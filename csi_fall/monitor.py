"""CSI 跌倒检测监视器 — 后台线程 + 告警桥接
在独立 daemon 线程中运行完整的 CSI 数据接收→处理→算法链,
检测到跌倒时直接调用 yolov26 已有告警基础设施。
"""
from __future__ import annotations
import threading
import time
import queue

from rehab_monitor.logging_setup import get_logger

from .config import (
    CSI_HOST, CSI_PORT, CSI_WIFI_PROTOCOL,
    FALL_SHORT_WINDOW, FALL_LONG_WINDOW, FALL_THRESHOLD,
    FALL_ALERT_FRAMES, FALL_COOLDOWN_FRAMES,
    EIGENVEC_WINDOW_SIZE,
    CSI_FALL_PUSH_COOLDOWN, CSI_FALL_LOCATION,
    CSI_SHOW_DIAG,
)
from .receiver import CSIReceiver
from .processor import CSIProcessor
from .algorithms.rssi_weighting import RSSIWeighting
from .algorithms.eigenvec_filter import EigenvecPerSubcarrierFilter
from .algorithms.fall_detection import FallDetection

logger = get_logger("csi.monitor")


class CSIMonitor:
    """CSI 跌倒检测监视器

    原理:
      - 在后台 daemon 线程中运行 TCP 接收 + CSI 处理 + 算法链
      - 通过算法链处理每帧 CSI 数据: RSSI加权 → 特征向量去噪 → 方差比跌倒检测
      - 识别到跌倒时, 直接调用 yolov26 告警基础设施:
        1. broadcast_fall_alert() → WebSocket → 微信小程序
        2. enqueue_fall() → OpenCV 桌面弹窗
        3. write_fall_alert() → SQLite 数据库
      - 支持可选 OpenCV 诊断窗口 (非 tkinter/matplotlib)

    Args:
        host: ESP32 IP 地址
        port: TCP 端口
        wifi_protocol: WiFi 协议 ("LLTF" | "HT40")
        show_diag: 是否显示 OpenCV 诊断窗口
    """

    def __init__(
        self,
        host: str = "",
        port: int = 0,
        wifi_protocol: str = "",
        show_diag: bool = False,
    ):
        self.host = host or CSI_HOST
        self.port = port or CSI_PORT
        self.wifi_protocol = wifi_protocol or CSI_WIFI_PROTOCOL
        self.show_diag = show_diag or CSI_SHOW_DIAG

        # 核心组件
        self.receiver = CSIReceiver(self.host, self.port)
        self.processor = CSIProcessor(self.wifi_protocol)
        self.fall_detector = FallDetection(
            short_window=FALL_SHORT_WINDOW,
            long_window=FALL_LONG_WINDOW,
            threshold=FALL_THRESHOLD,
            alert_frames=FALL_ALERT_FRAMES,
            cooldown_frames=FALL_COOLDOWN_FRAMES,
        )

        # 线程控制
        self._receive_thread: threading.Thread | None = None
        self._running: bool = False
        self._stop_event = threading.Event()

        # 最新状态 (线程安全, 供外部查询)
        self._state_lock = threading.Lock()
        self._latest_score: float = 0.0
        self._latest_alert: bool = False
        self._frame_count: int = 0
        self._fall_count: int = 0

        # 告警推送冷却 (时间维度, 秒)
        self._last_push_time: float = 0.0

        # 诊断显示队列
        self._diag_queue: queue.Queue | None = None

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    def start(self) -> threading.Thread | None:
        """启动 CSI 监视器 (非阻塞, 在后台 daemon 线程中运行)

        Returns:
            threading.Thread — 接收线程, 或 None (启动失败)
        """
        if self._running:
            logger.warning("CSI 监视器已在运行中")
            return self._receive_thread

        # 注册算法链 (顺序很重要: RSSI → 去噪 → 跌倒检测)
        self.processor.register_algorithm('rssi_weighting', RSSIWeighting())
        self.processor.register_algorithm(
            'eigenvec_filter',
            EigenvecPerSubcarrierFilter(sliding_window_size=EIGENVEC_WINDOW_SIZE),
        )
        self.processor.register_algorithm('fall_detection', self.fall_detector)

        # 建立 TCP 连接
        if not self.receiver.connect():
            logger.error("CSI 监视器启动失败: TCP 连接失败")
            return None

        self._running = True
        self._stop_event.clear()

        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            name="csi-receiver",
            daemon=True,
        )
        self._receive_thread.start()

        # 可选: 诊断显示
        if self.show_diag:
            self._start_diag()

        logger.info(
            "CSI 监视器已启动 %s:%d (协议=%s, 诊断=%s)",
            self.host, self.port, self.wifi_protocol, self.show_diag,
        )
        return self._receive_thread

    def stop(self):
        """停止 CSI 监视器"""
        self._running = False
        self._stop_event.set()
        self.receiver.disconnect()
        self.processor.clear()

        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)

        logger.info("CSI 监视器已停止")

    # ------------------------------------------------------------------
    # 接收循环 (后台线程)
    # ------------------------------------------------------------------

    def _receive_loop(self):
        """后台接收循环"""

        def on_packet(raw_data: dict):
            full_data = self.processor.process(raw_data)
            if full_data is not None:
                self._on_frame(full_data)

        try:
            self.receiver.receive_loop(on_packet)
        except Exception as e:
            logger.error("CSI 接收线程异常: %s", e, exc_info=True)
        finally:
            self._running = False

    # ------------------------------------------------------------------
    # 帧处理 + 告警桥接
    # ------------------------------------------------------------------

    def _on_frame(self, full_data: dict):
        """帧处理回调 — 检查跌倒告警并触发"""
        fall_alert = full_data.get('fall_alert', False)
        fall_score = full_data.get('fall_score', 0.0)

        with self._state_lock:
            self._frame_count += 1
            self._latest_score = fall_score
            self._latest_alert = fall_alert
            self._fall_count = full_data.get('fall_count', 0)
            fc = self._frame_count

        # 周期日志 (每 30 帧输出一次)
        if fc % 30 == 0:
            logger.debug(
                "CSI 帧 #%d, score=%.3f, alert=%s, falls=%d",
                fc, fall_score, fall_alert, self._fall_count,
            )

        # 跌倒告警触发
        if fall_alert:
            self._trigger_alert(fall_score)

        # 诊断显示更新
        if self.show_diag and self._diag_queue is not None:
            try:
                score_history = full_data.get('fall_history', [])
                if score_history:
                    self._diag_queue.put_nowait({
                        'score': fall_score,
                        'history': list(score_history),
                        'alert': fall_alert,
                        'frame': fc,
                        'fall_count': self._fall_count,
                    })
            except queue.Full:
                pass

    def _trigger_alert(self, score: float):
        """触发告警 — 桥接到 yolov26 已有告警基础设施

        三层告警:
          1. WebSocket → 微信小程序 (broadcast_fall_alert)
          2. OpenCV 桌面弹窗 (enqueue_fall)
          3. SQLite 数据库写入 (write_fall_alert)

        Args:
            score: 当前跌倒分数
        """
        now = time.time()

        # 时间冷却去重 (CSI 算法内部已有 50 帧冷却, 再加一道时间保护)
        if now - self._last_push_time < CSI_FALL_PUSH_COOLDOWN:
            return
        self._last_push_time = now

        loc = CSI_FALL_LOCATION

        # 1. WebSocket → 微信小程序
        try:
            from rehab_monitor.api_server import broadcast_fall_alert
            broadcast_fall_alert(loc, score)
            logger.info("CSI WebSocket 告警已推送 score=%.2f", score)
        except Exception as e:
            logger.error("CSI WebSocket 推送失败: %s", e)

        # 2. OpenCV 桌面弹窗
        try:
            from ubuntu.fall_popup import enqueue_fall
            enqueue_fall("csi", score, {
                "csi_score": round(score, 3),
                "location": list(loc),
            })
            logger.info("CSI 桌面弹窗已触发 score=%.2f", score)
        except Exception as e:
            logger.error("CSI 弹窗触发失败: %s", e)

        # 3. SQLite 数据库写入
        try:
            import rehab_monitor.api_server as _api
            db = getattr(_api, 'database', None)
            if db is not None and hasattr(db, 'write_fall_alert'):
                db.write_fall_alert(self._frame_count)
                logger.info("CSI 跌倒已写入数据库 frame=%d", self._frame_count)
        except Exception as e:
            logger.error("CSI 数据库写入失败: %s", e)

    # ------------------------------------------------------------------
    # 诊断显示
    # ------------------------------------------------------------------

    def _start_diag(self):
        """启动可选的 OpenCV 诊断窗口"""
        try:
            import cv2
            from .display import CSIDisplay

            self._diag_queue = queue.Queue(maxsize=10)
            self._display = CSIDisplay()
            self._diag_thread = threading.Thread(
                target=self._display.run,
                args=(self._diag_queue,),
                name="csi-diag",
                daemon=True,
            )
            self._diag_thread.start()
            logger.info("CSI 诊断窗口已启动")
        except ImportError:
            logger.warning("CSI 诊断窗口不可用 (缺少 cv2)")
            self.show_diag = False

    # ------------------------------------------------------------------
    # 外部查询接口
    # ------------------------------------------------------------------

    @property
    def is_fallen(self) -> bool:
        """是否处于跌倒告警状态 (供外部查询)"""
        with self._state_lock:
            return self._latest_alert

    @property
    def status(self) -> dict:
        """返回当前状态快照 (供 API 或调试)"""
        with self._state_lock:
            return {
                "running": self._running,
                "connected": self.receiver.connected,
                "frames": self._frame_count,
                "score": round(self._latest_score, 3),
                "fall_count": self._fall_count,
                "host": self.host,
                "port": self.port,
                "protocol": self.wifi_protocol,
            }
