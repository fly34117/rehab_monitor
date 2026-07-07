"""线程桥接 — OpenCV→PyQt 数据通道"""
import threading
from queue import Queue, Full

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from rehab_monitor.logging_setup import get_logger

logger = get_logger("thread_bridge")


class ThreadBridge(QObject):
    """桥接 rehab_monitor 后端和 PyQt 前端

    通过信号槽实现线程安全的数据传输
    """

    # 画面帧信号
    frame_signal = pyqtSignal(np.ndarray)
    # 帧计数信号
    frame_count_signal = pyqtSignal(int)

    # 跌倒状态信号
    fall_signal = pyqtSignal(dict)
    # 情绪数据信号
    emotion_signal = pyqtSignal(dict)
    # 步态指标信号
    metrics_signal = pyqtSignal(dict)
    # 呼吸数据信号
    breathing_signal = pyqtSignal(dict)
    # 人脸缩略图信号 (object 类型以支持 None 清除)
    face_thumbnail_signal = pyqtSignal(object)

    # LLM 分析响应信号 (response_text: str, status: str)
    llm_response_signal = pyqtSignal(str, str)
    # LLM 状态更新信号 (status: str, message: str)
    llm_status_signal = pyqtSignal(str, str)

    # 诊断画面信号
    diag_angle_signal = pyqtSignal(np.ndarray)
    diag_trajectory_signal = pyqtSignal(np.ndarray)
    diag_skeleton_signal = pyqtSignal(np.ndarray)
    diag_gait_signal = pyqtSignal(np.ndarray)
    diag_breath_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        """初始化线程桥接器"""
        super().__init__()
        # 队列用于异步数据传输
        self._frame_queue = Queue(maxsize=2)
        self._diag_angle_queue = Queue(maxsize=2)
        self._diag_trajectory_queue = Queue(maxsize=2)
        self._diag_skeleton_queue = Queue(maxsize=2)
        self._diag_gait_queue = Queue(maxsize=2)
        self._latest_frame = None
        self._latest_frame_num = 0
        self._latest_frame_seq = 0
        self._delivered_frame_seq = 0
        self._frame_lock = threading.Lock()

        self._running = False

        logger.info("线程桥接器已初始化")

    def start(self):
        """启动桥接器"""
        self._running = True
        logger.info("线程桥接器已启动")

    def stop(self):
        """停止桥接器"""
        self._running = False
        logger.info("线程桥接器已停止")

    def is_running(self):
        """检查是否运行中

        Returns:
            bool: 是否运行中
        """
        return self._running

    # ===== 主画面 =====

    def push_frame(self, frame):
        """推送主画面帧

        Args:
            frame: np.ndarray OpenCV BGR 格式帧
        """
        if not self._running:
            return
        # GUI 端按固定频率拉取最新帧，避免每帧 emit 造成 Qt 事件队列积压。
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_seq += 1

    def take_latest_frame(self):
        """取出尚未显示的最新帧。

        Returns:
            tuple: (frame_or_none, frame_num)
        """
        with self._frame_lock:
            if self._latest_frame_seq == self._delivered_frame_seq:
                return None, self._latest_frame_num
            self._delivered_frame_seq = self._latest_frame_seq
            return self._latest_frame, self._latest_frame_num

    # ===== 诊断画面 =====

    def push_diag_angle(self, frame):
        """推送关节角度画面

        Args:
            frame: np.ndarray OpenCV BGR 格式帧
        """
        if not self._running:
            return
        try:
            self._push_if_not_full(self._diag_angle_queue, frame)
            self.diag_angle_signal.emit(frame)
        except Full:
            pass

    def push_diag_trajectory(self, frame):
        """推送俯视轨迹画面

        Args:
            frame: np.ndarray OpenCV BGR 格式帧
        """
        if not self._running:
            return
        try:
            self._push_if_not_full(self._diag_trajectory_queue, frame)
            self.diag_trajectory_signal.emit(frame)
        except Full:
            pass

    def push_diag_skeleton(self, frame):
        """推送骨骼预览画面

        Args:
            frame: np.ndarray OpenCV BGR 格式帧
        """
        if not self._running:
            return
        try:
            self._push_if_not_full(self._diag_skeleton_queue, frame)
            self.diag_skeleton_signal.emit(frame)
        except Full:
            pass

    def push_diag_gait(self, frame):
        """推送步态指标画面"""
        if not self._running:
            return
        try:
            self._push_if_not_full(self._diag_gait_queue, frame)
            self.diag_gait_signal.emit(frame)
        except Full:
            pass

    def push_diag_breath(self, frame):
        """推送呼吸波形画面"""
        if not self._running:
            return
        if frame is not None and frame.size > 0:
            self.diag_breath_signal.emit(frame)

    def _push_if_not_full(self, queue, item):
        """推入队列, 满时丢弃旧项

        Args:
            queue: Queue 对象
            item: 要推入的项
        """
        if queue.full():
            try:
                queue.get_nowait()
            except Exception:
                pass
        queue.put_nowait(item)

    # ===== 状态数据 =====

    def push_fall(self, status, score=None, source="摄像头"):
        """推送跌倒状态

        Args:
            status: "safe" / "alert" / "lying"
            score: 跌倒分数
            source: 检测来源
        """
        if not self._running:
            return
        self.fall_signal.emit({
            "status": status,
            "score": score,
            "source": source,
        })

    def push_emotion(self, label, scores=None):
        """推送情绪数据

        Args:
            label: 情绪标签
            scores: 各情绪分数 dict
        """
        if not self._running:
            return
        self.emotion_signal.emit({
            "label": label,
            "scores": scores,
        })

    def push_metrics(self, data):
        """推送步态指标

        Args:
            data: dict 步态指标数据
        """
        if not self._running:
            return
        self.metrics_signal.emit(data)

    def push_breathing(self, status, bpm=None, remaining=0, total=30, has_person=True):
        """推送呼吸数据

        Args:
            status: 状态文本 ("空闲" / "检测中" / "完成")
            bpm: BPM 值 (None=尚未计算出)
            remaining: 剩余秒数
            total: 总秒数
            has_person: 是否检测到人体
        """
        if not self._running:
            return
        self.breathing_signal.emit({
            "status": status,
            "bpm": bpm,
            "remaining": remaining,
            "total": total,
            "has_person": has_person,
        })

    def push_frame_count(self, frame_num):
        """推送帧计数

        Args:
            frame_num: 当前帧号
        """
        if not self._running:
            return
        with self._frame_lock:
            self._latest_frame_num = frame_num

    def push_face_thumbnail(self, face_frame):
        """推送人脸缩略图（干净，无骨架叠加）

        Args:
            face_frame: np.ndarray 人脸区域 BGR 图像，None 表示清除缩略图
        """
        if not self._running:
            return
        self.face_thumbnail_signal.emit(face_frame)

    def push_llm_response(self, text, status="done"):
        """推送 LLM 回答

        Args:
            text: LLM 生成的报告文本
            status: "done" / "error"
        """
        if not self._running:
            return
        self.llm_response_signal.emit(text, status)

    def push_llm_status(self, status, message=""):
        """推送 LLM 状态更新

        Args:
            status: "idle" / "running" / "done" / "error"
            message: 状态消息
        """
        if not self._running:
            return
        self.llm_status_signal.emit(status, message)
