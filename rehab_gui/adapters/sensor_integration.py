"""手机/手环/CSI 跌倒检测集成

将外部传感器（ESP32-S3 手环、手机 IMU、CSI WiFi）的跌倒检测
集成到 GUI 帧循环中。所有传感器独立工作，不需要人脸锁定。

用法:
    sensor_manager = init_external_sensors(
        wristband=True, wristband_port=8081,
        phone=True,
        csi=True, csi_host="192.168.2.10", csi_port=8000, csi_protocol="LLTF",
        fall_detector=fall_detector,  # FallDetector 实例
    )
    # 每帧检查传感器跌倒状态
    sensor_manager.apply_to_fall_detector(fall_detector)
"""
import os
import sys
import re
import time
import threading
import subprocess as sp

from rehab_monitor.logging_setup import get_logger

logger = get_logger("sensor_integration")


class SensorManager:
    """手机/手环/CSI 传感器管理器"""

    def __init__(self):
        self.wristband_proc = None
        self.phone_data_source = None
        self.phone_thread = None
        self.phone_stop = None
        self.csi_monitor = None
        self.csi_thread = None
        self.wristband_fall = {"active": False, "score": 0.0, "magnitude": 0.0}
        self.phone_fall = {"active": False, "score": 0.0}
        self.csi_fall = {"active": False, "score": 0.0}
        # 跌倒超时自动重置 — 防止传感器状态永不消退
        self.SENSOR_FALL_TIMEOUT = 5.0  # 秒
        self._wristband_fall_time = 0.0
        self._csi_fall_time = 0.0

    def get_status(self):
        """获取所有传感器状态

        Returns:
            dict: 传感器状态字典
        """
        return {
            "wristband": {
                "active": self.wristband_proc is not None,
                "running": self.wristband_proc.poll() is None if self.wristband_proc else False,
                "fall": self.wristband_fall["active"],
                "score": self.wristband_fall["score"],
            },
            "phone": {
                "active": self.phone_data_source is not None,
                "connected": self.phone_data_source.is_active() if self.phone_data_source else False,
                "fall": self.phone_fall["active"],
                "score": self.phone_fall["score"],
            },
            "csi": {
                "active": self.csi_monitor is not None,
                "connected": self.csi_thread is not None and self.csi_thread.is_alive() if self.csi_thread else False,
                "fall": self.csi_fall["active"],
                "score": self.csi_fall["score"],
            },
        }

    def apply_to_fall_detector(self, fall_detector):
        """将传感器跌倒状态注入到 FallDetector

        传感器跌倒优先于摄像头视觉跌倒检测。
        当传感器未检测到跌倒时，会重置 FallDetector 状态。
        跌倒状态超时（默认5秒）后自动重置，防止永不消退。

        Args:
            fall_detector: FallDetector 实例
        """
        now = time.time()

        # 手环跌倒（带超时自动重置）
        if self.wristband_fall["active"]:
            if now - self._wristband_fall_time > self.SENSOR_FALL_TIMEOUT:
                self.wristband_fall["active"] = False
                self.wristband_fall["score"] = 0.0
            else:
                fall_detector.status = "alert"
                fall_detector.score = self.wristband_fall["score"]
                return ("alert", fall_detector.score)
        else:
            self.wristband_fall["active"] = False
            self.wristband_fall["score"] = 0.0

        # 手机跌倒
        if self.phone_data_source and self.phone_data_source.is_active():
            phone_status = self.phone_data_source.get_fall_status()
            if phone_status in ("ALERTING", "FALLEN"):
                fall_detector.status = "alert"
                fall_detector.score = self.phone_data_source.get_fall_score()
                self.phone_fall["active"] = True
                self.phone_fall["score"] = fall_detector.score
                return ("alert", fall_detector.score)
            else:
                self.phone_fall["active"] = False
                self.phone_fall["score"] = 0.0

        # CSI 跌倒（带超时自动重置，并联检测，不取代摄像头）
        if self.csi_monitor and self.csi_monitor.is_fallen:
            csi_status = self.csi_monitor.status
            csi_score = csi_status.get("score", 0.0)
            if csi_score > fall_detector.score:
                fall_detector.status = "alert"
                fall_detector.score = csi_score
                self.csi_fall["active"] = True
                self.csi_fall["score"] = csi_score
                self._csi_fall_time = now
                return ("alert", fall_detector.score)
            else:
                if now - self._csi_fall_time > self.SENSOR_FALL_TIMEOUT:
                    self.csi_fall["active"] = False
                    self.csi_fall["score"] = 0.0
        else:
            if now - self._csi_fall_time > self.SENSOR_FALL_TIMEOUT:
                self.csi_fall["active"] = False
                self.csi_fall["score"] = 0.0

        # 所有传感器均未触发跌倒 — 返回安全状态（视觉检测由调用方按需启用）
        fall_detector.status = "safe"
        fall_detector.score = 0.0
        return ("safe", 0.0)

    def close(self):
        """关闭所有传感器连接"""
        if self.wristband_proc:
            try:
                if self.wristband_proc.poll() is None:
                    self.wristband_proc.terminate()
                    self.wristband_proc.wait(timeout=3)
            except Exception:
                try:
                    self.wristband_proc.kill()
                    self.wristband_proc.wait(timeout=2)
                except Exception:
                    pass
            self.wristband_proc = None
            logger.info("手环服务器已关闭")

        if self.phone_stop:
            try:
                if hasattr(self.phone_stop, "set"):
                    self.phone_stop.set()
                elif callable(self.phone_stop):
                    self.phone_stop()
            except Exception:
                pass
            if self.phone_thread and self.phone_thread.is_alive():
                self.phone_thread.join(timeout=2.0)
            self.phone_thread = None
            self.phone_stop = None
            logger.info("手机监听器已关闭")

        if self.csi_monitor:
            try:
                if hasattr(self.csi_monitor, "stop"):
                    self.csi_monitor.stop()
            except Exception:
                pass
            self.csi_monitor = None
            self.csi_thread = None
            logger.info("CSI 监视器已停止")


def _init_wristband(wristband_port=8081, wristband_http=8080, sensor_manager=None):
    """初始化手环服务器

    Args:
        wristband_port: TCP 端口
        wristband_http: HTTP 端口
        sensor_manager: SensorManager 实例
    """
    # 清理端口残留进程
    for port in (wristband_port, wristband_http):
        try:
            sp.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=3)
        except Exception:
            pass

    # 查找项目根目录（rehab_gui 的父目录）
    rehab_gui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(rehab_gui_dir)
    server_script = os.path.join(project_root, "ubuntu", "_wristband_server.py")

    if not os.path.exists(server_script):
        logger.warning(f"手环服务器脚本不存在: {server_script}")
        return None

    fall_pattern = re.compile(r"FALL #\d+ \| Peak: ([\d.]+) m/s2 \| Conf: (\d+)%")
    http_fall_pattern = re.compile(r"FALL \(HTTP\)! mag=([\d.]+)")
    important = re.compile(r'(connected|Connected|Disconnected|subscriber|FALL|Error|error|Waiting for|mDNS|TCP)')

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = sp.Popen(
        [sys.executable, server_script],
        stdout=sp.PIPE, stderr=sp.STDOUT,
        text=True, bufsize=1,
        env=env,
    )

    def monitor(proc):
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            m = fall_pattern.search(line)
            hm = http_fall_pattern.search(line)
            if m:
                peak = float(m.group(1))
                conf = int(m.group(2)) / 100.0
                if sensor_manager:
                    sensor_manager.wristband_fall["active"] = True
                    sensor_manager.wristband_fall["score"] = conf
                    sensor_manager.wristband_fall["magnitude"] = peak
                    sensor_manager._wristband_fall_time = time.time()
                logger.info(f"手环跌倒: peak={peak:.1f} m/s2, conf={conf:.0%}")
                print(f"[手环] {line}")
                # 广播跌倒告警 (WebSocket → 小程序 + GUI 弹窗)
                try:
                    from rehab_monitor.api_server import broadcast_fall_alert
                    broadcast_fall_alert(None, conf)
                except Exception:
                    pass
            elif hm:
                conf = min(float(hm.group(1)) / 10.0, 1.0)
                if sensor_manager:
                    sensor_manager.wristband_fall["active"] = True
                    sensor_manager.wristband_fall["score"] = conf
                    sensor_manager.wristband_fall["magnitude"] = float(hm.group(1))
                    sensor_manager._wristband_fall_time = time.time()
                logger.info(f"手环跌倒(HTTP): mag={hm.group(1)}, conf={conf:.0%}")
                print(f"[手环] {line}")
                try:
                    from rehab_monitor.api_server import broadcast_fall_alert
                    broadcast_fall_alert(None, conf)
                except Exception:
                    pass
            elif important.search(line):
                print(f"[手环] {line}")

    thread = threading.Thread(target=monitor, args=(proc,),
                              name="wristband-monitor", daemon=True)
    thread.start()

    logger.info(f"✓ 手环服务器已启动 (TCP:{wristband_port} HTTP:{wristband_http})")
    return proc


def _init_phone(sensor_manager=None):
    """初始化手机 IMU 数据接入

    Args:
        sensor_manager: SensorManager 实例
    """
    try:
        from ubuntu.phone_data import phone_data_source, start_phone_listener

        phone_listener, phone_stop = start_phone_listener()
        sensor_manager.phone_data_source = phone_data_source
        sensor_manager.phone_thread = phone_listener
        sensor_manager.phone_stop = phone_stop

        logger.info("✓ 手机 IMU 数据接入已启动 (UDP:5002)")
        return True
    except ImportError:
        logger.warning("手机 IMU 模块不可用 (ubuntu/phone_data.py)")
        return False
    except Exception as e:
        logger.warning(f"手机 IMU 初始化失败: {e}")
        return False


def _init_csi(csi_host, csi_port, csi_protocol, show_diag=False):
    """初始化 CSI WiFi 跌倒检测

    Args:
        csi_host: ESP32 IP
        csi_port: TCP 端口
        csi_protocol: WiFi 协议
        show_diag: 是否显示诊断窗口

    Returns:
        CSIMonitor instance or None
    """
    try:
        from csi_fall.monitor import CSIMonitor

        monitor = CSIMonitor(
            host=csi_host, port=csi_port,
            wifi_protocol=csi_protocol,
            show_diag=show_diag,
        )
        thread = monitor.start()
        if thread:
            logger.info(f"✓ CSI 跌倒检测已启动 ({csi_host}:{csi_port}, {csi_protocol})")
            return monitor, thread
        else:
            logger.warning("CSI 跌倒检测启动失败 (ESP32 可能不在线)")
            return None, None
    except ImportError:
        logger.warning("CSI 模块不可用 (csi_fall/)")
        return None, None
    except Exception as e:
        logger.warning(f"CSI 初始化失败: {e}")
        return None, None


def init_external_sensors(config, fall_detector=None):
    """根据配置初始化所有外部传感器

    Args:
        config: 配置字典 (含 csi_wristband/wristband/csi, csi_host, csi_port, csi_protocol)
        fall_detector: FallDetector 实例（可选，用于绑定）

    Returns:
        SensorManager instance
    """
    manager = SensorManager()

    # CSI（手环）— 合并选项，兼容旧配置
    csi_enabled = (config.get("csi_wristband")
                   or config.get("wristband")
                   or config.get("csi"))
    if csi_enabled:
        wristband_port = config.get("wristband_port", 8081)
        wristband_http = config.get("wristband_http", 8080)
        manager.wristband_proc = _init_wristband(
            wristband_port, wristband_http, sensor_manager=manager)

        csi_host = config.get("csi_host", "192.168.2.10")
        csi_port = int(config.get("csi_port", 8000))
        csi_protocol = config.get("csi_protocol", "LLTF")
        csi_show_diag = config.get("csi_show_diag", False)
        monitor, thread = _init_csi(csi_host, csi_port, csi_protocol, csi_show_diag)
        manager.csi_monitor = monitor
        manager.csi_thread = thread
    else:
        # 即使未启用，也尝试启动手机 IMU（自动发现，无需手动启用）
        pass

    # 手机 IMU 始终尝试启动（UDP 自动发现，不阻塞）
    _init_phone(sensor_manager=manager)

    # 启动局域网发现服务（供小程序自动发现服务器）
    try:
        from rehab_monitor.discovery_service import start_discovery
        start_discovery()
    except Exception as e:
        logger.warning(f"UDP 发现服务启动失败（不影响核心功能）: {e}")

    return manager
