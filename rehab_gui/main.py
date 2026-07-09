"""GUI 入口点 — 简化集成模式（修复阻塞问题）"""
import sys
import os
import signal
import time
import threading
import subprocess
import urllib.request
import numpy as np

from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from rehab_gui.config import (
    CONFIG_DIR,
    LD_LIBRARY_PATH,
    ENV_VARS,
    LLM_MODEL_PATHS,
    LLM_PORT,
    LLAMA_SERVER,
    LLAMA_LIB_DIR,
    LLM_CPUSET,
    LLM_THREADS,
    LLM_NICE,
)
from rehab_gui.core.config_manager import ConfigManager
from rehab_gui.core.device_detector import DeviceDetector
from rehab_gui.core.thread_bridge import ThreadBridge
from rehab_gui.gui.main_window import MainWindow
from rehab_gui.gui.config_popup import ConfigPopup
from rehab_monitor.logging_setup import get_logger

logger = get_logger("gui_main")


def _setup_environment():
    """设置环境变量"""
    os.environ["LD_LIBRARY_PATH"] = f"{LD_LIBRARY_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    for key, value in ENV_VARS.items():
        os.environ[key] = value


def _load_theme(app):
    """加载深色主题"""
    theme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gui",
        "dark_theme.qss",
    )
    try:
        with open(theme_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass


class BackendInitThread(QThread):
    """后端初始化线程（简化版）"""

    backend_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config, bridge):
        super().__init__()
        self._config = config
        self._bridge = bridge
        self._backend = None

    def run(self):
        """在后台线程初始化后端（只初始化核心组件）"""
        try:
            logger.info("后台初始化后端组件（简化版）...")

            import cv2

            from rehab_monitor.config import POSE_MODEL_PATH, POSE_DEVICE, CAMERA_ID, USE_KALMAN
            from rehab_monitor.pose_detector import PoseDetector
            from rehab_monitor.gait_analyzer import GaitAnalyzer
            from rehab_monitor.fall_detector import FallDetector
            from rehab_monitor.display_overlay import DisplayOverlay
            from rehab_monitor.angle_monitor import AngleMonitor
            from rehab_monitor.trajectory_monitor import TrajectoryMonitor
            from rehab_monitor.skeleton_viewer import SkeletonViewer
            from rehab_monitor.gait_metrics_viewer import GaitMetricsViewer

            # 根据配置设置模型路径
            model_name = self._config.get("model", "yolo26n")
            device = self._config.get("device", "cpu")
            from rehab_gui.config import resolve_pose_model_path
            model_path, precision = resolve_pose_model_path(
                model_name, device, os.environ.get("REHAB_MODEL_PRECISION")
            )

            logger.info(f"加载模型: {model_path} (设备={device}, 精度={precision})")

            # 打开摄像头
            camera = cv2.VideoCapture(CAMERA_ID)
            if not camera.isOpened():
                raise RuntimeError("无法打开摄像头")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info("✓ 摄像头已打开")

            # 初始化姿态检测
            pose_detector = PoseDetector(model_path, device, USE_KALMAN)
            logger.info("✓ 姿态检测器已初始化")

            # 初始化核心分析器
            gait_analyzer = GaitAnalyzer()
            fall_detector = FallDetector()
            display_overlay = DisplayOverlay()

            # 初始化诊断窗口
            angle_monitor = AngleMonitor()
            trajectory_monitor = TrajectoryMonitor()
            skeleton_viewer = SkeletonViewer()
            gait_metrics_viewer = GaitMetricsViewer()

            logger.info("✓ 核心组件初始化完成")

            # 跳过非必需组件（避免阻塞）
            logger.info("跳过非必需组件（情绪识别、API 服务器、数据库）")

            # 创建后端集成对象
            self._backend = BackendIntegrator(
                self._bridge,
                camera,
                pose_detector,
                gait_analyzer,
                fall_detector,
                display_overlay,
                angle_monitor,
                trajectory_monitor,
                skeleton_viewer,
                gait_metrics_viewer,
            )

            logger.info("✓ 后端初始化完成")
            self.backend_ready.emit(self._backend)

        except Exception as e:
            logger.error(f"后端初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class BackendIntegrator:
    """后端集成器（简化版）"""

    def __init__(
        self,
        bridge,
        camera,
        pose_detector,
        gait_analyzer,
        fall_detector,
        display_overlay,
        angle_monitor,
        trajectory_monitor,
        skeleton_viewer,
        gait_metrics_viewer,
    ):
        self._bridge = bridge
        self._camera = camera
        self._pose_detector = pose_detector
        self._gait_analyzer = gait_analyzer
        self._fall_detector = fall_detector
        self._display_overlay = display_overlay
        self._angle_monitor = angle_monitor
        self._trajectory_monitor = trajectory_monitor
        self._skeleton_viewer = skeleton_viewer
        self._gait_metrics_viewer = gait_metrics_viewer

        self._frame_count = 0
        self._last_diag_frame = 0
        self._running = False

    def start(self):
        """启动处理"""
        self._running = True

    def process_frame(self):
        """处理单帧（非阻塞）"""
        if not self._running:
            return False

        try:
            import cv2

            ret, frame = self._camera.read()
            if not ret:
                return False

            # 姿态检测
            annotated_frame, keypoints = self._pose_detector.process_frame(frame)

            # 步态分析 (GaitAnalyzer 只支持单人分析，取第一个人的关键点)
            gait_keypoints = None
            if keypoints is not None and len(keypoints.shape) == 3 and keypoints.shape[0] > 0:
                # keypoints shape: (num_persons, 17, 3) 
                # GaitAnalyzer expects: (17, 2) or (17, 3)
                gait_keypoints = keypoints[0]  # 取第一个人的关键点
                self._gait_analyzer.update(gait_keypoints, time.time())

            # 跌倒检测 (FallDetector 支持多人，传完整关键点数组)
            # 计算第一个人的边界框
            bbox = None
            if keypoints is not None and len(keypoints.shape) == 3 and keypoints.shape[0] > 0:
                first_person = keypoints[0]  # (17, 3)
                # 筛选置信度 > 0.3 的点
                high_conf_mask = first_person[:, 2] > 0.3  # (17,)
                if len(high_conf_mask.shape) == 1 and np.any(high_conf_mask):
                    valid_points = first_person[high_conf_mask, :2]  # (N, 2)
                    if len(valid_points) > 0:
                        bbox = [float(valid_points[:, 0].min()), float(valid_points[:, 1].min()),
                               float(valid_points[:, 0].max()), float(valid_points[:, 1].max())]

            fall_status, fall_score = self._fall_detector.update(
                keypoints, bbox, 480  # frame_height
            )

            # 构建显示状态
            state = {
                "fps": 0,
                "person_count": keypoints.shape[0] if keypoints is not None else 0,
                "angles": self._gait_analyzer.compute_joint_angles(gait_keypoints) if gait_keypoints is not None else {},
                "fall_status": fall_status,
                "fall_score": fall_score,
                "frame_num": self._frame_count,
            }

            # 渲染叠加层
            display_frame = self._display_overlay.render(annotated_frame, state)

            # 推送到 GUI
            self._bridge.push_frame(display_frame)
            self._bridge.push_frame_count(self._frame_count)

            if fall_status != "safe":
                self._bridge.push_fall(fall_status, fall_score, "摄像头")

            self._bridge.push_metrics({"person_count": state["person_count"]})

            # 诊断窗口（每3帧渲染一次）
            if self._frame_count - self._last_diag_frame >= 3:
                self._last_diag_frame = self._frame_count

                angle_frame = self._angle_monitor.render()
                if angle_frame is not None:
                    self._bridge.push_diag_angle(angle_frame)

                traj_frame = self._trajectory_monitor.render()
                if traj_frame is not None:
                    self._bridge.push_diag_trajectory(traj_frame)

                skel_frame = self._skeleton_viewer.render()
                if skel_frame is not None:
                    self._bridge.push_diag_skeleton(skel_frame)

                gait_frame = self._gait_metrics_viewer.render()
                if gait_frame is not None:
                    self._bridge.push_diag_gait(gait_frame)

            self._frame_count += 1
            return True

        except Exception as e:
            logger.error(f"处理帧失败: {e}")
            return False

    def shutdown(self):
        """关闭所有组件"""
        logger.info("正在关闭后端组件...")
        self._running = False
        if self._camera:
            self._camera.release()
        logger.info("后端组件已关闭")


class InitDialog(QDialog):
    """启动画面 — 显示初始化进度"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rehab Monitor — 正在启动")
        self.setFixedSize(420, 180)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.FramelessWindowHint
        )
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
            }
            QLabel { color: #e0e0e0; }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                background: #2d2d2d;
                height: 18px;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0e639c, stop:1 #4fc3f7);
                border-radius: 3px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # 标题
        title = QLabel("🏥 Rehab Monitor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4fc3f7;")
        layout.addWidget(title)

        # 步骤文字
        self._step_label = QLabel("正在启动...")
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_label.setStyleSheet("font-size: 13px; color: #a0a0a0;")
        layout.addWidget(self._step_label)

        # 进度条
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # 版本信息
        ver = QLabel("YOLO + FaceNet + LLM 多模态康复监控")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(ver)

    def set_step(self, text, pct):
        """更新当前步骤和进度"""
        self._step_label.setText(text)
        self._progress.setValue(pct)
        QApplication.processEvents()


def main():
    """主入口函数"""
    _setup_environment()
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # 创建 QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Rehab Monitor")
    app.setFont(QFont("WenQuanYi Micro Hei", 13))

    # 加载主题
    _load_theme(app)

    # 配置管理
    config_manager = ConfigManager()

    # 每次启动显示配置弹窗（用户可选择模型/设备）
    # 预加载已有配置，弹窗中自动填充
    if config_manager.has_config():
        config_manager.load()  # 加载到内存，弹窗 _load_saved_config 会读取

    popup = ConfigPopup(config_manager)
    if popup.exec() != popup.DialogCode.Accepted:
        sys.exit(0)
    config = popup.get_result()

    # 自动检测设备
    if config.get("device") == "auto":
        detector = DeviceDetector()
        detected = detector.detect()
        config["device"] = detected

    logger.info(f"使用配置: {config}")

    # ── 第一步：创建主窗口（先不显示）──
    window = MainWindow(config=config)
    # 注意：window.show() 推迟到全部初始化完成后

    # 创建线程桥接器
    bridge = ThreadBridge()
    bridge.start()
    window.set_bridge(bridge)

    # LLM 状态信号（分析/对话共用）
    bridge.llm_status_signal.connect(window.update_llm_status)

    # ── llama-server 管理（Python 控制启停）──
    _llm_server_proc = [None]  # list 以便闭包修改
    llm_ready = [False]         # LLM 服务器是否就绪

    def _detect_gpu_layers(model_path):
        """检测可用 GPU 并返回推荐的 -ngl 值。

        策略:
        - NVIDIA/CUDA → 99 (独显有专用 VRAM，全部 offload)
        - Intel/AMD Vulkan: 检查模型是否超出 GPU 可用内存
          推理实测: Vulkan 5.07 t/s vs CPU 3.07 t/s（快 65%）
          但 7B 模型 (3.4GB) 超出 Intel iGPU 3.4GB 可用 → 回退 CPU
        - 无 GPU → 0 (纯 CPU)
        """
        import re as _re
        # 检查 Vulkan 设备列表
        try:
            result = subprocess.run(
                [LLAMA_SERVER, "--list-devices"],
                capture_output=True, text=True, timeout=5,
                env={**os.environ, "LD_LIBRARY_PATH": LLAMA_LIB_DIR}
            )
            output = result.stdout + result.stderr
            has_nvidia = "NVIDIA" in output.upper()
            has_cuda = "CUDA" in output.upper()
            has_vulkan = "Vulkan" in output

            if has_nvidia or has_cuda:
                logger.info("检测到 NVIDIA/CUDA GPU，使用 GPU offload (-ngl 99)")
                return 99

            if has_vulkan:
                # Intel/AMD 核显 — 检查模型大小 vs GPU 可用内存
                model_size_gb = os.path.getsize(model_path) / (1024**3) if os.path.exists(model_path) else 99
                # 解析 GPU 可用内存 (MiB)，如 "3748 MiB, 3373 MiB free"
                mem_match = _re.search(r'(\d+)\s*MiB[^,]*free', output)
                gpu_free_gb = int(mem_match.group(1)) / 1024.0 if mem_match else 0
                # 留 512MB margin 给 KV cache
                if model_size_gb + 0.5 < gpu_free_gb:
                    logger.info(
                        f"Vulkan GPU 可用 {gpu_free_gb:.1f}GB, 模型 {model_size_gb:.1f}GB → "
                        f"GPU offload (-ngl 99)"
                    )
                    return 99
                else:
                    logger.info(
                        f"Vulkan GPU 可用 {gpu_free_gb:.1f}GB < 模型 {model_size_gb:.1f}GB + 0.5GB margin → "
                        f"CPU 推理 (-ngl 0)"
                    )
                    return 0
        except Exception:
            pass
        # 回退：nvidia-smi
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            logger.info("检测到 NVIDIA GPU (nvidia-smi)，使用 GPU offload (-ngl 99)")
            return 99
        except Exception:
            pass
        logger.info("未检测到 GPU，使用纯 CPU 推理 (-ngl 0)")
        return 0

    def _start_llm_server(model_key):
        """启动 llama-server（自动检测 GPU/CPU 模式）"""
        model_path = LLM_MODEL_PATHS.get(model_key)
        if not model_path or not os.path.exists(model_path):
            logger.error(f"LLM 模型不存在: {model_path}")
            return False

        # 确保 llama-server 有执行权限（git 可能丢失 +x）
        if not os.access(LLAMA_SERVER, os.X_OK):
            try:
                os.chmod(LLAMA_SERVER, 0o755)
                logger.info("已修复 llama-server 执行权限")
            except Exception as e:
                logger.error(f"无法设置 llama-server 执行权限: {e}")
                return False

        _stop_llm_server()
        # 杀掉所有占用 LLM 端口的旧进程（防止残留）
        try:
            subprocess.run(["fuser", "-k", f"{LLM_PORT}/tcp"], stderr=subprocess.DEVNULL, timeout=5)
        except Exception:
            pass
        time.sleep(1)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{LLAMA_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
        os.environ["LLM_MODEL_ALIAS"] = model_key  # 同步给 llm_client.py
        os.environ["REHAB_LLM_PORT"] = str(LLM_PORT)

        ngl = _detect_gpu_layers(model_path)
        gpu_label = "GPU" if ngl > 0 else "CPU"

        # 通用启动参数
        base_cmd = [LLAMA_SERVER, "-m", model_path,
                    "--host", "127.0.0.1", "--port", str(LLM_PORT),
                    "-t", str(LLM_THREADS), "-c", "4096",
                    "-ngl", str(ngl), "--alias", model_key,
                    "--mlock"]  # 锁定内存防止 swap

        if "qwen3" in model_key.lower():
            cmd = base_cmd + ["--reasoning", "off"]
        else:
            cmd = base_cmd + ["--reasoning", "on", "--reasoning-budget", "256"]

        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            llama_log_path = os.path.join(log_dir, f"llama_server_{time.strftime('%Y%m%d_%H%M%S')}.log")
            llama_log = open(llama_log_path, "a", buffering=1, encoding="utf-8")

            launch_cmd = ["taskset", "-c", LLM_CPUSET, *cmd]
            if LLM_NICE:
                launch_cmd = ["nice", "-n", str(LLM_NICE), *launch_cmd]
            _llm_server_proc[0] = subprocess.Popen(launch_cmd, stdout=llama_log,
                                                    stderr=subprocess.STDOUT, env=env)
            llama_log.close()
            logger.info(
                f"llama-server 启动中: {model_key} (PID={_llm_server_proc[0].pid}, "
                f"{gpu_label} mode, ngl={ngl}, threads={LLM_THREADS}, cpuset={LLM_CPUSET}, nice={LLM_NICE}, "
                f"log={llama_log_path})"
            )
            return True
        except Exception as e:
            logger.error(f"llama-server 启动失败: {e}")
            return False

    def _stop_llm_server():
        """停止 llama-server"""
        proc = _llm_server_proc[0]
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            _llm_server_proc[0] = None
            logger.info("llama-server 已停止")

    # ── 显示启动画面 ──
    init_dialog = InitDialog()
    init_dialog.show()
    app.processEvents()

    # ── 启动 LLM 服务器（后台线程，与 YOLO 并行加载）──
    llm_model = config.get("llm_model", "qwen3-4b")
    threading.Thread(
        target=lambda: _start_llm_server(llm_model),
        name="LlmServerStarter",
        daemon=True,
    ).start()
    init_dialog.set_step("启动 LLM 服务器...", 5)
    app.processEvents()

    # ── 初始化后端组件 ──
    logger.info("后台初始化后端组件（简化版）...")

    try:
        import cv2
        from rehab_monitor.config import CAMERA_ID, USE_KALMAN
        from rehab_monitor.pose_detector import PoseDetector
        from rehab_monitor.gait_analyzer import GaitAnalyzer
        from rehab_monitor.fall_detector import FallDetector
        from rehab_monitor.display_overlay import DisplayOverlay
        from rehab_monitor.angle_monitor import AngleMonitor
        from rehab_monitor.trajectory_monitor import TrajectoryMonitor
        from rehab_monitor.skeleton_viewer import SkeletonViewer
        from rehab_monitor.gait_metrics_viewer import GaitMetricsViewer
        from rehab_monitor.spatial_mapper import SpatialMapper
        from rehab_monitor.data_logger import RehabDatabase
        from rehab_gui.adapters.monitor_capture import patch_all_monitors
        from rehab_gui.gui.utils import put_chinese_text, put_chinese_texts  # 支持中文的 OpenCV 文本渲染
        from rehab_gui.config import resolve_pose_model_path

        # 模型路径 — 根据设备动态选择精度（cpu→int8, gpu→fp16, npu→fp16）
        model_name = config.get("model", "yolo26n")
        device = config.get("device", "cpu")
        model_path, precision = resolve_pose_model_path(
            model_name, device, os.environ.get("REHAB_MODEL_PRECISION")
        )
        logger.info(f"加载模型: {model_path} (设备={device}, 精度={precision})")

        # 初始化摄像头
        camera = cv2.VideoCapture(CAMERA_ID)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("✓ 摄像头已打开")
        init_dialog.set_step("✓ 摄像头就绪 — 加载 YOLO 模型...", 20)
        app.processEvents()

        # 初始化检测器
        pose_detector = PoseDetector(model_path, device, USE_KALMAN)
        if "npu" in device.lower() and "cpu" in pose_detector.device.lower():
            logger.warning("⚠ NPU 编译失败（模型算子不兼容），已回退到 CPU")
        logger.info("✓ 姿态检测器已初始化（实际设备: %s）", pose_detector.device)
        init_dialog.set_step(f"✓ YOLO 就绪 ({pose_detector.device}) — 加载人脸识别...", 40)
        app.processEvents()

        # 初始化分析器
        gait_analyzer = GaitAnalyzer()
        fall_detector = FallDetector()
        # 视觉跌倒检测初始开关（启动后可运行时切换）
        fall_detector.visual_enabled = config.get("visual_fall", True)
        if hasattr(window, '_status_panel') and hasattr(window._status_panel, '_visual_fall_cb'):
            window._status_panel._visual_fall_cb.setChecked(fall_detector.visual_enabled)
        display_overlay = DisplayOverlay()
        spatial_mapper = SpatialMapper()

        # 初始化人脸锁定器
        face_locker = None
        try:
            from rehab_monitor.face_locker import FaceNetLocker
            # 使用绝对路径，与 on_lock_target 中写入的路径保持一致
            import os as _os
            face_db_abs = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "model", "face_db.json")
            face_db_abs = _os.path.abspath(face_db_abs)
            # MTCNN 检脸仍用 CPU；FaceNet embedding 优先 OpenVINO NPU，失败回退 CPU。
            # 该路径只在锁定/重锁时低频触发，不参与每帧 YOLO 推理。
            face_locker = FaceNetLocker(db_path=face_db_abs, device="npu")
            logger.info("✓ 人脸锁定器已初始化 (db=%s)", face_db_abs)
            # 预热 FaceNet（MTCNN + 嵌入模型），避免首次搜索时才加载导致前60帧卡顿
            face_locker._ensure_loaded()
            logger.info("✓ 人脸锁定器已预热")
        except Exception as e:
            logger.warning(f"人脸锁定器初始化失败（可选功能）: {e}")

        init_dialog.set_step("✓ 人脸识别就绪 — 初始化传感器...", 55)
        app.processEvents()

        # 初始化呼吸检测器（定时触发模式：点击按钮 → 采集 30s → 计算 BPM → 显示结果）
        from rehab_monitor.breathing_detector import BreathingDetector
        breathing_detector = BreathingDetector(sigma=1.5, min_bpm=5, max_bpm=100, auto_correct=True)
        breath_state = "空闲"           # 空闲 / 检测中 / 完成
        breath_signal_buffer = []       # 运动信号缓冲
        breath_prev_chest_gray = None   # 上一帧胸部区域灰度图
        breath_start_time = 0.0         # 检测开始时刻（秒，用 time.time()）
        breath_result_bpm = 0.0         # 最终 BPM 结果
        BREATH_DURATION_SEC = 30        # 检测时长（秒）
        logger.info("✓ 呼吸检测器已初始化（按钮触发，30秒定时检测）")

        # 静默 CSI 连接超时的 ERROR 日志（ESP32 不在线是正常情况，必须放在传感器初始化之前）
        import logging as py_logging
        for csi_name in ("rehab.csi.receiver", "rehab.csi.monitor"):
            csi_log = py_logging.getLogger(csi_name)
            csi_log.setLevel(py_logging.CRITICAL)
            csi_log.propagate = False

        # 初始化外部传感器（手机/手环/CSI 跌倒检测）
        from rehab_gui.adapters.sensor_integration import init_external_sensors, SensorManager

        sensor_mgr = init_external_sensors(config)
        logger.info(f"✓ 外部传感器已初始化（手环={sensor_mgr.wristband_proc is not None}, "
                    f"手机={sensor_mgr.phone_data_source is not None}, "
                    f"CSI={sensor_mgr.csi_monitor is not None}）")
        init_dialog.set_step("✓ 传感器就绪 — 初始化数据库...", 70)
        app.processEvents()

        # 初始化数据库（异步写入）
        db = RehabDatabase(async_mode=True)
        db_session_id = db.start_session()
        logger.info(f"✓ 数据库已初始化 (session={db_session_id})")

        # 启动 API 服务器（供小程序使用）
        try:
            from rehab_monitor.api_server import start_api_server
            api_thread = start_api_server(
                gait_analyzer, spatial_mapper, db, face_locker,
                pose_detector, None,
                debug=False,
            )
            logger.info("✓ API 服务器已启动 (HTTP:5000, WS:5001)")
        except Exception as e:
            logger.warning(f"API 服务器启动失败（不影响核心功能）: {e}")

        init_dialog.set_step("✓ 数据库就绪 — 等待 LLM 加载...", 80)
        app.processEvents()

        # 人脸锁定状态
        target_locked = [False]      # 用 list 以便闭包修改
        target_name = [""]
        target_similarity = [0.0]
        lock_confidence = [0]
        latest_fall_status = ["safe"]   # 供 LLM 回调读取
        latest_fall_score = [0.0]
        latest_fall_source = ["摄像头"]  # 跌倒来源
        llm_active_worker = [None]       # 保持 LLMWorker 引用，防止 GC 回收
        chat_worker_ref = [None]        # 保持 LLMChatWorker 引用
        _llm_streaming = [False]        # LLM 是否正在生成 token（分析 or 对话）
        LOCK_ENTER_THRESH = 5
        LOCK_INCREMENT = 2
        LOCK_DECREMENT = 1
        LOCK_MAX = 30
        search_mode = [False]  # 搜索模式：人脸丢失后进入

        # 在实例化诊断监控器之前，patch 它们：抑制独立窗口 + 捕获 render 输出
        patch_all_monitors(bridge)

        # 初始化诊断监控器（patch 后不会创建 cv2 窗口）
        angle_monitor = AngleMonitor()
        trajectory_monitor = TrajectoryMonitor()
        skeleton_viewer = SkeletonViewer()
        gait_metrics_viewer = GaitMetricsViewer()

        logger.info("✓ 所有后端组件已初始化")

        # 更新右侧状态面板的模型/设备/精度信息
        actual_device = pose_detector.device if hasattr(pose_detector, 'device') else device
        if hasattr(window, '_status_panel'):
            window._status_panel.update_model_info(model_name, actual_device, precision)
            window._status_panel.update_llm_model_info(llm_model)

        # 创建帧处理函数
        frame_count = 0
        last_diag_frame = 0
        frame_times = []  # 最近 90 帧的耗时 (ms)，用于滚动平均和 P99
        latest_keypoints = [None]  # 保存最新关键点供锁定回调使用
        latest_raw_frame = [None]  # 保存最新原始帧（无骨架，用于干净人脸提取）
        prev_target_locked = [False]  # 上一帧的锁定状态，用于检测变化
        paused = [False]  # 暂停状态
        shutting_down = [False]  # 正在退出，阻止帧处理
        cleanup_done = [False]  # 防止 Ctrl+C 和 aboutToQuit 重复清理
        last_fall_alert_time = [0.0]  # 跌倒告警冷却时间
        camera_fail_count = [0]  # 摄像头失败计数
        CAM_MAX_RECONNECT = 10  # 最大重连次数
        frame_worker_thread = [None]  # 后台帧处理线程
        frame_stop_event = threading.Event()
        llm_health_state = ["off"]
        last_llm_health_check = [0.0]

        def process_frame():
            """处理单帧并更新 GUI"""
            nonlocal frame_count, last_diag_frame
            nonlocal breath_state, breath_prev_chest_gray, breath_signal_buffer
            nonlocal breath_start_time, breath_result_bpm
            nonlocal prev_target_locked

            # 暂停状态 — 跳过所有处理，但保持调度存活
            if paused[0]:
                frame_stop_event.wait(0.1)  # 暂停时低频轮询，不卡住 GUI
                return

            # 正在退出 — 不再处理帧，也不重新调度
            if shutting_down[0]:
                return

            try:
                # 帧耗时分析（每 30 帧打印一次明细）
                _t0 = time.time()
                _t_last = _t0
                _timings = {}  # 逐组件耗时 (ms)
                def _tick(label):
                    nonlocal _t_last
                    _now = time.time()
                    _timings[label] = (_now - _t_last) * 1000
                    _t_last = _now

                # 初始化默认值（人脸锁定代码会在跌倒检测之前引用 fall_status）
                fall_status = "safe"
                fall_score = 0.0

                ret, frame = camera.read()
                _tick("camera")
                if not ret:
                    # 摄像头断线重连
                    camera_fail_count[0] += 1
                    if camera_fail_count[0] <= CAM_MAX_RECONNECT:
                        logger.warning(f"摄像头读取失败 ({camera_fail_count[0]}/{CAM_MAX_RECONNECT})，1秒后重试...")
                        frame_stop_event.wait(1.0)
                        if camera_fail_count[0] == CAM_MAX_RECONNECT:
                            logger.warning("尝试回退到摄像头 0...")
                            camera.release()
                            camera.open(0)
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            camera_fail_count[0] = 0  # 重置计数，让新摄像头有重试机会
                    return
                camera_fail_count[0] = 0  # 成功后重置计数

                # 保存原始帧（无骨架，用于干净人脸提取）
                latest_raw_frame[0] = frame.copy()

                # 姿态检测
                # PoseDetector.process_frame 返回 (annotated_frame, keypoints)
                # annotated_frame 已含骨骼绘制，keypoints 为 (N, 17, 3) 或 None
                annotated_frame, keypoints = pose_detector.process_frame(frame)
                _tick("yolo")

                # 步态分析（取锁定目标或第一个人的关键点）
                gait_keypoints = None
                total_persons = 0  # 画面中检测到的总人数

                # 人脸锁定：多线索融合（人脸 + 人体外观），移植自 CLI 版 rehab_monitor/main.py
                if keypoints is not None and len(keypoints.shape) == 3 and keypoints.shape[0] > 0:
                    latest_keypoints[0] = keypoints  # 保存供锁定回调使用
                    kpts_all = keypoints  # 未过滤副本，供人脸恢复扫描使用
                    total_persons = kpts_all.shape[0]  # 保存原始人数（keypoints 可能被窄化）

                    if target_locked[0]:
                        # === 已锁定 — Sticky Lock：只验证当前目标，不搜索其他人 ===
                        target_det_idx = pose_detector.get_target_det_idx()
                        if target_det_idx is not None and target_det_idx < kpts_all.shape[0]:
                            gait_keypoints = kpts_all[target_det_idx]
                            if kpts_all.shape[0] > 1:
                                keypoints = kpts_all[target_det_idx:target_det_idx + 1]

                            # ---- 躺下直通：跳过外观验证，强制锁定（对齐 CLI）----
                            if spatial_mapper.is_lying_down:
                                lock_confidence[0] = min(LOCK_MAX, lock_confidence[0] + LOCK_INCREMENT)
                                target_similarity[0] = 1.0
                            else:
                                # 定期外观验证（人脸15帧一次，人体5帧一次）
                                do_face = frame_count % 15 == 0
                                do_body = frame_count % 5 == 0

                                if do_face or do_body:
                                    person_k = kpts_all[target_det_idx]
                                    face_emb = None
                                    if do_face:
                                        face_pts = person_k[[0, 1, 2, 3, 4], :2]
                                        valid_fp = face_pts[(face_pts[:, 0] > 0) & (face_pts[:, 1] > 0)]
                                        if len(valid_fp) >= 2:
                                            x1, y1 = valid_fp.min(axis=0).astype(int)
                                            x2, y2 = valid_fp.max(axis=0).astype(int)
                                            face_emb = face_locker.extract_embedding_from_roi(frame, x1, y1, x2, y2)

                                    if do_body or face_emb is not None:
                                        is_match, sim, source = face_locker.match_combined(frame, person_k, face_emb)
                                    else:
                                        is_match, sim, source = False, 0.0, "none"

                                    # 人脸恢复扫描：检查其他检测是否更匹配
                                    best_face_pid = -1
                                    best_face_sim = 0.0
                                    if do_face and kpts_all.shape[0] > 1:
                                        for pid in range(kpts_all.shape[0]):
                                            if pid == target_det_idx:
                                                continue
                                            pk = kpts_all[pid]
                                            fp = pk[[0, 1, 2, 3, 4], :2]
                                            vfp = fp[(fp[:, 0] > 0) & (fp[:, 1] > 0)]
                                            if len(vfp) >= 2:
                                                x1, y1 = vfp.min(axis=0).astype(int)
                                                x2, y2 = vfp.max(axis=0).astype(int)
                                                other_face = face_locker.extract_embedding_from_roi(frame, x1, y1, x2, y2)
                                                if other_face is not None:
                                                    name, fs = face_locker.match(other_face, threshold=0.50)
                                                    if name is not None and fs > best_face_sim:
                                                        best_face_sim = fs
                                                        best_face_pid = pid

                                    # 融合判定
                                    if best_face_pid >= 0 and best_face_sim > sim + 0.05:
                                        lock_confidence[0] = min(LOCK_MAX, lock_confidence[0] + LOCK_INCREMENT * 2)
                                        target_similarity[0] = best_face_sim
                                        keypoints = kpts_all[best_face_pid:best_face_pid + 1]
                                        gait_keypoints = kpts_all[best_face_pid]
                                        # 保护：Kalman 槽位满时 _det_to_track 无此检测索引
                                        track_id = pose_detector._det_to_track.get(best_face_pid)
                                        if track_id is not None:
                                            pose_detector.target_track_id = track_id
                                    elif is_match:
                                        lock_confidence[0] = min(LOCK_MAX, lock_confidence[0] + LOCK_INCREMENT)
                                        target_similarity[0] = sim
                                    elif fall_status == "alert":
                                        pass  # 跌倒期间不降置信度
                                    else:
                                        lock_confidence[0] = max(0, lock_confidence[0] - LOCK_DECREMENT)
                                        if lock_confidence[0] <= 0:
                                            logger.info("目标丢失（外观验证失败），进入搜索模式")
                                            search_mode[0] = True
                                            target_locked[0] = False
                                            target_similarity[0] = 0.0
                                            pose_detector.target_track_id = None
                                            bridge.push_metrics({
                                                "lock_status": False,
                                                "lock_name": "",
                                                "lock_similarity": 0.0,
                                                "lock_searching": True,
                                            })
                        else:
                            # 目标轨道消失（人离开画面）— 渐进衰减，避免轨道抖动导致反复重锁
                            if spatial_mapper.is_lying_down or fall_status == "alert":
                                pass  # 躺下/跌倒期间轨道可能暂时失配
                            else:
                                lock_confidence[0] = max(0, lock_confidence[0] - LOCK_DECREMENT)
                                if lock_confidence[0] <= 0:
                                    logger.info("目标丢失（轨道消失），进入搜索模式")
                                    search_mode[0] = True
                                    target_locked[0] = False
                                    target_similarity[0] = 0.0
                                    pose_detector.target_track_id = None
                                    bridge.push_metrics({
                                        "lock_status": False,
                                        "lock_name": "",
                                        "lock_similarity": 0.0,
                                        "lock_searching": True,
                                    })

                    elif search_mode[0]:
                        # === 搜索模式：每4帧扫全员（无全帧回退），上限2人 ===
                        # MTCNN 实测 68ms，全帧回退每人多跑一次导致 N× 放大。
                        # 策略：4帧间隔扫全员（最多2人），去掉全帧回退，兼顾速度与检测率。
                        if face_locker is not None and frame_count % 4 == 0:
                            # 搜索时放宽阈值（匹配 CLI 版行为：face=0.30, body=0.55）
                            _orig_face_th = face_locker.face_threshold
                            _orig_body_th = face_locker.body_threshold
                            face_locker.face_threshold = 0.30
                            face_locker.body_threshold = 0.55

                            best_pid = -1
                            best_sim = 0.0
                            best_source = "none"
                            best_emb = None
                            # 上限 2 人，防止多人场景 MTCNN 累积失控
                            max_scan = min(kpts_all.shape[0], 2)
                            for pid in range(max_scan):
                                person_k = kpts_all[pid]
                                face_emb = None
                                face_pts = person_k[[0, 1, 2, 3, 4], :2]
                                valid_fp = face_pts[(face_pts[:, 0] > 0) & (face_pts[:, 1] > 0)]
                                if len(valid_fp) >= 1:
                                    x1, y1 = valid_fp.min(axis=0).astype(int)
                                    x2, y2 = valid_fp.max(axis=0).astype(int)
                                    if x1 == x2: x2 = x1 + 40
                                    if y1 == y2: y2 = y1 + 40
                                    face_emb = face_locker.extract_embedding_from_roi(frame, x1, y1, x2, y2)
                                # 不再回退全帧 MTCNN——ROI 失败全帧也不会好
                                if face_emb is not None:
                                    is_match, sim, source = face_locker.match_combined(frame, person_k, face_emb)
                                    if is_match and sim > best_sim:
                                        best_sim = sim
                                        best_pid = pid
                                        best_source = source
                                        best_emb = face_emb

                            # 恢复原始阈值
                            face_locker.face_threshold = _orig_face_th
                            face_locker.body_threshold = _orig_body_th

                            if best_pid >= 0:
                                lock_confidence[0] = min(LOCK_MAX, lock_confidence[0] + LOCK_INCREMENT)
                                if lock_confidence[0] >= LOCK_ENTER_THRESH:
                                    # 保护：Kalman 槽位满时 _det_to_track 无此检测索引，不锁定
                                    track_id = pose_detector._det_to_track.get(best_pid)
                                    if track_id is None:
                                        logger.warning("重锁: Kalman槽位满，留在搜索模式 (pid=%d)", best_pid)
                                        lock_confidence[0] = max(0, lock_confidence[0] - LOCK_INCREMENT)
                                    else:
                                        target_locked[0] = True
                                        search_mode[0] = False
                                        # 重锁后设满置信度缓冲（30帧≈1秒），防止轨道抖动立即导致再次丢失→搜索循环
                                        lock_confidence[0] = LOCK_MAX
                                        target_name[0] = "target"
                                        target_similarity[0] = best_sim
                                        pose_detector.target_track_id = track_id
                                        gait_keypoints = kpts_all[best_pid]
                                        if kpts_all.shape[0] > 1:
                                            keypoints = kpts_all[best_pid:best_pid + 1]
                                        # 录入人体外观特征（后备匹配线索）
                                        face_locker.enroll_body(frame, kpts_all[best_pid])
                                        # 用摄像头高质量人脸更新数据库嵌入（对齐 CLI：同源匹配更可靠）
                                        if best_emb is not None:
                                            try:
                                                import json as _json
                                                face_db_path_update = os.path.join(
                                                    os.path.dirname(__file__), "..", "model", "face_db.json")
                                                face_db_path_update = os.path.abspath(face_db_path_update)
                                                if os.path.exists(face_db_path_update):
                                                    with open(face_db_path_update, "r") as _f:
                                                        _db = _json.load(_f)
                                                    for _e in _db.get("entries", []):
                                                        if _e["name"] == "target":
                                                            _e["embedding"] = best_emb.tolist()
                                                            break
                                                    with open(face_db_path_update, "w") as _f:
                                                        _json.dump(_db, _f)
                                                    face_locker._load_db()
                                                    logger.info("已用摄像头人脸更新 target 的嵌入")
                                            except Exception:
                                                pass
                                        logger.info(f"搜索模式：重新锁定目标 (检测#{best_pid}, sim={best_sim:.3f}, source={best_source})")
                                        bridge.push_metrics({
                                            "lock_status": True,
                                            "lock_name": "target",
                                            "lock_similarity": best_sim,
                                            "lock_searching": False,
                                        })
                            else:
                                # 搜索无匹配：衰减置信度（对齐 CLI）
                                lock_confidence[0] = max(0, lock_confidence[0] - LOCK_DECREMENT)
                                if frame_count % 60 == 0:
                                    logger.debug(f"搜索模式: 已扫描 {kpts_all.shape[0]} 人，均未匹配 (最高 sim={best_sim:.4f})")
                        else:
                            gait_keypoints = kpts_all[0] if kpts_all.shape[0] > 0 else None
                    else:
                        # 未锁定未搜索 — 默认取第一个人
                        gait_keypoints = kpts_all[0] if kpts_all.shape[0] > 0 else None

                    if gait_keypoints is not None and target_locked[0]:
                        gait_analyzer.update(gait_keypoints, time.time())

                _tick("lock+face")
                sensor_result = sensor_mgr.apply_to_fall_detector(fall_detector)

                # 传感器优先，视觉检测仅在传感器未告警 + 开关开启时运行
                if sensor_result[0] == "alert":
                    # 传感器已触发跌倒 → 直接使用
                    fall_status, fall_score, fall_source = sensor_result
                    latest_fall_status[0] = fall_status
                    latest_fall_score[0] = fall_score
                    latest_fall_source[0] = fall_source
                elif fall_detector.visual_enabled:
                    # 传感器未告警 + 视觉检测开启 → 摄像头推断
                    bbox = None
                    if keypoints is not None and len(keypoints.shape) == 3 and keypoints.shape[0] > 0:
                        first_person = keypoints[0]
                        high_conf_mask = first_person[:, 2] > 0.3
                        if len(high_conf_mask.shape) == 1 and np.any(high_conf_mask):
                            valid_points = first_person[high_conf_mask, :2]
                            if len(valid_points) > 0:
                                bbox = [float(valid_points[:, 0].min()), float(valid_points[:, 1].min()),
                                       float(valid_points[:, 0].max()), float(valid_points[:, 1].max())]
                    fall_status, fall_score = fall_detector.update(keypoints, bbox, 480)
                    latest_fall_status[0] = fall_status
                    latest_fall_score[0] = fall_score
                    latest_fall_source[0] = "摄像头"
                else:
                    # 传感器安全 + 视觉关闭 → 使用传感器结果
                    fall_status, fall_score, fall_source = sensor_result
                    latest_fall_status[0] = fall_status
                    latest_fall_score[0] = fall_score
                    latest_fall_source[0] = fall_source

                _tick("fall")
                # 只有锁定人脸后才提取
                clean_face = None
                if target_locked[0] and gait_keypoints is not None:
                    face_pts = gait_keypoints[[0, 1, 2, 3, 4], :2]
                    valid_fp = face_pts[(face_pts[:, 0] > 0) & (face_pts[:, 1] > 0)]
                    if len(valid_fp) >= 2:
                        x_min, y_min = valid_fp.min(axis=0).astype(int)
                        x_max, y_max = valid_fp.max(axis=0).astype(int)
                        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                        half = int(max(x_max - x_min, y_max - y_min) * 1.2)  # 120% 边距
                        # 使用原始帧（无骨架）提取干净人脸
                        raw = latest_raw_frame[0]
                        if raw is not None:
                            h_raw, w_raw = raw.shape[:2]
                            x1 = max(0, cx - half)
                            y1 = max(0, cy - half)
                            x2 = min(w_raw, cx + half)
                            y2 = min(h_raw, cy + half)
                            if x2 > x1 and y2 > y1:
                                clean_face = raw[y1:y2, x1:x2].copy()

                # 呼吸检测：定时触发模式（按钮 → 采集 30s → 计算 BPM → 显示）
                # 只有人脸锁定后才进行信号采集
                if breath_state == "检测中":
                    if target_locked[0] and gait_keypoints is not None:
                        chest_roi = BreathingDetector.get_chest_roi_from_keypoints(
                            gait_keypoints, frame.shape
                        )
                        if chest_roi is not None:
                            import cv2 as cv2_lib
                            cx, cy, cw, ch = chest_roi
                            cx = max(0, min(cx, frame.shape[1] - 1))
                            cy = max(0, min(cy, frame.shape[0] - 1))
                            cw = min(cw, frame.shape[1] - cx)
                            ch = min(ch, frame.shape[0] - cy)
                            if cw > 10 and ch > 10:
                                gray = cv2_lib.cvtColor(frame[cy:cy+ch, cx:cx+cw], cv2_lib.COLOR_BGR2GRAY)
                                if breath_prev_chest_gray is not None and gray.shape == breath_prev_chest_gray.shape:
                                    diff = cv2_lib.absdiff(gray, breath_prev_chest_gray)
                                    movement = np.mean(diff.astype(np.float64))
                                    breath_signal_buffer.append(movement)
                                breath_prev_chest_gray = gray.copy()
                            else:
                                breath_prev_chest_gray = None
                        else:
                            breath_prev_chest_gray = None
                    else:
                        breath_prev_chest_gray = None

                    # 检查是否到达采集时间（基于实际时间，不受帧率影响）
                    import time as time_module
                    breath_elapsed = time_module.time() - breath_start_time
                    if breath_elapsed >= BREATH_DURATION_SEC:
                        # 采集完毕，计算 BPM
                        if len(breath_signal_buffer) >= 5:
                            eps = len(breath_signal_buffer) / max(breath_elapsed, 0.1)
                            bpm = breathing_detector.calculate_bpm(
                                list(breath_signal_buffer), eps, method='fft')
                            breath_result_bpm = bpm
                        else:
                            breath_result_bpm = 0.0
                        breath_state = "完成"
                        logger.info(f"呼吸检测完成: BPM={breath_result_bpm:.1f}")

                # 构建显示状态（仅保留 OpenCV overlay 实际使用的字段）
                state = {
                    "fall_status": fall_status,
                    "fall_score": fall_score,
                    "frame_num": frame_count,
                    "angles": gait_analyzer.compute_joint_angles(gait_keypoints) if gait_keypoints is not None else {},
                    "keypoints": keypoints,  # 供朝向箭头 + 关节角度标注
                }

                # 渲染叠加层 — 在 annotated_frame (含骨骼) 上叠加 HUD
                display_frame = display_overlay.render(annotated_frame, state)

                # 推送到 GUI
                if display_frame is not None and display_frame.size > 0:
                    bridge.push_frame_count(frame_count)
                    bridge.push_frame(display_frame)
                    # 人数变化即时更新（仅1个标签，开销极小）
                    bridge.push_metrics({"person_count": total_persons})

                _tick("overlay+push")

                if fall_status != "safe":
                    import time as _time
                    _now = _time.time()
                    if _now - last_fall_alert_time[0] >= 5.0:  # 5秒冷却
                        bridge.push_fall(fall_status, fall_score, latest_fall_source[0])
                        last_fall_alert_time[0] = _now

                # 推送实时步态指标到状态面板（每5帧，避免每帧更新20+标签触发Qt重排）
                if frame_count % 5 == 0:
                    gait_metrics = gait_analyzer.get_metrics()
                    sensor_status = sensor_mgr.get_status()
                    bridge.push_metrics({
                        "person_count": total_persons,
                        "gait_metrics": gait_metrics,
                        "fall_status": fall_status,
                        "fall_score": fall_score,
                        "lock_status": target_locked[0],
                        "lock_name": target_name[0],
                        "lock_similarity": target_similarity[0],
                        "lock_searching": search_mode[0],
                        "sensor_status": sensor_status,
                    })

                # 异步写入数据库（每5帧，仅目标锁定时）
                if frame_count % 5 == 0 and target_locked[0] and gait_keypoints is not None:
                    try:
                        angles = gait_analyzer.compute_joint_angles(gait_keypoints) if gait_keypoints is not None else {}
                        db.write_frame_snapshot(frame_count, gait_keypoints, angles,
                                                fall_status, fall_score)
                        gm = gait_analyzer.get_metrics()
                        db.write_gait_metrics(
                            frame_count,
                            step_count=gm.get("step_count", 0),
                            cadence=gm.get("cadence_spm", 0),
                            speed=gm.get("gait_velocity_mps", 0),
                            symmetry=gm.get("symmetry", 1.0),
                            stride_length_m=gm.get("stride_length_m", 0),
                            gait_velocity_mps=gm.get("gait_velocity_mps", 0),
                            step_width_m=gm.get("avg_step_width_m", 0),
                            stance_percentage=gm.get("stance_percentage", 0),
                            left_knee_rom=gm.get("left_knee_rom", 0),
                            right_knee_rom=gm.get("right_knee_rom", 0),
                            foot_clearance_cm=gm.get("foot_clearance_cm", 0),
                            gait_rehab_score=gm.get("gait_rehab_score", 0),
                            trunk_sway_deg=gm.get("trunk_sway_deg", 0),
                            double_support_ratio=gm.get("double_support_ratio", 0),
                        )
                    except Exception as _e:
                        logger.debug(f"DB 写入失败: {_e}")

                # 推送干净人脸缩略图到状态面板（每5帧，降低推送频率）
                if clean_face is not None and clean_face.size > 0 and frame_count % 5 == 0:
                    bridge.push_face_thumbnail(clean_face)
                elif frame_count % 15 == 0:
                    # 人脸丢失时每15帧推一次空数据，清除旧缩略图
                    bridge.push_face_thumbnail(None)

                # 推送呼吸数据到状态面板（按需：检测中每帧推送进度，完成/空闲时每30帧推送一次）
                lock_changed = (target_locked[0] != prev_target_locked[0])
                prev_target_locked[0] = target_locked[0]
                has_person = (gait_keypoints is not None)  # 是否检测到人体

                if breath_state == "检测中":
                    import time as time_module
                    breath_elapsed = time_module.time() - breath_start_time
                    remaining_sec = max(0, int(BREATH_DURATION_SEC - breath_elapsed))
                    bridge.push_breathing("检测中", None, remaining_sec, BREATH_DURATION_SEC, has_person)
                elif breath_state == "完成" and (frame_count % 30 == 0 or lock_changed):
                    bridge.push_breathing("完成", breath_result_bpm, 0, BREATH_DURATION_SEC, has_person)
                elif breath_state == "空闲" and (frame_count % 30 == 0 or lock_changed):
                    bridge.push_breathing("空闲", None, 0, BREATH_DURATION_SEC, has_person)

                # 诊断窗口（每10帧更新+渲染一次；未锁定时显示占位，降低渲染负载）
                if frame_count - last_diag_frame >= 10:
                    last_diag_frame = frame_count

                    if not target_locked[0]:
                        # 未锁定 — 推送"人脸未锁定"占位
                        try:
                            placeholder = np.ones((300, 400, 3), dtype=np.uint8) * 28
                            put_chinese_text(placeholder, "人脸未锁定", (105, 145),
                                           color=(120, 120, 120), size=22)
                            put_chinese_text(placeholder, "请先锁定目标人物", (80, 180),
                                           color=(100, 100, 100), size=17)
                            bridge.push_diag_angle(placeholder)
                            bridge.push_diag_trajectory(placeholder)
                            bridge.push_diag_skeleton(placeholder)
                            bridge.push_diag_gait(placeholder)
                        except Exception as e:
                            logger.debug(f"诊断占位渲染失败: {e}")
                    else:
                        # 已锁定 — 正常渲染诊断数据

                        # 计算角度数据
                        angles = state.get("angles", {})

                        # 1. 角度监控器 — 传入角度字典
                        try:
                            if angles:
                                angle_monitor.update(angles)
                            angle_monitor.render()
                        except Exception as e:
                            logger.debug(f"角度监控器渲染失败: {e}")

                        # 2. 轨迹监控器 — 传入世界坐标和轨迹
                        try:
                            world_pos = None
                            trajectory = []
                            if gait_keypoints is not None:
                                world_pos = spatial_mapper.get_person_position(gait_keypoints)
                                trajectory = spatial_mapper.get_trajectory()
                            yaw = angles.get("orientation_yaw", 0.0)
                            yaw_conf = angles.get("orientation_conf", 0.0)
                            facing_sign = angles.get("facing_sign", 1.0)
                            trajectory_monitor.update(
                                world_pos, trajectory, yaw, yaw_conf, facing_sign,
                                spatial_mapper._torso_depth, spatial_mapper._ground_depth
                            )
                            trajectory_monitor.render()
                        except Exception as e:
                            logger.debug(f"轨迹监控器渲染失败: {e}")

                        # 3. 骨骼监控器 — 传入单人关键点和姿态状态
                        try:
                            if gait_keypoints is not None:
                                skeleton_viewer.update(gait_keypoints)
                            skeleton_viewer.render(
                                fall_status,
                                angles.get("orientation_yaw", 0.0),
                                spatial_mapper._torso_angle_deg,
                                spatial_mapper.is_lying_down
                            )
                        except Exception as e:
                            logger.debug(f"骨骼监控器渲染失败: {e}")

                        # 4. 步态指标监控器 — 传入步态指标字典
                        try:
                            gait_metrics_viewer.update(gait_analyzer.get_metrics())
                            gait_metrics_viewer.render()
                        except Exception as e:
                            logger.debug(f"步态指标渲染失败: {e}")

                        # 5. 呼吸波形 — 检测中显示实时波形，空闲/完成显示占位
                        try:
                            import cv2 as cv2_lib
                            bw, bh = 560, 320
                            canvas = np.ones((bh, bw, 3), dtype=np.uint8) * 28
                            cn_texts = []  # 收集需要渲染的中文文本

                            if breath_state == "检测中" and breath_signal_buffer and len(breath_signal_buffer) >= 3:
                                data = np.array(list(breath_signal_buffer), dtype=np.float64)
                                data = data - data.mean()
                                abs_max = np.abs(data).max()
                                if abs_max > 1e-8:
                                    data = data / abs_max
                                margin, plot_h_m = 30, 40
                                plot_w = bw - 2 * margin
                                plot_h = bh - 2 * margin - plot_h_m
                                center_y = margin + plot_h // 2
                                for i in range(5):
                                    gy = int(margin + i * plot_h / 4)
                                    cv2_lib.line(canvas, (margin, gy), (bw - margin, gy), (45, 45, 45), 1)
                                cv2_lib.line(canvas, (margin, center_y), (bw - margin, center_y), (80, 80, 80), 1)
                                step = plot_w / max(len(data) - 1, 1)
                                for i in range(1, len(data)):
                                    p1 = (int(margin + (i-1)*step), int(center_y - data[i-1] * plot_h // 2))
                                    p2 = (int(margin + i*step), int(center_y - data[i] * plot_h // 2))
                                    cv2_lib.line(canvas, p1, p2, (0, 220, 100), 2)

                            # 标题（英文）
                            cv2_lib.putText(canvas, "Breathing Waveform", (10, 16),
                                          cv2_lib.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                            if breath_state == "检测中":
                                import time as time_module
                                breath_elapsed = time_module.time() - breath_start_time
                                elapsed_s = int(breath_elapsed)
                                cn_texts.append((f"Recording: {elapsed_s}s / {BREATH_DURATION_SEC}s",
                                                (10, bh - 8), (140, 180, 140), 13))
                            elif breath_state == "完成":
                                bpm_text = f"BPM: {breath_result_bpm:.1f}" if breath_result_bpm > 0 else "BPM: --"
                                cv2_lib.putText(canvas, bpm_text, (bw - 130, 16),
                                              cv2_lib.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)
                                cn_texts.append(("检测完成 — 点击按钮重新检测", (10, bh - 8),
                                                (120, 180, 120), 13))
                            elif target_locked[0]:
                                cn_texts.append(("空闲 — 点击侧边栏「开始检测」", (10, bh - 8),
                                                (120, 120, 120), 14))
                            else:
                                cn_texts.append(("人脸未锁定 — 请先锁定目标", (10, bh - 8),
                                                (120, 100, 80), 14))

                            # 批量渲染中文文本（一次 PIL 转换）
                            if cn_texts:
                                put_chinese_texts(canvas, cn_texts)

                            bridge.push_diag_breath(canvas)
                        except Exception as e:
                            logger.debug(f"呼吸波形渲染失败: {e}")

                frame_count += 1
                _tick("diag+other")

                # 帧耗时分析：每 30 帧打印一次完整耗时分布
                if frame_count % 30 == 0:
                    _total = (time.time() - _t0) * 1000
                    # 更新滚动窗口
                    frame_times.append(_total)
                    if len(frame_times) > 90:
                        frame_times.pop(0)
                    # 滚动统计
                    _avg = sum(frame_times) / len(frame_times)
                    _sorted = sorted(frame_times)
                    _p99 = _sorted[int(len(_sorted) * 0.99)] if len(_sorted) > 1 else _total
                    # 组件耗时明细
                    _detail = " | ".join(f"{k}={v:.0f}ms" for k, v in _timings.items())
                    # LLM 状态: "streaming"=正在生成, "idle"=服务器运行但无任务, "off"=服务器未运行
                    if _llm_streaming[0]:
                        _llm_state = "streaming"
                    else:
                        try:
                            import urllib.request as _ur
                            _now_health = time.time()
                            if _now_health - last_llm_health_check[0] >= 5.0:
                                r = _ur.urlopen("http://127.0.0.1:8088/health", timeout=0.1)
                                llm_health_state[0] = "idle" if r.status == 200 else "off"
                                last_llm_health_check[0] = _now_health
                            _llm_state = llm_health_state[0]
                        except:
                            llm_health_state[0] = "off"
                            last_llm_health_check[0] = time.time()
                            _llm_state = "off"
                    logger.debug("⏱ Frame#%d: cur=%.1fms avg=%.1fms p99=%.1fms (proc≈%.0fFPS target=%dFPS) LLM=%s | %s",
                              frame_count, _total, _avg, _p99,
                              1000/_avg if _avg > 0 else 0,
                              TARGET_FPS,
                              _llm_state, _detail)

                return

            except Exception as e:
                import traceback as _tb
                logger.error(f"处理帧失败: {e}\n{_tb.format_exc()}")
                frame_stop_event.wait(max(0.001, TARGET_MS / 1000.0))

        # 自适应帧调度：后台线程处理视频帧，GUI 线程只负责事件和绘制。
        # 右下角 FPS 受这里的目标帧率限制，可用 REHAB_TARGET_FPS 调整。
        from rehab_gui.config import TARGET_FPS
        TARGET_MS = 1000.0 / max(1, TARGET_FPS)

        def frame_loop():
            """后台帧处理循环，避免推理/人脸识别阻塞 Qt 主线程。"""
            logger.info("帧处理线程已启动")
            while not frame_stop_event.is_set() and not shutting_down[0]:
                loop_t0 = time.time()
                process_frame()
                elapsed = (time.time() - loop_t0) * 1000
                delay = max(0.001, (TARGET_MS - elapsed) / 1000.0)
                frame_stop_event.wait(delay)
            logger.info("帧处理线程已退出")

        frame_worker_thread[0] = threading.Thread(
            target=frame_loop,
            name="RehabFrameWorker",
            daemon=True,
        )
        frame_worker_thread[0].start()

        logger.info("✓ 后端组件全部就绪")

        # ── 等待 LLM 服务器就绪（轮询 health，最长 120 秒）──
        init_dialog.set_step("等待 LLM 模型加载...", 85)
        app.processEvents()
        llm_deadline = time.time() + 120
        llm_loaded = False
        while time.time() < llm_deadline:
            try:
                r = urllib.request.urlopen(
                    f"http://127.0.0.1:{LLM_PORT}/health", timeout=2
                )
                if r.status == 200:
                    llm_loaded = True
                    break
            except Exception:
                pass
            time.sleep(1)
            app.processEvents()  # 保持 UI 响应

        if llm_loaded:
            init_dialog.set_step("✓ 全部就绪 — 启动中...", 100)
            logger.info("✓ LLM 服务器已就绪")
        else:
            init_dialog.set_step("⚠ LLM 未能在 120s 内就绪（后续可重试）", 100)
            logger.warning("LLM 服务器未能在 120s 内就绪")

        app.processEvents()
        time.sleep(0.3)  # 让用户看到 100%

        # ── 显示主窗口 ──
        window.show()
        init_dialog.close()
        logger.info("✓ GUI 已启动")

        # === 人脸锁定/解锁回调 ===
        def on_lock_target():
            """录入人脸锁定"""
            # 防止重复锁定
            if target_locked[0]:
                logger.info("人脸已锁定，忽略重复锁定请求")
                return

            if face_locker is None:
                QMessageBox.warning(window, "提示", "人脸锁定器未初始化\n可能缺少 facenet-pytorch 依赖")
                return

            # 使用当前显示的帧（与 CLI 行为一致），避免抓新帧导致人脸丢失
            lock_frame = latest_raw_frame[0]
            if lock_frame is None:
                QMessageBox.warning(window, "提示", "无画面数据，请稍后重试")
                return

            # 录入人脸特征
            emb = face_locker.enroll_from_frame(lock_frame)
            if emb is None:
                QMessageBox.warning(window, "提示", "未检测到人脸\n请正对摄像头并重试")
                return

            # 保存到特征库
            import json
            import os
            face_db_path = os.path.join(os.path.dirname(__file__), "..", "model", "face_db.json")
            face_db_path = os.path.abspath(face_db_path)
            db = {"entries": [{"name": "target", "embedding": emb.tolist()}]}
            os.makedirs(os.path.dirname(face_db_path), exist_ok=True)
            with open(face_db_path, "w") as f:
                json.dump(db, f)
            face_locker.db = db

            # 录入人体特征（使用最新帧的关键点）
            cur_kpts = latest_keypoints[0]
            if cur_kpts is not None and len(cur_kpts.shape) == 3 and cur_kpts.shape[0] > 0:
                face_locker.enroll_body(lock_frame, cur_kpts[0])

            # 绑定到第一个人（保护：Kalman 槽位满时跳过轨道绑定）
            track_id = pose_detector._det_to_track.get(0)
            if track_id is not None:
                pose_detector.target_track_id = track_id
            target_locked[0] = True
            target_name[0] = "target"
            target_similarity[0] = 1.0
            lock_confidence[0] = LOCK_MAX  # 手动锁定时设满缓冲，防止轨道抖动导致误解锁
            search_mode[0] = False  # 录入后退出搜索模式

            # 持久化锁定状态（重启后自动搜索）
            import time as _time
            lock_state_path = os.path.join(os.path.dirname(__file__), "..", "model", "lock_state.json")
            lock_state_path = os.path.abspath(lock_state_path)
            with open(lock_state_path, "w") as f:
                json.dump({"target_name": "target", "timestamp": _time.time()}, f)

            window.update_lock_status(True, "target", 1.0)
            logger.info("✓ 人脸锁定成功")

        def on_unlock_target():
            """解除人脸锁定"""
            target_locked[0] = False
            target_name[0] = ""
            target_similarity[0] = 0.0
            lock_confidence[0] = 0
            search_mode[0] = False
            pose_detector.target_track_id = None
            pose_detector.reset_tracking()

            # 清除持久化状态和人脸库
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for fname in ["model/lock_state.json", "model/face_db.json"]:
                try:
                    os.remove(os.path.join(project_root, fname))
                except OSError:
                    pass
            if face_locker is not None:
                face_locker._load_db()
                face_locker.reference_body_feat = None

            window.update_lock_status(False)
            logger.info("✓ 已解除人脸锁定")

        window.lock_target.connect(on_lock_target)
        window.unlock_target.connect(on_unlock_target)

        # 启动时恢复上次锁定状态（进入搜索模式等待人脸匹配）
        import json as _json, os as _os
        lock_state_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "model", "lock_state.json")
        if _os.path.exists(lock_state_path):
            try:
                with open(lock_state_path, "r") as _f:
                    _saved = _json.load(_f)
                saved_name = _saved.get("target_name")
                if saved_name:
                    target_locked[0] = False  # 不直接锁定，进入搜索模式
                    target_name[0] = saved_name
                    search_mode[0] = True
                    lock_confidence[0] = LOCK_ENTER_THRESH  # 搜索模式起始值（匹配后会自动设满 LOCK_MAX）
                    logger.info("恢复锁定目标: %s（搜索模式，等待人脸匹配）", saved_name)
            except Exception as _e:
                logger.debug("恢复锁定状态失败: %s", _e)

        # 呼吸检测开始回调
        def on_breath_start():
            nonlocal breath_state, breath_signal_buffer, breath_prev_chest_gray
            nonlocal breath_start_time
            if not target_locked[0]:
                logger.warning("呼吸检测需要先锁定人脸")
                window._status_bar.showMessage("请先锁定人脸再使用呼吸检测", 3000)
                return
            if latest_keypoints[0] is None:
                logger.warning("未检测到人体，无法开始呼吸检测")
                window._status_bar.showMessage("未检测到人体，无法开始呼吸检测", 3000)
                return
            if breath_state == "检测中":
                logger.warning("呼吸检测已在运行中")
                return
            breath_state = "检测中"
            breath_signal_buffer = []
            breath_prev_chest_gray = None
            import time as time_module
            breath_start_time = time_module.time()
            logger.info("呼吸检测开始（30秒定时采集）")

        window.breath_start_requested.connect(on_breath_start)

        # AI 分析（本地 LLM）回调
        def _format_report_json(report, raw_text):
            """将 JSON 报告格式化为可读中文"""
            if not isinstance(report, dict):
                return raw_text

            # 指标中英文映射
            METRIC_CN = {
                "avg_speed": "平均步速", "symmetry": "对称性",
                "grs_score": "GRS康复评分", "notable": "值得注意",
                "stride_length": "步长", "cadence": "步频",
                "step_width": "步宽", "double_support": "双支撑比",
                "trunk_sway": "躯干侧倾", "knee_rom": "膝关节活动度",
                "foot_clearance": "足廓清",
            }

            lines = []
            if report.get("summary"):
                lines.append(f"📋 摘要: {report['summary']}")
            if report.get("gait_assessment"):
                lines.append(f"🚶 步态评价: {report['gait_assessment']}")
            if report.get("fall_risk"):
                risk_cn = {"none": "无风险", "low": "低风险", "medium": "中风险", "high": "高风险"}
                risk_emoji = {"none": "🟢", "low": "🟡", "medium": "🟠", "high": "🔴"}
                label = risk_cn.get(report["fall_risk"], report["fall_risk"])
                emoji = risk_emoji.get(report["fall_risk"], "⚪")
                lines.append(f"{emoji} 跌倒风险: {label}")
            if report.get("recommendation"):
                lines.append(f"💡 建议: {report['recommendation']}")
            km = report.get("key_metrics", {})
            if km and isinstance(km, dict):
                lines.append("📊 关键指标:")
                for k, v in km.items():
                    cn = METRIC_CN.get(k, k)
                    lines.append(f"   • {cn}: {v}")
            return "\n".join(lines) if lines else raw_text

        def on_llm_analysis(report_type, data):
            """启动本地 LLM 分析（后台线程）"""
            from rehab_gui.core.llm_worker import LLMWorker

            # 防止重复启动
            if llm_active_worker[0] is not None:
                logger.warning("LLM 分析已在运行中，忽略重复请求")
                bridge.push_llm_status("running", "已有分析任务在运行中，请等待完成")
                return

            gait_metrics = data.get("gait_metrics", {})

            # 检查是否有步态数据
            if not gait_metrics or gait_metrics.get("step_count", 0) < 1:
                msg = ("⚠ 步态数据不足，无法生成分析报告。\n\n"
                       "请确保:\n"
                       "1. 已锁定目标人物\n"
                       "2. 目标人物在画面中行走\n"
                       "3. 等待步态指标更新后再试")
                window.update_llm_status("error", "步态数据不足")
                if hasattr(window._diagnostic_tabs, "_llm_widget"):
                    # 追加警告到对话，不清除已有内容
                    existing = window._diagnostic_tabs._llm_widget._text_edit.toPlainText()
                    if existing:
                        window._diagnostic_tabs._llm_widget._text_edit.append("\n" + msg)
                    else:
                        window._diagnostic_tabs._llm_widget._text_edit.setPlainText(msg)
                    window._diagnostic_tabs._llm_widget.show_error("步态数据不足")
                if hasattr(window._diagnostic_tabs, "switch_to_llm_tab"):
                    window._diagnostic_tabs.switch_to_llm_tab()
                return

            # 更新状态为运行中
            window.update_llm_status("running", "正在启动本地 LLM...")

            # 创建工作线程
            worker = LLMWorker(
                report_type=report_type,
                gait_metrics=gait_metrics,
                fall_status=latest_fall_status[0],
                fall_score=latest_fall_score[0],
            )

            def on_response(text, json_report):
                from rehab_monitor.llm_client import (
                    extract_reasoning_text,
                    format_reasoning_response,
                )
                # 格式化显示：如果是有效 JSON 则美化，否则显示原文
                if isinstance(json_report, dict) and "raw" not in json_report:
                    # 有效 JSON → 格式化显示
                    final_text = _format_report_json(json_report, text)
                    reasoning = extract_reasoning_text(text)
                    formatted = format_reasoning_response(reasoning, final_text)
                else:
                    formatted = text
                window.finish_llm_text(formatted)
                window.update_llm_status("done", "分析完成")
                # 同步到 chat_store，小程序也能看到
                try:
                    from rehab_monitor.chat_store import chat_store
                    chat_store.add_message("user", "📋 请求生成康复分析报告")
                    chat_store.add_message("assistant", formatted)
                except Exception:
                    pass
                if hasattr(window, '_diagnostic_tabs') and hasattr(window._diagnostic_tabs, 'switch_to_llm_tab'):
                    window._diagnostic_tabs.switch_to_llm_tab()

            def on_partial(text):
                window.update_llm_partial(text)

            def on_error(error_msg):
                window.update_llm_status("error", error_msg)
                if hasattr(window._diagnostic_tabs, "_llm_widget"):
                    window._diagnostic_tabs._llm_widget.show_error(error_msg)
                _llm_streaming[0] = False

            def on_progress(msg):
                if msg == "thinking":
                    if hasattr(window._diagnostic_tabs, "_llm_widget"):
                        window._diagnostic_tabs._llm_widget.show_thinking()
                else:
                    bridge.push_llm_status("running", msg)

            def on_finished():
                llm_active_worker[0] = None
                _llm_streaming[0] = False

            worker.response_ready.connect(on_response)
            worker.partial_response.connect(on_partial)
            worker.error_occurred.connect(on_error)
            worker.progress_update.connect(on_progress)
            worker.finished.connect(on_finished)

            # 保持引用，防止 Python GC 在 QThread 运行时回收对象
            llm_active_worker[0] = worker

            # 在对话区追加分析报告头部（不清空历史记录）
            if hasattr(window._diagnostic_tabs, "_llm_widget"):
                window._diagnostic_tabs._llm_widget.start_analysis()
                window._diagnostic_tabs.switch_to_llm_tab()

            worker.start()
            _llm_streaming[0] = True

            logger.info(f"LLM 分析已启动 (类型={report_type})")

        window.llm_analysis_requested.connect(on_llm_analysis)

        # 对话回调（用户打字 → 本地 LLM 回复）

        def on_chat_message(text):
            """处理用户对话消息"""
            from rehab_gui.core.llm_worker import LLMChatWorker

            # 防止重复发送
            if chat_worker_ref[0] is not None:
                logger.warning("对话正在生成回复中，忽略重复消息")
                return

            # 获取对话历史
            history = window.get_chat_history()
            # 添加当前消息到历史
            history.append({"role": "user", "content": text})
            window.append_chat_history("user", text)

            # 更新状态 — 启动思考动画
            if hasattr(window._diagnostic_tabs, "_llm_widget"):
                window._diagnostic_tabs._llm_widget.show_running("思考中...")

            # 创建对话线程
            worker = LLMChatWorker(messages=history, model_key=llm_model)

            def on_partial(partial_text):
                window.update_chat_stream(partial_text)
                # 收到第一个 token 后停止思考动画
                if hasattr(window._diagnostic_tabs, "_llm_widget"):
                    window._diagnostic_tabs._llm_widget._stop_thinking()

            def on_response(full_text):
                window.finish_chat_stream(full_text)
                window.append_chat_history("assistant", full_text)
                window.update_llm_status("done", "对话完成")

            def on_error(error_msg):
                window.update_llm_status("error", error_msg)
                if hasattr(window._diagnostic_tabs, "_llm_widget"):
                    window._diagnostic_tabs._llm_widget.show_error(error_msg)
                _llm_streaming[0] = False

            def on_finished():
                chat_worker_ref[0] = None
                _llm_streaming[0] = False

            def on_progress(msg):
                if msg == "thinking" and hasattr(window._diagnostic_tabs, "_llm_widget"):
                    window._diagnostic_tabs._llm_widget.show_thinking()

            worker.partial_response.connect(on_partial)
            worker.response_ready.connect(on_response)
            worker.error_occurred.connect(on_error)
            worker.finished.connect(on_finished)
            worker.progress_update.connect(on_progress)

            chat_worker_ref[0] = worker
            _llm_streaming[0] = True
            worker.start()
            logger.info(f"对话已启动 (消息: {text[:30]}...)")

        window.llm_chat_requested.connect(on_chat_message)

        # === 系统控制回调（CLI 快捷键 → GUI 按钮） ===

        def on_pause_toggle():
            """暂停/恢复画面处理"""
            if paused[0]:
                paused[0] = False
                window._status_bar.showMessage("▶ 已恢复", 2000)
                if hasattr(window._status_panel, "update_pause_button"):
                    window._status_panel.update_pause_button(False)
            else:
                paused[0] = True
                window._status_bar.showMessage("⏸ 已暂停", 2000)
                if hasattr(window._status_panel, "update_pause_button"):
                    window._status_panel.update_pause_button(True)

        def on_reset():
            """重置 Kalman/步态/跌倒/人脸锁定"""
            pose_detector.reset_tracking()
            nonlocal gait_analyzer, fall_detector
            gait_analyzer = GaitAnalyzer()
            fall_detector = FallDetector()

            # 清除人脸数据
            import os as _os
            project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            for fname in ["model/lock_state.json", "model/face_db.json"]:
                try:
                    _os.remove(_os.path.join(project_root, fname))
                except OSError:
                    pass
            if face_locker is not None:
                face_locker._load_db()
                face_locker.reference_body_feat = None

            target_locked[0] = False
            target_name[0] = ""
            target_similarity[0] = 0.0
            lock_confidence[0] = 0
            search_mode[0] = False
            pose_detector.target_track_id = None

            # 清除步态数据库
            try:
                db.flush()
                db.clear_gait_data()
                if hasattr(gait_metrics_viewer, "reset"):
                    gait_metrics_viewer.reset()
                    gait_metrics_viewer.render()
                if hasattr(window, "clear_gait_metrics"):
                    window.clear_gait_metrics()
            except Exception as _e:
                logger.debug(f"清除步态数据失败: {_e}")

            window.update_lock_status(False)
            window._status_bar.showMessage("🔄 已重置 (Kalman/步态/人脸)", 3000)
            logger.info("✓ 系统已重置")

        def on_screenshot():
            """保存当前画面为 JPEG"""
            import time as _time
            raw = latest_raw_frame[0]
            if raw is None or raw.size == 0:
                window._status_bar.showMessage("⚠ 无画面可截图", 3000)
                return
            filename = f"rehab_screenshot_{_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, raw)
            window._status_bar.showMessage(f"📸 已保存: {filename}", 3000)
            logger.info(f"截图已保存: {filename}")

        def on_clear_gait():
            """清除步态数据库和当前内存步态状态"""
            nonlocal gait_analyzer
            try:
                db.flush()
                db.clear_gait_data()
                gait_analyzer = GaitAnalyzer()
                if hasattr(gait_metrics_viewer, "reset"):
                    gait_metrics_viewer.reset()
                    gait_metrics_viewer.render()
                if hasattr(window, "clear_gait_metrics"):
                    window.clear_gait_metrics()
                bridge.push_metrics({"gait_metrics": {}, "person_count": 0})
                window._status_bar.showMessage("🗑 步态数据已清除", 3000)
                logger.info("步态数据已清除（数据库 + 当前内存状态）")
            except Exception as _e:
                window._status_bar.showMessage(f"清除失败: {_e}", 3000)
                logger.error(f"清除步态数据失败: {_e}")

        def on_fps_toggle():
            """切换 FPS 标签显隐"""
            visible = window._fps_label.isVisible()
            window._fps_label.setVisible(not visible)
            window._status_bar.showMessage(
                f"FPS 显示: {'关' if visible else '开'}", 2000)

        # 连接系统控制信号
        window.pause_toggle_requested.connect(on_pause_toggle)
        window.reset_requested.connect(on_reset)
        window.screenshot_requested.connect(on_screenshot)
        window.clear_gait_requested.connect(on_clear_gait)
        window.fps_toggle_requested.connect(on_fps_toggle)
        window.help_requested.connect(window._on_help)

        # 视觉跌倒检测运行时切换
        def on_visual_fall_toggled(checked):
            fall_detector.visual_enabled = checked
            status = "开启" if checked else "关闭"
            logger.info(f"视觉跌倒检测已{status}")
            window._status_bar.showMessage(f"视觉跌倒检测: {status}", 2000)
            # 切换时重置全部跌倒状态，避免旧检测残留
            fall_detector.status = "safe"
            fall_detector.score = 0.0
            fall_detector._alert_frames = 0
            fall_detector._safe_frames = 0
            fall_detector._prev_head_y = None
            fall_detector._prev_time = None
            fall_detector._velocity_history.clear()
            fall_detector._long_vel_history.clear()
        window.visual_fall_toggled.connect(on_visual_fall_toggled)

    except Exception as e:
        init_dialog.accept()
        QMessageBox.critical(window, "错误", f"初始化失败:\n{e}")
        logger.error(f"初始化失败: {e}")
        app.quit()
        return

    # 存储 DB 引用供信号处理器使用
    _db_ref = db

    def _wait_llm_worker(ref, name):
        worker = ref[0]
        if worker is None:
            return
        try:
            if worker.isRunning():
                logger.info("等待 %s 线程退出...", name)
                worker.requestInterruption()
                if not worker.wait(5000):
                    logger.warning("%s 线程未及时退出，强制结束", name)
                    worker.terminate()
                    worker.wait(2000)
        except RuntimeError:
            pass
        finally:
            ref[0] = None
            _llm_streaming[0] = False

    def _cleanup():
        """清理退出"""
        if cleanup_done[0]:
            return
        cleanup_done[0] = True
        shutting_down[0] = True  # 阻止后续帧处理
        _stop_llm_server()  # 先停 HTTP 端，打断正在阻塞的 LLM 请求
        _wait_llm_worker(llm_active_worker, "LLM 分析")
        _wait_llm_worker(chat_worker_ref, "LLM 对话")
        frame_stop_event.set()
        worker = frame_worker_thread[0]
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        try:
            sensor_mgr.close()
        except Exception:
            logger.debug("传感器清理失败", exc_info=True)
        try:
            db.end_session()
            db.close()
        except Exception:
            pass
        try:
            camera.release()
        except Exception:
            pass

    signal.signal(signal.SIGINT, lambda *args: (_cleanup(), app.quit()))
    signal.signal(signal.SIGTERM, lambda *args: (_cleanup(), app.quit()))
    app.aboutToQuit.connect(_cleanup)  # 正常关闭（点X）也清理 llama-server
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
