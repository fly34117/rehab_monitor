"""GUI 入口点 - 简化版集成模式"""
import sys
import os

# 设置环境变量
os.environ["LD_LIBRARY_PATH"] = "/home/ubuntu224/miniconda3/envs/yolov26/lib/python3.11/site-packages/openvino/libs"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["REHAB_MODEL"] = "yolo26n"
os.environ["REHAB_DEVICE"] = "cpu"

from PyQt6.QtWidgets import QApplication, QMessageBox, QVBoxLayout, QLabel, QPushButton, QDialog
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QImage
import numpy as np

sys.path.insert(0, '.')

from rehab_gui.gui.main_window import MainWindow
from rehab_gui.core.thread_bridge import ThreadBridge
from rehab_monitor.logging_setup import get_logger

logger = get_logger("gui_main")


class SimpleBackendIntegrator:
    """简化的后端集成器"""

    def __init__(self, bridge):
        self._bridge = bridge
        self._running = False
        self._camera = None
        self._pose_detector = None

        # 延迟初始化的组件
        self._gait_analyzer = None
        self._fall_detector = None
        self._display_overlay = None

        self._frame_count = 0

    def initialize_minimal(self):
        """最小化初始化（只初始化摄像头和姿态检测）"""
        import cv2
        from rehab_monitor.config import CAMERA_ID
        from rehab_monitor.pose_detector import PoseDetector

        # 根据环境变量设置模型
        model_name = os.environ.get("REHAB_MODEL", "yolo26n")
        device = os.environ.get("REHAB_DEVICE", "cpu")

        # 查找模型路径
        from rehab_gui.config import MODEL_DIR, MODEL_PRECISION
        precision = MODEL_PRECISION.get(model_name, {}).get(device, "int8")
        model_path = os.path.join(MODEL_DIR, f"{model_name}-pose_{precision}_openvino_model")

        logger.info(f"加载模型: {model_path}")

        # 打开摄像头
        self._camera = cv2.VideoCapture(CAMERA_ID)
        if not self._camera.isOpened():
            raise RuntimeError("无法打开摄像头")

        # 初始化姿态检测
        self._pose_detector = PoseDetector(model_path, device, USE_KALMAN=True)

        logger.info("✓ 最小化初始化完成")
        self._running = True

    def process_frame(self):
        """处理单帧"""
        if not self._running or self._camera is None:
            return

        try:
            import cv2

            ret, frame = self._camera.read()
            if not ret:
                return

            # 姿态检测
            annotated_frame, keypoints = self._pose_detector.process_frame(frame)
            person_count = keypoints.shape[0] if keypoints is not None else 0

            # 添加状态文本
            cv2.putText(annotated_frame, f"Frame: {self._frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if person_count > 0:
                cv2.putText(annotated_frame, f"Persons: {person_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 推送到 GUI
            self._bridge.push_frame_count(self._frame_count)
            self._bridge.push_frame(annotated_frame)
            self._bridge.push_metrics({"person_count": person_count})

            self._frame_count += 1

        except Exception as e:
            logger.error(f"处理帧失败: {e}")


def main():
    """主入口"""
    # 创建 QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Rehab Monitor")
    app.setFont(QFont("WenQuanYi Micro Hei", 13))

    # 加载主题
    theme_path = "rehab_gui/gui/dark_theme.qss"
    try:
        with open(theme_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except:
        pass

    # 配置
    config = {
        "model": "yolo26n",
        "device": "cpu",
    }

    # 创建主窗口
    window = MainWindow(config=config)
    window.show()

    # 创建桥接器
    bridge = ThreadBridge()
    bridge.start()
    window.set_bridge(bridge)

    # 显示初始化对话框
    dialog = QDialog(window)
    dialog.setWindowTitle("初始化")
    dialog.setFixedSize(300, 100)
    layout = QVBoxLayout(dialog)
    label = QLabel("正在初始化摄像头和模型...")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    dialog.show()

    # 后台初始化
    backend = SimpleBackendIntegrator(bridge)

    def delayed_init():
        """延迟初始化"""
        try:
            backend.initialize_minimal()
            dialog.accept()

            # 启动定时器（40ms）
            timer = QTimer()
            timer.timeout.connect(backend.process_frame)
            timer.start(40)

            logger.info("✓ GUI 已启动（简化集成模式）")

        except Exception as e:
            dialog.accept()
            QMessageBox.critical(window, "错误", f"初始化失败:\n{e}")
            logger.error(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()

    # 100ms 后初始化（让 GUI 先显示）
    QTimer.singleShot(100, delayed_init)

    logger.info("✓ 程序已启动")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
