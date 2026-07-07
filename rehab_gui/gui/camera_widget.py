"""摄像头画面组件 — 显示 OpenCV 帧 + 叠加层"""
import cv2
import numpy as np
import time
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel

from rehab_gui.gui.utils import mat_to_qimage, create_placeholder_frame
from rehab_monitor.logging_setup import get_logger

logger = get_logger("camera_widget")


class CameraWidget(QLabel):
    """摄像头画面组件 — 显示 OpenCV 帧 + 叠加层"""

    def __init__(self, parent=None):
        """初始化摄像头组件

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.setMinimumSize(320, 240)
        # 注意: 不设置 background-color, 否则会覆盖 pixmap 显示
        self.setStyleSheet(
            "border: 1px solid #3c3c3c; border-radius: 3px;"
        )

        self._frame_count = 0
        self._display_samples = []

        # 占位帧
        self._placeholder = create_placeholder_frame(640, 480, "等待画面...")
        self._show_frame(self._placeholder)

        logger.info("摄像头组件已初始化")

    def update_frame(self, frame):
        """更新画面帧

        Args:
            frame: np.ndarray OpenCV BGR 格式帧 (含 DisplayOverlay 叠加层)
        """
        if frame is None:
            return

        try:
            self._show_frame(frame)
        except Exception as e:
            logger.warning(f"画面更新失败: {e}, 跳过当前帧")

    def _show_frame(self, frame):
        """显示画面帧 — 用 OpenCV 缩放代替 Qt SmoothTransformation

        Args:
            frame: np.ndarray OpenCV BGR 格式帧
        """
        t0 = time.perf_counter()
        t_last = t0
        timings = {}

        def tick(label):
            nonlocal t_last
            now = time.perf_counter()
            timings[label] = (now - t_last) * 1000
            t_last = now

        h, w = frame.shape[:2]
        tw, th = self.width(), self.height()
        # 只在窗口比视频小的时候缩小；不放大 640x480 视频，避免每帧 resize 拖慢 GUI。
        if (tw < w or th < h) and tw > 0 and th > 0:
            scale = min(tw / w, th / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        tick("resize")

        qimage = self._mat_to_qimage_fast(frame)
        tick("qimage")
        if qimage is not None and not qimage.isNull():
            pixmap = QPixmap.fromImage(qimage)
            tick("pixmap")
            self.setPixmap(pixmap)
            tick("set")
        else:
            logger.warning("QImage 转换失败或为空")
            return

        total = (time.perf_counter() - t0) * 1000
        self._frame_count += 1
        self._display_samples.append(total)
        if len(self._display_samples) > 120:
            self._display_samples.pop(0)
        if self._frame_count % 120 == 0:
            avg = sum(self._display_samples) / len(self._display_samples)
            worst = max(self._display_samples)
            detail = " | ".join(f"{k}={v:.1f}ms" for k, v in timings.items())
            logger.info("显示耗时: cur=%.1fms avg=%.1fms max=%.1fms | %s",
                        total, avg, worst, detail)

    def _mat_to_qimage_fast(self, frame):
        """OpenCV BGR 帧快速转 QImage，优先避免 BGR→RGB 复制。"""
        if frame is None:
            return None
        if len(frame.shape) == 2:
            return mat_to_qimage(frame)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return mat_to_qimage(frame)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        try:
            return QImage(
                frame.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_BGR888,
            )
        except Exception:
            return mat_to_qimage(frame)

    def clear(self):
        """清除画面, 显示占位符"""
        self._show_frame(self._placeholder)
