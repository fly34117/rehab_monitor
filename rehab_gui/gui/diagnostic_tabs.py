"""诊断标签页 — 5个Tab整合 (角度/轨迹/骨骼/步态/呼吸) + AI回答
支持双击放大查看"""
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QMouseEvent, QFont
from PyQt6.QtWidgets import (
    QTabWidget, QLabel, QVBoxLayout, QDialog,
    QPushButton, QHBoxLayout, QScrollArea, QWidget,
    QGraphicsDropShadowEffect,
)

from rehab_gui.gui.utils import mat_to_qimage, create_placeholder_frame
from rehab_monitor.logging_setup import get_logger

logger = get_logger("diagnostic_tabs")


class FullscreenPreview(QDialog):
    """诊断画面全屏预览 — 双击 Tab 时弹出"""

    def __init__(self, title, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(900, 700)
        self._pixmap = pixmap
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 标题
        title_label = QLabel(self.windowTitle())
        title_label.setFont(QFont("WenQuanYi Micro Hei", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #d4d4d4; padding: 4px;")
        layout.addWidget(title_label)

        # 图片滚动区（缩放显示）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background-color: #1e1e1e; border: 1px solid #3c3c3c; }"
        )

        # 图片标签 — 放大显示
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_scaled_pixmap()
        scroll.setWidget(self._img_label)

        layout.addWidget(scroll, stretch=1)

        # 关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        close_btn = QPushButton("关闭 (Esc)")
        close_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border: none; "
            "padding: 8px 24px; border-radius: 3px; font-size: 13px; }"
            "QPushButton:hover { background-color: #1177bb; }"
        )
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _update_scaled_pixmap(self):
        """根据窗口大小缩放图像"""
        if self._pixmap and not self._pixmap.isNull():
            # 放大到窗口的 90%
            target_w = int(self.width() * 0.9)
            target_h = int(self.height() * 0.85)
            if target_w > 0 and target_h > 0:
                scaled = self._pixmap.scaled(
                    target_w, target_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self._img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        """窗口大小变化时重新缩放图像"""
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class ClickableLabel(QLabel):
    """可点击/双击的标签 — 用于诊断 Tab 内容"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = title
        self._current_pixmap = None       # 显示用（已缩放）
        self._original_pixmap = None      # 原始分辨率（双击放大用）
        self._parent_widget = parent

    def set_title(self, title):
        self._title = title

    def setPixmap(self, pixmap):
        """设置缩放后的显示图像"""
        self._current_pixmap = pixmap
        super().setPixmap(pixmap)

    def setOriginalPixmap(self, pixmap):
        """保存原始分辨率图像，用于双击放大"""
        self._original_pixmap = pixmap

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._original_pixmap:
            if not self._original_pixmap.isNull():
                # 使用原始分辨率 pixmap 放大
                preview = FullscreenPreview(
                    self._title or "诊断画面",
                    self._original_pixmap,
                    self.window()
                )
                preview.show()
        super().mouseDoubleClickEvent(event)


class DiagnosticTabs(QTabWidget):
    """诊断标签页 — 整合5个诊断窗口 + AI 回答

    Tab 1: 关节角度 (Angle Monitor)
    Tab 2: 俯视轨迹 (Trajectory)
    Tab 3: 骨骼预览 (Skeleton)
    Tab 4: 步态指标 (Gait Metrics)
    Tab 5: 呼吸波形 (Breath Wave)
    Tab 6: AI 回答 (LLM Response)

    双击前5个 Tab 内容可放大查看。
    """

    # Tab 名称
    TAB_ANGLE = "关节角度"
    TAB_TRAJECTORY = "俯视轨迹"
    TAB_SKELETON = "骨骼预览"
    TAB_GAIT = "步态指标"
    TAB_BREATH = "呼吸波形"
    TAB_LLM = "🤖 AI 回答"

    def __init__(self, parent=None):
        """初始化诊断标签页

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTabPosition(QTabWidget.TabPosition.South)
        self.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3c3c3c;"
        )
        self.setMinimumHeight(220)

        # 记录每个 Tab 的最新帧（用于双击放大）
        self._latest_frames = {}

        self._setup_tabs()

    def _setup_tabs(self):
        """构建6个诊断标签页"""
        self._angle_widget = self._create_tab(self.TAB_ANGLE)
        self._trajectory_widget = self._create_tab(self.TAB_TRAJECTORY)
        self._skeleton_widget = self._create_tab(self.TAB_SKELETON)
        self._gait_widget = self._create_tab(self.TAB_GAIT)
        self._breath_widget = self._create_tab(self.TAB_BREATH)

        # AI 回答 Tab — 使用 LLMResponseWidget（文本显示，非图像）
        from rehab_gui.gui.llm_response_widget import LLMResponseWidget
        self._llm_widget = LLMResponseWidget()
        self.addTab(self._llm_widget, self.TAB_LLM)

        self._show_placeholder(self._angle_widget, "人脸未锁定")
        self._show_placeholder(self._trajectory_widget, "人脸未锁定")
        self._show_placeholder(self._skeleton_widget, "人脸未锁定")
        self._show_placeholder(self._gait_widget, "人脸未锁定")
        self._show_placeholder(self._breath_widget, "点击「开始检测」启动呼吸检测")

        logger.info("诊断标签页已初始化 (6个Tab, 双击可放大)")

    def _create_tab(self, title):
        """创建一个标签页

        Args:
            title: 标签页标题

        Returns:
            ClickableLabel: 可双击放大的图像面板
        """
        widget = ClickableLabel(title=title, parent=self)
        widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3c3c3c; border-radius: 2px;"
            "font-size: 12px; color: #666;"
        )
        self.addTab(widget, title)
        return widget

    def _show_placeholder(self, widget, text="无数据"):
        """显示占位帧

        Args:
            widget: ClickableLabel 面板
            text: 占位文本
        """
        frame = create_placeholder_frame(400, 300, text)
        qimage = mat_to_qimage(frame)
        if qimage:
            widget.setPixmap(QPixmap.fromImage(qimage))

    def update_angle(self, frame):
        """更新关节角度画面"""
        if frame is not None:
            self._latest_frames[self.TAB_ANGLE] = frame.copy()
            self._update_image(self._angle_widget, frame)

    def update_trajectory(self, frame):
        """更新俯视轨迹画面"""
        if frame is not None:
            self._latest_frames[self.TAB_TRAJECTORY] = frame.copy()
            self._update_image(self._trajectory_widget, frame)

    def update_skeleton(self, frame):
        """更新骨骼预览画面"""
        if frame is not None:
            self._latest_frames[self.TAB_SKELETON] = frame.copy()
            self._update_image(self._skeleton_widget, frame)

    def update_gait(self, frame):
        """更新步态指标画面"""
        if frame is not None:
            self._latest_frames[self.TAB_GAIT] = frame.copy()
            self._update_image(self._gait_widget, frame)

    def update_breath(self, frame):
        """更新呼吸波形画面"""
        if frame is not None:
            self._latest_frames[self.TAB_BREATH] = frame.copy()
            self._update_image(self._breath_widget, frame)

    def _update_image(self, widget, frame):
        """更新面板图像

        Args:
            widget: ClickableLabel 面板
            frame: np.ndarray OpenCV BGR 格式帧
        """
        try:
            qimage = mat_to_qimage(frame)
            if qimage is not None and not qimage.isNull():
                original = QPixmap.fromImage(qimage)
                # 保存原始分辨率（双击放大用）
                widget.setOriginalPixmap(original)
                # 缩放以适应当前 Tab 大小，保持宽高比
                scaled = original.scaled(
                    widget.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                widget.setPixmap(scaled)
        except Exception:
            logger.warning("诊断面板图像更新失败")

    # ---- LLM 回答 Tab 接口 ----

    def update_llm_response(self, text):
        """设置 LLM 回答文本

        Args:
            text: LLM 生成的完整报告文本
        """
        if hasattr(self, '_llm_widget'):
            self._llm_widget.set_text(text)

    def update_llm_status(self, status, message=""):
        """更新 LLM 回答 Tab 的状态

        Args:
            status: "idle" / "running" / "done" / "error"
            message: 状态描述
        """
        if not hasattr(self, '_llm_widget'):
            return
        if status == "running":
            self._llm_widget.show_running(message)
        elif status == "done":
            self._llm_widget.show_done(message)
        elif status == "error":
            self._llm_widget.show_error(message)
        else:
            self._llm_widget.show_idle()

    def switch_to_llm_tab(self):
        """切换到 AI 回答 Tab"""
        if hasattr(self, '_llm_widget'):
            for i in range(self.count()):
                if self.tabText(i) == self.TAB_LLM:
                    self.setCurrentIndex(i)
                    break
