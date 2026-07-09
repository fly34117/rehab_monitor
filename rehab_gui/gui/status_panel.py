"""状态面板 — 跌倒/步态/锁定/模型信息/呼吸检测"""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QMouseEvent
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QGroupBox,
    QScrollArea,
    QWidget,
    QPushButton,
    QCheckBox,
    QDialog,
)

from rehab_gui.gui.utils import mat_to_qimage
from rehab_monitor.logging_setup import get_logger

logger = get_logger("status_panel")


class FullscreenFacePreview(QDialog):
    """人脸缩略图全屏预览"""

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("人脸预览")
        self.setModal(False)
        self.resize(400, 400)
        self._pixmap = pixmap
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 缩放后的图像标签
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_scaled_pixmap()
        layout.addWidget(self._img_label, stretch=1)

        info = QLabel("双击关闭 | ESC 关闭")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #969696; font-size: 11px;")
        layout.addWidget(info)

    def _update_scaled_pixmap(self):
        """根据窗口大小缩放图像（放大显示）"""
        if self._pixmap and not self._pixmap.isNull():
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
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def mouseDoubleClickEvent(self, event):
        """双击关闭"""
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class ClickableFaceLabel(QLabel):
    """可双击放大的人脸缩略图标签"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(68, 68)
        self.setMaximumSize(100, 100)
        self.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3c3c3c; "
            "border-radius: 4px; color: #969696; font-size: 10px;"
        )
        self.setText("人脸未锁定")

    def set_face_frame(self, frame):
        """设置人脸帧（OpenCV BGR）"""
        if frame is None or frame.size == 0:
            return
        qimage = mat_to_qimage(frame)
        if qimage is not None and not qimage.isNull():
            pixmap = QPixmap.fromImage(qimage)
            self._current_pixmap = pixmap
            # 缩放到显示大小
            scaled = pixmap.scaled(
                68, 68,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._current_pixmap:
            if not self._current_pixmap.isNull():
                preview = FullscreenFacePreview(self._current_pixmap, self.window())
                preview.show()
        super().mouseDoubleClickEvent(event)


class StatusPanel(QFrame):
    """右侧状态面板 — 跌倒/步态/锁定/模型信息/呼吸检测/AI分析（实时更新）"""

    # 锁定/解锁信号
    lock_requested = pyqtSignal()
    unlock_requested = pyqtSignal()
    # 呼吸检测开始信号
    breath_start_requested = pyqtSignal()
    # AI 分析触发信号 (report_type: "simple"/"expert")
    llm_analysis_requested = pyqtSignal(str)
    # 系统控制信号（CLI 快捷键 → GUI 按钮）
    pause_toggle_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    clear_gait_requested = pyqtSignal()
    screenshot_requested = pyqtSignal()
    help_requested = pyqtSignal()
    # 视觉跌倒检测开关
    visual_fall_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "background-color: #252526; border-left: 1px solid #3c3c3c;"
        )
        self.setMinimumWidth(220)
        self.setMaximumWidth(300)

        self._is_locked = False  # 人脸是否已锁定

        self._setup_ui()

    def _setup_ui(self):
        """构建界面"""
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        # 用 QScrollArea 包裹，防止内容被遮挡
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: #252526; border: none; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # === 跌倒状态 ===
        self._fall_group = QGroupBox("跌倒状态")
        fall_layout = QVBoxLayout(self._fall_group)
        fall_layout.setSpacing(4)

        self._fall_status_label = QLabel("● 安全")
        self._fall_status_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #4ec9b0;"
        )
        self._fall_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fall_layout.addWidget(self._fall_status_label)

        self._fall_score_label = QLabel("分数: 0.0")
        self._fall_score_label.setStyleSheet("color: #969696; font-size: 11px;")
        self._fall_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fall_layout.addWidget(self._fall_score_label)

        self._fall_source_label = QLabel("来源: 摄像头")
        self._fall_source_label.setStyleSheet("color: #969696; font-size: 11px;")
        self._fall_source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fall_layout.addWidget(self._fall_source_label)

        self._visual_fall_cb = QCheckBox("视觉跌倒检测")
        self._visual_fall_cb.setToolTip("启用摄像头视觉跌倒检测\n（头部速度 + 姿态规则）")
        self._visual_fall_cb.setChecked(True)
        self._visual_fall_cb.setStyleSheet(
            "QCheckBox { color: #cccccc; font-size: 11px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; }"
        )
        self._visual_fall_cb.toggled.connect(lambda checked: self.visual_fall_toggled.emit(checked))
        fall_layout.addWidget(self._visual_fall_cb)

        layout.addWidget(self._fall_group)

        # === 人脸锁定（带按钮） ===
        self._lock_group = QGroupBox("人脸锁定")
        lock_layout = QVBoxLayout(self._lock_group)
        lock_layout.setSpacing(4)

        self._lock_status_label = QLabel("未锁定")
        self._lock_status_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #969696;"
        )
        self._lock_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lock_layout.addWidget(self._lock_status_label)

        self._lock_target_label = QLabel("目标: --")
        self._lock_target_label.setStyleSheet("color: #969696; font-size: 11px;")
        lock_layout.addWidget(self._lock_target_label)

        # 锁定/解锁按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self._lock_btn = QPushButton("🔒 锁定")
        self._lock_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border: none; "
            "padding: 4px 8px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #1177bb; }"
        )
        self._lock_btn.clicked.connect(lambda: self.lock_requested.emit())
        btn_layout.addWidget(self._lock_btn)

        self._unlock_btn = QPushButton("🔓 解锁")
        self._unlock_btn.setStyleSheet(
            "QPushButton { background-color: #6b3030; color: white; border: none; "
            "padding: 4px 8px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #8b4040; }"
        )
        self._unlock_btn.clicked.connect(lambda: self.unlock_requested.emit())
        btn_layout.addWidget(self._unlock_btn)

        lock_layout.addLayout(btn_layout)

        layout.addWidget(self._lock_group)

        # === 人脸缩略图 ===
        self._face_group = QGroupBox("人脸预览")
        face_layout = QVBoxLayout(self._face_group)
        face_layout.setSpacing(4)

        self._face_label = ClickableFaceLabel()
        face_layout.addWidget(self._face_label, alignment=Qt.AlignmentFlag.AlignCenter)

        face_hint = QLabel("双击放大")
        face_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        face_hint.setStyleSheet("color: #969696; font-size: 10px;")
        face_layout.addWidget(face_hint)

        layout.addWidget(self._face_group)

        # === 呼吸检测 ===
        self._breath_group = QGroupBox("呼吸检测")
        breath_layout = QVBoxLayout(self._breath_group)
        breath_layout.setSpacing(4)

        self._breath_bpm_label = QLabel("-- BPM")
        self._breath_bpm_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #969696;"
        )
        self._breath_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        breath_layout.addWidget(self._breath_bpm_label)

        self._breath_status_label = QLabel("状态: 就绪")
        self._breath_status_label.setStyleSheet("color: #969696; font-size: 11px;")
        self._breath_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        breath_layout.addWidget(self._breath_status_label)

        self._breath_progress_label = QLabel("")
        self._breath_progress_label.setStyleSheet("color: #969696; font-size: 10px;")
        self._breath_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        breath_layout.addWidget(self._breath_progress_label)

        # 开始检测按钮
        self._breath_btn = QPushButton("▶ 开始检测 (30s)")
        self._breath_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border: none; "
            "padding: 5px 10px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #1177bb; }"
            "QPushButton:disabled { background-color: #3c3c3c; color: #666; }"
        )
        self._breath_btn.clicked.connect(lambda: self.breath_start_requested.emit())
        breath_layout.addWidget(self._breath_btn)

        layout.addWidget(self._breath_group)

        # === 步态指标（实时） ===
        self._gait_group = QGroupBox("步态指标")
        gait_layout = QVBoxLayout(self._gait_group)
        gait_layout.setSpacing(3)

        self._gait_labels = {}
        gait_items = [
            ("step_count", "步数: 0"),
            ("cadence_spm", "步频: -- spm"),
            ("stride_length_m", "步长: -- m"),
            ("gait_velocity_mps", "步速: -- m/s"),
            ("avg_step_width_m", "步宽: -- m"),
            ("stance_percentage", "支撑期: -- %"),
            ("left_knee_rom", "左膝ROM: -- °"),
            ("right_knee_rom", "右膝ROM: -- °"),
            ("foot_clearance_cm", "足廓清: -- cm"),
            ("gait_rehab_score", "康复评分: --"),
            ("symmetry", "对称性: --"),
            ("trunk_sway_deg", "躯干摆动: -- °"),
        ]
        for key, text in gait_items:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #cccccc; font-size: 12px;")
            self._gait_labels[key] = lbl
            gait_layout.addWidget(lbl)

        layout.addWidget(self._gait_group)

        # === 外部传感器状态 ===
        self._sensor_group = QGroupBox("外部传感器")
        sensor_layout = QVBoxLayout(self._sensor_group)
        sensor_layout.setSpacing(4)

        self._csi_wristband_label = QLabel("📡 CSI（手环）: 未启用")
        self._csi_wristband_label.setStyleSheet("color: #969696; font-size: 12px;")
        sensor_layout.addWidget(self._csi_wristband_label)

        self._phone_label = QLabel("📱 手机 IMU: 未启动")
        self._phone_label.setStyleSheet("color: #969696; font-size: 12px;")
        sensor_layout.addWidget(self._phone_label)

        layout.addWidget(self._sensor_group)

        # === 模型信息 ===
        self._model_group = QGroupBox("模型信息")
        model_layout = QVBoxLayout(self._model_group)
        model_layout.setSpacing(4)

        self._model_name_label = QLabel("模型: --")
        self._model_name_label.setStyleSheet("color: #cccccc;")
        model_layout.addWidget(self._model_name_label)

        self._device_label = QLabel("设备: --")
        self._device_label.setStyleSheet("color: #cccccc;")
        model_layout.addWidget(self._device_label)

        self._precision_label = QLabel("精度: int8")
        self._precision_label.setStyleSheet("color: #cccccc;")
        model_layout.addWidget(self._precision_label)

        layout.addWidget(self._model_group)

        # === 系统控制（CLI 快捷键 → GUI 按钮） ===
        self._control_group = QGroupBox("⚙ 系统控制")
        control_layout = QVBoxLayout(self._control_group)
        control_layout.setSpacing(4)

        # 暂停/恢复按钮
        self._pause_btn = QPushButton("⏸ 暂停")
        self._pause_btn.setToolTip("暂停/恢复画面处理 (Space)")
        self._pause_btn.setStyleSheet(
            "QPushButton { background-color: #5a5a5a; color: white; border: none; "
            "padding: 5px 10px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #6e6e6e; }"
        )
        self._pause_btn.clicked.connect(lambda: self.pause_toggle_requested.emit())
        control_layout.addWidget(self._pause_btn)

        # 按钮行（重置 + 截图）
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(4)

        self._reset_btn = QPushButton("🔄 重置")
        self._reset_btn.setToolTip("重置 Kalman/步态/人脸锁定 (Ctrl+Shift+R)")
        self._reset_btn.setStyleSheet(
            "QPushButton { background-color: #6b3030; color: white; border: none; "
            "padding: 5px 8px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #8b4040; }"
        )
        self._reset_btn.clicked.connect(lambda: self.reset_requested.emit())
        btn_row1.addWidget(self._reset_btn)

        self._screenshot_btn = QPushButton("📸 截图")
        self._screenshot_btn.setToolTip("保存当前画面为 JPEG (S)")
        self._screenshot_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border: none; "
            "padding: 5px 8px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #1177bb; }"
        )
        self._screenshot_btn.clicked.connect(lambda: self.screenshot_requested.emit())
        btn_row1.addWidget(self._screenshot_btn)
        control_layout.addLayout(btn_row1)

        # 按钮行（清除步态 + 帮助）
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(4)

        self._clear_gait_btn = QPushButton("🗑 清除步态")
        self._clear_gait_btn.setToolTip("清除数据库中步态记录 (C)")
        self._clear_gait_btn.setStyleSheet(
            "QPushButton { background-color: #5a5a3a; color: white; border: none; "
            "padding: 5px 8px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #7a7a4a; }"
        )
        self._clear_gait_btn.clicked.connect(lambda: self.clear_gait_requested.emit())
        btn_row2.addWidget(self._clear_gait_btn)

        self._help_btn = QPushButton("❓ 帮助")
        self._help_btn.setToolTip("显示功能说明和快捷键列表 (H)")
        self._help_btn.setStyleSheet(
            "QPushButton { background-color: #3a5a3a; color: white; border: none; "
            "padding: 5px 8px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #4a7a4a; }"
        )
        self._help_btn.clicked.connect(lambda: self.help_requested.emit())
        btn_row2.addWidget(self._help_btn)
        control_layout.addLayout(btn_row2)

        layout.addWidget(self._control_group)

        # === AI 分析（本地 LLM） ===
        self._llm_group = QGroupBox("🤖 AI 分析 (本地)")
        llm_layout = QVBoxLayout(self._llm_group)
        llm_layout.setSpacing(6)

        # 状态标签
        self._llm_status_label = QLabel("状态: 就绪")
        self._llm_status_label.setStyleSheet("color: #969696; font-size: 11px;")
        self._llm_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        llm_layout.addWidget(self._llm_status_label)

        # 分析按钮
        self._llm_simple_btn = QPushButton("📋 简单分析")
        self._llm_simple_btn.setToolTip("生成本地 LLM 步态分析报告（约 30-60 秒）")
        self._llm_simple_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border: none; "
            "padding: 6px 10px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #1177bb; }"
            "QPushButton:disabled { background-color: #3c3c3c; color: #666; }"
        )
        self._llm_simple_btn.clicked.connect(lambda: self.llm_analysis_requested.emit("simple"))
        llm_layout.addWidget(self._llm_simple_btn)

        # 模型信息
        self._llm_model_label = QLabel("模型: Qwen2.5-3B")
        self._llm_model_label.setStyleSheet("color: #8a8a8a; font-size: 10px;")
        self._llm_model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        llm_layout.addWidget(self._llm_model_label)

        layout.addWidget(self._llm_group)

        # === 扫码连接（小程序） ===
        self._qr_group = QGroupBox("📱 扫码连接小程序")
        qr_layout = QVBoxLayout(self._qr_group)
        qr_layout.setSpacing(4)

        self._qr_label = QLabel()
        self._qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr_label.setMinimumSize(180, 180)
        self._qr_label.setMaximumSize(200, 200)
        self._qr_label.setStyleSheet("background-color: white; border: 2px solid #0e639c; border-radius: 6px;")
        qr_layout.addWidget(self._qr_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._qr_addr = QLabel("")
        self._qr_addr.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        self._qr_addr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr_addr.setWordWrap(True)
        qr_layout.addWidget(self._qr_addr)

        self._generate_qr_code()
        self._update_qr_addr_label()

        layout.addWidget(self._qr_group)

        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    def update_fall_status(self, status, score=None, source="摄像头"):
        """更新跌倒状态"""
        if status == "alert":
            self._fall_status_label.setText("⚠ 跌倒!")
            self._fall_status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #f44747;"
            )
        elif status == "lying":
            self._fall_status_label.setText("⏸ 躺下")
            self._fall_status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #dcdcaa;"
            )
        elif status == "warning":
            self._fall_status_label.setText("⚠ 警告")
            self._fall_status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #ce9178;"
            )
        else:
            self._fall_status_label.setText("● 安全")
            self._fall_status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #4ec9b0;"
            )

        if score is not None:
            self._fall_score_label.setText(f"分数: {score:.2f}")
        self._fall_source_label.setText(f"来源: {source}")

    def update_lock_status(self, locked, name="", similarity=0.0, searching=False):
        """更新人脸锁定状态

        Args:
            locked: 是否已锁定
            name: 目标名称
            similarity: 匹配置信度
            searching: 是否处于搜索模式
        """
        self._is_locked = locked

        if locked:
            self._lock_status_label.setText("已锁定")
            self._lock_status_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #4ec9b0;"
            )
            self._lock_target_label.setText(f"目标: {name} (相似度: {similarity:.2f})")
            self._lock_btn.setEnabled(False)
            self._unlock_btn.setEnabled(True)
        elif searching:
            self._lock_status_label.setText("搜索中...")
            self._lock_status_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #ce9178;"
            )
            self._lock_target_label.setText("正在寻找目标...")
            self._lock_btn.setEnabled(False)
            self._unlock_btn.setEnabled(False)
        else:
            self._lock_status_label.setText("未锁定")
            self._lock_status_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #969696;"
            )
            self._lock_target_label.setText("目标: --")
            self._lock_btn.setEnabled(True)
            self._unlock_btn.setEnabled(False)

    def update_face_thumbnail(self, face_frame):
        """更新人脸缩略图

        Args:
            face_frame: np.ndarray 干净的人脸区域（无骨架叠加）
        """
        if face_frame is not None and face_frame.size > 0:
            self._face_label.set_face_frame(face_frame)
        else:
            self._face_label.clear()
            if self._is_locked:
                self._face_label.setText("等待人脸...")
            else:
                self._face_label.setText("人脸未锁定")

    def update_breathing(self, bpm, status, remaining=0, total=30, has_person=True):
        """更新呼吸检测数据

        Args:
            bpm: 呼吸频率 (次/分钟)，0 表示未计算出
            status: 状态字符串 ("空闲" / "检测中" / "完成")
            remaining: 剩余秒数
            total: 总检测时长
            has_person: 是否检测到人体（即使已锁定也可能人物离开画面）
        """
        # 未锁定时禁用呼吸检测按钮
        if not self._is_locked:
            self._breath_bpm_label.setText("-- BPM")
            self._breath_bpm_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #969696;"
            )
            self._breath_status_label.setText("状态: 人脸未锁定")
            self._breath_status_label.setStyleSheet("color: #969696; font-size: 11px;")
            self._breath_progress_label.setText("")
            self._breath_btn.setText("请先锁定人脸")
            self._breath_btn.setEnabled(False)
            return

        # 已锁定但未检测到人体
        if not has_person:
            self._breath_bpm_label.setText("-- BPM")
            self._breath_bpm_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #969696;"
            )
            self._breath_status_label.setText("状态: 未检测到人体")
            self._breath_status_label.setStyleSheet("color: #ce9178; font-size: 11px;")
            self._breath_progress_label.setText("")
            self._breath_btn.setText("未检测到人体")
            self._breath_btn.setEnabled(False)
            return

        if status == "空闲":
            self._breath_bpm_label.setText("-- BPM")
            self._breath_bpm_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #969696;"
            )
            self._breath_status_label.setText("状态: 就绪")
            self._breath_status_label.setStyleSheet("color: #969696; font-size: 11px;")
            self._breath_progress_label.setText("")
            self._breath_btn.setText("▶ 开始检测 (30s)")
            self._breath_btn.setEnabled(True)
        elif status == "检测中":
            elapsed = total - remaining
            self._breath_bpm_label.setText("-- BPM")
            self._breath_bpm_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #ce9178;"
            )
            self._breath_status_label.setText(f"状态: 检测中 {elapsed}s/{total}s")
            self._breath_status_label.setStyleSheet("color: #ce9178; font-size: 11px;")
            # 进度条（文字模拟）
            bar_w = 12
            filled = int(bar_w * elapsed / max(total, 1))
            bar = "█" * filled + "░" * (bar_w - filled)
            self._breath_progress_label.setText(f"[{bar}]")
            self._breath_progress_label.setStyleSheet("color: #ce9178; font-size: 10px;")
            self._breath_btn.setText("⏳ 检测中...")
            self._breath_btn.setEnabled(False)
        else:  # "完成"
            if bpm > 0:
                self._breath_bpm_label.setText(f"{bpm:.1f} BPM")
                if 8 <= bpm <= 30:
                    self._breath_bpm_label.setStyleSheet(
                        "font-size: 18px; font-weight: bold; color: #4ec9b0;"
                    )
                else:
                    self._breath_bpm_label.setStyleSheet(
                        "font-size: 18px; font-weight: bold; color: #dcdcaa;"
                    )
            else:
                self._breath_bpm_label.setText("-- BPM")
                self._breath_bpm_label.setStyleSheet(
                    "font-size: 18px; font-weight: bold; color: #969696;"
                )
            self._breath_status_label.setText("状态: 完成")
            self._breath_status_label.setStyleSheet("color: #4ec9b0; font-size: 11px;")
            self._breath_progress_label.setText("")
            self._breath_btn.setText("▶ 重新检测 (30s)")
            self._breath_btn.setEnabled(True)

    def update_gait_metrics(self, metrics):
        """更新步态指标

        Args:
            metrics: dict from GaitAnalyzer.get_metrics()
        """
        # 未锁定时显示占位
        if not self._is_locked:
            for label_key, lbl in self._gait_labels.items():
                # 保留标签前缀，数值替换为 --
                base = lbl.text().split(":")[0] if ":" in lbl.text() else lbl.text()
                lbl.setText(f"{base}: --")
            return

        label_map = {
            "step_count": ("step_count", "步数: {}"),
            "cadence_spm": ("cadence_spm", "步频: {} spm"),
            "stride_length_m": ("stride_length_m", "步长: {:.2f} m"),
            "gait_velocity_mps": ("gait_velocity_mps", "步速: {:.2f} m/s"),
            "avg_step_width_m": ("avg_step_width_m", "步宽: {:.3f} m"),
            "stance_percentage": ("stance_percentage", "支撑期: {:.0f} %"),
            "left_knee_rom": ("left_knee_rom", "左膝ROM: {:.0f} °"),
            "right_knee_rom": ("right_knee_rom", "右膝ROM: {:.0f} °"),
            "foot_clearance_cm": ("foot_clearance_cm", "足廓清: {:.1f} cm"),
            "gait_rehab_score": ("gait_rehab_score", "康复评分: {:.0f}"),
            "symmetry": ("symmetry", "对称性: {:.3f}"),
            "trunk_sway_deg": ("trunk_sway_deg", "躯干摆动: {:.1f} °"),
        }

        for key, (label_key, fmt) in label_map.items():
            val = metrics.get(key)
            if val is not None and label_key in self._gait_labels:
                try:
                    if isinstance(val, (int,)):
                        self._gait_labels[label_key].setText(fmt.format(val))
                    else:
                        self._gait_labels[label_key].setText(fmt.format(val))
                except (ValueError, TypeError):
                    pass

    def clear_gait_metrics(self):
        """Clear gait metric labels after resetting gait data."""
        for label_key, lbl in self._gait_labels.items():
            base = lbl.text().split(":")[0] if ":" in lbl.text() else lbl.text()
            lbl.setText(f"{base}: --")

    def update_model_info(self, model, device, precision=None):
        """更新模型信息"""
        self._model_name_label.setText(f"模型: {model}-pose")
        self._device_label.setText(f"设备: {device.upper()}")
        if precision:
            self._precision_label.setText(f"精度: {precision}")

    def update_llm_model_info(self, model_key):
        """更新 AI 分析模型名称"""
        from rehab_gui.config import LLM_MODEL_OPTIONS
        label = LLM_MODEL_OPTIONS.get(model_key, model_key)
        self._llm_model_label.setText(f"模型: {label}")

    # ── 二维码（小程序扫码连接）──

    def _generate_qr_code(self):
        """生成服务器地址二维码"""
        try:
            import qrcode
            from io import BytesIO
            from PyQt6.QtGui import QPixmap
            import json
            from rehab_monitor.discovery_service import get_lan_ip

            ip = get_lan_ip()
            data = json.dumps({"ip": ip, "port": 5000}, ensure_ascii=False)

            qr = qrcode.QRCode(box_size=4, border=2)
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")

            buf = BytesIO()
            img = img.convert("RGB")  # Qt 不兼容 1-bit 模式，转 RGB
            img.save(buf, format="PNG")
            buf.seek(0)

            pixmap = QPixmap()
            ok = pixmap.loadFromData(buf.read())
            if not ok or pixmap.isNull():
                raise RuntimeError("QPixmap 加载失败")
            scaled = pixmap.scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            self._qr_label.setPixmap(scaled)
        except Exception as e:
            logger.warning("生成二维码失败: %s", e)
            self._qr_addr.setText(f"⚠ 二维码生成失败: {e}")

    def _update_qr_addr_label(self):
        """更新二维码地址提示"""
        from rehab_monitor.discovery_service import get_lan_ip
        ip = get_lan_ip()
        self._qr_addr.setText(f"服务器: {ip}:5000")

    def update_sensor_status(self, sensor_status):
        """更新外部传感器状态

        Args:
            sensor_status: dict from SensorManager.get_status()
        """
        if not sensor_status:
            return

        # CSI（手环）— 合并显示
        wb = sensor_status.get("wristband", {})
        csi = sensor_status.get("csi", {})
        wb_active = wb.get("active", False)
        csi_active = csi.get("active", False)

        if wb_active or csi_active:
            # 任一已启用
            wb_running = wb.get("running", False)
            csi_connected = csi.get("connected", False)
            wb_fall = wb.get("fall", False)
            csi_fall = csi.get("fall", False)

            if wb_fall or csi_fall:
                self._csi_wristband_label.setText("📡 CSI（手环）: ⚠ 跌倒!")
                self._csi_wristband_label.setStyleSheet(
                    "color: #f44747; font-size: 12px; font-weight: bold;")
            elif wb_running or csi_connected:
                status_parts = []
                if wb_running:
                    status_parts.append("手环在线")
                if csi_connected:
                    status_parts.append("CSI在线")
                self._csi_wristband_label.setText(
                    f"📡 CSI（手环）: {' + '.join(status_parts)}")
                self._csi_wristband_label.setStyleSheet(
                    "color: #4ec9b0; font-size: 12px;")
            else:
                self._csi_wristband_label.setText("📡 CSI（手环）: 等待连接")
                self._csi_wristband_label.setStyleSheet(
                    "color: #ce9178; font-size: 12px;")
        else:
            self._csi_wristband_label.setText("📡 CSI（手环）: 未启用")
            self._csi_wristband_label.setStyleSheet("color: #969696; font-size: 12px;")

        # 手机
        ph = sensor_status.get("phone", {})
        if ph.get("active"):
            if ph.get("connected"):
                if ph.get("fall"):
                    self._phone_label.setText("📱 手机 IMU: ⚠ 跌倒!")
                    self._phone_label.setStyleSheet(
                        "color: #f44747; font-size: 12px; font-weight: bold;")
                else:
                    self._phone_label.setText("📱 手机 IMU: 已连接")
                    self._phone_label.setStyleSheet("color: #4ec9b0; font-size: 12px;")
            else:
                self._phone_label.setText("📱 手机 IMU: 等待连接")
                self._phone_label.setStyleSheet("color: #ce9178; font-size: 12px;")
        else:
            self._phone_label.setText("📱 手机 IMU: 未启动")
            self._phone_label.setStyleSheet("color: #969696; font-size: 12px;")

    def update_llm_status(self, status, message=""):
        """更新 AI 分析按钮状态

        Args:
            status: "idle" / "running" / "done" / "error"
            message: 状态描述文本
        """
        if status == "running":
            self._llm_status_label.setText(f"状态: {message or '分析中...'}")
            self._llm_status_label.setStyleSheet(
                "color: #ce9178; font-size: 11px; font-weight: bold;")
            self._llm_simple_btn.setText("⏳ 分析中...")
            self._llm_simple_btn.setEnabled(False)
        elif status == "done":
            self._llm_status_label.setText("状态: ✅ 分析完成")
            self._llm_status_label.setStyleSheet("color: #4ec9b0; font-size: 11px;")
            self._llm_simple_btn.setText("📋 简单分析")
            self._llm_simple_btn.setEnabled(True)
        elif status == "error":
            self._llm_status_label.setText(f"状态: ❌ {message or '分析失败'}")
            self._llm_status_label.setStyleSheet(
                "color: #f44747; font-size: 11px; font-weight: bold;")
            self._llm_simple_btn.setText("📋 简单分析")
            self._llm_simple_btn.setEnabled(True)
        else:  # idle
            self._llm_status_label.setText("状态: 就绪")
            self._llm_status_label.setStyleSheet("color: #969696; font-size: 11px;")
            self._llm_simple_btn.setText("📋 简单分析")
            self._llm_simple_btn.setEnabled(True)

    def update_pause_button(self, paused):
        """更新暂停按钮文字

        Args:
            paused: True 表示当前已暂停，显示「恢复」；False 显示「暂停」
        """
        if paused:
            self._pause_btn.setText("▶ 恢复")
        else:
            self._pause_btn.setText("⏸ 暂停")
