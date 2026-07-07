"""主窗口 — QMainWindow 整合所有面板"""
import time
from collections import deque

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMenuBar,
    QStatusBar,
    QLabel,
    QMessageBox,
)

from rehab_gui.config import (
    DISPLAY_FPS,
    WINDOW_TITLE,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
)
from rehab_gui.gui.camera_widget import CameraWidget
from rehab_gui.gui.status_panel import StatusPanel
from rehab_gui.gui.diagnostic_tabs import DiagnosticTabs
from rehab_gui.gui.config_popup import ConfigPopup
from rehab_gui.gui.settings_dialog import SettingsDialog
from rehab_gui.core.config_manager import ConfigManager
from rehab_gui.core.thread_bridge import ThreadBridge

from rehab_monitor.logging_setup import get_logger

logger = get_logger("main_window")


class MainWindow(QMainWindow):
    """主窗口 — 整合摄像头/状态面板/诊断标签页"""

    # 信号: 启动后端管线
    backend_start = pyqtSignal(dict)
    # 信号: 停止后端管线
    backend_stop = pyqtSignal()
    # 信号: 录入人脸锁定
    lock_target = pyqtSignal()
    # 信号: 解除锁定
    unlock_target = pyqtSignal()
    # 信号: 开始呼吸检测
    breath_start_requested = pyqtSignal()
    # 信号: AI 分析请求 (report_type: "simple"/"expert")
    llm_analysis_requested = pyqtSignal(str, dict)
    # 信号: 对话消息 (message_text: str)
    llm_chat_requested = pyqtSignal(str)
    # 信号: 系统控制（CLI 快捷键 → GUI 按钮）
    pause_toggle_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    clear_gait_requested = pyqtSignal()
    screenshot_requested = pyqtSignal()
    help_requested = pyqtSignal()
    fps_toggle_requested = pyqtSignal()
    # 信号: 视觉跌倒检测开关
    visual_fall_toggled = pyqtSignal(bool)

    def __init__(self, config=None, parent=None):
        """初始化主窗口

        Args:
            config: 配置字典
            parent: 父窗口
        """
        super().__init__(parent)
        self._config = config or {}
        self._config_manager = ConfigManager()
        self._bridge = None
        self._frame_poll_timer = None
        self._backend_running = False

        # FPS 统计
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0.0
        self._frame_count = 0

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_shortcuts()
        self._apply_config()

        logger.info("主窗口已初始化")

    def _setup_ui(self):
        """构建界面"""
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # 主分割器 (左右)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧: 摄像头 + 诊断标签
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 摄像头画面
        self._camera_widget = CameraWidget()
        self._camera_widget.setSizePolicy(self._camera_widget.sizePolicy().Policy.Expanding,
                                          self._camera_widget.sizePolicy().Policy.Expanding)
        left_layout.addWidget(self._camera_widget, stretch=3)

        # 诊断标签页
        self._diagnostic_tabs = DiagnosticTabs()
        self._diagnostic_tabs.setMinimumHeight(220)
        left_layout.addWidget(self._diagnostic_tabs, stretch=2)

        splitter.addWidget(left_widget)

        # 右侧: 状态面板
        self._status_panel = StatusPanel()
        splitter.addWidget(self._status_panel)

        # 设置分割比例 (左侧占 75%, 右侧占 25%)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _setup_menu(self):
        """构建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        start_action = QAction("启动后端", self)
        start_action.setShortcut("Ctrl+S")
        start_action.triggered.connect(self._on_start_backend)
        file_menu.addAction(start_action)

        stop_action = QAction("停止后端", self)
        stop_action.setShortcut("Ctrl+T")
        stop_action.triggered.connect(self._on_stop_backend)
        file_menu.addAction(stop_action)

        file_menu.addSeparator()

        screenshot_action = QAction("截图", self)
        screenshot_action.setShortcut("S")
        screenshot_action.triggered.connect(lambda: self.screenshot_requested.emit())
        file_menu.addAction(screenshot_action)

        file_menu.addSeparator()

        quit_action = QAction("退出", self)
        quit_action.setShortcut("Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # 查看菜单
        view_menu = menubar.addMenu("查看")

        toggle_diag_action = QAction("切换诊断标签页", self)
        toggle_diag_action.setShortcut("D")
        view_menu.addAction(toggle_diag_action)

        # 配置菜单
        config_menu = menubar.addMenu("配置")

        settings_action = QAction("设置...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._on_settings)
        config_menu.addAction(settings_action)

        config_menu.addSeparator()

        config_popup_action = QAction("快速配置...", self)
        config_popup_action.setShortcut("Ctrl+P")
        config_popup_action.triggered.connect(self._on_config_popup)
        config_menu.addAction(config_popup_action)

        # 工具菜单
        tools_menu = menubar.addMenu("工具")

        lock_action = QAction("录入人脸锁定", self)
        lock_action.setShortcut("T")
        lock_action.triggered.connect(self._on_lock_target)
        tools_menu.addAction(lock_action)

        unlock_action = QAction("解除锁定", self)
        unlock_action.setShortcut("R")
        unlock_action.triggered.connect(self._on_unlock_target)
        tools_menu.addAction(unlock_action)

        tools_menu.addSeparator()

        tools_menu.addSeparator()

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self):
        """构建状态栏"""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._fps_label = QLabel("FPS: --")
        self._status_bar.addPermanentWidget(self._fps_label)

        self._persons_label = QLabel("人数: 0")
        self._status_bar.addPermanentWidget(self._persons_label)

        self._frame_label = QLabel("帧: 0")
        self._status_bar.addPermanentWidget(self._frame_label)

        self._model_label = QLabel("模型: --")
        self._status_bar.addPermanentWidget(self._model_label)

        self._device_label = QLabel("设备: --")
        self._status_bar.addPermanentWidget(self._device_label)

    def _setup_shortcuts(self):
        """绑定 CLI 兼容快捷键（与菜单快捷键并存）"""
        # Space → 暂停/恢复
        sc_pause = QShortcut(QKeySequence("Space"), self)
        sc_pause.activated.connect(lambda: self.pause_toggle_requested.emit())

        # Ctrl+Shift+R → 重置（R 已被解锁占用，用组合键避免冲突）
        sc_reset = QShortcut(QKeySequence("Ctrl+Shift+R"), self)
        sc_reset.activated.connect(lambda: self.reset_requested.emit())

        # C → 清除步态数据
        sc_clear = QShortcut(QKeySequence("C"), self)
        sc_clear.activated.connect(lambda: self.clear_gait_requested.emit())

        # H → 帮助
        sc_help = QShortcut(QKeySequence("H"), self)
        sc_help.activated.connect(lambda: self.help_requested.emit())

        # F → 切换 FPS 显示
        sc_fps = QShortcut(QKeySequence("F"), self)
        sc_fps.activated.connect(lambda: self.fps_toggle_requested.emit())

    def _apply_config(self):
        """应用配置到界面"""
        model = self._config.get("model", "yolo26n")
        device = self._config.get("device", "auto")

        self._model_label.setText(f"模型: {model}")
        self._device_label.setText(f"设备: {device}")

    def set_bridge(self, bridge):
        """设置数据桥接器

        Args:
            bridge: ThreadBridge 实例
        """
        self._bridge = bridge
        self._connect_signals()
        self._start_frame_polling()

    def _start_frame_polling(self):
        """按显示帧率拉取最新帧，丢弃旧帧以防 GUI 事件队列积压。"""
        if self._bridge is None or not hasattr(self._bridge, "take_latest_frame"):
            return
        if self._frame_poll_timer is not None:
            self._frame_poll_timer.stop()
        self._frame_poll_timer = QTimer(self)
        self._frame_poll_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._frame_poll_timer.timeout.connect(self._poll_latest_frame)
        interval_ms = max(1, int(1000 / max(1, DISPLAY_FPS)))
        self._frame_poll_timer.start(interval_ms)

    def _poll_latest_frame(self):
        if self._bridge is None:
            return
        frame, frame_num = self._bridge.take_latest_frame()
        if frame is None:
            return
        self._camera_widget.update_frame(frame)
        self._on_frame_count(frame_num)

    def _connect_signals(self):
        """连接数据桥接信号"""
        if self._bridge is None:
            return

        # 主画面由 _poll_latest_frame 定时拉取，避免每帧 signal 堆积。
        # 跌倒状态
        self._bridge.fall_signal.connect(self._on_fall_update)
        # 情绪数据
        self._bridge.emotion_signal.connect(self._on_emotion_update)
        # 诊断画面（直接连接）
        self._bridge.diag_angle_signal.connect(self._diagnostic_tabs.update_angle)
        self._bridge.diag_trajectory_signal.connect(
            self._diagnostic_tabs.update_trajectory
        )
        self._bridge.diag_skeleton_signal.connect(self._diagnostic_tabs.update_skeleton)
        self._bridge.diag_gait_signal.connect(self._diagnostic_tabs.update_gait)
        # 呼吸波形
        if hasattr(self._bridge, "diag_breath_signal"):
            self._bridge.diag_breath_signal.connect(self._diagnostic_tabs.update_breath)
        # 步态指标（小数据，直接连接即可）
        self._bridge.metrics_signal.connect(self._on_metrics_update)
        # 帧计数
        self._bridge.frame_count_signal.connect(self._on_frame_count)
        # 人脸缩略图
        if hasattr(self._bridge, "face_thumbnail_signal"):
            self._bridge.face_thumbnail_signal.connect(self._on_face_thumbnail)
        # 呼吸数据
        if hasattr(self._bridge, "breathing_signal"):
            self._bridge.breathing_signal.connect(self._on_breathing_update)

        # 状态面板的锁定/解锁按钮
        if hasattr(self._status_panel, "lock_requested"):
            self._status_panel.lock_requested.connect(self._on_lock_target)
        if hasattr(self._status_panel, "unlock_requested"):
            self._status_panel.unlock_requested.connect(self._on_unlock_target)
        # 呼吸检测开始按钮
        if hasattr(self._status_panel, "breath_start_requested"):
            self._status_panel.breath_start_requested.connect(self._on_breath_start)
        # AI 分析按钮
        if hasattr(self._status_panel, "llm_analysis_requested"):
            self._status_panel.llm_analysis_requested.connect(self._on_llm_analysis)

        # 系统控制按钮（CLI 快捷键 → GUI 按钮）
        if hasattr(self._status_panel, "pause_toggle_requested"):
            self._status_panel.pause_toggle_requested.connect(
                lambda: self.pause_toggle_requested.emit())
        if hasattr(self._status_panel, "reset_requested"):
            self._status_panel.reset_requested.connect(
                lambda: self.reset_requested.emit())
        if hasattr(self._status_panel, "clear_gait_requested"):
            self._status_panel.clear_gait_requested.connect(
                lambda: self.clear_gait_requested.emit())
        if hasattr(self._status_panel, "screenshot_requested"):
            self._status_panel.screenshot_requested.connect(
                lambda: self.screenshot_requested.emit())
        if hasattr(self._status_panel, "help_requested"):
            self._status_panel.help_requested.connect(self._on_help)
        # 视觉跌倒检测开关
        if hasattr(self._status_panel, "visual_fall_toggled"):
            self._status_panel.visual_fall_toggled.connect(
                lambda checked: self.visual_fall_toggled.emit(checked))

        # 对话输入（诊断标签页 → AI 回答 Tab）
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            self._diagnostic_tabs._llm_widget.message_sent.connect(self._on_chat_message)

    def _on_frame_count(self, frame_num):
        """帧计数回调

        Args:
            frame_num: 当前帧号
        """
        self._frame_count += 1
        # FPS 计算
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._fps_label.setText(f"FPS: {self._current_fps:.1f}")
            self._frame_count = 0
            self._fps_time = now

        self._frame_label.setText(f"帧: {frame_num}")

    def _on_fall_update(self, data):
        """跌倒状态回调

        Args:
            data: dict {status, score, source}
        """
        status = data.get("status", "safe")
        score = data.get("score")
        source = data.get("source", "摄像头")
        self._status_panel.update_fall_status(status, score, source)

    def _on_emotion_update(self, data):
        """情绪数据回调

        Args:
            data: dict {label, scores}
        """
        label = data.get("label", "neutral")
        scores = data.get("scores")
        if hasattr(self._status_panel, "update_emotion"):
            self._status_panel.update_emotion(label, scores)

    def _on_metrics_update(self, data):
        """步态指标回调 — 更新状态面板和状态栏

        Args:
            data: dict 步态指标数据
        """
        person_count = data.get("person_count", 0)
        self._persons_label.setText(f"人数: {person_count}")

        # 存储最新指标供 LLM 分析使用（仅当有实际步态数据时更新）
        gm = data.get("gait_metrics")
        if gm is not None:
            self._latest_metrics = gm

        # 更新状态面板的步态指标
        gait_metrics = data.get("gait_metrics")
        if gait_metrics and hasattr(self._status_panel, "update_gait_metrics"):
            self._status_panel.update_gait_metrics(gait_metrics)

        # 更新锁定状态（仅当数据中包含 lock_status 时才更新，避免 person_count 单条推送误清）
        if "lock_status" in data:
            lock_status = data.get("lock_status", False)
            lock_name = data.get("lock_name", "")
            lock_similarity = data.get("lock_similarity", 0.0)
            lock_searching = data.get("lock_searching", False)
            if hasattr(self._status_panel, "update_lock_status"):
                self._status_panel.update_lock_status(lock_status, lock_name, lock_similarity, lock_searching)

        # 更新跌倒状态到面板
        fall_status = data.get("fall_status")
        fall_score = data.get("fall_score")
        if fall_status is not None and hasattr(self._status_panel, "update_fall_status"):
            self._status_panel.update_fall_status(fall_status, fall_score)

        # 更新传感器状态到面板
        sensor_status = data.get("sensor_status")
        if sensor_status and hasattr(self._status_panel, "update_sensor_status"):
            self._status_panel.update_sensor_status(sensor_status)

    def clear_gait_metrics(self):
        """Clear cached gait metrics used by the side panel and LLM report."""
        self._latest_metrics = {}
        if hasattr(self._status_panel, "clear_gait_metrics"):
            self._status_panel.clear_gait_metrics()

    def _on_face_thumbnail(self, face_frame):
        """人脸缩略图回调

        Args:
            face_frame: np.ndarray 干净的人脸区域
        """
        if hasattr(self._status_panel, "update_face_thumbnail"):
            self._status_panel.update_face_thumbnail(face_frame)

    def _on_breathing_update(self, data):
        """呼吸数据回调

        Args:
            data: dict {bpm, status, remaining, total, has_person}
        """
        if hasattr(self._status_panel, "update_breathing"):
            bpm = data.get("bpm") or 0
            status = data.get("status", "空闲")
            remaining = data.get("remaining", 0)
            total = data.get("total", 30)
            has_person = data.get("has_person", True)
            self._status_panel.update_breathing(bpm, status, remaining, total, has_person)

    def _on_breath_start(self):
        """呼吸检测开始按钮回调 — 转发信号到后端"""
        self._status_bar.showMessage("开始呼吸检测 (30s)...", 3000)
        # 通过现有的 lock_target 信号链路不行，需要新增信号
        # 这里直接通过 bridge 通知，或新增信号
        self.breath_start_requested.emit()

    def _on_llm_analysis(self, report_type="simple"):
        """AI 分析按钮回调 — 转发 LLM 请求到后端

        Args:
            report_type: "simple" 或 "expert"
        """
        # 收集当前步态指标
        gait_metrics = {}
        if hasattr(self, '_latest_metrics'):
            gait_metrics = self._latest_metrics

        data = {
            "report_type": report_type,
            "gait_metrics": gait_metrics,
        }
        self.llm_analysis_requested.emit(report_type, data)
        self._status_bar.showMessage("正在启动本地 LLM 分析...", 3000)

    def update_llm_status(self, status, message=""):
        """更新 AI 分析状态（状态面板 + 诊断标签页）

        Args:
            status: "idle" / "running" / "done" / "error"
            message: 状态文本
        """
        if hasattr(self._status_panel, "update_llm_status"):
            self._status_panel.update_llm_status(status, message)
        if hasattr(self._diagnostic_tabs, "update_llm_status"):
            self._diagnostic_tabs.update_llm_status(status, message)

    def set_llm_response(self, text):
        """设置 LLM 回答文本并切换到 AI 回答 Tab

        Args:
            text: LLM 生成的报告文本
        """
        if hasattr(self._diagnostic_tabs, "update_llm_response"):
            self._diagnostic_tabs.update_llm_response(text)
        if hasattr(self._diagnostic_tabs, "switch_to_llm_tab"):
            self._diagnostic_tabs.switch_to_llm_tab()

    def update_llm_partial(self, text):
        """流式更新 LLM 文本（分析报告）"""
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            self._diagnostic_tabs._llm_widget.update_partial_text(text)

    def finish_llm_text(self, text):
        """完成流式 LLM 报告"""
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            self._diagnostic_tabs._llm_widget.finish_text(text)

    # ---- 对话功能 ----

    def _on_chat_message(self, text):
        """用户输入对话消息 — 转发到后端"""
        self._status_bar.showMessage(f"发送消息: {text[:30]}...", 2000)
        # 在显示区添加用户消息
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            self._diagnostic_tabs._llm_widget.append_user_message(text)
            self._diagnostic_tabs._llm_widget.show_running("思考中...")
        # 切换到 AI 回答 Tab
        if hasattr(self._diagnostic_tabs, "switch_to_llm_tab"):
            self._diagnostic_tabs.switch_to_llm_tab()
        # 发射信号给 main.py 处理
        self.llm_chat_requested.emit(text)

    def update_chat_stream(self, text):
        """流式更新对话回复"""
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            self._diagnostic_tabs._llm_widget.update_stream_text(text)

    def finish_chat_stream(self, text):
        """完成流式对话回复"""
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            self._diagnostic_tabs._llm_widget.finish_stream(text)
            self._diagnostic_tabs._llm_widget.show_done()

    def get_chat_history(self):
        """获取对话历史"""
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            return self._diagnostic_tabs._llm_widget.get_chat_history()
        return []

    def append_chat_history(self, role, content):
        """追加到对话历史"""
        if hasattr(self._diagnostic_tabs, "_llm_widget"):
            self._diagnostic_tabs._llm_widget.add_to_history(role, content)

    def _on_start_backend(self):
        """启动后端管线"""
        if self._backend_running:
            return
        self.backend_start.emit(self._config)
        self._backend_running = True
        self._status_bar.showMessage("后端管线已启动", 3000)

    def _on_stop_backend(self):
        """停止后端管线"""
        if not self._backend_running:
            return
        self.backend_stop.emit()
        self._backend_running = False
        self._status_bar.showMessage("后端管线已停止", 3000)

    def _on_lock_target(self):
        """录入人脸锁定"""
        self.lock_target.emit()
        self._status_bar.showMessage("正在录入人脸...", 3000)

    def _on_unlock_target(self):
        """解除人脸锁定"""
        self.unlock_target.emit()
        self._status_bar.showMessage("已解除锁定", 3000)

    def update_lock_status(self, locked, name="", similarity=0.0):
        """更新锁定状态显示

        Args:
            locked: 是否已锁定
            name: 目标名称
            similarity: 匹配置信度
        """
        if locked:
            self._status_bar.showMessage(f"已锁定: {name} (置信度: {similarity:.2f})", 5000)
        else:
            # 检查是否处于搜索模式
            # 搜索模式由 main.py 的 process_frame 管理
            self._status_bar.showMessage("正在搜索目标...", 3000)

    def _on_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self._config_manager, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            # 重新加载配置
            self._config = self._config_manager.load()
            self._apply_config()

    def _on_config_popup(self):
        """打开快速配置弹窗"""
        dialog = ConfigPopup(self._config_manager, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            result = dialog.get_result()
            if result:
                self._config = result
                self._apply_config()

    def _on_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            "Rehab Monitor GUI\n\n"
            "三模态智能康复监控系统\n"
            "YOLOv26-Pose + OpenVINO\n\n"
            "功能: 步态分析 · 跌倒检测 · 情绪识别 · 呼吸检测",
        )

    def _on_help(self):
        """显示功能帮助（快捷键速查）"""
        help_text = (
            "⌨ <b>快捷键速查</b><br><br>"
            "<table>"
            "<tr><td><b>Space</b></td><td>暂停 / 恢复</td></tr>"
            "<tr><td><b>S</b></td><td>保存截图</td></tr>"
            "<tr><td><b>F</b></td><td>切换 FPS 显示</td></tr>"
            "<tr><td><b>H</b></td><td>显示此帮助</td></tr>"
            "<tr><td><b>T</b></td><td>录入人脸锁定</td></tr>"
            "<tr><td><b>R</b></td><td>解除人脸锁定</td></tr>"
            "<tr><td><b>Ctrl+Shift+R</b></td><td>重置 Kalman/步态/人脸</td></tr>"
            "<tr><td><b>C</b></td><td>清除步态数据库</td></tr>"
            "<tr><td><b>Q</b></td><td>退出程序</td></tr>"
            "<tr><td><b>D</b></td><td>切换诊断标签页</td></tr>"
            "<tr><td><b>Ctrl+S</b></td><td>启动后端</td></tr>"
            "<tr><td><b>Ctrl+T</b></td><td>停止后端</td></tr>"
            "<tr><td><b>Ctrl+,</b></td><td>设置</td></tr>"
            "</table>"
            "<br>"
            "🖱 <b>鼠标操作</b><br>"
            "• 双击人脸缩略图 — 全屏预览<br>"
            "• 双击诊断画面 — 全屏预览<br>"
            "• 侧边栏按钮 — 锁定/解锁/呼吸/分析<br>"
            "<br>"
            "📋 <b>功能按钮</b> (右侧面板)<br>"
            "• ⏸ 暂停 — 暂停/恢复画面处理<br>"
            "• 🔄 重置 — 重置所有跟踪状态<br>"
            "• 📸 截图 — 保存当前画面<br>"
            "• 🗑 清除步态 — 清除数据库步态记录<br>"
            "• 📋 简单分析 — 生成本地 LLM 康复报告<br>"
        )
        QMessageBox.information(self, "功能帮助", help_text)

    def closeEvent(self, event):
        """窗口关闭事件

        Args:
            event: 关闭事件
        """
        if self._backend_running:
            reply = QMessageBox.question(
                self,
                "确认",
                "后端管线仍在运行, 确定要退出吗?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.backend_stop.emit()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
