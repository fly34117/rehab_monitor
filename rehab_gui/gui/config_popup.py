"""启动配置弹窗 — 首次启动快速配置"""
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QGroupBox,
    QMessageBox,
)

from rehab_gui.config import (
    MODEL_OPTIONS,
    DEFAULT_MODEL,
    DEVICE_OPTIONS,
    DEFAULT_CSI_WRISTBAND,
    DEFAULT_BREATHING,
    DEFAULT_VISUAL_FALL,
    LLM_MODEL_OPTIONS,
    DEFAULT_LLM_MODEL,
)
from rehab_gui.core.device_detector import DeviceDetector
from rehab_gui.core.config_manager import ConfigManager

from rehab_monitor.logging_setup import get_logger

logger = get_logger("config_popup")


class ConfigPopup(QDialog):
    """快速配置弹窗 — 单对话框, 非向导式"""

    def __init__(self, config_manager=None, parent=None):
        """初始化配置弹窗

        Args:
            config_manager: ConfigManager 实例
            parent: 父窗口
        """
        super().__init__(parent)
        self._config_manager = config_manager or ConfigManager()
        self._detector = DeviceDetector()
        self._result = None

        self.setWindowTitle("⚙️ 快速配置")
        self.setFixedSize(400, 560)
        self.setModal(True)

        self._setup_ui()
        self._load_saved_config()
        self._update_device_label()

    def _setup_ui(self):
        """构建界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 标题
        title = QLabel("快速配置")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #007acc;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QFormLayout(model_group)

        self._model_combo = QComboBox()
        for model in MODEL_OPTIONS:
            self._model_combo.addItem(model)
        self._model_combo.setCurrentText(DEFAULT_MODEL)
        model_layout.addRow("模型:", self._model_combo)

        layout.addWidget(model_group)

        # 设备选择
        device_group = QGroupBox("推理设备")
        device_layout = QFormLayout(device_group)

        self._device_combo = QComboBox()
        for device in DEVICE_OPTIONS:
            label = {"auto": "自动检测", "cpu": "CPU", "gpu": "GPU", "npu": "NPU"}[device]
            self._device_combo.addItem(label, device)
        self._device_combo.setCurrentIndex(0)  # 默认自动检测
        device_layout.addRow("设备:", self._device_combo)

        self._device_label = QLabel("检测设备中...")
        self._device_label.setStyleSheet("color: #969696; font-size: 11px;")
        device_layout.addRow("", self._device_label)

        layout.addWidget(device_group)

        # LLM 模型选择
        llm_group = QGroupBox("LLM 模型（本地大语言模型）")
        llm_layout = QFormLayout(llm_group)

        self._llm_model_combo = QComboBox()
        for key, label in LLM_MODEL_OPTIONS.items():
            self._llm_model_combo.addItem(label, key)
        self._llm_model_combo.setCurrentIndex(
            list(LLM_MODEL_OPTIONS.keys()).index(DEFAULT_LLM_MODEL))
        llm_layout.addRow("模型:", self._llm_model_combo)

        llm_hint = QLabel("4B 推荐 | 7B 需内存充裕")
        llm_hint.setStyleSheet("color: #969696; font-size: 11px;")
        llm_layout.addRow("", llm_hint)

        layout.addWidget(llm_group)

        # 功能开关
        feature_group = QGroupBox("功能选项")
        feature_layout = QVBoxLayout(feature_group)

        self._csi_wristband_check = QCheckBox("CSI（手环）跌倒检测 (ESP32-S3 / WiFi)")
        self._csi_wristband_check.setToolTip(
            "连接 ESP32-S3 手环或 CSI WiFi 传感器"
        )
        self._csi_wristband_check.setChecked(DEFAULT_CSI_WRISTBAND)
        feature_layout.addWidget(self._csi_wristband_check)

        self._breathing_check = QCheckBox("呼吸检测（胸腔ROI + FFT频谱分析）")
        self._breathing_check.setToolTip(
            "基于姿态关键点提取胸腔运动信号，实时计算呼吸频率 (BPM)"
        )
        self._breathing_check.setChecked(DEFAULT_BREATHING)
        feature_layout.addWidget(self._breathing_check)

        self._visual_fall_check = QCheckBox("视觉跌倒检测（头部速度 + 姿态规则）")
        self._visual_fall_check.setToolTip(
            "摄像头检测跌倒：头部垂直速度 > 1.5m/s + 身体姿态异常。\n"
            "关闭后仅使用外部传感器（手环/手机/CSI）检测跌倒。"
        )
        self._visual_fall_check.setChecked(DEFAULT_VISUAL_FALL)
        feature_layout.addWidget(self._visual_fall_check)

        layout.addWidget(QLabel(
            "<span style='color:#666; font-size:11px;'>"
            "提示：手机 App 通过 UDP 5002 自动发现，无需手动启用"
            "</span>"
        ))

        layout.addWidget(feature_group)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._start_btn = QPushButton("保存并启动")
        self._start_btn.setMinimumWidth(120)
        self._start_btn.clicked.connect(self._on_start)
        button_layout.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setMinimumWidth(80)
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)

        layout.addLayout(button_layout)

        # 设备检测
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)

    def _load_saved_config(self):
        """加载已保存的配置"""
        if self._config_manager.has_config():
            config = self._config_manager.load()
            self._model_combo.setCurrentText(config.get("model", DEFAULT_MODEL))
            device = config.get("device", "auto")
            idx = self._device_combo.findData(device)
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)
            # CSI（手环）— 合并选项
            csi_or_wb = config.get("csi_wristband") or config.get("wristband") or config.get("csi")
            self._csi_wristband_check.setChecked(bool(csi_or_wb))
            # 呼吸检测
            self._breathing_check.setChecked(config.get("breathing", DEFAULT_BREATHING))
            # 视觉跌倒检测
            self._visual_fall_check.setChecked(config.get("visual_fall", DEFAULT_VISUAL_FALL))
            # LLM 模型
            llm_model = config.get("llm_model", DEFAULT_LLM_MODEL)
            idx = self._llm_model_combo.findData(llm_model)
            if idx >= 0:
                self._llm_model_combo.setCurrentIndex(idx)

    def _update_device_label(self):
        """更新设备检测状态标签"""
        info = self._detector.get_info()
        parts = []
        if info["npu_available"]:
            parts.append("NPU✓")
        if info["gpu_available"]:
            parts.append("GPU✓")
        parts.append("CPU✓")
        detected = f"可用: {' / '.join(parts)}"
        self._device_label.setText(detected)

    def _on_device_changed(self):
        """设备选项变化时的回调"""
        self._update_device_label()

    def _on_start(self):
        """点击保存并启动"""
        device = self._device_combo.currentData()
        if device == "npu":
            # 验证 NPU 权限
            if not self._detector._has_npu():
                QMessageBox.warning(
                    self,
                    "NPU 不可用",
                    "NPU 设备不可用或权限不足。\n"
                    "请运行: sudo usermod -aG render $USER && newgrp render\n"
                    "已自动切换到 CPU。",
                )
                idx = self._device_combo.findData("cpu")
                self._device_combo.setCurrentIndex(idx)
                return

        self._result = {
            "model": self._model_combo.currentText(),
            "device": device,
            "llm_model": self._llm_model_combo.currentData(),
            "visual_fall": self._visual_fall_check.isChecked(),
            "csi_wristband": self._csi_wristband_check.isChecked(),
            "breathing": self._breathing_check.isChecked(),
            # 兼容旧配置
            "wristband": self._csi_wristband_check.isChecked(),
            "wristband_port": 8081,
            "wristband_http": 8080,
            "csi": self._csi_wristband_check.isChecked(),
            "csi_host": "192.168.2.10",
            "csi_port": 8000,
            "csi_protocol": "LLTF",
            "csi_show_diag": False,
        }

        self._config_manager.save(self._result)
        self.accept()

    def get_result(self):
        """获取配置结果

        Returns:
            dict: 配置字典, 未确认时返回 None
        """
        return self._result
