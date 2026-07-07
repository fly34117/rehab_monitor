"""设置对话框 — 运行时调整配置"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QGroupBox,
    QMessageBox,
    QTabWidget,
)

from rehab_gui.config import (
    MODEL_OPTIONS,
    DEVICE_OPTIONS,
)
from rehab_gui.core.config_manager import ConfigManager
from rehab_gui.core.device_detector import DeviceDetector

from rehab_monitor.logging_setup import get_logger

logger = get_logger("settings_dialog")


class SettingsDialog(QDialog):
    """运行时设置对话框"""

    def __init__(self, config_manager=None, parent=None):
        """初始化设置对话框

        Args:
            config_manager: ConfigManager 实例
            parent: 父窗口
        """
        super().__init__(parent)
        self._config_manager = config_manager or ConfigManager()
        self._detector = DeviceDetector()

        self.setWindowTitle("⚙️ 设置")
        self.setFixedSize(480, 520)
        self.setModal(True)

        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        """构建界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 标签页
        tabs = QTabWidget()

        # 常规设置
        general_tab = self._create_general_tab()
        tabs.addTab(general_tab, "常规")

        # 功能设置
        feature_tab = self._create_feature_tab()
        tabs.addTab(feature_tab, "功能")

        layout.addWidget(tabs)

        # 按钮
        button_layout = QHBoxLayout()

        self._reset_btn = QPushButton("重置为默认")
        self._reset_btn.clicked.connect(self._on_reset)
        button_layout.addWidget(self._reset_btn)

        button_layout.addStretch()

        self._save_btn = QPushButton("保存")
        self._save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(self._save_btn)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)

        layout.addLayout(button_layout)

    def _create_general_tab(self):
        """创建常规设置页

        Returns:
            QWidget: 常规设置页
        """
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setSpacing(10)

        # 模型选择
        self._model_combo = QComboBox()
        for model in MODEL_OPTIONS:
            self._model_combo.addItem(model)
        layout.addRow("模型:", self._model_combo)

        # 设备选择
        self._device_combo = QComboBox()
        device_labels = {"auto": "自动检测", "cpu": "CPU", "gpu": "GPU", "npu": "NPU"}
        for device in DEVICE_OPTIONS:
            self._device_combo.addItem(device_labels[device], device)
        layout.addRow("设备:", self._device_combo)

        layout.addRow(QLabel(""))

        # 设备信息
        info_group = QGroupBox("设备信息")
        info_layout = QFormLayout(info_group)
        info = self._detector.get_info()
        info_layout.addRow("NPU:", QLabel("可用" if info["npu_available"] else "不可用"))
        info_layout.addRow("GPU:", QLabel("可用" if info["gpu_available"] else "不可用"))
        info_layout.addRow("OpenVINO:", QLabel(info.get("openvino_version", "unknown")))
        layout.addRow(info_group)

        return widget

    def _create_feature_tab(self):
        """创建功能设置页

        Returns:
            QWidget: 功能设置页
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # CSI（手环） — 合并的 ESP32 跌倒检测
        self._csi_wristband_check = QCheckBox("CSI（手环）跌倒检测 (ESP32-S3 / WiFi)")
        self._csi_wristband_check.setToolTip(
            "连接 ESP32-S3 手环或 CSI WiFi 传感器进行跌倒检测"
        )
        layout.addWidget(self._csi_wristband_check)

        # 呼吸检测
        self._breathing_check = QCheckBox("呼吸检测（胸腔ROI + FFT频谱分析）")
        self._breathing_check.setToolTip(
            "基于姿态关键点提取胸腔运动信号，实时计算呼吸频率 (BPM)"
        )
        layout.addWidget(self._breathing_check)

        layout.addWidget(QLabel(
            "<span style='color:#666; font-size:11px;'>"
            "提示：手机 App 通过 UDP 5002 自动发现，无需手动启用"
            "</span>"
        ))

        layout.addStretch()

        return widget

    def _load_config(self):
        """加载当前配置"""
        config = self._config_manager.load()

        self._model_combo.setCurrentText(config.get("model", "yolo26n"))
        device = config.get("device", "auto")
        idx = self._device_combo.findData(device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

        # CSI（手环）— 合并选项
        csi_or_wb = config.get("csi_wristband") or config.get("wristband") or config.get("csi")
        self._csi_wristband_check.setChecked(bool(csi_or_wb))
        self._breathing_check.setChecked(config.get("breathing", True))

    def _on_reset(self):
        """重置为默认配置"""
        reply = QMessageBox.question(
            self,
            "确认",
            "确定要重置所有设置为默认值吗?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._config_manager.reset_to_defaults()
            self._load_config()
            logger.info("设置已重置为默认值")

    def _on_save(self):
        """保存配置"""
        config = {
            "model": self._model_combo.currentText(),
            "device": self._device_combo.currentData(),
            "csi_wristband": self._csi_wristband_check.isChecked(),
            "breathing": self._breathing_check.isChecked(),
            # 兼容旧配置（映射到新配置）
            "wristband": self._csi_wristband_check.isChecked(),
            "csi": self._csi_wristband_check.isChecked(),
        }
        self._config_manager.save(config)
        logger.info("设置已保存")
        QMessageBox.information(self, "成功", "配置已保存")
        self.accept()
