"""配置管理 — JSON 预设保存/加载"""
import json
import os
import shutil
from datetime import datetime

from rehab_gui.config import CONFIG_DIR, CONFIG_FILE


class ConfigManager:
    """GUI 配置管理器 — JSON 格式存储"""

    # 默认配置
    _DEFAULTS = {
        "model": "yolo26n",
        "device": "auto",
        "llm_model": "qwen3-4b",
        "visual_fall": True,
        "wristband": False,
        "wristband_port": 8081,
        "wristband_http": 8080,
        "phone": False,
        "csi": False,
        "csi_host": "192.168.2.10",
        "csi_port": 8000,
        "csi_protocol": "LLTF",
        "csi_show_diag": False,
        "last_used_device": None,
        "last_launch": None,
    }

    def __init__(self, config_path=None):
        """初始化配置管理器

        Args:
            config_path: 配置文件路径, 默认使用全局 CONFIG_FILE
        """
        self._config_path = config_path or CONFIG_FILE
        self._config = dict(self._DEFAULTS)

    def load(self):
        """加载配置文件

        Returns:
            dict: 配置字典, 文件不存在时返回默认值
        """
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # 合并默认值 (兼容旧配置)
                for key, value in self._DEFAULTS.items():
                    if key not in loaded:
                        loaded[key] = value
                self._config = loaded
            except (json.JSONDecodeError, IOError):
                self._config = dict(self._DEFAULTS)
        return dict(self._config)

    def save(self, config=None):
        """保存配置到文件

        Args:
            config: 配置字典, 为 None 时保存当前配置
        """
        if config is not None:
            self._config = config
        self._config["last_launch"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def get(self, key, default=None):
        """获取配置项

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self._config.get(key, default)

    def set(self, key, value):
        """设置配置项

        Args:
            key: 配置键
            value: 配置值
        """
        self._config[key] = value

    def has_config(self):
        """检查配置文件是否存在

        Returns:
            bool: 配置文件是否存在
        """
        return os.path.exists(self._config_path)

    def reset_to_defaults(self):
        """重置为默认配置"""
        self._config = dict(self._DEFAULTS)
        self.save()

    def get_presets_dir(self):
        """获取预设目录路径

        Returns:
            str: 预设目录路径
        """
        presets_dir = os.path.join(CONFIG_DIR, "presets")
        os.makedirs(presets_dir, exist_ok=True)
        return presets_dir

    def save_preset(self, name, config=None):
        """保存配置预设

        Args:
            name: 预设名称
            config: 配置字典, 为 None 时保存当前配置
        """
        presets_dir = self.get_presets_dir()
        path = os.path.join(presets_dir, f"{name}.json")
        save_config = config or dict(self._config)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_config, f, indent=2, ensure_ascii=False)

    def load_preset(self, name):
        """加载配置预设

        Args:
            name: 预设名称

        Returns:
            dict: 预设配置字典, 不存在时返回 None
        """
        presets_dir = self.get_presets_dir()
        path = os.path.join(presets_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_presets(self):
        """列出所有预设

        Returns:
            list[str]: 预设名称列表
        """
        presets_dir = self.get_presets_dir()
        if not os.path.exists(presets_dir):
            return []
        return [
            f.replace(".json", "")
            for f in os.listdir(presets_dir)
            if f.endswith(".json")
        ]
