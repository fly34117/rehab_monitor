"""
统一模型加载器 — 按设备自动选择精度变体
============================================
策略:
  NPU: FP16 (YOLO11n/s-pose) — NPU 不支持 FP32
  GPU: FP16 (最优) → INT8 → FP32
  CPU: INT8 (最优) → FP16 → FP32

模型目录命名: {model_name}-pose_{precision}_openvino_model
"""
import os
import logging
from .config import MODEL_DIR, DEVICE_PRECISION, AVAILABLE_MODELS, DEFAULT_MODEL

logger = logging.getLogger("rehab.model")


class ModelLoader:
    """异构模型加载器 — 按设备自动选精度"""

    def __init__(self, model_name=None, device=None):
        """
        Args:
            model_name: 模型名 (yolo11n, yolo11s, yolo26n)
            device: 设备类型 (npu, gpu, cpu), None=自动检测
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.device = device or "cpu"
        self._model = None

    # ------------------------------------------------------------------
    # 模型路径解析
    # ------------------------------------------------------------------

    def _get_precision(self):
        """获取当前设备的最优精度"""
        return DEVICE_PRECISION.get(self.device, "fp32")

    def _build_model_dir(self, precision):
        """构建模型目录名"""
        return f"{self.model_name}-pose_{precision}_openvino_model"

    def _find_best_model(self):
        """找到当前设备可用的最优模型"""
        precision = self._get_precision()
        model_dir = self._build_model_dir(precision)
        full_path = os.path.join(MODEL_DIR, model_dir)

        if os.path.isdir(full_path):
            return full_path, precision

        # 回退: 按精度优先级查找
        fallback_order = {
            "npu": ["fp16", "int8", "fp32"],
            "gpu": ["fp16", "fp32", "int8"],
            "cpu": ["int8", "fp16", "fp32"],
        }
        for prec in fallback_order.get(self.device, ["fp32"]):
            model_dir = self._build_model_dir(prec)
            full_path = os.path.join(MODEL_DIR, model_dir)
            if os.path.isdir(full_path):
                logger.warning("精度 %s 模型不存在, 回退到 %s", precision, prec)
                return full_path, prec

        raise FileNotFoundError(
            f"找不到模型 {self.model_name}-pose 的任何精度变体, "
            f"请先导出: yolo export model={self.model_name}-pose.pt format=openvino"
        )

    # ------------------------------------------------------------------
    # OpenVINO 设备名映射
    # ------------------------------------------------------------------

    @staticmethod
    def ov_device_name(device):
        """转为 OpenVINO / ultralytics 兼容的设备名"""
        return {
            "cpu": "cpu",         # pose_detector 转为 intel:cpu
            "gpu": "intel:gpu",
            "npu": "intel:npu",
        }.get(device, "cpu")

    # ------------------------------------------------------------------
    # 模型加载 (使用 ultralytics YOLO + OpenVINO 后端)
    # ------------------------------------------------------------------

    def load(self):
        """加载模型 — 返回 (model, device_str, model_path)"""
        model_path, precision = self._find_best_model()
        device_str = self.ov_device_name(self.device)

        logger.info("加载模型: %s", model_path)
        logger.info("设备: %s → %s  (精度: %s)", self.device, device_str, precision)

        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._model_path = model_path
        self._precision = precision

        return self._model, device_str, model_path

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("请先调用 load() 加载模型")
        return self._model

    @property
    def model_path(self):
        return getattr(self, '_model_path', None)

    @property
    def precision(self):
        return getattr(self, '_precision', None)

    def report(self):
        """打印模型信息"""
        path, prec = self._find_best_model()
        return (
            f"模型配置:\n"
            f"  名称: {self.model_name}-pose\n"
            f"  设备: {self.device.upper()}\n"
            f"  精度: {prec}\n"
            f"  路径: {path}"
        )
