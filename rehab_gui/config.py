"""GUI 配置常量 — 模型/设备/功能选项"""
import os

# ===== 模型选项 =====
MODEL_OPTIONS = [
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo11n",
    "yolo11s",
]
DEFAULT_MODEL = "yolo26n"

# ===== 设备选项 =====
DEVICE_OPTIONS = ["auto", "cpu", "gpu", "npu"]
DEFAULT_DEVICE = "gpu"

# ===== 模型精度映射 =====
MODEL_PRECISION = {
    "yolo11n": {"cpu": "int8", "gpu": "fp16", "npu": "fp16"},
    "yolo11s": {"cpu": "int8", "gpu": "fp16", "npu": "fp16"},
    # yolo26 系列 NPU 兼容已修复（end2end=False 重导出）
    "yolo26n": {"cpu": "int8", "gpu": "fp16", "npu": "fp16"},
    "yolo26s": {"cpu": "int8", "gpu": "fp16", "npu": "fp16"},
    "yolo26m": {"cpu": "fp32", "gpu": "fp32"},
}

# ===== 路径 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "ultralytics-8.4.46", "model")
CONFIG_DIR = os.path.expanduser("~/.rehab_gui")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def resolve_pose_model_path(model_name, device, precision_override=None):
    """Return the OpenVINO pose model directory and selected precision label."""
    precision = precision_override or MODEL_PRECISION.get(model_name, {}).get(device, "int8")
    if precision == "fp32":
        model_dir = f"{model_name}-pose_openvino_model"
    else:
        model_dir = f"{model_name}-pose_{precision}_openvino_model"
    return os.path.join(MODEL_DIR, model_dir), precision

# ===== 界面 =====
WINDOW_TITLE = "Rehab Monitor — 康复监控仪表盘"
WINDOW_MIN_WIDTH = 1280
WINDOW_MIN_HEIGHT = 820
CAMERA_DEFAULT_WIDTH = 640
CAMERA_DEFAULT_HEIGHT = 480
TARGET_FPS = int(os.environ.get("REHAB_TARGET_FPS", "30"))
DISPLAY_FPS = int(os.environ.get("REHAB_DISPLAY_FPS", "60"))

# ===== 功能开关默认值 =====
DEFAULT_PHONE = False
DEFAULT_CSI_WRISTBAND = False
DEFAULT_BREATHING = True  # 呼吸检测默认开启
DEFAULT_VISUAL_FALL = True  # 视觉跌倒检测默认开启

# ===== LLM 模型选项 =====
LLM_MODEL_OPTIONS = {
    "qwen2.5-3b": "Qwen2.5-3B",
    "qwen3-4b": "Qwen3-4B (实验性)",
    "qwen2.5-7b": "Qwen2.5-7B (推荐)",
}
LLM_MODEL_PATHS = {
    "qwen2.5-3b": os.path.join(PROJECT_ROOT, "llm_local", "models", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "qwen3-4b": os.path.join(PROJECT_ROOT, "llm_local", "models", "Qwen3-4B-Q4_K_M.gguf"),
    "qwen2.5-7b": os.path.join(PROJECT_ROOT, "llm_local", "models", "Qwen2.5-7B-Instruct-IQ3_M.gguf"),
}
DEFAULT_LLM_MODEL = "qwen2.5-7b"

LLAMA_SERVER = os.path.join(PROJECT_ROOT, "llm_local", "llama.cpp", "build", "bin", "llama-server")
LLAMA_LIB_DIR = os.path.join(PROJECT_ROOT, "llm_local", "llama.cpp", "build", "bin")
LLM_CPUSET = os.environ.get("REHAB_LLM_CPUSET", "4-13")
LLM_THREADS = int(os.environ.get("REHAB_LLM_THREADS", "4"))
LLM_NICE = int(os.environ.get("REHAB_LLM_NICE", "5"))
LLM_PORT = int(os.environ.get("REHAB_LLM_PORT", "8089"))

# ===== 环境 =====
CONDA_PREFIX = "/home/ubuntu224/miniconda3/envs/yolov26"
PYTHON_BIN = os.path.join(CONDA_PREFIX, "bin", "python")
LD_LIBRARY_PATH = os.path.join(CONDA_PREFIX, "lib", "python3.11", "site-packages", "openvino", "libs")

# ===== CPU 线程 (Arrow Lake-U 优化) =====
ENV_VARS = {
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OV_CPU_INFERENCE_NUM_THREADS": "8",
    "OV_CPU_NUM_STREAMS": "2",
    "OV_CPU_ENABLE_CPU_PINNING": "NO",
    "OV_CPU_HYPER_THREADING": "YES",
}
