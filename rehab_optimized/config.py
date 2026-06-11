"""异构计算配置 — 统一管理所有可调参数
support NPU / GPU / CPU three-device auto-selection and performance tuning

三阶段设备映射 (Stage → Device):
  --gpu: Decode=iGPU  Inference=iGPU  Post=CPU
  --npu: Decode=iGPU  Inference=NPU   Post=CPU
  --cpu: Decode=CPU   Inference=CPU   Post=CPU
"""
import os

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "ultralytics-8.4.46", "model")
DB_PATH = os.path.join(PROJECT_ROOT, "rehab_data.db")

# ============================================================
# 三阶段管线设备分配
# ============================================================
# key: 主推理设备, value: {Stage: 目标硬件}
# GPU 资源共享: VA-API (media engine) 和 OpenVINO GPU (compute engine)
# 使用 iGPU 不同子引擎, 可同时运行互不阻塞
STAGE_DEVICE_MAP = {
    "gpu": {
        "decode":    "iGPU",   # GStreamer VA-API / V4L2 MJPEG
        "inference": "iGPU",   # OpenVINO GPU (compute engine)
        "post":      "CPU",    # NumPy/OpenCV/Kalman
    },
    "npu": {
        "decode":    "iGPU",   # GStreamer VA-API
        "inference": "NPU",    # Intel AI Boost
        "post":      "CPU",
    },
    "cpu": {
        "decode":    "CPU",    # V4L2 MJPEG/YUYV
        "inference": "CPU",    # OpenVINO CPU
        "post":      "CPU",
    },
}

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "ultralytics-8.4.46", "model")
DB_PATH = os.path.join(PROJECT_ROOT, "rehab_data.db")

# ============================================================
# 设备选择策略
# ============================================================
# 优先级: NPU > GPU > CPU, 自动回退
DEVICE_PRIORITY = ["npu", "gpu", "cpu"]

# NPU 设备名 (OpenVINO)
NPU_DEVICE = "NPU"
# GPU 设备名 (OpenVINO)
GPU_DEVICE = "GPU"
# CPU 设备名 (OpenVINO)
CPU_DEVICE = "CPU"

# ============================================================
# 模型配置 — 按设备自动选精度
# ============================================================
# 设备 → 精度 映射
DEVICE_PRECISION = {
    "npu": "fp16",   # NPU 仅支持 FP16/INT8
    "gpu": "fp16",   # GPU FP16 最优
    "cpu": "int8",   # CPU INT8 最优
}

# 可用模型及其精度变体 (目录后缀)
AVAILABLE_MODELS = {
    "yolo11n": ["fp32", "fp16", "int8"],
    "yolo11s": ["fp32", "int8"],        # 没有 NPU FP16 导出
    "yolo26n": ["fp32", "fp16", "int8"],
}

# 默认模型
DEFAULT_MODEL = "yolo11n"

# 模型目录名模板: {model_name}-pose_{precision}_openvino_model
# 例: yolo11n-pose_fp16_openvino_model

# ============================================================
# 摄像头
# ============================================================
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS_TARGET = 30

# 帧队列 (解码→推理解耦)
FRAME_QUEUE_MAXSIZE = 2       # 只保留最新帧
RESULT_QUEUE_MAXSIZE = 1

# ============================================================
# GStreamer 硬件解码 (VA-API, 驱动需 >= 24.4.0)
# ============================================================
USE_GSTREAMER = False          # 默认 False, run.sh --hw-decode 时覆盖
# 设为 True 或通过环境变量 REHAB_USE_GSTREAMER=1 启用
# 需要 intel-media-va-driver >= 24.4.0 → sudo bash ubuntu/setup_hw_decode.sh
GST_PIPELINE_TEMPLATE = (
    "v4l2src device=/dev/video{device} ! "
    "video/x-raw,width={width},height={height},framerate={fps}/1 ! "
    "vaapipostproc ! video/x-raw,format=BGR ! "
    "appsink name=sink max-buffers=1 drop=true emit-signals=true"
)

# ============================================================
# 推理优化
# ============================================================
OV_NUM_STREAMS = 2            # CPU streams
OV_INFERENCE_THREADS = 8      # CPU 线程数 (混合架构最优)
OV_KALMAN_ENABLED = True

# ============================================================
# 原有步态/跌倒/表情/API参数 (从 rehab_monitor/config 继承)
# ============================================================
FALL_ASPECT_RATIO_THRESHOLD = 0.9
FALL_HIP_Y_RATIO = 0.7
FALL_TILT_DEG = 45.0
FALL_CONFIRM_FRAMES = 5
SQLITE_WRITE_INTERVAL = 1.0
EMOTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "emotion_model")
EMOTION_INTERVAL = 5
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT = 15.0
REPORT_INTERVAL = 30.0
API_HOST = "0.0.0.0"
API_PORT = 5000
WS_PORT = 5001

# COCO 关键点索引
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

# 卡尔曼
KALMAN_DT = 0.04
KALMAN_PROCESS_NOISE = 3e-4
KALMAN_MEASUREMENT_NOISE = 3e-3
KALMAN_MAX_PERSONS = 5
KALMAN_MAX_MATCH_DIST = 200
KALMAN_MAX_UNMATCHED = 30

# 空间定位
SPATIAL_CAMERA_HEIGHT_M = 1.5
SPATIAL_CAMERA_TILT_DEG = 0.0
SPATIAL_CALIB_POINTS = []

# 表情
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# 阈值
LOCK_SIMILARITY_THRESHOLD = 0.55
