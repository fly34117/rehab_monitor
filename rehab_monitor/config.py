"""所有可调参数集中管理"""
import os

# ===== 路径 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ULTRALYTICS_DIR = os.path.join(PROJECT_ROOT, "ultralytics-8.4.46")
MODEL_DIR = os.path.join(ULTRALYTICS_DIR, "model")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "yolo26n-pose_int8_openvino_model")
DB_PATH = os.path.join(PROJECT_ROOT, "rehab_data.db")

# ===== 摄像头 =====
CAMERA_ID = 0                # 首选摄像头，失败时自动回退到 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS_TARGET = 25       # 目标帧率 (用于卡尔曼 dt 估算)

# ===== 推理 =====
POSE_DEVICE = "cpu"           # "cpu" / "gpu" / "npu"
USE_KALMAN = True

# ===== 卡尔曼 =====
KALMAN_DT = 0.04
KALMAN_PROCESS_NOISE = 3e-4     # 过程噪声 (信任匀速模型)
KALMAN_MEASUREMENT_NOISE = 3e-3 # 测量噪声 (不信任原始检测，增强平滑)
KALMAN_MAX_PERSONS = 5
KALMAN_MAX_MATCH_DIST = 200    # 髋部中心最大匹配距离 (像素)
KALMAN_MAX_UNMATCHED = 30      # 连续未匹配帧数后释放槽位

# ===== 步态分析 =====
# COCO 17 关键点索引
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

# ===== 跌倒检测 =====
FALL_ASPECT_RATIO_THRESHOLD = 0.9   # 高/宽 < 0.9
FALL_HIP_Y_RATIO = 0.7              # 髋部 y > h * 0.7
FALL_TILT_DEG = 45.0                # 肩线倾角 > 45°
FALL_CONFIRM_FRAMES = 5             # 连续确认帧数

# ===== 存储 =====
SQLITE_WRITE_INTERVAL = 1.0   # 秒

# ===== DeepSeek API =====
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT = 15.0
REPORT_INTERVAL = 30.0   # 秒

# ===== 空间定位 (地平面映射) =====
SPATIAL_CAMERA_HEIGHT_M = 1.5       # 摄像头距地面高度 (米)
SPATIAL_CAMERA_TILT_DEG = 0.0       # 摄像头俯角 (0=水平平视, 90=垂直下视)
SPATIAL_ORIGIN_X_M = 0.0            # 世界原点 X (米), 默认摄像头正下方
SPATIAL_ORIGIN_Y_M = 0.0            # 世界原点 Y (米)
# 校准点: 4 组 (像素坐标 → 世界坐标 米), 用于计算单应矩阵
# 格式: [(px1, py1, wx1, wy1), (px2, py2, wx2, wy2), ...]
# 留空则使用简易针孔模型 (仅需高度+俯角)
SPATIAL_CALIB_POINTS = [
    # 示例: 地面4个角, 用卷尺实测世界坐标
    # (120, 420,  0.0, 1.0),   # 左下角: 像素(120,420) → 世界(0m, 1m)
    # (520, 420,  2.0, 1.0),   # 右下角
    # (120, 200,  0.0, 5.0),   # 左上角(远处)
    # (520, 200,  2.0, 5.0),   # 右上角(远处)
]

# ===== 表情识别 =====
EMOTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "emotion_model")
EMOTION_INTERVAL = 5     # 每 N 帧跑一次表情识别
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_INPUT_SIZE = 64  # 分类器输入尺寸 (64x64)
