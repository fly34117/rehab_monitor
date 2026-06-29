"""CSI 跌倒检测配置 — 所有可调参数集中管理"""
import os

# ===== 项目路径 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== TCP 连接 =====
CSI_HOST = os.environ.get("CSI_HOST", "192.168.2.10")
CSI_PORT = int(os.environ.get("CSI_PORT", "8000"))
CSI_CONNECT_TIMEOUT = 5.0          # socket 连接超时 (秒)
CSI_RECV_TIMEOUT = 0.1             # socket 接收超时 (秒)
CSI_PACKET_SIZE = 420              # ESP32 单个 CSI 数据包字节数
CSI_CACHE_MAX = 30                 # 粗时间戳帧缓存上限

# ===== WiFi 协议 =====
CSI_WIFI_PROTOCOL = os.environ.get("CSI_WIFI_PROTOCOL", "LLTF")  # "LLTF" | "HT40"
CSI_ENABLE_CALIBRATION = False     # 默认关闭校准 (无校准参考数据时)

# ===== 跌倒检测算法参数 =====
FALL_SHORT_WINDOW = 10             # 短窗口大小 (检测窗口), 帧
FALL_LONG_WINDOW = 100             # 长窗口大小 (基线窗口), 帧
FALL_THRESHOLD = 3.0               # 方差比阈值, 超过此值判定为跌倒
FALL_ALERT_FRAMES = 3              # 连续多少帧超阈值才触发告警 (防误报)
FALL_COOLDOWN_FRAMES = 50          # 告警后冷却帧数, 防止重复报警

# ===== 特征向量滤波 =====
EIGENVEC_WINDOW_SIZE = 15          # 滑动窗口大小 (帧)

# ===== 告警推送 =====
CSI_FALL_PUSH_COOLDOWN = 5.0       # 告警推送冷却时间 (秒), 防止重复推送
CSI_FALL_LOCATION = (0.0, 0.0)     # CSI 无空间定位, 默认原点坐标

# ===== 诊断显示 =====
CSI_SHOW_DIAG = os.environ.get("CSI_SHOW_DIAG", "").lower() in ("1", "true", "yes")
CSI_DIAG_WIDTH = 600
CSI_DIAG_HEIGHT = 240

# ===== 天线映射 =====
# ESP32 硬件天线索引 → 4×2 网格天线 ID
ANTENNA_MAPPING = {
    6: "00", 3: "01",
    5: "10", 2: "11",
    4: "20", 1: "21",
    7: "30", 0: "31",
}
ANTENNA_ORDER = ["00", "01", "10", "11", "20", "21", "30", "31"]
EXPECTED_ANTENNAS = set(ANTENNA_ORDER)
