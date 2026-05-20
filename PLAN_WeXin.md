# 康复监测系统 - 微信小程序 + Flask API 服务器（最终完整方案）

---

## 一、项目背景

基于现有的三模态康复监测系统（YOLO 姿态估计 + 人脸识别 + 情绪分类），开发微信小程序和配套 API 服务器，实现移动端康复监测。

**现有系统能力：**

| 模块 | 文件 | 核心能力 |
|------|------|---------|
| 步态分析 | `gait_analyzer.py` | `get_metrics()` 返回 23 项步态指标 |
| 空间定位 | `spatial_mapper.py` | `get_trajectory()` 返回世界坐标轨迹 |
| 跌倒检测 | `fall_detector.py` | 速度+姿态融合，rule-based |
| 数据存储 | `data_logger.py` | 异步 SQLite，6 表，WAL 模式 |
| AI 报告 | `api_client.py` | DeepSeek API，生成临床摘要 |
| 人脸识别 | `face_locker.py` | FaceNet embedding + HSV 直方图 |
| 表情识别 | `emotion_recognizer.py` | 7 类情绪，滑动窗口投票 |

**运行环境：** Intel i5-12500H CPU，Conda 环境 `yolov26`（Python 3.11.15）

---

## 二、技术栈

| 层级 | 技术选型 | 原因 |
|------|---------|------|
| 后端框架 | Flask + flask_sock | 轻量，与现有代码易于集成 |
| WebSocket | flask_sock 原生支持 | 无需 flask-socketio 的额外复杂度 |
| API 版本 | `/api/v1/` 前缀 | 便于后续升级，不影响现有客户端 |
| 限流 | 装饰器 + 内存滑动窗口 | 无外部依赖，够用 |
| 图表 | Canvas 2D 手绘 | echarts-for-weixin 已停止维护，无依赖 |
| 小程序 | 原生开发（非 uni-app） | 用户指定 |
| 图片传输 | Base64 JPEG, quality=60 | 平衡清晰度和传输速度 |

---

## 三、架构设计

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│  PC: rehab_monitor/main.py                              │
│                                                         │
│  ┌─ 主线程 ─────────────────────────────────────────┐  │
│  │  Camera → PoseDetector → GaitAnalyzer             │  │
│  │    ├─ SpatialMapper  (世界坐标 + 躺下检测)        │  │
│  │    ├─ FallDetector   (跌倒检测)                   │  │
│  │    │   └─ alert触发 → broadcast_fall_alert() ──┐  │  │
│  │    ├─ FaceNetLocker  (人脸锁定)                 │  │  │
│  │    ├─ EmotionThread  (表情识别, 后台线程)       │  │  │
│  │    └─ DataLogger     (SQLite写入, 后台线程)     │  │  │
│  └────────────────────────────────────────────────┘  │  │
│                                                         │  │
│  ┌─ Flask API Server [daemon线程, :5000] ───────────┐  │  │
│  │  REST:  /api/v1/health|metrics|lock|report|...    │  │  │
│  │  WS:    /ws/metrics  ← 3s推送 + fall_alert ◄─────┘  │  │
│  └──────────────────────────────────────────────────┘  │  │
└─────────────────────────────────────────────────────┘    │
         │ 局域网 HTTP / WebSocket                           │
         ▼                                                  │
┌─────────────────────────────────────────────────────┐    │
│  微信小程序 (WeXin_Server/)                          │    │
│                                                       │    │
│  ┌─ app.js ───────────────────────────────────────┐  │    │
│  │  全局状态 | WebSocket连接 | 离线缓存 | 网络监听  │  │    │
│  └────────────────────────────────────────────────┘  │    │
│                                                       │    │
│  ┌─ TabBar ───────────────────────────────────────┐  │    │
│  │  [首页]  [趋势]  [拍照]  [报告]  [我的]         │  │    │
│  └────────────────────────────────────────────────┘  │    │
│                                                       │    │
│  首页: Canvas仪表 | 指标网格 | 位置红点 | 心情图表     │    │
│  趋势: Canvas 2D折线图 | 时间范围切换                 │    │
│  拍照: Camera组件 | 图片压缩 | 锁定接口                │    │
│  报告: DeepSeek文本 | 富文本展示 | 分享                │    │
│  我的: 患者信息 | 设置 | 关于                          │    │
└─────────────────────────────────────────────────────┘    │
```

### 3.2 线程模型

```
main.py 进程
├── 主线程: 帧处理 (pose → gait → fall → display)
├── EmotionThread (daemon): 表情识别
├── DataLoggerThread (daemon): SQLite 异步写入
├── ReportThread (daemon): DeepSeek API 调用
└── Flask Thread (daemon): API + WebSocket
    └── WebSocket 推送定时器 (每 3 秒)
```

### 3.3 数据流

```
Camera Frame
  → PoseDetector (OpenVINO INT8 + Kalman)
  → keypoints (N×17×3)
      ├→ GaitAnalyzer.update() → get_metrics() → 23项指标
      │    ├→ API: GET /realtime, GET /history
      │    └→ WebSocket: type=metrics (每3秒)
      │
      ├→ FallDetector.update() → fall_status
      │    └→ status变为alert → broadcast_fall_alert() (5s去重)
      │
      ├→ SpatialMapper → world_pos, trajectory
      │    └→ API: GET /trajectory
      │
      ├→ FaceNetLocker → face_embedding
      │    └→ API: POST /lock (match → enroll)
      │
      ├→ EmotionRecognizer → emotion_label + scores
      │    └→ API: GET /emotion/stats (via DB)
      │
      └→ DataLogger → SQLite (每5帧写入)
           ├→ gait_metrics 表 (19字段 + timestamp)
           ├→ emotion_log 表
           └→ API: GET /history, GET /emotion/stats (查询)
```

---

## 四、API 接口设计

### 4.1 接口清单

| 方法 | 路径 | 说明 | 限流 | 认证 |
|------|------|------|------|------|
| GET | `/api/v1/health` | 增强健康检查 | 无 | 无 |
| GET | `/api/v1/metrics/realtime` | 实时步态指标 | 2次/秒 | Token |
| GET | `/api/v1/metrics/history?days=7` | 历史趋势 | 2次/秒 | Token |
| POST | `/api/v1/patient/lock` | 拍照锁定(合并) | 1次/3秒 | Token |
| POST | `/api/v1/report/generate?seconds=30` | 生成康复报告 | 1次/5秒 | Token |
| GET | `/api/v1/emotion/stats?days=7` | 心情统计 | 2次/秒 | Token |
| GET | `/api/v1/trajectory` | 位置轨迹 | 2次/秒 | Token |
| WS | `/ws/metrics` | 实时推送+跌倒告警 | - | - |

### 4.2 统一响应格式

```json
// 成功
{"code": 0, "data": {...}, "message": "ok"}

// 失败
{"code": -1, "message": "错误描述"}

// 限流
HTTP 429
{"code": -1, "message": "请求过于频繁，请稍后再试"}
```

### 4.3 接口详细设计

#### GET /api/v1/health
```json
{
  "code": 0,
  "data": {
    "status": "ok",
    "version": "v1",
    "uptime": 3600.5,
    "gait_ready": true,
    "db_ready": true,
    "face_locker_ready": true,
    "websocket_clients": 2,
    "timestamp": 1715900000.0
  }
}
```

#### GET /api/v1/metrics/realtime
```json
{
  "code": 0,
  "data": {
    "step_count": 120,
    "left_steps": 58,
    "right_steps": 62,
    "cadence_spm": 95.5,
    "symmetry": 0.925,
    "symmetry_source": "stride",
    "stride_length_m": 1.25,
    "avg_step_length_m": 0.62,
    "gait_velocity_mps": 0.98,
    "step_width_m": 0.12,
    "avg_step_width_m": 0.115,
    "stance_time_s": 0.65,
    "swing_time_s": 0.42,
    "stance_percentage": 60.5,
    "left_knee_rom": 55.2,
    "right_knee_rom": 53.8,
    "step_time_cv": 8.5,
    "step_length_cv": 10.2,
    "step_width_cv": 15.3,
    "foot_clearance_cm": 4.2,
    "gait_rehab_score": 72.5,
    "trunk_sway_deg": 3.1,
    "is_double_support": true,
    "double_support_ratio": 0.28,
    "speed_pxps": 120.5,
    "orientation_yaw": 5.2
  }
}
```

#### GET /api/v1/metrics/history?days=7
```json
{
  "code": 0,
  "data": [
    {
      "date": "2026-05-10",
      "gait_rehab_score": 68.5,
      "gait_velocity_mps": 0.85,
      "symmetry": 0.89,
      "cadence_spm": 88.0,
      "stride_length_m": 1.15,
      "left_knee_rom": 48.5,
      "right_knee_rom": 47.2,
      "step_time_cv": 12.5
    }
  ]
}
```

#### POST /api/v1/patient/lock
```json
// 请求
{"image": "data:image/jpeg;base64,/9j/4AAQ..."}

// 响应 - 匹配成功
{
  "code": 0,
  "data": {
    "action": "match",
    "patientId": "target",
    "similarity": 0.85,
    "message": "欢迎回来，target"
  }
}

// 响应 - 新录入
{
  "code": 0,
  "data": {
    "action": "enroll",
    "patientId": "P1715900000",
    "similarity": 1.0,
    "message": "新患者已录入"
  }
}
```

#### WebSocket /ws/metrics
```json
// 实时指标 (每3秒)
{"type": "metrics", "data": { /* 23项指标 */ }, "timestamp": 1715900000.0}

// 跌倒告警 (事件触发, 5s去重)
{
  "type": "fall_alert",
  "data": {
    "timestamp": 1715900000.0,
    "location": [1.5, 2.3],
    "score": 0.85,
    "severity": "high"
  }
}

// 心跳
{"type": "ping"}
→ 客户端回复 "pong"
```

### 4.4 限流策略

- **限流键**: `客户端IP_session_id` 组合（局域网多设备不互限）
- **算法**: 滑动窗口，1 秒内计数
- **存储**: 内存 `defaultdict(list)` + `Lock` 线程安全
- **清理**: 每次请求时惰性清理过期记录

---

## 五、核心实现要点

### 5.1 WebSocket 客户端管理（防止内存泄漏）

```python
websocket_clients = {}   # {uuid: ws_connection}
clients_lock = Lock()

@sock.route('/ws/metrics')
def metrics_socket(ws):
    client_id = str(uuid.uuid4())
    with clients_lock:
        websocket_clients[client_id] = ws

    try:
        while True:
            # 心跳检测 (30s)
            # 指标推送 (3s)
            time.sleep(WEBSOCKET_PUSH_INTERVAL)
    finally:
        with clients_lock:
            websocket_clients.pop(client_id, None)
```

### 5.2 跌倒告警去重（5 秒冷却）

```python
# main.py 全局变量
_last_fall_push_time = 0
FALL_PUSH_COOLDOWN = 5  # 秒

# 跌倒检测后
if fall_status == "alert" and previous_fall_status != "alert":
    now = time.time()
    if now - _last_fall_push_time > FALL_PUSH_COOLDOWN:
        api_server.broadcast_fall_alert(spatial.current_pos, fall_score)
        _last_fall_push_time = now
```

### 5.3 人脸锁定合并逻辑

```
POST /api/v1/patient/lock (Base64图片)
  → Base64 解码为 OpenCV BGR
  → face_locker.enroll_from_frame(frame) → embedding
  → face_locker.match(embedding)
      ├─ 匹配成功 → 返回 {action: "match", patientId, similarity}
      └─ 匹配失败 → 生成新ID → 保存到 face_db.json
                   → face_locker._load_db()
                   → 返回 {action: "enroll", patientId, similarity: 1.0}
```

### 5.4 数据库索引（启动时智能创建）

```python
def _create_index_if_not_exists(self, conn, idx_name, table, columns):
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{idx_name}'")
    if not cursor.fetchone():
        cursor.execute(f"CREATE INDEX {idx_name} ON {table}({columns})")
        logger.info(f"索引已创建: {idx_name}")

# 三个索引
# idx_gait_timestamp  ON gait_metrics(timestamp)
# idx_gait_session    ON gait_metrics(session_id, timestamp)
# idx_emotion_timestamp ON emotion_log(timestamp)
```

### 5.5 请求限流装饰器

```python
def rate_limit(per_second=2):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            session_id = request.headers.get('X-Session-Id', '')
            client_key = f"{request.remote_addr}_{session_id}"
            now = time.time()
            with rate_limit_lock:
                window = rate_limit_storage[client_key]
                window[:] = [t for t in window if now - t < 1.0]
                if len(window) >= per_second:
                    return jsonify({"code": -1, "message": "请求过于频繁"}), 429
                window.append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator
```

### 5.6 图片压缩流程

```
小程序拍照
  → wx.createCameraContext().takePhoto({quality: 'low'})
  → wx.compressImage({quality: IMAGE_UPLOAD_QUALITY})  // 60
  → wx.getFileSystemManager().readFileSync(tempPath, 'base64')
  → POST /api/v1/patient/lock {image: "data:image/jpeg;base64,..."}
```

### 5.7 Canvas 手绘图表（替代 echarts）

```javascript
// trend.js 核心绘制逻辑
drawLineChart(ctx, data, width, height) {
  // 1. 计算Y轴范围 (minScore ~ maxScore)
  // 2. 计算X轴步进 (width / data.length)
  // 3. ctx.beginPath() + moveTo/lineTo 绘制折线
  // 4. ctx.arc() 绘制数据点
  // 5. ctx.fillText() 绘制X轴标签
  // 6. ctx.fillText() 绘制Y轴刻度
}
```

### 5.8 小程序离线缓存

```javascript
// app.js
globalData: {
  lastMetrics: null,        // 最近一次指标数据
  offlineMode: false,
  pendingRequests: []       // 网络恢复后重试
}

// 网络监听
wx.onNetworkStatusChange((res) => {
  this.globalData.offlineMode = !res.isConnected
  if (res.isConnected) this.flushPendingRequests()
})
```

---

## 六、配置管理

### 6.1 config.py 新增项

```python
# ===== API 服务器 =====
API_VERSION = "v1"
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
API_CORS_ORIGINS = ["*"]
API_RATE_LIMIT_PER_SECOND = 2
WEBSOCKET_HEARTBEAT_INTERVAL = 30  # 秒
WEBSOCKET_PUSH_INTERVAL = 3        # 秒

# ===== 图片上传 =====
IMAGE_UPLOAD_QUALITY = 60          # 压缩质量 1-100
IMAGE_MAX_SIZE_KB = 100            # 最大上传大小
IMAGE_MAX_WIDTH = 1024             # 最大宽度
```

### 6.2 main.py 启动参数

```bash
python -m rehab_monitor.main                  # 正常启动
python -m rehab_monitor.main --api-debug      # API调试模式
python -m rehab_monitor.main --api-port 8080  # 自定义端口
```

---

## 七、开发步骤（分 5 个阶段，预计总工时 ~8 小时）

### 阶段总览

```
阶段1: 后端基础设施 (1h)
  ├── Step 1.1: 修改 config.py，新增 API 配置项
  ├── Step 1.2: 修改 data_logger.py，添加索引 + 查询优化
  └── Step 1.3: 安装依赖 pip install flask flask_sock

阶段2: API 服务器 (2.5h)
  ├── Step 2.1: 新建 api_server.py 框架 (Flask + CORS + Blueprint)
  ├── Step 2.2: 实现 6 个 REST 接口
  ├── Step 2.3: 实现 WebSocket (/ws/metrics)
  ├── Step 2.4: 实现限流装饰器
  ├── Step 2.5: 实现跌倒告警广播
  └── Step 2.6: 修改 main.py 集成启动

阶段3: 小程序基础设施 (1h)
  ├── Step 3.1: 新建 utils/api.js
  ├── Step 3.2: 新建 utils/websocket.js
  ├── Step 3.3: 覆写 app.json (TabBar + 5页面)
  └── Step 3.4: 覆写 app.js (全局状态 + 离线缓存)

阶段4: 小程序页面 (3h)
  ├── Step 4.1: 首页 (wxml + wxss + js)
  ├── Step 4.2: 拍照锁定页 (camera.js)
  ├── Step 4.3: 趋势图表页 (trend.js)
  ├── Step 4.4: 报告生成页 (report.js)
  └── Step 4.5: 个人中心页 (profile.js)

阶段5: 组件 + 收尾 (0.5h)
  ├── Step 5.1: rehab-gauge 仪表组件
  ├── Step 5.2: package.json
  └── Step 5.3: 端到端验证测试
```

---

### 阶段 1：后端基础设施（预计 1 小时）

#### Step 1.1 — 修改 `rehab_monitor/config.py`

**目标**: 新增 API 服务器和图片上传配置项

**操作**:
1. 打开 `rehab_monitor/config.py`
2. 在文件末尾追加以下配置块：

```python
# ===== API 服务器 =====
API_VERSION = "v1"
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
API_CORS_ORIGINS = ["*"]
API_RATE_LIMIT_PER_SECOND = 2
WEBSOCKET_HEARTBEAT_INTERVAL = 30  # 心跳间隔 (秒)
WEBSOCKET_PUSH_INTERVAL = 3        # 指标推送间隔 (秒)

# ===== 图片上传 =====
IMAGE_UPLOAD_QUALITY = 60          # JPEG 压缩质量 1-100
IMAGE_MAX_SIZE_KB = 100            # 单张图片上限
IMAGE_MAX_WIDTH = 1024             # 图片最大宽度

# ===== 患者锁定 =====
LOCK_SIMILARITY_THRESHOLD = 0.55   # 人脸匹配阈值
```

**验证**: `python -c "from rehab_monitor.config import API_PORT; print(API_PORT)"` 输出 `5000`

---

#### Step 1.2 — 修改 `rehab_monitor/data_logger.py`

**目标**: 添加数据库索引，加速历史查询；优化 `get_recent_data()` 查询

**操作**:

1. 在 `_create_tables()` 方法内（所有 `CREATE TABLE IF NOT EXISTS` 之后），添加索引创建辅助方法和调用：

```python
# 在 RehabDatabase 类中添加辅助方法
def _create_index_if_not_exists(self, conn, idx_name, table, columns):
    """仅当索引不存在时才创建，避免每次启动重复执行"""
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT name FROM sqlite_master WHERE type='index' AND name='{idx_name}'"
    )
    if not cursor.fetchone():
        cursor.execute(f"CREATE INDEX {idx_name} ON {table}({columns})")
        logger.info("索引已创建: %s ON %s(%s)", idx_name, table, columns)

# 在 _create_tables() 末尾的 conn.commit() 之前调用
self._create_index_if_not_exists(conn, "idx_gait_timestamp",
    "gait_metrics", "timestamp")
self._create_index_if_not_exists(conn, "idx_gait_session",
    "gait_metrics", "session_id, timestamp")
self._create_index_if_not_exists(conn, "idx_emotion_timestamp",
    "emotion_log", "timestamp")
```

2. 修改 `get_recent_data()` 方法，在 SQL 查询末尾添加 `ORDER BY timestamp DESC LIMIT 1000`

**验证**: `sqlite3 rehab_data.db "SELECT name FROM sqlite_master WHERE type='index';"` 应显示 3 个新索引

---

#### Step 1.3 — 安装 Python 依赖

```bash
conda activate yolov26
pip install flask flask_sock
```

**验证**: `python -c "import flask; import flask_sock; print('OK')"`

---

### 阶段 2：API 服务器（预计 2.5 小时）

#### Step 2.1 — 新建 `rehab_monitor/api_server.py` 框架

**目标**: 搭建 Flask 应用骨架

**实现内容**:
1. 导入依赖：`flask`, `flask_cors`, `flask_sock`, `json`, `time`, `uuid`, `base64`, `threading` 等
2. 定义模块级全局变量（由 `start_api_server()` 注入）：
   ```python
   gait_analyzer = None
   spatial_mapper = None
   database = None
   face_locker = None
   detector = None
   ```
3. 实现 `create_app()` 工厂函数：
   - `Flask(__name__)`
   - `CORS(app, origins=API_CORS_ORIGINS)`
   - `Sock(app)`
   - 注册 Blueprint（路由在后续步骤添加）
4. 实现 `start_api_server(gait, spatial, db, face_locker, detector, emotion, **kwargs)` 函数：
   - 注入全局变量
   - 在 daemon 线程中启动 `app.run(host=..., port=..., debug=..., threaded=True)`
   - 返回 thread 对象
5. 定义限流存储和锁：
   ```python
   rate_limit_storage = defaultdict(list)
   rate_limit_lock = Lock()
   ```
6. 定义 WebSocket 客户端管理和锁：
   ```python
   websocket_clients = {}  # {uuid: ws}
   clients_lock = Lock()
   app_start_time = time.time()
   ```

**验证**: 暂时无法独立验证，依赖 main.py 集成

---

#### Step 2.2 — 实现 6 个 REST 接口

**目标**: 逐个实现所有 HTTP 接口

**实现顺序和逻辑**:

##### 2.2a: `GET /api/v1/health`（无依赖，最先实现）

```python
@app.route('/api/v1/health')
def health():
    return jsonify({
        "code": 0,
        "data": {
            "status": "ok",
            "version": API_VERSION,
            "uptime": round(time.time() - app_start_time, 1),
            "gait_ready": gait_analyzer is not None,
            "db_ready": database is not None,
            "face_locker_ready": face_locker is not None,
            "websocket_clients": len(websocket_clients),
            "timestamp": time.time()
        }
    })
```

##### 2.2b: `GET /api/v1/metrics/realtime`（依赖 gait_analyzer）

- 调用 `gait_analyzer.get_metrics()`
- 用 `@rate_limit(per_second=2)` 装饰
- 异常时返回 `{"code": -1, "message": "步态分析器未就绪"}`

##### 2.2c: `GET /api/v1/metrics/history?days=7`

- 查询参数：`days`（默认 7），`limit`（默认 30）
- SQL: `SELECT DATE(timestamp) as date, AVG(gait_rehab_score), AVG(gait_velocity_mps), ... FROM gait_metrics WHERE timestamp >= ? GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT ?`
- `@rate_limit(per_second=2)`

##### 2.2d: `GET /api/v1/emotion/stats?days=7`

- SQL: `SELECT DATE(timestamp) as date, emotion_label, COUNT(*) as cnt FROM emotion_log WHERE timestamp >= ? GROUP BY date, emotion_label ORDER BY date`
- 按天 + 情绪类型聚合

##### 2.2e: `GET /api/v1/trajectory?limit=100`

- 调用 `spatial_mapper.get_trajectory()`
- 返回 `[(x, y), ...]`
- 支持 `limit` 参数截取最近 N 个点

##### 2.2f: `POST /api/v1/patient/lock`（核心接口）

完整逻辑流程：
```
1. 解析 request.json → image_base64
2. 去掉 "data:image/jpeg;base64," 前缀
3. base64.b64decode() → np.frombuffer() → cv2.imdecode()
4. face_locker.enroll_from_frame(frame) → embedding
5. 如果 embedding 为 None:
   → 返回 {"code": -1, "message": "未检测到人脸，请正对摄像头"}
6. face_locker.match(embedding) → (name, similarity)
7. 如果 name 不为 None 且 similarity >= LOCK_SIMILARITY_THRESHOLD:
   → 返回 {"code": 0, "data": {"action": "match", ...}}
8. 否则:
   → 生成 patient_id = f"P{int(time.time())}"
   → 更新 face_db.json，添加新条目
   → face_locker._load_db()
   → 返回 {"code": 0, "data": {"action": "enroll", ...}}
```

##### 2.2g: `POST /api/v1/report/generate`

- 查询参数 `seconds`（默认 30）
- 调用 `database.get_recent_data(seconds=seconds)`
- 调用 `api_client.generate_report(data)`（现有函数）
- 返回报告文本 + 摘要

**验证**: 
```bash
# 启动后测试
curl http://localhost:5000/api/v1/health
curl http://localhost:5000/api/v1/metrics/realtime
```

---

#### Step 2.3 — 实现 WebSocket `/ws/metrics`

**目标**: 实时推送步态指标，支持心跳和断线清理

**实现逻辑**:
```python
@sock.route('/ws/metrics')
def metrics_socket(ws):
    client_id = str(uuid.uuid4())
    with clients_lock:
        websocket_clients[client_id] = ws
    logger.info("WebSocket 客户端连接: %s (当前 %d 个)", client_id, len(websocket_clients))

    last_ping = time.time()
    try:
        while True:
            now = time.time()
            # 心跳 (30s)
            if now - last_ping > WEBSOCKET_HEARTBEAT_INTERVAL:
                ws.send(json.dumps({"type": "ping"}))
                last_ping = now
            # 指标推送 (3s)
            if gait_analyzer is not None:
                metrics = gait_analyzer.get_metrics()
                ws.send(json.dumps({
                    "type": "metrics",
                    "data": metrics,
                    "timestamp": now
                }))
            time.sleep(WEBSOCKET_PUSH_INTERVAL)
    except Exception as e:
        logger.debug("WebSocket 断开: %s (%s)", client_id, e)
    finally:
        with clients_lock:
            websocket_clients.pop(client_id, None)
        logger.info("WebSocket 客户端断开: %s (剩余 %d 个)", client_id, len(websocket_clients))
```

**验证**: `websocat ws://localhost:5000/ws/metrics` 每 3 秒收到数据

---

#### Step 2.4 — 实现限流装饰器

已在 Step 2.1 中准备好 `rate_limit_storage` 和 `rate_limit_lock`，在此步骤完善：

```python
from functools import wraps

def rate_limit(per_second=2):
    """请求限流装饰器，使用 session_id + IP 组合键"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            session_id = request.headers.get('X-Session-Id', 'default')
            client_key = f"{request.remote_addr}_{session_id}"
            now = time.time()
            with rate_limit_lock:
                window = rate_limit_storage[client_key]
                # 惰性清理过期记录
                window[:] = [t for t in window if now - t < 1.0]
                if len(window) >= per_second:
                    return jsonify({
                        "code": -1,
                        "message": "请求过于频繁，请稍后再试"
                    }), 429
                window.append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator
```

应用到接口：`@rate_limit(per_second=2)` 装饰 metrics/realtime、metrics/history 等

**验证**: 连续 3 次 `curl /api/v1/metrics/realtime`，第 3 次返回 429

---

#### Step 2.5 — 实现跌倒告警广播

**目标**: 提供 `broadcast_fall_alert()` 函数，由 main.py 调用

```python
def broadcast_fall_alert(location, score):
    """向所有 WebSocket 客户端广播跌倒告警（线程安全）"""
    if not websocket_clients:
        return
    alert = json.dumps({
        "type": "fall_alert",
        "data": {
            "timestamp": time.time(),
            "location": list(location) if location else [0, 0],
            "score": round(score, 3),
            "severity": "high" if score > 0.8 else "medium"
        }
    })
    with clients_lock:
        dead = []
        for cid, ws in websocket_clients.items():
            try:
                ws.send(alert)
            except Exception:
                dead.append(cid)
        for cid in dead:
            websocket_clients.pop(cid, None)
```

**验证**: 需等 main.py 集成后，模拟跌倒事件触发

---

#### Step 2.6 — 修改 `rehab_monitor/main.py` 集成 API 服务器

**目标**: 在 main.py 中启动 API 后台线程，集成跌倒告警调用

**具体修改点**:

1. **文件头部**：添加 `import argparse`

2. **main() 函数开头**（在 `def main():` 之后，初始化模块之前）：
```python
parser = argparse.ArgumentParser(description="康复监测系统")
parser.add_argument('--api-debug', action='store_true', help='启用API调试模式')
parser.add_argument('--api-port', type=int, default=API_PORT, help='API服务器端口')
args = parser.parse_args()
```

3. **初始化模块之后**（约 ~line 131，emotion_thread.start() 之后，session 创建之后）：
```python
# 启动 API 服务器
from .api_server import start_api_server
api_thread = start_api_server(
    detector, gait, spatial, fall, db, face_locker, emotion_recognizer,
    debug=args.api_debug,
    port=args.api_port
)
logger.info("API 服务器已启动 http://0.0.0.0:%d (debug=%s)", args.api_port, args.api_debug)
```

4. **循环开始前**（约 ~line 161，running=True 之后）添加：
```python
previous_fall_status = "safe"
_last_fall_push_time = 0
FALL_PUSH_COOLDOWN = 5  # 跌倒告警冷却时间 (秒)
```

5. **跌倒检测后**（约 ~line 408，fall_status 赋值之后）添加：
```python
if fall_status == "alert" and previous_fall_status != "alert":
    now_ts = time.time()
    if now_ts - _last_fall_push_time > FALL_PUSH_COOLDOWN:
        try:
            from .api_server import broadcast_fall_alert
            broadcast_fall_alert(spatial.current_pos if spatial else (0, 0), fall_score)
        except Exception:
            pass
        _last_fall_push_time = now_ts
previous_fall_status = fall_status
```

**验证**: 
```bash
python -m rehab_monitor.main --api-debug
# 控制台应打印 "API 服务器已启动 http://0.0.0.0:5000"
# curl http://localhost:5000/api/v1/health 应返回 JSON
```

---

### 阶段 3：小程序基础设施（预计 1 小时）

#### Step 3.1 — 新建 `WeXin_Server/utils/api.js`

**目标**: 封装 6 个 API 调用函数

**实现内容**:
```javascript
const API_BASE = 'http://192.168.1.100:5000/api/v1';  // 需根据实际 IP 修改
const SESSION_ID = 'wx_' + Date.now();

function request(method, path, data = null) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: API_BASE + path,
      method: method,
      data: data,
      header: {
        'Content-Type': 'application/json',
        'X-Session-Id': SESSION_ID
      },
      success(res) {
        if (res.statusCode === 200 && res.data.code === 0) {
          resolve(res.data.data);
        } else {
          reject(res.data);
        }
      },
      fail(err) {
        reject({ code: -1, message: '网络请求失败', error: err });
      }
    });
  });
}

// 6 个封装函数
export function getRealtimeMetrics() { return request('GET', '/metrics/realtime'); }
export function getHistoryTrend(days = 7) { return request('GET', `/metrics/history?days=${days}`); }
export function lockPatient(imageBase64) { return request('POST', '/patient/lock', { image: imageBase64 }); }
export function generateReport(seconds = 30) { return request('POST', `/report/generate?seconds=${seconds}`); }
export function getEmotionStats(days = 7) { return request('GET', `/emotion/stats?days=${days}`); }
export function getTrajectory(limit = 100) { return request('GET', `/trajectory?limit=${limit}`); }
```

**验证**: 在小程序页面中 `import { getRealtimeMetrics } from '../../utils/api'`

---

#### Step 3.2 — 新建 `WeXin_Server/utils/websocket.js`

**目标**: WebSocket 连接管理，支持心跳、重连、多消息类型

**实现内容**:
```javascript
class MetricsSocket {
  constructor() {
    this.socket = null;
    this.url = '';
    this.callbacks = [];
    this.reconnectTimer = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.heartbeatTimer = null;
  }

  connect(url) {
    this.url = url;
    this._doConnect();
  }

  _doConnect() {
    this.socket = wx.connectSocket({ url: this.url });
    
    this.socket.onOpen(() => {
      this.reconnectAttempts = 0;
      this._startHeartbeat();
    });

    this.socket.onMessage((res) => {
      const msg = JSON.parse(res.data);
      if (msg.type === 'ping') {
        this.socket.send({ data: 'pong' });
        return;
      }
      // 分发给所有订阅者
      this.callbacks.forEach(cb => {
        try { cb(msg); } catch (e) {}
      });
    });

    this.socket.onClose(() => this._onClose());
    this.socket.onError(() => this._onClose());
  }

  subscribe(callback) { this.callbacks.push(callback); }

  _startHeartbeat() {
    this.heartbeatTimer = setInterval(() => {
      if (this.socket) this.socket.send({ data: 'ping' });
    }, 30000);
  }

  _onClose() {
    clearInterval(this.heartbeatTimer);
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      this.reconnectTimer = setTimeout(() => {
        this._doConnect();
      }, 5000);
    }
  }

  disconnect() {
    clearInterval(this.heartbeatTimer);
    clearTimeout(this.reconnectTimer);
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}

module.exports = { MetricsSocket };
```

**验证**: 在其他模块 `const { MetricsSocket } = require('../../utils/websocket')`

---

#### Step 3.3 — 覆写 `WeXin_Server/app.json`

**目标**: 配置 5 个页面 + 底部 TabBar

```json
{
  "pages": [
    "pages/index/index",
    "pages/trend/trend",
    "pages/camera/camera",
    "pages/report/report",
    "pages/profile/profile"
  ],
  "window": {
    "navigationBarBackgroundColor": "#1a73e8",
    "navigationBarTitleText": "康复监测",
    "navigationBarTextStyle": "white"
  },
  "tabBar": {
    "color": "#999999",
    "selectedColor": "#1a73e8",
    "list": [
      { "pagePath": "pages/index/index", "text": "首页", "iconPath": "", "selectedIconPath": "" },
      { "pagePath": "pages/trend/trend", "text": "趋势", "iconPath": "", "selectedIconPath": "" },
      { "pagePath": "pages/camera/camera", "text": "拍照", "iconPath": "", "selectedIconPath": "" },
      { "pagePath": "pages/report/report", "text": "报告", "iconPath": "", "selectedIconPath": "" },
      { "pagePath": "pages/profile/profile", "text": "我的", "iconPath": "", "selectedIconPath": "" }
    ]
  },
  "style": "v2",
  "sitemapLocation": "sitemap.json"
}
```

---

#### Step 3.4 — 覆写 `WeXin_Server/app.js`

**目标**: 全局状态管理 + WebSocket 初始化 + 离线缓存

```javascript
const { MetricsSocket } = require('./utils/websocket');

App({
  globalData: {
    patientLocked: false,
    patientId: null,
    lastMetrics: null,
    lastEmotionStats: null,
    offlineMode: false,
    pendingRequests: [],
    wsConnected: false
  },

  socket: null,

  onLaunch() {
    // 检查本地存储的患者锁定状态
    const locked = wx.getStorageSync('patientLocked');
    if (locked) {
      this.globalData.patientLocked = true;
      this.globalData.patientId = wx.getStorageSync('patientId');
    }

    // 网络状态监听
    wx.onNetworkStatusChange((res) => {
      this.globalData.offlineMode = !res.isConnected;
      if (res.isConnected) {
        this._flushPendingRequests();
      }
    });

    // 初始化 WebSocket（如果有锁定患者）
    if (this.globalData.patientLocked) {
      this._initWebSocket();
    }
  },

  _initWebSocket() {
    this.socket = new MetricsSocket();
    // 默认回调：更新 lastMetrics
    this.socket.subscribe((msg) => {
      if (msg.type === 'metrics') {
        this.globalData.lastMetrics = msg.data;
      }
      if (msg.type === 'fall_alert') {
        // 触发全局跌倒告警事件（首页监听）
        this._emitFallAlert(msg.data);
      }
    });
    const wsUrl = this._getWsUrl();
    this.socket.connect(wsUrl);
  },

  _getWsUrl() {
    const apiBase = wx.getStorageSync('apiBaseUrl') || '192.168.1.100:5000';
    return `ws://${apiBase}/ws/metrics`;
  },

  _flushPendingRequests() {
    const requests = [...this.globalData.pendingRequests];
    this.globalData.pendingRequests = [];
    requests.forEach(req => {
      wx.request(req);
    });
  },

  _emitFallAlert(data) {
    // 通过事件总线通知各页面
    const pages = getCurrentPages();
    pages.forEach(page => {
      if (page.onFallAlert) {
        page.onFallAlert(data);
      }
    });
  }
});
```

---

### 阶段 4：小程序页面（预计 3 小时）

#### Step 4.1 — 首页（3 个文件）

##### `pages/index/index.wxml` — 页面结构

```
页面布局（自上而下）：
┌──────────────────────────────┐
│ 患者锁定状态卡片              │
│ [患者: target] [状态: 已锁定]  │
│ [重新锁定按钮]                │
├──────────────────────────────┤
│ 康复度仪表盘 (Canvas)         │
│ GRS: 72.5 分                │
├──────────────────────────────┤
│ 8 项关键指标网格 (2×4)       │
│ 步速 | 步长 | 对称性 | 步频   │
│ 左膝ROM | 右膝ROM | 风险 | 足廓清│
├──────────────────────────────┤
│ 实时位置简图 (Canvas)         │
│ 康复大厅地图 + 红点           │
├──────────────────────────────┤
│ 心情统计迷你图 (Canvas)       │
│ 近7天情绪分布               │
├──────────────────────────────┤
│ 快捷操作按钮                 │
│ [一键报告] [分享] [紧急求助]   │
└──────────────────────────────┘
```

关键 wxml 元素：
```xml
<view class="container">
  <!-- 患者锁定卡片 -->
  <view class="card lock-card">
    <text>患者: {{patientName || '未锁定'}}</text>
    <text>状态: {{locked ? '已锁定' : '未锁定'}}</text>
    <button bindtap="goToCamera">重新锁定</button>
  </view>

  <!-- 康复度仪表 (Canvas 2D) -->
  <canvas type="2d" id="gaugeCanvas" class="gauge-canvas"></canvas>
  <text class="grs-score">{{grsScore}} 分</text>

  <!-- 8 项指标网格 -->
  <view class="metrics-grid">
    <metric-card wx:for="{{metricsList}}" wx:key="label"
      label="{{item.label}}" value="{{item.value}}"
      unit="{{item.unit}}" status="{{item.status}}" />
  </view>

  <!-- 位置简图 -->
  <canvas type="2d" id="positionCanvas" class="position-canvas"></canvas>

  <!-- 心情统计 -->
  <canvas type="2d" id="emotionCanvas" class="emotion-canvas"></canvas>

  <!-- 快捷按钮 -->
  <view class="quick-actions">
    <button bindtap="quickReport">一键报告</button>
    <button open-type="share">分享</button>
    <button bindtap="emergency">紧急求助</button>
  </view>
</view>
```

##### `pages/index/index.wxss` — 页面样式

关键样式：
- `.container`: flex 纵向布局，padding 16rpx，背景色 #f5f5f5
- `.card`: 白色圆角卡片，阴影，margin-bottom 16rpx
- `.metrics-grid`: CSS Grid 2 列，gap 12rpx
- `.gauge-canvas`: 宽 300rpx，高 150rpx，居中

##### `pages/index/index.js` — 页面逻辑

核心逻辑：
```javascript
const { getRealtimeMetrics, getEmotionStats, getTrajectory, generateReport } = require('../../utils/api');
const app = getApp();

Page({
  data: {
    locked: false,
    patientName: '',
    grsScore: 0,
    metricsList: [],  // 8 个 {label, value, unit, status}
    lastFallAlert: null
  },

  onLoad() {
    this.setData({
      locked: app.globalData.patientLocked,
      patientName: app.globalData.patientId
    });
    this._startPolling();
    this._setupWebSocket();
  },

  // 每 3 秒轮询实时指标（WebSocket 备用）
  _startPolling() {
    this.pollTimer = setInterval(async () => {
      try {
        const metrics = await getRealtimeMetrics();
        this._updateMetrics(metrics);
      } catch (e) {
        // WebSocket 可用时静默失败
      }
    }, 3000);
  },

  // WebSocket 订阅
  _setupWebSocket() {
    if (!app.socket) app._initWebSocket();
    app.socket.subscribe((msg) => {
      if (msg.type === 'metrics') this._updateMetrics(msg.data);
    });
  },

  // 更新指标数据 → 设置 data + 绘制仪表
  _updateMetrics(metrics) {
    const list = [
      { label: '步速', value: metrics.gait_velocity_mps?.toFixed(2), unit: 'm/s',
        status: metrics.gait_velocity_mps > 0.8 ? 'good' : 'warn' },
      { label: '步长', value: metrics.stride_length_m?.toFixed(2), unit: 'm',
        status: metrics.stride_length_m > 1.0 ? 'good' : 'warn' },
      { label: '对称性', value: (metrics.symmetry * 100).toFixed(0), unit: '%',
        status: metrics.symmetry > 0.85 ? 'good' : 'warn' },
      { label: '步频', value: metrics.cadence_spm?.toFixed(0), unit: '步/分',
        status: metrics.cadence_spm > 70 ? 'good' : 'warn' },
      { label: '左膝ROM', value: metrics.left_knee_rom?.toFixed(0), unit: '°',
        status: metrics.left_knee_rom > 40 ? 'good' : 'warn' },
      { label: '右膝ROM', value: metrics.right_knee_rom?.toFixed(0), unit: '°',
        status: metrics.right_knee_rom > 40 ? 'good' : 'warn' },
      { label: '跌倒风险', value: metrics.step_time_cv < 15 ? '低' : '高', unit: '',
        status: metrics.step_time_cv < 15 ? 'good' : 'danger' },
      { label: '足廓清', value: metrics.foot_clearance_cm?.toFixed(1), unit: 'cm',
        status: metrics.foot_clearance_cm > 3 ? 'good' : 'warn' },
    ];
    this.setData({
      grsScore: metrics.gait_rehab_score?.toFixed(0) || 0,
      metricsList: list
    });
    this._drawGauge(metrics.gait_rehab_score || 0);
  },

  // Canvas 绘制康复度仪表盘
  _drawGauge(score) {
    const query = wx.createSelectorQuery();
    query.select('#gaugeCanvas').fields({ node: true, size: true }).exec((res) => {
      if (!res[0]) return;
      const canvas = res[0].node;
      const ctx = canvas.getContext('2d');
      const dpr = wx.getSystemInfoSync().pixelRatio;
      canvas.width = 300 * dpr;
      canvas.height = 150 * dpr;
      ctx.scale(dpr, dpr);

      // 绘制半圆仪表盘
      const cx = 150, cy = 130, r = 100;
      const startAngle = Math.PI;  // 180°
      const endAngle = startAngle + (score / 100) * Math.PI;  // 180° → 360°

      // 背景弧
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 16;
      ctx.stroke();

      // 前景弧 (颜色: 红<50 黄50-70 绿>70)
      const color = score < 50 ? '#e74c3c' : score < 70 ? '#f39c12' : '#27ae60';
      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, endAngle);
      ctx.strokeStyle = color;
      ctx.stroke();

      // 分数文字
      ctx.fillStyle = color;
      ctx.font = 'bold 36px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(Math.round(score), cx, cy - 10);
      ctx.font = '14px sans-serif';
      ctx.fillText('GRS 评分', cx, cy + 20);
    });
  },

  // 跌倒告警处理（由 app.js 事件总线触发）
  onFallAlert(data) {
    wx.showModal({
      title: '跌倒告警',
      content: `检测到跌倒事件！位置: (${data.location[0]?.toFixed(1)}, ${data.location[1]?.toFixed(1)})`,
      confirmText: '知道了',
      showCancel: false
    });
  },

  quickReport() { /* 调用 generateReport */ },
  goToCamera() { wx.navigateTo({ url: '/pages/camera/camera' }); },
  emergency() { wx.makePhoneCall({ phoneNumber: '120' }); },

  onUnload() {
    clearInterval(this.pollTimer);
  }
});
```

---

#### Step 4.2 — 新建 `pages/camera/camera.js`

**目标**: 拍照 → 压缩 → 调用锁定接口

核心流程：
```javascript
Page({
  takePhoto() {
    const ctx = wx.createCameraContext();
    ctx.takePhoto({
      quality: 'low',
      success: (res) => {
        wx.compressImage({
          src: res.tempImagePath,
          quality: 60,
          success: (compressRes) => {
            const fs = wx.getFileSystemManager();
            const base64 = fs.readFileSync(compressRes.tempImagePath, 'base64');
            this._lockPatient('data:image/jpeg;base64,' + base64);
          }
        });
      }
    });
  },

  chooseFromAlbum() {
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album'],
      success: (res) => {
        wx.compressImage({
          src: res.tempFilePaths[0],
          quality: 60,
          success: (compressRes) => {
            const fs = wx.getFileSystemManager();
            const base64 = fs.readFileSync(compressRes.tempImagePath, 'base64');
            this._lockPatient('data:image/jpeg;base64,' + base64);
          }
        });
      }
    });
  },

  async _lockPatient(imageBase64) {
    wx.showLoading({ title: '识别中...' });
    try {
      const { lockPatient } = require('../../utils/api');
      const result = await lockPatient(imageBase64);
      wx.hideLoading();

      const app = getApp();
      app.globalData.patientLocked = true;
      app.globalData.patientId = result.patientId;
      wx.setStorageSync('patientLocked', true);
      wx.setStorageSync('patientId', result.patientId);

      wx.showToast({ title: result.message, icon: 'success' });
      setTimeout(() => wx.navigateBack(), 1500);
    } catch (e) {
      wx.hideLoading();
      wx.showToast({ title: e.message || '锁定失败', icon: 'error' });
    }
  }
});
```

wxml 配套元素（此处只列出关键部分）：
```xml
<camera device-position="front" flash="off" binderror="onCameraError" />
<button bindtap="takePhoto">拍照锁定</button>
<button bindtap="chooseFromAlbum">从相册选择</button>
```

---

#### Step 4.3 — 新建 `pages/trend/trend.js`

**目标**: Canvas 2D 手绘康复评分趋势折线图 + 时间范围切换

核心逻辑：
```javascript
Page({
  data: {
    timeRange: 7,  // 7 / 30 / 90 天
    chartData: [],
    emotionData: []
  },

  onLoad() { this.loadData(); },

  onReady() {
    // 获取 Canvas 节点
    const query = wx.createSelectorQuery();
    query.select('#trendCanvas').fields({ node: true, size: true }).exec((res) => {
      if (res[0]) {
        this.canvas = res[0].node;
        this.ctx = this.canvas.getContext('2d');
        const dpr = wx.getSystemInfoSync().pixelRatio;
        this.canvas.width = res[0].width * dpr;
        this.canvas.height = res[0].height * dpr;
        this.ctx.scale(dpr, dpr);
        this.canvasWidth = res[0].width;
        this.canvasHeight = res[0].height;
      }
    });
  },

  async loadData() {
    const { getHistoryTrend, getEmotionStats } = require('../../utils/api');
    try {
      const [trend, emotion] = await Promise.all([
        getHistoryTrend(this.data.timeRange),
        getEmotionStats(this.data.timeRange)
      ]);
      this.setData({ chartData: trend, emotionData: emotion });
      this.drawChart();
    } catch (e) {
      wx.showToast({ title: '数据加载失败', icon: 'none' });
    }
  },

  switchRange(e) {
    const days = parseInt(e.currentTarget.dataset.days);
    this.setData({ timeRange: days });
    this.loadData();
  },

  drawChart() {
    if (!this.ctx || !this.data.chartData.length) return;
    const ctx = this.ctx;
    const w = this.canvasWidth - 40;   // 留出左边距
    const h = this.canvasHeight - 60;  // 留出上下边距

    // 清空
    ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

    const data = this.data.chartData;
    const scores = data.map(d => d.gait_rehab_score);
    const maxY = 100, minY = 0;
    const xStep = w / Math.max(data.length - 1, 1);
    const yScale = h / (maxY - minY);

    // ---- 折线 ----
    ctx.beginPath();
    ctx.strokeStyle = '#1a73e8';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    data.forEach((point, i) => {
      const x = 20 + i * xStep;
      const y = 10 + h - (point.gait_rehab_score - minY) * yScale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // ---- 数据点 ----
    data.forEach((point, i) => {
      const x = 20 + i * xStep;
      const y = 10 + h - (point.gait_rehab_score - minY) * yScale;
      ctx.beginPath();
      ctx.fillStyle = '#1a73e8';
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });

    // ---- X 轴标签 ----
    ctx.fillStyle = '#888';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    const step = Math.max(Math.floor(data.length / 6), 1);
    data.forEach((point, i) => {
      if (i % step === 0) {
        const x = 20 + i * xStep;
        ctx.fillText(point.date.slice(5), x, h + 30);
      }
    });
  }
});
```

wxml 关键元素：
```xml
<view class="time-tabs">
  <text data-days="7" bindtap="switchRange">近1周</text>
  <text data-days="30" bindtap="switchRange">近1月</text>
  <text data-days="90" bindtap="switchRange">近3月</text>
</view>
<canvas type="2d" id="trendCanvas" style="width:100%;height:500rpx;"></canvas>
```

---

#### Step 4.4 — 新建 `pages/report/report.js`

**目标**: 选择周期 → 调用 DeepSeek API → 展示文本报告 → 支持分享

核心逻辑：
```javascript
Page({
  data: {
    seconds: 30,
    reportText: '',
    reportSummary: '',
    loading: false
  },

  async generateReport() {
    this.setData({ loading: true });
    try {
      const { generateReport } = require('../../utils/api');
      const result = await generateReport(this.data.seconds);
      this.setData({
        reportText: result.text || '无报告内容',
        reportSummary: result.summary || '',
        loading: false
      });
    } catch (e) {
      this.setData({ loading: false });
      wx.showToast({ title: '报告生成失败', icon: 'error' });
    }
  },

  changePeriod(e) {
    this.setData({ seconds: parseInt(e.currentTarget.dataset.seconds) });
  },

  onShareAppMessage() {
    return {
      title: `康复报告 - ${this.data.reportSummary}`,
      path: '/pages/report/report'
    };
  }
});
```

---

#### Step 4.5 — 新建 `pages/profile/profile.js`

**目标**: 患者信息展示 + 设置 + 关于

```javascript
Page({
  data: {
    patientId: '',
    locked: false,
    notificationsEnabled: true,
    apiBaseUrl: ''
  },

  onLoad() {
    const app = getApp();
    this.setData({
      patientId: app.globalData.patientId || '未锁定',
      locked: app.globalData.patientLocked,
      apiBaseUrl: wx.getStorageSync('apiBaseUrl') || '192.168.1.100:5000'
    });
  },

  toggleNotification(e) {
    this.setData({ notificationsEnabled: e.detail.value });
  },

  changeApiBase() {
    wx.showModal({
      title: '修改服务器地址',
      editable: true,
      placeholderText: '192.168.1.100:5000',
      success: (res) => {
        if (res.confirm) {
          wx.setStorageSync('apiBaseUrl', res.content);
          this.setData({ apiBaseUrl: res.content });
        }
      }
    });
  },

  switchPatient() {
    wx.navigateTo({ url: '/pages/camera/camera' });
  }
});
```

---

### 阶段 5：组件 + 收尾（预计 0.5 小时）

#### Step 5.1 — 新建 `components/rehab-gauge/index.js`

**目标**: 可复用的康复度仪表盘 Canvas 组件

**实现**（Component 模式）:
```javascript
Component({
  properties: {
    score: { type: Number, value: 0, observer: '_draw' },
    size:  { type: Number, value: 150 }
  },

  data: { canvasId: 'gauge_' + Math.random().toString(36).slice(2, 8) },

  lifetimes: {
    attached() { this._draw(); }
  },

  methods: {
    _draw() {
      const query = this.createSelectorQuery();
      query.select('#' + this.data.canvasId).fields({ node: true, size: true }).exec((res) => {
        if (!res[0]) return;
        const canvas = res[0].node;
        const ctx = canvas.getContext('2d');
        const dpr = wx.getSystemInfoSync().pixelRatio;
        const s = this.properties.size;
        canvas.width = s * dpr;
        canvas.height = (s * 0.5 + 30) * dpr;
        ctx.scale(dpr, dpr);

        const score = this.properties.score;
        const color = score < 50 ? '#e74c3c' : score < 70 ? '#f39c12' : '#27ae60';

        // 背景弧
        ctx.beginPath();
        ctx.arc(s/2, s/2, s/2 - 10, Math.PI, 2 * Math.PI);
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 12;
        ctx.stroke();

        // 前景弧
        ctx.beginPath();
        ctx.arc(s/2, s/2, s/2 - 10, Math.PI, Math.PI + (score / 100) * Math.PI);
        ctx.strokeStyle = color;
        ctx.stroke();

        // 分数
        ctx.fillStyle = color;
        ctx.font = `bold ${s * 0.2}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(score), s/2, s/2 - 5);
      });
    }
  }
});
```

---

#### Step 5.2 — 新建 `WeXin_Server/package.json`

```json
{
  "name": "rehab-monitor-wechat",
  "version": "1.0.0",
  "description": "康复监测微信小程序",
  "main": "app.js",
  "dependencies": {},
  "devDependencies": {}
}
```

> 注意：使用 Canvas 2D 替代 echarts，因此无需 npm 依赖。如果后续需要 echarts，运行 `npm install echarts-for-weixin`。

---

#### Step 5.3 — 端到端验证测试

**Step 5.3a: 启动系统**
```bash
conda activate yolov26
cd f:/Code/Anaconda/yolov26
python -m rehab_monitor.main --api-debug
```

**Step 5.3b: 后端 API 逐项验证**
```bash
# 1. 健康检查 (预期: 200, status ok)
curl -s http://localhost:5000/api/v1/health | python -m json.tool

# 2. 实时指标 (预期: 200, 23个字段)
curl -s http://localhost:5000/api/v1/metrics/realtime | python -m json.tool

# 3. 限流测试 (预期: 第3次返回 -1)
for i in 1 2 3 4 5; do
  curl -s -H "X-Session-Id: test" http://localhost:5000/api/v1/metrics/realtime | python -c "import sys,json;print(json.load(sys.stdin)['code'])"
done

# 4. 数据库索引 (预期: 3个索引)
sqlite3 rehab_data.db "SELECT name FROM sqlite_master WHERE type='index';"

# 5. WebSocket (预期: 每3秒收到数据)
# 需安装 websocat: winget install websocat 或 pip install websocat
websocat ws://localhost:5000/ws/metrics
```

**Step 5.3c: 小程序端验证**
1. 微信开发者工具 → 导入项目 → 选择 `WeXin_Server/`
2. 修改 `utils/api.js` 中的 `API_BASE` 为 PC 实际 IP
3. 验证首页：仪表盘动画、指标卡片颜色、位置红点
4. 验证拍照：拍照 → 压缩 → 锁定 → 成功后返回首页
5. 验证趋势：折线图绘制、时间范围切换
6. 验证报告：点击生成 → 等待 DeepSeek → 展示文本
7. 验证离线：关闭 PC WiFi → 首页显示缓存数据

---

## 八、文件清单及输出顺序（17 个文件）

按依赖关系排序：先后端配置 → 后端存储 → 后端 API → 后端集成 → 小程序工具 → 小程序全局 → 小程序页面 → 小程序组件

| # | 文件 | 操作 | 行数估算 |
|---|------|------|---------|
| 1 | `rehab_monitor/config.py` | 修改 | +15 行 |
| 2 | `rehab_monitor/data_logger.py` | 修改 | +20 行 |
| 3 | `rehab_monitor/api_server.py` | **新建** | ~300 行 |
| 4 | `rehab_monitor/main.py` | 修改 | +35 行 |
| 5 | `WeXin_Server/utils/api.js` | **新建** | ~120 行 |
| 6 | `WeXin_Server/utils/websocket.js` | **新建** | ~100 行 |
| 7 | `WeXin_Server/app.json` | 覆写 | ~40 行 |
| 8 | `WeXin_Server/app.js` | 覆写 | ~80 行 |
| 9 | `WeXin_Server/pages/index/index.wxml` | **新建** | ~100 行 |
| 10 | `WeXin_Server/pages/index/index.wxss` | **新建** | ~120 行 |
| 11 | `WeXin_Server/pages/index/index.js` | **新建** | ~200 行 |
| 12 | `WeXin_Server/pages/camera/camera.js` | **新建** | ~150 行 |
| 13 | `WeXin_Server/pages/trend/trend.js` | **新建** | ~180 行 |
| 14 | `WeXin_Server/pages/report/report.js` | **新建** | ~120 行 |
| 15 | `WeXin_Server/pages/profile/profile.js` | **新建** | ~90 行 |
| 16 | `WeXin_Server/components/rehab-gauge/index.js` | **新建** | ~80 行 |
| 17 | `WeXin_Server/package.json` | **新建** | ~15 行 |

---

## 九、验证测试

### 8.1 后端 API 测试

```bash
# 启动后端（调试模式）
conda activate yolov26
cd f:/Code/Anaconda/yolov26
python -m rehab_monitor.main --api-debug

# 终端1: 健康检查
curl -s http://localhost:5000/api/v1/health | python -m json.tool

# 终端2: 实时指标（带 session-id）
curl -s -H "X-Session-Id: phone01" \
  http://localhost:5000/api/v1/metrics/realtime | python -m json.tool

# 终端3: 限流测试（连续5次，第3次起应返回429）
for i in 1 2 3 4 5; do
  echo -n "请求$i: "
  curl -s -H "X-Session-Id: phone01" \
    http://localhost:5000/api/v1/metrics/realtime | python -c "import sys,json;d=json.load(sys.stdin);print(d['code'])"
done
# 预期输出: 请求1: 0, 请求2: 0, 请求3: -1, 请求4: -1, 请求5: -1

# 终端4: 数据库索引验证
sqlite3 rehab_data.db "SELECT name FROM sqlite_master WHERE type='index';"
# 预期: idx_gait_timestamp, idx_gait_session, idx_emotion_timestamp

# WebSocket 测试 (websocat)
websocat ws://localhost:5000/ws/metrics
# 预期: 每3秒收到 {"type":"metrics","data":{...}}
```

### 8.2 小程序测试

1. 微信开发者工具打开 `f:/Code/Anaconda/yolov26/WeXin_Server/`
2. 在 `app.js` 中修改 `API_BASE_URL` 为 PC 的局域网 IP
3. 验证首页加载实时指标、仪表盘动画、指标卡片颜色
4. 拍照页面测试压缩和锁定接口
5. 趋势页面测试 Canvas 折线图绘制和时间范围切换
6. 关闭 WiFi 测试离线缓存降级

---

## 十、修改前后对比

| 功能 | 原项目状态 | 实现后 |
|------|-----------|--------|
| 数据查看 | 仅 PC 端显示器 | 手机小程序实时查看 |
| 人脸锁定 | PC 键盘 't' 键 | 小程序拍照一键锁定 |
| 跌倒告警 | PC 屏幕显示 | 手机 WebSocket 主动推送 |
| 历史趋势 | 命令行 analyze_session.py | 小程序 Canvas 折线图 |
| 康复报告 | 控制台日志 | 小程序富文本 + 分享 |
| 多设备访问 | 不支持 | 局域网多手机同时查看 |
| API 版本化 | 无 | /api/v1/ 前缀 |
| 离线能力 | 无 | 网络断开时显示缓存数据 |
| 请求保护 | 无 | 限流 2次/秒/设备 |

---

## 十一、不做的内容

- ❌ 微信登录授权（局域网场景不需要）
- ❌ 多患者并发（当前单患者场景）
- ❌ PDF 生成（DeepSeek 返回文本，非真正 PDF）
- ❌ 用户注册/注销系统
- ❌ 云端部署（纯局域网使用）
- ❌ PWA / H5 版本（仅微信小程序）

---

## 十二、风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| WebSocket 断线 | 中 | 心跳 + 自动重连 + 离线缓存降级 |
| 跌倒告警重复推送 | 低 | 5s 冷却去重 |
| 限流误伤多设备 | 低 | session_id+IP 组合键 |
| Canvas 图表复杂 | 低 | 只做折线图，不画复杂图表 |
| 图片 base64 过大 | 低 | quality=60，100KB 上限 |
| 数据库查询慢 | 低 | 3个索引 + LIMIT 1000 |

---

**方案版本**: v2.1 最终版  
**输出方式**: 逐文件输出，每个文件完成后等待确认再继续
