# Rehab Monitor — 三模态智能康复监测系统

[![GitHub stars](https://img.shields.io/github/stars/fly34117/rehab_monitor?style=flat-square&logo=github)](https://github.com/fly34117/rehab_monitor)
[![GitHub license](https://img.shields.io/github/license/fly34117/rehab_monitor?style=flat-square)](https://github.com/fly34117/rehab_monitor/blob/main/LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.4-orange?style=flat-square)](https://docs.openvino.ai/)
[![Gitee mirror](https://img.shields.io/badge/Gitee-镜像-red?style=flat-square&logo=gitee)](https://gitee.com/fiy-placid/rehab_monitor)

> **GitHub（主仓库）**: [github.com/fly34117/rehab_monitor](https://github.com/fly34117/rehab_monitor)  
> **Gitee（国内镜像）**: [gitee.com/fiy-placid/rehab_monitor](https://gitee.com/fiy-placid/rehab_monitor)

基于 YOLOv26-Pose + OpenVINO 的实时三模态智慧康复监测系统，集**步态分析**、**跌倒检测**、**情绪识别**于一体，运行于纯 CPU（Intel i5-12500H），目标帧率 20-25 FPS。

## 功能特性

- **姿态估计** — YOLOv26-Pose（nano INT8，38ms/帧）+ 卡尔曼滤波多人追踪
- **步态分析** — 14 项临床指标：步速、步长、步宽、对称性、膝关节 ROM、足廓清、躯干侧倾、双支撑比率等
- **跌倒检测** — 三规则加权投票（宽高比 + 髋部高度 + 肩线倾角）+ 躺下回溯分类（区分跌倒/主动躺下）
- **情绪识别** — 面部 ROI（COCO关键点定位）+ 7 类情绪分类 + 体态情绪后备（躺下时人脸不可靠）
- **人脸锁定** — FaceNet 嵌入 + HSV 人体外观双线索匹配，多人场景下自动追踪目标
- **空间定位** — 单应矩阵/针孔模型将像素坐标映射到真实地面 XY（米），支持 4 点校准
- **AI 报告** — DeepSeek API 每 30 秒生成周期康复摘要
- **SQLite 存储** — WAL 模式 + 异步写入，6 表完整记录
- **诊断窗口** — 关节角度监视器、轨迹俯瞰、骨架预览、步态指标仪表盘
- **离线分析** — `analyze_session.py` 生成 matplotlib 多面板报告图表

## 环境要求

| 项目 | 说明 |
|------|------|
| OS | Windows 10/11, Linux (macOS 未测试) |
| Python | 3.11+ |
| CPU | Intel Core i5-12500H 或更高（OpenVINO 加速） |
| 内存 | ≥ 8 GB RAM |
| 摄像头 | USB 或内置摄像头（640×480） |

## 快速开始

### 1. 克隆仓库

```bash
# GitHub（国际）
git clone https://github.com/fly34117/rehab_monitor.git
cd rehab_monitor

# Gitee（国内镜像，速度更快）
git clone https://gitee.com/fiy-placid/rehab_monitor.git
cd rehab_monitor
```

### 2. 创建并激活环境

```bash
conda create -n yolov26 python=3.11 -y
conda activate yolov26
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载姿态模型

从 [Ultralytics YOLO](https://docs.ultralytics.com/) 下载 `yolo26n-pose.pt`，放到 `model/` 目录。

或使用 YOLO CLI 直接导出：

```bash
# 下载并导出为 OpenVINO INT8 格式（可选，PyTorch 也可直接运行）
yolo export model=yolo26n-pose.pt format=openvino int8=True
```

> 参考：本项目的 `config.py` 默认使用 `model/yolo26n-pose_int8_openvino_model/`。

### 5. 运行

```bash
python -m rehab_monitor.main
```

## 配置

所有参数集中在 `rehab_monitor/config.py`，无需命令行参数。

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 摄像头 | `CAMERA_ID` | 0 | 首选摄像头 ID |
| 摄像头 | `FRAME_WIDTH` / `FRAME_HEIGHT` | 640 / 480 | 分辨率 |
| 推理 | `POSE_DEVICE` | `"cpu"` | cpu / gpu / npu |
| 推理 | `USE_KALMAN` | `True` | 卡尔曼滤波开关 |
| 跌倒 | `FALL_CONFIRM_FRAMES` | 5 | 连续确认帧数（防误报） |
| 跌倒 | `FALL_TILT_DEG` | 45.0 | 肩线倾角阈值 |
| API | `DEEPSEEK_API_KEY` | 环境变量 | DeepSeek API 密钥 |
| API | `REPORT_INTERVAL` | 30.0 | 报告生成间隔（秒） |
| 空间 | `SPATIAL_CAMERA_HEIGHT_M` | 1.5 | 摄像头距地面高度 |
| 空间 | `SPATIAL_CALIB_POINTS` | `[]` | 4 点校准坐标 |
| 表情 | `EMOTION_INTERVAL` | 5 | 每 N 帧跑一次表情 |

设置 DeepSeek API 密钥（可选，不设置则不生成报告）：

```bash
export DEEPSEEK_API_KEY="sk-your-key-here"      # Linux/macOS
set DEEPSEEK_API_KEY=sk-your-key-here            # Windows
```

## 运行时快捷键

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 截图保存 |
| `f` | 切换 FPS 显示 |
| `h` | 切换帮助覆盖层 |
| `SPACE` | 暂停 / 恢复 |
| `r` | 重置卡尔曼追踪 |
| `t` | 录入目标人脸（多人场景下锁定） |

## 输出指标

### 步态指标（每 5 帧记录）

| 指标 | 单位 | 说明 |
|------|------|------|
| Gait Velocity | m/s | 行走速度 |
| Stride Length | m | 跨步长度 |
| Step Length | m | 步长（分左右） |
| Step Width | m | 步宽 |
| Cadence | steps/min | 步频 |
| Symmetry | 0-1 | 左右对称性 |
| Stance Percentage | % | 支撑相占比 |
| Double Support Ratio | — | 双支撑帧占比 |
| Knee ROM (L/R) | ° | 膝关节活动范围 |
| Step Time CV | % | 步时变异系数 |
| Step Length CV | % | 步长变异系数 |
| Foot Clearance | cm | 足廓清高度 |
| Trunk Sway | ° | 躯干侧倾角 |
| GRS Score | 0-100 | 综合康复评分 |

### 跌倒状态

- **safe** — 正常站立/行走
- **alert** — 检测到跌倒（画面顶部红色告警栏 + SQLite 记录）

### 情绪类别

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

## 临床参考值

| 指标 | 正常范围 | 需关注 |
|------|----------|--------|
| 步速 | > 1.0 m/s | < 0.6 m/s |
| 对称性 | > 0.9 | < 0.7 |
| 步频 | 100-130 spm | < 80 spm |
| 双支撑比 | 20-30% | > 40% |
| 躯干侧倾 | < 5° | > 15° |
| 膝关节 ROM | 50-65°（行走时） | < 30° |
| 足廓清 | 1.5-2.5 cm | < 0.5 cm |
| 步时 CV | < 5% | > 15% |
| GRS 评分 | > 70（良好） | < 40（需重点关注） |

## 辅助工具

```bash
# 人脸录入（多人场景下用于锁定目标）
python enroll_face.py              # 交互式采集 5 张照片
python enroll_face.py --list       # 列出已录入人脸
python enroll_face.py --delete 0   # 删除指定条目

# 离线数据可视化
python analyze_session.py           # 分析最近一次会话
python analyze_session.py --session 5   # 指定会话 ID
python analyze_session.py --all          # 跨会话趋势对比

# 训练自定义情绪模型
python train_emotion.py             # 面部情绪（FER2013）
python train_body_emotion.py        # 体态情绪（关键点）
```

## 项目结构

```
rehab_monitor/
├── config.py              # 集中配置
├── main.py                # 主循环 + 线程协调
├── pose_detector.py       # OpenVINO 姿态检测 + 卡尔曼匹配
├── kalman_filter.py       # 卡尔曼滤波核心
├── gait_analyzer.py       # 14 项临床指标
├── fall_detector.py       # 跌倒规则引擎
├── emotion_recognizer.py  # 表情识别（人脸ROI + 分类 + 投票）
├── face_locker.py         # FaceNet 人脸锁定（嵌入 + 人体外观）
├── spatial_mapper.py      # 像素→世界坐标映射
├── data_logger.py         # SQLite 异步写入
├── api_client.py          # DeepSeek API 报告
├── display_overlay.py     # 主画面 UI 叠加
├── angle_monitor.py       # 关节角度监视窗
├── trajectory_monitor.py  # 轨迹俯瞰窗
├── skeleton_viewer.py     # 骨架预览窗
├── gait_metrics_viewer.py # 步态指标仪表盘
└── logging_setup.py       # 统一日志系统
```

## 许可证

[MIT License](LICENSE)

Copyright (c) 2025 fly34117

---

<p align="center">
  <a href="https://github.com/fly34117/rehab_monitor">GitHub</a> ·
  <a href="https://gitee.com/fiy-placid/rehab_monitor">Gitee 镜像</a>
</p>
