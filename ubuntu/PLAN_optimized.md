# 异构计算康复监测系统 — 全面优化方案

## Context

当前项目 YOLO26n-pose INT8 固定跑 CPU (~27ms)，未利用 Intel Arrow Lake-U 的 NPU (AI Boost) 和 GPU (Arc iGPU)。需要实现 NPU 推理 + GPU 解码 + CPU 后处理的异构管线，用于比赛答辩展示。

## 硬件环境

- Intel Core Ultra 5 225U (14核), Intel Arc iGPU, Intel AI Boost NPU
- OpenVINO: CPU(FP32/INT8), GPU(FP32/FP16/INT8), NPU(FP16/INT8)
- GStreamer 1.20.3 已安装, VA-API 驱动已安装
- OpenCV pip wheel 不含 GStreamer 后端 (需单独视频解码管道)

## 新增文件结构

```
rehab_optimized/              # 新异构管线包, 不影响原项目
├── __init__.py
├── config.py                 # 扩展配置: 设备优先级, 模型映射, GStreamer
├── device_selector.py        # 硬件自动检测 + 优先级回退
├── gst_camera.py             # GStreamer + VA-API 硬件解码管道
├── model_loader.py           # 统一模型加载: 按设备自动选精度
├── pipeline.py               # HeterogeneousPipeline 核心
├── monitor.py                # 性能监控: FPS, 设备负载, 延迟
├── main.py                   # 新入口, 集成原有步态/跌倒/表情/API
└── utils.py                  # 工具函数
```

## 核心设计

### 1. 设备选择优先级 (device_selector.py)
NPU > GPU > CPU, 自动检测可用性:
- NPU: render 组权限 + /dev/accel/accel0 存在
- GPU: OpenVINO GPU 设备可用
- CPU: 始终可用

### 2. 视频解码 (gst_camera.py)
GStreamer 管道: `v4l2src → vaapipostproc → appsink`
- 硬件解码卸载到 iGPU VA-API
- appsink 丢帧策略 (max-buffers=1, drop=true)
- 独立线程, 通过 queue 传递 frames

### 3. 模型加载 (model_loader.py)
设备→精度自动映射:
- NPU: FP16 (YOLO11n/s-pose) — NPU 不支持 FP32
- GPU: FP16/INT8
- CPU: INT8

推荐默认: YOLO11s-pose (NPU 兼容性好, 无 Floor 问题)

### 4. 异构管线 (pipeline.py)
```
GStreamer解码(GPU) → [帧队列] → NPU推理(NPU) → 后处理(CPU)
                                    ↓
                              步态/跌倒/表情/API (CPU)
```
- 解码器和推理异步运行
- 推理结果带时间戳, 支持掉帧
- 动态设备切换 (模型重载)

### 5. 依赖安装
- `gstreamer1.0-vaapi` — VA-API GStreamer 插件
- `gstreamer1.0-plugins-bad` — 额外编解码器

## 验证方式

1. `python -m rehab_optimized.main` 启动, 观察 FPS 和 设备负载
2. 对比 CPU-only vs NPU 模式 FPS
3. GStreamer 管道 `gst-launch-1.0 v4l2src ! vaapipostproc ! fakesink` 验证硬件解码
