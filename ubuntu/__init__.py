"""ubuntu — Linux/Ubuntu 适配层
==============================
隔离原则: 所有 Linux 适配代码都在此目录，绝不修改 rehab_monitor/ 任何文件。

模块:
  run.sh              — 启动脚本 (conda 激活 + 模型/设备选择 + 硬件解码)
  nmu_main.py          — 运行时入口 (monkey-patch config + 显示 + 腿纠正)
  leg_corrector.py     — 侧视步行腿关键点纠正 (摆动/支撑检测 + 骨长约束)
  hw_camera.py         — 硬件加速摄像头 (GStreamer VA-API / V4L2 MJPEG)
  setup_hw_decode.sh   — Intel 媒体驱动安装脚本 (Arrow Lake + Ubuntu 22.04)

使用方式:
  bash ubuntu/run.sh                      # 默认启动
  bash ubuntu/run.sh --model yolo11n --npu  # NPU 推理
  bash ubuntu/run.sh --hw-decode            # 启用硬件解码
  sudo bash ubuntu/setup_hw_decode.sh       # 安装硬件解码驱动
"""
