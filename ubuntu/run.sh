#!/bin/bash
# ============================================================
# 康复监测系统 — Ubuntu/Linux 启动脚本
# 隔离于 ubuntu/ 目录，不影响原始 Windows 项目文件
#
# 用法:
#   ./run.sh                              YOLO26n CPU INT8 (默认)
#   ./run.sh --model yolo11n --npu        YOLO11n NPU (最快)
#   ./run.sh --model yolo11s --cpu        YOLO11s CPU INT8
#   ./run.sh --gpu                        YOLO26n GPU
#   ./run.sh --hw-decode                  启用硬件解码 (GStreamer VA-API / MJPEG)
#   ./run.sh --optimized                  三阶段异构管线 (解码→iGPU, 推理→iGPU, 后处理→CPU)
#   ./run.sh --optimized --gpu            三阶段 + 强制 GPU 推理
#   ./run.sh --original                   原始配置 (不改模型/设备)
#
# 性能参考 (Intel Core Ultra 5 225U):
#   YOLO11n NPU  FP32 : 25.4ms  YOLO11n CPU INT8 : 27.5ms
#   YOLO11s NPU  FP32 : 28.9ms  YOLO11s CPU INT8 : 40.8ms
#   YOLO26n CPU  INT8 : 26.5ms  YOLO26n GPU FP32 : 24.1ms
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- Conda 环境 ----
MCD="/home/ubuntu224/miniconda3"
PY="$MCD/envs/yolov26/bin/python"

source "$MCD/etc/profile.d/conda.sh"
conda activate yolov26 2>/dev/null || {
    echo "[错误] 无法激活 conda 环境 'yolov26'"
    exit 1
}

# ---- 确保额外依赖 ----
$PY -c 'import flask, flask_cors, websockets' 2>/dev/null || {
    echo "[提示] 安装依赖..."
    $PY -m pip install flask flask-cors websockets -q
}

# ---- 动态库路径 ----
OV_LIBS="$MCD/envs/yolov26/lib/python3.11/site-packages/openvino/libs"
export LD_LIBRARY_PATH="$OV_LIBS:${LD_LIBRARY_PATH}"

# ---- 解析参数 ----
MODEL="yolo26n"
DEVICE="cpu"     # cpu / gpu / npu
ORIGINAL=false
HW_DECODE=false
OPTIMIZED=false
MAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        yolo11n|yolo11s|yolo26n|yolo26s|yolo26m)
            MODEL="$1"
            shift
            ;;
        --cpu)  DEVICE="cpu"; shift ;;
        --gpu)  DEVICE="gpu"; shift ;;
        --npu)  DEVICE="npu"; shift ;;
        --hw-decode)  HW_DECODE=true; shift ;;
        --optimized)  OPTIMIZED=true; shift ;;
        --original)   ORIGINAL=true; shift ;;
        *)      MAIN_ARGS+=("$1"); shift ;;
    esac
done

# ---- NPU 权限 ----
if [ "$DEVICE" = "npu" ]; then
    if ! groups | grep -q render; then
        echo "[提示] 不在 render 组，NPU 不可用，回退到 CPU"
        echo "       修复: sudo usermod -a -G render \$USER && newgrp render"
        DEVICE="cpu"
    else
        [ -c /dev/accel/accel0 ] && sudo chown root:render /dev/accel/accel0 2>/dev/null || true
        [ -c /dev/accel/accel0 ] && sudo chmod 660 /dev/accel/accel0 2>/dev/null || true
    fi
fi

# ============================================================
# 硬件优化 — Intel Core Ultra 5 225U (14核 Arrow Lake-U)
# ============================================================
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export KMP_DUPLICATE_LIB_OK=TRUE
export OV_CPU_INFERENCE_NUM_THREADS=8
export OV_CPU_NUM_STREAMS=2
export OV_CPU_ENABLE_CPU_PINNING=NO
export OV_CPU_HYPER_THREADING=YES

# ---- 硬件解码自动检测 ----
if $HW_DECODE; then
    # 检查 VA-API 是否可用
    if [ -n "$LIBVA_DRIVER_NAME" ] || vainfo &>/dev/null 2>&1; then
        export REHAB_HW_DECODE=1
        echo "[硬件解码] 已启用 (GStreamer VA-API / V4L2 MJPEG)"
    elif [ -c /dev/dri/renderD128 ] && groups "$USER" | grep -q render; then
        # GPU 节点和权限都存在, 可能缺驱动
        export REHAB_HW_DECODE=1
        echo "[硬件解码] 尝试启用 (如失败将自动回退 V4L2 MJPEG)"
        echo "           提示: 运行 ubuntu/setup_hw_decode.sh --check 诊断"
    else
        echo "[硬件解码] 未检测到硬件解码能力, 使用 CPU 解码"
        echo "           提示: sudo bash ubuntu/setup_hw_decode.sh 安装驱动"
    fi
elif [ -z "${REHAB_HW_DECODE:-}" ] && vainfo &>/dev/null 2>&1; then
    # 未显式指定 --hw-decode, 但 VA-API 可用 → 自动启用
    export REHAB_HW_DECODE=1
    echo "[硬件解码] 自动检测到 VA-API, 已启用"
fi

# ---- 启动信息 ----
PYVER=$($PY -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?')
OVVER=$($PY -c 'import openvino; print(openvino.__version__.split("-")[0])' 2>/dev/null || echo N/A)
OVDEV=$($PY -c 'import openvino as ov; print(", ".join(ov.Core().available_devices))' 2>/dev/null || echo N/A)

echo "============================================"
echo "  康复监测系统 — Ubuntu/Linux"
PIPELINE_MODE="monkey-patch"
$OPTIMIZED && PIPELINE_MODE="三阶段异构"
$ORIGINAL && PIPELINE_MODE="原始"
echo "  管线  : $PIPELINE_MODE"
echo "  模型  : ${MODEL^^}-pose | 设备: $DEVICE"
echo "  解码  : ${REHAB_HW_DECODE:+硬件加速}${REHAB_HW_DECODE:-CPU}"
echo "  HW    : $OVDEV"
echo "  Conda : yolov26 | Python $PYVER | OV $OVVER"
echo "============================================"
echo ""

# ---- 启动 ----
if $OPTIMIZED; then
    # 三阶段异构管线: Stage1 解码→iGPU, Stage2 推理→iGPU/NPU, Stage3 后处理→CPU
    OPT_ARGS=(
        --model "$MODEL"
        --camera 0
    )
    [ "$DEVICE" = "gpu" ] && OPT_ARGS+=(--gpu)
    [ "$DEVICE" = "npu" ] && OPT_ARGS+=(--npu)
    [ "$DEVICE" = "cpu" ] && OPT_ARGS+=(--cpu)
    [ "${REHAB_HW_DECODE:-}" = "1" ] && OPT_ARGS+=(--gstreamer)
    echo "[启动] rehab_optimized (三阶段异构管线)"
    exec $PY -m rehab_optimized.main "${OPT_ARGS[@]}" "${MAIN_ARGS[@]}"
elif $ORIGINAL; then
    exec $PY -m rehab_monitor.main "${MAIN_ARGS[@]}"
else
    export REHAB_MODEL="$MODEL"
    export REHAB_DEVICE="$DEVICE"
    exec $PY "$SCRIPT_DIR/npu_main.py" "${MAIN_ARGS[@]}"
fi
