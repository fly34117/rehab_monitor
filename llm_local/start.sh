#!/bin/bash
# ============================================================
# LLM 本地服务启动脚本
# 模型: Qwen2.5-3B-Instruct (Q4_K_M, ~2.1 GB)
# API:  OpenAI 兼容, http://localhost:8080/v1
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="${SCRIPT_DIR}/llama.cpp/build/bin/llama-server"
MODEL_PATH="${SCRIPT_DIR}/models/qwen2.5-3b-instruct-q4_k_m.gguf"

# 检查文件是否存在
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "错误: llama-server 未找到 ($LLAMA_SERVER)"
    echo "请先编译 llama.cpp"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件未找到 ($MODEL_PATH)"
    echo "请先下载模型到 models/ 目录"
    exit 1
fi

echo "============================================"
echo "  启动本地 LLM 服务"
echo "  模型: Qwen2.5-3B-Instruct (Q4_K_M)"
echo "  地址: http://localhost:8088"
echo "  API:  http://localhost:8088/v1"
echo "============================================"
echo ""

# 启动 llama-server
# -t 6:   使用 6 个 CPU 线程（留给其他项目余量）
# -c 4096: 上下文窗口 4096 tokens
# -ngl 0: 纯 CPU 推理（不占用 GPU/NPU）
# --mlock: 锁定模型在内存中，防止 swap
exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8088 \
    -t 6 \
    -c 4096 \
    -ngl 0 \
    --mlock \
    --alias "qwen2.5-3b"
