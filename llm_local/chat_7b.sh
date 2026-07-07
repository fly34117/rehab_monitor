#!/bin/bash
# ============================================================
# 命令行直接对话 - Qwen2.5-7B (IQ3_M, ~3.6 GB)
# 比 3B 强一档，通用能力更好
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CLI="${SCRIPT_DIR}/llama.cpp/build/bin/llama-cli"
MODEL_PATH="${SCRIPT_DIR}/models/Qwen2.5-7B-Instruct-IQ3_M.gguf"

if [ ! -f "$LLAMA_CLI" ]; then
    echo "错误: llama-cli 未找到"
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件未找到 ($MODEL_PATH)"
    exit 1
fi

echo "============================================"
echo "  Qwen2.5-7B 命令行对话 (IQ3_M)"
echo "  输入消息后回车，Ctrl+C 退出"
echo "============================================"
echo ""

exec "$LLAMA_CLI" \
    -m "$MODEL_PATH" \
    -t 4 \
    -c 3072 \
    -ngl 0 \
    --mlock \
    --conversation \
    --no-display-prompt
