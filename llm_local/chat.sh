#!/bin/bash
# ============================================================
# 命令行直接对话（不需要启动服务器，不用端口）
# 模型: Qwen2.5-3B-Instruct (Q4_K_M)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CLI="${SCRIPT_DIR}/llama.cpp/build/bin/llama-cli"
MODEL_PATH="${SCRIPT_DIR}/models/qwen2.5-3b-instruct-q4_k_m.gguf"

if [ ! -f "$LLAMA_CLI" ]; then
    echo "错误: llama-cli 未找到"
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件未找到"
    exit 1
fi

echo "============================================"
echo "  Qwen2.5-3B 命令行对话"
echo "  输入消息后回车，Ctrl+C 退出"
echo "============================================"
# 内存安全检查：可用内存低于 1.5 GB 时拒绝启动
AVAIL_MEM=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$AVAIL_MEM" -lt 1500 ]; then
    echo "警告: 可用内存仅 ${AVAIL_MEM} MB，不足 1.5 GB，拒绝启动以防系统卡死"
    exit 1
fi

echo ""

exec nice -n 10 "$LLAMA_CLI" \
    -m "$MODEL_PATH" \
    -t 4 \
    -c 2048 \
    -ngl 0 \
    --mlock \
    --conversation \
    --no-display-prompt
