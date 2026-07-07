#!/bin/bash
# LLM 推理 Shell 包装器 — 完全绕过 Python fork 死锁问题
# 用法: llm_runner.sh <input_json_file> <output_file>

LLAMA_DIR="/home/ubuntu224/桌面/llm_local"
LLAMA_CLI="$LLAMA_DIR/llama.cpp/build/bin/llama-cli"
MODEL="$LLAMA_DIR/models/qwen2.5-3b-instruct-q4_k_m.gguf"
export LD_LIBRARY_PATH="$LLAMA_DIR/llama.cpp/build/bin:$LD_LIBRARY_PATH"

INPUT="$1"
OUTPUT="$2"
PROMPT=$(python3 -c "import json,sys; print(json.load(open('$INPUT'))['prompt'])" 2>/dev/null)
MAX_TOKENS=$(python3 -c "import json,sys; print(json.load(open('$INPUT')).get('max_tokens',800))" 2>/dev/null)
TEMP=$(python3 -c "import json,sys; print(json.load(open('$INPUT')).get('temperature',0.3))" 2>/dev/null)
TIMEOUT=$(python3 -c "import json,sys; print(json.load(open('$INPUT')).get('timeout',180))" 2>/dev/null)

if [ -z "$PROMPT" ]; then
    echo '{"error":"Empty prompt"}' > "$OUTPUT"
    exit 1
fi

timeout "${TIMEOUT}" "$LLAMA_CLI" \
    -m "$MODEL" -t 4 -c 2048 \
    -n "${MAX_TOKENS:-800}" --temp "${TEMP:-0.3}" \
    --mlock --no-display-prompt -st \
    -p "$PROMPT" > "$OUTPUT" 2>/dev/null

RC=$?
if [ $RC -eq 124 ]; then
    echo '{"error":"TIMEOUT"}' > "$OUTPUT"
    exit 1
elif [ $RC -ne 0 ]; then
    echo "{\"error\":\"llama-cli exit $RC\"}" > "$OUTPUT"
    exit 1
fi

# 过滤输出，提取纯回复
FILTERED=$(python3 -c "
import sys, re
text = open('$OUTPUT').read()
lines = text.split('\n')
result = []
skip = True
for line in lines:
    while '\b' in line:
        line = re.sub(r'[^\b]\b', '', line, count=1)
    s = line.strip()
    if not s:
        if not skip: result.append('')
        continue
    if skip:
        if any(k in s for k in ['▄','█','▀','build ','model ','ftype ','modalities ','available commands','/exit','/regen']):
            continue
        skip = False
    if s.startswith('[ Prompt:') or s.startswith('[ Generation:') or s == 'Exiting...':
        continue
    if s.startswith('> ') and '<|im_start|>' in s:
        continue
    result.append(s)
print(json.dumps({'response': '\n'.join(result).strip()}))
" 2>/dev/null)

echo "$FILTERED" > "$OUTPUT"
