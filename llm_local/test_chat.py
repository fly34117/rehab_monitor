#!/usr/bin/env python3
"""
测试本地 LLM 服务的对话功能
使用 OpenAI 兼容 API, 默认地址 http://localhost:8080/v1
"""

import sys
import json
import urllib.request
import urllib.error


API_BASE = "http://localhost:8088/v1"
MODEL_NAME = "qwen2.5-3b"  # 与服务端 --alias 一致


def chat(message: str, stream: bool = True) -> str:
    """发送对话请求并返回回复"""
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": stream,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        if stream:
            return _handle_stream(req)
        else:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        print(f"\n❌ 连接失败: {e.reason}")
        print(f"   请确认 llama-server 已启动: ./start.sh")
        sys.exit(1)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"\n❌ API 错误 ({e.code}): {body}")
        sys.exit(1)


def _handle_stream(req) -> str:
    """处理流式响应"""
    full_reply = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_reply.append(content)
                except json.JSONDecodeError:
                    pass
    print()
    return "".join(full_reply)


def check_health() -> bool:
    """检查服务是否就绪"""
    try:
        url = f"{API_BASE}/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", [])]
            print(f"✅ 服务就绪, 可用模型: {models}")
            return True
    except Exception as e:
        print(f"❌ 服务未就绪: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  本地 LLM 对话测试")
    print("  模型: Qwen2.5-3B-Instruct")
    print("=" * 50)
    print()

    # 检查服务
    if not check_health():
        sys.exit(1)

    print()

    # 测试对话
    prompts = [
        "你好，请用一句话介绍你自己。",
        "请用中文写一首关于康复训练的五言绝句。",
        "1+1等于几？请直接回答。",
    ]

    if len(sys.argv) > 1:
        # 自定义提问
        prompts = [" ".join(sys.argv[1:])]

    for i, p in enumerate(prompts, 1):
        if len(prompts) > 1:
            print(f"--- 测试 {i}: {p}")
        else:
            print(f"💬 提问: {p}")
        print()
        reply = chat(p)
        if len(prompts) > 1:
            print()

    print("✅ 测试完成！")
