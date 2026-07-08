#!/usr/bin/env python3
"""Small llama-server diagnostic for Qwen3/Qwen2.5 backend behavior."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


LLAMA_SERVER = "/home/ubuntu224/桌面/llm_local/llama.cpp/build/bin/llama-server"
LLAMA_LIB_DIR = "/home/ubuntu224/桌面/llm_local/llama.cpp/build/bin"
MODELS = {
    "qwen2.5-3b": "/home/ubuntu224/桌面/llm_local/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    "qwen3-4b": "/home/ubuntu224/桌面/llm_local/models/Qwen3-4B-Q4_K_M.gguf",
}
PORT = 18089


def request_json(url: str, payload: dict | None = None, timeout: int = 5) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready() -> float:
    start = time.monotonic()
    deadline = start + 90
    while time.monotonic() < deadline:
        try:
            request_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            return time.monotonic() - start
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("server did not become ready")


def run_server(model_key: str, ngl: int, extra: list[str] | None = None) -> tuple[subprocess.Popen, str, float]:
    log_path = f"/tmp/llama_diag_{model_key.replace('.', '_')}_ngl{ngl}_{int(time.time())}.log"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LLAMA_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        LLAMA_SERVER,
        "-m",
        MODELS[model_key],
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "-t",
        "4",
        "-c",
        "2048",
        "-ngl",
        str(ngl),
        "--alias",
        model_key,
    ]
    if extra:
        cmd.extend(extra)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    log_file.close()
    ready_s = wait_ready()
    return proc, log_path, ready_s


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)


def run_case(name: str, model_key: str, ngl: int, endpoint: str, payload: dict, extra: list[str] | None = None) -> None:
    print(f"\n=== {name} ===", flush=True)
    proc = None
    log_path = ""
    try:
        proc, log_path, ready_s = run_server(model_key, ngl, extra)
        print(f"ready_s={ready_s:.2f} log={log_path}", flush=True)
        start = time.monotonic()
        data = request_json(f"http://127.0.0.1:{PORT}/v1/{endpoint}", payload, timeout=75)
        elapsed = time.monotonic() - start
        choice = data["choices"][0]
        text = choice.get("text") or choice.get("message", {}).get("content", "")
        reasoning = choice.get("message", {}).get("reasoning_content", "")
        print(f"ok elapsed_s={elapsed:.2f} text_len={len(text)} reasoning_len={len(reasoning)}", flush=True)
        print(f"sample={text[:160].replace(chr(10), ' ')}", flush=True)
    except urllib.error.HTTPError as exc:
        print(f"http_error code={exc.code} body={exc.read().decode('utf-8', errors='replace')[:400]}", flush=True)
    except Exception as exc:
        print(f"error type={type(exc).__name__} msg={exc}", flush=True)
    finally:
        if proc is not None:
            stop_server(proc)
        if log_path:
            print("log_tail:", flush=True)
            try:
                with open(log_path, encoding="utf-8", errors="replace") as handle:
                    lines = handle.readlines()[-18:]
                for line in lines:
                    print(line.rstrip(), flush=True)
            except OSError as exc:
                print(f"could not read log: {exc}", flush=True)


def main() -> int:
    chat_payload = {
        "model": "qwen3-4b",
        "messages": [
            {"role": "system", "content": "你是康复助手。回答要简短。"},
            {"role": "user", "content": "膝关节康复训练要注意什么？用三点回答。"},
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 96,
    }
    raw_prompt = (
        "<|im_start|>system\n你是康复助手。回答要简短。<|im_end|>\n"
        "<|im_start|>user\n膝关节康复训练要注意什么？用三点回答。<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    raw_payload = {
        "model": "qwen3-4b",
        "prompt": raw_prompt,
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 96,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }
    qwen25_payload = dict(chat_payload)
    qwen25_payload["model"] = "qwen2.5-3b"

    run_case("qwen3-4b CPU raw completions", "qwen3-4b", 0, "completions", raw_payload, ["--chat-template", "qwen3"])
    run_case("qwen3-4b Vulkan raw completions", "qwen3-4b", 99, "completions", raw_payload, ["--chat-template", "qwen3"])
    run_case("qwen3-4b Vulkan chat completions", "qwen3-4b", 99, "chat/completions", chat_payload, ["--chat-template", "qwen3"])
    run_case(
        "qwen3-4b Vulkan chat completions reasoning-on",
        "qwen3-4b",
        99,
        "chat/completions",
        chat_payload,
        ["--chat-template", "qwen3", "--reasoning", "on", "--reasoning-budget", "64"],
    )
    raw_payload_long = dict(raw_payload)
    raw_payload_long["max_tokens"] = 256
    run_case(
        "qwen3-4b Vulkan raw completions 256 tokens",
        "qwen3-4b",
        99,
        "completions",
        raw_payload_long,
        ["--chat-template", "qwen3", "--reasoning", "on", "--reasoning-budget", "64"],
    )
    run_case("qwen2.5-3b Vulkan chat completions", "qwen2.5-3b", 99, "chat/completions", qwen25_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
