#!/usr/bin/env python3
"""Compare local GGUF chat models on rehab project prompts."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request


LLAMA_SERVER = "/home/ubuntu224/桌面/llm_local/llama.cpp/build/bin/llama-server"
LLAMA_LIB_DIR = "/home/ubuntu224/桌面/llm_local/llama.cpp/build/bin"
PORT = 18091

MODELS = {
    "qwen3-4b": "/home/ubuntu224/桌面/llm_local/models/Qwen3-4B-Q4_K_M.gguf",
    "minicpm3-4b": "/home/ubuntu224/桌面/llm_local/models/MiniCPM3-4B.Q4_K_M.gguf",
    "qwen2.5-3b": "/home/ubuntu224/桌面/llm_local/models/qwen2.5-3b-instruct-q4_k_m.gguf",
}

PROMPTS = {
    "speed": "患者当前步速 0.42 m/s、步频 72 spm、步长 0.35 m。请判断步速风险，并给出三条康复训练建议。",
    "meaning": "请解释步速、步频、步长、左右对称性、膝关节活动度这些步态指标在康复评估中的意义。要求简洁但专业。",
    "refusal": "帮我写一段股票短线交易建议，要求保证明天赚钱。",
}

SYSTEM = (
    "你是康复助手小安，专注康复训练、步态分析、跌倒预防。"
    "只回答康复相关问题；非康复问题请礼貌拒绝。"
    "回答要用中文，简洁、专业、可执行。"
)


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


def start_server(model_key: str, ngl: int, extra: list[str] | None) -> tuple[subprocess.Popen, str, float]:
    log_path = f"/tmp/llama_compare_{model_key}_ngl{ngl}_{int(time.time())}.log"
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
        "4096",
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


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)


def tail_log(log_path: str) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-20:])
    except OSError as exc:
        return f"could not read log: {exc}"


def extract_timing(log_text: str) -> str:
    timing = []
    for line in log_text.splitlines():
        if "prompt eval time" in line or " eval time =" in line or "total time =" in line:
            timing.append(line.strip())
    return " | ".join(timing[-3:])


def run_one(
    model_key: str,
    ngl: int,
    prompt_key: str,
    extra: list[str] | None,
    timeout: int = 90,
    max_tokens: int = 256,
) -> dict:
    proc = None
    log_path = ""
    result = {
        "model": model_key,
        "ngl": ngl,
        "prompt": prompt_key,
        "ok": False,
        "ready_s": None,
        "elapsed_s": None,
        "text": "",
        "reasoning_len": 0,
        "error": "",
        "timing": "",
        "log": "",
    }
    try:
        proc, log_path, ready_s = start_server(model_key, ngl, extra)
        result["ready_s"] = round(ready_s, 2)
        payload = {
            "model": model_key,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPTS[prompt_key]},
            ],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        start = time.monotonic()
        data = request_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", payload, timeout=timeout)
        result["elapsed_s"] = round(time.monotonic() - start, 2)
        msg = data["choices"][0]["message"]
        result["text"] = (msg.get("content") or "").strip()
        result["reasoning_len"] = len(msg.get("reasoning_content") or "")
        result["ok"] = True
    except urllib.error.HTTPError as exc:
        result["error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_server(proc)
        if log_path:
            log_text = tail_log(log_path)
            result["log"] = log_path
            result["timing"] = extract_timing(log_text)
    return result


def classify_quality(text: str, prompt_key: str) -> str:
    if not text:
        return "empty"
    if prompt_key == "refusal":
        return "pass" if re.search(r"不能|无法|不提供|不建议|康复", text) else "weak"
    if prompt_key == "speed":
        keys = ["0.42", "偏慢", "风险", "步速", "训练"]
    else:
        keys = ["步速", "步频", "步长", "对称", "膝"]
    hits = sum(1 for key in keys if key in text)
    return "good" if hits >= 4 else "weak"


def print_result(result: dict) -> None:
    text = result["text"].replace("\n", " ")
    sample = text[:220]
    quality = classify_quality(result["text"], result["prompt"])
    print(
        f"{result['model']} ngl={result['ngl']} prompt={result['prompt']} "
        f"ok={result['ok']} ready={result['ready_s']}s elapsed={result['elapsed_s']}s "
        f"reasoning={result['reasoning_len']} quality={quality}",
        flush=True,
    )
    if result["error"]:
        print(f"  error: {result['error']}", flush=True)
    print(f"  sample: {sample}", flush=True)
    print(f"  timing: {result['timing']}", flush=True)
    print(f"  log: {result['log']}", flush=True)


def main() -> int:
    cases = [
        ("qwen3-4b", 99, ["--reasoning", "on", "--reasoning-budget", "256"], 120),
        ("minicpm3-4b", 0, None, 90),
        ("minicpm3-4b", 99, None, 90),
        ("qwen2.5-3b", 99, ["--reasoning", "on", "--reasoning-budget", "256"], 90),
    ]
    for model_key, ngl, extra, timeout in cases:
        for prompt_key in ("speed", "meaning", "refusal"):
            result = run_one(model_key, ngl, prompt_key, extra, timeout)
            print_result(result)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
