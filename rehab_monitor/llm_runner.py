#!/usr/bin/env python3
"""LLM 推理辅助进程 — 独立于父进程运行，避免 subprocess 挂起

用法:
    python llm_runner.py <prompt_file>

从 stdin 读取 JSON 配置，输出结果到 stdout。
与父进程完全隔离，不受 logging 模块或其他状态的影响。
"""
import sys
import os
import json
import subprocess
import time

LLAMA_DIR = os.path.expanduser("/home/ubuntu224/桌面/llm_local")
LLAMA_CLI = os.path.join(LLAMA_DIR, "llama.cpp", "build", "bin", "llama-cli")
MODEL_PATH = os.path.join(LLAMA_DIR, "models", "qwen2.5-3b-instruct-q4_k_m.gguf")
LLAMA_LIB_DIR = os.path.join(LLAMA_DIR, "llama.cpp", "build", "bin")


def run_once(full_prompt, max_tokens=800, temperature=0.3, timeout=180):
    """单次推理，返回 (stdout_text, returncode)"""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-t", "4",
        "-c", "2048",
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--mlock",
        "--no-display-prompt",
        "-st",
        "-p", full_prompt,
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return None, proc.returncode, "TIMEOUT"

    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    return stdout, proc.returncode, None


def filter_response(raw_output):
    """提取纯回复文本"""
    import re
    lines = raw_output.split("\n")
    result = []
    in_banner = True

    for line in lines:
        while "\b" in line:
            line = re.sub(r'[^\b]\b', '', line, count=1)
        stripped = line.strip()

        if not stripped:
            if not in_banner:
                result.append("")
            continue

        if in_banner:
            if any(kw in stripped for kw in [
                "▄", "█", "▀",
                "build ", "model ", "ftype ", "modalities ",
                "available commands", "/exit", "/regen", "/clear", "/read", "/glob",
            ]) or stripped.startswith("build "):
                continue
            in_banner = False

        if stripped.startswith("[ Prompt:") or stripped.startswith("[ Generation:") or stripped == "Exiting...":
            continue

        if stripped.startswith("> ") and "<|im_start|>" in stripped:
            continue

        result.append(stripped)

    return "\n".join(result).strip()


def main():
    """主入口：从 stdin 读取 JSON，执行推理，输出 JSON 到 stdout"""
    try:
        input_data = json.loads(sys.stdin.read())
    except Exception as e:
        print(json.dumps({"error": f"Invalid input: {e}"}))
        sys.exit(1)

    prompt = input_data.get("prompt", "")
    max_tokens = input_data.get("max_tokens", 800)
    temperature = input_data.get("temperature", 0.3)
    timeout = input_data.get("timeout", 180)

    if not prompt:
        print(json.dumps({"error": "Empty prompt"}))
        sys.exit(1)

    stdout, rc, timeout_err = run_once(prompt, max_tokens, temperature, timeout)

    if timeout_err:
        print(json.dumps({"error": timeout_err}))
        sys.exit(1)

    if rc != 0:
        print(json.dumps({"error": f"llama-cli exited with code {rc}", "raw": stdout}))
        sys.exit(1)

    filtered = filter_response(stdout) if stdout else ""

    print(json.dumps({"response": filtered, "raw": stdout}))


if __name__ == "__main__":
    main()
