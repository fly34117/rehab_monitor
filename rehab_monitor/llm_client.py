"""本地 LLM 客户端 — 通过 llama-server HTTP API 通信

启动时自动拉起 llama-server 后台进程，所有推理通过 OpenAI 兼容 API。
完全避免 Python subprocess fork 死锁问题。
"""
import json
import os
import time
import subprocess
import urllib.request
import urllib.error
import threading
import re

from .logging_setup import get_logger

logger = get_logger("llm_client")

# ===== 路径 =====
LLAMA_DIR = os.path.expanduser("/home/ubuntu224/桌面/llm_local")
LLAMA_SERVER = os.path.join(LLAMA_DIR, "llama.cpp", "build", "bin", "llama-server")
MODEL_PATH = os.path.join(LLAMA_DIR, "models", "qwen2.5-3b-instruct-q4_k_m.gguf")
LLAMA_LIB_DIR = os.path.join(LLAMA_DIR, "llama.cpp", "build", "bin")

# ===== 服务器配置 =====
SERVER_PORT = 8088
SERVER_HOST = "127.0.0.1"
API_BASE = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
SERVER_STARTUP_TIMEOUT = 30  # 等待服务器就绪的超时

# ===== 默认参数 =====
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 180

# ===== 全局状态 =====
_server_process = None
_server_ready = False
_server_lock = threading.Lock()

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_REASONING_HEADER = "### 思考过程"
_ANSWER_HEADER = "### 最终回答"


def strip_reasoning_text(text):
    """Remove model reasoning/thinking traces from user-visible text."""
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)
    return text.replace("</think>", "").strip()


def format_reasoning_response(reasoning, content):
    """Format reasoning and final content as two visible sections."""
    reasoning = (reasoning or "").strip()
    content = (content or "").strip()
    if reasoning and content:
        return f"{_REASONING_HEADER}\n{reasoning}\n\n{_ANSWER_HEADER}\n{content}"
    if reasoning:
        return f"{_REASONING_HEADER}\n{reasoning}"
    return content


def extract_reasoning_text(text):
    """Extract the visible reasoning section from formatted model output."""
    if not text or _REASONING_HEADER not in text:
        return ""
    body = text.split(_REASONING_HEADER, 1)[1]
    if _ANSWER_HEADER in body:
        body = body.split(_ANSWER_HEADER, 1)[0]
    return body.strip()


def extract_final_answer(text):
    """Extract the final answer section, falling back to stripping inline think tags."""
    if not text:
        return ""
    if _ANSWER_HEADER in text:
        return text.split(_ANSWER_HEADER, 1)[1].strip()
    return strip_reasoning_text(text)


def _ensure_server(timeout=SERVER_STARTUP_TIMEOUT):
    """等待 llama-server 就绪。

    GUI 会先启动视觉管线，再后台加载 llama-server。用户在模型加载期间点
    AI 对话时，不应立即报错，而是短暂等待 /v1/models 可用。
    """
    deadline = time.time() + max(0, timeout)
    while True:
        if _check_health():
            return True
        if time.time() >= deadline:
            logger.error("llama-server 未就绪（等待超时）")
            return False
        time.sleep(0.3)


def _check_health():
    """检查 llama-server 是否就绪"""
    try:
        url = f"{API_BASE}/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return len(data.get("data", [])) > 0
    except Exception:
        return False


def _api_chat(messages, max_tokens=800, temperature=0.3, timeout=DEFAULT_TIMEOUT, on_token=None):
    """调用 llama-server API，支持 SSE 流式回调

    Args:
        messages: [{"role": ..., "content": ...}]
        on_token: 若提供，逐 token 回调 on_token(accumulated_text: str)
    """
    if not _ensure_server(timeout=min(15, timeout)):
        return None, "llama-server 未就绪"

    stream = on_token is not None
    model_name = os.environ.get("LLM_MODEL_ALIAS", "qwen3-4b")
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not stream:
                data = json.loads(resp.read().decode())
                msg = data["choices"][0]["message"]
                return format_reasoning_response(
                    msg.get("reasoning_content") or "",
                    msg.get("content") or "",
                ), None

            # SSE 流式读取
            reasoning_parts = []
            content_parts = []
            last_visible = ""
            for line_bytes in resp:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            reasoning_parts.append(reasoning)
                        if content:
                            content_parts.append(content)
                        if reasoning or content:
                            visible = format_reasoning_response(
                                "".join(reasoning_parts),
                                strip_reasoning_text("".join(content_parts)),
                            )
                            if visible != last_visible:
                                last_visible = visible
                                on_token(visible)
                    except json.JSONDecodeError:
                        pass
            return format_reasoning_response(
                "".join(reasoning_parts),
                strip_reasoning_text("".join(content_parts)),
            ), None

    except urllib.error.URLError as e:
        return None, f"llama-server 连接失败: {e.reason}"
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return None, f"llama-server HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return None, f"API 错误: {str(e)[:200]}"


def _run_llm(system_prompt, user_prompt, max_tokens=DEFAULT_MAX_TOKENS,
             temperature=DEFAULT_TEMPERATURE):
    """非流式推理"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _api_chat(messages, max_tokens, temperature)


def _run_llm_stream(system_prompt, user_prompt, on_chunk, max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=DEFAULT_TEMPERATURE, timeout=None):
    """真流式推理 — SSE 逐 token 回调"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response, error = _api_chat(messages, max_tokens, temperature,
                                timeout or DEFAULT_TIMEOUT, on_token=on_chunk)
    return response, error


def _run_llm_stream_raw(full_prompt, on_chunk, max_tokens=DEFAULT_MAX_TOKENS,
                        temperature=DEFAULT_TEMPERATURE, timeout=None):
    """真流式（对话用）"""
    messages = _parse_chatml(full_prompt)
    response, error = _api_chat(messages, max_tokens, temperature,
                                timeout or DEFAULT_TIMEOUT, on_token=on_chunk)
    return response, error


def _parse_chatml(text):
    """从 ChatML 格式文本提取 messages 列表"""
    messages = []
    parts = text.split("<|im_start|>")
    for part in parts:
        part = part.strip()
        if not part or part.startswith("<|im_end|>"):
            continue
        lines = part.split("\n", 1)
        role = lines[0].strip()
        content = lines[1].split("<|im_end|>")[0].strip() if len(lines) > 1 else ""
        if role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": content})
    return messages


def _parse_response(content):
    """解析 LLM 返回的 JSON"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    if "```json" in content:
        try:
            s = content.index("```json") + 7
            e = content.index("```", s)
            return json.loads(content[s:e].strip())
        except (ValueError, json.JSONDecodeError):
            pass
    try:
        s = content.index("{")
        e = content.rindex("}") + 1
        return json.loads(content[s:e])
    except (ValueError, json.JSONDecodeError):
        return {"raw": content}


def _build_data_prompt(recent_data):
    lines = []
    gs = recent_data.get("gait_summary", {})
    lines.append(f"时间窗口: {recent_data.get('duration', 30)}秒")
    lines.append(f"步态记录: {recent_data.get('gait_records', 0)}条")
    lines.append(f"跌倒事件: {recent_data.get('fall_events', 0)}次")
    lines.append("")
    if gs:
        lines.append("## 临床指标 (最新值)")
        for key, (label, unit) in {
            "gait_velocity_mps": ("步速", "m/s"), "stride_length_m": ("步长", "m"),
            "symmetry": ("对称性", ""), "cadence_spm": ("步频", "spm"),
            "left_knee_rom": ("左膝ROM", "°"), "right_knee_rom": ("右膝ROM", "°"),
            "foot_clearance_cm": ("足廓清", "cm"), "gait_rehab_score": ("GRS评分", "/100"),
            "trunk_sway_deg": ("躯干侧倾", "°"),
        }.items():
            val = gs.get(key)
            if val is not None:
                lines.append(f"  {label}: {val:.2f}{unit}" if isinstance(val, float) else f"  {label}: {val}{unit}")
    else:
        lines.append("(步态数据不足)")
    if recent_data.get("fall_events", 0) > 0:
        lines.append(f"\n⚠ 检测到 {recent_data['fall_events']} 次跌倒事件")
    return "\n".join(lines)


# ---- 公开接口 ----

def generate_report(recent_data, api_key=None):
    from .api_client import SYSTEM_PROMPT
    data_text = _build_data_prompt(recent_data)
    response, error = _run_llm(SYSTEM_PROMPT, f"请分析以下康复监测数据并生成报告:\n\n{data_text}")
    if error: return None, None, error
    return response, _parse_response(response), None


def generate_report_from_metrics(gait_metrics, fall_status="safe", fall_score=0.0):
    from .api_client import SYSTEM_PROMPT
    gs = {k: v for k, v in (gait_metrics or {}).items() if v is not None}
    recent_data = {
        "duration": 30, "snapshot_count": 0,
        "gait_records": gait_metrics.get("step_count", 0) if gait_metrics else 0,
        "fall_events": 1 if fall_status == "alert" else 0,
        "emotion_records": 0, "gait_summary": gs,
    }
    return generate_report(recent_data)


def generate_expert_report(gait_stats, trend_data=None, api_key=None):
    from .expert_report import ExpertReportGenerator, EXPERT_SYSTEM_PROMPT
    expert = ExpertReportGenerator()
    if gait_stats is None: return None, [], None, "步态数据为空"
    user_prompt, cited_papers = expert._build_expert_prompt(gait_stats, trend_data)
    if len(user_prompt) > 1200: user_prompt = user_prompt[:1200] + "\n... [截断]"
    response, error = _run_llm(EXPERT_SYSTEM_PROMPT, user_prompt, max_tokens=1500)
    if error: return None, cited_papers, None, error
    return expert._parse_response(response), cited_papers, response, None


def is_available():
    return _ensure_server(), None


def shutdown_server():
    """关闭 llama-server（退出时调用）"""
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        logger.info("llama-server 已关闭")
