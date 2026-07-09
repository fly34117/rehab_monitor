"""LLM 推理工作线程 — 后台流式推理 + 对话模式

通过 QThread 将 llama-cli 调用放到后台执行，
支持流式输出（逐字显示）和多轮对话。
"""
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from rehab_monitor.logging_setup import get_logger

logger = get_logger("llm_worker")


class LLMWorker(QThread):
    """后台 LLM 推理线程（报告分析，流式输出）"""

    # 部分响应 (chunk: str) — 流式显示用
    partial_response = pyqtSignal(str)
    # 完整响应 (raw_text: str, json_report: dict)
    response_ready = pyqtSignal(str, dict)
    # 错误 (error_message: str)
    error_occurred = pyqtSignal(str)
    # 状态更新 (status_text: str)
    progress_update = pyqtSignal(str)

    def __init__(self, report_type="simple", gait_metrics=None,
                 fall_status="safe", fall_score=0.0):
        super().__init__()
        self.report_type = report_type
        self.gait_metrics = gait_metrics or {}
        self.fall_status = fall_status
        self.fall_score = fall_score
        self._full_text = []

    def run(self):
        try:
            self.progress_update.emit("thinking")  # 触发思考动画

            from rehab_monitor.llm_client import _run_llm_stream
            from rehab_monitor.api_client import SYSTEM_PROMPT, _build_data_prompt

            gs = {}
            if self.gait_metrics:
                for k, v in self.gait_metrics.items():
                    if v is not None:
                        gs[k] = v

            recent_data = {
                "duration": 30, "snapshot_count": 0,
                "gait_records": self.gait_metrics.get("step_count", 0),
                "fall_events": 1 if self.fall_status == "alert" else 0,
                "emotion_records": 0, "gait_summary": gs,
            }
            data_text = _build_data_prompt(recent_data)
            user_prompt = f"请分析以下康复监测数据并生成报告:\n\n{data_text}"

            max_tokens = 1500 if self.report_type == "expert" else 800

            def on_chunk(accumulated):
                # SSE 回调直接给累积文本
                self.partial_response.emit(accumulated)

            response, error = _run_llm_stream(
                SYSTEM_PROMPT, user_prompt, on_chunk,
                max_tokens=max_tokens,
            )

            if error and not response:
                self.error_occurred.emit(error)
            else:
                full_text = response or ""
                from rehab_monitor.llm_client import _parse_response, extract_final_answer
                report_json = _parse_response(extract_final_answer(full_text))
                self.response_ready.emit(full_text, report_json)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"LLM 工作线程异常: {e}\n{tb}")
            self.error_occurred.emit(f"LLM 推理异常: {str(e)[:200]}")


class LLMChatWorker(QThread):
    """后台 LLM 对话线程（多轮对话，流式输出）"""

    # 部分响应 (chunk: str)
    partial_response = pyqtSignal(str)
    # 完整响应 (response_text: str)
    response_ready = pyqtSignal(str)
    # 错误
    error_occurred = pyqtSignal(str)
    # 状态更新
    progress_update = pyqtSignal(str)

    # ── 模型差异化配置 ──
    # 小模型需要详细约束防跑偏，大模型精简提示词节省 token
    MODEL_CONFIGS = {
        "qwen2.5-3b": {
            "temperature": 0.2,
            "system": (
                "你是康复助手小安。你必须严格遵守以下规则：\n"
                "1. 你只能输出助手回复内容，绝对不能生成用户对话、角色标记（user/assistant/system）、或 XML 标签\n"
                '2. 只回答康复、步态分析、跌倒预防相关问题，其他问题回复「请咨询康复相关问题」\n'
                "3. 回答简洁专业，用中文，不超过200字\n"
                '4. 不确定就说「我暂时无法回答这个问题」，不要编造'
            ),
        },
        "qwen3-4b": {
            "temperature": 0.3,
            "system": (
                "你是康复助手小安，专注康复训练、步态分析、跌倒预防。\n"
                "用中文简洁专业回答。非康复问题礼貌拒绝。不要生成用户对话。"
            ),
            "max_tokens": 500,
            # 注: Qwen3 是思考模型，当前 llama.cpp 版本偶发超时，建议优先用 7B
        },
        "qwen2.5-7b": {
            "temperature": 0.4,
            "system": (
                "你是康复助手小安。用中文简洁专业地回答康复相关问题。"
                "非康复问题请礼貌拒绝。"
            ),
            "max_tokens": 800,
        },
    }
    # fallback 配置（未知模型）
    _DEFAULT_CONFIG = {
        "temperature": 0.3,
        "system": (
            "你是康复助手小安。用中文简洁专业地回答康复相关问题。"
        ),
    }

    @classmethod
    def get_model_config(cls, model_key):
        """获取模型对应的提示词和温度"""
        return cls.MODEL_CONFIGS.get(model_key, cls._DEFAULT_CONFIG)

    def __init__(self, messages=None, system_prompt=None, model_key=None):
        """初始化对话线程

        Args:
            messages: 对话历史 [{"role": "user"|"assistant", "content": "..."}, ...]
            system_prompt: 系统提示（None 使用模型默认）
            model_key: LLM 模型 key，用于选择温度（None 使用默认）
        """
        super().__init__()
        self.messages = messages or []
        cfg = self.get_model_config(model_key) if model_key else self._DEFAULT_CONFIG
        self.system_prompt = system_prompt or cfg["system"]
        self._temperature = cfg["temperature"]
        self._max_tokens = cfg.get("max_tokens", 600)
        self._model_key = model_key  # 用于选择推理路径
        self._full_text = []

    def run(self):
        try:
            self.progress_update.emit("thinking")
            self._full_text = []

            # Use chat/completions for all configured chat models. For Qwen3,
            # llama-server should use the GGUF embedded template so reasoning is
            # returned separately instead of being mixed into raw text.
            from rehab_monitor.llm_client import _run_llm_stream_raw, _strip_role_prefix

            parts = [f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"]
            for msg in self.messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            full_prompt = "".join(parts)

            def on_chunk(accumulated):
                # Qwen3 模型偶发以 role token 开头，流式时也过滤
                self.partial_response.emit(_strip_role_prefix(accumulated))

            response, error = _run_llm_stream_raw(
                full_prompt, on_chunk, max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            if error and not response:
                self.error_occurred.emit(error)
            else:
                self.response_ready.emit(response or "")

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"对话线程异常: {e}\n{tb}")
            self.error_occurred.emit(f"对话异常: {str(e)[:200]}")
