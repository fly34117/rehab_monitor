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
                max_tokens=max_tokens
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

    # 默认系统提示
    DEFAULT_SYSTEM = (
        "你是康复助手小安，专注于康复训练、步态分析、跌倒预防等健康话题。"
        "回答简洁专业、通俗易懂，用中文回复。"
    )

    def __init__(self, messages=None, system_prompt=None):
        """初始化对话线程

        Args:
            messages: 对话历史 [{"role": "user"|"assistant", "content": "..."}, ...]
            system_prompt: 系统提示（None 使用默认）
        """
        super().__init__()
        self.messages = messages or []
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self._full_text = []

    def run(self):
        try:
            self.progress_update.emit("thinking")
            self._full_text = []

            from rehab_monitor.llm_client import _run_llm_stream_raw

            parts = [f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"]
            for msg in self.messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            full_prompt = "".join(parts)

            def on_chunk(accumulated):
                self.partial_response.emit(accumulated)

            response, error = _run_llm_stream_raw(
                full_prompt, on_chunk, max_tokens=1024, temperature=0.7
            )

            if error and not response:
                self.error_occurred.emit(error)
            else:
                self.response_ready.emit(response or "")

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"对话线程异常: {e}\n{tb}")
            self.error_occurred.emit(f"对话异常: {str(e)[:200]}")
