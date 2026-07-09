"""LLM 回答 + 对话组件 — 流式文本显示 + 输入框

底部 "AI 回答" Tab 的内容组件，支持：
- 流式逐字显示 LLM 输出（分析报告 / 对话）
- 底部输入框 + 发送按钮（对话模式）
- 对话历史显示（用户消息 + AI 回复）
- 状态提示
"""
import time
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QKeyEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLabel, QPushButton, QLineEdit,
)
from rehab_monitor.chat_store import chat_store


class LLMResponseWidget(QWidget):
    """LLM 回答展示组件 + 对话输入

    信号:
        message_sent(str) — 用户发送消息时触发
    """

    message_sent = pyqtSignal(str)

    STYLE_IDLE = "color: #969696; font-size: 12px;"
    STYLE_RUNNING = "color: #ce9178; font-size: 12px; font-weight: bold;"
    STYLE_DONE = "color: #4ec9b0; font-size: 12px; font-weight: bold;"
    STYLE_ERROR = "color: #f44747; font-size: 12px; font-weight: bold;"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_report = ""
        self._streaming = False
        self._thinking_dots = 0
        self._thinking_timer = QTimer(self)
        self._thinking_timer.timeout.connect(self._animate_thinking)

        # 流式 UI 节流 — 避免每 token 触发 setPlainText 轰炸主线程
        self._pending_text = ""
        self._pending_mode = ""  # "chat" 或 "report"
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._flush_pending)
        self._last_flush = 0.0
        self._THROTTLE_MS = 0.05  # 50ms → 最多 20 次 setPlainText/秒

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # === 顶部状态栏 ===
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        self._status_label = QLabel("状态: 就绪 — 等待触发分析")
        self._status_label.setStyleSheet(self.STYLE_IDLE)
        header_layout.addWidget(self._status_label)

        header_layout.addStretch()

        self._clear_btn = QPushButton("清除对话")
        self._clear_btn.setStyleSheet(
            "QPushButton { background-color: #3c3c3c; color: #cccccc; border: none; "
            "padding: 2px 8px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #555; }"
        )
        self._clear_btn.clicked.connect(self.clear_chat)
        header_layout.addWidget(self._clear_btn)

        layout.addLayout(header_layout)

        # === 文本显示区域 ===
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("WenQuanYi Micro Hei Mono", 11))
        self._text_edit.setStyleSheet(
            "QTextEdit {"
            "  background-color: #1e1e1e;"
            "  color: #d4d4d4;"
            "  border: 1px solid #3c3c3c;"
            "  border-radius: 4px;"
            "  padding: 6px;"
            "  selection-background-color: #264f78;"
            "}"
        )
        self._show_placeholder()
        layout.addWidget(self._text_edit, stretch=1)

        # === 输入区域 ===
        input_layout = QHBoxLayout()
        input_layout.setSpacing(6)

        self._input_field = QLineEdit()
        self._input_field.setPlaceholderText("输入问题，按 Enter 发送...")
        self._input_field.setFont(QFont("WenQuanYi Micro Hei", 11))
        self._input_field.setStyleSheet(
            "QLineEdit {"
            "  background-color: #2d2d2d;"
            "  color: #d4d4d4;"
            "  border: 1px solid #3c3c3c;"
            "  border-radius: 4px;"
            "  padding: 6px 10px;"
            "}"
            "QLineEdit:focus { border-color: #0e639c; }"
        )
        self._input_field.returnPressed.connect(self._on_send)
        input_layout.addWidget(self._input_field, stretch=1)

        self._send_btn = QPushButton("发送")
        self._send_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border: none; "
            "padding: 6px 16px; border-radius: 3px; font-size: 12px; }"
            "QPushButton:hover { background-color: #1177bb; }"
            "QPushButton:disabled { background-color: #3c3c3c; color: #666; }"
        )
        self._send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(self._send_btn)

        layout.addLayout(input_layout)

        # 提示
        hint = QLabel("提示: 点击侧边栏「📋 简单分析」生成报告；在输入框打字对话")
        hint.setStyleSheet("color: #6a6a6a; font-size: 10px;")
        layout.addWidget(hint)

    # ---- 发送消息 ----

    def _on_send(self):
        text = self._input_field.text().strip()
        if not text or self._streaming:
            return
        self._input_field.clear()
        self.message_sent.emit(text)

    # ---- 流式更新（带节流） ----

    def _schedule_update(self, text, mode):
        """节流：累积文本，每 50ms 最多刷新一次 setPlainText"""
        self._pending_text = text
        self._pending_mode = mode
        now = time.time()
        if now - self._last_flush >= self._THROTTLE_MS:
            self._flush_pending()
        elif not self._throttle_timer.isActive():
            self._throttle_timer.start(int(self._THROTTLE_MS * 1000))

    def _flush_pending(self):
        """执行实际的 setPlainText 刷新"""
        if not self._pending_text:
            return
        text = self._pending_text
        self._pending_text = ""

        if self._pending_mode == "report":
            self._do_update_partial(text)
        elif self._pending_mode == "chat":
            self._do_update_stream(text)

        self._last_flush = time.time()

    def _do_update_partial(self, text):
        """报告模式实际 setPlainText"""
        plain = self._text_edit.toPlainText()
        idx = plain.rfind(self.ANALYSIS_MARKER)
        if idx >= 0:
            before = plain[:idx + len(self.ANALYSIS_MARKER)]
            self._text_edit.setPlainText(before + text + "▊")
        else:
            self._text_edit.setPlainText(plain + text + "▊")
        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _do_update_stream(self, text):
        """对话模式实际 setPlainText"""
        plain = self._text_edit.toPlainText()
        marker = "🤖 小安:\n"
        idx = plain.rfind(marker)
        if idx >= 0:
            before = plain[:idx + len(marker)]
            self._text_edit.setPlainText(before + text + "▊")
        else:
            self._text_edit.setPlainText(plain + text + "▊")
        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---- 对话显示 ----

    def append_user_message(self, text):
        """添加用户消息到显示区"""
        plain = self._text_edit.toPlainText()
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)
        # 如果已有内容，确保前面有换行分隔
        if plain and not plain.rstrip().endswith("\n\n"):
            self._text_edit.insertPlainText("\n\n" if plain.rstrip() else "")
        self._text_edit.insertPlainText(f"🧑 你:\n{text}\n\n")
        # AI 回复占位
        self._text_edit.insertPlainText("🤖 小安:\n▊\n")
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)

    def update_stream_text(self, text):
        """流式更新 AI 回复 — 节流到 50ms/次"""
        self._streaming = True
        self._schedule_update(text, "chat")

    def finish_stream(self, final_text):
        """流式完成 — 立即刷新最终文本"""
        self._streaming = False
        self._pending_text = final_text
        self._pending_mode = "chat"
        self._flush_pending()
        # 移除 ▊
        plain = self._text_edit.toPlainText()
        marker = "🤖 小安:\n"
        idx = plain.rfind(marker)
        if idx >= 0:
            before = plain[:idx + len(marker)]
            clean = final_text.replace("▊", "")
            self._text_edit.setPlainText(before + clean)
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)

    # ---- 报告显示（分析模式，追加到对话区） ----

    ANALYSIS_MARKER = "📋 分析报告:\n"

    def start_analysis(self):
        """在对话区添加分析报告头部标记（不清空历史）"""
        # 如果已有内容，先加一个空行分隔
        plain = self._text_edit.toPlainText().rstrip()
        if plain and not plain.endswith("\n\n"):
            self._text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self._text_edit.insertPlainText("\n\n" if plain else "")
        # 插入报告标记
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self._text_edit.insertPlainText(self.ANALYSIS_MARKER + "▊")
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)

    def update_partial_text(self, text):
        """流式更新报告文本 — 节流到 50ms/次"""
        self._streaming = True
        self._schedule_update(text, "report")

    def finish_text(self, final_text):
        """完成流式报告 — 立即刷新"""
        self._streaming = False
        self._pending_text = final_text
        self._pending_mode = "report"
        self._flush_pending()
        # 移除 ▊
        plain = self._text_edit.toPlainText()
        idx = plain.rfind(self.ANALYSIS_MARKER)
        if idx >= 0:
            before = plain[:idx + len(self.ANALYSIS_MARKER)]
            clean = final_text.replace("▊", "")
            self._text_edit.setPlainText(before + clean)
        else:
            self._text_edit.setPlainText(final_text.replace("▊", ""))
        self._text_edit.moveCursor(QTextCursor.MoveOperation.End)

    # ---- 状态 ----

    def show_running(self, message="⏳ 正在分析..."):
        self._status_label.setText(f"状态: {message}")
        self._status_label.setStyleSheet(self.STYLE_RUNNING)
        self._send_btn.setEnabled(False)
        self._input_field.setEnabled(False)

    def show_thinking(self):
        """显示思考动画（旋转点）"""
        self._status_label.setText("状态: 🤔 正在思考")
        self._status_label.setStyleSheet(self.STYLE_RUNNING)
        self._send_btn.setEnabled(False)
        self._input_field.setEnabled(False)
        self._thinking_dots = 0
        self._thinking_timer.start(400)  # 每 400ms 更新

    def _animate_thinking(self):
        """思考动画帧"""
        self._thinking_dots = (self._thinking_dots + 1) % 4
        dots = "." * self._thinking_dots
        self._status_label.setText(f"状态: 🤔 正在思考{dots}")
        if self._thinking_dots == 0:
            self._status_label.setText(f"状态: 🤔 正在思考")

    def _stop_thinking(self):
        self._thinking_timer.stop()

    def show_done(self, message="✅ 完成"):
        self._stop_thinking()
        self._status_label.setText(f"状态: {message}")
        self._status_label.setStyleSheet(self.STYLE_DONE)
        self._send_btn.setEnabled(True)
        self._input_field.setEnabled(True)

    def show_error(self, message):
        self._stop_thinking()
        self._status_label.setText(f"状态: ❌ {message}")
        self._status_label.setStyleSheet(self.STYLE_ERROR)
        self._send_btn.setEnabled(True)
        self._input_field.setEnabled(True)

    def show_idle(self):
        self._stop_thinking()
        self._status_label.setText("状态: 就绪 — 等待触发分析")
        self._status_label.setStyleSheet(self.STYLE_IDLE)
        self._send_btn.setEnabled(True)
        self._input_field.setEnabled(True)

    # ---- 清除 ----

    def clear_chat(self):
        """清除对话历史和显示"""
        chat_store.clear()
        self._current_report = ""
        self._streaming = False
        self._text_edit.clear()
        self._show_placeholder()
        self.show_idle()

    def _show_placeholder(self):
        self._text_edit.setPlaceholderText(
            "点击右侧「📋 简单分析」按钮，生成本地 LLM 康复分析报告...\n\n"
            "或在下方输入框打字对话：\n"
            "  • 康复建议咨询\n"
            "  • 步态分析解读\n"
            "  • 跌倒风险评估\n\n"
            "模型: Qwen2.5-3B-Instruct (本地运行)"
        )

    def get_chat_history(self):
        """获取对话历史"""
        return chat_store.get_history()

    def add_to_history(self, role, content):
        """添加到对话历史"""
        chat_store.add_message(role, content)
