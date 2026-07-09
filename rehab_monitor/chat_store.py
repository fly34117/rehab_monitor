"""共享对话历史存储 — 线程安全

GUI 的 LLMResponseWidget 和 Flask API 共用此模块，
确保对话记录在 GUI 和小程序间保持一致。
"""
import threading
from typing import List, Dict, Optional


class ChatStore:
    """线程安全的对话历史存储（模块级单例）"""

    def __init__(self):
        self._messages: List[Dict[str, str]] = []
        self._lock = threading.Lock()

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息到对话历史"""
        with self._lock:
            self._messages.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """返回对话历史副本"""
        with self._lock:
            return list(self._messages)

    def clear(self) -> int:
        """清除对话历史，返回清除的条数"""
        with self._lock:
            count = len(self._messages)
            self._messages.clear()
            return count

    def get_message_count(self) -> int:
        """获取当前消息数"""
        with self._lock:
            return len(self._messages)


# 模块级单例 — 整个进程共享同一份数据
chat_store = ChatStore()
