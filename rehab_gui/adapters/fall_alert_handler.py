"""跌倒告警处理 — Qt 弹窗替代 subprocess"""
import time
from functools import wraps

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel

from rehab_monitor.logging_setup import get_logger

logger = get_logger("fall_alert_handler")


class FallAlertHandler:
    """跌倒告警处理器 — 替代 ubuntu/fall_popup.py 的 subprocess 方式

    使用 Qt 对话框显示跌倒告警, 支持自动消音
    """

    ALERT_COOLDOWN = 3.0  # 冷却时间 (秒)

    def __init__(self, parent=None):
        """初始化跌倒告警处理器

        Args:
            parent: 父窗口
        """
        self._parent = parent
        self._last_alert_time = 0.0
        self._current_dialog = None
        self._auto_close_timer = None

        logger.info("跌倒告警处理器已初始化")

    def inject(self):
        """注入跌倒告警系统

        替换 rehab_monitor.api_server.broadcast_fall_alert
        """
        try:
            from rehab_monitor import api_server

            self._original_broadcast = api_server.broadcast_fall_alert

            @wraps(self._original_broadcast)
            def patched_broadcast(location=None, fall_score=None):
                """替换的广播函数"""
                # 调用原始广播 (发送给 WebSocket 客户端)
                try:
                    self._original_broadcast(location, fall_score)
                except Exception:
                    pass

                # 显示 Qt 告警
                self.show_alert(location, fall_score)

            api_server.broadcast_fall_alert = patched_broadcast
            logger.info("跌倒告警注入成功")
        except Exception as e:
            logger.error(f"跌倒告警注入失败: {e}")

    def show_alert(self, location=None, fall_score=None):
        """显示跌倒告警

        Args:
            location: 跌倒位置信息
            fall_score: 跌倒分数
        """
        now = time.time()
        # 冷却检查
        if now - self._last_alert_time < self.ALERT_COOLDOWN:
            return
        self._last_alert_time = now

        # 关闭旧对话框
        if self._current_dialog and self._current_dialog.isVisible():
            self._current_dialog.close()

        # 创建告警对话框
        dialog = QDialog(self._parent)
        dialog.setWindowTitle("⚠️ 跌倒告警")
        dialog.setWindowFlags(
            dialog.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
        )
        dialog.setModal(False)
        dialog.setMinimumSize(320, 160)

        # 布局
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        # 标题
        title = QLabel("⚠️ FALL DETECTED")
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #f44747;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 详情
        details = []
        if fall_score is not None:
            details.append(f"跌倒分数: {fall_score:.0f}")
        if location:
            details.append(f"位置: {location}")
        details.append(f"时间: {time.strftime('%H:%M:%S')}")

        detail_text = QLabel("\n".join(details))
        detail_text.setStyleSheet("font-size: 14px; color: #d4d4d4;")
        detail_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(detail_text)

        # 自动关闭
        self._auto_close_timer = QTimer(dialog)
        self._auto_close_timer.timeout.connect(dialog.close)
        self._auto_close_timer.start(8000)  # 8秒后自动关闭

        self._current_dialog = dialog
        dialog.show()

        logger.warning(f"跌倒告警已显示 (分数: {fall_score}, 位置: {location})")

    def clear_alert(self):
        """清除当前告警"""
        if self._current_dialog and self._current_dialog.isVisible():
            self._current_dialog.close()
