"""GUI 入口点 — 通过 WebSocket 连接后端 API"""
import sys
import os
import signal
import asyncio
import websockets
import json

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, QObject, pyqtSignal

from rehab_gui.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    LD_LIBRARY_PATH,
    ENV_VARS,
    WS_PORT,
)
from rehab_gui.core.config_manager import ConfigManager
from rehab_gui.gui.main_window import MainWindow
from rehab_monitor.logging_setup import get_logger

logger = get_logger("gui_main")


class WebSocketClient(QThread):
    """WebSocket 客户端 — 接收后端数据"""

    # 信号
    frame_received = pyqtSignal(dict)
    metrics_received = pyqtSignal(dict)
    fall_alert_received = pyqtSignal(dict)

    def __init__(self, host="localhost", port=5001):
        super().__init__()
        self._host = host
        self._port = port
        self._running = False

    def run(self):
        """WebSocket 连接循环"""
        import websockets
        uri = f"ws://{self._host}:{self._port}/ws/metrics"

        self._running = True
        while self._running:
            try:
                async def connect():
                    async with websockets.connect(uri) as ws:
                        logger.info(f"WebSocket 已连接: {uri}")
                        while self._running:
                            try:
                                msg = await ws.recv()
                                data = json.loads(msg)

                                if data.get("type") == "metrics":
                                    self.metrics_received.emit(data.get("data", {}))
                                elif data.get("type") == "fall_alert":
                                    self.fall_alert_received.emit(data.get("data", {}))

                            except Exception as e:
                                logger.warning(f"WebSocket 消息解析失败: {e}")
                                break

                asyncio.run(connect())
            except Exception as e:
                logger.error(f"WebSocket 连接失败: {e}")
                if self._running:
                    self.sleep(5000)  # 5秒后重连

    def stop(self):
        """停止连接"""
        self._running = False
        self.wait()


def _setup_environment():
    """设置环境变量"""
    os.environ["LD_LIBRARY_PATH"] = f"{LD_LIBRARY_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    for key, value in ENV_VARS.items():
        os.environ[key] = value


def _load_theme(app):
    """加载深色主题"""
    theme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gui",
        "dark_theme.qss",
    )
    try:
        with open(theme_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
        logger.info("深色主题已加载")
    except FileNotFoundError:
        logger.warning("主题文件未找到")


def main():
    """主入口函数"""
    _setup_environment()
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # 创建 QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Rehab Monitor")
    app.setOrganizationName("Rehab Monitor")
    from PyQt6.QtGui import QFont
    app.setFont(QFont("WenQuanYi Micro Hei", 13))

    # 加载主题
    _load_theme(app)

    # 检查后端是否运行
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:5000/api/v1/health", timeout=2)
        logger.info("后端 API 已连接")
    except:
        logger.warning("后端 API 未连接，请先启动: bash ubuntu/run.sh")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("警告")
        msg.setText("后端 API 未连接")
        msg.setInformativeText("请先启动后端服务:\n  bash ubuntu/run.sh")
        msg.exec()
        sys.exit(1)

    # 创建主窗口
    config_manager = ConfigManager()
    config = config_manager.load()

    window = MainWindow(config=config)
    window.show()

    # 创建 WebSocket 客户端
    ws_client = WebSocketClient()
    ws_client.metrics_received.connect(lambda data: window._on_metrics_update(data))
    ws_client.fall_alert_received.connect(
        lambda data: window._on_fall_update({
            "status": "alert",
            "score": data.get("score"),
            "source": f"WebSocket (severity={data.get('severity', 'medium')})",
        })
    )
    ws_client.start()

    logger.info("GUI 已启动 (WebSocket 模式)")

    signal.signal(signal.SIGINT, lambda *args: (ws_client.stop(), app.quit()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
