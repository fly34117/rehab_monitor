"""局域网 UDP 广播发现服务

后台线程每 3 秒向 255.255.255.255:5003 广播服务器地址，
供微信小程序同网段自动发现。
"""
import json
import socket
import time
import threading
import platform

from .logging_setup import get_logger

logger = get_logger("discovery")

BROADCAST_PORT = 5003
BROADCAST_INTERVAL = 3  # 秒
API_PORT = 5000
WS_PORT = 5001


def _get_lan_ip():
    """获取本机局域网 IP"""
    try:
        # 通过 UDP 连接探测获取实际出口 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("192.168.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    # 回退：遍历网卡
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return "127.0.0.1"


class DiscoveryService:
    """UDP 广播发现服务（后台线程）"""

    def __init__(self, api_port=API_PORT, ws_port=WS_PORT):
        self._api_port = api_port
        self._ws_port = ws_port
        self._running = False
        self._thread = None
        self._sock = None

    def start(self):
        """启动广播线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._broadcast_loop, daemon=True, name="discovery-svc"
        )
        self._thread.start()
        logger.info("UDP 发现服务已启动 (端口 %d, 间隔 %ds)", BROADCAST_PORT, BROADCAST_INTERVAL)

    def stop(self):
        """停止广播"""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        logger.info("UDP 发现服务已停止")

    def _broadcast_loop(self):
        ip = _get_lan_ip()
        hostname = platform.node()
        msg = json.dumps({
            "type": "rehab_server",
            "ip": ip,
            "port": self._api_port,
            "ws_port": self._ws_port,
            "hostname": hostname,
        }, ensure_ascii=False)

        while self._running:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                self._sock.settimeout(2)
                self._sock.sendto(msg.encode("utf-8"), ("255.255.255.255", BROADCAST_PORT))
                self._sock.close()
                self._sock = None
            except Exception as e:
                logger.debug("UDP 广播失败: %s", e)
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None

            # 等够间隔，但每 0.5s 检查一次 _running
            for _ in range(BROADCAST_INTERVAL * 2):
                if not self._running:
                    break
                time.sleep(0.5)


# 模块级单例
_discovery_service = None


def get_discovery_service():
    """获取或创建 DiscoveryService 单例"""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DiscoveryService()
    return _discovery_service


def start_discovery():
    """快捷启动"""
    get_discovery_service().start()


def stop_discovery():
    """快捷停止"""
    if _discovery_service:
        _discovery_service.stop()
