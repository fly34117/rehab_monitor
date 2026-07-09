"""局域网 mDNS 服务发现

通过 zeroconf 注册 _rehab._tcp 服务，小程序用
wx.startLocalServiceDiscovery 自动发现服务端 IP 和端口。
"""
import socket
import platform
import threading

from .logging_setup import get_logger

logger = get_logger("discovery")

API_PORT = 5000


def get_lan_ip():
    """获取本机局域网真实 IP — 走默认路由的网卡，排除虚拟网卡"""
    import subprocess
    import re
    try:
        # 方法1: 查默认路由对应的接口名，然后取它的 IP（最准确）
        route = subprocess.check_output(
            ["ip", "route", "show", "default"], text=True
        ).strip()
        m = re.search(r"dev\s+(\S+)", route)
        if m:
            iface = m.group(1)
            addr_out = subprocess.check_output(
                ["ip", "-4", "addr", "show", iface], text=True
            )
            m2 = re.search(r"inet\s+([\d.]+)", addr_out)
            if m2:
                return m2.group(1)
    except Exception:
        pass
    try:
        # 方法2: hostname -I, 过滤掉虚拟网段
        out = subprocess.check_output(["hostname", "-I"], text=True).strip()
        ips = out.split()
        for ip in ips:
            if ip.startswith("192.168.") and not ip.startswith("192.168.56."):
                return ip
        for ip in ips:
            if ip.startswith("10."):
                return ip
        if ips:
            return ips[0]
    except Exception:
        pass
    return "127.0.0.1"


# 内部别名（向后兼容）
_get_lan_ip = get_lan_ip


class DiscoveryService:
    """mDNS 服务注册（后台线程）"""

    def __init__(self, api_port=API_PORT):
        self._api_port = api_port
        self._running = False
        self._thread = None
        self._zc = None
        self._info = None

    def start(self):
        """启动 mDNS 注册"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._register_loop, daemon=True, name="mdns-svc"
        )
        self._thread.start()

    def stop(self):
        """停止 mDNS 注册"""
        self._running = False
        try:
            if self._info and self._zc:
                self._zc.unregister_service(self._info)
        except Exception:
            pass
        try:
            if self._zc:
                self._zc.close()
        except Exception:
            pass
        self._zc = None
        self._info = None
        logger.info("mDNS 发现服务已停止")

    def _register_loop(self):
        try:
            from zeroconf import Zeroconf, ServiceInfo
        except ImportError:
            logger.warning("zeroconf 未安装，mDNS 发现不可用: pip install zeroconf")
            return

        ip = _get_lan_ip()
        hostname = platform.node()
        props = {
            "hostname": hostname,
            "port": str(self._api_port),
            "type": "rehab_server",
        }

        self._zc = Zeroconf()
        self._info = ServiceInfo(
            type_="_rehab._tcp.local.",
            name=f"{hostname}._rehab._tcp.local.",
            addresses=[socket.inet_aton(ip)],
            port=self._api_port,
            properties=props,
        )

        try:
            self._zc.register_service(self._info, allow_name_change=True)
            logger.info("mDNS 发现服务已注册: %s (%s:%d)", hostname, ip, self._api_port)
        except Exception as e:
            logger.warning("mDNS 注册失败: %s", e)
            self._zc.close()
            self._zc = None
            return

        # 保持线程存活，直到 stop() 被调用
        while self._running:
            import time
            time.sleep(1)


# 模块级单例
_discovery_service = None


def start_discovery():
    """快捷启动"""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DiscoveryService()
    _discovery_service.start()


def stop_discovery():
    """快捷停止"""
    if _discovery_service:
        _discovery_service.stop()
