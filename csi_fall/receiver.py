"""CSI 数据接收 — TCP 套接字 + 420字节二进制包解析
从 ESP32 设备接收原始 CSI 数据包，解析头部和 I/Q 采样数据。
"""
from __future__ import annotations
import socket
import struct

from rehab_monitor.logging_setup import get_logger

from .config import CSI_HOST, CSI_PORT, CSI_CONNECT_TIMEOUT, CSI_RECV_TIMEOUT, CSI_PACKET_SIZE

logger = get_logger("csi.receiver")


class CSIReceiver:
    """CSI 数据接收器

    原理:
      - 通过 TCP 连接到 ESP32 (默认 192.168.2.10:8000)
      - 接收 420 字节二进制数据包
      - 解析包头 (1字节天线序号 + 419字节 CSI 采样数据)
      - 在后台线程中持续接收

    Args:
        host: ESP32 IP 地址
        port: TCP 端口

    Usage:
        receiver = CSIReceiver()
        receiver.connect()
        # 在单独线程中运行:
        receiver.receive_loop(callback)
    """

    def __init__(self, host: str = "", port: int = 0):
        self.host = host or CSI_HOST
        self.port = port or CSI_PORT
        self._socket: socket.socket | None = None
        self._connected: bool = False
        self._running: bool = False
        self._recv_buffer: bytes = b""
        self._packet_count: int = 0

    @property
    def connected(self) -> bool:
        """TCP 连接状态"""
        return self._connected

    @property
    def packet_count(self) -> int:
        """已接收的包数量"""
        return self._packet_count

    def connect(self) -> bool:
        """建立 TCP 连接

        Returns:
            True 连接成功, False 连接失败
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(CSI_CONNECT_TIMEOUT)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(CSI_RECV_TIMEOUT)
            self._connected = True
            logger.info("CSI TCP 已连接 %s:%d", self.host, self.port)
            return True
        except socket.timeout:
            logger.error("CSI TCP 连接超时 %s:%d (%.1fs)",
                         self.host, self.port, CSI_CONNECT_TIMEOUT)
            return False
        except ConnectionRefusedError:
            logger.error("CSI TCP 连接被拒绝 %s:%d (ESP32 是否在线?)",
                         self.host, self.port)
            return False
        except OSError as e:
            logger.error("CSI TCP 连接失败 %s:%d: %s", self.host, self.port, e)
            return False

    def disconnect(self):
        """断开 TCP 连接"""
        self._connected = False
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        self._recv_buffer = b""
        logger.info("CSI TCP 已断开 (共接收 %d 包)", self._packet_count)

    def receive_loop(self, callback: callable):
        """数据接收主循环 (阻塞, 应在后台 daemon 线程中运行)

        Args:
            callback: callable(raw_data_dict) — 每收到一个有效包时调用
        """
        self._running = True
        logger.info("CSI 接收循环开始")

        while self._running and self._connected:
            try:
                chunk = self._socket.recv(4096)
                if not chunk:
                    logger.warning("CSI TCP 连接已关闭 (对端断开)")
                    break
                self._recv_buffer += chunk

                while len(self._recv_buffer) >= CSI_PACKET_SIZE:
                    packet = self._recv_buffer[:CSI_PACKET_SIZE]
                    self._recv_buffer = self._recv_buffer[CSI_PACKET_SIZE:]
                    parsed = self._parse_packet(packet)
                    if parsed is not None:
                        self._packet_count += 1
                        callback(parsed)

            except socket.timeout:
                continue
            except OSError as e:
                logger.error("CSI 接收错误: %s", e)
                break

        logger.info("CSI 接收循环结束, 共接收 %d 包, 缓冲区残留 %d 字节",
                    self._packet_count, len(self._recv_buffer))

    def _parse_packet(self, data: bytes) -> dict | None:
        """解析 420 字节 CSI 数据包

        Packet layout (ESP32-S3):
          [0]       sequence_num (天线索引 0-7)
          [1:420]   csi_data:
            [0:4]     timestamp (uint32 LE, 微秒)
            [4:10]    mac (6 bytes)
            [10:16]   dmac (6 bytes)
            [16]      rssi (int8, dBm)
            [17]      noise_floor (int8)
            [18]      rate (uint8)
            [19]      sgi (uint8)
            [20]      sig_mode (uint8)
            [21]      mcs (uint8)
            [22]      cwb (uint8)
            [23]      channel (uint8)
            [24]      secondary_channel (uint8)
            [25]      rx_state (uint8)
            [26]      fft_gain (uint8)
            [27]      agc_gain (uint8)
            [28:32]   gain (float32 LE)
            [32]      first_word_invalid (uint8)
            [33:35]   csi_len (uint16 LE, 应为 384)
            [35:419]  csi_buffer (384 bytes, I/Q 交错 int8)

        Args:
            data: 420 字节原始二进制数据

        Returns:
            解析后的字典, 或 None (解析失败)
        """
        try:
            sequence_num = data[0]
            csi_data = data[1:420]

            timestamp = struct.unpack('<I', csi_data[0:4])[0]
            mac = csi_data[4:10]
            dmac = csi_data[10:16]
            rssi = struct.unpack('<b', csi_data[16:17])[0]
            noise_floor = struct.unpack('<b', csi_data[17:18])[0]
            rate = csi_data[18]
            sgi = csi_data[19]
            sig_mode = csi_data[20]
            mcs = csi_data[21]
            cwb = csi_data[22]
            channel = csi_data[23]
            secondary_channel = csi_data[24]
            rx_state = csi_data[25]
            fft_gain = csi_data[26]
            agc_gain = csi_data[27]
            gain = struct.unpack('<f', csi_data[28:32])[0]
            first_word_invalid = csi_data[32]
            csi_len = struct.unpack('<H', csi_data[33:35])[0]
            csi_buffer = csi_data[35:35 + 384]

            return {
                'antenna_index': sequence_num,
                'timestamp': timestamp,
                'rssi': rssi,
                'noise_floor': noise_floor,
                'mac': ':'.join(f'{b:02x}' for b in mac),
                'dmac': ':'.join(f'{b:02x}' for b in dmac),
                'rate': rate,
                'sgi': sgi,
                'sig_mode': sig_mode,
                'mcs': mcs,
                'cwb': cwb,
                'channel': channel,
                'secondary_channel': -1 if secondary_channel == 2 else secondary_channel,
                'rx_state': rx_state,
                'fft_gain': fft_gain,
                'agc_gain': agc_gain,
                'gain': gain,
                'first_word_invalid': first_word_invalid,
                'csi_len': csi_len,
                'csi_data': [b if b < 128 else b - 256 for b in csi_buffer],
            }
        except struct.error as e:
            logger.error("CSI 包解析错误 (结构体): %s", e)
            return None
        except IndexError as e:
            logger.error("CSI 包解析错误 (越界): %s", e)
            return None
        except Exception as e:
            logger.error("CSI 包解析错误: %s", e)
            return None
