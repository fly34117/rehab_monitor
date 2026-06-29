"""CSI 数据处理 — I/Q 重建 + 子载波重排 + 帧拼装 + 算法链
将原始 int8 I/Q 采样重建为 complex64 CSI, 按粗时间戳拼装 8 天线完整帧,
然后依次运行注册的算法链。
"""
from __future__ import annotations

import numpy as np

from rehab_monitor.logging_setup import get_logger

from .config import (
    ANTENNA_MAPPING, ANTENNA_ORDER, EXPECTED_ANTENNAS,
    CSI_WIFI_PROTOCOL, CSI_CACHE_MAX,
)

logger = get_logger("csi.processor")


class CSIProcessor:
    """CSI 数据处理器

    原理:
      - 将 int8 I/Q 采样重建为 complex64 CSI
      - 根据 WiFi 协议重排子载波 (LLTF: 64 subcarriers, HT40: 128)
      - 按粗时间戳 (10ms 桶) 将 8 天线数据拼装为一帧 full_data
      - 运行算法链: full_data 依次被每个注册的算法 mutate

    注入 full_data 的键 (每帧):
      'antenna_order'     : list[str], 天线 ID 有序列表
      '<antenna_id>'      : dict, 每个天线的处理后数据
        'csi_complex'     : np.ndarray (complex64)
        'csi_magnitude'   : np.ndarray (float64)
        'csi_phase'       : np.ndarray (float64)
        'rssi'            : int
        'timestamp'       : int
        'subcarriers_nums': int

    Args:
        wifi_protocol: "LLTF" (64 子载波) 或 "HT40" (128 子载波)
    """

    TIMESTAMP_SCALE = 10000  # 10ms 粗时间窗

    def __init__(self, wifi_protocol: str = ""):
        self.wifi_protocol = wifi_protocol or CSI_WIFI_PROTOCOL
        self.antenna_mapping = ANTENNA_MAPPING
        self.antenna_order = list(ANTENNA_ORDER)
        self.expected_antennas = EXPECTED_ANTENNAS

        self._cache: dict[int, list[dict]] = {}  # coarse_ts → list of processed dicts
        self._algorithms: dict[str, object] = {}   # name → algorithm instance
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # 算法注册
    # ------------------------------------------------------------------

    def register_algorithm(self, name: str, instance: object):
        """注册算法模块

        算法必须实现 apply(full_data, antenna_order) -> full_data 接口。
        可选实现 clear() 用于重置状态。

        算法按注册顺序执行，顺序很重要 (RSSI → 去噪 → 跌倒检测)。

        Args:
            name: 算法名称 (用于日志)
            instance: 算法实例
        """
        self._algorithms[name] = instance
        logger.info("CSI 算法已注册: %s (%s)", name, type(instance).__name__)

    @property
    def frame_count(self) -> int:
        """已完成的完整帧数"""
        return self._frame_count

    # ------------------------------------------------------------------
    # 核心处理
    # ------------------------------------------------------------------

    def process(self, raw_data: dict) -> dict | None:
        """处理单个原始数据包, 当 8 天线集齐时返回完整帧

        Args:
            raw_data: 来自 CSIReceiver._parse_packet() 的字典,
                      包含 antenna_index, timestamp, csi_data (list[int8]) 等

        Returns:
            完整帧 full_data dict (8 天线凑齐并运行算法链后),
            或 None (当前帧尚未集齐)
        """
        # 1. 天线索引 → 天线 ID
        antenna_index = raw_data['antenna_index']
        antenna_id = self.antenna_mapping.get(antenna_index)
        if antenna_id is None:
            logger.warning("未知天线索引: %d", antenna_index)
            return None

        # 2. 粗时间戳分桶 (10ms)
        coarse_ts = (raw_data['timestamp'] + self.TIMESTAMP_SCALE // 2) // self.TIMESTAMP_SCALE

        # 3. I/Q 重建: int8 交错 → complex64
        csi_int8 = np.array(raw_data['csi_data'], dtype=np.int8)
        # odd-index = real, even-index = imaginary
        csi_complex = (csi_int8[1::2] + 1j * csi_int8[0::2]).astype(np.complex64)

        # 4. 子载波重排 (将 DC 移到中心)
        if self.wifi_protocol == 'HT40':
            # 128 子载波模式: [64:128] + [128:192]
            part1 = csi_complex[64:128]
            part2 = csi_complex[128:192]
            csi_complex = np.concatenate([part2, part1])
        else:
            # LLTF 模式 (默认): [0:32] + [32:64] 交换 → [32:64] + [0:32]
            part1 = csi_complex[0:32]
            part2 = csi_complex[32:64]
            csi_complex = np.concatenate([part2, part1])

        # 5. 组装单天线数据
        processed = {
            'antenna_id': antenna_id,
            'timestamp': raw_data['timestamp'],
            'coarse_timestamp': coarse_ts,
            'rssi': raw_data.get('rssi', 0),
            'noise_floor': raw_data.get('noise_floor', 0),
            'gain': raw_data.get('gain', 1.0),
            'csi_complex': csi_complex,
            'csi_magnitude': np.abs(csi_complex).astype(np.float64),
            'csi_phase': np.angle(csi_complex).astype(np.float64),
            'subcarriers_nums': len(csi_complex),
        }

        # 6. 存入粗时间戳缓存
        if coarse_ts not in self._cache:
            self._cache[coarse_ts] = []
        self._cache[coarse_ts].append(processed)

        # 7. 清理过期缓存 (保留最近 N 个桶)
        while len(self._cache) > CSI_CACHE_MAX:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        # 8. 检查当前桶是否集齐 8 天线
        current_group = self._cache[coarse_ts]
        antennas_in_bucket = {item['antenna_id'] for item in current_group}
        if antennas_in_bucket != self.expected_antennas:
            return None

        # 9. 拼装 full_data
        full_data = {item['antenna_id']: item for item in current_group}
        del self._cache[coarse_ts]
        full_data['antenna_order'] = self.antenna_order

        # 10. 运行算法链
        full_data = self._run_algorithms(full_data)
        self._frame_count += 1

        return full_data

    def _run_algorithms(self, full_data: dict) -> dict:
        """依次运行所有注册的算法 (突变 full_data)

        Args:
            full_data: 待处理的完整帧

        Returns:
            算法链处理后的 full_data
        """
        for name, algo in self._algorithms.items():
            try:
                full_data = algo.apply(full_data, self.antenna_order)
            except Exception as e:
                logger.error("CSI 算法 '%s' 执行出错: %s", name, e)
        return full_data

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def clear(self):
        """清空所有内部状态"""
        self._cache.clear()
        self._frame_count = 0
        for algo in self._algorithms.values():
            if hasattr(algo, 'clear'):
                try:
                    algo.clear()
                except Exception as e:
                    logger.error("CSI 算法清理出错: %s", e)
        logger.info("CSI 处理器已重置")
