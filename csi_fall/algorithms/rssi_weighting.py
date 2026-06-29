"""RSSI 加权算法 — 按天线信号强度补偿 CSI 幅度"""
from __future__ import annotations

import numpy as np

from rehab_monitor.logging_setup import get_logger

logger = get_logger("csi.rssi")


class RSSIWeighting:
    """RSSI 增益补偿

    原理:
      - 不同天线的信号强度 (RSSI) 不同, 导致 CSI 幅度存在系统偏差
      - 补偿公式: CSI_weighted = CSI × 10^(RSSI/20)
      - 使得所有天线的 CSI 幅度处于相同基准, 便于后续跨天线分析

    注入 full_data 的键:
      修改 'csi_complex', 'csi_magnitude', 'csi_phase' (原地)
    """

    @staticmethod
    def _apply_rssi_weighting(csi_frame: np.ndarray, rssi_vals: np.ndarray) -> np.ndarray:
        """核心算法: CSI * 10^(RSSI/20)

        Args:
            csi_frame: shape (n_antennas, n_subcarriers), complex64
            rssi_vals: shape (n_antennas,), int

        Returns:
            加权后的 CSI 矩阵, shape (n_antennas, n_subcarriers)
        """
        weights = np.power(10.0, rssi_vals / 20.0)
        return csi_frame * weights[:, np.newaxis]

    def apply(self, full_data: dict, antenna_order: list[str]) -> dict:
        """应用 RSSI 加权补偿

        Args:
            full_data: 包含每个天线的 'csi_complex' 和 'rssi' 键
            antenna_order: 天线 ID 有序列表

        Returns:
            修改后的 full_data
        """
        # 收集数据
        csi_list = []
        rssi_list = []
        for aid in antenna_order:
            data = full_data.get(aid, {})
            csi = data.get('csi_complex')
            if csi is not None:
                csi_list.append(np.array(csi, dtype=np.complex64))
            else:
                csi_list.append(np.zeros(64, dtype=np.complex64))
            rssi_list.append(data.get('rssi', 0))

        csi_matrix = np.array(csi_list)
        rssi_vals = np.array(rssi_list, dtype=np.float64)

        # 加权补偿
        csi_weighted = self._apply_rssi_weighting(csi_matrix, rssi_vals)

        # 写回 full_data
        for i, aid in enumerate(antenna_order):
            if aid in full_data:
                full_data[aid]['csi_complex'] = csi_weighted[i]
                full_data[aid]['csi_magnitude'] = np.abs(csi_weighted[i])
                full_data[aid]['csi_phase'] = np.angle(csi_weighted[i])

        return full_data

    def clear(self):
        """重置状态 (此算法无状态, 空操作)"""
        pass
