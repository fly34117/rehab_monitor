"""特征向量滤波 — 逐子载波主成分去噪
对每个子载波独立计算协方差矩阵，提取主特征向量作为信号的可靠估计，
以 0 号天线为相位参考校正相位和幅度。
"""
from __future__ import annotations
from collections import deque

import numpy as np

from rehab_monitor.logging_setup import get_logger

# numba 加速 (可选, 未安装时静默回退为纯 NumPy)
try:
    from numba import jit
    _numba_available = True
except ImportError:
    def jit(*args, **kwargs):
        """numba 不可用时的无操作装饰器"""
        return lambda f: f
    _numba_available = False

logger = get_logger("csi.eigenvec")


class EigenvecPerSubcarrierFilter:
    """逐子载波特征向量去噪

    原理:
      - 对每个子载波, 收集滑动窗口内 (n_samples 帧) 8 天线的 CSI 复数向量
      - 形成 (n_samples, n_antennas) 矩阵, 计算协方差矩阵
      - 取主特征向量 (最大特征值对应) 作为去噪后的 CSI 估计
      - 以 0 号天线为相位参考: 乘以 exp(-j * angle(eigenvector[0]))
      - 幅度校正: 乘以 sqrt(max_eigenvalue)

    注入 full_data 的键:
      修改 'csi_complex', 'csi_magnitude', 'csi_phase' (原地)

    Args:
        sliding_window_size: 滑动窗口大小 (帧数), 默认 15
    """

    def __init__(self, sliding_window_size: int = 15):
        self.sliding_window_size = sliding_window_size
        self._windows: list[deque] | None = None
        self._current_num_subcarriers: int = 0

    # ------------------------------------------------------------------
    # 核心算法 (静态方法, 可 numba 加速)
    # ------------------------------------------------------------------

    @staticmethod
    @jit(nopython=True)
    def _interp_eigenvec(csi: np.ndarray) -> np.ndarray:
        """逐子载波特征值插值核心 (numba 加速)

        Args:
            csi: shape (n_samples, n_antennas, n_subcarriers), complex128

        Returns:
            去噪后的 CSI, shape (n_antennas, n_subcarriers), complex128
        """
        antenna_shape = csi.shape[1:-1]
        n_subcarriers = csi.shape[-1]

        # 展平天线维度: (n_samples, flat_antennas, n_subcarriers)
        csi_flat = csi.reshape(csi.shape[0], -1, n_subcarriers)

        # 逐子载波协方差矩阵: R[s, a, b] = sum_n csi[n, a, s] * conj(csi[n, b, s])
        R = np.einsum("nas,nbs->sab", csi_flat, np.conj(csi_flat))

        # 特征值分解
        eigvals, eigvecs = np.linalg.eig(R)
        # 按特征值绝对值降序排列
        idx = np.argsort(np.abs(eigvals), axis=1)[:, ::-1]
        eigvals = np.take_along_axis(eigvals, idx, axis=1)
        eigvecs = np.take_along_axis(eigvecs, idx[:, np.newaxis, :], axis=2)

        # 主特征向量 (最大特征值对应)
        principal_eigenvectors = eigvecs[:, :, 0]  # (n_subcarriers, flat_antennas)
        principal_eigenvalues = eigvals[:, 0]        # (n_subcarriers,)

        # 相位参考 0 号天线 + 幅度校正
        result_flat = (np.sqrt(principal_eigenvalues)[:, np.newaxis]
                       * principal_eigenvectors)
        result_flat = result_flat * np.exp(
            -1.0j * np.angle(principal_eigenvectors[:, 0][:, np.newaxis])
        )

        # 维度交换 + 恢复形状
        result_flat = np.swapaxes(result_flat, 0, 1)
        return result_flat.reshape(antenna_shape + (n_subcarriers,))

    # ------------------------------------------------------------------
    # 算法接口
    # ------------------------------------------------------------------

    def apply(self, full_data: dict, antenna_order: list[str]) -> dict:
        """应用逐子载波特征向量去噪

        Args:
            full_data: 当前帧的 CSI 数据 (每帧 8 天线)
            antenna_order: 天线 ID 有序列表

        Returns:
            修改后的 full_data
        """
        if not full_data or antenna_order[0] not in full_data:
            return full_data

        n_sc = full_data[antenna_order[0]]['subcarriers_nums']

        # 初始化窗口 (首次或子载波数变化时)
        if self._windows is None or self._current_num_subcarriers != n_sc:
            self._windows = [deque(maxlen=self.sliding_window_size) for _ in range(n_sc)]
            self._current_num_subcarriers = n_sc

        # 将当前帧各子载波的多天线 CSI 压入窗口
        for n in range(n_sc):
            try:
                csi_vector_n = np.array([
                    full_data[aid]['csi_complex'][n] for aid in antenna_order
                ], dtype=np.complex64)
                self._windows[n].append(csi_vector_n)
            except (KeyError, IndexError):
                continue

        # 窗口填满后执行特征向量去噪
        if all(len(w) == self.sliding_window_size for w in self._windows):
            # 构造窗口数据: (n_samples, n_antennas, n_subcarriers)
            csi_window = []
            for q in self._windows:
                csi_window.append(np.array(list(q), dtype=np.complex128))
            csi_matrix = np.stack(csi_window, axis=-1)  # (samples, ant, sc)

            # 调用去噪核心
            h_hat_all = self._interp_eigenvec(csi_matrix)

            # 写回 full_data
            for idx, aid in enumerate(antenna_order):
                if aid in full_data:
                    h_vec = h_hat_all[idx, :]
                    full_data[aid]['csi_complex'] = h_vec
                    full_data[aid]['csi_magnitude'] = np.abs(h_vec)
                    full_data[aid]['csi_phase'] = np.angle(h_vec)

        return full_data

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def clear(self):
        """清空所有窗口状态"""
        self._current_num_subcarriers = 0
        if self._windows:
            for w in self._windows:
                w.clear()
            self._windows = None
