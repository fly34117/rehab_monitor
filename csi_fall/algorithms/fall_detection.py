"""跌倒检测算法 — 基于CSI幅度方差的滑动窗口检测"""
from __future__ import annotations
from collections import deque

import numpy as np

from rehab_monitor.logging_setup import get_logger

from ..config import (
    FALL_SHORT_WINDOW, FALL_LONG_WINDOW, FALL_THRESHOLD,
    FALL_ALERT_FRAMES, FALL_COOLDOWN_FRAMES,
)

logger = get_logger("csi.fall_detect")


class FallDetection:
    """基于 WiFi CSI 的跌倒检测算法

    原理:
      - 人在跌倒时会引起多径环境的剧烈变化, 导致 CSI 幅度方差急剧增大
      - 使用双窗口机制: 短窗口 (检测窗口) vs 长窗口 (基线窗口)
      - 当短窗口方差远超长窗口方差时, 判定为跌倒事件
      - 连续多帧超阈值才触发告警 (防误报)
      - 告警后冷却 N 帧 (防重复报警)

    注入 full_data 的键:
        'fall_detected' : bool    — 当前帧是否检测到跌倒异常
        'fall_score'    : float   — 当前跌倒分数 (方差比)
        'fall_history'  : list    — 最近 N 帧的跌倒分数
        'fall_alert'    : bool    — 是否触发告警 (连续多帧超阈值)
        'fall_count'    : int     — 累计跌倒告警次数

    Args:
        short_window: 短窗口大小 (检测窗口), 单位：帧
        long_window:  长窗口大小 (基线窗口), 单位：帧
        threshold:    方差比阈值, 超过此值判定为跌倒
        alert_frames: 连续多少帧超阈值才触发告警 (防误报)
        cooldown_frames: 告警后冷却帧数, 防止重复报警
    """

    def __init__(
        self,
        short_window: int = 0,
        long_window: int = 0,
        threshold: float = 0.0,
        alert_frames: int = 0,
        cooldown_frames: int = 0,
    ):
        self.short_window = short_window or FALL_SHORT_WINDOW
        self.long_window = long_window or FALL_LONG_WINDOW
        self.threshold = threshold or FALL_THRESHOLD
        self.alert_frames = alert_frames or FALL_ALERT_FRAMES
        self.cooldown_frames = cooldown_frames or FALL_COOLDOWN_FRAMES

        self._variance_history: deque[float] = deque(maxlen=self.long_window)
        self._score_history: deque[float] = deque(maxlen=self.long_window)
        self._consecutive_detections: int = 0
        self._cooldown_counter: int = 0
        self._alert_active: bool = False
        self.fall_count: int = 0

    def apply(self, full_data: dict, antenna_order: list[str]) -> dict:
        """对一帧 CSI 数据运行跌倒检测

        Args:
            full_data: CSI 帧数据, 包含每个天线的 'csi_magnitude'
            antenna_order: 天线 ID 有序列表

        Returns:
            注入检测结果后的 full_data
        """
        mag_list = []
        for aid in antenna_order:
            data = full_data.get(aid, {})
            mag = data.get('csi_magnitude')
            if mag is not None and len(mag) > 0:
                mag_list.append(np.array(mag, dtype=np.float64))

        if not mag_list:
            full_data['fall_detected'] = False
            full_data['fall_score'] = 0.0
            full_data['fall_history'] = list(self._score_history)
            full_data['fall_alert'] = False
            full_data['fall_count'] = self.fall_count
            return full_data

        mag_matrix = np.array(mag_list)
        current_var = float(np.var(mag_matrix))
        self._variance_history.append(current_var)

        score = 0.0
        detected = False

        if len(self._variance_history) >= self.short_window + 1:
            var_list = list(self._variance_history)
            short_var = np.mean(var_list[-self.short_window:])
            long_var = np.mean(var_list)

            if long_var > 1e-10:
                score = float(short_var / long_var)

            if score > self.threshold:
                detected = True
                self._consecutive_detections += 1
            else:
                self._consecutive_detections = 0

        self._score_history.append(score)

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self._alert_active = False
        elif self._consecutive_detections >= self.alert_frames:
            self._alert_active = True
            self.fall_count += 1
            self._cooldown_counter = self.cooldown_frames
            self._consecutive_detections = 0
            logger.warning(
                "CSI 跌倒检测告警 #%d, score=%.2f, threshold=%.1f",
                self.fall_count, score, self.threshold,
            )

        full_data['fall_detected'] = detected
        full_data['fall_score'] = score
        full_data['fall_history'] = list(self._score_history)
        full_data['fall_alert'] = self._alert_active
        full_data['fall_count'] = self.fall_count

        return full_data

    @property
    def is_fallen(self) -> bool:
        """当前是否处于跌倒告警状态"""
        return self._alert_active

    @property
    def score(self) -> float:
        """最新一帧的跌倒分数"""
        if self._score_history:
            return self._score_history[-1]
        return 0.0

    def clear(self):
        """重置所有检测状态"""
        self._variance_history.clear()
        self._score_history.clear()
        self._consecutive_detections = 0
        self._cooldown_counter = 0
        self._alert_active = False
        self.fall_count = 0
