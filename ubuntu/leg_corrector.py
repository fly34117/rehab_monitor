"""
侧视步行腿关键点纠正器
解决: 近腿摆动时远腿(静止)关键点被干扰的问题

策略 (不修改原项目):
  1. 摆动/支撑状态感知: 根据踝关节Y轴速度判断哪条腿在动
  2. 支撑腿保护: 支撑腿关键点位置应稳定，若检测到异常位移则用Kalman预测值替代
  3. 交叉帧处理: 双腿X重叠时，锁定前一帧的左右分配
  4. 骨长约束: 大腿/小腿长度左右比异常时触发纠正
"""
import numpy as np
from collections import deque

LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

LEFT_LEG_IDX = [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]
RIGHT_LEG_IDX = [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]

# ---- 可调参数 ----
ANKLE_VEL_HISTORY = 8          # 踝关节速度滑动窗口
SWING_VEL_THRESH = 4.0         # Y轴速度 > 此值为摆动 (px/frame)
STANCE_MAX_JUMP = 18.0         # 支撑腿单帧最大位移 (px)
ANKLE_CROSS_GAP = 12           # 双踝X差 < 此值视为交叉帧
ALPHA_STANCE = 0.25            # 支撑腿平滑系数 (越小越平滑)
BONE_RATIO_ALERT = 2.0         # 左右骨长比超过此值报警


class LegCorrector:
    def __init__(self):
        self.prev_l_ankle = None   # (x, y)
        self.prev_r_ankle = None
        self.l_ankle_vel = deque(maxlen=ANKLE_VEL_HISTORY)
        self.r_ankle_vel = deque(maxlen=ANKLE_VEL_HISTORY)
        self.is_crossed = False
        self.cross_frames = 0
        self.swap_count = 0        # 累计纠正次数 (统计用)

    # ------------------------------------------------------------------
    def _ankle_vel_y(self, k, side):
        """计算踝关节Y轴速度"""
        idx = LEFT_ANKLE if side == 'L' else RIGHT_ANKLE
        prev = self.prev_l_ankle if side == 'L' else self.prev_r_ankle
        cy = float(k[idx, 1])
        if prev is None:
            return 0.0
        return cy - prev[1]

    def _smooth_stance_ankle(self, k, side):
        """对支撑腿踝关节做指数平滑"""
        idx = LEFT_ANKLE if side == 'L' else RIGHT_ANKLE
        prev = self.prev_l_ankle if side == 'L' else self.prev_r_ankle
        if prev is None:
            return
        cx, cy = float(k[idx, 0]), float(k[idx, 1])
        # 检查位移
        dist = np.sqrt((cx - prev[0])**2 + (cy - prev[1])**2)
        if dist > STANCE_MAX_JUMP:
            # 异常跳变 → 用平滑预测值替代
            sx = ALPHA_STANCE * cx + (1 - ALPHA_STANCE) * prev[0]
            sy = ALPHA_STANCE * cy + (1 - ALPHA_STANCE) * prev[1]
            k[idx, 0] = sx
            k[idx, 1] = sy

    def _detect_crossing(self, k):
        """双腿X轴是否交叉"""
        lx = float(k[LEFT_ANKLE, 0])
        rx = float(k[RIGHT_ANKLE, 0])
        return abs(lx - rx) < ANKLE_CROSS_GAP

    def _bone_len(self, k, idx_a, idx_b):
        return float(np.linalg.norm(
            np.array([k[idx_a, 0], k[idx_a, 1]]) -
            np.array([k[idx_b, 0], k[idx_b, 1]])))

    def _check_bone_swap(self, k):
        """骨长一致性检查: 左右大腿/小腿长度比异常 → 可能标反"""
        # 左大腿
        lt = self._bone_len(k, LEFT_HIP, LEFT_KNEE)
        rt = self._bone_len(k, RIGHT_HIP, RIGHT_KNEE)
        # 左小腿
        ls = self._bone_len(k, LEFT_KNEE, LEFT_ANKLE)
        rs = self._bone_len(k, RIGHT_KNEE, RIGHT_ANKLE)

        alerts = 0
        if lt > 2 and rt > 2:
            ratio = lt / rt if lt > rt else rt / lt
            if ratio > BONE_RATIO_ALERT:
                alerts += 1
        if ls > 2 and rs > 2:
            ratio = ls / rs if ls > rs else rs / ls
            if ratio > BONE_RATIO_ALERT:
                alerts += 1
        return alerts >= 1

    def _swap_legs(self, k):
        """交换左右腿关键点"""
        for la, ra in zip(LEFT_LEG_IDX, RIGHT_LEG_IDX):
            k[la], k[ra] = k[ra].copy(), k[la].copy()

    def _update_state(self, k):
        self.prev_l_ankle = (float(k[LEFT_ANKLE, 0]), float(k[LEFT_ANKLE, 1]))
        self.prev_r_ankle = (float(k[RIGHT_ANKLE, 0]), float(k[RIGHT_ANKLE, 1]))

    # ------------------------------------------------------------------
    def correct(self, kpts):
        """纠正腿关键点，返回 (kpts, was_corrected)"""
        if kpts is None or kpts.shape[0] != 1:
            return kpts, False

        k = kpts[0]  # (17, 3)
        corrected = False

        # 有效性
        valid_l = (float(k[LEFT_ANKLE, 2]) > 0.3 and
                   float(k[LEFT_KNEE, 2]) > 0.3 and
                   float(k[LEFT_HIP, 2]) > 0.3)
        valid_r = (float(k[RIGHT_ANKLE, 2]) > 0.3 and
                   float(k[RIGHT_KNEE, 2]) > 0.3 and
                   float(k[RIGHT_HIP, 2]) > 0.3)
        if not (valid_l and valid_r):
            if valid_l or valid_r:
                self._update_state(k)
            return kpts, False

        # ---- 1. 交叉检测 ----
        was_crossed = self.is_crossed
        self.is_crossed = self._detect_crossing(k)
        if self.is_crossed:
            self.cross_frames += 1
        else:
            self.cross_frames = max(0, self.cross_frames - 1)

        # ---- 2. 摆动/支撑状态识别 ----
        l_vy = self._ankle_vel_y(k, 'L')
        r_vy = self._ankle_vel_y(k, 'R')
        self.l_ankle_vel.append(abs(l_vy))
        self.r_ankle_vel.append(abs(r_vy))

        l_moving = (len(self.l_ankle_vel) >= 3 and
                    np.mean(self.l_ankle_vel) > SWING_VEL_THRESH)
        r_moving = (len(self.r_ankle_vel) >= 3 and
                    np.mean(self.r_ankle_vel) > SWING_VEL_THRESH)

        # ---- 3. 支撑腿保护 ----
        if l_moving and not r_moving and not self.is_crossed:
            self._smooth_stance_ankle(k, 'R')
        elif r_moving and not l_moving and not self.is_crossed:
            self._smooth_stance_ankle(k, 'L')

        # ---- 4. 骨长异常纠正 (非交叉帧 + 双腿均支撑) ----
        if not self.is_crossed and not l_moving and not r_moving:
            if self._check_bone_swap(k) and self.cross_frames == 0:
                self._swap_legs(k)
                self.swap_count += 1
                corrected = True

        self._update_state(k)
        return kpts, corrected
