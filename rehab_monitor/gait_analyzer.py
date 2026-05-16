"""步态分析 — 8项临床指标 + 关节角度 + 朝向估算"""
import numpy as np
from collections import deque

from .config import (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_SHOULDER, RIGHT_SHOULDER,
)


def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return None
    cos = np.dot(ba, bc) / (nba * nbc)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class GaitAnalyzer:
    """步态分析器 — 临床级步态指标"""

    LIFT_THRESH = 3.0
    FOCAL_LENGTH_PX = 554.0       # 焦距, 用于 px→m 转换

    def __init__(self):
        # ===== 步数 =====
        self.step_count = 0
        self.left_step_count = 0
        self.right_step_count = 0

        # ===== 脚部状态 =====
        self.prev_left_rel_y = None
        self.prev_right_rel_y = None
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.left_foot_up = False
        self.right_foot_up = False
        self.left_foot_on_ground = True
        self.right_foot_on_ground = True

        # ===== 步态间期 =====
        self._last_left_step_time = None
        self._last_right_step_time = None
        self._last_step_time = None
        self.left_stride_times = deque(maxlen=30)
        self.right_stride_times = deque(maxlen=30)

        # ===== 髋部追踪 (像素) =====
        self.prev_hip_center = None
        self.prev_time = None
        self.speed = 0.0
        self.cadence = 0.0

        # ===== 膝关节 ROM =====
        self.left_knee_vals = deque(maxlen=60)
        self.right_knee_vals = deque(maxlen=60)
        self.symmetry = 1.0
        self._symmetry_source = "none"

        # ===== 原有临床指标 =====
        self._prev_step_pos = None
        self.stride_length_m = 0.0
        self._stride_lengths = deque(maxlen=20)
        self.gait_velocity_mps = 0.0
        self._gait_velocities = deque(maxlen=20)
        self.trunk_sway_angle = 0.0
        self._sway_history = deque(maxlen=15)
        self.is_double_support = False
        self._double_support_frames = 0
        self._total_walking_frames = 0
        self.double_support_ratio = 0.0
        self._ankle_vel_history = deque(maxlen=5)

        # ============================================================
        #  新增8项临床指标
        # ============================================================

        # 1. Step Length (左右步长, m)
        self.left_step_lengths = deque(maxlen=20)
        self.right_step_lengths = deque(maxlen=20)
        self.step_lengths = deque(maxlen=30)    # 所有步长 (用于CV)
        self.step_length_m = 0.0
        self.avg_step_length_m = 0.0
        self._last_left_foot_pos = None
        self._last_right_foot_pos = None

        # 2. Step Width (步宽, m)
        self.step_widths = deque(maxlen=20)
        self.step_width_m = 0.0
        self.avg_step_width_m = 0.0

        # 3. Stance / Swing Time (支撑/摆动相, s)
        self.left_stance_times = deque(maxlen=20)
        self.right_stance_times = deque(maxlen=20)
        self.left_swing_times = deque(maxlen=20)
        self.right_swing_times = deque(maxlen=20)
        self.stance_time_s = 0.0
        self.swing_time_s = 0.0
        self.stance_percentage = 0.0     # 支撑相占步态周期比例
        self._left_toe_off_time = None   # 左脚离地时间戳
        self._right_toe_off_time = None

        # 4. Knee ROM (膝关节活动范围, 度)
        self.left_knee_rom = 0.0
        self.right_knee_rom = 0.0
        self._left_knee_cycle_vals = []  # 当前周期内所有膝角
        self._right_knee_cycle_vals = []

        # 5/7. Gait Variability CV (步态变异性, %)
        self.step_times = deque(maxlen=30)
        self.step_time_cv = 0.0
        self.step_length_cv = 0.0
        self.step_width_cv = 0.0

        # 6. Foot Clearance (足廓清高度, cm)
        self.foot_clearances = deque(maxlen=20)
        self.foot_clearance_cm = 0.0     # 平均足廓清
        self.left_clearance_cm = 0.0
        self.right_clearance_cm = 0.0
        self._left_swing_min_y = None    # 摆动期最小Y (最高点)
        self._right_swing_min_y = None

        # 8. Gait Rehabilitation Score (综合评分, 0-100)
        self.gait_rehab_score = 0.0

        # ===== 朝向估算 =====
        self.orientation_yaw = 0.0
        self.orientation_conf = 0.0
        self._ref_shoulder_w = 0.0
        self._ref_hip_w = 0.0
        self._yaw_history = deque(maxlen=15)

        # ===== 角度平滑 =====
        self._angle_smooth = {}
        self._ANGLE_ALPHA = 0.35

    # =================================================================
    #  姿态估计 (保留)
    # =================================================================

    def estimate_orientation(self, kpts):
        k = kpts[:, :2]
        valid = (k[:, 0] > 0) & (k[:, 1] > 0)
        sw = float(np.linalg.norm(k[LEFT_SHOULDER] - k[RIGHT_SHOULDER])) if valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER] else 0.0
        hw = float(np.linalg.norm(k[LEFT_HIP] - k[RIGHT_HIP])) if valid[LEFT_HIP] and valid[RIGHT_HIP] else 0.0
        if sw < 5 or hw < 5:
            return self.orientation_yaw, 0.0
        alpha = 0.995
        self._ref_shoulder_w = max(sw, self._ref_shoulder_w * alpha)
        self._ref_hip_w = max(hw, self._ref_hip_w * alpha)
        if self._ref_shoulder_w < 5 or self._ref_hip_w < 5:
            return self.orientation_yaw, 0.0
        ratio_sw = np.clip(sw / self._ref_shoulder_w, 0.15, 1.0)
        ratio_hw = np.clip(hw / self._ref_hip_w, 0.15, 1.0)
        yaw_raw = (np.degrees(np.arccos(ratio_sw)) + np.degrees(np.arccos(ratio_hw))) / 2.0
        agreement = 1.0 - abs(np.degrees(np.arccos(ratio_sw)) - np.degrees(np.arccos(ratio_hw))) / (abs(yaw_raw) + 1e-8)
        self._yaw_history.append(yaw_raw)
        self.orientation_yaw = float(np.mean(self._yaw_history))
        self.orientation_conf = float(np.clip(agreement, 0.0, 1.0))
        return self.orientation_yaw, self.orientation_conf

    def _correct_sagittal_angle(self, angle_2d, sin_yaw):
        if angle_2d is None or sin_yaw < 0.35:
            return angle_2d
        flexion_2d = 180.0 - angle_2d
        if flexion_2d <= 0.5:
            return angle_2d
        return 180.0 - min(flexion_2d / sin_yaw, 120.0)

    def compute_joint_angles(self, kpts):
        k = kpts[:, :2]
        sin_yaw = abs(np.sin(np.radians(self.estimate_orientation(kpts)[0]))) if kpts is not None else 0.0
        angles = {}
        raw_lk = calc_angle(k[LEFT_HIP], k[LEFT_KNEE], k[LEFT_ANKLE])
        raw_rk = calc_angle(k[RIGHT_HIP], k[RIGHT_KNEE], k[RIGHT_ANKLE])
        raw_le = calc_angle(k[LEFT_SHOULDER], k[7], k[9])
        raw_re = calc_angle(k[RIGHT_SHOULDER], k[8], k[10])

        for key, raw_val in [("left_knee", raw_lk), ("right_knee", raw_rk),
                              ("left_elbow", raw_le), ("right_elbow", raw_re)]:
            corrected = self._correct_sagittal_angle(raw_val, sin_yaw)
            if corrected is not None:
                prev = self._angle_smooth.get(key)
                if prev is not None:
                    corrected = self._ANGLE_ALPHA * corrected + (1 - self._ANGLE_ALPHA) * prev
                self._angle_smooth[key] = corrected
            angles[key] = corrected

        angles["left_knee_raw"] = raw_lk
        angles["right_knee_raw"] = raw_rk
        angles["left_elbow_raw"] = raw_le
        angles["right_elbow_raw"] = raw_re
        angles["orientation_yaw"] = self.orientation_yaw
        angles["orientation_conf"] = self.orientation_conf

        valid_pts = (k[:, 0] > 0) & (k[:, 1] > 0)
        if valid_pts[5] and valid_pts[6]:
            shoulder_vec = k[6] - k[5]
            facing_2d = np.array([-shoulder_vec[1], shoulder_vec[0]])
            if valid_pts[0]:
                nose_to_mid = k[0] - (k[5] + k[6]) / 2
                if np.dot(facing_2d, nose_to_mid) < 0:
                    facing_2d = -facing_2d
            angles["facing_sign"] = 1.0 if facing_2d[0] > 0 else -1.0
        else:
            angles["facing_sign"] = 1.0

        if valid_pts[5] and valid_pts[6] and valid_pts[11] and valid_pts[12]:
            sm = (k[LEFT_SHOULDER] + k[RIGHT_SHOULDER]) / 2
            hm = (k[LEFT_HIP] + k[RIGHT_HIP]) / 2
            tv = sm - hm
            nt = np.linalg.norm(tv)
            if nt > 1e-6:
                angles["torso_tilt"] = float(np.degrees(np.arccos(
                    np.clip(abs(np.dot(tv, np.array([0.0, -1.0]))) / nt, -1.0, 1.0))))
            else:
                angles["torso_tilt"] = None
        else:
            angles["torso_tilt"] = None

        if angles["left_knee"] is not None:
            self.left_knee_vals.append(angles["left_knee"])
        if angles["right_knee"] is not None:
            self.right_knee_vals.append(angles["right_knee"])

        return angles

    # =================================================================
    #  步态事件检测辅助
    # =================================================================

    def _px_to_m(self, px, depth_m):
        """像素距离 → 米 (使用当前深度)"""
        d = max(depth_m, 0.5)
        return px * d / self.FOCAL_LENGTH_PX

    def _cv(self, data_deque):
        """变异系数 CV = std/mean × 100%"""
        if len(data_deque) < 3:
            return 0.0
        arr = np.array(data_deque, dtype=np.float64)
        mean = arr.mean()
        if mean < 1e-8:
            return 0.0
        return float(np.std(arr) / mean * 100.0)

    def _ema_update(self, current, prev, alpha=0.3):
        if prev is None:
            return current
        return alpha * current + (1 - alpha) * prev

    # =================================================================
    #  主更新方法
    # =================================================================

    def update(self, kpts, timestamp, world_pos=None, depth_m=3.0):
        """更新所有步态指标

        Args:
            kpts: (17,2) or (17,3) 关键点
            timestamp: float 秒
            world_pos: (x_m, y_m) 世界坐标 or None
            depth_m: float 当前深度 (米)
        """
        if kpts is None:
            return self.step_count

        k = kpts[:, :2]
        valid = (k[:, 0] > 0) & (k[:, 1] > 0)
        if not (valid[LEFT_HIP] and valid[RIGHT_HIP]):
            return self.step_count

        hip_y = (k[LEFT_HIP][1] + k[RIGHT_HIP][1]) / 2
        left_rel_y = k[LEFT_ANKLE][1] - hip_y if valid[LEFT_ANKLE] else None
        right_rel_y = k[RIGHT_ANKLE][1] - hip_y if valid[RIGHT_ANKLE] else None
        left_ankle_y = k[LEFT_ANKLE][1] if valid[LEFT_ANKLE] else None
        right_ankle_y = k[RIGHT_ANKLE][1] if valid[RIGHT_ANKLE] else None

        step_occurred = False
        left_step_occurred = False
        right_step_occurred = False
        left_toe_off = False
        right_toe_off = False

        # ---- 左脚步态事件 ----
        if left_rel_y is not None and self.prev_left_rel_y is not None:
            dy = left_rel_y - self.prev_left_rel_y
            if not self.left_foot_up and dy < -self.LIFT_THRESH:
                # 脚尖离地 (Toe Off) → 摆动相开始
                self.left_foot_up = True
                left_toe_off = True
                self._left_toe_off_time = timestamp
                self._left_swing_min_y = left_ankle_y  # 初始化摆动最高点
                self.left_foot_on_ground = False
            elif self.left_foot_up and dy > 0:
                # 足跟着地 (Heel Strike) → 支撑相开始
                self.left_foot_up = False
                self.left_step_count += 1
                self.step_count += 1
                step_occurred = True
                left_step_occurred = True
                self.left_foot_on_ground = True
                if self._last_left_step_time is not None:
                    self.left_stride_times.append(timestamp - self._last_left_step_time)
                self._last_left_step_time = timestamp

        # ---- 右脚步态事件 ----
        if right_rel_y is not None and self.prev_right_rel_y is not None:
            dy = right_rel_y - self.prev_right_rel_y
            if not self.right_foot_up and dy < -self.LIFT_THRESH:
                self.right_foot_up = True
                right_toe_off = True
                self._right_toe_off_time = timestamp
                self._right_swing_min_y = right_ankle_y
                self.right_foot_on_ground = False
            elif self.right_foot_up and dy > 0:
                self.right_foot_up = False
                self.right_step_count += 1
                self.step_count += 1
                step_occurred = True
                right_step_occurred = True
                self.right_foot_on_ground = True
                if self._last_right_step_time is not None:
                    self.right_stride_times.append(timestamp - self._last_right_step_time)
                self._last_right_step_time = timestamp

        self.prev_left_rel_y = left_rel_y
        self.prev_right_rel_y = right_rel_y

        # ================================================================
        #  1. Step Length (左右步长, m)
        # ================================================================

        # 步宽: 双踝世界X坐标差 (像素差 × depth/f)
        if valid[LEFT_ANKLE] and valid[RIGHT_ANKLE]:
            ankle_dx_px = abs(k[LEFT_ANKLE][0] - k[RIGHT_ANKLE][0])
            sw = self._px_to_m(ankle_dx_px, depth_m)
            if 0.02 < sw < 0.8:
                self.step_widths.append(sw)
                self.step_width_m = sw
        if len(self.step_widths) >= 1:
            self.avg_step_width_m = float(np.mean(self.step_widths))

        # 步长: 同侧两次着地之间的世界坐标距离
        if left_step_occurred and world_pos is not None:
            pos = np.array(world_pos, dtype=np.float64)
            if self._last_left_foot_pos is not None:
                dist = float(np.linalg.norm(pos - self._last_left_foot_pos))
                if 0.1 < dist < 3.0:
                    self.left_step_lengths.append(dist)
                    self.step_lengths.append(dist)  # 用于CV
            self._last_left_foot_pos = pos
            if self._left_toe_off_time is not None and self._last_left_step_time is not None:
                swing_t = self._last_left_step_time - self._left_toe_off_time
                if 0.05 < swing_t < 2.0:
                    self.left_swing_times.append(swing_t)

        if right_step_occurred and world_pos is not None:
            pos = np.array(world_pos, dtype=np.float64)
            if self._last_right_foot_pos is not None:
                dist = float(np.linalg.norm(pos - self._last_right_foot_pos))
                if 0.1 < dist < 3.0:
                    self.right_step_lengths.append(dist)
                    self.step_lengths.append(dist)
            self._last_right_foot_pos = pos
            if self._right_toe_off_time is not None and self._last_right_step_time is not None:
                swing_t = self._last_right_step_time - self._right_toe_off_time
                if 0.05 < swing_t < 2.0:
                    self.right_swing_times.append(swing_t)

        # 当前步长 (取最新)
        all_step_lens = list(self.left_step_lengths) + list(self.right_step_lengths)
        if all_step_lens:
            self.step_length_m = all_step_lens[-1]
            self.avg_step_length_m = float(np.mean(all_step_lens))

        # 跨步步长 (stride = 左右各一步)
        if step_occurred and world_pos is not None:
            pos = np.array(world_pos, dtype=np.float64)
            if self._prev_step_pos is not None and self._last_step_time is not None:
                dt = timestamp - self._last_step_time
                dist = float(np.linalg.norm(pos - self._prev_step_pos))
                if 0.1 < dist < 3.0 and dt > 0.05:
                    self._stride_lengths.append(dist)
                    self._gait_velocities.append(dist / dt)
                    self.step_times.append(dt)
            self._prev_step_pos = pos
            self._last_step_time = timestamp

        if len(self._stride_lengths) >= 1:
            self.stride_length_m = float(np.mean(self._stride_lengths))
        if len(self._gait_velocities) >= 1:
            self.gait_velocity_mps = float(np.mean(self._gait_velocities))

        # ================================================================
        #  3. Stance Time / Swing Time
        # ================================================================

        # 支撑相时间 = 从 heel strike 到 toe off 的时长
        # 从 stride_times 和 swing_times 推导: stance = stride - swing
        if len(self.left_stride_times) >= 1 and len(self.left_swing_times) >= 1:
            avg_stride = sum(self.left_stride_times) / len(self.left_stride_times)
            avg_swing = sum(self.left_swing_times) / len(self.left_swing_times)
            stance = avg_stride - avg_swing
            if stance > 0:
                self.left_stance_times.append(stance)
                self.stance_time_s = stance
                self.swing_time_s = avg_swing
                if avg_stride > 0:
                    self.stance_percentage = stance / avg_stride * 100.0
        elif len(self.right_stride_times) >= 1 and len(self.right_swing_times) >= 1:
            avg_stride = sum(self.right_stride_times) / len(self.right_stride_times)
            avg_swing = sum(self.right_swing_times) / len(self.right_swing_times)
            stance = avg_stride - avg_swing
            if stance > 0:
                self.right_stance_times.append(stance)
                self.stance_time_s = stance
                self.swing_time_s = avg_swing
                if avg_stride > 0:
                    self.stance_percentage = stance / avg_stride * 100.0

        # ================================================================
        #  4. Knee ROM (步态周期内膝角范围)
        # ================================================================

        lk = calc_angle(k[LEFT_HIP], k[LEFT_KNEE], k[LEFT_ANKLE]) if valid[LEFT_HIP] and valid[LEFT_KNEE] and valid[LEFT_ANKLE] else None
        rk = calc_angle(k[RIGHT_HIP], k[RIGHT_KNEE], k[RIGHT_ANKLE]) if valid[RIGHT_HIP] and valid[RIGHT_KNEE] and valid[RIGHT_ANKLE] else None

        if lk is not None:
            self._left_knee_cycle_vals.append(lk)
        if rk is not None:
            self._right_knee_cycle_vals.append(rk)

        # 步态周期结束时计算 ROM (同侧heel strike → 下一次heel strike)
        if left_step_occurred and len(self._left_knee_cycle_vals) >= 5:
            vals = self._left_knee_cycle_vals
            rom = max(vals) - min(vals)
            if 5 < rom < 120:
                self.left_knee_rom = float(rom)
            self._left_knee_cycle_vals = []

        if right_step_occurred and len(self._right_knee_cycle_vals) >= 5:
            vals = self._right_knee_cycle_vals
            rom = max(vals) - min(vals)
            if 5 < rom < 120:
                self.right_knee_rom = float(rom)
            self._right_knee_cycle_vals = []

        # ================================================================
        #  5/7. Gait Variability CV (步态变异性, %)
        # ================================================================

        self.step_time_cv = self._cv(self.step_times)
        self.step_length_cv = self._cv(self.step_lengths)
        self.step_width_cv = self._cv(self.step_widths)

        # ================================================================
        #  6. Foot Clearance (足廓清高度, 像素)
        # ================================================================

        # 摆动相中跟踪踝关节最小Y (最高点)
        if self.left_foot_up and left_ankle_y is not None and self._left_swing_min_y is not None:
            if left_ankle_y < self._left_swing_min_y:
                self._left_swing_min_y = left_ankle_y
        elif left_step_occurred and self._left_swing_min_y is not None and left_ankle_y is not None:
            # 摆动结束: 记录足廓清
            clearance_px = left_ankle_y - self._left_swing_min_y
            if clearance_px > 0:
                clearance_cm = self._px_to_m(clearance_px, depth_m) * 100.0
                if 0.5 < clearance_cm < 50.0:
                    self.left_clearance_cm = float(self._ema_update(clearance_cm, self.left_clearance_cm, 0.4))
                    self.foot_clearances.append(self.left_clearance_cm)

        if self.right_foot_up and right_ankle_y is not None and self._right_swing_min_y is not None:
            if right_ankle_y < self._right_swing_min_y:
                self._right_swing_min_y = right_ankle_y
        elif right_step_occurred and self._right_swing_min_y is not None and right_ankle_y is not None:
            clearance_px = right_ankle_y - self._right_swing_min_y
            if clearance_px > 0:
                clearance_cm = self._px_to_m(clearance_px, depth_m) * 100.0
                if 0.5 < clearance_cm < 50.0:
                    self.right_clearance_cm = float(self._ema_update(clearance_cm, self.right_clearance_cm, 0.4))
                    self.foot_clearances.append(self.right_clearance_cm)

        if len(self.foot_clearances) >= 1:
            self.foot_clearance_cm = float(np.mean(self.foot_clearances))

        # ================================================================
        #  2. 躯干侧倾 + 双支撑 + 步频 + 对称性 (保留)
        # ================================================================

        if valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER] and valid[LEFT_HIP] and valid[RIGHT_HIP]:
            sm = (k[LEFT_SHOULDER] + k[RIGHT_SHOULDER]) / 2
            hm = (k[LEFT_HIP] + k[RIGHT_HIP]) / 2
            nt = float(np.linalg.norm(sm - hm))
            if nt > 5:
                cos_s = abs(np.dot(sm - hm, np.array([0.0, -1.0]))) / nt
                self._sway_history.append(float(np.degrees(np.arccos(np.clip(cos_s, 0.0, 1.0)))))
                self.trunk_sway_angle = float(np.mean(self._sway_history))

        if valid[LEFT_ANKLE] and valid[RIGHT_ANKLE] and valid[LEFT_HIP] and valid[RIGHT_HIP]:
            ankle_below = (left_ankle_y > hip_y * 0.85 and right_ankle_y > hip_y * 0.85)
            lv = rv = 0.0
            if self.prev_left_ankle_y is not None and self.prev_time is not None:
                dt = timestamp - self.prev_time
                if dt > 0.01:
                    lv = abs(left_ankle_y - self.prev_left_ankle_y) / dt
                    rv = abs(right_ankle_y - self.prev_right_ankle_y) / dt
            self._ankle_vel_history.append((lv, rv))
            sc = sum(1 for a, b in self._ankle_vel_history if a < 50.0 and b < 50.0)
            self.is_double_support = ankle_below and sc >= 3
            if self.step_count > 0 or self.is_double_support:
                self._total_walking_frames += 1
                if self.is_double_support:
                    self._double_support_frames += 1
                if self._total_walking_frames > 0:
                    self.double_support_ratio = self._double_support_frames / self._total_walking_frames
            self.prev_left_ankle_y = left_ankle_y
            self.prev_right_ankle_y = right_ankle_y

        hip_center = (k[LEFT_HIP] + k[RIGHT_HIP]) / 2
        if self.prev_hip_center is not None and self.prev_time is not None:
            dt = timestamp - self.prev_time
            if dt > 0.01:
                self.speed = 0.7 * self.speed + 0.3 * (abs(hip_center[0] - self.prev_hip_center[0]) / dt)
        self.prev_hip_center = hip_center
        self.prev_time = timestamp

        all_strides = list(self.left_stride_times) + list(self.right_stride_times)
        if len(all_strides) >= 2:
            avg = sum(all_strides) / len(all_strides)
            self.cadence = 60.0 / avg if avg > 0 else 0.0

        if len(self.left_stride_times) >= 1 and len(self.right_stride_times) >= 1:
            al = sum(self.left_stride_times) / len(self.left_stride_times)
            ar = sum(self.right_stride_times) / len(self.right_stride_times)
            if max(al, ar) > 1e-6:
                self.symmetry = min(al, ar) / max(al, ar)
                self._symmetry_source = "stride"
        elif len(self.left_knee_vals) >= 10 and len(self.right_knee_vals) >= 10:
            rl = max(self.left_knee_vals) - min(self.left_knee_vals)
            rr = max(self.right_knee_vals) - min(self.right_knee_vals)
            if max(rl, rr) > 5:
                self.symmetry = min(rl, rr) / max(rl, rr)
                self._symmetry_source = "knee_rom"

        # ================================================================
        #  8. Gait Rehabilitation Score (综合评分 0-100)
        # ================================================================
        self._compute_gait_score()

        return self.step_count

    # =================================================================
    #  综合康复评分
    # =================================================================

    def _score_linear(self, value, target, worse, weight=1.0, reverse=False):
        """线性评分: 越接近target分越高, 超过worse方向得0"""
        if value is None or value == 0:
            return 0.0
        if reverse:
            value = -value
            target = -target
            worse = -worse
        if worse > target:
            score = max(0.0, min(1.0, (value - worse) / (target - worse)))
        else:
            score = max(0.0, min(1.0, 1.0 - (value - target) / (worse - target + 1e-8)))
        return score * weight

    def _compute_gait_score(self):
        """GRS: 加权综合评分 0-100"""
        s_speed = self._score_linear(self.gait_velocity_mps, 1.2, 0.2) * 0.25
        s_sym = self._score_linear(self.symmetry, 1.0, 0.5) * 0.25
        s_cv = self._score_linear(self.step_length_cv, 5.0, 30.0, reverse=True) * 0.20
        s_rom = self._score_linear(max(self.left_knee_rom, self.right_knee_rom), 55.0, 10.0) * 0.15
        # 双支撑: 健康年轻人 ~20%, 偏离越大分越低 (双支撑过高=步态不稳)
        ds_best = 20.0
        ds_val = self.double_support_ratio * 100.0 if self.double_support_ratio > 0 else 20.0
        s_ds = max(0.0, 1.0 - abs(ds_val - ds_best) / 40.0) * 0.15

        self.gait_rehab_score = round((s_speed + s_sym + s_cv + s_rom + s_ds) * 100.0, 1)

    # =================================================================
    #  汇总接口
    # =================================================================

    def get_metrics(self):
        """返回所有临床指标的字典"""
        return {
            "step_count": self.step_count,
            "left_steps": self.left_step_count,
            "right_steps": self.right_step_count,
            "cadence_spm": round(self.cadence, 1),
            "symmetry": round(self.symmetry, 3),
            "symmetry_source": self._symmetry_source,
            # 步长
            "stride_length_m": round(self.stride_length_m, 3),
            "avg_step_length_m": round(self.avg_step_length_m, 3),
            # 步速
            "gait_velocity_mps": round(self.gait_velocity_mps, 3),
            # 步宽
            "step_width_m": round(self.step_width_m, 3),
            "avg_step_width_m": round(self.avg_step_width_m, 3),
            # 支撑/摆动
            "stance_time_s": round(self.stance_time_s, 3),
            "swing_time_s": round(self.swing_time_s, 3),
            "stance_percentage": round(self.stance_percentage, 1),
            # 膝 ROM
            "left_knee_rom": round(self.left_knee_rom, 1),
            "right_knee_rom": round(self.right_knee_rom, 1),
            # 变异性 CV
            "step_time_cv": round(self.step_time_cv, 1),
            "step_length_cv": round(self.step_length_cv, 1),
            "step_width_cv": round(self.step_width_cv, 1),
            # 足廓清
            "foot_clearance_cm": round(self.foot_clearance_cm, 1),
            # 综合评分
            "gait_rehab_score": round(self.gait_rehab_score, 1),
            # 其他
            "trunk_sway_deg": round(self.trunk_sway_angle, 1),
            "is_double_support": self.is_double_support,
            "double_support_ratio": round(self.double_support_ratio, 3),
            "speed_pxps": round(self.speed, 1),
            "orientation_yaw": round(self.orientation_yaw, 1),
        }

    def reset(self):
        yaw = self.orientation_yaw
        rsw = self._ref_shoulder_w
        rhw = self._ref_hip_w
        self.__init__()
        self.orientation_yaw = yaw
        self._ref_shoulder_w = rsw
        self._ref_hip_w = rhw
