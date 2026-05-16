"""跌倒检测 — 静态姿态规则 + 运动特征 (区分跌倒 vs 主动躺下)"""
from collections import deque
import numpy as np
import time

from .config import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    FALL_ASPECT_RATIO_THRESHOLD, FALL_HIP_Y_RATIO,
    FALL_TILT_DEG, FALL_CONFIRM_FRAMES,
)


class FallDetector:
    """跌倒检测器：姿态规则 + 运动速度(物理m/s, 距离自适应) + 时序确认

    核心区分逻辑:
      跌倒:  头部快速下坠 (>1.5 m/s) → 立刻告警
      躺下:  头部缓慢下降 (<0.6 m/s) → 不触发告警

    像素速度通过深度转换为物理速度: v_mps = v_pxps * depth_m / f_px
    """

    # 物理速度阈值 (m/s, 距离无关)
    FALL_VELOCITY_MPS = 1.5        # 头部下坠速度超过此值视为跌倒
    LIE_DOWN_VELOCITY_MPS = 0.6    # 低于此值视为主动躺下
    IMPACT_STILL_MPS = 0.2         # 坠落后头部接近静止的速度阈值
    IMPACT_STILL_FRAMES = 6        # 坠落后静止帧数确认

    FOCAL_LENGTH_PX = 554.0        # 焦距 (640px, 60° FOV)

    def __init__(self):
        self.status = "safe"        # "safe" / "warning" / "alert"
        self.score = 0.0

        # 姿态规则时序确认
        self._alert_frames = 0
        self._safe_frames = 0

        # ---- 运动追踪: 头部 (鼻子 COCO[0]) ----
        self._prev_head_y = None
        self._prev_time = None
        self._head_velocity_px = 0.0
        self._head_velocity_mps = 0.0
        self._velocity_history = deque(maxlen=12)      # 短期 (~0.4s), 用于实时检测
        self._long_vel_history = deque(maxlen=90)       # 长期 (~3s), 用于回溯分类
        self._fall_in_progress = False
        self._impact_frames = 0

        # ---- 回溯分类: 先确认躺下, 再回头看速度 ----
        self._lying_classification = "none"  # "none" / "fall" / "rest"
        self._lying_confirmed = False

    def update(self, kpts, bbox, frame_height, depth_m=0.0):
        """更新跌倒检测状态

        Args:
            kpts: (17,2) 关键点
            bbox: (x1,y1,x2,y2) or None
            frame_height: 图像高度
            depth_m: 目标深度 (米), 用于 px/s→m/s 转换

        Returns: (status, score)
        """
        if kpts is None:
            self._safe_frames += 1
            if self._safe_frames >= FALL_CONFIRM_FRAMES:
                self.status = "safe"
                self.score = 0.0
            self._prev_head_y = None
            self._prev_time = None
            return self.status, self.score

        k = kpts[:, :2]
        now = time.time()

        # ---- 运动分析: 头部垂直速度 (px/s → m/s) ----
        head_y = float(k[0][1]) if k[0][0] > 0 and k[0][1] > 0 else None

        if head_y is not None and self._prev_head_y is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.01:
                self._head_velocity_px = (head_y - self._prev_head_y) / dt  # +向下, px/s
                # 像素速度 → 物理速度 (m/s): v_mps = v_px * depth / f
                d = max(depth_m, 0.5)  # 最小 0.5m 防除零
                self._head_velocity_mps = self._head_velocity_px * d / self.FOCAL_LENGTH_PX
                self._velocity_history.append(self._head_velocity_mps)
                self._long_vel_history.append(self._head_velocity_mps)

        if head_y is not None:
            self._prev_head_y = head_y
        self._prev_time = now

        # 髋部 Y (供姿态规则使用)
        hip_center_y = float((k[LEFT_HIP][1] + k[RIGHT_HIP][1]) / 2)

        # ---- 姿态规则 ----
        votes = 0.0
        total = 0.0

        # 规则1: 边界框宽高比 (人倒下时 bbox 变宽变矮)
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                ratio = h / w
                if ratio < FALL_ASPECT_RATIO_THRESHOLD:
                    votes += 1.0
                total += 1.0

        # 规则2: 髋部 Y 位置 (在图像下方 = 躺在地上)
        if hip_center_y > frame_height * FALL_HIP_Y_RATIO:
            votes += 1.0
        total += 1.0

        # 规则3: 肩线倾角 (超过阈值 = 身体倾斜)
        if k[LEFT_SHOULDER][0] > 0 and k[RIGHT_SHOULDER][0] > 0:
            shoulder_vec = k[RIGHT_SHOULDER] - k[LEFT_SHOULDER]
            dx, dy = abs(shoulder_vec[0]), abs(shoulder_vec[1])
            if dx > 1e-6:
                tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
                if tilt_deg > FALL_TILT_DEG:
                    votes += 1.0
            total += 1.0

        posture_score = votes / total if total > 0 else 0.0

        # ---- 运动判定: 物理速度 (m/s) 区分跌倒 vs 主动躺下 ----
        peak_velocity = float(np.max(self._velocity_history)) if self._velocity_history else 0.0

        motion_fall = False
        motion_lie_down = False

        if peak_velocity > self.FALL_VELOCITY_MPS:
            motion_fall = True
            if not self._fall_in_progress:
                self._fall_in_progress = True
                self._peak_velocity = peak_velocity
                self._fall_start_time = now
        elif 0 < peak_velocity < self.LIE_DOWN_VELOCITY_MPS and posture_score >= 0.5:
            motion_lie_down = True
            self._fall_in_progress = False

        # 坠落冲击后静止确认
        if self._fall_in_progress:
            if abs(self._head_velocity_mps) < self.IMPACT_STILL_MPS:
                self._impact_frames += 1
            else:
                self._impact_frames = max(0, self._impact_frames - 1)

        # ---- 融合打分 ----
        if motion_fall and posture_score >= 0.33:
            # 运动特征强 → 立即触发
            self.score = max(posture_score, 0.6)
        elif motion_lie_down:
            # 主动躺下 → 降分
            self.score = 0.0
        else:
            self.score = posture_score

        # ---- 时序确认 ----
        if self.score >= 0.5:
            self._alert_frames += 1
            self._safe_frames = 0
        else:
            self._safe_frames += 1
            self._alert_frames = max(0, self._alert_frames - 1)

        if self._fall_in_progress and self._impact_frames >= self.IMPACT_STILL_FRAMES:
            # 坠落+冲击+静止 → 确认跌倒
            self.status = "alert"
        elif self._alert_frames >= FALL_CONFIRM_FRAMES:
            self.status = "alert"
        elif self._safe_frames >= FALL_CONFIRM_FRAMES * 2:
            self.status = "safe"
            self.score = 0.0
            self._fall_in_progress = False
            self._impact_frames = 0
        elif self._alert_frames > 0:
            self.status = "warning"

        return self.status, self.score

    @property
    def is_fallen(self):
        return self.status == "alert"

    def classify_lying_down(self):
        """回溯分类: 已确认躺下后, 检查速度缓冲区区分 跌倒 vs 休息

        必须在确认躺下后调用 (spatial.is_lying_down == True).
        查看过去 ~2-3s 的头部速度历史:
          峰值 > 1.2 m/s → 跌倒 (快速坠落)
          峰值 < 0.5 m/s → 休息 (缓慢躺下)

        Returns: "fall" | "rest" | "none"
        """
        if not self._lying_confirmed:
            return "none"
        if self._lying_classification != "none":
            return self._lying_classification  # 已分类, 直接返回

        if len(self._long_vel_history) < 10:
            return "none"

        # 取最近 2 秒的速度峰值
        vel_list = list(self._long_vel_history)
        peak = max(v for v in vel_list if v > 0) if any(v > 0 for v in vel_list) else 0

        if peak > self.FALL_VELOCITY_MPS:
            self._lying_classification = "fall"
        elif peak < self.LIE_DOWN_VELOCITY_MPS:
            self._lying_classification = "rest"
        else:
            self._lying_classification = "rest"  # 模棱两可算休息

        return self._lying_classification

    def reset_lying_state(self):
        """起身后重置躺下分类"""
        self._lying_classification = "none"
        self._lying_confirmed = False

    def get_motion_info(self):
        """返回运动诊断信息"""
        return {
            "head_velocity_mps": self._head_velocity_mps,
            "head_velocity_px": self._head_velocity_px,
            "fall_in_progress": self._fall_in_progress,
            "impact_frames": self._impact_frames,
        }
