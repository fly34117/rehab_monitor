"""骨架可视化窗口 — 距离归一化, 固定尺寸显示姿态"""
import cv2
import numpy as np
from collections import deque


# COCO 17 关键点骨架连接 (成对索引)
SKELETON_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # 头
    (5, 6),                                # 肩
    (5, 7), (7, 9), (6, 8), (8, 10),      # 手臂
    (5, 11), (6, 12), (11, 12),            # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16), # 腿
]

# 骨骼颜色: 左=蓝色, 右=红色, 中心=白色
LEFT_BONES = {(5, 7), (7, 9), (5, 11), (11, 13), (13, 15)}
RIGHT_BONES = {(6, 8), (8, 10), (6, 12), (12, 14), (14, 16)}
CENTER_BONES = {(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (11, 12)}


class SkeletonViewer:
    """独立窗口: 归一化骨架 (不受距离影响, 固定画布大小)"""

    def __init__(self, window_name="Skeleton (Normalized)"):
        self.window_name = window_name
        self.width = 300
        self.height = 400
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        self.kpts_smoothed = None  # EMA平滑后的关键点
        self._alpha = 0.4          # EMA系数

    def update(self, kpts_single):
        """更新骨架 (自动 EMA 平滑去抖)

        Args:
            kpts_single: (17, 2) 或 (17, 3) 单人关键点
        """
        if kpts_single is None:
            return

        k = kpts_single[:, :2].copy()
        if self.kpts_smoothed is None or self.kpts_smoothed.shape != k.shape:
            self.kpts_smoothed = k
        else:
            valid = (k[:, 0] > 0) & (k[:, 1] > 0)
            prev_valid = (self.kpts_smoothed[:, 0] > 0) & (self.kpts_smoothed[:, 1] > 0)
            for i in range(len(k)):
                if valid[i]:
                    if prev_valid[i]:
                        self.kpts_smoothed[i] = (self._alpha * k[i] +
                                                  (1 - self._alpha) * self.kpts_smoothed[i])
                    else:
                        self.kpts_smoothed[i] = k[i]

    def render(self, fall_status="safe", orientation_yaw=0.0, torso_angle=0.0, is_lying=False):
        """绘制归一化骨架"""
        w, h = self.width, self.height
        canvas = np.full((h, w, 3), (25, 25, 30), dtype=np.uint8)

        if self.kpts_smoothed is None:
            cv2.imshow(self.window_name, canvas)
            return

        k = self.kpts_smoothed
        valid = (k[:, 0] > 0) & (k[:, 1] > 0)

        if valid.sum() < 3:
            cv2.imshow(self.window_name, canvas)
            return

        # ---- 归一化: 缩放到固定画布 ----
        valid_pts = k[valid]
        x_min, y_min = valid_pts.min(axis=0)
        x_max, y_max = valid_pts.max(axis=0)

        body_w = x_max - x_min
        body_h = y_max - y_min
        if body_w < 1:
            body_w = 1
        if body_h < 1:
            body_h = 1

        # 目标显示区域
        margin = 30
        target_w = w - 2 * margin
        target_h = h - 2 * margin - 30  # 顶部留状态栏

        scale = min(target_w / body_w, target_h / body_h)
        # 平移使骨架居中
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        ox = w / 2 - cx * scale
        oy = margin + 20 + target_h / 2 - cy * scale

        def to_canvas(px, py):
            return int(px * scale + ox), int(py * scale + oy)

        # ---- 画骨骼 ----
        for i, j in SKELETON_BONES:
            if not (valid[i] and valid[j]):
                continue
            pair = (min(i, j), max(i, j))
            if pair in LEFT_BONES:
                color = (255, 140, 50)     # 蓝色(左)
            elif pair in RIGHT_BONES:
                color = (50, 140, 255)     # 红色(右)
            elif pair in CENTER_BONES:
                color = (200, 200, 200)    # 白色(中心)
            else:
                color = (140, 140, 140)

            pt1 = to_canvas(*k[i])
            pt2 = to_canvas(*k[j])
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

        # ---- 画关键点 ----
        for i in range(17):
            if not valid[i]:
                continue
            pt = to_canvas(*k[i])
            # 关键点大小恒定 (不随距离变化 — 已归一化)
            r = 4 if i < 5 else 3          # 头部略大
            if i in (5, 6, 11, 12):
                r = 5                       # 肩髋最大
            cv2.circle(canvas, pt, r, (220, 220, 220), -1)
            cv2.circle(canvas, pt, r + 1, (80, 80, 80), 1)

        # ---- 状态栏 ----
        # 跌倒状态
        s_color = {"safe": (0, 200, 0), "warning": (0, 200, 255), "alert": (0, 0, 255)}
        cv2.putText(canvas, fall_status.upper(), (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color.get(fall_status, (150, 150, 150)), 2)

        # 躯干倾角 (站=0° 躺=90°)
        angle_color = (0, 255, 0) if torso_angle < 30 else ((0, 255, 255) if torso_angle < 55 else (0, 0, 255))
        posture = "STAND" if torso_angle < 30 else ("LYING" if torso_angle > 55 else "TILT")
        cv2.putText(canvas, f"{posture} {torso_angle:.0f} deg", (w - 115, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, angle_color, 1)

        # 朝向
        orient_text = ("FRONT" if orientation_yaw < 20
                       else ("SIDE" if orientation_yaw > 70
                             else f"{orientation_yaw:.0f}d"))
        cv2.putText(canvas, orient_text, (w - 50, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyWindow(self.window_name)
