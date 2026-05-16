"""轨迹俯视图窗口 — 摄像头在底部中央, Y轴向上延伸为探测方向"""
import cv2
import numpy as np
from collections import deque


class TrajectoryMonitor:
    """独立窗口：俯视轨迹图 (X-Y 平面，米制坐标)

    坐标系: 摄像头位于世界原点 (0,0), 画布底部中央。
    Y轴向上 = 远离摄像头的深度方向。
    """

    MAX_HISTORY = 300           # 保留最近 300 个点
    CAM_FOV_DEG = 70.0          # 摄像头水平视场角

    def __init__(self, window_name="Trajectory (Top View)"):
        self.window_name = window_name
        self.trajectory = deque(maxlen=self.MAX_HISTORY)
        self.current_pos = (0.0, 0.0)
        self.orientation_yaw = 0.0
        self.orientation_conf = 0.0

        self.width = 520
        self.height = 560
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        self.view_range_y = 6.0   # Y 方向显示范围 (米)
        self.view_range_x = 4.0   # X 方向显示范围 (米)

    def update(self, world_pos, trajectory, yaw=0.0, yaw_conf=0.0, facing_sign=1.0,
               depth_body=0.0, depth_ground=0.0):
        self.current_pos = world_pos or (0.0, 0.0)
        self.orientation_yaw = yaw
        self.orientation_conf = yaw_conf
        self.facing_sign = facing_sign
        self.depth_body = depth_body
        self.depth_ground = depth_ground
        if trajectory:
            self.trajectory = deque(trajectory, maxlen=self.MAX_HISTORY)

    def render(self):
        w, h = self.width, self.height
        canvas = np.full((h, w, 3), (22, 22, 28), dtype=np.uint8)

        margin = 45
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin

        # ---- 自适应视图范围 ----
        pts = [(x, y) for x, y in self.trajectory if abs(x) < 50 and abs(y) < 50]
        if pts:
            xs = [abs(p[0]) for p in pts]
            ys = [p[1] for p in pts]
            self.view_range_x = max(max(xs) * 2.2, 2.0)
            self.view_range_y = max(max(ys) * 1.3, 3.0)

        # ---- 坐标系: 摄像头原点在画布底部中央 ----
        cam_sx = w // 2                      # 摄像头 X (屏幕)
        cam_sy = h - margin                  # 摄像头 Y (屏幕, 底部)
        scale_x = plot_w / self.view_range_x  # 像素/米 (X)
        scale_y = plot_h / self.view_range_y  # 像素/米 (Y)

        def to_screen(wx, wy):
            """世界 → 屏幕: 摄像头在 wx=0 屏幕=w/2, wy=0 屏幕=底部"""
            sx = int(cam_sx + wx * scale_x)
            sy = int(cam_sy - wy * scale_y)
            return sx, sy

        # ---- 网格 (X 方向, 世界坐标) ----
        grid_step = 1.0
        grid_color = (45, 45, 50)
        half_x = self.view_range_x / 2
        for g in np.arange(-half_x, half_x + grid_step, grid_step):
            sx, _ = to_screen(g, 0)
            if margin < sx < w - margin:
                cv2.line(canvas, (sx, margin), (sx, cam_sy), grid_color, 1)

        # 网格 (Y 方向, 世界坐标)
        for g in np.arange(0, self.view_range_y + grid_step, grid_step):
            if g < 0.01:
                continue
            _, sy = to_screen(0, g)
            if margin < sy < cam_sy:
                cv2.line(canvas, (margin, sy), (w - margin, sy), grid_color, 1)

        # ---- 摄像头 FOV 锥形 (半透明) ----
        fov_half = np.radians(self.CAM_FOV_DEG / 2)
        fov_left = to_screen(-self.view_range_y * np.tan(fov_half), self.view_range_y)
        fov_right = to_screen(self.view_range_y * np.tan(fov_half), self.view_range_y)

        fov_overlay = canvas.copy()
        fov_pts = np.array([(cam_sx, cam_sy), fov_left, fov_right], dtype=np.int32)
        cv2.fillPoly(fov_overlay, [fov_pts], (35, 40, 50))
        cv2.addWeighted(fov_overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.polylines(canvas, [fov_pts], True, (60, 65, 75), 1, cv2.LINE_AA)

        # ---- 坐标轴 ----
        axis_color = (75, 75, 80)
        # Y 轴 (从摄像头向上)
        cv2.line(canvas, (cam_sx, cam_sy), (cam_sx, margin), axis_color, 2)
        cv2.putText(canvas, "Y (depth)", (cam_sx + 6, margin + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
        # X 轴 (从摄像头左右)
        cv2.line(canvas, (margin, cam_sy), (w - margin, cam_sy), axis_color, 1)

        # X 刻度
        for g in np.arange(-half_x, half_x + grid_step, grid_step):
            if abs(g) < 0.02:
                continue
            sx, _ = to_screen(g, 0)
            if margin + 20 < sx < w - margin - 20:
                cv2.putText(canvas, f"{g:.0f}m", (sx - 12, cam_sy + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (110, 110, 110), 1)
                # 小刻度线
                cv2.line(canvas, (sx, cam_sy - 3), (sx, cam_sy + 3), (75, 75, 80), 1)

        # Y 刻度
        for g in np.arange(grid_step, self.view_range_y + grid_step, grid_step):
            _, sy = to_screen(0, g)
            if sy > margin + 10:
                cv2.putText(canvas, f"{g:.0f}m", (cam_sx + 8, sy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (110, 110, 110), 1)

        # X 标签
        cv2.putText(canvas, "X (m)", (w - 55, cam_sy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)

        # ---- 轨迹线 ----
        valid_pts = [(x, y) for x, y in pts if y >= 0]
        if len(valid_pts) >= 2:
            for i in range(1, len(valid_pts)):
                progress = i / len(valid_pts)
                color = (int(30 + 120 * progress),
                         int(90 + 165 * progress),
                         int(210 * (1 - progress)))
                p1 = to_screen(*valid_pts[i - 1])
                p2 = to_screen(*valid_pts[i])
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

        # ---- 摄像头图标 (底部中央) ----
        cam_icon_r = 8
        cv2.circle(canvas, (cam_sx, cam_sy), cam_icon_r + 2, (50, 55, 65), -1)
        cv2.circle(canvas, (cam_sx, cam_sy), cam_icon_r, (90, 100, 110), -1)
        cv2.circle(canvas, (cam_sx, cam_sy), 4, (160, 180, 200), -1)
        cv2.putText(canvas, "CAM", (cam_sx - 16, cam_sy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 180, 200), 2)

        # ---- 当前位置 + 朝向箭头 ----
        wx, wy = self.current_pos
        if wy > 0.01:  # 只显示在摄像头前方的目标
            pos_pt = to_screen(wx, wy)

            # 朝向角度: 0°=正对摄像头(箭头指向摄像头), yaw增大旋转
            yaw_rad = np.radians(self.orientation_yaw)
            sign = getattr(self, 'facing_sign', 1.0)
            # 从目标指向摄像头
            to_cam_angle = np.arctan2(cam_sy - pos_pt[1], cam_sx - pos_pt[0])
            facing_angle = to_cam_angle + sign * yaw_rad

            arrow_len = max(16, int(scale_x * 0.35))
            tip_x = int(pos_pt[0] + arrow_len * np.cos(facing_angle))
            tip_y = int(pos_pt[1] + arrow_len * np.sin(facing_angle))

            arrow_w = max(3, arrow_len // 2)
            perp = facing_angle + np.pi / 2
            base_x1 = int(pos_pt[0] + arrow_w * np.cos(perp))
            base_y1 = int(pos_pt[1] + arrow_w * np.sin(perp))
            base_x2 = int(pos_pt[0] - arrow_w * np.cos(perp))
            base_y2 = int(pos_pt[1] - arrow_w * np.sin(perp))

            arrow_pts = np.array([(tip_x, tip_y), (base_x1, base_y1), (base_x2, base_y2)], dtype=np.int32)
            cv2.fillPoly(canvas, [arrow_pts], (0, 210, 255))
            cv2.polylines(canvas, [arrow_pts], True, (0, 250, 255), 2)

            # 光晕
            cv2.circle(canvas, pos_pt, 8, (0, 180, 220), 2)
            cv2.circle(canvas, pos_pt, 4, (255, 255, 255), -1)

        # ---- 底部信息栏 ----
        orient_label = ("FRONT" if self.orientation_yaw < 20
                        else ("SIDE" if self.orientation_yaw > 70
                              else f"{self.orientation_yaw:.0f} deg"))
        db = getattr(self, 'depth_body', 0) or 0
        dg = getattr(self, 'depth_ground', 0) or 0
        depth_info = f"  depth: body={db:.1f}m ground={dg:.1f}m" if db > 0 else ""
        cv2.putText(canvas, f"X={wx:.2f}  Y={wy:.2f} m    Facing: {orient_label}{depth_info}",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

        # ---- 比例尺 ----
        bar_len = int(scale_y)  # 1m 像素
        bar_x, bar_y = w - margin + 2, h - margin - 10
        if bar_len > 10:
            cv2.line(canvas, (bar_x, bar_y), (bar_x, bar_y - bar_len), (200, 200, 200), 2)
            cv2.putText(canvas, "1m", (bar_x - 20, bar_y - bar_len // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyWindow(self.window_name)
