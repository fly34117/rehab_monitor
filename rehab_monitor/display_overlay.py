"""画面叠加 — 实时显示角度、FPS、跌倒状态"""
import cv2
import time
import numpy as np


class DisplayOverlay:
    """在帧上叠加康复监测信息"""

    # 颜色定义 (BGR)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    GRAY = (180, 180, 180)
    DARK_BG = (40, 40, 40)

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        self.session_start = time.time()

    def render(self, frame, state):
        """主渲染入口 — 只绘制 GUI 没有的原位标注

        state: dict with keys:
          fps, person_count, angles (dict), fall_status, fall_score,
          frame_num, keypoints
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 顶部状态栏（仅帧号 + 耗时，极轻量）
        self._draw_status_bar(out, state, w)

        # 朝向可视化（人体朝向箭头，原位标注）
        self._draw_orientation_overlay(out, state)

        # 关节角度（直接标在人物关节旁，原位标注）
        self._draw_angle_labels(out, state)

        # 跌倒告警横幅
        if state.get("fall_status") == "alert":
            self._draw_fall_banner(out, w, h)

        return out

    def _draw_status_bar(self, out, state, w):
        """顶部状态栏 — 仅帧号+会话时间，极轻量"""
        session_time = time.time() - self.session_start
        frame_num = state.get("frame_num", 0)
        bar = f"Frame: {frame_num} | Time: {session_time:.0f}s"

        cv2.rectangle(out, (0, 0), (w, 24), self.DARK_BG, -1)
        cv2.putText(out, bar, (8, 17), self.font, 0.45, self.GRAY, 1)

    def _draw_orientation_overlay(self, out, state):
        """人体朝向可视化: 肩上朝向箭头 + 左下角俯视图"""
        angles = state.get("angles") or {}
        kpts = state.get("keypoints")
        if kpts is None or len(kpts) == 0:
            return
        yaw = angles.get("orientation_yaw", 0)
        yaw_conf = angles.get("orientation_conf", 0)
        if yaw_conf < 0.2:
            return

        k = kpts[0][:, :2]
        h, w = out.shape[:2]
        valid = (k[:, 0] > 0) & (k[:, 1] > 0)

        if not (valid[5] and valid[6] and valid[11] and valid[12]):
            return

        # ---- 1. 人体上的朝向箭头 (肩中点出发，垂直于肩线) ----
        shoulder_mid = ((k[5] + k[6]) / 2).astype(int)
        shoulder_vec = k[6] - k[5]  # 左→右肩向量 (横向轴)
        # 朝向 = 肩线旋转90° (在2D图像里，指向画面外或内)
        # 用鼻子位置判断前后: 鼻子在肩中点上方 → 面对摄像头
        facing_dir = np.array([-shoulder_vec[1], shoulder_vec[0]])  # 旋转90°
        if valid[0]:  # nose
            nose_to_mid = k[0] - shoulder_mid
            # 如果朝向与鼻子方向点积为负，翻转
            if np.dot(facing_dir, nose_to_mid) < 0:
                facing_dir = -facing_dir
        # 如果鼻子不可见（背身），保持默认朝向
        facing_dir = facing_dir / (np.linalg.norm(facing_dir) + 1e-8)

        # 颜色：按视角质量编码
        sin_yaw = abs(np.sin(np.radians(yaw)))
        if sin_yaw > 0.85:
            arrow_color = (0, 255, 0)       # 绿=侧身(膝角最佳)
        elif sin_yaw > 0.5:
            arrow_color = (0, 255, 255)     # 黄=45°(中等纠正)
        else:
            arrow_color = (0, 165, 255)     # 橙=正对(膝角不可靠)

        # 画肩线
        cv2.line(out, tuple(k[5].astype(int)), tuple(k[6].astype(int)),
                 (200, 200, 200), 2, cv2.LINE_AA)
        # 画朝向箭头
        arrow_len = int(np.linalg.norm(shoulder_vec) * 0.5)
        tip = (shoulder_mid + facing_dir * arrow_len).astype(int)
        cv2.arrowedLine(out, tuple(shoulder_mid), tuple(tip),
                        arrow_color, 3, cv2.LINE_AA, tipLength=0.3)

        # 朝向文字
        if yaw < 20:
            ori_text = "FRONT"
        elif yaw > 70:
            ori_text = "SIDE"
        else:
            ori_text = f"{yaw:.0f} deg"
        cv2.putText(out, ori_text, (tip[0] - 25, tip[1] - 10),
                    self.font_small, 0.5, arrow_color, 2)

    def _draw_angle_labels(self, out, state):
        """在关节位置附近标注角度"""
        angles = state.get("angles") or {}
        kpts = state.get("keypoints")
        if kpts is None or len(kpts) == 0:
            return

        k = kpts[0][:, :2]  # person 0, (17, 2)

        angle_annotations = [
            ("left_knee", "L-Knee", 13),
            ("right_knee", "R-Knee", 14),
            ("left_elbow", "L-Elb", 7),
            ("right_elbow", "R-Elb", 8),
        ]

        for angle_key, label, kpt_idx in angle_annotations:
            angle = angles.get(angle_key)
            if angle is not None:
                x, y = int(k[kpt_idx][0]), int(k[kpt_idx][1])
                if x > 5 and y > 5:
                    txt = f"{label}: {angle:.0f}"
                    (tw, th), _ = cv2.getTextSize(txt, self.font_small, 0.5, 1)
                    cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y), (0, 0, 0), -1)
                    cv2.putText(out, txt, (x + 2, y - 4),
                                self.font_small, 0.5, self.CYAN, 1)

    def _draw_fall_banner(self, out, w, h):
        """跌倒告警横幅"""
        banner_h = 50
        # 红色半透明横幅
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

        # 闪烁效果（基于时间）
        t = time.time()
        if int(t * 2) % 2 == 0:
            cv2.putText(out, "! FALL DETECTED !", (w // 2 - 160, h - 12),
                        self.font, 1.2, self.RED, 3)
