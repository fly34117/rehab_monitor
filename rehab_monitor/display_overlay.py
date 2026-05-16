"""画面叠加 — 实时显示角度、FPS、跌倒状态"""
import cv2
import time
import numpy as np


class DisplayOverlay:
    """在帧上叠加康复监测信息"""

    # 颜色定义 (BGR)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    GRAY = (180, 180, 180)
    DARK_BG = (40, 40, 40)

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        self.session_start = time.time()
        # 趋势历史缓冲
        self.knee_history = []  # (left_knee, right_knee) 最近 60 个值
        self.MAX_HISTORY = 60

    def render(self, frame, state):
        """主渲染入口

        state: dict with keys:
          fps, person_count, angles (dict), step_count, symmetry, speed,
          fall_status, fall_score, emotion (label, scores),
          show_fps, use_kalman, frame_num
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 顶部状态栏
        self._draw_status_bar(out, state, w)

        # 右侧面板
        self._draw_side_panel(out, state, w, h)

        # 朝向可视化（人体朝向箭头）
        self._draw_orientation_overlay(out, state)

        # 关节角度（直接标在人物旁边）
        self._draw_angle_labels(out, state)

        # 跌倒告警
        if state.get("fall_status") == "alert":
            self._draw_fall_banner(out, w, h)

        return out

    def _draw_status_bar(self, out, state, w):
        """顶部状态栏"""
        session_time = time.time() - self.session_start
        fps = state.get("fps", 0)
        fall_status = state.get("fall_status", "safe")
        frame_num = state.get("frame_num", 0)

        bar = f"Session: {session_time:.0f}s | Frame: {frame_num} | FPS: {fps:.1f}"

        # 背景条
        cv2.rectangle(out, (0, 0), (w, 28), self.DARK_BG, -1)
        cv2.putText(out, bar, (8, 20), self.font, 0.5, self.WHITE, 1)

        # 跌倒状态颜色指示
        status_color = {"safe": self.GREEN, "warning": self.YELLOW, "alert": self.RED}
        color = status_color.get(fall_status, self.WHITE)
        cv2.circle(out, (w - 20, 14), 8, color, -1)

    def _draw_side_panel(self, out, state, w, h):
        """右侧信息面板"""
        panel_x = w - 220
        panel_w = 220

        # 半透明背景
        overlay = out.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

        y = 40
        lh = 22  # 行高

        # ---- 目标状态 ----
        if state.get("target_enrolled"):
            sim = state.get("target_similarity", 0)
            source = state.get("target_match_source", "none")
            if sim > 0:
                tname = state.get("target_name", "target")
                src_tag = "[FACE]" if source == "face" else "[BODY]"
                src_color = self.GREEN if source == "face" else self.CYAN
                cv2.putText(out, f"TARGET: {tname}", (panel_x + 8, y),
                            self.font, 0.5, self.GREEN, 1)
                cv2.putText(out, f"sim: {sim:.2f} {src_tag}", (panel_x + 105, y),
                            self.font_small, 0.4, src_color, 1)
            else:
                cv2.putText(out, "TARGET: SEARCHING...", (panel_x + 8, y),
                            self.font_small, 0.45, self.YELLOW, 1)
            y += lh + 2
            # 参考人脸缩略图
            ref_face = state.get("target_face")
            if ref_face is not None and ref_face.size > 0:
                ts = 36
                thumb = cv2.resize(ref_face, (ts, ts))
                if len(thumb.shape) == 2:
                    thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                out[y:y+ts, panel_x+panel_w-ts-10:panel_x+panel_w-10] = thumb
                y += ts + 4
        else:
            cv2.putText(out, "TARGET: press T", (panel_x + 8, y),
                        self.font_small, 0.45, self.GRAY, 1)
            y += lh

        # ---- 步态指标 ----
        cv2.putText(out, "GAIT METRICS", (panel_x + 8, y), self.font, 0.55, self.CYAN, 1)
        y += lh + 5

        angles = state.get("angles") or {}
        left_knee = angles.get("left_knee")
        right_knee = angles.get("right_knee")

        # ---- 人体朝向 + 躯干倾角 ----
        torso_angle = state.get("torso_angle", 0)
        is_lying = state.get("is_lying_down", False)
        if is_lying:
            lying_type = state.get("lying_type", "none")
            label = "FALL!" if lying_type == "fall" else "REST"
            lcolor = (0, 0, 255) if lying_type == "fall" else (255, 200, 0)
            cv2.putText(out, f"LYING({label}) {torso_angle:.0f}d", (panel_x + 8, y),
                        self.font_small, 0.5, lcolor, 2)
            y += lh

        for label, val, fmt in [
            ("Steps", state.get("step_count", 0), "{}"),
            ("Symmetry", state.get("symmetry", 1.0), "{:.2f}"),
        ]:
            v = val() if callable(val) else val
            txt = f"{label}: --" if v is None else f"{label}: {fmt.format(v)}"
            cv2.putText(out, txt, (panel_x + 8, y), self.font_small, 0.5, self.WHITE, 1)
            y += lh

        # 膝关节趋势线
        if left_knee is not None or right_knee is not None:
            self.knee_history.append((left_knee, right_knee))
            if len(self.knee_history) > self.MAX_HISTORY:
                self.knee_history.pop(0)
        if len(self.knee_history) >= 2:
            self._draw_knee_sparkline(out, panel_x, y, panel_w)
            y += 38

        # ---- 跌倒状态 ----
        y += 8
        cv2.putText(out, "FALL STATUS", (panel_x + 8, y), self.font, 0.55, self.CYAN, 1)
        y += lh + 2

        fall_status = state.get("fall_status", "safe")
        fall_score = state.get("fall_score", 0.0)
        status_color = {"safe": self.GREEN, "warning": self.YELLOW, "alert": self.RED}
        color = status_color.get(fall_status, self.WHITE)

        # 状态框
        cv2.rectangle(out, (panel_x + 8, y), (panel_x + panel_w - 16, y + 30), color, 2)
        cv2.putText(out, fall_status.upper(), (panel_x + 16, y + 22),
                    self.font, 0.7, color, 2)
        y += 38
        cv2.putText(out, f"Score: {fall_score:.2f}", (panel_x + 8, y),
                    self.font_small, 0.5, self.WHITE, 1)
        y += lh
        # 融合标注
        if state.get("fusion_active"):
            cv2.putText(out, "(emotion fusion)", (panel_x + 8, y),
                        self.font_small, 0.4, self.YELLOW, 1)
            y += lh
        y += 5

        # ---- 表情识别 ----
        y += 5
        cv2.putText(out, "EMOTION", (panel_x + 8, y), self.font, 0.55, self.CYAN, 1)
        y += lh + 2

        # 人脸缩略图（固定位置，优先显示）
        face_thumb = state.get("face_thumbnail")
        thumb_y = y
        if face_thumb is not None and face_thumb.size > 0:
            thumb_size = 68
            tx = panel_x + panel_w - thumb_size - 10
            if thumb_y + thumb_size < h - 10:
                cv2.rectangle(out, (tx - 1, thumb_y - 1),
                              (tx + thumb_size + 1, thumb_y + thumb_size + 1),
                              self.CYAN, 1)
                thumb = cv2.resize(face_thumb, (thumb_size, thumb_size))
                out[thumb_y:thumb_y + thumb_size, tx:tx + thumb_size] = thumb
                y = thumb_y + thumb_size + 4
            else:
                face_thumb = None  # 空间不够，跳过缩略图

        emotion_label = state.get("emotion_label", "neutral")
        emotion_scores = state.get("emotion_scores", {})
        emo_color = self.GREEN if emotion_label in ("happy", "neutral") else self.YELLOW
        cv2.putText(out, emotion_label.upper(), (panel_x + 8, y),
                    self.font, 0.65, emo_color, 2)
        y += lh + 2

        # 情绪柱状图 (仅非 neutral 时显示，减少渲染开销)
        if emotion_label != "neutral":
            bar_x = panel_x + 8
            bar_w = panel_w - 24
            bar_max_h = 8
            for cls_name, score in list(emotion_scores.items())[:7]:
                bar_y = y
                filled_w = int((bar_w - 40) * score)
                cv2.putText(out, cls_name[:3].upper(), (bar_x, y + bar_max_h - 1),
                            self.font_small, 0.35, self.GRAY, 1)
                if filled_w > 0:
                    cv2.rectangle(out, (bar_x + 28, bar_y),
                                  (bar_x + 28 + filled_w, bar_y + bar_max_h),
                                  emo_color, -1)
                cv2.rectangle(out, (bar_x + 28, bar_y),
                              (bar_x + 28 + bar_w - 40, bar_y + bar_max_h),
                              self.GRAY, 1)
                y += 12

        # ---- 模型信息 ----
        cv2.putText(out, "MODEL", (panel_x + 8, y), self.font, 0.55, self.CYAN, 1)
        y += lh
        cv2.putText(out, "yolo26n-pose INT8", (panel_x + 8, y),
                    self.font_small, 0.45, self.GRAY, 1)
        y += lh
        kalman = "ON" if state.get("use_kalman") else "OFF"
        cv2.putText(out, f"Kalman: {kalman}", (panel_x + 8, y),
                    self.font_small, 0.45, self.GRAY, 1)

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

    def _draw_knee_sparkline(self, out, px, y, panel_w):
        """绘制膝关节角度迷你趋势线"""
        spark_h = 30
        spark_w = panel_w - 20
        x0 = px + 10
        y0 = y

        # 背景
        cv2.rectangle(out, (x0, y0), (x0 + spark_w, y0 + spark_h), self.DARK_BG, -1)
        cv2.putText(out, "Knee trend", (x0 + 2, y0 + 10),
                    self.font_small, 0.3, self.GRAY, 1)

        # 取有效值
        l_vals = [(i, v[0]) for i, v in enumerate(self.knee_history) if v[0] is not None]
        r_vals = [(i, v[1]) for i, v in enumerate(self.knee_history) if v[1] is not None]

        def draw_line(pts, color):
            if len(pts) < 2:
                return
            xs = np.array([p[0] for p in pts], dtype=np.float32)
            ys = np.array([p[1] for p in pts], dtype=np.float32)
            ys = np.clip(ys, 0, 200)  # 裁剪异常值
            if ys.max() - ys.min() < 0.1:
                ys = ys * 0 + spark_h / 2
            else:
                ys = (ys - ys.min()) / (ys.max() - ys.min()) * (spark_h - 8) + 4
            xs = xs / (self.MAX_HISTORY - 1) * (spark_w - 4) + x0 + 2
            for i in range(len(xs) - 1):
                pt1 = (int(xs[i]), int(y0 + spark_h - ys[i]))
                pt2 = (int(xs[i + 1]), int(y0 + spark_h - ys[i + 1]))
                cv2.line(out, pt1, pt2, color, 1)

        draw_line(l_vals, (255, 150, 0))   # 蓝色-左膝
        draw_line(r_vals, (0, 150, 255))   # 红色-右膝
        cv2.putText(out, "L", (x0 + spark_w - 30, y0 + 10), self.font_small, 0.3, (255, 150, 0), 1)
        cv2.putText(out, "R", (x0 + spark_w - 18, y0 + 10), self.font_small, 0.3, (0, 150, 255), 1)

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
