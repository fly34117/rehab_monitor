"""步态临床指标可视化 — 步长/步速/步宽/支撑相/KneeROM/CV/足廓清/GRS评分"""
import cv2
import numpy as np
from collections import deque


class GaitMetricsViewer:
    MAX_HISTORY = 200

    def __init__(self, window_name="Gait Metrics"):
        self.window_name = window_name
        self.history = {
            "stride_length_m": deque(maxlen=self.MAX_HISTORY),
            "gait_velocity_mps": deque(maxlen=self.MAX_HISTORY),
            "avg_step_width_m": deque(maxlen=self.MAX_HISTORY),
            "stance_percentage": deque(maxlen=self.MAX_HISTORY),
            "left_knee_rom": deque(maxlen=self.MAX_HISTORY),
            "right_knee_rom": deque(maxlen=self.MAX_HISTORY),
            "step_length_cv": deque(maxlen=self.MAX_HISTORY),
            "foot_clearance_cm": deque(maxlen=self.MAX_HISTORY),
            "is_double_support": deque(maxlen=self.MAX_HISTORY),
            "gait_rehab_score": deque(maxlen=self.MAX_HISTORY),
        }
        self.width = 620
        self.height = 640
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def update(self, metrics):
        for key in self.history:
            val = metrics.get(key, None)
            if key == "is_double_support":
                self.history[key].append(1.0 if val else 0.0)
            else:
                self.history[key].append(float(val) if val is not None else np.nan)

    def reset(self):
        """Clear all gait metric history."""
        for values in self.history.values():
            values.clear()

    def render(self):
        w, h = self.width, self.height
        canvas = np.full((h, w, 3), (22, 22, 28), dtype=np.uint8)

        ml, mr = 55, 20
        ch, gap = 48, 2
        cw = w - ml - mr
        y = 8

        # 1. Stride Length
        self._draw_chart(canvas, y, ml, ch, cw,
                         self.history["stride_length_m"],
                         "Stride Length (m)", (100, 255, 150), 0.0, 2.0)
        y += ch + gap

        # 2. Gait Velocity
        self._draw_chart(canvas, y, ml, ch, cw,
                         self.history["gait_velocity_mps"],
                         "Gait Velocity (m/s)", (100, 200, 255), 0.0, 2.5)
        y += ch + gap

        # 3. Step Width
        self._draw_chart(canvas, y, ml, ch, cw,
                         self.history["avg_step_width_m"],
                         "Step Width (m)", (200, 180, 100), 0.0, 0.5)
        y += ch + gap

        # 4. Stance %
        self._draw_chart(canvas, y, ml, ch, cw,
                         self.history["stance_percentage"],
                         "Stance %", (180, 220, 255), 30.0, 80.0)
        y += ch + gap

        # 5. Knee ROM (L/R)
        self._draw_dual_chart(canvas, y, ml, ch, cw,
                              self.history["left_knee_rom"],
                              self.history["right_knee_rom"],
                              "Knee ROM (deg)", (100, 255, 200), (200, 150, 255), 0.0, 80.0)
        y += ch + gap

        # 6. Step Length CV
        self._draw_chart(canvas, y, ml, ch, cw,
                         self.history["step_length_cv"],
                         "Step Length CV (%)", (255, 150, 100), 0.0, 50.0)
        y += ch + gap

        # 7. Foot Clearance
        self._draw_chart(canvas, y, ml, ch, cw,
                         self.history["foot_clearance_cm"],
                         "Foot Clearance (cm)", (200, 255, 150), 0.0, 30.0)
        y += ch + gap

        # 8. Double Support
        self._draw_ds_chart(canvas, y, ml, 40, cw,
                            self.history["is_double_support"])
        y += 40 + gap

        # 9. GRS Score (大数字)
        self._draw_grs(canvas, y, ml, 40, cw,
                       self.history["gait_rehab_score"])
        y += 40 + gap

        cv2.imshow(self.window_name, canvas)

    def _draw_chart(self, canvas, y0, x0, ch, cw, data_deque,
                    title, color, y_min, y_max):
        y1 = y0 + ch
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (40, 40, 43), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (80, 80, 85), 1)
        cv2.putText(canvas, title, (x0 + 4, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        arr = np.array(data_deque, dtype=np.float32)
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return

        ys = y_max - y_min
        pts = [(int(x0 + i * cw / self.MAX_HISTORY),
                int(y1 - (np.clip(arr[i], y_min, y_max) - y_min) / ys * ch))
               for i in range(len(arr)) if valid[i]]

        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], color, 1, cv2.LINE_AA)

        last = arr[valid][-1]
        cv2.putText(canvas, f"{last:.1f}", (x0 + cw - 35, y0 + ch//2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    def _draw_dual_chart(self, canvas, y0, x0, ch, cw, d1, d2,
                         title, c1, c2, y_min, y_max):
        y1 = y0 + ch
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (40, 40, 43), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (80, 80, 85), 1)
        cv2.putText(canvas, title, (x0 + 4, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(canvas, "L", (x0 + cw - 30, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, c1, 1)
        cv2.putText(canvas, "R", (x0 + cw - 15, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, c2, 1)

        for data, color in [(d1, c1), (d2, c2)]:
            arr = np.array(data, dtype=np.float32)
            valid = ~np.isnan(arr)
            if valid.sum() < 2:
                continue
            ys = y_max - y_min
            pts = [(int(x0 + i * cw / self.MAX_HISTORY),
                    int(y1 - (np.clip(arr[i], y_min, y_max) - y_min) / ys * ch))
                   for i in range(len(arr)) if valid[i]]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], color, 1, cv2.LINE_AA)

    def _draw_ds_chart(self, canvas, y0, x0, ch, cw, data_deque):
        y1 = y0 + ch
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (40, 40, 43), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (80, 80, 85), 1)
        cv2.putText(canvas, "Double Support", (x0 + 4, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        arr = np.array(data_deque, dtype=np.float32)
        if len(arr) < 1:
            return
        bar_w = max(1, cw / self.MAX_HISTORY)
        for i in range(len(arr)):
            if arr[i] > 0.5:
                x = int(x0 + i * cw / self.MAX_HISTORY)
                cv2.rectangle(canvas, (x, y0 + ch//2),
                              (int(x + bar_w), y1 - 2), (0, 220, 200), -1)
        ratio = arr.mean() if len(arr) > 0 else 0
        cv2.putText(canvas, f"{ratio:.0%}", (x0 + cw - 35, y0 + ch//2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 2)

    def _draw_grs(self, canvas, y0, x0, ch, cw, data_deque):
        y1 = y0 + ch
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (40, 40, 43), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y1), (80, 80, 85), 1)
        cv2.putText(canvas, "Gait Rehab Score (GRS)", (x0 + 4, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        arr = np.array(data_deque, dtype=np.float32)
        valid = ~np.isnan(arr)
        if valid.sum() < 1:
            return

        last = arr[valid][-1]
        # 颜色: <40红, 40-70黄, >70绿
        color = (0, 0, 255) if last < 40 else ((0, 255, 255) if last < 70 else (0, 255, 0))

        # 分数条
        bar_w = int(cw * last / 100.0)
        cv2.rectangle(canvas, (x0, y0 + ch//2 + 2), (x0 + bar_w, y1 - 4), color, -1)

        cv2.putText(canvas, f"{last:.0f}/100", (x0 + cw - 55, y0 + ch//2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def close(self):
        cv2.destroyWindow(self.window_name)
