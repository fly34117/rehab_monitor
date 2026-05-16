"""角度诊断窗口 — 独立 opencv 窗口显示原始 vs 纠正角度滚动折线图"""
import cv2
import numpy as np
from collections import deque


class AngleMonitor:
    """独立窗口诊断关节角度变化，区分原始噪声与纠正噪声"""

    MAX_HISTORY = 150  # 滚动窗口帧数

    def __init__(self, window_name="Angle Monitor"):
        self.window_name = window_name
        self.history = {
            "left_knee": deque(maxlen=self.MAX_HISTORY),
            "left_knee_raw": deque(maxlen=self.MAX_HISTORY),
            "right_knee": deque(maxlen=self.MAX_HISTORY),
            "right_knee_raw": deque(maxlen=self.MAX_HISTORY),
            "left_elbow": deque(maxlen=self.MAX_HISTORY),
            "left_elbow_raw": deque(maxlen=self.MAX_HISTORY),
            "right_elbow": deque(maxlen=self.MAX_HISTORY),
            "right_elbow_raw": deque(maxlen=self.MAX_HISTORY),
            "yaw": deque(maxlen=self.MAX_HISTORY),
        }
        self.width = 680
        self.height = 540
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def update(self, angles):
        """记录一帧角度数据"""
        for key in self.history:
            val = angles.get(key)
            self.history[key].append(val if val is not None else np.nan)

    def render(self):
        """渲染折线图"""
        canvas = np.full((self.height, self.width, 3), (30, 30, 30), dtype=np.uint8)
        h, w = self.height, self.width

        # 图表区域布局: 4 张 105px + 偏航角 60px
        chart_h = 105
        gap = 4
        margin_t, margin_l, margin_r = 18, 50, 20
        y_curves = margin_t
        chart_w = w - margin_l - margin_r

        # ---- 左膝 ----
        self._draw_chart(canvas, y_curves, margin_l, chart_h, chart_w,
                         self.history["left_knee_raw"], self.history["left_knee"],
                         "Left Knee", (0, 170, 255), (0, 255, 100))
        y_curves += chart_h + gap

        # ---- 右膝 ----
        self._draw_chart(canvas, y_curves, margin_l, chart_h, chart_w,
                         self.history["right_knee_raw"], self.history["right_knee"],
                         "Right Knee", (0, 170, 255), (0, 255, 100))
        y_curves += chart_h + gap

        # ---- 左肘 ----
        self._draw_chart(canvas, y_curves, margin_l, chart_h, chart_w,
                         self.history["left_elbow_raw"], self.history["left_elbow"],
                         "Left Elbow", (0, 170, 255), (255, 200, 0))
        y_curves += chart_h + gap

        # ---- 右肘 ----
        self._draw_chart(canvas, y_curves, margin_l, chart_h, chart_w,
                         self.history["right_elbow_raw"], self.history["right_elbow"],
                         "Right Elbow", (0, 170, 255), (255, 150, 50))
        y_curves += chart_h + gap

        # ---- 偏航角 (底部) ----
        self._draw_yaw_chart(canvas, y_curves, margin_l, 60, chart_w)

        cv2.imshow(self.window_name, canvas)

    def _draw_chart(self, canvas, y0, x0, chart_h, chart_w,
                    raw_deque, corrected_deque, title,
                    raw_color, corrected_color):
        """画一张 (原始 vs 纠正) 对比折线图"""
        h_img, w_img = canvas.shape[:2]
        y1 = y0 + chart_h

        # 背景
        cv2.rectangle(canvas, (x0, y0), (x0 + chart_w, y1), (50, 50, 50), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + chart_w, y1), (100, 100, 100), 1)

        # 标题
        cv2.putText(canvas, title, (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        raw_arr = np.array(raw_deque, dtype=np.float32)
        cor_arr = np.array(corrected_deque, dtype=np.float32)

        # 有效值
        valid_raw = ~np.isnan(raw_arr)
        valid_cor = ~np.isnan(cor_arr)
        if not valid_raw.any() and not valid_cor.any():
            return

        # Y 范围 (110° – 185°)
        y_min, y_max = 110.0, 185.0
        y_span = y_max - y_min

        def to_px(val_arr, idx):
            x = int(x0 + idx * chart_w / self.MAX_HISTORY)
            y = int(y1 - (np.clip(val_arr[idx], y_min, y_max) - y_min) / y_span * chart_h)
            return x, y

        # 画纠正值 (绿色实线，更粗)
        if valid_cor.any():
            pts = []
            for i in range(len(cor_arr)):
                if valid_cor[i]:
                    pts.append(to_px(cor_arr, i))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], corrected_color, 2, cv2.LINE_AA)

        # 画原始值 (橙色虚线，半透明)
        if valid_raw.any():
            pts = []
            for i in range(len(raw_arr)):
                if valid_raw[i]:
                    pts.append(to_px(raw_arr, i))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], raw_color, 1, cv2.LINE_AA)

        # 当前值标注
        label_x = x0 + chart_w - 60
        if valid_cor.any():
            last_cor = cor_arr[valid_cor][-1]
            cv2.putText(canvas, f"{last_cor:.0f}", (label_x, y0 + chart_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, corrected_color, 2)
        if valid_raw.any():
            last_raw = raw_arr[valid_raw][-1]
            cv2.putText(canvas, f"raw:{last_raw:.0f}", (label_x, y0 + chart_h // 2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, raw_color, 1)

        # 图例
        cv2.line(canvas, (x0 + 60, y0 + 12), (x0 + 80, y0 + 12), raw_color, 1)
        cv2.putText(canvas, "raw", (x0 + 83, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, raw_color, 1)
        cv2.line(canvas, (x0 + 110, y0 + 12), (x0 + 130, y0 + 12), corrected_color, 2)
        cv2.putText(canvas, "cor", (x0 + 133, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, corrected_color, 1)

    def _draw_yaw_chart(self, canvas, y0, x0, chart_h, chart_w):
        """偏航角小图"""
        y1 = y0 + chart_h
        cv2.rectangle(canvas, (x0, y0), (x0 + chart_w, y1), (50, 50, 50), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + chart_w, y1), (100, 100, 100), 1)
        cv2.putText(canvas, "Yaw (orientation)", (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        yaw_arr = np.array(self.history["yaw"], dtype=np.float32)
        valid = ~np.isnan(yaw_arr)
        if not valid.any():
            return

        y_min, y_max = 0.0, 90.0
        y_span = y_max - y_min

        pts = []
        for i in range(len(yaw_arr)):
            if valid[i]:
                x = int(x0 + i * chart_w / self.MAX_HISTORY)
                y = int(y1 - (np.clip(yaw_arr[i], y_min, y_max) - y_min) / y_span * chart_h)
                pts.append((x, y))

        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 255), 1, cv2.LINE_AA)

        # 标注阈值线: sin(yaw)=0.35 → yaw≈20°
        thresh_y = int(y1 - 20 / y_span * chart_h)
        cv2.line(canvas, (x0, thresh_y), (x0 + chart_w, thresh_y), (80, 80, 80), 1)
        cv2.putText(canvas, "corr threshold", (x0 + 4, thresh_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        last_yaw = yaw_arr[valid][-1]
        cv2.putText(canvas, f"{last_yaw:.0f} deg", (x0 + chart_w - 50, y0 + chart_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def close(self):
        cv2.destroyWindow(self.window_name)
