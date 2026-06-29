"""CSI 诊断显示 — OpenCV 替代 tkinter/matplotlib
独立线程中运行, 从队列读取跌倒分数历史并绘制时间线图。
"""
from __future__ import annotations
import queue

import cv2
import numpy as np

from rehab_monitor.logging_setup import get_logger

from .config import FALL_THRESHOLD, CSI_DIAG_WIDTH, CSI_DIAG_HEIGHT

logger = get_logger("csi.diag")


class CSIDisplay:
    """OpenCV 跌倒分数时间线显示 (替代原 matplotlib Tab)

    原理:
      - 从队列读取最近的跌倒分数历史和告警状态
      - 绘制折线图显示分数随时间变化
      - 红色虚线标记阈值
      - 告警时红色边框闪烁

    Args:
        width: 窗口宽度 (像素), 默认 600
        height: 窗口高度 (像素), 默认 240
    """

    def __init__(self, width: int = 0, height: int = 0):
        self.width = width or CSI_DIAG_WIDTH
        self.height = height or CSI_DIAG_HEIGHT
        self._threshold = FALL_THRESHOLD

    def run(self, diag_queue: queue.Queue):
        """诊断显示主循环 (在后台线程中运行, 阻塞)

        Args:
            diag_queue: 每帧数据的队列, 元素为 dict {score, history, alert, frame, fall_count}
        """
        window_name = "CSI Fall Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)

        while True:
            try:
                data = diag_queue.get(timeout=1.0)
            except queue.Empty:
                # 检查窗口是否被关闭
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                continue

            try:
                canvas = self._draw(data)
                cv2.imshow(window_name, canvas)
            except cv2.error:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break

        cv2.destroyWindow(window_name)
        logger.info("CSI 诊断窗口已关闭")

    def _draw(self, data: dict) -> np.ndarray:
        """绘制单帧画面

        Args:
            data: {'score', 'history', 'alert', 'frame', 'fall_count'}

        Returns:
            BGR 图像 (np.ndarray)
        """
        W, H = self.width, self.height
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # 告警时红色背景
        if data.get('alert'):
            canvas[:] = (0, 0, 40)
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (0, 0, 255), 3)

        # ---- 标题栏 ----
        title = f"CSI Fall Monitor | Frame #{data.get('frame', 0)}"
        cv2.putText(canvas, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ---- 当前状态 ----
        current_score = data.get('score', 0.0)
        alert = data.get('alert', False)

        score_text = f"Score: {current_score:.2f}"
        score_color = (0, 0, 255) if alert else (0, 255, 0)
        cv2.putText(canvas, score_text, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, score_color, 2)

        falls_text = f"Falls: {data.get('fall_count', 0)}"
        cv2.putText(canvas, falls_text, (210, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 告警横幅
        if alert:
            banner_text = "FALL DETECTED!"
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            tx = (W - text_size[0]) // 2
            cv2.putText(canvas, banner_text, (tx, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # ---- 时间线图 ----
        history = data.get('history', [])
        if history:
            graph_x = 10
            graph_y = 65
            graph_w = W - 20
            graph_h = H - graph_y - 15

            # 背景
            cv2.rectangle(canvas, (graph_x, graph_y),
                          (graph_x + graph_w, graph_y + graph_h),
                          (30, 30, 30), -1)

            # 阈值线 (红色虚线)
            max_val = max(max(history), self._threshold * 1.5)
            thresh_y = int(graph_y + graph_h
                           - (self._threshold / max_val) * graph_h)
            cv2.line(canvas, (graph_x, thresh_y),
                     (graph_x + graph_w, thresh_y),
                     (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Th={self._threshold:.1f}",
                        (graph_x + graph_w - 70, thresh_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # 绘制分数折线
            points = []
            for i, val in enumerate(history):
                x = graph_x + int(i * graph_w / len(history))
                y = graph_y + graph_h - int((val / max_val) * graph_h)
                points.append((x, min(max(y, graph_y), graph_y + graph_h)))

            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i],
                         (0, 255, 255), 1, cv2.LINE_AA)

            # 最新值标记
            if len(points) > 1:
                color = (0, 0, 255) if alert else (0, 255, 0)
                cv2.circle(canvas, points[-1], 3, color, -1)

            # Y 轴标签
            cv2.putText(canvas, f"{max_val:.1f}",
                        (graph_x + 2, graph_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            cv2.putText(canvas, "0",
                        (graph_x + 2, graph_y + graph_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        return canvas
