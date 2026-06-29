"""
性能监控 — 实时追踪 FPS、各阶段延迟、设备负载
=================================================
追踪指标:
  - capture_fps: 摄像头捕获帧率
  - inference_ms: 推理延迟 (平均/最小/最大)
  - pipeline_fps: 端到端管线帧率
  - device_util: 设备使用情况 (OpenVINO 属性)
  - drop_rate: 丢帧率
"""
import time
import threading
import logging
from collections import deque

logger = logging.getLogger("rehab.monitor")


class PerformanceMonitor:
    """实时性能监控器 — 滑动窗口统计"""

    def __init__(self, window_size=60):
        self._lock = threading.Lock()
        self._window = window_size

        # 各阶段计时 (ms)
        self._capture_times = deque(maxlen=window_size)
        self._inference_times = deque(maxlen=window_size)
        self._postprocess_times = deque(maxlen=window_size)
        self._total_times = deque(maxlen=window_size)

        # 帧计数
        self._frame_count = 0
        self._drop_count = 0
        self._start_time = time.time()

        # 当前值
        self._fps = 0.0
        self._last_report = time.time()

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def record_capture(self, dt_ms):
        with self._lock:
            self._capture_times.append(dt_ms)

    def record_inference(self, dt_ms):
        with self._lock:
            self._inference_times.append(dt_ms)

    def record_postprocess(self, dt_ms):
        with self._lock:
            self._postprocess_times.append(dt_ms)

    def record_frame(self, total_ms):
        with self._lock:
            self._total_times.append(total_ms)
            self._frame_count += 1

    def record_drop(self):
        with self._lock:
            self._drop_count += 1

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def _stats(self, data):
        if not data:
            return 0, 0, 0
        return (sum(data) / len(data),
                min(data),
                max(data))

    def snapshot(self):
        """返回当前性能快照"""
        with self._lock:
            elapsed = max(time.time() - self._start_time, 0.001)
            fps = self._frame_count / elapsed
            total_frames = self._frame_count + self._drop_count
            drop_rate = (self._drop_count / total_frames * 100
                        if total_frames > 0 else 0)

            cap_avg, cap_min, cap_max = self._stats(self._capture_times)
            inf_avg, inf_min, inf_max = self._stats(self._inference_times)
            pp_avg, pp_min, pp_max = self._stats(self._postprocess_times)
            tot_avg, tot_min, tot_max = self._stats(self._total_times)

            self._fps = fps
            return {
                "pipeline_fps": round(fps, 1),
                "frame_count": self._frame_count,
                "drop_rate_pct": round(drop_rate, 1),
                "capture_ms":       round(cap_avg, 1),
                "inference_ms":     round(inf_avg, 1),
                "inference_ms_min": round(inf_min, 1),
                "inference_ms_max": round(inf_max, 1),
                "postprocess_ms":   round(pp_avg, 1),
                "total_ms":         round(tot_avg, 1),
                "total_ms_min":     round(tot_min, 1),
                "total_ms_max":     round(tot_max, 1),
            }

    def fps(self):
        return self._fps

    @property
    def frame_count(self):
        return self._frame_count

    # ------------------------------------------------------------------
    # 定时报告
    # ------------------------------------------------------------------

    def periodic_report(self, interval=5.0, device_info=""):
        """每隔 interval 秒打印性能报告"""
        now = time.time()
        if now - self._last_report < interval:
            return
        self._last_report = now

        s = self.snapshot()
        lines = [
            f"━━ 性能报告 ({device_info}) ━━",
            f"  Pipeline: {s['pipeline_fps']:.1f} FPS  "
            f"| 推理: {s['inference_ms']:.0f}ms "
            f"(min={s['inference_ms_min']:.0f} max={s['inference_ms_max']:.0f})",
            f"  采集: {s['capture_ms']:.0f}ms  "
            f"| 后处理: {s['postprocess_ms']:.0f}ms  "
            f"| 丢帧率: {s['drop_rate_pct']:.1f}%",
        ]
        logger.info("\n".join(lines))
