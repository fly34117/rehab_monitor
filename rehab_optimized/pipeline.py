"""
heterogeneous pipeline — 三阶段异构计算管线
=============================================
显式分离三个计算阶段，每阶段绑定特定硬件：

  Stage 1  视频解码  → iGPU  (GStreamer VA-API / V4L2 MJPEG)
  Stage 2  模型推理  → iGPU  (OpenVINO GPU) 或 NPU (AI Boost)
  Stage 3  后处理    → CPU   (卡尔曼/步态/跌倒/表情/标注)

设备分配策略 (由 config.STAGE_DEVICE_MAP 控制):
  --gpu  解码=iGPU  推理=iGPU  后处理=CPU  (iGPU 同时处理解码+推理)
  --npu  解码=iGPU  推理=NPU   后处理=CPU  (iGPU+NPU 异构并行)
  --cpu  解码=CPU   推理=CPU   后处理=CPU  (纯CPU模式)

GPU 资源共享: VA-API (media engine) 和 OpenVINO GPU (compute engine)
使用 iGPU 不同子引擎，可同时运行互不阻塞。
"""
from __future__ import annotations

import time
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from .config import (
    FRAME_QUEUE_MAXSIZE, OV_KALMAN_ENABLED,
    KALMAN_DT, KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE,
    KALMAN_MAX_PERSONS, KALMAN_MAX_MATCH_DIST, KALMAN_MAX_UNMATCHED,
)
from .device_selector import DeviceSelector
from .model_loader import ModelLoader
from .gst_camera import create_camera, _check_vaapi_working, _check_gst_vaapi
from .monitor import PerformanceMonitor

logger = logging.getLogger("rehab.pipeline")


# ============================================================
# 管线阶段定义
# ============================================================

class Stage(Enum):
    """管线三阶段"""
    DECODE     = "decode"      # 视频解码
    INFERENCE  = "inference"   # 模型推理
    POSTPROCESS = "postprocess"  # CPU 后处理


@dataclass
class StageInfo:
    """阶段运行状态"""
    name: str
    device: str           # 实际使用的硬件
    target_device: str    # 期望的硬件
    backend: str = ""     # 具体后端 (gstreamer_vaapi / v4l2_mjpeg / openvino_gpu...)
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    is_optimal: bool = True   # 是否达到目标设备


@dataclass
class PipelineTopology:
    """管线拓扑 — 描述三阶段硬件分配"""
    stages: Dict[Stage, StageInfo] = field(default_factory=dict)
    pipeline_device: str = "cpu"  # 主推理设备

    def summary(self) -> str:
        """拓扑摘要"""
        lines = [
            "╔══════════════════════════════════════╗",
            "║  异构计算管线拓扑                    ║",
            "╠══════════════════════════════════════╣",
        ]
        arrows = {
            Stage.DECODE:     "① 解码",
            Stage.INFERENCE:  "② 推理",
            Stage.POSTPROCESS: "③ 后处理",
        }
        for stage in (Stage.DECODE, Stage.INFERENCE, Stage.POSTPROCESS):
            info = self.stages.get(stage)
            if info:
                status = "✓" if info.is_optimal else "△"
                lines.append(
                    f"║ {arrows[stage]:6s} → {info.device:6s}  "
                    f"{status} {info.backend:<20s} ║"
                )
        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)

    def inline(self) -> str:
        """单行摘要"""
        parts = []
        for stage in (Stage.DECODE, Stage.INFERENCE, Stage.POSTPROCESS):
            info = self.stages.get(stage)
            if info:
                parts.append(f"{info.name}→{info.device}")
        return " | ".join(parts)


# ============================================================
# 设备分配策略
# ============================================================

def _resolve_stage_devices(main_device: str, vaapi_available: bool) -> Dict[Stage, StageInfo]:
    """根据主设备和硬件能力，分配每个阶段的设备

    Args:
        main_device: 主推理设备 (npu, gpu, cpu)
        vaapi_available: VA-API 硬件解码是否可用

    Returns:
        {Stage: StageInfo} 每个阶段的设备分配
    """
    stages = {}

    # Stage 1 — 解码
    if vaapi_available:
        stages[Stage.DECODE] = StageInfo(
            name="解码", device="iGPU", target_device="iGPU",
            backend="GStreamer VA-API", is_optimal=True,
        )
    else:
        # 检查是否有 GPU 可用 (即使 VA-API 不可用, MJPEG 也是好的回退)
        has_gpu_hw = main_device in ("gpu", "npu")
        stages[Stage.DECODE] = StageInfo(
            name="解码", device="CPU", target_device="iGPU",
            backend="V4L2 MJPEG" if has_gpu_hw else "V4L2 YUYV",
            is_optimal=False,  # 解码未命中 iGPU → 非最优
        )

    # Stage 2 — 推理
    infer_target = {"npu": "NPU", "gpu": "iGPU", "cpu": "CPU"}.get(main_device, "CPU")
    if main_device == "gpu":
        infer_actual = "iGPU"
    elif main_device == "npu":
        infer_actual = "NPU"
    else:
        infer_actual = "CPU"
    infer_hit_target = (infer_actual == infer_target)
    stages[Stage.INFERENCE] = StageInfo(
        name="推理", device=infer_actual, target_device=infer_target,
        backend=f"OpenVINO {infer_actual}",
        is_optimal=infer_hit_target,
    )

    # Stage 3 — 后处理 (始终 CPU)
    stages[Stage.POSTPROCESS] = StageInfo(
        name="后处理", device="CPU", target_device="CPU",
        backend="NumPy/OpenCV/Kalman",
        is_optimal=True,
    )

    return stages


# ============================================================
# 三阶段异构管线
# ============================================================

class HeterogeneousPipeline:
    """三阶段异构计算管线

    显式分离:
      Stage 1: 视频解码 (iGPU或CPU) — 异步采集线程
      Stage 2: 模型推理 (iGPU或NPU) — OpenVINO
      Stage 3: 后处理     (CPU)       — 卡尔曼/步态/跌倒/标注

    用法:
      pipeline = HeterogeneousPipeline(model_name="yolo11n", device="gpu")
      while True:
          ret, frame = pipeline.stage_decode()           # Stage 1
          if not ret: continue
          annotated, kpts = pipeline.stage_infer(frame)  # Stage 2
          state = pipeline.stage_postprocess(kpts)       # Stage 3 (在 main.py 完成)
    """

    def __init__(self, model_name=None, device=None, camera_id=0,
                 width=640, height=480, fps=30, use_gstreamer=False):
        self.width = width
        self.height = height
        self.target_fps = fps
        self.camera_id = camera_id

        # ---- 硬件检测 ----
        self.selector = DeviceSelector()
        self._vaapi_ok = _check_vaapi_working() and _check_gst_vaapi()

        # ---- 主推理设备 ----
        if device is None:
            device = self.selector.get_best_device()
        self.device = device
        self.pipeline_device = device  # 主设备标识

        # ---- 阶段设备分配 ----
        self._stage_info = _resolve_stage_devices(device, self._vaapi_ok)
        self.topology = PipelineTopology(
            stages=self._stage_info,
            pipeline_device=device,
        )

        # ---- 是否使用 GStreamer 硬件解码 ----
        self._use_gst = use_gstreamer and self._vaapi_ok

        # ---- Stage 1: 摄像头/解码 ----
        self.camera = create_camera(
            use_gstreamer=self._use_gst,
            camera_id=camera_id, width=width, height=height, fps=fps,
        )
        if not self.camera.is_opened:
            raise RuntimeError(f"无法打开摄像头 ID={camera_id}")

        # ---- Stage 2: 模型 ----
        self.model_name = model_name or "yolo11n"
        self.loader = ModelLoader(self.model_name, self.device)
        self.model, self.ov_device, self.model_path = self.loader.load()

        # ---- 卡尔曼滤波 (CPU 后处理的一部分, 但在 Stage 2 内执行) ----
        self.use_kalman = OV_KALMAN_ENABLED
        self.kalman = None
        if self.use_kalman:
            from rehab_monitor.kalman_filter import PoseKalmanSmoother
            self.kalman = PoseKalmanSmoother(
                num_kpts=17, dt=KALMAN_DT,
                process_noise=KALMAN_PROCESS_NOISE,
                measurement_noise=KALMAN_MEASUREMENT_NOISE,
                max_persons=KALMAN_MAX_PERSONS,
            )
            self._det_to_track = {}

        # ---- 性能监控 ----
        self.monitor = PerformanceMonitor()
        self._inference_times = []
        self._lock = threading.Lock()

        # 打印拓扑
        logger.info("管线初始化完成:\n%s", self.topology.summary())
        logger.info("设备拓扑: %s", self.topology.inline())

    # ==================================================================
    # Stage 1 — 视频解码 (iGPU / CPU)
    # ==================================================================

    def stage_decode(self):
        """Stage 1: 从摄像头读取一帧

        Returns: (ret, frame)
        解码设备: iGPU (GStreamer VA-API) 或 CPU (V4L2 MJPEG/YUYV)
        """
        t0 = time.time()
        ret, frame, cap_ts = self.camera.read()
        dt_ms = (time.time() - t0) * 1000
        self.monitor.record_capture(dt_ms)

        if not ret or frame is None:
            self.monitor.record_drop()
            return False, None, 0

        return True, frame, cap_ts

    # ==================================================================
    # Stage 2 — 模型推理 (iGPU / NPU / CPU)
    # ==================================================================

    def stage_infer(self, frame):
        """Stage 2: 模型推理 + 卡尔曼平滑

        Args:
            frame: BGR 图像 (np.ndarray, 来自 Stage 1)

        Returns: (annotated_frame, keypoints_array)
            推理设备: iGPU (OpenVINO GPU) / NPU / CPU
        """
        t_total = time.time()

        # ---- 2a. 模型推理 (iGPU/NPU) ----
        t0 = time.time()
        results = self.model(frame, device=self.ov_device, verbose=False)
        inf_ms = (time.time() - t0) * 1000
        self.monitor.record_inference(inf_ms)
        self._inference_times.append(inf_ms)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        # ---- 2b. 提取关键点 (CPU) ----
        raw = None
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            raw = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        # ---- 2c. 卡尔曼平滑 (CPU) ----
        smoothed = None
        if self.use_kalman and raw is not None:
            smoothed = self._kalman_smooth(raw)
            import torch
            results[0].keypoints.data = torch.from_numpy(smoothed)
        elif raw is not None:
            smoothed = raw

        # ---- 2d. 生成标注帧 (CPU) ----
        annotated = results[0].plot()
        pp_ms = (time.time() - t_total) * 1000
        self.monitor.record_postprocess(pp_ms)

        total_ms = (time.time() - t_total) * 1000
        self.monitor.record_frame(total_ms)

        # 更新阶段统计
        info = self._stage_info[Stage.INFERENCE]
        info.avg_ms = self.avg_inference_ms

        return annotated, smoothed

    # ==================================================================
    # 卡尔曼滤波 (CPU, Stage 2 内)
    # ==================================================================

    def _kalman_smooth(self, raw):
        """基于髋部中心距离的多目标卡尔曼匹配"""
        ks = self.kalman
        num_det = raw.shape[0]
        predicted = ks.predict_all()

        if not predicted:
            smoothed = np.zeros_like(raw)
            for i in range(min(num_det, KALMAN_MAX_PERSONS)):
                smoothed[i, :, :2] = ks.create_track(i, raw[i, :, :2])
                smoothed[i, :, 2] = raw[i, :, 2]
            return smoothed

        track_ids = list(predicted.keys())
        cost = np.full((num_det, len(track_ids)), np.inf)
        for di in range(num_det):
            for ti, tid in enumerate(track_ids):
                cost[di, ti] = ks._track_distance(raw[di, :, :2], predicted[tid])

        pairs = []
        for di in range(num_det):
            for ti, tid in enumerate(track_ids):
                if cost[di, ti] < KALMAN_MAX_MATCH_DIST:
                    pairs.append((cost[di, ti], di, tid))
        pairs.sort(key=lambda x: x[0])

        assignments = {}
        used_det = set()
        for _, di, tid in pairs:
            if di in used_det or tid in assignments:
                continue
            assignments[tid] = di
            used_det.add(di)

        smoothed = np.zeros_like(raw)
        det_to_track = {}
        for tid, di in assignments.items():
            smoothed[di, :, :2] = ks.update_track(tid, raw[di, :, :2])
            smoothed[di, :, 2] = raw[di, :, 2]
            det_to_track[di] = tid

        unmatched_det = set(range(num_det)) - used_det
        for di in unmatched_det:
            free = None
            for slot in range(KALMAN_MAX_PERSONS):
                if ks.filters[slot] is None:
                    free = slot
                    break
            if free is not None:
                smoothed[di, :, :2] = ks.create_track(free, raw[di, :, :2])
                smoothed[di, :, 2] = raw[di, :, 2]
                det_to_track[di] = free
            else:
                smoothed[di] = raw[di]

        for tid in track_ids:
            if tid not in assignments:
                ks.unmatched_frames[tid] += 1
                if ks.unmatched_frames[tid] >= KALMAN_MAX_UNMATCHED:
                    ks.reset_person(tid)

        self._det_to_track = det_to_track
        return smoothed

    # ==================================================================
    # Stage 3 — 后处理 (CPU) — 在 main.py 中完成
    # ==================================================================
    # Stage 3 的步态分析、跌倒检测、表情识别逻辑在 main.py 中，
    # 因为需要 GaitAnalyzer, FallDetector 等外部模块。
    # 这里仅提供管线状态数据结构。

    @staticmethod
    def stage_postprocess_status() -> StageInfo:
        """返回 Stage 3 的设备信息"""
        return StageInfo(
            name="后处理", device="CPU", target_device="CPU",
            backend="步态/跌倒/表情/API", is_optimal=True,
        )

    # ==================================================================
    # 设备热切换
    # ==================================================================

    def switch_device(self, new_device):
        """动态切换到新推理设备 (Stage 2 重载模型)"""
        if new_device == self.device:
            return
        if not self.selector.is_available(new_device):
            raise ValueError(f"设备 {new_device} 不可用")

        logger.info("设备切换: %s → %s", self.device.upper(), new_device.upper())
        self.device = new_device
        self.loader = ModelLoader(self.model_name, self.device)
        self.model, self.ov_device, self.model_path = self.loader.load()
        self._inference_times = []

        # 更新拓扑
        self._stage_info = _resolve_stage_devices(new_device, self._vaapi_ok)
        self.topology = PipelineTopology(
            stages=self._stage_info,
            pipeline_device=new_device,
        )
        logger.info("切换完成:\n%s", self.topology.summary())

    # ==================================================================
    # 状态查询
    # ==================================================================

    @property
    def avg_inference_ms(self):
        if not self._inference_times:
            return 0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def target_track_id(self):
        return getattr(self, '_det_to_track', {}) or None

    @target_track_id.setter
    def target_track_id(self, value):
        pass

    def get_state(self):
        """返回当前管线状态 (用于日志/API)"""
        return {
            "device": self.device,
            "model": self.model_name,
            "precision": self.loader.precision,
            "kalman": self.use_kalman,
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "topology": {
                "decode": self._stage_info[Stage.DECODE].device,
                "inference": self._stage_info[Stage.INFERENCE].device,
                "postprocess": "CPU",
            },
            **self.monitor.snapshot(),
        }

    def report(self):
        """打印完整管线报告"""
        s = self.monitor.snapshot()
        return "\n".join([
            self.topology.summary(),
            "",
            self.selector.report(),
            self.loader.report(),
            f"摄像头: {'GStreamer VA-API' if self._use_gst else 'V4L2 MJPEG/YUYV'}",
            f"卡尔曼: {'ON' if self.use_kalman else 'OFF'}",
            f"",
            f"── 性能 ──",
            f"  管线 FPS: {s['pipeline_fps']}",
            f"  Stage1 解码: {s['capture_ms']}ms",
            f"  Stage2 推理: {s['inference_ms']}ms (min={s['inference_ms_min']} max={s['inference_ms_max']})",
            f"  Stage3 后处理: {s['postprocess_ms']}ms",
            f"  总延迟: {s['total_ms']}ms  丢帧率: {s['drop_rate_pct']}%",
        ])

    # ==================================================================
    # 资源释放
    # ==================================================================

    def release(self):
        self.camera.release()
        if self.kalman:
            self.kalman = None
        logger.info("管线已释放")

    def __del__(self):
        self.release()
