"""姿态检测器 — 基于 OpenVINO + 卡尔曼滤波"""
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import (
    POSE_MODEL_PATH, USE_KALMAN,
    KALMAN_DT, KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE, KALMAN_MAX_PERSONS,
    KALMAN_MAX_MATCH_DIST, KALMAN_MAX_UNMATCHED,
)
from .kalman_filter import PoseKalmanSmoother
from .logging_setup import get_logger

logger = get_logger("pose")


class PoseDetector:
    """OpenVINO 姿态检测器，内置卡尔曼滤波"""

    def __init__(self, model_path=None, device="cpu", use_kalman=True):
        model_path = model_path or POSE_MODEL_PATH
        logger.info("加载模型: %s", model_path)
        self.model = YOLO(model_path)
        dev_lower = device.lower().replace("intel:", "")
        _DEVICE_MAP = {"cpu": "intel:cpu", "gpu": "intel:gpu", "npu": "intel:NPU"}
        self.device = _DEVICE_MAP.get(dev_lower, device)
        logger.info("推理设备: %s", self.device)

        self.use_kalman = use_kalman
        self.kalman_smoother = None
        if use_kalman:
            logger.info("卡尔曼滤波已启用 (dt=%s, Q=%s, R=%s)",
                        KALMAN_DT, KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE)
            self.kalman_smoother = PoseKalmanSmoother(
                num_kpts=17, dt=KALMAN_DT,
                process_noise=KALMAN_PROCESS_NOISE,
                measurement_noise=KALMAN_MEASUREMENT_NOISE,
                max_persons=KALMAN_MAX_PERSONS,
            )

        self.inference_times = []
        self.kalman_times = []

        # 多人目标锁定：空间轨迹追踪
        self.target_track_id = None   # 录入目标所在的卡尔曼轨道 ID
        self._det_to_track = {}       # 当前帧: detection_index → track_id

    def process_frame(self, frame):
        """处理单帧，返回 (annotated_frame, keypoints)

        keypoints shape: (num_persons, 17, 3) — [x, y, conf] 平滑后
        无人时返回 (annotated_frame, None)
        """
        # 推理
        t0 = time.time()
        results = self.model(frame, device=self.device, verbose=False)
        inf_time = (time.time() - t0) * 1000
        self.inference_times.append(inf_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        # 提取原始关键点
        raw = None
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            raw = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        # 卡尔曼平滑 — 基于空间距离匹配检测到轨道
        smoothed = None
        if self.use_kalman and raw is not None:
            t0 = time.time()
            smoothed = self._match_and_smooth(raw)
            kt = (time.time() - t0) * 1000
            self.kalman_times.append(kt)
            if len(self.kalman_times) > 100:
                self.kalman_times.pop(0)
            results[0].keypoints.data = torch.from_numpy(smoothed)
        elif raw is not None:
            smoothed = raw

        annotated = results[0].plot()
        return annotated, smoothed

    @property
    def avg_inference_ms(self):
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)

    @property
    def avg_kalman_ms(self):
        if not self.kalman_times:
            return 0
        return sum(self.kalman_times) / len(self.kalman_times)

    def reset_tracking(self):
        if self.kalman_smoother:
            for i in range(KALMAN_MAX_PERSONS):
                self.kalman_smoother.reset_person(i)
            self.target_track_id = None
            self._det_to_track = {}
            logger.info("卡尔曼跟踪状态已重置")

    def get_target_det_idx(self):
        """返回当前帧中目标轨道对应的检测索引，若目标轨道已失效则返回 None"""
        if self.target_track_id is None:
            return None
        for det_idx, track_id in self._det_to_track.items():
            if track_id == self.target_track_id:
                return det_idx
        return None

    def _match_and_smooth(self, raw):
        """基于髋部中心距离将检测匹配到卡尔曼轨道，消除多人交叉干扰"""
        ks = self.kalman_smoother
        num_det = raw.shape[0]

        # 1. 预测所有活跃轨道，获取预测位置
        predicted = ks.predict_all()

        if not predicted:
            # 无活跃轨道 → 为每个检测建立新轨道
            smoothed = np.zeros_like(raw)
            for i in range(min(num_det, KALMAN_MAX_PERSONS)):
                smoothed[i, :, :2] = ks.create_track(i, raw[i, :, :2])
                smoothed[i, :, 2] = raw[i, :, 2]
            return smoothed

        # 2. 构造代价矩阵 (检测 × 轨道)
        track_ids = list(predicted.keys())
        cost = np.full((num_det, len(track_ids)), np.inf)
        for di in range(num_det):
            for ti, tid in enumerate(track_ids):
                cost[di, ti] = ks._track_distance(raw[di, :, :2], predicted[tid])

        # 3. 贪心匹配 (按代价升序分配)
        assignments = {}       # track_id → det_idx
        used_detections = set()
        pairs = []
        for di in range(num_det):
            for ti, tid in enumerate(track_ids):
                if cost[di, ti] < KALMAN_MAX_MATCH_DIST:
                    pairs.append((cost[di, ti], di, tid))
        pairs.sort(key=lambda x: x[0])

        for _, di, tid in pairs:
            if di in used_detections or tid in assignments:
                continue
            assignments[tid] = di
            used_detections.add(di)

        # 4. 更新匹配的轨道
        smoothed = np.zeros_like(raw)
        det_to_track = {}  # detection_index → track_id
        for tid, di in assignments.items():
            smoothed[di, :, :2] = ks.update_track(tid, raw[di, :, :2])
            smoothed[di, :, 2] = raw[di, :, 2]
            det_to_track[di] = tid

        # 5. 未匹配检测 → 新轨道
        unmatched_det = set(range(num_det)) - used_detections
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
                smoothed[di] = raw[di]  # 无空闲槽位，回退原始值

        # 6. 未匹配轨道 → 递增计数，超时释放
        for tid in track_ids:
            if tid not in assignments:
                ks.unmatched_frames[tid] += 1
                if ks.unmatched_frames[tid] >= KALMAN_MAX_UNMATCHED:
                    ks.reset_person(tid)

        self._det_to_track = det_to_track
        return smoothed
