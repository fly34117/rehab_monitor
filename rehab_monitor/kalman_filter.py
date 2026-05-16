"""卡尔曼滤波器 — 从 openvino_pose_detection_with_kalman.py 抽取"""
import numpy as np


class KalmanPointFilter:
    """针对单个关键点 (x, y) 的卡尔曼滤波器，假设匀速运动模型"""

    def __init__(self, dt=0.04, process_noise=1e-5, measurement_noise=1e-1):#
        self.dt = dt
        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 100
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if not self.initialized and z is not None:
            self.x[:2] = np.array(z).reshape(2, 1)
            self.x[2:] = 0
            self.initialized = True
            return np.array(z)

        z_arr = np.array(z).reshape(2, 1)
        y = z_arr - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()


class PoseKalmanSmoother:
    """为人体多个关键点分别创建卡尔曼滤波器"""

    def __init__(self, num_kpts=17, dt=0.04, process_noise=1e-5,
                 measurement_noise=1e-1, max_persons=5):
        self.num_kpts = num_kpts
        self.max_persons = max_persons
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.filters = [None] * max_persons
        self.unmatched_frames = [0] * max_persons

    def _get_filters_for_person(self, person_id):
        if person_id >= self.max_persons:
            return None
        if self.filters[person_id] is None:
            self.filters[person_id] = [
                KalmanPointFilter(
                    dt=self.dt,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                ) for _ in range(self.num_kpts)
            ]
        return self.filters[person_id]

    def smooth(self, kpts_xy, person_id=0):
        if kpts_xy is None:
            return None
        filters = self._get_filters_for_person(person_id)
        if filters is None:
            return kpts_xy

        smoothed = np.zeros_like(kpts_xy)
        for i, (filter, point) in enumerate(zip(filters, kpts_xy)):
            if point[0] == 0 and point[1] == 0:
                filter.predict()
                smoothed[i] = filter.x[:2].flatten()
            else:
                filter.predict()
                smoothed[i] = filter.update(point)
        return smoothed

    def predict_all(self):
        """预测所有活跃轨道，返回 {track_id: predicted_positions(17,2)}"""
        predicted = {}
        for pid in range(self.max_persons):
            if self.filters[pid] is not None:
                pred = np.zeros((self.num_kpts, 2))
                for j, filt in enumerate(self.filters[pid]):
                    filt.predict()
                    pred[j] = filt.x[:2].flatten()
                predicted[pid] = pred
        return predicted

    def _track_distance(self, det_kpts, pred_kpts):
        """计算检测与预测之间的髋部中心距离"""
        # 髋部关键点 (11, 12)
        det_hip = (det_kpts[11] + det_kpts[12]) / 2
        pred_hip = (pred_kpts[11] + pred_kpts[12]) / 2
        if det_hip[0] > 0 and det_hip[1] > 0 and pred_hip[0] > 0 and pred_hip[1] > 0:
            return float(np.linalg.norm(det_hip - pred_hip))
        # 回退：用所有有效点的平均距离
        valid = (det_kpts[:, 0] > 0) & (pred_kpts[:, 0] > 0)
        if valid.any():
            return float(np.linalg.norm(det_kpts[valid] - pred_kpts[valid], axis=1).mean())
        return float("inf")

    def update_track(self, person_id, kpts_xy):
        """用匹配到的检测更新轨道 (必须先调用 predict_all)"""
        filters = self._get_filters_for_person(person_id)
        if filters is None:
            return kpts_xy
        self.unmatched_frames[person_id] = 0
        smoothed = np.zeros_like(kpts_xy)
        for i, (filt, pt) in enumerate(zip(filters, kpts_xy)):
            if pt[0] == 0 and pt[1] == 0:
                smoothed[i] = filt.x[:2].flatten()
            else:
                smoothed[i] = filt.update(pt)
        return smoothed

    def create_track(self, person_id, kpts_xy):
        """为新出现的检测创建轨道"""
        filters = self._get_filters_for_person(person_id)
        if filters is None:
            return kpts_xy
        self.unmatched_frames[person_id] = 0
        smoothed = np.zeros_like(kpts_xy)
        for i, (filt, pt) in enumerate(zip(filters, kpts_xy)):
            if pt[0] != 0 or pt[1] != 0:
                filt.predict()
                smoothed[i] = filt.update(pt)
            else:
                smoothed[i] = pt
        return smoothed

    def reset_person(self, person_id=0):
        if person_id < self.max_persons:
            self.filters[person_id] = None
            self.unmatched_frames[person_id] = 0
