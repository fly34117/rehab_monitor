"""空间定位 — 地平面单应矩阵将像素坐标映射到真实 XY 坐标 (米)"""
import numpy as np
import cv2
from collections import deque
from .logging_setup import get_logger

logger = get_logger("spatial")


class SpatialMapper:
    """通过地平面单应矩阵将图像踝关节像素坐标映射到地面 XY 坐标

    校准方法: 在地面上标记 4 个已知世界坐标的点，记录对应像素坐标。
    未校准时使用简化针孔模型（需提供摄像头高度和俯角）。
    """

    def __init__(self):
        self.H = None          # 3x3 单应矩阵 (像素 → 世界米)
        self.calibrated = False
        self.origin_pixel = None  # 世界原点对应的像素坐标 (用于简易模式)

        # 简易模式参数 (无完整校准时使用)
        self.camera_height_m = 1.5     # 摄像头距地面高度 (米)
        self.camera_tilt_deg = 0.0     # 摄像头俯角 (0=水平, 90=垂直下视)
        self.image_width = 640
        self.image_height = 480

        # 人体参考尺寸
        self.torso_height_m = 0.45     # 髋→肩高度 (米), 用于测距
        self._torso_depth = 0.0        # 人体身高法估算的深度
        self._ground_depth = 0.0       # 地面投影法估算的深度
        self._prev_torso_px = 0.0      # 上一帧躯干像素高度
        self._depth_latched = False     # 是否处于躺下状态
        self._torso_angle_deg = 0.0    # 躯干倾角 (0°=竖直, 90°=水平)

        # 位置历史 (世界坐标轨迹)
        self.position_history = deque(maxlen=600)  # ~20s @30fps
        self.current_pos = (0.0, 0.0)  # (x, y) 米

        # 定位稳定参数
        self.MAX_POS_DELTA = 0.15       # 单帧最大位移 (米), ~5m/s@30fps
        self.TORSO_CHANGE_RATIO = 2.0   # 躯干像素高度突变阈值 (倍)

    # ===== 校准 =====
    def calibrate_from_points(self, image_pts, world_pts):
        """从 4 组对应点计算单应矩阵

        Args:
            image_pts: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 像素坐标
            world_pts: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)] 世界坐标 (米)
        """
        if len(image_pts) < 4 or len(world_pts) < 4:
            logger.warning("需要至少 4 组对应点进行校准")
            return False

        src = np.array(image_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 2)
        self.H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

        if self.H is not None:
            self.calibrated = True
            logger.info("单应矩阵校准成功")
            return True
        logger.warning("校准失败，继续使用简易模型")
        return False

    # ===== 映射 =====
    def pixel_to_world(self, px, py):
        """将像素坐标映射到世界 XY"""
        if self.H is not None:
            pt = np.array([px, py, 1.0])
            wp = self.H @ pt
            wp = wp / wp[2]
            return float(wp[0]), float(wp[1])
        else:
            return self._simple_projection(px, py)

    def _simple_projection(self, px, py):
        """简易针孔投影 — 水平摄像头 + 地平面

        摄像头高度 H，光轴水平（平视）。
        地平线上方不可见地面，下方按 Y = f·H / (y - cy) 计算深度。
        """
        H = self.camera_height_m
        tilt = np.radians(self.camera_tilt_deg)
        h_img, w_img = self.image_height, self.image_width

        # 焦距估计 (假设 60° 水平视场角)
        fov_h = np.radians(60.0)
        f_px = (w_img / 2) / np.tan(fov_h / 2)

        # 主点
        cx, cy = w_img / 2, h_img / 2

        # 地平线像素 Y (光轴水平时在图像中心; 有俯角时上移)
        y_horizon = cy - f_px * np.tan(tilt)

        # 地面点必须在水平线以下
        y_below = py - y_horizon
        if y_below < 2:   # 太靠近水平线, 深度发散
            return 0.0, 0.0

        # 深度: Y = f * H / y_below
        world_Y = f_px * H / y_below

        # 横向: X = Y * (px - cx) / f
        world_X = world_Y * (px - cx) / f_px

        return float(world_X), float(world_Y)

    def _estimate_depth_from_torso(self, kpts):
        """从髋→肩像素高度反推深度 (已知躯干=0.45m)

        原理: 针孔模型中 真实高度/深度 = 像素高度/焦距
              → 深度 = f * 真实高度 / 像素高度
        不依赖摄像头俯角，比地面投影法更鲁棒。
        """
        k = kpts[:, :2]
        valid = (k[:, 0] > 0) & (k[:, 1] > 0)

        if not (valid[5] and valid[6] and valid[11] and valid[12]):
            return None

        shoulder_mid = (k[5] + k[6]) / 2
        hip_mid = (k[11] + k[12]) / 2
        pixel_height = float(np.linalg.norm(shoulder_mid - hip_mid))

        if pixel_height < 8:  # 太远或遮挡
            return None

        # 焦距 (同主投影)
        fov_h = np.radians(60.0)
        f_px = (self.image_width / 2) / np.tan(fov_h / 2)

        depth = f_px * self.torso_height_m / pixel_height
        return depth

    def _hip_ground_depth(self, k, valid):
        """髋关节中点 → 地面投影深度 (仅作粗略参考, 近处地平面不可见时偏大)"""
        if not (valid[11] and valid[12]):
            return None
        hip_mid_x = (k[11, 0] + k[12, 0]) / 2
        hip_mid_y = (k[11, 1] + k[12, 1]) / 2
        _, wy = self.pixel_to_world(hip_mid_x, hip_mid_y)
        if wy > 0.1:
            return float(wy)
        return None

    def _depth_from_shoulder_width(self, k, valid):
        """从肩宽估算深度 — 站立/躺下通用, 不依赖地面投影

        原理: 人体肩宽 ~0.40m, 像素宽度与深度成反比。
        """
        if not (valid[5] and valid[6]):
            return None
        sw_px = float(np.linalg.norm(k[5] - k[6]))
        if sw_px < 8:
            return None
        f_px = (self.image_width / 2) / np.tan(np.radians(60.0) / 2)
        return f_px * 0.40 / sw_px

    # ===== 获取人员位置 =====
    def get_person_position(self, kpts_single):
        """从单人关键点获取地面 XY 位置

        站姿: 躯干身高法 (髋→肩 0.45m) — 不受抬腿/手部动作影响
        躺下: 髋关节地面投影法 — 髋在地面, 投影=真实位置
        横向: 始终用髋部中点
        """
        if kpts_single is None:
            return None

        k = kpts_single[:, :2]
        valid = (k[:, 0] > 0) & (k[:, 1] > 0)

        fov_h = np.radians(60.0)
        f_px = (self.image_width / 2) / np.tan(fov_h / 2)
        cx = self.image_width / 2

        # ---- 三路深度并行计算 ----
        depth_torso = self._estimate_depth_from_torso(kpts_single)
        depth_hip = self._hip_ground_depth(k, valid)
        depth_shoulder = self._depth_from_shoulder_width(k, valid)

        self._torso_depth = depth_torso or 0
        self._ground_depth = depth_hip or 0

        # ---- 躯干像素高度 (用于异常检测) ----
        torso_px = 0.0
        if valid[5] and valid[6] and valid[11] and valid[12]:
            sm = (k[5] + k[6]) / 2
            hm = (k[11] + k[12]) / 2
            torso_px = float(np.linalg.norm(sm - hm))

        # ---- 横向 X: 髋部中点 ----
        if valid[11] and valid[12]:
            hip_mid_x = (k[11, 0] + k[12, 0]) / 2
        elif valid[15] or valid[16]:
            hip_mid_x = k[15, 0] if valid[15] else k[16, 0]
        else:
            return None

        # ---- 躯干倾角: 单帧判定站/躺 ----
        # 肩→髋向量与竖直轴(图像Y向上= [0,-1])的夹角
        # 站立: ≈0-20°  躺下: >55°
        torso_angle_deg = 0.0
        if valid[5] and valid[6] and valid[11] and valid[12]:
            sm = (k[5] + k[6]) / 2
            hm = (k[11] + k[12]) / 2
            torso_vec = sm - hm  # 肩→髋 (图像坐标: Y向下)
            norm_t = float(np.linalg.norm(torso_vec))
            if norm_t > 5:
                # 竖直方向 = [0, -1] (图像上方)
                cos_a = abs(np.dot(torso_vec, np.array([0.0, -1.0]))) / norm_t
                torso_angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, 0.0, 1.0))))
        self._torso_angle_deg = torso_angle_deg

        # ---- 躺下判定: 躯干倾角 ----
        LYING_ANGLE_THRESH = 55.0   # >55° = 水平躺下
        STAND_ANGLE_THRESH = 30.0   # <30° = 竖直站立

        if not self._depth_latched:
            if torso_angle_deg > LYING_ANGLE_THRESH:
                self._depth_latched = True
        else:
            if torso_angle_deg < STAND_ANGLE_THRESH:
                if not hasattr(self, '_stand_confirm'):
                    self._stand_confirm = 0
                self._stand_confirm += 1
                if self._stand_confirm >= 3:
                    self._depth_latched = False
                    self._stand_confirm = 0
            else:
                self._stand_confirm = 0

        # ---- 选择深度 ----
        if self._depth_latched:
            # 躺下期间: 肩宽法 (不受姿态影响) > 髋投影 (近处地平面不可见时偏大)
            if depth_shoulder is not None:
                depth = depth_shoulder
            elif depth_hip is not None:
                depth = depth_hip
            else:
                depth = depth_torso or depth_hip or 3.0
        elif depth_torso is not None:
            # 站立: 躯干身高法 (最准)
            depth = depth_torso
        elif depth_shoulder is not None:
            depth = depth_shoulder
        elif depth_hip is not None:
            depth = depth_hip
        else:
            return None

        world_X = depth * (hip_mid_x - cx) / f_px
        world_Y = depth

        # ---- 位移钳制 ----
        prev_x, prev_y = self.current_pos
        dx = world_X - prev_x
        dy = world_Y - prev_y
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > self.MAX_POS_DELTA and prev_x != 0 and prev_y != 0:
            scale = self.MAX_POS_DELTA / dist
            world_X = prev_x + dx * scale
            world_Y = prev_y + dy * scale

        self._prev_torso_px = torso_px
        self.current_pos = (world_X, world_Y)
        self.position_history.append((world_X, world_Y))
        return (world_X, world_Y)

    @property
    def is_lying_down(self):
        """目标是否处于躺下状态 (躯干水平)"""
        return self._depth_latched

    def set_lying_down(self, state=True):
        """外部强制设置躺下状态 (跌倒检测器触发)"""
        if state and not self._depth_latched:
            self._depth_latched = True

    def get_trajectory(self):
        """返回轨迹数组 [(x, y), ...]"""
        return list(self.position_history)

    def reset(self):
        self.position_history.clear()
        self.current_pos = (0.0, 0.0)
