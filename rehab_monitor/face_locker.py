"""Round 2: 基于 FaceNet 的目标锁定器 — 人脸识别 + 匹配"""
import os
import json
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from .logging_setup import get_logger

logger = get_logger("face")


class FaceNetLocker:
    """FaceNet 人脸特征提取 + 特征库匹配"""

    def __init__(self, db_path="model/face_db.json", device=None):
        self.db_path = db_path
        self.requested_device = (device or "cpu").lower()
        self.device = None
        self.resnet = None
        self.mtcnn = None
        self.ov_core = None
        self.ov_infer = None
        self.ov_input_name = None
        self.ov_device = None
        self.db = {"entries": []}
        self._loaded = False
        self._ov_ready = False
        self.reference_face = None       # 参考人脸缩略图
        self.reference_body_feat = None  # 参考人体外观特征
        self.face_threshold = 0.55
        self.body_threshold = 0.70       # 人体外观余弦相似度阈值

    def _ensure_loaded(self):
        if self._loaded:
            return
        self.device = torch.device("cpu")
        logger.info("设备: %s, 加载 FaceNet...", self.requested_device)
        self.mtcnn = MTCNN(keep_all=False, device=self.device,
                           thresholds=[0.5, 0.6, 0.6])
        self._try_load_openvino()
        if not self._ov_ready:
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            logger.info("FaceNet embedding 使用 PyTorch CPU")
        self._load_db()
        self._loaded = True

    def _try_load_openvino(self):
        """优先使用 OpenVINO FaceNet IR；MTCNN 仍保留 CPU。"""
        if self.requested_device not in ("npu", "gpu", "cpu"):
            return
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(
            project_root,
            "model",
            "facenet_ir",
            "facenet_inceptionresnetv1.xml",
        )
        if not os.path.exists(xml_path):
            return
        try:
            import openvino as ov
            self.ov_core = ov.Core()
            device_map = {"npu": "NPU", "gpu": "GPU", "cpu": "CPU"}
            ov_device = device_map[self.requested_device]
            if ov_device not in self.ov_core.available_devices:
                logger.warning("OpenVINO %s 不可用，FaceNet 回退 PyTorch CPU", ov_device)
                return
            ov_model = self.ov_core.read_model(xml_path)
            ov_model.reshape({ov_model.input(0): ov.PartialShape([1, 3, 160, 160])})
            compiled = self.ov_core.compile_model(ov_model, ov_device)
            self.ov_infer = compiled.create_infer_request()
            self.ov_input_name = compiled.input(0).get_any_name()
            self.ov_device = ov_device
            self._ov_ready = True
            logger.info("FaceNet embedding 使用 OpenVINO %s: %s", ov_device, xml_path)
        except Exception as e:
            logger.warning("OpenVINO FaceNet 加载失败，回退 PyTorch CPU: %s", str(e)[:160])
            self.ov_core = None
            self.ov_infer = None
            self.ov_input_name = None
            self.ov_device = None
            self._ov_ready = False

    def _embedding_from_face_tensor(self, face_tensor):
        """从 MTCNN 对齐后的 face_tensor 提取 512 维嵌入。"""
        if face_tensor is None:
            return None
        if self._ov_ready:
            x = face_tensor.detach().cpu().numpy().astype(np.float32)
            if x.ndim == 3:
                x = np.expand_dims(x, axis=0)
            out = self.ov_infer.infer({self.ov_input_name: x})
            return next(iter(out.values())).squeeze().astype(np.float32)

        with torch.no_grad():
            emb = self.resnet(face_tensor.unsqueeze(0).to(self.device))
            return emb.cpu().squeeze().numpy()

    def _mtcnn_face_tensor(self, rgb):
        """兼容 numpy/PIL 输入，返回 MTCNN 对齐后 tensor。"""
        try:
            return self.mtcnn(Image.fromarray(rgb))
        except Exception:
            return self.mtcnn(rgb)

    def _load_db(self):
        if not os.path.exists(self.db_path):
            self.db = {"entries": []}
            logger.info("特征库: 0 人")
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.db = loaded if isinstance(loaded, dict) else {"entries": []}
            if not isinstance(self.db.get("entries"), list):
                self.db["entries"] = []
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("特征库读取失败，已重置为空库: %s", e)
            self.db = {"entries": []}
        logger.info("特征库: %d 人", len(self.db.get("entries", [])))

    def extract_embedding_from_frame(self, frame_bgr):
        """从完整帧中提取 FaceNet 嵌入（用 MTCNN 检测+对齐）"""
        self._ensure_loaded()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            face_tensor = self._mtcnn_face_tensor(rgb)
            if face_tensor is None:
                return None, None
        except Exception as e:
            logger.debug("MTCNN 全帧检测失败: %s", e)
            return None, None
        emb = self._embedding_from_face_tensor(face_tensor)
        return emb, face_tensor

    def extract_embedding_from_roi(self, frame_bgr, x1, y1, x2, y2):
        """从帧中裁剪区域提取 FaceNet 嵌入（COCO粗定位+MTCNN精对齐）

        frame_bgr: 原始帧
        x1,y1,x2,y2: COCO关键点给出的人脸区域
        远距离时自动放大裁剪区域以帮助 MTCNN 检测
        """
        self._ensure_loaded()
        h, w = frame_bgr.shape[:2]
        face_w = max(x2 - x1, 10)
        face_h = max(y2 - y1, 10)
        # 扩大裁剪区域：近距离给 1.5x 边距，远距离用更大比例（至少 80px 边距）
        pad_x = max(int(face_w * 1.5), 80)
        pad_y = max(int(face_h * 1.5), 80)
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(w, x2 + pad_x)
        y2c = min(h, y2 + pad_y)
        if x2c <= x1c or y2c <= y1c:
            return None
        crop = frame_bgr[y1c:y2c, x1c:x2c]
        # 如果裁剪区域仍太小（< 80px），用全帧
        if crop.shape[1] < 80 or crop.shape[0] < 80:
            crop = frame_bgr

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # 限制 MTCNN 输入在 160~250px：小脸放大，大脸缩小，减少金字塔层数 (68ms→8-16ms)
        min_size, max_size = 160, 250
        if rgb.shape[0] < min_size or rgb.shape[1] < min_size:
            scale = min_size / min(rgb.shape[0], rgb.shape[1])
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)))
        elif rgb.shape[0] > max_size or rgb.shape[1] > max_size:
            scale = max_size / max(rgb.shape[0], rgb.shape[1])
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)))

        try:
            face_tensor = self._mtcnn_face_tensor(rgb)
        except Exception as e:
            logger.debug("MTCNN ROI 检测失败: %s", e)
            return None

        if face_tensor is None:
            return None

        return self._embedding_from_face_tensor(face_tensor)

    def extract_body_features(self, frame_bgr, kpts):
        """从关键点提取人体外观特征 — 上下半身分区的 HSV 颜色直方图

        上半身(肩→躯干中点) + 下半身(中点→髋) 分别捕捉衣服/裤子颜色，
        对侧身/背身鲁棒。

        Returns:
            归一化特征向量 (513 dims) 或 None
        """
        if kpts is None:
            return None
        k = kpts[:, :2]
        torso_pts = k[[5, 6, 11, 12]]
        valid = torso_pts[(torso_pts[:, 0] > 0) & (torso_pts[:, 1] > 0)]
        if len(valid) < 2:
            return None

        x1, y1 = valid.min(axis=0).astype(int)
        x2, y2 = valid.max(axis=0).astype(int)
        h_img, w_img = frame_bgr.shape[:2]
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.3)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w_img, x2 + pad_x)
        y2 = min(h_img, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        torso = frame_bgr[y1:y2, x1:x2]
        mid = (y2 - y1) // 2
        upper = torso[:mid, :]
        lower = torso[mid:, :]

        def hsv_hist(roi):
            if roi.size == 0:
                return np.zeros(8 * 8 * 4, dtype=np.float32)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4],
                             [0, 180, 0, 256, 0, 256])
            h = cv2.normalize(h, h).flatten()
            return h

        hist_upper = hsv_hist(upper)
        hist_lower = hsv_hist(lower)

        # 宽高比
        aspect = np.array([(y2 - y1) / ((x2 - x1) + 1e-8)], dtype=np.float32)

        feat = np.concatenate([hist_upper, hist_lower, aspect])  # 256+256+1=513
        return feat / (np.linalg.norm(feat) + 1e-8)

    def match_body(self, body_feat):
        """匹配人体外观特征，返回 (is_match, similarity)"""
        if self.reference_body_feat is None or body_feat is None:
            return False, 0.0
        sim = float(np.dot(body_feat, self.reference_body_feat))
        return sim >= self.body_threshold, sim

    def match_combined(self, frame_bgr, kpts, face_emb=None):
        """多线索融合匹配：人脸 > 人体外观

        Returns: (is_match, confidence, source)
            source: 'face' | 'body' | None
        """
        # 第1优先：人脸匹配（精度最高）
        if face_emb is not None:
            name, sim = self.match(face_emb, threshold=self.face_threshold)
            if name is not None:
                return True, sim, 'face'

        # 第2优先：人体外观匹配（侧身/背身时可用）
        body_feat = self.extract_body_features(frame_bgr, kpts)
        is_match, sim = self.match_body(body_feat)
        if is_match:
            return True, sim, 'body'

        return False, 0.0, None

    def enroll_from_frame(self, frame_bgr):
        """从全帧直接录入人脸（用 MTCNN 全帧检测，最准确）"""
        self._ensure_loaded()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            face_tensor = self._mtcnn_face_tensor(rgb)
            if face_tensor is None:
                return None
        except Exception as e:
            logger.debug("MTCNN 录入检测失败: %s", e)
            return None
        return self._embedding_from_face_tensor(face_tensor)

    def enroll_body(self, frame_bgr, kpts):
        """录入人体外观特征（侧身/背身时的后备匹配线索）"""
        self.reference_body_feat = self.extract_body_features(frame_bgr, kpts)
        if self.reference_body_feat is not None:
            logger.info("人体外观特征已录入 (dim=%d)", len(self.reference_body_feat))
            return True
        logger.warning("人体外观特征录入失败（关键点不足）")
        return False

    def match(self, emb, threshold=0.55):
        """匹配嵌入向量与特征库，返回 (name, similarity) 或 (None, max_sim)"""
        if self.db is None or not self.db.get("entries"):
            return None, 0.0

        best_name = None
        best_sim = 0.0
        for entry in self.db["entries"]:
            ref = np.array(entry["embedding"])
            sim = float(np.dot(emb, ref) /
                        (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_name = entry["name"]

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def process_frame(self, frame_bgr):
        """处理一帧：检测人脸 → 提取嵌入 → 匹配

        返回: (name_or_None, similarity, face_thumbnail)
        """
        self._ensure_loaded()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            face = self._mtcnn_face_tensor(rgb)
            if face is None:
                return None, 0.0, None
        except Exception as e:
            logger.debug("MTCNN process_frame 失败: %s", e)
            return None, 0.0, None

        emb_np = self._embedding_from_face_tensor(face)
        if emb_np is None:
            return None, 0.0, None

        # 缩略图
        thumbnail = cv2.resize(frame_bgr, (80, 80))

        name, sim = self.match(emb_np)
        self.reference_face = thumbnail
        return name, sim, thumbnail
