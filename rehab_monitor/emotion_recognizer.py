"""表情识别 — 人脸ROI提取 + 情绪分类"""
import os
import time
import threading
import cv2
import numpy as np
import torch

from .config import (
    EMOTION_MODEL_PATH, EMOTION_CLASSES, EMOTION_INPUT_SIZE,
)
from .logging_setup import get_logger

logger = get_logger("emotion")

# COCO keypoint indices for face
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4


class FaceROIExtractor:
    """从姿态关键点提取人脸ROI，Haar cascade 作为备选"""

    def __init__(self):
        # 加载 Haar cascade 作为备选
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(haar_path)
        self.use_haar_fallback = True

    def extract(self, frame, kpts):
        """提取人脸 ROI

        Args:
            frame: BGR 图像
            kpts: (17, 2) or (17, 3) 关键点数组（单个人）

        Returns:
            face_roi: 裁剪后的人脸图像 (64x64 grayscale)，或 None
            bbox: (x1, y1, x2, y2) 或 None
        """
        h, w = frame.shape[:2]
        bbox = None

        # 方法1: 从 COCO 关键点定位人脸框
        if kpts is not None:
            face_pts = kpts[[NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR], :2]
            valid = face_pts[(face_pts[:, 0] > 0) & (face_pts[:, 1] > 0)]
            if len(valid) >= 2:
                x_min, y_min = valid.min(axis=0)
                x_max, y_max = valid.max(axis=0)
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                size = max(x_max - x_min, y_max - y_min) * 1.8
                half = size / 2
                x1 = int(max(0, cx - half))
                y1 = int(max(0, cy - half))
                x2 = int(min(w, cx + half))
                y2 = int(min(h, cy + half))
                if x2 > x1 and y2 > y1:
                    bbox = (x1, y1, x2, y2)

        # 方法2: Haar cascade 备选
        if bbox is None and self.use_haar_fallback:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                bbox = (x, y, x + fw, y + fh)

        if bbox is None:
            return None, None

        # 裁剪并预处理
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None, None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (EMOTION_INPUT_SIZE, EMOTION_INPUT_SIZE))
        return resized, bbox


class EmotionClassifier:
    """情绪分类器，优先加载 OpenVINO 模型（NPU/CPU），回退到 PyTorch"""

    def __init__(self, model_path=None, device="cpu"):
        self.model_path = model_path or EMOTION_MODEL_PATH
        self.classes = EMOTION_CLASSES
        self.model = None
        self.backend = None      # "openvino" or "torch"
        self._ov_device = None   # "NPU" / "CPU"
        self._ov_infer = None    # OpenVINO infer request

        self._load_model(device)

    def _load_model(self, device):
        # 尝试 OpenVINO IR (XML+BIN 目录)
        if os.path.isdir(self.model_path):
            try:
                import openvino as ov
                xml_path = os.path.join(self.model_path, "emotion_model.xml")
                if not os.path.exists(xml_path):
                    # 尝试找目录内第一个 xml
                    xmls = [f for f in os.listdir(self.model_path) if f.endswith('.xml')]
                    if xmls:
                        xml_path = os.path.join(self.model_path, xmls[0])

                if os.path.exists(xml_path):
                    core = ov.Core()
                    ov_model = core.read_model(xml_path)

                    # 设备选择：优先 NPU，回退 CPU
                    ov_device = "NPU" if device.lower() == "npu" and "NPU" in core.available_devices else "CPU"
                    try:
                        compiled = core.compile_model(ov_model, ov_device)
                        self._ov_infer = compiled.create_infer_request()
                        self._ov_device = ov_device
                        self.backend = "openvino"
                        logger.info("OpenVINO 模型已加载 (%s): %s", ov_device, self.model_path)
                        return
                    except Exception as e:
                        if ov_device == "NPU":
                            logger.warning("NPU 编译失败，回退 CPU: %s", str(e)[:100])
                            compiled = core.compile_model(ov_model, "CPU")
                            self._ov_infer = compiled.create_infer_request()
                            self._ov_device = "CPU"
                            self.backend = "openvino"
                            logger.info("OpenVINO 模型已加载 (CPU): %s", self.model_path)
                            return
                        raise
            except Exception as e:
                logger.warning("OpenVINO 加载失败: %s", e)

        # 尝试 PyTorch 权重
        torch_path = self.model_path + ".pt"
        if os.path.exists(torch_path):
            try:
                import torch
                self.model = _EmotionCNN()
                self.model.load_state_dict(torch.load(torch_path, map_location="cpu"))
                self.model.eval()
                self.backend = "torch"
                logger.info("PyTorch 模型已加载: %s", torch_path)
                return
            except Exception as e:
                logger.warning("PyTorch 加载失败: %s", e)

        logger.warning("未找到模型，将使用模拟输出。运行 train_emotion.py 训练模型。")
        self.backend = "mock"

    def predict(self, face_roi):
        """预测情绪

        Args:
            face_roi: (64, 64) uint8 grayscale image

        Returns:
            (label, scores_dict): label="neutral", scores={class: prob}
        """
        if self.backend == "mock" or face_roi is None:
            return "neutral", {c: 0.0 for c in self.classes}

        try:
            if self.backend == "openvino":
                return self._predict_openvino(face_roi)
            elif self.backend == "torch":
                return self._predict_torch(face_roi)
        except Exception as e:
            logger.error("推理失败: %s", e)

        return "neutral", {c: 0.0 for c in self.classes}

    def _predict_openvino(self, face_roi):
        # face_roi: (64, 64) uint8 grayscale → float32 [0,1], shape (1, 1, 64, 64)
        x = face_roi.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, 1))  # (1, 1, 64, 64)
        out = self._ov_infer.infer([x])[0]
        probs = out.squeeze()
        idx = int(probs.argmax())
        label = self.classes[idx]
        scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return label, scores

    def _predict_torch(self, face_roi):
        import torch
        x = face_roi.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        idx = int(probs.argmax())
        label = self.classes[idx]
        scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return label, scores


class _EmotionCNN(torch.nn.Module):
    """轻量级情绪分类 CNN (~200K 参数)"""

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---- 后台线程 ----

def run_emotion_thread(recognizer, stop_event, frame_ref, result_queue, interval=5):
    """后台情绪识别线程

    Args:
        recognizer: EmotionRecognizer 实例
        stop_event: threading.Event, 退出信号
        frame_ref: 共享帧引用 (list of 1 element)
        result_queue: queue.Queue, 结果输出
        interval: 每隔多少帧处理一次
    """
    logger.info("后台线程已启动")
    frame_count = 0
    last_result = ("neutral", {c: 0.0 for c in EMOTION_CLASSES}, None, None)

    while not stop_event.is_set():
        stop_event.wait(0.04)  # ~25fps
        frame_count += 1

        if frame_count % interval != 0:
            continue

        frame = frame_ref[0]
        if frame is None:
            continue

        try:
            face_roi, bbox = recognizer.extractor.extract(frame, recognizer._current_kpts)
            if face_roi is not None:
                label, scores = recognizer.classifier.predict(face_roi)
                last_result = (label, scores, bbox, face_roi)
        except Exception as e:
            logger.error("后台线程错误: %s", e)

        # 非阻塞放入队列
        if result_queue.empty():
            try:
                result_queue.put_nowait(last_result)
            except queue.Full:
                pass

    logger.info("后台线程已退出")


class BodyEmotionClassifier:
    """体态情绪分类器 — 用 17 个关键点推断情绪，不依赖人脸

    当人体水平（躺下）时人脸不可靠，体态分类器权重提高。
    """

    def __init__(self):
        self.classes = EMOTION_CLASSES
        self.model = _BodyEmotionNet(num_classes=7)
        self.model.eval()
        self._loaded = False
        self._try_load()

    def _try_load(self):
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "model", "body_emotion.pt")
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path, map_location="cpu"))
                self._loaded = True
                logger.info("体态情绪模型已加载: %s", path)
            except Exception as e:
                logger.warning("体态情绪模型加载失败: %s", e)
        else:
            logger.info("未找到 body_emotion.pt，体态情绪不可用")

    def predict(self, kpts):
        """输入 (17, 2) 或 (17, 3) 关键点，返回 (label, scores) 或 None"""
        if not self._loaded or kpts is None:
            return None
        kpts = np.asarray(kpts[:, :2])  # 只取 x, y
        # 归一化：以髋部中心为原点，肩宽为单位
        hips = kpts[11:13]  # left_hip, right_hip
        shoulders = kpts[5:7]  # left_shoulder, right_shoulder
        hip_center = hips.mean(axis=0)
        shoulder_width = np.linalg.norm(shoulders[0] - shoulders[1]) + 1e-8
        normalized = (kpts - hip_center) / shoulder_width
        x = torch.from_numpy(normalized.astype(np.float32)).flatten().unsqueeze(0)  # (1, 34)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        idx = int(probs.argmax())
        return self.classes[idx], {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}


class _BodyEmotionNet(torch.nn.Module):
    """轻量 MLP：34 维输入 (17 关键点 × 2)，7 维输出"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(34, 64),
            torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EmotionVoter:
    """多帧投票缓冲：收集最近 N 帧预测，返回众数"""

    def __init__(self, window_size=15):
        self.window = []  # list of (label, scores_dict)
        self.window_size = window_size

    def add(self, label, scores):
        self.window.append((label, scores))
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def get(self):
        """返回投票结果：众数标签 + 平均分数"""
        if not self.window:
            return "neutral", {c: 0.0 for c in EMOTION_CLASSES}
        # 统计标签众数
        from collections import Counter
        labels = [w[0] for w in self.window]
        voted_label = Counter(labels).most_common(1)[0][0]
        # 平均分数
        n = len(self.window)
        avg_scores = {}
        for cls_name in EMOTION_CLASSES:
            avg_scores[cls_name] = float(sum(w[1].get(cls_name, 0) for w in self.window) / n)
        return voted_label, avg_scores


class TargetFaceEnroller:
    """目标人脸注册与匹配 — 用 EmotionCNN 特征层做嵌入"""

    def __init__(self, classifier):
        self.classifier = classifier
        self.reference_embedding = None  # 参考嵌入向量
        self.reference_face = None       # 参考人脸缩略图 (用于显示)
        self.threshold = 0.55            # 余弦相似度阈值

    def _get_embedding(self, face_roi):
        """从人脸 ROI 提取 128 维嵌入向量"""
        if self.classifier.backend != "torch" or self.classifier.model is None:
            return None
        import torch
        x = face_roi.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        with torch.no_grad():
            features = self.classifier.model.features(x)  # (1, 128)
            return features.squeeze().numpy()

    def enroll(self, face_roi):
        """录入参考人脸"""
        emb = self._get_embedding(face_roi)
        if emb is not None:
            self.reference_embedding = emb
            self.reference_face = face_roi.copy()
            logger.info("目标人脸已录入")
            return True
        logger.warning("录入失败")
        return False

    def match(self, face_roi):
        """匹配人脸，返回 (is_match, similarity)"""
        if self.reference_embedding is None or face_roi is None:
            return False, 0.0
        emb = self._get_embedding(face_roi)
        if emb is None:
            return False, 0.0
        sim = float(np.dot(emb, self.reference_embedding) /
                    (np.linalg.norm(emb) * np.linalg.norm(self.reference_embedding) + 1e-8))
        return sim >= self.threshold, sim

    def is_enrolled(self):
        return self.reference_embedding is not None


class EmotionRecognizer:
    """表情识别器：人脸提取 + 分类 + 多帧投票 + 目标匹配"""

    def __init__(self, model_path=None, device="cpu"):
        self.extractor = FaceROIExtractor()
        self.classifier = EmotionClassifier(model_path, device=device)
        self.voter = EmotionVoter(window_size=15)
        self.target = TargetFaceEnroller(self.classifier)
        self._current_kpts = None

    def set_keypoints(self, kpts):
        self._current_kpts = kpts

    def extract(self, frame, kpts=None):
        k = kpts if kpts is not None else self._current_kpts
        return self.extractor.extract(frame, k)

    def predict(self, face_roi):
        return self.classifier.predict(face_roi)

    def predict_voted(self, face_roi):
        """单帧预测 → 加入投票缓冲 → 返回投票结果"""
        label, scores = self.predict(face_roi)
        self.voter.add(label, scores)
        return self.voter.get()

    def enroll_target(self, frame, kpts):
        """录入当前帧中的人脸作为跟踪目标"""
        face_roi, bbox = self.extract(frame, kpts)
        if face_roi is not None:
            return self.target.enroll(face_roi)
        logger.warning("未检测到人脸，请正对摄像头")
        return False

    def match_target(self, face_roi):
        """检查人脸是否匹配注册目标"""
        return self.target.match(face_roi)
