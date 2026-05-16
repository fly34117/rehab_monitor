"""训练体态情绪模型 — 用 17 关键点推断情绪

思路：模拟不同情绪对应的身体姿态，训练轻量MLP。
躺下时人脸不可靠，体态信号作为主要判断依据。

用法:
    python train_body_emotion.py
输出:
    model/body_emotion.pt
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# ---- 模型（与 emotion_recognizer.py 保持一致）----

class BodyEmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(34, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    def forward(self, x): return self.net(x)


# ---- 姿态模板生成 ----

def generate_posture_samples():
    """根据情绪生成模拟姿态关键点

    每种情绪定义特定的姿态特征：
    - happy: 直立，手臂张开，肩膀舒展
    - sad: 低头，塌肩，手臂下垂内收
    - angry: 肩膀耸起，身体前倾，握拳
    - fear: 保护姿态，手臂内收，身体微缩
    - neutral: 自然站立
    - disgust: 身体后仰，头偏转
    - surprise: 手臂微抬，身体直立
    """
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    samples = []

    # 基础站立姿态 (COCO 17 keypoints, 归一化坐标)
    base = np.array([
        [0.00, -0.90],  # 0 nose
        [-0.04, -0.93], # 1 left_eye
        [0.04, -0.93],  # 2 right_eye
        [-0.08, -0.92], # 3 left_ear
        [0.08, -0.92],  # 4 right_ear
        [-0.15, -0.60], # 5 left_shoulder
        [0.15, -0.60],  # 6 right_shoulder
        [-0.25, -0.30], # 7 left_elbow
        [0.25, -0.30],  # 8 right_elbow
        [-0.30, 0.00],  # 9 left_wrist
        [0.30, 0.00],   # 10 right_wrist
        [-0.10, 0.20],  # 11 left_hip
        [0.10, 0.20],   # 12 right_hip
        [-0.10, 0.60],  # 13 left_knee
        [0.10, 0.60],   # 14 right_knee
        [-0.10, 1.00],  # 15 left_ankle
        [0.10, 1.00],   # 16 right_ankle
    ])

    # 每个情绪的扰动参数 (shoulder, arm, head, lean)
    perturbations = {
        "happy":    {"shoulder_y": -0.05, "arm_out": 0.15, "head_y": 0.0, "lean": 0.0},
        "sad":      {"shoulder_y": 0.08, "arm_out": -0.10, "head_y": 0.05, "lean": 0.0},
        "angry":    {"shoulder_y": -0.08, "arm_out": 0.05, "head_y": -0.02, "lean": 0.06},
        "fear":     {"shoulder_y": 0.05, "arm_out": -0.15, "head_y": 0.02, "lean": -0.04},
        "neutral":  {"shoulder_y": 0.0, "arm_out": 0.0, "head_y": 0.0, "lean": 0.0},
        "disgust":  {"shoulder_y": 0.0, "arm_out": 0.05, "head_y": -0.03, "lean": -0.08},
        "surprise": {"shoulder_y": -0.03, "arm_out": 0.10, "head_y": -0.04, "lean": 0.0},
    }

    for label_idx, emotion in enumerate(emotions):
        p = perturbations[emotion]
        for _ in range(800):  # 每类 800 个样本
            kpts = base.copy()
            noise = np.random.randn(17, 2) * 0.02

            # 肩膀上下
            kpts[5:7, 1] += p["shoulder_y"] + np.random.randn() * 0.03
            # 手臂内外
            kpts[7:11, 0] += p["arm_out"] + np.random.randn(4) * 0.05
            # 头部前倾后仰
            kpts[0:5, 1] += p["head_y"] + np.random.randn() * 0.02
            # 身体前倾后仰
            kpts[0:11, 0] += p["lean"] + np.random.randn() * 0.02

            kpts += noise
            # 翻转增强（左右镜像）
            if np.random.rand() > 0.5:
                kpts[:, 0] *= -1
                kpts[[1, 3, 5, 7, 9, 11, 13, 15]] = kpts[[2, 4, 6, 8, 10, 12, 14, 16]]

            samples.append((kpts.flatten().astype(np.float32), label_idx))

    return samples


# ---- 训练 ----

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[训练] 设备: {device}")

    samples = generate_posture_samples()
    np.random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_data = samples[:split]
    val_data = samples[split:]

    model = BodyEmotionNet(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    X_train = torch.from_numpy(np.array([s[0] for s in train_data])).to(device)
    y_train = torch.tensor([s[1] for s in train_data], dtype=torch.long).to(device)
    X_val = torch.from_numpy(np.array([s[0] for s in val_data])).to(device)
    y_val = torch.tensor([s[1] for s in val_data], dtype=torch.long).to(device)

    batch_size = 128
    best_acc = 0.0

    for epoch in range(200):
        # 训练
        model.train()
        perm = torch.randperm(len(train_data))
        train_correct, train_total = 0, 0
        for i in range(0, len(train_data), batch_size):
            idx = perm[i:i + batch_size]
            x, y = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_correct += (model(x).argmax(1) == y).sum().item()
            train_total += len(y)

        # 验证
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_correct = (outputs.argmax(1) == y_val).sum().item()
            val_acc = 100 * val_correct / len(val_data)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/body_emotion.pt")

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/200 | "
                  f"Train Acc: {100*train_correct/train_total:.1f}% | "
                  f"Val Acc: {val_acc:.1f}%")

    print(f"\n[训练] 完成! 最佳验证准确率: {best_acc:.1f}%")
    print("[训练] 模型已保存: model/body_emotion.pt")


if __name__ == "__main__":
    train()
