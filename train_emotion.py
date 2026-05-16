"""训练表情识别模型 — 支持 FER2013 图片格式 + CSV 格式

用法:
    conda activate yolov26
    python train_emotion.py

输出:
    model/emotion_model.pt           # PyTorch 权重
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image

# ---- 配置 ----
DATA_DIR = "fer2013_data"
MODEL_OUT = "model/emotion_model.pt"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
IMG_SIZE = 64
NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ---- 数据集 ----

class EmotionImageDataset(Dataset):
    """从文件夹结构加载 FER2013 图片 (train/emotion/*.jpg)"""

    def __init__(self, root_dir, img_size=64, train=True):
        self.samples = []
        self.img_size = img_size
        subdir = "train" if train else "test"
        data_dir = os.path.join(root_dir, subdir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"目录不存在: {data_dir}")

        for label_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.isdir(emotion_dir):
                for fname in os.listdir(emotion_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(emotion_dir, fname), label_idx))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomApply([transforms.RandomRotation(degrees=45)], p=0.6),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.3)], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img), label


class EmotionCSVDataset(Dataset):
    """从 CSV 加载 FER2013 (col0=emotion, col1=pixels, col2=usage)"""

    def __init__(self, csv_path, img_size=64, train=True):
        self.img_size = img_size
        self.samples = []
        import csv
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                usage = row[2]
                if train and usage == "Training":
                    self.samples.append((row[1], int(row[0])))
                elif not train and usage in ("PublicTest", "PrivateTest"):
                    self.samples.append((row[1], int(row[0])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pixels_str, label = self.samples[idx]
        pixels = np.array([int(p) for p in pixels_str.split()], dtype=np.float32)
        img = pixels.reshape(48, 48)
        img = torch.from_numpy(img).unsqueeze(0)
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear"
        ).squeeze(0)
        return img / 255.0, int(label)


# ---- 模型 ----

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---- 数据下载 ----

def download_fer2013():
    """下载 FER2013 数据集"""
    os.makedirs(DATA_DIR, exist_ok=True)

    # 检查图片格式 (train/emotion/*.jpg)
    if os.path.isdir(os.path.join(DATA_DIR, "train")):
        print(f"[数据] FER2013 图片数据集已存在: {DATA_DIR}")
        return "image"

    # 检查 CSV 格式
    csv_path = os.path.join(DATA_DIR, "fer2013.csv")
    if os.path.exists(csv_path):
        print(f"[数据] FER2013 CSV 已存在: {csv_path}")
        return "csv"

    # 用 kagglehub 下载图片版
    try:
        import kagglehub
        print("[数据] 通过 kagglehub 下载 FER2013 (图片格式)...")
        dl_dir = kagglehub.dataset_download("msambare/fer2013")
        import shutil
        for sub in ["train", "test"]:
            src = os.path.join(dl_dir, sub)
            dst = os.path.join(DATA_DIR, sub)
            if os.path.isdir(src) and not os.path.isdir(dst):
                shutil.copytree(src, dst)
        print(f"[数据] 下载完成: {DATA_DIR}")
        return "image"
    except Exception as e:
        print(f"[数据] kagglehub 失败: {e}")

    print(f"\n[错误] 自动下载失败。请手动操作：")
    print(f"  方法A (推荐): 访问 https://www.kaggle.com/datasets/msambare/fer2013 下载")
    print(f"             解压后 train/ 和 test/ 放到 {DATA_DIR}/")
    print(f"  方法B: 下载 fer2013.csv 放到 {csv_path}")
    print(f"  然后重新运行 python train_emotion.py")
    sys.exit(1)


# ---- 训练 ----

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[训练] 设备: {device}")

    # 数据
    data_type = download_fer2013()

    if data_type == "image":
        train_ds = EmotionImageDataset(DATA_DIR, img_size=IMG_SIZE, train=True)
        val_ds = EmotionImageDataset(DATA_DIR, img_size=IMG_SIZE, train=False)
    else:
        csv_path = os.path.join(DATA_DIR, "fer2013.csv")
        train_ds = EmotionCSVDataset(csv_path, img_size=IMG_SIZE, train=True)
        val_ds = EmotionCSVDataset(csv_path, img_size=IMG_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"[数据] 训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

    # 模型
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)

    # 类别权重
    labels = [train_ds[i][1] for i in range(len(train_ds))]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = 1.0 / (class_counts + 1)
    weights = weights / weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
    print(f"[数据] 类别分布: {dict(zip(EMOTIONS, class_counts.astype(int)))}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (model(imgs).argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.1f}%")

    print(f"\n[训练] 完成! 最佳验证准确率: {best_acc:.1f}%")
    print(f"[训练] 模型已保存: {MODEL_OUT}")
    print(f"[训练] 运行主程序即可自动加载模型进行表情识别。")


if __name__ == "__main__":
    train()
