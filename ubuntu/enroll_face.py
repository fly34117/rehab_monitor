"""Round 1: 人脸录入脚本 — 基于 FaceNet + MTCNN

用法:
    python enroll_face.py              # 录入新人脸
    python enroll_face.py --list       # 列出已录入的人
    python enroll_face.py --delete 1   # 删除编号为1的人

输出:
    model/face_db.json                 # 人脸特征库
"""
import sys
import os
import json
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

DB_PATH = "model/face_db.json"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {"entries": []}

def save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def list_entries():
    db = load_db()
    if not db["entries"]:
        print("[录入库] 暂无记录")
        return
    print(f"\n{'ID':<4} {'Name':<15} {'Date':<12}")
    print("-" * 35)
    for i, e in enumerate(db["entries"]):
        print(f"{i:<4} {e['name']:<15} {e.get('date','?')[:10]:<12}")
    print()

def delete_entry(idx):
    db = load_db()
    if idx < 0 or idx >= len(db["entries"]):
        print(f"[错误] ID {idx} 不存在")
        return
    name = db["entries"][idx]["name"]
    db["entries"].pop(idx)
    save_db(db)
    print(f"[录入库] 已删除: {name}")

def enroll():
    """从摄像头录入人脸"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[录入] 设备: {device}")

    # 加载模型
    print("[录入] 加载 FaceNet 模型...")
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    name = input("[录入] 请输入姓名: ").strip()
    if not name:
        print("[录入] 已取消")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[录入] 无法打开摄像头")
        return

    print(f"\n[录入] 即将为 '{name}' 拍照，请正对摄像头...")
    print("[录入] 按 空格键 拍照, 按 q 取消\n")

    embeddings = []
    while len(embeddings) < 5:
        ret, frame = cap.read()
        if not ret:
            break

        # 显示画面
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Captured: {len(embeddings)}/5", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: capture | Q: quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Enroll Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') or key == 32:
            # MTCNN 检测人脸 → FaceNet 提取特征
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)
            if face is not None:
                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device))
                    embeddings.append(emb.cpu().squeeze().numpy().tolist())
                print(f"[录入] 已采集 {len(embeddings)}/5 张")
            else:
                print("[录入] 未检测到人脸，请调整位置")

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("[录入] 未采集到任何人脸，退出")
        return

    # 平均融合多张照片
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    db = load_db()
    db["entries"].append({
        "name": name,
        "embedding": avg_embedding,
        "samples": len(embeddings),
        "date": __import__("datetime").datetime.now().isoformat()[:19],
    })
    save_db(db)
    print(f"\n[录入] 成功! '{name}' 的人脸特征已保存 ({len(embeddings)} 样本)")

if __name__ == "__main__":
    if "--list" in sys.argv:
        list_entries()
    elif "--delete" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--delete") + 1])
        delete_entry(idx)
    else:
        enroll()
