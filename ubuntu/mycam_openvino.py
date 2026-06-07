#!/usr/bin/env python3
"""
YOLO Pose 实时摄像头检测 — Ubuntu/Linux 适配版
原文件: ultralytics-8.4.46/mycam_openvino.py (未修改)

用法:
  python ubuntu/mycam_openvino.py              # NPU FP32 (默认, 需render组)
  python ubuntu/mycam_openvino.py --cpu        # CPU INT8
  python ubuntu/mycam_openvino.py --gpu        # GPU FP32
  python ubuntu/mycam_openvino.py --model yolo26n  # 切换模型
"""
import os, sys, time, argparse

# ---- 硬件优化 (必须在 torch/openvino 前设置) ----
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 确保能找到 rehab_monitor 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
from ultralytics import YOLO

# ---- 模型路径 (相对于项目根目录) ----
MODEL_DIR = os.path.join(PROJECT_ROOT, "ultralytics-8.4.46", "model")
MODELS = {
    "yolo11n": {
        "fp32": os.path.join(MODEL_DIR, "yolo11n-pose_openvino_model"),
        "int8": os.path.join(MODEL_DIR, "yolo11n-pose_int8_openvino_model"),
    },
    "yolo11s": {
        "fp32": os.path.join(MODEL_DIR, "yolo11s-pose_openvino_model"),
        "int8": os.path.join(MODEL_DIR, "yolo11s-pose_int8_openvino_model"),
    },
    "yolo26n": {
        "fp32": os.path.join(MODEL_DIR, "yolo26n-pose_openvino_model"),
        "fp16": os.path.join(MODEL_DIR, "yolo26n-pose_fp16_openvino_model"),
        "int8": os.path.join(MODEL_DIR, "yolo26n-pose_int8_openvino_model"),
    },
}


def get_device_str(device: str) -> str:
    """将简写设备名转为 ultralytics/OpenVINO 可识别的格式"""
    mapping = {
        "cpu": "intel:cpu",
        "gpu": "intel:gpu",
        "npu": "intel:npu",
    }
    return mapping.get(device, "intel:cpu")


class OpenVINOPoseDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.device = get_device_str(device)
        print(f"推理设备: {self.device}")

    def process_frame(self, frame):
        results = self.model(frame, device=self.device, verbose=False)
        annotated = results[0].plot()
        kpts = results[0].keypoints
        kpts_data = kpts.data.cpu().numpy() if kpts is not None and len(kpts) > 0 else None
        return annotated, kpts_data


def main():
    parser = argparse.ArgumentParser(description="YOLO Pose 实时检测 (Ubuntu)")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 推理")
    parser.add_argument("--npu", action="store_true", help="使用 NPU 推理 (默认, 需 render 组)")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU INT8")
    parser.add_argument("--model", default="yolo11n", choices=["yolo11n", "yolo11s", "yolo26n"],
                        help="选择模型 (默认 yolo11n)")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--no-fps", action="store_true", help="隐藏帧率")
    args = parser.parse_args()

    # 确定设备和模型
    if args.npu:
        device, precision = "npu", "fp32"
    elif args.gpu:
        device, precision = "gpu", "fp32"
    elif args.cpu:
        device, precision = "cpu", "int8"
    else:
        device, precision = "npu", "fp32"   # 默认 NPU

    model_name = args.model
    model_path = MODELS[model_name].get(precision, MODELS[model_name]["fp32"])
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 {model_path}")
        sys.exit(1)

    # NPU 权限检查
    if device == "npu":
        import grp
        user_groups = {g.gr_name for g in grp.getgrall() if os.getuid() in g.gr_mem}
        if "render" not in user_groups:
            print("[警告] 不在 render 组，NPU 不可用，回退到 CPU")
            device, precision = "cpu", "int8"

    print(f"模型: {model_name} | 精度: {precision} | 设备: {device}")
    print(f"路径: {model_path}")

    detector = OpenVINOPoseDetector(model_path, device=device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("错误: 无法打开摄像头!")
        sys.exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头: /dev/video{args.camera} | {fw}x{fh}")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    show_fps = not args.no_fps
    save_counter = 0
    show_help = False

    print("操作: q=退出 s=截图 f=切换FPS h=帮助")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        annotated, kpts = detector.process_frame(frame)

        if show_fps:
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if kpts is not None:
            cv2.putText(annotated, f"Persons: {kpts.shape[0]}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        tag = f"{model_name.upper()}-Pose | {device.upper()} | OpenVINO"
        cv2.putText(annotated, tag, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("YOLO Pose Detection - Ubuntu/Linux", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出")
            break
        elif key == ord('s'):
            save_counter += 1
            fname = f"screenshot_{save_counter}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"截图: {fname}")
        elif key == ord('f'):
            show_fps = not show_fps
            print(f"FPS: {'ON' if show_fps else 'OFF'}")
        elif key == ord('h'):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()
    print("结束")


if __name__ == "__main__":
    main()
