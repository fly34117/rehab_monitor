"""Ubuntu 启动入口 — 不修改原项目，运行时覆盖配置
支持所有 OpenVINO 模型的 CPU/GPU/NPU 切换
"""
import os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_name = os.environ.get("REHAB_MODEL", "yolo26n")
device = os.environ.get("REHAB_DEVICE", "cpu")

import rehab_monitor.config as cfg

# ---- 模型-精度映射 ----
MODEL_PRECISION = {
    "yolo11n": {"cpu": "int8", "gpu": "fp32", "npu": "fp32"},
    "yolo11s": {"cpu": "int8", "gpu": "fp32", "npu": "fp32"},
    "yolo26n": {"cpu": "int8", "gpu": "fp32", "npu": "fp16"},
    "yolo26s": {"cpu": "int8", "gpu": "fp32", "npu": "fp32"},
    "yolo26m": {"cpu": "fp32", "gpu": "fp32", "npu": "fp32"},
}

if model_name not in MODEL_PRECISION:
    print(f"[ubuntu] 未知模型 '{model_name}'，回退到 yolo26n")
    model_name = "yolo26n"

precision = MODEL_PRECISION[model_name].get(device, "fp32")

if precision == "int8":
    model_dir = f"{model_name}-pose_int8_openvino_model"
elif precision == "fp16":
    model_dir = f"{model_name}-pose_fp16_openvino_model"
else:
    model_dir = f"{model_name}-pose_openvino_model"

cfg.POSE_MODEL_PATH = os.path.join(cfg.MODEL_DIR, model_dir)
cfg.POSE_DEVICE = {
    "cpu": "cpu",
    "gpu": "intel:gpu",
    "npu": "intel:npu",
}[device]

if not os.path.exists(cfg.POSE_MODEL_PATH):
    print(f"[ubuntu] 模型不存在: {cfg.POSE_MODEL_PATH}")
    sys.exit(1)

MODEL_TAG = f"{model_name}-pose {precision} {device}"
print(f"[ubuntu] 模型={model_name}-pose  精度={precision}  设备={cfg.POSE_DEVICE}")

# ---- Monkey-patch: 侧面板 MODEL 区域显示真实配置 ----
import rehab_monitor.display_overlay as ov_module
_orig_draw_side = ov_module.DisplayOverlay._draw_side_panel

def _patched_draw_side(self, out, state, w, h):
    _orig_draw_side(self, out, state, w, h)
    # 覆盖侧面板底部的 MODEL 信息 (原代码硬编码 "yolo26n-pose INT8")
    panel_x = w - 220
    # MODEL 区域在面板底部: 从 h-55 附近往上搜索 "MODEL" 文字位置
    # 直接覆盖固定位置 — 原代码在 _draw_side_panel 的 y 递增序列末尾
    # y 最终位置约在 h 的 85%-95% 区间，取决于面板内容
    # 安全做法: 找 "MODEL" 文字并用真实值覆盖其下方两行
    pass

# 更简洁可靠的做法: 在 render 后直接在侧面板底部追加真实模型信息
_orig_render = ov_module.DisplayOverlay.render

def _patched_render(self, frame, state):
    out = _orig_render(self, frame, state)
    h, w = out.shape[:2]
    panel_x = w - 220
    # 在侧面板最底部覆盖模型信息 (原 MODEL 区域下方)
    y_base = h - 45
    # 清除原硬编码文字区域 (画背景条覆盖)
    cv2.rectangle(out, (panel_x, y_base - 5), (w, h), (40, 40, 40), -1)
    # 重新绘制
    cv2.putText(out, "MODEL", (panel_x + 8, y_base + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(out, MODEL_TAG, (panel_x + 8, y_base + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    kalman = "ON" if state.get("use_kalman") else "OFF"
    cv2.putText(out, f"Kalman: {kalman}", (panel_x + 8, y_base + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    return out

ov_module.DisplayOverlay.render = _patched_render

print(f"[ubuntu] 画面叠加已更新: {MODEL_TAG}")

# ---- Monkey-patch: 腿关键点交叉干扰纠正 ----
from ubuntu.leg_corrector import LegCorrector
import rehab_monitor.pose_detector as pd_module

_corrector = LegCorrector()
_orig_process_frame = pd_module.PoseDetector.process_frame

def _corrected_process_frame(self, frame):
    annotated, kpts = _orig_process_frame(self, frame)
    if kpts is not None and kpts.shape[0] == 1:
        kpts, swapped = _corrector.correct(kpts)
        if swapped:
            # 更新 annotated 帧上的关键点位置 (仅修正点, 连线会自然跟随)
            pass  # annotated 来自 results[0].plot(), 此处不对画图做修改
    return annotated, kpts

pd_module.PoseDetector.process_frame = _corrected_process_frame
print(f"[ubuntu] 腿关键点纠正器已启用")

import rehab_monitor.main
rehab_monitor.main.main()
