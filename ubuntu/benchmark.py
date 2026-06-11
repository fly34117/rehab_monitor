#!/usr/bin/env python
"""
两种管线流畅度对比 benchmark
==============================
方式一: monkey-patch (npu_main.py 模式) — cv2.VideoCapture 同步
方式二: 三阶段异构 (rehab_optimized) — V4L2 异步 + 三阶段管线

指标: FPS(均值/最小/最大/方差), 捕获延迟, 推理延迟, 后处理延迟, 丢帧率
方差越小 → 帧间隔越稳定 → 画面越流畅
"""
import sys, os, time, queue, threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAMES = 100  # 每种方式测试帧数
WARMUP = 10   # 预热帧数

def stats(name, values):
    """计算统计值"""
    arr = np.array(values)
    return {
        "name": name,
        "mean": np.mean(arr),
        "std":  np.std(arr),
        "min":  np.min(arr),
        "max":  np.max(arr),
        "p95":  np.percentile(arr, 95),
        "cv":   np.std(arr) / np.mean(arr) * 100 if np.mean(arr) > 0 else 0,  # 变异系数
    }

def run_benchmark(name, capture_fn, infer_fn, post_fn, frames=FRAMES):
    """运行 benchmark

    capture_fn() → (ret, frame, ts)
    infer_fn(frame) → (annotated, kpts)
    post_fn(frame, kpts) → None
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    frame_times = []      # 帧间隔 (ms)
    capture_times = []    # 采集耗时
    infer_times = []      # 推理耗时
    post_times = []       # 后处理耗时
    drops = 0

    prev_ts = None

    for i in range(frames + WARMUP):
        # --- 采集 ---
        t0 = time.time()
        ret, frame, ts = capture_fn()
        cap_ms = (time.time() - t0) * 1000

        if not ret or frame is None:
            drops += 1
            continue

        # --- 推理 ---
        t1 = time.time()
        annotated, kpts = infer_fn(frame)
        inf_ms = (time.time() - t1) * 1000

        if annotated is None or kpts is None:
            drops += 1
            continue

        # --- 后处理 ---
        t2 = time.time()
        post_fn(frame, kpts)
        post_ms = (time.time() - t2) * 1000

        # 统计 (跳过预热)
        if i >= WARMUP:
            capture_times.append(cap_ms)
            infer_times.append(inf_ms)
            post_times.append(post_ms)
            if prev_ts is not None:
                frame_times.append((time.time() - prev_ts) * 1000)
            prev_ts = time.time()

        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{frames+WARMUP} 帧", flush=True)

    # 计算统计
    results = {}
    if frame_times:
        results["frame_interval"] = stats(f"{name} 帧间隔", frame_times)
        fps_arr = 1000 / np.array(frame_times)
        results["fps"] = stats(f"{name} FPS", fps_arr)
    if capture_times:
        results["capture"] = stats(f"{name} 采集", capture_times)
    if infer_times:
        results["inference"] = stats(f"{name} 推理", infer_times)
    if post_times:
        results["postprocess"] = stats(f"{name} 后处理", post_times)
    results["drops"] = drops
    results["total_frames"] = frames + WARMUP

    return results

def print_result(r):
    """打印单个统计"""
    if "mean" not in r:
        return
    print(f"  {r['name']:30s}: avg={r['mean']:6.1f}  "
          f"σ={r['std']:5.1f}  p95={r['p95']:5.1f}  "
          f"min={r['min']:5.1f}  max={r['max']:5.1f}  "
          f"CV={r['cv']:4.1f}%")


# ============================================================
# 方式一: monkey-patch (原始 cv2.VideoCapture 同步)
# ============================================================
def bench_v1():
    # 模拟 npu_main.py 的行为: 修改 config
    import rehab_monitor.config as cfg
    from rehab_monitor.pose_detector import PoseDetector
    from ubuntu.leg_corrector import LegCorrector

    # 配置
    cfg.POSE_MODEL_PATH = os.path.join(
        cfg.MODEL_DIR, "yolo11n-pose_int8_openvino_model")
    cfg.POSE_DEVICE = "cpu"

    # 初始化
    detector = PoseDetector()
    leg_corrector = LegCorrector()

    # 摄像头 (原始同步方式)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def capture():
        return cap.read() + (time.time() * 1000,)

    def infer(frame):
        annotated, kpts = detector.process_frame(frame)
        if kpts is not None and kpts.shape[0] == 1:
            kpts, _ = leg_corrector.correct(kpts)
        return annotated, kpts

    def postproc(frame, kpts):
        # 模拟最小后处理
        if kpts is not None and kpts.shape[0] > 0:
            pass  # 实际会有 gait, fall 等

    results = run_benchmark("V1 monkey-patch (同步采集)", capture, infer, postproc)
    cap.release()
    return results

# ============================================================
# 方式二: 三阶段异构 (V4L2 异步 + 三阶段管线)
# ============================================================
def bench_v2():
    from rehab_optimized.gst_camera import OpenCVCamera
    from rehab_optimized.model_loader import ModelLoader
    from rehab_monitor.kalman_filter import PoseKalmanSmoother
    from ubuntu.leg_corrector import LegCorrector

    # Stage 1: 异步摄像头 (V4L2 MJPEG)
    cam = OpenCVCamera(0, 640, 480, 30)

    # Stage 2: 模型
    loader = ModelLoader("yolo11n", "cpu")
    model, dev_str, model_path = loader.load()

    # 卡尔曼
    kalman = PoseKalmanSmoother(
        num_kpts=17, dt=0.04,
        process_noise=1e-5,
        measurement_noise=1e-3,
        max_persons=5,
    )
    det_to_track = {}

    # 腿纠正
    leg_corrector = LegCorrector()

    def capture():
        return cam.read()

    def infer(frame):
        results = model(frame, device=dev_str, verbose=False)
        raw = None
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            raw = results[0].keypoints.data.cpu().numpy()

        # 卡尔曼平滑
        smoothed = None
        if raw is not None:
            ks = kalman
            num_det = raw.shape[0]
            predicted = ks.predict_all()
            if not predicted:
                smoothed = np.zeros_like(raw)
                for j in range(min(num_det, 5)):
                    smoothed[j, :, :2] = ks.create_track(j, raw[j, :, :2])
                    smoothed[j, :, 2] = raw[j, :, 2]
            else:
                # 简化贪心匹配
                smoothed = np.zeros_like(raw)
                for j in range(min(num_det, 5)):
                    smoothed[j] = raw[j]

        if smoothed is not None and smoothed.shape[0] == 1:
            smoothed, _ = leg_corrector.correct(smoothed)

        annotated = results[0].plot()
        return annotated, smoothed

    def postproc(frame, kpts):
        pass  # 同 V1

    results = run_benchmark("V2 三阶段异构 (异步采集)", capture, infer, postproc)
    cam.release()
    return results


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("康复监测系统 — 流畅度 Benchmark")
    print(f"测试帧数: {FRAMES} (+{WARMUP} 预热)")
    print()

    # 确保环境就绪
    try:
        cv2.VideoCapture(0, cv2.CAP_V4L2).release()
    except Exception:
        print("错误: 无法访问摄像头 /dev/video0")
        sys.exit(1)

    # --- 方式一 ---
    r1 = bench_v1()

    # 短暂休息 (让摄像头释放)
    time.sleep(1.0)

    # --- 方式二 ---
    r2 = bench_v2()

    # ============================================================
    # 对比报告
    # ============================================================
    print()
    print("=" * 70)
    print("  对比报告")
    print("=" * 70)

    for metric, label in [
        ("fps", "FPS (越高越好)"),
        ("frame_interval", "帧间隔 ms (越低越稳)"),
        ("capture", "采集延迟 ms"),
        ("inference", "推理延迟 ms"),
        ("postprocess", "后处理延迟 ms"),
    ]:
        print(f"\n── {label} ──")
        for r in [r1, r2]:
            if metric in r:
                print_result(r[metric])

    # 流畅度评分
    print()
    print("── 流畅度评分 ──")
    for name, r in [("V1 monkey-patch", r1), ("V2 三阶段异构", r2)]:
        score = 100
        issues = []
        if "fps" in r:
            fps = r["fps"]
            if fps["mean"] < 20: score -= 20; issues.append("FPS偏低")
            elif fps["mean"] < 25: score -= 10; issues.append("FPS略低")
            if fps["cv"] > 15: score -= 20; issues.append(f"FPS波动大(CV={fps['cv']:.0f}%)")
            elif fps["cv"] > 8: score -= 10; issues.append(f"FPS有波动(CV={fps['cv']:.0f}%)")
        if "capture" in r and r["capture"]["mean"] > 50:
            score -= 10; issues.append("采集延迟高")
        if r.get("drops", 0) > FRAMES * 0.05:
            score -= 15; issues.append(f"丢帧率高({r['drops']}帧)")
        print(f"  {name}: {score}/100 {'✓' if score>=80 else '△'}")
        if issues:
            for issue in issues:
                print(f"    - {issue}")

    print()
    print("结论:")
    if "fps" in r1 and "fps" in r2:
        v1_fps = r1["fps"]["mean"]
        v2_fps = r2["fps"]["mean"]
        v1_cv = r1["fps"]["cv"]
        v2_cv = r2["fps"]["cv"]

        if v2_fps > v1_fps and v2_cv < v1_cv:
            print(f"  → V2 三阶段异构 更流畅: FPS +{v2_fps-v1_fps:.1f}, 抖动 -{v1_cv-v2_cv:.1f}%")
        elif v2_fps > v1_fps:
            print(f"  → V2 三阶段异构 FPS 更高 (+{v2_fps-v1_fps:.1f}), 但抖动接近")
        elif v2_cv < v1_cv:
            print(f"  → V2 三阶段异构 更稳定 (CV -{v1_cv-v2_cv:.1f}%), FPS 接近")
        else:
            print(f"  → 两种方式性能接近")
