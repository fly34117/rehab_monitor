"""
heterogeneous computing rehab monitor — main entry
====================================================
三阶段异构管线:

  Stage 1 — 视频解码 → iGPU  (GStreamer VA-API / V4L2 MJPEG)
  Stage 2 — 模型推理 → iGPU  (OpenVINO) 或 NPU
  Stage 3 — 后处理   → CPU   (步态/跌倒/表情/API/显示)

用法:
  python -m rehab_optimized.main                  # 自动选择最优设备
  python -m rehab_optimized.main --gpu            # 解码=iGPU, 推理=iGPU
  python -m rehab_optimized.main --npu            # 解码=iGPU, 推理=NPU
  python -m rehab_optimized.main --cpu            # 纯CPU
  python -m rehab_optimized.main --model yolo11s  # 切换模型
"""
from __future__ import annotations

import argparse
import time
import queue
import threading
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rehab_optimized.config import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FPS_TARGET,
    USE_GSTREAMER, EMOTION_INTERVAL, EMOTION_CLASSES,
    REPORT_INTERVAL, API_PORT,
    SPATIAL_CAMERA_HEIGHT_M, SPATIAL_CAMERA_TILT_DEG, SPATIAL_CALIB_POINTS,
)
from rehab_optimized.pipeline import HeterogeneousPipeline, Stage
from rehab_optimized.monitor import PerformanceMonitor
from rehab_optimized.utils import setup_logging

# 原分析模块 (不改原代码, 仅导入)
from rehab_monitor.gait_analyzer import GaitAnalyzer
from rehab_monitor.fall_detector import FallDetector
from rehab_monitor.data_logger import RehabDatabase
from rehab_monitor.display_overlay import DisplayOverlay
from rehab_monitor.spatial_mapper import SpatialMapper
from rehab_monitor.emotion_recognizer import EmotionRecognizer, run_emotion_thread
from rehab_monitor.face_locker import FaceNetLocker
from rehab_monitor.angle_monitor import AngleMonitor
from rehab_monitor.trajectory_monitor import TrajectoryMonitor
from rehab_monitor.skeleton_viewer import SkeletonViewer
from rehab_monitor.gait_metrics_viewer import GaitMetricsViewer
from rehab_monitor.api_client import generate_report
from ubuntu.leg_corrector import LegCorrector

logger = setup_logging("rehab")


def _keypoints_to_bbox(kpts):
    if kpts is None:
        return None
    valid = kpts[kpts[:, 2] > 0.3]
    if len(valid) == 0:
        return None
    x, y = valid[:, 0], valid[:, 1]
    return np.array([x.min(), y.min(), x.max(), y.max()])


# ============================================================
# Stage 3 后处理函数 — 全部在 CPU 上执行
# ============================================================

def stage3_postprocess(
    frame, kpts, person_kpts, state, frame_num,
    gait, fall, spatial, emotion, face_locker, leg_corrector, db,
    emotion_queue, frame_ref,
):
    """Stage 3: CPU 后处理

    包含: 腿纠正 → 空间定位 → 步态分析 → 跌倒检测 → 表情识别
          → 人脸锁定 → 数据存储

    全部在 CPU 上执行，不阻塞 Stage 1/2。
    """
    t0 = time.time()

    # ---- 3a. 腿关键点纠正 ----
    kpts, leg_swapped = leg_corrector.correct(kpts)
    person_kpts = kpts[0] if kpts is not None and kpts.shape[0] > 0 else None

    # ---- 3b. 空间定位 ----
    world_pos = spatial.get_person_position(person_kpts) if person_kpts is not None else None
    trajectory = spatial.get_trajectory()

    # ---- 3c. 步态分析 ----
    angles = gait.compute_joint_angles(person_kpts) if person_kpts is not None else {}
    step_count = gait.update(person_kpts, time.time(), world_pos) if person_kpts is not None else 0

    # ---- 3d. 跌倒检测 ----
    bbox = _keypoints_to_bbox(person_kpts)
    depth_m = getattr(spatial, '_torso_depth', 3.0) or 3.0
    fall_status, fall_score = fall.update(person_kpts, bbox, frame.shape[0], depth_m)

    # ---- 3e. 表情识别 (后台线程, 这里触发) ----
    emotion_bbox = None
    if person_kpts is not None:
        emotion.set_keypoints(person_kpts)
        if frame_num % EMOTION_INTERVAL == 0:
            frame_ref[0] = frame.copy()

    # ---- 3f. 人脸识别锁定 ----
    target_enrolled = False
    target_similarity = 0.0
    target_name = ""
    target_match_source = "none"
    if person_kpts is not None and frame_num % 15 == 0:
        try:
            enrolled, sim, name, source = face_locker.match(frame, person_kpts)
            target_enrolled = enrolled
            target_similarity = sim
            target_name = name
            target_match_source = source
        except Exception:
            pass

    # ---- 3g. 人脸缩略图 ----
    face_thumbnail = None
    if person_kpts is not None:
        fp = person_kpts[[0, 1, 2, 3, 4], :2]
        v = fp[(fp[:, 0] > 0) & (fp[:, 1] > 0)]
        if len(v) >= 2:
            x1, y1 = v.min(axis=0).astype(int)
            x2, y2 = v.max(axis=0).astype(int)
            h_img, w_img = frame.shape[:2]
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(w_img, x2 + 10), min(h_img, y2 + 10)
            if x2 > x1 and y2 > y1:
                face_thumbnail = frame[y1:y2, x1:x2]
                emotion_bbox = (x1, y1, x2, y2)

    # ---- 3h. 数据存储 (每5帧) ----
    if frame_num % 5 == 0 and person_kpts is not None:
        db.write_frame_snapshot(frame_num, kpts, angles, fall_status, fall_score)
        m = gait.get_metrics()
        db.write_gait_metrics(frame_num,
                              m["step_count"], m["cadence_spm"],
                              m["speed_pxps"], m["symmetry"],
                              m["stride_length_m"], m["gait_velocity_mps"],
                              m["avg_step_width_m"], m["stance_percentage"],
                              m["left_knee_rom"], m["right_knee_rom"],
                              m["step_time_cv"], m["step_length_cv"],
                              m["foot_clearance_cm"], m["gait_rehab_score"],
                              m["trunk_sway_deg"], m["double_support_ratio"])
        if fall_status == "alert":
            db.write_fall_alert(frame_num)
        if state.get("emotion_label", "neutral") != "neutral":
            db.write_emotion(frame_num, state["emotion_label"],
                             state["emotion_scores"], None)

    # ---- 更新状态字典 (与原始 rehab_monitor/main.py 完全一致) ----
    state.update({
        "person_count": kpts.shape[0] if kpts is not None else 0,
        "keypoints": kpts,
        "angles": angles,
        "step_count": step_count,
        "symmetry": gait.symmetry,
        "speed": gait.speed,
        "cadence": gait.cadence,
        "stride_length_m": gait.stride_length_m,
        "gait_velocity_mps": gait.gait_velocity_mps,
        "trunk_sway_deg": gait.trunk_sway_angle,
        "is_double_support": gait.is_double_support,
        "double_support_ratio": gait.double_support_ratio,
        "fall_status": fall_status,
        "fall_score": fall_score,
        "torso_angle": getattr(spatial, '_torso_angle_deg', 0),
        "is_lying_down": spatial.is_lying_down,
        "lying_type": fall._lying_classification,
        "target_enrolled": target_enrolled,
        "target_similarity": target_similarity,
        "target_name": target_name,
        "target_match_source": target_match_source,
        "target_face": getattr(face_locker, 'reference_face', None),
        "emotion_bbox": emotion_bbox,
        "face_thumbnail": face_thumbnail,
        "world_pos": world_pos,
        "trajectory": trajectory,
        "spatial_calibrated": spatial.calibrated,
    })

    post_ms = (time.time() - t0) * 1000
    return state, post_ms


# ============================================================
# 管线拓扑可视化
# ============================================================

def _draw_topology_bar(output, pipeline, MODEL_TAG):
    """在画面底部绘制三阶段拓扑条"""
    h, w = output.shape[:2]
    bar_y = h - 20
    bar_h = 20
    panel_x = w - 420  # 足够容纳三阶段信息

    # 半透明背景
    overlay = output.copy()
    cv2.rectangle(overlay, (panel_x - 5, bar_y - 2), (w, h), (20, 20, 20), -1)
    output[:] = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

    # 三阶段指示灯 — 绿=最优设备, 黄=回退设备
    stages = [
        ("①Decode", "iGPU" if pipeline._use_gst else "CPU"),
        ("②Infer", pipeline.device.upper()),
        ("③Post", "CPU"),
    ]
    x_pos = panel_x
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, dev in stages:
        color = (0, 255, 100) if dev in ("iGPU", "NPU") else (0, 200, 255)
        cv2.putText(output, f"{label}:{dev}", (x_pos, bar_y + 14),
                    font, 0.38, color, 1)
        x_pos += 110

    # 模型标签
    cv2.putText(output, f"{MODEL_TAG} | {pipeline.monitor.fps():.0f}FPS",
                (panel_x, bar_y - 6), font, 0.35, (200, 200, 200), 1)


def _draw_help(frame):
    """快捷键帮助覆盖层"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 2 - 200, h // 2 - 160),
                  (w // 2 + 200, h // 2 + 160), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
    lines = [
        "KEYBOARD SHORTCUTS", "",
        "q - Quit", "s - Screenshot",
        "f - Toggle FPS", "h - Toggle Help",
        "SPACE - Pause/Resume", "r - Reset Tracking",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        y = h // 2 - 110 + i * 28
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        size = 0.7 if i == 0 else 0.55
        cv2.putText(frame, line, (w // 2 - 130, y), font, size, color, 1)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="异构计算康复监测系统")
    parser.add_argument("--gpu", action="store_true", help="解码=iGPU 推理=iGPU")
    parser.add_argument("--npu", action="store_true", help="解码=iGPU 推理=NPU")
    parser.add_argument("--cpu", action="store_true", help="纯CPU模式")
    parser.add_argument("--model", default="yolo11n",
                        choices=["yolo11n", "yolo11s", "yolo26n"],
                        help="选择模型 (默认 yolo11n)")
    parser.add_argument("--camera", type=int, default=CAMERA_ID, help="摄像头 ID")
    parser.add_argument("--api-port", type=int, default=API_PORT, help="API 端口")
    parser.add_argument("--api-debug", action="store_true", help="Flask 调试模式")
    parser.add_argument("--gstreamer", action="store_true", help="启用 GStreamer VA-API 硬件解码")
    parser.add_argument("--no-diag", action="store_true", help="关闭诊断窗口")
    parser.add_argument("--debug-video", action="store_true", help="调试模式: 仅显示原始帧")
    parser.add_argument("--no-api", action="store_true", help="关闭 API 服务器")
    args = parser.parse_args()

    # ---- 确定主推理设备 ----
    device = None
    if args.npu:
        device = "npu"
    elif args.gpu:
        device = "gpu"
    elif args.cpu:
        device = "cpu"

    # ================================================================
    # 1. 三阶段异构管线
    # ================================================================
    logger.info("初始化三阶段异构管线...")
    pipeline = HeterogeneousPipeline(
        model_name=args.model,
        device=device,
        camera_id=args.camera,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=CAMERA_FPS_TARGET,
        use_gstreamer=args.gstreamer,
    )
    MODEL_TAG = f"{args.model}-pose {pipeline.loader.precision} {pipeline.device}"
    logger.info("管线就绪: %s", MODEL_TAG)

    # ================================================================
    # 2. Stage 3 模块 — 全部在 CPU 上
    # ================================================================
    gait = GaitAnalyzer()
    fall = FallDetector()
    db = RehabDatabase()
    overlay = DisplayOverlay()
    spatial = SpatialMapper()
    spatial.camera_height_m = SPATIAL_CAMERA_HEIGHT_M
    spatial.camera_tilt_deg = SPATIAL_CAMERA_TILT_DEG
    if SPATIAL_CALIB_POINTS and len(SPATIAL_CALIB_POINTS) >= 4:
        img_pts = [(p[0], p[1]) for p in SPATIAL_CALIB_POINTS]
        wld_pts = [(p[2], p[3]) for p in SPATIAL_CALIB_POINTS]
        spatial.calibrate_from_points(img_pts, wld_pts)

    emotion = EmotionRecognizer()
    face_locker = FaceNetLocker()
    leg_corrector = LegCorrector()

    # 诊断窗口
    angle_monitor = AngleMonitor()
    traj_monitor = TrajectoryMonitor()
    skeleton_viewer = SkeletonViewer()
    gait_viewer = GaitMetricsViewer()

    # ================================================================
    # 3. 表情后台线程
    # ================================================================
    frame_ref = [None]
    emotion_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    emotion_thread = threading.Thread(
        target=run_emotion_thread,
        args=(emotion, stop_event, frame_ref, emotion_queue, EMOTION_INTERVAL),
        daemon=True,
    )
    emotion_thread.start()

    # ================================================================
    # 4. 数据库 + API 服务器
    # ================================================================
    session_id = db.start_session()
    logger.info("会话 #%d 开始", session_id)

    if not args.no_api:
        from rehab_monitor.api_server import start_api_server
        api_thread = start_api_server(
            gait, spatial, db, face_locker, pipeline, emotion,
            debug=args.api_debug, port=args.api_port,
        )
        logger.info("API 服务器: http://0.0.0.0:%d", args.api_port)
    else:
        logger.info("API 服务器: 已关闭")

    # ================================================================
    # 5. 主循环 — 显式三阶段
    # ================================================================
    running = True
    paused = False
    show_fps = True
    show_help = False
    frame_num = 0
    last_report_time = time.time()
    emotion_label = "neutral"
    emotion_scores = {c: 0.0 for c in EMOTION_CLASSES}

    logger.info("实时监测开始 (三阶段异构管线)...")
    logger.info("快捷键: q=退出 s=截图 f=FPS h=帮助 space=暂停 r=重置")

    try:
        while running:
            # ========================================================
            # Stage 1 — 视频解码 (iGPU/CPU)
            # ========================================================
            ret, frame, cap_ts = pipeline.stage_decode()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            frame_num += 1

            # 暂停
            if paused:
                if show_help:
                    _draw_help(frame)
                cv2.imshow("Rehab Monitor — Heterogeneous", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '):
                    paused = False
                elif key == ord('q'):
                    running = False
                continue

            # ========================================================
            # Stage 2 — 模型推理 (iGPU/NPU)
            # ========================================================
            annotated, kpts = pipeline.stage_infer(frame)

            # 帧校验
            if annotated is None or annotated.size == 0:
                logger.error("annotated 帧为空!")
                continue

            # 表情投票 (从后台线程取结果 — 在 Stage3 之前更新)
            try:
                raw_label, raw_scores, emo_bbox, face_roi = emotion_queue.get_nowait()
                if face_roi is not None:
                    emotion_label, emotion_scores = emotion.predict_voted(face_roi)
                else:
                    emotion_label, emotion_scores = raw_label, raw_scores
            except (queue.Empty, ValueError):
                pass

            # ========================================================
            # Stage 3 — 后处理 (CPU)
            # ========================================================
            state = {
                "fps": pipeline.monitor.fps(),
                "show_fps": show_fps,
                "use_kalman": pipeline.use_kalman,
                "frame_num": frame_num,
                "emotion_label": emotion_label,
                "emotion_scores": emotion_scores,
            }

            person_kpts = kpts[0] if kpts is not None and kpts.shape[0] > 0 else None
            state, post_ms = stage3_postprocess(
                frame, kpts, person_kpts, state, frame_num,
                gait, fall, spatial, emotion, face_locker, leg_corrector, db,
                emotion_queue, frame_ref,
            )
            pipeline.monitor.record_postprocess(post_ms)

            # ---- 调试模式 ----
            if args.debug_video:
                if frame_num == 1:
                    cv2.imwrite("/tmp/debug_frame.jpg", annotated)
                    logger.info("首帧已保存: /tmp/debug_frame.jpg")
                try:
                    cv2.imshow("Rehab Monitor — DEBUG", annotated)
                except Exception as e:
                    logger.error("imshow 失败: %s", e)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    running = False
                continue

            # ---- 画面叠加 ----
            try:
                output = overlay.render(annotated, state)
            except Exception as e:
                logger.error("overlay.render 失败: %s", e)
                output = annotated

            # ---- 三阶段拓扑条 (覆盖原硬编码 MODEL) ----
            _draw_topology_bar(output, pipeline, MODEL_TAG)

            if show_help:
                _draw_help(output)
            cv2.imshow("Rehab Monitor — Heterogeneous", output)

            # ---- 诊断窗口 (每3帧) ----
            if frame_num % 3 == 0 and not args.no_diag:
                angles = state.get("angles", {})
                if angles:
                    angle_monitor.update(angles)
                    angle_monitor.render()
                world_pos = state.get("world_pos")
                if world_pos is not None:
                    yaw = angles.get("orientation_yaw", 0)
                    traj_monitor.update(world_pos, state.get("trajectory", []),
                                        yaw, 1.0, 1.0,
                                        getattr(spatial, '_torso_depth', 3.0),
                                        getattr(spatial, '_ground_depth', 3.0))
                    traj_monitor.render()
                if person_kpts is not None:
                    skeleton_viewer.update(person_kpts)
                    skeleton_viewer.render(
                        state.get("fall_status", "safe"),
                        angles.get("orientation_yaw", 0),
                        getattr(spatial, '_torso_angle_deg', 0),
                        spatial.is_lying_down,
                    )
                gait_viewer.update(gait.get_metrics())
                gait_viewer.render()

            # ---- 周期性报告 ----
            now = time.time()
            if now - last_report_time >= REPORT_INTERVAL and db.session_id is not None:
                last_report_time = now
                sid = db.session_id

                def _do_report():
                    try:
                        data = db.get_recent_data(seconds=int(REPORT_INTERVAL))
                        text, summary, err = generate_report(data)
                        db.write_report(text or "", summary, err)
                    except Exception as e:
                        logger.error("报告失败: %s", e)
                threading.Thread(target=_do_report, daemon=True).start()

            # ---- 性能报告 ----
            pipeline.monitor.periodic_report(
                5.0, f"{MODEL_TAG} | {pipeline.topology.inline()}"
            )

            # ---- 键盘 ----
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                logger.info("用户退出")
                running = False
            elif key == ord('s'):
                fname = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, output)
                logger.info("截图: %s", fname)
            elif key == ord('f'):
                show_fps = not show_fps
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord(' '):
                paused = not paused
                logger.info("暂停" if paused else "恢复")
            elif key == ord('r'):
                pipeline.kalman.reset_tracking() if pipeline.kalman else None
                gait = GaitAnalyzer()
                fall = FallDetector()
                logger.info("已重置")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
    finally:
        # ---- 清理 ----
        stop_event.set()
        emotion_thread.join(timeout=2.0)

        logger.info("性能统计:")
        logger.info("  Stage1 解码: %.1fms", pipeline.monitor.snapshot()["capture_ms"])
        logger.info("  Stage2 推理: %.1fms (平均)", pipeline.avg_inference_ms)
        logger.info("  Stage3 后处理: %.1fms", pipeline.monitor.snapshot()["postprocess_ms"])
        s = pipeline.monitor.snapshot()
        logger.info("  管线 FPS: %.1f  丢帧率: %.1f%%",
                     s["pipeline_fps"], s["drop_rate_pct"])

        db.end_session()
        db.close()
        angle_monitor.close()
        traj_monitor.close()
        skeleton_viewer.close()
        gait_viewer.close()
        pipeline.release()
        cv2.destroyAllWindows()
        logger.info("会话 #%d 结束", session_id)


if __name__ == "__main__":
    main()
