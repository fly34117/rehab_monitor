"""主循环 — 三模态康复监测系统入口"""
import argparse
import time
import queue
import threading
from collections import deque
import cv2
import numpy as np

from .config import (
    POSE_MODEL_PATH, POSE_DEVICE, USE_KALMAN, CAMERA_ID,
    REPORT_INTERVAL, EMOTION_INTERVAL,
    API_PORT,
    SPATIAL_CAMERA_HEIGHT_M, SPATIAL_CAMERA_TILT_DEG,
    SPATIAL_CALIB_POINTS,
    EMOTION_CLASSES,
    BREATHING_DURATION, BREATHING_MIN_BPM, BREATHING_MAX_BPM, BREATHING_SIGMA,
)
from .pose_detector import PoseDetector
from .gait_analyzer import GaitAnalyzer
from .fall_detector import FallDetector
from .data_logger import RehabDatabase
from .display_overlay import DisplayOverlay
from .angle_monitor import AngleMonitor
from .spatial_mapper import SpatialMapper
from .trajectory_monitor import TrajectoryMonitor
from .skeleton_viewer import SkeletonViewer
from .gait_metrics_viewer import GaitMetricsViewer
from .api_client import generate_report
from .emotion_recognizer import EmotionRecognizer, run_emotion_thread
from .face_locker import FaceNetLocker
from .breathing_detector import (
    BreathingDetector, draw_breathing_waveform, draw_breathing_result,
    draw_breathing_result_frame,
)
from .logging_setup import setup_logging, get_logger

setup_logging()
logger = get_logger("main")

# 摄像头重连参数
CAM_RECONNECT_DELAY = 1.0      # 重连间隔 (秒)
CAM_MAX_RECONNECT_ATTEMPTS = 10


def _keypoints_to_bbox(kpts):
    if kpts is None:
        return None
    valid = kpts[kpts[:, 2] > 0.3]
    if len(valid) == 0:
        return None
    x = valid[:, 0]
    y = valid[:, 1]
    return np.array([x.min(), y.min(), x.max(), y.max()])


def _open_camera(camera_id):
    """打开摄像头，失败返回 None"""
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("摄像头已打开 (ID=%d)", camera_id)
        return cap
    return None


def _reconnect_camera(old_cap, preferred_id):
    """释放旧摄像头并尝试重新打开，返回新 cap 或 None"""
    if old_cap is not None:
        try:
            old_cap.release()
        except Exception:
            pass
    for attempt in range(CAM_MAX_RECONNECT_ATTEMPTS):
        logger.info("摄像头重连尝试 %d/%d...", attempt + 1, CAM_MAX_RECONNECT_ATTEMPTS)
        time.sleep(CAM_RECONNECT_DELAY)
        cap = _open_camera(preferred_id)
        if cap is not None:
            return cap
        if preferred_id != 0:
            cap = _open_camera(0)
            if cap is not None:
                return cap
    return None


def _draw_help_overlay(frame):
    """在帧上绘制快捷键帮助覆盖层"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # 半透明背景
    cv2.rectangle(overlay, (w // 2 - 200, h // 2 - 175),
                  (w // 2 + 200, h // 2 + 175), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

    lines = [
        "KEYBOARD SHORTCUTS",
        "",
        "q - Quit",
        "s - Screenshot",
        "f - Toggle FPS display",
        "h - Toggle this help",
        "SPACE - Pause / Resume",
        "r - Reset Kalman tracking",
        "t - Enroll target face",
        "c - Clear gait data (DB)",
        "b - Breathing detection (after t)",
    ]
    y0 = h // 2 - 125
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        y = y0 + i * 28
        if i == 0:
            cv2.putText(frame, line, (w // 2 - 130, y), font, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, line, (w // 2 - 170, y), font, 0.55, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="康复监测系统")
    parser.add_argument('--api-debug', action='store_true', help='启用API调试模式')
    parser.add_argument('--api-port', type=int, default=API_PORT, help='API服务器端口')
    parser.add_argument('--camera', type=int, default=CAMERA_ID, help='摄像头ID (默认0)')
    api_args = parser.parse_args()

    # ---- 初始化 ----
    detector = PoseDetector(model_path=POSE_MODEL_PATH, device=POSE_DEVICE,
                            use_kalman=USE_KALMAN)
    gait = GaitAnalyzer()
    fall = FallDetector()
    db = RehabDatabase()
    overlay = DisplayOverlay()
    angle_monitor = AngleMonitor()
    traj_monitor = TrajectoryMonitor()
    skeleton_viewer = SkeletonViewer()
    gait_metrics_viewer = GaitMetricsViewer()

    # 空间定位
    spatial = SpatialMapper()
    spatial.camera_height_m = SPATIAL_CAMERA_HEIGHT_M
    spatial.camera_tilt_deg = SPATIAL_CAMERA_TILT_DEG
    if SPATIAL_CALIB_POINTS and len(SPATIAL_CALIB_POINTS) >= 4:
        img_pts = [(p[0], p[1]) for p in SPATIAL_CALIB_POINTS]
        wld_pts = [(p[2], p[3]) for p in SPATIAL_CALIB_POINTS]
        spatial.calibrate_from_points(img_pts, wld_pts)
    emotion_recognizer = EmotionRecognizer()
    face_locker = FaceNetLocker()
    breathing_det = BreathingDetector(sigma=BREATHING_SIGMA,
                                       min_bpm=BREATHING_MIN_BPM,
                                       max_bpm=BREATHING_MAX_BPM)

    # ---- 表情后台线程 ----
    frame_ref = [None]        # 共享帧引用
    emotion_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    emotion_thread = threading.Thread(
        target=run_emotion_thread,
        args=(emotion_recognizer, stop_event, frame_ref, emotion_queue, EMOTION_INTERVAL),
        daemon=True,
    )
    emotion_thread.start()

    session_id = db.start_session()
    logger.info("会话 #%d 开始", session_id)

    # 启动 API 服务器 (后台 daemon 线程)
    from .api_server import start_api_server
    api_thread = start_api_server(
        gait, spatial, db, face_locker, detector, emotion_recognizer,
        debug=api_args.api_debug, port=api_args.api_port
    )
    logger.info("API 服务器已启动 http://0.0.0.0:%d (debug=%s)",
                api_args.api_port, api_args.api_debug)

    # ---- 循环状态 ----
    running = True
    frame_num = 0
    fps = 0.0
    frame_count = 0
    prev_fps_time = time.time()
    last_report_time = time.time()
    show_fps = True
    show_help = False
    paused = False

    # 跌倒告警去重
    previous_fall_status = "safe"
    _last_fall_push_time = 0
    FALL_PUSH_COOLDOWN = 5  # 冷却时间 (秒)

    # 表情状态
    emotion_label = "neutral"
    emotion_scores = {c: 0.0 for c in EMOTION_CLASSES}
    emotion_bbox = None

    logger.info("实时康复监测开始...")
    target_enrolled = False
    target_person_id = -1  # -1 = 未锁定/搜索中
    target_similarity = 0.0
    target_name = ""
    target_match_source = "none"  # 'face' / 'body' / 'none'
    fall_status = "safe"         # 预初始化，供匹配段使用

    # 目标锁定迟滞状态：快速锁定、缓慢释放，防止闪烁
    lock_confidence = 0           # 0-30
    LOCK_INCREMENT = 2            # 每帧匹配成功 +2
    LOCK_DECREMENT = 1            # 每帧匹配失败 -1
    LOCK_ENTER_THRESH = 5         # 锁定阈值
    LOCK_MAX = 30

    def _save_lock_state(name):
        """持久化锁定状态到 lock_state.json（重启后自动恢复）"""
        import json as _json, os as _os
        _os.makedirs("model", exist_ok=True)
        with open("model/lock_state.json", "w") as _f:
            _json.dump({"target_name": name, "timestamp": time.time()}, _f)

    def _load_lock_state():
        """读取持久化的锁定状态，返回 target_name 或 None"""
        import os as _os, json as _json
        path = _os.path.join(_os.path.dirname(__file__), "..", "model", "lock_state.json")
        if _os.path.exists(path):
            with open(path, "r") as _f:
                data = _json.load(_f)
            return data.get("target_name")
        return None

    # 启动时尝试恢复上次的锁定目标（进入搜索模式）
    _saved_target = _load_lock_state()
    if _saved_target:
        target_enrolled = True
        target_name = _saved_target
        target_person_id = -1  # 搜索中，等待人脸匹配
        lock_confidence = LOCK_ENTER_THRESH  # 跳过迟滞，快速锁定
        logger.info("恢复锁定目标: %s（搜索中...）", _saved_target)

    logger.info("快捷键: q=退出 s=截图 f=切换FPS r=重置 c=清空数据 b=呼吸 h=帮助 space=暂停")

    # ── 呼吸检测状态 (非阻塞状态机: idle → active → result → idle) ──
    breathing_state = 'idle'          # 'idle' | 'active' | 'result'
    breathing_roi = None              # (x, y, w, h) 锁定 ROI
    breathing_start_time = 0.0
    breathing_result_start_time = 0.0
    breathing_movements = []          # 运动信号序列
    breathing_wave_buf = deque(maxlen=300)
    breathing_frame_count = 0
    breathing_prev_gray = None
    breathing_bpm = 0.0
    breathing_win = "Breathing Waveform"

    def _start_breathing(target_kpts, ref_frame):
        """初始化呼吸检测状态 (非阻塞, 在主循环中逐帧采集)"""
        nonlocal breathing_state, breathing_roi, breathing_start_time
        nonlocal breathing_movements, breathing_wave_buf, breathing_frame_count
        nonlocal breathing_prev_gray, breathing_bpm

        roi = BreathingDetector.get_chest_roi_from_keypoints(
            target_kpts, ref_frame.shape)
        if roi is None:
            logger.error("无法从目标关键点计算胸腔 ROI, 请确保目标正对摄像头")
            return False

        breathing_roi = roi
        breathing_start_time = time.time()
        breathing_movements = []
        breathing_wave_buf = deque(maxlen=300)
        breathing_frame_count = 0
        breathing_prev_gray = None
        breathing_bpm = 0.0
        breathing_state = 'active'

        cv2.namedWindow(breathing_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(breathing_win, 640, 400)
        logger.info("呼吸检测开始: ROI=%s, 时长=%ds (非阻塞模式)", roi, BREATHING_DURATION)
        return True

    def _cleanup_breathing():
        """清理呼吸检测状态, 关闭波形窗口"""
        nonlocal breathing_state, breathing_roi, breathing_prev_gray
        breathing_state = 'idle'
        breathing_roi = None
        breathing_prev_gray = None
        try:
            cv2.destroyWindow(breathing_win)
        except Exception:
            pass

    logger.info("快捷键: q=退出 s=截图 f=切换FPS r=重置(清除锁定+人脸库) h=帮助 space=暂停 t=录入人脸 b=呼吸检测(需先t锁定) c=清空数据")

    # 延迟打开摄像头 — 等 OpenVINO 模型编译完成后再开，避免 MSMF 流超时
    cap = _open_camera(api_args.camera)
    if cap is None:
        logger.error("无法打开摄像头 (ID=%d)，尝试回退到 ID=0...", api_args.camera)
        cap = _open_camera(0)
    if cap is None:
        logger.critical("所有摄像头尝试失败，退出")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("摄像头读取失败，尝试重连...")
                cap = _reconnect_camera(cap, api_args.camera)
                if cap is None:
                    logger.error("摄像头重连失败，退出")
                    break
                continue

            t_start = time.time()
            frame_num += 1

            # 暂停时跳过所有处理，仅显示画面
            if paused:
                if show_help:
                    _draw_help_overlay(frame)
                cv2.imshow("Rehab Monitor - YOLOv26 Pose + Gait + Fall + Emotion", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    paused = False
                    logger.info("恢复")
                elif key == ord('q'):
                    logger.info("用户退出")
                    running = False
                elif key == ord('h'):
                    show_help = not show_help
                continue

            # 1. 姿态检测 + 卡尔曼
            annotated, kpts = detector.process_frame(frame)

            # 1.5a 空间过滤（每帧）：已锁定时用卡尔曼轨道隔离目标，杜绝跳变
            kpts_all = kpts  # 保存未过滤副本，供人脸恢复扫描使用
            if target_enrolled and target_person_id >= 0 and kpts is not None and kpts.shape[0] > 0:
                target_det = detector.get_target_det_idx()
                if target_det is not None and 0 <= target_det < kpts.shape[0]:
                    if kpts.shape[0] > 1:
                        kpts = kpts[target_det:target_det + 1]
                    target_person_id = 0
                elif spatial.is_lying_down or fall_status == "alert":
                    # 躺下/跌倒期间轨道可能暂时失配, 不误解锁
                    pass
                else:
                    # 目标轨道消失（人离开画面）→ 立即解锁
                    target_person_id = -1
                    target_similarity = 0.0
                    target_match_source = "none"
                    lock_confidence = max(0, lock_confidence - LOCK_DECREMENT * 3)

            # 1.5b API 待处理解锁请求（来自微信小程序）
            try:
                from .api_server import consume_pending_unlock
                if consume_pending_unlock():
                    target_enrolled = False
                    target_person_id = -1
                    target_name = ""
                    target_similarity = 0.0
                    target_match_source = "none"
                    lock_confidence = 0
                    face_locker.reference_body_feat = None
                    try:
                        import os as _os
                        _os.remove("model/lock_state.json")
                        _os.remove("model/face_db.json")
                        face_locker._load_db()
                    except Exception:
                        pass
                    logger.info("API 解锁: 锁定状态和人脸库已清除")
            except Exception:
                pass

            # 1.5c API 待处理锁定请求（来自微信小程序拍照）
            lock_name, lock_emb = None, None
            try:
                from .api_server import consume_pending_lock
                lock_name, lock_emb = consume_pending_lock()
            except Exception:
                pass

            if lock_name is not None:
                # 进入搜索模式：用数据库匹配（同源），不用手机照片跨源匹配
                target_enrolled = True
                target_name = lock_name
                target_person_id = -1  # 搜索中
                target_match_source = "none"
                lock_confidence = LOCK_ENTER_THRESH  # 快速锁定
                _save_lock_state(lock_name)
                logger.info("API 锁定请求: %s → 搜索模式（等待人脸匹配）", lock_name)

            # 1.5d 外观验证（分级频率）：锁定后只验证不搜索，丢失时才搜索所有人
            if target_enrolled and kpts is not None and kpts.shape[0] > 0:
                lost = target_person_id < 0
                do_face = lost or (frame_num % 15 == 0)
                do_body = lost or (frame_num % 5 == 0)

                if do_face or do_body:
                    if not lost:
                        if spatial.is_lying_down:
                            # ---- 躺下直通: 位置唯一, 跳过外观验证, 强制锁定 ----
                            lock_confidence = min(LOCK_MAX, lock_confidence + LOCK_INCREMENT)
                            target_similarity = 1.0
                            target_match_source = "lying"
                            if kpts.shape[0] > 1:
                                kpts = kpts[0:1]
                            target_person_id = 0
                        else:
                            # ---- STICKY LOCK：只验证当前目标，不检查其他人 ----
                            person_k = kpts[0]
                            face_emb = None
                            if do_face:
                                face_pts = person_k[[0, 1, 2, 3, 4], :2]
                                valid_fp = face_pts[(face_pts[:, 0] > 0) & (face_pts[:, 1] > 0)]
                                if len(valid_fp) >= 2:
                                    x1, y1 = valid_fp.min(axis=0).astype(int)
                                    x2, y2 = valid_fp.max(axis=0).astype(int)
                                    face_emb = face_locker.extract_embedding_from_roi(frame, x1, y1, x2, y2)

                            if do_body or face_emb is not None:
                                is_match, sim, source = face_locker.match_combined(frame, person_k, face_emb)
                            else:
                                is_match, sim, source = False, 0.0, "none"

                            # 人脸恢复扫描
                            best_face_pid = -1
                            best_face_sim = 0.0
                            if do_face and kpts_all is not None and kpts_all.shape[0] > 1:
                                target_det = detector.get_target_det_idx()
                                for pid in range(kpts_all.shape[0]):
                                    if pid == target_det:
                                        continue
                                    pk = kpts_all[pid]
                                    fp = pk[[0, 1, 2, 3, 4], :2]
                                    vfp = fp[(fp[:, 0] > 0) & (fp[:, 1] > 0)]
                                    if len(vfp) >= 2:
                                        x1, y1 = vfp.min(axis=0).astype(int)
                                        x2, y2 = vfp.max(axis=0).astype(int)
                                        other_face = face_locker.extract_embedding_from_roi(frame, x1, y1, x2, y2)
                                        if other_face is not None:
                                            name, fs = face_locker.match(other_face, threshold=0.50)
                                            if name is not None and fs > best_face_sim:
                                                best_face_sim = fs
                                                best_face_pid = pid

                            # 融合判定
                            if best_face_pid >= 0 and best_face_sim > sim + 0.05:
                                logger.debug("人脸恢复: 检测%d face_sim=%.3f > 当前%.3f",
                                             best_face_pid, best_face_sim, sim)
                                lock_confidence = min(LOCK_MAX, lock_confidence + LOCK_INCREMENT * 2)
                                target_similarity = best_face_sim
                                target_match_source = "face"
                                kpts = kpts_all[best_face_pid:best_face_pid + 1]
                                target_person_id = 0
                                detector.target_track_id = detector._det_to_track.get(best_face_pid)
                            elif is_match:
                                lock_confidence = min(LOCK_MAX, lock_confidence + LOCK_INCREMENT)
                                target_similarity = sim
                                target_match_source = source
                            elif fall_status == "alert":
                                pass  # 跌倒期间不降置信度
                            else:
                                lock_confidence = max(0, lock_confidence - LOCK_DECREMENT)
                                if lock_confidence <= 0:
                                    logger.debug("目标丢失")
                                    target_person_id = -1
                                    target_similarity = 0.0
                                    target_match_source = "none"
                    else:
                        # ---- SEARCHING：检查所有检测找到目标 ----
                        # 有指定目标名时（API/恢复锁定），用极低阈值匹配（手机照片 vs 远距离摄像头）
                        _orig_face_th = face_locker.face_threshold
                        _orig_body_th = face_locker.body_threshold
                        if target_name:
                            face_locker.face_threshold = 0.30   # 极宽：远距离跨源匹配
                            face_locker.body_threshold = 0.55   # 放宽

                        best = {"pid": -1, "sim": 0.0, "source": "none", "emb": None}
                        for pid in range(kpts.shape[0]):
                            person_k = kpts[pid]
                            face_emb = None
                            matched_name = None
                            if do_face:
                                face_pts = person_k[[0, 1, 2, 3, 4], :2]
                                valid_fp = face_pts[(face_pts[:, 0] > 0) & (face_pts[:, 1] > 0)]
                                if len(valid_fp) >= 1:
                                    # 单点也尝试（用大边距裁剪）
                                    x1, y1 = valid_fp.min(axis=0).astype(int)
                                    x2, y2 = valid_fp.max(axis=0).astype(int)
                                    if x1 == x2: x2 = x1 + 40
                                    if y1 == y2: y2 = y1 + 40
                                    face_emb = face_locker.extract_embedding_from_roi(frame, x1, y1, x2, y2)
                                if face_emb is None:
                                    face_emb = face_locker.enroll_from_frame(frame)
                                # 验证匹配的人名
                                if face_emb is not None and target_name:
                                    matched_name, _ = face_locker.match(face_emb, threshold=0.30)

                            if do_body or face_emb is not None:
                                is_match, sim, source = face_locker.match_combined(frame, person_k, face_emb)
                                # 有指定目标名时，优先接受名称匹配的人；也接受无名但相似度够高的
                                if is_match and target_name and matched_name:
                                    if matched_name != target_name:
                                        is_match = False  # 匹配到错误的人
                                    else:
                                        sim += 0.10  # 名称匹配加分
                                if is_match and sim > best["sim"]:
                                    best = {"pid": pid, "sim": sim, "source": source, "emb": face_emb}

                        # 恢复原始阈值
                        face_locker.face_threshold = _orig_face_th
                        face_locker.body_threshold = _orig_body_th

                        if best["pid"] >= 0:
                            lock_confidence = min(LOCK_MAX, lock_confidence + LOCK_INCREMENT)
                            if lock_confidence >= LOCK_ENTER_THRESH:
                                if target_person_id < 0:
                                    logger.info("目标已锁定! sim=%.3f source=%s (宽阈值搜索)",
                                                best["sim"], best["source"])
                                # 先提取目标关键点（过滤前），再缩小 kpts
                                target_kpts = kpts[best["pid"]]
                                if kpts.shape[0] > 1:
                                    kpts = kpts[best["pid"]:best["pid"] + 1]
                                target_person_id = 0
                                target_similarity = best["sim"]
                                target_match_source = best["source"]
                                detector.target_track_id = detector._det_to_track.get(best["pid"])
                                # 录入人体外观特征（后续帧可用 body 匹配，不需要人脸）
                                face_locker.enroll_body(frame, target_kpts)
                                # 用摄像头高质量人脸更新数据库嵌入（同源匹配更可靠）
                                if best["emb"] is not None and target_name:
                                    try:
                                        import json as _json, os as _os
                                        db_path = _os.path.join(_os.path.dirname(__file__), "..", "model", "face_db.json")
                                        if _os.path.exists(db_path):
                                            with open(db_path, "r") as _f:
                                                _db = _json.load(_f)
                                            for _e in _db.get("entries", []):
                                                if _e["name"] == target_name:
                                                    _e["embedding"] = best["emb"].tolist()
                                                    break
                                            with open(db_path, "w") as _f:
                                                _json.dump(_db, _f)
                                            face_locker._load_db()
                                            logger.info("已用摄像头人脸更新 %s 的嵌入", target_name)
                                    except Exception:
                                        pass
                        else:
                            lock_confidence = max(0, lock_confidence - LOCK_DECREMENT)
                            if lock_confidence <= 0 and target_person_id >= 0:
                                logger.debug("目标丢失")
                                target_person_id = -1
                                target_similarity = 0.0
                                target_match_source = "none"

            # 在画面上标记目标（绿色高亮框）
            if target_enrolled and target_person_id >= 0 and kpts is not None and kpts.shape[0] > 0:
                tk = kpts[0][:, :2]
                valid = tk[(tk[:, 0] > 0) & (tk[:, 1] > 0)]
                if len(valid) >= 3:
                    x1, y1 = valid.min(axis=0).astype(int)
                    x2, y2 = valid.max(axis=0).astype(int)
                    cv2.rectangle(annotated, (x1-8, y1-8), (x2+8, y2+8), (0, 255, 0), 3)
                    source_tag = "FACE" if target_match_source == "face" else "BODY"
                    source_color = (0, 255, 0) if target_match_source == "face" else (255, 200, 0)
                    cv2.putText(annotated, f"TARGET:{target_name} [{source_tag}]",
                                (x1, y1-14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, source_color, 2)
                    cv2.putText(annotated, f"sim:{target_similarity:.2f}",
                                (x1, y2+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, source_color, 1)

            # 1.6 呼吸检测数据采集 (非阻塞, 逐帧累加运动信号)
            if breathing_state == 'active' and breathing_roi is not None:
                bx, by, brw, brh = breathing_roi
                if (bx >= 0 and by >= 0
                        and bx + brw <= frame.shape[1]
                        and by + brh <= frame.shape[0]):
                    roi_frame = frame[by:by + brh, bx:bx + brw]
                    if roi_frame.size > 0:
                        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                        if breathing_prev_gray is not None:
                            motion = float(cv2.absdiff(breathing_prev_gray, gray).mean())
                            breathing_movements.append(motion)
                            breathing_wave_buf.append(motion)
                            breathing_frame_count += 1
                        breathing_prev_gray = gray

                # 检查是否到达检测时长
                elapsed = time.time() - breathing_start_time
                if elapsed >= BREATHING_DURATION:
                    # 计算 BPM, 切换到结果展示状态
                    actual_fps = (breathing_frame_count / elapsed
                                  if elapsed > 0 else 25.0)
                    if len(breathing_movements) >= actual_fps:
                        breathing_bpm = breathing_det.calculate_bpm(
                            breathing_movements, actual_fps, method='fft')
                        logger.info("呼吸检测完成: BPM=%.1f, 帧数=%d, FPS=%.1f",
                                   breathing_bpm, breathing_frame_count, actual_fps)
                    else:
                        breathing_bpm = 0.0
                        logger.warning("呼吸检测数据不足 (%d 帧)",
                                      len(breathing_movements))
                    breathing_state = 'result'
                    breathing_result_start_time = time.time()

            # 2. 步态分析 (先空间定位 → 再步态, 因为真实步长需要世界坐标)
            angles = {}
            step_count = 0
            symmetry = 1.0
            speed = 0.0
            cadence = 0.0
            bbox = None
            person_kpts = None
            world_pos = None
            trajectory = []

            if kpts is not None and kpts.shape[0] > 0 and target_enrolled and target_person_id >= 0:
                pid = target_person_id
                if pid < kpts.shape[0]:
                    person_kpts = kpts[pid]

                # 先空间定位 (步态需要 world_pos)
                world_pos = spatial.get_person_position(person_kpts)
                trajectory = spatial.get_trajectory()

                # 再步态分析
                angles = gait.compute_joint_angles(person_kpts)
                step_count = gait.update(person_kpts, t_start, world_pos)
                symmetry = gait.symmetry
                speed = gait.speed
                cadence = gait.cadence
                bbox = _keypoints_to_bbox(person_kpts)

            # 3. 跌倒检测（仅锁定后运行）
            if target_enrolled and target_person_id >= 0:
                frame_h = frame.shape[0]
                depth_m = spatial._torso_depth if spatial._torso_depth > 0 else 3.0
                fall_status, fall_score = fall.update(
                    person_kpts if kpts is not None and kpts.shape[0] > 0 else None,
                    bbox, frame_h, depth_m)
            else:
                fall_status = "safe"
                fall_score = 0.0

            # 躺下回溯分类: 先确认躯干水平, 再回看速度区分跌倒/休息（仅锁定后）
            if target_enrolled and target_person_id >= 0:
                if spatial.is_lying_down:
                    fall._lying_confirmed = True
                    lying_type = fall.classify_lying_down()
                    if lying_type == "fall" and fall_status != "alert":
                        fall_status = "alert"
                        fall_score = 0.8
                    # 通知空间定位
                    spatial.set_lying_down(True)
                else:
                    if fall._lying_confirmed:
                        fall.reset_lying_state()
                        if fall_status == "alert":
                            fall_status = "safe"
                            fall_score = 0.0

            # 跌倒告警广播 (5s 冷却去重)
            if target_enrolled and fall_status == "alert" and previous_fall_status != "alert":
                now_ts = time.time()
                if now_ts - _last_fall_push_time > FALL_PUSH_COOLDOWN:
                    try:
                        from .api_server import broadcast_fall_alert
                        loc = spatial.current_pos if spatial else (0, 0)
                        broadcast_fall_alert(loc, fall_score)
                        logger.info("跌倒告警已广播 score=%.2f pos=(%.1f,%.1f)",
                                    fall_score, loc[0], loc[1])
                    except Exception:
                        pass
                    _last_fall_push_time = now_ts
            previous_fall_status = fall_status

            # 4. 表情识别 — 传递关键点 + 更新共享帧（仅锁定后）
            if target_enrolled and target_person_id >= 0:
                emotion_recognizer.set_keypoints(person_kpts)
                if frame_num % EMOTION_INTERVAL == 0:
                    frame_ref[0] = frame.copy()
            else:
                emotion_label = "neutral"
                emotion_scores = {c: 0.0 for c in EMOTION_CLASSES}

            # 非阻塞读取表情结果 → 多帧投票
            try:
                raw_label, raw_scores, emotion_bbox, face_roi = emotion_queue.get_nowait()
                if face_roi is not None:
                    emotion_label, emotion_scores = emotion_recognizer.predict_voted(face_roi)
                else:
                    emotion_label, emotion_scores = raw_label, raw_scores
            except (queue.Empty, ValueError):
                pass

            # 裁剪人脸缩略图（主线程直接用关键点裁剪，更快更可靠）
            face_thumbnail = None
            if person_kpts is not None:
                face_pts = person_kpts[[0, 1, 2, 3, 4], :2]  # nose, eyes, ears
                valid = face_pts[(face_pts[:, 0] > 0) & (face_pts[:, 1] > 0)]
                if len(valid) >= 2:
                    x_min, y_min = valid.min(axis=0).astype(int)
                    x_max, y_max = valid.max(axis=0).astype(int)
                    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                    half = int(max(x_max - x_min, y_max - y_min) * 0.9)
                    h_img, w_img = frame.shape[:2]
                    x1 = max(0, cx - half)
                    y1 = max(0, cy - half)
                    x2 = min(w_img, cx + half)
                    y2 = min(h_img, cy + half)
                    if x2 > x1 and y2 > y1:
                        face_thumbnail = frame[y1:y2, x1:x2]

            # 5. 异步数据存储（仅锁定后每5帧写入）
            if target_enrolled and target_person_id >= 0 and frame_num % 5 == 0:
                if kpts is not None:
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
                if emotion_label != "neutral":
                    db.write_emotion(frame_num, emotion_label, emotion_scores, emotion_bbox)

            # 5.5 周期康复报告 (后台线程, 仅锁定后每30s触发)
            now = time.time()
            if target_enrolled and target_person_id >= 0 and now - last_report_time >= REPORT_INTERVAL and db.session_id is not None:
                last_report_time = now
                sid = db.session_id
                def _do_report():
                    try:
                        data = db.get_recent_data(seconds=int(REPORT_INTERVAL))
                        text, summary, err = generate_report(data)
                        db.write_report(text or "", summary, err)
                        tag = "OK" if text else f"ERR:{err[:40] if err else 'unknown'}"
                        logger.info("周期报告已生成 (%s)", tag)
                    except Exception as e:
                        logger.error("报告生成失败: %s", e)
                threading.Thread(target=_do_report, daemon=True).start()

            # 6. FPS 计算
            if show_fps:
                frame_count += 1
                if now - prev_fps_time >= 1.0:
                    fps = frame_count / (now - prev_fps_time)
                    frame_count = 0
                    prev_fps_time = now

            # 7. 画面叠加
            state = {
                "fps": fps,
                "person_count": kpts.shape[0] if kpts is not None else 0,
                "keypoints": kpts,
                "angles": angles,
                "step_count": step_count,
                "symmetry": symmetry,
                "speed": speed,
                "cadence": cadence,
                "stride_length_m": gait.stride_length_m,
                "gait_velocity_mps": gait.gait_velocity_mps,
                "trunk_sway_deg": gait.trunk_sway_angle,
                "is_double_support": gait.is_double_support,
                "double_support_ratio": gait.double_support_ratio,
                "fall_status": fall_status,
                "fall_score": fall_score,
                "torso_angle": spatial._torso_angle_deg,
                "is_lying_down": spatial.is_lying_down,
                "lying_type": fall._lying_classification,
                "show_fps": show_fps,
                "use_kalman": USE_KALMAN,
                "frame_num": frame_num,
                "target_enrolled": target_enrolled,
                "target_similarity": target_similarity,
                "target_name": target_name,
                "target_match_source": target_match_source,
                "target_face": face_locker.reference_face,
                "emotion_label": emotion_label,
                "emotion_scores": emotion_scores,
                "emotion_bbox": emotion_bbox,
                "face_thumbnail": face_thumbnail,
                "world_pos": world_pos,
                "trajectory": trajectory,
                "spatial_calibrated": spatial.calibrated,
            }
            # 同步锁定状态到 API 服务器（供微信小程序查询）
            # 只有真正跟踪到人才报告 locked=true（搜索中不算）
            try:
                from .api_server import update_current_lock
                update_current_lock(target_enrolled and target_person_id >= 0,
                                    target_name, target_similarity, target_match_source)
            except Exception:
                pass

            output = overlay.render(annotated, state)
            if show_help:
                _draw_help_overlay(output)

            # 在主画面右上角叠加呼吸检测状态 + 锁定 ROI 框
            if breathing_state == 'active':
                elapsed = time.time() - breathing_start_time
                cv2.putText(output, "BREATHING", (output.shape[1] - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
                cv2.putText(output,
                            f"{elapsed:.0f}s/{BREATHING_DURATION}s",
                            (output.shape[1] - 200, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
                if breathing_roi is not None:
                    bx, by, brw, brh = breathing_roi
                    cv2.rectangle(output, (bx, by), (bx + brw, by + brh),
                                  (255, 180, 0), 2)
                    cv2.putText(output, "ROI LOCKED", (bx, by - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)
            elif breathing_state == 'result':
                cv2.putText(output,
                            f"BREATHING BPM: {breathing_bpm:.1f}",
                            (output.shape[1] - 280, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

            cv2.imshow("Rehab Monitor - YOLOv26 Pose + Gait + Fall + Emotion", output)

            # 7.5 独立诊断窗 (每3帧刷新)
            if frame_num % 3 == 0:
                if angles:
                    angle_monitor.update(angles)
                    angle_monitor.render()
                if world_pos is not None:
                    yaw = angles.get("orientation_yaw", 0) if angles else 0
                    yaw_conf = angles.get("orientation_conf", 0) if angles else 0
                    fsign = angles.get("facing_sign", 1.0) if angles else 1.0
                    traj_monitor.update(world_pos, trajectory, yaw, yaw_conf, fsign,
                                        spatial._torso_depth, spatial._ground_depth)
                    traj_monitor.render()
                if person_kpts is not None:
                    skeleton_viewer.update(person_kpts)
                    skeleton_viewer.render(fall_status,
                                           angles.get("orientation_yaw", 0) if angles else 0,
                                           spatial._torso_angle_deg,
                                           spatial.is_lying_down)
                gait_metrics_viewer.update(gait.get_metrics())
                gait_metrics_viewer.render()

            # 7.6 呼吸检测波形窗口 (非阻塞, 独立窗口与其他诊断窗并行)
            if breathing_state == 'active':
                wave_canvas = draw_breathing_waveform(
                    breathing_wave_buf, breathing_frame_count,
                    BREATHING_DURATION,
                    cap.get(cv2.CAP_PROP_FPS) or 25.0,
                    BREATHING_MIN_BPM, BREATHING_MAX_BPM)
                cv2.imshow(breathing_win, wave_canvas)
            elif breathing_state == 'result':
                remaining = 5.0 - (time.time() - breathing_result_start_time)
                if remaining > 0:
                    result_canvas = draw_breathing_result_frame(
                        breathing_bpm, remaining)
                    cv2.imshow(breathing_win, result_canvas)
                else:
                    _cleanup_breathing()
                    logger.info("呼吸检测结果窗口已销毁 (BPM=%.1f)", breathing_bpm)

            # 8. 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("用户退出")
                running = False
            elif key == ord('s'):
                filename = f"rehab_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, output)
                logger.info("截图已保存: %s", filename)
            elif key == ord('f'):
                show_fps = not show_fps
                logger.debug("FPS 显示: %s", "ON" if show_fps else "OFF")
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord(' '):
                paused = not paused
                logger.info("暂停" if paused else "恢复")
            elif key == ord('r'):
                detector.reset_tracking()
                gait = GaitAnalyzer()
                fall = FallDetector()
                target_enrolled = False
                target_person_id = -1
                target_name = ""
                target_similarity = 0.0
                target_match_source = "none"
                lock_confidence = 0
                # 清除持久化的锁定状态和人脸库
                try:
                    import os as _os
                    _os.remove("model/lock_state.json")
                except Exception:
                    pass
                try:
                    import os as _os
                    _os.remove("model/face_db.json")
                    face_locker._load_db()  # 重新加载空库
                    face_locker.reference_body_feat = None
                except Exception:
                    pass
                logger.info("卡尔曼跟踪已重置，锁定状态和人脸库已清除")
            elif key == ord('t'):
                # 多模态录入：人脸 + 人体外观
                emb = face_locker.enroll_from_frame(frame)
                body_ok = False
                if person_kpts is not None:
                    body_ok = face_locker.enroll_body(frame, person_kpts)

                if emb is not None or body_ok:
                    import json, os
                    # 保存人脸嵌入（如果有）
                    if emb is not None:
                        entry = {
                            "name": "target",
                            "embedding": emb.tolist(),
                            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        db_data = {"entries": [entry]}
                        os.makedirs("model", exist_ok=True)
                        with open("model/face_db.json", "w") as f:
                            json.dump(db_data, f)
                        face_locker._load_db()

                    target_enrolled = True
                    target_person_id = 0
                    target_name = "target"
                    lock_confidence = LOCK_ENTER_THRESH
                    target_match_source = "face" if emb is not None else "body"
                    _save_lock_state("target")
                    # 记录目标所在的卡尔曼轨道（多人环境下用于空间定位）
                    if kpts is not None and kpts.shape[0] > 0:
                        detector.target_track_id = detector._det_to_track.get(0)
                    cues = []
                    if emb is not None:
                        cues.append("人脸")
                    if body_ok:
                        cues.append("人体外观")
                    logger.info("目标已录入! (线索: %s)", "+".join(cues))
                else:
                    logger.warning("录入失败 - 未检测到人脸且关键点不足。请正对摄像头")
            elif key == ord('c'):
                # 清空步态数据（保留会话记录）
                db.clear_gait_data()
                logger.info("步态数据已清空 (会话 #%d 继续运行)", session_id)
            elif key == ord('b'):
                # 呼吸检测: 需先用 t 锁定目标人脸, 非阻塞运行
                if breathing_state != 'idle':
                    # 正在检测中, 按 b 中断
                    logger.info("呼吸检测被用户中断 (已采集 %d 帧)",
                               breathing_frame_count)
                    _cleanup_breathing()
                elif not target_enrolled or target_person_id < 0:
                    logger.warning(
                        "请先按 't' 锁定目标人脸, 再按 'b' 开始呼吸检测")
                elif kpts is None or kpts.shape[0] == 0:
                    logger.warning("未检测到目标关键点, 无法开始呼吸检测")
                elif person_kpts is None:
                    logger.warning("目标关键点无效, 无法定位胸腔 ROI")
                else:
                    _start_breathing(person_kpts, frame)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt 中断退出")
    finally:
        # ---- 清理 ----
        stop_event.set()
        emotion_thread.join(timeout=2.0)

        logger.info("性能统计:")
        logger.info("  平均推理时间: %.1fms", detector.avg_inference_ms)
        if USE_KALMAN and detector.kalman_times:
            logger.info("  卡尔曼滤波耗时: %.1fms", detector.avg_kalman_ms)

        db.end_session()
        db.close()
        angle_monitor.close()
        traj_monitor.close()
        skeleton_viewer.close()
        gait_metrics_viewer.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("会话 #%d 结束", session_id)


if __name__ == "__main__":
    main()
