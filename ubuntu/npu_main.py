"""Ubuntu 启动入口 — 不修改原项目，运行时覆盖配置
支持所有 OpenVINO 模型的 CPU/GPU/NPU 切换
"""
import os, sys, cv2, numpy as np, time
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

# ---- 手机轨迹独立窗口 ----
_PHONE_WIN = "Phone Trajectory"

def _draw_phone_window(traj, raw_x, raw_y):
    """Draw phone IMU trajectory in a standalone OpenCV window."""
    W, H = 400, 400
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (W, H), (20, 20, 30), -1)

    if len(traj) < 2:
        cv2.imshow(_PHONE_WIN, canvas)
        cv2.waitKey(1)
        return

    # Find bounds
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 0.5)
    span_y = max(max_y - min_y, 0.5)
    span = max(span_x, span_y) * 1.2
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

    def to_px(x, y):
        px = int((x - cx) / span * 180 + W // 2)
        py = int((cy - y) / span * 180 + H // 2)
        return px, py

    # Grid
    for i in range(5):
        cv2.line(canvas, (40, 80 + i * 80), (W - 40, 80 + i * 80), (40, 40, 50), 1)
        cv2.line(canvas, (80 + i * 80, 40), (80 + i * 80, H - 40), (40, 40, 50), 1)

    # Trajectory line
    pts = [to_px(p[0], p[1]) for p in traj]
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 200), 2)

    # Current position dot
    if pts:
        cv2.circle(canvas, pts[-1], 6, (0, 255, 255), -1)

    # Header
    cv2.putText(canvas, "PHONE IMU", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
    cv2.putText(canvas, f"Pos: ({raw_x:.2f}, {raw_y:.2f})", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(canvas, f"Pts: {len(traj)}", (W - 70, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow(_PHONE_WIN, canvas)
    cv2.waitKey(1)


# ---- Monkey-patch: 腿关键点交叉干扰纠正 ----
from ubuntu.leg_corrector import LegCorrector
import rehab_monitor.pose_detector as pd_module

_corrector = LegCorrector()
_orig_process_frame = pd_module.PoseDetector.process_frame

def _corrected_process_frame(self, frame):
    annotated, kpts = _orig_process_frame(self, frame)
    # 人体存在检测: 关键点平均置信度 < 0.5 → 假检测 → 丢弃
    if kpts is not None and kpts.shape[0] > 0:
        mean_conf = kpts[0, :, 2].mean()
        if mean_conf < 0.5:
            kpts = None

    # 人体丢失时用手机IMU位置
    if kpts is None:
        from ubuntu.phone_data import phone_data_source as _pds
        if _pds.is_active():
            pp = _pds.get_position()
            if pp is not None:
                import rehab_monitor.spatial_mapper as _spm
                _spatial = getattr(_spm, '_phone_spatial_ref', None)
                if _spatial is not None:
                    if not hasattr(_spatial, '_phone_off'):
                        _spatial._phone_off = (
                            _spatial.current_pos[0] - pp[0],
                            _spatial.current_pos[1] - pp[1])
                        _spatial._phone_traj = []
                        print(f"[手机] 接管 (偏移={_spatial._phone_off[0]:.1f},{_spatial._phone_off[1]:.1f})")
                    wx = pp[0] + _spatial._phone_off[0]
                    wy = pp[1] + _spatial._phone_off[1]
                    _spatial.current_pos = (wx, wy)
                    _spatial.position_history.append((wx, wy))
                    # 追加到手机专属轨迹
                    _spatial._phone_traj.append((wx, wy))
                    if len(_spatial._phone_traj) > 300:
                        _spatial._phone_traj = _spatial._phone_traj[-300:]
                    # 画独立手机轨迹窗口
                    _draw_phone_window(_spatial._phone_traj, pp[0], pp[1])
    elif kpts is not None:
        import rehab_monitor.spatial_mapper as _spm2
        _spatial = getattr(_spm2, '_phone_spatial_ref', None)
        if _spatial is not None and hasattr(_spatial, '_phone_traj'):
            del _spatial._phone_traj
            if hasattr(_spatial, '_phone_off'):
                del _spatial._phone_off
            # 销毁手机轨迹窗口
            try:
                cv2.destroyWindow("Phone Trajectory")
            except Exception:
                pass
            print("[手机] 切回摄像头")

    if kpts is not None and kpts.shape[0] == 1:
        kpts, swapped = _corrector.correct(kpts)
    return annotated, kpts

pd_module.PoseDetector.process_frame = _corrected_process_frame
print(f"[ubuntu] 腿关键点纠正器已启用")

# ---- Monkey-patch: 硬件加速摄像头 ----
if os.environ.get("REHAB_HW_DECODE", "").lower() in ("1", "true", "yes"):
    from ubuntu.hw_camera import monkey_patch_cv2_videocapture, detect_best_backend
    best = detect_best_backend()
    print(f"[ubuntu] 硬件解码: 检测到最优后端 = {best}")
    monkey_patch_cv2_videocapture()
else:
    print("[ubuntu] 硬件解码: 未启用 (设置 REHAB_HW_DECODE=1 启用)")

# ---- Wristband + Fall Popup integration ----
if os.environ.get("REHAB_WRISTBAND", "").lower() in ("1", "true", "yes"):
    import subprocess as _sp
    import threading as _th
    import re as _re

    # 1. Monkey-patch broadcast_fall_alert — add camera-source popup
    from ubuntu.fall_popup import enqueue_fall as _popup_enqueue
    import rehab_monitor.api_server as _api_module
    _orig_broadcast = _api_module.broadcast_fall_alert

    def _patched_broadcast(location, score):
        _orig_broadcast(location, score)
        _popup_enqueue("camera", score,
                       {"location": list(location) if location else [0, 0]})

    _api_module.broadcast_fall_alert = _patched_broadcast
    print("[ubuntu] broadcast_fall_alert 已注入桌面弹窗")

    # 2. Launch original server.py as subprocess (zero modifications)
    _server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_wristband_server.py")
    _wristband_fall = [0.0, 0.0]  # [score, timestamp] — set when wristband detects fall

    _fall_pattern = _re.compile(r"FALL #\d+ \| Peak: ([\d.]+) m/s2 \| Conf: (\d+)%")

    # Clean up leftover processes from a previous run
    _wristband_port = int(os.environ.get("REHAB_WRISTBAND_PORT", "8081"))
    _wristband_http = int(os.environ.get("REHAB_WRISTBAND_HTTP", "8080"))
    for _port in (_wristband_http, _wristband_port):
        try:
            _sp.run(["fuser", "-k", f"{_port}/tcp"],
                    capture_output=True, timeout=3)
        except Exception:
            pass

    # Only print important lines: connections, disconnections, falls, errors
    _important = _re.compile(
        r'(connected|Connected|Disconnected|subscriber|FALL|Error|error|'
        r'Waiting for|mDNS|TCP)')

    def _monitor_wristband(proc):
        """Read stdout from the original server.py and detect fall events."""
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            m = _fall_pattern.search(line)
            if m:
                peak = float(m.group(1))
                conf = int(m.group(2)) / 100.0
                _popup_enqueue("wristband", conf, {"peak_magnitude": peak})
                _wristband_fall[0] = conf
                _wristband_fall[1] = time.time()
                try:
                    import rehab_monitor.api_server as _api
                    _api.broadcast_fall_alert((0, 0), conf)
                except Exception:
                    pass
                print(f"[手环] {line}")
            elif _important.search(line):
                print(f"[手环] {line}")

    _wristband_env = os.environ.copy()
    _wristband_env["PYTHONUNBUFFERED"] = "1"
    _wristband_proc = _sp.Popen(
        [sys.executable, _server_script],
        stdout=_sp.PIPE, stderr=_sp.STDOUT,
        text=True, bufsize=1,
        env=_wristband_env,
    )
    _th.Thread(target=_monitor_wristband, args=(_wristband_proc,),
               name="wristband-monitor", daemon=True).start()
    print(f"[ubuntu] 手环服务器已启动 (原始 server.py, pid={_wristband_proc.pid})")

    print(f"[ubuntu] 手环服务器已启动 (原始 server.py, pid={_wristband_proc.pid})")

# ---- Phone IMU data integration ----
if os.environ.get("REHAB_PHONE", "").lower() in ("1", "true", "yes"):
    from ubuntu.phone_data import (
        phone_data_source, _register_phone_routes, start_phone_listener)

    # 1. Register phone API routes on the Flask app (monkey-patch create_app)
    import rehab_monitor.api_server as _api_module
    _orig_create_app = _api_module.create_app

    def _patched_create_app():
        app = _orig_create_app()
        _register_phone_routes(app)
        return app

    _api_module.create_app = _patched_create_app
    print("[ubuntu] 手机 API 路由已注册 (/api/v1/phone/data, /api/v1/phone/status)")

    # 2. Start UDP discovery listener
    _phone_listener, _phone_stop = start_phone_listener()
    print("[ubuntu] 手机 UDP 发现服务已启动 (端口 5002)")

    # 3. Capture spatial reference for phone takeover
    import rehab_monitor.spatial_mapper as _sp_module
    _sp_module._phone_spatial_ref = None
    _orig_sp_init = _sp_module.SpatialMapper.__init__

    def _patched_spatial_init(self, *a, **kw):
        _orig_sp_init(self, *a, **kw)
        _sp_module._phone_spatial_ref = self
    _sp_module.SpatialMapper.__init__ = _patched_spatial_init

    print("[ubuntu] 手机接管: 人体丢失→手机IMU, 人体恢复→摄像头")

    print("[ubuntu] SpatialMapper 已注入手机轨迹同步")

# ---- CSI fall detection integration ----
if os.environ.get("REHAB_CSI", "").lower() in ("1", "true", "yes"):
    from csi_fall.monitor import CSIMonitor as _CSIMonitor
    from csi_fall.config import (
        CSI_HOST as _CFG_CSI_HOST, CSI_PORT as _CFG_CSI_PORT,
        CSI_WIFI_PROTOCOL as _CFG_CSI_PROTO,
    )

    # 1. 环境变量覆盖 ESP32 网络配置
    _csi_host = os.environ.get("CSI_HOST", _CFG_CSI_HOST)
    _csi_port = int(os.environ.get("CSI_PORT", str(_CFG_CSI_PORT)))
    _csi_proto = os.environ.get("CSI_WIFI_PROTOCOL", _CFG_CSI_PROTO)
    _csi_diag = os.environ.get("CSI_SHOW_DIAG", "").lower() in ("1", "true", "yes")

    # 2. 启动 CSI 监视器 (daemon 线程)
    _csi_monitor = _CSIMonitor(
        host=_csi_host, port=_csi_port,
        wifi_protocol=_csi_proto, show_diag=_csi_diag,
    )
    _csi_thread = _csi_monitor.start()
    if _csi_thread:
        print(f"[ubuntu] CSI 跌倒检测已启动 ({_csi_host}:{_csi_port}, protocol={_csi_proto})")
    else:
        print("[ubuntu] CSI 跌倒检测启动失败 (检查 ESP32 是否在线)")

# ---- Unified FallDetector patch (checks phone + wristband + CSI) ----
_need_fall_patch = (
    os.environ.get("REHAB_PHONE", "") == "1"
    or os.environ.get("REHAB_WRISTBAND", "") == "1"
    or os.environ.get("REHAB_CSI", "").lower() in ("1", "true", "yes")
)
if _need_fall_patch:
    import time as _fall_time
    import rehab_monitor.fall_detector as _fd_module
    _orig_fall_update = _fd_module.FallDetector.update

    def _patched_fall_update(self, person_kpts, bbox, frame_h, depth_m):
        # Check phone (replaces camera detection)
        try:
            from ubuntu.phone_data import phone_data_source
            if phone_data_source.is_active():
                ps = phone_data_source.get_fall_status()
                if ps in ("ALERTING", "FALLEN"):
                    self.status = "alert"
                    self.score = phone_data_source.get_fall_score()
                    return ("alert", self.score)
        except Exception:
            pass
        # Check wristband (replaces camera detection)
        try:
            if _fall_time.time() - _wristband_fall[1] < 3.0:
                self.status = "alert"
                self.score = _wristband_fall[0]
                return ("alert", self.score)
        except NameError:
            pass

        # Run original camera-based fall detection first
        cam_status, cam_score = _orig_fall_update(
            self, person_kpts, bbox, frame_h, depth_m)

        # If phone or wristband is active (but didn't trigger above),
        # camera fall detection is disabled — return safe
        try:
            _phone_active = False
            try:
                from ubuntu.phone_data import phone_data_source
                _phone_active = phone_data_source.is_active()
            except Exception:
                pass
            _wristband_active = False
            try:
                _wristband_active = True  # wristband was enabled
            except NameError:
                pass
            if _phone_active or _wristband_active:
                return ("safe", 0.0)
        except Exception:
            pass

        # CSI fall detection — runs in PARALLEL with camera (not replaces)
        # CSI and YOLO detect falls using different physical principles
        # (RF multipath vs. visual pose), so they are complementary
        try:
            if _csi_monitor.is_fallen:
                csi_st = _csi_monitor.status
                csi_score = csi_st.get('score', 0.0)
                if csi_score > cam_score:
                    self.status = "alert"
                    self.score = csi_score
                    return ("alert", self.score)
        except NameError:
            pass

        return (cam_status, cam_score)

    _fd_module.FallDetector.update = _patched_fall_update
    print("[ubuntu] FallDetector 已注入 (手机+手环: 替换, CSI: 并联, 摄像头: 回退)")

import rehab_monitor.main
rehab_monitor.main.main()
