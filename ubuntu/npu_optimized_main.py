"""Ubuntu wrapper for the heterogeneous (optimized) pipeline.

Adds fall-alert desktop popup + wristband-server + phone-IMU support
on top of rehab_optimized.main.  All integration is done via
monkey-patch — zero modifications to rehab_optimized/.

Usage (indirect, via run.sh):
    bash ubuntu/run.sh --optimized --wristband
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Always-active: broadcast_fall_alert → desktop popup
# (moved out of REHAB_WRISTBAND block so camera-detected falls always pop up)
# ---------------------------------------------------------------------------
from ubuntu.fall_popup import enqueue_fall as _popup_enqueue
import rehab_monitor.api_server as _api_module
_orig_broadcast = _api_module.broadcast_fall_alert


def _patched_broadcast(location, score):
    _orig_broadcast(location, score)
    _popup_enqueue("camera", score,
                   {"location": list(location) if location else [0, 0]})


_api_module.broadcast_fall_alert = _patched_broadcast
print("[ubuntu] broadcast_fall_alert 已注入桌面弹窗")

# ---------------------------------------------------------------------------
# Wristband + Fall Popup integration  (REHAB_WRISTBAND=1)
# ---------------------------------------------------------------------------
if os.environ.get("REHAB_WRISTBAND", "").lower() in ("1", "true", "yes"):
    import subprocess as _sp
    import threading as _th
    import re as _re

    # 1. Launch original server.py as subprocess (zero modifications)
    _server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_wristband_server.py")
    _fall_pattern = _re.compile(r"FALL #\d+ \| Peak: ([\d.]+) m/s2 \| Conf: (\d+)%")
    _http_fall_pattern = _re.compile(r"FALL \(HTTP\)! mag=([\d.]+)")

    # Clean up leftover processes from a previous run
    for _port in (8080, 8081):
        try:
            _sp.run(["fuser", "-k", f"{_port}/tcp"],
                    capture_output=True, timeout=3)
        except Exception:
            pass

    _important = _re.compile(
        r'(connected|Connected|Disconnected|subscriber|FALL|Error|error|'
        r'Waiting for|mDNS|TCP)')

    def _monitor_wristband(proc):
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            m = _fall_pattern.search(line)
            hm = _http_fall_pattern.search(line)
            if m:
                peak = float(m.group(1))
                conf = int(m.group(2)) / 100.0
                _popup_enqueue("wristband", conf, {"peak_magnitude": peak})
                # 同时推送到 WebSocket
                try:
                    import rehab_monitor.api_server as _api
                    _api.broadcast_fall_alert((0, 0), conf)
                except Exception:
                    pass
                print(f"[手环] {line}")
            elif hm:
                conf = float(hm.group(1)) / 10.0  # mag 10.0 → score 1.0
                conf = min(conf, 1.0)
                _popup_enqueue("wristband", conf, {"magnitude": float(hm.group(1))})
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

# ---- Phone IMU data integration ----
if os.environ.get("REHAB_PHONE", "").lower() in ("1", "true", "yes"):
    from ubuntu.phone_data import (
        phone_data_source, _register_phone_routes, start_phone_listener)

    # 1. Register phone API routes
    import rehab_monitor.api_server as _api_module
    _orig_create_app = _api_module.create_app

    def _patched_create_app():
        app = _orig_create_app()
        _register_phone_routes(app)
        return app

    _api_module.create_app = _patched_create_app
    print("[ubuntu] 手机 API 路由已注册")

    # 2. Start UDP discovery
    _phone_listener, _phone_stop = start_phone_listener()
    print("[ubuntu] 手机 UDP 发现服务已启动 (端口 5002)")

    # 3. Monkey-patch for phone takeover during camera occlusion
    import rehab_monitor.face_locker as _fl_module
    _orig_mc = _fl_module.FaceNetLocker.match_combined
    _face_lost = [False]

    def _patched_match_combined(self, *a, **kw):
        r = _orig_mc(self, *a, **kw)
        src = r[2] if len(r) > 2 else None
        _face_lost[0] = (not r[0] or src == 'body')
        return r
    _fl_module.FaceNetLocker.match_combined = _patched_match_combined

    import rehab_monitor.spatial_mapper as _sp_module
    _orig_get_pos = _sp_module.SpatialMapper.get_person_position

    def _patched_get_position(self, person_kpts):
        pos = _orig_get_pos(self, person_kpts)

        need_phone = (pos is None or _face_lost[0]) and phone_data_source.is_active()

        if need_phone:
            pp = phone_data_source.get_position()
            if pp is not None:
                if not hasattr(self, '_phone_off'):
                    ref = pos if pos is not None else self.current_pos
                    self._phone_off = (ref[0] - pp[0], ref[1] - pp[1])
                    print(f"[手机] 接管 (偏移={self._phone_off[0]:.1f},{self._phone_off[1]:.1f})")
                pos = (pp[0] + self._phone_off[0], pp[1] + self._phone_off[1])
                self.current_pos = pos
                self.position_history.append(pos)
        elif hasattr(self, '_phone_off'):
            del self._phone_off
            print("[手机] 切回摄像头")

        return pos

    _sp_module.SpatialMapper.get_person_position = _patched_get_position
    print("[ubuntu] 手机接管就绪 (人脸丢失→手机, 人脸恢复→摄像头)")

    # 4. Monkey-patch FallDetector for phone fall alerts
    import rehab_monitor.fall_detector as _fd_module
    _orig_fall_update = _fd_module.FallDetector.update

    def _patched_fall_update(self, person_kpts, bbox, frame_h, depth_m):
        if phone_data_source.is_active():
            phone_status = phone_data_source.get_fall_status()
            if phone_status in ("ALERTING", "FALLEN"):
                self.status = "alert"
                self.score = phone_data_source.get_fall_score()
                return ("alert", self.score)
        return _orig_fall_update(self, person_kpts, bbox, frame_h, depth_m)

    _fd_module.FallDetector.update = _patched_fall_update
    print("[ubuntu] FallDetector 已注入手机摔倒接管")

# ---------------------------------------------------------------------------
# Delegate to the real optimized main
# ---------------------------------------------------------------------------
import rehab_optimized.main
rehab_optimized.main.main()
