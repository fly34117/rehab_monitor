"""Phone IMU data source — receives trajectory + fall events from phone app.

Architecture (from 手机数据接入Ubuntu指南):
  Phone App → UDP discover (port 5002) → Ubuntu responds with IP
  Phone App → HTTP POST /api/v1/phone/data → Flask → PhoneDataSource
  Main loop → PhoneDataSource.get_position() → takeover when camera fails

The UDP discovery listener runs as a daemon thread.  Data is received
via Flask routes registered by monkey-patching create_app().
"""

import json
import logging
import socket
import threading
import time

log = logging.getLogger("phone_data")


# ---------------------------------------------------------------------------
# PhoneDataSource — thread-safe cache of the latest phone data
# ---------------------------------------------------------------------------

class PhoneDataSource:
    """Thread-safe container for the latest phone IMU trajectory and fall state."""

    def __init__(self):
        self._lock = threading.RLock()  # reentrant — get_status_dict nests calls
        self._last_trajectory = None   # dict: {x, y, heading, stepCount, distance, timestamp}
        self._last_fall = None         # dict: {status, currentAccelMag, alertCountdown, ...}
        self._last_update_time = 0.0
        self._traj_count = 0           # debug counter

    # -- write side (called from Flask route) --

    def update_trajectory(self, data):
        with self._lock:
            self._last_trajectory = data
            self._last_update_time = time.time()
            self._traj_count += 1

    def update_fall(self, data):
        with self._lock:
            self._last_fall = data
            self._last_update_time = time.time()

    # -- read side (called from main loop) --

    def get_position(self):
        """Return (x, y) tuple or None."""
        with self._lock:
            if self._last_trajectory:
                return (self._last_trajectory.get("x", 0.0),
                        self._last_trajectory.get("y", 0.0))
            return None

    def get_heading(self):
        with self._lock:
            return (self._last_trajectory.get("heading", 0.0)
                    if self._last_trajectory else 0.0)

    def get_fall_status(self):
        """Return fall state string: NORMAL / FREE_FALL / IMPACT / ... / ALERTING."""
        with self._lock:
            if self._last_fall:
                return self._last_fall.get("status", "NORMAL")
            return "NORMAL"

    def get_fall_score(self):
        """Convert phone impact acceleration to a 0–1 score."""
        with self._lock:
            if self._last_fall:
                impact = self._last_fall.get("impactAccel") or \
                         self._last_fall.get("currentAccelMag", 0)
                return min(float(impact) / 40.0, 1.0)
            return 0.0

    def is_active(self, timeout=3.0):
        """True if phone has pushed data within *timeout* seconds."""
        with self._lock:
            return (time.time() - self._last_update_time) < timeout

    def get_status_dict(self):
        """Full debug snapshot (reads internal state directly to avoid nested locking)."""
        with self._lock:
            now = time.time()
            active = (now - self._last_update_time) < 3.0
            pos = None
            if self._last_trajectory:
                pos = (self._last_trajectory.get("x", 0.0),
                       self._last_trajectory.get("y", 0.0))
            heading = (self._last_trajectory.get("heading", 0.0)
                       if self._last_trajectory else 0.0)
            fall_status = (self._last_fall.get("status", "NORMAL")
                           if self._last_fall else "NORMAL")
            if self._last_fall:
                impact = (self._last_fall.get("impactAccel") or
                          self._last_fall.get("currentAccelMag", 0))
                fall_score = min(float(impact) / 40.0, 1.0)
            else:
                fall_score = 0.0
            return {
                "active": active,
                "position": pos,
                "heading": heading,
                "fall_status": fall_status,
                "fall_score": fall_score,
                "trajectory_data": (dict(self._last_trajectory)
                                    if self._last_trajectory else None),
                "fall_data": (dict(self._last_fall)
                              if self._last_fall else None),
            }


# Global singleton — shared between Flask route and main-loop patches
phone_data_source = PhoneDataSource()

# Set by FaceNetLocker monkey-patch — True when face tracking is lost
face_tracking_lost = False


# ---------------------------------------------------------------------------
# Phone API — Flask route handlers
# These are registered on the rehab API app via monkey-patch of create_app().
# ---------------------------------------------------------------------------

def _register_phone_routes(app):
    """Add /api/v1/phone/* routes to the existing Flask app."""

    # Suppress Flask's per-request log for phone data (arrives every 200ms)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.route("/api/v1/phone/data", methods=["POST"])
    def phone_data():
        """Receive trajectory or fall data from the phone app."""
        from flask import request, jsonify
        try:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"code": -1, "message": "empty body"}), 400

            msg_type = data.get("type")
            if msg_type == "trajectory":
                phone_data_source.update_trajectory(data)
                if phone_data_source._traj_count % 50 == 1:
                    print(f"[手机] 轨迹 #{phone_data_source._traj_count} pos=({data.get('x',0):.2f},{data.get('y',0):.2f})")
                return jsonify({"code": 0, "message": "ok"})
            elif msg_type == "fall":
                phone_data_source.update_fall(data)
                # Fall events carry x/y position — also update trajectory
                if "x" in data and "y" in data:
                    phone_data_source.update_trajectory(data)
                status = data.get("status", "")
                # 手机确认的摔倒状态: ALERTING/FALLEN/IMPACT
                if status in ("ALERTING", "FALLEN", "IMPACT"):
                    pos = phone_data_source.get_position() or (0, 0)
                    score = phone_data_source.get_fall_score()
                    print(f"[手机] 收到摔倒: status={status} score={score:.2f} pos={pos}")
                    # WebSocket → 微信小程序
                    try:
                        from rehab_monitor.api_server import broadcast_fall_alert
                        broadcast_fall_alert(pos, score)
                    except Exception:
                        pass
                    # OpenCV 弹窗
                    try:
                        from ubuntu.fall_popup import enqueue_fall
                        enqueue_fall("phone", score,
                                     {"location": list(pos) if pos else [0, 0]})
                        print("[手机] 弹窗已触发")
                    except Exception as _exc:
                        print(f"[手机] 弹窗触发失败: {_exc}")
                return jsonify({"code": 0, "message": "ok"})
            else:
                return jsonify({"code": -1,
                                "message": f"unknown type: {msg_type}"}), 400
        except Exception as exc:
            log.exception("phone_data route error")
            return jsonify({"code": -1, "message": str(exc)}), 500

    @app.route("/api/v1/phone/status")
    def phone_status():
        """Debug endpoint — current phone data source state."""
        from flask import jsonify
        return jsonify({"code": 0, "data": phone_data_source.get_status_dict()})

    print("[手机] API 路由已注册: /api/v1/phone/data, /api/v1/phone/status")


# ---------------------------------------------------------------------------
# UDP discovery listener  (phone app sends INDOOR_TRACKER_DISCOVER → port 5002)
# ---------------------------------------------------------------------------

DISCOVERY_PORT = 5002
DISCOVERY_MSG = b"INDOOR_TRACKER_DISCOVER"
RESPONSE_PREFIX = "INDOOR_TRACKER_HERE"


def _get_local_ip():
    """Return the best-guess LAN IP address, preferring real WiFi/LAN interfaces."""
    import subprocess
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True).strip()
        ips = out.split()
        for ip in ips:
            if ip.startswith("192.168."):
                return ip
        for ip in ips:
            if ip.startswith("10."):
                return ip
        if ips:
            return ips[0]
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127.") and not ip.startswith("198.18."):
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def _udp_discovery_loop(stop_event):
    """Run UDP discovery responder in a daemon thread."""
    local_ip = _get_local_ip()
    print(f"[手机] 本机IP: {local_ip}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        sock.bind(("0.0.0.0", DISCOVERY_PORT))
    except OSError as e:
        print(f"[手机] UDP 绑定失败 (端口 {DISCOVERY_PORT}): {e}")
        return

    sock.settimeout(1.0)
    print(f"[手机] UDP 发现服务已启动 (端口 {DISCOVERY_PORT})，等待手机广播...")

    _discovery_count = 0
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(1024)
            if data.strip() == DISCOVERY_MSG:
                response = f"{RESPONSE_PREFIX}:{local_ip}".encode()
                sock.sendto(response, addr)
                _discovery_count += 1
                if _discovery_count % 10 == 1:   # print every ~30s
                    print(f"[手机] 发现响应 #{_discovery_count} → {addr[0]}")
        except socket.timeout:
            continue
        except Exception:
            pass

    sock.close()
    print("[手机] UDP 发现服务已停止")


def start_phone_listener():
    """Launch the UDP discovery thread.  Returns (thread, stop_event)."""
    stop_event = threading.Event()
    t = threading.Thread(target=_udp_discovery_loop, args=(stop_event,),
                         name="phone-udp", daemon=True)
    t.start()
    return t, stop_event
