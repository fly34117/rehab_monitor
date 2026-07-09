"""ESP32-S3 wristband fall-detection server — embeddable in rehab pipeline.

Listens on a TCP port for JSON-lines sensor/fall data from an ESP32-S3
wristband, serves a web dashboard over HTTP, and bridges fall events into
the unified fall_popup module for on-screen alerting.

Adapted from 桌面/esp32s3/server.py — wrapped as a class so it can run as a
background daemon inside the rehab pipeline.
"""

import json
import logging
import math
import os
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from zeroconf import ServiceInfo, Zeroconf
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False

# Try real tracker3d first; fall back to our stub.
try:
    from tracker3d import IMUDisplacementTracker
except ImportError:
    from .wristband_tracker_stub import IMUDisplacementTracker

log = logging.getLogger("wristband")

# ---------------------------------------------------------------------------
# Embedded HTML dashboard  (identical to original; self-contained)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Fall Detection</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#0f0f23;color:#e0e0e0}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;text-align:center;border-bottom:2px solid #0ff}
.header h1{color:#0ff;font-size:24px}
.status{display:flex;justify-content:center;gap:20px;margin-top:10px}
.badge{padding:6px 16px;border-radius:20px;font-size:13px}
.ok{background:#0a3d0a;border:1px solid #0f0;color:#0f0}
.alert{background:#3d0a0a;border:1px solid #f00;color:#f00}
.container{max-width:900px;margin:20px auto;padding:0 20px}
.card{background:#1a1a2e;border-radius:10px;padding:20px;margin-bottom:20px;border:1px solid #333}
.card h2{color:#0ff;margin-bottom:15px;font-size:18px}
.sensor{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:15px}
.s{text-align:center;padding:10px;background:#16213e;border-radius:8px}
.s .l{font-size:11px;color:#666}.s .v{font-size:20px;font-weight:bold}
.vx{color:#ff6b6b}.vy{color:#51cf66}.vz{color:#339af0}.vm{color:#ff0}
.list{max-height:400px;overflow-y:auto}
.item{background:#16213e;padding:15px;border-radius:8px;margin-bottom:10px;border-left:4px solid #f00}
.time{color:#888;font-size:12px}
.empty{text-align:center;color:#666;padding:40px}
.cnt{background:#0ff;color:#000;padding:2px 10px;border-radius:10px;font-size:14px}
</style></head><body>
<div class="header"><h1>Fall Detection Monitor</h1>
<div class="status"><div class="badge ok">Server Running</div>
<div class="badge ok">Events: <span class="cnt" id="n">0</span></div></div></div>
<div class="container">
<div class="card"><h2>Real-time Sensor Data</h2>
<div class="sensor">
<div class="s"><div class="l">Accel X</div><div class="v vx" id="ax">0.00</div></div>
<div class="s"><div class="l">Accel Y</div><div class="v vy" id="ay">0.00</div></div>
<div class="s"><div class="l">Accel Z</div><div class="v vz" id="az">0.00</div></div>
<div class="s"><div class="l">Magnitude</div><div class="v vm" id="mag">0.00</div></div>
<div class="s"><div class="l">Gyro Mag</div><div class="v" id="gm" style="color:#ff8800">0</div></div>
<div class="s"><div class="l">State</div><div class="v" id="st" style="color:#20c997">Normal</div></div>
</div></div>
<div class="card"><h2>Position Tracking (ZUPT + Drift Comp) <button onclick="fetch('/api/origin')" style="float:right;background:#0ff;border:none;color:#000;padding:4px 12px;border-radius:4px;cursor:pointer;font-weight:bold">Set Origin</button></h2>
<div class="sensor">
<div class="s"><div class="l">X (m)</div><div class="v vx" id="px">0.0000</div></div>
<div class="s"><div class="l">Y (m)</div><div class="v vy" id="py">0.0000</div></div>
<div class="s"><div class="l">Z (m)</div><div class="v vz" id="pz">0.0000</div></div>
<div class="s"><div class="l">State</div><div class="v" id="pstate" style="color:#20c997">INIT</div></div>
<div class="s"><div class="l">S-Val</div><div class="v" id="psval" style="color:#ff8800">0.00</div></div>
<div class="s"><div class="l">Frames</div><div class="v" id="pframes" style="color:#888">0</div></div>
</div></div>
<div class="card"><h2>Fall Events</h2>
<div class="list" id="lst"><div class="empty">Waiting for fall events...</div></div></div></div>
<script>
async function load(){
try{
const r=await fetch('/api/fall');const d=await r.json();
document.getElementById('n').textContent=d.length;
const el=document.getElementById('lst');
if(!d.length){el.innerHTML='<div class="empty">Waiting...</div>';return}
el.innerHTML=d.reverse().map(e=>`<div class="item"><div class="time">${e.received_at||''}</div>
<div style="margin-top:8px">Peak: ${e.peak_magnitude?.toFixed(1)||0} m/s2 | Conf: ${(e.confidence*100)?.toFixed(0)||0}%</div></div>`).join('');
}catch(e){}
try{
const r=await fetch('/api/sensor');const s=await r.json();
document.getElementById('ax').textContent=(s.ax||0).toFixed(2);
document.getElementById('ay').textContent=(s.ay||0).toFixed(2);
document.getElementById('az').textContent=(s.az||0).toFixed(2);
document.getElementById('mag').textContent=(s.mag||0).toFixed(2);
document.getElementById('gm').textContent=(s.gyro_mag||0).toFixed(0);
const st=document.getElementById('st');
st.textContent=s.state||'Normal';
st.style.color=s.state=='DETECTED'?'#f00':'#20c997';
}catch(e){}
try{
const r=await fetch('/api/position');const p=await r.json();
document.getElementById('px').textContent=(p.px||0).toFixed(4);
document.getElementById('py').textContent=(p.py||0).toFixed(4);
document.getElementById('pz').textContent=(p.pz||0).toFixed(4);
const ps=document.getElementById('pstate');
ps.textContent=p.stationary?'STATIONARY':'MOVING';
ps.style.color=p.stationary?'#20c997':'#ff8800';
document.getElementById('psval').textContent=(p.stationary_val||0).toFixed(3);
document.getElementById('pframes').textContent=p.frames||0;
const conf=p.confidence||0;
const px=document.getElementById('px');
if(conf>0.6){px.style.color='#51cf66'}else if(conf>0.3){px.style.color='#ff0'}else{px.style.color='#f00'}
document.getElementById('py').style.color=px.style.color;
}catch(e){}
}load();setInterval(load,500);
</script></body></html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _now():
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _HTTPHandler(BaseHTTPRequestHandler):
    """Serves the dashboard + REST API.  Uses closure over the server instance
    stored on the HTTPServer object."""

    @property
    def server_state(self):
        # The WristbandServer instance is stashed on the HTTPServer.
        return self.server._wristband

    def do_GET(self):
        state = self.server_state
        if self.path == '/api/fall':
            body = json.dumps(state.fall_events, ensure_ascii=False).encode()
            self._respond_json(body)
        elif self.path == '/api/sensor':
            body = json.dumps(state.sensor_data).encode()
            self._respond_json(body)
        elif self.path == '/api/position':
            body = json.dumps(state.pos_data).encode()
            self._respond_json(body)
        elif self.path == '/api/origin':
            state.tracker.set_origin()
            state.pos_data = state.tracker.get_state()
            self._respond_json(b'{"origin":"set","ok":true}')
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        if self.path == '/api/fall':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                data['received_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.server_state.fall_events.append(data)
                mag = data.get('peak_magnitude', 0)
                print(f"[{data['received_at']}] FALL (HTTP)! mag={mag:.1f}")
                # 直接推送 WebSocket + 桌面弹窗
                try:
                    from rehab_monitor.api_server import broadcast_fall_alert
                    conf = min(mag / 10.0, 1.0)
                    broadcast_fall_alert((0, 0), conf)
                except Exception:
                    pass
                try:
                    from .fall_popup import enqueue_fall
                    enqueue_fall("wristband", conf, {"magnitude": mag})
                except Exception:
                    pass
            except Exception as e:
                print(f"[手环] HTTP fall 解析错误: {e}")
            self._respond_json(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, *args):
        pass

    def _respond_json(self, body_bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body_bytes)))
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body_bytes)


# ---------------------------------------------------------------------------
# Terminal viz  (disabled by default; opt-in for standalone use)
# ---------------------------------------------------------------------------

def _terminal_viz_loop(state):
    """ANSI terminal dashboard — reads from *state* (a WristbandServer)."""
    last = 0
    while state._running:
        time.sleep(0.1)
        if state.sensor_count == last:
            continue
        last = state.sensor_count

        s = state.sensor_data
        p = state.pos_data
        is_fall = s.get('state', 'Normal') == 'DETECTED'
        c = '\033[1;31m' if is_fall else '\033[0m'

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"  Samples: {state.sensor_count}  |  Falls: {state.fall_count}  |  {c}State: {s.get('state','Normal')}\033[0m")
        print()
        print(f"  Accel X: {s.get('ax',0):8.2f} m/s2     Gyro X: {s.get('gx',0):8.1f} dps")
        print(f"  Accel Y: {s.get('ay',0):8.2f} m/s2     Gyro Y: {s.get('gy',0):8.1f} dps")
        print(f"  Accel Z: {s.get('az',0):8.2f} m/s2     Gyro Z: {s.get('gz',0):8.1f} dps")
        print(f"  Mag:     {s.get('mag',0):8.2f} m/s2     Gyro Mag: {s.get('gyro_mag',0):8.0f} dps")
        print(f"  Confidence: {s.get('confidence',0):6.0%}")
        print()

        conf = p.get('confidence', 0)
        if conf > 0.6:
            pc = '\033[1;32m'
        elif conf > 0.3:
            pc = '\033[1;33m'
        else:
            pc = '\033[1;31m'
        st = 'STATIONARY' if p.get('stationary') else 'MOVING'
        stc = '\033[1;32m' if p.get('stationary') else '\033[1;33m'
        drift = math.hypot(p.get('drift_vx', 0), p.get('drift_vy', 0))
        print(f"  {pc}Position (m):  X={p.get('px',0):8.4f}  Y={p.get('py',0):8.4f}  Z={p.get('pz',0):8.4f}\033[0m"
              f"  conf={conf:.0%}")
        print(f"  Velocity (m/s): X={p.get('vx',0):8.4f}  Y={p.get('vy',0):8.4f}  Z={p.get('vz',0):8.4f}")
        print(f"  {stc}{st}\033[0m  |  Drift rate: {drift:.5f} m/s/s  |  S-val: {p.get('stationary_val',0):.3f}")
        print(f"  Frames: {p.get('frames',0)}  |  Roll={p.get('roll',0):6.1f}  Pitch={p.get('pitch',0):6.1f}  Yaw={p.get('yaw',0):6.1f}")
        print()
        if is_fall:
            print(f"  \033[1;31m╔════════════════════════════╗")
            print(f"  ║   FALL ALERT ACTIVE      ║")
            print(f"  ╚════════════════════════════╝\033[0m")
        print()
        print(f"  [Web dashboard: http://{_get_local_ip()}:{state.http_port}]")


# ---------------------------------------------------------------------------
# WristbandServer
# ---------------------------------------------------------------------------

class WristbandServer:
    """TCP + HTTP server for ESP32-S3 wristband fall-detection data.

    Usage::

        ws = WristbandServer(http_port=8080, tcp_port=8081)
        ws.start()          # non-blocking — spawns daemon threads
        ...
        ws.stop()           # clean shutdown
    """

    def __init__(self, http_port=8080, tcp_port=8081,
                 enable_viz=False, verbose=True):
        self.http_port = http_port
        self.tcp_port = tcp_port
        self.enable_viz = enable_viz
        self.verbose = verbose

        # Internal state  (was module-level globals in original server.py)
        self.fall_events = []
        self.sensor_data = {
            "ax": 0, "ay": 0, "az": 0,
            "gx": 0, "gy": 0, "gz": 0,
            "mag": 0, "gyro_mag": 0,
            "state": "Normal", "confidence": 0,
        }
        self.sensor_count = 0
        self.fall_count = 0
        self.tracker = IMUDisplacementTracker(fs=333.0)
        self.tracker.set_origin()
        self.pos_data = self.tracker.get_state()

        # Threading
        self._running = False
        self._sock = None
        self._httpd = None
        self._zc = None
        self._zc_info = None
        self._subscribers = []
        self._sub_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Launch HTTP + TCP listeners in daemon threads.  Non-blocking."""
        if self._running:
            return

        self._running = True
        local_ip = _get_local_ip()

        if self.verbose:
            print(f"  HTTP dashboard: http://{local_ip}:{self.http_port}")
            print(f"  TCP stream:     {local_ip}:{self.tcp_port}")

        # mDNS
        self._register_mdns()

        # HTTP server thread
        self._httpd = HTTPServer(('0.0.0.0', self.http_port), _HTTPHandler)
        self._httpd._wristband = self        # stash ref for handler
        threading.Thread(target=self._httpd.serve_forever,
                         name="wristband-http", daemon=True).start()

        # Terminal viz thread (opt-in)
        if self.enable_viz:
            threading.Thread(target=_terminal_viz_loop, args=(self,),
                             name="wristband-viz", daemon=True).start()

        # TCP listener thread
        threading.Thread(target=self._tcp_accept_loop,
                         name="wristband-tcp", daemon=True).start()

    def stop(self):
        """Shut down all listeners and threads."""
        self._running = False

        # Close TCP socket
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        # Shut down HTTP server
        if self._httpd:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            self._httpd = None

        # Unregister mDNS
        if self._zc and self._zc_info:
            try:
                self._zc.unregister_service(self._zc_info)
                self._zc.close()
            except Exception:
                pass
            self._zc = None
            self._zc_info = None

        print("[手环] 服务器已停止")

    # ------------------------------------------------------------------
    # mDNS
    # ------------------------------------------------------------------

    def _register_mdns(self):
        if not HAS_ZEROCONF:
            if self.verbose:
                print("  [!] zeroconf not installed, mDNS disabled")
                print("  [!] Run: pip install zeroconf")
            return
        try:
            local_ip = _get_local_ip()
            self._zc = Zeroconf()
            self._zc_info = ServiceInfo(
                "_fall-detector._tcp.local.",
                "Fall Detector._fall-detector._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=self.tcp_port,
                properties={},
                server="fall-detector.local.",
            )
            self._zc.register_service(self._zc_info)
            if self.verbose:
                print(f"  mDNS: fall-detector.local -> {local_ip}")
        except Exception as e:
            print(f"  [!] mDNS failed: {e}")

    # ------------------------------------------------------------------
    # TCP accept loop
    # ------------------------------------------------------------------

    def _tcp_accept_loop(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(('0.0.0.0', self.tcp_port))
        except OSError as e:
            print(f"[手环] 无法绑定 TCP:{self.tcp_port} — {e}")
            return
        self._sock.listen(5)
        self._sock.settimeout(1.0)

        print(f"[手环] TCP 监听已就绪 :{self.tcp_port}")
        print("[手环] 等待 ESP32 + viz 客户端连接...")
        print()

        while self._running:
            try:
                conn, addr = self._sock.accept()
                threading.Thread(target=self._client_handler,
                                 args=(conn, addr),
                                 name=f"wristband-client-{addr[1]}",
                                 daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    # ------------------------------------------------------------------
    # Per-client handler  (TCP, JSON-lines)
    # ------------------------------------------------------------------

    def _client_handler(self, conn, addr):
        conn.settimeout(5.0)
        buf = ""
        is_sub = False
        label = 'Viz' if addr[1] != self.tcp_port else 'ESP32'
        print(f"[{_now()}] {label} 已连接: {addr[0]}:{addr[1]}")

        while self._running:
            try:
                data = conn.recv(4096)
                if not data:
                    print(f"[{_now()}] 断开连接: {addr[0]}:{addr[1]}")
                    break
                buf += data.decode('utf-8', errors='ignore')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get('type')

                    if msg_type == 'subscribe':
                        is_sub = True
                        with self._sub_lock:
                            self._subscribers.append(conn)
                        print(f"[{_now()}] Viz 订阅者已添加 (总计: {len(self._subscribers)})")
                        conn.sendall(b'{"type":"ok"}\n')

                    elif msg_type == 'sensor':
                        self.sensor_count += 1
                        self.sensor_data = msg
                        dt = 1.0 / 333.0
                        self.tracker.update(
                            msg.get('ax', 0), msg.get('ay', 0), msg.get('az', 0),
                            msg.get('gx', 0), msg.get('gy', 0), msg.get('gz', 0),
                            dt,
                        )
                        self.pos_data = self.tracker.get_state()
                        if self.sensor_count % 30 == 0:
                            print(f"[{_now()}] #{self.sensor_count} "
                                  f"Accel=({msg.get('ax',0):6.2f},{msg.get('ay',0):6.2f},{msg.get('az',0):6.2f}) "
                                  f"State={msg.get('state','Normal')} Conf={msg.get('confidence',0):.0%}")
                        self._broadcast(line + '\n')

                    elif msg_type == 'fall':
                        self.fall_count += 1
                        msg['received_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.fall_events.append(msg)
                        peak = msg.get('peak_magnitude', 0)
                        conf = msg.get('confidence', 0)
                        print(f"\n{'='*55}")
                        print(f"  FALL #{self.fall_count} | Peak: {peak:.1f} m/s2 | Conf: {conf:.0%}")
                        print(f"{'='*55}\n")
                        self._broadcast(line + '\n')

                        # Bridge: WebSocket → 微信小程序 + 桌面弹窗 + 数据库
                        try:
                            from rehab_monitor.api_server import broadcast_fall_alert
                            broadcast_fall_alert((0, 0), conf)
                            print("[手环] WebSocket 告警已推送")
                        except Exception as exc:
                            print(f"[手环] WebSocket 推送失败: {exc}")

                        try:
                            from .fall_popup import enqueue_fall
                            enqueue_fall("wristband", conf,
                                         {"peak_magnitude": peak})
                            print("[手环] 跌倒弹窗已触发")
                        except Exception as exc:
                            print(f"[手环] 弹窗触发失败: {exc}")

                        try:
                            import rehab_monitor.api_server as _api_mod
                            db = getattr(_api_mod, 'database', None)
                            if db is not None and hasattr(db, 'write_fall_alert'):
                                db.write_fall_alert(self.sensor_count)
                        except Exception:
                            pass

            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError):
                break
            except Exception as e:
                print(f"[手环] TCP 错误: {e}")
                break

        if is_sub:
            with self._sub_lock:
                if conn in self._subscribers:
                    self._subscribers.remove(conn)
        try:
            conn.close()
        except Exception:
            pass

    def _broadcast(self, line):
        """Send *line* to every registered viz subscriber."""
        with self._sub_lock:
            dead = []
            for c in self._subscribers:
                try:
                    c.sendall(line.encode())
                except Exception:
                    dead.append(c)
            for d in dead:
                self._subscribers.remove(d)


# ---------------------------------------------------------------------------
# Standalone entry point  (for testing outside the rehab pipeline)
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    ap = argparse.ArgumentParser(description="ESP32-S3 Wristband Fall Detection Server")
    ap.add_argument('--http-port', type=int, default=8080)
    ap.add_argument('--tcp-port', type=int, default=8081)
    ap.add_argument('--viz', action='store_true', help='Enable terminal dashboard')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    server = WristbandServer(
        http_port=args.http_port,
        tcp_port=args.tcp_port,
        enable_viz=args.viz,
        verbose=not args.quiet,
    )
    server.start()

    print("\nPress Ctrl+C to stop...\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()


if __name__ == '__main__':
    main()
