import socket, json, threading, time, sys, os, math
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from zeroconf import ServiceInfo, Zeroconf
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False

from tracker3d import IMUDisplacementTracker

fall_events = []
sensor_data = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0, "mag": 0, "gyro_mag": 0, "state": "Normal", "confidence": 0}
sensor_count = 0
fall_count = 0

tracker = IMUDisplacementTracker(fs=333.0)
tracker.set_origin()  # origins at first stationary moment
pos_data = tracker.get_state()

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def register_mdns():
    if not HAS_ZEROCONF:
        print("  [!] zeroconf not installed, mDNS disabled")
        print("  [!] Run: pip install zeroconf")
        return None, None
    try:
        local_ip = get_local_ip()
        zc = Zeroconf()
        info = ServiceInfo(
            "_fall-detector._tcp.local.",
            "Fall Detector._fall-detector._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=SOCKET_PORT,
            properties={},
            server="fall-detector.local.",
        )
        zc.register_service(info)
        print(f"  mDNS: fall-detector.local -> {local_ip}")
        return zc, info
    except Exception as e:
        print(f"  [!] mDNS failed: {e}")
        return None, None

class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/fall':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(fall_events, ensure_ascii=False).encode())
        elif self.path == '/api/sensor':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(sensor_data).encode())
        elif self.path == '/api/position':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(pos_data).encode())
        elif self.path == '/api/origin':
            global tracker
            tracker.set_origin()
            pos_data = tracker.get_state()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"origin":"set","ok":true}')
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
                fall_events.append(data)
                mag = data.get('peak_magnitude', 0)
                print(f"[{data['received_at']}] FALL (HTTP)! mag={mag:.1f}")
                # 直接推送 WebSocket（同进程模式） + 桌面弹窗
                try:
                    from rehab_monitor.api_server import broadcast_fall_alert
                    conf = min(mag / 10.0, 1.0)
                    broadcast_fall_alert((0, 0), conf)
                except Exception:
                    pass
                try:
                    from ubuntu.fall_popup import enqueue_fall
                    enqueue_fall("wristband", conf, {"magnitude": mag})
                except Exception:
                    pass
            except Exception as e:
                print(f"Parse error: {e}")
            response = b'{"ok":true}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.send_header('Connection', 'close')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, *args):
        pass

HTML_PAGE = """<!DOCTYPE html>
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
}load();setInterval(load,500);
</script></body></html>"""

HTTP_PORT = 8080
SOCKET_PORT = 8081

def now():
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]

def terminal_viz():
    """Terminal real-time sensor display — reads globals, zero I/O to ESP32."""
    global sensor_data, sensor_count, fall_count, pos_data
    last = 0
    while True:
        time.sleep(0.1)
        if sensor_count == last: continue
        last = sensor_count

        s = sensor_data
        p = pos_data
        is_fall = s.get('state', 'Normal') == 'DETECTED'
        c = '\033[1;31m' if is_fall else '\033[0m'

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"  Samples: {sensor_count}  |  Falls: {fall_count}  |  {c}State: {s.get('state','Normal')}\033[0m")
        print()
        print(f"  Accel X: {s.get('ax',0):8.2f} m/s2     Gyro X: {s.get('gx',0):8.1f} dps")
        print(f"  Accel Y: {s.get('ay',0):8.2f} m/s2     Gyro Y: {s.get('gy',0):8.1f} dps")
        print(f"  Accel Z: {s.get('az',0):8.2f} m/s2     Gyro Z: {s.get('gz',0):8.1f} dps")
        print(f"  Mag:     {s.get('mag',0):8.2f} m/s2     Gyro Mag: {s.get('gyro_mag',0):8.0f} dps")
        print(f"  Confidence: {s.get('confidence',0):6.0%}")
        print()
        # Position display
        conf = p.get('confidence', 0)
        if conf > 0.6: pc = '\033[1;32m'  # green
        elif conf > 0.3: pc = '\033[1;33m'  # yellow
        else: pc = '\033[1;31m'  # red
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
            print(f"  ╔════════════════════════════╗")
            print(f"  ║   FALL ALERT ACTIVE      ║")
            print(f"  ╚════════════════════════════╝")
        print()
        print("  [Web dashboard: http://{}:{}]".format(get_local_ip(), HTTP_PORT))
        print("  [Set Origin:  GET /api/origin  or press Ctrl+O in browser]")

def main():
    local_ip = get_local_ip()
    viz = '--viz' in sys.argv

    print("=" * 55)
    print("  Fall Detection Server")
    print("=" * 55)
    print(f"  HTTP:  http://{local_ip}:{HTTP_PORT}")
    print(f"  Stream: {local_ip}:{SOCKET_PORT}")
    if viz: print(f"  Terminal viz: ON")
    print()

    zc, info = register_mdns()

    print("=" * 55)
    print()

    HTTPServer.allow_reuse_address = True   # 允许快速重启时端口复用
    threading.Thread(target=lambda: HTTPServer(('0.0.0.0', HTTP_PORT), HTTPHandler).serve_forever(), daemon=True).start()
    if viz: threading.Thread(target=terminal_viz, daemon=True).start()

    subscribers = []
    sub_lock = threading.Lock()

    def broadcast(line):
        with sub_lock:
            dead = []
            for c in subscribers:
                try: c.sendall(line.encode())
                except Exception as e:
                    print(f"[{now()}] Broadcast error: {e}")
                    dead.append(c)
            for d in dead: subscribers.remove(d)

    def client_handler(conn, addr):
        global sensor_count, fall_count, sensor_data
        conn.settimeout(5.0)
        buf = ""
        is_sub = False
        print(f"[{now()}] {'Viz' if addr[1] != SOCKET_PORT else 'ESP32'} connected: {addr[0]}:{addr[1]}")
        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    print(f"[{now()}] Disconnected: {addr[0]}:{addr[1]}")
                    break
                buf += data.decode('utf-8', errors='ignore')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if not line: continue
                    try:
                        msg = json.loads(line)
                        if msg.get('type') == 'subscribe':
                            is_sub = True
                            with sub_lock: subscribers.append(conn)
                            print(f"[{now()}] Viz subscriber added (total: {len(subscribers)})")
                            conn.sendall(b'{"type":"ok"}\n')
                        elif msg.get('type') == 'sensor':
                            sensor_count += 1
                            sensor_data = msg
                            # Update IMU displacement tracker
                            global pos_data, tracker
                            # ESP32 sends ~333Hz (3ms interval). Use real
                            # timestamp delta when available; fall back to
                            # 1/333 s nominal rate.
                            dt = 1.0 / 333.0
                            p, v, conf, zupt = tracker.update(
                                msg.get('ax', 0), msg.get('ay', 0), msg.get('az', 0),
                                msg.get('gx', 0), msg.get('gy', 0), msg.get('gz', 0),
                                dt
                            )
                            pos_data = tracker.get_state()
                            if sensor_count % 30 == 0:
                                print(f"[{now()}] #{sensor_count} "
                                      f"Accel=({msg.get('ax',0):6.2f},{msg.get('ay',0):6.2f},{msg.get('az',0):6.2f}) "
                                      f"State={msg.get('state','Normal')} Conf={msg.get('confidence',0):.0%}")
                            broadcast(line + '\n')
                        elif msg.get('type') == 'fall':
                            fall_count += 1
                            msg['received_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            fall_events.append(msg)
                            peak = msg.get('peak_mag', 0)
                            conf = msg.get('confidence', 0)
                            print(f"\n{'='*55}")
                            print(f"  FALL #{fall_count} | Peak: {peak:.1f} m/s2 | Conf: {conf:.0%}")
                            print(f"{'='*55}\n")
                            broadcast(line + '\n')

                            # Bridge: WebSocket → 微信小程序 (same-process mode)
                            try:
                                from rehab_monitor.api_server import broadcast_fall_alert
                                broadcast_fall_alert((0, 0), conf)
                            except Exception:
                                pass
                    except json.JSONDecodeError: pass
            except socket.timeout: continue
            except ConnectionResetError: break
            except Exception as e: print(f"Error: {e}"); break
        if is_sub:
            with sub_lock:
                if conn in subscribers:
                    subscribers.remove(conn)
        conn.close()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', SOCKET_PORT))
    sock.listen(5)
    sock.settimeout(1.0)

    print("Waiting for ESP32 + viz clients...")
    print()

    try:
        while True:
            try:
                conn, addr = sock.accept()
                threading.Thread(target=client_handler, args=(conn, addr), daemon=True).start()
            except socket.timeout: continue
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
    finally:
        sock.close()
        if zc and info:
            try: zc.unregister_service(info); zc.close()
            except: pass

if __name__ == '__main__':
    main()
