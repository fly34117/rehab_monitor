"""Flask API 服务器 — 为微信小程序提供 REST + WebSocket 接口"""
import json
import time
import uuid
import base64
import queue
import asyncio
import threading
from functools import wraps
from collections import defaultdict
from threading import Lock

import cv2
import numpy as np
import websockets
from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS

from .config import (
    API_VERSION, API_HOST, API_PORT, WS_PORT, API_DEBUG, API_CORS_ORIGINS,
    API_RATE_LIMIT_PER_SECOND,
    WEBSOCKET_HEARTBEAT_INTERVAL, WEBSOCKET_PUSH_INTERVAL,
    LOCK_SIMILARITY_THRESHOLD,
)
from .logging_setup import get_logger

logger = get_logger("api")

# ===== 模块级全局变量（由 start_api_server 注入）=====
gait_analyzer = None
spatial_mapper = None
database = None
face_locker = None
emotion_recognizer = None

# ===== 当前锁定状态（由主循环更新）=====
current_lock = {"locked": False, "name": "", "similarity": 0.0, "source": "none"}
current_lock_lock = Lock()

def update_current_lock(locked, name="", similarity=0.0, source="none"):
    """主循环调用：更新当前锁定状态，供 API health 查询"""
    global current_lock
    with current_lock_lock:
        current_lock = {"locked": locked, "name": name,
                        "similarity": similarity, "source": source}

# ===== 限流 =====
rate_limit_storage = defaultdict(list)
rate_limit_lock = Lock()

# ===== 待处理锁定请求（API → 主循环通信）=====
pending_lock_request = None  # {"name": str, "embedding": np.ndarray} 或 None
pending_unlock_request = False  # 待处理解锁请求
pending_lock_lock = Lock()

def consume_pending_lock():
    """主循环调用：取出并清除待处理的锁定请求。返回 (name, embedding) 或 (None, None)"""
    global pending_lock_request
    with pending_lock_lock:
        if pending_lock_request is None:
            return None, None
        req = pending_lock_request
        pending_lock_request = None
        return req["name"], req["embedding"]

def consume_pending_unlock():
    """主循环调用：检查是否有待处理的解锁请求"""
    global pending_unlock_request
    with pending_lock_lock:
        if pending_unlock_request:
            pending_unlock_request = False
            return True
        return False

# ===== WebSocket 客户端管理（websockets 库）=====
websocket_clients = set()  # {websocket}
clients_lock = Lock()
fall_alert_queue = queue.Queue()  # 跌倒告警队列
app_start_time = time.time()


async def _ws_send_all(message):
    """向所有连接的 WebSocket 客户端广播消息"""
    dead = set()
    with clients_lock:
        clients = list(websocket_clients)
    for ws in clients:
        try:
            await ws.send(message)
        except Exception:
            dead.add(ws)
    if dead:
        with clients_lock:
            websocket_clients.difference_update(dead)


def broadcast_fall_alert(location, score):
    """向所有 WebSocket 客户端广播跌倒告警（线程安全）"""
    alert = json.dumps({
        "type": "fall_alert",
        "data": {
            "timestamp": time.time(),
            "location": list(location) if location else [0, 0],
            "score": round(score, 3),
            "severity": "high" if score > 0.8 else "medium"
        }
    })
    fall_alert_queue.put(alert)


async def _ws_handler(ws):
    """单个 WebSocket 客户端处理协程"""
    client_id = str(uuid.uuid4())[:8]
    with clients_lock:
        websocket_clients.add(ws)
    logger.info("WebSocket 连接: %s (当前 %d 个)", client_id, len(websocket_clients))

    # 发送连接确认
    try:
        await ws.send(json.dumps({"type": "connected", "timestamp": time.time()}))
    except Exception:
        pass

    try:
        # 接收循环：只用于检测断线 + 处理 pong
        async for message in ws:
            try:
                msg = json.loads(message)
                if msg.get("type") == "pong":
                    pass  # 心跳回复
            except Exception:
                pass
    except Exception:
        pass
    finally:
        with clients_lock:
            websocket_clients.discard(ws)
        logger.info("WebSocket 断开: %s (剩余 %d 个)", client_id, len(websocket_clients))


async def _ws_broadcast_loop():
    """后台广播循环：定期推送指标 + 跌倒告警"""
    last_heartbeat = time.time()
    while True:
        await asyncio.sleep(WEBSOCKET_PUSH_INTERVAL)
        now = time.time()

        # 心跳
        if now - last_heartbeat > WEBSOCKET_HEARTBEAT_INTERVAL:
            await _ws_send_all(json.dumps({"type": "ping"}))
            last_heartbeat = now

        # 跌倒告警（非阻塞检查队列）
        try:
            while True:
                alert = fall_alert_queue.get_nowait()
                await _ws_send_all(alert)
        except queue.Empty:
            pass

        # 指标推送
        if gait_analyzer is not None:
            try:
                metrics = gait_analyzer.get_metrics()
                with current_lock_lock:
                    metrics["_locked"] = current_lock["locked"]
                msg = json.dumps({
                    "type": "metrics",
                    "data": metrics,
                    "timestamp": now
                })
                await _ws_send_all(msg)
            except Exception:
                pass


def _run_ws_server(host, port):
    """在独立线程中运行 websockets 服务器（同步包装）"""
    async def _serve():
        async with websockets.serve(_ws_handler, host, port, ping_interval=None):
            # 同时运行广播循环
            await _ws_broadcast_loop()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_serve())
    except Exception as e:
        logger.error("WebSocket 服务器异常: %s", e)


def rate_limit(per_second=2):
    """请求限流装饰器，使用 session_id + IP 组合键，线程安全"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            session_id = request.headers.get('X-Session-Id', 'default')
            client_key = f"{request.remote_addr}_{session_id}"
            now = time.time()
            with rate_limit_lock:
                window = rate_limit_storage[client_key]
                window[:] = [t for t in window if now - t < 1.0]
                if len(window) >= per_second:
                    return jsonify({
                        "code": -1,
                        "message": "请求过于频繁，请稍后再试"
                    }), 429
                window.append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def create_app():
    """创建并配置 Flask 应用"""
    app = Flask(__name__)
    CORS(app, origins=API_CORS_ORIGINS)

    # ---- REST 路由 ----

    @app.route(f'/api/{API_VERSION}/health')
    def health():
        with current_lock_lock:
            lock = dict(current_lock)
        import platform
        return jsonify({
            "code": 0,
            "data": {
                "status": "ok",
                "version": API_VERSION,
                "uptime": round(time.time() - app_start_time, 1),
                "gait_ready": gait_analyzer is not None,
                "db_ready": database is not None,
                "face_locker_ready": face_locker is not None,
                "websocket_clients": len(websocket_clients),
                "target_locked": lock,
                "hostname": platform.node(),
                "timestamp": time.time()
            }
        })

    @app.route(f'/api/{API_VERSION}/metrics/realtime')
    @rate_limit(per_second=API_RATE_LIMIT_PER_SECOND)
    def metrics_realtime():
        if gait_analyzer is None:
            return jsonify({"code": -1, "message": "步态分析器未就绪"})
        with current_lock_lock:
            locked = current_lock["locked"]
        try:
            data = gait_analyzer.get_metrics()
            data["_locked"] = locked
            return jsonify({"code": 0, "data": data})
        except Exception as e:
            logger.error("获取实时指标失败: %s", e)
            return jsonify({"code": -1, "message": str(e)})

    @app.route(f'/api/{API_VERSION}/metrics/history')
    @rate_limit(per_second=API_RATE_LIMIT_PER_SECOND)
    def metrics_history():
        if database is None:
            return jsonify({"code": -1, "message": "数据库未就绪"})
        days = request.args.get('days', 7, type=int)
        limit = request.args.get('limit', 30, type=int)
        cutoff = time.time() - days * 86400
        try:
            c = database.conn.cursor()
            c.execute("""
                SELECT DATE(timestamp, 'unixepoch') as date,
                       AVG(gait_rehab_score), AVG(gait_velocity_mps),
                       AVG(symmetry), AVG(cadence), AVG(stride_length_m),
                       AVG(left_knee_rom), AVG(right_knee_rom),
                       AVG(step_time_cv)
                FROM gait_metrics
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp, 'unixepoch')
                ORDER BY date DESC LIMIT ?
            """, (cutoff, limit))
            rows = c.fetchall()
            result = []
            for row in rows:
                result.append({
                    "date": row[0],
                    "gait_rehab_score": round(row[1], 1) if row[1] else 0,
                    "gait_velocity_mps": round(row[2], 2) if row[2] else 0,
                    "symmetry": round(row[3], 3) if row[3] else 0,
                    "cadence_spm": round(row[4], 1) if row[4] else 0,
                    "stride_length_m": round(row[5], 2) if row[5] else 0,
                    "left_knee_rom": round(row[6], 1) if row[6] else 0,
                    "right_knee_rom": round(row[7], 1) if row[7] else 0,
                    "step_time_cv": round(row[8], 1) if row[8] else 0,
                })
            return jsonify({"code": 0, "data": result})
        except Exception as e:
            logger.error("获取历史趋势失败: %s", e)
            return jsonify({"code": -1, "message": str(e)})

    @app.route(f'/api/{API_VERSION}/emotion/stats')
    @rate_limit(per_second=API_RATE_LIMIT_PER_SECOND)
    def emotion_stats():
        if database is None:
            return jsonify({"code": -1, "message": "数据库未就绪"})
        days = request.args.get('days', 7, type=int)
        cutoff = time.time() - days * 86400
        try:
            c = database.conn.cursor()
            c.execute("""
                SELECT DATE(timestamp, 'unixepoch') as date,
                       emotion_label, COUNT(*) as cnt
                FROM emotion_log
                WHERE timestamp >= ?
                GROUP BY date, emotion_label
                ORDER BY date
            """, (cutoff,))
            rows = c.fetchall()
            result = {}
            for date, label, cnt in rows:
                if date not in result:
                    result[date] = {}
                result[date][label] = cnt
            return jsonify({"code": 0, "data": result})
        except Exception as e:
            logger.error("获取心情统计失败: %s", e)
            return jsonify({"code": -1, "message": str(e)})

    @app.route(f'/api/{API_VERSION}/trajectory')
    @rate_limit(per_second=API_RATE_LIMIT_PER_SECOND)
    def trajectory():
        if spatial_mapper is None:
            return jsonify({"code": -1, "message": "空间定位器未就绪"})
        limit = request.args.get('limit', 100, type=int)
        try:
            traj = spatial_mapper.get_trajectory()
            points = traj[-limit:] if len(traj) > limit else traj
            result = [{"x": round(p[0], 3), "y": round(p[1], 3)} for p in points]
            return jsonify({"code": 0, "data": result})
        except Exception as e:
            logger.error("获取轨迹失败: %s", e)
            return jsonify({"code": -1, "message": str(e)})

    @app.route(f'/api/{API_VERSION}/patient/lock', methods=['POST'])
    @rate_limit(per_second=3)
    def patient_lock():
        """拍照锁定患者：先尝试匹配，无匹配则自动录入"""
        global pending_lock_request
        if face_locker is None:
            return jsonify({"code": -1, "message": "人脸识别器未就绪"})

        try:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"code": -1, "message": "请求体为空"})

            image_base64 = data.get('image', '')
            if not image_base64:
                return jsonify({"code": -1, "message": "缺少图片数据"})

            # 去掉 data:image/...;base64, 前缀
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            # Base64 → OpenCV BGR
            img_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"code": -1, "message": "图片解码失败"})

            # 提取人脸嵌入
            emb = face_locker.enroll_from_frame(frame)
            if emb is None:
                return jsonify({"code": -1, "message": "未检测到人脸，请正对摄像头"})

            # 尝试匹配已有数据库
            name, similarity = face_locker.match(emb, threshold=LOCK_SIMILARITY_THRESHOLD)
            if name is not None:
                # 通知主循环锁定该患者
                with pending_lock_lock:
                    pending_lock_request = {"name": name, "embedding": emb}
                logger.info("API 锁定请求: %s (match, sim=%.3f)", name, similarity)
                return jsonify({
                    "code": 0,
                    "data": {
                        "action": "match",
                        "patientId": name,
                        "similarity": round(similarity, 3),
                        "message": f"欢迎回来，{name}"
                    }
                })

            # 无匹配 → 录入新患者
            import os
            patient_id = f"P{int(time.time())}"
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "face_db.json")
            db_data = {"entries": []}
            if os.path.exists(db_path):
                with open(db_path, "r") as f:
                    db_data = json.load(f)

            db_data["entries"].append({
                "name": patient_id,
                "embedding": emb.tolist(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            with open(db_path, "w") as f:
                json.dump(db_data, f)
            face_locker._load_db()

            # 通知主循环锁定该患者
            with pending_lock_lock:
                pending_lock_request = {"name": patient_id, "embedding": emb}
            logger.info("API 锁定请求: %s (enroll)", patient_id)

            return jsonify({
                "code": 0,
                "data": {
                    "action": "enroll",
                    "patientId": patient_id,
                    "similarity": 1.0,
                    "message": "新患者已录入"
                }
            })

        except Exception as e:
            logger.error("锁定患者失败: %s", e)
            return jsonify({"code": -1, "message": str(e)})

    @app.route(f'/api/{API_VERSION}/patient/unlock', methods=['POST'])
    def patient_unlock():
        """取消锁定：通知主循环清除锁定状态和人脸库"""
        global pending_unlock_request
        with pending_lock_lock:
            pending_unlock_request = True
            pending_lock_request = None  # 同时取消待处理的锁定
        logger.info("API 解锁请求")
        return jsonify({
            "code": 0,
            "data": {"message": "已发送解锁指令"}
        })

    @app.route(f'/api/{API_VERSION}/report/generate', methods=['POST'])
    @rate_limit(per_second=2)
    def report_generate():
        if database is None:
            return jsonify({"code": -1, "message": "数据库未就绪"})
        seconds = request.args.get('seconds', 30, type=int)
        try:
            data = database.get_recent_data(seconds=seconds)
            # 检查是否有足够的步态数据
            gs = data.get("gait_summary", {}) if data else {}
            gait_count = data.get("gait_records", 0) if data else 0
            if not gs or gait_count < 1:
                return jsonify({
                    "code": -1,
                    "message": "步态数据不足，请先锁定患者并让其在画面中行走一段距离后再试"
                })
            from .llm_client import generate_report
            text, summary, err = generate_report(data)
            if err:
                return jsonify({
                    "code": -1,
                    "message": f"报告生成失败: {err[:80]}"
                })
            # 同步到对话历史（小程序和 GUI 都能看到）
            try:
                from .chat_store import chat_store
                chat_store.add_message("user", "📋 请求生成康复分析报告")
                chat_store.add_message("assistant", text or "")
            except Exception:
                pass
            return jsonify({
                "code": 0,
                "data": {
                    "text": text,
                    "summary": summary,
                    "period_seconds": seconds,
                    "timestamp": time.time()
                }
            })
        except Exception as e:
            logger.error("生成报告失败: %s", e)
            return jsonify({"code": -1, "message": str(e)})

    @app.route(f'/api/{API_VERSION}/report/expert', methods=['POST'])
    @rate_limit(per_second=1)
    def report_expert():
        """专家知识库分析报告 — 结合 13 篇权威论文 + DeepSeek API"""
        if database is None:
            return jsonify({"code": -1, "message": "数据库未就绪"})

        seconds = request.args.get('seconds', 60, type=int)
        trend_days = request.args.get('trend_days', 0, type=int)

        try:
            # 1. 获取步态统计数据
            sid = database.session_id
            if sid is None:
                # 取最近一次会话
                c = database.conn.cursor()
                c.execute("SELECT MAX(id) FROM sessions")
                row = c.fetchone()
                if row and row[0]:
                    sid = row[0]
                else:
                    return jsonify({"code": -1, "message": "无会话数据"})

            gait_stats = database.get_gait_stats(session_id=sid, seconds=seconds)
            if gait_stats is None:
                return jsonify({"code": -1, "message": f"最近 {seconds}s 内无步态数据"})

            # 2. 获取趋势数据（可选）
            trend_data = None
            if trend_days > 0:
                trend_data = database.get_monthly_gait_trend(days=trend_days)

            # 3. 调用本地 LLM 专家分析
            from .llm_client import generate_expert_report
            report_json, cited_papers, raw, err = generate_expert_report(
                gait_stats, trend_data
            )

            if err:
                return jsonify({
                    "code": -1,
                    "message": f"专家分析失败: {err[:100]}"
                })

            # 4. 格式化报告文本
            if isinstance(report_json, dict):
                from .expert_report import ExpertReportGenerator
                expert = ExpertReportGenerator()
                formatted = expert.format_report(report_json, cited_papers)
            else:
                formatted = raw or str(report_json)

            # 5. 保存到数据库
            if raw:
                database.write_report(
                    formatted,
                    report_json if isinstance(report_json, dict) else {"raw": str(report_json)},
                )

            return jsonify({
                "code": 0,
                "data": {
                    "text": formatted,
                    "summary": report_json.get("summary", "") if isinstance(report_json, dict) else "",
                    "report_json": report_json,
                    "cited_papers": [p["filename"] for p in cited_papers],
                    "cited_count": len(cited_papers),
                    "period_seconds": seconds,
                    "trend_days": trend_days,
                    "timestamp": time.time()
                }
            })

        except Exception as e:
            logger.error("专家报告生成失败: %s", e)
            return jsonify({"code": -1, "message": str(e)[:200]})

    # ── 对话历史（本地 LLM 共享）──

    @app.route(f'/api/{API_VERSION}/chat/history')
    def chat_history():
        """获取本地 LLM 对话历史"""
        try:
            from .chat_store import chat_store
            messages = chat_store.get_history()
            return jsonify({
                "code": 0,
                "data": {
                    "messages": messages,
                    "count": len(messages),
                }
            })
        except Exception as e:
            logger.error("获取对话历史失败: %s", e)
            return jsonify({"code": -1, "message": str(e)[:200]})

    @app.route(f'/api/{API_VERSION}/chat/clear', methods=['POST'])
    def chat_clear():
        """清除本地 LLM 对话历史"""
        try:
            from .chat_store import chat_store
            cleared = chat_store.clear()
            logger.info("对话历史已清除 (%d 条)", cleared)
            return jsonify({
                "code": 0,
                "data": {"cleared": cleared}
            })
        except Exception as e:
            logger.error("清除对话历史失败: %s", e)
            return jsonify({"code": -1, "message": str(e)[:200]})

    @app.route(f'/api/{API_VERSION}/chat/send', methods=['POST'])
    @rate_limit(per_second=3)
    def chat_send():
        """发送消息到本地 LLM 并获取回复"""
        try:
            data = request.get_json(force=True, silent=True)
            if not data or "message" not in data:
                return jsonify({"code": -1, "message": "缺少 message 字段"})
            msg = data["message"].strip()
            if not msg:
                return jsonify({"code": -1, "message": "消息不能为空"})

            from .chat_store import chat_store
            chat_store.add_message("user", msg)

            system_prompt = (
                "你是康复助手小安。用中文简洁专业回答康复相关问题。"
                "非康复问题请礼貌拒绝。回答不超过200字。"
            )

            # 先检查 llama-server 可达性
            from .llm_client import _check_health
            if not _check_health():
                return jsonify({"code": -1, "message": "LLM 服务未启动，请确认 GUI 已加载模型"})

            from .llm_client import _run_llm
            reply, error = _run_llm(system_prompt, msg, max_tokens=300, temperature=0.3)

            if error:
                logger.error("LLM 调用失败: %s", error)
                return jsonify({"code": -1, "message": error[:200]})

            reply = (reply or "").strip()
            chat_store.add_message("assistant", reply)

            return jsonify({
                "code": 0,
                "data": {"reply": reply, "user_message": msg}
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error("对话发送失败:\n%s", tb)
            return jsonify({"code": -1, "message": f"{type(e).__name__}: {e}"})

    return app


def start_api_server(gait, spatial, db, face_locker_, detector, emotion,
                     debug=False, port=None, ws_port=None):
    """在后台 daemon 线程启动 Flask API 服务器 + WebSocket 服务器

    Args:
        gait: GaitAnalyzer 实例
        spatial: SpatialMapper 实例
        db: RehabDatabase 实例
        face_locker_: FaceNetLocker 实例
        detector: PoseDetector 实例
        emotion: EmotionRecognizer 实例
        debug: 是否启用 Flask debug 模式
        port: API 端口 (默认使用 config.API_PORT)
        ws_port: WebSocket 端口 (默认使用 config.WS_PORT)

    Returns:
        (flask_thread, ws_thread) 两个 threading.Thread 对象
    """
    global gait_analyzer, spatial_mapper, database, face_locker, emotion_recognizer
    gait_analyzer = gait
    spatial_mapper = spatial
    database = db
    face_locker = face_locker_
    emotion_recognizer = emotion

    app = create_app()

    _port = port if port is not None else API_PORT
    _ws_port = ws_port if ws_port is not None else WS_PORT

    # Flask REST API 线程
    def _run_flask():
        logger.info("API 服务器启动 http://%s:%d (debug=%s)", API_HOST, _port, debug)
        app.run(host=API_HOST, port=_port, debug=debug, threaded=True, use_reloader=False)

    flask_thread = threading.Thread(target=_run_flask, daemon=True, name="api-server")
    flask_thread.start()

    # WebSocket 服务线程（websockets 库）
    ws_thread = threading.Thread(
        target=_run_ws_server, args=(API_HOST, _ws_port),
        daemon=True, name="ws-server"
    )
    ws_thread.start()
    logger.info("WebSocket 服务器启动 ws://%s:%d", API_HOST, _ws_port)

    return flask_thread
