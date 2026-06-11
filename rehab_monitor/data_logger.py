"""数据存储 — SQLite 建表 + 写入 + 查询 + 统计分析 (支持异步写入)"""
import sqlite3
import json
import time
import queue
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict

from .config import DB_PATH
from .logging_setup import get_logger

logger = get_logger("db")


class RehabDatabase:
    """康复监测数据库"""

    def __init__(self, db_path=None, async_mode=True):
        self.db_path = db_path or DB_PATH
        self.async_mode = async_mode
        self.session_id = None

        # 主线程连接（读 + 建表）
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        self._migrate(self.conn)

        if async_mode:
            self._write_queue = queue.Queue(maxsize=500)
            self._running = True
            self._write_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._write_thread.start()
            logger.info("异步写入模式已启用 (后台线程 + 专用连接)")

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT 'active'
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS frame_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL,
            person_count INTEGER DEFAULT 0,
            keypoints_json TEXT,
            joint_angles_json TEXT,
            fall_status TEXT DEFAULT 'safe',
            fall_score REAL DEFAULT 0.0,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS gait_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL,
            step_count INTEGER DEFAULT 0,
            cadence REAL DEFAULT 0.0,
            speed REAL DEFAULT 0.0,
            symmetry REAL DEFAULT 1.0,
            stride_length_m REAL DEFAULT 0.0,
            gait_velocity_mps REAL DEFAULT 0.0,
            step_width_m REAL DEFAULT 0.0,
            stance_percentage REAL DEFAULT 0.0,
            left_knee_rom REAL DEFAULT 0.0,
            right_knee_rom REAL DEFAULT 0.0,
            step_time_cv REAL DEFAULT 0.0,
            step_length_cv REAL DEFAULT 0.0,
            foot_clearance_cm REAL DEFAULT 0.0,
            gait_rehab_score REAL DEFAULT 0.0,
            trunk_sway_deg REAL DEFAULT 0.0,
            double_support_ratio REAL DEFAULT 0.0,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS emotion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL,
            emotion_label TEXT,
            emotion_scores_json TEXT,
            face_bbox_json TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS fall_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL,
            alert_level TEXT DEFAULT 'alert',
            duration_frames INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            report_type TEXT DEFAULT '30s',
            generated_at REAL NOT NULL,
            report_text TEXT,
            summary_json TEXT,
            error_message TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )""")
        self._create_index_if_not_exists(c, "idx_gait_timestamp",
            "gait_metrics", "timestamp")
        self._create_index_if_not_exists(c, "idx_gait_session",
            "gait_metrics", "session_id, timestamp")
        self._create_index_if_not_exists(c, "idx_emotion_timestamp",
            "emotion_log", "timestamp")
        self.conn.commit()

    def _create_index_if_not_exists(self, conn_or_cursor, idx_name, table, columns):
        """仅当索引不存在时才创建，避免每次启动重复执行"""
        c = conn_or_cursor
        c.execute(
            f"SELECT name FROM sqlite_master WHERE type='index' AND name='{idx_name}'"
        )
        if not c.fetchone():
            c.execute(f"CREATE INDEX {idx_name} ON {table}({columns})")

    def _migrate(self, conn):
        """自动迁移：为旧数据库添加缺失的列"""
        expected = {
            "gait_metrics": {
                "stride_length_m": "REAL DEFAULT 0.0",
                "gait_velocity_mps": "REAL DEFAULT 0.0",
                "step_width_m": "REAL DEFAULT 0.0",
                "stance_percentage": "REAL DEFAULT 0.0",
                "left_knee_rom": "REAL DEFAULT 0.0",
                "right_knee_rom": "REAL DEFAULT 0.0",
                "step_time_cv": "REAL DEFAULT 0.0",
                "step_length_cv": "REAL DEFAULT 0.0",
                "foot_clearance_cm": "REAL DEFAULT 0.0",
                "gait_rehab_score": "REAL DEFAULT 0.0",
                "trunk_sway_deg": "REAL DEFAULT 0.0",
                "double_support_ratio": "REAL DEFAULT 0.0",
            },
        }
        c = conn.cursor()
        for table, columns in expected.items():
            c.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in c.fetchall()}
            for col_name, col_def in columns.items():
                if col_name not in existing:
                    try:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
                        logger.info("迁移: %s.%s 已添加", table, col_name)
                    except Exception as e:
                        logger.warning("迁移失败 %s.%s: %s", table, col_name, e)
        conn.commit()

    def _writer_loop(self):
        """后台写入线程：专用连接，从队列取任务立即写入并提交"""
        wconn = sqlite3.connect(self.db_path, check_same_thread=False)
        wconn.execute("PRAGMA journal_mode=WAL")
        wconn.execute("PRAGMA synchronous=OFF")  # 更快写入，WAL 保证安全性
        self._create_tables_on(wconn)
        self._migrate(wconn)

        while self._running:
            try:
                task = self._write_queue.get(timeout=0.5)
                if task is None:
                    break
                task(wconn)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("异步写入错误: %s", e)

        wconn.close()

    def _create_tables_on(self, conn):
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL, end_time TEXT, status TEXT DEFAULT 'active')""")
        c.execute("""CREATE TABLE IF NOT EXISTS frame_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL, timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL, person_count INTEGER DEFAULT 0,
            keypoints_json TEXT, joint_angles_json TEXT,
            fall_status TEXT DEFAULT 'safe', fall_score REAL DEFAULT 0.0,
            FOREIGN KEY (session_id) REFERENCES sessions(id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS gait_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL, timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL, step_count INTEGER DEFAULT 0,
            cadence REAL DEFAULT 0.0, speed REAL DEFAULT 0.0,
            symmetry REAL DEFAULT 1.0,
            stride_length_m REAL DEFAULT 0.0,
            gait_velocity_mps REAL DEFAULT 0.0,
            step_width_m REAL DEFAULT 0.0,
            stance_percentage REAL DEFAULT 0.0,
            left_knee_rom REAL DEFAULT 0.0,
            right_knee_rom REAL DEFAULT 0.0,
            step_time_cv REAL DEFAULT 0.0,
            step_length_cv REAL DEFAULT 0.0,
            foot_clearance_cm REAL DEFAULT 0.0,
            gait_rehab_score REAL DEFAULT 0.0,
            trunk_sway_deg REAL DEFAULT 0.0,
            double_support_ratio REAL DEFAULT 0.0,
            FOREIGN KEY (session_id) REFERENCES sessions(id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS emotion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL, timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL, emotion_label TEXT,
            emotion_scores_json TEXT, face_bbox_json TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS fall_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL, timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL, alert_level TEXT DEFAULT 'alert',
            duration_frames INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL, report_type TEXT DEFAULT '30s',
            generated_at REAL NOT NULL, report_text TEXT,
            summary_json TEXT, error_message TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id))""")
        conn.commit()

    def _enqueue(self, task):
        """非阻塞入队；队列满时丢弃最旧任务"""
        try:
            self._write_queue.put_nowait(task)
        except queue.Full:
            try:
                self._write_queue.get_nowait()
                self._write_queue.put_nowait(task)
            except queue.Full:
                pass  # 极端情况，丢弃

    # ----- 会话 -----
    def start_session(self):
        c = self.conn.cursor()
        c.execute("INSERT INTO sessions (start_time, status) VALUES (?, 'active')",
                  (datetime.now().isoformat(),))
        self.conn.commit()
        self.session_id = c.lastrowid
        return self.session_id

    def end_session(self):
        if self.session_id is None:
            return
        sid = self.session_id
        end_time = datetime.now().isoformat()

        def do_write(conn):
            conn.execute(
                "UPDATE sessions SET end_time=?, status='completed' WHERE id=?",
                (end_time, sid))
            conn.commit()

        if self.async_mode:
            self._enqueue(do_write)
        else:
            do_write(self.conn)

    # ----- 每帧快照 -----
    def write_frame_snapshot(self, frame_num, kpts, angles, fall_status, fall_score):
        if self.session_id is None:
            return
        sid = self.session_id
        ts = time.time()
        kpts_json = json.dumps(kpts.tolist()) if kpts is not None else None
        angles_json = json.dumps({k: round(v, 1) if v else None
                                  for k, v in angles.items()}) if angles else None
        person_count = kpts.shape[0] if kpts is not None else 0
        fs = round(fall_score, 3)

        def do_write(conn):
            conn.execute(
                """INSERT INTO frame_snapshots
                   (session_id, timestamp, frame_number, person_count,
                    keypoints_json, joint_angles_json, fall_status, fall_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, ts, frame_num, person_count,
                 kpts_json, angles_json, fall_status, fs))
            conn.commit()

        if self.async_mode:
            self._enqueue(do_write)
        else:
            do_write(self.conn)

    # ----- 步态指标 -----
    def write_gait_metrics(self, frame_num, step_count, cadence, speed, symmetry,
                            stride_length_m=0.0, gait_velocity_mps=0.0,
                            step_width_m=0.0, stance_percentage=0.0,
                            left_knee_rom=0.0, right_knee_rom=0.0,
                            step_time_cv=0.0, step_length_cv=0.0,
                            foot_clearance_cm=0.0, gait_rehab_score=0.0,
                            trunk_sway_deg=0.0, double_support_ratio=0.0):
        if self.session_id is None:
            return
        sid = self.session_id
        ts = time.time()

        def do_write(conn):
            conn.execute(
                """INSERT INTO gait_metrics
                   (session_id, timestamp, frame_number, step_count,
                    cadence, speed, symmetry,
                    stride_length_m, gait_velocity_mps,
                    step_width_m, stance_percentage,
                    left_knee_rom, right_knee_rom,
                    step_time_cv, step_length_cv,
                    foot_clearance_cm, gait_rehab_score,
                    trunk_sway_deg, double_support_ratio)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, ts, frame_num, step_count,
                 float(round(cadence, 2)), float(round(speed, 2)),
                 float(round(symmetry, 3)),
                 float(round(stride_length_m, 3)),
                 float(round(gait_velocity_mps, 3)),
                 float(round(step_width_m, 3)),
                 float(round(stance_percentage, 1)),
                 float(round(left_knee_rom, 1)),
                 float(round(right_knee_rom, 1)),
                 float(round(step_time_cv, 1)),
                 float(round(step_length_cv, 1)),
                 float(round(foot_clearance_cm, 1)),
                 float(round(gait_rehab_score, 1)),
                 float(round(trunk_sway_deg, 1)),
                 float(round(double_support_ratio, 3))))
            conn.commit()

        if self.async_mode:
            self._enqueue(do_write)
        else:
            do_write(self.conn)

    # ----- 跌倒告警 -----
    def write_fall_alert(self, frame_num):
        if self.session_id is None:
            return
        sid = self.session_id
        ts = time.time()

        def do_write(conn):
            conn.execute(
                """INSERT INTO fall_alerts
                   (session_id, timestamp, frame_number, alert_level)
                   VALUES (?, ?, ?, 'alert')""",
                (sid, ts, frame_num))
            conn.commit()

        if self.async_mode:
            self._enqueue(do_write)
        else:
            do_write(self.conn)

    # ----- 表情日志 -----
    def write_emotion(self, frame_num, emotion_label, scores, face_bbox):
        if self.session_id is None:
            return
        sid = self.session_id
        ts = time.time()
        scores_json = json.dumps({k: float(v) for k, v in scores.items()}) if scores else None
        bbox_json = json.dumps([int(v) for v in face_bbox]) if face_bbox else None

        def do_write(conn):
            conn.execute(
                """INSERT INTO emotion_log
                   (session_id, timestamp, frame_number, emotion_label,
                    emotion_scores_json, face_bbox_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sid, ts, frame_num, emotion_label, scores_json, bbox_json))
            conn.commit()

        if self.async_mode:
            self._enqueue(do_write)
        else:
            do_write(self.conn)

    # ----- 报告 -----
    def write_report(self, report_text, summary, error=None):
        if self.session_id is None:
            return
        sid = self.session_id
        ts = time.time()
        summary_json = json.dumps(summary, ensure_ascii=False) if summary else None

        def do_write(conn):
            conn.execute(
                """INSERT INTO reports
                   (session_id, report_type, generated_at, report_text, summary_json, error_message)
                   VALUES (?, '30s', ?, ?, ?, ?)""",
                (sid, ts, report_text, summary_json, error))
            conn.commit()

        if self.async_mode:
            self._enqueue(do_write)
        else:
            do_write(self.conn)

    # ----- 查询 -----
    def get_recent_data(self, seconds=30):
        """获取最近 N 秒的数据用于生成报告"""
        cutoff = time.time() - seconds
        c = self.conn.cursor()
        c.execute(
            "SELECT timestamp, frame_number, joint_angles_json, fall_status, fall_score "
            "FROM frame_snapshots WHERE session_id=? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT 1000",
            (self.session_id, cutoff))
        snapshots = c.fetchall()

        c.execute(
            "SELECT timestamp, step_count, cadence, speed, symmetry, "
            "stride_length_m, gait_velocity_mps, step_width_m, stance_percentage, "
            "left_knee_rom, right_knee_rom, step_time_cv, step_length_cv, "
            "foot_clearance_cm, gait_rehab_score, trunk_sway_deg, double_support_ratio "
            "FROM gait_metrics WHERE session_id=? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT 1000",
            (self.session_id, cutoff))
        gait = c.fetchall()

        c.execute(
            "SELECT timestamp, emotion_label, emotion_scores_json "
            "FROM emotion_log WHERE session_id=? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT 1000",
            (self.session_id, cutoff))
        emotions = c.fetchall()

        c.execute(
            "SELECT COUNT(*) FROM fall_alerts WHERE session_id=? AND timestamp >= ?",
            (self.session_id, cutoff))
        fall_count = c.fetchone()[0]

        # 汇总步态指标 (取最新非零值)
        gait_summary = {}
        if gait:
            cols = ["stride_length_m", "gait_velocity_mps", "step_width_m",
                    "stance_percentage", "left_knee_rom", "right_knee_rom",
                    "step_time_cv", "step_length_cv", "foot_clearance_cm",
                    "gait_rehab_score", "trunk_sway_deg", "double_support_ratio"]
            for row in reversed(gait):
                for i, col in enumerate(cols):
                    if col not in gait_summary or gait_summary[col] == 0:
                        val = row[7 + i] if 7 + i < len(row) else 0
                        if val:
                            gait_summary[col] = round(val, 2)

        return {
            "duration": seconds,
            "snapshot_count": len(snapshots),
            "gait_records": len(gait),
            "fall_events": fall_count,
            "emotion_records": len(emotions),
            "snapshots": snapshots,
            "gait": gait,
            "emotions": emotions,
            "gait_summary": gait_summary,
        }

    # ----- 步态统计分析 -----
    def get_gait_stats(self, session_id=None, seconds=None):
        """获取步态指标的统计摘要

        Args:
            session_id: 指定会话ID，None=当前会话
            seconds: 时间窗口（秒），None=全部数据

        Returns:
            dict: 每项指标的 {mean, std, min, max, count, latest}
        """
        sid = session_id or self.session_id
        if sid is None:
            return None

        c = self.conn.cursor()
        cols = [
            "cadence", "speed", "symmetry",
            "stride_length_m", "gait_velocity_mps", "step_width_m",
            "stance_percentage", "left_knee_rom", "right_knee_rom",
            "step_time_cv", "step_length_cv", "foot_clearance_cm",
            "gait_rehab_score", "trunk_sway_deg", "double_support_ratio",
            "step_count",
        ]

        if seconds is not None:
            # 用该会话最新数据的时间戳作为参考点（而非 time.time()），
            # 这样会话关闭后再分析也能正确找到"最后 N 秒"的数据
            c.execute(
                "SELECT MAX(timestamp) FROM gait_metrics WHERE session_id=?",
                (sid,))
            row = c.fetchone()
            if row and row[0]:
                cutoff = row[0] - seconds
            else:
                cutoff = time.time() - seconds
            c.execute(
                f"SELECT {', '.join(cols)} FROM gait_metrics "
                "WHERE session_id=? AND timestamp >= ? ORDER BY timestamp",
                (sid, cutoff))
        else:
            c.execute(
                f"SELECT {', '.join(cols)} FROM gait_metrics "
                "WHERE session_id=? ORDER BY timestamp", (sid,))

        rows = c.fetchall()
        if not rows:
            return None

        arr = np.array(rows, dtype=np.float64)
        stats = {}
        for i, col in enumerate(cols):
            vals = arr[:, i]
            nonzero = vals[vals > 0] if col != "step_count" else vals
            if len(nonzero) == 0:
                nonzero = vals
            stats[col] = {
                "mean": round(float(np.mean(nonzero)), 2),
                "std": round(float(np.std(nonzero)), 2),
                "min": round(float(np.min(nonzero)), 2),
                "max": round(float(np.max(nonzero)), 2),
                "latest": round(float(vals[-1]), 2),
                "count": len(nonzero),
            }
        stats["total_records"] = len(rows)
        return stats

    def get_monthly_gait_trend(self, days=30):
        """获取过去 N 天的每日步态均值趋势（跨所有会话）

        Args:
            days: 统计天数

        Returns:
            dict: {dates: [...], metrics: {col: [daily_avg, ...]}}
        """
        cutoff = time.time() - days * 86400
        c = self.conn.cursor()

        cols = [
            "gait_velocity_mps", "symmetry", "cadence",
            "gait_rehab_score", "left_knee_rom", "right_knee_rom",
            "step_length_cv", "foot_clearance_cm", "double_support_ratio",
            "trunk_sway_deg",
        ]

        c.execute(
            f"SELECT timestamp, {', '.join(cols)} FROM gait_metrics "
            "WHERE timestamp >= ? ORDER BY timestamp", (cutoff,))
        rows = c.fetchall()

        if not rows:
            return None

        # 按天分组
        daily = defaultdict(lambda: defaultdict(list))
        for row in rows:
            ts = row[0]
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            for i, col in enumerate(cols):
                val = row[i + 1]
                if val is not None and val > 0:
                    daily[day][col].append(float(val))

        dates = sorted(daily.keys())
        trend = {"dates": dates, "metrics": {}}
        for col in cols:
            trend["metrics"][col] = [
                round(float(np.mean(daily[d][col])), 2) if daily[d][col] else 0.0
                for d in dates
            ]

        return trend

    def clear_gait_data(self, session_id=None):
        """清空步态数据

        Args:
            session_id: 指定会话ID，None=清空所有步态数据
        """
        c = self.conn.cursor()
        if session_id is not None:
            c.execute("DELETE FROM gait_metrics WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM frame_snapshots WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM fall_alerts WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM emotion_log WHERE session_id=?", (session_id,))
            logger.info("已清空会话 #%d 的步态/帧/跌倒/情绪数据", session_id)
        else:
            c.execute("DELETE FROM gait_metrics")
            c.execute("DELETE FROM frame_snapshots")
            c.execute("DELETE FROM fall_alerts")
            c.execute("DELETE FROM emotion_log")
            logger.info("已清空全部步态/帧/跌倒/情绪数据")
        self.conn.commit()

    def flush(self, timeout=3.0):
        """等待异步队列清空"""
        if not self.async_mode:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._write_queue.empty():
                break
            time.sleep(0.05)

    def close(self):
        if self.async_mode:
            self.flush()
            self._running = False
            try:
                self._write_queue.put_nowait(None)
            except queue.Full:
                pass
            self._write_thread.join(timeout=3.0)
        if self.conn:
            self.conn.close()
