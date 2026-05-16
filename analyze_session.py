"""步态康复数据分析与可视化

用法:
    python analyze_session.py              # 分析最近一次会话
    python analyze_session.py --session 5  # 分析指定会话
    python analyze_session.py --all        # 分析所有会话趋势

输出:
    analysis/session_<id>_report.png       # 单次会话综合图表
    analysis/trend_report.png              # 跨会话趋势
"""
import sqlite3
import json
import os
import sys
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = "rehab_data.db"
OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def safe_float(val):
    """修复二进制存储的 float 值"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, bytes):
        try:
            return struct.unpack("d", val)[0]
        except:
            return 0.0
    return float(val)


def load_session(db, session_id):
    """加载单次会话的所有数据"""
    cur = db.cursor()

    # 步态数据
    cur.execute(
        "SELECT timestamp, frame_number, step_count, cadence, speed, symmetry "
        "FROM gait_metrics WHERE session_id=? ORDER BY timestamp",
        (session_id,))
    gait_rows = cur.fetchall()
    gait_times = [datetime.fromtimestamp(r[0]) for r in gait_rows]
    gait_data = {
        "times": gait_times,
        "steps": [r[2] for r in gait_rows],
        "cadence": [safe_float(r[3]) for r in gait_rows],
        "speed": [safe_float(r[4]) for r in gait_rows],
        "symmetry": [safe_float(r[5]) for r in gait_rows],
    }

    # 关节角度数据
    cur.execute(
        "SELECT timestamp, joint_angles_json FROM frame_snapshots "
        "WHERE session_id=? AND joint_angles_json IS NOT NULL ORDER BY timestamp",
        (session_id,))
    angle_rows = cur.fetchall()
    angle_times = []
    left_knees, right_knees = [], []
    for r in angle_rows:
        try:
            angles = json.loads(r[1])
            if angles.get("left_knee") is not None:
                angle_times.append(datetime.fromtimestamp(r[0]))
                left_knees.append(angles["left_knee"])
                right_knees.append(angles["right_knee"])
        except:
            pass

    # 情绪数据
    cur.execute(
        "SELECT timestamp, emotion_label FROM emotion_log "
        "WHERE session_id=? ORDER BY timestamp", (session_id,))
    emotion_rows = cur.fetchall()
    emotion_times = [datetime.fromtimestamp(r[0]) for r in emotion_rows]
    emotion_labels = [r[1] for r in emotion_rows]

    # 跌倒事件
    cur.execute(
        "SELECT timestamp FROM fall_alerts WHERE session_id=? ORDER BY timestamp",
        (session_id,))
    fall_rows = cur.fetchall()
    fall_times = [datetime.fromtimestamp(r[0]) for r in fall_rows]

    return gait_data, (angle_times, left_knees, right_knees), \
           (emotion_times, emotion_labels), fall_times


def plot_session(session_id, gait, angles, emotions, falls, output_path):
    """绘制单次会话综合图表"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Rehab Session #{session_id} - Gait Analysis Report",
                 fontsize=16, fontweight="bold")

    gait_t, angle_t, emo_t = gait["times"], angles[0], emotions[0]
    fall_ts = falls

    # ---- 面板1: 膝关节角度 ----
    ax = axes[0]
    if angle_t:
        ax.plot(angle_t, angles[1], "b-", alpha=0.7, linewidth=0.8, label="Left Knee")
        ax.plot(angle_t, angles[2], "r-", alpha=0.7, linewidth=0.8, label="Right Knee")
        ax.axhline(y=170, color="gray", linestyle="--", alpha=0.5, label="Normal (170)")
    ax.set_ylabel("Knee Angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 面板2: 步态对称性 ----
    ax = axes[1]
    if gait_t:
        ax.plot(gait_t, gait["symmetry"], "g-", linewidth=1.0, label="Symmetry")
        ax.axhline(y=0.85, color="orange", linestyle="--", alpha=0.5, label="Threshold (0.85)")
    ax.set_ylabel("Symmetry Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 面板3: 步速 + 步频 ----
    ax = axes[2]
    if gait_t:
        ax.plot(gait_t, gait["speed"], "purple", linewidth=1.0, label="Speed (px/s)")
        ax2 = ax.twinx()
        ax2.plot(gait_t, gait["cadence"], "brown", linewidth=1.0, alpha=0.6, label="Cadence (steps/min)")
        ax2.set_ylabel("Cadence (steps/min)", color="brown")
        ax2.tick_params(axis="y", colors="brown")
    ax.set_ylabel("Speed (px/s)", color="purple")
    ax.tick_params(axis="y", colors="purple")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 面板4: 情绪 + 跌倒标记 ----
    ax = axes[3]
    emotion_map = {"happy": 6, "surprise": 5, "neutral": 4, "sad": 3,
                   "fear": 2, "angry": 1, "disgust": 0}
    if emo_t:
        emo_vals = [emotion_map.get(e, 4) for e in emotions[1]]
        ax.scatter(emo_t, emo_vals, c=emo_vals, cmap="RdYlGn", s=15, alpha=0.6, label="Emotion")
    # 跌倒竖线
    for ft in fall_ts:
        ax.axvline(x=ft, color="red", linestyle="-", alpha=0.3, linewidth=2)
    if fall_ts:
        ax.axvline(x=fall_ts[0], color="red", linestyle="-", alpha=0.3, linewidth=2,
                   label=f"Fall Alert ({len(fall_ts)})")
    ax.set_ylabel("Emotion Level")
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels(["Disgust", "Fear", "Neutral", "Happy"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 时间格式化
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[报告] 已保存: {output_path}")


def plot_trend(all_sessions, output_path):
    """跨会话趋势对比"""
    if len(all_sessions) < 2:
        print("[趋势] 需要至少 2 个会话才能生成趋势图表")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Cross-Session Gait Trend", fontsize=14, fontweight="bold")

    sids = [s["id"] for s in all_sessions]
    labels = [f"S{s['id']}" for s in all_sessions]

    # 平均对称性
    ax = axes[0, 0]
    syms = [np.mean(s["symmetry"]) for s in all_sessions if s["symmetry"]]
    ax.bar(labels[:len(syms)], syms, color="steelblue")
    ax.axhline(y=0.85, color="orange", linestyle="--")
    ax.set_title("Avg Symmetry")
    ax.set_ylim(0, 1.05)

    # 平均步速
    ax = axes[0, 1]
    speeds = [np.mean(s["speed"]) for s in all_sessions if s["speed"]]
    ax.bar(labels[:len(speeds)], speeds, color="purple")
    ax.set_title("Avg Speed (px/s)")

    # 跌倒次数
    ax = axes[1, 0]
    falls = [s.get("fall_count", 0) for s in all_sessions]
    ax.bar(labels, falls, color="red" if any(f > 0 for f in falls) else "gray")
    ax.set_title("Fall Events")

    # 主要情绪
    ax = axes[1, 1]
    emotions = [s.get("dominant_emotion", "neutral") for s in all_sessions]
    emo_counts = defaultdict(int)
    for e in emotions: emo_counts[e] += 1
    ax.pie(emo_counts.values(), labels=emo_counts.keys(), autopct="%1.1f%%",
           colors=plt.cm.Pastel1.colors)
    ax.set_title("Dominant Emotion")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[趋势] 已保存: {output_path}")


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    cur = db.cursor()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, help="分析指定会话 ID")
    parser.add_argument("--all", action="store_true", help="分析所有会话趋势")
    args = parser.parse_args()

    if args.all:
        # 收集所有会话摘要
        cur.execute("SELECT DISTINCT session_id FROM gait_metrics ORDER BY session_id")
        all_sids = [r[0] for r in cur.fetchall()]
        all_summaries = []
        for sid in all_sids:
            cur.execute(
                "SELECT AVG(symmetry), AVG(speed), COUNT(*) "
                "FROM gait_metrics WHERE session_id=? AND speed > 0", (sid,))
            r = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM fall_alerts WHERE session_id=?", (sid,))
            fc = cur.fetchone()[0]
            cur.execute(
                "SELECT emotion_label FROM emotion_log WHERE session_id=? "
                "GROUP BY emotion_label ORDER BY COUNT(*) DESC LIMIT 1", (sid,))
            er = cur.fetchone()
            all_summaries.append({
                "id": sid,
                "symmetry": [r[0]] if r[0] else [],
                "speed": [r[1]] if r[1] else [],
                "fall_count": fc,
                "dominant_emotion": er[0] if er else "unknown",
            })
        plot_trend(all_summaries, f"{OUT_DIR}/trend_report.png")
    else:
        sid = args.session
        if sid is None:
            cur.execute("SELECT MAX(id) FROM sessions")
            sid = cur.fetchone()[0]
            if sid is None:
                print("没有找到任何会话数据")
                return
        print(f"[分析] 加载会话 #{sid}...")
        gait, angles, emotions, falls = load_session(db, sid)
        plot_session(sid, gait, angles, emotions, falls,
                     f"{OUT_DIR}/session_{sid}_report.png")

    db.close()


if __name__ == "__main__":
    main()
