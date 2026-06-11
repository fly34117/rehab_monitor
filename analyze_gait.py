"""步态数据统计分析与可视化

用法:"""
import sys
# Windows GBK 命令行 UTF-8 支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
__doc__ += """
    python analyze_gait.py                        # 分析最近 1 分钟
    python analyze_gait.py --last-minute          # 分析最近 1 分钟
    python analyze_gait.py --minutes 5            # 分析最近 5 分钟
    python analyze_gait.py --month                # 分析近 30 天趋势
    python analyze_gait.py --month --days 90      # 分析近 90 天趋势
    python analyze_gait.py --session 3            # 分析指定会话
    python analyze_gait.py --expert               # 专家知识库分析 (API + 文献)
    python analyze_gait.py --expert --month       # 专家月度趋势分析

输出:
    analysis/gait_1min_report.png                 # 1 分钟综合图表
    analysis/gait_monthly_trend.png               # 月度趋势图表
    analysis/expert_report_*.txt                  # 专家分析报告
"""
import sqlite3
import json
import os
import sys
import struct
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = "rehab_data.db"
OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 指标中文名映射
METRIC_NAMES_CN = {
    "gait_velocity_mps": "步速 (m/s)",
    "symmetry": "对称性",
    "cadence": "步频 (spm)",
    "gait_rehab_score": "GRS 康复评分",
    "left_knee_rom": "左膝 ROM (°)",
    "right_knee_rom": "右膝 ROM (°)",
    "step_length_cv": "步长 CV (%)",
    "step_time_cv": "步时 CV (%)",
    "foot_clearance_cm": "足廓清 (cm)",
    "double_support_ratio": "双支撑比例",
    "trunk_sway_deg": "躯干侧倾 (°)",
    "stride_length_m": "跨步长 (m)",
    "step_width_m": "步宽 (m)",
    "stance_percentage": "支撑相 (%)",
    "speed": "速度 (px/s)",
    "step_count": "步数",
}

# 临床参考值 (正常范围)
CLINICAL_REF = {
    "gait_velocity_mps": (1.0, 0.6, ">1.0 正常, <0.6 需关注"),
    "symmetry": (0.9, 0.7, ">0.9 正常, <0.7 需关注"),
    "gait_rehab_score": (70, 40, ">70 良好, <40 需关注"),
    "double_support_ratio": (0.25, 0.40, "20-30% 正常, >40% 需关注"),
    "trunk_sway_deg": (5, 15, "<5° 正常, >15° 需关注"),
    "step_length_cv": (5, 15, "<5% 正常, >15% 需关注"),
    "foot_clearance_cm": (2.0, 0.5, "1.5-2.5 正常, <0.5 需关注"),
}


def safe_float(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, bytes):
        try:
            return struct.unpack("d", val)[0]
        except Exception:
            return 0.0
    return float(val)


def print_stats(stats, title="步态统计摘要"):
    """打印统计结果到控制台"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    key_metrics = [
        "gait_rehab_score", "gait_velocity_mps", "symmetry", "cadence",
        "left_knee_rom", "right_knee_rom", "step_length_cv", "step_time_cv",
        "foot_clearance_cm", "double_support_ratio", "trunk_sway_deg",
    ]

    if "total_records" in stats:
        print(f"  总记录数: {stats['total_records']}")

    for col in key_metrics:
        if col in stats:
            s = stats[col]
            name = METRIC_NAMES_CN.get(col, col)
            print(f"  {name:20s}  "
                  f"均值={s['mean']:8.2f}  "
                  f"最新={s['latest']:8.2f}  "
                  f"范围=[{s['min']:.2f}, {s['max']:.2f}]  "
                  f"σ={s['std']:.2f}")
            # 临床参考
            if col in CLINICAL_REF:
                ref_ok, ref_warn, hint = CLINICAL_REF[col]
                latest = s['latest']
                status = "[OK]" if latest >= ref_ok else ("[WARN]" if latest >= ref_warn else "[RISK]")
                print(f"  {'':20s}  临床: {status} {hint}")

    print(f"{'='*60}\n")


def plot_last_minute(db, session_id, seconds=60):
    """绘制最近 N 秒的步态综合图表"""
    cur = db.cursor()

    # 用该会话最新数据时间戳作为参考点（而非 time.time()），
    # 确保会话关闭后离线分析也能正确工作
    cur.execute(
        "SELECT MAX(timestamp) FROM gait_metrics WHERE session_id=?",
        (session_id,))
    row = cur.fetchone()
    if row and row[0]:
        cutoff = row[0] - seconds
    else:
        cutoff = time.time() - seconds

    cols = [
        "timestamp", "cadence", "speed", "symmetry",
        "gait_velocity_mps", "gait_rehab_score",
        "left_knee_rom", "right_knee_rom",
        "step_length_cv", "step_time_cv",
        "foot_clearance_cm", "double_support_ratio",
        "trunk_sway_deg",
    ]

    cur.execute(
        f"SELECT {', '.join(cols)} FROM gait_metrics "
        "WHERE session_id=? AND timestamp >= ? ORDER BY timestamp",
        (session_id, cutoff))
    rows = cur.fetchall()

    if not rows:
        print(f"[警告] 最近 {seconds} 秒内无步态数据 (session #{session_id})")
        return

    times = [datetime.fromtimestamp(r[0]) for r in rows]
    data = {}
    for i, col in enumerate(cols[1:], 1):
        data[col] = [safe_float(r[i]) for r in rows]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"步态实时分析 — 最近 {seconds} 秒 (Session #{session_id})",
                 fontsize=16, fontweight="bold")

    # ---- 面板1: GRS 康复评分 ----
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(times, data["gait_rehab_score"], "b-", linewidth=1.5, label="GRS 评分")
    ax1.axhline(y=70, color="green", linestyle="--", alpha=0.6, label="良好线 (70)")
    ax1.axhline(y=40, color="red", linestyle="--", alpha=0.6, label="警戒线 (40)")
    ax1.fill_between(times, 0, data["gait_rehab_score"], alpha=0.15, color="blue")
    ax1.set_ylabel("GRS 评分 (0-100)")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("综合康复评分 (GRS)")

    # ---- 面板2: 步速 + 步频 (双Y轴) ----
    ax2 = fig.add_subplot(3, 2, 2)
    line1 = ax2.plot(times, data["gait_velocity_mps"], "purple", linewidth=1.2, label="步速")
    ax2.set_ylabel("步速 (m/s)", color="purple")
    ax2.tick_params(axis="y", colors="purple")
    ax2.axhline(y=1.0, color="purple", linestyle="--", alpha=0.4)
    ax2.axhline(y=0.6, color="red", linestyle="--", alpha=0.4)
    ax2b = ax2.twinx()
    line2 = ax2b.plot(times, data["cadence"], "brown", linewidth=1.0, alpha=0.7, label="步频")
    ax2b.set_ylabel("步频 (spm)", color="brown")
    ax2b.tick_params(axis="y", colors="brown")
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("步速 & 步频")

    # ---- 面板3: 对称性 ----
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(times, data["symmetry"], "green", linewidth=1.5, label="对称性")
    ax3.axhline(y=0.9, color="green", linestyle="--", alpha=0.6, label="良好 (0.9)")
    ax3.axhline(y=0.7, color="orange", linestyle="--", alpha=0.6, label="警戒 (0.7)")
    ax3.fill_between(times, 0, data["symmetry"], alpha=0.1, color="green")
    ax3.set_ylabel("对称性 (0-1)")
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("步态对称性")

    # ---- 面板4: 膝关节 ROM ----
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(times, data["left_knee_rom"], "blue", linewidth=1.2, alpha=0.8, label="左膝")
    ax4.plot(times, data["right_knee_rom"], "red", linewidth=1.2, alpha=0.8, label="右膝")
    ax4.fill_between(times, 0, data["left_knee_rom"], alpha=0.08, color="blue")
    ax4.fill_between(times, 0, data["right_knee_rom"], alpha=0.08, color="red")
    ax4.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="正常下限 (50°)")
    ax4.set_ylabel("ROM (°)")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("膝关节活动范围 (ROM)")

    # ---- 面板5: 变异性 CV ----
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(times, data["step_length_cv"], "orange", linewidth=1.2, label="步长 CV")
    ax5.plot(times, data["step_time_cv"], "teal", linewidth=1.2, alpha=0.7, label="步时 CV")
    ax5.axhline(y=5, color="green", linestyle="--", alpha=0.5, label="良好线 (5%)")
    ax5.axhline(y=15, color="red", linestyle="--", alpha=0.5, label="警戒线 (15%)")
    ax5.set_ylabel("CV (%)")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_title("步态变异性 (CV)")

    # ---- 面板6: 足廓清 + 躯干侧倾 + 双支撑 ----
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(times, data["foot_clearance_cm"], "blue", linewidth=1.2, label="足廓清")
    ax6.set_ylabel("足廓清 (cm)", color="blue")
    ax6.tick_params(axis="y", colors="blue")
    ax6.axhline(y=1.5, color="blue", linestyle="--", alpha=0.4)
    ax6b = ax6.twinx()
    ax6b.plot(times, data["trunk_sway_deg"], "orange", linewidth=1.0, alpha=0.7, label="躯干侧倾")
    ax6b.set_ylabel("角度 (°)", color="orange")
    ax6b.tick_params(axis="y", colors="orange")
    ax6b.axhline(y=5, color="orange", linestyle="--", alpha=0.4)
    lines6_1, labels6_1 = ax6.get_legend_handles_labels()
    lines6_2, labels6_2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines6_1 + lines6_2, labels6_1 + labels6_2, loc="upper right", fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_title("足廓清 & 躯干侧倾")

    # 时间格式化
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/gait_1min_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[报告] 已保存: {out_path}  ({len(rows)} 条记录)")


def plot_monthly_trend(db, days=30):
    """绘制近 N 天的步态每日趋势"""
    cutoff = time.time() - days * 86400
    cur = db.cursor()

    cols = [
        "timestamp", "gait_velocity_mps", "symmetry", "cadence",
        "gait_rehab_score", "left_knee_rom", "right_knee_rom",
        "step_length_cv", "foot_clearance_cm", "double_support_ratio",
        "trunk_sway_deg",
    ]

    cur.execute(
        f"SELECT {', '.join(cols)} FROM gait_metrics "
        "WHERE timestamp >= ? ORDER BY timestamp", (cutoff,))
    rows = cur.fetchall()

    if not rows:
        print(f"[警告] 近 {days} 天内无步态数据")
        return

    # 按天分组
    daily = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ts = row[0]
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        for i, col in enumerate(cols[1:], 1):
            val = safe_float(row[i])
            if val > 0:
                daily[day][col].append(val)

    dates_str = sorted(daily.keys())
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates_str]

    def day_avg(col):
        return [np.mean(daily[d][col]) if daily[d][col] else 0.0 for d in dates_str]

    def day_std(col):
        return [np.std(daily[d][col]) if len(daily[d][col]) > 1 else 0.0 for d in dates_str]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"步态月度趋势分析 — 近 {days} 天",
                 fontsize=16, fontweight="bold")

    # ---- 面板1: GRS 康复评分趋势 ----
    ax1 = fig.add_subplot(3, 2, 1)
    scores = day_avg("gait_rehab_score")
    errs = day_std("gait_rehab_score")
    ax1.fill_between(dates, [max(0, s - e) for s, e in zip(scores, errs)],
                     [min(100, s + e) for s, e in zip(scores, errs)],
                     alpha=0.2, color="blue")
    ax1.plot(dates, scores, "b-o", linewidth=2, markersize=6, label="GRS 评分")
    ax1.axhline(y=70, color="green", linestyle="--", alpha=0.6, label="良好 (70)")
    ax1.axhline(y=40, color="red", linestyle="--", alpha=0.6, label="警戒 (40)")
    ax1.set_ylabel("GRS 评分 (0-100)")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("GRS 康复评分 (日均值 ± σ)")

    # 添加趋势箭头
    if len(scores) >= 2:
        delta = scores[-1] - scores[0]
        direction = "↑ 改善" if delta > 2 else ("↓ 下降" if delta < -2 else "→ 持平")
        color = "green" if delta > 2 else ("red" if delta < -2 else "gray")
        ax1.annotate(f"{direction} ({delta:+.1f})", xy=(dates[-1], scores[-1]),
                     xytext=(30, 15), textcoords="offset points",
                     fontsize=10, color=color, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=color))

    # ---- 面板2: 步速趋势 ----
    ax2 = fig.add_subplot(3, 2, 2)
    vel = day_avg("gait_velocity_mps")
    ax2.fill_between(dates, [max(0, v - e) for v, e in zip(vel, day_std("gait_velocity_mps"))],
                     [v + e for v, e in zip(vel, day_std("gait_velocity_mps"))],
                     alpha=0.2, color="purple")
    ax2.plot(dates, vel, "s-", color="purple", linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.6, label="正常 (>1.0)")
    ax2.axhline(y=0.6, color="red", linestyle="--", alpha=0.6, label="警戒 (<0.6)")
    ax2.set_ylabel("步速 (m/s)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("步速趋势 (日均值 ± σ)")

    # ---- 面板3: 对称性趋势 ----
    ax3 = fig.add_subplot(3, 2, 3)
    sym = day_avg("symmetry")
    ax3.fill_between(dates, [max(0, s - e) for s, e in zip(sym, day_std("symmetry"))],
                     [min(1, s + e) for s, e in zip(sym, day_std("symmetry"))],
                     alpha=0.2, color="green")
    ax3.plot(dates, sym, "o-", color="green", linewidth=2, markersize=6)
    ax3.axhline(y=0.9, color="green", linestyle="--", alpha=0.5)
    ax3.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5)
    ax3.set_ylabel("对称性 (0-1)")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("步态对称性趋势 (日均值 ± σ)")

    # ---- 面板4: 膝关节 ROM 对比 ----
    ax4 = fig.add_subplot(3, 2, 4)
    l_rom = day_avg("left_knee_rom")
    r_rom = day_avg("right_knee_rom")
    x = np.arange(len(dates_str))
    w = 0.35
    ax4.bar(x - w/2, l_rom, w, label="左膝", color="steelblue", alpha=0.8)
    ax4.bar(x + w/2, r_rom, w, label="右膝", color="coral", alpha=0.8)
    ax4.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="正常下限 (50°)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.strftime("%m-%d") for d in dates], rotation=45, ha="right", fontsize=7)
    ax4.set_ylabel("ROM (°)")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_title("膝关节 ROM 日均值对比")

    # ---- 面板5: 变异性 + 足廓清趋势 ----
    ax5 = fig.add_subplot(3, 2, 5)
    cv = day_avg("step_length_cv")
    ax5.plot(dates, cv, "D-", color="orange", linewidth=1.5, markersize=5, label="步长 CV")
    ax5.axhline(y=5, color="green", linestyle="--", alpha=0.5)
    ax5.axhline(y=15, color="red", linestyle="--", alpha=0.5)
    ax5.set_ylabel("步长 CV (%)", color="orange")
    ax5.tick_params(axis="y", colors="orange")
    ax5b = ax5.twinx()
    fc = day_avg("foot_clearance_cm")
    ax5b.plot(dates, fc, "v-", color="teal", linewidth=1.5, markersize=5, label="足廓清")
    ax5b.set_ylabel("足廓清 (cm)", color="teal")
    ax5b.tick_params(axis="y", colors="teal")
    ax5b.axhline(y=1.5, color="teal", linestyle="--", alpha=0.4)
    lines5_1, labels5_1 = ax5.get_legend_handles_labels()
    lines5_2, labels5_2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines5_1 + lines5_2, labels5_1 + labels5_2, loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_title("步长变异性 & 足廓清趋势")

    # ---- 面板6: 综合统计表 ----
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis("off")

    # 计算汇总统计
    summary_metrics = [
        ("gait_rehab_score", "GRS 评分"),
        ("gait_velocity_mps", "步速 (m/s)"),
        ("symmetry", "对称性"),
        ("cadence", "步频 (spm)"),
        ("left_knee_rom", "左膝 ROM (°)"),
        ("right_knee_rom", "右膝 ROM (°)"),
        ("step_length_cv", "步长 CV (%)"),
        ("foot_clearance_cm", "足廓清 (cm)"),
    ]

    table_data = []
    for col, name in summary_metrics:
        vals = day_avg(col)
        if vals and any(v > 0 for v in vals):
            overall_mean = np.mean([v for v in vals if v > 0])
            trend_val = vals[-1] - vals[0] if len(vals) >= 2 else 0
            trend_arrow = "↑" if trend_val > 0.5 else ("↓" if trend_val < -0.5 else "→")
            table_data.append([name, f"{overall_mean:.1f}", f"{trend_val:+.1f}", trend_arrow])

    if table_data:
        table = ax6.table(
            cellText=table_data,
            colLabels=["指标", "均值", "变化", "趋势"],
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.25, 0.25, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        # 表头样式
        for j in range(4):
            table[0, j].set_facecolor("#40466e")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # 趋势列着色
        for i in range(len(table_data)):
            trend = table_data[i][3]
            color = "#c8e6c9" if trend == "↑" else ("#ffcdd2" if trend == "↓" else "#f5f5f5")
            for j in range(4):
                table[i + 1, j].set_facecolor(color)

    ax6.set_title("月度变化汇总", fontweight="bold", fontsize=12)

    # 时间格式化
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        if ax != ax4:  # ax4 uses bar chart with string labels
            ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
            ax.xaxis.set_major_locator(DayLocator(interval=max(1, days // 10)))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/gait_monthly_trend.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[趋势报告] 已保存: {out_path}  ({len(rows)} 条记录, {len(dates_str)} 天)")

    # 控制台输出每日汇总
    print(f"\n{'='*80}")
    print(f"  每日步态汇总 — 近 {days} 天")
    print(f"{'='*80}")
    header = f"  {'日期':12s} {'GRS':>6s} {'步速':>7s} {'对称性':>7s} {'步频':>7s} {'左ROM':>6s} {'右ROM':>6s}"
    print(header)
    print(f"  {'-'*60}")
    for i, d in enumerate(dates_str):
        s = day_avg("gait_rehab_score")[i]
        v = day_avg("gait_velocity_mps")[i]
        sy = day_avg("symmetry")[i]
        c = day_avg("cadence")[i]
        lr = day_avg("left_knee_rom")[i]
        rr = day_avg("right_knee_rom")[i]
        print(f"  {d:12s} {s:6.1f} {v:7.2f} {sy:7.3f} {c:7.1f} {lr:6.1f} {rr:6.1f}")
    print(f"{'='*80}\n")


def _run_expert_analysis(db, stats, trend_data):
    """运行专家知识库分析"""
    print(f"\n{'='*60}")
    print(f"  启动专家知识库分析...")
    print(f"  加载 13 篇权威论文 + DeepSeek API 生成报告")
    print(f"{'='*60}\n")

    from rehab_monitor.expert_report import ExpertReportGenerator

    expert = ExpertReportGenerator()
    print(f"  知识库已加载: {len(expert.index)} 篇论文")

    report_json, cited_papers, raw, error = expert.generate(stats, trend_data)

    if error:
        print(f"\n[专家分析失败] {error}")
        return None, None

    # 格式化并打印报告
    formatted = expert.format_report(report_json, cited_papers)
    print(formatted)

    # 保存报告
    out_path = f"{OUT_DIR}/expert_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted)
        # 也保存原始 JSON
        json_path = out_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    print(f"\n[报告已保存] {out_path}")
    print(f"[JSON已保存] {json_path}")

    return report_json, formatted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="步态数据统计分析与可视化")
    parser.add_argument("--last-minute", action="store_true", help="分析最近 1 分钟步态数据")
    parser.add_argument("--minutes", type=int, default=1, help="分析最近 N 分钟 (默认 1)")
    parser.add_argument("--month", action="store_true", help="分析近 30 天步态趋势")
    parser.add_argument("--days", type=int, default=30, help="月度趋势天数 (默认 30)")
    parser.add_argument("--session", type=int, help="指定会话 ID")
    parser.add_argument("--expert", action="store_true",
                        help="启用专家知识库分析 (调用 DeepSeek API + 权威文献)")
    args = parser.parse_args()

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # 默认行为：分析最近 1 分钟
    if not args.month and not args.last_minute and args.minutes == 1 and args.session is None:
        args.last_minute = True

    # 引入 RehabDatabase 用于统计查询
    from rehab_monitor.data_logger import RehabDatabase
    rdb = RehabDatabase.__new__(RehabDatabase)
    rdb.db_path = DB_PATH
    rdb.conn = db

    # 获取会话 ID
    cur = db.cursor()
    sid = args.session
    if sid is None:
        cur.execute("SELECT MAX(id) FROM sessions")
        sid = cur.fetchone()[0]

    # ---- 收集趋势数据（月度 + 专家分析共用） ----
    trend_data = None
    if args.month or args.expert:
        print(f"[分析] 加载近 {args.days} 天步态趋势...")
        trend_data = rdb.get_monthly_gait_trend(days=args.days)

    # ---- 收集实时统计（1分钟 + 专家分析共用） ----
    gait_stats = None
    if args.last_minute or args.minutes != 1 or args.session is not None or args.expert:
        seconds = args.minutes * 60
        if sid is None:
            print("没有找到任何会话数据。请先运行康复监测系统。")
            db.close()
            return

        rdb.session_id = sid
        print(f"[分析] 加载会话 #{sid} 最近 {args.minutes} 分钟数据...")
        gait_stats = rdb.get_gait_stats(session_id=sid, seconds=seconds)

    # ---- 可视化 + 控制台输出 ----
    if args.month:
        plot_monthly_trend(db, days=args.days)
        if trend_data:
            print(f"  覆盖 {len(trend_data['dates'])} 天的数据")

    if args.last_minute or args.minutes != 1 or args.session is not None:
        if gait_stats:
            print_stats(gait_stats, f"步态统计 — 会话 #{sid} 最近 {args.minutes} 分钟")
            plot_last_minute(db, sid, seconds=seconds)
        else:
            print(f"[警告] 会话 #{sid} 最近 {args.minutes} 分钟内无数据")

    # ---- 专家知识库分析 ----
    if args.expert:
        if gait_stats is None:
            print("[专家分析] 无步态统计数据，跳过")
        else:
            _run_expert_analysis(db, gait_stats, trend_data)

    db.close()
    print("分析完成。")


if __name__ == "__main__":
    main()
