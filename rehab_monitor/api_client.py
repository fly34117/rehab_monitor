"""DeepSeek API 周期康复报告生成"""
import json
import time
import requests

from .config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL, DEEPSEEK_TIMEOUT,
)


SYSTEM_PROMPT = """你是一位资深康复医学分析师。根据患者最近30秒的步态数据和姿态快照，生成一份简洁的临床摘要。

## 输出格式 (严格JSON)
{
  "summary": "1-2句话概括当前状态",
  "gait_assessment": "步态评价 (步速/对称性/步长/双支撑)",
  "fall_risk": "none/low/medium/high",
  "recommendation": "简短建议 (如继续训练/休息/调整步态/需要关注)",
  "key_metrics": {
    "avg_speed": "? m/s",
    "symmetry": "?%",
    "grs_score": "?",
    "notable": "值得注意的异常"
  }
}

## 评分参考
- 步速: >1.0 m/s 正常, <0.6 缓慢
- 对称性: >0.9 正常, <0.7 异常
- GRS评分: >70 良好, 40-70 一般, <40 需关注
- 双支撑比: 20-30% 正常, >40% 步态不稳
- 躯干侧倾: <5° 正常, >15° 异常

请基于实际数据生成报告，数据不足时标注"数据不足"。只返回JSON，不要其他文字。"""


def generate_report(recent_data, api_key=None):
    """生成周期康复报告

    Args:
        recent_data: data_logger.get_recent_data() 的返回值
        api_key: DeepSeek API key (默认从 config 读取)

    Returns:
        (report_text, summary_json, error_message)
    """
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        return None, None, "DEEPSEEK_API_KEY 未设置"

    # 构建数据摘要
    data_text = _build_data_prompt(recent_data)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请分析以下康复监测数据并生成报告:\n\n{data_text}"},
    ]

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 600,
            },
            timeout=DEEPSEEK_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        msg = body["choices"][0]["message"]
        from .llm_client import (
            extract_final_answer,
            format_reasoning_response,
        )
        content = format_reasoning_response(
            msg.get("reasoning_content") or "",
            msg.get("content") or "",
        )

        # 解析 JSON
        report_json = _parse_response(extract_final_answer(content))
        return content, report_json, None

    except requests.exceptions.Timeout:
        return None, None, f"API 超时 (>{DEEPSEEK_TIMEOUT}s)"
    except requests.exceptions.ConnectionError:
        return None, None, "API 连接失败 (检查网络)"
    except Exception as e:
        return None, None, f"API 错误: {str(e)[:200]}"


def _build_data_prompt(data):
    """将数据库数据转为文本摘要"""
    lines = []
    gs = data.get("gait_summary", {})
    lines.append(f"时间窗口: {data['duration']}秒")
    lines.append(f"姿态快照: {data['snapshot_count']}帧")
    lines.append(f"步态记录: {data['gait_records']}条")
    lines.append(f"跌倒事件: {data['fall_events']}次")
    lines.append(f"表情记录: {data['emotion_records']}条")
    lines.append("")

    if gs:
        lines.append("## 临床指标 (最新值)")
        metric_labels = {
            "gait_velocity_mps": ("步速", "m/s"),
            "stride_length_m": ("步长", "m"),
            "step_width_m": ("步宽", "m"),
            "symmetry": ("对称性", ""),
            "cadence_spm": ("步频", "spm"),
            "stance_percentage": ("支撑相占比", "%"),
            "left_knee_rom": ("左膝ROM", "°"),
            "right_knee_rom": ("右膝ROM", "°"),
            "step_time_cv": ("步时CV", "%"),
            "step_length_cv": ("步长CV", "%"),
            "foot_clearance_cm": ("足廓清", "cm"),
            "gait_rehab_score": ("GRS评分", "/100"),
            "trunk_sway_deg": ("躯干侧倾", "°"),
            "double_support_ratio": ("双支撑比", ""),
        }
        for key, (label, unit) in metric_labels.items():
            val = gs.get(key)
            symmetry_val = data.get("gait", [])
            if val is not None and val != 0:
                if key == "symmetry" and symmetry_val:
                    val = round(symmetry_val[-1][4], 2) if len(symmetry_val[0]) > 4 else val
                lines.append(f"  {label}: {val}{unit}")
    else:
        lines.append("(步态数据不足)")

    # 跌倒统计
    if data["fall_events"] > 0:
        lines.append(f"\n⚠ 检测到 {data['fall_events']} 次跌倒事件")

    # 情绪
    emotions = data.get("emotions", [])
    if emotions:
        from collections import Counter
        emo_counts = Counter(e[1] for e in emotions if e[1])
        if emo_counts:
            top_emo = emo_counts.most_common(2)
            lines.append(f"\n情绪分布: {', '.join(f'{k}({v}次)' for k, v in top_emo)}")

    return "\n".join(lines)


def _parse_response(content):
    """解析 API 返回的 JSON"""
    try:
        # 尝试直接解析
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # 尝试提取 ```json ... ``` 代码块
    if "```json" in content:
        try:
            start = content.index("```json") + 7
            end = content.index("```", start)
            return json.loads(content[start:end].strip())
        except (ValueError, json.JSONDecodeError):
            pass
    # 提取 { ... }
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        return json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"raw": content}
