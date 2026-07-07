"""专家知识库驱动步态分析 — DeepSeek API + 权威论文文献

将患者的步态数据与专家知识库（13篇权威论文）一起发送给 DeepSeek API，
生成带有文献引用的临床分析报告。

用法:
    from rehab_monitor.expert_report import ExpertReportGenerator
    expert = ExpertReportGenerator()
    report = expert.generate(gait_stats, trend_data)

    # 或直接用 CLI 脚本
    python analyze_gait.py --expert
"""
import json
import os
import re
import time
import requests

from .config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL, DEEPSEEK_TIMEOUT,
)


# ---- 知识库路径 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(PROJECT_ROOT, "expert_kb")
KB_INDEX_PATH = os.path.join(KB_DIR, "papers_index.json")
KB_TEXTS_DIR = os.path.join(KB_DIR, "texts")

# ---- 指标 → 相关论文主题映射 ----
METRIC_TO_TOPICS = {
    "gait_velocity_mps": ["spatiotemporal", "gait_parameters", "stride", "cadence"],
    "symmetry": ["symmetry", "gait_variability"],
    "cadence": ["cadence", "spatiotemporal", "gait_parameters"],
    "gait_rehab_score": ["gait_analysis", "gait_parameters", "biomechanics"],
    "left_knee_rom": ["knee_rom", "kinematics", "biomechanics"],
    "right_knee_rom": ["knee_rom", "kinematics", "biomechanics"],
    "step_length_cv": ["gait_variability", "step_width", "stride"],
    "step_time_cv": ["gait_variability", "cadence"],
    "foot_clearance_cm": ["foot_clearance", "fall_risk"],
    "double_support_ratio": ["double_support", "gait_parameters"],
    "trunk_sway_deg": ["trunk_sway", "fall_risk"],
    "step_width_m": ["step_width", "fall_risk", "gait_parameters"],
    "stride_length_m": ["stride", "spatiotemporal"],
    "stance_percentage": ["double_support", "gait_parameters"],
}


EXPERT_SYSTEM_PROMPT = """你是一位资深康复医学与步态分析专家，同时熟悉以下权威文献数据库中的临床参考标准。

## 你的任务
基于患者的客观步态数据和权威文献标准，生成一份专业的临床步态评估报告。

## 输出格式 (严格JSON)
{
  "summary": "2-3句话概括当前步态状态",
  "gait_assessment": {
    "overall": "总体评价 (正常/轻度异常/中度异常/严重异常)",
    "grs_score": "康复评分解读",
    "key_findings": ["发现1", "发现2", ...]
  },
  "literature_comparison": [
    {
      "metric": "指标名",
      "patient_value": "患者值",
      "normal_range": "文献中的正常范围",
      "paper_reference": "引用文献文件名",
      "assessment": "正常/偏高/偏低/需关注"
    }
  ],
  "fall_risk_assessment": {
    "level": "none/low/medium/high",
    "risk_factors": ["因素1", ...],
    "literature_evidence": "文献依据"
  },
  "recommendations": [
    {
      "action": "建议内容",
      "priority": "high/medium/low",
      "basis": "文献依据或临床推理"
    }
  ],
  "literature_cited": ["文献1文件名", "文献2文件名"]
}

## 评分参考标准 (来自文献数据库)
- 步速 (gait velocity): >1.0 m/s 正常 (Rssler 2024), <0.6 m/s 缓慢，<0.4 严重异常
- 对称性 (symmetry): >0.90 正常, 0.70-0.90 轻度不对称, <0.70 显著不对称 (Gait Variability Review)
- GRS 康复评分: >70 良好, 40-70 一般, <40 需重点关注
- 膝关节 ROM: 行走时 50-65° 正常, <30° 受限 (Gait Kinematic Parameters)
- 足廓清: 1.5-2.5 cm 正常, <0.5 cm 绊倒风险显著增加 (Foot Clearance Fall Risk)
- 双支撑比: 20-30% 正常, >40% 提示步态不稳 (Gait Parameters)
- 躯干侧倾: <5° 正常, >15° 异常 (Trunk Accelerometer Gait)
- 步长CV/步时CV: <5% 正常, 5-15% 轻度变异, >15% 高度变异 (Gait Variability Older Adults)
- 步宽变异: 增加与跌倒风险正相关 (Predicting fall risk through step width variability)

## 重要指示
1. 引用文献时使用文件名（如 "Rssleretal.2024_Spatiotemporalgaitcharacteristics..."）
2. 数据不足时标注"数据不足以做出可靠评估"
3. 只返回 JSON，不要其他文字
4. 你的分析必须基于客观数据 + 文献标准，不可猜测"""


class ExpertReportGenerator:
    """专家知识库驱动的报告生成器"""

    def __init__(self):
        self.index = None
        self._text_cache = {}
        self._paper_relevance = {}
        self._load_kb()

    def _load_kb(self):
        """加载知识库索引"""
        if not os.path.exists(KB_INDEX_PATH):
            self.index = []
            return

        with open(KB_INDEX_PATH, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        # 构建主题 → 论文映射
        for paper in self.index:
            for cat in paper.get("categories", []):
                if cat not in self._paper_relevance:
                    self._paper_relevance[cat] = []
                self._paper_relevance[cat].append(paper["id"])

    def _get_paper_text(self, paper_id):
        """获取论文全文（缓存）"""
        if paper_id in self._text_cache:
            return self._text_cache[paper_id]

        for paper in self.index:
            if paper["id"] == paper_id:
                txt_path = os.path.join(KB_TEXTS_DIR, paper.get("text_file", ""))
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    self._text_cache[paper_id] = text
                    return text
        return ""

    def _select_relevant_papers(self, metrics, max_papers=5):
        """根据指标选择最相关的论文

        优先选择与异常指标最相关的论文
        """
        paper_scores = {}

        for metric_name, metric_value in metrics.items():
            if metric_name not in METRIC_TO_TOPICS:
                continue

            topics = METRIC_TO_TOPICS[metric_name]
            for topic in topics:
                if topic not in self._paper_relevance:
                    continue
                for pid in self._paper_relevance[topic]:
                    paper_scores[pid] = paper_scores.get(pid, 0) + 1

        # 按得分排序，取前 max_papers
        sorted_papers = sorted(paper_scores.items(), key=lambda x: -x[1])
        selected_ids = [pid for pid, _ in sorted_papers[:max_papers]]

        # 确保覆盖主要主题
        if len(selected_ids) < 3 and self.index:
            for paper in self.index:
                if paper["id"] not in selected_ids:
                    selected_ids.append(paper["id"])
                    if len(selected_ids) >= max_papers:
                        break

        return selected_ids

    def _extract_relevant_excerpts(self, paper_id, metrics, max_chars=1500):
        """从论文中提取与指标最相关的段落"""
        text = self._get_paper_text(paper_id)
        if not text:
            return ""

        # 提取摘要（前 2000 字符）
        lines = text.split("\n")
        excerpt_lines = []

        # 方法1: 搜索关键词
        relevant_keywords = set()
        for metric_name in metrics:
            if metric_name in METRIC_TO_TOPICS:
                for topic in METRIC_TO_TOPICS[metric_name]:
                    relevant_keywords.add(topic.replace("_", " "))

        # 额外关键词
        search_terms = [
            "normal", "mean", "average", "threshold", "range",
            "significant", "conclusion", "result", "finding",
            "velocity", "speed", "symmetry", "cadence", "stride",
            "knee", "clearance", "sway", "variability", "fall",
            "m/s", "degree", "cm", "score",
        ]

        # 寻找包含关键数值的行
        for line in lines:
            line_lower = line.lower()
            # 包含关键指标词 + 数值
            has_number = bool(re.search(r'[\d]+\.[\d]+|[\d]+', line))
            has_term = any(t in line_lower for t in search_terms)
            if has_number and has_term and len(line) > 20:
                excerpt_lines.append(line.strip()[:200])

        # 如果提取太少，取摘要
        if len(excerpt_lines) < 5:
            excerpt_lines = [l.strip() for l in lines[:30] if len(l.strip()) > 30]

        excerpt = "\n".join(excerpt_lines[:30])  # 限制行数
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars] + "\n... [截断]"
        return excerpt

    def _build_expert_prompt(self, gait_stats, trend_data=None, conversation_history=None):
        """构建包含文献上下文的专家提示"""

        # 1. 患者数据摘要
        data_lines = []
        data_lines.append("## 患者步态数据\n")

        if gait_stats:
            data_lines.append("### 当前统计 (均值 ± 标准差 | 最新值)")

            key_metrics_order = [
                ("gait_rehab_score", "GRS 康复评分", "/100"),
                ("gait_velocity_mps", "步速", "m/s"),
                ("symmetry", "对称性", ""),
                ("cadence", "步频", "spm"),
                ("stride_length_m", "跨步长", "m"),
                ("step_width_m", "步宽", "m"),
                ("left_knee_rom", "左膝 ROM", "°"),
                ("right_knee_rom", "右膝 ROM", "°"),
                ("step_length_cv", "步长 CV", "%"),
                ("step_time_cv", "步时 CV", "%"),
                ("foot_clearance_cm", "足廓清", "cm"),
                ("double_support_ratio", "双支撑比", ""),
                ("stance_percentage", "支撑相占比", "%"),
                ("trunk_sway_deg", "躯干侧倾", "°"),
            ]

            if "total_records" in gait_stats:
                data_lines.append(f"记录数: {gait_stats['total_records']}条\n")

            for col, name, unit in key_metrics_order:
                s = gait_stats.get(col)
                if s:
                    data_lines.append(
                        f"  {name}: 均值={s['mean']:.2f}{unit} "
                        f"σ={s['std']:.2f} "
                        f"范围=[{s['min']:.2f}, {s['max']:.2f}] "
                        f"最新={s['latest']:.2f}{unit}"
                    )

        # 2. 趋势数据（如果有）
        if trend_data and trend_data.get("dates"):
            dates = trend_data["dates"]
            data_lines.append(f"\n### 趋势数据 ({len(dates)} 天)")

            metrics_trend = trend_data.get("metrics", {})
            for col in ["gait_rehab_score", "gait_velocity_mps", "symmetry", "step_length_cv"]:
                if col in metrics_trend and len(metrics_trend[col]) == len(dates):
                    first = metrics_trend[col][0]
                    last = metrics_trend[col][-1]
                    delta = last - first
                    name = col
                    data_lines.append(f"  {name}: {first:.2f} → {last:.2f} (变化 {delta:+.2f})")

        data_text = "\n".join(data_lines)

        # 3. 选择相关论文并提取摘录
        metrics_dict = {}
        if gait_stats:
            for col in METRIC_TO_TOPICS:
                if col in gait_stats:
                    metrics_dict[col] = gait_stats[col]["latest"]

        selected_ids = self._select_relevant_papers(metrics_dict, max_papers=4)

        literature_text = []
        literature_text.append("## 权威文献参考\n")
        literature_text.append("以下是从专家数据库中选取的最相关论文摘要：\n")

        for pid in selected_ids:
            paper = None
            for p in self.index:
                if p["id"] == pid:
                    paper = p
                    break
            if not paper:
                continue

            literature_text.append(f"### [{pid}] {paper['filename']}")
            literature_text.append(f"分类: {', '.join(paper.get('categories', []))}")
            literature_text.append(f"页数: {paper.get('pages', '?')} 页")

            if paper.get("abstract"):
                literature_text.append(f"\n摘要:\n{paper['abstract'][:400]}")

            if paper.get("keywords"):
                literature_text.append(f"\n关键词: {paper['keywords'][:200]}")

            # 提取相关摘录
            excerpt = self._extract_relevant_excerpts(pid, metrics_dict, max_chars=1200)
            if excerpt:
                literature_text.append(f"\n关键摘录:\n{excerpt}")

            literature_text.append("\n---\n")

        literature_combined = "\n".join(literature_text)

        # 组合最终 prompt
        full_prompt = data_text + "\n\n" + literature_combined
        return full_prompt, [p for p in self.index if p["id"] in selected_ids]

    def generate(self, gait_stats, trend_data=None, api_key=None):
        """生成专家级步态分析报告

        Args:
            gait_stats: data_logger.get_gait_stats() 返回值
            trend_data: data_logger.get_monthly_gait_trend() 返回值 (可选)
            api_key: DeepSeek API key

        Returns:
            (report_json, cited_papers, raw_response, error)
        """
        key = api_key or DEEPSEEK_API_KEY
        if not key:
            return None, [], None, "DEEPSEEK_API_KEY 未设置"

        if gait_stats is None:
            return None, [], None, "步态数据为空，无法生成报告"

        # 构建 expert prompt
        user_prompt, cited_papers = self._build_expert_prompt(gait_stats, trend_data)

        messages = [
            {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
                    "max_tokens": 2000,
                },
                timeout=max(DEEPSEEK_TIMEOUT, 60.0),  # 专家分析需更长超时
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

            if not content:
                return None, cited_papers, None, "API 返回空内容 (可能是推理模型未完成)"

            # 解析 JSON
            report_json = self._parse_response(extract_final_answer(content))
            return report_json, cited_papers, content, None

        except requests.exceptions.Timeout:
            return None, cited_papers, None, f"API 超时 (>{DEEPSEEK_TIMEOUT}s)"
        except requests.exceptions.ConnectionError:
            return None, cited_papers, None, "API 连接失败 (检查网络)"
        except Exception as e:
            return None, cited_papers, None, f"API 错误: {str(e)[:200]}"

    def _parse_response(self, content):
        """解析 API 返回的 JSON"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        if "```json" in content:
            try:
                start = content.index("```json") + 7
                end = content.index("```", start)
                return json.loads(content[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass

        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"raw": content}

    def format_report(self, report_json, cited_papers):
        """格式化报告为可打印的文本"""
        lines = []
        lines.append("=" * 70)
        lines.append("  专家知识库步态分析报告")
        lines.append("  Expert Knowledge-Based Gait Analysis Report")
        lines.append("=" * 70)

        if isinstance(report_json, dict):
            # 摘要
            if report_json.get("summary"):
                lines.append(f"\n【摘要】\n{report_json['summary']}")

            # 步态评估
            ga = report_json.get("gait_assessment", {})
            if ga:
                lines.append(f"\n【步态评估】")
                lines.append(f"  总体评价: {ga.get('overall', 'N/A')}")
                if ga.get("grs_score"):
                    lines.append(f"  GRS 解读: {ga['grs_score']}")
                for f in ga.get("key_findings", []):
                    lines.append(f"  • {f}")

            # 文献对比
            lc = report_json.get("literature_comparison", [])
            if lc:
                lines.append(f"\n【文献对比分析】")
                lines.append(f"  {'指标':12s} {'患者值':10s} {'文献正常范围':18s} {'评估'}")
                lines.append(f"  {'-'*56}")
                for item in lc:
                    lines.append(
                        f"  {item.get('metric', ''):12s} "
                        f"{str(item.get('patient_value', '')):10s} "
                        f"{str(item.get('normal_range', '')):18s} "
                        f"{item.get('assessment', '')}"
                    )

            # 跌倒风险
            fra = report_json.get("fall_risk_assessment", {})
            if fra:
                lines.append(f"\n【跌倒风险评估】")
                lines.append(f"  风险等级: {fra.get('level', 'N/A').upper()}")
                for f in fra.get("risk_factors", []):
                    lines.append(f"  • {f}")
                if fra.get("literature_evidence"):
                    lines.append(f"  文献依据: {fra['literature_evidence']}")

            # 建议
            recs = report_json.get("recommendations", [])
            if recs:
                lines.append(f"\n【临床建议】")
                for r in recs:
                    priority_mark = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}.get(
                        r.get("priority", "low"), "*")
                    lines.append(f"  {priority_mark} [{r.get('priority', 'low').upper()}] {r.get('action', '')}")
                    if r.get("basis"):
                        lines.append(f"     依据: {r['basis']}")

        # 引用文献
        if report_json.get("literature_cited") or cited_papers:
            lines.append(f"\n【引用文献】")
            cited_names = report_json.get("literature_cited", [])
            if cited_names:
                for name in cited_names:
                    lines.append(f"  • {name}")
            else:
                for p in cited_papers:
                    lines.append(f"  • [{p['id']}] {p['filename']}")
            lines.append(f"\n  (共引用 {len(cited_papers)} 篇权威论文)")

        lines.append(f"\n{'=' * 70}")
        lines.append(f"  报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  数据来源: {len(self.index)} 篇权威论文 + 患者步态数据库")
        lines.append(f"{'=' * 70}")

        return "\n".join(lines)


def generate_expert_report(gait_stats, trend_data=None, api_key=None):
    """快捷函数：生成专家报告"""
    generator = ExpertReportGenerator()
    report_json, cited_papers, raw, error = generator.generate(
        gait_stats, trend_data, api_key
    )
    if error:
        return None, error
    formatted = generator.format_report(report_json, cited_papers)
    return formatted, None
