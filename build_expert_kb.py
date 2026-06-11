"""构建专家知识库 — 从权威论文 PDF 中提取文本并建立索引

一次性运行:
    python build_expert_kb.py

输出:
    expert_kb/                           # 知识库目录
    ├── papers_index.json                # 论文索引 + 元数据
    └── texts/                           # 各论文提取的纯文本
"""
import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(PROJECT_ROOT, "Authoritative Literature & Expert Database")
KB_DIR = os.path.join(PROJECT_ROOT, "expert_kb")
TEXTS_DIR = os.path.join(KB_DIR, "texts")


def extract_pdf_text(pdf_path, max_pages=None):
    """提取 PDF 文本，可选限制页数"""
    doc = fitz.open(pdf_path)
    text_parts = []
    pages = min(len(doc), max_pages) if max_pages else len(doc)

    for i in range(pages):
        page = doc[i]
        text = page.get_text("text")
        if text.strip():
            text_parts.append(f"--- 第 {i+1} 页 ---\n{text}")

    doc.close()
    return "\n\n".join(text_parts)


def extract_abstract_and_key(text):
    """尝试提取摘要和关键发现（前 3000 字符）"""
    # 提取前 3000 字符作为摘要
    abstract = text[:3000]
    # 尝试找关键词部分
    keywords = ""
    kw_match = re.search(r'(?:Keywords|KEYWORDS|Key words)[:\s]*(.*?)(?:\n\n|\n[A-Z])', text, re.DOTALL)
    if kw_match:
        keywords = kw_match.group(1).strip()[:500]
    return abstract, keywords


def classify_paper(filename, text):
    """根据文件名和内容分类论文主题"""
    name_lower = filename.lower()
    categories = []

    if any(w in name_lower for w in ['foot_clearance', 'foot clearance']):
        categories.append("foot_clearance")
    if any(w in name_lower for w in ['trunk', 'accelerometer']):
        categories.append("trunk_sway")
    if any(w in name_lower for w in ['variability', 'variabilit']):
        categories.append("gait_variability")
    if any(w in name_lower for w in ['parkinson']):
        categories.append("parkinsons")
    if any(w in name_lower for w in ['fall_risk', 'fall risk', 'predicting_fall']):
        categories.append("fall_risk")
    if any(w in name_lower for w in ['spatiotemporal', 'lifespan', 'adult']):
        categories.append("spatiotemporal")
    if any(w in name_lower for w in ['wearable']):
        categories.append("wearable_sensors")
    if any(w in name_lower for w in ['kinematic']):
        categories.append("kinematics")
    if 'jpts' in name_lower:
        categories.append("gait_analysis")
    if 'nihms' in name_lower:
        categories.append("gait_parameters")
    if 'fbioe' in name_lower:
        categories.append("biomechanics")

    # 从内容中检测更多分类
    if 'step width' in text.lower() or 'step_width' in text.lower():
        categories.append("step_width")
    if 'cadence' in text.lower():
        categories.append("cadence")
    if 'stride' in text.lower():
        categories.append("stride")
    if 'double support' in text.lower():
        categories.append("double_support")
    if 'knee' in text.lower() and ('rom' in text.lower() or 'range' in text.lower()):
        categories.append("knee_rom")
    if 'clearance' in text.lower():
        categories.append("foot_clearance")
    if 'symmetry' in text.lower():
        categories.append("symmetry")

    return list(set(categories))


def main():
    os.makedirs(TEXTS_DIR, exist_ok=True)
    papers_dir = Path(PAPERS_DIR)
    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[错误] 未找到 PDF 文件: {PAPERS_DIR}")
        return

    print(f"找到 {len(pdf_files)} 篇论文，开始提取...\n")

    papers_index = []
    total_pages = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{i}/{len(pdf_files)}] {pdf_path.name[:70]}...", end=" ", flush=True)
            full_text = extract_pdf_text(str(pdf_path))
            abstract, keywords = extract_abstract_and_key(full_text)

            # 统计
            pages = full_text.count("--- 第 ")
            total_pages += pages
            chars = len(full_text)

            # 分类
            categories = classify_paper(pdf_path.name, full_text)

            # 保存纯文本
            txt_name = pdf_path.stem[:60] + ".txt"
            txt_path = os.path.join(TEXTS_DIR, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # 索引条目
            entry = {
                "id": i,
                "filename": pdf_path.name,
                "text_file": txt_name,
                "pages": pages,
                "chars": chars,
                "categories": categories,
                "abstract": abstract[:500] if abstract else "",
                "keywords": keywords,
            }
            papers_index.append(entry)
            print(f"[OK] {pages}p {chars}c {categories}")

        except Exception as e:
            print(f"[ERR] {e}")
            papers_index.append({
                "id": i,
                "filename": pdf_path.name,
                "error": str(e),
            })

    # 保存索引
    index_path = os.path.join(KB_DIR, "papers_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(papers_index, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  Knowledge Base Built!")
    print(f"  Papers: {len(papers_index)}")
    print(f"  Pages:  {total_pages}")
    print(f"  Index:  {index_path}")
    print(f"  Texts:  {TEXTS_DIR}")
    print(f"\n  Topics covered:")
    all_cats = set()
    for p in papers_index:
        all_cats.update(p.get("categories", []))
    for c in sorted(all_cats):
        count = sum(1 for p in papers_index if c in p.get("categories", []))
        print(f"    {c}: {count} papers")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
