"""
成书编译：把已采纳的案例按"前言 + 按维度分章 + 附录一/二/三"的固定版式
拼装成一份 Word 文档，并把知识图谱截图嵌入附录之后。
"""
import io
from collections import defaultdict

import docx
from docx.shared import Inches

from db import Case, CaseKnowledgeMapping, DIMENSIONS
from doc_writer import write_case_section, set_default_font
from knowledge_graph import build_graph, render_graph_png

# 对应 README 里"五维度固定值，对应大纲第二至六章"的映射关系
CHAPTER_TITLES = {
    "政治认同": "第二章 政治认同",
    "家国情怀": "第三章 家国情怀",
    "文化素养": "第四章 文化素养",
    "宪法法治意识": "第五章 宪法法治意识",
    "道德修养": "第六章 道德修养",
}

PREFACE_TEMPLATE = (
    "本书围绕《讲好数字中国故事：人工智能类课程思政案例集》的编写目标，收录经审核采纳的思政案例，"
    "按政治认同、家国情怀、文化素养、宪法法治意识、道德修养五个维度分章编排，供高校人工智能相关"
    "课程教学参考使用。\n"
    "（本段前言为原型自动生成的模板文字，并非原始大纲文本，正式出版前请人工核校替换。）"
)


def _write_case_overview_table(doc, cases: list):
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "案例编号", "标题", "所属维度", "审核状态"
    for c in cases:
        cells = table.add_row().cells
        cells[0].text = c.case_code or ""
        cells[1].text = c.title or ""
        cells[2].text = c.dimension or ""
        cells[3].text = c.status or ""


def _write_knowledge_mapping_table(doc, db, cases: list):
    case_ids = [c.id for c in cases]
    mappings = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id.in_(case_ids), CaseKnowledgeMapping.status == "已采纳")
        .all()
    )
    if not mappings:
        doc.add_paragraph("（暂无已采纳的知识点关联，可在「知识点匹配」标签页运行匹配并采纳后重新导出）")
        return

    case_by_id = {c.id: c for c in cases}
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "案例编号", "关联课程/章节", "知识点描述", "融入方式建议"
    for m in mappings:
        case = case_by_id.get(m.case_id)
        kp = m.knowledge_point
        cells = table.add_row().cells
        cells[0].text = case.case_code if case else str(m.case_id)
        cells[1].text = f"{kp.course_name}　{kp.chapter or ''}" if kp else ""
        cells[2].text = kp.description if kp else ""
        cells[3].text = m.suggestion_text or ""


def _write_resource_index(doc, cases: list):
    seen = set()
    rows = []
    for c in cases:
        for r in (c.to_dict().get("further_reading") or []):
            url = r.get("url") or ""
            key = (r.get("title"), url)
            if key in seen:
                continue
            seen.add(key)
            rows.append((c.case_code, r.get("type", ""), r.get("title", ""), url))

    if not rows:
        doc.add_paragraph("（暂无延伸阅读资源）")
        return

    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "来源案例", "类型", "标题", "链接"
    for case_code, rtype, title, url in rows:
        cells = table.add_row().cells
        cells[0].text = case_code or ""
        cells[1].text = rtype
        cells[2].text = title
        cells[3].text = url


def build_book_docx(db, status_filter: str = "已采纳") -> bytes:
    doc = docx.Document()
    set_default_font(doc)
    doc.add_heading("讲好数字中国故事：人工智能类课程思政案例集（自动编译）", level=0)

    doc.add_heading("前言", level=1)
    doc.add_paragraph(PREFACE_TEMPLATE)
    doc.add_page_break()

    cases = (
        db.query(Case)
        .filter(Case.status == status_filter)
        .order_by(Case.case_code)
        .all()
    )
    by_dimension = defaultdict(list)
    for c in cases:
        by_dimension[c.dimension].append(c)

    for dim in DIMENSIONS:
        doc.add_heading(CHAPTER_TITLES.get(dim, dim), level=1)
        dim_cases = by_dimension.get(dim, [])
        if not dim_cases:
            doc.add_paragraph(f"（{dim}维度暂无「{status_filter}」状态的案例）")
            continue
        for c in dim_cases:
            write_case_section(doc, c.to_dict(), heading_level=2)

    doc.add_heading("附录一 案例总览表", level=1)
    _write_case_overview_table(doc, cases)
    doc.add_page_break()

    doc.add_heading("附录二 案例与知识点关联表", level=1)
    _write_knowledge_mapping_table(doc, db, cases)
    doc.add_page_break()

    doc.add_heading("附录三 延伸阅读资源索引", level=1)
    _write_resource_index(doc, cases)
    doc.add_page_break()

    doc.add_heading("知识图谱", level=1)
    graph = build_graph(db)
    png_bytes = render_graph_png(graph)
    doc.add_picture(io.BytesIO(png_bytes), width=Inches(6.3))

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
