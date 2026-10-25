"""
成书编译：把已采纳的案例按"前言 + 第一章课程思政元素总览 + 按维度分章 + 附录一/二 +
附录三知识图谱"的固定版式拼装成一份 Word 文档。汇总知识图谱放在全书最后一部分。
"""
import io
from collections import defaultdict

import docx
from docx.shared import Inches

from book_front_matter import (
    PREFACE_SECTIONS, SIZHENG_ELEMENT_DEFINITIONS, GRAPH_USAGE_NOTE, CHAPTER_SUBTITLES,
)
from db import Case, CaseKnowledgeMapping, DIMENSIONS
from doc_writer import write_case_section, set_default_font
from knowledge_graph import case_tree_mermaid, summary_mermaid, summary_graph_data, summary_empty_reason
from mermaid_render import render_mermaid_batch

# 六个思政维度对应第二至七章；"第一章"是"课程思政元素总览"，从"第二章"起接上
CHAPTER_TITLES = {
    "政治认同": "第二章 政治认同",
    "家国情怀": "第三章 家国情怀",
    "文化素养": "第四章 文化素养",
    "科学精神": "第五章 科学精神",
    "宪法法治意识": "第六章 宪法法治意识",
    "道德修养": "第七章 道德修养",
}


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
    """附录二：案例编号 | 关联课程/章节 | 融入方式建议。只列已经写入案例的知识点关联（bound），
    跟案例正文"适用课程举例"里的内容一致；融入方式建议优先用案例里那一行的文字（"补充"时
    按案例内容生成的），没有再用匹配时的建议"""
    case_by_id = {c.id: c for c in cases}
    mappings = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id.in_(list(case_by_id)), CaseKnowledgeMapping.bound.is_(True))
        .order_by(CaseKnowledgeMapping.case_id, CaseKnowledgeMapping.relevance_score.desc())
        .all()
    )
    mappings = [m for m in mappings if m.knowledge_point]
    if not mappings:
        doc.add_paragraph("（暂无已写入案例的知识点关联，可在「知识点匹配」标签页采纳知识点并点击「补充」后重新导出）")
        return

    row_suggestion = {}  # (case_id, knowledge_point_id) -> 案例"适用课程举例"里的融入方式建议
    for c in cases:
        for row in (c.to_dict().get("applicable_courses") or []):
            if isinstance(row, dict) and row.get("knowledge_point_id") is not None:
                row_suggestion[(c.id, row["knowledge_point_id"])] = row.get("融入方式建议") or ""

    mappings.sort(key=lambda m: (case_by_id[m.case_id].case_code or "", -(m.relevance_score or 0)))
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = "案例编号", "关联课程/章节", "融入方式建议"
    for m in mappings:
        kp = m.knowledge_point
        cells = table.add_row().cells
        cells[0].text = case_by_id[m.case_id].case_code or ""
        cells[1].text = f"{kp.course_name}　{kp.chapter or ''}"
        cells[2].text = row_suggestion.get((m.case_id, m.knowledge_point_id)) or m.suggestion_text or ""


def _write_knowledge_graph_appendix(doc, db):
    """全书最后一部分：汇总知识图谱（思政维度 → 案例 → 课程 → 章节 → 知识点）。按维度一张图分开
    画——全部维度画成一张图会长到超出一页，按页宽缩小后字就看不清了"""
    doc.add_heading("附录三 课程思政知识图谱", level=1)
    doc.add_paragraph("图谱结构：思政维度 → 案例 → 课程 → 章节 → 知识点（每个知识点节点是从该章节教学内容中"
                      "选出的、与案例最相关的一条，每个章节最多5个）。仅汇总审核状态为「已采纳」的案例。")
    doc.add_paragraph(GRAPH_USAGE_NOTE)

    reason = summary_empty_reason(db)
    if reason:
        doc.add_paragraph(f"（{reason}）")
        return
    dims = [dim for dim, _items in summary_graph_data(db)]
    pngs = render_mermaid_batch([summary_mermaid(db, only_dimension=dim) for dim in dims])
    for i, (dim, png) in enumerate(zip(dims, pngs), start=1):
        doc.add_heading(f"（{i}）{dim}", level=2)
        if png:
            doc.add_picture(io.BytesIO(png), width=Inches(6.3))
        else:
            doc.add_paragraph("（知识图谱图片生成失败：服务器需要安装Playwright浏览器内核，执行 playwright install chromium）")


def build_book_docx(db, status_filter: str = "已采纳") -> bytes:
    doc = docx.Document()
    set_default_font(doc)
    doc.add_heading("讲好数字中国故事：人工智能类课程思政案例集", level=0)

    doc.add_heading("前言", level=1)
    for title, body in PREFACE_SECTIONS:
        doc.add_heading(title, level=2)
        doc.add_paragraph(body)
    doc.add_page_break()

    doc.add_heading("第一章 课程思政元素总览", level=1)
    for title, body in SIZHENG_ELEMENT_DEFINITIONS:
        doc.add_paragraph(f"{title}：{body}")
    doc.add_paragraph("全部案例与课程知识点的关联关系，见全书最后的「附录三 课程思政知识图谱」。")
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

    # 树状图渲染（起无头浏览器）比较慢，全书可能有几十个案例，这里对所有案例只批量渲染
    # 一次，不要在下面的分章节循环里逐个案例单独渲染
    case_dicts = {c.id: c.to_dict() for c in cases}
    tree_pngs = dict(zip(
        case_dicts.keys(),
        render_mermaid_batch([case_tree_mermaid(db, c) for c in cases]),
    ))

    for dim in DIMENSIONS:
        chapter_title = CHAPTER_TITLES.get(dim, dim)
        subtitle = CHAPTER_SUBTITLES.get(dim)
        doc.add_heading(f"{chapter_title}：{subtitle}" if subtitle else chapter_title, level=1)
        dim_cases = by_dimension.get(dim, [])
        if not dim_cases:
            doc.add_paragraph(f"（{dim}维度暂无「{status_filter}」状态的案例）")
            continue
        for c in dim_cases:
            write_case_section(doc, case_dicts[c.id], heading_level=2, course_tree_png=tree_pngs.get(c.id),
                               course_table=False)

    doc.add_heading("附录一 案例总览表", level=1)
    _write_case_overview_table(doc, cases)
    doc.add_page_break()

    doc.add_heading("附录二 课程思政案例与人工智能相关课程关联信息总表", level=1)
    _write_knowledge_mapping_table(doc, db, cases)
    doc.add_page_break()

    _write_knowledge_graph_appendix(doc, db)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
