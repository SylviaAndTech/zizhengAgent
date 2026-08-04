"""
把一个案例的七段式内容写入 python-docx 的 Document 对象。
供「勾选案例导出Word」和「成书编译」两处共用，避免两套导出各写一份格式。
"""


def write_case_section(doc, case_dict: dict, heading_level: int = 1):
    """case_dict: Case.to_dict() 的结果"""
    d = case_dict
    doc.add_heading(f"案例{d['case_code']}：{d['title'] or ''}", level=heading_level)
    doc.add_paragraph(f"所属维度：{d['dimension'] or ''}　审核状态：{d['status']}")

    sub = heading_level + 1

    doc.add_heading("一、完整案例", level=sub)
    doc.add_paragraph(d["full_narrative"] or "")

    doc.add_heading("二、案例教学目标", level=sub)
    for k, v in (d["teaching_objectives"] or {}).items():
        doc.add_paragraph(f"{k}：{v}")

    doc.add_heading("三、课程思政元素", level=sub)
    for k, v in (d["sizheng_elements"] or {}).items():
        doc.add_paragraph(f"{k}：{v}")

    doc.add_heading("四、适用课程举例", level=sub)
    courses = d["applicable_courses"] or []
    if courses:
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text, hdr[2].text = "课程名称", "适用章节", "融入方式建议"
        for row in courses:
            cells = table.add_row().cells
            cells[0].text = str(row.get("课程名称", ""))
            cells[1].text = str(row.get("适用章节", ""))
            cells[2].text = str(row.get("融入方式建议", ""))

    doc.add_heading("五、教学设计", level=sub)
    for k, v in (d["teaching_design"] or {}).items():
        doc.add_paragraph(f"{k}：{v}")

    doc.add_heading("六、课程评价与成效", level=sub)
    for k, v in (d["evaluation"] or {}).items():
        doc.add_paragraph(f"{k}：{v}")

    doc.add_heading("七、延伸阅读", level=sub)
    for r in (d["further_reading"] or []):
        doc.add_paragraph(f"[{r.get('type', '')}] {r.get('title', '')} {r.get('url', '')}")

    doc.add_page_break()
