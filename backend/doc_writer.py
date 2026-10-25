"""
把一个案例的七段式内容写入 python-docx 的 Document 对象。
供「勾选案例导出Word」和「成书编译」两处共用，避免两套导出各写一份格式。
"""
import io
import re

from docx.oxml.ns import qn
from docx.shared import Inches

# 案例正文里的句级引用标注，如 [素材1]、[素材2:关键词片段]（第二种带定位短语，供前端hover预览用）；
# 导出成Word给人看的最终稿时，这些标注是给内部溯源用的，不应该出现在正式文档里，统一去掉
_CITATION_RE = re.compile(r"\[素材\d+(?::[^\]]*)?\]")

DEFAULT_FONT = "宋体"
_STYLED_NAMES = ("Normal", "Title", "Heading 1", "Heading 2", "Heading 3", "Heading 4", "Heading 5", "Heading 6")


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text or "")


def set_default_font(doc, font_name: str = DEFAULT_FONT):
    """python-docx设置字体要中英文分开设：font.name只管西文(ascii/hAnsi)，中文走的是
    w:eastAsia这个单独的属性，两个都不设的话Word里中文还是显示成默认字体，看不出变化"""
    for style_name in _STYLED_NAMES:
        try:
            style = doc.styles[style_name]
        except KeyError:
            continue
        style.font.name = font_name
        rpr = style.element.get_or_add_rPr()
        rpr.rFonts.set(qn("w:eastAsia"), font_name)


_CN_NUMS = "一二三四五六七八九十"


def sizheng_sections(se: dict) -> dict | None:
    """课程思政元素分析里各维度的论述。早期版本的键叫"六维度阐述"（当时六个维度都写），
    现在只写案例体现了的维度、键改叫"维度阐述"；两种都认。都不是（更早的五维度旧格式）返回None"""
    for key in ("维度阐述", "六维度阐述"):
        if isinstance(se.get(key), dict):
            return se[key]
    return None


def _write_sizheng_analysis(doc, se: dict):
    """课程思政元素分析：总述一段，再按维度"一、政治认同方面"……逐段论述"""
    if se.get("对应维度"):
        doc.add_paragraph(f"所属维度：{se['对应维度']}")
    doc.add_paragraph(f"案例的课程思政元素分析：{se.get('总述') or ''}")
    for i, (dim, text) in enumerate(sizheng_sections(se).items()):
        p = doc.add_paragraph()
        p.add_run(f"{_CN_NUMS[i]}、{dim}方面").bold = True
        doc.add_paragraph(str(text or ""))


def _write_legacy_sizheng(doc, se: dict):
    """旧格式（五维度：摘录表述+解释）的课程思政元素，原样输出"""
    for k, v in se.items():
        if k == "五维度阐释" and isinstance(v, dict):
            doc.add_paragraph("五维度阐释：")
            for dim, detail in v.items():
                if isinstance(detail, dict):
                    doc.add_paragraph(f"　　{dim} - 摘录表述：{detail.get('摘录表述', '')}")
                    doc.add_paragraph(f"　　{dim} - 解释：{detail.get('解释', '')}")
                else:
                    doc.add_paragraph(f"　　{dim}：{detail}")
        else:
            doc.add_paragraph(f"{k}：{v}")


def write_case_section(doc, case_dict: dict, heading_level: int = 1, course_tree_png: bytes | None = None,
                       course_table: bool = True):
    """case_dict: Case.to_dict() 的结果。
    course_tree_png: 可选，该案例知识图谱（案例 → 课程 → 章节 → 知识点）预先渲染好的PNG字节
    ——渲染本身（Playwright起无头浏览器跑mermaid.js）比较慢，同一次导出可能有几十个案例，
    要在调用方批量渲染一次后逐个传进来，不要在这个函数里每个案例单独渲染一次，见
    mermaid_render.render_mermaid_batch()。不传或传None就只写文字版表格，不嵌图片
    （没有适用课程举例数据，或者图渲染失败时就是这种情况，不影响其余内容照常导出）。"""
    d = case_dict
    doc.add_heading(f"案例{d['case_code']}：{d['title'] or ''}", level=heading_level)
    doc.add_paragraph(f"所属维度：{d['dimension'] or ''}　审核状态：{d['status']}")

    sub = heading_level + 1

    doc.add_heading("一、完整案例", level=sub)
    doc.add_paragraph(strip_citations(d["full_narrative"]))

    doc.add_heading("二、案例教学目标", level=sub)
    for k, v in (d["teaching_objectives"] or {}).items():
        doc.add_paragraph(f"{k}：{v}")

    doc.add_heading("三、课程思政元素", level=sub)
    se = d["sizheng_elements"] or {}
    if sizheng_sections(se) is not None:
        _write_sizheng_analysis(doc, se)
    else:
        _write_legacy_sizheng(doc, se)

    doc.add_heading("四、适用课程举例", level=sub)
    courses = d["applicable_courses"] or []
    # course_table=False：只放知识图谱、不出表格（成书用——课程/章节/融入方式建议在书末附录二里
    # 统一汇总，每个案例里再出一遍表格显得重复）
    if courses and course_table:
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text, hdr[2].text = "课程名称", "适用章节", "融入方式建议"
        for row in courses:
            cells = table.add_row().cells
            cells[0].text = str(row.get("课程名称", ""))
            cells[1].text = str(row.get("适用章节", ""))
            cells[2].text = str(row.get("融入方式建议", ""))

    if course_tree_png:
        doc.add_paragraph("案例知识图谱（案例 → 课程 → 章节 → 知识点）：")
        doc.add_picture(io.BytesIO(course_tree_png), width=Inches(6.3))
    elif not course_table:
        # 不出表格时图又没有，这一节就空了，写明原因
        doc.add_paragraph("（暂无，案例尚未写入课程知识点）" if not courses
                          else "（知识图谱图片生成失败：服务器需要安装Playwright浏览器内核，执行 playwright install chromium）")

    doc.add_heading("五、教学设计", level=sub)
    for k, v in (d["teaching_design"] or {}).items():
        if k == "依据知识点ID":  # 内部记录用的字段（教学设计依据哪条知识点生成），不是正文内容
            continue
        doc.add_paragraph(f"{k}：{v}")

    doc.add_heading("六、思考与讨论", level=sub)
    for i, q in enumerate(d.get("discussion_questions") or [], start=1):
        doc.add_paragraph(f"{i}. {q}")

    doc.add_heading("七、延伸阅读", level=sub)
    for r in (d["further_reading"] or []):
        doc.add_paragraph(f"[{r.get('type', '')}] {r.get('title', '')} {r.get('url', '')}")

    doc.add_page_break()
