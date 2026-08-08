"""
课程大纲数字PDF（有文字层，不是扫描件）里"知识单元/教学内容(知识点)/教学目标/..."这类
表格的解析。按坐标分列，全程不调用大模型，免费、瞬间出结果。

（扫描件走的是另一套方案，见 syllabus_vision_ocr.py：扫描件没有可靠的文字层坐标，
按坐标分列常常认错——尤其是窄列里被拉伸变形的英文单词，Tesseract逐字符识别没有上下文
纠错能力；扫描件改用视觉大模型直接"看图"识别，效果好得多，但要花钱调用，所以只在真正
遇到扫描件时才用。）

文字层里同一行会把"知识点"列和右边"教学目标"列的文字揉在一起（两列都用"1. 2. 3."编号，
混在一起后没法区分哪条是知识点、哪条是教学目标）。这里用每行文字的横坐标做一次简化的
"列还原"：先在表头行里找到"知识单元"列和"教学内容/知识点"列各自的横坐标，之后每一行
正文按横坐标落在哪一列，就归到哪一列，从而只取"知识点"列的编号条目，不混入旁边教学目标
列的内容。

只处理这一种表格结构（知识单元+知识点两列相邻、知识点条目用"N."编号），不是通用表格解析器；
但这是高校OBE培养方案教学大纲里最常见的标准格式，具备一定的通用性。
"""
import re

from ocr_utils import native_page_lines

_NUM_ITEM_RE = re.compile(r"^\s*(\d{1,2})[.\、]\s*(\S.*)$")
_TOP_SECTION_RE = re.compile(r"[六七八九十]、|课程考核方式")
_UNIT_KEYWORDS = ("知识单元", "实验单元")
_POINT_KEYWORDS = ("知识点", "教学内容")
_HEADER_MATCH_MAX_DIST = 40  # 表头里"单元"和"知识点"两个关键词的y坐标允许的最大偏差
_UNIT_RUN_GAP = 30  # 章节列里，y间距在这个范围内的几行文字算同一个单元标题的换行


def _find_column_anchors(lines):
    """在本页所有行里找"知识单元"列和"知识点"列表头各自的x中心。
    不假设表头一定在页面顶部（有的页表格前还有别的大纲正文段落）。"""
    unit_candidates = []
    point_candidates = []
    for x0, y0, x1, y1, text in lines:
        cx = (x0 + x1) / 2
        if any(kw in text for kw in _UNIT_KEYWORDS) or text.strip() == "单元":
            unit_candidates.append((y0, cx))
        if any(kw in text for kw in _POINT_KEYWORDS):
            point_candidates.append((y0, cx))
    if not unit_candidates or not point_candidates:
        return None

    best = None
    for py0, pcx in point_candidates:
        for uy0, ucx in unit_candidates:
            dist = abs(py0 - uy0)
            if best is None or dist < best[0]:
                best = (dist, ucx, pcx)
    if best is None or best[0] > _HEADER_MATCH_MAX_DIST:
        return None
    return best[1], best[2]  # unit_cx, point_cx


def extract_units_and_points(doc, get_page_lines=native_page_lines) -> list[tuple[str | None, str]]:
    """
    doc: 已打开的 fitz.Document（有文字层的数字PDF）
    get_page_lines: 单页取"带坐标文本行"的方式，默认直接读文字层（ocr_utils.native_page_lines）。
      理论上也能传ocr_page_lines给扫描件用，但扫描件现在走的是效果好得多的
      syllabus_vision_ocr.py，不会再用到这里。
    返回 [(章节名或None, 知识点描述), ...]，按出现顺序
    """
    results: list[tuple[str | None, str]] = []
    last_anchors = None
    last_active_page = None
    last_known_chapter = None

    for pno in range(len(doc)):
        lines = get_page_lines(doc[pno])

        # 顶级章节分界（比如"六、课程考核方式"）之后就不是知识单元表格了，切掉
        cutoff_y = None
        for x0, y0, x1, y1, text in lines:
            if _TOP_SECTION_RE.search(text):
                cutoff_y = y0 if cutoff_y is None else min(cutoff_y, y0)
        if cutoff_y is not None:
            lines = [l for l in lines if l[1] < cutoff_y]

        anchors = _find_column_anchors(lines)

        # 中间隔了一整页没匹配上表头/知识点的，说明已经离开了这张表，缓存的列位置作废
        if last_active_page is not None and pno - last_active_page > 1:
            last_anchors = None
        if anchors:
            last_anchors = anchors
        if not last_anchors:
            continue

        unit_cx, point_cx = last_anchors
        mid = (unit_cx + point_cx) / 2
        right_bound = point_cx + (point_cx - unit_cx) / 2

        unit_lines = []
        point_lines = []
        for x0, y0, x1, y1, text in lines:
            cx = (x0 + x1) / 2
            if cx <= mid:
                unit_lines.append((y0, text))
            elif cx <= right_bound:
                m = _NUM_ITEM_RE.match(text)
                if m:
                    point_lines.append((y0, m.group(2).strip()))

        # 章节列的文字通常在合并单元格的纵向居中位置，不是顶部对齐，所以用"离哪个单元标题片段
        # 最近"而不是"顺序落在哪个单元标题之后"来归属
        unit_runs = []
        for y0, text in unit_lines:
            if unit_runs and y0 - unit_runs[-1][-1][0] <= _UNIT_RUN_GAP:
                unit_runs[-1].append((y0, text))
            else:
                unit_runs.append([(y0, text)])
        run_reprs = [
            (sum(y for y, _ in run) / len(run), "".join(t for _, t in run))
            for run in unit_runs
        ]

        if anchors or point_lines:
            last_active_page = pno

        for y0, text in point_lines:
            if run_reprs:
                chapter = min(run_reprs, key=lambda r: abs(r[0] - y0))[1]
            else:
                chapter = last_known_chapter
            results.append((chapter, text))

        if run_reprs:
            last_known_chapter = run_reprs[-1][1]

    return results
