"""
把"案例→课程→章节→知识点"关系画成Mermaid树状图的定义字符串——是frontend/index.html里
buildApplicableCoursesMermaid()/mermaidEsc()的Python移植版，两边逻辑必须保持一致
（节点ID、classDef样式、分组规则、换行宽度都相同），这样"导出Word里嵌入的图"和"页面上
看到的图"才是同一张图，不会跑出两份不一致的实现。

之所以需要一份独立的Python实现：Mermaid本身只是一段文本定义，真正画图靠mermaid.js，
之前只有浏览器端在跑；现在要把这张图也导出进Word（见mermaid_render.py），后端需要能
独立生成同样的定义字符串去渲染，而"怎么分组、怎么转义、怎么换行"这部分逻辑不涉及DOM/
浏览器API，照搬到Python不需要跑在JS环境里。
"""
import re

_ESC_RE = re.compile(r'["\[\]]')
_NEWLINE_RE = re.compile(r"\s*\n+\s*")

# 跟前端保持一致的单行换行宽度。Mermaid不会自动给节点内的文字换行——文字多长，节点就
# 撑多宽，所以超过这个字符数就手动按字符数插入<br/>分行，而不是像旧版那样直接截断丢弃
# 后半段内容（那样会导致"知识点简述"这类本来就偏长的技术描述被腰斩，看不全）
WRAP_LEN = 16


def mermaid_escape(s, wrap_len: int | None = WRAP_LEN) -> str:
    """mermaid节点标签只能用双引号包裹的文本，标签内不能再出现裸的双引号/方括号，否则会
    破坏语法结构；换行统一转成<br/>（mermaid标签支持这个标签）。wrap_len传None就不做任何
    换行处理（比如案例编号"2.2"这种本来就很短的文本不需要）。"""
    t = _NEWLINE_RE.sub("<br/>", _ESC_RE.sub("", str(s or "").strip()))
    if wrap_len:
        segments = []
        for seg in t.split("<br/>"):
            if len(seg) <= wrap_len:
                segments.append(seg)
            else:
                segments.append("<br/>".join(seg[i:i + wrap_len] for i in range(0, len(seg), wrap_len)))
        t = "<br/>".join(segments)
    return t


def build_applicable_courses_mermaid(case_dict: dict) -> str | None:
    """案例详情页"④ 适用课程举例"的树状图：第一层案例编号+名字，第二层课程名字，第三层
    章节，第四层知识点简述（同一课程/章节下可能有多条已采纳的知识点关联，各自一个叶子节点）。
    case_dict是Case.to_dict()的结果；没有适用课程举例数据时返回None（没有已采纳的知识点
    关联，或者还没跑过知识点匹配）。"""
    courses = case_dict.get("applicable_courses") or []
    if not courses:
        return None

    lines = [
        "graph LR",
        "classDef caseNode fill:#4F46E5,stroke:#3730A3,stroke-width:2px,color:#FFFFFF,rx:8px,ry:8px;",
        "classDef courseNode fill:#0EA5E9,stroke:#0369A1,stroke-width:2px,color:#FFFFFF,rx:6px,ry:6px;",
        "classDef chapterNode fill:#10B981,stroke:#047857,stroke-width:2px,color:#FFFFFF,rx:6px,ry:6px;",
        "classDef kpNode fill:#F3F4F6,stroke:#D1D5DB,stroke-width:1.5px,color:#1F2937,rx:6px,ry:6px;",
        f'CASE["案例 {mermaid_escape(case_dict.get("case_code"), None)}<br/>'
        f'{mermaid_escape(case_dict.get("title"))}"]:::caseNode',
    ]

    course_map: dict[str, dict[str, list[str]]] = {}  # 课程名 -> {章节名: [知识点简述, ...]}
    for row in courses:
        course_name = (row.get("课程名称") or "").strip() or "(未命名课程)"
        chapter_name = (row.get("适用章节") or "").strip() or "(未标注章节)"
        kp = (row.get("知识点简述") or "").strip()
        chapter_map = course_map.setdefault(course_name, {})
        kp_list = chapter_map.setdefault(chapter_name, [])
        if kp:
            kp_list.append(kp)

    for ci, (course_name, chapter_map) in enumerate(course_map.items(), start=1):
        cid = f"C{ci}"
        lines.append(f'CASE --> {cid}["课程：{mermaid_escape(course_name)}"]:::courseNode')
        for chi, (chapter_name, kps) in enumerate(chapter_map.items(), start=1):
            chid = f"{cid}_CH{chi}"
            lines.append(f'{cid} --> {chid}["{mermaid_escape(chapter_name)}"]:::chapterNode')
            for ki, kp in enumerate(kps, start=1):
                kpid = f"{chid}_KP{ki}"
                lines.append(f'{chid} --> {kpid}["知识点：{mermaid_escape(kp)}"]:::kpNode')

    return "\n".join(lines)
