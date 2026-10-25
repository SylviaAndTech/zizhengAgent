"""
知识图谱：两种
1. 单个案例的知识图谱：案例 → 课程 → 章节 → 知识点，显示在案例详情"④ 适用课程举例"里（横向树状），
   也嵌进导出的Word里该案例的"适用课程举例"部分。
2. 汇总知识图谱：思政维度 → 案例 → 课程 → 章节 → 知识点，覆盖全部"已采纳"（审核通过）的案例，
   页面上"知识图谱"卡片查看，也放在"导出成书"Word的最后一部分。

数据都直接取自数据库里"已写入案例"（CaseKnowledgeMapping.bound）的知识点关联，课程名/章节/
知识点描述都是知识点库里的原文——不用案例"适用课程举例"里模型写的"课程名称/适用章节"
文字：那是模型生成的，偶尔会把整段教学内容塞进"适用章节"里，拿来画图会很乱。

图本身用Mermaid横向树（graph LR）表示：同一份定义字符串，页面上用mermaid.js直接渲染，
导出Word时用mermaid_render（无头浏览器）渲染成PNG，两边是同一张图。可交互版本用pyvis
（vis.js）的分层布局，节点可以拖拽，层次跟Mermaid树一致。

文字完整呈现：节点里的文字一律不截断，过长的按固定字数手动换行（Mermaid不会自动给节点内
文字换行，文字多长节点就撑多宽）。

知识点节点不是整段知识点描述，而是从描述里挑出的跟案例最相关的1~5条原文条目，每条一个
节点，挂在所属章节（知识单元）节点下面（见kp_points.py）。
"""
import json
import re

from pyvis.network import Network

from db import Case, CaseKnowledgeMapping, DIMENSIONS
from kp_points import fallback_points

# 会破坏Mermaid语法（双引号、方括号）或被当成HTML标签（尖括号）的字符，换成对应的全角
# 字符——不能直接删掉，要求文字完整呈现
_ESC_TABLE = str.maketrans({'"': "＂", "[": "［", "]": "］", "<": "＜", ">": "＞"})
_NEWLINE_RE = re.compile(r"\s*\n+\s*")

# 各层节点每行字数
WRAP = {"dimension": 8, "case": 16, "course": 14, "chapter": 18, "kp": 24}

# 所有节点都是浅色底、深色字，各层用不同色系区分
LAYER_STYLE = {  # 层 -> (底色, 边框, 文字)
    "dimension": ("#FDECEC", "#F19A9D", "#8F1F23"),
    "case": ("#EEF2FF", "#6366F1", "#312E81"),
    "course": ("#E0F2FE", "#38BDF8", "#075985"),
    "chapter": ("#FFFBEB", "#FCD34D", "#78350F"),
    "kp": ("#F0FDF4", "#86EFAC", "#14532D"),
}
_CLASS_NAMES = {"dimension": "dimNode", "case": "caseNode", "course": "courseNode",
                "chapter": "chapterNode", "kp": "kpNode"}
CLASS_DEFS = [
    f"classDef {_CLASS_NAMES[layer]} fill:{fill},stroke:{stroke},stroke-width:1px,color:{color},rx:4px,ry:4px;"
    for layer, (fill, stroke, color) in LAYER_STYLE.items()
]

# 每个知识单元（章节）在图上最多几个知识点节点
MAX_POINTS_PER_CHAPTER = 5

# 图的版式（字号、间距、连线形状）统一在mermaid_render.MERMAID_CONFIG里，页面和导出共用
# 树状图不需要箭头（层级从左到右已经表达了方向），用不带箭头的细线，看着更干净
EDGE_STYLE = "linkStyle default stroke:#94A3B8,stroke-width:1px;"


def _header() -> list[str]:
    return ["graph LR", *CLASS_DEFS]


def _wrap_lines(text, width: int) -> list[str]:
    """按字数切行（不截断），原文里的换行保留"""
    lines = []
    for seg in _NEWLINE_RE.split(str(text or "").strip()):
        lines.extend(seg[i:i + width] for i in range(0, len(seg), width))
    return lines or [""]


def _label(text, width: int) -> str:
    """Mermaid节点标签：去掉会破坏语法/被当成HTML的字符，按字数换行成<br/>"""
    return "<br/>".join(line.translate(_ESC_TABLE) for line in _wrap_lines(text, width))


def _mapping_points(m) -> list[str]:
    """这条关联在图谱里显示的知识点：写入案例时提取好的结果，还没提取过就用兜底规则"""
    if m.graph_points:
        try:
            points = json.loads(m.graph_points)
            if isinstance(points, list) and points:
                return [str(p) for p in points]
        except ValueError:
            pass
    return fallback_points(m.knowledge_point.description)


# ---------- 取数 ----------

def case_course_groups(db, case_id: int) -> list[tuple[str, list]]:
    """某个案例已写入的知识点，按课程、章节（知识单元）两级分组：
    [(课程名, [(章节名, [知识点, ...]), ...]), ...]。课程、章节都按其中最高相关度排序；
    每个章节最多MAX_POINTS_PER_CHAPTER个知识点——同一课程下同名的章节（比如不同大纲文件里
    都有"第一章"）会合并成一个章节节点，合起来也不超过上限"""
    mappings = [
        m for m in db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id == case_id, CaseKnowledgeMapping.bound.is_(True))
        .order_by(CaseKnowledgeMapping.relevance_score.desc())
        .all()
        if m.knowledge_point
    ]
    groups: dict[str, dict[str, list]] = {}  # 课程 -> {章节: [知识点, ...]}，dict保持插入顺序=相关度顺序
    for m in mappings:
        kp = m.knowledge_point
        points = groups.setdefault(kp.course_name, {}).setdefault(kp.chapter or "(未标注章节)", [])
        for point in _mapping_points(m):
            if len(points) >= MAX_POINTS_PER_CHAPTER:
                break
            if point not in points:
                points.append(point)
    return [(course, list(chapters.items())) for course, chapters in groups.items()]


def summary_graph_data(db) -> list[tuple[str, list]]:
    """汇总图谱的数据：[(维度, [(案例, [(课程名, [知识点...])...]), ...]), ...]。
    只纳入"已采纳"（审核通过）的案例；没有已写入知识点的案例也保留（挂在维度下，没有子节点）"""
    cases = db.query(Case).filter(Case.status == "已采纳").order_by(Case.case_code).all()
    by_dim: dict[str, list] = {}
    for c in cases:
        by_dim.setdefault(c.dimension or "(未标注维度)", []).append((c, case_course_groups(db, c.id)))
    order = {d: i for i, d in enumerate(DIMENSIONS)}
    return sorted(by_dim.items(), key=lambda kv: order.get(kv[0], len(order)))


def summary_empty_reason(db) -> str | None:
    """汇总图谱为空时告诉用户为什么（不然只看到一张空图，不知道该去做什么）"""
    approved = db.query(Case).filter(Case.status == "已采纳").count()
    if approved == 0:
        counts: dict[str, int] = {}
        for (status,) in db.query(Case.status).all():
            counts[status] = counts.get(status, 0) + 1
        detail = "、".join(f"{s}{n}个" for s, n in counts.items()) or "还没有任何案例"
        return f"当前没有审核通过（已采纳）的案例（{detail}），知识图谱只汇总已采纳的案例。可以在案例库里把审核通过的案例状态改为「已采纳」。"
    has_bound = (
        db.query(CaseKnowledgeMapping).join(Case, Case.id == CaseKnowledgeMapping.case_id)
        .filter(Case.status == "已采纳", CaseKnowledgeMapping.bound.is_(True)).first()
    )
    if not has_bound:
        return (f"已有{approved}个已采纳的案例，但它们都还没有写入知识点。请在「知识点匹配」页面采纳知识点，"
                "再点「补充」按钮写入案例。")
    return None


# ---------- Mermaid ----------

def _course_subtree(lines: list, parent_id: str, prefix: str, groups: list):
    """课程 → 章节 → 知识点 三层"""
    for ci, (course_name, chapters) in enumerate(groups, start=1):
        cid = f"{prefix}C{ci}"
        lines.append(f'{parent_id} --- {cid}["课程：{_label(course_name, WRAP["course"])}"]:::courseNode')
        for hi, (chapter, points) in enumerate(chapters, start=1):
            hid = f"{cid}H{hi}"
            lines.append(f'{cid} --- {hid}["{_label(chapter, WRAP["chapter"])}"]:::chapterNode')
            for ki, point in enumerate(points, start=1):
                lines.append(f'{hid} --- {hid}K{ki}["{_label(point, WRAP["kp"])}"]:::kpNode')


def _case_label(case) -> str:
    return f'案例 {_label(case.case_code, 20)}<br/>{_label(case.title, WRAP["case"])}'


def case_tree_mermaid(db, case) -> str | None:
    """单个案例的知识图谱（案例 → 课程 → 章节 → 知识点）。还没有写入任何知识点时返回None"""
    groups = case_course_groups(db, case.id)
    if not groups:
        return None
    lines = [*_header(), f'CASE["{_case_label(case)}"]:::caseNode']
    _course_subtree(lines, "CASE", "", groups)
    return "\n".join([*lines, EDGE_STYLE])


def summary_mermaid(db, only_dimension: str | None = None) -> str | None:
    """汇总知识图谱（思政维度 → 案例 → 课程 → 章节 → 知识点）。only_dimension：只画这一个维度
    （导出Word时按维度分开画，一张图塞下全部维度会长到超出一页，缩小后字看不清）"""
    data = [(d, items) for d, items in summary_graph_data(db) if only_dimension in (None, d)]
    if not data:
        return None
    lines = _header()
    for di, (dim, items) in enumerate(data, start=1):
        did = f"D{di}"
        lines.append(f'{did}["{_label(dim, WRAP["dimension"])}"]:::dimNode')
        for xi, (case, groups) in enumerate(items, start=1):
            xid = f"{did}X{xi}"
            lines.append(f'{did} --- {xid}["{_case_label(case)}"]:::caseNode')
            _course_subtree(lines, xid, xid, groups)
    return "\n".join([*lines, EDGE_STYLE])


# ---------- 可交互HTML（pyvis） ----------

def render_summary_html(db) -> str:
    """汇总知识图谱的可交互版本：vis.js分层布局（从左到右四层），节点可拖拽、可缩放，
    文字完整显示（按字数换行，不截断）"""
    net = Network(height="850px", width="100%", bgcolor="#ffffff", font_color="#2b2622",
                  cdn_resources="in_line", directed=True)

    def add(node_id, text_lines, level, kind):
        fill, stroke, color = LAYER_STYLE[kind]
        net.add_node(
            node_id, label="\n".join(text_lines), level=level, shape="box",
            color={"background": fill, "border": stroke}, font={"color": color, "size": 12},
            borderWidth=1, shapeProperties={"borderRadius": 4}, margin=6,
        )

    reason = summary_empty_reason(db)
    if reason:
        add("empty", _wrap_lines(reason, 30), 0, "kp")
    for di, (dim, items) in enumerate(summary_graph_data(db), start=1):
        did = f"D{di}"
        add(did, [dim], 0, "dimension")
        for xi, (case, groups) in enumerate(items, start=1):
            xid = f"{did}X{xi}"
            add(xid, [f"案例 {case.case_code}", *_wrap_lines(case.title, WRAP["case"])], 1, "case")
            net.add_edge(did, xid)
            for ci, (course_name, chapters) in enumerate(groups, start=1):
                cid = f"{xid}C{ci}"
                add(cid, _wrap_lines(f"课程：{course_name}", WRAP["course"]), 2, "course")
                net.add_edge(xid, cid)
                for hi, (chapter, points) in enumerate(chapters, start=1):
                    hid = f"{cid}H{hi}"
                    add(hid, _wrap_lines(chapter, WRAP["chapter"]), 3, "chapter")
                    net.add_edge(cid, hid)
                    for ki, point in enumerate(points, start=1):
                        kid = f"{hid}K{ki}"
                        add(kid, _wrap_lines(point, WRAP["kp"]), 4, "kp")
                        net.add_edge(hid, kid)

    net.set_options("""{
      "layout": {"hierarchical": {"enabled": true, "direction": "LR", "sortMethod": "directed",
                                  "levelSeparation": 240, "nodeSpacing": 18, "treeSpacing": 30}},
      "physics": {"enabled": false},
      "edges": {"arrows": {"to": {"enabled": false}}, "color": "#94A3B8", "width": 1,
                "smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.4}},
      "interaction": {"dragNodes": true, "zoomView": true, "dragView": true}
    }""")
    return net.generate_html(notebook=False)
