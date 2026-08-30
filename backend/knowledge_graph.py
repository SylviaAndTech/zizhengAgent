"""
知识图谱：基于"已采纳"状态的案例、以及案例已采纳的知识点关联，构建 案例↔知识点 两层关系图，
导出静态图片或可交互HTML。

节点/关系用 langchain_community.graphs.graph_document 里的 Node/Relationship/GraphDocument
表示——这是LangChain的LLMGraphTransformer的标准输出schema，这里借用同一套数据结构来表达图，
但不实际调用LLM：案例和知识点的关联本来就是CaseKnowledgeMapping表里已经确定的结构化数据，
不需要（也不应该）靠LLM从文本里再"猜"一遍已经明确知道的关系。

知识点节点的显示名称（课程名+章节）经常很长，图上全量显示会让节点互相压字、很难看清，
所以knowledge point节点的label会截断成短名字；截断前的完整名字放进节点属性里，可交互HTML
版本用pyvis原生支持的title属性做hover提示（vis.js的label和title是两个独立属性，鼠标悬停
才显示title，不用额外写JS）；静态PNG是一张位图，没有"悬停"这回事，只能一直显示截断后的
短名字，这是PNG这个媒介本身的局限，不是遗漏。
"""
import io

import matplotlib
matplotlib.use("Agg")  # 无GUI环境下渲染，不依赖系统显示后端
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_core.documents import Document

from db import Case, CaseKnowledgeMapping

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "SimHei", "Microsoft YaHei",
]
plt.rcParams["axes.unicode_minus"] = False

NODE_COLORS = {
    "Case": "#c17817",
    "KnowledgePoint": "#3a7d44",
}

KP_LABEL_MAX_LEN = 12  # 知识点显示名称的截断长度；案例名称不截断，按要求完整显示


def build_graph_document(db) -> GraphDocument:
    """
    只纳入"已采纳"状态的案例、以及它们"已采纳"的知识点关联——
    图谱反映的应该是审核通过的成果，不是一堆待定的草稿和推荐。
    """
    nodes: dict[str, Node] = {}
    relationships: list[Relationship] = []

    cases = db.query(Case).filter(Case.status == "已采纳").all()
    for c in cases:
        case_id = f"case:{c.id}"
        full_label = f"{c.case_code} {c.title or ''}".strip()
        nodes[case_id] = Node(
            id=case_id, type="Case",
            properties={"label": full_label, "full_label": full_label},
        )

    mappings = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.status == "已采纳")
        .all()
    )
    for m in mappings:
        case_id = f"case:{m.case_id}"
        if case_id not in nodes:
            continue  # 案例本身不是"已采纳"状态，不纳入图谱
        kp = m.knowledge_point
        if not kp:
            continue
        kp_id = f"kp:{kp.id}"
        full_label = f"{kp.course_name} {(kp.chapter or '')}".strip()
        short_label = full_label if len(full_label) <= KP_LABEL_MAX_LEN else full_label[:KP_LABEL_MAX_LEN] + "…"
        if kp_id not in nodes:
            nodes[kp_id] = Node(
                id=kp_id, type="KnowledgePoint",
                properties={"label": short_label, "full_label": full_label},
            )
        relationships.append(Relationship(source=nodes[case_id], target=nodes[kp_id], type="COVERS"))

    return GraphDocument(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=Document(page_content="已采纳案例与已采纳知识点关联"),
    )


# 保留旧函数名作为别名，避免调用方还没来得及跟着改名字就直接报错
build_graph = build_graph_document


def _to_networkx(graph_doc: GraphDocument) -> nx.Graph:
    """GraphDocument只是数据结构，不负责画图；实际渲染还是靠networkx+matplotlib/pyvis，
    这里做一次轻量转换，把LangChain的Node/Relationship铺进networkx.Graph里"""
    g = nx.Graph()
    for n in graph_doc.nodes:
        g.add_node(
            n.id,
            label=n.properties.get("label", n.id),
            full_label=n.properties.get("full_label", n.id),
            type=n.type,
        )
    for r in graph_doc.relationships:
        g.add_edge(r.source.id, r.target.id)
    return g


def render_graph_png(graph_doc: GraphDocument) -> bytes:
    g = _to_networkx(graph_doc)
    fig, ax = plt.subplots(figsize=(12, 9))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "还没有已采纳的案例/知识点关联，暂无图谱数据", ha="center", va="center")
        ax.axis("off")
    else:
        pos = nx.spring_layout(g, k=0.7, seed=42)
        colors = [NODE_COLORS.get(g.nodes[n]["type"], "#888") for n in g.nodes]
        labels = {n: g.nodes[n]["label"] for n in g.nodes}  # 静态图没法hover，只能一直显示截断后的短名字
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#ccc")
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=900)
        nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=8)
        ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_graph_html(graph_doc: GraphDocument) -> str:
    g = _to_networkx(graph_doc)
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#2b2622", cdn_resources="in_line")
    for n, data in g.nodes(data=True):
        # label是节点上直接显示的文字（知识点节点是截断后的短名字，案例节点是完整名字）；
        # title是鼠标hover时弹出的提示条，放完整名字——vis.js原生支持这两个独立属性，不用额外写JS
        net.add_node(n, label=data["label"], title=data["full_label"], color=NODE_COLORS.get(data["type"], "#888"))
    for u, v in g.edges():
        net.add_edge(u, v)
    net.repulsion(node_distance=180, spring_length=180)
    return net.generate_html(notebook=False)
