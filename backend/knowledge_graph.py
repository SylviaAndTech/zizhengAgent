"""
知识图谱：基于"已采纳"状态的案例、以及案例已采纳的知识点关联，
构建 维度 → 案例 → 知识点 三层关系图，导出静态图片或可交互HTML。
"""
import io

import matplotlib
matplotlib.use("Agg")  # 无GUI环境下渲染，不依赖系统显示后端
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

from db import Case, CaseKnowledgeMapping, DIMENSIONS

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "SimHei", "Microsoft YaHei",
]
plt.rcParams["axes.unicode_minus"] = False

NODE_COLORS = {
    "dimension": "#b3282d",
    "case": "#c17817",
    "knowledge": "#3a7d44",
}


def build_graph(db) -> nx.Graph:
    """
    只纳入"已采纳"状态的案例、以及它们"已采纳"的知识点关联——
    图谱反映的应该是审核通过的成果，不是一堆待定的草稿和推荐。
    """
    g = nx.Graph()
    for d in DIMENSIONS:
        g.add_node(f"dim:{d}", label=d, type="dimension")

    cases = db.query(Case).filter(Case.status == "已采纳").all()
    for c in cases:
        case_node = f"case:{c.id}"
        label = f"{c.case_code} {c.title or ''}"[:24]
        g.add_node(case_node, label=label, type="case")
        if c.dimension and g.has_node(f"dim:{c.dimension}"):
            g.add_edge(f"dim:{c.dimension}", case_node)

    mappings = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.status == "已采纳")
        .all()
    )
    for m in mappings:
        case_node = f"case:{m.case_id}"
        if not g.has_node(case_node):
            continue  # 案例本身不是"已采纳"状态，不纳入图谱
        kp = m.knowledge_point
        if not kp:
            continue
        kp_node = f"kp:{kp.id}"
        label = f"{kp.course_name} {(kp.chapter or '')}"[:24]
        g.add_node(kp_node, label=label, type="knowledge")
        g.add_edge(case_node, kp_node)

    return g


def render_graph_png(g: nx.Graph) -> bytes:
    fig, ax = plt.subplots(figsize=(12, 9))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "还没有已采纳的案例/知识点关联，暂无图谱数据", ha="center", va="center")
        ax.axis("off")
    else:
        pos = nx.spring_layout(g, k=0.7, seed=42)
        colors = [NODE_COLORS.get(g.nodes[n]["type"], "#888") for n in g.nodes]
        labels = {n: g.nodes[n]["label"] for n in g.nodes}
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#ccc")
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=900)
        nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=8)
        ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_graph_html(g: nx.Graph) -> str:
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#2b2622", cdn_resources="in_line")
    for n, data in g.nodes(data=True):
        net.add_node(n, label=data.get("label", n), color=NODE_COLORS.get(data.get("type"), "#888"))
    for u, v in g.edges():
        net.add_edge(u, v)
    net.repulsion(node_distance=180, spring_length=180)
    return net.generate_html(notebook=False)
