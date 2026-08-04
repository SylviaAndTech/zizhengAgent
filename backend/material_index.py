"""
素材库的检索索引：把 RawMaterial 的正文切块、向量化、存入 ChromaDB，
供 AI 助手在聊天里用自然语言检索相关素材（跟"按case_code精确查"是两回事，
用户在聊天里说"有没有关于XX的素材"这种模糊描述时靠这个）。

跟 parse_document.py 的关注点不同：那边是"识别案例边界"，这边是"切块供检索"，
两套切分逻辑不共用。
"""
import os

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index_setup import configure_llama_index

load_dotenv()

CHROMA_HOST = os.environ.get("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
COLLECTION_NAME = "material_chunks"

MAX_QUERY_CHARS = 2000  # DashScope embedding接口对单条文本长度有限制，检索用的query要截断

_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
_index = None


def _get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        configure_llama_index()
        vector_store = ChromaVectorStore(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            collection_name=COLLECTION_NAME,
            collection_kwargs={"metadata": {"hnsw:space": "cosine"}},
        )
        _index = VectorStoreIndex.from_vector_store(vector_store)
    return _index


def index_material(material) -> int:
    """把一条素材的正文切块、向量化，存进素材检索向量库。material: RawMaterial对象。返回索引了几个chunk"""
    text = (material.fetched_text or "").strip()
    if not text:
        return 0
    chunks = _splitter.split_text(text)
    nodes = [
        TextNode(
            id_=f"material-{material.id}-{idx}",
            text=chunk,
            metadata={
                "material_id": material.id,
                "case_code": material.case_code or "",
                "source": material.source_title or material.url or "",
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
    _get_index().insert_nodes(nodes)
    return len(nodes)


def search_materials(query: str, top_k: int = 5, case_code: str | None = None) -> list[dict]:
    """语义检索素材库，返回命中的片段+来源信息；可选按case_code过滤"""
    filters = None
    if case_code:
        filters = MetadataFilters(filters=[MetadataFilter(key="case_code", value=case_code)])
    retriever = _get_index().as_retriever(similarity_top_k=top_k, filters=filters)
    hits = retriever.retrieve(query[:MAX_QUERY_CHARS])
    return [
        {
            "material_id": h.node.metadata.get("material_id"),
            "case_code": h.node.metadata.get("case_code"),
            "source": h.node.metadata.get("source"),
            "snippet": h.node.text[:300],
            "score": round(float(h.score or 0), 3),
        }
        for h in hits
    ]
