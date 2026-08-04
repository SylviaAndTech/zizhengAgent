"""
知识点抽取（启发式，来自上传的课程教学大纲文档）
与 案例↔知识点 匹配：LlamaIndex(ChromaDB)向量粗筛(ANN检索) + Qwen LLM复核精排（结合教学语境重新打分、给融入建议）
"""
import json
import os
import re

import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index_setup import configure_llama_index
from parse_document import extract_text_from_docx, extract_text_from_pdf
from prompts import KNOWLEDGE_MATCH_SYSTEM_PROMPT, build_knowledge_match_prompt
from qwen_client import get_client, require_api_key, CHAT_MODEL

load_dotenv()

CHROMA_HOST = os.environ.get("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
COLLECTION_NAME = "knowledge_point_vectors"

MIN_KP_CHARS = 6
MAX_KP_CHARS = 200
MAX_QUERY_CHARS = 2000  # DashScope embedding接口对单条文本长度有限制，案例正文可能很长，检索前要截断

COARSE_TOP_K = 15       # 向量粗筛阶段保留的候选数，交给LLM复核
FINE_TOP_K = 8          # 最终展示给用户的候选数

# 大纲标题里常见的"课程名+大纲"套话，用于从标题里剥离出干净的课程名
_COURSE_SUFFIX_RE = re.compile(r"(课程)?(教学)?大纲.*$")
_BOOK_TITLE_RE = re.compile(r"《([^》]+)》")


def guess_course_name(filename: str, lines: list[tuple[str, bool]]) -> str:
    """从文档前几行的标题/正文里猜课程名，猜不出就退化用文件名"""
    for text, is_heading in lines[:5]:
        candidate = text.strip()
        if not candidate:
            continue
        book_match = _BOOK_TITLE_RE.search(candidate)
        if book_match:
            return book_match.group(1).strip()
        if is_heading and len(candidate) <= 30:
            cleaned = _COURSE_SUFFIX_RE.sub("", candidate).strip("《》 　:：-")
            if cleaned:
                return cleaned

    base = re.sub(r"\.(docx|pdf)$", "", filename, flags=re.I)
    base = _COURSE_SUFFIX_RE.sub("", base).strip("《》 　:：-_")
    return base or filename


def extract_knowledge_points(
    filename: str, file_bytes: bytes, course_name_override: str | None = None
) -> tuple[str, list[dict]]:
    """
    把大纲文档按标题分节，非标题的正文行作为候选知识点条目。
    course_name_override 非空时直接采用（适合同一门课程分多个文件上传）；
    为空则自动从文档标题/文件名猜课程名（适合一次批量上传多门课程的大纲）。
    返回 (最终使用的课程名, 知识点列表)
    """
    lower_name = filename.lower()
    if lower_name.endswith(".docx"):
        lines = extract_text_from_docx(file_bytes)
    elif lower_name.endswith(".pdf"):
        lines = extract_text_from_pdf(file_bytes)
    else:
        raise ValueError("仅支持 .docx 或 .pdf 文件")

    course_name = (
        course_name_override.strip()
        if course_name_override and course_name_override.strip()
        else guess_course_name(filename, lines)
    )

    points = []
    current_chapter = None
    for text, is_heading in lines:
        if is_heading:
            current_chapter = text
            continue
        if MIN_KP_CHARS <= len(text) <= MAX_KP_CHARS:
            points.append({
                "course_name": course_name,
                "chapter": current_chapter,
                "description": text,
            })
    return course_name, points


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


def index_knowledge_point(kp) -> None:
    """把一条知识点的描述向量化存入知识点向量库。kp: KnowledgePoint对象。
    每条知识点本身就是抽取阶段已经拆好的一条原子描述，不用再切块，一条对应一个向量"""
    node = TextNode(
        id_=str(kp.id),
        text=kp.description,
        metadata={"course_name": kp.course_name, "chapter": kp.chapter or ""},
    )
    _get_index().insert_nodes([node])


def match_case_to_knowledge(db, case) -> list[dict]:
    """
    两阶段匹配：
    1. 向量粗筛：把案例正文丢给LlamaIndex(ChromaDB)做ANN检索，从知识点向量库里召回候选集
    2. LLM复核精排：Qwen结合教学语境重新打分、给出具体的融入方式建议（比纯向量分数更懂"是否真的适合当案例引入"）
    返回 [{"knowledge_point": KnowledgePoint对象, "relevance_score": 0-100, "suggestion_text": str}, ...]，按分数降序
    """
    require_api_key()

    case_text = f"{case.title or ''}\n{case.full_narrative or ''}".strip()
    if not case_text:
        raise ValueError("案例还没有正文内容，无法匹配知识点")

    retriever = _get_index().as_retriever(similarity_top_k=COARSE_TOP_K)
    hits = [h for h in retriever.retrieve(case_text[:MAX_QUERY_CHARS]) if (h.score or 0) > 0]
    if not hits:
        return []

    from db import KnowledgePoint
    kp_ids = [int(h.node.id_) for h in hits]
    kp_by_id = {kp.id: kp for kp in db.query(KnowledgePoint).filter(KnowledgePoint.id.in_(kp_ids)).all()}

    candidates = []
    candidate_payload = []
    for h in hits:
        kp = kp_by_id.get(int(h.node.id_))
        if not kp:
            continue
        candidates.append(kp)
        candidate_payload.append({
            "id": kp.id, "course_name": kp.course_name, "chapter": kp.chapter,
            "description": kp.description, "coarse_score": round((h.score or 0) * 100),
        })
    if not candidates:
        return []

    prompt = build_knowledge_match_prompt(case.title or "", case.full_narrative or "", candidate_payload)

    try:
        response = get_client().chat.completions.create(
            model=CHAT_MODEL,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": KNOWLEDGE_MATCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except openai.APIError as e:
        raise ValueError(f"调用 Qwen API 失败: {str(e)}")

    raw_text = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"知识点复核阶段模型输出不是合法JSON: {e}\n原始输出前500字: {raw_text[:500]}")

    kp_by_id = {kp.id: kp for kp in candidates}
    results = []
    for m in parsed.get("matches", []):
        kp = kp_by_id.get(m.get("knowledge_point_id"))
        if not kp:
            continue
        score = int(m.get("relevance_score") or 0)
        if score <= 0:
            continue
        results.append({
            "knowledge_point": kp,
            "relevance_score": score,
            "suggestion_text": m.get("suggestion_text") or "",
        })

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:FINE_TOP_K]
