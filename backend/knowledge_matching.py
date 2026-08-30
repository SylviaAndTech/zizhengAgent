"""
知识点抽取（启发式，来自上传的课程教学大纲文档）
与 案例↔知识点 匹配：LlamaIndex(ChromaDB)向量粗筛(ANN检索) + Qwen LLM复核精排（结合教学语境重新打分、给融入建议）
"""
import json
import os
import re

import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index_setup import configure_llama_index
from ocr_utils import doc_is_scanned, native_page_lines, ocr_page_lines
from parse_document import extract_text_from_docx, extract_text_from_pdf
from prompts import KNOWLEDGE_MATCH_SYSTEM_PROMPT, build_knowledge_match_prompt
from qwen_client import get_client, require_api_key, CHAT_MODEL
from syllabus_table_ocr import extract_units_and_points
from syllabus_vision_ocr import extract_units_and_points as extract_units_and_points_vision

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
# "一、课程基本信息"里的"课程名称：xxx"字段——高校OBE大纲里几乎都有，比封面装饰性大标题
# 靠谱得多：封面大标题常是艺术字/分行排版，OCR按坐标排序时顺序经常被打乱，"《...》"书名号
# 正则容易把旁边不相关的文字（比如"课程教学大纲"这几个字）也框进去。这个字段后面通常紧跟
# "/English翻译"或下一个编号字段（"3.课程类别"），用来界定课程名到哪里结束
_COURSE_NAME_FIELD_RE = re.compile(r"课程名称?[，,：:]\s*([^/]{2,40}?)(?=/|[0-9]{1,2}[.\、]|$)")


def _course_name_from_field(blob: str) -> str | None:
    """从"课程名称：xxx"结构化字段里提取课程名，取不到返回None交给调用方走其他猜测方式"""
    m = _COURSE_NAME_FIELD_RE.search(blob)
    if not m:
        return None
    name = m.group(1).strip(" 　:：，,。")
    return name or None


def guess_course_name(filename: str, lines: list[tuple[str, bool]]) -> str:
    """从文档前几行的标题/正文里猜课程名，猜不出就退化用文件名"""
    field_name = _course_name_from_field("".join(text for text, _ in lines[:20]))
    if field_name:
        return field_name

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


def _resolve_course_name(course_name_override, filename, lines) -> str:
    return (
        course_name_override.strip()
        if course_name_override and course_name_override.strip()
        else guess_course_name(filename, lines)
    )


def _points_from_heading_lines(course_name: str, lines: list[tuple[str, bool]]) -> list[dict]:
    """按标题分节：标题行更新当前章节，非标题行落在长度区间内的作为候选知识点"""
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
    return points


def extract_knowledge_points(
    filename: str, file_path: str, course_name_override: str | None = None
) -> tuple[str, list[dict]]:
    """
    file_path: 磁盘上的临时文件路径（调用方负责上传落盘和之后的清理），不是读进内存的bytes——
    大纲PDF/Word体积可能到几百MB，整个读进内存在小内存服务器上容易顶不住。
    course_name_override 非空时直接采用（适合同一门课程分多个文件上传）；
    为空则自动从文档标题/文件名猜课程名（适合一次批量上传多门课程的大纲）。
    返回 (最终使用的课程名, 知识点列表)

    PDF分两条路径：
    - 正常数字PDF（有文字层）：先尝试按坐标分列识别"知识单元/教学内容(知识点)"表格
      （syllabus_table_ocr，免费、不用等大模型）；识别不出这种表格结构就退回到按标题
      样式分节的启发式（跟docx共用同一套逻辑）。
    - 扫描版PDF（没有文字层，比如打印后扫描/拍照转的PDF）：这类没有可靠的"标题样式"信息，
      按坐标分列也常常因为窄列文字被拉伸变形而认错（尤其是英文单词）。改用
      syllabus_vision_ocr：先用免费OCR定位"知识单元/实验单元"这两张表格分别在哪几页，
      只把这几页截图发给视觉大模型识别——模型能结合上下文语义纠错，比逐字符OCR准得多。
    """
    lower_name = filename.lower()
    if lower_name.endswith(".docx"):
        lines = extract_text_from_docx(file_path)
        course_name = _resolve_course_name(course_name_override, filename, lines)
        return course_name, _points_from_heading_lines(course_name, lines)

    if not lower_name.endswith(".pdf"):
        raise ValueError("仅支持 .docx 或 .pdf 文件")

    doc = fitz.open(file_path)
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError("PDF已加密，无法在没有密码的情况下解析")

        scanned = doc_is_scanned(doc)

        if scanned:
            try:
                units_points = extract_units_and_points_vision(doc)
            except Exception as e:
                raise ValueError(f"这是一份扫描版PDF（没有文字层），表格识别失败: {e}") from e
            course_name = _course_name_from_pdf_title(doc, ocr_page_lines, course_name_override, filename)
            points = [
                {"course_name": course_name, "chapter": chapter, "description": description}
                for chapter, description in units_points
                if MIN_KP_CHARS <= len(description) <= MAX_KP_CHARS
            ]
            return course_name, points

        # 数字PDF：先试免费的按坐标分列，识别不出表格结构就静默回退到标题分节启发式
        try:
            units_points = extract_units_and_points(doc, native_page_lines)
        except Exception:
            units_points = []

        if units_points:
            course_name = _course_name_from_pdf_title(doc, native_page_lines, course_name_override, filename)
            points = [
                {"course_name": course_name, "chapter": chapter, "description": description}
                for chapter, description in units_points
                if MIN_KP_CHARS <= len(description) <= MAX_KP_CHARS
            ]
            return course_name, points
    finally:
        doc.close()

    # 非扫描件、且没识别出"知识单元/知识点"表格结构的PDF：退回到原来的标题分节启发式
    lines = extract_text_from_pdf(file_path)
    course_name = _resolve_course_name(course_name_override, filename, lines)
    return course_name, _points_from_heading_lines(course_name, lines)


def _course_name_from_pdf_title(doc, get_page_lines, course_name_override, filename) -> str:
    if course_name_override and course_name_override.strip():
        return course_name_override.strip()
    first_page_texts = [text for *_, text in get_page_lines(doc[0])] if len(doc) else []
    return _guess_course_name_from_page_lines(first_page_texts, filename)


def _guess_course_name_from_page_lines(line_texts: list[str], filename: str) -> str:
    """给按坐标定位的表格解析路径用课程名：这条路径没有"标题样式"这种概念，优先找
    "一、课程基本信息"里的"课程名称：xxx"字段（结构化、可靠），找不到再退化找封面书名号
    （艺术字封面标题OCR顺序常被打乱，容易框进不相关文字，只作为兜底），
    再找不到就退化用guess_course_name的文件名兜底逻辑"""
    blob = "".join(line_texts[:20])
    field_name = _course_name_from_field(blob)
    if field_name:
        return field_name
    book_match = _BOOK_TITLE_RE.search(blob)
    if book_match:
        return book_match.group(1).strip()
    return guess_course_name(filename, [(t, False) for t in line_texts])


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


def remove_knowledge_point_from_index(kp_id: int) -> None:
    _get_index().vector_store.delete_nodes([str(kp_id)])


def reindex_knowledge_point(kp) -> None:
    """编辑知识点描述后重新建索引。Chroma对已存在的id是"跳过不写"，不是"覆盖更新"——
    实测过，同一个id直接insert_nodes()两次，向量库里存的还是第一次写入的旧文本，
    所以编辑后必须先删掉旧向量再插入新的，不能指望insert_nodes自己处理更新语义"""
    remove_knowledge_point_from_index(kp.id)
    index_knowledge_point(kp)


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


def enrich_case_from_accepted_mappings(db, case) -> dict:
    """
    案例当前"已采纳"的知识点关联集合发生变化后调用（不管是新采纳了一条、还是取消采纳了
    一条，都要调这个函数重新算一遍）：
    - 如果当前一条"已采纳"的关联都没有，"适用课程举例"和"教学设计"清空回"尚未生成"状态
      （不留旧内容——旧内容对应的知识点可能已经不再是"已采纳"状态了，留着会误导）。
    - 否则重新生成这两个字段：适用课程举例对已采纳的每一条知识点关联都生成一条；
      教学设计只结合relevance_score最高的那一条来设计，不是把全部已采纳的糅在一起
      （多个不相关的知识点糅进一份教学设计里会变得空泛、失焦）。

    调用方负责commit/refresh/写审计日志——这个函数只改case对象的字段值、返回变化了
    哪些字段（供写审计日志用），不落库、不处理事务边界。
    返回值：{字段名: {"old":.., "new":..}, ...}，没有实际变化时返回空dict。
    """
    from db import CaseKnowledgeMapping  # 延迟import，避免和db.py出现循环依赖

    accepted = (
        db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id == case.id, CaseKnowledgeMapping.status == "已采纳")
        .order_by(CaseKnowledgeMapping.relevance_score.desc())
        .all()
    )
    accepted_payload = [
        {
            "course_name": m.knowledge_point.course_name,
            "chapter": m.knowledge_point.chapter,
            "description": m.knowledge_point.description,
            "suggestion_text": m.suggestion_text,
            "relevance_score": m.relevance_score,
        }
        for m in accepted
        if m.knowledge_point
    ]

    changes = {}
    if not accepted_payload:
        for field in ("applicable_courses", "teaching_design"):
            old_value = getattr(case, field, None)
            if old_value is not None:
                changes[field] = {"old": old_value, "new": None}
            setattr(case, field, None)
        return changes

    from generate_case import enrich_case_with_knowledge
    enrichment = enrich_case_with_knowledge(case.to_dict(), accepted_payload)

    for field in ("applicable_courses", "teaching_design"):
        if field not in enrichment:
            continue
        old_value = getattr(case, field, None)
        new_value = json.dumps(enrichment[field], ensure_ascii=False)
        if old_value != new_value:
            changes[field] = {"old": old_value, "new": new_value}
        setattr(case, field, new_value)

    return changes
