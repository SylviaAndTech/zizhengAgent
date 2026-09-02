"""
知识点抽取（启发式，来自上传的课程教学大纲文档）
与 案例↔知识点 匹配：
  向量粗筛(ANN检索) + 关键词粗筛(BM25) 两路RRF融合 + Qwen LLM复核精排（结合教学语境重新打分、给融入建议）

两路粗筛为什么都要：案例正文是叙事体（人物/事件/引语），知识点描述是教学大纲里的技术条目，
两种文本语域差异很大，纯向量相似度实测下来区分度很弱（真实案例测过：语义完全不相关的候选
跟真正相关的候选，cosine分数常常只差0.02~0.05）。加一路BM25关键词检索能兜住"案例正文里
直接出现了知识点描述里的原词（比如都提到'云计算'），但整体语义分数没能体现出来"这类情况，
两路各有侧重，用RRF（只看排名不看具体分数值，两路分数量纲完全不同没法直接比）融合。
"""
import hashlib
import json
import logging
import os
import re

import fitz  # PyMuPDF
import jieba
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi

from llama_index_setup import configure_llama_index
from ocr_utils import doc_is_scanned, native_page_lines, ocr_page_lines
from parse_document import extract_text_from_docx, extract_text_from_pdf
from prompts import KNOWLEDGE_MATCH_SYSTEM_PROMPT, build_knowledge_match_prompt
from qwen_client import get_client, require_api_key, CHAT_MODEL
from syllabus_table_ocr import extract_units_and_points
from syllabus_vision_ocr import extract_units_and_points as extract_units_and_points_vision

load_dotenv()

logger = logging.getLogger("uvicorn.error")

CHROMA_HOST = os.environ.get("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
COLLECTION_NAME = "knowledge_point_vectors"

MIN_KP_CHARS = 6
# 扫描件走的syllabus_vision_ocr现在整格转录"教学内容"（含编号、重点/难点），不再拆成
# 单条短语，同一个知识单元的描述可能有小几百字；这个上限是"过滤明显异常的超长内容"用的
# 兜底，不是"内容应该多短"的期望值，给足够宽裕的余量，避免正常的完整格子内容被误伤过滤掉
MAX_KP_CHARS = 800
MAX_QUERY_CHARS = 2000  # DashScope embedding接口对单条文本长度有限制，案例正文可能很长，检索前要截断

COARSE_TOP_K = 15       # 向量/关键词粗筛阶段各自保留的候选数，融合后再交给LLM复核
FINE_TOP_K = 8          # 最终展示给用户的候选数
RRF_K = 60              # RRF融合常数，业界常用默认值，不是需要为这个项目单独调的参数

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


def _kp_embedding_text(course_name: str, chapter: str | None, description: str) -> str:
    """知识点用于embedding/BM25检索的文本——course_name+chapter+description拼起来，
    比单独用description本身多一点语境。注意：这个拼接只影响"喂给检索的文本"，不影响
    description字段本身的存储/展示——description必须保持逐字忠实（下游"适用课程举例"/
    "教学设计"生成都依赖这一点），不能为了检索效果反过来污染这个字段"""
    return f"{course_name} {chapter or ''} {description}".strip()


def index_knowledge_point(kp) -> None:
    """把一条知识点存入向量库（course_name+chapter+description拼接后embedding）。
    kp: KnowledgePoint对象。每条知识点本身就是抽取阶段已经拆好的一条原子描述，
    不用再切块，一条对应一个向量"""
    node = TextNode(
        id_=str(kp.id),
        text=_kp_embedding_text(kp.course_name, kp.chapter, kp.description),
        metadata={"course_name": kp.course_name, "chapter": kp.chapter or ""},
    )
    _get_index().insert_nodes([node])
    _invalidate_bm25_cache()


def remove_knowledge_point_from_index(kp_id: int) -> None:
    _get_index().vector_store.delete_nodes([str(kp_id)])
    _invalidate_bm25_cache()


def reindex_knowledge_point(kp) -> None:
    """编辑知识点描述后重新建索引。Chroma对已存在的id是"跳过不写"，不是"覆盖更新"——
    实测过，同一个id直接insert_nodes()两次，向量库里存的还是第一次写入的旧文本，
    所以编辑后必须先删掉旧向量再插入新的，不能指望insert_nodes自己处理更新语义"""
    remove_knowledge_point_from_index(kp.id)
    index_knowledge_point(kp)


_bm25_cache: tuple | None = None  # (BM25Okapi实例或None, [kp_id,...])，None表示需要重建


def _invalidate_bm25_cache() -> None:
    """知识点库有变更（新增/删除/编辑）时调用，逼下次匹配请求重新从MySQL建一次BM25索引。
    不在每次匹配请求里都重新对全量语料分词——语料涨到几万条时，现场分词会成为明显的
    延迟来源；缓存把这块开销从"每次匹配"移到"每次知识点库变更"（后者频率低得多）"""
    global _bm25_cache
    _bm25_cache = None


def _tokenize_for_bm25(text: str) -> list[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip()]


def _get_bm25_index(db) -> tuple:
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    from db import KnowledgePoint
    points = db.query(KnowledgePoint).all()
    kp_ids = [kp.id for kp in points]
    corpus = [
        _tokenize_for_bm25(_kp_embedding_text(kp.course_name, kp.chapter, kp.description))
        for kp in points
    ]
    bm25 = BM25Okapi(corpus) if corpus else None
    _bm25_cache = (bm25, kp_ids)
    return _bm25_cache


def _rrf_fuse(*ranked_id_lists: list[int], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion：只看每个候选在各路检索结果里的排名，不看具体分数值——
    向量cosine分数和BM25分数完全不是一个量纲，直接加权求和没有意义，RRF不需要为
    分数做归一化/调参，更稳健。返回按融合分数降序排列的候选id列表"""
    scores: dict[int, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, kp_id in enumerate(ranked_ids):
            scores[kp_id] = scores.get(kp_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def _ensure_topic_keywords(case) -> str:
    """案例↔知识点向量检索的查询文本用"主题关键词"而不是完整正文（叙事体语域跟知识点
    技术条目差异太大，见模块顶部说明）。懒生成+按title/full_narrative内容哈希缓存：
    只有正文实际变了才重新调用LLM提炼一次，不是每次匹配请求都调用。
    这个函数只改case对象的字段值（topic_keywords/topic_keywords_source_hash），不落库，
    commit交给调用方——跟enrich_case_from_accepted_mappings是同一个模式。
    LLM调用失败时不缓存失败状态（不写source_hash，下次匹配还会重试），这次退化用
    "标题+正文"本身当查询文本，不让匹配功能因为这一步失败就彻底不可用"""
    source = f"{case.title or ''}\n{case.full_narrative or ''}"
    current_hash = hashlib.md5(source.encode("utf-8")).hexdigest()

    if case.topic_keywords and case.topic_keywords_source_hash == current_hash:
        return case.topic_keywords

    from generate_case import extract_topic_keywords
    try:
        keywords = extract_topic_keywords(case.title or "", case.full_narrative or "")
    except Exception as e:
        logger.warning(f"案例(id={case.id})提炼主题关键词失败，本次匹配退化用原始正文做向量检索: {e}")
        return source

    case.topic_keywords = keywords
    case.topic_keywords_source_hash = current_hash
    return keywords


def match_case_to_knowledge(db, case) -> list[dict]:
    """
    三阶段匹配：
    1. 向量粗筛：把案例的"主题关键词"（不是完整正文，见_ensure_topic_keywords）丢给
       LlamaIndex(ChromaDB)做ANN检索，从知识点向量库里召回候选集
    2. 关键词粗筛：把案例正文丢给BM25，召回字面/术语重合的候选集
    3. 两路用RRF融合，交给Qwen LLM复核精排：结合教学语境重新打分、给出具体的融入方式
       建议（比粗筛信号更懂"是否真的适合当案例引入"）
    返回 [{"knowledge_point": KnowledgePoint对象, "relevance_score": 0-100, "suggestion_text": str}, ...]，按分数降序
    """
    require_api_key()

    if not (case.title or "").strip() and not (case.full_narrative or "").strip():
        raise ValueError("案例还没有正文内容，无法匹配知识点")

    query_keywords = _ensure_topic_keywords(case)
    case_full_text = f"{case.title or ''}\n{case.full_narrative or ''}".strip()

    retriever = _get_index().as_retriever(similarity_top_k=COARSE_TOP_K)
    vector_hits = [h for h in retriever.retrieve(query_keywords[:MAX_QUERY_CHARS]) if (h.score or 0) > 0]
    vector_ranked_ids = [int(h.node.id_) for h in vector_hits]

    bm25, bm25_kp_ids = _get_bm25_index(db)
    keyword_ranked_ids = []
    if bm25 is not None and bm25_kp_ids:
        bm25_scores = bm25.get_scores(_tokenize_for_bm25(case_full_text[:MAX_QUERY_CHARS]))
        ranked_pairs = sorted(zip(bm25_kp_ids, bm25_scores), key=lambda p: p[1], reverse=True)
        keyword_ranked_ids = [kp_id for kp_id, score in ranked_pairs[:COARSE_TOP_K] if score > 0]

    fused_ids = _rrf_fuse(vector_ranked_ids, keyword_ranked_ids)[:COARSE_TOP_K]
    if not fused_ids:
        return []

    from db import KnowledgePoint
    kp_by_id = {kp.id: kp for kp in db.query(KnowledgePoint).filter(KnowledgePoint.id.in_(fused_ids)).all()}

    vector_id_set = set(vector_ranked_ids)
    keyword_id_set = set(keyword_ranked_ids)

    candidates = []
    candidate_payload = []
    for kp_id in fused_ids:
        kp = kp_by_id.get(kp_id)
        if not kp:
            continue
        matched_by = []
        if kp_id in vector_id_set:
            matched_by.append("向量语义")
        if kp_id in keyword_id_set:
            matched_by.append("关键词")
        candidates.append(kp)
        candidate_payload.append({
            "id": kp.id, "course_name": kp.course_name, "chapter": kp.chapter,
            "description": kp.description, "matched_by": matched_by,
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

    courses = enrichment.get("applicable_courses")
    if isinstance(courses, list) and len(courses) == len(accepted_payload):
        # 模型按提示词要求，为每条已采纳知识点按顺序各输出一条适用课程举例；这里直接按位置
        # 把知识点描述原文逐字复原回对应行（供前端画树状图用的第四层叶子节点），不依赖模型
        # 自己把description重新抄一遍——那样容易复述走样，见_kp_embedding_text()的注释
        for row, mapping in zip(courses, accepted_payload):
            if isinstance(row, dict):
                row["知识点简述"] = mapping["description"]

    for field in ("applicable_courses", "teaching_design"):
        if field not in enrichment:
            continue
        old_value = getattr(case, field, None)
        new_value = json.dumps(enrichment[field], ensure_ascii=False)
        if old_value != new_value:
            changes[field] = {"old": old_value, "new": new_value}
        setattr(case, field, new_value)

    return changes
