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
import unicodedata

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
from syllabus_docx_table import extract_units_and_points as extract_units_and_points_docx
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


def _normalize_cjk_radicals(text: str) -> str:
    """把康熙部首/部首补充区的字符（U+2E80–U+2FDF）换成对应的常用汉字。
    有些大纲是从PDF转回Word的，"计算机网络"里的"网"其实是康熙部首"⽹"(U+2F79)，
    肉眼看不出区别，但存进库里之后按"网络"搜索就搜不到。只转这一段字符，不对整串做
    NFKC——那样会把全角括号"（Java）"也变成半角，改变课程名原本的写法"""
    return "".join(
        unicodedata.normalize("NFKC", ch) if "⺀" <= ch <= "⿟" else ch
        for ch in text
    )


def _course_name_from_field(blob: str) -> str | None:
    """从"课程名称：xxx"结构化字段里提取课程名，取不到返回None交给调用方走其他猜测方式"""
    m = _COURSE_NAME_FIELD_RE.search(blob)
    if not m:
        return None
    # 有的大纲把"中文名/英文名"整体用括号括起来："课程名称：（数据结构/data structure）"，
    # 正则截到"/"为止会剩下一个孤零零的左括号
    name = m.group(1).strip(" 　:：，,。").lstrip("（(").strip()
    return _normalize_cjk_radicals(name) or None


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


# 大纲文件名常见写法："林依天+工业互联网应用系统开发+10830745+2025软件工程V3.docx"、
# "姚家朋-数据结构-10830401-25计算机科学与技术（专升本）.docx"——教师名、课程名、课程代码、
# 年级专业用+/-/_拼起来，顺序不固定。下面这些片段肯定不是课程名，剔除后取剩下的第一段
_FILENAME_SPLIT_RE = re.compile(r"[+＋\-_－—]")
_FILENAME_NOISE_RE = re.compile(r"^\d+$|^(20)?\d{2}(级|届)?|级|专业|大纲|最终版|新版|^V\d|^[一-龥]{2,3}(任课)?$")


def _course_name_from_filename(filename: str) -> str:
    base = re.sub(r"\.(docx|pdf)$", "", filename, flags=re.I).strip()
    book_match = _BOOK_TITLE_RE.search(base)
    if book_match:
        return book_match.group(1).strip()
    for seg in _FILENAME_SPLIT_RE.split(base):
        seg = re.sub(r"\(\d+\)$|（\d+）$", "", seg.strip()).strip()  # "xxx(1).docx"这种副本后缀
        if seg and not _FILENAME_NOISE_RE.search(seg):
            return _COURSE_SUFFIX_RE.sub("", seg).strip() or seg
    return _COURSE_SUFFIX_RE.sub("", base).strip("《》 　:：-_") or filename


def _resolve_docx_course_name(course_name_override, filename, lines) -> tuple[str, list[str]]:
    """Word大纲的课程名：手动填写的优先；否则以正文开头"2.课程名称：xxx/English"字段里
    的中文名为准，并跟文件名核对——文件名里找不到这个课程名时给出提示让用户核实（文件名
    是人手动起的，经常是简称、旧名或者干脆写错，正文字段更可信）。正文里没有这个字段时，
    才退回从文件名提取。返回(课程名, 提示列表)"""
    if course_name_override and course_name_override.strip():
        return course_name_override.strip(), []

    from_filename = _course_name_from_filename(filename)
    from_content = _course_name_from_field("".join(text for text, _ in lines[:20]))
    if not from_content:
        return from_filename, [f"{filename}：正文开头没有找到「课程名称」字段，课程名按文件名识别为「{from_filename}」，请核实"]

    def squash(s):
        return re.sub(r"[\s《》]", "", unicodedata.normalize("NFKC", s)).lower()

    if squash(from_content) in squash(filename):
        return from_content, []
    return from_content, [
        f"{filename}：文件名识别为「{from_filename}」，与文档内课程名称「{from_content}」不一致，已采用「{from_content}」"
    ]


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
) -> tuple[str, list[dict], list[str]]:
    """
    file_path: 磁盘上的临时文件路径（调用方负责上传落盘和之后的清理），不是读进内存的bytes——
    大纲PDF/Word体积可能到几百MB，整个读进内存在小内存服务器上容易顶不住。
    course_name_override 非空时直接采用（适合同一门课程分多个文件上传）；
    为空则自动从文档标题/文件名猜课程名（适合一次批量上传多门课程的大纲）。
    返回 (最终使用的课程名, 知识点列表, 需要提示用户核实的信息列表)

    Word(.docx)：先按表格解析"表5-1 知识单元/教学内容"、"表5-2 实验单元/教学内容"
    （syllabus_docx_table，一个单元一条知识点）；文档里没有这类表格（比如毕业实习大纲）
    才退回到按标题样式分节的启发式。

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
        course_name, warnings = _resolve_docx_course_name(course_name_override, filename, lines)
        units_points = extract_units_and_points_docx(file_path)
        if not units_points:
            return course_name, _points_from_heading_lines(course_name, lines), warnings
        # 表格路径不按MAX_KP_CHARS过滤：一个单元的教学内容整格照录，偶尔会比较长，
        # 悄悄丢掉会让这个单元在知识点库里凭空消失，违背"一个单元一条记录"；照常入库，
        # 只提示一下让用户留意
        for chapter, description in units_points:
            if len(description) > MAX_KP_CHARS:
                warnings.append(f"{filename}：「{chapter}」的教学内容有{len(description)}字，偏长，已完整入库，请留意是否混入了其他内容")
        points = [
            {"course_name": course_name, "chapter": chapter, "description": description}
            for chapter, description in units_points
            if len(description) >= MIN_KP_CHARS
        ]
        return course_name, points, warnings

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
            return course_name, points, []

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
            return course_name, points, []
    finally:
        doc.close()

    # 非扫描件、且没识别出"知识单元/知识点"表格结构的PDF：退回到原来的标题分节启发式
    lines = extract_text_from_pdf(file_path)
    course_name = _resolve_course_name(course_name_override, filename, lines)
    return course_name, _points_from_heading_lines(course_name, lines), []


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


# 教学设计只结合一条知识点生成，记下是哪一条，存在teaching_design这个dict里；以后这条知识点
# 被去掉时才知道教学设计要跟着重新生成，没被去掉就原样保留
TEACHING_DESIGN_BASIS_KEY = "依据知识点ID"


def _load_json(value, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def _mapping_payload(m) -> dict:
    return {
        "course_name": m.knowledge_point.course_name,
        "chapter": m.knowledge_point.chapter,
        "description": m.knowledge_point.description,
        "suggestion_text": m.suggestion_text,
        "relevance_score": m.relevance_score,
    }


def _row_matches_mapping(row: dict, m) -> bool:
    """旧数据的适用课程举例行没有knowledge_point_id，按知识点描述原文（"知识点简述"）
    认领，再没有就按课程名+章节认领"""
    kp = m.knowledge_point
    if row.get("知识点简述"):
        return row["知识点简述"] == kp.description
    return (str(row.get("课程名称") or "").strip("《》 ") == kp.course_name
            and (row.get("适用章节") or None) == kp.chapter)


def _attach_row_ids(courses: list, bound_mappings: list) -> None:
    """给还没有knowledge_point_id的旧行补上ID（一行只认领给一条知识点），之后增删都按ID精确定位"""
    for m in bound_mappings:
        if any(r.get("knowledge_point_id") == m.knowledge_point_id for r in courses):
            continue
        for r in courses:
            if r.get("knowledge_point_id") is None and _row_matches_mapping(r, m):
                r["knowledge_point_id"] = m.knowledge_point_id
                break


def _generate_rows(case, mappings: list) -> tuple[list, dict | None]:
    """为给定的知识点关联（已按相关度降序）各生成一条适用课程举例，顺带生成一份结合第一条
    知识点的教学设计。行里的知识点描述原文和ID直接按位置写回，不依赖模型复述
    （见_kp_embedding_text的注释）——所以模型返回的条数必须跟输入一一对应，对不上宁可
    报错让用户重试，也不能把行错配到别的知识点上（之后按ID删除时会删错）"""
    from generate_case import enrich_case_with_knowledge

    enrichment = enrich_case_with_knowledge(case.to_dict(), [_mapping_payload(m) for m in mappings])
    rows = enrichment.get("applicable_courses")
    if not isinstance(rows, list) or len(rows) != len(mappings) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("模型返回的适用课程举例条数与知识点条数对不上，无法逐条对应，请重试")
    for row, m in zip(rows, mappings):
        # 课程名称/适用章节也用知识点库原文覆盖：提示词要求模型照抄，但模型偶尔会把整段
        # 教学内容塞进"适用章节"里
        row["课程名称"] = m.knowledge_point.course_name
        row["适用章节"] = m.knowledge_point.chapter or ""
        row["知识点简述"] = m.knowledge_point.description
        row["knowledge_point_id"] = m.knowledge_point_id
    td = enrichment.get("teaching_design")
    if isinstance(td, dict):
        td[TEACHING_DESIGN_BASIS_KEY] = mappings[0].knowledge_point_id
    else:
        td = None
    return rows, td


def _fill_graph_points(case, mappings: list, on_stage=None) -> int:
    """给还没提取过知识图谱节点的已写入关联提取一次（见kp_points.py）。提取失败不影响
    写入案例本身——知识图谱只是展示，画图时会用兜底规则，下次同步再补。返回补上的条数"""
    from kp_points import extract_graph_points

    pending = [m for m in mappings if m.knowledge_point and not m.graph_points]
    if not pending:
        return 0
    (on_stage or (lambda _msg: None))("提取知识图谱的知识点")
    try:
        extracted = extract_graph_points(case, pending)
    except Exception as e:
        logger.warning(f"案例{case.id}的知识图谱知识点提取失败，先用兜底规则显示：{e}")
        return 0
    for m in pending:
        if m.id in extracted:
            m.graph_points = json.dumps(extracted[m.id], ensure_ascii=False)
    return len(extracted)


def _apply_binding_changes(db, case, to_add: list, to_remove: list,
                           allow_llm: bool = True, on_stage=None) -> tuple[dict, str]:
    """把to_add里的知识点写进案例、把to_remove里的从案例里删掉，其余已绑定的内容原样不动。
    allow_llm=False：只做不需要调模型的部分（只能删、不能加），用在强制删除知识点这种在HTTP
    请求里同步执行的场景；教学设计需要重新生成时先清空，由调用方另起后台任务补
    （regenerate_missing_teaching_design）。on_stage：调模型前回调一句进度描述，给任务进度条用。
    - 新增：只为新增的这几条调模型生成适用课程举例，追加在原有行后面
    - 删除：直接删掉对应的行，不调模型
    - 教学设计：原样保留；只有它依据的知识点被删掉、或者原本就没有教学设计时，才改用剩下
      已绑定知识点里相关度最高的那条重新生成；一条绑定都不剩了就清空
    同时维护每条关联的bound标记。调用方负责commit和写审计日志（中途模型调用失败抛异常时，
    调用方不commit，bound标记也就跟着回滚，不会出现"标成已绑定、内容却没写进去"）。
    返回 ({字段名: {"old":.., "new":..}}, 给用户看的一句话结果)"""
    from db import CaseKnowledgeMapping  # 延迟import，避免和db.py出现循环依赖

    by_score = lambda m: m.relevance_score or 0  # noqa: E731
    bound_all = [
        m for m in db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id == case.id, CaseKnowledgeMapping.bound.is_(True)).all()
        if m.knowledge_point
    ]
    to_remove = [m for m in to_remove if m.bound]
    to_add = sorted((m for m in to_add if not m.bound and m.knowledge_point), key=by_score, reverse=True)
    remove_ids = {m.knowledge_point_id for m in to_remove}
    still_bound = [m for m in bound_all if m.knowledge_point_id not in remove_ids]
    final_bound = still_bound + to_add

    courses = _load_json(case.applicable_courses, [])
    courses = courses if isinstance(courses, list) else []
    td = _load_json(case.teaching_design, None)
    td = td if isinstance(td, dict) and td else None

    _attach_row_ids(courses, bound_all)
    basis_id = td.get(TEACHING_DESIGN_BASIS_KEY) if td else None
    if td and basis_id is None and bound_all:
        # 旧数据没记依据哪条：旧逻辑一直是用相关度最高的那条已采纳知识点生成教学设计
        basis_id = max(bound_all, key=by_score).knowledge_point_id
        td[TEACHING_DESIGN_BASIS_KEY] = basis_id

    courses = [r for r in courses if r.get("knowledge_point_id") not in remove_ids]
    stage = on_stage or (lambda _msg: None)
    td_note = ""
    if not final_bound:
        new_courses, new_td = None, None
        if td:
            td_note = "，教学设计已清空"
    else:
        new_td = td
        need_td = td is None or basis_id not in {m.knowledge_point_id for m in still_bound}
        candidate = max(final_bound, key=by_score)
        if not allow_llm:
            if to_add:
                raise ValueError("allow_llm=False时不能新增知识点")
            if need_td:
                new_td, td_note = None, "，教学设计待重新生成"
                need_td = False
        if to_add:
            stage(f"生成新增的{len(to_add)}条适用课程举例")
            rows, generated_td = _generate_rows(case, to_add)
            courses.extend(rows)
            if need_td and candidate is to_add[0]:
                new_td, need_td = generated_td, False
                td_note = "，教学设计已重新生成"
        if need_td:
            stage("重新生成教学设计")
            _, new_td = _generate_rows(case, [candidate])  # 只要教学设计，附带生成的那一行丢掉
            td_note = "，教学设计已重新生成"
        new_courses = courses

    for m in to_remove:
        m.bound = False
        m.graph_points = None  # 以后重新写入时案例内容可能已经变了，重新提取
    for m in to_add:
        m.bound = True
    if allow_llm and final_bound:
        _fill_graph_points(case, final_bound, on_stage=stage)

    changes = {}
    for field, value in (("applicable_courses", new_courses), ("teaching_design", new_td)):
        old_value = getattr(case, field, None)
        new_value = json.dumps(value, ensure_ascii=False) if value is not None else None
        if old_value != new_value:
            changes[field] = {"old": old_value, "new": new_value}
        setattr(case, field, new_value)

    parts = []
    if to_add:
        parts.append(f"新增{len(to_add)}条")
    if to_remove:
        parts.append(f"移除{len(to_remove)}条")
    return changes, ("、".join(parts) or "无变化") + td_note


def sync_case_with_accepted_mappings(db, case, on_stage=None) -> tuple[dict, str]:
    """"补充"按钮：让案例里写入的知识点跟当前"已采纳"的集合保持一致——已采纳但还没写进去
    的追加进去，写进去了但已经不是已采纳的删掉；两边一致就什么都不做（不调模型）。
    返回值同_apply_binding_changes"""
    from db import CaseKnowledgeMapping  # 延迟import，避免和db.py出现循环依赖

    mappings = [
        m for m in db.query(CaseKnowledgeMapping).filter(CaseKnowledgeMapping.case_id == case.id).all()
        if m.knowledge_point
    ]
    to_add = [m for m in mappings if m.status == "已采纳" and not m.bound]
    to_remove = [m for m in mappings if m.bound and m.status != "已采纳"]
    if not to_add and not to_remove:
        # 内容已经一致，只补一下旧数据缺的知识图谱节点提取（提取结果不属于案例字段，不算内容变更）
        filled = _fill_graph_points(case, [m for m in mappings if m.bound], on_stage=on_stage)
        return {}, f"已是最新，无需更新（补全了{filled}条知识图谱的知识点提取）" if filled else "已是最新，无需更新"
    return _apply_binding_changes(db, case, to_add, to_remove, on_stage=on_stage)


def unbind_mapping_from_case(db, case, mapping, on_stage=None) -> tuple[dict, str]:
    """知识点库里解绑：只把这一条从案例里删掉（教学设计正好依据它时重新生成），不顺带
    同步其他还没写进去的已采纳知识点——那个要用户自己点"补充"按钮。返回值同上"""
    return _apply_binding_changes(db, case, [], [mapping], on_stage=on_stage)


def remove_points_from_case(db, case, mappings: list) -> tuple[dict, str]:
    """强制删除知识点之前调用：把这些知识点从案例里删掉，不调模型（在HTTP请求里同步执行）。
    教学设计依据的知识点被删时先清空，调用方据此另起后台任务重新生成。返回值同上"""
    return _apply_binding_changes(db, case, [], mappings, allow_llm=False)


def regenerate_missing_teaching_design(db, case, on_stage=None) -> tuple[dict, str]:
    """教学设计为空、但还有已写入案例的知识点时，用其中相关度最高的一条重新生成教学设计
    （强制删除知识点后的后台补生成）。教学设计已经有了或者一条绑定都没有就什么都不做"""
    from db import CaseKnowledgeMapping  # 延迟import，避免和db.py出现循环依赖

    if _load_json(case.teaching_design, None):
        return {}, "教学设计已存在，无需更新"
    bound = [
        m for m in db.query(CaseKnowledgeMapping)
        .filter(CaseKnowledgeMapping.case_id == case.id, CaseKnowledgeMapping.bound.is_(True)).all()
        if m.knowledge_point
    ]
    if not bound:
        return {}, "没有已写入案例的知识点，教学设计保持为空"
    (on_stage or (lambda _msg: None))("重新生成教学设计")
    _, td = _generate_rows(case, [max(bound, key=lambda m: m.relevance_score or 0)])
    new_value = json.dumps(td, ensure_ascii=False) if td else None
    changes = {"teaching_design": {"old": case.teaching_design, "new": new_value}} if new_value != case.teaching_design else {}
    case.teaching_design = new_value
    return changes, "教学设计已重新生成"
