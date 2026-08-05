"""
上传的 Word/PDF 文档解析：提取正文，并按标题/规则拆分成若干个候选"案例"段落。
拆分是启发式的（无需调用LLM），拆不出明显结构时整篇作为单个候选素材兜底，
避免用户上传的文档因为格式特殊而被静默丢弃。
"""
import io
import re

import docx
import fitz  # PyMuPDF

from ocr_utils import doc_is_scanned, ocr_document_plain_text

# 常见的案例/条目标题模式：如 "案例3.4"、"3.4 DeepSeek..."、"（一）..."、"一、..."
HEADING_PATTERNS = [
    re.compile(r"^\s*案例\s*[0-9一二三四五六七八九十]+(\.[0-9]+)?[：:、\s].*"),
    re.compile(r"^\s*[0-9]+\.[0-9]+\s+\S.*"),
    re.compile(r"^\s*[（(][一二三四五六七八九十0-9]+[)）]\s*\S.*"),
    re.compile(r"^\s*[一二三四五六七八九十]+、\s*\S.*"),
]

MIN_SEGMENT_CHARS = 30  # 短于这个长度的候选段落视为噪声（页眉页脚等），忽略


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 60:
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)


def extract_text_from_docx(file_bytes: bytes) -> list[tuple[str, bool]]:
    """返回 [(段落文本, 是否为Heading样式或形似标题), ...]"""
    document = docx.Document(io.BytesIO(file_bytes))
    lines = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        is_heading = para.style.name.startswith("Heading") or _looks_like_heading(text)
        lines.append((text, is_heading))
    return lines


def extract_text_from_pdf(file_bytes: bytes) -> list[tuple[str, bool]]:
    """PDF没有样式信息，只能靠文本规则判断是否像标题。
    用PyMuPDF而不是pypdf：pypdf对国产办公软件导出的中文PDF（自定义CID字体编码）经常提取出
    乱码或大段丢字，PyMuPDF的字体/编码解析明显更稳，是目前公认解决中文PDF乱码问题的主流方案。"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError("PDF已加密，无法在没有密码的情况下解析")

        if doc_is_scanned(doc):
            # 扫描件（整页是图片，没有文字层），文本抽取拿不到任何内容，走OCR兜底
            try:
                raw_lines = ocr_document_plain_text(doc)
            except Exception as e:
                raise ValueError(f"这是一份扫描版PDF（没有文字层），OCR识别失败: {e}") from e
            return [(text, _looks_like_heading(text)) for text in raw_lines]

        lines = []
        for page in doc:
            page_text = page.get_text("text") or ""
            for raw_line in page_text.split("\n"):
                text = raw_line.strip()
                if not text:
                    continue
                lines.append((text, _looks_like_heading(text)))
        return lines
    finally:
        doc.close()


def split_into_cases(lines: list[tuple[str, bool]]) -> list[dict]:
    """
    按标题行切分成多个段落。如果全篇找不到任何标题特征，整篇作为一个候选案例返回，
    并标记 fallback=True，提示用户人工确认这份文档实际上并未按案例分节。
    """
    segments = []
    current_title = None
    current_body: list[str] = []

    def _flush():
        body_text = "\n".join(current_body).strip()
        if len(body_text) >= MIN_SEGMENT_CHARS:
            segments.append({
                "title": current_title,
                "text": body_text,
            })

    for text, is_heading in lines:
        if is_heading and current_body:
            _flush()
            current_title = text
            current_body = []
        elif is_heading and not current_body and current_title is None:
            current_title = text
        else:
            current_body.append(text)
    _flush()

    if not segments:
        full_text = "\n".join(t for t, _ in lines).strip()
        if not full_text:
            return []
        return [{
            "title": None,
            "text": full_text,
            "fallback": True,
        }]

    for seg in segments:
        seg.setdefault("fallback", False)
    return segments


def parse_uploaded_document(filename: str, file_bytes: bytes) -> list[dict]:
    """入口函数：按扩展名选择解析器，返回拆分后的候选案例列表"""
    lower_name = filename.lower()
    if lower_name.endswith(".docx"):
        lines = extract_text_from_docx(file_bytes)
    elif lower_name.endswith(".pdf"):
        lines = extract_text_from_pdf(file_bytes)
    else:
        raise ValueError("仅支持 .docx 或 .pdf 文件")

    return split_into_cases(lines)
