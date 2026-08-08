"""
扫描版PDF（没有文字层，整页是图片）的OCR兜底。
用Tesseract（通过PyMuPDF的OCR集成）逐页识别；本机需要安装tesseract + 中文语言包
（macOS: brew install tesseract tesseract-lang）。

这里只提供"把一页OCR成带坐标的文本行"这一个基础能力，给两处调用方用：
- parse_document.py：素材/案例文档上传，只要按阅读顺序把文字拼出来即可；
- knowledge_matching.py：课程大纲的"知识单元/知识点"表格，需要按列位置把知识点列
  和教学目标列分开，不能简单按阅读顺序拼接，所以由它自己在这些文本行基础上做列重建。
"""
import fitz  # PyMuPDF

OCR_LANGUAGE = "chi_sim"
OCR_DPI = 300


def page_has_text(page) -> bool:
    return bool((page.get_text("text") or "").strip())


def doc_is_scanned(doc) -> bool:
    """整份PDF所有页都没有文字层，才判定为扫描件——避免个别页面是空白页时误判"""
    return not any(page_has_text(page) for page in doc)


def ocr_page_lines(page, dpi: int = OCR_DPI) -> list[tuple[float, float, float, float, str]]:
    """OCR单页，返回 [(x0, y0, x1, y1, 行文本), ...]，按y从上到下排序（同一y的按x从左到右）。
    需要本机装了tesseract，没装的话PyMuPDF会抛异常，调用方要接住并给出清晰的报错提示。"""
    tp = page.get_textpage_ocr(flags=0, language=OCR_LANGUAGE, dpi=dpi, full=True)
    d = page.get_text("dict", textpage=tp)
    lines = []
    for block in d["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append((x0, y0, x1, y1, text))
    lines.sort(key=lambda l: (round(l[1] / 5), l[0]))
    return lines


def ocr_document_plain_text(doc) -> list[str]:
    """把扫描版PDF按阅读顺序OCR成一行行文本，供不需要表格结构的场景（素材/案例文档）使用"""
    all_lines = []
    for page in doc:
        for x0, y0, x1, y1, text in ocr_page_lines(page):
            all_lines.append(text)
    return all_lines


def native_page_lines(page) -> list[tuple[float, float, float, float, str]]:
    """跟ocr_page_lines返回同样的结构（每行文字+坐标），但直接读PDF自带的文字层，不用OCR。
    "知识单元/教学内容(知识点)"这种表格排版不只出现在扫描件里，数字PDF（有文字层）一样可能用
    这种表格，所以按坐标分列的解析逻辑（syllabus_table_ocr.py）需要能同时接两种"取行"的方式。"""
    d = page.get_text("dict")
    lines = []
    for block in d["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append((x0, y0, x1, y1, text))
    lines.sort(key=lambda l: (round(l[1] / 5), l[0]))
    return lines
