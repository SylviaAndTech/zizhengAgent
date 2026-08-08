"""
扫描版课程大纲PDF里"知识单元/教学内容(知识点)"、"实验单元/教学内容(知识点)"这两类表格的解析。

跟 syllabus_table_ocr.py（按坐标分列，纯本地免费）不是同一套方案：那套方案对这份大纲里
"知识单元"这种窄列、字符间距被拉伸的英文单词识别经常出错（Tesseract逐字符识别，没有上下文
纠错能力）。这里换成两步走：
1. 先用免费的Tesseract整页OCR，只是为了搜关键词、定位"知识单元/实验单元"这两张表分别在
   哪几页（含续页），不要求这一步文字识别多准，只要能搜到关键词就行；
2. 只把定位到的这几页截图，逐页发给视觉大模型（能"看懂"整页版面+结合上下文语义纠错，比逐字符
   OCR准得多，尤其是窄列变形文字），直接问它要结构化的"章节+知识点列表"JSON。

这样只有真正含目标表格的页面才会调用大模型（省钱），非目标页面（课程基本信息、考核标准、
参考书目等）完全不产生视觉模型调用。

【踩过的坑，别再犯】
- 试过"不做页面定位，把整份PDF所有页一次性发给模型"：在12页的真实大纲上出现了明显幻觉——
  编造出原表格里不存在的章节、把不同章节的内容互相串了。
- 试过"把定位到的几页(比如7页)打包成一次API调用"：幻觉依然存在，比如把"Sping框架概述"
  这个单元的真实内容错误地安到了"Spring MVC原理"这个章节名下面。也就是说不是"页数太多"
  的问题，现在用的 Qwen3-VL-8B-Instruct 这个模型只要一次调用里塞超过1张图，就容易把
  不同图片的内容搞混——参数量较小的视觉模型在这类多图结构化抽取任务上似乎就是不如
  单图任务稳。
- 试过换成 deepseek-ai/DeepSeek-OCR：这个模型压根不理会"按JSON schema输出"这种指令，
  会输出它自己的整页HTML/Markdown表格转录（有时候转录质量很好，忠实保留了原文错别字），
  但同一张图连续调用几次，产出内容极不稳定——出现过好几次输出跟这份文档完全无关的内容
  （物理题、不知道哪来的客户数据表格），在硅基流动上这个模型目前不可信，先排除掉。
- 早先"每次调用只给1张图"是对的，但当时给每一页都塞了一句"上一页最后一个单元是XXX，
  这页续着写"的文字提示，想靠这个帮模型接上跨页续接的章节名。换成参数量更大的
  Qwen/Qwen3.6-35B-A3B 测试时发现这句提示反而是个坑：只要某一页其实不是续页（比如
  这页本身就是一张新表格的开头），这句断言就是错的，跟这页图片实际内容对不上，模型会
  直接返回空数组——比不给提示还糟。换成让模型"看不出续页就把章节留空，我们代码用上一个
  已知章节名兜底"（extract_units_and_points里`chapter = row.get("章节") or last_chapter`
  这行本来就有这个兜底逻辑），不再往prompt里塞可能是错的断言，两个模型上都测过表现更好。

所以保留"每次调用只给1张图"这个设定，跨页续接靠代码兜底、不靠prompt里的断言，这是几个
方案里唯一没出现编造内容问题的，准确度优先。
"""
import base64
import json
import logging
import os
import re

from ocr_utils import ocr_page_lines
from qwen_client import get_client, require_api_key

logger = logging.getLogger("uvicorn.error")

# 这两张表各自标题里的关键词，命中即认为该页是某张目标表格的起始页；往这个列表里加词
# 就能适配措辞略有出入的大纲（比如别的学校可能叫"实践单元"而不是"实验单元"）
TABLE_TITLE_KEYWORDS = ["知识单元", "实验单元"]
# 续页判断用：命中标题的页之后，只要连续下一页还有这些字样，就认为还在同一张表里
CONTINUATION_KEYWORDS = ["知识点", "教学内容"]
# 顶级章节分界（比如"六、课程考核方式"）之后就不是目标表格了
_TOP_SECTION_RE = re.compile(r"[六七八九十]、|课程考核方式")

# 可以用 QWEN_VISION_MODEL 环境变量换模型对比效果，不用改代码；默认用Qwen3-VL-8B-Instruct
# （deepseek-ai/DeepSeek-OCR 实测在硅基流动上输出不稳定，会编造跟文档无关的内容，不要用）
VISION_MODEL = os.environ.get("QWEN_VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VISION_DPI = 200
_JSON_FENCE_RE = re.compile(r"^```(json)?|```$")
_POINT_NOISE_RE = re.compile(r"^(重点|难点)[：:]")


def _page_ocr_text(page) -> str:
    """只用来搜关键词定位页面，不要求识别质量，所以可以直接复用ocr_page_lines拼起来"""
    return "".join(text for *_, text in ocr_page_lines(page))


def find_table_pages(doc) -> list[int]:
    """定位属于"知识单元/实验单元"这类表格的页码（0-based，含续页），按顺序返回"""
    pages: list[int] = []
    last_hit_page: int | None = None

    for pno in range(len(doc)):
        text = _page_ocr_text(doc[pno])
        cutoff = _TOP_SECTION_RE.search(text)
        text_before_cutoff = text[: cutoff.start()] if cutoff else text

        is_title = any(kw in text_before_cutoff for kw in TABLE_TITLE_KEYWORDS)
        is_continuation = (
            last_hit_page is not None
            and pno - last_hit_page == 1
            and any(kw in text_before_cutoff for kw in CONTINUATION_KEYWORDS)
        )
        if is_title or is_continuation:
            pages.append(pno)
            last_hit_page = pno

    return pages


def _extract_json(raw_text: str):
    text = _JSON_FENCE_RE.sub("", raw_text.strip()).strip()
    return json.loads(text)


def _extract_page_via_vision(page) -> list[dict]:
    """返回这一页识别出的 [{"章节": str, "知识点": [str, ...]}, ...]。
    如果这页顶部是续行、没出现新的知识单元/实验单元名称，"章节"允许留空——
    续页的章节名由调用方（extract_units_and_points）用上一页的章节名兜底补上，
    不在这里靠prompt硬塞"上一页是XXX"这种断言：早先这么做过，遇到参数量更大、更
    "较真"的模型时，只要这句断言在个别页面上不成立（这页其实不是续页），模型就会
    因为断言和图片内容对不上直接返回空数组，比不给任何提示还糟。"""
    pix = page.get_pixmap(dpi=VISION_DPI)
    # JPEG比PNG小不少，这张图既要占内存又要经公网发给SiliconFlow，云上小带宽实例上
    # 体积越小越好；文档类图片JPEG的有损压缩对识别效果影响可以忽略
    b64 = base64.b64encode(pix.tobytes("jpg", jpg_quality=85)).decode()

    prompt = (
        "请提取这张课程教学大纲表格图片里的内容，按JSON数组输出，每个元素是 "
        '{"章节": "知识单元/实验单元列的内容", "知识点": ["教学内容(知识点)列里每一条编号内容，'
        '一条编号对应数组里一个字符串", ...]}。\n'
        "严格要求：\n"
        "0. 如果这张图片最上面几行知识点看起来是接着上一页表格没写完的内容（这张图开头"
        '没有出现新的知识单元/实验单元名称），"章节"这个字段留空字符串就行，不用猜测'
        "或编造一个章节名。\n"
        '1. "教学内容(知识点)"这一列里，只要看到以"重点："或"难点："开头的整行，把这一整行'
        '（包括"重点/难点"后面跟着的具体内容）彻底丢弃，不要以任何形式出现在"知识点"数组里——'
        '既不要保留"重点："这个前缀，也不要把它后面的内容当成独立的一条知识点收进来。'
        '举例：这一列原文是"Spring Boot 入门项目实践\\n重点：熟悉开发环境\\n难点：理解工作机制"，'
        '"知识点"数组应该只有一个元素["Spring Boot 入门项目实践"]，不能出现"熟悉开发环境"'
        '或"理解工作机制"，这条规则优先级最高。\n'
        '2. 排除"重点/难点"之后剩下的内容：如果有"1. 2. 3."编号，按编号拆分，每条编号对应'
        '数组里一个字符串，不要合并；如果没有编号、只是一段完整的实验/项目名称，就把这段内容'
        '整体作为数组里唯一的一个字符串。\n'
        '3. 不要把右边"教学目标"列的内容（通常是"能够..."这种表述学生要达到什么能力的句子）'
        '当成知识点混进来。\n'
        "只输出JSON数组本身，不要用markdown代码块包裹，不要有任何其他文字。"
    )

    response = get_client().chat.completions.create(
        model=VISION_MODEL,
        # 有的视觉模型（比如Qwen3.6-35B-A3B这种"思考"型模型）会先生成一段内部推理过程再
        # 给最终答案，2000太容易被推理过程本身耗光，导致finish_reason="length"、真正的
        # JSON内容被截断成空——这里给足够宽裕的余量，避免这种"推理没写完答案就被砍掉"
        max_tokens=6000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    raw_text = response.choices[0].message.content or "[]"
    data = _extract_json(raw_text)
    if not isinstance(data, list):
        raise ValueError(f"视觉模型返回的不是JSON数组: {raw_text[:200]}")
    return data


# 实测这个视觉模型有小概率（大致15%~20%）单次调用要么直接报错（比如JSON解析失败），
# 要么"成功"但返回空数组——不是哪一页内容有问题，纯粹是这次调用运气不好，同一页立刻
# 重新问一次基本都能成功。定位到的页面理论上不该是空的（find_table_pages已经筛过），
# 所以"返回空数组"也当成一种失败，一起重试
_MAX_ATTEMPTS = 3


def _extract_page_via_vision_with_retry(page, pno: int) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            rows = _extract_page_via_vision(page)
        except Exception as e:
            last_error = e
            logger.warning(f"第{pno + 1}页视觉识别第{attempt}次尝试出错: {e}")
            continue
        if rows:
            return rows
        logger.warning(f"第{pno + 1}页视觉识别第{attempt}次尝试返回空结果，重试")
    if last_error is not None:
        raise last_error
    return []


def extract_units_and_points(doc) -> list[tuple[str | None, str]]:
    """
    doc: 已打开的 fitz.Document（扫描版PDF）
    返回 [(章节名或None, 知识点描述), ...]，按出现顺序。
    单页识别失败（重试用完仍然失败）不影响其他页，跳过该页并记录warning。
    """
    require_api_key()
    table_pages = find_table_pages(doc)

    results: list[tuple[str | None, str]] = []
    last_chapter: str | None = None

    for pno in table_pages:
        try:
            page_rows = _extract_page_via_vision_with_retry(doc[pno], pno)
        except Exception as e:
            logger.warning(f"第{pno + 1}页表格视觉识别失败（已重试{_MAX_ATTEMPTS}次），跳过这一页: {e}")
            continue

        for row in page_rows:
            if not isinstance(row, dict):
                continue
            chapter = (row.get("章节") or "").strip() or last_chapter
            for point in row.get("知识点") or []:
                point = str(point).strip()
                # 模型偶尔还是会把"重点：xxx"/"难点：xxx"整行当成一条知识点混进来，
                # prompt里三令五申了还是有概率漏，这里用代码兜底再过滤一遍，不额外花钱调用
                if point and not _POINT_NOISE_RE.match(point):
                    results.append((chapter, point))
            if chapter:
                last_chapter = chapter

    return results
