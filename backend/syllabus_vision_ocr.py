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
- 表格某一行恰好跨页断开、续页顶部不重复表头时（真实遇到过：一个"实验单元"格的内容
  被从中间切开，一半在上一页页底，一半在下一页页顶），试过借用`syllabus_table_ocr.
  _find_column_anchors`去算列的坐标位置、转成"大约百分之多少宽度"塞进续页的prompt里
  当空间锚点——这个函数是给数字PDF的精确文字层坐标设计的，套在扫描件的Tesseract免费
  OCR结果上不可靠：Tesseract在"这一步不要求识别质量、只求搜到关键词"的宽松要求下，
  会在明明没有表头的续页上，把某处笔迹误读出"单元""(知识点)"这类碎片、凑巧被判定成
  "这页有表头"，反而没触发提示、问题依旧。这跟上面那条"跨页断言"的教训是同一类坑：
  想靠额外的跨页信息去补续页缺失的上下文，只要这个额外信息本身不够可靠，就会在"判断
  错误"的那些页面上引入新的错误，得不偿失。这类"表格行恰好跨页断开"的情况暂时接受
  作为已知的边界局限，不再继续加码修。
- 真实遇到过另一种表格：不是每个"知识单元"一行、内容较短，而是每一"章"一整行、内容
  很长（一章的编号列表+重点+难点能占满一整页甚至跨页），这种表格的续页顶部经常不重复
  "知识单元/教学内容"这两个词（find_table_pages原本靠这两个词判断续页，遇到这种表格
  会在第二页就直接断掉，只识别出第一章）——已经改成"进表格后一律当续页，直到遇到下一个
  顶级章节标题才结束"，不再逐页找关键词。表格起始页本身如果上面还有一大段无关说明文字、
  表格只占页面下半一小块，标题行的章节名（比如"第一章"）这种小字有较高概率被模型看漏
  返回空——加了本地免费OCR定位表格标题的y坐标、把上面无关内容裁掉再发图，配合起始页
  专属的重试（_extract_page_via_vision_with_retry的require_leading_chapter），实测能
  稳定读对。
- 同一份"每章一整行"的大纲还发现：如果某一页顶部是上一章遗留的"重点/难点"续行（没有
  章节名），紧接着同一张图下半部分又完整出现了下一章的新内容，模型有不低的概率
  （实测大概3-4成）把这段没有名字的续行片段整个漏掉、只返回下一章那一条——不是拼错，
  是干脆不写。已经在prompt规则0里明确要求"即使同一张图后面还有完整的新章节，这段续行
  也必须单独占数组里的第一个元素，不能省略"，能把漏掉的概率降下来但没能完全消除。
  这跟前面几条"额外信息不可靠、越修越糟"的坑不是一回事——影响范围小（丢的是某一章
  的重点/难点这类补充说明，不是整章内容或整份表格），要根治得让每一页都不管有没有
  出错都强制再问一次拿去比较，等于把总耗时和调用成本翻倍去应对一个只在特定页面结构
  下才出现、且不影响主干数据的小概率问题，暂时不做，先接受作为已知限制。
"""
import base64
import json
import logging
import os
import re
import time

import fitz

from ocr_utils import ocr_page_lines
from qwen_client import get_client, require_api_key

logger = logging.getLogger("uvicorn.error")

# 这两张表各自标题里的关键词，命中即认为该页是某张目标表格的起始页；往这个列表里加词
# 就能适配措辞略有出入的大纲（比如别的学校可能叫"实践单元"而不是"实验单元"）
TABLE_TITLE_KEYWORDS = ["知识单元", "实验单元"]
# 顶级章节分界（比如"六、课程考核方式"）之后就不是目标表格了。别把"思政融合点"这种
# 说明段落关键词也塞进这条正则——实测有的大纲模板会把"思政融合点X：第Y章"这种说明
# 直接穿插在表格每一章内容中间（不是全部拼在表格最后），当成全局分界的话反而会在
# 表格中途就提前误判"表格结束"，把后面几章截断掉
_TOP_SECTION_RE = re.compile(r"[六七八九十]、|课程考核方式")
# 但如果一整页从头开始就是"思政融合点"说明文字（前面完全没有表格内容、真正的表格已经
# 在上一页就结束了），这一页就不该再被当成续页发给视觉大模型——发了也是白发，纯说明
# 文字没有"知识单元/教学内容"这种结构，模型容易乱编。只在页面开头判断，不影响上面
# "思政融合点跟真表格内容混在同一页中间"的情况
_PURE_COMMENTARY_START_RE = re.compile(r"^\s{0,5}思政融合点")

# 可以用 QWEN_VISION_MODEL 环境变量换模型对比效果，不用改代码；默认用Qwen3-VL-8B-Instruct
# （deepseek-ai/DeepSeek-OCR 实测在硅基流动上输出不稳定，会编造跟文档无关的内容，不要用）
VISION_MODEL = os.environ.get("QWEN_VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VISION_DPI = 200
_JSON_FENCE_RE = re.compile(r"^```(json)?|```$")


def _page_ocr_text(page) -> str:
    """只用来搜关键词定位页面，不要求识别质量，所以可以直接复用ocr_page_lines拼起来"""
    return "".join(text for *_, text in ocr_page_lines(page))


def _find_table_top_y(page) -> float | None:
    """定位这页里表格标题（'知识单元'/'实验单元'关键词）所在行的y坐标（PDF点数），
    找不到就返回None——通常是续页，本来就没有独立的表格标题可定位，不需要裁剪"""
    for x0, y0, x1, y1, text in ocr_page_lines(page):
        if any(kw in text for kw in TABLE_TITLE_KEYWORDS):
            return y0
    return None


def find_table_pages(doc) -> list[int]:
    """定位属于"知识单元/实验单元"这类表格的页码（0-based，含续页），按顺序返回"""
    pages: list[int] = []
    last_hit_page: int | None = None
    table_ended = True  # 还没进入任何目标表格之前，谈不上"续页"

    for pno in range(len(doc)):
        text = _page_ocr_text(doc[pno])
        # 页面开头就是纯说明文字的情况，位置必然是0，比正文中间才出现的顶级章节分界
        # 更早，所以优先判它
        cutoff = _PURE_COMMENTARY_START_RE.match(text) or _TOP_SECTION_RE.search(text)
        text_before_cutoff = text[: cutoff.start()] if cutoff else text

        is_title = any(kw in text_before_cutoff for kw in TABLE_TITLE_KEYWORDS)
        if is_title:
            table_ended = False

        # 续页顶部是合并单元格的延续，通常不会重复"知识单元/教学内容"这些表头文字
        # （真实遇到过：一份大纲的表格每章一行、跨好几页，续页整页都是编号列表和
        # "重点/难点"，完全没有这两个词），不能靠关键词判断续页在不在表里；只要上一页
        # 还在表里、还没遇到过顶级章节分界，就认为还在同一张表格里。一旦遇到过一次就
        # 不再回头——不然表格结束后的其他章节（比如"七、八、九、十"）如果某一页恰好
        # 没再出现这些分界词，会被误判成还在续表格里，越滚越多
        is_continuation = (
            not table_ended
            and last_hit_page is not None
            and pno - last_hit_page == 1
            and bool(text_before_cutoff.strip())
        )
        if is_title or is_continuation:
            pages.append(pno)
            last_hit_page = pno

        if cutoff is not None:
            table_ended = True

    return pages


def _extract_json(raw_text: str):
    text = _JSON_FENCE_RE.sub("", raw_text.strip()).strip()
    return json.loads(text)


def _extract_page_via_vision(page) -> list[dict]:
    """返回这一页识别出的 [{"章节": str, "教学内容": str}, ...]——一个知识单元/实验单元对应
    一条记录，"教学内容"是那一整格的原文，不拆分成多条、不删减"重点/难点"（下游知识点匹配
    要求"章节"跟"知识单元"一致、"内容描述"跟"教学内容"一致，逐字对应原表格，不能有取舍）。
    如果这页顶部是续行、没出现新的知识单元/实验单元名称，"章节"允许留空——
    续页的章节名由调用方（extract_units_and_points）用上一页的章节名兜底补上，
    不在这里靠prompt硬塞"上一页是XXX"这种断言：早先这么做过，遇到参数量更大、更
    "较真"的模型时，只要这句断言在个别页面上不成立（这页其实不是续页），模型就会
    因为断言和图片内容对不上直接返回空数组，比不给任何提示还糟。"""
    # 表格标题页有时上半页是一大段跟表格无关的说明文字（比如"课程教学方法和课堂形式"），
    # 表格本身只占页面下半部分一小块——整页发过去，表格第一行的章节名这种小字很容易被
    # 模型看漏（实测确认过）。这里先用本地免费OCR定位表格标题的位置，找到了就把上面
    # 无关的部分裁掉，让表格在图片里占比更大；续页本来就没有独立的表格标题可定位，
    # 找不到就不裁剪，发整页
    top_y = _find_table_top_y(page)
    if top_y is not None and top_y > 60:
        clip = fitz.Rect(0, max(0, top_y - 30), page.rect.width, page.rect.height)
        pix = page.get_pixmap(dpi=VISION_DPI, clip=clip)
    else:
        pix = page.get_pixmap(dpi=VISION_DPI)
    # JPEG比PNG小不少，这张图既要占内存又要经公网发给SiliconFlow，云上小带宽实例上
    # 体积越小越好；文档类图片JPEG的有损压缩对识别效果影响可以忽略
    b64 = base64.b64encode(pix.tobytes("jpg", jpg_quality=85)).decode()

    prompt = (
        "这张图片是课程教学大纲表格，左边一列是'知识单元/实验单元'（可能是合并单元格，"
        "同一个知识单元名称跨好几行），右边一列是'教学内容(知识点)'——同一个知识单元对应的"
        "这一整格内容，版面上经常被横线分隔成好几个小块（比如上半部分是编号列表"
        "'1. 2. 3.'，下面单独一行'重点：...'，再下面单独一行'难点：...'），这几个小块"
        "虽然版面上看着分开，但都属于同一个知识单元格，不是几条独立的记录。\n\n"
        "请按'知识单元'（左边合并单元格）为单位输出JSON数组，一个知识单元只对应数组里"
        '一个元素，格式是 {"章节": "知识单元/实验单元列的内容", "教学内容": "这个知识单元'
        '对应的教学内容列，把里面所有小块（编号列表、重点、难点）全部拼接成的完整字符串"}。\n'
        "严格要求：\n"
        "0. 如果这张图片最上面几行看起来是接着上一页表格没写完的内容（这张图开头"
        '没有出现新的知识单元/实验单元名称），仍然必须把这部分内容作为数组的第一个'
        '元素单独输出，"章节"这个字段留空字符串就行（不用猜测或编造一个章节名），'
        "绝对不能因为它没有章节名就跳过不写，也不能把它并入这张图片后面出现的新知识"
        "单元里——哪怕这张图片同时还包含一个完整的新知识单元（有自己的名称、内容也"
        "齐全），这两部分也要分别输出成数组里两个独立的元素，一个都不能少。\n"
        "0.1 有时这张图片里表格只占页面下半部分一小块（上半部分是大段无关的说明文字），"
        "这种情况下第一行'知识单元'那一格的文字（比如'第一章'）容易因为字小、占比小被"
        "看漏、误判成空——请专门找到表格实际开始的位置，仔细看清楚这一格里写的什么字，"
        "不要因为图片大部分是无关文字就把这格也当成没有章节名的续行。\n"
        '1. 【最重要，务必遵守】"教学内容"字段必须把同一个知识单元对应的编号列表、'
        '"重点：..."、"难点：..."这几部分全部拼接成同一个字符串（用"\\n"换行分隔），'
        "绝对不能因为它们中间隔着横线、版面上是分开的小格子，就拆成JSON数组里的多个"
        "元素——一个知识单元只能对应数组里的一条记录，不能是两条、三条。反面例子：如果"
        '右边教学内容格里是"1. Spring MVC 的工作原理;\\n...\\n5. Spring MVC 的基本配置。'
        '\\n重点：Spring MVC 的工作原理\\n难点：Spring MVC 的工作原理"，绝对不能拆成三个'
        'JSON元素分别装编号列表、重点、难点，必须整个拼成一个字符串放进一条记录的'
        '"教学内容"字段里。\n'
        '2. "教学内容"字段内部要逐字转录原文——包括编号、"重点："、"难点："这些文字，'
        "一个字都不要删减，也不要概括或改写。\n"
        '3. 不要把右边"教学目标"列的内容（通常是"能够..."这种表述学生要达到什么能力的'
        '句子，跟"教学内容"不是同一列）当成"教学内容"混进来。\n'
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


def _extract_page_via_vision_with_retry(
    page, pno: int, require_leading_chapter: bool = False
) -> list[dict]:
    """require_leading_chapter=True 用于一张表格真正的起始页——这页的第一条记录理论上
    必须有章节名（它前面没有上一页可以续），如果模型漏读成空，不像续页那样可以靠上一个
    已知章节名兜底，只能重试。实测这个模型在"页面大部分是无关说明文字、表格只占一小块"
    的起始页上，即使返回了内容，也有不低的概率把第一行的章节名看漏，重试基本能读对。"""
    last_error: Exception | None = None
    last_rows: list[dict] | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            rows = _extract_page_via_vision(page)
        except Exception as e:
            last_error = e
            logger.warning(f"第{pno + 1}页视觉识别第{attempt}次尝试出错（耗时{time.time()-t0:.1f}s）: {e}")
            continue
        if rows:
            last_rows = rows
            if require_leading_chapter and not str(rows[0].get("章节") or "").strip():
                logger.warning(f"第{pno + 1}页是表格起始页，第{attempt}次尝试漏读了第一行的章节名（耗时{time.time()-t0:.1f}s），重试")
                continue
            logger.info(f"第{pno + 1}页视觉识别成功，第{attempt}次尝试，耗时{time.time()-t0:.1f}s")
            return rows
        logger.warning(f"第{pno + 1}页视觉识别第{attempt}次尝试返回空结果（耗时{time.time()-t0:.1f}s），重试")
    if last_rows is not None:
        logger.warning(f"第{pno + 1}页重试{_MAX_ATTEMPTS}次仍未读到表格起始行的章节名，使用最后一次的结果")
        return last_rows
    if last_error is not None:
        raise last_error
    return []


def extract_units_and_points(doc) -> list[tuple[str | None, str]]:
    """
    doc: 已打开的 fitz.Document（扫描版PDF）
    返回 [(章节名或None, 教学内容原文), ...]，按出现顺序——一个知识单元对应一条记录，
    "教学内容"是那一格的原文整体，不拆分成多条子知识点、不删减重点/难点。
    单页识别失败（重试用完仍然失败）不影响其他页，跳过该页并记录warning。
    """
    require_api_key()
    t_start = time.time()
    table_pages = find_table_pages(doc)
    logger.info(f"定位到{len(table_pages)}页目标表格页面: {[p+1 for p in table_pages]}，耗时{time.time()-t_start:.1f}s")

    results: list[tuple[str | None, str]] = []
    last_chapter: str | None = None

    for pno in table_pages:
        t_page = time.time()
        try:
            page_rows = _extract_page_via_vision_with_retry(
                doc[pno], pno, require_leading_chapter=(pno == table_pages[0])
            )
        except Exception as e:
            logger.warning(f"第{pno + 1}页表格视觉识别失败（已重试{_MAX_ATTEMPTS}次，耗时{time.time()-t_page:.1f}s），跳过这一页: {e}")
            continue

        for row in page_rows:
            if not isinstance(row, dict):
                continue
            chapter = (row.get("章节") or "").strip() or last_chapter
            content = str(row.get("教学内容") or "").strip()
            if content:
                # prompt里三令五申"编号列表/重点/难点要拼成一条记录"，模型还是有概率把
                # 同一个知识单元格里视觉上分开的这几个小块拆成JSON数组里的多个元素——
                # 这里用代码兜底：只要章节名跟上一条记录相同，就说明这是同一个知识单元
                # 格的后续小块，直接拼接回上一条，不新开一条记录，不完全指望模型每次都
                # 严格遵守指令
                if results and results[-1][0] == chapter:
                    prev_chapter, prev_content = results[-1]
                    results[-1] = (prev_chapter, prev_content + "\n" + content)
                else:
                    results.append((chapter, content))
            if chapter:
                last_chapter = chapter

    logger.info(f"全部页面识别完成，共{len(results)}条知识单元记录，总耗时{time.time()-t_start:.1f}s")
    return results
