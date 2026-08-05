"""
原始素材抓取模块
输入一批URL，抓取正文并落盘存档，作为案例真实性的证据链。
"""
import json
import re

import requests
import trafilatura

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

TIMEOUT_SECONDS = 15

# 少数站点（实测政务网站用的某些建站模板）的页面正文中间会提前混入一个 </html> 或 </body>，
# 真正的正文其实还在后面。trafilatura底层用的libxml2解析器认字面意思，一碰到闭合标签就
# 认为文档结束，后面真实的正文反而被整段丢弃——不是编码问题，是这个多出来的闭合标签把解析器
# 提前截断了。这里在解析前把非最后一个的闭合标签去掉，只保留文档真正末尾那一个。
# 用bytes级别的正则处理（不解码），标签本身在任何编码下都是纯ASCII，不会破坏中文字符。
_PREMATURE_CLOSE_TAG_PATTERNS = [
    re.compile(rb"</\s*html\s*>", re.IGNORECASE),
    re.compile(rb"</\s*body\s*>", re.IGNORECASE),
]


def _strip_premature_closing_tags(raw: bytes) -> bytes:
    for pattern in _PREMATURE_CLOSE_TAG_PATTERNS:
        matches = list(pattern.finditer(raw))
        if len(matches) > 1:
            parts = []
            last_end = 0
            for m in matches[:-1]:
                parts.append(raw[last_end:m.start()])
                last_end = m.end()
            parts.append(raw[last_end:])
            raw = b"".join(parts)
    return raw


def fetch_url_text(url: str) -> dict:
    """
    抓取单个URL的正文内容。
    返回: {"status": "success"/"failed", "text": str, "title": str|None, "error": str|None}
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()

        # 直接把原始字节交给trafilatura，让它自己判断编码（比依赖requests按响应头猜编码更准，
        # 很多中文站点声明的charset不准确，用resp.text会导致中文乱码）
        content = _strip_premature_closing_tags(resp.content)
        extracted = trafilatura.extract(
            content,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )

        text = None
        title = None
        if extracted:
            data = json.loads(extracted)
            text = (data.get("text") or "").strip()
            title = (data.get("title") or "").strip() or None

        if not text or len(text) < 30:
            return {
                "status": "failed",
                "text": None,
                "title": title,
                "error": "正文提取为空或过短，可能是反爬页面/需要JS渲染，建议人工核实",
            }

        return {
            "status": "success",
            "text": text,
            "title": title,
            "error": None,
        }

    except requests.exceptions.RequestException as e:
        return {"status": "failed", "text": None, "title": None, "error": f"请求失败: {str(e)}"}
    except Exception as e:
        return {"status": "failed", "text": None, "title": None, "error": f"抓取异常: {str(e)}"}
