"""
原始素材抓取模块
输入一批URL，抓取正文并落盘存档，作为案例真实性的证据链。
"""
import requests
import trafilatura

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

TIMEOUT_SECONDS = 15
MAX_TEXT_LENGTH = 20000  # 单条素材正文最大保留长度，避免过长拖慢生成


def fetch_url_text(url: str) -> dict:
    """
    抓取单个URL的正文内容。
    返回: {"status": "success"/"failed", "text": str, "error": str|None}
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        html = resp.text

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )

        if not extracted or len(extracted.strip()) < 30:
            return {
                "status": "failed",
                "text": None,
                "error": "正文提取为空或过短，可能是反爬页面/需要JS渲染，建议人工核实",
            }

        return {
            "status": "success",
            "text": extracted.strip()[:MAX_TEXT_LENGTH],
            "error": None,
        }

    except requests.exceptions.RequestException as e:
        return {"status": "failed", "text": None, "error": f"请求失败: {str(e)}"}
    except Exception as e:
        return {"status": "failed", "text": None, "error": f"抓取异常: {str(e)}"}
