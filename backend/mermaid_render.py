"""
用 Playwright 驱动无头 Chromium，把 Mermaid 图定义字符串渲染成 PNG 图片——供"导出勾选为
Word"/"导出成书"两处调用，把案例详情页里能看到的"适用课程举例"树状图也嵌进导出的Word文档里
（之前这两个导出流程只写了文字版的表格，没有把图带出去）。

Mermaid本身只在浏览器里跑（mermaid.js），Python后端没有原生渲染能力，这里用Playwright起
一个无头浏览器页面、加载跟前端同一个CDN上的mermaid.js（只是加载这个开源JS库本身，不会把
任何案例内容发给第三方——真正的图定义字符串是本地生成、在无头浏览器本机内存里渲染的，
不经过网络）、把定义字符串塞进页面里渲染成SVG，再截图成PNG。

同一次导出可能要渲染几十张图（每个案例一张），这里用同一个浏览器实例批量渲染，不是每张图
都重新起一次浏览器进程——那样会非常慢。

依赖：需要先 `pip install playwright` 并且执行一次 `playwright install chromium`
下载无头浏览器内核（这一步不是pip安装能带的，是Playwright自己的下载命令，只需要在
部署环境里跑一次）。渲染失败（比如没装浏览器内核、单张图渲染超时）不应该导致整个导出
失败——只是这张图导不出来，调用方拿到None就跳过嵌入图片，案例详情里原有的文字版表格
还在，不影响内容完整性。
"""
import logging

logger = logging.getLogger("uvicorn.error")

MERMAID_CDN_URL = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"


def _page_html(definition: str) -> str:
    return f"""<!doctype html><html><body style="margin:0;background:white;">
<div id="graph" class="mermaid">{definition}</div>
<script src="{MERMAID_CDN_URL}"></script>
<script>mermaid.initialize({{startOnLoad:true, securityLevel:"strict", theme:"base"}});</script>
</body></html>"""


def render_mermaid_batch(
    definitions: list[str | None], width: int = 1400, timeout_ms: int = 15000,
) -> list[bytes | None]:
    """definitions里的None元素直接跳过（对应没有适用课程举例、不需要画图的案例），返回
    等长的结果列表，每项要么是PNG字节，要么是None（没有图，或者渲染失败）。"""
    results: list[bytes | None] = [None] * len(definitions)
    pending = [(i, d) for i, d in enumerate(definitions) if d]
    if not pending:
        return results

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning(
            "未安装playwright，跳过Mermaid图导出（Word文档里仍保留文字版表格）；"
            "需要执行 pip install playwright 并 playwright install chromium"
        )
        return results

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                for i, definition in pending:
                    page = None
                    try:
                        page = browser.new_page(viewport={"width": width, "height": 900})
                        page.set_content(_page_html(definition))
                        page.wait_for_selector("#graph svg", timeout=timeout_ms)
                        svg_el = page.query_selector("#graph svg")
                        results[i] = svg_el.screenshot(type="png")
                    except Exception as e:
                        logger.warning(f"第{i}张Mermaid图渲染失败，跳过：{e}")
                    finally:
                        if page:
                            page.close()
            finally:
                browser.close()
    except Exception as e:
        logger.warning(
            "启动无头浏览器失败，跳过全部Mermaid图导出（需要先执行一次"
            f"`playwright install chromium`下载浏览器内核）：{e}"
        )

    return results
