"""
知识图谱里的"知识点"节点：从知识点描述（知识单元那一整格教学内容）里，挑出跟案例最相关的
那几条编号条目，每条一个节点。

- 只摘原文、不改写：先用代码按"1. 2. 3."编号把描述拆成条目，模型只负责从中挑出相关的编号，
  不自己写文字；描述里没有编号时，模型从原文里截一段，代码校验这段文字确实出现在原文里。
- "重点："/"难点："之后的内容不参与：那是对前面编号条目的重复强调，拿来做节点会重复。
- 每个知识单元1~5条，最多5条：太多会把图撑得很大。
- 在"补充"同步写入案例时提取一次、存进CaseKnowledgeMapping.graph_points，画图时不调模型；
  还没提取过（或提取失败）时用fallback_points兜底。
"""
import json
import logging
import re

from prompts import GRAPH_POINTS_SYSTEM_PROMPT, build_graph_points_prompt
from qwen_client import CHAT_MODEL, chat_json, require_api_key

logger = logging.getLogger("uvicorn.error")

MAX_POINTS = 5  # 跟knowledge_graph.MAX_POINTS_PER_CHAPTER一致
MAX_CASE_CHARS = 2500  # 案例正文只给前面一段做相关性判断，够用且省token

_FOCUS_RE = re.compile(r"(重点|难点)\s*[:：]")
_BOUNDARY_CHARS = "\n；;。，,"
_TRAILING_PUNCT = " \t\n；;。，,、"


def _strip_focus(description: str) -> str:
    """去掉"重点：/难点："及其后面的内容"""
    m = _FOCUS_RE.search(description or "")
    return (description[:m.start()] if m else description or "").strip()


# 编号的几种写法：1. / 1、 / (1) / （1） / [1] / 【1】 / ① / 行首直接跟汉字的"1实现……"
_MARKER_RE = re.compile(
    r"[（(](\d{1,2})[）)]\s*"
    r"|[\[【](\d{1,2})[\]】]\s*"
    r"|([\u2460-\u2469])\s*"
    r"|(\d{1,2})\s*[.．、]\s*"
    r"|(\d{1,2})(?=[\u4e00-\u9fff])"
)
# 按行拆分时去掉行首的编号（含"2.1"这种多级编号）
_LINE_PREFIX_RE = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,2})+|[（(\[【]?\d{1,2}[）)\]】]|[\u2460-\u2469]|\d{1,2})\s*[.．、]?\s*")


def split_items(description: str) -> list[str]:
    """把描述里的编号条目按顺序拆出来（不含重点/难点部分），去掉编号和末尾标点。

    只认"下一个应该出现的编号"（1、2、3……依次递增），并且编号前面得是行首/分隔标点，
    或者紧跟在汉字/右括号后面且编号后不是数字（"方法2.知识库"这种原文漏了分隔符的写法）——
    这样"工业 1.0 到工业 3.0"里的"1.0""3.0"不会被误当成编号。按编号拆不出至少两条、但
    原文有多行时（比如每行"2.1 线性回归""2.2 最小二乘法"），按行拆。都拆不出来返回空列表，
    交给调用方按"没有编号的整段描述"处理"""
    text = _strip_focus(description)
    starts = []  # [(编号起始位置, 内容起始位置)]
    expected = 1
    for m in _MARKER_RE.finditer(text):
        paren, bracket, circled, dotted, bare = m.groups()
        num = ord(circled) - 0x245F if circled else int(paren or bracket or dotted or bare)
        if num != expected:
            continue
        before = text[:m.start()].rstrip(" \t")
        prev = before[-1:] if before else ""
        after = text[m.end():m.end() + 1]
        at_boundary = prev == "" or prev in _BOUNDARY_CHARS
        # 前一个字是汉字或右括号/右引号（"案例库）3.基础"），且编号后不是数字
        glued_to_cjk = ("\u4e00" <= prev <= "\u9fff" or prev in "）)】]”’\"'") and after and not after.isdigit()
        # 括号/方括号/圈号这些写法本身就不会跟正文里的数字混淆，不要求前面是分隔符；
        # "1实现"这种没有任何标点的写法最容易误认，只认行首的
        if (paren or bracket or circled) or (dotted and (at_boundary or glued_to_cjk)) or (bare and at_boundary):
            starts.append((m.start(), m.end()))
            expected += 1
    items = []
    if len(starts) >= 2:
        for i, (_, content_start) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            items.append(text[content_start:end])
    else:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) >= 2:
            items = [_LINE_PREFIX_RE.sub("", ln, count=1) for ln in lines]
    items = [it.strip().rstrip(_TRAILING_PUNCT).strip() for it in items]
    return [it for it in items if it]


def fallback_points(description: str) -> list[str]:
    """还没提取过时的兜底：有编号就取前MAX_POINTS条，没有就用去掉重点/难点后的整段原文"""
    items = split_items(description)
    if items:
        return items[:MAX_POINTS]
    text = _strip_focus(description).rstrip(_TRAILING_PUNCT)
    text = re.sub(r"^\s*1\s*[.．、]\s*", "", text)  # 只有一条编号时去掉那个"1."
    return [text] if text else []


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def extract_graph_points(case, mappings: list) -> dict[int, list[str]]:
    """一次模型调用，给这批知识点关联各自挑出跟案例最相关的条目。
    返回 {mapping.id: [条目原文, ...]}；某条没有拿到合法结果的就不在返回值里（调用方保持
    graph_points为空，画图时走fallback_points）。只有一条编号/很短的描述不需要模型判断，
    直接用原文。"""
    results: dict[int, list[str]] = {}
    ask = []  # [(mapping, items或None)]
    for m in mappings:
        desc = m.knowledge_point.description
        items = split_items(desc)
        if not items and len(_strip_focus(desc)) <= 40:
            results[m.id] = fallback_points(desc)
        else:
            ask.append((m, items))
    if not ask:
        return results

    require_api_key()
    payload = [
        {"id": m.id, "知识单元": m.knowledge_point.chapter or "",
         **({"编号条目": {str(i): it for i, it in enumerate(items, start=1)}} if items
            else {"整段描述": _strip_focus(m.knowledge_point.description)})}
        for m, items in ask
    ]
    prompt = build_graph_points_prompt(
        case.title or "", (case.full_narrative or "")[:MAX_CASE_CHARS], json.dumps(payload, ensure_ascii=False, indent=2),
    )
    parsed = chat_json(GRAPH_POINTS_SYSTEM_PROMPT, prompt, max_tokens=1500, model=CHAT_MODEL, temperature=0.1)

    by_id = {str(r.get("id")): r for r in parsed.get("results", []) if isinstance(r, dict)}
    for m, items in ask:
        r = by_id.get(str(m.id))
        if not r:
            continue
        if items:
            picked = []
            for n in r.get("items") or []:
                try:
                    idx = int(n)
                except (TypeError, ValueError):
                    continue
                if 1 <= idx <= len(items) and items[idx - 1] not in picked:
                    picked.append(items[idx - 1])
            if picked:
                # 按原文顺序排列，跟描述里的条目顺序一致
                results[m.id] = sorted(picked, key=items.index)[:MAX_POINTS]
        else:
            excerpts = [str(e).strip().rstrip(_TRAILING_PUNCT) for e in (r.get("excerpts") or [])]
            source = _norm(m.knowledge_point.description)
            valid = [e for e in excerpts if e and _norm(e) in source]  # 必须是原文里真有的文字
            if valid:
                results[m.id] = valid[:MAX_POINTS]
            elif excerpts:
                logger.warning(f"知识点关联{m.id}：模型摘出的片段不在原文里，改用兜底：{excerpts}")
    return results
