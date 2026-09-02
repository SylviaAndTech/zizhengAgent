"""
Writer→Judge→Reviser 多Agent循环 —— LangGraph 状态图搭建 + 三个节点的实现。

流程：
    写作者(writer) → 评审者(judge) → [路由判断] → 通过/达到轮次上限 → 结束
                                              → 触发事实忠实性否决 → 打回写作者重写
                                              → 结构硬伤（引用/篇幅格式） → 修订者局部修
                                              → 仅文笔类问题未达标 → 修订者局部修
    修订/重写之后都回到评审者，形成循环，直到通过或达到最大轮次。

跟参考设计相比的两处改动：
1. 参考设计里的 writer节点每次被调用都会重新跑一遍"事实提炼"，注释说"已有facts时应该
   跳过"，但代码本身并没有做这个判断（是原型自身的实现和文档不一致）。这里的事实提炼
   在 generate_case.py 里作为独立步骤只跑一次，writer_agent 只消费已经提炼好的
   facts/narrative_arc，无论是首次写初稿还是被否决后重写，都不重新抽取事实。
2. 新增了一层不依赖LLM judge自觉的纯Python结构校验（style_rubric.validate_structure），
   在路由函数里单独判断——引用标注格式、两段式结构、篇幅这些是下游"点击引用定位到原文"
   功能能否工作的硬约束，checked独立于rubric打分之外。

所有节点都是同步函数（不用async），走现有的 qwen_client.chat_json/chat_text 走
Qwen/DeepSeek（跟generate_case.py其余步骤保持同一套"防截断超时"的调用方式），
LangGraph的StateGraph同样支持同步的.invoke()，不需要为此引入异步事件循环。
"""
import os

from langgraph.graph import StateGraph, START, END

from case_agent_state import CaseState
from case_narrative_examples import get_examples_by_dimension
from prompts import CASE_NARRATIVE_STYLE_PROMPT, build_narrative_user_message, build_reviser_prompt
from qwen_client import chat_text, chat_json, CHAT_MODEL, NARRATIVE_MODEL
from style_rubric import (
    build_judge_prompt, build_judge_user_message, compute_pass, validate_structure,
)

# 附件默认给3轮；生产环境正文写作/修订单次调用都是max_tokens=8000的重模型调用，
# 3轮意味着最多"1次写作 + 3次评审 + 最多3次修订/重写"，耗时和成本会明显超过现有
# 单次流水线，默认给2轮更稳妥。可通过环境变量调整。
DEFAULT_MAX_ITERATIONS = int(os.environ.get("CASE_NARRATIVE_MAX_ITERATIONS", "2"))

JUDGE_SYSTEM = build_judge_prompt()

REVISER_ROLE_NOTE = (
    "\n\n你现在的角色是修订者：在保留原文优点的基础上，只针对评审给出的具体问题精修，"
    "不要推倒重来，不要改动没有问题的段落。"
)


def writer_agent(state: CaseState) -> dict:
    """写作者：拿事实骨架+行文脉络建议写正文。首次调用写初稿；因事实忠实性被否决后
    打回重写时，复用同一份facts（不重新提炼事实），只是重新创作一遍正文。"""
    facts = state["facts"]
    narrative_arc = state.get("narrative_arc")
    topic_hint = facts.get("主题", "") if isinstance(facts, dict) else ""

    messages = []
    for ex in get_examples_by_dimension(None, limit=2, topic_hint=topic_hint):
        messages.append({"role": "user", "content": build_narrative_user_message(ex["example_facts"])})
        messages.append({"role": "assistant", "content": ex["example_output"]})
    messages.append({"role": "user", "content": build_narrative_user_message(facts, narrative_arc)})

    narrative = chat_text(
        CASE_NARRATIVE_STYLE_PROMPT, messages, max_tokens=8000,
        model=NARRATIVE_MODEL, temperature=0.8,
    )

    is_first_pass = not state.get("narrative")
    update = {
        "narrative": narrative,
        "iteration": 1,
        "history": [
            f"[写作者] 第{state.get('iteration', 0) + 1}轮："
            f"{'生成初稿' if is_first_pass else '因事实问题重写'}，{len(narrative)}字"
        ],
    }
    if is_first_pass:
        update["first_draft"] = narrative
    return update


def judge_agent(state: CaseState) -> dict:
    """评审者：按rubric打分+事实忠实性一票否决，另外单独跑一遍纯Python的结构校验
    （不依赖LLM judge是否自觉遵守引用格式/两段式结构/篇幅要求）。"""
    narrative = state["narrative"]
    result = chat_json(
        JUDGE_SYSTEM, build_judge_user_message(state["facts"], narrative),
        max_tokens=2000, model=CHAT_MODEL, temperature=0.2,
    )

    scores = result.get("scores", {})
    veto = result.get("factual_fidelity", {}).get("veto_triggered", False)
    fabrications = result.get("factual_fidelity", {}).get("suspected_fabrications", [])
    structure_errors = validate_structure(narrative)

    verdict = compute_pass(scores, veto_triggered=veto)

    return {
        "scores": scores,
        "issues": result.get("issues", {}),
        "veto_triggered": veto,
        "suspected_fabrications": fabrications,
        "revision_checklist": result.get("revision_checklist", []),
        "weighted_ratio": verdict.get("ratio"),
        "passed": verdict["passed"] and not structure_errors,
        "structure_errors": structure_errors,
        "history": [
            f"[评审者] 得分比 {verdict.get('ratio')}，通过={verdict['passed']}，"
            f"否决={veto}，结构问题={structure_errors or '无'}"
        ],
    }


def reviser_agent(state: CaseState) -> dict:
    """修订者：针对评审给出的具体问题清单局部精修，不推倒重来。若上一轮是因为结构校验
    不通过（引用格式/两段式/篇幅），把这些问题也并入修改清单，确保修订者能看到并修正。"""
    checklist = list(state.get("revision_checklist") or [])
    checklist.extend(state.get("structure_errors") or [])

    issues_str = "\n".join(
        f"- {k}: {v}" for k, v in (state.get("issues") or {}).items() if v and v != "无"
    ) or "（无具体文笔问题，仅需修正下面的结构问题）"
    checklist_str = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(checklist)) or "（无）"

    prompt = build_reviser_prompt(state["narrative"], state["facts"], issues_str, checklist_str)
    narrative = chat_text(
        CASE_NARRATIVE_STYLE_PROMPT + REVISER_ROLE_NOTE,
        [{"role": "user", "content": prompt}], max_tokens=8000,
        model=NARRATIVE_MODEL, temperature=0.7,
    )

    return {
        "narrative": narrative,
        "iteration": 1,
        "history": [f"[修订者] 第{state.get('iteration', 0)}轮修订完成，{len(narrative)}字"],
    }


def route_after_judge(state: CaseState) -> str:
    """评审后的路由决策。判断顺序（跟附件的end/rewrite/revise三分支相比，新增了
    结构硬伤这一档，且优先级最高——格式问题必须修，不管rubric打分高低）："""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    # 结构硬伤（引用标注/两段式/篇幅）：必须修，不算一次"事实否决"，走修订分支
    if state.get("structure_errors"):
        if iteration >= max_iter:
            return "end"  # 达到轮次上限，交人工处理，不再无限循环
        return "revise"

    # 达标 或 达到轮次上限 → 结束（未达标时把当前最好版本+评审反馈交给人工终审）
    if state.get("passed") or iteration >= max_iter:
        return "end"

    # 触发事实忠实性否决 → 打回写作者重写（修订解决不了编造内容的问题）
    if state.get("veto_triggered"):
        return "rewrite"

    # 纯文笔类问题未达标 → 交修订者局部精修
    return "revise"


def build_case_graph(on_stage=None):
    """工厂函数：编译一次LangGraph状态图。闭包捕获on_stage回调用于展示"当前跑到第几轮
    评审/修订"的进度——跟generate_case.py现有的_stage()闭包用法保持同一种风格，不用
    contextvar（那是chat_agent.py为了跨线程传递SSE进度专门用的，这里不需要）。
    每次调用都现编译一次图，编译开销可忽略，不做成模块级单例——这里每次都要闭包不同的
    on_stage，跟chat_agent.py里`_agent`那种无状态工具agent单例是不同的场景。"""

    def _stage(name: str):
        if on_stage:
            on_stage(name)

    def _writer_node(state: CaseState) -> dict:
        _stage("正文初稿" if not state.get("narrative") else f"因事实问题重写(第{state.get('iteration', 0) + 1}轮)")
        return writer_agent(state)

    def _judge_node(state: CaseState) -> dict:
        _stage(f"AI评审(第{state.get('iteration', 0)}轮)")
        return judge_agent(state)

    def _reviser_node(state: CaseState) -> dict:
        _stage(f"内容修订(第{state.get('iteration', 0)}轮)")
        return reviser_agent(state)

    graph = StateGraph(CaseState)
    graph.add_node("writer", _writer_node)
    graph.add_node("judge", _judge_node)
    graph.add_node("reviser", _reviser_node)

    graph.add_edge(START, "writer")
    graph.add_edge("writer", "judge")
    graph.add_conditional_edges(
        "judge", route_after_judge,
        {"end": END, "revise": "reviser", "rewrite": "writer"},
    )
    graph.add_edge("reviser", "judge")

    return graph.compile()
