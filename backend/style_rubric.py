"""
思政案例正文评审用的评分表(Rubric) + 结构校验。

用途：
1. 作为 case_agents.py 里 judge_agent 的评分依据，让评审模型对生成的正文按维度打分
   并给出具体问题句，供后续路由到修订/重写。
2. 把"真实性"单独设为一票否决项——只要出现编造的具体事实，无论其余维度打分多高，
   立即判定不合格。
3. validate_structure() 是纯Python的结构校验，检查判定不应该完全依赖LLM judge的自觉——
   引用标注格式（[素材N:定位短语]）和【背景阅读】【详细内容】两段式结构是下游"点击引用
   定位到原文"功能能否工作的硬前提，必须用代码兜底，不能只靠prompt自觉，LLM judge打分
   再高也不能替代这层检查。

设计说明：
- 每个维度都给出"什么样算高分/什么样算低分"的可复核锚点，而不是"写得好"这种主观描述。
- 锚点直接来自 prompts.py 里 CASE_NARRATIVE_STYLE_PROMPT 总结的真实写法，可复现、可核查。
- 把"真实性"单独作为一票否决项——真实性不达标，无论其余多高分都判定"不合格"。

评分说明：
- 6 个加权维度，每项 1-5 分，满分 30。
- 1 个否决项（真实性），不计入加权分，但只要触发就直接判定"不合格"，无论其余分数多高。
- 建议 pass 线：加权得分 ≥ 22 且真实性未触发否决。
"""
import json
import re

# ---- 权重：人物具体性和真实性是重中之重，权重更高 ----
RUBRIC = [
    {
        "id": "human_concreteness",
        "name": "人物具体性（最重要）",
        "weight": 2.0,
        "question": "文中的宏观成果、数据、政策，是否都能落到'具体的人'身上，是否用了'宏观数据+具体人物+命运转折'的写法？",
        "anchors": {
            "5": "几乎每个重要成果都能落到具体人物上，不是'转述'（如把医疗费用从7-8万降到3000多元。从放弃治疗到治得起病，人物有名有姓有身份。",
            "3": "有部分成果能落到人物，但仍有明显段落停在纯宏观数据和政策罗列上。",
            "1": "通篇几乎都是宏观数据和政策罗列，看不到具体的人，读起来像工作总结而非案例。",
        },
    },
    {
        "id": "authentic_quotes",
        "name": "人物原话引用",
        "weight": 1.5,
        "question": "是否引用了素材中的真实人物直接引语，是否优先选用了带着生活质感、乡土气息的老百姓大白话？",
        "anchors": {
            "5": "多处使用真实引语，不乏类似'手款多了，房子大了，票子百了''现在我觉得比种人还香'这类朴素而有感染力的大白话。",
            "3": "有少量引语，但偏官方、偏书面，缺少鲜活的百姓语言。",
            "1": "几乎没有直接引语，或引语明显是拼凑/改写的（未在素材中出现）。",
        },
    },
    {
        "id": "opening_hook",
        "name": "开篇吸引力",
        "weight": 1.0,
        "question": "开篇是否用了打动人的瞬间、人物话语、具体困境或悬念开头，还是落入'XX年，XX启动了XX'的年间体？",
        "anchors": {
            "5": "开篇有画面感、有悬念或有明确认领，如以当事人一句'如果知道会遇到这么多困难，当时甚至可能不放开始'开头，或以设问/时空对比开头。",
            "3": "开篇平稳但不出彩，交代了背景但铺开一般。",
            "1": "开篇是典型年间体，'XX年，XX市启动了XX改革'，毫无吸引力。",
        },
    },
    {
        "id": "narrative_arc",
        "name": "叙事弧线与详略",
        "weight": 1.0,
        "question": "是否用时间线/阶段搭出了起承转合的叙事弧线，长短句是否交替、有节奏，还是并列罗列知识点？",
        "anchors": {
            "5": "有明确的叙事弧线（如从'治乱'到'治病'到'治未病'三阶段），段落间推进感强，长短句交替。",
            "3": "结构清楚但偏平铺直叙，缺少推进的张力。",
            "1": "各段落是并列的知识点/成果罗列，没有故事内在的推进逻辑。",
        },
    },
    {
        "id": "vivid_language",
        "name": "语言形象化",
        "weight": 1.0,
        "question": "是否用了贴切的比喻来拆解抽象概念或政策机制，比喻是否自然、不浮夸？",
        "anchors": {
            "5": "多次使用形象化表达（如'牵鼻子''第一枪''荆棘丛生'），让政策机制有了画面感。",
            "3": "语言平实准确，但缺少让人眼前一亮的形象化表达。",
            "1": "满是抽象术语和政策黑话，读起来枯燥、疏离感强。",
        },
    },
    {
        "id": "no_cliche",
        "name": "免俗套（无套话反复）",
        "weight": 1.5,
        "question": "文章中段是否反复出现'这一……体现了/彰显了/展现了/生动诠释了'这类总结性句式，价值升华是否克制、只在结尾自然出现一次？",
        "anchors": {
            "5": "全篇几乎没有中段套话，价值升华集中在结尾一处自然出现，不贴标签、不喊口号。",
            "3": "有个别套话或中段升华，但不严重。",
            "1": "通篇反复'这一……体现了……''生动诠释了……'，说教感强，存在贴标签和喊口号。",
        },
    },
]

# ---- 一票否决项：真实性 ----
VETO_CRITERION = {
    "id": "factual_fidelity",
    "name": "事实忠实性（一票否决）",
    "question": "文中所有时间、地点、人物姓名、机构、数据、事件经过、人物原话，是否都能在给定素材中找到依据，有没有为了'生动'而编造的具体事实？",
    "veto_rule": "只要发现任何一处素材中不存在的具体事实（比如新造的人名/数据/引语/举措等），立即判定不合格，无论其余多好；字面细节没有明确标注来源的，也应标记出来要求人工核实。",
}

# ---- 计算参数 ----
MAX_WEIGHTED_SCORE = sum(c["weight"] * 5 for c in RUBRIC)  # 满分
PASS_THRESHOLD = 0.75  # 加权得分达到75%视为通过（可自行调整）

RUBRIC_BY_ID = {c["id"]: c for c in RUBRIC}

# ---- 结构硬约束：引用标注格式、两段式结构、篇幅 ----
_CITATION_RE = re.compile(r"\[素材\d+[:：][^\[\]]+\]")
_UNBALANCED_BRACKET_RE = re.compile(r"[\[\]]")
MIN_DETAIL_CHARS = 2200


def validate_structure(narrative: str) -> list[str]:
    """纯Python结构校验，不依赖LLM判断。返回发现的问题列表，空列表表示结构合格。
    检查的都是下游功能（引用定位、两段式展示）能否正常工作的硬约束，不是文笔好坏，
    LLM judge打分再高也不能替代这层检查——所以在route_after_judge里，只要这里有问题，
    不管rubric通过与否都要强制走修订分支。"""
    errors = []

    if "【背景阅读】" not in narrative:
        errors.append("缺少【背景阅读】小标题")
    if "【详细内容】" not in narrative:
        errors.append("缺少【详细内容】小标题")

    if "【详细内容】" in narrative:
        detail = narrative.split("【详细内容】", 1)[1]
        detail_len = len(detail.strip())
        if detail_len < MIN_DETAIL_CHARS:
            errors.append(f"详细内容部分只有{detail_len}字，少于要求的{MIN_DETAIL_CHARS}字")

    # 方括号总数必须是偶数（每个引用标注都要有一对完整的[...]），且能被引用格式正则
    # 完整匹配掉——如果匹配掉所有[素材N:定位短语]之后还剩下方括号，说明存在格式被破坏的标注
    remaining = _CITATION_RE.sub("", narrative)
    leftover_brackets = _UNBALANCED_BRACKET_RE.findall(remaining)
    if leftover_brackets:
        errors.append(f"存在格式不完整或不规范的引用标注（残留{len(leftover_brackets)}个未配对方括号）")

    return errors


def compute_pass(scores: dict, veto_triggered: bool) -> dict:
    """
    scores: {criterion_id: 1-5 的整数}
    veto_triggered: 真实性否决是否触发
    返回判定结果。
    """
    if veto_triggered:
        return {
            "passed": False,
            "reason": "触发事实忠实性一票否决（存在素材中不存在的编造内容），必须打回重写或交人工核定。",
            "weighted_score": None,
        }

    weighted = sum(RUBRIC_BY_ID[cid]["weight"] * s for cid, s in scores.items())
    ratio = weighted / MAX_WEIGHTED_SCORE
    return {
        "passed": ratio >= PASS_THRESHOLD,
        "reason": f"加权得分比例 {ratio:.0%}（阈值 {PASS_THRESHOLD:.0%}）",
        "weighted_score": round(weighted, 1),
        "max_score": MAX_WEIGHTED_SCORE,
        "ratio": round(ratio, 3),
    }


def build_judge_prompt() -> str:
    """生成给评审模型的完整系统提示词。"""
    lines = [
        "你是一位严格且专业的思政案例文笔评审专家。",
        "请对下面给出的《案例正文》按评分表逐项打分（1-5的整数），",
        "并针对每一项打分**具体的问题所在**（引用原文中的问题句子），而不是泛泛而谈。",
        "",
        "# 评分维度",
    ]
    for c in RUBRIC:
        lines.append(f"\n《{c['name']}》(id: {c['id']}, 权重{c['weight']})")
        lines.append(f"评判问题：{c['question']}")
        lines.append(f"  5分：{c['anchors']['5']}")
        lines.append(f"  3分：{c['anchors']['3']}")
        lines.append(f"  1分：{c['anchors']['1']}")

    lines.append(f"\n# 一票否决项")
    lines.append(f"《{VETO_CRITERION['name']}》(id: {VETO_CRITERION['id']})")
    lines.append(f"评判问题：{VETO_CRITERION['question']}")
    lines.append(f"规则：{VETO_CRITERION['veto_rule']}")

    lines.append("""
# 待评审材料由用户消息给出，格式为"【事实骨架】...【待评审案例正文】..."。
# 输出格式（只输出JSON，不要有其他文字，不要用markdown代码块包裹）：
{
  "scores": {
    "human_concreteness": 整数1-5,
    "authentic_quotes": 整数1-5,
    "opening_hook": 整数1-5,
    "narrative_arc": 整数1-5,
    "vivid_language": 整数1-5,
    "no_cliche": 整数1-5
  },
  "issues": {
    "human_concreteness": "指出具体问题（引用原文问题句），若无问题写'无'",
    "authentic_quotes": "...",
    "opening_hook": "...",
    "narrative_arc": "...",
    "vivid_language": "...",
    "no_cliche": "..."
  },
  "factual_fidelity": {
    "veto_triggered": true 或 false,
    "suspected_fabrications": ["列出疑似编造或素材中未支持的具体内容，供人工核定；若无留空数组"]
  },
  "revision_checklist": ["按优先级排序的、具体可执行的修改清单"]
}""")
    return "\n".join(lines)


def build_judge_user_message(facts: dict, narrative: str) -> str:
    """跟build_judge_prompt()（system prompt）配套的user消息，把待评审的事实骨架和
    正文交给评审模型。事实骨架用于核对真实性（正文里的具体事实是否都能在骨架中找到依据，
    有没有超出这个范围编造），不是评分依据本身。"""
    facts_str = json.dumps(facts, ensure_ascii=False, indent=2)
    return (
        f"【事实骨架】（用于核对事实忠实性，正文不能超出此范围编造）\n{facts_str}\n\n"
        f"【待评审案例正文】\n{narrative}\n\n"
        "请严格按系统提示里的JSON格式输出评分。"
    )
