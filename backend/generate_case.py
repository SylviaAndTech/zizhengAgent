"""
调用 Qwen（阿里云 DashScope）生成案例草稿

案例正文生成分两大段：①事实提炼（低温，追求准确）；②正文写作与评审——由
case_agent_graph.py 里的 writer/judge/reviser 三节点循环完成初稿写作、rubric打分、
按问题类型路由到局部修订或整体重写，收敛后再进入③结构化字段（标题/教学目标/
延伸阅读等偏公文语气的字段，跟叙事体正文分开生成，避免模型把整体语域拉平、变成正确但
无趣的八股文——这也是客户反馈"案例过于AI"的根源）。
详见 prompts.py 里 FACT_EXTRACTION/CASE_NARRATIVE_STYLE/CASE_STRUCTURED_FIELDS 三块
prompt 上面的注释，以及 case_agent_graph.py/case_agents.py/style_rubric.py 里
写作-评审循环的设计说明。
④课程思政元素分析：总述+六个维度的论述段落+所属维度，篇幅长、更看重文笔，单独一步用
写正文的强模型生成（generate_sizheng_analysis），已有案例也可以单独重新生成这一部分。
⑤思考与讨论：给学生读完案例后思考的问题（generate_discussion_questions），同样可单独重新生成。
"""
from prompts import (
    CASE_ENRICH_SYSTEM_PROMPT, build_enrich_prompt,
    FACT_EXTRACTION_SYSTEM_PROMPT, build_fact_extraction_prompt,
    NARRATIVE_ARC_SYSTEM_PROMPT, build_narrative_arc_prompt,
    CASE_STRUCTURED_FIELDS_SYSTEM_PROMPT, build_structured_fields_prompt,
    TOPIC_KEYWORDS_SYSTEM_PROMPT, build_topic_keywords_prompt,
    SIZHENG_ANALYSIS_SYSTEM_PROMPT, build_sizheng_analysis_prompt,
    DISCUSSION_QUESTIONS_SYSTEM_PROMPT, build_discussion_questions_prompt,
)
from db import DIMENSIONS
from doc_writer import strip_citations
from case_agent_graph import build_case_graph, DEFAULT_MAX_ITERATIONS
from qwen_client import require_api_key, CHAT_MODEL, NARRATIVE_MODEL, chat_json, chat_text


def _extract_facts(case_code: str, materials: list[dict]) -> dict:
    """第①步：从原始素材提炼事实骨架（低温，追求准确不追求文采）。
    事实骨架要保留人物原话+细节+来源标注，素材丰富时很容易超过3000 token被截断成
    半截JSON——踩过这个坑（跟syllabus_vision_ocr.py早前那次max_tokens截断是同一类问题），
    给足够宽裕的余量。现在正文要求写到2200~3000字（详见CASE_NARRATIVE_STYLE_PROMPT的篇幅
    要求），骨架需要提供的人物/数据/细节条目相应更多，max_tokens给到8000留足空间"""
    prompt = build_fact_extraction_prompt(case_code, materials)
    return chat_json(FACT_EXTRACTION_SYSTEM_PROMPT, prompt, max_tokens=8000, temperature=0.1)


def _suggest_narrative_arc(facts: dict) -> str:
    """折中方案：给写作者一个轻量的"行文脉络建议"（先讲什么、再讲什么、怎么收尾的提纲式顺序，
    不含具体句子、不含引用标注），单独用中等温度（0.3）生成，不跟①的精确摘抄共用一次调用
    ——摘抄需要的低温（0.1，保证定位短语跟原文逐字匹配）和组织顺序建议需要的一点创造力
    如果硬塞进同一次调用，两者会互相拖累：温度压低摘抄准了但骨架建议会很呆板，温度调高
    骨架建议更合理但摘抄精度会下降。用CHAT_MODEL而不是NARRATIVE_MODEL，因为这一步是
    "分析型"的顺序编排，不是正文本身的文学化写作，不需要为此多花钱用强模型。"""
    # 要求的是100~200字一段话，但模型经常会写超；这里是纯文本调用（不是chat_json），
    # 截断了不会像JSON模式那样报JSONDecodeError，只会安安静静地返回一段被腰斩的建议，
    # 没有任何报错信号——给足够宽裕的余量，避免这种不容易第一时间发现的静默截断
    prompt = build_narrative_arc_prompt(facts)
    return chat_text(
        NARRATIVE_ARC_SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=1500,
        model=CHAT_MODEL, temperature=0.3,
    )


def _generate_structured_fields(case_code: str, materials: list[dict], facts: dict, narrative_final: str) -> dict:
    """第③步：结合已定稿的正文，补全标题/教学目标/课程思政元素/适用课程/教学设计/评价/延伸阅读"""
    prompt = build_structured_fields_prompt(case_code, materials, facts, narrative_final)
    return chat_json(CASE_STRUCTURED_FIELDS_SYSTEM_PROMPT, prompt, max_tokens=4000)


def generate_sizheng_analysis(case_title: str, narrative: str) -> dict:
    """第④步：课程思政元素分析，返回 {"对应维度", "总述", "维度阐述": {维度: 一段话}}。
    模型先判断案例体现了六个维度里的哪几个，只对这几个展开（不是六个都写）。
    正文里的[素材N:定位短语]引用标注先去掉，不然模型会把它们抄进分析里。
    一个维度都没写、写了不认识的维度名、对应维度不在展开的维度里，重试一次，还不行就报错"""
    require_api_key()
    prompt = build_sizheng_analysis_prompt(case_title or "", strip_citations(narrative))
    problem = ""
    for _attempt in range(2):
        result = chat_json(SIZHENG_ANALYSIS_SYSTEM_PROMPT, prompt, max_tokens=8000,
                           model=NARRATIVE_MODEL, temperature=0.5)
        sections = result.get("维度阐述") if isinstance(result, dict) else None
        sections = {k: str(v).strip() for k, v in (sections or {}).items() if str(v or "").strip()} \
            if isinstance(sections, dict) else {}
        unknown = [k for k in sections if k not in DIMENSIONS]
        if not sections or unknown:
            problem = f"维度名称不对：{'、'.join(unknown)}" if unknown else "一个维度都没有展开"
            continue
        if result.get("对应维度") not in sections:
            problem = f"对应维度（{result.get('对应维度')}）不在展开的维度里"
            continue
        if not str(result.get("总述") or "").strip():
            problem = "缺少总述"
            continue
        return {
            "对应维度": result["对应维度"],
            "总述": str(result["总述"]).strip(),
            "维度阐述": {d: sections[d] for d in DIMENSIONS if d in sections},  # 按六个维度的固定顺序
        }
    raise ValueError(f"课程思政元素分析生成结果不完整（{problem}），请重试")


def generate_discussion_questions(case_title: str, narrative: str) -> list[str]:
    """第⑤步："思考与讨论"，给学生读完案例后思考的问题（一道思考题+一道实践探究题）"""
    require_api_key()
    prompt = build_discussion_questions_prompt(case_title or "", strip_citations(narrative))
    result = chat_json(DISCUSSION_QUESTIONS_SYSTEM_PROMPT, prompt, max_tokens=1500,
                       model=NARRATIVE_MODEL, temperature=0.6)
    questions = [str(q).strip() for q in (result.get("思考与讨论") or []) if str(q).strip()]
    if not questions:
        raise ValueError("「思考与讨论」生成结果为空，请重试")
    return questions


def generate_case_draft(case_code: str, materials: list[dict], on_stage=None) -> dict:
    """
    materials: [{"id": int, "url": str, "title": str, "text": str}, ...]
    on_stage: 可选回调 (stage_name: str) -> None，在每个阶段（含写作-评审循环的每一轮）
    开始前调用一次，用于调用方展示"当前跑到第几步"（比如AI助手聊天里的实时进度提示）；
    不传就是纯粹的静默调用，不影响返回结果。
    返回解析后的案例草稿 dict（结构见 prompts.py 中的JSON schema），额外多一个
    full_narrative_draft字段——写作者给出的第一版正文（评审/修订循环开始之前），跟收敛后的
    full_narrative一起返回，方便调用方两份都存下来、都能在案例详情里对照查看。
    所属思政维度不是入参——"课程思政元素分析"一步先判断案例体现了哪几个思政维度、只论述这几个，并由模型结合
    案例内容判断"对应维度"（书稿六个维度章节里的哪一个）。
    """
    if not materials:
        raise ValueError("没有可用的素材，无法生成案例（避免模型凭空编造）")

    require_api_key()

    def _stage(name: str):
        if on_stage:
            on_stage(name)

    _stage("事实提炼")
    facts = _extract_facts(case_code, materials)
    narrative_arc = _suggest_narrative_arc(facts)

    graph = build_case_graph(on_stage=_stage)
    result = graph.invoke({
        "case_code": case_code,
        "facts": facts,
        "narrative_arc": narrative_arc,
        "iteration": 0,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "history": [],
    })
    narrative_draft = result["first_draft"]
    narrative_final = result["narrative"]

    _stage("结构化字段")
    structured = _generate_structured_fields(case_code, materials, facts, narrative_final)

    _stage("课程思政元素分析")
    structured["sizheng_elements"] = generate_sizheng_analysis(structured.get("title"), narrative_final)
    _stage("思考与讨论")
    structured["discussion_questions"] = generate_discussion_questions(structured.get("title"), narrative_final)

    structured["full_narrative"] = narrative_final
    structured["full_narrative_draft"] = narrative_draft
    return structured


def extract_topic_keywords(case_title: str, case_narrative: str) -> str:
    """从案例标题+正文提炼一段贴近知识点语域的主题关键词，专供knowledge_matching.py做
    向量检索查询用（不对外展示）——叙事体正文跟知识点的技术条目语域差异太大，直接拿
    整段正文去embedding检索效果差，见knowledge_matching._ensure_topic_keywords()。
    用CHAT_MODEL（便宜快）而不是NARRATIVE_MODEL：这是"提炼关键词"，不是文学化写作，
    不需要为此多花钱用强模型。"""
    require_api_key()
    prompt = build_topic_keywords_prompt(case_title, case_narrative)
    return chat_text(
        TOPIC_KEYWORDS_SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=300,
        model=CHAT_MODEL, temperature=0.2,
    )


def enrich_case_with_knowledge(case: dict, accepted_mappings: list[dict]) -> dict:
    """
    根据人工采纳的知识点关联，补充/更新案例的"适用课程举例"与"教学设计"两个字段。
    case: Case.to_dict() 的结果
    accepted_mappings: [{"course_name":.., "chapter":.., "description":.., "suggestion_text":..,
    "relevance_score":..}, ...]，调用方要预先按relevance_score降序排好——适用课程举例会
    对应列表里每一条各生成一条，教学设计只会结合排在第一条（相关度最高）的那条知识点来设计，
    这个"哪条最相关"的判断是调用方排序决定的，不是让模型自己从分数里挑。
    """
    if not accepted_mappings:
        raise ValueError("没有已采纳的知识点关联，无法据此补充案例")

    require_api_key()
    prompt = build_enrich_prompt(case, accepted_mappings)
    return chat_json(CASE_ENRICH_SYSTEM_PROMPT, prompt, max_tokens=2000)
