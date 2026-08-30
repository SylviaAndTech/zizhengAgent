"""
调用 Qwen（阿里云 DashScope）生成案例草稿

案例正文用四段式流水线生成（事实提炼→正文初稿→去AI味改写→结构化字段），而不是像
enrich_case_with_knowledge那样一次JSON调用搞定——正文是最吃文笔的部分，跟其余结构化
字段（教学目标、评价指标等偏公文语气）挤在同一次"填JSON表单"里生成，模型会不自觉把
整体语域拉平、变成正确但无趣的八股文，客户反馈的"案例过于AI"根源就在这里。
详见 prompts.py 里 FACT_EXTRACTION/CASE_NARRATIVE_STYLE/CASE_STRUCTURED_FIELDS 三块
prompt 上面的注释。
"""
import json
import re

import openai

from prompts import (
    CASE_ENRICH_SYSTEM_PROMPT, build_enrich_prompt,
    FACT_EXTRACTION_SYSTEM_PROMPT, build_fact_extraction_prompt,
    NARRATIVE_ARC_SYSTEM_PROMPT, build_narrative_arc_prompt,
    CASE_NARRATIVE_STYLE_PROMPT, build_narrative_user_message, build_ai_flavor_revision_prompt,
    CASE_STRUCTURED_FIELDS_SYSTEM_PROMPT, build_structured_fields_prompt,
)
from case_narrative_examples import get_examples_by_dimension
from qwen_client import get_client, require_api_key, CHAT_MODEL, NARRATIVE_MODEL


def _extract_json(raw_text: str) -> dict:
    """兜底处理：万一模型输出带了markdown代码块包裹，去掉再解析"""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return json.loads(text.strip())


def _chat_json(
    system_prompt: str, user_prompt: str, max_tokens: int,
    model: str = CHAT_MODEL, temperature: float | None = None,
) -> dict:
    """调用Qwen，要求输出JSON对象；DashScope兼容模式支持response_format强制JSON输出。
    用stream=True请求，避免生成内容较长时模型思考耗时过长、连接被中间层判定超时挂断；
    但这里的返回值本来就要整体丢给json.loads解析，用户也看不到逐字输出的过程
    （这个函数是被案例生成/知识点补充接口和AI助手的工具调用在后台使用，不是聊天窗口直接展示的内容），
    所以流式只用来提高请求稳定性，还是要把所有chunk拼完整再解析。"""
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        stream = get_client().chat.completions.create(**kwargs)
        raw_text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices
        )
    except openai.APIError as e:
        raise ValueError(f"调用 API 失败: {str(e)}")

    try:
        return _extract_json(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"模型输出不是合法JSON，需要人工检查原始输出。解析错误: {e}\n"
            f"原始输出前500字: {raw_text[:500]}"
        )


def _chat_text(
    system_prompt: str, messages: list[dict], max_tokens: int,
    model: str, temperature: float | None = None,
) -> str:
    """跟_chat_json同样的流式拼接写法，但不强制JSON输出，返回纯文本。案例正文这类创造性
    写作不适合用response_format=json_object——会把模型拉进"填表单"模式，行文容易被拉匀、
    变呆，所以单独拆出来一个纯文本版本"""
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        stream=True,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        stream = get_client().chat.completions.create(**kwargs)
        raw_text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices
        )
    except openai.APIError as e:
        raise ValueError(f"调用 API 失败: {str(e)}")

    if not raw_text.strip():
        raise ValueError("模型没有返回任何正文内容")
    return raw_text.strip()


def _extract_facts(case_code: str, materials: list[dict]) -> dict:
    """第①步：从原始素材提炼事实骨架（低温，追求准确不追求文采）。
    事实骨架要保留人物原话+细节+来源标注，素材丰富时很容易超过3000 token被截断成
    半截JSON——踩过这个坑（跟syllabus_vision_ocr.py早前那次max_tokens截断是同一类问题），
    给足够宽裕的余量。现在正文要求写到2200~3000字（详见CASE_NARRATIVE_STYLE_PROMPT的篇幅
    要求），骨架需要提供的人物/数据/细节条目相应更多，max_tokens给到8000留足空间"""
    prompt = build_fact_extraction_prompt(case_code, materials)
    return _chat_json(FACT_EXTRACTION_SYSTEM_PROMPT, prompt, max_tokens=8000, temperature=0.1)


def _suggest_narrative_arc(facts: dict) -> str:
    """折中方案：给②一个轻量的"行文脉络建议"（先讲什么、再讲什么、怎么收尾的提纲式顺序，
    不含具体句子、不含引用标注），单独用中等温度（0.3）生成，不跟①的精确摘抄共用一次调用
    ——摘抄需要的低温（0.1，保证定位短语跟原文逐字匹配）和组织顺序建议需要的一点创造力
    如果硬塞进同一次调用，两者会互相拖累：温度压低摘抄准了但骨架建议会很呆板，温度调高
    骨架建议更合理但摘抄精度会下降。用CHAT_MODEL而不是NARRATIVE_MODEL，因为这一步是
    "分析型"的顺序编排，不是正文本身的文学化写作，不需要为此多花钱用强模型。"""
    # 要求的是100~200字一段话，但模型经常会写超；这里是纯文本调用（不是_chat_json），
    # 截断了不会像JSON模式那样报JSONDecodeError，只会安安静静地返回一段被腰斩的建议，
    # 没有任何报错信号——给足够宽裕的余量，避免这种不容易第一时间发现的静默截断
    prompt = build_narrative_arc_prompt(facts)
    return _chat_text(
        NARRATIVE_ARC_SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=1500,
        model=CHAT_MODEL, temperature=0.3,
    )


def _write_narrative_draft(facts: dict, narrative_arc: str | None = None) -> str:
    """第②步：拿事实骨架（+可选的行文脉络建议）写正文初稿（高温+风格规则+few-shot范文，
    用更强的NARRATIVE_MODEL）。
    few-shot范文现在还不按维度筛选——这一步案例最终会归到哪个思政维度，要等第④步结合
    定稿正文才判断得出来，第②步阶段还不知道，所以get_examples_by_dimension传None，
    退化成"现有全部范文都当通用文笔示范"，不影响使用（现在范文库本来也就2篇）"""
    messages = []
    for ex in get_examples_by_dimension(None, limit=2):
        messages.append({"role": "user", "content": build_narrative_user_message(ex["example_facts"])})
        messages.append({"role": "assistant", "content": ex["example_output"]})
    messages.append({"role": "user", "content": build_narrative_user_message(facts, narrative_arc)})
    # 正文要求2200~3000字，中文在DeepSeek系tokenizer上大约1字对应1~1.5 token，
    # 再算上几十个[素材N:定位短语]标注的开销，max_tokens给到8000留足余量，
    # 避免正文写到一半被截断（比空间不够更明显的信号是JSON模式下的Unterminated
    # string报错，这里是纯文本模式，截断只会表现为正文戛然而止，不容易第一时间发现）
    return _chat_text(
        CASE_NARRATIVE_STYLE_PROMPT, messages, max_tokens=8000,
        model=NARRATIVE_MODEL, temperature=0.8,
    )


def _revise_ai_flavor(narrative_draft: str, facts: dict) -> str:
    """第③步：去AI味二次改写——对照写作规则自查初稿、把违规的句子改掉，
    不改事实、不删引用标注"""
    prompt = build_ai_flavor_revision_prompt(narrative_draft, facts)
    return _chat_text(
        CASE_NARRATIVE_STYLE_PROMPT, [{"role": "user", "content": prompt}], max_tokens=8000,
        model=NARRATIVE_MODEL, temperature=0.7,
    )


def _generate_structured_fields(case_code: str, materials: list[dict], facts: dict, narrative_final: str) -> dict:
    """第④步：结合已定稿的正文，补全标题/教学目标/课程思政元素/适用课程/教学设计/评价/延伸阅读"""
    prompt = build_structured_fields_prompt(case_code, materials, facts, narrative_final)
    return _chat_json(CASE_STRUCTURED_FIELDS_SYSTEM_PROMPT, prompt, max_tokens=4000)


def generate_case_draft(case_code: str, materials: list[dict], on_stage=None) -> dict:
    """
    materials: [{"id": int, "url": str, "title": str, "text": str}, ...]
    on_stage: 可选回调 (stage_name: str) -> None，在四个阶段各自开始前调用一次，
    用于调用方展示"当前跑到第几步"（比如AI助手聊天里的实时进度提示）；不传就是纯粹
    的静默调用，不影响返回结果。
    返回解析后的案例草稿 dict（结构见 prompts.py 中的JSON schema），额外多一个
    full_narrative_draft字段——去AI味改写之前的正文初稿，跟改写后的full_narrative
    一起返回，方便调用方两份都存下来、都能在案例详情里对照查看。
    所属思政维度不是入参——每个案例现在都要求对五个官方思政维度逐一给出摘录表述+解释
    （sizheng_elements.五维度阐释），"对应维度"（书稿五个章节分类里的哪一个）改由模型结合
    案例内容自主判断给出。
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
    _stage("正文初稿")
    narrative_draft = _write_narrative_draft(facts, narrative_arc)
    _stage("去AI味改写")
    narrative_final = _revise_ai_flavor(narrative_draft, facts)
    _stage("结构化字段")
    structured = _generate_structured_fields(case_code, materials, facts, narrative_final)

    structured["full_narrative"] = narrative_final
    structured["full_narrative_draft"] = narrative_draft
    return structured


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
    return _chat_json(CASE_ENRICH_SYSTEM_PROMPT, prompt, max_tokens=2000)
