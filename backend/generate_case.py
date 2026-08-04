"""
调用 Qwen（阿里云 DashScope）生成案例草稿
"""
import json
import re

import openai

from prompts import (
    CASE_GENERATION_SYSTEM_PROMPT, build_user_prompt,
    CASE_ENRICH_SYSTEM_PROMPT, build_enrich_prompt,
)
from qwen_client import get_client, require_api_key, CHAT_MODEL


def _extract_json(raw_text: str) -> dict:
    """兜底处理：万一模型输出带了markdown代码块包裹，去掉再解析"""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return json.loads(text.strip())


def _chat_json(system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    """调用Qwen，要求输出JSON对象；DashScope兼容模式支持response_format强制JSON输出"""
    try:
        response = get_client().chat.completions.create(
            model=CHAT_MODEL,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except openai.APIError as e:
        raise ValueError(f"调用 Qwen API 失败: {str(e)}")

    raw_text = response.choices[0].message.content or ""

    try:
        return _extract_json(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"模型输出不是合法JSON，需要人工检查原始输出。解析错误: {e}\n"
            f"原始输出前500字: {raw_text[:500]}"
        )


def generate_case_draft(case_code: str, dimension: str, materials: list[dict]) -> dict:
    """
    materials: [{"id": int, "url": str, "title": str, "text": str}, ...]
    返回解析后的案例草稿 dict（结构见 prompts.py 中的JSON schema）
    """
    if not materials:
        raise ValueError("没有可用的素材，无法生成案例（避免模型凭空编造）")

    require_api_key()
    user_prompt = build_user_prompt(case_code, dimension, materials)
    return _chat_json(CASE_GENERATION_SYSTEM_PROMPT, user_prompt, max_tokens=4000)


def enrich_case_with_knowledge(case: dict, accepted_mappings: list[dict]) -> dict:
    """
    根据人工采纳的知识点关联，补充/更新案例的"适用课程举例"与"教学设计"两个字段。
    case: Case.to_dict() 的结果
    accepted_mappings: [{"course_name":.., "chapter":.., "description":.., "suggestion_text":..}, ...]
    """
    if not accepted_mappings:
        raise ValueError("没有已采纳的知识点关联，无法据此补充案例")

    require_api_key()
    prompt = build_enrich_prompt(case, accepted_mappings)
    return _chat_json(CASE_ENRICH_SYSTEM_PROMPT, prompt, max_tokens=2000)
