"""
统一的 Qwen（阿里云 DashScope，OpenAI 兼容模式）客户端与模型配置。
所有需要调用大模型/向量模型的地方都从这里取 client 和模型名，避免各处散着写。
"""
import json
import os
import re

import httpx
import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# langchain-openai(>=1.4)在检测到本机代理环境变量时，只要ChatOpenAI收到了自定义http_client
# （我们下面就是这么做的，为了绕开本机代理），就会照常算一遍它自己那套内核级TCP keepalive参数、
# 并打印"injected a custom httpx transport..."这条warning——即便这些参数根本不会被用到我们
# 自己传入的http_client上（库内部是"self.http_client or 它自己构造的client"，我们的client已经
# 非空，keepalive参数从头到尾都是死代码）。这是它1.4.1版本里的一个逻辑疏漏，不是我们代码有问题。
# 官方建议的两种消音方式之一就是设这个环境变量，效果是让它直接算出空的keepalive参数、不再触发那条
# warning，对我们本来就没用到的keepalive功能没有任何实际影响。
os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")

DASHSCOPE_BASE_URL = "https://api.siliconflow.cn/v1"

CHAT_MODEL = os.environ.get("QWEN_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
EMBEDDING_MODEL = os.environ.get("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
# 专门写案例"正文"用的模型——正文是全案例里最吃文笔的部分，值得单独换一个更强
# （也更贵、更慢）的模型，其余结构化字段继续用上面更便宜快的CHAT_MODEL
NARRATIVE_MODEL = os.environ.get("QWEN_NARRATIVE_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
MAX_TOOL_ROUNDS = int(os.environ.get("QWEN_MAX_TOOL_ROUNDS", "6"))

_client = None


def has_api_key() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def require_api_key():
    """没配key时给出清晰报错，而不是让底层SDK抛一个不知所云的错误"""
    if not has_api_key():
        raise ValueError(
            "未检测到 DASHSCOPE_API_KEY 环境变量。请先在 backend/.env 里配置key "
        )


def _no_proxy_http_client() -> httpx.Client:
    """禁止httpx读取本机代理相关环境变量（HTTP_PROXY/HTTPS_PROXY/ALL_PROXY）。
    这里直连固定的API域名，不需要走本机代理；用户本机若配置了SOCKS代理但没装socksio，
    会导致请求直接报错'Using SOCKS proxy, but the socksio package is not installed'"""
    return httpx.Client(trust_env=False)


def get_client() -> OpenAI:
    """延迟初始化：避免没配置key或网络异常时，服务器一启动就崩溃"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=DASHSCOPE_BASE_URL,
            http_client=_no_proxy_http_client(),
        )
    return _client


def get_langchain_llm():
    """LangChain用的Qwen聊天模型（走DashScope的OpenAI兼容协议），给AI助手的agent用"""
    require_api_key()
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=CHAT_MODEL,
        base_url=DASHSCOPE_BASE_URL,
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        http_client=_no_proxy_http_client(),
    )


def get_llama_index_llm():
    """LlamaIndex用的聊天模型。我们目前的检索链路只用LlamaIndex做向量存取，不靠它做生成，
    这里主要是为了把Settings.llm钉死住，避免LlamaIndex内部逻辑悄悄尝试调用真正的OpenAI。

    注意：不能用 llama_index.llms.dashscope.DashScope —— 那个类是阿里云DashScope原生SDK，
    请求硬编码打到 dashscope.aliyuncs.com，跟我们其他地方统一走的 DASHSCOPE_BASE_URL
    （硅基流动的OpenAI兼容endpoint）完全是两个不同的平台；我们的API key是硅基流动的key，
    打到阿里云自己的endpoint上必然认证失败。要用通用的OpenAI兼容客户端，把base_url显式指过去。"""
    from llama_index.llms.openai_like import OpenAILike
    return OpenAILike(
        model=CHAT_MODEL,
        api_base=DASHSCOPE_BASE_URL,
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        is_chat_model=True,
    )


def get_llama_index_embed_model():
    """LlamaIndex用的向量模型。同样不能用 llama_index.embeddings.dashscope.DashScopeEmbedding
    ——原因和上面get_llama_index_llm一样，那个类打的是阿里云自己的endpoint，不是硅基流动，
    拿硅基流动的key去请求必然401。这里用OpenAIEmbedding+显式api_base，跟get_client()/
    get_langchain_llm()保持同一套"OpenAI兼容协议+自定义base_url"的写法。
    （旧版DashScopeEmbedding有个text_type参数区分document/query两种向量，OpenAI兼容协议的
    embeddings接口没有这个概念，统一用同一个模型即可，调用方不用再传这个参数了。）"""
    from llama_index.embeddings.openai import OpenAIEmbedding
    return OpenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        api_base=DASHSCOPE_BASE_URL,
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )


def _extract_json(raw_text: str) -> dict:
    """兜底处理：万一模型输出带了markdown代码块包裹，去掉再解析"""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return json.loads(text.strip())


def chat_json(
    system_prompt: str, user_prompt: str, max_tokens: int,
    model: str = None, temperature: float | None = None,
) -> dict:
    """调用Qwen，要求输出JSON对象；DashScope兼容模式支持response_format强制JSON输出。
    用stream=True请求，避免生成内容较长时模型思考耗时过长、连接被中间层判定超时挂断；
    但这里的返回值本来就要整体丢给json.loads解析，用户也看不到逐字输出的过程，
    所以流式只用来提高请求稳定性，还是要把所有chunk拼完整再解析。
    （从generate_case.py搬到这里，供案例正文生成流水线和writer/judge/reviser多agent循环共用，
    避免各处重新发明一遍同样的"防截断超时"调用方式。）"""
    kwargs = dict(
        model=model or CHAT_MODEL,
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


def chat_text(
    system_prompt: str, messages: list[dict], max_tokens: int,
    model: str, temperature: float | None = None,
) -> str:
    """跟chat_json同样的流式拼接写法，但不强制JSON输出，返回纯文本。案例正文这类创造性
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
