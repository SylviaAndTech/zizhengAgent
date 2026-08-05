"""
统一的 Qwen（阿里云 DashScope，OpenAI 兼容模式）客户端与模型配置。
所有需要调用大模型/向量模型的地方都从这里取 client 和模型名，避免各处散着写。
"""
import os

import httpx
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
    """LlamaIndex用的Qwen聊天模型（原生DashScope SDK）。
    我们目前的检索链路只用LlamaIndex做向量存取，不靠它做生成，
    这里主要是为了把Settings.llm钉死在Qwen上，避免LlamaIndex内部逻辑悄悄尝试调用真正的OpenAI。"""
    from llama_index.llms.dashscope import DashScope
    return DashScope(model_name=CHAT_MODEL, api_key=os.environ.get("DASHSCOPE_API_KEY"))


def get_llama_index_embed_model(text_type: str = "document"):
    """LlamaIndex用的Qwen向量模型。text_type='document'用于建索引；
    检索时LlamaIndex内部会自动用text_type='query'，不需要单独再配一个query用的实例"""
    from llama_index.embeddings.dashscope import DashScopeEmbedding
    return DashScopeEmbedding(
        model_name=EMBEDDING_MODEL,
        text_type=text_type,
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )
