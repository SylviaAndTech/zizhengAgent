"""
统一的 Qwen（阿里云 DashScope，OpenAI 兼容模式）客户端与模型配置。
所有需要调用大模型/向量模型的地方都从这里取 client 和模型名，避免各处散着写。
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CHAT_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
EMBEDDING_MODEL = os.environ.get("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
MAX_TOOL_ROUNDS = int(os.environ.get("QWEN_MAX_TOOL_ROUNDS", "6"))

_client = None


def has_api_key() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def require_api_key():
    """没配key时给出清晰报错，而不是让底层SDK抛一个不知所云的错误"""
    if not has_api_key():
        raise ValueError(
            "未检测到 DASHSCOPE_API_KEY 环境变量。请先在 backend/.env 里配置阿里云 "
            "DashScope 的 API Key（https://dashscope.console.aliyun.com/apiKey）后重启后端"
        )


def get_client() -> OpenAI:
    """延迟初始化：避免没配置key或网络异常时，服务器一启动就崩溃"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=DASHSCOPE_BASE_URL,
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
