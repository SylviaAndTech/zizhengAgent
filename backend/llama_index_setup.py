"""
LlamaIndex 全局配置：把 Settings.llm / Settings.embed_model 都钉死指向 Qwen(DashScope)，
避免 LlamaIndex 内部某些逻辑在没找到配置时，悄悄尝试调用真正的 OpenAI。
"""
from llama_index.core import Settings

from qwen_client import get_llama_index_llm, get_llama_index_embed_model

_configured = False


def configure_llama_index():
    global _configured
    if _configured:
        return
    Settings.llm = get_llama_index_llm()
    Settings.embed_model = get_llama_index_embed_model(text_type="document")
    _configured = True
