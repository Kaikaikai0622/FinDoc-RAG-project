"""生成模块 - LLM 问答"""
from src.generation.llm_service import (
    BaseLLMService,
    QwenLLMService,
    KimiLLMService,
    get_llm_service,
)
from src.generation.prompts import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, format_context
from src.generation.qa_chain import QAChain, ask

__all__ = [
    "BaseLLMService",
    "QwenLLMService",
    "KimiLLMService",
    "get_llm_service",
    "QA_SYSTEM_PROMPT",
    "QA_USER_TEMPLATE",
    "format_context",
    "QAChain",
    "ask",
]
