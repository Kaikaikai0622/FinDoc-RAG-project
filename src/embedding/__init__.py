"""Embedding 模块"""
from src.embedding.base import BaseEmbeddingService
from src.embedding.bge_m3 import BGEm3EmbeddingService

__all__ = [
    "BaseEmbeddingService",
    "BGEm3EmbeddingService",
]
