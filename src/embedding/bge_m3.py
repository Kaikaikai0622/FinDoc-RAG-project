"""BGE-M3 Embedding 服务实现"""
import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from src.embedding.base import BaseEmbeddingService
from config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class BGEm3EmbeddingService(BaseEmbeddingService):
    """基于 BAAI/bge-m3 的 Embedding 服务实现

    BGE-M3 是一个多语言 embedding 模型，支持密集检索、稀疏检索和混合检索。
    向量维度为 1024。
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """初始化 BGE-M3 模型

        Args:
            model_name: HuggingFace 模型名称，默认为 BAAI/bge-m3
        """
        self.model_name = model_name
        self._dimension = EMBEDDING_DIM
        # 延迟加载模型，避免启动时耗时过长
        self._model = None
        # 记录设备信息
        self._device = None

    def _get_device_info(self) -> str:
        """获取设备信息并打印日志

        Returns:
            设备信息字符串
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self._device = "cuda"
            logger.info(f"[Embedding] 使用 GPU: {device_name}")
            return f"cuda:0 ({device_name})"
        else:
            self._device = "cpu"
            logger.warning("[Embedding] 未检测到 GPU，将使用 CPU")
            return "cpu"

    @property
    def model(self) -> SentenceTransformer:
        """懒加载模型

        Returns:
            SentenceTransformer 模型实例
        """
        if self._model is None:
            # 打印设备信息
            device_info = self._get_device_info()
            logger.info(f"[Embedding] 加载模型: {self.model_name}, 设备: {device_info}")

            self._model = SentenceTransformer(self.model_name)
            # 明确将模型移到GPU
            if self._device == "cuda":
                self._model = self._model.to("cuda")
                logger.info(f"[Embedding] 模型已加载到 GPU")

        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表（批量编码）

        Args:
            texts: 待向量化的文本列表

        Returns:
            向量列表
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """获取 embedding 向量维度

        Returns:
            向量维度，BGE-M3 为 1024
        """
        return self._dimension
