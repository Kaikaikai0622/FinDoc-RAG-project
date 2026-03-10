"""Reranker 模块 - 对粗检索结果进行精排"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Reranker 抽象基类。

    子类必须实现 rerank()，接口签名与 Retriever.search() 的返回值保持兼容：
    输入 docs 是 search() 返回的 List[Dict]，输出也是同结构（附加 rerank_score 字段）。
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """对候选文档重新打分并返回 top_k 条。

        Args:
            query:  用户原始问题
            docs:   粗检索返回的候选文档列表（含 chunk_text、score 等字段）
            top_k:  精排后最终返回数量

        Returns:
            精排后的文档列表（降序），每条附加 rerank_score 字段
        """


class BGERerankerV2M3(BaseReranker):
    """基于 BAAI/bge-reranker-v2-m3 的精排器。

    使用 sentence-transformers CrossEncoder 接口，零额外依赖。
    模型在首次调用 rerank() 时懒加载，USE_RERANKER=False 时不会占用显存/内存。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model = None  # 懒加载

    def _load_model(self) -> None:
        """首次调用时加载模型（懒加载）。"""
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 只在 CUDA 可用时才有意义；通过 model_kwargs 传入兼容当前 sentence-transformers 版本
        model_kwargs = {}
        if self._use_fp16 and device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        logger.info(
            "加载 Reranker 模型 %s (device=%s, fp16=%s)",
            self._model_name, device, bool(model_kwargs),
        )
        self._model = CrossEncoder(
            self._model_name,
            device=device,
            model_kwargs=model_kwargs or None,
        )
        logger.info("Reranker 模型加载完毕")

    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """对候选文档进行交叉编码打分，返回精排后 top_k 条。

        Args:
            query:  用户原始问题
            docs:   粗检索候选列表（每项须含 chunk_text 字段）
            top_k:  精排最终返回数量

        Returns:
            精排后文档列表（附加 rerank_score 字段，降序排列）
        """
        if not docs:
            return docs

        if self._model is None:
            self._load_model()

        pairs = [(query, doc["chunk_text"]) for doc in docs]
        scores = self._model.predict(pairs)  # numpy array，长度与 docs 相同

        # 将 rerank_score 写入每条 doc（拷贝避免修改原始数据）
        scored_docs = []
        for doc, score in zip(docs, scores):
            entry = dict(doc)
            entry["rerank_score"] = float(score)
            scored_docs.append(entry)

        # 按 rerank_score 降序，截取 top_k
        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored_docs[:top_k]

