"""RerankRetriever - 二阶段检索器（粗检索 → 精排）"""
from __future__ import annotations

import logging
from typing import Any

from config import RETRIEVAL_TOP_K, RERANK_TOP_K
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import BaseReranker, BGERerankerV2M3

logger = logging.getLogger(__name__)


class RerankRetriever:
    """二阶段检索器：向量粗检索 → Reranker 精排。

    接口签名与 Retriever.search() 完全相同，对上层（QAChain、评估脚本）透明替换。

    流程：
        1. 调用底层 Retriever 粗检索 RETRIEVAL_TOP_K(=20) 条候选
        2. 调用 BGERerankerV2M3 精排，返回 RERANK_TOP_K(=5) 条结果
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        """初始化二阶段检索器。

        Args:
            retriever: 粗检索器，默认构造 Retriever()
            reranker:  精排器，默认构造 BGERerankerV2M3()（懒加载模型）
        """
        self.retriever = retriever or Retriever()
        self.reranker = reranker or BGERerankerV2M3()

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """检索并精排相关文档片段。

        Args:
            query:  用户问题
            top_k:  精排最终返回数量，None 时使用 config.RERANK_TOP_K

        Returns:
            精排后的文档列表，每项含 chunk_id、chunk_text、source_file、
            page_number、score（向量相似度）、rerank_score（精排分数）
        """
        rerank_top_k = top_k if top_k is not None else RERANK_TOP_K

        # 第一阶段：粗检索，固定取 RETRIEVAL_TOP_K 条候选
        candidates = self.retriever.search(query, top_k=RETRIEVAL_TOP_K)
        logger.debug("粗检索返回 %d 条候选", len(candidates))

        if not candidates:
            return candidates

        # 第二阶段：精排
        results = self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        logger.debug("精排后返回 %d 条结果", len(results))

        return results

