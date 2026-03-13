"""Query Router - 查询路由器

职责：
1. 接收查询和可选过滤条件
2. 调用 QueryClassifier 进行分类
3. 根据分类结果执行相应的检索策略
4. 支持回退机制（自动过滤失败时回退到全局检索）
5. 返回统一的 RetrievedContext

支持路径：
- filtered: 单一过滤检索
- global: 全局检索
- filtered_then_global: 过滤检索失败后回退到全局
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from config import QUERY_ROUTER_EMPTY_RESULT_THRESHOLD
from src.retrieval import RerankRetriever
from src.routing.models import QueryClassification, RetrievedChunk, RetrievedContext
from src.routing.query_classifier import QueryClassifier

if TYPE_CHECKING:
    from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class QueryRouter:
    """查询路由器

    编排查询分类和检索执行，统一输出 RetrievedContext。

    不负责：
    - 生成答案
    - prompt 组织
    - API 格式化
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        classifier: QueryClassifier | None = None,
        empty_result_threshold: int | None = None,
    ) -> None:
        """初始化 QueryRouter

        Args:
            retriever: 检索器实例，默认使用 RerankRetriever
            classifier: 分类器实例，默认新建 QueryClassifier
            empty_result_threshold: 回退阈值（结果数 <= 此值时回退），None 使用配置
        """
        self.retriever = retriever or RerankRetriever()
        self.classifier = classifier or QueryClassifier()
        self.empty_result_threshold = (
            empty_result_threshold
            if empty_result_threshold is not None
            else QUERY_ROUTER_EMPTY_RESULT_THRESHOLD
        )

    def route(
        self,
        query: str,
        filter_file: str | None = None,
    ) -> RetrievedContext:
        """执行查询路由和检索

        Args:
            query: 用户查询文本
            filter_file: 显式指定的过滤条件（可选）

        Returns:
            RetrievedContext 包含完整检索上下文
        """
        start_time = time.time()

        # Step 1: 分类
        classification = self.classifier.classify(query, filter_file)
        logger.debug(
            "Query classified: scene=%s, generation=%s, scope=%s, filter=%s",
            classification.scene,
            classification.generation_mode,
            classification.retrieval_scope,
            classification.filter_file,
        )

        # Step 2: 根据分类执行检索
        chunks, retrieval_mode, fallback_triggered, fallback_reason = self._execute_retrieval(
            query=query,
            classification=classification,
        )

        retrieval_time = time.time() - start_time

        # Step 3: 组装结果
        retrieved_chunks = [
            RetrievedChunk.from_retriever_result(chunk) for chunk in chunks
        ]

        return RetrievedContext(
            query=query,
            classification=classification,
            retrieval_mode=retrieval_mode,
            filter_used=classification.filter_file,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            chunks=retrieved_chunks,
            retrieval_time=retrieval_time,
        )

    def _execute_retrieval(
        self,
        query: str,
        classification: QueryClassification,
    ) -> tuple[list[dict], str, bool, str | None]:
        """执行检索策略

        Args:
            query: 查询文本
            classification: 查询分类结果

        Returns:
            (chunks列表, 检索模式, 是否回退, 回退原因)
        """
        # 情况 1: 全局检索
        if classification.retrieval_scope == "global":
            logger.debug("Executing global retrieval")
            chunks = self.retriever.search(query, filter_file=None)
            return chunks, "global", False, None

        # 情况 2: 单公司过滤检索
        filter_file = classification.filter_file
        assert filter_file is not None  # scope=single_company 时 filter 必存在

        logger.debug("Executing filtered retrieval: filter=%s", filter_file)
        chunks = self.retriever.search(query, filter_file=filter_file)

        # 检查是否需要回退
        filtered_count = len(chunks)
        if filtered_count <= self.empty_result_threshold:
            if classification.fallback_allowed:
                logger.info(
                    "Fallback triggered: filtered retrieval returned %d chunks "
                    "(threshold=%d), falling back to global",
                    filtered_count,
                    self.empty_result_threshold,
                )
                chunks = self.retriever.search(query, filter_file=None)
                return (
                    chunks,
                    "filtered_then_global",
                    True,
                    f"filtered_returned_{filtered_count}_chunks",
                )
            else:
                logger.debug(
                    "Fallback not allowed: filtered retrieval returned %d chunks, "
                    "staying with filtered results",
                    len(chunks),
                )
                return chunks, "filtered", False, None

        # 正常返回过滤结果
        return chunks, "filtered", False, None
