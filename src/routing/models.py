"""Query Router 数据模型 - 定义路由相关的数据结构"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QueryClassification:
    """查询分类结果

    由 QueryClassifier 输出，描述查询的意图类型、生成模式和检索范围。

    Attributes:
        scene: 查询场景类型
        generation_mode: 生成模式（单步/两步）
        filter_file: 过滤条件（文件名或公司名）
        filter_source: 过滤条件来源
        retrieval_scope: 检索范围
        fallback_allowed: 是否允许回退到全局检索
        confidence: 分类置信度（0-1）
        reason_codes: 分类理由代码列表（用于调试和追踪）
    """

    scene: Literal["factual", "comparison", "extraction", "policy_qa", "unknown"]
    generation_mode: Literal["single_step", "two_step"]
    filter_file: str | None = None
    filter_source: Literal["explicit", "auto_company", "none"] = "none"
    retrieval_scope: Literal["single_company", "global"] = "global"
    fallback_allowed: bool = True
    confidence: float = 1.0
    reason_codes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证数据一致性"""
        if self.filter_source == "explicit":
            # 显式过滤默认不允许回退（可通过配置覆盖）
            pass
        if self.filter_file and self.filter_source == "none":
            raise ValueError("filter_file 存在时 filter_source 不能为 none")


@dataclass
class RetrievedChunk:
    """检索到的文档片段

    作为 RetrievedContext 的组成部分，统一描述一个 chunk 的完整信息。

    Attributes:
        chunk_id: chunk 唯一标识
        chunk_text: chunk 原文内容
        source_file: 来源文件名
        page_number: 来源页码
        score: 向量检索相似度分数
        rerank_score: 精排分数（可选）
    """

    chunk_id: str
    chunk_text: str
    source_file: str
    page_number: int
    score: float
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        """转换为字典格式（用于序列化）"""
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "score": self.score,
            "rerank_score": self.rerank_score,
        }

    @classmethod
    def from_retriever_result(cls, result: dict) -> RetrievedChunk:
        """从现有检索器结果格式创建

        Args:
            result: 来自 Retriever.search() 的字典格式

        Returns:
            RetrievedChunk 实例
        """
        return cls(
            chunk_id=result["chunk_id"],
            chunk_text=result["chunk_text"],
            source_file=result["source_file"],
            page_number=result["page_number"],
            score=result["score"],
            rerank_score=result.get("rerank_score"),
        )


@dataclass
class RetrievedContext:
    """完整的检索上下文

    由 QueryRouter 输出，包含检索过程中的所有元信息和检索结果。
    作为 QAChain、API、Evaluator 共享的真实检索上下文。

    Attributes:
        query: 原始查询文本
        classification: 查询分类结果
        retrieval_mode: 实际执行的检索模式
        filter_used: 实际使用的过滤条件
        fallback_triggered: 是否触发了回退
        fallback_reason: 回退原因（如触发）
        chunks: 检索到的 chunks 列表
        chunks_count: chunks 数量（冗余字段，方便序列化）
        retrieval_time: 检索耗时（秒）
    """

    query: str
    classification: QueryClassification
    retrieval_mode: Literal["filtered", "global", "filtered_then_global"]
    filter_used: str | None
    fallback_triggered: bool
    fallback_reason: str | None
    chunks: list[RetrievedChunk]
    retrieval_time: float
    # 派生字段，由 __post_init__ 计算
    chunks_count: int = field(init=False)

    def __post_init__(self) -> None:
        """计算派生字段"""
        object.__setattr__(self, "chunks_count", len(self.chunks))

    def to_sources_list(self) -> list[dict]:
        """转换为 API 兼容的 sources 格式

        Returns:
            轻量级的来源信息列表，用于 API 响应
        """
        sources = []
        for chunk in self.chunks:
            source = {
                "file": chunk.source_file,
                "page": chunk.page_number,
                "score": round(chunk.score, 3),
            }
            if chunk.rerank_score is not None:
                source["rerank_score"] = round(chunk.rerank_score, 3)
            sources.append(source)
        return sources

    def to_dict(self) -> dict:
        """转换为完整字典格式（用于序列化和调试）"""
        return {
            "query": self.query,
            "classification": {
                "scene": self.classification.scene,
                "generation_mode": self.classification.generation_mode,
                "filter_file": self.classification.filter_file,
                "filter_source": self.classification.filter_source,
                "retrieval_scope": self.classification.retrieval_scope,
                "fallback_allowed": self.classification.fallback_allowed,
                "confidence": self.classification.confidence,
                "reason_codes": self.classification.reason_codes,
            },
            "retrieval_mode": self.retrieval_mode,
            "filter_used": self.filter_used,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "chunks_count": self.chunks_count,
            "retrieval_time": self.retrieval_time,
        }
