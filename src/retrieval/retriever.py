"""检索模块 - 从向量数据库检索相关文档"""
from typing import List, Dict, Any

from src.embedding import BaseEmbeddingService, BGEm3EmbeddingService
from src.storage import VectorStore, DocStore
from config import TOP_K


class Retriever:
    """文档检索器

    接收用户问题，检索相关文档片段。
    流程：问题 → 向量化 → Chroma 向量检索 → SQLite 取原文
    """

    def __init__(
        self,
        embedding_service: BaseEmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        doc_store: DocStore | None = None,
    ) -> None:
        """初始化检索器

        Args:
            embedding_service: Embedding 服务实例
            vector_store: 向量存储实例
            doc_store: 文档存储实例
        """
        self.embedding_service = embedding_service or BGEm3EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.doc_store = doc_store or DocStore()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_file: str | None = None,
    ) -> List[Dict[str, Any]]:
        """检索与问题相关的文档片段

        Args:
            query: 用户问题
            top_k: 返回 top k 结果，默认从 settings.TOP_K 读取
            filter_file: 按来源文件名过滤（支持部分匹配，如"陕国投"匹配"陕国投A：2025年年度报告.pdf"）

        Returns:
            相关文档片段列表，每项包含 chunk_id, chunk_text, source_file, page_number, score
        """
        if top_k is None:
            top_k = TOP_K

        # 构建 ChromaDB where 过滤条件
        where_filter = None
        if filter_file:
            # 使用 $contains 进行部分匹配
            where_filter = {"source_file": {"$contains": filter_file}}

        # 1. 将问题向量化
        query_embedding = self.embedding_service.embed([query])[0]

        # 2. 从 Chroma 向量库检索 Top-K
        vector_results = self.vector_store.query(
            embedding=query_embedding,
            top_k=top_k,
            where=where_filter,
        )

        # 3. 从 SQLite 获取原文和元信息
        results = []
        for item in vector_results:
            chunk_id = item["id"]
            score = item["distance"]

            # 从 SQLite 获取原文
            chunk_info = self.doc_store.get_chunk_by_id(chunk_id)
            if chunk_info:
                results.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_info["chunk_text"],
                    "source_file": chunk_info["metadata"]["source_file"],
                    "page_number": chunk_info["metadata"]["page_number"],
                    "score": 1 - score,  # 转为相似度分数（距离越小越相似）
                })

        return results


def retrieve(query: str, top_k: int | None = None, filter_file: str | None = None) -> List[Dict[str, Any]]:
    """便捷检索函数

    Args:
        query: 用户问题
        top_k: 返回 top k 结果
        filter_file: 按来源文件名过滤

    Returns:
        相关文档片段列表
    """
    retriever = Retriever()
    return retriever.search(query, top_k, filter_file)
