"""Chroma 向量存储模块

存储 chunk 的 embedding 向量。
collection name 包含 embedding 模型标识，方便后期多模型共存。
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


class VectorStore:
    """Chroma 向量存储类

    用于存储和检索 chunk 的 embedding 向量。
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        """初始化 VectorStore

        Args:
            persist_dir: Chroma 持久化目录
            embedding_model: embedding 模型名称，用于生成 collection name
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

        # 生成包含模型标识的 collection name
        # 将模型名中的 / 替换为 _，避免 Chroma collection name 冲突
        safe_model_name = embedding_model.replace("/", "_")
        self._collection_name = f"findoc_{safe_model_name}"

    @property
    def client(self) -> chromadb.PersistentClient:
        """获取 Chroma 客户端（懒加载）"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """获取或创建 collection"""
        if self._collection is None:
            # 如果 collection 已存在则获取，否则创建
            try:
                self._collection = self.client.get_collection(
                    name=self._collection_name
                )
            except Exception:
                self._collection = self.client.create_collection(
                    name=self._collection_name,
                    metadata={"model": self.embedding_model}
                )
        return self._collection

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """批量添加 embeddings

        Args:
            ids: chunk id 列表
            embeddings: embedding 向量列表
            metadatas: 元信息列表（可选）
        """
        if not ids or not embeddings:
            return

        # 如果没有提供 metadatas，创建一个空的
        if metadatas is None:
            metadatas = [{}] * len(ids)

        # 直接添加（Chroma会自动覆盖已存在的id）
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """查询相似向量（预留接口，本阶段不调用）

        Args:
            embedding: 查询向量
            top_k: 返回 top k 结果
            where: 过滤条件

        Returns:
            查询结果列表，每项包含 id, distance, metadata
        """
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )

        # 格式化返回结果
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })

        return output

    def count(self) -> int:
        """统计向量总数

        Returns:
            向量数量
        """
        return self.collection.count()

    def delete_all(self) -> None:
        """删除所有向量（用于测试或重建索引）"""
        self.collection.delete(where={})
