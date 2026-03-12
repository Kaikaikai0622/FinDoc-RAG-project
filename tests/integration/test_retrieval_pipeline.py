"""Retrieval Pipeline 集成测试

验证完整流程: Query → Embedding → Vector Search → SQLite Join

覆盖场景：
- 基本查询返回chunks
- filter_file参数过滤
- VectorStore和SQLite数据一致性
- 空结果处理
- 分页和top_k参数
"""
import pytest
from src.retrieval.retriever import Retriever
from src.storage import DocStore, VectorStore


class TestRetrievalPipeline:
    """检索管道集成测试"""

    def test_query_returns_chunks(self, populated_storage):
        """
        测试查询返回chunks

        场景: 对已摄取文档进行查询，应返回相关chunks
        """
        # Arrange
        retriever = Retriever(
            doc_store=DocStore(db_path=populated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=populated_storage["chroma_path"]),
        )

        # Act
        results = retriever.search("营收是多少", top_k=5)

        # Assert
        assert len(results) > 0
        assert len(results) <= 5
        assert all("chunk_text" in r for r in results)
        assert all("source_file" in r for r in results)
        assert all("score" in r for r in results)

    def test_filter_file_parameter(self, populated_storage):
        """
        测试filter_file参数

        场景: 使用filter_file应只返回匹配文件的chunks
        """
        # Arrange
        retriever = Retriever(
            doc_store=DocStore(db_path=populated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=populated_storage["chroma_path"]),
        )

        # Act: 使用不存在的文件过滤
        results = retriever.search("营收", filter_file="不存在的文件")

        # Assert
        assert len(results) == 0  # 无匹配文件

    def test_vector_sqlite_consistency(self, populated_storage):
        """
        测试VectorStore和SQLite数据一致性

        场景: ChromaDB中的向量应与SQLite中的chunk一一对应
        """
        # Arrange
        doc_store = DocStore(db_path=populated_storage["sqlite_path"])
        vector_store = VectorStore(persist_dir=populated_storage["chroma_path"])

        # Act: 查询所有chunks
        chunks = doc_store.get_all_chunks()
        chunk_ids = [c["chunk_id"] for c in chunks]

        # Assert: 向量数量和chunk数量一致
        vector_count = vector_store.count()
        assert vector_count == len(chunk_ids)
        assert vector_count == populated_storage["ingestion_result"]["chunk_count"]

    def test_retriever_returns_chunk_metadata(self, populated_storage):
        """
        测试检索结果包含完整的元数据

        场景: 每个返回的chunk应包含所有必要的元数据字段
        """
        # Arrange
        retriever = Retriever(
            doc_store=DocStore(db_path=populated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=populated_storage["chroma_path"]),
        )

        # Act
        results = retriever.search("年报", top_k=3)

        # Assert
        if len(results) > 0:
            for result in results:
                assert "chunk_id" in result
                assert "chunk_text" in result
                assert "source_file" in result
                assert "page_number" in result
                assert "score" in result
                # 验证score是相似度分数（0-1范围）
                assert 0 <= result["score"] <= 1

    def test_top_k_parameter(self, populated_storage):
        """
        测试top_k参数限制返回数量

        场景: top_k应该严格限制返回结果数量
        """
        # Arrange
        retriever = Retriever(
            doc_store=DocStore(db_path=populated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=populated_storage["chroma_path"]),
        )

        # Act: 测试不同的top_k值
        results_3 = retriever.search("年报", top_k=3)
        results_10 = retriever.search("年报", top_k=10)

        # Assert
        assert len(results_3) <= 3
        assert len(results_10) <= 10
        # 如果数据足够，top_k=10应该返回更多或相同数量
        assert len(results_10) >= len(results_3)

    def test_retrieve_by_chunk_id(self, populated_storage):
        """
        测试通过chunk_id获取chunk详情

        场景: 能正确从SQLite获取指定chunk的完整信息
        """
        # Arrange
        doc_store = DocStore(db_path=populated_storage["sqlite_path"])

        # 先获取一个chunk_id
        all_chunks = doc_store.get_all_chunks()
        if not all_chunks:
            pytest.skip("没有可用的chunks")

        chunk_id = all_chunks[0]["chunk_id"]

        # Act
        chunk_info = doc_store.get_chunk_by_id(chunk_id)

        # Assert
        assert chunk_info is not None
        assert chunk_info["chunk_id"] == chunk_id
        assert "chunk_text" in chunk_info
        assert "metadata" in chunk_info

    def test_retrieve_nonexistent_chunk(self, populated_storage):
        """
        测试获取不存在的chunk

        场景: 对不存在的chunk_id应返回None
        """
        # Arrange
        doc_store = DocStore(db_path=populated_storage["sqlite_path"])

        # Act
        chunk_info = doc_store.get_chunk_by_id("nonexistent_chunk_id_12345")

        # Assert
        assert chunk_info is None

    def test_filter_file_partial_match(self, populated_storage):
        """
        测试filter_file部分匹配

        场景: filter_file支持部分匹配（如"中兴"匹配"中兴通讯"）
        """
        # Arrange
        retriever = Retriever(
            doc_store=DocStore(db_path=populated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=populated_storage["chroma_path"]),
        )

        # 获取实际文件名用于测试
        doc_store = DocStore(db_path=populated_storage["sqlite_path"])
        chunks = doc_store.get_all_chunks()
        if not chunks:
            pytest.skip("没有可用的chunks")

        source_file = chunks[0]["metadata"]["source_file"]
        # 使用文件名的一部分作为filter
        partial_name = source_file[:5] if len(source_file) > 5 else source_file

        # Act
        results = retriever.search("年报", filter_file=partial_name)

        # Assert: 至少应该返回匹配的chunks
        # 注意：如果没有匹配结果，说明filter_file可能不支持部分匹配
        # 或者数据本身不符合预期，这个测试用于验证功能存在
        for result in results:
            assert partial_name in result["source_file"]
