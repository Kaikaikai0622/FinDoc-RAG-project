"""Ingestion Pipeline 集成测试

验证完整流程: PDF解析 → 切块 → Embedding → 存储

覆盖场景：
- 完整PDF摄取流程
- Chunk元数据完整性
- 重复摄取幂等性
- 批量文件摄取
- 错误处理
"""
from src.ingestion.pipeline import IngestionPipeline
from src.storage import DocStore, VectorStore


class TestIngestionPipeline:
    """文档摄取管道集成测试"""

    def test_pdf_to_vector_store_e2e(self, isolated_storage, sample_pdf_path):
        """
        测试完整PDF摄取流程

        场景: 摄取样本PDF，验证SQLite和ChromaDB都有数据
        """
        # Arrange
        pipeline = IngestionPipeline()

        # 注入隔离存储路径
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])

        # Act
        result = pipeline.run(sample_pdf_path)

        # Assert
        assert result["chunk_count"] > 0
        assert result["doc_store_count"] > 0
        assert result["vector_store_count"] > 0
        assert result["doc_store_count"] == result["vector_store_count"]
        assert result["page_count"] > 0
        assert result["elapsed_time"] > 0

    def test_chunk_metadata_integrity(self, isolated_storage, sample_pdf_path):
        """
        测试chunk元数据完整性

        场景: 摄取后验证每个chunk的元数据正确
        """
        # Arrange & Act
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])
        result = pipeline.run(sample_pdf_path)

        # Assert: 从SQLite读取并验证
        doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        chunks = doc_store.get_all_chunks()

        assert len(chunks) == result["chunk_count"]

        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "chunk_text" in chunk
            assert chunk["chunk_text"].strip() != ""
            assert "metadata" in chunk
            assert chunk["metadata"]["source_file"] != ""
            assert chunk["metadata"]["page_number"] >= 0
            assert chunk["metadata"]["chunk_index"] >= 0

    def test_reingest_idempotency(self, isolated_storage, sample_pdf_path):
        """
        测试重复摄取幂等性

        场景: 同一文件摄取两次，不应产生重复数据
        """
        # Arrange
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])

        # Act: 摄取两次
        result1 = pipeline.run(sample_pdf_path)
        result2 = pipeline.run(sample_pdf_path)

        # Assert: 第二次不增加新数据
        assert result2["chunk_count"] == result1["chunk_count"]
        assert result2["doc_store_count"] == result1["doc_store_count"]
        assert result2["vector_store_count"] == result1["vector_store_count"]

    def test_batch_processing(self, isolated_storage, sample_pdf_path):
        """
        测试批量文件处理

        场景: 同时处理多个文件
        """
        # Arrange
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])

        # Act: 批量处理（同一文件两次模拟多文件）
        results = pipeline.run_batch([sample_pdf_path, sample_pdf_path])

        # Assert
        assert len(results) == 2
        # 第二次是幂等的，所以chunk_count应该相同
        assert results[0]["chunk_count"] == results[1]["chunk_count"]

    def test_backward_compatibility_pdf_path(self, isolated_storage, sample_pdf_path):
        """
        测试向后兼容的pdf_path参数

        场景: 使用旧的pdf_path关键字参数应该正常工作
        """
        # Arrange
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])

        # Act: 使用旧的pdf_path参数
        result = pipeline.run(pdf_path=sample_pdf_path)

        # Assert
        assert result["chunk_count"] > 0
        assert "pdf_path" in result  # 确保向后兼容字段存在

    def test_ingestion_result_structure(self, isolated_storage, sample_pdf_path):
        """
        测试摄取结果结构完整性

        场景: 验证返回的统计信息包含所有必要字段
        """
        # Arrange & Act
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])
        result = pipeline.run(sample_pdf_path)

        # Assert: 验证所有预期字段存在
        expected_fields = [
            "file_path", "pdf_path", "page_count", "element_count",
            "chunk_count", "embedding_dim", "doc_store_count",
            "vector_store_count", "elapsed_time"
        ]
        for field in expected_fields:
            assert field in result, f"结果缺少字段: {field}"

        # 验证字段类型和值范围
        assert isinstance(result["chunk_count"], int)
        assert isinstance(result["embedding_dim"], int)
        assert result["embedding_dim"] > 0
        assert result["elapsed_time"] > 0
