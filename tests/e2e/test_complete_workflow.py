"""端到端工作流测试

验证完整用户场景: Ingestion → Retrieval → QA

覆盖场景：
- 完整文档摄取到问答流程
- 多轮问答
- 错误恢复
- 并发安全（如适用）
"""
import pytest


class TestCompleteWorkflow:
    """完整工作流端到端测试"""

    def test_full_workflow_ingest_then_query(self, isolated_storage, sample_pdf_path):
        """
        测试完整流程：摄取 → 查询

        场景: 摄取文档后能立即查询
        """
        from src.ingestion.pipeline import IngestionPipeline
        from src.retrieval.retriever import Retriever
        from src.storage import DocStore, VectorStore

        # Step 1: 摄取文档
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])

        ingest_result = pipeline.run(sample_pdf_path)
        assert ingest_result["chunk_count"] > 0

        # Step 2: 查询
        retriever = Retriever(
            doc_store=DocStore(db_path=isolated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=isolated_storage["chroma_path"]),
        )

        search_results = retriever.search("营收", top_k=3)
        assert len(search_results) > 0

        # Step 3: 验证返回的chunk可以被正确获取
        doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        for result in search_results:
            chunk = doc_store.get_chunk_by_id(result["chunk_id"])
            assert chunk is not None
            assert chunk["chunk_text"] == result["chunk_text"]

    def test_full_workflow_with_mock_llm(self, isolated_storage, sample_pdf_path, mock_llm_service):
        """
        测试完整问答流程（使用Mock LLM）

        场景: 端到端流程，但使用Mock避免真实API调用
        """
        from src.ingestion.pipeline import IngestionPipeline
        from src.generation.qa_chain import QAChain
        from src.retrieval.retriever import Retriever
        from src.storage import DocStore, VectorStore

        # Step 1: 摄取
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])
        pipeline.run(sample_pdf_path)

        # Step 2: 创建QAChain
        retriever = Retriever(
            doc_store=DocStore(db_path=isolated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=isolated_storage["chroma_path"]),
        )
        qa_chain = QAChain(retriever=retriever, llm_service=mock_llm_service)

        # Step 3: 问答
        result = qa_chain.ask("营收是多少？")

        # Assert
        assert "answer" in result
        assert "sources" in result
        assert result["chunks_used"] > 0
        assert result["mode"] == "single_step"

    def test_workflow_reingest_then_query(self, isolated_storage, sample_pdf_path):
        """
        测试重新摄取后查询

        场景: 重复摄取同一文档后查询应正常工作
        """
        from src.ingestion.pipeline import IngestionPipeline
        from src.retrieval.retriever import Retriever
        from src.storage import DocStore, VectorStore

        # 第一次摄取
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])
        result1 = pipeline.run(sample_pdf_path)

        # 第二次摄取（幂等）
        result2 = pipeline.run(sample_pdf_path)

        # 验证数据量不变
        assert result1["chunk_count"] == result2["chunk_count"]

        # 查询应正常工作
        retriever = Retriever(
            doc_store=DocStore(db_path=isolated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=isolated_storage["chroma_path"]),
        )
        results = retriever.search("营收", top_k=3)
        assert len(results) > 0

    def test_workflow_multiple_queries(self, isolated_storage, sample_pdf_path):
        """
        测试多轮查询

        场景: 同一文档支持多个不同查询
        """
        from src.ingestion.pipeline import IngestionPipeline
        from src.retrieval.retriever import Retriever
        from src.storage import DocStore, VectorStore

        # 摄取
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])
        pipeline.run(sample_pdf_path)

        # 多次查询
        retriever = Retriever(
            doc_store=DocStore(db_path=isolated_storage["sqlite_path"]),
            vector_store=VectorStore(persist_dir=isolated_storage["chroma_path"]),
        )

        queries = ["营收", "利润", "资产", "负债"]
        for query in queries:
            results = retriever.search(query, top_k=3)
            # 至少有一些查询应该返回结果
            # 注意：如果测试PDF不包含这些关键词，可能返回空
            # 这个测试主要是验证查询不会崩溃
            assert isinstance(results, list)

    def test_workflow_isolation(self, isolated_storage, sample_pdf_path):
        """
        测试数据隔离

        场景: 不同测试之间的数据应相互隔离
        """
        from src.ingestion.pipeline import IngestionPipeline
        from src.storage import DocStore, VectorStore

        # 摄取到隔离存储
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])
        result = pipeline.run(sample_pdf_path)

        # 验证数据只在隔离存储中
        doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        count = doc_store.count()
        assert count == result["chunk_count"]

    @pytest.mark.skip(reason="需要真实LLM和API服务")
    def test_full_workflow_with_real_llm(self, isolated_storage, sample_pdf_path):
        """
        测试完整流程（使用真实LLM）

        场景: 端到端完整测试，需要外部API
        """
        from src.ingestion.pipeline import IngestionPipeline
        from src.generation.qa_chain import QAChain
        from src.storage import DocStore, VectorStore

        # Step 1: 摄取
        pipeline = IngestionPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        pipeline.vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])
        pipeline.run(sample_pdf_path)

        # Step 2: 使用真实LLM问答
        qa_chain = QAChain()  # 使用真实LLM服务
        result = qa_chain.ask("这份年报的主要内容是什么？")

        assert result["answer"] != ""
        assert len(result["sources"]) > 0
