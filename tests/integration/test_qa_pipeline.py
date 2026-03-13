"""QA Pipeline 集成测试

验证完整流程: Question → Retrieval → LLM Generation

覆盖场景：
- 问答返回完整结果
- Comparison问题触发两步生成
- 自动公司名称提取
- 空检索结果处理
- 单步vs两步生成策略
"""
import pytest
from src.generation.qa_chain import QAChain
from src.retrieval import Retriever
from unittest.mock import MagicMock


class TestQAPipeline:
    """问答管道集成测试"""

    def test_qa_returns_complete_result(self, qa_chain_with_mock_llm):
        """
        测试问答返回完整结果

        场景: 提问应返回包含答案、来源、耗时等的完整结果
        """
        # Act
        result = qa_chain_with_mock_llm.ask("2025年营收是多少？")

        # Assert
        assert "answer" in result
        assert "sources" in result
        assert "retrieval_time" in result
        assert "generation_time" in result
        assert "total_time" in result
        assert "mode" in result  # single_step 或 two_step
        assert "question" in result
        assert "chunks_used" in result

        # 验证返回的问题与提问一致
        assert result["question"] == "2025年营收是多少？"

        # 验证时间和数量字段是合理的
        assert result["retrieval_time"] >= 0
        assert result["generation_time"] >= 0
        assert result["total_time"] >= 0
        assert result["chunks_used"] >= 0

    def test_two_step_generation_for_comparison(self, qa_chain_with_mock_llm):
        """
        测试comparison问题触发两步生成

        场景: 含"相比"的问题应使用two_step模式
        """
        # Act
        result = qa_chain_with_mock_llm.ask("与去年相比营收如何？")

        # Assert
        assert result["mode"] == "two_step"
        assert result["answer"] != ""

    def test_two_step_generation_for_policy(self, qa_chain_with_mock_llm):
        """
        测试政策类问题触发两步生成

        场景: 含"政策"、"规定"等词的问题应使用two_step模式
        """
        # Act
        result = qa_chain_with_mock_llm.ask("分红政策是如何规定的？")

        # Assert
        assert result["mode"] == "two_step"

    def test_two_step_generation_for_extraction(self, qa_chain_with_mock_llm):
        """
        测试提取类问题触发两步生成

        场景: 含"列出"、"说明"等词的问题应使用two_step模式
        """
        # Act
        result = qa_chain_with_mock_llm.ask("列出主要财务指标")

        # Assert
        assert result["mode"] == "two_step"

    def test_single_step_for_simple_factual(self, qa_chain_with_mock_llm):
        """
        测试简单事实问题使用单步生成

        场景: 简单事实问题应使用single_step模式
        """
        # Act
        result = qa_chain_with_mock_llm.ask("2025年营收是多少？")

        # Assert
        assert result["mode"] == "single_step"

    def test_qa_with_manual_filter(self, populated_storage, mock_llm_service):
        """
        测试手动指定filter_file

        场景: 手动提供的filter_file应该被使用
        """
        # Arrange
        retriever = Retriever(
            doc_store=__import__('src.storage', fromlist=['DocStore']).DocStore(
                db_path=populated_storage["sqlite_path"]
            ),
            vector_store=__import__('src.storage', fromlist=['VectorStore']).VectorStore(
                persist_dir=populated_storage["chroma_path"]
            ),
        )
        qa_chain = QAChain(retriever=retriever, llm_service=mock_llm_service)

        # Act: 使用一个肯定不存在的filter
        result = qa_chain.ask("营收是多少？", filter_file="不存在的公司")

        # Assert
        assert "filter_used" in result
        assert result["filter_used"] == "不存在的公司"
        # 由于filter不存在，应该返回0个chunks
        assert result["chunks_used"] == 0

    def test_qa_sources_structure(self, qa_chain_with_mock_llm):
        """
        测试问答来源信息的结构

        场景: sources应包含文件、页码、分数等信息
        """
        # Act
        result = qa_chain_with_mock_llm.ask("营收是多少？")

        # Assert
        assert "sources" in result
        if result["chunks_used"] > 0:
            for source in result["sources"]:
                assert "file" in source
                assert "page" in source
                assert "score" in source
                # score应在合理范围内
                assert 0 <= source["score"] <= 1

    def test_qa_with_no_results(self, populated_storage):
        """
        测试无检索结果的情况

        场景: 当没有匹配的chunks时，应正确处理
        """
        # Arrange
        from src.generation.qa_chain import QAChain

        # 使用一个Mock LLM
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "未找到相关信息"

        retriever = Retriever(
            doc_store=__import__('src.storage', fromlist=['DocStore']).DocStore(
                db_path=populated_storage["sqlite_path"]
            ),
            vector_store=__import__('src.storage', fromlist=['VectorStore']).VectorStore(
                persist_dir=populated_storage["chroma_path"]
            ),
        )
        qa_chain = QAChain(retriever=retriever, llm_service=mock_llm)

        # Act: 使用不存在的公司filter
        result = qa_chain.ask("营收是多少？", filter_file="不存在的公司")

        # Assert
        assert result["chunks_used"] == 0
        assert result["answer"] != ""

    def test_qa_convenience_function(self, populated_storage, mock_llm_service):
        """
        测试便捷的ask函数

        场景: ask()函数应该能正确工作
        """
        # Arrange - 注入mock
        from src.generation import qa_chain as qa_module
        original_qa_chain = qa_module.QAChain

        class MockQAChain:
            def __init__(self, *args, **kwargs):
                pass
            def ask(self, question, filter_file=None):
                return {
                    "question": question,
                    "answer": "测试答案",
                    "sources": [],
                    "chunks_used": 0,
                    "retrieval_time": 0.1,
                    "generation_time": 0.2,
                    "total_time": 0.3,
                    "mode": "single_step",
                    "filter_used": filter_file,
                    "filter_auto": False,
                }

        qa_module.QAChain = MockQAChain

        try:
            # Act
            from src.generation.qa_chain import ask
            result = ask("测试问题")

            # Assert
            assert result["question"] == "测试问题"
            assert "answer" in result
        finally:
            # 恢复
            qa_module.QAChain = original_qa_chain

    def test_two_step_generation_with_trend_keywords(self, qa_chain_with_mock_llm):
        """
        测试趋势类关键词触发两步生成

        场景: 含"增长"、"下降"、"同比"等词的问题应使用two_step模式
        """
        test_cases = [
            "营收同比增长多少？",
            "利润下降了多少？",
            "与去年相比有何变化？",
        ]

        for question in test_cases:
            result = qa_chain_with_mock_llm.ask(question)
            assert result["mode"] == "two_step", f"问题'{question}'应该触发两步生成"


class TestQAPipelineRouterFields:
    """Router 字段验证测试（Phase 7 新增）"""

    def test_qa_returns_router_fields_when_enabled(self, qa_chain_with_mock_llm):
        """Router 启用时返回新增字段"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("陕国投A的营收是多少？")

        # 验证 Router 字段存在
        assert "route_label" in result
        assert "retrieval_mode" in result
        assert "fallback_triggered" in result
        assert "query_classifier" in result
        assert "retrieved_context" in result

    def test_qa_retrieval_mode_filtered(self, qa_chain_with_mock_llm):
        """识别公司名时走 filtered 模式"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("陕国投A的营收是多少？")

        # 应使用过滤检索
        assert result["retrieval_mode"] in ["filtered", "filtered_then_global"]

    def test_qa_retrieval_mode_global(self, qa_chain_with_mock_llm):
        """未识别公司名时走 global 模式"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        # 使用不太可能匹配任何公司名的查询
        result = qa_chain_with_mock_llm.ask("介绍一下公司的业务")

        assert result["retrieval_mode"] == "global"

    def test_qa_classifier_structure(self, qa_chain_with_mock_llm):
        """query_classifier 结构完整"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("陕国投A的营收是多少？")

        qc = result["query_classifier"]
        assert "scene" in qc
        assert "generation_mode" in qc
        assert "filter_source" in qc
        assert "retrieval_scope" in qc
        assert "confidence" in qc

    def test_qa_route_label_factual(self, qa_chain_with_mock_llm):
        """factual 问题的 route_label"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("营收是多少？")
        assert result["route_label"] == "factual"

    def test_qa_route_label_comparison(self, qa_chain_with_mock_llm):
        """comparison 问题的 route_label"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("与去年相比营收如何？")
        assert result["route_label"] == "comparison"

    def test_qa_route_label_policy(self, qa_chain_with_mock_llm):
        """policy_qa 问题的 route_label"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("公司章程中关于分红的规定是什么？")
        assert result["route_label"] == "policy_qa"

    def test_qa_filter_auto_true(self, qa_chain_with_mock_llm):
        """自动识别公司时 filter_auto 为 True"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("陕国投A的营收是多少？")
        # 如果自动识别了公司名，filter_auto 应为 True
        if result["filter_used"]:
            assert result["filter_auto"] is True

    def test_qa_filter_auto_false(self, qa_chain_with_mock_llm):
        """显式指定 filter 时 filter_auto 为 False"""
        from config import ENABLE_QUERY_ROUTER

        if not ENABLE_QUERY_ROUTER:
            pytest.skip("Router 未启用，跳过测试")

        result = qa_chain_with_mock_llm.ask("营收是多少？", filter_file="陕国投A")
        assert result["filter_auto"] is False
        assert result["filter_used"] == "陕国投A"
