"""Evaluator Router 集成测试

验证 Evaluator 使用真实检索上下文：
1. 使用 result["retrieved_context"] 而不是重新检索
2. details 中记录路由元信息
3. config 中记录 Router 开关状态
"""
import pytest
from unittest.mock import MagicMock, patch

from src.evaluation.evaluator import Evaluator, EvalResult
from src.evaluation.dataset import EvalDataset


class TestEvaluatorUsesRealContext:
    """验证 Evaluator 使用真实上下文"""

    def test_evaluator_uses_retrieved_context(self):
        """Evaluator 应使用 QAChain 返回的 retrieved_context"""
        # Mock QAChain 返回包含 retrieved_context 的结果
        mock_qa_chain = MagicMock()
        mock_qa_chain.ask.return_value = {
            "question": "测试问题",
            "answer": "测试答案",
            "sources": [
                {"file": "test.pdf", "page": 1, "score": 0.95}
            ],
            "chunks_used": 1,
            "retrieval_time": 0.5,
            "generation_time": 1.0,
            "total_time": 1.5,
            "mode": "single_step",
            "filter_used": "test.pdf",
            "filter_auto": False,
            # Router 返回的上下文
            "route_label": "factual",
            "retrieval_mode": "filtered",
            "fallback_triggered": False,
            "query_classifier": {
                "scene": "factual",
                "generation_mode": "single_step",
                "filter_source": "explicit",
                "retrieval_scope": "single_company",
                "confidence": 0.9,
                "reason_codes": ["test"],
            },
            "retrieved_context": {
                "query": "测试问题",
                "chunks": [
                    {
                        "chunk_id": "test_1_0",
                        "chunk_text": "这是真实的 chunk 文本",
                        "source_file": "test.pdf",
                        "page_number": 1,
                        "score": 0.95,
                    }
                ],
                "chunks_count": 1,
                "retrieval_mode": "filtered",
                "fallback_triggered": False,
            },
        }

        evaluator = Evaluator(qa_chain=mock_qa_chain)

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.get_all.return_value = [
            {
                "id": "s1",
                "question": "测试问题",
                "ground_truth": "测试答案",
                "scene": "factual",
                "difficulty": "easy",
                "source_file": "test.pdf",
                "source_page": 1,
            }
        ]

        # Mock LLM judge 返回正确
        with patch.object(evaluator, '_check_answer_correct', return_value=True):
            result = evaluator.run(mock_dataset, run_name="test", include_ragas=False)

        # 验证使用了真实上下文（而不是重新检索）
        detail = result.details[0]
        # chunk 文本应该来自 retrieved_context，而不是重新检索
        assert detail.get("route_label") == "factual"
        assert detail.get("retrieval_mode") == "filtered"
        assert detail.get("fallback_triggered") is False

    def test_evaluator_fallback_to_old_mode(self):
        """当 retrieved_context 不存在时回退到旧模式"""
        mock_qa_chain = MagicMock()
        mock_qa_chain.ask.return_value = {
            "question": "测试问题",
            "answer": "测试答案",
            "sources": [
                {"file": "test.pdf", "page": 1, "score": 0.95}
            ],
            "chunks_used": 1,
            "retrieval_time": 0.5,
            "generation_time": 1.0,
            "total_time": 1.5,
            "mode": "single_step",
            "filter_used": "test.pdf",
            "filter_auto": False,
            # 无 retrieved_context（旧模式）
        }

        evaluator = Evaluator(qa_chain=mock_qa_chain)

        mock_dataset = MagicMock()
        mock_dataset.get_all.return_value = [
            {
                "id": "s1",
                "question": "测试问题",
                "ground_truth": "测试答案",
                "scene": "factual",
                "source_file": "test.pdf",
                "source_page": 1,
            }
        ]

        # 应该能正常运行（回退到旧模式）
        with patch.object(evaluator, '_check_answer_correct', return_value=True):
            result = evaluator.run(mock_dataset, run_name="test", include_ragas=False)

        # 回退模式下路由元信息应为 None
        detail = result.details[0]
        assert detail.get("route_label") is None
        assert detail.get("retrieval_mode") is None


class TestEvaluatorConfig:
    """验证 Evaluator Config 记录 Router 状态"""

    def test_config_contains_router_flag(self):
        """配置应包含 Router 开关状态"""
        evaluator = Evaluator()
        config = evaluator._get_config()

        assert "enable_query_router" in config

    def test_config_contains_router_settings_when_enabled(self):
        """Router 启用时配置应包含相关设置"""
        from config import ENABLE_QUERY_ROUTER

        evaluator = Evaluator()
        config = evaluator._get_config()

        if ENABLE_QUERY_ROUTER:
            assert "router_allow_auto_fallback" in config
            assert "router_allow_explicit_fallback" in config
            assert "router_empty_threshold" in config


class TestEvaluatorDetailsStructure:
    """验证 Details 结构包含路由元信息"""

    def test_details_structure(self):
        """验证 details 条目结构"""
        mock_qa_chain = MagicMock()
        mock_qa_chain.ask.return_value = {
            "question": "测试",
            "answer": "答案",
            "sources": [],
            "retrieved_context": {"chunks": []},
            "route_label": "factual",
            "retrieval_mode": "global",
            "fallback_triggered": False,
            "query_classifier": {
                "scene": "factual",
                "generation_mode": "single_step",
                "filter_source": "none",
                "retrieval_scope": "global",
                "confidence": 0.8,
            },
        }

        evaluator = Evaluator(qa_chain=mock_qa_chain)

        mock_dataset = MagicMock()
        mock_dataset.get_all.return_value = [
            {
                "id": "s1",
                "question": "测试",
                "ground_truth": "答案",
                "scene": "factual",
            }
        ]

        with patch.object(evaluator, '_check_answer_correct', return_value=True):
            result = evaluator.run(mock_dataset, run_name="test", include_ragas=False)

        detail = result.details[0]

        # 基础字段
        assert "id" in detail
        assert "question" in detail
        assert "model_answer" in detail
        assert "answer_correct" in detail

        # 路由元信息字段
        assert "route_label" in detail
        assert "retrieval_mode" in detail
        assert "fallback_triggered" in detail
        assert "query_classifier" in detail

        # query_classifier 结构
        qc = detail["query_classifier"]
        assert "scene" in qc
        assert "generation_mode" in qc
        assert "filter_source" in qc
        assert "retrieval_scope" in qc
        assert "confidence" in qc


class TestEvaluatorContextConsistency:
    """验证评估上下文与真实回答上下文一致"""

    def test_no_duplicate_retrieval(self):
        """评估时不应再次调用检索器"""
        mock_qa_chain = MagicMock()
        mock_qa_chain.ask.return_value = {
            "question": "测试",
            "answer": "答案",
            "sources": [{"file": "test.pdf", "page": 1, "score": 0.9}],
            "retrieved_context": {
                "chunks": [
                    {
                        "chunk_id": "1",
                        "chunk_text": "真实文本",
                        "source_file": "test.pdf",
                        "page_number": 1,
                        "score": 0.9,
                    }
                ]
            },
        }

        # Mock retriever（验证不会被调用）
        mock_retriever = MagicMock()
        mock_qa_chain.retriever = mock_retriever

        evaluator = Evaluator(qa_chain=mock_qa_chain)

        mock_dataset = MagicMock()
        mock_dataset.get_all.return_value = [
            {
                "id": "s1",
                "question": "测试",
                "ground_truth": "答案",
                "scene": "factual",
            }
        ]

        with patch.object(evaluator, '_check_answer_correct', return_value=True):
            evaluator.run(mock_dataset, run_name="test", include_ragas=False)

        # 验证 retriever.search 没有被调用（因为我们已经有 retrieved_context）
        mock_retriever.search.assert_not_called()
