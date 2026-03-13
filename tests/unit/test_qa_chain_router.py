"""QAChain Router 集成测试

验证 QAChain 与 QueryRouter 的集成：
1. __init__ 支持注入 router
2. ask() 返回新增字段
3. 保留向后兼容的旧字段
4. 旧模式回退（ENABLE_QUERY_ROUTER=False）
"""
from unittest.mock import MagicMock, patch

from src.generation.qa_chain import QAChain


class TestQAChainInit:
    """初始化测试"""

    def test_init_with_router(self):
        """支持注入 router"""
        mock_router = MagicMock()
        mock_router.route.return_value = MagicMock(
            chunks=[],
            retrieval_time=0.5,
            classification=MagicMock(
                scene="factual",
                generation_mode="single_step",
                filter_file=None,
                filter_source="none",
                confidence=0.8,
                reason_codes=["test"],
            ),
            retrieval_mode="global",
            filter_used=None,
            fallback_triggered=False,
            fallback_reason=None,
            to_dict=lambda: {"test": "data"},
        )

        chain = QAChain(router=mock_router)

        assert chain.router is mock_router

    def test_init_without_router_uses_config(self):
        """不传入 router 时根据配置决定"""
        from config import ENABLE_QUERY_ROUTER

        chain = QAChain()

        if ENABLE_QUERY_ROUTER:
            assert chain.router is not None
        else:
            assert chain.router is None

    def test_retriever_property_backward_compatible(self):
        """retriever 属性向后兼容"""
        chain = QAChain()

        # 旧代码可能访问 chain.retriever
        retriever = chain.retriever

        assert retriever is not None
        assert retriever is chain._retriever


class TestQAChainReturnStructure:
    """返回结构测试"""

    def _create_mock_router(self, chunks=None, scene="factual", generation_mode="single_step"):
        """Helper: 创建 mock router"""
        mock_router = MagicMock()

        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = {
            "chunk_id": "1",
            "chunk_text": "test",
            "source_file": "test.pdf",
            "page_number": 1,
            "score": 0.9,
        }

        mock_router.route.return_value = MagicMock(
            chunks=[mock_chunk] if chunks is None else chunks,
            retrieval_time=0.5,
            classification=MagicMock(
                scene=scene,
                generation_mode=generation_mode,
                filter_file="test.pdf",
                filter_source="explicit",
                retrieval_scope="single_company",
                confidence=0.8,
                reason_codes=["test:reason"],
            ),
            retrieval_mode="filtered",
            filter_used="test.pdf",
            fallback_triggered=False,
            fallback_reason=None,
            to_dict=lambda: {"query": "test", "chunks_count": 1},
        )

        return mock_router

    def test_return_has_all_old_fields(self):
        """返回包含所有旧字段"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "测试答案"

        mock_router = self._create_mock_router()

        chain = QAChain(router=mock_router, llm_service=mock_llm)

        with patch.object(chain, '_single_step_generate', return_value=("测试答案", "single_step")):
            result = chain.ask("测试问题", filter_file="test.pdf")

        # 旧字段必须存在
        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert "chunks_used" in result
        assert "retrieval_time" in result
        assert "generation_time" in result
        assert "total_time" in result
        assert "mode" in result
        assert "filter_used" in result
        assert "filter_auto" in result

        # 旧字段类型正确
        assert isinstance(result["sources"], list)
        assert isinstance(result["chunks_used"], int)
        assert result["filter_used"] == "test.pdf"

    def test_return_has_new_router_fields(self):
        """返回包含 Router 新增字段"""
        mock_llm = MagicMock()

        mock_router = self._create_mock_router()

        chain = QAChain(router=mock_router, llm_service=mock_llm)

        with patch.object(chain, '_single_step_generate', return_value=("测试答案", "single_step")):
            result = chain.ask("测试问题")

        # 新增字段必须存在
        assert "route_label" in result
        assert "retrieval_mode" in result
        assert "fallback_triggered" in result
        assert "query_classifier" in result
        assert "retrieved_context" in result

        # 新增字段内容正确
        assert result["route_label"] == "factual"
        assert result["retrieval_mode"] == "filtered"
        assert result["fallback_triggered"] is False

        # query_classifier 结构
        qc = result["query_classifier"]
        assert "scene" in qc
        assert "generation_mode" in qc
        assert "filter_source" in qc
        assert "confidence" in qc
        assert "reason_codes" in qc

    def test_mode_backward_compatible(self):
        """mode 字段向后兼容（single_step/two_step）"""
        mock_llm = MagicMock()

        # 测试 single_step
        mock_router = self._create_mock_router(generation_mode="single_step")
        chain = QAChain(router=mock_router, llm_service=mock_llm)

        with patch.object(chain, '_single_step_generate', return_value=("答案", "single_step")):
            result = chain.ask("问题")

        assert result["mode"] in ("single_step", "two_step")


class TestQAChainGenerationMode:
    """生成模式路由测试"""

    def test_single_step_generation(self):
        """single_step 模式调用正确"""
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.route.return_value = MagicMock(
            chunks=[],
            retrieval_time=0.5,
            classification=MagicMock(
                scene="factual",
                generation_mode="single_step",
                filter_file=None,
                filter_source="none",
                confidence=0.8,
                reason_codes=[],
            ),
            retrieval_mode="global",
            filter_used=None,
            fallback_triggered=False,
            fallback_reason=None,
            to_dict=lambda: {},
        )

        chain = QAChain(router=mock_router, llm_service=mock_llm)

        with patch.object(chain, '_single_step_generate') as mock_single:
            mock_single.return_value = ("答案", "single_step")
            chain.ask("问题")

            mock_single.assert_called_once()

    def test_two_step_generation(self):
        """two_step 模式调用正确"""
        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.route.return_value = MagicMock(
            chunks=[],
            retrieval_time=0.5,
            classification=MagicMock(
                scene="policy_qa",
                generation_mode="two_step",
                filter_file="test.pdf",
                filter_source="explicit",
                confidence=0.9,
                reason_codes=[],
            ),
            retrieval_mode="filtered",
            filter_used="test.pdf",
            fallback_triggered=False,
            fallback_reason=None,
            to_dict=lambda: {},
        )

        chain = QAChain(router=mock_router, llm_service=mock_llm)

        with patch.object(chain, '_two_step_generate') as mock_two:
            mock_two.return_value = ("答案", "two_step")
            chain.ask("公司章程中关于分红的规定是什么？", filter_file="test.pdf")

            mock_two.assert_called_once()


class TestQAChainBackwardCompatibility:
    """向后兼容性测试"""

    def test_ask_function_signature_unchanged(self):
        """ask 方法签名不变"""
        import inspect

        sig = inspect.signature(QAChain.ask)
        params = list(sig.parameters.keys())

        assert "question" in params
        assert "filter_file" in params

    def test_result_keys_additive_only(self):
        """结果键只增不减"""
        # 期望的旧字段集合
        old_fields = {
            "question", "answer", "sources", "chunks_used",
            "retrieval_time", "generation_time", "total_time",
            "mode", "filter_used", "filter_auto",
        }

        mock_llm = MagicMock()
        mock_router = MagicMock()
        mock_router.route.return_value = MagicMock(
            chunks=[MagicMock(to_dict=lambda: {})],
            retrieval_time=0.5,
            classification=MagicMock(
                scene="factual",
                generation_mode="single_step",
                filter_file=None,
                filter_source="none",
                confidence=0.8,
                reason_codes=[],
            ),
            retrieval_mode="global",
            filter_used=None,
            fallback_triggered=False,
            fallback_reason=None,
            to_dict=lambda: {},
        )

        chain = QAChain(router=mock_router, llm_service=mock_llm)

        with patch.object(chain, '_single_step_generate', return_value=("答案", "single_step")):
            result = chain.ask("问题")

        # 所有旧字段都必须存在
        for field in old_fields:
            assert field in result, f"旧字段 {field} 缺失"
