"""QueryRouter 单元测试

验证路由器的核心行为：
1. filtered 路径
2. global 路径
3. filtered_then_global 路径（回退）
4. 显式 filter 不回退
5. rerank_score 不丢失
6. RetrievedContext 结构完整
"""
import pytest
from unittest.mock import MagicMock, patch

from src.routing.query_router import QueryRouter
from src.routing.models import RetrievedContext, QueryClassification

# 正确的 mock 路径：extract_company_filter 在 query_classifier 中调用
PATCH_PATH = 'src.routing.query_classifier.extract_company_filter'


class TestQueryRouterGlobalPath:
    """全局检索路径测试"""

    def test_global_path_no_filter(self):
        """无过滤条件时走 global 路径"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "test", "source_file": "a.pdf", "page_number": 1, "score": 0.9}
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = None  # 无自动识别

            result = router.route("介绍一下这家公司")

        assert result.retrieval_mode == "global"
        assert result.fallback_triggered is False
        assert result.filter_used is None
        mock_retriever.search.assert_called_once_with("介绍一下这家公司", filter_file=None)

    def test_global_path_with_chunks(self):
        """global 路径返回 chunks"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "chunk1", "source_file": "a.pdf", "page_number": 1, "score": 0.9},
            {"chunk_id": "2", "chunk_text": "chunk2", "source_file": "b.pdf", "page_number": 2, "score": 0.8},
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = None

            result = router.route("query")

        assert len(result.chunks) == 2
        assert result.chunks_count == 2
        assert result.chunks[0].chunk_id == "1"
        assert result.chunks[1].source_file == "b.pdf"


class TestQueryRouterFilteredPath:
    """过滤检索路径测试"""

    def test_filtered_path_with_explicit_filter(self):
        """显式 filter 走 filtered 路径"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "test", "source_file": "陕国投A.pdf", "page_number": 1, "score": 0.9}
        ]

        router = QueryRouter(retriever=mock_retriever)

        result = router.route("营收是多少？", filter_file="陕国投A")

        assert result.retrieval_mode == "filtered"
        assert result.fallback_triggered is False
        assert result.filter_used == "陕国投A"
        mock_retriever.search.assert_called_once_with("营收是多少？", filter_file="陕国投A")

    def test_filtered_path_with_auto_filter(self):
        """自动识别公司走 filtered 路径"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "test", "source_file": "陕国投A.pdf", "page_number": 1, "score": 0.9}
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "陕国投A"

            result = router.route("陕国投的营收是多少？")

        assert result.retrieval_mode == "filtered"
        assert result.filter_used == "陕国投A"

    def test_filtered_path_with_results_no_fallback(self):
        """过滤检索有结果时不回退"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "test", "source_file": "a.pdf", "page_number": 1, "score": 0.9},
            {"chunk_id": "2", "chunk_text": "test2", "source_file": "a.pdf", "page_number": 2, "score": 0.8},
        ]

        router = QueryRouter(retriever=mock_retriever, empty_result_threshold=0)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "陕国投A"

            result = router.route("陕国投的营收是多少？")

        assert result.retrieval_mode == "filtered"
        assert result.fallback_triggered is False
        assert result.fallback_reason is None
        # 只调用一次检索
        assert mock_retriever.search.call_count == 1


class TestQueryRouterFallbackPath:
    """回退路径测试"""

    def test_fallback_triggered_when_empty(self):
        """过滤检索为空时触发回退"""
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [],  # 第一次：过滤检索返回空
            [{"chunk_id": "1", "chunk_text": "global", "source_file": "b.pdf", "page_number": 1, "score": 0.8}]
        ]

        router = QueryRouter(retriever=mock_retriever, empty_result_threshold=0)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "不存在的公司"

            result = router.route("不存在的公司的营收是多少？")

        assert result.retrieval_mode == "filtered_then_global"
        assert result.fallback_triggered is True
        assert result.fallback_reason == "filtered_returned_0_chunks"
        # 调用了两次检索
        assert mock_retriever.search.call_count == 2

    def test_fallback_returns_global_results(self):
        """回退后返回全局检索结果"""
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [],  # 过滤检索为空
            [
                {"chunk_id": "g1", "chunk_text": "global1", "source_file": "x.pdf", "page_number": 1, "score": 0.85},
                {"chunk_id": "g2", "chunk_text": "global2", "source_file": "y.pdf", "page_number": 2, "score": 0.75},
            ]
        ]

        router = QueryRouter(retriever=mock_retriever, empty_result_threshold=0)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "未知公司"

            result = router.route("未知公司的信息？")

        # 返回的是全局结果
        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == "g1"

    def test_explicit_filter_no_fallback(self):
        """显式 filter 默认不回退"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []  # 返回空

        router = QueryRouter(retriever=mock_retriever, empty_result_threshold=0)

        result = router.route("query", filter_file="指定公司")

        # 显式 filter 默认不回退，即使为空
        assert result.retrieval_mode == "filtered"
        assert result.fallback_triggered is False
        # 只调用一次检索
        assert mock_retriever.search.call_count == 1

    def test_auto_filter_allows_fallback(self):
        """自动识别允许回退"""
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [],  # 过滤检索为空
            [{"chunk_id": "1", "chunk_text": "global", "source_file": "b.pdf", "page_number": 1, "score": 0.8}]
        ]

        router = QueryRouter(retriever=mock_retriever, empty_result_threshold=0)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "自动识别的公司"

            result = router.route("自动识别的公司的信息？")

        assert result.fallback_triggered is True
        assert result.retrieval_mode == "filtered_then_global"

    def test_custom_threshold_fallback(self):
        """自定义回退阈值"""
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [{"chunk_id": "1", "chunk_text": "only one", "source_file": "a.pdf", "page_number": 1, "score": 0.9}],
            [
                {"chunk_id": "g1", "chunk_text": "global1", "source_file": "x.pdf", "page_number": 1, "score": 0.85},
                {"chunk_id": "g2", "chunk_text": "global2", "source_file": "y.pdf", "page_number": 2, "score": 0.75},
            ]
        ]

        # 阈值设为 1，<=1 条时回退
        router = QueryRouter(retriever=mock_retriever, empty_result_threshold=1)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "公司"

            result = router.route("query")

        # 有 1 条结果但阈值是 1，应触发回退
        assert result.fallback_triggered is True


class TestQueryRouterRerankScore:
    """rerank_score 保留测试"""

    def test_rerank_score_preserved(self):
        """精排分数应保留在结果中"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {
                "chunk_id": "1",
                "chunk_text": "test",
                "source_file": "a.pdf",
                "page_number": 1,
                "score": 0.9,
                "rerank_score": 0.95
            }
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = None

            result = router.route("query")

        assert result.chunks[0].rerank_score == 0.95

    def test_no_rerank_score_when_none(self):
        """无精排分数时应为 None"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {
                "chunk_id": "1",
                "chunk_text": "test",
                "source_file": "a.pdf",
                "page_number": 1,
                "score": 0.9,
                # 无 rerank_score
            }
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = None

            result = router.route("query")

        assert result.chunks[0].rerank_score is None


class TestQueryRouterContextStructure:
    """RetrievedContext 结构测试"""

    def test_context_structure_complete(self):
        """验证 RetrievedContext 结构完整"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "test", "source_file": "a.pdf", "page_number": 1, "score": 0.9}
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "公司"

            result = router.route("query")

        assert isinstance(result, RetrievedContext)
        assert result.query == "query"
        assert isinstance(result.classification, QueryClassification)
        assert result.retrieval_mode == "filtered"
        assert result.filter_used == "公司"
        assert result.fallback_triggered is False
        assert result.fallback_reason is None
        assert len(result.chunks) == 1
        assert result.chunks_count == 1
        assert result.retrieval_time >= 0

    def test_to_sources_list_format(self):
        """验证 sources_list 格式正确"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "test", "source_file": "a.pdf", "page_number": 1, "score": 0.9, "rerank_score": 0.95}
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = None

            result = router.route("query")

        sources = result.to_sources_list()
        assert len(sources) == 1
        assert sources[0]["file"] == "a.pdf"
        assert sources[0]["page"] == 1
        assert "score" in sources[0]
        assert "rerank_score" in sources[0]

    def test_to_dict_format(self):
        """验证 to_dict 格式正确"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"chunk_id": "1", "chunk_text": "test", "source_file": "a.pdf", "page_number": 1, "score": 0.9}
        ]

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = None

            result = router.route("query")

        d = result.to_dict()
        assert "query" in d
        assert "classification" in d
        assert "retrieval_mode" in d
        assert "filter_used" in d
        assert "fallback_triggered" in d
        assert "chunks" in d
        assert "chunks_count" in d
        assert "retrieval_time" in d


class TestQueryRouterEdgeCases:
    """边界情况测试"""

    def test_empty_query(self):
        """空查询处理"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        router = QueryRouter(retriever=mock_retriever)

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = None

            result = router.route("")

        assert isinstance(result, RetrievedContext)
        assert result.retrieval_mode == "global"

    def test_retriever_exception_propagated(self):
        """检索器异常应抛出"""
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = Exception("Retriever error")

        router = QueryRouter(retriever=mock_retriever)

        with pytest.raises(Exception, match="Retriever error"):
            router.route("query", filter_file="test")
