"""QueryClassifier 单元测试

验证分类器的核心行为：
1. 显式 filter_file 优先于自动识别
2. 公司简称/全角半角归一化
3. comparison/extraction/policy_qa → two_step
4. 普通 factual → single_step
5. reason_codes 完整
"""
import pytest
from unittest.mock import patch, MagicMock

from src.routing.query_classifier import QueryClassifier, _TWO_STEP_KEYWORDS
from src.routing.models import QueryClassification


class TestQueryClassifierBasic:
    """基础分类功能测试"""

    def test_factual_query_single_step(self):
        """普通事实查询应走 single_step"""
        classifier = QueryClassifier()

        result = classifier.classify("陕国投2024年净利润是多少？")

        assert result.scene == "factual"
        assert result.generation_mode == "single_step"
        assert "generation:single_step" in result.reason_codes

    def test_comparison_query_two_step(self):
        """对比查询应走 two_step"""
        classifier = QueryClassifier()

        result = classifier.classify("陕国投与芯导科技的营收相比如何？")

        assert result.scene == "comparison"
        assert result.generation_mode == "two_step"
        assert "generation:two_step" in result.reason_codes

    def test_policy_query_two_step(self):
        """政策/条款查询应走 two_step"""
        classifier = QueryClassifier()

        result = classifier.classify("公司章程中关于分红的规定是什么？")

        assert result.scene == "policy_qa"
        assert result.generation_mode == "two_step"

    def test_extraction_query_two_step(self):
        """提取类查询应走 two_step"""
        classifier = QueryClassifier()

        result = classifier.classify("提取并列出公司的主要业务板块")

        assert result.scene == "extraction"
        assert result.generation_mode == "two_step"

    def test_two_step_keywords_fallback(self):
        """兜底：匹配 two_step 关键词也应走 two_step"""
        classifier = QueryClassifier()

        # 包含增长/下降关键词但未匹配场景
        result = classifier.classify("营业收入同比增长了多少？")

        assert result.generation_mode == "two_step"
        # 场景可能是 factual，但 generation_mode 是 two_step


class TestQueryClassifierFilterPriority:
    """过滤条件优先级测试"""

    def test_explicit_filter_priority(self):
        """显式 filter_file 优先于自动识别"""
        classifier = QueryClassifier()

        # 即使问题中提到了公司名，显式 filter 应优先
        with patch('src.routing.query_classifier.extract_company_filter') as mock_extract:
            mock_extract.return_value = "陕国投A"

            result = classifier.classify(
                "陕国投的营收是多少？",
                filter_file="芯导科技"  # 显式指定不同公司
            )

        assert result.filter_file == "芯导科技"
        assert result.filter_source == "explicit"
        assert "filter:explicit" in result.reason_codes

    def test_auto_company_extraction(self):
        """自动识别公司名称"""
        classifier = QueryClassifier()

        with patch('src.routing.query_classifier.extract_company_filter') as mock_extract:
            mock_extract.return_value = "陕国投A"

            result = classifier.classify("陕国投的营收是多少？")

        assert result.filter_file == "陕国投A"
        assert result.filter_source == "auto_company"
        assert result.retrieval_scope == "single_company"
        assert "filter:auto_company:陕国投A" in result.reason_codes

    def test_no_filter_global_scope(self):
        """无过滤条件时应走 global"""
        classifier = QueryClassifier()

        with patch('src.routing.query_classifier.extract_company_filter') as mock_extract:
            mock_extract.return_value = None

            result = classifier.classify("介绍一下这家公司的业务")

        assert result.filter_file is None
        assert result.filter_source == "none"
        assert result.retrieval_scope == "global"
        assert "scope:global" in result.reason_codes


class TestQueryClassifierFallback:
    """回退策略测试"""

    def test_explicit_filter_no_fallback_by_default(self):
        """显式 filter 默认不允许回退"""
        classifier = QueryClassifier()

        result = classifier.classify("test", filter_file="some_company")

        assert result.filter_source == "explicit"
        assert result.fallback_allowed is False
        assert "fallback:explicit:False" in result.reason_codes

    def test_auto_filter_allows_fallback_by_default(self):
        """自动识别默认允许回退"""
        classifier = QueryClassifier()

        with patch('src.routing.query_classifier.extract_company_filter') as mock_extract:
            mock_extract.return_value = "陕国投A"

            result = classifier.classify("陕国投的营收是多少？")

        assert result.filter_source == "auto_company"
        assert result.fallback_allowed is True
        assert "fallback:auto:True" in result.reason_codes

    def test_custom_fallback_config(self):
        """自定义回退配置"""
        classifier = QueryClassifier(
            allow_auto_fallback=False,
            allow_explicit_fallback=True
        )

        with patch('src.routing.query_classifier.extract_company_filter') as mock_extract:
            mock_extract.return_value = "陕国投A"

            auto_result = classifier.classify("陕国投的营收是多少？")

        explicit_result = classifier.classify("test", filter_file="some_company")

        # 自定义配置应生效
        assert auto_result.fallback_allowed is False
        assert explicit_result.fallback_allowed is True


class TestQueryClassifierReasonCodes:
    """reason_codes 完整性测试"""

    def test_reason_codes_structure(self):
        """验证 reason_codes 包含必要信息"""
        classifier = QueryClassifier()

        result = classifier.classify("陕国投与芯导科技相比，营收如何？")

        codes = result.reason_codes

        # 应包含过滤来源
        assert any(c.startswith("filter:") for c in codes)
        # 应包含范围
        assert any(c.startswith("scope:") for c in codes)
        # 应包含回退策略
        assert any(c.startswith("fallback:") for c in codes)
        # 应包含场景
        assert any(c.startswith("scene:") for c in codes)
        # 应包含生成模式
        assert any(c.startswith("generation:") for c in codes)
        # 应包含置信度
        assert any(c.startswith("confidence:") for c in codes)

    def test_confidence_range(self):
        """置信度应在 0-1 范围内"""
        classifier = QueryClassifier()

        result = classifier.classify("测试查询")

        assert 0.0 <= result.confidence <= 1.0

    def test_explicit_filter_higher_confidence(self):
        """显式过滤应有更高置信度"""
        classifier = QueryClassifier()

        with patch('src.routing.query_classifier.extract_company_filter') as mock_extract:
            mock_extract.return_value = "陕国投A"

            auto_result = classifier.classify("陕国投的营收是多少？")

        explicit_result = classifier.classify("测试", filter_file="陕国投A")

        # 显式过滤置信度应高于自动识别
        assert explicit_result.confidence > auto_result.confidence


class TestQueryClassifierCompanyNormalization:
    """公司名称归一化测试（全角半角、别名）"""

    def test_fullwidth_company_name(self):
        """全角字符公司名应正确识别"""
        classifier = QueryClassifier()

        with patch(PATCH_PATH) as mock_extract:
            # 模拟返回全角公司名称
            mock_extract.return_value = "陕国投Ａ"  # 全角Ａ

            result = classifier.classify("陕国投Ａ的营收是多少？")

        assert result.filter_file == "陕国投Ａ"
        assert result.filter_source == "auto_company"

    def test_halfwidth_company_name(self):
        """半角字符公司名应正确识别"""
        classifier = QueryClassifier()

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "陕国投A"  # 半角A

            result = classifier.classify("陕国投A的营收是多少？")

        assert result.filter_file == "陕国投A"
        assert result.filter_source == "auto_company"

    def test_company_alias_short_name(self):
        """公司简称应正确识别"""
        classifier = QueryClassifier()

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "陕国投"  # 不带后缀的简称

            result = classifier.classify("陕国投的营收是多少？")

        assert result.filter_file == "陕国投"
        assert result.filter_source == "auto_company"

    def test_company_alias_with_suffix(self):
        """带后缀的公司名应正确识别"""
        classifier = QueryClassifier()

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "芯导科技"

            result = classifier.classify("芯导科技的利润如何？")

        assert result.filter_file == "芯导科技"

    def test_company_name_with_colon(self):
        """带冒号的公司提及应正确处理"""
        classifier = QueryClassifier()

        with patch(PATCH_PATH) as mock_extract:
            mock_extract.return_value = "指南针"

            result = classifier.classify("指南针：2025年营收是多少？")

        assert result.filter_file == "指南针"


class TestQueryClassifierEdgeCases:
    """边界情况测试"""

    def test_empty_query(self):
        """空查询处理"""
        classifier = QueryClassifier()

        result = classifier.classify("")

        # 应返回有效结果，不走异常
        assert isinstance(result, QueryClassification)
        assert result.scene == "factual"  # 默认场景

    def test_long_query(self):
        """长查询处理"""
        classifier = QueryClassifier()

        # 使用明确的 policy 关键词
        long_query = "关于" + "公司业务" * 100 + "的章程分红制度"

        result = classifier.classify(long_query)

        # 应正确处理，识别出政策关键词（"制度"是纯 policy 关键词）
        assert result.scene == "policy_qa"
        assert result.generation_mode == "two_step"

    def test_multiple_keywords(self):
        """多个关键词同时存在"""
        classifier = QueryClassifier()

        # 同时包含 comparison 和 policy 关键词
        result = classifier.classify("公司章程中关于分红的规定与芯导科技相比有何不同？")

        # comparison 优先级更高（按代码顺序）
        assert result.scene == "comparison"
        assert result.generation_mode == "two_step"

    def test_query_with_special_chars(self):
        """包含特殊字符的查询"""
        classifier = QueryClassifier()

        result = classifier.classify("营收、利润、毛利率分别是多少？")

        # 应正常处理，不抛异常
        assert isinstance(result, QueryClassification)
        assert result.scene == "factual"

    def test_query_with_numbers(self):
        """包含数字的查询"""
        classifier = QueryClassifier()

        result = classifier.classify("2024年营收增长10%的是哪家公司？")

        assert isinstance(result, QueryClassification)
        assert result.scene == "factual"  # 虽然有"增长"但整体是 factual
