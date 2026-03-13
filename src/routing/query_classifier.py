"""Query Classifier - 查询分类器

根据查询文本和可选的过滤条件，输出统一的 QueryClassification。

当前实现（Phase 1）：基于规则的分类器
- 复用现有的 _should_use_two_step() 规则语义
- 复用 extract_company_filter() 进行公司识别

未来扩展：可接入 LLM-based classifier
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from config import ENABLE_QUERY_ROUTER
from src.routing.models import QueryClassification
from src.utils.company_resolver import extract_company_filter

if TYPE_CHECKING:
    from collections.abc import Sequence


# 触发两步生成流程的场景关键词（从 qa_chain.py 迁移）
_TWO_STEP_KEYWORDS = re.compile(
    r"政策|条款|规定|约定|制度|规则|办法|细则|章程|"
    r"分红|派息|利润分配|股息|送股|转增|"
    r"贷款|担保|抵押|质押|回购|"
    r"如何|怎么|如何规定|如何约定|"
    r"提取|摘录|列出|说明|描述|介绍|"
    r"与.*相比|和.*相比|同.*相比|跟.*相比|"
    r"有.*变化|有.*差异|有.*区别|有.*不同|"
    r"增长|下降|增加|减少|同比|环比"
)

# 场景识别关键词
_COMPARISON_KEYWORDS = re.compile(r"与.*相比|和.*相比|同.*相比|跟.*相比|对比|比较")
_EXTRACTION_KEYWORDS = re.compile(r"提取|摘录|列出|说明|描述|介绍|如何规定|如何约定")
_POLICY_KEYWORDS = re.compile(r"政策|条款|规定|约定|制度|规则|办法|细则|章程|公司章程")


class QueryClassifier:
    """查询分类器

    职责：
    1. 根据问题与可选 filter_file 做分类
    2. 输出统一的 QueryClassification
    3. 复用现有规则：extract_company_filter()、_should_use_two_step()

    规则优先级：
    1. 显式 filter_file 优先于自动识别
    2. 自动公司识别允许回退（配置决定）
    3. 显式 filter_file 默认不回退（配置决定）
    """

    def __init__(
        self,
        allow_auto_fallback: bool | None = None,
        allow_explicit_fallback: bool | None = None,
    ) -> None:
        """初始化分类器

        Args:
            allow_auto_fallback: 自动识别公司时是否允许回退，None 时使用配置
            allow_explicit_fallback: 显式 filter 是否允许回退，None 时使用配置
        """
        # 如未传入，使用配置默认值
        from config import (
            QUERY_ROUTER_ALLOW_AUTO_FILTER_FALLBACK,
            QUERY_ROUTER_ALLOW_EXPLICIT_FILTER_FALLBACK,
        )

        self.allow_auto_fallback = (
            allow_auto_fallback
            if allow_auto_fallback is not None
            else QUERY_ROUTER_ALLOW_AUTO_FILTER_FALLBACK
        )
        self.allow_explicit_fallback = (
            allow_explicit_fallback
            if allow_explicit_fallback is not None
            else QUERY_ROUTER_ALLOW_EXPLICIT_FILTER_FALLBACK
        )

    def classify(
        self,
        query: str,
        filter_file: str | None = None,
    ) -> QueryClassification:
        """对查询进行分类

        Args:
            query: 用户查询文本
            filter_file: 显式指定的过滤条件（可选）

        Returns:
            QueryClassification 分类结果
        """
        reason_codes: list[str] = []

        # Step 1: 确定 filter_file 来源
        if filter_file:
            # 显式指定
            final_filter = filter_file
            filter_source = "explicit"
            reason_codes.append("filter:explicit")
        else:
            # 尝试自动识别
            auto_filter = extract_company_filter(query)
            if auto_filter:
                final_filter = auto_filter
                filter_source = "auto_company"
                reason_codes.append(f"filter:auto_company:{auto_filter}")
            else:
                final_filter = None
                filter_source = "none"
                reason_codes.append("filter:none")

        # Step 2: 确定 retrieval_scope
        if final_filter:
            retrieval_scope = "single_company"
            reason_codes.append("scope:single_company")
        else:
            retrieval_scope = "global"
            reason_codes.append("scope:global")

        # Step 3: 确定 fallback_allowed
        if filter_source == "explicit":
            fallback_allowed = self.allow_explicit_fallback
            reason_codes.append(f"fallback:explicit:{fallback_allowed}")
        elif filter_source == "auto_company":
            fallback_allowed = self.allow_auto_fallback
            reason_codes.append(f"fallback:auto:{fallback_allowed}")
        else:
            fallback_allowed = True  # 无过滤时，回退无意义但允许
            reason_codes.append("fallback:n/a")

        # Step 4: 确定 scene 和 generation_mode
        scene = self._detect_scene(query)
        generation_mode = self._determine_generation_mode(query, scene)
        reason_codes.append(f"scene:{scene}")
        reason_codes.append(f"generation:{generation_mode}")

        # Step 5: 计算置信度（当前简单规则，后续可接入 LLM 置信度）
        confidence = self._calculate_confidence(
            filter_source, scene, bool(final_filter)
        )
        reason_codes.append(f"confidence:{confidence:.2f}")

        return QueryClassification(
            scene=scene,
            generation_mode=generation_mode,
            filter_file=final_filter,
            filter_source=filter_source,
            retrieval_scope=retrieval_scope,
            fallback_allowed=fallback_allowed,
            confidence=confidence,
            reason_codes=reason_codes,
        )

    def _detect_scene(self, query: str) -> str:
        """检测查询场景类型

        Args:
            query: 查询文本

        Returns:
            场景类型: factual/comparison/extraction/policy_qa/unknown
        """
        if _COMPARISON_KEYWORDS.search(query):
            return "comparison"
        if _EXTRACTION_KEYWORDS.search(query):
            return "extraction"
        if _POLICY_KEYWORDS.search(query):
            return "policy_qa"
        # 默认 factual
        return "factual"

    def _determine_generation_mode(self, query: str, scene: str) -> str:
        """确定生成模式

        Args:
            query: 查询文本
            scene: 已检测的场景类型

        Returns:
            "single_step" 或 "two_step"
        """
        # 命中 comparison / extraction / policy 时走 two_step
        if scene in ("comparison", "extraction", "policy_qa"):
            return "two_step"

        # 兜底：检查 two_step 关键词
        if _TWO_STEP_KEYWORDS.search(query):
            return "two_step"

        return "single_step"

    def _calculate_confidence(
        self,
        filter_source: str,
        scene: str,
        has_filter: bool,
    ) -> float:
        """计算分类置信度

        当前简单实现，后续可基于:
        - 公司识别置信度
        - 场景识别模糊度
        - 历史查询模式

        Args:
            filter_source: 过滤条件来源
            scene: 场景类型
            has_filter: 是否有过滤条件

        Returns:
            置信度分数 (0-1)
        """
        confidence = 0.8  # 基础置信度

        # 显式过滤更可靠
        if filter_source == "explicit":
            confidence += 0.15
        elif filter_source == "auto_company":
            confidence += 0.05  # 自动识别有一定不确定性

        # unknown 场景降低置信度
        if scene == "unknown":
            confidence -= 0.2

        return min(max(confidence, 0.0), 1.0)
