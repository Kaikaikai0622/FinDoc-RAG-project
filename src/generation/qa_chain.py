"""QA Chain - 串联检索和生成的完整问答流程"""
import time
from typing import Dict, Any

from src.retrieval import Retriever, RerankRetriever
from src.generation.llm_service import BaseLLMService, get_llm_service
from src.generation.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_USER_TEMPLATE,
    CONCLUDE_SYSTEM_PROMPT,
    CONCLUDE_USER_TEMPLATE,
    format_context,
)
from src.routing import QueryRouter, QueryClassification
from config import USE_RERANKER, ENABLE_QUERY_ROUTER



class QAChain:
    """问答链

    串联检索器和 LLM 服务，实现完整的问答流程：
    问题 → QueryRouter 路由检索 → 格式化上下文 → LLM 生成回答

    场景路由：
    - policy_qa / extraction 类问题 → 两步重量级：Step1 纯抽取 + Step2 结论生成
    - factual / comparison 类问题  → 单步：直接生成（原有流程）

    USE_RERANKER=True  → 默认使用 RerankRetriever（粗检索 30 → 精排 10）
    USE_RERANKER=False → 默认使用 Retriever（Baseline）
    ENABLE_QUERY_ROUTER=True → 使用 QueryRouter 进行路由和检索编排
    ENABLE_QUERY_ROUTER=False → 回退到旧逻辑（直接检索）
    """

    def __init__(
        self,
        retriever: Retriever | RerankRetriever | None = None,
        llm_service: BaseLLMService | None = None,
        router: QueryRouter | None = None,
    ) -> None:
        """初始化 QA Chain

        Args:
            retriever:   检索器实例；None 时根据 USE_RERANKER 自动选择
            llm_service: LLM 服务实例
            router:      QueryRouter 实例；None 时根据 ENABLE_QUERY_ROUTER 决定是否创建
        """
        # 保存 retriever 用于非 Router 模式或注入 Router
        if retriever is not None:
            self._retriever = retriever
        elif USE_RERANKER:
            self._retriever = RerankRetriever()
        else:
            self._retriever = Retriever()

        self.llm_service = llm_service or get_llm_service()

        # 初始化 QueryRouter（如果启用）
        if ENABLE_QUERY_ROUTER:
            # 如果传入 router 则使用，否则新建（使用上面的 _retriever）
            self.router = router or QueryRouter(retriever=self._retriever)
        else:
            self.router = None

    @property
    def retriever(self):
        """向后兼容：提供 retriever 属性访问。

        旧代码可能直接访问 qa_chain.retriever，此属性保证兼容性。
        """
        return self._retriever

    # ──────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────

    def ask(self, question: str, filter_file: str | None = None) -> Dict[str, Any]:
        """执行问答

        Args:
            question: 用户问题
            filter_file: 按来源文件名过滤（支持部分匹配，如"陕国投"）
                         若为 None，自动从问题中提取公司名称

        Returns:
            结构化结果，包含问题、答案、引用来源、检索耗时、生成耗时、mode、filter_used
            （以及 Router 相关的新增字段，如果 ENABLE_QUERY_ROUTER=True）
        """
        total_start = time.time()

        if self.router is not None:
            # ═══════════════════════════════════════════════════════════════
            # 新模式：使用 QueryRouter 进行路由和检索
            # ═══════════════════════════════════════════════════════════════
            retrieved_context = self.router.route(question, filter_file)

            # 提取检索结果
            chunks = [chunk.to_dict() for chunk in retrieved_context.chunks]
            retrieval_time = retrieved_context.retrieval_time

            # 使用 classification 决定生成模式
            generation_mode = retrieved_context.classification.generation_mode

            # 打印自动识别信息（保持与旧模式一致的用户体验）
            if (
                retrieved_context.classification.filter_source == "auto_company"
                and retrieved_context.classification.filter_file
            ):
                print(f"[自动识别] 从问题中提取到公司: {retrieved_context.classification.filter_file}")

        else:
            # ═══════════════════════════════════════════════════════════════
            # 旧模式：直接检索（向后兼容，ENABLE_QUERY_ROUTER=False 时）
            # ═══════════════════════════════════════════════════════════════
            from src.utils.company_resolver import extract_company_filter

            # 自动提取公司名称（如果未提供 filter_file）
            auto_filter = None
            if filter_file is None:
                auto_filter = extract_company_filter(question)
                if auto_filter:
                    print(f"[自动识别] 从问题中提取到公司: {auto_filter}")
            else:
                auto_filter = filter_file

            # 检索相关段落
            retrieval_start = time.time()
            chunks = self._retriever.search(question, filter_file=auto_filter)
            retrieval_time = time.time() - retrieval_start

            # 决定生成模式（旧逻辑）
            generation_mode = "two_step" if self._should_use_two_step(question) else "single_step"

            # 构建模拟的 classification（用于返回结构一致性）
            retrieved_context = None  # type: ignore

        # 格式化上下文
        context = format_context(chunks)

        # 执行生成
        generation_start = time.time()
        if generation_mode == "two_step":
            answer, mode = self._two_step_generate(context, question)
        else:
            answer, mode = self._single_step_generate(context, question)
        generation_time = time.time() - generation_start

        # 整理引用来源
        sources = []
        for chunk in chunks:
            source = {
                "file": chunk.get("source_file", ""),
                "page": chunk.get("page_number", 0),
                "score": chunk.get("score", 0),
            }
            if "rerank_score" in chunk:
                source["rerank_score"] = chunk["rerank_score"]
            sources.append(source)

        total_time = time.time() - total_start

        # 构建返回结果（保留所有旧字段，确保向后兼容）
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "mode": mode,
            "filter_used": (
                retrieved_context.filter_used
                if retrieved_context
                else (auto_filter if 'auto_filter' in locals() else filter_file)
            ),
            "filter_auto": (
                retrieved_context.classification.filter_source == "auto_company"
                if retrieved_context
                else (filter_file is None and 'auto_filter' in locals() and auto_filter is not None)
            ),
        }

        # 添加 Router 相关的新字段（如果使用了 Router）
        if retrieved_context is not None:
            result.update({
                "route_label": retrieved_context.classification.scene,
                "retrieval_mode": retrieved_context.retrieval_mode,
                "fallback_triggered": retrieved_context.fallback_triggered,
                "query_classifier": {
                    "scene": retrieved_context.classification.scene,
                    "generation_mode": retrieved_context.classification.generation_mode,
                    "filter_source": retrieved_context.classification.filter_source,
                    "retrieval_scope": retrieved_context.classification.retrieval_scope,
                    "confidence": retrieved_context.classification.confidence,
                    "reason_codes": retrieved_context.classification.reason_codes,
                },
                "retrieved_context": retrieved_context.to_dict(),
            })

        return result

    @staticmethod
    def _should_use_two_step(question: str) -> bool:
        """判断是否需要使用两步重量级流程（旧逻辑保留）。

        启发式：问题含 policy/extraction 类关键词时返回 True。
        """
        import re
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
        return bool(_TWO_STEP_KEYWORDS.search(question))

    # ──────────────────────────────────────────────
    # 生成策略：单步
    # ──────────────────────────────────────────────

    def _single_step_generate(self, context: str, question: str) -> tuple[str, str]:
        """单次 LLM 调用生成答案（factual / comparison 场景）。

        Returns:
            (answer_text, mode="single_step")
        """
        user_message = QA_USER_TEMPLATE.format(context=context, question=question)
        answer = self.llm_service.chat(QA_SYSTEM_PROMPT, user_message)
        return answer, "single_step"

    # ──────────────────────────────────────────────
    # 生成策略：两步重量级（policy_qa / extraction）
    # ──────────────────────────────────────────────

    def _two_step_generate(self, context: str, question: str) -> tuple[str, str]:
        """两次 LLM 调用：Step1 纯抽取条款原文，Step2 基于条款生成结论。

        Step1 指令模型只做"复制粘贴"式引用，禁止推断，确保 grounding。
        Step2 将 Step1 输出作为唯一依据，生成综合结论。

        Returns:
            (combined_answer, mode="two_step")
        """
        # ── Step 1：条款抽取 ──
        extract_user = EXTRACT_USER_TEMPLATE.format(context=context, question=question)
        extracted_clauses = self.llm_service.chat(EXTRACT_SYSTEM_PROMPT, extract_user)

        # ── Step 2：基于抽取结果生成结论 ──
        conclude_user = CONCLUDE_USER_TEMPLATE.format(
            question=question,
            extracted_clauses=extracted_clauses,
        )
        conclusion = self.llm_service.chat(CONCLUDE_SYSTEM_PROMPT, conclude_user)

        # 合并两步结果为完整回答
        combined = f"{extracted_clauses}\n\n{conclusion}"
        return combined, "two_step"


def ask(question: str, filter_file: str | None = None) -> Dict[str, Any]:
    """便捷问答函数

    Args:
        question: 用户问题
        filter_file: 按来源文件名过滤

    Returns:
        问答结果
    """
    qa_chain = QAChain()
    return qa_chain.ask(question, filter_file=filter_file)
