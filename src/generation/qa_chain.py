"""QA Chain - 串联检索和生成的完整问答流程"""
import re
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
from config import USE_RERANKER

# 触发两步重量级流程的场景关键词
# 命中任意一个词 → 走 extraction + conclude 两次调用
_TWO_STEP_KEYWORDS = re.compile(
    r"政策|条款|规定|约定|制度|规则|办法|细则|章程|"
    r"分红|派息|利润分配|股息|送股|转增|"
    r"贷款|担保|抵押|质押|回购|"
    r"如何|怎么|如何规定|如何约定|"
    r"提取|摘录|列出|说明|描述|介绍"
)


def _should_use_two_step(question: str) -> bool:
    """判断是否需要使用两步重量级流程。

    启发式：问题含 policy/extraction 类关键词时返回 True。
    """
    return bool(_TWO_STEP_KEYWORDS.search(question))


class QAChain:
    """问答链

    串联检索器和 LLM 服务，实现完整的问答流程：
    问题 → 检索相关段落 → 格式化上下文 → LLM 生成回答

    场景路由：
    - policy_qa / extraction 类问题 → 两步重量级：Step1 纯抽取 + Step2 结论生成
    - factual / comparison 类问题  → 单步：直接生成（原有流程）

    USE_RERANKER=True  → 默认使用 RerankRetriever（粗检索 30 → 精排 10）
    USE_RERANKER=False → 默认使用 Retriever（Baseline）
    """

    def __init__(
        self,
        retriever: Retriever | RerankRetriever | None = None,
        llm_service: BaseLLMService | None = None,
    ) -> None:
        """初始化 QA Chain

        Args:
            retriever:   检索器实例；None 时根据 USE_RERANKER 自动选择
            llm_service: LLM 服务实例
        """
        if retriever is not None:
            self.retriever = retriever
        elif USE_RERANKER:
            self.retriever = RerankRetriever()
        else:
            self.retriever = Retriever()
        self.llm_service = llm_service or get_llm_service()

    # ──────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """执行问答

        Args:
            question: 用户问题

        Returns:
            结构化结果，包含问题、答案、引用来源、检索耗时、生成耗时、mode
        """
        total_start = time.time()

        # 1. 检索相关段落
        retrieval_start = time.time()
        chunks = self.retriever.search(question)
        retrieval_time = time.time() - retrieval_start

        # 2. 格式化上下文
        context = format_context(chunks)

        # 3. 场景路由 → 选择生成策略
        generation_start = time.time()
        if _should_use_two_step(question):
            answer, mode = self._two_step_generate(context, question)
        else:
            answer, mode = self._single_step_generate(context, question)
        generation_time = time.time() - generation_start

        # 4. 整理引用来源
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

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "mode": mode,
        }

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


def ask(question: str) -> Dict[str, Any]:
    """便捷问答函数

    Args:
        question: 用户问题

    Returns:
        问答结果
    """
    qa_chain = QAChain()
    return qa_chain.ask(question)
