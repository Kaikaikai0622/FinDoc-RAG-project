"""评估引擎模块

实现 RAG 系统的自动化评估，包括基础指标和 Ragas 高级指标。
"""
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.generation.qa_chain import QAChain
from src.evaluation.dataset import EvalDataset
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    QWEN_MODEL,
    QWEN_BASE_URL,
    KIMI_MODEL,
    KIMI_BASE_URL,
    ENABLE_QUERY_ROUTER,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果数据结构"""
    run_name: str
    timestamp: str
    config: dict
    total_questions: int

    # 基础指标（全部问题）
    accuracy: float
    retrieval_hit_rate: float
    avg_retrieval_rank: float

    # 场景权重修正后的准确率（仅 synthetic 问题，基于 synthetic 自身场景分布）
    # manual 问题不计算加权准确率
    # 参考理想权重（不参与计算，仅供对比）: factual=35%, extraction=35%, policy_qa=20%, comparison=10%
    weighted_accuracy: float = 0.0

    # Synthetic 问题独立指标（决策2：synthetic 准确率分离计算）
    synthetic_accuracy: float = 0.0           # synthetic 子集初始准确率
    synthetic_weighted_accuracy: float = 0.0  # synthetic 子集加权准确率
    synthetic_count: int = 0                  # synthetic 问题数量

    # Manual 问题独立指标
    manual_accuracy: float = 0.0              # manual 子集准确率（无加权）
    manual_count: int = 0                     # manual 问题数量

    # Ragas 指标
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # 按场景分组的指标
    scene_metrics: dict = field(default_factory=dict)

    # 逐条明细
    details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    def save(self, path: str) -> None:
        """保存为 JSON 文件"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"评估结果已保存到: {path}")


class Evaluator:
    """RAG 系统评估器

    评估流程：
    1. 遍历评估数据集
    2. 对每条问题调用 QA Chain 获取答案
    3. 计算基础指标（准确率、检索命中率等）
    4. 使用 Ragas 计算高级指标
    5. 生成评估报告
    """

    def __init__(self, qa_chain: QAChain | None = None, llm_service=None):
        """初始化评估器

        Args:
            qa_chain:    QA Chain 实例
            llm_service: LLM 服务实例，用于 LLM judge；None 时自动选取与 QAChain 相同的 provider
        """
        from src.generation.llm_service import get_llm_service
        self.qa_chain = qa_chain or QAChain()
        self._llm_service = llm_service or get_llm_service()

    def run(
        self,
        dataset: EvalDataset,
        run_name: str = "default",
        include_ragas: bool = True,
    ) -> EvalResult:
        """运行评估

        Args:
            dataset: 评估数据集
            run_name: 运行名称（用于标识本次评估）
            include_ragas: 是否计算 Ragas 指标

        Returns:
            评估结果
        """
        logger.info(f"开始评估 run_name={run_name}, 数据集大小={len(dataset.get_all())}")

        # 获取所有 QA 对
        qa_pairs = dataset.get_all()

        if not qa_pairs:
            logger.warning("评估数据集为空")
            return EvalResult(
                run_name=run_name,
                timestamp=datetime.now().isoformat(),
                config=self._get_config(),
                total_questions=0,
                accuracy=0.0,
                retrieval_hit_rate=0.0,
                avg_retrieval_rank=0.0,
            )

        # 逐条评估
        details = []
        correct_count = 0
        hit_count = 0
        rank_sum = 0
        hit_count_for_avg = 0

        # 收集 Ragas 需要的批量数据
        questions = []
        ground_truths = []
        model_answers = []
        contexts = []

        # 决策2：分离 manual 和 synthetic 问题的统计
        manual_correct = 0
        manual_count = 0
        synthetic_correct = 0
        synthetic_count = 0
        synthetic_details = []  # 仅 synthetic 问题的 details 用于加权计算

        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            ground_truth = qa["ground_truth"]
            scene = qa.get("scene", "unknown")
            difficulty = qa.get("difficulty", "unknown")
            # 支持 source_files/source_pages 数组格式（优先）或 source_file/source_page 字符串格式（向后兼容）
            expected_source_files = qa.get("source_files") or ([qa.get("source_file", "")] if qa.get("source_file") else [])
            expected_source_pages = qa.get("source_pages") or ([qa.get("source_page", 0)] if qa.get("source_page") else [])

            logger.info(f"[{i+1}/{len(qa_pairs)}] 评估问题: {question[:50]}...")

            try:
                # 调用 QA Chain
                result = self.qa_chain.ask(question)
                model_answer = result.get("answer", "")
                retrieved_sources = result.get("sources", [])

                # ═══════════════════════════════════════════════════════════════
                # Phase 6 Fix: 使用真实检索上下文，而不是重新检索
                # ═══════════════════════════════════════════════════════════════
                if "retrieved_context" in result and result["retrieved_context"]:
                    # Router 模式：使用 QAChain 返回的真实上下文
                    retrieved_chunks = [
                        {
                            "chunk_id": chunk.get("chunk_id", ""),
                            "chunk_text": chunk.get("chunk_text", ""),
                            "source_file": chunk.get("source_file", ""),
                            "page_number": chunk.get("page_number", 0),
                            "score": chunk.get("score", 0),
                        }
                        for chunk in result["retrieved_context"].get("chunks", [])
                    ]
                    # 提取路由元信息
                    router_metadata = {
                        "route_label": result.get("route_label"),
                        "retrieval_mode": result.get("retrieval_mode"),
                        "fallback_triggered": result.get("fallback_triggered"),
                        "query_classifier": result.get("query_classifier"),
                    }
                else:
                    # 旧模式回退：兼容未启用 Router 的情况
                    # 需要将 sources 转换为 chunks 格式
                    retrieved_chunks = [
                        {
                            "chunk_id": source.get("file", "") + f"_{i}",
                            "chunk_text": "",  # 旧模式可能没有 chunk_text
                            "source_file": source.get("file", ""),
                            "page_number": source.get("page", 0),
                            "score": source.get("score", 0),
                        }
                        for i, source in enumerate(retrieved_sources)
                    ]
                    router_metadata = None

                # 计算基础指标
                basic_metrics = self._compute_basic_metrics(
                    question=question,
                    ground_truth=ground_truth,
                    model_answer=model_answer,
                    retrieved_sources=retrieved_sources,
                    retrieved_chunks=retrieved_chunks,
                    expected_source_files=expected_source_files,
                    expected_source_pages=expected_source_pages,
                    scene=scene,
                )

                # 收集 Ragas 数据
                if include_ragas:
                    questions.append(question)
                    ground_truths.append(ground_truth)
                    model_answers.append(model_answer)
                    contexts.append([chunk.get("chunk_text", "") for chunk in retrieved_chunks])

                # 统计基础指标
                is_correct = basic_metrics["answer_correct"]
                if is_correct:
                    correct_count += 1
                if basic_metrics["retrieval_hit"]:
                    hit_count += 1
                    rank_sum += basic_metrics["retrieval_rank"]
                    hit_count_for_avg += 1

                # 决策2：识别问题来源并分别统计
                qid = qa.get("id", f"q{i+1}")
                is_manual = qid.startswith("m")  # manual 问题 ID 以 m 开头
                is_synthetic = not is_manual     # synthetic 问题 ID 以 s 开头或数字开头

                if is_manual:
                    manual_count += 1
                    if is_correct:
                        manual_correct += 1
                else:
                    synthetic_count += 1
                    if is_correct:
                        synthetic_correct += 1
                    # 收集 synthetic 问题的详情用于后续加权计算
                    synthetic_details.append({
                        "id": qid,
                        "scene": scene,
                        "answer_correct": is_correct,
                    })

                # 记录详情
                detail_entry = {
                    "id": qid,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "scene": scene,
                    "difficulty": difficulty,
                    "is_manual": is_manual,
                    "is_synthetic": is_synthetic,
                    "sources": retrieved_sources,
                    **basic_metrics,
                }
                # 添加路由元信息（如果存在）
                if router_metadata:
                    detail_entry["route_label"] = router_metadata.get("route_label")
                    detail_entry["retrieval_mode"] = router_metadata.get("retrieval_mode")
                    detail_entry["fallback_triggered"] = router_metadata.get("fallback_triggered")
                    if router_metadata.get("query_classifier"):
                        qc = router_metadata["query_classifier"]
                        detail_entry["query_classifier"] = {
                            "scene": qc.get("scene"),
                            "generation_mode": qc.get("generation_mode"),
                            "filter_source": qc.get("filter_source"),
                            "retrieval_scope": qc.get("retrieval_scope"),
                            "confidence": qc.get("confidence"),
                        }
                details.append(detail_entry)

            except Exception as e:
                logger.error(f"评估问题 {question[:30]}... 时出错: {e}")
                qid = qa.get("id", f"q{i+1}")
                is_manual = qid.startswith("m")
                is_synthetic = not is_manual

                # 错误时也统计到对应类型
                if is_manual:
                    manual_count += 1
                else:
                    synthetic_count += 1
                    synthetic_details.append({
                        "id": qid,
                        "scene": scene,
                        "answer_correct": False,
                    })

                details.append({
                    "id": qid,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"ERROR: {e}",
                    "scene": scene,
                    "difficulty": difficulty,
                    "is_manual": is_manual,
                    "is_synthetic": is_synthetic,
                    "answer_correct": False,
                    "retrieval_hit": False,
                    "retrieval_rank": -1,
                    "error": str(e),
                })

        # 计算基础指标
        total = len(qa_pairs)
        accuracy = correct_count / total if total > 0 else 0.0
        retrieval_hit_rate = hit_count / total if total > 0 else 0.0
        avg_retrieval_rank = rank_sum / hit_count_for_avg if hit_count_for_avg > 0 else -1.0

        # 计算 Ragas 指标
        if include_ragas and questions:
            try:
                ragas_metrics = self._compute_ragas_metrics(
                    questions=questions,
                    ground_truths=ground_truths,
                    model_answers=model_answers,
                    contexts=contexts,
                )
            except Exception as e:
                logger.warning(f"Ragas 指标计算失败: {e}")
                ragas_metrics = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                }
        else:
            ragas_metrics = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

        # 按场景分组统计
        scene_metrics = self._compute_scene_metrics(details)

        # 决策2：计算 manual 和 synthetic 的独立指标
        # manual 准确率（无加权）
        manual_accuracy_val = manual_correct / manual_count if manual_count > 0 else 0.0

        # synthetic 准确率
        synthetic_accuracy_val = synthetic_correct / synthetic_count if synthetic_count > 0 else 0.0

        # synthetic 加权准确率（仅基于 synthetic 自身场景分布）
        synthetic_weighted_accuracy_val = self._compute_weighted_accuracy(
            synthetic_details, synthetic_accuracy_val
        ) if synthetic_count > 0 else 0.0

        # 全局加权准确率（基于全部问题的场景分布，用于向后兼容）
        weighted_accuracy = self._compute_weighted_accuracy(details, accuracy)

        # 构建结果
        result = EvalResult(
            run_name=run_name,
            timestamp=datetime.now().isoformat(),
            config=self._get_config(),
            total_questions=total,
            accuracy=accuracy,
            retrieval_hit_rate=retrieval_hit_rate,
            avg_retrieval_rank=avg_retrieval_rank,
            # 加权准确率字段
            weighted_accuracy=weighted_accuracy,
            # Synthetic 独立指标
            synthetic_accuracy=synthetic_accuracy_val,
            synthetic_weighted_accuracy=synthetic_weighted_accuracy_val,
            synthetic_count=synthetic_count,
            # Manual 独立指标
            manual_accuracy=manual_accuracy_val,
            manual_count=manual_count,
            # Ragas 指标
            faithfulness=ragas_metrics.get("faithfulness", 0.0),
            answer_relevancy=ragas_metrics.get("answer_relevancy", 0.0),
            context_precision=ragas_metrics.get("context_precision", 0.0),
            context_recall=ragas_metrics.get("context_recall", 0.0),
            scene_metrics=scene_metrics,
            details=details,
        )

        logger.info(f"评估完成: accuracy={accuracy:.2%}, hit_rate={retrieval_hit_rate:.2%}")

        return result

    def _compute_basic_metrics(
        self,
        question: str,
        ground_truth: str,
        model_answer: str,
        retrieved_sources: list[dict],
        retrieved_chunks: list[dict],
        expected_source_files: list[str] | None = None,
        expected_source_pages: list[int] | None = None,
        scene: str = "",
    ) -> dict:
        """计算基础指标

        Args:
            question: 问题文本（保留用于后续扩展）
            ground_truth: 标准答案
            model_answer: 模型生成的答案
            retrieved_sources: QA Chain 返回的来源列表
            retrieved_chunks: 检索到的 chunks
            expected_source_files: 标注来源文件列表
            expected_source_pages: 标注来源页码列表
            scene: 问题场景

        Returns:
            基础指标字典
        """
        # 检查答案是否正确（模糊匹配）
        answer_correct = self._check_answer_correct(ground_truth, model_answer)

        # 检查检索是否命中
        retrieval_hit = False
        retrieval_rank = -1

        # out_of_scope 场景不要求命中具体来源
        if scene == "out_of_scope":
            retrieval_hit = True
            retrieval_rank = 1
        else:
            # 默认空列表
            if expected_source_files is None:
                expected_source_files = []
            if expected_source_pages is None:
                expected_source_pages = []

            # 优先按来源文件 + 页码判断命中（任意匹配即命中）
            for i, source in enumerate(retrieved_sources):
                source_file = str(source.get("file", ""))
                source_page = int(source.get("page", 0) or 0)
                if self._source_match(expected_source_files, expected_source_pages, source_file, source_page):
                    retrieval_hit = True
                    retrieval_rank = i + 1
                    break

            # 若来源字段未命中，回退到文本包含判定
            if not retrieval_hit:
                for i, chunk in enumerate(retrieved_chunks):
                    chunk_text = chunk.get("chunk_text", "")
                    if self._check_chunk_contains_answer(ground_truth, chunk_text):
                        retrieval_hit = True
                        retrieval_rank = i + 1  # 1-based
                        break

        return {
            "answer_correct": answer_correct,
            "retrieval_hit": retrieval_hit,
            "retrieval_rank": retrieval_rank,
        }

    def _source_match(
        self,
        expected_files: list[str],
        expected_pages: list[int],
        actual_file: str,
        actual_page: int,
    ) -> bool:
        """判断标注来源与检索来源是否匹配（支持列表，任意匹配即命中）"""
        if not expected_files:
            return False

        # 检查是否有"任意文件"标记
        if any(f in {"（任意文件）", "(任意文件)"} for f in expected_files):
            return True

        # 遍历所有标注的 (file, page) 组合，任意匹配即命中
        for expected_file, expected_page in zip(expected_files, expected_pages):
            # 页码匹配：expected_page=0 表示忽略页码，或精确匹配
            page_match = (expected_page == 0) or (expected_page == actual_page)

            # 文件名做宽松匹配
            ef = expected_file.replace("：", ":").strip()
            af = actual_file.replace("：", ":").strip()
            file_match = (ef == af) or (ef in af) or (af in ef)

            if page_match and file_match:
                return True

        return False

    def _check_answer_correct(
        self,
        ground_truth: str,
        model_answer: str,
    ) -> bool:
        """检查模型答案是否包含标准答案的关键信息。

        流程：LLM judge → 失败时退回启发式规则。

        Args:
            ground_truth: 标准答案
            model_answer: 模型答案

        Returns:
            是否正确
        """
        # 特殊情况：不可回答问题，检查模型是否正确拒答
        if ground_truth.strip() == "__UNANSWERABLE__":
            refuse_patterns = [
                "无法找到", "无法回答", "没有相关信息", "无法确定",
                "未找到", "未披露", "未在", "无法获取", "没有找到",
                "cannot find", "unable to", "no information", "not found",
            ]
            return any(p in model_answer.lower() for p in refuse_patterns)

        # LLM judge 优先
        try:
            return self._llm_judge(ground_truth, model_answer)
        except Exception as e:
            logger.warning(f"LLM judge 失败，退回启发式: {e}")
            return self._heuristic_check(ground_truth, model_answer)

    def _llm_judge(self, ground_truth: str, model_answer: str) -> bool:
        """调用 LLM 判断模型答案是否包含 ground truth 的核心信息。

        判分原则：
        - 模型答案包含了 GT 的关键信息，即便更详细，也判为正确
        - 只有核心结论（数值/是非/主体）不一致时，才判为错误
        """
        system_prompt = (
            "你是一个严格但公平的 RAG 系统评分员。\n\n"
            "判分规则（必须严格遵守）：\n"
            "1. 如果模型答案包含了标准答案的关键信息（核心数值、是非结论、主体名称），"
            "即便模型答案更详细或有额外说明，也应判为【正确】\n"
            "2. 只有当模型答案的核心结论（数值/是否/主体）与标准答案明显不一致时，才判为【错误】\n"
            "3. 模型答案「更详细」不是错误；模型答案「有额外说明」不是错误\n"
            "4. 若模型答案明确表示无法找到信息，但标准答案有具体内容，判为【错误】\n\n"
            "输出格式（只输出这一行，不要任何解释）：\n"
            "CORRECT 或 INCORRECT"
        )
        user_message = (
            f"标准答案：\n{ground_truth}\n\n"
            f"模型回答：\n{model_answer}\n\n"
            "请判断模型回答是否正确（CORRECT / INCORRECT）："
        )
        response = self._llm_service.chat(system_prompt, user_message).strip().upper()
        if "INCORRECT" in response:
            return False
        if "CORRECT" in response:
            return True
        # 无法解析时退回启发式
        logger.warning(f"LLM judge 输出无法解析: {response!r}，退回启发式")
        return self._heuristic_check(ground_truth, model_answer)

    def _heuristic_check(self, ground_truth: str, model_answer: str) -> bool:
        """启发式规则判断（LLM judge 的兜底）。"""
        import re

        def normalize_numbers(text: str) -> set:
            text = text.replace(",", "")
            text = re.sub(r'(\d+\.?\d*)\s*亿', lambda m: str(float(m.group(1)) * 1e8), text)
            text = re.sub(r'(\d+\.?\d*)\s*万', lambda m: str(float(m.group(1)) * 1e4), text)
            text = re.sub(r'(\d+\.?\d*)\s*%', r'\1', text)
            return set(re.findall(r'\d+\.?\d*', text))

        gt_numbers = normalize_numbers(ground_truth)
        ma_numbers = normalize_numbers(model_answer)
        if gt_numbers:
            gt_main = {n for n in gt_numbers if len(n) > 2}
            if gt_main:
                match_count = sum(1 for n in gt_main if any(n in m or m in n for m in ma_numbers))
                if match_count >= len(gt_main) * 0.4:
                    return True
            elif gt_numbers.issubset(ma_numbers):
                return True

        gt_kw = set(re.findall(r'[\u4e00-\u9fff]{2,}', ground_truth))
        ma_kw = set(re.findall(r'[\u4e00-\u9fff]{2,}', model_answer))
        if gt_kw and len(gt_kw & ma_kw) >= len(gt_kw) * 0.4:
            return True
        return False

    def _check_chunk_contains_answer(
        self,
        ground_truth: str,
        chunk_text: str,
    ) -> bool:
        """检查 chunk 中是否包含答案信息

        Args:
            ground_truth: 标准答案
            chunk_text: chunk 文本

        Returns:
            是否包含
        """
        import re

        # 提取数字进行比对
        gt_numbers = set(re.findall(r'[\d,.]+', ground_truth))
        chunk_numbers = set(re.findall(r'[\d,.]+', chunk_text))

        # 如果有数字，检查是否有交集
        if gt_numbers and gt_numbers & chunk_numbers:
            return True

        # 检查关键词
        gt_keywords = set(re.findall(r'[\u4e00-\u9fff]{3,}', ground_truth))
        chunk_keywords = set(re.findall(r'[\u4e00-\u9fff]{3,}', chunk_text))

        if gt_keywords:
            match_count = len(gt_keywords & chunk_keywords)
            if match_count >= len(gt_keywords) * 0.3:  # 30% 以上匹配
                return True

        return False

    def _compute_ragas_metrics(
        self,
        questions: list[str],
        ground_truths: list[str],
        model_answers: list[str],
        contexts: list[list[str]],
    ) -> dict:
        """使用 Ragas 计算高级指标

        Args:
            questions: 问题列表
            ground_truths: 标准答案列表
            model_answers: 模型答案列表
            contexts: 检索上下文列表

        Returns:
            Ragas 指标字典
        """
        try:
            from ragas import EvaluationDataset
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI
            from langchain_community.embeddings import HuggingFaceEmbeddings

            # Ragas 0.4.x 要求每条样本是 mapping（SingleTurnSample 字段）
            samples = []
            for question, answer, context_list, ground_truth in zip(
                questions, model_answers, contexts, ground_truths
            ):
                samples.append({
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": context_list,
                    "reference": ground_truth,
                })
            eval_dataset = EvaluationDataset.from_list(samples)

            # 优先读取显式的 Ragas/OpenAI 兼容配置（必须 key+base 同时存在）
            ragas_env_key = os.environ.get("RAGAS_OPENAI_API_KEY")
            ragas_env_base = os.environ.get("RAGAS_OPENAI_BASE_URL")
            ragas_env_model = os.environ.get("RAGAS_OPENAI_MODEL")
            openai_key = os.environ.get("OPENAI_API_KEY")
            openai_base = os.environ.get("OPENAI_BASE_URL")

            if ragas_env_key and ragas_env_base:
                ragas_key = ragas_env_key
                ragas_base = ragas_env_base
                ragas_model = ragas_env_model or QWEN_MODEL
            elif openai_key and openai_base:
                ragas_key = openai_key
                ragas_base = openai_base
                ragas_model = ragas_env_model or QWEN_MODEL
            elif LLM_PROVIDER == "kimi":
                ragas_key = os.environ.get("MOONSHOT_API_KEY")
                ragas_base = KIMI_BASE_URL
                ragas_model = KIMI_MODEL
            else:
                ragas_key = os.environ.get("DASHSCOPE_API_KEY")
                ragas_base = QWEN_BASE_URL
                ragas_model = QWEN_MODEL

            ragas_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=ragas_model,
                    api_key=ragas_key,
                    base_url=ragas_base,
                    temperature=0.0,
                )
            )
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                )
            )

            # 运行评估
            logger.info("开始计算 Ragas 指标...")
            results = evaluate(
                dataset=eval_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False,
            )

            # 提取指标均值
            df = results.to_pandas()

            return {
                "faithfulness": self._safe_metric_mean(df, "faithfulness"),
                "answer_relevancy": self._safe_metric_mean(df, "answer_relevancy"),
                "context_precision": self._safe_metric_mean(df, "context_precision"),
                "context_recall": self._safe_metric_mean(df, "context_recall"),
            }

        except Exception as e:
            logger.warning(f"Ragas 评估失败: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

    def _safe_metric_mean(self, df: Any, col: str) -> float:
        """安全计算指标均值（处理缺列与 NaN）。

        Returns:
            float: 正常计算结果或 0.0（当计算失败时）
            注意：返回 0.0 可能表示两种情况：(1) 真实计算结果；(2) 计算失败
        """
        import logging
        logger = logging.getLogger(__name__)

        if col not in df:
            logger.warning(f"Ragas metric '{col}' not found in results, treating as N/A")
            return 0.0
        try:
            # 检查是否有有效值
            valid_values = df[col].dropna()
            if len(valid_values) == 0:
                logger.warning(f"Ragas metric '{col}' has no valid values, treating as N/A")
                return 0.0
            value = float(valid_values.mean())
        except Exception as e:
            logger.warning(f"Ragas metric '{col}' calculation failed: {e}, treating as N/A")
            return 0.0
        if math.isnan(value) or math.isinf(value):
            logger.warning(f"Ragas metric '{col}' is NaN or Inf, treating as N/A")
            return 0.0
        return value

    def _compute_scene_metrics(self, details: list[dict]) -> dict:
        """按场景分组计算指标

        Args:
            details: 详细结果列表

        Returns:
            场景指标字典
        """
        from collections import defaultdict

        scene_stats = defaultdict(lambda: {
            "count": 0,
            "correct": 0,
            "hits": 0,
            "total_rank": 0,
            "rank_count": 0,
        })

        for detail in details:
            scene = detail.get("scene", "unknown")
            scene_stats[scene]["count"] += 1

            if detail.get("answer_correct"):
                scene_stats[scene]["correct"] += 1
            if detail.get("retrieval_hit"):
                scene_stats[scene]["hits"] += 1
                scene_stats[scene]["total_rank"] += detail.get("retrieval_rank", 0)
                scene_stats[scene]["rank_count"] += 1

        # 计算指标
        scene_metrics = {}
        for scene, stats in scene_stats.items():
            count = stats["count"]
            scene_metrics[scene] = {
                "count": count,
                "accuracy": stats["correct"] / count if count > 0 else 0.0,
                "hit_rate": stats["hits"] / count if count > 0 else 0.0,
                "avg_rank": stats["total_rank"] / stats["rank_count"] if stats["rank_count"] > 0 else -1.0,
            }

        return scene_metrics

    def _compute_weighted_accuracy(self, details: list[dict], overall_accuracy: float) -> float:
        """按评估集实际场景分布加权计算准确率。

        只对 4 个核心场景（factual / extraction / policy_qa / comparison）计算，
        剔除 out_of_scope / unknown 等干扰项，然后以各场景在核心题目中的
        实际占比作为权重重新归一化。

        公式：
            w_i       = count_i / Σ count_j   （仅在核心场景 j 上求和）
            weighted  = Σ w_i × accuracy_i

        参考理想权重（不参与计算，仅供对比）:
            factual=35%, extraction=35%, policy_qa=20%, comparison=10%

        Args:
            details:          评估明细列表
            overall_accuracy: 全局准确率（当核心场景无数据时作为兜底返回）

        Returns:
            加权准确率
        """
        CORE_SCENES = {"factual", "extraction", "policy_qa", "comparison"}

        from collections import defaultdict
        stats: dict = defaultdict(lambda: {"count": 0, "correct": 0})

        for detail in details:
            scene = detail.get("scene", "unknown")
            if scene not in CORE_SCENES:
                continue
            stats[scene]["count"] += 1
            if detail.get("answer_correct"):
                stats[scene]["correct"] += 1

        total_core = sum(s["count"] for s in stats.values())
        if total_core == 0:
            # 无核心场景数据，退回全局准确率
            return overall_accuracy

        weighted = 0.0
        for scene, s in stats.items():
            w_i = s["count"] / total_core          # 实际占比
            acc_i = s["correct"] / s["count"]      # 该场景准确率
            weighted += w_i * acc_i

        return weighted

    def _get_config(self) -> dict:
        """获取当前配置快照"""
        from config import USE_RERANKER, RETRIEVAL_TOP_K, RERANK_TOP_K, RERANKER_MODEL
        cfg = {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "llm_provider": LLM_PROVIDER,
            "use_reranker": USE_RERANKER,
            "enable_query_router": ENABLE_QUERY_ROUTER,  # Phase 6: 记录 Router 开关
        }
        if USE_RERANKER:
            cfg["retrieval_top_k"] = RETRIEVAL_TOP_K   # 粗检索候选数
            cfg["rerank_top_k"] = RERANK_TOP_K         # 精排输出数（LLM 实际看到的）
            cfg["reranker_model"] = RERANKER_MODEL
            cfg["top_k"] = RERANK_TOP_K                # 向后兼容 report 读取
        else:
            cfg["top_k"] = TOP_K

        # Phase 6: 记录 Router 配置（如果启用）
        if ENABLE_QUERY_ROUTER:
            from config import (
                QUERY_ROUTER_ALLOW_AUTO_FILTER_FALLBACK,
                QUERY_ROUTER_ALLOW_EXPLICIT_FILTER_FALLBACK,
                QUERY_ROUTER_EMPTY_RESULT_THRESHOLD,
            )
            cfg["router_allow_auto_fallback"] = QUERY_ROUTER_ALLOW_AUTO_FILTER_FALLBACK
            cfg["router_allow_explicit_fallback"] = QUERY_ROUTER_ALLOW_EXPLICIT_FILTER_FALLBACK
            cfg["router_empty_threshold"] = QUERY_ROUTER_EMPTY_RESULT_THRESHOLD

        return cfg


def run_evaluation(
    dataset_path: str = "data/eval/manual_qa.json",
    run_name: str = "baseline",
    output_path: str = "data/eval/results.json",
    include_ragas: bool = True,
) -> EvalResult:
    """便捷函数：运行评估

    Args:
        dataset_path: 评估数据文件路径
        run_name: 运行名称
        output_path: 结果输出路径
        include_ragas: 是否计算 Ragas 指标

    Returns:
        评估结果
    """
    # 加载数据集
    dataset = EvalDataset()
    dataset.load_manual(dataset_path)

    if not dataset.get_all():
        raise ValueError(f"评估数据集为空: {dataset_path}")

    # 创建评估器
    evaluator = Evaluator()

    # 运行评估
    result = evaluator.run(dataset, run_name=run_name, include_ragas=include_ragas)

    # 保存结果
    result.save(output_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 评估工具")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/manual_qa.json",
        help="评估数据文件路径",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="运行名称",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/results.json",
        help="结果输出路径",
    )

    args = parser.parse_args()

    result = run_evaluation(
        dataset_path=args.dataset,
        run_name=args.name,
        output_path=args.output,
    )

    # 打印摘要
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total questions: {result.total_questions}")
    print(f"Overall Accuracy: {result.accuracy:.2%}")
    print(f"Retrieval Hit Rate: {result.retrieval_hit_rate:.2%}")
    print(f"Avg Retrieval Rank: {result.avg_retrieval_rank:.1f}")
    print(f"Faithfulness: {result.faithfulness:.2%}")
    print(f"Answer Relevancy: {result.answer_relevancy:.2%}")
    print(f"Context Precision: {result.context_precision:.2%}")
    print(f"Context Recall: {result.context_recall:.2%}")
    print("-" * 60)
    print(f"Manual Questions: {result.manual_count} | Accuracy: {result.manual_accuracy:.2%}")
    print(f"Synthetic Questions: {result.synthetic_count} | Accuracy: {result.synthetic_accuracy:.2%} | Weighted: {result.synthetic_weighted_accuracy:.2%}")
    print("=" * 60)
