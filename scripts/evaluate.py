#!/usr/bin/env python
"""FinDoc-RAG 评估工具。"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPERIMENT_CHUNK_SIZES, EXPERIMENT_TOP_KS
from src.evaluation.dataset import EvalDataset, load_dataset
from src.evaluation.evaluator import Evaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _print_detail_rows(details: list[dict]) -> None:
    for i, item in enumerate(details, 1):
        print("-" * 80)
        print(f"[{i}] 问题: {item.get('question', '')}")
        print(f"回答: {item.get('model_answer', '')}")
        print(f"答案是否正确: {item.get('answer_correct', False)}")
        print(f"检索是否命中: {item.get('retrieval_hit', False)}")
        print(f"命中排名: {item.get('retrieval_rank', -1)}")


def _print_summary(result, with_ragas: bool) -> None:
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"Run: {result.run_name}")
    print(f"Total: {result.total_questions}")
    print(f"accuracy: {result.accuracy:.4f}")
    print(f"weighted_accuracy: {result.weighted_accuracy:.4f}")
    print(f"retrieval_hit_rate: {result.retrieval_hit_rate:.4f}")
    print(f"avg_retrieval_rank: {result.avg_retrieval_rank:.4f}")
    if with_ragas:
        print(f"faithfulness: {result.faithfulness:.4f}")
        print(f"answer_relevancy: {result.answer_relevancy:.4f}")
        print(f"context_precision: {result.context_precision:.4f}")
        print(f"context_recall: {result.context_recall:.4f}")
        print("scene_metrics:")
        for scene, metrics in result.scene_metrics.items():
            print(
                f"  - {scene}: accuracy={metrics.get('accuracy', 0.0):.4f}, "
                f"hit_rate={metrics.get('hit_rate', 0.0):.4f}, avg_rank={metrics.get('avg_rank', -1.0):.4f}"
            )
    print("=" * 60)


def _run_manual(output_dir: Path, chunk_size: int = None, top_k: int = None) -> None:
    """运行 manual 数据集的评估"""
    from src.retrieval import Retriever, RerankRetriever
    from src.storage import DocStore, VectorStore
    from src.generation.qa_chain import QAChain
    from config import EMBEDDING_MODEL, USE_RERANKER

    dataset = EvalDataset()
    dataset.load_manual("data/eval/manual_qa.json")

    # 如果指定了 chunk_size，使用对应的实验数据库
    if chunk_size:
        exp_dir = output_dir / "experiments" / f"chunk_size_{chunk_size}"
        chroma_dir = exp_dir / "chroma_db"
        sqlite_path = exp_dir / "doc_store.db"

        if not chroma_dir.exists():
            raise ValueError(f"chunk_size={chunk_size} 的实验数据不存在，请先运行 experiment")

        doc_store = DocStore(db_path=str(sqlite_path))
        vector_store = VectorStore(
            persist_dir=str(chroma_dir),
            embedding_model=f"{EMBEDDING_MODEL}_chunk_{chunk_size}",
        )
        base_retriever = Retriever(vector_store=vector_store, doc_store=doc_store)
    else:
        base_retriever = Retriever()

    # A/B 切换
    if USE_RERANKER and top_k is None:
        retriever = RerankRetriever(retriever=base_retriever)
        logger.info("检索模式: RerankRetriever（粗检索→精排）")
    else:
        retriever = base_retriever
        logger.info("检索模式: Retriever（Baseline）")

    if top_k:
        from src.evaluation.experiment import _FixedTopKRetriever
        retriever = _FixedTopKRetriever(base_retriever, top_k=top_k)

    qa_chain = QAChain(retriever=retriever)
    evaluator = Evaluator(qa_chain=qa_chain)
    run_name = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = evaluator.run(dataset=dataset, run_name=run_name, include_ragas=True)
    result_path = output_dir / "manual_eval_result.json"
    result.save(str(result_path))
    history_dir = output_dir / "history"
    history_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result.save(str(history_dir / f"manual_{ts}.json"))
    _print_detail_rows(result.details)
    _print_summary(result, with_ragas=True)


def _run_synthetic(output_dir: Path, chunk_size: int = None, top_k: int = None) -> None:
    """运行 synthetic 数据集的评估（单独）"""
    from src.retrieval import Retriever, RerankRetriever
    from src.storage import DocStore, VectorStore
    from src.generation.qa_chain import QAChain
    from config import EMBEDDING_MODEL, USE_RERANKER

    # 只加载 synthetic 数据
    dataset = EvalDataset()
    dataset.load_synthetic("data/eval/synthetic_qa.json")

    # 如果指定了 chunk_size，使用对应的实验数据库
    if chunk_size:
        exp_dir = output_dir / "experiments" / f"chunk_size_{chunk_size}"
        chroma_dir = exp_dir / "chroma_db"
        sqlite_path = exp_dir / "doc_store.db"

        if not chroma_dir.exists():
            raise ValueError(f"chunk_size={chunk_size} 的实验数据不存在，请先运行 experiment")

        doc_store = DocStore(db_path=str(sqlite_path))
        vector_store = VectorStore(
            persist_dir=str(chroma_dir),
            embedding_model=f"{EMBEDDING_MODEL}_chunk_{chunk_size}",
        )
        base_retriever = Retriever(vector_store=vector_store, doc_store=doc_store)
    else:
        base_retriever = Retriever()

    # A/B 切换
    if USE_RERANKER and top_k is None:
        retriever = RerankRetriever(retriever=base_retriever)
        logger.info("检索模式: RerankRetriever（粗检索→精排）")
    else:
        retriever = base_retriever
        logger.info("检索模式: Retriever（Baseline）")

    if top_k:
        from src.evaluation.experiment import _FixedTopKRetriever
        retriever = _FixedTopKRetriever(base_retriever, top_k=top_k)

    qa_chain = QAChain(retriever=retriever)
    evaluator = Evaluator(qa_chain=qa_chain)
    run_name = f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = evaluator.run(dataset=dataset, run_name=run_name, include_ragas=True)
    result_path = output_dir / "synthetic_eval_result.json"
    result.save(str(result_path))
    history_dir = output_dir / "history"
    history_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result.save(str(history_dir / f"synthetic_{ts}.json"))
    _print_summary(result, with_ragas=True)


def _run_full(output_dir: Path, chunk_size: int = None, top_k: int = None) -> None:
    from src.retrieval import Retriever, RerankRetriever
    from src.storage import DocStore, VectorStore
    from src.generation.qa_chain import QAChain
    from config import EMBEDDING_MODEL, USE_RERANKER

    dataset = load_dataset(
        manual_path="data/eval/manual_qa.json",
        synthetic_path="data/eval/synthetic_qa.json",
    )

    # 如果指定了 chunk_size，使用对应的实验数据库
    if chunk_size:
        exp_dir = output_dir / "experiments" / f"chunk_size_{chunk_size}"
        chroma_dir = exp_dir / "chroma_db"
        sqlite_path = exp_dir / "doc_store.db"

        if not chroma_dir.exists():
            raise ValueError(f"chunk_size={chunk_size} 的实验数据不存在，请先运行 experiment")

        doc_store = DocStore(db_path=str(sqlite_path))
        vector_store = VectorStore(
            persist_dir=str(chroma_dir),
            embedding_model=f"{EMBEDDING_MODEL}_chunk_{chunk_size}",
        )
        base_retriever = Retriever(vector_store=vector_store, doc_store=doc_store)
    else:
        base_retriever = Retriever()

    # A/B 切换：USE_RERANKER=True 走二阶段精排，False 退回 Baseline
    if USE_RERANKER and top_k is None:
        retriever = RerankRetriever(retriever=base_retriever)
        logger.info("检索模式: RerankRetriever（粗检索→精排）")
    else:
        retriever = base_retriever
        logger.info("检索模式: Retriever（Baseline）")

    # 如果指定了 top_k，使用 FixedTopKRetriever（仅 Baseline 模式下有意义）
    if top_k:
        from src.evaluation.experiment import _FixedTopKRetriever
        retriever = _FixedTopKRetriever(base_retriever, top_k=top_k)

    qa_chain = QAChain(retriever=retriever)
    evaluator = Evaluator(qa_chain=qa_chain)
    run_name = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = evaluator.run(dataset=dataset, run_name=run_name, include_ragas=True)
    result_path = output_dir / "full_eval_result.json"
    result.save(str(result_path))
    history_dir = output_dir / "history"
    history_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result.save(str(history_dir / f"full_{ts}.json"))
    _print_summary(result, with_ragas=True)


def _run_generate(num: int) -> None:
    from src.evaluation.testset_generator import generate_synthetic_qa
    from src.generation.llm_service import get_llm_service

    llm_service = get_llm_service()
    # --num 为目标生成题目数；采样 chunk 数 = num × 2，上限 200（由 generate_synthetic_qa 内部处理）
    qa_pairs = generate_synthetic_qa(
        llm_service=llm_service,
        num_questions=num,
        output_path="data/eval/synthetic_qa.json",
    )
    print(f"Generated {len(qa_pairs)} synthetic QA pairs")


def _run_experiment(variable: str) -> None:
    from src.evaluation.experiment import ExperimentRunner

    runner = ExperimentRunner()
    dataset = load_dataset("data/eval/manual_qa.json", "data/eval/synthetic_qa.json")
    if variable == "chunk_size":
        results = runner.run_chunk_size_experiment(EXPERIMENT_CHUNK_SIZES, dataset)
    else:
        results = runner.run_topk_experiment(EXPERIMENT_TOP_KS, dataset)
    print(runner.compare(results))


def _run_report(output_dir: Path, include_experiments: bool = False) -> None:
    """生成评估报告

    Args:
        output_dir: 输出目录
        include_experiments: 是否包含参数实验数据（默认 False）
    """
    from src.evaluation.report import ReportGenerator

    dataset = load_dataset("data/eval/manual_qa.json", "data/eval/synthetic_qa.json")
    if dataset.summary()["total"] == 0:
        raise ValueError("数据集为空，无法生成报告")

    results = []
    manual_path = output_dir / "manual_eval_result.json"
    full_path = output_dir / "full_eval_result.json"

    if manual_path.exists():
        from src.evaluation.evaluator import EvalResult
        import json
        manual = json.loads(manual_path.read_text(encoding="utf-8"))
        results.append(EvalResult(**manual))

    if full_path.exists():
        from src.evaluation.evaluator import EvalResult
        import json
        full = json.loads(full_path.read_text(encoding="utf-8"))
        results.append(EvalResult(**full))

    if not results:
        raise ValueError("未找到评估结果，请先运行 manual/full")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamped_path = str(reports_dir / f"report_{ts}.md")
    report = ReportGenerator().generate(
        results,
        output_path=timestamped_path,
        include_experiments=include_experiments
    )
    import shutil
    shutil.copy(timestamped_path, str(output_dir / "report.md"))
    print(f"Report generated: {report}")
    if not include_experiments:
        print("Note: 参数对比实验数据未包含（使用 --include-experiments 包含）")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinDoc-RAG 评估工具")
    parser.add_argument(
        "--mode",
        choices=["full", "manual", "synthetic", "generate", "experiment", "report"],
        required=True,
    )
    parser.add_argument("--num", type=int, default=50, help="目标生成题目数（采样 chunk 数 = num × 2，上限 200）")
    parser.add_argument("--variable", choices=["chunk_size", "top_k"], help="实验变量")
    parser.add_argument("--output-dir", type=str, default="data/eval", help="结果输出目录")
    parser.add_argument("--chunk-size", type=int, help="指定 chunk_size（用于 full/manual/synthetic 模式）")
    parser.add_argument("--top-k", type=int, help="指定 top_k（用于 full/manual/synthetic 模式）")
    parser.add_argument("--include-experiments", action="store_true", help="报告包含参数对比实验数据")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "manual":
        _run_manual(output_dir, chunk_size=args.chunk_size, top_k=args.top_k)
    elif args.mode == "synthetic":
        _run_synthetic(output_dir, chunk_size=args.chunk_size, top_k=args.top_k)
    elif args.mode == "full":
        _run_full(output_dir, chunk_size=args.chunk_size, top_k=args.top_k)
    elif args.mode == "generate":
        _run_generate(args.num)
    elif args.mode == "experiment":
        if not args.variable:
            raise ValueError("--mode experiment 需要 --variable")
        _run_experiment(args.variable)
    elif args.mode == "report":
        _run_report(output_dir, include_experiments=args.include_experiments)


if __name__ == "__main__":
    main()
