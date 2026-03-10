"""参数对比实验模块。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import (
    DATA_RAW_DIR,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EXPERIMENT_CHUNK_SIZES,
    EXPERIMENT_TOP_KS,
)
from src.evaluation.dataset import EvalDataset, load_dataset
from src.evaluation.evaluator import Evaluator, EvalResult
from src.generation.qa_chain import QAChain
from src.retrieval import Retriever
from src.storage import DocStore, VectorStore


class _FixedTopKRetriever:
    """强制使用固定 top_k 的检索器包装器。"""

    def __init__(self, retriever: Retriever, top_k: int):
        self._retriever = retriever
        self._top_k = top_k

    def search(self, query: str, top_k: int | None = None):
        return self._retriever.search(query=query, top_k=self._top_k)


class ExperimentRunner:
    """参数对比实验执行器。"""

    def __init__(self) -> None:
        self._exp_dir = Path("data/eval/experiments")
        self._exp_dir.mkdir(parents=True, exist_ok=True)

    def run_chunk_size_experiment(
        self,
        sizes: list[int] | None = None,
        dataset: EvalDataset | None = None,
    ) -> list[EvalResult]:
        sizes = sizes or EXPERIMENT_CHUNK_SIZES
        dataset = dataset or load_dataset()
        results: list[EvalResult] = []

        pdf_files = sorted(Path(DATA_RAW_DIR).glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"未在 {DATA_RAW_DIR} 找到 PDF 文件")

        try:
            from src.ingestion.chunker import Chunker
            from src.ingestion.pipeline import IngestionPipeline
        except Exception as e:
            raise RuntimeError("chunk_size 实验依赖 ingestion/llama_index 组件，请先安装对应依赖") from e

        for size in sizes:
            run_name = f"chunk_size_{size}"
            run_dir = self._exp_dir / run_name
            chroma_dir = run_dir / "chroma_db"
            sqlite_path = run_dir / "doc_store.db"

            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            doc_store = DocStore(db_path=str(sqlite_path))
            vector_store = VectorStore(
                persist_dir=str(chroma_dir),
                embedding_model=f"{EMBEDDING_MODEL}_chunk_{size}",
            )
            pipeline = IngestionPipeline(doc_store=doc_store, vector_store=vector_store)
            pipeline.chunker = Chunker(chunk_size=size, chunk_overlap=CHUNK_OVERLAP)

            for pdf in pdf_files:
                pipeline.run(str(pdf))

            retriever = Retriever(vector_store=vector_store, doc_store=doc_store)
            evaluator = Evaluator(qa_chain=QAChain(retriever=retriever))
            result = evaluator.run(dataset=dataset, run_name=run_name, include_ragas=True)
            result.config["chunk_size"] = size
            result.save(str(run_dir / "result.json"))
            results.append(result)

        self._save_batch("chunk_size_results.json", results)
        return results

    def run_topk_experiment(
        self,
        ks: list[int] | None = None,
        dataset: EvalDataset | None = None,
    ) -> list[EvalResult]:
        # TODO: USE_RERANKER=True 时，此处 EXPERIMENT_TOP_KS 的语义是"粗检索候选数"
        # 而非"精排最终输出数"。Reranker 稳定后请使用 run_rerank_topk_experiment()
        # 扫描 RERANK_TOP_K 变量，与本方法形成语义隔离。
        ks = ks or EXPERIMENT_TOP_KS
        dataset = dataset or load_dataset()
        results: list[EvalResult] = []

        # 复用 chunk_size=512 的数据库
        chunk_512_dir = self._exp_dir / "chunk_size_512"
        chroma_dir = chunk_512_dir / "chroma_db"
        sqlite_path = chunk_512_dir / "doc_store.db"
        doc_store = DocStore(db_path=str(sqlite_path))
        vector_store = VectorStore(
            persist_dir=str(chroma_dir),
            embedding_model=f"{EMBEDDING_MODEL}_chunk_512",
        )
        base_retriever = Retriever(vector_store=vector_store, doc_store=doc_store)

        for k in ks:
            run_name = f"top_k_{k}"
            retriever = _FixedTopKRetriever(base_retriever, top_k=k)
            evaluator = Evaluator(qa_chain=QAChain(retriever=retriever))
            result = evaluator.run(dataset=dataset, run_name=run_name, include_ragas=True)
            result.config["top_k"] = k
            results.append(result)

        self._save_batch("top_k_results.json", results)
        return results

    def compare(self, results: list[EvalResult]) -> str:
        """按 (accuracy + retrieval_hit_rate) / 2 排序的 Markdown 表格。"""
        rows = sorted(
            results,
            key=lambda r: (r.accuracy + r.retrieval_hit_rate) / 2,
            reverse=True,
        )
        lines = [
            "| Run | Accuracy | Hit Rate | Faithfulness | Context Precision | 综合得分 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            score = (r.accuracy + r.retrieval_hit_rate) / 2
            lines.append(
                f"| {r.run_name} | {r.accuracy:.2f} | {r.retrieval_hit_rate:.2f} | "
                f"{r.faithfulness:.2f} | {r.context_precision:.2f} | {score:.2f} |"
            )
        return "\n".join(lines)

    def run_rerank_topk_experiment(
        self,
        rerank_ks: list[int] | None = None,
        dataset: EvalDataset | None = None,
    ) -> list[EvalResult]:
        """扫描 RERANK_TOP_K（精排最终输出数）的实验。

        与 run_topk_experiment 语义隔离：
        - run_topk_experiment     → 变量是粗检索候选数（Baseline）
        - run_rerank_topk_experiment → 变量是精排输出数（Reranker 开启后）

        TODO: Reranker 稳定后实现此方法，替代 run_topk_experiment 做精排参数调优。
        """
        raise NotImplementedError(
            "run_rerank_topk_experiment 尚未实现，"
            "请在 Reranker 评估稳定后补充实现。"
        )

    def _save_batch(self, filename: str, results: list[EvalResult]) -> None:
        data = json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)
        path = self._exp_dir / filename
        path.write_text(data, encoding="utf-8")
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = Path(filename).stem
        history_dir = self._exp_dir / "history"
        history_dir.mkdir(exist_ok=True)
        (history_dir / f"{stem}_{ts}.json").write_text(data, encoding="utf-8")
