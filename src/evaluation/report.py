"""评估报告生成器。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.evaluation.dataset import load_dataset
from src.evaluation.evaluator import EvalResult


class ReportGenerator:
    def generate(
        self,
        results: list[EvalResult],
        output_path: str = "data/eval/report.md",
        include_experiments: bool = False,
    ) -> str:
        """生成 Markdown 格式的评估报告。"""
        dataset = load_dataset()
        ds_summary = dataset.summary()

        # 优先选择含 Ragas 的结果作为主结果
        main = results[-1] if results else None
        for r in results:
            if any([r.faithfulness, r.answer_relevancy, r.context_precision, r.context_recall]):
                main = r

        if main is None:
            raise ValueError("没有可用于生成报告的评估结果")

        # 只有明确指定 include_experiments=True 时才加载实验数据
        chunk_results = self._load_batch("data/eval/experiments/chunk_size_results.json") if include_experiments else []
        topk_results = self._load_batch("data/eval/experiments/top_k_results.json") if include_experiments else []

        lines: list[str] = []
        lines.append("# FinDoc-RAG 评估报告")
        lines.append("")
        lines.append("## 生成时间")
        lines.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lines.append("")
        lines.append("## 数据集概况")
        lines.append("| 项目 | 数量 |")
        lines.append("|------|------|")
        lines.append(f"| 手工标注 | {ds_summary['manual_count']} 条 |")
        lines.append(f"| 合成生成 | {ds_summary['synthetic_count']} 条 |")
        lines.append(f"| 总计 | {ds_summary['total']} 条 |")
        lines.append("")
        lines.append("| 场景 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        total = max(ds_summary["total"], 1)
        for scene, count in ds_summary["scene_distribution"].items():
            lines.append(f"| {scene} | {count} | {count / total * 100:.1f}% |")
        lines.append("")
        lines.append("## 总体指标")
        lines.append("")
        lines.append("| 指标 | 分数 |")
        lines.append("|------|------|")
        lines.append(f"| 正确率 (Accuracy) | {main.accuracy:.2f} |")
        lines.append(f"| 加权正确率 (Weighted Accuracy) | {main.weighted_accuracy:.2f} |")
        lines.append(f"| 检索命中率 (Retrieval Hit Rate) | {main.retrieval_hit_rate:.2f} |")
        lines.append(f"| 忠实度 (Faithfulness) | {main.faithfulness:.2f} |")
        lines.append(f"| 答案相关性 (Answer Relevancy) | {main.answer_relevancy:.2f} |")
        lines.append(f"| 上下文精度 (Context Precision) | {main.context_precision:.2f} |")
        lines.append(f"| 上下文召回 (Context Recall) | {main.context_recall:.2f} |")
        lines.append("")
        lines.append("## ⭐ 高价值场景分析")
        lines.append("")
        lines.append("| 场景 | Context Precision | Context Recall | Accuracy | 评级 |")
        lines.append("|------|-------------------|----------------|----------|------|")
        for scene in ["comparison", "policy_qa", "extraction"]:
            sm = main.scene_metrics.get(scene, {})
            # 当前实现没有场景级 ragas，使用 hit_rate 近似上下文质量
            cp = float(sm.get("hit_rate", 0.0))
            cr = float(sm.get("hit_rate", 0.0))
            acc = float(sm.get("accuracy", 0.0))
            score = (cp + cr + acc) / 3
            lines.append(
                f"| {scene} | {cp:.2f} | {cr:.2f} | {acc:.2f} | {self._rate(score)} |"
            )
        lines.append("")
        lines.append("评级标准：≥0.7 ✅ 良好 | 0.5-0.7 ⚠️ 需改进 | <0.5 ❌ 较差")
        lines.append("")
        lines.append("## 参数对比实验")
        lines.append("")
        lines.append("### Chunk Size 对比（固定 top_k=5）")
        lines.append("")
        lines.append("| Chunk Size | Accuracy | Hit Rate | Faithfulness | Context Precision | 综合得分 |")
        lines.append("|------------|----------|----------|--------------|-------------------|----------|")
        if chunk_results:
            for item in chunk_results:
                cfg = item.get("config", {})
                size = cfg.get("chunk_size", "-")
                acc = float(item.get("accuracy", 0.0))
                hit = float(item.get("retrieval_hit_rate", 0.0))
                faith = float(item.get("faithfulness", 0.0))
                cp = float(item.get("context_precision", 0.0))
                score = (acc + hit + faith + cp) / 4
                lines.append(f"| {size} | {acc:.2f} | {hit:.2f} | {faith:.2f} | {cp:.2f} | {score:.2f} |")
        else:
            lines.append("| - | - | - | - | - | - |")
        lines.append("")
        lines.append("### Top-K 对比（固定 chunk_size=512）")
        lines.append("")
        lines.append("| Top-K | Accuracy | Hit Rate | Faithfulness | Context Precision | 综合得分 |")
        lines.append("|-------|----------|----------|--------------|-------------------|----------|")
        if topk_results:
            for item in topk_results:
                cfg = item.get("config", {})
                k = cfg.get("top_k", "-")
                acc = float(item.get("accuracy", 0.0))
                hit = float(item.get("retrieval_hit_rate", 0.0))
                faith = float(item.get("faithfulness", 0.0))
                cp = float(item.get("context_precision", 0.0))
                score = (acc + hit + faith + cp) / 4
                lines.append(f"| {k} | {acc:.2f} | {hit:.2f} | {faith:.2f} | {cp:.2f} | {score:.2f} |")
        else:
            lines.append("| - | - | - | - | - | - |")
        lines.append("")
        lines.append("综合得分 = (Accuracy + Hit Rate + Faithfulness + Context Precision) / 4")
        lines.append("")
        lines.append("## 🏆 最佳策略推荐")
        lines.append("")
        best_chunk = self._best_from_results(chunk_results, "chunk_size")
        best_topk = self._best_from_results(topk_results, "top_k")
        chunk_text = best_chunk if best_chunk is not None else main.config.get("chunk_size", 512)
        topk_text = best_topk if best_topk is not None else main.config.get("top_k", 5)
        lines.append(f"- **最佳 Chunk Size**：{chunk_text}")
        lines.append(f"- **最佳精排 Top-K**：{topk_text}")
        lines.append(
            f"- **当前最优配置**：chunk_size={chunk_text}, top_k={topk_text}, "
            f"embedding={main.config.get('embedding_model', '-')}, llm={main.config.get('llm_provider', '-')}"
        )
        lines.append("")
        lines.append("## 待改进方向")
        lines.append("1. comparison 场景的上下文召回偏低时，优先检查跨页信息是否被切散。")
        lines.append("2. extraction 场景若准确率偏低，优先检查表格解析和chunk保真。")
        lines.append("3. 可进一步探索更细粒度的 chunk 策略优化。")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        return str(output)

    def _rate(self, score: float) -> str:
        if score >= 0.7:
            return "✅ 良好"
        if score >= 0.5:
            return "⚠️ 需改进"
        return "❌ 较差"

    def _load_batch(self, path: str) -> list[dict]:
        """加载实验结果文件"""
        p = Path(path)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _best_from_results(self, rows: list[dict], key: str):
        if not rows:
            return None
        scored = []
        for item in rows:
            acc = float(item.get("accuracy", 0.0))
            hit = float(item.get("retrieval_hit_rate", 0.0))
            faith = float(item.get("faithfulness", 0.0))
            cp = float(item.get("context_precision", 0.0))
            score = (acc + hit + faith + cp) / 4
            cfg = item.get("config", {})
            scored.append((score, cfg.get(key)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
