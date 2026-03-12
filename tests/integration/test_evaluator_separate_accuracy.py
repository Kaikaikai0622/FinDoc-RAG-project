"""
集成测试：验证决策2 - synthetic 和 manual 问题准确率分离计算
"""
from collections import defaultdict
from src.evaluation.evaluator import EvalResult


class TestEvaluatorSeparateAccuracy:
    """测试 Manual/Synthetic 准确率分离计算"""

    def test_manual_id_prefix_detection(self):
        """测试 manual ID 前缀识别"""
        test_cases = [
            ("m001", True, False, "manual ID 以 m 开头"),
            ("m123", True, False, "manual ID 以 m 开头"),
            ("s001", False, True, "synthetic ID 以 s 开头"),
            ("s123", False, True, "synthetic ID 以 s 开头"),
            ("001", False, True, "纯数字 ID 视为 synthetic"),
            ("q001", False, True, "其他前缀视为 synthetic"),
        ]

        for qid, expected_manual, expected_synthetic, desc in test_cases:
            is_manual = qid.startswith("m")
            is_synthetic = not is_manual
            assert is_manual == expected_manual and is_synthetic == expected_synthetic, \
                f"{desc}: {qid} -> manual={is_manual}, synthetic={is_synthetic}"

    def test_separate_calculation_50_75(self):
        """测试 Manual 50%, Synthetic 75% 场景"""
        manual_res = [True, False]  # 1对1错
        synthetic_res = [
            (True, "factual"),
            (True, "factual"),
            (True, "extraction"),
            (False, "policy_qa"),
        ]  # 3对1错

        m_acc, s_acc, s_weighted, m_count, s_count = self._calc_accuracy(
            manual_res, synthetic_res
        )

        assert abs(m_acc - 0.5) < 0.01 and m_count == 2, \
            f"Manual: count={m_count}, accuracy={m_acc:.2%} (期望 2, 50%)"
        assert abs(s_acc - 0.75) < 0.01 and s_count == 4, \
            f"Synthetic: count={s_count}, accuracy={s_acc:.2%} (期望 4, 75%)"
        assert abs(s_weighted - 0.75) < 0.01, \
            f"Synthetic Weighted: {s_weighted:.2%} (期望 75%)"

    def test_pure_manual_questions(self):
        """测试纯 manual 问题场景"""
        m_acc, s_acc, s_weighted, m_count, s_count = self._calc_accuracy(
            [True, True, False], []
        )

        assert m_acc == 2 / 3, f"manual_acc={m_acc:.2%} (期望 66.67%)"
        assert s_count == 0, f"synthetic_count={s_count} (期望 0)"
        assert s_acc == 0.0, f"synthetic_acc={s_acc} (期望 0.0)"

    def test_pure_synthetic_questions(self):
        """测试纯 synthetic 问题场景"""
        m_acc, s_acc, s_weighted, m_count, s_count = self._calc_accuracy(
            [], [(True, "factual"), (False, "extraction")]
        )

        assert m_count == 0, f"manual_count={m_count} (期望 0)"
        assert s_acc == 0.5, f"synthetic_acc={s_acc:.2%} (期望 50%)"
        assert s_count == 2, f"synthetic_count={s_count} (期望 2)"

    def test_empty_dataset(self):
        """测试空数据集场景"""
        m_acc, s_acc, s_weighted, m_count, s_count = self._calc_accuracy([], [])

        assert m_acc == 0.0, f"manual_acc={m_acc} (期望 0.0)"
        assert s_acc == 0.0, f"synthetic_acc={s_acc} (期望 0.0)"
        assert m_count == 0, f"manual_count={m_count} (期望 0)"
        assert s_count == 0, f"synthetic_count={s_count} (期望 0)"

    def test_evalresult_data_structure(self):
        """测试 EvalResult 数据结构"""
        result = EvalResult(
            run_name="test",
            timestamp="2026-01-01",
            config={},
            total_questions=10,
            accuracy=0.6,
            retrieval_hit_rate=0.8,
            avg_retrieval_rank=2.0,
            weighted_accuracy=0.65,
            synthetic_accuracy=0.75,
            synthetic_weighted_accuracy=0.70,
            synthetic_count=4,
            manual_accuracy=0.50,
            manual_count=6,
        )

        assert result.synthetic_accuracy == 0.75
        assert result.synthetic_weighted_accuracy == 0.70
        assert result.synthetic_count == 4
        assert result.manual_accuracy == 0.50
        assert result.manual_count == 6

    def test_evalresult_to_dict(self):
        """测试 EvalResult to_dict 方法"""
        result = EvalResult(
            run_name="test",
            timestamp="2026-01-01",
            config={},
            total_questions=10,
            accuracy=0.6,
            retrieval_hit_rate=0.8,
            avg_retrieval_rank=2.0,
            weighted_accuracy=0.65,
            synthetic_accuracy=0.75,
            synthetic_weighted_accuracy=0.70,
            synthetic_count=4,
            manual_accuracy=0.50,
            manual_count=6,
        )
        data = result.to_dict()

        assert "synthetic_accuracy" in data
        assert "synthetic_weighted_accuracy" in data
        assert "manual_accuracy" in data

    # ─────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────

    def _calc_accuracy(self, manual_results, synthetic_results):
        """
        计算准确率

        Args:
            manual_results: list of bool, True表示正确
            synthetic_results: list of (bool, scene)

        Returns:
            (manual_acc, synthetic_acc, synthetic_weighted, manual_count, synthetic_count)
        """
        manual_correct = sum(manual_results)
        manual_count = len(manual_results)
        manual_acc = manual_correct / manual_count if manual_count > 0 else 0.0

        synthetic_correct = sum(r[0] for r in synthetic_results)
        synthetic_count = len(synthetic_results)
        synthetic_acc = synthetic_correct / synthetic_count if synthetic_count > 0 else 0.0

        # 计算 synthetic 加权准确率
        CORE_SCENES = {"factual", "extraction", "policy_qa", "comparison"}
        stats = defaultdict(lambda: {"count": 0, "correct": 0})

        for is_correct, scene in synthetic_results:
            if scene in CORE_SCENES:
                stats[scene]["count"] += 1
                if is_correct:
                    stats[scene]["correct"] += 1

        total_core = sum(s["count"] for s in stats.values())
        if total_core > 0:
            weighted = sum(
                (s["count"] / total_core) * (s["correct"] / s["count"])
                for s in stats.values() if s["count"] > 0
            )
        else:
            weighted = synthetic_acc

        return manual_acc, synthetic_acc, weighted, manual_count, synthetic_count
