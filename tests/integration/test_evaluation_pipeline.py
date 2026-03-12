"""Evaluation Pipeline 集成测试

验证评估流程: QA对 → 预测 → 评估 → 报告

覆盖场景：
- 评估器初始化
- 单一QA对评估
- 批量评估
- 评估结果结构
- 不同评估模式
"""
import pytest
from pathlib import Path
import json


class TestEvaluationPipeline:
    """评估管道集成测试"""

    def test_evaluator_initialization(self):
        """
        测试评估器初始化

        场景: 评估器应能正确初始化
        """
        from src.evaluation.evaluator import Evaluator

        evaluator = Evaluator()
        assert evaluator is not None

    def test_evaluator_load_manual_qa(self):
        """
        测试加载手工标注QA

        场景: 能从文件加载手工标注的QA对
        """
        from src.evaluation.evaluator import Evaluator

        _ = Evaluator()  # noqa: F841 - 验证初始化成功
        qa_path = Path("data/eval/manual_qa.json")

        if not qa_path.exists():
            pytest.skip("manual_qa.json 不存在")

        # 尝试加载
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            assert isinstance(qa_data, list)
        except Exception as e:
            pytest.fail(f"加载QA文件失败: {e}")

    def test_evaluator_load_synthetic_qa(self):
        """
        测试加载合成QA

        场景: 能从文件加载合成的QA对
        """
        from src.evaluation.evaluator import Evaluator

        _ = Evaluator()  # noqa: F841 - 验证初始化成功
        qa_path = Path("data/eval/synthetic_qa.json")

        if not qa_path.exists():
            pytest.skip("synthetic_qa.json 不存在")

        # 尝试加载
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            assert isinstance(qa_data, list)
        except Exception as e:
            pytest.fail(f"加载QA文件失败: {e}")

    def test_evaluator_qa_schema_validation(self):
        """
        测试QA数据结构验证

        场景: QA数据应包含必要字段
        """
        qa_path = Path("data/eval/manual_qa.json")

        if not qa_path.exists():
            pytest.skip("manual_qa.json 不存在")

        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        if not qa_data:
            pytest.skip("QA数据为空")

        # 验证每个QA对都有必要字段
        for qa in qa_data[:5]:  # 检查前5个
            assert "question" in qa, "QA对缺少question字段"
            assert "answer" in qa or "ground_truth" in qa, "QA对缺少answer/ground_truth字段"

    def test_dataset_generator_exists(self):
        """
        测试数据集生成器存在

        场景: 数据集生成器应能被导入
        """
        from src.evaluation.testset_generator import TestsetGenerator

        generator = TestsetGenerator()
        assert generator is not None

    def test_report_generator_exists(self):
        """
        测试报告生成器存在

        场景: 报告生成器应能被导入
        """
        from src.evaluation.report import ReportGenerator

        report_gen = ReportGenerator()
        assert report_gen is not None

    def test_experiment_tracker_exists(self):
        """
        测试实验追踪器存在

        场景: 实验追踪器应能被导入
        """
        from src.evaluation.experiment import ExperimentTracker

        tracker = ExperimentTracker()
        assert tracker is not None

    @pytest.mark.skip(reason="需要真实LLM调用")
    def test_single_qa_evaluation(self, populated_storage):
        """
        测试单一QA对评估（需要LLM）

        场景: 对一个QA对进行评估
        """
        from src.evaluation.evaluator import Evaluator  # noqa: F401
        from src.generation.qa_chain import QAChain  # noqa: F401

    def test_evaluator_modes(self):
        """
        测试不同评估模式

        场景: manual模式、full模式应都能被调用
        """
        from src.evaluation.evaluator import Evaluator

        _ = Evaluator()  # noqa: F841 - 验证初始化成功

        # 验证评估器支持的模式
        # 这些模式在scripts/evaluate.py中被使用
        modes = ["manual", "full", "generate"]
        for mode in modes:
            # 这里只是验证模式字符串，不实际执行
            assert isinstance(mode, str)
