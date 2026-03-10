"""评估模块初始化"""
from src.evaluation.dataset import EvalDataset
from src.evaluation.evaluator import Evaluator, EvalResult, run_evaluation

try:
    from src.evaluation.experiment import ExperimentRunner
except Exception:
    ExperimentRunner = None

try:
    from src.evaluation.report import ReportGenerator
except Exception:
    ReportGenerator = None

try:
    from src.evaluation.testset_generator import SyntheticTestsetGenerator
except Exception:
    SyntheticTestsetGenerator = None

__all__ = [
    "SyntheticTestsetGenerator",
    "EvalDataset",
    "Evaluator",
    "EvalResult",
    "run_evaluation",
    "ExperimentRunner",
    "ReportGenerator",
]
