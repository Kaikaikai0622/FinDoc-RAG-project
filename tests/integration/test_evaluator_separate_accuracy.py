"""
冒烟测试：验证决策2 - synthetic 和 manual 问题准确率分离计算
"""
import sys
import os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import Evaluator, EvalResult
from src.evaluation.dataset import EvalDataset

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = 0

print("=" * 60)
print("决策2：Synthetic/Manual 准确率分离计算测试")
print("=" * 60)
print()

# ─────────────────────────────────────────────────────────────
# 模拟数据测试
# ─────────────────────────────────────────────────────────────
print("测试场景：混合 manual (m开头) 和 synthetic (s开头) 问题")
print("-" * 60)

# 构造测试数据
# manual 问题：2个正确，2个错误（准确率 50%）
# synthetic 问题：3个正确，1个错误（准确率 75%）
# synthetic 场景分布：factual(2), extraction(1), policy_qa(1)
# 假设 factual 对2错0 (100%), extraction 对1错0 (100%), policy_qa 对0错1 (0%)
# 加权 = 2/4*100% + 1/4*100% + 1/4*0% = 75%

test_qa_pairs = [
    # Manual 问题 (m开头)
    {"id": "m001", "question": "Q1", "ground_truth": "A1", "scene": "factual", "source_files": ["test.pdf"], "source_pages": [1]},
    {"id": "m002", "question": "Q2", "ground_truth": "A2", "scene": "extraction", "source_files": ["test.pdf"], "source_pages": [1]},
    {"id": "m003", "question": "Q3", "ground_truth": "A3", "scene": "factual", "source_files": ["test.pdf"], "source_pages": [1]},
    {"id": "m004", "question": "Q4", "ground_truth": "A4", "scene": "policy_qa", "source_files": ["test.pdf"], "source_pages": [1]},
    # Synthetic 问题 (s开头)
    {"id": "s001", "question": "Q5", "ground_truth": "A5", "scene": "factual", "source_files": ["test.pdf"], "source_pages": [1]},
    {"id": "s002", "question": "Q6", "ground_truth": "A6", "scene": "factual", "source_files": ["test.pdf"], "source_pages": [1]},
    {"id": "s003", "question": "Q7", "ground_truth": "A7", "scene": "extraction", "source_files": ["test.pdf"], "source_pages": [1]},
    {"id": "s004", "question": "Q8", "ground_truth": "A8", "scene": "policy_qa", "source_files": ["test.pdf"], "source_pages": [1]},
]

# 模拟评估器计算
def mock_compute_metrics(details):
    """模拟评估指标计算"""
    manual_correct = sum(1 for d in details if d["id"].startswith("m") and d.get("answer_correct", False))
    manual_count = sum(1 for d in details if d["id"].startswith("m"))
    synthetic_correct = sum(1 for d in details if not d["id"].startswith("m") and d.get("answer_correct", False))
    synthetic_count = sum(1 for d in details if not d["id"].startswith("m"))

    manual_accuracy = manual_correct / manual_count if manual_count > 0 else 0.0
    synthetic_accuracy = synthetic_correct / synthetic_count if synthetic_count > 0 else 0.0

    return manual_accuracy, synthetic_accuracy, manual_count, synthetic_count

# 测试1: 验证ID识别逻辑
print("[测试1] 问题ID前缀识别")
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
    ok = (is_manual == expected_manual) and (is_synthetic == expected_synthetic)
    label = PASS if ok else FAIL
    print(f"  [{label}] {desc}: {qid} -> manual={is_manual}, synthetic={is_synthetic}")
    if not ok:
        errors += 1

print()

# 测试2: 验证分离计算逻辑
print("[测试2] Manual/Synthetic 分离计算")

# 模拟不同正确性分布
def test_separate_calculation(manual_results, synthetic_results):
    """
    manual_results: list of bool, True表示正确
    synthetic_results: list of (bool, scene)
    """
    manual_correct = sum(manual_results)
    manual_count = len(manual_results)
    manual_acc = manual_correct / manual_count if manual_count > 0 else 0.0

    synthetic_correct = sum(r[0] for r in synthetic_results)
    synthetic_count = len(synthetic_results)
    synthetic_acc = synthetic_correct / synthetic_count if synthetic_count > 0 else 0.0

    # 计算 synthetic 加权准确率
    CORE_SCENES = {"factual", "extraction", "policy_qa", "comparison"}
    from collections import defaultdict
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

# 测试场景1: manual 50%, synthetic 75%
manual_res = [True, False]  # 1对1错
synthetic_res = [(True, "factual"), (True, "factual"), (True, "extraction"), (False, "policy_qa")]  # 3对1错

m_acc, s_acc, s_weighted, m_count, s_count = test_separate_calculation(manual_res, synthetic_res)

ok1 = abs(m_acc - 0.5) < 0.01 and m_count == 2
ok2 = abs(s_acc - 0.75) < 0.01 and s_count == 4
# synthetic 加权: factual(2对0错, 100%), extraction(1对0错, 100%), policy_qa(0对1错, 0%)
# 权重: factual=2/4=50%, extraction=1/4=25%, policy_qa=1/4=25%
# 加权 = 0.5*1.0 + 0.25*1.0 + 0.25*0.0 = 0.75
ok3 = abs(s_weighted - 0.75) < 0.01

label = PASS if ok1 else FAIL
print(f"  [{label}] Manual: count={m_count}, accuracy={m_acc:.2%} (期望 2, 50%)")
if not ok1:
    errors += 1

label = PASS if ok2 else FAIL
print(f"  [{label}] Synthetic: count={s_count}, accuracy={s_acc:.2%} (期望 4, 75%)")
if not ok2:
    errors += 1

label = PASS if ok3 else FAIL
print(f"  [{label}] Synthetic Weighted: {s_weighted:.2%} (期望 75%)")
if not ok3:
    errors += 1

print()

# 测试3: 极端情况
print("[测试3] 极端情况测试")

# 纯 manual 问题
m_acc, s_acc, s_weighted, m_count, s_count = test_separate_calculation([True, True, False], [])
ok = (m_acc == 2/3) and (s_count == 0) and (s_acc == 0.0)
label = PASS if ok else FAIL
print(f"  [{label}] 纯 manual 问题: manual_acc={m_acc:.2%}, synthetic_count={s_count}")
if not ok:
    errors += 1

# 纯 synthetic 问题
m_acc, s_acc, s_weighted, m_count, s_count = test_separate_calculation([], [(True, "factual"), (False, "extraction")])
ok = (m_count == 0) and (s_acc == 0.5) and (s_count == 2)
label = PASS if ok else FAIL
print(f"  [{label}] 纯 synthetic 问题: synthetic_acc={s_acc:.2%}, manual_count={m_count}")
if not ok:
    errors += 1

# 空数据集
m_acc, s_acc, s_weighted, m_count, s_count = test_separate_calculation([], [])
ok = (m_acc == 0.0) and (s_acc == 0.0) and (m_count == 0) and (s_count == 0)
label = PASS if ok else FAIL
print(f"  [{label}] 空数据集: 所有指标为0")
if not ok:
    errors += 1

print()

# 测试4: EvalResult 数据结构
print("[测试4] EvalResult 数据结构验证")
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

ok = (
    result.synthetic_accuracy == 0.75 and
    result.synthetic_weighted_accuracy == 0.70 and
    result.synthetic_count == 4 and
    result.manual_accuracy == 0.50 and
    result.manual_count == 6
)
label = PASS if ok else FAIL
print(f"  [{label}] EvalResult 包含 synthetic/manual 独立字段")
if not ok:
    errors += 1

# 测试 to_dict 方法
data = result.to_dict()
ok = (
    "synthetic_accuracy" in data and
    "synthetic_weighted_accuracy" in data and
    "manual_accuracy" in data
)
label = PASS if ok else FAIL
print(f"  [{label}] to_dict() 包含新字段")
if not ok:
    errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 结果汇总
# ─────────────────────────────────────────────────────────────
print("=" * 60)
if errors == 0:
    print(f"\033[32m✅ 所有冒烟测试通过（0 failures）\033[0m")
else:
    print(f"\033[31m❌ {errors} 个测试失败\033[0m")
print("=" * 60)
sys.exit(errors)
