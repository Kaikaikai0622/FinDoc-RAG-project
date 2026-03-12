"""
冒烟测试：验证 comparison 类型问题纳入两步生成策略
测试 _TWO_STEP_KEYWORDS 正则是否正确匹配 comparison 类问题
"""
import sys
import os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.qa_chain import _should_use_two_step

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = 0

# ─────────────────────────────────────────────────────────────
# Comparison 类型问题测试
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Comparison 类型问题 - 两步生成触发测试")
print("=" * 60)

comparison_cases = [
    # (问题, 期望触发两步, 描述)
    ("芯导科技公司的上年度研发投入占比是多少？", False, "factual类问题不触发"),
    ("陕国投A与上年相比，合并利润表中的营业总收入是增长还是下降？", True, "与...相比 结构"),
    ("芯导科技公司2025年研发投入占营业收入的比例较2024年是增加还是减少？", True, "与...相比（较...是）结构"),
    ("联科科技在2025年度营业收入同比增速上有何差异？", True, "同比 关键词"),
    ("陕国投Ａ的信托财务报告中，信托资产总计期末余额与年初余额相比是增加还是减少？", True, "与...相比 结构"),
    ("山东药玻2025年年度报告中，持股数量排名前三的流通股股东分别是谁？", False, "extraction类问题不触发"),
    ("中兴通讯关于年度利润分配预案是怎么安排的？", True, "政策类问题触发（原有逻辑）"),
    ("公司今年和去年相比有什么变化？", True, "和...相比 结构"),
    ("同上年相比，营收有增长吗？", True, "同...相比 结构"),
    ("跟竞争对手相比，我们的优势是什么？", True, "跟...相比 结构"),
    ("今年业绩有什么变化？", True, "有...变化 结构"),
    ("两项业务有什么差异？", True, "有...差异 结构"),
    ("两个报表有什么区别？", True, "有...区别 结构"),
    ("前后两次披露有什么不同？", True, "有...不同 结构"),
    ("营业收入同比增长多少？", True, "同比增长 关键词"),
    ("环比增长率是多少？", True, "环比 关键词"),
    ("利润下降了百分之几？", True, "下降 关键词"),
    ("成本有所增加吗？", True, "增加 关键词"),
]

for question, expected, desc in comparison_cases:
    result = _should_use_two_step(question)
    ok = result == expected
    label = PASS if ok else FAIL
    print(f"[{label}] {desc}")
    print(f"       问题: {question[:50]}...")
    print(f"       期望: {'两步生成' if expected else '单步生成'}, 实际: {'两步生成' if result else '单步生成'}")
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