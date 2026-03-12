"""
冒烟测试：验证公司名称解析模块
"""
import sys
import os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.company_resolver import CompanyResolver

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = 0

print("=" * 60)
print("公司名称解析模块测试")
print("=" * 60)
print()

# ─────────────────────────────────────────────────────────────
# 测试1: 文件名解析
# ─────────────────────────────────────────────────────────────
print("[测试1] 从文件名提取公司名称")

resolver = CompanyResolver()

test_filenames = [
    ("陕国投Ａ：2025年年度报告.pdf", "陕国投Ａ"),
    ("芯导科技：2025年年度报告.pdf", "芯导科技"),
    ("指南针：2025年年度报告.pdf", "指南针"),
    ("中兴通讯：2025年年度报告.pdf", "中兴通讯"),
    ("山东药玻：2025年年度报告.pdf", "山东药玻"),
    ("联科科技：2025年年度报告.pdf", "联科科技"),
]

for filename, expected in test_filenames:
    result = resolver._extract_company_from_filename(filename)
    ok = result == expected
    label = PASS if ok else FAIL
    print(f"  [{label}] {filename[:20]}... -> {result}")
    if not ok:
        print(f"       期望: {expected}, 实际: {result}")
        errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 测试2: 字符标准化（全角转半角）
# ─────────────────────────────────────────────────────────────
print("[测试2] 字符标准化（全角转半角）")

test_chars = [
    ("陕国投Ａ", "陕国投A"),  # 全角A转半角
    ("芯导科技", "芯导科技"),  # 无变化
    ("ＡＢＣ", "ABC"),  # 全角字母
    ("１２３", "123"),  # 全角数字
    ("中文：英文", "中文:英文"),  # 全角冒号
]

for input_text, expected in test_chars:
    result = resolver._normalize_chars(input_text)
    ok = result == expected
    label = PASS if ok else FAIL
    print(f"  [{label}] {input_text!r} -> {result!r}")
    if not ok:
        print(f"       期望: {expected!r}")
        errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 测试3: 别名生成
# ─────────────────────────────────────────────────────────────
print("[测试3] 别名映射生成")

# 重新初始化以清空别名
resolver = CompanyResolver()
resolver._add_aliases("陕国投A", "陕国投A：2025年年度报告.pdf")
resolver._add_aliases("芯导科技", "芯导科技：2025年年度报告.pdf")

aliases_to_check = [
    ("陕国投A", "陕国投A"),  # 原始名
    ("陕国投", "陕国投A"),   # 去A后缀
    ("芯导科技", "芯导科技"),
    ("芯导", "芯导科技"),    # 去科技后缀
]

for alias, expected_company in aliases_to_check:
    result = resolver._company_aliases.get(alias)
    ok = result == expected_company
    label = PASS if ok else FAIL
    print(f"  [{label}] 别名 '{alias}' -> 公司: {result}")
    if not ok:
        print(f"       期望: {expected_company}, 实际: {result}")
        errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 测试4: 从问题提取公司名
# ─────────────────────────────────────────────────────────────
print("[测试4] 从用户问题提取公司名")

# 模拟已加载的公司
resolver._company_map = {
    "陕国投A": "陕国投Ａ：2025年年度报告.pdf",
    "芯导科技": "芯导科技：2025年年度报告.pdf",
    "指南针": "指南针：2025年年度报告.pdf",
}
resolver._company_aliases = {
    "陕国投A": "陕国投A",
    "陕国投": "陕国投A",
    "芯导科技": "芯导科技",
    "芯导": "芯导科技",
    "指南针": "指南针",
}
resolver._initialized = True

test_questions = [
    ("陕国投A的营收是多少？", "陕国投A"),
    ("陕国投的营收是多少？", "陕国投A"),  # 简称
    ("芯导科技的利润如何？", "芯导科技"),
    ("芯导今年业绩怎么样？", "芯导科技"),  # 简称
    ("指南针的股价是多少？", "指南针"),
    ("介绍一下这家公司的业务", None),  # 无公司名
]

for question, expected_company in test_questions:
    result = resolver.extract_company_from_question(question)
    ok = result == expected_company
    label = PASS if ok else FAIL
    print(f"  [{label}] {question[:25]}... -> {result}")
    if not ok:
        print(f"       期望: {expected_company}, 实际: {result}")
        errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 测试5: 全角半角混合匹配
# ─────────────────────────────────────────────────────────────
print("[测试5] 全角半角混合匹配")

test_mixed = [
    ("陕国投Ａ今年怎么样？", "陕国投A"),  # 全角Ａ
    ("陕国投A今年怎么样？", "陕国投A"),   # 半角A
]

for question, expected in test_mixed:
    result = resolver.extract_company_from_question(question)
    ok = result == expected
    label = PASS if ok else FAIL
    print(f"  [{label}] {question} -> {result}")
    if not ok:
        print(f"       期望: {expected}, 实际: {result}")
        errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 结果汇总
# ─────────────────────────────────────────────────────────────
print("=" * 60)
if errors == 0:
    print("\033[32m✅ 所有冒烟测试通过（0 failures）\033[0m")
else:
    print(f"\033[31m❌ {errors} 个测试失败\033[0m")
print("=" * 60)
sys.exit(errors)
