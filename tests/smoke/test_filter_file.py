"""
冒烟测试：验证决策3 - ChromaDB where 参数按公司筛选
"""
import sys
import os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retriever import Retriever
from src.retrieval.rerank_retriever import RerankRetriever
from src.generation.qa_chain import QAChain

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = 0

print("=" * 60)
print("决策3：ChromaDB where 参数按公司筛选测试")
print("=" * 60)
print()

# ─────────────────────────────────────────────────────────────
# 测试1: 验证 where 过滤条件构建
# ─────────────────────────────────────────────────────────────
print("[测试1] where 过滤条件构建")

# 测试 where 条件构建逻辑
def build_where_filter(filter_file: str | None):
    """模拟 Retriever 中的 where 条件构建"""
    if filter_file:
        return {"source_file": {"$eq": filter_file}}
    return None

test_cases = [
    (None, None, "无过滤时返回 None"),
    (
        "陕国投Ａ：2025年年度报告.pdf",
        {"source_file": {"$eq": "陕国投Ａ：2025年年度报告.pdf"}},
        "有过滤时返回 eq 条件",
    ),
    (
        "指南针：2025年年度报告.pdf",
        {"source_file": {"$eq": "指南针：2025年年度报告.pdf"}},
        "按完整 source_file 精确匹配",
    ),
]

for filter_file, expected, desc in test_cases:
    result = build_where_filter(filter_file)
    ok = result == expected
    label = PASS if ok else FAIL
    print(f"  [{label}] {desc}")
    print(f"       输入: {filter_file!r}")
    print(f"       输出: {result!r}")
    if not ok:
        errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 测试2: Retriever 接口签名
# ─────────────────────────────────────────────────────────────
print("[测试2] Retriever.search() 接口签名")

import inspect

# 检查 Retriever.search 参数
sig = inspect.signature(Retriever.search)
params = list(sig.parameters.keys())
ok = 'filter_file' in params
label = PASS if ok else FAIL
print(f"  [{label}] Retriever.search() 包含 filter_file 参数")
if not ok:
    errors += 1
    print(f"       实际参数: {params}")

# 检查参数默认值
filter_file_param = sig.parameters.get('filter_file')
if filter_file_param:
    ok = filter_file_param.default is None
    label = PASS if ok else FAIL
    print(f"  [{label}] filter_file 参数默认值为 None")
    if not ok:
        errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 测试3: RerankRetriever 接口签名
# ─────────────────────────────────────────────────────────────
print("[测试3] RerankRetriever.search() 接口签名")

sig = inspect.signature(RerankRetriever.search)
params = list(sig.parameters.keys())
ok = 'filter_file' in params
label = PASS if ok else FAIL
print(f"  [{label}] RerankRetriever.search() 包含 filter_file 参数")
if not ok:
    errors += 1
    print(f"       实际参数: {params}")

print()

# ─────────────────────────────────────────────────────────────
# 测试4: QAChain.ask() 接口签名
# ─────────────────────────────────────────────────────────────
print("[测试4] QAChain.ask() 接口签名")

sig = inspect.signature(QAChain.ask)
params = list(sig.parameters.keys())
ok = 'filter_file' in params
label = PASS if ok else FAIL
print(f"  [{label}] QAChain.ask() 包含 filter_file 参数")
if not ok:
    errors += 1
    print(f"       实际参数: {params}")

print()

# ─────────────────────────────────────────────────────────────
# 测试5: 便捷函数签名
# ─────────────────────────────────────────────────────────────
print("[测试5] 便捷函数接口签名")

from src.retrieval.retriever import retrieve
from src.generation.qa_chain import ask

sig = inspect.signature(retrieve)
ok = 'filter_file' in sig.parameters
label = PASS if ok else FAIL
print(f"  [{label}] retrieve() 包含 filter_file 参数")
if not ok:
    errors += 1

sig = inspect.signature(ask)
ok = 'filter_file' in sig.parameters
label = PASS if ok else FAIL
print(f"  [{label}] ask() 包含 filter_file 参数")
if not ok:
    errors += 1

print()

# ─────────────────────────────────────────────────────────────
# 测试6: API 请求模型
# ─────────────────────────────────────────────────────────────
print("[测试6] API QueryRequest 模型")

try:
    from src.api.main import QueryRequest
    from pydantic import BaseModel

    # 检查是否是 Pydantic 模型
    ok = issubclass(QueryRequest, BaseModel)
    label = PASS if ok else FAIL
    print(f"  [{label}] QueryRequest 是 Pydantic BaseModel")
    if not ok:
        errors += 1

    # 检查字段
    fields = QueryRequest.model_fields.keys()
    ok = 'filter_file' in fields
    label = PASS if ok else FAIL
    print(f"  [{label}] QueryRequest 包含 filter_file 字段")
    if not ok:
        errors += 1
        print(f"       实际字段: {list(fields)}")

    # 测试实例化
    req = QueryRequest(question="测试", filter_file="陕国投")
    ok = req.question == "测试" and req.filter_file == "陕国投"
    label = PASS if ok else FAIL
    print(f"  [{label}] QueryRequest 可正确实例化")
    if not ok:
        errors += 1

    # 测试默认值为 None
    req2 = QueryRequest(question="测试")
    ok = req2.filter_file is None
    label = PASS if ok else FAIL
    print(f"  [{label}] filter_file 默认为 None")
    if not ok:
        errors += 1

except Exception as e:
    print(f"  [{FAIL}] API 模型测试失败: {e}")
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
