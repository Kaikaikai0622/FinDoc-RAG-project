"""
冒烟测试：验证两个 chunker 修复
  Fix 1: TableSummary 表头只在分隔行之前查找，防止数据行被误判为表头导致数值丢失
  Fix 2: 政策段落跨页拼接（Chunker 阶段2），相邻政策页文本合并后整段保留
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from src.ingestion.chunker import (
    _extract_summary_chunks,
    _is_policy_section,
    _combined_text_is_policy,
    Chunker,
)
from src.ingestion.models import ParsedElement, ParsedDocument

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = 0

# ─────────────────────────────────────────────────────────────
# Fix 1: TableSummary 表头检测修复
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Fix 1: TableSummary 表头检测修复")
print("=" * 60)

# Case 1a: 正常表格（表头在分隔行之前）
injected_with_prefix = """【信托项目资产负债汇总表·信托资产,期末余额,年初余额】
| 信托资产 | 期末余额 | 年初余额 |
|---|---|---|
| 货币资金 | 63,963,365,627.93 | 142,810,162,018.81 |
| 信托资产总计 | 699,003,796,375.97 | 599,976,194,824.03 |"""

chunks = _extract_summary_chunks(
    injected=injected_with_prefix,
    table_title="信托项目利润及利润分配表",
    section_title="信托财务报告",
    source_file="陕国投Ａ：2025年年度报告.pdf",
    page_number=65,
)

ok = len(chunks) == 1
ok = ok and "信托项目资产负债汇总表" in chunks[0]
ok = ok and "699,003,796,375.97" in chunks[0]
ok = ok and "第65页" in chunks[0]

label = PASS if ok else FAIL
print(f"[{label}] 1a. 正常表格：从【注入行】提取正确表名，数值完整")
if chunks:
    print(f"       内容: {chunks[0]}")
if not ok:
    errors += 1

# Case 1b: ★真实 bug 场景★ 续表首行是数据行
injected_continuation = """【永续债·应付职工薪酬,777,352,422.86,613,919,572.42】
| 应付职工薪酬 | 777,352,422.86 | 613,919,572.42 |
|---|---|---|
| 应交税费 | 246,066,074.99 | 328,402,522.17 |
| 应付款项 |  |  |
| 负债合计 | 10,507,484,710.92 | 7,563,963,713.85 |
| 负债和股东权益总计 | 29,451,008,031.13 | 25,451,475,840.16 |"""

chunks_cont = _extract_summary_chunks(
    injected=injected_continuation,
    table_title="永续债",
    section_title="",
    source_file="陕国投Ａ：2025年年度报告.pdf",
    page_number=95,
)

summary_with_total = [c for c in chunks_cont if "负债和股东权益总计" in c]
ok_1b = len(summary_with_total) >= 1
ok_1b = ok_1b and "29,451,008,031.13" in summary_with_total[0]
ok_1b = ok_1b and "25,451,475,840.16" in summary_with_total[0]
label_1b = PASS if ok_1b else FAIL
print(f"[{label_1b}] 1b. ★续表场景★ 首行是数据行→表头正确识别→数值不丢失")
if summary_with_total:
    print(f"       内容: {summary_with_total[0]}")
else:
    print(f"       ❌ 未找到含'负债和股东权益总计'的 summary chunk")
    print(f"       所有 chunks: {chunks_cont}")
if not ok_1b:
    errors += 1

# Case 1c: 6列宽表
injected_wide = """【信托项目利润及利润分配表·衍生金融资产,卖出回购金融资产款,48956535029.11,21592305903.10】
| 衍生金融资产 |  |  | 卖出回购金融资产款 | 48,956,535,029.11 | 21,592,305,903.10 |
|---|---|---|---|---|---|
| 应收清算款 | 10,483,835,304.84 | 480,865,628.09 | 应付受托人报酬 | 558,153,716.46 | 541,476,745.73 |
| 信托资产总计 | 699,003,796,375.97 | 599,976,194,824.03 | 信托负债和净资产总计 | 699,003,796,375.97 | 599,976,194,824.03 |"""

chunks_wide = _extract_summary_chunks(
    injected=injected_wide,
    table_title="",
    section_title="",
    source_file="test.pdf",
    page_number=65,
)
summary_trust = [c for c in chunks_wide if "信托资产总计" in c]
ok_1c = len(summary_trust) >= 1 and "699,003,796,375.97" in summary_trust[0]
label_1c = PASS if ok_1c else FAIL
print(f"[{label_1c}] 1c. 6列宽表 → 总计行含数值")
if summary_trust:
    print(f"       内容: {summary_trust[0]}")
if not ok_1c:
    errors += 1

# Case 1d: 无分隔行 fallback
injected_nosep = """| 信托资产总计 | 699,003,796,375.97 | 599,976,194,824.03 |"""

chunks_nosep = _extract_summary_chunks(
    injected=injected_nosep,
    table_title="测试表",
    section_title="",
    source_file="test.pdf",
    page_number=1,
)
ok_1d = len(chunks_nosep) >= 1 and "699,003,796,375.97" in chunks_nosep[0]
label_1d = PASS if ok_1d else FAIL
print(f"[{label_1d}] 1d. 无分隔行 fallback → 数值不丢失")
if chunks_nosep:
    print(f"       内容: {chunks_nosep[0]}")
if not ok_1d:
    errors += 1

# ─────────────────────────────────────────────────────────────
# Fix 2: 政策段落跨页拼接
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Fix 2: 政策段落跨页拼接")
print("=" * 60)

# Case 2a: 基础政策检测
elem_midtext = ParsedElement(
    text="公司经本次董事会审议通过的利润分配预案为：以总股本5,113,970,358股为基数。",
    category="Text",
    page_or_index=2,
    metadata={"section_title": "重要提示、目录", "page": 2},
)
ok_2a = _is_policy_section([elem_midtext]) is True
label_2a = PASS if ok_2a else FAIL
print(f"[{label_2a}] 2a. 正文含'利润分配预案' → _is_policy_section=True")
if not ok_2a:
    errors += 1

# Case 2b: 普通段落不误触发
ok_2b = _combined_text_is_policy("信托资产规模达6990亿元") is False
label_2b = PASS if ok_2b else FAIL
print(f"[{label_2b}] 2b. 普通段落 → _combined_text_is_policy=False")
if not ok_2b:
    errors += 1

# Case 2c: ★真实 bug 场景★ 跨页政策段落合并
page55_text = (
    "十一、公司利润分配及资本公积金转增股本情况\n"
    "报告期内利润分配政策，特别是现金分红政策的制定、执行或调整情况\n"
    "适用 □不适用\n"
    "2024 年度利润分配方案：以总股本5,113,970,358股为基数，"
    "2024年半年度已按每10股派发现金红利0.10元（含税）。\n"
    "该方案已于 2025 年 6 月 4 日实施完毕。\n"
    "2025年半年度利润分配方案：以 2025 年 6 月末总股本 5,113,970,358 股为基数，按每 10 股派发"
)
page56_text = (
    "现金红利 0.10 元（含税）。该方案已于 2025年 10 月 22 日实施。\n"
    "2025 年三季度利润分配方案：以 2025 年 9 月末总股本 5,113,970,358 股为基数，"
    "按每 10 股派发现金红利 0.20 元（含税）。该方案已于 2025 年 12 月 16 日实施。"
)

doc_cross_page = ParsedDocument(
    source_file="test_policy.pdf",
    file_type="pdf",
    elements=[
        ParsedElement(
            text=page55_text,
            category="Text",
            page_or_index=55,
            metadata={"section_title": "公司治理、环境和社会", "page": 55},
        ),
        ParsedElement(
            text=page56_text,
            category="Text",
            page_or_index=56,
            metadata={"section_title": "公司治理、环境和社会", "page": 56},
        ),
    ],
)

chunker = Chunker()
chunks_cross = chunker.chunk_document(doc_cross_page)

ok_2c = len(chunks_cross) == 1
ok_2c = ok_2c and "按每 10 股派发" in chunks_cross[0]["chunk_text"]
ok_2c = ok_2c and "该方案已于 2025 年 12 月 16 日实施" in chunks_cross[0]["chunk_text"]
label_2c = PASS if ok_2c else FAIL
print(f"[{label_2c}] 2c. ★跨页政策段落★ Page 55+56 → {len(chunks_cross)} chunk（期望1）")
if chunks_cross:
    txt = chunks_cross[0]["chunk_text"]
    print(f"       chunk 长度: {len(txt)} 字符")
    print(f"       含Page55末尾: {'按每 10 股派发' in txt}")
    print(f"       含Page56内容: {'该方案已于 2025 年 12 月 16 日实施' in txt}")
if not ok_2c:
    errors += 1

# Case 2d: 普通文本跨页不误合并
doc_normal = ParsedDocument(
    source_file="test_normal.pdf",
    file_type="pdf",
    elements=[
        ParsedElement(
            text="公司主要从事信托业务，报告期内信托资产规模达6990亿元。",
            category="Text",
            page_or_index=10,
            metadata={"section_title": "管理层讨论", "page": 10},
        ),
        ParsedElement(
            text="公司持续优化资产配置，加强风险管控能力。",
            category="Text",
            page_or_index=11,
            metadata={"section_title": "管理层讨论", "page": 11},
        ),
    ],
)
chunks_normal = chunker.chunk_document(doc_normal)
ok_2d = len(chunks_normal) == 2
label_2d = PASS if ok_2d else FAIL
print(f"[{label_2d}] 2d. 普通文本跨页不误合并 → {len(chunks_normal)} chunks（期望2）")
if not ok_2d:
    errors += 1

# ─────────────────────────────────────────────────────────────
# 端到端：TableSummary 完整流程
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("端到端：Chunker TableSummary 含正确表名和数值")
print("=" * 60)

table_md = """| 信托资产 | 期末余额 | 年初余额 |
|---|---|---|
| 货币资金 | 63,963,365,627.93 | 142,810,162,018.81 |
| 交易性金融资产 | 484,474,530,071.39 | 244,179,611,390.63 |
| 信托资产总计 | 699,003,796,375.97 | 599,976,194,824.03 |"""

doc_table = ParsedDocument(
    source_file="陕国投A.pdf",
    file_type="pdf",
    elements=[
        ParsedElement(
            text=table_md,
            category="Table",
            page_or_index=65,
            metadata={
                "page": 65,
                "section_title": "信托财务报告",
                "table_title": "信托项目资产负债汇总表",
                "table_index": 1,
                "column_names": "信托资产,期末余额,年初余额",
                "col_count": 3,
            },
        )
    ],
)

chunks_table = chunker.chunk_document(doc_table)
summary_chunks = [c for c in chunks_table if "总计" in c["chunk_text"] and "来源" in c["chunk_text"]]

ok_ts1 = len(summary_chunks) >= 1
ok_ts2 = ok_ts1 and "信托项目资产负债汇总表" in summary_chunks[0]["chunk_text"]
ok_ts3 = ok_ts1 and "699,003,796,375.97" in summary_chunks[0]["chunk_text"]
ok_ts4 = ok_ts1 and summary_chunks[0]["metadata"]["element_category"] == "TableSummary"

label_ts = PASS if (ok_ts1 and ok_ts2 and ok_ts3 and ok_ts4) else FAIL
print(f"[{label_ts}] TableSummary chunk 生成: {len(summary_chunks)} 个")
if summary_chunks:
    print(f"       内容:\n{summary_chunks[0]['chunk_text']}")
    print(f"       element_category: {summary_chunks[0]['metadata']['element_category']}")
if not (ok_ts1 and ok_ts2 and ok_ts3 and ok_ts4):
    errors += 1
    if not ok_ts2: print("       ❌ 表名不含'信托资产负债汇总表'")
    if not ok_ts3: print("       ❌ 未包含数值 699,003,796,375.97")
    if not ok_ts4: print("       ❌ element_category 不是 TableSummary")

# ─────────────────────────────────────────────────────────────
# 结果汇总
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
if errors == 0:
    print(f"\033[32m✅ 所有冒烟测试通过（0 failures）\033[0m")
else:
    print(f"\033[31m❌ {errors} 个测试失败\033[0m")
print("=" * 60)
sys.exit(errors)

