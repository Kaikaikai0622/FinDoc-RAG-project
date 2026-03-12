"""BGERerankerV2M3 smoke test — 验证模型加载 + 推理正常"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.reranker import BGERerankerV2M3

# ── 1. 单独测试 reranker ──────────────────────────────────────────────────
print("=" * 60)
print("[1] BGERerankerV2M3 单元 smoke test")
print("=" * 60)

reranker = BGERerankerV2M3()
assert reranker._model is None, "懒加载失败：构造时不应加载模型"
print("✓ 懒加载验证通过（构造时 _model is None）")

# 构造假 docs，模拟 Retriever.search() 返回结构
fake_docs = [
    {"chunk_id": "c1", "chunk_text": "净利润是扣除所有费用后的盈余。",        "source_file": "a.pdf", "page_number": 1, "score": 0.82},
    {"chunk_id": "c2", "chunk_text": "营业收入是企业主营业务产生的收入总额。",  "source_file": "a.pdf", "page_number": 2, "score": 0.75},
    {"chunk_id": "c3", "chunk_text": "资产负债率反映企业偿债能力。",            "source_file": "b.pdf", "page_number": 3, "score": 0.71},
    {"chunk_id": "c4", "chunk_text": "毛利率等于毛利除以营业收入。",            "source_file": "b.pdf", "page_number": 4, "score": 0.68},
    {"chunk_id": "c5", "chunk_text": "每股收益反映普通股股东获得的利润。",      "source_file": "c.pdf", "page_number": 5, "score": 0.65},
]

query = "什么是净利润？"
results = reranker.rerank(query, fake_docs, top_k=3)

assert reranker._model is not None, "懒加载失败：首次 rerank 后 _model 应已加载"
print("✓ 懒加载验证通过（首次 rerank 后 _model 已加载）")

assert len(results) == 3, f"期望返回 3 条，实际 {len(results)} 条"
print(f"✓ top_k=3 截断正确（返回 {len(results)} 条）")

assert all("rerank_score" in d for d in results), "结果缺少 rerank_score 字段"
print("✓ rerank_score 字段存在")

scores = [d["rerank_score"] for d in results]
assert scores == sorted(scores, reverse=True), "精排结果未按 rerank_score 降序排列"
print(f"✓ 降序排列正确，scores={[f'{s:.4f}' for s in scores]}")

print("\n精排 Top-3 结果：")
for i, d in enumerate(results, 1):
    print(f"  [{i}] score={d['rerank_score']:.4f} | {d['chunk_text'][:30]}")

# ── 2. 测试 edge case：空列表 ────────────────────────────────────────────
empty_result = reranker.rerank(query, [], top_k=3)
assert empty_result == [], "空输入应返回空列表"
print("\n✓ 空输入 edge case 通过")

# ── 3. 测试 top_k > len(docs) ────────────────────────────────────────────
small_docs = fake_docs[:2]
result_small = reranker.rerank(query, small_docs, top_k=5)
assert len(result_small) == 2, f"docs 少于 top_k 时应返回全部，实际 {len(result_small)} 条"
print("✓ top_k > len(docs) edge case 通过（返回全部可用文档）")

# ── 4. 验证原始 docs 未被修改（无副作用）────────────────────────────────
assert "rerank_score" not in fake_docs[0], "rerank 不应修改原始 docs（副作用检测）"
print("✓ 无副作用验证通过（原始 docs 未被修改）")

print("\n" + "=" * 60)
print("全部 smoke test 通过 ✓")
print("=" * 60)

