# Testing Guide

## Philosophy

本项目以**评估测试**为主，而非传统单元测试。

## Smoke Tests（每次改动后必跑）

```bash
# Chunker bug 修复验证（TableSummary 数值 + 政策段落跨页拼接）
python tests/test_chunker_fixes.py

# Embedding model 加载（首次运行会下载 ~2.2GB 模型）
python tests/test_embedding.py

# 全部单元测试
python -m pytest tests/ -v
```

## Evaluation Tests

```bash
# Manual eval（手工标注集，含 Ragas 指标）
python scripts/evaluate.py --mode manual

# Synthetic eval
python scripts/evaluate.py --mode synthetic

# 生成合成 QA 集（50 题）
python scripts/evaluate.py --mode generate --num 50
```

## Chunk 质量验证

```bash
# 导出指定公司 chunks，人工核查
python scripts/export_chunks.py --file "陕国投" --output data/eval/chunks_verify.txt

# 验证 manual_qa.json 格式
python scripts/validate_qa.py
```

重点检查项：
1. **TableSummary**：搜索"总计"，确认同行有数值（如 `699,003,796,375.97`）
2. **政策段落跨页**：搜索"按每 10 股派发"，确认同一 chunk 内有接续内容
3. **跨页表格合并**：P64 末尾表头 + P65 续表数据应在同一逻辑块

## 添加单元测试

```bash
pip install pytest pytest-asyncio
# 在 tests/ 目录下创建 test_*.py
pytest tests/
```

重点测试区域：`chunker.py`（边界逻辑）、`doc_store.py`（SQLite 操作）、`retriever.py`（检索精度）

