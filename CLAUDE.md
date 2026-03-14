# CLAUDE.md

**FinDoc-RAG** — 中文财务年报 RAG 问答系统（Python 3.12）。
解析 PDF/DOCX/XLSX 等格式，向量检索 + 两阶段精排，LLM 生成答案，支持自动评估。
当前阶段：单文档问答，向量检索（bge-m3）+ Reranker，即将引入 hybrid search。

## Tech Stack

- **Runtime**: Python 3.12, FastAPI, Uvicorn
- **Parsing**: pdfplumber, python-docx, python-pptx, openpyxl
- **Chunking**: LlamaIndex SentenceSplitter（512 chars / 80 overlap）
- **Embedding**: BAAI/bge-m3（1024-dim, sentence-transformers）
- **Vector DB**: ChromaDB（持久化, `data/chroma_db/`）
- **Doc Store**: SQLite（`data/doc_store.db`）
- **Reranker**: BAAI/bge-reranker-v2-m3（cross-encoder）
- **LLM**: Kimi（kimi-k2）或 Qwen（qwen3.5），通过 `LLM_PROVIDER` 切换
- **Query Router**: 查询路由与检索编排（自动公司识别 + 回退机制）
- **Eval**: Ragas 0.4.x + 自定义 LLM judge

## Key Commands

### 核心操作

```bash
# 启动 API 服务
uvicorn src.api.main:app --reload

# 交互式命令行问答
python cli.py

# Ingest 全部 PDF
python scripts/ingest.py

# Ingest 单个文件
python scripts/ingest.py --file "data/raw/文件名.pdf"

# 生成合成 QA 集
python scripts/evaluate.py --mode generate --num 50

# 运行评估
python scripts/evaluate.py --mode manual
python scripts/evaluate.py --mode full

# 导出 chunks（人工核查）
python scripts/export_chunks.py --file "陕国投" --output data/eval/chunks_verify.txt
```

### 测试命令（四层金字塔）

```bash
# 运行全部测试
pytest tests/ -v

# 分层运行
pytest tests/unit -v              # 单元测试（~15秒）
pytest tests/smoke -v             # 冒烟测试（~1分钟）
pytest tests/integration -v       # 集成测试（~10分钟，真实数据库交互）
pytest tests/e2e -v               # 端到端测试

# 运行特定测试文件/用例
pytest tests/integration/test_qa_pipeline.py -v
pytest tests/integration/test_qa_pipeline.py::TestQAPipeline::test_two_step_generation -v

# 覆盖率报告
pytest tests/unit tests/integration --cov=src --cov-report=html
```

## Environment Variables (`.env`)

| Variable | Used By |
|----------|---------|
| `MOONSHOT_API_KEY` | Kimi LLM |
| `DASHSCOPE_API_KEY` | Qwen LLM |

## Query Router 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_QUERY_ROUTER` | `True` | 总开关，False 时回退到旧逻辑 |
| `QUERY_ROUTER_ALLOW_AUTO_FILTER_FALLBACK` | `True` | 自动识别公司时允许回退到全局 |
| `QUERY_ROUTER_ALLOW_EXPLICIT_FILTER_FALLBACK` | `False` | 显式 filter 不允许回退 |
| `QUERY_ROUTER_EMPTY_RESULT_THRESHOLD` | `0` | 结果数 <= 此值时触发回退 |

## Data Layout

```
data/
├── raw/            # 原始文档（PDF/DOCX/XLSX/…）
├── chroma_db/      # ChromaDB 向量索引
├── doc_store.db    # SQLite：chunk text + metadata
└── eval/
    ├── manual_qa.json          # 手工标注 QA
    ├── synthetic_qa.json       # LLM 生成 QA
    ├── router_sensitive_qa.json # Router 敏感评估样本（16条）
    └── experiments/            # 参数扫描结果

src/
├── routing/                    # Query Router 模块
│   ├── __init__.py             # 导出 QueryClassifier, QueryRouter
│   ├── models.py               # QueryClassification, RetrievedContext
│   ├── query_classifier.py     # 查询分类器（场景识别 + 公司提取）
│   └── query_router.py         # 查询路由器（检索编排 + 回退机制）
├── retrieval/
├── generation/
├── ...

tests/                          # 测试套件（四层金字塔）
├── unit/                       # 单元测试
├── smoke/                      # 组件/冒烟测试
├── integration/                # 集成测试（多组件协作 + 真实数据库）
│   ├── conftest.py             # 共享 Fixtures
│   └── fixtures/               # 测试数据目录
└── e2e/                        # 端到端测试
```

## 延伸文档

需要了解具体细节时，按需读取对应文件：

| 文档 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 完整项目结构、目录说明 |
| `agent_docs/architecture.md` | 完整数据流、模块职责表、支持格式、Query Router 架构 |
| `agent_docs/ingestion.md` | PDF 解析（跨页合并、列名提取）、Chunker 三阶段逻辑、TableSummary 规则 |
| `agent_docs/retrieval-generation.md` | 两阶段检索配置、Query Router 检索流程、Prompt 强制规则、LLM 切换方式 |
| `agent_docs/evaluation.md` | Evaluator 指标体系、5-Block Flow、QA Schema v1.1 |
| `agent_docs/testing-guide.md` | 冒烟测试、评估测试、chunk 质量验证清单 |
| `tests/README.md` | 测试套件完整指南（四层金字塔、Fixtures、执行命令） |
| `tests/IMPLEMENTATION_SUMMARY.md` | 集成测试实施总结 |
| `mission.md` | Query Router 实施计划（Phase 1-7） |
| `docs/QUERY_ROUTER_PHASE1.md` | Phase 1 完成记录：数据契约与配置 |
| `docs/QUERY_ROUTER_PHASE2.md` | Phase 2 完成记录：QueryClassifier 实现 |
| `docs/QUERY_ROUTER_PHASE3.md` | Phase 3 完成记录：QueryRouter 实现 |
| `docs/QUERY_ROUTER_PHASE4.md` | Phase 4 完成记录：QAChain 接入 |
| `docs/QUERY_ROUTER_PHASE5.md` | Phase 5 完成记录：API 接入 |
| `docs/QUERY_ROUTER_PHASE6.md` | Phase 6 完成记录：Evaluator 接入 |
| `docs/QUERY_ROUTER_PHASE7.md` | Phase 7 完成记录：测试与样本扩充 |
