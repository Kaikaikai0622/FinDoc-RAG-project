# CLAUDE.md

**FinDoc-RAG** — 中文财务年报 RAG 问答系统（Python 3.12）。
解析 PDF/DOCX/XLSX 等格式，向量检索 + 两阶段精排，LLM 生成答案，支持自动评估。
当前阶段：单文档问答，向量检索（bge-m3）+ Reranker，即将引入 hybrid search。

## Tech Stack

- **Runtime**: Python 3.12, FastAPI, Uvicorn
- **Parsing**: pdfplumber, python-docx, python-pptx, openpyxl
- **Chunking**: LlamaIndex SentenceSplitter（256 chars / 50 overlap）
- **Embedding**: BAAI/bge-m3（1024-dim, sentence-transformers）
- **Vector DB**: ChromaDB（持久化, `data/chroma_db/`）
- **Doc Store**: SQLite（`data/doc_store.db`）
- **Reranker**: BAAI/bge-reranker-v2-m3（cross-encoder）
- **LLM**: Kimi（kimi-k2）或 Qwen（qwen3.5），通过 `LLM_PROVIDER` 切换
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

## Data Layout

```
data/
├── raw/            # 原始文档（PDF/DOCX/XLSX/…）
├── chroma_db/      # ChromaDB 向量索引
├── doc_store.db    # SQLite：chunk text + metadata
└── eval/
    ├── manual_qa.json       # 手工标注 QA
    ├── synthetic_qa.json    # LLM 生成 QA
    └── experiments/         # 参数扫描结果

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
| `agent_docs/architecture.md` | 完整数据流、模块职责表、支持格式 |
| `agent_docs/ingestion.md` | PDF 解析（跨页合并、列名提取）、Chunker 三阶段逻辑、TableSummary 规则 |
| `agent_docs/retrieval-generation.md` | 两阶段检索配置、Prompt 强制规则、LLM 切换方式 |
| `agent_docs/evaluation.md` | Evaluator 指标体系、5-Block Flow、QA Schema v1.1 |
| `agent_docs/testing-guide.md` | 冒烟测试、评估测试、chunk 质量验证清单 |
| `tests/README.md` | 测试套件完整指南（四层金字塔、Fixtures、执行命令） |
| `tests/IMPLEMENTATION_SUMMARY.md` | 集成测试实施总结 |
