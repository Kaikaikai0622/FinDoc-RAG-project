# Architecture

## Data Flow

### Ingestion Flow
```
data/raw/ → DocumentRouter → Parser → Chunker（三阶段）
  → BGEm3EmbeddingService → ChromaDB + SQLite
```

### Query Flow (with Query Router)
```
Query → QueryRouter.route()
  ├── QueryClassifier.classify() → QueryClassification
  │     ├── extract_company_filter() → auto filter
  │     └── scene detection → generation_mode
  ├── Retrieval (filtered / global / filtered_then_global)
  │     └── RerankRetriever（粗检索 top-30 → BGE-Reranker 精排 top-10）
  └── RetrievedContext → QAChain
        → LLM（Kimi / Qwen）→ JSON response
```

### Query Router Paths
| Path | Condition | Description |
|------|-----------|-------------|
| `global` | No company filter | Search across all documents |
| `filtered` | Has filter & results > threshold | Search within filtered company |
| `filtered_then_global` | Auto filter empty & fallback allowed | Fallback to global when auto filter returns empty |

## Module Map

### Query Router (新增)
| Path | Responsibility |
|------|----------------|
| `src/routing/models.py` | 数据契约：QueryClassification、RetrievedChunk、RetrievedContext |
| `src/routing/query_classifier.py` | 查询分类：场景识别(factual/comparison/extraction/policy_qa) + 公司提取 + 生成模式判定 |
| `src/routing/query_router.py` | 检索编排：三路径检索(global/filtered/filtered_then_global) + 回退机制 |

### Core Modules
| Path | Responsibility |
|------|----------------|
| `config/settings.py` | 所有可调参数（chunk size、top-k、LLM、路径、policy keywords、Query Router 配置） |
| `src/ingestion/pipeline.py` | 文档→chunks→embed→store 编排 |
| `src/ingestion/document_router.py` | 多格式路由（PDF/DOCX/PPTX/XLSX/TXT/MD/CSV） |
| `src/ingestion/pdf_parser.py` | pdfplumber；表格→Markdown；列名提取（D1）；跨页合并（D2） |
| `src/ingestion/chunker.py` | 三阶段 chunker；政策跨页拼接（D4）；TableSummary（D3） |
| `src/embedding/bge_m3.py` | Lazy-load BAAI/bge-m3，GPU-aware |
| `src/storage/vector_store.py` | ChromaDB persistent client |
| `src/storage/doc_store.py` | SQLite：完整 chunk text + metadata |
| `src/retrieval/retriever.py` | 向量检索 → SQLite join |
| `src/retrieval/rerank_retriever.py` | 两阶段检索（粗检索 → 精排） |
| `src/generation/llm_service.py` | `BaseLLMService` + Qwen / Kimi via OpenAI SDK |
| `src/generation/qa_chain.py` | 检索 + LLM 编排，返回 timing metadata |
| `src/generation/prompts.py` | QA system prompt（含报表歧义/占比/总计/冲突规则） |
| `src/api/main.py` | FastAPI：`GET /health`，`POST /query` |
| `src/evaluation/evaluator.py` | LLM judge + Ragas + 场景加权准确率 |
| `src/evaluation/testset_generator.py` | 5-Block Flow 合成 QA 生成器 |
| `src/evaluation/dataset.py` | EvalDataset，支持 Schema v1.1 |

## Supported Formats

| Extension | Parser |
|-----------|--------|
| `.pdf` | PDFParser（pdfplumber） |
| `.docx` | DocxParser（python-docx） |
| `.pptx` | PptxParser（python-pptx） |
| `.xlsx` | XlsxParser（openpyxl） |
| `.xls` | XlsxParser（抛出友好错误，建议另存为 .xlsx） |
| `.txt` / `.md` / `.csv` | PlainTextParser |

## Dependency Injection

所有主要类（`IngestionPipeline`、`Retriever`、`QAChain`）构造函数接受可选子服务参数，默认使用标准实现，便于测试时注入 mock。

