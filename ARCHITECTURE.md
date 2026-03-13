RAG_1/
├── CLAUDE.md                   # 根文件（指针层，~80 行）
├── ARCHITECTURE.md             # 本文件
├── pyproject.toml
├── .env / .env.example
├── cli.py                      # 交互式命令行问答
├── config/
│   └── settings.py             # 全局参数（chunk_size, top_k, LLM provider, policy keywords 等）
├── agent_docs/                 # 详细设计文档（按需读取）
│   ├── architecture.md         # 数据流、模块职责表
│   ├── ingestion.md            # PDF 解析、三阶段 Chunker、TableSummary 规则
│   ├── retrieval-generation.md # 检索配置、Prompt 规则、LLM 切换
│   ├── evaluation.md           # Evaluator 指标、5-Block Flow、QA Schema v1.1
│   └── testing-guide.md        # 冒烟测试、评估测试、chunk 质量验证
├── src/
│   ├── routing/
│   │   ├── __init__.py         # Query Router 模块入口
│   │   ├── models.py           # QueryClassification / RetrievedChunk / RetrievedContext
│   │   ├── query_classifier.py # 查询分类器：场景识别 + 公司提取 + 生成模式判定
│   │   └── query_router.py     # 查询路由器：检索编排 + 回退机制
│   ├── ingestion/
│   │   ├── pipeline.py         # 串联：路由→解析→切块→embedding→入库
│   │   ├── document_router.py  # 多格式路由（PDF/DOCX/PPTX/XLSX/TXT/MD/CSV）
│   │   ├── pdf_parser.py       # pdfplumber；列名提取（D1）；跨页表格合并（D2）
│   │   ├── docx_parser.py
│   │   ├── pptx_parser.py
│   │   ├── xlsx_parser.py
│   │   ├── plain_text_parser.py
│   │   ├── chunker.py          # 三阶段 chunker；政策跨页拼接（D4）；TableSummary（D3）
│   │   └── models.py           # ParsedDocument / ParsedElement
│   ├── embedding/
│   │   ├── base.py             # EmbeddingService 接口
│   │   └── bge_m3.py           # BAAI/bge-m3（1024-dim, lazy-load, GPU-aware）
│   ├── storage/
│   │   ├── doc_store.py        # SQLite：chunk text + metadata
│   │   └── vector_store.py     # ChromaDB persistent client
│   ├── retrieval/
│   │   ├── retriever.py        # 向量检索 → SQLite join
│   │   ├── reranker.py         # BGE-Reranker-v2-m3 cross-encoder
│   │   └── rerank_retriever.py # 两阶段：粗检索 top-30 → 精排 top-10
│   ├── generation/
│   │   ├── llm_service.py      # BaseLLMService + Kimi / Qwen（OpenAI SDK）
│   │   ├── qa_chain.py         # 检索 + Prompt + LLM，返回 timing metadata
│   │   └── prompts.py          # QA system prompt（报表歧义/占比/总计/冲突规则）
│   ├── evaluation/
│   │   ├── evaluator.py        # LLM judge + Ragas + 场景加权准确率
│   │   ├── testset_generator.py# 5-Block Flow 合成 QA（含 Block 6 数值/单位回查）
│   │   ├── dataset.py          # EvalDataset，Schema v1.1
│   │   ├── experiment.py       # 参数扫描
│   │   └── report.py           # Markdown 报告生成
│   └── api/
│       └── main.py             # FastAPI：GET /health，POST /query
├── scripts/
│   ├── ingest.py               # 批量/单文件 ingest
│   ├── evaluate.py             # 评估入口（manual / synthetic / full / generate）
│   ├── export_chunks.py        # 导出 chunks 供人工核查
│   └── validate_qa.py          # 校验 manual_qa.json 格式
├── data/
│   ├── raw/                    # 原始文档（PDF/DOCX/XLSX/…）
│   ├── chroma_db/              # ChromaDB 向量索引
│   ├── doc_store.db            # SQLite
│   └── eval/
│       ├── manual_qa.json      # 手工标注 QA
│       ├── synthetic_qa.json   # LLM 生成 QA
│       ├── experiments/        # 参数扫描结果
│       └── reports/            # 评估报告
└── tests/                          # 测试套件（四层测试金字塔）
    ├── unit/                       # 单元测试（函数/类级别，大量使用 Mock）
    │   ├── test_document_router.py # DocumentRouter 多格式路由测试
    │   ├── test_query_classifier.py # QueryClassifier 单元测试（22个）
    │   ├── test_query_router.py    # QueryRouter 单元测试（17个）
    │   ├── test_qa_chain_router.py # QAChain Router 集成测试（10个）
    │   └── check_imports.py        # 依赖导入健康检查
    ├── smoke/                      # 组件/冒烟测试（单组件完整功能）
    │   ├── test_chunker_fixes.py   # Chunker 冒烟测试（TableSummary + 跨页政策段落）
    │   ├── test_embedding.py       # Embedding 冒烟测试（首次运行下载 ~2.2GB 模型）
    │   ├── test_reranker.py        # Reranker 单元测试
    │   ├── test_company_resolver.py # 公司名称解析测试
    │   ├── test_comparison_two_step.py # 两步生成策略测试
    │   ├── test_filter_file.py     # 文件过滤功能测试
    │   ├── test_defensive_check.py # 防御性检查测试
    │   └── test_ragas_cols.py      # Ragas 指标列名兼容性测试
    ├── integration/                # 集成测试（多组件协作 + 真实数据库交互）
    │   ├── conftest.py             # 共享 Fixtures（隔离存储、Mock LLM、预填充数据）
    │   ├── test_ingestion_pipeline.py   # Ingestion Pipeline（PDF→Chunk→Embedding→存储）
    │   ├── test_retrieval_pipeline.py   # Retrieval Pipeline（Query→检索→SQLite Join）
    │   ├── test_qa_pipeline.py          # QA Pipeline（Question→检索→LLM Generation）
    │   ├── test_evaluation_pipeline.py  # Evaluation Pipeline（QA对→评估→报告）
    │   ├── test_api.py                  # API 集成测试（FastAPI TestClient）
    │   ├── test_api_router.py           # API Router 集成测试（6个）
    │   ├── test_evaluator_router.py     # Evaluator Router 集成测试（6个）
    │   └── fixtures/                    # 测试数据目录
    │       ├── sample_pdfs/             # PDF 样本文件
    │       ├── sample_qa/               # 测试 QA 数据集
    │       └── expected_outputs/        # 预期输出
    └── e2e/                        # 端到端测试（完整用户场景）
        └── test_complete_workflow.py    # 完整工作流：Ingest→Retrieval→QA
