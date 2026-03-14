# FinDoc-RAG

**中文财务年报问答系统**

选择中国上市公司年报作为 RAG 落地场景，基于以下三点判断：
1. 数据公开可得：年报为法定披露文件，无数据获取与合规障碍。
2. 文档复杂度高：单份年报通常达 200 多页，包含混合长文本、嵌套表格、跨页条款等内容。这些复杂的文档结构是 RAG 解析层的主要难点。
3. 问答需求真实且可验证：分红政策、担保事项、数据提取等问题均有明确的标准答案，能够有效支撑评估体系中 Ground Truth 的构建。
   

### 项目目标

构建一套端到端 RAG Pipeline，核心交付：
1. 一条从 PDF 解析到答案生成的完整链路，覆盖表格、长文本、跨页条款。
2. 一套可重复执行、成本可控的自动化评估基础设施，保证每次检索生成策略都有量化依据。 
3. 一组经 Re-ranker 与 Query Router 验证的检索与生成策，附指标对比。

## 技术亮点

### 1. SDG Hub — Synthetic QA 生成器

- **解决 Ragas 痛点**：本方案采用 **Block-Flow-Contract** 架构解决存在Ragas generate流程黑盒、领域适配困难、成本不可控等问题。
- **Ground Truth 与检索单元对齐**：生成器从 DocStore 读 chunk 而非原始 PDF，确保评估数据的 ground truth context 与检索索引单元严格一致。
- **模块化 pipeline 设计**：（Block → Flow → Contract），支持一键重新生成评估数据集，单次生成 50 条问答成本稳定在 **¥1 以内**。
- **可持续评估闭环、成本可控**：pipeline 与 Hybrid Search 的 chunk 粒度完全对齐。
- 后续 Reranker、Router 等模块的收益均基于该数据集量化验证（见下方指标）

## 2. 基于 PDFPlumber 构建的文档轻量化解析架构

- **领域驱动语义拼接**：针对"分红、担保、利润分配"等核心条款设计跨页逻辑合并策略，通过关键词触发动态窗口延伸
- **长表格逻辑重构**：支持大规格表格的按行分段解析，并自动识别提取"总计/合计"行构建聚合 Chunk
- **上下文前缀注入**：实现**"标题+表头"语义锚定**，将父级标题与列名元数据自动注入每一个 Table Chunk，从底层消除 RAG 链路中表格数据的语义断裂与上下文丢失。

### 3. Query Router（MVP）

- **准确性与检索命中稳定提升**：统一处理显式 `filter_file` 与自动公司识别，按规则输出检索范围与生成模式
- **复杂问题覆盖更广**：新增 可观测字段，针对对比、条款、抽取类问题的处理能力更均衡，提升系统在业务场景中的可用边界。
- **降低"黑盒感"**：结果同时提供来源与处理标记，确保"回答所用上下文"与"评估上下文"一致
- **配套配置完善**：提供 Router 开关、回退阈值与置信度阈值配置，并通过分类器与路由器单测覆盖核心路径

### Query Router引入+Chunk策略调整

| 评估维度       | 指标名称 (Metrics)              | Run: manual_ | Run: synthetic_ |
|:-----------|:----------------------------|:-------------|:----------------|
| **基础规模**   | 样本总量 (Total)                | 30           | 65              |
| **准确性**    | 准确率 (Accuracy)              | 0.7333       | 0.8692          |
| **检索质量**   | 检索命中率 (Retrieval Hit Rate)  | 0.9333       | 0.9323          |
|            | 平均检索排名 (Avg Retrieval Rank) | 1.6786       | 1.2690          |
| **RAG 质量** | 忠实度 (Faithfulness)          | 0.8885       | 0.8963          |
|            | 回答相关性 (Answer Relevancy)    | 0.5085       | 0.6748          |
|            | 上下文精确度 (Context Precision)  | 0.5214       | 0.7774          |
|            | 上下文召回率 (Context Recall)     | 0.7489       | 0.9385          |

### 4. 场景化生成策略（动态路由）

对问题处理等级进行路由：

- **两步生成**：政策条款类问题 → Step1 纯抽取原文 → Step2 生成结论（保证 grounding）
- **单步生成**：事实类问题直接回答（省 token）
- **启发式判断问题类型**：正则匹配 policy/extraction 关键词自动切换

### 5. 两阶段检索（RerankRetriever）

生产级 RAG 标准优化：

- **粗检索**：ChromaDB 向量检索 Top-30 候选
- **精排**：BGE Reranker-v2-m3 交叉编码器 → Top-7 最终结果
- **懒加载模型**，禁用时不占显存

### 引入 Reranker 前后对比

| 指标 | Baseline (无Reranker) | +Reranker |
|------|----------------------|-----------|
| 检索命中率 (Hit Rate) | 0.88 | **0.92** |
| 忠实度 (Faithfulness) | 0.79 | **0.82** |
| 答案相关性 (Answer Relevancy) | 0.68 | **0.70** |
| 上下文精度 (Context Precision) | 0.32 | **0.47** |

---
---

## 技术架构
```
PDF/DOCX/XLSX -> 文档解析 -> 智能切块 -> 向量化(bge-m3) -> ChromaDB
                                                            |
用户问题 -> Query Router(查询分类: specific/global) ---------+
             |                                      |
             | (specific_query)                     | (global_summary)
             v                                      v
       向量检索(Top-30) -> 精排(bge-reranker, Top-7)    粗召回/相关性过滤 -> 分层摘要(Map-Reduce)
                               |                                      |
                               +------------- 上下文融合 --------------+
                                              |
                                              v
                                     答案生成 -> LLM(Kimi/Qwen)
```

**核心模型**：
- Embedding: BAAI/bge-m3 (1024维)
- Reranker: BAAI/bge-reranker-v2-m3
- LLM: Kimi (kimi-k2) / Qwen (qwen3.5)

---
## 项目结构

```
RAG_1/
├── src/
│   ├── api/           # FastAPI 接口
│   ├── ingestion/     # 文档解析 + 切块
│   ├── embedding/     # bge-m3 向量化
│   ├── storage/       # ChromaDB + SQLite
│   ├── retrieval/     # 检索 + Reranker
│   ├── generation/    # QA Chain + LLM
│   └── evaluation/    # Ragas 评估
├── scripts/           # 入口脚本
├── tests/             # 测试套件（四层金字塔）
│   ├── unit/          # 单元测试
│   ├── smoke/         # 组件/冒烟测试
│   ├── integration/   # 集成测试
│   └── e2e/           # 端到端测试
├── config/            # 配置参数
└── data/              # 数据目录
```
---

## 快速开始

### 环境准备

```bash
# 1. 克隆项目
git clone <repo-url>
cd RAG_1

# 2. 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate  # Windows

# 3. 安装依赖
pip install -e .

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 MOONSHOT_API_KEY 或 DASHSCOPE_API_KEY
```

### 导入文档

```bash
# 导入单个 PDF
python scripts/ingest.py --file "data/raw/年报.pdf"

# 导入全部文档
python scripts/ingest.py
```

### 启动服务

```bash
# CLI 交互式问答
python cli.py

# 或启动 FastAPI 服务
uvicorn src.api.main:app --reload
# 访问 http://localhost:8000/docs
```
## 评估命令

```bash
# 生成合成 QA
python scripts/evaluate.py --mode generate --num 50

# 运行评估
python scripts/evaluate.py --mode manual   # 手工评估
python scripts/evaluate.py --mode full     # 完整评估
```

## 测试体系

采用四层测试金字塔模型：

```
┌─────────────────────────────────────────┐
│  E2E Tests (端到端测试)                  │  tests/e2e/
│  - 完整用户场景验证                      │
├─────────────────────────────────────────┤
│  Integration Tests (集成测试)            │  tests/integration/
│  - 多组件协作 + 真实数据库交互            │
├─────────────────────────────────────────┤
│  Component Tests (组件/冒烟测试)         │  tests/smoke/
│  - 单组件完整功能验证                    │
├─────────────────────────────────────────┤
│  Unit Tests (单元测试)                   │  tests/unit/
│  - 函数/类级别测试                       │
└─────────────────────────────────────────┘
```

### 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定层级
pytest tests/unit -v
pytest tests/smoke -v
pytest tests/integration -v -m "not slow"
pytest tests/e2e -v

# 运行特定测试文件
pytest tests/integration/test_qa_pipeline.py -v

# 运行特定测试用例
pytest tests/integration/test_qa_pipeline.py::TestQAPipeline::test_two_step_generation -v

# 覆盖率报告
pytest tests/unit tests/integration --cov=src --cov-report=html
```

---

## Tech Stack

- Python 3.12 / FastAPI / Uvicorn
- 文档解析: pdfplumber, python-docx, python-pptx, openpyxl
- Chunking: LlamaIndex SentenceSplitter
- Embedding: sentence-transformers (BAAI/bge-m3)
- Vector DB: ChromaDB
- Reranker: BAAI/bge-reranker-v2-m3
- LLM: Kimi / Qwen (OpenAI 兼容 API)
- Evaluation: Ragas 0.4.x
