# FinDoc-RAG 测试套件

本目录包含 FinDoc-RAG 项目的完整测试体系，采用四层测试金字塔模型。

## 测试分层

```
┌─────────────────────────────────────────┐
│  Layer 4: E2E Tests (端到端测试)         │  tests/e2e/
│  - 完整用户场景验证                      │
│  - 可能需要外部服务(LLM)                 │
├─────────────────────────────────────────┤
│  Layer 3: Integration Tests (集成测试)   │  tests/integration/
│  - 多组件协作验证                        │
│  - 真实数据库交互                        │
├─────────────────────────────────────────┤
│  Layer 2: Component Tests (组件测试)     │  tests/smoke/
│  - 单组件完整功能验证                    │
│  - 现有冒烟测试属于此层                  │
├─────────────────────────────────────────┤
│  Layer 1: Unit Tests (单元测试)          │  tests/unit/
│  - 函数/类级别测试                       │
│  - 大量使用Mock                          │
└─────────────────────────────────────────┘
```

## 目录结构

```
tests/
├── README.md                    # 本文件
├── unit/                        # 单元测试
│   ├── __init__.py
│   ├── test_document_router.py  # 文档路由单元测试
│   └── check_imports.py         # 导入检查
├── smoke/                       # 冒烟测试（组件测试）
│   ├── __init__.py
│   ├── test_chunker_fixes.py    # Chunker修复测试
│   ├── test_company_resolver.py # 公司名称解析测试
│   ├── test_comparison_two_step.py  # 两步生成测试
│   ├── test_defensive_check.py  # 防御性检查测试
│   ├── test_embedding.py        # Embedding服务测试
│   ├── test_filter_file.py      # 文件过滤测试
│   ├── test_ragas_cols.py       # RAGAS列测试
│   └── test_reranker.py         # Reranker测试
├── integration/                 # 集成测试
│   ├── __init__.py
│   ├── conftest.py              # 共享fixtures
│   ├── fixtures/                # 测试数据
│   │   ├── README.md
│   │   ├── sample_pdfs/         # PDF样本
│   │   ├── sample_qa/           # QA数据
│   │   └── expected_outputs/    # 预期输出
│   ├── test_ingestion_pipeline.py   # 摄取管道测试
│   ├── test_retrieval_pipeline.py   # 检索管道测试
│   ├── test_qa_pipeline.py          # QA管道测试
│   ├── test_evaluation_pipeline.py  # 评估管道测试
│   └── test_api.py                  # API接口测试
└── e2e/                         # 端到端测试
    ├── __init__.py
    └── test_complete_workflow.py    # 完整工作流测试
```

## 运行测试

### 运行所有测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定层级
pytest tests/unit -v
pytest tests/smoke -v
pytest tests/integration -v
pytest tests/e2e -v
```

### 运行特定测试文件

```bash
# 单元测试
pytest tests/unit/test_document_router.py -v

# 冒烟测试
python tests/smoke/test_chunker_fixes.py

# 集成测试
pytest tests/integration/test_qa_pipeline.py -v

# E2E测试
pytest tests/e2e/test_complete_workflow.py -v
```

### 运行特定测试用例

```bash
# 运行特定类
pytest tests/integration/test_qa_pipeline.py::TestQAPipeline -v

# 运行特定方法
pytest tests/integration/test_qa_pipeline.py::TestQAPipeline::test_two_step_generation_for_comparison -v

# 按关键词匹配
pytest tests/integration -v -k "two_step"
```

### 覆盖率报告

```bash
# 生成覆盖率报告
pytest tests/unit tests/integration --cov=src --cov-report=html --cov-report=term

# 打开HTML报告
open htmlcov/index.html
```

## 测试环境配置

### 本地测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 或者手动安装
pip install pytest pytest-asyncio pytest-cov httpx
```

### CI环境

CI使用GitHub Actions配置（见 `.github/workflows/ci.yml`）：

1. **Lint阶段**: 代码风格检查
2. **Unit Tests阶段**: 单元测试
3. **Smoke Tests阶段**: 快速冒烟测试
4. **Integration Tests阶段**: 集成测试
5. **E2E Tests阶段**: 端到端测试
6. **Coverage阶段**: 覆盖率报告

## 测试数据

### 样本PDF

集成测试使用 `data/raw/` 目录下的真实PDF文件，或 `tests/integration/fixtures/sample_pdfs/` 下的专用测试样本。

### Fixtures

共享fixtures定义在 `tests/integration/conftest.py`：

- `isolated_storage`: 隔离的SQLite和ChromaDB存储
- `sample_pdf_path`: 样本PDF路径
- `mock_llm_service`: Mock LLM服务
- `populated_storage`: 预填充测试数据的存储
- `retriever_with_data`: 配置了预填充数据的Retriever
- `qa_chain_with_mock_llm`: 使用Mock LLM的QAChain

## 编写新测试

### 单元测试示例

```python
# tests/unit/test_new_feature.py
import pytest
from src.module import NewFeature

class TestNewFeature:
    def test_basic_functionality(self):
        feature = NewFeature()
        result = feature.process("input")
        assert result == "expected"
```

### 集成测试示例

```python
# tests/integration/test_new_pipeline.py
import pytest

class TestNewPipeline:
    def test_pipeline_e2e(self, isolated_storage):
        from src.pipeline import NewPipeline
        from src.storage import DocStore

        pipeline = NewPipeline()
        pipeline.doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
        result = pipeline.run()

        assert result["success"] is True
```

## 注意事项

1. **隔离性**: 每个测试应该独立运行，不依赖其他测试的状态
2. **清理**: 使用fixtures自动清理临时资源
3. **Mock外部服务**: 避免在测试中调用真实LLM API
4. **测试数据**: 不要将大文件(>1MB)提交到git
5. **命名规范**: 测试方法名应描述被测试的行为

## 调试技巧

```bash
# 详细输出
pytest tests/integration -v -s

# 遇到第一个错误停止
pytest tests/integration -x

# 失败时进入pdb
pytest tests/integration --pdb

# 只运行上次失败的测试
pytest tests/integration --lf
```
