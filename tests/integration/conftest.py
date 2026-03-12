"""Integration Tests 共享 Fixtures

为集成测试提供隔离的存储环境、Mock服务和样本数据。
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return TEST_DATA_DIR


@pytest.fixture(scope="function")
def isolated_storage(tmp_path):
    """
    为每个测试提供隔离的存储环境

    返回:
        dict: 包含 sqlite_path 和 chroma_path 的字典
    """
    sqlite_path = tmp_path / "test_doc_store.db"
    chroma_path = tmp_path / "test_chroma_db"

    yield {
        "sqlite_path": str(sqlite_path),
        "chroma_path": str(chroma_path),
        "tmp_path": tmp_path,
    }

    # 清理：测试完成后删除临时目录
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_pdf_path(test_data_dir):
    """样本PDF路径

    优先使用 fixtures/sample_pdfs/ 下的样本文件，
    如果不存在则使用 data/raw/ 下的第一个PDF。
    """
    # 首先检查专用测试样本
    sample_dir = test_data_dir / "sample_pdfs"
    if sample_dir.exists():
        samples = list(sample_dir.glob("*.pdf"))
        if samples:
            return str(samples[0])

    # 回退到 data/raw/
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        pdfs = list(raw_dir.glob("*.pdf"))
        if pdfs:
            return str(pdfs[0])

    # 如果没有PDF，返回None，测试应该跳过
    pytest.skip("没有找到可用的PDF样本文件")


@pytest.fixture(scope="function")
def mock_llm_service():
    """Mock LLM服务

    模拟LLM响应，避免在集成测试中调用真实API。
    """
    mock = MagicMock()
    mock.chat.return_value = "这是一个测试答案。根据文档内容，相关信息如下..."
    return mock


@pytest.fixture(scope="function")
def populated_storage(isolated_storage, sample_pdf_path):
    """
    预填充测试数据的存储

    使用真实的IngestionPipeline摄取样本PDF，
    为检索和QA测试提供数据基础。
    """
    from src.ingestion.pipeline import IngestionPipeline
    from src.storage import DocStore, VectorStore

    # 创建隔离存储实例
    doc_store = DocStore(db_path=isolated_storage["sqlite_path"])
    vector_store = VectorStore(persist_dir=isolated_storage["chroma_path"])

    # 创建并配置Pipeline
    pipeline = IngestionPipeline(
        doc_store=doc_store,
        vector_store=vector_store,
    )

    # 执行摄取
    result = pipeline.run(sample_pdf_path)

    # 验证摄取成功
    assert result["chunk_count"] > 0, "样本PDF摄取失败，没有生成chunks"

    return {
        **isolated_storage,
        "ingestion_result": result,
    }


@pytest.fixture(scope="function")
def retriever_with_data(populated_storage):
    """
    配置了预填充数据的Retriever实例
    """
    from src.retrieval.retriever import Retriever
    from src.storage import DocStore, VectorStore

    doc_store = DocStore(db_path=populated_storage["sqlite_path"])
    vector_store = VectorStore(persist_dir=populated_storage["chroma_path"])

    return Retriever(
        doc_store=doc_store,
        vector_store=vector_store,
    )


@pytest.fixture(scope="function")
def qa_chain_with_mock_llm(populated_storage, mock_llm_service):
    """
    使用Mock LLM的QAChain

    使用预填充数据的存储，但使用Mock LLM避免API调用。
    """
    from src.generation.qa_chain import QAChain
    from src.retrieval.retriever import Retriever
    from src.storage import DocStore, VectorStore

    doc_store = DocStore(db_path=populated_storage["sqlite_path"])
    vector_store = VectorStore(persist_dir=populated_storage["chroma_path"])

    retriever = Retriever(
        doc_store=doc_store,
        vector_store=vector_store,
    )

    return QAChain(
        retriever=retriever,
        llm_service=mock_llm_service,
    )
