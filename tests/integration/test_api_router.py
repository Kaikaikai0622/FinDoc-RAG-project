"""API Router 集成测试

验证 API 响应包含 Router 相关字段：
1. 旧响应字段仍存在
2. 新字段为 additive
3. 默认不返回原始 chunk 文本
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_qa_chain():
    """Mock QAChain 返回包含 Router 字段的结果"""
    with patch('src.api.main.qa_chain') as mock_chain:
        mock_chain.ask.return_value = {
            "question": "测试问题",
            "answer": "测试答案",
            "sources": [
                {"file": "test.pdf", "page": 1, "score": 0.95}
            ],
            "chunks_used": 1,
            "retrieval_time": 0.5,
            "generation_time": 1.0,
            "total_time": 1.5,
            "mode": "single_step",
            "filter_used": "test.pdf",
            "filter_auto": False,
            # Router 新增字段
            "route_label": "factual",
            "retrieval_mode": "filtered",
            "fallback_triggered": False,
            "query_classifier": {
                "scene": "factual",
                "generation_mode": "single_step",
                "filter_source": "explicit",
                "retrieval_scope": "single_company",
                "confidence": 0.9,
                "reason_codes": ["test"],
            },
        }
        yield mock_chain


def test_api_response_has_old_fields(mock_qa_chain):
    """API 响应包含所有旧字段"""
    from src.api.main import app
    client = TestClient(app)

    response = client.post("/query", json={"question": "测试"})

    assert response.status_code == 200
    data = response.json()

    # 旧字段
    assert "question" in data
    assert "answer" in data
    assert "sources" in data
    assert "chunks_used" in data
    assert "retrieval_time" in data
    assert "generation_time" in data


def test_api_response_has_new_router_fields(mock_qa_chain):
    """API 响应包含 Router 新字段"""
    from src.api.main import app
    client = TestClient(app)

    response = client.post("/query", json={"question": "测试"})

    assert response.status_code == 200
    data = response.json()

    # 新增字段
    assert "total_time" in data
    assert "mode" in data
    assert "filter_used" in data
    assert "filter_auto" in data
    assert "route_label" in data
    assert "retrieval_mode" in data
    assert "fallback_triggered" in data
    assert "query_classifier" in data


def test_api_response_no_raw_chunks(mock_qa_chain):
    """API 默认不返回原始 chunk 文本"""
    from src.api.main import app
    client = TestClient(app)

    response = client.post("/query", json={"question": "测试"})

    assert response.status_code == 200
    data = response.json()

    # 不应包含原始 chunk 文本
    assert "retrieved_context" not in data
    assert "chunks" not in data


def test_api_query_classifier_structure(mock_qa_chain):
    """query_classifier 字段结构正确"""
    from src.api.main import app
    client = TestClient(app)

    response = client.post("/query", json={"question": "测试"})

    assert response.status_code == 200
    data = response.json()

    qc = data.get("query_classifier")
    assert qc is not None
    assert "scene" in qc
    assert "generation_mode" in qc
    assert "filter_source" in qc
    assert "retrieval_scope" in qc
    assert "confidence" in qc
    # reason_codes 不暴露（内部调试用）
    assert "reason_codes" not in qc


def test_api_empty_question_returns_400(mock_qa_chain):
    """空问题返回 400"""
    from src.api.main import app
    client = TestClient(app)

    response = client.post("/query", json={"question": ""})

    assert response.status_code == 400


def test_api_backward_compatible_when_no_router(mock_qa_chain):
    """当 Router 未启用时，新字段为 null/None"""
    from src.api.main import app
    client = TestClient(app)

    # 模拟无 Router 字段的情况
    mock_qa_chain.ask.return_value = {
        "question": "测试",
        "answer": "答案",
        "sources": [],
        "chunks_used": 0,
        "retrieval_time": 0.1,
        "generation_time": 0.2,
        "total_time": 0.3,
        "mode": "single_step",
        "filter_used": None,
        "filter_auto": False,
        # 无 Router 字段
    }

    response = client.post("/query", json={"question": "测试"})

    assert response.status_code == 200
    data = response.json()

    # 旧字段存在
    assert data["question"] == "测试"
    # 新字段为 None
    assert data.get("route_label") is None
    assert data.get("retrieval_mode") is None
