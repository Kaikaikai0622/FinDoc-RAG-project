"""API 集成测试

使用FastAPI TestClient测试HTTP接口

覆盖场景：
- 健康检查接口
- 问答接口
- 参数验证错误
- 空问题处理
- 根路径接口
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestAPI:
    """API接口集成测试"""

    @pytest.fixture(scope="class")
    def test_client(self):
        """创建测试客户端"""
        from src.api.main import app
        return TestClient(app)

    def test_root_endpoint(self, test_client):
        """测试根路径接口"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert data["docs"] == "/docs"

    def test_health_endpoint(self, test_client):
        """测试健康检查接口"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "chunks_count" in data
        assert "llm_provider" in data

    def test_query_endpoint_success(self, test_client):
        """测试问答接口成功场景"""
        # 模拟QAChain的ask方法
        mock_result = {
            "question": "测试问题",
            "answer": "这是一个测试答案",
            "sources": [
                {"file": "test.pdf", "page": 1, "score": 0.95}
            ],
            "chunks_used": 1,
            "retrieval_time": 0.5,
            "generation_time": 1.0,
            "total_time": 1.5,
            "mode": "single_step",
            "filter_used": None,
            "filter_auto": False,
        }

        with patch('src.api.main.qa_chain') as mock_qa:
            mock_qa.ask.return_value = mock_result

            response = test_client.post("/query", json={
                "question": "测试问题",
                "filter_file": None
            })

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "测试问题"
        assert data["answer"] == "这是一个测试答案"
        assert data["chunks_used"] == 1
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file"] == "test.pdf"

    def test_query_endpoint_with_filter(self, test_client):
        """测试问答接口带filter_file"""
        mock_result = {
            "question": "测试问题",
            "answer": "答案",
            "sources": [],
            "chunks_used": 0,
            "retrieval_time": 0.1,
            "generation_time": 0.2,
            "total_time": 0.3,
            "mode": "single_step",
            "filter_used": "中兴",
            "filter_auto": False,
        }

        with patch('src.api.main.qa_chain') as mock_qa:
            mock_qa.ask.return_value = mock_result

            response = test_client.post("/query", json={
                "question": "测试问题",
                "filter_file": "中兴"
            })

        assert response.status_code == 200
        # 验证filter_file被正确传递
        mock_qa.ask.assert_called_once_with("测试问题", filter_file="中兴")

    def test_query_validation_error_empty_question(self, test_client):
        """测试参数验证错误 - 空问题"""
        response = test_client.post("/query", json={
            "question": ""
        })
        assert response.status_code == 400
        assert "不能为空" in response.json()["detail"]

    def test_query_validation_error_whitespace_question(self, test_client):
        """测试参数验证错误 - 仅空白字符的问题"""
        response = test_client.post("/query", json={
            "question": "   "
        })
        assert response.status_code == 400
        assert "不能为空" in response.json()["detail"]

    def test_query_validation_error_missing_question(self, test_client):
        """测试参数验证错误 - 缺少question字段"""
        response = test_client.post("/query", json={
            "filter_file": "test"
        })
        # FastAPI会验证Pydantic模型，缺少必填字段会返回422
        assert response.status_code == 422

    def test_query_endpoint_error_handling(self, test_client):
        """测试问答接口错误处理"""
        with patch('src.api.main.qa_chain') as mock_qa:
            mock_qa.ask.side_effect = Exception("内部错误")

            response = test_client.post("/query", json={
                "question": "测试问题"
            })

        assert response.status_code == 500
        assert "处理失败" in response.json()["detail"]

    def test_query_response_model(self, test_client):
        """测试问答响应模型完整性"""
        mock_result = {
            "question": "测试",
            "answer": "答案",
            "sources": [
                {"file": "a.pdf", "page": 1, "score": 0.9, "rerank_score": 0.95}
            ],
            "chunks_used": 1,
            "retrieval_time": 0.1,
            "generation_time": 0.2,
            "total_time": 0.3,
            "mode": "single_step",
            "filter_used": None,
            "filter_auto": False,
        }

        with patch('src.api.main.qa_chain') as mock_qa:
            mock_qa.ask.return_value = mock_result

            response = test_client.post("/query", json={
                "question": "测试"
            })

        assert response.status_code == 200
        data = response.json()

        # 验证所有预期字段存在
        expected_fields = ["question", "answer", "sources", "chunks_used",
                          "retrieval_time", "generation_time"]
        for field in expected_fields:
            assert field in data, f"响应缺少字段: {field}"

        # 验证score被正确四舍五入
        assert data["sources"][0]["score"] == 0.9
