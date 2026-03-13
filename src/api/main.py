"""FastAPI 问答接口"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.generation import QAChain
from src.storage import DocStore
from config import LLM_PROVIDER

# 创建 FastAPI 应用
app = FastAPI(
    title="FinDoc-RAG API",
    description="财务报表 RAG 问答系统接口",
    version="0.1.0",
)

# 初始化 QA Chain
qa_chain = QAChain()


class QueryRequest(BaseModel):
    """问答请求"""
    question: str
    filter_file: str | None = None  # 按来源文件名过滤（支持部分匹配）


class SourceInfo(BaseModel):
    """来源信息"""
    file: str
    page: int
    score: float


class QueryClassifierInfo(BaseModel):
    """查询分类器信息（轻量）"""
    scene: str
    generation_mode: str
    filter_source: str
    retrieval_scope: str
    confidence: float


class QueryResponse(BaseModel):
    """问答响应"""
    question: str
    answer: str
    sources: List[SourceInfo]
    chunks_used: int
    retrieval_time: float
    generation_time: float
    total_time: float  # 新增
    mode: str  # 新增: single_step / two_step
    filter_used: str | None  # 新增
    filter_auto: bool  # 新增
    route_label: str | None = None  # 新增: Router 场景标签
    retrieval_mode: str | None = None  # 新增: filtered / global / filtered_then_global
    fallback_triggered: bool | None = None  # 新增: 是否触发回退
    query_classifier: QueryClassifierInfo | None = None  # 新增: 分类器详情


@app.get("/")
def root():
    """根路径"""
    return {"message": "FinDoc-RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    """健康检查接口"""
    doc_store = DocStore()
    chunks_count = doc_store.count()

    return {
        "status": "ok",
        "chunks_count": chunks_count,
        "llm_provider": LLM_PROVIDER,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """问答接口

    Args:
        request: 问答请求，包含问题和可选的过滤条件

    Returns:
        问答响应，包含答案和引用来源，以及 Router 元信息（如果启用）
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = qa_chain.ask(request.question, filter_file=request.filter_file)

        # 转换 sources 格式
        sources = [
            SourceInfo(
                file=source["file"],
                page=source["page"],
                score=round(source["score"], 3),
            )
            for source in result["sources"]
        ]

        # 构建基础响应（始终存在的字段）
        response_data = {
            "question": result["question"],
            "answer": result["answer"],
            "sources": sources,
            "chunks_used": result["chunks_used"],
            "retrieval_time": result["retrieval_time"],
            "generation_time": result["generation_time"],
            "total_time": result.get("total_time", result["retrieval_time"] + result["generation_time"]),
            "mode": result["mode"],
            "filter_used": result["filter_used"],
            "filter_auto": result["filter_auto"],
        }

        # 添加 Router 相关字段（如果存在）
        if "route_label" in result:
            response_data["route_label"] = result["route_label"]
        if "retrieval_mode" in result:
            response_data["retrieval_mode"] = result["retrieval_mode"]
        if "fallback_triggered" in result:
            response_data["fallback_triggered"] = result["fallback_triggered"]

        # 添加 query_classifier 信息（如果存在）
        if "query_classifier" in result:
            qc = result["query_classifier"]
            response_data["query_classifier"] = QueryClassifierInfo(
                scene=qc["scene"],
                generation_mode=qc["generation_mode"],
                filter_source=qc["filter_source"],
                retrieval_scope=qc["retrieval_scope"],
                confidence=qc["confidence"],
            )

        return QueryResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
