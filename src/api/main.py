"""FastAPI 问答接口"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

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


class QueryResponse(BaseModel):
    """问答响应"""
    question: str
    answer: str
    sources: List[SourceInfo]
    chunks_used: int
    retrieval_time: float
    generation_time: float


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
        问答响应，包含答案和引用来源
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

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            chunks_used=result["chunks_used"],
            retrieval_time=result["retrieval_time"],
            generation_time=result["generation_time"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
