#!/usr/bin/env python
"""命令行问答交互工具"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import QAChain
from src.storage import DocStore
from config import LLM_PROVIDER, EMBEDDING_MODEL, TOP_K, USE_RERANKER, RETRIEVAL_TOP_K, RERANK_TOP_K


def print_config() -> None:
    """打印当前配置信息"""
    print("=" * 60)
    print("FinDoc-RAG 问答系统")
    print("=" * 60)
    doc_store = DocStore()
    chunk_count = doc_store.count()
    print(f"LLM Provider:    {LLM_PROVIDER}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    if USE_RERANKER:
        print(f"检索模式:        RerankRetriever（粗检索 {RETRIEVAL_TOP_K} → 精排 {RERANK_TOP_K}）")
    else:
        print(f"检索模式:        Retriever Baseline（Top-K={TOP_K}）")
    print(f"向量库 Chunk 数量: {chunk_count}")
    print("=" * 60)
    print()


def print_result(result: dict) -> None:
    """打印问答结果"""
    print("\n" + "=" * 60)
    print("【问题】")
    print(result["question"])

    print("\n【回答】")
    print(result["answer"])

    print("\n【引用来源】")
    for i, source in enumerate(result["sources"], 1):
        rerank_info = ""
        if "rerank_score" in source:
            rerank_info = f", 精排分: {source['rerank_score']:.4f}"
        print(f"  {i}. {source['file']}, 第{source['page']}页 (向量相似度: {source['score']:.3f}{rerank_info})")

    print("\n【统计信息】")
    print(f"  使用 chunk 数: {result['chunks_used']}")
    print(f"  检索耗时: {result['retrieval_time']:.3f}秒")
    print(f"  生成耗时: {result['generation_time']:.3f}秒")
    print(f"  总耗时: {result['total_time']:.3f}秒")
    print("=" * 60)


def main() -> None:
    """主函数"""
    print_config()
    print("初始化组件...")
    qa_chain = QAChain()  # 内部自动根据 USE_RERANKER 选择检索器
    mode = "RerankRetriever" if USE_RERANKER else "Retriever (Baseline)"
    print(f"初始化完成！检索器: {mode}\n")

    print("请输入问题（输入 quit 退出）：")
    print("-" * 60)

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break

        try:
            result = qa_chain.ask(question)
            print_result(result)
        except Exception as e:
            print(f"\nError: {e}")
            print("请尝试其他问题或检查配置。\n")


if __name__ == "__main__":
    main()
