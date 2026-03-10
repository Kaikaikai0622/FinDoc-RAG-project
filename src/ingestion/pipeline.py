"""Ingestion Pipeline 模块

串联 PDF 解析、文档切块、Embedding、存储的完整流程。
"""
import logging
import time
from typing import List, Dict, Any

from src.ingestion.document_router import DocumentRouter
from src.ingestion.chunker import Chunker
from src.embedding import BaseEmbeddingService, BGEm3EmbeddingService
from src.storage import DocStore, VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """文档摄取管道

    完整流程：PDF 解析 → 切块 → Embedding → 存储
    """

    def __init__(
        self,
        embedding_service: BaseEmbeddingService | None = None,
        doc_store: DocStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """初始化 Ingestion Pipeline

        Args:
            embedding_service: Embedding 服务实例
            doc_store: 文档存储实例
            vector_store: 向量存储实例
        """
        self.router = DocumentRouter()
        self.chunker = Chunker()
        self.embedding_service = embedding_service or BGEm3EmbeddingService()
        self.doc_store = doc_store or DocStore()
        self.vector_store = vector_store or VectorStore()

    def run(self, file_path: str = None, *, pdf_path: str = None) -> Dict[str, Any]:
        """运行完整的 ingestion 流程

        Args:
            file_path: 文件路径（支持 .pdf / .docx / .pptx / .xlsx / .txt / .md / .csv）
            pdf_path: 向后兼容参数，等价于 file_path

        Returns:
            统计信息字典
        """
        # 向后兼容：支持旧版 pdf_path 关键字参数
        file_path = file_path or pdf_path
        if not file_path:
            raise ValueError("必须提供 file_path 参数")

        start_time = time.time()

        # 步骤 1: 通过 DocumentRouter 解析文件 → ParsedDocument
        logger.info(f"开始解析文件: {file_path}")
        doc = self.router.route(file_path)
        page_count = doc.page_count
        logger.info(f"解析完成，共 {page_count} 页，{len(doc.elements)} 个元素")

        # 步骤 2: 切分文档
        logger.info("开始切分文档...")
        chunks = self.chunker.chunk_document(doc)
        logger.info(f"切分完成，共 {len(chunks)} 个 chunks")

        # 步骤 3: 计算 Embedding
        logger.info("开始计算 Embedding...")
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self.embedding_service.embed(chunk_texts)
        logger.info(f"Embedding 计算完成，向量维度: {len(embeddings[0]) if embeddings else 0}")

        # 步骤 4: 存储到 DocStore (SQLite)
        logger.info("开始存储到 DocStore...")
        self.doc_store.save_chunks(chunks)
        doc_store_count = self.doc_store.count()
        logger.info(f"DocStore 存储完成，当前共 {doc_store_count} 个 chunks")

        # 步骤 5: 存储到 VectorStore (Chroma)
        logger.info("开始存储到 VectorStore...")
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        self.vector_store.add_embeddings(chunk_ids, embeddings, metadatas)
        vector_store_count = self.vector_store.count()
        logger.info(f"VectorStore 存储完成，当前共 {vector_store_count} 个向量")

        # 统计耗时
        elapsed_time = time.time() - start_time

        result = {
            "file_path": file_path,
            "pdf_path": file_path,  # 向后兼容
            "page_count": page_count,
            "element_count": len(doc.elements),
            "chunk_count": len(chunks),
            "embedding_dim": self.embedding_service.get_dimension(),
            "doc_store_count": doc_store_count,
            "vector_store_count": vector_store_count,
            "elapsed_time": elapsed_time,
        }

        logger.info(f"Pipeline 执行完成，耗时 {elapsed_time:.2f} 秒")

        return result

    def run_batch(self, file_paths: List[str] = None, *, pdf_paths: List[str] = None) -> List[Dict[str, Any]]:
        """批量处理多个文件

        Args:
            file_paths: 文件路径列表
            pdf_paths: 向后兼容参数，等价于 file_paths

        Returns:
            每个文件的统计信息列表
        """
        file_paths = file_paths or pdf_paths or []
        results = []
        for file_path in file_paths:
            try:
                result = self.run(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                results.append({
                    "file_path": file_path,
                    "pdf_path": file_path,  # 向后兼容
                    "error": str(e),
                })
        return results
