"""SQLite 文档存储模块

存储 chunk 原文和元信息，与向量库分离。
后期换 embedding 模型时，从 SQLite 取原文重新计算向量，不需要重新解析 PDF。
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import SQLITE_DB_PATH


class DocStore:
    """SQLite 文档存储类

    用于存储 chunk 的原文和元信息。
    表结构：
    - chunk_id: 主键，chunk 唯一标识
    - chunk_text: chunk 原文内容
    - source_file: 来源文件名
    - page_number: 源 PDF 页码
    - chunk_index: 在文档中的 chunk 序号
    - created_at: 创建时间戳
    """

    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        """初始化 DocStore

        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self.init_db()

    def _ensure_db_dir(self) -> None:
        """确保数据库目录存在"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 支持列名访问
        return conn

    def init_db(self) -> None:
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                source_file TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                element_category TEXT DEFAULT 'Text',
                table_title TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 迁移：为旧库补充新列（SQLite 不支持 IF NOT EXISTS，通过捕获异常处理）
        for col, definition in [
            ("element_category", "TEXT DEFAULT 'Text'"),
            ("table_title",      "TEXT DEFAULT ''"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col} {definition}")
            except Exception:
                pass  # 列已存在，忽略

        conn.commit()
        conn.close()

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """批量写入 chunks

        Args:
            chunks: chunk 列表，每项包含 chunk_id, chunk_text, metadata
        """
        if not chunks:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        data = [
            (
                chunk["chunk_id"],
                chunk["chunk_text"],
                chunk["metadata"]["source_file"],
                chunk["metadata"]["page_number"],
                chunk["metadata"]["chunk_index"],
                chunk["metadata"].get("element_category", "Text"),
                chunk["metadata"].get("table_title", ""),
                now,
            )
            for chunk in chunks
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, chunk_text, source_file, page_number, chunk_index,
             element_category, table_title, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data
        )

        conn.commit()
        conn.close()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """根据 chunk_id 查询单个 chunk

        Args:
            chunk_id: chunk 唯一标识

        Returns:
            chunk 信息字典，不存在则返回 None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_id, chunk_text, source_file, page_number, chunk_index,
                   element_category, table_title, created_at
            FROM chunks WHERE chunk_id = ?
            """,
            (chunk_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return {
            "chunk_id": row["chunk_id"],
            "chunk_text": row["chunk_text"],
            "metadata": {
                "source_file": row["source_file"],
                "page_number": row["page_number"],
                "chunk_index": row["chunk_index"],
                "element_category": row["element_category"] or "Text",
                "table_title": row["table_title"] or "",
            },
            "created_at": row["created_at"],
        }

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """查询所有 chunks

        Returns:
            所有 chunk 信息列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_id, chunk_text, source_file, page_number, chunk_index,
                   element_category, table_title, created_at
            FROM chunks ORDER BY source_file, chunk_index
            """
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "chunk_id": row["chunk_id"],
                "chunk_text": row["chunk_text"],
                "metadata": {
                    "source_file": row["source_file"],
                    "page_number": row["page_number"],
                    "chunk_index": row["chunk_index"],
                    "element_category": row["element_category"] or "Text",
                    "table_title": row["table_title"] or "",
                },
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def count(self) -> int:
        """统计 chunk 总数

        Returns:
            chunk 数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]

        conn.close()
        return count
