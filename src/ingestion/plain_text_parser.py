"""纯文本解析模块

处理 .txt / .md / .csv 三种格式，零第三方依赖（全标准库）。
- TXT/MD：按 line_chunk_size 行分段 → category="Text"
- CSV：用 csv.DictReader 按行分组 → category="SheetRow"
"""
import csv
import logging
from pathlib import Path
from typing import List

from .models import ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)

LINE_CHUNK_SIZE = 30   # TXT/MD 每段行数
CSV_ROW_CHUNK_SIZE = 15  # CSV 每组行数


class PlainTextParser:
    """纯文本解析器

    支持 .txt / .md / .csv，返回统一的 ParsedDocument。
    """

    def __init__(self, line_chunk_size: int = LINE_CHUNK_SIZE,
                 csv_row_chunk_size: int = CSV_ROW_CHUNK_SIZE) -> None:
        self.line_chunk_size = line_chunk_size
        self.csv_row_chunk_size = csv_row_chunk_size

    def parse(self, file_path: str) -> ParsedDocument:
        """解析纯文本文件

        Args:
            file_path: 文件路径

        Returns:
            ParsedDocument 对象
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        source_file = path.name

        if suffix == ".csv":
            return ParsedDocument(
                source_file=source_file,
                file_type="csv",
                elements=self._parse_csv(file_path),
            )
        elif suffix == ".md":
            return ParsedDocument(
                source_file=source_file,
                file_type="md",
                elements=self._parse_text(file_path),
            )
        else:
            return ParsedDocument(
                source_file=source_file,
                file_type="txt",
                elements=self._parse_text(file_path),
            )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _parse_text(self, file_path: str) -> List[ParsedElement]:
        """解析 .txt / .md 文件，按 line_chunk_size 行分段"""
        elements: List[ParsedElement] = []

        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        chunk_index = 0
        for start in range(0, len(lines), self.line_chunk_size):
            end = min(start + self.line_chunk_size, len(lines))
            chunk_lines = lines[start:end]
            text = "".join(chunk_lines).strip()
            if not text:
                continue

            elements.append(ParsedElement(
                text=text,
                category="Text",
                page_or_index=chunk_index,
                metadata={
                    "line_range": f"{start + 1}-{end}",
                },
            ))
            chunk_index += 1

        return elements

    def _parse_csv(self, file_path: str) -> List[ParsedElement]:
        """解析 .csv 文件，按 csv_row_chunk_size 行分组"""
        elements: List[ParsedElement] = []

        with open(file_path, encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            column_names = reader.fieldnames or []
            rows = list(reader)

        if not rows:
            return elements

        col_str = ",".join(str(c) for c in column_names)
        chunk_index = 0

        for start in range(0, len(rows), self.csv_row_chunk_size):
            end = min(start + self.csv_row_chunk_size, len(rows))
            chunk_rows = rows[start:end]

            # 序列化：header行 + 每行数据
            lines = [" | ".join(str(c) for c in column_names)]
            for row in chunk_rows:
                lines.append(" | ".join(str(row.get(c, "")) for c in column_names))
            text = "\n".join(lines)

            elements.append(ParsedElement(
                text=text,
                category="SheetRow",
                page_or_index=chunk_index,
                metadata={
                    "row_index": start + 1,
                    "column_names": col_str,
                },
            ))
            chunk_index += 1

        return elements


def parse_plain_text(file_path: str) -> ParsedDocument:
    """解析纯文本文件的便捷函数"""
    parser = PlainTextParser()
    return parser.parse(file_path)

