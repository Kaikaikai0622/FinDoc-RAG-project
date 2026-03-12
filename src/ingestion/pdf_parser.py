"""PDF 解析模块

使用 pdfplumber 解析 PDF，提取文本内容和页码。
表格元素转为 Markdown 格式文本。

新增功能：
  - 从表格第一行自动提取 column_names，注入 metadata
  - 跨页同结构表格自动合并为一个逻辑 chunk
"""
from pathlib import Path
from typing import List
import logging
import re

from .models import ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.error("pdfplumber 库不可用，无法解析 PDF")


class PDFParser:
    """PDF 解析器

    使用 pdfplumber 解析 PDF 文件，返回统一的 ParsedDocument。
    """

    def __init__(self) -> None:
        """初始化解析器"""
        pass

    def parse(self, pdf_path: str) -> ParsedDocument:
        """解析 PDF 文件

        Args:
            pdf_path: PDF 文件路径

        Returns:
            ParsedDocument 对象
        """
        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("pdfplumber 库不可用，请安装 pdfplumber")

        source_file = Path(pdf_path).name
        file_type = Path(pdf_path).suffix.lower().lstrip(".")

        elements = self._parse_with_pdfplumber(pdf_path)

        return ParsedDocument(
            source_file=source_file,
            file_type=file_type,
            elements=elements,
        )

    def _parse_with_pdfplumber(self, pdf_path: str) -> List[ParsedElement]:
        """使用 pdfplumber 解析 PDF

        Args:
            pdf_path: PDF 文件路径

        Returns:
            ParsedElement 列表（跨页表格已合并）
        """
        results: List[ParsedElement] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 提取文本
                text = page.extract_text()
                section_title = ""
                page_lines: List[str] = []
                if text and text.strip():
                    section_title = self._extract_section_title(text)
                    page_lines = [line for line in text.split("\n") if line.strip()]

                    results.append(ParsedElement(
                        text=text,
                        category="Text",
                        page_or_index=page_num,
                        metadata={
                            "page": page_num,
                            "section_title": section_title,
                        },
                    ))

                # 提取表格（如果存在）
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    table_text = self._table_to_markdown(table)
                    if table_text.strip():
                        table_title = self._extract_table_title(
                            page_lines, table_idx, len(tables)
                        )
                        # ── D1: 从第一行提取列名 ──
                        column_names = self._extract_column_names(table)

                        results.append(ParsedElement(
                            text=table_text,
                            category="Table",
                            page_or_index=page_num,
                            metadata={
                                "page": page_num,
                                "section_title": section_title,
                                "table_index": table_idx + 1,
                                "table_title": table_title,
                                "column_names": column_names,
                                "col_count": len(table[0]) if table else 0,
                            },
                        ))

        # ── D2: 跨页表格合并后处理 ──
        results = self._merge_cross_page_tables(results)
        return results

    # ──────────────────────────────────────────────
    # D1: 列名提取
    # ──────────────────────────────────────────────

    def _extract_column_names(self, table: List[List]) -> str:
        """从表格第一行提取列名，返回逗号分隔字符串。

        Args:
            table: pdfplumber 原始表格（二维列表）

        Returns:
            逗号分隔的列名字符串，如 "项目,金额(万元),占营业收入比例"
            若无法提取则返回空字符串
        """
        if not table or not table[0]:
            return ""

        header_row = table[0]
        names = []
        for cell in header_row:
            name = str(cell).strip() if cell else ""
            # 过滤掉空字符串和纯数字/标点（不像列名）
            if name and not name.isdigit() and len(name) < 30:
                names.append(name)

        return ",".join(names) if names else ""

    # ──────────────────────────────────────────────
    # D2: 跨页表格合并
    # ──────────────────────────────────────────────

    def _merge_cross_page_tables(
        self, elements: List[ParsedElement]
    ) -> List[ParsedElement]:
        """检测并合并跨页的连续同结构表格。

        合并条件（同时满足）：
        1. 两个相邻 Table 元素的页码差为 1
        2. 两表列数相同（或差值 ≤ 1，容忍合并单元格）
        3. 第二张表第一行与第一张表第一行的"表头相似度"低
           （即第二页是续表数据行，而非新表头）

        Args:
            elements: 解析后的 ParsedElement 列表

        Returns:
            合并后的 ParsedElement 列表
        """
        if not elements:
            return elements

        merged: List[ParsedElement] = []
        i = 0
        while i < len(elements):
            elem = elements[i]

            # 只对 Table 元素尝试合并
            if elem.category != "Table" or i + 1 >= len(elements):
                merged.append(elem)
                i += 1
                continue

            next_elem = elements[i + 1]

            # 条件 1: 下一个也是 Table 且页码差为 1
            if (
                next_elem.category == "Table"
                and next_elem.page_or_index == elem.page_or_index + 1
            ):
                col_count_curr = elem.metadata.get("col_count", 0)
                col_count_next = next_elem.metadata.get("col_count", 0)

                # 条件 2: 列数相同或差值 ≤ 1
                if col_count_curr > 0 and abs(col_count_curr - col_count_next) <= 1:
                    # 条件 3: 第二页第一行不像独立表头（是续表数据行）
                    if self._is_continuation_table(elem, next_elem):
                        merged_elem = self._do_merge_tables(elem, next_elem)
                        logger.info(
                            "跨页表格合并: 第%d页 + 第%d页 → 合并为一个 Table chunk",
                            elem.page_or_index, next_elem.page_or_index,
                        )
                        # 将合并后的元素替换 elem，继续尝试与再下一页合并
                        elements[i] = merged_elem
                        elements.pop(i + 1)
                        continue  # 不递增 i，继续检查合并后元素与新 i+1

            merged.append(elem)
            i += 1

        return merged

    def _is_continuation_table(
        self, base: ParsedElement, candidate: ParsedElement
    ) -> bool:
        """判断 candidate 是否是 base 的续表（而非全新表格）。

        启发式：若 candidate 的第一行内容与 base 的第一行高度相似
        （文字重叠 > 50%），则认为 candidate 也有独立表头，不合并。
        否则视为续表。

        Args:
            base:      当前页的 Table 元素
            candidate: 下一页的 Table 元素

        Returns:
            True = 应该合并（candidate 是续表），False = 不合并
        """
        # 从 Markdown 文本中提取第一行（| col1 | col2 | ... |）
        base_first = self._get_markdown_header_row(base.text)
        cand_first = self._get_markdown_header_row(candidate.text)

        if not base_first or not cand_first:
            return False  # 无法判断，保守选择不合并

        # 计算字符级 Jaccard 相似度
        set_base = set(re.split(r"[\s|,，。、]+", base_first.lower()))
        set_cand = set(re.split(r"[\s|,，。、]+", cand_first.lower()))
        set_base.discard("")
        set_cand.discard("")

        if not set_base or not set_cand:
            return False

        intersection = len(set_base & set_cand)
        union = len(set_base | set_cand)
        similarity = intersection / union if union > 0 else 0

        # 相似度 > 0.5 → 下一页也有独立表头 → 不是续表
        # 相似度 ≤ 0.5 → 下一页是数据行 → 是续表
        return similarity <= 0.5

    def _get_markdown_header_row(self, markdown_text: str) -> str:
        """从 Markdown 表格文本中提取第一行内容。

        Args:
            markdown_text: Markdown 格式表格文本

        Returns:
            第一行文本（去除注入前缀 【...】）
        """
        lines = markdown_text.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            # 跳过注入的 【...】 前缀行
            if stripped.startswith("【") and stripped.endswith("】"):
                continue
            if stripped.startswith("|") and "---" not in stripped:
                return stripped
        return ""

    def _do_merge_tables(
        self, base: ParsedElement, continuation: ParsedElement
    ) -> ParsedElement:
        """将 continuation 的数据行追加到 base 表格后，生成合并后的 ParsedElement。

        Args:
            base:         基础 Table 元素（第 N 页）
            continuation: 续表 Table 元素（第 N+1 页）

        Returns:
            合并后的新 ParsedElement，page_or_index 保持为 base 的页码
        """
        base_lines = base.text.strip().split("\n")
        cont_lines = continuation.text.strip().split("\n")

        # 找 continuation 中的分隔行（|---|）位置，跳过表头和分隔行，只取数据行
        sep_idx = next(
            (i for i, line in enumerate(cont_lines)
             if line.strip().startswith("|") and "---" in line),
            None,
        )
        if sep_idx is not None:
            # 跳过表头行（sep_idx 之前）和分隔行本身，只取数据行
            data_lines = cont_lines[sep_idx + 1:]
        else:
            # 没找到分隔行，跳过第一行（可能是表头）
            data_lines = cont_lines[1:] if len(cont_lines) > 1 else cont_lines

        # 过滤掉空行
        data_lines = [line for line in data_lines if line.strip()]

        merged_text = "\n".join(base_lines + data_lines)

        # 更新 metadata
        new_metadata = dict(base.metadata)
        new_metadata["pages"] = [base.page_or_index, continuation.page_or_index]
        new_metadata["merged_from_pages"] = True

        return ParsedElement(
            text=merged_text,
            category="Table",
            page_or_index=base.page_or_index,
            metadata=new_metadata,
        )

    # ──────────────────────────────────────────────
    # 已有辅助方法（保持不变）
    # ──────────────────────────────────────────────

    def _extract_table_title(
        self, page_lines: List[str], table_idx: int, total_tables: int
    ) -> str:
        """从页面文本行中推断表格标题。

        策略：
        - 若该页只有一张表，取最后一个长度 < 40 且非纯数字/标点的行（通常是表名）
        - 若该页有多张表，按 table_idx 等分行数后，取对应分段的最后一行作为候选

        Args:
            page_lines: 页面非空文本行列表
            table_idx:  当前表格索引（0-based）
            total_tables: 该页表格总数

        Returns:
            推断出的表格标题，未能识别则返回空字符串
        """
        if not page_lines:
            return ""

        if total_tables <= 1:
            candidates = page_lines
        else:
            seg_size = max(1, len(page_lines) // total_tables)
            start = table_idx * seg_size
            end = start + seg_size
            candidates = page_lines[start:end]

        for line in reversed(candidates):
            stripped = line.strip()
            if (
                stripped
                and len(stripped) < 40
                and not stripped.isdigit()
                and not all(c in "| -—\t" for c in stripped)
            ):
                return stripped

        return ""

    def _extract_section_title(self, page_text: str) -> str:
        """从页面文本中提取章节标题

        取前两行作为可能的标题（年报通常标题在前）

        Args:
            page_text: 页面完整文本

        Returns:
            提取的标题，如果无法识别则返回空字符串
        """
        lines = page_text.strip().split("\n")
        if not lines:
            return ""

        candidates = [line.strip() for line in lines[:2] if line.strip()]

        if len(candidates) >= 2:
            title = candidates[0] if len(candidates[0]) <= len(candidates[1]) else candidates[1]
        elif len(candidates) == 1:
            title = candidates[0]
        else:
            title = ""

        # 过滤掉明显不是标题的候选（如页码、日期等）
        if title and len(title) < 50 and not title.isdigit():
            return title

        return ""

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """将 pdfplumber 表格转换为 Markdown 格式

        Args:
            table: 表格二维列表

        Returns:
            Markdown 格式的表格字符串
        """
        if not table:
            return ""

        markdown_rows = []
        for row in table:
            # 清理单元格内容
            cells = [str(cell).strip() if cell else "" for cell in row]
            markdown_rows.append("| " + " | ".join(cells) + " |")

        # 添加分隔行
        if len(markdown_rows) > 1:
            header_len = len(table[0])
            markdown_rows.insert(1, "|" + "|".join(["---"] * header_len) + "|")

        return "\n".join(markdown_rows)


def parse_pdf(pdf_path: str) -> ParsedDocument:
    """解析 PDF 文件的便捷函数

    Args:
        pdf_path: PDF 文件路径

    Returns:
        ParsedDocument 对象
    """
    parser = PDFParser()
    return parser.parse(pdf_path)
