"""文档解析数据模型

定义统一的文档解析结果Schema，支持多格式解析器。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal


@dataclass
class ParsedElement:
    """解析后的单个元素

    Attributes:
        text: 元素文本内容
        category: 元素类型 - Text(纯文本) | Table(表格) | Title(标题) | Slide(Slide页) | SheetRow(Excel行)
        page_or_index: 位置标识
            - PDF/DOCX: 页码
            - PPTX: slide number
            - Excel: 行范围起始
        metadata: 额外元信息，透传给下游
    """
    text: str
    category: Literal["Text", "Table", "Title", "Slide", "SheetRow"]
    page_or_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_injectable(self) -> bool:
        """判断该类型元素是否需要注入上下文前缀"""
        return self.category in {"Table", "SheetRow", "Slide"}


@dataclass
class ParsedDocument:
    """解析后的完整文档

    Attributes:
        source_file: 原始文件名（含扩展名）
        file_type: 文档类型 - pdf | docx | pptx | xlsx | txt | md | csv
        elements: 按阅读顺序排列的元素列表
    """
    source_file: str
    file_type: Literal["pdf", "docx", "pptx", "xlsx", "txt", "md", "csv"]
    elements: List[ParsedElement] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        """文档页数/ sheet 数"""
        if not self.elements:
            return 0
        return max(elem.page_or_index for elem in self.elements) + 1


# ========== 各格式 metadata 字段约定 ==========

# PDF (pdfplumber)
# ParsedElement(
#     text="信托资产总计 | 12,345,678 | 11,234,567",
#     category="Table",
#     page_or_index=11,
#     metadata={
#         "page": 11,
#         "section_title": "十八、信托财务报告",  # 解析器尽力提取
#         "table_index": 2,                        # 该页第几张表
#     }
# )

# DOCX (python-docx)
# ParsedElement(
#     text="...",
#     category="Text",   # 或 "Title"（来自 Heading 样式）
#     page_or_index=0,   # python-docx 无页码，用段落索引
#     metadata={
#         "paragraph_index": 42,
#         "style": "Heading 2",
#         "section_title": "...",   # 从最近一个 Heading 提取
#     }
# )

# PPTX (python-pptx)
# ParsedElement(
#     text="Q4 净利润同比增长 23%",
#     category="Slide",
#     page_or_index=5,
#     metadata={
#         "slide_number": 5,
#         "slide_title": "Q4 营收分析",
#         "shape_type": "text_box",
#     }
# )

# Excel (pandas + openpyxl)
# ParsedElement(
#     text="项目 | 本年末 | 上年末\n信托资产总计 | 12,345,678 | 11,234,567",
#     category="SheetRow",
#     page_or_index=0,
#     metadata={
#         "sheet_name": "资产负债表",
#         "row_range": "1-15",
#         "column_names": "项目,本年末,上年末",
#     }
# )

# TXT / MD
# ParsedElement(
#     text="...",
#     category="Text",
#     page_or_index=0,
#     metadata={"line_range": "1-30"}
# )
