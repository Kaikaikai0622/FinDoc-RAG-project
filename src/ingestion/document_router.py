"""DocumentRouter — 多格式文档路由器

根据文件扩展名自动选择对应解析器，返回统一的 ParsedDocument。
各格式解析器采用懒加载（首次使用时实例化并缓存），避免不必要的 import 开销。

支持格式：
    .pdf   → PDFParser
    .docx  → DocxParser
    .pptx  → PptxParser
    .xlsx  → XlsxParser
    .xls   → XlsxParser (会抛出友好错误，提示另存为 .xlsx)
    .txt   → PlainTextParser
    .md    → PlainTextParser
    .csv   → PlainTextParser
"""
import logging
from pathlib import Path
from typing import Dict, Any

from .models import ParsedDocument

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 自定义异常
# ------------------------------------------------------------------

class UnsupportedFileTypeError(ValueError):
    """不支持的文件格式异常

    Attributes:
        file_path: 传入的文件路径
        suffix: 文件扩展名
    """

    def __init__(self, file_path: str, suffix: str) -> None:
        self.file_path = file_path
        self.suffix = suffix
        supported = ".pdf / .docx / .pptx / .xlsx / .txt / .md / .csv"
        super().__init__(
            f"不支持的文件格式 '{suffix}'（文件: {file_path}）。"
            f"支持的格式：{supported}"
        )


class DocumentParseError(RuntimeError):
    """文档解析失败异常

    包装底层解析器抛出的异常，保留 __cause__ 以便追踪原始错误。

    Attributes:
        file_path: 解析失败的文件路径
    """

    def __init__(self, file_path: str, cause: Exception) -> None:
        self.file_path = file_path
        super().__init__(f"解析文件失败: {file_path}。原因: {cause}")


# ------------------------------------------------------------------
# DocumentRouter
# ------------------------------------------------------------------

class DocumentRouter:
    """多格式文档路由器

    根据文件扩展名自动分发到对应解析器，返回 ParsedDocument。
    解析器实例懒加载并缓存，同一 Router 实例多次调用不会重复初始化。

    Usage:
        router = DocumentRouter()
        doc = router.route("/path/to/report.pdf")
        doc = router.route("/path/to/data.xlsx")
    """

    # 扩展名 → file_type 映射
    EXTENSION_MAP: Dict[str, str] = {
        ".pdf":  "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".xlsx": "xlsx",
        ".xls":  "xlsx",   # 路由到 XlsxParser，会抛出友好错误
        ".txt":  "txt",
        ".md":   "md",
        ".csv":  "csv",
    }

    def __init__(self) -> None:
        # 解析器实例缓存：file_type → parser instance
        self._cache: Dict[str, Any] = {}

    def route(self, file_path: str) -> ParsedDocument:
        """解析文件，自动选择解析器

        Args:
            file_path: 文件路径（任意支持格式）

        Returns:
            ParsedDocument 对象

        Raises:
            UnsupportedFileTypeError: 不支持的文件格式
            DocumentParseError: 解析过程中发生错误
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.EXTENSION_MAP:
            raise UnsupportedFileTypeError(file_path=file_path, suffix=suffix)

        file_type = self.EXTENSION_MAP[suffix]
        parser = self._get_parser(file_type)

        logger.info(f"[DocumentRouter] 使用 {type(parser).__name__} 解析: {path.name}")

        try:
            return parser.parse(file_path)
        except (UnsupportedFileTypeError, RuntimeError):
            # 直接透传已处理的异常（如 .xls 不支持、依赖缺失）
            raise
        except Exception as exc:
            raise DocumentParseError(file_path=file_path, cause=exc) from exc

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_parser(self, file_type: str) -> Any:
        """懒加载并缓存解析器实例

        Args:
            file_type: 格式类型字符串（见 EXTENSION_MAP）

        Returns:
            对应的解析器实例
        """
        if file_type in self._cache:
            return self._cache[file_type]

        if file_type == "pdf":
            from .pdf_parser import PDFParser
            self._cache["pdf"] = PDFParser()

        elif file_type == "docx":
            from .docx_parser import DocxParser
            self._cache["docx"] = DocxParser()

        elif file_type == "pptx":
            from .pptx_parser import PptxParser
            self._cache["pptx"] = PptxParser()

        elif file_type == "xlsx":
            from .xlsx_parser import XlsxParser
            self._cache["xlsx"] = XlsxParser()

        elif file_type in ("txt", "md", "csv"):
            from .plain_text_parser import PlainTextParser
            # txt / md / csv 共用同一个 PlainTextParser 实例
            parser = PlainTextParser()
            self._cache["txt"] = parser
            self._cache["md"] = parser
            self._cache["csv"] = parser

        else:
            raise UnsupportedFileTypeError(file_path="", suffix=f".{file_type}")

        return self._cache[file_type]

    @property
    def supported_extensions(self) -> list:
        """返回支持的文件扩展名列表"""
        return list(self.EXTENSION_MAP.keys())

