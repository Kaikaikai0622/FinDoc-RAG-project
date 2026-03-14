"""公司名称解析模块

实现从用户问题中提取公司名称，并映射到对应的文件名。
"""
import re
from typing import List, Dict, Optional, Tuple
from difflib import get_close_matches


class CompanyResolver:
    """公司名称解析器

    功能：
    1. 从文件名提取公司名称（如"陕国投Ａ：2025年年度报告.pdf"→"陕国投A"）
    2. 从用户问题中提取提到的公司名称
    3. 模糊匹配（处理全角半角、简称等）
    """

    def __init__(self, doc_store=None):
        """初始化解析器

        Args:
            doc_store: DocStore 实例，用于获取所有文件名
        """
        self._doc_store = doc_store
        self._company_map: Dict[str, str] = {}  # 公司名 -> 完整文件名
        self._company_aliases: Dict[str, str] = {}  # 别名 -> 标准公司名
        self._initialized = False

    def _load_companies(self) -> None:
        """从 DocStore 加载所有文件名并解析公司名称"""
        if self._initialized:
            return

        if self._doc_store is None:
            from src.storage import DocStore
            self._doc_store = DocStore()

        # 获取所有 chunks 的 source_file
        all_chunks = self._doc_store.get_all_chunks()
        files = set()
        for chunk in all_chunks:
            files.add(chunk["metadata"]["source_file"])

        # 解析文件名中的公司名称
        for filename in files:
            company = self._extract_company_from_filename(filename)
            if company:
                self._company_map[company] = filename
                # 添加别名（处理全角半角等）
                self._add_aliases(company, filename)

        self._initialized = True

    def _extract_company_from_filename(self, filename: str) -> Optional[str]:
        """从文件名提取公司名称

        文件名格式: "陕国投Ａ：2025年年度报告.pdf"
        提取: "陕国投A"

        Args:
            filename: 源文件名

        Returns:
            公司名称，提取失败返回 None
        """
        # 移除扩展名
        name = filename.replace('.pdf', '').replace('.PDF', '')
        # 分割冒号（中文或英文冒号）
        parts = re.split(r'[：:]', name, 1)
        if parts:
            return parts[0].strip()
        return None

    def _add_aliases(self, company: str, filename: str) -> None:
        """添加公司名别名映射

        例如：
        - "陕国投Ａ" → "陕国投A"（全角转半角）
        - "陕国投" → "陕国投A"（去掉后缀）
        """
        # 标准化：全角转半角
        normalized = self._normalize_chars(company)
        self._company_aliases[normalized] = company

        # 去掉常见后缀的变体
        suffixes = ['A', 'B', 'H', '股份', '集团', '科技', '有限']
        for suffix in suffixes:
            if normalized.endswith(suffix):
                short = normalized[:-len(suffix)].strip()
                if short and short not in self._company_aliases:
                    self._company_aliases[short] = company

        # 原始公司名也作为别名
        if company not in self._company_aliases:
            self._company_aliases[company] = company

    def _normalize_chars(self, text: str) -> str:
        """字符标准化（全角转半角）"""
        # 全角字母数字转半角
        result = []
        for char in text:
            code = ord(char)
            # 全角空格
            if code == 0x3000:
                result.append(' ')
            # 全角字符（字母、数字）
            elif 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            else:
                result.append(char)
        return ''.join(result)

    def get_all_companies(self) -> List[str]:
        """获取所有公司名称列表"""
        self._load_companies()
        return list(self._company_map.keys())

    def extract_company_from_question(self, question: str) -> Optional[str]:
        """从用户问题中提取公司名称

        策略：
        1. 匹配所有已知公司名称（按长度降序，优先匹配长的）
        2. 使用模糊匹配处理轻微差异

        Args:
            question: 用户问题

        Returns:
            匹配到的公司名称，未匹配返回 None
        """
        self._load_companies()

        if not self._company_aliases:
            return None

        # 标准化问题文本
        normalized_question = self._normalize_chars(question)

        # 1. 精确匹配（优先匹配长的公司名，避免"陕国投"匹配到"陕国投A"的片段）
        sorted_aliases = sorted(
            self._company_aliases.keys(),
            key=len,
            reverse=True
        )

        for alias in sorted_aliases:
            if alias in normalized_question:
                return self._company_aliases[alias]

        # 2. 模糊匹配（处理如"芯导"匹配"芯导科技"）
        # 提取2-4字的片段尝试匹配
        for length in range(min(4, len(normalized_question)), 1, -1):
            for i in range(len(normalized_question) - length + 1):
                fragment = normalized_question[i:i+length]
                if len(fragment) >= 2:
                    matches = get_close_matches(
                        fragment,
                        sorted_aliases,
                        n=1,
                        cutoff=0.8
                    )
                    if matches:
                        return self._company_aliases[matches[0]]

        return None

    def resolve(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """解析问题中的公司名称并返回完整文件名

        Args:
            question: 用户问题

        Returns:
            (公司名称, 完整文件名)，未匹配返回 (None, None)
        """
        self._load_companies()

        company = self.extract_company_from_question(question)
        if company:
            return company, self._company_map.get(company)
        return None, None

    def get_filter_for_question(self, question: str) -> Optional[str]:
        """获取问题对应的文件过滤条件

        Args:
            question: 用户问题

        Returns:
            用于 filter_file 的完整 source_file，未匹配返回 None
        """
        _, filename = self.resolve(question)
        return filename


# 全局实例（单例模式）
_resolver_instance: Optional[CompanyResolver] = None


def get_company_resolver(doc_store=None) -> CompanyResolver:
    """获取 CompanyResolver 单例实例"""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = CompanyResolver(doc_store)
    return _resolver_instance


def extract_company_filter(question: str) -> Optional[str]:
    """便捷函数：从问题提取文件过滤条件

    Args:
        question: 用户问题

    Returns:
        用于 filter_file 的字符串，未匹配返回 None

    示例：
        >>> extract_company_filter("陕国投的营收是多少？")
        '陕国投Ａ：2025年年度报告.pdf'
        >>> extract_company_filter("芯导科技的利润如何？")
        '芯导科技：2025年年度报告.pdf'
        >>> extract_company_filter("介绍一下这家公司的业务")  # 未提及具体公司
        None
    """
    resolver = get_company_resolver()
    return resolver.get_filter_for_question(question)
