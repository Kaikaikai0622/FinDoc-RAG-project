"""文档切块模块

使用 LlamaIndex 的 SentenceSplitter 进行文本切分。
每个 chunk 生成唯一 chunk_id，并携带元信息。

表格处理策略：
  - Table / SheetRow 元素直接独立成 chunk，不经过 SentenceSplitter
  - 若单张表格字符数超过 TABLE_MAX_CHARS，按行分段后每段独立成 chunk
  - 含"总计/合计"行的表格额外生成聚合 chunk（TableSummary）

文本处理策略：
  - 普通文本：合并后经 SentenceSplitter 按 CHUNK_SIZE 切分
  - 政策/条款类文本：整段合并为单个 chunk（上限 POLICY_MAX_CHARS），保留完整条款
"""
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core.node_parser import SentenceSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, POLICY_SECTION_KEYWORDS, POLICY_MAX_CHARS
from .models import ParsedElement, ParsedDocument

logger = logging.getLogger(__name__)

# 需要注入上下文的元素类别
INJECT_CATEGORIES = {"Table", "SheetRow", "Slide"}

# 表格独立 chunk 时的最大字符数阈值（对应 bge-m3 约 8192 token 上限的保守估计）
# 中文字符平均约 1.5 token，8192 * (1/1.5) ≈ 5400，留余量取 4000
TABLE_MAX_CHARS = 4000

# 按行分段时，每段最大行数（Markdown 表格每行约 50-200 字符，30 行约 1500-6000 字符）
TABLE_ROWS_PER_SEGMENT = 30

# 匹配总计/合计/小计行的正则
_SUMMARY_ROW_RE = re.compile(r"总计|合计|小计|汇总")


def _inject_context(elem: ParsedElement) -> str:
    """为可注入元素添加上下文前缀

    Args:
        elem: ParsedElement 实例

    Returns:
        注入后的文本
    """
    if elem.category not in INJECT_CATEGORIES:
        return elem.text

    # 提取各类格式的标题信息
    prefix_parts = [
        elem.metadata.get("table_title") or
        elem.metadata.get("section_title") or
        elem.metadata.get("slide_title") or
        elem.metadata.get("sheet_name", ""),
        elem.metadata.get("column_names", "")
    ]
    prefix = "·".join(p for p in prefix_parts if p)

    if prefix:
        return f"【{prefix}】\n{elem.text}"
    return elem.text


def _is_policy_section(text_elems: List[ParsedElement]) -> bool:
    """判断一组文本元素是否属于政策/条款类段落。

    检查 section_title 或文本内容（全文）是否命中 POLICY_SECTION_KEYWORDS。

    Args:
        text_elems: 同一页的文本元素列表

    Returns:
        True 表示应整段保留，不按 CHUNK_SIZE 切分
    """
    for elem in text_elems:
        # 检查 section_title
        section_title = elem.metadata.get("section_title", "")
        for kw in POLICY_SECTION_KEYWORDS:
            if kw in section_title:
                return True
        # 检查文本内容全文（原来只检查前200字，会漏掉段落中间出现的关键词）
        for kw in POLICY_SECTION_KEYWORDS:
            if kw in elem.text:
                return True
    return False


def _combined_text_is_policy(combined_text: str) -> bool:
    """快速判断合并后的文本是否含政策关键词（用于 chunker 主循环双重检查）。

    Args:
        combined_text: 页面文本拼接后的完整字符串

    Returns:
        True 表示应使用政策段落分割器
    """
    for kw in POLICY_SECTION_KEYWORDS:
        if kw in combined_text:
            return True
    return False


def _extract_summary_chunks(
    injected: str,
    table_title: str,
    section_title: str,
    source_file: str,
    page_number: int,
) -> List[str]:
    """从表格的 Markdown 文本中提取含"总计/合计"关键词的行，生成聚合 chunk 文本列表。

    每个聚合 chunk 格式：
    【table_title·总计行】
    字段1: 值1 | 字段2: 值2 | ...
    （来源：source_file 第page_number页）

    Args:
        injected:      已注入上下文前缀的表格 Markdown 文本
        table_title:   表格标题（fallback，优先从注入前缀行解析）
        section_title: 章节标题
        source_file:   来源文件名
        page_number:   页码

    Returns:
        聚合 chunk 文本列表（通常 0-3 个）
    """
    lines = injected.split("\n")

    # 优先从注入前缀行（【...】格式）提取真实表格标题
    # 避免 table_title 参数传入了推断错误的标题（如相邻表格的名称）
    for line in lines[:3]:
        stripped = line.strip()
        m = re.match(r"^【(.+?)(?:·[^】]*)?】$", stripped)
        if m:
            candidate = m.group(1).strip()
            # 只用看起来合理的标题（非列名列表样式）
            if candidate and "," not in candidate and len(candidate) < 50:
                table_title = candidate
            break

    # 找 Markdown 表头行：必须在分隔行（|---|）之前出现
    # 分隔行之后的 | 行全部是数据行，不应被当作表头
    header_cells: List[str] = []
    sep_line_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and "---" in stripped:
            sep_line_idx = idx
            break

    if sep_line_idx is not None:
        # 从分隔行往前找最近的 | 行作为表头
        for idx in range(sep_line_idx - 1, -1, -1):
            stripped = lines[idx].strip()
            if stripped.startswith("【") and stripped.endswith("】"):
                continue
            if stripped.startswith("|"):
                header_cells = [c.strip() for c in stripped.strip("|").split("|")]
                break

    summary_chunks: List[str] = []

    for line in lines:
        stripped = line.strip()
        # 跳过非表格行、分隔行
        if not stripped.startswith("|") or "---" in stripped:
            continue
        # 检查是否含总计关键词
        if not _SUMMARY_ROW_RE.search(stripped):
            continue

        row_cells = [c.strip() for c in stripped.strip("|").split("|")]

        # 将列名与值配对
        if header_cells and len(header_cells) >= len(row_cells):
            pairs = []
            for col_name, value in zip(header_cells, row_cells):
                if col_name and value:
                    pairs.append(f"{col_name}: {value}")
            row_summary = " | ".join(pairs) if pairs else stripped
        else:
            # Fallback：没有有效表头或列数不匹配，用位置索引标注
            pairs = []
            for i, value in enumerate(row_cells):
                if value:
                    pairs.append(value)
            row_summary = " | ".join(pairs) if pairs else stripped

        # 找出行中第一个含关键词的字段作为 chunk 标题
        row_title_part = row_cells[0] if row_cells else "总计"
        title_label = (
            f"{table_title}·{row_title_part}" if table_title
            else (f"{section_title}·{row_title_part}" if section_title else row_title_part)
        )

        chunk_text = (
            f"【{title_label}】\n"
            f"{row_summary}\n"
            f"（来源：{source_file} 第{page_number}页）"
        )
        summary_chunks.append(chunk_text)

    return summary_chunks


def _split_table_by_rows(header_prefix: str, body_lines: List[str]) -> List[str]:
    """将超长 Markdown 表格按行分段。

    每段保留表头前缀（含 【标题】 行和 Markdown 表头/分隔行），
    后跟最多 TABLE_ROWS_PER_SEGMENT 条数据行。

    Args:
        header_prefix: 表格头部文本（标题注入行 + Markdown 表头两行），已含换行
        body_lines:    Markdown 表格数据行列表（不含表头两行）

    Returns:
        分段后的字符串列表，每项为一个独立 chunk 的文本
    """
    segments: List[str] = []
    total = len(body_lines)
    for start in range(0, total, TABLE_ROWS_PER_SEGMENT):
        batch = body_lines[start: start + TABLE_ROWS_PER_SEGMENT]
        seg_text = header_prefix + "\n".join(batch)
        segments.append(seg_text)
    return segments if segments else [header_prefix]


class Chunker:
    """文档切块器

    将 ParsedDocument 切分为 chunks，每个 chunk 携带元信息。
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """初始化切块器

        Args:
            chunk_size: 每个 chunk 的最大字符数
            chunk_overlap: 相邻 chunk 之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 使用 LlamaIndex 的 SentenceSplitter
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 保留分隔符
            separator="\n\n",
        )
        # 政策段落专用分割器（更宽松的 chunk_size）
        self._policy_splitter = SentenceSplitter(
            chunk_size=POLICY_MAX_CHARS,
            chunk_overlap=chunk_overlap,
            separator="\n\n",
        )

    def chunk_document(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        """将 ParsedDocument 切分为 chunks

        Args:
            doc: ParsedDocument 对象

        Returns:
            chunks 列表，每项包含 chunk_id, chunk_text, metadata
        """
        # 按页码/索引分组元素
        elements_by_page: Dict[int, List[ParsedElement]] = {}
        for elem in doc.elements:
            page_idx = elem.page_or_index
            if page_idx not in elements_by_page:
                elements_by_page[page_idx] = []
            elements_by_page[page_idx].append(elem)

        all_chunks = []
        global_chunk_index = 0
        source_file = doc.source_file

        # ── 阶段 1：按页收集并分离文本/表格，同时做政策段落跨页拼接 ──
        sorted_pages = sorted(elements_by_page.keys())
        # 每页的文本元素和表格元素
        page_text_data: List[Dict[str, Any]] = []  # [{page_idx, text_elems, combined_text}]
        page_table_data: List[Dict[str, Any]] = []  # [{page_idx, table_elems}]

        for page_idx in sorted_pages:
            page_elements = elements_by_page[page_idx]
            table_elems = [e for e in page_elements if e.category in {"Table", "SheetRow"}]
            text_elems  = [e for e in page_elements if e.category not in {"Table", "SheetRow"}]

            if table_elems:
                page_table_data.append({"page_idx": page_idx, "table_elems": table_elems})

            if text_elems:
                processed_texts = [_inject_context(e) for e in text_elems]
                combined_text = "\n\n".join(processed_texts)
                page_text_data.append({
                    "page_idx": page_idx,
                    "text_elems": text_elems,
                    "combined_text": combined_text,
                })

        # ── 阶段 1.5：政策段落跨页拼接 ──
        # 当 Page N 的文本是政策段落且以未完成句子结尾时，
        # 向后吸收 Page N+1（乃至 N+2）的文本，合并为一个逻辑块
        merged_text_data: List[Dict[str, Any]] = []
        skip_indices: set = set()

        for i, td in enumerate(page_text_data):
            if i in skip_indices:
                continue

            combined = td["combined_text"]
            base_page = td["page_idx"]
            base_elems = list(td["text_elems"])
            is_policy = _is_policy_section(td["text_elems"]) or _combined_text_is_policy(combined)

            if is_policy:
                # 尝试向后拼接相邻页（页码差 1-2 且总长不超限）
                j = i + 1
                while j < len(page_text_data):
                    next_td = page_text_data[j]
                    # 只拼接页码连续的页面
                    if next_td["page_idx"] - page_text_data[j - 1]["page_idx"] > 2:
                        break
                    candidate = combined + "\n\n" + next_td["combined_text"]
                    if len(candidate) > POLICY_MAX_CHARS:
                        break
                    # 下一页文本也含政策关键词或者上一页末尾是未完成句子
                    next_is_policy = (
                        _is_policy_section(next_td["text_elems"])
                        or _combined_text_is_policy(next_td["combined_text"])
                    )
                    ends_incomplete = not combined.rstrip().endswith(("。", "）", ")", "】", "》"))
                    if next_is_policy or ends_incomplete:
                        combined = candidate
                        base_elems.extend(next_td["text_elems"])
                        skip_indices.add(j)
                        j += 1
                        logger.info(
                            "政策段落跨页拼接: 第%d页 → 第%d页 (%d chars)",
                            base_page, next_td["page_idx"], len(combined),
                        )
                    else:
                        break

            merged_text_data.append({
                "page_idx": base_page,
                "text_elems": base_elems,
                "combined_text": combined,
                "is_policy": is_policy,
            })

        # ── 阶段 2：生成文本 chunks ──
        for td in merged_text_data:
            page_idx = td["page_idx"]
            text_elems = td["text_elems"]
            combined_text = td["combined_text"]
            is_policy = td["is_policy"] or _combined_text_is_policy(combined_text)

            if is_policy:
                if len(combined_text) <= POLICY_MAX_CHARS:
                    chunks_texts = [combined_text]
                    logger.debug(
                        "政策段落整段保留 (%d chars): 来源=%s 第%d页",
                        len(combined_text), source_file, page_idx,
                    )
                else:
                    chunks_texts = self._policy_splitter.split_text(combined_text)
                    logger.debug(
                        "政策段落宽松切分 (%d chars → %d chunks): 来源=%s 第%d页",
                        len(combined_text), len(chunks_texts), source_file, page_idx,
                    )
            else:
                chunks_texts = self._splitter.split_text(combined_text)

            first_elem = text_elems[0]
            for chunk_text in chunks_texts:
                chunk_id = self._generate_chunk_id(source_file, page_idx, global_chunk_index)
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "metadata": {
                        "source_file": source_file,
                        "page_number": page_idx,
                        "chunk_index": global_chunk_index,
                        "element_category": first_elem.category,
                        "section_title": first_elem.metadata.get("section_title", ""),
                    },
                })
                global_chunk_index += 1

        # ── 阶段 3：处理表格元素（每张表独立成 chunk，超长时按行分段）──
        for td in page_table_data:
            page_idx = td["page_idx"]
            table_elems = td["table_elems"]
            for elem in table_elems:
                injected = _inject_context(elem)
                table_title = (
                    elem.metadata.get("table_title")
                    or elem.metadata.get("section_title", "")
                )
                section_title = elem.metadata.get("section_title", "")

                # 实际存储的页码（跨页合并后取 base 页码）
                actual_page = elem.metadata.get("page", page_idx)

                if len(injected) <= TABLE_MAX_CHARS:
                    # 正常大小：整张表作为一个 chunk
                    chunk_texts_for_table = [injected]
                else:
                    # 超长：拆分为按行分段
                    logger.warning(
                        "表格超长 (%d chars > %d)，按行分段: 来源=%s 第%d页 table_index=%s",
                        len(injected), TABLE_MAX_CHARS, source_file, page_idx,
                        elem.metadata.get("table_index", "?"),
                    )
                    lines = injected.split("\n")
                    sep_idx = next(
                        (i for i, line in enumerate(lines) if line.startswith("|") and "---" in line),
                        None,
                    )
                    if sep_idx is not None and sep_idx + 1 < len(lines):
                        header_lines = lines[:sep_idx + 1]
                        body_lines   = lines[sep_idx + 1:]
                        header_prefix = "\n".join(header_lines) + "\n"
                    else:
                        header_prefix = ""
                        body_lines = lines

                    chunk_texts_for_table = _split_table_by_rows(header_prefix, body_lines)

                for chunk_text in chunk_texts_for_table:
                    chunk_id = self._generate_chunk_id(source_file, page_idx, global_chunk_index)
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "metadata": {
                            "source_file": source_file,
                            "page_number": actual_page,
                            "chunk_index": global_chunk_index,
                            "element_category": "Table",
                            "section_title": section_title,
                            "table_title": table_title,
                        },
                    })
                    global_chunk_index += 1

                # D3: 额外生成总计/合计聚合 chunk
                summary_texts = _extract_summary_chunks(
                    injected, table_title, section_title, source_file, actual_page
                )
                for summary_text in summary_texts:
                    chunk_id = self._generate_chunk_id(source_file, page_idx, global_chunk_index)
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_text": summary_text,
                        "metadata": {
                            "source_file": source_file,
                            "page_number": actual_page,
                            "chunk_index": global_chunk_index,
                            "element_category": "TableSummary",
                            "section_title": section_title,
                            "table_title": table_title,
                        },
                    })
                    global_chunk_index += 1
                    logger.debug(
                        "生成聚合 chunk: %s 来源=%s 第%d页",
                        summary_text[:50], source_file, actual_page,
                    )

        return all_chunks

    def _generate_chunk_id(
        self,
        source_file: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """生成唯一的 chunk_id

        格式：{filename_hash}_{page_number}_{chunk_index}

        Args:
            source_file: 来源文件名
            page_number: 页码
            chunk_index: chunk 序号

        Returns:
            唯一的 chunk_id
        """
        filename = Path(source_file).stem
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        chunk_id = f"{file_hash}_{page_number}_{chunk_index}"
        return chunk_id


def chunk_document(doc: ParsedDocument) -> List[Dict[str, Any]]:
    """将 ParsedDocument 切分为 chunks 的便捷函数

    Args:
        doc: ParsedDocument 对象

    Returns:
        chunks 列表
    """
    chunker = Chunker()
    return chunker.chunk_document(doc)


# ========== 向后兼容函数 ==========

def chunk_elements(
    elements: List[Dict[str, Any]],
    source_file: str,
) -> List[Dict[str, Any]]:
    """将旧格式元素列表切分为 chunks（向后兼容）

    Args:
        elements: 旧格式元素列表，每项包含 text, page_number, element_type
        source_file: 来源文件名

    Returns:
        chunks 列表
    """
    parsed_elements = []
    for elem in elements:
        parsed_elements.append(ParsedElement(
            text=elem["text"],
            category=elem.get("element_type", "Text"),
            page_or_index=elem["page_number"],
            metadata={"page": elem["page_number"]},
        ))

    doc = ParsedDocument(
        source_file=source_file,
        file_type="pdf",
        elements=parsed_elements,
    )

    return chunk_document(doc)
