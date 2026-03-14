# Ingestion & Chunker

## PDF Parser（`src/ingestion/pdf_parser.py`）

**D1 列名提取 `_extract_column_names`**
- 取 `table[0]` 作为表头候选，过滤空值和纯数字
- 结果注入 metadata `column_names`，同时生成前缀行 `【table_title·col1,col2,…】`

**D2 跨页表格合并 `_merge_cross_page_tables`**
- 合并条件：页码差=1 ＋ 列数差≤1 ＋ 第二页首行与第一页表头 Jaccard 相似度 ≤ 50%
- 相似度 > 50% → 认为是新表头，不合并
- 合并后 `page_or_index` 保持基础页码；`metadata.pages` 记录合并的页码列表

## Chunker 三阶段（`src/ingestion/chunker.py`）

```
阶段 1    按页收集，分离 文本元素 / 表格元素

阶段 1.5  政策段落跨页拼接
          触发：命中 POLICY_SECTION_KEYWORDS（利润分配/分红/公司章程…）
               或页面末尾是未完成句子（不以 。）】》 结尾）
          行为：向后吸收连续页面文本，最多合并至 POLICY_MAX_CHARS=2000 字符

阶段 2    文本 chunk 生成
          政策段落 → 整段保留（超出 POLICY_MAX_CHARS 才用宽松分割器）
          普通段落 → SentenceSplitter（CHUNK_SIZE=512, CHUNK_OVERLAP=80）

阶段 3    表格 chunk 生成
          整表作为 1 个 chunk；超 TABLE_MAX_CHARS=4000 则按行分段
          含"总计/合计/小计"行 → 额外生成 TableSummary chunk：
            【表名·总计行】
            列1: 值1 | 列2: 值2 | ...
            （来源：file.pdf 第N页）
```

**TableSummary 表头检测规则（已修复 Bug）**
- 表头 = Markdown 分隔行 `|---|` **之前**的最近一个 `|` 行
- 分隔行之后的所有 `|` 行均为数据行，不作为表头候选
- 无有效表头时 fallback：直接拼接总计行所有非空单元格

**chunk_id 格式：** `{md5(filename)[:8]}_{page}_{global_index}`

## Key Constants

| Constant | Value | Notes |
|----------|-------|-------|
| `CHUNK_SIZE` | 512   | SentenceSplitter 字符上限 |
| `CHUNK_OVERLAP` | 80    | 相邻 chunk 重叠字符数 |
| `POLICY_MAX_CHARS` | 2000  | 政策段落整段保留上限 |
| `TABLE_MAX_CHARS` | 4000  | 单表 chunk 字符上限，超出则按行分段 |
| `TABLE_ROWS_PER_SEGMENT` | 30    | 按行分段时每段最大行数 |

