# Evaluation System

## Evaluator（`src/evaluation/evaluator.py`）

### 指标体系

| 指标 | 类型 | 说明 |
|------|------|------|
| `accuracy` | 基础 | LLM judge；失败退回启发式规则 |
| `weighted_accuracy` | 基础 | 按评估集实际场景分布加权（4 核心场景），消除场景比例随机影响 |
| `retrieval_hit_rate` | 基础 | 标注来源文件+页码是否命中检索结果（支持文本包含 fallback） |
| `avg_retrieval_rank` | 基础 | 命中时的平均排名（越小越好） |
| `faithfulness` | Ragas | 答案忠实度 |
| `answer_relevancy` | Ragas | 答案与问题相关性 |
| `context_precision` | Ragas | 检索精确率 |
| `context_recall` | Ragas | 检索召回率 |

### LLM Judge 规则

- 包含 GT 核心信息（数值/是非/主体）即判 CORRECT，允许模型答案更详细
- 仅当核心结论明显不一致时判 INCORRECT
- `ground_truth = "__UNANSWERABLE__"` → 检测模型是否正确拒答

### 场景分组

`factual` / `comparison` / `extraction` / `policy_qa` / `out_of_scope` / `unknown`

### 加权准确率计算

仅对 4 核心场景（`factual`/`extraction`/`policy_qa`/`comparison`）计算；
以各场景在本次评估集中的**实际占比**作为权重，不使用固定理想分布。

---

## Synthetic QA Generator（`src/evaluation/testset_generator.py`）

### 5-Block Flow

```
DocStore chunks
  ↓ _sample_chunks()：按 source_file 均匀采样，噪声过滤（< 80 字符等）
  ↓
Block 1  Topic Extraction    → 1-3 个核心主题词（JSON）
Block 2  Question Generation → 初始问题（JSON）
Block 3  Question Evolution  → 自然语言改写
           后处理①：口语称谓检测（老X/小X + 姓氏集合）→ 命中则丢弃
           后处理②：报表范围检测 → report_scope = "unspecified" / "specified"
Block 4  Grounded Answer     → 仅基于 chunk 回答 + supporting_excerpt
                               标注 scene / difficulty
Block 6  Numeric/Unit Check  纯规则，置于 Block 5 之前节省 LLM 调用：
           ① 数值回查：answer 中每个数值在 chunk_text 字符串查找（含千分位变体）
           ② 单位检查：万元/亿元/% 需在 chunk 找到对应 pattern
              "元"特殊处理：chunk 有万元/亿元/人民币/≥3个数值 → 放行交 Block 5
Block 5  Groundedness Filter → LLM 语义判断（无幻觉 + 能唯一回答）
  ↓
输出 Schema v1.1 JSON
```

### 噪声过滤

- `_is_noise_chunk()`：长度 < 80、全数字/标点、纯页码行 → 跳过
- `_is_colloquial_name_match()`：两步（粗匹配 + 姓氏集合精确验证）

---

## QA Dataset Schema（v1.1）

```json
{
  "_schema_version": "1.1",
  "questions": [
    {
      "id": "m001",
      "question": "...",
      "ground_truth": "...（__UNANSWERABLE__ 表示不可回答）",
      "source_files": ["文件名.pdf"],
      "source_pages": [10, 11],
      "scene": "factual|comparison|extraction|policy_qa|out_of_scope",
      "difficulty": "easy|medium|hard",
      "metadata": {
        "report_scope": "unspecified",
        "generalization_excluded": true
      }
    }
  ]
}
```

**字段说明**
- `source_files`：数组；`"（任意文件）"` 忽略文件来源匹配
- `source_pages`：数组；`0` 忽略页码匹配
- `metadata.report_scope = "unspecified"`：问题未明确母公司/合并范围，供下游过滤
- `metadata.generalization_excluded = true`：不参与泛化评测加权
- 向后兼容旧版 `source_file`（字符串）和 `source_page`（整数）格式

