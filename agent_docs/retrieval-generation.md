# Retrieval & Generation

## 检索流程（Two-Stage）

**默认模式（`USE_RERANKER=True`）**
1. 向量检索：embed query → ChromaDB ANN → 取 `RETRIEVAL_TOP_K=30` 候选
2. 精排：BGE-Reranker-v2-m3 交叉编码，返回 `RERANK_TOP_K=10` 结果

**Baseline 模式（`USE_RERANKER=False`）**
- 仅向量检索，直接返回 `RETRIEVAL_TOP_K` 条结果

## Generation Prompt 强制规则（`src/generation/prompts.py`）

所有规则均注入 `QA_SYSTEM_PROMPT`，不可绕过：

| 规则 | 触发场景 | 处理方式 |
|------|---------|---------|
| 报表类型歧义 | 问题未指明母公司/合并，context 同时含两种数据 | 声明歧义或默认合并报表并注明 |
| 多处冲突信息 | context 同一事项有多处不一致表述 | 逐条列出 + 标注来源 + 综合结论 |
| 总额/合计字段 | 问题涉及"合计""总额"等 | 只引用明确标注"合计/总计"的字段；禁止自行加总 |
| 占比/比例数据 | 问题涉及"占比""比例""比重"等 | 优先引用含百分比原文；无百分比禁止自行计算 |

## LLM Provider 切换

`config/settings.py` 中设置 `LLM_PROVIDER = "kimi"` 或 `"qwen"`。

| Provider | 模型 | Base URL |
|----------|------|----------|
| `kimi` | `kimi-k2-0905-preview` | `https://api.moonshot.cn/v1` |
| `qwen` | `qwen3.5-397b-a17b` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

Ragas 评估也使用同一 LLM provider，通过 `RAGAS_OPENAI_API_KEY` / `RAGAS_OPENAI_BASE_URL` 环境变量覆盖。

