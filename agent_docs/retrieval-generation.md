# Retrieval & Generation

## 检索流程

### 方式一：Query Router 模式（默认，`ENABLE_QUERY_ROUTER=True`）

Query Router 提供统一的查询分类、检索编排和回退机制：

```
┌─────────────────────────────────────────────────────────────┐
│                     QueryRouter.route()                      │
├─────────────────────────────────────────────────────────────┤
│  1. QueryClassifier.classify(query, filter_file)            │
│     ├── 确定 filter_source: explicit / auto_company / none  │
│     ├── 确定 retrieval_scope: single_company / global       │
│     ├── 确定 fallback_allowed: True / False                 │
│     ├── 检测 scene: factual/comparison/extraction/policy_qa │
│     └── 确定 generation_mode: single_step / two_step        │
│                                                             │
│  2. Execute Retrieval                                       │
│     ├── scope=global → global search                        │
│     └── scope=single_company → filtered search              │
│         ├── results > threshold → return filtered          │
│         └── results <= threshold & fallback_allowed        │
│             → fallback to global (filtered_then_global)     │
│                                                             │
│  3. Return RetrievedContext                                 │
│     ├── chunks: List[RetrievedChunk]                        │
│     ├── retrieval_mode: filtered/global/filtered_then_global│
│     ├── fallback_triggered: bool                            │
│     └── classification: QueryClassification                 │
└─────────────────────────────────────────────────────────────┘
```

**检索路径详解**

| 路径 | 触发条件 | 回退 | 适用场景 |
|------|----------|------|----------|
| `global` | 无公司名识别 | - | 跨公司比较、通用问题 |
| `filtered` | 显式/自动识别公司 + 有结果 | - | 单公司问题 |
| `filtered_then_global` | 自动识别公司 + 空结果 + 允许回退 | 是 | 误识别公司时兜底 |

**配置项** (`config/settings.py`)

```python
ENABLE_QUERY_ROUTER = True                           # 总开关
QUERY_ROUTER_ALLOW_AUTO_FILTER_FALLBACK = True       # 自动识别允许回退
QUERY_ROUTER_ALLOW_EXPLICIT_FILTER_FALLBACK = False  # 显式过滤不允许回退
QUERY_ROUTER_EMPTY_RESULT_THRESHOLD = 0              # 空结果才回退
QUERY_ROUTER_CONFIDENCE_THRESHOLD = 0.6              # 分类置信度阈值
```

### 方式二：传统模式（`ENABLE_QUERY_ROUTER=False`）

直接使用 RerankRetriever，无路由和回退机制。

---

## Two-Stage 检索（在 Router 内部使用）

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

