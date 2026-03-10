# FinDoc-RAG 评估报告

## 生成时间
2026-03-10 11:05:10

## 数据集概况
| 项目 | 数量 |
|------|------|
| 手工标注 | 25 条 |
| 合成生成 | 31 条 |
| 总计 | 56 条 |

| 场景 | 数量 | 占比 |
|------|------|------|
| factual | 25 | 44.6% |
| comparison | 3 | 5.4% |
| extraction | 20 | 35.7% |
| policy_qa | 8 | 14.3% |

## 总体指标

| 指标 | 分数 |
|------|------|
| 正确率 (Accuracy) | 0.52 |
| 加权正确率 (Weighted Accuracy) | 0.00 |
| 检索命中率 (Retrieval Hit Rate) | 0.79 |
| 忠实度 (Faithfulness) | 0.79 |
| 答案相关性 (Answer Relevancy) | 0.62 |
| 上下文精度 (Context Precision) | 0.45 |
| 上下文召回 (Context Recall) | 0.58 |

## ⭐ 高价值场景分析

| 场景 | Context Precision | Context Recall | Accuracy | 评级 |
|------|-------------------|----------------|----------|------|
| comparison | 0.50 | 0.50 | 0.50 | ⚠️ 需改进 |
| policy_qa | 0.92 | 0.92 | 0.25 | ⚠️ 需改进 |
| extraction | 0.81 | 0.81 | 0.57 | ✅ 良好 |

评级标准：≥0.7 ✅ 良好 | 0.5-0.7 ⚠️ 需改进 | <0.5 ❌ 较差

## 参数对比实验

### Chunk Size 对比（固定 top_k=5）

| Chunk Size | Accuracy | Hit Rate | Faithfulness | Context Precision | 综合得分 |
|------------|----------|----------|--------------|-------------------|----------|
| - | - | - | - | - | - |

### Top-K 对比（固定 chunk_size=512）

| Top-K | Accuracy | Hit Rate | Faithfulness | Context Precision | 综合得分 |
|-------|----------|----------|--------------|-------------------|----------|
| - | - | - | - | - | - |

综合得分 = (Accuracy + Hit Rate + Faithfulness + Context Precision) / 4

## 🏆 最佳策略推荐

- **最佳 Chunk Size**：256
- **最佳精排 Top-K**：7
- **当前最优配置**：chunk_size=256, top_k=7, embedding=BAAI/bge-m3, llm=kimi

## 待改进方向
1. comparison 场景的上下文召回偏低时，优先检查跨页信息是否被切散。
2. extraction 场景若准确率偏低，优先检查表格解析和chunk保真。
3. 可进一步探索更细粒度的 chunk 策略优化。