# 集成测试 Fixtures

本目录包含集成测试所需的测试数据和样本文件。

## 目录结构

```
fixtures/
├── sample_pdfs/          # PDF样本文件
│   ├── sample_annual_report.pdf      # < 100KB的简化年报
│   ├── sample_cross_page_table.pdf   # 跨页表格测试
│   └── sample_policy_section.pdf     # 政策段落测试
├── sample_qa/            # 测试QA对
│   └── test_qa_dataset.json          # 结构化QA数据
└── expected_outputs/     # 预期输出
    ├── expected_chunks.json
    └── expected_answers.json
```

## 样本文件规范

### PDF样本要求

1. **大小**: 每个PDF < 100KB（用于快速测试）
2. **内容**: 包含可解析的文本内容
3. **页数**: 1-5页（保持测试快速）
4. **来源**: 可使用公开财报的前几页

### 创建样本PDF

```bash
# 从现有PDF提取前几页创建样本
python scripts/create_sample_pdf.py \
    --input "data/raw/中兴通讯：2025年年度报告.pdf" \
    --output "tests/integration/fixtures/sample_pdfs/sample_annual_report.pdf" \
    --pages 1-3
```

## QA数据集格式

```json
[
  {
    "question": "2025年营收是多少？",
    "answer": "2025年营收为100亿元",
    "ground_truth": "100亿元",
    "category": "factual",
    "source_file": "sample_annual_report.pdf",
    "expected_mode": "single_step"
  }
]
```

## 注意事项

- 不要将大文件(>1MB)提交到git
- 敏感/私有数据不要放入fixtures
- 定期更新样本以反映格式变化
