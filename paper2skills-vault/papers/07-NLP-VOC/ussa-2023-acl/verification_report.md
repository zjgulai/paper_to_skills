---
title: VOC Semantic Blueprint 验证报告
doc_type: verification
module: nlp-voc
topic: voc-semantic-blueprint
status: stable
created: 2026-04-28
updated: 2026-04-28
owner: self
source: ai
---

# VOC Semantic Blueprint Extractor - 验证报告

## 验证概要

| 项目 | 结果 |
|------|------|
| **论文** | USSA: A Unified Table Filling Scheme for Structured Sentiment Analysis (ACL 2023) |
| **代码路径** | `paper2skills-code/nlp_voc/voc_semantic_blueprint/model.py` |
| **验证日期** | 2026-04-28 |
| **验证人** | AI (Claude Code) |

## 验证项

### 1. 语法检查
```bash
python3 -m py_compile model.py
```
**结果**: ✅ 通过

### 2. 单元测试
```bash
python3 model.py
```
**结果**: ✅ 通过（3/3 测试用例）
- Test 1: 英文评论提取 → 2 个节点
- Test 2: 含场景评论提取 → 1 个节点，正确识别 positive sentiment
- Test 3: 空文本处理 → 0 个节点，无异常

### 3. 数据 POC 验证
**数据集**: Momcozy 高质量评论采样 (`momcozy_voc_high_quality_sampled.csv`)
**样本量**: 100 条评论

**统计结果**:
| 指标 | 数值 |
|------|------|
| 处理评论数 | 100 |
| 总提取节点 | 370 |
| 平均每评论节点 | 3.70 |
| 不重复方面数 | 82 |

**示例输出**:
```json
{
  "raw_text": "This has worked great for me so far! I had a c-section...",
  "nodes": [
    {"aspect": "pricey but", "opinion": "great for", "sentiment": "positive"},
    {"aspect": "pricey but", "opinion": "comfortable", "sentiment": "positive"},
    {"aspect": "happen", "opinion": "comfortable", "sentiment": "positive", "cause": "because i remember my previous one"}
  ]
}
```

### 4. 结构检查
- ✅ 输出为合法的 `VOCBlueprint` 对象（dataclass）
- ✅ 每个节点包含必需的 (aspect, opinion, sentiment) 字段
- ✅ 可选字段 (cause, scene) 正确填充或留空
- ✅ `to_dict()` 方法可序列化为 JSON

## 已知局限

1. **方面提取精度**: 规则基线方法提取的方面名不够准确（如 "suction but", "pricey but" 等），需要训练好的模型替代
2. **场景识别**: 基于关键词匹配，覆盖率有限
3. **多语言**: 当前仅支持英文，中文支持需要扩展 tokenizer
4. **Overlap/Discontinuity**: 规则方法无法处理复杂 overlap 场景，USSA 模型的核心优势在此

## 改进建议

| 优先级 | 改进项 | 预估工作量 |
|--------|--------|-----------|
| P1 | 接入预训练语言模型做 aspect/opinion 识别 | 3-5 天 |
| P2 | 增加中文 tokenizer 和词典 | 1-2 天 |
| P3 | 接入 LLM 做原因/场景提取 | 2-3 天 |
| P4 | 训练 USSA 模型 on 母婴领域数据 | 2-3 周 |

## 结论

**验证通过**。代码模板可运行，POC 验证成功。当前版本为规则基线，可作为快速原型和概念验证。生产环境建议按优先级逐步改进。
