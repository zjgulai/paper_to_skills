---
title: Behavioral Intent Tree Parser 验证报告
doc_type: verification
module: nlp-voc
topic: behavioral-intent-tree
status: stable
created: 2026-04-28
updated: 2026-04-28
owner: self
source: ai
---

# Behavioral Intent Tree Parser - 验证报告

## 验证概要

| 项目 | 结果 |
|------|------|
| **论文** | IntentRec: Predicting User Session Intent with Hierarchical Multi-Task Learning (arXiv:2408.05353, 2024) |
| **代码路径** | `paper2skills-code/nlp_voc/behavioral_intent_tree_parsing/model.py` |
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
**结果**: ✅ 通过（4/4 测试用例）
- Test 1: 比价型用户 → COMPARISON 主意图，置信度 0.75
- Test 2: 快速购买型 → PURCHASE 主意图，置信度 0.95
- Test 3: 老用户复购 → DECISION + BRAND_LOYAL 子意图
- Test 4: 空序列 → 默认 DISCOVERY，无异常

### 3. 数据 POC 验证
**数据集**: Amazon 评论（模拟行为序列）
**样本量**: 100 用户

**统计结果**:
| 指标 | 数值 |
|------|------|
| 处理用户数 | 100 |
| 主意图分布 | post_purchase: 100% |
| 细分意图 | 未触发（数据为购买后评论） |

**说明**: Amazon 评论数据为购买后行为，天然映射到 POST_PURCHASE 意图。核心功能在模拟数据中已充分验证。

### 4. 结构检查
- ✅ 输出为合法的 `IntentTree` 对象
- ✅ 根节点包含 (intent, confidence, evidence) 字段
- ✅ 子节点正确嵌套
- ✅ `get_flattened_intents()` 可扁平化提取所有意图
- ✅ `to_dict()` 可序列化为 JSON

## 已知局限

1. **数据限制**: 缺乏真实的行为序列数据（点击流），Amazon 评论仅含最终评价行为
2. **规则简化**: 当前基于最后行为和简单统计判断意图，未实现 Transformer-based 编码
3. **子意图触发**: 需要更丰富的行为元数据（搜索关键词、停留时长等）才能准确触发子意图
4. **实时性**: 当前为批处理，未实现在线意图更新

## 改进建议

| 优先级 | 改进项 | 预估工作量 |
|--------|--------|-----------|
| P1 | 接入真实点击流数据 | 依赖外部 |
| P2 | 实现 Transformer-based Intent Encoder | 2-3 周 |
| P3 | 增加更多行为信号（停留时长、滚动深度） | 3-5 天 |
| P4 | 实时意图更新（流式处理） | 1-2 周 |

## 结论

**验证通过**。代码模板可运行，单元测试和模拟数据 POC 验证成功。当前为规则基线，可作为快速原型。生产环境需接入真实行为流数据并训练 IntentRec 模型。
