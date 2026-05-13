---
title: Dialogue-to-Action Graph Parser 验证报告
doc_type: verification
module: nlp-voc
topic: dialogue-to-action-graph
status: stable
created: 2026-04-28
updated: 2026-04-28
owner: self
source: ai
---

# Dialogue-to-Action Graph Parser - 验证报告

## 验证概要

| 项目 | 结果 |
|------|------|
| **论文** | TOD-Flow: Modeling the Structure of Task-Oriented Dialogues (arXiv:2312.04668, 2023) |
| **代码路径** | `paper2skills-code/nlp_voc/dialogue_to_action_graph/model.py` |
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
- Test 1: 改地址对话 → 4 个节点，3 条 SHOULD 边，推荐路径正确
- Test 2: 产品故障对话 → 5 个节点，路径到 resolution
- Test 3: 空文本处理 → 0 个节点，无异常

### 3. 数据 POC 验证
**数据集**: Zendesk 客服工单 (`zendesk_momcozy_voc_sampled.csv`)
**样本量**: 100 条工单

**统计结果**:
| 指标 | 数值 |
|------|------|
| 处理工单数 | 100 |
| 平均轮次 | 1.0 |
| 解决率 | 0.0 |
| 常见模式 | user_issue (100%) |

**说明**: Zendesk 数据为单轮客户留言（非多轮对话），每条仅含用户问题。代码模板的核心功能在多轮模拟对话中已验证。真实多轮对话数据需从客服系统导出。

**示例输出**:
```
🗣️  客服对话决策图
轮次: 1
节点: 1, 边: 0

❓ [n0] user_issue: Hello , My name is Crystal Olivas . I was wondering if it po...
```

### 4. 结构检查
- ✅ 输出为合法的 `DecisionGraph` 对象
- ✅ 节点包含必需的 (id, type, text) 字段
- ✅ 边包含 (source, target, relation) 字段，relation 为 Can/Should/ShouldNot
- ✅ `get_path_to_resolution()` 可沿 SHOULD 边找到推荐路径
- ✅ `to_dict()` 可序列化为 JSON

## 已知局限

1. **数据限制**: Zendesk 数据为单轮工单，非多轮对话。多轮功能仅在模拟数据中验证
2. **节点分类精度**: 基于关键词匹配，无法处理复杂语义
3. **关系推断**: 仅基于相邻轮次的简单规则，未实现 TOD-Flow 的图学习
4. **Should Not 关系**: 当前规则覆盖较少，需更多业务规则补充

## 改进建议

| 优先级 | 改进项 | 预估工作量 |
|--------|--------|-----------|
| P1 | 获取多轮对话数据（从客服系统导出） | 依赖外部 |
| P2 | 训练 Dialog-Act 分类器 | 1-2 周 |
| P3 | 实现 TOD-Flow 图学习算法 | 2-3 周 |
| P4 | 接入 LLM 做实时对话状态跟踪 | 3-5 天 |

## 结论

**验证通过**。代码模板可运行，模拟对话测试验证成功。当前主要限制是缺乏多轮对话数据。建议在获取真实多轮数据后，优先训练 Dialog-Act 分类器。
