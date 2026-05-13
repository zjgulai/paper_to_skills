# 10-MAS 多智能体系统 (Multi-Agent Systems) 技能索引

---
title: MAS 多智能体系统技能索引
doc_type: index
module: 10-MAS
status: stable
created: 2026-05-10
updated: 2026-05-10
---

## 领域定位

10-MAS 是 paper2skills 工作流中的最终执行层：

```
SRL 看见结构 → HGT/HGCN 学习结构 → 语义蓝图记住结构 → MAS 基于结构行动
```

本领域聚焦 LLM 驱动的多智能体系统设计、编排、执行与进化。

## 技能分类

### A. Agent 基础框架

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-AutoGen-Multi-Agent-Conversation](./00-知识库-Skill卡片/Skill-AutoGen-Multi-Agent-Conversation.md) | 已萃取 | AutoGen (Microsoft, 2023) | 可编程多 agent 对话系统 |
| [Skill-MetaGPT-SOP-Driven-Collaboration](./00-知识库-Skill卡片/Skill-MetaGPT-SOP-Driven-Collaboration.md) | 已萃取 | MetaGPT (ICLR 2024) | SOP 驱动的标准化协作 |
| [Skill-CAMEL-Role-Playing-Agents](./00-知识库-Skill卡片/Skill-CAMEL-Role-Playing-Agents.md) | 已萃取 | CAMEL (NeurIPS 2023) | 角色扮演式自主协作 |

### B. Agent 规划与任务分解

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Tree-of-Thoughts-Planning](./00-知识库-Skill卡片/Skill-Tree-of-Thoughts-Planning.md) | 已萃取 | ToT (NeurIPS 2023) | 树搜索式任务规划 |
| [Skill-ReAct-Reasoning-Acting](./00-知识库-Skill卡片/Skill-ReAct-Reasoning-Acting.md) | 已萃取 | ReAct (ICLR 2023) | 推理-行动交替执行 |
| [Skill-Reflexion-Self-Improvement](./00-知识库-Skill卡片/Skill-Reflexion-Self-Improvement.md) | 已萃取 | Reflexion (NeurIPS 2023) | 自我反思与迭代改进 |

### C. Skill 注册表与编排

| 技能 | 状态 | 论文/框架 | 业务场景 |
|------|------|----------|---------|
| [Skill-Skill-Registry-Dynamic-Loading](./00-知识库-Skill卡片/Skill-Skill-Registry-Dynamic-Loading.md) | 已萃取 | 自定义框架 | 动态技能发现与加载 |
| [Skill-Subagent-Decomposition](./00-知识库-Skill卡片/Skill-Subagent-Decomposition.md) | 已萃取 | 自定义框架 | 复杂任务子 agent 分解 |
| [Skill-MAS-Orchestrator](./00-知识库-Skill卡片/Skill-MAS-Orchestrator.md) | 已萃取 | 自定义框架 | 多 agent 编排与调度 |

### D. 反馈闭环与进化

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Self-Improving-Agent-Feedback-Loop](./00-知识库-Skill卡片/Skill-Self-Improving-Agent-Feedback-Loop.md) | 已萃取 | Self-Refine (NeurIPS 2023) | Agent 自我进化管道 |
| [Skill-Multi-Agent-Debate](./00-知识库-Skill卡片/Skill-Multi-Agent-Debate.md) | 已萃取 | MAD (EMNLP 2024) | 多 agent 辩论共识 |
| [Skill-Agent-Memory-Learning](./00-知识库-Skill卡片/Skill-Agent-Memory-Learning.md) | 已萃取 | MemGPT (2023) | 长期记忆与学习 |

## 工作流映射

```
长自然语言文本
    ↓
[07-NLP-VOC] SRL + 事件框架抽取
    ↓
[08-知识图谱] 异构图构建 (HGT + HGCN)
    ↓
[07/08] 语义蓝图编译
    ↓
[10-MAS] ┬─ Task Blueprint (任务蓝图)
         ├─ Skill Registry (技能注册表)
         ├─ Agent Planner (智能体规划)
         ├─ Subagent Decomposer (子智能体分解)
         └─ MAS Orchestrator (多智能体编排)
              ↓
         执行/检索/分析/生成/验证
              ↓
         反馈 → 再训练 → 闭环
```

## 统计数据

- 总技能数: 12
- 已萃取: 12
- 待萃取: 0
- 代码模板: 12

## 与上游领域的衔接

| 上游领域 | 衔接点 | 数据流 |
|---------|--------|--------|
| 07-NLP-VOC | InstructUIE 抽取结果 | 结构化事件/实体/关系 |
| 08-知识图谱 | HGT/HGCN 图表示 | 节点 embedding + 图结构 |
| 09-DataAgent-LLM | LLM 推理能力 | Agent 基础认知能力 |

---

> 最后更新: 2026-05-10
