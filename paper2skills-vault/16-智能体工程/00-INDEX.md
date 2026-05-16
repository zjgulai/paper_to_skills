---
title: 16-智能体工程技能索引
doc_type: index
module: 16-智能体工程
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# 16-智能体工程 (LLM Agent Engineering) 技能索引

## 领域定位

`16-智能体工程` 与 `09-DataAgent-LLM` / `10-MAS` 三个领域形成分层互补:

```
应用层(09-DataAgent-LLM)    -- 业务 Agent 怎么帮分析师做事
架构层(10-MAS)              -- 多智能体怎么协作、角色怎么分配
工程层(16-智能体工程)        -- Skill/Context/协议怎么具体落地
```

本领域聚焦 LLM Agent 的工程实践层面:Skill 设计、上下文管理、工具调用、协议落地、模型能力训练。

## 技能分类

### A. Agent Skills/Tools 工程

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Auto-Skill-Synthesis](./Skill-Auto-Skill-Synthesis.md) | 已萃取 P0 | SkillForge (SIGIR 2026, arxiv:2604.08618) | 从客服历史对话自动生成 Skill |
| Skill-Skill-Lifecycle-Design | 已萃取 P1 | SoK Agentic Skills (arxiv:2602.20867) | Skill 卡设计哲学与生命周期 |
| Skill-Co-Evolutionary-Skill-Verification | 待萃取 P2 | EvoSkills (arxiv:2604.01687) | Skill 自动迭代与验证 |
| Skill-Progressive-Disclosure-Architecture | 待萃取 P2 | Agent Skills Survey (arxiv:2602.12430) | SKILL.md + 渐进式上下文加载 |

### B. Context Engineering 上下文工程

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| Skill-Context-Compression | 待萃取 P1 | ACON (arxiv:2510.00615) | 长对话/长 VOC 压缩(内存降 26-54%) |
| Skill-Agentic-Memory-Management | 待萃取 P1 | AgeMem (arxiv:2601.01885) | 客户长期偏好记忆(母婴 1-3 年周期) |
| Skill-Active-Context-Pruning | 待萃取 P2 | Focus (arxiv:2601.07190) | 仿生粘菌策略的主动剪枝 |
| Skill-Memory-as-Action | 待萃取 P2 | MemAct (arxiv:2510.12635) | Memory 操作嵌入 policy |

### C. MAS 协作工程层(与 10-MAS 互补,聚焦协议)

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| Skill-MCP-A2A-Protocol-Stack | 待萃取 P1 | MAS Orchestration Survey (arxiv:2601.13671) | MCP + A2A 双协议栈选型 |
| Skill-Orchestration-Trace-RL | 待萃取 P2 | RL via Orchestration Traces (arxiv:2605.02801) | 用 trace 训 RL 优化编排决策 |
| Skill-Task-Adaptive-Topology | 待萃取 P2 | AdaptOrch (arxiv:2602.16873) | 动态拓扑(比 static 提升 12-23%) |

### D. LLM Function Calling / Tool Use

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| Skill-Open-Source-Tool-Use-Model | 待萃取 P2 | Hermes 4 (arxiv:2508.18255, 破例纳入) | 开源 tool calling 基座选型 |
| Skill-SLM-Tool-Calling-Optimization | 待萃取 P2 | SLM Tool Calling (arxiv:2512.15943) | 客服/工单分类成本优化 |

### E. MCP 协议 + Tool 描述质量

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| Skill-Tool-Description-Audit | 待萃取 P2 | MCP Descriptions Smelly (arxiv:2602.14878) | Skill/Tool 描述质量审核 6 维评分 |
| Skill-MCP-Tool-Use-Benchmark | 待萃取 P3 | MCPAgentBench (arxiv:2512.24565) | 工具选择与区分能力评估 |

### F. 电商 Agent 落地(与项目业务场景强契合)

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| Skill-Long-Term-Preference-Memory | 已萃取 P0 | Shopping Companion (arxiv:2603.14864) | 母婴用户长周期偏好 RL Agent |
| Skill-Agent-Stage-Evaluation | 已萃取 P0 | EComStage (arxiv:2601.02752) | Perception/Planning/Action 三阶段评估 |

## P0 优先执行顺序(用户已确认"顺序做")

1. **Skill-Auto-Skill-Synthesis** (SkillForge) — 客服记录自动萃取 Skill
2. **Skill-Long-Term-Preference-Memory** (Shopping Companion) — 母婴用户长周期偏好
3. **Skill-Agent-Stage-Evaluation** (EComStage) — Agent 三阶段评估

完成 P0 三篇后进入 P1:Skill-Skill-Lifecycle-Design (SoK Agentic Skills) 作为方法论底座。

## 工作流映射

```
论文(arxiv 2026)
    ↓
[paper-选题] 按 6 个子方向筛选
    ↓
[paper-萃取] MasterPrompt → 5 模块 Skill 卡
    ↓
[paper-审核] 总分 ≥ 7/10 且代码 ≥ 7/10
    ↓
[paper-同步] vault + github + feishu
    ↓
落入 paper2skills-vault/16-智能体工程/00-知识库-Skill卡片/
```

## 与其他领域的衔接

| 领域 | 衔接点 | 数据流 |
|------|--------|--------|
| 09-DataAgent-LLM | DataAgent 的 Skill 复用 | 业务 Agent 调用工程层 Skill |
| 10-MAS | MAS 算法层与协议层 | 算法选型 → 协议落地 |
| 07-NLP-VOC | VOC 长文本处理 | Context Compression 应用 |
| 14-用户分析 | 用户长周期建模 | Agentic Memory 应用 |

## 统计数据

- 规划技能数:15
- 已萃取:4
- 待萃取 P0:0 / P1:3 / P2:7 / P3:1
- 论文来源:arxiv 2026-01 至 2026-05

---

> 选题草稿:`drafts/analysis/llm-agent-engineering-paper-selection-draft-20260516.md`
> 最后更新:2026-05-16
