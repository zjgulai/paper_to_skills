---
title: 10-MAS 2026 萃取计划 — 按知识图谱填充路径排序
doc_type: analysis
module: paper-选题
topic: mas-2026
status: active
created: 2026-06-04
updated: 2026-06-04
owner: self
source: human+ai
---

# 10-MAS 2026 萃取计划（图谱填充路径版）

## 背景与动机

### 图谱现状（2026-06-04）

| 指标 | 数据 |
|------|------|
| 全图节点数 | 287 个 Skills |
| 全图边数 | 5,152 条 |
| MAS 领域现有 Skills | 25 个 |
| 本次新增计划 | 9 个方向，约 15-18 张 Skill 卡 |

### 现有 MAS 覆盖层次

```
基础框架层  ✅  AutoGen / MetaGPT / CAMEL
规划推理层  ✅  ReAct / ToT / Reflexion
编排调度层  ✅  SDOF / Dynamic DAG / ParaManager / MAS Orchestrator
通信协议层  ✅  G²CP (Graph-Grounded Communication Protocol)
评估层      ✅  MASEval
进化学习层  ✅  EvoSC / Self-Improving
业务落地层  ✅  Flowr(供应链) / AIM-RM(库存) / AgenticPay(采购)
```

### 本次填补的结构性空白

```
生产工程层  ❌  资源调度 / 规模化管理 / 测试验证  ← 最大盲区
安全信任层  ❌  动态信任 / 攻防对抗               ← 完全空白
理论基础层  ❌  共识机制                           ← 图谱缺失
跨域桥梁层  ❌  MAS↔广告 / MAS↔KG动态            ← 两大域零连接
协议互操作  ❌  跨组织 Agent 协议                  ← 标准化进行中
```

---

## Sprint 计划（3个 Sprint，6周）

### Sprint 1（第1-2周）：P0 生产工程基础

**目标**：填补 MAS 生产层最大空白，3 个 Skill 方向，6 张卡

| # | Skill 名 | 论文 | 优先理由 |
|---|---------|------|---------|
| 1.1 | Skill-MAS-Resource-Scheduling | HiveMind + AgentRM + MCPP | 工程可落地性最高，MIT开源即插即用 |
| 1.2 | Skill-MAS-Testing-Verification | FLARE + MAESTRO | 首创性 Fuzzing，有CI/CD集成路径 |
| 1.3 | Skill-MAS-Dynamic-Trust | DynaTrust + A-Trust + ECL | 信任层完全空白，防御效果突出 |

**Sprint 1 交付物**：
- 3 个新 Skill 方向的完整卡片（共 6 张，含代码模板）
- 更新 10-MAS/00-INDEX.md
- 更新 skills_graph_report.md（新增 edges）

---

### Sprint 2（第3-4周）：P1 重要桥梁与理论基础

**目标**：建立 3 条跨域桥梁，填补共识理论与规模化管理空白

| # | Skill 名 | 论文 | 优先理由 |
|---|---------|------|---------|
| 2.1 | Skill-LLM-AutoBidding-MAS | DARA + LBM | mas↔advertising 首条桥梁，WWW'26 顶会 |
| 2.2 | Skill-MAS-Consensus-Mechanism | Aegean + DySCo + SAC | 共识理论完全空白，有形式化证明 |
| 2.3 | Skill-MAS-Scale-Management | MegaFlow + MonoScale + OrgAgent | 万级并发生产验证，公司制架构洞见 |

**Sprint 2 交付物**：
- 3 个新 Skill 方向（共 6-7 张卡）
- mas ↔ advertising 跨域 edge 建立
- 更新 00-INDEX.md 与图谱

---

### Sprint 3（第5-6周）：P2 安全攻防与跨域精补

**目标**：补全安全层，建立剩余跨域桥梁

| # | Skill 名 | 论文 | 优先理由 |
|---|---------|------|---------|
| 3.1 | Skill-MAS-Adversarial-Defense | GroupGuard + FlowSteer + Conjunctive | 2026爆发子方向，攻防对称 |
| 3.2 | Skill-Cross-Org-Agent-Protocol | MPAC + ACP | 与16-智能体工程 MCP-A2A Skill 互补 |
| 3.3 | Skill-MAS-Dynamic-KG-Collaboration | MemGraphRAG + MAGE | mas↔KG动态协同桥梁 |

**Sprint 3 交付物**：
- 3 个新 Skill 方向（共 5-6 张卡）
- mas ↔ knowledge_graph 跨域 edge 建立
- 整体图谱更新，总结报告

---

## 论文索引（全部 arXiv 已确认）

### Sprint 1 论文

#### 1.1 MAS 资源调度

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| HiveMind: OS-Inspired Scheduling for LLM Agent Workloads | [2604.17111](https://arxiv.org/abs/2604.17111) | 2026-04 | ⭐⭐⭐⭐⭐ |
| AgentRM: OS-Inspired Resource Manager | [2603.13110](https://arxiv.org/abs/2603.13110) | 2026-03 | ⭐⭐⭐⭐⭐ |
| On Time, Within Budget: MCPP | [2605.06110](https://arxiv.org/abs/2605.06110) | 2026-05 | ⭐⭐⭐⭐ |

**补充参考**：
- AMRO-S: Ant Colony Optimization for MAS Routing → [2603.12933](https://arxiv.org/abs/2603.12933)（ACO路由，可作为调度扩展）
- LAMaS: Latency-Aware Orchestration → [2601.10560](https://arxiv.org/abs/2601.10560)（关键路径优化）

#### 1.2 MAS 测试验证

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| FLARE: Coverage-Guided Fuzzing for MAS | [2604.05289](https://arxiv.org/abs/2604.05289) | 2026-04 | ⭐⭐⭐⭐ |
| MAESTRO: Multi-Agent Test Suite | [2601.00481](https://arxiv.org/abs/2601.00481) | 2026-01 | ⭐⭐⭐⭐⭐ |

**补充参考**：
- MAS-ProVe: Process Verification → [2602.03053](https://arxiv.org/abs/2602.03053)（验证稳定性分析）
- MAT: Message-Action Traces Assurance → [2603.18096](https://arxiv.org/abs/2603.18096)（契约式回放）

#### 1.3 动态信任管理

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| DynaTrust: Dynamic Trust Graph vs Sleeper Agents | [2603.15661](https://arxiv.org/abs/2603.15661) | 2026-03 | ⭐⭐⭐⭐⭐ |
| A-Trust: Attention-Based Trust Management | [2506.02546](https://arxiv.org/abs/2506.02546) | 2026-06 | ⭐⭐⭐⭐ |
| ECL: Epistemic Context Learning | [2601.21742](https://arxiv.org/abs/2601.21742) | 2026-01 | ⭐⭐⭐⭐ |

---

### Sprint 2 论文

#### 2.1 MAS × 广告自动竞价

| 论文 | arXiv/Venue | 年份 | 工程评级 |
|------|------------|------|---------|
| DARA: Few-shot Budget Allocation | [2601.14711](https://arxiv.org/abs/2601.14711) · WWW'26 | 2026-01 | ⭐⭐⭐⭐⭐ |
| LBM: Hierarchical Large Auto-Bidding Model | [2603.05134](https://arxiv.org/abs/2603.05134) · WWW'26 | 2026-03 | ⭐⭐⭐⭐⭐ |

**补充参考**：
- SemBid: Semantic Bidding Feature Fusion → [2605.05833](https://arxiv.org/abs/2605.05833)（语义+数值融合理论基础）
- MAS Marketing Attribution → ACM ICAIC 2026, [doi:10.1145/3807246.3807341](https://dl.acm.org/doi/10.1145/3807246.3807341)

#### 2.2 MAS 共识机制

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| Aegean: Consensus Protocol for Stochastic Agents | [2512.20184](https://arxiv.org/abs/2512.20184) | 2026-01 | ⭐⭐⭐⭐⭐ |
| DySCo: Dynamic Trust-Aware Sparse Consensus | [2606.01828](https://arxiv.org/abs/2606.01828) | 2026-06 | ⭐⭐⭐⭐ |
| SAC: Self-Anchored Consensus under Byzantine Faults | [2605.09076](https://arxiv.org/abs/2605.09076) | 2026-05 | ⭐⭐⭐⭐ |

**补充参考**：
- Coalition Formation / CoalT → [2604.14386](https://arxiv.org/abs/2604.14386)（博弈论联盟稳定性）

#### 2.3 大规模 Agent 集群管理

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| MegaFlow: Large-Scale Distributed Orchestration | [2601.07526](https://arxiv.org/abs/2601.07526) | 2026-01 | ⭐⭐⭐⭐⭐ |
| MonoScale: Scaling MAS with Monotonic Improvement | [2601.23219](https://arxiv.org/abs/2601.23219) | 2026-01 | ⭐⭐⭐⭐ |
| OrgAgent: Organize MAS like a Company | [2604.01020](https://arxiv.org/abs/2604.01020) | 2026-04 | ⭐⭐⭐⭐ |

**补充参考**：
- ROMA: Recursive Open Meta-Agent → [2602.01848](https://arxiv.org/abs/2602.01848)（递归分解长任务）
- Multi²: Hierarchical Decision-Making → [2606.03698](https://arxiv.org/abs/2606.03698)（含benchmark数据集）
- AdaptOrch → [2602.16873](https://arxiv.org/abs/2602.16873)（拓扑自适应选择）

---

### Sprint 3 论文

#### 3.1 MAS 安全攻防

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| GroupGuard: Collusive Attack Defense | [2603.13940](https://arxiv.org/abs/2603.13940) | 2026-03 | ⭐⭐⭐⭐ |
| FlowSteer/FlowGuard: Planning-Time Attack | [2605.11514](https://arxiv.org/abs/2605.11514) | 2026-05 | ⭐⭐⭐⭐ |
| Conjunctive Prompt Attacks | [2604.16543](https://arxiv.org/abs/2604.16543) | 2026-04 | ⭐⭐⭐⭐ |

**补充参考**：
- MAS-FIRE: Fault Injection Framework → [2602.19843](https://arxiv.org/abs/2602.19843)（15类故障分类法）
- ResMAS: Resilience via Topology+Prompt → [2601.04694](https://arxiv.org/abs/2601.04694)（GNN弹性预测）

#### 3.2 跨组织 Agent 协议

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| MPAC: Multi-Principal Agent Coordination | [2604.09744](https://arxiv.org/abs/2604.09744) | 2026-04 | ⭐⭐⭐⭐⭐ |
| ACP: Unified Agent Communication Protocol | [2602.15055](https://arxiv.org/abs/2602.15055) | 2026-02 | ⭐⭐⭐⭐ |

**补充参考**：
- AWCP: Workspace Delegation Protocol → [2602.20493](https://arxiv.org/abs/2602.20493)
- IETF MACP draft → [draft-li-dmsc-macp-05](https://datatracker.ietf.org/doc/html/draft-li-dmsc-macp-05)（标准化轨道参考）
- RAPS: Reputation-Aware Pub-Sub → [2602.08009](https://arxiv.org/abs/2602.08009)（自适应订阅协调）

#### 3.3 MAS × 知识图谱动态协同

| 论文 | arXiv | 年份 | 工程评级 |
|------|-------|------|---------|
| MemGraphRAG: Multi-Agent Dynamic KG | [2606.00610](https://arxiv.org/abs/2606.00610) | 2026-06 | ⭐⭐⭐⭐ |
| MAGE: Co-Evolutionary Knowledge Graphs | [2605.10064](https://arxiv.org/abs/2605.10064) | 2026-05 | ⭐⭐⭐⭐ |

**补充参考**：
- DIAL-KG: Streaming Incremental KG → [2603.20059](https://arxiv.org/abs/2603.20059)（实时KG更新三阶段）
- RoMem: KG Temporal Reasoning → [2604.11544](https://arxiv.org/abs/2604.11544)（时序KG推理SOTA）

---

## 通信效率方向（跨 Sprint 补充，视进度安排）

> 当前已有 G²CP 覆盖图结构通信，但 Latent 通信、Ad-hoc Teaming 是新范式

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| HyLaT: Hybrid Latent-Text Communication | [2605.25421](https://arxiv.org/abs/2605.25421) | Token使用降低10.6×，双通道通信 |
| CONCAT: Consensus-Driven Ad Hoc Teaming | [2605.29612](https://arxiv.org/abs/2605.29612) | 效率提升2.02×，无需训练 |
| MOC: Multi-Order Communication | [2606.02359](https://arxiv.org/abs/2606.02359) | 多跳证据流，6数据集一致提升 |

**建议**：HyLaT 萃取价值高（新范式），可在 Sprint 2 或 Sprint 3 插入。

---

## 形式化协调协议方向（跨 Sprint 补充）

> 现有 SDOF/DAG 是实用编排，ZipperGen 提供形式化正确性保证

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| ZipperGen: Provable Coordination via MSC | [2604.17612](https://arxiv.org/abs/2604.17612) | 形式化无死锁协调，ITU标准 |
| MPAC（已列入3.2） | [2604.09744](https://arxiv.org/abs/2604.09744) | 多委托人标准协议 |

---

## 萃取执行规范

### 每个 Skill 卡标准结构
参考 [llm-agent-engineering-paper-selection-draft-20260516.md](./llm-agent-engineering-paper-selection-draft-20260516.md) 中已萃取 Skill 格式：

```yaml
---
title: [论文核心贡献短名]
doc_type: knowledge
module: 10-MAS
topic: [kebab-case-topic]
source_paper: [arXiv ID 或 venue]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### 图谱 edge 更新规则

每完成一个 Skill 方向，同步更新以下 edges：

| 新 Skill | 应建立的 edges |
|---------|--------------|
| Skill-MAS-Resource-Scheduling | → Skill-Agent-Production-Engineering (extends) |
| Skill-MAS-Resource-Scheduling | → Skill-Cost-Aware-Agent-Scheduling (combinable) |
| Skill-MAS-Testing-Verification | → Skill-MASEval-System-Evaluation (extends) |
| Skill-MAS-Testing-Verification | → Skill-ReliabilityBench-Agent-Reliability (combinable) |
| Skill-MAS-Dynamic-Trust | → Skill-AgentTrust-Runtime-Safety-Interception (prerequisite) |
| Skill-MAS-Dynamic-Trust | → Skill-Agent-Safety-Guardrails (combinable) |
| Skill-LLM-AutoBidding-MAS | → Skill-ROAS-Budget-Optimization (extends) ← **跨域桥梁** |
| Skill-LLM-AutoBidding-MAS | → Skill-MAS-Orchestrator (prerequisite) |
| Skill-MAS-Dynamic-KG-Collaboration | → Skill-Helicase-Supply-Chain-KG-MAS (extends) ← **跨域桥梁** |
| Skill-MAS-Dynamic-KG-Collaboration | → Skill-Graph-Grounded-MAS-Protocol (combinable) |

### 硬约束（沿用项目规范）

1. **必须有开源代码或可复现实现**（或论文本身提供足够的算法描述可复现）
2. **必须能映射到至少一个业务场景**（广告/供应链/电商/推荐/客服）
3. **排除纯 survey**，除非覆盖空白极大且无更好选择
4. **论文年份**：2026 优先，2025 年下半年次之，更早需特殊理由

---

## 里程碑与验收

| 里程碑 | 时间 | 交付标准 |
|--------|------|---------|
| Sprint 1 完成 | 第2周末 | 6张Skill卡入库，INDEX更新，edges建立 |
| Sprint 2 完成 | 第4周末 | 6-7张Skill卡，mas↔advertising桥梁建立 |
| Sprint 3 完成 | 第6周末 | 5-6张Skill卡，mas↔KG桥梁建立，全图报告更新 |
| 全部完成 | 第6周末 | MAS领域 Skills 从 25 增至约 40，新增3条跨域桥梁 |

---

## 历史参考

| 类似选题 | 完成日期 | 参考文件 |
|---------|---------|---------|
| 16-LLM-Agent-Engineering 首批萃取 | 2026-05-16 | [llm-agent-engineering-paper-selection-draft-20260516.md](./llm-agent-engineering-paper-selection-draft-20260516.md) |
| semantic-blueprint 选题 | 2026-04-28 | [semantic-blueprint-paper-selection-draft-20260428.md](./semantic-blueprint-paper-selection-draft-20260428.md) |
