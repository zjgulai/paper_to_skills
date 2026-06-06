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

## 图谱填充三原则（优先级顺序）

```
原则 1: missing_prerequisite 断链 → 最高优先，阻塞下游
原则 2: 孤立节点关联 → 已有 Skill 接入图谱主体
原则 3: 延伸链 + 跨域桥梁 → 扩大图谱覆盖密度
```

**对应到 MAS 2026 计划的映射逻辑：**

| 图谱规则 | MAS 场景 | 结论 |
|---------|---------|------|
| prerequisite 断链阻塞下游 | `Skill-Dynamic-Trust` 是 `AgentTrust` 的前置，缺失会造成断链 | **最先萃取** |
| 孤立节点已存在但零连接 | `Skill-MAS-Testing-Verification` 目前无任何 extends/combinable 指向 | **第二批萃取** |
| Hub Skill 高被依赖 → 优先补强其延伸链 | `Skill-MAS-Orchestrator` 被依赖多，其生产化方向（资源调度）是自然延伸 | **第三批** |
| 跨域桥梁 → 最后，连通大图 | `mas↔advertising`、`mas↔KG` 需要双侧 Skills 都成熟后再建桥 | **最后批次** |

---

## 当前 MAS 图谱结构分析

### Layer 分布（按图谱操作手册四层体系）

```
Layer 1 基础层 ──── ReAct / ToT / AutoGen / CAMEL / MetaGPT
Layer 2 核心层 ──── Reflexion / MAS Orchestrator / Subagent Decomposition
Layer 3 进阶层 ──── Dynamic DAG / SDOF / ParaManager / G²CP / MASEval
Layer 4 桥接层 ──── Agent-Production-Engineering (MAS↔LLM-Eng)
                    Flowr (MAS↔供应链) / AIM-RM / AgenticPay / Event-Driven-Demand
```

### 现有 prerequisite 链（已建立）

```
ReAct → MAS Orchestrator → Subagent Decomposition → Dynamic DAG
AutoGen/MetaGPT → MAS Orchestrator
EvoSC → Self-Improving-Agent
G²CP → Graph-Grounded Communication
```

### 新增 9 个方向的依赖拓扑分析

```
[信任层] Dynamic Trust
    prerequisite ← AgentTrust-Runtime-Safety (16-智能体工程，已存在)
    extends → Agent Safety Guardrails (16-智能体工程，已存在)
    ★ 现在是 AgentTrust 的 prerequisite 断链 → 必须最先萃取

[测试层] MAS Testing Verification
    prerequisite ← MASEval (已存在)
    extends → ReliabilityBench (16-智能体工程，已存在)
    ★ 孤立节点问题：MASEval 有延伸指向空白 → 第二批

[资源调度] MAS Resource Scheduling
    prerequisite ← MAS Orchestrator (已存在)
    extends → Cost-Aware-Agent-Scheduling (16-智能体工程，已存在)
    ★ Orchestrator 的自然延伸链 → 第三批

[共识机制] MAS Consensus Mechanism
    prerequisite ← Multi-Agent-Debate (已存在)
    extends → G²CP (已存在)
    ★ G²CP 的深化延伸 → 第三批

[规模化管理] MAS Scale Management
    prerequisite ← MAS Orchestrator + Skill Registry (已存在)
    extends → Agent Registry Discovery (已存在)
    ★ Orchestrator 延伸链的另一分支 → 第四批

[广告桥梁] LLM AutoBidding MAS
    prerequisite ← MAS Orchestrator (已存在) + ROAS-Budget (13-广告，已存在)
    ★ 跨域桥梁，双侧前置已备 → 第四批

[安全攻防] MAS Adversarial Defense
    prerequisite ← Dynamic Trust (新增，本计划) + Agent Safety Guardrails (已存在)
    ★ 依赖 Dynamic Trust，必须 Dynamic Trust 完成后 → 第五批

[跨组织协议] Cross-Org Agent Protocol
    prerequisite ← G²CP (已存在) + MCP-A2A (16-智能体工程，已存在)
    ★ 协议层延伸 → 第五批

[KG动态协同] MAS Dynamic KG Collaboration
    prerequisite ← Graph-Grounded MAS Protocol (已存在) + Helicase (已存在)
    ★ 跨域桥梁，双侧前置已备 → 第六批
```

---

## 萃取顺序（拓扑排序结果）

### 🔴 第一波：断链修复（Week 1）

**核心逻辑**：`AgentTrust-Runtime-Safety-Interception`（16-智能体工程，已存在）需要一个 prerequisite，当前断链。`Skill-MAS-Dynamic-Trust` 就是这个缺失的前置。不萃取它，整个信任链路在图谱中是孤立的。

---

#### W1-1 · `Skill-MAS-Dynamic-Trust`

**图谱角色**：Layer 3 进阶层 → 修复 `AgentTrust` 断链，同时为后续 `Skill-MAS-Adversarial-Defense` 提供 prerequisite

**核心论文（3篇，互补组合）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **DynaTrust**: Dynamic Trust Graph vs Sleeper Agents | [2603.15661](https://arxiv.org/abs/2603.15661) | Beta分布Bayesian信任平滑+陪审团共识，DSR 92.4% | 主干算法：动态信任图构建 |
| **A-Trust**: Attention-Based Trust Management | [2506.02546](https://arxiv.org/abs/2506.02546) | 6维信任+LLM Attention量化，无需外部验证器 | 补充：无侵入式信任评估 |
| **ECL**: Epistemic Context Learning | [2601.21742](https://arxiv.org/abs/2601.21742) | 两阶段解耦+RL，4B模型超越无感知30B | 补充：历史感知信任聚合 |

**Skill 卡结构要点**：
- 算法原理：动态信任图（DTG）构建 + Beta 分布 Bayesian 更新 + 接收端评分（非发送端自报）
- 业务场景：多 Agent 采购谈判（AgenticPay）+ 库存决策（AIM-RM）中的可信度校验
- 代码模板：`DynamicTrustGraph` 类，含 `update_trust(agent_id, message, outcome)` + `get_trusted_agents(threshold)`

**图谱 edges（萃取后立即添加）**：
```
Skill-MAS-Dynamic-Trust
  prerequisite ← Skill-Multi-Agent-Debate                    (辩论→信任评估的自然前置)
  prerequisite ← Skill-CAMEL-Role-Playing-Agents             (角色扮演中的信任建立)
  extends      → Skill-AgentTrust-Runtime-Safety-Interception (修复断链！)
  extends      → Skill-Agent-Safety-Guardrails               (信任→安全防护延伸)
  combinable   ↔ Skill-Graph-Grounded-MAS-Protocol           (图结构通信+信任双轨)
  combinable   ↔ Skill-MASEval-System-Evaluation             (评估框架中的信任维度)
```

---

### 🟠 第二波：孤立节点接入（Week 1-2）

**核心逻辑**：`MASEval` 已存在但其 extends 链断裂（没有指向"如何做测试"的下游），`ReliabilityBench`（16-智能体工程）也是孤立的。`Skill-MAS-Testing-Verification` 同时接通这两个断点。

---

#### W2-1 · `Skill-MAS-Testing-Verification`

**图谱角色**：Layer 3 进阶层 → 接通 `MASEval` 的延伸链，同时为 `ReliabilityBench` 提供 prerequisite

**核心论文（2篇主干 + 2篇补充）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **FLARE**: Coverage-Guided Fuzzing for MAS | [2604.05289](https://arxiv.org/abs/2604.05289) | 首个MAS Fuzzing，发现56个未知失败，96.9%覆盖率 | 主干：测试方法论 |
| **MAESTRO**: Multi-Agent Test Suite | [2601.00481](https://arxiv.org/abs/2601.00481) | OpenTelemetry跨框架追踪，12个MAS架构>模型选择 | 主干：测试工具链 |
| MAS-ProVe (补充) | [2602.03053](https://arxiv.org/abs/2602.03053) | 过程验证不稳定性分析 | 反面案例 |
| MAT Assurance (补充) | [2603.18096](https://arxiv.org/abs/2603.18096) | 契约式回放+故障注入 | 治理层补充 |

**Skill 卡结构要点**：
- 算法原理：覆盖制导模糊测试（FLARE）+ 基于 OpenTelemetry 的跨框架执行轨迹采集（MAESTRO）
- 业务场景：上线前 MAS 回归测试、CI/CD 集成的自动化 inter-agent 覆盖率验证
- 代码模板：`MASTestOracle` 类，含 `inject_fault(agent_id, fault_type)` + `measure_coverage()`

**图谱 edges**：
```
Skill-MAS-Testing-Verification
  prerequisite ← Skill-MASEval-System-Evaluation             (评估→测试的自然延伸)
  prerequisite ← Skill-Agent-Production-Engineering          (生产化前提：测试)
  extends      → Skill-ReliabilityBench-Agent-Reliability    (接通孤立节点！)
  extends      → Skill-Agent-Stage-Evaluation                (16-智能体工程)
  combinable   ↔ Skill-MAS-Dynamic-Trust                     (测试×信任双轨验证)
  combinable   ↔ Skill-Atomix-Transactional-Tool-Calls       (事务性工具调用可测性)
```

---

### 🟡 第三波：Hub Skill 延伸链（Week 2-3）

**核心逻辑**：`Skill-MAS-Orchestrator` 是 MAS 领域的 Hub（高被依赖数）。其延伸链有两个自然方向：① 资源调度（生产化）② 共识机制（理论深化）。按"技术栈完整性"先补调度（工程），再补共识（理论）。

---

#### W3-1 · `Skill-MAS-Resource-Scheduling`

**图谱角色**：Layer 3 进阶层 → `MAS-Orchestrator` 的生产化延伸，同时接通 `Cost-Aware-Agent-Scheduling` 断链

**核心论文（3篇主干）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **HiveMind**: OS-Inspired Scheduling | [2604.17111](https://arxiv.org/abs/2604.17111) | 5大OS调度原语，并发失败率72%→0，MIT开源 | 主干：调度机制 |
| **AgentRM**: OS-Inspired Resource Manager | [2603.13110](https://arxiv.org/abs/2603.13110) | MLFQ+三层上下文生命周期，P95延迟降86% | 主干：资源管理 |
| **MCPP** (On Time, Within Budget) | [2605.06110](https://arxiv.org/abs/2605.06110) | 双硬约束（预算+截止时间）最大化完成率 | 补充：SLA约束 |

**图谱 edges**：
```
Skill-MAS-Resource-Scheduling
  prerequisite ← Skill-MAS-Orchestrator                      (编排→调度的自然前置)
  prerequisite ← Skill-Agent-Production-Engineering          (生产化基础)
  extends      → Skill-Cost-Aware-Agent-Scheduling           (接通孤立节点！)
  extends      → Skill-Agent-SLO-Manager                     (SLO管理的实现层)
  combinable   ↔ Skill-MAS-Testing-Verification              (调度+测试的生产运维组合)
  combinable   ↔ Skill-ParaManager-Parallel-Orchestration    (并行调度互补)
```

---

#### W3-2 · `Skill-MAS-Consensus-Mechanism`

**图谱角色**：Layer 3 进阶层 → `Multi-Agent-Debate` 的理论深化，`G²CP` 的算法基础补充

**核心论文（3篇）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **Aegean**: Consensus Protocol | [2512.20184](https://arxiv.org/abs/2512.20184) | quorum detection，延迟降1.2-20×，形式化证明 | 主干：分布式共识 |
| **DySCo**: Dynamic Sparse Consensus | [2606.01828](https://arxiv.org/abs/2606.01828) | 动态边价值评估，Token消耗-70% | 主干：稀疏共识 |
| **SAC**: Byzantine Fault Consensus | [2605.09076](https://arxiv.org/abs/2605.09076) | MSR算法+r-robustness形式化容错 | 补充：Byzantine容错 |

**图谱 edges**：
```
Skill-MAS-Consensus-Mechanism
  prerequisite ← Skill-Multi-Agent-Debate                    (辩论是非正式共识，本Skill是形式化版)
  prerequisite ← Skill-Graph-Grounded-MAS-Protocol           (图通信→共识协议)
  extends      → Skill-Agent-QMix-Topology-Learning          (拓扑学习+共识互补)
  combinable   ↔ Skill-MAS-Dynamic-Trust                     (共识×信任双保障)
  combinable   ↔ Skill-SDOF-State-Constrained-Orchestration  (状态约束+共识状态机)
```

---

### 🟢 第四波：规模化 + 跨域桥梁（Week 3-4）

**核心逻辑**：完成 Layer 3 补强后，扩展 Layer 4 桥接层。`MAS-Scale-Management` 是 `Agent-Registry-Discovery` 的上层架构，先补强领域内的规模化能力，再建跨域桥梁 `mas↔advertising`。

---

#### W4-1 · `Skill-MAS-Scale-Management`

**图谱角色**：Layer 4 桥接层 → `Agent-Registry-Discovery` 的架构延伸

**核心论文（3篇主干）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **MegaFlow**: Large-Scale Distributed Orchestration | [2601.07526](https://arxiv.org/abs/2601.07526) | 三服务解耦，万级并发，130,000+生产记录 | 主干：规模化架构 |
| **MonoScale**: Monotonic Improvement Scaling | [2601.23219](https://arxiv.org/abs/2601.23219) | Agent池扩展性能单调保证，Bandit形式化 | 主干：动态扩缩容 |
| **OrgAgent**: MAS like a Company | [2604.01020](https://arxiv.org/abs/2604.01020) | 公司制三层架构，F1+102%，Token-74% | 补充：层次化组织 |

**图谱 edges**：
```
Skill-MAS-Scale-Management
  prerequisite ← Skill-Agent-Registry-Discovery              (发现→规模化管理的前置)
  prerequisite ← Skill-MAS-Orchestrator                      (编排→规模化的基础)
  extends      → Skill-Dynamic-DAG-Orchestration             (DAG在大规模下的运行)
  combinable   ↔ Skill-MAS-Resource-Scheduling               (规模化+资源调度组合)
  combinable   ↔ Skill-ParaManager-Parallel-Orchestration    (并行管理互补)
```

---

#### W4-2 · `Skill-LLM-AutoBidding-MAS` ← **`mas ↔ advertising` 跨域桥梁**

**图谱角色**：Layer 4 桥接层 → 首条 `mas ↔ advertising` 桥梁，连通两个最大领域（25+20=45 Skills）

**核心论文（2篇 WWW'26）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **DARA**: Few-shot Budget Allocation | [2601.14711](https://arxiv.org/abs/2601.14711) · WWW'26 | 双Agent(Reasoner+Optimizer)+GRPO-Adaptive | 主干：双Agent架构 |
| **LBM**: Hierarchical Large Auto-Bidding | [2603.05134](https://arxiv.org/abs/2603.05134) · WWW'26 | Think+Act双层+GQPO消除竞价幻觉 | 主干：推理-执行分离 |

**图谱 edges**（跨域桥梁必须双向建立）：
```
Skill-LLM-AutoBidding-MAS
  prerequisite ← Skill-MAS-Orchestrator                      (MAS侧前置)
  prerequisite ← Skill-ROAS-Budget-Optimization              (广告侧前置，13-广告)
  extends      → Skill-AgenticPay-Procurement-Negotiation    (MAS侧延伸：竞价→谈判)
  extends      → Skill-Ad-Attribution-Modeling               (广告侧延伸，13-广告)
  combinable   ↔ Skill-Multi-Agent-Debate                    (竞价策略中的多方博弈)
  combinable   ↔ Skill-DARA-Agentic-MMM                      (15-营销，预算分配闭环)
  [BRIDGE EDGE] mas ↔ advertising                           ← 建立跨域连接
```

---

### 🔵 第五波：安全攻防 + 协议标准化（Week 4-5）

**核心逻辑**：`MAS-Adversarial-Defense` 依赖 `Dynamic-Trust`（第一波），必须等其完成。`Cross-Org-Agent-Protocol` 是 `G²CP` + `MCP-A2A` 的协议上层，需要前两者稳定后再建。

---

#### W5-1 · `Skill-MAS-Adversarial-Defense`

**图谱角色**：Layer 3 进阶层 → `Dynamic-Trust` 的应用场景延伸，`Agent-Safety-Guardrails` 的 MAS 级别补充

**核心论文（3篇攻防对称）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **GroupGuard**: Collusive Attack Defense | [2603.13940](https://arxiv.org/abs/2603.13940) | 群体合谋攻击形式化，88%检测准确率 | 主干：群体攻击防御 |
| **FlowSteer/FlowGuard**: Planning-Time | [2605.11514](https://arxiv.org/abs/2605.11514) | 一条prompt劫持DAG规划，+FlowGuard防御 | 主干：规划时攻击 |
| **Conjunctive Attacks** | [2604.16543](https://arxiv.org/abs/2604.16543) | 路由合取绕过现有全系列防御 | 补充：新型攻击向量 |

**图谱 edges**：
```
Skill-MAS-Adversarial-Defense
  prerequisite ← Skill-MAS-Dynamic-Trust                     (信任层是攻防的基础)
  prerequisite ← Skill-Agent-Safety-Guardrails               (单Agent安全→MAS安全)
  prerequisite ← Skill-SDOF-State-Constrained-Orchestration  (状态约束防御前提)
  extends      → Skill-AgentTrust-Runtime-Safety-Interception (运行时拦截升级)
  combinable   ↔ Skill-MAS-Consensus-Mechanism               (共识+攻防双保障)
  combinable   ↔ Skill-MAS-Testing-Verification              (安全测试组合)
```

---

#### W5-2 · `Skill-Cross-Org-Agent-Protocol`

**图谱角色**：Layer 4 桥接层 → `MCP-A2A-Protocol-Stack`（16-智能体工程）的上层架构，跨组织边界

**核心论文（2篇主干 + IETF 标准参考）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **MPAC**: Multi-Principal Coordination | [2604.09744](https://arxiv.org/abs/2604.09744) | 21消息类型，5层协调语义，有PyPI包 | 主干：多委托人协议 |
| **ACP**: Federated Agent Communication | [2602.15055](https://arxiv.org/abs/2602.15055) | DID身份+动态SLA，通信延迟-40% | 主干：联邦编排 |
| IETF MACP draft | [draft-li-dmsc-macp-05](https://datatracker.ietf.org/doc/html/draft-li-dmsc-macp-05) | 标准化轨道 | 补充：协议标准参考 |

**图谱 edges**：
```
Skill-Cross-Org-Agent-Protocol
  prerequisite ← Skill-Graph-Grounded-MAS-Protocol           (图通信→跨组织协议)
  prerequisite ← Skill-MCP-A2A-Protocol-Stack                (16-智能体工程，协议栈基础)
  extends      → Skill-Agent-Registry-Discovery              (注册发现的跨组织扩展)
  combinable   ↔ Skill-MAS-Dynamic-Trust                     (跨组织=跨信任边界)
  combinable   ↔ Skill-LDP-Identity-Aware-Protocol           (16-智能体工程，身份认证)
```

---

### 🟣 第六波：KG 跨域桥梁（Week 5-6）

**核心逻辑**：`mas ↔ knowledge_graph` 桥梁需要 MAS 侧（`Graph-Grounded-MAS-Protocol` + `Helicase`）和 KG 侧（`Audience-KG`、`CausalRAG`）都成熟后建立。当前 KG 侧已有 16 个 Skills，时机成熟。

---

#### W6-1 · `Skill-MAS-Dynamic-KG-Collaboration` ← **`mas ↔ knowledge_graph` 跨域桥梁**

**图谱角色**：Layer 4 桥接层 → `Helicase`（静态KG）的动态化演进，连通 MAS 与 KG 两大领域

**核心论文（2篇主干 + 2篇补充）**：

| 论文 | arXiv | 核心贡献 | 萃取重点 |
|------|-------|---------|---------|
| **MemGraphRAG**: Multi-Agent Dynamic KG | [2606.00610](https://arxiv.org/abs/2606.00610) | 三层共享记忆+三Agent协同，冲突解决闭环，0.061s检索 | 主干：动态KG构建 |
| **MAGE**: Co-Evolutionary Knowledge Graphs | [2605.10064](https://arxiv.org/abs/2605.10064) | 四子图协同演化KG，9benchmark验证 | 主干：KG协同进化 |
| DIAL-KG (补充) | [2603.20059](https://arxiv.org/abs/2603.20059) | 流式增量KG构建三阶段 | 补充：实时更新 |
| RoMem (补充) | [2604.11544](https://arxiv.org/abs/2604.11544) | 时序KG推理SOTA，MRR 72.6 | 补充：时序推理 |

**图谱 edges**（双侧桥梁）：
```
Skill-MAS-Dynamic-KG-Collaboration
  prerequisite ← Skill-Helicase-Supply-Chain-KG-MAS          (静态KG→动态KG进化)
  prerequisite ← Skill-Graph-Grounded-MAS-Protocol           (图通信基础)
  prerequisite ← Skill-GraphRAG-Knowledge-Enhanced           (08-知识图谱，RAG基础)
  extends      → Skill-AIM-RM-LLM-Inventory-MAS-Memory       (记忆KG应用到库存)
  combinable   ↔ Skill-AgeMem-Unified-Agent-Memory           (16-智能体工程，记忆管理)
  combinable   ↔ Skill-MAS-Dynamic-Trust                     (知识可信度+Agent信任)
  [BRIDGE EDGE] mas ↔ knowledge_graph                       ← 建立跨域连接
```

---

## 完整时序总览

```
Week 1 ────────────────────────────────────────────────────────────
│  W1-1: Skill-MAS-Dynamic-Trust                                    │
│         [修复 AgentTrust 断链，为 Adversarial-Defense 铺路]       │
│         论文: DynaTrust + A-Trust + ECL                           │
│                                                                    │
│  W2-1: Skill-MAS-Testing-Verification                             │
│         [接通 MASEval 延伸链，修复 ReliabilityBench 孤立]         │
│         论文: FLARE + MAESTRO                                      │
└──────────────────────────────────────────────────────────────────

Week 2 ────────────────────────────────────────────────────────────
│  W3-1: Skill-MAS-Resource-Scheduling                              │
│         [MAS-Orchestrator 生产化延伸链]                           │
│         论文: HiveMind + AgentRM + MCPP                           │
│                                                                    │
│  W3-2: Skill-MAS-Consensus-Mechanism                              │
│         [Multi-Agent-Debate 理论深化]                              │
│         论文: Aegean + DySCo + SAC                                 │
└──────────────────────────────────────────────────────────────────

Week 3 ────────────────────────────────────────────────────────────
│  W4-1: Skill-MAS-Scale-Management                                 │
│         [Agent-Registry 架构延伸]                                  │
│         论文: MegaFlow + MonoScale + OrgAgent                      │
│                                                                    │
│  W4-2: Skill-LLM-AutoBidding-MAS  ★ mas↔advertising 桥梁         │
│         [首条跨域桥梁，连通广告领域]                               │
│         论文: DARA + LBM (WWW'26 × 2)                             │
└──────────────────────────────────────────────────────────────────

Week 4 ────────────────────────────────────────────────────────────
│  W5-1: Skill-MAS-Adversarial-Defense                              │
│         [Dynamic-Trust 的应用场景延伸]                             │
│         论文: GroupGuard + FlowSteer + Conjunctive                 │
│                                                                    │
│  W5-2: Skill-Cross-Org-Agent-Protocol                             │
│         [G²CP + MCP-A2A 的协议上层]                               │
│         论文: MPAC + ACP + IETF MACP                              │
└──────────────────────────────────────────────────────────────────

Week 5-6 ──────────────────────────────────────────────────────────
│  W6-1: Skill-MAS-Dynamic-KG-Collaboration  ★ mas↔KG 桥梁         │
│         [Helicase 静态KG→动态协同演进]                             │
│         论文: MemGraphRAG + MAGE                                   │
└──────────────────────────────────────────────────────────────────
```

---

## 图谱增量预测

| 阶段 | 新增 Skill | 新增 edges | 修复断链/孤立 | 累计节点 | 累计边 |
|------|-----------|-----------|-------------|---------|--------|
| 基线 | — | — | — | 287 | 5,152 |
| Week 1 (W1+W2) | +2 | +12 | 2个断链修复 | 289 | 5,164 |
| Week 2 (W3-1+W3-2) | +2 | +12 | Orchestrator延伸链 | 291 | 5,176 |
| Week 3 (W4-1+W4-2) | +2 | +12 | mas↔advertising桥梁 | 293 | 5,188 |
| Week 4 (W5-1+W5-2) | +2 | +12 | 安全链完整 | 295 | 5,200 |
| Week 5-6 (W6-1) | +1 | +7 | mas↔KG桥梁 | 296 | 5,207 |

**结构性改善**：
- 断链修复：2 条（AgentTrust、ReliabilityBench）
- 孤立节点接通：2 个
- 新增跨域桥梁：2 条（mas↔advertising、mas↔KG）
- MAS 领域 Skills：25 → 34

---

## 每个 Skill 萃取后的验收检查

参考项目规范（`知识图谱架构与分类体系.md` § 附录C）：

```
□ ① 算法原理：核心论文方法，含关键公式或伪代码
□ ② 业务场景：母婴跨境/DTC 具体场景，有量化指标
□ ③ 代码模板：可运行 Python 代码，含核心类和方法签名
□ ④ 图谱关联：prerequisite ≥1 / extends ≥1 / combinable ≥2
□ ⑤ 审核评分：4 维度各 ≥1.5，总分 ≥7/10
```

**特殊要求（MAS 2026 方向）**：
- 所有跨域桥梁 Skill（W4-2, W6-1）：双侧 prerequisite 必须已存在再萃取
- 安全方向（W5-1）：必须 W1-1 Dynamic Trust 完成后再开始

---

## 论文快速索引

| Skill | 主干论文 arXiv | 补充论文 arXiv |
|-------|--------------|--------------|
| W1-1 Dynamic Trust | 2603.15661 · 2506.02546 | 2601.21742 |
| W2-1 Testing Verification | 2604.05289 · 2601.00481 | 2602.03053 · 2603.18096 |
| W3-1 Resource Scheduling | 2604.17111 · 2603.13110 | 2605.06110 |
| W3-2 Consensus Mechanism | 2512.20184 · 2606.01828 | 2605.09076 |
| W4-1 Scale Management | 2601.07526 · 2601.23219 | 2604.01020 |
| W4-2 AutoBidding MAS | 2601.14711 · 2603.05134 | — |
| W5-1 Adversarial Defense | 2603.13940 · 2605.11514 | 2604.16543 |
| W5-2 Cross-Org Protocol | 2604.09744 · 2602.15055 | IETF draft |
| W6-1 Dynamic KG | 2606.00610 · 2605.10064 | 2603.20059 · 2604.11544 |

---

## 历史参考

| 类似选题 | 完成日期 | 参考文件 |
|---------|---------|---------|
| 16-LLM-Agent-Engineering 首批萃取 | 2026-05-16 | [llm-agent-engineering-paper-selection-draft-20260516.md](./llm-agent-engineering-paper-selection-draft-20260516.md) |
| semantic-blueprint 选题 | 2026-04-28 | [semantic-blueprint-paper-selection-draft-20260428.md](./semantic-blueprint-paper-selection-draft-20260428.md) |
| MAS 2026 原始选题（未排序版） | 2026-06-04 | [mas-2026-paper-selection-plan-20260604.md](./mas-2026-paper-selection-plan-20260604.md) |
