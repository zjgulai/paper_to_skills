---
title: 16-LLM-Agent-Engineering 选题方向与 2026 论文推荐
doc_type: analysis
module: paper-选题
topic: llm-agent-engineering
status: draft
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# 16-LLM-Agent-Engineering 选题方向与 2026 论文推荐

## 1. 选题动机与边界

### 1.1 为什么新建顶级领域

当前 vault 已有：

- `09-DataAgent-LLM/`:DataAgent + LLM-powered analytics 应用层
- `10-MAS/`:Multi-Agent System 算法与架构理论

新选题 `16-LLM-Agent-Engineering/` 与上述两者的边界:

| 领域 | 关注层面 | 典型问题 |
|------|----------|----------|
| 09-DataAgent-LLM | 业务应用 | DataAgent 怎么帮分析师做数据分析 |
| 10-MAS | 系统架构 | 多智能体怎么协作、角色怎么分配 |
| **16-LLM-Agent-Engineering** | **工程实践** | **Skill/Tool 怎么设计、Context 怎么管理、协议怎么落地** |

### 1.2 4 个聚焦方向（已与用户确认）

1. **Agent Skills/Tools 工程**：技能复用、工具选择、Skill-based 任务分解
2. **Context Engineering**：长上下文管理、Memory、Compaction 策略
3. **MAS 多智能体协作**：角色分配、通信协议、Orchestrator-Worker
4. **LLM 模型本身能力**：Hermes、Function Calling、Tool Use 训练

### 1.3 实践性硬约束

- 必须有 GitHub 开源代码或可复现实现
- 必须能映射到母婴跨境电商场景（VOC/Funnel/客服/广告/推荐）
- 排除纯理论 framework paper、survey-only paper

---

## 2. 推荐论文清单（2026 优先）

### 方向 1：Agent Skills / Tools 工程

**⭐ 核心推荐 1：SoK: Agentic Skills — Beyond Tool Use in LLM Agents**

- arxiv: [2602.20867](https://arxiv.org/abs/2602.20867)（2026-02）
- 一句话:把"技能层"作为一种独立于 prompt/tool 的抽象,提出技能全生命周期（discovery → practice → distillation → storage → composition → evaluation → update）的双轴 taxonomy
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：直接对应本项目的 Skill 卡设计哲学,可作为 Skill 卡格式 V2 的理论支撑
- 与项目契合度：极高（本项目就是把论文转 Skill 卡）

**⭐ 核心推荐 2：Agent Skills for LLMs - Architecture, Acquisition, Security**

- arxiv: [2602.12430](https://arxiv.org/abs/2602.12430)（2026-02）
- 一句话:首次系统化 SKILL.md 规范 + Progressive Disclosure + MCP 集成,提出 26.1% 社区 Skill 存在漏洞的安全分析
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：作为本项目 paper-审核 阶段的安全检查清单依据
- 与项目契合度：极高

**推荐 3：EvoSkills - Self-Evolving Agent Skills via Co-Evolutionary Verification**

- arxiv: [2604.01687](https://arxiv.org/html/2604.01687v1)（2026-04）
- 一句话:解决 tool-skill gap,通过协同进化验证让 Skill 自动迭代,配套 SkillsBench(87 任务, 20 领域, 二元 pass/fail 验证)
- 工程价值：⭐⭐⭐⭐
- 落地建议：可作为本项目 evolve/ 机制的算法基础

**推荐 4：SkillForge - Domain-Specific Self-Evolving Skills in Cloud Tech Support**

- arxiv: [2604.08618](https://arxiv.org/html/2604.08618v1)（2026-04,SIGIR 2026 Industry Track）
- 一句话:Tool Mining + Knowledge Extraction + Skill Synthesis 三步法从客服历史对话中自动生成 Skill
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：直接对应跨境电商客服场景,从 VOC + 客服记录中萃取 Skill
- 与项目契合度：极高（业务场景几乎完全对齐）

### 方向 2：Context Engineering 上下文工程

**⭐ 核心推荐 5：ACON - Optimizing Context Compression for Long-horizon LLM Agents**

- arxiv: [2510.00615](https://arxiv.org/abs/2510.00615)（2025-10,持续被 2026 工作引用）
- 一句话:用失败驱动的自然语言压缩准则优化,实测内存降低 26-54%, 准确率保留 95%+
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：可用于本项目长期客服对话/长篇 VOC 分析场景

**⭐ 核心推荐 6：Agentic Memory (AgeMem)**

- arxiv: [2601.01885](https://arxiv.org/abs/2601.01885)（2026-01）
- 一句话:把 LTM/STM 管理操作变成 tool-based actions,用三阶段渐进式 RL + step-wise GRPO 训练
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：母婴跨境电商客户长期偏好记忆（用户 1-3 年内购买周期）

**推荐 7：Active Context Compression (Focus)**

- arxiv: [2601.07190](https://arxiv.org/abs/2601.07190)（2026-01）
- 一句话:仿生粘菌探索策略,Agent 自主决策何时把关键学习固化为 Knowledge 块并主动剪枝原始历史
- 工程价值：⭐⭐⭐⭐

**推荐 8：Memory as Action (MemAct)**

- arxiv: [2510.12635](https://arxiv.org/html/2510.12635v1)（2025-10,持续更新）
- 一句话:把 Memory 操作整合进 policy,提出 Dynamic Context Policy Optimization 解决 trajectory fractures
- 工程价值：⭐⭐⭐⭐

**推荐 9：Memory for Autonomous LLM Agents（Survey）**

- arxiv: [2603.07670](https://arxiv.org/html/2603.07670v1)（2026-03）
- 一句话:覆盖 2022-2026 早期的 Memory 工作,提出三维 taxonomy（temporal scope / representational substrate / control policy）和 5 大机制族
- 工程价值：⭐⭐⭐⭐
- 落地建议：作为本项目 Memory 选型的决策依据

### 方向 3：MAS 多智能体协作（与 10-MAS 互补,聚焦工程协议层）

**⭐ 核心推荐 10：The Orchestration of Multi-Agent Systems - Architectures, Protocols, Enterprise Adoption**

- arxiv: [2601.13671](https://arxiv.org/html/2601.13671v1)（2026-01）
- 一句话:系统对比 MCP（Host↔Server）vs A2A（Agent↔Agent）双协议栈,给出企业级落地路径
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：本项目 paper-同步 等子流程的协议选择参考

**推荐 11：RL for LLM-MAS through Orchestration Traces**

- arxiv: [2605.02801](https://arxiv.org/abs/2605.02801)（2026-05）
- 一句话:用 orchestration trace（时序交互图）训 RL,识别出 8 类 reward family 和 5 类 orchestration 决策（spawn/delegate/communicate/aggregate/stop）
- 工程价值：⭐⭐⭐⭐
- 落地建议：可指导 MAS 项目的 reward shaping

**推荐 12：AdaptOrch - Task-Adaptive Multi-Agent Orchestration**

- arxiv: [2602.16873](https://arxiv.org/html/2602.16873v1)（2026-02）
- 一句话:提出 Adaptive Synthesis Protocol（含可证终止性）,topology-aware 比 static topology 提升 12-23%
- 工程价值：⭐⭐⭐⭐
- 落地建议：本项目 paper-workflow 可借鉴动态拓扑

**推荐 13：Multi-Agent LLM Orchestration for Incident Response**

- arxiv: [2511.15755](https://arxiv.org/abs/2511.15755)（2025-11,但工程价值高,列入参考）
- 一句话:348 controlled trials 显示 MAS 比单 Agent 在 actionable 推荐率上 1.7% → 100%,80 倍提升
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：跨境电商售后/退货异常处理 MAS 设计

### 方向 4：LLM Function Calling / Tool Use（Hermes 系列 + 训练方法）

**⭐ 核心推荐 14：Hermes 4 Technical Report**

- arxiv: [2508.18255](https://arxiv.org/pdf/2508.18255)（2025-08,Hermes 系列最新）
- 一句话:整合 self-reflective reasoning + 广泛指令竞争力,30k 截断 SFT 训练策略,中性对齐
- 工程价值：⭐⭐⭐⭐⭐
- 注:Hermes 系列论文本身是 2024-2025,但 hermes-agent（GitHub）持续更新到 2026
- 落地建议：开源 tool calling 基座模型选型参考

**⭐ 核心推荐 15：Small Language Models for Efficient Agentic Tool Calling**

- arxiv: [2512.15943](https://arxiv.org/abs/2512.15943)（2025-12,2026-03 更新）
- 一句话:目标 fine-tuning 让 SLM 在 tool calling 上超过大模型,显著降本
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：跨境电商客服/工单分类等高频低复杂度场景的成本优化

**推荐 16：Data-Driven Function Calling for Online Financial QA**

- arxiv: [2604.05387](https://arxiv.org/html/2604.05387v1)（2026-04）
- 一句话:xLAM 标注 + SFT + RL 两阶段,数据集随用户查询周期性更新
- 工程价值：⭐⭐⭐⭐
- 落地建议：跨境电商财务/订单 QA 场景

### 方向 5：MCP 协议 + Tool 描述质量（与方向 1/3 交叉）

**推荐 17：MCP Tool Descriptions Are Smelly!**

- arxiv: [2602.14878](https://arxiv.org/html/2602.14878v1)（2026-02）
- 一句话:实证 856 个 tool / 103 个 MCP server,提出 6 维 description 质量评分 + tool description smells 形式化
- 工程价值：⭐⭐⭐⭐
- 落地建议：本项目 Skill 卡描述质量的审核 checklist

**推荐 18：MCPAgentBench - Real-world Task Benchmark for MCP Tool Use**

- arxiv: [2512.24565](https://arxiv.org/abs/2512.24565)（2025-12,2026-01 更新）
- 一句话:动态沙盒 + distractor 工具列表,测试 tool selection 与 discrimination
- 工程价值：⭐⭐⭐⭐

### 方向 6：电商 Agent 落地（贴合本项目母婴跨境场景）

**⭐ 核心推荐 19：EComStage - Stage-wise Benchmarking for LLMs in E-commerce**

- arxiv: [2601.02752](https://arxiv.org/html/2601.02752v1)（2026-01）
- 一句话:不再只看任务最终是否完成,而是按 Perception/Planning/Action 三阶段评估,首次同时覆盖 customer + merchant 视角
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：评估本项目 Agent 在不同业务节点的瓶颈
- 与项目契合度：极高

**⭐ 核心推荐 20：Shopping Companion - Memory-Augmented LLM Agent for E-Commerce**

- arxiv: [2603.14864](https://arxiv.org/abs/2603.14864)（2026-03）
- 一句话:120w 商品规模的长期偏好购物 benchmark,dual-reward RL with tool-wise rewards
- 工程价值：⭐⭐⭐⭐⭐
- 落地建议：母婴用户 1-3 年生命周期偏好建模
- 与项目契合度：极高

---

## 3. 转 Skill 卡的优先级排序

按"工程价值 × 与母婴跨境电商契合度"排序的 Top 8 优先转 Skill:

| 优先级 | 论文 | arxiv | 拟 Skill 名 | 业务场景 |
|--------|------|-------|------------|----------|
| P0 | SkillForge | 2604.08618 | Skill-Auto-Skill-Synthesis | 从客服记录自动生成 Skill |
| P0 | Shopping Companion | 2603.14864 | Skill-Long-Term-Preference-Memory | 母婴用户长周期偏好 |
| P0 | EComStage | 2601.02752 | Skill-Agent-Stage-Evaluation | Agent 三阶段评估 |
| P1 | Agentic Memory (AgeMem) | 2601.01885 | Skill-Agentic-Memory-Management | 客户长期记忆管理 |
| P1 | ACON | 2510.00615 | Skill-Context-Compression | 长对话 / 长 VOC 压缩 |
| P1 | SoK Agentic Skills | 2602.20867 | Skill-Skill-Lifecycle-Design | Skill 卡设计哲学 |
| P2 | MCP Tool Descriptions Smelly | 2602.14878 | Skill-Tool-Description-Audit | Skill/Tool 描述质量审核 |
| P2 | Hermes 4 | 2508.18255 | Skill-Open-Source-Tool-Use-Model | 开源 tool calling 基座选型 |

---

## 4. 待确认事项

1. **目录命名**:`16-LLM-Agent-Engineering/` vs `16-智能体工程/` —— 当前 vault 用中文目录名,建议跟随中文命名
2. **关键词库**:是否需要把这 6 个方向的英文关键词补充进 `paper2skills-vault/07-资源库/关键词库.md`
3. **Hermes 系列的处理**:Hermes 3/4 严格说是 2024-2025 的,但 hermes-agent 仓库 2026 持续更新,是否破例纳入
4. **优先级路径**:先做 P0 三篇(SkillForge / Shopping Companion / EComStage)还是先把"Skill 设计哲学"类(SoK Agentic Skills)做掉作为方法论底座

---

## 4.1 用户决议(2026-05-16)

| 事项 | 用户选择 |
|------|----------|
| 目录命名 | `16-智能体工程/`(跟随中文命名) |
| 关键词库 | 同步更新到 `07-资源库/关键词库.md` |
| Hermes 系列 | 破例纳入(Hermes 4 作为 P2) |
| 优先级路径 | 顺序做:P0 三篇业务落地优先,再做方法论底座 |

执行进度:

- [x] 创建 `paper2skills-vault/16-智能体工程/00-INDEX.md`(规划 15 个 Skill)
- [x] 更新 `paper2skills-vault/07-资源库/关键词库.md`(新增 6 个智能体工程子方向)
- [x] 更新 `paper2skills-vault/README.md`(补完所有 16 个领域)
- [x] 更新 `CLAUDE.md`(Project Structure + Domain Mapping 表)
- [ ] paper-workflow:P0-1 SkillForge → Skill-Auto-Skill-Synthesis
- [ ] paper-workflow:P0-2 Shopping Companion → Skill-Long-Term-Preference-Memory
- [ ] paper-workflow:P0-3 EComStage → Skill-Agent-Stage-Evaluation
- [ ] paper-workflow:P1-1 SoK Agentic Skills → Skill-Skill-Lifecycle-Design

---

## 5. 下一步建议

1. 用户对清单选择性 PICK(每个方向选 1-2 篇先做)
2. 按本项目 paper-workflow 走完整流程(选题 → 萃取 → 审核 → 同步)
3. 把通过审核的 Skill 卡落入 `paper2skills-vault/16-LLM-Agent-Engineering/`
4. 关键词库同步更新

---

## 附录:数据源

- arxiv.org（2026-01 至 2026-05 检索）
- VoltAgent/awesome-ai-agent-papers GitHub 仓库(每周从 arxiv 自动更新)
- Anthropic compaction API 文档(compact-2026-01-12)
