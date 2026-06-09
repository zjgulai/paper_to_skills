---
title: CAMEL — 角色扮演式自主协作多 Agent 框架
doc_type: knowledge
module: 10-MAS
topic: camel-role-playing-agents
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill: CAMEL — 角色扮演式自主协作多 Agent 框架

---

## ① 算法原理

### 核心思想

**CAMEL** (Communicative Agents for "Mind" Exploration of Large Language Model Society) 提出了一种基于**角色扮演（Role-Playing）**的多 Agent 自主协作范式。核心洞察：**当两个互补角色的 Agent（指令发出者 vs 执行者）在结构化协议约束下对话时，可以自主完成复杂任务，无需人工逐步干预**。

CAMEL 的三个核心机制：

1. **Role-Playing 角色分离**：
   - **AI User**：负责提出指令、定义目标、评判结果
   - **AI Assistant**：负责理解指令、执行任务、返回结果
   - 严格的角色边界防止"角色翻转"（双方互相推诿或都等对方行动）

2. **Inception Prompting 递归提示**：
   - 将任务描述、角色定义、通信协议、终止条件**嵌入 Agent 的系统提示中**
   - 每个 Agent 的系统提示都包含"对方是谁、自己要做什么、如何回应、何时停止"
   - Agent 在对话中**相互提示**，形成自约束的闭环

3. **Task Specifier 任务细化**：
   - 人类指令往往模糊（如"分析一下我们的竞品"）
   - Task Specifier Agent 先将模糊指令转化为具体、可执行的任务描述
   - 再分发给 Role-Playing Agent 对执行

### 数学直觉

**Inception Prompting 的形式化**：

设任务为 $T$，AI User 角色为 $R_u$，AI Assistant 角色为 $R_a$。Inception Prompt 构造：

$$P_u = \text{System}(R_u, R_a, T, \text{protocol})$$
$$P_a = \text{System}(R_a, R_u, T, \text{protocol})$$

对话轮次 $t$：
$$m_t^u = \text{LLM}(P_u, \{m_{<t}\}) \quad \text{(User 发出指令)}$$
$$m_t^a = \text{LLM}(P_a, \{m_{<t}, m_t^u\}) \quad \text{(Assistant 响应)}$$

终止条件由协议定义（如"当 Assistant 返回 `<CAMEL_TASK_DONE>` 时停止"）。

**角色稳定性约束**：

Inception Prompt 中显式约束每个 Agent 的输出格式，防止角色混淆：

```
AI User 永远不会说 "我来帮你" 或 "我来执行"
AI Assistant 永远不会反问 "你觉得呢" 或 "请给我更多指令"
```

### 关键假设

1. **角色互补性有效**：指令-执行分离能覆盖大多数任务类型
2. **LLM 遵循角色约束**：模型能稳定保持系统提示中定义的角色
3. **任务可细化**：模糊任务可以通过一次 LLM 调用转化为具体任务
4. **对话能收敛**：双 Agent 对话不会无限循环或偏离主题

---

## ② 母婴出海应用案例

### 场景一：VOC 评论分析的角色扮演协作

**业务问题**：

分析母婴产品评论需要"业务视角"（关注什么属性重要）和"数据视角"（从文本中抽取具体信息）的结合。单 Agent 难以同时保持两种视角，容易遗漏关键洞察。

**数据要求**：

- 用户评论文本（Amazon/Trustpilot/Zendesk）
- 产品品类信息
- 分析目标（如"找出 Spectra S1 吸奶器的 Top 3 负面驱动因素"）

**预期产出**：

```
Task Specifier 细化:
  输入: "分析一下 Spectra S1 吸奶器的用户反馈"
  输出: "从 Amazon 和 Trustpilot 评论中，提取 Spectra S1 吸奶器的
         (1) 高频正面属性 Top 5, (2) 高频负面属性 Top 5,
         (3) 与竞品 Medela 的差异化评价, (4) 用户推荐意愿的驱动因素"

Role-Playing 对话:

[AI User = 产品经理]
  "请分析 Spectra S1 的评论，先给出高频正面属性 Top 5"

[AI Assistant = 数据分析师]
  "根据 1,247 条评论分析：
   1. 静音效果好 (提及率 34.2%, 情感正向 91%)
   2. 双边设计省时 (提及率 28.7%, 情感正向 88%)
   3. 夜间模式便利 (提及率 22.1%, 情感正向 85%)
   4. 吸力可调范围广 (提及率 19.5%, 情感正向 82%)
   5. 配件易清洁 (提及率 17.3%, 情感正向 79%)"

[AI User = 产品经理]
  "负面属性 Top 5 是什么？与 Medela 的差异呢？"

[AI Assistant = 数据分析师]
  "负面 Top 5: ...
   与 Medela 差异: Spectra 在静音上胜 (+23pp)，Medela 在便携性上胜 (+15pp)"

<CAMEL_TASK_DONE>
```

**业务价值**：
- 业务视角和数据视角的互补，减少分析盲区
- 无需人工逐步引导，Agent 自主推进分析深度
- 一次对话完成多维度分析，效率提升 3-5x

---

### 场景二：跨品类竞品对标报告自动生成

**业务问题**：

母婴出海需要持续监控竞品动态。传统方式是人工收集数据 → 写报告，周期长（1-2 周）。需要自动化生成结构化的竞品对标报告。

**数据要求**：

- 竞品产品评论数据
- 产品规格参数
- 价格数据
- 市场份额估算

**预期产出**：

```
Task Specifier 细化:
  "生成 Spectra S1 vs Medela Pump In Style vs Elvie Pump 的对标报告，
   包含: (1) 产品规格对比表, (2) 用户评价情感对比, (3) 价格竞争力分析,
   (4) 各产品 SWOT, (5) 市场定位建议"

Agent 对协作:
  [AI User = 市场战略经理]  — 定义分析框架、追问洞察
  [AI Assistant = 竞品分析师] — 收集数据、填充分析、生成报告

输出: 结构化 Markdown 报告，可直接用于管理层汇报
```

**业务价值**：
- 竞品报告生成从 1-2 周缩短到 10-30 分钟
- 分析框架标准化，不同分析师产出一致性高
- 可配置定期自动生成（周报/月报）

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/camel_role_playing/camel_agent.py`

核心组件：
- `RolePlayingAgent`: 带角色的 Agent（AI User / AI Assistant）
- `InceptionPromptBuilder`: 构建 Inception Prompt
- `TaskSpecifier`: 模糊任务 → 具体任务描述
- `CAMELConversation`: 编排角色扮演对话循环

运行方式：
```bash
cd paper2skills-code/mas/camel_role_playing
python camel_agent.py
```

生产环境建议：
1. 使用真实 LLM API 替代 mock 生成器
2. 增加对话轮次上限和超时机制
3. 实现对话历史持久化（用于审计和复盘）
4. 结合 Self-Refinement 机制，让 AI User 对 Assistant 输出进行质量评判
5. 支持多对 Agent 并行协作（如多品类同时分析）

---

## ④ 技能关联

### 前置技能
- **LLM Prompt Engineering**：理解系统提示、角色提示、 Few-shot 提示
- **AutoGen**：理解多 Agent 对话框架的基础概念

### 延伸技能
- **DyLAN**：动态路由的多 Agent 网络（CAMEL 的扩展方向）
- **Multi-Agent Debate**：多 Agent 辩论共识机制
- **Society Simulation**：大规模 Agent 社会模拟

### 可组合技能
- **AutoGen**：CAMEL 的角色扮演可以作为 AutoGen GroupChat 的一种对话模式
- **MetaGPT**：CAMEL 的 Task Specifier 可以与 MetaGPT 的 SOP 流程结合
- **Self-Refine**：AI User 可以作为反馈者，对 Assistant 输出进行批评和改进
- **InstructUIE**：Assistant 执行的数据分析可以调用 InstructUIE 进行结构化抽取

---


- **可组合**：[[Skill-MAS-Orchestrator]] / [[Skill-ReAct-Reasoning-Acting]]

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| VOC 评论自动分析 | 分析效率提升 3-5x，人工耗时减少 70% | 开发 1-2 周 | 12-18x |
| 竞品报告自动生成 | 报告周期从 1-2 周缩短到 10-30 分钟 | 开发 2-3 周 | 15-25x |
| 多维度洞察挖掘 | 减少分析盲区，发现单视角遗漏的关键洞察 | 开发 1 周 | 10-15x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：低，依赖 LLM 能力，无需额外训练数据
- 技术门槛：中，需要理解角色设计和 Inception Prompting
- 工程复杂度：中低，核心是 Prompt 工程和对话状态管理
- 维护成本：低，Prompt 调整即可适应新场景

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- **概念优雅**：Role-Playing 是直觉上最自然的协作方式
- **无需训练**：纯 Prompt 工程，零模型训练成本
- **与 AutoGen 互补**：AutoGen 提供基础设施，CAMEL 提供角色协作模式
- **可落地性强**：1-2 周可完成 MVP

---

## 参考论文

1. **CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society** (NeurIPS 2023)
   - Li, G., Hammoud, H., Itani, H., Khizbullin, D., & Ghanem, B. (KAUST)
   - 核心贡献：Role-Playing + Inception Prompting + Task Specifier 三机制
   - arXiv：2303.17760
   - 代码：https://github.com/camel-ai/camel

---

## 与 AutoGen / MetaGPT 的对比

| 维度 | CAMEL | AutoGen | MetaGPT |
|------|-------|---------|---------|
| 协作模式 | 固定角色对（User-Assistant） | 灵活对话拓扑（任意数量 Agent） | SOP 驱动的工作流 |
| 角色定义 | AI User + AI Assistant | 自定义角色 | 预定义岗位角色 |
| 任务分解 | Task Specifier 一次性细化 | Agent 自主协商 | 按 SOP 步骤执行 |
| 人工干预 | 零干预（纯自主） | 可选 Human-in-the-Loop | 设计时人工定义 SOP |
| 最佳场景 | 探索性任务、创意生成 | 通用多 Agent 应用 | 标准化流程任务 |

**组合建议**：
- 用 **MetaGPT** 定义标准化分析流程（SOP）
- 流程中的每个步骤用 **CAMEL** 角色对执行
- 用 **AutoGen** 作为底层对话基础设施
