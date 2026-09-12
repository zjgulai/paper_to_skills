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
paper_id: 2303.17760
paper: "CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society"
evidence_basis: paper-verbatim
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

---

## ⑥ 原文引用

> **底本**：本卡 frontmatter 的 `paper_id` 即 arXiv 编号；其全文存档为 `paper2skills-vault/papers/` 下本论文目录的 `fulltext.md`（LaTeXML HTML 转 Markdown，章节号完整）。
> 下列引文均为底本中的**连续子串**，未改标点、未改词、未把两句话缝成一句（由 `quote_check.py` 逐字核验）。
> 本段只覆盖 ① 与 ⑤ 中**能回溯到论文原文**的断言；② 的场景数字与 ⑤ 的 ROI 表均为作者举例/自估，论文中无对应数字。

### A. 核心思想：角色扮演 + Inception Prompting（对应 ①核心思想）

> 原文:"To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing . Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions."
> 出处：2303.17760 Abstract

> 原文:"This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents and provide insight into their “cognitive” processes."
> 出处：2303.17760 Abstract

### B. 三机制之一：Role-Playing 角色分离（对应 ①机制 1「AI User / AI Assistant」）

> 原文:"Our proposed framework is a novel role-playing approach for studying multiple communicative agents. Specifically, we concentrate on task-oriented role-playing that involves one AI assistant and one AI user. After the multi-agent system receives a preliminary idea and the role assignment from human users, a task-specifier agent will provide a detailed description to make the idea specific and then the AI assistant and AI user will cooperate on completing the specified task through multi-turn conversations until the AI user determines the task is done."
> 出处：2303.17760 §3.1 Role-playing Framework

> 原文:"The AI user is responsible for giving instructions to the AI assistant and directing the conversation toward task completion. On the other hand, the AI assistant is designed to follow the instructions from the AI user and respond with specific solutions."
> 出处：2303.17760 §3.1 Role-playing Framework

> 原文:"After the role assignment is completed, the AI assistant $\mathcal{A}$ and AI user $\mathcal{U}$ will collaborate in an instruction-following manner to accomplish the task. In the AI assistant-user scenario, the AI user is responsible for providing instructions, and the assistant is expected to respond with a solution that fulfills the instructions."
> 出处：2303.17760 §3.1 Role-playing Framework（Conversation Towards Task-Solving）

> 原文:"After the task specification, The AI assistant role and the AI user role will be assigned to the user agent and the assistant agent correspondingly to complete the specified task. In practice, a system message is passed to each agent declaring roles to each."
> 出处：2303.17760 §3.1 Role-playing Framework（AI Assistant-User Role Assignment）

### C. 三机制之二：Inception Prompting 递归提示（对应 ①机制 2）

> 原文:"Unlike other techniques for conversational language models, our prompt engineering occurs solely at the beginning of role-playing, for task specification and role assignment. Once the conversation phase commences, the AI assistant and AI user prompt each other automatically in a loop until termination."
> 出处：2303.17760 §3.2 Inception Prompting

### D. 终止条件与角色稳定性（对应 ①「终止条件由协议定义」与 ①关键假设 2、4）

> 原文:"End of Task Token: If the user believes that the task has been solved, they are expected to say <CAMEL_TASK_DONE> to signify the completion of the task. Once this message is received, the conversation is terminated to ensure that the data generated accurately reflects the completion of the task."
> 出处：2303.17760 §4.1 Role-Playing for AI Society and Code Scenarios（Termination Conditions）

> 原文:"User No Instruct: If the user does not instruct the assistant for 3 rounds, the conversation is terminated."
> 出处：2303.17760 §4.1（Termination Conditions）

> 原文:"Assistant Instruct: If the assistant provides an instruction to the user, it indicates a role reversal, and the conversation is terminated."
> 出处：2303.17760 §4.1（Termination Conditions）

> 原文:"Maximum Number of Messages: To keep the cost of generated chats in check, we have set a maximum limit of 40 messages. This limit guarantees a long enough conversation between the user and assistant while also ensuring that the data generated is not too costly to produce. The cost grows quadratically with the length of the conversation, making it essential to set a limit."
> 出处：2303.17760 §4.1（Termination Conditions）

### E. 论文实测的四类协作失效（对应 ①关键假设 2「LLM 遵循角色约束」的反例）

> 原文:"Role Flipping: One challenge we encountered was role flipping, where the assistant and user switch roles during the conversation. This issue typically arises when the assistant starts providing instructions or commands instead of following the user’s prompts, which can lead to confusion and a reversal of roles. To avoid role flipping, it is crucial for the assistant not to ask questions, as this can also contribute to the problem."
> 出处：2303.17760 §4.1（Challenges and Observations）

> 原文:"Assistant Repeats Instruction: Another challenge that we observed was the assistant simply repeating the user’s instructions without any role flipping occurring."
> 出处：2303.17760 §4.1（Challenges and Observations）

> 原文:"Flake Replies: We also observed instances where the assistant agent responds with a flake reply, often taking the form of "I will…". These messages do not contribute to the task at hand, as the assistant promises to take action but ultimately fails to follow through."
> 出处：2303.17760 §4.1（Challenges and Observations）

> 原文:"Infinite Loop of Messages: A particularly interesting challenge that we encountered was when the assistant and user engage in an infinite loop of meaningless conversation, such as repeatedly thanking each other or saying goodbye without making any progress in the conversation."
> 出处：2303.17760 §4.1（Challenges and Observations）

### F. 数据规模与产出（对应 ①机制 3 与 ③「可复用资产」）

> 原文:"*Figure 5: Generated Meta Data. The meta data generated by LLMs for AI Society and Code datasets. 50 assistant roles and 50 user role are generated for AI Society. 20 programming languages and 50 domains are generated for Code.*"
> 出处：2303.17760 §4.1（Figure 5）

> 原文:"Our library, which we make publicly available, provides modular functionality, implementations of different agents, well-crafted prompts, and data explorers, thereby simplifying the utilization of the library for future research in various areas such as multi-agent systems, cooperative AI, game theory simulations, social analysis, AI ethics, AI alignment, and beyond."
> 出处：2303.17760 §1 Introduction

### G. 结论与自承风险（对应 ⑤「可落地性」与边界）

> 原文:"Our approach enables communicative agents to collaborate autonomously toward completing tasks while requiring minimal human intervention. Through our analysis, we show that achieving autonomous cooperation is challenging due to issues like hallucination, conversation deviation, role flipping, and termination conditions."
> 出处：2303.17760 §5 Conclusion

> 原文:"We are aware of the potential risks and limitations of this work. For the risks, since existing LLMs are not fully tuned to be harmless, they can be easily exploited by malicious users for harmful purposes."
> 出处：2303.17760 §5 Conclusion（Risk, Limitation and Future Work）
