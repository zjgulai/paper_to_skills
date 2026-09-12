---
title: MemGPT — 长期记忆与虚拟上下文管理
doc_type: knowledge
module: 10-MAS
topic: agent-memory-learning
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
paper_id: 2310.08560
paper: "MemGPT: Towards LLMs as Operating Systems"
evidence_basis: paper-verbatim
---

# Skill: MemGPT — 长期记忆与虚拟上下文管理

---

## ① 算法原理

### 核心思想

**MemGPT** 将操作系统的虚拟内存管理思想引入 LLM Agent 的记忆系统。核心洞察：**LLM 的上下文窗口就像物理 RAM——容量有限且昂贵，而 Agent 需要处理的任务往往远超这个容量。解决方案是构建一个分层记忆体系，让 LLM 主动管理自己的记忆**。

MemGPT 的三层记忆架构（类比 OS 内存层次）：

| 层级 | OS 类比 | 功能 | 容量 | 速度 |
|------|---------|------|------|------|
| **Main Context** | 物理 RAM | 当前对话/任务状态/活跃记忆 | 有限（LLM 上下文窗口） | 最快 |
| **Recall Storage** | 磁盘缓存 | 近期对话历史/最近访问的记忆 | 中等（数千条） | 快 |
| **Archival Memory** | 磁盘/长期存储 | 所有历史对话/学习到的知识/用户画像 | 无限（向量数据库） | 慢（需检索） |

### 虚拟上下文管理

MemGPT 的核心创新是让 **LLM 自己控制记忆操作**，通过函数调用管理三层记忆：

1. **`core_memory_replace`** / **`core_memory_append`** — 编辑主内存（Main Context）
2. **`archival_memory_search`** — 从档案记忆检索（向量搜索）
3. **`archival_memory_insert`** — 写入档案记忆
4. **`recall_memory_search`** — 从回忆存储检索

**中断式控制流**：
- 当上下文窗口接近满载（~70% 预警，100% 强制），系统自动触发"页置换"
- 将不活跃的上下文数据换出到 Recall Storage 或 Archival Memory
- 生成递归摘要防止信息丢失
- LLM 通过函数调主动请求加载所需记忆

### 与 RAG 的区别

| 维度 | 传统 RAG | MemGPT |
|------|---------|--------|
| 检索触发 | 外部系统决定何时检索 | LLM 自己决定何时检索 |
| 记忆更新 | 被动（预构建索引） | 主动（LLM 决定存什么） |
| 上下文管理 | 单次检索后固定 | 动态换入换出 |
| 长期记忆 | 静态知识库 | 动态学习到的经验 |

### 关键假设

1. **LLM 能管理自己的记忆**：模型知道什么信息重要、何时需要检索
2. **分层存储有效**：不同活跃度的信息适合不同层级
3. **函数调用可靠**：LLM 能正确调用记忆管理函数
4. **检索质量可接受**：向量检索能召回相关记忆

---

## ② 母婴出海应用案例

### 场景一：长期用户对话 Agent

**业务问题**：

母婴产品的用户咨询是长期关系（从孕期到孩子 3 岁），Agent 需要记住用户的历史偏好、购买记录、孩子的成长阶段、之前的咨询问题。传统 LLM 的上下文窗口无法承载这么长的历史。

**数据要求**：

- 用户历史对话记录
- 购买记录和产品反馈
- 用户画像（偏好、阶段、关注点）
- 产品知识库

**预期产出**：

```
用户: "我上次咨询的吸奶器，现在宝宝 3 个月了，需要换吗？"

MemGPT 记忆检索过程:
  1. Main Context: 当前对话 + 用户基本信息
     → 发现缺少历史购买记录

  2. LLM 调用: archival_memory_search("用户 上次咨询 吸奶器 购买")
     → 检索到: "2025-12: 用户购买 Spectra S1，当时宝宝即将出生"
     → 检索到: "2026-01: 用户反馈吸力够用，但体积大不便携"

  3. LLM 调用: core_memory_append("用户: 有 Spectra S1，反馈体积大不便携")

  4. LLM 生成回复:
     "您 3 个月前购买的 Spectra S1 应该还能用。
      但考虑到您之前提到体积大不便携，
      如果宝宝开始外出频繁，可以考虑便携版 Spectra 9 Plus（仅 0.3kg）。
      需要我对比两款的具体差异吗？"

记忆更新:
  archival_memory_insert: "2026-03: 用户宝宝 3 个月，咨询是否需要换吸奶器"
```

**业务价值**：
- 个性化服务体验（用户感觉 Agent "记得"自己）
- 跨会话连续性，无需重复描述背景
- 长期用户关系维护，提升复购率

---

### 场景二：VOC 知识累积与复用

**业务问题**：

VOC 分析 Agent 在持续处理评论的过程中会积累大量洞察（如"静音是吸奶器的关键购买因素"、"价格敏感度在 Q4 上升"）。这些洞察需要在后续分析中被主动调用，而不是每次都重新学习。

**数据要求**：

- 历史分析结果和洞察
- 时间序列趋势数据
- 品类知识库

**预期产出**：

```
新任务: "分析本周 Spectra S1 的评论"

MemGPT 自动检索相关知识:
  1. archival_memory_search("Spectra S1 历史洞察")
     → "2025-10: 静音是 Spectra S1 的核心卖点，提及率 34%"
     → "2025-11: 价格负面情感从 8% 上升至 15%"
     → "2025-12: 竞品 Medela 推出静音款，构成威胁"

  2. core_memory_append 关键背景:
     → "Spectra S1 核心卖点: 静音"
     → "价格敏感度上升中"
     → "竞品 Medela 静音款威胁"

  3. 分析时自动关注:
     - 静音提及率是否变化？
     - 价格负面是否继续上升？
     - Medela 静音款的影响是否显现？

  4. 新洞察写入档案:
     "2026-03: 静音提及率降至 28%（-6pp），可能受 Medela 竞品影响"
```

**业务价值**：
- 分析洞察跨任务复用，避免重复劳动
- 趋势变化自动检测（与历史基线对比）
- 知识库持续积累，Agent 越用越"懂"业务

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/agent_memory_learning/agent_memory.py`

核心组件：
- `MainContext`: 主内存（当前活跃上下文）
- `RecallStorage`: 回忆存储（近期历史）
- `ArchivalMemory`: 档案记忆（长期向量存储）
- `MemGPTAgent`: 带记忆管理的 Agent
  - `core_memory_replace` / `core_memory_append`
  - `archival_memory_search` / `archival_memory_insert`
  - `recall_memory_search`
- `ContextManager`: 上下文压力管理（换入换出、摘要生成）

运行方式：
```bash
cd paper2skills-code/mas/agent_memory_learning
python agent_memory.py
```

生产环境建议：
1. 使用向量数据库（Pinecone/Milvus/Weaviate）作为 Archival Memory
2. 使用 Redis 作为 Recall Storage
3. 实现记忆去重和合并（避免重复存储相似信息）
4. 定期归档和压缩（旧记忆生成摘要，删除细节）
5. 与 Reflexion 集成：反思结果自动写入 Archival Memory
6. 考虑 Letta（原 MemGPT 商业版）用于生产部署

---

## ④ 技能关联

### 前置技能
- **向量检索**：理解 embedding、相似度搜索、向量数据库
- **LLM Function Calling**：模型调用外部函数的能力

### 延伸技能
- **Letta**：MemGPT 的商业化演进版本
- **Mem0**：轻量级 Agent 记忆层
- **Graphiti / Zep**：基于图结构的长期记忆

### 可组合技能
- **Reflexion**：反思经验存入 Archival Memory，形成长期学习
- **Self-Refine**：改进过程中的中间状态存入 Recall Storage
- **MAS Orchestrator**：多 Agent 共享 Archival Memory 作为知识库
- **CAMEL**：角色对之间的对话历史由 MemGPT 管理

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 长期用户对话 | 个性化体验提升，复购率 +15-20% | 开发 2-3 周 | 15-25x |
| VOC 知识累积 | 分析效率提升 30%，洞察复用 | 开发 2 周 | 12-18x |
| 跨会话一致性 | 用户满意度提升，客服成本降低 | 开发 1-2 周 | 10-15x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：中，需要历史数据初始化记忆库
- 技术门槛：中高，需要理解虚拟内存管理和向量检索
- 工程复杂度：中高，三层存储的协调是核心挑战
- 维护成本：中，记忆库需要定期清理和压缩

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **根本性问题**：上下文窗口限制是 LLM Agent 的核心瓶颈
- **OS 级创新**：将操作系统经典思想应用于 AI，概念优雅
- **已验证效果**：文档分析和多轮对话中显著超越基线
- **商业化成熟**：已演进为 Letta 平台，有生产级支持

---

## 参考论文

1. **MemGPT: Towards LLMs as Operating Systems** (2023)
   - Packer, C. et al. (UC Berkeley)
   - 核心贡献：虚拟上下文管理、三层记忆架构、LLM 主动记忆控制
   - arXiv：2310.08560
   - 代码/平台：https://memgpt.ai / Letta

---

## 三层记忆架构示意

```
用户查询
    ↓
┌──────────────────────────────────────────────┐
│ Main Context (RAM)                           │
│ 容量: 8K-128K tokens                         │
│ 内容: 当前对话 + 活跃记忆 + 任务状态          │
│                                              │
│ [用户: "我上次买的吸奶器..."]                 │
│ [Agent 状态: 需要检索历史购买记录]            │
└──────────────────────────────────────────────┘
    ↓ 未命中
┌──────────────────────────────────────────────┐
│ Recall Storage (磁盘缓存)                     │
│ 容量: 数千条近期记录                          │
│ 内容: 近期对话历史、最近访问的记忆             │
│                                              │
│ [2026-01: 用户反馈 Spectra S1 体积大]         │
│ [2025-12: 用户购买 Spectra S1]                │
└──────────────────────────────────────────────┘
    ↓ 未命中
┌──────────────────────────────────────────────┐
│ Archival Memory (磁盘/向量库)                  │
│ 容量: 无限                                     │
│ 内容: 所有历史对话、学习到的知识、用户画像      │
│                                              │
│ 向量搜索: "Spectra S1 用户 购买 历史"         │
│ 结果: [2025-12 购买记录, 2026-01 反馈记录, ...]│
└──────────────────────────────────────────────┘
    ↓
检索结果换入 Main Context
    ↓
生成回复
```

---

## ⑥ 原文引用

> **底本**：本卡 frontmatter 的 `paper_id` 即 arXiv 编号；其全文存档为 `paper2skills-vault/papers/` 下本论文目录的 `fulltext.md`（LaTeXML HTML 转 Markdown，章节号完整）。
> 下列引文均为底本中的**连续子串**，未改标点、未改词、未把两句话缝成一句（由 `quote_check.py` 逐字核验）。
> 本段只覆盖 ① 与 ⑤ 中**能回溯到论文原文**的断言；② 的场景数字与 ⑤ 的 ROI 表均为作者举例/自估。
> ⚠️ ① 的上下文压力阈值（预警/强制两档）与 `core_memory_replace` 等函数名在本底本中零命中，详见本卡维护记录。

### A. 核心洞察与虚拟上下文管理（对应 ①核心思想）

> 原文:"Large language models (LLMs) have revolutionized AI, but are constrained by limited context windows, hindering their utility in tasks like extended conversations and document analysis. To enable using context beyond limited context windows, we propose virtual context management, a technique drawing inspiration from hierarchical memory systems in traditional operating systems that provide the appearance of large memory resources through data movement between fast and slow memory."
> 出处：2310.08560 Abstract

> 原文:"Using this technique, we introduce MemGPT (Memory-GPT), a system that intelligently manages different memory tiers in order to effectively provide extended context within the LLM’s limited context window, and utilizes interrupts to manage control flow between itself and the user."
> 出处：2310.08560 Abstract

> 原文:"In this paper, we introduced MemGPT, a novel LLM system inspired by operating systems to manage the limited context windows of large language models. By designing a memory hierarchy and control flow analogous to traditional OSes, MemGPT provides the illusion of larger context resources for LLMs."
> 出处：2310.08560 §5 Concluding remarks and future directions

### B. 分层记忆体系（对应 ①「三层记忆架构」表）

> 原文:"In MemGPT we refer to the LLM inputs (that are bound by the maximum number of input tokens) as the system’s main context."
> 出处：2310.08560 §2.1 Main context

> 原文:"In our experiments on multi-session chat and document analysis, we further divide main context into three components: system instructions, which hold the base LLM instructions (e.g., information describing MemGPT functions and control flow to the LLM), conversational context, which holds a first-in-first-out (FIFO) queue of recent event history (e.g., messages between the agent and user), and working context, which serves as a working memory scratchpad for the agent."
> 出处：2310.08560 §2.1 Main context

> 原文:"External context refers to out-of-context storage that lies outside the context window of the LLM processor, analogous to disk memory (i.e. disk storage) in OSes. Information in external context is not immediately visible to the LLM processor, however, it can be brought into main context through appropriate function calls."
> 出处：2310.08560 §2.2 External context

> 原文:"We make a distinction between two types of external context: recall storage, which stores the entire history of events processed by the LLM processor (in essense the full uncompressed queue from active memory), and archival storage, which serves as a general read-write datastore that the agent can utilize as overflow for the in-context read-write core memory."
> 出处：2310.08560 §2.2 External context

> 原文:"In the context of conversational agents, archival storage allows MemGPT to store facts, experiences, preferences, etc. about the agent or user beyond the strict token limit of main context, and search over recall storage allows the MemGPT to find past interactions related to a particular query or within a specific time period."
> 出处：2310.08560 §2.2 External context

### C. 与 RAG 的区别：记忆操作由 LLM 自主发起（对应 ①「与 RAG 的区别」表第 1、2 行）

> 原文:"MemGPT orchestrates data movement between main context and external context via function calls that are generated by the LLM processor. Memory edits and retrieval are entirely self-directed: MemGPT autonomously updates and searches through its own memory based on the current context."
> 出处：2310.08560 §2.3 Self-directed editing and retrieval

> 原文:"We implement self-directed editing and retrieval by providing explicit instructions within the preprompt that guide the system on how to interact with its memory systems."
> 出处：2310.08560 §2.3 Self-directed editing and retrieval

### D. 中断式控制流（对应 ①「中断式控制流」；论文只写了 warns on token limitations，未给出具体预警阈值）

> 原文:"Awareness of context limits is a key aspect in making the self-editing mechanism work effectively, to this end MemGPT prompts the processor with warnings regarding token limitations to guide its memory management decisions"
> 出处：2310.08560 §2.3 Self-directed editing and retrieval

> 原文:"Our memory retrieval mechanisms are designed to be cognizant of these token constraints and implement pagination to prevent retrieval calls from overflowing the context window."
> 出处：2310.08560 §2.3 Self-directed editing and retrieval

> 原文:"In MemGPT, events trigger LLM inference: events are generalized inputs to MemGPT and can consist of user messages (in chat applications), system messages (e.g. main context capacity warnings), user interactions (e.g. an alert that a user just logged in, or an alert that they finished uploading a document), and timed events that are run on a regular schedule (allowing MemGPT to run ‘unprompted’ without user intervention)."
> 出处：2310.08560 §2.4 Control flow and function chaining

> 原文:"Function chaining allows MemGPT to execute multiple function calls sequentially before returning control to the user."
> 出处：2310.08560 §2.4 Control flow and function chaining

### E. 论文自报效果（直接支撑 ⑤「已验证效果：文档分析和多轮对话中显著超越基线」）

> 原文:"We evaluate our OS-inspired design in two domains where the limited context windows of modern LLMs severely handicaps their performance: document analysis, where MemGPT is able to analyze large documents that far exceed the underlying LLM’s context window, and multi-session chat, where MemGPT can create conversational agents that remember, reflect, and evolve dynamically through long-term interactions with their users."
> 出处：2310.08560 Abstract

> 原文:"MemGPT significantly outperforms both GPT-4 and GPT-3.5 in both LLM judge accuracy and ROUGE-L scores: instead of relying on recursive summaries to extend context, MemGPT is able to query past conversation history in its Recall Memory to answer the DMR questions."
> 出处：2310.08560 §3.1.2 Deep memory retrieval task (consistency)

> 原文:"| MemGPT | dialogue1:5 (Recall) | 82.4% | 0.173 | 0.106 | 0.743 |"
> 出处：2310.08560 §3.1.2（Table 2 Deep memory retrieval）

> 原文:"| gpt-4 | summary1:5 | 63.0% | 0.159 | 0.101 | 0.607 |"
> 出处：2310.08560 §3.1.2（Table 2 Deep memory retrieval）

> 原文:"MemGPT utilizes memory to increase engagement: As seen in Table and Figure , MemGPT is able to craft engaging openers that perform similarly to and occasionally exceed the hand-written human openers."
> 出处：2310.08560 §3.1.3 Conversation opener task (engagement)

> 原文:"While GPT-3.5 and GPT-4 have good performance on the original KV tasks, both struggle in the nested KV task. GPT-3.5 is unable to complete the nested variant of the task and has an immediate dropoff in performance, hitting 0 percent accuracy at 1 nesting level (we observe that its primary failure mode is to simply returns the original value). GPT-4 is better than GPT-3.5, but also suffers from a similar dropoff, and hits 0 percent accuracy by 4 nesting levels."
> 出处：2310.08560 §3.2.2 Nested key-value retrieval (KV)

> 原文:"MemGPT on the other hand is unaffected with the number of nesting levels and is able to perform the nested lookup by accessing the key-value pairs stored in main memory repeatedly via function queries. MemGPT performance on the nested KV task demonstrates its ability to combine multiple queries to perform multi-hop lookups."
> 出处：2310.08560 §3.2.2 Nested key-value retrieval (KV)

### F. 边界与权衡（论文自承局限；卡片正文未声明，此处仅备查）

> 原文:"There is also a tradeoff in retrieved document capacity created by MemGPT more complex operation: assuming MemGPT has the same token budget as a fixed-context baseline (using the same LLM), a non-trivial portion of MemGPT’s token budget will be consumed by system intstructions required for MemGPT’s OS components (e.g. function call schemas for memory management), meaning that the total number of documents that can be held in-context at any given time is lower for MemGPT than the baselines."
> 出处：2310.08560 §3.2.1 Multi-document question-answering (Doc-QA)

> 原文:"This tradeoff is observed in Figure : MemGPT has a lower average accuracy than GPT-4 (though higher than GPT-3.5), but can trivially scale to larger numbers of documents."
> 出处：2310.08560 §3.2.1 Multi-document question-answering (Doc-QA)
