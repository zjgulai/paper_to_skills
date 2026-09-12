---
title: AgeMem — 统一 LTM+STM 管理的 Agentic Memory
doc_type: knowledge
module: 16-智能体工程
topic: agentic-memory-management
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: AgeMem — 统一 LTM+STM 管理的 Agentic Memory

---

## ① 算法原理

### 核心思想

**AgeMem(Agentic Memory)** 颠覆了传统 LTM/STM 分离架构,把**记忆管理整合到 Agent 的 policy 本身**。现有方法把 LTM 与 STM 当作两个独立模块,要么用 trigger-based 启发式,要么外挂 Memory Manager,导致:

- LTM/STM 分别优化,组合时各种 ad-hoc
- 训练时记忆操作的稀疏/不连续 reward 难处理
- 部署时需要额外 expert LLM,推理成本翻倍

AgeMem 三大创新:

1. **6 个 memory tools 作为 action space**:
   - LTM: `Add` / `Update` / `Delete`
   - STM: `Retrieve` / `Summary` / `Filter`
   - LLM 自主决定何时调用哪个,无需外部 controller

2. **三阶段渐进式 RL**:
   - **Stage 1 LTM Construction**:闲聊场景下,学习把关键信息存入 LTM
   - **Stage 2 STM Control**:重置 context,注入干扰内容,学过滤/总结
   - **Stage 3 Integrated Reasoning**:正式任务,协调 LTM 检索 + STM 管理 + 答案生成

3. **Step-wise GRPO**:把 trajectory 终局 reward 广播到所有中间步骤,解决"memory 操作 reward 稀疏不连续"难题

### 数学直觉

**状态与策略**:

$$
s_t = (C_t, \mathcal{M}_t, \mathcal{T})
$$

其中 $C_t$ 是 STM(active context),$\mathcal{M}_t$ 是 LTM store,$\mathcal{T}$ 是任务规格。Agent 在 hybrid action space 中选择 $a_t$(语言生成 + 6 个 memory tool 调用):

$$
\pi_\theta(a_t | s_t) = P(a_t | s_t; \theta)
$$

**复合 reward**:

$$
R(\tau) = w_{\text{task}} R_{\text{task}} + w_{\text{context}} R_{\text{context}} + w_{\text{memory}} R_{\text{memory}} + P_{\text{penalty}}
$$

- $R_{\text{task}}$: LLM-judge 任务完成度 ∈ [0, 1]
- $R_{\text{context}}$: STM 三因子(compression 效率 + 预防性操作 + 信息保留)
- $R_{\text{memory}}$: LTM 三因子(storage 质量 + 维护 + 语义相关性)
- $P_{\text{penalty}}$: context overflow / 超 turn 限制

**Step-wise GRPO advantage 广播**:

$$
A_T^{(k,q)} = \frac{r_T^{(k,q)} - \mu_{G_q}}{\sigma_{G_q} + \epsilon}, \quad A_t^{(k,q)} = A_T^{(k,q)} \text{ (broadcast)}
$$

终局 advantage 复制到所有 step,使得 Stage 1 的 memory 决策也能从 Stage 3 的任务结果中拿到学习信号。

**最终目标**:

$$
J(\theta) = \mathbb{E}_{(e_t, A_t) \sim \mathcal{E}} [\rho_t A_t - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]]
$$

### 关键实证发现

- **Qwen2.5-7B**:AgeMem 41.96% vs no-memory 28.05% (+49.59% 相对)
- **Qwen3-4B**:AgeMem 54.31% vs no-memory 43.97% (+23.52% 相对)
- RL 训练 +8.5-8.7pp(vs noRL)
- 比最强 baseline (Mem0/A-Mem) +4.82-8.57pp
- token 使用 -3-5%(更精准的 STM 管理)
- Memory Quality:0.533 / 0.605(LLM judge 评估存储记忆与 ground-truth 的相关度)

### 关键假设

1. Memory 操作可以被表达为 discrete tool calls(论文 6 个 tool)
2. trajectory 终局 reward 能传递到中间 memory 决策(GRPO broadcast 假设)
3. 三阶段训练顺序可以让 LTM/STM 能力逐步建立(progressive)
4. LLM judge 与人类一致性足够高

### 关键约束

- **训练成本**:RL fine-tune 不便宜,论文用 Trinity 框架,但仍需 GPU 资源
- **冷启动**:全新 deployment 时 $\mathcal{M}_t$ 为空,前几次交互无法享受 LTM 收益
- **GRPO 假设**:K 个独立 rollout 同任务,组内归一化,需要相对稳定的环境

---

## ①b 反例与适用边界（负结果证据）

> **口径声明**：本卡讲「记忆管理」,但记忆这件事有**两半** —— **抽取/结构化**那一半与
> **遗忘/剪枝**那一半。`2608.28978`（*Selective Forgetting*，arXiv preprint，LongMemEval，500 题）
> 这篇**负结果**论文给出的最有价值的一条判据就是：**这两半不能一起信**。以下按两半分开写。
> ⚠️ 论文把自己的结论严格限定在「基于抽取的流水线」范围内,**不是**「图结构记忆普遍无效」——
> 见第 4 条,这是本节最重要的一处口径。

**1.「图结构抽取」那一半：负结果,不要当 Agent 记忆的主存储**

在**匹配 5 检索根候选生成预算**下,把每轮对话抽成 typed node + 属性边的图记忆
**没有跑赢**平铺向量基线：token F1 **0.417 vs 0.468**，
配对 bootstrap（500 问）**Δ = −0.050，95% CI [−0.085, −0.016]**（区间不含 0）；
judge 准确率 **0.454 vs 0.536**（⑥ Q1/Q15）。

差距最大的是**需要回忆某条历史 assistant 发言**的问题：judge **0.911 → 0.607**（⑥ Q2/Q12）。
机制解释：平铺基线能**逐字**取回原始 assistant 那一轮，
而图抽取把它**分解成实体与关系**,丢掉了这类问题依赖的**表层形式（surface form）**（⑥ Q2/Q12）。

→ **对 AgeMem 的直接含义**：`Summary` / `Filter` 若把原话压成「要素」,就踩同一个坑。
**LTM 必须保留可回溯的原文通道**（例如 `Add` 时同时落一条原文指针 / hash）,
否则「客户上次说的**具体那句话**」这类检索会系统性劣化。

**2.「遗忘 / 剪枝」那一半：正结果,可以复用（本卡最该拿走的一条）**

论文的遗忘模块反而是**成功**的：对一张 **27,021 节点**的持久图应用**一次**，
移除 **9.8% 节点 / 9.5% 存储字节**（2,653 节点、2,560 边，440.6 MB → 398.6 MB）；
token F1 **基本不变（+0.001，95% CI [−0.015, +0.016]）**，
judge 正确率下降 **1.6 个点**，95% 区间把损失上界压在 **3.8 个点**（[−0.038, +0.006]）（⑥ Q4/Q6）。
打分口径：recency / access frequency / degree centrality / turn age 加权，
阈值以下节点**连同关联边**一并删除，每 **400 轮**触发一次（⑥ Q13）。

→ **可复用的是「剪枝策略」,不是「抽取管线」。** 对一个已经在跑的 Agent LTM,
先上「重要度打分 + 尾部剪枝」拿存储收益,比先重构成知识图谱风险低得多；
这条与本卡 ③ 段的 `Delete` 与 STM 的 `Filter` 是同一件事的两种实现。

**3. 但这一半也有两个没关掉的口子（论文自承,本卡照抄不加工）**

- **没有匹配压缩对照**：论文把「随机剪掉同样比例的节点」列为下一步**第一优先级**实验（⑥ Q11）——
  「按重要度剪枝有效」**尚未**与「随便剪都一样」区分开。
  本卡 ③ 段的三因子打分在本卡内**没有独立证据**证明它优于随机剪枝,落地时必须自建这个对照。
- **judge 确实掉了 1.6 个点**：论文承认被剪掉的信息仍可能对正确答案有贡献,
  这是明确的**效率 ↔ 信息保留**权衡（⑥ Q7）,不是零成本。
- 实验规模有限：只跑了 **4 次完整 run**,且**没有对保留参数做 sweep**（⑥ Q16/Q17）——
  这条结论的置信度应按此打折。

**4. 论文自设的范围限定（本卡强制口径 —— 最重要的一条）**

- ✅ 可以说：「这个**基于抽取的**图记忆流水线在 LongMemEval 上没跑赢平铺向量基线。」
- ❌ 不能说：「图结构记忆普遍无效 / 不要在 Agent 记忆上投图。」
  论文原话把结论限定为 **this extraction-based pipeline** / **at this model scale**,
  明确排除 **graph-structured memory in general**（⑥ Q3/Q10）；
  抽取器是**单个小模型（GPT-4o-mini）**、只评了**一个 benchmark**（⑥ Q14）。

**5. 与本卡 `Update` 工具直接相关的一类失败模式**

知识更新类问题 F1 **0.456 vs 0.511**：论文检查失败样本发现,在**没有显式置信度**时,
冲突消解策略会**保留旧的属性值而不替换为新值**（⑥ Q8）。
→ 对本卡的 `Update` 是硬要求：**必须显式定义冲突消解规则**（论文建议对事实 / 数值型属性用
last-write-wins）,否则「宝宝月龄 / 偏好品牌」这类随时间变化的字段会**静默过期**,
而 Agent 自己不会报错。

**6. 论文未讨论 / 未报告**

母婴出海、跨境电商、多语言客服与平台站内数据 —— **论文未讨论**；
抽取调用的 token 成本与端到端时延 —— **论文未报告**；
除 LongMemEval 之外的第二 benchmark —— **论文未报告**,且已列为下一步工作（⑥ Q11）。

---

## ② 母婴出海应用案例

### 场景一:母婴用户 0-3 岁全生命周期 LTM/STM 协同管理

**业务问题**:

母婴用户从孕期到 3 岁的完整生命周期,跨越 4 年、可能数百次交互。传统系统要么用 RAG(STM 一直膨胀)要么固定 trigger(每月强制总结),都不够精准。需要 Agent 自主决定:

- 哪些信息(过敏原、品牌偏好、宝宝月龄、产品满意度) 该存进 LTM
- 哪些 context 该用 Filter 剔除(广告导流、闲聊)
- 哪些 LTM 该 Update(月龄随时间增长) / Delete(过期偏好)

**数据要求**:

- 用户 4 年内全部对话历史 + 商品互动 + 客服记录
- HotpotQA 风格的"上下文 + 干扰 + 任务"训练对
- Ground-truth memory 标注(用于 RL 训 memory quality reward)

**预期产出**:

```
用户 ID U001 的 LTM 演化:

孕期 (Stage 1 等价):
  Add: pregnancy_due_date=2024-08-15
  Add: previous_allergy=peanut

新生儿期:
  Add: baby_born_2024-08-10  (Update due_date → real)
  Add: formula_brand_prefer=PampersBrand
  Add: diaper_size_history=NB→S→M (各 2 个月)

注:Delete: pregnancy_due_date(已过期)

干扰 (Stage 2 等价):
  Filter:广告页面内容(无 LTM 写入)
  Summary:多轮"客户咨询发货时间"压缩为 1 句

正式任务 (Stage 3 等价):
  User: "宝宝 6 个月了能换 L 码吗?"
  Retrieve: baby_born_2024-08-10, diaper_size_history
  Compute: 6 月龄 + 上次 M 码穿到 8.5kg
  Answer: 建议 L 码,推荐 Pampers L 码(基于 brand_prefer LTM)
```

**业务价值**:

- 用户体验:跨 session 偏好继承,无需用户重复说明
- LTV 提升:精准的全周期推荐,复购率预期 +20-30%
- 客服成本:Filter + Summary 让 context token 降低 3-5%(论文实测)

---

### 场景二:跨境客服多 session 历史智能管理

**业务问题**:

跨境多语言客服,1 个客户可能在 7 天内联系 5+ 次,涉及不同问题(发货、退货、再购买)。每次新 session,客服 Agent 都从头开始读历史会浪费 token,且关键信息(订单号、过敏症状)可能因为 context 长度被 RAG 错过。

**数据要求**:

- 客户多 session 历史(中英 + 平台数据)
- bad case:之前看过但忘记关键信息导致的客户投诉
- RL 训练集:HotpotQA 风格,5000-10000 例

**预期产出**:

```
新 session 开始:
  Agent: Retrieve(query="客户历史关键事件") → 拿 LTM
  Agent 看到: { order_history: [ORD123_退货_过敏],
                allergen: peanut, 
                next_visit_due: 0-9 月龄推荐 }
  
新 session 中:
  User: "再买一次配方奶粉,但要避免过敏"
  Agent: Retrieve(allergen) → "peanut"
  Agent: 推荐非花生类配方
  
session 结束:
  Agent: Add(satisfaction_score=4/5)  
  Agent: Update(last_purchase=2026-05-15)
```

**业务价值**:

- 跨 session 一致性:Agent 不会"忘记"上次客户的关键信息
- token 节省:STM Filter + Summary 让对话 context 维持紧凑
- 客户满意度 +15-25%(从行业经验估算)

---

## ③ 代码模板

代码位置:`paper2skills-code/llm_agent_engineering/agentic_memory/agemem.py`

核心组件:

- `LTMStore`:LTM 持久化(支持 Add / Update / Delete)
- `STMContext`:STM 当前 context(支持 Retrieve / Summary / Filter)
- `MemoryTool` enum:6 个 tool 的统一接口
- `AgentState`:$s_t = (C_t, \mathcal{M}_t, \mathcal{T})$ 状态封装
- `MemoryAgent`:简化版 policy,可触发 6 个 memory tool
- `ThreeStageRollout`:三阶段 trajectory 生成
- `StepwiseGRPO`:简化版 step-wise GRPO advantage 计算
- `CompositeReward`:R_task + R_context + R_memory + penalty

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/agentic_memory
python agemem.py
```

生产环境建议:

1. 真实部署用 Trinity 框架(论文用)做 RL fine-tune,base 模型选 Qwen3-4B/7B
2. 6 个 memory tool 的实现接入向量库(Pinecone / Weaviate / FAISS)
3. Reward 中的 LLM judge 接入 GPT-5 / Claude / Qwen3-Max
4. 三阶段训练数据用 HotpotQA 风格自动生成(参考论文 Stage 1 + Stage 2 + Stage 3 设计)
5. K(每任务 rollout 数) 推荐 8-16,trajectories 长度 50-150 turn

---

## ④ 技能关联

### 前置技能

- **02-A_B实验**:GRPO 组内对比的基础思想
- **16-智能体工程 Skill-Context-Compression**(ACON):STM 压缩前置
- **16-智能体工程 Skill-Skill-Lifecycle-Design**(SoK):memory 在 Skill 中的角色

### 延伸技能

- **16-智能体工程 Skill-Long-Term-Preference-Memory**(Shopping Companion):Dual-Reward 在购物场景的应用
- **16-智能体工程 Skill-Active-Context-Pruning**(Focus):仿生剪枝
- **16-智能体工程 Skill-Memory-as-Action**(MemAct):Memory 操作作 action 的另一变体

### 可组合技能

- **05-推荐系统**:基于 LTM 偏好的个性化推荐
- **07-NLP-VOC 自动打标签**:从对话中萃取 LTM 候选条目
- **08-知识图谱**:LTM 存储可结构化为 KG ⚠️ **与 2608.28978 的负结果冲突,见 ①b 段** —— 该论文实测「把对话抽成 KG」这一半**没有**跑赢平铺向量基线（token F1 0.417 vs 0.468）;若要做,须按 ①b 第 1 条保留原文回溯通道,并按第 4 条限定结论口径。

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 母婴全生命周期 LTM 协同 | 复购率 +20-30%, LTV +25-40% | RL 训练 4-8 周 + 数据标注 | 15-25x |
| 客服多 session 历史管理 | token 节省 5-10%, 客户满意 +15-25% | RL 训练 3-4 周 | 10-15x |
| LTM/STM 联合监控 | bad case 自动检测 -30-50% 人工质检 | 工程 2 周 | 6-10x |

### 实施难度

**评分:⭐⭐⭐⭐☆(4/5 星)**

- 数据要求:高,需要 HotpotQA 风格的标注训练集 5000+
- 技术门槛:高,Step-wise GRPO + 三阶段 progressive RL 需要 RL 经验
- 工程复杂度:高,需要部署 RL 训练管道 + LLM judge 评估
- 维护成本:中,RL 训练后 policy 可以长期复用

### 优先级评分

**评分:⭐⭐⭐⭐☆(4/5 星)**

- **业务价值极高**:与跨境母婴长周期场景高度契合
- **小模型友好**:论文 Qwen2.5-7B / Qwen3-4B 都能用,显著降成本
- **完整方法论**:三阶段 RL + 复合 reward + Step-wise GRPO 是完整 receipe
- **学习曲线陡**:需要 RL 基础,本项目其他 Skill 完成后再做更稳

### 评估依据

1. **实验充分**:5 个长 horizon benchmark + 2 个 LLM backbone + 4 baseline 对比
2. **数据扎实**:Qwen3-4B +23.52% 相对增益,小模型 4B 即可超越 baseline
3. **Memory Quality 量化**:LLM judge 评估存储记忆质量,而非只看任务成功
4. **作者背景**:阿里巴巴 + 武汉大学,产业实战 + 学术严谨

---

---

## ⑥ 原文引用

> **本段引文的论文与范围**：本段全部引文均来自 `2608.28978`
> *Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents*
> （arXiv preprint，全文存档：`papers/16-智能体工程/p2s-2026-0030/fulltext.md`）。
> ⚠️ 本卡的**正面主张来自另一篇论文**（AgeMem，arXiv:2601.01885），那张论文的全文
> 未收录在本仓库存档中，因此**本段不引用它**（无法逐字核验的引文一律不写）。
> 本段引文全部用于 **①b 反例与适用边界**：抽取那一半是负结果，遗忘/剪枝那一半是正结果。


**A. 负结果：抽取那一半没跑赢平铺向量基线**

> 原文:"On LongMemEval, the graph does not outperform a flat vector baseline at a matched candidate-generation budget of five retrieval roots: token F1 is $0.417$ against $0.468$, and a paired bootstrap over 500 questions gives $\Delta=-0.050$ (95% CI $[-0.085,-0.016]$)."
> 出处：2608.28978 §Abstract｜Q1

> 原文:"The gap is widest on questions that require recalling a specific prior assistant turn, where judged correctness falls from $0.911$ to $0.607$, suggesting that decomposing a turn into entities discards the surface form these questions depend on."
> 出处：2608.28978 §Abstract｜Q2

> 原文:"Because our extractor is a single small model evaluated on one benchmark, these results characterise this extraction-based pipeline rather than graph-structured memory in general."
> 出处：2608.28978 §Abstract｜Q3

**B. 正结果：遗忘 / 剪枝那一半是有效的（本卡可复用的部分）**

> 原文:"The forgetting module is more successful. Applied once to a persistent 27,021-node graph, it removes 9.8% of nodes and 9.5% of stored bytes; token F1 is unchanged ($+0.001$, 95% CI $[-0.015,+0.016]$) and judged correctness falls by $1.6$ points, with the 95% interval bounding any loss at $3.8$ points ($[-0.038,+0.006]$)."
> 出处：2608.28978 §Abstract｜Q4

> 原文:"The proposed forgetting module contributes a retention mechanism whose cost we can bound: pruning the low-importance tail of a 27,021-node store removed 9.8% of nodes and 9.5% of bytes, and a paired bootstrap over 500 questions detects no significant change in any of the four metrics (Table 7)."
> 出处：2608.28978 §6 Conclusion｜Q5

> 原文:"Applying the forgetting mechanism removes 2,653 nodes (9.8%) and 2,560 edges (5.5%), reducing the graph size from 440.6 MB to 398.6 MB, a 9.5% reduction."
> 出处：2608.28978 §5 Discussion and Limitations / Table 2｜Q6

> 原文:"The reduction in LLM-judge accuracy, however, indicates that some pruned information can still contribute to correct answers, highlighting a trade-off between memory efficiency and information retention."
> 出处：2608.28978 §5 Discussion and Limitations｜Q7

**C. 失败模式与论文自设的范围限定（照抄不加工）**

> 原文:"Graph RAG also underperforms on knowledge-update questions (F1: 0.456 vs. 0.511). Inspection of failures indicates that the current conflict-resolution policy can retain an earlier attribute value instead of replacing it with a more recent value when no explicit confidence score is available."
> 出处：2608.28978 §5 Discussion and Limitations｜Q8

> 原文:"First, representing conversational memory as a knowledge graph does not uniformly improve retrieval over a flat vector store."
> 出处：2608.28978 §5 Discussion and Limitations｜Q9

> 原文:"Accordingly, our findings should be read as characterising this extraction-based graph memory pipeline at this model scale, not graph-structured memory in general."
> 出处：2608.28978 §A.1 Note to Reviewers on Experimental Scope and AI Use｜Q10

> 原文:"Given additional budget, our order of priority would be: a matched-compression control that prunes the same fraction of nodes at random, to isolate the contribution of the importance function; a second benchmark; and a stronger extraction model."
> 出处：2608.28978 §A.1 Note to Reviewers on Experimental Scope and AI Use｜Q11

**D. 方法口径（打分因子、抽取器、检索口径）**

> 原文:"The flat baseline can retrieve the original assistant turn verbatim, whereas graph extraction decomposes the turn into entities and relationships. In doing so, it may lose information about which item or statement was specifically emphasized."
> 出处：2608.28978 §5 Discussion and Limitations｜Q12

> 原文:"Every 400 turns, the retention stage scores each node by recency, access frequency, centrality, and turn age, pruning nodes whose importance falls below the threshold together with their incident edges."
> 出处：2608.28978 §3.1 Architecture / Figure 1 caption｜Q13

> 原文:"Each conversational turn is processed by a single LLM extraction call (GPT-4o-mini) using a structured system prompt that defines the ontology, output schema, and extraction rules."
> 出处：2608.28978 §3.3 Extraction Pipeline｜Q14

**E. 整体读数与实验规模（用于判断结论强度）**

> 原文:"In Experiment 1, the baseline RAG system outperforms Graph RAG overall, achieving a token F1 of 0.468 compared with 0.417 and an LLM-judge accuracy of 0.536 compared with 0.454."
> 出处：2608.28978 §5 Discussion and Limitations｜Q15

> 原文:"We chose to spend it on four full runs (Experiment 1 treatment and control, Experiment 2 treatment and control) at $n=500$ with temperature $=0$, and to report paired bootstrap intervals over those runs, rather than on a larger number of partially evaluated configurations."
> 出处：2608.28978 §A.1 Note to Reviewers on Experimental Scope and AI Use｜Q16

> 原文:"We did not perform a full sweep over the retention parameters; the consequences of this are discussed at the end of this section."
> 出处：2608.28978 §A.2 Justification for parameter values｜Q17

<details><summary>Q 编号 ↔ 论文位置对照</summary>

| Q | 论文位置 | 行号 |
|---|---|---|
| Q1 | §Abstract | L15 |
| Q2 | §Abstract | L15 |
| Q3 | §Abstract | L15 |
| Q4 | §Abstract | L15 |
| Q5 | §6 Conclusion | L181 |
| Q6 | §5 Discussion and Limitations / Table 2 | L169 |
| Q7 | §5 Discussion and Limitations | L173 |
| Q8 | §5 Discussion and Limitations | L167 |
| Q9 | §5 Discussion and Limitations | L159 |
| Q10 | §A.1 Note to Reviewers on Experimental Scope and AI Use | L267 |
| Q11 | §A.1 Note to Reviewers on Experimental Scope and AI Use | L267 |
| Q12 | §5 Discussion and Limitations | L165 |
| Q13 | §3.1 Architecture / Figure 1 caption | L63 |
| Q14 | §3.3 Extraction Pipeline | L73 |
| Q15 | §5 Discussion and Limitations | L161 |
| Q16 | §A.1 Note to Reviewers on Experimental Scope and AI Use | L265 |
| Q17 | §A.2 Justification for parameter values | L275 |

</details>

---


---

## 参考论文

1. **Agentic Memory (AgeMem): Learning Unified Long-Term and Short-Term Memory Management for LLM Agents** (2026-01)
   - Yu, Y., Yao, L., Xie, Y., Tan, Q., Feng, J., Li, Y., Wu, L. — Alibaba + Wuhan University
   - 核心贡献:LTM + STM 统一 tool-based 管理 + 三阶段渐进 RL + Step-wise GRPO
   - arxiv:[2601.01885](https://arxiv.org/abs/2601.01885)

## 相关基础

- **GRPO** (arxiv:2402.03300):DeepSeekMath 提出的 Group Relative Policy Optimization
- **HotpotQA** (EMNLP 2018):多跳问答 benchmark,论文主训练数据
- **Mem0** (arxiv:2504.19413):scalable extract-update LTM pipeline(对照 baseline)
- **A-Mem**:Zettelkasten 风格 LTM(对照 baseline)
- **LangMem**:LangChain 风格 LTM(对照 baseline)

---

## 与同领域 Skill 的对比

| 维度 | AgeMem | ACON(P1-2) | Shopping Companion(P0-2) |
|------|--------|------------|---------------------------|
| 关注层面 | 全 Memory 管理(LTM + STM) | Context Compression(单向) | 任务执行 + 偏好记忆 |
| 训练方法 | 三阶段 RL + Step-wise GRPO | Gradient-free guideline opt | Dual-Reward RL |
| Tool 数量 | 6 个 memory tool | 0(纯 prompt opt) | 5 个 retrieval/check tool |
| 训练成本 | 高(需 RL fine-tune) | 低(无 fine-tune) | 中(RL fine-tune) |
| 母婴场景 | 全生命周期 | 长对话 / 长报告 | 复购 / 凑单 |

**互补使用**:
- **底层 Memory 能力**用 AgeMem 训
- **Context 压缩快速 win**用 ACON(无需 fine-tune)
- **购物场景任务执行**用 Shopping Companion
