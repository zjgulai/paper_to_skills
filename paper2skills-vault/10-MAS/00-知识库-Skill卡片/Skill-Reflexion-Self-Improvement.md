---
title: Reflexion — 言语强化学习与自我反思
doc_type: knowledge
module: 10-MAS
topic: reflexion-self-improvement
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
venue_tier: preprint
venue_source: EXPLICIT_RETIERS(Skill-Reflexion-Self-Improvement)
paper_id: 2303.11366
paper: "Reflexion: an autonomous agent with dynamic memory and self-reflection"
evidence_basis: paper-verbatim
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-02
l2_domain: 产品与创新
l3_id: DOM-02-019
l3_business: VOC编码
l3_all: VOC编码
l1_l2_l3: 业务运营/产品与创新/VOC编码
---

# Skill: Reflexion — 言语强化学习与自我反思

---

## ① 算法原理

### 核心思想

**Reflexion** 提出了一种**言语强化学习（Verbal Reinforcement Learning）**机制。核心洞察：**传统 RL 需要更新模型权重，成本高且难以解释；而 LLM 可以通过自然语言形式的"自我反思"来改进策略，无需任何权重更新**。

Reflexion 的三组件架构：

1. **Actor**：执行任务，生成输出（如代码、分析、决策）
2. **Evaluator**：评估 Actor 的输出，给出成功/失败信号和分数
3. **Self-Reflection Model**：生成 verbal reinforcement——用自然语言总结"哪里做错了、为什么、下次怎么做"

**Episodic Memory（情节记忆）**：
- 将 Self-Reflection 的输出存入记忆
- 下次执行同类任务时，从记忆中检索相关经验
- Actor 的输入 = 任务 + 从记忆中检索到的相关反思

### 数学直觉

**Reflexion 循环**：

$$
\begin{aligned}
o_t &= \text{Actor}(x, M_{t-1}) \\
r_t &= \text{Evaluator}(o_t, y^*) \\
m_t &= \text{SelfReflection}(x, o_t, r_t) \\
M_t &= M_{t-1} \cup \{m_t\}
\end{aligned}
$$

其中：
- $o_t$：第 $t$ 次尝试的输出
- $r_t$：评估结果（成功/失败 + 分数）
- $m_t$：生成的反思（自然语言）
- $M_t$：累积的情节记忆
- $x$：输入任务，$y^*$： ground truth（如有）

**记忆检索**：

执行新任务 $x'$ 时，从记忆中检索最相关的反思：

$$M^* = \{m \in M \mid \text{sim}(x', m.\text{task}) > \theta\}$$

Actor 的输入增强为：$(x', M^*)$

### 关键假设

1. **LLM 能自我诊断**：模型能识别自己的错误并总结教训
2. **反思可迁移**：对一个任务的反思能帮助解决相似任务
3. **评估信号可靠**：Evaluator 能准确判断输出质量
4. **记忆检索有效**：相似任务的经验确实相关

---

## ② 母婴出海应用案例

### 场景一：VOC 打标质量自我提升

**业务问题**：

LLM 打标新类型评论时容易出错（如新品类、新品牌、新表达方式）。传统方式是人工审核后修正，效率低。需要 Agent 能自动识别错误、总结规律、避免再犯。

**数据要求**：

- 历史打标结果
- 人工审核后的修正记录
- 评估标准（标签准确率、覆盖率、一致性）

**预期产出**：

```
第 1 次尝试:
  输入: "Momcozy S12  wearable pump  is super convenient"
  输出: [品牌_Momcozy, 便利性_正面]
  评估: 部分失败 — 遗漏了 "穿戴式" 关键属性

Self-Reflection:
  "错误原因: 未识别 'wearable pump' 为产品类型标签。
   教训: 'wearable' / 'hands-free' 等词应映射到 '穿戴式_吸奶器' 标签。
   改进: 下次遇到产品类型描述词时，优先检查是否对应品类标签。"

存入记忆。

第 2 次尝试（新任务）:
  输入: "Elvie  hands-free  pump is quiet"
  检索记忆 → 找到之前的反思
  增强输入: 任务 + "注意: 产品类型描述词(hands-free/wearable)需映射到品类标签"
  输出: [品牌_Elvie, 品类_穿戴式吸奶器, 静音_正面, 便利性_正面]
  评估: 成功
```

**业务价值**：
- 错误率逐次下降，无需重新训练模型
- 反思经验可跨任务复用
- 人工审核聚焦在高价值案例上

---

### 场景二：评论分类策略迭代优化

**业务问题**：

评论分类策略（如先粗分再细分、还是一步多标签）的效果受数据分布影响。当新产品上线或评论风格变化时，原有策略可能不再最优。需要自动检测策略失效并迭代改进。

**数据要求**：

- 分类策略定义
- 历史分类结果和准确率
- 新批次评论数据

**预期产出**：

```
Reflexion 迭代:

Epoch 1 — 策略: 先情感 → 再属性
  准确率: 82.3%
   evaluator: "情感分类准确(91%)，但属性映射错误率较高(15%)"

Self-Reflection:
  "问题: 情感先分类导致属性识别时丢失了上下文。
   例: '价格贵但值得' → 情感=正面，但属性'价格'映射失败。
   改进: 改为先识别所有属性提及，再分别判断情感。"

存入记忆 → 更新策略

Epoch 2 — 策略: 先属性 → 再情感
  准确率: 87.1% (+4.8pp)
  evaluator: "属性识别提升(95%)，情感准确率略降(88%)"

Self-Reflection:
  "整体提升，但否定句式情感判断仍有问题。
   例: '不是说不静音' → 误判为负面。
   改进: 增加否定句式识别规则。"

Epoch 3 — 策略: 先属性 → 否定检测 → 再情感
  准确率: 89.4% (+2.3pp)
  evaluator: "满足阈值(≥88%)，停止迭代。"
```

**业务价值**：
- 策略自动迭代，无需人工调参
- 每次迭代的改进方向有明确解释
- 策略演进历史可追踪，便于审计

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/reflexion_self_reflect/reflexion_agent.py`

核心组件：
- `Actor`: 执行任务的 Agent
- `Evaluator`: 评估输出质量
- `SelfReflection`: 生成 verbal reinforcement
- `EpisodicMemory`: 情节记忆存储与检索
- `ReflexionLoop`: 主循环（执行 → 评估 → 反思 → 记忆）

运行方式：
```bash
cd paper2skills-code/mas/reflexion_self_reflect
python reflexion_agent.py
```

生产环境建议：
1. 使用向量数据库存储记忆，支持语义检索
2. 设置评估阈值，只有"有意义"的失败才触发反思（避免噪声）
3. 定期清理陈旧记忆，防止知识过时
4. 结合 RLHF：人工审核反思质量，奖励好的反思
5. 与 Self-Refine 结合：Reflexion 负责跨任务经验，Self-Refine 负责单任务内迭代

---

## ④ 技能关联

### 前置技能
- **Self-Refine**：单任务内的自我改进（Reflexion 是其跨任务扩展）
- **LLM Prompt Engineering**：理解 system prompt 和 few-shot 设计

### 延伸技能
- **RLHF**：人类反馈强化学习（Reflexion 的轻量级替代）
- **Meta-Learning**：跨任务学习通用策略
- **Experience Replay**：从记忆中采样经验进行训练

### 可组合技能
- **Self-Refine**：Reflexion 管跨任务记忆，Self-Refine 管单任务迭代
- **ReAct**：ReAct 的 Observation 可作为 Evaluator 的输入
- **CAMEL**：AI User 作为 Evaluator，对 AI Assistant 的输出进行评判
- **MAS Orchestrator**：全局记忆库供多个 Agent 共享

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 打标质量自我提升 | 错误率逐次下降 30-50%，无需重训 | 开发 1-2 周 | 15-20x |
| 策略自动迭代优化 | 策略调参从人工 2-3 周缩短到自动 1 天 | 开发 2 周 | 12-18x |
| 跨任务经验复用 | 新场景冷启动时间缩短 60% | 开发 1-2 周 | 10-15x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：中，需要评估信号和 ground truth
- 技术门槛：中高，需要理解记忆检索和反思生成机制
- 工程复杂度：中高，涉及多组件协调
- 维护成本：中，记忆库需要管理

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **无需重训**：最大的优势，零模型训练成本
- **可解释性强**：反思是自然语言，人类可读可审核
- **NeurIPS 2023 验证**：HumanEval 91% pass@1，超越 GPT-4
- **与 Self-Refine 互补**：两者结合覆盖单任务和跨任务改进

---

## 参考论文

1. **Reflexion: Language Agents with Verbal Reinforcement Learning** (NeurIPS 2023)
   - Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S. (Northeastern / MIT)
   - 核心贡献：言语强化学习，无需权重更新的 Agent 自我改进
   - arXiv：2303.11366

---

## 与 Self-Refine 的关系

| 维度 | Self-Refine | Reflexion |
|------|-------------|-----------|
| 粒度 | 单任务内迭代 | 跨任务经验积累 |
| 改进方式 | 同一任务的多次尝试 | 相似任务的经验复用 |
| 记忆 | 短期（当前任务） | 长期（情节记忆库） |
| 更新机制 | 无需更新 | 无需权重更新 |
| 最佳组合 | **两者结合**：Self-Refine 解决当前任务，Reflexion 积累长期经验 |

```
新任务到来
    ↓
[Reflexion] 检索相似任务的经验
    ↓
[Actor] 使用经验增强的输入生成初稿
    ↓
[Self-Refine] 在当前任务内迭代改进
    ↓
[Evaluator] 评估最终输出
    ↓
[Reflexion] 生成反思并存入记忆
    ↓
（下一个任务）
```

---

## ⑥ 原文引用

> **底本**：本卡 frontmatter 的 `paper_id` 即 arXiv 编号；其全文存档为 `paper2skills-vault/papers/` 下本论文目录的 `fulltext.md`（LaTeXML HTML 转 Markdown，章节号完整）。
> ⚠️ **底本版本警告**：该存档取自 **arXiv v1**（首页题名为 *Reflexion: an autonomous agent with dynamic
> memory and self-reflection*，作者为 Shinn / Labash / Gopinath），只含 **AlfWorld 与 HotPotQA** 两个 benchmark，
> **不含 HumanEval / MBPP**。卡片「参考论文」栏写的是 NeurIPS 2023 正式版
> （*Language Agents with Verbal Reinforcement Learning*，作者含 Cassano / Narasimhan / Yao），
> 两者不是同一版本。故 ⑤ 的 HumanEval pass@1 断言在**本底本中零命中**，
> 详见本卡维护记录（不改正文）。
> 下列引文均为底本中的**连续子串**（由 `quote_check.py` 逐字核验）。

### A. 核心机制：言语强化学习与三组件（对应 ①核心思想、①数学直觉的循环）

> 原文:"Building on recent research, we propose Reflexion, an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities."
> 出处：2303.11366 Abstract

> 原文:"In this study, Reflexion leverages ReAct (Yao et al.,, 2023), but any decision-making approach can be used in future implementations."
> 出处：2303.11366 §2 Architecture

> 原文:"The reflection loop aims to help the agent correct common cases of hallucination and inefficiency through trial and error. The model used for self-reflection is an LLM prompted with two-shot learning examples of domain-specific failed trajectory and ideal reflection pairs."
> 出处：2303.11366 §2.3 Reflexion

> 原文:"A heuristic $\operatorname{h}(s_{t},a_{t},\Omega,\varepsilon,\left[a_{o},o_{0},\ldots,a_{t-1},o_{t-1}\right])$ is defined to tell the agent when to reflect, where $t$ is the time step, $s_{t}$ is the current state, $\Omega$ and $\varepsilon$ are hyperparameters for the maximum number of repetitive action cycles and the maximum number of total actions allowed, and $\left[a_{o},o_{0}\ldots,a_{t-1},o_{t-1}\right])$ is the trajectory history."
> 出处：2303.11366 §2.2 Heuristics

### B. 评估信号：二值奖励（对应 ①「Evaluator 给出成功/失败信号」与 ①关键假设 3）

> 原文:"Typically, designing or training an effective yet broadly-applicable reward model can be challenging. In this work, we limit the agent to a binary reward model. A binary reward model is a type of reward function that assigns a value of 0 or 1 to an action taken by the agent in the current state. 1 indicates a successful outcome and 0 indicates an unsuccessful outcome."
> 出处：2303.11366 §2.4 Reward model

### C. 论文自报效果（对应 ⑤「无需重训」「效果验证」的量级参照）

> 原文:"To assess our approach, we evaluate the agent’s ability to complete decision-making tasks in AlfWorld environments and knowledge-intensive, search-based question-and-answer tasks in HotPotQA environments. We observe success rates of 97% and 51%, respectively, and provide a discussion on the emergent property of self-reflection."
> 出处：2303.11366 Abstract

> 原文:"We ran the agent without reflection for a single trial to establish a starting point, which achieved 63% accuracy (Fig. 2), which is in-line with the results from (Yao et al.,, 2023)"
> 出处：2303.11366 §3.1 AlfWorld（Results）

> 原文:"Given the ability to reflect, the agent used ReAct to solve 97% of the given environments in 12 trials, failing to solve only 4 out of 134 tasks."
> 出处：2303.11366 §3.1 AlfWorld（Results）

> 原文:"Reflexion enabled the agent to successfully answer 54% of the questions from the dataset, outperforming the base ReAct agent by 20% (Fig. 3) and its first trial attempt by 22%."
> 出处：2303.11366 §3.2 HotPotQA（Results）

> 原文:"While the base ReAct agent achieved 34% accuracy and the Reflexion agent achieved 32% accuracy in the first trials (Fig. 3), the Reflexion agent was able to outperform the base ReAct agent over the course of 7 trials."
> 出处：2303.11366 §3.2 HotPotQA（Results）

> 原文:"It is important to note that the comparison between Reflexion + ReAct and ReAct is not one of accuracy for state-of-the-art performance, but rather a demonstration of improved accuracy by self-improved learning rather than success by retry."
> 出处：2303.11366 §3.1 AlfWorld（Results）

### D. 边界与失败模式（论文自承局限；卡片正文未声明，此处仅备查）

> 原文:"While Reflexion enabled the agent to discover new problem-solving techniques in AlfWorld decision-making tasks and HotPotQA knowledge-intensive tasks, we observed a shortcoming in its ability to improve on its baseline performance in a third benchmark, WebShop (Yao et al., rint, )."
> 出处：2303.11366 §4.4 Limitations of Reflexion

> 原文:"However, after only 4 trials, we terminated the baseline and Reflexion runs as the agent did not show improvement in accuracy (Fig. 6) and was not generating helpful, intuitive self-reflections."
> 出处：2303.11366 §4.4 Limitations of Reflexion

> 原文:"The agent achieved a $33\%\rightarrow 34\%$ accuracy improvement in the baseline run and a mere $33\%\rightarrow 35\%$ accuracy improvement in the Reflexion run"
> 出处：2303.11366 §4.4 Limitations of Reflexion

> 原文:"In fact, by simply retrying, the base ReAct agent was not able to successfully answer any additional questions."
> 出处：2303.11366 §3.2 HotPotQA（Results）

> 原文:"We demonstrated learning curves on the AlfWorld and HotPotQA benchmarks that significantly outperform base ReAct agents. In addition, we include an inconclusive attempt to improve performance on the WebShop benchmark and provide a discussion that highlights a few limitations of this approach."
> 出处：2303.11366 §5 Conclusion
