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
- **08-知识图谱**:LTM 存储可结构化为 KG

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
