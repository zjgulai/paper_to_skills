---
title: Memory-as-Action — RL 内嵌式记忆操作策略 (DCPO 训练)
doc_type: knowledge
module: 16-智能体工程
topic: memory-as-action
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: Memory-as-Action — 记忆操作嵌入策略 + DCPO 训练

---

## ① 算法原理

### 核心思想

**MemAct(Memory-as-Action)** 把"记忆管理"从外部启发式控制器(sliding window / 外部 summarizer)升级为 **agent policy 内嵌的可学习 action**:

- **传统范式**:agent π_task 只负责任务,memory 由外部 controller 用 rule-based 启发式管理(MemGPT / MemOS)
- **MemAct 范式**:统一 policy π_θ 同时输出 task action 和 memory action,**端到端 RL 训练**

### 与已有方案的关键差异

| 方案 | Memory 控制 | 训练方式 | 端到端? |
|------|-----------|--------|--------|
| MemGPT | 外部 paging | 无 | ❌ |
| Sliding Window | rule-based 截断 | 无 | ❌ |
| Focus (P2-2) | Agent 自主 + prompt 引导 | 无(纯 prompt) | ❌ |
| **MemAct(本)** | **Policy 输出 memory action** | **DCPO RL** | **✅** |

### 形式化:MDP 重新定义

把交互建模为 MDP $\langle S,\mathcal{A},T,R\rangle$:

- **State**:$s_t = H_t$,工作记忆(历史观察序列)
- **Action Space**:$\mathcal{A} = \mathcal{A}_{\text{task}} \cup \mathcal{A}_{\text{mem}}$
  - $\mathcal{A}_{\text{task}}$:任务 action(调用外部 tool / 回答)
  - $\mathcal{A}_{\text{mem}}$:记忆 action(prune / compress / insert summary)
- **Transition**:
  - Task action: $H_{t+1} = H_t \oplus (a_t, o_t)$(append-only)
  - **Memory action: $H_{t+1} = a_t(H_t)$**(**可覆写、可删除,打破 prefix 累积假设**)
- **Reward**:稀疏终态奖励 $R(\tau)$ = task success ($r_{\text{task}}{=}+1$) - resource violation penalty ($r_{\text{pen}}{=}-0.1$)

### 核心挑战:Trajectory Fracture

Standard RL for LLM (GRPO/PPO) 假设 **prefix 累积**:每个 context 是前一个的扩展,policy gradient 计算依赖这个假设。

Memory action 打破这个假设:

```
传统 trajectory:    [x₀] → [x₀, a₁, o₁] → [x₀, a₁, o₁, a₂, o₂] → ...
                    ↑      ↑              ↑
                    prefix prefix         prefix (累积)

MemAct trajectory:  [x₀] → [x₀, a₁, o₁] → [x₀, summary_of_(a₁,o₁), a₂, o₂] → ...
                    ↑      ↑              ↑
                    prefix prefix         ❌ 不再是前一步的 prefix
                                          (trajectory fracture)
```

直接应用 GRPO 会产生 **mismatched context**,梯度计算错误 → 训练不稳定。

### DCPO:Dynamic Context Policy Optimization

论文提出的解决方案:**在每个 memory action 处切分 trajectory**,把每段视为独立的 prefix-extended segment。

**Segmentation**:让 $t_1^{\text{mem}},\ldots,t_K^{\text{mem}}$ 是 memory action 发生的时刻。每个 segment $\sigma_i$ 是 $(t_i^{\text{mem}}, t_{i+1}^{\text{mem}}]$ 之间的子序列,共享 prefix $H_{t_i^{\text{mem}}}$:

```
Segment σ₀: prefix=H_0, generated=(y_1,...,y_{t₁})
Segment σ₁: prefix=H_{t₁} (post-memory-edit), generated=(y_{t₁+1},...,y_{t₂})
...
```

**Trajectory-level Advantage**(GRPO 风格):

$$
A(\tau) = \frac{R(\tau) - \mu_u}{\sigma_u}
$$

其中 $\mu_u, \sigma_u$ 是同 prompt $u$ 下 $N_{\text{traj}}$ 条 trajectory 的 return 均值/方差。

**Loss**:

$$
\mathcal{L}(\theta) = -\mathbb{E}_u\!\left[\frac{1}{|\mathcal{G}(u)|}\sum_{\tau\in\mathcal{G}(u)}\sum_{\sigma_i\in\Sigma(\tau)}\sum_{t\in\sigma_i} m_t^{\sigma_i}\cdot A(\tau)\cdot\log\pi_\theta(y_t\mid H_t)\right]
$$

其中 $m_t^{\sigma_i}$ 是 binary mask,只对每个 segment 中**新生成的 token** 算梯度(不对 prefix 算)。

关键好处:**每个 token 的梯度都用其生成时的精确 context**,避免 mismatch。

### 训练流程

```
1. Cold-Start SFT:
   - DeepSeek-V3.1 模拟 MemAct 风格生成 800 条轨迹
   - 用 segmented SFT(loss masking)训练 7B/14B 模型
   - 6 epoch, lr=5e-5

2. DCPO RL Training:
   - 数据:8k multi-hop QA + 8k multi-objective QA(偏向简单任务,逼模型泛化)
   - N_traj=8 trajectories per prompt, N_seg=16 segments per update
   - lr=1e-6, batch=128, max_turns=35
   - GRPO-compatible
```

### prune_context Tool 设计

```python
prune_context(
    summary: str,        # 模型生成的关键信息总结
    ids_to_prune: list[str],  # 要删除的历史记录 id 列表
) -> None
```

每个 tool call 输出都有唯一 ID 作为 handle,模型决定保留哪些 / 删除哪些 + 写一段 summary 替换。

### 关键实证结果

**Multi-Objective QA**(2-8 个独立子问题):

| 模型 | Avg Acc | Tokens/Round |
|------|---------|--------------|
| Qwen3-235B-A22B | <59% | — |
| Search-R1-14B | <59% | 8,625 |
| **MemAct-14B-RL** | **59.1%** | **3,447** |

**关键洞察**:**14B with MemAct ≥ 235B without MemAct**,且 token 消耗 -60%。

**Multi-hop QA(5 个 benchmark 平均)**:

| Method | 2Wiki | Bamboogle | HotpotQA | Musique | Frames | Avg |
|--------|-------|-----------|----------|---------|--------|-----|
| Base (Qwen2.5-14B) | 0.580 | 0.488 | 0.655 | 0.233 | 0.275 | 0.446 |
| Sliding Window | 0.535 | 0.472 | 0.560 | 0.271 | 0.215 | 0.411 |
| Sliding Window + Summary | 0.540 | 0.442 | 0.692 | 0.268 | 0.335 | 0.455 |
| Search-R1 (Cold-Start) | 0.775 | 0.624 | 0.723 | 0.364 | 0.376 | **0.572** |
| MemAct-SFT | 0.764 | 0.616 | 0.705 | 0.330 | 0.359 | 0.555 |
| **MemAct-RL** | 0.767 | 0.618 | 0.710 | 0.353 | 0.385 | **0.567** |

MemAct-RL ≈ Search-R1,但 **token 消耗显著更低**(论文 Figure 2b)。

**训练效率**:7B 模型用 DCPO + MemAct → **rollout phase -40%, policy update -25%**。

### 两种自适应策略(emergent behavior)

RL 后,不同模型大小自动学到不同策略:

- **14B(强模型)**:效率导向 → 比 SFT 用更少 external tool calls
- **7B(弱模型)**:补偿导向 → 用更多 tool calls + 更频繁 memory management(补内部知识不足)

这证明 MemAct 不强加固定策略,而是给 RL 一个机制去发现自适应策略。

### 关键挑战

- **训练成本高**:DCPO 需要 cold-start SFT + RL 双阶段,工程量比 prompt-based(Focus)大
- **Reward 设计敏感**:稀疏终态 reward 对长 horizon 学习信号弱,需要 r_pen 显式约束 resource
- **trajectory fracture 增加复杂度**:分段 + 采样 + advantage 映射,实现比 GRPO 多 30% 代码

### 关键假设

1. 任务有可机评 success/failure 信号(用 gpt-oss 评估)
2. memory action 数量有限(论文 max 35 turns)
3. 模型够大能学到非平凡 memory policy(7B 已可,但更小模型未验证)

---

## ② 母婴出海应用案例

### 场景一:多目标客服 Agent 训练(替代 prompt-based 方案)

**业务问题**:

跨境母婴客服 agent 处理"多目标"客诉:一次对话里客户同时问"过敏退货 + 物流追踪 + 关税咨询 + 推荐替代品",每个子问题需要独立工具调用。

现状用 Focus(P2-2) 纯 prompt 控制,但有两个问题:
1. 子目标多到 4 个以上时,Focus 频繁压缩导致丢失上下文,准确率从 80% 跌到 60%
2. Prompt 不适配模型,Haiku 4.5 调好的 prompt 换到 Qwen3-7B 失效

**MemAct 落地方案**:

```
1. Cold-Start SFT:
   - 用 Claude Opus 4.6 模拟 MemAct 风格,生成 1000 条客服多目标轨迹
   - 每条轨迹标注: task_action 序列 + prune_context 调用时机 + summary
   - 用 segmented SFT 训练 Qwen2.5-7B 客服基座

2. DCPO RL:
   - Reward: task_success (+1.0), context_overflow (-0.1)
   - 数据: 5k 历史多目标客诉 (脱敏 + 增强生成)
   - 让模型学到: 子目标 1-2 个时少压缩, 5+ 时频繁压缩

3. 部署:
   - 7B 模型 + MemAct 替代原 Haiku 4.5 + Focus
   - 论文实证: 14B-RL 超越 235B baseline → 7B-RL 估算超越当前 Haiku 4.5

预期效果(对应论文数据):
- 多目标准确率: 60% → 75% (+15pp)
- Tokens/Round: 8,625 → 3,447 (-60%)
- 推理成本: 同等准确率下 -50 ~ -70%
```

**业务价值**:

- 模型成本:从 Haiku 4.5 ($1/Mtok output) 切到自训 Qwen2.5-7B ($0.1/Mtok 自建) = -90%
- 训练投入:Cold-Start SFT + DCPO RL ≈ 2-3 周 + 8x H100 训练 ≈ $5k
- ROI: 月度 100k 工单 × ($1 - $0.1) × 0.5 Mtok = $45k/月节省, 1 个月回本

### 场景二:商家选品 Agent 长周期推理(Multi-Objective)

**业务问题**:

商家选品决策需要多目标推理:
- 目标 1:筛 8 个国家可上架品类
- 目标 2:计算每个品类的财务模型
- 目标 3:对比 10+ 竞品的销售数据
- 目标 4:输出选品建议

每个目标涉及 10-15 个 tool call,总计 60+ tool call,context 容易爆 50k 上限。

**MemAct 落地方案**:

```
对应论文 Multi-Objective QA 实验设置:
- 训练数据: 历史选品决策记录 (500 单)
- 训练: 同上 cold-start SFT + DCPO RL
- 部署: 7B/14B 自训模型

实测预期(对应论文 Figure 3, objectives=4-8 区间):
- Baseline (Qwen3-235B 直接调用): accuracy 50%, tokens 200k+
- MemAct-14B-RL: accuracy 59-65%, tokens 80k
- 改进: +9 ~ +15pp accuracy + -60% tokens
```

**业务价值**:

- 选品决策准确率:50% → 65% (+15pp) = 每 100 单选品多 15 单选对
- 单次任务成本:200k token × $1/Mtok = $0.2 → 80k × $0.1/Mtok = $0.008,**-96%**
- 月度 5k 次选品任务:$1000 → $40 = $11.5k/年节省

---

## ③ 代码模板

代码位置:`paper2skills-code/llm_agent_engineering/memory_as_action/memact.py`

核心组件:

- `Action` 抽象类 + `TaskAction` / `MemoryAction` 子类
- `WorkingMemory`:状态管理,每条记录带唯一 ID
- `PruneContextTool`:`prune_context(summary, ids_to_prune)` 实现
- `MemActAgent`:统一 policy,同时输出 task / memory action
- `Trajectory` + `Segment` + `TrajectoryFracturePoint`:trajectory 分段管理
- `DCPOTrainer`:DCPO 训练算法(简化版,演示 segmentation + advantage 计算)
- `ComputeAdvantages`:GRPO-style group-normalized advantage
- 母婴客服 multi-objective demo:模拟"多目标问答 + memory pruning"

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/memory_as_action
python3 memact.py
```

生产环境建议:

1. **基座选型**:Qwen2.5-7B / 14B,论文实证 7B 已可用,14B 性价比更高
2. **Cold-Start SFT**:用 Claude/GPT 模拟生成 ~800 条轨迹,segmented SFT 6 epoch
3. **DCPO RL**:N_traj=8, N_seg=16, lr=1e-6, max_turns=35, 8x H100 训练 ~2 周
4. **Reward 设计**:任务 +1.0, resource 违规 -0.1, 其他 0
5. **Memory tool 接 MCP**:`prune_context` 作为 MCP tool 暴露给 agent
6. **与 Focus 互补**:小任务用 Focus(prompt) 即可,大规模训练用 MemAct(RL)

---

## ④ 技能关联

### 前置技能

- **16-智能体工程 Skill-Active-Context-Pruning**(Focus,P2-2):无训练版本,理解 memory action 的 prompt 端模拟
- **16-智能体工程 Skill-Agentic-Memory-Management**(AgeMem,P1-3):AgeMem 也用 GRPO 风格 + step-wise advantage,但聚焦 inter-task LTM
- **16-智能体工程 Skill-Context-Compression**(ACON,P1-2):被动外部压缩对比

### 延伸技能

- **16-智能体工程 Skill-Orchestration-Trace-RL**(待萃取 P2-5):用 trace 训 orchestration 策略,理念类似但目标不同
- **16-智能体工程 Skill-Co-Evolutionary-Skill-Verification**(EvoSkills,P2-1):skill 演化时若 context 累积可应用 MemAct

### 可组合技能

- **10-MAS Skill-MAS-Orchestrator**:MemAct policy 集成到 MAS 的 worker agent
- **16-智能体工程 Skill-Skill-Lifecycle-Design**(SoK):memory action 作为 first-class skill 注册
- **本项目 paper-审核**:审核长论文时 agent 可用 MemAct 自动剪枝中间笔记

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 多目标客服 Agent 训练 | 模型成本 -90%, 准确率 +15pp, 月省 $45k | 工程 2-3 周 + 训练 ~$5k | 9-12x (首月回本) |
| 商家选品长推理 | 单次成本 -96%, 准确率 +15pp | 工程 3-4 周 + 训练 $8k | 长期 15-20x |
| 通用 Agent base model 训练 | 跨场景复用, 替代 GPT-4 调用 | 工程 4-6 周 + 训练 $15k | 战略价值 |

### 实施难度

**评分:⭐⭐⭐⭐⭐(5/5 星)**

- 数据要求:中高,需 800+ 条 cold-start 轨迹 + 8k+ RL 训练样本
- 技术门槛:**高**,需懂 RL (GRPO/DCPO) + 分布式训练 + segmentation 工程
- 工程复杂度:**高**,DCPO segmentation + advantage mapping + segmented SFT
- 维护成本:中,模型迭代时需重新训练,但 cold-start 流程可复用

### 优先级评分

**评分:⭐⭐⭐☆☆(3/5 星)**

- **方法论价值极高**:把 memory 视为 first-class action 是范式级创新
- **实施门槛极高**:小团队不易承担 RL 训练 + 工程基建
- **替代方案**:小规模用 Focus(P2-2) prompt 即可,大规模才上 MemAct
- **建议**:作为 P3 战略储备,先用 Focus 验证假设,再考虑训练 MemAct base
- **风险**:Cold-start SFT 依赖大模型(Claude/GPT)生成数据,有泄露风险

### 评估依据

1. **论文实证强**:7B/14B 模型 vs 235B 同等准确率,token -60%
2. **理论完备**:trajectory fracture 问题 + DCPO 解法清晰
3. **训练效率**:rollout -40%, update -25% 数据可信
4. **可复现性**:代码框架 GRPO-compatible, 不需重新造轮子
5. **业务相关性**:多目标 QA 与跨境客服多目标客诉高度同构

---

## 参考论文

1. **Memory-as-Action: Autonomous Context Curation for Long-Horizon Agentic Tasks** (2025-10)
   - Yuxiang Zhang, Jiangming Shu, Ye Ma, Xueyuan Lin, Shangxi Wu, Jitao Sang
   - 北京交通大学 + Hithink Research + 华为诺亚方舟
   - 核心贡献:Memory action 嵌入 policy + DCPO 训练算法 + trajectory fracture 处理
   - arxiv:[2510.12635](https://arxiv.org/abs/2510.12635)

## 相关基础

- **GRPO** (DeepSeek-Math, arxiv:2402.03300):group-normalized advantage 基础
- **MemGPT** (arxiv:2310.08560):OS-style memory hierarchy (外部)
- **MemOS** (Z. Li et al.):memory OS for AI system (外部)
- **Search-R1** (arxiv:2503.05292):RL 训练的 search agent (无 memory action)
- **Asearcher** (arxiv:2508.07976):async RL for long-horizon search agent
- **Yarn** (arxiv:2309.00071):long-context window extension

---

## 与同领域 Skill 的对比

| 维度 | MemAct (本) | Focus (P2-2) | AgeMem (P1-3) | ACON (P1-2) |
|------|-----------|-------------|---------------|-------------|
| Memory 控制 | Policy action | Agent 自主 + prompt | RL policy | 外部 module |
| 训练 | DCPO RL | 无 (prompt) | 三阶段 RL | trajectory + distill |
| 范围 | Intra-task | Intra-task | Inter-task LTM | History + obs |
| 实证收益 | -60% token + 59.1% acc | -22.7% token, acc 不变 | acc +10pp | -26~-54% memory |
| 实施周期 | **8-12 周(含训练)** | 短(2-3 周) | 长(8-12 周) | 中(4-6 周) |
| 适用规模 | 大规模 prod | 任何 | LTM 重场景 | 推理时压缩 |

**互补使用**:
- **验证假设阶段**用 Focus(prompt) 快速验证
- **大规模生产阶段**用 MemAct 训练专用模型
- **LTM 持久化**用 AgeMem 三阶段 RL
- **推理时压缩**用 ACON 蒸馏小模型
- **四者叠加**:Focus 在线 prompt + MemAct 训练 base + AgeMem 长期记忆 + ACON 推理压缩
