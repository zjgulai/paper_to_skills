---
title: ACON — Agent 长上下文压缩与 NL 准则优化
doc_type: knowledge
module: 16-智能体工程
topic: context-compression
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: ACON — Agent 长上下文压缩与失败驱动准则优化

---

## ① 算法原理

### 核心思想

**ACON(Agent Context Optimization)** 解决长 horizon LLM Agent 的核心瓶颈:**上下文随交互无界增长**。Agent 在每一步要积累 observation + action,十几步后 context 就爆炸,带来高成本 + 长上下文稀释相关信息。

ACON 的两个核心创新:

1. **双重压缩**:
   - **History Compression**:压缩历史交互(observation + action 序列)
   - **Observation Compression**:压缩当前步的长观察(API 返回值、长文档等)
   - 都用阈值触发:$|h_t| > T_{\text{hist}}$ 或 $|o_t| > T_{\text{obs}}$

2. **失败驱动的 NL guideline 优化**(gradient-free):
   - 收集对比 trajectory:无压缩成功 vs 压缩后失败
   - 让 LLM 分析失败原因 → 自然语言反馈
   - 类似 textual gradient descent 更新 compression guideline
   - **完全不需要 fine-tuning**,适用于闭源 API 模型

### 数学直觉

**优化目标**:

$$
\max_{\psi} \underbrace{\mathbb{E}[\mathcal{R}(s_T(\psi))]}_{\text{maximize task reward}} - \lambda \underbrace{\mathbb{E}[C(\bm{H}'(\psi))]}_{\text{minimize context cost}}
$$

其中 $\psi = (\phi, \mathcal{P})$ 是 compressor 参数 + guideline,$\mathcal{R}$ 是任务 reward,$C$ 是 context 总成本(token 数)。

**两阶段优化**:

- **UT(Utility Maximization)** 步骤:从 contrastive failure 中学习,更新 guideline 使其保留更多关键信息

$$
\text{Feedback}_i = \text{LLM}(\text{Feedback Instr}, \bm{H}_i, \bm{H}_i')
$$

$$
\mathcal{P}^{(1)} = \text{LLM}(\text{Update Instr}, \mathcal{P}^{(0)}, \|_{i=1}^n \text{Feedback}_i)
$$

- **CO(Compression Maximization)** 步骤:在 success trajectory 上学"哪些信息真正被用上",进一步压缩

**蒸馏**:把优化好的大 compressor($\phi_T$)蒸馏到小 compressor($\phi_S$):

$$
\min_{\phi_S} \mathbb{E}_{(x, y) \sim \mathcal{D}^+_{\text{train}}} \left[ -\sum_{n=1}^{L_y} \log f(y_n | x, y_{<n}; \phi_S, \mathcal{P}^*) \right]
$$

### 关键实证发现

- **AppWorld**:peak tokens -25%,准确率与无压缩相当(56.5% vs 56.0%)
- **OfficeBench**:peak tokens -30%,准确率 74.74%(无压缩 76.84%)
- **8-objective QA**:peak tokens -54.5%,EM/F1 甚至**超过**无压缩基线
- **蒸馏**:Qwen3-14B/8B/Phi-4 学 gpt-4.1 compressor,保留 95%+ 性能
- **小 Agent 受益更大**:Qwen3-14B + ACON 在 AppWorld 上 +32%,在 QA 上 +46%

### 关键假设

1. POMDP 框架适用(确定性 transition + 离散 action)
2. LLM 能从 success/failure trajectory pair 中提炼准则(textual gradient 假设)
3. 历史压缩可由阈值 + 子任务相关性触发(论文给阈值 4096 history, 1024 obs)
4. 蒸馏 student 比 teacher 小但仍保持 understanding compression intent 的能力

### 关键局限

- **History compression 不一定省 cost**:KV cache overhead 抵消压缩收益
- **Observation compression 才是真省 cost**:输入压缩直接降 token
- **Compressor 模块本身有额外调用 cost**:蒸馏小模型缓解此问题

---

## ② 母婴出海应用案例

### 场景一:跨境客服长对话 VOC 分析(observation compression)

**业务问题**:

跨境母婴客服 1 次对话经常 10-30+ 轮,Agent 在执行 RCA(Root Cause Analysis)、生成回复、生成报告时需要历史完整对话 + 多次 API 返回(订单详情、物流数据、产品规格)。这些 API 返回非常长(单个 API 可能 5000+ token),累积后超出大多数模型的 128k context。

**数据要求**:

- 跨境客服历史对话 10-30 轮(input)
- 多次 API 返回的原始 JSON(订单/物流/产品)
- 一份"成功 trajectory"(完整 context 下 Agent 给出正确建议)的标注
- 一份"失败 trajectory"(压缩后 Agent 给出错误建议)的标注

**预期产出**:

```
原始 context:
  对话 25 轮 + 5 次 API 返回 = ~80k tokens
  Agent 推理速度慢,部分关键信息被稀释

ACON 优化:
  Stage 1 (UT) 优化 guideline:
    "在压缩 API 返回时,必须保留:订单状态码 + 物流时效 + 产品过敏原列表"
  Stage 2 (CO) 进一步压缩:
    "FAQ 类问答可仅保留结论性段落,跳过推理过程"
  
  压缩后 context: ~25k tokens (-69%)
  Agent 推理准确率: 0.485 → 0.494 EM (持平甚至提升)
  推理延迟: 12s → 4s
```

**业务价值**:

- 长对话客服推理成本 -60-70%(对应 API token 费节省)
- 客服首响时延 -60-70%
- AppWorld benchmark 显示压缩后 Agent 反而**准确率不降**——长上下文稀释效应被消除

---

### 场景二:小模型长报告生成(distillation + small agent)

**业务问题**:

跨境母婴运营每月要做多个长报告:VOC 总结、广告 ROAS 分析、库存周转预警。这些任务都是 long-horizon Agent(10-30 步:查询 SQL → 取数 → 计算 → 检索 KB → 撰写章节)。如果全用 GPT-5 / Claude Opus,月成本极高。

**数据要求**:

- 历史月报范本(作为 reward signal:能生成出与范本一致的就算成功)
- 长查询日志(SQL + KB 检索 + 模型调用 trace)

**预期产出**:

```
方案对比:

A. GPT-5 Agent + 无压缩:
   成本: $0.50/报告 × 30 月报 = $15/月/类
   总成本: 5 类 × $15 = $75/月

B. Qwen3-14B Agent + ACON (蒸馏 compressor):
   compressor: 蒸馏自 GPT-4.1, 保留 95% 性能
   agent: Qwen3-14B + 压缩 context (AppWorld 上 +32% 性能)
   成本: $0.05/报告 × 30 月报 = $1.5/月/类
   总成本: 5 类 × $1.5 = $7.5/月
   
   节省: 90% 成本, 性能接近 GPT-5
```

**业务价值**:

- 月成本节省 90%(从 $75/月 → $7.5/月)
- 蒸馏 compressor 一次性投入,长期复用
- Qwen3-14B 自部署可控,无 API 调用风险
- 论文 Qwen3-14B 在 AppWorld 上 26.8% → 33.9% (+7pp),母婴场景预期类似

---

## ③ 代码模板

代码位置:`paper2skills-code/llm_agent_engineering/context_compression/acon.py`

核心组件:

- `ContextCompressor`:基础压缩器接口
- `HistoryCompressor`:压缩交互历史(阈值触发)
- `ObservationCompressor`:压缩单次观察(阈值触发)
- `GuidelineOptimizer`:失败驱动的 NL guideline 优化(UT + CO 两阶段)
- `CompressorDistiller`:把大模型 compressor 蒸馏到小模型的训练接口
- `TrajectoryCollector`:收集 success/failure trajectory pair
- `Acon`:统一 orchestrator

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/context_compression
python acon.py
```

生产环境建议:

1. 替换简化版的 `_mock_llm_compress` 为真实 LLM API(GPT-4.1 / Claude / Qwen3-Max)
2. `GuidelineOptimizer` 接入 o3 / GPT-5 / Claude Opus 作 prompt optimizer
3. 蒸馏 student 用 LoRA(参考论文 §4.3)节省训练成本
4. 历史压缩阈值默认 4096 token,观察压缩阈值默认 1024 token
5. CO 阶段只用 success trajectory,避免污染

---

## ④ 技能关联

### 前置技能

- **02-A_B实验**:UT/CO 步骤本质上是 A/B 测试 guideline 候选
- **07-NLP-VOC 文本摘要**:压缩 guideline 即"领域特化的摘要规则"
- **16-智能体工程 Skill-Skill-Lifecycle-Design**(SoK):理解 Skill 全生命周期前提

### 延伸技能

- **16-智能体工程 Skill-Agentic-Memory-Management**(AgeMem):把压缩从 prompt 升到 RL policy
- **16-智能体工程 Skill-Active-Context-Pruning**(Focus):仿生粘菌主动剪枝
- **16-智能体工程 Skill-Memory-as-Action**(MemAct):memory 操作整合进 policy

### 可组合技能

- **16-智能体工程 Skill-Auto-Skill-Synthesis**(SkillForge):压缩后的 trajectory 用于 SkillForge 萃取
- **09-DataAgent-LLM Skill-SQL-Agent**:多步 SQL Agent 长 trajectory 压缩典型场景
- **07-NLP-VOC 长篇 VOC 分析**:长 review 阵列的逐步分析压缩

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 客服长对话压缩 | 推理成本 -60-70%, 延迟 -60-70% | 工程 4 周 + guideline 迭代 | 15-20x |
| 月报告生成蒸馏 | 月成本 -90% ($75 → $7.5) | 工程 6 周 + 蒸馏训练 | 25-40x |
| 小 Agent 长任务加速 | Qwen3-14B 性能 +32-46% | 工程 3 周 + 蒸馏 | 12-18x |

### 实施难度

**评分:⭐⭐⭐⭐☆(4/5 星)**

- 数据要求:中,需要 success/failure trajectory pair(50-200 对每场景)
- 技术门槛:中-高,UT/CO 两阶段优化 + LoRA 蒸馏
- 工程复杂度:中,gradient-free + 接现成 LLM API,无 RL 复杂度
- 维护成本:中,guideline 需随业务场景演化定期更新

### 优先级评分

**评分:⭐⭐⭐⭐⭐(5/5 星)**

- **直接降本**:任何长 horizon Agent 场景都受益,本项目 paper-workflow 自身就是长流程
- **小模型增益巨大**:与 Hermes 4 / Qwen3-4B 这种开源模型组合,可实现 GPT-5 级体验 + 1/10 成本
- **gradient-free**:无需 fine-tune backbone,直接套现有 API LLM,投资风险低
- **生效快**:guideline 优化 1-2 周可见效,蒸馏 2-3 周可量产

### 评估依据

1. **实验充分**:3 个 long-horizon benchmark + 9+ baseline 对比
2. **数据扎实**:peak tokens -26-54%,准确率持平或提升,蒸馏保 95%
3. **方法通用**:gradient-free,适用任何 backbone(开源/闭源)
4. **作者背景**:Microsoft + KAIST,工程性强

---

## 参考论文

1. **ACON: Optimizing Context Compression for Long-horizon LLM Agents** (2025-10)
   - Kang, M., Chen, W.-N., Han, D., Inan, H. A., Wutschitz, L., Chen, Y., Sim, R., Rajmohan, S. — KAIST + Microsoft + University of Cambridge
   - 核心贡献:失败驱动 NL guideline 优化 + 双重压缩(history + observation) + compressor 蒸馏
   - arxiv:[2510.00615](https://arxiv.org/abs/2510.00615)

## 相关基础

- **AppWorld** (arxiv:2407.18901):9 应用模拟环境的 long-horizon benchmark
- **OfficeBench** (arxiv:2407.19056):Office 文档生产力 benchmark
- **TextGrad** (arxiv:2406.07496):textual gradient descent 理论锚
- **OPRO** (arxiv:2309.03409):LLM 作 prompt optimizer 范式

---

## 与同领域 Skill 的对比

| 维度 | ACON | AgeMem(P1-3) | Focus(P2) |
|------|------|---------------|------------|
| 压缩对象 | history + observation 两条 | 统一 memory store | history actively pruned |
| 训练范式 | gradient-free guideline opt | 三阶段 RL + GRPO | RL with biological inspiration |
| 是否需 fine-tune | 否(只优化 prompt) | 是(RL 训 policy) | 是(RL) |
| 适合阶段 | 立即落地, 1-2 周见效 | 中长期, 需 RL 训练 | 中期, 需 biological-inspired 探索 |
| 母婴契合 | 客服长对话 / 长报告 | 客户长周期偏好 | 多 session 长持续 |

**互补使用**:
- **先用 ACON** 在现有 Agent 上做 quick win(API-based, 1-2 周见效)
- **再用 AgeMem** 在客户长周期偏好场景做 RL fine-tune
- **长期用 Focus / MemAct** 探索 memory-policy 一体化
