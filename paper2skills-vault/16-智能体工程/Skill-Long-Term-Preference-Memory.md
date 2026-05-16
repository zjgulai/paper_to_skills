---
title: Shopping Companion — 记忆增强的长期偏好购物 Agent
doc_type: knowledge
module: 16-智能体工程
topic: long-term-preference-memory
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: Shopping Companion — 记忆增强的长期偏好购物 Agent

---

## ① 算法原理

### 核心思想

**Shopping Companion** 解决两个长期被忽视的问题:(1) 缺少能评估跨 session 偏好记忆的端到端购物 benchmark;(2) 现有方法把"偏好识别"和"购物执行"当作独立模块,没有端到端联合优化。它把购物 Agent 形式化为 **POMDP**,并提出**两阶段统一框架** + **双奖励 RL 训练**。

两阶段架构:

1. **Stage 1 - Preference Identification(偏好识别)**:通过 memory tools 检索跨 session 对话历史,提取隐含偏好(品牌偏好、尺码历史、价格档),并向用户呈现确认,允许用户介入修正
2. **Stage 2 - Shopping Assistance(购物执行)**:基于确认后的偏好,迭代检索产品 + 校验约束(预算、组合、品类),直到任务完成

5 个工具(基于 memory + product 两个检索引擎):memory search、product search、preference extraction、constraint check、recommendation output。

### 数学直觉

**任务成功条件**:Agent 终态推荐必须同时满足"指令需求"和"偏好约束":

$$
C_{\mathcal{I}} = \bigwedge_{n \in \mathcal{N}(\mathcal{I})} \text{Satisfy}(s_T, n), \quad C_{\mathcal{M}} = \bigwedge_{p \in \mathcal{P}(\mathcal{M})} \text{Match}(s_T, p)
$$

$$
\text{Success}(s_T) = \mathbb{I}[C_{\mathcal{I}} \wedge C_{\mathcal{M}}]
$$

**双奖励**(对应两阶段):

$$
R_1(\tau_1) = \frac{q_1 + m_1 + b \cdot c_1}{1 + F + b \cdot N}, \quad R_2(\tau_2) = \frac{p_2 + q_2 + m_2 + b \cdot (n_2 + u_2)}{2 + F + b \cdot (N+1)}
$$

其中 $b \in \{0, 1\}$ 区分单品任务 vs 组合套装任务,$F$ 是偏好属性数,$N$ 是组合大小。$q$ 是查询相关性,$m$ 是属性匹配,$p$ 是产品有效性,$u$ 是预算可行。

**Tool-wise reward** 解决多轮工具调用的稀疏 reward 问题:

$$
R_{\text{tool}}(\tau) = \frac{1}{|\mathcal{U}(\tau)|}\sum_{u \in \mathcal{U}(\tau)} r(u)
$$

**最终 reward**:$R(\tau) = R_z(\tau_z) + R_{\text{tool}}(\tau) + R_{\text{fmt}}(\tau)$

### 关键假设

1. 用户跨 session 的偏好是稳定的(短期内不会大幅变化)
2. 隐含偏好可以从历史对话中通过 retrieval 提取(非生成)
3. 用户愿意在 Stage 1 提供 confirmation/intervention(否则退化为 single-stage)
4. LLM-judge 与人工标注一致率 ≥ 90%(论文实测达到)

### 实验亮点

- **任务难度**:GPT-5 在 add-on-deals 任务上 success 仅 54%(单品 75%)
- **小模型胜过大模型**:Qwen3-4B + Dual&Tool-wise RL → avg success 63.5%,超过 GPT-4o(49%) 和 GPT-4.1(51%)
- **Tool-wise reward 关键**:相比仅 Dual-reward(+59.5% avg),加 Tool-wise 提升到 63.5%(+4 pp)

---

## ② 母婴出海应用案例

### 场景一:母婴用户长周期偏好建模与精准复购推荐

**业务问题**:

母婴用户的生命周期天然就是"长周期":怀孕 9 个月 + 0-3 岁宝宝 = 接近 4 年的购物窗口。期间偏好不断演化:

- 早期:孕妇装、孕妇维生素、待产包
- 0-6 个月:奶粉品牌、纸尿裤尺码、湿巾过敏属性
- 6-12 个月:辅食、米粉品牌、餐具材质
- 1-3 岁:童装、玩具、英语启蒙

跨境平台(独立站 + Amazon + TikTok Shop)对话是分裂的,传统推荐忽略跨 session 偏好,导致复购推荐不精准。

**数据要求**:

- 用户历史对话(跨 session,跨平台):至少 6 个月以上历史
- 产品库:多平台 SKU 元数据(品牌、尺寸、过敏原、价格、库存)
- 偏好属性 schema:品牌偏好、避免品牌、尺寸历史、过敏原、价格档、风格

**预期产出**:

```
对话示例:

[Stage 1 - 偏好识别]
User: 帮我推荐一个安抚奶嘴

Agent:[检索 memory] 找到 3 个相关历史 session:
  · session_2026_03: "宝宝乳胶过敏,我们不能用乳胶制品"
  · session_2026_04: "上次买的XX牌不喜欢,口感太硬"
  · session_2026_05: "宝宝 4 个月了"

请确认以下偏好:
  · 避免乳胶材质     ✓ 确认 ✗ 修正
  · 避免 XX 品牌     ✓ 确认 ✗ 修正
  · 适合 4-6 月龄    ✓ 确认 ✗ 修正

User: 确认

[Stage 2 - 购物执行]
Agent:[检索产品库] 筛选 silicone 材质 + 非 XX 品牌 + 4-6M 尺码
       → 找到 5 个 SKU
       [check constraints] 全部满足
       → 按推荐分排序,Top 3 给用户
```

**业务价值**:

- 复购推荐 CTR 提升:Shopping Companion 论文 +15-25pp,预期跨境母婴 +10-15pp
- 用户满意度:用户介入(intervention)机制减少"AI 推荐与我偏好不符"投诉
- LTV 提升:精准长周期推荐使购物频次提高 20-30%

---

### 场景二:跨境组合套装(Add-on Deals)智能凑单

**业务问题**:

跨境物流成本高,平台普遍设有"满 $X 免邮"或"组合满减"。母婴用户经常需要"凑单"——但凑单不应破坏偏好(凑个不喜欢的品牌反而退货)。

**数据要求**:

- 凑单规则(预算阈值、品类组合、最少件数)
- 用户偏好(同场景一)
- 产品库(含库存、价格、品类)

**预期产出**:

```
User: 帮我凑够 $80 免邮,我已经买了 $45 的奶粉

Agent:[Stage 1] 检索偏好:
       · 品牌偏好:Pampers, Huggies
       · 避免:含香料的湿巾
       · 价格档:中端($10-25/件)

[Stage 2] 工具序列:
  product_search("纸尿裤 size M")      → 5 个候选
  constraint_check(budget=$35, fragrance=False) → 过滤 2 个
  product_search("无香湿巾 80片")       → 4 个候选
  constraint_check(...)                  → 过滤到 1 个
  
最终推荐:
  · 纸尿裤 M 64片 $25 (Pampers)
  · 无香湿巾 80片 ×1 $11 (Huggies)
  总价: $45 + $25 + $11 = $81 ✓ 免邮
```

**业务价值**:

- 凑单转化率提升:从粗糙的"加价购"提升到偏好驱动推荐,CVR +5-10%
- 客单价 AOV 提升:add-on success 75-80% 时,AOV 提升 18-22%
- 退货率下降:偏好驱动凑单的退货率比随机推荐低 30-40%

---

## ③ 代码模板

代码位置:`paper2skills-code/llm_agent_engineering/long_term_preference_memory/shopping_companion.py`

核心组件:

- `LTMStore`:用 embedding + cosine 检索的长期记忆库
- `ProductIndex`:BM25-like 简化版产品检索(实际生产用 Pyserini/BM25)
- `PreferenceExtractor`:从 retrieved 历史对话提取偏好(Stage 1)
- `ConstraintChecker`:检验产品是否满足偏好和指令约束
- `ShoppingCompanion`:Stage 1 + Stage 2 两阶段 orchestrator
- `DualReward`:R_1/R_2 + tool-wise + format reward 计算(用于 offline 评估)

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/long_term_preference_memory
python shopping_companion.py
```

生产环境建议:

1. embedding 模型从 all-MiniLM-L6-v2 升级到中英文混合模型(适合跨境)
2. 产品索引接入真实 BM25(Pyserini)或 Vespa
3. Dual-reward 中的 LLM-judge 接入 GPT-5 / Claude / Qwen3-Max
4. Stage 1 偏好确认 UI 集成进现有客服或聊天 widget
5. 用 LoRA 微调小模型(论文建议 Qwen3-4B,可降本)

---

## ④ 技能关联

### 前置技能

- **05-推荐系统**:产品检索 + 排序的基础知识
- **14-用户分析**:RFM / Cohort 等用户分层方法,辅助偏好属性设计
- **07-NLP-VOC 自动打标签**:从历史对话萃取偏好属性的能力

### 延伸技能

- **16-智能体工程 Skill-Agentic-Memory-Management**(AgeMem):把 Memory 操作训练为 tool actions
- **16-智能体工程 Skill-Context-Compression**(ACON):长对话压缩
- **16-智能体工程 Skill-Auto-Skill-Synthesis**(SkillForge):从对话历史萃取 Skill

### 可组合技能

- **05-推荐系统 Cold-Start-Meta-Learning**:新用户冷启动时退化为传统冷启动
- **10-MAS Skill-AutoGen-Multi-Agent-Conversation**:把 Stage 1/Stage 2 拆为两个 agent
- **08-知识图谱**:产品 KG 增强属性约束检查

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 长周期偏好建模 | 复购 CTR +10-15pp, LTV +20-30% | 工程 6-8 周 + LoRA 训练 | 12-20x |
| 凑单智能推荐 | 凑单 CVR +5-10%, AOV +18-22%, 退货 -30-40% | 工程 3-4 周 + 产品索引 | 10-15x |
| 偏好确认/介入机制 | 推荐满意度 NPS +15, 投诉率 -20% | 工程 2-3 周 + UI 集成 | 6-10x |

### 实施难度

**评分:⭐⭐⭐⭐☆(4/5 星)**

- 数据要求:中高,需要 6+ 月历史对话和结构化产品库
- 技术门槛:高,需要 RL fine-tuning + dual reward 设计
- 工程复杂度:高,需要 memory + product 两套检索基础设施
- 维护成本:中,工具调用 schema 变更需要重训

### 优先级评分

**评分:⭐⭐⭐⭐⭐(5/5 星)**

- **业务价值极高**:母婴长周期天然适配,跨境多平台天然需要跨 session memory
- **小模型友好**:论文用 Qwen3-4B + LoRA + RL 击败 GPT-4o,显著降本
- **可分阶段落地**:可以先做 Stage 1(纯检索 + LLM 抽取)再加 Stage 2 和 RL
- **数据闭环天然**:客服 + 推荐场景每天产生大量 (preference, success) 信号

### 评估依据

1. **任务真实**:1.2M 真实产品库,1000 实例,需求设计经手动验证
2. **指标可量化**:Acc.(偏好提取) + Succ.(最终推荐) 双指标可分阶段优化
3. **小模型可落地**:4B 模型 + LoRA + RL 即可超越 GPT-4o,推理成本可控
4. **跨境母婴契合**:用户跨 session 偏好 + 跨平台凑单 + 长周期 LTV 是核心场景

---

## 参考论文

1. **Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks** (2026)
   - Yu, Z., Xiao, K., Zhao, H., Luo, T., Zeng, X. — Alibaba International Digital Commercial Group
   - 核心贡献:首个支持 LTM + 真实任务 + 用户介入的统一购物 benchmark + Dual&Tool-wise RL 训练
   - arxiv:[2603.14864](https://arxiv.org/abs/2603.14864)

## 相关基础

- **LongMemEval** (arxiv:2410.10813):长期记忆评估 benchmark
- **WebShop** (NeurIPS 2022):e-commerce agent 基础 benchmark
- **Agentic Memory (AgeMem)** (arxiv:2601.01885):memory operations 训练为 tool actions
- **GRPO**:Group Relative Policy Optimization

---

## 与同领域 Skill 的对比

| 维度 | Shopping Companion | AgeMem(P1-3) | SkillForge(P0-1) |
|------|--------------------|---------------|--------------------|
| 关注层面 | 端到端购物任务 | Memory 管理本身 | Skill 萃取与优化 |
| 训练方法 | Dual + Tool-wise RL | 三阶段渐进式 RL | 失败驱动 prompt 优化 |
| 业务最强 | 复购 / 凑单 / 长周期推荐 | 任何长会话场景 | 客服 / 工单类 |
| 学习曲线 | 中(RL + LoRA) | 中高(三阶段 RL) | 中(prompt + ReAct) |

**互补使用**:
- 用 **AgeMem** 作为底层 memory 操作能力
- 用 **Shopping Companion** 在 e-commerce 场景上做端到端微调
- 用 **SkillForge** 把购物对话失败案例自动萃取为可演化 Skill
