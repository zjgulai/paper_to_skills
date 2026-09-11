---
title: ReAct — 推理与行动交替执行
doc_type: knowledge
module: 10-MAS
topic: react-reasoning-acting
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
---

# Skill: ReAct — 推理与行动交替执行

---

## ① 算法原理

### 核心思想

**ReAct** (Reasoning + Acting) 提出了一种**推理与行动交织**的范式。核心洞察：**纯推理（Chain-of-Thought）容易幻觉，纯行动（Tool Use）缺乏规划——只有把两者交替进行，才能既保持思维连贯性又确保信息准确性**。

ReAct 的核心循环：

```
Thought（推理）→ Action（行动）→ Observation（观察）→ Thought → Action → ...
```

每个循环中：
1. **Thought**：Agent 进行内部推理——计划下一步、分析现状、更新策略
2. **Action**：Agent 执行具体行动——调用 API、搜索、查询数据库
3. **Observation**：Agent 接收外部反馈——API 返回结果、搜索结果、数据库记录

ReAct 解决了两个关键问题：
- **CoT 的幻觉问题**：推理不再仅依赖内部知识，每一步都可以获取外部真实信息
- **Tool Use 的规划问题**：行动不再盲目，每一步都有明确的推理支撑

### 数学直觉

**ReAct 轨迹**：

设任务为 $Q$，环境为 $\mathcal{E}$。ReAct 生成轨迹：

$$\tau = [(\hat{r}_1, a_1, o_1), (\hat{r}_2, a_2, o_2), ..., (\hat{r}_n, a_n, o_n)]$$

其中 $\hat{r}_t$ 是 reasoning trace（思维过程），$a_t$ 是 action，$o_t = \mathcal{E}(a_t)$ 是 observation。

每步生成：

$$(\hat{r}_t, a_t) \sim \text{LLM}(\text{prompt} | Q, \tau_{<t})$$

即基于任务和所有历史轨迹生成下一步的推理和行动。

**与 CoT 的对比**：

CoT 轨迹：$\tau_\text{CoT} = [\hat{r}_1, \hat{r}_2, ..., \hat{r}_n]$（仅推理，无外部交互）

ReAct 轨迹：$\tau_\text{ReAct} = [(\hat{r}_1, a_1, o_1), ...]$（推理+行动+观察闭环）

### 关键假设

1. **环境可交互**：存在外部工具/数据源可以查询
2. **推理可显式化**：LLM 能将内部推理过程用自然语言表达
3. **行动空间定义清晰**：Agent 知道有哪些行动可用
4. **观察信息有价值**：外部反馈能有效纠正推理偏差

---

## ② 母婴出海应用案例

### 场景一：竞品情报收集 Agent

**业务问题**：

分析竞品需要实时获取多方信息（评论、价格、规格、市场份额），但信息分散在不同平台。单 Agent 要么靠内部知识（容易过时/幻觉），要么盲目查询（效率低）。

**数据要求**：

- 搜索引擎 API / 电商平台 API
- 评论数据库
- 产品规格数据库

**预期产出**：

```
ReAct 执行轨迹:

[Thought 1] 需要分析 Spectra S1 的竞品。先搜索当前主要竞品有哪些。
[Action 1] search("Spectra S1 breast pump competitors 2024")
[Observation 1] 主要竞品: Medela Pump In Style, Elvie Pump, Willow Go, Momcozy S12

[Thought 2] 好的，有 5 个竞品。先从 Amazon 获取各产品的用户评分和评论数。
[Action 2] query_amazon(product_ids=["Spectra-S1", "Medela-PIS", "Elvie-Pump"])
[Observation 2] Spectra: 4.6★ (2,341 reviews), Medela: 4.4★ (5,892 reviews), Elvie: 4.2★ (1,203 reviews)

[Thought 3] Spectra 评分最高但评论数较少。需要深入分析评论内容。提取各产品的 Top 3 正面/负面属性。
[Action 3] analyze_reviews(product="Spectra-S1", top_k_attributes=3)
[Observation 3] 正面: [静音, 双边, 吸力可调], 负面: [价格, 体积, 配件贵]

[Thought 4] 与 Medela 对比，Spectra 在静音上有优势。验证这个洞察是否普遍。
[Action 4] compare_reviews(product_a="Spectra-S1", product_b="Medela-PIS", attribute="静音")
[Observation 4] Spectra 静音提及率 34.2% (正向 91%), Medela 静音提及率 11.0% (正向 62%)

[Thought 5] 数据支撑了洞察。生成竞品对标报告。
[Action 5] generate_report(format="markdown", sections=["概览", "详细对比", "SWOT", "建议"])
[Observation 5] 报告生成完成。
```

**业务价值**：
- 每一步推理都有外部数据支撑，消除幻觉
- 行动有明确目的，查询效率高
- 完整的思考过程可审计、可复盘

---

### 场景二：VOC 异常检测与根因分析

**业务问题**：

当某个产品的负面评论突然增加时，需要快速定位根因。涉及多维度数据（评论内容、时间趋势、竞品动态、产品批次），需要系统性地探索。

**数据要求**：

- 历史评论数据（时间序列）
- 产品批次/版本信息
- 竞品同期数据
- 客服工单数据

**预期产出**：

```
ReAct 根因分析:

[Thought] Spectra S1 本周负面评论率从 12% 上升到 23%。需要找根因。
[Action] query_trend(product="Spectra-S1", metric="negative_rate", period="30d")
[Observation] 负面率在 3 天前开始跳升，与某批次发货时间吻合。

[Thought] 可能是批次问题。查看该批次的常见负面关键词。
[Action] extract_keywords(reviews, filter="batch_X202405", sentiment="negative")
[Observation] 高频词: ["漏奶"(45次), "吸力不足"(32次), "配件松动"(28次)]

[Thought] "漏奶"是新品控问题？检查客服工单是否有类似投诉。
[Action] query_tickets(product="Spectra-S1", keyword="漏奶", period="7d")
[Observation] 工单增加 3x，主要集中在批次 X202405。

[Thought] 根因确认：批次 X202405 的密封圈质量问题导致漏奶。
建议：暂停该批次销售，启动召回，联系供应商。
[Action] generate_alert(severity="high", root_cause="批次质量问题", action_items=["暂停销售", "启动召回"])
```

**业务价值**：
- 根因分析从 2-3 天缩短到 10-30 分钟
- 推理过程可追溯，便于人工复核
- 自动化生成预警和行动建议

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/react_reasoning_acting/react_agent.py`

核心组件：
- `ReActStep`: 单步数据结构（Thought + Action + Observation）
- `ActionRegistry`: 行动注册表（搜索、查询、分析、生成）
- `ReActAgent`: ReAct Agent（推理 → 行动 → 观察循环）
- `Environment`: 模拟环境（API 调用、数据库查询）

运行方式：
```bash
cd paper2skills-code/mas/react_reasoning_acting
python react_agent.py
```

生产环境建议：
1. 接入真实 API（搜索引擎、电商平台、数据库）
2. 实现 action 超时和重试机制
3. 限制最大步数（防止无限循环）
4. 记录完整轨迹用于审计
5. 结合 ToT 在高层规划时使用树搜索，ReAct 在底层执行时使用推理-行动循环

---

## ④ 技能关联

### 前置技能
- **Chain-of-Thought**：理解推理链的概念
- **Tool Use / Function Calling**：理解 LLM 调用外部工具

### 延伸技能
- **WebGPT / WebAgent**：结合网络搜索的 ReAct 变体
- **Code Interpreter**：结合代码执行的 ReAct 变体
- **Multi-hop QA**：多跳问答的 ReAct 应用

### 可组合技能
- **ToT**：ToT 负责高层规划，ReAct 负责每步执行
- **AutoGen**：ReAct Agent 作为 AutoGen 的一种 Agent 类型
- **Self-Refine**：ReAct 轨迹中的 Observation 可作为反馈信号
- **CAMEL**：AI Assistant 使用 ReAct 循环执行任务

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 竞品情报自动收集 | 信息收集时间从 1 天缩短到 30 分钟 | 开发 2-3 周 | 15-20x |
| 异常根因自动分析 | 响应时间从 2-3 天缩短到 30 分钟 | 开发 2 周 | 20-30x |
| 多源数据整合分析 | 消除信息孤岛，一次查询覆盖全维度 | 开发 1-2 周 | 10-15x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：中，需要外部 API/数据源
- 技术门槛：中，核心是 Prompt 设计和行动空间定义
- 工程复杂度：中，需要环境接口和错误处理
- 维护成本：中，API 变动需要适配

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **基础性地位**：ReAct 是现代 LLM Agent 的基石范式
- **广泛适用**：几乎所有需要外部信息的 Agent 场景都适用
- **与现有框架兼容**：LangChain、AutoGen、OpenAI Function Calling 都基于 ReAct
- **可落地性强**：1-2 周可完成 MVP

---

## 参考论文

1. **ReAct: Synergizing Reasoning and Acting in Language Models** (ICLR 2023)
   - Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y. (Princeton / Google)
   - 核心贡献：推理与行动交替的闭环范式，解决 CoT 幻觉和 Tool Use 盲目性问题
   - arXiv：2210.03629

---

## 与 ToT / CoT 的关系

```
简单问题: CoT 足够
  问题 → 推理链 → 答案

需要外部信息: ReAct
  问题 → Thought → Action → Observation → Thought → ... → 答案

需要探索多个方案: ToT
  问题 → [方案A, 方案B, 方案C] → 评估 → 最优方案 → 答案

复杂问题: ToT + ReAct
  问题 → ToT 规划最优路径 → ReAct 执行每步 → 答案
```
