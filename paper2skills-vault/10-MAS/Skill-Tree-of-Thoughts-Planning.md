---
title: Tree of Thoughts — 树搜索式任务规划
doc_type: knowledge
module: 10-MAS
topic: tree-of-thoughts-planning
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
---

# Skill: Tree of Thoughts — 树搜索式任务规划

---

## ① 算法原理

### 核心思想

**Tree of Thoughts (ToT)** 将 LLM 的推理过程从线性链式思维（Chain-of-Thought）扩展为**树状搜索**。核心洞察：**人类解决复杂问题时会探索多条路径、评估中间进展、在死胡同回溯**——LLM 也应该具备这种"深思熟虑"的能力。

ToT 与 CoT 的本质区别：

| 维度 | CoT | ToT |
|------|-----|-----|
| 结构 | 单一路径 | 分支树 |
| 回溯能力 | 无 | 可剪枝回溯 |
| 探索能力 | 一次生成 | 多路径并行探索 |
| 评估 | 仅最终输出 | 中间节点可评估 |
| 适用任务 | 简单推理 | 需要探索的复杂问题 |

ToT 的四个步骤：
1. **Thought Decomposition**：将问题分解为中间推理步骤（thoughts）
2. **Thought Generation**：从每个节点生成 $k$ 个候选 thoughts（采样或提议）
3. **State Evaluation**：评估每个 thought 的"前景"（用 LLM 自身或启发式函数打分）
4. **Search Algorithm**：用 BFS 或 DFS 搜索最优路径

### 数学直觉

**Thought Tree**：

设问题为 $Q$，thought 空间为 $\mathcal{T}$。ToT 构建搜索树：

$$\mathcal{G} = (\mathcal{S}, \mathcal{E})$$

其中节点 $s_i \in \mathcal{S}$ 是 partial solution（由 thoughts 序列组成），边 $(s_i, s_j) \in \mathcal{E}$ 表示添加一个 thought。

**Thought 生成**：

从节点 $s$ 生成 $k$ 个候选 thoughts：

$$\{t_1, ..., t_k\} \sim \text{LLM}(\text{prompt}_\text{propose}(s))$$

**状态评估**：

对每个候选 thought 打分（价值或置信度）：

$$v(s \oplus t_i) = \text{LLM}(\text{prompt}_\text{evaluate}(s, t_i))$$

**搜索**：
- BFS：每层保留 top-$k$ 最有前景的节点
- DFS：深度优先探索，到达叶子后回溯

### 关键假设

1. **问题可分解**：复杂问题可以分解为一系列中间推理步骤
2. **Thought 可评估**：LLM 能判断中间步骤的质量
3. **搜索空间可控**：分支因子和深度在计算预算内
4. **多条路径有价值**：探索不同思路比单一路径更有可能找到最优解

---

## ② 母婴出海应用案例

### 场景一：VOC 标签体系设计策略搜索

**业务问题**：

设计一个新的 VOC 标签体系时，面临多个设计决策（层级深度、粒度、覆盖范围、与现有体系兼容性）。每个决策影响后续决策，形成决策树。传统方式是人工逐一决策，容易陷入局部最优。

**数据要求**：

- 现有标签体系（v3.9 字典 602 个 tag）
- 评论样本数据
- 设计约束（如"层级不超过 3 层"、"标签数在 500-800 之间"）

**预期产出**：

```
ToT 搜索过程:

根节点: 设计新标签体系
  ├─ 分支1: 按产品功能分类
  │   ├─ 子分支1.1: 3层结构 (评分: 0.72)
  │   ├─ 子分支1.2: 4层结构 (评分: 0.68)
  │   └─ 子分支1.3: 2层扁平结构 (评分: 0.65)
  ├─ 分支2: 按用户旅程分类
  │   ├─ 子分支2.1: 购买前→购买中→使用后 (评分: 0.78) ← 最优路径
  │   └─ 子分支2.2: 认知→决策→体验→推荐 (评分: 0.71)
  └─ 分支3: 混合分类
      └─ ...

最优策略 (BFS 搜索后):
  按用户旅程分类 + 3层结构
  预期覆盖率: 94.2%
  预期人工仲裁率: 3.8%
```

**业务价值**：
- 系统性地探索标签体系设计空间，避免局部最优
- 量化评估不同设计策略的预期效果
- 设计决策有据可依，减少试错成本

---

### 场景二：评论分类策略搜索与优化

**业务问题**：

一条评论可能涉及多个标签（如"Spectra S1 静音好但价格贵"→[静音_正面, 价格_负面]）。分类顺序和策略影响最终标签质量。需要找到最优的分类策略（先粗分类再细分类？先情感再属性？）。

**数据要求**：

- 标注好的评论样本
- 候选分类策略集合
- 评估指标（准确率、覆盖率、一致性）

**预期产出**：

```
ToT 搜索:

根节点: 评论分类策略
  ├─ 路径A: 先情感 → 再属性 → 再细分 (F1: 0.82)
  ├─ 路径B: 先产品 → 再情感 → 再属性 (F1: 0.79)
  ├─ 路径C: 一步多标签分类 (F1: 0.71)
  ├─ 路径D: 先属性 → 再情感确认 (F1: 0.85) ← 最优
  └─ 路径E: 分层迭代细化 (F1: 0.83)

最优策略: 先属性识别 → 再情感确认
  - 属性识别准确率: 91.2%
  - 情感确认准确率: 88.7%
  - 整体 F1: 0.85
```

**业务价值**：
- 自动搜索最优分类策略，超越人工经验
- 策略效果可量化对比
- 新标签加入时可快速重新搜索最优策略

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/tot_planning/tot_planner.py`

核心组件：
- `ThoughtNode`: 树节点（thought 内容 + 得分 + 父节点）
- `ThoughtTree`: 推理树（生成、评估、搜索）
- `BFSPlanner` / `DFSPlanner`: BFS/DFS 搜索策略
- `evaluate_thought`: LLM-based thought 评估

运行方式：
```bash
cd paper2skills-code/mas/tot_planning
python tot_planner.py
```

生产环境建议：
1. 使用真实 LLM API 替代 mock 评估器
2. 对搜索空间剪枝（设置分支因子上限、深度上限）
3. 缓存中间评估结果，避免重复计算
4. 结合并行生成加速 thought 探索
5. 对于实时性要求高的场景，使用 Beam Search 限制搜索宽度

---

## ④ 技能关联

### 前置技能
- **Chain-of-Thought**：理解线性推理链的概念
- **LLM Prompt Engineering**：理解 few-shot、system prompt 设计

### 延伸技能
- **Monte Carlo Tree Search (MCTS)**：用 MCTS 替代 BFS/DFS，更高效的搜索
- **Beam Search**：限制宽度的贪心搜索，平衡质量和成本
- **A* Search**：启发式引导的最优路径搜索

### 可组合技能
- **ReAct**：ToT 负责规划路径，ReAct 负责路径上的执行
- **CAMEL**：AI User 用 ToT 规划分析框架，AI Assistant 执行每条分支
- **MetaGPT**：SOP 中的决策节点用 ToT 搜索最优策略

---


- **可组合**：[[Skill-MAS-Orchestrator]] / [[Skill-ReAct-Reasoning-Acting]]

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 标签体系设计优化 | 覆盖率提升 5-10pp，设计周期缩短 50% | 开发 2 周 | 10-15x |
| 分类策略自动搜索 | F1 提升 3-5pp，超越人工经验 | 开发 1-2 周 | 12-18x |
| 复杂决策辅助 | 多方案量化对比，决策质量提升 | 开发 1 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：低，纯 LLM 推理，无需训练数据
- 技术门槛：中高，需要理解搜索算法和 LLM 评估机制
- 工程复杂度：中高，搜索空间管理和成本控制是关键
- 维护成本：低，Prompt 调整即可适应新场景

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- **核心差异化**：探索式推理是超越 CoT 的关键能力
- **技术前沿**：NeurIPS 2023 高引论文，Game of 24 上 4%→74% 的震撼提升
- **通用性强**：不仅限于 Agent，任何需要决策的问题都适用
- **成本注意**：搜索调用量是 CoT 的数十倍，需要成本控制机制

---

## 参考论文

1. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** (NeurIPS 2023)
   - Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T.L., Cao, Y., Narasimhan, K. (Princeton / Google)
   - 核心贡献：将 LLM 推理扩展为树搜索，支持探索、评估、回溯
   - arXiv：2305.10601
   - 代码：https://github.com/princeton-nlp/tree-of-thought-llm

---

## 与 ReAct / CoT 的关系

```
CoT:  问题 → Thought1 → Thought2 → Thought3 → 答案
      （单一路径，无回溯）

ReAct: 问题 → Thought → Action → Observation → Thought → ...
      （与外部世界交互的线性路径）

ToT:        问题
           /  |  \
        T1a  T1b  T1c
        /|\   |    |
      T2a... T2b  T2c
       |
      答案
      （分支树，可评估、剪枝、回溯）

最佳组合: ToT 负责高层规划 + ReAct 负责每步执行
```
