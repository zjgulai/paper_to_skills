---
title: EComStage — 电商 Agent 三阶段(Perception/Planning/Action)双向 Benchmark
doc_type: knowledge
module: 16-智能体工程
topic: agent-stage-evaluation
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: EComStage — 电商 Agent 三阶段双向评估框架

---

## ① 算法原理

### 核心思想

**EComStage** 解决现有 e-commerce benchmark 只看"最终任务是否成功"的盲点。它把 LLM Agent 的推理过程分解为三阶段评估,并首次**同时覆盖 customer-oriented 和 merchant-oriented 两类视角**:

- **Perception(感知)**:理解用户意图,识别上下文中的关键信号
- **Planning(规划)**:基于感知做行动方案,选择正确的工作流路径
- **Action(执行)**:输出最终决策或回复

7 个代表性任务覆盖三阶段 × 两视角:

| 阶段 | 任务 | 视角 | 样本数 |
|------|------|------|--------|
| Perception | Query Rewrite | 客户 | 233 |
| Perception | Attitude Classification | 商家 | 424 |
| Perception | Query Match | 客户 | 1927 |
| Perception | Intent Recognition | 客户 | 1367 |
| Planning | Scenario Route | 商家 | 164 |
| Action | Solution Decision | 客户 | 487 |
| Action | RAG-QA | 双向 | 202 |
| **合计** | | | **4804** |

### 数学直觉

**评估指标**:

- close-ended 任务(分类、匹配):accuracy

$$
\text{Acc} = \frac{|\{i : \hat{y}_i = y_i\}|}{N}
$$

- open-ended 任务(生成):用 Qwen3-Embedding-8B 编码后 cosine similarity

$$
\text{Sim}(\hat{r}, r_{\text{ref}}) = \frac{\mathbf{e}(\hat{r}) \cdot \mathbf{e}(r_{\text{ref}})}{\|\mathbf{e}(\hat{r})\| \cdot \|\mathbf{e}(r_{\text{ref}})\|}
$$

**关键实证发现**:

- **没有单一模型在所有阶段都最优**:Claude Sonnet 4 整体最强(84.21),但 Gemini 2.5-Pro 在 Solution Decision 上更好
- **GPT-4o 在 merchant-oriented 任务上反而较弱**(82.30 vs Claude 84.21)
- **Qwen3-4B-Instruct(小模型)接近顶尖**:82.26 ≈ Claude Sonnet 4 - 2pp,显著降本可能
- **Solution Decision 最不稳定**:依赖多轮上下文,各模型方差大

### 关键假设

1. 三阶段可分离评估,虽然现实可能存在 error propagation(论文 limitation 承认)
2. 单次任务评估能反映 Agent 真实能力,即使任务独立(无 cross-stage 影响)
3. 人工标注与 LLM 一致性检查可保证 ground truth 质量
4. 7 个任务能代表 e-commerce 全场景(论文承认有限,可扩展)

---

## ② 母婴出海应用案例

### 场景一:跨境母婴客服 Agent 能力体检

**业务问题**:

跨境母婴客服 Agent 上线 3-6 个月后,运营经常发现"总指标差,但具体哪里差不知道"。需要一套 stage-wise 评估,把整体满意度下降问题定位到具体能力短板:

- 是 **Perception** 出问题(没听懂客户问什么)
- 还是 **Planning** 出问题(听懂但选错处理路径)
- 还是 **Action** 出问题(路径对但回复质量差)

**数据要求**:

- 跨境母婴客服历史会话 5000+(每类任务 200-500)
- 人工标注(双语):意图标签、attitude 标签、scenario route 标签、solution 选项、reference answer
- 用 Qwen3 / GPT-4o 等做一致性检查

**预期产出**:

```
母婴客服 Agent 体检报告 (v1.0, 2026-05):

Perception 阶段:
  · Query Rewrite       85.2% (强项)
  · Intent Recognition  72.8% (弱项 — 妈妈表达情绪含糊,需加强)
  · Query Match         92.1% (强项)

Planning 阶段:
  · Scenario Route      68.5% (弱项 — 母婴退货流程多,路径选错率高)

Action 阶段:
  · Solution Decision   71.3% (中等)
  · RAG-QA              65.4% (弱项 — 多国不同政策 RAG 检索准确率低)

诊断结论:
  · 主要短板:跨国政策检索 + 退货路径选择
  · 建议:升级 RAG 索引 + 增加 retraining 数据
  · 优先级:Action.RAG-QA > Planning.Scenario Route > Perception.Intent
```

**业务价值**:

- 精准诊断:从"客户投诉率上升 12%" → 定位到 "RAG-QA 65.4% 是主因",改造方向明确
- 选型决策:同样支出下能选最匹配业务短板的 LLM
- 持续监控:每月体检报告对比,识别新出现的能力衰退

---

### 场景二:商家端运营 Agent(merchant-oriented)评估

**业务问题**:

母婴跨境平台不仅服务消费者,还要服务商家(品牌方/小商家)。商家 Agent 处理:

- 促销规则解释(Promotion Management)
- 内容审核(Content Review)
- 退款审核(Refund Decision for Advertisers)
- 商品上架自动化

传统 benchmark 只评估 customer-oriented,商家场景能力不可见。EComStage 提供:

**数据要求**:

- 商家工单历史(中英 + 跨境平台 SOP)
- 内容审核记录(图文 + 标签)
- 促销规则数据库

**预期产出**:

```
商家 Agent 评估对比:

任务: Attitude Classification (商家是否传递负面情绪给客户)
  · GPT-4o          79.95
  · Claude Sonnet 4 83.49 ✓ 最优
  · Qwen3-4B-Inst   82.78 (轻量替代)
  · DeepSeek-V3     88.68 ✓ 整体最优,但贵

任务: Scenario Route (商家工单路由)
  · Qwen2.5-72B     89.02 ✓
  · Qwen3-235B      82.32
  · Claude Sonnet 4 88.41

任务: RAG-QA (政策类问答)
  · Qwen3-235B-Inst 69.76
  · Gemini 2.5-Pro  69.97
  · 整体均低于 72%, 是行业难点
```

**业务价值**:

- 商家服务质量量化:从粗放"商家流失率"到精细化"工单路由准确率"
- 模型采购指引:不同场景采用不同模型 → ROI 优化
- 提前发现 GPT-4o 在 merchant 场景的弱点,避免错误选型

---

## ③ 代码模板

代码位置:`paper2skills-code/llm_agent_engineering/agent_stage_evaluation/ecomstage_eval.py`

核心组件:

- `EComStageBenchmark`:7 任务的统一数据结构 + 加载/划分
- `PerceptionTask`:Query Rewrite / Attitude Classification / Query Match / Intent Recognition
- `PlanningTask`:Scenario Route
- `ActionTask`:Solution Decision / RAG-QA
- `Evaluator`:close-ended accuracy + open-ended cosine similarity
- `StageReport`:三阶段 + 双向 dashboard 报告

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/agent_stage_evaluation
python ecomstage_eval.py
```

生产环境建议:

1. 替换 cosine similarity 的 embedding 模型为 Qwen3-Embedding-8B(中文跨境)或多语言模型
2. 数据集构造接入真实工单的多级过滤管道(参考论文 §3.1)
3. 评估并行化(8 GPU 并行如论文 setup)
4. 评估周期建议月度,与业务 KPI 联动
5. 给每个任务设阈值告警:某任务连续 2 月下降 ≥3pp 就触发预警

---

## ④ 技能关联

### 前置技能

- **07-NLP-VOC 自动打标签**:任务标签萃取与对齐
- **05-推荐系统**:RAG-QA 的检索基础
- **09-DataAgent-LLM Skill-SQL-Agent**:商家工单数据处理

### 延伸技能

- **16-智能体工程 Skill-Auto-Skill-Synthesis**(SkillForge):用评估结果驱动 Skill 演化
- **16-智能体工程 Skill-Long-Term-Preference-Memory**(Shopping Companion):Solution Decision 阶段配套 LTM
- **10-MAS Skill-Multi-Agent-Debate**:把多模型同时评估变为 ensemble 决策

### 可组合技能

- **02-A_B实验**:把模型选型变为 A/B 测试
- **14-用户分析 Funnel**:把三阶段评估嵌入到客户旅程漏斗
- **15-营销投放分析**:商家端评估指标与营销 ROI 联动

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 客服 Agent 三阶段体检 | 投诉根因定位准确率 60% → 90%, 优化迭代周期 -50% | 工程 4 周 + 标注 200/任务 | 10-15x |
| 商家 Agent 双向评估 | 模型选型节省 30-50% LLM 成本(小模型 cover) | 工程 3 周 + 标注 200/任务 | 12-20x |
| 月度自动体检 | 每月发现 1-2 个能力衰退,提前 1-2 月干预 | 工程 2 周 + CI 集成 | 8-12x |

### 实施难度

**评分:⭐⭐⭐☆☆(3/5 星)**

- 数据要求:中,需要每个任务 200-500 标注样本
- 技术门槛:低-中,主要是评估流水线 + 报告生成
- 工程复杂度:低,close-ended + cosine sim 都很标准
- 维护成本:低,数据集定期补充即可

### 优先级评分

**评分:⭐⭐⭐⭐⭐(5/5 星)**

- **业务价值极高**:本项目 paper-审核 工作流可直接整合 EComStage,变成 Agent 能力体检
- **指导选型**:论文实证 Qwen3-4B-Instruct 接近 Claude Sonnet 4,跨境母婴成本可降 80%+
- **可立即落地**:无需训练,只需标注 + 评估管道,2 周内可上线
- **双向覆盖**:本项目商家(B端) + 客户(C端)两套场景同时受益

### 评估依据

1. **基准充分**:30+ 主流 LLM(1B~235B)在 4804 真实场景标注样本上的实证
2. **方法论扎实**:三阶段分解 + 双向评估 + 多级过滤,可直接迁移
3. **小模型可行性已验证**:Qwen3-4B-Instruct 82.26 ≈ Claude Sonnet 4 - 2pp,显著降本
4. **代码可立即跑**:close-ended 用 accuracy,open-ended 用 cosine sim,实现非常标准

---

## 参考论文

1. **EComStage: Stage-wise and Orientation-specific Benchmarking for Large Language Models in E-commerce** (2026-01)
   - Zhao, K., Meng, Z., Xie, Z., Duan, J., Hu, Y., Liu, Z., Cao, S. — UTokyo / 浙大 / 小红书
   - 核心贡献:首个三阶段(Perception/Planning/Action) + 双向(Customer/Merchant)电商 Agent benchmark
   - arxiv:[2601.02752](https://arxiv.org/abs/2601.02752)

## 相关基础

- **τ-Bench**:retail + airline 工具-Agent 交互 benchmark
- **ECom-Bench** (arxiv:2507.05639):多模态客户支持 benchmark
- **Mix-Ecom**:混合类型 e-commerce 对话 benchmark

---

## 与同领域 Skill 的对比

| 维度 | EComStage | Shopping Companion | SkillForge |
|------|-----------|---------------------|--------------|
| 角色 | 评估 benchmark | 训练 + 推理 framework | Skill 萃取与演化 |
| 阶段覆盖 | Perception/Planning/Action | Stage 1 偏好 + Stage 2 购物 | Failure→Skill 闭环 |
| 视角 | 客户 + 商家 | 客户 | 客户(客服) |
| 数据规模 | 4804 标注样本 | 1.2M 产品 + 1000 实例 | 1883 ticket / 3737 task |
| 立即落地 | 是(2 周) | 中(需 RL 训练) | 中(需历史工单) |

**互补使用**:
- **先用 EComStage 体检**当前 Agent 能力 → 找到短板
- **用 SkillForge 自动萃取/优化** Skill 针对短板
- **用 Shopping Companion 训** 长偏好场景的专门 Agent
- **每月用 EComStage 复测**,形成 PDCA 闭环
