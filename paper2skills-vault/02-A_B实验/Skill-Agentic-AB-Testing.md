---
title: Agent A/B Testing - 交互式 LLM 智能体全自动 A/B 测试框架
doc_type: knowledge
module: 02-A_B实验
topic: agentic-ab-testing
status: stable
created: 2026-05-19
updated: 2026-05-19
owner: self
source: human+ai
paper: arXiv:2504.09723
---

# Skill: Agent A/B Testing — 交互式 LLM 智能体全自动 A/B 测试

> 论文:**Automated and Scalable Web A/B Testing with Interactive LLM Agents** (2025-04) · arXiv:2504.09723

---

## ① 算法原理

### 核心思想

传统 A/B 测试有三大致命痛点：需要真实流量（新站/小站测不起）、测试周期长（数周）、试错成本高（错误版本直接伤害真实用户）。**Agent A/B** 提出用成千上万个拥有不同 Persona 的 LLM 智能体替代真实用户跑 A/B 实验，在产品正式上线前（Pre-deployment）完成转化率、会话时长、点击热图等核心指标的验证。

### 四模块流程

| 模块 | 功能 | 技术要点 |
|---|---|---|
| **Agent Generation** | 基于目标人群分布生成虚拟用户画像池 | LLM 按 Persona 模板批量生成，覆盖年龄/价格敏感度/偏好等维度 |
| **Testing Preparation** | 随机分流到 Control/Treatment 并做协变量平衡检验 | Standardized Mean Difference (SMD) < 0.1 视为均衡 |
| **Autonomous Simulation** | Agent 在 Staging 网页上执行 Perceive-Decide-Act 闭环 | LLM 感知截图 → 决策操作 → Playwright/Selenium 执行 DOM 事件 |
| **Post-Testing Analysis** | 汇总转化率等指标，输出双比例 z 检验结论 | 支持子群分析（不同 Persona 分组的异质效应） |

### 数学直觉

**双比例 z 检验**（核心显著性判断）：

$$z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}_{pool}(1-\hat{p}_{pool})\left(\frac{1}{n_C}+\frac{1}{n_T}\right)}}$$

其中 $\hat{p}_{pool} = \frac{\text{conversions}_C + \text{conversions}_T}{n_C + n_T}$，$p\text{-value} = P(Z > z)$，$\alpha = 0.05$ 为显著水平。

**协变量平衡检验（SMD）**：

$$\text{SMD} = \frac{|\bar{x}_C - \bar{x}_T|}{\sqrt{(s_C^2 + s_T^2)/2}} < 0.1$$

### 关键假设

1. LLM 生成的 Persona 行为分布与真实目标用户足够接近（需要业务人员审核 Persona 配置）
2. Staging 环境与生产环境功能等价（无后端差异干扰）
3. Agent 行为不受其他 Agent 行为影响（无 SUTVA 违反，因为 Agent 间互相隔离）

---

## ② 母婴出海应用案例

### 场景一：黑五落地页上线前压测（DTC 女装独立站）

- **业务问题**：黑五新版 Landing Page 上线前无法确定转化率是否优于旧版，流量昂贵、试错成本极高（预计损失可达数十万美元）
- **数据要求**：
  - Staging 服务器已部署新/旧两版页面（URL 可访问）
  - 目标人群画像配置：年龄段分布、价格敏感度、偏好风格（追求颜值 vs 性价比）
  - 历史转化率基准（Control 组基线，约 10-15%）
- **执行方案**：
  1. 生成 1000 个 Agent：500 个"追求性价比的宝妈"（35-44岁，价格敏感度 > 0.7）+ 500 个"追求颜值的年轻女性"（18-24岁，style=trendy）
  2. 随机分流到旧版（Control）/ 新版（Treatment）
  3. Agent 自主浏览：点击商品图、加购、模拟结账，记录 DOM 事件流
  4. 输出报告：分组转化率差异 + p-value + 子群分析（宝妈群体结账页停留时长变化）
- **预期产出**：
  - 总体转化率对比及显著性结论
  - **子群洞察**：新版对哪个 Persona 提升最大（如 trendy 年轻女性 +18% vs 宝妈 +6%）
  - DOM 热点报告：哪些元素被频繁点击/忽略
- **业务价值**：将"上线后听天由命"变为"上线前数字孪生压测"。以黑五 2 周 GMV 500 万元估算，避免错误版本上线可防止约 5-15% GMV 损失 = **25-75 万元/次**

### 场景二：商品详情页 UI 改版实验（母婴 3C 配件品牌）

- **业务问题**：重新设计的商品详情页（新增视频展示 + 简化参数对比表）是否能提升加购率，但改版工程量大，需要在正式上线前量化价值
- **数据要求**：新旧两版商品页部署到 Staging；目标用户分布（以"有 0-3 岁宝宝的父母"为主）
- **执行方案**：
  1. 配置 Persona：父亲（practical偏好）50% + 母亲（trendy + 价格敏感）50%
  2. 500 个 Agent 跑 2 小时（等效真实用户 2 周实验量）
  3. 重点关注"视频播放率"和"参数表展开率"两个中间指标
- **预期产出**：加购率提升幅度与 95% 置信区间；中间漏斗对比（视频播放 → 加购 → 结账各环节流失率）
- **业务价值**：工程改版成本约 20 工作日，若 Agent 测试显示提升 < 5% 则取消改版，节省约 **10-15 万元/次**人力成本

---

## ③ 代码模板

> 完整可运行代码见 [`paper2skills-code/02-A_B实验/agent_ab_testing_2025/model.py`](../../paper2skills-code/02-A_B实验/agent_ab_testing_2025/model.py)
> 自测方式：`python3 model.py`（无需第三方依赖，6 个测试用例全绿）

```python
"""
Agent A/B Testing 核心使用示例
完整实现见 model.py（含 6 个自测用例）
"""
from model import run_agent_ab_test

result = run_agent_ab_test(
    n_agents=1000,        # 虚拟 Agent 总数
    treatment_lift=0.14,  # Mock 模式下模拟的 Treatment 提升幅度
    seed=42,
    verbose=True,
)

analysis = result["analysis"]
print(f"Control 转化率: {analysis['control']['conversion_rate']:.2%}")
print(f"Treatment 转化率: {analysis['treatment']['conversion_rate']:.2%}")
print(f"相对提升: {analysis['relative_lift']:.2%}  p-value: {analysis['p_value']:.4f}")
print(analysis["recommendation"])
```

**关键类说明**：

| 类/函数 | 职责 |
|---|---|
| `AgentGenerationModule.generate(n, seed)` | 按年龄/价格敏感度/风格分布生成 Persona 池 |
| `TestingPreparationModule.assign(personas)` | 随机分流，返回 control/treatment 两组 |
| `TestingPreparationModule.check_balance(c, t)` | SMD 平衡检验，确保两组可比 |
| `AutonomousSimulationModule.run_simulation(personas)` | 批量仿真，输出 DOM 交互轨迹 |
| `PostTestingAnalysisModule.analyze(traces)` | 双比例 z 检验，输出推荐结论 |

**对接真实 LLM（生产扩展）**：

```python
from model import AgentGenerationModule, TestingPreparationModule, PostTestingAnalysisModule

# 1. 生成 Persona（可替换为 LLM 生成的精细画像）
gen = AgentGenerationModule()
personas = gen.generate(n_agents=500, seed=42)

# 2. 分流
prep = TestingPreparationModule()
control, treatment = prep.assign(personas)
balance = prep.check_balance(control, treatment)
print(f"SMD: {balance['price_sensitivity_smd']} - {'均衡' if balance['is_balanced'] else '需重新分流'}")

# 3. 真实仿真（生产中替换为 LLM + Playwright 调用）
# from your_llm_sim import run_real_simulation
# traces = run_real_simulation(control + treatment, control_url="...", treatment_url="...")

# 4. 分析
# analysis = PostTestingAnalysisModule.analyze(traces)
```

---

## ④ 技能关联

### 前置技能
- [Skill-AB-Experimental-Design](./Skill-AB-Experimental-Design.md) — Agent A/B 是标准 A/B 实验在无流量场景的范式升级，需要掌握基础实验设计原则
- [Skill-Power-Analysis-Sample-Size](./Skill-Power-Analysis-Sample-Size.md) — Agent 数量（样本量）的选择同样需要功效分析支撑

### 延伸技能
- [Skill-AB-Test-Result-Interpretation](./Skill-AB-Test-Result-Interpretation.md) — z 检验结论需要严谨解读（多重检验校正、实际显著性 vs 统计显著性）
- [Skill-Multi-Armed-Bandit](./Skill-Multi-Armed-Bandit.md) — 当需要同时测试多个版本时，MAB 可替代固定分配实验
- [Skill-Switchback-Experiment-Design](./Skill-Switchback-Experiment-Design.md) — 当网页测试存在 SUTVA 违反（如库存效应）时结合 Switchback

### 可组合
- `16-智能体工程/Skill-ReAct` — Agent 自主仿真的 Perceive-Decide-Act 闭环即 ReAct 框架的直接落地
- `14-用户分析/Skill-AGRS` — Agent 生成的子群可对接用户分析的异质效应估计，发现"哪类用户对改版最敏感"

---

## ⑤ 商业价值评估

### ROI 预估

**场景一（黑五落地页压测）**：
- 每次 Agent 测试成本：1000 Agent × 2h API 调用 ≈ **$20-50**（GPT-4o-mini 单价估算）
- 避免错误版本上线的 GMV 保护：**25-75 万元/次**
- **ROI ≈ 5000-15000 倍**

**场景二（商品详情页改版决策）**：
- Agent 测试成本：**$10-30**
- 避免无效改版人力浪费：**10-15 万元/次**
- **ROI ≈ 3000-5000 倍**（最保守估算）

### 实施难度：⭐⭐⭐☆☆ (3/5)

- **易**：Persona 配置灵活，无需历史数据积累（不像 Empirical Bayes 类方法）
- **易**：Mock 模式可立即跑通验证全链路（`python3 model.py` 即可）
- **难**：LLM Persona 行为的真实度需要业务人员审核（Persona 写得差=结论失真）
- **难**：Playwright + LLM 组合的稳定性工程（页面 DOM 变更、超时处理）

### 优先级评分：⭐⭐⭐⭐⭐ (5/5)

**评估依据**：
1. **场景独特性**：解决了"新站/小站测不起 A/B"的根本性痛点，无可替代
2. **可立即落地**：无需真实流量，Staging 环境即可启动，DTC 出海品牌黑五前即可使用
3. **零风险试错**：哪怕测最激进的改版（隐藏运费、移除评价）也不影响真实用户
4. **填补图谱缺口**：02-A_B实验 内首个面向"Pre-deployment 仿真"的 Skill，与其余 5 个 Skill 互补而非竞争
5. **2025年前沿**：论文 2025-04 发布，技术红利窗口期内实施有先发优势
