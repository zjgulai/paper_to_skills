---
title: Agentic AB Testing — AI Agent 驱动 A/B 实验：假设→设计→解读→决策
doc_type: knowledge
module: 02-A_B实验
topic: agentic-ab-testing-automation
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: Agentic AB Testing — AI Agent 驱动 A/B 实验全自动化

---

## ① 算法原理

### 核心思想

传统 A/B 测试有三大痛点：**① 依赖统计专业知识**（功效分析、多重检验校正），**② 分析耗时**（从数据到决策通常需要 2-3 周），**③ 假设生成靠人工经验**（容易受认知偏差影响）。

**Agentic AB Testing** 将整个实验生命周期交给 LLM Agent 驱动：

```
历史数据 → 假设生成 → 实验设计 → 执行监控 → 结果解读 → 决策建议
```

### 关键算法

**1. 自动假设生成（HypothesisGenerator）**：
- 分析历史指标的异常波动（标准差 > 2σ 的时间窗口）
- 对比同类实验的历史提升量（lift baseline）
- 基于"当前指标 × 可优化空间"生成可测试假设

**2. 样本量计算（ExperimentDesigner）**：

$$n = \frac{2(z_{\alpha/2} + z_\beta)^2 \cdot p(1-p)}{(\Delta p)^2}$$

- α = 0.05（I 类错误率），β = 0.20（II 类错误率，功效 80%）
- MDE（最小可检测效应）由历史实验数据自动推荐

**3. 多重检验校正（Bonferroni）**：
- 同时测试 k 个指标时，每个指标的显著性阈值调整为 α/k
- 防止"数据挖掘"导致的假阳性

**4. 自然语言决策解读**：
- 统计显著 + 业务相关 → 输出"推荐 Variant B：CTR 提升 12.3%，置信度 95%，预计年化收益 $X"
- 不显著 → 输出"样本不足/效应量太小/建议延长实验 N 天"

### MAB 自适应分配（Multi-Armed Bandit 增强）

多方案实验时用 Thompson Sampling 动态分配流量，减少对劣质方案的曝光损耗：
- 每日更新 Beta 分布先验
- 表现好的方案自动获得更多流量

---

## ② 母婴出海应用案例

### 场景一：Listing 主图 A/B 自动化（WF-B 工作流）

**背景**：婴儿奶粉旗舰店主图点击率（CTR）低于类目均值 1.2%。

**Agent 执行流程**：
1. **假设生成**：扫描历史数据 → 检测到"使用场景图"类实验历史平均提升 +15% CTR → 生成假设「将主图改为婴儿实际使用场景（妈妈哺乳/喂食）预计提升 CTR 8-15%」
2. **实验设计**：计算样本量（基线 CTR=2.3%，MDE=0.3pp，α=0.05，Power=80%）→ 需要每组 9,800 次曝光，预计运行 7 天
3. **执行**：流量 50/50 分配，日监控心跳（持续监测置信度变化）
4. **结果解读**：`Variant B CTR=2.61% vs Control CTR=2.30%，z=2.41，p=0.016 < 0.05` → 输出：「✅ 推荐 Variant B 上线：CTR 提升 13.5%，统计置信度 98.4%，预计月均 GMV 增量 $18,000」

### 场景二：定价策略三方案 MAB 测试

**背景**：奶粉 SKU 从 $44 调价，测试 $42 / $45 / $48 三种定价。

**Agent 执行**：
- 使用 Thompson Sampling MAB（而非等比例分流）
- 实验第 3 天：$48 方案的 Beta 后验均值最高（Revenue/Session 指标）
- 系统自动将 $48 方案流量从 33% 提升至 52%，同时继续收集数据
- 第 14 天：输出「$48 定价方案 Revenue/Session 较基线提升 9.2%（p=0.008），推荐上线；$42 方案流量降至 8%（已基本淘汰）」

**效果**：相比传统等比例 3-arm A/B，MAB 减少约 35% 的"差方案"流量浪费。

---

## ③ 代码模板

代码位置：`paper2skills-code/ab_testing/agentic_ab_testing/model.py`

**核心类**：
- `Hypothesis`：假设数据类（指标、预期提升、风险等级）
- `HypothesisGenerator`：从历史数据模式自动生成假设
- `ExperimentDesigner`：样本量计算 + 流量分配策略
- `ResultInterpreter`：统计显著性 + 业务含义自然语言解读
- `AgenticABTestRunner`：全流程编排（假设→设计→执行→解读）

**使用示例**：
```python
from agentic_ab_testing import AgenticABTestRunner

runner = AgenticABTestRunner()
hypothesis = runner.generate_hypothesis(
    metric="ctr",
    baseline_value=0.023,
    historical_experiments=[...]
)
design = runner.design_experiment(hypothesis, daily_traffic=3000)
result = runner.interpret_result(control_data, treatment_data, hypothesis)
print(result.recommendation)
```

---

## ④ 技能关联

### 前置技能
- [[Skill-AB-Experimental-Design]]：基础实验设计（样本量、随机化、分层抽样）
- [[Skill-STATE-Robust-Variance-Reduction]]：方差缩减方法（CUPED/MLRATE）
- [[Skill-Power-Analysis-Sample-Size]]：统计功效与样本量计算

### 延伸技能
- [[Skill-BCCB-Causal-Bandits]]：因果 Bandit 更先进的自适应实验框架
- [[Skill-Thompson-Sampling-MAB]]：多臂 Bandit 自适应流量分配

### 可组合技能
- [[Skill-ATLAS-Gradient-Free-Continual]]：持续学习驱动的实验策略演化
- [[Skill-Listing-Quality-Scoring]]：Listing 质量评分与实验目标对齐

---

## ⑤ 商业价值

| 维度 | 指标 |
|------|------|
| **效率提升** | 实验周期：3 周 → 1.5 周（减少 50%） |
| **人力节省** | 无需专职数据分析师全程介入 |
| **决策质量** | Bonferroni 校正防止假阳性，减少错误上线率 |
| **MAB 增益** | 相比固定分流减少约 35% 劣质方案流量损耗 |
| **实现难度** | ⭐⭐⭐☆☆（中等，统计部分可用标准库） |
| **优先级** | ⭐⭐⭐⭐⭐（高，直接影响产品迭代速度） |

### 注意事项

- ⚠️ 新奇效应：实验前 48 小时数据不稳定，建议从第 3 天起计算结果
- ⚠️ 网络效应：Amazon 平台算法调整会干扰实验，建议控制组与实验组在相同时间窗口
- ⚠️ 多重检验：同时监测 5+ 指标时必须应用 Bonferroni 或 FDR 校正
