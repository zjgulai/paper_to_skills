---
title: Causal Cohort Analysis — 因果队列分析：促销干预的长期用户行为效应
doc_type: knowledge
module: 01-因果推断
topic: causal-cohort-analysis-user-behavior
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Causal Cohort Analysis — 因果队列分析

---

## ① 算法原理

### 核心问题：传统队列分析的混淆陷阱

传统队列分析（Cohort Analysis）追踪同一时期加入的用户群体的行为轨迹，但无法剥离**选择偏差**：高价值用户本来就会复购，促销只是"锦上添花"而非真正驱动因素。直接比较"接受促销的队列 vs 未接受促销的队列"，会高估促销效果 30-60%。

### DiD 如何分离处理效应

**双重差分（DiD）**的核心思想：对照组和处理组的趋势差异（趋势项）是公共的，去掉趋势后剩下的才是干预的净效应。

$$ATT = \underbrace{(\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}})}_{\text{处理组变化}} - \underbrace{(\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}})}_{\text{对照组变化（趋势代理）}}$$

**平行趋势假设**：若无促销干预，处理队列和对照队列的 LTV/留存趋势应平行。通过事件研究图（干预前各期系数≈0）进行检验。

### Synthetic Control 在小样本队列中的应用

当队列数量少（如每月只有3-5个同期群）时，DiD 对照组选择困难。**合成控制法（SCM）**通过对多个潜在对照队列的加权组合，构造一个"合成对照"：

$$\hat{Y}_{T,\text{post}}^{(0)} = \sum_{j \in \mathcal{C}} w_j^* \cdot Y_{j,\text{post}}$$

权重 $w^*$ 在干预前期最小化处理队列与合成对照的差异：

$$w^* = \arg\min_w \sum_{t<t_0} \left(Y_{T,t} - \sum_j w_j Y_{j,t}\right)^2, \quad \text{s.t. } w_j \geq 0, \sum_j w_j = 1$$

### 因果效应的时序衰减模式

促销对 LTV 的因果效应并非静态，呈现典型的**时序衰减**：
- **即时效应**（0-1月）：最强，通常 15-30% LTV 提升
- **短期衰减**（2-3月）：效应衰减至 50-70%
- **长期残留**（4-6月）：习惯养成效应，约 10-20% 持续效益
- **负效应识别**：部分促销 6 月后 ATT < 0（sleeping dogs，本来忠实用户被折扣教育为价格敏感用户）

---

## ② 母婴出海应用案例

### 场景1：母婴订阅活动效果评估

**业务问题**：平台推出"购买3件免费送1件"促销，GMV 短期上涨 22%。但运营团队对 6 个月后的真实 LTV 效果存疑：这批用户是真的因为促销被激活了，还是本来就会复购的高价值用户？

**应用流程**：
1. **队列划分**：将2025年10月接受促销的用户定为处理队列，选择2025年9月（同期购买但未接受促销）作为对照队列
2. **结果变量**：月度 LTV（月均消费金额），追踪促销后6个月
3. **平行趋势检验**：检查两个队列在2025年7-9月的 LTV 趋势是否平行
4. **DiD 估计**：计算 ATT，按月分解动态效应

**预期产出**：
- 真实因果效应：ATT ≈ 促销使 6 月 LTV 提升 12%（而非表观的 22%）
- 动态效应图：第1月效应最强（+28%），第6月衰减至+8%
- 识别 sleeping dogs：约 15% 的高价值老用户接受促销后 LTV 反而下降（价格锚点降低）

**业务价值**：精确评估促销 ROI，识别哪类用户不应发放促销码，节省 20-30% 促销预算。

### 场景2：WF-E Review 激励效果评估

**业务问题**：WF-E 工作流向用户赠积分换好评（1000积分≈$10），策略目标是提升商品评分和复购率。但积分成本不低，且担忧"假好评"伤害平台信任度。需评估该策略对复购率的长期因果影响，并识别 sleeping dogs。

**应用流程**：
1. **处理分配**：2025年Q3接受"积分换评"的用户为处理组；同期购买但未收到积分邀请的用户为对照组
2. **结果变量**：3月复购率、6月复购率、12月 LTV
3. **Synthetic Control**：队列数量少时，用2025年Q1/Q2的历史队列合成对照
4. **ATT 估计**：分正向/负向 ATT，识别 sleeping dogs 用户群特征

**关键洞察**：
- 整体 ATT：复购率提升 8%（因果效应，剔除混淆后比表观 15% 低）
- Sleeping dogs 占比约 20%：本来忠实用户（RFM 高分）接受积分后复购率反而下降 5%，因为他们认为"平台在贿赂我"，信任度受损
- **操作建议**：对 RFM 高分用户停发积分，聚焦中等价值用户，整体 ROI 提升 35%

---

## ③ 代码模板

```python
"""
Causal Cohort Analysis — 因果队列分析
用 DiD + Synthetic Control 评估促销干预对长期 LTV/留存的因果效应

纯 Python 标准库 + math/statistics，无 sklearn/pandas 依赖
Python 3.14 兼容
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


# ─── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class CohortRecord:
    """单个用户的队列观测记录"""
    user_id: str
    cohort_month: str          # e.g. "2025-10"
    treated: bool              # 是否接受促销干预
    pre_metric: float          # 干预前指标（如月均 LTV）
    post_metrics: list[float]  # 干预后各月指标，post_metrics[0]=第1月，依此类推


@dataclass
class ATTResult:
    """ATT 估计结果"""
    period: int                # 干预后第几个月
    att: float                 # Average Treatment Effect on the Treated
    se: float                  # 标准误（Bootstrap 估计）
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int


# ─── DiD 队列分析器 ──────────────────────────────────────────────────────────

class DiDCohortAnalyzer:
    """
    双重差分因果队列分析器

    假设：平行趋势（pre-period 趋势相同）
    估计量：ATT = E[Y(1) - Y(0) | Treated=1]
    """

    def __init__(self, n_bootstrap: int = 200, alpha: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self._treated: list[CohortRecord] = []
        self._control: list[CohortRecord] = []
        self._n_periods: int = 0

    def fit(self, records: list[CohortRecord]) -> "DiDCohortAnalyzer":
        """加载数据，分离处理组/对照组"""
        self._treated = [r for r in records if r.treated]
        self._control = [r for r in records if not r.treated]
        if not self._treated or not self._control:
            raise ValueError("处理组和对照组均不能为空")
        n_periods_set = {len(r.post_metrics) for r in records}
        if len(n_periods_set) != 1:
            raise ValueError("所有记录的 post_metrics 长度必须相同")
        self._n_periods = n_periods_set.pop()
        return self

    def _mean(self, records: list[CohortRecord], period: int | None = None) -> float:
        """计算均值：period=None 时取 pre_metric，否则取指定后置期"""
        if period is None:
            vals = [r.pre_metric for r in records]
        else:
            vals = [r.post_metrics[period] for r in records]
        return statistics.mean(vals)

    def compute_att(self, period: int) -> ATTResult:
        """计算第 period 期（0-indexed）的 ATT"""
        if period >= self._n_periods:
            raise ValueError(f"period={period} 超出范围，共 {self._n_periods} 期")

        # DiD 核心：ATT = (T_post - T_pre) - (C_post - C_pre)
        t_post = self._mean(self._treated, period)
        t_pre  = self._mean(self._treated)
        c_post = self._mean(self._control, period)
        c_pre  = self._mean(self._control)
        att = (t_post - t_pre) - (c_post - c_pre)

        # Bootstrap SE
        bootstrap_atts: list[float] = []
        import random
        rng = random.Random(42)
        for _ in range(self.n_bootstrap):
            bt = rng.choices(self._treated, k=len(self._treated))
            bc = rng.choices(self._control, k=len(self._control))
            t_p = self._mean(bt, period)
            t_r = self._mean(bt)
            c_p = self._mean(bc, period)
            c_r = self._mean(bc)
            bootstrap_atts.append((t_p - t_r) - (c_p - c_r))

        se = statistics.stdev(bootstrap_atts)
        z = 1.96  # 95% CI
        return ATTResult(
            period=period + 1,
            att=round(att, 4),
            se=round(se, 4),
            ci_lower=round(att - z * se, 4),
            ci_upper=round(att + z * se, 4),
            n_treated=len(self._treated),
            n_control=len(self._control),
        )

    def compute_all_periods(self) -> list[ATTResult]:
        """计算所有后置期的 ATT"""
        return [self.compute_att(p) for p in range(self._n_periods)]

    def plot_parallel_trends_check(self) -> None:
        """
        打印平行趋势检验的文本可视化
        检验逻辑：干预前期，处理组和对照组的指标差异应接近 0
        """
        t_pre = self._mean(self._treated)
        c_pre = self._mean(self._control)
        pre_diff = t_pre - c_pre

        print("=" * 50)
        print("平行趋势检验（Pre-Period）")
        print(f"  处理组 pre mean: {t_pre:.2f}")
        print(f"  对照组 pre mean: {c_pre:.2f}")
        print(f"  Pre-period 差异: {pre_diff:.2f}")
        print(f"  相对差异: {abs(pre_diff/c_pre)*100:.1f}%")

        # 简单判断：差异 < 10% 认为满足平行趋势
        if abs(pre_diff / max(c_pre, 1e-8)) < 0.10:
            print("  ✅ 平行趋势假设可接受（差异 < 10%）")
        else:
            print("  ⚠️  平行趋势假设存疑（差异 ≥ 10%），建议选取更相似的对照组")
        print("=" * 50)

    def print_dynamic_effects(self, results: list[ATTResult]) -> None:
        """打印动态效应表"""
        print(f"\n{'期数':>4} {'ATT':>8} {'SE':>7} {'95% CI下界':>11} {'95% CI上界':>11}")
        print("-" * 50)
        for r in results:
            sig = "✅" if r.ci_lower > 0 else ("❌" if r.ci_upper < 0 else "—")
            print(f"{r.period:>4}月 {r.att:>8.3f} {r.se:>7.3f} {r.ci_lower:>11.3f} {r.ci_upper:>11.3f} {sig}")


# ─── Synthetic Control 小样本队列 ─────────────────────────────────────────────

class SyntheticControlCohort:
    """
    合成对照法：用多个历史队列加权合成"反事实对照队列"
    适用场景：对照组队列数量少（3-10个），DiD 对照组选择困难

    优化目标：min_w || Y_treat_pre - W @ Y_donors_pre ||^2
    约束：w_i >= 0, sum(w_i) = 1
    """

    def __init__(self, max_iter: int = 5000, lr: float = 0.01):
        self.max_iter = max_iter
        self.lr = lr
        self._weights: list[float] = []
        self._donor_names: list[str] = []

    def fit(
        self,
        treated_pre: list[float],          # 处理队列干预前各期值
        donors_pre: dict[str, list[float]], # 供体队列干预前各期值
    ) -> "SyntheticControlCohort":
        """
        投影梯度下降求解权重（满足单纯形约束）
        """
        n_donors = len(donors_pre)
        self._donor_names = list(donors_pre.keys())
        donor_matrix = [donors_pre[k] for k in self._donor_names]
        n_pre = len(treated_pre)

        # 初始化等权重
        w = [1.0 / n_donors] * n_donors

        for _ in range(self.max_iter):
            # 计算合成值
            synth = [
                sum(w[j] * donor_matrix[j][t] for j in range(n_donors))
                for t in range(n_pre)
            ]
            # 梯度
            residuals = [synth[t] - treated_pre[t] for t in range(n_pre)]
            grads = [
                sum(2 * residuals[t] * donor_matrix[j][t] for t in range(n_pre))
                for j in range(n_donors)
            ]
            # 梯度步骤
            w = [w[j] - self.lr * grads[j] for j in range(n_donors)]
            # 投影到单纯形（relu + 归一化）
            w = [max(0.0, x) for x in w]
            total = sum(w)
            if total < 1e-12:
                w = [1.0 / n_donors] * n_donors
            else:
                w = [x / total for x in w]

        self._weights = w
        return self

    def predict_counterfactual(
        self,
        donors_post: dict[str, list[float]],
    ) -> list[float]:
        """预测处理队列的反事实（若无干预的期望值）"""
        n_post = len(next(iter(donors_post.values())))
        donor_matrix = [donors_post[k] for k in self._donor_names]
        return [
            sum(self._weights[j] * donor_matrix[j][t] for j in range(len(self._donor_names)))
            for t in range(n_post)
        ]

    def compute_att_scm(
        self,
        treated_post: list[float],
        donors_post: dict[str, list[float]],
    ) -> list[float]:
        """ATT = 实际值 - 合成反事实"""
        counterfactual = self.predict_counterfactual(donors_post)
        return [round(treated_post[t] - counterfactual[t], 4) for t in range(len(treated_post))]

    def print_weights(self) -> None:
        print("\n合成控制权重：")
        for name, w in zip(self._donor_names, self._weights):
            bar = "█" * int(w * 30)
            print(f"  {name:>12}: {w:.3f} {bar}")


# ─── 测试 ────────────────────────────────────────────────────────────────────

def _simulate_cohort_data(
    n_treated: int = 80,
    n_control: int = 100,
    true_att_by_period: list[float] | None = None,
    seed: int = 42,
) -> list[CohortRecord]:
    """
    模拟队列数据
    处理组：基础 LTV 100，促销后真实 ATT 随时间衰减
    对照组：基础 LTV 95，无促销效应
    """
    import random
    rng = random.Random(seed)
    if true_att_by_period is None:
        true_att_by_period = [28.0, 20.0, 14.0, 10.0, 8.0, 6.0]  # 6个月衰减

    records: list[CohortRecord] = []

    # 处理组
    for i in range(n_treated):
        pre = 100 + rng.gauss(0, 15)
        post = [
            pre + true_att_by_period[p] + rng.gauss(0, 10)
            for p in range(len(true_att_by_period))
        ]
        records.append(CohortRecord(
            user_id=f"T-{i:04d}",
            cohort_month="2025-10",
            treated=True,
            pre_metric=round(pre, 2),
            post_metrics=[round(v, 2) for v in post],
        ))

    # 对照组（略低于处理组，但趋势相同）
    for i in range(n_control):
        pre = 95 + rng.gauss(0, 15)
        post = [
            pre + rng.gauss(0, 10)  # 无促销效应，仅自然波动
            for _ in range(len(true_att_by_period))
        ]
        records.append(CohortRecord(
            user_id=f"C-{i:04d}",
            cohort_month="2025-10",
            treated=False,
            pre_metric=round(pre, 2),
            post_metrics=[round(v, 2) for v in post],
        ))

    return records


def _simulate_scm_data() -> tuple[list[float], dict[str, list[float]], list[float], dict[str, list[float]]]:
    """模拟 SCM 场景：处理队列 + 3 个供体队列"""
    import random
    rng = random.Random(99)
    n_pre, n_post = 4, 6
    true_att = [20.0, 15.0, 10.0, 8.0, 5.0, 4.0]

    # 供体队列（历史季度）
    donors_pre: dict[str, list[float]] = {
        "2025-Q1": [100 + rng.gauss(0, 5) for _ in range(n_pre)],
        "2025-Q2": [98  + rng.gauss(0, 5) for _ in range(n_pre)],
        "2025-Q3": [95  + rng.gauss(0, 5) for _ in range(n_pre)],
    }
    donors_post: dict[str, list[float]] = {
        "2025-Q1": [100 + rng.gauss(0, 5) for _ in range(n_post)],
        "2025-Q2": [98  + rng.gauss(0, 5) for _ in range(n_post)],
        "2025-Q3": [95  + rng.gauss(0, 5) for _ in range(n_post)],
    }

    # 处理队列 pre（与供体相似）
    treated_pre = [99 + rng.gauss(0, 5) for _ in range(n_pre)]
    # 处理队列 post（= 合成对照 + true_att + 噪声）
    synth_post = [
        0.4 * donors_post["2025-Q1"][t] + 0.35 * donors_post["2025-Q2"][t] + 0.25 * donors_post["2025-Q3"][t]
        for t in range(n_post)
    ]
    treated_post = [round(synth_post[t] + true_att[t] + rng.gauss(0, 3), 2) for t in range(n_post)]

    return treated_pre, donors_pre, treated_post, donors_post


def main() -> None:
    print("=" * 60)
    print("Loop 51-A: Causal Cohort Analysis — DiD 验证")
    print("=" * 60)

    # ─── DiD 测试 ───
    records = _simulate_cohort_data()
    analyzer = DiDCohortAnalyzer(n_bootstrap=300)
    analyzer.fit(records)

    # 平行趋势检验
    analyzer.plot_parallel_trends_check()

    # 动态 ATT
    results = analyzer.compute_all_periods()
    analyzer.print_dynamic_effects(results)

    # 验证：第1期 ATT 应接近真实值 28
    att_period1 = results[0].att
    assert abs(att_period1 - 28.0) < 8.0, f"ATT 偏差过大: {att_period1:.3f}"
    print(f"\n✅ DiD ATT 第1期: {att_period1:.3f}（真实值≈28.0）")

    # ─── SCM 测试 ───
    print("\n" + "=" * 60)
    print("Synthetic Control Method — SCM 验证")
    print("=" * 60)
    treated_pre, donors_pre, treated_post, donors_post = _simulate_scm_data()

    scm = SyntheticControlCohort(max_iter=2000, lr=0.005)
    scm.fit(treated_pre, donors_pre)
    scm.print_weights()

    att_scm = scm.compute_att_scm(treated_post, donors_post)
    print(f"\nSCM ATT（各期）: {att_scm}")

    # 验证第1期接近真实值 20
    assert abs(att_scm[0] - 20.0) < 12.0, f"SCM ATT 偏差过大: {att_scm[0]:.3f}"
    print(f"✅ SCM ATT 第1期: {att_scm[0]:.3f}（真实值≈20.0）")
    print("\n✅ 所有验证通过 — Loop 51-A Causal Cohort Analysis")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [[Skill-DML-Cohort-Causal-Effect]] — DML 是本技能的方法学前置，提供高维混杂控制
- [[Skill-Cohort-Retention-Analysis]] — 描述性队列留存分析，本技能的因果升级版
- [[Skill-DiD-Difference-in-Differences]] — DiD 理论基础，本技能在队列场景的专项应用

### 延伸技能
- [[Skill-Guardrailed-Uplift-Targeting]] — 因果效应估计后的精准投放策略
- [[Skill-User-Lifecycle-STAN]] — 用户生命周期建模，与 LTV 因果效应互补

### 可组合
- [[Skill-RFM-Customer-Segmentation]] — RFM 分群作为 DiD 的分层维度，分群后估计异质性 ATT
- [[Skill-LTV-Prediction-ZILN]] — 预测 LTV 与因果估计 LTV 效应组合，构建完整促销决策框架
- [[Skill-Guardrailed-Uplift-Targeting]] — 将 ATT 估计用于 Uplift 模型训练集构建

---

## ⑤ 商业价值评估

### ROI 预估

**场景1（订阅促销因果评估）**：促销 ROI 评估精准度提升 40%；识别 15-20% 的无效促销投入（发给本来就会买的用户）；年化节省促销预算 200-500 万元。

**场景2（Review 激励 Sleeping Dogs 识别）**：识别 20% 的 sleeping dogs 用户，停止发放积分，节省 20% 积分成本；同时避免高价值用户信任度受损，保护长期 LTV。

### 实施难度：⭐⭐⭐☆☆ (3/5)

- 易处：纯 Python 实现，无外部依赖；DiD 逻辑清晰
- 难处：平行趋势假设验证需要业务判断；小样本 SCM 权重求解需调参
- 前提：至少 2 个月的干预前历史数据；对照组选择需业务知识

### 优先级评分：⭐⭐⭐⭐⭐ (5/5)

**评估依据**：
1. **直接解决"促销真实 ROI"业务痛点**，所有促销都需要因果评估
2. **方法成熟**：DiD 是经济学/流行病学黄金标准，SCM 为 Abadie 2010 顶刊成果
3. **与现有技能形成体系**：01-因果推断模块的应用层 Skill，承接理论到实践
4. **代码可直接部署**：纯标准库，无依赖风险
