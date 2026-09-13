---
title: Skill-Incrementality-Measurement
module: 14-用户分析
topic: 「被助攻的乌龙球」下的增量测量——渠道完整结果的受众级 ITT 随机化 + 个体级 PIE 扩展
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2607.09608
paper: "Media Measurement and the Assisted Own Goal: Attribution, Marketing-Mix Models, and Individual-Level Incrementality"
venue: arXiv preprint
venue_tier: preprint
venue_source: frontmatter-as-is
evidence_grade: A
verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py（G2/G3）+ 人工逐条核对 registry 与原文
verified_at: 2026-09-12
supersedes:
related: Skill-Ad-Attribution-Modeling.md, Skill-Cannibalization-Corrected-Attribution.md, Skill-Causal-Budget-Allocation.md, Skill-Marketing-Mix-Modeling.md, Skill-Uplift-Modeling.md, Skill-User-Funnel-Analysis.md
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-05
l2_domain: 品牌与增长
l3_id: DOM-05-105
l3_business: 增量分析
l3_all: 增量分析 / 实验设计
l1_l2_l3: 业务运营/品牌与增长/增量分析
---

# Skill Card: 渠道完整结果的受众级增量测量（Channel-Complete ITT Incrementality Measurement）

**与同目录两张卡的分工**：`Skill-User-Funnel-Analysis.md` 回答「漏斗每一步流失多少」，`Skill-Cohort-Retention-Analysis.md` 回答「按批次看留存」。两张卡都建立在**同一份可观测事件表**之上。本卡问的是更上游的问题：漏斗顶端那些「种草了、但在别的平台成交」的增量，在归因数据里**根本不存在**——不是漏斗哪一步漏了，而是这笔转化从未被记到发起渠道名下。因此本卡与 `Skill-Ad-Attribution-Modeling.md`（归因口径）是**替代关系**，不是叠加关系。

**⚠️ 证据强度声明（先读这一条再看数字）**：这篇论文**没有任何真实数据**。它的第 6 节是一个模拟研究，作者自己在贡献列表里也是写 "outline a simulation design"（⑥ Q5）。因此本卡里凡是带数字的地方，只有三种合法身份：① ⑥ 段逐字引文里的**论文原文数字**（其中大部分是**模拟设定值/模拟输出**）；② 围栏内的**本卡推算或本卡模拟输出**（B 类·可自行复算）；③ frontmatter 里的结构性字段。**任何地方都不会出现「实测效果」。**

---

## ① 算法原理

**核心思想**：上游「种草」平台制造了增量购买，但消费者不信任它的站内结算，转去下游可信 marketplace 成交，于是这笔转化被归因系统记在 marketplace 头上——论文称之为**被助攻的乌龙球**（assisted own goal）：助攻由发起平台完成，进球算在对手账上。**换归因模型救不了**：多触点归因只是在「可见触点」间重新分功，而这笔转化被整条移出了可见集合（⑥ Q11）。

**数学直觉**：可观测因子 κ = 1 − τ(1 − φ)：τ 是被不信任挤到下游成交的比例，φ 是跨 marketplace 的回收率。测量 ROAS = κ × 真实增量 ROAS（Proposition 1，⑥ Q8）；按 ROAS 阈值分配时投入只剩 first-best 的 κ 的 1/(1−β) 次方（Proposition 2，⑥ Q9/Q10）——越会种草，账面越难看。治法是把结果换成**渠道完整**口径（在哪成交都算），在受众级随机化下做 ITT：τ、φ、收割方的记功比例 η 同时被消掉（Proposition 3，⑥ Q18）。

**关键假设**：① 分配必须早于曝光；② 分配对象在激活时可枚举；③ 结果是渠道完整的第一方结果；④ 跨实验交互可忽略、条件效应可迁移到未实验受众（论文 §5.4 与 Limitations，⑥ Q30/Q31）。

---

## ①b 反例与适用边界

**什么时候不要用这个算法**

1. **拿不到渠道完整结果**（只在第三方 marketplace 卖、只有平台后台报表）。此时 τ 与 φ 不可分离，本卡降级为「κ 的诊断区间 + 预算口径复核」，**不得声称任何增量因果**。
2. **受众在激活时不可枚举**。论文明确说 lookalike / broad prospecting 这类拉新广告违反枚举约束：第一方 universe 之外的用户可以被促成转化，却从未被分配过。两个「事后补救」都被论文否掉——事后随机分配转化者会稀释 ITT 到零，按曝光分配则把归因偏差原样搬回来（⑥ Q22）。
3. **想用平台自带的 lift study 替代**。论文指出平台侧随机化虽然交给了能枚举的一方，但其结果**不是渠道完整的**，乌龙球在 lift test 内部依然存在（⑥ Q23）。
4. **需要周/季度粒度的宏观规划**。此时 MMM 更合适——但要注意它的收割方花费内生跟随到店需求，会把转移需求再记一次功（⑥ Q12/Q13）。

**已知的失败模式**

1. **用「预测响应」代替「预测增量」**。论文的数据生成过程里，高响应人群与高增量人群不是同一批人（本卡 ③ 的合成数据复刻了这一点）：按 propensity 排序会把预算推向「不投也会买」的人。
2. **把 φ 当成可以靠工程解决的像素问题**。论文的机制是 marketplace 系统性压制第三方信号，跨 marketplace 交易上「加像素/加回传」大多办不到（⑥ Q6）。
3. **主指标窗口短于「种草 → 在别处成交」的决策时滞**。窗口太短会把增量记成噪声。
4. **把 MMM 的渠道系数直接当因果弹性用**。内生性未解决时它只能定方向、不能定金额。

**论文自己承认的局限**（§7 Discussion & Conclusion → Limitations，逐字见 ⑥ Q30/Q31）

- 模型刻意精简：单一产品、单一 generator 与单一 retailer、静态一次性分配，τ 是 reduced-form 参数（未内生化）。
- 未建模：τ 作为结账投资均衡的内生结果、多触点归因窗口、retailer 策略性压制 φ。
- 测量模型自带假设：渠道完整的第一方结果、跨实验交互可忽略、条件效应可迁移到未实验受众、可寻址受众不变。
- **本卡必须额外声明**：论文没有真实数据（§6 是模拟研究），也**未报告**任何企业的实测收益、实施成本或周期。个体级扩展（§5.3）在本论文里**只有设计、没有个体级实验结果**；论文转引的 PIE 证据（2,226 个 Meta RCT、R² 0.88）来自 Gordon et al. (2023)，是 **campaign 粒**的结果，不能当作本卡个体级映射的验证（⑥ Q20）。

---

## ② 母婴出海应用案例

### 场景 1：独立站站外种草 → 消费者转去 Amazon 成交 → 独立站 ROAS 被系统性低估

- **业务问题**：母婴出海品牌（吸奶器、纸尿裤、奶瓶）用独立站（Shopify）承接，同时在 TikTok 投达人种草。用户刷到短视频、形成购买意图，但因为不放心独立站的支付与退货，转去 Amazon 搜品牌词下单。结果：**这笔增量被记到 Amazon 品牌词广告（收割位）上**，而 TikTok 与独立站在报表上垫底。月度预算评审按 measured ROAS 排序，把钱从种草挪到品牌词——越会种草，账面越难看。这正是论文 §3 的 own goal，也是母婴跨境最贵的测量缺口。
- **数据要求**：① 受众级分配表（可枚举的第一方标识：邮箱/手机号哈希、MAID），每次受众激活自带 control；② **渠道完整**的成交台账——独立站订单、Amazon 订单、线下/其他渠道订单都能回到同一个 user/order key；③ 日粒度，覆盖 ≥1 个完整旺季周期（母婴品类有宝宝月龄生命周期，窗口太短会把生命周期效应误判成投放效果）；④ 两个辅助参数：τ（可用成交渠道的回传覆盖率近似）、φ（平台回传配置）。
- **数据可得性**：`部分可得（需补充 X）`。独立站侧第一方数据与自建 holdout 可得；**Amazon 侧拿不到受众级随机化**，channel-complete 结果要靠「品牌 + ASIN + 订单号/邮箱」做订单回传，覆盖率有限。若回传覆盖率不足以覆盖 τ 的主要部分，则本卡只能产出 κ 的诊断区间，**不得声称增量因果**——这与 registry 的 note「纯平台站内卖家须降级为诊断框架」一致。
- **预期产出**：① κ 的诊断区间与「归因口径 / 真实口径」对照表；② 受众级 ITT 的增量转化率与真实增量 ROAS 区间；③ 个体级 δ̂(x) 打分表，用于挑下一个受众。
- **业务价值**：见 ⑤。价值来自两处：把预算从零增量的收割位挪走；以及把排序口径从 measured ROAS 换成增量口径后，避免**反复砍掉真正在制造需求的渠道**。

### 场景 2：旺季前六周——种草预算与收割位预算的切分

- **业务问题**：黑五前六周，同一笔预算要在「TikTok 达人种草」与「Amazon SP 品牌词」之间切分。现在的切法看上周 measured ROAS，品牌词常年最高。
- **做法**：用 ambient 随机化让**每次受众激活自带 control**（不必为每个 campaign 单独立项），把实验库当成训练资源；再用个体级映射把结论**投射到还没做过实验的受众**（论文 §5.3 的 projection，⑥ Q21）。
- **数据要求与可得性**：`部分可得（需补充 X）`。需要受众激活日志（平台侧可得）+ 渠道完整结果（见场景 1 的回传缺口）+ 分配前特征（消费频次、宝宝月龄段、历史客单）。**分配必须在激活时确定**，不能等转化发生后再补（⑥ Q22）。
- **预期产出**：旺季六周的渠道预算切分建议 + 「哪些受众的增量集中在站外」的清单。
- **业务价值**：即使暂时不上个体级模型，**单是「每次激活自带 control + 渠道完整结果」这一层**，就能把 measured ROAS 的系统性低估暴露出来。

---

## ③ 代码模板

- 依赖：**只需要 numpy**（不引入 sklearn / statsmodels；个体级映射用带极小 ridge 的线性概率 T-learner 闭式求解），因此能过 K1 的断网执行校验。
- 结构：论文解析式（式 (4)、(7)、(8) 与 Proposition 2）→ 合成数据的「乌龙球」模拟 → 渠道完整 ITT 与归因口径对照 → 市场级 MMM 的再记功 → 个体级增量映射与投射 → 断言测试 → 业务演示。
- 与论文的对应：`observability_factor` = 式 (4) 的 κ；`spend_ratio_vs_first_best` = Proposition 2 的 κ 的 1/(1−β) 次方；`itt_estimate` = 式 (9)；`fit_individual_incrementality` / `predict_delta` = 式 10 与 §5.3 的 projection。
- ⚠️ **代码内所有样本都是 numpy 合成的模拟数据。** 论文本身没有真实数据（§6 是模拟研究），本卡的合成数据也**不是**论文的数据，更不代表任何企业的真实效果。代码里的 τ、φ、基线转化率取的是论文 §6 panel (c) 的**模拟设定值**，所以输出数值与论文报告值处于同一量级是**设定相同**的结果，**不构成对论文的独立验证**（论文未公开它的随机数流）。
- ⚠️ 本卡模拟输出与论文报告值**不可互相印证**：⑤ 的收益推算用的是论文的**公式 + 模拟设定值**，不是任何实测结果。

```python
"""
渠道完整结果的受众级 ITT 增量测量（channel-complete ITT + 个体级增量映射）
论文：2607.09608 §2.5 归因测量层 / §2.7 Proposition 2 / §5.2 Proposition 3 /
      §5.3 式 (10) / §6 模拟研究

⚠️ 数据声明：该论文没有真实数据（§6 是一个模拟研究）。本文件里的全部样本都由
   numpy 随机数生成器合成，只用于验证公式自洽与代码可运行；它们既不是论文的输出，
   也不代表任何企业的真实投放效果。
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# 1. 论文的解析公式（式 (4)(7)(8) 与 Proposition 2）—— 闭式解，不含任何实测值
# ---------------------------------------------------------------------------
def observability_factor(tau: float, phi: float) -> float:
    """κ(τ, φ) = (1 − τ) + φ·τ = 1 − τ(1 − φ)。论文式 (4)。

    τ：消费者不信任发起平台的站内结算、转到下游 marketplace 成交的比例；
    φ：跨 marketplace 的回收率（成交发生在 R 时仍能被匹配回发起平台 G 的概率）。
    """
    return (1.0 - tau) + phi * tau


def spend_ratio_vs_first_best(tau: float, phi: float, beta: float) -> float:
    """论文 Proposition 2：s_attr / s_true = κ 的 1/(1−β) 次方（τ>0 且 φ<1 时 < 1）。"""
    return observability_factor(tau, phi) ** (1.0 / (1.0 - beta))


def demand_generated(alpha: float, s: float, beta: float) -> float:
    """论文式 (1)：q(s) = α·s^β（diminishing returns）。"""
    return alpha * float(s) ** beta


def spend_from_roas_threshold(m: float, alpha: float, beta: float,
                              lam: float, kappa: float) -> float:
    """论文式 (7)(8)：按 measured ROAS 打到 hurdle λ 的投入 = (m·κ·α·β/λ) 的 1/(1−β) 次方。

    κ 是归因口径下的可观测因子（真实口径代入 κ=1）。
    """
    return (m * kappa * alpha * beta / lam) ** (1.0 / (1.0 - beta))


def margin_loss_from_mismeasurement(m: float, alpha: float, beta: float,
                                    lam: float, tau: float, phi: float) -> dict:
    """按论文模型推算「归因制下少投」造成的增量毛利缺口。

    这是**本卡用论文 §2.2–2.7 的公式做的推算**（B 类·本地可复算），不是论文报告的收益；
    参数 α、β、m 必须由企业自测，λ 是自定的 ROAS hurdle。下面 _business_demo 里
    代入的 m/α/λ 是**任意演示值**，既不是论文给的，也不是任何公司的真实数据。
    """
    kappa = observability_factor(tau, phi)
    s_attr = spend_from_roas_threshold(m, alpha, beta, lam, kappa)
    s_true = spend_from_roas_threshold(m, alpha, beta, lam, 1.0)
    q_attr = demand_generated(alpha, s_attr, beta)
    q_true = demand_generated(alpha, s_true, beta)
    return {"kappa": kappa, "s_attr": s_attr, "s_true": s_true,
            "q_attr": q_attr, "q_true": q_true,
            "extra_margin": m * (q_true - q_attr),
            "extra_spend": s_true - s_attr}


# ---------------------------------------------------------------------------
# 2. 模拟「被助攻的乌龙球」：归因口径 vs 真实口径（论文 §6 panel a 的数据生成过程）
# ---------------------------------------------------------------------------
def simulate_attributed_ratio(tau: float, phi: float, n_intents: int = 200_000,
                              seed: int = 0) -> float:
    """模拟「归因口径能看到的转化 / 真实增量」这一比值。

    生成过程与论文 §6 一致：每一条 intent 都是真实增量购买；成交以概率 τ 记在
    marketplace 上；站内成交的匹配率为 1，跨平台成交的回收率为 φ。
    返回的是**本卡模拟值**，不是论文的报告值。
    """
    rng = np.random.default_rng(seed)
    booked_on_marketplace = rng.random(n_intents) < tau
    recovered = rng.random(n_intents) < phi
    observed = (~booked_on_marketplace) | (booked_on_marketplace & recovered)
    return float(observed.mean())


def simulate_channel_complete_experiment(n_per_arm: int = 200_000,
                                         baseline_rate: float = 0.02,
                                         true_lift: float = 0.015,
                                         tau: float = 0.7, phi: float = 0.1,
                                         seed: int = 11) -> dict:
    """一个受众级 ITT 实验，结果同时按两种口径保存。

      y      —— **渠道完整**结果：无论在 G 还是 R 成交都算（论文 §5.2）；
      y_attr —— 归因口径：只看到站内成交，以及 φ 比例被回收的跨平台成交。
    true_lift 是「每被分配 1 个用户新增的转化概率」，0.015 即每 1,000 人 15 个转化；
    τ、φ、baseline_rate 取论文 §6 panel (c) 的模拟设定值（不是实测参数）。
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm
    z = np.repeat([0, 1], n_per_arm)                      # 随机分配（分配早于曝光）
    y0 = rng.random(n) < baseline_rate                    # 未触达时的结果
    y1 = rng.random(n) < (baseline_rate + true_lift)      # 触达时的结果
    y = np.where(z == 1, y1, y0)
    booked_on_marketplace = rng.random(n) < tau
    recovered = rng.random(n) < phi
    observable = (~booked_on_marketplace) | (booked_on_marketplace & recovered)
    y_attr = (y & observable).astype(float)
    return {"z": z, "y": y.astype(float), "y_attr": y_attr,
            "n_per_arm": n_per_arm, "tau": tau, "phi": phi,
            "kappa": observability_factor(tau, phi)}


def itt_estimate(y: np.ndarray, z: np.ndarray) -> dict:
    """论文式 (9)：Δ^ITT = E[Y|Z=1] − E[Y|Z=0]，由随机化识别，不需要任何模型。"""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=int)
    t, c = y[z == 1], y[z == 0]
    delta = float(t.mean() - c.mean())
    se = math.sqrt(float(t.var(ddof=1)) / t.size + float(c.var(ddof=1)) / c.size)
    return {"delta": delta, "se": se,
            "ci95": (delta - 1.96 * se, delta + 1.96 * se),
            "treat_mean": float(t.mean()), "control_mean": float(c.mean())}


# ---------------------------------------------------------------------------
# 3. MMM 的再记功：市场级回归把被转移的需求记到收割方头上（论文 §4.2 / §6 panel c）
# ---------------------------------------------------------------------------
def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """最小二乘（截距列由调用方自行加）。"""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def simulate_market_cells(n_cells: int = 600, gamma: float = 1.0,
                          diverted_share: float = 0.5, seed: int = 3) -> dict:
    """市场级（channel × week）面板的简化模拟。

    设定（与论文 §4.2 的机制一致，参数由本卡自行设定）：
      assigned_reach —— 发起平台的分配触达（外生，由预算决定）；
      diverted       —— 该市场被转移的需求，跨市场独立波动（信任水平不同）；
      retailer_spend —— 收割方的站内赞助花费，**内生地跟随到店需求**：γ·diverted + 噪声；
      总销量         —— 由 assigned_reach 与 diverted 共同构成，收割方的广告本身不创造任何东西。
    因此把总销量回归到 (assigned_reach, retailer_spend) 时，收割方花费的系数为正 ——
    这正是论文说的「MMM 看得到总量，却把功劳再记给收割方」。
    """
    rng = np.random.default_rng(seed)
    assigned_reach = rng.normal(10.0, 2.0, n_cells)
    diverted = diverted_share * np.abs(rng.normal(5.0, 2.0, n_cells))
    retailer_spend = gamma * diverted + rng.normal(0.0, 0.5, n_cells)
    total_sales = 2.0 * assigned_reach + 1.0 * diverted + rng.normal(0.0, 1.0, n_cells)
    return {"assigned_reach": assigned_reach, "diverted": diverted,
            "retailer_spend": retailer_spend, "total_sales": total_sales}


def mmm_regression(cells: dict) -> dict:
    """把总销量回归到「发起平台分配触达 + 收割方赞助花费」上（论文 §6 panel c 的做法）。"""
    X = np.column_stack([np.ones(len(cells["total_sales"])),
                         cells["assigned_reach"], cells["retailer_spend"]])
    coef = ols(cells["total_sales"], X)
    return {"intercept": float(coef[0]), "coef_generator": float(coef[1]),
            "coef_retailer": float(coef[2])}


# ---------------------------------------------------------------------------
# 4. 个体级增量映射：把 PIE 从 campaign 粒推到 individual 粒（论文 §5.3 式 (10)）
# ---------------------------------------------------------------------------
def simulate_individual_records(n: int = 60_000, seed: int = 5) -> dict:
    """个体级记录 (X, Z, Y)。合成设定：

    δ(x) = 0.01 + 0.02·x1 − 0.01·x2  —— 增量效应按分配前特征异质（本卡自设）；
    baseline(x) = 0.03 + 0.01·x1     —— 预后型特征（高响应人群未必高增量）；
    Y 是渠道完整结果（在线性概率模型下生成，便于闭式检验）。
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2])
    z = (rng.random(n) < 0.5).astype(int)
    delta = 0.01 + 0.02 * x1 - 0.01 * x2
    baseline = 0.03 + 0.01 * x1
    p = np.clip(baseline + delta * z, 1e-4, 1.0 - 1e-4)
    y = (rng.random(n) < p).astype(float)
    return {"X": X, "z": z, "y": y, "delta": delta}


def fit_individual_incrementality(X: np.ndarray, z: np.ndarray,
                                  y: np.ndarray, ridge: float = 1e-8) -> dict:
    """T-learner：分别拟合 E[Y|Z=1,X] 与 E[Y|Z=0,X]，其差即 δ̂(x)（论文式 (10)）。

    δ(x) = E[Y|Z=1, X=x] − E[Y|Z=0, X=x]，用实验库里的 (X, Z, Y) 训练；
    这里用带极小 ridge 的线性概率模型，便于闭式求解与小样本自检。
    """
    def _fit(mask: np.ndarray) -> np.ndarray:
        Xd = np.hstack([np.ones((int(mask.sum()), 1)), X[mask]])
        A = Xd.T @ Xd + ridge * np.eye(Xd.shape[1])
        return np.linalg.solve(A, Xd.T @ y[mask])

    b1, b0 = _fit(z == 1), _fit(z == 0)
    return {"b_treat": b1, "b_control": b0}


def predict_delta(X: np.ndarray, model: dict) -> np.ndarray:
    """把拟合好的映射投射到任意受众（论文 §5.3 的 projection onto campaigns without holdouts）。"""
    Xd = np.hstack([np.ones((X.shape[0], 1)), X])
    return Xd @ (model["b_treat"] - model["b_control"])
```

```python
# ---------------------------------------------------------------------------
# 5. 测试
# ---------------------------------------------------------------------------
def test_observability_factor_edges():
    # τ=0（无 distrust）或 φ=1（完美回传）时，归因口径不吃亏
    assert abs(observability_factor(0.0, 0.0) - 1.0) < 1e-12
    assert abs(observability_factor(0.9, 1.0) - 1.0) < 1e-12
    assert abs(observability_factor(0.75, 0.0) - 0.25) < 1e-12
    # κ 关于 τ 递减、关于 φ 递增（Proposition 1）
    assert observability_factor(0.8, 0.2) < observability_factor(0.4, 0.2)
    assert observability_factor(0.4, 0.8) > observability_factor(0.4, 0.2)


def test_proposition2_underinvestment():
    # 只要 τ>0 且 φ<1，归因制投入严格低于 first-best（Proposition 2）
    for tau in (0.25, 0.5, 0.75):
        for phi in (0.0, 0.3):
            assert spend_ratio_vs_first_best(tau, phi, 0.5) < 1.0
    # 与闭式解一致：s_attr/s_true == κ 的 1/(1−β) 次方
    m, alpha, beta, lam = 0.35, 120.0, 0.5, 2.0
    s_attr = spend_from_roas_threshold(m, alpha, beta, lam,
                                       observability_factor(0.75, 0.0))
    s_true = spend_from_roas_threshold(m, alpha, beta, lam, 1.0)
    assert abs(s_attr / s_true - spend_ratio_vs_first_best(0.75, 0.0, beta)) < 1e-9
    # 少投 → 少生成需求 → 毛利缺口为正
    gap = margin_loss_from_mismeasurement(m, alpha, beta, lam, 0.75, 0.0)
    assert gap["extra_margin"] > 0.0 and gap["s_true"] > gap["s_attr"]


def test_simulated_ratio_tracks_kappa():
    # 模拟出的「归因/真实」比应收敛到 κ（论文 §6 panel a 的结论；这里是本卡自跑的模拟）
    for tau, phi in ((0.75, 0.0), (0.5, 0.3), (0.25, 0.0)):
        est = simulate_attributed_ratio(tau, phi, n_intents=200_000, seed=1)
        assert abs(est - observability_factor(tau, phi)) < 0.01


def test_channel_complete_itt_is_unbiased_but_attribution_is_not():
    exp = simulate_channel_complete_experiment()
    itt = itt_estimate(exp["y"], exp["z"])
    attr = itt_estimate(exp["y_attr"], exp["z"])
    # 渠道完整 ITT 落在真值（每 1,000 人 15 个转化 = 0.015）的抽样误差内
    assert abs(itt["delta"] - 0.015) < 4 * itt["se"]
    # 归因口径系统性低估，且低估幅度约等于 κ
    kappa = exp["kappa"]
    assert attr["delta"] < itt["delta"]
    assert abs(attr["delta"] / itt["delta"] - kappa) < 0.08


def test_mmm_recredits_diverted_demand_to_the_harvester():
    cells = simulate_market_cells()
    fit = mmm_regression(cells)
    # 收割方的赞助花费不创造任何东西，但回归系数为正 —— 论文 §4.2 的「再记功」
    assert fit["coef_retailer"] > 0.0
    assert fit["coef_generator"] > 0.0


def test_individual_map_recovers_heterogeneity_and_nests_campaign_grain():
    rec = simulate_individual_records()
    model = fit_individual_incrementality(rec["X"], rec["z"], rec["y"])
    d_hat = predict_delta(rec["X"], model)
    d_true = rec["delta"]
    # 逐个体排序：学到的映射与真实异质效应同向
    assert float(np.corrcoef(d_hat, d_true)[0, 1]) > 0.5
    # 均值回到 campaign 粒（论文 §5.3：个体粒嵌套 campaign 粒）
    assert abs(float(d_hat.mean()) - float(d_true.mean())) < 0.005
    itt = itt_estimate(rec["y"], rec["z"])
    assert abs(float(d_hat.mean()) - itt["delta"]) < 4 * itt["se"] + 0.002
    # 按 δ̂ 排序能挑出高增量人群（这就是「个体粒多出来的能力」）
    top = np.argsort(-d_hat)[: len(d_hat) // 10]
    assert float(d_true[top].mean()) > float(d_true.mean())


# ---------------------------------------------------------------------------
# 6. 业务演示
# ---------------------------------------------------------------------------
def _business_demo() -> None:
    print("== A. 归因楔子（解析式 κ = 1 − τ(1−φ)，无实测数据）==")
    for tau in (0.25, 0.5, 0.75):
        for phi in (0.0, 0.3):
            k = observability_factor(tau, phi)
            print(f"  τ={tau:.2f} φ={phi:.1f} → κ={k:.4f}"
                  f" | 归因制投入 / first-best = "
                  f"{spend_ratio_vs_first_best(tau, phi, 0.5):.6f}")
    print("  [模拟核对] τ=0.75 φ=0 时，200,000 条合成 intents 得到归因/真实比 = "
          f"{simulate_attributed_ratio(0.75, 0.0, 200_000, seed=1):.4f}"
          "（本卡合成数据，非论文输出）")

    exp = simulate_channel_complete_experiment()
    itt = itt_estimate(exp["y"], exp["z"])
    attr = itt_estimate(exp["y_attr"], exp["z"])
    print("\n== B. 同一个实验，两种口径（本卡合成数据：τ=0.7, φ=0.1, 每组 200,000 人）==")
    print(f"  渠道完整 ITT : {itt['delta'] * 1000:.2f} ± {1.96 * itt['se'] * 1000:.2f} "
          "转化/千人（95% CI）")
    print(f"  归因口径     : {attr['delta'] * 1000:.2f} ± {1.96 * attr['se'] * 1000:.2f} "
          f"转化/千人 —— 比值 {attr['delta'] / itt['delta']:.3f}"
          f"（κ={exp['kappa']:.3f}）")

    fit = mmm_regression(simulate_market_cells())
    print("\n== C. 市场级 MMM 的再记功（本卡合成数据，600 个 market cells）==")
    print(f"  系数：发起平台分配触达 = {fit['coef_generator']:.3f}，"
          f"收割方赞助花费 = {fit['coef_retailer']:.3f}"
          "（收割方广告本身不创造任何需求）")

    rec = simulate_individual_records()
    model = fit_individual_incrementality(rec["X"], rec["z"], rec["y"])
    d_hat = predict_delta(rec["X"], model)
    top = np.argsort(-d_hat)[: len(d_hat) // 10]
    print("\n== D. 个体级增量映射（本卡合成数据，60,000 条个体记录）==")
    print(f"  全体平均 δ̂ = {d_hat.mean():.4f}，δ̂ 最高的那 10% 人群真实平均增量 = "
          f"{rec['delta'][top].mean():.4f}（全体真实平均 {rec['delta'].mean():.4f}）")


if __name__ == "__main__":
    _business_demo()
```

**本卡代码的真实输出**（B 类·可复现；下列数值全部来自上面这段代码在本机跑出的结果，不是论文数据）：

```text
== A. 归因楔子（解析式 κ = 1 − τ(1−φ)，无实测数据）==
  τ=0.25 φ=0.0 → κ=0.7500 | 归因制投入 / first-best = 0.562500
  τ=0.25 φ=0.3 → κ=0.8250 | 归因制投入 / first-best = 0.680625
  τ=0.50 φ=0.0 → κ=0.5000 | 归因制投入 / first-best = 0.250000
  τ=0.50 φ=0.3 → κ=0.6500 | 归因制投入 / first-best = 0.422500
  τ=0.75 φ=0.0 → κ=0.2500 | 归因制投入 / first-best = 0.062500
  τ=0.75 φ=0.3 → κ=0.4750 | 归因制投入 / first-best = 0.225625
  [模拟核对] τ=0.75 φ=0 时，200,000 条合成 intents 得到归因/真实比 = 0.2505（本卡合成数据，非论文输出）

== B. 同一个实验，两种口径（本卡合成数据：τ=0.7, φ=0.1, 每组 200,000 人）==
  渠道完整 ITT : 16.16 ± 1.02 转化/千人（95% CI）
  归因口径     : 6.25 ± 0.62 转化/千人 —— 比值 0.387（κ=0.370）

== C. 市场级 MMM 的再记功（本卡合成数据，600 个 market cells）==
  系数：发起平台分配触达 = 1.957，收割方赞助花费 = 0.773（收割方广告本身不创造任何需求）

== D. 个体级增量映射（本卡合成数据，60,000 条个体记录）==
  全体平均 δ̂ = 0.0104，δ̂ 最高的那 10% 人群真实平均增量 = 0.0493（全体真实平均 0.0101）
```

> ⚠️ 上面 B 段的渠道完整 ITT 估计，与论文 §6 panel (c) 报告的模拟值（15.4 ± 1.0）**看起来接近，但不可互相印证**：两者用的是同一组模拟设定（τ=0.7、φ=0.1、每组 200,000 人），但随机数流各不相同。**这不构成对论文的复现或验证**，只是同一公式在两个独立模拟里的表现。

---

## ④ 技能关联

- **替代关系｜`Skill-Ad-Attribution-Modeling.md`**（13-广告分析）：Shapley / Markov 归因解决「可见触点之间怎么分功劳」。**逻辑依赖**：本卡说明当转化被移出可见集合时，任何在可见触点上重新加权的方法都拿不回它（⑥ Q11）。正确组合是「归因做日常监控 + 渠道完整 ITT 做校准」，而不是拿归因输出替代增量测量。
- **同域组合｜`Skill-Cannibalization-Corrected-Attribution.md`**（13-广告分析）：那张卡处理「自有渠道之间互相蚕食」造成的归因错配，本卡处理**跨平台**（发起平台 → marketplace）的错配。同一族问题的两个层级：先在站内做蚕食校正，再在站间做渠道完整校正。
- **下游衔接｜`Skill-Causal-Budget-Allocation.md`**（13-广告分析）：本卡产出的真实增量 ROAS / δ̂(x) 正是那张卡预算求解器**必须**的输入——那张卡的 ①b 已经警告「平台归因天然高估，不能直接当 τ̃ 输入」。数据流：本卡（测量校正）→ 那张卡（约束下分配）。
- **对照｜`Skill-Marketing-Mix-Modeling.md`**（15-营销投放分析）：MMM 的总销量口径**看得到**被转移的需求，但收割方花费内生跟随到店需求，会把功劳再记一次；它适合周/季度宏观规划，不能驱动日粒度阈值决策（⑥ Q13）。
- **前置｜`Skill-Uplift-Modeling.md`**（01-因果推断）：式 10 给出的条件 ITT 函数 δ(x) 就是 CATE 估计的一个特例（处理变量是分配 Z 而不是曝光 D）。个体级映射可以直接换成该卡的 S/T/X-Learner，本卡用线性 T-learner 只是为了让代码零依赖。
- **同域定位｜`Skill-User-Funnel-Analysis.md` / `Skill-Cohort-Retention-Analysis.md`**（14-用户分析）：这两张卡解释「漏斗内」与「批次间」的现象；本卡解释漏斗顶端**记不到账**的增量，是它们的前提修正。

---

## ⑤ 商业价值评估

**ROI 公式**（本卡自建；论文**未报告任何真实收益或成本**）

ROI = (Δ增量毛利 − C_sys) / C_sys，其中 Δ增量毛利 = [q(s_true) − q(s_attr)] × m，
q(s) = α·s^β 取自论文式 (1)，s_attr / s_true = κ 的 1/(1−β) 次方取自论文 Proposition 2。

| 参数 | 含义 | 来源 |
|---|---|---|
| κ | 可观测因子 = 1 − τ(1 − φ) | **必须企业自测**：τ 由渠道完整成交台账估、φ 由平台回传覆盖率估。论文 §6 里的 τ、φ 只是**模拟设定值**，不能直接当贵司参数 |
| β | 需求生成函数的规模弹性 | 论文模拟设为 0.5（**模拟设定值**）；企业应用自己的花费-需求曲线拟合，不要照搬 |
| α | 需求生成函数的效率参数 | 企业自有数据拟合；论文未给任何行业的 α |
| m | 增量成交的单位毛利（已扣平台佣金、头程/FBA 履约、退货） | 企业财务口径 |
| λ | ROAS 门槛（hurdle） | 企业自定 |
| C_sys | 一次性建设 + 运行成本：渠道完整结果打通、受众级随机化改造、实验库与个体级模型 | **论文未报告任何成本量级**，需企业自估 |

**可复算的缺口比例**（本卡按论文公式推算，代入的是论文 §6 的**模拟设定值**）

```text
输入（全部来自论文 §6 的模拟设定，不是实测参数）：
  τ = 0.75        φ = 0        β = 0.5
推算（本卡按论文式 (1) 与式 (4)、Proposition 2 自己算的）：
  κ               = 1 − τ(1 − φ)   = 1 − 0.75 × (1 − 0) = 0.25
  s_attr / s_true = κ^(1/(1−β))    = 0.25^2              = 0.0625
  q_attr / q_true = κ^(β/(1−β))    = 0.25^1              = 0.25
推论：在这组模拟设定下，按归因制 ROAS 分配只会投出 first-best 的 6.25% 预算，
      最终只生成 first-best 需求的 25% —— 缺口的 75% 来自「测量口径」而非「投放能力」。
      ⚠️ 这是公式演练：τ、φ、β 未经任何企业数据校准，不得当作收益预测。
```

**保守结论**：本卡**不给金额结论**。收益侧的关键未知量是 κ（即 τ 与 φ，须企业自测）与渠道完整结果的回传覆盖率；论文没有提供任何可用于外推的实测数字——它的 §6 全部是模拟。

- **实施难度**：⭐⭐⭐⭐☆ —— 算法本身很简单（③ 的全部代码只用 numpy），真正的成本在数据工程：把独立站、marketplace、线下成交并到同一个 user/order key，并把「分配」在受众激活时就固化下来。
- **优先级**：⭐⭐⭐⭐☆ —— 如果企业正在用 measured ROAS 决定种草预算的去留，这张卡是**前置修正**：先修口径，再谈优化。若企业只做平台站内（拿不到渠道完整结果），优先级降到 2 星，只做 κ 的诊断。
- **评估依据**：论文的贡献是**口径与识别策略**，不是提升幅度——它一个真实数据都没有。因此本卡的价值主张是「把被系统性低估的渠道救回来」，而不是承诺一个提升率。

---

## ⑥ 原文引用

**A. own goal 的定义与机制**

> 原文："We term the un-credited, demand-generating impression an assist, in the football sense: it set up the goal. But because the assist is invisible to the measurement system in attribution-based systems, it is scored against the originating platform—an own goal."
> 出处：2607.09608 §1 Introduction（fulltext.md L33）｜Q3

> 原文："Hence the assisted own goal hypothesis: the act of successfully generating demand can, through the attribution layer, either not increase, or in some cases even reduce the generating platform’s measured performance and therefore its advertising revenue."
> 出处：2607.09608 §1 Introduction（fulltext.md L33）｜Q4

> 原文："When the purchase occurs on a trusted marketplace that does not pass conversion signals from the originating platform back to the advertiser – and that is true for most inter-marketplace transactions – the conversion is simply not observed by the advertiser’s attribution system."
> 出处：2607.09608 §1 Introduction（fulltext.md L31）｜Q2

> 原文："Because ITT contrasts are computed on channel-complete outcomes, the estimator is unbiased and the own goal disappears."
> 出处：2607.09608 §Abstract（fulltext.md L19）｜Q1

**B. 归因为什么看不见（τ 与 φ 的机制）**

> 原文："Trusted marketplaces characteristically suppress third-party signal or share any data of its referrer (in this case $G$), so empirically $\phi$ is close to $0$."
> 出处：2607.09608 §2.5 The attribution measurement layer（fulltext.md L123）｜Q6

> 原文："$\kappa(\tau,\phi)\;\equiv\;(1-\tau)+\phi\,\tau\;=\;1-\tau(1-\phi)\;\in(0,1].$"
> 出处：2607.09608 §2.5 The attribution measurement layer 式 (4)（fulltext.md L129）｜Q7

> 原文："It is worth pointing out that the failure is not the crediting heuristic (i.e. last- vs. multi-touch), but the observability constraint: while multi-touch or data-driven attribution re-divides credit among observed touchpoints, the own goal removes the conversion from the observable universe. No re-weighting of visible touchpoints can restore credit for this invisible conversion."
> 出处：2607.09608 §4.1 Attribution-based measurement（fulltext.md L203）｜Q11

**C. 理论结果（Proposition 1 / 2 / 3）**

> 原文："Measured ROAS understates true incremental ROAS by exactly the observability factor $1-\tau(1-\phi)$ with equality if and only if either there is no distrust ($\tau=0$) or off-platform purchases are perfectly back-propagated ($\phi=1$). The wedge is strictly increasing in distrust $\tau$ and strictly decreasing in the recovery rate $\phi$."
> 出处：2607.09608 §2.6 Proposition 1（fulltext.md L143）｜Q8

> 原文："Under ROAS-thresholding allocation, the demand-generating platform receives strictly less than first-best spend whenever $\tau>0$ and $\phi<1$:"
> 出处：2607.09608 §2.7 Proposition 2（fulltext.md L157）｜Q9

> 原文："$\frac{s_{G}^{\text{attr}}}{s_{G}^{\text{true}}}=\kappa(\tau,\phi)^{\frac{1}{1-\beta}}=\big(1-\tau(1-\phi)\big)^{\frac{1}{1-\beta}}\;<\;1.$"
> 出处：2607.09608 §2.7 Proposition 2 的比值式（fulltext.md L159）｜Q10

> 原文："Under randomization of $Z_{i}$ and channel-complete outcome measurement, $\Delta^{\text{ITT}}$ identifies the average incremental purchase effect per assigned user and is invariant to the diversion share $\tau$, the recovery rate $\phi$, and the marketplace claim share $\eta$."
> 出处：2607.09608 §5.2 Proposition 3（fulltext.md L243）｜Q18

> 原文："The proposition states the formal sense in which the own goal is a measurement artifact: the same assist that is invisible to attribution is fully recoverable by a channel-complete ITT under a randomized experiment."
> 出处：2607.09608 §5.2（fulltext.md L245）｜Q19

**D. 另外两个测量口径为什么不解决它**

> 原文："First, and most directly tied to our mechanism, the harvester’s spend is endogenous to the generator’s demand: $R$ prices and sells sponsored placements against the arriving intent, so any typical regression model regressing total purchase intent on $s_{R}$ and $S_{G}$ does not know how much of the generated demand to attribute to the harvesting channel—the own goal reappears as simultaneity or multi-colinearity bias rather than signal loss."
> 出处：2607.09608 §4.2 Marketing-mix models（fulltext.md L209）｜Q12

> 原文："Third, MMM resolves channels by week or quarter, not by day, so it cannot drive the thresholding decisions even when its aggregate reading is correct."
> 出处：2607.09608 §4.2 Marketing-mix models（fulltext.md L209）｜Q13

> 原文："Such estimates recover $\mathrm{ROAS}^{\text{true}}_{G}$ and are, by construction, invariant to the diversion share $\tau$: a holdout simply buys less of the product in total, wherever those sales would have occurred."
> 出处：2607.09608 §4.3 Incrementality-based experimentation（fulltext.md L213）｜Q14

**E. 测量方案的两个部件（ambient 随机化 + 个体级扩展）**

> 原文："Assignment is a deterministic hash of the user identifier salted by an audience-specific key: user $i$ is assigned to control in audience $a$ if and only if $h(i,a)\bmod 100<100\,c_{a}$, where $c_{a}$ is the audience’s control percentage."
> 出处：2607.09608 §5.1 Ambient audience-level randomization（fulltext.md L231）｜Q15

> 原文："the outcome measured in the brand’s first-party transaction data—channel-complete by construction, in the sense that it aggregates purchases wherever they are booked: on the generator’s storefront, on the marketplace, or offline."
> 出处：2607.09608 §5.2 Intent-to-treat as the estimand（fulltext.md L235）｜Q16

> 原文："Using assignment rather than exposure avoids conditioning on the ad platform’s endogenous delivery decisions (who saw the ad is algorithmically selected; who was assigned is controled by GrowthLoop’s randomization procedure), and matches the advertiser’s decision variable: budget buys assignment, not exposure."
> 出处：2607.09608 §5.2（fulltext.md L239）｜Q17

> 原文："using 2,226 Meta RCTs, it trains a model mapping campaign features—including post-determined aggregates such as exposure rates and last-click conversions, which would be invalid controls in a causal regression but are valid predictors once identification is handled by the experiments—to experiment-identified incrementality, achieving out-of-sample $R^{2}=0.88$ against $R^{2}=0.19$ for seven-day last-click attribution."
> 出处：2607.09608 §5.3（fulltext.md L249，**论文转引 Gordon et al. (2023) 的 campaign 级结果，不是本论文的实验**）｜Q20

> 原文："The feature coefficients can now be projected onto any audience with no holdout given that the audience in question has the same features at both individual- and campaign-level as the initial set of experiments."
> 出处：2607.09608 §5.3 Projection onto campaigns without holdouts（fulltext.md L259）｜Q21

> 原文："Assigning converters randomly after the fact is independent of treatment by construction and dilutes the intent-to-treat effect toward zero; assigning by observed exposure conditions on the platform’s endogenous delivery reproduces the attribution bias. Assignment must precede exposure."
> 出处：2607.09608 §5.4 Acquisition advertising and the enumerability constraint（fulltext.md L263）｜Q22

> 原文："Note that platform-side lift studies delegate individual-level randomization to the party that can enumerate at auction time—though their platform-observed outcomes are not channel-complete, so the own goal survives inside the lift test itself."
> 出处：2607.09608 §5.4（fulltext.md L265）｜Q23

**F. 模拟研究（§6）—— 论文**没有**真实数据，本段全部是模拟设定与模拟输出**

> 原文："We outline a simulation design under which the attribution bias, the induced revenue loss, and the accuracy of the individual-level model can be quantified (Section 6)."
> 出处：2607.09608 §1 Contributions（fulltext.md L79）｜Q5

> 原文："Consumer journeys are drawn from the data-generating process of Section 2: spend creates intent ($\beta=0.5$), each intent converts on the marketplace with probability $\tau$ and on the generator’s storefront with probability $1-\tau$, and the attribution layer matches on-platform purchases perfectly but recovers diverted purchases only with recovery rate $\varphi$."
> 出处：2607.09608 §6 Simulation Study（fulltext.md L273）｜Q24

> 原文："For 200,000 simulated intents per cell we compute the ratio of attributed to true conversions across $\tau\in\{0,.25,.5,.75,.95\}$ and $\varphi\in\{0,0.3\}$. The simulated ratios (circles in panel a) mimic $\kappa=1-\tau(1-\varphi)$ to three decimal places: at $\tau=0.75$ with no signal recovery, the generator is credited with exactly one quarter of the conversions it caused."
> 出处：2607.09608 §6 The wedge（fulltext.md L277）｜Q25

> 原文："First-best spend grows with $\alpha$ throughout, but attributed revenue peaks at $\alpha\approx 1.7$ and then declines: by $\alpha=2.6$ the generator has lost nearly half of its peak revenue while being more than twice as effective as at baseline."
> 出处：2607.09608 §6 The backfire（fulltext.md L281）｜Q26

> 原文："Finally we run the ambient experiment of Section 5 at $\tau=0.7$, $\varphi=0.1$: 200,000 users per arm, a 2% baseline conversion rate on channel-complete outcomes, and a true incremental effect of 15 conversions per 1,000 assigned users."
> 出处：2607.09608 §6 Recovery（fulltext.md L285）｜Q27

> 原文："The ITT contrast estimates $15.4\pm 1.0$—the truth, within sampling error—while last-touch attribution reports $5.5$, which is $\kappa\times$truth to the decimal (panel c)."
> 出处：2607.09608 §6 Recovery（fulltext.md L285）｜Q28

> 原文："Between the two sits the MMM benchmark: aggregating the same journeys to 600 market-level cells and regressing total sales on assigned reach and the retailer’s sponsored spend—which endogenously tracks arriving demand—yields $9.7\pm 2.4$, with the shortfall re-credited to $R$: the coefficient on $R$’s spend is positive even though $R$’s advertising causes nothing. In other words, MMM sees the diverted demand in aggregate but falsely credits part of it to the harvester, exactly as Section 4 anticipates."
> 出处：2607.09608 §6 Recovery（fulltext.md L285）｜Q29

**G. 论文自承局限（§7 Limitations）**

> 原文："The model is deliberately parsimonious: a single product, a single generator and retailer, a static one-shot allocation, and a reduced-form trust parameter."
> 出处：2607.09608 §7 Limitations（fulltext.md L303）｜Q30

> 原文："The measurement model of Section 5 adds its own assumptions: channel-complete first-party outcomes, negligible cross-experiment interaction (Section 5.1), and transportability of conditional effects to un-experimented audiences; non-changing addressable audiences."
> 出处：2607.09608 §7 Limitations（fulltext.md L303）｜Q31
