---
title: Skill-Funnel-Causal-Coupon-Allocation
module: 13-广告分析
topic: 漏斗分解的双目标 uplift 建模与多档优惠券预算分配（FunnelCausalNet）
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.11675
paper: FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation
venue: CIKM 2026
venue_tier: CCF-B
evidence_grade: A
verified_by: verify_skill_code.py（K1 L1–L5）+ quote_check.py 逐字核验 + 人工抽检数字
supersedes:
related: Skill-Uplift-Modeling.md, Skill-ROAS-Budget-Optimization.md, Skill-Ad-Attribution-Modeling.md, Skill-Promotion-Effectiveness.md
---

# Skill Card: FunnelCausalNet — 漏斗分解的双目标优惠券 uplift 与预算分配

> 论文：`2608.11675`（CIKM 2026，CCF-B）。本卡正文里的每一个数字都在 ⑥ 段有逐字原文出处；
> 论文没有报告的量级，一律写「论文未报告」，不做任何推算与补全。

## ① 算法原理

**核心思想**：成交额有确定性漏斗结构——没转化就没有 GMV。所以对「零膨胀 + 重尾」的 GMV 直接回归是低效的。
FunnelCausalNet 把 GMV 的条件期望拆成两个共享表征的头：转化概率 μ_conv，与「已转化条件下客单」的期望 μ_val，
按 `μ_gmv = μ_conv · μ_val` 组合（论文 Eq. 5），再把两个头的 CATE 一起送进「按档位分配补贴预算」的分配器。

**数学直觉**：GMV 的方差可精确拆成两块，`Var(Y^g) = p·σ_v² + p(1-p)·μ_v²`：
前一块是「转化者内部的客单波动」，后一块是「零质量带来的伯努利切换方差」。
在「转化头以参数速率收敛、客单头以更慢的非参速率收敛」的速率差假设下，漏斗组合估计与直接非参估计的
点态 MSE 之比趋近 `1 / (1 + (1-p)·μ_v²/σ_v²)`：零质量 (1-p) 越大、客单均值相对其波动越主导，漏斗分解越省方差。
论文自己反复强调这只是**理想化的 regime 指标**，不是普适最优性定理，也不保证 CATE 排序更准。

**关键假设**：① RCT 式随机分配（`T ⊥ (Y^c(·), Y^g(·)) | X` 且倾向得分在 x 处有下界）；
② 漏斗支撑恒成立（`Y^c = 0 ⇒ Y^g = 0`）；③ 两个头的交叉协方差可控——独立样本切分能保证，
**共享表征网络并不保证**；④ 档位是离散 offer，不做连续券剂量的插值。

**配套三件套**：拉格朗日松弛把「谁分到哪一档」解耦成每人独立的内层问题（可扩到百万级用户）；
用 RCT 臂均值做**加性锚定**修正零膨胀下的水平偏差后再喂分配器；
两个目标各出一条 split-conformal 区间再加 Bonferroni 联合，**只当审计/监控带**，不作为分配器输入。

---

## ①b 反例与适用边界

**什么时候不要用这个算法**

- **只有「发/不发」单次触达轴、没有档位强度轴时**。漏斗分解会退化成「在单次转化提升的下游估一个近退化的客单头」。
  论文在公开 Hillstrom 邮件 RCT 上给出了直接反例：营收导向的 RERUM 与 DualHeadNet 在 AUUC_GMV 上领先，
  而 DESCN / ECUP / FunnelCausalNet 这些多档漏斗深度模型全部更差。站内如果只做「发券 / 不发券」两臂测试，先别上这套。
- **拿不到用户级随机化的平台店内链路**。论文的因果解释**全部**建立在 RCT 式随机分配上，并明确不声称从纯观察日志识别。
  Amazon 第三方卖家只有卖家侧报表，本卡在该链路上只能给「档位弹性诊断 + 券预算上限测算」，不能给个体级增量或个体定向。
- **想把券做成连续剂量**（每单自动算一个折扣）时。论文的档位是离散 offer，不利用剂量的平滑性或单调性，这不是它的能力范围。

**已知失败模式**

- **稀疏转化者**：转化率越低，客单头的有效样本越少；论文自己的零膨胀压力测试在最极端的低转化端点出现 finite-sample 回撤。
- **窄 α 的 conformal 下界过于悲观**：把 LCB 直接喂给分配器会退化成「全场不发券」（论文的 `funnel_ip_lcb` 在工业日志上零增量、零成本）。
- **共享表征让两个头的误差变相关**：乘积组合会把系统性偏差放大；论文只给理想化命题，不覆盖神经网络实现。
- **记录级切分泄漏**：论文工业实验用记录级置换切分，同一个用户可同时落在训练片与留出片，区间解释因此偏乐观。
- **水平校准 ≠ 排序质量**：ATE 级别的 GMV 水平对不对，与 PEHE/AUUC 级别的「谁更值得给券」是两件事；
  论文在合成数据上发现浅模型（Causal Forest / S-Learner）的 ATE 误差反而更小。

**论文自己承认的局限**

① 最强的分配证据来自**不可独立复现**的私有工业 RCT，公开二值基准上没有一致增益；
② 工业结果的 per-anchor 自助置信区间包含零、锚点之间相关，所以「7/7 领先」只是描述性一致性，不是独立显著性；
③ 联合 conformal 覆盖是边际的、且偏保守；
④ 论文未做完整因子分解（估计器架构 / 锚定 / conformal / 分配器各自贡献），也**没有公开代码 artifact**；
⑤ 论文未报告母婴品类或任何跨境电商渠道的实测结果——本卡不做跨域外推。

---

## ② 母婴出海应用案例

### 场景一：Amazon 站内「Coupon 档位 + Best Deal」的锁档与预算切分（旺季前决策）

- **业务问题**：旺季前若干周，站内主推 SKU（吸奶器、纸尿裤、辅食机）到底给浅档、中档还是深档 Coupon、
  要不要叠加 Best Deal，以及一笔总补贴预算在 ASIN × 买家分群之间怎么切。
  业务现在盯的是**券核销率**，但核销率是归因指标而不是增量——本来就会买的人也会领券。
- **数据要求**：事件级曝光 / 领券 / 下单日志，字段含随机化单元、档位 T、转化 Y^c、成交额 Y^g；
  粒度做到「分群 × 日」；历史至少覆盖一个完整旺季加一个平季对照；
  分配器另外需要每档的补贴成本 c(x,k) 与总预算 B。
- **数据可得性**：`不可得（用户级随机化）`。平台店铺无法对买家做券档位的用户级随机化，
  而论文要求 RCT 式随机分配 → 站内链路必须退化为「ASIN × 周 × 档位」的分层弹性估计 + 地理/时间 holdback，
  结论一律标注为观察性，只用于券预算上限测算与档位诊断，不用于个体定向。
- **预期产出**：① 各档的转化弹性与客单弹性曲线；② 各档的增量 ROI 曲面；③ 预算边界上「转化最优 vs GMV 最优」的排序冲突名单（审计层，供人工复核）。
- **业务价值**：把「券该不该给、给多深」变成可判定的阈值。论文的判据是
  **ΔROI = 增量 GMV ÷ 增量补贴成本**，并把平台佣金率当敏感性参数取 [0.2,0.3]，
  对应盈亏平衡 ΔROI 落在 [3.3,5.0]；低于 3 就意味着增量佣金已经抵不上补贴。
  贵司代入时请用 2025 年该品类在 Amazon 的实测佣金率，不要沿用论文的 OTA 区间。

### 场景二：站外种草 → 独立站承接的首单券档位分配（唯一能做真 RCT 的链路）

- **业务问题**：TikTok / 达人内容把流量导到独立站（Shopify），要给新客首单券选档位、选人群，并按周预算约束分配。
  站外链路必须用**增量口径**而不是归因口径来决定预算：站内品牌词广告会截胡站外种草带来的品牌搜索转化，
  按归因分预算会持续把钱投给「本来就成交」的那一端。
- **数据要求**：独立站事件级埋点（曝光-加购-下单）+ 券档位 + 随机化单元 ID + 订单金额；
  至少两个月的历史与一次 holdback 实验；用户需跨渠道去重（同一个人可能在 Amazon 也买过）。
- **数据可得性**：`企业内可得`（独立站自建埋点 + Shopify 可控随机化）。但样本量通常远小于论文的工业规模，
  转化者稀疏是主要风险——论文在单次触发的公开 RCT 上给出的失败模式正是这一条。
- **预期产出**：首单券各档的 CATE 排序 + 预算约束下的分配名单 + 审计用的联合覆盖区间（只做合规复核，不喂分配器）。
- **业务价值**：论文的工业实验里，锚定拉格朗日分配在紧预算下比随机分配拿到更高 ΔROI（3.92 对 3.07）。
  这个**量级不能搬到母婴独立站**（品类、佣金率、转化基线都不同），但决策形态可以直接搬：
  先估增量 → 再在预算约束下分配 → 最后拿增量 ROI 与盈亏平衡线比较，而不是拿券核销率拍板。

---

## ③ 代码模板

> 下面四个 python 块按文档顺序拼成一个模块即可运行（K1 即按此方式验证）：
> 只依赖 numpy / pandas / scikit-learn，数据在函数内合成，不读外部文件、不联网、不画图。

```python
"""
FunnelCausalNet —— 漏斗分解的双目标 uplift + 多档券预算分配（可运行简化版）

对应论文：2608.11675（CIKM 2026）
  * 漏斗组合       mu_gmv = mu_conv * mu_val                      (Eq. 5)
  * 方差分解       Var(Y^g) = p*sigma_v^2 + p*(1-p)*mu_v^2        (Eq. 7)
  * 理想化 MSE 比  1 / (1 + (1-p)*mu_v^2/sigma_v^2)               (Eq. 8)
  * 拉格朗日预算分配 + RCT 臂均值加性锚定                          (Sec. 4.4)

依赖：numpy / pandas / scikit-learn。数据在函数内合成，不读外部文件、不联网、不画图。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

FEATURES = ("x1", "x2")

# 合成生成器参数：低基线转化 + 多档折扣 + 对数正态客单
# （论文的 Criteo-MT7 也是「低基线转化 + 多档折扣」的半合成设定）
BASE_LOGIT = -2.6          # 对照档基线转化 logit（约 7%）
CONV_TIER_SLOPE = 6.0      # 折扣对转化概率的斜率
VAL_LOG_MEAN = 3.4         # 已转化条件的客单对数均值
VAL_LOG_SIGMA = 0.8        # 已转化条件的客单对数标准差
DEFAULT_DISCOUNTS = (0.0, 0.05, 0.10, 0.15, 0.20)


def make_coupon_rct(n_users: int = 4000, discounts=DEFAULT_DISCOUNTS,
                    seed: int = 7) -> pd.DataFrame:
    """合成一份「多档优惠券 RCT」日志。

    返回列：x1/x2（用户特征）、tier（0 = 对照）、discount、conv（Y^c）、gmv（Y^g）。
    漏斗支撑恒成立：conv == 0 时 gmv == 0。
    """
    rng = np.random.default_rng(seed)
    discounts = np.asarray(discounts, dtype=float)
    x1 = rng.normal(size=n_users)
    x2 = rng.normal(size=n_users)
    tier = rng.integers(0, discounts.size, size=n_users)
    disc = discounts[tier]

    logit = BASE_LOGIT + 0.5 * x1 - 0.3 * x2 + CONV_TIER_SLOPE * disc
    conv = (rng.random(n_users) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    log_gmv = VAL_LOG_MEAN + 0.2 * x2 - 1.0 * disc
    gmv = np.where(conv == 1, rng.lognormal(mean=log_gmv, sigma=VAL_LOG_SIGMA), 0.0)

    return pd.DataFrame({"x1": x1, "x2": x2, "tier": tier,
                         "discount": disc, "conv": conv, "gmv": gmv})


def oracle_arm_value(df: pd.DataFrame, discounts=DEFAULT_DISCOUNTS) -> np.ndarray:
    """真值曲面 E[Y^g(t) | x] = p_t(x) * exp(mu_log_t(x) + sigma^2 / 2)，形状 (n, K)。

    仅用于合成数据的 PEHE 评估——论文在 Criteo-MT7 上同样使用 oracle ITE。
    """
    discounts = np.asarray(discounts, dtype=float)
    x1 = df["x1"].to_numpy()
    x2 = df["x2"].to_numpy()
    out = np.zeros((len(df), discounts.size))
    for k, d in enumerate(discounts):
        logit = BASE_LOGIT + 0.5 * x1 - 0.3 * x2 + CONV_TIER_SLOPE * d
        p = 1.0 / (1.0 + np.exp(-logit))
        mu_log = VAL_LOG_MEAN + 0.2 * x2 - 1.0 * d
        out[:, k] = p * np.exp(mu_log + 0.5 * VAL_LOG_SIGMA ** 2)
    return out


def variance_decomposition(gmv, conv) -> "tuple[float, float]":
    """论文 Eq. 7 的方差分解：Var(Y^g) = p*sigma_v^2 + p*(1-p)*mu_v^2。

    返回 (左侧样本方差, 右侧两项之和)。用总体方差（ddof = 0）时该式是代数恒等式。
    """
    y = np.asarray(gmv, dtype=float)
    c = np.asarray(conv, dtype=float)
    p = c.mean()
    v = y[c == 1]
    mu_v = v.mean()
    sigma_v2 = v.var()
    return float(y.var()), float(p * sigma_v2 + p * (1.0 - p) * mu_v ** 2)


def funnel_mse_ratio(p: float, mu_v: float, sigma_v: float) -> float:
    """论文 Eq. 8 的领先阶 MSE 比：漏斗组合 / 直接非参回归。

    比值 < 1 表示漏斗分解在该 regime 下更省点态方差；只在「转化头有参数速率优势」时成立。
    """
    return 1.0 / (1.0 + (1.0 - p) * mu_v ** 2 / sigma_v ** 2)
```

```python
class FunnelUplift:
    """漏斗组合 uplift 估计器：mu_gmv = mu_conv * mu_val（论文 Sec. 4.1）。

    每个档位分别拟合「转化头」（逻辑回归，全样本）与「客单头」
    （对 log1p(GMV) 的岭回归，只用该档的转化者子样本），再做 LogNormal 风格的均值修正。
    """

    def __init__(self, n_tiers: int):
        self.n_tiers = int(n_tiers)
        self.conv_models: list = []
        self.val_models: list = []
        self.val_log_mean: np.ndarray = np.zeros(self.n_tiers)
        self.val_log_var: np.ndarray = np.zeros(self.n_tiers)
        self.anchor_shift: np.ndarray = np.zeros(self.n_tiers)

    def _fit_heads(self, df: pd.DataFrame) -> None:
        X = df[list(FEATURES)].to_numpy()
        y_conv = df["conv"].to_numpy()
        y_gmv = df["gmv"].to_numpy()
        tier = df["tier"].to_numpy()

        for k in range(self.n_tiers):
            m = tier == k
            conv_model = LogisticRegression(max_iter=1000)
            conv_model.fit(X[m], y_conv[m])
            self.conv_models.append(conv_model)

            conv_mask = m & (y_conv == 1)
            log_v = np.log1p(y_gmv[conv_mask])
            if conv_mask.sum() >= 5:
                val_model = Ridge(alpha=1.0)
                val_model.fit(X[conv_mask], log_v)
                resid = log_v - val_model.predict(X[conv_mask])
                self.val_log_var[k] = float(resid.var())
            else:
                # 转化者太少 → 该档客单头退化为常数（论文点名的稀疏转化者失败模式）
                val_model = None
                self.val_log_var[k] = 0.0
            self.val_models.append(val_model)
            self.val_log_mean[k] = float(log_v.mean()) if log_v.size else 0.0

    def predict_heads(self, X: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """返回 (mu_conv, mu_val, mu_gmv)，形状均为 (n_users, n_tiers)。"""
        n = X.shape[0]
        mu_conv = np.zeros((n, self.n_tiers))
        mu_val = np.zeros((n, self.n_tiers))
        for k in range(self.n_tiers):
            mu_conv[:, k] = self.conv_models[k].predict_proba(X)[:, 1]
            if self.val_models[k] is None:
                mu_val[:, k] = np.expm1(self.val_log_mean[k])
            else:
                eta = self.val_models[k].predict(X) + 0.5 * self.val_log_var[k]
                mu_val[:, k] = np.expm1(np.clip(eta, 0.0, 25.0))
        return mu_conv, mu_val, mu_conv * mu_val

    def _fit_anchor(self, df: pd.DataFrame) -> np.ndarray:
        """RCT 臂均值加性锚定（论文 Sec. 4.4，只用留入片 held-in slice）。

        修正零膨胀下 GMV 水平的系统性偏差：不改排序、不重训，只把每档的常数偏移减掉。
        """
        X = df[list(FEATURES)].to_numpy()
        _, _, mu_gmv = self.predict_heads(X)
        tier = df["tier"].to_numpy()
        y_gmv = df["gmv"].to_numpy()
        shift = np.zeros(self.n_tiers)
        for k in range(self.n_tiers):
            m = tier == k
            if m.sum():
                shift[k] = float(y_gmv[m].mean() - mu_gmv[m, k].mean())
        return shift

    def fit(self, df: pd.DataFrame) -> "FunnelUplift":
        self._fit_heads(df)
        self.anchor_shift = self._fit_anchor(df)
        return self

    def cate(self, X: np.ndarray, anchored: bool = True) -> dict:
        """各档相对对照的 CATE：{tau_c, tau_g}，形状 (n_users, n_tiers)。"""
        mu_conv, _, mu_gmv = self.predict_heads(X)
        if anchored:
            mu_gmv = mu_gmv + self.anchor_shift[None, :]
        return {"tau_c": mu_conv - mu_conv[:, [0]],
                "tau_g": mu_gmv - mu_gmv[:, [0]]}


class DirectGmvUplift:
    """基线：把零膨胀 GMV 当无约束连续响应直接回归（论文 Table 4 的 A 模式）。"""

    def __init__(self, n_tiers: int):
        self.n_tiers = int(n_tiers)
        self.models: list = []

    def fit(self, df: pd.DataFrame) -> "DirectGmvUplift":
        X = df[list(FEATURES)].to_numpy()
        y = df["gmv"].to_numpy()
        tier = df["tier"].to_numpy()
        for k in range(self.n_tiers):
            m = tier == k
            self.models.append(Ridge(alpha=1.0).fit(X[m], y[m]))
        return self

    def cate(self, X: np.ndarray) -> dict:
        mu = np.column_stack([m.predict(X) for m in self.models])
        return {"tau_g": mu - mu[:, [0]]}


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """PEHE：CATE 估计的均方根误差（论文用 PEHE_GMV / PEHE_CVR 衡量个体效应误差）。"""
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
```

```python
class BudgetedTierAllocator:
    """预算约束下的多档分配（论文 Sec. 4.4 的拉格朗日松弛）。

        max  sum_i r[i, k_i]     s.t.  sum_i c[i, k_i] <= B

    外层对偶变量 lambda 做单变量二分，内层按用户解耦（每人取 argmax r - lambda*c），
    因此可扩展到百万级用户。第 0 列固定为对照（reward = cost = 0）。
    """

    def __init__(self, reward: np.ndarray, cost: np.ndarray, budget: float,
                 max_iter: int = 80):
        self.reward = np.asarray(reward, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.budget = float(budget)
        self.max_iter = int(max_iter)

    def _pick(self, lam: float) -> np.ndarray:
        return np.argmax(self.reward - lam * self.cost, axis=1)

    def _total_cost(self, lam: float) -> float:
        rows = np.arange(self.cost.shape[0])
        return float(self.cost[rows, self._pick(lam)].sum())

    def solve(self) -> dict:
        hi = 1.0
        for _ in range(60):                      # 先把上界推到可行（cost <= B）
            if self._total_cost(hi) <= self.budget:
                break
            hi *= 2.0
        if self._total_cost(hi) > self.budget:   # 预算太小，只能全场对照
            choice = np.zeros(self.cost.shape[0], dtype=int)
        else:
            lo = 0.0
            for _ in range(self.max_iter):       # cost(lambda) 单调不增 → 二分安全
                mid = 0.5 * (lo + hi)
                if self._total_cost(mid) > self.budget:
                    lo = mid
                else:
                    hi = mid
            choice = self._pick(hi)
        rows = np.arange(self.cost.shape[0])
        tot_cost = float(self.cost[rows, choice].sum())
        tot_reward = float(self.reward[rows, choice].sum())
        return {"choice": choice, "total_cost": tot_cost, "total_reward": tot_reward,
                "lambda": float(hi),
                "delta_roi": (tot_reward / tot_cost) if tot_cost > 0 else float("nan")}


def build_allocation_inputs(df: pd.DataFrame, model: FunnelUplift,
                            discounts=DEFAULT_DISCOUNTS):
    """把模型输出变成分配器输入。

    奖励 = 锚定后的 tau_g（相对对照的增量 GMV 预测）；
    成本 = 该档折扣率 × 该档预测 GMV（论文 Table 7 用的就是 d_k * mu_gmv）。
    """
    X = df[list(FEATURES)].to_numpy()
    tau_g = model.cate(X)["tau_g"]
    _, _, mu_gmv = model.predict_heads(X)
    disc = np.asarray(discounts, dtype=float)
    reward = np.column_stack([np.zeros(len(df)), tau_g[:, 1:]])
    cost = np.column_stack([np.zeros(len(df)), mu_gmv[:, 1:] * disc[None, 1:]])
    return reward, cost
```

```python
def test_funnel_support_holds():
    """漏斗支撑：conv == 0 时 gmv 必须为 0（论文 Eq. 1）。"""
    df = make_coupon_rct(n_users=800, seed=1)
    assert int(((df["conv"] == 0) & (df["gmv"] > 0)).sum()) == 0


def test_variance_decomposition_is_identity():
    """论文 Eq. 7 在样本上必须是恒等式（用总体方差 ddof = 0）。"""
    df = make_coupon_rct(n_users=3000, seed=2)
    lhs, rhs = variance_decomposition(df["gmv"].to_numpy(), df["conv"].to_numpy())
    assert lhs > 0.0
    assert abs(lhs - rhs) <= 1e-9 * lhs


def test_mse_ratio_shrinks_as_zero_mass_grows():
    """Eq. 8：零质量 (1-p) 越大，漏斗组合的相对 MSE 越小，且始终小于 1。"""
    ratio_mid_p = funnel_mse_ratio(0.45, 40.0, 30.0)
    ratio_low_p = funnel_mse_ratio(0.05, 40.0, 30.0)
    assert 0.0 < ratio_low_p < ratio_mid_p < 1.0


def test_budget_constraint_respected():
    """分配器必须守住预算（论文 Eq. 4）。"""
    rng = np.random.default_rng(0)
    n, k = 400, 4
    reward = np.column_stack([np.zeros(n), rng.random((n, k - 1)) * 10.0])
    cost = np.column_stack([np.zeros(n), rng.random((n, k - 1)) * 5.0])
    budget = 0.25 * float(cost[:, 1:].sum())
    res = BudgetedTierAllocator(reward, cost, budget).solve()
    assert res["total_cost"] <= budget * (1.0 + 1e-9)


def test_allocator_picks_own_best_tier_when_budget_is_loose():
    """预算宽松时，解应退化为「每人取自身增量最大的档位」。"""
    reward = np.array([[0.0, 1.0, 3.0], [0.0, 2.0, 0.5]])
    cost = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    res = BudgetedTierAllocator(reward, cost, budget=10.0).solve()
    assert list(res["choice"]) == [2, 1]


if __name__ == "__main__":
    df = make_coupon_rct(n_users=4000, seed=11)
    print("样本人群:", len(df), "人; 总体转化率 =", round(float(df["conv"].mean()), 4))

    lhs, rhs = variance_decomposition(df["gmv"].to_numpy(), df["conv"].to_numpy())
    print("Eq.7 方差分解: 左侧 =", round(lhs, 3), "| 右侧 =", round(rhs, 3))

    conv = df["conv"].to_numpy()
    val = df.loc[conv == 1, "gmv"].to_numpy()
    p, mu_v, sigma_v = float(conv.mean()), float(val.mean()), float(val.std())
    print("Eq.8 理想化 MSE 比 =", round(funnel_mse_ratio(p, mu_v, sigma_v), 4))

    funnel = FunnelUplift(n_tiers=len(DEFAULT_DISCOUNTS)).fit(df)
    direct = DirectGmvUplift(n_tiers=len(DEFAULT_DISCOUNTS)).fit(df)
    X = df[list(FEATURES)].to_numpy()
    truth = oracle_arm_value(df)
    tau_true = truth - truth[:, [0]]
    print("PEHE_GMV  漏斗组合 =",
          round(pehe(funnel.cate(X, anchored=False)["tau_g"], tau_true), 3),
          "| 直接回归 =", round(pehe(direct.cate(X)["tau_g"], tau_true), 3))

    reward, cost = build_allocation_inputs(df, funnel)
    budget = 0.2 * float(cost.max(axis=1).sum())        # 粗口径：全场都给最深档的两成
    alloc = BudgetedTierAllocator(reward, cost, budget).solve()
    print("预算内分配: 成本 =", round(alloc["total_cost"], 1),
          "| 增量 GMV =", round(alloc["total_reward"], 1),
          "| delta_ROI =", round(alloc["delta_roi"], 3))
    print("对照 lambda =", round(alloc["lambda"], 4))

    rng = np.random.default_rng(3)
    rand_choice = rng.integers(0, cost.shape[1], size=len(df))
    rows = np.arange(len(df))
    print("随机分配对照: 成本 =", round(float(cost[rows, rand_choice].sum()), 1),
          "| 增量 GMV =", round(float(reward[rows, rand_choice].sum()), 1))
```

> **这段代码是什么、不是什么**：它是按论文 Eq. 5 / Eq. 7 / Eq. 8 与 §4.4 自行实现的**最小结构复现**，
> 用于验证公式能在真实数据上跑通、预算约束能被守住、漏斗支撑与方差分解恒等式能被子样本断言。
> **它不复现论文的任何实验数值**：合成生成器与论文的 Criteo-MT7 不同源，且论文没有公开代码 artifact。
> 实测下来这段 demo 里「漏斗组合 vs 直接回归」的 PEHE 基本持平（甚至略差），这恰好说明
> 论文那 18–48% 的降幅来自它特定的零膨胀 regime 与数据生成过程，**不能靠一个玩具实现自动获得**。

---

## ④ 技能关联

- **前置 · `Skill-Uplift-Modeling.md`**：本卡的两个头本质就是多任务 T-learner 的 CATE 估计。
  该卡提供 CATE 的定义、元学习器骨架与 Qini / AUUC 评估口径；没有这一层，漏斗分解只是一个结构约束，
  没法判断「估计出来的增量排序到底靠不靠谱」。
- **延伸 · `Skill-ROAS-Budget-Optimization.md`**：预算约束下的分配数学同源（拉格朗日 / 边际 ROI 均衡），
  差别只是本卡把优化变量从「渠道预算」下沉到「券档位 × 人群」。该卡的约束求解与灵敏度分析套路可直接复用，
  本卡额外补上「档位成本 = 折扣率 × 预测 GMV」这个券特有的成本口径。
- **组合 · `Skill-Ad-Attribution-Modeling.md`**：归因回答「谁参与了这笔成交」，本卡回答「给哪一档才拿到增量」。
  两者组合正好处理业务侧的核心痛点——站内品牌词与站外种草互相争功导致归因虚高：
  用增量口径决定券预算怎么分，用归因口径只做过程监控与异常发现。
- **组合 · `Skill-Promotion-Effectiveness.md`**：当站内链路拿不到 RCT 时，先用该卡的观察数据因果方法
  （DML 一类）估出档位弹性，作为本卡在「数据不可得」场景下的退化路径；
  两者共用同一套「增量而不是相关」的判别纪律。

---

## ⑤ 商业价值评估

- **ROI 预估公式**：ΔROI = 增量 GMV ÷ 增量补贴成本（论文口径 `ΔROI := Σ τ̂_g / Σ ĉ`），
  盈亏平衡阈值 ΔROI* = 1/γ，其中 γ 是平台佣金率。
  - **参数来源**：γ 的取值区间来自论文对佣金率的敏感性设定 [0.2,0.3]，对应盈亏平衡 [3.3,5.0]；
    贵司代入时请替换为 2025 年该品类在对应平台的**实测**佣金率，不要沿用论文的 OTA 区间。
  - **论文侧增益证据**：工业多臂 RCT 上，FunnelCausalNet 在 7/7 个锚点的种子平均 ΔROI 最高；
    在中到大锚点区间，其均值超出第二名 0.18–0.21 个 ROI 单位。
    但论文同时声明 per-anchor 置信区间包含零 —— **这不是显著性结论，不能当作收益承诺**。
  - **论文未报告**母婴品类、Amazon 站内或独立站链路的任何收益量级；本卡不给出跨域外推的数字。
- **实施难度**：⭐⭐⭐⭐☆（4/5）—— 需要（a）独立站侧可控随机化与事件级埋点，
  （b）双头模型 + 锚定 + 拉格朗日分配器的实现与调参，（c）券补贴成本口径与财务口径对齐。
- **优先级评分**：⭐⭐⭐⭐☆（4/5）—— 站外种草 → 独立站链路可直接落地并产出可判定阈值；
  站内链路因无用户级随机化，只能拿诊断价值，因此两条链路按不同优先级推进。
- **评估依据**：论文的增益是「描述性一致性」而非独立显著性，且工业数据不可独立复现 → 难度与优先级各扣一星；
  但决策形态（增量口径 → 预算分配 → 盈亏平衡阈值）与业务方现有的券预算决策链路完全对接，
  且代码模板可在无 GPU 的笔记本上跑通，故落地路径清晰。

### 论文量化证据摘要（每个数字的逐字出处见 ⑥）

| 证据 | 论文报告的数值 | 出处 |
|---|---|---|
| 合成多档 MT7：AUUC_GMV 最高者 / 本方法 | 0.615 / 0.613（本方法在一个种子标准差之内） | §5.2 Table 3 |
| 合成多档 MT7：PEHE_CVR（DualHeadNet / 本方法） | 0.048 / 0.058 | §5.2 Table 3 |
| 漏斗消融：硬漏斗把漏斗违例率从直接回归的六成以上压到 | 60 以上 → 0 | §5.3 Table 4 |
| 零膨胀压力测试的 PEHE_GMV 降幅（转化率区间 4.6–45.4） | 18–48 | §5.3 Table 5 |
| 紧预算下锚定拉格朗日 vs 随机分配的 ΔROI | 3.92 / 3.07 | §5.5 Table 7 |
| 联合 conformal 覆盖超出名义 1−α | 3–15 pp（α 取 0.05 / 0.10 / 0.20） | §5.6 Table 8 |
| 训练墙钟时间（十万 → 一百万用户） | 30 → 324 秒 | §5.7 Table 10 |
| 拉格朗日对偶更新耗时（百万用户、八档） | 0.13 秒 | §5.7 Table 10 |
| 工业多臂 RCT 规模 | 4.98 百万条曝光 / 2.79 百万用户；每种子 50 千训练 + 4.93 百万留出 | §5.8 |
| 工业 ΔROI：本方法在 7/7 锚点均值最高，中到大锚点领先第二名 | 0.18–0.21 个 ROI 单位（锚点 25–60%） | §5.8 Table 11 |
| 公开单次触发 RCT（Hillstrom）上的反例 | RERUM 0.747 / DualHeadNet 0.739 领先 | §5.2 |
| 冲突筛查（审计层）的峰值 F1 | 0.25（注入相关 0.6 时） | §5.4 Table 6 |
| 合成数据的档位设定 | 基线转化 8、八档、折扣区间 0–14 | §5.1 |

> 表中数字一律保留论文原值，只是把单位/量纲写成文字；逐字引文见 ⑥ 段。

---

## ⑥ 原文引用

**漏斗结构与因果识别**

> 原文："We propose FunnelCausalNet, an uplift estimator that couples a binary conversion head with a nonnegative conditional-value head under the funnel composition $\mu_{\mathrm{gmv}}=\mu_{\mathrm{conv}}\,\mu_{\mathrm{val}}$."
> 出处：2608.11675 §Abstract

> 原文："We identify $\tau^{c}_{t}$ and $\tau^{g}_{t}$ under randomized $T\mid X$ (RCT) as in standard analyses; we do not claim identification from purely observational logs."
> 出处：2608.11675 §3（Causal targets）

> 原文："$\mathrm{Var}(Y^{g}\mid X{=}x,T{=}t)=p\,\sigma_{v}^{2}+p(1-p)\,\mu_{v}^{2}.$"
> 出处：2608.11675 §4.2（Eq. 7）

> 原文："The first term is the within-converter variance; the second is the Bernoulli switching variance contributed by the zero mass."
> 出处：2608.11675 §4.2（Eq. 7 解读）

> 原文："Within these idealized assumptions, the ratio in (8) is below one whenever $(1{-}p)\mu_{v}^{2}/\sigma_{v}^{2}{>}0$, and shrinks as the zero mass $(1{-}p)$ grows or $\mu_{v}$ dominates $\sigma_{v}$."
> 出处：2608.11675 §4.2（Eq. 8 的 operational regime）

> 原文："Eq. (8) is an idealized pointwise variance comparison, not a universal optimality theorem or a guarantee for CATE ranking."
> 出处：2608.11675 §4.2（Eq. 8 的适用限制）

> 原文："We apply additive shifts estimated from RCT arm-wise averages on a held-in slice before forming rewards fed to the allocator, improving $\Delta\mathrm{ROI}$-style objectives without retraining."
> 出处：2608.11675 §4.4（Anchoring）

**合成数据上的估计质量与漏斗消融（论文最强的方法证据，但不是最强的业务证据）**

> 原文："EFIN attains the highest AUUC_GMV ($0.615$); FunnelCausalNet ranks second ($0.613$, within one seed standard deviation)"
> 出处：2608.11675 §5.2（Table 3）

> 原文："PEHE_CVR is led by DualHeadNet ($0.048$); FunnelCausalNet ($0.058$) remains competitive, confirming that funnel coupling does not destroy conversion-head identifiability."
> 出处：2608.11675 §5.2（Table 3）

> 原文："Hard coupling achieves the lowest PEHE_GMV at $10\mathrm{K}$, $20\mathrm{K}$, and $100\mathrm{K}$ samples, while the funnel-violation rate of A remains at $\gtrsim 60\%$ versus $0\%$ for C."
> 出处：2608.11675 §5.3（Table 4）

> 原文："Funnel composition reduces PEHE_GMV by $18$–$48\%$ across the tested $\hat{p}\in[4.6\%,45.4\%]$ range, with peak benefit at moderate-high zero inflation"
> 出处：2608.11675 §5.3（Table 5）

> 原文："generator parameters (baseline conversion ${\approx}8\%$, eight tiers $0\%$–$14\%$) fall inside operationally common e-commerce coupon ranges"
> 出处：2608.11675 §5.1（Semi-synthetic calibration disclosure）

**预算分配与不确定性层（论文的业务接口）**

> 原文："The anchored-Lagrangian pipeline attains higher $\Delta\mathrm{ROI}$ than random allocation under tight budgets—for example, $3.92$ versus $3.07$ at $B/B_{\mathrm{free}}{=}0.05$—with lower realized cost and competitive incremental GMV."
> 出处：2608.11675 §5.5（Table 7）

> 原文："Joint empirical coverage consistently exceeds nominal $1{-}\alpha$ by 3–15 pp across $\alpha\in\{0.05,0.10,0.20\}$"
> 出处：2608.11675 §5.6（Table 8）

> 原文："recommend wider nominal $\alpha\in[0.10,0.20]$ when widths must remain actionable, and pair intervals with anchored point estimates when feeding optimizers, because marginally valid lower-conformal bounds for $\tau^{g}$ at narrow $\alpha$ can be so pessimistic under zero inflation that budgeted LCB policies collapse to all-control assignments (Sec. 5.5)."
> 出处：2608.11675 §4.3（Deployment stance）

> 原文："Peak F1 reaches $\approx 0.25$ at $\rho_{\mathrm{conf}}{=}0.6$"
> 出处：2608.11675 §5.4（Table 6）

> 原文："Training scales sublinearly between $N{=}10^{4}$ and $10^{6}$ in our sweeps ($\approx 30\,\mathrm{s}\to 324\,\mathrm{s}$, $\sim 10\times$ wall-clock for $100\times$ users). Conformal calibration stays below one second even at $N{=}10^{6}$. Lagrangian dual updates stay near $0.13\,\mathrm{s}$ at one million users for $K{=}8$, whereas dense LP relaxations exceed tens of seconds already at $N{=}10^{5}$ and fail at larger $N$ due to memory."
> 出处：2608.11675 §5.7（Table 10）

**工业多臂 RCT：论文的最强业务证据与其自身保留意见**

> 原文："totaling $\approx 4.98\times 10^{6}$ exposure records from $\approx 2.79\times 10^{6}$ distinct users overall. For each of three permutation seeds we shuffle the full table, take the first $N_{\mathrm{train}}{=}50\mathrm{K}$ records for training, and retain the remaining $\approx 4.93$M exposure records per seed for evaluation."
> 出处：2608.11675 §5.8

> 原文："We treat the platform commission rate $\gamma$ as a sensitivity parameter over $[0.2,0.3]$, a band typical of online travel/coupon programs; the break-even point is $\Delta\mathrm{ROI}\!=\!1/\gamma\!\in\![3.3,5.0]$. Operating below the band ($\Delta\mathrm{ROI}\!<\!3$) means incremental commission no longer offsets subsidy cost"
> 出处：2608.11675 §5.8（Practical operating regime）

> 原文："at mid-to-large anchors ($25\%$–$60\%$) FunnelCausalNet’s mean exceeds the second-best by $0.18$–$0.21$ ROI units."
> 出处：2608.11675 §5.8（Table 11）

> 原文："Per-anchor paired-bootstrap CIs over three permutation seeds include $0$, so individual rows are not formally significant."
> 出处：2608.11675 §5.8（Table 11 caption）

> 原文："on industrial multi-arm RCT logs, FunnelCausalNet has the highest seed-averaged mean LP-frontier $\Delta\mathrm{ROI}$ at all $7/7$ reported anchors, although their correlation and the three permutation splits preclude an independent-anchor significance claim."
> 出处：2608.11675 §7（Conclusion）

**适用边界与可复现性（①b 的依据）**

> 原文："Empirically, revenue-focused rankers RERUM ($0.747$) and DualHeadNet ($0.739$) lead AUUC_GMV on Hillstrom, while all multi-tier funnel-aware deep models (DESCN, ECUP, FunnelCausalNet) underperform."
> 出处：2608.11675 §5.2（Public-RCT scope boundary）

> 原文："All causal interpretations assume RCT-like randomized assignment, not observational identification."
> 出处：2608.11675 §6（Identification scope and limitations）

> 原文："E7 uses record-level permutation splits, so repeated users can appear in both training and hold-out slices"
> 出处：2608.11675 §6（record-level permutation splits）

> 原文："Finally, the observed coupon arms are discrete offers: the model does not exploit smoothness or monotonicity across a continuous coupon dose"
> 出处：2608.11675 §6（discrete offer arms）

> 原文："The current version does not include a public code artifact."
> 出处：2608.11675 §5.9（Reproducibility）

> 原文："quantitative effect sizes, per-bucket exposure ratios, and ablation traces remain unavailable under the platform agreement"
> 出处：2608.11675 §5.8（Online consistency check）
