---
title: Skill-Cannibalization-Corrected-Attribution
module: 13-广告分析
topic: 用稀疏增量实验当因果锚点，校正日均归因中「非增量」的蚕食部分，并按业务层级分摊到 campaign 粒度
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2606.26690
paper: Attributed, But Not Incremental: Cannibalization-Corrected Attribution for Large-Scale Advertising
venue: ADKDD 2026
venue_tier: preprint
venue_source: EXPLICIT_RETIERS
evidence_grade: A
verified_by: verify_skill_code.py (K1 PASS) + quote_check.py (引文逐字核验 VERBATIM) + gate_check.py G2
verified_at: 2026-09-12
supersedes:
related: Skill-Ad-Attribution-Modeling.md, Skill-ROAS-Budget-Optimization.md, Skill-DiD-Difference-in-Differences.md, Skill-Power-Analysis-Sample-Size.md, Skill-Marketing-Mix-Modeling.md
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-05
l2_domain: 品牌与增长
l3_id: DOM-05-105
l3_business: 增量分析
l3_all: 增量分析 / 预算分配
l1_l2_l3: 业务运营/品牌与增长/增量分析
---

# Skill-Cannibalization-Corrected-Attribution

> **venue 说明（R3 要求显式标注）**：本文发表于 **ADKDD 2026**（KDD 的 workshop，2026-08-10 于济州），
> **不是 KDD 主会**。之所以仍纳入萃取：它同时给出了完整方法（§3）、离线前向验证（§4，
> 覆盖多个市场的多轮渠道级增量实验）与生产上线结果（§5），并非 workshop 短文或 demo。
> 但按 `venue-whitelist.md` 的层级规则，不应被当作 CCF-A 证据引用。

---

## ① 算法原理

**核心思想**：生产归因（last-touch / DDA）擅长「及时、细粒度」，但它是**观测**的，会把本来就会发生的
自然成交记在付费头上；增量实验能识别反事实增量，却**稀疏、延迟、贵**。本文不替换归因，而是把
实验当作**因果标尺**去校正归因。

**数学直觉**：设名义付费归因量 $A_t$、真实增量 $Lift_t$，则

- 蚕食率 $C_t = 1 - Lift_t / A_t$ —— 「记在我头上但不是我带来的」占比
- 校正后增量归因 $\hat{I}_t = A_t(1-\hat{C}_t)$

$C_t$ 在生产中**不可直接观测**，只能靠实验锚定。框架分两层：
**ETDC** 用稀疏的渠道-窗口实验读数训练 $f(\phi_{c,t})\to\widehat{Lift}$（$\phi$ 由自然基线代理、时间结构、
投放/渠道状态三块组成），把实验的因果**尺度**外推到日粒度；
**HCA** 再把已标定的蚕食总量按业务层级往下分摊，只保证聚合一致与可行性，
**不重新估计细粒度因果效应**。

**关键假设**：① 有可用的增量实验作锚点；② 代理变量与自然需求稳定相关、且不机械响应短期投放开关。

---

## ①b 反例与适用边界

**什么时候不要用**：

- **一期增量实验都没有**。ETDC 的因果尺度完全来自实验；没有实验就只能得到未标定的归因，
  此时本框架退化为「无锚可校」，**不要声称得到了增量口径的数字**。
- **需要单位级（某个 ASIN × 某个广告位）的因果结论**。HCA 的输出是**分摊**，不是识别。

**已知的失败模式**：

- 代理变量被投放动作污染（例如用「站内搜索量」当自然基线，但品牌词广告会直接抬高它），
  会让校正层把付费效果又学回去。
- 标定集过小 + 特征过多会过拟合。本模板实测：早期版本放了 7 个星期哑变量使参数数超过标定样本数，
  估计蚕食率明显偏离；补上渠道交互项、去掉近共线的二次项后收敛。
- 大促/新品上市会破坏代理变量的外生性，需要重新标定。

**论文自己承认的局限**（§6，原文引用见 ⑥）：

- 框架依赖实验的质量与覆盖：实验稀疏、置信区间宽、处置隔离不彻底、前拉效应（pull-forward）未解决，
  都会把不确定性传导进标定层。
- 代理外推要求「相关性可监控 + 近似外生」，而新品发布、季节性、市场冲击、获客渠道迁移都会削弱它。
- 细粒度输出应被理解为**受标定总量约束的运营分摊**，而非独立识别出的单位级因果效应。
- 负蚕食、自然外溢与渠道互补只被当作**诊断信号**，并未被完整建模。

---

## ② 母婴出海应用案例

### 场景一：站内品牌词广告「抢」自然品牌搜索的功劳

- **业务问题**：平台店铺（Amazon）的 SP 品牌词广告报表显示高 ROAS，同期自然搜索订单占比却在下滑。
  典型争功链路是：用户先在站外被种草（TikTok / 小红书），回到 Amazon **直接搜品牌词**，
  最后点了品牌词广告成交 —— 这笔单被记成「品牌词广告的功劳」，而它本来就会发生。
  每月做预算再分配时，这类虚高会让预算持续流向「收割既有需求」的广告位，而不是创造新需求的位置。
- **数据要求**：
  - **归因侧**：渠道 × 日粒度的名义付费归因转化量 $A_{c,t}$（Amazon 广告报表 / 独立站 GA4 均可导出），
    需与广告花费、活动周期字段对齐。
  - **实验侧**：渠道 × 时间窗的**增量实验读数**。平台店铺拿不到受众级随机化，可行做法是
    ①品牌词广告 on/off 分时测试；②按地区（geo holdout）分组的投放开关；③独立站侧的地理 holdout。
    每轮只需覆盖**一个渠道的一个窗口**，因此总量需求很小。
  - **代理变量**：自然需求代理（品牌词自然搜索量、自然流量会话数、Search Query Performance 里
    的非广告曝光）、时间结构（星期、节假日、旺季标记）、投放状态（花费、活动生命周期）。
- **数据可得性**：`部分可得（需补充增量实验读数）`。归因报表与代理变量在平台店铺与独立站内**都可得**；
  **缺口是增量实验**——需要主动设计品牌词 on/off 或 geo holdout，且每季度至少跑若干轮。
  若完全无法做实验，只能把本卡当作**诊断框架**用（看 $A$ 与自然基线的背离方向），**不得**对外给出增量口径数字。
- **预期产出**：日粒度校正后的增量归因 $\hat{I}_{c,t}$；按 campaign / placement 分摊的蚕食量；
  以及「负蚕食」诊断清单（自然外溢或漏记归因触点的可疑渠道）。
- **业务价值**：校正后的归因是后续所有预算决策的**输入层**。归因虚高的方向性后果是预算被系统性
  投向低增量渠道；把 $\hat{C}$ 显式估出来，就能把「品牌词收割」与「种草创造」分开计价，
  再据此调整两者的预算配比。

### 场景二：站内广告 vs 站外种草 的跨渠道预算再分配

- **业务问题**：平台店铺的站内广告与站外内容投放（引流至独立站）由不同团队负责、各自用各自的报表
  证明自己有效，两边都虚高，**无法比较**。需要一把共同的尺子来回答「下个月这块钱该给站内还是站外」。
- **数据要求**：同场景一，但要求两个渠道**各自**都至少有若干轮实验读数，否则无法标定渠道间的相对尺度。
- **数据可得性**：`企业内可得`（前提是两边都跑过至少少量实验；站外种草通常更容易做地理 holdout，
  可先用站外的尺度锚定站内的相对位置）。
- **预期产出**：统一的增量口径渠道对比表 + 层级分摊后的 campaign 级蚕食量。
- **业务价值**：把「渠道 SOI」换成「渠道增量 SOI」，让跨渠道预算可比。

---

## ③ 代码模板

> **本模板是业务化的简化实现，不是论文的忠实复现。** 论文生产版用的是 GLM（见 ⑥ 引用）；
> 下面用标准化 + Huber-IRLS 的线性模型实现同一职责，依赖仅 numpy，便于嵌入现有数据栈。
> ETDC 与 HCA 的接口与公式编号均对齐论文 §3.4 / §3.5。

### 第一段：特征与实验数据

```python
# -*- coding: utf-8 -*-
"""蚕食校正归因（ETDC + HCA）业务模板"""
from __future__ import annotations

import numpy as np

CHANNELS = ("站内SP品牌词", "站外种草")
# 直接按**意图蚕食率**参数化，而不是间接设 beta/kappa —— 后者的真实蚕食率
# 由 organic/spend 量级比值主导，两个渠道会被压到几乎相同的 C，与业务先验相反。
#   站内品牌词：品牌自然搜索本就会成交，蚕食重
#   站外种草：内容带来更多净新增需求，蚕食相对轻
C_BASE = np.array([0.72, 0.55])
ALPHA = np.array([0.55, 0.40])   # 归因报表中随自然需求走的部分
GAMMA = np.array([0.60, 0.35])   # 归因报表中随投放强度走的部分
N_DAYS = 180
EXPERIMENT_ROUNDS = 36            # 一半做标定、一半做评估 → 评估集 18 轮，对齐论文 §4.2

def make_market_panel(n_days: int = N_DAYS, seed: int = 20260912):
    """构造一个「归因虚高」的日粒度面板。

    因果结构（对齐论文 §3.1 的定义）：
        organic_t   自然需求基线（含星期效应与旺季抬升）
        A_{c,t}     生产归因报出的名义付费归因量
        C_true      真实蚕食率 = 1 - Lift/A（生产中**不可直接观测**，只能靠实验锚定）
        true_lift   = A * (1 - C_true)，即广告真正创造的增量

    未校正时 ARE = C/(1-C)：C=0.72 → 257%，C=0.55 → 122%，
    与论文 §4.3 报告的 slice 级 179%–334% 同量级（见卡片 ⑥ 引用）。
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    dow = days % 7
    dow_effect = np.array([1.10, 1.05, 1.00, 1.00, 1.05, 1.20, 1.15])[dow]
    season = 1.0 + 0.35 * (days >= 120)          # 第 120 天起进入旺季

    panel = {}
    for ci, name in enumerate(CHANNELS):
        organic = 1000.0 * season * dow_effect * (1 + 0.05 * rng.standard_normal(n_days))
        spend = np.clip(300.0 * season * (1 + 0.10 * rng.standard_normal(n_days)), 30, None)
        a_nominal = ALPHA[ci] * organic + GAMMA[ci] * spend
        c_true = np.clip(C_BASE[ci] + 0.03 * rng.standard_normal(n_days), 0.05, 0.95)
        panel[name] = {
            "channel_idx": ci,
            "A": a_nominal,
            "C_true": c_true,
            "true_lift": a_nominal * (1.0 - c_true),
            "organic": organic,
            "spend": spend,
            "dow": dow,
        }
    panel["n_days"] = n_days
    return panel

def build_features(row: dict, idx: np.ndarray) -> np.ndarray:
    """特征三块 φ = (φ^P 自然基线代理, φ^T 时间结构, φ^S 投放/渠道状态)。

    对齐论文 §3.4：代理变量须「与自然增长稳定相关」且「不机械响应短期投放开关」。

    ⚠️ 三个实测踩出来的设计要点：
    ① **必须带渠道交互项**。真实增量随渠道斜率不同；若只放渠道哑变量、不放交互项，
       一个汇总模型只能拟合平均斜率，两个渠道会同时偏。
    ② **参数个数必须少于标定样本数**。这里 8 个参数 / 18 个标定样本。
    ③ **不要放 organic²**。organic 只在窄区间内变动，二次项与一次项近似共线。
    """
    organic = row["organic"][idx] / 1000.0
    spend = row["spend"][idx] / 300.0
    dow = row["dow"][idx]
    is_ch1 = float(row["channel_idx"] == 1)
    n = len(idx)

    phi_p = np.column_stack([organic])
    phi_t = np.column_stack([(dow >= 5).astype(float), np.cos(2 * np.pi * dow / 7)])
    phi_s = np.column_stack([
        spend,
        np.full(n, is_ch1),
        organic * is_ch1,      # 渠道 × 自然基线 交互
        spend * is_ch1,        # 渠道 × 投放强度 交互
    ])
    return np.column_stack([np.ones(n), phi_p, phi_t, phi_s])

def make_experiments(panel: dict, rounds: int = EXPERIMENT_ROUNDS, win: int = 7,
                     holdout_frac: float = 0.34, seed: int = 7):
    """把渠道-窗口级增量实验读数切成「标定集」与「评估集」。

    对齐论文 §4.1 的 forward-in-time 协议：**同一轮实验不得既标定又评估**。
    实验读数是稀疏、带噪的因果锚点 —— 每轮只覆盖一个渠道的一个 7 天窗口。
    """
    rng = np.random.default_rng(seed)
    n = panel["n_days"]
    recs = []
    for r in range(rounds):
        ch = CHANNELS[r % len(CHANNELS)]
        start = int(rng.integers(0, n - win))
        idx = np.arange(start, start + win)
        row = panel[ch]
        true_win = float(row["true_lift"][idx].mean())
        # 实验读数：真实增量 + 采样噪声（论文称其为 noisy causal supervision）
        lift_exp = true_win * (1 + 0.12 * rng.standard_normal())
        recs.append({
            "channel": ch, "idx": idx, "lift_exp": lift_exp,
            "A_win": float(row["A"][idx].mean()),
        })
    n_hold = max(1, int(round(len(recs) * holdout_frac)))
    # 按时间排序后取后段做评估（forward-in-time）
    order = np.argsort([r["idx"][0] for r in recs])
    recs = [recs[i] for i in order]
    return recs[:-n_hold], recs[-n_hold:]
```

### 第二段：ETDC 标定层与 HCA 分摊层

```python
def huber_irls(X: np.ndarray, y: np.ndarray, delta: float = 1.0,
               n_iter: int = 100, l2: float = 1e-6) -> np.ndarray:
    """Huber 损失的 IRLS 求解（对齐论文 §3.4：用 Huber 抑制极端实验点）。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(n_iter):
        r = y - X @ beta
        s = np.maximum(np.abs(r), 1e-12)
        w = np.where(s <= delta, 1.0, delta / s)
        A_mat = X.T @ (w[:, None] * X) + l2 * np.eye(p)
        beta_new = np.linalg.solve(A_mat, X.T @ (w * y))
        if np.allclose(beta_new, beta, atol=1e-12):
            beta = beta_new
            break
        beta = beta_new
    return beta

class ETDCCalibrator:
    """Experiment-to-Daily Cannibalization 标定层。

    把稀疏、延迟的实验读数拟合成日粒度增量预测 f(φ)，再按论文式 (7) 反推蚕食率：
        Ĉ = 1 - Lift_hat / A ,  Î = A (1 - Ĉ)
    """

    def __init__(self, delta: float = 1.0, c_max: float = 0.95, l2: float = 1e-3):
        self.delta = delta
        self.c_max = c_max          # 生产中把 Ĉ 限制在既定运营区间内（§3.4）
        self.l2 = l2                # 标准化之后的 ridge 强度
        self.beta_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """标准化（截距列除外）。标定样本只有十几到几十条，不标准化时
        ridge 对量纲差异巨大的列（organic ~1.3 vs 常数 1）形同虚设。"""
        X = np.asarray(X, float)
        if fit:
            self.mu_ = X[:, 1:].mean(axis=0)
            sd = X[:, 1:].std(axis=0)
            self.sigma_ = np.where(sd < 1e-12, 1.0, sd)
        Z = X.copy()
        Z[:, 1:] = (X[:, 1:] - self.mu_) / self.sigma_
        return Z

    def fit(self, panel: dict, experiments: list[dict]) -> "ETDCCalibrator":
        Xs, ys = [], []
        for e in experiments:
            if e["channel"] not in panel:
                continue
            Xs.append(build_features(panel[e["channel"]], e["idx"]).mean(axis=0))
            ys.append(e["lift_exp"])
        if not Xs:
            raise ValueError("标定集为空：至少需要一轮增量实验读数")
        X = self._standardize(np.vstack(Xs), fit=True)
        self.beta_ = huber_irls(X, np.asarray(ys), delta=self.delta, l2=self.l2)
        return self

    def predict_daily(self, panel: dict, channel: str) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("请先 fit()")
        idx = np.arange(panel["n_days"])
        X = build_features(panel[channel], idx)
        return self._standardize(X) @ self.beta_

    def correct(self, panel: dict, channel: str) -> dict:
        """返回日粒度 Ĉ 与校正后增量归因 Î（以及负值诊断标记）。"""
        a = panel[channel]["A"]
        lift_hat = np.maximum(self.predict_daily(panel, channel), 0.0)
        c_hat = 1.0 - lift_hat / np.maximum(a, 1e-9)
        negative = c_hat < 0            # 负蚕食：可能的自然外溢/渠道互补（§3.1）
        c_clipped = np.clip(c_hat, 0.0, self.c_max)
        return {
            "C_hat": c_clipped,
            "I_hat": a * (1.0 - c_clipped),
            "negative_diagnostic": negative,
        }

def hierarchical_allocate(total: float, propensities: np.ndarray,
                          caps: np.ndarray, tol: float = 1e-9,
                          n_iter: int = 200) -> np.ndarray:
    """HCA 层级分摊：把父节点总额分给子节点，满足论文式 (8) 的三个约束。

        ① 聚合一致  sum(alloc) == total
        ② 可行性    0 <= alloc_j <= caps_j
        ③ 局部性    仅在兄弟集合内重分配（由调用方按层级逐层调用保证）

    触顶子节点的残额在同级兄弟间继续按倾向重分配。
    """
    w = np.asarray(propensities, float).copy()
    caps = np.asarray(caps, float)
    total = float(total)
    if np.any(caps < 0):
        raise ValueError("caps 不得为负")
    if total < 0:
        raise ValueError("total 不得为负")
    if total > caps.sum() + 1e-9:
        raise ValueError(f"不可行：总额 {total:.4f} 超过可行上界 {caps.sum():.4f}")
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)          # 无倾向信息时退化为按可行余量均分

    alloc = np.zeros_like(w)
    active = np.ones(len(w), bool)
    for _ in range(n_iter):
        remaining = total - alloc.sum()
        if remaining <= tol or not active.any():
            break
        wa = w[active]
        if wa.sum() <= 0:
            wa = np.ones_like(wa)
        share = remaining * wa / wa.sum()
        room = caps[active] - alloc[active]
        add = np.minimum(share, room)
        idx = np.where(active)[0]
        alloc[idx] += add
        active = alloc < caps - tol

    if abs(alloc.sum() - total) > 1e-6:
        raise RuntimeError(f"分摊未收敛：{alloc.sum():.6f} != {total:.6f}")
    return alloc

def allocate_tree(node: dict, c_hat_root: float, tol: float = 1e-6) -> dict:
    """按业务层级（渠道 → campaign/placement）逐层分摊蚕食量。"""
    out = {}
    children = node.get("children", {})
    if not children:
        return out
    names = list(children)
    props = np.array([children[k].get("propensity", 1.0) for k in names], float)
    caps = np.array([children[k]["A"] for k in names], float)
    alloc = hierarchical_allocate(c_hat_root, props, caps, tol=tol)
    for k, v in zip(names, alloc):
        out[k] = v
        out.update(allocate_tree(children[k], v, tol=tol))
    return out

def are(pred: np.ndarray, truth: np.ndarray) -> float:
    """绝对相对误差 ARE（论文式 12），lift 加权以抑制低增量实验的不稳定。"""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.average(np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-9),
                            weights=np.abs(truth)))
```

### 第三段：自检与演示

```python
def test_hca_sums_to_parent_and_respects_caps():
    """HCA 必须同时满足聚合一致与可行性（论文式 8）。"""
    alloc = hierarchical_allocate(total=100.0,
                                 propensities=np.array([3.0, 1.0, 1.0]),
                                 caps=np.array([40.0, 500.0, 500.0]))
    assert abs(alloc.sum() - 100.0) < 1e-6, alloc
    assert np.all(alloc <= np.array([40.0, 500.0, 500.0]) + 1e-9), alloc
    assert abs(alloc[0] - 40.0) < 1e-6, "倾向最高但触顶的子节点应恰好取满"

def test_hca_rejects_infeasible_total():
    try:
        hierarchical_allocate(10.0, np.array([1.0, 1.0]), np.array([3.0, 3.0]))
    except ValueError:
        return
    raise AssertionError("总额超上界时应抛 ValueError")

def test_correction_removes_non_incremental_part():
    """校正后归因必须显著低于生产归因 —— 这正是论文要解决的问题。"""
    panel = make_market_panel()
    cal, hold = make_experiments(panel)
    model = ETDCCalibrator().fit(panel, cal)

    raw_are, etc_are = [], []
    for e in hold:
        ch, idx = e["channel"], e["idx"]
        raw_are.append(e["A_win"])
        etc_are.append(float(model.predict_daily(panel, ch)[idx].mean()))
    truth = np.array([e["lift_exp"] for e in hold])
    assert are(np.array(etc_are), truth) < are(np.array(raw_are), truth), \
        "ETDC 标定后的 ARE 应低于未校正的生产归因"
    assert float(np.mean(etc_are)) < float(np.mean(raw_are)), \
        "校正应把归因往下拉（去掉非增量部分）"

def test_cannibalization_rate_is_within_bounds():
    panel = make_market_panel()
    cal, _ = make_experiments(panel)
    model = ETDCCalibrator(c_max=0.95).fit(panel, cal)
    for ch in CHANNELS:
        res = model.correct(panel, ch)
        assert res["C_hat"].min() >= 0.0 and res["C_hat"].max() <= 0.95 + 1e-9
        assert np.all(res["I_hat"] <= panel[ch]["A"] + 1e-6), "校正后归因不得超过名义归因"

def test_brand_channel_has_higher_cannibalization():
    """业务先验：站内品牌词蚕食率应高于站外种草（自然品牌搜索本就会成交）。"""
    panel = make_market_panel()
    cal, _ = make_experiments(panel)
    model = ETDCCalibrator().fit(panel, cal)
    c_brand = model.correct(panel, CHANNELS[0])["C_hat"].mean()
    c_offsite = model.correct(panel, CHANNELS[1])["C_hat"].mean()
    assert c_brand > c_offsite, (c_brand, c_offsite)

def test_negative_diagnostic_flags_organic_spillover():
    """实测 lift 超过名义归因时（负蚕食），应打诊断标记而非静默截断（论文 §3.1）。"""
    panel = make_market_panel()
    cal, _ = make_experiments(panel)
    model = ETDCCalibrator().fit(panel, cal)
    probe = dict(panel)
    probe[CHANNELS[0]] = dict(panel[CHANNELS[0]], A=panel[CHANNELS[0]]["A"] * 0.05)
    res = model.correct(probe, CHANNELS[0])
    assert res["negative_diagnostic"].all(), "应全部标记为负蚕食诊断"
    assert res["C_hat"].min() >= 0.0, "报告值仍须截断到合法区间"

if __name__ == "__main__":
    panel = make_market_panel()
    cal, hold = make_experiments(panel)
    model = ETDCCalibrator().fit(panel, cal)

    truth = np.array([e["lift_exp"] for e in hold])
    raw_pred = np.array([e["A_win"] for e in hold])
    etc_pred = np.array([float(model.predict_daily(panel, e["channel"])[e["idx"]].mean())
                         for e in hold])

    print("=== ETDC 标定效果（留出实验轮，越低越好）===")
    print(f"  未校正生产归因 ARE : {are(raw_pred, truth):.4f}")
    print(f"  ETDC 标定后  ARE : {are(etc_pred, truth):.4f}")
    print(f"  误差下降           : {(1 - are(etc_pred, truth) / are(raw_pred, truth)) * 100:.1f}%")

    for ch in CHANNELS:
        res = model.correct(panel, ch)
        print(f"  {ch} 名义归因 {panel[ch]['A'].mean():.1f}  真实蚕食率 "
              f"{panel[ch]['C_true'].mean():.3f}  估计蚕食率 {res['C_hat'].mean():.3f}")

    tree = {"children": {
        "站内SP品牌词": {"A": 700.0, "propensity": 3.0, "children": {
            "brand_exact": {"A": 400.0, "propensity": 4.0},
            "brand_broad": {"A": 300.0, "propensity": 1.0}}},
        "站外种草": {"A": 500.0, "propensity": 1.0}}}
    root_total = float(model.correct(panel, CHANNELS[0])["C_hat"].mean()
                       * panel[CHANNELS[0]]["A"].mean())
    alloc = allocate_tree(tree, root_total)
    lvl1 = alloc["站内SP品牌词"] + alloc["站外种草"]
    print(f"  HCA 根层汇总 {lvl1:.6f} vs 根节点蚕食量 {root_total:.6f} "
          f"（聚合一致性 {'OK' if abs(lvl1 - root_total) < 1e-6 else '不一致'}）")
    print("全部自检通过 ✅")
```

**本模板的实际运行输出**（`python3 <本卡代码>.py` 可复现；**这是模拟数据的结果，不是论文数据**）：

```text
=== ETDC 标定效果（留出实验轮，越低越好）===
  未校正生产归因 ARE : 2.2205
  ETDC 标定后  ARE : 0.1722
  误差下降           : 92.2%
  站内SP品牌词 名义归因 860.5  真实蚕食率 0.726  估计蚕食率 0.713
  站外种草 名义归因 598.6  真实蚕食率 0.548  估计蚕食率 0.579
  HCA 根层汇总 613.589351 vs 根节点蚕食量 613.589351 （聚合一致性 OK）
全部自检通过 ✅
```

> ⚠️ **不要把上表那个「误差下降」百分比与论文表 1 的数字混为一谈。** 两者数值接近纯属巧合：
> 论文的 91.38% 是真实生产系统的渠道级标定误差下降（见 ⑥ 引用）；
> 上表那个「误差下降」百分比来自本模板的合成面板，只说明实现逻辑自洽。模拟中未校正误差的**量级**
> 是我**故意**调到论文 §4.3 报告的 slice 级区间内，以便量级可比 —— 它不是独立验证，也不是复现。

---

## ④ 技能关联

| 关联卡片 | 关系 | 为什么组合 |
|---|---|---|
| `Skill-Ad-Attribution-Modeling.md` | 前置 | 本卡不替换归因，而是**叠在**归因之上做因果校正；先用它产出 $A_{c,t}$，本卡再估计 $\hat{C}$。 |
| `Skill-ROAS-Budget-Optimization.md` | 下游 | 校正后的 $\hat{I}$ 是预算优化的**输入**。直接拿未校正归因做 ROAS 优化，等于把预算推向低增量渠道。 |
| `Skill-DiD-Difference-in-Differences.md` | 前置（实验替代方案） | 平台店铺往往做不了受众级随机化；geo holdout 的读数正是 DiD 的标准产出，可直接喂给 ETDC 标定层。 |
| `Skill-Power-Analysis-Sample-Size.md` | 前置（实验设计） | 论文 §3.3 会剔除「合格样本不足」的实验；设计阶段就要算清楚窗口长度与最小可检测效应，否则实验读数进不了标定集。 |
| `Skill-Marketing-Mix-Modeling.md` | 互补（交叉校验） | MMM 从**时间聚合**侧估渠道贡献，本卡从**渠道-日**侧校正归因。两条独立路线若给出方向相反的结论，说明代理变量或被污染了。 |

---

## ⑤ 商业价值评估

**ROI 预估**（给公式与参数来源，不直接给编造的数）：

```
预算再分配增益 = 被误配的预算额 × 校正后识别出的低增量渠道占比 × 该部分预算的增量ROAS提升
```

参数来源：①**被误配的预算额**——贵司站内广告月花费 × 品牌词广告占比（来自广告报表）；
②**低增量渠道占比**——本卡产出的 $\hat{C}$ 按渠道排序后，取 $\hat{C}$ 显著高于其他渠道的部分；
③**增量ROAS提升**——用校正前后同一批实验轮的 $\widehat{Lift}$ 对比得到（本卡内置的 forward-in-time 评估协议）。
三个参数都能在**企业内**算出来，无需外部基准。

论文给出的方向性参照：部署后整体蚕食率下降了约 15 个百分点（§5，见 ⑥ 引用）——
即被识别为「非增量」的流量份额发生了实质性变化，说明该校正层确实改变了预算流向。
**注意这是论文的生产系统读数，不是贵司的预期收益**；贵司量级取决于品牌词广告占比与实验覆盖度。

- **实施难度**：⭐⭐⭐☆☆（3/5）。方法本身不难（线性模型 + 分摊算法，本卡已给可运行实现），
  真正的成本在**组织侧**：需要推动品牌词 on/off 或 geo holdout 实验，并让投放团队接受
  「自己的渠道被下调」。技术风险低、协作风险高。
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）。归因是**上游**。上游口径错了，下游所有优化都会放大错误：
  预算分配、活动评估、渠道诊断全部受污染。修上游的杠杆最大。
- **评估依据**：①论文在生产系统上验证过（§5），非纯离线；②依赖的数据在平台店铺 + 独立站内**大部分可得**，
  唯一缺口是实验，而实验设计成本可控（每季若干轮、每轮一个渠道-窗口）；
  ③与已有卡片 `Skill-Ad-Attribution-Modeling` / `Skill-ROAS-Budget-Optimization` 直接串联，
  不是孤立新领域。

---

## ⑥ 原文引用

> 原文："We refer to this gap between credited conversions and causal incremental conversions as the attribution–cannibalization mismatch"
> 出处：2606.26690 §1（PDF 第 1 页）

> 原文："Production attribution is timely, granular, and continuously available, but observational by construction."
> 出处：2606.26690 §1（PDF 第 1 页）

> 原文："we define the cannibalization rate as the fraction of nominal paid-attributed conversions that is not truly incremental"
> 出处：2606.26690 §3.1 式 (2) 前（PDF 第 3 页）

> 原文："In deployment, we use a generalized linear model for robustness and operational simplicity."
> 出处：2606.26690 §3.4（PDF 第 4 页）

> 原文："as Huber loss to reduce the influence of extreme daily experimental points that remain after input filtering"
> 出处：2606.26690 §3.4（PDF 第 4 页）

> 原文："Negative cannibalization estimates are treated as diagnostic signals for potential organic spillover, channel complementarity, or missing attribution touchpoints"
> 出处：2606.26690 §3.4（PDF 第 4 页）

> 原文："producing actionable fine-grained corrections while satisfying three operational constraints: aggregate consistency, feasibility, and locality"
> 出处：2606.26690 §3.5（PDF 第 5 页）

> 原文："the same experiment is not used both for calibration and evaluation within the same model version"
> 出处：2606.26690 §4.1（PDF 第 5 页）

> 原文："This section summarizes the overall performance across 18 rounds of channel-level A/B incrementality experiments, covering up to eight markets."
> 出处：2606.26690 §4.2（PDF 第 5 页）

> 原文："Device ML reduces normalized calibration error by 69.11% relative to Raw Attribution, but its signed-error distribution remains relatively wide across experiment slices."
> 出处：2606.26690 §4.2（PDF 第 5 页）

> 原文："ETDC+HCA achieves the best overall calibration, reducing normalized calibration error by 91.38%, with median signed error close to zero and a narrower interquartile range."
> 出处：2606.26690 §4.2（PDF 第 5 页）

> 原文："| ETDC+HCA | 0.09 | 91.38% | 0.60% | [-8.11%, 7.24%] |"
> 出处：2606.26690 §4.2 表 1（ETDC+HCA 行；PDF 第 5 页）

> 原文："Raw Attribution consistently overestimates incremental contribution in these slices, with signed relative errors ranging from 179% to 334%."
> 出处：2606.26690 §4.3（PDF 第 6 页）

> 原文："In contrast, ETDC+HCA stays close to experimental lift across all reported slices, with signed relative errors ranging from -7% to 10%."
> 出处：2606.26690 §4.3（PDF 第 6 页）

> 原文："the measured overall cannibalization rate subsequently decreased by approximately 15 percentage points"
> 出处：2606.26690 §5（PDF 第 6 页）

> 原文："The framework depends on the quality and coverage of incrementality experiments. Sparse experiments, wide confidence intervals, incomplete treatment isolation, or unresolved pull-forward effects can propagate uncertainty into calibration."
> 出处：2606.26690 §6 Limitations（PDF 第 6 页）

> 原文："Proxy-based extrapolation also requires monitored relevance and approximate exogeneity; product launches, seasonality, market shocks, or acquisition-channel shifts may weaken these assumptions and require recalibration."
> 出处：2606.26690 §6 Limitations（PDF 第 6 页）

> 原文："fine-grained outputs should be interpreted as calibrated allocations rather than independently identified unit-level causal effects"
> 出处：2606.26690 §6 Limitations（PDF 第 6 页）

---

## 附：证据链与核验方式

- **全文底本**：`paper2skills-vault/papers/13-广告分析/p2s-2026-0001/fulltext.md`
  （由 `paper2skills-research/scripts/fetch_fulltext.py` 从 arXiv LaTeXML HTML 转换，保留章节号）
- **引文逐字核验**：`python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <本卡>`
  —— 每条 ⑥ 引用块都会回查全文底本，报告 `VERBATIM` / `FUZZY` / `FABRICATED`
- **K1 代码可执行**：`python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <本卡>`
- **K2 门禁**：`python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <本卡>`
- **本地可复现数字的约定**：论文事实数字一律进 ⑥ 引用块；本模板运行输出与按贵司参数代入的算式
  放进代码/输出围栏并标注「可复现」——围栏内的数字是**可自行验证**的，不冒充论文结论。
