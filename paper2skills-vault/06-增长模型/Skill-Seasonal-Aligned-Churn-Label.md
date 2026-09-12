---
title: Skill-Seasonal-Aligned-Churn-Label
module: 06-增长模型
topic: 把衰退预警标签的基线换成「去年同期同样的 k 个日历月」，从源头消除季节性误报
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.18174
paper: Seasonal false alarms in customer churn and decline early-warning systems: adjacent-window labels confound seasonality with decline, and a year-over-year correction
venue: arXiv preprint
venue_tier: preprint
evidence_grade: A
verified_by: verify_skill_code.py（K1=PASS）+ quote_check.py（引文逐字命中）+ gate_check.py --only G2（passed）+ 人工抽检 Table 1 / Table 3 数字
supersedes:
related: Skill-Customer-Churn-Prediction.md, Skill-Uplift-Churn-Prediction.md, Skill-Cohort-Retention-Analysis.md, Skill-Prophet-Forecasting.md, Skill-User-Lifecycle-STAN.md
---

# Skill Card: Seasonal-Aligned Churn Label（季节性对齐的衰退标签）

> **与 `Skill-Customer-Churn-Prediction.md` 的分工（先读这一行）**
> 那张卡解决「**给定**流失标签，怎么把用户排序、怎么选阈值」；本卡解决「**标签本身错了**」。
> 论文的核心发现：相邻窗口（trailing vs forward）阈值比标签的两个窗口覆盖**不同的日历月**，
> 于是季节性实体的「淡季」会被读成「衰退」，而模型学得越好，越像一个季节性检测器。
> 本卡修的是标签，不是模型。两者可以叠加，但**先修标签，再训模型**。

---

## ① 算法原理

**核心思想**
非契约型业务里「衰退」只能从交易历史自己造出来：业界标准做法是相邻窗口阈值比 ——
锚点月**前** k 个月合计为基线 `B_adj`，**后** k 个月为结果窗口 `F`，`F ≤ θ·B_adj` 即判「衰退」。
本卡把基线换成**与结果窗口同日历月的去年同期** `B_yoy`（year-over-year 对齐），k、θ、特征、模型一律不动。

**数学直觉**
纯季节实体上定义锚点月比值 `ρ(m) = W(m+1)/W(m−k+1)`（W 为 k 个月窗口和）。论文 Proposition 1 三条：
**(i)** 相邻标签只依赖**锚点的日历月**、与实体无关，每年在同一锚点同开同关；
**(ii)** 十二个 `ρ(m)` 的**几何平均恰为 1** —— 只要窗口和不全相等，必有月份落在 parity 之下，
事件率于是变成日历月的函数；**(iii)** 对齐标签下 `F = B_yoy`，纯季节实体比值恒为 1，永不触发。
一句话：**把「和上一段时间比」换成「和去年这个时候比」。**

**关键假设**
① 乘法季节性（加性下依赖仍在，但随实体量级衰减）；② 年度周期 + 12 个月滞后（非年度周期未测试）；
③ 历史更长：约 12+k 个月才出第一个标签，相邻窗口只要 2k；④ 两者都不泄漏：特征只用锚点之前的月份。

---

## ①b 反例与适用边界

- **什么时候不要用这个算法**
  1. **历史不足**：出海首年（不足 12+k 个月）根本算不出 yoy 基线，只能走 ② 场景二的先验退路；
  2. **不活跃型流失定义**：基于「N 天未活跃」的流失标签有相关但**结构不同**的季节混淆（季节性安静会被读成流失），论文的命题**不覆盖**这一类；
  3. **持续强增长期**：去年基线低于当前水平，对齐切点实际上更严 —— 论文在合成数据上实测召回从 0.86 掉到 0.70；
  4. **上一年窗口本身被污染**：一次促销尖峰或断货会污染之后最多 k 个月的标签，而相邻基线 k 个月就自愈；
  5. **需要判断「当下状态」而非「新发衰退」**：衰退后企稳（decline then stabilization）是对齐标签的结构性盲点，得另开一条 current-state watchlist；
  6. **非年度周期 / 移动节假日主导**的市场（论文明确未测试）。
- **已知的失败模式**
  - **把诊断量读成业务事实**：anchor-month 事件率曲线的起伏不是「客户基础固有的流失季节性」。论文点名一篇同行评审研究正是这样读的（把波动解释为「the inherent seasonality of customer defection」），它没有把波动追溯到标签构造，因此既没量化、也没修。
  - **只在训练集层面打补丁**：多切片池化 / 整年窗口 / 月份平移的重叠训练集能平衡训练混合，但**改不了任何单个样本的戳**，被池化的错误标签原样留在训练集里。
  - **把日历月塞进特征**：论文在合成真值上实测，给相邻标签模型加上 month-of-anchor 指示特征后，对真衰退的排序**没有任何改善**（0.44）。
  - **用一个全品类季节指数顶替同比**：只能修一部分，而且会把误报高峰换个月份（见 ② 场景二与 ③ 的运行输出）。
- **论文自己承认的局限**（§7 四条）
  1. 模型层的受控对比只在**单一组织**上做过，迁移性未验证；
  2. 对齐构造假设年度周期与 12 个月滞后，**非年度周期未测试**；
  3. 对齐基线**重放历史**（一次异常污染最多 k 个月），且实体在头 12+k 个月没有标签 —— 恰是非契约型流失最集中的早期关系期；
  4. 论文只修**阈值比**这一类标签，不活跃型定义与 latent-attrition 模型族不在其射程内。

---

## ② 母婴出海应用案例

### 场景一：Amazon 平台店铺复购面板 + 独立站 email/SMS 用户的季度「衰退预警清单」重标

- **业务问题**：每季度要交给客服/运营一份「看起来在衰退」的复购用户清单（吸奶器耗材、奶瓶配件、辅食等），
  每上一个名字就占用**一个干预位**。现状是：黑五（11 月）旺季一过，1 月的清单暴涨，
  客服照着名单给刚囤完货的妈妈发「我们想你了」召回券与优惠券，既浪费预算又伤体验。
  母婴还有一层专属混淆：**宝宝月龄生命周期**（0–6 月龄高频买耗材 → 12 月龄后自然降级到水杯/餐具），
  相邻窗口会把「宝宝长大了、需求降级了」读成「用户要流失了」；这两者与季节性三者叠加，
  正是 MasterPrompt 增长模型领域规则点名的高风险区。
- **数据要求**：月度面板 `user_id × 月份 × GMV`（或订单数），平台店铺复购用户与独立站 email/SMS 用户各一份；
  **≥2 个完整年度**（论文口径：12+k 个月才出第一个 yoy 标签）；带材料性围栏 ——
  论文公共口径是「过去 12 个可用月中至少 75% 为正」，生产系统还额外加了金额门槛（50K / 100K / 200K 三档都测过）。
- **数据可得性**：`部分可得（需补充宝宝月龄与首购日期才更准）`。
  平台店铺的**用户级**月度复购口径要从平台报表导出并做用户级去重（Amazon 侧拿不到用户在站外的全部行为）；
  独立站 email/SMS 用户可得；宝宝月龄需从订单 SKU 与订阅记录反推，部分用户缺失 —— 缺失时用首购月份当代理。
  **若出海不足 2 年：直接走场景二，不要硬做同比。**
- **预期产出**：每季度一张「衰退」清单 + 两个诊断量：
  ① anchor-month 事件率曲线（12 个日历月的事件率、CV、max/min）；
  ② 对齐不一致率（相邻标签的事件里，有多少在 yoy 定义下不成立）。
  这两个量都是「几行代码」级别的审计工具，正好可以直接放进季度复盘。
- **业务价值**：见 ⑤ 的 ROI 公式与参数来源。论文部署系统里，重标之后**行动清单缩减三分之一**（119 → 79 个账号），
  离开清单的正是那些没有对齐对应物的季节性形状事件。
- **口径核对（重要）**：立项材料里记的是「生产系统约三分之一行动清单是误报」，这一点在原文中**确认成立** ——
  论文摘要写的是「每三个被服务的行动位里有一个给了在季节性对齐标签下会消失的标记」，引言里换成「约每三个被服务的标记里有一个」。
  但要注意口径分层：三分之一是**服务清单**（部署系统）层面的数字；
  在论文统一协议下的**材料队列**上，对齐不一致率是 50%，而公共面板是 37%–69%。三者不是同一个分母，引用时不要混用。

### 场景二：历史不足 2 年时的退路 —— 品类季节性先验，以及它到底丢掉了什么

- **业务问题**：出海首年只有 8–14 个月数据，做不了同比。改用**品类季节性先验**
  （平台类目报表的月度销量/搜索热度、Google Trends、行业报告的月度指数）当月度季节指数，
  把月度活跃度除以该指数，再套原来的相邻窗口标签。
- **数据要求**：12 个值的品类月度季节指数，月份与自营面板对齐；至少 12 个月自营数据用于校准量级。
- **数据可得性**：`部分可得（指数可得，但它未必是你店铺的相位）`。
- **预期产出**：一份「打了折」的清单 —— 必须同时在评审材料里写清哪几条结论不可信。
- **业务价值与损失（必须一起写进评审）**：论文实测这条退路**只能修一部分**。
  合成队列上 pooled 指数臂的误报率与相邻窗口臂几乎持平（9.2% vs 9.5%）；
  真实面板上它留下的离散度是对齐标签的两三倍（原文表述：two to more than three times，见 ⑥）；根因是**一个面板级指数服务不了峰月不同的实体**
  （母婴各细分类目峰月本就不同：吸奶器耗材、童装、辅食的旺季并不重合）。
  ③ 的演示代码在自造面板上复现了同一现象：先验把误报高峰从 12–1 月**搬到** 3–6 月，
  总量降下来了，但曲线**没有被压平**（离散度反而略升）。
  **结论：先验只能当过渡；满两个完整年度立刻切同比，并让两种清单并行跑一个季度做对照。**
  另外，论文把「非年度周期需对齐到相关周期的同一相位」列为**未测试**的外推 ——
  母婴场景里若想按宝宝月龄分层后再做同比，那属于本卡明确的**未验证假设**，需要自己留出验证集。

---

## ③ 代码模板

下面两段代码按文档顺序拼成一个模块：第一段是引擎（自造月度面板 → 三种窗口标签 → 两个诊断量 → 品类先验退路），
第二段是断言与演示。面板是**自造的合成数据**（含明确季节性与一个真衰退队列），
运行输出用于说明机制，**不是论文数字**。

**关于论文自带的复现包**：论文声明随预印本提供复现材料 —— 脚本、**预指定**的分析计划、
以及带时间戳的结果工件，足以复现全部合成与公共面板数字；公共面板（M5 / 澳大利亚旅游 / UCI Online Retail II）
的清洗步骤也在其中；生产面板是专有数据，只以比率、比值与计数形式出现，不公开。
因此本卡的做法是：**机制与阈值口径照论文，代码按业务面板重写**（论文的脚本面向论文的数据格式，
不能直接读我们的用户级月度面板）。

```python
"""季节性对齐的衰退标签（Seasonal-Aligned Decline Label）— 母婴出海复购预警改造模板。

出处：arXiv:2608.18174。核心结论：相邻窗口（trailing vs forward）阈值比标签的两
个窗口覆盖不同日历月，对季节性实体而言"淡季/旺季"本身就会被读成"衰退"；把基线换成
"去年同期同样的 k 个日历月"（yoy 对齐）可以在源头压平 anchor-month 事件率曲线。

本模块只依赖 numpy / pandas，输入是自造月度面板（不读外部文件）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# 品类月度季节性指数（示例：11 月黑五/圣诞备货峰值、2 月低谷，峰谷比约 4.4）
CATEGORY_SEASONAL_INDEX = np.array(
    [0.60, 0.52, 0.55, 0.60, 0.68, 0.75, 0.82, 0.95, 1.20, 1.60, 2.30, 1.45]
)


def make_monthly_panel(
    n_entities: int = 600,
    n_months: int = 48,
    declining_share: float = 0.5,
    drop: float = 0.60,
    noise_sigma: float = 0.12,
    seed: int = 7,
    phase_choices: Iterable[int] = (0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 2, 2),
) -> pd.DataFrame:
    """生成月度复购面板（长表）。

    phase_choices 控制峰月相位的分布：多数实体跟随品类主相位（黑五/圣诞），
    少数细分品类整体前移或后移 1-2 个月 —— 这正是「一个全品类共用的季节先验
    只服务得了一部分实体」的来源，也是论文 pooled-index 退路只修一部分的原因。

    两类实体：
      * seasonal_only —— 只有季节性、没有趋势（论文的 clean cohort，任何"衰退"事件都是误报）；
      * declining     —— 在随机月份发生一次性水平下滑（论文的 -60% level shift），
                         用来检验修正后的标签是否仍能抓到真衰退。

    每个实体的季节峰月随机（复刻论文 synthetic panel 的 random peak month 设计），
    因此"一个全品类共用的季节先验"必然错配一部分实体。

    返回列：entity / month / value / phase / cohort / step_month
    """
    rng = np.random.default_rng(seed)
    n_declining = int(round(n_entities * declining_share))
    level = rng.lognormal(0.0, 0.35, n_entities)          # 实体规模异质性（对数正态）
    months = np.arange(n_months)
    phases = rng.choice(np.asarray(list(phase_choices)), size=n_entities)
    # 每个实体一条按峰月相位滚动过的季节曲线
    curves = np.stack([np.roll(CATEGORY_SEASONAL_INDEX, int(p))[months % 12] for p in phases])
    noise = rng.lognormal(0.0, noise_sigma, size=(n_entities, n_months))
    values = level[:, None] * curves * noise

    step = np.full(n_entities, 10 ** 6, dtype=int)        # 10**6 = 从不衰退
    if n_declining:
        step[:n_declining] = rng.integers(13, max(14, n_months - 6), size=n_declining)
    values = values * np.where(months[None, :] >= step[:, None], 1.0 - drop, 1.0)

    cohort = np.where(np.arange(n_entities) < n_declining, "declining", "seasonal_only")
    return pd.DataFrame(
        {
            "entity": np.repeat(np.arange(n_entities), n_months),
            "month": np.tile(months, n_entities),
            "value": values.ravel(),
            "phase": np.repeat(phases, n_months),
            "cohort": np.repeat(cohort, n_months),
            "step_month": np.repeat(step, n_months),
        }
    )


def pivot_panel(panel: pd.DataFrame, n_months: int | None = None) -> np.ndarray:
    """长表 → (实体 × 月份) 宽表。缺失月按 0 计（无购买 = 无活跃）。"""
    wide = panel.pivot_table(index="entity", columns="month", values="value", aggfunc="sum")
    if n_months is None:
        n_months = int(wide.columns.max()) + 1
    wide = wide.reindex(columns=range(n_months)).fillna(0.0).sort_index()
    return wide.to_numpy(dtype=float)


def build_window_labels(
    panel: pd.DataFrame,
    k: int = 6,
    theta: float = 0.5,
    pos_floor: float = 0.75,
    values: np.ndarray | None = None,
) -> pd.DataFrame:
    """构造三种窗口标签，返回 anchor 级（实体 × 锚点月）明细表。

    对锚点月 a（0-based）：
      B_adj = 过去 k 个月合计（相邻窗口基线，业界标准做法）
      F     = 未来 k 个月合计（结果窗口）
      B_yoy = 去年同期同样的 k 个日历月合计（论文提出的对齐基线）
      D_adj = 1{F <= theta * B_adj}；D_yoy = 1{F <= theta * B_yoy}

    有效样本围栏（fence）：过去 12 个可用月中至少 pos_floor 比例为正，
    且两个基线都严格为正 —— 与论文 §4.2 的口径一致。
    """
    if values is None:
        values = pivot_panel(panel)
    n_entities, n_months = values.shape
    meta = panel.groupby("entity").agg(
        cohort=("cohort", "first"), step_month=("step_month", "first")
    )
    positive = values > 0
    blocks: list[pd.DataFrame] = []
    for a in range(12, n_months - k):
        b_adj = values[:, a - k:a].sum(axis=1)
        forward = values[:, a + 1: a + 1 + k].sum(axis=1)
        b_yoy = values[:, a + 1 - 12: a + 1 + k - 12].sum(axis=1)
        fence = positive[:, a - 11: a + 1].mean(axis=1) >= pos_floor
        keep = fence & (b_adj > 0) & (b_yoy > 0)
        if not keep.any():
            continue
        idx = np.arange(n_entities)[keep]
        blocks.append(
            pd.DataFrame(
                {
                    "entity": idx,
                    "anchor": a,
                    "cal_month": (a % 12) + 1,
                    "B_adj": b_adj[keep],
                    "F": forward[keep],
                    "B_yoy": b_yoy[keep],
                    "cohort": meta["cohort"].to_numpy()[keep],
                    "step_month": meta["step_month"].to_numpy()[keep],
                }
            )
        )
    labels = pd.concat(blocks, ignore_index=True)
    labels["D_adj"] = (labels["F"] <= theta * labels["B_adj"]).astype(int)
    labels["D_yoy"] = (labels["F"] <= theta * labels["B_yoy"]).astype(int)
    # 合成数据才有真值：结果窗口整体落在下滑之后、去年基线整体落在下滑之前
    labels["decline_in_progress"] = (
        (labels["cohort"] == "declining")
        & (labels["anchor"] + 1 >= labels["step_month"])
        & (labels["anchor"] + k - 12 < labels["step_month"])
    )
    return labels


def anchor_month_rate_curve(
    labels: pd.DataFrame, col: str = "D_adj", subset: pd.Series | None = None
) -> tuple[pd.Series, float, float]:
    """论文的第一个诊断量：事件率随锚点日历月的曲线 + 离散度（CV 与 max/min）。

    CV 越大说明"事件率"里混进了越多日历效应。max/min 在出现 0 事件率的月份时无定义，
    此时返回 nan。
    """
    data = labels if subset is None else labels[subset]
    curve = data.groupby("cal_month")[col].mean().reindex(range(1, 13)).fillna(0.0)
    mean_rate = float(curve.mean())
    cv = float(curve.std(ddof=0) / mean_rate) if mean_rate > 0 else 0.0
    ratio = float(curve.max() / curve.min()) if float(curve.min()) > 0 else float("nan")
    return curve, cv, ratio


def aligned_disagreement_share(labels: pd.DataFrame, subset: pd.Series | None = None) -> float:
    """论文的第二个诊断量：相邻标签的事件中，有多少在 yoy 对齐定义下并不成立。"""
    data = labels if subset is None else labels[subset]
    events = data[data["D_adj"] == 1]
    if events.empty:
        return 0.0
    return float((events["D_yoy"] == 0).mean())


def audit_decline_label(panel: pd.DataFrame, k: int = 6, theta: float = 0.5) -> dict:
    """一站式体检：曲线离散度 + 对齐不一致率 + （合成数据才有的）真衰退召回。"""
    labels = build_window_labels(panel, k=k, theta=theta)
    clean = labels["cohort"] == "seasonal_only"
    curve_adj, cv_adj, ratio_adj = anchor_month_rate_curve(labels, "D_adj", clean)
    curve_yoy, cv_yoy, ratio_yoy = anchor_month_rate_curve(labels, "D_yoy", clean)
    truth = labels["decline_in_progress"]
    return {
        "rows": int(len(labels)),
        "entities": int(labels["entity"].nunique()),
        "clean_curve_adj": curve_adj.to_dict(),
        "clean_cv_adj": cv_adj,
        "clean_ratio_adj": ratio_adj,
        "clean_curve_yoy": curve_yoy.to_dict(),
        "clean_cv_yoy": cv_yoy,
        "clean_ratio_yoy": ratio_yoy,
        "clean_fp_adj": float(labels.loc[clean, "D_adj"].mean()),
        "clean_fp_yoy": float(labels.loc[clean, "D_yoy"].mean()),
        "clean_disagreement": aligned_disagreement_share(labels, clean),
        "decline_rows": int(truth.sum()),
        "recall_adj": float(labels.loc[truth, "D_adj"].mean()) if truth.any() else float("nan"),
        "recall_yoy": float(labels.loc[truth, "D_yoy"].mean()) if truth.any() else float("nan"),
    }


def prior_deseasonalized_labels(
    panel: pd.DataFrame, prior_index: Iterable[float], k: int = 6, theta: float = 0.5
) -> pd.DataFrame:
    """出海不足 2 年时的退路：用**品类季节性先验**替代同比基线。

    做法是把月度活跃度除以品类季节指数后再套相邻窗口标签。论文实测这条退路只能部分
    修复：合成队列上 pooled 指数臂的误报率与相邻窗口臂几乎持平，因为峰月不同的实体
    无法被一个全品类共用的指数校正。
    """
    values = pivot_panel(panel)
    n_months = values.shape[1]
    prior = np.asarray(list(prior_index), dtype=float)
    prior = prior / prior.mean()
    scale = np.tile(prior[np.arange(n_months) % 12], (values.shape[0], 1))
    return build_window_labels(panel, k=k, theta=theta, values=values / scale)
```

```python
# ============================ 断言与演示 ============================

_CACHE: dict = {}


def cached_panel() -> pd.DataFrame:
    """测试用：同一份自造面板只生成一次。"""
    if "panel" not in _CACHE:
        _CACHE["panel"] = make_monthly_panel()
    return _CACHE["panel"]


def cached_labels(k: int = 6) -> pd.DataFrame:
    key = f"labels_k{k}"
    if key not in _CACHE:
        _CACHE[key] = build_window_labels(cached_panel(), k=k)
    return _CACHE[key]


def test_adjacent_label_fires_on_purely_seasonal_entities_while_aligned_stays_quiet():
    """纯季节队列上：相邻窗口标签大量误报，同比对齐标签基本不报。"""
    labels = cached_labels(6)
    clean = labels["cohort"] == "seasonal_only"
    fp_adj = float(labels.loc[clean, "D_adj"].mean())
    fp_yoy = float(labels.loc[clean, "D_yoy"].mean())
    assert fp_adj > 0.05, f"相邻窗口标签应大量误报，实测 {fp_adj:.4f}"
    assert fp_yoy < 0.01, f"对齐标签应停在噪声底，实测 {fp_yoy:.4f}"
    assert fp_adj > 20 * max(fp_yoy, 1e-4), (fp_adj, fp_yoy)


def test_aligned_label_flattens_the_anchor_month_rate_curve():
    """anchor-month 事件率曲线：相邻标签剧烈起伏，对齐标签被压平。"""
    labels = cached_labels(6)
    clean = labels["cohort"] == "seasonal_only"
    _, cv_adj, _ = anchor_month_rate_curve(labels, "D_adj", clean)
    _, cv_yoy, _ = anchor_month_rate_curve(labels, "D_yoy", clean)
    assert cv_adj > 0.5, f"相邻标签的曲线离散度应明显，实测 CV={cv_adj:.3f}"
    assert cv_adj > 10 * max(cv_yoy, 1e-6), (cv_adj, cv_yoy)


def test_aligned_disagreement_share_is_total_on_the_seasonal_cohort():
    """纯季节队列上，相邻标签的每一个事件都没有对齐对应物（不一致率=100%）。"""
    labels = cached_labels(6)
    clean = labels["cohort"] == "seasonal_only"
    share = aligned_disagreement_share(labels, clean)
    assert labels.loc[clean, "D_adj"].sum() > 0, "该队列应至少产生一些相邻标签事件"
    assert share > 0.95, f"不一致率应接近 1，实测 {share:.3f}"


def test_aligned_label_keeps_recall_of_genuine_decline():
    """对齐不是"把事件全掐掉"：真衰退（下滑已进行中）的召回反而更高。"""
    labels = cached_labels(6)
    truth = labels["decline_in_progress"]
    assert int(truth.sum()) > 0
    recall_adj = float(labels.loc[truth, "D_adj"].mean())
    recall_yoy = float(labels.loc[truth, "D_yoy"].mean())
    assert recall_yoy > 0.9, f"对齐标签真衰退召回应很高，实测 {recall_yoy:.3f}"
    assert recall_yoy > recall_adj + 0.2, (recall_yoy, recall_adj)


def test_category_prior_only_partially_repairs_the_label():
    """出海不足 2 年的退路（品类季节性先验）：能降误报，但压不平曲线、也回不到 0。"""
    panel = cached_panel()
    prior_labels = prior_deseasonalized_labels(panel, CATEGORY_SEASONAL_INDEX)
    clean = prior_labels["cohort"] == "seasonal_only"
    reference = cached_labels(6)
    ref_clean = reference["cohort"] == "seasonal_only"
    fp_prior = float(prior_labels.loc[clean, "D_adj"].mean())
    fp_yoy = float(reference.loc[ref_clean, "D_yoy"].mean())
    fp_adj = float(reference.loc[ref_clean, "D_adj"].mean())
    _, cv_prior, _ = anchor_month_rate_curve(prior_labels, "D_adj", clean)
    assert fp_prior < fp_adj, (fp_prior, fp_adj)
    assert fp_prior > 20 * max(fp_yoy, 1e-4), f"先验之后仍留有误报，实测 {fp_prior:.4f}"
    assert cv_prior > 0.5, f"先验并未压平曲线，实测 CV={cv_prior:.3f}"


def test_annual_window_label_is_calendar_neutral():
    """k=12（整年窗口）本身即日历中性 —— 代价是检测窗口翻倍、事件定义变了。"""
    _, cv_k6, _ = anchor_month_rate_curve(cached_labels(6), "D_adj",
                                          cached_labels(6)["cohort"] == "seasonal_only")
    labels_k12 = cached_labels(12)
    _, cv_k12, _ = anchor_month_rate_curve(labels_k12, "D_adj",
                                           labels_k12["cohort"] == "seasonal_only")
    assert cv_k12 < cv_k6, (cv_k12, cv_k6)


def main() -> None:
    """演示：一张自造月度面板上，相邻窗口标签 vs 同比对齐标签的体检对比。"""
    panel = make_monthly_panel()
    report = audit_decline_label(panel)
    labels = build_window_labels(panel)
    clean = labels["cohort"] == "seasonal_only"

    print("=" * 72)
    print("季节性对齐标签体检（合成面板：自造数据，非论文数字）")
    print("=" * 72)
    print(f"面板规模：{report['entities']} 个实体 × 48 个月，解出 {report['rows']} 条有效锚点观测")
    print("纯季节队列（真值：无任何衰退）")
    print(f"  相邻窗口标签误报率 = {report['clean_fp_adj']:.4f}")
    print(f"  同比对齐标签误报率 = {report['clean_fp_yoy']:.4f}")
    print(f"  anchor-month 曲线离散度 CV：相邻 {report['clean_cv_adj']:.3f} vs 对齐 {report['clean_cv_yoy']:.3f}")
    print(f"  相邻事件的对齐不一致率 = {report['clean_disagreement']:.3f}")
    print("真衰退队列（水平一次性下滑 -60%，只看「下滑已进行中」的锚点）")
    print(f"  相邻窗口标签召回 = {report['recall_adj']:.4f}")
    print(f"  同比对齐标签召回 = {report['recall_yoy']:.4f}")
    print(f"  可判真值的锚点观测 {report['decline_rows']} 条")

    prior_labels = prior_deseasonalized_labels(panel, CATEGORY_SEASONAL_INDEX)
    prior_clean = prior_labels["cohort"] == "seasonal_only"
    _, cv_prior, _ = anchor_month_rate_curve(prior_labels, "D_adj", prior_clean)
    print("退路：品类季节性先验（代替同比）")
    print(f"  误报率 = {float(prior_labels.loc[prior_clean, 'D_adj'].mean()):.4f}"
          f"（相邻 {report['clean_fp_adj']:.4f}，对齐 {report['clean_fp_yoy']:.4f}）")
    print(f"  曲线离散度 CV = {cv_prior:.3f} —— 误报只是被挪到了别的月份，没有被消除")

    pooled_adj = anchor_month_rate_curve(labels, "D_adj")[1]
    pooled_yoy = anchor_month_rate_curve(labels, "D_yoy")[1]
    print("全队列（生产口径：把两个队列混在一起看）")
    print(f"  曲线离散度 CV：相邻 {pooled_adj:.3f} vs 对齐 {pooled_yoy:.3f}")
    print(f"  对齐不一致率 = {aligned_disagreement_share(labels):.3f}")
    print("=" * 72)
    print("上线动作：把预警标签的基线从「过去 k 个月」换成「去年同期同样的 k 个月」，"
          "再用上面两个诊断量复核曲线是否被压平。")
    print("=" * 72)


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

- **前置技能**
  - `Skill-Cohort-Retention-Analysis.md` —— 本卡的输入就是「实体 × 月份」面板与队列口径，先把面板和留存曲线立起来再谈标签；
  - `Skill-Prophet-Forecasting.md` —— 场景二需要一条月度季节指数，Prophet 的年度季节项可以直接给出 12 个月的指数（注意：这是**替代路径**，论文未测试先验形式）。
- **并列 / 被本卡替换**
  - `Skill-Customer-Churn-Prediction.md`、`Skill-Deep-Learning-Churn-Prediction.md` —— 它们消费的正是本卡要修的标签。标签里混着季节性时，模型 AUC 的一部分是在测「模型学会了多少季节」；先修标签，这两张卡的评估才有意义。
- **延伸技能**
  - `Skill-Uplift-Churn-Prediction.md` —— 清单从 119 缩到 79 之后，省下来的干预容量应该投给**增量最大**而不是**风险最高**的用户，正好是 uplift 卡的输入；
  - `Skill-RFM-Customer-Segmentation.md` —— 在「月龄 × 品类 × 最近购买」上分层，把生命周期造成的自然降级与真实衰退分开看；
  - `Skill-User-Lifecycle-STAN.md` —— 把宝宝月龄当独立维度建模，是缓解「月龄生命周期 vs 季节性」混淆的另一条路（论文未测试，属外推）。
- **组合数据流**：`Cohort 面板` → `本卡：季节性对齐标签` → `RFM/月龄分层` → `Churn / Uplift 模型` → `干预清单`。
  标签错则后面四步全部失真，所以本卡处在数据流的**最上游**。

---

## ⑤ 商业价值评估

- **ROI 预估（容量口径，参数来源逐项标注）**
  记 `L_adj`、`L_yoy` 为相邻/对齐两份清单的长度，`G` 为两份清单共有的真衰退人数，
  `r` 为「成功干预一人」的收益（以单次干预成本为单位）。

  ```text
  每周期节省 = L_adj − L_yoy = 40 个干预位          ← 论文部署系统 119 → 79（参数来源：⑥「模型层与决策层」的原文引文）
  净收益     = G × r − L                            ← 论文给的净收益口径（$Gr-79$ vs $Gr-119$）
  年化货币节省 = (L_adj − L_yoy) × 单账号单次干预成本 × 每年周期数
  其中 单账号单次干预成本 = 运营人均小时成本 × 单次触达耗时   ← 企业内参数，论文未给（单位经济性保密）
  ```

  论文给出的边界情形：`G = 79` 时，`r = 2` 净收益约翻倍（原文：roughly double）、`r = 5` 高 14%、`r = 10` 高 6%。
  **论文没有报告任何货币金额**（原文明确：换算成货币需要每账号毛利与干预成本，而这些是保密的），
  所以本卡也不写货币数字 —— 货币换算需要企业自己的单账号干预成本。
- **实施难度**：⭐☆☆☆☆（1/5）。只改标签定义：同一份特征、同一个学习器、同一个时间留出集，只换目标
  （论文的受控对比就是这么做的）。真正的门槛是**数据历史与围栏**，不是算法；代码量以「行」计。
- **优先级评分**：⭐⭐⭐⭐☆（4/5）。
  - 已有 ≥2 个完整年度月度面板 → 立刻做：一次性改标签 + 一次双诊断，同时拿到「清单更短」与「模型更准」；
  - 历史不足 2 年 → 降级为场景二的过渡方案，优先级 ⭐⭐☆☆☆，因为退路只能部分修复。
- **评估依据**
  1. 论文在同一模型、同一特征、同一时间留出集下，只换标签就让 holdout ROC-AUC 从 0.767 升到 0.864 —— 说明瓶颈在**目标**而不在模型容量（换用更强的表格基础模型 TabPFN，PR-AUC 反而变化 −0.015）；
  2. 清单缩减三分之一直接对应客服/运营的干预容量，是**可被业务方当场核对**的指标；
  3. 两个诊断量（曲线离散度、对齐不一致率）成本极低，可以做成季度例检，防止标签在换品类/换市场后重新漂移。

---

## ⑥ 原文引用

### 机制与命题

> 原文："The construction has a defect: its two windows cover different calendar months. For a seasonal entity the two windows sit at different points of the seasonal cycle. The label therefore fires on seasonal descent and heals on seasonal ascent, whether or not the entity’s trajectory has changed."
> 出处：2608.18174 §1 Introduction

> 原文："For a purely seasonal entity anchored just after its peak, the trailing baseline sums the high season and the outcome window sums the low season. The adjacent ratio then reads 0.5, and the label declares “decay” with no decline anywhere in the series. The year-over-year baseline reads parity."
> 出处：2608.18174 §3.2 Why the adjacent label is calendar-dependent

> 原文："Part (ii) is the mechanism in one line. The twelve anchor-month ratios have geometric mean exactly one. Any profile whose window sums are not all equal therefore places some anchors below parity."
> 出处：2608.18174 §3.2 Why the adjacent label is calendar-dependent

> 原文："Across its 10,426 buyer–month training observations, the decay-event rate ranged from 2.4% at April anchors to 9.3% at October anchors. About half of the decay events dissolved under a seasonally aligned definition."
> 出处：2608.18174 §1 Introduction


### 生产系统与公共面板的实测

> 原文："Raising it from zero through $50K, $100K, and $200K of trailing annual value moves the ratio $1.3\to 2.1\to 2.4\to 3.4$."
> 出处：2608.18174 §5.1 The anchor-month artifact replicates on public panels

> 原文："On the production material cohort (the deployed materiality floor applied to the full panel) the adjacent decay rate troughs at 3.9% in April and peaks at 9.1% in November. That is a max/min ratio of 2.4, with a coefficient of variation of 0.284. The aligned label sits nearly flat between 7.1% and 8.6% (CV 0.054). Half of the adjacent events (50%, interval $[45,54]$) have no aligned counterpart."
> 出处：2608.18174 §5.1 The anchor-month artifact replicates on public panels

> 原文："Seasonality is substantial: 214 of 383 material buyers swing more than 50% from peak to trough within a year."
> 出处：2608.18174 §4.1 Panels

> 原文："The adjacent label’s false-event rate rises from below 0.01% at $a=0$ through 3.0% at $a=0.4$ to 27.3% at $a=0.8$. The aligned label stays at or below 0.03% at every amplitude."
> 出处：2608.18174 §5.2 The synthetic sweep isolates the mechanism

> 原文："On the injected true declines, recall is 0.905 aligned against 0.820 adjacent (Figure 2b)."
> 出处：2608.18174 §5.2 The synthetic sweep isolates the mechanism

> 原文："Pooled across every anchor month, 37–69% of the adjacent label’s decay events on the public panels have no aligned counterpart."
> 出处：2608.18174 §5.3 Pooling does not repair the labels

> 原文："Pooling fixes the composition of the training set without fixing the labels inside it."
> 出处：2608.18174 §5.3 Pooling does not repair the labels

> 原文："An entity–anchor observation is valid when at least 75% of the trailing twelve available months are positive (the public analogue of the production materiality floor)."
> 出处：2608.18174 §4.2 Protocol


### 修正的代价、替代方案与稳健性

> 原文："Pooled indices fail where seasonal phases are heterogeneous — a panel-level index cannot serve entities that peak in different months."
> 出处：2608.18174 §5.5 Alternative remedies, measured

> 原文："On M5 and production, where profiles drift and history is shorter, they leave two to more than three times the aligned label’s dispersion. They also demand a year more history."
> 出处：2608.18174 §5.5 Alternative remedies, measured

> 原文："| Adjacent | 12 | 9.2% | 0.82 | 0.235 | 0.195 | 0.267 |"
> 出处：2608.18174 §5.5 Alternative remedies, measured（Table 2，Adjacent 行）

> 原文："| Deseason. (pooled idx) | 18 | 9.5% | 0.83 | 0.216 | 0.186 | 0.121 |"
> 出处：2608.18174 §5.5 Alternative remedies, measured（Table 2，Deseason. (pooled idx) 行）

> 原文："Among the arms that flatten the curve, only the aligned label keeps the six-month horizon, at tied-lowest history cost."
> 出处：2608.18174 §5.5 Alternative remedies, measured

> 原文："The aligned label fires on 27–41% of following-year anchors against 6–19% for the adjacent label, whose trailing baseline ratchets down with the decline."
> 出处：2608.18174 §5.4 What the disagreement events actually are

> 原文："Judged without precedence, however, 83% of the production disagreements on which the deseasonalized label is defined are removed by seasonal adjustment alone (70% on M5)."
> 出处：2608.18174 §5.4 What the disagreement events actually are

> 原文："On the production material cohort the adjacent growth label’s anchor-month ratio is 2.1, and 30% of its events lack an aligned counterpart."
> 出处：2608.18174 §5.7 Robustness


### 模型层与决策层

> 原文："ROC-AUC rises from 0.767 to 0.864 for the decay direction and from 0.736 to 0.857 for growth. Precision–recall AUC rises from 0.253 to 0.367 and from 0.430 to 0.743."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

> 原文："The caution: the two rows of each pair score different response variables with different prevalences (decay: 5.5% adjacent against 7.0% aligned at training). The comparison therefore does not show one model beating another on a fixed task."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

> 原文："Detecting declines in progress, the aligned-trained model reaches ROC 0.78. The adjacent-trained model scores 0.44, below chance, having learned to rank seasonal descent above genuine decline. The calendar features repair nothing (0.44)."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

> 原文："Swapping learners under the same labels moved holdout skill insignificantly or negatively. The strongest challenger, the TabPFN tabular foundation model (Hollmann et al., 2025), changed precision–recall AUC by $-0.015$ in both directions."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

> 原文："Against its own labels the adjacent-trained model scores 0.96 — a model can look excellent against a defective target."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

> 原文："The served action list is the unit of cost: every flagged account consumes account-manager attention."
> 出处：2608.18174 §6 The correction in practice

> 原文："Relabeling shrank the served “shrinking” classification by a third (119 to 79 accounts)."
> 出处：2608.18174 §6 The correction in practice

> 原文："The adjacent list spends 119 intervention units per cycle to reach what the aligned list reaches with 79. That is a saving of 40 units per cycle at any $G$. In net-benefit terms ($Gr-79$ against $Gr-119$) the advantage is those same 40 units at every $r$."
> 出处：2608.18174 §6 The correction in practice

> 原文："At the boundary case $G=79$ this amounts to roughly double the net benefit at $r=2$, 14% more at $r=5$, and 6% more at $r=10$."
> 出处：2608.18174 §6 The correction in practice

> 原文："The aligned baseline needs the outcome window’s calendar months one year back: roughly $12+k$ months of history before an entity’s first label, against $2k$ for the adjacent construction."
> 出处：2608.18174 §6 The correction in practice

> 原文："A currency translation requires per-account margins and intervention costs that are confidential."
> 出处：2608.18174 §6 The correction in practice


### 复现材料与边界

> 原文："Scripts, the pre-specified analysis plan, and timestamped result artifacts sufficient to reproduce every synthetic and public-panel number accompany this preprint as arXiv ancillary files."
> 出处：2608.18174 Data and code availability

> 原文："Each has multiplicative month-of-year seasonality with a random peak month and amplitude $a\in\{0,0.2,0.4,0.6,0.8\}$ (400 entities per amplitude), lognormal multiplicative noise, and no trend and no true decline."
> 出处：2608.18174 §4.1 Panels

> 原文："First, the controlled model-skill comparison (Table 3) is single-organization. The public panels support the artifact, the mechanism, and the disagreement accounting. The synthetic ground-truth experiment supports the model-level claim. The transfer of the production ROC gains is demonstrated in one system only. Second, our aligned construction assumes annual periodicity with a twelve-month lag. Entities with non-annual cycles, or panels dominated by mobile holidays — the calendar effects that survive year-on-year comparison in official-statistics practice (Eurostat, 2024) — need alignment to the same phase of the relevant cycle. We have not tested that. Third, the aligned baseline replays history. One anomalous prior-year window (a promotion spike, an outage) corrupts up to $k$ of the following year’s labels, where an adjacent baseline would self-heal within $k$ months. Entities are also unlabeled for their first $12+k$ months. That excludes the early-relationship segment where non-contractual defection concentrates, a cost young customer bases bear most. Fourth, the construction we correct is the threshold-ratio label. Inactivity-based churn definitions carry a related seasonal confound (seasonal quiet reads as churn) with a different structure our proposition does not cover. The latent-attrition lineage (Bachmann et al., 2021; Wünderlich et al., 2022) avoids discrete labels altogether. Our results say nothing against those choices. They say that where threshold-ratio window labels are used on seasonal entities, the calendar belongs in the label’s definition and not only in the model’s features. The features-only version of that advice is now measured and found wanting (Section 5.6)."
> 出处：2608.18174 §7 Discussion and limitations（局限 1）

> 原文："Second, our aligned construction assumes annual periodicity with a twelve-month lag. Entities with non-annual cycles, or panels dominated by mobile holidays — the calendar effects that survive year-on-year comparison in official-statistics practice (Eurostat, 2024) — need alignment to the same phase of the relevant cycle. We have not tested that."
> 出处：2608.18174 §7 Discussion and limitations（局限 2）

> 原文："Third, the aligned baseline replays history. One anomalous prior-year window (a promotion spike, an outage) corrupts up to $k$ of the following year’s labels, where an adjacent baseline would self-heal within $k$ months. Entities are also unlabeled for their first $12+k$ months. That excludes the early-relationship segment where non-contractual defection concentrates, a cost young customer bases bear most."
> 出处：2608.18174 §7 Discussion and limitations（局限 3）

> 原文："Fourth, the construction we correct is the threshold-ratio label. Inactivity-based churn definitions carry a related seasonal confound (seasonal quiet reads as churn) with a different structure our proposition does not cover."
> 出处：2608.18174 §7 Discussion and limitations（局限 4）

> 原文："Where strong growth is sustained, the cut is also effectively stricter (synthetic recall 0.70 against the adjacent label’s 0.86)."
> 出处：2608.18174 §7 Discussion and limitations

> 原文："The case alignment truly cannot see is decline followed by stabilization: a current-state question rather than a new-decline question."
> 出处：2608.18174 §6 The correction in practice

> 原文："Of the adjacent-window decay events, 37–69% on the public panels and 28–50% in production have no counterpart under a seasonally aligned definition."
> 出处：2608.18174 §Abstract


---

## 变更记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-09-12 | v1 | 首版：paper 2608.18174 萃取，落位 `06-增长模型/`（K1 PASS / 引文逐字核验 / G2 passed） |
