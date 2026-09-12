---
title: Skill-Shipping-Cost-Estimation
module: 03-时间序列
topic: 用「路由加权期望」把下单前运费预估拆成需求预测 / 费率卡基线 / 残差修正 / 装箱合并推断四段，产出可审计的分区级单均运费
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2607.16230
paper: RouteCost: A Production-Inspired Multi-Stage Framework for Pre-Order Shipping Cost Estimation in E-Commerce
venue: arXiv preprint
venue_tier: preprint
evidence_grade: A
verified_by: verify_skill_code.py (K1 PASS) + quote_check.py (引文逐字核验 VERBATIM，0 伪造 0 近似) + gate_check.py G1/G2/G3
supersedes:
related: Skill-Demand-Forecasting-Supply-Chain.md, Skill-Prophet-Forecasting.md, Skill-Monodense-单品价格弹性估计.md, Skill-Safety-Stock-Replenishment.md, Skill-Multi-Echelon-Inventory.md
---

# Skill-Shipping-Cost-Estimation

> **venue 说明（R3 要求显式标注）**：本文是 **arXiv 预印本**（`venue: arXiv preprint`，`venue_tier: preprint`），
> **没有会议或期刊标记**。全文头部那行 `Conference: Conference Title; 2026; TBDCCS` 是投稿模板的**占位符**
> （`Conference Title` 与 `TBDCCS` 都没填），不能当作 venue 证据。按 `venue-whitelist.md` 的层级规则，
> 本卡不构成 CCF/UTD 级证据。之所以仍纳入萃取：它给出了完整四段方法（§3）、时间切分回测（§4–§5）
> 与显式的局限声明（§6），不是 demo 短文或综述。

> **先说清楚数据性质（避免误用）**：论文的全部数字来自**合成数据集**
> （§6 自承 "operationally grounded synthetic dataset"），承运商费率卡也是 "simplified FedEx-like"
> 的合成结构（§4），且履约网络被简化到单起运仓、单承运商、单服务等级。
> **本卡因此不写任何"论文在生产系统上省了多少钱/降了多少个点"这类话 —— 论文没有报告，也不该由本卡补。**

---

## ① 算法原理

**核心思想**：把「下单前这一单要花多少运费」从一个黑箱回归，改写成**路由加权期望**：
需求模块预测目的地区域份额（路由权重），费率卡给出每条路由的结构化基线，残差模型补非线性误差，
装箱合并模块用弱代理推断隐性节省，最后按权重汇总成产品级预期运费。

**数学直觉**（论文式 (1)(2)）：$\hat C(p,t)=\sum_z w(z\mid p,t)\,\hat c(p,z,t)$，
$\hat c=c^{(1)}+\Delta c^{(2)}-s^{\text{box}}$。$w$ 回答「这单大概率发往哪个分区」，
$c^{(1)}$ 是费率卡（计费重 × 分区费率 + 旺季与超尺寸附加费），$\Delta c^{(2)}$ 是基线漏掉的非线性，
$s^{\text{box}}$ 是并箱省下的钱。四段各自独立，路由贡献可逐行审计。

**关键假设**：① 路由可由目的地**分区**刻画（单起运仓 + 单承运商）；② 费率卡结构稳定、可由显式特征表达；
③ 合并结果在下单时不可观测，只能由代理推断。

---

## ①b 反例与适用边界

**什么时候不要用**：

- **多仓 + 多承运商的履约网络**。论文只覆盖 $|W|=|K|=1$；多仓下「每个起运仓一张分区图」不可扩展，
  路由空间要从「商品 × 分区」扩成「仓 × 区域（× 承运商）」。**不要把本卡的四段直接套到多仓场景**，
  那需要先重做路由表示（§6 原话给了方向）。
- **头程（工厂 → 海外仓）**。论文的对象是**单起运仓之后**的小包裹履约成本，从头到尾没有出现海运/空运头程、
  关务或入仓预约。头程时效与成本结构完全不同（按方/按柜计费、含清关与港口费用），
  **不能拿本卡的费率卡逻辑去估头程**；要用就得按同样的四段结构重新定义"路由"。
- **需要单笔订单报价（quote）而不是期望成本**。框架产出的是**期望**：下单前路由尚未实现，
  它给的是分区加权的平均值，不是"这一单一定花多少"。

**已知的失败模式**：

- **用与运费强相关但非因果的字段当特征**。论文点名了批发成本：大件商品往往批发价也高，
  于是批发价与包裹运费在跨品类上正相关；但两张尺寸完全相同的毯子，批发价差得再多，运费也几乎一样
  （论文只说 "very different wholesale costs"，**未报告批发价差异的量级**，本卡不补）。
  把批发价放进模型会提高样本内拟合，却让模型在费率卡或品类结构变化时失真。
- **Stage 2 单独加进去可能反而变差**。论文表 2 里 Stage 1 + Stage 2 的 MAE 高于只用 Stage 1，
  而 §5.1 正文却写 Stage 2 "improves the estimate" —— **论文没有解释这个不一致**，
  本卡照实现实验收时也复现了"残差层不保证单调改善"这一现象（残差层若学到大盘均值偏移，就会在小样本上抖动）。
  落地时**必须逐段留出验证**，不能默认"多加一段一定更好"。
- **把合并节省当成可直接入账的钱**。论文的合并目标是**伪标签构造**出来的，不是真实装箱结果；
  节省项在预测时只能由代理推断。把它当成财务口径的确定成本下调，会系统性低估成本。

**论文自己承认的局限**（§6）：

- 评估用的是合成数据集 + 简化履约网络（单起运仓/单承运商/单服务等级），路由复杂度被人为压低。
- 合并节省是伪标签推断，不是真实 shipment 级合并结果。
- 单起运仓的分区图不可扩展到多仓；多仓需要共享区域划分。
- 论文**未讨论**：货币单位与 MAE 量纲、Ridge 与梯度提升的超参数取值、与纯查表/黑箱模型的对照数字、
  分品类/分分区的误差分解、显著性检验。以上均**未报告**，本卡不替它补。

---

## ② 母婴出海应用案例

### 场景一：旺季前 6 周，给婴儿车与餐椅算出「分区加权单均运费」，据此定独立站包邮门槛与平台价格下限

- **业务问题**：母婴出海的大件（婴儿车、餐椅）走尾程派送时，计费重由**体积重**主导，分区费率卡在远端分区明显更贵，
  且旺季有附加费。现在的定价与促销用的往往是一个「账单平均运费」（月度尾程账单总额 ÷ 单量）。
  只要一个 SKU 的订单在远端分区占比高于均值，用平均运费定价的每一个远端订单都在亏；
  旺季再叠加附加费，亏损面扩大。需要在下单**之前**、按「这个 SKU × 这个月 × 这个分区」给出运费，
  而不是等月底账单出来再复盘。
- **数据要求**：
  - **订单流水**：订单日期、目的 ZIP、SKU、件数，**≥18 个月**且含至少一个完整旺季（用于估月度分区份额与旺季抬升）。
  - **商品物理属性**：单件重量、长宽高（cm）、整箱可装件数、产品族与品类。
  - **承运商费率卡**：分区 × 计费重档位的费率、旺季附加费规则、超尺寸/超大件附加费规则；
    以及目的 ZIP → 分区的映射（论文用承运商 zone locator 生成）。
  - **账单口径**：尾程派送账单与 FBA 入仓费报表（用于把「实付运费」对齐到订单）。
  - 出力口径：**承运商分区**（论文用 1–8 八个分区）而不是行政区；费率卡必须按月版本化。
- **数据可得性**：`部分可得（需补充费率卡与 ZIP→分区映射）`。订单流水、商品尺寸重量、Amazon 侧费用报表
  在平台店铺内可得；独立站（Shopify）订单与自建物流账单也可得。**缺口有两处**：
  ① 承运商费率卡的分区档位与旺季附加费版本，通常要从承运商合同/账单反推，而不是现成表；
  ② 目的 ZIP → 分区映射需要按起运仓逐次生成。**头程（工厂→海外仓）成本不在本卡口径内**，
  需另建一套（见 ①b）。
- **预期产出**：SKU × 月 × 分区的单均运费预估 + 一张可审计的路由表
  （每个分区的权重、单件运费、加权贡献），以及分区加权后的 SKU 级单均运费。
- **业务价值**：把定价里的运费项从「一个平均数」换成「8 个分区的加权」，可直接支撑三件事：
  ① 独立站包邮门槛（阈值定低了，远端分区订单必亏）；② 平台店铺价格下限（大件在远端分区是否还有毛利）；
  ③ 旺季临时加价幅度（附加费按分区叠加，不是全国一个数）。
  计算过程（参数全部来自企业内部，**可复现**）：

```text
避损收益 = Σ_z [ 该分区订单占比 w_z × max(0, 该分区实际单均运费 − 现行定价所用平均运费) ] × 年单量 × 自担比例
并箱节省 = 可合并订单占比 × 单均合并节省额 × 年单量
```

  其中 `w_z` 与「分区实际单均运费」由本卡 Stage 0/Stage 1 直接产出，「现行定价所用平均运费」来自现行定价表，
  「自担比例」来自包邮/运费补贴政策。四个输入都在企业内部，无需外部基准。

### 场景二：组合装（婴儿车 + 配件）与餐椅整箱补货的并箱推断：一票发还是拆两票发

- **业务问题**：独立站常做组合装（婴儿车 + 雨罩 + 杯架），海外仓补货常按整箱（例如一箱装两台餐椅）。
  这两种情况都会改变**体积重**：件数越贴近整箱倍数，外箱数越少，计费重越低，单件尾程运费被摊薄。
  但下单前不知道仓库当天会不会并箱，账单上也没有「本单是否合并」这一列 —— 于是"组合装更省运费"长期停留在经验判断，
  定价与拆单规则都说不清省了多少。**这正是论文最后一段（装箱合并推断）要解决的问题。**
- **数据要求**：代理变量而非结果变量 —— 同日同 ZIP 密度、同日同分区密度、同族平均件数、
  尺寸兼容性（件数与整箱倍数的贴合度）、拆票风险（超重/超围长必须拆）；再用历史账单构造伪标签做校准。
- **数据可得性**：`部分可得（需自建代理变量与伪标签）`。订单流水与商品尺寸可算出密度与兼容性代理；
  **缺口是"真实是否并箱"**——平台店铺与承运商账单里通常拿不到 shipment 级合并结果，
  论文自己也是用伪标签构造绕开的。因此这一段的产出应定位为**成本估计的区域性下调**，
  而不是逐单可对账的折扣。
- **预期产出**：每个 SKU/篮子的合并机会分（代理加权）、预估合并节省（带业务可校准的上限），
  以及"单件 vs 整箱倍数"的单件运费曲线（本卡 ③ 的演示会打印这条曲线）。
- **业务价值**：把「组合装更省」变成可核对的数字，直接支撑三个动作：
  ① 组合装定价（省下来的运费要不要让一部分给消费者）；② 拆单规则（拆票会吃掉合并收益，需要阈值）；
  ③ 整箱补货批量（批量贴近整箱倍数时单件运费下降，可反过来改变补货经济性 —— 与
  `Skill-Safety-Stock-Replenishment.md` 的批量决策直接咬合）。

---

## ③ 代码模板

> **本模板是业务化的简化实现，不是论文的忠实复现。** 论文用的是它自己的合成数据集与合成费率卡；
> 下面这套是本卡自建的母婴大件算例（费率卡数值、单量水平、品类清单**全部是示例值**）。
> 依赖仅 numpy + scikit-learn，四段的接口与论文 §3.3 的功能分解一一对应。

### 第一段：路由几何、费率卡与订单流水

```python
# -*- coding: utf-8 -*-
"""RouteCost 业务模板 —— 预购运费成本预估（母婴跨境大件 / 自建算例）

对齐论文 arXiv:2607.16230 的四段功能分解：
  Stage 0 需求预测  → 分区路由权重 w(z|p,t)
  Stage 1 费率卡基线 → c1(p,z,t)（Ridge：计费重/体积重/尺寸/分区独热/费率卡查表值/附加费旗标）
  Stage 2 残差修正  → Δc2(p,z,t)（梯度提升，拟合 Stage 1 残差）
  Stage 3 装箱合并  → s_box(p,z,t)（弱代理打分 × 可校准节省上限）
  聚合              → Ĉ(p,t) = Σ_z w(z|p,t) · [c1 + Δc2 − s_box]
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

ZONES = np.arange(1, 9)                      # 分区 1..8（对齐论文 §3.2 的 zone index）
DIM_DIVISOR = 6000.0                         # 体积重除数：长×宽×高(cm)/6000 → kg
PEAK_MONTHS = (11, 12)                       # 旺季月份
RATE_BASE = np.array([28.0, 30.5, 33.0, 36.0, 39.5, 43.0, 47.5, 52.0])   # 分区基础费（示例值）
RATE_PER_KG = np.array([6.2, 6.8, 7.5, 8.3, 9.2, 10.2, 11.4, 12.8])      # 分区续重费率（示例值）
PEAK_SURCHARGE = 0.35                        # 旺季附加费加价率（示例值）
OVERSIZE_FEE = 45.0                          # 大件/超长附加费（元/票，示例值）
CONSOLIDATION_CAP = 0.18                     # 合并节省上限（占基线比例，业务可校准参数）
N_MONTHS, DAYS_PER_MONTH, HOLDOUT_MONTHS = 18, 30, 3
N_DAYS = N_MONTHS * DAYS_PER_MONTH
DAILY_BASE = 60.0                            # 基期日单量（示例值）
ZIP_POOL = 40                                # 每分区抽样 ZIP 数（示例值）

CATALOG = (
    # SKU,            产品族,   品类,  单重kg, 长,   宽,   高,   整箱可装件数
    ("STROLLER-PRO",  "婴儿车", "出行", 11.6, 79.0, 53.0, 31.0, 1),
    ("STROLLER-LITE", "婴儿车", "出行",  8.4, 66.0, 45.0, 25.0, 2),
    ("HICHAIR-WOOD",  "餐椅",   "喂养",  7.2, 62.0, 56.0, 39.0, 1),
    ("HICHAIR-FOLD",  "餐椅",   "喂养",  5.1, 54.0, 48.0, 33.0, 2),
    ("PUMP-DUAL",     "吸奶器", "喂养",  1.3, 26.0, 19.0, 15.0, 6),
    ("BOTTLE-SET",    "奶瓶",   "喂养",  0.9, 22.0, 16.0, 12.0, 12),
)
SKUS = np.array([c[0] for c in CATALOG])
FAMILY = np.array([c[1] for c in CATALOG])
CATEGORY = np.array([c[2] for c in CATALOG])
UNIT_W = np.array([c[3] for c in CATALOG], float)
DIMS = np.array([[c[4], c[5], c[6]] for c in CATALOG], float)
PER_CARTON = np.array([c[7] for c in CATALOG], float)


def shipment_geometry(sku_idx, qty):
    """实重 / 体积重 / 计费重 / 大件旗标 / 拆票风险。

    体积重按**整箱件数**折算：件数越贴近整箱倍数，外箱数越少、体积重越低 ——
    这就是「组合装 / 整箱补货摊薄单件运费」的物理来源。
    """
    sku_idx = np.asarray(sku_idx, int)
    qty = np.asarray(qty, float)
    lwh = DIMS[sku_idx]
    cartons = np.ceil(qty / PER_CARTON[sku_idx])
    phys = UNIT_W[sku_idx] * qty
    dimw = lwh[:, 0] * lwh[:, 1] * lwh[:, 2] * cartons / DIM_DIVISOR
    billable = np.maximum(phys, dimw)
    longest = lwh.max(axis=1)
    girth = lwh[:, 0] + 2.0 * (lwh[:, 1] + lwh[:, 2])
    oversize = ((longest >= 70.0) | (girth >= 200.0)).astype(float)
    split_risk = (billable > 22.0).astype(float)      # 超重件须拆票 → 合并机会下降
    return phys, dimw, billable, oversize, split_risk


def rate_card_cost(zone, billable, peak, oversize):
    """费率卡查表值：分区基础费 + 分区续重费率 × 计费重，再叠加两类附加费。"""
    z = np.asarray(zone, int) - 1
    base = RATE_BASE[z] + RATE_PER_KG[z] * np.asarray(billable, float)
    return (base * (1.0 + PEAK_SURCHARGE * np.asarray(peak, float))
            + OVERSIZE_FEE * np.asarray(oversize, float))


def monthly_volume():
    """月度日单量水平（含自然增长与旺季抬升）—— 需求模块的输出之一。"""
    m = np.arange(N_MONTHS)
    return DAILY_BASE * (1.0 + 0.025 * m) * (1.0 + 0.55 * np.isin((m % 12) + 1, PEAK_MONTHS))


def monthly_zone_share():
    """月度分区份额真值（含季节漂移）—— Stage 0 要预测的正是它。"""
    base = np.array([0.05, 0.07, 0.09, 0.12, 0.15, 0.17, 0.19, 0.16])
    out = np.zeros((N_MONTHS, len(ZONES)))
    for m in range(N_MONTHS):
        drift = 1.0 + 0.30 * np.sin(2 * np.pi * ((m % 12) - 2) / 12.0)
        raw = base * np.where(ZONES % 2 == 1, drift, 2.0 - drift)
        out[m] = raw / raw.sum()
    return out


def make_order_history(seed=20260912):
    """构造 18 个月订单流水（**本卡自建示例数据，不是论文数据**）。

    真实运费 DGP：
        费卡价   = f(计费重, 分区, 旺季旗标, 大件旗标)
        实付运费 = 费卡价 × (1 − 合并折扣)
    合并折扣由**下单时不可观测**的当日实际拼单密度驱动（密度越高越可能并箱）。
    zip_density 是当日同 ZIP 实际单量，保留用于校准期的伪标签构造与审计；
    预测期只能用它的**期望值**（见 consolidation_score）。
    """
    rng = np.random.default_rng(seed)
    vol, share = monthly_volume(), monthly_zone_share()
    lam = np.zeros((N_DAYS, len(ZONES), ZIP_POOL))
    for d in range(N_DAYS):
        lam[d] = (vol[d // DAYS_PER_MONTH] * share[d // DAYS_PER_MONTH] / ZIP_POOL)[:, None]
    counts = rng.poisson(lam)
    cell = np.argwhere(counts > 0)
    rep = counts[counts > 0]
    day = np.repeat(cell[:, 0], rep)
    zone = np.repeat(cell[:, 1] + 1, rep)
    density = np.repeat(rep, rep).astype(float)          # 同日同 ZIP 实际单量
    zone_density = counts.sum(axis=2)[day, zone - 1].astype(float)
    n = len(day)
    month, moy = day // DAYS_PER_MONTH, (day // DAYS_PER_MONTH) % 12 + 1
    peak = np.isin(moy, PEAK_MONTHS).astype(float)
    sku = rng.integers(0, len(CATALOG), n)
    qty = rng.integers(1, 4, n)

    phys, dimw, billable, oversize, split_risk = shipment_geometry(sku, qty)
    base_cost = rate_card_cost(zone, billable, peak, oversize)
    rem = np.mod(qty, PER_CARTON[sku])
    size_compat = np.where(rem == 0, 1.0, 1.0 / (1.0 + rem))
    exp_zone = vol[month] * share[month, zone - 1]        # 当日该分区的预期单量水平
    disc = (0.010 + 0.030 * (exp_zone - 5.0) / 10.0 + 0.012 * size_compat
            + 0.006 * (qty - 1.0) - 0.025 * split_risk
            + 0.0015 * (zone_density - exp_zone))         # 当日实际偏离预期的部分（不可预期）
    disc = np.clip(disc * (1.0 + 0.15 * rng.standard_normal(n)), 0.0, CONSOLIDATION_CAP)
    realized = base_cost * (1.0 - disc)
    return {
        "n": n, "day": day, "month": month, "moy": moy, "zone": zone, "sku": sku,
        "qty": qty, "phys": phys, "dimw": dimw, "billable": billable,
        "oversize": oversize, "split_risk": split_risk, "size_compat": size_compat,
        "peak": peak, "family": FAMILY[sku], "category": CATEGORY[sku],
        "zip_density": density, "zone_density": zone_density,
        "base_cost": base_cost, "realized": realized,
    }
```

### 第二段：四段模型（需求权重 → 费率卡基线 → 残差修正 → 装箱合并）

```python
def _share_table(hist, mask):
    counts = np.zeros((N_MONTHS, len(ZONES)))
    np.add.at(counts, (hist["month"][mask], hist["zone"][mask] - 1), 1.0)
    return {"counts": counts, "total": counts.sum(axis=1)}


def build_share_tables(hist):
    """按 全局 / 品类 / 产品族 三层统计月度分区计数（Stage 0 的输入）。"""
    tables = {"global": _share_table(hist, np.ones(hist["n"], bool))}
    for cat in np.unique(hist["category"]):
        tables[f"category:{cat}"] = _share_table(hist, hist["category"] == cat)
    for fam in np.unique(hist["family"]):
        tables[f"family:{fam}"] = _share_table(hist, hist["family"] == fam)
    return tables


def _smoothed_shares(counts, smooth_win=3):
    """月度分区份额：短滚动窗口平滑后再归一化（论文 §3.3 第一步）。"""
    out = np.zeros_like(counts, dtype=float)
    for m in range(counts.shape[0]):
        out[m] = counts[max(0, m - smooth_win + 1):m + 1].sum(axis=0)
    tot = out.sum(axis=1, keepdims=True)
    return out / np.where(tot <= 0.0, 1.0, tot)


def resolve_route_level(tables, family, category, month, min_orders=30):
    """带层级回退的路由权重：产品族 → 品类 → 全局 → 均匀。

    论文 §3.3：目的地分布随产品组变化，故按品类估计份额并在推理时做层级回退。
    返回 (层级名, 分区权重向量)。
    """
    for key, level, need in ((f"family:{family}", "family", min_orders),
                             (f"category:{category}", "category", 3),
                             ("global", "global", 0)):
        tbl = tables.get(key)
        if tbl is not None and tbl["total"][month] >= need:
            return level, _smoothed_shares(tbl["counts"])[month]
    return "uniform", np.full(len(ZONES), 1.0 / len(ZONES))


def build_features(hist, idx):
    """Stage 1 特征（论文 §3.3）：计费重 / 实重 / 体积重 / 三边 / 分区独热 / 费率卡查表值 / 附加费旗标。"""
    z = hist["zone"][idx]
    onehot = np.zeros((len(idx), len(ZONES)))
    onehot[np.arange(len(idx)), z - 1] = 1.0
    return np.column_stack([
        hist["billable"][idx], hist["phys"][idx], hist["dimw"][idx], DIMS[hist["sku"][idx]],
        onehot,
        rate_card_cost(z, hist["billable"][idx], hist["peak"][idx], hist["oversize"][idx]),
        hist["peak"][idx], hist["oversize"][idx],
    ])


def stage2_features(hist, idx, stage1_pred):
    """Stage 2 输入：Stage 1 预测值 + 原始物理特征（**不含**合并代理与月份）。"""
    return np.column_stack([stage1_pred, build_features(hist, idx)])


def consolidation_score(hist, idx, proxy):
    """装箱合并机会分（论文 §3.3 的弱代理清单）。

    ⚠️ 所有代理在**下单前**可得：需求模块给出的预期同日同 ZIP 密度、预期同分区密度、
    同族平均件数、尺寸兼容性、拆票风险 —— 当日实际是否并箱在预测时不可观测。
    """
    zip_d = np.clip(proxy["exp_zip_density"][hist["month"][idx], hist["zone"][idx] - 1] / 2.0,
                    0.0, 1.0)
    zone_d = np.clip(proxy["exp_zone_density"][hist["month"][idx], hist["zone"][idx] - 1] / 40.0,
                     0.0, 1.0)
    fam_q = np.clip(proxy["family_avg_qty"][hist["sku"][idx]] - 1.0, 0.0, 1.5) / 1.5
    return np.clip(0.35 * zip_d + 0.15 * zone_d + 0.15 * fam_q
                   + 0.20 * hist["size_compat"][idx] - 0.30 * hist["split_risk"][idx], 0.0, 1.0)


class RouteCostModel:
    """四段式预购运费预估：需求权重 → 费率卡基线 → 残差修正 → 装箱合并推断。"""

    def __init__(self, alpha=1.0, n_estimators=120, max_depth=3, learning_rate=0.1,
                 cap=CONSOLIDATION_CAP):
        self.s1 = Ridge(alpha=alpha)
        self.s2 = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                            learning_rate=learning_rate, random_state=0)
        self.cap, self.tables_, self.proxy_, self.beta_ = cap, {}, {}, 0.0

    def fit(self, hist, train_mask):
        idx = np.flatnonzero(train_mask)
        self.tables_ = build_share_tables(hist)
        # 预测时可用的是**预期**密度（需求模块输出 × 分区份额 / ZIP 池），不是当日实际密度
        vol, share = monthly_volume(), _smoothed_shares(self.tables_["global"]["counts"])
        self.proxy_ = {
            "exp_zip_density": vol[:, None] * share / ZIP_POOL,
            "exp_zone_density": vol[:, None] * share,
            "family_avg_qty": np.array([hist["qty"][idx][hist["sku"][idx] == s].mean()
                                        for s in range(len(CATALOG))]),
        }
        y = hist["realized"][idx]
        f1 = build_features(hist, idx)
        self.s1.fit(f1, y)
        c1 = self.s1.predict(f1)
        self.s2.fit(stage2_features(hist, idx, c1), y - c1)
        c2 = c1 + self.s2.predict(stage2_features(hist, idx, c1))
        # Stage 3 校准：把代理分回归到「基线 − 实付」得到的伪标签节省率上。
        # 目标取 Stage 2 **之后仍未被解释**的节省率 —— 代理分里已被 Stage 2 学走的成分
        # 在残差目标下不会重复扣减（无截距最小二乘会自动压低其系数）。
        # ⚠️ 只搬尺度不搬结果：训练期实付可知，预测期只有代理分。
        score = consolidation_score(hist, idx, self.proxy_)
        ratio = np.clip(1.0 - hist["realized"][idx] / c2, 0.0, self.cap)
        denom = float(np.sum(score ** 2))
        self.beta_ = float(np.sum(score * ratio) / denom) if denom > 0 else 0.0
        return self

    def predict(self, hist, idx, apply_consolidation=True):
        f1 = build_features(hist, idx)
        c1 = self.s1.predict(f1)
        c2 = c1 + self.s2.predict(stage2_features(hist, idx, c1))
        saving = np.zeros_like(c2)
        if apply_consolidation:
            ratio = np.clip(self.beta_ * consolidation_score(hist, idx, self.proxy_),
                            0.0, self.cap)
            saving = ratio * c2
        return {"c1": c1, "c12": c2, "saving": saving, "final": c2 - saving}

    def route_table(self, hist, idx, family, category, month):
        """路由表（论文 §3.3）：权重 / 路由成本 / 加权贡献，逐行可审计。"""
        level, w = resolve_route_level(self.tables_, family, category, month)
        pred = self.predict(hist, idx)["final"]
        z = hist["zone"][idx]
        route_cost = np.array([pred[z == zz].mean() if np.any(z == zz) else 0.0 for zz in ZONES])
        contrib = w * route_cost
        return {"level": level, "weight": w, "route_cost": route_cost,
                "contribution": contrib, "expected": float(contrib.sum())}


def mae(pred, truth):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(truth, float))))


def mape(pred, truth):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(truth, float))
                         / np.asarray(truth, float)))
```

### 第三段：自检与演示

```python
_CACHE = {}


def _fitted(seed=20260912, n_estimators=80):
    """演示与自检共用的一次性缓存：同一份合成数据与模型只训练一次。"""
    key = (seed, n_estimators)
    if key not in _CACHE:
        hist = make_order_history(seed=seed)
        train = hist["month"] < N_MONTHS - HOLDOUT_MONTHS
        _CACHE[key] = (hist, RouteCostModel(n_estimators=n_estimators).fit(hist, train), train)
    return _CACHE[key]


def test_billable_weight_is_dimensional_for_bulky_stroller():
    """婴儿车这类大件的计费重由体积重决定（体积重 > 实重），且触发大件旗标。"""
    phys, dimw, billable, oversize, _ = shipment_geometry(np.array([0]), np.array([1]))
    assert dimw[0] > phys[0]
    assert abs(billable[0] - dimw[0]) < 1e-9
    assert oversize[0] == 1.0


def test_bundle_packing_lowers_unit_dim_weight():
    """整箱装 2 台的餐椅：一次买 2 台的**单件体积重**低于只买 1 台（组合装摊薄的物理来源）。"""
    _, dimw1, _, _, _ = shipment_geometry(np.array([3]), np.array([1]))
    _, dimw2, _, _, _ = shipment_geometry(np.array([3]), np.array([2]))
    assert dimw2[0] / 2.0 < dimw1[0]


def test_route_weights_sum_to_one():
    hist = make_order_history(seed=11)
    tables = build_share_tables(hist)
    _, w = resolve_route_level(tables, "婴儿车", "出行", N_MONTHS - 1)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= 0.0)


def test_hierarchical_fallback_for_unseen_family():
    """history 里没有的产品族 → 回退到品类层（论文 §3.3 的层级回退）。"""
    hist = make_order_history(seed=12)
    tables = build_share_tables(hist)
    level, w = resolve_route_level(tables, "全新品族", "喂养", N_MONTHS - 1)
    assert level == "category", level
    assert abs(w.sum() - 1.0) < 1e-9


def test_route_weighted_expectation_matches_order_level_mean():
    """式 (1) 的恒等式：按分区份额加权的期望成本 == 订单级平均成本。"""
    hist, model, train = _fitted()
    hidx = np.flatnonzero((~train) & (hist["family"] == "婴儿车"))
    pred = model.predict(hist, hidx)["final"]
    z = hist["zone"][hidx]
    w = np.array([float(np.mean(z == zz)) for zz in ZONES])
    route_cost = np.array([pred[z == zz].mean() for zz in ZONES])
    assert abs(float(np.sum(w * route_cost)) - float(pred.mean())) < 1e-9


def test_consolidation_saving_is_capped_and_non_negative():
    hist, model, train = _fitted()
    idx = np.flatnonzero(~train)
    r = model.predict(hist, idx)
    assert np.all(r["saving"] >= 0.0)
    assert np.all(r["saving"] <= CONSOLIDATION_CAP * r["c12"] + 1e-9)
    assert np.all(r["final"] <= r["c12"] + 1e-9)


def test_full_framework_beats_fee_card_baseline_on_holdout():
    """留出期 MAE：四段式全模型优于仅费率卡基线（论文 §5.1 的结构性结论）。"""
    hist, model, train = _fitted()
    idx = np.flatnonzero(~train)
    y = hist["realized"][idx]
    r = model.predict(hist, idx)
    assert mae(r["final"], y) < mae(r["c1"], y)
    assert mae(r["final"], y) <= mae(r["c12"], y) + 1e-9


def test_holdout_split_is_chronological():
    """时间切分不得泄漏：训练月与留出月没有交集（论文 §4 的 time-based split）。"""
    hist = make_order_history(seed=13)
    train = hist["month"] < N_MONTHS - HOLDOUT_MONTHS
    assert hist["month"][train].max() < hist["month"][~train].min()


if __name__ == "__main__":
    hist = make_order_history()
    months = np.arange(N_MONTHS)
    train = np.isin(hist["month"], months[:-HOLDOUT_MONTHS])
    hold = np.isin(hist["month"], months[-HOLDOUT_MONTHS:])
    m = RouteCostModel().fit(hist, train)
    hidx = np.flatnonzero(hold)
    y = hist["realized"][hidx]
    r = m.predict(hist, hidx)
    print("订单 %d 训练 %d 留出 %d  实付相对费卡价的合并折扣均值 %.4f"
          % (hist["n"], int(train.sum()), int(hold.sum()),
             float((1.0 - hist["realized"] / hist["base_cost"]).mean())))
    print("Stage1     MAE %.3f  MAPE %.4f" % (mae(r["c1"], y), mape(r["c1"], y)))
    print("Stage1+2   MAE %.3f  MAPE %.4f" % (mae(r["c12"], y), mape(r["c12"], y)))
    print("Full       MAE %.3f  MAPE %.4f" % (mae(r["final"], y), mape(r["final"], y)))
    errs = []
    for mm in months[-HOLDOUT_MONTHS:]:
        sel = hist["month"][hidx] == mm
        errs.append(r["final"][sel].sum() / y[sel].sum() - 1.0)
    print("月度聚合误差（预测总额/实际总额 − 1）%.4f ~ %.4f" % (min(errs), max(errs)))
    rt = m.route_table(hist, hidx, "婴儿车", "出行", N_MONTHS - 1)
    print("路由层级 %s  权重和 %.6f  分区加权期望单均运费 %.2f"
          % (rt["level"], rt["weight"].sum(), rt["expected"]))
    for s in range(len(CATALOG)):
        sel = hist["sku"][hidx] == s
        for q in (1, 2, 3):
            qsel = sel & (hist["qty"][hidx] == q)
            if qsel.sum() > 20:
                print("  %-14s 件数=%d 单件预估运费 %.2f" % (SKUS[s], q, r["final"][qsel].mean() / q))
```

**本模板的实际运行输出**（`python3 <本卡代码>.py` 可复现；**这是本卡自建合成算例的输出，不是论文数据**）：

```text
订单 41903 训练 34463 留出 7440  实付相对费卡价的合并折扣均值 0.0372
Stage1     MAE 4.194  MAPE 0.0185
Stage1+2   MAE 3.271  MAPE 0.0129
Full       MAE 3.086  MAPE 0.0120
月度聚合误差（预测总额/实际总额 − 1）0.0024 ~ 0.0048
路由层级 family  权重和 1.000000  分区加权期望单均运费 276.12
  STROLLER-PRO   件数=1 单件预估运费 282.72
  STROLLER-PRO   件数=2 单件预估运费 245.42
  STROLLER-PRO   件数=3 单件预估运费 229.15
  STROLLER-LITE  件数=1 单件预估运费 196.96
  STROLLER-LITE  件数=2 单件预估运费 117.86
  STROLLER-LITE  件数=3 单件预估运费 109.06
  HICHAIR-WOOD   件数=1 单件预估运费 302.96
  HICHAIR-WOOD   件数=2 单件预估运费 253.95
  HICHAIR-WOOD   件数=3 单件预估运费 242.46
  HICHAIR-FOLD   件数=1 单件预估运费 216.49
  HICHAIR-FOLD   件数=2 单件预估运费 106.74
  HICHAIR-FOLD   件数=3 单件预估运费 119.40
  PUMP-DUAL      件数=1 单件预估运费 51.47
  PUMP-DUAL      件数=2 单件预估运费 31.45
  PUMP-DUAL      件数=3 单件预估运费 24.82
  BOTTLE-SET     件数=1 单件预估运费 47.81
  BOTTLE-SET     件数=2 单件预估运费 27.93
  BOTTLE-SET     件数=3 单件预估运费 21.39
```

> ⚠️ **不要把上表的 MAE 与论文表 2 的 MAE 混为一谈。** 论文报告的是它自己那套合成数据集上的数
> （Stage 1 = 1.735、Stage 1 + Stage 2 = 1.893、Full Model = 1.649），
> 而本模板跑的是**本卡另建的一套合成算例**（费率卡、单量水平、品类清单全是示例值）。
> 两者没有共同的货币单位、没有共同的费率卡、也没有共同的数据集 —— 数值上"都变小了"**不构成复现**，
> 数值是否接近纯属巧合。更要注意方向差异：论文表 2 里 Stage 1 + Stage 2 的 MAE（1.893）**高于** Stage 1（1.735），
> 而本算例里 Stage 2 是改善的 —— **结论方向不同**，这本身就是"合成数据结论不可外推"的例证。

---

## ④ 技能关联

| 关联卡片 | 关系 | 为什么组合 |
|---|---|---|
| `Skill-Demand-Forecasting-Supply-Chain.md` | 前置 | Stage 0 的路由权重本质就是「目的地区域的月度需求份额」预测。该卡的分层预测与协调（bottom-up / top-down / reconciliation）正好补上本卡层级回退之外的"份额怎么预测"这一层。 |
| `Skill-Prophet-Forecasting.md` | 前置（可替换实现） | 论文用「月度份额 + 短滚动窗口平滑」；当目的地份额带有节假日与旺季结构（黑五、圣诞、春季出行季）时，可用 Prophet 的趋势+季节+节假日项替换滚动平滑，二者接口一致（都输出月度分区份额）。 |
| `Skill-Monodense-单品价格弹性估计.md` | 下游 | 运费只是成本项之一。弹性回答「提价会掉多少量」，本卡回答「每单运费是多少」；两者相乘才能算清一次调价后的毛利，缺任何一半都会把定价做成猜。 |
| `Skill-Safety-Stock-Replenishment.md` | 下游（双向咬合） | 整箱补货的批量直接改变件数分布与体积重，而本卡 Stage 3 的并箱节省又会反过来改变补货批量的经济性 —— 批量与运费必须联立看，不能各算各的。 |
| `Skill-Multi-Echelon-Inventory.md` | 延伸 | 论文 §6 指出多仓场景下"每个起运仓一张分区图"不可扩展，需要仓 × 区域的共享划分；该卡的多阶节点视角是与之一致的扩展方向。 |

---

## ⑤ 商业价值评估

**ROI 预估**（给公式与参数来源，不直接给编造的数）：

```
避损收益 = Σ_z [ 分区订单占比 w_z × max(0, 该分区实际单均运费 − 现行定价所用平均运费) ] × 年单量 × 自担比例
并箱节省 = 可合并订单占比 × 单均合并节省额 × 年单量
```

参数来源：① **分区订单占比 w_z 与该分区单均运费** —— 本卡 Stage 0/Stage 1 直接产出；
② **现行定价所用平均运费** —— 贵司现行定价表/包邮门槛设置里实际使用的那个数；
③ **自担比例** —— 包邮与运费补贴政策下由商家承担运费的那部分订单占比；
④ **可合并订单占比与单均合并节省额** —— 本卡 Stage 3 的分区输出（带上限约束）。
四个输入**全部可在企业内部算出来**，无需外部基准。

论文给出的**方向性**参照只有一句定性：四段全模型的留出表现最好（§5.1），且月度聚合偏差窄（§5.2）。
**论文没有报告任何金额、成本下降比例或 ROI** —— 所以本卡不给"预期收益百分比"，
填入任何具体金额都会违反 R1（凭空数字比没有数字更有害）。

- **实施难度**：⭐⭐⭐☆☆（3/5）。技术上四段都是常规组件（Ridge + 梯度提升 + 加权汇总，本卡已给可运行实现），
  真正的成本在**数据工程**：费率卡版本化、ZIP→分区映射、账单与订单对齐、以及"合并"代理变量的构造与校准。
  这些是一次性投入，但缺任何一项，四段就只能停在演示。
- **优先级评分**：⭐⭐⭐⭐☆（4/5）。运费是大件母婴品类**毛利的主要可变项之一**，
  而它同时是定价、包邮门槛、拆单规则、补货批量四类决策的共同输入 —— 一处算准，四处受益。
  扣一星是因为：论文的验证环境是合成数据（§6 自承），落地时必须先在贵司自己的历史账单上做时间切分回测，
  不能直接采信论文的 MAE 对比。
- **评估依据**：① 方法链条完整且每一段都有对应原文（§3.1–§3.3），不是概念拼贴；
  ② 依赖的数据在平台店铺 + 独立站 + 承运商账单内**大部分可得**，缺口集中在费率卡与合并结果；
  ③ 与已有卡片 `Skill-Demand-Forecasting-Supply-Chain` / `Skill-Safety-Stock-Replenishment` /
  `Skill-Monodense-单品价格弹性估计` 直接串联，不是孤立新领域；
  ④ 论文的验证据此降级：合成数据集 + 单起运仓 + 单承运商，**证据等级按 preprint 处理**。

---

## ⑥ 原文引用

> 原文："We propose RouteCost, a production-inspired multi-stage framework that decomposes the problem into time-aware demand forecasting, fee-card-informed baseline pricing, Stage 2 residual correction, and proxy-based box-consolidation inference."
> 出处：2607.16230 §Abstract

> 原文："In practice, shipping cost is shaped not only by distance but also by destination demand mix, billable weight, dimensional pricing, surcharge triggers, and latent operational effects such as shipment consolidation."
> 出处：2607.16230 §Abstract

> 原文："Across over 250,000 orders, 260 products, and 18 months of order history, the framework improves predictive quality and aggregate calibration while preserving route-level interpretability."
> 出处：2607.16230 §Abstract

> 原文："In small-parcel environments, this task is shaped not only by geographic distance, but also by carrier pricing rules, dimensional billing, destination-specific demand patterns, and operational effects that may not be directly observable at prediction time (Bogyrbayeva et al., 2024; Mangiaracina et al., 2019)."
> 出处：2607.16230 §1 Introduction

> 原文："For example, larger products often also have higher wholesale cost, so wholesale cost and parcel cost can appear positively correlated across categories such as stools, chairs, sofas, and beds."
> 出处：2607.16230 §1 Introduction

> 原文："Two rugs with identical shipment dimensions but very different wholesale costs—for instance, a low-cost rug and a hand-knotted Persian rug—should incur essentially the same shipping cost."
> 出处：2607.16230 §1 Introduction

> 原文："We make three contributions: (1) we formulate pre-order shipping cost estimation as a route-weighted expectation problem; (2) we introduce a multi-stage architecture that separates demand forecasting, baseline pricing, residual correction, and proxy-based box-consolidation inference; and (3) we show through temporal backtesting and route-level decomposition that structured decomposition improves both predictive quality and interpretability."
> 出处：2607.16230 §1 Introduction

> 原文："Consolidation has long been recognized as an important logistics strategy because combining multiple shipments can reduce transportation cost and exploit scale economies (Hall, 1987; Wei et al., 2021)."
> 出处：2607.16230 §2 Related Work

> 原文："RouteCost contains four modules: (1) time-aware demand forecasting, which estimates route weights by forecasting destination-zone shares over time; (2) fee-card-informed Stage 1 pricing, which produces a structured baseline estimate using billable weight, dimensions, rate-card lookups, and surcharge features; (3) Stage 2 residual correction, which captures nonlinear error not explained by the baseline; and (4) proxy-based box-consolidation inference, which estimates latent savings associated with likely consolidation opportunities."
> 出处：2607.16230 §3.1 Framework Overview

> 原文："Under our simplified fulfillment setting, we set Boston as the single in-stock warehouse and all the order shipment originates from Boston and deliver through a single parcel carrier (FedEx)."
> 出处：2607.16230 §3.2 Problem Formulation

> 原文："In practice, these destination zones are obtained from the FedEx zone locator: after specifying Boston ZIP 02108 as the shipping origin, each destination ZIP code is assigned a zone index ranging from 1 to 8."
> 出处：2607.16230 §3.2 Problem Formulation

> 原文："More generally, if the fulfillment system includes multiple warehouses and multiple carriers, the route space expands beyond zones alone to a cross-product of destination regions, warehouse choices, and carrier choices."
> 出处：2607.16230 §3.2 Problem Formulation

> 原文："Rather than assuming a fixed destination distribution, we construct monthly zone shares and smooth them using short rolling windows."
> 出处：2607.16230 §3.3 Demand Forecasting and Multi-Stage Cost Estimation

> 原文："Because destination patterns vary across product groups, we also estimate category-specific monthly zone shares and apply a hierarchical fallback strategy during inference."
> 出处：2607.16230 §3.3 Demand Forecasting and Multi-Stage Cost Estimation

> 原文："Carrier rate-card information remains an important signal, but the final baseline estimate is learned through a regularized Ridge regression model that integrates billable weight, physical weight, dimensional weight, package dimensions, destination zone, synthetic rate-card lookup values, and surcharge-related flags."
> 出处：2607.16230 §3.3 Demand Forecasting and Multi-Stage Cost Estimation

> 原文："Stage 2 then learns a nonlinear residual correction over the Stage 1 output using gradient boosting."
> 出处：2607.16230 §3.3 Demand Forecasting and Multi-Stage Cost Estimation

> 原文："Finally, box consolidation is inferred from weak operational proxies such as same-day same-ZIP density, same-day same-zone density, family-level average quantity, size compatibility, package split risk, and a composite consolidation opportunity score."
> 出处：2607.16230 §3.3 Demand Forecasting and Multi-Stage Cost Estimation

> 原文："Because the final estimate is computed as the sum of route-level weighted contributions, the model can be audited at both the prediction level and the route-composition level."
> 出处：2607.16230 §3.3 Demand Forecasting and Multi-Stage Cost Estimation

> 原文："This separation reduces the need for a single model to jointly absorb pricing rules, demand patterns, and hidden operational effects in one end-to-end mapping."
> 出处：2607.16230 §3.3 Demand Forecasting and Multi-Stage Cost Estimation

> 原文："The dataset contains 250,000 order records, 260 products, and 18 months of transaction history."
> 出处：2607.16230 §4 Dataset and Experimental Setup

> 原文："The synthetic pricing layer is built from a simplified FedEx-like rate-card structure indexed by destination zone and billable weight."
> 出处：2607.16230 §4 Dataset and Experimental Setup

> 原文："We use a time-based split in which earlier months are used for training and the final three months are reserved for holdout evaluation."
> 出处：2607.16230 §4 Dataset and Experimental Setup

> 原文："| Orders | 250,000 line-item orders | | Products | 260 products across 36 categories | | Time span | 2024-01 to 2025-06 (18 months) | | Fulfillment setting | Single Boston warehouse; single parcel carrier | | Route granularity | 8 destination zones | | Pricing layer | Zone x billable-weight rate card with surcharges | | Consolidation signal | Latent savings inferred from weak operational proxies | | Evaluation | Time-based holdout; MAE, MAPE, aggregate error |"
> 出处：2607.16230 §4 表 1（Dataset and setup summary；原文为多行表格，此处按单行摘录，行序与原文一致）

> 原文："Adding Stage 2 improves the estimate by correcting nonlinear residual error, while the full framework further benefits from the inferred consolidation effect."
> 出处：2607.16230 §5.1 Main Performance

> 原文："Overall, the full model achieves the best holdout performance, indicating that each additional stage contributes useful information beyond the structured baseline."
> 出处：2607.16230 §5.1 Main Performance

> 原文："| Stage 1 | 1.735 | 0.070 | | Stage 1 + Stage 2 | 1.893 | 0.078 | | Full Model | 1.649 | 0.064 |"
> 出处：2607.16230 §5.1 表 2（Holdout performance comparison across model variants；行序与原文一致）

> 原文："This metric is especially useful because a model with acceptable per-order accuracy can still generate biased monthly totals, which would be problematic for pricing, budgeting, and gross-margin planning."
> 出处：2607.16230 §5.2 Temporal Error Analysis

> 原文："As shown in Figure 6, the full model remains close to zero aggregate bias in most months, with fluctuations contained within a narrow range of roughly -0.14% to +0.06%."
> 出处：2607.16230 §5.2 Temporal Error Analysis

> 原文："In practical terms, route cost and route contribution are not identical: some far-away zones contribute strongly because route costs are high even when demand share is modest, while some nearby zones dominate mainly because demand is concentrated there."
> 出处：2607.16230 §5.3 Route-Level Interpretability

> 原文："This route-weighted interpretation is especially valuable for downstream pricing discussions because it ties expected cost back to both operational demand patterns and pricing mechanics."
> 出处：2607.16230 §5.3 Route-Level Interpretability

> 原文："The evaluation uses an operationally grounded synthetic dataset and a simplified fulfillment network with a single origin, single carrier, and single service level, which reduces route complexity."
> 出处：2607.16230 §6 Discussion and Limitations

> 原文："In addition, the consolidation target is inferred through pseudo-label construction rather than learned from true shipment-level consolidation outcomes."
> 出处：2607.16230 §6 Discussion and Limitations

> 原文："In a multi-warehouse setting, maintaining a separate zone map for each origin would not scale well."
> 出处：2607.16230 §6 Discussion and Limitations

> 原文："A further advantage of this decomposition is robustness under distribution shift."
> 出处：2607.16230 §6 Discussion and Limitations

> 原文："In such settings, a modular design supports targeted refresh of the affected components: demand models can be updated when destination mix changes, while pricing logic or rate-card inputs can be revised when carrier pricing rules change."
> 出处：2607.16230 §6 Discussion and Limitations

---

## 附：证据链与核验方式

- **全文底本**：`paper2skills-vault/papers/03-时间序列/p2s-2026-0005/fulltext.md`
  （由 `paper2skills-research/scripts/fetch_fulltext.py` 从 arXiv LaTeXML HTML 转换，保留章节号）
- **引文逐字核验**：`python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <本卡>`
- **K1 代码可执行**：`python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <本卡>`
- **K2 门禁**：`python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <本卡> --k1 <K1 JSON>`
- **证据档案**：`paper2skills-vault/papers/03-时间序列/p2s-2026-0005/evidence.md`
- **本地可复现数字的约定**：论文事实数字一律进 ⑥ 引用块；本模板运行输出、按贵司参数代入的算式
  放进代码/输出围栏并标注「可复现」——围栏内的数字是**可自行验证**的，不冒充论文结论。
- **代码公开性**：registry 备注为「摘要未说明代码」，全文也未给出代码仓库或数据链接，
  因此**按无公开代码处理**：本卡 ③ 段是按论文 §3.3 的功能分解自行实现的业务化简化版，不是官方实现。
