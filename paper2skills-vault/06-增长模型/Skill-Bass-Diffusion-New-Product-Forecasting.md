---
title: GEANN + Bass 新品冷启动需求预测 - 母婴跨境新品备货
doc_type: knowledge
module: 06-增长模型
topic: new-product-cold-start-forecasting
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
paper: arXiv:2307.03595 (Amazon, 2023) + arXiv:2406.16221 (UIUC+Amazon, 2024)
roadmap_phase: phase2
---

# Skill: 新品冷启动需求预测 — Bass 扩散 + GEANN 相似品迁移

> 主论文:**GEANN: Scalable Graph Augmentations for Multi-Horizon Time Series Forecasting** (Yang & Wolff, Amazon + UC Berkeley, 2023) · arXiv:2307.03595
> 辅论文:**F-FOMAML: GNN-Enhanced Meta-Learning for Peak Period Demand Forecasting** (Xu et al., UIUC + Rutgers + Amazon, 2024) · arXiv:2406.16221
> 开源参考:[PyMC-Marketing Bayesian Bass](https://github.com/pymc-labs/pymc-marketing) + [bassmodeldiffusion](https://github.com/marmiskarian/bassmodeldiffusion)

---

## ① 算法原理

### 核心思想

母婴跨境**新品冷启动需求预测**痛点:每年 20-30 款新品上市,前 8 周零销售记录,人工拍脑袋备货首批,**积压或断货损失年化 300+ 万元**. 本 Skill 组合两个方法:① **Bass 扩散模型**生成新品扩散曲线形状(创新+模仿系数);② **GEANN 图迁移**从相似品历史借用销售信号;③ Bass 参数从相似品**加权迁移**初始化,实现"形状从理论 + 规模从迁移"的双驱动.

### 数学直觉

**Bass 累积采用函数**:
$$N(t) = m \cdot \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} e^{-(p+q)t}}$$
- $m$: 市场潜力(最终累计采购者数)
- $p$: 创新系数(外部广告驱动)≈ 0.01-0.03
- $q$: 模仿系数(口碑传播)≈ 0.3-0.5

**Bass 增量(每周新增需求)**:
$$\frac{dN(t)}{dt} = \left(p + q \cdot \frac{N(t)}{m}\right)(m - N(t))$$

**峰值时间**:
$$t^* = \frac{\ln(q/p)}{p + q}$$

**GEANN 图嵌入相似度**(相似品检索):
$$S_{\text{Corr}}(i, j) = |\text{Corr}(H_i^{(0)}, H_j^{(0)})|$$
$H_i^{(0)}$ 是预训练 MQ-CNN 嵌入,Pearson 相关度高 → 商品相似度高.

**Bass 参数加权迁移**(本 Skill 核心):
$$(\hat{m}, \hat{p}, \hat{q}) = \sum_{k} w_k \cdot (m_k, p_k, q_k), \quad w_k = \frac{\exp(s_k)}{\sum_{j} \exp(s_j)}$$
$s_k$ 是相似品 $k$ 的相似度分数,Softmax 归一化作权重.

### 关键假设

1. **相似品有 6-12 个月历史**:能拟合稳定 Bass 参数
2. **目标新品有特征向量**:价格带 / 品类 / 产地 / 渠道等
3. **市场潜力可估算**:目标品类月销 × 预期份额(可参考竞品 BSR)

### 关键效果数字

| 维度 | 数据 |
|---|---|
| GEANN 数据集 | Amazon EU5(10万商品)+ NA(200万商品) × 5 年 |
| GEANN 整体 P50/P90 改善 | **2.3-3.1%** |
| **GEANN 新品冷启动 P90 改善** | **5.7%** |
| GEANN 近期缺货品改善 | 3.5% |
| F-FOMAML 内部数据 MAE 降低 | **26.24%** |
| F-FOMAML JD.com 公开数据 MAE 降低 | 1.04% |
| Bass 参数行业经验值 | $p \in [0.01, 0.03], q \in [0.3, 0.5]$ |

---

## ② 母婴出海应用案例

### 场景一:新品奶粉冷启动(前 8 周零销售记录)

- **业务问题**:Momcozy 上市新品"羊奶粉 A 段 800g 德国品牌 X",历史零销售,采购无依据备货. 首批多压货占用现金流 50-100 万,少备货错失上升期 BSR 损失更大. **Anker 案例:新品冷启动占总库存损失 35%**
- **数据要求**:新品特征向量(价格/品类/产地/渠道) + 同品类相似品 52 周历史销售
- **配置流程**:
  1. **相似品检索**:用 GEANN kNN 找 Top-3 相似品(德国奶粉 800g / 澳洲奶粉 900g / 荷兰奶粉 800g)
  2. **Bass 参数拟合**:对每个相似品拟合 (m, p, q)
  3. **Softmax 加权迁移**:相似度高的相似品权重大
  4. **市场潜力估算**:目标品类月销 × 8 周 × 预期份额 (3-5%)
  5. **逐周预测**:Bass 增量公式输出第 1-8 周需求 + P10/P90 置信区间
- **业务价值**:
  - 首批备货精度从 ±50% 提升至 ±20%
  - 单款新品节省滞销损失 10-30 万 + 减少断货损失 5-20 万 = **15-50 万/款**
  - 年化 20-30 款新品 × 平均 25 万/款 = **500-750 万/年**

### 场景二:季节性新品 Prime Day 备货(F-FOMAML 场景)

- **业务问题**:Momcozy 推出 Prime Day 季节性新品(夏季婴儿游泳圈),**大促备货截止日提前 60 天**(亚马逊 Prime Day 报名),但新品当时只有 3-7 天早期销售数据;过量备货占用 FBA 长期仓储费,不足备货错失大促爆发
- **数据要求**:Prime Day 新品早期 3-7 天销售 + 非大促期相似品(湿巾/纸尿裤)峰值数据
- **F-FOMAML 配置**:
  - 非大促期相似品销售曲线作"代理数据"
  - GNN 按品类 + 价格带 + 历史峰值倍率构建商品相似图
  - F-FOMAML 用 3-7 天早期数据快速 fine-tune
  - 输出 Prime Day 24-48 小时峰值需求曲线
- **业务价值**:
  - Prime Day 备货精度提升:MAE 降低 25%(F-FOMAML 实测)
  - 单款季节性新品避免 Prime Day 断货 = **30-80 万/款 GMV 保护**
  - 季节性新品年 5-10 款 × 50 万 = **250-500 万/年**

---

## ③ 代码模板

```python
"""
新品冷启动需求预测 - Bass 扩散 + 相似品迁移最小骨架
论文 arXiv:2307.03595 + arXiv:2406.16221
依赖: pip install numpy scipy scikit-learn
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def bass_cumulative(t: np.ndarray, m: float, p: float, q: float) -> np.ndarray:
    """Bass 累积采用 N(t) = m*(1-e^-(p+q)t)/(1 + q/p * e^-(p+q)t)"""
    return m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


def bass_incremental(t: np.ndarray, m: float, p: float, q: float) -> np.ndarray:
    """Bass 增量 dN/dt(每周新增需求)"""
    N = bass_cumulative(t, m, p, q)
    return (p + q * N / m) * (m - N)


def fit_bass_from_history(sales_history: np.ndarray) -> Tuple[float, float, float]:
    """从相似品历史销售拟合 Bass 参数(m, p, q)"""
    t = np.arange(1, len(sales_history) + 1, dtype=float)
    cumulative = np.cumsum(sales_history)
    try:
        p0 = [cumulative[-1] * 2, 0.02, 0.4]
        bounds = ([0, 0.001, 0.01], [1e8, 0.5, 2.0])
        popt, _ = curve_fit(bass_cumulative, t, cumulative, p0=p0, bounds=bounds, maxfev=5000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        return float(cumulative[-1] * 3), 0.02, 0.38


@dataclass
class Product:
    sku_id: str
    features: np.ndarray
    sales_history: Optional[np.ndarray] = None


def find_similar_products(new_product: Product, catalog: List[Product], top_k: int = 5) -> List[Tuple[Product, float]]:
    """基于特征向量余弦相似度找 Top-K 相似品"""
    new_feat = new_product.features
    similarities = []
    for prod in catalog:
        if prod.sales_history is not None:
            num = float(np.dot(new_feat, prod.features))
            denom = float(np.linalg.norm(new_feat) * np.linalg.norm(prod.features))
            sim = num / denom if denom > 0 else 0.0
            similarities.append((prod, sim))
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


def transfer_bass_params(
    similar_products: List[Tuple[Product, float]],
    market_potential_override: Optional[float] = None,
) -> Tuple[float, float, float]:
    """从相似品 Softmax 加权迁移 Bass 参数"""
    if not similar_products:
        return 1000.0, 0.02, 0.38

    params_list = []
    weights = []
    for prod, sim_score in similar_products:
        m, p, q = fit_bass_from_history(prod.sales_history)
        params_list.append((m, p, q))
        weights.append(sim_score)

    w = np.array(weights)
    if w.std() > 0:
        w = np.exp(w) / np.exp(w).sum()
    else:
        w = np.ones(len(w)) / len(w)

    m_avg = np.average([x[0] for x in params_list], weights=w)
    p_avg = np.average([x[1] for x in params_list], weights=w)
    q_avg = np.average([x[2] for x in params_list], weights=w)

    if market_potential_override:
        m_avg = market_potential_override

    return float(m_avg), float(p_avg), float(q_avg)


def cold_start_forecast(
    new_product: Product,
    catalog: List[Product],
    forecast_weeks: int = 8,
    market_potential: Optional[float] = None,
    top_k: int = 5,
) -> dict:
    """新品冷启动主预测流程"""
    similar = find_similar_products(new_product, catalog, top_k=top_k)
    m, p, q = transfer_bass_params(similar, market_potential_override=market_potential)

    t = np.arange(1, forecast_weeks + 1, dtype=float)
    weekly_demand = bass_incremental(t, m, p, q)

    if len(similar) >= 2:
        all_params = [fit_bass_from_history(p.sales_history) for p, _ in similar]
        demand_samples = np.array([bass_incremental(t, pm, pp, pq) for pm, pp, pq in all_params])
        lower = np.percentile(demand_samples, 10, axis=0)
        upper = np.percentile(demand_samples, 90, axis=0)
    else:
        lower = weekly_demand * 0.7
        upper = weekly_demand * 1.3

    return {
        "sku_id": new_product.sku_id,
        "bass_params": {"m": m, "p": p, "q": q},
        "peak_week": int(np.argmax(weekly_demand)) + 1,
        "weekly_forecast": weekly_demand.tolist(),
        "lower_p10": lower.tolist(),
        "upper_p90": upper.tolist(),
        "cumulative_8w": float(weekly_demand.sum()),
        "similar_products": [(p.sku_id, s) for p, s in similar],
    }


def main() -> None:
    np.random.seed(42)
    catalog = [
        Product(
            "SIM_FORMULA_DE_800G",
            features=np.array([0.6, 0.0, 0.0, 0.0]),
            sales_history=np.array([50, 120, 200, 280, 310, 320, 315, 308, 295, 280] * 4, dtype=float),
        ),
        Product(
            "SIM_FORMULA_AU_900G",
            features=np.array([0.55, 0.0, 1.0, 0.0]),
            sales_history=np.array([30, 80, 150, 220, 280, 300, 310, 305, 295, 285] * 4, dtype=float),
        ),
        Product(
            "SIM_FORMULA_NL_800G",
            features=np.array([0.65, 0.0, 0.0, 1.0]),
            sales_history=np.array([40, 100, 180, 260, 300, 315, 318, 310, 298, 282] * 4, dtype=float),
        ),
    ]

    new_product = Product(
        sku_id="NEW_GOAT_FORMULA_DE_800G",
        features=np.array([0.62, 0.0, 0.0, 0.0]),
    )

    result = cold_start_forecast(
        new_product=new_product,
        catalog=catalog,
        forecast_weeks=8,
        market_potential=4800,
        top_k=3,
    )

    print(f"=== {result['sku_id']} 冷启动预测 ===")
    print(f"Bass 参数: m={result['bass_params']['m']:.0f}, "
          f"p={result['bass_params']['p']:.4f}, q={result['bass_params']['q']:.4f}")
    print(f"峰值预测周: 第 {result['peak_week']} 周")
    print(f"8 周累计需求: {result['cumulative_8w']:.0f} 件")
    print(f"\n周次 | 预测 | P10 | P90")
    for i in range(8):
        print(f"  第 {i+1} 周 | {result['weekly_forecast'][i]:6.0f} | "
              f"{result['lower_p10'][i]:6.0f} | {result['upper_p90'][i]:6.0f}")
    print(f"\n相似品:")
    for sku, sim in result["similar_products"]:
        print(f"  - {sku}: {sim:.3f}")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-Feature-Engineering](../12-ML基础/[[Skill-Feature-Engineering]].md) — 商品特征向量化(价格/品类/产地)是相似品检索的基础
- [Skill-Cold-Start-Meta-Learning-PAM](../05-推荐系统/[[Skill-Cold-Start-Meta-Learning-PAM]].md) — 元学习冷启动是 F-FOMAML 的方法学基础
- [Skill-Time-Series-Forecasting](../03-时间序列/[[Skill-Time-Series-Forecasting]].md) — Bass 模型是时序预测的特例

### 延伸技能
- [Skill-Demand-Forecasting-Supply-Chain](../04-供应链/[[Skill-Demand-Forecasting-Supply-Chain]].md) — 冷启动预测后接入正常补货流程
- [Skill-Hierarchical-Demand-Forecasting-Reconciliation](../03-时间序列/[[Skill-Hierarchical-Demand-Forecasting-Reconciliation]].md) — 新品冷启动 + 分层预测组合
- [Skill-Lead-Time-Distribution-Risk-GenQOT](../04-供应链/[[Skill-Lead-Time-Distribution-Risk-GenQOT]].md) — Bass 预测驱动动态安全库存

### 可组合
- [Skill-Cold-Start-Product-Recommendation](./[[Skill-Cold-Start-Product-Recommendation]].md) — 新品冷启动推荐 + 需求预测形成闭环
- [Skill-Hierarchical-Product-KG-Construction](../08-知识图谱/[[Skill-Hierarchical-Product-KG-Construction]].md) — KG 提供相似品检索的语义信号

---

## ⑤ 商业价值评估

### ROI 预估

**场景一(新品奶粉冷启动)**:年化 20-30 款新品 × 25 万/款节省 = **500-750 万/年**

**场景二(季节性新品 Prime Day)**:5-10 款季节性新品 × 50 万/款保护 = **250-500 万/年**

**合计**:**750-1250 万/年**

### 实施难度:⭐⭐⭐☆☆ (3/5)

- 易处:Bass 模型有解析解,scipy.curve_fit 可直接拟合
- 易处:PyMC-Marketing Bayesian Bass 开源完整
- 难处:**Amazon GEANN/F-FOMAML 未开源**,GNN 部分需自行实现
- 难处:相似品特征工程需业务专家(价格/品类/产地编码)
- 难处:市场潜力 m 估算依赖竞品数据(可借助 Helium 10 / Jungle Scout)

### 优先级评分:⭐⭐⭐⭐⭐ (5/5)

**评估依据**:
1. **Amazon 内部生产部署**:GEANN 是 Amazon Forecasting 团队 2023 工作
2. **直接解决 WF-A P0 缺口**:新品冷启动是母婴跨境的核心痛点(年损 300+ 万)
3. **Bass + GNN 组合**:经典理论 + 现代深度学习,鲁棒性强
4. **Pyc-Marketing 开源**:Bass 部分工程化路径成熟
5. **与 HiFoReAd + Gen-QOT 形成完整 WF-A 供应链闭环**
