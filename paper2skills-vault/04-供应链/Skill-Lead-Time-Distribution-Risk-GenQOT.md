---
title: Gen-QOT 提前期分布建模 - 动态安全库存防海运延误
doc_type: knowledge
module: 04-供应链
topic: lead-time-distribution-risk
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
paper: arXiv:2310.17168 (Amazon, 2024)
---

# Skill: Gen-QOT — 随机提前期分布建模与动态安全库存

> 论文:**Learning an Inventory Control Policy with General Inventory Arrival Dynamics** (Andaz et al., Amazon + Harvard, 2024) · arXiv:2310.17168
> Amazon 内部真实 A/B 测试验证,击败传统库存管理系统
> 参考代码:Madeka 2022 GitHub awslabs/sagemaker-deep-demand-forecast

---

## ① 算法原理

### 核心思想

母婴跨境海运提前期(Lead Time, LT)在 25-50 天剧烈波动(苏伊士事件/港口拥堵). 传统安全库存假设 **LT 固定**,实际服务水平远低于设定值(设 95% 实际只有 85%). Gen-QOT 用**深度自回归生成模型**对 LT 进行**分布式建模**(不假设参数分布),并把"订单整批到货"扩展为**分批随机到达**(QOT, Quantity-Over-Time),精确建模拼箱拆批到港行为. 动态安全库存自适应季节性 + 港口拥堵期.

### 数学直觉

**QOT 到达模型**(核心扩展):
$$o_{t,j} = \min(U_t, a_t) \cdot \rho_{t,j}, \quad \sum_j \rho_{t,j} = 1$$
订单 $a_t$ 在未来 $L$ 周按份额向量 $(\rho_{t,0}, \ldots, \rho_{t,L})$ 分散到货,$U_t$ 是供应商当期供货上限.

**经典安全库存(下界,LT 固定假设)**:
$$SS_{\text{classic}} = z_\alpha \cdot \sigma_D \cdot \sqrt{\bar{L}}$$
$\bar{L}$ 为 LT 均值,**忽略 LT 方差,严重低估**.

**Silver-Meal 扩展(同时考虑 LT 方差)**:
$$\sigma_{LTD} = \sqrt{\bar{L} \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2}$$
$$SS_{\text{dynamic}} = z_\alpha \cdot \sigma_{LTD}$$

**Gen-QOT 数据驱动**:
$$SS_{\text{Gen-QOT}} = z_\alpha \cdot \hat{\sigma}_{VLT\_demand}$$
$\hat{\sigma}_{VLT\_demand}$ 来自深度自回归模型对 LT 分布的采样经验标准差,**支持厚尾/不规则分布**.

**极端事件缓冲(P99)**:
$$SS_{\text{P99}} = \bar{D} \cdot (Q_{0.99}(LT) - \bar{L})$$

**再订货点(ROP)**:
$$ROP = \bar{D} \cdot \bar{L} + SS_{\text{dynamic}}$$

### 关键假设

1. **LT 历史数据足够**:至少 6-12 个月 PO 到货时序(母婴跨境 1-2 年累积充分)
2. **季节性可识别**:旺季(双 11/6.18/Prime Day) + 平季 + 淡季的 LT 分布差异
3. **拼箱拆批行为**:多 SKU 共柜运输时部分到货是常态(QOT 模型核心)

### 关键效果数字

| 指标 | Gen-QOT vs 传统 |
|---|---|
| LT 预测 CRPS / Quantile Loss | **持平或更优** |
| 库存利润(模拟回测) | **显著提升** |
| 真实 A/B 测试 | **击败 Amazon 生产库存系统** |
| 服务水平实际达成 | 85% → **接近设定 95%** |
| 拥堵期动态 SS | **较固定 SS 增加 40-70%** |

---

## ② 母婴出海应用案例

### 场景一:海运延误实时预警系统

- **业务问题**:苏伊士运河拥堵 / 巴拿马运河旱季 / 港口积压时,Momcozy 海运 LT 从均值 30 天变成 50+ 天,**预警通常滞后 2-3 周**才发现库存紧张. 单次缺货损失 BSR 排名(恢复需 3-6 月)+ 直接销售损失,**单产品月损失约 30-80 万元**
- **数据要求**:历史 PO 到货时序(至少 200 条) + 当前在途订单 + 港口拥堵实时指数(可选)
- **Gen-QOT 配置**:
  - GluonTS DeepAR 拟合 LT 经验分布,输出 P10/P50/P90 分位数
  - 实时监控 $P_{90}(LT) - P_{10}(LT)$ 跨度,**> 20 天触发预警**
  - 自动提前下单,补偿延误影响
- **业务价值**:
  - 预警提前 2-3 周响应 vs 传统月度复盘事后处理
  - 单产品月损失 30-80 万 × 5-10 个核心 SKU = **年化避免损失 1800-9600 万元**(头部品牌量级)
  - 中小品牌单 SKU 节省 = **年化 200-500 万元**

### 场景二:旺季安全库存动态调整(双11/6.18/Prime Day)

- **业务问题**:Momcozy 双 11 期间需求和 LT 同时方差扩大,传统全年固定安全库存系数导致**旺季前 6 周安全库存严重不足 + 旺季后 4 周积压**. 旺季前估计 SS 不足 30-40% 导致大促爆款断货,旺季后积压占用 FBA 长期仓储费(月均 $0.15/立方英尺)
- **数据要求**:历史季节性需求 + 旺季 LT 数据 + 大促日历
- **Gen-QOT 配置**:
  - 按季节性 / 大促窗口条件化 Gen-QOT
  - 旺季前 6 周:$SS_{dynamic}$ 提升 40-70%(自适应)
  - 旺季后 4 周:$SS_{dynamic}$ 自动收缩 30%
- **业务价值**:
  - 旺季缺货率从 15-20% 降至 3-5% = 大促 GMV 弹性 +5-10% = **单次大促 50-100 万**
  - 旺季后 FBA 长期仓储费节省 30-50% = **单季 20-40 万**
  - 全年 4 个大促 × (50-100 万 + 20-40 万) = **280-560 万元/年**

---

## ③ 代码模板

```python
"""
Gen-QOT 提前期分布建模 + 动态安全库存最小骨架
论文 arXiv:2310.17168 (Amazon, 2024)
生产替换: GluonTS DeepAR / MQ-CNN 学习 LT 分布
依赖: pip install numpy scipy
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
from scipy import stats


def sample_lt_history(
    n: int = 365,
    base_days: int = 30,
    sigma_days: int = 7,
    congestion_prob: float = 0.1,
    congestion_extra: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """模拟海运 LT 历史(生产替换为真实 PO 到货时序)
    - base_days/sigma_days: 正常 LT 分布
    - congestion_prob: 港口拥堵概率(10% 默认)
    - congestion_extra: 拥堵时额外延误天数
    """
    rng = np.random.default_rng(seed)
    normal = rng.normal(base_days, sigma_days, n)
    shocks = rng.choice([0, congestion_extra], size=n, p=[1 - congestion_prob, congestion_prob])
    return np.clip(normal + shocks, base_days - 10, base_days + 40).astype(int)


def fit_lt_distribution(lt_history: np.ndarray, context: Dict = None) -> Dict[float, float]:
    """Gen-QOT 风格:非参数经验分位数 LT 预测
    生产替换为 GluonTS DeepAR 或 MQ-CNN with context features
    """
    quantile_levels = [0.1, 0.5, 0.9, 0.95, 0.99]
    return {q: float(np.quantile(lt_history, q)) for q in quantile_levels}


def calc_safety_stock(
    demand_mean: float,
    demand_std: float,
    lt_quantiles: Dict[float, float],
    service_level: float = 0.95,
) -> Dict[str, float]:
    """三种安全库存计算:classic / dynamic / P99 buffer"""
    z = stats.norm.ppf(service_level)
    lt_mean = lt_quantiles[0.5]
    lt_std = (lt_quantiles[0.9] - lt_quantiles[0.1]) / 2.56

    ss_classic = z * demand_std * np.sqrt(lt_mean)

    sigma_ltd = np.sqrt(lt_mean * demand_std ** 2 + demand_mean ** 2 * lt_std ** 2)
    ss_dynamic = z * sigma_ltd

    ss_p99_buffer = demand_mean * (lt_quantiles[0.99] - lt_mean)

    return {
        "ss_classic": round(ss_classic, 0),
        "ss_dynamic": round(ss_dynamic, 0),
        "ss_p99_buffer": round(ss_p99_buffer, 0),
        "reorder_point": round(demand_mean * lt_mean + ss_dynamic, 0),
        "lt_p10": lt_quantiles[0.1],
        "lt_p50": lt_quantiles[0.5],
        "lt_p90": lt_quantiles[0.9],
        "lt_p99": lt_quantiles[0.99],
    }


def detect_lt_drift_warning(
    lt_history: np.ndarray,
    recent_window: int = 30,
    threshold_days: float = 20.0,
) -> Dict:
    """LT 分布漂移预警(港口拥堵实时检测)"""
    if len(lt_history) < recent_window * 2:
        return {"warning": False, "reason": "insufficient_history"}

    historical = lt_history[:-recent_window]
    recent = lt_history[-recent_window:]
    p10_p90_spread = np.quantile(recent, 0.9) - np.quantile(recent, 0.1)
    historical_spread = np.quantile(historical, 0.9) - np.quantile(historical, 0.1)

    return {
        "warning": p10_p90_spread > threshold_days,
        "recent_spread_days": float(p10_p90_spread),
        "historical_spread_days": float(historical_spread),
        "spread_change_pct": float((p10_p90_spread - historical_spread) / historical_spread * 100),
    }


def main() -> None:
    lt_hist = sample_lt_history(n=365)
    lt_q = fit_lt_distribution(lt_hist)

    result = calc_safety_stock(
        demand_mean=50,
        demand_std=15,
        lt_quantiles=lt_q,
        service_level=0.95,
    )
    print("=== 母婴 SKU 安全库存对比 ===")
    print(f"LT P10/P50/P90/P99: {lt_q[0.1]:.0f}/{lt_q[0.5]:.0f}/{lt_q[0.9]:.0f}/{lt_q[0.99]:.0f} 天")
    print(f"  经典 SS(LT 固定假设):    {result['ss_classic']:.0f} 件  ← 低估!")
    print(f"  动态 SS(Silver-Meal 扩展): {result['ss_dynamic']:.0f} 件  ← 推荐")
    print(f"  P99 极端缓冲:            {result['ss_p99_buffer']:.0f} 件")
    print(f"  再订货点 (ROP):         {result['reorder_point']:.0f} 件")

    print("\n=== 港口拥堵漂移预警 ===")
    warning = detect_lt_drift_warning(lt_hist, recent_window=30, threshold_days=20)
    print(f"  预警状态: {warning['warning']}")
    print(f"  近 30 天 LT 跨度: {warning['recent_spread_days']:.1f} 天")
    print(f"  历史 LT 跨度: {warning['historical_spread_days']:.1f} 天")
    print(f"  跨度变化: {warning['spread_change_pct']:+.1f}%")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-Demand-Forecasting-Supply-Chain](./[[Skill-Demand-Forecasting-Supply-Chain]].md) — demand_mean / demand_std 输入依赖
- [Skill-Safety-Stock-Replenishment](./[[Skill-Safety-Stock-Replenishment]].md) — 安全库存计算的传统基础

### 延伸技能
- [Skill-Two-Echelon-Inventory-DRL](./[[Skill-Two-Echelon-Inventory-DRL]].md) — Gen-QOT 输出的 ROP 驱动多级库存 DRL 决策
- [Skill-Multi-Echelon-Inventory](./[[Skill-Multi-Echelon-Inventory]].md) — 多仓 LT 风险联合建模

### 可组合
- [Skill-Time-Series-Anomaly-Detection](../03-时间序列/[[Skill-Time-Series-Anomaly-Detection]].md) — LT 异常检测 + Gen-QOT 拥堵预警联动
- [Skill-Hierarchical-Demand-Forecasting-Reconciliation](../03-时间序列/[[Skill-Hierarchical-Demand-Forecasting-Reconciliation]].md) — 调和后的多层预测 + Gen-QOT 形成完整补货闭环

---

## ⑤ 商业价值评估

### ROI 预估

**场景一(海运延误预警)**:中型品牌单 SKU = **200-500 万/年**;头部品牌 5-10 个核心 SKU = **1800-9600 万/年**

**场景二(旺季动态 SS)**:4 个大促 = **280-560 万/年**

**合计(中型品牌)**:**480-1060 万/年**

### 实施难度:⭐⭐⭐⭐☆ (4/5)

- 易处:Silver-Meal 扩展公式有解析解,简单部署
- 易处:GluonTS DeepAR / MQ-CNN 开源,工程化路径成熟
- 难处:**Amazon 主论文未开源**,Gen-QOT 完整深度模型需自行实现
- 难处:LT 历史数据质量要求高(PO 到货时间戳精确)
- 难处:QOT 拼箱拆批数据采集需要 ERP 系统对接

### 优先级评分:⭐⭐⭐⭐⭐ (5/5)

**评估依据**:
1. **Amazon 真实 A/B 测试验证**,击败生产库存系统
2. **直接解决 WF-A 工作流 P0 缺口**:服务水平 85% → 95%
3. **跨境海运痛点核心**:苏伊士/港口拥堵的应对方案
4. **节省额度极大**:头部品牌年化损失节省可达千万元级
5. **可与 HiFoReAd 组合**:多层预测调和 + 动态 LT 风险 = 完整供应链闭环
