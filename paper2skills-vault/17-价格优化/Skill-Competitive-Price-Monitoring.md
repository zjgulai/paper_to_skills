# Skill Card: Competitive Price Monitoring（竞品价格监测与响应）

> **领域**: 17-价格优化 | **类型**: 综合萃取

---

## ① 算法原理

### 核心思想
竞品价格监测不只是"看别人卖多少钱"，而是建立价格-转化率的因果响应模型，在竞品降价时量化"不跟降会损失多少"和"跟降能获得多少"，做出有数据支撑的响应决策。

### 数学直觉

**价格竞争力指数 (PCI)**：
$$PCI = \frac{P_{our}}{P_{market\_median}}$$

当 PCI > 1.05（我们比市场中位数贵 5%+）时，需评估响应。

**竞品响应模型**——基于历史数据估计跟降的预期收益：
$$E[\Delta Q] = \hat{\epsilon} \cdot \frac{\Delta P}{P} \cdot Q_{baseline}$$

其中 $\hat{\epsilon}$ 是通过准实验（DiD / 合成控制）估计的 cross-price elasticity（交叉价格弹性）。注意区分：
- Own-price elasticity：自己降价 → 自己销量变化
- Cross-price elasticity：竞品降价 → 我们销量变化

### 关键假设
- 竞品价格变化是外生的（不是针对我们的反应）
- 交叉弹性在品类内稳定（吸奶器 vs 吸奶器 ≈ -0.8 到 -1.2）

---

## ② 母婴出海应用案例

### 场景：Amazon 吸奶器品类的竞品价格战监测

**业务问题**：5 个竞品（Momcozy/Medela/Spectra/Bellababy/Elvie）在 Amazon US 频繁调价。需要每日监测+自动预警+响应建议。

**数据要求**：每日竞品价格爬虫 + 我们的日销量数据（用于估计交叉弹性）

**预期产出**：
- 竞品价格仪表盘：5 竞品 × 3 变体（单边/双边/穿戴式）价格热力图
- 预警规则：PCI > 1.10 且竞品降价 > 10% 时触发橙色预警
- 响应建议：预计不跟降损失 15-20% 日销量 vs 跟降 10% 保护份额但利润率降 3pp

**业务价值**：避免被动反应（跟降太多/太少），预计每月优化定价决策节约 $2,000-5,000

---

## ③ 代码模板

```python
"""Competitive Price Monitoring — PCI 计算 + 响应决策"""

import numpy as np
from typing import List, Dict, Tuple


def price_competitiveness_index(
    our_price: float, competitor_prices: List[float]
) -> float:
    """价格竞争力指数"""
    if not competitor_prices:
        return 1.0
    return our_price / np.median(competitor_prices)


def estimate_cross_elasticity(
    our_sales: List[float], competitor_prices: List[float],
    window: int = 7
) -> float:
    """用滚动窗口估计交叉价格弹性"""
    if len(our_sales) < window * 2:
        return -0.8  # 默认值
    
    elasticities = []
    for t in range(window, len(our_sales) - window):
        comp_change = (competitor_prices[t] - competitor_prices[t-window]) / competitor_prices[t-window]
        sales_change = (our_sales[t+window] - our_sales[t]) / max(our_sales[t], 0.01)
        if abs(comp_change) > 0.01:
            elasticities.append(sales_change / comp_change)
    
    return np.median(elasticities) if elasticities else -0.8


def price_response_decision(
    our_price: float, our_cost: float, our_sales: float,
    competitor_prices: List[float], cross_elasticity: float = -0.8,
) -> Dict:
    """竞品响应决策"""
    pci = price_competitiveness_index(our_price, competitor_prices)
    min_comp = min(competitor_prices)
    
    # 场景分析
    if pci < 1.03:
        return {'action': 'maintain', 'pci': pci, 'reason': '价格有竞争力'}
    
    # 跟降 10% 的效果
    target_price = min_comp * 1.02
    price_drop_pct = (our_price - target_price) / our_price
    expected_sales_gain = cross_elasticity * (-price_drop_pct) * our_sales
    profit_change = (target_price - our_cost) * (our_sales + expected_sales_gain) - \
                    (our_price - our_cost) * our_sales
    
    if profit_change > 0 and price_drop_pct > 0.03:
        return {
            'action': 'match', 'target_price': round(target_price, 2),
            'pci': pci, 'expected_sales_gain': round(expected_sales_gain),
            'profit_change': round(profit_change), 'drop_pct': f'{price_drop_pct:.1%}'
        }
    elif pci > 1.10:
        return {'action': 'partial_match', 'pci': pci,
                'target_price': round(our_price * 0.95, 2),
                'reason': 'PCI过高但利润不允许完全匹配'}
    else:
        return {'action': 'maintain', 'pci': pci, 'reason': '利润变化为负'}


# ============ 测试 ============

if __name__ == '__main__':
    result = price_response_decision(
        our_price=129, our_cost=80, our_sales=100,
        competitor_prices=[119, 115, 125, 99, 109],  # 竞品中位数 115
        cross_elasticity=-0.8
    )
    print(f"PCI={result['pci']:.2f} → 决策={result['action']}")
    if 'target_price' in result:
        print(f"  目标价=${result['target_price']}, "
              f"预计销量变化={result.get('expected_sales_gain','?')}件, "
              f"利润变化=${result.get('profit_change','?')}")
    
    assert price_competitiveness_index(100, [90, 100, 110]) > 0.9
    print("\n[✓] Competitive Price Monitoring 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Dynamic-Pricing-Elasticity]] | [[Skill-Ad-Attribution-Modeling]]（DiD/SCM 准实验方法）
- **延伸技能**：[[Skill-Cross-Border-Price-Harmonization]] | [[Skill-Markdown-Optimization]]
- **可组合技能**：[[Skill-CABB-Cross-Category-Attribution]]（跨品类的交叉弹性估计）

---

## ⑤ 商业价值评估

- **ROI 预估**：优化竞品响应决策，每月节约 $2,000-5,000；年化 **25-60 万元**
- **实施难度**：⭐⭐☆☆☆（2 星）— 爬虫 + PCI 计算简单
- **优先级评分**：⭐⭐⭐⭐☆（4 星）— 跨境价格战是日常挑战
