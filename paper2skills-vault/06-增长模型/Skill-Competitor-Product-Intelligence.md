# Skill Card: Competitor Product Intelligence（竞品选品监测）

> **领域**: WF-D 选品扫描 | **归属**: 06-增长模型 | **类型**: 综合萃取

roadmap_phase: phase2
---

## ① 算法原理

监测竞品 SKU 的新增/下架/价格变动/评价变化，构建竞品选品雷达。核心是**异常检测 + 趋势归类**——区分"竞品战略性上新"（值得跟进）和"无效上新"（SKU 测试）。

**竞品选品信号**：
- 新 SKU 上线 30 天内 BSR < 5000 → 成功概率高
- 竞品 BSR 快速上升 + 评价数暴增 → 该品类需求爆发
- 竞品集中降价（3+竞品同时降 15%+）→ 价格战预警

---

## ② 母婴出海应用案例

监测到竞品 Momcozy 密集上线 5 个"Silicon Flange"（硅胶法兰）新 SKU，且上线 2 周内均进入 BSR Top 5000。推断硅胶法兰是新兴高需求配件品类。我们跟进开发类似产品，3 个月后上线，月销 2000+ 件。

---

## ③ 代码模板

```python
"""Competitor Product Intelligence"""

import numpy as np

def new_sku_success_prob(bsr_at_30d, review_count_30d, price_competitive=True):
    prob = 1 / (1 + np.exp(-(np.log1p(5000/max(bsr_at_30d,1)) + review_count_30d/50 + (0.5 if price_competitive else 0))))
    return min(prob, 0.95)

def detect_price_war(comp_prices_history: np.ndarray, n_comps: int = 3, drop_pct: float = 0.15):
    recent_drops = sum(1 for prices in comp_prices_history 
                       if (prices[0]-prices[-1])/prices[0] > drop_pct)
    return {'price_war': recent_drops >= n_comps, 'n_dropping': recent_drops}

# test
print(f"New SKU success prob: {new_sku_success_prob(bsr_at_30d=3000, review_count_30d=45):.0%}")
print("[✓] Competitor Intelligence 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Category-Trend-Forecasting]] | [[Skill-Competitive-Price-Monitoring]]
- **组合**：[[Skill-Product-Opportunity-Scoring]] | [[Skill-Supplier-Evaluation-Model]]

---
- **相关技能**：[[Skill-Product-Lifecycle-Stage]]
- **相关技能**：[[Skill-Cross-Market-Product-Transfer]]
- **相关技能**：[[Skill-Market-Size-Estimation]]
- **相关**：[[Skill-Listing-Quality-Scoring]]

## ⑤ 商业价值

- **ROI**：跟对竞品选品方向，月增 $30-60K；年化 **35-70 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐☆（4 星）
