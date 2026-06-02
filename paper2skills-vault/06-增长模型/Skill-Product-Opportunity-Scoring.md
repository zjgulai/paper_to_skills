# Skill Card: Product Opportunity Scoring（新品机会评分卡）

> **领域**: WF-D 选品扫描 | **归属**: 06-增长模型 | **类型**: 综合萃取

---

## ① 算法原理

多维度加权评分卡，综合评估新品机会：

$$Score = \sum_{d} w_d \cdot \text{normalize}(metric_d)$$

六大维度：
1. **市场规模**（搜索量 + BSR 品类总量，$w$=0.25）
2. **竞争强度**（竞品数量 + 集中度 HHI，$w$=0.20，反比）
3. **利润空间**（估算毛利 = 售价 - 成本 - FBA 费，$w$=0.20）
4. **趋势方向**（来自 Category Trend Forecasting，$w$=0.15）
5. **合规风险**（认证要求 + 专利风险，$w$=0.10，反比）
6. **运营复杂度**（体积/重量/退货率，$w$=0.10，反比）

$Score > 0.65$ → 高优先级选品；$0.45-0.65$ → 候选池；$<0.45$ → 暂缓。

---

## ② 母婴出海应用案例

候选新品"智能温控奶瓶"评分：市场大（0.8）× 竞争中等（0.6）× 利润高（0.85）× 趋势强（0.75）× 合规风险中等（0.5）× 复杂度中等（0.6）→ 综合 0.71。**高优先级**，进入选品短名单。

候选"婴儿电动摇椅"评分 0.38（合规风险极高 + 退货率高）→ **暂缓**。

---

## ③ 代码模板

```python
"""Product Opportunity Scoring"""

def opportunity_score(metrics: dict, weights: dict = None):
    w = weights or {'market_size': 0.25, 'competition': 0.20, 'margin': 0.20,
                     'trend': 0.15, 'compliance': 0.10, 'complexity': 0.10}
    # competition, compliance, complexity are inverse (higher=worse)
    if 'competition' in metrics: metrics['competition'] = 1 - metrics['competition']
    if 'compliance' in metrics: metrics['compliance'] = 1 - metrics['compliance']
    if 'complexity' in metrics: metrics['complexity'] = 1 - metrics['complexity']
    return sum(w.get(k, 0) * v for k, v in metrics.items())

def classify(score):
    return "HIGH" if score > 0.65 else ("CANDIDATE" if score > 0.45 else "LOW")

# test
s = opportunity_score({'market_size':0.8, 'competition':0.4, 'margin':0.85, 'trend':0.75, 'compliance':0.5, 'complexity':0.4})
print(f"Score: {s:.2f} → {classify(s)}")
assert s > 0.65
print("[✓] Product Opportunity Scoring 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Category-Trend-Forecasting]] | [[Skill-Competitor-Product-Intelligence]]
- **组合**：[[Skill-Supplier-Evaluation-Model]] | [[Skill-Dynamic-Pricing-Elasticity]]

---
- **相关技能**：[[Skill-Product-Lifecycle-Stage]]
- **相关技能**：[[Skill-Cross-Market-Product-Transfer]]
- **相关技能**：[[Skill-Market-Size-Estimation]]
- **相关技能**：[[Skill-Cross-Border-Cold-Start-Forecast]]

## ⑤ 商业价值

- **ROI**：系统化选品减少试错成本 50%+；年化 **40-80 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐⭐（5 星）
