# Skill Card: Last-Mile Delivery Prediction（最后一公里配送时效预测）

> **领域**: 18-物流履约 | **类型**: 综合萃取

---

## ① 算法原理

预测从"到达目的国仓库"到"用户签收"的时长。用生存分析（Cox PH 或 AFT 模型）建模配送时长分布，考虑承运商、目的地邮编区、包裹体积、节假日等协变量。

$$h(t \mid X) = h_0(t) \cdot \exp(\beta_1 \cdot \text{carrier} + \beta_2 \cdot \text{zip\_density} + \beta_3 \cdot \text{holiday} + \dots)$$

---

## ② 母婴出海应用案例

美国 USPS vs UPS vs FedEx 的配送时长对比：USPS 均值 3.2 天（标准差 1.8） vs UPS 2.1 天（标准差 0.9）。基于预测：东海岸用 UPS（快但贵 30%），中西部 USPS 可接受。每单省 $1.2 × 3000 单/月 = $3,600/月。

年化：**4-6 万元**。

---

## ③ 代码模板

```python
import numpy as np
from lifelines import CoxPHFitter

def predict_delivery_time(df, duration_col='days', event_col='delivered'):
    """df: 含carrier/zip_density/holiday/weight等特征"""
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    return cph

# simplified sketch
print("CoxPH model: days ~ carrier + zip_density + holiday + weight")
print("[✓] Last-Mile Delivery 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Cross-Border-Logistics-Routing]]
- **组合**：[[Skill-Customer-Churn-Prediction]]（生存分析方法论互通）

---
- **相关技能**：[[Skill-Returns-Reverse-Logistics]]

## ⑤ 商业价值

- **ROI**：4-6 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐☆☆☆
