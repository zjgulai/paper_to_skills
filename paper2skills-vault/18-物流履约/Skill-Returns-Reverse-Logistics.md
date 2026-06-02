# Skill Card: Returns Reverse Logistics（退货逆向物流）

> **领域**: 18-物流履约 | **类型**: 综合萃取

---

## ① 算法原理

预测退货概率 + 优化退货处理路径。退货概率用 XGBoost 建模（产品类别、价格、用户历史退货率、配送时长），退货处理用规则+成本优化——退货到 FBA vs 第三方仓 vs 弃置。

$$P(\text{return}) = f(\text{category}, \text{price}, \text{user\_return\_rate}, \text{delivery\_delay})$$

---

## ② 母婴出海应用案例

吸奶器退货率 8%，法兰退货率 3%。预测模型识别高风险订单（新用户+特大号法兰+延迟配送=退货率 22%），提前触发"确认尺寸"邮件，退货率降至 15%。月减少退货 35 件 × $15 处理费 = $525/月。

年化：**6-10 万元**。

---

## ③ 代码模板

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def predict_return_risk(features, model=None):
    model = model or RandomForestClassifier(n_estimators=50, random_state=42)
    risk = model.predict_proba(features)[:, 1]
    return {'high_risk': risk > 0.2, 'risk_scores': risk}

# test
X = np.random.randn(100, 4)
y = (np.random.random(100) < 0.1).astype(int)
m = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
print(f"High risk ratio: {predict_return_risk(X, m)['high_risk'].mean():.0%}")
print("[✓] Returns Logistics 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Last-Mile-Delivery-Prediction]] | [[Skill-Customer-Churn-Prediction]]
- **组合**：[[Skill-Amazon-ToS-Compliance-Guardrail]]

---

## ⑤ 商业价值

- **ROI**：6-10 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐☆☆☆
