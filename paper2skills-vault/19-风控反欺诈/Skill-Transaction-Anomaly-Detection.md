# Skill Card: Transaction Anomaly Detection（异常交易检测）

> **领域**: 19-风控反欺诈 | **类型**: 综合萃取

---

## ① 算法原理

Isolation Forest + 动态阈值检测异常交易模式。特征：订单金额、支付方式、IP 国家 vs 收货国家、下单到支付间隔、同一 IP 下单频率、地址变更次数。

**动态阈值**：滚动 30 天窗口计算每个特征的 baseline($\mu, \sigma$)，异常分数 $z = \max_i |(x_i - \mu_i) / \sigma_i|$。

---

## ② 母婴出海应用案例

同一个 IP（印尼）在 10 分钟内下了 5 单吸奶器，收货地址美国不同州，使用 5 张不同信用卡。$z$-score 4.8 → 触发高风险预警，自动 hold 订单等待人工审核。拦截盗刷订单 $5000。

年化止损：**3-8 万元**。

---

## ③ 代码模板

```python
import numpy as np
from sklearn.ensemble import IsolationForest

def transaction_anomaly_score(features, window_history):
    """z-score anomaly detection with rolling baseline"""
    mu, sigma = window_history.mean(axis=0), window_history.std(axis=0) + 1e-6
    z_scores = np.abs((features - mu) / sigma)
    max_z = z_scores.max(axis=1)
    return {'z_scores': max_z, 'high_risk': max_z > 3.5}

# test
hist = np.random.randn(500, 6) * 0.5
curr = np.array([[3.0, 4.2, -2.8, 3.5, 0.1, 4.5]])  # anomalous
r = transaction_anomaly_score(curr, hist)
print(f"z-score: {r['z_scores'][0]:.1f}, high_risk: {r['high_risk'][0]}")
assert r['high_risk'][0]
print("[✓] Transaction Anomaly 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Feature-Engineering]]
- **组合**：[[Skill-Review-Fraud-Detection]]（统一风控体系）

---
- **相关技能**：[[Skill-Click-Fraud-Detection]]
- **相关技能**：[[Skill-FraudSquad-LLM-Review-Detection]]
- **相关技能**：[[Skill-DS-DGA-GCN-Fake-Review-Group]]
- **关联**：[[Skill-ROAS-Budget-Optimization]]

## ⑤ 商业价值

- **ROI**：3-8 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
