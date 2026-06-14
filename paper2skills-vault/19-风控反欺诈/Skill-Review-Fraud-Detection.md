# Skill Card: Review Fraud Detection（虚假评论检测）

> **领域**: 19-风控反欺诈 | **类型**: 综合萃取

roadmap_phase: phase1
---

## ① 算法原理

GNN 图神经网络检测虚假评论——不是看单条评论文本，而是看**评论者-产品-评分之间的关系图**。虚假评论团通常呈现异常图模式：同批次账号、评分极端（1 或 5 星）、评论时间集中、文本相似度高。

**异构图**：节点=用户/产品/评论，边=写了/属于/评分。GCN 聚合邻居特征，检测异常子图密度和评分偏差。

$$\text{AnomalyScore}(u) = \|\text{GNN}(u) - \text{GNN}(\text{normal\_neighbors})\|_2$$

---

## ② 母婴出海应用案例

吸奶器 listing 突然收到 20 条 5 星好评，全部来自新账号（注册<30 天），文本相似度 0.85+。GNN 检测为虚假评论团，触发删除+上报 Amazon。避免被竞品恶意刷评导致 list 被限流。

年化止损：**5-15 万元**（避免 listing 降权 + 处罚）。

---

## ③ 代码模板

```python
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_review_fraud(features: np.ndarray):
    """features: [user_age, rating, text_similarity_to_others, time_cluster, ...]"""
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(features)
    fraud_ratio = (preds == -1).mean()
    return {'fraud_ratio': fraud_ratio, 'fraud_indices': np.where(preds == -1)[0]}

# test: 100 reviews, 5% fraud
X = np.random.randn(100, 5)
# inject anomaly cluster
X[-5:] = np.array([[0.1, 5.0, 0.9, 0.05, 0.95]]*5)
r = detect_review_fraud(X)
print(f"Fraud detected: {r['fraud_ratio']:.0%}, last 5 indices: {r['fraud_indices'][-5:]}")
assert all(i >= 94 for i in r['fraud_indices'][-5:])
print("[✓] Review Fraud Detection 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Feature-Engineering]] | [[Skill-Imbalanced-Data-Handling]]
- **组合**：[[Skill-AGRS-Aspect-Guided-Review-Summarization]]（过滤虚假后做真实摘要）

---
- **相关技能**：[[Skill-FraudSquad-LLM-Review-Detection]]
- **相关技能**：[[Skill-DS-DGA-GCN-Fake-Review-Group]]
- **相关技能**：[[Skill-Transaction-Anomaly-Detection]]
- **跨域关联**：[[Skill-KG-Auto-Construction-Agent-Driven]]

## ⑤ 商业价值

- **ROI**：5-15 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐⭐☆（填补图谱 HIGH 缺口）
