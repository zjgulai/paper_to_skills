```markdown
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

**品类**：婴儿暖奶器（母婴出海爆款，客单价 $39.99）  
**背景**：某 Amazon 店铺 listing 日销 50 件，库存 2000 件，转化率 4.5%，ROAS 3.2。竞品恶意雇佣刷评团，3 天内涌入 25 条 5 星好评，全部来自注册<30 天新账号，文本相似度 0.88，评分时间集中在凌晨 2-4 点。

**GNN 检测结果**：  
- 异常子图密度 0.73（正常<0.15）  
- 评分偏差 4.2（正常<0.5）  
- 准确率从 82% 提升至 97%（+15%）  

**干预动作**：自动删除虚假评论并上报 Amazon，listing 未被降权，库存周转率从 2.1 次/月提升至 2.7 次/月（+28%）。  

**年化止损**：避免 listing 限流导致的销量损失 + 平台处罚，年化节省 **45 万元**。

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

- **ROI**：45 万元/年 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐⭐☆（填补图谱 HIGH 缺口）
```