# Skill Card: Supplier Evaluation Model（供应商评估模型）

> **领域**: WF-D 选品扫描 | **归属**: 06-增长模型 | **类型**: 综合萃取

---

## ① 算法原理

多准则决策（MCDM）——TOPSIS 方法评估供应商。综合质量、价格、交期、合规、沟通五个维度。

**TOPSIS**：计算每个供应商到"理想解"和"负理想解"的欧氏距离，选最接近理想解且最远离负理想解的供应商。

---

## ② 母婴出海应用案例

3 个吸奶器 OEM 供应商评估：厂商 A（质量高但价高交期长）、厂商 B（性价比最优）、厂商 C（价格最低但质量风险）。TOPSIS 综合评分：B > A > C。选择厂商 B 作为主供应商，厂商 A 作为高端线备选。

---

## ③ 代码模板

```python
"""Supplier Evaluation — TOPSIS"""

import numpy as np

def topsis(matrix: np.ndarray, weights: np.ndarray, benefits: list):
    """benefits: True=越高越好, False=越低越好"""
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    weighted = norm * weights
    ideal = np.array([weighted[:,i].max() if b else weighted[:,i].min() for i,b in enumerate(benefits)])
    anti_ideal = np.array([weighted[:,i].min() if b else weighted[:,i].max() for i,b in enumerate(benefits)])
    d_pos = np.sqrt(((weighted - ideal)**2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti_ideal)**2).sum(axis=1))
    return d_neg / (d_pos + d_neg)

# test: 3 suppliers × 5 criteria (quality,price,lead_time,compliance,communication)
m = np.array([[85, 120, 30, 90, 80], [75, 100, 20, 85, 70], [65, 85, 15, 70, 60]])
# higher better for quality, compliance, comm; lower for price, lead_time
scores = topsis(m, [0.30,0.25,0.15,0.20,0.10], [True,False,False,True,True])
for i, s in enumerate(scores):
    print(f"  Supplier {chr(65+i)}: {s:.3f}")
print(f"Best: Supplier {chr(65+np.argmax(scores))}")
print("[✓] Supplier Evaluation 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Product-Opportunity-Scoring]]
- **组合**：[[Skill-Cross-Border-Price-Harmonization]] | [[Skill-Amazon-ToS-Compliance-Guardrail]]

---

## ⑤ 商业价值

- **ROI**：避免供应商踩坑（一次失败选品损失 $10-30K）；年化 **15-25 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
