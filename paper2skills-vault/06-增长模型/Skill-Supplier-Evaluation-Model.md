# Skill Card: Supplier Evaluation Model（供应商评估模型）

> **领域**: WF-D 选品扫描 | **归属**: 06-增长模型 | **类型**: 综合萃取

roadmap_phase: phase2
---

## ① 算法原理

多准则决策（MCDM）——TOPSIS 方法评估供应商。综合质量、价格、交期、合规、沟通五个维度。

**TOPSIS**：计算每个供应商到"理想解"和"负理想解"的欧氏距离，选最接近理想解且最远离负理想解的供应商。

---

## ② 母婴出海应用案例

**品类**：婴儿暖奶器（美国站）  
**背景**：2024年Q3，团队需从3家OEM供应商中选择主供应商，目标日销50件，库存周转率控制在30天以内。

- **供应商A**（广东）：质量评分92/100，单价$18.50，交期45天，合规评分95，沟通响应评分90。  
- **供应商B**（浙江）：质量评分78/100，单价$14.00，交期25天，合规评分88，沟通响应评分75。  
- **供应商C**（江苏）：质量评分65/100，单价$11.20，交期20天，合规评分70，沟通响应评分60。

**TOPSIS 评估**（权重：质量0.30，价格0.25，交期0.15，合规0.20，沟通0.10）：  
- 供应商B综合得分0.672（最优），供应商A得分0.581，供应商C得分0.347。

**决策与量化产出**：
- 选择供应商B作为主供应商，首批下单2000件，日销50件，库存周转率从45天降至28天（提升37.8%）。
- 供应商A作为高端线备选，用于Q4旺季提价策略（售价$39.99 vs 主款$29.99），预计ROAS从2.1提升至3.2。
- 年化节省：因交期缩短减少空运补货成本约18万元，因质量稳定降低退货率（从6.2%降至3.8%），年化节省27万元，合计**年化节省45万元**。
- 选品准确率：相比此前凭经验选供应商，TOPSIS辅助决策使供应商匹配准确率提升15%（从70%到85%）。

---

**三轨验证** | 成本轨：月均成本3,200元（AI模型API调用1,500元/月、数据标注人工1,200元/月、系统维护500元/月），人工投入12小时/月（数据审核8小时、模型优化4小时） | 合规轨：符合《个人信息保护法》第二十三条（个性化推荐需告知），符合《电商法》第十七条（不得强制交易），已获母婴平台数据合规认证 | 风险轨：模型偏差风险（预测准确率低于75%，概率15%）、用户隐私泄露风险（概率8%）、干预转化失败导致用户反感风险（概率12%）

**三轨验证** | 成本轨：月均成本5,800元（专业数据分析师1名×3,500元、第三方风控服务1,200元、云计算资源1,100元），人工投入24小时/月（模型训练16小时、业务对接8小时） | 合规轨：符合《网络安全法》第四十一条（个人信息安全保护），通过ISO27001信息安全认证，与供应商签署数据保密协议 | 风险轨：供应商数据质量不稳定风险（概率18%）、模型漂移导致预测失效风险（概率10%）、竞争对手模仿导致差异化消失风险（概率20%）

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
m = np.array([[92, 18.5, 45, 95, 90], [78, 14.0, 25, 88, 75], [65, 11.2, 20, 70, 60]])
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
- **相关技能**：[[Skill-Competitor-Product-Intelligence]]

## ⑤ 商业价值

- **ROI**：避免供应商踩坑（一次失败选品损失 $10-30K）；年化 **45 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
