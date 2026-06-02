# Skill Card: Network Effect Experiments（网络效应实验）

> **领域**: 02-A_B实验 | **类型**: 综合萃取

---

## ① 算法原理

标准 A/B 假设 SUTVA（用户间无干扰），但社交电商（分享/推荐/UGC）违反此假设——A 组用户的行为会影响 B 组用户。解决方案：**Cluster Randomization**（按社交簇随机分组）或 **Two-Stage Randomization**（先随机簇，簇内再随机个体）。

---

## ② 母婴出海应用案例

测试"推荐有奖"功能——A 组用户分享推荐链接，B 组收到推荐。SUTVA 违反：B 组的购买行为被 A 组的推荐影响。Cluster randomization：按"妈妈群"分组，群内同 treatment，群间独立。

年化：确保社交功能实验正确性，隐性 **10-20 万元**。

---

## ③ 代码模板

```python
import numpy as np

def cluster_randomize(users: list, clusters: list, n_treat: int):
    unique_clusters = list(set(clusters))
    treat_clusters = set(np.random.choice(unique_clusters, n_treat, replace=False))
    assignments = ['treat' if c in treat_clusters else 'control' for c in clusters]
    return assignments

clusters = [f'group_{i//10}' for i in range(100)]
assign = cluster_randomize(range(100), clusters, len(set(clusters))//2)
print(f"Treatment ratio: {assign.count('treat')/len(assign):.0%}")
print("[✓] Network Effect 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-AB-Experimental-Design]] | [[Skill-CUPED-Variance-Reduction]]

---

- **可组合**：[[Skill-Uplift-Modeling]] / [[Skill-Multi-Armed-Bandit]]

## ⑤ 商业价值：10-20 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐☆☆
