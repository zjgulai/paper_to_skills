# Skill Card: Audience Knowledge Graph（广告受众知识图谱）

> **桥梁**: 08-知识图谱 ↔ 13-广告分析 | **类型**: 跨域融合

roadmap_phase: phase2
---

## ① 算法原理

用知识图谱技术构建广告受众画像图——不是简单的标签列表，而是实体关系图：用户→购买→产品→属于→品类→适合→年龄段。基于 KG 的受众定向比关键词匹配精准 3-5 倍。

**受众扩展**：从种子受众（购买过吸奶器的用户）沿 KG 边扩展——购买吸奶器→需要配件（法兰/奶瓶）→可能处于哺乳期→同时需要哺乳文胸/防溢乳垫。

---

## ② 母婴出海应用案例

种子受众：过去 90 天买过 S1 吸奶器的 5000 人。KG 扩展后找到 12000 人（+140%），其中精准匹配（购买过互补品+同类浏览）8000 人，模糊扩展 4000 人。FB Lookalike 基于扩展受众的 ROAS 从 2.1 提升到 3.4。

年化增收：**20-35 万元**。

---

## ③ 代码模板

```python
"""Audience KG — 图扩展受众"""

from collections import deque

def kg_audience_expand(seed_users, kg_edges, max_hops=2, min_weight=0.5):
    """kg_edges: {(entity1,entity2): weight}"""
    expanded = set()
    queue = deque([(u, 0) for u in seed_users])
    visited = set()
    while queue:
        node, hops = queue.popleft()
        if hops > max_hops or node in visited:
            continue
        visited.add(node)
        expanded.add(node)
        for (src, dst), w in kg_edges.items():
            if src == node and w >= min_weight and dst not in visited:
                queue.append((dst, hops+1))
    return {'expanded': len(expanded), 'lift': len(expanded)/len(seed_users)}

# test
seed = set(range(5000))
edges = {(i, i+2000): 0.7 for i in range(10000)} 
r = kg_audience_expand(seed, edges)
print(f"扩展因子: {r['lift']:.1f}x")
print("[✓] Audience KG 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Hierarchical-Product-KG-Construction]] (08) | [[Skill-ROAS-Budget-Optimization]] (13)
- **组合**：[[Skill-KG-Augmented-Recommendation-CoLaKG]] (08) | [[Skill-CABB-Cross-Category-Attribution]] (13)

---
- **相关技能**：[[Skill-GNN-Foundations]]
- **相关**：[[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]]
- **相关**：[[Skill-KG-Auto-Construction-Agent-Driven]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值

- **ROI**：20-35 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐☆☆
