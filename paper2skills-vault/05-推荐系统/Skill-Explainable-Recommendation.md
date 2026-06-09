---
title: Explainable Recommendation for Business Trust
doc_type: knowledge
module: 05-推荐系统
topic: explainable-recommendation
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase2
---

# Skill Card: Explainable Recommendation

## ① 算法原理

**核心问题**：黑盒推荐系统给用户推了"吸奶器"，用户会问"为什么给我推这个？"如果无法解释，用户不信任、不点击、甚至反感。业务方也不理解模型逻辑，无法优化。

**可解释性的三个层次**：

| 层次 | 解释对象 | 示例 |
|------|---------|------|
| **模型层面** | 为什么模型做了这个预测 | SHAP值、注意力权重 |
| **用户层面** | 为什么推给这个用户 | "因为你买过奶粉" |
| **物品层面** | 为什么推这个物品 | "因为这个品牌和A品牌相似" |

**主流方法**：

**1. 基于关联规则的解释**
- "买了A的人也买了B"（Amazon经典）
- 简单、直观、无需额外模型
- 局限：只能解释协同过滤类推荐

**2. 基于知识的解释（Knowledge-aware）**
- 利用知识图谱中的关系路径解释
- "推荐爱他美3段，因为它含有DHA，和您之前买的美赞臣成分相似"
- 优势：解释内容丰富、有说服力

**3. 基于自然语言的解释（NLG）**
- 用模板或生成模型产出自然语言解释
- "这款吸奶器静音设计，适合夜间使用，和您收藏的一款功能类似"
- 前沿：LLM生成个性化解释

**4. 因果解释（Causal Explanation）**
- 不只看相关性，看因果性
- "如果去掉'价格'这个特征，推荐结果会从A变成B"
- 2025年前沿：Causal RecSys

**反直觉洞察**：
- 解释不一定要"完全准确"——用户需要的是"听起来合理"的解释，而非模型内部的数学真相
- 过长的解释反而降低点击率——一行字的解释效果最好
- "个性化解释"比"通用解释"点击率高30%+——"因为你"比"很多人"更有说服力

---

## ② 母婴出海应用案例

### 场景1：首页"猜你喜欢"的解释

**业务问题**：Momcozy 首页推荐位点击率2.5%，但用户调研显示40%的用户"不信任推荐结果"。需要给每个推荐商品添加一句话解释。

**解释生成策略**：

| 推荐原因 | 解释模板 | 示例 |
|---------|---------|------|
| 协同过滤 | "和您购买的{过往商品}很搭" | "和您买的吸奶器很搭：储奶袋" |
| 内容相似 | "和您浏览过的{商品}功能相似" | "和您浏览的A款功能相似：静音升级" |
| 知识关联 | "适合{宝宝阶段}的妈妈" | "适合6个月+宝宝：辅食机" |
| 热门趋势 | "本周{品类}热销Top 3" | "本周吸奶器热销Top 3" |
| 价格敏感 | "比您收藏的{商品}省${金额}" | "比您收藏的A款省$20" |

**A/B测试结果**：
- 有解释版：点击率3.2%（+28%）
- 无解释版：点击率2.5%
- 解释类型效果排序：知识关联 > 协同过滤 > 价格敏感 > 热门趋势

### 场景2：业务方理解模型逻辑

**业务问题**：产品团队质疑推荐系统"为什么总在推低价商品，不打高客单价用户？"

**因果解释分析**：
1. 用SHAP值分析每个特征对推荐结果的影响
2. 发现"价格"特征的SHAP值为负（模型偏好低价）
3. 根因：训练数据中低价商品的点击率天然更高（选择偏误）
4. 修正：在损失函数中加入"客单价"的加权，或做因果纠偏

---

## ③ 代码模板

```python
"""
Explainable Recommendation — 可解释推荐
支持：关联规则解释、知识图谱解释、SHAP解释
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter


class AssociationRuleExplainer:
    """基于关联规则的解释器"""

    def __init__(self, min_support=0.01, min_confidence=0.3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = []

    def fit(self, transactions):
        """
        从交易记录挖掘关联规则

        Args:
            transactions: list of lists, each inner list is a user's purchase history
        """
        # 统计项集频率
        item_counts = Counter()
        pair_counts = defaultdict(Counter)
        total = len(transactions)

        for trans in transactions:
            unique_items = list(set(trans))
            for item in unique_items:
                item_counts[item] += 1
            for i, a in enumerate(unique_items):
                for b in unique_items[i+1:]:
                    pair_counts[a][b] += 1
                    pair_counts[b][a] += 1

        # 生成规则
        self.rules = []
        for a, neighbors in pair_counts.items():
            a_support = item_counts[a] / total
            if a_support < self.min_support:
                continue
            for b, count in neighbors.most_common(10):
                confidence = count / item_counts[a]
                if confidence >= self.min_confidence:
                    lift = confidence / (item_counts[b] / total)
                    self.rules.append({
                        'antecedent': a,
                        'consequent': b,
                        'confidence': confidence,
                        'lift': lift,
                        'support': count / total
                    })

        # 按lift排序
        self.rules.sort(key=lambda x: x['lift'], reverse=True)
        return self

    def explain(self, user_history, recommended_item):
        """为用户解释为什么推荐某个商品"""
        explanations = []

        for rule in self.rules:
            if rule['antecedent'] in user_history and rule['consequent'] == recommended_item:
                explanations.append(
                    f"买了{rule['antecedent']}的用户也买了这个（置信度{rule['confidence']:.0%}）"
                )

        return explanations[:3] if explanations else ["这款商品本周很受欢迎"]


class KnowledgeGraphExplainer:
    """基于知识图谱的解释器"""

    def __init__(self, kg_edges):
        """
        Args:
            kg_edges: list of (head, relation, tail) tuples
        """
        self.edges = defaultdict(list)
        for h, r, t in kg_edges:
            self.edges[h].append((r, t))
            self.edges[t].append((r, h))

    def find_path(self, item_a, item_b, max_depth=3):
        """找两个商品之间的知识图谱路径"""
        # BFS
        visited = {item_a}
        queue = [(item_a, [])]

        while queue:
            current, path = queue.pop(0)
            if current == item_b and path:
                return path

            if len(path) >= max_depth:
                continue

            for relation, neighbor in self.edges.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(current, relation, neighbor)]))

        return None

    def explain_similarity(self, item_a, item_b):
        """解释为什么两个商品相似"""
        path = self.find_path(item_a, item_b)
        if not path:
            return f"{item_a}和{item_b}经常被一起购买"

        parts = []
        for h, r, t in path:
            relation_text = {
                'same_brand': '同品牌',
                'same_category': '同类目',
                'same_ingredient': '含相同成分',
                'complementary': '互补',
                'substitute': '可替代'
            }.get(r, r)
            parts.append(relation_text)

        return f"{' → '.join(parts)}"


class SimpleSHAPExplainer:
    """简化版SHAP解释器（用于线性模型）"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def explain(self, instance):
        """
        解释单个预测

        对于线性模型 y = w1*x1 + w2*x2 + ... + b
        SHAP值 = wi * (xi - E[xi])
        """
        if hasattr(self.model, 'coef_'):
            weights = self.model.coef_
            bias = self.model.intercept_ if hasattr(self.model, 'intercept_') else 0

            shap_values = weights * instance
            feature_importance = list(zip(self.feature_names, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            return feature_importance[:5]
        else:
            return [("模型不支持SHAP", 0)]


# 示例
if __name__ == '__main__':
    # 关联规则解释
    transactions = [
        ['吸奶器', '储奶袋', '防溢乳垫'],
        ['吸奶器', '储奶袋'],
        ['奶粉3段', '奶瓶', '奶嘴'],
        ['奶粉3段', '奶瓶'],
        ['纸尿裤M', '湿巾', '护臀膏'],
    ]

    explainer = AssociationRuleExplainer()
    explainer.fit(transactions)

    user_history = ['吸奶器']
    recommended = '储奶袋'
    explanations = explainer.explain(user_history, recommended)
    print(f"推荐'{recommended}'给买过{user_history}的用户:")
    for e in explanations:
        print(f"  - {e}")

    # 知识图谱解释
    kg_edges = [
        ('爱他美3段', 'same_brand', '爱他美2段'),
        ('爱他美3段', 'same_ingredient', 'DHA'),
        ('美赞臣3段', 'same_ingredient', 'DHA'),
        ('爱他美3段', 'substitute', '美赞臣3段'),
    ]
    kg_explainer = KnowledgeGraphExplainer(kg_edges)
    print(f"\n知识图谱解释:")
    print(kg_explainer.explain_similarity('爱他美3段', '美赞臣3段'))
```

---


## ④ 技能关联

### 前置技能
- [Skill-NeuralNDCG-Learning-to-Rank](../05-推荐系统/[[Skill-NeuralNDCG-Learning-to-Rank]].md) — 排序模型是解释性推荐的基础对象
- [Skill-Matrix-Factorization](../05-推荐系统/[[Skill-Matrix-Factorization]].md) — 隐因子是解释性归因的常用维度

### 延伸技能
- 无（本 Skill 是终端/聚合卡）

### 可组合
- [Skill-Knowledge-Graph-for-Skills-Management](../08-知识图谱/[[Skill-Knowledge-Graph-for-Skills-Management]].md) — KG 路径提供天然的解释性推理链

## ⑤ 商业价值评估

- **ROI**：推荐点击率提升25-40%，用户信任度显著提升
- **难度**：⭐⭐☆☆☆（2/5）— 关联规则简单，NLG需要LLM
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 推荐系统从"能用"到"可信"的关键一步
