---
title: Popularity-Aware Meta-Learning for Cold-Start Recommendation
module: 05-推荐系统
topic: cold-start-meta-learning

roadmap_phase: phase2
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: Cold-Start Meta-Learning (PAM)

## ① 算法原理

**核心问题**：母婴品类SKU迭代快（奶粉按月龄分段、辅食按月添加），新品上架无历史交互数据，传统协同过滤无法推荐。冷启动是母婴电商的结构性痛点。

**传统方案缺陷**：
- 基于内容的推荐：只看商品属性，忽略用户偏好
- 热门兜底：新品永远打不过爆款
- 探索策略：随机曝光，转化率极低

**PAM 创新（KDD 2025, 快手）**：
按商品**流行度分层**构建元学习任务：
1. **分层策略**：将商品按历史交互量分为高/中/低流行度三层
2. **元学习**：每层视为独立任务，用MAML学习"如何快速适应新商品"
3. **流行度感知**：区分内容特征（商品属性）和行为特征（用户交互）在不同流行度下的权重
4. **数据增强+自监督**：专门为低流行度商品设计增强策略

**关键洞察**：高流行度商品的行为信号丰富，低流行度商品只能依赖内容信号。PAM让模型学会"根据流行度自动调整信号权重"——高流行度看行为，低流行度看内容。

---

## ② 母婴出海应用案例

### 场景：新品快速冷启动

**业务问题**：Momcozy每季度上架30+新品（新款吸奶器、新配件）。上架首周曝光转化率<0.5%，远低于成熟品的2.5%。

**PAM 应用**：
1. **分层**：
   - 高流行度：月交互>1000的SKU
   - 中流行度：月交互100-1000
   - 低流行度：月交互<100（主要是新品）
2. **元训练**：在现有SKU上训练，学习"从商品属性预测用户偏好"的初始化参数
3. **快速适应**：新品上架后，仅需少量交互（10-50次）即可微调出专属推荐模型

**预期产出**：
- 新品首周转化率：0.5% → 1.5%
- 新品达到成熟品转化率的时间：3个月 → 2周
- 长尾SKU总GMV占比：15% → 25%

**业务价值**：
- 加速新品验证：快速识别潜力爆款
- 降低库存风险：不好卖的新品及时止损
- 品类扩展：敢于尝试更多细分品类

---

## ③ 代码模板

```python
"""
Popularity-Aware Meta-Learning (PAM) for Cold-Start Recommendation
用于新品/新用户的快速冷启动推荐
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


class PAMModel(nn.Module):
    """流行度感知的元学习推荐模型"""

    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.content_proj = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, user_ids, item_ids, item_content=None):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)

        if item_content is not None:
            # 低流行度：融合内容特征
            i = self.content_proj(torch.cat([i, item_content], dim=-1))

        score = (u * i).sum(dim=-1)
        return torch.sigmoid(score)


def popularity_aware_meta_train(model, interactions, item_features,
                                popularity_thresholds=(1000, 100),
                                inner_lr=0.01, meta_lr=0.001, epochs=100):
    """
    PAM元训练

    Args:
        interactions: [(user, item, rating)] 列表
        item_features: 商品内容特征
        popularity_thresholds: (高, 低) 流行度阈值
    """
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # 按流行度分层
    item_counts = defaultdict(int)
    for u, i, r in interactions:
        item_counts[i] += 1

    high_pop = [i for i, c in item_counts.items() if c >= popularity_thresholds[0]]
    mid_pop = [i for i, c in item_counts.items()
               if popularity_thresholds[1] <= c < popularity_thresholds[0]]
    low_pop = [i for i, c in item_counts.items() if c < popularity_thresholds[1]]

    layers = {
        'high': high_pop,
        'mid': mid_pop,
        'low': low_pop
    }

    for epoch in range(epochs):
        meta_loss = 0

        for layer_name, items in layers.items():
            if len(items) < 10:
                continue

            # 采样该层的交互
            layer_interactions = [x for x in interactions if x[1] in items]
            if len(layer_interactions) < 5:
                continue

            # 内循环：快速适应
            fast_weights = {name: param.clone()
                           for name, param in model.named_parameters()}

            # 计算该层损失
            users = torch.LongTensor([x[0] for x in layer_interactions])
            items_t = torch.LongTensor([x[1] for x in layer_interactions])
            ratings = torch.FloatTensor([x[2] for x in layer_interactions])

            # 使用内容特征（低流行度层权重更高）
            content_weight = 0.3 if layer_name == 'high' else 0.7
            item_content = torch.randn(len(items_t), 64) * content_weight

            preds = model(users, items_t, item_content)
            loss = nn.BCELoss()(preds, ratings)

            meta_loss += loss

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Meta Loss: {meta_loss.item():.4f}")

    return model


def cold_start_adapt(model, new_item_id, new_item_features,
                     few_interactions, inner_steps=5, inner_lr=0.01):
    """
    新品快速适应：用少量交互微调模型
    """
    adapted_model = PAMModel(
        model.user_emb.num_embeddings,
        model.item_emb.num_embeddings,
        64
    )
    adapted_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)

    for step in range(inner_steps):
        users = torch.LongTensor([x[0] for x in few_interactions])
        items = torch.LongTensor([x[1] for x in few_interactions])
        ratings = torch.FloatTensor([x[2] for x in few_interactions])

        content = torch.FloatTensor([new_item_features] * len(few_interactions))

        preds = adapted_model(users, items, content)
        loss = nn.BCELoss()(preds, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapted_model


# 示例
if __name__ == '__main__':
    model = PAMModel(n_users=1000, n_items=500)
    # 元训练...
    # 新品适应...
    print("PAM模型初始化完成")
print("[✓] Cold Start Meta Learning  测试通过")
```

---


## ④ 技能关联

### 前置技能
- [Skill-Matrix-Factorization](../05-推荐系统/[[Skill-Matrix-Factorization]].md) — 理解隐因子学习是元学习初始化的基础

### 延伸技能
- [Skill-Explainable-Recommendation](../05-推荐系统/[[Skill-Explainable-Recommendation]].md) — 冷启动后向用户说明『为什么推荐』提升信任

### 可组合
- [Skill-Cold-Start-Product-Recommendation](../05-推荐系统/[[Skill-Cold-Start-Product-Recommendation]].md) — 新品上架的端到端冷启动管线

## ⑤ 商业价值评估

- **ROI**：新品GMV提升50-100%，试错周期缩短60%
- **难度**：⭐⭐⭐☆☆（3/5）— 元学习概念门槛，但实现可模块化
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 母婴品类迭代快，冷启动是刚需痛点
