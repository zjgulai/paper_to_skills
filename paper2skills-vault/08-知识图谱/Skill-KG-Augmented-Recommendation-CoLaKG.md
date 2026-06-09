---
title: 知识图谱增强推荐 - CoLaKG (LLM × KG)
doc_type: knowledge
module: 08-知识图谱
topic: kg-augmented-recommendation
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
paper: arXiv:2410.12229 SIGIR 2025
roadmap_phase: phase2
---

# Skill: CoLaKG — LLM 增强的知识图谱推荐

> 论文:**Comprehending Knowledge Graphs with Large Language Models for Recommender Systems** (Cui et al., SIGIR 2025) · arXiv:2410.12229 · [GitHub](https://github.com/ziqiangcui/CoLaKG-SIGIR25)

---

## ① 算法原理

### 核心思想

传统 KG 推荐(KGAT/KGIN)把 KG 结构硬编码为 embedding,缺失语义理解。CoLaKG 用 **LLM 读懂 KG**:对每个 item 提取局部子图 → LLM 生成语义文本理解 → 文本 embedding → 通过余弦相似度构建全局 item-item 语义图 → 与 CF 协同 embedding 门控融合。**推断期不调用 LLM**(离线预计算),工程友好。

### 数学直觉

**Step 1 - 局部 KG 理解**:
$$\mathbf{s}_v = \mathcal{P}\bigl(\text{LLM}(\mathcal{I}_v, \mathcal{D}_v, \mathcal{D}'_v)\bigr)$$
其中 $\mathcal{I}_v$ 是 item 信息,$\mathcal{D}_v, \mathcal{D}'_v$ 是 1-2 跳子图三元组,$\mathcal{P}$ 是文本 embedding 模型。

**Step 2 - 全局语义图**:
$$r_{(v_i, v_j)} = \cos(\mathbf{s}_{v_i}, \mathbf{s}_{v_j})$$
突破多跳 KG 的层数爆炸,直接捕获语义相近 item。

**Step 3 - 门控融合**(CF + 语义):
$$\text{item\_repr} = g \odot \mathbf{e}_v + (1-g) \odot \mathbf{s}_v^{\text{agg}}, \quad g = \sigma(W[\mathbf{e}_v; \mathbf{s}_v^{\text{agg}}])$$

### 关键效果数字

| 数据集 | Recall@20 | NDCG@20 | vs. KGAT | vs. KGIN |
|---|---|---|---|---|
| MovieLens | **0.2642** | **0.3974** | +7.8% | +3.1% |
| Last-FM | **0.3803** | **0.3471** | +13.7% | +4.7% |
| MIND | **0.1087** | **0.0684** | +15.7% | +7.4% |

---

## ② 母婴出海应用案例

### 场景一:基于品牌×成分×认证三元组的奶粉推荐

- **业务问题**:海外华人妈妈购买奶粉需综合考量品牌(HiPP/Aptamil)、成分(DHA/HMO 益生元)、段位(1段/2段)、认证(EU 有机/Non-GMO),传统 CF 无法解读这些维度。新品奶粉(无购买历史)冷启动困难
- **数据要求**:商品属性 KG(品牌-成分-认证三元组) + 用户购买历史
- **CoLaKG 配置**:
  - 节点:奶粉 SKU + 属性节点(品牌/成分/认证)
  - 局部子图:每款奶粉的 1-2 跳邻居(同品牌/同成分/同认证)
  - LLM 语义化:"这是一款适合 0-6 月新生儿、含有机 HMO 益生元、持 EU 有机认证的德国奶粉"
- **业务价值**:新品奶粉冷启动 Recall@10 提升 8-12%,小品牌 GMV 增量 100-200 万元/月(以印尼站 5000 万 GMV 计)

### 场景二:基于"适用月龄图谱"的成长路径推荐

- **业务问题**:母婴用品有显著时序性消费(纸尿裤 NB→S→M→L,辅食工具 4M→1Y→3Y),传统推荐无法跨品类关联成长轨迹
- **数据要求**:商品 KG(适用月龄边) + 用户购买序列
- **CoLaKG 配置**:
  - LLM 理解每个 item:"NB 纸尿裤适合体重<5kg 新生儿"
  - 全局语义图捕获不同品类同月龄的 item(NB 纸尿裤 ↔ 3M 安抚奶嘴 ↔ 防胀气奶瓶)
  - 跨品类 proactive 触达
- **业务价值**:跨品类购买转化率提升 15-20%,冷启动新用户(1-2 次购买)效果显著;月均增量 GMV 80-150 万元

---

## ③ 代码模板

```python
"""
CoLaKG 最小骨架
论文 arXiv:2410.12229 (SIGIR 2025)
官方代码: https://github.com/ziqiangcui/CoLaKG-SIGIR25
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGSemanticAggregator(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Linear(embed_dim * 2, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = x[src], x[dst]
        alpha_raw = self.attn(torch.cat([h_dst, h_src], dim=-1)).squeeze(-1) * edge_weight

        alpha_exp = torch.exp(alpha_raw - alpha_raw.max())
        denom = torch.zeros(x.size(0), device=x.device).index_add(0, dst, alpha_exp) + 1e-10
        alpha = alpha_exp / denom[dst]

        messages = alpha.unsqueeze(-1) * h_src
        agg = torch.zeros_like(x).index_add(0, dst, messages)
        return agg


class CoLaKG(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, llm_embed_dim: int = 768):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.semantic_proj = nn.Linear(llm_embed_dim, embed_dim)
        self.kg_agg = KGSemanticAggregator(embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        item_item_edge_index: torch.Tensor,
        item_item_weight: torch.Tensor,
        llm_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        s = self.semantic_proj(llm_embeddings)
        s_aug = self.kg_agg(s, item_item_edge_index, item_item_weight)

        e_v = self.item_emb.weight
        gate = self.gate(torch.cat([e_v, s_aug], dim=-1))
        item_repr = gate * e_v + (1 - gate) * s_aug

        u = self.user_emb(users)
        v = item_repr[items]
        return (u * v).sum(dim=-1)


def main() -> None:
    N_U, N_I, D, D_LLM = 500, 1000, 64, 768
    model = CoLaKG(N_U, N_I, D, D_LLM)
    edge_idx = torch.randint(0, N_I, (2, 5000))
    edge_w = torch.rand(5000)
    llm_embs = torch.randn(N_I, D_LLM)
    users = torch.randint(0, N_U, (32,))
    items = torch.randint(0, N_I, (32,))
    scores = model(users, items, edge_idx, edge_w, llm_embs)
    print(f"Output shape: {scores.shape}")
    print(f"Sample scores: {scores[:5].detach().tolist()}")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-Knowledge-Graph-for-Skills-Management](./[[Skill-Knowledge-Graph-for-Skills-Management]].md) — KG 基础结构理解
- [Skill-Matrix-Factorization](../05-推荐系统/[[Skill-Matrix-Factorization]].md) — CF 协同嵌入是 CoLaKG 的基底

### 延伸技能
- [Skill-Hierarchical-Product-KG-Construction](./[[Skill-Hierarchical-Product-KG-Construction]].md) — 自动构建的产品 KG 直接喂给 CoLaKG
- [Skill-Explainable-Recommendation](../05-推荐系统/[[Skill-Explainable-Recommendation]].md) — LLM 生成的语义理解天然可解释

### 可组合
- [Skill-Dense-Retrieval-Ecommerce-Semantic-Search](./[[Skill-Dense-Retrieval-Ecommerce-Semantic-Search]].md) — KG 语义 embedding 共享,搜推一体化
- [Skill-GraphRAG-Knowledge-Enhanced-Retrieval](./[[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]].md) — 推荐 + RAG 检索增强

---

## ⑤ 商业价值评估

### ROI 预估

**场景一(奶粉品牌×成分推荐)**:小品牌 GMV 增量 100-200 万元/月 = **1200-2400 万元/年**;LLM 推理离线一次性成本 5-10 万元;**ROI ≈ 100-200 倍**

**场景二(成长路径推荐)**:跨品类 GMV 增量 80-150 万元/月 = **960-1800 万元/年**

### 实施难度:⭐⭐⭐☆☆ (3/5)

- 易处:**官方 PyTorch 开源代码完整**,有预计算 embedding 可直接复用
- 难处:LLM 语义化需要 GPT-4o/Qwen2.5 调用预算(一次性,~5-10 万)
- 难处:商品 KG 必须先构建,可配合 Hierarchical-Product-KG Skill

### 优先级评分:⭐⭐⭐⭐⭐ (5/5)

**评估依据**:
1. **SIGIR 2025 顶会**,2025-04 发表方法新颖
2. **官方完整开源代码**,工程化路径清晰
3. **+7.8% / +13.7% / +15.7% Recall** 提升幅度大
4. **关键桥梁**:08-知识图谱 ↔ 05-推荐系统 直接连通,是图谱缺口高优先级填补项
