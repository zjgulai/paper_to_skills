# Skill Card: GNN Foundations（图神经网络基础）

> **领域**: 08-知识图谱 | **类型**: 综合萃取

roadmap_phase: phase2
---

## ① 算法原理

GNN 三大基础架构的统一入门：GCN（图卷积，邻居特征加权平均 $\mathbf{h}_v^{(l+1)} = \sigma(\mathbf{W}^{(l)} \sum_{u \in N(v)} \frac{\mathbf{h}_u^{(l)}}{\sqrt{d_v d_u}})$）→ GAT（加注意力权重）→ GraphSAGE（归纳式采样聚合，适合大规模图）。

---

## ② 母婴出海应用案例

产品共购图：节点=SKU，边=同时购买频率。GCN 学习节点嵌入→发现"硅胶法兰"和"乳头霜"在嵌入空间接近（同一用户群购买），但传统协同过滤未能捕获——因为这两个产品没有共同购买者但有相似的购买者画像。

年化：补充 KG 推荐基础能力，隐性 **10-20 万元**。

---

## ③ 代码模板

```python
import numpy as np

def gcn_layer(adj_norm, features, weights):
    """简化 GCN: H' = σ(D⁻½ A D⁻½ H W)"""
    return np.maximum(0, adj_norm @ features @ weights)  # ReLU

# test: 4-node graph
adj = np.array([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]])
deg = np.diag(1/np.sqrt(adj.sum(axis=1)))
adj_norm = deg @ adj @ deg
feat = np.eye(4); W = np.random.randn(4, 2)*0.1
emb = gcn_layer(adj_norm, feat, W)
print(f"Node embeddings shape: {emb.shape}")
assert emb.shape == (4, 2)
print("[✓] GNN Foundations 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-HGT-Heterogeneous-Graph-Transformer]] | [[Skill-HGCN-Hyperbolic-Graph]]
- **组合**：[[Skill-Audience-Knowledge-Graph]]

---

- **可组合**：[[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]] / [[Skill-KG-Auto-Construction-Agent-Driven]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值：10-20 万元 | **难度**：⭐⭐⭐⭐☆ | **优先级**：⭐⭐⭐☆☆
