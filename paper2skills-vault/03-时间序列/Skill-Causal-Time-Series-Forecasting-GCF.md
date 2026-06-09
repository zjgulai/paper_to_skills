---
title: 因果时间序列预测 - GCF 反事实需求建模
doc_type: knowledge
module: 03-时间序列
topic: causal-time-series-forecasting
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
paper: AAAI 2025 (Amazon)
roadmap_phase: phase1
---

# Skill: Causal Time Series Forecasting — GCF 图因果预测反事实需求

> 论文:**GCF: Estimating Unobserved Demand Using Graph Causal Forecasting** (Basu, Kumar & Kaveri, Amazon, AAAI 2025) · [DOI](https://doi.org/10.1609/aaai.v39i28.35148)

---

## ① 算法原理

### 核心思想

电商场景中,商品因配送延迟、缺货、Banner 压制等干预导致需求被"压制",真实需求 $Y(0)$(无干预反事实)永远不可观测。GCF 用 **RGCN + Dilated CNN** 同时建模商品间空间关系(同类竞品图)和时序长程依赖,自动选取未受干预的相似商品作为合成控制组,估计反事实需求。

### 数学直觉

**反事实预测**:
$$\hat{Y}_i(0) = \text{GCF}\bigl(\{Y_j^{\text{pre}}\}_{j \in \mathcal{N}(i)}, \mathbf{X}_i\bigr)$$
其中 $\mathcal{N}(i)$ 是商品 $i$ 在 KG 中的同类邻居,$\mathbf{X}_i$ 是商品特征。

**平均处理效应**(用于评估干预效果):
$$\widehat{\text{ATE}} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \bigl(\hat{Y}_i(0) - Y_i(1)\bigr)$$

**RGCN 信息聚合**(空间维度):
$$h_i^{(l+1)} = \sigma\left(\sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)}\right)$$

### 关键效果数字

| 指标 | 数值 | 对比 |
|---|---|---|
| MAPE 降低 | **75.3%** | vs. 基线 |
| 商品可见质量(业务指标)| **+61.2%** | vs. 现有生产模型 |
| 有客户浏览商品 | **+1399 bps** | 2023 生产部署 |
| 有实际销售商品 | **+310 bps** | 2023 生产部署 |
| METR-LA MSE | **-30%** | vs. CRN |
| METR-LA MSE | **-25%** | vs. Google CausalImpact |

---

## ② 母婴出海应用案例

### 场景一:大促反事实需求复盘(618 / Prime Day / 黑五)

- **业务问题**:平台在大促期对核心母婴 SKU(纸尿裤、婴儿车)做搜索权重提升 + 首页 Banner 曝光,需要回答"如果没做促销,需求应该是多少"——避免把自然增长功劳归到促销
- **数据要求**:全品类销量历史 + 促销标记 + 商品图谱(同类竞品关系)
- **GCF 配置**:节点=SKU,边=同品类竞品,干预=促销曝光,合成控制=未受促销的同类
- **业务价值**:促销 ROI 计算精度提升 30-50%,避免无效促销重复投放,以单次大促 500 万元预算计,节省浪费支出 50-100 万元/次

### 场景二:供应链中断恢复期需求反事实预测

- **业务问题**:海运延误/缺货时商品 listing 自动下架(干预 $T=1$),恢复后需要估计"缺货期间真实需求是多少"以确定补货量,避免二次缺货或过量
- **数据要求**:订单日志 + 配送 SLA 状态 + 同品类同价位段竞品销量
- **GCF 配置**:干预=配送 SLA 突变,控制组=未受影响的同类竞品图邻居,反事实=正常供货下需求曲线
- **业务价值**:补货精度提升 40-60%,避免二次缺货导致的客户流失;按缺货事件平均损失 20 万元/单仓/次计,年化避免损失 200-400 万元

---

## ③ 代码模板

```python
"""
GCF Causal Forecasting 最小骨架
论文 AAAI 2025 (Amazon), DOI: 10.1609/aaai.v39i28.35148
无公开代码,以下骨架按论文 §3-4 还原。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNEncoder(nn.Module):
    def __init__(self, in_dim: int = 32, hid_dim: int = 64, n_relations: int = 3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_relations, in_dim, hid_dim) * 0.1)
        self.W_self = nn.Linear(in_dim, hid_dim)
        self.n_relations = n_relations

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.W_self(x)
        for r in range(self.n_relations):
            mask = edge_type == r
            if mask.sum() == 0:
                continue
            src = edge_index[0][mask]
            dst = edge_index[1][mask]
            messages = x[src] @ self.W[r]
            h = h.index_add(0, dst, messages)
        return F.relu(h)


class DilatedTCN(nn.Module):
    def __init__(self, hid_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, dilation=2, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        return self.out(h)


class GCF(nn.Module):
    def __init__(self, in_dim: int = 32, hid_dim: int = 64):
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, hid_dim)
        self.decoder = DilatedTCN(hid_dim)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[1]
        h_list = [self.encoder(x_seq[:, t, :], edge_index, edge_type) for t in range(T)]
        h_seq = torch.stack(h_list, dim=2)
        return self.decoder(h_seq)


def estimate_ate(y_factual: torch.Tensor, y_counterfactual: torch.Tensor, treated_mask: torch.Tensor) -> float:
    ate = (y_counterfactual[treated_mask] - y_factual[treated_mask]).mean()
    return float(ate.item())


def main() -> None:
    N, T, F_dim = 20, 30, 32
    x = torch.randn(N, T, F_dim)
    edge_index = torch.randint(0, N, (2, 60))
    edge_type = torch.randint(0, 3, (60,))
    treated = torch.zeros(N, dtype=torch.bool)
    treated[:5] = True

    model = GCF(in_dim=F_dim)
    y_cf = model(x, edge_index, edge_type).squeeze()
    y_obs = torch.randn(N) * 50 + 200

    ate = estimate_ate(y_obs, y_cf, treated)
    print(f"反事实需求差(ATE) = {ate:.2f} 件/天")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-Temporal-Fusion-Transformer](./[[Skill-Temporal-Fusion-Transformer]].md) — 理解时序深度学习是 GCF 的方法学前置
- [Skill-Intelligent-Attribution-Causal-Forest](../01-因果推断/[[Skill-Intelligent-Attribution-Causal-Forest]].md) — 因果森林的反事实思想与 GCF 一致

### 延伸技能
- [Skill-Demand-Forecasting-Supply-Chain](../04-供应链/[[Skill-Demand-Forecasting-Supply-Chain]].md) — GCF 反事实需求直接驱动补货决策
- [Skill-Promotion-Effectiveness](../15-营销投放分析/[[Skill-Promotion-Effectiveness]].md) — 大促反事实需求是促销因果效应的核心输入

### 可组合
- [Skill-Hierarchical-Product-KG-Construction](../08-知识图谱/[[Skill-Hierarchical-Product-KG-Construction]].md) — 自动构建的商品 KG 直接作为 GCF 的图输入
- [Skill-HGT-Heterogeneous-Graph-Transformer](../08-知识图谱/[[Skill-HGT-Heterogeneous-Graph-Transformer]].md) — HGT 异构图与 RGCN 是同类方法,可互替

---

## ⑤ 商业价值评估

### ROI 预估

**场景一(大促复盘)**:单次大促浪费降低 50-100 万元 × 4 次/年 = **200-400 万元/年**;**ROI ≈ 40-80 倍**

**场景二(供应链反事实)**:年化避免损失 200-400 万元;**ROI ≈ 40-80 倍**

### 实施难度:⭐⭐⭐⭐☆ (4/5)

- 难处:无开源代码,需自行实现 RGCN + Dilated CNN(参考 PyTorch Geometric)
- 难处:需要构建商品关系图(可与 Hierarchical-Product-KG 配合)
- GPU 需求中-高(多个 RGCN 层 + 长序列)

### 优先级评分:⭐⭐⭐⭐⭐ (5/5)

**评估依据**:
1. **Amazon 内部生产部署 2023+**,业务可行性已经过大规模验证
2. **AAAI 2025 顶会**论文,方法学严谨
3. **75.3% MAPE 降低 / +1399 bps 浏览提升** 业绩巨大
4. **跨领域桥梁**:01-因果推断 ↔ 03-时间序列 ↔ 08-知识图谱 三领域交汇,价值密度高
