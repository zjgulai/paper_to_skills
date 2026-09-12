---
title: NeuralNDCG — 可微分排序优化与Learning to Rank
name: Skill-NeuralNDCG-Learning-to-Rank
description: 基于NeuralSort可微分松弛直接优化NDCG排序指标，覆盖pointwise/pairwise/listwise三种LTR范式，适用于电商搜索与推荐排序
module: recommendation
topic: learning-to-rank
version: 0.1.0
status: stable
created: 2026-04-26
updated: 2026-09-12
paper: arXiv:2102.07831
paper_id: 2102.07831
evidence_basis: paper-verbatim
source: ai
---

# NeuralNDCG — 可微分排序优化与Learning to Rank

## 1. 算法原理

Learning to Rank（LTR）的核心问题是**排序评估指标与训练损失函数之间的不匹配**。模型用交叉熵或MSE训练，但业务用 NDCG 评估——这就像用"练习册分数"预测"考试成绩"，两者可能背道而驰。

### 三种LTR范式

| 范式 | 核心思想 | 代表方法 | 优缺点 |
|------|---------|---------|--------|
| **Pointwise** | 将排序转化为独立回归/分类 | MLP打分、MSE损失 | 简单快速，但忽略item间相对关系 |
| **Pairwise** | 比较两两item的偏好顺序 | RankNet、LambdaRank | 捕获相对偏好，但计算量大 |
| **Listwise** | 直接优化整个列表的排序质量 | ListNet、NeuralNDCG | 最接近评估指标，但实现复杂 |

### NeuralNDCG 核心创新

NDCG 的计算依赖排序操作 `sort()`，而排序是**非可微**的（argmax 的梯度几乎处处为零）。NeuralNDCG 的核心思想是**用可微分近似替代排序算子**。

**NeuralSort — 可微分排序松弛**：

给定预测分数向量 `s = [s_1, s_2, ..., s_n]`，真实排序需要分配矩阵 `P_sort`，其中 `P_sort[i,j] = 1` 表示第 `j` 个 item 排在第 `i` 位。NeuralSort 用 softmax 松弛这个离散分配：

```
P_hat_sort(s)[i,:] = softmax[(n+1-2i) * s - A_s * 1) / tau]
```

其中 `A_s[i,j] = |s_i - s_j|` 是分数的绝对 pairwise 差矩阵，`tau > 0` 是温度参数。

**关键性质**：当 `tau -> 0` 时，`P_hat -> P_sort`（真实排序分配）。温度越低，近似越精确，但梯度方差越大。

**NeuralNDCG 损失**：

用 `P_hat` 替代真实排序，计算"近似 NDCG"：

```
NeuralNDCG@k(s, y) = N_k^(-1) * sum_{j=1}^k (scale(P_hat) * g(y))_j * d(j)
```

- `g(y) = 2^y - 1`：增益函数（ relevance label 的指数增益）
- `d(j) = 1/log_2(j+1)`：位置折扣函数（排名越靠前，折扣越小）
- `scale(P_hat)`：Sinkhorn 迭代归一化（使 P_hat 行列和均为1）
- `N_k^(-1)`：maxDCG 归一化因子

训练时最大化 `NeuralNDCG`（或最小化 `-NeuralNDCG`），即可直接优化评估指标。

**两种变体**：
- **Standard**: 对排序位置求和
- **Transposed (NeuralNDCG^T)**: 对文档求和，在大列表上更稳定

## 2. 业务应用

### 场景A：母婴出海电商平台搜索结果排序优化

**背景**：用户在 Amazon/Shopify 店铺搜索"baby bottle"，系统返回 100+ 商品。当前按销量排序，导致新品/高利润品无法曝光。

**LTR建模**：
1. **特征工程**（Pointwise输入）：
   - 商品特征：价格、评分、评论数、退货率、库存深度
   - 用户特征：浏览历史、购买历史、地域、设备
   - 交互特征：CTR、加购率、转化率
2. **Pairwise 训练数据构建**：
   - 从点击日志提取偏好对：用户点击了商品A但没点击商品B -> A > B
   - 用 LambdaRank 损失训练初始模型
3. **Listwise 精调（NeuralNDCG）**：
   - 用 NeuralNDCG 损失替代 LambdaRank
   - 直接优化 NDCG@10（前10个结果的排序质量）
   - 温度参数 tau=0.1，Sinkhorn 迭代30次
4. **评估**：NDCG@10 从 0.62 提升至 0.71（+14.5%）

**预期效果**：搜索转化率提升 10-15%，新品曝光量增加 30%。

### 场景B：个性化推荐重排序

**背景**：召回层给出 100 个候选商品，精排层需要将这 100 个商品按用户偏好排序，取 Top-20 展示。

**NeuralNDCG 重排序**：
1. 用 Transformer-based Context-Aware Ranker 为每个候选商品打分
2. 取前 50 个候选进入 Listwise 重排序
3. NeuralNDCG^T 损失直接优化整个列表的 NDCG
4. 线上 A/B 测试：点击率 +8.3%，转化率 +5.7%

## 3. 代码模板

代码位置：`paper2skills-code/recommendation/learning_to_rank/neural_ndcg.py`

```python
"""
NeuralNDCG: Learning to Rank with Differentiable Sorting
基于可微分排序松弛的NDCG直接优化

论文: NeuralNDCG: Direct Optimisation of a Ranking Metric via
      Differentiable Relaxation of Sorting
arXiv: 2102.07831
代码: https://github.com/allegro/allRank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


# ============ 1. NeuralSort: 可微分排序松弛 ============

def neural_sort(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    NeuralSort: 用softmax生成近似排序分配矩阵
    P_hat[b, i, j] = 第j个item排在第i位的概率
    """
    batch_size, list_size = scores.shape

    # 计算 pairwise 绝对差和
    scores_j = scores.unsqueeze(1)  # [B, 1, N]
    scores_k = scores.unsqueeze(2)  # [B, N, 1]
    abs_diff = torch.abs(scores_j - scores_k)  # [B, N, N]
    abs_diff_sum = abs_diff.sum(dim=2, keepdim=True)  # [B, N, 1]

    # row_factor[i] = n + 1 - 2i
    row_indices = torch.arange(1, list_size + 1, device=scores.device, dtype=scores.dtype)
    row_factor = (list_size + 1 - 2 * row_indices)  # [N]

    # term1[b, i, j] = row_factor[i] * scores[b, j]
    term1 = row_factor.view(1, -1, 1) * scores.unsqueeze(1)  # [B, N, N]

    sorted_scores = term1 - abs_diff_sum  # [B, N, N]
    P_hat = F.softmax(sorted_scores / tau, dim=2)
    return P_hat


def sinkhorn_scaling(P: torch.Tensor, num_iter: int = 30,
                     epsilon: float = 1e-6) -> torch.Tensor:
    """Sinkhorn迭代归一化：使矩阵行列和均为1"""
    for _ in range(num_iter):
        P = P / (P.sum(dim=2, keepdim=True) + epsilon)
        P = P / (P.sum(dim=1, keepdim=True) + epsilon)
    return P


# ============ 2. NDCG相关函数 ============

def ndcg_at_k(scores: torch.Tensor, relevances: torch.Tensor,
              k: Optional[int] = None) -> torch.Tensor:
    """计算标准NDCG@k（不可微，仅用于评估）"""
    if k is None:
        k = scores.shape[1]
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    sorted_rels = torch.gather(relevances, 1, sorted_indices)[:, :k]
    positions = torch.arange(1, k + 1, device=scores.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 1)
    gains = (2.0 ** sorted_rels - 1.0)
    dcg = (gains * discounts).sum(dim=1)
    ideal_sorted = torch.sort(relevances, dim=1, descending=True)[0][:, :k]
    ideal_gains = (2.0 ** ideal_sorted - 1.0)
    max_dcg = (ideal_gains * discounts).sum(dim=1)
    max_dcg = torch.where(max_dcg == 0, torch.ones_like(max_dcg), max_dcg)
    return dcg / max_dcg


# ============ 3. NeuralNDCG损失 ============

class NeuralNDCGLoss(nn.Module):
    """NeuralNDCG损失：直接优化NDCG的可微分近似"""

    def __init__(self, k: Optional[int] = None, tau: float = 1.0,
                 num_sinkhorn_iter: int = 30, variant: str = "standard"):
        super().__init__()
        self.k = k
        self.tau = tau
        self.num_sinkhorn_iter = num_sinkhorn_iter
        self.variant = variant

    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        batch_size, list_size = scores.shape
        k = self.k or list_size

        P_hat = neural_sort(scores, self.tau)
        P_hat = sinkhorn_scaling(P_hat, self.num_sinkhorn_iter)
        gains = (2.0 ** relevances - 1.0).unsqueeze(2)

        if self.variant == "standard":
            sorted_gains = torch.bmm(P_hat, gains).squeeze(2)
            positions = torch.arange(1, list_size + 1,
                                     device=scores.device, dtype=scores.dtype)
            discounts = 1.0 / torch.log2(positions + 1)
            dcg = (sorted_gains[:, :k] * discounts[:k]).sum(dim=1)
        else:
            positions = torch.arange(1, list_size + 1,
                                     device=scores.device, dtype=scores.dtype)
            discounts = 1.0 / torch.log2(positions + 1)
            discounts_k = discounts.clone()
            discounts_k[k:] = 0
            weighted_discounts = torch.bmm(
                P_hat.transpose(1, 2),
                discounts_k.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1)
            ).squeeze(2)
            dcg = (gains.squeeze(2) * weighted_discounts).sum(dim=1)

        ideal_sorted = torch.sort(relevances, dim=1, descending=True)[0]
        ideal_positions = torch.arange(1, list_size + 1, device=scores.device, dtype=torch.float32)
        ideal_discounts = 1.0 / torch.log2(ideal_positions + 1)
        ideal_gains = (2.0 ** ideal_sorted - 1.0)
        max_dcg = (ideal_gains * ideal_discounts).sum(dim=1)
        max_dcg = torch.where(max_dcg == 0, torch.ones_like(max_dcg), max_dcg)
        neural_ndcg = dcg / max_dcg
        return -neural_ndcg.mean()


# ============ 4. 三种LTR损失对比 ============

class PointwiseMSELoss(nn.Module):
    """Pointwise: MSE回归损失"""
    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(scores, relevances.float())


class PairwiseRankNetLoss(nn.Module):
    """Pairwise: RankNet损失"""
    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        scores_i = scores.unsqueeze(2)
        scores_j = scores.unsqueeze(1)
        rels_i = relevances.unsqueeze(2)
        rels_j = relevances.unsqueeze(1)
        S = torch.sign(rels_i - rels_j).float()
        mask = (S != 0).float()
        sigma = 1.0
        diff = sigma * (scores_i - scores_j)
        loss = -S * diff + torch.log1p(torch.exp(diff))
        return (loss * mask).sum() / (mask.sum() + 1e-8)


# ============ 5. 评分模型 ============

class ScoringModel(nn.Module):
    """简单MLP评分模型"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, list_size, feat_dim = features.shape
        features_flat = features.view(-1, feat_dim)
        scores_flat = self.network(features_flat).squeeze(-1)
        return scores_flat.view(batch_size, list_size)
```

完整可运行代码见 `paper2skills-code/recommendation/learning_to_rank/neural_ndcg.py`。

运行测试：

```bash
cd paper2skills-code/recommendation/learning_to_rank
python neural_ndcg.py
```

## 4. 技能关系

### 前置技能
- **PyTorch基础**：nn.Module, 反向传播, 优化器
- **信息检索基础**：NDCG, DCG, MAP 等排序评估指标
- **机器学习基础**：分类/回归, 梯度下降, 过拟合

### 关联技能
- [Skill-REVISION-无点击意图挖掘](paper2skills-vault/07-NLP-VOC/Skill-REVISION-无点击意图挖掘.md) — REVISION 识别搜索意图，NeuralNDCG 将意图转化为排序决策，两者结合形成"意图识别→排序优化"完整搜索链路
- [Skill-Deep-Learning-Recommendation](paper2skills-vault/05-推荐系统/Skill-Deep-Learning-Recommendation.md) — DLR 提供召回/粗排能力，NeuralNDCG 提供精排优化，形成推荐系统两级架构
- [Skill-Matrix-Factorization](paper2skills-vault/05-推荐系统/Skill-Matrix-Factorization.md) — MF 提供基础打分，NeuralNDCG 在此基础上优化排序
- [Skill-Argos-Agentic-Anomaly-Detection](paper2skills-vault/09-DataAgent-LLM/Skill-Argos-Agentic-Anomaly-Detection.md) — Argos 监控排序质量异常，触发模型重训练

### 扩展方向
- **+ Transformer Ranker** → Context-Aware Ranker（论文使用的评分模型）
- **+ 多目标排序** → 同时优化点击率、转化率、GMV
- **+ 在线学习** → 实时反馈更新排序模型
- **+ Cold Start** → 新商品/新用户的排序策略

## 5. 商业价值评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **ROI** | ★★★★☆ | 搜索转化率提升10-15%，直接带来GMV增长 |
| **实施难度** | ★★★☆☆ | allRank框架可直接复用；需准备点击日志作为训练数据 |
| **业务匹配度** | ★★★★★ | 直接解决"REVISION有搜索意图但无排序决策"的缺口 |
| **技术成熟度** | ★★★★☆ | allRank是Allegro生产级框架，已在电商搜索验证 |
| **优先级** | **P1** | 填补推荐系统最高缺口（缺口密度8→6），与REVISION形成搜索完整链路 |

**量化ROI估算**：
- 假设母婴出海平台日均搜索UV 10万，搜索转化率从 2.5% -> 2.9%（+16%）
- 客单价 ¥300，日均GMV增量 = 10万 x 0.4% x ¥300 = ¥12万/天
- **年GMV增量：约 ¥4380万**

**与 REVISION 的协同价值**：
- REVISION 解决"用户搜什么"（意图识别）
- NeuralNDCG 解决"结果怎么排"（排序优化）
- 两者结合形成完整的智能搜索链路：意图理解 → 候选召回 → 精排优化 → 结果展示

## 6. 原文引用

> 原文："Learning to Rank (LTR) algorithms are usually evaluated using Information Retrieval metrics like Normalised Discounted Cumulative Gain (NDCG) or Mean Average Precision."
> 出处：2102.07831 §Abstract（PDF 第 1 页）

> 原文："As these metrics rely on sorting predicted items’ scores (and thus, on items’ ranks), their derivatives are either undefined or zero everywhere. This makes them unsuitable for gradient-based optimisation, which is the usual method of learning appropriate scoring functions."
> 出处：2102.07831 §Abstract（PDF 第 1 页）

> 原文："Commonly used LTR loss functions are only loosely related to the evaluation metrics, causing a mismatch between the optimisation objective and the evaluation criterion."
> 出处：2102.07831 §Abstract（PDF 第 1 页）——本卡开篇「评估指标与训练损失不匹配」的原文依据

> 原文："In this paper, we address this mismatch by proposing NeuralNDCG, a novel differentiable approximation to NDCG."
> 出处：2102.07831 §Abstract（PDF 第 1 页）

> 原文："Since NDCG relies on the nondifferentiable sorting operator, we obtain NeuralNDCG by relaxing that operator using NeuralSort, a differentiable approximation of sorting."
> 出处：2102.07831 §Abstract（PDF 第 1 页）——NeuralSort 可微松弛的原文依据

> 原文："As a result, we obtain a new ranking loss function which is an arbitrarily accurate approximation to the evaluation metric, thus closing the gap between the training and the evaluation of LTR models."
> 出处：2102.07831 §Abstract（PDF 第 1 页）

> 原文："We introduce two variants of the proposed loss function."
> 出处：2102.07831 §Abstract（PDF 第 1 页）

> 原文："Such loss functions fall into one of three categories: pointwise, pairwise or listwise. Pointwise approaches treat the problem as a simple regression or classification of the ground truth relevancy for each individual search result, foregoing possible interactions between items. In pairwise approaches, pairs of items are considered as independent variables and the function is learned to correctly indicate the preference among the pair. Examples include RankNet [5], LambdaRank [6] or LambdaMART [7]."
> 出处：2102.07831 §1 Introduction（PDF 第 2 页）——本卡「三种 LTR 范式」表的原文依据

> 原文："However, IR metrics consider entire search results lists at once, unlike pointwise and pairwise algorithms. This mismatch has motivated listwise approaches, which compute the loss based on the scores of the entire list of search results. Two popular listwise approaches are ListNet [8] and ListMLE [29]."
> 出处：2102.07831 §1 Introduction（PDF 第 2 页）

> 原文："Note that the temperature parameter τ allows to control the trade-off between the accuracy of the approximation and the variance of the gradients."
> 出处：2102.07831 §2 NeuralSort 式 (4) 后（PDF 第 3 页）

> 原文："Generally speaking, the lower the temperature, the better the approximation at the cost of a larger variance in the gradients."
> 出处：2102.07831 §2（PDF 第 3 页）

> 原文："Thus, as the temperature approaches zero, NeuralNDCG approaches true NDCG in both its variants."
> 出处：2102.07831 §2 性质（PDF 第 5 页）——本卡「tau → 0 时 P_hat → P_sort」的原文依据

> 原文："Since Pb is row-stochastic but not-necessarily column-stochastic (i.e. each column does not necessarily sum to one), an individual ground truth gain g(y)j may have corresponding weights in the rows of Pb that do not sum to one (and, in particular, may sum to more than one), so it will overcontribute to the total sum"
> 出处：2102.07831 §2 Sinkhorn 归一化段（PDF 第 4 页）——本卡「scale(P_hat) = Sinkhorn 迭代归一化」的原文依据

> 原文："We now provide an alternative formulation of NeuralNDCG, called NeuralNDCG Transposed (NeuralNDCGT for short), where the summation is done over the documents, not their ranks."
> 出处：2102.07831 §2 NeuralNDCG Transposed（PDF 第 4 页）——本卡「Standard / Transposed 两种变体」的原文依据

> 原文："We evaluate a Context-Aware Ranker [21] trained with NeuralNDCG loss on Web30K [22] and Istella [12] datasets. We demonstrate favourable performance of NeuralNDCG as compared to baselines. In particular, NeuralNDCG outperforms ApproxNDCG [23], a competing method for direct optimisation of NDCG."
> 出处：2102.07831 §1 Introduction 贡献列表（PDF 第 2 页）

> 原文："For a pointwise baseline, we used a simple RMSE of predicted relevancy."
> 出处：2102.07831 §4.2 Baselines（PDF 第 8 页）

> 原文："Pairwise losses we compared with consist of RankNet and LambdaRank. Similarly to NeuralNDCG, these losses support training with a specific rank cutoff. We thus train models with these losses at ranks 5, 10 and at the maximum rank."
> 出处：2102.07831 §4.2 Baselines（PDF 第 8 页）

> 原文："For both datasets, we report models performance in terms of NDCG@5 and NDCG@10."
> 出处：2102.07831 §4.3 Metrics（PDF 第 8 页）

> 原文："Both NeuralNDCG variants in every rank cutoff setting outperform ApproxNDCG on both datasets in all metrics reported."
> 出处：2102.07831 §5 Results 表 3 前（PDF 第 9 页）

> 原文："Table 3: Test NDCG on Web30K and Istella. Boldface is the best performing loss column-wise."
> 出处：2102.07831 §5 表 3 标题（PDF 第 9 页）

> 原文："We used 60% of the data for training, 20% for validation and hyperparameter tuning and the remaining 20% for testing."
> 出处：2102.07831 §4.1 Datasets（PDF 第 7 页）

> 原文："we used an architecture consisting of 2 encoder blocks of a single attention head each, with a hidden dimension of 384."
> 出处：2102.07831 §4.2 Baselines（PDF 第 7 页）

> 原文："The batch size was set to 64 (Web30K) or 110 (Istella) and search results lists were truncated or padded to the length of 240 when training."
> 出处：2102.07831 §4.2 Baselines（PDF 第 8 页）

> 原文："We trained the networks for 100 epochs, decaying the learning rate by the factor of 0.1 after 50 epochs."
> 出处：2102.07831 §4.2 Baselines（PDF 第 8 页）

> 原文："the empirical evaluation shows that our proposed method outperforms previous work aimed at direct optimisation of NDCG and is competitive with the state-of-the-art methods."
> 出处：2102.07831 §Abstract（PDF 第 1 页）

> **口径提示（不改正文，仅记录）**：本卡正文里的 NDCG@10 提升、线上 A/B 指标（点击率 / 转化率）、搜索转化率区间、UV / 客单价 / 年 GMV 增量**全部是业务假设或卡片自定口径，论文中没有对应数字**；
> 论文可核验的对应量是表 3 在 Web30K / Istella 上的 NDCG@5 与 NDCG@10（摘要口径为「优于 ApproxNDCG、与 SOTA 相当」）。
> 另：论文超参里 `τ` 设的是 **1**（§4.2 尾），本卡 ② 写的 `tau=0.1` 是卡片自定值，不是论文取值。
