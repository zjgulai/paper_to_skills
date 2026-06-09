---
title: 反事实推荐 - 双重校准估计器（DCE）
doc_type: knowledge
module: 05-推荐系统
topic: counterfactual-recommendation
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
paper: arXiv:2403.00817
roadmap_phase: phase2
---

# Skill: Counterfactual Recommendation — 双重校准估计器（DCE）破解 MNAR 选择偏差

> 论文:**Doubly Calibrated Estimator for Recommendation on Data Missing Not At Random** (Kweon & Yu, WWW 2024 oral) · arXiv:2403.00817 · [GitHub](https://github.com/WonbinKweon/DCE_WWW2024)

---

## ① 算法原理

### 核心思想

电商推荐系统的核心痛点是 **MNAR（Missing Not At Random）选择偏差**:用户只对系统曝光过的商品产生反馈,而曝光本身受热度/历史 CTR 影响,导致推荐模型陷入"自我强化"循环。DCE 用**双重校准**同时校准倾向分(propensity)与插补误差(imputation),即插即用与所有 DR 变体兼容。

### 数学直觉

**IPS 估计器**(无偏理想损失):
$$\hat{L}_{\text{IPS}} = \frac{1}{|D|} \sum_{(u,i) \in O} \frac{e_{u,i}}{\hat{p}_{u,i}}$$

**DR 估计器**(双重鲁棒,倾向分 OR 插补只需一个准确):
$$\hat{L}_{\text{DR}} = \sum_{(u,i)\in D} \hat{e}_{u,i} + \sum_{(u,i)\in O} \frac{e_{u,i} - \hat{e}_{u,i}}{\hat{p}_{u,i}}$$

**DCE 校准专家**(本文核心 - 用户自适应温度缩放):
$$\bar{p}_{u,i} = \sum_{k=1}^{K} \pi_k(u) \cdot \sigma\bigl(a_k \cdot \text{logit}(\hat{p}_{u,i}) + b_k\bigr)$$

其中 $\pi_k(u)$ 是基于用户 embedding 的 Gumbel-Softmax 路由权重,$(a_k, b_k)$ 是第 $k$ 个专家的温度缩放参数。

### 关键假设

1. **MNAR 偏差**:观测的 $(u, i)$ 对非随机抽样,曝光机制对 $\hat{p}_{u,i}$ 校准要求严格
2. **校准异质性**:不同用户群对原始倾向分的校准函数不同(高频 vs 低频用户的曝光分布差异)
3. **路由可学习**:Gumbel-Softmax 路由可端到端反传,无需离散决策

### 关键效果数字(论文 Table)

| 数据集 | DCE-DR vs DR-JL |
|---|---|
| Coat (random test) | MSE -8.37%, NDCG@10 提升 |
| Yahoo! R3 | AUC ~+1.2% |
| KuaiRec (full matrix) | NDCG@50 显著提升 |

---

## ② 母婴出海应用案例

### 场景一:东南亚母婴电商奶粉品牌信仰偏差去除

- **业务问题**:Shopee 印尼站某德国奶粉因历史曝光高 CTR 数据被高估,推荐模型持续压制澳洲/新西兰品牌的同质量 SKU。新妈妈点击不到优质冷门品牌,小品牌冷启动失败率 80%+。
- **数据要求**:用户行为日志(曝光/点击/购买) + 商品特征(品牌/价格/认证)
- **预期产出**:DCE-DR 模型给出的去偏 CTR 预测,新品牌召回率提升
- **业务价值**:小品牌冷启动 ROI 提升 30-50%,平台 SKU 多样性扩大,小品牌入驻意愿↑;按印尼站月 GMV 5000 万元计,长尾品牌 GMV 增量约 200-400 万元/月

### 场景二:哺乳期妈妈"购买窗口期"反事实评估

- **业务问题**:母婴电商的核心特征是月龄驱动的时间窗口需求(0-6月奶粉1段, 6-12月辅食),用户在"错误时期"未购买 ≠ 无需求,但模型把"未购"当作"不喜欢"。
- **数据要求**:用户行为日志 + 宝宝月龄 + 商品适用月龄属性
- **预期产出**:对每个用户做"如果在正确月龄推送,购买概率是多少"的反事实预测,前置 1-2 月推送
- **业务价值**:前置触达转化率提升 25-40%,以美亚母婴专区 100 万月活计,GMV 增量约 80-150 万元/月

---

## ③ 代码模板

```python
"""
DCE (Doubly Calibrated Estimator) 最小骨架
论文 arXiv:2403.00817, WWW 2024 (oral)
官方完整实现: https://github.com/WonbinKweon/DCE_WWW2024
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibratedPropensityModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32, n_experts: int = 5):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.router = nn.Sequential(nn.Linear(emb_dim, n_experts), nn.Softmax(dim=1))
        self.a = nn.Parameter(torch.ones(n_experts))
        self.b = nn.Parameter(-torch.ones(n_experts))

    def forward(self, users: torch.Tensor, items: torch.Tensor, T: float = 1e-3) -> torch.Tensor:
        u = self.user_emb(users)
        v = self.item_emb(items)
        logit = (u * v).sum(-1)

        pi = self.router(u)
        g = -torch.log(-torch.log(torch.rand_like(pi) + 1e-10) + 1e-10)
        pi = F.softmax((pi.log() + g) / T, dim=1)

        logit_exp = logit.unsqueeze(1).expand(-1, self.a.size(0))
        p_cal = torch.sigmoid(logit_exp * self.a + self.b)
        return (p_cal * pi).sum(1).clamp(1e-4, 1 - 1e-4)


def dce_dr_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    prop: torch.Tensor,
    imp_pred: torch.Tensor,
    gamma: float = 0.05,
) -> torch.Tensor:
    inv_p = 1.0 / prop.detach().clamp(gamma, 1.0)
    ips_term = F.binary_cross_entropy(pred, label, weight=inv_p, reduction="mean")
    imp_term = F.binary_cross_entropy(pred, imp_pred.detach(), reduction="mean")
    return ips_term - imp_term


def main() -> None:
    n_users, n_items = 5000, 2000
    model = CalibratedPropensityModel(n_users, n_items, emb_dim=32, n_experts=5)

    users = torch.randint(0, n_users, (128,))
    items = torch.randint(0, n_items, (128,))
    labels = torch.randint(0, 2, (128,)).float()

    prop = model(users, items)
    print(f"校准倾向分均值: {prop.mean():.4f}, 标准差: {prop.std():.4f}")

    pred = torch.sigmoid(torch.randn(128))
    imp = torch.sigmoid(torch.randn(128))
    loss = dce_dr_loss(pred, labels, prop, imp)
    print(f"DR Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-Matrix-Factorization](./[[Skill-Matrix-Factorization]].md) — 理解 MF 隐因子是 DCE 倾向模型的方法学基础
- [Skill-Intelligent-Prediction-Doubly-Robust](../03-时间序列/[[Skill-Intelligent-Prediction-Doubly-Robust]].md) — DR 估计是 DCE 的方法学骨干

### 延伸技能
- [Skill-NeuralNDCG-Learning-to-Rank](./[[Skill-NeuralNDCG-Learning-to-Rank]].md) — DCE 输出的去偏 CTR 直接接入 L2R 精排
- [Skill-Uplift-Modeling](../01-因果推断/[[Skill-Uplift-Modeling]].md) — 反事实推荐是 Uplift 思想在推荐场景的具体应用

### 可组合
- [Skill-Cold-Start-Meta-Learning-PAM](./[[Skill-Cold-Start-Meta-Learning-PAM]].md) — DCE 解决长尾偏差,元学习解决冷启动,二者互补
- [Skill-Intelligent-Attribution-Causal-Forest](../01-因果推断/[[Skill-Intelligent-Attribution-Causal-Forest]].md) — 因果森林估计 CATE,DCE 估计去偏 CTR,联合驱动反事实推荐

---

## ⑤ 商业价值评估

### ROI 预估

**场景一(品牌偏差去除)**:
- 长尾品牌 GMV 增量:200-400 万元/月(以印尼站 5000 万 GMV 计)
- 模型部署成本:GPU 训练 ~2 万元/月 + 工程 1 人月
- **ROI ≈ 100-200 倍/月**

**场景二(月龄前置触达)**:
- 前置触达 GMV 增量:80-150 万元/月(美亚母婴专区 100 万月活)
- **年化收益:1000-1800 万元**

### 实施难度:⭐⭐⭐☆☆ (3/5)

- 易处:有官方 PyTorch 开源代码可直接复用
- 难处:Coat/Yahoo 是公开 benchmark,真实业务数据需先做曝光-行为日志的 schema 对齐
- GPU 需求中等(单 V100 可训),适合 100 万-1000 万级用户规模

### 优先级评分:⭐⭐⭐⭐⭐ (5/5)

**评估依据**:
1. **业务相关度极高**:MNAR 偏差是所有电商推荐系统的"原罪",DCE 是 2024 年最强的 plug-and-play 解
2. **WWW 2024 oral 顶会论文**,方法学严谨,引用快速增长
3. **官方 GitHub 完整开源**,工程化路径清晰
4. **填补图谱关键缺口**:连接 01-因果推断 ↔ 05-推荐系统 双领域桥梁,提升 5-推荐系统 内的因果质量
