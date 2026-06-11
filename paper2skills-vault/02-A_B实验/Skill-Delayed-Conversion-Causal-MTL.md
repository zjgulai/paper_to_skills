---
title: 促销延迟转化因果多任务建模 — 大促期用户行为的反事实分析
doc_type: knowledge
module: 02-A_B实验
topic: delayed-conversion-causal-multitask
status: stable
created: 2026-06-11
updated: 2026-06-11
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 促销延迟转化因果多任务建模

> **论文**：Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion
> **arXiv**：2604.21675 | 2026年4月 | **桥梁**: 02-A/B实验 ↔ 23-运营财务 | **类型**: 跨域融合
> **背景**：大促前用户"加购不下单"行为大量积累，如何区分"真正被促销触动"vs"本来就会买"是计算促销 ROI 的核心难题

---

## ① 算法原理

### 核心思想

**大促前的延迟转化**有两种本质不同的路径：

```
路径A（真实增量）：促销信息 → 用户决策改变 → 下单（Treated + would not have bought）
路径B（自然转化）：用户本来就计划购买 → 等到促销后下单（Control contamination）
```

传统 A/B 实验**无法直接分离**这两条路径，因为大促期间所有用户都暴露在同样的促销环境中（无干净对照组）。

**Counterfactual Multi-task Learning** 的解法：
1. 在促销前（Pre-Promotion）观测用户行为序列（浏览、加购、搜索、时长）
2. 构建 **Counterfactual 反事实模型**：估算"如果没有促销，该用户是否仍会在此时购买"
3. 用 **Multi-task Learning** 同时优化：(a) 促销后实际转化预测 (b) 反事实无促销转化预测
4. 两者之差 = **真实促销增量转化率（AICR，Actual Incremental Conversion Rate）**

### 数学框架

设 $T_i \in \{0, 1\}$ 为是否被促销处理，$Y_i(t)$ 为潜在结果（购买=1）：

$$\text{AICR}_i = E[Y_i(1) | X_i] - E[Y_i(0) | X_i]$$

多任务网络同时学习：
- **任务1**：$\hat{p}_1 = P(Y=1 | X, T=1)$（促销下的转化率）
- **任务2**：$\hat{p}_0 = P(Y=1 | X, T=0)$（无促销下的自然转化率，用历史非促销期数据监督）
- **AICR**：$\hat{p}_1 - \hat{p}_0$

共享底层特征提取器（用户行为 embedding），两个任务头独立输出，用 **IPW（Inverse Propensity Weighting）** 修正选择偏差。

### 关键假设
- 历史非促销期数据可作为 $T=0$ 的代理标签
- 用户行为（Pre-Promotion 阶段的浏览/加购）是可观测协变量
- 忽略网络效应（用户间相互影响较弱的场景）

---

## ② 母婴出海应用案例

### 场景A：Prime Day 促销真实 ROI 测量

**业务问题**：每次 Prime Day 后，广告团队说"Prime Day 带来了 3,000 单增量"，但 CFO 质疑"其中多少是本来就会买的？"。传统方法：总 Prime Day 销量 - 同期去年销量 = 增量（非常粗糙，忽略自然增长趋势）。

**CMTL 处理**：
1. 收集 Prime Day 前 14 天用户行为（浏览次数/收藏/加购/停留时长）
2. 训练 Counterfactual 模型：用历史非大促期数据标定 $P(Y=1|X,T=0)$
3. 对比 Prime Day 期间实际转化率与模型预测的"自然转化率"
4. AICR = 促销增量购买率

**数据要求**：
- 用户前14天行为序列（每日浏览次数、加购次数、搜索词、页面停留时长）
- 历史非促销期（普通周）的购买记录（用于训练反事实模型）
- Prime Day 期间的购买记录（实际 $Y$）

**预期产出**：
- **真实 AICR**：区分 3,000 单中，哪些是真正被促销驱动（假设结果：1,200单是增量，1,800单是自然转化）
- 促销 ROI 计算更准确：投入广告费 $10,000，真实带来 1,200 增量单 × $20 利润 = $24,000 → 真实 ROI = 2.4x（而非表面的 $10,000 / 3,000单 × $20 = 6x 的虚高估计）

**ROI**：优化广告投入决策，避免在低增量促销上过度投入，年化节省 15-25% 广告费 = 母婴中型卖家节省 $20,000-50,000/年

### 场景B：双11 优惠券发放策略优化

**业务问题**：给所有加购用户发券 → 成本高（许多人本来就会买）。给购买意向低的用户发券 → 转化率低。

**CMTL 处理**：
- 用 AICR 对用户排序：AICR 高 = 该用户被促销驱动的边际效应大（"说服型"用户）
- 只给 AICR > 阈值的用户发券（"可说服者" Persuadable）
- 策略：AICR Top 30% 用户发高价值券（$15 off），Bottom 70% 发低价值券或不发

**ROI**：优惠券预算减少 30%，同等 GMV 增量 → 优惠券 ROI 提升 40-60%

---

## ③ 代码模板

```python
"""
CMTL - 促销延迟转化因果多任务建模
估算每个用户的真实促销增量转化率 (AICR)

依赖: numpy, scikit-learn
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────
# 特征工程：大促前行为序列
# ─────────────────────────────────────────────

def extract_pre_promo_features(behavior_logs: List[dict]) -> np.ndarray:
    """
    从大促前14天行为日志提取特征
    
    behavior_logs 格式：
    [{'user_id': str, 'days_before_promo': int(1-14),
      'page_views': int, 'add_to_cart': int, 'searches': int,
      'session_duration_min': float, 'wishlist_add': int}]
    
    返回：每用户的特征矩阵 shape=(n_users, n_features)
    """
    from collections import defaultdict
    user_data = defaultdict(lambda: {
        'total_pv': 0, 'total_atc': 0, 'total_search': 0,
        'total_duration': 0, 'total_wishlist': 0,
        'active_days': 0, 'recent_3d_pv': 0
    })
    
    for log in behavior_logs:
        uid = log['user_id']
        d = log['days_before_promo']
        user_data[uid]['total_pv'] += log.get('page_views', 0)
        user_data[uid]['total_atc'] += log.get('add_to_cart', 0)
        user_data[uid]['total_search'] += log.get('searches', 0)
        user_data[uid]['total_duration'] += log.get('session_duration_min', 0)
        user_data[uid]['total_wishlist'] += log.get('wishlist_add', 0)
        user_data[uid]['active_days'] += 1
        if d <= 3:
            user_data[uid]['recent_3d_pv'] += log.get('page_views', 0)
    
    uids = list(user_data.keys())
    X = np.array([
        [
            user_data[u]['total_pv'],
            user_data[u]['total_atc'],
            user_data[u]['total_search'],
            user_data[u]['total_duration'],
            user_data[u]['total_wishlist'],
            user_data[u]['active_days'],
            user_data[u]['recent_3d_pv'] / max(user_data[u]['total_pv'], 1),
        ]
        for u in uids
    ])
    return uids, X


# ─────────────────────────────────────────────
# CMTL: Counterfactual Multi-task Learner
# ─────────────────────────────────────────────

class CounterfactualMTL:
    """
    双塔反事实多任务学习模型
    
    任务1 (treated_model): 促销期转化预测 P(Y=1|X, T=1)
    任务2 (control_model): 自然转化预测 P(Y=1|X, T=0)
    AICR = 任务1输出 - 任务2输出
    """
    
    def __init__(self):
        self.treated_model = LogisticRegression(C=1.0, max_iter=500)
        self.control_model = LogisticRegression(C=1.0, max_iter=500)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self,
            X_treated: np.ndarray, y_treated: np.ndarray,
            X_control: np.ndarray, y_control: np.ndarray,
            sample_weight_treated: np.ndarray = None,
            sample_weight_control: np.ndarray = None):
        """
        X_treated: 促销期用户特征（大促期间观测）
        y_treated: 促销期购买标签（1=下单）
        X_control: 非促销期用户特征（历史普通周）
        y_control: 非促销期购买标签
        sample_weight_*: IPW 权重（修正选择偏差，可选）
        """
        X_all = np.vstack([X_treated, X_control])
        self.scaler.fit(X_all)
        
        Xt_scaled = self.scaler.transform(X_treated)
        Xc_scaled = self.scaler.transform(X_control)
        
        self.treated_model.fit(Xt_scaled, y_treated, sample_weight=sample_weight_treated)
        self.control_model.fit(Xc_scaled, y_control, sample_weight=sample_weight_control)
        self.fitted = True
    
    def predict_aicr(self, X: np.ndarray) -> np.ndarray:
        """
        计算每个用户的 AICR（真实促销增量转化率）
        
        Returns: shape=(n_users,) 的 AICR 数组
          正值 = 该用户被促销驱动（可说服者）
          负值 = 该用户促销后转化率反而低（促销敏感的价格歧视问题）
        """
        if not self.fitted:
            raise RuntimeError("模型未训练")
        X_scaled = self.scaler.transform(X)
        p_treated = self.treated_model.predict_proba(X_scaled)[:, 1]
        p_control = self.control_model.predict_proba(X_scaled)[:, 1]
        return p_treated - p_control
    
    def segment_users(self, X: np.ndarray, uids: list) -> dict:
        """用 AICR 对用户分层"""
        aicr = self.predict_aicr(X)
        p_control = self.control_model.predict_proba(self.scaler.transform(X))[:, 1]
        
        segments = {'persuadables': [], 'sure_things': [], 'lost_causes': [], 'sleeping_dogs': []}
        
        for i, uid in enumerate(uids):
            a = aicr[i]
            p0 = p_control[i]
            if a > 0.1 and p0 < 0.5:
                segments['persuadables'].append((uid, a))  # 高 AICR，低自然转化 → 最值得发券
            elif a <= 0.1 and p0 >= 0.5:
                segments['sure_things'].append((uid, a))   # 本来就会买 → 不需要发券
            elif a <= 0 and p0 < 0.3:
                segments['sleeping_dogs'].append((uid, a)) # 促销反而降低转化 → 谨慎
            else:
                segments['lost_causes'].append((uid, a))   # 低转化且低 AICR
        
        return {k: sorted(v, key=lambda x: -x[1]) for k, v in segments.items()}


def run_cmtl_demo():
    """演示 Prime Day 促销增量分析"""
    print("="*60)
    print("CMTL — Prime Day 促销增量转化率分析演示")
    print("="*60)
    
    np.random.seed(42)
    n_treated, n_control = 1000, 2000
    
    # 模拟促销期用户（大促期更高基础购买意愿）
    X_treated = np.random.randn(n_treated, 7)
    X_treated[:, 1] += 1.5  # 加购次数更高（大促期用户更活跃）
    y_treated = (X_treated[:, 1] + X_treated[:, 0] * 0.5 + np.random.randn(n_treated) > 1.0).astype(int)
    
    # 模拟非促销期用户（控制组，自然行为）
    X_control = np.random.randn(n_control, 7)
    y_control = (X_control[:, 1] + X_control[:, 0] * 0.5 + np.random.randn(n_control) > 2.0).astype(int)
    
    # 训练模型
    model = CounterfactualMTL()
    model.fit(X_treated, y_treated, X_control, y_control)
    
    # 对新一批用户估算 AICR（例如下次大促前的用户池）
    n_new = 500
    X_new = np.random.randn(n_new, 7)
    uids = [f'user_{i:04d}' for i in range(n_new)]
    
    aicr = model.predict_aicr(X_new)
    segments = model.segment_users(X_new, uids)
    
    print(f"\n用户分层结果（共 {n_new} 人）:")
    print(f"  ✅ Persuadables（高 AICR，最值得发券）: {len(segments['persuadables'])} 人")
    print(f"  💰 Sure Things（本来就会买，节省券）:   {len(segments['sure_things'])} 人")
    print(f"  😴 Lost Causes（低转化，放弃）:         {len(segments['lost_causes'])} 人")
    print(f"  ⚠️  Sleeping Dogs（促销反效果）:         {len(segments['sleeping_dogs'])} 人")
    
    print(f"\nAICR 分布统计:")
    print(f"  均值: {aicr.mean():.3f}")
    print(f"  中位数: {np.median(aicr):.3f}")
    print(f"  >0.1（强增量）: {(aicr > 0.1).sum()} 人 ({(aicr > 0.1).mean():.0%})")
    print(f"  <0（促销伤害）: {(aicr < 0).sum()} 人 ({(aicr < 0).mean():.0%})")
    
    # 优惠券策略对比
    budget_all = n_new * 15  # 全员发 $15 券
    persuadables_only = len(segments['persuadables'])
    budget_targeted = persuadables_only * 15  # 只给 Persuadables 发
    budget_saving = budget_all - budget_targeted
    
    print(f"\n优惠券策略对比 ($15/张):")
    print(f"  全员发券预算: ${budget_all:,}")
    print(f"  精准发券预算: ${budget_targeted:,}")
    print(f"  节省预算:     ${budget_saving:,} ({budget_saving/budget_all:.0%})")
    
    print("\n[✓] CMTL 演示完成")
    return model, aicr, segments


if __name__ == "__main__":
    run_cmtl_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Uplift-Churn-Prediction]]（Uplift 建模基础：AICR 是 Uplift 的电商促销专化版本）
- **前置（prerequisite）**：[[Skill-DiD-Difference-in-Differences]]（双重差分：理解反事实估计框架）
- **延伸（extends）**：[[Skill-Sequential-AB-Testing]]（序列实验：大促ROI测量后，如何持续做在线实验优化促销策略）
- **可组合（combinable）**：[[Skill-FBA-Fee-Intelligence]]（跨域桥梁 02↔23：精准 AICR 帮助财务团队准确计算每次大促的真实利润贡献，而非虚高的销售增长数字）
- **可组合（combinable）**：[[Skill-PL-Attribution-Analysis]]（跨域桥梁 02↔23：AICR 提供促销增量的因果估计，P&L 归因分析分配利润来源，两者结合完成"促销到利润"的完整因果链）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 优惠券精准投放节省：30% 预算 = 月均 $5,000 优惠券支出节省 $1,500/月 = **$18,000/年**
  - 更准确的促销 ROI 测量 → 避免在低效促销上追加投入：年化节省 **$30,000-80,000**（中型卖家）
  - 大卖家（月销 $1M+）：优化促销预算的 15% = **$180,000/年**

- **实施难度**：⭐⭐⭐☆☆
  - 需要2-3个月历史非促销期数据作为 control 训练集
  - 特征工程（前14天行为序列）需要数据工程配合
  - 模型本身逻辑较清晰，LogisticRegression 即可有效实现

- **优先级评分**：⭐⭐⭐⭐⭐
  - **跨域桥梁**：连接 A/B实验方法论（02域）和运营财务决策（23域）
  - 每个做大促的卖家都面临"这次促销到底有多少是真实增量"的问题
  - 直接影响 CFO 的促销预算决策

- **评估依据**：arXiv 2604.21675，在真实电商大促数据上（前促销期行为序列长度14天），AICR 估计误差相比 DID 降低 23%，相比 PSM（倾向得分匹配）降低 18%
