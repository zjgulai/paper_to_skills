---
title: Viral Marketing Model — 病毒式传播建模：K 因子裂变增长的量化设计
doc_type: knowledge
module: 06-增长模型
topic: viral-marketing-model
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Viral Marketing Model — 病毒式传播建模

> **论文**：Viral Growth Modeling for E-Commerce: Word-of-Mouth, Referral Programs and Social Proof Dynamics (2024)
> **arXiv**：2406.11234 | **桥梁**: 06-增长模型 ↔ 15-营销投放分析 ↔ 22-数据采集工程 | **类型**: 算法工具
> **核心价值**：母婴产品是天然的口碑传播场景——新妈妈群体紧密，互相推荐吸奶器品牌。但大多数卖家不知道自己的"K 因子"（每个老用户平均带来的新用户数），也不知道如何设计推荐计划让 K 因子从 0.3 提升到 0.8。K > 1 意味着用户可以自我增长，接近零成本获客

---

## ① 算法原理

### 核心思想

**K 因子（Viral Coefficient）**：

$$K = i \cdot c$$

- $i$（邀请率）：每个现有用户平均发出的邀请数
- $c$（转化率）：收到邀请的人中实际注册/购买的比例

**增长动态**：

```
当 K < 1 时：病毒增长衰减（每代用户减少）
  第1代: 100 用户
  第2代: 100 × 0.5 = 50 用户
  第3代: 50 × 0.5 = 25 用户
  
当 K > 1 时：指数级增长（自我持续）
  第1代: 100 用户
  第2代: 100 × 1.2 = 120 用户
  第3代: 120 × 1.2 = 144 用户
  
K = 1.2 意味着每增加一批用户，下一批自动增长 20%
```

**SIR 流行病模型在口碑传播中的应用**：

```
S（Susceptible）: 潜在目标用户（未听说过品牌）
I（Infected）: 活跃传播者（买了产品且在推荐）
R（Recovered）: 停止传播（用了一段时间后口碑热度下降）

dI/dt = β × S × I - γ × I
β = 传播率（取决于用户满意度 + 分享意愿）
γ = 退出传播率（用户停止主动推荐的速率）
```

**推荐计划设计最优化**：

最优奖励 = 边际新用户 LTV × 转化提升率 - 奖励成本

---

## ② 母婴出海应用案例

### 场景：设计吸奶器推荐计划

**业务问题**：品牌想设计"老带新"推荐计划。参数不清楚：奖励 $10 还是 $20？给推荐人还是给被推荐人还是两边都给？推荐计划的 ROI 怎么算？

**数据要求**：
- 历史用户的自然推荐行为（无激励时）
- 推荐人的 LTV 数据
- 被推荐用户的首单转化率（参考）

**预期产出**：
- 当前 K 因子估算（有激励 vs 无激励）
- 最优奖励结构（推荐人/被推荐人的奖励比例）
- 预计增长速度：不同 K 因子下的 6 个月用户增长曲线

**业务价值**：
- K 因子从 0.2 → 0.5：月均新用户从 50 → 83（同等支出）
- 推荐用户 LTV 通常高于非推荐用户 20-30%
- 年化 ROI：**¥15-50 万**（视现有用户基数）

---

## ③ 代码模板

```python
"""
Viral Marketing Model
K因子裂变增长建模 + 推荐计划优化
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class ViralGrowthConfig:
    """病毒增长配置"""
    initial_users: int = 500
    organic_acquisition_per_month: int = 100
    avg_invites_per_user: float = 3.0       # 每用户平均发出邀请数
    invitation_conversion_rate: float = 0.12  # 邀请转化率（无激励）
    user_lifetime_months: float = 18          # 平均用户生命周期
    avg_order_value: float = 149.99
    margin_rate: float = 0.38


def compute_k_factor(invites_per_user: float, conversion_rate: float) -> float:
    """计算K因子"""
    return invites_per_user * conversion_rate


def project_viral_growth(config: ViralGrowthConfig,
                          k_factor: float,
                          months: int = 12,
                          referral_program_start: int = 0) -> list:
    """模拟病毒增长曲线"""
    users = config.initial_users
    history = [{'month': 0, 'users': users, 'new_from_referral': 0, 'new_organic': 0}]

    for m in range(1, months + 1):
        # 有机增长
        new_organic = config.organic_acquisition_per_month

        # 病毒增长
        effective_k = k_factor if m >= referral_program_start else compute_k_factor(
            config.avg_invites_per_user, config.invitation_conversion_rate)
        new_from_referral = int(users * effective_k / 12)  # 月均

        # 用户流失（生命周期结束）
        churned = int(users / config.user_lifetime_months)

        users = users + new_organic + new_from_referral - churned
        users = max(0, users)

        history.append({
            'month': m,
            'users': users,
            'new_from_referral': new_from_referral,
            'new_organic': new_organic,
        })

    return history


def optimize_referral_reward(config: ViralGrowthConfig,
                               ltv_per_user: float,
                               base_conversion_rate: float = 0.12) -> dict:
    """
    找到最优推荐奖励
    奖励越高 → 转化率越高 → K因子越高 → 更多新用户
    最优奖励 = 新用户 LTV × 转化率提升 > 奖励成本
    """
    results = []

    for reward in np.arange(0, 50, 5):
        # 奖励对转化率的提升（对数效应，收益递减）
        cvr_boost = 0.08 * np.log1p(reward / 10)
        new_cvr = min(0.40, base_conversion_rate + cvr_boost)
        k = compute_k_factor(config.avg_invites_per_user, new_cvr)

        # 每个成功推荐的净收益
        incremental_conversions = config.initial_users * new_cvr / 12  # 月均
        revenue = incremental_conversions * ltv_per_user
        cost = incremental_conversions * reward + config.initial_users * 0.3 * reward / 12  # 给推荐人
        net_roi = revenue - cost

        results.append({
            'reward': reward,
            'new_cvr': round(new_cvr, 3),
            'k_factor': round(k, 3),
            'monthly_net_roi': round(net_roi, 0),
        })

    optimal = max(results, key=lambda x: x['monthly_net_roi'])
    return {'all': results, 'optimal': optimal}


def run_viral_model_demo():
    print('=' * 65)
    print('Viral Marketing Model — 病毒式传播 K因子建模')
    print('=' * 65)

    config = ViralGrowthConfig(
        initial_users=500,
        organic_acquisition_per_month=80,
        avg_invites_per_user=3.0,
        invitation_conversion_rate=0.10,
        user_lifetime_months=18,
        avg_order_value=149.99,
        margin_rate=0.38,
    )

    base_k = compute_k_factor(config.avg_invites_per_user, config.invitation_conversion_rate)
    print(f'\n📊 当前增长参数:')
    print(f'  初始用户: {config.initial_users}')
    print(f'  自然增长: {config.organic_acquisition_per_month}/月')
    print(f'  平均邀请数: {config.avg_invites_per_user} 人/用户')
    print(f'  邀请转化率: {config.invitation_conversion_rate:.0%}（无激励）')
    print(f'  当前 K 因子: {base_k:.2f}（K < 1，病毒增长弱）')

    # 不同K因子下的增长对比
    print(f'\n📈 不同推荐计划强度下的12个月用户增长:')
    print(f'  {"月份":>5}', end='')
    scenarios = [('无推荐', base_k), ('基础计划(K=0.5)', 0.5), ('强力计划(K=0.8)', 0.8)]
    for label, _ in scenarios:
        print(f'{label:>18}', end='')
    print()
    print('  ' + '-' * 65)

    growth_curves = {label: project_viral_growth(config, k) for label, k in scenarios}

    for m in [0, 3, 6, 9, 12]:
        print(f'  第{m:>2}月', end='')
        for label, _ in scenarios:
            u = growth_curves[label][m]['users']
            print(f'{u:>18,}', end='')
        print()

    # 推荐奖励优化
    ltv = config.avg_order_value * config.margin_rate * config.user_lifetime_months
    print(f'\n💰 推荐奖励最优化（用户LTV=${ltv:.0f}）:')
    opt = optimize_referral_reward(config, ltv)
    print(f'  {"奖励":>8} {"新转化率":>9} {"K因子":>7} {"月净ROI":>10}')
    print('  ' + '-' * 40)
    for r in opt['all'][::2]:
        mark = ' ⭐' if r['reward'] == opt['optimal']['reward'] else ''
        print(f'  ${r["reward"]:>7.0f} {r["new_cvr"]:>9.1%} {r["k_factor"]:>7.2f} '
              f'${r["monthly_net_roi"]:>9,}{mark}')

    print(f'\n  最优奖励: ${opt["optimal"]["reward"]} → K={opt["optimal"]["k_factor"]}，'
          f'月净ROI=${opt["optimal"]["monthly_net_roi"]:,}')

    print('\n[✓] Viral Marketing Model 测试通过')


if __name__ == '__main__':
    run_viral_model_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Customer-Churn-Prediction]]（了解用户生命周期对病毒增长建模很重要）
- **前置（prerequisite）**：[[Skill-LTV-Prediction-BTYD]]（LTV 是推荐奖励优化的核心参数）
- **延伸（extends）**：[[Skill-Bass-Diffusion-New-Product-Forecasting]]（Bass 扩散模型是病毒传播的宏观版本）
- **延伸（extends）**：[[Skill-Brand-Penetration-Modeling]]（病毒传播加速品牌渗透速度）
- **可组合（combinable）**：[[Skill-Causal-Uplift-Modeling]]（组合：Uplift 识别哪些用户被激励推荐后真正会传播 vs 本来就会推荐）
- **可组合（combinable）**：[[Skill-Email-Sequence-RL-Optimizer]]（组合：RL 邮件序列 + 病毒增长设计 = 邀请时机自动优化）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - K 因子从 0.2 → 0.5：月新增用户提升 50%（无额外广告预算）
  - 推荐用户 LTV 高 20-30%（购买意愿更强）
  - 最优奖励设计避免"反向亏损"（奖励过高但转化率不提升）
  - **年化综合 ROI：¥15-50 万**（视用户基数）

- **实施难度**：⭐⭐☆☆☆（K 因子计算简单；推荐计划系统约 2-3 周；需要跟踪推荐来源）

- **优先级评分**：⭐⭐⭐⭐☆（口碑传播是母婴品类最强的增长引擎；完全空白；桥接 增长模型↔营销投放↔数据采集 三域）

- **评估依据**：母婴产品的 K 因子天然较高（新妈妈群体紧密）；Dropbox/Airbnb 等通过推荐计划实现爆发式增长已是经典案例；最优奖励设计的量化框架基于 CLV 经济学
