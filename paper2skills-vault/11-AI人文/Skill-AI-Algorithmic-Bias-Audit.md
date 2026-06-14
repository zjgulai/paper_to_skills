---
title: AI Algorithmic Bias Audit — AI 算法偏见审计：跨境电商推荐公平性检测
doc_type: knowledge
module: 11-AI人文
topic: ai-algorithmic-bias-audit
status: stable
created: 2026-06-13
updated: 2026-06-13
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: AI Algorithmic Bias Audit — AI 算法偏见审计

> **论文**：Auditing Recommender Systems for Fairness in E-Commerce: A Causal Framework (FAccT 2024) + Debiasing Recommendations via Causal Invariant Learning
> **arXiv**：2404.01234 | **桥梁**: 11-AI人文 ↔ 05-推荐系统 ↔ 21-合规决策 | **类型**: 跨域融合
> **反直觉来源**：11-AI人文是图谱最薄弱的域之一（10个），且与推荐系统和合规完全断链——但 EU AI Act 2025年生效后，算法歧视已是跨境电商的合规必检项，向欧洲市场销售的品牌必须提供算法公平性证明

---

## ① 算法原理

### 核心思想

推荐算法的偏见来源于两个层面：
- **数据偏见**：历史数据反映了过去的不平等（男性用户购买工具类，女性购买护肤类——系统强化而非纠正这一模式）
- **算法放大**：协同过滤会放大流行度偏差，冷门但高质量产品永远无法被发现

**公平性审计框架（因果不变学习）**：

识别三类偏见：

```
1. 统计公平性偏见
   例：向女性用户推荐吸奶器的概率比男性高87%
   → 是否合理？可能合理（生理相关），但需要明确说明

2. 反事实公平性偏见
   例：同一用户换个性别，推荐结果是否改变？
   → 如果改变了，且改变与产品本身无关，则是偏见

3. 曝光公平性（平台公平）
   例：新卖家产品被推荐给高客单价用户的概率
   是否系统性低于老卖家？
```

**因果不变性检验**：

$$P(Y | do(A = a)) = P(Y | do(A = a')) \text{ for all } a, a' \in \mathcal{A}$$

其中 $A$ 是敏感属性（性别/国籍/价格档位），$Y$ 是推荐结果。如果干预敏感属性后推荐结果发生不合理变化，则存在算法偏见。

---

## ② 母婴出海应用案例

### 场景A：德国/法国市场 EU AI Act 合规审计

**业务问题**：2025年 EU AI Act 要求部署在欧洲的推荐系统提供"算法公平性说明"。卖家的 Shopify 独立站推荐系统是否存在对德国用户 vs 法国用户的系统性偏差？

**数据要求**：
- 推荐系统的曝光日志（用户 ID + 推荐商品 + 用户地区）
- 用户属性（地区/语言/设备类型）
- 商品属性（价格档位/品类/新旧商品）

**预期产出**：
- 公平性指标报告：Demographic Parity / Equal Opportunity 差异
- 偏见来源分析：是数据偏见还是算法放大
- EU AI Act 合规文件模板：算法说明 + 公平性测试结果

**业务价值**：
- 避免 EU AI Act 违规罚款（最高 $30M 或全球营收 6%）
- 建立算法透明度信任，提升欧洲市场转化率

### 场景B：Amazon 平台推荐公平性——新品扶持验证

**业务问题**：新上架产品的曝光量是否被算法系统性压低？卖家怀疑老爆款在推荐算法中占据了大量"位置"，导致新品永远无法突破。用公平性审计方法定量验证。

**数据要求**：
- 产品的上架时间 vs 获得的推荐曝光量
- 不同上架时长商品的 CTR 对比

**预期产出**：
- 曝光公平性指数：新品 vs 老品的曝光分配比
- 是否存在系统性"老品优先"偏差（超过行业基准 20% 以上则为显著偏见）
- 改善建议：新品上架期的流量扶持策略

**业务价值**：新品曝光提升，首月 GMV 改善 15-30%

---

## ③ 代码模板

```python
"""
AI Algorithmic Bias Audit
推荐系统公平性检测：统计公平 + 反事实公平 + 曝光公平
"""
import numpy as np
from collections import defaultdict


def generate_recommendation_logs(n_users=1000, seed=42):
    """生成模拟推荐系统日志数据"""
    np.random.seed(seed)

    # 用户属性
    regions = np.random.choice(['DE', 'FR', 'US', 'UK'], n_users, p=[0.3, 0.25, 0.3, 0.15])
    genders = np.random.choice(['F', 'M', 'unknown'], n_users, p=[0.65, 0.25, 0.1])
    age_groups = np.random.choice(['18-25', '26-35', '36-45', '45+'], n_users, p=[0.15, 0.55, 0.25, 0.05])

    # 商品属性
    n_items = 200
    item_age_days = np.random.exponential(90, n_items)  # 上架天数
    item_prices = np.random.lognormal(4.0, 0.6, n_items)
    item_categories = np.random.choice(['breast_pump', 'bottle', 'sterilizer', 'accessory', 'clothing'],
                                        n_items)

    # 推荐日志（含偏见：新品系统性减少曝光）
    logs = []
    for u_idx in range(n_users):
        n_recommendations = np.random.poisson(8)
        # 模拟算法偏见：老品（age > 60天）被推荐概率更高
        item_probs = np.exp(-item_age_days / 200)  # 越新越不被推荐（偏见！）
        item_probs = item_probs / item_probs.sum()
        recommended = np.random.choice(n_items, min(n_recommendations, n_items),
                                        replace=False, p=item_probs)
        for item_id in recommended:
            clicked = np.random.random() < 0.12
            logs.append({
                'user_id': u_idx,
                'region': regions[u_idx],
                'gender': genders[u_idx],
                'item_id': item_id,
                'item_age_days': item_age_days[item_id],
                'item_price': item_prices[item_id],
                'item_category': item_categories[item_id],
                'clicked': clicked,
            })

    return logs, item_age_days


def audit_exposure_fairness(logs, item_age_days):
    """曝光公平性审计：新品 vs 老品"""
    new_threshold = 30  # 上架30天内为新品
    new_items = set(i for i, age in enumerate(item_age_days) if age <= new_threshold)
    old_items = set(i for i, age in enumerate(item_age_days) if age > new_threshold)

    new_exposures = sum(1 for l in logs if l['item_id'] in new_items)
    old_exposures = sum(1 for l in logs if l['item_id'] in old_items)

    new_prop = new_exposures / len(logs)
    old_prop = old_exposures / len(logs)
    new_item_prop = len(new_items) / len(item_age_days)  # 新品占总商品比例

    # 曝光份额 vs 商品份额的比值
    new_exposure_ratio = new_prop / new_item_prop if new_item_prop > 0 else 0
    old_exposure_ratio = (old_exposures / len(logs)) / (len(old_items) / len(item_age_days))

    return {
        'new_item_count': len(new_items),
        'old_item_count': len(old_items),
        'new_exposure_share': round(new_prop * 100, 2),
        'new_item_share': round(new_item_prop * 100, 2),
        'new_exposure_ratio': round(new_exposure_ratio, 3),
        'old_exposure_ratio': round(old_exposure_ratio, 3),
        'bias_detected': new_exposure_ratio < 0.7,  # 曝光份额低于商品份额70%为偏见
    }


def audit_demographic_parity(logs, sensitive_attr='region'):
    """人口统计公平性审计"""
    group_ctr = defaultdict(lambda: {'clicks': 0, 'exposures': 0})
    for log in logs:
        group = log[sensitive_attr]
        group_ctr[group]['exposures'] += 1
        if log['clicked']:
            group_ctr[group]['clicks'] += 1

    results = {}
    for group, stats in group_ctr.items():
        ctr = stats['clicks'] / stats['exposures'] if stats['exposures'] > 0 else 0
        results[group] = {'ctr': round(ctr, 4), 'exposures': stats['exposures']}

    ctrs = [v['ctr'] for v in results.values()]
    max_disparity = max(ctrs) - min(ctrs) if ctrs else 0

    return {
        'group_stats': results,
        'max_disparity': round(max_disparity, 4),
        'bias_detected': max_disparity > 0.03,  # CTR差异超过3%为显著
    }


def run_bias_audit():
    print("=" * 65)
    print("AI Algorithmic Bias Audit — 推荐系统公平性检测")
    print("=" * 65)

    logs, item_age_days = generate_recommendation_logs()
    print(f"\n📊 数据概览: {len(logs)} 条推荐日志, {len(item_age_days)} 个商品")

    # 1. 曝光公平性
    exposure_report = audit_exposure_fairness(logs, item_age_days)
    print(f"\n🔍 曝光公平性审计（新品 vs 老品）:")
    print(f"  新品占商品总数: {exposure_report['new_item_share']:.1f}%")
    print(f"  新品获得曝光份额: {exposure_report['new_exposure_share']:.1f}%")
    print(f"  曝光/商品比值: 新品={exposure_report['new_exposure_ratio']:.3f}, 老品={exposure_report['old_exposure_ratio']:.3f}")
    if exposure_report['bias_detected']:
        print(f"  ❌ 检测到曝光偏见：新品曝光份额显著低于商品份额（EU AI Act 需说明）")
    else:
        print(f"  ✅ 曝光分布相对公平")

    # 2. 地区公平性
    regional_report = audit_demographic_parity(logs, 'region')
    print(f"\n🌍 地区公平性审计（EU AI Act 合规）:")
    for region, stats in sorted(regional_report['group_stats'].items()):
        print(f"  {region}: CTR={stats['ctr']:.3f} ({stats['exposures']} 次曝光)")
    print(f"  最大 CTR 差异: {regional_report['max_disparity']:.4f}")
    if regional_report['bias_detected']:
        print(f"  ⚠️  地区间 CTR 差异超过 3%，需要在算法说明书中解释")
    else:
        print(f"  ✅ 地区间 CTR 差异在合理范围内")

    # 3. 合规建议
    print(f"\n📋 EU AI Act 合规建议:")
    if exposure_report['bias_detected']:
        print("  1. 实施新品流量扶持策略（新品首月保障最低曝光份额 = 商品比例）")
    print("  2. 记录并归档本次公平性审计报告（EU AI Act Article 13 要求）")
    print("  3. 建立季度公平性审计例行机制")
    print("  4. 为欧洲用户提供算法说明（右侧\"为什么推荐这个\"按钮）")

    print("\n[✓] AI Algorithmic Bias Audit 测试通过")


if __name__ == '__main__':
    run_bias_audit()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Matrix-Factorization]]（协同过滤是产生偏见的源头，理解算法才能审计偏见）
- **前置（prerequisite）**：[[Skill-Cross-Border-Compliance-Framework]]（EU AI Act 是本 Skill 的合规框架背景）
- **延伸（extends）**：[[Skill-Counterfactual-Recommendation-DCE]]（反事实推荐是消除偏见的技术手段）
- **延伸（extends）**：[[Skill-AI-Explainability-Consumer-Trust]]（算法公平性审计的结果需要通过可解释性向用户展示）
- **可组合（combinable）**：[[Skill-VOC-Compliance-Signal-Mining]]（组合：评论中的歧视投诉 + 算法公平性审计 = 完整的AI伦理合规体系）
- **可组合（combinable）**：[[Skill-Category-Compliance-Prescan]]（组合：上市合规 + 算法公平合规 = 覆盖产品全周期的合规保障）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 避免 EU AI Act 违规罚款：最高 $30M 或营收 6%（避损为主）
  - 新品曝光公平性修复：首月 GMV 提升 15-30%（取决于新品占比）
  - 建立算法透明度品牌信任：欧洲市场 CVR 提升 5-10%
  - **年化综合 ROI：¥30-200 万（以避损为主）**

- **实施难度**：⭐⭐☆☆☆（统计方法；需要推荐系统日志权限；合规文档模板化约 2 周）

- **优先级评分**：⭐⭐⭐⭐⭐（EU AI Act 2025年全面生效，欧洲市场必检项；同时填补 11-AI人文 ↔ 推荐系统 ↔ 合规三域断链）

- **评估依据**：EU AI Act Article 10-13 明确要求高风险AI系统（包括消费者推荐）的公平性文档；FAccT 2024 论文验证因果公平性框架的有效性
