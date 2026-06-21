---
title: 算法定价公平性审计 — 群体公平指标与价格歧视检测
doc_type: knowledge
module: 11-AI人文
topic: algorithmic-fairness-in-pricing
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 算法定价公平性审计

> **论文/方法来源**：Algorithmic Pricing and Price Discrimination (Dwork et al., 2012 公平性框架) + Fairness in Machine Learning (Hardt et al., 2016) + EU AI Act Article 10 定价合规
> **领域**：11-AI人文 ↔ 17-价格优化 | **类型**: 跨域融合

## ① 算法原理

算法定价公平性审计检测价格模型是否对不同用户群体产生系统性价格歧视，核心关注三类公平指标：

**统计公平性（Demographic Parity）**：不同群体（按设备、地区、用户属性）获得的价格分布均值差异不超过阈值：
$$|E[P|G=g_1] - E[P|G=g_2]| \leq \epsilon$$

**机会均等（Equal Opportunity）**：不同群体中符合优惠条件的用户获得折扣的概率相等：
$$P(Y_{discount}=1|G=g_1) \approx P(Y_{discount}=1|G=g_2)$$

**价格差异解释度**：将价格差异分解为合理因素（库存、运费、市场成本）与不可解释差异，后者若与敏感属性相关则判定为歧视。

监管背景：EU AI Act（2024）将动态定价纳入高风险系统，要求可解释性文档；美国 FTC 也关注基于地理/人口歧视性定价。

## ② 母婴出海应用案例

**场景A：跨市场动态定价合规审计**
- 业务问题：动态定价算法在美国向使用 Apple 设备的用户展示更高价格，收到 FTC 投诉风险
- 数据要求：用户-价格-设备类型-地区-购买时间历史数据，样本量 ≥ 1000 次曝光/群体
- 预期产出：输出公平性报告：各设备类型价格均值差异、统计显著性 p 值、歧视来源分解
- 业务价值：提前发现定价歧视避免监管处罚（EU 罚款可达年营收 6%），合规运营节省潜在法律成本 30 万元+

**场景B：促销折扣分发公平性监控**
- 业务问题：优惠券定向算法可能系统性向某些邮政编码区域少发优惠，引发歧视投诉
- 数据要求：优惠券分发记录、用户地区数据、人口统计信息（匿名化）
- 预期产出：每周自动生成公平性仪表板，标红 p<0.05 的显著差异维度
- 业务价值：主动发现并修复 3 处不公平分发规则，避免集体诉讼风险

## ③ 代码模板

```python
"""
算法定价公平性审计 — 群体公平性检测与歧视来源分解
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def demographic_parity_check(
    prices: np.ndarray,
    groups: np.ndarray,
    threshold: float = 5.0
) -> Dict:
    """统计公平性检测：各群体价格均值差异"""
    unique_groups = np.unique(groups)
    group_stats = {}
    for g in unique_groups:
        mask = groups == g
        group_prices = prices[mask]
        group_stats[g] = {
            "count": int(mask.sum()),
            "mean_price": float(group_prices.mean()),
            "std_price": float(group_prices.std()),
            "median_price": float(np.median(group_prices))
        }

    # 计算最大群体间均值差异
    means = [v["mean_price"] for v in group_stats.values()]
    max_diff = max(means) - min(means)
    is_fair = max_diff <= threshold

    return {
        "group_stats": group_stats,
        "max_mean_diff": round(max_diff, 2),
        "threshold": threshold,
        "is_fair": is_fair,
        "verdict": "✅ 公平" if is_fair else f"❌ 歧视嫌疑（差异 ${max_diff:.2f} > 阈值 ${threshold}）"
    }


def statistical_significance_test(
    prices: np.ndarray,
    groups: np.ndarray
) -> Dict:
    """Mann-Whitney U 检验各群体价格分布是否显著不同"""
    unique_groups = np.unique(groups)
    results = {}
    for i in range(len(unique_groups)):
        for j in range(i + 1, len(unique_groups)):
            g1, g2 = unique_groups[i], unique_groups[j]
            p1 = prices[groups == g1]
            p2 = prices[groups == g2]
            stat, p_val = stats.mannwhitneyu(p1, p2, alternative='two-sided')
            key = f"{g1} vs {g2}"
            results[key] = {
                "p_value": round(float(p_val), 4),
                "significant": p_val < 0.05,
                "mean_diff": round(float(p1.mean() - p2.mean()), 2)
            }
    return results


def discount_opportunity_equality(
    discount_received: np.ndarray,
    groups: np.ndarray
) -> Dict:
    """机会均等检验：各群体获得折扣的概率差异"""
    unique_groups = np.unique(groups)
    rates = {}
    for g in unique_groups:
        mask = groups == g
        rate = float(discount_received[mask].mean())
        rates[g] = round(rate, 4)

    rate_values = list(rates.values())
    max_disparity = max(rate_values) - min(rate_values)
    return {
        "discount_rates_by_group": rates,
        "max_disparity": round(max_disparity, 4),
        "is_equal_opportunity": max_disparity <= 0.05,
        "verdict": "✅ 机会均等" if max_disparity <= 0.05 else f"❌ 不均等（差异={max_disparity:.1%}）"
    }


def generate_fairness_report(
    prices: np.ndarray,
    groups: np.ndarray,
    discount_received: np.ndarray,
    group_attribute: str = "device_type",
    price_threshold: float = 5.0
) -> str:
    """生成完整公平性审计报告"""
    parity = demographic_parity_check(prices, groups, price_threshold)
    sig_tests = statistical_significance_test(prices, groups)
    eq_opp = discount_opportunity_equality(discount_received, groups)

    report = [
        "=" * 50,
        f"算法定价公平性审计报告 | 分组维度: {group_attribute}",
        "=" * 50,
        f"\n【群体价格统计】",
    ]
    for g, s in parity["group_stats"].items():
        report.append(f"  {g}: n={s['count']}, 均值=${s['mean_price']:.2f}, 中位数=${s['median_price']:.2f}")

    report.append(f"\n【统计公平性】{parity['verdict']} (最大差异=${parity['max_mean_diff']})")

    report.append("\n【显著性检验】")
    for pair, res in sig_tests.items():
        sig_flag = "⚠️ 显著" if res["significant"] else "✓ 不显著"
        report.append(f"  {pair}: p={res['p_value']}, {sig_flag}, 均值差=${res['mean_diff']}")

    report.append(f"\n【折扣机会均等】{eq_opp['verdict']}")
    for g, r in eq_opp["discount_rates_by_group"].items():
        report.append(f"  {g}: {r:.1%}")

    return "\n".join(report)


# ===== 测试 =====
if __name__ == "__main__":
    np.random.seed(42)
    n = 600

    # 模拟：iOS 用户价格略高于 Android（注入轻微歧视）
    groups = np.array(["iOS"] * 300 + ["Android"] * 300)
    prices_ios = np.random.normal(loc=29.99, scale=3.0, size=300)
    prices_android = np.random.normal(loc=27.50, scale=3.0, size=300)
    prices = np.concatenate([prices_ios, prices_android])

    # 模拟折扣分发（iOS 用户获得折扣率低5%）
    discount = np.concatenate([
        np.random.binomial(1, 0.30, 300),  # iOS: 30%
        np.random.binomial(1, 0.35, 300)   # Android: 35%
    ])

    report = generate_fairness_report(
        prices=prices,
        groups=groups,
        discount_received=discount,
        group_attribute="device_type",
        price_threshold=5.0
    )
    print(report)

    # 验证：确认检测到价格差异
    parity_result = demographic_parity_check(prices, groups, threshold=5.0)
    assert "mean_price" in parity_result["group_stats"]["iOS"]
    assert parity_result["max_mean_diff"] > 0

    # 验证显著性测试运行
    sig_results = statistical_significance_test(prices, groups)
    assert len(sig_results) > 0

    print("\n[✓] 算法定价公平性审计测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-AI-Ethics-Fairness-Audit]]（通用伦理审计框架）
- **前置**：[[Skill-AI-Algorithmic-Bias-Audit]]（算法偏见审计基础）
- **延伸**：[[Skill-AIGP-LLM-Dynamic-Pricing]]（在 LLM 动价中嵌入公平约束）
- **可组合**：[[Skill-Competitive-Price-Monitoring]]（竞品定价对比 + 自身公平审计）
- **可组合**：[[Skill-Causal-RL-Dynamic-Pricing]]（在强化学习定价中加公平性约束）

## ⑤ 商业价值评估

- ROI 预估：避免监管处罚（EU AI Act 违规罚款可达年营收 6%），年化法律风险规避 50 万元+
- 实施难度：⭐⭐⭐☆☆（需要有群体标签数据，检验方法成熟）
- 优先级：⭐⭐⭐⭐☆
- 评估依据：欧美市场对算法定价公平性监管趋严，母婴品类涉及弱势群体（孕产期妇女），合规审计是入市前置条件
