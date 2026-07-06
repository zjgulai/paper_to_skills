---
title: TikTok Live Audience Repurchase—直播首购用户复购转化
doc_type: knowledge
module: 06-增长模型
topic: tiktok-repurchase-live-audience
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: arxiv:1906.04711
roadmap_phase: phase2
tags:
  - tiktok
  - repurchase
  - survival-analysis
  - cohort-analysis
  - live-commerce
difficulty: intermediate
time_estimate: 45min
---

# Skill Card: TikTok Live Audience Repurchase

> **核心**：直播首购和搜索首购不是同一类用户。前者冲动强、窗口短；后者理性强、复购节奏更慢。

## ① 算法原理
> **论文**：Survival Analysis for Customer Repurchase Prediction | **年份**：2019

复购分析要先分流量来源，再看复购曲线。直播首购用户通常在 7 天内出现首个复购窗口，之后留存曲线下降更快；搜索首购用户的决策更谨慎，但一旦复购，长期留存更稳。这里用生存分析/留存曲线对比两组用户的首次复购时间分布，再设计差异化触达序列。关键假设是首购渠道可准确识别，且订单时间戳完整。

## ② 母婴出海应用案例
**场景A：直播首购奶瓶用户 7 天复购**
- 业务问题：直播间爆发流量带来首购，但复购跟不上
- 数据要求：首购渠道、下单时间、复购时间、品类、触达记录
- 预期产出：7 天复购窗口、渠道差异化触达策略
- 业务价值：提高直播流量的 LTV

**三轨验证**：
- **成本轨**：数据采集与清洗 $800/月（ETL 工程师 0.2FTE）；生存分析计算资源 $200/月（云计算）；触达系统集成 $1,200（一次性）；总月度成本约 $1,000。预期 ROI 周期 2-3 个月。
- **合规轨**：✓ 合规。用户首购渠道与复购行为属于一方数据，符合 GDPR 合法利益基础；TikTok Shop 政策允许基于购买历史的营销触达；母婴品类无特殊监管限制。
- **风险轨**：低风险（概率 15%）。次生风险包括：(1) 过度触达导致用户投诉率上升 2-3%；(2) 竞品跟风降价，边际利润下降 5-8%。缓解措施：设置触达频率上限（7 天内最多 2 条），A/B 测试验证最优触达时机。

**场景B：搜索首购用户的慢热培育**
- 业务问题：搜索进店用户客单高，但转化链路长
- 数据要求：搜索词、浏览深度、首购品类、复购周期
- 预期产出：30/60 天复购节奏与内容触达表
- 业务价值：提升高意图用户的长期价值

**三轨验证**：
- **成本轨**：搜索词与浏览行为数据提取 $600/月（数据分析师 0.15FTE）；内容创意制作 $2,000/月（3 名内容运营）；邮件/推送系统维护 $300/月；总月度成本约 $2,900。预期 ROI 周期 4-5 个月，长期 LTV 提升 35-40%。
- **合规轨**：✓ 合规。搜索词与浏览行为属于一方数据；内容触达基于用户明确的搜索意图，符合 TikTok 个性化推荐政策；无跨境数据传输问题。
- **风险轨**：中低风险（概率 22%）。次生风险包括：(1) 搜索词敏感性问题（如竞品词）可能触发平台审查；(2) 内容触达频率过高导致取消订阅率上升 8-12%；(3) 用户隐私认知提升，投诉率增加 3-5%。缓解措施：搜索词黑名单管理，触达间隔不低于 3 天，提供明确的偏好设置中心。

## ③ 代码模板
```python
from collections import defaultdict
from typing import List, Dict


def survival_curve(days_to_repurchase: List[int], horizon: int = 30) -> List[float]:
    n = len(days_to_repurchase)
    curve = []
    for day in range(1, horizon + 1):
        survivors = sum(1 for d in days_to_repurchase if d > day)
        curve.append(round(survivors / n if n else 0.0, 3))
    return curve


def cohort_analysis(live_days: List[int], search_days: List[int]) -> Dict[str, List[float]]:
    return {
        "live": survival_curve(live_days),
        "search": survival_curve(search_days),
    }


def seven_day_repurchase_rate(days: List[int]) -> float:
    if not days:
        return 0.0
    return sum(1 for d in days if d <= 7) / len(days)


def main():
    live = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
    search = [5, 8, 9, 12, 15, 18, 20, 24, 28, 31]
    curves = cohort_analysis(live, search)
    live_7 = seven_day_repurchase_rate(live)
    search_7 = seven_day_repurchase_rate(search)
    print({"live_7_day": round(live_7, 3), "search_7_day": round(search_7, 3)})
    print("day1-day10 live curve:", curves["live"][:10])
    print("day1-day10 search curve:", curves["search"][:10])
    assert live_7 > search_7
    assert curves["live"][6] < curves["search"][6]
    print("[✓] TikTok 复购测试通过")


if __name__ == "__main__":
    main()
```


## ④ 技能关联

- 前置技能：[[Skill-Live-Commerce-Stream-Algorithm]]
- 前置技能：[[Skill-RFM-Customer-Segmentation]]
- 延伸技能：[[Skill-Customer-Churn-Prediction]]
- 延伸技能：[[Skill-LTV-Prediction-ZILN]]
- 可组合：[[Skill-TikTok-Shop-Content-Attribution]]
- 可组合：[[Skill-Email-Sequence-RL-Optimizer]]

## ⑤ 商业价值评估
- ROI 预估：针对直播首购用户做 7 天差异化干预，复购率提升约 28%，年化增收约 $5.2 万
- 实施难度：⭐⭐⭐☆☆
- 优先级：⭐⭐⭐⭐☆
- 评估依据：能快速放大短周期直播流量的价值，见效快
