---
title: Long-Tail-Opportunity-Auto-Capture — 新兴长尾词周搜索量激增自动创建定向广告组
doc_type: knowledge
module: 25-搜索流量工程
topic: long-tail-opportunity-auto-capture
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Long-Tail-Opportunity-Auto-Capture

> **配对分析层**：[[Skill-Long-Tail-Keyword-Discovery]]
> **决策类型**: 自动触发型 | **触发条件**: 新兴长尾词搜索量周增 > 50% | **执行动作**: 自动创建定向广告组，抢占早期流量窗口

## ① 算法原理

核心是「搜索量趋势监控 + 新兴词识别 + 机会评分 + 广告组自动创建」：

1. **新兴词识别**：通过关键词搜索量 API（Amazon Keyword Trends / Helium 10 / Brand Analytics）获取每周搜索量，计算周环比增速。
2. **触发条件**：
   - 周搜索量增速 ≥ 50%（连续 2 周）
   - 当前竞争强度低（竞品广告数量 < 5 个）
   - 与当前商品相关性 ≥ 70%（基于语义相似度）
3. **机会评分**：
   - 高机会（≥ 80 分）：搜索量 > 1,000/月 + 增速 > 80% + 竞品 < 3 个
   - 中机会（60-79 分）：满足触发条件但不满足高机会
4. **广告组创建**：
   - 匹配方式：Broad + Phrase（新词阶段先收集数据）
   - 初始出价：按 CPC 历史均值的 0.8x（低价探索阶段）
   - 每日预算：$20（控制初始风险）
5. **优化节点**：7 天后评估转化率，若 CVR ≥ 基准则提升出价，否则暂停。

## ② 母婴出海应用案例

**场景：「postpartum recovery」相关搜索词爆发（产后恢复趋势）**
- 触发条件：关键词「postpartum belly wrap」周搜索量从 2,800 → 5,100（+82%），竞品广告 2 个，与产后护理品类相关性 85%
- 执行动作：
  - 自动创建广告组「LT-Capture-postpartum-belly-wrap」
  - Broad + Phrase 双匹配，初始出价 $0.85（行业均值 $1.05 × 0.8）
  - 每日预算 $20，持续 14 天
- 结果：14 天收集数据，CVR 3.2%（高于基准 2.5%），提升出价至 $1.30，转入正式广告组
- 业务价值：提前 3 周抢占新兴词流量，首月带来增量 GMV $12,000，CPC 仅 $0.9（vs 成熟词 $2.5）

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta

def long_tail_opportunity_auto_capture(
    keywords: List[Dict],
    now: Optional[datetime] = None,
    growth_threshold: float = 0.50,      # 周增速触发阈值
    min_weekly_searches: int = 500,       # 最低搜索量
    max_competitors: int = 5,            # 最大竞品广告数
    min_relevance_score: float = 0.70,   # 最低相关性分数
    initial_bid_discount: float = 0.80,  # 初始出价 = 行业均值 × 折扣
    daily_budget: float = 20.0,         # 每日预算
    exploration_days: int = 14          # 探索期天数
) -> Dict:
    """
    新兴长尾词自动捕获触发器
    
    参数:
        keywords: [{
            "keyword": str,
            "weekly_searches_current": int,  # 本周搜索量
            "weekly_searches_prior": int,    # 上周搜索量
            "competitor_ad_count": int,
            "relevance_score": float,        # 与当前品类相关性（0-1）
            "industry_avg_cpc": float,
            "already_targeting": bool        # 是否已有广告组
        }]
    
    返回:
        {"campaigns": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    campaigns = []
    
    for kw in keywords:
        keyword = kw["keyword"]
        searches_current = kw.get("weekly_searches_current", 0)
        searches_prior = max(kw.get("weekly_searches_prior", 1), 1)
        competitor_ads = kw.get("competitor_ad_count", 99)
        relevance = kw.get("relevance_score", 0.0)
        avg_cpc = kw.get("industry_avg_cpc", 1.5)
        already_targeting = kw.get("already_targeting", False)
        
        # 已有广告组跳过
        if already_targeting:
            campaigns.append({"keyword": keyword, "action": "SKIP", "reason": "已有广告组在投放"})
            continue
        
        # 计算增速
        growth_rate = (searches_current - searches_prior) / searches_prior
        
        # 检查触发条件
        condition_growth = growth_rate >= growth_threshold
        condition_volume = searches_current >= min_weekly_searches
        condition_competition = competitor_ads <= max_competitors
        condition_relevance = relevance >= min_relevance_score
        
        if not (condition_growth and condition_volume and condition_relevance):
            campaigns.append({
                "keyword": keyword,
                "action": "NOT_TRIGGERED",
                "growth_rate": round(growth_rate, 2),
                "weekly_searches": searches_current,
                "relevance": relevance,
                "reason": f"未满足触发条件：增速{growth_rate:.0%}（需{growth_threshold:.0%}），搜量{searches_current}，相关性{relevance:.0%}"
            })
            continue
        
        # 机会评分（0-100）
        growth_score = min(growth_rate / 2.0, 1.0) * 40  # 100%增速得40分
        volume_score = min(searches_current / 5000, 1.0) * 30  # 5000搜量得30分
        competition_score = max(0, (max_competitors - competitor_ads) / max_competitors) * 20  # 竞品越少得分越高
        relevance_score = relevance * 10
        opportunity_score = growth_score + volume_score + competition_score + relevance_score
        
        tier = "HIGH" if opportunity_score >= 80 else "MEDIUM"
        
        # 广告组配置
        initial_bid = round(avg_cpc * initial_bid_discount, 2)
        end_date = (now + timedelta(days=exploration_days)).strftime("%Y-%m-%d")
        
        campaign = {
            "keyword": keyword,
            "action": "CREATE_AD_GROUP",
            "opportunity_tier": tier,
            "opportunity_score": round(opportunity_score, 1),
            "growth_rate": round(growth_rate, 2),
            "weekly_searches": searches_current,
            "competitor_ads": competitor_ads,
            "relevance_score": round(relevance, 2),
            "ad_group_name": f"LT-Capture-{keyword[:20].replace(' ', '-')}",
            "targeting": [
                {"match_type": "broad",  "keyword": keyword, "bid": initial_bid},
                {"match_type": "phrase", "keyword": keyword, "bid": round(initial_bid * 1.1, 2)}
            ],
            "daily_budget": daily_budget * (1.5 if tier == "HIGH" else 1.0),
            "exploration_end_date": end_date,
            "optimization_trigger": f"探索期{exploration_days}天后，若CVR≥基准则提升出价至${round(avg_cpc, 2)}，否则暂停",
            "avg_cpc_industry": avg_cpc,
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%S")
        }
        campaigns.append(campaign)
    
    created = [c for c in campaigns if c.get("action") == "CREATE_AD_GROUP"]
    
    return {
        "total_keywords": len(keywords),
        "ad_groups_created": len(created),
        "high_opportunity": sum(1 for c in created if c.get("opportunity_tier") == "HIGH"),
        "medium_opportunity": sum(1 for c in created if c.get("opportunity_tier") == "MEDIUM"),
        "campaigns": campaigns,
        "total_daily_budget": sum(c.get("daily_budget", 0) for c in created)
    }


# 测试
keywords = [
    {
        "keyword": "postpartum belly wrap", "weekly_searches_current": 5100,
        "weekly_searches_prior": 2800, "competitor_ad_count": 2,
        "relevance_score": 0.85, "industry_avg_cpc": 1.05, "already_targeting": False
    },
    {
        "keyword": "eco friendly baby wipes", "weekly_searches_current": 3200,
        "weekly_searches_prior": 3000, "competitor_ad_count": 3,  # 增速不足
        "relevance_score": 0.90, "industry_avg_cpc": 0.80, "already_targeting": False
    },
    {
        "keyword": "bamboo diaper", "weekly_searches_current": 8500,
        "weekly_searches_prior": 4200, "competitor_ad_count": 1,
        "relevance_score": 0.95, "industry_avg_cpc": 1.50, "already_targeting": False
    },
    {
        "keyword": "electric breast pump", "weekly_searches_current": 45000,
        "weekly_searches_prior": 30000, "competitor_ad_count": 4,
        "relevance_score": 1.0, "industry_avg_cpc": 2.20, "already_targeting": True  # 已有广告
    },
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = long_tail_opportunity_auto_capture(keywords, now=now)

assert result["total_keywords"] == 4
assert result["ad_groups_created"] == 2  # postpartum + bamboo diaper

camp_map = {c["keyword"]: c for c in result["campaigns"]}
assert camp_map["postpartum belly wrap"]["action"] == "CREATE_AD_GROUP"
assert camp_map["eco friendly baby wipes"]["action"] == "NOT_TRIGGERED"  # 增速仅6.7%
assert camp_map["bamboo diaper"]["action"] == "CREATE_AD_GROUP"
assert camp_map["electric breast pump"]["action"] == "SKIP"

# bamboo diaper增速102% → 高机会
bamboo = camp_map["bamboo diaper"]
assert bamboo["opportunity_tier"] == "HIGH"

print("[✓] Long Tail Opportunity Auto Capture 测试通过")
print(f"  总关键词: {result['total_keywords']}，创建广告组: {result['ad_groups_created']}（高机会:{result['high_opportunity']}）")
print(f"  总日预算: ${result['total_daily_budget']:.0f}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Long-Tail-Keyword-Discovery]]（新兴词挖掘和搜索量趋势分析）
- **延伸（extends）**：[[Skill-Keyword-Bid-Auto-Adjuster]]（探索期结束后的智能出价调整）
- **可组合（combinable）**：[[Skill-Search-Rank-Recovery-Auto-Action]]（新词排名建立后联合监控）

## ⑤ 商业价值评估
- **ROI量化**：提前3周抢占新兴词流量，CPC $0.9 vs 成熟词 $2.5（节省64%），首月增量GMV $12,000；探索期预算 $560，ROI 21:1
- **实施难度**：⭐⭐⭐☆☆（需关键词趋势 API + 竞争密度数据 + 广告平台写入 API）
- **优先级**：⭐⭐⭐⭐☆（新兴词早期竞争低、获客成本低，是流量扩张的最优路径）
