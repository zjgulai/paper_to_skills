---
title: Brand-Keyword-Hijack-Alert — 品牌词搜索下竞品展示份额超30%触发品牌防守广告扩展
doc_type: knowledge
module: 25-搜索流量工程
topic: brand-keyword-hijack-alert
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Brand-Keyword-Hijack-Alert

> **配对分析层**：[[Skill-Brand-Search-Share-Analytics]]
> **决策类型**: 自动触发型 | **触发条件**: 品牌词搜索下竞品展示份额 > 30% | **执行动作**: 触发品牌词防守广告自动扩展 + 竞品情报采集

## ① 算法原理

核心是「品牌词竞品份额监控 + 阈值告警 + 防守广告自动扩展」：

1. **品牌词定义**：包含品牌名称的关键词（如「Medela 吸奶器」「Philips Avent 奶瓶」）。
2. **份额监控**：每日拉取品牌词下各 ASIN 的展示份额，聚合计算竞品总份额（非品牌方产品）。
3. **触发规则**：竞品展示份额 > 30%（即品牌词搜索下每3次展示就有1次是竞品）。
4. **防守广告扩展**：
   - 品牌词 Sponsored Brands（SB）广告：确保品牌词搜索首位始终有品牌广告横幅
   - Sponsored Products（SP）品牌词精准匹配：确保产品广告位覆盖
   - 出价提升至「Top of Search 优先」
5. **竞品情报**：记录抢占品牌词的竞品 ASIN，触发价格/评分/Listing 对比分析。
6. **品牌词保护注册**：若品牌已注册 Amazon Brand Registry，可申请「Brand Keyword Protection」。

## ② 母婴出海应用案例

**场景：某母婴品牌「MomsChoice」品牌词被竞品大量投放**
- 触发条件：搜索「MomsChoice breast pump」时，竞品总展示份额 = 38%（阈值 30%）
- 执行动作：
  - 创建/更新 SB 广告「MomsChoice 吸奶器」，出价提升至 $4.50（Top of Search）
  - SP 广告「MomsChoice」Exact Match，出价 +30%（$2.80 → $3.64）
  - 自动记录竞品 ASINs（B0XXXX, B0YYYY），触发竞品情报报告
  - 通知品牌团队：检查是否可申请「Brand Keyword Protection」
- 业务价值：品牌词自然份额从 62% → 85%，品牌词带来的订单 +23%，月均增量 GMV $35,000

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime

def brand_keyword_hijack_alert(
    brand_keywords: List[Dict],
    now: Optional[datetime] = None,
    hijack_threshold: float = 0.30,
    severe_threshold: float = 0.50,
    bid_increase_sp: float = 0.30,
    bid_increase_sb: float = 0.20,
    top_of_search_multiplier: float = 1.5
) -> Dict:
    """
    品牌词劫持告警与防守广告触发器
    
    参数:
        brand_keywords: [{
            "brand": str, "keyword": str,
            "our_impression_share": float,      # 我方品牌词展示份额
            "competitor_impression_share": float, # 竞品总展示份额
            "top_competitor_asins": List[str],
            "current_sp_bid": float,    # 当前SP出价
            "current_sb_bid": float,    # 当前SB出价（若无SB则为0）
            "has_sb_campaign": bool,    # 是否有SB广告活动
            "brand_registered": bool    # 是否已注册Brand Registry
        }]
    
    返回:
        {"alerts": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    alerts = []
    
    for kw in brand_keywords:
        brand = kw["brand"]
        keyword = kw["keyword"]
        our_share = kw.get("our_impression_share", 1.0)
        comp_share = kw.get("competitor_impression_share", 0.0)
        top_asins = kw.get("top_competitor_asins", [])
        sp_bid = kw.get("current_sp_bid", 1.5)
        sb_bid = kw.get("current_sb_bid", 0)
        has_sb = kw.get("has_sb_campaign", False)
        brand_registered = kw.get("brand_registered", False)
        
        if comp_share < hijack_threshold:
            alerts.append({
                "keyword": keyword,
                "action": "HEALTHY",
                "competitor_share": comp_share,
                "our_share": our_share,
                "reason": f"竞品份额{comp_share:.0%}，低于{hijack_threshold:.0%}阈值"
            })
            continue
        
        # 判断严重程度
        if comp_share >= severe_threshold:
            severity = "SEVERE"
            sp_bid_multiplier = 1 + bid_increase_sp * 1.5
            escalate = True
        else:
            severity = "MODERATE"
            sp_bid_multiplier = 1 + bid_increase_sp
            escalate = False
        
        new_sp_bid = round(sp_bid * sp_bid_multiplier, 2)
        new_sb_bid = round(max(sb_bid, sp_bid) * (1 + bid_increase_sb) if has_sb else sp_bid * 1.2, 2)
        
        # 防守行动
        defense_actions = [
            {
                "type": "UPDATE_SP_BRAND_KEYWORD",
                "action": f"更新SP广告品牌词精准匹配出价: ${sp_bid} → ${new_sp_bid}",
                "placement": "Top of Search",
                "new_bid": new_sp_bid
            }
        ]
        
        if has_sb:
            defense_actions.append({
                "type": "UPDATE_SB_BRAND_CAMPAIGN",
                "action": f"更新SB品牌广告出价: ${sb_bid} → ${new_sb_bid}",
                "placement": "Top Banner",
                "new_bid": new_sb_bid
            })
        else:
            defense_actions.append({
                "type": "CREATE_SB_CAMPAIGN",
                "action": "创建 Sponsored Brands 品牌广告（需Brand Registry）" if brand_registered else "建议注册Brand Registry后创建SB广告",
                "priority": "HIGH" if brand_registered else "MEDIUM"
            })
        
        if brand_registered and comp_share >= severe_threshold:
            defense_actions.append({
                "type": "BRAND_KEYWORD_PROTECTION",
                "action": "申请Amazon Brand Keyword Protection（阻止竞品在品牌词中展示广告）",
                "link": "Seller Central > Brand Registry > Brand Protection"
            })
        
        # 竞品情报触发
        intel_task = {
            "type": "COMPETITOR_INTEL",
            "competitor_asins": top_asins,
            "intel_items": [
                "竞品定价 vs 我方定价",
                "竞品评分/评论数对比",
                "竞品Listing关键词分析",
                "竞品广告策略（出价估算）"
            ]
        }
        
        alert = {
            "brand": brand,
            "keyword": keyword,
            "action": "BRAND_HIJACK_DEFENSE_TRIGGERED",
            "severity": severity,
            "our_impression_share": our_share,
            "competitor_impression_share": comp_share,
            "hijack_level": f"竞品每100次品牌词展示中占{int(comp_share*100)}次",
            "top_competitor_asins": top_asins,
            "defense_actions": defense_actions,
            "competitor_intel": intel_task,
            "escalate_to_human": escalate,
            "triggered_at": now.strftime("%Y-%m-%dT%H:%M:%S")
        }
        alerts.append(alert)
    
    triggered = [a for a in alerts if a.get("action") == "BRAND_HIJACK_DEFENSE_TRIGGERED"]
    
    return {
        "total_brand_keywords": len(brand_keywords),
        "hijack_detected": len(triggered),
        "severe_count": sum(1 for a in triggered if a["severity"] == "SEVERE"),
        "alerts": alerts
    }


# 测试
brand_keywords = [
    {
        "brand": "MomsChoice", "keyword": "MomsChoice breast pump",
        "our_impression_share": 0.62, "competitor_impression_share": 0.38,
        "top_competitor_asins": ["B0XXXX01", "B0YYYY02"],
        "current_sp_bid": 2.80, "current_sb_bid": 3.50,
        "has_sb_campaign": True, "brand_registered": True
    },
    {
        "brand": "MomsChoice", "keyword": "MomsChoice baby bottle",
        "our_impression_share": 0.88, "competitor_impression_share": 0.12,
        "top_competitor_asins": [],
        "current_sp_bid": 1.20, "current_sb_bid": 0,
        "has_sb_campaign": False, "brand_registered": True
    },
    {
        "brand": "BabyFirst", "keyword": "BabyFirst stroller",
        "our_impression_share": 0.45, "competitor_impression_share": 0.55,  # 严重
        "top_competitor_asins": ["B0ZZZZ01", "B0AAAA02", "B0BBBB03"],
        "current_sp_bid": 3.00, "current_sb_bid": 0,
        "has_sb_campaign": False, "brand_registered": False
    },
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = brand_keyword_hijack_alert(brand_keywords, now=now)

assert result["total_brand_keywords"] == 3
assert result["hijack_detected"] == 2  # MomsChoice pump (38%) 和 BabyFirst stroller (55%)
assert result["severe_count"] == 1  # BabyFirst stroller 55%

alert_map = {a["keyword"]: a for a in result["alerts"]}
assert alert_map["MomsChoice breast pump"]["severity"] == "MODERATE"
assert alert_map["MomsChoice baby bottle"]["action"] == "HEALTHY"
assert alert_map["BabyFirst stroller"]["severity"] == "SEVERE"

print("[✓] Brand Keyword Hijack Alert 测试通过")
print(f"  总品牌词: {result['total_brand_keywords']}，检测到劫持: {result['hijack_detected']}（严重:{result['severe_count']}）")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Brand-Search-Share-Analytics]]（品牌词展示份额数据分析）
- **延伸（extends）**：[[Skill-Competitor-Ad-Surge-Defense-Trigger]]（品牌词防御与泛类目防御联动）
- **可组合（combinable）**：[[Skill-Search-Rank-Recovery-Auto-Action]]（品牌词排名跌落联动恢复）

## ⑤ 商业价值评估
- **ROI量化**：品牌词自然份额从 62% → 85%，月均增量 GMV $35,000；防守广告追加成本 $800/月，ROI 43:1
- **实施难度**：⭐⭐⭐☆☆（需广告份额报告 API + SB 广告权限（Brand Registry）+ 自动化出价接口）
- **优先级**：⭐⭐⭐⭐⭐（品牌词被劫持是「用自己的品牌为竞品引流」，属极高优先级防御任务）
