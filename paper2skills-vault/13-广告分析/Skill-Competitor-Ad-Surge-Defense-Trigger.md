---
title: Competitor-Ad-Surge-Defense-Trigger — 竞品广告份额单日激增自动触发防御性出价提升
doc_type: knowledge
module: 13-广告分析
topic: competitor-ad-surge-defense-trigger
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Competitor-Ad-Surge-Defense-Trigger

> **配对分析层**：[[Skill-Ad-Competitive-Intelligence]]
> **决策类型**: 自动触发型 | **触发条件**: 竞品广告展示份额单日上升 > 15% | **执行动作**: 触发防御性出价提升 + 品牌词防守广告激活

## ① 算法原理

核心是「竞品广告份额监控 + 异常变化检测 + 防御性出价响应」：

1. **竞品份额监控**：通过 Amazon 广告 API 的「竞品指标报告」或第三方工具（Helium 10/Jungle Scout），每日拉取竞品在目标关键词上的「展示份额（Impression Share）」。
2. **异常检测**：计算竞品份额的单日变化量（Δ），若 Δ > 15%（绝对值），触发防御响应。
3. **防御策略**：
   - 轻度冲击（15% ≤ Δ < 30%）：品牌核心关键词出价提升 20%
   - 中度冲击（30% ≤ Δ < 50%）：品牌词 + 品类词出价提升 30%，开启「精准匹配」追加投放
   - 重度冲击（Δ ≥ 50%）：紧急响应，全词出价提升 40%，通知广告负责人，触发竞品情报收集
4. **持续时间**：防御出价调整持续 3 天，若竞品份额回落则自动恢复原出价。
5. **成本控制**：防御期间设置每日预算上限（原预算 × 1.5），避免无限制烧钱。

## ② 母婴出海应用案例

**场景：竞品在「吸奶器」类目发起广告攻势**
- 触发条件：竞品 ASIN B0XXXX 在关键词「electric breast pump」的展示份额单日从 12% → 31%（+19%）
- 执行动作：
  - 核心品牌词出价 +20%（$1.80 → $2.16）
  - 品类精准词补充出价 +20%
  - 每日预算上限提升至原 1.5x（$300 → $450）
  - 通知广告团队：竞品新上线（BSR 上升明显，建议查看其 Listing 改动）
- 持续 3 天后竞品份额回落至 15%，系统自动恢复原出价
- 业务价值：防守期间自然份额损失从预估 -25% 降至 -8%，保护了 $12,000 的周 GMV

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta

def competitor_ad_surge_defense_trigger(
    keywords: List[Dict],
    now: Optional[datetime] = None,
    light_threshold: float = 0.15,   # 轻度冲击
    medium_threshold: float = 0.30,  # 中度冲击
    heavy_threshold: float = 0.50,   # 重度冲击
    light_bid_increase: float = 0.20,
    medium_bid_increase: float = 0.30,
    heavy_bid_increase: float = 0.40,
    defense_days: int = 3,
    budget_multiplier: float = 1.5
) -> Dict:
    """
    竞品广告冲击防御触发器
    
    参数:
        keywords: [{
            "keyword_id": str, "keyword_text": str,
            "our_impression_share_yesterday": float,  # 昨日我方份额
            "our_impression_share_today": float,      # 今日我方份额
            "competitor_share_yesterday": float,      # 昨日竞品份额
            "competitor_share_today": float,          # 今日竞品份额
            "competitor_id": str,
            "current_bid": float,
            "daily_budget": float,
            "is_brand_keyword": bool
        }]
    
    返回:
        {"defenses": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    defenses = []
    
    for kw in keywords:
        kwid = kw["keyword_id"]
        kwtext = kw.get("keyword_text", kwid)
        comp_share_yesterday = kw.get("competitor_share_yesterday", 0)
        comp_share_today = kw.get("competitor_share_today", 0)
        our_share_change = kw.get("our_impression_share_today", 0) - kw.get("our_impression_share_yesterday", 0)
        comp_id = kw.get("competitor_id", "unknown")
        current_bid = kw["current_bid"]
        daily_budget = kw.get("daily_budget", 100.0)
        is_brand = kw.get("is_brand_keyword", False)
        
        # 计算竞品份额单日变化
        delta = comp_share_today - comp_share_yesterday
        
        if delta < light_threshold:
            defenses.append({
                "keyword_id": kwid,
                "keyword_text": kwtext,
                "action": "NO_DEFENSE_NEEDED",
                "competitor_share_delta": round(delta, 3),
                "reason": f"竞品份额变化{delta:+.1%}，未达{light_threshold:.0%}防御阈值"
            })
            continue
        
        # 确定防御级别
        if delta >= heavy_threshold:
            level = "HEAVY"
            bid_increase = heavy_bid_increase
            additional_actions = [
                "开启所有精准匹配词追加投放",
                "触发竞品情报收集（ASIN、新品分析）",
                "通知广告负责人，评估是否启动促销对抗"
            ]
        elif delta >= medium_threshold:
            level = "MEDIUM"
            bid_increase = medium_bid_increase
            additional_actions = [
                "品类词补充精准匹配",
                "检查竞品 Listing 更新和价格变动"
            ]
        else:
            level = "LIGHT"
            bid_increase = light_bid_increase
            additional_actions = ["监控竞品份额后续变化"]
        
        # 品牌词额外保护
        if is_brand:
            bid_increase = min(bid_increase + 0.05, 0.50)  # 品牌词额外+5%，上限50%
        
        new_bid = round(current_bid * (1 + bid_increase), 2)
        new_budget = round(daily_budget * budget_multiplier, 2)
        defense_until = (now + timedelta(days=defense_days)).strftime("%Y-%m-%d")
        
        defense = {
            "keyword_id": kwid,
            "keyword_text": kwtext,
            "action": "DEFENSE_ACTIVATED",
            "defense_level": level,
            "competitor_id": comp_id,
            "competitor_share_delta": round(delta, 3),
            "competitor_share_today": round(comp_share_today, 3),
            "our_share_change": round(our_share_change, 3),
            "current_bid": current_bid,
            "defense_bid": new_bid,
            "bid_increase_pct": round(bid_increase * 100, 1),
            "daily_budget": daily_budget,
            "defense_budget": new_budget,
            "defense_until": defense_until,
            "auto_recover_condition": f"竞品份额回落至{comp_share_yesterday:.0%}以下，自动恢复原出价",
            "additional_actions": additional_actions,
            "is_brand_keyword": is_brand
        }
        defenses.append(defense)
    
    activated = [d for d in defenses if d.get("action") == "DEFENSE_ACTIVATED"]
    level_counts = {}
    for d in activated:
        lv = d.get("defense_level", "")
        level_counts[lv] = level_counts.get(lv, 0) + 1
    
    return {
        "total_keywords": len(keywords),
        "defense_activated": len(activated),
        "level_summary": level_counts,
        "defenses": defenses,
        "total_defense_budget_increase": sum(d.get("defense_budget", 0) - d.get("daily_budget", 0) for d in activated)
    }


# 测试
keywords = [
    {
        "keyword_id": "KW001", "keyword_text": "electric breast pump",
        "our_impression_share_yesterday": 0.35, "our_impression_share_today": 0.28,
        "competitor_share_yesterday": 0.12, "competitor_share_today": 0.31,  # +19%，轻度
        "competitor_id": "ASIN-B0XXXX",
        "current_bid": 1.80, "daily_budget": 300.0, "is_brand_keyword": False
    },
    {
        "keyword_id": "KW002", "keyword_text": "BrandName breast pump",
        "our_impression_share_yesterday": 0.80, "our_impression_share_today": 0.55,
        "competitor_share_yesterday": 0.05, "competitor_share_today": 0.45,  # +40%，中度，品牌词
        "competitor_id": "ASIN-B0YYYY",
        "current_bid": 2.50, "daily_budget": 200.0, "is_brand_keyword": True
    },
    {
        "keyword_id": "KW003", "keyword_text": "baby bottle nipple",
        "our_impression_share_yesterday": 0.20, "our_impression_share_today": 0.19,
        "competitor_share_yesterday": 0.15, "competitor_share_today": 0.18,  # +3%，不触发
        "competitor_id": "ASIN-B0ZZZZ",
        "current_bid": 0.90, "daily_budget": 80.0, "is_brand_keyword": False
    },
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = competitor_ad_surge_defense_trigger(keywords, now=now)

assert result["total_keywords"] == 3
assert result["defense_activated"] == 2

def_map = {d["keyword_id"]: d for d in result["defenses"]}
assert def_map["KW001"]["defense_level"] == "LIGHT"
assert def_map["KW002"]["defense_level"] == "MEDIUM"
assert def_map["KW003"]["action"] == "NO_DEFENSE_NEEDED"

# 品牌词应有额外保护
brand_defense = def_map["KW002"]
assert brand_defense["bid_increase_pct"] > 30  # 30% + 5% = 35%

print("[✓] Competitor Ad Surge Defense Trigger 测试通过")
print(f"  总关键词: {result['total_keywords']}，触发防御: {result['defense_activated']}")
print(f"  防御级别: {result['level_summary']}，预算增量: ${result['total_defense_budget_increase']:.0f}/天")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Ad-Competitive-Intelligence]]（竞品广告份额数据来源）
- **延伸（extends）**：[[Skill-Brand-Keyword-Hijack-Alert]]（品牌词被劫持场景的专项防御）
- **可组合（combinable）**：[[Skill-Keyword-Bid-Auto-Adjuster]]（防御结束后恢复正常出价优化）

## ⑤ 商业价值评估
- **ROI量化**：防守期间份额损失从 -25% → -8%，保护周 GMV $12,000；防守预算增量 $150/天 × 3天 = $450，ROI 26:1
- **实施难度**：⭐⭐⭐☆☆（需竞品广告份额 API + 实时监控 + 广告平台写入权限）
- **优先级**：⭐⭐⭐⭐☆（竞品突袭在大促前后频发，防御响应时效关键）
