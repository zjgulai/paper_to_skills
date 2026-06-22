---
title: Search-Rank-Recovery-Auto-Action — 核心关键词排名跌出Page1自动触发三步恢复行动
doc_type: knowledge
module: 25-搜索流量工程
topic: search-rank-recovery-auto-action
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Search-Rank-Recovery-Auto-Action

> **配对分析层**：[[Skill-Search-Rank-Tracking-Analytics]]
> **决策类型**: 自动触发型 | **触发条件**: 核心关键词排名跌出 Page 1（> 第16位） | **执行动作**: 触发Listing优化 + 广告补位 + 索引检查三步行动

## ① 算法原理

核心是「排名变化检测 + 跌出 Page1 识别 + 三步恢复行动并行触发」：

1. **排名监控**：每日拉取核心关键词自然排名（Organic Rank），与前日对比，识别「今日排名 > 16 且昨日排名 ≤ 16」的跌出事件。
2. **跌出严重程度分级**：
   - 轻度跌出（排名 17-30）：触发 Listing 优化检查 + 广告小幅补位
   - 中度跌出（排名 31-60）：三步行动全触发，优先处理
   - 重度跌出（排名 > 60 或索引丢失）：紧急响应，升级给运营负责人
3. **三步行动**：
   - **Step 1 Listing 优化**：检查标题关键词密度、五点、后端 Search Terms 是否包含该关键词
   - **Step 2 广告补位**：在搜索结果页顶部（Top of Search）创建广告活动，确保有广告展示
   - **Step 3 索引检查**：使用 `site:amazon.com + ASIN + keyword` 验证产品是否仍被索引
4. **恢复效果追踪**：7 天后自动检查排名是否恢复到 Page 1（≤ 16）。

## ② 母婴出海应用案例

**场景：「electric breast pump」核心词排名暴跌**
- 触发条件：ASIN B0PUMP01 在「electric breast pump」自然排名从 第8位 → 第34位（跌出Page1，中度）
- 根因排查：竞品新上线（BSR 上升）+ Listing 标题关键词被更新时遗漏主词
- 三步行动：
  - Step 1：修复标题（加回「electric breast pump」）+ 后端 Search Terms 补全
  - Step 2：创建 SP 广告活动「electric breast pump Exact Match」，Top of Search 出价 $3.50
  - Step 3：索引验证通过（正常），标记为无需处理
- 恢复结果：7 天后排名回到 Page 1 第11位
- 业务价值：避免核心词排名跌出 Page1 损失流量 ~60%，保护周均 GMV $8,500

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta

def search_rank_recovery_auto_action(
    keyword_rankings: List[Dict],
    now: Optional[datetime] = None,
    page1_threshold: int = 16,
    light_threshold: int = 30,
    medium_threshold: int = 60,
    top_of_search_bid_multiplier: float = 1.5
) -> Dict:
    """
    搜索排名跌出Page1自动恢复触发器
    
    参数:
        keyword_rankings: [{
            "asin": str, "keyword": str,
            "rank_today": int, "rank_yesterday": int,
            "current_bid": float,
            "listing_has_keyword_in_title": bool,
            "listing_has_keyword_in_backend": bool,
            "indexed": bool  # 是否被Amazon索引
        }]
        page1_threshold: Page1阈值（默认第16位）
        top_of_search_bid_multiplier: Top of Search出价倍数
    
    返回:
        {"actions": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    actions = []
    
    for kw_rank in keyword_rankings:
        asin = kw_rank["asin"]
        keyword = kw_rank["keyword"]
        rank_today = kw_rank.get("rank_today", 999)
        rank_yesterday = kw_rank.get("rank_yesterday", 999)
        current_bid = kw_rank.get("current_bid", 1.5)
        title_has_kw = kw_rank.get("listing_has_keyword_in_title", True)
        backend_has_kw = kw_rank.get("listing_has_keyword_in_backend", True)
        indexed = kw_rank.get("indexed", True)
        
        # 检查是否跌出 Page1（今日排名 > 16 且昨日排名 ≤ 16）
        dropped_out = rank_today > page1_threshold and rank_yesterday <= page1_threshold
        
        if not dropped_out:
            actions.append({
                "asin": asin, "keyword": keyword,
                "action": "NO_ACTION",
                "rank_today": rank_today,
                "rank_change": rank_today - rank_yesterday,
                "reason": "排名未跌出Page1" if rank_today <= page1_threshold else "昨日已不在Page1"
            })
            continue
        
        rank_drop = rank_today - rank_yesterday
        
        # 确定严重程度
        if not indexed:
            severity = "CRITICAL"
        elif rank_today > medium_threshold:
            severity = "HEAVY"
        elif rank_today > light_threshold:
            severity = "MEDIUM"
        else:
            severity = "LIGHT"
        
        # Step 1: Listing 优化建议
        listing_tasks = []
        if not title_has_kw:
            listing_tasks.append(f"【高优先级】在标题中添加关键词「{keyword}」（位置：前50字符）")
        if not backend_has_kw:
            listing_tasks.append(f"在后端 Search Terms 添加「{keyword}」")
        listing_tasks.append("检查关键词在五点和描述中的密度（目标出现2-3次）")
        if severity in ["HEAVY", "CRITICAL"]:
            listing_tasks.append("申请 A+ Content 更新，增加关键词内容密度")
        
        # Step 2: 广告补位
        top_bid = round(current_bid * top_of_search_bid_multiplier, 2)
        ad_campaign = {
            "action": "CREATE_SP_CAMPAIGN",
            "campaign_name": f"Rank-Recovery-{asin}-{keyword[:15]}",
            "targeting": f'"{keyword}" Exact Match',
            "bid": top_bid,
            "placement": "Top of Search",
            "daily_budget": round(top_bid * 20, 2),  # 20次点击的预算
            "duration_days": 14,
            "note": f"排名恢复期间维持Top of Search广告位，恢复后可调降出价"
        }
        
        # Step 3: 索引检查
        index_check = {
            "action": "INDEX_VERIFICATION",
            "method": f'Amazon搜索验证：搜索「{keyword} {asin}」，确认产品出现',
            "indexed": indexed,
            "reindex_action": "联系 Amazon Seller Central 提交 Reindex 申请" if not indexed else "索引正常，无需操作"
        }
        
        recovery_action = {
            "asin": asin,
            "keyword": keyword,
            "action": "RANK_RECOVERY_TRIGGERED",
            "severity": severity,
            "rank_yesterday": rank_yesterday,
            "rank_today": rank_today,
            "rank_drop": rank_drop,
            "step1_listing": {"tasks": listing_tasks, "priority": "HIGH" if not title_has_kw else "MEDIUM"},
            "step2_ad": ad_campaign,
            "step3_index": index_check,
            "recovery_check_date": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "recovery_target": f"7天内恢复至Page1（排名≤{page1_threshold}）",
            "escalate_to_human": severity in ["HEAVY", "CRITICAL"]
        }
        actions.append(recovery_action)
    
    triggered = [a for a in actions if a.get("action") == "RANK_RECOVERY_TRIGGERED"]
    
    return {
        "total_keywords": len(keyword_rankings),
        "recovery_triggered": len(triggered),
        "critical_count": sum(1 for a in triggered if a["severity"] == "CRITICAL"),
        "heavy_count": sum(1 for a in triggered if a["severity"] == "HEAVY"),
        "actions": actions
    }


# 测试
keyword_rankings = [
    {
        "asin": "B0PUMP01", "keyword": "electric breast pump",
        "rank_today": 34, "rank_yesterday": 8,  # 跌出Page1，中度
        "current_bid": 2.0,
        "listing_has_keyword_in_title": False,  # 标题缺失关键词
        "listing_has_keyword_in_backend": True,
        "indexed": True
    },
    {
        "asin": "B0BOTTLE01", "keyword": "baby bottle",
        "rank_today": 5, "rank_yesterday": 4,  # 未跌出Page1
        "current_bid": 1.2,
        "listing_has_keyword_in_title": True,
        "listing_has_keyword_in_backend": True,
        "indexed": True
    },
    {
        "asin": "B0WIPES01", "keyword": "baby wipes sensitive",
        "rank_today": 95, "rank_yesterday": 12,  # 跌出Page1，重度
        "current_bid": 0.8,
        "listing_has_keyword_in_title": False,
        "listing_has_keyword_in_backend": False,
        "indexed": False  # 索引丢失，CRITICAL
    },
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = search_rank_recovery_auto_action(keyword_rankings, now=now)

assert result["total_keywords"] == 3
assert result["recovery_triggered"] == 2
assert result["critical_count"] == 1  # B0WIPES01

action_map = {a["asin"]: a for a in result["actions"]}
assert action_map["B0PUMP01"]["severity"] == "MEDIUM"
assert action_map["B0BOTTLE01"]["action"] == "NO_ACTION"
assert action_map["B0WIPES01"]["severity"] == "CRITICAL"
assert action_map["B0WIPES01"]["step3_index"]["indexed"] == False

print("[✓] Search Rank Recovery Auto Action 测试通过")
print(f"  总关键词: {result['total_keywords']}，触发恢复: {result['recovery_triggered']}（CRITICAL:{result['critical_count']}）")
for a in result["actions"]:
    if a.get("action") == "RANK_RECOVERY_TRIGGERED":
        print(f"  [{a['severity']}] {a['asin']} 「{a['keyword']}」: {a['rank_yesterday']}位→{a['rank_today']}位")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Search-Rank-Tracking-Analytics]]（每日关键词排名追踪数据来源）
- **延伸（extends）**：[[Skill-Brand-Keyword-Hijack-Alert]]（品牌词排名异常专项处理）
- **可组合（combinable）**：[[Skill-Keyword-Bid-Auto-Adjuster]]（排名恢复后出价策略动态调整）

## ⑤ 商业价值评估
- **ROI量化**：核心词跌出Page1流量损失约60%，快速恢复保护周均GMV $8,500；恢复广告投入约$350，ROI 24:1
- **实施难度**：⭐⭐⭐☆☆（需关键词排名追踪工具 + 广告 API + Listing 编辑权限）
- **优先级**：⭐⭐⭐⭐⭐（搜索排名是自然流量的核心，跌出Page1直接影响80%自然订单）
