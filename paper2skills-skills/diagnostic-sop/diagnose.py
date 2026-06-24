#!/usr/bin/env python3
"""
paper2skills 通用业务诊断 SOP
=============================
输入：SKU 名称 + 渠道 + 问题类型 + 数据文件（可选）
输出：三层诊断报告（诊断根因 / 处置行动 / 预防机制）+ 行动清单

用法（交互模式）:
    python3 diagnose.py

用法（参数模式）:
    python3 diagnose.py --sku "暖奶器" --channel amazon --problem repurchase_drop

用法（带真实数据）:
    python3 diagnose.py --sku "奶粉" --channel amazon --problem repurchase_drop \
        --reviews reviews.csv --prices prices.csv --users users.csv

支持的问题类型:
    repurchase_drop   复购率下降
    roas_decline      广告 ROAS 下滑
    traffic_drop      ASIN 流量异常
    new_cvr_low       新客转化率低
    inventory_issue   库存积压/断货
    review_attack     差评攻击/评分下滑
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

VERSION = "1.0.0"
SKILL_BASE_URL = "https://skills.lute-tlz-dddd.top/skills"


# ══════════════════════════════════════════════════════
# 配置：问题类型 → 诊断模块映射
# ══════════════════════════════════════════════════════

PROBLEM_CONFIG = {
    "repurchase_drop": {
        "name": "复购率下降",
        "modules": ["voc", "competitor_price", "baby_age_clock", "causal_attribution"],
        "primary_skills": [
            "Skill-VOC-Aspect-Sentiment-Extraction",
            "Skill-Review-Pain-Point-Mining",
            "Skill-Competitor-Price-Intelligence",
            "Skill-Baby-Age-Clock-RFM-Enhancement",
            "Skill-Causal-Churn-Retention-Attribution",
        ],
        "treatment_skills": [
            "Skill-SCRABLE-Review-Response-Generation",
            "Skill-Review-Defense-Vine-Optimizer",
            "Skill-Repurchase-Trigger-Timing-Model",
            "Skill-Baby-Age-Aware-Recommendation",
            "Skill-Listing-AI-Copywriting",
        ],
        "prevention_skills": [
            "Skill-Membership-Churn-Early-Warning-Graph",
            "Skill-Real-Time-Competitive-Repricing",
            "Skill-Cohort-Retention-Analysis",
        ],
    },
    "roas_decline": {
        "name": "广告 ROAS 下滑",
        "modules": ["competitor_price", "creative_fatigue", "causal_attribution"],
        "primary_skills": [
            "Skill-Creative-Fatigue-Detection",
            "Skill-Competitor-Price-Intelligence",
            "Skill-Counterfactual-Ad-Attribution-Debiasing",
            "Skill-ROAS-Budget-Optimization",
        ],
        "treatment_skills": [
            "Skill-Ad-Creative-Personalization-Bandit",
            "Skill-Constrained-Multi-Objective-Ad-Delivery",
            "Skill-Cross-Channel-Budget-Pacing-Controller",
            "Skill-Negative-Keyword-Safe-Guard",
        ],
        "prevention_skills": [
            "Skill-Autobidding-Budget-Allocation-Optimization",
            "Skill-Share-of-Voice-Tracking",
        ],
    },
    "traffic_drop": {
        "name": "ASIN 流量异常下跌",
        "modules": ["voc", "competitor_price", "causal_attribution"],
        "primary_skills": [
            "Skill-Amazon-A10-Algorithm-Ranking",
            "Skill-VOC-Aspect-Sentiment-Extraction",
            "Skill-Competitor-Product-Intelligence",
            "Skill-Keyword-Competition-Scoring",
        ],
        "treatment_skills": [
            "Skill-Listing-AI-Copywriting",
            "Skill-Long-Tail-Search-Embedding-SEO",
            "Skill-Amazon-External-Traffic-Boost",
            "Skill-Review-Defense-Vine-Optimizer",
        ],
        "prevention_skills": [
            "Skill-SEO-Organic-Ranking-Optimization",
            "Skill-Listing-Health-Diagnostic",
        ],
    },
    "new_cvr_low": {
        "name": "新客转化率低",
        "modules": ["voc", "competitor_price"],
        "primary_skills": [
            "Skill-Listing-Health-Diagnostic",
            "Skill-Listing-Quality-Scoring",
            "Skill-VOC-Aspect-Sentiment-Extraction",
            "Skill-Competitor-Product-Intelligence",
        ],
        "treatment_skills": [
            "Skill-Listing-AI-Copywriting",
            "Skill-Shopify-Landing-Page-CRO",
            "Skill-Social-Proof-Amplification",
        ],
        "prevention_skills": [
            "Skill-Listing-AB-Testing-Automation",
            "Skill-Purchase-Intent-Prediction",
        ],
    },
    "inventory_issue": {
        "name": "库存积压/断货",
        "modules": ["competitor_price", "causal_attribution"],
        "primary_skills": [
            "Skill-ITO-Three-Phase-Health-Tracking",
            "Skill-Dynamic-ABC-Stratification-Adaptive-Policy",
            "Skill-Forecast-Driven-Inventory",
            "Skill-Infant-Lifecycle-Purchase-Rhythm",
        ],
        "treatment_skills": [
            "Skill-Long-Tail-SKU-Clearance-Optimization",
            "Skill-Markdown-Optimization",
            "Skill-Multi-Channel-Inventory-Sync",
        ],
        "prevention_skills": [
            "Skill-Safety-Stock-Replenishment",
            "Skill-Baby-Age-Clock-RFM-Enhancement",
        ],
    },
    "review_attack": {
        "name": "差评攻击/评分下滑",
        "modules": ["voc"],
        "primary_skills": [
            "Skill-Fake-Review-Detection",
            "Skill-DS-DGA-GCN-Fake-Review-Group",
            "Skill-VOC-Aspect-Sentiment-Extraction",
            "Skill-Review-Pain-Point-Mining",
        ],
        "treatment_skills": [
            "Skill-SCRABLE-Review-Response-Generation",
            "Skill-Review-Defense-Vine-Optimizer",
            "Skill-Post-Purchase-Review-Request-Dispatcher",
        ],
        "prevention_skills": [
            "Skill-AutoQual-Review-Quality-Assessment",
            "Skill-Brand-Listing-Hijacking-Detection",
        ],
    },
}


# ══════════════════════════════════════════════════════
# 诊断模块：VOC 评价分析
# ══════════════════════════════════════════════════════

def run_voc_module(reviews_df: Optional[pd.DataFrame], sku: str) -> Dict[str, Any]:
    """VOC 评价诊断（Skill-VOC-Aspect-Sentiment-Extraction）"""
    if reviews_df is None or reviews_df.empty:
        # 生成模拟数据
        np.random.seed(42)
        n = 300
        periods = ['before'] * 150 + ['after'] * 150
        ratings_before = np.random.choice([1,2,3,4,5], 150, p=[0.03,0.05,0.10,0.32,0.50])
        ratings_after  = np.random.choice([1,2,3,4,5], 150, p=[0.12,0.16,0.17,0.28,0.27])
        pain_before = ['质量稳定','操作简单','物超所值','宝宝喜欢','安全放心']
        pain_after  = ['加热不稳定','噪音大','存在安全隐患','加热故障','退货处理差',
                       '漏水','做工粗糙','不值这个价']
        texts = ([np.random.choice(pain_before) for _ in range(150)] +
                 [np.random.choice(pain_after)  for _ in range(150)])
        reviews_df = pd.DataFrame({'rating': np.concatenate([ratings_before, ratings_after]),
                                    'period': periods, 'review_text': texts})
        data_source = "⚠️  模拟数据（建议导入真实评价数据）"
    else:
        data_source = "✅ 真实数据"

    before = reviews_df[reviews_df['period'] == 'before']
    after  = reviews_df[reviews_df['period'] == 'after']
    avg_before = before['rating'].mean()
    avg_after  = after['rating'].mean()
    bad_before = (before['rating'] <= 2).mean()
    bad_after  = (after['rating'] <= 2).mean()

    from collections import Counter
    low_texts = after[after['rating'] <= 2]['review_text'].tolist()
    pain_counter = Counter(low_texts)
    top_pains = pain_counter.most_common(5)

    critical_kw = ['安全隐患', '漏水', '烧坏', '故障', '危险', 'hazard', 'unsafe']
    critical_cnt = sum(1 for t in low_texts if any(k in str(t) for k in critical_kw))
    critical_pct = critical_cnt / max(len(low_texts), 1)

    severity = "🔴 严重" if avg_after < 3.5 or critical_pct > 0.3 else (
               "🟠 中等" if avg_after < 4.0 else "🟡 轻微")

    return {
        "module": "VOC 评价分析",
        "data_source": data_source,
        "avg_before": round(avg_before, 2),
        "avg_after":  round(avg_after, 2),
        "rating_drop": round(avg_before - avg_after, 2),
        "bad_rate_before": round(bad_before, 3),
        "bad_rate_after":  round(bad_after, 3),
        "top_pains": top_pains,
        "critical_pct": round(critical_pct, 2),
        "severity": severity,
        "conclusion": (
            f"评分从 ★{avg_before:.1f} 下降到 ★{avg_after:.1f}（-{avg_before-avg_after:.1f}分），"
            f"差评率上升 {(bad_after-bad_before):.0%}，"
            f"{'存在安全类严重差评' if critical_pct > 0.2 else '无安全类差评但体验类差评增多'}"
        ),
    }


# ══════════════════════════════════════════════════════
# 诊断模块：竞品价格监控
# ══════════════════════════════════════════════════════

def run_competitor_price_module(prices_df: Optional[pd.DataFrame], sku: str) -> Dict[str, Any]:
    """竞品价格诊断（Skill-Competitor-Price-Intelligence）"""
    if prices_df is None or prices_df.empty:
        np.random.seed(42)
        n = 60
        own_price = 38.99
        prices_df = pd.DataFrame({
            'day': list(range(n)),
            'own_price': own_price,
            'comp_a': [38.99 - (0 if i < 30 else 4.5) + np.random.uniform(-0.5, 0.5) for i in range(n)],
            'comp_b': [44.99 + np.random.uniform(-0.8, 0.8) for _ in range(n)],
            'comp_c': [29.99 - (0 if i < 25 else 5.5) + np.random.uniform(-0.3, 0.3) for i in range(n)],
        })
        data_source = "⚠️  模拟数据"
    else:
        data_source = "✅ 真实数据"

    own_price = prices_df['own_price'].mean()
    comp_cols = [c for c in prices_df.columns if c not in ['day', 'date', 'own_price']]
    mid = len(prices_df) // 2

    results = []
    for col in comp_cols:
        p_before = prices_df[col][:mid].mean()
        p_after  = prices_df[col][mid:].mean()
        delta    = p_after - p_before
        gap_now  = own_price - p_after
        action   = "⚠️ 降价抢市场" if delta < -2 else ("✅ 价格稳定" if abs(delta) < 1 else "轻微变化")
        results.append({"comp": col, "before": round(p_before,2), "after": round(p_after,2),
                        "delta": round(delta,2), "gap": round(gap_now,2), "action": action})

    cheapest = min(r['after'] for r in results)
    premium  = own_price - cheapest
    severity = "🔴 严重" if premium > own_price * 0.15 else (
               "🟠 中等" if premium > own_price * 0.05 else "🟡 轻微")

    return {
        "module": "竞品价格监控",
        "data_source": data_source,
        "own_price": round(own_price, 2),
        "competitors": results,
        "cheapest_comp": round(cheapest, 2),
        "premium_usd": round(premium, 2),
        "premium_pct": round(premium / own_price, 3),
        "severity": severity,
        "conclusion": (
            f"市场最低竞品价 ${cheapest:.2f}，我方溢价 ${premium:.2f}"
            f"（{premium/own_price:.0%}），"
            f"{'竞品价格战明显，价格敏感用户流失风险高' if premium > 3 else '价格差距在可接受范围内'}"
        ),
    }


# ══════════════════════════════════════════════════════
# 诊断模块：婴儿月龄时钟（VOID 框架贡献）
# ══════════════════════════════════════════════════════

def run_baby_age_clock_module(users_df: Optional[pd.DataFrame], sku: str) -> Dict[str, Any]:
    """月龄时钟诊断（Skill-Baby-Age-Clock-RFM-Enhancement）"""
    baby_relevant = any(kw in sku.lower() for kw in
                        ['奶', '暖', '辅食', '尿', '婴', 'formula', 'warmer',
                         'diaper', 'baby', 'infant', 'feeder', '哺'])

    if not baby_relevant:
        return {"module": "婴儿月龄时钟", "applicable": False,
                "conclusion": "该 SKU 非母婴专属品类，月龄时钟模块不适用"}

    if users_df is None or users_df.empty:
        np.random.seed(42)
        n = 200
        ages = np.random.choice(range(1, 13), n,
                                 p=[0.06,0.10,0.13,0.14,0.13,0.12,0.10,0.08,0.07,0.05,0.01,0.01])
        churned = [1 if (a >= 7 and np.random.random() < 0.55) or
                       (a < 7 and np.random.random() < 0.20) else 0 for a in ages]
        users_df = pd.DataFrame({'baby_age_months': ages, 'churned': churned})
        data_source = "⚠️  模拟数据"
    else:
        data_source = "✅ 真实数据"

    churned_df = users_df[users_df['churned'] == 1]
    total_churn = len(churned_df)

    natural_exit = (churned_df['baby_age_months'] >= 7).sum()
    real_churn   = (churned_df['baby_age_months'] < 7).sum()
    natural_pct  = natural_exit / max(total_churn, 1)
    real_pct     = real_churn   / max(total_churn, 1)

    age_dist = churned_df['baby_age_months'].value_counts().sort_index()

    return {
        "module": "婴儿月龄时钟",
        "applicable": True,
        "data_source": data_source,
        "total_churned": total_churn,
        "natural_exit": int(natural_exit),
        "real_churn":   int(real_churn),
        "natural_pct":  round(natural_pct, 3),
        "real_pct":     round(real_pct, 3),
        "age_distribution": age_dist.to_dict(),
        "void_insight": True,
        "conclusion": (
            f"流失用户中 {natural_pct:.0%} 是月龄驱动的自然退出（宝宝≥7月龄），"
            f"真正需要挽回的只有 {real_churn} 人（{real_pct:.0%}）。"
            f"传统 RFM 会将自然退出误判为流失，浪费约 {natural_pct:.0%} 的干预资源。"
        ),
    }


# ══════════════════════════════════════════════════════
# 诊断模块：因果归因
# ══════════════════════════════════════════════════════

def run_causal_module(voc_result: Dict, price_result: Dict,
                      age_result: Dict) -> Dict[str, Any]:
    """因果归因（Skill-Causal-Churn-Retention-Attribution）"""
    factors = {}

    if voc_result.get("rating_drop", 0) > 0.5:
        voc_contribution = min(0.6, voc_result["rating_drop"] / 2)
        factors["差评质量退化"] = round(voc_contribution, 2)

    if price_result.get("premium_pct", 0) > 0.1:
        price_contribution = min(0.5, price_result["premium_pct"] * 2)
        factors["竞品价格战"] = round(price_contribution, 2)

    if age_result.get("applicable") and age_result.get("natural_pct", 0) > 0.2:
        age_contribution = age_result["natural_pct"] * 0.8
        factors["月龄自然退出"] = round(age_contribution, 2)

    if not factors:
        factors["待进一步分析"] = 1.0

    total = sum(factors.values())
    normalized = {k: round(v / total, 2) for k, v in factors.items()}

    primary = max(normalized, key=normalized.get)

    return {
        "module": "因果归因",
        "factors": normalized,
        "primary_driver": primary,
        "conclusion": (
            f"主要驱动因素：{primary}（贡献 {normalized[primary]:.0%}）。"
            f"完整因素分解：" + "，".join(f"{k} {v:.0%}" for k, v in normalized.items())
        ),
    }


# ══════════════════════════════════════════════════════
# 报告生成器
# ══════════════════════════════════════════════════════

def generate_action_plan(problem_type: str, voc: Dict, price: Dict,
                          age: Dict, causal: Dict) -> list:
    """根据诊断结果生成优先级行动队列"""
    cfg = PROBLEM_CONFIG.get(problem_type, PROBLEM_CONFIG["repurchase_drop"])
    actions = []

    # VOC 紧急处置
    if voc.get("rating_drop", 0) > 0.5:
        actions.append({
            "day": "Day 1-2", "priority": "🔴 紧急",
            "skill": "Skill-Review-Pain-Point-Mining",
            "action": f"聚类差评主要痛点：{', '.join(p for p,_ in voc.get('top_pains',[])[:3])}",
            "url": f"{SKILL_BASE_URL}/Skill-Review-Pain-Point-Mining.html",
        })
        if voc.get("critical_pct", 0) > 0.2:
            actions.append({
                "day": "Day 1-2", "priority": "🔴 紧急",
                "skill": "Skill-SCRABLE-Review-Response-Generation",
                "action": f"对 {int(voc.get('critical_pct',0) * 50)} 条安全类差评逐一专业回复",
                "url": f"{SKILL_BASE_URL}/Skill-SCRABLE-Review-Response-Generation.html",
            })
        actions.append({
            "day": "Day 2-4", "priority": "🟠 高优先",
            "skill": "Skill-Review-Defense-Vine-Optimizer",
            "action": "申请 Amazon Vine 计划补充高质量评价，目标2周内新增20条★4以上",
            "url": f"{SKILL_BASE_URL}/Skill-Review-Defense-Vine-Optimizer.html",
        })

    # 竞品价格处置
    if price.get("premium_pct", 0) > 0.1:
        actions.append({
            "day": "Day 3-5", "priority": "🟠 高优先",
            "skill": "Skill-Listing-AI-Copywriting",
            "action": "重写 Listing 强化差异化卖点（安全认证/质保），与低价竞品拉开认知距离",
            "url": f"{SKILL_BASE_URL}/Skill-Listing-AI-Copywriting.html",
        })

    # 月龄时钟：挽回 vs 转品类
    if age.get("applicable") and age.get("real_churn", 0) > 0:
        actions.append({
            "day": "Day 5-7", "priority": "🟡 中优先",
            "skill": "Skill-Repurchase-Trigger-Timing-Model",
            "action": f"对 {age.get('real_churn')} 名真实流失用户（婴儿<7月龄）发定向复购优惠",
            "url": f"{SKILL_BASE_URL}/Skill-Repurchase-Trigger-Timing-Model.html",
        })
    if age.get("applicable") and age.get("natural_exit", 0) > 0:
        actions.append({
            "day": "Day 7+", "priority": "🟡 中优先",
            "skill": "Skill-Baby-Age-Aware-Recommendation",
            "action": f"对 {age.get('natural_exit')} 名自然退出用户推荐下一品类（辅食工具/幼儿零食）",
            "url": f"{SKILL_BASE_URL}/Skill-Baby-Age-Aware-Recommendation.html",
        })

    return actions


def print_report(sku: str, channel: str, problem_type: str,
                 voc: Dict, price: Dict, age: Dict, causal: Dict,
                 actions: list) -> None:
    """打印完整诊断报告"""
    cfg = PROBLEM_CONFIG.get(problem_type, PROBLEM_CONFIG["repurchase_drop"])
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "═" * 68)
    print(f"  paper2skills 业务诊断报告  v{VERSION}")
    print("═" * 68)
    print(f"  SKU：{sku}    渠道：{channel.upper()}    问题：{cfg['name']}")
    print(f"  诊断时间：{ts}")
    print("═" * 68)

    # ── 诊断结论汇总
    print("\n┌─ 综合结论 " + "─" * 55)
    severities = [voc.get("severity",""), price.get("severity","")]
    overall = "🔴 多因素叠加" if severities.count("🔴 严重") >= 2 else (
              "🟠 需要关注" if "🔴 严重" in severities else "🟡 轻微异常")
    print(f"│  整体严重程度：{overall}")
    print(f"│  主要驱动：{causal.get('primary_driver','待分析')} "
          f"（{causal.get('factors',{}).get(causal.get('primary_driver',''),0):.0%}）")
    if age.get("applicable") and age.get("natural_pct",0) > 0.2:
        print(f"│  ⚡ VOID 洞察：{age.get('natural_pct',0):.0%} 的'流失'是月龄自然退出，非真实流失")
    print("└" + "─" * 66)

    # ── 层1：诊断根因
    print("\n【诊断层】根因分析")
    print("─" * 55)

    # VOC
    print(f"\n  📊 {voc['module']}  {voc.get('data_source','')}")
    print(f"     评分：★{voc.get('avg_before','-')} → ★{voc.get('avg_after','-')} "
          f"（下降 {voc.get('rating_drop',0):.2f}）  严重程度：{voc.get('severity','')}")
    if voc.get("top_pains"):
        print(f"     差评高频词：{' | '.join(p for p,_ in voc['top_pains'][:4])}")
    print(f"     → {voc.get('conclusion','')}")

    # 竞品
    print(f"\n  💰 {price['module']}  {price.get('data_source','')}")
    print(f"     我方价格：${price.get('own_price',0):.2f}  "
          f"市场最低竞品：${price.get('cheapest_comp',0):.2f}  "
          f"溢价：{price.get('premium_pct',0):.0%}  {price.get('severity','')}")
    for c in price.get('competitors', []):
        if c.get('action') != '✅ 价格稳定':
            print(f"     {c['comp']}: ${c['before']} → ${c['after']} ({c['delta']:+.2f})  {c['action']}")
    print(f"     → {price.get('conclusion','')}")

    # 月龄时钟
    if age.get("applicable"):
        print(f"\n  🍼 {age['module']}  {age.get('data_source','')}  ⚡ VOID框架")
        print(f"     流失用户 {age.get('total_churned',0)} 人：")
        print(f"       🔵 自然退出（月龄≥7月）：{age.get('natural_exit',0)} 人 "
              f"({age.get('natural_pct',0):.0%}) → 不应干预")
        print(f"       🔴 真实流失（月龄<7月）：{age.get('real_churn',0)} 人 "
              f"({age.get('real_pct',0):.0%}) → 需要干预")
        print(f"     → {age.get('conclusion','')}")

    # 因果归因
    print(f"\n  🔍 {causal['module']}")
    factors_str = "  |  ".join(f"{k}: {v:.0%}" for k, v in causal.get('factors', {}).items())
    print(f"     驱动因素：{factors_str}")
    print(f"     → {causal.get('conclusion','')}")

    # ── 层2：处置行动
    print("\n\n【处置层】立即执行行动队列")
    print("─" * 55)
    for a in actions:
        print(f"\n  {a['day']}  {a['priority']}")
        print(f"  → {a['skill']}")
        print(f"     {a['action']}")
        print(f"     📎 {a['url']}")

    # ── 层3：预防机制
    print("\n\n【预防层】长效机制建议")
    print("─" * 55)
    prev_skills = PROBLEM_CONFIG.get(problem_type, {}).get("prevention_skills", [])
    for sk in prev_skills:
        print(f"  • {sk}")
        print(f"    📎 {SKILL_BASE_URL}/{sk}.html")

    if age.get("applicable"):
        print(f"\n  ⚡ 月龄感知预警（VOID框架专属）：")
        print(f"  • Skill-Baby-Age-Clock-RFM-Enhancement")
        print(f"    → 自动区分自然退出 vs 真实流失，节省约 {age.get('natural_pct',0):.0%} 无效干预")

    # ── 参考链接
    print("\n\n【参考链接】相关 Skill 卡片")
    print("─" * 55)
    all_skills = (PROBLEM_CONFIG.get(problem_type, {}).get("primary_skills", []) +
                  PROBLEM_CONFIG.get(problem_type, {}).get("treatment_skills", []))
    for sk in all_skills[:8]:
        print(f"  {sk}")
        print(f"  {SKILL_BASE_URL}/{sk}.html")

    print("\n" + "═" * 68)
    print("  诊断完成。如需深入分析，请访问：")
    print("  https://skills.lute-tlz-dddd.top/diagnostic.html")
    print("═" * 68)


# ══════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════

def interactive_mode():
    """交互式诊断向导"""
    print("\n" + "=" * 55)
    print("  paper2skills 业务诊断 SOP  v" + VERSION)
    print("=" * 55)

    sku = input("\n  SKU 名称（如：暖奶器、奶粉、推车）：").strip() or "未知SKU"
    channel = input("  渠道（amazon/tiktok/dtc）[默认 amazon]：").strip() or "amazon"

    print("\n  问题类型：")
    for i, (k, v) in enumerate(PROBLEM_CONFIG.items(), 1):
        print(f"    {i}. {v['name']}")
    choice = input("  选择（输入数字，默认 1）：").strip() or "1"
    try:
        problem_type = list(PROBLEM_CONFIG.keys())[int(choice) - 1]
    except (ValueError, IndexError):
        problem_type = "repurchase_drop"

    print(f"\n  开始诊断：{sku} | {channel.upper()} | {PROBLEM_CONFIG[problem_type]['name']}")
    print("  使用模拟数据（如有真实数据请使用参数模式）")

    return sku, channel, problem_type, None, None, None


def main():
    parser = argparse.ArgumentParser(description="paper2skills 通用业务诊断 SOP")
    parser.add_argument("--sku",      default=None, help="SKU 名称")
    parser.add_argument("--channel",  default="amazon", help="渠道")
    parser.add_argument("--problem",  default="repurchase_drop",
                        choices=list(PROBLEM_CONFIG.keys()), help="问题类型")
    parser.add_argument("--reviews",  default=None, help="评价数据 CSV 路径")
    parser.add_argument("--prices",   default=None, help="竞品价格 CSV 路径")
    parser.add_argument("--users",    default=None, help="用户月龄数据 CSV 路径")
    args = parser.parse_args()

    if args.sku is None:
        sku, channel, problem_type, reviews_path, prices_path, users_path = interactive_mode()
    else:
        sku          = args.sku
        channel      = args.channel
        problem_type = args.problem
        reviews_path = args.reviews
        prices_path  = args.prices
        users_path   = args.users

    reviews_df = pd.read_csv(reviews_path) if reviews_path and os.path.exists(reviews_path) else None
    prices_df  = pd.read_csv(prices_path)  if prices_path  and os.path.exists(prices_path)  else None
    users_df   = pd.read_csv(users_path)   if users_path   and os.path.exists(users_path)   else None

    print("\n  运行诊断模块...")
    voc    = run_voc_module(reviews_df, sku)
    price  = run_competitor_price_module(prices_df, sku)
    age    = run_baby_age_clock_module(users_df, sku)
    causal = run_causal_module(voc, price, age)
    actions = generate_action_plan(problem_type, voc, price, age, causal)

    print_report(sku, channel, problem_type, voc, price, age, causal, actions)


if __name__ == "__main__":
    main()
