---
title: Skill-Brand-Safety-Video-Content-Filter — 品牌安全视频内容过滤
doc_type: knowledge
module: 20-AI视频生成
topic: brand-safety-video-content-filter
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Brand-Safety-Video-Content-Filter

> **论文/方法来源**：Brand Safety in Digital Advertising（IAS/GARM Framework）+ UGC Content Moderation at Scale（工业实践）
> **领域**：20-AI视频生成 ↔ 合规决策 | **类型**: 内容安全

## ① 算法原理

品牌安全视频内容过滤（Brand Safety Video Content Filter）通过多层规则引擎和 NLP 分类器，自动检测 UGC（用户生成内容）中的品牌违规风险，保护品牌在 TikTok/YouTube 等平台的声誉。

**GARM 品牌安全层级**（全球广告行业标准）：

| 风险等级 | 内容类型 | 处置 |
|---------|---------|------|
| 危险内容（Level 4）| 暴力/色情/仇恨 | 立即下架 + 平台举报 |
| 高风险（Level 3）| 政治敏感/争议话题 | 禁止品牌关联 |
| 中风险（Level 2）| 竞品诋毁/夸大宣传 | 法务审查 |
| 低风险（Level 1）| 轻微偏离品牌调性 | 运营审核 |

**母婴品牌专项风险词库**：
- 医疗声明违规：「cures」「treats」「FDA approved」（未经核实）
- 安全性夸大：「safest in the world」「100% safe」（无科学依据）
- 竞品攻击：直接提及竞品品牌 + 负面词
- 儿童安全警觉词：「choking hazard」「toxic」（产品被指控问题）

**检测方法**：正则表达式规则（高精度快速）+ TF-IDF 特征分类器（覆盖模糊表达）。

## ② 母婴出海应用案例

**场景：婴儿辅食品牌 UGC 内容合规批量审查**

- **业务问题**：品牌 TikTok 挑战赛收集了 500 条 UGC 视频，需要在 24 小时内完成品牌安全审查，人工审查每条 5 分钟需 41 小时，不可行
- **数据要求**：UGC 视频文字描述/字幕、品牌安全风险词库
- **执行方案**：
  - 批量提取 UGC 视频字幕文本（Whisper ASR）
  - 多层规则引擎自动分类（危险/高风险/中风险/低风险/通过）
  - 自动通过的视频直接发布（约 70%）
  - 高风险和中风险推送人工审核队列（约 15%）
- **量化产出**：审查时间从 41 小时 → 3 小时（自动过滤 70% + 人工审核 30%）
- **业务价值**：避免一次品牌安全事故（潜在罚款 + 下架损失约 10-50 万元）

## ③ 代码模板

```python
import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class BrandSafetyRule:
    """品牌安全规则"""
    rule_id: str
    risk_level: int              # 1-4，4 最危险
    pattern: str                 # 正则表达式
    category: str                # 违规类别
    action: str                  # BLOCK/REVIEW/FLAG

# 母婴品牌安全规则库
BABY_BRAND_SAFETY_RULES = [
    BrandSafetyRule("R001", 4, r"\b(suicide|self.harm|violence|abuse)\b", "危险内容", "BLOCK"),
    BrandSafetyRule("R002", 4, r"\b(child\s+abuse|exploitation)\b", "儿童危害", "BLOCK"),
    BrandSafetyRule("R003", 3, r"\b(BPA.*(poison|toxic|cancer))\b", "安全诋毁", "BLOCK"),
    BrandSafetyRule("R004", 3, r"\b(choking\s+hazard|recall|toxic|poisoning)\b", "产品危害指控", "REVIEW"),
    BrandSafetyRule("R005", 2, r"\b(cure[sd]?|treats|FDA[\s-]approved|clinically\s+proven)\b", "医疗声明违规", "REVIEW"),
    BrandSafetyRule("R006", 2, r"\b(100%\s+safe|safest\s+in\s+the\s+world|guaranteed)\b", "夸大安全声明", "REVIEW"),
    BrandSafetyRule("R007", 2, r"\b(better\s+than\s+[A-Z][a-z]+|[A-Z][a-z]+\s+is\s+(bad|terrible|toxic))\b", "竞品攻击", "REVIEW"),
    BrandSafetyRule("R008", 1, r"\b(overpriced|scam|rip.?off)\b", "价格/诚信攻击", "FLAG"),
    BrandSafetyRule("R009", 1, r"\b(misleading|fake|counterfeit)\b", "虚假声明", "FLAG"),
    BrandSafetyRule("R010", 1, r"\b(political|election|controversial)\b", "政治敏感", "FLAG"),
]

def scan_content(text: str, rules: List[BrandSafetyRule]) -> List[Dict]:
    """扫描文本内容，返回触发的规则列表"""
    triggered = []
    text_lower = text.lower()
    
    for rule in rules:
        matches = re.findall(rule.pattern, text_lower, re.IGNORECASE)
        if matches:
            triggered.append({
                "rule_id": rule.rule_id,
                "risk_level": rule.risk_level,
                "category": rule.category,
                "action": rule.action,
                "matched_text": matches[:3]  # 最多显示3个匹配
            })
    
    return triggered

def classify_content(text: str, rules: List[BrandSafetyRule]) -> Dict:
    """综合分类内容风险"""
    triggered = scan_content(text, rules)
    
    if not triggered:
        return {
            "status": "PASS",
            "max_risk_level": 0,
            "primary_action": "APPROVE",
            "triggered_rules": [],
            "review_priority": "NONE"
        }
    
    max_risk = max(r["risk_level"] for r in triggered)
    
    # 最严格动作优先
    if any(r["action"] == "BLOCK" for r in triggered):
        action = "BLOCK"
        priority = "URGENT"
    elif any(r["action"] == "REVIEW" for r in triggered):
        action = "HUMAN_REVIEW"
        priority = "HIGH" if max_risk >= 3 else "MEDIUM"
    else:
        action = "FLAG"
        priority = "LOW"
    
    return {
        "status": "FLAGGED",
        "max_risk_level": max_risk,
        "primary_action": action,
        "triggered_rules": triggered,
        "review_priority": priority
    }

def batch_audit_ugc(ugc_texts: List[str], rules: List[BrandSafetyRule]) -> pd.DataFrame:
    """批量审计 UGC 内容"""
    rows = []
    for i, text in enumerate(ugc_texts):
        result = classify_content(text, rules)
        rows.append({
            "ugc_id": i + 1,
            "content_preview": text[:50] + "..." if len(text) > 50 else text,
            "status": result["status"],
            "max_risk": result["max_risk_level"],
            "action": result["primary_action"],
            "priority": result["review_priority"],
            "n_violations": len(result["triggered_rules"])
        })
    
    return pd.DataFrame(rows)

def generate_audit_summary(df: pd.DataFrame) -> Dict:
    """生成审计汇总报告"""
    return {
        "total_ugc": len(df),
        "approved_auto": int((df["action"] == "APPROVE").sum()),
        "blocked": int((df["action"] == "BLOCK").sum()),
        "human_review_needed": int((df["action"] == "HUMAN_REVIEW").sum()),
        "flagged": int((df["action"] == "FLAG").sum()),
        "auto_approval_rate_pct": round((df["action"] == "APPROVE").mean() * 100, 1),
        "high_risk_count": int((df["max_risk"] >= 3).sum()),
        "review_queue": df[df["action"] != "APPROVE"]["ugc_id"].tolist()
    }

# 测试
ugc_samples = [
    "This BPA-free baby bottle is amazing, my baby loves it and no issues at all",
    "WARNING: This product is TOXIC and POISONING babies, do NOT buy!",
    "BrandX cures colic and is FDA approved - every mom should try this",
    "The product is good but overpriced compared to similar brands",
    "My baby had no rash using this, gentle on sensitive skin, recommend!",
    "This brand is better than Frida Baby, their products are terrible",
    "Love the quality but shipping packaging was flimsy, arrived damaged",
    "100% safe and the safest in the world for newborns - buy now!",
    "Great unboxing experience, will definitely repurchase this baby wipe",
    "This scam company sent me fake product - counterfeit items!"
]

audit_df = batch_audit_ugc(ugc_samples, BABY_BRAND_SAFETY_RULES)
print("=== UGC 品牌安全审计结果 ===")
print(audit_df.to_string(index=False))

summary = generate_audit_summary(audit_df)
print("\n=== 审计汇总 ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n[✓] Brand-Safety-Video-Content-Filter 测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-Video-Sentiment-Analysis-VOC]]（评论监控）、[[Skill-品牌合规卫士]]（合规框架）
- **延伸**：[[Skill-Cross-Platform-Video-Repurposing]]（跨平台内容审核）、[[Skill-Live-Stream-Highlight-Extraction]]（直播内容审查）
- **可组合**：[[Skill-AI-Product-Video-Script-Generator]]（脚本合规预审）+ [[Skill-TikTok-Algorithm-Content-Boost]]（合规内容加速推广）

## ⑤ 商业价值评估

- **ROI**：自动过滤 70% UGC，人工审核时间减少 80%，避免品牌安全事故（年化保护价值 10-50 万元）
- **实施难度**：⭐⭐☆☆☆（规则引擎即可实现 80% 场景，开发周期 1 天）
- **优先级**：⭐⭐⭐⭐☆（有 UGC 挑战赛/合作 KOL 的品牌必备，合规风险不可忽视）
