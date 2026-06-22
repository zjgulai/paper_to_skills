---
title: New-Market-Entry-Readiness-Gate — 新市场进入评分超阈值自动生成进入Checklist并分配任务
doc_type: knowledge
module: 06-增长模型
topic: new-market-entry-readiness-gate
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: New-Market-Entry-Readiness-Gate

> **配对分析层**：[[Skill-Market-Opportunity-Scoring]]
> **决策类型**: 自动触发型 | **触发条件**: 新市场进入评分 > 阈值（默认70分） | **执行动作**: 自动生成市场进入 Checklist 并分配责任人任务

## ① 算法原理

核心是「多维评分聚合 + 阈值门控 + 差异化 Checklist 生成 + 任务分配」：

1. **评分维度（满分100）**：
   - 市场规模潜力（30分）：目标市场 GMV 规模、增速、竞争密度
   - 监管合规就绪度（25分）：产品认证状态（CE/FCC/各国标准）
   - 物流履约可行性（20分）：目标国仓储网络、清关成功率历史数据
   - 本地化准备度（15分）：语言/货币/支付方式支持
   - 财务可行性（10分）：目标市场预估毛利率
2. **门控判断**：综合评分 ≥ 70 且所有强制项（合规 ≥ 20分）通过 → 触发进入流程。
3. **Checklist 生成**：根据每个维度的得分和缺口，自动生成对应的准备任务（低分项生成更多任务）。
4. **任务分配**：按任务类型自动分配给对应团队（合规→法务团队、物流→供应链团队等）。

## ② 母婴出海应用案例

**场景：从 Amazon US 扩展进入 Amazon DE（德国市场）**
- 评分结果：综合评分 76 分（市场规模 24/30，合规就绪度 18/25，物流 16/20，本地化 12/15，财务 6/10）
- 触发动作：
  - 自动生成 24 项 Checklist（重点：DE 市场 WEEE 注册、VAT 注册、德语 Listing 翻译）
  - 分配任务：法务团队（4项，合规）、供应链团队（6项，德国仓储）、运营团队（8项，Listing 本地化）
  - 设置 30 天进入准备 Deadline，每周自动检查进度
- 业务价值：系统化推进避免遗漏关键合规项，DE 市场首月 GMV $45,000，6 个月后 $120,000/月

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# 评分维度配置
SCORING_DIMENSIONS = [
    {"name": "market_potential", "label": "市场规模潜力", "max_score": 30, "mandatory_min": 15,
     "tasks_template": ["目标市场TAM调研报告", "竞品密度分析（Top 20 ASIN）", "类目增速验证（近12月）"]},
    {"name": "compliance_readiness", "label": "监管合规就绪度", "max_score": 25, "mandatory_min": 20,
     "tasks_template": ["产品认证清单核查（CE/FCC/本国标准）", "VAT/GST注册评估", "本地强制标注要求确认", "WEEE/EPR合规评估"]},
    {"name": "logistics_feasibility", "label": "物流履约可行性", "max_score": 20, "mandatory_min": 10,
     "tasks_template": ["目标国FBA仓容量评估", "关税税率确认", "清关流程文件准备", "本地退货地址设置", "平均配送时效调研"]},
    {"name": "localization_readiness", "label": "本地化准备度", "max_score": 15, "mandatory_min": 8,
     "tasks_template": ["Listing本地语言翻译（标题/五点/描述）", "本地支付方式支持确认", "货币定价策略制定"]},
    {"name": "financial_viability", "label": "财务可行性", "max_score": 10, "mandatory_min": 5,
     "tasks_template": ["目标市场定价空间分析", "毛利率预测模型（含关税/物流/VAT）"]},
]

TASK_ASSIGNEE = {
    "market_potential": "market_team",
    "compliance_readiness": "legal_team",
    "logistics_feasibility": "supply_chain_team",
    "localization_readiness": "ops_team",
    "financial_viability": "finance_team"
}

def new_market_entry_readiness_gate(
    market_assessments: List[Dict],
    now: Optional[datetime] = None,
    entry_score_threshold: float = 70.0,
    checklist_deadline_days: int = 30
) -> Dict:
    """
    新市场进入就绪门控
    
    参数:
        market_assessments: [{
            "market_id": str, "market_name": str,
            "scores": {
                "market_potential": float (0-30),
                "compliance_readiness": float (0-25),
                "logistics_feasibility": float (0-20),
                "localization_readiness": float (0-15),
                "financial_viability": float (0-10)
            }
        }]
    
    返回:
        {"decisions": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    decisions = []
    
    for assessment in market_assessments:
        mid = assessment["market_id"]
        mname = assessment["market_name"]
        scores = assessment.get("scores", {})
        
        # 计算总分
        total_score = sum(scores.get(dim["name"], 0) for dim in SCORING_DIMENSIONS)
        
        # 检查强制项
        mandatory_failures = []
        for dim in SCORING_DIMENSIONS:
            score = scores.get(dim["name"], 0)
            if score < dim["mandatory_min"]:
                mandatory_failures.append({
                    "dimension": dim["label"],
                    "score": score,
                    "required_min": dim["mandatory_min"],
                    "gap": dim["mandatory_min"] - score
                })
        
        # 门控判断
        passes_threshold = total_score >= entry_score_threshold
        passes_mandatory = len(mandatory_failures) == 0
        
        if not passes_threshold or not passes_mandatory:
            decision = {
                "market_id": mid,
                "market_name": mname,
                "action": "ENTRY_BLOCKED",
                "total_score": round(total_score, 1),
                "threshold": entry_score_threshold,
                "passes_threshold": passes_threshold,
                "mandatory_failures": mandatory_failures,
                "reason": f"评分{total_score:.0f}分（{'未达阈值' if not passes_threshold else ''}）{'+ 强制项未通过' if mandatory_failures else ''}",
                "blockers": [f"{mf['dimension']}不足（{mf['score']:.0f}/{mf['required_min']}）" for mf in mandatory_failures]
            }
        else:
            # 生成 Checklist
            checklist = []
            for dim in SCORING_DIMENSIONS:
                score = scores.get(dim["name"], 0)
                gap_ratio = 1 - score / dim["max_score"]
                # 低分维度生成完整任务列表，高分维度生成最少任务
                tasks_count = max(1, int(len(dim["tasks_template"]) * gap_ratio + 0.5))
                assigned_tasks = dim["tasks_template"][:tasks_count]
                for task in assigned_tasks:
                    deadline_days = 7 if gap_ratio > 0.4 else 14
                    checklist.append({
                        "task": task,
                        "dimension": dim["label"],
                        "assignee": TASK_ASSIGNEE[dim["name"]],
                        "priority": "HIGH" if gap_ratio > 0.4 else "MEDIUM",
                        "deadline": (now + timedelta(days=deadline_days)).strftime("%Y-%m-%d"),
                        "score_gap": round(gap_ratio * 100, 0)
                    })
            
            # 按团队分组任务
            team_tasks = {}
            for item in checklist:
                team = item["assignee"]
                team_tasks[team] = team_tasks.get(team, 0) + 1
            
            decision = {
                "market_id": mid,
                "market_name": mname,
                "action": "ENTRY_APPROVED",
                "total_score": round(total_score, 1),
                "threshold": entry_score_threshold,
                "score_breakdown": {dim["name"]: round(scores.get(dim["name"], 0), 1) for dim in SCORING_DIMENSIONS},
                "checklist": checklist,
                "checklist_count": len(checklist),
                "team_task_distribution": team_tasks,
                "overall_deadline": (now + timedelta(days=checklist_deadline_days)).strftime("%Y-%m-%d"),
                "entry_recommendation": f"✅ 评分{total_score:.0f}分，建议30天内完成{len(checklist)}项准备工作后进入{mname}市场"
            }
        
        decisions.append(decision)
    
    return {
        "total_markets": len(market_assessments),
        "approved": sum(1 for d in decisions if d.get("action") == "ENTRY_APPROVED"),
        "blocked": sum(1 for d in decisions if d.get("action") == "ENTRY_BLOCKED"),
        "decisions": decisions
    }


# 测试
assessments = [
    {
        "market_id": "DE", "market_name": "德国市场(Amazon DE)",
        "scores": {
            "market_potential": 24,
            "compliance_readiness": 18,  # 低于强制最低20分 → 应被阻断
            "logistics_feasibility": 16,
            "localization_readiness": 12,
            "financial_viability": 6
        }
    },
    {
        "market_id": "JP", "market_name": "日本市场(Amazon JP)",
        "scores": {
            "market_potential": 26,
            "compliance_readiness": 22,
            "logistics_feasibility": 17,
            "localization_readiness": 11,
            "financial_viability": 7
        }
    },
    {
        "market_id": "AU", "market_name": "澳大利亚市场(Amazon AU)",
        "scores": {
            "market_potential": 18,
            "compliance_readiness": 20,
            "logistics_feasibility": 12,
            "localization_readiness": 8,
            "financial_viability": 4  # 财务低于强制项5分 → 阻断
        }
    },
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = new_market_entry_readiness_gate(assessments, now=now)

assert result["total_markets"] == 3
dec_map = {d["market_id"]: d for d in result["decisions"]}
assert dec_map["DE"]["action"] == "ENTRY_BLOCKED"  # 合规未达强制最低
assert dec_map["JP"]["action"] == "ENTRY_APPROVED"
assert dec_map["AU"]["action"] == "ENTRY_BLOCKED"  # 财务未达强制最低

jp_decision = dec_map["JP"]
assert jp_decision["checklist_count"] > 0
assert "legal_team" in jp_decision["team_task_distribution"]

print("[✓] New Market Entry Readiness Gate 测试通过")
print(f"  评估市场: {result['total_markets']}，批准进入: {result['approved']}，阻断: {result['blocked']}")
if result["approved"] > 0:
    approved = next(d for d in result["decisions"] if d["action"] == "ENTRY_APPROVED")
    print(f"  {approved['market_name']} 生成 {approved['checklist_count']} 项任务")
    print(f"  团队分配: {approved['team_task_distribution']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Market-Opportunity-Scoring]]（提供各维度评分数据）
- **延伸（extends）**：[[Skill-Compliance-Violation-Auto-Escalation]]（市场进入后合规风险实时监控）
- **可组合（combinable）**：[[Skill-Pre-Launch-Compliance-Gate]]（进入审批通过后再走产品上架合规预检）

## ⑤ 商业价值评估
- **ROI量化**：系统化进入避免合规遗漏，DE 市场 6 个月 GMV $120,000/月；避免一次合规违规罚款约 $30,000
- **实施难度**：⭐⭐⭐☆☆（需评分数据输入接口 + 任务管理系统对接）
- **优先级**：⭐⭐⭐⭐☆（多市场扩张是增长的核心路径，但遗漏合规风险极高）
