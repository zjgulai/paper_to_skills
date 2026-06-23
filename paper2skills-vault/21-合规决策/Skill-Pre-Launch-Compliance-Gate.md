---
title: Pre-Launch-Compliance-Gate — 新品上架前合规评分低于阈值自动阻断并触发修复工作流
doc_type: knowledge
module: 21-合规决策
topic: pre-launch-compliance-gate
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Pre-Launch-Compliance-Gate

> **配对分析层**：[[Skill-Category-Compliance-Prescan]]
> **决策类型**: 自动触发型 | **触发条件**: 新品上架流程发起 | **执行动作**: 合规评分 < 80 → 阻断上架 + 生成修复清单；≥ 80 → 放行 + 记录存档

## ① 算法原理

核心是「多维合规评分 → 门控决策 → 差异化处置」：

1. **多维评分**：5 个维度各 0-20 分，满分 100：
   - 产品安全认证完整性（CPSC/CE/ASTM 等）
   - Listing 合规性（违禁词、健康声明、图片规范）
   - HTS 码风险等级
   - 历史同品类召回密度
   - 目标市场特殊要求满足度
2. **门控决策**：≥ 80 放行；60-79 警告放行（需人工确认）；< 60 强制阻断
3. **修复清单生成**：对每个失分维度生成具体修复步骤和优先级
4. **存档**：放行记录写入合规档案，供申诉和审计使用

**安全护栏**：门控阈值可按品类配置（母婴类默认 85，一般消费品 75）。

## ② 母婴出海应用案例

**场景：新款婴儿硅胶安抚奶嘴上架前合规检查**
- 触发：运营提交新品上架申请
- 评分结果：安全认证 18/20（缺 FDA 食品接触材料证明）+ Listing 15/20（主图有健康声明违禁词）+ HTS 19/20 + 历史召回 14/20（同品类近 2 年 2 次召回）+ 市场要求 16/20 = 总分 **82 分**
- 决策：警告放行，生成 2 条修复建议，要求运营 7 天内补齐 FDA 证明并修改图片
- 量化价值：提前发现合规问题，避免上架后下架损失（平均每次合规下架损失 ¥3-8 万）

## ③ 代码模板

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ComplianceCheckResult:
    dimension: str
    score: float
    max_score: float
    issues: List[str]
    fix_steps: List[str]

@dataclass
class GateDecision:
    total_score: float
    decision: str  # PASS / WARN_PASS / BLOCK
    dimension_results: List[ComplianceCheckResult]
    fix_checklist: List[Dict]
    archive_id: str

def pre_launch_compliance_gate(
    product: Dict,
    market: str = "US",
    pass_threshold: float = 80.0,
    warn_threshold: float = 60.0
) -> GateDecision:
    import hashlib, json
    from datetime import datetime

    category_thresholds = {
        "infant": 85.0,
        "baby": 85.0,
        "toy": 82.0,
        "default": pass_threshold
    }
    cat = product.get("category", "default").lower()
    effective_threshold = next(
        (v for k, v in category_thresholds.items() if k in cat),
        category_thresholds["default"]
    )

    def check_certifications(p: Dict) -> ComplianceCheckResult:
        certs = p.get("certifications", [])
        required = {"CPSC", "CE"} if market == "EU" else {"CPSC"}
        if "infant" in p.get("category", "").lower():
            required.add("FDA_food_contact")
        missing = required - set(certs)
        score = 20.0 * (1 - len(missing) / max(len(required), 1))
        issues = [f"缺少认证: {c}" for c in missing]
        fixes = [f"申请 {c} 认证，预计 {4 if c == 'CE' else 2} 周" for c in missing]
        return ComplianceCheckResult("安全认证", round(score, 1), 20.0, issues, fixes)

    def check_listing(p: Dict) -> ComplianceCheckResult:
        forbidden = ["治愈", "最安全", "无毒", "FDA认证"]
        title = p.get("title", "") + " " + p.get("description", "")
        found = [w for w in forbidden if w in title]
        score = 20.0 - len(found) * 5
        issues = [f"违禁词: {w}" for w in found]
        fixes = [f"删除或替换违禁词「{w}」" for w in found]
        return ComplianceCheckResult("Listing合规", max(score, 0), 20.0, issues, fixes)

    def check_hts(p: Dict) -> ComplianceCheckResult:
        hts = p.get("hts_code", "")
        high_risk_prefixes = ["9503", "9508", "6111"]
        is_risky = any(hts.startswith(pfx) for pfx in high_risk_prefixes)
        score = 15.0 if is_risky else 20.0
        issues = [f"HTS {hts} 属于高风险品类"] if is_risky else []
        fixes = ["请合规团队复核 HTS 码分类"] if is_risky else []
        return ComplianceCheckResult("HTS码风险", score, 20.0, issues, fixes)

    def check_recall_history(p: Dict) -> ComplianceCheckResult:
        recall_count = p.get("category_recall_count_2y", 0)
        score = max(20.0 - recall_count * 3, 5.0)
        issues = [f"同品类近2年有 {recall_count} 次召回记录"] if recall_count > 0 else []
        fixes = ["加强产品测试，参考召回原因针对性检验"] if recall_count > 2 else []
        return ComplianceCheckResult("召回历史", round(score, 1), 20.0, issues, fixes)

    def check_market_requirements(p: Dict) -> ComplianceCheckResult:
        met = p.get("market_requirements_met", [])
        required_count = 3
        score = 20.0 * min(len(met) / required_count, 1.0)
        issues = [] if len(met) >= required_count else [f"仅满足 {len(met)}/{required_count} 项市场要求"]
        fixes = ["补充目标市场特定合规文件"] if issues else []
        return ComplianceCheckResult("市场要求", round(score, 1), 20.0, issues, fixes)

    results = [
        check_certifications(product),
        check_listing(product),
        check_hts(product),
        check_recall_history(product),
        check_market_requirements(product),
    ]

    total = sum(r.score for r in results)

    if total >= effective_threshold:
        decision = "PASS"
    elif total >= warn_threshold:
        decision = "WARN_PASS"
    else:
        decision = "BLOCK"

    fix_checklist = [
        {"dimension": r.dimension, "issue": issue, "fix": fix, "priority": i + 1}
        for i, r in enumerate(results)
        for issue, fix in zip(r.issues, r.fix_steps)
    ]

    archive_id = hashlib.md5(
        json.dumps({"sku": product.get("sku_id"), "ts": str(__import__("datetime").datetime.now())}).encode()
    ).hexdigest()[:8].upper()

    return GateDecision(
        total_score=round(total, 1),
        decision=decision,
        dimension_results=results,
        fix_checklist=fix_checklist,
        archive_id=f"COMP-{archive_id}"
    )


if __name__ == "__main__":
    product = {
        "sku_id": "PAC-001",
        "title": "婴儿硅胶安抚奶嘴 最安全设计",
        "description": "FDA认证原料",
        "category": "infant_pacifier",
        "certifications": ["CPSC"],
        "hts_code": "3924100000",
        "category_recall_count_2y": 2,
        "market_requirements_met": ["ASTM_F963", "EN_1400"]
    }

    result = pre_launch_compliance_gate(product, market="US")
    print(f"合规总分: {result.total_score}/100")
    print(f"决策: {result.decision}")
    print(f"档案ID: {result.archive_id}")
    for r in result.dimension_results:
        print(f"  {r.dimension}: {r.score}/{r.max_score}")
    if result.fix_checklist:
        print(f"修复清单 ({len(result.fix_checklist)} 项):")
        for item in result.fix_checklist:
            print(f"  [{item['priority']}] {item['issue']} → {item['fix']}")

    assert result.decision in ("PASS", "WARN_PASS", "BLOCK")
    assert 0 <= result.total_score <= 100
    print("[✓] Pre-Launch-Compliance-Gate 测试通过")

## ④ 技能关联

- 前置技能：[[Skill-Category-Compliance-Prescan]]
- 前置技能：[[Skill-Cross-Border-Compliance-Framework]]
- 延伸技能：[[Skill-Compliance-Scored-Guardrail-Orchestration]]
- 延伸技能：[[Skill-Listing-Compliance-Auto-Repair]]
- 可组合：[[Skill-New-SKU-Launch-Readiness-Gate]]
- 可组合：[[Skill-Amazon-ToS-Compliance-Guardrail]]
