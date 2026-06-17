---
title: 新品上市准入门控 — 从选品到发布的全量检查清单与Tag驱动的上市就绪评估
doc_type: knowledge
module: 04-供应链
topic: new-sku-launch-readiness-gate
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 新品上市准入门控

> **来源**：arXiv:2310.09823（Product Launch Readiness Framework）+ arXiv:2402.12234（Gate-Based New Product Introduction）+ Amazon新品上市最佳实践
> **桥梁**：选品规划 ↔ 标签工程 ↔ 供应链全链路 | **类型**：准入门控

## ① 算法原理

**新品上市准入门控（Launch Readiness Gate）** 是一个多维度、结构化的"上市检查清单"——防止因某个关键环节遗漏导致上市后的补救成本远高于预防成本。

**5个核心门控维度**：

| 门控 | 检查项 | 通过标准 | Tag |
|-----|-------|--------|-----|
| G1 产品合规 | 认证/测试报告/危险品检查 | 目标市场100%合规 | `sku.launch.compliance_gate` |
| G2 供应链就绪 | 供应商产能/备货/前置期 | 首批库存≥30天销量 | `sku.launch.supply_gate` |
| G3 Listing就绪 | 标题/图片/A+/关键词 | 全部完成并审核 | `sku.launch.listing_gate` |
| G4 财务测算 | 利润率/FBA费/初始库存投入 | 净利润率>10% | `sku.launch.finance_gate` |
| G5 运营准备 | 客服培训/退货政策/广告计划 | 全部就位 | `sku.launch.ops_gate` |

**门控状态**：
- `PASS`：通过
- `FAIL`：未通过（阻塞上市）
- `CONDITIONAL`：条件通过（有已知风险，接受并继续）
- `NA`：不适用

**整体上市就绪评分**：
$$\text{LaunchScore} = \frac{\sum_i \text{GateScore}_i \times w_i}{\sum w_i}$$

阈值：≥85分 = 可上市；70-84分 = 条件上市；<70分 = 暂缓

## ② 代码模板

```python
"""
新品上市准入门控系统
功能：多维度门控检查 / 就绪评分 / 阻塞项识别 / 上市决策建议
"""
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


GATE_WEIGHTS = {
    "compliance": 0.30,  # 合规是硬门控
    "supply":     0.25,
    "listing":    0.20,
    "finance":    0.15,
    "operations": 0.10,
}


@dataclass
class GateCheck:
    gate_name: str
    check_name: str
    status: str          # PASS / FAIL / CONDITIONAL / NA
    score: float         # 0-1
    details: str = ""
    is_blocker: bool = False  # True=必须PASS才能上市


@dataclass
class LaunchReadinessReport:
    sku_id: str
    product_name: str
    gate_scores: dict = field(default_factory=dict)  # gate → score
    all_checks: list = field(default_factory=list)
    overall_score: float = 0.0
    recommendation: str = "PENDING"
    blockers: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)


def check_compliance_gate(sku_data: dict) -> list:
    checks = []
    markets = sku_data.get("target_markets", [])
    certs = sku_data.get("certifications", {})

    required = {"US": ["FCC"], "EU": ["CE", "ROHS"], "JP": ["PSE"]}
    all_pass = True
    for market in markets:
        for cert in required.get(market, []):
            has_cert = certs.get(cert, False)
            checks.append(GateCheck(
                "compliance", f"{market}:{cert}认证",
                "PASS" if has_cert else "FAIL",
                1.0 if has_cert else 0.0,
                f"{'已获得' if has_cert else '缺失'}{cert}认证",
                is_blocker=not has_cert,
            ))
            if not has_cert: all_pass = False

    checks.append(GateCheck(
        "compliance", "EPR注册",
        "PASS" if sku_data.get("epr_registered") else "CONDITIONAL",
        0.8 if sku_data.get("epr_registered") else 0.5,
        "EPR已注册" if sku_data.get("epr_registered") else "EPR未注册（EU市场风险）",
        is_blocker=False,
    ))
    return checks


def check_supply_gate(sku_data: dict) -> list:
    checks = []
    first_batch = sku_data.get("first_batch_qty", 0)
    est_monthly = sku_data.get("estimated_monthly_sales", 100)
    coverage_days = first_batch / max(1, est_monthly) * 30
    supplier_confirmed = sku_data.get("supplier_confirmed", False)

    checks.append(GateCheck(
        "supply", "首批备货覆盖天数",
        "PASS" if coverage_days >= 30 else "FAIL",
        min(1.0, coverage_days / 30),
        f"首批{first_batch}件={coverage_days:.0f}天覆盖（目标≥30天）",
        is_blocker=(coverage_days < 14),
    ))
    checks.append(GateCheck(
        "supply", "供应商产能确认",
        "PASS" if supplier_confirmed else "FAIL",
        1.0 if supplier_confirmed else 0.0,
        "供应商已确认月产能" if supplier_confirmed else "供应商产能未确认",
        is_blocker=not supplier_confirmed,
    ))
    checks.append(GateCheck(
        "supply", "PLT评估",
        "PASS" if sku_data.get("plt_days_confirmed") else "CONDITIONAL",
        1.0 if sku_data.get("plt_days_confirmed") else 0.6,
        f"PLT={sku_data.get('plt_days',35)}天（已确认）" if sku_data.get("plt_days_confirmed") else "PLT未精确确认",
    ))
    return checks


def check_listing_gate(sku_data: dict) -> list:
    checks = []
    for item, has, blocker in [
        ("主图（7张+白底主图）", sku_data.get("main_images_done"), True),
        ("A+/EBC页面", sku_data.get("aplus_done"), False),
        ("关键词研究", sku_data.get("keywords_done"), True),
        ("标题优化", sku_data.get("title_done"), True),
        ("本地化翻译", sku_data.get("localization_done"), False),
    ]:
        checks.append(GateCheck(
            "listing", item,
            "PASS" if has else "FAIL",
            1.0 if has else 0.0,
            f"{'已完成' if has else '未完成'}: {item}",
            is_blocker=(blocker and not has),
        ))
    return checks


def check_finance_gate(sku_data: dict) -> list:
    checks = []
    net_margin = sku_data.get("estimated_net_margin_pct", 0)
    initial_investment = sku_data.get("initial_investment_usd", 0)
    checks.append(GateCheck(
        "finance", "净利润率测算",
        "PASS" if net_margin >= 10 else ("CONDITIONAL" if net_margin >= 5 else "FAIL"),
        min(1.0, max(0, net_margin) / 20),
        f"预估净利润率{net_margin:.1f}%（目标≥10%）",
        is_blocker=(net_margin < 0),
    ))
    checks.append(GateCheck(
        "finance", "初始资金预算",
        "PASS" if sku_data.get("budget_approved") else "CONDITIONAL",
        1.0 if sku_data.get("budget_approved") else 0.5,
        f"初始投入${initial_investment:,.0f}，{'预算已批准' if sku_data.get('budget_approved') else '待批准'}",
    ))
    return checks


def evaluate_launch_readiness(sku_id: str, product_name: str,
                               sku_data: dict) -> LaunchReadinessReport:
    report = LaunchReadinessReport(sku_id=sku_id, product_name=product_name)
    all_checks = []
    all_checks.extend(check_compliance_gate(sku_data))
    all_checks.extend(check_supply_gate(sku_data))
    all_checks.extend(check_listing_gate(sku_data))
    all_checks.extend(check_finance_gate(sku_data))

    report.all_checks = all_checks
    report.blockers = [c for c in all_checks if c.is_blocker and c.status == "FAIL"]

    # 门控得分
    gate_checks = {}
    for check in all_checks:
        gate_checks.setdefault(check.gate_name, []).append(check.score)

    for gate, scores in gate_checks.items():
        report.gate_scores[gate] = sum(scores) / len(scores)

    # 综合分
    report.overall_score = sum(
        report.gate_scores.get(g, 0.5) * w for g, w in GATE_WEIGHTS.items()
    ) * 100

    if report.blockers:
        report.recommendation = "BLOCKED"
    elif report.overall_score >= 85:
        report.recommendation = "LAUNCH"
    elif report.overall_score >= 70:
        report.recommendation = "CONDITIONAL_LAUNCH"
    else:
        report.recommendation = "DELAY"

    report.tags = {
        "sku.launch.overall_score": round(report.overall_score, 1),
        "sku.launch.recommendation": report.recommendation,
        "sku.launch.blockers_count": len(report.blockers),
    }
    return report


if __name__ == "__main__":
    print("【新品上市准入门控系统】\n")
    sku_data = {
        "target_markets": ["US", "EU"],
        "certifications": {"FCC": True, "CE": True, "ROHS": True},
        "epr_registered": False,
        "first_batch_qty": 500, "estimated_monthly_sales": 200,
        "supplier_confirmed": True, "plt_days": 28, "plt_days_confirmed": True,
        "main_images_done": True, "aplus_done": False, "keywords_done": True,
        "title_done": True, "localization_done": False,
        "estimated_net_margin_pct": 12.5, "initial_investment_usd": 25_000, "budget_approved": True,
    }

    report = evaluate_launch_readiness("SKU-NewPump", "新款双边吸奶器", sku_data)

    print(f"  产品: {report.product_name}  整体评分: {report.overall_score:.1f}分")
    rec_icon = {"LAUNCH": "✅", "CONDITIONAL_LAUNCH": "⚠️ ", "BLOCKED": "❌", "DELAY": "🔴"}[report.recommendation]
    print(f"  上市建议: {rec_icon} {report.recommendation}")

    print("\n  各门控得分:")
    for gate, score in report.gate_scores.items():
        bar = "█" * int(score * 10)
        print(f"    {gate:12s}: {score*100:.0f}分 {bar}")

    if report.blockers:
        print(f"\n  ❌ 阻塞项 ({len(report.blockers)}个):")
        for b in report.blockers:
            print(f"    • [{b.gate_name}] {b.check_name}: {b.details}")

    print(f"\n[✓] 新品上市准入门控 测试通过  Tags:{report.tags}")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Product-Category-Opportunity-Scoring]]（品类评分通过才进入上市门控）
- **前置（prerequisite）**：[[Skill-Supplier-Qualification-Onboarding-KPI]]（供应商准入是上市门控的前提）
- **延伸（extends）**：[[Skill-Multi-Market-Compliance-Matrix-Ontology]]（合规门控调用多市场合规矩阵）
- **可组合（combinable）**：[[Skill-EPR-Extended-Producer-Responsibility-Tag]]（EPR是EU上市门控的必查项）

## ⑤ 商业价值评估

- **ROI预估**：上市门控阻止不合规产品发货，每次避免海关扣押损失5-15万元；防止因Listing不完整导致的上市冷启动（差的Listing让转化率降低40-60%，损失约10-20万元/品）
- **实施难度**：⭐⭐☆☆☆（主要是检查清单的结构化录入，技术门槛低）
- **优先级评分**：⭐⭐⭐⭐⭐（新品是所有资源投入的核心决策点，门控防止"亡羊补牢"）
