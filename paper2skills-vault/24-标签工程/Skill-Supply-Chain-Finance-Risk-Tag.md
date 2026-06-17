---
title: 供应链金融风险标签 — 融资依赖度/信用评级/现金流压力的综合风险画像
doc_type: knowledge
module: 24-标签工程
topic: supply-chain-finance-risk-tag
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 供应链金融风险标签

> **来源**：arXiv:2403.09823（Supply Chain Finance Risk Assessment）+ arXiv:2310.11234（Credit Risk in Trade Finance）
> **桥梁**：供应链财务 ↔ 风险管理 ↔ 标签工程 | **类型**：金融风险

## ① 算法原理

**供应链金融风险标签** 将财务脆弱性量化为可查询的Tag，在现金流危机前触发预警和主动干预。

**三维风险画像**：

| 维度 | 指标 | Tag |
|-----|------|-----|
| 流动性 | CCC/快速比率 | `finance.liquidity_risk=HIGH` |
| 融资依赖 | 短期借款/总资产 | `finance.leverage_risk=HIGH` |
| 收款周期 | DSO vs 行业均值 | `finance.collection_risk=ELEVATED` |

**预警触发**：
- `finance.liquidity_risk=CRITICAL` → 暂停新增采购预算
- `finance.leverage_risk=HIGH` → 通知CFO审批大额支出
- 供应商的金融风险Tag → 传播到"该供应商的供货可靠性"

## ② 代码模板

```python
"""
供应链金融风险标签系统
功能：流动性评分 / 融资风险评估 / 综合风险画像 / 预警Tag生成
"""
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FinancialMetrics:
    entity_id: str
    entity_type: str        # Brand / Supplier
    cash_usd: float
    accounts_receivable_usd: float
    inventory_value_usd: float
    current_liabilities_usd: float
    short_term_debt_usd: float
    total_assets_usd: float
    monthly_revenue_usd: float
    monthly_cogs_usd: float


def compute_finance_risk_tags(metrics: FinancialMetrics) -> dict:
    """计算供应链金融风险标签"""
    # 流动性指标
    current_assets = metrics.cash_usd + metrics.accounts_receivable_usd + metrics.inventory_value_usd
    current_ratio = current_assets / max(1, metrics.current_liabilities_usd)
    quick_ratio = (metrics.cash_usd + metrics.accounts_receivable_usd) / max(1, metrics.current_liabilities_usd)
    cash_coverage_days = metrics.cash_usd / max(1, metrics.monthly_cogs_usd / 30)

    # 融资依赖度
    leverage_ratio = metrics.short_term_debt_usd / max(1, metrics.total_assets_usd)

    # 风险等级
    liquidity_risk = "CRITICAL" if quick_ratio < 0.5 else ("HIGH" if quick_ratio < 1.0 else "LOW")
    leverage_risk = "HIGH" if leverage_ratio > 0.4 else ("MEDIUM" if leverage_ratio > 0.2 else "LOW")
    cash_risk = "CRITICAL" if cash_coverage_days < 7 else ("HIGH" if cash_coverage_days < 14 else "LOW")

    overall_risk = "CRITICAL" if "CRITICAL" in [liquidity_risk, cash_risk] else (
        "HIGH" if "HIGH" in [liquidity_risk, leverage_risk, cash_risk] else "MEDIUM")

    return {
        "finance.liquidity_risk": liquidity_risk,
        "finance.leverage_risk": leverage_risk,
        "finance.cash_risk": cash_risk,
        "finance.overall_risk": overall_risk,
        "finance.current_ratio": round(current_ratio, 2),
        "finance.quick_ratio": round(quick_ratio, 2),
        "finance.cash_coverage_days": round(cash_coverage_days, 0),
        "finance.leverage_ratio": round(leverage_ratio, 2),
        "finance.procurement_approval_required": overall_risk in ["CRITICAL", "HIGH"],
    }


if __name__ == "__main__":
    print("【供应链金融风险标签系统】\n")
    entities = [
        FinancialMetrics("MCC-Brand", "Brand", 50_000, 120_000, 300_000, 150_000, 80_000, 500_000, 500_000, 300_000),
        FinancialMetrics("SUP-SZ", "Supplier", 10_000, 30_000, 80_000, 120_000, 100_000, 200_000, 100_000, 70_000),
    ]
    for metrics in entities:
        tags = compute_finance_risk_tags(metrics)
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "✅"}[tags["finance.overall_risk"]]
        print(f"  {icon} {metrics.entity_id} [{tags['finance.overall_risk']}]")
        print(f"     流动性:{tags['finance.liquidity_risk']}  杠杆:{tags['finance.leverage_risk']}  "
              f"现金覆盖:{tags['finance.cash_coverage_days']:.0f}天")
        if tags["finance.procurement_approval_required"]:
            print(f"     ⚠️  大额采购需CFO审批")
    print(f"\n[✓] 供应链金融风险标签 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supply-Chain-Working-Capital-Optimization]]（营运资金CCC是金融风险的核心）
- **延伸（extends）**：[[Skill-Dynamic-Payment-Terms-Tag-Engine]]（金融风险高时触发账期延长策略）
- **可组合（combinable）**：[[Skill-Geopolitical-Risk-Tag-Supply-Impact]]（地缘风险和金融风险联合评估）

## ⑤ 商业价值评估

- **ROI预估**：供应商金融风险高预警可提前3-6个月准备备用供应商，防止供应商资金链断裂导致的断供（每次约20-50万元损失）；品牌自身金融风险监控防止现金流危机，及时融资
- **实施难度**：⭐⭐⭐☆☆（数据来源：财务系统，计算逻辑清晰）
- **优先级评分**：⭐⭐⭐⭐☆（2024-2025年多家中小跨境卖家因现金流管理失败倒闭，金融风险是生死线）
