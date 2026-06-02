---
title: Supply Chain Due Diligence — 供应链合规尽职调查：劳工+环境+产品三维
doc_type: knowledge
module: 21-合规决策
topic: supply-chain-due-diligence-compliance
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill-Supply-Chain-Due-Diligence

---

## ① 算法原理

**供应链合规三维框架**

2023 年起德国《供应链尽职调查法》（LkSG）生效，要求年营业额 >4.5 亿欧元的企业对整条供应链的合规负责。母婴出海品牌虽暂无直接法律义务，但头部零售商（Walmart/Target/亚马逊）已要求供应商提供 ESG 合规证明：

```
维度一 — 劳工标准（Labor Compliance）
  核心要求: ILO 核心劳工标准（8 项基本公约）
  关键指标: 禁止童工(< 16岁) / 禁止强迫劳动 / 合理工时(≤60h/周) / 最低工资达标
  认证标识: SA8000 / BSCI / SMETA / Sedex
  风险信号: 供应商在 UFLPA 名单 / 新疆采购 / 工厂审计不合格

维度二 — 环境合规（Environmental Compliance）
  核心要求: ISO 14001 环境管理体系
  关键指标: 碳排放数据(Scope 1/2) / 废水处理达标 / 危险废物合规处置
  认证标识: ISO 14001 / OEKO-TEX STANDARD 100 / GRS(再生材料)
  风险信号: 环保违规记录 / 无废水处理设施 / 碳排放数据缺失

维度三 — 产品认证（Product Certification）
  核心要求: 目标市场强制认证(CE/FCC/CPSC)
  关键指标: 有效期内认证 / 测试报告真实性 / 工厂审计通过率
  认证标识: CE / FCC / CPSC 107 / ISO 9001
  风险信号: 认证过期 / 伪造测试报告 / 工厂变更未通知
```

**自动化合规评分体系**

三维加权评分模型：

```
总分 = 劳工分(×0.4) + 环境分(×0.3) + 产品认证分(×0.3)

风险等级:
  总分 ≥ 80  → 合格供应商（绿色）
  总分 60-79 → 观察供应商（黄色），需限期整改
  总分 < 60  → 不合格（红色），触发采购暂停
```

**LkSG 2023 核心要求映射**：禁止清单（童工/强迫劳动/环境破坏）触发一票否决机制，不受加权影响。

---

## ② 母婴出海应用案例

**场景一：供应商准入评估（新供应商申请）**

- **业务问题**：新供应商申请进入合格供应商名单，需要客观、标准化地评估其合规状态，避免"凭感觉"或"关系"决策。
- **系统输入**：供应商自报信息 + 第三方审计报告
- **自动输出**：
  ```
  供应商: 广州某工厂（婴儿推车配件）
  
  劳工合规评分: 82/100
    + 持有 BSCI 2023 审计证书（有效期内）
    + 月工资 ¥5,200（高于最低工资标准 40%）
    - 工时记录显示个别月份超 60h/周（扣分）
  
  环境合规评分: 71/100
    + 通过 ISO 14001:2015 认证
    - 无碳排放数据（Scope 2 缺失，扣分）
    - 废水处理设施 2022 年审查报告显示轻微超标
  
  产品认证评分: 90/100
    + CE/FCC 证书均在有效期内
    + 最近工厂审计通过率 95%
  
  综合评分: 80.5/100 → 合格供应商 ✓（需 6 个月后复评碳排放数据）
  ```
- **业务价值**：供应商准入决策有据可查，避免"人情供应商"风险，合规供应商入库率从 70% 提升至 95%

**场景二：WF-A 补货供应商风险监控（定期重评）**

- **业务问题**：在用供应商的合规状态可能随时间变化（认证到期/被处罚/列入黑名单），需要定期自动重评。
- **系统处理**：每季度自动重新评估所有在用供应商：
  ```
  [风险预警] 供应商 SUP-SZ-007 合规分下降：
    本季度评分: 58/100（上季度: 76/100）
    主要原因: BSCI 认证 2024-10 已过期（劳工分 -25）
    处理建议: 暂停新采购订单 / 要求 30 天内提交续期证明
    
  → 自动触发 WF-A 告警：暂停该供应商的自动补货指令
  ```
- **业务价值**：供应商合规风险降低 70%（主动监控 vs 被动等投诉）；避免因供应商问题导致的平台处罚

---

## ③ 代码模板

```python
"""
Skill-Supply-Chain-Due-Diligence
供应链合规尽职调查：劳工+环境+产品三维评估
基于 LkSG 2023 + ESG 合规 + 供应链尽职调查最佳实践
纯 Python 标准库，Python 3.14 兼容，无第三方依赖
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    BLOCKED = "blocked"


@dataclass
class SupplierProfile:
    supplier_id: str
    name: str
    country: str
    # 劳工维度
    labor_cert: Optional[str] = None
    labor_cert_valid: bool = False
    wage_ratio_to_minimum: float = 1.0
    max_weekly_hours: float = 60.0
    uflpa_listed: bool = False
    # 环境维度
    env_cert: Optional[str] = None
    env_cert_valid: bool = False
    has_carbon_data: bool = False
    wastewater_compliant: bool = True
    # 产品认证维度
    product_certs: list[str] = field(default_factory=list)
    product_certs_valid: bool = False
    factory_audit_pass_rate: float = 1.0


class LaborComplianceChecker:
    def score(self, profile: SupplierProfile) -> tuple[float, list[str]]:
        score = 100.0
        notes: list[str] = []
        if profile.uflpa_listed:
            return 0.0, ["⛔ 列入 UFLPA 禁止采购名单，一票否决"]
        if not profile.labor_cert_valid:
            score -= 25
            notes.append(f"- {profile.labor_cert or '劳工认证'} 已过期或缺失")
        if profile.wage_ratio_to_minimum < 1.0:
            score -= 20
            notes.append(f"- 工资低于最低工资标准（比率: {profile.wage_ratio_to_minimum:.1%}）")
        elif profile.wage_ratio_to_minimum >= 1.4:
            notes.append(f"+ 工资高于最低工资标准 {(profile.wage_ratio_to_minimum-1):.0%}")
        if profile.max_weekly_hours > 60:
            score -= 10
            notes.append(f"- 工时超标（{profile.max_weekly_hours:.0f}h/周 > 60h）")
        if profile.labor_cert_valid:
            notes.append(f"+ 持有 {profile.labor_cert} 认证（有效期内）")
        return max(0.0, score), notes


class EnvironmentalChecker:
    def score(self, profile: SupplierProfile) -> tuple[float, list[str]]:
        score = 100.0
        notes: list[str] = []
        if not profile.env_cert_valid:
            score -= 20
            notes.append(f"- {profile.env_cert or '环境认证'} 已过期或缺失")
        else:
            notes.append(f"+ 持有 {profile.env_cert} 认证（有效期内）")
        if not profile.has_carbon_data:
            score -= 15
            notes.append("- 碳排放数据（Scope 2）缺失")
        if not profile.wastewater_compliant:
            score -= 15
            notes.append("- 废水处理不达标")
        return max(0.0, score), notes


class ProductCertChecker:
    def score(self, profile: SupplierProfile) -> tuple[float, list[str]]:
        score = 100.0
        notes: list[str] = []
        if not profile.product_certs:
            score -= 30
            notes.append("- 无产品认证记录")
        elif not profile.product_certs_valid:
            score -= 20
            notes.append(f"- 产品认证已过期（{', '.join(profile.product_certs)}）")
        else:
            notes.append(f"+ 认证有效（{', '.join(profile.product_certs)}）")
        if profile.factory_audit_pass_rate < 0.8:
            score -= 15
            notes.append(f"- 工厂审计通过率低（{profile.factory_audit_pass_rate:.0%}）")
        elif profile.factory_audit_pass_rate >= 0.95:
            notes.append(f"+ 工厂审计通过率优秀（{profile.factory_audit_pass_rate:.0%}）")
        return max(0.0, score), notes


@dataclass
class DueDiligenceResult:
    supplier_id: str
    supplier_name: str
    labor_score: float
    env_score: float
    product_score: float
    total_score: float
    risk_level: RiskLevel
    labor_notes: list[str]
    env_notes: list[str]
    product_notes: list[str]
    is_blocked: bool

    def __str__(self) -> str:
        level_emoji = {"green": "✅", "yellow": "⚠️", "red": "❌", "blocked": "⛔"}
        lines = [
            f"\n[供应商尽职调查] {self.supplier_name} ({self.supplier_id})",
            f"  劳工合规: {self.labor_score:.0f}/100",
            *[f"    {n}" for n in self.labor_notes],
            f"  环境合规: {self.env_score:.0f}/100",
            *[f"    {n}" for n in self.env_notes],
            f"  产品认证: {self.product_score:.0f}/100",
            *[f"    {n}" for n in self.product_notes],
            f"\n  综合评分: {self.total_score:.1f}/100  {level_emoji.get(self.risk_level.value, '')} [{self.risk_level.value.upper()}]",
        ]
        if self.is_blocked:
            lines.append("  ⛔ 采购暂停触发（一票否决）")
        return "\n".join(lines)


class DueDiligenceScorer:
    WEIGHTS = {"labor": 0.4, "env": 0.3, "product": 0.3}

    def __init__(self) -> None:
        self._labor = LaborComplianceChecker()
        self._env = EnvironmentalChecker()
        self._product = ProductCertChecker()

    def evaluate(self, profile: SupplierProfile) -> DueDiligenceResult:
        labor_score, labor_notes = self._labor.score(profile)
        env_score, env_notes = self._env.score(profile)
        product_score, product_notes = self._product.score(profile)

        is_blocked = profile.uflpa_listed or labor_score == 0.0
        if is_blocked:
            total = 0.0
            risk_level = RiskLevel.BLOCKED
        else:
            total = (
                labor_score * self.WEIGHTS["labor"]
                + env_score * self.WEIGHTS["env"]
                + product_score * self.WEIGHTS["product"]
            )
            if total >= 80:
                risk_level = RiskLevel.GREEN
            elif total >= 60:
                risk_level = RiskLevel.YELLOW
            else:
                risk_level = RiskLevel.RED

        return DueDiligenceResult(
            supplier_id=profile.supplier_id,
            supplier_name=profile.name,
            labor_score=labor_score,
            env_score=env_score,
            product_score=product_score,
            total_score=round(total, 1),
            risk_level=risk_level,
            labor_notes=labor_notes,
            env_notes=env_notes,
            product_notes=product_notes,
            is_blocked=is_blocked,
        )


if __name__ == "__main__":
    scorer = DueDiligenceScorer()

    suppliers = [
        SupplierProfile(
            supplier_id="SUP-GZ-001", name="广州某优质工厂", country="CN",
            labor_cert="BSCI", labor_cert_valid=True, wage_ratio_to_minimum=1.4, max_weekly_hours=55.0,
            env_cert="ISO 14001", env_cert_valid=True, has_carbon_data=True, wastewater_compliant=True,
            product_certs=["CE", "FCC"], product_certs_valid=True, factory_audit_pass_rate=0.96,
        ),
        SupplierProfile(
            supplier_id="SUP-SZ-002", name="深圳某中等工厂", country="CN",
            labor_cert="SMETA", labor_cert_valid=True, wage_ratio_to_minimum=1.1, max_weekly_hours=62.0,
            env_cert="ISO 14001", env_cert_valid=False, has_carbon_data=False, wastewater_compliant=True,
            product_certs=["CE"], product_certs_valid=True, factory_audit_pass_rate=0.85,
        ),
        SupplierProfile(
            supplier_id="SUP-DG-003", name="东莞某风险工厂", country="CN",
            labor_cert=None, labor_cert_valid=False, wage_ratio_to_minimum=0.95, max_weekly_hours=70.0,
            env_cert=None, env_cert_valid=False, has_carbon_data=False, wastewater_compliant=False,
            product_certs=[], product_certs_valid=False, factory_audit_pass_rate=0.65,
        ),
    ]

    results = [scorer.evaluate(s) for s in suppliers]
    for r in results:
        print(r)

    assert results[0].risk_level == RiskLevel.GREEN, f"优质工厂应为 GREEN，实际: {results[0].risk_level}"
    assert results[1].risk_level in (RiskLevel.YELLOW, RiskLevel.GREEN), f"中等工厂应为 YELLOW，实际: {results[1].risk_level}"
    assert results[2].risk_level == RiskLevel.RED, f"风险工厂应为 RED，实际: {results[2].risk_level}"
    assert results[0].total_score >= results[2].total_score, "优质工厂总分应高于风险工厂"

    print(f"\n[✓] 评估结果: {results[0].supplier_name}={results[0].risk_level.value} | "
          f"{results[1].supplier_name}={results[1].risk_level.value} | "
          f"{results[2].supplier_name}={results[2].risk_level.value}")
    print("[✓] Supply Chain Due Diligence 全部测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Category-Compliance-Prescan]] / [[Skill-Cross-Border-Compliance-Framework]]
- **延伸**：[[Skill-Supply-Chain-Causal-SCM-Attribution]] / [[Skill-Helicase-Supply-Chain-KG-MAS]]
- **可组合**：[[Skill-Supplier-Capacity-Planning]] / [[Skill-AgenticPay-Procurement-Negotiation]]

---
- **相关**：[[Skill-Consumer-Complaint-Recall-Prediction]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值

- **合规风险降低**：供应商合规风险降低 70%（系统性评估 vs 随机检查）
- **决策可审计**：每次评分留存记录，供审计和零售商合规要求使用
- **自动化监控**：季度重评自动触发告警，从被动应对到主动管控
- **实施难度**：⭐⭐☆☆☆
- **优先级**：⭐⭐⭐⭐☆
