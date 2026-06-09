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
    labor_cert: Optional[str] = None
    labor_cert_valid: bool = False
    wage_ratio_to_minimum: float = 1.0
    max_weekly_hours: float = 60.0
    uflpa_listed: bool = False
    env_cert: Optional[str] = None
    env_cert_valid: bool = False
    has_carbon_data: bool = False
    wastewater_compliant: bool = True
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
        else:
            notes.append(f"+ 持有 {profile.labor_cert} 认证（有效期内）")
        if profile.wage_ratio_to_minimum < 1.0:
            score -= 20
            notes.append(f"- 工资低于最低工资标准（比率: {profile.wage_ratio_to_minimum:.1%}）")
        elif profile.wage_ratio_to_minimum >= 1.4:
            notes.append(f"+ 工资高于最低工资标准 {(profile.wage_ratio_to_minimum-1):.0%}")
        if profile.max_weekly_hours > 60:
            score -= 10
            notes.append(f"- 工时超标（{profile.max_weekly_hours:.0f}h/周 > 60h）")
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
    assert results[0].total_score >= results[2].total_score

    print(f"\n[✓] {results[0].supplier_name}={results[0].risk_level.value} | "
          f"{results[1].supplier_name}={results[1].risk_level.value} | "
          f"{results[2].supplier_name}={results[2].risk_level.value}")
    print("[✓] Supply Chain Due Diligence 全部测试通过")
