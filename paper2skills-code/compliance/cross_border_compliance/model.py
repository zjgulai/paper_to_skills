"""
Cross-Border E-Commerce Compliance Framework — 跨境电商多辖区合规自动映射
多辖区合规矩阵 + 规则引擎，自动生成国家专项合规清单

依赖：纯 Python 标准库（无外部依赖）
Python 版本：3.8+
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Market(Enum):
    US = "US"
    EU = "EU"
    UK = "UK"
    CA = "CA"
    AU = "AU"


class RequirementType(Enum):
    BLOCKING = "BLOCKING"
    MANDATORY = "MANDATORY"
    LABELING = "LABELING"
    ADVISORY = "ADVISORY"

    @property
    def priority(self) -> int:
        return {
            RequirementType.BLOCKING: 0,
            RequirementType.MANDATORY: 1,
            RequirementType.LABELING: 2,
            RequirementType.ADVISORY: 3,
        }[self]


@dataclass
class ComplianceRequirement:
    regulation: str
    requirement_type: RequirementType
    description: str
    mandatory: bool
    certification: Optional[str] = None
    estimated_cost_usd: Optional[int] = None
    estimated_weeks: Optional[int] = None

    def is_blocking(self) -> bool:
        return self.requirement_type == RequirementType.BLOCKING


@dataclass
class MarketComplianceResult:
    market: Market
    requirements: List[ComplianceRequirement]
    is_blocked: bool
    blocking_reasons: List[str]

    def requirements_by_type(self, req_type: RequirementType) -> List[ComplianceRequirement]:
        return [r for r in self.requirements if r.requirement_type == req_type]

    @property
    def total_estimated_cost(self) -> int:
        return sum(r.estimated_cost_usd or 0 for r in self.requirements if r.mandatory)

    @property
    def max_estimated_weeks(self) -> int:
        return max((r.estimated_weeks or 0 for r in self.requirements if r.mandatory), default=0)


@dataclass
class ComplianceReport:
    product_category: str
    target_markets: List[Market]
    market_results: Dict[Market, MarketComplianceResult]
    summary: str

    def blocked_markets(self) -> List[Market]:
        return [m for m, r in self.market_results.items() if r.is_blocked]

    def clearable_markets(self) -> List[Market]:
        return [m for m, r in self.market_results.items() if not r.is_blocked]


class ComplianceMatrix:
    """
    多辖区合规矩阵
    键: (product_category, Market) → List[ComplianceRequirement]
    """

    def __init__(self):
        self._matrix: Dict[str, Dict[Market, List[ComplianceRequirement]]] = {}
        self._load_builtin_rules()

    def get_requirements(
        self, product_category: str, market: Market
    ) -> List[ComplianceRequirement]:
        category_rules = self._matrix.get(product_category, {})
        return sorted(
            category_rules.get(market, []),
            key=lambda r: r.requirement_type.priority,
        )

    def register(
        self,
        product_category: str,
        market: Market,
        requirements: List[ComplianceRequirement],
    ) -> None:
        if product_category not in self._matrix:
            self._matrix[product_category] = {}
        self._matrix[product_category][market] = requirements

    def _load_builtin_rules(self) -> None:
        self._load_infant_formula_rules()
        self._load_baby_monitor_rules()

    def _load_infant_formula_rules(self) -> None:
        category = "infant_formula"

        self.register(category, Market.US, [
            ComplianceRequirement(
                regulation="FDA 21 CFR 107",
                requirement_type=RequirementType.BLOCKING,
                description="营养成分须符合 FDA 营养素最低要求（铁、蛋白质等 29 项）",
                mandatory=True,
                certification="FDA Facility Registration",
                estimated_cost_usd=15000,
                estimated_weeks=24,
            ),
            ComplianceRequirement(
                regulation="FDA Form 3537",
                requirement_type=RequirementType.MANDATORY,
                description="FDA 进口设施注册，每两年更新",
                mandatory=True,
                certification=None,
                estimated_cost_usd=0,
                estimated_weeks=4,
            ),
            ComplianceRequirement(
                regulation="21 CFR 107.10",
                requirement_type=RequirementType.LABELING,
                description="英文标签 + 冲泡说明，需包含警告语",
                mandatory=True,
                certification=None,
                estimated_cost_usd=2000,
                estimated_weeks=2,
            ),
        ])

        self.register(category, Market.EU, [
            ComplianceRequirement(
                regulation="IFP Regulation 2016/127 + Delegated Reg 2021/571",
                requirement_type=RequirementType.BLOCKING,
                description="组合物符合 EU 营养要求，蛋白质/脂肪/碳水比例须在规定范围内",
                mandatory=True,
                certification="EU Market Notification",
                estimated_cost_usd=20000,
                estimated_weeks=36,
            ),
            ComplianceRequirement(
                regulation="EU Market Notification",
                requirement_type=RequirementType.MANDATORY,
                description="上市前需通知成员国主管机构，提供产品成分完整报告",
                mandatory=True,
                certification=None,
                estimated_cost_usd=3000,
                estimated_weeks=8,
            ),
            ComplianceRequirement(
                regulation="EU Labeling Directive",
                requirement_type=RequirementType.LABELING,
                description="多语言标签（目标市场官方语言），禁止母乳替代宣传",
                mandatory=True,
                certification=None,
                estimated_cost_usd=5000,
                estimated_weeks=4,
            ),
        ])

        self.register(category, Market.UK, [
            ComplianceRequirement(
                regulation="UK PARNUTS (Post-Brexit)",
                requirement_type=RequirementType.BLOCKING,
                description="符合英国营养食品规定（继承 EU 但已独立更新至 2024 版）",
                mandatory=True,
                certification="UK Responsible Person",
                estimated_cost_usd=8000,
                estimated_weeks=16,
            ),
            ComplianceRequirement(
                regulation="UK Labeling Regulations",
                requirement_type=RequirementType.LABELING,
                description="英文标签 + 英国责任人（UK RP）地址，禁止欧盟地址",
                mandatory=True,
                certification=None,
                estimated_cost_usd=2000,
                estimated_weeks=2,
            ),
        ])

    def _load_baby_monitor_rules(self) -> None:
        category = "baby_monitor_smart"

        self.register(category, Market.US, [
            ComplianceRequirement(
                regulation="FCC Part 15",
                requirement_type=RequirementType.BLOCKING,
                description="无线通信设备必须通过 FCC 认证，含 WiFi/BT 模块",
                mandatory=True,
                certification="FCC ID",
                estimated_cost_usd=8000,
                estimated_weeks=12,
            ),
            ComplianceRequirement(
                regulation="CPSC 16 CFR 1303",
                requirement_type=RequirementType.MANDATORY,
                description="电气安全标准，防止儿童触电风险",
                mandatory=True,
                certification="UL Listing",
                estimated_cost_usd=6000,
                estimated_weeks=10,
            ),
            ComplianceRequirement(
                regulation="CCPA",
                requirement_type=RequirementType.MANDATORY,
                description="若采集用户数据（视频/音频/位置），须符合 CCPA 隐私要求",
                mandatory=True,
                certification=None,
                estimated_cost_usd=5000,
                estimated_weeks=8,
            ),
            ComplianceRequirement(
                regulation="ASTM F2090",
                requirement_type=RequirementType.ADVISORY,
                description="婴儿监视器自愿性安全标准（建议满足以提升竞争力）",
                mandatory=False,
                certification=None,
                estimated_cost_usd=2000,
                estimated_weeks=4,
            ),
        ])

        self.register(category, Market.EU, [
            ComplianceRequirement(
                regulation="RED Directive 2014/53/EU",
                requirement_type=RequirementType.BLOCKING,
                description="无线电设备指令，含 WiFi 设备必须通过 CE + RED 认证",
                mandatory=True,
                certification="CE Mark (RED)",
                estimated_cost_usd=12000,
                estimated_weeks=16,
            ),
            ComplianceRequirement(
                regulation="LVD 2014/35/EU",
                requirement_type=RequirementType.BLOCKING,
                description="低压指令，所有交流供电设备必须符合",
                mandatory=True,
                certification="CE Mark (LVD)",
                estimated_cost_usd=5000,
                estimated_weeks=8,
            ),
            ComplianceRequirement(
                regulation="GDPR (EU) 2016/679",
                requirement_type=RequirementType.MANDATORY,
                description="视频/音频数据采集须符合 GDPR，需 DPA 和数据处理协议",
                mandatory=True,
                certification="GDPR Compliance",
                estimated_cost_usd=8000,
                estimated_weeks=12,
            ),
            ComplianceRequirement(
                regulation="REACH (EC) 1907/2006",
                requirement_type=RequirementType.MANDATORY,
                description="化学品注册，产品含 SVHC 物质须申报",
                mandatory=True,
                certification="REACH Declaration",
                estimated_cost_usd=3000,
                estimated_weeks=6,
            ),
            ComplianceRequirement(
                regulation="RoHS Directive 2011/65/EU",
                requirement_type=RequirementType.MANDATORY,
                description="限制有害物质（铅/汞/镉等），需第三方检测报告",
                mandatory=True,
                certification="RoHS Test Report",
                estimated_cost_usd=2000,
                estimated_weeks=4,
            ),
        ])


class ComplianceChecker:
    def __init__(self, matrix: Optional[ComplianceMatrix] = None):
        self._matrix = matrix or ComplianceMatrix()

    def check_product(
        self,
        product_category: str,
        target_markets: List[Market],
    ) -> ComplianceReport:
        market_results: Dict[Market, MarketComplianceResult] = {}

        for market in target_markets:
            requirements = self._matrix.get_requirements(product_category, market)
            blocking_reasons = [
                r.description for r in requirements if r.is_blocking()
            ]
            market_results[market] = MarketComplianceResult(
                market=market,
                requirements=requirements,
                is_blocked=len(blocking_reasons) > 0,
                blocking_reasons=blocking_reasons,
            )

        blocked = [m for m, r in market_results.items() if r.is_blocked]
        clearable = [m for m, r in market_results.items() if not r.is_blocked]

        parts = []
        if blocked:
            parts.append(f"⚠️ {len(blocked)} 个市场存在封禁要求待满足: {[m.value for m in blocked]}")
        if clearable:
            parts.append(f"✅ {len(clearable)} 个市场无封禁障碍（仍需满足强制认证）: {[m.value for m in clearable]}")

        return ComplianceReport(
            product_category=product_category,
            target_markets=target_markets,
            market_results=market_results,
            summary=" | ".join(parts) if parts else "无合规要求",
        )


def run_demo():
    print("=" * 60)
    print("Cross-Border Compliance Framework — 演示")
    print("=" * 60)

    checker = ComplianceChecker()

    print("\n📦 场景 A: 婴儿配方奶粉 → US + EU + UK 三市场")
    report_a = checker.check_product(
        product_category="infant_formula",
        target_markets=[Market.US, Market.EU, Market.UK],
    )
    _print_report(report_a)

    print("\n📦 场景 B: 智能婴儿监视器 → US + EU 两市场")
    report_b = checker.check_product(
        product_category="baby_monitor_smart",
        target_markets=[Market.US, Market.EU],
    )
    _print_report(report_b)

    _validate(report_a, report_b)
    return report_a, report_b


def _print_report(report: ComplianceReport) -> None:
    print(f"\n  产品类别: {report.product_category}")
    print(f"  总结: {report.summary}")
    for market, result in report.market_results.items():
        print(f"\n  [{market.value}] 合规要求 ({len(result.requirements)} 项)")
        print(f"    预估总认证费用: ${result.total_estimated_cost:,}")
        print(f"    最长认证周期: {result.max_estimated_weeks} 周")
        for req_type in RequirementType:
            reqs = result.requirements_by_type(req_type)
            if reqs:
                print(f"    [{req_type.value}]")
                for r in reqs:
                    cert = f" [{r.certification}]" if r.certification else ""
                    print(f"      - {r.regulation}{cert}: {r.description}")


def _validate(report_a: ComplianceReport, report_b: ComplianceReport) -> None:
    assert len(report_a.market_results) == 3, "场景 A 应包含 3 个市场"
    assert len(report_b.market_results) == 2, "场景 B 应包含 2 个市场"

    for market, result in report_b.market_results.items():
        blocking = result.requirements_by_type(RequirementType.BLOCKING)
        assert len(blocking) > 0, f"{market.value} 智能设备应有封禁要求"

    us_result = report_b.market_results[Market.US]
    eu_result = report_b.market_results[Market.EU]
    assert len(eu_result.requirements) > len(us_result.requirements), \
        "EU 认证要求数应多于 US"

    print("\n✅ 所有断言通过 — 模块验证成功")


if __name__ == "__main__":
    run_demo()
