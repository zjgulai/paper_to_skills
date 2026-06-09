from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProductCategory(Enum):
    INFANT_FORMULA = "infant_formula"
    TOY_0_3 = "toy_0_3"
    STROLLER = "stroller"
    INFANT_CARRIER = "infant_carrier"
    CRIB = "crib"
    NIPPLE = "nipple"
    CLOTHING = "clothing"
    SKINCARE = "skincare"


class Market(Enum):
    US = "US"
    EU = "EU"
    UK = "UK"
    CA = "CA"
    AU = "AU"


class CompliancePriority(Enum):
    BLOCKING = "BLOCKING"
    MANDATORY = "MANDATORY"
    ADVISORY = "ADVISORY"


class RiskLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SafetyTestRequirement:
    test_name: str
    standard: str
    priority: CompliancePriority
    cost_low_usd: int
    cost_high_usd: int
    duration_weeks: int
    market: Market
    notes: str = ""

    @property
    def cost_range_str(self) -> str:
        return f"${self.cost_low_usd:,}-${self.cost_high_usd:,}"

    def __str__(self) -> str:
        return (
            f"  [{self.priority.value:10s}] {self.test_name:40s} "
            f"{self.cost_range_str:15s} {self.duration_weeks}周  ({self.standard})"
        )


CATEGORY_RISK: dict[ProductCategory, RiskLevel] = {
    ProductCategory.NIPPLE: RiskLevel.HIGH,
    ProductCategory.CRIB: RiskLevel.HIGH,
    ProductCategory.INFANT_FORMULA: RiskLevel.HIGH,
    ProductCategory.TOY_0_3: RiskLevel.MEDIUM,
    ProductCategory.STROLLER: RiskLevel.MEDIUM,
    ProductCategory.INFANT_CARRIER: RiskLevel.MEDIUM,
    ProductCategory.CLOTHING: RiskLevel.LOW,
    ProductCategory.SKINCARE: RiskLevel.LOW,
}

TEST_REQUIREMENTS_DB: dict[tuple[ProductCategory, Market], list[SafetyTestRequirement]] = {
    (ProductCategory.STROLLER, Market.US): [
        SafetyTestRequirement("ASTM F833 婴儿车安全标准", "ASTM F833-23", CompliancePriority.BLOCKING, 3000, 5000, 8, Market.US),
        SafetyTestRequirement("CPSC 重金属含量检测", "16 CFR 1220", CompliancePriority.BLOCKING, 1000, 1500, 4, Market.US),
        SafetyTestRequirement("JPMA 婴儿产品协会认证", "JPMA Certification", CompliancePriority.MANDATORY, 2500, 3500, 6, Market.US),
        SafetyTestRequirement("UL 安全实验室验证", "UL 2088", CompliancePriority.MANDATORY, 1200, 2000, 4, Market.US),
        SafetyTestRequirement("亚马逊 PPAI 产品安全合规", "Amazon PPAI", CompliancePriority.ADVISORY, 800, 1500, 2, Market.US),
    ],
    (ProductCategory.STROLLER, Market.EU): [
        SafetyTestRequirement("EN 1888-2 婴儿车欧盟标准", "EN 1888-2:2018", CompliancePriority.BLOCKING, 2500, 3500, 8, Market.EU),
        SafetyTestRequirement("EN 71-3 有害物质迁移", "EN 71-3:2019", CompliancePriority.BLOCKING, 700, 1200, 4, Market.EU),
        SafetyTestRequirement("CE 符合性声明", "EU 2001/95/EC", CompliancePriority.MANDATORY, 400, 800, 2, Market.EU),
        SafetyTestRequirement("REACH 化学物质限制", "REACH Regulation", CompliancePriority.MANDATORY, 600, 1000, 3, Market.EU),
    ],
    (ProductCategory.TOY_0_3, Market.US): [
        SafetyTestRequirement("ASTM F963 玩具安全标准", "ASTM F963-23", CompliancePriority.BLOCKING, 2000, 3500, 6, Market.US),
        SafetyTestRequirement("CPSC 铅含量/表面涂层", "16 CFR 1303", CompliancePriority.BLOCKING, 800, 1200, 3, Market.US),
        SafetyTestRequirement("CPSC 小零件窒息风险", "16 CFR 1501", CompliancePriority.BLOCKING, 500, 800, 2, Market.US),
        SafetyTestRequirement("CPSIA 第三方测试认证", "CPSIA Section 102", CompliancePriority.MANDATORY, 1500, 2500, 4, Market.US),
    ],
    (ProductCategory.TOY_0_3, Market.EU): [
        SafetyTestRequirement("EN 71-1 机械物理安全", "EN 71-1:2014+A1:2018", CompliancePriority.BLOCKING, 1500, 2500, 6, Market.EU),
        SafetyTestRequirement("EN 71-2 阻燃性能", "EN 71-2:2011+A1:2014", CompliancePriority.BLOCKING, 800, 1200, 4, Market.EU),
        SafetyTestRequirement("EN 71-3 化学特性", "EN 71-3:2019+A1:2021", CompliancePriority.BLOCKING, 1000, 1500, 4, Market.EU),
        SafetyTestRequirement("CE 标志认证", "EU Toy Safety Directive 2009/48/EC", CompliancePriority.MANDATORY, 500, 1000, 2, Market.EU),
    ],
    (ProductCategory.CLOTHING, Market.US): [
        SafetyTestRequirement("CPSC 睡衣阻燃标准", "16 CFR 1615/1616", CompliancePriority.BLOCKING, 600, 1000, 3, Market.US),
        SafetyTestRequirement("CPSIA 铅/邻苯测试", "CPSIA Section 101", CompliancePriority.BLOCKING, 500, 800, 2, Market.US),
        SafetyTestRequirement("ASTM F1816 拉绳安全", "ASTM F1816", CompliancePriority.MANDATORY, 300, 600, 2, Market.US),
    ],
}


class SafetyTestMapper:
    def get_requirements(
        self,
        category: ProductCategory,
        market: Market,
    ) -> list[SafetyTestRequirement]:
        reqs = TEST_REQUIREMENTS_DB.get((category, market), [])
        if not reqs:
            return [SafetyTestRequirement(
                f"{category.value} 在 {market.value} 市场无预置测试清单",
                "N/A", CompliancePriority.ADVISORY, 0, 0, 0, market,
                notes="请咨询专业测试机构获取最新要求",
            )]
        return sorted(reqs, key=lambda r: list(CompliancePriority).index(r.priority))

    def estimate_total_cost(
        self,
        requirements: list[SafetyTestRequirement],
    ) -> tuple[int, int]:
        low = sum(r.cost_low_usd for r in requirements)
        high = sum(r.cost_high_usd for r in requirements)
        return low, high


@dataclass
class ComplianceTimeline:
    category: ProductCategory
    markets: list[Market]
    total_weeks: int
    critical_path_weeks: int
    requirements_by_market: dict[Market, list[SafetyTestRequirement]]
    total_cost_low: int
    total_cost_high: int

    def __str__(self) -> str:
        lines = [
            f"\n[合规时间轴] {self.category.value} → {[m.value for m in self.markets]}",
            f"  关键路径: {self.critical_path_weeks} 周 | 总周期（含缓冲）: {self.total_weeks} 周",
            f"  总测试成本: ${self.total_cost_low:,} - ${self.total_cost_high:,}",
        ]
        for market, reqs in self.requirements_by_market.items():
            lines.append(f"\n  --- {market.value} 市场 ---")
            for r in reqs:
                lines.append(str(r))
        return "\n".join(lines)


class ComplianceTimelinePlanner:
    def __init__(self) -> None:
        self._mapper = SafetyTestMapper()

    def plan(
        self,
        category: ProductCategory,
        markets: list[Market],
        buffer_weeks: int = 4,
    ) -> ComplianceTimeline:
        reqs_by_market: dict[Market, list[SafetyTestRequirement]] = {}
        all_reqs: list[SafetyTestRequirement] = []
        for market in markets:
            reqs = self._mapper.get_requirements(category, market)
            reqs_by_market[market] = reqs
            all_reqs.extend(reqs)

        critical_path = max(
            (max((r.duration_weeks for r in reqs), default=0) for reqs in reqs_by_market.values()),
            default=0,
        )
        max_total_per_market = max(
            (sum(r.duration_weeks for r in reqs) for reqs in reqs_by_market.values()),
            default=0,
        )
        total_weeks = max_total_per_market + buffer_weeks
        total_low, total_high = self._mapper.estimate_total_cost(all_reqs)
        return ComplianceTimeline(
            category=category,
            markets=markets,
            total_weeks=total_weeks,
            critical_path_weeks=critical_path,
            requirements_by_market=reqs_by_market,
            total_cost_low=total_low,
            total_cost_high=total_high,
        )


if __name__ == "__main__":
    planner = ComplianceTimelinePlanner()
    timeline = planner.plan(ProductCategory.STROLLER, [Market.US, Market.EU])
    print(timeline)

    us_reqs = timeline.requirements_by_market[Market.US]
    blocking_count = sum(1 for r in us_reqs if r.priority == CompliancePriority.BLOCKING)
    assert blocking_count >= 1, "STROLLER/US 应有至少1项 BLOCKING 要求"
    assert timeline.total_cost_low > 0, "总成本应 > 0"
    assert timeline.critical_path_weeks > 0, "关键路径应 > 0"
    print(f"\n[✓] 婴儿推车 US+EU: {blocking_count} 项 BLOCKING | 关键路径 {timeline.critical_path_weeks} 周 | 成本 ${timeline.total_cost_low:,}-${timeline.total_cost_high:,}")

    toy_timeline = planner.plan(ProductCategory.TOY_0_3, [Market.US, Market.EU])
    print(toy_timeline)
    print("\n[✓] Product Safety Testing Requirements 全部测试通过")
