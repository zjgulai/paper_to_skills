"""
Regulatory Change Monitoring — 法规变更自动监控：受影响品类实时映射
paper2skills-code: 21-合规决策 | 母婴出海跨境电商

纯 Python 标准库实现（无外部依赖）
Python 3.14 兼容
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# 枚举
# ──────────────────────────────────────────────

class ChangeType(Enum):
    """法规变更类型"""
    NEW_REGULATION = "new_regulation"    # 全新法规
    AMENDMENT = "amendment"              # 修订现有法规
    ENFORCEMENT_ACTION = "enforcement"  # 执法行动（即时效力）
    GUIDANCE_UPDATE = "guidance"         # 指导方针更新


class EffectiveStatus(Enum):
    """生效状态"""
    IMMEDIATE = "immediate"   # 立即生效（无缓冲期）
    TRANSITION = "transition" # 过渡期内（有时间准备）
    FUTURE = "future"         # 未来生效（>90天）
    ALREADY_EFFECTIVE = "already_effective"  # 已生效


class AlertPriority(Enum):
    """告警优先级"""
    CRITICAL = "critical"    # 立即处理（执法行动/立即生效）
    HIGH = "high"            # 高优先（过渡期 < 90天）
    MEDIUM = "medium"        # 中优先（过渡期 90-180天）
    LOW = "low"              # 低优先（> 180天或指导方针）


# ──────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────

@dataclass
class RegulationUpdate:
    """单条法规更新记录"""
    reg_id: str                          # 法规唯一 ID（如 "EU-GPSR-2023/988"）
    agency: str                          # 监管机构（CPSC / FDA / EU / MHRA 等）
    title: str                           # 法规标题
    effective_date: date                 # 生效日期
    affected_categories: list[str]       # 受影响产品品类列表
    change_type: ChangeType              # 变更类型
    description: str = ""               # 变更描述
    source_url: str = ""                # 官方信息来源
    markets: list[str] = field(default_factory=list)  # 适用市场


@dataclass
class AffectedProduct:
    """受法规变更影响的产品信息"""
    product_category: str
    regulation: RegulationUpdate
    days_until_effective: int
    required_actions: list[str]


@dataclass
class ComplianceAlert:
    """合规告警"""
    priority: AlertPriority
    regulation: RegulationUpdate
    effective_status: EffectiveStatus
    affected_products: list[AffectedProduct]
    days_remaining: int                 # 距生效剩余天数（负数=已过期）
    action_required: str               # 需要立即采取的行动


# ──────────────────────────────────────────────
# 品类→关键词映射（用于匹配受影响品类）
# ──────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "婴儿奶粉": ["infant formula", "baby formula", "formula", "奶粉", "配方奶"],
    "婴儿食品": ["baby food", "infant food", "puree", "辅食", "婴儿食品"],
    "婴儿玩具": ["toy", "baby toy", "infant toy", "玩具"],
    "婴儿服装": ["baby clothing", "infant clothing", "onesie", "服装", "童装"],
    "婴儿床具": ["baby crib", "bassinet", "sleep product", "婴儿床", "睡篮"],
    "婴儿安全设备": ["baby gate", "car seat", "safety", "安全座椅", "防护"],
    "婴儿护肤品": ["lotion", "cream", "baby skincare", "护肤", "润肤"],
    "吸乳器": ["breast pump", "nursing", "哺乳", "吸奶器"],
    "婴儿监视器": ["baby monitor", "camera", "监视器", "婴儿监控"],
    "通用婴儿产品": ["baby", "infant", "child", "children", "婴儿", "幼儿", "儿童"],
}


def _match_categories(regulation: RegulationUpdate) -> list[str]:
    """从法规内容中推断受影响品类（基于关键词匹配）"""
    text = f"{regulation.title} {regulation.description}".lower()
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw.lower() in text for kw in keywords):
            matched.append(category)
    return matched if matched else regulation.affected_categories


# ──────────────────────────────────────────────
# 法规数据库
# ──────────────────────────────────────────────

class RegulationDatabase:
    """
    法规更新存储与查询。

    功能：
    1. 存储法规更新记录
    2. 按品类/机构/市场查询受影响的法规
    3. 查询特定品类的所有相关法规
    """

    def __init__(self) -> None:
        self._records: dict[str, RegulationUpdate] = {}

    def add(self, update: RegulationUpdate) -> None:
        """添加法规更新记录"""
        self._records[update.reg_id] = update

    def get(self, reg_id: str) -> Optional[RegulationUpdate]:
        """按 ID 查询"""
        return self._records.get(reg_id)

    def all(self) -> list[RegulationUpdate]:
        """返回所有记录"""
        return list(self._records.values())

    def find_by_category(self, category: str) -> list[RegulationUpdate]:
        """查询影响特定品类的所有法规"""
        result = []
        for reg in self._records.values():
            # 直接匹配 affected_categories
            if any(category.lower() in cat.lower() for cat in reg.affected_categories):
                result.append(reg)
                continue
            # 动态推断匹配
            inferred = _match_categories(reg)
            if any(category.lower() in cat.lower() for cat in inferred):
                result.append(reg)
        return result

    def find_by_agency(self, agency: str) -> list[RegulationUpdate]:
        """查询特定机构的法规更新"""
        return [r for r in self._records.values() if agency.lower() in r.agency.lower()]

    def find_by_market(self, market: str) -> list[RegulationUpdate]:
        """查询适用于特定市场的法规"""
        return [
            r for r in self._records.values()
            if not r.markets or market.upper() in [m.upper() for m in r.markets]
        ]

    def find_upcoming(self, within_days: int = 180, as_of: Optional[date] = None) -> list[RegulationUpdate]:
        """查询即将生效的法规（在 within_days 天内）"""
        today = as_of or date.today()
        result = []
        for reg in self._records.values():
            delta = (reg.effective_date - today).days
            if 0 <= delta <= within_days:
                result.append(reg)
        return result


# ──────────────────────────────────────────────
# 合规告警引擎
# ──────────────────────────────────────────────

class ComplianceAlertEngine:
    """
    法规变更 → 产品影响分析 → 优先级告警。

    告警触发条件：
    - 执法行动：立即 CRITICAL 告警
    - 生效日 <= 30天：CRITICAL
    - 生效日 31-90天：HIGH
    - 生效日 91-180天：MEDIUM
    - 生效日 > 180天：LOW
    - 指导方针更新：默认 LOW（降级处理）
    """

    def __init__(self, db: RegulationDatabase) -> None:
        self._db = db

    def _calc_effective_status(self, reg: RegulationUpdate, as_of: date) -> EffectiveStatus:
        delta = (reg.effective_date - as_of).days
        if delta < 0:
            return EffectiveStatus.ALREADY_EFFECTIVE
        elif delta == 0 or reg.change_type == ChangeType.ENFORCEMENT_ACTION:
            return EffectiveStatus.IMMEDIATE
        elif delta <= 90:
            return EffectiveStatus.TRANSITION
        else:
            return EffectiveStatus.FUTURE

    def _calc_priority(
        self,
        reg: RegulationUpdate,
        status: EffectiveStatus,
        days_remaining: int,
    ) -> AlertPriority:
        # 执法行动或立即生效 → CRITICAL
        if reg.change_type == ChangeType.ENFORCEMENT_ACTION or status == EffectiveStatus.IMMEDIATE:
            return AlertPriority.CRITICAL
        # 已生效 → CRITICAL（需立即审查合规状态）
        if status == EffectiveStatus.ALREADY_EFFECTIVE:
            return AlertPriority.CRITICAL
        # 按剩余天数分级
        if days_remaining <= 30:
            return AlertPriority.CRITICAL
        elif days_remaining <= 90:
            return AlertPriority.HIGH
        elif days_remaining <= 180:
            return AlertPriority.MEDIUM
        else:
            # 指导方针更新降级
            if reg.change_type == ChangeType.GUIDANCE_UPDATE:
                return AlertPriority.LOW
            return AlertPriority.LOW

    def _required_actions(
        self,
        reg: RegulationUpdate,
        priority: AlertPriority,
    ) -> list[str]:
        """根据法规类型和优先级生成所需行动清单"""
        actions = []
        if priority in (AlertPriority.CRITICAL, AlertPriority.HIGH):
            actions.append(f"立即审查受影响品类的合规状态：{reg.affected_categories}")
        if reg.change_type == ChangeType.NEW_REGULATION:
            actions.append("评估新法规对现有产品线的影响，更新合规文档")
            actions.append("与法律顾问确认产品是否需要重新认证")
        elif reg.change_type == ChangeType.AMENDMENT:
            actions.append("对比变更前后条款，识别需更新的产品文档/标签")
        elif reg.change_type == ChangeType.ENFORCEMENT_ACTION:
            actions.append("紧急停售相关产品，联系法律团队评估暴露风险")
            actions.append("准备客户沟通方案和库存处置计划")
        if priority == AlertPriority.CRITICAL:
            actions.append("在24小时内提交合规行动计划")
        return actions

    def analyze(
        self,
        regulation: RegulationUpdate,
        as_of: Optional[date] = None,
    ) -> ComplianceAlert:
        """
        分析单条法规变更，生成合规告警。

        Args:
            regulation: 法规更新记录
            as_of: 分析基准日期（默认 today）
        """
        today = as_of or date.today()
        days_remaining = (regulation.effective_date - today).days
        status = self._calc_effective_status(regulation, today)
        priority = self._calc_priority(regulation, status, days_remaining)

        # 推断受影响品类
        categories = regulation.affected_categories or _match_categories(regulation)

        # 生成受影响产品列表
        required_actions = self._required_actions(regulation, priority)
        affected_products = [
            AffectedProduct(
                product_category=cat,
                regulation=regulation,
                days_until_effective=max(days_remaining, 0),
                required_actions=required_actions,
            )
            for cat in categories
        ]

        # 行动摘要
        if priority == AlertPriority.CRITICAL:
            action_required = f"【紧急】24小时内完成合规审查：{regulation.title}"
        elif priority == AlertPriority.HIGH:
            action_required = f"【高优】30天内完成合规更新：{regulation.title}"
        elif priority == AlertPriority.MEDIUM:
            action_required = f"【中优】90天内完成合规准备：{regulation.title}"
        else:
            action_required = f"【低优】跟踪关注：{regulation.title}"

        return ComplianceAlert(
            priority=priority,
            regulation=regulation,
            effective_status=status,
            affected_products=affected_products,
            days_remaining=days_remaining,
            action_required=action_required,
        )

    def analyze_all(self, as_of: Optional[date] = None) -> list[ComplianceAlert]:
        """分析数据库中所有法规，按优先级排序"""
        alerts = [self.analyze(reg, as_of) for reg in self._db.all()]
        # 按优先级排序：CRITICAL > HIGH > MEDIUM > LOW
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3,
        }
        return sorted(alerts, key=lambda a: priority_order[a.priority])

    def check_category_risk(
        self,
        category: str,
        as_of: Optional[date] = None,
    ) -> list[ComplianceAlert]:
        """
        WF-D 选品合规门控：检查特定品类的法规变更风险。

        Returns:
            按优先级排序的告警列表（空列表 = 无已知法规风险）
        """
        regs = self._db.find_by_category(category)
        alerts = [self.analyze(reg, as_of) for reg in regs]
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3,
        }
        return sorted(alerts, key=lambda a: priority_order[a.priority])


# ──────────────────────────────────────────────
# 测试：3条法规更新
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 60)
    print("Regulatory Change Monitoring — 法规变更监控测试")
    print("=" * 60)

    # 使用固定的分析基准日期以确保测试可重复
    as_of = date(2026, 6, 1)

    # 构建数据库
    db = RegulationDatabase()

    # 法规 1：EU GPSR 2023/988（2024-12-13 已生效，当前日期 2026-06-01 → 已过期/已生效）
    reg_gpsr = RegulationUpdate(
        reg_id="EU-GPSR-2023/988",
        agency="EU",
        title="EU General Product Safety Regulation 2023/988",
        effective_date=date(2024, 12, 13),
        affected_categories=["通用婴儿产品", "婴儿玩具", "婴儿床具", "婴儿安全设备"],
        change_type=ChangeType.NEW_REGULATION,
        description=(
            "EU General Product Safety Regulation 替代 GPSD，"
            "强制要求所有 baby 和 child 产品提供数字产品护照和追溯信息。"
        ),
        source_url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R0988",
        markets=["EU", "UK"],
    )
    db.add(reg_gpsr)

    # 法规 2：CPSC 婴儿睡眠产品新标准（2026-09-30 生效 → 约 121 天后）
    reg_cpsc = RegulationUpdate(
        reg_id="CPSC-SLEEP-2025-001",
        agency="CPSC",
        title="CPSC Updated Infant Sleep Product Safety Standard 2025",
        effective_date=date(2026, 9, 30),
        affected_categories=["婴儿床具", "婴儿安全设备"],
        change_type=ChangeType.AMENDMENT,
        description=(
            "修订婴儿睡眠产品安全标准 16 CFR Part 1236，"
            "新增对便携式睡眠产品的侧壁高度和通气性要求。"
        ),
        source_url="https://www.cpsc.gov/Regulations-Laws--Standards",
        markets=["US"],
    )
    db.add(reg_cpsc)

    # 法规 3：FDA 执法行动（立即生效）
    reg_fda = RegulationUpdate(
        reg_id="FDA-FORMULA-ENFORCEMENT-2026-06",
        agency="FDA",
        title="FDA Enforcement Action: Infant Formula Contamination Alert",
        effective_date=date(2026, 6, 1),
        affected_categories=["婴儿奶粉"],
        change_type=ChangeType.ENFORCEMENT_ACTION,
        description=(
            "FDA 针对特定批次婴儿配方奶粉发布强制召回令，"
            "涉及批次号 2026-A到2026-C，相关品牌需立即下架。"
        ),
        source_url="https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts",
        markets=["US"],
    )
    db.add(reg_fda)

    engine = ComplianceAlertEngine(db)

    print(f"\n分析基准日期: {as_of}")

    # 测试 1：分析 EU GPSR（已生效 → CRITICAL）
    print("\n[法规 1] EU GPSR 2023/988 — 已生效分析")
    alert1 = engine.analyze(reg_gpsr, as_of=as_of)
    assert alert1.effective_status == EffectiveStatus.ALREADY_EFFECTIVE, (
        f"应为 ALREADY_EFFECTIVE，实际: {alert1.effective_status}"
    )
    assert alert1.priority == AlertPriority.CRITICAL, (
        f"已生效法规应为 CRITICAL，实际: {alert1.priority}"
    )
    print(f"  ✓ 状态={alert1.effective_status.value} | 优先级={alert1.priority.value}")
    print(f"  ✓ 受影响品类: {[p.product_category for p in alert1.affected_products]}")
    print(f"  ✓ 行动: {alert1.action_required}")

    # 测试 2：分析 CPSC 睡眠标准（121天后 → MEDIUM）
    print("\n[法规 2] CPSC 婴儿睡眠标准修订 — 未来生效分析")
    alert2 = engine.analyze(reg_cpsc, as_of=as_of)
    assert alert2.days_remaining > 90, f"应 >90天，实际: {alert2.days_remaining}"
    assert alert2.priority in (AlertPriority.MEDIUM, AlertPriority.LOW), (
        f"应为 MEDIUM 或 LOW，实际: {alert2.priority}"
    )
    print(f"  ✓ 剩余天数={alert2.days_remaining} | 优先级={alert2.priority.value}")
    print(f"  ✓ 状态={alert2.effective_status.value}")
    print(f"  ✓ 行动: {alert2.action_required}")

    # 测试 3：分析 FDA 执法行动（立即 → CRITICAL）
    print("\n[法规 3] FDA 婴儿奶粉执法行动 — 即时效力分析")
    alert3 = engine.analyze(reg_fda, as_of=as_of)
    assert alert3.priority == AlertPriority.CRITICAL, (
        f"执法行动应为 CRITICAL，实际: {alert3.priority}"
    )
    print(f"  ✓ 优先级={alert3.priority.value} ← 立即处理！")
    print(f"  ✓ 变更类型={reg_fda.change_type.value}")
    print(f"  ✓ 行动清单: {alert3.affected_products[0].required_actions[:2]}")

    # 测试 4：WF-D 选品合规门控
    print("\n[门控测试] 婴儿床具品类合规风险检查")
    risks = engine.check_category_risk("婴儿床具", as_of=as_of)
    assert len(risks) >= 1, "婴儿床具应有法规风险"
    print(f"  ✓ 发现 {len(risks)} 条法规风险")
    for r in risks:
        print(f"    - {r.regulation.reg_id} | {r.priority.value} | {r.action_required[:50]}...")

    # 测试 5：全量分析并按优先级排序
    print("\n[全量分析] 所有法规按优先级排序")
    all_alerts = engine.analyze_all(as_of=as_of)
    critical_count = sum(1 for a in all_alerts if a.priority == AlertPriority.CRITICAL)
    print(f"  ✓ 总告警: {len(all_alerts)} | CRITICAL: {critical_count}")
    for alert in all_alerts:
        print(f"    [{alert.priority.value.upper():8}] {alert.regulation.reg_id}")

    print("\n" + "=" * 60)
    print("[✓] 所有场景验证通过 — Regulatory Change Monitoring")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
