"""
User Profile Long Memory — 跨会话用户画像积累
来源: Personalized Memory Architecture + LLM Personal Assistant 2025-2026
场景: 育儿阶段感知 + 品牌偏好 + 价格敏感度的持久化画像管理
"""

import math
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# 枚举定义
# ─────────────────────────────────────────────

class UserProfileDimension(Enum):
    """用户画像维度枚举"""
    PARENTING_STAGE = "parenting_stage"       # 育儿阶段
    BRAND_PREFERENCE = "brand_preference"     # 品牌偏好
    PRICE_SENSITIVITY = "price_sensitivity"   # 价格敏感度
    ORGANIC_PREFERENCE = "organic_preference" # 有机偏好
    CATEGORY_AFFINITY = "category_affinity"   # 品类亲和度
    QUALITY_WEIGHT = "quality_weight"         # 品质权重
    BABY_AGE_MONTHS = "baby_age_months"       # 宝宝月龄（推断值）


class ParentingStage(Enum):
    """育儿阶段枚举"""
    NEWBORN = "newborn"                 # 新生儿（0-3月）
    INFANT = "infant"                   # 婴儿期（4-6月）
    CRAWLER = "crawler"                 # 爬行期（7-9月）
    TODDLER_EARLY = "toddler_early"     # 学步初期（10-12月）
    TODDLER_MID = "toddler_mid"         # 学步中期（13-18月）
    TODDLER_LATE = "toddler_late"       # 学步晚期（19-24月）
    PRESCHOOL = "preschool"             # 学前期（25+月）
    UNKNOWN = "unknown"                 # 未知阶段


# ─────────────────────────────────────────────
# 数据类定义
# ─────────────────────────────────────────────

@dataclass
class ProfileAttribute:
    """单一画像属性，含置信度与衰减机制"""
    value: object
    confidence: float                          # 置信度 [0,1]
    last_updated: str                          # ISO 8601 时间戳
    decay_rate: float = 0.05                   # 月衰减率（λ）
    source: str = "purchase_history"           # 来源标注
    evidence_count: int = 1                    # 支撑证据条数

    def effective_confidence(self, now: Optional[datetime] = None) -> float:
        """计算时间衰减后的有效置信度：v = confidence * exp(-λ * Δt_months)"""
        if now is None:
            now = datetime.now(timezone.utc)
        last = datetime.fromisoformat(self.last_updated)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta_months = (now - last).days / 30.0
        decayed = self.confidence * math.exp(-self.decay_rate * delta_months)
        return round(max(0.0, min(1.0, decayed)), 4)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            "decay_rate": self.decay_rate,
            "source": self.source,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileAttribute":
        return cls(
            value=d["value"],
            confidence=d["confidence"],
            last_updated=d["last_updated"],
            decay_rate=d.get("decay_rate", 0.05),
            source=d.get("source", "unknown"),
            evidence_count=d.get("evidence_count", 1),
        )


@dataclass
class PurchaseEvent:
    """购买事件，用于育儿阶段推断"""
    product_category: str           # 如 "formula_stage1", "baby_puree"
    purchase_date: str              # ISO 8601
    brand: Optional[str] = None
    price: Optional[float] = None
    quantity: int = 1


# ─────────────────────────────────────────────
# 育儿阶段推断器
# ─────────────────────────────────────────────

class ParentingStageInferrer:
    """从购买历史推断育儿阶段与宝宝月龄"""

    # 品类 → (最低月龄, 最高月龄, 推断月龄中值)
    CATEGORY_AGE_MAP: dict = {
        "formula_stage1": (0, 6, 2),
        "formula_stage2": (6, 12, 8),
        "formula_stage3": (12, 36, 18),
        "baby_puree":     (4, 8, 6),
        "baby_cereal":    (4, 12, 7),
        "finger_food":    (8, 18, 12),
        "sippy_cup":      (6, 18, 10),
        "walker":         (9, 15, 12),
        "push_toy":       (10, 24, 15),
        "toddler_snack":  (12, 36, 18),
    }

    @classmethod
    def infer_from_purchase(cls, event: PurchaseEvent) -> tuple:
        """从单次购买推断宝宝月龄，返回 (月龄, 置信度)"""
        cat = event.product_category
        if cat not in cls.CATEGORY_AGE_MAP:
            return 0, 0.0
        _, _, midpoint = cls.CATEGORY_AGE_MAP[cat]
        return midpoint, 0.5

    @classmethod
    def infer_from_history(cls, purchases: list) -> tuple:
        """从多次购买历史综合推断月龄（按时间最新的购买权重最高）"""
        if not purchases:
            return 0, 0.0

        sorted_events = sorted(purchases, key=lambda e: e.purchase_date, reverse=True)

        weighted_age_sum = 0.0
        total_weight = 0.0
        confidence_complement = 1.0

        for idx, event in enumerate(sorted_events):
            age, conf = cls.infer_from_purchase(event)
            if conf == 0:
                continue
            time_weight = max(0.1, 1.0 - idx * 0.15)
            purchase_dt = datetime.fromisoformat(event.purchase_date)
            if purchase_dt.tzinfo is None:
                purchase_dt = purchase_dt.replace(tzinfo=timezone.utc)
            months_since = (datetime.now(timezone.utc) - purchase_dt).days / 30.0
            current_age = age + months_since

            weighted_age_sum += current_age * time_weight * conf
            total_weight += time_weight * conf
            confidence_complement *= (1 - conf * time_weight)

        if total_weight == 0:
            return 0, 0.0

        final_age = int(weighted_age_sum / total_weight)
        final_confidence = round(min(0.95, 1 - confidence_complement), 3)
        return final_age, final_confidence

    @classmethod
    def age_to_stage(cls, age_months: int) -> ParentingStage:
        """月龄映射到育儿阶段"""
        if age_months <= 3:
            return ParentingStage.NEWBORN
        elif age_months <= 6:
            return ParentingStage.INFANT
        elif age_months <= 9:
            return ParentingStage.CRAWLER
        elif age_months <= 12:
            return ParentingStage.TODDLER_EARLY
        elif age_months <= 18:
            return ParentingStage.TODDLER_MID
        elif age_months <= 24:
            return ParentingStage.TODDLER_LATE
        else:
            return ParentingStage.PRESCHOOL


# ─────────────────────────────────────────────
# 用户画像存储
# ─────────────────────────────────────────────

class UserProfileStore:
    """跨会话用户画像存储：支持更新/检索/衰减/隐私保护"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._attributes: dict = {}
        self._inferrer = ParentingStageInferrer()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def update(
        self,
        dimension: UserProfileDimension,
        value: object,
        confidence: float,
        decay_rate: float = 0.05,
        source: str = "purchase_history",
    ) -> None:
        """更新或创建画像属性（置信度叠加）"""
        key = dimension.value
        if key in self._attributes:
            existing = self._attributes[key]
            combined_conf = min(0.97, 1 - (1 - existing.confidence) * (1 - confidence))
            self._attributes[key] = ProfileAttribute(
                value=value,
                confidence=round(combined_conf, 4),
                last_updated=self._now_iso(),
                decay_rate=decay_rate,
                source=source,
                evidence_count=existing.evidence_count + 1,
            )
        else:
            self._attributes[key] = ProfileAttribute(
                value=value,
                confidence=confidence,
                last_updated=self._now_iso(),
                decay_rate=decay_rate,
                source=source,
                evidence_count=1,
            )

    def ingest_purchase(self, event: PurchaseEvent) -> None:
        """从购买事件更新画像（品牌偏好 + 价格区间 + 品类亲和度）"""
        if event.brand:
            self.update(
                UserProfileDimension.BRAND_PREFERENCE,
                event.brand,
                confidence=0.6,
                decay_rate=0.05,
                source="purchase",
            )
        if event.price:
            self.update(
                UserProfileDimension.PRICE_SENSITIVITY,
                round(event.price, 2),
                confidence=0.5,
                decay_rate=0.1,
                source="purchase",
            )
        self.update(
            UserProfileDimension.CATEGORY_AFFINITY,
            event.product_category,
            confidence=0.55,
            decay_rate=0.08,
            source="purchase",
        )

    def ingest_purchase_history(self, purchases: list) -> None:
        """批量处理购买历史，更新育儿阶段画像"""
        for event in purchases:
            self.ingest_purchase(event)
        age, conf = ParentingStageInferrer.infer_from_history(purchases)
        if conf > 0.1:
            stage = ParentingStageInferrer.age_to_stage(age)
            self.update(
                UserProfileDimension.BABY_AGE_MONTHS,
                age,
                confidence=conf,
                decay_rate=0.01,
                source="inferred_from_purchases",
            )
            self.update(
                UserProfileDimension.PARENTING_STAGE,
                stage.value,
                confidence=conf,
                decay_rate=0.01,
                source="inferred_from_purchases",
            )

    def get(
        self,
        dimension: UserProfileDimension,
        min_confidence: float = 0.2,
    ) -> Optional[ProfileAttribute]:
        """检索画像属性，低于置信度阈值返回 None"""
        attr = self._attributes.get(dimension.value)
        if attr is None:
            return None
        if attr.effective_confidence() < min_confidence:
            return None
        return attr

    def get_all_active(self, min_confidence: float = 0.2) -> dict:
        """返回所有有效置信度超过阈值的画像维度"""
        result = {}
        now = datetime.now(timezone.utc)
        for key, attr in self._attributes.items():
            eff_conf = attr.effective_confidence(now)
            if eff_conf >= min_confidence:
                result[key] = {
                    "value": attr.value,
                    "effective_confidence": eff_conf,
                    "evidence_count": attr.evidence_count,
                    "source": attr.source,
                }
        return result

    def reset(self) -> None:
        """用户主动重置画像（GDPR/CCPA 合规）"""
        self._attributes.clear()

    def export_anonymized(self) -> dict:
        """导出脱敏画像（不含用户 ID 原始信息）"""
        return {
            "dimensions": {k: v.to_dict() for k, v in self._attributes.items()},
            "exported_at": self._now_iso(),
        }

    def to_json(self) -> str:
        data = {
            "user_id_hash": hash(self.user_id) % (10**9),
            "attributes": {k: v.to_dict() for k, v in self._attributes.items()},
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, user_id: str, json_str: str) -> "UserProfileStore":
        store = cls(user_id)
        data = json.loads(json_str)
        for key, attr_dict in data.get("attributes", {}).items():
            store._attributes[key] = ProfileAttribute.from_dict(attr_dict)
        return store


# ─────────────────────────────────────────────
# 测试验证
# ─────────────────────────────────────────────

def test_user_profile_accumulation() -> None:
    """5次购买历史，验证画像积累和育儿阶段推断"""
    print("=" * 55)
    print("测试：用户画像积累 + 育儿阶段推断")
    print("=" * 55)

    store = UserProfileStore(user_id="user_001")

    purchases = [
        PurchaseEvent("formula_stage1", "2026-03-01T10:00:00+00:00", brand="Aptamil", price=72.0),
        PurchaseEvent("formula_stage1", "2026-03-28T14:00:00+00:00", brand="Aptamil", price=72.0),
        PurchaseEvent("baby_cereal",    "2026-04-15T09:00:00+00:00", brand="Gerber",  price=18.5),
        PurchaseEvent("baby_puree",     "2026-05-01T11:00:00+00:00", brand="Ella's",  price=24.0),
        PurchaseEvent("formula_stage2", "2026-05-20T16:00:00+00:00", brand="Aptamil", price=78.0),
    ]

    store.ingest_purchase_history(purchases)

    print("\n【活跃画像维度】")
    active = store.get_all_active(min_confidence=0.15)
    for dim, info in active.items():
        print(
            f"  {dim:30s} → value={str(info['value']):20s} "
            f"conf={info['effective_confidence']:.3f}  (n={info['evidence_count']})"
        )

    stage_attr = store.get(UserProfileDimension.PARENTING_STAGE)
    age_attr   = store.get(UserProfileDimension.BABY_AGE_MONTHS)
    brand_attr = store.get(UserProfileDimension.BRAND_PREFERENCE)

    print("\n【关键断言】")
    assert stage_attr is not None, "❌ 育儿阶段未推断"
    assert age_attr is not None,   "❌ 宝宝月龄未推断"
    assert brand_attr is not None, "❌ 品牌偏好未记录"
    assert brand_attr.value == "Aptamil", f"❌ 品牌应为 Aptamil，实际={brand_attr.value}"

    inferred_age = age_attr.value
    assert 3 <= inferred_age <= 18, f"❌ 推断月龄异常: {inferred_age}"

    print(f"  ✅ 育儿阶段推断: {stage_attr.value} (conf={stage_attr.effective_confidence():.3f})")
    print(f"  ✅ 宝宝月龄推断: {inferred_age} 月 (conf={age_attr.effective_confidence():.3f})")
    print(f"  ✅ 品牌偏好记录: {brand_attr.value} (conf={brand_attr.effective_confidence():.3f})")

    price_attr = store.get(UserProfileDimension.PRICE_SENSITIVITY)
    assert price_attr is not None
    print(f"  ✅ 价格区间记录: {price_attr.value}")

    # JSON 往返
    json_str = store.to_json()
    restored = UserProfileStore.from_json("user_001", json_str)
    assert restored.get(UserProfileDimension.PARENTING_STAGE) is not None
    print("  ✅ JSON 序列化/反序列化 往返正常")

    print("\n✅ 所有断言通过！")


if __name__ == "__main__":
    test_user_profile_accumulation()
