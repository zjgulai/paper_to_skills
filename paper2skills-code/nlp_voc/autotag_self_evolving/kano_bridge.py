"""Aspect-Kano 桥接模块

将 AutoTag 产出的 L3/L4 标签 + 情感强度映射到 Kano 需求分类，
直接对接 Kano 需求优先级排序和 iReFeed 决策引擎。

Kano 模型回顾：
- 基本型(Must-be): 缺失→极度不满, 存在→无感。如"不漏尿"
- 期望型(One-dimensional): 线性满足。如"越柔软越好"
- 兴奋型(Attractive): 缺失→无感, 存在→惊喜。如"附赠试用装"
- 无差异型(Indifferent): 有无都不影响满意度
- 反向型(Reverse): 存在反而降低满意度
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sentiment_intensity import IntensityResult


@dataclass
class KanoMapping:
    """单条标签的 Kano 映射结果"""

    label_l3: str
    label_l4: Optional[str]
    kano_type: str            # "must_be" | "one_dim" | "attractive" | "indifferent" | "reverse"
    kano_name: str            # 中文名称
    sentiment_intensity: float  # -5 ~ +5
    priority_score: float     # 0-10，越大越需要优先处理
    action: str               # 建议行动

    def to_dict(self) -> dict:
        return {
            "label_l3": self.label_l3,
            "label_l4": self.label_l4,
            "kano_type": self.kano_type,
            "kano_name": self.kano_name,
            "sentiment_intensity": round(self.sentiment_intensity, 2),
            "priority_score": round(self.priority_score, 2),
            "action": self.action,
        }


@dataclass
class KanoReport:
    """Kano 桥接报告"""

    total_items: int
    kano_distribution: dict[str, int]
    priority_ranking: list[KanoMapping]
    action_summary: dict[str, list[str]]

    def to_dict(self) -> dict:
        return {
            "total_items": self.total_items,
            "kano_distribution": self.kano_distribution,
            "priority_ranking": [m.to_dict() for m in self.priority_ranking],
            "action_summary": self.action_summary,
        }


class KanoMapper:
    """Kano 需求分类映射器

    基于业务规则将 VOC 标签映射到 Kano 五类需求模型。
    映射规则由母婴出海业务场景定义，可按品类扩展。
    """

    # Kano 类型映射规则：关键词 → Kano 类型
    # 优先级：长词优先匹配（避免短词误匹配）
    KANO_RULES: dict[str, str] = {
        # 基本型(Must-be): 安全、合规、基础功能
        "漏": "must_be",
        "过敏": "must_be",
        "异味": "must_be",
        "发霉": "must_be",
        "甲醛": "must_be",
        "有害": "must_be",
        "安全隐患": "must_be",
        "不合格": "must_be",
        "破损": "must_be",
        "失效": "must_be",
        # 期望型(One-dimensional): 性能、效率、体验
        "柔软": "one_dim",
        "透气": "one_dim",
        "吸水量": "one_dim",
        "干爽": "one_dim",
        "尺码": "one_dim",
        "物流": "one_dim",
        "配送": "one_dim",
        "价格": "one_dim",
        "性价比": "one_dim",
        "耐用": "one_dim",
        "溶解": "one_dim",
        "粘性": "one_dim",
        "效果": "one_dim",
        # 兴奋型(Attractive): 惊喜、超预期
        "赠品": "attractive",
        "试用装": "attractive",
        "礼品": "attractive",
        "惊喜": "attractive",
        "贴心": "attractive",
        "设计感": "attractive",
        "夜光": "attractive",
        "温感": "attractive",
        "智能": "attractive",
        "环保": "attractive",
        "可持续": "attractive",
        # 反向型(Reverse): 过度反而不好
        "太厚": "reverse",
        "太重": "reverse",
        "过度": "reverse",
        "香味太浓": "reverse",
        "包装过度": "reverse",
    }

    KANO_NAMES = {
        "must_be": "基本型",
        "one_dim": "期望型",
        "attractive": "兴奋型",
        "indifferent": "无差异型",
        "reverse": "反向型",
    }

    def __init__(self, custom_rules: Optional[dict[str, str]] = None):
        """初始化

        Args:
            custom_rules: 自定义映射规则，覆盖默认规则
        """
        self.rules = dict(self.KANO_RULES)
        if custom_rules:
            self.rules.update(custom_rules)
        # 按关键词长度降序排列，优先匹配长词
        self.sorted_keywords = sorted(self.rules.keys(), key=len, reverse=True)

    def map_label(
        self,
        label_l3: str,
        label_l4: Optional[str],
        intensity: float,
    ) -> KanoMapping:
        """将单条标签映射到 Kano 分类

        Args:
            label_l3: L3 标签名称
            label_l4: L4 标签名称（可选）
            intensity: 情感强度 -5~+5

        Returns:
            Kano 映射结果
        """
        # 1. 确定 Kano 类型
        kano_type = self._detect_kano_type(label_l3, label_l4)
        kano_name = self.KANO_NAMES.get(kano_type, "未知")

        # 2. 计算优先级分数
        priority = self._compute_priority(kano_type, intensity)

        # 3. 生成行动建议
        action = self._generate_action(kano_type, intensity, label_l3, label_l4)

        return KanoMapping(
            label_l3=label_l3,
            label_l4=label_l4,
            kano_type=kano_type,
            kano_name=kano_name,
            sentiment_intensity=intensity,
            priority_score=priority,
            action=action,
        )

    def map_batch(
        self,
        labels_l3: list[str],
        labels_l4: list[Optional[str]],
        intensities: list[float],
    ) -> list[KanoMapping]:
        """批量映射"""
        results = []
        for l3, l4, intensity in zip(labels_l3, labels_l4, intensities):
            results.append(self.map_label(l3, l4, intensity))
        return results

    def build_report(
        self,
        mappings: list[KanoMapping],
    ) -> KanoReport:
        """构建 Kano 桥接报告"""
        # Kano 分布
        distribution: dict[str, int] = {}
        for m in mappings:
            distribution[m.kano_type] = distribution.get(m.kano_type, 0) + 1

        # 按优先级排序
        sorted_mappings = sorted(mappings, key=lambda x: x.priority_score, reverse=True)

        # 行动汇总
        actions: dict[str, list[str]] = {}
        for m in sorted_mappings:
            if m.kano_type not in actions:
                actions[m.kano_type] = []
            # 去重
            action_key = f"{m.label_l3}/{m.label_l4}: {m.action}"
            if action_key not in actions[m.kano_type]:
                actions[m.kano_type].append(action_key)

        return KanoReport(
            total_items=len(mappings),
            kano_distribution=distribution,
            priority_ranking=sorted_mappings,
            action_summary=actions,
        )

    def _detect_kano_type(
        self,
        label_l3: str,
        label_l4: Optional[str],
    ) -> str:
        """检测标签对应的 Kano 类型

        优先匹配 L4（更具体），L4 无匹配时再匹配 L3。
        避免 L3 的宽泛词（如"质量"）覆盖 L4 的精确词（如"漏尿"）。
        """
        # 1. 先在 L4 中搜索（优先）
        if label_l4:
            for keyword in self.sorted_keywords:
                if keyword in label_l4:
                    return self.rules[keyword]

        # 2. L4 无匹配，在 L3 中搜索
        for keyword in self.sorted_keywords:
            if keyword in label_l3:
                return self.rules[keyword]

        # 默认
        return "indifferent"

    def _compute_priority(self, kano_type: str, intensity: float) -> float:
        """计算优先级分数 (0-10)

        优先级逻辑：
        - 基本型 + 负面 → 最高（安全问题）
        - 期望型 + 负面 → 中高（竞争力下降）
        - 兴奋型 + 负面 → 中（机会点）
        - 基本型 + 正面 → 低（基线达标）
        - 兴奋型 + 正面 → 中高（差异化优势）
        """
        abs_intensity = abs(intensity)
        is_negative = intensity < -0.2

        if kano_type == "must_be":
            # 基本型：负面 = 危机，正面 = 达标
            base = 8.0 if is_negative else 2.0
            return min(base + abs_intensity * 0.3, 10.0)

        elif kano_type == "one_dim":
            # 期望型：线性递减/递增
            if is_negative:
                return min(5.0 + abs_intensity * 0.8, 9.0)
            else:
                return min(3.0 + abs_intensity * 0.5, 6.0)

        elif kano_type == "attractive":
            # 兴奋型：负面 = 无所谓（本来就没有期望）
            #         正面 = 差异化优势
            if is_negative:
                return min(2.0 + abs_intensity * 0.3, 4.0)
            else:
                return min(4.0 + abs_intensity * 0.8, 8.0)

        elif kano_type == "reverse":
            # 反向型：负面 = 需要减少（反而是好事）
            if is_negative:
                return min(3.0 + abs_intensity * 0.5, 6.0)
            else:
                return min(5.0 + abs_intensity * 0.5, 8.0)

        else:  # indifferent
            return min(2.0 + abs_intensity * 0.3, 4.0)

    def _generate_action(
        self,
        kano_type: str,
        intensity: float,
        label_l3: str,
        label_l4: Optional[str],
    ) -> str:
        """生成行动建议"""
        is_negative = intensity < -0.2
        is_strong = abs(intensity) >= 2.5
        detail = label_l4 or label_l3

        if kano_type == "must_be":
            if is_negative:
                return f"立即整改: {detail} 是基本安全/合规需求，负面反馈{'严重' if is_strong else '较多'}"
            return f"维持现状: {detail} 基本达标"

        elif kano_type == "one_dim":
            if is_negative:
                return f"优化提升: {detail} 是竞争力要素，需{'紧急' if is_strong else '持续'}改进"
            return f"保持优势: {detail} 表现良好，可作为卖点"

        elif kano_type == "attractive":
            if is_negative:
                return f"观察: {detail} 负面反馈有限，兴奋型需求暂不优先处理"
            return f"放大亮点: {detail} 是超预期惊喜，建议{'大力' if is_strong else '适度'}宣传"

        elif kano_type == "reverse":
            if is_negative:
                return f"反向机会: {detail} 消费者嫌{'过' if is_strong else '略'}多，可减少"
            return f"注意: {detail} 消费者反馈正面，但可能过度"

        return f"观察: {detail} 影响有限，暂不行动"


# ── 与 AutoTag 直接集成的便捷接口 ─────────────────────────────

def bridge_autotag_to_kano(
    predictions: list[dict],
    intensity_results: list[IntensityResult],
    kano_mapper: Optional[KanoMapper] = None,
) -> KanoReport:
    """一键桥接：AutoTag 预测结果 → Kano 分类报告

    Args:
        predictions: AutoTag PredictionResult.to_dict() 列表
        intensity_results: SentimentIntensityQuantifier 输出列表
        kano_mapper: 可选的自定义 KanoMapper

    Returns:
        Kano 桥接报告
    """
    mapper = kano_mapper or KanoMapper()

    labels_l3 = [p.get("l3") or "未分类" for p in predictions]
    labels_l4 = [p.get("l4") for p in predictions]
    intensities = [r.intensity for r in intensity_results]

    mappings = mapper.map_batch(labels_l3, labels_l4, intensities)
    return mapper.build_report(mappings)


# ── 测试 ──────────────────────────────────────────────────────

def test_kano_bridge():
    print("=" * 60)
    print("测试: KanoMapper")
    print("=" * 60)

    mapper = KanoMapper()

    # 测试用例：覆盖 Kano 五类
    test_cases = [
        # (L3, L4, intensity, expected_type)
        ("质量", "漏尿", -4.2, "must_be"),           # 基本型 + 强负面 = 危机
        ("质量", "过敏反应", -3.5, "must_be"),       # 基本型 + 强负面 = 危机
        ("质量", "柔软度", -2.0, "one_dim"),         # 期望型 + 中度负面
        ("质量", "柔软度", +3.0, "one_dim"),         # 期望型 + 正面
        ("物流", "配送时效", -1.5, "one_dim"),       # 期望型 + 轻度负面
        ("服务", "赠品", +4.0, "attractive"),        # 兴奋型 + 强正面
        ("服务", "试用装", -0.5, "attractive"),      # 兴奋型 + 轻微负面
        ("质量", "香味太浓", -2.5, "reverse"),       # 反向型 + 负面
        ("包装", "颜色", +0.5, "indifferent"),       # 无差异
    ]

    print("\n--- 单条映射测试 ---")
    for l3, l4, intensity, expected in test_cases:
        result = mapper.map_label(l3, l4, intensity)
        status = "✓" if result.kano_type == expected else "✗"
        print(
            f"  {status} [{result.kano_name}] 优先级:{result.priority_score:.1f} "
            f"{l3}/{l4} (强度:{intensity:+.1f})"
        )
        print(f"      行动: {result.action}")
        assert result.kano_type == expected, (
            f"期望 {expected}, 实际 {result.kano_type}"
        )

    # 批量映射 + 报告
    print("\n--- 批量报告测试 ---")
    labels_l3 = [t[0] for t in test_cases]
    labels_l4 = [t[1] for t in test_cases]
    intensities = [t[2] for t in test_cases]

    mappings = mapper.map_batch(labels_l3, labels_l4, intensities)
    report = mapper.build_report(mappings)

    print(f"\n  总计: {report.total_items} 条")
    print(f"  Kano 分布:")
    for ktype, count in sorted(report.kano_distribution.items()):
        name = mapper.KANO_NAMES.get(ktype, ktype)
        print(f"    {name}: {count}")

    print(f"\n  优先级 Top-3:")
    for i, m in enumerate(report.priority_ranking[:3], 1):
        print(f"    [{i}] {m.label_l3}/{m.label_l4}: {m.priority_score:.1f}分 [{m.kano_name}]")

    # 验证：基本型负面应排第一
    top = report.priority_ranking[0]
    assert top.kano_type == "must_be" and top.sentiment_intensity < 0, (
        "基本型负面应优先"
    )

    # 集成接口测试
    print("\n--- AutoTag 集成接口测试 ---")
    predictions = [
        {"l3": "质量", "l4": "漏尿"},
        {"l3": "服务", "l4": "赠品"},
        {"l3": "物流", "l4": "配送时效"},
    ]
    intensities = [
        IntensityResult(text="", raw_sentiment=-1, intensity=-4.0,
                        intensity_level="extreme_neg", confidence=0.9),
        IntensityResult(text="", raw_sentiment=1, intensity=+3.5,
                        intensity_level="strong_pos", confidence=0.8),
        IntensityResult(text="", raw_sentiment=-1, intensity=-1.5,
                        intensity_level="moderate_neg", confidence=0.6),
    ]
    report2 = bridge_autotag_to_kano(predictions, intensities)
    print(f"  集成报告: {report2.total_items} 条, 优先级最高 = "
          f"{report2.priority_ranking[0].label_l4} "
          f"({report2.priority_ranking[0].priority_score:.1f}分)")

    print("\n" + "=" * 60)
    print("Kano 桥接测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_kano_bridge()
