"""Label Function 定义与执行

Label Function 是 ALCHEmist 的核心：一段轻量代码，输入文本，输出标签或弃权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class LabelFunction:
    """单个标注程序

    Attributes:
        name: 程序名称
        func: 标注函数 (text) -> label | None
        description: 程序描述
        coverage: 覆盖率（非弃权比例）
        accuracy: 在验证集上的准确率
        source: 来源 ("llm_generated" | "human_written" | "evolved")
    """

    name: str
    func: Callable[[str], Optional[str]]
    description: str = ""
    coverage: float = 0.0
    accuracy: float = 0.0
    source: str = "llm_generated"

    def __call__(self, text: str) -> Optional[str]:
        """执行标注程序"""
        try:
            return self.func(text)
        except Exception:
            return None  # 执行失败 = 弃权

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        """在验证集上评估程序性能"""
        correct = 0
        covered = 0
        total = len(texts)

        for text, true_label in zip(texts, labels):
            pred = self(text)
            if pred is not None:
                covered += 1
                if pred == true_label:
                    correct += 1

        coverage = covered / total if total > 0 else 0
        accuracy = correct / covered if covered > 0 else 0

        self.coverage = coverage
        self.accuracy = accuracy

        return {
            "coverage": coverage,
            "accuracy": accuracy,
            "total": total,
            "covered": covered,
            "correct": correct,
        }


class LFRegistry:
    """标注程序注册表

    管理多个 label functions，支持按标签分组。
    """

    def __init__(self):
        self.functions: list[LabelFunction] = []
        self._by_label: dict[str, list[LabelFunction]] = {}

    def add(self, lf: LabelFunction) -> None:
        """注册一个标注程序"""
        self.functions.append(lf)
        self._by_label = {}  # 清空缓存

    def get_for_label(self, label: str) -> list[LabelFunction]:
        """获取标注某标签的所有程序"""
        if label not in self._by_label:
            self._by_label[label] = [
                lf for lf in self.functions
                # 通过抽样估计标签关联性
            ]
        return self._by_label.get(label, [])

    def apply_all(self, text: str) -> list[tuple[str, Optional[str]]]:
        """对所有程序应用同一文本，返回 (程序名, 输出) 列表"""
        return [(lf.name, lf(text)) for lf in self.functions]

    def filter_by_accuracy(self, threshold: float = 0.6) -> list[LabelFunction]:
        """过滤掉低准确率程序"""
        return [lf for lf in self.functions if lf.accuracy >= threshold]

    def summary(self) -> dict:
        """注册表统计"""
        return {
            "total_functions": len(self.functions),
            "avg_coverage": sum(lf.coverage for lf in self.functions) / max(len(self.functions), 1),
            "avg_accuracy": sum(lf.accuracy for lf in self.functions) / max(len(self.functions), 1),
            "sources": {
                source: sum(1 for lf in self.functions if lf.source == source)
                for source in set(lf.source for lf in self.functions)
            },
        }


# ── 预设 Label Functions（母婴出海场景）────────────────────────

def _preset_lfs() -> list[LabelFunction]:
    """预定义一些常见母婴产品标签的标注程序"""
    return [
        LabelFunction(
            name="lf_size_keywords",
            func=lambda t: "尺码偏差" if any(kw in t for kw in ["尺码", "大小", "偏大", "偏小", "不合身"]) else None,
            description="关键词匹配：尺码相关词汇",
        ),
        LabelFunction(
            name="lf_size_english",
            func=lambda t: "尺码偏差" if any(kw in t.lower() for kw in ["size", "sizing", "too big", "too small", "tight", "loose"]) else None,
            description="英文关键词：尺码相关",
        ),
        LabelFunction(
            name="lf_material_keywords",
            func=lambda t: "材质问题" if any(kw in t for kw in ["材质", "面料", "硬", "粗糙", "不舒服"]) else None,
            description="关键词匹配：材质相关",
        ),
        LabelFunction(
            name="lf_leak_keywords",
            func=lambda t: "漏尿" if any(kw in t for kw in ["漏", "漏尿", "leak", "渗漏"]) else None,
            description="关键词匹配：漏尿相关",
        ),
        LabelFunction(
            name="lf_allergy_keywords",
            func=lambda t: "过敏反应" if any(kw in t for kw in ["过敏", "红疹", "发红", "rash", "allergy"]) else None,
            description="关键词匹配：过敏相关",
        ),
        LabelFunction(
            name="lf_allergy_body_part",
            func=lambda t: "过敏反应" if (
                any(b in t for b in ["皮肤", "屁股", "大腿", "腰部"])
                and any(r in t for r in ["红", "疹", "肿", "痒"])
            ) else None,
            description="身体部位 + 皮肤反应模式",
        ),
        LabelFunction(
            name="lf_shipping_keywords",
            func=lambda t: "物流延迟" if any(kw in t for kw in ["物流", "快递", "慢", "延迟", "shipping"]) else None,
            description="关键词匹配：物流相关",
        ),
        LabelFunction(
            name="lf_price_keywords",
            func=lambda t: "价格问题" if any(kw in t for kw in ["贵", "便宜", "价格", "expensive", "price"]) else None,
            description="关键词匹配：价格相关",
        ),
    ]


# ── 测试 ──────────────────────────────────────────────────────

def test_label_function():
    """测试 Label Function"""
    print("=" * 60)
    print("测试: LabelFunction")
    print("=" * 60)

    lfs = _preset_lfs()
    registry = LFRegistry()
    for lf in lfs:
        registry.add(lf)

    test_texts = [
        "这个尺码偏小，建议买大一码",
        "腰贴总是粘不住，宝宝一动就开了",
        "物流太慢了，清关等了两周",
        "宝宝用了皮肤发红，是不是过敏了",
        "面料太硬了，摩擦得皮肤不舒服",
    ]

    print("\n--- 单程序标注 ---")
    lf = lfs[0]  # lf_size_keywords
    for text in test_texts:
        result = lf(text)
        status = f"→ {result}" if result else "→ [弃权]"
        print(f"  {lf.name}: '{text}' {status}")

    print("\n--- 全程序投票 ---")
    for text in test_texts:
        votes = registry.apply_all(text)
        non_abstain = [(name, label) for name, label in votes if label is not None]
        print(f"\n  文本: '{text}'")
        for name, label in non_abstain:
            print(f"    {name}: {label}")

    print("\n--- 注册表统计 ---")
    print(registry.summary())

    # 评估准确率
    print("\n--- 验证集评估 ---")
    val_texts = [
        "这个尺码偏小",
        "面料太硬了",
        "晚上漏尿严重",
        "宝宝皮肤过敏了",
        "物流太慢",
        "质量很好",  # 无匹配，应该弃权
    ]
    val_labels = ["尺码偏差", "材质问题", "漏尿", "过敏反应", "物流延迟", None]

    for lf in lfs[:3]:
        stats = lf.evaluate(val_texts, val_labels)
        print(f"  {lf.name}: 覆盖率={stats['coverage']:.1%}, 准确率={stats['accuracy']:.1%}")

    print("\n" + "=" * 60)
    print("Label Function 测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_label_function()
