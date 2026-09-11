"""Shopping Companion: 长期偏好记忆驱动的购物 Agent.

参考论文:Yu, Z. et al. (2026) Shopping Companion: A Memory-Augmented LLM Agent
for Real-World E-Commerce Tasks. arxiv:2603.14864.

本实现是简化版:
- LTM 检索用 token 集合 Jaccard 替代真实 embedding cosine
- Product 检索用关键词匹配替代 BM25
- 偏好提取用规则替代 LLM
- Reward 函数符合论文公式,可用于 offline 评估真实 trajectory

生产环境:embedding 接入 all-MiniLM-L6-v2 / E5-multilingual;Product 接入 Pyserini;
偏好抽取与 reward 接入 GPT-5 / Claude / Qwen3-Max.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Data structures -----------------------------------------------------------


@dataclass
class ConversationTurn:
    session_id: str
    turn_id: int
    speaker: str  # "user" 或 "assistant"
    content: str
    timestamp: int


@dataclass
class Preference:
    attribute: str  # 例:"brand", "size", "material", "fragrance"
    value: str
    polarity: str  # "prefer" 或 "avoid"
    source_turn: Optional[int] = None


@dataclass
class Product:
    sku: str
    title: str
    brand: str
    price: float
    category: str
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class Instruction:
    text: str
    task_type: str  # "single_product" 或 "add_on_deals"
    budget_remaining: Optional[float] = None
    required_count: Optional[int] = None


@dataclass
class Trajectory:
    instruction: Instruction
    extracted_prefs: list[Preference]
    tool_calls: list[dict]
    final_products: list[Product]
    user_confirmed: bool = False


# Stage 1 prerequisites: memory + preference --------------------------------


class LTMStore:
    """长期记忆库. 简化版:token 集合 Jaccard 替代 embedding cosine."""

    def __init__(self) -> None:
        self.turns: list[ConversationTurn] = []

    def add(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def retrieve(self, query: str, top_k: int = 5) -> list[ConversationTurn]:
        if not self.turns:
            return []
        q_tokens = self._tokenize(query)
        scored = [
            (self._jaccard(q_tokens, self._tokenize(t.content)), t)
            for t in self.turns
        ]
        scored.sort(key=lambda x: -x[0])
        return [t for score, t in scored[:top_k] if score > 0]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        # 英文按词,中文按单字,简化版替代真实 embedding.
        return set(re.findall(r"[a-z0-9]+|[一-鿿]", text.lower()))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


class PreferenceExtractor:
    """从 retrieved turns 提取偏好. 简化版:规则 + 关键词."""

    AVOID_MARKERS = ["过敏", "不能用", "避免", "不喜欢", "allergic", "avoid", "hate"]
    PREFER_MARKERS = ["喜欢", "偏好", "love", "prefer", "always buy"]
    # 简化版规则匹配:value 须为多字符避免误匹配.生产用 LLM 抽取.
    ATTR_KEYWORDS = {
        "material": ["乳胶", "硅胶", "silicone", "latex", "棉", "竹纤维"],
        "fragrance": ["香料", "无香", "fragrance", "scented"],
        "brand": ["pampers", "huggies", "merries", "好奇", "花王"],
    }

    def extract(self, turns: list[ConversationTurn]) -> list[Preference]:
        prefs: list[Preference] = []
        for turn in turns:
            text = turn.content.lower()
            polarity = self._detect_polarity(text)
            if polarity is None:
                continue
            for attr, kws in self.ATTR_KEYWORDS.items():
                for kw in kws:
                    if kw in text:
                        prefs.append(
                            Preference(attribute=attr, value=kw, polarity=polarity, source_turn=turn.turn_id)
                        )
        return self._deduplicate(prefs)

    def _detect_polarity(self, text: str) -> Optional[str]:
        if any(m in text for m in self.AVOID_MARKERS):
            return "avoid"
        if any(m in text for m in self.PREFER_MARKERS):
            return "prefer"
        return None

    @staticmethod
    def _deduplicate(prefs: list[Preference]) -> list[Preference]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[Preference] = []
        for p in prefs:
            key = (p.attribute, p.value, p.polarity)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique


# Stage 2 prerequisites: product retrieval + constraint check ---------------


class ProductIndex:
    def __init__(self, products: list[Product]) -> None:
        self.products = products

    def search(self, query: str, top_k: int = 10) -> list[Product]:
        q_tokens = set(re.findall(r"[a-z0-9]+|[一-鿿]", query.lower()))
        scored = []
        for p in self.products:
            tokens = set(re.findall(r"[a-z0-9]+|[一-鿿]", f"{p.title} {p.category} {p.brand}".lower()))
            overlap = len(q_tokens & tokens)
            if overlap > 0:
                scored.append((overlap, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]


class ConstraintChecker:
    def check(self, product: Product, prefs: list[Preference]) -> tuple[bool, list[str]]:
        violations: list[str] = []
        for pref in prefs:
            attr_value = product.attributes.get(pref.attribute, "").lower()
            value = pref.value.lower()
            # avoid: 仅当 product 显式包含该 value 时 fail
            if pref.polarity == "avoid" and attr_value and value in attr_value:
                violations.append(f"avoid {pref.attribute}={pref.value} but product has it")
            # prefer: product 显式有该 attr 但不含 value 时 fail; product 缺失该 attr 时跳过
            elif pref.polarity == "prefer" and attr_value and value not in attr_value:
                violations.append(f"prefer {pref.attribute}={pref.value} but product lacks it")
        return len(violations) == 0, violations

    def fits_budget(self, products: list[Product], budget: float) -> bool:
        return sum(p.price for p in products) <= budget


# Orchestrator --------------------------------------------------------------


@dataclass
class ShoppingCompanion:
    ltm: LTMStore
    products: ProductIndex
    extractor: PreferenceExtractor = field(default_factory=PreferenceExtractor)
    checker: ConstraintChecker = field(default_factory=ConstraintChecker)

    def stage1_identify_preferences(
        self, instruction: Instruction, user_confirms: bool = True
    ) -> tuple[list[Preference], list[dict]]:
        tool_calls: list[dict] = []
        retrieved = self.ltm.retrieve(instruction.text, top_k=5)
        tool_calls.append({"tool": "memory_search", "query": instruction.text, "n_results": len(retrieved)})
        prefs = self.extractor.extract(retrieved)
        tool_calls.append({"tool": "preference_extraction", "n_prefs": len(prefs)})
        if not user_confirms:
            prefs = []  # 模拟用户拒绝偏好
        return prefs, tool_calls

    def stage2_shop(
        self,
        instruction: Instruction,
        prefs: list[Preference],
    ) -> tuple[list[Product], list[dict]]:
        tool_calls: list[dict] = []
        candidates = self.products.search(instruction.text, top_k=20)
        tool_calls.append({"tool": "product_search", "query": instruction.text, "n_results": len(candidates)})
        valid: list[Product] = []
        for product in candidates:
            ok, _ = self.checker.check(product, prefs)
            tool_calls.append({"tool": "constraint_check", "sku": product.sku, "passed": ok})
            if ok:
                valid.append(product)
        if instruction.task_type == "add_on_deals" and instruction.budget_remaining:
            valid = self._select_bundle(valid, instruction.budget_remaining, instruction.required_count or 1)
            tool_calls.append({"tool": "bundle_selector", "n_selected": len(valid)})
        else:
            valid = valid[:1]  # 单品任务返回 top 1
        tool_calls.append({"tool": "recommendation_output", "n_products": len(valid)})
        return valid, tool_calls

    @staticmethod
    def _select_bundle(candidates: list[Product], budget: float, min_count: int) -> list[Product]:
        # 贪心:按价格降序选,直到接近预算或达到件数
        sorted_cand = sorted(candidates, key=lambda p: -p.price)
        bundle: list[Product] = []
        total = 0.0
        for p in sorted_cand:
            if total + p.price <= budget:
                bundle.append(p)
                total += p.price
            if len(bundle) >= min_count and total >= budget * 0.85:
                break
        return bundle

    def run(self, instruction: Instruction, user_confirms: bool = True) -> Trajectory:
        prefs, calls_1 = self.stage1_identify_preferences(instruction, user_confirms)
        products, calls_2 = self.stage2_shop(instruction, prefs)
        return Trajectory(
            instruction=instruction,
            extracted_prefs=prefs,
            tool_calls=calls_1 + calls_2,
            final_products=products,
            user_confirmed=user_confirms,
        )


# Dual reward (offline evaluator) -------------------------------------------


@dataclass
class DualReward:
    """符合论文 R_1 + R_2 + R_tool + R_fmt 的离线评估器."""

    def evaluate(
        self,
        trajectory: Trajectory,
        reference_prefs: list[Preference],
        reference_products: list[Product],
    ) -> dict[str, float]:
        b = 1.0 if trajectory.instruction.task_type == "add_on_deals" else 0.0
        F = max(len(reference_prefs), 1)
        N = max(trajectory.instruction.required_count or 1, 1)

        q1, m1, c1 = self._stage1_signals(trajectory, reference_prefs, b)
        R1 = (q1 + m1 + b * c1) / (1.0 + F + b * N)

        p2, q2, m2, n2, u2 = self._stage2_signals(trajectory, reference_prefs, reference_products, b)
        R2 = (p2 + q2 + m2 + b * (n2 + u2)) / (2.0 + F + b * (N + 1.0))

        R_tool = self._tool_reward(trajectory)
        R_fmt = self._format_reward(trajectory)

        R_total = (R1 if not b else 0) + (R2 if b else R2) + R_tool + R_fmt
        return {"R1": R1, "R2": R2, "R_tool": R_tool, "R_fmt": R_fmt, "R_total": R_total}

    @staticmethod
    def _stage1_signals(traj: Trajectory, ref_prefs: list[Preference], b: float) -> tuple[float, float, float]:
        q1 = 1.0 if any(call.get("tool") == "memory_search" for call in traj.tool_calls) else 0.0
        matched = sum(
            1 for rp in ref_prefs
            if any(ep.attribute == rp.attribute and ep.polarity == rp.polarity for ep in traj.extracted_prefs)
        )
        m1 = matched
        c1 = 1.0 if b and traj.final_products else 0.0
        return q1, m1, c1

    @staticmethod
    def _stage2_signals(
        traj: Trajectory,
        ref_prefs: list[Preference],
        ref_products: list[Product],
        b: float,
    ) -> tuple[float, float, float, float, float]:
        p2 = 1.0 if traj.final_products else 0.0
        ref_skus = {p.sku for p in ref_products}
        relevant = any(p.sku in ref_skus for p in traj.final_products)
        q2 = 1.0 if relevant else 0.0
        all_match = all(
            ConstraintChecker().check(p, ref_prefs)[0] for p in traj.final_products
        )
        m2 = 1.0 if all_match and traj.final_products else 0.0
        if b:
            n2 = 1.0 if traj.instruction.required_count and len(traj.final_products) >= traj.instruction.required_count else 0.0
            total = sum(p.price for p in traj.final_products)
            u2 = 1.0 if traj.instruction.budget_remaining and total <= traj.instruction.budget_remaining else 0.0
        else:
            n2 = u2 = 0.0
        return p2, q2, m2, n2, u2

    @staticmethod
    def _tool_reward(traj: Trajectory) -> float:
        if not traj.tool_calls:
            return 0.0
        valid_calls = sum(1 for c in traj.tool_calls if c.get("tool") and "n_results" not in c or c.get("n_results", 1) > 0)
        return valid_calls / len(traj.tool_calls)

    @staticmethod
    def _format_reward(traj: Trajectory) -> float:
        f_ans = 1.0 if traj.final_products else 0.0
        f_th = 1.0  # 简化版假设 thinking tags 完整
        f_tc = 1.0 if all("tool" in c for c in traj.tool_calls) else 0.0
        f_rec = 1.0 if traj.final_products else 0.0
        return (f_ans + f_th + f_tc + f_rec) / 4.0


# Demo & tests --------------------------------------------------------------


def _demo_ltm() -> LTMStore:
    ltm = LTMStore()
    ltm.add(ConversationTurn("s1", 1, "user", "宝宝乳胶过敏,我们不能用乳胶制品", 1700000000))
    ltm.add(ConversationTurn("s1", 2, "assistant", "明白,会避免乳胶材质", 1700000060))
    ltm.add(ConversationTurn("s2", 1, "user", "上次买的XX牌奶嘴不喜欢,口感太硬", 1700100000))
    ltm.add(ConversationTurn("s3", 1, "user", "宝宝 4 个月了", 1700200000))
    ltm.add(ConversationTurn("s4", 1, "user", "湿巾要无香的,避免香料", 1700300000))
    ltm.add(ConversationTurn("s5", 1, "user", "我喜欢 Pampers 品牌的纸尿裤", 1700400000))
    return ltm


def _demo_products() -> list[Product]:
    return [
        Product(
            sku="P001", title="硅胶安抚奶嘴 4-6M", brand="Brand A", price=15.99,
            category="安抚奶嘴", attributes={"material": "硅胶 silicone", "size": "m"},
        ),
        Product(
            sku="P002", title="乳胶安抚奶嘴 4-6M", brand="Brand B", price=12.99,
            category="安抚奶嘴", attributes={"material": "乳胶 latex", "size": "m"},
        ),
        Product(
            sku="P003", title="无香湿巾 80 片", brand="Huggies", price=8.99,
            category="湿巾", attributes={"fragrance": "无香 fragrance-free"},
        ),
        Product(
            sku="P004", title="Pampers 纸尿裤 M 64 片", brand="Pampers", price=25.99,
            category="纸尿裤", attributes={"brand": "pampers", "size": "m"},
        ),
        Product(
            sku="P005", title="Huggies 纸尿裤 M 56 片", brand="Huggies", price=22.99,
            category="纸尿裤", attributes={"brand": "huggies", "size": "m"},
        ),
    ]


def _demo_single_product() -> None:
    print("=== Demo 1: 单品推荐(安抚奶嘴) ===")
    companion = ShoppingCompanion(ltm=_demo_ltm(), products=ProductIndex(_demo_products()))
    traj = companion.run(Instruction(text="帮我推荐一个安抚奶嘴", task_type="single_product"))
    print(f"提取偏好: {[(p.attribute, p.value, p.polarity) for p in traj.extracted_prefs]}")
    print(f"推荐产品: {[(p.sku, p.title) for p in traj.final_products]}")
    print(f"工具调用次数: {len(traj.tool_calls)}")


def _demo_add_on_deals() -> None:
    print("\n=== Demo 2: 凑单(纸尿裤 + 湿巾) ===")
    companion = ShoppingCompanion(ltm=_demo_ltm(), products=ProductIndex(_demo_products()))
    traj = companion.run(
        Instruction(
            text="帮我凑够 35 美元的纸尿裤 湿巾",
            task_type="add_on_deals",
            budget_remaining=35.0,
            required_count=2,
        )
    )
    print(f"提取偏好: {[(p.attribute, p.value, p.polarity) for p in traj.extracted_prefs]}")
    print(f"凑单结果: {[(p.sku, p.brand, p.price) for p in traj.final_products]}")
    print(f"总价: ${sum(p.price for p in traj.final_products):.2f}")


def _demo_reward() -> None:
    print("\n=== Demo 3: Dual Reward 离线评估 ===")
    companion = ShoppingCompanion(ltm=_demo_ltm(), products=ProductIndex(_demo_products()))
    traj = companion.run(Instruction(text="帮我推荐一个安抚奶嘴", task_type="single_product"))
    ref_prefs = [Preference("material", "乳胶", "avoid"), Preference("size", "m", "prefer")]
    ref_products = [_demo_products()[0]]  # P001 silicone is correct
    reward = DualReward().evaluate(traj, ref_prefs, ref_products)
    for k, v in reward.items():
        print(f"  {k}: {v:.3f}")


def test_pipeline() -> None:
    companion = ShoppingCompanion(ltm=_demo_ltm(), products=ProductIndex(_demo_products()))
    traj = companion.run(Instruction(text="帮我推荐一个安抚奶嘴", task_type="single_product"))
    assert traj.extracted_prefs, "must extract at least one preference"
    assert any(p.polarity == "avoid" for p in traj.extracted_prefs), "should detect 'avoid 乳胶' from LTM"
    assert traj.final_products, "must return at least one product"
    avoided_sku = "P002"  # latex product
    assert all(p.sku != avoided_sku for p in traj.final_products), "must filter out latex product"

    bundle_traj = companion.run(
        Instruction(
            text="凑够 35 的纸尿裤 湿巾",
            task_type="add_on_deals",
            budget_remaining=35.0,
            required_count=2,
        )
    )
    total = sum(p.price for p in bundle_traj.final_products)
    assert total <= 35.0, "bundle must respect budget"

    reward = DualReward().evaluate(
        traj,
        [Preference("material", "乳胶", "avoid"), Preference("size", "m", "prefer")],
        [_demo_products()[0]],
    )
    assert 0.0 <= reward["R_total"] <= 5.0, "reward should be in reasonable range"
    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    _demo_single_product()
    _demo_add_on_deals()
    _demo_reward()
