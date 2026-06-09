import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BiddingContext:
    keyword: str
    current_bid: float
    market_avg_cpc: float
    budget_remaining: float
    budget_total: float
    target_acos: float
    current_acos: float
    current_roas: float
    history_clicks: int
    history_conversions: int
    time_remaining_hours: float = 24.0


@dataclass
class BiddingDecision:
    keyword: str
    recommended_bid: float
    reasoning: str
    intent: str
    confidence: float
    within_budget: bool


class DARAReasonerAgent:
    def __init__(self, min_history_for_confidence: int = 50):
        self.min_history = min_history_for_confidence

    def reason(self, ctx: BiddingContext) -> Tuple[str, float]:
        data_confidence = min(1.0, ctx.history_clicks / self.min_history)
        acos_gap = ctx.current_acos - ctx.target_acos

        if acos_gap > 0.1:
            direction, magnitude = "reduce", min(0.2, acos_gap)
            reasoning = f"ACOS {ctx.current_acos:.1%} 超目标 {ctx.target_acos:.1%}，降价 {magnitude:.0%}"
        elif acos_gap > 0:
            direction, magnitude = "slight_reduce", 0.05
            reasoning = "ACOS 略高，微降维持竞争力"
        elif ctx.budget_remaining / ctx.budget_total < 0.3:
            direction, magnitude = "reduce", 0.1
            reasoning = "预算剩余不足 30%，降价保存预算"
        else:
            direction, magnitude = "maintain", 0.0
            reasoning = f"ROAS {ctx.current_roas:.1f} 良好，维持当前出价"

        confidence = data_confidence * 0.6 + 0.4
        return f"{direction}|{magnitude}|{reasoning}", confidence


class DARAOptimizerAgent:
    def __init__(self, bid_floor: float = 0.01, bid_ceil_multiplier: float = 3.0):
        self.bid_floor = bid_floor
        self.bid_ceil_multiplier = bid_ceil_multiplier

    def optimize(self, ctx: BiddingContext, reasoning_output: str) -> float:
        parts = reasoning_output.split("|")
        direction = parts[0] if parts else "maintain"
        magnitude = float(parts[1]) if len(parts) > 1 else 0.0

        if direction in ("reduce", "slight_reduce"):
            new_bid = ctx.current_bid * (1.0 - magnitude)
        elif direction == "increase":
            new_bid = ctx.current_bid * (1.0 + magnitude)
        else:
            new_bid = ctx.current_bid * (1.0 + random.uniform(-0.01, 0.01))

        bid_ceil = ctx.market_avg_cpc * self.bid_ceil_multiplier
        return round(max(self.bid_floor, min(new_bid, bid_ceil)), 2)


class DARABiddingSystem:
    def __init__(self):
        self.reasoner = DARAReasonerAgent()
        self.optimizer = DARAOptimizerAgent()

    def decide(self, ctx: BiddingContext) -> BiddingDecision:
        reasoning_output, confidence = self.reasoner.reason(ctx)
        bid = self.optimizer.optimize(ctx, reasoning_output)
        parts = reasoning_output.split("|")
        intent = parts[0]
        reasoning_text = parts[2] if len(parts) > 2 else ""
        return BiddingDecision(
            keyword=ctx.keyword, recommended_bid=bid,
            reasoning=reasoning_text, intent=intent,
            confidence=confidence, within_budget=bid <= ctx.budget_remaining * 0.1,
        )


class LBMThinkAgent:
    def think(self, ctx: BiddingContext) -> Tuple[str, float]:
        budget_ratio = ctx.budget_remaining / max(ctx.budget_total, 1)
        acos_ratio = ctx.current_acos / max(ctx.target_acos, 0.01)

        if acos_ratio > 1.5 or budget_ratio < 0.2:
            return "reduce_aggressive", 0.9
        elif acos_ratio > 1.1:
            return "reduce", 0.8
        elif acos_ratio < 0.8 and budget_ratio > 0.5:
            return "increase", 0.75
        elif acos_ratio < 0.6 and budget_ratio > 0.7:
            return "increase_aggressive", 0.7
        return "maintain", 0.85


class LBMActAgent:
    INTENT_MULTIPLIERS = {
        "reduce_aggressive": (0.7, 0.85),
        "reduce": (0.88, 0.97),
        "maintain": (0.98, 1.02),
        "increase": (1.03, 1.12),
        "increase_aggressive": (1.13, 1.25),
    }

    def __init__(self, bid_floor: float = 0.01):
        self.bid_floor = bid_floor

    def act(self, ctx: BiddingContext, intent: str, confidence: float) -> float:
        lo, hi = self.INTENT_MULTIPLIERS.get(intent, (0.98, 1.02))
        multiplier = lo + (hi - lo) * confidence
        new_bid = ctx.current_bid * multiplier
        max_affordable = ctx.budget_remaining * 0.05
        return round(max(self.bid_floor, min(new_bid, max_affordable, ctx.market_avg_cpc * 2.5)), 2)


class LBMBiddingSystem:
    def __init__(self):
        self.think_agent = LBMThinkAgent()
        self.act_agent = LBMActAgent()

    def decide(self, ctx: BiddingContext) -> BiddingDecision:
        intent, confidence = self.think_agent.think(ctx)
        bid = self.act_agent.act(ctx, intent, confidence)
        return BiddingDecision(
            keyword=ctx.keyword, recommended_bid=bid,
            reasoning=f"LBM-Think: {intent} (conf={confidence:.2f})",
            intent=intent, confidence=confidence,
            within_budget=bid <= ctx.budget_remaining * 0.1,
        )


def make_ctx(acos=0.38, budget_rem=450, budget_total=800, roas=3.2,
             clicks=150, conversions=8, current_bid=0.72):
    return BiddingContext(
        keyword="baby diaper rash cream",
        current_bid=current_bid, market_avg_cpc=0.72,
        budget_remaining=budget_rem, budget_total=budget_total,
        target_acos=0.30, current_acos=acos, current_roas=roas,
        history_clicks=clicks, history_conversions=conversions,
    )


def test_dara_reduces_bid_when_acos_high():
    random.seed(42)
    ctx = make_ctx(acos=0.45, current_bid=0.72)
    system = DARABiddingSystem()
    decision = system.decide(ctx)
    assert decision.recommended_bid < ctx.current_bid, \
        f"Expected bid reduction, got {decision.recommended_bid} vs {ctx.current_bid}"
    assert decision.intent in ("reduce", "slight_reduce")
    print(f"[PASS] dara_reduce: {ctx.current_bid:.2f} → {decision.recommended_bid:.2f}, intent={decision.intent}")


def test_dara_sparse_data_conservative():
    random.seed(42)
    ctx_sparse = make_ctx(acos=0.32, clicks=10)
    ctx_rich = make_ctx(acos=0.32, clicks=200)
    system = DARABiddingSystem()
    d_sparse = system.decide(ctx_sparse)
    d_rich = system.decide(ctx_rich)
    assert d_sparse.confidence < d_rich.confidence, \
        f"Sparse data should have lower confidence: {d_sparse.confidence} vs {d_rich.confidence}"
    print(f"[PASS] dara_sparse: sparse_conf={d_sparse.confidence:.2f} < rich_conf={d_rich.confidence:.2f}")


def test_lbm_no_hallucination():
    ctx = make_ctx(acos=0.25, current_bid=0.95, budget_rem=450)
    system = LBMBiddingSystem()
    decision = system.decide(ctx)
    assert decision.recommended_bid <= ctx.market_avg_cpc * 2.5, \
        f"Bid should not exceed 2.5x market CPC (hallucination): {decision.recommended_bid}"
    assert decision.recommended_bid >= 0.01
    print(f"[PASS] lbm_no_hallucination: bid={decision.recommended_bid:.2f} (market={ctx.market_avg_cpc:.2f})")


def test_lbm_increases_bid_when_roas_good():
    ctx = make_ctx(acos=0.18, roas=5.5, budget_rem=600, budget_total=800, current_bid=0.60)
    system = LBMBiddingSystem()
    decision = system.decide(ctx)
    assert decision.intent in ("increase", "increase_aggressive"), \
        f"Should increase when ROAS high: {decision.intent}"
    print(f"[PASS] lbm_increase: intent={decision.intent}, bid={decision.recommended_bid:.2f}")


def test_lbm_emergency_reduce_when_budget_low():
    ctx = make_ctx(acos=0.31, budget_rem=100, budget_total=800, current_bid=0.95)
    system = LBMBiddingSystem()
    decision = system.decide(ctx)
    assert decision.intent == "reduce_aggressive", \
        f"Should reduce_aggressive when budget <20%: {decision.intent}"
    print(f"[PASS] lbm_emergency: intent={decision.intent}, bid={decision.recommended_bid:.2f}")


if __name__ == "__main__":
    random.seed(42)
    test_dara_reduces_bid_when_acos_high()
    test_dara_sparse_data_conservative()
    test_lbm_no_hallucination()
    test_lbm_increases_bid_when_roas_good()
    test_lbm_emergency_reduce_when_budget_low()
    print("\n✅ All tests passed")
