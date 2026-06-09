"""
AgenticPay — LLM 多 Agent 采购谈判框架
arXiv:2602.06008 | Python 3.14+ | 仅标准库
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NegotiationOffer:
    """单轮谈判报价"""
    price: float
    moq: int
    delivery_days: int
    payment_terms: int     # NET days
    round_num: int
    party: str             # "buyer" or "seller"
    rationale: str = ""


@dataclass
class BATNA:
    """最佳替代方案（谈判底线）"""
    walk_away_price: float
    alternative_supplier: str
    min_moq: int = 0
    max_moq: int = 999999


@dataclass
class NegotiationResult:
    """谈判结果"""
    success: bool
    final_offer: Optional[NegotiationOffer]
    total_rounds: int
    history: list[NegotiationOffer] = field(default_factory=list)
    failure_reason: str = ""


class BuyerAgent:
    """Buyer Agent：保守启动 → 逐步让步，坚守 BATNA 策略"""

    def __init__(self, batna: BATNA, initial_offer_factor: float = 0.75,
                 concession_decay: float = 0.3):
        self.batna = batna
        self.initial_offer_factor = initial_offer_factor
        self.concession_decay = concession_decay
        self._last_offer: Optional[NegotiationOffer] = None

    def generate_offer(self, round_num: int, product: str) -> NegotiationOffer:
        if round_num == 1:
            price = self.batna.walk_away_price * self.initial_offer_factor
        else:
            last_price = self._last_offer.price if self._last_offer else (
                self.batna.walk_away_price * self.initial_offer_factor
            )
            gap = self.batna.walk_away_price - last_price
            step = gap * (1 - math.exp(-self.concession_decay * round_num))
            price = min(self.batna.walk_away_price, last_price + step * 0.4)

        moq = max(self.batna.min_moq, 500)
        offer = NegotiationOffer(
            price=round(price, 2),
            moq=moq,
            delivery_days=30,
            payment_terms=45,
            round_num=round_num,
            party="buyer",
            rationale=f"基于替代价格 {self.batna.walk_away_price}，本轮报价 {price:.2f}",
        )
        self._last_offer = offer
        return offer

    def evaluate_counter_offer(self, offer: NegotiationOffer) -> bool:
        price_ok = offer.price <= self.batna.walk_away_price
        moq_ok = offer.moq <= self.batna.max_moq
        return price_ok and moq_ok


class SellerAgent:
    """Seller Agent：成本底线保护，递减让步"""

    def __init__(self, cost_floor: float, target_margin: float = 0.15,
                 initial_markup: float = 0.35, concession_decay: float = 0.25):
        self.cost_floor = cost_floor
        self.target_margin = target_margin
        self.min_price = cost_floor * (1 + target_margin)
        self.initial_markup = initial_markup
        self.concession_decay = concession_decay
        self._last_offer: Optional[NegotiationOffer] = None

    def respond_to_offer(self, buyer_offer: NegotiationOffer) -> NegotiationOffer:
        round_num = buyer_offer.round_num

        if round_num == 1:
            price = self.min_price * (1 + self.initial_markup)
        else:
            last_price = self._last_offer.price if self._last_offer else (
                self.min_price * (1 + self.initial_markup)
            )
            gap = last_price - self.min_price
            step = gap * math.exp(-self.concession_decay * round_num) * 0.3
            price = max(self.min_price, last_price - step)

        moq = max(500, buyer_offer.moq + 200)
        offer = NegotiationOffer(
            price=round(price, 2),
            moq=moq,
            delivery_days=25,
            payment_terms=30,
            round_num=round_num,
            party="seller",
            rationale=f"成本 {self.cost_floor}，目标毛利 {self.target_margin:.0%}，本轮 {price:.2f}",
        )
        self._last_offer = offer
        return offer

    def evaluate_buyer_offer(self, offer: NegotiationOffer) -> bool:
        return offer.price >= self.min_price and offer.moq >= 400


class MediatorAgent:
    """Mediator Agent：ZOPA 区间计算 + 折中建议"""

    def estimate_zopa(self, buyer: BuyerAgent, seller: SellerAgent) -> tuple[float, float]:
        return (seller.min_price, buyer.batna.walk_away_price)

    def suggest_compromise(self, buyer_offer: NegotiationOffer,
                           seller_offer: NegotiationOffer,
                           buyer: BuyerAgent,
                           seller: SellerAgent) -> NegotiationOffer:
        zopa_low, zopa_high = self.estimate_zopa(buyer, seller)

        if zopa_low > zopa_high:
            return NegotiationOffer(
                price=buyer_offer.price,
                moq=buyer_offer.moq,
                delivery_days=28,
                payment_terms=38,
                round_num=buyer_offer.round_num,
                party="mediator",
                rationale="ZOPA 不存在，建议终止谈判",
            )

        compromise_price = round((zopa_low + zopa_high) / 2, 2)
        compromise_moq = (buyer_offer.moq + seller_offer.moq) // 2
        return NegotiationOffer(
            price=compromise_price,
            moq=compromise_moq,
            delivery_days=28,
            payment_terms=38,
            round_num=buyer_offer.round_num,
            party="mediator",
            rationale=f"ZOPA [{zopa_low:.2f}, {zopa_high:.2f}]，折中建议 {compromise_price:.2f}",
        )


class NegotiationSession:
    """谈判会话：驱动三方 Agent 完成多轮谈判"""

    def __init__(self, buyer: BuyerAgent, seller: SellerAgent,
                 mediator: MediatorAgent, max_rounds: int = 8,
                 stall_threshold: int = 3):
        self.buyer = buyer
        self.seller = seller
        self.mediator = mediator
        self.max_rounds = max_rounds
        self.stall_threshold = stall_threshold

    def run_negotiation(self, product: str = "采购商品") -> NegotiationResult:
        history: list[NegotiationOffer] = []
        stall_counter = 0
        last_gap = float("inf")

        print(f"\n[AgenticPay] 开始谈判: {product}")
        print(f"买家 BATNA: {self.buyer.batna.walk_away_price}，"
              f"卖家底线: {self.seller.min_price:.2f}")

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n── 第 {round_num} 轮 ──")

            buyer_offer = self.buyer.generate_offer(round_num, product)
            history.append(buyer_offer)
            print(f"  买家报价: ¥{buyer_offer.price} | MOQ={buyer_offer.moq}")

            if self.seller.evaluate_buyer_offer(buyer_offer):
                print("  ✅ 卖家接受买家报价！")
                return NegotiationResult(
                    success=True,
                    final_offer=buyer_offer,
                    total_rounds=round_num,
                    history=history,
                )

            seller_offer = self.seller.respond_to_offer(buyer_offer)
            history.append(seller_offer)
            print(f"  卖家还价: ¥{seller_offer.price} | MOQ={seller_offer.moq}")

            if self.buyer.evaluate_counter_offer(seller_offer):
                print("  ✅ 买家接受卖家还价！")
                return NegotiationResult(
                    success=True,
                    final_offer=seller_offer,
                    total_rounds=round_num,
                    history=history,
                )

            current_gap = seller_offer.price - buyer_offer.price
            print(f"  价差: ¥{current_gap:.2f}")
            if current_gap >= last_gap * 0.95:
                stall_counter += 1
            else:
                stall_counter = 0
            last_gap = current_gap

            if stall_counter >= self.stall_threshold:
                print(f"  🔔 连续 {stall_counter} 轮停滞，Mediator 介入...")
                compromise = self.mediator.suggest_compromise(
                    buyer_offer, seller_offer, self.buyer, self.seller
                )
                history.append(compromise)
                print(f"  Mediator: ¥{compromise.price} | {compromise.rationale}")

                if (self.buyer.evaluate_counter_offer(compromise)
                        and self.seller.evaluate_buyer_offer(compromise)):
                    return NegotiationResult(
                        success=True,
                        final_offer=compromise,
                        total_rounds=round_num,
                        history=history,
                    )

        return NegotiationResult(
            success=False,
            final_offer=None,
            total_rounds=self.max_rounds,
            history=history,
            failure_reason=f"超过最大轮次 {self.max_rounds}，未达成协议",
        )


def test_baby_formula_negotiation():
    """测试：奶粉供应商谈判，验证在 BATNA 范围内达成协议"""
    print("=" * 60)
    print("AgenticPay 谈判测试：母婴奶粉供应商 MOQ/价格谈判")
    print("=" * 60)

    buyer = BuyerAgent(
        batna=BATNA(
            walk_away_price=120.0,
            alternative_supplier="B厂（¥118，MOQ=600）",
            min_moq=100,
            max_moq=700,
        ),
        initial_offer_factor=0.75,
        concession_decay=0.3,
    )
    seller = SellerAgent(
        cost_floor=95.0,
        target_margin=0.15,
        initial_markup=0.35,
        concession_decay=0.25,
    )
    mediator = MediatorAgent()

    session = NegotiationSession(
        buyer=buyer,
        seller=seller,
        mediator=mediator,
        max_rounds=8,
        stall_threshold=3,
    )
    result = session.run_negotiation("A2 有机婴儿配方奶粉 900g")

    print("\n── 谈判结果 ──")
    print(f"成功: {result.success}")
    print(f"总轮次: {result.total_rounds}")
    if result.final_offer:
        print(f"最终价格: ¥{result.final_offer.price}")
        print(f"最终 MOQ: {result.final_offer.moq}")

    if result.success:
        assert result.final_offer is not None
        assert result.final_offer.price <= buyer.batna.walk_away_price, \
            f"最终价格 {result.final_offer.price} 超过买家 BATNA {buyer.batna.walk_away_price}"
        assert result.final_offer.price >= seller.min_price, \
            f"最终价格 {result.final_offer.price} 低于卖家底线 {seller.min_price}"
        assert result.total_rounds <= 8, f"轮次 {result.total_rounds} 超过预期"
        print("\n✅ 测试通过：在 BATNA 范围内达成协议")
    else:
        print(f"\n⚠️  谈判失败: {result.failure_reason}")


if __name__ == "__main__":
    test_baby_formula_negotiation()
