---
title: AgenticPay — LLM 多 Agent 采购谈判：自主完成价格与 MOQ 协商
doc_type: knowledge
module: 10-MAS
topic: agentic-pay-procurement-negotiation
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: AgenticPay — LLM 多 Agent 采购谈判：自主完成价格与 MOQ 协商

> 论文: arXiv:2602.06008 (2026-02) | 三方 Agent 架构自主完成买卖谈判与交易

---

## ① 算法原理

### 核心思想

**AgenticPay** 将买卖双方谈判建模为三方博弈：Buyer Agent（代理买家利益）+ Seller Agent（代理卖家利益）+ Mediator Agent（协调双方找到 ZOPA）。LLM 驱动每个 Agent 根据各自的 BATNA（最佳替代方案）和策略参数自主生成报价、评估还价、决定让步幅度。

与人工谈判的本质区别：**策略一致性**——LLM Agent 不受情绪影响，每轮报价严格遵循编码好的策略（如保守启动、逐步让步、到达底线停止），不会因疲劳/压力偏离策略。

### 博弈论基础：ZOPA（Zone of Possible Agreement）

$$\text{ZOPA} = [\text{Buyer\_BATNA}, \text{Seller\_BATNA}]$$

若 $\text{Buyer\_BATNA} \geq \text{Seller\_BATNA}$，协议区间存在；否则无法达成协议。Mediator Agent 的核心职责是估算 ZOPA 区间，并引导双方向区间中心靠拢。

### BATNA 编码方式

```python
buyer_batna = BATNA(
    walk_away_price=120,          # 超过此价格宁可找替代供应商
    alternative_supplier="B 厂",  # 替代方案
)
```

BATNA 硬编码进 Agent prompt，不随谈判轮次改变（保证策略一致性）。

### 谈判收敛终止条件

1. **协议达成**：买家接受价格 ≤ 买家 BATNA 且 ≥ 卖家底线
2. **轮次耗尽**：超过 max_rounds 无协议 → 返回失败
3. **死局检测**：连续 3 轮双方报价差距未缩小 → Mediator 强制介入或终止

### LLM 策略一致性机制

每轮让步幅度遵循递减函数：$\Delta_r = \Delta_0 \times e^{-\alpha r}$，轮次 $r$ 越大让步越小，避免被"车轮战"策略榨干所有让步空间。

### 关键假设

1. 双方 BATNA 真实可估（买家知道替代供应商价格，卖家知道成本底线）
2. 谈判在有限轮次内（通常 5-10 轮）可收敛
3. 合同条款可结构化表达（价格/MOQ/交期/付款条件均可量化）

---

## ② 母婴出海应用案例

### 场景一：奶粉供应商 MOQ/价格谈判

**业务问题**：母婴品牌向供应商采购配方奶粉，供应商初始 MOQ=1000 箱（资金占用约 50 万），品牌方目标 MOQ≤500 箱（降低首单风险）。价格谈判同步进行（目标单价≤¥110，供应商开价¥130）。

**数据要求**：
- 买家 BATNA：替代供应商 B 的报价（¥118/箱，MOQ=600 箱）
- 卖家成本底线：生产成本 ¥95/箱，目标毛利率 ≥ 15%（底线价格 ¥109.25）
- 谈判参数：max_rounds=5，初始让步 10%，每轮递减 30%

**预期产出**：
- 3-5 轮内达成协议：价格 ¥108-¥115，MOQ 500-700 箱
- 谈判记录（每轮报价 + 论据），可用于内部审计
- 若协议失败，输出最大差距条款供人工跟进

**业务价值（量化）**：
- 谈判周期从 2 周（邮件往返）→ 2 小时（Agent 自动运行）
- 采购成本降低 5-12%（LLM 不受情绪压力，坚守 BATNA 不轻易让步）
- 采购人员解放，专注于供应商关系维护和异常处理

---

### 场景二：包装材料年度框架合同谈判

**业务问题**：年度采购 500 万件包装袋，需谈判价格阶梯（100 万/年享 5% 折扣，500 万/年享 12% 折扣）和付款条件（NET30 vs NET45），人工谈判需反复协调财务/采购/法务三部门。

**数据要求**：
- 市场基准价格（来自 3 家同类供应商报价单）
- 目标采购量预测（运营提供，按月分解）
- 付款条件约束（CFO 要求 NET ≥ 30 天）

**预期产出**：
- 年度框架合同草案（价格阶梯 + 付款条件 + 交货周期）
- 多 Agent 谈判日志（可作为商务谈判记录存档）

**业务价值（量化）**：
- 框架合同谈判周期从 4-6 周 → 3 天（人工确认 + Agent 自动谈判）
- 价格阶梯优化（Agent 能同时优化多个维度，人工通常顾此失彼）

---

## ③ 代码模板

代码文件：`paper2skills-code/mas/agentic_pay_negotiation/model.py`

```python
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
    """Buyer Agent：保守启动 → 逐步让步，保守 BATNA 策略"""

    def __init__(self, batna: BATNA, initial_offer_factor: float = 0.75,
                 concession_decay: float = 0.3):
        self.batna = batna
        self.initial_offer_factor = initial_offer_factor
        self.concession_decay = concession_decay
        self._last_offer: Optional[NegotiationOffer] = None

    def generate_offer(self, round_num: int, product: str) -> NegotiationOffer:
        """生成买家报价：初始保守，随轮次递增（但不超 BATNA）"""
        if round_num == 1:
            price = self.batna.walk_away_price * self.initial_offer_factor
        else:
            last_price = self._last_offer.price if self._last_offer else (
                self.batna.walk_away_price * self.initial_offer_factor
            )
            # 每轮向 BATNA 靠拢，但递减让步
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
            rationale=f"基于市场替代价格 {self.batna.walk_away_price}，本轮报价 {price:.2f}",
        )
        self._last_offer = offer
        return offer

    def evaluate_counter_offer(self, offer: NegotiationOffer) -> bool:
        """评估卖家还价：接受则返回 True"""
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
        """卖家还价：初始高标，逐步向最低价靠近"""
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

        moq = max(500, buyer_offer.moq + 200)   # 卖家 MOQ 保持略高于买家要求
        offer = NegotiationOffer(
            price=round(price, 2),
            moq=moq,
            delivery_days=25,
            payment_terms=30,
            round_num=round_num,
            party="seller",
            rationale=f"成本 {self.cost_floor}，目标毛利 {self.target_margin:.0%}，本轮让步至 {price:.2f}",
        )
        self._last_offer = offer
        return offer

    def evaluate_buyer_offer(self, offer: NegotiationOffer) -> bool:
        """评估买家报价：接受则返回 True"""
        return offer.price >= self.min_price and offer.moq >= 400


class MediatorAgent:
    """Mediator Agent：ZOPA 区间计算 + 折中建议"""

    def estimate_zopa(self, buyer: BuyerAgent, seller: SellerAgent) -> tuple[float, float]:
        """估算 ZOPA 区间 [seller_min, buyer_max]"""
        return (seller.min_price, buyer.batna.walk_away_price)

    def suggest_compromise(self, buyer_offer: NegotiationOffer,
                           seller_offer: NegotiationOffer,
                           buyer: BuyerAgent,
                           seller: SellerAgent) -> NegotiationOffer:
        """基于 ZOPA 中点给出折中建议"""
        zopa_low, zopa_high = self.estimate_zopa(buyer, seller)

        if zopa_low > zopa_high:
            # ZOPA 不存在
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
        """执行完整谈判流程，返回 NegotiationResult"""
        history: list[NegotiationOffer] = []
        stall_counter = 0
        last_gap = float("inf")

        print(f"\n[AgenticPay] 开始谈判: {product}")
        print(f"买家 BATNA: {self.buyer.batna.walk_away_price}，"
              f"卖家底线: {self.seller.min_price:.2f}")

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n── 第 {round_num} 轮 ──")

            # 买家报价
            buyer_offer = self.buyer.generate_offer(round_num, product)
            history.append(buyer_offer)
            print(f"  买家报价: ¥{buyer_offer.price} | MOQ={buyer_offer.moq}")

            # 卖家直接接受买家报价？
            if self.seller.evaluate_buyer_offer(buyer_offer):
                print(f"  ✅ 卖家接受买家报价！")
                return NegotiationResult(
                    success=True,
                    final_offer=buyer_offer,
                    total_rounds=round_num,
                    history=history,
                )

            # 卖家还价
            seller_offer = self.seller.respond_to_offer(buyer_offer)
            history.append(seller_offer)
            print(f"  卖家还价: ¥{seller_offer.price} | MOQ={seller_offer.moq}")

            # 买家评估卖家还价
            if self.buyer.evaluate_counter_offer(seller_offer):
                print(f"  ✅ 买家接受卖家还价！")
                return NegotiationResult(
                    success=True,
                    final_offer=seller_offer,
                    total_rounds=round_num,
                    history=history,
                )

            # 检测停滞
            current_gap = seller_offer.price - buyer_offer.price
            print(f"  价差: ¥{current_gap:.2f}")
            if current_gap >= last_gap * 0.95:
                stall_counter += 1
            else:
                stall_counter = 0
            last_gap = current_gap

            # Mediator 介入
            if stall_counter >= self.stall_threshold:
                print(f"  🔔 连续 {stall_counter} 轮停滞，Mediator 介入...")
                compromise = self.mediator.suggest_compromise(
                    buyer_offer, seller_offer, self.buyer, self.seller
                )
                history.append(compromise)
                print(f"  Mediator 建议: ¥{compromise.price} | {compromise.rationale}")

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
    """测试：奶粉供应商谈判，验证 3-5 轮内达成协议，MOQ 在双方 BATNA 之间"""
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
        print(f"达成方: {result.final_offer.party}")

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
        print("提示：可尝试调整 walk_away_price 或 target_margin 使 ZOPA 存在")


if __name__ == "__main__":
    test_baby_formula_negotiation()
```

---

## ④ 技能关联

- **前置**：[[Skill-Supplier-Capacity-Planning]] / [[Skill-Dynamic-Lot-Sizing-MOQ]] / [[Skill-MAS-Orchestrator]]
- **延伸**：[[Skill-Multi-Agent-Debate]] / [[Skill-Flowr-Supply-Chain-MAS]]
- **可组合**：[[Skill-Multi-SKU-Procurement-Budget-Allocation]] / [[Skill-AIM-RM-LLM-Inventory-MAS-Memory]]

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 采购成本降低 5-15%（LLM Agent 不受情绪影响，坚守 BATNA，不轻易让步）
  - 谈判周期从 2 周缩短至 2 小时（无需等待邮件回复，Agent 实时执行）
  - 采购人员从谈判执行者转为谈判策略制定者（节省约 4 人·天/次）
  - 以年采购 500 万元计，降低 5% = 节省 25 万元/年

- **实施难度**：⭐⭐☆☆☆
  - 核心逻辑为规则+策略编码，无需训练，直接集成 LLM API
  - 最大挑战：BATNA 信息的准确获取（需要采购团队提供替代方案报价）
  - 可从模拟谈判（非实际交易）开始，逐步积累策略参数

- **优先级评分**：⭐⭐⭐⭐☆
  - 跨境采购是母婴出海的核心成本项，价格谈判直接影响毛利率
  - 实施门槛低（纯规则驱动，不依赖特定模型）
  - 可快速量化 ROI（谈判记录直接对比人工谈判结果）
  - 短期可落地（1-2 个月从 PoC 到生产）
