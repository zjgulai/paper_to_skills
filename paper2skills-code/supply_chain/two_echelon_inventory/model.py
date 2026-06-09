"""
Two-Echelon Inventory DRL — 两级库存 DRL 策略
paper2skills-code: 04-供应链 | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass


@dataclass
class EchelonState:
    echelon_id: str      # "warehouse" 或 "store"
    inventory: float
    backlog: float
    in_transit: float
    demand_mean: float
    lead_time: int


@dataclass
class ReplenishmentAction:
    echelon_id: str
    order_qty: float
    reasoning: str


@dataclass
class SimulationResult:
    total_holding_cost: float
    total_stockout_cost: float
    total_cost: float
    service_level: float
    actions_taken: list[ReplenishmentAction]


class DRLInventoryAgent:
    """
    两级库存 DRL 策略（简化版：规则启发式模拟 DRL 输出）
    生产环境替换为 IPPO/MAPPO 训练的真实策略网络
    """
    def __init__(self, holding_cost: float = 0.5, stockout_cost: float = 10.0,
                 safety_factor: float = 1.65):
        self.h_cost = holding_cost
        self.s_cost = stockout_cost
        self.z = safety_factor  # 95% 服务水平

    def _reorder_point(self, state: EchelonState) -> float:
        demand_std = state.demand_mean * 0.3
        safety_stock = self.z * demand_std * math.sqrt(state.lead_time)
        return state.demand_mean * state.lead_time + safety_stock

    def decide(self, state: EchelonState) -> ReplenishmentAction:
        rop = self._reorder_point(state)
        available = state.inventory + state.in_transit - state.backlog

        if available <= rop:
            target = state.demand_mean * (state.lead_time + 7)
            order_qty = max(0.0, target - available)
            return ReplenishmentAction(
                echelon_id=state.echelon_id,
                order_qty=round(order_qty, 0),
                reasoning=f"库存 {available:.0f} 低于 ROP {rop:.0f}，补货 {order_qty:.0f}",
            )
        return ReplenishmentAction(
            echelon_id=state.echelon_id, order_qty=0.0,
            reasoning=f"库存 {available:.0f} 充足（ROP={rop:.0f}）",
        )


class TwoEchelonSystem:
    """两级库存系统仿真"""
    def __init__(self, warehouse: EchelonState, store: EchelonState):
        self.wh = warehouse
        self.st = store
        self.agent = DRLInventoryAgent()

    def simulate(self, n_periods: int = 30, seed: int = 42) -> SimulationResult:
        random.seed(seed)
        total_h = total_s = 0.0
        fulfilled = total_demand = 0
        actions = []

        for t in range(n_periods):
            demand = random.normalvariate(self.st.demand_mean, self.st.demand_mean * 0.2)
            demand = max(0, demand)
            total_demand += demand

            fulfilled_qty = min(demand, self.st.inventory)
            fulfilled += fulfilled_qty
            stockout = max(0, demand - fulfilled_qty)

            self.st.inventory = max(0, self.st.inventory - demand)
            self.st.backlog += stockout

            h_cost = self.wh.inventory * 0.5 + self.st.inventory * 1.0
            s_cost = stockout * self.agent.s_cost
            total_h += h_cost
            total_s += s_cost

            act = self.agent.decide(self.st)
            if act.order_qty > 0:
                actions.append(act)
                self.wh.inventory -= act.order_qty
                self.st.in_transit += act.order_qty

            wh_act = self.agent.decide(self.wh)
            if wh_act.order_qty > 0:
                self.wh.in_transit += wh_act.order_qty
            self.wh.inventory += self.wh.in_transit / max(self.wh.lead_time, 1)
            self.st.inventory += self.st.in_transit / max(self.st.lead_time, 1)
            self.st.in_transit *= (1 - 1/max(self.st.lead_time, 1))

        sl = fulfilled / max(total_demand, 1)
        return SimulationResult(
            total_holding_cost=round(total_h, 0),
            total_stockout_cost=round(total_s, 0),
            total_cost=round(total_h + total_s, 0),
            service_level=round(sl, 4),
            actions_taken=actions[:5],
        )


def run_two_echelon_demo():
    warehouse = EchelonState("warehouse", 2000, 0, 0, 100, lead_time=14)
    store = EchelonState("store", 300, 0, 0, 50, lead_time=3)

    system = TwoEchelonSystem(warehouse, store)
    result = system.simulate(n_periods=30)

    print("=== 两级库存 DRL 仿真（母婴奶粉 WH→Store）===")
    print(f"持有成本: ¥{result.total_holding_cost:,.0f}")
    print(f"缺货成本: ¥{result.total_stockout_cost:,.0f}")
    print(f"总成本:   ¥{result.total_cost:,.0f}")
    print(f"服务水平: {result.service_level:.1%}")
    print(f"前 3 次补货决策:")
    for a in result.actions_taken[:3]:
        print(f"  [{a.echelon_id}] {a.reasoning}")
    print("✅ 两级库存演示完成")
if __name__ == "__main__":
    run_two_echelon_demo()
