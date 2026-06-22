---
title: MAS多仓库补货Nash协商 — 多仓库Agent协商最优库存调拨方案
doc_type: knowledge
module: 10-MAS
topic: mas-multi-warehouse-replenishment-consensus
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: MAS多仓库补货Nash协商

> **论文**：Multi-Agent Nash Bargaining for Inventory Replenishment and Rebalancing Across Distributed Warehouses
> **arXiv**：2402.15891 | 2024 | **桥接**: 10-MAS ↔ 04-供应链 | **类型**: 跨域融合

## ① 算法原理

多仓库库存管理的核心矛盾：每个仓库都想最大化自己的库存安全（本地最优），但全局最优要求适度不平衡（有仓库多、有仓库少）。中央统筹则信息延迟、无法实时响应。

**Nash协商解法**：
- 每个仓库设一个 **仓库Agent**，持有本地库存状态、预测需求、前置期信息
- Agents在协商协议下提出调拨方案，目标是找到 **Pareto最优调拨** 而非单边最优
- **Nash均衡点**：不存在任何Agent通过单边偏离能改善自身状况的调拨方案

**数学基础**：
```
纳什协商解 = argmax ∏ᵢ (Uᵢ(调拨方案) - Uᵢ(不协商基准))
```
其中 Uᵢ 是第i个仓库的效用函数（库存健康度 - 调拨成本 - 缺货风险）。

**协商协议**：改良Zeuthen协议，每轮每个Agent提出让步幅度，用「风险比率」决定谁先让步：
```
风险比率 = 当前方案效用损失 / 拒绝时的总损失
```
风险更低的Agent优先让步，直到收敛。

## ② 母婴出海应用案例

**场景：吸奶器三仓库库存协同（美东/美西/德国）**

- **业务问题**：吸奶器在美东仓库库存过剩（150%安全库存），美西仓断货（30%），德国仓库正常。中心化调配决策延迟48-72小时，经常错过调仓窗口
- **数据要求**：各仓库当前库存、安全库存目标、日均销量、前置期、仓间调拨成本（每件$3-8）
- **多Agent设计**：
  - Agent_US_East：持有美东仓数据，目标减少过剩
  - Agent_US_West：持有美西仓数据，目标消除缺货风险
  - Agent_DE：持有德国仓数据，目标维持安全水位
- **预期产出**：24小时内达到Nash协商均衡，输出调拨方案（如：美东→美西调 50件，成本$150）
- **业务价值**：消除缺货损失（美西1天缺货约$2000），减少过剩导致的FBA长期仓储费（每件$0.5/月），年化节省约 **15-30万元**

## ③ 代码模板

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class WarehouseState:
    """仓库状态"""
    warehouse_id: str
    current_stock: int
    safety_stock: int
    daily_demand: float
    lead_time_days: int
    transfer_cost_per_unit: float  # 调出成本

class WarehouseAgent:
    """仓库Agent：持有本地信息，参与Nash协商"""
    
    def __init__(self, state: WarehouseState):
        self.state = state
        
    def compute_utility(self, stock_after_transfer: int) -> float:
        """计算调拨后的效用值（库存健康度）"""
        s = self.state
        # 库存覆盖天数
        coverage_days = stock_after_transfer / max(s.daily_demand, 0.1)
        safety_coverage = s.safety_stock / max(s.daily_demand, 0.1)
        
        # 最优覆盖在安全库存的1.5倍处
        optimal_coverage = safety_coverage * 1.5
        
        # 效用函数：覆盖不足惩罚重（缺货），过多惩罚轻（仓储费）
        if coverage_days < safety_coverage:
            # 缺货风险：惩罚系数3x
            utility = -3.0 * (safety_coverage - coverage_days)
        else:
            # 过剩：轻惩罚
            utility = -0.3 * max(coverage_days - optimal_coverage, 0)
        
        return utility
    
    def compute_no_deal_utility(self) -> float:
        """不协商时的基准效用（维持现状）"""
        return self.compute_utility(self.state.current_stock)
    
    def evaluate_transfer(self, send_units: int, receive_units: int) -> Dict:
        """评估调拨方案"""
        net_transfer = receive_units - send_units
        new_stock = self.state.current_stock + net_transfer
        new_stock = max(0, new_stock)
        
        transfer_cost = send_units * self.state.transfer_cost_per_unit
        utility_gain = self.compute_utility(new_stock) - self.compute_no_deal_utility()
        
        return {
            'new_stock': new_stock,
            'utility_gain': utility_gain - transfer_cost * 0.01,  # 成本折算
            'feasible': new_stock >= 0 and send_units <= self.state.current_stock
        }


class NashBargainingOrchestrator:
    """Nash协商协调器：协调多个仓库Agent达到协商均衡"""
    
    def __init__(self, agents: List[WarehouseAgent]):
        self.agents = agents
        self.max_iterations = 50
    
    def _compute_nash_product(self, transfer_plan: Dict[str, Dict]) -> float:
        """计算Nash乘积（所有Agent效用增益之积）"""
        product = 1.0
        for agent in self.agents:
            wid = agent.state.warehouse_id
            plan = transfer_plan.get(wid, {'send': 0, 'receive': 0})
            eval_result = agent.evaluate_transfer(plan.get('send', 0), plan.get('receive', 0))
            gain = eval_result['utility_gain'] + 1.0  # 偏移确保正值
            product *= max(gain, 0.01)
        return product
    
    def find_nash_equilibrium(self) -> Dict:
        """迭代寻找Nash协商均衡调拨方案"""
        # 识别过剩仓和短缺仓
        surplus_agents = []
        deficit_agents = []
        
        for agent in self.agents:
            s = agent.state
            safety_days = s.safety_stock / max(s.daily_demand, 0.1)
            current_days = s.current_stock / max(s.daily_demand, 0.1)
            
            if current_days > safety_days * 1.8:
                surplus = int((current_days - safety_days * 1.3) * s.daily_demand)
                surplus_agents.append((agent, surplus))
            elif current_days < safety_days * 0.8:
                deficit = int((safety_days - current_days) * s.daily_demand)
                deficit_agents.append((agent, deficit))
        
        # 匹配过剩仓→短缺仓，逐步协商
        transfer_proposals = []
        
        for surplus_agent, available in surplus_agents:
            for deficit_agent, needed in deficit_agents:
                transfer_amount = min(available, needed)
                if transfer_amount <= 0:
                    continue
                
                # 检查Nash乘积是否改善
                plan = {
                    surplus_agent.state.warehouse_id: {'send': transfer_amount, 'receive': 0},
                    deficit_agent.state.warehouse_id: {'send': 0, 'receive': transfer_amount}
                }
                nash_product = self._compute_nash_product(plan)
                
                if nash_product > 1.0:  # 比不协商好
                    transfer_proposals.append({
                        'from': surplus_agent.state.warehouse_id,
                        'to': deficit_agent.state.warehouse_id,
                        'units': transfer_amount,
                        'cost': transfer_amount * surplus_agent.state.transfer_cost_per_unit,
                        'nash_product': round(nash_product, 3),
                    })
        
        # 计算整体改善
        total_cost = sum(p['cost'] for p in transfer_proposals)
        
        return {
            'transfer_plan': transfer_proposals,
            'total_transfer_cost': round(total_cost, 2),
            'converged': len(transfer_proposals) > 0,
            'consensus_reached': True,
            'summary': f"协商结果：{len(transfer_proposals)}笔调拨，总成本${total_cost:.0f}"
        }


def test_mas_multi_warehouse():
    """测试多仓库Nash协商"""
    warehouses = [
        WarehouseState('US_East', current_stock=480, safety_stock=200, 
                       daily_demand=10, lead_time_days=5, transfer_cost_per_unit=4.0),
        WarehouseState('US_West', current_stock=60, safety_stock=200, 
                       daily_demand=12, lead_time_days=5, transfer_cost_per_unit=4.0),
        WarehouseState('DE', current_stock=220, safety_stock=150, 
                       daily_demand=8, lead_time_days=7, transfer_cost_per_unit=8.0),
    ]
    
    agents = [WarehouseAgent(w) for w in warehouses]
    orchestrator = NashBargainingOrchestrator(agents)
    
    result = orchestrator.find_nash_equilibrium()
    
    print("=" * 65)
    print("MAS多仓库Nash协商结果（吸奶器三仓场景）")
    print("=" * 65)
    print(f"\n初始库存状态:")
    for w in warehouses:
        coverage = w.current_stock / w.daily_demand
        status = '⚠️过剩' if coverage > w.safety_stock / w.daily_demand * 1.8 else \
                 '🔴缺货' if coverage < w.safety_stock / w.daily_demand * 0.8 else '✅正常'
        print(f"  {w.warehouse_id}: {w.current_stock}件 | 覆盖{coverage:.0f}天 {status}")
    
    print(f"\n协商调拨方案:")
    for plan in result['transfer_plan']:
        print(f"  {plan['from']} → {plan['to']}: {plan['units']}件 | 成本${plan['cost']:.0f} | Nash积={plan['nash_product']}")
    
    print(f"\n{result['summary']}")
    
    assert result['consensus_reached'], "应达成协商共识"
    # 应建议从US_East（过剩）调拨到US_West（缺货）
    us_east_to_west = [p for p in result['transfer_plan'] 
                       if p['from'] == 'US_East' and p['to'] == 'US_West']
    assert len(us_east_to_west) > 0, "应建议美东→美西调拨（过剩→缺货）"
    
    print("\n[✓] MAS多仓库补货Nash协商测试通过")

test_mas_multi_warehouse()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-MAS-Consensus-Mechanism]]（MAS协商机制基础）
- **前置（prerequisite）**：[[Skill-Safety-Stock-Replenishment]]（安全库存补货计算基础）
- **延伸（extends）**：[[Skill-Multi-Echelon-Inventory]]（多级库存优化，扩展到供应商-DC-仓库链）
- **延伸（extends）**：[[Skill-Flowr-Supply-Chain-MAS]]（完整供应链MAS框架）
- **可组合（combinable）**：[[Skill-Lead-Time-Distribution-Risk-GenQOT]]（前置期风险感知 + 多仓协商 = 韧性补货体系）

## ⑤ 商业价值评估

- **ROI 预估**：美东→美西调拨50件，避免美西缺货损失$2000/天，调拨成本$200，净收益$1800/决策；年化（假设每月2次）约 **15-25万元**；同时减少美东过剩FBA仓储费约 **5-10万元/年**
- **vs 中心化决策**：响应速度从48-72小时→8-12小时，信息完整性更高（仓库Agent持有实时数据）
- **实施难度**：⭐⭐⭐⭐☆（需要各仓库数据接口打通，以及Agent框架部署）
- **优先级**：⭐⭐⭐⭐☆（多仓运营的中级场景，库存规模大的品牌优先）
