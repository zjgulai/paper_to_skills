"""
Auto-extracted from: paper2skills-vault/04-供应链/Skill-Safety-Stock-Replenishment.md
Skill: Skill-Safety-Stock-Replenishment
Domain: 04-供应链
"""
"""
Safety Stock and Replenishment Strategy — 安全库存与补货策略
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_safety_stock(demand_std, lead_time, service_level=0.95):
    """
    计算安全库存

    Args:
        demand_std: 需求标准差（单位时间）
        lead_time: 提前期（单位时间）
        service_level: 目标服务水平
    """
    z = stats.norm.ppf(service_level)
    sigma_ltd = demand_std * np.sqrt(lead_time)
    ss = z * sigma_ltd
    return ss


def calculate_rop(avg_demand, lead_time, safety_stock):
    """计算再订货点"""
    return avg_demand * lead_time + safety_stock


def calculate_eoq(annual_demand, order_cost, holding_cost_per_unit):
    """计算经济订货量"""
    return int(np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit))


def calculate_inventory_metrics(avg_demand, demand_std, lead_time,
                                 order_cost, holding_cost, unit_cost,
                                 service_level=0.95):
    """
    计算完整库存指标

    Returns:
        dict with safety_stock, rop, eoq, avg_inventory, turns, stockout_prob
    """
    annual_demand = avg_demand * 52
    ss = calculate_safety_stock(demand_std, lead_time, service_level)
    rop = calculate_rop(avg_demand, lead_time, ss)
    eoq = calculate_eoq(annual_demand, order_cost, holding_cost)

    # 平均库存 = 周期库存/2 + 安全库存
    avg_inventory = eoq / 2 + ss

    # 库存周转率
    inventory_value = avg_inventory * unit_cost
    turns = (annual_demand * unit_cost) / inventory_value if inventory_value > 0 else 0

    # 年总成本
    annual_ordering_cost = (annual_demand / eoq) * order_cost
    annual_holding_cost = avg_inventory * holding_cost
    total_cost = annual_ordering_cost + annual_holding_cost

    return {
        'safety_stock': int(ss),
        'reorder_point': int(rop),
        'eoq': int(eoq),
        'avg_inventory': int(avg_inventory),
        'inventory_turns': turns,
        'annual_ordering_cost': annual_ordering_cost,
        'annual_holding_cost': annual_holding_cost,
        'total_annual_cost': total_cost
    }


class ReplenishmentPlanner:
    """补货计划器"""

    def __init__(self, skus_config):
        """
        Args:
            skus_config: list of dicts, each with keys:
                sku_id, avg_demand, demand_std, lead_time,
                current_stock, in_transit, unit_cost, order_cost, holding_cost
        """
        self.skus = pd.DataFrame(skus_config)

    def plan(self, service_level=0.95):
        """生成补货计划"""
        plans = []
        for _, row in self.skus.iterrows():
            metrics = calculate_inventory_metrics(
                row['avg_demand'], row['demand_std'], row['lead_time'],
                row['order_cost'], row['holding_cost'], row['unit_cost'],
                service_level
            )

            available = row['current_stock'] + row.get('in_transit', 0)
            need_replenish = available <= metrics['reorder_point']

            plans.append({
                'sku_id': row['sku_id'],
                'current_stock': row['current_stock'],
                'available': available,
                'reorder_point': metrics['reorder_point'],
                'need_replenish': need_replenish,
                'order_qty': metrics['eoq'] if need_replenish else 0,
                'safety_stock': metrics['safety_stock'],
                'projected_stockout_date': self._estimate_stockout(row, metrics)
            })

        return pd.DataFrame(plans)

    def _estimate_stockout(self, row, metrics):
        """估算缺货日期"""
        if row['avg_demand'] <= 0:
            return None
        days_until_rop = (row['current_stock'] - metrics['reorder_point']) / row['avg_demand'] * 7
        return pd.Timestamp.now() + pd.Timedelta(days=max(0, days_until_rop))


# 示例
if __name__ == '__main__':
    # 奶粉SKU示例
    metrics = calculate_inventory_metrics(
        avg_demand=200, demand_std=40, lead_time=6,
        order_cost=500, holding_cost=2.4, unit_cost=30
    )
    print("奶粉SKU库存指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:,.0f}" if isinstance(v, float) else f"  {k}: {v}")

    # 多SKU补货计划
    skus = [
        {'sku_id': '奶粉3段', 'avg_demand': 200, 'demand_std': 40, 'lead_time': 6,
         'current_stock': 1500, 'in_transit': 0, 'unit_cost': 30, 'order_cost': 500, 'holding_cost': 2.4},
        {'sku_id': '纸尿裤M', 'avg_demand': 500, 'demand_std': 80, 'lead_time': 4,
         'current_stock': 800, 'in_transit': 2000, 'unit_cost': 15, 'order_cost': 300, 'holding_cost': 1.2},
    ]
    planner = ReplenishmentPlanner(skus)
    plan = planner.plan()
    print("\n补货计划:")
    print(plan[['sku_id', 'current_stock', 'reorder_point', 'need_replenish', 'order_qty']].to_string(index=False))
