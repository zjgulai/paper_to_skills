from __future__ import annotations

from mas.skills.mcp_servers.base import BaseMCPServer
from mas.skills.supply_chain_tools import (
    demand_forecast,
    safety_stock_reorder_point,
    causal_counterfactual_forecast,
)
from typing import Any, Dict, List


class SupplyChainServer(BaseMCPServer):
    server_name = "supply_chain_server"
    domain = "supply_chain"

    def _register_tools(self) -> None:
        self._add("supply_demand_forecast", "需求预测(供应链版),输入历史销售/季节性返回各 SKU 未来 N 周需求",
                  lambda **kw: demand_forecast(**kw))
        self._add("supply_safety_stock_replenishment", "安全库存与再订货点计算",
                  lambda **kw: safety_stock_reorder_point(**kw))
        self._add("supply_causal_gcf", "GCF 反事实需求预测(处理促销/缺货干预)",
                  lambda **kw: causal_counterfactual_forecast(**kw))
        self._add("supply_multi_echelon_inventory", "多级库存优化,跨仓位补货分配建模",
                  lambda **kw: {"skill": "supply_multi_echelon_inventory", "status": "stub_ok", **kw})
        self._add("supply_two_echelon_drl", "两级库存 DRL 策略,自动给出补货量与时点",
                  lambda **kw: {"skill": "supply_two_echelon_drl", "status": "stub_ok", **kw})
        self._add("supply_monodense_price_elasticity", "单 SKU 价格弹性估计",
                  lambda **kw: {"skill": "supply_monodense_price_elasticity", "status": "stub_ok", **kw})
        self._add("supply_hiforead_reconciliation", "HiFoReAd 分层需求预测调和(SKU×仓×市场)",
                  lambda **kw: {"skill": "supply_hiforead_reconciliation", "status": "stub_ok", **kw})
        self._add("supply_multi_channel_inventory_pooling", "多渠道库存池化,跨Amazon/独立站/TikTok调拨",
                  lambda **kw: {"skill": "supply_multi_channel_inventory_pooling", "status": "stub_ok", **kw})
        self._add("supply_genqot_lead_time_risk", "Gen-QOT 提前期分布风险,海运波动+旺季动态安全库存",
                  lambda **kw: {"skill": "supply_genqot_lead_time_risk", "status": "stub_ok", **kw})
