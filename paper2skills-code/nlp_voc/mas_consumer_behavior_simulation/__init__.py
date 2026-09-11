"""MAS Consumer Behavior Simulation

LLM-Based Multi-Agent System for Simulating and Analyzing Consumer Behavior.
Based on arXiv:2510.18155.

Core Components:
- ConsumerAgent: Heterogeneous consumer agent with profile, memory, and decision logic
- SimulationEngine: Orchestrates multi-agent interactions in a virtual market
- SimulationAnalyzer: Analyzes promotion effects, market share, and loyalty
- PromotionConfig: Configurable promotion scenarios

Usage:
    from mas_consumer_behavior_simulation import (
        SimulationEngine,
        ConsumerAgent,
        SimulationAnalyzer,
        create_momcozy_scenario,
        PromotionConfig,
    )

    promotion = PromotionConfig(shop_id="momcozy_store", discount_rate=0.20, start_day=2, end_day=4)
    engine = create_momcozy_scenario(n_agents=30, promotion=promotion)
    results = engine.run(n_days=7)

    analyzer = SimulationAnalyzer(engine.records, engine.day_logs)
    effect = analyzer.promotion_effect("Momcozy Flagship", (2, 4))
"""

from model import (
    AgentProfile,
    ConsumerAgent,
    Location,
    MemoryEntry,
    Product,
    PromotionConfig,
    PurchaseRecord,
    Shop,
    SimulationAnalyzer,
    SimulationEngine,
    create_momcozy_scenario,
    run_simulation_with_promotion,
)

__all__ = [
    "AgentProfile",
    "ConsumerAgent",
    "Location",
    "MemoryEntry",
    "Product",
    "PromotionConfig",
    "PurchaseRecord",
    "Shop",
    "SimulationAnalyzer",
    "SimulationEngine",
    "create_momcozy_scenario",
    "run_simulation_with_promotion",
]
