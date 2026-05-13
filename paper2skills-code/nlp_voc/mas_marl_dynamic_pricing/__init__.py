"""MARL Multi-Agent Dynamic Pricing in Supply Chains

Multi-Agent Reinforcement Learning for dynamic pricing optimization.
Based on arXiv:2507.02698.

Core Components:
- PricingAgent: Base class for pricing strategies
- StaticMarkupAgent: Fixed cost-plus pricing baseline
- CompetitorMatchingAgent: Follow competitor prices
- DemandResponsiveAgent: Adjust price based on demand signals
- SeasonalPricingAgent: Seasonal adjustment pricing
- QLearningAgent: Table-based Q-learning (simplified MARL)
- MarketEnvironment: Competitive market simulation
- PricingSimulator: Strategy comparison engine

Usage:
    from mas_marl_dynamic_pricing import (
        PricingSimulator,
        QLearningAgent,
        MarketEnvironment,
        Product,
    )

    simulator = PricingSimulator()
    results = simulator.compare_strategies(product, n_agents=10, n_weeks=52)
"""

from model import (
    CompetitorMatchingAgent,
    DemandModel,
    DemandResponsiveAgent,
    EpisodeResult,
    MarketEnvironment,
    MarketState,
    PricingAction,
    PricingAgent,
    PricingSimulator,
    Product,
    QLearningAgent,
    SeasonalPricingAgent,
    StaticMarkupAgent,
    create_momcozy_pricing_scenario,
)

__all__ = [
    "CompetitorMatchingAgent",
    "DemandModel",
    "DemandResponsiveAgent",
    "EpisodeResult",
    "MarketEnvironment",
    "MarketState",
    "PricingAction",
    "PricingAgent",
    "PricingSimulator",
    "Product",
    "QLearningAgent",
    "SeasonalPricingAgent",
    "StaticMarkupAgent",
    "create_momcozy_pricing_scenario",
]
