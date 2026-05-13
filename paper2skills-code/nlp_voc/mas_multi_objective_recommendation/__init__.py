"""Multi-Agent Multi-Objective Recommendation System

Based on arXiv:2512.24325 (MaRCA).

Core Components:
- ClickAgent: Optimize click-through rate
- ConversionAgent: Optimize conversion rate
- ProfitAgent: Optimize profit margin
- DiversityAgent: Optimize catalog coverage
- Coordinator: Learn optimal weight combination
- MultiObjectiveRecommender: Main system integrating all agents
- RecommenderEnvironment: Simulate user behavior

Usage:
    from mas_multi_objective_recommendation import (
        MultiObjectiveRecommender,
        create_momcozy_catalog,
        create_demo_users,
    )

    recommender = MultiObjectiveRecommender(catalog)
    for user in users:
        recommender.observe_feedback(user)
    metrics = recommender.evaluate()
"""

from model import (
    ClickAgent,
    ConversionAgent,
    Coordinator,
    DiversityAgent,
    MultiObjectiveRecommender,
    ObjectiveAgent,
    Product,
    ProfitAgent,
    Recommendation,
    RecommenderEnvironment,
    User,
    UserFeedback,
    create_demo_users,
    create_momcozy_catalog,
    run_ab_test,
)

__all__ = [
    "ClickAgent",
    "ConversionAgent",
    "Coordinator",
    "DiversityAgent",
    "MultiObjectiveRecommender",
    "ObjectiveAgent",
    "Product",
    "ProfitAgent",
    "Recommendation",
    "RecommenderEnvironment",
    "User",
    "UserFeedback",
    "create_demo_users",
    "create_momcozy_catalog",
    "run_ab_test",
]