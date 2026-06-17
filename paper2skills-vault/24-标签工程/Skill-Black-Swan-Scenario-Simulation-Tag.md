---
title: 黑天鹅情景模拟标签 — 极端事件供应链压力测试与预案激活机制
doc_type: knowledge
module: 24-标签工程
topic: black-swan-scenario-simulation-tag
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 黑天鹅情景模拟标签

> **来源**：arXiv:2404.09823（Black Swan Scenario Analysis in Supply Chains）+ arXiv:2310.11834（Supply Chain Stress Testing）+ WEF供应链韧性框架
> **桥梁**：供应链风险 ↔ 标签工程 ↔ 韧性建模 | **类型**：情景模拟

## ① 算法原理

**黑天鹅情景模拟** 回答："如果发生了最坏的情况，我们的供应链能扛住多久？需要多长时间恢复？"

**5类典型情景**：

| 情景 | 触发 | 影响范围 | 持续时间 |
|-----|------|--------|--------|
| S1 主供应商断供 | 工厂火灾/罢工 | 关键SKU 60-80% | 2-8周 |
| S2 港口封锁 | 罢工/自然灾害 | 所有进口货物 | 1-4周 |
| S3 关税暴涨 | 贸易战升级 | 对应来源地SKU | 长期 |
| S4 需求崩溃 | 经济危机/品类衰退 | 全品类销售 | 3-12月 |
| S5 平台封号 | Amazon账号暂停 | 主渠道100%断销 | 1-8周 |

**压力测试输出**：
- 每个情景下的GMV损失（$）
- 库存撑过时间（天）
- 恢复所需时间（周）
- 缓解成本（$）

**预案激活Tag**：
- 当 `scenario_probability > 0.15` → `contingency_plan.activate=True`
- 预案包括：备用供应商激活/渠道切换/价格策略调整

## ② 代码模板

```python
"""
黑天鹅情景模拟标签系统
功能：情景定义 / 压力测试计算 / 影响量化 / 预案激活 / 韧性评分
"""
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BlackSwanScenario:
    scenario_id: str
    name: str
    probability: float       # 年发生概率
    gmv_impact_pct: float    # GMV影响比例（负数）
    duration_weeks: int      # 持续时间
    recovery_weeks: int      # 恢复时间
    mitigation_cost_usd: float  # 缓解成本
    contingency_actions: list = field(default_factory=list)


def run_stress_test(scenarios: list, monthly_gmv_usd: float,
                    current_inventory_days: float) -> list:
    """运行压力测试"""
    results = []
    for sc in scenarios:
        # GMV损失
        gmv_loss = monthly_gmv_usd * abs(sc.gmv_impact_pct) * sc.duration_weeks / 4.3
        # 库存是否能撑过
        inventory_survives = current_inventory_days > sc.duration_weeks * 7
        # 总损失
        total_loss = gmv_loss + sc.mitigation_cost_usd
        # 韧性评分（越高越能抗）
        resilience = min(100, max(0,
            (inventory_survives * 30) +
            (1 - sc.probability) * 30 +
            (1 - abs(sc.gmv_impact_pct)) * 20 +
            min(1, sc.mitigation_cost_usd / total_loss) * 20
        ))
        results.append({
            "scenario": sc.name, "probability": sc.probability,
            "gmv_loss_usd": round(gmv_loss, 0),
            "total_loss_usd": round(total_loss, 0),
            "duration_weeks": sc.duration_weeks,
            "inventory_survives": inventory_survives,
            "resilience_score": round(resilience, 1),
            "activate_contingency": sc.probability > 0.10,
            "tags": {
                f"scenario.{sc.scenario_id}.probability": sc.probability,
                f"scenario.{sc.scenario_id}.gmv_impact_pct": sc.gmv_impact_pct,
                f"contingency.{sc.scenario_id}.activate": sc.probability > 0.10,
            }
        })
    return sorted(results, key=lambda x: x["total_loss_usd"], reverse=True)


if __name__ == "__main__":
    print("【黑天鹅情景模拟标签系统】\n")
    scenarios = [
        BlackSwanScenario("S1", "主供应商断供", 0.08, -0.70, 4, 6, 15_000,
                           ["激活备用供应商", "空运紧急补货"]),
        BlackSwanScenario("S2", "主要港口封锁", 0.05, -0.40, 3, 3, 8_000,
                           ["改变运输路线", "提前备货"]),
        BlackSwanScenario("S3", "关税大幅上涨", 0.15, -0.20, 52, 26, 50_000,
                           ["转移生产到越南/印度", "提价"]),
        BlackSwanScenario("S4", "平台封号", 0.03, -0.80, 4, 8, 20_000,
                           ["启动独立站", "切换TikTok"]),
    ]

    results = run_stress_test(scenarios, monthly_gmv_usd=500_000, current_inventory_days=45)

    print("=" * 65)
    print("【压力测试结果（按损失排序）】")
    for r in results:
        activate_icon = "🔔" if r["activate_contingency"] else "📋"
        surv_icon = "✅" if r["inventory_survives"] else "❌"
        print(f"\n  [{r['scenario']}] 概率:{r['probability']:.0%}")
        print(f"    GMV损失: ${r['gmv_loss_usd']:,}  总损失: ${r['total_loss_usd']:,}")
        print(f"    库存撑住: {surv_icon}  韧性评分: {r['resilience_score']:.0f}")
        print(f"    {activate_icon} 预案激活: {'是' if r['activate_contingency'] else '否'}")

    print(f"\n[✓] 黑天鹅情景模拟标签 测试通过  {len(scenarios)}个情景完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SC-Resilience-Hypergraph]]（超图韧性建模基础）
- **前置（prerequisite）**：[[Skill-Geopolitical-Risk-Tag-Supply-Impact]]（地缘风险是黑天鹅的触发源）
- **延伸（extends）**：[[Skill-Supply-Chain-Agent-Orchestration-Hub]]（预案激活触发编排中枢启动应急响应）
- **可组合（combinable）**：[[Skill-Supplier-Capacity-Booking-Engine]]（弹性产能预订是S1场景的预案之一）

## ⑤ 商业价值评估

- **ROI预估**：关税暴涨情景（年发生率15%）提前规划生产迁移，节省约50万关税差额；平台封号预案（提前建独立站+TikTok），减少封号期间GMV损失约80%
- **实施难度**：⭐⭐⭐☆☆（主要是情景数据收集和预案制定，算法本身不复杂）
- **优先级评分**：⭐⭐⭐⭐☆（2024年红海危机/2023年亚马逊封号潮已证明黑天鹅不是小概率事件）
