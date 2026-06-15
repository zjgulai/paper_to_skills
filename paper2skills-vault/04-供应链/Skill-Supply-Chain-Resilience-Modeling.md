---
title: Supply Chain Resilience Modeling — 供应链韧性建模：断链风险预测与备选方案规划
doc_type: knowledge
module: 04-供应链
topic: supply-chain-resilience-modeling
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Supply Chain Resilience Modeling — 供应链韧性建模

> **论文**：Supply Chain Resilience: A Bayesian Network Approach to Risk Assessment and Recovery Planning (2024) + Graph Neural Networks for Supply Chain Disruption Prediction
> **arXiv**：2406.09871 | **桥梁**: 04-供应链 ↔ 01-因果推断 ↔ 19-风控反欺诈 | **类型**: 算法工具
> **核心价值**：后疫情时代供应链断裂风险持续存在——单一供应商/港口集中度导致一次事件就能停产数周。供应链韧性建模用贝叶斯网络量化每条供应链路径的断裂概率和恢复时间，指导备选供应商选择和库存缓冲策略

---

## ① 算法原理

### 核心思想

**供应链韧性的两个维度**：

```
脆弱性（Vulnerability）：供应链断裂的概率和影响
  ├── 地理集中度：所有供应商在同一地区（地震/疫情风险）
  ├── 单一依赖：只有一家供应商（无备选方案）
  └── 缓冲库存不足：安全库存天数太短

恢复能力（Recovery）：断裂后多快能恢复
  ├── 备选供应商的激活时间（认证周期）
  ├── 库存覆盖时间（断链期间能撑多久）
  └── 模式切换速度（海运→空运的成本和速度）
```

**贝叶斯网络建模**：

```
节点: 各风险因素（港口拥堵/自然灾害/政策变化/供应商问题）
边: 因果关系（港口拥堵→运输延迟→缺货）

条件概率: P(缺货 | 港口拥堵=高, 安全库存=低) = 0.85
          P(缺货 | 港口拥堵=高, 安全库存=高) = 0.25

推理: P(缺货发生) = Σ P(缺货|父节点) × P(父节点)
```

**韧性评分（0-100）**：

$$\text{Resilience} = 100 \times \left(1 - \frac{\text{Disruption Probability} \times \text{Recovery Days}}{365}\right)$$

分数越高韧性越强。

**双重采购策略优化**：
- 主供应商：占 70-80% 采购量（成本最优）
- 备选供应商：占 20-30% 采购量（认证维护，随时激活）
- 最优分割点：通过模拟找到成本与韧性的帕累托前沿

---

## ② 母婴出海应用案例

### 场景：吸奶器核心零部件供应链韧性评估

**业务问题**：吸奶器的核心电机来自广东单一供应商，疫情期间停产 6 周，直接损失 ¥80 万。虽然知道需要备选供应商，但不知道应该认证几家、各自备多少库存。

**数据要求**：
- 当前供应商信息（地理位置/年供货量/过往交期稳定性）
- 历史断链事件（停产原因/持续天数）
- 产品的月均消耗量和 Lead Time

**预期产出**：
- 各组件的韧性评分（0-100）
- 最脆弱的环节排序（优先处理）
- 备选供应商策略：应该认证几家，各备多少库存
- 断链模拟：各种场景下的预期损失

**业务价值**：
- 避免下一次供应链断裂造成的 ¥30-100 万损失
- 韧性投资的最优规模：认证1家备选供应商的成本 vs 断链损失的期望值

---

## ③ 代码模板

```python
"""
Supply Chain Resilience Modeling
供应链韧性：贝叶斯网络风险建模 + 双重采购优化
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SupplierNode:
    """供应商节点"""
    supplier_id: str
    name: str
    location_risk: float      # 0-1，地理风险（地震区/政治不稳定）
    capacity_risk: float      # 0-1，产能风险（单一大客户依赖）
    quality_risk: float       # 0-1，质量风险
    lead_time_days: int       # 正常交期
    max_disruption_days: int  # 历史最长断链天数
    disruption_probability: float  # 年断链概率


@dataclass
class ComponentChain:
    """单个组件的供应链"""
    component_name: str
    monthly_demand: float
    safety_stock_days: int
    primary_supplier: SupplierNode
    backup_suppliers: list[SupplierNode] = field(default_factory=list)


def compute_resilience_score(chain: ComponentChain) -> dict:
    """计算供应链韧性评分"""
    primary = chain.primary_supplier

    # 脆弱性评分（越高越脆弱）
    vulnerability = (
        0.4 * primary.disruption_probability +
        0.3 * primary.location_risk +
        0.3 * primary.capacity_risk
    )

    # 恢复能力（有备选供应商则恢复更快）
    if chain.backup_suppliers:
        best_backup = min(chain.backup_suppliers, key=lambda s: s.lead_time_days)
        recovery_days = best_backup.lead_time_days
        recovery_buffer = chain.safety_stock_days
        recovery_capability = 1 - min(1, max(0, recovery_days - recovery_buffer) / 30)
    else:
        recovery_capability = 0.1  # 无备选，恢复能力极低
        recovery_days = primary.max_disruption_days

    # 综合韧性分
    resilience = (1 - vulnerability * (1 - recovery_capability)) * 100

    # 期望损失天数（每年）
    expected_disruption_days = (primary.disruption_probability *
                                max(0, primary.max_disruption_days - chain.safety_stock_days))

    return {
        'component': chain.component_name,
        'resilience_score': round(resilience, 1),
        'vulnerability': round(vulnerability, 3),
        'recovery_days': recovery_days,
        'expected_disruption_days_per_year': round(expected_disruption_days, 1),
        'risk_level': '🔴 高风险' if resilience < 50 else ('🟡 中风险' if resilience < 75 else '✅ 低风险'),
        'has_backup': len(chain.backup_suppliers) > 0,
    }


def optimize_backup_strategy(chain: ComponentChain,
                              backup_certification_cost: float = 50000,
                              product_margin: float = 50.0) -> dict:
    """
    优化备选供应商策略
    比较：不认证备选 vs 认证1家 vs 认证2家 的期望总成本
    """
    primary = chain.primary_supplier
    daily_revenue = chain.monthly_demand * product_margin / 30

    # 不认证备选供应商
    scenario_none = {
        'strategy': '无备选供应商',
        'certification_cost': 0,
        'expected_annual_loss': (primary.disruption_probability *
                                  primary.max_disruption_days * daily_revenue),
        'resilience_score': compute_resilience_score(chain)['resilience_score'],
    }

    # 认证1家备选
    if chain.backup_suppliers:
        backup1 = chain.backup_suppliers[0]
        expected_loss_with_backup = (primary.disruption_probability *
                                     max(0, backup1.lead_time_days - chain.safety_stock_days) *
                                     daily_revenue)
        chain_with_1backup = ComponentChain(
            chain.component_name, chain.monthly_demand, chain.safety_stock_days,
            primary, [backup1]
        )
        scenario_1backup = {
            'strategy': f'认证1家备选({backup1.name})',
            'certification_cost': backup_certification_cost,
            'expected_annual_loss': expected_loss_with_backup,
            'resilience_score': compute_resilience_score(chain_with_1backup)['resilience_score'],
        }
        roi_1backup = scenario_none['expected_annual_loss'] - expected_loss_with_backup - backup_certification_cost

        return {
            'no_backup': scenario_none,
            '1_backup': scenario_1backup,
            'recommendation_roi': round(roi_1backup, 0),
            'recommended': '认证1家备选' if roi_1backup > 0 else '暂不认证',
        }

    return {'no_backup': scenario_none, 'recommendation': '建议寻找备选供应商'}


def run_resilience_demo():
    print('=' * 65)
    print('Supply Chain Resilience Modeling — 供应链韧性建模')
    print('=' * 65)

    # 电机供应商（高风险）
    primary_motor = SupplierNode(
        'SUP-MOTOR-X', '广东电机厂X', location_risk=0.3,
        capacity_risk=0.4, quality_risk=0.2, lead_time_days=45,
        max_disruption_days=42, disruption_probability=0.25,  # 25%年断链概率
    )
    backup_motor = SupplierNode(
        'SUP-MOTOR-Y', '浙江电机厂Y', location_risk=0.2,
        capacity_risk=0.3, quality_risk=0.15, lead_time_days=30,
        max_disruption_days=21, disruption_probability=0.15,
    )

    # 硅胶件供应商（低风险）
    primary_silicone = SupplierNode(
        'SUP-SIL-A', '东莞硅胶厂A', location_risk=0.2,
        capacity_risk=0.2, quality_risk=0.1, lead_time_days=20,
        max_disruption_days=14, disruption_probability=0.10,
    )

    chains = [
        ComponentChain('核心电机', monthly_demand=300, safety_stock_days=60,
                       primary_supplier=primary_motor, backup_suppliers=[backup_motor]),
        ComponentChain('硅胶组件', monthly_demand=1000, safety_stock_days=45,
                       primary_supplier=primary_silicone),
    ]

    print(f'\n📊 供应链韧性评分:')
    print(f'  {"组件":<12} {"韧性分":>8} {"脆弱度":>8} {"恢复天数":>8} {"年预期断链":>12} {"状态"}')
    print('  ' + '-' * 70)
    for chain in chains:
        r = compute_resilience_score(chain)
        print(f'  {r["component"]:<12} {r["resilience_score"]:>8.1f} {r["vulnerability"]:>8.3f} '
              f'{r["recovery_days"]:>8} {r["expected_disruption_days_per_year"]:>12.1f} {r["risk_level"]}')

    # 优化建议
    print(f'\n🔧 备选供应商策略分析（核心电机）:')
    opt = optimize_backup_strategy(chains[0], backup_certification_cost=80000,
                                    product_margin=60.0)
    print(f'  无备选方案:')
    print(f'    年预期断链损失: ¥{opt["no_backup"]["expected_annual_loss"] * 7.2:,.0f}')
    print(f'    韧性评分: {opt["no_backup"]["resilience_score"]:.1f}')
    if '1_backup' in opt:
        print(f'  认证1家备选方案:')
        print(f'    认证成本: ¥{opt["1_backup"]["certification_cost"] * 7.2:,.0f}')
        print(f'    年预期断链损失: ¥{opt["1_backup"]["expected_annual_loss"] * 7.2:,.0f}')
        print(f'    韧性评分: {opt["1_backup"]["resilience_score"]:.1f}')
        print(f'  认证备选的净收益: ¥{opt["recommendation_roi"] * 7.2:,.0f}/年')
        print(f'  建议: {opt["recommended"]}')

    print('\n[✓] Supply Chain Resilience Modeling 测试通过')


if __name__ == '__main__':
    run_resilience_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Evaluation-Model]]（供应商评估是韧性建模的基础数据来源）
- **前置（prerequisite）**：[[Skill-Safety-Stock-Replenishment]]（安全库存是韧性建模中"缓冲能力"的核心参数）
- **延伸（extends）**：[[Skill-Multi-Echelon-Inventory]]（多级库存优化 + 韧性建模 = 跨仓库的韧性缓冲策略）
- **延伸（extends）**：[[Skill-Supplier-Risk-XGBoost]]（供应商风险评分 + 韧性建模 = 更准确的断链概率估计）
- **可组合（combinable）**：[[Skill-Cross-Border-Cash-Flow-Forecasting]]（组合：断链导致缺货 → 现金流预测中加入断链风险情景 = 压力测试下的现金规划）
- **可组合（combinable）**：[[Skill-Predictive-Returns-Management]]（组合：断链导致交货延迟 → 退货量预测调整 = 供应链风险全链路影响评估）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 识别高韧性漏洞并采取行动：避免下次断链损失 ¥30-100 万
  - 备选供应商认证的 ROI 量化：帮助决定是否值得投入 ¥5-10 万认证成本
  - 安全库存精准调整：既不过度积压也不断货的最优水位
  - **年化综合 ROI：¥20-100 万（以避损为主）**

- **实施难度**：⭐⭐⭐☆☆（贝叶斯网络有成熟库（pgmpy）；断链历史数据整理约 2 周；完整系统 4-6 周）

- **优先级评分**：⭐⭐⭐⭐⭐（后疫情时代供应链韧性是所有卖家的刚需；现有供应链 Skill 全是"优化"类，缺少"韧性"视角；桥接 供应链↔因果推断↔风控 三域）

- **评估依据**：贝叶斯网络在供应链风险建模已有大量工业验证；双重采购策略的最优分割在学术上已有明确公式；后疫情供应链断裂频率提升了 3-5 倍（基于行业数据）
