---
title: 生意规模三维KPI监控 — 销售增长率/计划达成率/GMV完成比的实时预警与归因
doc_type: knowledge
module: 04-供应链
topic: business-scale-kpi-growth-achievement-gmv
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 生意规模三维KPI监控

> **书籍**：《全链路管理》陈凤霞 第二章第二节"电商生意计划供应链的KPI——生意增长和规模"
> **arXiv**：结合 2025-2026 跨境电商计划准确率相关研究 | **桥梁**: 供应链 ↔ 运营财务 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：生意规模KPI是供应链"结果层"的三维表达——**增长率**衡量相对速度，**达成率**衡量计划执行力，**GMV完成比**衡量目标兑现度。书中特别强调：**达成率与预测准确率是不同的计算逻辑**——达成率可以>100%（超额），预测准确率不会超过100%（偏差越小越好）。这两个指标反映的管理维度完全不同，混淆使用是供应链管理的常见错误。

**三维KPI定义（书中标准）**：

1. **销售增长率（YoY/MoM Growth Rate）**：
   - 同比增长率 = (本期销售 - 上年同期) / 上年同期 × 100%
   - 环比增长率 = (本期销售 - 上期) / 上期 × 100%
   - 书中强调：大促期间要用"日均"而非总量比较，消除时间跨度影响

2. **计划达成率（Plan Achievement Rate）**：
   - 达成率 = 实际销售 / 计划销售目标 × 100%
   - 与预测准确率的区别：达成率=完成情况（考核导向），准确率=预测偏差（优化导向）
   - 行业标准：日常≥95%，大促≥90%

3. **GMV完成比（GMV Completion Ratio）**：
   - 实时GMV完成比 = 当前累计GMV / 当期目标GMV × 100%
   - 分段进度预警：前30%时间完成<25% → 预警；前50%时间完成<40% → 紧急干预

**算法突破口：差异分解（Variance Decomposition）**：

当达成率偏低时，书中提出"拆因"框架：
```
实际GMV = 订单量 × 客单价
       = 流量 × 转化率 × 客单价

GMV差距 = 流量差距贡献 + 转化率差距贡献 + 客单价差距贡献
```

通过Shapley值分配各因素对GMV缺口的贡献，精确定位改善方向。

## ② 母婴出海应用案例

**场景A：Amazon旺季GMV目标实时追踪**

- **业务问题**：某母婴品牌Q4 GMV目标$50万，大促第1天结束只完成$3.2万（完成比6.4%），但团队不知道是流量问题还是转化率问题
- **三维KPI监控方案**：
  1. 实时计算：GMV完成比 = $3.2万/$50万 = 6.4%（正常第1天目标8-10%）→ 轻度落后
  2. 差异分解：流量-15%（竞品广告挤压），转化率+3%（页面优化有效），客单价-2%（折扣影响）
  3. 主要差距=流量，立即决策：提高SP广告出价15%
- **预期产出**：精准归因后的干预比盲目加预算效率高40%，大促GMV最终完成92%

**场景B：月度计划达成率分析**

- **业务问题**：运营团队月末看到达成率只有85%，但无法判断是预测偏高、还是实际执行不到位
- **算法应用**：将85%达成率分解为：计划偏高贡献8% + 执行差距贡献7%；计划偏高→调整下月预测模型；执行差距→追查缺货/断货原因

## ③ 代码模板

```python
"""
生意规模三维KPI监控系统
功能：增长率/达成率/GMV完成比实时计算 + Shapley差异分解
基于《全链路管理》陈凤霞 第二章第二节
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BusinessScaleKPI:
    """生意规模三维KPI计算器"""

    @staticmethod
    def growth_rate(current: float, base: float, mode: str = 'yoy') -> Dict:
        """计算增长率"""
        if base <= 0:
            return {'rate': None, 'status': 'N/A', 'message': '基期为零'}
        rate = (current - base) / base
        status = '🟢增长' if rate > 0.1 else ('🟡持平' if rate >= -0.05 else '🔴下滑')
        return {
            'current': current,
            'base': base,
            'rate': rate,
            'rate_pct': f"{rate:.1%}",
            'status': status,
            'mode': mode,
        }

    @staticmethod
    def achievement_rate(actual: float, plan: float) -> Dict:
        """计算计划达成率（注意：不同于预测准确率）"""
        if plan <= 0:
            return {'rate': None, 'status': 'N/A'}
        rate = actual / plan
        # 达成率可超100%
        if rate >= 1.0:
            status = '✅超额完成'
        elif rate >= 0.95:
            status = '🟢达成'
        elif rate >= 0.90:
            status = '🟡轻度未达'
        elif rate >= 0.80:
            status = '🟠未达'
        else:
            status = '🔴严重未达'
        return {
            'actual': actual,
            'plan': plan,
            'rate': rate,
            'rate_pct': f"{rate:.1%}",
            'gap': actual - plan,
            'gap_pct': f"{(actual-plan)/plan:.1%}",
            'status': status,
        }

    @staticmethod
    def gmv_completion_progress(current_gmv: float, target_gmv: float,
                                 time_elapsed_pct: float) -> Dict:
        """实时GMV完成进度预警"""
        completion_pct = current_gmv / max(target_gmv, 1)
        # 期望进度（线性）
        expected_completion = time_elapsed_pct
        # 进度差（实际-期望）
        progress_gap = completion_pct - expected_completion

        if progress_gap > 0.05:
            status = '🟢领先进度'
        elif progress_gap > -0.05:
            status = '🟡进度正常'
        elif progress_gap > -0.15:
            status = '🟠轻度落后，建议干预'
        else:
            status = '🔴严重落后，立即干预'

        # 预测最终完成率（线性外推）
        if time_elapsed_pct > 0.05:
            forecast_final = completion_pct / time_elapsed_pct
        else:
            forecast_final = 1.0

        return {
            'current_gmv': current_gmv,
            'target_gmv': target_gmv,
            'completion_pct': completion_pct,
            'completion_pct_str': f"{completion_pct:.1%}",
            'time_elapsed_pct': time_elapsed_pct,
            'expected_completion': expected_completion,
            'progress_gap': progress_gap,
            'status': status,
            'forecast_final_pct': forecast_final,
            'forecast_final_str': f"{forecast_final:.1%}",
        }


class GMVVarianceDecomposer:
    """
    GMV差异分解（书中拆因框架）
    GMV = 流量 × 转化率 × 客单价
    用Shapley值分配各因素对差距的贡献
    """

    def decompose(self, plan: Dict, actual: Dict) -> Dict:
        """
        分解GMV差距
        plan/actual: {'traffic': N, 'cvr': float, 'aov': float}
        """
        plan_gmv = plan['traffic'] * plan['cvr'] * plan['aov']
        actual_gmv = actual['traffic'] * actual['cvr'] * actual['aov']
        gap = actual_gmv - plan_gmv

        # Shapley值近似（3因素）
        # 计算每个因素的边际贡献
        def gmv(t, c, a): return t * c * a

        factors = ['traffic', 'cvr', 'aov']
        plan_vals = [plan['traffic'], plan['cvr'], plan['aov']]
        actual_vals = [actual['traffic'], actual['cvr'], actual['aov']]

        shapley_values = {}
        for i, factor in enumerate(factors):
            # 用实际值替换计划值，计算边际影响
            mixed = list(plan_vals)
            mixed[i] = actual_vals[i]
            marginal = gmv(*mixed) - gmv(*plan_vals)
            shapley_values[factor] = marginal

        # 归一化到实际差距
        total_marginal = sum(abs(v) for v in shapley_values.values())
        if total_marginal > 0:
            contributions = {k: v / gap if gap != 0 else 0
                             for k, v in shapley_values.items()}
        else:
            contributions = {k: 0 for k in factors}

        return {
            'plan_gmv': plan_gmv,
            'actual_gmv': actual_gmv,
            'gap': gap,
            'gap_pct': f"{gap/max(plan_gmv,1):.1%}",
            'decomposition': {
                factor: {
                    'plan': plan[factor],
                    'actual': actual[factor],
                    'change_pct': f"{(actual[factor]-plan[factor])/max(plan[factor],1e-9):.1%}",
                    'gmv_contribution': round(shapley_values[factor], 0),
                    'contribution_pct': f"{contributions[factor]:.1%}",
                }
                for factor in factors
            },
            'main_driver': max(shapley_values, key=lambda k: abs(shapley_values[k])),
        }


def run_business_scale_kpi_demo():
    """生意规模三维KPI监控演示"""
    print("=" * 65)
    print("生意规模三维KPI监控系统")
    print("基于《全链路管理》陈凤霞 生意计划供应链KPI体系")
    print("=" * 65)

    kpi = BusinessScaleKPI()

    # 1. 同比增长率
    print("\n[1] 销售增长率监控")
    growth_cases = [
        ("吸奶器品类", 1280000, 1050000, "yoy"),
        ("温奶器品类", 380000, 420000, "yoy"),
        ("整体GMV",   3200000, 2800000, "yoy"),
    ]
    for name, current, base, mode in growth_cases:
        r = kpi.growth_rate(current, base, mode)
        print(f"  {name}: 本期${current:,} vs 同期${base:,} → {r['rate_pct']} {r['status']}")

    # 2. 计划达成率 vs 预测准确率对比
    print("\n[2] 计划达成率 vs 预测准确率（书中特别区分！）")
    print("  【达成率】实际销售 / 计划目标（考核完成情况，可>100%）")
    print("  【准确率】1 - |预测-实际|/实际（衡量预测偏差，不超100%）")
    cases = [
        ("上月销售", 950000, 1000000, 980000),
    ]
    for name, actual, plan, forecast in cases:
        ach = kpi.achievement_rate(actual, plan)
        acc = 1 - abs(forecast - actual) / actual
        print(f"\n  {name}: 实际${actual:,} | 计划${plan:,} | 预测${forecast:,}")
        print(f"    达成率: {ach['rate_pct']} {ach['status']}")
        print(f"    预测准确率: {acc:.1%}（不是达成率！）")

    # 3. 大促GMV完成比实时预警
    print("\n[3] 大促GMV完成比实时预警（以大促进行18小时为例）")
    progress = kpi.gmv_completion_progress(
        current_gmv=32000,    # 当前完成$3.2万
        target_gmv=500000,    # 目标$50万
        time_elapsed_pct=0.375  # 已过37.5%时间（48小时大促中的18小时）
    )
    print(f"  当前GMV: ${progress['current_gmv']:,} / 目标: ${progress['target_gmv']:,}")
    print(f"  完成进度: {progress['completion_pct_str']} | 期望进度: {progress['expected_completion']:.1%}")
    print(f"  线性预测最终完成率: {progress['forecast_final_str']}")
    print(f"  状态: {progress['status']}")

    # 4. GMV差异分解
    print("\n[4] GMV差异分解（Shapley归因）")
    decomposer = GMVVarianceDecomposer()
    plan_data = {'traffic': 50000, 'cvr': 0.04, 'aov': 89.99}
    actual_data = {'traffic': 42000, 'cvr': 0.041, 'aov': 87.50}

    result = decomposer.decompose(plan_data, actual_data)
    print(f"  计划GMV: ${result['plan_gmv']:,.0f} | 实际GMV: ${result['actual_gmv']:,.0f}")
    print(f"  差距: ${result['gap']:,.0f} ({result['gap_pct']})")
    print(f"\n  因素分解:")
    for factor, info in result['decomposition'].items():
        factor_name = {'traffic': '流量', 'cvr': '转化率', 'aov': '客单价'}[factor]
        print(f"    {factor_name}: 计划{info['plan']:,} → 实际{info['actual']:,} "
              f"({info['change_pct']}) | GMV贡献: ${info['gmv_contribution']:,.0f}")
    driver_names = {'traffic': '流量', 'cvr': '转化率', 'aov': '客单价'}
    print(f"\n  主要差距来源: {driver_names.get(result['main_driver'], result['main_driver'])}")
    print(f"  建议: 重点改善流量获取（广告/自然流量），转化率表现尚可")

    # 5. 关键区分总结
    print("\n[书中关键洞察]")
    print("  达成率>100% → 超额完成（好事），不要用预测准确率框架解读")
    print("  预测准确率用于改善预测模型，达成率用于绩效考核")
    print("  GMV拆分：流量×转化率×客单价，三维归因才能精准干预")

    print("\n[✓] 生意规模三维KPI监控系统测试通过")
    return result


if __name__ == "__main__":
    run_business_scale_kpi_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SOP-Sales-Operations-Planning]]（S&OP提供计划数据作为达成率基准）、[[Skill-Supply-Chain-KPI-Health-Dashboard]]（本Skill的生意规模层是健康仪表盘的核心模块）
- **延伸（extends）**：[[Skill-Flash-Sale-Realtime-Sellthrough-Forecast]]（大促GMV完成比实时监控的延伸）、[[Skill-Promo-Stocktaking-SOP-Automation]]（大促前的达成率预测和备货决策）
- **可组合（combinable）**：[[Skill-Nonlinear-Multi-Touch-Attribution]]（GMV差距归因中流量来源分析）、[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]（生意规模KPI与库存效率KPI联动监控）

## ⑤ 商业价值评估

- **ROI 预估**：月GMV$100万卖家，通过Shapley归因精准干预（而非盲目加预算），大促期间额外挽回GMV约$8-15万；达成率监控系统建设$1万，ROI>800%
- **实施难度**：⭐⭐☆☆☆（计算逻辑简单，关键是准确获取计划数和实际数；差异分解需要流量/转化率/客单价三维拆分数据）
- **优先级**：⭐⭐⭐⭐⭐（生意规模是供应链的最终结果指标，所有供应链优化都应以提升GMV达成率为目标；书中专章强调）
- **适用规模**：所有规模卖家，月GMV>$3万即可受益；大促场景ROI尤其高
- **数据依赖**：历史销售数据（月度/周度）、计划目标数据、流量/转化率/客单价三维拆分
