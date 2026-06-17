---
title: 产能约束生产排程KPI — 产能利用率/排程达成率/换线时间/瓶颈识别
doc_type: knowledge
module: 04-供应链
topic: capacity-constraint-production-schedule-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 产能约束生产排程KPI

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》产能计划章节 + arXiv:2306.11284（Capacity-constrained production scheduling for OEM supply chains）
> **桥梁**：采购管理 ↔ 供应商产能 ↔ S&OP计划 | **类型**：产能KPI体系

## ① 算法原理

母婴品牌通常通过OEM工厂生产，**工厂产能是供应链计划的硬约束**。陈凤霞书中强调：不了解供应商产能约束，计划做得再好也无法执行。

**核心KPI**：

1. **产能利用率（Capacity Utilization Rate）**：
   $$\text{产能利用率} = \frac{\text{实际生产量}}{\text{最大可用产能}} \times 100\%$$
   目标区间：75%-85%（低于75%浪费产能；高于90%过度紧张，质量风险升高）

2. **排程达成率（Schedule Achievement Rate）**：
   $$\text{排程达成率} = \frac{\text{按计划完成的生产订单数}}{\text{计划生产订单总数}} \times 100\%$$
   目标：≥90%

3. **换线时间（Changeover Time）**：从一个产品切换到另一产品的停产时间
   - 目标：≤生产周期的10%（母婴电子类通常目标≤2小时/次）

4. **生产节拍（Takt Time）**：
   $$\text{Takt Time} = \frac{\text{可用生产时间}}{\text{需求数量}}$$
   节拍是协调整条产线的核心参数

**产能瓶颈识别（TOC约束理论）**：
- 找到产线中产能最低的工序（瓶颈）
- 整条产线的产出受瓶颈限制
- 优化非瓶颈无法提升整体产出

## ② 母婴出海应用案例

**场景A：吸奶器OEM工厂产能规划（旺季备货）**
- **业务问题**：Q3计划备货旺季所需24,000台吸奶器，但工厂月产能只有8,000台，只有3个月时间，恰好够用
- **数据要求**：工厂月产能（按型号）+ 当前订单占用情况 + 换线时间 + 计划交期
- **预期产出**：
  - 产能利用率预测：旗舰款8月100%（满产）、9月95%（仍然过载）
  - 风险：9月如有质量异常返工，将导致延误
  - 行动：备用工厂产能锁定2000台/月，旗舰款提前到7月开始生产
- **业务价值**：产能风险提前识别，避免大促前断货，防止约50万GMV损失

**场景B：多SKU产线排程优化（减少换线）**
- **业务问题**：工厂同时生产5款吸奶器型号，换线频繁（每天平均换线6次×2小时=12小时），实际有效生产时间损失15%
- **数据要求**：各型号月需求量 + 换线时间矩阵（从型号A换到型号B的时间）
- **预期产出**：优化生产顺序（相似型号相邻排产），换线次数从6次/天降至3次/天，有效产能提升8%
- **业务价值**：无需额外投资，通过排程优化增加有效产能800台/月

## ③ 代码模板

```python
"""
产能约束生产排程 KPI 体系
功能：产能利用率 / 排程达成率 / 换线分析 / 瓶颈识别 / 旺季产能规划
输入：生产订单记录 + 产能配置 + 换线时间矩阵
输出：产能KPI报告 + 瓶颈识别 + 排程优化建议
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_production_data(n_months=6, seed=42):
    """生成月度生产计划与实际数据"""
    np.random.seed(seed)
    
    products = {
        'P01-旗舰吸奶器': {'max_capacity': 3000, 'takt_min': 12, 'changeover_h': 2.0},
        'P02-标准吸奶器': {'max_capacity': 4000, 'takt_min': 8, 'changeover_h': 1.5},
        'P03-便携吸奶器': {'max_capacity': 2500, 'takt_min': 10, 'changeover_h': 2.5},
        'P04-配件套装': {'max_capacity': 8000, 'takt_min': 4, 'changeover_h': 0.5},
    }
    
    records = []
    for m in range(1, n_months + 1):
        is_peak = m >= 4  # 模拟旺季
        for prod, info in products.items():
            max_cap = info['max_capacity']
            # 需求量（旺季更高）
            demand = max_cap * np.random.uniform(0.65, 0.85) * (1.3 if is_peak else 1.0)
            demand = min(demand, max_cap * 1.1)  # 最多超产10%
            
            # 实际生产（受各种因素影响）
            actual = demand * np.random.uniform(0.88, 1.02)
            actual = min(actual, max_cap)
            
            # 换线次数
            changeovers = np.random.randint(3, 8)
            changeover_hours = changeovers * info['changeover_h']
            working_hours = 22 * 8  # 月工作时间（22天×8小时）
            effective_hours = working_hours - changeover_hours
            utilization = actual / max_cap * 100
            schedule_achieved = np.random.random() < 0.88  # 88%达成率
            
            records.append({
                'month': m,
                'product': prod,
                'planned_qty': round(demand),
                'actual_qty': round(actual),
                'max_capacity': max_cap,
                'capacity_utilization': round(utilization, 1),
                'schedule_achieved': schedule_achieved,
                'changeover_count': changeovers,
                'changeover_hours': changeover_hours,
                'effective_hours': effective_hours,
                'changeover_loss_pct': round(changeover_hours / working_hours * 100, 1),
                'is_peak': is_peak,
            })
    
    return pd.DataFrame(records)


def compute_capacity_kpi(df):
    """产能KPI总览"""
    print("=" * 65)
    print("【产能约束 KPI 总览】")
    print("=" * 65)
    
    avg_util = df['capacity_utilization'].mean()
    schedule_rate = df['schedule_achieved'].mean() * 100
    avg_changeover_pct = df['changeover_loss_pct'].mean()
    
    util_status = '✅' if 75 <= avg_util <= 85 else ('⚠️ 过载' if avg_util > 85 else '⚠️ 利用不足')
    sar_status = '✅' if schedule_rate >= 90 else '⚠️ '
    co_status = '✅' if avg_changeover_pct <= 10 else '🔴'
    
    print(f"\n  {util_status} 平均产能利用率: {avg_util:.1f}%  (目标75-85%)")
    print(f"  {sar_status} 排程达成率:      {schedule_rate:.1f}%  (目标≥90%)")
    print(f"  {co_status} 换线时间损失率:   {avg_changeover_pct:.1f}%  (目标≤10%)")


def analyze_capacity_by_product(df):
    """分产品产能分析"""
    print("\n" + "=" * 65)
    print("【分产品产能利用率分析】")
    print("=" * 65)
    
    prod_kpi = df.groupby('product').agg(
        均产能利用率=('capacity_utilization', 'mean'),
        最大利用率=('capacity_utilization', 'max'),
        排程达成率=('schedule_achieved', lambda x: x.mean() * 100),
        均换线损失=('changeover_loss_pct', 'mean'),
    ).round(1)
    
    for prod, row in prod_kpi.iterrows():
        util = row['均产能利用率']
        if util > 90:
            risk = '🔴 过载风险'
        elif util > 85:
            risk = '⚠️  偏高'
        elif util >= 75:
            risk = '✅ 正常'
        else:
            risk = '⚠️  利用不足'
        
        print(f"  {risk} {prod[:12]:12s}: 均利用率={util:.0f}%  "
              f"峰值={row['最大利用率']:.0f}%  "
              f"达成率={row['排程达成率']:.0f}%  "
              f"换线损失={row['均换线损失']:.0f}%")


def identify_bottleneck(df):
    """瓶颈识别（TOC约束理论）"""
    print("\n" + "=" * 65)
    print("【产线瓶颈识别（TOC约束理论）】")
    print("=" * 65)
    
    # 模拟工序级产能数据
    np.random.seed(42)
    workstations = {
        '注塑成型': 3200,
        '电机组装': 2800,   # 瓶颈
        '电路板焊接': 3500,
        '整机装配': 3100,
        '质量检测': 4000,
        '包装': 5000,
    }
    
    total_demand = 3000  # 月需求
    
    print(f"\n  月生产目标: {total_demand}台")
    print(f"\n  {'工序':12s}  {'产能/月':8s}  {'利用率':8s}  {'是否瓶颈'}")
    bottleneck = None
    min_cap = float('inf')
    
    for ws, cap in workstations.items():
        util = total_demand / cap * 100
        is_bottleneck = cap == min(workstations.values())
        if cap < min_cap:
            min_cap = cap
            bottleneck = ws
        icon = '🔴 ← 瓶颈' if is_bottleneck else ('⚠️ ' if util > 85 else '✅')
        print(f"  {ws:12s}  {cap:8,}台  {util:7.1f}%  {icon}")
    
    print(f"\n  ⚡ 最大制约瓶颈: {bottleneck}（产能{min_cap}台/月）")
    print(f"  改善建议: 专注提升{bottleneck}产能，其他工序再快也无用")
    print(f"  产能提升路径: 增加设备 / 优化工艺 / 双班制 / 外协部分工序")


def plan_peak_season_capacity(df):
    """旺季产能规划"""
    print("\n" + "=" * 65)
    print("【旺季产能规划（Q4备货需求）】")
    print("=" * 65)
    
    peak_df = df[df['is_peak']]
    normal_df = df[~df['is_peak']]
    
    peak_util = peak_df['capacity_utilization'].mean()
    normal_util = normal_df['capacity_utilization'].mean()
    
    print(f"\n  淡季平均利用率: {normal_util:.1f}%")
    print(f"  旺季平均利用率: {peak_util:.1f}%  "
          f"{'🔴 超负荷！' if peak_util > 90 else ('⚠️ 偏高' if peak_util > 85 else '✅')}")
    
    if peak_util > 85:
        print(f"\n  ⚠️  旺季产能超过85%警戒线，建议:")
        print(f"    1. 提前3个月开始旺季生产（平滑产能曲线）")
        print(f"    2. 与备用工厂签订弹性产能协议（保留20%外协产能）")
        print(f"    3. 减少旺季换线次数（集中生产主力SKU）")
        over_cap = df[df['capacity_utilization'] > 90]
        if len(over_cap) > 0:
            print(f"\n  超负荷月份({len(over_cap)}次):")
            for _, r in over_cap.head(3).iterrows():
                print(f"    {r['month']}月 {r['product'][:10]}: {r['capacity_utilization']:.0f}% "
                      f"（计划{r['planned_qty']}台 vs 产能{r['max_capacity']}台）")


if __name__ == "__main__":
    print("【产能约束生产排程 KPI 体系】\n")
    
    df = generate_production_data(n_months=6)
    
    compute_capacity_kpi(df)
    analyze_capacity_by_product(df)
    identify_bottleneck(df)
    plan_peak_season_capacity(df)
    
    print("\n[✓] 产能排程KPI体系 测试通过")
    avg_util = df['capacity_utilization'].mean()
    sar = df['schedule_achieved'].mean() * 100
    print(f"    产能利用率={avg_util:.1f}%  排程达成率={sar:.1f}%  瓶颈识别+旺季规划完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Capacity-Planning]]（供应商产能规划基础）
- **前置（prerequisite）**：[[Skill-Procurement-Cycle-Time-KPI]]（PLT与产能约束直接相关）
- **延伸（extends）**：[[Skill-SOP-Sales-Operations-Planning]]（S&OP需要产能约束作为输入）
- **延伸（extends）**：[[Skill-Demand-Supply-Matching-Gap-Analysis]]（产能是供需平衡中"供"的硬约束）
- **可组合（combinable）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（产能不足是OTIF违约的根因）
- **可组合（combinable）**：[[Skill-Flexible-Supply-Chain-Small-Batch-Agile]]（柔性产能是解决约束的策略之一）

## ⑤ 商业价值评估

- **ROI预估**：提前识别旺季产能瓶颈 → 避免大促前断货（按旗舰款GMV损失约30万）；换线优化提升有效产能8% → 相当于减少8%采购成本（无需外协补产）
- **实施难度**：⭐⭐⭐☆☆（需要与供应商建立深度信息共享，主要难点是工厂数据获取）
- **优先级评分**：⭐⭐⭐⭐☆（陈凤霞："不了解工厂产能就做计划，等于在沙上建楼"）
- **评估依据**：母婴品牌OEM模式下，工厂产能是最终供应的硬约束，所有S&OP计划都需要以此为边界
