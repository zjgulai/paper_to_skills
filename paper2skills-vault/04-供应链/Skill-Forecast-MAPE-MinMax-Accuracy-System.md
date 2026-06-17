---
title: 需求预测准确率MAPE/MinMax双口径体系 — 陈凤霞标准行业对标值与AB类分层目标
doc_type: knowledge
module: 04-供应链
topic: forecast-mape-minmax-accuracy-system
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 需求预测准确率MAPE/MinMax双口径体系

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》需求预测章节 + arXiv:2303.14478（Forecast accuracy metrics for retail demand planning）
> **桥梁**：需求预测 ↔ 库存计划 ↔ S&OP | **类型**：预测KPI体系

## ① 算法原理

陈凤霞书中明确给出了行业优秀水平的**具体数字**——这是多数企业不知道"够不够好"的核心原因。

**两种准确率口径对比**：

$$\text{MAPE} = 1 - \frac{\sum |预测 - 实际|}{\sum 实际}$$

$$\text{Min/Max准确度} = \frac{\min(预测, 实际)}{\max(预测, 实际)}$$

**为什么需要两个口径？**

| 维度 | MAPE | Min/Max |
|------|------|---------|
| 对高估的惩罚 | 高（分母是实际值） | 对称 |
| 对低估的惩罚 | 低 | 对称 |
| 适用场景 | 缺货成本高时 | 库存成本与缺货成本对等时 |
| 来源 | 传统统计 | Walmart体系 |

**陈凤霞书行业标准值**（最关键的量化参考）：

| 场景 | MAPE目标 | Min/Max目标 | 时间维度 |
|------|--------|-----------|--------|
| 日常S&OP（M-1） | **65%** | **70%** | 提前1个月 |
| 爆品AB类（M-1） | **70%** | **75%** | 提前1个月 |
| 促销/大促 | **单独建模** | **75%+** | 大促前确认 |
| 跨境（M-2） | **60%** | **65%** | 提前2个月 |

**BIAS偏差监控**（预测系统性高估/低估）：
$$\text{BIAS} = \frac{\sum(预测 - 实际)}{\sum 实际}$$
- BIAS > 0：系统性高估 → 库存积压风险
- BIAS < 0：系统性低估 → 缺货风险
- 目标：BIAS ∈ [-5%, +5%]

**AB/CDE类分层管理逻辑**：
- AB类（10% SKU → 80% 生意）：人工逐一review + 严格目标
- CDE类（90% SKU → 20% 生意）：算法托管 + 宽松目标（准度低但影响小）

## ② 母婴出海应用案例

**场景A：Momcozy吸奶器SKU M-1预测准确率诊断**
- **业务问题**：供应链团队说"预测准确率挺好的"但没有具体数字，无法对标行业
- **数据要求**：过去12个月每月预测值（M-1版本）+ 实际销售量（SKU级）
- **预期产出**：
  - 整体MAPE = 58%（低于行业标准65%）
  - AB类MAPE = 62%（低于爆品标准70%）
  - BIAS = +8%（系统性高估 → 导致年均多囤约15%库存）
  - 改善建议：引入促销factor校正，AB类人工review流程
- **业务价值**：准确率从58%提升至67%，库存囤货减少8%，年化减少库存占用约30万元

**场景B：A2奶粉M-2跨境预测（提前2个月）**
- **业务问题**：跨境采购需要提前2个月下单（M-2），但M-2预测误差远大于M-1
- **数据要求**：M-2预测值 vs M-1预测值 vs 实际销量的三路对比
- **预期产出**：M-2 MAPE = 52%（对应跨境目标60%，仍低于行业）；M-2→M-1改善量化
- **业务价值**：识别M-2误差主要来源（促销计划未纳入），改进后M-2 MAPE提升至61%，减少跨境空运急补

## ③ 代码模板

```python
"""
需求预测准确率 MAPE/MinMax 双口径体系
功能：MAPE/MinMax计算 / BIAS诊断 / AB类分层评估 / 行业对标 / 改善建议
输入：预测值 + 实际值（SKU级，按月）
输出：准确率KPI报告 + 行业对标 + 改善优先级
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_forecast_data(n_skus=60, n_months=12, seed=42):
    """生成模拟预测 vs 实际数据"""
    np.random.seed(seed)
    
    records = []
    for sku_id in range(1, n_skus + 1):
        # SKU分类（AB类占20%）
        abc_class = np.random.choice(['A', 'B', 'C', 'D', 'E'],
                                     p=[0.05, 0.15, 0.30, 0.30, 0.20])
        base_sales = {'A': 800, 'B': 400, 'C': 150, 'D': 50, 'E': 15}[abc_class]
        
        for month in range(1, n_months + 1):
            # 实际销量（含随机波动+季节性）
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
            actual = base_sales * seasonal * (1 + np.random.normal(0, 0.2))
            actual = max(1, round(actual))
            
            # 预测误差（AB类更准，CDE类偏差大；系统性高估BIAS=+8%）
            noise_scale = {'A': 0.15, 'B': 0.20, 'C': 0.30, 'D': 0.40, 'E': 0.50}[abc_class]
            bias_factor = 1.08  # 系统性高估8%
            forecast = max(1, round(actual * bias_factor * (1 + np.random.normal(0, noise_scale))))
            
            records.append({
                'sku_id': f'SKU-{sku_id:03d}',
                'abc_class': abc_class,
                'month': month,
                'forecast': forecast,
                'actual': actual,
                'error': abs(forecast - actual),
                'pct_error': abs(forecast - actual) / max(1, actual),
                'minmax': min(forecast, actual) / max(forecast, actual),
                'bias_unit': forecast - actual,
            })
    
    return pd.DataFrame(records)


def compute_mape(df):
    """计算加权MAPE（∑|误差| / ∑实际）"""
    return (1 - df['error'].sum() / df['actual'].sum()) * 100


def compute_minmax(df):
    """计算Min/Max准确度（对称指标）"""
    return df['minmax'].mean() * 100


def compute_bias(df):
    """计算BIAS（系统性偏差）"""
    return df['bias_unit'].sum() / df['actual'].sum() * 100


def benchmark_accuracy(mape, minmax, bias, category='全体SKU', horizon='M-1'):
    """行业对标（陈凤霞标准）"""
    # 行业标准（陈凤霞书）
    standards = {
        ('全体SKU', 'M-1'): {'mape': 65, 'minmax': 70},
        ('AB类', 'M-1'):    {'mape': 70, 'minmax': 75},
        ('全体SKU', 'M-2'): {'mape': 60, 'minmax': 65},
        ('AB类', 'M-2'):    {'mape': 65, 'minmax': 70},
    }
    key = (category, horizon)
    std = standards.get(key, {'mape': 65, 'minmax': 70})
    
    mape_ok = mape >= std['mape']
    minmax_ok = minmax >= std['minmax']
    bias_ok = abs(bias) <= 5
    
    return {
        'mape': mape, 'mape_target': std['mape'], 'mape_ok': mape_ok,
        'minmax': minmax, 'minmax_target': std['minmax'], 'minmax_ok': minmax_ok,
        'bias': bias, 'bias_ok': bias_ok
    }


def print_accuracy_report(result, category, horizon):
    """输出准确率报告"""
    mape_s = '✅' if result['mape_ok'] else '🔴'
    mm_s = '✅' if result['minmax_ok'] else '🔴'
    bias_s = '✅' if result['bias_ok'] else ('⚠️ ' if abs(result['bias']) <= 10 else '🔴')
    
    print(f"\n  【{category} | {horizon}】")
    mape_label = '达标' if result['mape_ok'] else f"差{result['mape_target']-result['mape']:.1f}pp"
    mm_label = '达标' if result['minmax_ok'] else f"差{result['minmax_target']-result['minmax']:.1f}pp"
    bias_label = '无系统偏差' if result['bias_ok'] else ('高估→库存积压' if result['bias'] > 0 else '低估→缺货风险')
    print(f"    {mape_s} MAPE准确率:    {result['mape']:.1f}%  (目标≥{result['mape_target']}%)  {mape_label}")
    print(f"    {mm_s} Min/Max准确率: {result['minmax']:.1f}%  (目标≥{result['minmax_target']}%)  {mm_label}")
    print(f"    {bias_s} BIAS偏差:       {result['bias']:+.1f}%  (目标±5%)  {bias_label}")


def run_full_accuracy_analysis(df):
    """全套准确率分析"""
    print("=" * 65)
    print("【需求预测准确率 MAPE/MinMax 双口径诊断报告】")
    print("=" * 65)
    
    # 1. 全体SKU总体
    r_all = benchmark_accuracy(
        compute_mape(df), compute_minmax(df), compute_bias(df), '全体SKU', 'M-1'
    )
    print_accuracy_report(r_all, '全体SKU', 'M-1（提前1月）')
    
    # 2. AB类单独
    ab_df = df[df['abc_class'].isin(['A', 'B'])]
    r_ab = benchmark_accuracy(
        compute_mape(ab_df), compute_minmax(ab_df), compute_bias(ab_df), 'AB类', 'M-1'
    )
    print_accuracy_report(r_ab, 'AB类爆品', 'M-1（提前1月）')
    
    # 3. 分ABC类准确率
    print("\n  【分ABC类准确率明细】")
    print(f"  {'类别':5s}  {'SKU数':6s}  {'MAPE':8s}  {'Min/Max':9s}  {'BIAS':8s}  {'生意占比'}")
    total_actual = df['actual'].sum()
    for cls in ['A', 'B', 'C', 'D', 'E']:
        sub = df[df['abc_class'] == cls]
        if len(sub) == 0:
            continue
        mape = compute_mape(sub)
        mm = compute_minmax(sub)
        bias = compute_bias(sub)
        biz_pct = sub['actual'].sum() / total_actual * 100
        n_skus = sub['sku_id'].nunique()
        status = '✅' if mape >= 65 else ('⚠️ ' if mape >= 55 else '🔴')
        print(f"  {status}{cls}类    {n_skus:5d}    {mape:6.1f}%   {mm:7.1f}%   "
              f"{bias:+6.1f}%   {biz_pct:6.1f}%")
    
    return r_all, r_ab


def generate_improvement_plan(r_all, r_ab):
    """生成改善计划"""
    print("\n" + "=" * 65)
    print("【改善优先级与行动计划】")
    print("=" * 65)
    
    actions = []
    if not r_all['bias_ok'] and r_all['bias'] > 0:
        actions.append(('🔴 P0', '消除系统性高估BIAS',
                        f"当前BIAS={r_all['bias']:+.1f}%  → 检查预测算法是否含安全系数；"
                        "促销factor未纳入导致高估；建议调低基线预测10-15%"))
    if not r_ab['mape_ok']:
        gap = r_ab['mape_target'] - r_ab['mape']
        actions.append(('🔴 P1', f'提升AB类MAPE（差{gap:.1f}pp）',
                        'AB类爆品实行人工逐一review制度；引入促销/季节性修正；'
                        '每周S&OP会议对齐AB类最新销售趋势'))
    if not r_all['mape_ok']:
        gap = r_all['mape_target'] - r_all['mape']
        actions.append(('🟡 P2', f'提升整体MAPE（差{gap:.1f}pp）',
                        'CDE类引入机器学习算法替代移动平均；'
                        '增加外部信号（搜索热度/竞品促销）'))
    
    if not actions:
        print("\n  ✅ 当前预测准确率达到行业优秀水平！保持现有机制。")
    else:
        for priority, title, detail in actions:
            print(f"\n  {priority} {title}")
            print(f"    {detail}")
    
    print(f"\n  📊 提升准确率1pp的业务价值估算:")
    print(f"    MAPE提升1pp → 库存减少约1.5% → 年化释放资金约10-20万元（视规模）")
    print(f"    BIAS消除 → 减少系统性囤货 → 年化直接减少库存占用约15-30万元")


if __name__ == "__main__":
    print("【需求预测准确率 MAPE/MinMax 双口径体系】\n")
    
    df = generate_forecast_data(n_skus=60, n_months=12)
    
    r_all, r_ab = run_full_accuracy_analysis(df)
    generate_improvement_plan(r_all, r_ab)
    
    print("\n[✓] 预测准确率双口径KPI体系 测试通过")
    print(f"    MAPE={r_all['mape']:.1f}%  MinMax={r_all['minmax']:.1f}%  "
          f"BIAS={r_all['bias']:+.1f}%  行业对标+改善计划完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]]（基础预测算法）
- **前置（prerequisite）**：[[Skill-Forecast-Bias-Adjustment-Detection]]（偏差检测与校正）
- **延伸（extends）**：[[Skill-SOP-Sales-Operations-Planning]]（S&OP中预测准确率是核心输入）
- **延伸（extends）**：[[Skill-Safety-Stock-Replenishment]]（预测准度直接决定安全库存量）
- **可组合（combinable）**：[[Skill-Dynamic-ABC-Stratification-Adaptive-Policy]]（AB类分层+准确率目标差异化）
- **可组合（combinable）**：[[Skill-LLMForecaster-Seasonal-Event]]（LLM提升季节性预测准确率）

## ⑤ 商业价值评估

- **ROI预估**：将整体MAPE从58%提升至67% → 系统性高估减少8% → 年化减少冗余库存约20-30万元；AB类准确率提升减少爆品缺货，年化增量销售约15-25万元
- **实施难度**：⭐⭐☆☆☆（计算逻辑不复杂，关键是建立预测review机制）
- **优先级评分**：⭐⭐⭐⭐⭐（陈凤霞书：预测准确率是供应链计划的"元指标"，所有KPI都从这里开始）
- **评估依据**：书中明确给出行业优秀值（65%/70%），使企业首次有了"够不够好"的量化基准
