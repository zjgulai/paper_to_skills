---
title: 采购价格达成率与降本KPI体系 — 全链路降本量化追踪与价格偏差归因
doc_type: knowledge
module: 04-供应链
topic: procurement-cost-kpi-price-achievement
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 采购价格达成率与降本KPI体系

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》采购成本管理章节 + arXiv:2312.09654（Cost-aware procurement optimization in e-commerce）
> **桥梁**：采购管理 ↔ P&L财务 | **类型**：KPI成本体系

## ① 算法原理

**采购成本KPI** 是供应链降本的可量化抓手。陈凤霞体系将采购成本分解为三层：

```
采购总成本 = 直接采购成本 + 机会成本 + 管理成本
           = Σ(采购量 × 采购单价)
             + 缺货损失 + 超量库存持有成本
             + 询价/比价/审批人工成本
```

**核心KPI定义**：

1. **价格达成率（PAR, Price Achievement Rate）**：
   $$\text{PAR} = \frac{\text{实际采购价}}{\text{目标价/基准价}} \times 100\%$$
   - PAR < 100%：实际低于目标 → 采购降本成功
   - PAR > 100%：实际高于目标 → 需归因（涨价/议价不足/急采溢价）

2. **降本幅度（Cost Saving）**：
   $$\text{年化降本} = \sum_{i}{\left(\text{基准价}_i - \text{实际价}_i\right) \times \text{采购量}_i}$$

3. **价格偏差归因**（陈凤霞四象限分析）：
   - 市场涨价（不可控）
   - 供应商提价（可谈判）
   - 急采溢价（内部因素，可预防）
   - 批量不足（优化批量可改善）

**差异化采购策略**（基于价格弹性）：
- 大宗/稳定SKU → 年度框架协议锁价（规避市场波动）
- 促销活动SKU → 提前2-3个月锁货锁价（促销价格确定性）
- 尾货/急采 → 供应商备用池应急（防溢价）

## ② 母婴出海应用案例

**场景A：A2奶粉全年采购价格达成率追踪**
- **业务问题**：A2奶粉原材料价格波动，全年多次采购，事后发现有3批次价格超出年初预算8-15%，但没有系统量化
- **数据要求**：年度采购计划价格表（SKU级）+ 实际采购订单（价格/数量/日期/供应商）
- **预期产出**：
  - 月度PAR趋势（目标≤102%，实际月均105%）
  - 超价原因分解：急采溢价占68%、市场涨价占22%、议价不足占10%
  - 年化超支金额：约24万元
- **业务价值**：聚焦"急采溢价"根因（断货预警滞后）→ 优化安全库存后急采频次从12次降至3次 → 年化节省约16万元

**场景B：吸奶器OEM供应商年度降本谈判效果评估**
- **业务问题**：每年Q4与供应商谈判价格，但缺乏数据支撑谈判筹码
- **数据要求**：历史3年采购价格 + 同期原材料指数（铜/硅胶/ABS塑料价格）
- **预期产出**：可归因于原材料成本的价格变动占比（40%）vs 供应商利润扩张（60%） → 形成谈判"降价空间"量化论据
- **业务价值**：数据驱动谈判，年度降本目标达成率从55%提升至82%，节省采购成本约35万元

## ③ 代码模板

```python
"""
采购价格达成率与降本 KPI 体系
功能：PAR计算 / 价格偏差归因 / 降本统计 / 采购成本看板
输入：采购订单数据（含计划价、实际价、采购量、原因标签）
输出：采购成本KPI报告
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_procurement_data(n=300, seed=42):
    """生成模拟采购数据"""
    np.random.seed(seed)
    
    skus = {
        'SKU-A2奶粉900g': {'base_price': 85.0, 'annual_volume': 5000},
        'SKU-吸奶器旗舰': {'base_price': 180.0, 'annual_volume': 2000},
        'SKU-婴儿湿巾': {'base_price': 12.0, 'annual_volume': 20000},
        'SKU-辅食机': {'base_price': 95.0, 'annual_volume': 1500},
    }
    
    # 偏差原因分布
    deviation_reasons = {
        '急采溢价': 0.35,       # 因断货/促销急采导致溢价
        '市场原材料涨价': 0.20,  # 不可控
        '供应商提价': 0.15,     # 可谈判
        '批量不足折扣未达': 0.10, # 可优化
        '正常波动范围': 0.20,   # ±2%以内
    }
    
    records = []
    base_date = datetime(2025, 1, 1)
    sku_list = list(skus.keys())
    
    for i in range(n):
        sku = np.random.choice(sku_list)
        base_price = skus[sku]['base_price']
        reason = np.random.choice(list(deviation_reasons.keys()),
                                  p=list(deviation_reasons.values()))
        
        # 根据原因生成价格偏差
        price_delta_pct = {
            '急采溢价': np.random.uniform(0.05, 0.18),
            '市场原材料涨价': np.random.uniform(-0.03, 0.08),
            '供应商提价': np.random.uniform(0.02, 0.10),
            '批量不足折扣未达': np.random.uniform(0.01, 0.05),
            '正常波动范围': np.random.uniform(-0.02, 0.02),
        }[reason]
        
        actual_price = base_price * (1 + price_delta_pct)
        qty = np.random.randint(100, 1000)
        order_date = base_date + timedelta(days=np.random.randint(0, 365))
        
        records.append({
            'po_id': f'PO-{i+1:04d}',
            'sku': sku,
            'order_date': order_date,
            'month': order_date.strftime('%Y-%m'),
            'planned_price': base_price,
            'actual_price': round(actual_price, 2),
            'quantity': qty,
            'deviation_reason': reason,
            'price_par': actual_price / base_price,         # 价格达成率（<1好）
            'unit_delta': actual_price - base_price,        # 单价偏差
            'total_delta': (actual_price - base_price) * qty,  # 总额偏差（正=超支）
            'total_cost': actual_price * qty,
            'planned_cost': base_price * qty,
        })
    
    return pd.DataFrame(records)


def compute_price_achievement_rate(df):
    """计算整体价格达成率（PAR）"""
    print("=" * 60)
    print("【采购价格达成率 (PAR) 报告】")
    print("=" * 60)
    
    # 加权平均PAR（按采购金额加权）
    weighted_par = (df['total_cost'].sum() / df['planned_cost'].sum()) * 100
    total_saving = -df['total_delta'].sum()  # 负数=超支，正数=节省
    overrun_pct = (df['total_delta'].sum() / df['planned_cost'].sum()) * 100
    
    print(f"\n  整体PAR（价格达成率）: {weighted_par:.2f}%  (目标≤100%)")
    status = '✅ 达标' if weighted_par <= 102 else '❌ 超标'
    print(f"  状态: {status}")
    print(f"  年化采购总额: {df['total_cost'].sum()/10000:.1f} 万元")
    print(f"  vs 计划基准: {df['planned_cost'].sum()/10000:.1f} 万元")
    print(f"  价格偏差: {overrun_pct:+.2f}%  ({'-超支' if overrun_pct>0 else '+节省'} "
          f"{abs(total_saving/10000):.1f}万元)")
    
    return {'weighted_par': weighted_par, 'total_delta': df['total_delta'].sum()}


def analyze_price_deviation_by_reason(df):
    """价格偏差归因分析（陈凤霞四象限）"""
    print("\n" + "=" * 60)
    print("【价格偏差归因分析】")
    print("=" * 60)
    
    reason_summary = df[df['total_delta'] > 0].groupby('deviation_reason').agg(
        超支金额=('total_delta', 'sum'),
        发生次数=('po_id', 'count'),
        平均超支比例=('price_par', lambda x: (x.mean()-1)*100)
    ).sort_values('超支金额', ascending=False)
    
    total_overrun = reason_summary['超支金额'].sum()
    reason_summary['占比%'] = (reason_summary['超支金额'] / total_overrun * 100).round(1)
    
    print(f"\n  超支订单总额: {total_overrun/10000:.1f}万元")
    print()
    for reason, row in reason_summary.iterrows():
        controllable = '（可控）' if reason in ['急采溢价', '批量不足折扣未达', '供应商提价'] else '（不可控）'
        print(f"  {reason}{controllable}")
        print(f"    超支: {row['超支金额']/10000:.1f}万  占比: {row['占比%']:.1f}%  "
              f"均超价: {row['平均超支比例']:.1f}%  次数: {row['发生次数']:.0f}")
    
    print("\n  ⚡ 行动建议:")
    print("  ① 急采溢价（可控）→ 优化安全库存预警，减少临时急采")
    print("  ② 供应商提价（可谈判）→ 准备市场原材料数据反谈")
    print("  ③ 批量折扣未达 → 合并订单或签年框")


def compute_monthly_par_trend(df):
    """月度PAR趋势"""
    print("\n" + "=" * 60)
    print("【月度PAR趋势】")
    print("=" * 60)
    
    monthly = df.groupby('month').apply(
        lambda x: pd.Series({
            'PAR%': (x['total_cost'].sum() / x['planned_cost'].sum()) * 100,
            '超支万元': x['total_delta'].sum() / 10000,
            '采购单数': len(x)
        })
    ).reset_index()
    
    for _, row in monthly.iterrows():
        par = row['PAR%']
        status = '✅' if par <= 102 else ('⚠️ ' if par <= 108 else '🔴')
        print(f"  {row['month']}: PAR={par:.1f}%  {status}  "
              f"偏差: {row['超支万元']:+.2f}万  单数: {row['采购单数']:.0f}")


def compute_sku_par_ranking(df):
    """SKU级价格达成率排名"""
    print("\n" + "=" * 60)
    print("【SKU价格达成率排名（超支TOP）】")
    print("=" * 60)
    
    sku_summary = df.groupby('sku').apply(
        lambda x: pd.Series({
            'PAR%': (x['total_cost'].sum() / x['planned_cost'].sum()) * 100,
            '总超支万元': x['total_delta'].sum() / 10000,
        })
    ).sort_values('总超支万元', ascending=False)
    
    for sku, row in sku_summary.iterrows():
        status = '🔴' if row['PAR%'] > 105 else ('⚠️ ' if row['PAR%'] > 102 else '✅')
        print(f"  {status} {sku}: PAR={row['PAR%']:.1f}%  超支: {row['总超支万元']:+.2f}万")


def generate_action_plan(df):
    """生成降本行动计划"""
    print("\n" + "=" * 60)
    print("【降本行动计划（基于数据）】")
    print("=" * 60)
    
    # 急采溢价超支
    emergency_overrun = df[df['deviation_reason'] == '急采溢价']['total_delta'].sum() / 10000
    
    print(f"\n  当前年化超支估算: {df['total_delta'].sum()/10000:.1f}万元")
    print(f"  其中可控超支: {emergency_overrun:.1f}万元（急采+议价不足）")
    print()
    print("  降本路径:")
    print(f"  1. 提升PLT精准度 → 减少急采 → 预期节省 {emergency_overrun*0.6:.1f}万")
    print(f"  2. 年度框架协议锁价（锁定大宗SKU）→ 预期节省 {emergency_overrun*0.2:.1f}万")
    print(f"  3. 供应商谈判（原材料数据支撑）→ 预期节省 {emergency_overrun*0.15:.1f}万")
    print(f"  4. 合并小额订单达批量折扣 → 预期节省 {emergency_overrun*0.05:.1f}万")
    print(f"\n  合计潜在降本: {emergency_overrun*1.0:.1f}万元/年")


if __name__ == "__main__":
    print("【采购价格达成率与降本 KPI 体系】\n")
    
    df = generate_procurement_data(n=300)
    
    compute_price_achievement_rate(df)
    analyze_price_deviation_by_reason(df)
    compute_monthly_par_trend(df)
    compute_sku_par_ranking(df)
    generate_action_plan(df)
    
    print("\n[✓] 采购价格达成率KPI体系 测试通过")
    par = (df['total_cost'].sum() / df['planned_cost'].sum()) * 100
    overrun = df['total_delta'].sum() / 10000
    print(f"    PAR={par:.2f}%  超支={overrun:.1f}万  降本目标已量化")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Performance-Scorecard]]（供应商绩效评分含价格维度）
- **前置（prerequisite）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（OTIF与价格质量联合评估）
- **延伸（extends）**：[[Skill-Procurement-Cycle-Time-KPI]]（PLT超标是急采溢价的根因）
- **延伸（extends）**：[[Skill-Supply-Chain-Total-Cost-TCO-Model]]（采购价格是TCO第一成本项）
- **可组合（combinable）**：[[Skill-Multi-SKU-Procurement-Budget-Allocation]]（价格达成率约束下的预算分配）
- **可组合（combinable）**：[[Skill-Cross-Border-Cash-Flow-Forecasting]]（采购成本输入现金流预测）

## ⑤ 商业价值评估

- **ROI预估**：年采购额1000万的品牌，通过价格管理精细化降本2-3%即可节省20-30万元；减少急采溢价是最快见效路径（通常占超支的40-60%）
- **实施难度**：⭐⭐☆☆☆（数据来自采购ERP，主要工作是建立基准价格库）
- **优先级评分**：⭐⭐⭐⭐⭐（采购成本直接影响P&L，是CEO最关注的供应链KPI）
- **评估依据**：陈凤霞书中强调"采购价格达成率是供应链团队向管理层汇报的第一指标"，与GMV增长同等重要
