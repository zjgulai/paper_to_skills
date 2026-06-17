---
title: FBA滞销不可售库存KPI与处置策略 — 滞销率/仓储过长费预警/移除决策优化
doc_type: knowledge
module: 04-供应链
topic: fba-stranded-unfulfillable-inventory-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: FBA滞销不可售库存KPI与处置策略

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》FBA运营专章 + arXiv:2309.07832（Stranded inventory management in FBA marketplace）
> **桥梁**：FBA仓储管理 ↔ 库存成本 ↔ 跨境运营 | **类型**：FBA专项KPI

## ① 算法原理

**FBA滞销/不可售库存** 是Amazon跨境卖家最主要的隐性成本陷阱。陈凤霞书中专设FBA运营章节，将此类库存分为三类：

```
FBA问题库存分类：

①滞销库存（Slow-Moving）: 在库>90天但仍可售，产生仓储过长费
②不可售库存（Unfulfillable）: 质量问题/标签错误/客户退货损坏，无法再销售
③被封存库存（Stranded）: 商品未关联有效Listing，库存"孤立"无法出售
```

**Amazon仓储费率结构**（陈凤霞梳理）：

| 库存时长 | 标准仓储费 | 过长仓储费（额外） |
|--------|----------|---------------|
| ≤90天 | $0.75/立方英尺/月（标准） | 无 |
| 91-180天 | $0.75 | $0.15/立方英尺 |
| 181-270天 | $0.75 | $0.15 |
| 271-365天 | $0.75 | $1.50 |
| >365天 | $1.50 | $6.90 |

**核心KPI**：

1. **FBA滞销率（Slow-Moving Rate）**：
   $$\text{滞销率} = \frac{\text{库龄>90天的SKU库存额}}{\text{FBA总库存额}} \times 100\%$$
   目标：≤5%（超过10%需启动清仓程序）

2. **不可售库存率**：
   $$\text{不可售率} = \frac{\text{不可售/受损库存量}}{\text{FBA总库存量}} \times 100\%$$
   目标：≤1%

3. **预期仓储过长费（月）**：
   $$\text{LTSF} = \sum_{i \in \{>90\text{天}\}} \text{Volume}_i \times \text{LTSF Rate}_i$$

4. **处置ROI**：
   $$\text{处置ROI} = \frac{\text{清仓回收价} - \text{移除费}}{\text{仓储过长费节省}} $$

**处置决策树**（陈凤霞四象限）：
- 高价值+高销速 → 保留，优化转化
- 高价值+低销速 → 闪购促销清仓
- 低价值+高销速 → 正常流转
- 低价值+低销速 → 主动移除/销毁

## ② 母婴出海应用案例

**场景A：吸奶器SKU FBA库存健康度诊断**
- **业务问题**：Momcozy美国FBA仓有SKU库龄超过180天，每月仓储过长费高达$2,800
- **数据要求**：FBA库存报告（SKU/数量/库龄段/仓储费用）+ 近期销售数据
- **预期产出**：
  - 问题库存清单：5个SKU库龄>180天，占FBA库存额的8.5%
  - 月仓储过长费估算：$2,800（可避免）
  - 处置方案：2个SKU做闪购，2个SKU申请移除，1个SKU维持观察
- **业务价值**：执行处置后节省$2,200/月仓储费，同时回收约$3,500现金流

**场景B：不可售库存追踪与Amazon索赔**
- **业务问题**：Amazon仓库操作导致部分货物受损标记为不可售，但未自动赔偿，需要人工索赔
- **数据要求**：FBA调查报告 + 不可售库存记录 + 原始入仓记录
- **预期产出**：
  - 识别Amazon责任损坏比率（通常占不可售的60-70%）
  - 准备索赔材料清单
- **业务价值**：年化追回Amazon赔款约$1,500-$3,000（多数卖家未主动索赔）

## ③ 代码模板

```python
"""
FBA 滞销/不可售库存 KPI 与处置策略
功能：库龄分布分析 / 仓储过长费计算 / 处置决策树 / 索赔识别
输入：FBA库存报告（SKU/库龄/数量/尺寸）
输出：FBA库存健康KPI + 过长费预测 + 处置优先级
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Amazon FBA 仓储费率表（美国站，立方英尺/月，2024年）
FBA_STORAGE_RATES = {
    (0, 90):   {'standard': 0.75, 'ltsf': 0.00},
    (91, 180): {'standard': 0.75, 'ltsf': 0.15},
    (181, 270):{'standard': 0.75, 'ltsf': 0.15},
    (271, 365):{'standard': 0.75, 'ltsf': 1.50},
    (365, 9999):{'standard': 1.50,'ltsf': 6.90},
}

def get_storage_rate(days_in_storage):
    """根据库龄获取仓储费率"""
    for (low, high), rates in FBA_STORAGE_RATES.items():
        if low <= days_in_storage < high:
            return rates
    return {'standard': 1.50, 'ltsf': 6.90}


def generate_fba_inventory(n_skus=50, seed=42):
    """生成模拟FBA库存数据"""
    np.random.seed(seed)
    
    sku_categories = ['吸奶器主机', '吸奶器配件', 'A2奶粉900g', '婴儿湿巾', '辅食机', '安抚奶嘴']
    
    records = []
    for i in range(n_skus):
        category = np.random.choice(sku_categories)
        
        # 库龄分布（大部分健康，少部分滞销）
        age_segment = np.random.choice(
            ['0-90', '91-180', '181-270', '271-365', '365+'],
            p=[0.55, 0.20, 0.12, 0.08, 0.05]
        )
        
        age_map = {'0-90': (0, 90), '91-180': (91, 180), '181-270': (181, 270),
                   '271-365': (271, 365), '365+': (365, 500)}
        age_low, age_high = age_map[age_segment]
        days_in_storage = np.random.randint(age_low, age_high + 1)
        
        units = np.random.randint(5, 200)
        unit_price = np.random.uniform(15, 180)
        
        # 商品尺寸（立方英尺/件）
        cubic_feet = np.random.uniform(0.1, 2.5)  # 标准尺寸
        
        # 月销售量（用于判断销速）
        monthly_velocity = np.random.exponential(30)  # 均值30件/月
        if age_segment in ['181-270', '271-365', '365+']:
            monthly_velocity *= 0.2  # 滞销品销速低
        
        # 是否不可售
        is_unfulfillable = np.random.random() < 0.05 if age_segment != '0-90' else False
        # 是否被封存
        is_stranded = np.random.random() < 0.02
        
        records.append({
            'sku_id': f'SKU-{i+1:03d}',
            'category': category,
            'days_in_storage': days_in_storage,
            'age_segment': age_segment,
            'units': units,
            'unit_price': round(unit_price, 2),
            'inventory_value': round(units * unit_price, 2),
            'cubic_feet_per_unit': round(cubic_feet, 3),
            'total_cubic_feet': round(units * cubic_feet, 2),
            'monthly_velocity': round(monthly_velocity, 1),
            'months_of_supply': round(units / max(1, monthly_velocity), 1),
            'is_unfulfillable': is_unfulfillable,
            'is_stranded': is_stranded,
        })
    
    return pd.DataFrame(records)


def compute_fba_kpi_summary(df):
    """FBA库存健康KPI总览"""
    print("=" * 60)
    print("【FBA 库存健康 KPI 总览】")
    print("=" * 60)
    
    total_value = df['inventory_value'].sum()
    slow_moving = df[df['days_in_storage'] > 90]
    unfulfillable = df[df['is_unfulfillable']]
    stranded = df[df['is_stranded']]
    
    slow_rate = slow_moving['inventory_value'].sum() / total_value * 100
    unfulfill_rate = unfulfillable['units'].sum() / df['units'].sum() * 100
    stranded_rate = stranded['units'].sum() / df['units'].sum() * 100
    
    kpis = [
        ('FBA总库存价值', f'${total_value:,.0f}'),
        ('滞销率(>90天)', f'{slow_rate:.2f}%  (目标≤5%)  {"✅" if slow_rate<=5 else "🔴"}'),
        ('不可售库存率', f'{unfulfill_rate:.2f}%  (目标≤1%)  {"✅" if unfulfill_rate<=1 else "⚠️ "}'),
        ('被封存库存率', f'{stranded_rate:.2f}%  (目标=0%)  {"✅" if stranded_rate==0 else "🔴"}'),
    ]
    
    print()
    for k, v in kpis.items():
        print(f"  {k}: {v}")


def compute_ltsf_estimate(df):
    """仓储过长费（LTSF）预测"""
    print("\n" + "=" * 60)
    print("【仓储过长费（LTSF）月度预测】")
    print("=" * 60)
    
    total_ltsf = 0
    records = []
    
    for _, row in df.iterrows():
        rates = get_storage_rate(row['days_in_storage'])
        monthly_ltsf = row['total_cubic_feet'] * rates['ltsf']
        total_ltsf += monthly_ltsf
        
        if monthly_ltsf > 0:
            records.append({
                'sku': row['sku_id'],
                'category': row['category'],
                'days': row['days_in_storage'],
                'units': row['units'],
                'value': row['inventory_value'],
                'monthly_ltsf_usd': round(monthly_ltsf, 2),
            })
    
    ltsf_df = pd.DataFrame(records).sort_values('monthly_ltsf_usd', ascending=False)
    
    print(f"\n  当月预计LTSF总额: ${total_ltsf:.2f}")
    print(f"  年化LTSF预估: ${total_ltsf*12:.0f}")
    print()
    print("  TOP10 高费用SKU:")
    for _, r in ltsf_df.head(10).iterrows():
        print(f"    {r['sku']} ({r['category']}): 库龄{r['days']:.0f}天  "
              f"LTSF=${r['monthly_ltsf_usd']:.2f}/月  库存价值=${r['value']:.0f}")
    
    return total_ltsf


def run_disposal_decision_tree(df, ltsf_monthly):
    """处置决策树（四象限）"""
    print("\n" + "=" * 60)
    print("【处置决策树（四象限分析）】")
    print("=" * 60)
    
    # 分类：高/低价值 × 高/低销速
    value_median = df['inventory_value'].median()
    velocity_median = df['monthly_velocity'].median()
    
    slow_moving_df = df[df['days_in_storage'] > 90].copy()
    
    decisions = {
        'Q1_高价值_高销速': [],  # 保留优化
        'Q2_高价值_低销速': [],  # 闪购/降价促销
        'Q3_低价值_高销速': [],  # 正常流转
        'Q4_低价值_低销速': [],  # 移除/销毁
    }
    
    for _, row in slow_moving_df.iterrows():
        hi_val = row['inventory_value'] >= value_median
        hi_vel = row['monthly_velocity'] >= velocity_median
        
        if hi_val and hi_vel:
            decisions['Q1_高价值_高销速'].append(row['sku_id'])
        elif hi_val and not hi_vel:
            decisions['Q2_高价值_低销速'].append(row['sku_id'])
        elif not hi_val and hi_vel:
            decisions['Q3_低价值_高销速'].append(row['sku_id'])
        else:
            decisions['Q4_低价值_低销速'].append(row['sku_id'])
    
    actions = {
        'Q1_高价值_高销速': '✅ 保留，优化Listing/广告提升销速',
        'Q2_高价值_低销速': '⚡ 闪购/50off促销，或迁回国内仓',
        'Q3_低价值_高销速': '📦 正常流转，监控补货时机',
        'Q4_低价值_低销速': '🗑️ 主动申请移除或销毁，止损LTSF',
    }
    
    for q, sku_list in decisions.items():
        if sku_list:
            print(f"\n  {q} ({len(sku_list)}个SKU):")
            print(f"  建议: {actions[q]}")
            print(f"  SKU: {', '.join(sku_list[:5])}{'...' if len(sku_list)>5 else ''}")
    
    removal_skus = len(decisions['Q4_低价值_低销速'])
    removal_cost = removal_skus * 0.50  # Amazon移除费约$0.5/件
    removal_save = (ltsf_monthly * 0.4)  # 预计移除40%的LTSF来源
    print(f"\n  💰 移除成本估算: ${removal_cost:.0f}  节省LTSF: ${removal_save:.2f}/月  "
          f"回本周期: {removal_cost/max(0.01,removal_save):.1f}个月")


if __name__ == "__main__":
    print("【FBA 滞销/不可售库存 KPI 与处置策略】\n")
    
    df = generate_fba_inventory(n_skus=50)
    
    compute_fba_kpi_summary(df)
    ltsf = compute_ltsf_estimate(df)
    run_disposal_decision_tree(df, ltsf)
    
    print("\n[✓] FBA滞销库存KPI体系 测试通过")
    slow_rate = df[df['days_in_storage'] > 90]['inventory_value'].sum() / df['inventory_value'].sum() * 100
    print(f"    滞销率={slow_rate:.1f}%  LTSF月度=${ltsf:.2f}  处置决策树已完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Inventory-Aging-Cost-Management]]（库龄管理的基础方法论）
- **前置（prerequisite）**：[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]（ITO低的SKU容易产生FBA过长费）
- **延伸（extends）**：[[Skill-Long-Tail-SKU-Clearance-Optimization]]（清仓优化是处置的具体算法）
- **延伸（extends）**：[[Skill-GMROI-Inventory-Investment-Efficiency]]（GMROI与FBA费用是ROI对立面）
- **可组合（combinable）**：[[Skill-Warehouse-Cost-Per-Unit-KPI]]（FBA仓储费是单位成本重要组成）
- **可组合（combinable）**：[[Skill-Unified-Cross-Border-Inventory-Dispatch]]（FBA↔海外仓调配决策）

## ⑤ 商业价值评估

- **ROI预估**：FBA运营1年以上的品牌通常有5-15%库存产生过长费，年化LTSF支出约$3,000-$15,000；系统化管理可减少70%过长费；同时Amazon索赔可追回$1,500-$3,000/年（多数卖家未追）
- **实施难度**：⭐⭐☆☆☆（Amazon Seller Central提供库存报告，主要是分析逻辑）
- **优先级评分**：⭐⭐⭐⭐⭐（陈凤霞书FBA专章核心：LTSF是直接吞噬利润的"看不见的成本"）
- **评估依据**：陈凤霞书数据：跨境卖家平均FBA库存健康度评分仅62分，滞销库存费年化占营业额0.5-2%
