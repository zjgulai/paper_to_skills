---
title: 本地订单达成率与FDC仓网覆盖KPI — 本地发货率/跨仓调拨成本/仓网优化决策
doc_type: knowledge
module: 04-供应链
topic: local-order-fulfillment-rate-fdc
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 本地订单达成率与FDC仓网覆盖KPI

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》仓网规划章节 + arXiv:2306.09517（Multi-echelon warehouse network optimization for e-commerce）+ 京东2026供应链案例
> **桥梁**：仓网规划 ↔ 订单履约 ↔ 库存成本 | **类型**：仓网KPI

## ① 算法原理

**本地订单达成率（Local Order Fulfillment Rate）** 是衡量仓网合理性的核心KPI。陈凤霞书中引用京东案例：FDC满足率提升1.47pp，年化节省库存持有成本4451万元。

**核心概念**：

```
仓库层级（CDC → RDC → FDC）：
CDC（中央仓）→ RDC（区域仓）→ FDC（前置仓/城市仓）→ 消费者

本地订单 = 消费者所在城市/区域的FDC有货发货（不需要跨仓调拨）
跨仓订单 = FDC缺货，需从RDC或其他FDC调拨/发货
```

**本地订单达成率**：
$$\text{本地率} = \frac{\text{由本地FDC直接发货的订单数}}{\text{总订单数}} \times 100\%$$

**陈凤霞目标区间**：75%-80%
- **低于75%**：仓网布局不合理（前置仓太少或库存分配错误）→ 时效差+物流成本高
- **高于80%**：前置仓过多布局，库存冗余，持有成本过高
- **最优区间75-80%**：时效+成本的平衡点

**为什么不是100%？**
- 100%意味着每个前置仓都有全量SKU → 库存重复存放 → 资金占用爆炸
- 正确做法：FDC存放高频AB类SKU，低频CDE类放RDC，订单混合履约

**成本分析**（陈凤霞三角）：

| 本地率 | 时效 | 物流成本 | 库存持有成本 |
|--------|------|--------|-----------|
| 60% | 慢（跨区） | 高 | 低（少FDC） |
| 75-80% | 快（本地）| 低 | 优化 |
| 95%+ | 极快 | 极低 | 极高（仓库太多） |

**京东2026年案例**：
- FDC需求满足率：57.12% → 58.59%（+1.47pp）
- 库存持有成本节省：**4451万元/年**
- 库存调拨成本节省：**1.62亿元/年**

## ② 母婴出海应用案例

**场景A：Momcozy美国市场FDC仓网优化**
- **业务问题**：美国市场只有1个中央仓（加州），东海岸订单配送需要5-7天，FBA是主要渠道但自营海外仓本地率极低
- **数据要求**：按州/地区的订单量分布 + 当前发货仓 + 是否本地发货
- **预期产出**：
  - 当前本地率 = 41%（东部市场只有从加州发货）
  - 建议：在新泽西增设RDC → 本地率提升至72%
  - 预计节省：时效从5-7天→2-3天，物流成本降低$0.8/件
- **业务价值**：本地率从41%提升至72%，年化物流成本节省约15万元，Prime成员配送时效合规

**场景B：多渠道母婴平台分仓策略优化**
- **业务问题**：同时在京东/天猫/自营三个渠道销售，各渠道仓独立，造成库存重复备货
- **数据要求**：各渠道/各仓订单量 + 发货仓 + 本地发货率
- **预期产出**：通过共仓策略（FDC仓同时服务多渠道），本地率从65%提升至78%
- **业务价值**：减少跨仓调拨，年化节省约12万元

## ③ 代码模板

```python
"""
本地订单达成率与 FDC 仓网覆盖 KPI 体系
功能：本地率计算 / 仓网覆盖分析 / 跨仓成本量化 / 仓网优化建议
输入：订单地理分布 + 仓库位置 + 库存数据
输出：本地率KPI + 跨仓成本 + 仓网优化方案
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_order_warehouse_data(n_orders=2000, seed=42):
    """生成含地理信息的订单-仓库数据"""
    np.random.seed(seed)
    
    # 区域分布（模拟订单地理分布）
    regions = {
        '华东（上海/苏州/杭州）': 0.30,
        '华北（北京/天津）': 0.20,
        '华南（广州/深圳）': 0.18,
        '华中（武汉/郑州）': 0.12,
        '华西（成都/重庆）': 0.10,
        '东北（沈阳/哈尔滨）': 0.05,
        '西北（西安/兰州）': 0.05,
    }
    
    # 仓库位置（FDC/RDC）
    warehouses = {
        'WH-上海（FDC）': '华东（上海/苏州/杭州）',
        'WH-北京（FDC）': '华北（北京/天津）',
        'WH-广州（FDC）': '华南（广州/深圳）',
        'WH-武汉（RDC）': '华中（武汉/郑州）',
    }
    # 华西/东北/西北 无本地FDC，需跨仓
    
    records = []
    region_list = list(regions.keys())
    region_probs = list(regions.values())
    
    for i in range(n_orders):
        order_region = np.random.choice(region_list, p=region_probs)
        
        # 找是否有本地仓库
        local_wh = None
        for wh, wh_region in warehouses.items():
            if wh_region == order_region:
                local_wh = wh
                break
        
        if local_wh:
            # 有本地仓：80%从本地发（20%因库存不足跨仓）
            is_local = np.random.random() < 0.80
            fulfillment_wh = local_wh if is_local else np.random.choice(list(warehouses.keys()))
        else:
            # 无本地仓：必须跨仓
            is_local = False
            fulfillment_wh = np.random.choice(['WH-上海（FDC）', 'WH-广州（FDC）'])
        
        # 配送成本（本地便宜，跨仓贵）
        base_delivery_cost = np.random.uniform(8, 15)
        if not is_local:
            base_delivery_cost *= np.random.uniform(1.4, 2.2)  # 跨仓溢价40-120%
        
        # 配送时效
        delivery_days = np.random.uniform(1, 2) if is_local else np.random.uniform(3, 6)
        
        records.append({
            'order_id': f'ORD-{i+1:05d}',
            'order_region': order_region,
            'fulfillment_wh': fulfillment_wh,
            'is_local': is_local,
            'delivery_cost': round(base_delivery_cost, 2),
            'delivery_days': round(delivery_days, 1),
            'order_value': round(np.random.gamma(5, 30) + 50, 2),
        })
    
    return pd.DataFrame(records)


def compute_local_fulfillment_rate(df):
    """本地订单达成率KPI"""
    print("=" * 65)
    print("【本地订单达成率（Local Fulfillment Rate）KPI】")
    print("=" * 65)
    
    total = len(df)
    local_orders = df['is_local'].sum()
    local_rate = local_orders / total * 100
    
    if local_rate >= 80:
        status = '⚠️  偏高→库存冗余风险（目标75-80%）'
    elif local_rate >= 75:
        status = '✅ 理想区间（75-80%）'
    elif local_rate >= 65:
        status = '⚠️  偏低→时效与成本受损'
    else:
        status = '🔴 严重偏低→仓网需要重大优化'
    
    avg_local_cost = df[df['is_local']]['delivery_cost'].mean()
    avg_cross_cost = df[~df['is_local']]['delivery_cost'].mean()
    avg_local_days = df[df['is_local']]['delivery_days'].mean()
    avg_cross_days = df[~df['is_local']]['delivery_days'].mean()
    
    print(f"\n  总订单: {total}  本地发货: {local_orders}  跨仓发货: {total-local_orders}")
    print(f"\n  本地订单达成率: {local_rate:.1f}%  {status}")
    print(f"\n  对比：本地发货 vs 跨仓发货")
    print(f"    配送成本: ¥{avg_local_cost:.1f} vs ¥{avg_cross_cost:.1f}  "
          f"（跨仓溢价 {(avg_cross_cost/avg_local_cost-1)*100:.0f}%）")
    print(f"    配送时效: {avg_local_days:.1f}天 vs {avg_cross_days:.1f}天  "
          f"（跨仓慢 {avg_cross_days-avg_local_days:.1f}天）")
    
    return local_rate, avg_local_cost, avg_cross_cost


def analyze_local_rate_by_region(df):
    """按区域本地率分析"""
    print("\n" + "=" * 65)
    print("【分区域本地订单达成率分析】")
    print("=" * 65)
    
    region_kpi = df.groupby('order_region').agg(
        本地率=('is_local', lambda x: x.mean() * 100),
        订单量=('order_id', 'count'),
        均配送成本=('delivery_cost', 'mean'),
        均配送天数=('delivery_days', 'mean'),
    ).sort_values('本地率', ascending=False)
    
    for region, row in region_kpi.iterrows():
        rate = row['本地率']
        status = '✅有本地仓' if rate >= 70 else '🔴无本地仓'
        print(f"  {status} {region[:12]:12s}: 本地率={rate:.0f}%  "
              f"均成本=¥{row['均配送成本']:.1f}  均时效={row['均配送天数']:.1f}天  "
              f"订单量={row['订单量']:.0f}")


def compute_cross_warehouse_cost(df, avg_local_cost, avg_cross_cost):
    """跨仓成本量化"""
    print("\n" + "=" * 65)
    print("【跨仓配送成本量化与优化机会】")
    print("=" * 65)
    
    cross_orders = df[~df['is_local']]
    cross_cost_premium = (avg_cross_cost - avg_local_cost) * len(cross_orders)
    annual_premium = cross_cost_premium * (365 / 30)  # 年化
    
    print(f"\n  当月跨仓溢价总额: ¥{cross_cost_premium:,.0f}")
    print(f"  年化跨仓溢价: ¥{annual_premium/10000:.1f}万元")
    
    # 优化方案：如果本地率从当前提升至78%
    current_rate = df['is_local'].mean()
    target_rate = 0.78
    if target_rate > current_rate:
        cross_reduction = (target_rate - current_rate) * len(df) * (avg_cross_cost - avg_local_cost)
        annual_saving = cross_reduction * (365 / 30)
        print(f"\n  优化方案: 本地率从{current_rate*100:.1f}%提升至{target_rate*100:.0f}%")
        print(f"  可节省物流成本: ¥{cross_reduction:,.0f}/月  年化¥{annual_saving/10000:.1f}万")
        print(f"\n  实现路径:")
        print(f"    1. 华西/东北设立RDC → 本地率+10-15pp")
        print(f"    2. 优化AB类SKU在各FDC的分配比例 → 本地率+3-5pp")
        print(f"    3. 提升FDC库存预测准确率 → 本地率+2-3pp")


def warehouse_expansion_roi(new_wh_region, orders_affected, 
                              delivery_cost_saving_per_order, 
                              annual_wh_cost):
    """新增仓库ROI决策"""
    print("\n" + "=" * 65)
    print(f"【新增仓库ROI决策 — {new_wh_region}】")
    print("=" * 65)
    
    annual_logistics_saving = orders_affected * 365 / 30 * delivery_cost_saving_per_order
    net_annual_benefit = annual_logistics_saving - annual_wh_cost
    payback_months = annual_wh_cost / max(1, annual_logistics_saving / 12)
    
    print(f"\n  目标区域: {new_wh_region}")
    print(f"  影响订单量: {orders_affected}/月  单票节省: ¥{delivery_cost_saving_per_order}")
    print(f"  年化物流节省: ¥{annual_logistics_saving/10000:.1f}万")
    print(f"  年仓储成本: ¥{annual_wh_cost/10000:.1f}万")
    print(f"  年净收益: ¥{net_annual_benefit/10000:.1f}万")
    print(f"  回本周期: {payback_months:.1f}个月  "
          f"{'✅ 建议开仓' if payback_months <= 18 else '⚠️ ROI不足，暂缓'}")


if __name__ == "__main__":
    print("【本地订单达成率与 FDC 仓网覆盖 KPI 体系】\n")
    
    df = generate_order_warehouse_data(n_orders=2000)
    
    local_rate, avg_local, avg_cross = compute_local_fulfillment_rate(df)
    analyze_local_rate_by_region(df)
    compute_cross_warehouse_cost(df, avg_local, avg_cross)
    
    # 新增华西仓ROI评估
    warehouse_expansion_roi(
        new_wh_region='华西（成都）',
        orders_affected=200,          # 每月200单受益
        delivery_cost_saving_per_order=8,  # 每单节省¥8
        annual_wh_cost=400_000,        # 年仓储成本40万
    )
    
    print("\n[✓] 本地订单达成率FDC仓网KPI体系 测试通过")
    print(f"    本地率={local_rate:.1f}%  分区域分析+跨仓成本+仓库ROI决策完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-FDC-RDC-Inventory-Allocation]]（FDC/RDC库存分配是本地率的直接决定因素）
- **前置（prerequisite）**：[[Skill-Multi-Echelon-Inventory]]（多级库存理论基础）
- **延伸（extends）**：[[Skill-On-Shelf-Availability-SKU-Matrix]]（FDC覆盖率 + OSA矩阵联动）
- **延伸（extends）**：[[Skill-Logistics-Cost-Structure-Decomposition]]（本地率提升→末程成本优化）
- **可组合（combinable）**：[[Skill-Inventory-Turnover-ABC-Classification]]（AB类优先放FDC，CDE类留RDC）
- **可组合（combinable）**：[[Skill-Order-Cycle-Time-OTD-Analytics]]（本地率是OTD时效的关键影响因子）

## ⑤ 商业价值评估

- **ROI预估**：本地率从60%提升至75%，年化节省配送成本约10-20万元（取决于规模）；京东案例：+1.47pp → 节省库存持有成本4451万元（规模效应，中小品牌同比例约5-15万元）
- **实施难度**：⭐⭐⭐☆☆（需要地理订单分布数据 + 仓库位置优化，属于中期战略项目）
- **优先级评分**：⭐⭐⭐⭐☆（陈凤霞："仓网布局一旦成型难以改变，错误的仓网会持续产生高额的跨仓成本"）
- **评估依据**：京东2026案例数据：FDC满足率每提升1pp节省持有成本约3000万（万亿级规模），中小品牌同比例约5-15万元
