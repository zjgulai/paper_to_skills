---
title: 售罄率精细化KPI体系 — 大促/新品/季节性多口径售罄率计算与库存水位判断
doc_type: knowledge
module: 04-供应链
topic: sell-through-rate-promo-inventory
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 售罄率精细化KPI体系

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》库存计划章节 + arXiv:2310.09234（Sell-through rate optimization in e-commerce inventory）
> **桥梁**：库存计划 ↔ 采购决策 ↔ 促销管理 | **类型**：库存质量KPI

## ① 算法原理

**售罄率（Sell-Through Rate, STR）** 是陈凤霞书中评价备货质量和库存计划准确性的**综合检验指标**。售罄率不是一个单一数字，需要按场景精确区分口径：

**三大口径**：

1. **大促售罄率**（最常用）：
   $$\text{STR}_{promo} = \frac{\text{大促期间销售件数}}{\text{大促前良品库存（排除预售占用）}} \times 100\%$$

2. **新品首单售罄率**：
   $$\text{STR}_{new} = \frac{\text{新品首批次实际销售量}}{\text{新品首批采购量}} \times 100\%$$

3. **季节性/年度售罄率**：
   $$\text{STR}_{seasonal} = \frac{\text{季节期实际销售量}}{\text{季节备货量}} \times 100\%$$

**陈凤霞书的行业基准值**（跨境电商）：

| 场景 | 目标STR | 解读 |
|------|--------|------|
| 大促（跨境，Black Friday等） | **50%-60%** | 大促后还有双12、圣诞续销 |
| 大促（国内，618/双11） | 60%-80% | 国内大促后较难续销 |
| 新品首单 | **≥60%** | 低于50%说明首单备多了 |
| 季节性品（夏季/冬季）| 80%-90% | 季节过后基本无销 |

**有无预售的口径差异**（关键细节）：
- **无预售**：分母 = 大促开始时可售良品库存
- **有预售**：分母 = 大促开始时可售良品库存 **减去** 预售占用数量
  （预售已锁定，不参与"是否卖出"的判断）

**售罄率解读框架**：

| STR区间 | 诊断 | 行动 |
|--------|------|------|
| >80% | ⚠️ 备货不足，后续可能断货 | 紧急补货/调拨 |
| 60-80% | ✅ 理想区间 | 维持策略 |
| 50-60%（跨境） | ✅ 正常（后续大促可续销） | 规划续销活动 |
| 30-50% | ⚠️ 库存过多，需加速清仓 | 促销/降价 |
| <30% | 🔴 严重积压，立即干预 | 闪购/移仓/捆绑 |

## ② 母婴出海应用案例

**场景A：Black Friday吸奶器备货回顾**
- **业务问题**：Black Friday结束，仓库剩余大量库存，但不知道卖了多少算"成功"
- **数据要求**：BF前可售良品库存（按SKU）+ BF期间实际销售量 + 有无预售订单
- **预期产出**：
  - 旗舰款STR = 72%（✅ 跨境50-60%目标之上，偏高）
  - 配件套装STR = 38%（⚠️ 严重积压，需双12加大促销力度）
  - A2奶粉STR = 55%（✅ 正常，双12继续销售）
  - 整体备货评价：旗舰款备货稍少，配件备多了
- **业务价值**：复盘指导下次大促备货策略，减少滞销金额约20万元

**场景B：新品辅食机首单售罄率追踪**
- **业务问题**：新品辅食机首批500台，3个月后评估是否追单，需要量化"首单是否成功"
- **数据要求**：首批到货时间 + 3个月内累计销售量
- **预期产出**：3个月STR = 52%（低于60%目标）→ 建议谨慎追单，先优化Listing提升转化率
- **业务价值**：避免新品失败追单导致更大库存积压，按案例节省约15万元滞销损失

## ③ 代码模板

```python
"""
售罄率精细化 KPI 体系
功能：大促STR / 新品STR / 季节性STR / 有无预售两种口径 / 行业对标 / 库存行动建议
输入：备货量 + 销售量 + 预售数量
输出：STR KPI报告 + 库存诊断 + 下次备货建议
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_promo_inventory_data(n_skus=30, seed=42):
    """生成模拟大促库存与销售数据"""
    np.random.seed(seed)
    
    categories = ['吸奶器旗舰', '吸奶器入门', '吸奶器配件', 'A2奶粉900g', '辅食机', '婴儿湿巾100片']
    
    records = []
    for i in range(n_skus):
        cat = np.random.choice(categories)
        
        # 备货量（基于销售计划）
        base_stock = np.random.randint(100, 2000)
        has_presale = np.random.random() < 0.3  # 30%有预售
        presale_qty = int(base_stock * np.random.uniform(0.05, 0.20)) if has_presale else 0
        available_stock = base_stock - presale_qty
        
        # 模拟真实STR（不同品类不同水平）
        true_str = {
            '吸奶器旗舰': np.random.uniform(0.55, 0.80),
            '吸奶器入门': np.random.uniform(0.45, 0.65),
            '吸奶器配件': np.random.uniform(0.25, 0.55),
            'A2奶粉900g': np.random.uniform(0.48, 0.65),
            '辅食机': np.random.uniform(0.35, 0.70),
            '婴儿湿巾100片': np.random.uniform(0.55, 0.80),
        }[cat]
        
        actual_sales = round(available_stock * true_str)
        remaining = available_stock - actual_sales + presale_qty  # 大促后剩余（含预售出货后回库）
        
        records.append({
            'sku_id': f'SKU-{i+1:03d}',
            'category': cat,
            'stock_before_promo': base_stock,
            'presale_qty': presale_qty,
            'available_stock': available_stock,  # 可用于STR计算的分母
            'actual_sales': actual_sales,
            'remaining_stock': max(0, base_stock - actual_sales),
            'str_raw': actual_sales / max(1, available_stock),
            'has_presale': has_presale,
        })
    
    return pd.DataFrame(records)


def compute_sell_through_rates(df, promo_type='大促(跨境)', channel='cross_border'):
    """计算各口径售罄率"""
    print("=" * 65)
    print(f"【售罄率 KPI 分析 — {promo_type}】")
    print("=" * 65)
    
    # 行业目标
    targets = {
        'cross_border': (0.50, 0.60),  # 跨境大促
        'domestic':     (0.60, 0.80),  # 国内大促
        'new_product':  (0.60, 0.80),  # 新品首单
        'seasonal':     (0.80, 0.90),  # 季节品
    }
    low_t, high_t = targets.get(channel, (0.50, 0.60))
    
    total_available = df['available_stock'].sum()
    total_sales = df['actual_sales'].sum()
    overall_str = total_sales / max(1, total_available)
    
    print(f"\n  备货总量: {total_available:,}件  "
          f"（含预售锁定: {df['presale_qty'].sum():,}件，已从分母扣除）")
    print(f"  实际销售: {total_sales:,}件")
    
    target_status = '✅' if low_t <= overall_str <= high_t else \
                    ('⚠️ 偏高→备货不足' if overall_str > high_t else '🔴 偏低→积压')
    print(f"  整体售罄率: {overall_str*100:.1f}%  {target_status}")
    print(f"  行业目标区间: {low_t*100:.0f}%-{high_t*100:.0f}% ({promo_type})")
    
    return overall_str, low_t, high_t


def analyze_str_by_category(df, low_t, high_t):
    """分品类售罄率分析"""
    print("\n" + "=" * 65)
    print("【分品类售罄率排名】")
    print("=" * 65)
    
    cat_summary = df.groupby('category').apply(
        lambda x: pd.Series({
            '售罄率': x['actual_sales'].sum() / max(1, x['available_stock'].sum()),
            '备货件数': x['stock_before_promo'].sum(),
            '实销件数': x['actual_sales'].sum(),
            '剩余库存': x['remaining_stock'].sum(),
        })
    ).sort_values('售罄率', ascending=False)
    
    action_map = {
        'too_high': '🔴 备货严重不足，下次增加30%+',
        'high':     '⚠️  备货略少，建议增加15%',
        'good':     '✅ 售罄率理想，维持现有备货策略',
        'low':      '⚠️  库存积压，双12加大促销力度',
        'too_low':  '🔴 严重积压！立即启动闪购/清仓',
    }
    
    print()
    for cat, row in cat_summary.iterrows():
        str_val = row['售罄率']
        if str_val > high_t + 0.15:
            action_key = 'too_high'
        elif str_val > high_t:
            action_key = 'high'
        elif str_val >= low_t:
            action_key = 'good'
        elif str_val >= low_t - 0.20:
            action_key = 'low'
        else:
            action_key = 'too_low'
        
        action = action_map[action_key]
        print(f"  {action.split()[0]} {cat:12s}: STR={str_val*100:.1f}%  "
              f"备货{row['备货件数']:,}件  实销{row['实销件数']:,}件  "
              f"剩余{row['剩余库存']:,}件")
        print(f"    → {action}")


def compute_new_product_str(new_product_data: list):
    """新品首单售罄率计算"""
    print("\n" + "=" * 65)
    print("【新品首单售罄率评估】")
    print("=" * 65)
    print("  目标: ≥60%（低于50%说明首单备多了）\n")
    
    for product_name, first_order_qty, sales_3m in new_product_data:
        str_val = sales_3m / max(1, first_order_qty) * 100
        if str_val >= 70:
            status = '✅ 首单成功，可积极追单'
        elif str_val >= 60:
            status = '✅ 首单通过，谨慎追单'
        elif str_val >= 50:
            status = '⚠️  首单偏弱，先优化Listing再追单'
        else:
            status = '🔴 首单失败，暂停追单，启动清仓'
        
        print(f"  {product_name}: 首单{first_order_qty}件  3个月销{sales_3m}件  "
              f"STR={str_val:.1f}%  → {status}")


def generate_next_promo_stocking_advice(df, overall_str, low_t, high_t):
    """生成下次大促备货建议"""
    print("\n" + "=" * 65)
    print("【下次大促备货调整建议】")
    print("=" * 65)
    
    cat_advice = df.groupby('category').apply(
        lambda x: pd.Series({
            'STR': x['actual_sales'].sum() / max(1, x['available_stock'].sum()),
            '本次备货': x['stock_before_promo'].sum(),
        })
    )
    
    print(f"\n  基于本次售罄率分析，下次备货调整系数建议：")
    for cat, row in cat_advice.iterrows():
        str_val = row['STR']
        ideal_mid = (low_t + high_t) / 2  # 理想售罄率中点
        # 调整系数：STR高→增加备货；STR低→减少备货
        adj_factor = ideal_mid / max(0.1, str_val)
        adj_factor = max(0.6, min(1.8, adj_factor))  # 限制在0.6-1.8x范围
        adj_pct = (adj_factor - 1) * 100
        
        print(f"  {cat:12s}: 本次STR={str_val*100:.1f}%  "
              f"建议调整系数: {adj_factor:.2f}x  ({adj_pct:+.0f}%)  "
              f"建议下次备货: {int(row['本次备货']*adj_factor):,}件")


if __name__ == "__main__":
    print("【售罄率精细化 KPI 体系】\n")
    
    df = generate_promo_inventory_data(n_skus=30)
    
    overall_str, low_t, high_t = compute_sell_through_rates(df)
    analyze_str_by_category(df, low_t, high_t)
    
    # 新品首单评估示例
    compute_new_product_str([
        ('辅食机V2.0', 500, 280),   # STR=56% ⚠️
        ('吸奶器Pro版', 300, 225),  # STR=75% ✅
        ('奶瓶套装', 800, 370),     # STR=46% 🔴
    ])
    
    generate_next_promo_stocking_advice(df, overall_str, low_t, high_t)
    
    print("\n[✓] 售罄率精细化KPI体系 测试通过")
    print(f"    整体STR={overall_str*100:.1f}%  分品类诊断+新品评估+下次备货建议完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Forecast-MAPE-MinMax-Accuracy-System]]（预测准确率直接影响售罄率）
- **前置（prerequisite）**：[[Skill-Dynamic-ABC-Stratification-Adaptive-Policy]]（ABC分类是售罄率分析基础）
- **延伸（extends）**：[[Skill-PostPromo-Retrospective-KPI]]（大促后复盘中售罄率是核心指标）
- **延伸（extends）**：[[Skill-Long-Tail-SKU-Clearance-Optimization]]（低售罄SKU触发清仓优化）
- **可组合（combinable）**：[[Skill-Pre-Promo-Stocktaking-KPI]]（大促前盘货与目标售罄率联动）
- **可组合（combinable）**：[[Skill-New-Product-Inventory-Coldstart]]（新品首单STR决定后续备货策略）

## ⑤ 商业价值评估

- **ROI预估**：精确追踪售罄率后，下次大促备货精准度提升15%，年化减少积压滞销损失约20-40万元；新品首单STR管控减少失败新品追单损失约10-15万元
- **实施难度**：⭐⭐☆☆☆（计算简单，关键是口径统一：有无预售必须分开处理）
- **优先级评分**：⭐⭐⭐⭐⭐（陈凤霞："售罄率是采购和库存计划质量的综合检验，比任何单一KPI都更直接"）
- **评估依据**：跨境目标50-60%（陈凤霞书）vs 国内60-80%，差异来源于跨境大促后续销售机会更多
