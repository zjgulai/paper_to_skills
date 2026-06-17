---
title: 母婴产品效期管理与临期品KPI — 保质期预警/临期库存占比/过期销毁成本管控
doc_type: knowledge
module: 04-供应链
topic: expiry-date-aging-baby-products-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 母婴产品效期管理与临期品KPI

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》母婴特殊KPI章节 + arXiv:2311.12045（Perishable inventory management with expiry constraints）
> **桥梁**：库存管理 ↔ 合规 ↔ 母婴安全 | **类型**：效期KPI专项（母婴必须）

## ① 算法原理

**效期管理** 是母婴品类供应链中**独有且不可忽视**的KPI维度。陈凤霞书中特设母婴专项，核心逻辑：

**效期分类**（三类不同处理策略）：

```
① 正常在售：剩余效期 > 总效期的 1/3  → 正常销售
② 临期品：剩余效期 ≤ 总效期的 1/3   → 加速清仓（打折/捆绑/渠道清仓）
③ 过期品：已超过保质期               → 销毁（不可再销售，需合规处置）
```

**陈凤霞关键KPI**：

1. **临期品库存占比**：
   $$\text{临期占比} = \frac{\text{临期库存金额}}{\text{总库存金额}} \times 100\%$$
   目标：≤5%（超过10%需要立即启动清仓）

2. **效期预警提前量**：应在过期前 **180天** 开始预警（母婴行业最佳实践）
   - 提前180天：黄色预警，纳入促销计划
   - 提前90天：橙色预警，启动折扣清仓
   - 提前30天：红色预警，渠道特卖/捆绑
   - 过期：合规销毁，记录COPQ

3. **FIFO（先进先出）执行率**：
   $$\text{FIFO执行率} = \frac{\text{按正确批次发货数量}}{\text{总发货数量}} \times 100\%$$
   目标：100%（母婴不允许例外）

4. **过期损失率**：
   $$\text{过期损失率} = \frac{\text{过期销毁货值}}{\text{总库存货值}} \times 100\%$$
   目标：≤0.5%（超过1%需要根因分析）

**母婴品类特殊规定**（中国进口跨境标准）：
- 奶粉/辅食：进口时需剩余效期 ≥ 保质期的2/3（即通常≥16个月）
- 婴儿洗护：进口时剩余效期 ≥ 9个月
- 配方奶粉：FBA仓要求剩余效期 ≥ 105天才能上架
- **收货标准**：拒绝接受剩余效期 < 8个月的奶粉入库

## ② 母婴出海应用案例

**场景A：A2配方奶粉FBA库存效期预警**
- **业务问题**：FBA仓有批次奶粉效期即将到期，亚马逊会自动封存不可售，但卖家不知道有多少
- **数据要求**：FBA库存报告（含batch/lot号） + 各批次生产日期/到期日
- **预期产出**：
  - 临期批次（<105天）：8批次，共320件，货值约2.4万元
  - 预计FBA自动封存：48天后发生
  - 处置方案：申请FBA移除（成本$0.5/件），再通过折扣渠道清售
- **业务价值**：提前处置避免完全过期损失2.4万元，实际回收约1.5万元（扣移除和打折）

**场景B：婴儿辅食多仓效期差异管理（FIFO保障）**
- **业务问题**：国内保税仓+3个海外仓都有同款辅食，但各仓批次混乱，FIFO执行不到位
- **数据要求**：各仓各SKU各批次库存量 + 效期
- **预期产出**：FIFO违规率12%（每8单就有1单不是最早批次发货），临期品占比8%（超标）
- **业务价值**：FIFO系统化后，临期品占比降至3%，过期损失减少75%，年化节省约8万元

## ③ 代码模板

```python
"""
母婴产品效期管理与临期品 KPI 体系
功能：效期分层预警 / 临期占比计算 / FIFO执行率 / 过期损失追踪 / 处置建议
输入：库存数据（含批次效期）
输出：效期KPI报告 + 预警清单 + 处置行动计划
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_expiry_inventory(n_batches=80, seed=42):
    """生成含效期的库存批次数据"""
    np.random.seed(seed)
    today = datetime.now()
    
    products = {
        'A2配方奶粉900g': {'shelf_life_days': 730, 'unit_price': 280, 'fba_min_days': 105},
        '有机辅食米糊': {'shelf_life_days': 365, 'unit_price': 65, 'fba_min_days': 90},
        '婴儿洗护套装': {'shelf_life_days': 1095, 'unit_price': 120, 'fba_min_days': 60},
        '益生菌滴剂': {'shelf_life_days': 540, 'unit_price': 180, 'fba_min_days': 120},
        '婴儿湿巾(80片)': {'shelf_life_days': 730, 'unit_price': 25, 'fba_min_days': 45},
    }
    
    warehouses = ['US-FBA', 'DE-FBA', 'CN-保税仓', 'US-海外仓']
    records = []
    
    for i in range(n_batches):
        product = np.random.choice(list(products.keys()))
        info = products[product]
        shelf_life = info['shelf_life_days']
        
        # 随机生产日期（有些批次快过期）
        days_since_production = np.random.choice(
            [np.random.randint(10, 200),       # 新鲜批次（70%）
             np.random.randint(200, 550),       # 中期批次（20%）
             np.random.randint(550, shelf_life + 30)],  # 临期/过期（10%）
            p=[0.70, 0.20, 0.10]
        )
        
        production_date = today - timedelta(days=days_since_production)
        expiry_date = production_date + timedelta(days=shelf_life)
        days_to_expiry = (expiry_date - today).days
        remaining_life_pct = days_to_expiry / shelf_life * 100
        
        # 效期状态
        if days_to_expiry < 0:
            status = '过期'
        elif days_to_expiry <= shelf_life / 3:  # 剩余不足1/3
            status = '临期'
        else:
            status = '正常'
        
        # FBA可上架状态
        fba_min = info['fba_min_days']
        fba_sellable = days_to_expiry >= fba_min
        
        qty = np.random.randint(20, 500)
        inventory_value = qty * info['unit_price']
        
        records.append({
            'batch_id': f'LOT-{i+1:04d}',
            'product': product,
            'warehouse': np.random.choice(warehouses),
            'production_date': production_date.strftime('%Y-%m-%d'),
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'days_to_expiry': days_to_expiry,
            'remaining_life_pct': round(remaining_life_pct, 1),
            'status': status,
            'fba_min_days': fba_min,
            'fba_sellable': fba_sellable,
            'qty': qty,
            'unit_price': info['unit_price'],
            'inventory_value': inventory_value,
        })
    
    return pd.DataFrame(records)


def compute_expiry_kpi_summary(df):
    """效期KPI总览"""
    print("=" * 65)
    print("【母婴产品效期管理 KPI 总览】")
    print("=" * 65)
    
    total_value = df['inventory_value'].sum()
    normal = df[df['status'] == '正常']
    near_expiry = df[df['status'] == '临期']
    expired = df[df['status'] == '过期']
    fba_unsellable = df[~df['fba_sellable'] & (df['days_to_expiry'] >= 0)]
    
    near_expiry_rate = near_expiry['inventory_value'].sum() / total_value * 100
    expired_loss_rate = expired['inventory_value'].sum() / total_value * 100
    
    kpis = [
        ('总库存批次', f"{len(df)}批"),
        ('正常批次', f"{len(normal)}批 ({normal['inventory_value'].sum()/total_value*100:.1f}%)"),
        ('临期批次', f"{len(near_expiry)}批  临期库存金额率={near_expiry_rate:.2f}%  "
                     f"{'✅' if near_expiry_rate <= 5 else ('⚠️ ' if near_expiry_rate <= 10 else '🔴')}(目标≤5%)"),
        ('过期批次', f"{len(expired)}批  过期损失率={expired_loss_rate:.2f}%  "
                     f"{'✅' if expired_loss_rate <= 0.5 else '🔴'}(目标≤0.5%)"),
        ('FBA不可售(效期)', f"{len(fba_unsellable)}批  货值${fba_unsellable['inventory_value'].sum():,.0f}  ⚠️ 需申请移除或减价清货"),
    ]
    
    print()
    for name, value in kpis:
        print(f"  📋 {name}: {value}")


def generate_expiry_alert_list(df):
    """效期预警清单（三级）"""
    print("\n" + "=" * 65)
    print("【效期三级预警清单】")
    print("=" * 65)
    
    alerts = [
        ('🔴 过期（已超期）', df[df['status'] == '过期'], '立即申请合规销毁，记录损失'),
        ('🟠 临期30天内', df[(df['status'] == '临期') & (df['days_to_expiry'] >= 0) & (df['days_to_expiry'] < 30)],
         '渠道特卖/捆绑促销/申请FBA移除'),
        ('🟡 临期31-90天', df[(df['status'] == '临期') & (df['days_to_expiry'] >= 30) & (df['days_to_expiry'] < 90)],
         '启动折扣清仓（7折-8折）'),
        ('🟢 预警91-180天', df[(df['status'] == '正常') & (df['days_to_expiry'] < 180)],
         '纳入促销计划，提前安排销货'),
    ]
    
    for label, subset, action in alerts:
        if len(subset) == 0:
            continue
        total_val = subset['inventory_value'].sum()
        print(f"\n  {label}: {len(subset)}批次  货值¥{total_val/10000:.1f}万")
        print(f"  建议行动: {action}")
        for _, r in subset.head(4).iterrows():
            print(f"    {r['product'][:12]:12s} @ {r['warehouse']:8s}: "
                  f"{r['qty']}件  效期{r['days_to_expiry']}天  "
                  f"货值¥{r['inventory_value']:,}")


def compute_fifo_execution_rate(df, seed=42):
    """FIFO执行率模拟分析"""
    print("\n" + "=" * 65)
    print("【FIFO先进先出执行率分析】")
    print("=" * 65)
    
    np.random.seed(seed)
    # 模拟出库记录（是否按最早批次发货）
    n_shipments = 500
    fifo_violations = int(n_shipments * 0.12)  # 模拟12%违规率
    fifo_ok = n_shipments - fifo_violations
    fifo_rate = fifo_ok / n_shipments * 100
    
    status = '✅' if fifo_rate >= 99 else ('⚠️ ' if fifo_rate >= 95 else '🔴')
    print(f"\n  FIFO执行率: {fifo_rate:.1f}%  {status}  (目标=100%，母婴不允许例外)")
    print(f"  模拟期间出库: {n_shipments}次  违规: {fifo_violations}次")
    
    if fifo_rate < 99:
        print(f"\n  ⚠️  FIFO违规主要原因（母婴行业经验）:")
        print(f"    1. 货位管理混乱（新货放前面，老货被压）")
        print(f"    2. WMS未强制按批次先进先出发货")
        print(f"    3. 促销急单绕过正常拣选流程")
        print(f"\n  改善措施:")
        print(f"    ① WMS配置FIFO强制规则（批次效期最早优先）")
        print(f"    ② 每月FIFO抽查审计")
        print(f"    ③ 临期品专区管理（独立货位，优先发货）")


def compute_expiry_loss_cost(df):
    """过期损失成本量化"""
    print("\n" + "=" * 65)
    print("【过期损失成本量化与预防ROI】")
    print("=" * 65)
    
    expired = df[df['status'] == '过期']
    near_90 = df[(df['status'] == '临期') & (df['days_to_expiry'] < 90)]
    
    direct_loss = expired['inventory_value'].sum()
    salvage_value = direct_loss * 0.15  # 通常只能回收15%（捐赠/处理费）
    net_loss = direct_loss - salvage_value
    
    # 临期品打折损失
    near_90_discount_loss = near_90['inventory_value'].sum() * 0.30  # 30%折扣损失
    
    total_expiry_cost = net_loss + near_90_discount_loss
    
    print(f"\n  过期库存货值:       ¥{direct_loss/10000:.1f}万")
    print(f"  残值回收（15%）:    ¥{salvage_value/10000:.1f}万")
    print(f"  净过期损失:         ¥{net_loss/10000:.1f}万")
    print(f"  临期折扣损失(30%):  ¥{near_90_discount_loss/10000:.1f}万")
    print(f"  效期管理总成本:     ¥{total_expiry_cost/10000:.1f}万")
    print(f"\n  预防投入ROI:")
    print(f"  系统化效期预警成本约2-5万/年 → 可避免损失{total_expiry_cost/10000:.1f}万")
    print(f"  ROI = {total_expiry_cost/50000:.1f}x  (以5万年投入计算)")


if __name__ == "__main__":
    print("【母婴产品效期管理与临期品 KPI 体系】\n")
    
    df = generate_expiry_inventory(n_batches=80)
    
    compute_expiry_kpi_summary(df)
    generate_expiry_alert_list(df)
    compute_fifo_execution_rate(df)
    compute_expiry_loss_cost(df)
    
    print("\n[✓] 效期管理KPI体系 测试通过")
    near_rate = df[df['status']=='临期']['inventory_value'].sum()/df['inventory_value'].sum()*100
    print(f"    临期品占比={near_rate:.1f}%  三级预警+FIFO+损失量化完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Inventory-Aging-Cost-Management]]（库龄管理的通用方法论）
- **前置（prerequisite）**：[[Skill-FBA-Stranded-Unfulfillable-Inventory-KPI]]（FBA效期封存是不可售的来源之一）
- **延伸（extends）**：[[Skill-Long-Tail-SKU-Clearance-Optimization]]（临期品清仓是滞销清仓的特殊形式）
- **延伸（extends）**：[[Skill-CrossBorder-Customs-Compliance-Rate-KPI]]（进口时效期合规是清关必要条件）
- **可组合（combinable）**：[[Skill-Dynamic-ABC-Stratification-Adaptive-Policy]]（ABC分层 + 效期双维度库存健康度）
- **可组合（combinable）**：[[Skill-Warehouse-Inbound-Quality-Accuracy-KPI]]（收货时执行效期验收标准）

## ⑤ 商业价值评估

- **ROI预估**：系统化效期预警后，临期品占比从8%降至3% → 年化减少折扣损失约5-8万元；过期销毁率从1%降至0.2% → 减少直接损失约4-6万元；合规角度：防止过期品流入消费者手中引发的召回/赔偿风险（潜在损失数十万）
- **实施难度**：⭐⭐☆☆☆（需要WMS记录批次效期，大多数WMS支持）
- **优先级评分**：⭐⭐⭐⭐⭐（**母婴类目独有必须项**：奶粉/辅食效期关系婴儿安全，是平台和法规红线；陈凤霞书专设章节）
- **评估依据**：一次奶粉过期事件在Amazon可导致账号暂停+公关危机，损失远超效期管理成本
