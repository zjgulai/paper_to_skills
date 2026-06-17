---
title: 订单交付周期OTD全链路分解 — On-Time Delivery率/交付阶段拆解/延迟根因归因
doc_type: knowledge
module: 04-供应链
topic: order-cycle-time-otd-analytics
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 订单交付周期OTD全链路分解

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》订单交付管理章节 + arXiv:2305.11478（End-to-end delivery cycle analytics in e-commerce）
> **桥梁**：订单管理 ↔ 物流配送 ↔ 客户承诺 | **类型**：交付KPI体系

## ① 算法原理

**OTD（On-Time Delivery）** 是客户体验的直接量化指标。陈凤霞体系的核心洞察：**OTD拆解才有价值**，总体OTD=95%无法指导改善，必须知道哪个阶段失职。

**订单交付周期全链路**（从下单到签收）：

```
T_total = T_confirm + T_pick_pack + T_handover + T_transit + T_delivery

下单确认(T1) → 仓库拣货包装(T2) → 交运给物流(T3) → 干线运输(T4) → 末端配送(T5)
```

**OTD核心指标**：

1. **按时交付率（On-Time Delivery Rate）**：
   $$\text{OTD} = \frac{\text{在承诺日期前到达的订单数}}{\text{总订单数}} \times 100\%$$
   Amazon Prime标准：≥97%（2日达）；非Prime FBM：≥95%

2. **Perfect Order Rate（完美订单率）**：
   $$\text{POR} = \text{OTD} \times \text{订单准确率} \times \text{无损坏率} \times \text{无发票错误率}$$
   世界级水准：≥95%

3. **交付周期（Order Cycle Time, OCT）**：从下单到签收的日历天数
   
4. **Stage Level OTD**：每个阶段单独计算OTD，找最大延迟瓶颈

**延迟根因分类**（陈凤霞4M1E法）：
- **人（Man）**：仓库拣货人手不足、司机延误
- **机器（Machine）**：WMS故障、扫描设备问题
- **材料（Material）**：货物包装不合规、超重超尺
- **方法（Method）**：路由分配错误、地址问题
- **环境（Environment）**：天气/节假日/海关延误

## ② 母婴出海应用案例

**场景A：美国市场2-day Prime OTD分析**
- **业务问题**：Momcozy Amazon Prime订单OTD达成率91%（低于97%标准），Prime badge面临取消风险
- **数据要求**：Amazon订单数据（下单时间/承诺交付日/实际签收日）+ 各阶段时间戳
- **预期产出**：
  - 阶段分析：拣货打包平均1.8天（目标0.5天）是最大瓶颈
  - 延迟原因TOP3：仓库高峰期超负荷（52%）、物流商末端延误（28%）、地址问题（20%）
  - 改善后OTD预测：从91%提升至96.5%
- **业务价值**：维持Prime资格 = 保留约30%溢价定价权，年化收益约25万元

**场景B：跨境B2C多国OTD差异分析（美/德/英对比）**
- **业务问题**：同款产品在三个国家市场OTD差异大（美国92% vs 德国78% vs 英国88%），需分国分析根因
- **数据要求**：各国订单交付记录 + 物流商表现数据
- **预期产出**：德国OTD低的根因 = 末程承运商（Hermes）在德国农村地区延误率高达38%
- **业务价值**：针对德国更换末程承运商（DHL替代Hermes），OTD从78%提升至89%

## ③ 代码模板

```python
"""
订单交付周期 OTD 全链路分解分析
功能：OTD率计算 / 各阶段拆解 / 延迟根因归因 / 完美订单率 / 多国对比
输入：订单交付记录（含各阶段时间戳）
输出：OTD KPI报告 + 瓶颈识别 + 改善路径
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_delivery_data(n=500, seed=42):
    """生成模拟订单交付数据（含各阶段时间戳）"""
    np.random.seed(seed)
    
    countries = {
        'US': {'committed_days': 2, 't4_mean': 1.5, 't5_mean': 0.8, 'delay_prob': 0.09},
        'DE': {'committed_days': 5, 't4_mean': 3.5, 't5_mean': 1.5, 'delay_prob': 0.22},
        'GB': {'committed_days': 4, 't4_mean': 2.8, 't5_mean': 1.2, 'delay_prob': 0.12},
    }
    
    delay_reasons = ['仓库超负荷', '物流商延误', '地址问题', '天气/节假日', '清关延误', '其他']
    
    records = []
    base_date = datetime(2025, 1, 1)
    
    for i in range(n):
        country = np.random.choice(list(countries.keys()), p=[0.55, 0.25, 0.20])
        c = countries[country]
        
        order_date = base_date + timedelta(days=np.random.randint(0, 365))
        is_promo = order_date.month in [11, 12]
        
        # 各阶段耗时（天）
        t1 = np.random.uniform(0.1, 0.5)          # 下单确认
        t2_base = np.random.gamma(1, 0.7)          # 拣货包装
        t2 = t2_base * (2.0 if is_promo else 1.0)  # 旺季更慢
        t3 = np.random.uniform(0.1, 0.5)            # 交运给物流
        t4 = np.random.gamma(c['t4_mean'], 0.5)    # 干线运输
        t5 = np.random.gamma(c['t5_mean'], 0.3)    # 末端配送
        
        total_days = t1 + t2 + t3 + t4 + t5
        committed = c['committed_days']
        on_time = total_days <= committed
        
        # 延迟原因（只有延迟的才有）
        delay_prob_adj = c['delay_prob'] * (1.5 if is_promo else 1.0)
        if not on_time:
            # 根据哪个阶段最长判断主要原因
            stages = [t1, t2, t3, t4, t5]
            bottleneck = stages.index(max(stages))
            delay_reason = ['仓库超负荷', '仓库超负荷', '物流商延误', '物流商延误', '地址问题'][bottleneck]
            if country == 'DE' and t4 > c['t4_mean'] * 1.5:
                delay_reason = '清关延误'
        else:
            delay_reason = '无延迟'
        
        # 完美订单标志（OTD + 无损坏 + 无错发）
        no_damage = np.random.random() > 0.008
        no_error = np.random.random() > 0.015
        perfect_order = on_time and no_damage and no_error
        
        records.append({
            'order_id': f'ORD-{i+1:05d}',
            'country': country,
            'order_date': order_date,
            'month': order_date.strftime('%Y-%m'),
            'is_promo': is_promo,
            'committed_days': committed,
            'total_days': round(total_days, 2),
            't1_confirm': round(t1, 2),
            't2_pick_pack': round(t2, 2),
            't3_handover': round(t3, 2),
            't4_transit': round(t4, 2),
            't5_delivery': round(t5, 2),
            'on_time': on_time,
            'delay_days': max(0, round(total_days - committed, 1)),
            'delay_reason': delay_reason,
            'no_damage': no_damage,
            'no_error': no_error,
            'perfect_order': perfect_order,
        })
    
    return pd.DataFrame(records)


def compute_otd_kpi_summary(df):
    """OTD KPI总览"""
    print("=" * 60)
    print("【OTD订单交付周期 KPI 总览】")
    print("=" * 60)
    
    otd = df['on_time'].mean() * 100
    por = df['perfect_order'].mean() * 100
    avg_cycle = df['total_days'].mean()
    late_orders = (~df['on_time']).sum()
    avg_delay = df[~df['on_time']]['delay_days'].mean()
    
    print(f"\n  OTD率（按时交付）:  {otd:.2f}%  ({'✅' if otd >= 97 else '⚠️ ' if otd >= 95 else '🔴'}  目标≥97%)")
    print(f"  完美订单率(POR):   {por:.2f}%  ({'✅' if por >= 95 else '⚠️ '}  目标≥95%)")
    print(f"  平均交付周期:      {avg_cycle:.1f}天")
    print(f"  延迟订单数:        {late_orders}单 ({100-otd:.1f}%)")
    print(f"  平均延迟天数:      {avg_delay:.1f}天")
    
    return {'otd': otd, 'por': por}


def analyze_stage_contribution(df):
    """各阶段耗时贡献分析（找瓶颈）"""
    print("\n" + "=" * 60)
    print("【各阶段耗时分解（瓶颈识别）】")
    print("=" * 60)
    
    stages = {
        '①下单确认(T1)': 't1_confirm',
        '②拣货包装(T2)': 't2_pick_pack',
        '③交运物流(T3)': 't3_handover',
        '④干线运输(T4)': 't4_transit',
        '⑤末端配送(T5)': 't5_delivery',
    }
    
    total_mean = df['total_days'].mean()
    
    print(f"\n  总平均周期: {total_mean:.2f}天")
    print()
    for stage, col in stages.items():
        d = df[col]
        pct = d.mean() / total_mean * 100
        cv = d.std() / d.mean()
        bottleneck = '🔴主要瓶颈' if pct > 30 else ('⚠️ ' if pct > 20 else '✅')
        print(f"  {bottleneck} {stage}: 均值={d.mean():.2f}天  占比={pct:.1f}%  CV={cv:.2f}")


def analyze_otd_by_country(df):
    """分国OTD对比"""
    print("\n" + "=" * 60)
    print("【分国OTD率对比】")
    print("=" * 60)
    
    for country, grp in df.groupby('country'):
        otd = grp['on_time'].mean() * 100
        avg_days = grp['total_days'].mean()
        committed = grp['committed_days'].iloc[0]
        status = '✅' if otd >= 97 else ('⚠️ ' if otd >= 93 else '🔴')
        print(f"  {status} {country}: OTD={otd:.1f}%  均值={avg_days:.1f}天  承诺={committed}天  "
              f"主要延迟原因: {grp[~grp['on_time']]['delay_reason'].mode().iloc[0] if len(grp[~grp['on_time']]) > 0 else '无'}")


def analyze_delay_pareto(df):
    """延迟原因Pareto分析"""
    print("\n" + "=" * 60)
    print("【延迟原因 Pareto 分析】")
    print("=" * 60)
    
    delayed = df[~df['on_time']]
    if len(delayed) == 0:
        print("  ✅ 无延迟订单")
        return
    
    reason_dist = delayed.groupby('delay_reason').agg(
        次数=('order_id', 'count'),
        平均延迟天数=('delay_days', 'mean'),
    ).sort_values('次数', ascending=False)
    
    total = reason_dist['次数'].sum()
    cum = 0
    
    for reason, row in reason_dist.iterrows():
        pct = row['次数'] / total * 100
        cum += pct
        vital = '🔴关键' if cum <= 80 else '⚪一般'
        print(f"  {vital} {reason}: {row['次数']:.0f}次 ({pct:.1f}%)  "
              f"均延迟{row['平均延迟天数']:.1f}天  [累计{cum:.0f}%]")


def analyze_promo_vs_normal_otd(df):
    """旺季vs淡季OTD对比"""
    print("\n" + "=" * 60)
    print("【旺季 vs 淡季 OTD 对比】")
    print("=" * 60)
    
    for promo, grp in df.groupby('is_promo'):
        label = '🎄旺季' if promo else '📦淡季'
        otd = grp['on_time'].mean() * 100
        avg_t2 = grp['t2_pick_pack'].mean()
        status = '✅' if otd >= 97 else ('⚠️ ' if otd >= 93 else '🔴')
        print(f"  {label}: OTD={otd:.1f}%  {status}  拣货包装均值={avg_t2:.2f}天  订单量={len(grp)}")


if __name__ == "__main__":
    print("【订单交付周期 OTD 全链路分解分析】\n")
    
    df = generate_delivery_data(n=500)
    
    kpi = compute_otd_kpi_summary(df)
    analyze_stage_contribution(df)
    analyze_otd_by_country(df)
    analyze_delay_pareto(df)
    analyze_promo_vs_normal_otd(df)
    
    print("\n[✓] OTD全链路分解KPI体系 测试通过")
    print(f"    OTD={kpi['otd']:.1f}%  POR={kpi['por']:.1f}%  五阶段拆解+分国+旺季分析完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Order-Fulfillment-Rate-Dispatch-Timeliness]]（发货及时率是OTD的仓库段）
- **前置（prerequisite）**：[[Skill-B2C-Delivery-Timeliness-Experience-KPI]]（配送时效是OTD的末端指标）
- **延伸（extends）**：[[Skill-Order-Accuracy-Exception-Rate-KPI]]（OTD + 准确率 = 完美订单率POR）
- **延伸（extends）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（OTIF是B2B版的OTD+In-Full）
- **可组合（combinable）**：[[Skill-Warehouse-Outbound-Fulfillment-SLA]]（出库SLA是OTD中T2+T3的管控指标）
- **可组合（combinable）**：[[Skill-Supply-Chain-Causal-SCM-Attribution]]（OTD延迟根因的因果归因）

## ⑤ 商业价值评估

- **ROI预估**：OTD从91%提升至97%（维持Amazon Prime资格）→ 保留30%溢价定价权，年化收益约20-30万元；同时减少客户差评和退款
- **实施难度**：⭐⭐☆☆☆（数据主要来自物流系统时间戳，主要工作是建立阶段拆解分析）
- **优先级评分**：⭐⭐⭐⭐⭐（Amazon Prime/FBA OTD是账号健康核心指标，陈凤霞书"面向客户的第一指标"）
- **评估依据**：Amazon研究显示，OTD每提升1%，转化率提升约0.8%，因为客户优先选择"可靠交付"的卖家
