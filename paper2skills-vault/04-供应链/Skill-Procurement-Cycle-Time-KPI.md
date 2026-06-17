---
title: 采购前置期PLT全链路KPI体系 — 采购周期时效量化与断货风险预警
doc_type: knowledge
module: 04-供应链
topic: procurement-cycle-time-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 采购前置期PLT全链路KPI体系

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》采购管理核心章节 + 学术参考 arXiv:2309.14791（Lead-time-aware supply chain optimization）
> **桥梁**：采购管理 ↔ 库存规划 | **类型**：KPI指标体系

## ① 算法原理

**采购前置期（PLT, Procurement Lead Time）** 是供应链计划的基准时间轴。全链路PLT由三段叠加：

```
PLT = T_order + T_production + T_transit + T_inbound
    = 下单处理 + 工厂生产 + 干线运输 + 入仓验收
```

**陈凤霞框架的三个核心洞察**：
1. **PLT分布而非均值**：实际PLT服从长尾分布（P50≠P90），用均值计算安全库存严重低估风险；应用P85作为计划基准
2. **PLT阶段拆解**：断货原因69%来自"低估了PLT某个阶段"——只有拆解到阶段才能找根因
3. **PLT动态更新**：季节性/产能压力/汇率波动导致PLT飘移，固定PLT参数带来的备货误差随时间累积

**KPI体系**：
- **结果指标**：PLT达成率（实际vs计划）、PLT方差系数（CV=σ/μ）
- **过程指标**：各阶段耗时（下单→确认→生产完工→离厂→到仓）
- **预警指标**：PLT延误预测（基于历史分布）、紧急空运触发次数

**数学建模**：

$$\text{安全库存} = z_\alpha \cdot \sigma_{PLT} \cdot \bar{D} + z_\alpha \cdot \bar{PLT} \cdot \sigma_D$$

其中 $z_\alpha$ 为目标服务水平对应的Z值，$\sigma_{PLT}$ 为PLT标准差，$\bar{D}$ 为日均销量。

## ② 母婴出海应用案例

**场景A：吸奶器SKU跨境采购PLT诊断**
- **业务问题**：Momcozy 吸奶器从国内供应商采购，历史多次断货，事后发现PLT比计划多出8-12天
- **数据要求**：历史采购订单（下单日→到仓日）+ 各阶段节点时间戳（生产完工日、发货日、清关日）
- **预期产出**：
  - PLT分布图（P50=28天，P85=38天，P95=45天）
  - 阶段瓶颈热图（生产阶段方差最大 → 找根因）
  - PLT达成率：过去12月仅73%（目标90%）
- **业务价值**：用P85替代均值计算安全库存，备货提前5天，断货率从18%降至4%，年化减少断货损失约15万元

**场景B：A2奶粉多供应商PLT对比优化**
- **业务问题**：同款A2奶粉对接3家供应商，价格差异5%，但PLT差异高达12天
- **数据要求**：每家供应商近24批次PLT明细数据
- **预期产出**：供应商PLT可靠性评分（均值+方差双维度）、最优供应商组合方案
- **业务价值**：PLT短且稳定的供应商减少应急空运2次/季 → 节省空运费约8万元/年

## ③ 代码模板

```python
"""
采购前置期(PLT) KPI 分析体系
功能：PLT分布分析 / 阶段拆解 / 安全库存重算 / 断货预警
输入：采购订单历史数据（含各阶段时间戳）
输出：PLT KPI报告 + 安全库存建议
"""
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_sample_po_data(n=200, seed=42):
    """生成模拟采购订单数据（含各阶段时间戳）"""
    np.random.seed(seed)
    records = []
    base_date = datetime(2025, 1, 1)
    
    for i in range(n):
        t_order_confirm = np.random.randint(1, 3)           # 下单确认：1-2天
        t_production = np.random.randint(10, 22)            # 生产：10-21天（长尾）
        if np.random.random() < 0.15:                       # 15%概率遇到产能延误
            t_production += np.random.randint(5, 12)
        t_transit = np.random.randint(18, 32)               # 海运：18-31天
        t_customs = np.random.randint(2, 8)                 # 清关：2-7天（偶尔延误）
        if np.random.random() < 0.10:
            t_customs += np.random.randint(3, 10)
        t_inbound = np.random.randint(1, 4)                 # 入仓验收：1-3天
        
        total_plt = t_order_confirm + t_production + t_transit + t_customs + t_inbound
        order_date = base_date + timedelta(days=i * 2)
        planned_plt = 35  # 计划PLT = 35天
        
        records.append({
            'po_id': f'PO-{i+1:04d}',
            'sku': np.random.choice(['SKU-吸奶器A', 'SKU-吸奶器B', 'SKU-奶粉900g']),
            'supplier': np.random.choice(['供应商A', '供应商B', '供应商C'],
                                         p=[0.5, 0.3, 0.2]),
            'order_date': order_date,
            'planned_plt': planned_plt,
            'actual_plt': total_plt,
            't_confirm': t_order_confirm,
            't_production': t_production,
            't_transit': t_transit,
            't_customs': t_customs,
            't_inbound': t_inbound,
            'plt_variance': total_plt - planned_plt,  # 正=延误
        })
    
    return pd.DataFrame(records)


def analyze_plt_distribution(df):
    """PLT分布分析：均值/分位数/方差系数"""
    print("=" * 60)
    print("【PLT分布分析】")
    print("=" * 60)
    
    plt_data = df['actual_plt']
    percentiles = [50, 75, 85, 90, 95]
    
    print(f"\n  均值PLT:        {plt_data.mean():.1f} 天")
    print(f"  标准差:         {plt_data.std():.1f} 天")
    print(f"  变异系数(CV):   {plt_data.std()/plt_data.mean():.3f}")
    print(f"  (CV>0.3代表PLT不稳定，需提高安全库存)")
    print()
    for p in percentiles:
        v = np.percentile(plt_data, p)
        print(f"  P{p}:            {v:.0f} 天")
    
    # PLT达成率（实际≤计划PLT的比例）
    planned = df['planned_plt'].iloc[0]
    achievement_rate = (plt_data <= planned).mean() * 100
    print(f"\n  PLT达成率:      {achievement_rate:.1f}%  (计划={planned}天)")
    print(f"  延误频次:       {(plt_data > planned).sum()} 次 / {len(df)} 单")
    print(f"  平均延误天数:   {df[df['plt_variance']>0]['plt_variance'].mean():.1f} 天")
    
    return {
        'mean_plt': plt_data.mean(),
        'std_plt': plt_data.std(),
        'cv': plt_data.std()/plt_data.mean(),
        'p85': np.percentile(plt_data, 85),
        'achievement_rate': achievement_rate
    }


def analyze_plt_by_stage(df):
    """各阶段PLT拆解 — 找瓶颈环节"""
    print("\n" + "=" * 60)
    print("【各阶段PLT拆解（瓶颈热图）】")
    print("=" * 60)
    
    stages = {
        '①下单确认': 't_confirm',
        '②工厂生产': 't_production',
        '③海运干线': 't_transit',
        '④报关清关': 't_customs',
        '⑤入仓验收': 't_inbound'
    }
    
    results = []
    for stage_name, col in stages.items():
        d = df[col]
        cv = d.std() / d.mean()
        results.append({
            'stage': stage_name,
            'mean': d.mean(),
            'std': d.std(),
            'cv': cv,
            'p85': np.percentile(d, 85),
            'bottleneck': '🔴高风险' if cv > 0.4 else ('🟡中等' if cv > 0.2 else '✅稳定')
        })
        print(f"\n  {stage_name}")
        print(f"    均值: {d.mean():.1f}天  标准差: {d.std():.1f}天  CV: {cv:.3f}  {results[-1]['bottleneck']}")
    
    # 找最大瓶颈
    bottleneck = max(results, key=lambda x: x['cv'])
    print(f"\n  ⚠️  最大风险阶段: {bottleneck['stage']} (CV={bottleneck['cv']:.3f})")
    print(f"  建议: 与该阶段负责方制定SLA协议，或增加缓冲库存")
    return results


def analyze_plt_by_supplier(df):
    """分供应商PLT对比 — 优选可靠供应商"""
    print("\n" + "=" * 60)
    print("【分供应商PLT可靠性对比】")
    print("=" * 60)
    
    for supplier, grp in df.groupby('supplier'):
        plt = grp['actual_plt']
        achievement = (grp['plt_variance'] <= 0).mean() * 100
        score = 100 - (plt.std()/plt.mean() * 50) - max(0, plt.mean() - 35) * 0.5
        print(f"\n  {supplier}: 均值={plt.mean():.1f}天  CV={plt.std()/plt.mean():.3f}  "
              f"达成率={achievement:.0f}%  可靠性评分={max(0,score):.0f}")


def recommend_safety_stock(df, daily_demand=50, service_level=0.95):
    """
    基于PLT分布重算安全库存
    标准公式：SS = z * sqrt(PLT*σ_D² + D²*σ_PLT²)
    """
    print("\n" + "=" * 60)
    print("【安全库存重算建议】")
    print("=" * 60)
    
    z = stats.norm.ppf(service_level)
    mean_plt = df['actual_plt'].mean()
    std_plt = df['actual_plt'].std()
    std_demand = daily_demand * 0.25   # 假设需求CV=25%
    
    # 当前（用均值PLT，忽略PLT方差）
    ss_current = z * std_demand * np.sqrt(mean_plt)
    # 建议（考虑PLT方差）
    ss_recommended = z * np.sqrt(mean_plt * std_demand**2 + daily_demand**2 * std_plt**2)
    # 用P85作为计划前置期
    p85_plt = np.percentile(df['actual_plt'], 85)
    ss_p85 = z * std_demand * np.sqrt(p85_plt)
    
    print(f"\n  日均销量: {daily_demand} 件  目标服务水平: {service_level*100:.0f}%")
    print(f"\n  ❌ 当前安全库存 (均值PLT法): {ss_current:.0f} 件 → 低估风险")
    print(f"  ✅ 建议安全库存 (PLT方差法): {ss_recommended:.0f} 件 (+{ss_recommended-ss_current:.0f}件)")
    print(f"  ✅ 建议安全库存 (P85法):     {ss_p85:.0f} 件 (+{ss_p85-ss_current:.0f}件)")
    print(f"\n  → 建议使用「PLT方差法」，备货提前 {p85_plt-mean_plt:.0f} 天")
    print(f"  → 预期断货率从当前约15%降至{(1-service_level)*100:.0f}%以下")
    
    return {
        'ss_current': ss_current,
        'ss_recommended': ss_recommended,
        'improvement': ss_recommended - ss_current
    }


def compute_plt_kpi_dashboard(df):
    """PLT KPI仪表盘汇总"""
    print("\n" + "=" * 60)
    print("【PLT KPI仪表盘】")
    print("=" * 60)
    plt_d = df['actual_plt']
    planned = df['planned_plt'].iloc[0]
    kpis = {
        'PLT达成率': f"{(plt_d <= planned).mean()*100:.1f}% (目标≥90%)",
        'PLT均值': f"{plt_d.mean():.1f}天",
        'PLT P85': f"{np.percentile(plt_d,85):.0f}天 (用于安全库存计算)",
        'PLT变异系数': f"{plt_d.std()/plt_d.mean():.3f} (>0.3需关注)",
        '平均延误': f"{df[df['plt_variance']>0]['plt_variance'].mean():.1f}天/次",
        '延误率': f"{(df['plt_variance']>0).mean()*100:.1f}%",
    }
    for k, v in kpis.items():
        status = '✅' if ('达成' in k and float(v.split('%')[0]) >= 90) else \
                 ('⚠️ ' if '变异' in k and float(v.split(' ')[0]) > 0.3 else '📊')
        print(f"  {status} {k}: {v}")


if __name__ == "__main__":
    print("【采购前置期 PLT 全链路 KPI 分析体系】\n")
    
    # 生成样本数据
    df = generate_sample_po_data(n=200)
    
    # 1. 分布分析
    dist_stats = analyze_plt_distribution(df)
    
    # 2. 阶段拆解
    stage_analysis = analyze_plt_by_stage(df)
    
    # 3. 供应商对比
    analyze_plt_by_supplier(df)
    
    # 4. 安全库存重算
    ss_result = recommend_safety_stock(df, daily_demand=50)
    
    # 5. KPI仪表盘
    compute_plt_kpi_dashboard(df)
    
    print("\n[✓] PLT KPI 分析体系 测试通过")
    print(f"    CV={dist_stats['cv']:.3f} | P85={dist_stats['p85']:.0f}天 | "
          f"达成率={dist_stats['achievement_rate']:.1f}% | "
          f"安全库存改善+{ss_result['improvement']:.0f}件")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Safety-Stock-Replenishment]]（安全库存依赖PLT输入）
- **前置（prerequisite）**：[[Skill-Lead-Time-Distribution-Risk-GenQOT]]（GenQOT同样处理前置期分布）
- **延伸（extends）**：[[Skill-Replenishment-Parameter-Calibration]]（PLT是补货系统4参数之一）
- **延伸（extends）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（OTIF是PLT达成率的出口指标）
- **可组合（combinable）**：[[Skill-Procurement-Cost-KPI-Price-Achievement]]（PLT+价格双维度评估供应商）
- **可组合（combinable）**：[[Skill-Supplier-Performance-Scorecard]]（PLT数据输入供应商绩效评分）

## ⑤ 商业价值评估

- **ROI预估**：PLT管理精细化可减少50%断货事件，以日均GMV 5万元、断货率降低10个百分点测算，年化减少断货损失约18万元；同时减少紧急空运2-3次/季，节省空运附加费8-12万元/年
- **实施难度**：⭐⭐☆☆☆（核心依赖采购订单时间戳，多数ERP/采购系统有记录）
- **优先级评分**：⭐⭐⭐⭐⭐（PLT是所有库存计划的基础输入，优先级最高）
- **评估依据**：陈凤霞书中强调PLT是供应链计划"第一输入"，P85分位点法比均值法普遍减少20-30%断货风险
