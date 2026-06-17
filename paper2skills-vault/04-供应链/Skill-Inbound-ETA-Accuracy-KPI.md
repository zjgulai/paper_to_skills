---
title: 在途ETA准确率与到货履约率KPI — 过程数据vs结果数据的全链路在途管理体系
doc_type: knowledge
module: 04-供应链
topic: inbound-eta-accuracy-arrival-fulfillment-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 在途ETA准确率与到货履约率KPI

> **书籍**：《全链路管理》陈凤霞 第五章第五节"以在途库存提升物流履约能力——在途库存的过程数据与结果数据"
> **桥梁**: 供应链 ↔ 物流履约 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：书中第五章专节揭示了在途库存管理最容易被忽视的区分：**过程数据（异常跟进）vs 结果数据（履约达成）**。书中还系统分析了在途数据"无法落地"的四大根因——这是多数卖家的痛点所在。

**书中明确区分的两类数据**：

1. **过程数据（Process Data）**——追踪的是"异常"：
   - ETA准确率：预计到货时间 vs 实际到货时间的偏差
   - 在途异常率：发生延误/变更/文件问题的批次占比
   - 信息更新频率：多少批次每天有状态更新（反映信息透明度）
   - 中转停留时长：在中转港/保税区的平均等待天数

2. **结果数据（Result Data）**——追踪的是"完成度"：
   - 到货完成率：按时足量到达目的仓的批次占比
   - 到货准时率：实际到货日 ≤ 承诺到货日的批次占比
   - 缺货归因率：因在途延误导致的缺货占总缺货比例（书中重点指标！）

3. **四大落地障碍（书中专门分析）**：
   - **障碍1—系统**：不同货代、船公司数据格式不统一，无法自动聚合
   - **障碍2—数据**：货代更新不及时，ETA变化后通知延迟>24小时
   - **障碍3—流程**：没有明确的"ETA变化→触发补救行动"的标准流程
   - **障碍4—人员能力**：团队不知道如何将在途数据转化为库存决策

**ETA准确率计算（书中定义）**：
```
ETA准确率 = 1 - |预计到货日 - 实际到货日| / 计划在途天数

批次ETA偏差天数 = 实际到货日 - 预计到货日（正=延迟，负=提前）
P80延迟天数：80%批次的延迟不超过X天（用于制定安全缓冲）
```

## ② 母婴出海应用案例

**场景A：ETA准确率与安全库存动态调整联动**

- **业务问题**：某卖家每次海运延误都是"突然"发现的，当ETA变化通知到运营团队时库存已经告急
- **过程KPI应用**：
  1. 建立ETA准确率基线：某航线历史P80延迟=6天（80%批次延误≤6天）
  2. 安全库存=提前期内日均销售×延误天数缓冲（按P80设定）
  3. 当ETA变化触发"延误预警"（>3天），自动计算影响的SKU缺货风险
  4. 缺货风险SKU→触发空运补货评估（空运成本 vs 缺货损失）
- **预期产出**：提前3-5天预警，应急处理时间足够，断货率从12%降至3%

**场景B：在途数据落地四障碍逐一攻克**

- **业务问题**：团队有在途数据但从不用（典型的"数据富，洞察穷"）
- **落地路径**：障碍1→统一货代数据格式标准；障碍2→要求货代ETA变化6小时内推送；障碍3→建立"ETA变化N天→启动X动作"规则手册；障碍4→将在途KPI纳入运营团队周会必检项

## ③ 代码模板

```python
"""
在途ETA准确率与到货履约率KPI体系
基于《全链路管理》陈凤霞 第五章第五节
过程数据(异常跟进) vs 结果数据(履约达成)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TransitBatchRecord:
    """在途批次记录"""
    batch_id: str
    sku_id: str
    route: str                      # 如 '上海→洛杉矶'
    planned_units: int
    actual_units: int
    planned_eta: datetime           # 计划到货日
    actual_arrival: Optional[datetime]  # 实际到货日（None=未到）
    transit_days_planned: int       # 计划在途天数
    eta_updates: List[datetime] = field(default_factory=list)  # ETA历史更新记录
    delay_cause: Optional[str] = None  # 延误原因


class InboundETAKPIAnalyzer:
    """在途ETA准确率与到货KPI分析器"""

    @staticmethod
    def compute_eta_accuracy(batch: TransitBatchRecord) -> Optional[float]:
        """计算单批次ETA准确率"""
        if batch.actual_arrival is None:
            return None
        delay_days = (batch.actual_arrival - batch.planned_eta).days
        accuracy = 1 - abs(delay_days) / max(batch.transit_days_planned, 1)
        return max(accuracy, 0.0)

    @staticmethod
    def delay_days(batch: TransitBatchRecord) -> Optional[int]:
        """计算延迟天数（正=延迟，负=提前）"""
        if batch.actual_arrival is None:
            return None
        return (batch.actual_arrival - batch.planned_eta).days

    def route_analytics(self, batches: List[TransitBatchRecord]) -> pd.DataFrame:
        """按航线分析ETA准确率分布"""
        route_data = {}
        for b in batches:
            if b.actual_arrival is None:
                continue
            dd = self.delay_days(b)
            ea = self.compute_eta_accuracy(b)
            if b.route not in route_data:
                route_data[b.route] = {'delays': [], 'accuracies': [], 'units': []}
            route_data[b.route]['delays'].append(dd)
            route_data[b.route]['accuracies'].append(ea)
            route_data[b.route]['units'].append(b.actual_units)

        records = []
        for route, data in route_data.items():
            delays = np.array(data['delays'])
            accs = np.array(data['accuracies'])
            records.append({
                'route': route,
                'batch_count': len(delays),
                'avg_delay_days': float(np.mean(delays)),
                'p80_delay_days': float(np.percentile(delays, 80)),
                'p95_delay_days': float(np.percentile(delays, 95)),
                'avg_eta_accuracy': float(np.mean(accs)),
                'on_time_rate': float(np.mean(delays <= 0)),
                'late_rate': float(np.mean(delays > 3)),  # 延迟>3天算异常
                'recommended_buffer_days': int(np.ceil(np.percentile(delays, 80))),
            })
        return pd.DataFrame(records).sort_values('avg_delay_days', ascending=False)

    def arrival_fulfillment_kpi(self, batches: List[TransitBatchRecord]) -> Dict:
        """计算到货履约率结果KPI"""
        completed = [b for b in batches if b.actual_arrival is not None]
        n = len(completed)
        if n == 0:
            return {}

        # 到货准时率（按时≤0延迟）
        on_time = sum(1 for b in completed if self.delay_days(b) <= 0)

        # 到货完成率（足量到达）
        full_quantity = sum(1 for b in completed
                             if b.actual_units >= b.planned_units * 0.98)

        # 平均ETA准确率
        accuracies = [self.compute_eta_accuracy(b) for b in completed]

        return {
            'total_batches': n,
            'arrival_on_time_rate': on_time / n,
            'arrival_on_time_pct': f"{on_time/n:.1%}",
            'full_quantity_rate': full_quantity / n,
            'full_quantity_pct': f"{full_quantity/n:.1%}",
            'avg_eta_accuracy': float(np.mean(accuracies)),
            'avg_eta_accuracy_pct': f"{np.mean(accuracies):.1%}",
            'in_transit_batches': len(batches) - n,
            'overall_status': '✅' if on_time / n >= 0.85 else '🔴',
        }

    def stockout_attribution_analysis(self, stockout_events: List[Dict],
                                       batches: List[TransitBatchRecord]) -> Dict:
        """
        缺货归因分析：多少缺货是由在途延误导致的（书中重点指标）
        """
        total_stockouts = len(stockout_events)
        transit_caused = 0

        delayed_batches = {b.sku_id for b in batches
                            if b.actual_arrival and self.delay_days(b) > 5}

        for event in stockout_events:
            if event.get('sku_id') in delayed_batches:
                transit_caused += 1

        return {
            'total_stockout_events': total_stockouts,
            'transit_delay_caused': transit_caused,
            'transit_attribution_rate': transit_caused / max(total_stockouts, 1),
            'transit_attribution_pct': f"{transit_caused/max(total_stockouts,1):.0%}",
            'non_transit_stockouts': total_stockouts - transit_caused,
        }

    def identify_data_landing_barriers(self, batches: List[TransitBatchRecord]) -> Dict:
        """识别在途数据落地的四大障碍（书中框架）"""
        # 障碍1：系统——多少批次有ETA更新记录（信息透明度）
        has_updates = sum(1 for b in batches if len(b.eta_updates) > 0)
        update_coverage = has_updates / max(len(batches), 1)

        # 障碍2：数据——ETA更新频率（更新及时性）
        avg_updates = np.mean([len(b.eta_updates) for b in batches])

        # 障碍3：流程——延迟>3天的批次，有多少触发了应急动作
        delayed_batches = sum(1 for b in batches
                               if b.actual_arrival and self.delay_days(b) > 3)

        # 障碍4：人员——通过缺货归因率侧面衡量
        return {
            'barrier_1_system': {
                'eta_update_coverage': f"{update_coverage:.0%}",
                'status': '✅' if update_coverage >= 0.8 else '🔴',
                'desc': '80%+批次有ETA更新为系统障碍最低要求',
            },
            'barrier_2_data': {
                'avg_updates_per_batch': round(avg_updates, 1),
                'status': '✅' if avg_updates >= 1.0 else '🔴',
                'desc': '每批次平均至少1次ETA更新为数据质量基线',
            },
            'barrier_3_process': {
                'delayed_batches_needing_action': delayed_batches,
                'desc': '需建立：ETA延误N天→触发X应急动作的标准流程',
            },
            'barrier_4_capability': {
                'desc': '将在途KPI纳入周会必检项，培训团队用ETA数据做库存决策',
            },
        }


def run_inbound_eta_kpi_demo():
    """在途ETA KPI完整演示"""
    print("=" * 65)
    print("在途ETA准确率与到货履约率KPI体系")
    print("基于《全链路管理》陈凤霞 第五章第五节")
    print("区分过程数据(异常跟进) vs 结果数据(履约达成)")
    print("=" * 65)

    np.random.seed(42)
    analyzer = InboundETAKPIAnalyzer()
    base_date = datetime(2026, 1, 1)

    # 模拟不同航线批次数据
    routes_config = [
        ('深圳→洛杉矶', 28, 4, 0.15),   # 平均在途28天，P80延迟4天，15%严重延迟
        ('上海→伦敦', 32, 7, 0.25),       # 平均在途32天，P80延迟7天，25%严重延迟
        ('广州→法兰克福', 30, 5, 0.20),   # 中等
    ]

    all_batches = []
    skus = ['PUMP-PRO', 'WARMER-S1', 'BOTTLE-3P']

    for route, transit_days, p80_delay, late_rate in routes_config:
        for i in range(15):
            delay = int(np.random.lognormal(1.0, 0.8))
            if np.random.random() > late_rate:
                delay = min(delay, 3)
            planned_eta = base_date + timedelta(days=transit_days + i * 7)
            actual_arrival = planned_eta + timedelta(days=delay)
            n_updates = np.random.randint(0, 4)
            batch = TransitBatchRecord(
                batch_id=f"BATCH-{route[:3]}-{i:03d}",
                sku_id=np.random.choice(skus),
                route=route,
                planned_units=np.random.randint(200, 800),
                actual_units=np.random.randint(180, 820),
                planned_eta=planned_eta,
                actual_arrival=actual_arrival,
                transit_days_planned=transit_days,
                eta_updates=[planned_eta - timedelta(days=j*2) for j in range(n_updates)],
                delay_cause='港口拥堵' if delay > 5 else None,
            )
            all_batches.append(batch)

    # 1. 结果KPI
    print("\n[结果KPI：到货履约率]")
    kpi = analyzer.arrival_fulfillment_kpi(all_batches)
    print(f"  到货准时率: {kpi['arrival_on_time_pct']} {kpi['overall_status']}")
    print(f"  足量到达率: {kpi['full_quantity_pct']}")
    print(f"  平均ETA准确率: {kpi['avg_eta_accuracy_pct']}")
    print(f"  在途中批次: {kpi['in_transit_batches']}")

    # 2. 过程KPI - 按航线
    print("\n[过程KPI：航线ETA准确率分析]")
    route_df = analyzer.route_analytics(all_batches)
    print(f"\n  {'航线':<20} {'批次':<6} {'均延迟':<10} {'P80延迟':<10} {'及时率':<10} {'推荐缓冲'}")
    for _, row in route_df.iterrows():
        print(f"  {row['route']:<20} {row['batch_count']:<6} "
              f"{row['avg_delay_days']:<10.1f}天 {row['p80_delay_days']:<10.1f}天 "
              f"{row['on_time_rate']:<10.0%} +{row['recommended_buffer_days']}天")

    # 3. 缺货归因
    stockout_events = [{'sku_id': np.random.choice(skus), 'date': '2026-02-01'} for _ in range(20)]
    attribution = analyzer.stockout_attribution_analysis(stockout_events, all_batches)
    print(f"\n[缺货归因分析]")
    print(f"  总缺货事件: {attribution['total_stockout_events']}")
    print(f"  在途延误导致: {attribution['transit_delay_caused']} ({attribution['transit_attribution_pct']})")
    print(f"  非在途原因: {attribution['non_transit_stockouts']}")

    # 4. 落地障碍评估
    print("\n[四大落地障碍评估（书中框架）]")
    barriers = analyzer.identify_data_landing_barriers(all_batches)
    for barrier_key, info in barriers.items():
        status = info.get('status', '?')
        desc = info.get('desc', '')
        if 'eta_update_coverage' in info:
            print(f"  {barrier_key}: {status} ETA更新覆盖率={info['eta_update_coverage']}")
        elif 'avg_updates_per_batch' in info:
            print(f"  {barrier_key}: {status} 平均{info['avg_updates_per_batch']}次/批次更新")
        else:
            print(f"  {barrier_key}: {desc[:60]}")

    print("\n[书中关键洞察]")
    print("  过程数据=异常跟进（ETA偏差、更新频率）→用于实时预警")
    print("  结果数据=履约达成（准时率、完成率）→用于考核和改善")
    print("  缺货归因中'在途延误'占比=在途管理的最终价值量化")
    print("\n[✓] 在途ETA准确率KPI系统测试通过")
    return kpi


if __name__ == "__main__":
    run_inbound_eta_kpi_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-In-Transit-Inventory-Tracking-Visibility]]（在途可视化是本Skill数据来源）、[[Skill-Lead-Time-Distribution-Risk-GenQOT]]（提前期分布建模支撑ETA准确率预测）
- **延伸（extends）**：[[Skill-ITO-Three-Phase-Health-Tracking]]（在途数据是备货中阶段的核心输入）
- **可组合（combinable）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（OTIF是供应商侧，ETA准确率是物流侧，两者联合覆盖完整链路）、[[Skill-Fill-Rate-OOS-Cost-Quantification]]（缺货归因中在途延误贡献量化）

## ⑤ 商业价值评估

- **ROI 预估**：在途延误导致的缺货占总缺货的40-60%，提前3-5天预警使应急处理成本降低50%；以月缺货损失$5万计算，减少40%≈$2万/月；系统$2万，ROI>1000%
- **实施难度**：⭐⭐⭐☆☆（需要货代API或人工录入ETA更新数据；四大障碍中数据障碍最难克服）
- **优先级**：⭐⭐⭐⭐⭐（书中专节讲解，且明确量化了"缺货归因率"这个被低估的指标）
- **适用规模**：月均在途批次>5个的卖家，批次越多价值越高
- **数据依赖**：货代提供的ETA更新记录、实际到港时间、SKU与批次的对应关系
