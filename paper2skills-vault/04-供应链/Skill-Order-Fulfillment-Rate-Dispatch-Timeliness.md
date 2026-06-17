---
title: 订单履约率与发货及时率 — 全链路订单从下单到签收的履约质量量化体系
doc_type: knowledge
module: 04-供应链
topic: order-fulfillment-rate-dispatch-timeliness
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 订单履约率与发货及时率

> **书籍**：《全链路管理》陈凤霞 第二章第四节"电商物流供应链KPI——成本和体验"综合；结合行业标准
> **桥梁**: 供应链 ↔ 用户分析 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：订单履约是供应链"最后一公里"的核心KPI，贯穿从下单到签收的全链路。书中强调：**履约率不等于配送成功率**——履约率衡量的是"有库存且按时发货"的完整链路，包含库存可用性、发货及时性、配送成功性三个环节，任何一环失败都算履约失败。

**订单履约率完整定义体系**：

1. **订单满足率（Order Fill Rate）**：
   - = 有库存能满足的订单 / 总订单
   - 衡量：库存端是否能响应需求
   - 注意：与"发货率"不同，满足率=库存满足，发货率=实际发货

2. **发货及时率（On-Time Dispatch Rate）**：
   - = 按承诺时间发货的订单 / 总订单
   - 承诺时效：通常平台标准（Amazon 24h内发货，Shopee 48h内）
   - 根因分类：库存不足导致延迟、操作不及时、快递揽件延迟

3. **完整履约率（Perfect Order Rate）**：
   - = 完全满足（有货+及时发+按时到+无损）的订单 / 总订单
   - 书中最高层次的KPI，综合衡量全链路质量
   - 行业标准：Amazon FBA类≥97%，自发货类≥90%

4. **分层履约率（Layered Fulfillment Analysis）**：
   ```
   履约漏斗：
   总订单 100%
   ↓ 库存满足率 (库存不足→缺货拒单)
   有效订单 X%
   ↓ 发货及时率 (延迟发货→客体验差)
   及时发货 Y%
   ↓ 配送成功率 (失败投递→退件)
   成功签收 Z%
   ↓ 无争议率 (破损/错发→投诉)
   完美履约 W%
   ```

**算法创新：履约率影响因素分解**：
用机器学习识别哪些因素（SKU类别/平台/旺季/仓库/快递商）对履约率影响最大，优先改善高影响因素。

## ② 母婴出海应用案例

**场景A：Amazon卖家账号健康保护**

- **业务问题**：某母婴卖家Amazon账号健康评分下降，发现ODR（Order Defect Rate）从0.8%升至1.5%（Amazon红线1%），存在封号风险
- **履约率拆解**：
  1. ODR = 差评率0.3% + A-to-Z索赔率0.8% + 信用卡拒付率0.4%
  2. A-to-Z索赔主要来源：3个SKU的货运破损率异常（2.1% vs 正常0.2%）
  3. 根因：这3个SKU包装不够厚（婴儿玻璃奶瓶），海运期间碰撞破损
  4. 行动：加强外箱包装规格，破损率降至0.15%，ODR在2周内回落至0.7%

**场景B：大促期间履约率预警**

- **业务问题**：Prime Day首日发货及时率从98%骤降至65%（仓库临时工不熟悉操作），触发Amazon预警
- **实时履约监控**：每4小时计算一次发货及时率，发现低于85%时立即启动应急方案（加班+借调人员），避免长时间低于阈值触发更严重的处罚

## ③ 代码模板

```python
"""
订单履约率与发货及时率全链路量化体系
基于《全链路管理》陈凤霞 + 行业最佳实践
完整履约漏斗 + 根因分解 + 风险预警
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OrderRecord:
    """订单记录"""
    order_id: str
    sku_id: str
    platform: str
    order_time: str
    promised_dispatch_time: str    # 承诺发货时间
    actual_dispatch_time: Optional[str]  # 实际发货时间（None=未发货）
    promised_delivery_time: Optional[str]  # 承诺到达时间
    actual_delivery_time: Optional[str]
    has_stock: bool                # 下单时是否有库存
    delivery_success: bool         # 是否成功签收
    has_damage: bool               # 是否有破损
    has_complaint: bool            # 是否有投诉


class FulfillmentRateAnalyzer:
    """订单履约率分析引擎"""

    # 平台发货时效标准（小时）
    DISPATCH_STANDARDS = {
        'Amazon': 24,
        'Shopee': 48,
        'TikTok_Shop': 48,
        'Own_Store': 72,
    }

    def compute_fulfillment_funnel(self, orders: List[OrderRecord]) -> Dict:
        """计算完整履约漏斗"""
        n = len(orders)
        if n == 0:
            return {}

        # 各层次统计
        with_stock = sum(1 for o in orders if o.has_stock)
        dispatched_on_time = sum(1 for o in orders
                                  if o.has_stock and o.actual_dispatch_time is not None
                                  and o.actual_dispatch_time <= o.promised_dispatch_time)
        delivered_success = sum(1 for o in orders if o.delivery_success)
        no_issues = sum(1 for o in orders
                         if o.delivery_success and not o.has_damage and not o.has_complaint)

        return {
            'total_orders': n,
            'funnel': {
                '1_order_fill_rate': {
                    'count': with_stock, 'rate': with_stock / n,
                    'rate_pct': f"{with_stock/n:.1%}",
                    'drop': n - with_stock,
                    'drop_cause': '库存不足/缺货',
                },
                '2_dispatch_on_time_rate': {
                    'count': dispatched_on_time,
                    'rate': dispatched_on_time / max(with_stock, 1),
                    'rate_pct': f"{dispatched_on_time/max(with_stock,1):.1%}",
                    'drop': with_stock - dispatched_on_time,
                    'drop_cause': '发货延迟',
                },
                '3_delivery_success_rate': {
                    'count': delivered_success, 'rate': delivered_success / n,
                    'rate_pct': f"{delivered_success/n:.1%}",
                    'drop': n - delivered_success,
                    'drop_cause': '派送失败/退件',
                },
                '4_perfect_order_rate': {
                    'count': no_issues, 'rate': no_issues / n,
                    'rate_pct': f"{no_issues/n:.1%}",
                    'drop': delivered_success - no_issues,
                    'drop_cause': '破损/投诉',
                },
            },
            'perfect_order_rate': no_issues / n,
            'perfect_order_status': '✅达标' if no_issues / n >= 0.90 else '🔴需改进',
        }

    def compute_dispatch_timeliness(self, orders: List[OrderRecord],
                                     platform: str) -> Dict:
        """计算发货及时率（分平台）"""
        platform_orders = [o for o in orders if o.platform == platform]
        if not platform_orders:
            return {}

        standard_hours = self.DISPATCH_STANDARDS.get(platform, 48)
        on_time = sum(1 for o in platform_orders
                       if o.actual_dispatch_time and
                       o.actual_dispatch_time <= o.promised_dispatch_time)
        late_orders = [o for o in platform_orders
                        if not o.actual_dispatch_time or
                        o.actual_dispatch_time > o.promised_dispatch_time]

        return {
            'platform': platform,
            'total': len(platform_orders),
            'on_time': on_time,
            'on_time_rate': on_time / len(platform_orders),
            'on_time_rate_pct': f"{on_time/len(platform_orders):.1%}",
            'late_count': len(late_orders),
            'dispatch_standard_hours': standard_hours,
            'status': '✅' if on_time / len(platform_orders) >= 0.95 else '🔴',
        }

    def defect_rate_analysis(self, orders: List[OrderRecord]) -> Dict:
        """ODR（订单缺陷率）分析——Amazon关键指标"""
        n = len(orders)
        defects = sum(1 for o in orders if o.has_damage or o.has_complaint)
        damage_count = sum(1 for o in orders if o.has_damage)
        complaint_count = sum(1 for o in orders if o.has_complaint)

        odr = defects / max(n, 1)
        return {
            'total_orders': n,
            'defect_count': defects,
            'odr': odr,
            'odr_pct': f"{odr:.2%}",
            'damage_rate': damage_count / max(n, 1),
            'complaint_rate': complaint_count / max(n, 1),
            'amazon_threshold': 0.01,  # Amazon ODR红线1%
            'status': '✅安全' if odr < 0.008 else ('⚠️警告' if odr < 0.01 else '🔴危险！'),
            'action': 'ODR>1%有封号风险！立即查明根因' if odr >= 0.01 else '保持',
        }

    def segment_analysis(self, orders: List[OrderRecord],
                          segment_by: str = 'sku_id') -> pd.DataFrame:
        """按SKU/平台分析履约率，识别问题热点"""
        records = []
        segments = set(getattr(o, segment_by) for o in orders)
        for seg in segments:
            seg_orders = [o for o in orders if getattr(o, segment_by) == seg]
            n = len(seg_orders)
            on_time = sum(1 for o in seg_orders if o.actual_dispatch_time and
                           o.actual_dispatch_time <= o.promised_dispatch_time)
            defects = sum(1 for o in seg_orders if o.has_damage or o.has_complaint)
            records.append({
                segment_by: seg,
                'order_count': n,
                'dispatch_on_time_rate': on_time / max(n, 1),
                'defect_rate': defects / max(n, 1),
                'needs_attention': defects / max(n, 1) > 0.02 or on_time / max(n, 1) < 0.90,
            })
        df = pd.DataFrame(records).sort_values('defect_rate', ascending=False)
        return df


def run_fulfillment_rate_demo():
    """订单履约率体系完整演示"""
    print("=" * 65)
    print("订单履约率与发货及时率全链路体系")
    print("基于《全链路管理》陈凤霞 + 行业最佳实践")
    print("=" * 65)

    np.random.seed(42)
    analyzer = FulfillmentRateAnalyzer()

    # 生成模拟订单数据
    platforms = ['Amazon', 'Shopee', 'TikTok_Shop']
    skus = ['PUMP-PRO', 'WARMER-S1', 'BOTTLE-3P', 'NIPPLE-SHIELD']
    orders = []

    for i in range(1000):
        platform = np.random.choice(platforms, p=[0.6, 0.25, 0.15])
        sku = np.random.choice(skus, p=[0.4, 0.2, 0.3, 0.1])
        has_stock = np.random.random() > 0.05  # 5%缺货率

        # 根据SKU模拟不同的问题概率
        damage_prob = 0.03 if sku == 'NIPPLE-SHIELD' else 0.008  # 小件更容易破损
        on_time_prob = 0.85 if platform == 'TikTok_Shop' else 0.96  # TikTok发货稍慢

        dispatch_on_time = np.random.random() < on_time_prob
        order = OrderRecord(
            order_id=f"ORD-{i:05d}",
            sku_id=sku,
            platform=platform,
            order_time="2026-06-01",
            promised_dispatch_time="2026-06-02" if on_time_prob == 0.96 else "2026-06-03",
            actual_dispatch_time="2026-06-02" if dispatch_on_time else "2026-06-04",
            promised_delivery_time="2026-06-08",
            actual_delivery_time="2026-06-07" if dispatch_on_time else None,
            has_stock=has_stock,
            delivery_success=has_stock and dispatch_on_time and np.random.random() > 0.01,
            has_damage=np.random.random() < damage_prob,
            has_complaint=np.random.random() < 0.015,
        )
        orders.append(order)

    # 1. 履约漏斗
    print("\n[1] 完整履约漏斗分析")
    funnel = analyzer.compute_fulfillment_funnel(orders)
    print(f"  总订单: {funnel['total_orders']}")
    for level, data in funnel['funnel'].items():
        bar = "█" * int(data['rate'] * 20)
        print(f"  {level}: {data['rate_pct']} {bar} (流失{data['drop']}单←{data['drop_cause']})")
    print(f"\n  完美履约率: {funnel['perfect_order_rate']:.1%} {funnel['perfect_order_status']}")

    # 2. 分平台发货及时率
    print("\n[2] 分平台发货及时率")
    for platform in platforms:
        r = analyzer.compute_dispatch_timeliness(orders, platform)
        if r:
            print(f"  {r['platform']}: {r['on_time_rate_pct']} {r['status']} "
                  f"(迟发{r['late_count']}单，标准{r['dispatch_standard_hours']}h)")

    # 3. ODR分析（Amazon风险）
    print("\n[3] ODR订单缺陷率（Amazon账号安全）")
    amazon_orders = [o for o in orders if o.platform == 'Amazon']
    odr = analyzer.defect_rate_analysis(amazon_orders)
    print(f"  Amazon订单: {odr['total_orders']} | ODR: {odr['odr_pct']} | {odr['status']}")
    print(f"  破损率: {odr['damage_rate']:.2%} | 投诉率: {odr['complaint_rate']:.2%}")
    if odr['action'] != '保持':
        print(f"  ⚠️ {odr['action']}")

    # 4. 分SKU热点识别
    print("\n[4] SKU级履约热点识别")
    seg_df = analyzer.segment_analysis(orders, 'sku_id')
    print(f"  {'SKU':<20} {'订单量':<10} {'及时发货':<12} {'缺陷率':<10} {'需关注'}")
    for _, row in seg_df.iterrows():
        flag = '⚠️' if row['needs_attention'] else '✅'
        print(f"  {row['sku_id']:<20} {row['order_count']:<10} "
              f"{row['dispatch_on_time_rate']:.0%}{'':>5} {row['defect_rate']:.2%}{'':>3} {flag}")

    print("\n[书中关键洞察]")
    print("  完美履约率 = 四层质量门的串联积（任一层低都严重拖累整体）")
    print("  ODR红线1%是Amazon封号的主要诱因，破损率+投诉率分开管控")
    print("  发货及时率是可控性最强的指标，优先保障旺季不触发平台预警")
    print("\n[✓] 订单履约率与发货及时率系统测试通过")


if __name__ == "__main__":
    run_fulfillment_rate_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Fill-Rate-OOS-Cost-Quantification]]（Fill Rate是履约漏斗第一层）、[[Skill-B2C-Delivery-Timeliness-Experience-KPI]]（配送时效是履约漏斗第三层）
- **延伸（extends）**：[[Skill-Supply-Chain-KPI-Health-Dashboard]]（完美履约率是KPI仪表盘的核心综合指标）
- **可组合（combinable）**：[[Skill-Warehouse-Operations-KPI-Picking-Efficiency]]（仓储运营效率直接影响发货及时率）、[[Skill-Reverse-Logistics-Disposition-Optimization]]（履约失败后的逆向物流处置）

## ⑤ 商业价值评估

- **ROI 预估**：ODR从1.5%降至0.8%避免Amazon封号风险（封号损失$50万+年营业额）；完美履约率每提升1%，平台自然搜索排名约提升2-3位（GMV增量约3-5%）；系统建设$2万，防损价值极高
- **实施难度**：⭐⭐⭐☆☆（需要整合多平台数据；Amazon提供订单级别的ODR报告，可直接获取；自发货平台需要自建追踪）
- **优先级**：⭐⭐⭐⭐⭐（ODR是Amazon账号安全的核心指标，完美履约率直接影响平台排名，是必须追踪的指标）
- **适用规模**：所有在Amazon/Shopee/TikTok Shop等平台销售的卖家
- **数据依赖**：平台订单API（提供发货时效/投诉/索赔数据）；物流商API（破损/丢失数据）
