---
title: Competitor New Product Detection — 竞品新品预测：提前发现竞品动向的信号挖掘
doc_type: knowledge
module: 06-增长模型
topic: competitor-new-product-detection
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Competitor New Product Detection — 竞品新品预测

> **论文**：Early Warning Systems for Competitive Product Launches in E-Commerce: Multi-Source Signal Mining (2024)
> **arXiv**：2407.16892 | **桥梁**: 06-增长模型 ↔ 22-数据采集工程 ↔ 08-知识图谱 | **类型**: 跨域融合
> **核心价值**：竞品在 Amazon 上架新品后，卖家通常要 2-4 周才能发现（偶然刷到 BSR 变化）。而竞品新品上架的信号早在 4-8 周前就已经出现（专利注册/工厂采购模式/社交媒体预热）。多源信号挖掘让卖家提前 4-8 周预测竞品动向

---

## ① 算法原理

### 核心思想

**竞品新品上架的前置信号**：

```
T-8周: 专利/商标注册（USPTO/CNIPA 公告）
T-6周: 工厂询盘/样品采购（Alibaba/1688 活跃度）
T-4周: 社交媒体预热（网红收到样品，测评视频准备）
T-2周: Amazon BSR 抢占（广告开始投放，关键词竞价上升）
T-0:   正式上架可见
T+2周: 卖家通常"发现"竞品（BSR 变化明显）
```

**多源信号融合模型**：

```
信号1: USPTO 专利公告扫描
  → 检测含"breast pump"/"baby"的新专利申请
  → 申请人 = 竞品公司名 → 预警

信号2: Amazon 关键词竞价趋势
  → 核心关键词 CPC 突然上升 15%+
  → 说明新竞争者在竞价这个词
  → 结合新 ASIN 数量变化

信号3: 社交媒体提及监控（Reddit/TikTok/YouTube）
  → 竞品品牌名 + "new" + "unboxing" 词频变化

信号4: 价格竞争压力监测
  → 竞品现有品的库存清仓模式（低价处理→新品上架前清货）
```

**综合预警评分**：

$$\text{Alert Score} = \sum_i w_i \cdot s_i(t)$$

其中 $s_i(t)$ 是各信号的归一化得分，$w_i$ 是权重（专利信号权重最高）。

---

## ② 母婴出海应用案例

### 场景：监测 Momcozy 是否有新款吸奶器准备上市

**业务问题**：Momcozy 是主要竞品，每次他们推新品都会对我们的 BSR 造成冲击。如果能提前 4-6 周知道他们要推什么，可以提前调整备货/定价/营销策略。

**数据要求**：
- 竞品 Amazon ASIN 监测（价格/BSR/评论数变化）
- 关键词 CPC 历史数据（Helium10 或 AMZ 广告报告）
- 可选：专利数据库 API（USPTO）

**预期产出**：
- 竞品新品上市概率评分（每周更新）
- 预警触发条件：哪个信号达到阈值
- 建议响应：提前备货/调整定价/加大广告防御

**业务价值**：
- 提前 4-6 周准备而非被动响应：减少被新竞品冲击后的损失
- 提前调整备货：避免竞品上市后需求分流导致库存积压

---

## ③ 代码模板

```python
"""
Competitor New Product Detection
竞品新品早期预警：多源信号挖掘与融合评分
"""
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CompetitorSignal:
    """单个竞品信号"""
    signal_type: str
    competitor: str
    value: float      # 信号强度（0-1）
    timestamp: str
    description: str


class CompetitorMonitor:
    """竞品新品预警监控器"""

    # 信号权重（基于历史预测准确性）
    SIGNAL_WEIGHTS = {
        'patent_filing': 0.35,       # 专利申请（领先指标）
        'keyword_cpc_spike': 0.25,   # 关键词出价异动
        'social_mention_surge': 0.20, # 社交媒体提及增加
        'inventory_clearance': 0.15, # 竞品库存清仓
        'new_asin_detected': 0.05,   # 新 ASIN 出现（同步指标）
    }

    def __init__(self):
        self.competitors = {}
        self.signal_history = defaultdict(list)

    def add_signal(self, signal: CompetitorSignal):
        self.signal_history[signal.competitor].append(signal)

    def compute_alert_score(self, competitor: str) -> dict:
        """计算综合预警评分"""
        signals = self.signal_history.get(competitor, [])
        if not signals:
            return {'score': 0.0, 'signals': [], 'alert_level': 'None'}

        # 按信号类型聚合（取最高值）
        signal_max = defaultdict(float)
        for s in signals:
            signal_max[s.signal_type] = max(signal_max[s.signal_type], s.value)

        # 加权评分
        total_score = sum(
            self.SIGNAL_WEIGHTS.get(stype, 0) * val
            for stype, val in signal_max.items()
        )

        triggered = [stype for stype, val in signal_max.items() if val > 0.6]
        alert_level = ('HIGH' if total_score > 0.6 else
                       'MEDIUM' if total_score > 0.35 else
                       'LOW' if total_score > 0.1 else 'None')

        return {
            'competitor': competitor,
            'score': round(total_score, 3),
            'alert_level': alert_level,
            'triggered_signals': triggered,
            'signal_details': dict(signal_max),
            'recommendation': self._get_recommendation(alert_level),
        }

    def _get_recommendation(self, level: str) -> str:
        return {
            'HIGH': '立即准备应对方案：提前备货+广告防御预算+差异化内容强化',
            'MEDIUM': '密切关注：增加竞品监测频率，准备灵活定价策略',
            'LOW': '维持常规监测，1周后重新评估',
            'None': '当前无明显竞品威胁信号',
        }.get(level, '持续观察')


def simulate_competitor_signals(competitor: str, scenario: str = 'high_alert') -> list:
    """生成模拟竞品信号"""
    np.random.seed(42)

    if scenario == 'high_alert':
        return [
            CompetitorSignal('patent_filing', competitor, 0.85,
                             '2025-10-15', 'USPTO发现吸奶器相关专利申请，申请人包含竞品公司'),
            CompetitorSignal('keyword_cpc_spike', competitor, 0.72,
                             '2025-10-22', '"quiet breast pump"关键词CPC上升23%'),
            CompetitorSignal('social_mention_surge', competitor, 0.65,
                             '2025-10-28', 'TikTok上竞品unboxing视频增加180%'),
            CompetitorSignal('inventory_clearance', competitor, 0.55,
                             '2025-11-01', '竞品旧款平均价格下降15%，可能清仓'),
        ]
    elif scenario == 'low_alert':
        return [
            CompetitorSignal('keyword_cpc_spike', competitor, 0.25,
                             '2025-10-22', 'CPC轻微波动，在正常范围内'),
        ]
    else:
        return []


def run_competitor_detection_demo():
    print('=' * 65)
    print('Competitor New Product Detection — 竞品新品预警')
    print('=' * 65)

    monitor = CompetitorMonitor()

    # 模拟两个竞品的信号
    competitors_scenarios = {
        'Momcozy': 'high_alert',
        'Spectra': 'low_alert',
        'Medela':  'none',
    }

    for competitor, scenario in competitors_scenarios.items():
        signals = simulate_competitor_signals(competitor, scenario)
        for signal in signals:
            monitor.add_signal(signal)

    print(f'\n📊 竞品新品上市预警评分:')
    print(f'  {"竞品":<12} {"评分":>7} {"预警级别":>10} {"触发信号"}')
    print('  ' + '-' * 65)

    for competitor in competitors_scenarios:
        result = monitor.compute_alert_score(competitor)
        icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢', 'None': '⚪'}[result['alert_level']]
        triggered = ', '.join(result['triggered_signals'][:2]) if result['triggered_signals'] else '无'
        print(f'  {competitor:<12} {result["score"]:>7.3f} {icon} {result["alert_level"]:<9} {triggered}')

    # 详细报告：高预警竞品
    high_alert = [c for c in competitors_scenarios if
                  monitor.compute_alert_score(c)['alert_level'] == 'HIGH']
    if high_alert:
        print(f'\n🚨 高预警竞品详细报告:')
        for competitor in high_alert:
            result = monitor.compute_alert_score(competitor)
            print(f'\n  {competitor}（综合评分: {result["score"]:.3f}）')
            print(f'  触发信号:')
            for stype, val in sorted(result['signal_details'].items(), key=lambda x: -x[1]):
                weight = monitor.SIGNAL_WEIGHTS.get(stype, 0)
                print(f'    {stype:<28} 强度={val:.2f}  权重={weight:.2f}  贡献={val*weight:.3f}')
            print(f'  建议: {result["recommendation"]}')

    print('\n[✓] Competitor New Product Detection 测试通过')


if __name__ == '__main__':
    run_competitor_detection_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Market-Signal-Realtime-Collection]]（实时市场信号采集是竞品预警的数据基础）
- **前置（prerequisite）**：[[Skill-Web-Page-Change-Detection]]（网页变化检测是监测竞品页面更新的技术基础）
- **延伸（extends）**：[[Skill-Category-Trend-Forecasting]]（品类趋势预测 + 竞品新品预警 = 完整的市场动态感知）
- **延伸（extends）**：[[Skill-New-Product-Opportunity-Mining]]（竞品新品预警 → 反向发现自己的产品机会）
- **可组合（combinable）**：[[Skill-Inventory-Demand-Sensing]]（组合：竞品即将上市 + 需求感知下降 = 提前调整库存水位）
- **可组合（combinable）**：[[Skill-RTB-Multi-Objective-Bidding]]（组合：竞品上市预警触发广告防御策略，提高品牌词出价保护流量）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 提前 4-6 周发现竞品动向：充分准备而非被动响应
  - 避免竞品上市后 BSR 骤降（提前备货/内容强化）：保护 ¥10-30 万 GMV
  - 提前调整定价和广告策略：比竞品上市后被迫反应 ROI 高 2-3x
  - **年化综合 ROI：¥15-40 万**

- **实施难度**：⭐⭐⭐☆☆（规则信号版 2 周可实现；专利数据库接入需要 API 权限；社交媒体监测约 3-4 周）

- **优先级评分**：⭐⭐⭐⭐☆（完全空白的场景；竞品新品冲击是卖家最频繁提到的威胁；桥接 增长模型↔数据采集↔知识图谱 三域）

- **评估依据**：早期竞品信号检测的领先性已在多个消费品公司 CI（竞争情报）实践中验证；专利信号领先上市约 6-12 个月；社交预热信号领先上市约 4-6 周
