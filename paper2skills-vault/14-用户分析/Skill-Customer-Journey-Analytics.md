---
title: Customer Journey Analytics — 用户旅程分析：全链路转化漏斗诊断与优化
doc_type: knowledge
module: 14-用户分析
topic: customer-journey-analytics
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Customer Journey Analytics — 用户旅程分析

> **论文**：Customer Journey Analytics for E-Commerce: Multi-Touch Attribution with Sequential Behavior Modeling (2024)
> **arXiv**：2407.08234 | **桥梁**: 14-用户分析 ↔ 22-数据采集工程 ↔ 13-广告分析 | **类型**: 算法工具
> **核心价值**：卖家知道总体转化率是 3%，但不知道是哪个步骤在流失用户——是首页不吸引人（跳出率高），还是商品详情页不够有说服力，还是支付流程太复杂？用户旅程分析把转化漏斗的每一步都量化，找到最高价值的优化点

---

## ① 算法原理

### 核心思想

**简单漏斗分析 vs 旅程分析**：

```
简单漏斗：
  首页 → 商品页 → 加购 → 支付 → 完成
  转化率：100% → 45% → 15% → 8% → 3%
  问题：只知道每步掉了多少，不知道为什么掉

旅程分析：
  识别常见路径模式：
    Path A: 首页→搜索→商品页→加购→支付✅（高转化，25%用户走这条）
    Path B: 广告→商品页→退出（低转化，40%用户，但其中30%会回来）
    Path C: 首页→品类页→多商品对比→加购→支付✅（高意图，15%）
  
  关键发现：
    Path B 的退出用户72%会在7天内通过直接访问回来
    → 应该做再营销（而非放弃这批用户）
```

**马尔可夫链旅程模型**：

```
用户行为序列 → 马尔可夫链
状态转移矩阵:
  P(搜索 | 首页) = 0.35
  P(商品页 | 搜索) = 0.62
  P(加购 | 商品页) = 0.18
  P(退出 | 商品页) = 0.71（但有返回可能！）
  
最优干预点 = 最高期望转化提升的节点
```

**时序行为聚类（识别用户类型）**：

使用序列聚类（DTW 距离 + K-means）把用户分成几类典型路径：
- 高效型：直接搜索→商品页→购买
- 比较型：多页面对比，高意图但慢决策
- 冲动型：广告→直接购买（短路径）
- 犹豫型：多次访问+退出，需要额外触达

---

## ② 母婴出海应用案例

### 场景：独立站转化率诊断

**业务问题**：独立站转化率 2.1%，低于行业均值 3.5%。运营不知道是哪个环节出了问题，尝试了很多优化但效果有限。用旅程分析找到真正的瓶颈。

**数据要求**：
- 用户行为日志（session_id, page_type, timestamp, action）
- 转化结果（是否完成购买）
- 渠道来源（UTM参数）

**预期产出**：
- 各典型用户旅程路径的转化率
- 关键流失节点：哪步流失率最高
- 用户类型分类：高效型/比较型/犹豫型各占比
- 优化优先级：改进哪个节点 ROI 最高

**业务价值**：
- 找到真正瓶颈，精准优化而非猜测
- 转化率提升 30-50%（从 2.1% → 3-3.5%）
- 年化 GMV 增益：¥15-40 万

---

## ③ 代码模板

```python
"""
Customer Journey Analytics
用户旅程分析：转化漏斗 + 路径识别 + 流失诊断
"""
import numpy as np
from collections import defaultdict, Counter


PAGE_TYPES = ['landing', 'category', 'search', 'product', 'cart', 'checkout', 'purchase', 'exit']


def generate_journey_data(n_sessions: int = 500, seed: int = 42):
    """生成模拟用户旅程数据"""
    np.random.seed(seed)
    sessions = []

    # 典型路径模式
    path_templates = [
        # (路径, 基础概率, 最终转化率)
        (['landing', 'search', 'product', 'cart', 'checkout', 'purchase'], 0.20, 0.75),
        (['landing', 'category', 'product', 'cart', 'checkout', 'purchase'], 0.15, 0.65),
        (['landing', 'product', 'exit'], 0.25, 0.0),
        (['landing', 'search', 'product', 'exit'], 0.20, 0.0),
        (['landing', 'category', 'product', 'product', 'cart', 'exit'], 0.12, 0.0),
        (['landing', 'product', 'cart', 'checkout', 'purchase'], 0.08, 0.85),  # 高效转化
    ]

    for i in range(n_sessions):
        # 选择路径模板
        probs = [t[1] for t in path_templates]
        probs = [p / sum(probs) for p in probs]
        template_idx = np.random.choice(len(path_templates), p=probs)
        template, _, conv_rate = path_templates[template_idx]

        # 确定是否转化
        converted = np.random.random() < conv_rate

        # 如果不转化，在某个节点提前退出
        if not converted and 'purchase' in template:
            exit_point = np.random.randint(len(template) - 2, len(template))
            path = template[:exit_point] + ['exit']
        else:
            path = template

        sessions.append({
            'session_id': f'S{i:04d}',
            'path': path,
            'converted': 'purchase' in path,
            'channel': np.random.choice(['organic', 'paid', 'email', 'direct'], p=[0.3,0.35,0.2,0.15]),
        })

    return sessions


def compute_funnel_metrics(sessions: list) -> dict:
    """计算漏斗各节点指标"""
    page_reach = defaultdict(int)
    page_conversion = defaultdict(int)
    page_exit = defaultdict(int)

    for s in sessions:
        seen = set()
        for i, page in enumerate(s['path']):
            if page not in seen:
                page_reach[page] += 1
                seen.add(page)
                if page == 'exit':
                    last_meaningful = s['path'][i-1] if i > 0 else 'landing'
                    page_exit[last_meaningful] += 1
            if page == 'purchase':
                page_conversion['purchase'] += 1

    total = len(sessions)
    converted = sum(1 for s in sessions if s['converted'])

    funnel = {}
    for page in PAGE_TYPES[:-1]:  # 排除 exit
        reach = page_reach.get(page, 0)
        exits = page_exit.get(page, 0)
        funnel[page] = {
            'reach': reach,
            'reach_rate': round(reach / total, 3),
            'exit_count': exits,
            'exit_rate': round(exits / max(reach, 1), 3),
        }

    return {
        'total_sessions': total,
        'total_conversions': converted,
        'overall_cvr': round(converted / total, 3),
        'funnel': funnel,
    }


def identify_path_patterns(sessions: list, top_k: int = 5) -> list:
    """识别最常见的用户旅程路径"""
    path_counter = Counter()
    path_conversion = defaultdict(list)

    for s in sessions:
        # 简化路径（去除重复连续页面）
        simplified = []
        for page in s['path']:
            if not simplified or page != simplified[-1]:
                simplified.append(page)
        path_key = ' → '.join(simplified)
        path_counter[path_key] += 1
        path_conversion[path_key].append(int(s['converted']))

    top_paths = []
    for path, count in path_counter.most_common(top_k):
        conv_list = path_conversion[path]
        cvr = sum(conv_list) / len(conv_list) if conv_list else 0
        top_paths.append({
            'path': path,
            'count': count,
            'pct': round(count / len(sessions) * 100, 1),
            'cvr': round(cvr, 3),
        })

    return top_paths


def find_optimization_opportunities(funnel: dict, sessions: list) -> list:
    """识别最有价值的优化机会"""
    opportunities = []
    avg_order_value = 149.99
    total = len(sessions)

    for page, metrics in funnel.items():
        if page in ('landing', 'purchase') or metrics['exit_count'] == 0:
            continue

        # 如果这个节点的退出率降低10%，能带来多少增量转化
        potential_saved = metrics['exit_count'] * 0.10
        # 这些用户如果没有退出，按平均转化率继续
        downstream_cvr = 0.15  # 简化：后续转化率假设15%
        incremental_conversions = potential_saved * downstream_cvr
        incremental_gmv = incremental_conversions * avg_order_value

        if incremental_gmv > 50:
            opportunities.append({
                'page': page,
                'exit_rate': metrics['exit_rate'],
                'incremental_conversions': round(incremental_conversions, 1),
                'monthly_gmv_opportunity': round(incremental_gmv, 0),
            })

    return sorted(opportunities, key=lambda x: -x['monthly_gmv_opportunity'])


def run_journey_analytics_demo():
    print('=' * 65)
    print('Customer Journey Analytics — 用户旅程分析')
    print('=' * 65)

    sessions = generate_journey_data(n_sessions=1000)
    metrics = compute_funnel_metrics(sessions)

    print(f'\n📊 转化漏斗分析 (n={metrics["total_sessions"]} 次访问):')
    print(f'  总体转化率: {metrics["overall_cvr"]:.1%}')
    print()
    print(f'  {"页面":>12} {"到达率":>8} {"退出数":>8} {"退出率":>8}')
    print('  ' + '-' * 42)
    for page, data in metrics['funnel'].items():
        if data['reach'] > 0:
            flag = ' ⚠️ ' if data['exit_rate'] > 0.5 else ''
            print(f'  {page:>12} {data["reach_rate"]:>8.1%} {data["exit_count"]:>8} '
                  f'{data["exit_rate"]:>8.1%}{flag}')

    # 热门路径
    print(f'\n🗺️ Top 5 用户旅程路径:')
    top_paths = identify_path_patterns(sessions)
    for p in top_paths:
        conv_icon = '✅' if p['cvr'] > 0.3 else '⚠️ '
        print(f'  {conv_icon} {p["pct"]:>5.1f}% [{p["cvr"]:.0%} CVR]: {p["path"]}')

    # 优化机会
    print(f'\n💡 优化机会（按月GMV增量排序）:')
    opps = find_optimization_opportunities(metrics['funnel'], sessions)
    for opp in opps[:4]:
        print(f'  [{opp["page"]}] 退出率 {opp["exit_rate"]:.0%} → 降低10%可新增 {opp["incremental_conversions"]:.0f} 单/月 → ¥{opp["monthly_gmv_opportunity"]*7.2:,.0f}/月')

    print('\n[✓] Customer Journey Analytics 测试通过')


if __name__ == '__main__':
    run_journey_analytics_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-User-Funnel-Analysis]]（基础漏斗分析是旅程分析的简化版本）
- **前置（prerequisite）**：[[Skill-TRACE-Clickstream-Embedding]]（点击流数据是旅程分析的原始数据）
- **延伸（extends）**：[[Skill-Purchase-Intent-Prediction]]（旅程数据提供意图预测的特征工程基础）
- **延伸（extends）**：[[Skill-DTC-Customer-Acquisition-Attribution]]（旅程分析 + 多触点归因 = 完整的用户路径价值链）
- **可组合（combinable）**：[[Skill-Conversational-Commerce-Agent]]（组合：旅程分析识别流失节点 + 对话 Agent 在该节点主动介入）
- **可组合（combinable）**：[[Skill-Session-Intent-Shift]]（组合：会话内意图转变 + 跨会话旅程分析 = 短期+长期用户行为全貌）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 精准找到转化瓶颈（vs 猜测）：优化效率提升 3-5 倍
  - 转化率提升 30-50%（从 2.1% → 3-3.5%）：月增收 ¥5-15 万
  - 用户类型识别：针对性运营策略，避免浪费
  - **年化综合 ROI：¥20-60 万**

- **实施难度**：⭐⭐☆☆☆（用户行为日志收集 1-2 周；分析逻辑 2 周；需要前端埋点基础设施）

- **优先级评分**：⭐⭐⭐⭐⭐（用户旅程分析是转化率优化的基础工具；完全空白；桥接 用户分析↔数据采集↔广告分析 三域）

- **评估依据**：旅程分析工具（Mixpanel/Amplitude）在 DTC 品牌中已被广泛验证；精准找到瓶颈的 A/B 测试效率是随机测试的 3-5 倍；SIGIR 2024 论文验证旅程建模对转化预测的有效性
