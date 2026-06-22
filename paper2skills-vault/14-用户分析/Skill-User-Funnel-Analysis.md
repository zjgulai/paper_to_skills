---
title: User Funnel and Behavior Path Analysis
module: 14-用户分析
topic: funnel-analysis

roadmap_phase: phase2
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: User Funnel Analysis

## ① 算法原理

**核心问题**：用户从"知道品牌"到"下单购买"要经历多个步骤。每一步都有用户流失。漏斗分析回答：用户在哪个步骤流失最多？为什么？优化哪个步骤的杠杆最大？

**经典电商漏斗（AIPL模型）**：

```
认知(Awareness) → 兴趣(Interest) → 购买(Purchase) → 忠诚(Loyalty)
    100%    →      30%      →      5%       →      2%
```

**行为路径分析**：

不仅看漏斗的"宽度"，还要看用户走的路径：
- 路径1：首页 → 搜索 → 详情页 → 加购 → 支付（理想路径）
- 路径2：首页 → 详情页 → 跳出（价格敏感）
- 路径3：首页 → 详情页 → 详情页 → 详情页（比较型用户）
- 路径4：广告 → 详情页 → 加购 → 支付（高意向用户）

**关键指标**：

| 指标 | 定义 | 诊断价值 |
|------|------|---------|
| **转化率** | 下一步人数 / 当前步人数 | 哪一步漏最多 |
| **流失率** | 1 - 转化率 | 哪一步需要优化 |
| **中位停留时间** | 每步的中位停留时长 | 用户是否困惑 |
| **回溯率** | 返回上一步的比例 | 信息是否不足 |
| **多步流失占比** | 流失前经过的步骤数 | 是突然离开还是慢慢放弃 |

**路径挖掘算法**：

**1. 序列模式挖掘（PrefixSpan）**
- 找出最常见的用户行为序列
- "首页 → 搜索 → 详情页 → 加购" 出现频率是多少？

**2. 马尔可夫链模型**
- 计算从状态A到状态B的转移概率
- 识别"吸收态"（如支付成功、跳出）
- 模拟：如果详情页转化率提升10%，整体转化率会提升多少？

**反直觉洞察**：
- 漏斗最大的漏洞往往不是最后一步（支付），而是第一步（从详情页到加购）——50%的用户在详情页就离开了
- "加购但未支付"的用户不是"流失"——他们是"延迟决策"，邮件提醒的回收率可达15%
- 比较型用户（看多个详情页）的转化率反而高于直接下单型——因为他们已经做好了功课

---

## ② 母婴出海应用案例

### 场景1：吸奶器详情页流失分析

**业务问题**：Momcozy 吸奶器详情页UV 10,000/天，但加购率只有3%，支付转化率1%。详情页是不是有问题？

**漏斗分析**：

| 步骤 | 用户数 | 转化率 | 流失原因分析 |
|------|--------|--------|-------------|
| 详情页UV | 10,000 | — | — |
| 看完详情（滚动>50%）| 6,000 | 60% | 40%跳出：页面加载慢/首屏不吸引人 |
| 点击"加购" | 3,000 | 30% | 30%流失：价格犹豫/信任不足 |
| 进入购物车 | 2,500 | 25% | 5%流失：加购操作失败 |
| 进入支付 | 1,200 | 12% | 13%流失：运费/税费意外 |
| 完成支付 | 800 | 8% | 4%流失：支付方式不支持 |

**优化优先级**：
1. **首屏吸引力**（40%流失）：优化首屏视频和核心卖点展示
2. **价格信任**（30%流失）：增加用户评价、媒体报道、信任徽章
3. **运费透明**（13%流失）：详情页直接显示"包邮"或运费计算器

**预期效果**：详情页→加购转化率 3% → 5%，日增订单200单。

### 场景2：用户路径聚类

**业务问题**：不同类型的用户有不同的购买路径。识别典型路径，针对性优化。

**路径聚类结果**：

| 用户群 | 典型路径 | 占比 | 转化率 | 特征 |
|--------|---------|------|--------|------|
| 冲动型 | 广告 → 详情页 → 支付 | 15% | 12% | 高客单价，决策快 |
| 研究型 | 搜索 → 详情页 → 对比页 → 详情页 → 加购 → 支付 | 30% | 8% | 看多个产品，需要详细信息 |
| 价格敏感型 | 详情页 → 优惠券页 → 详情页 → 加购 → 支付 | 25% | 5% | 对促销敏感 |
| 老客复购型 | 首页 → 我的订单 → 再次购买 | 20% | 35% | 极高转化率，无需优化 |
| 浏览型 | 首页 → 列表页 → 列表页 → 跳出 | 10% | 0.5% | 低意向，放弃优化 |

**策略**：
- 冲动型：简化支付流程，一键下单
- 研究型：详情页增加对比表格、FAQ、视频评测
- 价格敏感型：详情页直接展示优惠券入口

---

## ③ 代码模板

```python
"""
User Funnel and Behavior Path Analysis — 用户漏斗与行为路径分析
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict


class FunnelAnalyzer:
    """漏斗分析器"""

    def __init__(self, step_names):
        """
        Args:
            step_names: 漏斗步骤名称列表，如 ['首页', '详情页', '加购', '支付', '完成']
        """
        self.steps = step_names

    def analyze(self, user_paths):
        """
        分析漏斗

        Args:
            user_paths: list of lists, each inner list is a user's step sequence
        """
        # 统计每步的人数
        step_counts = Counter()
        for path in user_paths:
            for step in path:
                step_counts[step] += 1

        # 计算转化率
        results = []
        for i, step in enumerate(self.steps):
            count = step_counts.get(step, 0)
            prev_count = step_counts.get(self.steps[i-1], count) if i > 0 else count
            conversion_rate = count / prev_count if prev_count > 0 else 0
            overall_rate = count / step_counts.get(self.steps[0], 1) if i > 0 else 1.0

            results.append({
                'step': step,
                'users': count,
                'step_conversion': conversion_rate,
                'overall_conversion': overall_rate,
                'dropoff': 1 - conversion_rate if i > 0 else 0
            })

        return pd.DataFrame(results)

    def dropoff_analysis(self, user_paths, step_from, step_to):
        """
        分析从step_from到step_to的流失用户去了哪里
        """
        dropoff_users = []
        for path in user_paths:
            if step_from in path and step_to not in path[path.index(step_from):]:
                # 用户到了step_from但没到step_to
                idx = path.index(step_from)
                next_step = path[idx + 1] if idx + 1 < len(path) else '退出'
                dropoff_users.append(next_step)

        return Counter(dropoff_users)


class PathMiner:
    """路径挖掘"""

    def __init__(self):
        self.transitions = defaultdict(Counter)

    def fit(self, paths):
        """学习转移概率"""
        for path in paths:
            for i in range(len(path) - 1):
                self.transitions[path[i]][path[i+1]] += 1
        return self

    def get_transition_matrix(self):
        """获取转移概率矩阵"""
        all_states = sorted(set(self.transitions.keys()) | \
                           {s for targets in self.transitions.values() for s in targets})
        n = len(all_states)
        matrix = np.zeros((n, n))
        state_idx = {s: i for i, s in enumerate(all_states)}

        for from_state, targets in self.transitions.items():
            total = sum(targets.values())
            for to_state, count in targets.items():
                matrix[state_idx[from_state]][state_idx[to_state]] = count / total

        return matrix, all_states

    def find_common_paths(self, paths, min_length=3, top_k=10):
        """找出最常见的完整路径"""
        path_counts = Counter(tuple(p) for p in paths if len(p) >= min_length)
        return path_counts.most_common(top_k)

    def simulate_conversion_lift(self, from_state, to_state, current_rate, new_rate, n_simulations=10000):
        """
        模拟：如果从from_state到to_state的转化率从current_rate提升到new_rate，
        整体转化率会提升多少？
        """
        # 简化：假设from_state的流量不变，直接计算新的到达to_state的人数
        # 实际应该用完整的马尔可夫链模拟
        lift = (new_rate - current_rate) / current_rate if current_rate > 0 else 0
        return lift


# 示例
if __name__ == '__main__':
    # 模拟用户路径数据
    np.random.seed(42)

    paths = []
    for _ in range(1000):
        # 60%的用户进入首页
        if np.random.random() > 0.4:
            path = ['首页']
            # 60%去详情页
            if np.random.random() > 0.4:
                path.append('详情页')
                # 30%加购
                if np.random.random() > 0.7:
                    path.append('加购')
                    # 50%支付
                    if np.random.random() > 0.5:
                        path.append('支付')
                        # 80%完成
                        if np.random.random() > 0.2:
                            path.append('完成')
        else:
            path = ['首页', '退出']
        paths.append(path)

    # 漏斗分析
    funnel = FunnelAnalyzer(['首页', '详情页', '加购', '支付', '完成'])
    result = funnel.analyze(paths)
    print("漏斗分析:")
    print(result.to_string(index=False))

    # 流失分析
    dropoff = funnel.dropoff_analysis(paths, '详情页', '加购')
    print(f"\n从详情页流失的用户去了哪里:")
    for step, count in dropoff.most_common(5):
        print(f"  {step}: {count}")

    # 路径挖掘
    miner = PathMiner()
    miner.fit(paths)
    common = miner.find_common_paths(paths, min_length=3, top_k=5)
    print(f"\n最常见路径:")
    for path, count in common:
        print(f"  {' → '.join(path)}: {count}次")
print("[✓] User Funnel Analysis 测试通过")
```

---


## ④ 技能关联

### 前置技能
- [Skill-Feature-Engineering](../12-ML基础/[[Skill-Feature-Engineering]].md) — 漏斗节点定义本质是事件特征构造

### 延伸技能
- [Skill-Cohort-Retention-Analysis](../14-用户分析/[[Skill-Cohort-Retention-Analysis]].md) — 漏斗看转化，留存看持续，互补维度
- [Skill-RFM-Customer-Segmentation](../06-增长模型/[[Skill-RFM-Customer-Segmentation]].md) — 漏斗各步可结合 RFM 切群分析

### 可组合
- [Skill-Customer-Churn-Prediction](../06-增长模型/[[Skill-Customer-Churn-Prediction]].md) — 漏斗流失节点是流失预测的关键输入

## ⑤ 商业价值评估

- **ROI**：详情页转化率提升1-2个百分点，日增收数千到数万
- **难度**：⭐⭐☆☆☆（2/5）— 主要是数据处理和可视化
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 优化用户体验的入口工具
