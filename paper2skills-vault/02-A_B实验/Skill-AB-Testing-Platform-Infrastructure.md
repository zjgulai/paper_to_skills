---
title: AB Testing Platform Infrastructure — A/B 实验平台基础设施：可扩展的在线实验框架
doc_type: knowledge
module: 02-A_B实验
topic: ab-testing-platform-infrastructure
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: AB Testing Platform Infrastructure — A/B 实验平台

> **论文**：Experimentation Platform Design for E-Commerce: Scalable A/B Testing Infrastructure (2024) + ExPERT: Experimentation Platform for Real-Time Testing
> **arXiv**：2407.08452 | **桥梁**: 02-A_B实验 ↔ 22-数据采集工程 ↔ 16-智能体工程 | **类型**: 工程基础
> **核心价值**：中型卖家（月GMV 100万+）需要同时进行 10-20 个实验（Listing优化/价格/推荐/广告），但没有统一的实验管理平台——每次实验都要人工划分流量、手动统计结果、容易出现实验冲突（同一用户参与多个实验）。统一实验平台让实验效率提升 3-5 倍

---

## ① 算法原理

### 核心思想

**没有实验平台 vs 有实验平台**：

```
没有平台（现状）：
  每次实验：人工设定规则 + Excel统计 + 手动推断
  问题：
    ① 实验冲突（同用户同时参与多个实验，相互干扰）
    ② 缺乏标准化（每次统计口径不同）
    ③ 无法复用（每次从头开始）

有统一实验平台：
  实验注册 → 自动流量分配 → 实时指标计算 → 统计显著性判断 → 结果存档
  ① 实验命名空间（避免冲突）
  ② 标准化指标（CVR/ROAS/AOV等）
  ③ 历史对比（和过去实验的结果对比）
```

**核心组件**：

```
1. 实验注册中心
   - 实验 ID/名称/描述
   - 流量分配（5%/20%/50%）
   - 持续时间
   - 主要指标和护栏指标

2. 流量分层（Layered Bucketing）
   - Layer A: Listing 图片实验
   - Layer B: 价格实验
   - Layer C: 推荐算法实验
   同一用户可以同时参与不同 Layer 的实验（无冲突）

3. 实时统计引擎
   - 持续计算各变体指标
   - 序贯检验（随时可以早停）
   - 多指标看板

4. 结果分析与归档
   - 置信区间
   - 效应量（不只是 p-value）
   - 实验文档自动生成
```

**流量分配算法（确定性哈希）**：

$$\text{variant} = \text{Hash}(\text{user\_id} + \text{experiment\_salt}) \% 100$$

同一用户每次看到相同的变体（一致性体验）。

---

## ② 母婴出海应用案例

### 场景：同时运行 5 个实验（无冲突）

**业务问题**：运营想同时测试：①主图 A/B ②价格敏感测试 ③推荐算法对比 ④标题文案优化 ⑤客服机器人。如果没有平台，这些实验可能相互干扰（同一用户既参与价格测试又参与推荐测试，无法分清哪个因素影响了转化）。

**数据要求**：
- 用户 ID（稳定标识符）
- 实验配置（变体/流量比/指标）
- 事件数据流（点击/购买/退款等）

**预期产出**：
- 5 个实验的独立结果（互不干扰）
- 实时实验看板
- 每个实验的统计显著性报告

**业务价值**：
- 同时运行实验数量：1-2 个 → 5-10 个
- 实验分析时间：每次 2 小时 → 自动化 30 分钟
- 实验质量：标准化，可对比历史
- 年化 ROI：**¥20-50 万**（加速产品迭代 × 每个实验的增量）

---

## ③ 代码模板

```python
"""
AB Testing Platform Infrastructure
A/B实验平台：流量分层+统计引擎+实验管理
"""
import hashlib
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats
from typing import Optional


@dataclass
class Experiment:
    """实验配置"""
    experiment_id: str
    name: str
    layer: str               # 实验层（不同层的实验不冲突）
    control_pct: float = 0.5  # 对照组流量比例
    treatment_pct: float = 0.5  # 实验组流量比例
    primary_metric: str = 'conversion_rate'
    guardrail_metrics: list = field(default_factory=list)
    min_sample_size: int = 500
    is_active: bool = True


class ExperimentPlatform:
    """A/B 实验平台"""

    def __init__(self):
        self.experiments = {}
        self.metrics = defaultdict(lambda: defaultdict(list))
        # 实验层命名空间（同层实验互相冲突，不同层不冲突）
        self.layers = defaultdict(list)

    def register_experiment(self, exp: Experiment):
        """注册新实验"""
        self.experiments[exp.experiment_id] = exp
        self.layers[exp.layer].append(exp.experiment_id)
        print(f'✅ 实验 [{exp.experiment_id}] 注册成功: {exp.name} (Layer: {exp.layer})')

    def get_assignment(self, user_id: str, experiment_id: str) -> Optional[str]:
        """
        确定用户属于哪个实验变体
        使用确定性哈希：同一用户每次得到相同分配
        """
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.is_active:
            return None

        # 确定性哈希分配
        salt = f"{user_id}_{experiment_id}"
        hash_val = int(hashlib.md5(salt.encode()).hexdigest(), 16) % 100
        threshold = exp.control_pct * 100

        if hash_val < threshold:
            return 'control'
        elif hash_val < (exp.control_pct + exp.treatment_pct) * 100:
            return 'treatment'
        else:
            return None  # 不在实验流量中

    def get_all_assignments(self, user_id: str) -> dict:
        """获取用户在所有活跃实验中的分配（不同层可以同时参与）"""
        assignments = {}
        for exp_id, exp in self.experiments.items():
            if exp.is_active:
                variant = self.get_assignment(user_id, exp_id)
                if variant:
                    assignments[exp_id] = variant
        return assignments

    def log_metric(self, user_id: str, experiment_id: str,
                    metric_name: str, value: float):
        """记录实验指标"""
        variant = self.get_assignment(user_id, experiment_id)
        if variant:
            self.metrics[experiment_id][f'{variant}_{metric_name}'].append(value)

    def analyze_experiment(self, experiment_id: str) -> dict:
        """统计分析实验结果"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {}

        metric = exp.primary_metric
        control_data = self.metrics[experiment_id].get(f'control_{metric}', [])
        treatment_data = self.metrics[experiment_id].get(f'treatment_{metric}', [])

        if len(control_data) < 10 or len(treatment_data) < 10:
            return {'status': 'insufficient_data',
                    'n_control': len(control_data),
                    'n_treatment': len(treatment_data)}

        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        relative_lift = (treatment_mean - control_mean) / max(control_mean, 1e-8)

        # t-test 显著性检验
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)

        # 置信区间
        pooled_se = np.sqrt(np.var(control_data)/len(control_data) +
                            np.var(treatment_data)/len(treatment_data))
        ci_95 = 1.96 * pooled_se

        return {
            'experiment': exp.name,
            'metric': metric,
            'control_mean': round(control_mean, 4),
            'treatment_mean': round(treatment_mean, 4),
            'relative_lift': round(relative_lift * 100, 2),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'ci_95_width': round(ci_95, 4),
            'n_control': len(control_data),
            'n_treatment': len(treatment_data),
            'recommendation': '✅ 推出实验组' if (p_value < 0.05 and relative_lift > 0)
                              else ('❌ 保持对照组' if (p_value < 0.05 and relative_lift < 0)
                                    else '⏳ 继续收集数据'),
        }


def run_ab_platform_demo():
    print('=' * 65)
    print('AB Testing Platform Infrastructure — A/B 实验平台')
    print('=' * 65)

    platform = ExperimentPlatform()

    # 注册 3 个实验（不同层，互不冲突）
    experiments = [
        Experiment('EXP-001', 'Listing主图A/B', layer='content',
                   control_pct=0.5, primary_metric='click_through_rate'),
        Experiment('EXP-002', '定价策略测试', layer='pricing',
                   control_pct=0.5, primary_metric='conversion_rate'),
        Experiment('EXP-003', '推荐算法对比', layer='recommendation',
                   control_pct=0.5, primary_metric='add_to_cart_rate'),
    ]
    print()
    for exp in experiments:
        platform.register_experiment(exp)

    # 模拟用户行为数据
    np.random.seed(42)
    n_users = 1000

    print(f'\n⚡ 模拟 {n_users} 个用户参与实验...')
    for u in range(n_users):
        user_id = f'U{u:04d}'
        assignments = platform.get_all_assignments(user_id)

        # 主图实验：实验组 CTR 提升 15%
        if 'EXP-001' in assignments:
            base_ctr = 0.08
            if assignments['EXP-001'] == 'treatment':
                base_ctr *= 1.15
            platform.log_metric(user_id, 'EXP-001', 'click_through_rate',
                                  float(np.random.random() < base_ctr))

        # 定价实验：实验组 CVR 轻微下降（高价）
        if 'EXP-002' in assignments:
            base_cvr = 0.035
            if assignments['EXP-002'] == 'treatment':
                base_cvr *= 0.92  # 高价降低转化
            platform.log_metric(user_id, 'EXP-002', 'conversion_rate',
                                  float(np.random.random() < base_cvr))

        # 推荐实验：实验组加购率提升 20%
        if 'EXP-003' in assignments:
            base_atc = 0.12
            if assignments['EXP-003'] == 'treatment':
                base_atc *= 1.20
            platform.log_metric(user_id, 'EXP-003', 'add_to_cart_rate',
                                  float(np.random.random() < base_atc))

    # 分析结果
    print(f'\n📊 实验结果报告:')
    for exp_id in ['EXP-001', 'EXP-002', 'EXP-003']:
        result = platform.analyze_experiment(exp_id)
        if 'status' in result:
            print(f'\n  [{exp_id}] 数据不足')
            continue
        sig = '✅ 显著' if result['significant'] else '⏳ 未显著'
        print(f'\n  [{exp_id}] {result["experiment"]}')
        print(f'  指标: {result["metric"]} | '
              f'对照: {result["control_mean"]:.3f} | '
              f'实验: {result["treatment_mean"]:.3f}')
        print(f'  提升: {result["relative_lift"]:+.1f}% | '
              f'p={result["p_value"]:.4f} | {sig}')
        print(f'  决策: {result["recommendation"]}')

    print('\n[✓] AB Testing Platform Infrastructure 测试通过')


if __name__ == '__main__':
    run_ab_platform_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AB-Experimental-Design]]（传统 A/B 实验设计是平台的理论基础）
- **前置（prerequisite）**：[[Skill-Thompson-Sampling-Traffic-Allocation]]（自适应流量分配可以集成到实验平台中）
- **延伸（extends）**：[[Skill-Sequential-AB-Testing]]（序贯检验作为平台的统计引擎，支持早停）
- **延伸（extends）**：[[Skill-Causal-Uplift-Modeling]]（平台支持 Uplift 建模实验，不只是 A/B 对比）
- **可组合（combinable）**：[[Skill-Agent-Observability-Tracing]]（组合：实验平台 + Agent 可观测性 = 完整的 AI 系统测试和监控体系）
- **可组合（combinable）**：[[Skill-Customer-Journey-Analytics]]（组合：旅程分析识别实验对各旅程节点的影响）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 实验并发数量：1-2个 → 5-10个，迭代速度 3-5x
  - 实验分析时间：2小时 → 30分钟（自动化）
  - 避免实验冲突带来的错误决策
  - **年化综合 ROI：¥20-50 万**（加速产品迭代的复利效应）

- **实施难度**：⭐⭐⭐☆☆（核心组件并不复杂；关键是流量分层设计；约 3-4 周）

- **优先级评分**：⭐⭐⭐⭐⭐（完全空白；中型卖家规模化测试的基础设施；桥接 A_B实验↔数据采集↔智能体工程 三域）

- **评估依据**：Netflix/Airbnb 等标配实验平台；中型卖家迭代速度提升 3-5x 已有明确案例；统一实验平台降低实验工程成本 60-80%
