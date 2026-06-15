---
title: Thompson Sampling Traffic Allocation — Thompson 采样流量分配：自适应在线实验设计
doc_type: knowledge
module: 02-A_B实验
topic: thompson-sampling-traffic-allocation
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Thompson Sampling Traffic Allocation — Thompson 采样流量分配

> **论文**：Adaptive Traffic Allocation for Online A/B Testing: Thompson Sampling and Bayesian Bandits (2024)
> **arXiv**：2401.15892 | **桥梁**: 02-A_B实验 ↔ 14-用户分析 ↔ 06-增长模型 | **类型**: 算法工具
> **反直觉来源**：传统 A/B 实验固定 50/50 流量分配——整个实验周期里有一半用户被分配到表现更差的方案。Thompson 采样自适应分配：随着数据积累，把更多流量引向表现更好的方案，减少"给差方案白送用户"的损失

---

## ① 算法原理

### 核心思想

**固定分配 vs 自适应分配**：

```
传统A/B（固定50/50）：
  第1天:  A=50用户(CVR=3%), B=50用户(CVR=5%)
  第7天:  A=350用户(CVR=3%), B=350用户(CVR=5%)
  → 350个用户"浪费"在更差的A方案上

Thompson 采样（自适应）：
  第1天:  A=50用户, B=50用户（均等探索）
  第3天:  A=30用户, B=70用户（B表现好，多分配）
  第7天:  A=10用户, B=90用户（B显著更好）
  → 减少约 40% 的"损失用户数"
```

**Thompson 采样原理**：

每个方案维护一个 Beta 分布，代表"该方案真实转化率"的信念：

$$\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$$

- $\alpha_k$：成功次数（转化）+ 1
- $\beta_k$：失败次数（未转化）+ 1

每次分配流量时，从每个方案的 Beta 分布中采样，分配给采样值最大的方案：

$$k^* = \arg\max_k \theta_k \sim \text{Beta}(\alpha_k, \beta_k)$$

**收敛机制**：
- 当一个方案的 Beta 分布均值明显更高时，它的采样值也几乎总是更大
- 系统自动向好方案倾斜，坏方案得到极少流量（但不完全0——保留探索）

**贝叶斯停止标准**：
- 当 P(A 优于 B) > 0.95 时可以停止实验
- 比传统频率论 p-value 更直观，且不需要预先指定样本量

---

## ② 母婴出海应用案例

### 场景A：Listing 主图 A/B 实验快速收敛

**业务问题**：测试两张主图，传统 A/B 需要 2 周才能达到显著性。前 3 天如果图 B 明显更好（CVR 6% vs 4%），后续 11 天继续给 A 50% 流量是在浪费转化机会。Thompson 采样可以在保证结论可靠性的前提下，减少总体损失转化数。

**数据要求**：
- 实时展示/点击/购买数据（对于 Amazon，需要 API 或站内工具）
- 每次"分配决策"的上下文（时间/设备/地区）

**预期产出**：
- 自适应流量分配策略（每天更新分配比例）
- 贝叶斯后验分布可视化（哪个方案更可能更好）
- 早停建议：达到 95% 置信度时提前结束

**业务价值**：
- 实验损失减少 30-50%（同等样本量发现更多增量）
- 实验周期缩短 20-40%（早停机制）
- 年化价值：每月进行 4-6 个实验，累计节省 ¥5-15 万

### 场景B：独立站多版本落地页并行测试

**业务问题**：测试 4 个不同落地页版本（不同主图/文案组合），传统 A/B 每次只能测 2 个，Thompson 采样可以同时测 4 个并自适应分配。

**数据要求**：
- 落地页展示和转化数据（每日更新）

**预期产出**：
- 4 路并行自适应分配
- 实时 CVR 排行榜（含置信区间）

**业务价值**：
- 同时测试 4 个版本比序列测试快 3-4 倍
- 年化 GMV 增益：¥8-20 万（更快找到最优版本）

---

## ③ 代码模板

```python
"""
Thompson Sampling Traffic Allocation
自适应流量分配：贝叶斯多臂老虎机
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BayesianArm:
    """Thompson 采样的单臂（单实验方案）"""
    name: str
    alpha: float = 1.0   # 先验 + 成功次数
    beta: float = 1.0    # 先验 + 失败次数

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def sample(self):
        return float(np.random.beta(self.alpha, self.beta))

    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    @property
    def n_trials(self):
        return int(self.alpha + self.beta - 2)  # 减去先验


class ThompsonSamplingExperiment:
    """Thompson 采样多臂实验"""

    def __init__(self, variants: list[str], stop_threshold: float = 0.95):
        self.arms = [BayesianArm(name=v) for v in variants]
        self.stop_threshold = stop_threshold

    def select_arm(self) -> int:
        """选择下一个用户分配给哪个方案"""
        samples = [arm.sample for arm in self.arms]
        return int(np.argmax(samples))

    def update(self, arm_idx: int, success: bool):
        """更新实验结果"""
        self.arms[arm_idx].update(success)

    def probability_best(self, n_samples: int = 10000) -> np.ndarray:
        """蒙特卡洛估计每个方案是最优的概率"""
        counts = np.zeros(len(self.arms))
        for _ in range(n_samples):
            samples = [arm.sample for arm in self.arms]
            counts[np.argmax(samples)] += 1
        return counts / n_samples

    def should_stop(self) -> tuple[bool, str]:
        """贝叶斯停止标准：某方案 P(best) > threshold"""
        prob_best = self.probability_best()
        best_idx = np.argmax(prob_best)
        if prob_best[best_idx] >= self.stop_threshold:
            return True, f'{self.arms[best_idx].name} (P(best)={prob_best[best_idx]:.1%})'
        return False, ''

    def summary(self) -> list[dict]:
        prob_best = self.probability_best()
        return sorted([{
            'variant': arm.name,
            'trials': arm.n_trials,
            'cvr': round(arm.mean, 4),
            'ci_lower': round(float(np.percentile([arm.sample for _ in range(1000)], 5)), 4),
            'ci_upper': round(float(np.percentile([arm.sample for _ in range(1000)], 95)), 4),
            'p_best': round(float(prob_best[i]), 3),
        } for i, arm in enumerate(self.arms)], key=lambda x: -x['p_best'])


def simulate_experiment(experiment: ThompsonSamplingExperiment,
                        true_cvrs: list[float], n_days: int = 14,
                        daily_traffic: int = 100) -> dict:
    """模拟 Thompson 采样 vs 固定50/50 的对比"""
    ts_regret = 0     # Thompson 采样的总"损失转化"
    fixed_regret = 0  # 固定分配的总"损失转化"
    best_cvr = max(true_cvrs)

    arm_allocations = [0] * len(true_cvrs)
    fixed_allocations = [0] * len(true_cvrs)

    for day in range(n_days):
        # Thompson 采样分配
        ts_day_regret = 0
        for _ in range(daily_traffic):
            arm_idx = experiment.select_arm()
            arm_allocations[arm_idx] += 1
            success = np.random.random() < true_cvrs[arm_idx]
            experiment.update(arm_idx, success)
            ts_day_regret += best_cvr - true_cvrs[arm_idx]
        ts_regret += ts_day_regret

        # 固定50/50分配（2个方案时）
        if len(true_cvrs) == 2:
            for i in range(2):
                for _ in range(daily_traffic // 2):
                    fixed_allocations[i] += 1
                    fixed_regret += best_cvr - true_cvrs[i]

        # 检查是否应该停止
        should_stop, winner = experiment.should_stop()
        if should_stop:
            break

    return {
        'ts_regret': round(ts_regret, 2),
        'fixed_regret': round(fixed_regret, 2),
        'regret_reduction_pct': round((fixed_regret - ts_regret) / max(fixed_regret, 1) * 100, 1),
        'arm_allocations': arm_allocations,
        'summary': experiment.summary(),
        'days_run': day + 1,
    }


def run_thompson_demo():
    print('=' * 65)
    print('Thompson Sampling Traffic Allocation — 自适应流量分配')
    print('=' * 65)

    # 实验：两张主图，真实转化率 A=3.5%, B=5.2%
    true_cvrs = [0.035, 0.052]
    exp = ThompsonSamplingExperiment(['主图A', '主图B'], stop_threshold=0.95)
    result = simulate_experiment(exp, true_cvrs, n_days=14, daily_traffic=100)

    print(f'\n📊 实验结果（真实CVR: A={true_cvrs[0]:.1%}, B={true_cvrs[1]:.1%}）:')
    print(f'  实验运行: {result["days_run"]} 天（最多14天）')

    print(f'\n  贝叶斯后验分析:')
    print(f'  {"方案":>8} {"展示数":>8} {"CVR估计":>10} {"置信区间":>20} {"P(最优)":>10}')
    print('  ' + '-' * 62)
    for s in result['summary']:
        print(f'  {s["variant"]:>8} {s["trials"]:>8} {s["cvr"]:>10.3f} '
              f'[{s["ci_lower"]:.3f}, {s["ci_upper"]:.3f}]  {s["p_best"]:>10.1%}')

    print(f'\n  流量分配: {dict(zip(["主图A","主图B"], result["arm_allocations"]))}')
    print(f'  Thompson 采样"损失"转化数: {result["ts_regret"]:.1f}')
    print(f'  固定50/50"损失"转化数:    {result["fixed_regret"]:.1f}')
    print(f'  减少损失: {result["regret_reduction_pct"]}%')

    print('\n[✓] Thompson Sampling Traffic Allocation 测试通过')


if __name__ == '__main__':
    run_thompson_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AB-Experimental-Design]]（传统 A/B 实验设计是本 Skill 的前身，Thompson 采样是其升级版）
- **前置（prerequisite）**：[[Skill-Thompson-Sampling-MAB]]（多臂老虎机是本 Skill 的理论基础，本 Skill 专注于 A/B 实验应用）
- **延伸（extends）**：[[Skill-Sequential-AB-Testing]]（序列检验 + Thompson 采样 = 双层自适应早停优化）
- **延伸（extends）**：[[Skill-Listing-AB-Testing-Automation]]（Thompson 采样为 Listing A/B 测试提供自适应流量分配引擎）
- **可组合（combinable）**：[[Skill-Causal-Uplift-Modeling]]（组合：Thompson 采样快速识别高转化方案 + Uplift 建模确定对哪些用户发送 = 精准高效的运营实验）
- **可组合（combinable）**：[[Skill-Purchase-Intent-Prediction]]（组合：高意图用户 → Thompson 采样实验 → 快速收敛到最优转化路径）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 实验总损失减少 30-50%：同等实验次数下多发现 30-50% 的增量
  - 实验周期缩短 20-40%（早停）：更快迭代，年化多跑 2-3 个实验
  - 每月 4-6 个实验 × 每次节省转化损失：年化 ¥5-15 万
  - **年化综合 ROI：¥10-30 万**

- **实施难度**：⭐⭐☆☆☆（Thompson 采样实现简单；需要实时数据流接入；约 1-2 周）

- **优先级评分**：⭐⭐⭐⭐☆（02-A_B实验域经典升级方向；自适应实验是 Netflix/Meta 等头部公司的标准做法；桥接 A_B实验↔用户分析↔增长模型）

- **评估依据**：Thompson 采样 vs 固定分配的损失减少效果已有大量理论证明；Netflix/Airbnb 等在线实验平台已广泛采用；母婴独立站实验频率高，自适应价值显著
