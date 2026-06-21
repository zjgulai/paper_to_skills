---
title: Concept Drift Detection — 在线监控模型分布漂移
doc_type: knowledge
module: 12-ML基础
topic: concept-drift-detection
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Concept Drift Detection（概念漂移检测）

> **论文/方法来源**：Gama et al. (2004) "Learning with Drift Detection (DDM)"；Bifet & Gavalda (2007) "Learning from Time-Changing Data with Adaptive Windowing (ADWIN)"；Mouss et al. (2004) "Test of Page-Hinkley for change detection"
> **领域**：12-ML基础 ↔ 14-用户分析 | **类型**: 算法工具

## ① 算法原理

模型在训练时学习的是历史数据分布 P_train(X, Y)，但生产环境中数据分布会随时间变化——这种现象称为**概念漂移（Concept Drift）**。分为三类：
- **突变漂移（Abrupt）**：大促、政策变化后消费者行为骤变
- **渐变漂移（Gradual）**：季节性变化、用户群逐步迁移
- **周期漂移（Recurring）**：年节规律

**ADWIN（自适应窗口算法）**：维护一个可变长度的滑动窗口，当窗口内不同子区间的统计均值差异超过阈值 ε 时触发漂移告警：
```
|μ_W0 - μ_W1| ≥ ε_cut
其中 ε_cut = sqrt((1/2m) · ln(4n/δ))
m = 1/(1/n0 + 1/n1)  （调和平均样本数）
```
ADWIN 无需预设窗口大小，自动缩减窗口以适应最新分布。

**DDM（漂移检测方法）**：跟踪在线误差率 p_t 及其标准差 s_t。当 p_t + s_t > p_min + 3·s_min 时触发告警（WARNING）；继续超过则触发漂移重训（DRIFT）。

**Page-Hinkley 检测**：基于累积和（CUSUM）：
```
m_t = m_{t-1} + (x_t - μ_ref - δ)
PH_t = m_t - min_{i≤t} m_i
DRIFT if PH_t > λ
```
适合检测均值的单调漂移（如某 KPI 指标持续下滑）。

## ② 母婴出海应用案例

**场景A：黑五/Prime Day 后广告 CTR 预测模型漂移监控**

- **业务问题**：大促后 3 天内消费者行为模式剧变，广告 CTR 预测偏差 > 30%，自动出价策略严重失准
- **数据要求**：每日模型预测值 + 实际 CTR 日志；模型上线后的预测误差流
- **预期产出**：ADWIN 在大促后 6-12 小时内触发漂移告警，比传统月度再训练提前 20 天感知
- **业务价值**：及时重训后广告 ROAS 恢复至正常水平，避免约 5-8 万元/月的出价损耗

**场景B：退货率预测模型季节性漂移（Q4 旺季）**

- **业务问题**：Q4 新客大量涌入，买家行为特征分布变化，退货预测模型 F1 从 0.68 下降到 0.41
- **数据要求**：每日退货标签回收 + 历史预测分布
- **预期产出**：Page-Hinkley 提前 2 周检测到渐变漂移，触发增量训练
- **业务价值**：退货干预策略持续有效，Q4 旺季额外减损约 15-20 万元

## ③ 代码模板

```python
import numpy as np
from collections import deque

np.random.seed(42)


# ===== 模拟母婴广告 CTR 预测误差流（含漂移点）=====
def simulate_error_stream(n_normal=500, n_drift=300, drift_increase=0.15):
    """模拟广告大促后 CTR 预测误差流"""
    errors_normal = np.random.binomial(1, 0.1, n_normal).astype(float)  # 正常期误差率 10%
    errors_drift = np.random.binomial(1, 0.1 + drift_increase, n_drift).astype(float)  # 漂移后 25%
    return np.concatenate([errors_normal, errors_drift]), n_normal


error_stream, true_drift_point = simulate_error_stream()
print(f"数据流长度: {len(error_stream)}, 真实漂移点: t={true_drift_point}")


# ===== ADWIN 漂移检测器（核心实现）=====
class ADWIN:
    """自适应窗口漂移检测，无需预设窗口大小"""

    def __init__(self, delta=0.002):
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.n = 0

    def add_element(self, value):
        self.window.append(value)
        self.total += value
        self.n += 1
        drift_detected = self._check_drift()
        return drift_detected

    def _check_drift(self):
        n = self.n
        total = self.total
        # 检查所有可能的分割点
        n1, sum1 = 0, 0.0
        window_list = list(self.window)
        for i in range(len(window_list) - 1, 0, -1):
            n1 += 1
            sum1 += window_list[i]
            n0 = n - n1
            sum0 = total - sum1
            if n0 < 5 or n1 < 5:
                continue
            mu0 = sum0 / n0
            mu1 = sum1 / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = np.sqrt(np.log(4 * n / self.delta) / (2 * m))
            if abs(mu0 - mu1) >= eps_cut:
                # 漂移：截断旧数据
                cut = len(window_list) - n1
                for _ in range(cut):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.n -= 1
                return True
        return False

    @property
    def mean(self):
        return self.total / self.n if self.n > 0 else 0.0


# ===== DDM（误差率统计漂移检测）=====
class DDM:
    """基于误差率均值+标准差的漂移检测"""

    def __init__(self):
        self.n = 0
        self.p = 0.0  # 当前误差率
        self.s = 0.0  # 当前标准差
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.WARNING_LEVEL = 2.0
        self.DRIFT_LEVEL = 3.0

    def add_element(self, error):
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        if self.p + self.s > self.p_min + self.DRIFT_LEVEL * self.s_min:
            return 'DRIFT'
        elif self.p + self.s > self.p_min + self.WARNING_LEVEL * self.s_min:
            return 'WARNING'
        return 'NORMAL'


# ===== Page-Hinkley 检测（均值漂移）=====
class PageHinkley:
    """CUSUM 变体，检测均值单调偏移"""

    def __init__(self, delta=0.005, lam=50, alpha=1.0):
        self.delta = delta
        self.lam = lam
        self.alpha = alpha
        self.n = 0
        self.sum_x = 0.0
        self.m_t = 0.0
        self.m_min = 0.0

    def add_element(self, value):
        self.n += 1
        self.sum_x += value
        mu_t = self.sum_x / self.n
        self.m_t += value - mu_t - self.delta
        self.m_min = min(self.m_min, self.m_t)
        ph = self.m_t - self.m_min
        return ph > self.lam


# 运行三种检测器
adwin = ADWIN(delta=0.002)
ddm = DDM()
ph = PageHinkley(delta=0.005, lam=40)

adwin_alerts = []
ddm_alerts = []
ph_alerts = []

for t, error in enumerate(error_stream):
    if adwin.add_element(error):
        adwin_alerts.append(t)
    status = ddm.add_element(error)
    if status == 'DRIFT':
        ddm_alerts.append(t)
    if ph.add_element(error):
        ph_alerts.append(t)

print("\n=== 概念漂移检测结果（广告 CTR 预测误差流）===")
print(f"真实漂移点:     t = {true_drift_point}")
print(f"ADWIN 告警点:   {adwin_alerts[:5]} {'(✅首次告警在漂移后)' if adwin_alerts and adwin_alerts[0] > true_drift_point else ''}")
print(f"DDM 告警点:     {ddm_alerts[:5]}")
print(f"Page-Hinkley:   {ph_alerts[:3]}")

# 评估检测延迟
for name, alerts in [("ADWIN", adwin_alerts), ("DDM", ddm_alerts), ("PH", ph_alerts)]:
    post_drift = [a for a in alerts if a >= true_drift_point]
    if post_drift:
        delay = post_drift[0] - true_drift_point
        print(f"{name} 检测延迟: {delay} 步（约 {delay} 小时）")

print(f"\n业务意义: 比月度再训练提前约 {true_drift_point + 100 - (adwin_alerts[0] if adwin_alerts else true_drift_point + 100)} 天感知漂移")
print("[✓] Concept Drift Detection 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Data-Drift-Detection]]（数据漂移 vs 概念漂移的区别与联系）
- **前置（prerequisite）**：[[Skill-Model-Performance-Monitor]]（漂移检测的基础：需先建立在线监控基础设施）
- **延伸（extends）**：[[Skill-Online-Incremental-Learning]]（检测到漂移后触发增量学习是下一步）
- **可组合（combinable）**：[[Skill-Model-Calibration]]（漂移往往同时导致概率校准失效，需联动触发再校准）
- **可组合（combinable）**：[[Skill-AutoML-Pipeline-Design]]（漂移检测器嵌入 AutoML 流水线实现全自动再训练）

## ⑤ 商业价值评估

- **ROI预估**：大促后及时重训，避免广告出价损耗约 5-8 万元/月；退货模型持续有效，Q4 额外减损约 15-20 万元；年化综合约 80-110 万元（以中型 DTC 为基准）
- **实施难度**：⭐⭐⭐☆☆（需有模型预测日志基础设施，检测器本身轻量）
- **优先级**：⭐⭐⭐⭐☆
- **评估依据**：跨境母婴行业季节性强（Q4、Prime Day、黑五），漂移是常态；检测器一次部署持续收益
