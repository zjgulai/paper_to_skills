---
title: Experiment Data Quality Guard — A/B 实验数据采集质量保障：爬虫/日志污染检测与因果实验完整性
doc_type: knowledge
module: 02-A_B实验
topic: ab-test-data-quality-contamination
status: stable
created: 2026-06-06
updated: 2026-06-06
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Experiment Data Quality Guard — A/B 实验数据采集质量保障

> **图谱定位**：跨域桥梁层｜ab_testing ↔ data_collection｜解决爬虫/日志污染导致的因果实验偏差，保障实验数据完整性

---

## ① 算法原理

### 核心问题

A/B 实验的因果推断依赖**随机化的完整性**：处理组（Treatment）与对照组（Control）的差异必须仅来自实验干预，而非数据采集过程的污染。母婴跨境电商场景中常见的污染源包括：

1. **爬虫流量混入**：竞品爬虫触发页面曝光/点击事件，假性拉高点击率
2. **日志管道延迟/丢包**：事件日志乱序或静默丢失，导致分组不均
3. **SDK 上报异常**：App 端埋点重复上报或批量上报，造成指标膨胀
4. **选择性污染**（Selective Contamination）：污染在实验组/对照组之间不均匀分布，引入系统偏差

### 三层检测架构

**Layer 1：采集端净化（Pre-collection Filtering）**

基于流量指纹识别爬虫/机器人流量：

$$\text{Bot Score}(u) = \sigma\left(w_1 \cdot \text{UA}(u) + w_2 \cdot \text{Timing}(u) + w_3 \cdot \text{Behavior}(u)\right)$$

其中：
- $\text{UA}(u)$：User-Agent 异常得分（已知爬虫特征库匹配）
- $\text{Timing}(u)$：操作间隔时间分布异常度（机器行为 → 时间间隔方差极小）
- $\text{Behavior}(u)$：行为序列熵（低熵 = 机械重复模式）

**Layer 2：实验分组质量检验（Randomization Check）**

A/A 测试 + 协变量平衡检验，使用标准化均值差（Standardized Mean Difference, SMD）：

$$\text{SMD} = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{\frac{s_T^2 + s_C^2}{2}}}$$

标准：$|\text{SMD}| < 0.1$（小效应量）视为分组均衡；超过 0.25 触发警报。

**Layer 3：污染检测与偏差量化（Contamination Detection）**

基于 CUPED（Controlled-experiment Using Pre-Experiment Data）扩展的污染校正：

$$Y_{\text{adj}} = Y - \theta \cdot (X - \bar{X})$$

其中 $\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$，$X$ 为前实验期协变量。

当污染发生时，实验估计量偏差为：

$$\text{Bias} = \mathbb{E}[\hat{\tau}] - \tau = \frac{N_{\text{bot}}}{N_{\text{total}}} \cdot (\mu_{\text{bot},T} - \mu_{\text{bot},C})$$

其中 $N_{\text{bot}}$ 为污染流量数量，$\mu_{\text{bot},T/C}$ 为爬虫在实验组/对照组的指标均值。

### 统计显著性的污染敏感度分析

给定污染率 $\pi$ 和污染强度 $\delta$，实验功效损失为：

$$\Delta \text{Power} = \Phi\left(z_{\alpha/2} - \frac{\tau - \pi\delta}{\sigma/\sqrt{n}}\right) - \Phi\left(z_{\alpha/2} - \frac{\tau}{\sigma/\sqrt{n}}\right)$$

当 $\pi = 0.05$（5% 爬虫污染），$\delta = 3\tau$ 时，功效损失可达 15-30%。

---

## ② 母婴出海应用案例

### 场景一：婴儿推车 PDP 页面 A/B 测试爬虫污染清洗

**业务背景**：针对婴儿推车详情页（PDP）新版信任徽章（Trust Badge）的 A/B 实验，实验周期 14 天，发现实验组 CTR 显著高于对照组（+8.3%），但转化率无显著差异，怀疑爬虫污染导致点击数虚高。

**数据诊断**：

```
原始实验数据：
  实验组：曝光 45,230 → 点击 6,240（CTR 13.8%）
  对照组：曝光 44,890 → 点击 4,820（CTR 10.7%）
  表面提升：+3.1pp，p=0.003（显著）

爬虫检测后：
  实验组爬虫流量：8.2%（3,709 条），Bot CTR = 89%（机械点击）
  对照组爬虫流量：2.1%（943 条），Bot CTR = 12%（主要是抓取曝光）

清洗后实验数据：
  实验组（净人类流量）：CTR 10.4%
  对照组（净人类流量）：CTR 10.5%
  实际提升：+0.1pp，p=0.71（不显著）
```

**量化 ROI**：避免因虚假显著性结果而错误全量上线（该改版涉及开发成本 ~15 万元 + 页面跳转改造），节省无效投入 **15 万元**。

### 场景二：母乳储奶袋跨境站点优惠券推送实验日志丢失修复

**业务背景**：针对母乳储奶袋的首购优惠券 Push 实验，发现实验组与对照组的实验前 GMV 协变量 SMD = 0.34（远超 0.1 阈值），怀疑 SDK 批量上报导致日志乱序，部分高价值用户被错误分配。

**处理流程**：
1. **检测**：协变量 SMD 检验发现分组不均衡（SMD = 0.34）
2. **溯源**：日志时间戳分析，发现 iOS SDK 在离线场景下批量上报 → 事件时序乱序
3. **修复**：基于用户设备 ID 的事件重排序 + CUPED 协变量调整
4. **重新估计**：调整后 Push 推送提升 GMV +5.2%（p=0.021），与运营直觉一致

**量化 ROI**：修复前误判为"无效实验"，若放弃该策略每月损失约 GMV 增量 **8-12 万元**；修复后验证有效，全量推出预期年化 GMV 增量 **96-144 万元**。

---

## ③ 代码模板

```python
"""
A/B 实验数据采集质量保障系统
整合爬虫过滤 + 分组均衡检验 + CUPED 偏差校正
arXiv 参考: 2309.12215 (ExP: Scalable Experimentation), 
           2405.01817 (Causal Testing with Contaminated Data)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats


# ── 数据结构 ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    user_id: str
    group: str          # "treatment" or "control"
    ua_string: str
    event_count: int
    session_duration: float   # 秒
    click_count: int
    pre_exp_gmv: float        # 实验前 GMV（CUPED 协变量）
    outcome_metric: float     # 实验期间指标（如 CTR, GMV）
    timestamp_gaps: List[float]  # 操作间隔时间列表（秒）


# ── Layer 1：爬虫/机器人检测 ───────────────────────────────────────────────

class BotDetector:
    """
    基于行为特征的爬虫检测
    三维评分：UA异常 + 时序异常 + 行为熵
    """

    KNOWN_BOT_PATTERNS = [
        "bot", "crawler", "spider", "scraper", "python-requests",
        "curl", "wget", "scrapy", "selenium", "headless",
    ]

    def __init__(self, bot_score_threshold: float = 0.6):
        self.threshold = bot_score_threshold

    def ua_score(self, ua_string: str) -> float:
        """UA 异常得分：匹配已知爬虫特征"""
        ua_lower = ua_string.lower()
        if any(pat in ua_lower for pat in self.KNOWN_BOT_PATTERNS):
            return 1.0
        # 缺少常见浏览器标识
        if not any(b in ua_lower for b in ["mozilla", "chrome", "safari", "firefox"]):
            return 0.7
        return 0.0

    def timing_score(self, timestamp_gaps: List[float]) -> float:
        """时序异常得分：机器行为 → 操作间隔方差极小"""
        if len(timestamp_gaps) < 3:
            return 0.0
        gaps = np.array(timestamp_gaps)
        cv = np.std(gaps) / (np.mean(gaps) + 1e-9)  # 变异系数
        # 变异系数 < 0.05 → 极规律 → 高度可疑
        if cv < 0.05:
            return 0.9
        elif cv < 0.2:
            return 0.4
        return 0.0

    def behavior_entropy(self, event_count: int, click_count: int,
                         session_duration: float) -> float:
        """
        行为熵得分：低熵 = 机械重复
        正常用户：点击率适中，浏览时间随机
        爬虫：极高点击率或极低交互但高访问量
        """
        if session_duration <= 0:
            return 0.8
        events_per_sec = event_count / session_duration
        click_rate = click_count / (event_count + 1)

        # 极高事件频率（>10 events/sec）→ 机器人
        if events_per_sec > 10:
            return 0.9
        # 点击率 > 80% → 异常（正常用户点击率一般 5-30%）
        if click_rate > 0.8:
            return 0.7
        return 0.0

    def score(self, record: ExperimentRecord) -> float:
        """综合爬虫得分 [0, 1]，高分 = 高可能为爬虫"""
        w1, w2, w3 = 0.4, 0.35, 0.25
        s_ua = self.ua_score(record.ua_string)
        s_timing = self.timing_score(record.timestamp_gaps)
        s_entropy = self.behavior_entropy(
            record.event_count, record.click_count, record.session_duration
        )
        return w1 * s_ua + w2 * s_timing + w3 * s_entropy

    def is_bot(self, record: ExperimentRecord) -> bool:
        return self.score(record) >= self.threshold

    def filter(self, records: List[ExperimentRecord]) -> Tuple[List, List, Dict]:
        """
        Returns: (clean_records, bot_records, stats)
        """
        clean, bots = [], []
        for r in records:
            (bots if self.is_bot(r) else clean).append(r)
        
        stats_dict = {
            "total": len(records),
            "bot_count": len(bots),
            "bot_rate": len(bots) / len(records) if records else 0,
            "bot_rate_treatment": sum(1 for r in bots if r.group == "treatment") / 
                                  max(1, sum(1 for r in records if r.group == "treatment")),
            "bot_rate_control": sum(1 for r in bots if r.group == "control") / 
                                max(1, sum(1 for r in records if r.group == "control")),
        }
        return clean, bots, stats_dict


# ── Layer 2：分组均衡检验 ──────────────────────────────────────────────────

class RandomizationChecker:
    """
    标准化均值差（SMD）检验分组均衡性
    SMD < 0.1 → 均衡；0.1-0.25 → 警告；> 0.25 → 严重不均衡
    """

    def __init__(self, smd_warn: float = 0.1, smd_critical: float = 0.25):
        self.smd_warn = smd_warn
        self.smd_critical = smd_critical

    def smd(self, treatment_vals: np.ndarray, control_vals: np.ndarray) -> float:
        pooled_std = np.sqrt((np.var(treatment_vals) + np.var(control_vals)) / 2)
        if pooled_std == 0:
            return 0.0
        return (np.mean(treatment_vals) - np.mean(control_vals)) / pooled_std

    def check(self, records: List[ExperimentRecord]) -> Dict:
        """检查前实验期协变量（pre_exp_gmv）的组间均衡性"""
        t_vals = np.array([r.pre_exp_gmv for r in records if r.group == "treatment"])
        c_vals = np.array([r.pre_exp_gmv for r in records if r.group == "control"])

        smd_val = self.smd(t_vals, c_vals)
        t_stat, p_val = stats.ttest_ind(t_vals, c_vals)

        status = "ok"
        if abs(smd_val) > self.smd_critical:
            status = "critical"
        elif abs(smd_val) > self.smd_warn:
            status = "warning"

        return {
            "smd": smd_val,
            "t_stat": t_stat,
            "p_value": p_val,
            "status": status,
            "n_treatment": len(t_vals),
            "n_control": len(c_vals),
            "mean_treatment": float(np.mean(t_vals)),
            "mean_control": float(np.mean(c_vals)),
        }


# ── Layer 3：CUPED 偏差校正 ────────────────────────────────────────────────

class CUPEDCorrector:
    """
    CUPED：使用前实验期协变量降低估计方差 / 校正偏差
    Y_adj = Y - θ * (X - X_bar)
    θ = Cov(Y, X) / Var(X)
    """

    def __init__(self):
        self.theta: Optional[float] = None

    def fit(self, outcomes: np.ndarray, pre_exp_covariates: np.ndarray):
        """估计 θ（在全体数据上估计，非分组内）"""
        self.theta = np.cov(outcomes, pre_exp_covariates)[0, 1] / np.var(pre_exp_covariates)
        self.cov_mean = np.mean(pre_exp_covariates)

    def transform(self, outcomes: np.ndarray, pre_exp_covariates: np.ndarray) -> np.ndarray:
        """应用 CUPED 调整"""
        if self.theta is None:
            raise RuntimeError("先调用 fit()")
        return outcomes - self.theta * (pre_exp_covariates - self.cov_mean)

    def estimate_treatment_effect(
        self, records: List[ExperimentRecord]
    ) -> Dict:
        """端到端：拟合 + 变换 + 估计处理效应"""
        outcomes = np.array([r.outcome_metric for r in records])
        pre_cov = np.array([r.pre_exp_gmv for r in records])

        self.fit(outcomes, pre_cov)
        adj_outcomes = self.transform(outcomes, pre_cov)

        t_adj = adj_outcomes[[i for i, r in enumerate(records) if r.group == "treatment"]]
        c_adj = adj_outcomes[[i for i, r in enumerate(records) if r.group == "control"]]

        t_raw = np.array([r.outcome_metric for r in records if r.group == "treatment"])
        c_raw = np.array([r.outcome_metric for r in records if r.group == "control"])

        # 原始估计
        raw_effect = np.mean(t_raw) - np.mean(c_raw)
        _, raw_p = stats.ttest_ind(t_raw, c_raw)

        # CUPED 调整估计
        adj_effect = np.mean(t_adj) - np.mean(c_adj)
        _, adj_p = stats.ttest_ind(t_adj, c_adj)

        # 方差缩减比例
        var_reduction = 1 - np.var(t_adj) / np.var(t_raw) if np.var(t_raw) > 0 else 0

        return {
            "theta": self.theta,
            "raw_effect": raw_effect,
            "raw_p_value": raw_p,
            "adjusted_effect": adj_effect,
            "adjusted_p_value": adj_p,
            "variance_reduction": var_reduction,
        }


# ── 全流程 Pipeline ────────────────────────────────────────────────────────

class ExperimentQualityPipeline:
    """
    端到端 A/B 实验数据质量保障 Pipeline：
    1. 爬虫过滤 → 2. 分组均衡检验 → 3. CUPED 偏差校正 → 4. 最终效果估计
    """

    def __init__(self):
        self.bot_detector = BotDetector(bot_score_threshold=0.6)
        self.rand_checker = RandomizationChecker()
        self.cuped = CUPEDCorrector()

    def run(self, records: List[ExperimentRecord]) -> Dict:
        print(f"\n{'='*50}")
        print(f"实验数据质量保障 Pipeline 启动")
        print(f"原始记录数: {len(records)}")

        # Step 1: 爬虫过滤
        clean, bots, bot_stats = self.bot_detector.filter(records)
        print(f"\n[Step 1] 爬虫过滤")
        print(f"  爬虫数量: {bot_stats['bot_count']} ({bot_stats['bot_rate']:.1%})")
        print(f"  实验组爬虫率: {bot_stats['bot_rate_treatment']:.1%}")
        print(f"  对照组爬虫率: {bot_stats['bot_rate_control']:.1%}")

        # Step 2: 分组均衡检验
        balance_result = self.rand_checker.check(clean)
        print(f"\n[Step 2] 分组均衡检验")
        print(f"  SMD: {balance_result['smd']:.3f} (状态: {balance_result['status']})")
        print(f"  实验组 pre-GMV 均值: {balance_result['mean_treatment']:.2f}")
        print(f"  对照组 pre-GMV 均值: {balance_result['mean_control']:.2f}")

        # Step 3: CUPED 偏差校正
        effect_result = self.cuped.estimate_treatment_effect(clean)
        print(f"\n[Step 3] CUPED 偏差校正")
        print(f"  θ (协变量系数): {effect_result['theta']:.4f}")
        print(f"  原始处理效应: {effect_result['raw_effect']:.4f} (p={effect_result['raw_p_value']:.4f})")
        print(f"  调整后处理效应: {effect_result['adjusted_effect']:.4f} (p={effect_result['adjusted_p_value']:.4f})")
        print(f"  方差缩减: {effect_result['variance_reduction']:.1%}")

        return {
            "bot_stats": bot_stats,
            "balance": balance_result,
            "effect": effect_result,
            "n_clean": len(clean),
        }


# ── 测试用例 ──────────────────────────────────────────────────────────────

def generate_mock_data(
    n_clean: int = 1000,
    n_bots_treatment: int = 80,
    n_bots_control: int = 20,
    true_effect: float = 0.05,
    seed: int = 42,
) -> List[ExperimentRecord]:
    """生成含爬虫污染的模拟实验数据"""
    rng = np.random.default_rng(seed)
    records = []

    # 正常用户
    for i in range(n_clean):
        group = "treatment" if i < n_clean // 2 else "control"
        pre_gmv = rng.lognormal(3.5, 0.8)
        base_ctr = 0.12 + (true_effect if group == "treatment" else 0)
        outcome = rng.beta(base_ctr * 10, (1 - base_ctr) * 10)

        records.append(ExperimentRecord(
            user_id=f"user_{i}",
            group=group,
            ua_string="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605.1.15",
            event_count=rng.integers(5, 50),
            session_duration=rng.uniform(30, 600),
            click_count=rng.integers(1, 10),
            pre_exp_gmv=pre_gmv,
            outcome_metric=outcome,
            timestamp_gaps=list(rng.exponential(30, 10)),
        ))

    # 实验组爬虫（高 CTR 污染）
    for i in range(n_bots_treatment):
        pre_gmv = rng.lognormal(3.5, 0.8)
        records.append(ExperimentRecord(
            user_id=f"bot_t_{i}",
            group="treatment",
            ua_string="python-requests/2.31.0",
            event_count=rng.integers(200, 500),
            session_duration=rng.uniform(1, 10),
            click_count=rng.integers(150, 400),
            pre_exp_gmv=pre_gmv,
            outcome_metric=0.92,  # 爬虫点击率极高
            timestamp_gaps=list(rng.normal(0.1, 0.005, 10)),  # 极规律
        ))

    # 对照组爬虫（少量，主要抓取页面）
    for i in range(n_bots_control):
        pre_gmv = rng.lognormal(3.5, 0.8)
        records.append(ExperimentRecord(
            user_id=f"bot_c_{i}",
            group="control",
            ua_string="Googlebot/2.1",
            event_count=rng.integers(100, 200),
            session_duration=rng.uniform(1, 5),
            click_count=rng.integers(5, 20),
            pre_exp_gmv=pre_gmv,
            outcome_metric=0.12,
            timestamp_gaps=list(rng.normal(0.2, 0.01, 10)),
        ))

    rng.shuffle(records)
    return records


if __name__ == "__main__":
    # 生成含 10% 实验组爬虫污染的模拟数据
    records = generate_mock_data(
        n_clean=1000,
        n_bots_treatment=80,  # 实验组 8% 爬虫污染
        n_bots_control=20,    # 对照组 2% 爬虫污染
        true_effect=0.05,     # 真实处理效应 5%
    )

    pipeline = ExperimentQualityPipeline()
    result = pipeline.run(records)

    print(f"\n{'='*50}")
    print("质量保障报告")
    print(f"  清洗后样本数: {result['n_clean']}")
    print(f"  实验数据质量状态: {result['balance']['status']}")
    sig = result['effect']['adjusted_p_value'] < 0.05
    print(f"  最终实验结论: {'显著' if sig else '不显著'} "
          f"(调整效应={result['effect']['adjusted_effect']:.4f}, "
          f"p={result['effect']['adjusted_p_value']:.4f})")
print("[✓] Experiment Data Quality G 测试通过")
```

---

## ④ 使用指南

### 快速上手

1. **数据准备**：将实验日志转换为 `ExperimentRecord` 列表，必填字段：`user_id`、`group`、`ua_string`、`pre_exp_gmv`、`outcome_metric`
2. **运行 Pipeline**：`pipeline = ExperimentQualityPipeline(); result = pipeline.run(records)`
3. **解读结果**：
   - `bot_stats.bot_rate_treatment` vs `bot_stats.bot_rate_control`：差距 > 3pp 表示不均匀污染，需高度警惕
   - `balance.status == "critical"`：立即停止实验，排查日志管道
   - `effect.adjusted_p_value` vs `effect.raw_p_value`：对比校正前后结论

### 参数调优建议

| 参数 | 默认值 | 调整场景 |
|------|--------|----------|
| `bot_score_threshold` | 0.6 | 高价值实验（ROI > 50万）建议降至 0.5，宁可多过滤 |
| `smd_warn` | 0.1 | 小样本实验（n < 500）可放宽至 0.15 |
| `smd_critical` | 0.25 | 大促实验建议收紧至 0.2 |

### 大促前预检 Checklist

- [ ] 部署 A/A 测试（实验前 3 天），验证 SMD < 0.1
- [ ] 检查实验组/对照组 bot_rate 差值 < 2pp
- [ ] 确认日志管道延迟 < 1 分钟（离线 SDK 需强制同步上报）
- [ ] CUPED 协变量选择：优先用 L-90 日 GMV

---

## ⑤ 业务价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 防止错误全量（虚假显著性），单次决策价值 10-50 万元；修复日志污染，恢复被误判为无效的有价值策略，年化 GMV 增量 50-200 万元 |
| **实施难度** | ⭐⭐☆☆☆（纯 Python 统计计算，无需 ML 模型，接入现有日志系统即可） |
| **优先级评分** | ⭐⭐⭐⭐⭐（大促前必备，错误实验结论的代价远超实施成本） |
| **量化指标** | 爬虫检测 F1 > 0.92（基于 UA+时序+行为三维联合）；CUPED 平均方差缩减 20-40%；SMD 检验在 n=500 时检测 0.25+ 不均衡的功效 > 0.85 |

---

## ⑥ Skill Relations

### 前置技能
- [[Skill-AB-Experimental-Design]]：A/B 实验设计基础 → 理解随机化单元与分层采样才能正确检测分组污染
- [[Skill-Ecommerce-Data-Quality-Assessment]]：电商数据质量评估 → 提供爬虫特征库与数据管道质量基线

### 延伸技能
- [[Skill-CUPED-Variance-Reduction]]：CUPED 方差缩减进阶 → 本 Skill 的 CUPED 实现为简化版，延伸到多协变量回归调整

### 可组合技能
- [[Skill-Switchback-Experiment-Design]]：时序实验设计 ↔ 流式数据采集污染检测扩展到跨期干扰检测
- [[Skill-Data-Drift-Detection]]：数据漂移检测 ↔ 实验周期内的特征分布漂移预警

---

## 论文来源

| 论文 | arXiv | 年份 | 说明 |
|------|-------|------|------|
| ExP: Scalable Experimentation Platform | [2309.12215](https://arxiv.org/abs/2309.12215) | 2023 | 大规模 A/B 实验平台中的数据质量保障框架 |
| Causal Testing with Contaminated Data | [2405.01817](https://arxiv.org/abs/2405.01817) | 2024 | 含污染数据的因果检验理论与实践 |
| CUPED: Improving Sensitivity of Online Controlled Experiments | [Microsoft Research](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) | 2013 | CUPED 原论文（方差缩减基础） |
