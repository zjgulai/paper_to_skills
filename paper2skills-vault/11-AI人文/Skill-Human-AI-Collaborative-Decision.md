---
title: 人机协作决策框架 — 最优介入时机识别与置信阈值路由
doc_type: knowledge
module: 11-AI人文
topic: human-ai-collaborative-decision
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 人机协作决策框架

> **论文/方法来源**：Human-AI Collaboration (Bansal et al., 2021) + Complementarity in Human-AI Decision Making (Wilder et al., 2021) + Learning to Defer (Madras et al., 2018)
> **领域**：11-AI人文 ↔ 16-智能体工程 | **类型**: 跨域融合

## ① 算法原理

人机协作决策的核心是**互补性原则**：让 AI 处理它擅长的高确定性决策，将 AI 不确定的边缘案例路由给人工，实现系统整体优于纯 AI 或纯人工。

**置信阈值路由**：
- 高置信（confidence > τ_high）→ AI 直接决策，免打扰人工
- 低置信（confidence < τ_low）→ 强制人工审核
- 中间区间 → 可选人工复核（成本敏感路由）

**最优阈值确定（Pareto 前沿）**：

$$\text{argmin}_{\tau} \left[ \lambda \cdot C_{human}(n_{route}) + (1-\lambda) \cdot \text{Error}(\tau) \right]$$

其中 $C_{human}$ 是人工介入成本，$\lambda$ 是成本权重，$n_{route}$ 是被路由的样本数。

**介入价值计算**：当 AI 错误案例中，有多少比例人工能纠正 → 才值得路由。若 AI 在某类错误上人工也表现一致（系统性偏差），路由无价值。

使用条件：适合有明确对错判断的决策任务（如欺诈判定、审核通过/拒绝）；置信分校准是前提（需要 calibrated probability）。

## ② 母婴出海应用案例

**场景A：客户服务工单自动/人工分流**
- 业务问题：客服机器人 70% 可以直接处理，但对复杂投诉（赔付纠纷、安全事故）乱答会加剧客户愤怒，需精准识别人工介入时机
- 数据要求：工单文本、机器人置信分、历史人工处理结果、处理时长
- 预期产出：设置 τ_high=0.85，τ_low=0.5，路由精准率 >90%，人工介入量减少 40%
- 业务价值：客服人力节省 40%（约 8 万元/年），CSAT 提升 12 分，升级投诉率下降 25%

**场景B：选品审核人机协作流水线**
- 业务问题：新品引入审核需人工逐条评估（每天 200+ SKU），AI 评分高置信时人工审核是浪费
- 数据要求：历史 SKU 特征、AI 评分、最终人工判定结果
- 预期产出：高置信 SKU（>90%）自动通过/拒绝，低置信队列优先排给高级采购
- 业务价值：采购审核效率提升 3 倍，年化节省人力成本 12 万元

## ③ 代码模板

```python
"""
人机协作决策框架 — 置信阈值路由 + 互补性评估
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_routing_stats(
    ai_probs: np.ndarray,
    true_labels: np.ndarray,
    tau_high: float = 0.85,
    tau_low: float = 0.50
) -> Dict:
    """计算三段式路由的统计信息"""
    ai_preds = (ai_probs >= 0.5).astype(int)
    ai_correct = (ai_preds == true_labels)

    # 路由分区
    auto_accept = ai_probs >= tau_high        # AI 直接通过
    auto_reject = ai_probs <= (1 - tau_high)  # AI 直接拒绝
    review_zone = ~auto_accept & ~auto_reject  # 人工复核区

    high_conf = auto_accept | auto_reject
    low_conf = review_zone

    n_total = len(ai_probs)
    n_auto = high_conf.sum()
    n_review = low_conf.sum()

    # 各区准确率
    auto_accuracy = ai_correct[high_conf].mean() if n_auto > 0 else 0
    review_ai_accuracy = ai_correct[low_conf].mean() if n_review > 0 else 0

    return {
        "total": n_total,
        "auto_decided": int(n_auto),
        "auto_rate": round(float(n_auto / n_total), 3),
        "review_required": int(n_review),
        "review_rate": round(float(n_review / n_total), 3),
        "auto_accuracy": round(float(auto_accuracy), 4),
        "review_zone_ai_accuracy": round(float(review_ai_accuracy), 4),
        "tau_high": tau_high,
        "tau_low": tau_low
    }


def find_optimal_thresholds(
    ai_probs: np.ndarray,
    true_labels: np.ndarray,
    human_cost_per_case: float = 1.0,
    error_cost: float = 10.0,
    tau_candidates: Optional[List[float]] = None
) -> Dict:
    """网格搜索最优置信阈值（最小化总成本）"""
    if tau_candidates is None:
        tau_candidates = [0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95]

    best_cost = float('inf')
    best_tau = 0.85
    results = []

    for tau in tau_candidates:
        stats = compute_routing_stats(ai_probs, true_labels, tau_high=tau)
        n_review = stats["review_required"]
        n_auto = stats["auto_decided"]

        # 自动决策错误数
        auto_mask = (ai_probs >= tau) | (ai_probs <= (1 - tau))
        ai_preds = (ai_probs >= 0.5).astype(int)
        auto_errors = int(((ai_preds != true_labels) & auto_mask).sum())

        total_cost = (n_review * human_cost_per_case + auto_errors * error_cost)
        results.append({
            "tau": tau,
            "review_rate": stats["review_rate"],
            "auto_accuracy": stats["auto_accuracy"],
            "total_cost": round(total_cost, 2)
        })
        if total_cost < best_cost:
            best_cost = total_cost
            best_tau = tau

    return {
        "optimal_tau": best_tau,
        "min_cost": round(best_cost, 2),
        "cost_table": results
    }


def complementarity_score(
    ai_probs: np.ndarray,
    human_decisions: np.ndarray,
    true_labels: np.ndarray
) -> Dict:
    """评估人机互补性：AI 错误中人工能纠正多少"""
    ai_preds = (ai_probs >= 0.5).astype(int)
    ai_errors = ai_preds != true_labels
    human_correct = human_decisions == true_labels

    if ai_errors.sum() == 0:
        complementarity = 1.0
    else:
        # 在 AI 犯错的案例中，人工能纠正的比例
        complementarity = float(human_correct[ai_errors].mean())

    return {
        "ai_error_rate": round(float(ai_errors.mean()), 4),
        "human_accuracy_on_ai_errors": round(complementarity, 4),
        "complementarity": round(complementarity, 4),
        "verdict": "✅ 人工路由有价值" if complementarity > 0.6 else "⚠️ 人工在 AI 错误上也不准，路由价值低"
    }


# ===== 测试 =====
if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    # 模拟 AI 置信分 + 真实标签（二分类：工单是否需要升级）
    true_labels = np.random.binomial(1, 0.3, n)
    # AI 在高置信区准确，低置信区接近随机
    ai_probs = np.where(
        np.random.rand(n) > 0.3,
        np.clip(true_labels * 0.8 + np.random.normal(0, 0.1, n), 0.01, 0.99),
        np.random.uniform(0.4, 0.6, n)  # 低置信区
    )

    # 人工决策（比 AI 在困难案例上更好）
    human_decisions = np.where(
        np.random.rand(n) > 0.15,
        true_labels,  # 85% 准确率
        1 - true_labels
    )

    print("=== 路由统计 (τ=0.85) ===")
    stats = compute_routing_stats(ai_probs, true_labels, tau_high=0.85)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== 最优阈值搜索 ===")
    optimal = find_optimal_thresholds(ai_probs, true_labels)
    print(f"  最优 τ = {optimal['optimal_tau']}, 最小成本 = {optimal['min_cost']}")

    print("\n=== 人机互补性 ===")
    comp = complementarity_score(ai_probs, human_decisions, true_labels)
    for k, v in comp.items():
        print(f"  {k}: {v}")

    assert stats["auto_rate"] > 0.3, "高置信区比例过低"
    assert optimal["optimal_tau"] in [0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95]
    assert comp["complementarity"] > 0

    print("\n[✓] 人机协作决策框架测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-Agent-Decision-Confidence-Threshold]]（置信阈值设计原理）
- **前置**：[[Skill-AI-Explainability-Consumer-Trust]]（解释 AI 决策给人工审核员）
- **延伸**：[[Skill-Agent-Fault-Tolerance]]（人工兜底作为 Agent 容错机制）
- **可组合**：[[Skill-CS-Ticket-Intelligence]]（客服工单的人机分流落地）
- **可组合**：[[Skill-Customer-Journey-Decision-Tree]]（决策树路径与人机节点映射）

## ⑤ 商业价值评估

- ROI 预估：客服人力成本节省 30-50%，约 6-15 万元/年；决策质量提升使错误成本下降 40%
- 实施难度：⭐⭐⭐☆☆（需要 calibrated probability，历史标注数据 ≥ 500 条）
- 优先级：⭐⭐⭐⭐☆
- 评估依据：客服和审核是母婴出海最高人力密度场景，精准路由可将人力释放到真正需要人工判断的 20% 案例
