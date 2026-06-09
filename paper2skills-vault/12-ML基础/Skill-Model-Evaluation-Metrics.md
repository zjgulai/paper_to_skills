# Skill Card: Model Evaluation Metrics（模型评估体系）

---

## ① 算法原理

### 核心思想
模型评估体系解决"**模型到底好不好**"这个问题——不是凭感觉，而是用标准化的量化指标从多个维度衡量模型表现。这是所有预测建模的基础能力，也是 ML 工程中模型选型、A/B 测试结果判读、生产监控的必备技能。

### 数学直觉

**混淆矩阵（Confusion Matrix）** — 一切评估指标的源头：

|  | 预测正类 | 预测负类 |
|---|---|---|
| 实际正类 | TP (True Positive) | FN (False Negative) |
| 实际负类 | FP (False Positive) | TN (True Negative) |

**核心指标**：

- **准确率 (Accuracy)**：`(TP+TN) / (TP+TN+FP+FN)` — 最简单的指标，但在不平衡数据上欺骗性强（如流失率 5% 时，全预测"不流失"就有 95% 准确率）
- **精确率 (Precision)**：`TP / (TP+FP)` — 预测为正的样本中有多少是真的。适用于"宁缺毋滥"场景（如高价值用户定向营销）
- **召回率 (Recall)**：`TP / (TP+FN)` — 真正的正类被找出了多少。适用于"宁可错杀不可放过"场景（如欺诈检测）
- **F1-Score**：`2 × Precision × Recall / (Precision + Recall)` — Precision 和 Recall 的调和平均
- **AUC-ROC**：ROC 曲线下面积。随机猜测 = 0.5，完美分类器 = 1.0。核心直觉：随机选一个正样本和一个负样本，模型给正样本打分更高的概率
- **对数损失 (Log Loss)**：`- (y log(p) + (1-y) log(1-p))` — 惩罚"自信的误判"，对概率校准敏感

### 关键假设
- 测试集与训练集同分布（否则评估结果无意义）
- 业务场景决定了哪个指标是主指标（不存在"最好"的指标，只有"最合适"的指标）
- 概率校准评估（Calibration）需要在独立的验证集上进行

---

## ② 母婴出海应用案例

### 场景一：吸奶器流失预测模型的评估与选型

**业务问题**：
我们训练了 3 个流失预测模型（XGBoost / LightGBM / Logistic Regression），需要决定用哪个模型上线到 WF-A 智能补货系统，触发对高风险用户的挽留优惠券。但流失用户只占 5%，简单看准确率会误判——一个"全预测不流失"的模型也有 95% 准确率但完全没用。

**数据要求**：
- 测试集：30,000 条用户样本（1500 个实际流失 + 28500 个留存），需包含 ground truth 标签
- 概率预测：每个模型输出 predict_proba

**预期产出**：
- 多维度评估报告：Precision/Recall/F1/AUC-ROC/PR-AUC
- 业务敏感度分析：在不同阈值下（top-10%/top-20%/top-50%），实际能触达多少流失用户
- 模型选型建议 + 最佳决策阈值

**业务价值**：
- 每个流失用户挽留价值约 $200（LTV），测试集 1500 个流失用户
- 若 Recall 从 0.6 提升到 0.85（选对模型+阈值），多挽留 375 个用户 → $75,000/月
- 同时控制 Precision ≥ 0.3（确保发券 ROI 为正）

### 场景二：A/B 实验结果的多维度判读

**业务问题**：
我们在 FB 广告上测试两个素材（A vs B），需要判断 B 是否显著优于 A。但只看转化率差 2% 不够——需要结合置信区间、统计功效、效应量来判断"这是真的提升还是随机波动"。

**数据要求**：
- A 组：5,000 曝光，250 转化
- B 组：5,000 曝光，265 转化
- 用户特征：国家、设备、时段

**预期产出**：
- 转化率差异的 95% 置信区间
- 统计功效分析（样本量是否足够）
- 分段评估（按国家/设备拆分看是否有 Simpson 悖论）
- 决策建议："上线 B" / "继续实验" / "停止"

**业务价值**：
- 避免"假阳性"误上线（历史上 30% 的 A/B "胜出"实际上并不显著）
- 广告预算月均 30 万，每次误判上线投入 2-3 周 = 浪费 5-8 万

---

## ③ 代码模板

```python
"""
Model Evaluation Toolkit
模型评估工具集 — 分类模型多维度评估

适用场景：模型选型、A/B测试结果判读、生产模型健康监控
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """模型评估结果"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    brier_score: float
    confusion: np.ndarray
    
    def summary(self) -> str:
        return (
            f"Accuracy={self.accuracy:.3f} | "
            f"Precision={self.precision:.3f} | "
            f"Recall={self.recall:.3f} | "
            f"F1={self.f1:.3f} | "
            f"AUC-ROC={self.auc_roc:.3f} | "
            f"AUC-PR={self.auc_pr:.3f} | "
            f"LogLoss={self.log_loss:.3f} | "
            f"Brier={self.brier_score:.3f}"
        )


def evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> EvaluationResult:
    """
    分类模型完整评估
    
    Args:
        y_true: 真实标签 [0,1]
        y_prob: 预测概率 [0,1]
        threshold: 分类阈值
    
    Returns:
        EvaluationResult 包含所有评估指标
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    return EvaluationResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc_roc=roc_auc_score(y_true, y_prob),
        auc_pr=average_precision_score(y_true, y_prob),
        log_loss=log_loss(y_true, y_prob),
        brier_score=brier_score_loss(y_true, y_prob),
        confusion=cm
    )


def compare_models(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    多模型对比评估
    
    Args:
        y_true: 真实标签
        model_predictions: {模型名: 概率预测数组}
        threshold: 分类阈值
    
    Returns:
        DataFrame: 各模型多维度对比
    """
    results = []
    for name, y_prob in model_predictions.items():
        r = evaluate_classifier(y_true, y_prob, threshold)
        results.append({
            'Model': name,
            'Accuracy': r.accuracy,
            'Precision': r.precision,
            'Recall': r.recall,
            'F1': r.f1,
            'AUC-ROC': r.auc_roc,
            'AUC-PR': r.auc_pr,
            'LogLoss': r.log_loss,
            'Brier': r.brier_score,
        })
    return pd.DataFrame(results).set_index('Model').round(4)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = 'f1',
    cost_fp: float = 1.0,
    cost_fn: float = 1.0
) -> Tuple[float, float]:
    """
    在业务约束下寻找最优阈值
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        objective: 'f1' | 'profit' | 'recall_at_precision'
        cost_fp: 误报成本（如发券成本）
        cost_fn: 漏报成本（如流失损失）
    
    Returns:
        (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -np.inf
    best_threshold = 0.5
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        if objective == 'f1':
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        elif objective == 'profit':
            score = -cost_fp * fp - cost_fn * fn  # 成本越小越好（取负号转最大化）
        elif objective == 'recall_at_precision':
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = rec if prec >= 0.3 else -1  # 目标 Precision ≥ 0.3
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score


def plot_evaluation_curves(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    绘制评估曲线图：ROC / PR / Calibration / Threshold Analysis
    
    Args:
        y_true: 真实标签
        model_predictions: {模型名: 概率预测}
        figsize: 图大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. ROC Curve
    ax = axes[0, 0]
    for name, y_prob in model_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    
    # 2. Precision-Recall Curve (更适合不平衡数据)
    ax = axes[0, 1]
    for name, y_prob in model_predictions.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f'{name} (AP={ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    
    # 3. Calibration Curve (概率校准)
    ax = axes[1, 0]
    for name, y_prob in model_predictions.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=name)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    
    # 4. Threshold Analysis
    ax = axes[1, 1]
    thresholds = np.linspace(0.01, 0.99, 50)
    for name, y_prob in model_predictions.items():
        precisions, recalls = [], []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
        ax.plot(thresholds, precisions, '--', label=f'{name} Precision')
        ax.plot(thresholds, recalls, '-', label=f'{name} Recall')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall vs Threshold')
    ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.show()


# ============ 测试用例 ============

def _generate_test_data(
    n_samples: int = 10000,
    pos_ratio: float = 0.05
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """生成模拟测试数据 — 模拟母婴流失预测场景"""
    np.random.seed(42)
    y_true = (np.random.random(n_samples) < pos_ratio).astype(int)
    
    models = {}
    # 好的模型：概率与真实标签相关
    good_prob = y_true * np.random.beta(8, 2, n_samples) + \
                (1 - y_true) * np.random.beta(2, 8, n_samples)
    models['Good-XGBoost'] = good_prob
    
    # 一般的模型：中等区分能力
    mid_prob = y_true * np.random.beta(6, 4, n_samples) + \
               (1 - y_true) * np.random.beta(3, 7, n_samples)
    models['Mid-LightGBM'] = mid_prob
    
    # 差的模型：接近随机
    bad_prob = np.random.beta(2, 2, n_samples)
    models['Bad-LogisticReg'] = bad_prob
    
    return y_true, models


if __name__ == '__main__':
    print("=" * 60)
    print("模型评估体系 — 测试运行")
    print("=" * 60)
    
    # 1. 生成测试数据
    y_true, model_preds = _generate_test_data(n_samples=10000, pos_ratio=0.05)
    print(f"\n[数据] {len(y_true)} 条样本 | 正类={y_true.sum()} ({y_true.mean():.1%})")
    
    # 2. 多模型对比
    comparison = compare_models(y_true, model_preds)
    print(f"\n[模型对比]\n{comparison}")
    
    # 3. 最优阈值
    for name, y_prob in model_preds.items():
        if 'Good' in name:
            best_t, best_score = find_optimal_threshold(
                y_true, y_prob, objective='recall_at_precision'
            )
            print(f"\n[阈值优化] {name}: 最优阈值={best_t:.2f}, Recall={best_score:.3f}")
    
    # 4. 可视化（在支持的环境中运行）
    try:
        plot_evaluation_curves(y_true, model_preds)
        print("\n[可视化] 评估曲线已绘制")
    except Exception as e:
        print(f"\n[可视化] 仅在交互环境显示: {e}")
    
    # 5. 断言测试
    r = evaluate_classifier(y_true, model_preds['Good-XGBoost'])
    assert r.auc_roc > 0.7, f"AUC-ROC should be > 0.7, got {r.auc_roc:.3f}"
    assert 0 <= r.precision <= 1
    assert 0 <= r.recall <= 1
    print("\n[验证] ✓ 所有断言通过")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Feature-Engineering]] — 模型输入质量决定评估可信度
- **延伸技能**：
  - [[Skill-Cross-Validation-Strategies]] — 从单次评估到稳健评估
  - [[Skill-Imbalanced-Data-Handling]] — 理解 Precision/Recall 后学习如何处理不平衡
  - [[Skill-Hyperparameter-Optimization]] — 评估驱动超参选择
- **可组合**：
  - **[[Skill-AB-Experimental-Design]]** — A/B 结果的置信区间判读直接依赖评估体系
  - **[[Skill-Customer-Churn-Prediction]]** — 流失模型的 Precision-Recall 阈值选择
  - **[[Skill-ROAS-Budget-Optimization]]** — 广告模型的概率校准影响预算分配精度
  - **[[Skill-Ensemble-Methods]]** — 评估体系驱动集成策略选择

---
- **相关技能**：[[Skill-Feature-Selection]]
- **相关技能**：[[Skill-Data-Drift-Detection]]
- **关联**：[[Skill-GraphDeepAR-Demand-Forecasting]]

## ⑤ 商业价值评估

- **ROI 预估**：模型评估能力贯穿所有 ML 项目。仅流失预测场景——正确的模型选型（从 60% Recall 提升到 85%）每月可多挽留价值 $75,000 的用户；A/B 评估避免"假阳性"上线每次节省 5-8 万试错成本。年化贡献 **200-500 万元**（中型品牌）。
- **实施难度**：⭐⭐☆☆☆（2 星）— scikit-learn 原生支持，纯评估逻辑无需训练，集成成本低
- **优先级评分**：⭐⭐⭐⭐⭐（5 星）— 所有模型工作流的前置基础，图谱被依赖度仅次于 Feature Engineering
- **评估依据**：
  - 当前图谱中 16 个 Skill 依赖 Feature Engineering 作为前置，但无一指向评估体系——说明大量模型 Skill 的评估环节是空白的
  - 商业产出：ROI 高、实施难度低、影响面广
  - 战略价值：ML 基础层最核心的缺失组件
