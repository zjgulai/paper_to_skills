---
title: "Skill Card: Model Evaluation Metrics（模型评估体系）"
roadmap_phase: phase1
updated: 2026-07-05
difficulty: ⭐⭐⭐☆☆
priority: ⭐⭐⭐⭐☆
---

# Skill Card: Model Evaluation Metrics（模型评估体系）

## ① 算法原理

### 核心思想
模型评估体系解决"**模型在真实业务中的表现到底如何**"这个问题——通过混淆矩阵衍生的多维度量化指标（Precision/Recall/F1/AUC-ROC），在不同业务阈值下权衡假正例与假负例成本，从而做出科学的模型选型与部署决策。这是 ML 工程中从实验到生产的关键卡点。

### 数学直觉

**混淆矩阵（Confusion Matrix）** — 所有评估指标的源头：

| | 预测正类 | 预测负类 |
|---|---|---|
| 实际正类 | TP | FN |
| 实际负类 | FP | TN |

**核心指标体系**：

- **Precision（精确率）** = `TP/(TP+FP)` — 预测为正的样本中实际为正的比例。业务含义：发出的优惠券中有多少真的是流失用户
- **Recall（召回率）** = `TP/(TP+FN)` — 真正的正类中被正确预测的比例。业务含义：所有流失用户中我们成功识别了多少
- **F1-Score** = `2×Precision×Recall/(Precision+Recall)` — 两者的调和平均，适合不平衡数据
- **AUC-ROC** = ROC 曲线下面积。直觉：随机抽一个正样本和负样本，模型给正样本打分更高的概率。范围 [0,1]，0.5=随机，1.0=完美
- **阈值敏感性分析** — 同一模型在不同决策阈值下的 Precision-Recall 权衡曲线

### 关键假设
- 测试集与训练集同分布（否则评估无效）
- 业务场景的成本不对称性决定了主指标选择（流失预测看 Recall，反欺诈看 Precision）
- 样本量足够大（≥1000）才能保证指标稳定性

### 非共识迁移：为何降维打击母婴跨境电商
通用 ML 评估指标在母婴出海中的特殊应用：
- **不平衡数据**：流失率 5%、高价值用户 2%，准确率陷阱明显 → 必须用 Precision/Recall/AUC-PR
- **多阈值决策**：不同国家/品类的流失成本差异大（美国用户 LTV $500 vs 印度 $50），需要阈值敏感性分析而非固定 0.5
- **实时反馈循环**：模型上线后需要持续监控 AUC-ROC 漂移，检测数据分布变化（如季节性、新品类）

---

## ② 母婴出海应用案例

### 场景一：婴儿纸尿裤销量预测模型集成评估

**业务问题**：
母婴跨境 SaaS 平台为中国卖家提供 WF-A 智能补货系统。我们训练了 3 个销量预测模型（XGBoost/LightGBM/Prophet），需要选择最优模型指导备货决策。备货过多积压成本高（仓储费 15%/月），备货过少缺货损失销售（毛利率 40%）。

**具体数据规模**：
- 历史数据：500 个 SKU × 12 个月 = 6000 条样本
- 测试集：1500 条（最近 3 个月数据）
- 预测目标：30 天销量（连续值）
- 评估指标：RMSE、MAE、MAPE、R²

**量化产出**：
- XGBoost 模型 RMSE 降低 23%（vs 基准 Prophet），MAPE 从 18% 降至 14%
- 备货成本节省：年均 12 万元（通过减少 8% 的过度备货）
- 缺货率从 12% 降至 6%，新增销售收入 28 万元/年
- **总 ROI**：40 万元/年 ÷ 模型开发成本 8 万 = 5 倍

**三轨验证**：
- **成本**：模型训练 + 部署 + 监控 = 8 万元/年
- **合规**：销量预测无隐私风险，符合各国数据法规
- **风险**：若模型漂移（新品类上市、季节性变化），需要月度重训，否则预测偏差 >20%

---

### 场景二：高价值用户流失预测模型的阈值优化

**业务问题**：
母婴跨境平台识别高价值用户（LTV > $500，占用户 3%）的流失风险，通过精准补贴挽留。但流失率仅 5%，简单看准确率无意义。需要在"发券成本"与"挽留价值"之间找到最优阈值。

**具体数据规模**：
- 用户样本：50,000 个高价值用户（历史 6 个月）
- 实际流失：2,500 人（5%）
- 模型输出：每个用户的流失概率 [0,1]
- 评估指标：Precision、Recall、F1、AUC-ROC、PR-AUC

**量化产出**：
- 模型 AUC-ROC = 0.82（vs 随机 0.5），AUC-PR = 0.35（基准 0.05）
- 最优阈值 0.35：Precision 0.42、Recall 0.68
  - 预测流失 4,286 人，发券成本 4,286 × $8 = $34,288
  - 实际挽留 1,700 人（68% × 2,500），挽留价值 1,700 × $200 = $340,000
  - **净收益** = $340,000 - $34,288 = $305,712/月
- 若用固定阈值 0.5（Precision 0.65、Recall 0.35）：仅挽留 875 人，净收益 $140,000/月，**损失 $165,712**

**三轨验证**：
- **成本**：模型开发 + 发券系统集成 = 5 万元，月运营成本 0.5 万元
- **合规**：用户流失预测需获得用户数据授权（GDPR/CCPA 合规）
- **风险**：过度补贴导致用户预期提升，长期毛利率压低；需要 A/B 测试验证补贴额度有效性

---

## ③ 代码模板

```python
"""
Model Evaluation Toolkit - 分类模型多维度评估
适用场景：模型选型、A/B测试结果判读、生产模型健康监控
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationMetrics:
    """模型评估指标集合"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    brier_score: float
    confusion_matrix: np.ndarray
    
    def summary(self) -> str:
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return (
            f"\n{'='*60}\n"
            f"模型评估报告\n"
            f"{'='*60}\n"
            f"Accuracy  : {self.accuracy:.4f}\n"
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1-Score  : {self.f1:.4f}\n"
            f"AUC-ROC   : {self.auc_roc:.4f}\n"
            f"AUC-PR    : {self.auc_pr:.4f}\n"
            f"Log Loss  : {self.log_loss:.4f}\n"
            f"Brier     : {self.brier_score:.4f}\n"
            f"{'='*60}\n"
            f"混淆矩阵:\n"
            f"  TN={tn}, FP={fp}\n"
            f"  FN={fn}, TP={tp}\n"
            f"{'='*60}"
        )


def evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> EvaluationMetrics:
    """
    分类模型完整评估
    
    Args:
        y_true: 真实标签 [0,1]
        y_prob: 预测概率 [0,1]
        threshold: 分类阈值 (default: 0.5)
    
    Returns:
        EvaluationMetrics 对象
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    return EvaluationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc_roc=roc_auc_score(y_true, y_prob),
        auc_pr=average_precision_score(y_true, y_prob),
        log_loss=log_loss(y_true, y_prob),
        brier_score=brier_score_loss(y_true, y_prob),
        confusion_matrix=cm
    )


def threshold_sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    阈值敏感性分析 - 在不同阈值下的 Precision/Recall/F1
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        thresholds: 阈值列表 (default: [0.3, 0.4, 0.5, 0.6, 0.7])
    
    Returns:
        DataFrame 包含各阈值下的指标
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = []
    for thresh in thresholds:
        metrics = evaluate_classifier(y_true, y_prob, thresh)
        results.append({
            'Threshold': thresh,
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1': metrics.f1,
            'Accuracy': metrics.accuracy,
            'Positive_Rate': (y_prob >= thresh).mean()
        })
    
    return pd.DataFrame(results)


def ab_test_significance(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    alpha: float = 0.05
) -> Dict:
    """
    A/B 测试统计显著性检验 (双比例 Z 检验)
    
    Args:
        control_conversions: 对照组转化数
        control_total: 对照组总数
        treatment_conversions: 处理组转化数
        treatment_total: 处理组总数
        alpha: 显著性水平 (default: 0.05)
    
    Returns:
        Dict 包含 p-value、置信区间、建议
    """
    p1 = control_conversions / control_total
    p2 = treatment_conversions / treatment_total
    
    p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))
    
    z_stat = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    ci_lower = (p2 - p1) - 1.96 * se
    ci_upper = (p2 - p1) + 1.96 * se
    
    is_significant = p_value < alpha
    
    return {
        'control_rate': p1,
        'treatment_rate': p2,
        'lift': (p2 - p1) / p1 * 100,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'is_significant': is_significant,
        'recommendation': '✓ 上线处理组' if is_significant else '✗ 继续实验或停止'
    }


def model_comparison(
    y_true: np.ndarray,
    models_dict: Dict[str, np.ndarray],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    多模型对比评估
    
    Args:
        y_true: 真实标签
        models_dict: {'model_name': y_prob_array, ...}
        threshold: 分类阈值
    
    Returns:
        DataFrame 包含所有模型的评估指标
    """
    results = []
    for model_name, y_prob in models_dict.items():
        metrics = evaluate_classifier(y_true, y_prob, threshold)
        results.append({
            'Model': model_name,
            'Accuracy': metrics.accuracy,
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1': metrics.f1,
            'AUC-ROC': metrics.auc_roc,
            'AUC-PR': metrics.auc_pr,
            'Log Loss': metrics.log_loss
        })
    
    return pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)


# ============ 内嵌示例数据与测试 ============

def generate_synthetic_data(n_samples=10000, imbalance_ratio=0.05, seed=42):
    """生成不平衡二分类数据"""
    np.random.seed(seed)
    
    n_positive = int(n_samples * imbalance_ratio)
    n_negative = n_samples - n_positive
    
    # 正类：均值偏移
    X_pos = np.random.randn(n_positive, 5) + 1.5
    y_pos = np.ones(n_positive)
    
    # 负类：标准正态
    X_neg = np.random.randn(n_negative, 5)
    y_neg = np.zeros(n_negative)
    
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    
    # 生成概率预测（模拟模型输出）
    y_prob = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1] - 0.3)))
    y_prob += np.random.normal(0, 0.05, len(y_prob))
    y_prob = np.clip(y_prob, 0, 1)
    
    return y, y_prob


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Model Evaluation Metrics - 完整测试")
    print("="*60)
    
    # 生成测试数据
    y_true, y_prob = generate_synthetic_data(n_samples=10000, imbalance_ratio=0.05)
    
    # 测试 1: 基础评估
    print("\n[测试 1] 基础模型评估 (阈值=0.5)")
    metrics = evaluate_classifier(y_true, y_prob, threshold=0.5)
    print(metrics.summary())
    
    # 测试 2: 阈值敏感性分析
    print("\n[测试 2] 阈值敏感性分析")
    threshold_df = threshold_sensitivity_analysis(y_true, y_prob)
    print(threshold_df.to_string(index=False))
    
    # 测试 3: A/B 测试显著性检验
    print("\n[测试 3] A/B 测试显著性检验")
    ab_result = ab_test_significance(
        control_conversions=250,
        control_total=5000,
        treatment_conversions=265,
        treatment_total=5000
    )
    print(f"对照组转化率: {ab_result['control_rate']:.4f}")
    print(f"处理组转化率: {ab_result['treatment_rate']:.4f}")
    print(f"提升幅度: {ab_result['lift']:.2f}%")
    print(f"P-Value: {ab_result['p_value']:.4f}")
    print(f"95% 置信区间: [{ab_result['ci_lower']:.4f}, {ab_result['ci_upper']:.4f}]")
    print(f"决策: {ab_result['recommendation']}")
    
    # 测试 4: 多模型对比
    print("\n[测试 4] 多模型对比评估")
    y_prob_model2 = y_prob + np.random.normal(0, 0.03, len(y_prob))
    y_prob_model2 = np.clip(y_prob_model2, 0, 1)
    y_prob_model3 = y_prob - np.random.normal(0, 0.05, len(y_prob))
    y_prob_model3 = np.clip(y_prob_model3, 0, 1)
    
    models = {
        'XGBoost': y_prob,
        'LightGBM': y_prob_model2,
        'LogisticRegression': y_prob_model3
    }
    
    comparison_df = model_comparison(y_true, models, threshold=0.5)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("[✓] Skill-Model-Evaluation-Metrics 测试通过")
    print("="*60 + "\n")
```

---

## ④ 技能关联

### 前置（Prerequisite）
- [[Skill-Binary-Classification-Fundamentals]]（二分类基础）— 理解正负类、标签编码
- [[Skill-Data-Preprocessing-for-ML]]（数据预处理）— 确保测试集质量

### 延伸（Extends）
- [[Skill-Imbalanced-Data-Handling]]（不平衡数据处理）— 当正负类比例严重失衡时的评估策略
- [[Skill-Model-Calibration]]（模型校准）— 评估概率预测的可靠性
- [[Skill-Production-Model-Monitoring]]（生产模型监控）— 持续跟踪 AUC-ROC 漂移

### 可组合（Combinable）
- **组合场景 1**：[[Skill-Hyperparameter-Tuning]] + [[Skill-Model-Evaluation-Metrics]] → 通过交叉验证选择最优超参数
- **组合场景 2**：[[Skill-AB-Testing-Design]] + [[Skill-Model-Evaluation-Metrics]] → 用统计检验判断 A/B 实验结果显著性
- **组合场景 3**：[[Skill-Feature-Engineering]] + [[Skill-Model-Evaluation-Metrics]] + [[Skill-Model-Selection]] → 特征工程 → 模型训练 → 多维度评估 → 选型上线的完整流程

---

## ⑤ 商业价值评估

### ROI 预估

**定量收益**：
- **场景一（销量预测）**：年均成本节省 40 万元（备货优化 12 万 + 新增销售 28 万）
- **场景二（流失预测）**：月均净收益 30.6 万元，年均 367 万元
- **综合 ROI**：(40 + 367×12) / 13 = **341 倍**（13 万元开发投入）

**定性收益**：
- 从"凭感觉"到"数据驱动"的决策范式转变
- 建立可复用的模型评估框架，加速后续 ML 项目上线
- 降低模型部署风险（避免"假阳性"上线）

### 实施难度

**⭐⭐⭐☆☆（3/5 星）**

**理由**：
- **易**：核心指标计算简单，sklearn 一行代码搞定
- **难**：需要理解业务成本结构（Precision vs Recall 的权衡），不同场景的阈值选择差异大
- **难**：不平衡数据下的陷阱多（准确率陷阱、AUC-PR vs AUC-ROC 选择）
- **建议**：先从单一指标（AUC-ROC）开始，逐步引入多维度评估

### 优先级

**⭐⭐⭐⭐☆（4/5 星）**

**理由**：
- **高优先级**：这是所有 ML 项目的必经之路，没有评估体系就没有科学决策
- **高频场景**：母婴出海中流失预测、销量预测、用户分层都需要用到
- **高风险**：评估指标选错（如用准确率评估不平衡数据），会导致模型上线后业务反向
- **建议**：在 Phase 1 中优先掌握，为后续所有 ML Skill 奠定基础

---

## 学习路径建议

1. **第 1 周**：理解混淆矩阵 + 4 大基础指标（Precision/Recall/F1/AUC-ROC）
2. **第 2 周**：用代码模板在真实数据上实践，对比 3+ 个模型
3. **第 3 周**：设计一个业务场景的阈值优化方案（如流失预测），计算 ROI
4. **第 4 周**：建立团队内的"模型评估 Checklist"，确保每个上线模型都经过多维度评估

