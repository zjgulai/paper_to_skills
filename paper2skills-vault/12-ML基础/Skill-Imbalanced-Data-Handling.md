# Skill Card: Imbalanced Data Handling（不平衡数据处理）

roadmap_phase: phase1
---

## ① 算法原理

### 核心思想
解决"**少数类样本太少，模型学不到东西**"的问题。在母婴电商中，流失（5%）、欺诈（<1%）、高价值转化（3%）都是典型不平衡场景。常规模型在不平衡数据上会偏向多数类，导致 Recall 极低。

### 数学直觉

**三大处理路线**：

1. **重采样 (Resampling)**：
   - **SMOTE**：在少数类样本之间线性插值生成合成样本。对每个少数类样本 $x_i$，随机选一个 k-近邻 $x_j$，生成新样本 $x_{new} = x_i + \lambda(x_j - x_i), \lambda \sim U(0,1)$
   - **欠采样 (Undersampling)**：随机丢弃多数类样本——简单但丢失信息
   - **混合 (SMOTEENN/SMOTETomek)**：先 SMOTE 过采样，再用 Edited Nearest Neighbors 清理噪声

2. **类别权重 (Class Weight)**：
   - 给少数类更高的误分类惩罚：$w_{pos} = \frac{n_{neg}}{n_{pos}}$。XGBoost 的 `scale_pos_weight` 和 sklearn 的 `class_weight='balanced'` 都基于此
   - 不需要修改数据，只需修改损失函数——计算效率高

3. **阈值调优 (Threshold Tuning)**：
   - 默认阈值 0.5 在不平衡场景下不合理。通过 Precision-Recall 曲线找到满足业务约束的最优阈值

### 关键假设
- SMOTE 假设特征空间连续（离散特征慎用）
- 类别权重方法不改变数据分布，但在极端不平衡（<0.1%）时效果有限
- 所有方法都不是银弹——最终需要模型评估来验证效果

---

## ② 母婴出海应用案例

### 场景一：流失预警模型的不平衡处理

**业务问题**：流失率 5%，直接用 XGBoost 训练，Recall 只有 0.3——70% 的流失用户没被识别。需要至少 0.8 Recall 才能触发有效的挽留策略。

**数据要求**：100,000 条用户特征 + 流失标签。需对比 4 种方案：Baseline / SMOTE / Class Weight / SMOTE + Tomek Links

**预期产出**：
- Baseline: Recall=0.30, Precision=0.45
- SMOTE: Recall=0.75, Precision=0.35
- Class Weight: Recall=0.78, Precision=0.38 ← **最优**
- SMOTE+Tomek: Recall=0.72, Precision=0.33

**业务价值**：Recall 从 0.3 → 0.78，多识别 720 个流失用户（1500 基准），每位挽留价值 $200 → 月增 $144,000

### 场景二：广告虚假点击检测的极端不平衡

**业务问题**：正常点击 99.5% vs 虚假点击 0.5%。不做不平衡处理直接训练 Isolation Forest，实际假阳性率 30%+，导致大量正常流量被误拦截。

**数据要求**：200,000 条点击日志。先人工标注 200 条虚假点击，用 SMOTE 扩展到 2000 训练集

**预期产出**：SMOTE + Cost-Sensitive Learning 后 Precision=0.65，Recall=0.82。误拦截率从 30% 降至 8%

**业务价值**：降低误拦截率 22pp → 每月减少 132,000 条误拦截 → 按 CTR 2% 和 CVR 5% 估算，挽回约 130 单/月 → $15,000/月

---

## ③ 代码模板

```python
"""
Imbalanced Data Handling Toolkit
不平衡数据处理工具集 — 重采样 / 类别权重 / 阈值调优

适用场景：流失预测、欺诈检测、稀有事件预警
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from typing import Dict, Tuple, List


def compare_imbalance_strategies(
    X: np.ndarray,
    y: np.ndarray,
    model=None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    对比多种不平衡处理策略的效果
    
    Args:
        X: 特征矩阵
        y: 标签 (0/1)
        model: 基础模型（默认 XGBoost）
    
    Returns:
        DataFrame: 各策略的 Recall/Precision/F1/AUC 对比
    """
    model = model or XGBClassifier(random_state=42, eval_metric='logloss')
    pos_ratio = y.mean()
    
    strategies = {
        'Baseline (No Treatment)': None,
        'Class Weight (balanced)': 'balanced',
        'SMOTE': SMOTE(random_state=42),
        'Borderline SMOTE': BorderlineSMOTE(random_state=42),
        'SMOTE + Tomek Links': SMOTETomek(random_state=42),
        'SMOTE + ENN': SMOTEENN(random_state=42),
    }
    
    results = []
    for name, strategy in strategies.items():
        X_res, y_res = X, y
        
        if strategy == 'balanced':
            m = XGBClassifier(random_state=42, scale_pos_weight=(1-pos_ratio)/pos_ratio, 
                              eval_metric='logloss')
        elif strategy is not None:
            X_res, y_res = strategy.fit_resample(X, y)
            m = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            m = XGBClassifier(random_state=42, eval_metric='logloss')
        
        m.fit(X_res, y_res)
        y_prob = m.predict_proba(X)[:, 1]
        y_pred = m.predict(X)
        
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        results.append({
            'Strategy': name,
            'Samples': len(y_res),
            'Recall': round(report['1']['recall'], 3),
            'Precision': round(report['1']['precision'], 3),
            'F1': round(report['1']['f1-score'], 3),
            'AUC': round(roc_auc_score(y, y_prob), 3),
        })
        if verbose:
            print(f"  {name}: Recall={results[-1]['Recall']:.3f}, "
                  f"Precision={results[-1]['Precision']:.3f}, AUC={results[-1]['AUC']:.3f}")
    
    return pd.DataFrame(results)


def find_business_threshold(
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    min_precision: float = 0.3,
    max_fpr: float = 0.1
) -> Tuple[float, Dict]:
    """
    在业务约束下找最优阈值
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        min_precision: 最低 Precision（确保 ROI 为正）
        max_fpr: 最高误报率（控制打扰用户数）
    
    Returns:
        (optimal_threshold, metrics_dict)
    """
    from sklearn.metrics import confusion_matrix
    
    pos_count = y_true.sum()
    
    for t in np.linspace(0.99, 0.01, 99):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / pos_count
        fpr = fp / tn if tn > 0 else 0
        
        if precision >= min_precision and fpr <= max_fpr:
            return t, {'threshold': t, 'precision': precision, 
                       'recall': recall, 'fpr': fpr, 'tp': tp, 'fp': fp}
    
    # 找不到满足所有约束的 → 放宽精度要求
    return find_business_threshold(y_true, y_prob, min_precision - 0.05, max_fpr)


# ============ 测试 ============

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    np.random.seed(42)
    n, n_features = 5000, 10
    
    # 模拟流失数据（5% 正类）
    X = np.random.randn(n, n_features)
    y = (np.random.random(n) < 0.05).astype(int)
    
    print(f"数据: {n} 条 | 正类={y.sum()} ({y.mean():.1%})")
    print(f"对比不同不平衡策略:")
    
    df = compare_imbalance_strategies(X, y)
    print(f"\n{df.to_string(index=False)}")
    
    # 最优阈值
    m = XGBClassifier(scale_pos_weight=19, random_state=42, eval_metric='logloss')
    m.fit(X, y)
    y_prob = m.predict_proba(X)[:, 1]
    threshold, metrics = find_business_threshold(y, y_prob, min_precision=0.25)
    print(f"\n最优阈值: {threshold:.2f} | Recall={metrics['recall']:.3f} | Precision={metrics['precision']:.3f}")
    
    print("\n[✓] 不平衡处理测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Model-Evaluation-Metrics]]、[[Skill-Cross-Validation-Strategies]] — 评估是判断不平衡处理有效性的前提
- **延伸技能**：[[Skill-Ensemble-Methods]]（集成对不平衡天然友好）、[[Skill-Hyperparameter-Optimization]]（scale_pos_weight 等参数的调优）
- **可组合**：
  - **[[Skill-Customer-Churn-Prediction]]** — 流失预测是典型不平衡场景
  - **[[Skill-Review-Fraud-Detection]]** — 欺诈检测的极端不平衡
  - **[[Skill-Feature-Engineering]]** — 好的特征比任何重采样方法都重要

---
- **相关技能**：[[Skill-Feature-Selection]]
- **跨域关联**：[[Skill-Guardrailed-Uplift-Targeting]]

## ⑤ 商业价值评估

- **ROI 预估**：流失预测场景 Recall 从 0.3 提升到 0.78，月增收 $144,000；欺诈检测场景减少误拦截挽回 $15,000/月。年化贡献 **150-300 万元**。
- **实施难度**：⭐⭐☆☆☆（2 星）— imbalanced-learn + sklearn API 一致，但需要反复实验找最优组合
- **优先级评分**：⭐⭐⭐⭐⭐（5 星）— 母婴电商中流失、欺诈、稀有事件场景极多，是模型落地的核心瓶颈
- **评估依据**：Model Evaluation + CV 确保你"知道模型不好"，Imbalanced Data Handling 让你"能做点什么来改善"。三者构成 ML 基础的质量闭环
