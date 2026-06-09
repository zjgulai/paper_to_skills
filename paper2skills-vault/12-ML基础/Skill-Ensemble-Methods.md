# Skill Card: Ensemble Methods（集成学习方法）

---

## ① 算法原理

### 核心思想
**三个臭皮匠胜过诸葛亮**——将多个弱学习器组合成强学习器，通过"群体智慧"降低偏差（Boosting）或方差（Bagging），获得比任何单一模型更好的预测性能。

### 数学直觉

**三大集成范式**：

1. **Bagging（并行）**：Bootstrap 采样生成 D 个训练子集，独立训练 D 个模型，投票/平均。核心公式——方差缩减：
   $$\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{D}\sigma^2$$
   其中 $\rho$ 是模型间相关性。Bagging 通过 Bootstrap 降低 $\rho$，从而降低总体方差。**代表：Random Forest**

2. **Boosting（串行）**：序贯训练，每个新模型专注于前一个模型的错误。核心——梯度提升的加法模型：
   $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
   其中 $h_m(x)$ 拟合的是前一轮的负梯度（残差），$\eta$ 是学习率控制步长。**代表：XGBoost, LightGBM, CatBoost**

3. **Stacking（元学习）**：用第一层多个异构模型（RF + XGBoost + LR）的输出作为特征，训练第二层元模型（通常用简单 LR）做最终预测。核心直觉：不同模型在不同数据子空间各有优势，元模型学会"何时信任谁"。

### 关键假设
- Bagging 要求基学习器低偏差高方差（如深度决策树），否则无法从集成获益
- Boosting 对噪声敏感——一个错误标签会被后续模型反复"关注"导致过拟合
- Stacking 需要第一层模型有足够的多样性（同质化模型 Stacking 无意义）

---

## ② 母婴出海应用案例

### 场景一：流失预测的多模型 Stacking 集成

**业务问题**：单独用 XGBoost 预测流失 AUC=0.82，单独用 LightGBM AUC=0.81，单独用 Random Forest AUC=0.78。希望进一步提升到 0.85+ 以提升挽留策略精度。

**数据要求**：100K 用户特征 + 流失标签。第一层：XGBoost + LightGBM + CatBoost + Random Forest，第二层：Logistic Regression

**预期产出**：Stacking 集成后 AUC=0.86（+0.04），Recall 在相同 Precision 下提升 5pp。相比单一最优模型可多识别 75 个流失用户/月

**业务价值**：每月额外挽留 $15,000（75 用户 × $200），年化 $180,000。实施成本几乎为零（无需新数据，仅调整训练流程）

### 场景二：广告转化预测的 Blending 策略

**业务问题**：FB/Google/TikTok 三个渠道的广告数据特征空间不同，统一模型在跨渠道表现不一致。需要渠道级模型 + 全局 Blending。

**数据要求**：按渠道拆分训练独立模型，再对预测概率加权平均。权重随渠道历史表现动态调整

**预期产出**：渠道级 XGBoost + 全局 Blending，跨渠道平均 AUC 从 0.78 提升到 0.83

**业务价值**：更精准的广告预算分配，ROAS 预计提升 8-12%。月广告预算 30 万 → 等价增收 $24,000-36,000/月

---

## ③ 代码模板

```python
"""
Ensemble Methods Toolkit
集成学习工具集 — Bagging / Boosting / Stacking / Blending
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, List, Tuple


def stacking_cv(
    X: np.ndarray,
    y: np.ndarray,
    base_models: Dict[str, object],
    meta_model=None,
    n_folds: int = 5,
    random_state: int = 42
) -> Tuple[object, np.ndarray]:
    """
    Stacking with cross-validation (防过拟合)
    
    Step 1: K-Fold 训练每个基模型，收集 out-of-fold 预测
    Step 2: 用 out-of-fold 预测训练元模型
    
    Args:
        X: 特征矩阵
        y: 标签
        base_models: {模型名: 模型实例}
        meta_model: 元模型（默认 LogisticRegression）
        n_folds: 折数
    
    Returns:
        (trained_meta_model, stacking_features) 
    """
    meta_model = meta_model or LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # 存储每个基模型的 out-of-fold 预测
    oof_predictions = np.zeros((len(y), len(base_models)))
    
    # 存储训练好的基模型（用于后续 full fit）
    trained_base_models = {name: [] for name in base_models}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for i, (name, model) in enumerate(base_models.items()):
            m = clone_model(model, random_state + fold)
            m.fit(X_train, y_train)
            oof_predictions[val_idx, i] = m.predict_proba(X_val)[:, 1]
            trained_base_models[name].append(m)
    
    # 训练元模型
    meta_model.fit(oof_predictions, y)
    
    # 为后续使用保留 full-fit 的基模型
    for name in base_models:
        base_models[name] = clone_model(base_models[name], random_state)
        base_models[name].fit(X, y)
    
    return meta_model, oof_predictions


def clone_model(model, random_state: int):
    """深拷贝模型并设置 random_state"""
    import copy
    m = copy.deepcopy(model)
    if hasattr(m, 'random_state'):
        m.random_state = random_state
    if hasattr(m, 'set_params'):
        try:
            m.set_params(random_state=random_state)
        except Exception:
            pass
    return m


def blending_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    models: Dict[str, object],
    weights: Dict[str, float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Blending: 各模型独立训练后加权平均
    
    Args:
        models: {模型名: 模型实例}
        weights: {模型名: 权重}，None 则等权
    
    Returns:
        (blended_predictions, used_weights)
    """
    from sklearn.model_selection import train_test_split
    
    if weights is None:
        # 用 holdout 集动态学习最优权重
        X_tr, X_hold, y_tr, y_hold = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train
        )
        
        hold_preds = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            hold_preds[name] = model.predict_proba(X_hold)[:, 1]
        
        # 简单策略：按 holdout AUC 分配权重
        weights = {}
        total = 0
        for name, y_prob in hold_preds.items():
            w = roc_auc_score(y_hold, y_prob)
            weights[name] = max(w - 0.5, 0.01)  # 低于 0.5 给极小权重
            total += weights[name]
        weights = {k: v/total for k, v in weights.items()}
    
    # Full fit + Blending
    test_preds = np.zeros(len(X_test))
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_preds += weights[name] * model.predict_proba(X_test)[:, 1]
    
    return test_preds, weights


def compare_ensemble_vs_single(
    X: np.ndarray, y: np.ndarray, cv: int = 5
) -> pd.DataFrame:
    """对比单一模型 vs 集成模型的 CV 表现"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }
    
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results.append({'Model': name, 'AUC_Mean': scores.mean(), 'AUC_Std': scores.std()})
    
    # Simple voting ensemble
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    from sklearn.ensemble import VotingClassifier
    voting = VotingClassifier(
        [('rf', rf), ('xgb', xgb), ('lgbm', LGBMClassifier(random_state=42, verbose=-1))],
        voting='soft'
    )
    scores = cross_val_score(voting, X, y, cv=cv, scoring='roc_auc')
    results.append({'Model': 'Soft Voting Ensemble', 'AUC_Mean': scores.mean(), 'AUC_Std': scores.std()})
    
    return pd.DataFrame(results)


# ============ 测试 ============

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    np.random.seed(42)
    n, n_features = 5000, 15
    X = np.random.randn(n, n_features)
    # 构建有一些真实信号的数据
    y = ((X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(n) * 2) > 0).astype(int)
    
    print(f"数据: {n} 条 | 正类={y.sum()} ({y.mean():.1%})")
    
    # 单一 vs 集成对比
    df = compare_ensemble_vs_single(X, y)
    print(f"\n单一 vs 集成:\n{df.to_string(index=False)}")
    
    # Stacking
    base = {
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    meta, oof = stacking_cv(X, y, base)
    stacking_score = roc_auc_score(y, meta.predict_proba(oof)[:, 1])
    print(f"\nStacking AUC: {stacking_score:.3f}")
    
    print("\n[✓] 集成学习测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Model-Evaluation-Metrics]]、[[Skill-Cross-Validation-Strategies]]
- **延伸技能**：[[Skill-Hyperparameter-Optimization]]（集成模型超参极多，调优收益高）
- **可组合**：
  - **[[Skill-Customer-Churn-Prediction]]** — Stacking 可显著提升流失预测
  - **[[Skill-Imbalanced-Data-Handling]]** — 不平衡处理 + Boosting 的 scale_pos_weight 协同
  - **[[Skill-ROAS-Budget-Optimization]]** — 广告预测模型几乎必用 XGBoost/LightGBM

---
- **相关技能**：[[Skill-Feature-Selection]]

## ⑤ 商业价值评估

- **ROI 预估**：Stacking 在流失/广告场景可提升 AUC 3-5pp，对应月增收 $40,000-50,000。实施成本低（仅代码集成，无需新数据/新系统）。年化贡献 **40-80 万元**。
- **实施难度**：⭐⭐⭐☆☆（3 星）— 需要理解 Bagging/Boosting/Stacking 的区别和适用场景
- **优先级评分**：⭐⭐⭐⭐☆（4 星）— XGBoost/LightGBM 是 Kaggle 和工业界的事实标准，所有预测模型几乎都会用到
- **评估依据**：ML 基础层的高阶技能。学会了可以让已有模型无痛提升 3-5% 性能
