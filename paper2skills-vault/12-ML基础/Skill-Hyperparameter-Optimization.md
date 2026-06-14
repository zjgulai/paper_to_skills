# Skill Card: Hyperparameter Optimization（超参调优）

roadmap_phase: phase1
---

## ① 算法原理

### 核心思想
**模型有一堆"旋钮"（超参），拧对了性能飞跃，拧错了白费算力。** 超参调优系统化地搜索最优超参组合，而非手动试错。

### 数学直觉

**三种搜索策略**：

1. **Grid Search**：遍历所有指定组合。保证找到"网格内"最优，但维度灾难——10 个参数各 3 个值 = $3^{10}=59049$ 次训练。

2. **Random Search**：随机采样。Bergstra & Bengio (2012) 证明在高维空间中，Random Search 比 Grid Search 更高效——因为大部分超参对结果影响不大，Grid Search 浪费了大量计算在无关维度上。

3. **Bayesian Optimization (BO)**：用概率模型（高斯过程 / TPE）建模"超参→性能"的映射，每次选择"最可能提升"的下一组超参。
   - 采集函数 (Acquisition Function) 平衡探索与利用：
     - Expected Improvement: $EI(x) = E[\max(f(x) - f^*, 0)]$
   - **Optuna** 是基于 TPE (Tree-structured Parzen Estimator) 的 BO 实现，支持剪枝（早停表现差的 trial）

4. **Hyperband**：多臂老虎机思想——给有潜力的配置更多资源（epochs），差的及早停止。结合 BO → BOHB

### 关键假设
- BO 假设超参空间平滑（相近的超参→相近的性能），对离散超参效果差
- 早停(Pruning) 假设学习曲线单调——对震荡剧烈的 loss 不适用
- 超参调优不是银弹——数据质量 > 特征工程 > 超参调优

---

## ② 母婴出海应用案例

### 场景一：XGBoost 流失预测的超参调优

**业务问题**：默认参数的 XGBoost 流失预测 AUC=0.78，希望通过超参调优提升到 0.82+。关键超参：`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `scale_pos_weight`。

**数据要求**：100K 数据，在 10K 子集上快速迭代（Optuna 50 trials），确认方向后再在全集上精调

**预期产出**：
- 最优配置：max_depth=6, lr=0.05, n_estimators=300, subsample=0.8
- AUC 从 0.78 → 0.83，+0.05
- 最佳 trial 在 50 轮内找到（vs Grid Search 需 729 轮）

**业务价值**：AUC +0.05 = Recall +6pp → 多挽留 90 用户/月 → +$18,000/月

### 场景二：LightGBM 广告预测的 Hyperband 加速

**业务问题**：完整训练一次 LightGBM 需要 20 分钟。Grid Search 100 组合 = 33 小时。需要用 Hyperband 在 5 小时内找到近似最优配置。

**数据要求**：50K 样本，LightGBM 10 个主要超参

**预期产出**：Hyperband 5 小时内完成 100 trials（90 个早停 + 10 个完整训练），找到 AUC=0.81 配置（Grid Search 的 0.815 差距可接受）

**业务价值**：调优时间从 33 小时缩减到 5 小时（6x 加速），支持每日迭代优化

---

## ③ 代码模板

```python
"""
Hyperparameter Optimization Toolkit
超参调优工具集 — Grid / Random / Bayesian (Optuna) / Hyperband
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from typing import Dict, Tuple


def grid_search_xgboost(
    X: np.ndarray, y: np.ndarray,
    cv: int = 3, n_jobs: int = -1
) -> Tuple[object, pd.DataFrame]:
    """XGBoost Grid Search（小空间适用）"""
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9],
    }
    
    gs = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid, cv=cv, scoring='roc_auc', n_jobs=n_jobs, verbose=0
    )
    gs.fit(X, y)
    
    results = pd.DataFrame(gs.cv_results_)
    results = results.sort_values('rank_test_score')
    
    return gs.best_estimator_, results[['params', 'mean_test_score', 'std_test_score']].head(5)


def random_search_xgboost(
    X: np.ndarray, y: np.ndarray,
    n_iter: int = 50, cv: int = 3, n_jobs: int = -1
) -> Tuple[object, pd.DataFrame]:
    """XGBoost Random Search（中等空间适用）"""
    from scipy.stats import uniform, randint
    
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'n_estimators': randint(100, 500),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
    }
    
    rs = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_dist, n_iter=n_iter, cv=cv, scoring='roc_auc',
        n_jobs=n_jobs, random_state=42
    )
    rs.fit(X, y)
    
    return rs.best_estimator_, pd.DataFrame({
        'rank': range(1, 6),
        'mean_score': sorted(rs.cv_results_['mean_test_score'], reverse=True)[:5]
    })


def optuna_optimize(
    X: np.ndarray, y: np.ndarray,
    n_trials: int = 50, cv_folds: int = 3,
    timeout: int = 600
) -> Dict:
    """
    Optuna Bayesian Optimization（推荐）
    
    pip install optuna
    """
    try:
        import optuna
    except ImportError:
        print("[ERROR] pip install optuna")
        return {}
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0, log=True),
        }
        
        model = XGBClassifier(**params, random_state=42, 
                               eval_metric='logloss', use_label_encoder=False)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_prob = model.predict_proba(X[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], y_prob))
        
        return np.mean(scores)
    
    # Optuna study with pruning
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }


# ============ 测试 ============

if __name__ == '__main__':
    np.random.seed(42)
    n, n_features = 2000, 10
    X = np.random.randn(n, n_features)
    y = ((X[:, :3].sum(axis=1) + np.random.randn(n)) > 0).astype(int)
    
    # Baseline
    base = XGBClassifier(random_state=42, eval_metric='logloss')
    base.fit(X, y)
    baseline_auc = roc_auc_score(y, base.predict_proba(X)[:, 1])
    print(f"Baseline AUC: {baseline_auc:.3f}")
    
    # Random Search（快速）
    _, rs_results = random_search_xgboost(X, y, n_iter=15, cv=3)
    best_random = rs_results.iloc[0]
    print(f"Random Search best AUC: {best_random['mean_score']:.3f}")
    
    # Optuna
    opt_result = optuna_optimize(X, y, n_trials=20)
    if opt_result:
        print(f"Optuna best AUC: {opt_result['best_score']:.3f}")
        print(f"  trials={opt_result['n_trials']}, pruned={opt_result.get('pruned_trials', 0)}")
    
    print("\n[✓] 超参调优测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Model-Evaluation-Metrics]]、[[Skill-Cross-Validation-Strategies]]（Nested CV 是超参调优的正确做法）
- **延伸技能**：[[Skill-Ensemble-Methods]]（调优后的单模型作为 Stacking 基学习器）
- **可组合**：
  - **[[Skill-Customer-Churn-Prediction]]** — 流失模型超参敏感度高
  - **[[Skill-Imbalanced-Data-Handling]]** — `scale_pos_weight` 是核心调优超参
  - **[[Skill-Feature-Selection]]** — 特征选择后的精简模型调优更快收敛

---

## ⑤ 商业价值评估

- **ROI 预估**：典型场景下超参调优可提升 AUC 3-5pp（尤其是默认参数在高维/不平衡数据上表现差）。流失预测 AUC +0.05 = 月增收 $18,000。年化贡献 **50-100 万元**。
- **实施难度**：⭐⭐⭐☆☆（3 星）— Optuna API 友好，但需要理解超参含义和搜索策略
- **优先级评分**：⭐⭐⭐⭐☆（4 星）— 是所有模型训练的"最后一步"，投入产出比极高（Optuna 50 trials 通常只需 30 分钟）
- **评估依据**：ML 基础层 6 张卡片的最后一张，形成完整闭环：评估 → 交叉验证 → 不平衡处理 → 集成 → 特征选择 → 超参调优
