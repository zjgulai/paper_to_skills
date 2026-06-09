# Skill Card: Cross-Validation Strategies（交叉验证策略）

---

## ① 算法原理

### 核心思想
交叉验证解决"**模型在未知数据上表现如何**"的问题——不是依赖一次 train/test split 的"运气"，而是通过多次切分平均来获得稳健的泛化能力估计。

### 数学直觉

**K-Fold CV**：将数据等分 K 份，每次用 K-1 份训练、1 份验证，重复 K 次取平均。

$$\text{CV Score} = \frac{1}{K} \sum_{i=1}^{K} \text{Score}(\text{Model trained on folds } \neq i, \text{ fold } i)$$

**为什么需要变体**：
- **Stratified K-Fold**：保持每折中类别比例一致。不平衡数据（如流失率 5%）下随机 K-Fold 可能导致某折一个正样本都没有
- **TimeSeries Split**：时序数据不能用随机切分（会用到未来信息"预测"过去）。必须保证训练集时间 < 验证集时间
- **Group K-Fold**：同一用户的多条记录不能在训练集和验证集间泄漏。按 group（如 user_id）切分
- **Nested CV**：外层 CV 估计泛化误差，内层 CV 做超参选择——避免超参调优时的过拟合

### 关键假设
- 数据独立同分布（i.i.d.）— 非 i.i.d. 场景（时序/分组/空间相关）必须用对应的 CV 变体
- K 的选择：K=5 或 K=10 是经验平衡点。K 越大偏差越小但方差越大

---

## ② 母婴出海应用案例

### 场景一：吸奶器销量预测模型的时序验证

**业务问题**：我们用过去 24 个月数据训练 Prophet 预测下月销量。用随机 K-Fold 会导致"用 12 月数据预测 6 月销量"的荒谬情况——模型看到了未来信息。

**数据要求**：24 个月 × 30 SKU 的日销量数据，需按月份做 TimeSeries Split

**预期产出**：正确的时序 CV 评估——3 折滚动验证（每次用前 18 个月训、后 3 个月验），MAPE 稳定在 15-20%

**业务价值**：避免因错误验证方式导致的高估（随机 CV 可能低报 MAPE 50%），正确评估可节省因预测偏差造成的库存积压损失约 10-20 万/月

### 场景二：新客首单预测模型的用户级 Group CV

**业务问题**：训练集包含同一用户的多次会话记录。随机切分会将同一用户的部分会话放入训练集、部分放入验证集——导致数据泄漏，验证 AUC 虚高 0.05-0.10。

**数据要求**：50,000 条会话记录，归属 12,000 个用户。Group K-Fold 按 user_id 分组

**预期产出**：GroupCV 评估的 AUC 通常比随机 CV 低 0.05-0.10，但更接近上线真实表现。避免模型选型错误

**业务价值**：避免因验证乐观而上线表现差的模型，减少因模型失效导致的营销预算浪费（每次误判约 3-5 万）

---

## ③ 代码模板

```python
"""
Cross-Validation Strategies Toolkit
适用场景：模型稳健评估、时序验证、分组数据防泄漏
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate
)
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CVResult:
    """交叉验证结果"""
    strategy: str
    fold_scores: List[float]
    
    @property
    def mean(self) -> float:
        return np.mean(self.fold_scores)
    
    @property
    def std(self) -> float:
        return np.std(self.fold_scores)
    
    @property
    def ci95(self) -> Tuple[float, float]:
        se = self.std / np.sqrt(len(self.fold_scores))
        return (self.mean - 1.96 * se, self.mean + 1.96 * se)
    
    def summary(self) -> str:
        return f"{self.strategy}: {self.mean:.4f} ± {self.std:.4f} [95%CI: {self.ci95[0]:.4f}, {self.ci95[1]:.4f}]"


def select_cv_strategy(
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str = 'classification',
    groups: Optional[np.ndarray] = None,
    is_time_series: bool = False,
    n_splits: int = 5
) -> object:
    """
    根据数据特征自动选择交叉验证策略
    
    Args:
        X: 特征矩阵
        y: 标签
        problem_type: 'classification' | 'regression'
        groups: 分组标签（如 user_id），用于 GroupKFold
        is_time_series: 是否为时序数据
        n_splits: 折数
    """
    if is_time_series:
        return TimeSeriesSplit(n_splits=n_splits)
    elif groups is not None:
        return GroupKFold(n_splits=n_splits)
    elif problem_type == 'classification' and len(np.unique(y)) > 1:
        # 检查不平衡程度决定是否用 Stratified
        pos_ratio = y.mean()
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def compare_cv_strategies(
    X: np.ndarray,
    y: np.ndarray,
    model,
    groups: Optional[np.ndarray] = None,
    scoring: str = 'roc_auc',
    n_splits: int = 5
) -> List[CVResult]:
    """
    对比不同 CV 策略在同一数据上的表现差异
    
    Returns:
        各策略的 CVResult 列表 — 差异大说明存在数据泄漏
    """
    strategies = {
        'Standard-KFold': KFold(n_splits=n_splits, shuffle=True, random_state=42),
        'Stratified-KFold': StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
    }
    if groups is not None:
        strategies['Group-KFold'] = GroupKFold(n_splits=n_splits)
        strategies['TimeSeries-Split'] = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    for name, cv in strategies.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results.append(CVResult(strategy=name, fold_scores=list(scores)))
        except Exception as e:
            print(f"  [{name}] failed: {e}")
    
    return results


def nested_cv_hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: List[Dict],
    inner_cv=None,
    outer_cv=None,
    n_outer: int = 5,
    n_inner: int = 3
) -> Tuple[float, float]:
    """
    Nested CV：外层评估泛化误差，内层做超参选择
    
    防止超参调优时的过拟合——如果只用单层 CV 同时做选择和评估，
    选出的最优参数会过度拟合验证集
    
    Returns:
        (mean_score, std_score) — 无偏泛化误差估计
    """
    from sklearn.model_selection import GridSearchCV
    
    outer_cv = outer_cv or StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    inner_cv = inner_cv or StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
    
    outer_scores = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        inner_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=inner_cv, scoring='roc_auc'
        )
        inner_search.fit(X_train, y_train)
        
        best_model = inner_search.best_estimator_
        test_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        outer_scores.append(test_score)
    
    return float(np.mean(outer_scores)), float(np.std(outer_scores))


# ============ 测试 ============

if __name__ == '__main__':
    np.random.seed(42)
    
    # 模拟母婴流失预测数据
    n = 5000
    X = np.random.randn(n, 10)
    y = (np.random.random(n) < 0.05).astype(int)
    groups = np.repeat(range(1000), 5)[:n]  # 每用户 5 条记录
    
    # 自动策略选择
    cv = select_cv_strategy(X, y, is_time_series=False, groups=None)
    print(f"自动选择: {type(cv).__name__}")
    
    # 对比不同策略
    results = compare_cv_strategies(
        X, y, 
        model=RandomForestClassifier(n_estimators=50, random_state=42),
        groups=groups
    )
    for r in results:
        print(f"  {r.summary()}")
    
    # 检查泄漏
    if len(results) >= 2:
        diff = results[0].mean - results[-1].mean if len(results) > 1 else 0
        if abs(diff) > 0.03:
            print(f"\n⚠️ 策略间差异 {diff:.3f} > 0.03 — 可能存在数据泄漏!")
        else:
            print(f"\n✓ 策略间差异 {diff:.3f} ≤ 0.03 — 无明显泄漏")
    
    print("\n[✓] 交叉验证策略测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Model-Evaluation-Metrics]] — 理解评估指标后再学习如何稳健评估
- **延伸技能**：[[Skill-Hyperparameter-Optimization]]（Nested CV 是超参调优的正确做法）、[[Skill-Imbalanced-Data-Handling]]（不平衡数据的 CV 策略选择）
- **可组合**：
  - **[[Skill-AB-Experimental-Design]]** — 交叉验证的方差估计与 A/B 检验的置信区间计算共享统计基础
  - **[[Skill-Time-Series-Forecasting]]** — 时序 CV 是预测模型的必备验证手段
  - **[[Skill-Customer-Churn-Prediction]]** — 流失预测的评估必须用 StratifiedCV + 时序 CV 双重验证

---
- **相关技能**：[[Skill-Ensemble-Methods]]
- **相关技能**：[[Skill-Model-Performance-Monitor]]
- **跨域关联**：[[Skill-AIGP-LLM-Dynamic-Pricing]]
- **跨域关联**：[[Skill-Conformal-Prediction-Demand-UQ]]
- **关联**：[[Skill-Category-Compliance-Prescan]]

## ⑤ 商业价值评估

- **ROI 预估**：正确的 CV 策略可避免数据泄漏导致的模型虚高评估。母婴场景下，一次错误的模型选型导致上线后表现差，营销预算浪费 3-5 万/次。年化避免损失 **30-80 万元**。
- **实施难度**：⭐⭐☆☆☆（2 星）— scikit-learn 原生支持，仅需理解业务数据特征后选择对应策略
- **优先级评分**：⭐⭐⭐⭐☆（4 星）— 所有模型的正确评估前提，ML 基础层第二核心
- **评估依据**：Model Evaluation 教你"看什么指标"，Cross Validation 教你"怎么看才对"。二者互补。不平衡数据 + 时序数据场景在母婴电商中极为常见（流失率 5%、月度/季度周期性），必须掌握对应 CV 策略
