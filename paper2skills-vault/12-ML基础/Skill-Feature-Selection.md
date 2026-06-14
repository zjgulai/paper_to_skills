# Skill Card: Feature Selection（特征选择）

roadmap_phase: phase1
---

## ① 算法原理

### 核心思想
**少即是多**——从大量特征中筛选出真正有用的子集，提升模型性能、降低过拟合风险、减少训练/推理成本、增强可解释性。

### 数学直觉

**三大流派**：

1. **过滤法 (Filter)**：在训练前，用统计指标独立评估每个特征的重要性。
   - **方差阈值**：方差 ≈ 0 的特征无信息量
   - **互信息 (Mutual Information)**：$I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$ — 衡量特征与标签的非线性相关性
   - **相关系数**：线性相关度量，对非线性关系不敏感

2. **包裹法 (Wrapper)**：用模型性能作为特征子集的评价标准，迭代搜索最优子集。
   - **RFE (Recursive Feature Elimination)**：从全量特征开始，每轮去掉最不重要的特征
   - 计算代价高但效果好——每次迭代都要重新训练模型

3. **嵌入法 (Embedded)**：特征选择融入模型训练过程。
   - **L1 正则化 (Lasso)**：$\min_w \sum_i (y_i - w^T x_i)^2 + \lambda \sum_j |w_j|$ — 弱特征权重被压缩到 0
   - **树模型特征重要性**：基于分裂带来的 impurity reduction 排名
   - **SHAP**：基于博弈论的归因——每个特征对预测的边际贡献。一致性最强

### 关键假设
- Filter 方法忽略特征间交互，可能遗漏"单个无用但组合有用"的特征
- 树模型的特征重要性偏向高基数特征（需用 permutation importance 纠偏）
- SHAP 计算代价高（$O(2^n)$），需用 TreeSHAP 等近似算法

---

## ② 母婴出海应用案例

### 场景一：流失预测模型的特征精简

**业务问题**：我们从多个数据源（CRM、广告平台、网站分析、客服系统）汇总了 200+ 特征。但很多冗余特征不仅没用，还拖慢训练速度和增加过拟合。需要筛选出真正影响流失的 top-20 特征。

**数据要求**：100K 用户 × 200 特征。先用 SHAP 做全局重要性排序，再用 RFE 验证

**预期产出**：SHAP 识别出 top-20 特征贡献了 90% 的预测力。精简后模型：训练速度快 3x，AUC 仅下降 0.005

**业务价值**：训练时间从 45 分钟降到 15 分钟（支持每日重训），特征采集成本降低 60%（可减少 API 调用/数据存储）。运营成本节省约 $5,000/月

### 场景二：广告转化预测的 Boruta 特征筛选

**业务问题**：FB 广告 API 返回 50+ 受众特征，但大多数对转化预测无帮助。需要找到最关键的 5-8 个定向特征（如"是否新手妈妈""浏览吸奶器详情页次数"）来优化广告定向。

**数据要求**：50,000 条广告转化记录 × 50 特征。Boruta 算法（基于 Random Forest 的包装法）自动判定特征是否有用

**预期产出**：8 个核心特征（设备类型、浏览页数、是否加购、时段、国家、年龄带、近 7 天访问次数、是否搜索过吸奶器），可解释 85%+ 的转化差异

**业务价值**：广告定向从盲目投放变为精准 8 维定向，CPA 降低 25%，月省 $7,500

---

## ③ 代码模板

```python
"""
Feature Selection Toolkit
特征选择工具集 — Filter / Wrapper / Embedded / SHAP
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    mutual_info_classif, RFE, RFECV, SelectKBest, SelectFromModel
)
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier
from typing import List, Dict, Tuple


def filter_features(
    X: pd.DataFrame, y: np.ndarray, 
    k: int = 20, method: str = 'mutual_info'
) -> Tuple[List[str], pd.DataFrame]:
    """
    过滤法特征选择
    
    Returns:
        (selected_columns, importance_scores)
    """
    if method == 'mutual_info':
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == 'variance':
        scores = X.var().values
    else:
        raise ValueError(f"Unknown method: {method}")
    
    importance = pd.DataFrame({
        'feature': X.columns, 'score': scores
    }).sort_values('score', ascending=False)
    
    selected = importance.head(k)['feature'].tolist()
    return selected, importance


def rfe_feature_selection(
    X: np.ndarray, y: np.ndarray,
    estimator=None, n_features: int = 20, cv: bool = True
) -> List[int]:
    """
    RFE (Recursive Feature Elimination) 特征选择
    """
    estimator = estimator or RandomForestClassifier(n_estimators=50, random_state=42)
    
    if cv:
        selector = RFECV(estimator, min_features_to_select=n_features, 
                         cv=3, scoring='roc_auc')
    else:
        selector = RFE(estimator, n_features_to_select=n_features)
    
    selector.fit(X, y)
    return list(np.where(selector.support_)[0])


def shap_feature_importance(
    X: np.ndarray, y: np.ndarray,
    model=None, max_samples: int = 5000
) -> pd.DataFrame:
    """
    SHAP 特征重要性（需要 pip install shap）
    """
    try:
        import shap
    except ImportError:
        print("[WARN] shap 未安装，回退到 permutation importance")
        return permutation_importance(X, y, model)
    
    model = model or XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    
    # 采样加速
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    importance = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        'feature_idx': range(len(importance)),
        'shap_importance': importance
    }).sort_values('shap_importance', ascending=False)


def permutation_importance(
    X: np.ndarray, y: np.ndarray, model=None
) -> pd.DataFrame:
    """Permutation Importance（无模型依赖的特征重要性）"""
    from sklearn.inspection import permutation_importance as pi
    
    model = model or RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    result = pi(model, X, y, n_repeats=5, random_state=42, scoring='roc_auc')
    return pd.DataFrame({
        'feature_idx': range(X.shape[1]),
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)


# ============ 测试 ============

if __name__ == '__main__':
    np.random.seed(42)
    n, n_features = 3000, 30
    
    # 只有前 8 个特征有信号
    X_signal = np.random.randn(n, 8)
    X_noise = np.random.randn(n, n_features - 8) * 0.3
    X = np.hstack([X_signal, X_noise])
    y = ((X[:, :8].sum(axis=1) + np.random.randn(n) * 2) > 0).astype(int)
    
    columns = [f'feat_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=columns)
    
    # Filter
    selected, imp = filter_features(X_df, y, k=15)
    print(f"Filter: Top-5 = {selected[:5]}")
    assert all(f.startswith('feat_') for f in selected[:5])
    
    # RFE
    rfe_indices = rfe_feature_selection(X, y, n_features=10, cv=False)
    print(f"RFE: selected {len(rfe_indices)} features, top indices = {sorted(rfe_indices)[:5]}")
    
    # Permutation
    perm_imp = permutation_importance(X, y)
    top5 = perm_imp.head(5)['feature_idx'].tolist()
    print(f"Permutation: top-5 feature indices = {top5}")
    # 验证：top-5 应该是前 8 个信号特征
    assert all(i < 8 for i in top5[:3]), f"Top features should be signal features, got {top5[:3]}"
    
    print("\n[✓] 特征选择测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Feature-Engineering]]（特征选择的前提是有特征可选）
- **延伸技能**：[[Skill-Model-Evaluation-Metrics]]（评估特征选择效果）、[[Skill-Ensemble-Methods]]（集成模型对特征冗余更敏感）
- **可组合**：
  - **[[Skill-Customer-Churn-Prediction]]** — 200 特征→20 特征的典型场景
  - **[[Skill-ROAS-Budget-Optimization]]** — 广告定向特征筛选
  - **[[Skill-Imbalanced-Data-Handling]]** — 不平衡场景下特征选择需特别小心

---
- **相关技能**：[[Skill-Hyperparameter-Optimization]]

## ⑤ 商业价值评估

- **ROI 预估**：特征精简可降低训练成本 60%+、推理成本 70%+、数据采集成本 50%+。月省 $5,000-10,000；同时更好的特征集提升模型 AUC 1-3pp。年化贡献 **30-60 万元**。
- **实施难度**：⭐⭐⭐⭐☆（4 星）— 需要理解不同方法的适用场景和局限性
- **优先级评分**：⭐⭐⭐☆☆（3 星）— 模型已经有 200 特征时才紧迫，50 特征以下收益有限
- **评估依据**：特征工程是图谱 #1 被依赖节点，特征选择是其下游最自然的延伸
