---
title: AutoML 流水线设计 — Optuna TPE + FLAML 自动化建模
doc_type: knowledge
module: 12-ML基础
topic: automl-pipeline-design
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: AutoML 流水线设计

> **论文**：FLAML: A Fast and Lightweight AutoML Library (Wang et al., 2021)
> **arXiv**：2005.01571 | 2021 | **桥梁**: 12-ML基础 ↔ 04-供应链 | **类型**: 算法工具

---

## ① 算法原理

**核心思想**：AutoML 将「特征工程 → 算法选择 → 超参调优」三个步骤自动化，通过贝叶斯优化在搜索空间中高效找到最优配置，让非 ML 专家也能获得接近专家水平的模型。

**数学直觉**：
- **TPE（Tree-structured Parzen Estimator）**：将超参搜索转化为密度比估计问题
  - $p(y < y^* | \mathbf{x}) \propto l(\mathbf{x}) / g(\mathbf{x})$，其中 $l$ 为好的超参分布，$g$ 为差的超参分布
  - 最大化 $l/g$ 比率，优先采样高概率区域
- **FLAML 成本感知搜索**：在每步搜索时考虑训练时间 $t_k$，按 $\text{score}/t_k$ 排序，优先试验「快且好」的配置
- **嵌套交叉验证**：外层评估泛化性能，内层用于超参选择，防止数据泄露

**关键假设**：
- 搜索空间定义合理（包含真正优秀的配置）
- 数据量足够支持交叉验证（推荐 ≥ 1000 样本）
- 目标指标已明确（RMSE/AUC/MAPE 等）

---

## ② 母婴出海应用案例

**场景A：销量预测模型自动优化**

- **业务问题**：数据工程师每季度手动调优 XGBoost 预测模型，耗时 3 天，调优结果依赖个人经验，MAPE 长期停在 18% 无法突破
- **数据要求**：历史销量数据（时间序列特征矩阵，≥5000 行）、特征列表（价格/促销/节假日/竞品价格）
- **预期产出**：AutoML 在 2 小时内自动试验 200+ 配置（LightGBM/XGBoost/Random Forest + 超参），MAPE 从 18% → 12%，最优模型自动导出
- **业务价值**：预测精度提升 6pp → 库存准确率提升 → 年化节省缺货损失约 **80 万元**；工程师从 3 天/季 → 0.5 天，节省人力成本约 **5 万元/年**

**场景B：Listing 质量分类模型快速迭代**

- **业务问题**：新品上架时需要预测 Listing 质量评分（高/中/低），人工打标昂贵，需要自动分类模型
- **数据要求**：历史 Listing 数据 2000 条（标题长度/图片数/子弹点数/评分/类目等特征）+ 人工标注质量标签
- **预期产出**：AutoML 自动搜索分类模型，F1 从手工调 LR 的 0.71 → 自动的 0.84，搜索时间 30 分钟
- **业务价值**：Listing 优化命中率提升 18%，年化新品首月 GMV 提升估算约 **25 万元**

---

## ③ 代码模板

```python
"""
AutoML 流水线设计 — Optuna TPE 贝叶斯超参搜索
（不依赖 FLAML/AutoSklearn，用 Optuna + sklearn 实现核心逻辑）
"""
import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_pipeline(trial: optuna.Trial, X: np.ndarray) -> Pipeline:
    """
    定义搜索空间：算法选择 + 超参
    """
    algo = trial.suggest_categorical("algorithm", ["gbm", "rf", "ridge"])

    if algo == "gbm":
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("gbm_n_estimators", 50, 300),
            max_depth=trial.suggest_int("gbm_max_depth", 2, 8),
            learning_rate=trial.suggest_float("gbm_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("gbm_subsample", 0.6, 1.0),
            random_state=42,
        )
    elif algo == "rf":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
            max_depth=trial.suggest_int("rf_max_depth", 3, 15),
            min_samples_leaf=trial.suggest_int("rf_min_leaf", 1, 10),
            random_state=42,
        )
    else:
        model = Ridge(
            alpha=trial.suggest_float("ridge_alpha", 0.01, 100.0, log=True),
        )

    use_scaler = trial.suggest_categorical("use_scaler", [True, False])
    steps = ([("scaler", StandardScaler())] if use_scaler else []) + [("model", model)]
    return Pipeline(steps)


def automl_search(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv: int = 5,
    metric: str = "neg_mean_absolute_percentage_error"
) -> dict:
    """
    AutoML 超参搜索主入口
    返回：最优算法配置 + 验证分数
    """
    def objective(trial):
        pipe = make_pipeline(trial, X)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    return {
        "best_params": best_trial.params,
        "best_cv_score": -best_trial.value,  # 转回正值 MAPE
        "n_trials": len(study.trials),
        "improvement": None,  # 后续填入 baseline 对比
    }


def fit_best_model(X: np.ndarray, y: np.ndarray, best_params: dict) -> Pipeline:
    """用最优参数在全量数据训练最终模型"""
    trial_like = type("T", (), {"params": best_params})()
    # 简化重建：直接用 best_params 中的 algorithm 重建
    algo = best_params.get("algorithm", "gbm")
    if algo == "gbm":
        model = GradientBoostingRegressor(
            n_estimators=best_params.get("gbm_n_estimators", 100),
            max_depth=best_params.get("gbm_max_depth", 4),
            learning_rate=best_params.get("gbm_lr", 0.1),
            subsample=best_params.get("gbm_subsample", 0.8),
            random_state=42,
        )
    elif algo == "rf":
        model = RandomForestRegressor(
            n_estimators=best_params.get("rf_n_estimators", 100),
            max_depth=best_params.get("rf_max_depth", 8),
            min_samples_leaf=best_params.get("rf_min_leaf", 3),
            random_state=42,
        )
    else:
        model = Ridge(alpha=best_params.get("ridge_alpha", 1.0))

    use_scaler = best_params.get("use_scaler", False)
    steps = ([("scaler", StandardScaler())] if use_scaler else []) + [("model", model)]
    pipe = Pipeline(steps)
    pipe.fit(X, y)
    return pipe


# ─── 测试用例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    # 模拟母婴销量特征数据（价格、促销力度、节假日、历史销量）
    n_samples = 500
    X = np.column_stack([
        np.random.uniform(50, 300, n_samples),   # 售价
        np.random.uniform(0, 0.5, n_samples),    # 折扣率
        np.random.randint(0, 2, n_samples),      # 是否节假日
        np.random.uniform(0, 100, n_samples),    # 上周销量
        np.random.uniform(1, 5, n_samples),      # 评分
    ])
    # 真实销量（含非线性关系）
    y = (
        200 - X[:, 0] * 0.3
        + X[:, 1] * 150
        + X[:, 2] * 80
        + X[:, 3] * 0.7
        + np.random.normal(0, 15, n_samples)
    )

    # 基线（Ridge 默认参数）
    baseline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    baseline_scores = cross_val_score(baseline, X, y, cv=3,
                                      scoring="neg_mean_absolute_percentage_error")
    baseline_mape = -baseline_scores.mean()
    print(f"基线 MAPE: {baseline_mape:.4f}")

    # AutoML 搜索（30 次快速验证）
    print("AutoML 搜索中（30 trials）...")
    result = automl_search(X, y, n_trials=30, cv=3)
    result["improvement"] = baseline_mape - result["best_cv_score"]

    print(f"最优算法: {result['best_params'].get('algorithm', 'N/A')}")
    print(f"AutoML MAPE: {result['best_cv_score']:.4f}")
    print(f"MAPE 改善: -{result['improvement']:.4f} ({result['improvement']/baseline_mape*100:.1f}%)")

    # 训练最终模型
    best_model = fit_best_model(X, y, result["best_params"])
    y_pred = best_model.predict(X[:10])
    print(f"样本预测（前10）: {y_pred[:3].round(1)}")

    # 验证
    assert result["best_cv_score"] > 0, "MAPE 不应为负"
    assert len(result["best_params"]) > 0, "无最优参数"
    assert result["n_trials"] == 30, "试验次数不对"
    print("\n[✓] AutoML 流水线设计 测试通过")
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Hyperparameter-Optimization]]（理解贝叶斯优化基础）、[[Skill-Feature-Selection]]（AutoML 前做特征筛选可减少搜索空间）
- **延伸（extends）**：[[Skill-Online-Incremental-Learning]]（AutoML 找到最优架构后，用增量学习持续更新）
- **可组合（combinable）**：[[Skill-Model-Performance-Monitor]]（AutoML 上线后监控性能漂移，触发重新搜索）、[[Skill-Data-Drift-Detection]]（检测到数据漂移时自动触发 AutoML 重训）

---

## ⑤ 商业价值评估

- **ROI 预估**：预测精度提升 6pp 带来库存节省 **80 万元/年**；工程师人力节省约 **5 万元/年**。总年化约 **85 万元**
- **实施难度**：⭐⭐⭐☆☆（需要 optuna 库，搜索时间随 n_trials 增加；本地 CPU 50 trials 约 5-15 分钟）
- **优先级**：⭐⭐⭐⭐⭐（通用性极强，所有需要建模的 Skill 都可受益；一次实现多处复用）
- **评估依据**：FLAML 论文在多个 benchmark 上比手工调优平均提升 15-30%；母婴供应链场景数据量适中（103 万 SKU），AutoML 完全可行
