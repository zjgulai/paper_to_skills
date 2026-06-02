"""
Hyperparameter Optimization — Optuna/Bayesian/Hyperband 超参调优
paper2skills-code: 12-ML基础 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field


@dataclass
class Trial:
    trial_id: int
    params: dict
    score: float
    duration_s: float = 1.0


@dataclass
class HPOResult:
    method: str
    best_params: dict
    best_score: float
    all_trials: list[Trial]
    total_trials: int


def mock_objective(params: dict, seed: int = 0) -> float:
    """模拟目标函数（AUC，越高越好）"""
    lr = params.get("learning_rate", 0.01)
    n_est = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    random.seed(seed + int(lr * 1000) + n_est + max_depth)
    optimal = 0.88
    lr_penalty = abs(math.log(lr) - math.log(0.03)) * 0.02
    depth_penalty = abs(max_depth - 5) * 0.01
    noise = random.gauss(0, 0.01)
    return round(min(0.95, optimal - lr_penalty - depth_penalty + noise), 4)


def grid_search(param_grid: dict, max_trials: int = 27) -> HPOResult:
    """网格搜索（枚举）"""
    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))[:max_trials]
    trials = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        score = mock_objective(params, seed=i)
        trials.append(Trial(i, params, score))
    best = max(trials, key=lambda t: t.score)
    return HPOResult("GridSearch", best.params, best.score, trials, len(trials))


def random_search(param_space: dict, n_trials: int = 30) -> HPOResult:
    """随机搜索"""
    random.seed(42)
    trials = []
    for i in range(n_trials):
        params = {k: random.choice(v) if isinstance(v, list) else
                  v[0] + random.random() * (v[1] - v[0])
                  for k, v in param_space.items()}
        params = {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}
        score = mock_objective(params, seed=i)
        trials.append(Trial(i, params, score))
    best = max(trials, key=lambda t: t.score)
    return HPOResult("RandomSearch", best.params, best.score, trials, n_trials)


def bayesian_search(param_space: dict, n_trials: int = 20) -> HPOResult:
    """贝叶斯优化（简化版：基于历史最优邻域搜索）"""
    random.seed(99)
    trials = []
    best_score = -1.0
    best_params = {k: (v[0] if isinstance(v, list) else (v[0] + v[1]) / 2)
                   for k, v in param_space.items()}

    for i in range(n_trials):
        if i < 5 or random.random() < 0.3:
            params = {k: random.choice(v) if isinstance(v, list) else
                      v[0] + random.random() * (v[1] - v[0])
                      for k, v in param_space.items()}
        else:
            params = {}
            for k, v in param_space.items():
                if isinstance(v, list):
                    params[k] = random.choice(v)
                else:
                    center = best_params.get(k, (v[0] + v[1]) / 2)
                    span = (v[1] - v[0]) * 0.3
                    params[k] = min(v[1], max(v[0], center + random.gauss(0, span / 3)))
        params = {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}
        score = mock_objective(params, seed=i + 100)
        trials.append(Trial(i, params, score))
        if score > best_score:
            best_score = score
            best_params = params

    best = max(trials, key=lambda t: t.score)
    return HPOResult("Bayesian", best.params, best.score, trials, n_trials)


def run_hpo_demo():
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
    }
    param_space = {
        "learning_rate": [0.001, 0.2],
        "n_estimators": [50, 200, 100, 150],
        "max_depth": [3, 5, 7, 9],
    }

    print("=== 超参调优方法对比（母婴 Churn 预测模型）===\n")
    for method_fn, name in [
        (lambda: grid_search(param_grid), "GridSearch"),
        (lambda: random_search(param_space, 30), "RandomSearch"),
        (lambda: bayesian_search(param_space, 20), "Bayesian"),
    ]:
        result = method_fn()
        print(f"  {name:15s}: 最优 AUC {result.best_score:.4f}"
              f" | 尝试 {result.total_trials} 次 | 最优参数: {result.best_params}")

    print("\n✅ 超参调优演示完成")


if __name__ == "__main__":
    run_hpo_demo()
