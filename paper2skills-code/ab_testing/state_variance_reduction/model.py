"""
Skill-STATE-Robust-Variance-Reduction
STATE: Student's t-distribution for Robust ATE Estimation in Online Experiments
arXiv:2407.16337 | 美团 2024 | 重尾指标鲁棒 A/B 方差减少

参考论文：
  Zhu et al. "STATE: Student's t-distribution for Robust ATE Estimation
  in Online Experiments". arXiv:2407.16337, 2024.

核心贡献：
  将机器学习回归调整（CUPAC/MLRATE）与 Student's t-分布结合，
  解决电商平台重尾指标（GMV、订单量）下 CUPED 方差缩减效率不足的问题。
  
美团真实 A/A 测试效果：
  - 订单量指标方差缩减 70.5%（vs CUPED）
  - GMV 指标方差缩减 80.7%
  - 相比 CUPAC/MLRATE 额外缩减 36%+
  - 同等功效下实验时长缩短约 50%
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import stats


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class HeavyTailMetrics:
    """实验指标数据容器（重尾场景）"""
    metric_values: np.ndarray        # 实验期指标 Y（GMV / 订单量）
    pre_exp_covariates: np.ndarray   # 实验前协变量 X（可多列）
    treatment: np.ndarray            # 处理标记：1=实验组，0=对照组

    def __post_init__(self):
        n = len(self.metric_values)
        assert len(self.treatment) == n, "treatment length mismatch"
        if self.pre_exp_covariates.ndim == 1:
            self.pre_exp_covariates = self.pre_exp_covariates.reshape(-1, 1)
        assert len(self.pre_exp_covariates) == n, "covariate length mismatch"

    @property
    def treatment_idx(self) -> np.ndarray:
        return self.treatment == 1

    @property
    def control_idx(self) -> np.ndarray:
        return self.treatment == 0


@dataclass
class ATEResult:
    """ATE 估计结果"""
    ate: float                          # 平均处理效应
    variance: float                     # 估计量方差
    std_error: float                    # 标准误
    p_value: float                      # 双侧 p-value
    ci_lower: float                     # 95% 置信区间下界
    ci_upper: float                     # 95% 置信区间上界
    t_df: Optional[float] = None        # t 分布自由度（STATE 专属）
    t_scale: Optional[float] = None     # t 分布尺度（STATE 专属）
    variance_reduction_vs_raw: float = 0.0  # vs 原始指标方差的缩减率

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def __str__(self) -> str:
        sig = "✓ 显著" if self.is_significant() else "✗ 不显著"
        lines = [
            f"ATE={self.ate:.4f}  SE={self.std_error:.4f}  "
            f"p={self.p_value:.4f}  {sig}",
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"方差缩减（vs 原始）={self.variance_reduction_vs_raw:.1%}",
        ]
        if self.t_df is not None:
            lines.append(f"t_df={self.t_df:.2f}  t_scale={self.t_scale:.4f}")
        return "\n".join(lines)


# ─── CUPED 基线 ────────────────────────────────────────────────────────────────

class CUPEDEstimator:
    """
    标准 CUPED（Controlled-experiment Using Pre-Experiment Data）

    核心思想：
      Y_cuped = Y - theta * (X - E[X])
      theta = Cov(Y, X) / Var(X)
      方差缩减率 ≈ 1 - ρ(Y,X)²

    局限性：
      假设残差为高斯分布；重尾数据下 ρ 偏低，缩减效果有限。
    """

    def _ols_adjust(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """OLS 回归调整，返回 Y 的残差"""
        X_aug = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
        return Y - X_aug @ coef

    def estimate_ate(self, data: HeavyTailMetrics) -> ATEResult:
        Y = data.metric_values
        X = data.pre_exp_covariates
        T = data.treatment

        # 原始方差（用于计算缩减率）
        nt, nc = int(T.sum()), int((1 - T).sum())
        raw_var = float(np.var(Y[T == 1], ddof=1) / nt
                        + np.var(Y[T == 0], ddof=1) / nc)

        # OLS 调整
        Y_adj = self._ols_adjust(Y, X)
        Yt_adj = Y_adj[data.treatment_idx]
        Yc_adj = Y_adj[data.control_idx]

        ate = float(np.mean(Yt_adj) - np.mean(Yc_adj))
        var_t = float(np.var(Yt_adj, ddof=1) / len(Yt_adj))
        var_c = float(np.var(Yc_adj, ddof=1) / len(Yc_adj))
        var_ate = var_t + var_c
        se = math.sqrt(var_ate)

        z = ate / se
        p_value = float(2 * stats.norm.sf(abs(z)))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        var_reduction = 1.0 - var_ate / raw_var if raw_var > 0 else 0.0

        return ATEResult(
            ate=ate, variance=var_ate, std_error=se,
            p_value=p_value, ci_lower=ci_lower, ci_upper=ci_upper,
            variance_reduction_vs_raw=var_reduction,
        )


# ─── STATE 估计器 ──────────────────────────────────────────────────────────────

class STATEEstimator:
    """
    STATE: Student's t Regression Adjustment for ATE Estimation
    arXiv:2407.16337

    算法流程：
      1. 初步 OLS 得到残差 ε = Y - f(X)
      2. EM 估计残差 t 分布参数 (ν, σ)
         - E步: w_i = (ν+1) / (ν + ε_i²/σ²)    # 隐变量期望权重
         - M步: σ² ← Σw_i·ε_i² / n             # 加权方差
                ν  ← argmax L(ν; w)              # 数值优化
      3. 用权重 w_i 进行加权 OLS 重新调整
      4. 基于 t 分布的方差估计和置信区间

    关键性质：
      - ν 小 → 尾部重（如 ν=3，适合 GMV）
      - ν → ∞ → 退化为高斯（普通 CUPED）
      - 大离群值通过权重 w_i ≈ ν/ε_i² 被自动降权
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.df_: Optional[float] = None    # 拟合的自由度 ν
        self.scale_: Optional[float] = None  # 拟合的尺度 σ

    def fit_t_distribution(
        self, residuals: np.ndarray
    ) -> Tuple[float, float]:
        """
        变分 EM 估计 Student-t 分布参数

        Args:
            residuals: 回归残差数组

        Returns:
            (df, scale): 自由度 ν 和尺度 σ
        """
        r = residuals - np.mean(residuals)  # 中心化
        n = len(r)

        # 初始化：正态矩估计
        sigma = float(np.std(r)) + 1e-8
        nu = 10.0

        for _ in range(self.max_iter):
            # E步：辅助变量期望（t 分布的 EM 权重）
            r_std = r / sigma
            weights = (nu + 1) / (nu + r_std ** 2)

            # M步：更新 σ（加权 MLE）
            sigma_new = float(np.sqrt(np.sum(weights * r ** 2) / n)) + 1e-8

            # M步：更新 ν（grid search 近似 MLE）
            nu_new = self._update_nu(weights, nu)

            # 收敛检验
            if (abs(sigma_new - sigma) < self.tol
                    and abs(nu_new - nu) < self.tol):
                sigma, nu = sigma_new, nu_new
                break
            sigma, nu = sigma_new, nu_new

        # 保证 ν > 2（方差存在条件）
        self.df_ = max(nu, 2.01)
        self.scale_ = sigma
        return self.df_, self.scale_

    @staticmethod
    def _update_nu(weights: np.ndarray, nu_prev: float) -> float:
        """
        对数似然关于 ν 的 grid search 优化

        目标函数（负对数似然 ∝）：
          L(ν) = n·[lgamma((ν+1)/2) - lgamma(ν/2) - 0.5·log(ν)]
               + 0.5·(ν+1)·Σ[log(w_i) - w_i]
        """
        log_w_sum = float(np.sum(np.log(weights + 1e-10) - weights))

        def neg_ll(nu: float) -> float:
            if nu <= 2.0:
                return 1e10
            return -(
                math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                - 0.5 * math.log(nu)
                + 0.5 * (nu + 1) * log_w_sum / len(weights)
            )

        candidates = np.arange(2.1, 50.5, 0.5)
        losses = [neg_ll(float(c)) for c in candidates]
        return float(candidates[int(np.argmin(losses))])

    def _weighted_ols(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """加权 OLS：min Σw_i·(Y_i - Xβ)²，返回残差"""
        X_aug = np.column_stack([np.ones(len(X)), X])
        # WLS 正规方程：(X'WX)β = X'WY
        W_sqrt = np.sqrt(weights)
        Xw = X_aug * W_sqrt[:, None]
        Yw = Y * W_sqrt
        coef, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
        return Y - X_aug @ coef

    def estimate_ate(self, data: HeavyTailMetrics) -> ATEResult:
        """
        STATE ATE 估计

        Args:
            data: HeavyTailMetrics 实验数据

        Returns:
            ATEResult 含 ATE、标准误、p-value、CI、t 参数
        """
        Y = data.metric_values
        X = data.pre_exp_covariates
        T = data.treatment
        nt, nc = int(T.sum()), int((1 - T).sum())

        # 原始方差
        raw_var = float(np.var(Y[T == 1], ddof=1) / nt
                        + np.var(Y[T == 0], ddof=1) / nc)

        # 步骤1：初步 OLS 残差
        X_aug = np.column_stack([np.ones(len(X)), X])
        coef0, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
        residuals0 = Y - X_aug @ coef0

        # 步骤2：EM 估计 t 参数
        nu, sigma = self.fit_t_distribution(residuals0)

        # 步骤3：t 加权 OLS
        weights = (nu + 1) / (nu + (residuals0 / sigma) ** 2)
        Y_adj = self._weighted_ols(Y, X, weights)

        # 步骤4：ATE 估计
        Yt_adj = Y_adj[data.treatment_idx]
        Yc_adj = Y_adj[data.control_idx]
        wt = weights[data.treatment_idx]
        wc = weights[data.control_idx]

        ate = float(np.mean(Yt_adj) - np.mean(Yc_adj))

        # t 加权方差估计
        def weighted_var(y: np.ndarray, w: np.ndarray) -> float:
            w_sum = w.sum()
            w_sum2 = (w ** 2).sum()
            y_wmean = np.sum(w * y) / w_sum
            return float(np.sum(w * (y - y_wmean) ** 2)
                         / (w_sum - w_sum2 / w_sum))

        var_t = weighted_var(Yt_adj, wt) / nt
        var_c = weighted_var(Yc_adj, wc) / nc
        var_ate = var_t + var_c
        se = math.sqrt(var_ate)

        # t 检验（使用 Satterthwaite df 近似）
        df_test = min(nt, nc) - 1
        t_stat = ate / se
        p_value = float(2 * stats.t.sf(abs(t_stat), df=df_test))
        t_crit = stats.t.ppf(0.975, df=df_test)
        ci_lower = ate - t_crit * se
        ci_upper = ate + t_crit * se

        var_reduction = 1.0 - var_ate / raw_var if raw_var > 0 else 0.0

        return ATEResult(
            ate=ate, variance=var_ate, std_error=se,
            p_value=p_value, ci_lower=ci_lower, ci_upper=ci_upper,
            t_df=nu, t_scale=sigma,
            variance_reduction_vs_raw=var_reduction,
        )

    def variance_reduction_vs_cuped(self, data: HeavyTailMetrics) -> float:
        """
        计算 STATE 相对 CUPED 的额外方差缩减率

        Returns:
            float: 额外缩减率（正值表示 STATE 方差更小）
        """
        cuped = CUPEDEstimator()
        cuped_result = cuped.estimate_ate(data)
        state_result = self.estimate_ate(data)
        if cuped_result.variance <= 0:
            return 0.0
        return max(0.0, 1.0 - state_result.variance / cuped_result.variance)


# ─── 样本量计算 ────────────────────────────────────────────────────────────────

class ABTestPowerCalculator:
    """
    基于 STATE 方差缩减的 A/B 实验样本量与实验天数计算

    用法：
        calc = ABTestPowerCalculator(alpha=0.05, power=0.80)
        result = calc.experiment_days(
            daily_users=5000, sigma=300, mde=15,
            variance_reduction=0.70
        )
    """

    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        self.alpha = alpha
        self.power = power

    def _critical_values(self) -> Tuple[float, float]:
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)
        return z_alpha, z_beta

    def sample_size_traditional(self, sigma: float, mde: float) -> int:
        """标准 t 检验每组所需样本量"""
        z_alpha, z_beta = self._critical_values()
        n = 2 * ((z_alpha + z_beta) * sigma / mde) ** 2
        return math.ceil(n)

    def sample_size_state(
        self,
        sigma: float,
        mde: float,
        variance_reduction: float,
    ) -> int:
        """STATE 方差缩减后每组所需样本量"""
        n_trad = self.sample_size_traditional(sigma, mde)
        return max(math.ceil(n_trad * (1 - variance_reduction)), 10)

    def experiment_days(
        self,
        daily_users: int,
        sigma: float,
        mde: float,
        variance_reduction: float = 0.70,
    ) -> dict:
        """
        计算实验天数对比

        Args:
            daily_users: 每日实验用户量
            sigma: 指标标准差
            mde: 最小可检测效应（绝对值）
            variance_reduction: STATE 方差缩减率（默认 0.70）

        Returns:
            dict: 传统方法 vs STATE 的天数、样本量、加速比
        """
        n_trad = self.sample_size_traditional(sigma, mde)
        n_state = self.sample_size_state(sigma, mde, variance_reduction)
        days_trad = math.ceil(n_trad * 2 / max(daily_users, 1))
        days_state = math.ceil(n_state * 2 / max(daily_users, 1))

        return {
            "n_per_group_traditional": n_trad,
            "n_per_group_state": n_state,
            "days_traditional": days_trad,
            "days_state": days_state,
            "days_saved": days_trad - days_state,
            "speedup_ratio": round(days_trad / max(days_state, 1), 2),
        }


# ─── 测试函数 ──────────────────────────────────────────────────────────────────

def _simulate_heavy_tail_gmv(n: int, seed: int = 42) -> HeavyTailMetrics:
    """
    模拟重尾 GMV 数据
    - 90% 普通用户：GMV ~ Normal(200, 50)
    - 10% 大客户（批量采购/月子中心）：GMV ~ Normal(2000, 500)
    真实 ATE = 10 元（新 Listing 提升 GMV）
    """
    rng = np.random.default_rng(seed)
    n_heavy = max(int(n * 0.1), 1)
    n_normal = n - n_heavy

    # 实验前协变量（过去 14 天 GMV）
    X_normal = rng.normal(180, 40, n_normal)
    X_heavy = rng.normal(1800, 400, n_heavy)
    X = np.concatenate([X_normal, X_heavy])

    # 随机分组
    treatment = rng.integers(0, 2, n)

    # 实验期指标
    Y_normal = rng.normal(200, 50, n_normal)
    Y_heavy = rng.normal(2000, 500, n_heavy)
    Y_base = np.concatenate([Y_normal, Y_heavy])
    Y = Y_base + 10.0 * treatment  # 真实 ATE = 10

    return HeavyTailMetrics(
        metric_values=Y,
        pre_exp_covariates=X,
        treatment=treatment,
    )


def run_test() -> None:
    """STATE vs CUPED 对比测试"""
    print("=" * 60)
    print("STATE 重尾 GMV 方差缩减测试")
    print("=" * 60)

    data = _simulate_heavy_tail_gmv(n=2000)
    print(f"\n数据: N={len(data.metric_values)}, "
          f"GMV均值={np.mean(data.metric_values):.1f}, "
          f"GMV标准差={np.std(data.metric_values):.1f}")
    print(f"最大GMV={np.max(data.metric_values):.0f}（重尾特征）")

    cuped = CUPEDEstimator()
    cuped_result = cuped.estimate_ate(data)
    print(f"\n[CUPED]\n{cuped_result}")

    state = STATEEstimator()
    state_result = state.estimate_ate(data)
    print(f"\n[STATE]\n{state_result}")

    extra = state.variance_reduction_vs_cuped(data)
    print(f"\nSTATE vs CUPED 额外方差缩减: {extra:.1%}")

    sigma = float(np.std(data.metric_values))
    calc = ABTestPowerCalculator()
    days_info = calc.experiment_days(
        daily_users=5000, sigma=sigma, mde=10.0,
        variance_reduction=state_result.variance_reduction_vs_raw,
    )
    print(f"\n[实验天数对比]")
    for k, v in days_info.items():
        print(f"  {k}: {v}")

    # 基本正确性断言
    assert state_result.variance <= cuped_result.variance * 1.2, \
        "STATE 方差应不大于 CUPED（允许20%误差）"
    assert days_info["days_state"] <= days_info["days_traditional"], \
        "STATE 实验天数应不超过传统方法"
    print("\n[✓] 所有断言通过")


if __name__ == "__main__":
    run_test()
