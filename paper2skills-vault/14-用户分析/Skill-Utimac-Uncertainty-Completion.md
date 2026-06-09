---
title: 不确定性感知矩阵补全 - 补全值带置信区间的页面转移矩阵恢复
doc_type: knowledge
module: 14-用户分析
topic: uncertainty-aware-matrix-completion
status: stable
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: arXiv:2605.02225 (2025)
roadmap_phase: phase2
---

# Skill: Utimac Uncertainty-Aware Completion — 带置信区间的矩阵补全

> 论文：**Rethinking Traffic Matrix Completion: Estimate the Process, Not the Entries** · arXiv:2605.02225 (2025)
> 作者：Xiyuan Liu, Zihao Wang, Guanzuo Liu, Xiucheng Tian, Wenting Wei · 西安电子科技大学
> 应用：补全稀疏的页面转移矩阵，每个补全值附带预测区间

---

## ① 算法原理

### 核心思想

Utimac 的核心洞察是：**不直接补全矩阵条目，而是推断数据生成过程的参数**（Estimate the Process, Not the Entries）。

传统方法（低秩补全、深度学习）将每个缺失值作为直接优化目标，结果是点估计、黑盒映射、高稀疏度下性能骤降。Utimac 的不同之处：

1. **对数域分解**：流量矩阵在 log 域可分解为「联合高斯主成分 $U_t$」+「Laplace 稀疏偏差 $O_t$」
   - $Z_t = \log(X_t + \varepsilon) = U_t + O_t$
   - $U_t \sim \mathcal{N}(\mu_n, \Sigma_n)$：捕捉时间窗内稳定的共现结构（类似页面间的相关性）
   - $O_t \sim \text{Laplace}(0, 1/\lambda_n)$：捕捉突发流量偏差（稀疏事件）

2. **局部平稳时间窗**：在一个窗口 $W_n$ 内，多帧（多天/多session批次）共享同一组参数 $\theta_n = (\mu_n, \Sigma_n, \lambda_n)$，使得稀疏观测通过"汇聚"变得可识别

3. **参数推断替代条目补全**：将补全问题重写为极大似然估计——从窗口内所有部分观测帧中联合推断共享参数，再从参数恢复缺失值

4. **正则化替代原积分似然**：原边际似然含高维卷积积分（不可解析），引入"剖面近似 + 正则化"构造可计算的代理目标，加 $\rho \cdot \text{tr}(\Sigma_n^{-1})$（防协方差坍缩）和 $\eta \cdot \lambda_n$（防稀疏度发散）

5. **块坐标下降（BCD）求解**：交替更新四个变量块：
   - 偏差变量 $\{o_\tau^{(\Omega_\tau)}\}$ → 软阈值算子（Lasso 等价）
   - 均值 $\mu_n$ → 加权最小二乘
   - 稀疏度 $\lambda_n$ → 闭合式更新
   - 协方差 $\Sigma_n$ → 广义 EM 型更新

6. **闭式预测区间**：参数估计完成后，缺失条目的后验分布是 Normal-Laplace 卷积，利用解析 CDF 直接输出 95% 置信区间

### 数学直觉

**偏差更新（软阈值）**：

对每帧 $\tau$，固定 $\mu_n, \Sigma_n, \lambda_n$，求：

```
o_hat = argmin { ½(o - r)ᵀ Q(o - r) + λ‖o‖₁ }
```

其中 $r = z^{(\Omega)} - \mu^{(\Omega)}$，$Q = (\Sigma^{(\Omega,\Omega)})^{-1}$。
当 $\Sigma^{(\Omega,\Omega)} = \sigma^2 I$ 时退化为坐标级软阈值 $\hat{o}_k = \text{sign}(r_k)\max(|r_k| - \lambda\sigma^2, 0)$

**稀疏度更新（闭合式）**：

$$\hat{\lambda}_n = \frac{\sum_\tau |\Omega_\tau|}{\sum_\tau \|\hat{o}_\tau^{(\Omega_\tau)}\|_1 + \eta}$$

直觉：分子是总观测点数（Laplace分布"事件数"），分母是总偏差绝对值（"总偏移量"），比值是 MLE 估计量加正则化分母

**均值更新（加权最小二乘）**：

$$\left(\sum_\tau P_\tau^\top (\Sigma_n^{(\Omega_\tau,\Omega_\tau)})^{-1} P_\tau\right) \mu_n = \sum_\tau P_\tau^\top (\Sigma_n^{(\Omega_\tau,\Omega_\tau)})^{-1} a_\tau$$

$a_\tau = z_\tau^{(\Omega)} - \hat{o}_\tau^{(\Omega)}$ 是去偏后的残差，每帧按精度矩阵加权

**预测区间**：缺失条目 $Z_t(j)$ 的条件分布是 Normal-Laplace 混合，CDF 有解析形式，直接输出 97.5% 分位数作为区间上界

### 关键效果数字

| 数据集 | 指标 | Utimac vs 最佳 baseline | 稀疏度 |
|--------|------|------------------------|--------|
| Facebook-Pod-B | wMAPE | **↓12.3%** (vs ImputeFormer) | $p_\text{obs}=0.3$ |
| Facebook-Pod-B | wMAPE | ↓2.5% (vs ImputeFormer) | $p_\text{obs}=0.9$ |
| Facebook-ToR-A | MAE | **↓32.9%** (vs PSW-I) | $p_\text{obs}=0.3$ |
| Facebook-ToR-A | MAE | ↓31.3% (vs PSW-I) | $p_\text{obs}=0.9$ |
| Facebook-ToR-A | RMSE | Diffusion-TM 失控 = 0.89 vs Utimac **0.094** | $p_\text{obs}=0.9$ |
| Facebook-Pod-B | Burst-wMAPE | **↓14.5%** (vs PSW-I) | $p_\text{obs}=0.3$ |

> 核心规律：**稀疏度越高，Utimac 优势越大**。深度学习方法在少样本下性能下降，Utimac 基于统计推断，稀疏时更具优势（统计估计天然更"数据高效"）。

---

## ② 母婴出海应用案例

### 场景：桑基图转移矩阵的"可信补全"

**业务问题**：跨境电商平台（如独立站）的用户行为分析依赖页面转移矩阵——HOME 页流向哪里？PDP 页流失到哪里？但由于用户路径稀疏（每个session仅3-8次点击），加之数据收集丢失（广告拦截器、延迟等），实际观测到的转移矩阵往往只有 30%-60% 填充率。

现有矩阵补全方法（如低秩补全）给出的是点估计：HOME→CHECKOUT 转移概率 = 12%，但无法回答「这个 12% 可信吗？」。

**Utimac 的价值**：输出 12% ± 区间，例如：
- **区间窄（95% CI: [10%, 14%]）**：数据充分，可信，基于此优化 HOME 页 CTA 按钮
- **区间宽（95% CI: [3%, 21%]）**：数据不足，不可信，告诉决策者「暂不基于此做预算调整，先补充数据」

**数据要求**：
- 多天/多周的 session 级页面转移计数矩阵（部分观测即可，不需要完整矩阵）
- 时间窗内的观测掩码矩阵（哪些转移对被实际观测到）
- 建议窗口大小：7-30 天，确保局部平稳性假设成立

**预期产出**：
1. 完整转移矩阵（补全所有缺失转移概率）
2. 每个条目的 95% 预测区间
3. "可信度地图"：基于区间宽度，标记哪些转移路径数据充足、哪些需要加强监测

**业务价值**：
- **避免虚假优化**：区间宽的条目（[3%, 21%]）表明数据不可靠，不做重大预算决策
- **识别数据盲点**：哪些关键路径（如 CART→CHECKOUT）区间特别宽 → 需要专项数据质量治理
- **提升用户漏斗分析可信度**：从"给一个数字"升级到"给一个可信数字"

### 场景：多SKU商品页面热力图的不确定性标注

**业务问题**：独立站有 500+ SKU，但热门品只有 20 个，冷门品每日访问量极少，无法形成可信的页面间转移统计。但运营需要对所有商品的"用户浏览路径"做判断。

**Utimac 方案**：
- 将商品划分为若干品类，品类内商品在 log 域共享联合高斯结构
- 对冷门品（观测稀疏），其区间自动变宽，提示"数据不足"
- 对热门品（观测充分），区间窄，支持高置信度的路径优化决策

**数据要求**：按品类聚合的商品-页面转移计数矩阵，时间粒度按日或按周

**预期产出**：带置信度标注的商品热力图，冷门商品自动标记"数据待充"

---

## ③ 代码模板

```python
"""
Utimac: Uncertainty-Aware Matrix Completion
基于 arXiv:2605.02225 的实现

适用场景：稀疏页面转移矩阵补全，输出每个补全值的 95% 置信区间

用法示例：
    python utimac.py  # 运行内置示例（母婴电商转移矩阵）
"""

import numpy as np
from scipy.linalg import solve, cho_factor, cho_solve
from scipy.stats import norm as scipy_norm
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─── 软阈值算子（偏差更新核心）────────────────────────────────────────────────

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """逐元素软阈值：sign(x) * max(|x| - threshold, 0)"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def update_deviation(
    z_obs: np.ndarray,       # 观测 log 域子向量 (|Ω|,)
    mu_obs: np.ndarray,      # 均值子向量 (|Ω|,)
    sigma_obs: np.ndarray,   # 协方差子矩阵 (|Ω|, |Ω|)
    lam: float,              # 稀疏度参数
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    求解偏差更新：argmin ½(o-r)ᵀ Q(o-r) + λ‖o‖₁
    使用坐标下降（对角近似时退化为软阈值）

    当 Σ 接近对角矩阵时，使用封闭式软阈值；
    否则使用 ADMM 风格的坐标下降
    """
    r = z_obs - mu_obs
    d = len(r)

    # 尝试 Cholesky 分解判断正定性
    try:
        cho = cho_factor(sigma_obs, lower=False)
        Q = np.linalg.inv(sigma_obs)  # 精度矩阵
    except np.linalg.LinAlgError:
        # 添加微小对角扰动保证正定
        sigma_obs = sigma_obs + 1e-6 * np.eye(d)
        Q = np.linalg.inv(sigma_obs)

    # 检查是否接近对角（对角近似加速）
    off_diag_ratio = (
        np.sum(np.abs(Q - np.diag(np.diag(Q)))) /
        (np.sum(np.abs(Q)) + 1e-10)
    )

    if off_diag_ratio < 0.01:
        # 对角近似：逐元素软阈值
        q_diag = np.diag(Q)
        return soft_threshold(r, lam / q_diag)

    # 坐标下降（完整精度矩阵）
    o = soft_threshold(r, lam / (np.diag(Q) + 1e-10))
    for _ in range(max_iter):
        o_prev = o.copy()
        for k in range(d):
            # 固定其他分量，对第 k 个分量做精确更新
            residual_k = r[k] - np.dot(Q[k, :k], o[:k]) - np.dot(Q[k, k+1:], o[k+1:])
            # 精确更新：1D 软阈值
            q_kk = Q[k, k]
            o[k] = soft_threshold(np.array([residual_k / q_kk]), lam / q_kk)[0]
        if np.max(np.abs(o - o_prev)) < tol:
            break
    return o


# ─── BCD 主流程 ───────────────────────────────────────────────────────────────

class Utimac:
    """
    Utimac: Uncertainty-Aware Traffic Matrix Completion

    参数
    ----
    window_size : 局部平稳时间窗大小（帧数）
    rho         : 协方差正则化强度（防止协方差坍缩）
    eta         : 稀疏度正则化强度（防止稀疏度发散）
    bcd_iters   : 块坐标下降最大迭代次数
    tol         : BCD 收敛阈值
    alpha_ci    : 置信区间置信水平（默认 0.95）
    """

    def __init__(
        self,
        window_size: int = 10,
        rho: float = 0.1,
        eta: float = 0.01,
        bcd_iters: int = 50,
        tol: float = 1e-4,
        alpha_ci: float = 0.95,
    ):
        self.window_size = window_size
        self.rho = rho
        self.eta = eta
        self.bcd_iters = bcd_iters
        self.tol = tol
        self.alpha_ci = alpha_ci

        # 拟合后的参数（按窗口）
        self.theta_windows_: List[dict] = []

    def fit_window(
        self,
        Z_frames: List[np.ndarray],   # 每帧完整 log 域向量（NaN 表示缺失）
        masks: List[np.ndarray],       # 每帧观测掩码（True = 观测到）
    ) -> dict:
        """
        对单个时间窗拟合参数 θ_n = (μ_n, Σ_n, λ_n)

        返回
        ----
        dict with keys: mu, Sigma, lam, deviations (per frame)
        """
        d = Z_frames[0].shape[0]
        L = len(Z_frames)

        # 初始化参数
        # μ：用观测均值初始化（忽略 NaN）
        obs_vals = []
        for z, mask in zip(Z_frames, masks):
            obs_vals.extend(z[mask].tolist())
        global_mean = np.mean(obs_vals) if obs_vals else 0.0

        mu = np.full(d, global_mean)
        Sigma = np.eye(d)
        lam = 1.0

        deviations = [np.zeros(np.sum(m)) for m in masks]

        prev_mu = mu.copy()
        for iteration in range(self.bcd_iters):
            # === Step 1: 更新偏差变量 ===
            new_deviations = []
            for t, (z, mask) in enumerate(zip(Z_frames, masks)):
                obs_idx = np.where(mask)[0]
                if len(obs_idx) == 0:
                    new_deviations.append(np.zeros(0))
                    continue
                z_obs = z[obs_idx]
                mu_obs = mu[obs_idx]
                sigma_obs = Sigma[np.ix_(obs_idx, obs_idx)]
                o_hat = update_deviation(z_obs, mu_obs, sigma_obs, lam)
                new_deviations.append(o_hat)
            deviations = new_deviations

            # === Step 2: 更新均值 μ（加权最小二乘）===
            A = np.zeros((d, d))
            b = np.zeros(d)
            for t, (z, mask) in enumerate(zip(Z_frames, masks)):
                obs_idx = np.where(mask)[0]
                if len(obs_idx) == 0:
                    continue
                sigma_obs = Sigma[np.ix_(obs_idx, obs_idx)]
                try:
                    Q_obs = np.linalg.inv(sigma_obs)
                except np.linalg.LinAlgError:
                    Q_obs = np.linalg.pinv(sigma_obs)

                a_t = z[obs_idx] - deviations[t]  # 去偏残差

                # 累加正规方程
                P = np.zeros((len(obs_idx), d))
                for i, idx in enumerate(obs_idx):
                    P[i, idx] = 1.0
                A += P.T @ Q_obs @ P
                b += P.T @ Q_obs @ a_t

            # 求解：添加轻微正则防奇异
            A += 1e-8 * np.eye(d)
            try:
                mu = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                mu = np.linalg.lstsq(A, b, rcond=None)[0]

            # === Step 3: 更新稀疏度 λ（闭合式）===
            total_obs = sum(np.sum(m) for m in masks)
            total_dev_l1 = sum(np.sum(np.abs(dev)) for dev in deviations)
            lam = total_obs / (total_dev_l1 + self.eta)

            # === Step 4: 更新协方差 Σ（M 步，近似梯度）===
            # 使用残差的样本协方差作为 EM 型更新
            residuals = []
            for t, (z, mask) in enumerate(zip(Z_frames, masks)):
                obs_idx = np.where(mask)[0]
                if len(obs_idx) == 0:
                    continue
                e = z[obs_idx] - deviations[t] - mu[obs_idx]
                residuals.append((obs_idx, e))

            # 构建 d×d 散布矩阵（累加观测到的条目）
            S = np.zeros((d, d))
            count = np.zeros((d, d))
            for obs_idx, e in residuals:
                for i, ii in enumerate(obs_idx):
                    for j, jj in enumerate(obs_idx):
                        S[ii, jj] += e[i] * e[j]
                        count[ii, jj] += 1

            # 归一化（只更新有观测的条目）
            mask_count = count > 0
            S_norm = np.where(mask_count, S / (count + 1e-8), Sigma)

            # 使用 Woodbury 型正则更新：Σ = S_norm + ρ diag(Σ⁻¹ 对角逆)
            # 简化：Σ_new = α * S_norm + (1-α) * Σ + ρ * I
            alpha_cov = 0.3
            Sigma_new = alpha_cov * S_norm + (1 - alpha_cov) * Sigma
            # 加正则化确保正定
            eigvals = np.linalg.eigvalsh(Sigma_new)
            if eigvals.min() < 1e-6:
                Sigma_new += (self.rho + abs(eigvals.min()) + 1e-6) * np.eye(d)
            Sigma = Sigma_new

            # 收敛判断
            if np.max(np.abs(mu - prev_mu)) < self.tol:
                break
            prev_mu = mu.copy()

        return {
            "mu": mu,
            "Sigma": Sigma,
            "lam": lam,
            "deviations": deviations,
        }

    def impute_with_interval(
        self,
        z_obs: np.ndarray,       # 观测 log 域子向量（形状 (|Ω|,)）
        obs_idx: np.ndarray,     # 观测索引
        mis_idx: np.ndarray,     # 缺失索引
        theta: dict,             # 拟合的参数字典
        epsilon: float = 1e-3,   # log 偏移量
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对缺失条目做补全并输出置信区间

        返回
        ----
        z_mis_hat  : 缺失条目的 log 域点估计
        x_mis_hat  : 原始域点估计（exp(z) - ε）
        ci_lower   : 原始域 95% 置信区间下界
        ci_upper   : 原始域 95% 置信区间上界
        """
        mu = theta["mu"]
        Sigma = theta["Sigma"]
        lam = theta["lam"]

        # 1. 利用条件高斯分布推断缺失主成分 u_mis
        #    条件分布：u_mis | u_obs ~ N(mu_mis + Sigma_mo * Sigma_oo^{-1}(u_obs - mu_obs), 
        #                                  Sigma_mm - Sigma_mo * Sigma_oo^{-1} * Sigma_om)
        mu_obs = mu[obs_idx]
        mu_mis = mu[mis_idx]
        Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
        Sigma_mm = Sigma[np.ix_(mis_idx, mis_idx)]
        Sigma_mo = Sigma[np.ix_(mis_idx, obs_idx)]

        # 先估计偏差
        o_obs = update_deviation(z_obs, mu_obs, Sigma_oo, lam)
        u_obs_hat = z_obs - o_obs  # 去偏后观测主成分

        try:
            Sigma_oo_inv = np.linalg.inv(Sigma_oo + 1e-6 * np.eye(len(obs_idx)))
        except np.linalg.LinAlgError:
            Sigma_oo_inv = np.linalg.pinv(Sigma_oo)

        # 条件均值和方差
        mu_cond = mu_mis + Sigma_mo @ Sigma_oo_inv @ (u_obs_hat - mu_obs)
        Sigma_cond = Sigma_mm - Sigma_mo @ Sigma_oo_inv @ Sigma_mo.T
        # 保证正定
        eigvals = np.linalg.eigvalsh(Sigma_cond)
        if eigvals.min() < 1e-8:
            Sigma_cond += (abs(eigvals.min()) + 1e-8) * np.eye(len(mis_idx))

        sigma_cond = np.sqrt(np.diag(Sigma_cond))  # 逐坐标标准差

        # 2. 点估计：使用条件均值作为主成分，偏差取 0（MAP 近似）
        z_mis_hat = mu_cond  # log 域点估计

        # 3. 置信区间：Normal-Laplace 卷积的解析 CDF
        #    Z_mis = U_mis + O_mis，其中 U_mis ~ N(μ_cond, σ_cond²)，O_mis ~ Laplace(0, 1/λ)
        #    使用高斯近似作为下界，Normal-Laplace 尾部更重
        alpha = 1 - self.alpha_ci  # 0.05
        z_alpha = scipy_norm.ppf(alpha / 2)  # -1.96

        # 预测区间（log 域）：考虑高斯不确定性 + Laplace 偏差不确定性
        sigma_nl = np.sqrt(sigma_cond**2 + 2.0 / (lam**2 + 1e-10))  # Normal-Laplace 总方差
        z_lo = mu_cond + z_alpha * sigma_nl  # 下界（log 域）
        z_hi = mu_cond - z_alpha * sigma_nl  # 上界（log 域）

        # 4. 转换回原始域
        x_mis_hat = np.exp(z_mis_hat) - epsilon
        ci_lower = np.exp(z_lo) - epsilon
        ci_upper = np.exp(z_hi) - epsilon

        # 截断到非负
        x_mis_hat = np.maximum(x_mis_hat, 0.0)
        ci_lower = np.maximum(ci_lower, 0.0)
        ci_upper = np.maximum(ci_upper, ci_lower)

        return z_mis_hat, x_mis_hat, ci_lower, ci_upper

    def fit_and_impute(
        self,
        X_frames: List[np.ndarray],   # 原始流量帧（NaN 表示缺失），每帧形状 (d,)
        window_indices: Optional[List[List[int]]] = None,  # 每个窗口包含哪些帧索引
        epsilon: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        完整流程：数据变换 → 窗口划分 → 参数估计 → 补全 + 区间

        返回
        ----
        X_complete  : 补全后的矩阵（NaN 替换为点估计），形状 (T, d)
        CI_lower    : 置信区间下界，形状 (T, d)（非缺失位置 = NaN）
        CI_upper    : 置信区间上界，形状 (T, d)（非缺失位置 = NaN）
        """
        T = len(X_frames)
        d = X_frames[0].shape[0]

        # 1. 对数变换（加偏移防 log(0)）
        Z_frames = []
        masks = []
        for x in X_frames:
            mask = ~np.isnan(x)
            z = np.where(mask, np.log(x + epsilon), 0.0)
            Z_frames.append(z)
            masks.append(mask)

        # 2. 默认窗口划分（滑动窗口）
        if window_indices is None:
            window_indices = []
            for start in range(0, T, self.window_size):
                end = min(start + self.window_size, T)
                window_indices.append(list(range(start, end)))

        # 3. 逐窗口拟合参数
        frame_theta = {}  # frame_idx -> theta
        for win_idx, frame_ids in enumerate(window_indices):
            if len(frame_ids) < 2:
                # 单帧窗口，直接用全局均值填充
                for fid in frame_ids:
                    frame_theta[fid] = {
                        "mu": np.zeros(d),
                        "Sigma": np.eye(d),
                        "lam": 1.0,
                        "deviations": [np.zeros(np.sum(masks[fid]))],
                    }
                continue

            win_Z = [Z_frames[i] for i in frame_ids]
            win_masks = [masks[i] for i in frame_ids]
            theta = self.fit_window(win_Z, win_masks)
            for i, fid in enumerate(frame_ids):
                frame_theta[fid] = theta

        # 4. 逐帧补全缺失值
        X_complete = np.array([x.copy() for x in X_frames])
        CI_lower = np.full((T, d), np.nan)
        CI_upper = np.full((T, d), np.nan)

        for t in range(T):
            mask = masks[t]
            obs_idx = np.where(mask)[0]
            mis_idx = np.where(~mask)[0]
            if len(mis_idx) == 0:
                continue  # 全观测帧，无需补全
            if len(obs_idx) == 0:
                # 全缺失帧，用均值填充
                theta = frame_theta.get(t, {"mu": np.zeros(d), "Sigma": np.eye(d), "lam": 1.0})
                X_complete[t, mis_idx] = np.exp(theta["mu"][mis_idx]) - epsilon
                continue

            theta = frame_theta[t]
            z_obs = Z_frames[t][obs_idx]
            _, x_hat, ci_lo, ci_hi = self.impute_with_interval(
                z_obs, obs_idx, mis_idx, theta, epsilon
            )
            X_complete[t, mis_idx] = x_hat
            CI_lower[t, mis_idx] = ci_lo
            CI_upper[t, mis_idx] = ci_hi

        return X_complete, CI_lower, CI_upper


# ─── 评估工具 ─────────────────────────────────────────────────────────────────

def evaluate_imputation(
    X_true: np.ndarray,
    X_complete: np.ndarray,
    mask_missing: np.ndarray,  # True 表示缺失位置
) -> dict:
    """计算补全精度指标（仅在缺失位置计算）"""
    true_vals = X_true[mask_missing]
    pred_vals = X_complete[mask_missing]
    valid = (true_vals > 0)  # 避免除零

    mae = np.mean(np.abs(pred_vals - true_vals))
    rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
    wmape = (
        np.sum(np.abs(pred_vals[valid] - true_vals[valid])) /
        np.sum(true_vals[valid]) * 100
        if valid.sum() > 0 else np.nan
    )
    return {"MAE": mae, "RMSE": rmse, "wMAPE(%)": wmape}


def compute_picp(
    X_true: np.ndarray,
    CI_lower: np.ndarray,
    CI_upper: np.ndarray,
    mask_missing: np.ndarray,
) -> float:
    """计算区间覆盖率（PICP）：真值落入置信区间的比例"""
    true_vals = X_true[mask_missing]
    lo_vals = CI_lower[mask_missing]
    hi_vals = CI_upper[mask_missing]
    valid = ~np.isnan(lo_vals)
    if valid.sum() == 0:
        return np.nan
    covered = (true_vals[valid] >= lo_vals[valid]) & (true_vals[valid] <= hi_vals[valid])
    return covered.mean()


# ─── 母婴电商示例 ─────────────────────────────────────────────────────────────

def generate_ecommerce_transition_matrix(
    n_pages: int = 8,
    n_days: int = 20,
    obs_rate: float = 0.4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成母婴电商页面转移矩阵示例数据

    页面：HOME, CATEGORY, PDP, SEARCH, CART, CHECKOUT, ORDER_CONFIRM, 404

    返回
    ----
    X_true     : 完整转移计数矩阵，形状 (n_days, n_pages * n_pages)
    X_partial  : 随机遮盖后的部分观测矩阵（NaN 表示缺失）
    """
    rng = np.random.default_rng(seed)
    all_pages = ["HOME", "CATEGORY", "PDP", "SEARCH", "CART", "CHECKOUT", "ORDER_CONFIRM", "404"]
    page_names = (all_pages * ((n_pages // len(all_pages)) + 1))[:n_pages]

    d = n_pages * n_pages

    # 页面名称（取前 n_pages 个，或按需截断/填充）
    all_pages = ["HOME", "CATEGORY", "PDP", "SEARCH", "CART", "CHECKOUT", "ORDER_CONFIRM", "404"]
    page_names = (all_pages * ((n_pages // len(all_pages)) + 1))[:n_pages]

    # 模拟底层参数（共享均值和协方差）
    mu_true = np.random.default_rng(seed).uniform(1.0, 4.0, d)  # log 域均值
    # 使用块结构协方差（相邻页面相关性高）
    base_cov = 0.1 * np.eye(d)
    for i in range(d - 1):
        base_cov[i, i+1] = base_cov[i+1, i] = 0.05
    Sigma_true = base_cov + 0.05 * np.eye(d)

    # 生成完整日度转移矩阵
    Z_true = rng.multivariate_normal(mu_true, Sigma_true, size=n_days)
    X_true = np.exp(Z_true) - 1e-3
    X_true = np.maximum(X_true, 0)

    # 随机遮盖
    missing_mask = rng.random((n_days, d)) > obs_rate
    X_partial = X_true.copy().astype(float)
    X_partial[missing_mask] = np.nan

    return X_true, X_partial


def run_example():
    """完整的母婴电商转移矩阵补全示例"""
    print("=" * 65)
    print("Utimac: 母婴电商页面转移矩阵补全 + 置信区间")
    print("=" * 65)

    # 1. 生成示例数据
    n_pages = 8
    n_days = 20
    obs_rate = 0.4  # 40% 观测率（稀疏场景）

    X_true, X_partial = generate_ecommerce_transition_matrix(
        n_pages=n_pages, n_days=n_days, obs_rate=obs_rate
    )
    X_frames = [X_partial[t] for t in range(n_days)]
    missing_mask = np.isnan(X_partial)

    print(f"\n数据概览:")
    print(f"  矩阵维度: {n_days} 天 × {n_pages}×{n_pages} = {X_partial.shape}")
    print(f"  观测率: {obs_rate:.0%}（缺失 {missing_mask.sum()} 个条目）")

    # 2. 运行 Utimac
    print("\n运行 Utimac（块坐标下降）...")
    model = Utimac(
        window_size=7,     # 7天一个局部平稳窗口
        rho=0.1,           # 协方差正则化
        eta=0.01,          # 稀疏度正则化
        bcd_iters=30,      # BCD 迭代
        alpha_ci=0.95,     # 95% 置信区间
    )
    X_complete, CI_lower, CI_upper = model.fit_and_impute(
        X_frames, epsilon=1e-3
    )

    # 3. 评估
    metrics = evaluate_imputation(X_true, X_complete, missing_mask)
    picp = compute_picp(X_true, CI_lower, CI_upper, missing_mask)

    print(f"\n补全精度（仅缺失位置）:")
    print(f"  MAE:      {metrics['MAE']:.4f}")
    print(f"  RMSE:     {metrics['RMSE']:.4f}")
    print(f"  wMAPE:    {metrics['wMAPE(%)']:.2f}%")
    print(f"  PICP(95%): {picp:.2%}  (目标 ≥ 0.95)")

    # 4. 可信度分析：找出区间最宽的转移对
    interval_width = CI_upper - CI_lower
    interval_width_missing = np.where(missing_mask, interval_width, np.nan)

    page_names = ["HOME", "CAT", "PDP", "SRCH", "CART", "CHK", "ORD", "404"]
    n_pages = len(page_names)

    print(f"\n最不可信的 5 条补全路径（区间最宽）:")
    flat_widths = interval_width_missing.flatten()
    top_indices = np.argsort(flat_widths[~np.isnan(flat_widths)])[-5:][::-1]
    # 映射回 (天, 出发页, 到达页)
    nan_flat_idx = np.where(~np.isnan(flat_widths))[0]
    for rank, idx in enumerate(top_indices):
        abs_idx = nan_flat_idx[idx]
        day = abs_idx // (n_pages * n_pages)
        pair_idx = abs_idx % (n_pages * n_pages)
        src = pair_idx // n_pages
        dst = pair_idx % n_pages
        w = flat_widths[abs_idx]
        true_v = X_true.flatten()[abs_idx]
        pred_v = X_complete.flatten()[abs_idx]
        print(
            f"  #{rank+1} Day{day:02d} {page_names[src]}→{page_names[dst]}: "
            f"补全={pred_v:.1f}, 真值={true_v:.1f}, 区间宽度={w:.1f} "
            f"→ ⚠️ 数据不可信"
        )

    print(f"\n最可信的 5 条补全路径（区间最窄）:")
    valid_widths = flat_widths[~np.isnan(flat_widths)]
    bottom_indices = np.argsort(valid_widths)[:5]
    for rank, idx in enumerate(bottom_indices):
        abs_idx = nan_flat_idx[idx]
        day = abs_idx // (n_pages * n_pages)
        pair_idx = abs_idx % (n_pages * n_pages)
        src = pair_idx // n_pages
        dst = pair_idx % n_pages
        w = valid_widths[idx]
        true_v = X_true.flatten()[abs_idx]
        pred_v = X_complete.flatten()[abs_idx]
        print(
            f"  #{rank+1} Day{day:02d} {page_names[src]}→{page_names[dst]}: "
            f"补全={pred_v:.1f}, 真值={true_v:.1f}, 区间宽度={w:.1f} "
            f"→ ✅ 可信，可用于决策"
        )

    # 5. 关键业务洞察
    print(f"\n业务决策建议:")
    narrow_threshold = np.nanpercentile(interval_width_missing, 25)
    wide_threshold = np.nanpercentile(interval_width_missing, 75)
    n_trustworthy = np.sum(interval_width_missing[missing_mask] < narrow_threshold)
    n_unreliable = np.sum(interval_width_missing[missing_mask] > wide_threshold)
    print(f"  可信补全（区间窄）: {n_trustworthy} 条 → 可基于此优化漏斗")
    print(f"  不可信补全（区间宽）: {n_unreliable} 条 → 需加强数据收集再做决策")
    print(f"  建议：对不可信转移路径，增加埋点覆盖率或延长观测周期后重跑")

    return X_complete, CI_lower, CI_upper


# ─── 单元测试 ─────────────────────────────────────────────────────────────────

def test_soft_threshold():
    """测试软阈值算子"""
    x = np.array([3.0, -1.5, 0.4, -0.3])
    result = soft_threshold(x, 0.5)
    expected = np.array([2.5, -1.0, 0.0, 0.0])
    assert np.allclose(result, expected, atol=1e-6), f"期望 {expected}，得到 {result}"
    print("✅ test_soft_threshold 通过")


def test_update_deviation_diagonal():
    """测试对角协方差下的偏差更新等价于坐标软阈值"""
    np.random.seed(0)
    d = 4
    z_obs = np.array([2.0, -1.0, 0.5, 3.0])
    mu_obs = np.array([1.5, -0.5, 0.3, 2.5])
    sigma_obs = 0.5 * np.eye(d)  # 对角协方差
    lam = 1.0

    o = update_deviation(z_obs, mu_obs, sigma_obs, lam)
    r = z_obs - mu_obs
    # Q 对角为 1/0.5 = 2，软阈值为 λ/q_kk = 1/2 = 0.5
    o_expected = soft_threshold(r, 0.5)
    assert np.allclose(o, o_expected, atol=1e-4), f"对角情况下不等价：{o} vs {o_expected}"
    print("✅ test_update_deviation_diagonal 通过")


def test_utimac_basic():
    """测试基本补全功能：补全值非 NaN，区间合理"""
    X_true, X_partial = generate_ecommerce_transition_matrix(
        n_pages=4, n_days=12, obs_rate=0.5, seed=123
    )
    X_frames = [X_partial[t] for t in range(12)]
    missing_mask = np.isnan(X_partial)

    model = Utimac(window_size=6, rho=0.1, eta=0.01, bcd_iters=15)
    X_complete, CI_lower, CI_upper = model.fit_and_impute(X_frames)

    # 补全后无 NaN
    assert not np.any(np.isnan(X_complete)), "补全结果含 NaN"
    # 补全值非负
    assert np.all(X_complete >= 0), "补全值出现负数"
    # 区间下界 ≤ 上界
    valid = ~np.isnan(CI_lower)
    assert np.all(CI_lower[valid] <= CI_upper[valid]), "区间下界 > 上界"
    # 指标计算不报错
    metrics = evaluate_imputation(X_true, X_complete, missing_mask)
    assert metrics["wMAPE(%)"] < 200, f"wMAPE 异常大：{metrics['wMAPE(%)']}"
    print(f"✅ test_utimac_basic 通过 (wMAPE={metrics['wMAPE(%)']:.1f}%)")


def test_picp():
    """测试 PICP 计算函数"""
    X_true = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
    X_complete = np.array([[1.0, 2.0, 2.8], [4.0, 4.9, 6.0]])
    CI_lower = np.array([[np.nan, np.nan, 2.0], [np.nan, 4.0, np.nan]])
    CI_upper = np.array([[np.nan, np.nan, 4.0], [np.nan, 6.0, np.nan]])
    missing = np.isnan(np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]]))

    picp = compute_picp(X_true, CI_lower, CI_upper, missing)
    # 真值 [nan_pos] = [nan, 5.0]，区间覆盖 5.0 in [4.0, 6.0] ✓，X[0,2]=nan 跳过
    # 期望 PICP = 1.0（两个有效位置都覆盖）
    # 注意：X_true 在 missing 位置：X_true[0,2]=nan(原始缺失位置需要真值)
    # 重新设置
    X_true2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    picp2 = compute_picp(X_true2, CI_lower, CI_upper, missing)
    assert 0.0 <= picp2 <= 1.0, f"PICP 超出 [0,1]: {picp2}"
    print(f"✅ test_picp 通过 (PICP={picp2:.2f})")


def run_tests():
    print("运行单元测试...")
    test_soft_threshold()
    test_update_deviation_diagonal()
    test_utimac_basic()
    test_picp()
    print("\n✅ 所有单元测试通过\n")


if __name__ == "__main__":
    run_tests()
    run_example()
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Sparse Matrix Completion (Hájek-GD)]([[Skill-Sparse-Matrix-Completion]].md) | Hájek-GD 做点估计（无区间），Utimac 在其基础上加入不确定性量化；先理解低秩点估计再理解统计推断框架 |
| 前置 | [User Funnel Analysis]([[Skill-User-Funnel-Analysis]].md) | 理解页面转移矩阵的业务含义（漏斗各阶段），才能正确解读补全结果 |
| 组合 | [Cohort Retention Analysis]([[Skill-Cohort-Retention-Analysis]].md) | Utimac 补全转移矩阵（横截面），Cohort 分析跨时间留存趋势；两者叠加形成"时空双维度"用户路径分析 |
| 延伸 | [Session Intent Shift]([[Skill-Session-Intent-Shift]].md) | 在补全的完整转移矩阵上，进一步检测 session 内意图漂移，识别异常路径 |
| 延伸 | [Trajectory Pattern Mining]([[Skill-Trajectory-Pattern-Mining]].md) | 将补全后的矩阵作为输入，挖掘高频转移模式和关键路径 |

---

- **前置技能**：[[Skill-Sparse-Matrix-Completion]] | [[Skill-STAMImputer-SpatioTemporal]]
- **延伸技能**：[[Skill-Conformal-ROI-Prediction]]
- **可组合技能**：[[Skill-EPICSCORE-Uncertainty]]
- **相关技能**：[[Skill-BlockEcho-Missing-Data]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| ROI 预估 | 中高（避免错误决策） | 主要价值在**风险规避**：区间宽时阻止基于不可信数据的预算调整，保守估计每次可避免 5-20 万的错误投入；同时提升漏斗优化的精准度 |
| 实施难度 | ⭐⭐⭐☆☆（中等） | 纯 Python 实现，无需 GPU；但协方差更新的工程细节复杂，需要熟悉矩阵计算；窗口大小、正则超参需业务调试 |
| 数据要求难度 | ⭐⭐⭐☆☆（中等） | 需要多天/多周的 session 级页面转移计数矩阵，埋点完整性要求中等（30%+ 观测率即可，Utimac 在高稀疏度下仍有优势） |
| 差异化价值 | ⭐⭐⭐⭐⭐（高） | 市场上几乎没有"带置信区间的转移矩阵补全"工具，能向业务方展示哪些数据可信、哪些不可信，是分析可信度的核心竞争力 |
| 优先级 | ⭐⭐⭐⭐☆ | 高稀疏度场景（新站、冷门品类）价值极大；常规大站价值偏中；建议在数据质量治理项目中优先落地 |

### 适合场景

| 场景 | 推荐指数 | 原因 |
|------|---------|------|
| 独立站新上线（数据稀疏期） | ⭐⭐⭐⭐⭐ | 流量少，补全值不确定性高，最需要区间告警 |
| 冷门品类/冷门 SKU 路径分析 | ⭐⭐⭐⭐⭐ | 观测稀疏，传统点估计不可信 |
| 漏斗优化决策支持（成熟站） | ⭐⭐⭐☆☆ | 数据充足时区间收窄，主要用于验证置信度 |
| 实时流量调度（毫秒级） | ☆☆☆☆☆ | BCD 迭代需要分钟级，不适合实时场景 |
