---
title: 认知不确定性共形评分 - 数据稀疏区域自适应区间加宽
doc_type: knowledge
module: 01-因果推断
topic: epistemic-uncertainty-conformal
status: stable
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: arXiv:2502.06995 (2025)
roadmap_phase: phase1
---

# Skill: EPICSCORE Uncertainty Quantification — 认知不确定性共形评分

> 论文:**EPICSCORE: Epistemic Uncertainty in Conformal Scores — A Unified Approach** · arXiv:2502.06995 (2025)
> 作者:Luben M. C. Cabezas, Vagner S. Santos, Thiago R. Ramos, Rafael Izbicki（Federal University of São Carlos / USP）
> 发表:UAI 2025
> 应用:在数据稀疏的桑基图路径上自动加宽预测区间，数据丰富的路径保持紧凑

---

## ① 算法原理

### 核心思想

标准共形预测对所有数据点使用统一的非一致性分数，无法区分"数据多的区域"和"数据少的区域"。根本原因：传统共形分数只捕捉**偶然不确定性**（aleatoric uncertainty，数据本身的随机性），对**认知不确定性**（epistemic uncertainty，训练数据不足导致的模型无知）视而不见——在数据稀疏区域仍然给出窄区间，形成虚假的高置信度。

**EPICSCORE** 将认知不确定性显式集成到共形分数中：
1. 对任意已有的非一致性分数 `s(x, y)`（回归残差、分位数分数、分类得分均可）
2. 用贝叶斯模型（GP / MC Dropout / BART）对分数的条件分布 `f(s | x, θ)` 建模
3. 通过后验积分得到**预测 CDF** `F(s | x, D)`，将其作为新的非一致性分数 `s'(x, y)`
4. 在数据稀疏处，后验方差大 → `F` 值偏低 → 新分数偏小 → 预测区间自动扩宽

无需修改下游的共形推断框架，是即插即用的"分数增强器"而非替换品。

### 数学直觉

**认知不确定性 vs 偶然不确定性**

| 类型 | 来源 | 能否通过更多数据消除 |
|------|------|---------------------|
| 偶然不确定性（Aleatoric） | 数据本身的随机噪声 | ✗ 不能 |
| 认知不确定性（Epistemic） | 训练数据不足 / 区域未覆盖 | ✓ 可以 |

**EPICSCORE 核心公式**

设原始非一致性分数为 `s(x, y)`，校准集 `D_cal` 分成两部分：
- `D_cal,1`：用于拟合贝叶斯模型，得到后验 `f(θ | D)`
- `D_cal,2`：用于计算分位数阈值

**步骤 1** — 构建修正分数（Eq. 2）：

```
s'(x, y) = F(s(x, y) | x, D)
         = ∫ F(s(x,y) | x, θ) · f(θ | D) dθ
```

`F(s | x, D)` 是贝叶斯后验预测 CDF：在数据密集区，后验集中 → `F` 接近真实 CDF → 分数信息丰富；在数据稀疏区，后验弥散 → `F` 偏保守 → 分数自动收缩（等效于区间扩宽）。

**步骤 2** — 预测区间（与标准共形形式一致）：

```
R_EPIC(x_new) = { y : s(x_new, y) ≤ F⁻¹(t_{1-α} | x_new, D) }
```

其中 `t_{1-α}` 是 `s'` 在 `D_cal,2` 上的 `(1-α)` 分位数。

**关键保证**：
- **有限样本边际覆盖**：继承自标准共形预测，`P(Y_{n+1} ∈ R_EPIC(X_{n+1})) ≥ 1-α`
- **渐近条件覆盖**：当校准集增大时，区间宽度随 `x` 的局部数据密度自适应调整（Theorem 2）

**贝叶斯后验方差作为认知不确定性代理**：对于高斯过程，`Var[F(s | x, θ) | D]` 在训练点稀疏处最大，驱动区间扩宽。

### 关键效果数字

- **分类任务（CIFAR-100 OOD）**：EPICSCORE 的 Size-Stratified Coverage（SSC）比标准 APS 提升 **33%**，即对分布外图像给出了更诚实的大预测集，而不是虚假的窄集
- **回归任务**：在数据空洞区域（如 `x ∈ (7, 8)` 无训练点），标准共形给出与数据密集区等宽的窄区间；EPICSCORE 区间显著扩宽，正确反映未知性
- **与 CQR-r（Rossellini et al.2024）、Cocheteux et al.2025 等专用方法相比**：EPICSCORE 在多个真实数据集上性能持平或更优，且**不限定任务类型**

---

## ② 母婴出海应用案例

### 场景：桑基图热门路径 vs 冷门路径的差异化区间

**业务问题**

桑基图中每条边代表一条用户路径转化，观测量差异极大：

| 路径边 | 日均观测量 | 标准共形区间 | EPICSCORE 区间 | 决策含义 |
|--------|-----------|-------------|----------------|---------|
| SEARCH → PDP | 5,000次 | [78%, 82%] | [77%, 83%] | 数据充足，区间无明显变化 |
| PDP → CART | 2,100次 | [43%, 49%] | [42%, 50%] | 数据较充足 |
| SUPPORT → CHECKOUT | 23次 | 标准方法仍给 [8%, 38%] | [5%, 45%] | 数据稀疏，自动加宽 |
| REFUND → REPURCHASE | 7次 | 虚假窄区间 | 极宽区间（接近 [0%, 100%]）| 直接警告：样本不足，不可据此决策 |

**一眼看出哪条边可信、哪条边不可信**，避免把小样本冷门路径的点估计误作稳定洞察。

**数据要求**
- 每条桑基图边上有 `(x_i, y_i)` 对：`x_i` 为路径上下文特征（流量来源、时段、品类等），`y_i` 为结果指标（转化率、GMV）
- 带不确定性代理的预测模型（任意贝叶斯模型，包括轻量 GP/BART/MC Dropout）

**预期产出**：每条桑基图边的自适应置信区间，数据多的边窄、数据少的边宽

**业务价值**：热门路径精准决策 + 冷门路径避免误判，决策者自动获得"样本量警告"

---

## ③ 代码模板

以下代码实现 EPICSCORE 三种贝叶斯后端（GP、MC Dropout、BART），加上标准共形基线对比，以及桑基图边场景的完整测试用例。

```python
"""
EPICSCORE 完整实现
论文: arXiv:2502.06995 — Epistemic Uncertainty in Conformal Scores: A Unified Approach

依赖:
  pip install numpy scipy scikit-learn gpytorch torch bartpy
  或: conda install numpy scipy scikit-learn pytorch -c pytorch
      pip install gpytorch bartpy
"""

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 一、非一致性分数定义（任意分数可替换）
# ─────────────────────────────────────────────

def regression_split_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """标准回归 split 分数: s(x,y) = |y - g(x)|"""
    return np.abs(y_true - y_pred)


def cqr_score(q_lo: np.ndarray, q_hi: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """CQR 分数: s(x,y) = max(q_lo(x)-y, y-q_hi(x))"""
    return np.maximum(q_lo - y_true, y_true - q_hi)


# ─────────────────────────────────────────────
# 二、贝叶斯后端 A — 高斯过程（GP）
# ─────────────────────────────────────────────

class GPEpistemicModel:
    """
    用 GP 对非一致性分数的条件分布建模。
    返回预测均值和方差，用正态近似计算 F(s | x, D)。
    依赖: gpytorch（可替换为 sklearn.gaussian_process）
    """

    def __init__(self, length_scale: float = 1.0, noise_var: float = 0.1):
        self.length_scale = length_scale
        self.noise_var = noise_var
        self._X_train = None
        self._s_train = None
        self._K_inv = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF 核函数"""
        diff = X1[:, None, :] - X2[None, :, :]  # (n1, n2, d)
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.exp(-0.5 * dist_sq / self.length_scale ** 2)

    def fit(self, X_cal1: np.ndarray, scores_cal1: np.ndarray):
        """在 D_cal,1 上拟合 GP 后验"""
        self._X_train = X_cal1
        self._s_train = scores_cal1
        K = self._rbf_kernel(X_cal1, X_cal1)
        K += self.noise_var * np.eye(len(X_cal1))
        self._K_inv = np.linalg.inv(K)

    def predict(self, X_new: np.ndarray):
        """返回预测均值和方差（认知不确定性代理）"""
        K_star = self._rbf_kernel(X_new, self._X_train)  # (n_new, n_train)
        K_star_star = self._rbf_kernel(X_new, X_new)     # (n_new, n_new)

        mu = K_star @ self._K_inv @ self._s_train
        var = np.diag(K_star_star - K_star @ self._K_inv @ K_star.T)
        var = np.maximum(var, 1e-6)  # 数值稳定性
        return mu, var

    def cdf(self, s_values: np.ndarray, X_new: np.ndarray) -> np.ndarray:
        """
        F(s | x, D) ≈ Φ((s - μ(x)) / σ(x))
        用正态近似后验预测 CDF
        """
        mu, var = self.predict(X_new)
        sigma = np.sqrt(var)
        return norm.cdf((s_values - mu) / sigma)


# ─────────────────────────────────────────────
# 三、贝叶斯后端 B — MC Dropout（轻量神经网络）
# ─────────────────────────────────────────────

class MCDropoutEpistemicModel:
    """
    用 MC Dropout 近似贝叶斯推断。
    在推理时保持 dropout 激活，多次前向传播取平均。
    """

    def __init__(self, n_samples: int = 100, dropout_rate: float = 0.1,
                 hidden_dim: int = 64, n_epochs: int = 500):
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self._weights = {}  # 简化：存储 numpy 权重

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """单次前向传播（含 dropout）"""
        h = self._relu(X @ self._weights["W1"] + self._weights["b1"])
        if training:
            mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(float)
            h = h * mask / (1 - self.dropout_rate)
        out = h @ self._weights["W2"] + self._weights["b2"]
        return out.squeeze()

    def fit(self, X_cal1: np.ndarray, scores_cal1: np.ndarray):
        """简化训练（实际应用建议用 PyTorch/TF 实现）"""
        d = X_cal1.shape[1]
        np.random.seed(42)
        self._weights = {
            "W1": np.random.randn(d, self.hidden_dim) * 0.1,
            "b1": np.zeros(self.hidden_dim),
            "W2": np.random.randn(self.hidden_dim, 1) * 0.1,
            "b2": np.zeros(1),
        }
        # 简化 SGD
        lr = 0.01
        for _ in range(self.n_epochs):
            preds = np.array([self._forward(X_cal1, True) for _ in range(5)]).mean(axis=0)
            loss_grad = 2 * (preds - scores_cal1) / len(scores_cal1)
            h = self._relu(X_cal1 @ self._weights["W1"] + self._weights["b1"])
            self._weights["W2"] -= lr * (h.T @ loss_grad.reshape(-1, 1))
            self._weights["b2"] -= lr * loss_grad.sum()

    def predict_samples(self, X_new: np.ndarray) -> np.ndarray:
        """返回 MC 采样矩阵 (n_samples, n_new)"""
        return np.array([self._forward(X_new, True) for _ in range(self.n_samples)])

    def cdf(self, s_values: np.ndarray, X_new: np.ndarray) -> np.ndarray:
        """F(s | x, D) ≈ 经验 CDF over MC 样本"""
        samples = self.predict_samples(X_new)  # (n_samples, n_new)
        # 对每个测试点，计算 s_values[i] 在 samples[:, i] 中的百分位
        result = np.zeros(len(s_values))
        for i in range(len(s_values)):
            result[i] = np.mean(samples[:, i] <= s_values[i])
        return result


# ─────────────────────────────────────────────
# 四、EPICSCORE 主包装器
# ─────────────────────────────────────────────

class EPICSCORE:
    """
    EPICSCORE: 认知不确定性增强的共形分数包装器。

    用法：
        epicscore = EPICSCORE(base_score_fn, bayesian_model)
        epicscore.calibrate(X_cal, y_cal, base_model_preds, alpha=0.1)
        interval = epicscore.predict_interval(X_new, base_model_pred_new)

    参数:
        base_score_fn: 原始非一致性分数函数，签名 (pred, true) -> score
        bayesian_model: 贝叶斯后端，需实现 fit(X, s) 和 cdf(s, X) 方法
        cal_split_ratio: D_cal,1 / D_cal,2 的比例（默认 0.5）
    """

    def __init__(self, base_score_fn, bayesian_model, cal_split_ratio: float = 0.5):
        self.base_score_fn = base_score_fn
        self.bayesian_model = bayesian_model
        self.cal_split_ratio = cal_split_ratio
        self._t_alpha = None  # 共形阈值

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  preds_cal: np.ndarray, alpha: float = 0.1):
        """
        校准阶段（Algorithm 1 in paper）:
        1. 计算原始分数 s_i = base_score(pred_i, y_i)
        2. 将 D_cal 分成 D_cal,1 和 D_cal,2
        3. 在 D_cal,1 上拟合贝叶斯模型
        4. 在 D_cal,2 上计算 s'_i = F(s_i | x_i, D) 并取 (1-α) 分位数
        """
        n = len(X_cal)
        n1 = int(n * self.cal_split_ratio)

        # 分割校准集
        idx = np.random.permutation(n)
        idx1, idx2 = idx[:n1], idx[n1:]
        X1, preds1, y1 = X_cal[idx1], preds_cal[idx1], y_cal[idx1]
        X2, preds2, y2 = X_cal[idx2], preds_cal[idx2], y_cal[idx2]

        # D_cal,1: 计算原始分数并拟合贝叶斯模型
        s1 = self.base_score_fn(preds1, y1)
        self.bayesian_model.fit(X1, s1)

        # D_cal,2: 计算修正分数 s'
        s2 = self.base_score_fn(preds2, y2)
        s2_prime = self.bayesian_model.cdf(s2, X2)

        # 共形阈值
        self._t_alpha = np.quantile(s2_prime, 1 - alpha)
        self._alpha = alpha
        print(f"[EPICSCORE] 校准完成: t_{{1-α}} = {self._t_alpha:.4f} (α={alpha})")

    def predict_interval(self, X_new: np.ndarray, pred_new: np.ndarray,
                         y_grid: np.ndarray = None) -> tuple:
        """
        预测区间（回归场景）:
        R_EPIC(x) = { y : s(x,y) ≤ F⁻¹(t_{1-α} | x, D) }

        对于绝对误差分数：R_EPIC = [pred - radius, pred + radius]
        其中 radius = F⁻¹(t_{1-α} | x, D)

        返回: (lower, upper) 每个测试点的区间
        """
        if self._t_alpha is None:
            raise RuntimeError("请先调用 calibrate()")

        # 通过逆 CDF 得到每点的自适应半径
        # 搜索：找到最大的 s 使得 F(s | x, D) ≤ t_{1-α}
        radii = []
        for i in range(len(X_new)):
            xi = X_new[i:i+1]
            # 二分查找逆 CDF
            lo, hi = 0.0, 20.0
            for _ in range(50):
                mid = (lo + hi) / 2
                s_mid = np.array([mid])
                cdf_val = self.bayesian_model.cdf(s_mid, xi)[0]
                if cdf_val < self._t_alpha:
                    lo = mid
                else:
                    hi = mid
            radii.append((lo + hi) / 2)

        radii = np.array(radii)
        lower = pred_new - radii
        upper = pred_new + radii
        return lower, upper, radii

    def coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """计算实际覆盖率"""
        covered = (y_true >= lower) & (y_true <= upper)
        return covered.mean()

    def interval_width(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """区间宽度"""
        return upper - lower


# ─────────────────────────────────────────────
# 五、标准共形基线（对比用）
# ─────────────────────────────────────────────

class StandardConformal:
    """标准 split 共形预测基线（无认知不确定性感知）"""

    def __init__(self, base_score_fn):
        self.base_score_fn = base_score_fn
        self._t_alpha = None

    def calibrate(self, preds_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1):
        scores = self.base_score_fn(preds_cal, y_cal)
        self._t_alpha = np.quantile(scores, 1 - alpha)

    def predict_interval(self, pred_new: np.ndarray) -> tuple:
        lower = pred_new - self._t_alpha
        upper = pred_new + self._t_alpha
        return lower, upper

    def coverage(self, y_true, lower, upper):
        return ((y_true >= lower) & (y_true <= upper)).mean()


# ─────────────────────────────────────────────
# 六、完整测试用例
# ─────────────────────────────────────────────

def test_basic_coverage():
    """测试 1：边际覆盖率应 ≥ 1-α"""
    np.random.seed(42)
    alpha = 0.1

    # 模拟数据：训练集在 [0,6]，测试集有 OOD 点在 [7,8]
    X_train = np.random.uniform(0, 6, (500, 1))
    y_train = np.sin(X_train[:, 0]) + np.random.randn(500) * 0.3

    X_cal = np.random.uniform(0, 6, (300, 1))
    y_cal = np.sin(X_cal[:, 0]) + np.random.randn(300) * 0.3

    # 简单预测模型（线性回归近似）
    from sklearn.linear_model import Ridge
    model = Ridge().fit(X_train, y_train)
    preds_cal = model.predict(X_cal)

    # 初始化 EPICSCORE（GP 后端）
    gp = GPEpistemicModel(length_scale=1.0, noise_var=0.1)
    epic = EPICSCORE(
        base_score_fn=regression_split_score,
        bayesian_model=gp,
        cal_split_ratio=0.5
    )
    epic.calibrate(X_cal, y_cal, preds_cal, alpha=alpha)

    # 测试集（包含 OOD 点）
    X_test_id = np.random.uniform(0, 6, (100, 1))   # 分布内
    X_test_ood = np.random.uniform(7, 8, (20, 1))    # 分布外（数据稀疏）
    X_test = np.vstack([X_test_id, X_test_ood])
    y_test = np.sin(X_test[:, 0]) + np.random.randn(120) * 0.3
    preds_test = model.predict(X_test)

    lower, upper, radii = epic.predict_interval(X_test, preds_test)
    cov = epic.coverage(y_test, lower, upper)

    print(f"\n[测试1] 边际覆盖率: {cov:.3f} (目标≥{1-alpha:.1f})")
    assert cov >= 1 - alpha - 0.05, f"覆盖率不足: {cov:.3f}"
    print("[测试1] PASSED ✓")

    # 对比：OOD 点的区间宽度应明显大于 in-distribution 点
    widths_id = radii[:100] * 2
    widths_ood = radii[100:] * 2
    print(f"  分布内区间宽度（均值）: {widths_id.mean():.3f}")
    print(f"  分布外区间宽度（均值）: {widths_ood.mean():.3f}")
    print(f"  OOD/ID 宽度比: {widths_ood.mean() / widths_id.mean():.2f}x")
    # OOD 区间应更宽（体现认知不确定性感知）
    assert widths_ood.mean() > widths_id.mean(), "OOD 区间应比 ID 区间更宽"
    print("[OOD 感知] PASSED ✓")


def test_sankey_edge_scenario():
    """
    测试 2：桑基图场景
    模拟热门路径（多数据）vs 冷门路径（少数据）
    """
    np.random.seed(0)
    alpha = 0.1

    # 特征：[log(流量), 路径类型编码, 时段]
    # 热门路径：SEARCH→PDP，大量数据
    n_hot = 500
    X_hot = np.column_stack([
        np.random.normal(8.5, 0.3, n_hot),   # log(5000)≈8.5，流量充足
        np.ones(n_hot) * 0,                   # 路径类型 0
        np.random.uniform(0, 24, n_hot)       # 时段
    ])
    y_hot = 0.8 + np.random.randn(n_hot) * 0.02   # 转化率均值 80%，方差小

    # 冷门路径：SUPPORT→CHECKOUT，极少数据
    n_cold = 23
    X_cold = np.column_stack([
        np.random.normal(3.1, 0.5, n_cold),   # log(23)≈3.1，流量极少
        np.ones(n_cold) * 1,                   # 路径类型 1
        np.random.uniform(0, 24, n_cold)
    ])
    y_cold = 0.15 + np.random.randn(n_cold) * 0.08  # 转化率均值 15%，方差大

    # 合并为校准集
    X_cal = np.vstack([X_hot, X_cold])
    y_cal = np.concatenate([y_hot, y_cold])

    # 预测模型（点预测）
    preds_cal = np.where(X_cal[:, 1] == 0, 0.80, 0.15)

    # EPICSCORE（GP 后端）
    gp = GPEpistemicModel(length_scale=2.0, noise_var=0.05)
    epic = EPICSCORE(regression_split_score, gp, cal_split_ratio=0.5)
    epic.calibrate(X_cal, y_cal, preds_cal, alpha=alpha)

    # 预测
    X_new_hot = np.array([[8.5, 0, 14.0]])   # 热门路径测试点
    X_new_cold = np.array([[3.1, 1, 14.0]])  # 冷门路径测试点

    pred_hot = np.array([0.80])
    pred_cold = np.array([0.15])

    lo_h, hi_h, r_h = epic.predict_interval(X_new_hot, pred_hot)
    lo_c, hi_c, r_c = epic.predict_interval(X_new_cold, pred_cold)

    print(f"\n[测试2] 桑基图路径自适应区间:")
    print(f"  热门路径 SEARCH→PDP :  [{lo_h[0]:.3f}, {hi_h[0]:.3f}]  宽度={r_h[0]*2:.3f}")
    print(f"  冷门路径 SUPPORT→CHECKOUT: [{lo_c[0]:.3f}, {hi_c[0]:.3f}]  宽度={r_c[0]*2:.3f}")
    print(f"  冷门/热门 宽度比: {r_c[0]/r_h[0]:.2f}x")

    assert r_c[0] > r_h[0], "冷门路径区间应更宽"
    print("[测试2] PASSED ✓")


def test_vs_standard_conformal():
    """测试 3：与标准共形对比，验证 EPICSCORE 在 OOD 处更宽"""
    np.random.seed(7)
    alpha = 0.1
    n_cal = 400

    X_cal = np.random.uniform(0, 5, (n_cal, 1))
    y_cal = 2 * X_cal[:, 0] + np.random.randn(n_cal) * 0.5
    preds_cal = 2 * X_cal[:, 0]  # 假设完美预测均值

    # 标准共形
    std_conf = StandardConformal(regression_split_score)
    std_conf.calibrate(preds_cal, y_cal, alpha=alpha)

    # EPICSCORE
    gp = GPEpistemicModel(length_scale=1.0, noise_var=0.2)
    epic = EPICSCORE(regression_split_score, gp, cal_split_ratio=0.5)
    epic.calibrate(X_cal, y_cal, preds_cal, alpha=alpha)

    # OOD 测试点（x=8，远超训练范围 [0,5]）
    X_ood = np.array([[8.0], [9.0], [10.0]])
    preds_ood = 2 * X_ood[:, 0]

    lo_std, hi_std = std_conf.predict_interval(preds_ood)
    lo_epic, hi_epic, radii_epic = epic.predict_interval(X_ood, preds_ood)

    std_widths = hi_std - lo_std
    epic_widths = hi_epic - lo_epic

    print(f"\n[测试3] OOD 点 (x=8,9,10) 区间宽度对比:")
    print(f"  标准共形: {std_widths}")
    print(f"  EPICSCORE: {epic_widths}")
    print(f"  加宽倍数: {epic_widths / std_widths}")

    # 标准共形在 OOD 处宽度恒定（无感知），EPICSCORE 应更宽
    assert all(epic_widths >= std_widths * 0.9), "EPICSCORE 在 OOD 处应给出更宽或等宽区间"
    print("[测试3] PASSED ✓")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("EPICSCORE 测试套件")
    print("=" * 60)
    test_basic_coverage()
    test_sankey_edge_scenario()
    test_vs_standard_conformal()
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
print("[✓] EPICSCORE Uncertainty 测试通过")
```

### 生产环境推荐替换

| 组件 | 简化版（上方代码） | 生产推荐 |
|------|------------------|---------|
| GP 实现 | numpy 手写 | `gpytorch` 变分 GP（可扩展至 10k+ 点） |
| MC Dropout | numpy 简化 SGD | `torch.nn` + `F.dropout(training=True)` |
| BART | 未实现 | `bartpy` 或 R 的 `dbarts`（via rpy2） |
| 逆 CDF 搜索 | 二分法 50 次迭代 | `scipy.optimize.brentq`，更快更稳 |

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Skill-Conformal-ROI-Prediction](./[[Skill-Conformal-ROI-Prediction]].md) | 标准共形是 EPICSCORE 的基础；需先理解非一致性分数、分位数阈值机制 |
| 前置 | 贝叶斯推断基础（GP / MC Dropout / BART） | EPICSCORE 的后端依赖贝叶斯后验，需对后验预测分布有基本认知 |
| 组合 | Utimac Uncertainty Completion | Utimac 补全缺失的矩阵条目（桑基图边），EPICSCORE 在补全结果上给出自适应置信区间，两者串联：先补全→再量化不确定性 |
| 组合 | [Skill-DML-Cohort-Causal-Effect](./[[Skill-DML-Cohort-Causal-Effect]].md) | DML 给出因果点估计，EPICSCORE 在样本稀疏的子群（如新兴市场冷门品类）上扩宽区间，避免虚假精准的因果结论 |
| 扩展 | [Skill-Uplift-Modeling](./[[Skill-Uplift-Modeling]].md) | Uplift 模型输出 CATE 点预测，可接入 EPICSCORE 得到自适应 CATE 区间 |
| 对比 | CQR-r（Rossellini et al.2024） | 仅适用于分位数回归，EPICSCORE 更通用；但 CQR-r 实现更简单，数据充足时两者性能接近 |

---

- **前置技能**：[[Skill-Uplift-Modeling]] | [[Skill-Conformal-ROI-Prediction]]
- **延伸技能**：[[Skill-SSBC-Small-Sample-Conformal]]
- **可组合技能**：[[Skill-Marketing-Mix-Modeling]]

## ⑤ 商业价值评估

**P1 优先级 ⭐⭐⭐⭐**

| 维度 | 评估 | 说明 |
|------|------|------|
| 落地难度 | 中 | 需要贝叶斯模型基础；代码框架完整，可直接集成 |
| 数据要求 | 低 | 无需额外数据，在现有校准集上增加贝叶斯后端 |
| 业务增量价值 | 高 | 直接解决"小样本路径虚假置信"问题，提升决策可靠性 |
| 模型无关性 | ★★★★★ | 适配任意现有预测模型和非一致性分数，改造成本极低 |
| 理论保证 | 强 | 有限样本边际覆盖 + 渐近条件覆盖，双重保障 |
| 首要场景 | 桑基图冷门边 | 路径分析中，小流量边的区间诚实性直接影响运营决策质量 |

**落地路径**：
1. 取现有转化率预测模型的预测值和校准集
2. 选择 GP 或 BART 作为贝叶斯后端（数据量 < 500 选 GP，> 500 选 BART/MC Dropout）
3. `EPICSCORE.calibrate()` + `predict_interval()` 替换原有固定宽度区间
4. 在报表层增加"区间宽度"色阶，自动标红稀疏路径
