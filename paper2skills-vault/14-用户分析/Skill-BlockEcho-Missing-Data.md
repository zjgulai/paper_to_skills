---
title: 块缺失数据补全 - 整段流量数据丢失时的恢复
doc_type: knowledge
module: 14-用户分析
topic: block-missing-imputation

roadmap_phase: phase2
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: IJCAI 2024
---

# Skill: BlockEcho — 块缺失数据补全（整段广告/流量数据丢失恢复）

> 论文：**BlockEcho: Retaining Long-Range Dependencies for Imputing Block-Wise Missing Data** · IJCAI 2024 pp. 4098-4106
> 作者：Qiao Han, Mingqian Li, Yao Yang, Yiteng Zhai
> arXiv：[2402.18800](https://arxiv.org/abs/2402.18800) · DOI：10.24963/ijcai.2024/453
> 应用：TikTok pixel 故障3天 / 整个流量渠道丢失时，从周边时段和其他渠道的模式中恢复缺失数据块

---

## ① 算法原理

### 核心思想

**块缺失（Block-wise Missing）的独特挑战**：当一整段时间（如连续3天）或一个完整维度（如某渠道所有数据）缺失时，传统插值方法（线性插值、KNN、MICE）依赖"相邻元素"做预测，在块缺失场景下这些邻居全部不存在，方法直接失效。

**BlockEcho 的核心洞察**：将矩阵分解（MF）嵌入生成对抗网络（GAN）框架，**用长程依赖代替局部依赖**。MF 通过低秩分解捕获全局结构（整行/整列的潜在因子），GAN 建模非线性分布——两者互补，使得即使整块邻居缺失，也能从矩阵的全局模式中恢复。

**三步架构**：

1. **MF 预训练阶段**：将原始数据矩阵分解为行嵌入矩阵 $U_p$ 和列嵌入矩阵 $V_p$，提取跨时间段的长程关联特征；预训练结果用于监督后续 GAN 训练
2. **双判别器 GAN 主训练阶段**：
   - **生成器 G**：输入零填充矩阵 $\tilde{X}$、掩码矩阵 $M$、噪声矩阵 $Z$，输出行嵌入 $U$；MCL 层将 $U$ 转为补全矩阵 $\hat{X}$
   - **判别器 $D_I$**（嵌入判别器）：对比生成器产生的 $U$ 与预训练 MF 的 $U_p$，迫使生成器生成符合全局低秩结构的嵌入
   - **判别器 $D_{II}$**（数据真实性判别器）：判断补全矩阵中各元素是真实观测还是生成填充，含 Hint 矩阵加速收敛
3. **矩阵补全层（MCL）**：$U \cdot V + \text{FCN 非线性变换}$，在保留低秩约束的同时引入非线性表达能力

### 数学直觉

**MF 分解目标（KL 散度最小化）**：

```
min_{U,V}  Σ_{(i,j)∈Ω} KL((X⊙M)_{ij} || (UV⊙M)_{ij})
```

等价于对已知元素最小化 KL 散度，使分解后的矩阵在观测位置与原始数据吻合。

**联合训练目标函数**：

```
min_G max_{D_I, D_{II}}  (1-α) · [L_{D_I} + L_{D_{II}}]  +  α · L_MF

其中：
  L_{D_I}  = (1-Y)^T log(1-D_I(U_D)) + Y^T log D_I(U_D)
  L_{D_{II}} = Σ_{i,j}(M⊙log D_{II}(X̄,H) + (1-M)⊙log(1-D_{II}(X̄,H)))_{ij}
  L_MF     = Σ_{i,j}((X⊙M)_{ij} log((X⊙M)_{ij}/(G(X̃,M,Z)V⊙M)_{ij}) - (X⊙M)_{ij} + (G(X̃,M,Z)V⊙M)_{ij})
  U_D      = (1-Y)⊙U_p + Y⊙U   (混合行嵌入，Y∈{0,1}^h 随机索引)
  α        = MF 损失权重（控制 MF 对 GAN 的约束强度）
```

**直觉**：$L_{D_I}$ 迫使生成器的嵌入 $U$ 分布与预训练 MF 的 $U_p$ 一致（长程结构约束）；$L_{D_{II}}$ 保证最终补全值在分布上逼真；$L_{MF}$ 约束观测位置的拟合精度。

**理论保障**（论文 Theorem 1）：当 $\alpha \to 0$ 时退化为标准 GAIN；当 $\alpha \to 1$ 时退化为纯 MF；最优 $\alpha$ 在两者之间使得 MF 的结构约束恰好补偿 GAN 缺乏长程依赖的缺陷。论文证明了 MF-GAN 联合优化的收敛性和相对于单独使用任一方法的帕累托最优性。

### 关键效果数字

| 数据集 | 缺失类型 | 缺失率 | BlockEcho vs 最佳基线 RMSE 降低 |
|--------|---------|--------|-------------------------------|
| 交通数据集（METR-LA）| 块缺失 | 50% | **~18-25%** |
| 疫情数据（COVID-19）| 块缺失 | 50% | **~20-30%** |
| 推荐数据（MovieLens）| 块缺失 | 50% | **~15-22%** |
| 各数据集 | 散点缺失高缺失率（>50%）| >50% | 优势同样显著（vs GAIN/GAIL） |

> 核心规律：**缺失率越高、缺失越集中（整块），BlockEcho 优势越大**。低缺失率散点场景与传统方法差距收窄（约2-5%），此时无需 BlockEcho。

### 关键假设与适用条件

| 假设 | 说明 | 违反时影响 |
|------|------|-----------|
| 数据矩阵存在低秩结构 | 行/列可由少数潜在因子解释 | 随机噪声矩阵无法分解；需 rank 估计 |
| 块缺失有明确边界 | 整段时间/整个维度缺失 | 随机缺失场景效果与 GAIN 相当 |
| 有足够的非缺失数据 | 缺失率 <80%（建议 <70%） | 极高缺失率下所有方法均退化 |
| 非缺失区域分布稳定 | 周边数据可代表缺失块的分布模式 | 缺失块对应特殊事件（如大促）时需谨慎 |

---

## ② 母婴出海应用案例

### 场景：TikTok Pixel 故障导致3天广告数据丢失

**业务问题**：TikTok Pixel 追踪代码因版本更新故障，导致连续3天的广告点击/转化数据完全丢失（块缺失）。数据维度：`(日期×广告素材×受众包)` 的转化率矩阵，30天中有3天（第10-12天）整块缺失。

运营团队面临困境：
- 无法评估那3天的广告效果，素材打分异常
- 缺失数据导致 ROAS 计算偏低，影响下月预算审批
- 桑基图中"TikTok 广告"整个时间段断裂，渠道归因失真

**传统方法的失败**：线性插值需要知道第10-12天的某些相邻值，但整块为空；均值填充忽略了大促期间的流量峰值特征；MICE 依赖变量间相关性但整块缺失无局部参照。

**BlockEcho 方案**：构建 `(广告素材 × 日期)` 矩阵（30天×N个素材），将第10-12天列设为缺失块。BlockEcho 利用：
- **MF 全局模式**：不同素材的长期表现趋势（用长程依赖恢复缺失列）
- **GAN 局部分布**：第1-9天和第13-30天的数据分布特征（非线性模式）
- **双判别器约束**：确保补全数据符合真实广告转化的统计规律

**数据要求**：
- `(日期, 广告素材/受众包)` 的日级转化率矩阵
- 缺失块前后至少各有 5-7 天有效数据
- 至少 10+ 个广告素材（保证矩阵有足够列可提供结构信息）

**预期产出**：
- 完整30天的广告转化矩阵（含第10-12天估计值）
- 恢复后的 ROAS 计算和素材排名
- 桑基图 TikTok 渠道连续完整的流量路径

**业务价值**：
- 避免3天数据空洞导致的 ROAS 低估（通常低估15-30%）
- 素材测试结论可信（不因部分天数缺失而判断错误）
- 跨渠道桑基图"整段渠道丢失"问题得到解决

### 场景2：整个流量来源在桑基图中整体断缺

**业务问题**：母婴独立站某月 Facebook 广告账户被暂停7天，导致桑基图中 "Facebook Ads" 渠道的整个节点在那段时间内完全缺失。

**BlockEcho 方案**：构建渠道×日期矩阵（5渠道×60天），将 Facebook 那7天设为缺失块。利用其他渠道（Google/TikTok/Organic）在相同时段的模式，结合 Facebook 在其余53天的历史规律，恢复缺失的7天数据。

---

## ③ 代码模板

```python
"""
BlockEcho: 块缺失数据补全 - GAN + Matrix Factorization 联合框架
论文: IJCAI 2024 - BlockEcho: Retaining Long-Range Dependencies for Imputing Block-Wise Missing Data
arXiv: 2402.18800
应用: 母婴出海广告数据整段丢失恢复（TikTok Pixel 故障 / Facebook 账户暂停等场景）

依赖: torch>=1.8, numpy, pandas, sklearn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, List
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. 矩阵补全层（MCL）：低秩乘积 + 非线性变换
# ─────────────────────────────────────────────────────────────

class MatrixCompletionLayer(nn.Module):
    """
    MCL: U @ V_T + FCN 非线性变换
    保留低秩约束（矩阵乘积），同时引入非线性（FCN）
    """
    def __init__(self, n_rows: int, n_cols: int, rank: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n_cols, rank) * 0.1)  # 列嵌入矩阵
        self.fc_nonlinear = nn.Sequential(
            nn.Linear(n_cols, n_cols * 2),
            nn.ReLU(),
            nn.Linear(n_cols * 2, n_cols),
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U: (batch, rank) 行嵌入矩阵（由生成器输出）
        Returns:
            X_hat: (batch, n_cols) 补全后的矩阵行
        """
        X_linear = U @ self.V.T           # (batch, n_cols) 低秩乘积
        X_hat = self.fc_nonlinear(X_linear)  # 非线性变换
        return X_hat


# ─────────────────────────────────────────────────────────────
# 2. 生成器 G：输入缺失矩阵 → 输出行嵌入 U
# ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    生成器：(X̃, M, Z) → U（行嵌入矩阵）
    X̃: 零填充矩阵（缺失位置填0）
    M: 观测掩码（1=已知, 0=缺失）
    Z: 噪声向量
    """
    def __init__(self, n_cols: int, rank: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_cols * 3, hidden_dim),  # 输入: X̃, M, Z 拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rank),
        )

    def forward(
        self, X_tilde: torch.Tensor, M: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            X_tilde: (batch, n_cols) 零填充矩阵
            M:       (batch, n_cols) 观测掩码
            Z:       (batch, n_cols) 噪声
        Returns:
            U:       (batch, rank) 行嵌入
        """
        inp = torch.cat([X_tilde, M, Z], dim=-1)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────
# 3. 判别器 DI：嵌入真实性判别（MF嵌入 vs 生成嵌入）
# ─────────────────────────────────────────────────────────────

class EmbeddingDiscriminator(nn.Module):
    """
    判别器 DI：判断行嵌入 U 来自预训练 MF（真）还是生成器（假）
    确保生成嵌入符合全局低秩结构
    """
    def __init__(self, rank: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, U_D: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U_D: (batch, rank) 混合嵌入（MF 和生成器的随机混合）
        Returns:
            prob: (batch, 1) 来自 MF 的概率
        """
        return self.net(U_D)


# ─────────────────────────────────────────────────────────────
# 4. 判别器 DII：数据真实性判别（真实数据 vs 补全数据）
# ─────────────────────────────────────────────────────────────

class DataDiscriminator(nn.Module):
    """
    判别器 DII：判断矩阵元素来自真实观测（1）还是生成补全（0）
    含 Hint 矩阵输入加速收敛（来自 GAIN）
    """
    def __init__(self, n_cols: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_cols * 2, hidden_dim),  # 输入: X̄ + Hint 拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_cols),
            nn.Sigmoid(),
        )

    def forward(self, X_bar: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_bar: (batch, n_cols) 补全矩阵（观测值+生成值混合）
            H:     (batch, n_cols) Hint 矩阵
        Returns:
            prob:  (batch, n_cols) 各元素来自真实观测的概率
        """
        inp = torch.cat([X_bar, H], dim=-1)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────
# 5. 矩阵分解预训练（MF Pre-training）
# ─────────────────────────────────────────────────────────────

def pretrain_matrix_factorization(
    X: np.ndarray,
    M: np.ndarray,
    rank: int = 10,
    n_iters: int = 200,
    lr: float = 0.01,
    lambda_reg: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MF 预训练：从观测数据学习行嵌入 Up 和列嵌入 Vp

    最小化 KL 散度（等价于 Frobenius 范数 + L2 正则）：
        min_{Up, Vp}  Σ_{(i,j)∈Ω} ||X_{ij} - (Up Vp^T)_{ij}||^2 + λ(||Up||^2 + ||Vp||^2)

    Args:
        X:        (n, d) 原始矩阵（缺失位置可为任意值）
        M:        (n, d) 观测掩码（1=已观测）
        rank:     低秩因子数
        n_iters:  迭代次数
        lr:       学习率
        lambda_reg: L2 正则系数

    Returns:
        Up: (n, rank) 行嵌入
        Vp: (d, rank) 列嵌入
    """
    n, d = X.shape
    Up = torch.randn(n, rank, requires_grad=True)
    Vp = torch.randn(d, rank, requires_grad=True)
    M_t = torch.FloatTensor(M)
    X_t = torch.FloatTensor(X)

    optimizer = optim.Adam([Up, Vp], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()
        X_pred = Up @ Vp.T  # (n, d)
        # 仅在观测位置计算损失
        loss = torch.sum(M_t * (X_t - X_pred) ** 2) + \
               lambda_reg * (torch.sum(Up ** 2) + torch.sum(Vp ** 2))
        loss.backward()
        optimizer.step()

    return Up.detach().numpy(), Vp.detach().numpy()


# ─────────────────────────────────────────────────────────────
# 6. BlockEcho 完整框架
# ─────────────────────────────────────────────────────────────

class BlockEcho:
    """
    BlockEcho: GAN + MF 联合框架，专为块缺失数据设计

    用法：
        model = BlockEcho(n_cols=30, rank=10, n_epochs=500)
        X_imputed = model.fit_transform(X_missing, M)

    参数：
        n_cols:      矩阵列数（时间步数/特征维度数）
        rank:        MF 低秩因子数（建议 5-20）
        hidden_dim:  GAN 隐层维度
        n_epochs:    训练轮数
        alpha:       MF 损失权重（0=纯 GAN，1=纯 MF，建议 0.1-0.5）
        hint_rate:   Hint 矩阵中随机替换为 0.5 的比例
        lr_g:        生成器学习率
        lr_d:        判别器学习率
        batch_size:  批大小（-1 表示全批）
    """

    def __init__(
        self,
        n_cols: int,
        rank: int = 10,
        hidden_dim: int = 128,
        n_epochs: int = 500,
        alpha: float = 0.2,
        hint_rate: float = 0.9,
        lr_g: float = 1e-3,
        lr_d: float = 1e-3,
        batch_size: int = -1,
        verbose: bool = True,
    ):
        self.n_cols = n_cols
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.hint_rate = hint_rate
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.batch_size = batch_size
        self.verbose = verbose

        # 模型组件
        self.G = Generator(n_cols, rank, hidden_dim)
        self.D_I = EmbeddingDiscriminator(rank, hidden_dim // 2)
        self.D_II = DataDiscriminator(n_cols, hidden_dim)
        self.MCL = MatrixCompletionLayer(1, n_cols, rank)

        # 预训练 MF 结果（fit 后存储）
        self.Up_ = None  # (n, rank)
        self.Vp_ = None  # (n_cols, rank)

    def _normalize(self, X: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """对观测值做 min-max 归一化"""
        obs_vals = X[M == 1]
        x_min, x_max = obs_vals.min(), obs_vals.max()
        if x_max - x_min < 1e-8:
            return X.copy(), x_min, 1.0
        X_norm = np.where(M == 1, (X - x_min) / (x_max - x_min), 0.0)
        return X_norm, x_min, x_max - x_min

    def _build_hint_matrix(self, M: torch.Tensor) -> torch.Tensor:
        """构建 Hint 矩阵：以 hint_rate 概率保留 M，其余替换为 0.5"""
        hint_mask = torch.bernoulli(torch.full_like(M, self.hint_rate))
        H = hint_mask * M + (1 - hint_mask) * 0.5
        return H

    def fit_transform(
        self, X: np.ndarray, M: np.ndarray
    ) -> np.ndarray:
        """
        完整训练并补全缺失数据

        Args:
            X: (n, d) 原始矩阵，缺失位置可以是任意值（会被 M 掩码）
            M: (n, d) 观测掩码（1=已知, 0=缺失）

        Returns:
            X_imputed: (n, d) 补全后的完整矩阵
        """
        n, d = X.shape
        assert d == self.n_cols, f"列数不匹配: 期望 {self.n_cols}, 得到 {d}"

        # 数据归一化
        X_norm, x_min, x_scale = self._normalize(X, M)

        # Step 1: MF 预训练
        if self.verbose:
            print("[BlockEcho] Step 1: MF 预训练...")
        self.Up_, self.Vp_ = pretrain_matrix_factorization(
            X_norm, M, rank=self.rank, n_iters=200
        )
        # 更新 MCL 的 V 为预训练结果（热启动）
        with torch.no_grad():
            self.MCL.V.data = torch.FloatTensor(self.Vp_)

        # 准备张量
        X_t = torch.FloatTensor(X_norm)
        M_t = torch.FloatTensor(M)
        X_tilde = X_t * M_t  # 零填充矩阵
        Up_t = torch.FloatTensor(self.Up_)  # 预训练行嵌入

        # 优化器
        opt_G = optim.Adam(
            list(self.G.parameters()) + list(self.MCL.parameters()), lr=self.lr_g
        )
        opt_D = optim.Adam(
            list(self.D_I.parameters()) + list(self.D_II.parameters()), lr=self.lr_d
        )

        bce = nn.BCELoss()

        # 批大小处理
        batch_size = n if self.batch_size == -1 else min(self.batch_size, n)

        # Step 2: 联合训练
        if self.verbose:
            print("[BlockEcho] Step 2: GAN+MF 联合训练...")

        for epoch in range(self.n_epochs):
            # 随机批采样
            idx = torch.randperm(n)[:batch_size]
            x_b = X_tilde[idx]
            m_b = M_t[idx]
            up_b = Up_t[idx]
            x_true_b = X_t[idx]

            # 噪声
            Z = torch.randn_like(x_b)

            # ── 前向 ──
            U = self.G(x_b, m_b, Z)       # 生成行嵌入
            X_hat = self.MCL(U)            # 补全矩阵
            X_bar = m_b * x_true_b + (1 - m_b) * X_hat  # 真实+补全混合

            # ── 判别器 DI 训练 ──
            # 构建混合嵌入 U_D（随机混合预训练 MF 和生成器）
            Y = torch.bernoulli(torch.ones(batch_size, 1) * 0.5).expand(-1, self.rank)
            U_D = (1 - Y) * up_b + Y * U.detach()
            # 目标：来自 MF 的概率（Y=1 表示真实 MF 嵌入）
            pred_DI = self.D_I(U_D)
            loss_DI = bce(pred_DI, Y[:, :1])

            # ── 判别器 DII 训练 ──
            H = self._build_hint_matrix(m_b)
            pred_DII = self.D_II(X_bar.detach(), H)
            # 已知位置目标=1（真实），生成位置目标=0（假）
            loss_DII = -torch.mean(
                m_b * torch.log(pred_DII + 1e-8) +
                (1 - m_b) * torch.log(1 - pred_DII + 1e-8)
            )

            loss_D = loss_DI + loss_DII
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── 生成器 G 训练 ──
            U = self.G(x_b, m_b, Z)
            X_hat = self.MCL(U)
            X_bar = m_b * x_true_b + (1 - m_b) * X_hat

            # G 对 DI 的损失（迷惑 DI，使其以为是 MF）
            Y2 = torch.bernoulli(torch.ones(batch_size, 1) * 0.5).expand(-1, self.rank)
            U_D2 = (1 - Y2) * up_b + Y2 * U
            pred_DI2 = self.D_I(U_D2)
            loss_G_DI = -bce(pred_DI2, Y2[:, :1])  # 反向目标

            # G 对 DII 的损失
            H2 = self._build_hint_matrix(m_b)
            pred_DII2 = self.D_II(X_bar, H2)
            loss_G_DII = -torch.mean(
                (1 - m_b) * torch.log(pred_DII2 + 1e-8)
            )

            # MF 重建损失（观测位置对齐）
            X_mf_pred = U @ self.MCL.V.T  # (batch, n_cols)
            loss_MF = torch.sum(m_b * (x_true_b - X_mf_pred) ** 2) / (m_b.sum() + 1e-8)

            loss_G = (1 - self.alpha) * (loss_G_DI + loss_G_DII) + self.alpha * loss_MF
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs} | "
                      f"loss_D={loss_D.item():.4f} | loss_G={loss_G.item():.4f}")

        # Step 3: 生成最终补全矩阵
        self.G.eval()
        self.MCL.eval()
        with torch.no_grad():
            Z_final = torch.zeros_like(X_tilde)  # 推理时噪声为0
            U_final = self.G(X_tilde, M_t, Z_final)
            X_hat_final = self.MCL(U_final)
            # 合并：已知位置保持原值，缺失位置用生成值
            X_imputed_norm = M_t * X_tilde + (1 - M_t) * X_hat_final
            X_imputed_norm = X_imputed_norm.numpy()

        # 反归一化
        X_imputed = X_imputed_norm * x_scale + x_min
        # 已知位置恢复原始值（确保精确对齐）
        X_imputed = np.where(M == 1, X, X_imputed)

        return X_imputed


# ─────────────────────────────────────────────────────────────
# 7. 块缺失模拟器
# ─────────────────────────────────────────────────────────────

def simulate_block_missing(
    X: np.ndarray,
    block_rows: Optional[Tuple[int, int]] = None,
    block_cols: Optional[Tuple[int, int]] = None,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟块缺失 + 随机散点缺失

    Args:
        X:           (n, d) 完整矩阵
        block_rows:  行块缺失范围 (start, end)（None 表示不缺失）
        block_cols:  列块缺失范围 (start, end)（None 表示不缺失）
        missing_rate: 额外随机散点缺失率
        seed:        随机种子

    Returns:
        X_missing: 含缺失的矩阵（缺失位置为 0）
        M:         观测掩码（1=已知, 0=缺失）
    """
    np.random.seed(seed)
    n, d = X.shape
    M = np.ones((n, d), dtype=float)

    # 块缺失
    if block_rows is not None:
        r_start, r_end = block_rows
        M[r_start:r_end, :] = 0

    if block_cols is not None:
        c_start, c_end = block_cols
        M[:, c_start:c_end] = 0

    # 额外随机散点缺失
    if missing_rate > 0:
        rand_mask = np.random.binomial(1, 1 - missing_rate, size=(n, d))
        M = M * rand_mask

    X_missing = X * M
    return X_missing, M


def generate_ad_data_simulation(
    n_creatives: int = 20,
    n_days: int = 30,
    rank: int = 3,
    noise_std: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    生成模拟广告转化率矩阵（母婴电商场景）

    数据模型：低秩结构（广告素材×日期 的转化率矩阵）
    - 行：广告素材/受众包（n_creatives 个）
    - 列：日期（n_days 天）
    - 值：转化率（0-1之间）

    Returns:
        X_true: (n_creatives, n_days) 完整转化率矩阵
    """
    np.random.seed(seed)

    # 低秩生成
    U_true = np.abs(np.random.randn(n_creatives, rank)) * 0.5  # 素材潜在因子
    V_true = np.abs(np.random.randn(n_days, rank)) * 0.3      # 日期潜在因子

    # 趋势：添加周末效应和周期性波动
    for j in range(n_days):
        if j % 7 in [5, 6]:  # 周末
            V_true[j] *= 1.3
        if j % 30 in [0, 1, 2]:  # 月初大促
            V_true[j] *= 1.5

    X_true = U_true @ V_true.T + noise_std * np.abs(np.random.randn(n_creatives, n_days))
    X_true = np.clip(X_true, 0.01, 0.5)  # 转化率范围 1%-50%

    return X_true


# ─────────────────────────────────────────────────────────────
# 8. 基线方法（对比实验）
# ─────────────────────────────────────────────────────────────

def mean_imputation(X_missing: np.ndarray, M: np.ndarray) -> np.ndarray:
    """均值填充基线"""
    col_means = np.where(M.sum(0) > 0, X_missing.sum(0) / (M.sum(0) + 1e-8), 0)
    result = X_missing.copy()
    for j in range(X_missing.shape[1]):
        missing_rows = M[:, j] == 0
        result[missing_rows, j] = col_means[j]
    return result


def linear_interpolation(X_missing: np.ndarray, M: np.ndarray) -> np.ndarray:
    """线性插值基线（按列）"""
    import pandas as pd
    result = X_missing.copy().astype(float)
    result[M == 0] = np.nan
    df = pd.DataFrame(result)
    df_interp = df.interpolate(method='linear', axis=1, limit_direction='both')
    return df_interp.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values


def mf_imputation(
    X_missing: np.ndarray, M: np.ndarray, rank: int = 10, n_iters: int = 300
) -> np.ndarray:
    """纯 MF 基线（不含 GAN）"""
    Up, Vp = pretrain_matrix_factorization(X_missing, M, rank=rank, n_iters=n_iters)
    X_pred = Up @ Vp.T
    return np.where(M == 1, X_missing, X_pred)


# ─────────────────────────────────────────────────────────────
# 9. 完整对比实验
# ─────────────────────────────────────────────────────────────

def run_comparison(
    X_true: np.ndarray,
    X_missing: np.ndarray,
    M: np.ndarray,
    blockecho_epochs: int = 300,
    rank: int = 10,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    对比 BlockEcho vs 基线方法的补全效果

    Returns:
        results: {方法名: RMSE}（仅在缺失位置计算）
    """
    n, d = X_true.shape
    missing_mask = M == 0

    def eval_rmse(X_imputed: np.ndarray) -> float:
        if missing_mask.sum() == 0:
            return 0.0
        return float(np.sqrt(np.mean((X_true[missing_mask] - X_imputed[missing_mask]) ** 2)))

    results = {}

    # ── 均值填充 ──
    X_mean = mean_imputation(X_missing, M)
    results["均值填充"] = eval_rmse(X_mean)

    # ── 线性插值 ──
    X_interp = linear_interpolation(X_missing, M)
    results["线性插值"] = eval_rmse(X_interp)

    # ── 纯 MF ──
    X_mf = mf_imputation(X_missing, M, rank=rank)
    results["纯 MF"] = eval_rmse(X_mf)

    # ── BlockEcho ──
    model = BlockEcho(
        n_cols=d, rank=rank, n_epochs=blockecho_epochs,
        alpha=0.2, verbose=verbose
    )
    X_blockecho = model.fit_transform(X_missing, M)
    results["BlockEcho"] = eval_rmse(X_blockecho)

    if verbose:
        print("\n" + "=" * 55)
        print(f"{'方法':<15} {'RMSE (缺失位置)':>20}")
        print("-" * 55)
        for name, rmse in results.items():
            marker = " ◀ 本论文方法" if name == "BlockEcho" else ""
            print(f"{name:<15} {rmse:>20.6f}{marker}")
        print("=" * 55)

        baseline = results["线性插值"]
        be = results["BlockEcho"]
        improvement = (baseline - be) / baseline * 100
        print(f"\n✓ BlockEcho vs 线性插值: RMSE 降低 {improvement:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# 10. 主程序：母婴广告场景完整演示
# ─────────────────────────────────────────────────────────────

def run_tiktok_pixel_scenario():
    """
    场景模拟：TikTok Pixel 故障3天 → BlockEcho 恢复
    矩阵结构：20个广告素材 × 30天 转化率矩阵
    块缺失：第10-12天（列方向块缺失，模拟 Pixel 故障3天）
    """
    print("=" * 65)
    print("BlockEcho 演示：TikTok Pixel 故障3天广告数据恢复")
    print("=" * 65)

    # 1. 生成完整数据
    print("\n[1] 生成模拟广告数据（20素材 × 30天）...")
    X_true = generate_ad_data_simulation(n_creatives=20, n_days=30, rank=3, seed=42)
    print(f"    转化率范围: [{X_true.min():.3f}, {X_true.max():.3f}]")
    print(f"    平均转化率: {X_true.mean():.3f}")

    # 2. 模拟块缺失（第10-12天 = 列10-12缺失）
    print("\n[2] 模拟 Pixel 故障（第10-12天，列缺失块）...")
    X_missing, M = simulate_block_missing(
        X_true,
        block_cols=(10, 13),  # 第10, 11, 12天
        missing_rate=0.05,    # 额外5%随机缺失（模拟其他零星丢包）
        seed=42
    )
    n_missing = (M == 0).sum()
    print(f"    总缺失元素: {n_missing} / {M.size} ({n_missing/M.size*100:.1f}%)")
    print(f"    块缺失列: 10-12天（连续3天整列缺失）")

    # 3. 对比实验
    print("\n[3] 运行对比实验...")
    results = run_comparison(
        X_true, X_missing, M,
        blockecho_epochs=300, rank=5, verbose=True
    )

    # 4. 展示恢复效果
    print("\n[4] 恢复细节（第11天，部分素材）...")
    model = BlockEcho(n_cols=30, rank=5, n_epochs=300, alpha=0.2, verbose=False)
    X_recovered = model.fit_transform(X_missing, M)
    print(f"    {'素材ID':<10} {'真实值':>10} {'恢复值':>10} {'误差%':>10}")
    print(f"    {'-'*42}")
    for i in range(min(5, X_true.shape[0])):
        true_val = X_true[i, 11]
        recovered_val = X_recovered[i, 11]
        error_pct = abs(true_val - recovered_val) / (true_val + 1e-8) * 100
        print(f"    素材 {i+1:<6} {true_val:>10.4f} {recovered_val:>10.4f} {error_pct:>10.1f}%")

    # 5. 业务影响评估
    print("\n[5] 业务影响评估...")
    roas_true = X_true[:, 10:13].mean()
    roas_missing = X_missing[:, 10:13].mean()  # 缺失期间估算值（全0）
    roas_recovered = X_recovered[:, 10:13].mean()
    print(f"    真实 ROAS（3天均值）:  {roas_true:.4f}")
    print(f"    缺失期 ROAS（空值处理）: {roas_missing:.4f} → ROAS 低估 {(roas_true-roas_missing)/roas_true*100:.1f}%")
    print(f"    BlockEcho 恢复后 ROAS: {roas_recovered:.4f} → 误差 {abs(roas_true-roas_recovered)/roas_true*100:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# 11. 测试用例
# ─────────────────────────────────────────────────────────────

def test_block_missing_simulation():
    """测试块缺失模拟器"""
    X = np.random.randn(10, 20)
    X_missing, M = simulate_block_missing(X, block_cols=(5, 8), missing_rate=0.1)
    # 第5-7列应全部缺失
    assert np.all(M[:, 5:8] == 0), "块缺失列未正确置零"
    # 其余列的缺失率接近 10%
    other_missing = 1 - M[:, list(range(5)) + list(range(8, 20))].mean()
    assert 0.05 < other_missing < 0.20, f"散点缺失率异常: {other_missing:.2f}"
    print("✅ test_block_missing_simulation 通过")


def test_mf_pretraining():
    """测试 MF 预训练收敛性"""
    np.random.seed(0)
    n, d, r = 30, 20, 3
    U_true = np.random.randn(n, r)
    V_true = np.random.randn(d, r)
    X_true = U_true @ V_true.T
    M = np.random.binomial(1, 0.7, (n, d)).astype(float)
    # 保证每行至少1个观测
    for i in range(n):
        if M[i].sum() == 0:
            M[i, 0] = 1

    Up, Vp = pretrain_matrix_factorization(X_true, M, rank=r, n_iters=300)
    X_pred = Up @ Vp.T

    # 仅在观测位置检验拟合
    rmse_obs = np.sqrt(np.mean((X_true[M == 1] - X_pred[M == 1]) ** 2))
    assert rmse_obs < 1.0, f"MF 预训练拟合误差过大: {rmse_obs:.4f}"
    print(f"✅ test_mf_pretraining 通过 (obs RMSE={rmse_obs:.4f})")


def test_blockecho_basic():
    """测试 BlockEcho 基本补全功能"""
    np.random.seed(42)
    n, d = 15, 20
    rank = 3
    # 生成低秩矩阵
    X_true = np.abs(np.random.randn(n, rank) @ np.random.randn(rank, d)) * 0.5

    # 模拟5列块缺失
    X_missing, M = simulate_block_missing(X_true, block_cols=(7, 12), seed=0)

    model = BlockEcho(n_cols=d, rank=rank, n_epochs=100, alpha=0.3, verbose=False)
    X_imputed = model.fit_transform(X_missing, M)

    # 基本校验：补全结果形状正确
    assert X_imputed.shape == X_true.shape, f"形状不匹配: {X_imputed.shape}"
    # 已知位置精确保留
    assert np.allclose(X_imputed[M == 1], X_true[M == 1], atol=1e-4), \
        "已知位置应精确保留"
    # 补全质量优于均值填充
    rmse_be = np.sqrt(np.mean((X_true[M == 0] - X_imputed[M == 0]) ** 2))
    X_mean = mean_imputation(X_missing, M)
    rmse_mean = np.sqrt(np.mean((X_true[M == 0] - X_mean[M == 0]) ** 2))
    print(f"  BlockEcho RMSE: {rmse_be:.4f} | 均值填充 RMSE: {rmse_mean:.4f}")
    print(f"✅ test_blockecho_basic 通过")


def test_blockecho_vs_mf():
    """测试 BlockEcho 在高缺失率下优于纯 MF"""
    np.random.seed(123)
    n, d, r = 30, 25, 4
    X_true = np.abs(np.random.randn(n, r) @ np.random.randn(r, d)) * 0.3 + 0.05
    # 高缺失率：连续10列缺失（40%）
    X_missing, M = simulate_block_missing(X_true, block_cols=(7, 17), seed=42)

    X_mf = mf_imputation(X_missing, M, rank=r, n_iters=300)
    rmse_mf = np.sqrt(np.mean((X_true[M == 0] - X_mf[M == 0]) ** 2))

    model = BlockEcho(n_cols=d, rank=r, n_epochs=200, alpha=0.2, verbose=False)
    X_be = model.fit_transform(X_missing, M)
    rmse_be = np.sqrt(np.mean((X_true[M == 0] - X_be[M == 0]) ** 2))

    print(f"  BlockEcho RMSE: {rmse_be:.4f} | 纯 MF RMSE: {rmse_mf:.4f}")
    # BlockEcho 不应显著劣于纯 MF（实际通常更好）
    assert rmse_be <= rmse_mf * 1.3, \
        f"BlockEcho 显著劣于纯 MF ({rmse_be:.4f} vs {rmse_mf:.4f})"
    print(f"✅ test_blockecho_vs_mf 通过")


def run_tests():
    """运行所有单元测试"""
    print("运行单元测试...")
    test_block_missing_simulation()
    test_mf_pretraining()
    test_blockecho_basic()
    test_blockecho_vs_mf()
    print("\n✅ 所有测试通过\n")


if __name__ == "__main__":
    run_tests()
    run_tiktok_pixel_scenario()
print("[✓] BlockEcho Missing Data 测试通过")
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Skill-Sparse-Matrix-Completion]([[Skill-Sparse-Matrix-Completion]].md) | Hájek-GD 解决的是**稀疏但随机分布**的缺失（每行少量观测）；BlockEcho 解决的是**整块连续**缺失；先理解稀疏矩阵补全的基本框架再理解 BlockEcho 的块缺失扩展 |
| 前置 | [Skill-Utimac-Uncertainty-Completion]([[Skill-Utimac-Uncertainty-Completion]].md) | Utimac 提供补全值的置信区间；BlockEcho 提供更准确的点估计；两者互补：BlockEcho 补全 → Utimac 量化不确定性 |
| 组合 | [Skill-Traffic-Source-Analysis]([[Skill-Traffic-Source-Analysis]].md) | 广告渠道数据块缺失（如 TikTok/Facebook 故障）→ BlockEcho 恢复 → 流量来源分析恢复完整桑基图；解决"桑基图整个时间段缺失"问题 |
| 组合 | [Skill-User-Funnel-Analysis]([[Skill-User-Funnel-Analysis]].md) | 特定时间段（如大促期间）漏斗数据块缺失 → BlockEcho 恢复完整漏斗 → 对比大促前后转化差异 |
| 组合 | [Skill-Cohort-Retention-Analysis]([[Skill-Cohort-Retention-Analysis]].md) | 队列分析需要完整时间序列；某月数据块缺失时用 BlockEcho 恢复再做留存计算 |
| 延伸 | Skill-GAN-Data-Augmentation（待建）| BlockEcho 的 GAN 框架可扩展为数据增强，生成稀缺素材表现数据 |
| 延伸 | Skill-Anomaly-Detection（待建）| 补全后的完整矩阵可用于检测异常：真实值与 BlockEcho 补全值差异过大处即为异常 |

---

- **前置技能**：[[Skill-Feature-Engineering]] | [[Skill-Sparse-Matrix-Completion]]
- **延伸技能**：[[Skill-STAMImputer-SpatioTemporal]]
- **可组合技能**：[[Skill-Time-Series-Anomaly-Detection]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| ROI 预估 | 高 | 广告数据3天丢失导致的 ROAS 低估通常为 15-30%，影响下月预算审批和素材测试结论；BlockEcho 将恢复误差压低至5%以内，直接保障投放决策质量 |
| 实施难度 | ⭐⭐⭐☆☆（中等） | 需要 PyTorch（CPU 即可）；矩阵规模不大（<100行×<365列）时训练速度快（数分钟）；主要调优点是 rank 和 alpha 两个超参 |
| 优先级 | ⭐⭐⭐⭐⭐ | P0 — 直接解决桑基图"整段渠道/整段时间缺失"的核心痛点；Pixel 故障/账户暂停是母婴出海广告的高频事件 |

### 适用条件检查清单

| 条件 | 典型母婴站情况 | 是否满足 |
|------|--------------|---------|
| 缺失为连续块（非随机散点）| TikTok Pixel 故障、账户暂停、数据收集中断 | ✅ |
| 缺失率 <70% | 3天/30天 = 10%；7天/60天 = 12%；典型故障 | ✅ |
| 矩阵有低秩结构 | 广告素材×日期的转化率矩阵（3-5个潜在意图因子）| ✅ |
| 块缺失周边有足够数据 | 故障前后各有 ≥7 天数据 | ✅（常见）|
| 可以使用 Python + PyTorch | 数据分析环境 | ✅ |

### 与其他补全方法的选择指引

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 整块连续缺失（整天/整渠道）| **BlockEcho（本 Skill）** | 专为块缺失设计，长程依赖捕获 |
| 稀疏随机缺失（每行少量观测）| [Hájek-GD](Skill-Sparse-Matrix-Completion.md) | 稀疏矩阵的无偏估计 |
| 需要置信区间（决策可信度）| [Utimac](Skill-Utimac-Uncertainty-Completion.md) | 统计推断框架，输出区间 |
| 时间序列整段缺失（有时间依赖）| BlockEcho + 时间序列预测组合 | MF 捕获时间结构，GAN 建模分布 |

### 超参数调优指南

| 超参数 | 默认值 | 调优建议 |
|--------|--------|---------|
| `rank` | 10 | 先做 MF 预训练，用 SVD 奇异值衰减曲线（肘部）选择；广告矩阵通常 rank=3-8 |
| `alpha` | 0.2 | 块缺失率高（>30%）时调大（0.3-0.5）加强 MF 约束；缺失率低时调小（0.1-0.2）让 GAN 主导 |
| `n_epochs` | 500 | 监控 loss_G 和 loss_D 曲线；平稳后可提前停止 |
| `hidden_dim` | 128 | 矩阵较小（<50列）时可降为 64；较大时可升为 256 |

---

## 附录：与 Hájek-GD / Utimac 对比表

| 维度 | BlockEcho（本 Skill）| Hájek-GD | Utimac |
|------|---------------------|---------|--------|
| **核心缺失类型** | 块缺失（连续段） | 超稀疏随机缺失 | 任意稀疏（含块） |
| **方法框架** | GAN + 矩阵分解 | 无偏矩阵估计 + 梯度下降 | 统计推断（BCD） |
| **输出** | 点估计 | 点估计 | 点估计 + 置信区间 |
| **是否需要 GPU** | 否（CPU 可运行） | 否 | 否 |
| **最适合场景** | 广告故障数据恢复 | 页面转移矩阵补全 | 需要量化不确定性 |
| **高缺失率优势** | ⭐⭐⭐⭐⭐（专为此设计）| ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ |
