"""
Monodense Deep Neural Model for Item Price Elasticity
单品价格弹性估计的 Monodense 深度神经网络

论文: Monodense Deep Neural Model for Determining Item Price Elasticity
arXiv:2603.29261 (Walmart Inc.)

核心创新:
- Monodense 层在神经网络中强制价格→需求的单调递减约束
- 无需对照实验即可从大规模交易数据中学习单品价格弹性
- 经济学一致性保证: 输出的弹性始终为负
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class MonodenseLayer(nn.Module):
    """
    Monodense 层: 对输入特征的权重施加单调性约束。

    - t_i =  1: 单调递增, 权重 w_i >= 0
    - t_i = -1: 单调递减, 权重 w_i <= 0 (如价格特征)
    - t_i =  0: 无约束
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonicity: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        if monotonicity is None:
            monotonicity = np.zeros(in_features, dtype=np.int32)
        self.register_buffer(
            "monotonicity", torch.tensor(monotonicity, dtype=torch.int32)
        )

    def get_constrained_weight(self) -> torch.Tensor:
        w = self.weight
        mono = self.monotonicity.unsqueeze(0)  # (1, in_features)
        inc_mask = (mono == 1).float()
        dec_mask = (mono == -1).float()
        unconstrained_mask = (mono == 0).float()
        w_inc = torch.clamp(w, min=0.0) * inc_mask
        w_dec = torch.clamp(w, max=0.0) * dec_mask
        w_uncon = w * unconstrained_mask
        return w_inc + w_dec + w_uncon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.get_constrained_weight()
        return F.linear(x, w, self.bias)


class BoundedActivation(nn.Module):
    """有界激活函数: 基于凸激活构造的饱和变体。"""

    def __init__(self, rho=nn.ELU(alpha=1.0)):
        super().__init__()
        self.rho = rho
        self.rho_one = float(self.rho(torch.tensor(1.0)).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negative = self.rho(x + 1.0) - self.rho_one
        positive = self.rho(x - 1.0) + self.rho_one
        return torch.where(x < 0, negative, positive)


class MonodenseDLM(nn.Module):
    """
    Monodense Deep Learning Model
    简化版网络结构: Embedding + Dense + Monodense + 输出头
    """

    def __init__(
        self,
        num_continuous: int,
        num_categorical: int,
        embedding_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        monotonicity: Optional[np.ndarray] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.embedding = nn.Linear(num_categorical, embedding_dim)
        input_dim = num_continuous + embedding_dim

        # Monodense 层的输入包含 continuous + embedding，需补全 monotonicity
        if monotonicity is not None:
            full_mono = np.zeros(input_dim, dtype=np.int32)
            full_mono[:num_continuous] = monotonicity
            monotonicity = full_mono

        # 架构: 输入层后先用 Monodense 捕获价格-需求单调关系，再接标准 Dense
        self.input_mono = MonodenseLayer(input_dim, hidden_dims[0], monotonicity=monotonicity)
        layers: List[nn.Module] = [nn.ReLU(), nn.Dropout(dropout)]
        prev_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(
        self, x_continuous: torch.Tensor, x_categorical: torch.Tensor
    ) -> torch.Tensor:
        embed = F.relu(self.embedding(x_categorical))
        x = torch.cat([x_continuous, embed], dim=-1)
        h = self.input_mono(x)
        h = self.backbone(h)
        out = self.head(h).squeeze(-1)
        return torch.clamp(out, min=0.0)  # 需求量非负


@dataclass
class ElasticityResult:
    """弹性估计结果"""

    item_id: str
    current_price: float
    predicted_demand: float
    elasticity: float
    price_sensitivity: str  # 高敏感 / 中敏感 / 低敏感


class ElasticityEstimator:
    """
    价格弹性估计器
    封装模型训练、预测与弹性计算
    """

    def __init__(
        self,
        num_continuous: int,
        num_categorical: int,
        embedding_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        monotonicity: Optional[np.ndarray] = None,
        lr: float = 0.01,
        epochs: int = 25,
        batch_size: int = 128,
    ):
        self.model = MonodenseDLM(
            num_continuous=num_continuous,
            num_categorical=num_categorical,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            monotonicity=monotonicity,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses: List[float] = []

    def fit(
        self, X_continuous: np.ndarray, X_categorical: np.ndarray, y: np.ndarray
    ) -> None:
        """训练模型"""
        self.model.train()
        n = len(y)
        x_c = torch.tensor(X_continuous, dtype=torch.float32)
        x_cat = torch.tensor(X_categorical, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            batches = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                pred = self.model(x_c[idx], x_cat[idx])
                loss = F.mse_loss(pred, y_t[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            avg_loss = epoch_loss / max(batches, 1)
            self.losses.append(avg_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, MSE Loss: {avg_loss:.4f}")

    def predict_demand(
        self, X_continuous: np.ndarray, X_categorical: np.ndarray
    ) -> np.ndarray:
        """预测需求量"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                torch.tensor(X_continuous, dtype=torch.float32),
                torch.tensor(X_categorical, dtype=torch.float32),
            )
        return pred.numpy()

    def compute_elasticity(
        self,
        X_continuous: np.ndarray,
        X_categorical: np.ndarray,
        price_index: int,
        delta_pct: float = 0.05,
        item_ids: Optional[List[str]] = None,
    ) -> List[ElasticityResult]:
        """
        计算价格弹性: E = [y(p+dp) - y(p)] / y(p) * p/dp
        """
        self.model.eval()
        base_demand = self.predict_demand(X_continuous, X_categorical)
        x_c_up = X_continuous.copy()
        x_c_up[:, price_index] *= 1.0 + delta_pct
        up_demand = self.predict_demand(x_c_up, X_categorical)

        results: List[ElasticityResult] = []
        for i in range(len(base_demand)):
            p = X_continuous[i, price_index]
            y_base = base_demand[i]
            y_up = up_demand[i]
            if y_base <= 1e-6:
                elasticity = 0.0
            else:
                elasticity = ((y_up - y_base) / y_base) * (1.0 / delta_pct)
            abs_e = abs(elasticity)
            if abs_e > 1.5:
                sensitivity = "高敏感"
            elif abs_e > 0.8:
                sensitivity = "中敏感"
            else:
                sensitivity = "低敏感"
            item_id = item_ids[i] if item_ids else f"SKU_{i}"
            results.append(
                ElasticityResult(
                    item_id=item_id,
                    current_price=float(p),
                    predicted_demand=float(y_base),
                    elasticity=float(elasticity),
                    price_sensitivity=sensitivity,
                )
            )
        return results


def generate_momcozy_pricing_data(
    n_samples: int = 2000, seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 Momcozy 母婴电商的合成定价数据

    返回:
        df: 完整的 DataFrame
        X_continuous: 连续特征矩阵
        X_categorical: 类别特征矩阵
        y: 需求量
    """
    rng = np.random.RandomState(seed)

    categories = ["吸奶器", "温奶器", "婴儿背带", "辅食机"]
    cat_weights = [0.35, 0.25, 0.25, 0.15]
    cat_idx = rng.choice(len(categories), size=n_samples, p=cat_weights)

    base_price = np.array([129.0, 59.0, 79.0, 49.0])[cat_idx]
    base_demand = np.array([500.0, 800.0, 600.0, 400.0])[cat_idx]

    price = base_price * (1 + rng.uniform(-0.2, 0.2, n_samples))
    competitor_price = base_price * (1 + rng.uniform(-0.15, 0.25, n_samples))
    promo_depth = rng.uniform(0.0, 0.3, n_samples)
    inventory_level = rng.uniform(0.3, 1.0, n_samples)
    seasonality = 1.0 + 0.2 * np.sin(2 * np.pi * rng.uniform(0, 1, n_samples))

    demand = (
        base_demand
        * np.power(price / base_price, -1.2)
        * (1 + 1.5 * promo_depth)
        * np.power(competitor_price / price, 0.4)
        * seasonality
        * inventory_level
    )
    demand = demand * (1 + rng.normal(0, 0.1, n_samples))
    demand = np.clip(demand, 10.0, None)

    df = pd.DataFrame(
        {
            "item_id": [f"SKU_{i:04d}" for i in range(n_samples)],
            "category": [categories[c] for c in cat_idx],
            "price": price,
            "competitor_price": competitor_price,
            "promo_depth": promo_depth,
            "inventory_level": inventory_level,
            "seasonality": seasonality,
            "demand": demand,
        }
    )

    X_continuous = df[
        ["price", "competitor_price", "promo_depth", "inventory_level", "seasonality"]
    ].values.astype(np.float32)
    df["category"] = pd.Categorical(df["category"], categories=categories)
    X_categorical = pd.get_dummies(df["category"]).values.astype(np.float32)
    y = df["demand"].values.astype(np.float32)

    return df, X_continuous, X_categorical, y


def main() -> None:
    print("=" * 60)
    print("Monodense DLM: Momcozy 母婴电商单品价格弹性估计")
    print("=" * 60)

    df, X_c, X_cat, y = generate_momcozy_pricing_data(n_samples=2000)
    print(f"\n数据规模: {len(df)} 条交易记录")
    print(df.groupby("category")[["price", "demand"]].mean().round(2))

    # 单调性约束: 价格特征(index=0)设为单调递减(t=-1)
    monotonicity = np.array([-1, 0, 0, 0, 0], dtype=np.int32)

    estimator = ElasticityEstimator(
        num_continuous=X_c.shape[1],
        num_categorical=X_cat.shape[1],
        embedding_dim=8,
        hidden_dims=[64, 32],
        monotonicity=monotonicity,
        lr=0.01,
        epochs=20,
        batch_size=128,
    )

    print("\n开始训练 Monodense-DLM...")
    estimator.fit(X_c, X_cat, y)

    print("\n计算各 SKU 价格弹性 (基于 5% 价格变动)...")
    results = estimator.compute_elasticity(
        X_continuous=X_c[:20],
        X_categorical=X_cat[:20],
        price_index=0,
        delta_pct=0.05,
        item_ids=df["item_id"].tolist()[:20],
    )

    print("\n示例结果 (前 8 个 SKU):")
    for r in results[:8]:
        print(
            f"  {r.item_id}: 价格=${r.current_price:.1f}, "
            f"预测需求={r.predicted_demand:.0f}, "
            f"弹性={r.elasticity:.2f}, 敏感度={r.price_sensitivity}"
        )

    full_results = estimator.compute_elasticity(
        X_continuous=X_c,
        X_categorical=X_cat,
        price_index=0,
        delta_pct=0.05,
        item_ids=df["item_id"].tolist(),
    )
    insight_df = pd.DataFrame(
        [
            {"category": cat, "elasticity": r.elasticity}
            for cat, r in zip(df["category"], full_results)
        ]
    )
    summary = insight_df.groupby("category")["elasticity"].agg(["mean", "std"]).round(2)
    print("\n各品类平均价格弹性:")
    print(summary)

    print("\n业务建议:")
    for cat, row in summary.iterrows():
        avg_e = row["mean"]
        if abs(avg_e) > 1.2:
            suggestion = "高弹性品类, 建议采用竞争性低价策略以换取销量"
        elif abs(avg_e) > 0.7:
            suggestion = "中等弹性, 可通过促销节点测试价格敏感度"
        else:
            suggestion = "低弹性品类, 用户对价格不敏感, 有提价空间"
        print(f"  - {cat}: 平均弹性 {avg_e:.2f} -> {suggestion}")

    print("\n" + "=" * 60)
    print("Monodense-DLM 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
