"""
LTV预测 - 零膨胀对数正态模型 (ZILN)
用于母婴出海电商新客价值预测和会员等级划分

基于论文:
- A Deep Probabilistic Model for Customer Lifetime Value Prediction: arXiv:1912.07753 (Google Research, 2019)
- HT-GNN: Hyper-Temporal Graph Neural Network for LTV Prediction: arXiv:2601.13013 (Baidu, 2026)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ZILNModel(nn.Module):
    """
    Zero-Inflated Lognormal 神经网络模型
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 dropout_rate: float = 0.2):
        """
        初始化 ZILN 模型

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout 率
        """
        super(ZILNModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # 构建共享网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # 三个输出头
        self.churn_head = nn.Linear(prev_dim, 1)  # 流失概率
        self.mu_head = nn.Linear(prev_dim, 1)     # 对数均值
        self.sigma_head = nn.Linear(prev_dim, 1)  # 对数标准差

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            churn_prob: 流失概率
            mu: 对数均值
            sigma: 对数标准差
        """
        shared = self.shared_layers(x)

        churn_logits = self.churn_head(shared)
        churn_prob = torch.sigmoid(churn_logits)

        mu = self.mu_head(shared)
        sigma = torch.nn.functional.softplus(self.sigma_head(shared)) + 1e-6

        return churn_prob, mu, sigma

    def predict_ltv(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测期望 LTV

        Args:
            x: 输入特征

        Returns:
            expected_ltv: 期望 LTV 值
        """
        churn_prob, mu, sigma = self.forward(x)

        # E[Y] = (1 - p) * exp(mu + sigma^2 / 2)
        expected_log_ltv = mu + 0.5 * (sigma ** 2)
        expected_ltv = (1 - churn_prob) * torch.exp(expected_log_ltv)

        return expected_ltv.squeeze()


class ZILNLoss(nn.Module):
    """ZILN 损失函数"""

    def __init__(self, eps: float = 1e-6):
        super(ZILNLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true: torch.Tensor, churn_prob: torch.Tensor,
                mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算 ZILN 损失

        Args:
            y_true: 真实 LTV 值
            churn_prob: 预测流失概率
            mu: 预测对数均值
            sigma: 预测对数标准差

        Returns:
            损失值
        """
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(1)

        # 零值掩码
        positive_mask = (y_true > 0).float()

        # 零膨胀部分损失
        churn_prob = torch.clamp(churn_prob, self.eps, 1 - self.eps)
        zero_inflated_loss = -(
            (1 - positive_mask) * torch.log(churn_prob + self.eps) +
            positive_mask * torch.log(1 - churn_prob + self.eps)
        )

        # 对数正态部分损失
        log_y = torch.log(y_true + self.eps)
        lognormal_loss = positive_mask * (
            torch.log(sigma + self.eps) +
            0.5 * ((log_y - mu) / (sigma + self.eps)) ** 2 +
            log_y
        )

        return torch.mean(zero_inflated_loss + lognormal_loss)


class LTVDataset(Dataset):
    """LTV 数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ZILNTrainer:
    """ZILN 模型训练器"""

    def __init__(self, model: ZILNModel, learning_rate: float = 0.001,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = ZILNLoss()
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_model_state = None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            churn_prob, mu, sigma = self.model(batch_x)
            loss = self.criterion(batch_y, churn_prob, mu, sigma)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                churn_prob, mu, sigma = self.model(batch_x)
                loss = self.criterion(batch_y, churn_prob, mu, sigma)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, patience: int = 10) -> 'ZILNTrainer':
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break

        return self


class LTVEvaluator:
    """LTV 模型评估器"""

    @staticmethod
    def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算归一化 Gini 系数

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            归一化 Gini 系数
        """
        def gini(actual, pred):
            assert len(actual) == len(pred)
            # 合并数据并排序
            data = np.column_stack([actual, pred, np.arange(len(actual))]).astype(float)
            data = data[np.lexsort((data[:, 2], -1 * data[:, 1]))]
            total_losses = data[:, 0].sum()
            gini_sum = data[:, 0].cumsum().sum() / total_losses
            gini_sum -= (len(actual) + 1) / 2.0
            return gini_sum / len(actual)

        return gini(y_true, y_pred) / gini(y_true, y_true)

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算 LTV 预测评估指标

        Args:
            y_true: 真实 LTV
            y_pred: 预测 LTV

        Returns:
            指标字典
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        gini = LTVEvaluator.normalized_gini(y_true, y_pred)

        # Top 10% 捕获率
        top_10_pct = int(0.1 * len(y_true))
        top_true = set(np.argsort(y_true)[-top_10_pct:])
        top_pred = set(np.argsort(y_pred)[-top_10_pct:])
        capture_rate = len(top_true & top_pred) / top_10_pct

        return {
            'MAE': mae,
            'RMSE': rmse,
            'Normalized_Gini': gini,
            'Top10_Capture_Rate': capture_rate
        }


class LTVSegmentAnalyzer:
    """LTV 分层分析器"""

    @staticmethod
    def segment_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                         n_segments: int = 5) -> pd.DataFrame:
        """
        分析 LTV 分层效果

        Args:
            y_true: 真实 LTV
            y_pred: 预测 LTV
            n_segments: 分层数

        Returns:
            DataFrame: 各层分析结果
        """
        segments = pd.qcut(y_pred, q=n_segments,
                          labels=['Low', 'D-Low', 'Mid', 'D-High', 'High'])

        analysis = []
        for seg in ['Low', 'D-Low', 'Mid', 'D-High', 'High']:
            mask = segments == seg
            if mask.sum() > 0:
                analysis.append({
                    'Segment': seg,
                    'Count': mask.sum(),
                    'Pred_LTV_Mean': y_pred[mask].mean(),
                    'True_LTV_Mean': y_true[mask].mean(),
                    'Lift': y_true[mask].mean() / y_true.mean()
                })

        return pd.DataFrame(analysis)

    @staticmethod
    def get_operational_strategy(segment: str) -> str:
        """
        获取运营策略建议

        Args:
            segment: 分层名称

        Returns:
            策略建议
        """
        strategies = {
            'High': 'VIP服务 + 专属优惠 + 优先客服',
            'D-High': '定期邮件 + 会员权益 + 生日礼',
            'Mid': '自动化营销 + 标准权益',
            'D-Low': '低成本触达 + 激活优惠',
            'Low': '减少投放 + 自然留存'
        }
        return strategies.get(segment, '标准运营')


def generate_ltv_data(n_samples: int = 5000,
                      random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成母婴出海电商 LTV 模拟数据

    场景：吸奶器新客 LTV 预测
    """
    np.random.seed(random_state)

    # 用户画像特征
    age = np.clip(np.random.normal(31, 5, n_samples), 22, 45)
    is_first_time = np.random.binomial(1, 0.65, n_samples)
    income_level = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.35, 0.35, 0.1])

    # 首购行为特征
    first_order_value = np.random.lognormal(4.2, 0.4, n_samples)
    days_to_first_order = np.random.exponential(3, n_samples)
    pages_viewed = np.random.poisson(8, n_samples)
    used_coupon = np.random.binomial(1, 0.4, n_samples)

    # 渠道特征
    channel = np.random.choice(['facebook', 'tiktok', 'google', 'organic'],
                               n_samples, p=[0.4, 0.3, 0.2, 0.1])

    X = pd.DataFrame({
        'age': age,
        'is_first_time': is_first_time,
        'income_level': income_level,
        'first_order_value': first_order_value,
        'days_to_first_order': days_to_first_order,
        'pages_viewed': pages_viewed,
        'used_coupon': used_coupon,
        'channel_facebook': (channel == 'facebook').astype(int),
        'channel_tiktok': (channel == 'tiktok').astype(int),
        'channel_google': (channel == 'google').astype(int)
    })

    # 生成 LTV (零膨胀 + 对数正态)
    churn_logit = (
        -1.5 +
        0.5 * (1 - is_first_time) +
        0.3 * (income_level - 2.5) +
        0.02 * (first_order_value - 60) +
        0.1 * (pages_viewed - 8) +
        0.2 * used_coupon
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    is_churned = np.random.binomial(1, churn_prob)

    mu = 3.5 + 0.2 * (income_level - 2.5) + 0.01 * (first_order_value - 60) + 0.05 * is_first_time
    sigma = 0.6

    ltv = np.zeros(n_samples)
    non_churned = ~is_churned.astype(bool)
    ltv[non_churned] = np.random.lognormal(mu[non_churned], sigma, non_churned.sum())

    return X, ltv


def main():
    """主函数：演示 ZILN 在母婴 LTV 预测中的应用"""
    print("=" * 70)
    print("母婴出海 - ZILN LTV 预测")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    X, ltv = generate_ltv_data(n_samples=5000)
    print(f"   总样本: {len(X)}")
    print(f"   零 LTV 用户: {(ltv == 0).sum()} ({(ltv == 0).mean()*100:.1f}%)")
    print(f"   平均 LTV: ${ltv.mean():.2f}")

    # 2. 数据预处理
    print("\n[2] 数据预处理...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, ltv, test_size=0.2, random_state=42
    )

    # 3. 创建数据加载器
    train_dataset = LTVDataset(X_train, y_train)
    test_dataset = LTVDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 4. 训练模型
    print("\n[3] 训练 ZILN 模型...")
    model = ZILNModel(input_dim=X.shape[1], hidden_dims=[64, 32])
    trainer = ZILNTrainer(model, learning_rate=0.001)
    trainer.fit(train_loader, test_loader, epochs=100, patience=15)

    # 5. 预测
    print("\n[4] 预测 LTV...")
    model.eval()
    with torch.no_grad():
        y_pred = model.predict_ltv(torch.FloatTensor(X_test)).numpy()

    # 6. 评估
    print("\n[5] 模型评估...")
    evaluator = LTVEvaluator()
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    print(f"   MAE: ${metrics['MAE']:.2f}")
    print(f"   RMSE: ${metrics['RMSE']:.2f}")
    print(f"   归一化 Gini: {metrics['Normalized_Gini']:.3f}")
    print(f"   Top10% 捕获率: {metrics['Top10_Capture_Rate']*100:.1f}%")

    # 7. 分层分析
    print("\n[6] LTV 分层分析...")
    analyzer = LTVSegmentAnalyzer()
    segment_df = analyzer.segment_analysis(y_test, y_pred)
    print(segment_df.to_string(index=False))

    # 8. 运营策略
    print("\n[7] 运营策略建议...")
    for seg in ['High', 'D-High', 'Mid', 'D-Low', 'Low']:
        strategy = analyzer.get_operational_strategy(seg)
        print(f"   {seg}: {strategy}")

    print("\n" + "=" * 70)
    return model, scaler, metrics


if __name__ == '__main__':
    model, scaler, metrics = main()
