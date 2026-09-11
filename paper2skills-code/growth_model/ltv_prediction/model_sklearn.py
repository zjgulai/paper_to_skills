"""
LTV预测 - ZILN 简化版 (基于 scikit-learn)
用于母婴出海电商新客价值预测

当 PyTorch 不可用时使用此版本，功能与 ZILN 类似但使用 MLPRegressor
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleLTVModel:
    """
    简化版 LTV 预测模型
    使用 MLPRegressor 处理零膨胀和长尾分布
    """

    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=500):
        """
        初始化模型

        Args:
            hidden_layer_sizes: 隐藏层大小
            max_iter: 最大迭代次数
        """
        # 两个模型：一个预测是否流失（分类），一个预测 LTV（回归）
        self.churn_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42
        )
        self.ltv_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.churn_threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型

        Args:
            X: 特征矩阵
            y: LTV 值（含0表示流失）
        """
        X_scaled = self.scaler.fit_transform(X)

        # 1. 训练流失预测模型（y=0 为流失）
        is_churned = (y == 0).astype(float)
        self.churn_model.fit(X_scaled, is_churned)

        # 2. 训练 LTV 预测模型（仅对未流失用户）
        non_zero_mask = y > 0
        if non_zero_mask.sum() > 0:
            # 对 LTV 取对数处理长尾分布
            log_ltv = np.log(y[non_zero_mask] + 1)
            self.ltv_model.fit(X_scaled[non_zero_mask], log_ltv)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测 LTV

        Args:
            X: 特征矩阵

        Returns:
            LTV 预测值
        """
        X_scaled = self.scaler.transform(X)

        # 预测流失概率
        churn_prob = self.churn_model.predict(X_scaled)
        churn_prob = np.clip(churn_prob, 0, 1)

        # 预测 LTV（对未流失用户）
        log_ltv_pred = self.ltv_model.predict(X_scaled)
        ltv_pred = np.exp(log_ltv_pred) - 1
        ltv_pred = np.clip(ltv_pred, 0, None)

        # 综合预测：如果可能流失，降低 LTV 预期
        expected_ltv = (1 - churn_prob) * ltv_pred

        return expected_ltv

    def predict_churn_prob(self, X: np.ndarray) -> np.ndarray:
        """预测流失概率"""
        X_scaled = self.scaler.transform(X)
        return np.clip(self.churn_model.predict(X_scaled), 0, 1)


class LTVEvaluator:
    """LTV 模型评估器"""

    @staticmethod
    def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算归一化 Gini 系数"""
        def gini(actual, pred):
            assert len(actual) == len(pred)
            data = np.column_stack([actual, pred, np.arange(len(actual))]).astype(float)
            data = data[np.lexsort((data[:, 2], -1 * data[:, 1]))]
            total_losses = data[:, 0].sum()
            gini_sum = data[:, 0].cumsum().sum() / total_losses
            gini_sum -= (len(actual) + 1) / 2.0
            return gini_sum / len(actual)

        return gini(y_true, y_pred) / gini(y_true, y_true)

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
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


def generate_ltv_data(n_samples: int = 5000, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """生成母婴出海电商 LTV 模拟数据"""
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
    """主函数：演示 LTV 预测"""
    print("=" * 70)
    print("母婴出海 - LTV 预测 (sklearn 简化版)")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    X, ltv = generate_ltv_data(n_samples=5000)
    print(f"   总样本: {len(X)}")
    print(f"   零 LTV 用户: {(ltv == 0).sum()} ({(ltv == 0).mean()*100:.1f}%)")
    print(f"   平均 LTV: ${ltv.mean():.2f}")
    print(f"   LTV 中位数: ${np.median(ltv):.2f}")

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, ltv, test_size=0.2, random_state=42
    )
    print(f"\n[2] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 3. 训练模型
    print("\n[3] 训练模型...")
    model = SimpleLTVModel(hidden_layer_sizes=(64, 32))
    model.fit(X_train.values, y_train)
    print("   训练完成")

    # 4. 预测
    print("\n[4] 预测 LTV...")
    y_pred = model.predict(X_test.values)

    # 5. 评估
    print("\n[5] 模型评估...")
    evaluator = LTVEvaluator()
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    print(f"   MAE: ${metrics['MAE']:.2f}")
    print(f"   RMSE: ${metrics['RMSE']:.2f}")
    print(f"   归一化 Gini: {metrics['Normalized_Gini']:.3f}")
    print(f"   Top10% 捕获率: {metrics['Top10_Capture_Rate']*100:.1f}%")

    # 6. 分层分析
    print("\n[6] LTV 分层分析 (按预测值)...")
    segments = pd.qcut(y_pred, q=5, labels=['Low', 'D-Low', 'Mid', 'D-High', 'High'])
    for seg in ['High', 'D-High', 'Mid', 'D-Low', 'Low']:
        mask = segments == seg
        if mask.sum() > 0:
            avg_ltv = y_test[mask].mean()
            print(f"   {seg}: 平均真实 LTV ${avg_ltv:.2f}, 样本数 {mask.sum()}")

    # 7. 高价值用户分析
    print("\n[7] 高价值用户分析 (Top 20%)...")
    high_value_mask = y_pred > np.percentile(y_pred, 80)
    high_value_ltv = y_test[high_value_mask].mean()
    print(f"   预测高价值用户的平均真实 LTV: ${high_value_ltv:.2f}")
    print(f"   是整体平均的 {high_value_ltv / y_test.mean():.1f} 倍")

    print("\n" + "=" * 70)
    print("LTV 预测分析完成！")
    print("=" * 70)

    return model, metrics


if __name__ == '__main__':
    model, metrics = main()
