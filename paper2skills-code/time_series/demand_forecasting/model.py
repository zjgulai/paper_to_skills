"""
Time Series Forecasting
用于母婴出海电商销量预测
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """需求预测器"""

    def __init__(self, model_type='simple'):
        """
        初始化预测器

        Args:
            model_type: 'simple', 'exponential_smoothing', 'prophet', 'lstm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def fit(self, dates, values):
        """
        训练模型

        Args:
            dates: 日期序列
            values: 销量序列
        """
        if self.model_type == 'simple':
            self._fit_simple(values)
        elif self.model_type == 'exponential_smoothing':
            self._fit_exponential_smoothing(dates, values)
        elif self.model_type == 'prophet':
            self._fit_prophet(dates, values)
        elif self.model_type == 'lstm':
            self._fit_lstm(values)

        self.is_fitted = True
        return self

    def predict(self, n_periods, future_dates=None):
        """
        预测未来销量

        Args:
            n_periods: 预测周期数
            future_dates: 未来日期（可选）

        Returns:
            predictions: 预测值
            lower_bound: 下界
            upper_bound: 上界
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.model_type == 'simple':
            return self._predict_simple(n_periods)
        elif self.model_type == 'exponential_smoothing':
            return self._predict_exponential_smoothing(n_periods)
        elif self.model_type == 'prophet':
            return self._predict_prophet(n_periods, future_dates)
        elif self.model_type == 'lstm':
            return self._predict_lstm(n_periods)

    def _fit_simple(self, values):
        """简单模型：移动平均 + 趋势"""
        self.avg_value = np.mean(values)
        self.trend = (np.mean(values[-30:]) - np.mean(values[:30])) / len(values)
        self.std_value = np.std(values[-30:])

    def _predict_simple(self, n_periods):
        """简单预测"""
        predictions = self.avg_value + self.trend * np.arange(1, n_periods + 1)
        lower = predictions - 1.96 * self.std_value
        upper = predictions + 1.96 * self.std_value
        return predictions, lower, upper

    def _fit_exponential_smoothing(self, dates, values):
        """指数平滑"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            print("Warning: statsmodels not installed, falling back to simple model")
            self.model_type = 'simple'
            self._fit_simple(values)
            return

        df = pd.DataFrame({'ds': dates, 'y': values})
        df = df.set_index('ds')

        self.model = ExponentialSmoothing(
            df['y'],
            seasonal_periods=7,
            trend='add',
            seasonal='add'
        ).fit()

    def _predict_exponential_smoothing(self, n_periods):
        """指数平滑预测"""
        predictions = self.model.forecast(n_periods)
        std = self.model.resid.std()
        lower = predictions - 1.96 * std
        upper = predictions + 1.96 * std
        return predictions.values, lower.values, upper.values

    def _fit_prophet(self, dates, values):
        """Prophet 模型"""
        try:
            from prophet import Prophet
        except ImportError:
            print("Warning: prophet not installed, falling back to simple model")
            self.model_type = 'simple'
            self._fit_simple(values)
            return

        df = pd.DataFrame({'ds': dates, 'y': values})
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        self.model.fit(df)

    def _predict_prophet(self, n_periods, future_dates):
        """Prophet 预测"""
        if future_dates is None:
            future = self.model.make_future_dataframe(periods=n_periods, freq='W')
        else:
            future = pd.DataFrame({'ds': future_dates})

        forecast = self.model.predict(future)
        predictions = forecast['yhat'].values[-n_periods:]
        lower = forecast['yhat_lower'].values[-n_periods:]
        upper = forecast['yhat_upper'].values[-n_periods:]

        return predictions, lower, upper

    def _fit_lstm(self, values):
        """LSTM 模型（简化版，使用滑动窗口）"""
        window = 7
        X, y = [], []
        for i in range(window, len(values)):
            X.append(values[i-window:i])
            y.append(values[i])

        X = np.array(X)
        y = np.array(y)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = LinearRegression()
        self.model.fit(X_scaled, y)

        self.window = window
        self.std_value = np.std(values[-30:])

    def _predict_lstm(self, n_periods):
        """LSTM 预测（简化）"""
        last_window = values[-self.window:]
        predictions = []

        for _ in range(n_periods):
            X_scaled = self.scaler.transform(last_window.reshape(1, -1))
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred

        predictions = np.array(predictions)
        lower = predictions - 1.96 * self.std_value
        upper = predictions + 1.96 * self.std_value

        return predictions, lower, upper


def evaluate_forecast(actual, predicted):
    """
    评估预测效果

    Args:
        actual: 实际值
        predicted: 预测值

    Returns:
        metrics: 评估指标
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


# ==================== 示例代码 ====================

def generate_sample_data(n_weeks=104):
    """生成模拟销量数据"""
    np.random.seed(42)

    dates = pd.date_range('2023-01-01', periods=n_weeks, freq='W')

    base = 100
    trend = np.linspace(0, 0.2, n_weeks)
    seasonality = 0.3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

    holiday_effect = np.zeros(n_weeks)
    for i, d in enumerate(dates):
        if d.month == 6 and d.day >= 1 and d.day <= 30:
            holiday_effect[i] += 0.5
        if d.month == 11 and d.day >= 10 and d.day <= 20:
            holiday_effect[i] += 0.8

    noise = np.random.normal(0, 0.1, n_weeks)

    values = base * (1 + trend + seasonality + holiday_effect + noise)
    values = np.maximum(values, 0)

    return dates, values


def main():
    """主函数"""
    print("=" * 60)
    print("Time Series Forecasting 测试")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    dates, values = generate_sample_data(n_weeks=104)
    print(f"   数据周数: {len(values)}")
    print(f"   平均周销量: {values.mean():.1f}")
    print(f"   销量标准差: {values.std():.1f}")

    # 2. 划分训练测试集
    print("\n[2] 划分训练/测试集...")
    train_size = int(len(values) * 0.8)
    train_dates, test_dates = dates[:train_size], dates[train_size:]
    train_values, test_values = values[:train_size], values[train_size:]
    print(f"   训练集: {len(train_values)} 周")
    print(f"   测试集: {len(test_values)} 周")

    # 3. 训练简单模型
    print("\n[3] 训练预测模型...")
    model = DemandForecaster(model_type='simple')
    model.fit(train_dates, train_values)
    print(f"   模型类型: {model.model_type}")

    # 4. 预测
    print("\n[4] 预测未来...")
    predictions, lower, upper = model.predict(len(test_values))
    print(f"   预测周数: {len(predictions)}")

    # 5. 评估
    print("\n[5] 评估效果...")
    metrics = evaluate_forecast(test_values, predictions)
    print(f"   MAE: {metrics['MAE']:.2f}")
    print(f"   RMSE: {metrics['RMSE']:.2f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")

    # 6. 展示预测结果
    print("\n[6] 预测结果示例 (前8周):")
    print("-" * 60)
    for i in range(min(8, len(predictions))):
        print(f"   Week {i+1}: 预测={predictions[i]:.0f}, "
              f"区间=[{lower[i]:.0f}, {upper[i]:.0f}], "
              f"实际={test_values[i]:.0f}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return model


if __name__ == '__main__':
    model = main()
