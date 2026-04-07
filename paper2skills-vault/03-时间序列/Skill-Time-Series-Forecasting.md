# Skill Card: Time Series Forecasting (时间序列预测)

---

## ① 算法原理

### 核心思想
时间序列预测解决的核心问题是：**基于历史销售数据，预测未来一段时间的需求量**，从而指导备货、定价和供应链决策。与简单外推不同， modern 时间序列模型能捕捉季节性、趋势、节假日效应和外部变量（促销、竞品）的影响。

### 数学直觉

**分解模型 (Additive)**：
$$Y(t) = Trend(t) + Seasonality(t) + Holiday(t) + Noise(t)$$

- **Trend（趋势）**：长期增长或下降趋势
- **Seasonality（季节性）**：周期性的波动（如每周、每月、每季）
- **Holiday（节假日）**：节假日效应（如双11、黑五）
- **Noise（噪声）**：随机波动

**Prophet 模型**：
$$g(t) = \frac{C}{1 + e^{-k(t - m)}} + b t \quad \text{(logistic trend)}$$
$$s(t) = \sum_{n=1}^{N} a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right)$$

**LSTM/GRU**：
- 门控机制：输入门、遗忘门、输出门
- 长期依赖：$C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t$
- 时序记忆：隐藏状态 $h_t$ 包含历史信息

### 关键假设
- **历史可重复**：未来模式与历史相似
- **独立同分布噪声**：残差服从正态分布
- **无外部冲击**：不考虑突发事件（可通过外部变量引入）

---

## ② 吸奶器出海应用案例

### 场景一：吸奶器周销量预测

**业务问题**：
母婴出海电商需要预测未来 4 周的销量，以指导海外仓补货。传统方法是基于移动平均或简单指数平滑，但无法捕捉：
- 周期性：周末销量通常高于工作日
- 季节性：奶粉、尿裤在大促季（618、双11、黑五）销量激增
- 趋势：新品牌上线后有爬坡期

**数据要求**：
- 历史销量：至少 2 年的日/周销量数据
- 节假日：春节、618、双11、黑五、圣诞节
- 促销标记：是否有活动、活动力度
- 外部变量：竞品价格、搜索指数

**预期产出**：
- 未来 4 周销量预测（点预测 + 置信区间）
- 预测误差评估（MAPE、RMSE）
- 关键影响因子贡献度

**业务价值**：
- 库存周转提升 15-25%
- 缺货率降低 30-50%
- 滞销库存减少 10-20%

---

### 场景二：爆款生命周期预测

**业务问题**：
新款婴儿推车、 安全座椅上市后，需要预测其生命周期曲线：导入期、成长期、成熟期、衰退期。这决定了：
- 首批采购量（多了压库存，少了丢销售）
- 价格策略（成长期可维持高价，衰退期需清仓）
- 备货节奏（成长期需频繁补货）

**数据要求**：
- 新品上市后前 4-8 周的销售数据
- 同品类历史新品曲线（参考相似产品）
- 竞品上市信息

**预期产出**：
- 未来 12 周销量预测曲线
- 峰值销量和峰值时间预测
- 生命周期阶段判断

**业务价值**：
- 新品首批库存准确率提升 30%+
- 价格策略优化增加毛利 5-10%
- 避免滞销品积压

---

## ③ 代码模板

```python
"""
Time Series Forecasting
用于母婴出海电商销量预测
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


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

    def _predict_simple(self, n_periods):
        """简单预测"""
        predictions = self.avg_value + self.trend * np.arange(1, n_periods + 1)
        std = np.std(values[-30:])
        lower = predictions - 1.96 * std
        upper = predictions + 1.96 * std
        return predictions, lower, upper

    def _fit_exponential_smoothing(self, dates, values):
        """指数平滑"""
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels not installed")

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
        # 简化置信区间
        std = self.model.resid.std()
        lower = predictions - 1.96 * std
        upper = predictions + 1.96 * std
        return predictions.values, lower.values, upper.values

    def _fit_prophet(self, dates, values):
        """Prophet 模型"""
        if not HAS_PROPHET:
            raise ImportError("prophet not installed")

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
        # 准备训练数据
        window = 7
        X, y = [], []
        for i in range(window, len(values)):
            X.append(values[i-window:i])
            y.append(values[i])

        X = np.array(X)
        y = np.array(y)

        # 归一化
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 简化：用线性回归模拟 LSTM 输出
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)

        self.window = window

    def _predict_lstm(self, n_periods):
        """LSTM 预测（简化）"""
        # 使用最后窗口作为起点
        last_window = values[-self.window:]
        predictions = []

        for _ in range(n_periods):
            X_scaled = self.scaler.transform(last_window.reshape(1, -1))
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred

        predictions = np.array(predictions)
        std = np.std(values[-30:])
        lower = predictions - 1.96 * std
        upper = predictions + 1.96 * std

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

    # 生成 2 年周数据
    dates = pd.date_range('2023-01-01', periods=n_weeks, freq='W')

    # 基础销量
    base = 100

    # 趋势：年增长 10%
    trend = np.linspace(0, 0.2, n_weeks)

    # 季节性：52周周期
    seasonality = 0.3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

    # 节假日效应（618、双11）
    holiday_effect = np.zeros(n_weeks)
    for i, d in enumerate(dates):
        if d.month == 6 and d.day >= 1 and d.day <= 30:  # 618
            holiday_effect[i] += 0.5
        if d.month == 11 and d.day >= 10 and d.day <= 20:  # 双11
            holiday_effect[i] += 0.8

    # 噪声
    noise = np.random.normal(0, 0.1, n_weeks)

    # 计算最终销量
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
```

---

## ④ 技能关联

### 前置技能
- **基础统计**：理解均值、标准差、正态分布
- **Python 基础**：熟练使用 pandas、numpy
- **数据库查询**：能从 SQL 取数

### 延伸技能
- **深度学习时序**：使用 Transformer、TFT 进行复杂预测
- **异常检测**：检测销量异常波动
- **因果推断**：分析促销等因素的增量效应

### 可组合技能
- **库存优化**：结合预测结果计算安全库存
- **Uplift Modeling**：预测不同营销活动的效果
- **动态定价**：基于需求预测调整价格

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 周销量预测 | 库存周转提升 15-25% | 开发 2 周，数据接入 1 周 | 8-12x |
| 生命周期预测 | 首批准确率提升 30%+ | 开发 1 周 | 10-15x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要至少 2 年的历史销量数据
- 技术门槛：中等，Prophet 较简单，深度学习需要更多资源
- 工程复杂度：中等，需要定时跑批
- 维护成本：中等，需要定期重新训练

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- 业务价值极高：预测是所有决策的基础
- 见效快：1-2 周可完成 POC
- 可落地性强：母婴出海场景明确
- 数据依赖：已有历史数据可用

### 评估依据
1. **需求预测**是供应链的核心 input，预测准确率提升 10% 可节省 5-10% 库存成本
2. Prophet 上手简单，可快速验证
3. 与库存优化、动态定价形成完整闭环
4. 优先从周销量预测切入，逐步扩展到日SKU级别
