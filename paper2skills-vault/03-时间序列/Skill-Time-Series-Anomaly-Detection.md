```markdown
---
title: Time Series Anomaly Detection for E-Commerce Monitoring
doc_type: knowledge
module: 03-时间序列
topic: anomaly-detection
status: stable
created: 2026-05-15
updated: 2026-05-15
owner: self
source: arxiv:1902.08387
roadmap_phase: phase1
---

# Skill Card: Time Series Anomaly Detection

---

## ① 算法原理

> **论文**：Anomaly Detection in Time Series: A Comprehensive Evaluation | **年份**：2019

**核心问题**：母婴出海电商的关键指标（GMV、订单量、转化率、退货率）时刻波动。如何区分"正常波动"和"真实异常"？异常检测的本质是：建立正常行为的概率模型，将低概率事件标记为异常。

**三种主流方法**：

### 统计方法（基线）

**Z-Score**：
$$z_t = \frac{x_t - \mu}{\sigma}$$
$|z_t| > 3$ 视为异常。简单但假设数据服从正态分布。

**STL分解 + 残差检验**：
1. 将时序分解为趋势（Trend）、季节性（Seasonal）、残差（Residual）
2. 对残差应用Z-Score或IQR规则
3. 分离季节性后，异常更易识别

### 机器学习方法

**Isolation Forest**：
- 核心思想：异常点是"容易被孤立"的点
- 随机选择特征和切分点构建多棵决策树
- 异常点的平均路径长度显著短于正常点
- 优势：无需标注数据，对高维特征友好

**Prophet + 区间外检测**：
- Prophet预测未来值及其置信区间
- 实际值落在置信区间外即标记为异常
- 优势：天然处理季节性和节假日效应

### 深度学习方法（前沿）

**AutoEncoder重构误差**：
- 用正常数据训练AutoEncoder学习"正常模式"
- 异常数据的重构误差显著大于正常数据
- 优势：捕获复杂的非线性模式

**VAE（变分自编码器）概率异常检测**：
- 不仅看重构误差，还看后验概率
- 异常点的后验分布与正常分布差异大
- 2025年前沿：结合Transformer的时序VAE

**反直觉洞察**：
- 95%的"异常告警"是误报——因为业务指标天然波动大（促销、周末、节假日）
- 好的异常检测不是灵敏度越高越好，而是**上下文感知**——知道今天是黑五，GMV翻倍是正常的
- 最简单的方法（STL + 3-sigma）在80%的场景下足够，深度学习只在复杂多变量场景有优势

---

## ② 母婴出海应用案例

### 场景1：订单量异常监控

**业务问题**：Momcozy 的日订单量通常在500-800单之间波动。某天订单量突然降到200单——是系统Bug、支付通道故障、还是正常的市场波动？

**检测流程**：
1. **数据预处理**：取过去90天日订单量
2. **STL分解**：分离趋势、季节性（周内模式）、残差
3. **残差异常检测**：残差超过3倍标准差标记为异常
4. **上下文校验**：检查当天是否为节假日、是否有已知促销活动
5. **多指标交叉验证**：同时检查转化率、客单价是否同步异常

**预期产出**：
- 异常检测：订单量残差 = -4.2σ → 标记为异常
- 根因定位：转化率正常，但支付成功率从98%降到72% → 支付通道故障
- 告警延迟：<30分钟（实时检测）

**业务价值**：
- 支付通道故障的平均发现时间：4小时 → 30分钟
- 避免损失：故障期间订单量损失约50%，快速修复可减少80%损失

### 场景2：退货率异常预警

**业务问题**：某批次婴儿推车退货率从正常的5%飙升到15%。需要尽早发现，避免更多问题订单发出。

**检测流程**：
1. **滚动窗口监控**：7天滚动退货率
2. **Prophet预测**：基于历史数据预测正常退货率区间
3. **异常标记**：实际值超出95%置信区间
4. **维度下钻**：按SKU、仓库、物流商拆解，定位问题来源

**预期产出**：
- 告警触发：退货率超出预测区间上限
- 根因：某仓库发货的SKU混入了错误配件
- 止损：及时暂停该仓库发货，避免问题扩大

---

## ③ 代码模板

```python
"""
Time Series Anomaly Detection — 时序异常检测
用于电商关键指标的监控与告警

支持：STL分解、Isolation Forest、Z-Score检测
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


# ==================== 统计方法：Z-Score ====================

class ZScoreAnomalyDetector:
    """基于Z-Score的异常检测器"""

    def __init__(self, threshold=3.0):
        """
        Args:
            threshold: Z-Score阈值
        """
        self.threshold = threshold
        self.mean = None
        self.std = None

    def fit(self, series):
        """拟合模型"""
        self.series = series
        self.mean = series.mean()
        self.std = series.std()
        return self

    def predict(self, series=None):
        """预测异常"""
        if series is None:
            data = self.series
        else:
            data = series

        z_scores = (data - self.mean) / (self.std + 1e-8)
        anomalies = np.abs(z_scores) > self.threshold

        return {
            'is_anomaly': anomalies,
            'z_scores': z_scores,
            'mean': self.mean,
            'std': self.std
        }


# ==================== 统计方法：STL分解 ====================

class STLAnomalyDetector:
    """基于STL分解的异常检测器"""

    def __init__(self, period=7, threshold=3.0):
        """
        Args:
            period: 季节性周期（7=周，30=月）
            threshold: Z-Score阈值
        """
        self.period = period
        self.threshold = threshold
        self.trend = None
        self.seasonal = None
        self.residual = None
        self.residual_mean = None
        self.residual_std = None

    def _decompose_stl(self, series):
        """简化的STL分解实现"""
        n = len(series)
        
        # 计算趋势（使用中心移动平均）
        window = max(3, self.period)
        trend = pd.Series(series).rolling(window=window, center=True).mean()
        trend = trend.fillna(method='bfill').fillna(method='ffill')
        
        # 去趋势
        detrended = series - trend
        
        # 计算季节性（按周期平均）
        seasonal = np.zeros(n)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            if len(indices) > 0:
                seasonal[indices] = np.mean(detrended[indices])
        
        # 平滑季节性
        seasonal_smooth = pd.Series(seasonal).rolling(window=3, center=True).mean()
        seasonal_smooth = seasonal_smooth.fillna(method='bfill').fillna(method='ffill')
        seasonal = seasonal_smooth.values
        
        # 计算残差
        residual = series - trend - seasonal
        
        return trend, seasonal, residual

    def fit(self, series):
        """拟合STL模型"""
        self.series = series
        self.trend, self.seasonal, self.residual = self._decompose_stl(series)
        self.residual_mean = self.residual.mean()
        self.residual_std = self.residual.std()
        return self

    def predict(self, series=None):
        """预测异常"""
        if series is None:
            resid = self.residual
        else:
            # 对新数据使用训练期的趋势/季节性作为基线
            last_trend = np.mean(self.trend[-self.period:])
            last_seasonal = self.seasonal[-self.period:]
            resid = series - last_trend - last_seasonal[:len(series)]

        z_scores = (resid - self.residual_mean) / (self.residual_std + 1e-8)
        anomalies = np.abs(z_scores) > self.threshold

        return {
            'is_anomaly': anomalies,
            'z_scores': z_scores,
            'residuals': resid,
            'trend': self.trend if series is None else None,
            'seasonal': self.seasonal if series is None else None
        }


# ==================== 机器学习方法：Isolation Forest ====================

class IFAnomalyDetector:
    """基于Isolation Forest的异常检测器"""

    def __init__(self, contamination=0.05, n_estimators=100):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.contamination = contamination

    def fit(self, features):
        """
        训练模型

        Args:
            features: DataFrame或ndarray，包含时序特征
        """
        self.model.fit(features)
        return self

    def predict(self, features):
        """预测异常"""
        labels = self.model.predict(features)  # -1=异常, 1=正常
        scores = self.model.decision_function(features)  # 负值越大越异常

        return {
            'is_anomaly': labels == -1,
            'anomaly_score': -scores  # 转换为正向分数（越大越异常）
        }


# ==================== 特征工程：时序特征构建 ====================

def build_time_series_features(series, lags=[1, 7, 14], windows=[7, 14, 30]):
    """
    构建时序特征（用于Isolation Forest）

    Args:
        series: 时间序列数据
        lags: 滞后阶数
        windows: 滚动窗口大小
    
    Returns:
        DataFrame: 特征矩阵
    """
    df = pd.DataFrame({'value': series})

    # 滞后特征
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # 滚动统计特征
    for window in windows:
        df[f'ma_{window}'] = df['value'].rolling(window=window).mean()
        df[f'std_{window}'] = df['value'].rolling(window=window).std()
        df[f'min_{window}'] = df['value'].rolling(window=window).min()
        df[f'max_{window}'] = df['value'].rolling(window=window).max()

    # 时间特征
    if hasattr(series.index, 'dayofweek'):
        df['dayofweek'] = series.index.dayofweek
        df['is_weekend'] = (series.index.dayofweek >= 5).astype(int)

    if hasattr(series.index, 'month'):
        df['month'] = series.index.month

    # 差分特征
    df['diff_1'] = df['value'].diff(1)
    df['diff_7'] = df['value'].diff(7)

    # 去掉NaN
    df = df.dropna()

    return df


# ==================== 母婴出海业务专用函数 ====================

def generate_ecommerce_ts_data(days=90, random_state=42):
    """
    生成母婴出海电商的模拟时序数据

    场景：日订单量，含趋势、季节性、节假日效应、异常注入
    
    Returns:
        tuple: (时序数据, 真实异常标签)
    """
    np.random.seed(random_state)
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')

    # 趋势
    trend = np.linspace(500, 700, days)

    # 季节性（周内模式：周末高）
    seasonal = np.array([20, 10, 5, 0, -5, 30, 35] * (days // 7 + 1))[:days]

    # 节假日效应
    holiday_effect = np.zeros(days)
    black_friday = 45  # 模拟黑五
    if black_friday < days:
        holiday_effect[max(0, black_friday-3):min(days, black_friday+3)] = [50, 80, 150, 200, 100, 50, 20][:min(7, days-black_friday+3)]

    # 噪声
    noise = np.random.normal(0, 30, days)

    # 基础序列
    base = trend + seasonal + holiday_effect + noise

    # 注入异常
    anomalies = np.zeros(days)
    # 异常1：支付通道故障（订单量骤降）
    anomalies[45:47] = [-200, -180]
    # 异常2：促销活动（订单量激增）
    anomalies[60] = 300
    # 异常3：系统Bug（重复订单）
    anomalies[75:77] = [250, 280]

    values = base + anomalies
    values = np.maximum(values, 0)

    return pd.Series(values, index=dates), pd.Series(anomalies != 0, index=dates)


def multi_metric_anomaly_detector(metrics_dict, method='stl', threshold=3.0):
    """
    多指标联合异常检测

    Args:
        metrics_dict: {metric_name: series}
        method: 'stl' 或 'zscore'
        threshold: 异常阈值
    
    Returns:
        tuple: (检测结果字典, 联合异常标签)
    """
    results = {}

    for name, series in metrics_dict.items():
        if method == 'stl':
            detector = STLAnomalyDetector(period=7, threshold=threshold)
            detector.fit(series)
            result = detector.predict()
        else:
            detector = ZScoreAnomalyDetector(threshold=threshold)
            detector.fit(series)
            result = detector.predict()

        results[name] = result

    # 联合判断：多个指标同时异常才是真异常
    n_metrics = len(metrics_dict)
    n_anomalies = sum(results[name]['is_anomaly'].astype(int) for name in results)
    joint_anomaly = n_anomalies >= max(2, n_metrics // 2)

    return results, joint_anomaly


# ==================== 示例代码 ====================

def main():
    """主函数：演示时序异常检测"""
    print("=" * 70)
    print("母婴出海 — 时序异常检测")
    print("=" * 70)

    # 1. 生成模拟数据
    print("\n[1] 生成模拟数据...")
    series, true_anomalies = generate_ecommerce_ts_data(days=90)
    print(f"   数据长度: {len(series)} 天")
    print(f"   均值: {series.mean():.0f}, 标准差: {series.std():.0f}")
    print(f"   真实异常天数: {true_anomalies.sum()}")

    # 2. Z-Score异常检测
    print("\n[2] Z-Score异常检测...")
    zscore_detector = ZScoreAnomalyDetector(threshold=3.0)
    zscore_detector.fit(series)
    zscore_result = zscore_detector.predict()

    detected = zscore_result['is_anomaly'].sum()
    true_positives = (zscore_result['is_anomaly'] & true_anomalies).sum()
    false_positives = (zscore_result['is_anomaly'] & ~true_anomalies).sum()
    false_negatives = (~zscore_result['is_anomaly'] & true_anomalies).sum()

    print(f"   检测到的异常数: {detected}")
    print(f"   真正例(TP): {true_positives}, 假正例(FP): {false_positives}, 假反例(FN): {false_negatives}")
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f"   精确率: {precision:.2f}, 召回率: {recall:.2f}")

    # 3. STL分解异常检测
    print("\n[3] STL分解异常检测...")
    stl_detector = STLAnomalyDetector(period=7, threshold=3.0)
    stl_detector.fit(series)
    stl_result = stl_detector.predict()

    detected = stl_result['is_anomaly'].sum()
    true_positives = (stl_result['is_anomaly'] & true_anomalies).sum()
    false_positives = (stl_result['is_anomaly'] & ~true_anomalies).sum()
    false_negatives = (~stl_result['is_anomaly'] & true_anomalies).sum()

    print(f"   检测到的异常数: {detected}")
    print(f"   真正例(TP): {true_positives}, 假正例(FP): {false_positives}, 假反例(FN): {false_negatives}")
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f"   精确率: {precision:.2f}, 召回率: {recall:.2f}")

    # 4. Isolation Forest
    print("\n[4] Isolation Forest 异常检测...")
    features = build_time_series_features(series)
    if_detector = IFAnomalyDetector(contamination=0.05)
    if_detector.fit(features)
    if_result = if_detector.predict(features)

    # 对齐索引
    if_anomalies = pd.Series(if_result['is_anomaly'], index=features.index)
    true_anomalies_aligned = true_anomalies[features.index]
    
    detected = if_anomalies.sum()
    true_positives = (if_anomalies & true_anomalies_aligned).sum()
    false_positives = (if_anomalies & ~true_anomalies_aligned).sum()
    false_negatives = (~if_anomalies & true_anomalies_aligned).sum()

    print(f"   检测到的异常数: {detected}")
    print(f"   真正例(TP): {true_positives}, 假正例(FP): {false_positives}, 假反例(FN): {false_negatives}")
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f"   精确率: {precision:.2f}, 召回率: {recall:.2f}")

    # 5. 多指标联合检测
    print("\n[5] 多指标联合异常检测...")
    metrics = {
        'orders': series,
        'conversion': pd.Series(np.random.normal(0.025, 0.005, len(series)), index=series.index),
        'aov': pd.Series(np.random.normal(120, 20, len(series)), index=series.index)
    }
    # 让异常同时影响多个指标
    for day in series.index[true_anomalies]:
        metrics['conversion'][day] *= 0.5
        metrics['aov'][day] *= 0.8

    results, joint = multi_metric_anomaly_detector(metrics, method='stl')
    print(f"   联合异常判断: {'是' if joint else '否'}")
    for name, result in results.items():
        print(f"   {name}: 异常数={result['is_anomaly'].sum()}")

    print("\n" + "=" * 70)
    print("异常检测完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
```

---

## ④ 技能关联

### 前置技能
- [[Skill-Time-Series-Forecasting]] — 理解时序分解、趋势、季节性
- **基础统计推断** — 理解正态分布、标准差、置信区间

### 延伸技能
- **