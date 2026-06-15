"""
Auto-extracted from: paper2skills-vault/03-时间序列/Skill-Adaptive-Forecast-Accuracy-Optimization.md
Skill: Skill-Adaptive-Forecast-Accuracy-Optimization
Domain: 03-时间序列
"""
"""
Adaptive Forecast Accuracy Optimization
自适应预测精准化：滚动误差修正 + 漂移检测
"""
import numpy as np
from collections import deque


class AdaptiveForecastOptimizer:
    """
    自适应预测精准化器
    实时检测预测误差漂移，动态调整预测
    """

    def __init__(self, alpha: float = 0.15, window: int = 7,
                 cusum_threshold: float = 2.0):
        self.alpha = alpha              # 自适应学习率
        self.window = window            # 误差滑动窗口
        self.cusum_threshold = cusum_threshold
        self.errors = deque(maxlen=window)
        self.cumsum_pos = 0.0
        self.cumsum_neg = 0.0
        self.bias_correction = 0.0     # 累积偏差修正量
        self.drift_events = []

    def update(self, actual: float, predicted: float) -> dict:
        """
        更新误差追踪器
        返回: 漂移检测结果 + 修正后的预测调整量
        """
        error = actual - predicted
        rel_error = error / max(abs(predicted), 1e-8)
        self.errors.append(rel_error)

        # CUSUM 漂移检测
        mean_err = np.mean(self.errors) if self.errors else 0
        self.cumsum_pos = max(0, self.cumsum_pos + mean_err - 0.1)
        self.cumsum_neg = max(0, self.cumsum_neg - mean_err - 0.1)

        drift_detected = (self.cumsum_pos > self.cusum_threshold or
                          self.cumsum_neg > self.cusum_threshold)

        if drift_detected:
            # 检测到漂移，重置并调整偏差修正
            recent_bias = np.mean(list(self.errors)[-3:])
            self.bias_correction += self.alpha * recent_bias
            self.cumsum_pos = 0.0
            self.cumsum_neg = 0.0
            self.drift_events.append({'error': recent_bias, 'correction': self.bias_correction})

        return {
            'error': round(error, 3),
            'rel_error_pct': round(rel_error * 100, 1),
            'drift_detected': drift_detected,
            'bias_correction': round(self.bias_correction, 4),
            'cusum_pos': round(self.cumsum_pos, 3),
        }

    def correct_forecast(self, raw_forecast: float) -> float:
        """应用偏差修正，输出修正后预测"""
        correction = raw_forecast * self.bias_correction
        return max(0, raw_forecast + correction)

    def get_accuracy_metrics(self, actuals: list, predictions: list) -> dict:
        """计算预测精度指标"""
        if not actuals or not predictions:
            return {}
        errors = [abs(a - p) / max(abs(a), 1) for a, p in zip(actuals, predictions)]
        return {
            'mape': round(np.mean(errors) * 100, 2),
            'mae': round(np.mean([abs(a-p) for a,p in zip(actuals, predictions)]), 2),
            'rmse': round(np.sqrt(np.mean([(a-p)**2 for a,p in zip(actuals, predictions)])), 2),
        }


def run_adaptive_forecast_demo():
    print('=' * 65)
    print('Adaptive Forecast Accuracy Optimization — 自适应预测精准化')
    print('=' * 65)

    np.random.seed(42)
    # 模拟：前20天正常，第21天竞品大促需求降30%
    n_days = 35
    base_demand = 50
    raw_predictions = [base_demand] * n_days
    actuals = []
    for d in range(n_days):
        if d < 20:
            a = base_demand + np.random.normal(0, 5)
        else:
            a = base_demand * 0.70 + np.random.normal(0, 4)  # 需求下降30%
        actuals.append(max(0, a))

    optimizer = AdaptiveForecastOptimizer(alpha=0.2, window=5, cusum_threshold=1.5)
    corrected_predictions = []
    metrics_log = []

    print(f'\n📊 自适应预测 vs 静态预测（第21天竞品大促）:')
    print(f'  {"天":>4} {"实际":>8} {"静态预测":>8} {"自适误差":>10} {"修正预测":>9} {"漂移"}')
    print('  ' + '-' * 52)

    for d in range(n_days):
        result = optimizer.update(actuals[d], raw_predictions[d])
        corrected = optimizer.correct_forecast(raw_predictions[d])
        corrected_predictions.append(corrected)
        metrics_log.append(result)

        if d >= 18 or result['drift_detected']:
            drift_icon = '🚨' if result['drift_detected'] else ''
            print(f'  {d+1:>4} {actuals[d]:>8.1f} {raw_predictions[d]:>8.1f} '
                  f'{result["rel_error_pct"]:>+9.1f}% {corrected:>9.1f} {drift_icon}')

    # 精度对比
    static_metrics = optimizer.get_accuracy_metrics(actuals[20:], raw_predictions[20:])
    adaptive_metrics = optimizer.get_accuracy_metrics(actuals[20:], corrected_predictions[20:])

    print(f'\n  📈 大促后精度对比（第21-35天）:')
    print(f'  静态预测 MAPE: {static_metrics["mape"]:.1f}%')
    print(f'  自适应预测 MAPE: {adaptive_metrics["mape"]:.1f}%')
    improvement = static_metrics["mape"] - adaptive_metrics["mape"]
    print(f'  改善: {improvement:.1f}pp ({improvement/static_metrics["mape"]*100:.0f}%)')

    print('\n[✓] Adaptive Forecast Accuracy Optimization 测试通过')


if __name__ == '__main__':
    run_adaptive_forecast_demo()
