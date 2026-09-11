"""
Demand Forecasting 预测服务
集成数据加载和预测模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import SalesDataLoader
from time_series.demand_forecasting import DemandForecaster
import pandas as pd
from datetime import datetime, timedelta


class ForecastService:
    """销量预测服务"""

    def __init__(self):
        self.sales_loader = SalesDataLoader()
        self.forecasters = {}
        self.initialized = False

    def initialize(self, skus: list = None):
        """初始化预测模型"""
        if skus is None:
            skus = self.sales_loader.get_skus()

        for sku in skus:
            self._train_forecaster(sku)

        self.initialized = True
        print(f"预测服务初始化完成，SKU数量: {len(skus)}")
        return self

    def _train_forecaster(self, sku: str):
        """训练单个SKU的预测模型"""
        ts_data = self.sales_loader.get_sku_time_series(sku)
        dates = ts_data['date']
        values = ts_data['sales_quantity'].values

        forecaster = DemandForecaster(model_type='simple')
        forecaster.fit(dates, values)
        self.forecasters[sku] = forecaster

    def predict(self, sku: str, n_periods: int = 4) -> dict:
        """预测未来销量"""
        if sku not in self.forecasters:
            self._train_forecaster(sku)

        forecaster = self.forecasters[sku]
        predictions, lower, upper = forecaster.predict(n_periods)

        last_date = self.sales_loader.get_sku_time_series(sku)['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=7),
            periods=n_periods,
            freq='W'
        )

        results = []
        for i in range(n_periods):
            results.append({
                'week': i + 1,
                'date': future_dates[i].strftime('%Y-%m-%d'),
                'prediction': round(predictions[i], 0),
                'lower_bound': round(lower[i], 0),
                'upper_bound': round(upper[i], 0),
                'confidence': 0.95
            })

        return {
            'sku': sku,
            'forecast': results,
            'generated_at': datetime.now().isoformat()
        }

    def predict_all(self, n_periods: int = 4) -> dict:
        """预测所有SKU"""
        results = {}
        for sku in self.forecasters.keys():
            results[sku] = self.predict(sku, n_periods)
        return results

    def get_inventory_recommendations(self, n_periods: int = 4) -> dict:
        """生成库存建议"""
        predictions = self.predict_all(n_periods)

        recommendations = {}
        for sku, pred in predictions.items():
            total_pred = sum([f['prediction'] for f in pred['forecast']])
            max_upper = max([f['upper_bound'] for f in pred['forecast']])
            safety_stock = max_upper - (total_pred / n_periods)

            recommendations[sku] = {
                'recommended_stock': int(total_pred + safety_stock * 0.5),
                'safety_stock': int(safety_stock * 0.5),
                'weekly_forecast': [f['prediction'] for f in pred['forecast']],
                'priority': 'high' if total_pred > 500 else 'medium'
            }

        return recommendations


def main():
    print("=" * 60)
    print("Demand Forecasting 预测服务测试")
    print("=" * 60)

    print("\n[1] 初始化预测服务...")
    service = ForecastService()
    service.initialize()

    print("\n[2] 预测 SKU 销量...")
    result = service.predict('奶粉_1段', n_periods=4)
    print(f"   SKU: {result['sku']}")
    for week in result['forecast']:
        print(f"   Week {week['week']}: {week['prediction']} [{week['lower_bound']}, {week['upper_bound']}]")

    print("\n[3] 生成库存建议...")
    recommendations = service.get_inventory_recommendations()
    for sku, rec in recommendations.items():
        print(f"   {sku}: 推荐库存={rec['recommended_stock']}, 安全库存={rec['safety_stock']}, 优先级={rec['priority']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()