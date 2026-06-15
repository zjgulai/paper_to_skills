"""
Auto-extracted from: paper2skills-vault/04-供应链/Skill-Demand-Forecasting-Supply-Chain.md
Skill: Skill-Demand-Forecasting-Supply-Chain
Domain: 04-供应链
"""
"""
Demand Forecasting for Supply Chain — 供应链需求预测
支持：分层预测、促销效应建模、多SKU批量预测
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


class SupplyChainDemandForecaster:
    """供应链需求预测器"""

    def __init__(self, model=None):
        self.model = model or GradientBoostingRegressor(n_estimators=100, max_depth=4)
        self.is_fitted = False

    def build_features(self, df):
        """构建供应链需求预测特征"""
        df = df.copy()
        df = df.sort_values(['sku', 'date'])

        # 滞后特征
        for lag in [1, 2, 4, 12]:
            df[f'sales_lag_{lag}'] = df.groupby('sku')['sales'].shift(lag)

        # 滚动统计
        for window in [4, 12]:
            df[f'sales_ma_{window}'] = df.groupby('sku')['sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        # 促销特征
        if 'is_promo' in df.columns:
            df['promo_depth'] = df.get('discount_rate', 0)
            df['promo_lag_1'] = df.groupby('sku')['is_promo'].shift(1)

        # 季节特征
        df['week_of_year'] = pd.to_datetime(df['date']).dt.isocalendar().week
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['is_holiday'] = df['week_of_year'].isin([47, 48, 49, 50, 51]).astype(int)

        # 价格特征
        if 'price' in df.columns:
            df['price_lag_1'] = df.groupby('sku')['price'].shift(1)
            df['price_change'] = (df['price'] - df['price_lag_1']) / df['price_lag_1']

        return df.dropna()

    def fit(self, df, feature_cols, target_col='sales'):
        """训练模型"""
        df_train = df.dropna(subset=feature_cols + [target_col])
        X = df_train[feature_cols]
        y = df_train[target_col]
        self.model.fit(X, y)
        self.feature_cols = feature_cols
        self.is_fitted = True
        return self

    def predict(self, df):
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = df[self.feature_cols].fillna(0)
        return np.maximum(self.model.predict(X), 0)

    def evaluate(self, df, feature_cols, target_col='sales'):
        """评估：WAPE（加权绝对百分比误差）"""
        df_eval = df.dropna(subset=feature_cols + [target_col])
        preds = self.predict(df_eval)
        actuals = df_eval[target_col].values
        wape = np.sum(np.abs(preds - actuals)) / np.sum(actuals)
        mae = mean_absolute_error(actuals, preds)
        return {'wape': wape, 'mae': mae}


def hierarchical_reconcile(bottom_up_preds, proportions):
    """
    分层预测协调

    Args:
        bottom_up_preds: SKU级预测 {sku: pred}
        proportions: 各SKU占总需求的历史比例
    """
    total_pred = sum(bottom_up_preds.values())
    reconciled = {}
    for sku, pred in bottom_up_preds.items():
        # 用历史比例校准
        reconciled[sku] = total_pred * proportions.get(sku, pred / total_pred)
    return reconciled


def generate_supply_chain_data(n_skus=10, n_weeks=104, random_state=42):
    """生成供应链需求预测模拟数据"""
    np.random.seed(random_state)
    dates = pd.date_range(end='2026-05-15', periods=n_weeks, freq='W')

    data = []
    for sku_id in range(n_skus):
        base_demand = np.random.uniform(50, 200)
        trend = np.linspace(0, 20, n_weeks)
        seasonal = 30 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

        # 促销（第45、47、49周——黑五期间）
        promo = np.zeros(n_weeks)
        promo[[44, 46, 48]] = [100, 150, 80]

        noise = np.random.normal(0, 15, n_weeks)
        sales = base_demand + trend + seasonal + promo + noise
        sales = np.maximum(sales, 0)

        for i, date in enumerate(dates):
            data.append({
                'sku': f'SKU_{sku_id:03d}',
                'date': date,
                'sales': sales[i],
                'is_promo': 1 if promo[i] > 0 else 0,
                'discount_rate': 0.3 if promo[i] > 0 else 0,
                'price': 100 if promo[i] == 0 else 70
            })

    return pd.DataFrame(data)


if __name__ == '__main__':
    df = generate_supply_chain_data()
    forecaster = SupplyChainDemandForecaster()
    df_features = forecaster.build_features(df)

    feature_cols = [c for c in df_features.columns if c.startswith(('sales_lag', 'sales_ma', 'promo', 'week', 'price'))]
    train = df_features[df_features['date'] < '2026-01-01']
    test = df_features[df_features['date'] >= '2026-01-01']

    forecaster.fit(train, feature_cols)
    metrics = forecaster.evaluate(test, feature_cols)
    print(f"WAPE: {metrics['wape']:.2%}, MAE: {metrics['mae']:.1f}")
