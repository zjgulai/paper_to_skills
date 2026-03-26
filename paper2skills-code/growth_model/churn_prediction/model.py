"""
Customer Churn Prediction
用于母婴出海电商用户流失预测
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def _create_features(self, df):
        """创建特征"""
        features = pd.DataFrame()

        features['days_since_registration'] = (pd.Timestamp.now() - df['register_date']).dt.days
        features['days_since_last_purchase'] = (pd.Timestamp.now() - df['last_purchase_date']).dt.days
        features['total_purchase_count'] = df['purchase_count']
        features['total_purchase_amount'] = df['purchase_amount']
        features['avg_purchase_amount'] = df['purchase_amount'] / df['purchase_count'].clip(lower=1)

        features['browse_count_7d'] = df['browse_count_7d']
        features['browse_count_30d'] = df['browse_count_30d']
        features['browse_count_90d'] = df['browse_count_90d']
        features['cart_add_count'] = df['cart_add_count']
        features['wishlist_count'] = df['wishlist_count']

        features['active_days_30d'] = df['active_days_30d']
        features['active_days_90d'] = df['active_days_90d']
        features['purchase_frequency_30d'] = df['purchase_count_30d']
        features['purchase_frequency_90d'] = df['purchase_count_90d']

        features['browse_to_purchase_ratio'] = df['purchase_count'] / df['browse_count_90d'].clip(lower=1)
        features['recency_frequency'] = (1 / (features['days_since_last_purchase'] + 1)) * df['purchase_count_90d']

        return features

    def fit(self, df, target):
        X = self._create_features(df)
        self.feature_names = X.columns.tolist()
        X = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        self.model.fit(X_scaled, target)
        self.is_fitted = True
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        X = self._create_features(df).fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_churn_risk(self, df):
        probs = self.predict(df)
        risk_levels = []
        for p in probs:
            if p >= 0.7: risk_levels.append('极高')
            elif p >= 0.5: risk_levels.append('高')
            elif p >= 0.3: risk_levels.append('中')
            elif p >= 0.15: risk_levels.append('低')
            else: risk_levels.append('极低')
        return risk_levels, probs


def generate_sample_data(n_users=5000):
    np.random.seed(42)
    data = {
        'user_id': range(1, n_users + 1),
        'register_date': pd.date_range('2023-01-01', periods=n_users, freq='h'),
        'last_purchase_date': pd.date_range('2024-01-01', periods=n_users, freq='h') - pd.Timedelta(days=np.random.randint(0, 90, n_users)),
        'purchase_count': np.random.poisson(3, n_users),
        'purchase_amount': np.random.exponential(500, n_users),
        'browse_count_7d': np.random.poisson(10, n_users),
        'browse_count_30d': np.random.poisson(30, n_users),
        'browse_count_90d': np.random.poisson(80, n_users),
        'cart_add_count': np.random.poisson(2, n_users),
        'wishlist_count': np.random.poisson(1, n_users),
        'active_days_30d': np.random.poisson(5, n_users),
        'active_days_90d': np.random.poisson(15, n_users),
        'purchase_count_30d': np.random.poisson(1, n_users),
        'purchase_count_90d': np.random.poisson(3, n_users),
    }
    df = pd.DataFrame(data)
    churn_prob = 0.3 * (df['active_days_30d'] < 2).astype(int) + 0.3 * (df['purchase_count_30d'] == 0).astype(int) + 0.2 * (df['browse_count_30d'] < 5).astype(int) + 0.2 * np.random.random(n_users)
    target = (np.random.random(n_users) < np.clip(churn_prob, 0, 1)).astype(int)
    return df, target


def main():
    print("=" * 60)
    print("Customer Churn Prediction 测试")
    print("=" * 60)

    print("\n[1] 生成模拟数据...")
    df, target = generate_sample_data(5000)
    print(f"   用户数: {len(df)}, 流失率: {target.mean()*100:.1f}%")

    print("\n[2] 训练模型...")
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
    predictor = ChurnPredictor('gradient_boosting')
    predictor.fit(X_train, y_train)

    print("\n[3] 预测评估...")
    probs = predictor.predict(X_test)
    auc = roc_auc_score(y_test, probs)
    print(f"   AUC: {auc:.4f}")

    print("\n[4] 风险分层...")
    risks, _ = predictor.predict_churn_risk(X_test)
    for level in ['极高', '高', '中', '低', '极低']:
        count = sum([r == level for r in risks])
        print(f"   {level}: {count} ({count/len(risks)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("测试完成!")
    return predictor


if __name__ == '__main__':
    main()
