# Skill Card: Customer Churn Prediction (用户流失预测)

---

## ① 算法原理

### 核心思想
用户流失预测解决的核心问题是：**识别哪些用户即将停止使用产品/服务**，从而提前采取挽留措施。与被动等待用户流失后分析不同，预测模型可以提前 7-30 天预警，让运营团队有足够时间干预。

### 数学直觉

**Logistic 回归模型**：
$$P(churn=1|X) = \frac{1}{1 + e^{-z}}$$
其中 $z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$

- 将线性组合映射到 (0,1) 区间
- $\beta_i$ 为特征权重，可解释

**特征重要性（基于树模型）**：
- **Information Gain**: $IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$
- **Gain**: $Gain(S, A) = \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)$

### 关键假设
- **历史可预测未来**：过去流失模式可预测未来
- **特征稳定性**：特征分布不随时间剧烈变化
- **定义清晰**：明确定义"流失"（如 90 天未活跃）

---

## ② 吸奶器出海应用案例

### 场景一：吸奶器配件复购用户流失预警

**业务问题**：
购买吸奶器的妈妈用户（如定期更换配件：喇叭罩、鸭嘴阀、储奶袋）是核心复购用户。但部分复购用户会逐渐减少购买甚至不再访问，需要提前识别并挽留。

**数据要求**：
- 用户特征：注册时间、首次购买吸奶器时间、历史购买次数/金额
- 行为特征：浏览配件页面数、加购未购次数、收藏商品数
- 时序特征：近 7/30/90 天活跃天数、登录频次
- 标签：90 天未购买配件 = 流失

**预期产出**：
- 每个用户的流失概率（0-1）
- 高风险用户清单（top 20%）
- 挽留优先级排序

**业务价值**：
- 流失率降低 15-25%
- 挽留成本降低 30%（精准触达）
- 挽回收入：假设月流失用户贡献 50 万，挽回 20% = 10 万/月

---

### 场景二：沉默用户激活预测

**业务问题**：
部分注册用户首次购买后，逐渐沉默（不再访问网站）。需要识别哪些沉默用户可以通过优惠激活，哪些会自然回流。

**数据要求**：
- 沉默标记：30 天未访问
- 历史行为：购买频次、加购行为、浏览深度
- 营销响应历史：历史领券/点击记录

**预期产出**：
- 激活概率评分
- 最优触达策略（发券/推送/短信）
- 预期 ROI

**业务价值**：
- 沉默用户激活率提升 20%+
- 营销成本降低 25%
- 预算聚焦高ROI用户

---

## ③ 代码模板

```python
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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, f1_score
)
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    """用户流失预测器"""

    def __init__(self, model_type='gradient_boosting'):
        """
        初始化

        Args:
            model_type: 'logistic', 'random_forest', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def _create_features(self, df):
        """创建特征"""
        features = pd.DataFrame()

        # 基础特征
        features['days_since_registration'] = (pd.now() - df['register_date']).dt.days
        features['days_since_last_purchase'] = (pd.now() - df['last_purchase_date']).dt.days
        features['total_purchase_count'] = df['purchase_count']
        features['total_purchase_amount'] = df['purchase_amount']
        features['avg_purchase_amount'] = df['purchase_amount'] / df['purchase_count'].clip(lower=1)

        # 行为特征
        features['browse_count_7d'] = df['browse_count_7d']
        features['browse_count_30d'] = df['browse_count_30d']
        features['browse_count_90d'] = df['browse_count_90d']
        features['cart_add_count'] = df['cart_add_count']
        features['wishlist_count'] = df['wishlist_count']

        # 时序特征
        features['active_days_30d'] = df['active_days_30d']
        features['active_days_90d'] = df['active_days_90d']
        features['purchase_frequency_30d'] = df['purchase_count_30d']
        features['purchase_frequency_90d'] = df['purchase_count_90d']

        # 衍生特征
        features['browse_to_purchase_ratio'] = (
            df['purchase_count'] / df['browse_count_90d'].clip(lower=1)
        )
        features['cart_to_purchase_ratio'] = (
            df['purchase_count'] / df['cart_add_count'].clip(lower=1)
        )
        features['recency_frequency'] = (
            1 / (features['days_since_last_purchase'] + 1)
        ) * df['purchase_count_90d']

        return features

    def fit(self, df, target):
        """
        训练模型

        Args:
            df: 用户数据 DataFrame
            target: 流失标签 (1=流失, 0=未流失)
        """
        X = self._create_features(df)
        self.feature_names = X.columns.tolist()

        # 填充缺失值
        X = X.fillna(0)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 选择模型
        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        self.model.fit(X_scaled, target)
        self.is_fitted = True
        return self

    def predict(self, df):
        """
        预测流失概率

        Args:
            df: 用户数据 DataFrame

        Returns:
            probabilities: 流失概率
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X = self._create_features(df)
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_churn_risk(self, df, threshold=0.5):
        """
        预测流失风险等级

        Args:
            df: 用户数据
            threshold: 阈值

        Returns:
            risk_levels: 风险等级
        """
        probabilities = self.predict(df)

        risk_levels = []
        for p in probabilities:
            if p >= 0.7:
                risk_levels.append('极高')
            elif p >= 0.5:
                risk_levels.append('高')
            elif p >= 0.3:
                risk_levels.append('中')
            elif p >= 0.15:
                risk_levels.append('低')
            else:
                risk_levels.append('极低')

        return risk_levels, probabilities

    def get_feature_importance(self, top_n=10):
        """获取特征重要性"""
        if self.model_type in ['random_forest', 'gradient_boosting']:
            importance = self.model.feature_importances_
            sorted_idx = np.argsort(importance)[::-1][:top_n]

            return {
                self.feature_names[i]: importance[i]
                for i in sorted_idx
            }
        else:
            # Logistic 回归使用系数
            importance = np.abs(self.model.coef_[0])
            sorted_idx = np.argsort(importance)[::-1][:top_n]

            return {
                self.feature_names[i]: importance[i]
                for i in sorted_idx
            }


# ==================== 示例代码 ====================

def generate_sample_data(n_users=5000):
    """生成模拟数据"""
    np.random.seed(42)

    # 模拟用户数据
    data = {
        'user_id': range(1, n_users + 1),
        'register_date': pd.date_range('2023-01-01', periods=n_users, freq='h'),
        'last_purchase_date': pd.date_range('2024-01-01', periods=n_users, freq='h') - pd.Timedelta(days=np.random.randint(0, 90, n_users)),
        'purchase_count': np.random.poisson(3, n_users),
        'purchase_amount': np.random.exponential(500, n_users),
    }

    # 行为数据
    data['browse_count_7d'] = np.random.poisson(10, n_users)
    data['browse_count_30d'] = np.random.poisson(30, n_users)
    data['browse_count_90d'] = np.random.poisson(80, n_users)
    data['cart_add_count'] = np.random.poisson(2, n_users)
    data['wishlist_count'] = np.random.poisson(1, n_users)

    # 时序数据
    data['active_days_30d'] = np.random.poisson(5, n_users)
    data['active_days_90d'] = np.random.poisson(15, n_users)
    data['purchase_count_30d'] = np.random.poisson(1, n_users)
    data['purchase_count_90d'] = np.random.poisson(3, n_users)

    df = pd.DataFrame(data)

    # 生成流失标签（基于规则）
    # 流失概率与活跃度负相关
    churn_prob = (
        0.3 * (df['active_days_30d'] < 2).astype(int) +
        0.3 * (df['purchase_count_30d'] == 0).astype(int) +
        0.2 * (df['browse_count_30d'] < 5).astype(int) +
        0.2 * np.random.random(n_users)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    target = (np.random.random(n_users) < churn_prob).astype(int)

    return df, target


def main():
    """主函数"""
    print("=" * 60)
    print("Customer Churn Prediction 测试")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    df, target = generate_sample_data(n_users=5000)
    print(f"   用户数: {len(df)}")
    print(f"   流失率: {target.mean()*100:.1f}%")

    # 2. 划分数据
    print("\n[2] 划分训练/测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.2, random_state=42
    )
    print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 3. 训练模型
    print("\n[3] 训练预测模型...")
    predictor = ChurnPredictor(model_type='gradient_boosting')
    predictor.fit(X_train, y_train)
    print(f"   模型类型: {predictor.model_type}")

    # 4. 预测
    print("\n[4] 预测流失概率...")
    probabilities = predictor.predict(X_test)

    # 5. 评估
    print("\n[5] 评估模型...")
    auc = roc_auc_score(y_test, probabilities)
    print(f"   AUC: {auc:.4f}")

    # 阈值优化
    thresholds = [0.3, 0.4, 0.5, 0.6]
    for th in thresholds:
        predictions = (probabilities >= th).astype(int)
        f1 = f1_score(y_test, predictions)
        precision = (predictions & y_test).sum() / predictions.sum() if predictions.sum() > 0 else 0
        recall = (predictions & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
        print(f"   阈值 {th}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    # 6. 风险分层
    print("\n[6] 风险分层...")
    risk_levels, probs = predictor.predict_churn_risk(X_test)
    print(f"   极高风险: {sum([r=='极高' for r in risk_levels])} ({sum([r=='极高' for r in risk_levels])/len(risk_levels)*100:.1f}%)")
    print(f"   高风险: {sum([r=='高' for r in risk_levels])} ({sum([r=='高' for r in risk_levels])/len(risk_levels)*100:.1f}%)")
    print(f"   中风险: {sum([r=='中' for r in risk_levels])} ({sum([r=='中' for r in risk_levels])/len(risk_levels)*100:.1f}%)")
    print(f"   低风险: {sum([r=='低' for r in risk_levels])} ({sum([r=='低' for r in risk_levels])/len(risk_levels)*100:.1f}%)")

    # 7. 特征重要性
    print("\n[7] 特征重要性 (Top 10):")
    importance = predictor.get_feature_importance(10)
    for feature, score in importance.items():
        print(f"   {feature}: {score:.4f}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return predictor


if __name__ == '__main__':
    predictor = main()
