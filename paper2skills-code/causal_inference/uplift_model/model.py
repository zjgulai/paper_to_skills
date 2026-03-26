"""
Uplift Modeling - 元学习框架实现
用于母婴出海电商广告投放归因和促销效果评估
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class UpliftModel:
    """Uplift Modeling 元学习框架"""

    def __init__(self, method='xlearner'):
        """
        初始化 Uplift Model

        Args:
            method: 'tlearner', 'slearner', 或 'xlearner'
        """
        self.method = method
        self.model_treatment = None
        self.model_control = None
        self.model_uplift = None
        self.propensity_model = None
        self.is_fitted = False

    def fit(self, X, treatment, outcome):
        """
        训练 Uplift Model

        Args:
            X: 特征矩阵 (DataFrame 或 np.array)
            treatment: 干预标志 (1=干预组, 0=对照组)
            outcome: 结果变量 (0/1 二分类或连续值)
        """
        X = self._preprocess(X)
        treatment = np.array(treatment)
        outcome = np.array(outcome)

        if self.method == 'tlearner':
            self._fit_tlearner(X, treatment, outcome)
        elif self.method == 'slearner':
            self._fit_slearner(X, treatment, outcome)
        elif self.method == 'xlearner':
            self._fit_xlearner(X, treatment, outcome)

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        预测 uplift score

        Args:
            X: 特征矩阵

        Returns:
            uplift_scores: 每个样本的 uplift score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = self._preprocess(X)

        if self.method == 'tlearner':
            return self._predict_tlearner(X)
        elif self.method == 'slearner':
            return self._predict_slearner(X)
        elif self.method == 'xlearner':
            return self._predict_xlearner(X)

    def _preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        return X

    def _fit_tlearner(self, X, treatment, outcome):
        """T-Learner: 分别训练干预组和对照组模型"""
        X_t = X[treatment == 1]
        X_c = X[treatment == 0]
        y_t = outcome[treatment == 1]
        y_c = outcome[treatment == 0]

        self.model_treatment = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.model_control = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )

        self.model_treatment.fit(X_t, y_t)
        self.model_control.fit(X_c, y_c)

    def _predict_tlearner(self, X):
        """T-Learner 预测"""
        pred_t = self.model_treatment.predict_proba(X)[:, 1]
        pred_c = self.model_control.predict_proba(X)[:, 1]
        return pred_t - pred_c

    def _fit_slearner(self, X, treatment, outcome):
        """S-Learner: 单模型，将干预作为特征"""
        X_with_treatment = np.column_stack([X, treatment])
        self.model_treatment = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.model_treatment.fit(X_with_treatment, outcome)

    def _predict_slearner(self, X):
        """S-Learner 预测"""
        X_treatment = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])
        pred_t = self.model_treatment.predict_proba(X_treatment)[:, 1]
        pred_c = self.model_treatment.predict_proba(X_control)[:, 1]
        return pred_t - pred_c

    def _fit_xlearner(self, X, treatment, outcome):
        """X-Learner: 两阶段元学习"""
        X_t = X[treatment == 1]
        X_c = X[treatment == 0]
        y_t = outcome[treatment == 1]
        y_c = outcome[treatment == 0]

        # 阶段一：训练基础模型
        self.model_treatment = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.model_control = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.model_treatment.fit(X_t, y_t)
        self.model_control.fit(X_c, y_c)

        # 计算 imputed treatment effects
        # 干预组的伪对照结果
        D1 = y_t - self.model_control.predict_proba(X_t)[:, 1]
        # 对照组的伪干预结果
        D0 = self.model_treatment.predict_proba(X_c)[:, 1] - y_c

        # 阶段二：训练 CATE 模型
        self.model_uplift_t = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.model_uplift_c = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.model_uplift_t.fit(X_t, D1)
        self.model_uplift_c.fit(X_c, D0)

        # 训练倾向评分模型
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.propensity_model.fit(X, treatment)

    def _predict_xlearner(self, X):
        """X-Learner 预测"""
        tau1 = self.model_uplift_t.predict(X)
        tau0 = self.model_uplift_c.predict(X)
        propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 组合预测：τ(x) = τ₁(x) + e(x)·(τ₀(x) - τ₁(x))
        return tau1 + propensity * (tau0 - tau1)

    def predict_user_segments(self, X, thresholds=None):
        """
        预测用户分群

        Args:
            X: 特征矩阵
            thresholds: 分群阈值 [negative, low, medium, high]

        Returns:
            segments: 用户分群标签
        """
        uplift_scores = self.predict(X)

        if thresholds is None:
            thresholds = np.percentile(uplift_scores, [20, 40, 60, 80])

        segments = []
        for score in uplift_scores:
            if score < thresholds[0]:
                segments.append('负_uplift')
            elif score < thresholds[1]:
                segments.append('低_uplift')
            elif score < thresholds[2]:
                segments.append('中_uplift')
            elif score < thresholds[3]:
                segments.append('高_uplift')
            else:
                segments.append('最高_uplift')

        return np.array(segments), uplift_scores


# ==================== 示例代码 ====================

def generate_sample_data(n_samples=5000, random_state=42):
    """
    生成模拟数据用于测试

    Args:
        n_samples: 样本数量
        random_state: 随机种子

    Returns:
        X: 特征矩阵
        treatment: 干预标志
        outcome: 结果变量
    """
    np.random.seed(random_state)

    # 生成用户特征
    age = np.random.normal(35, 8, n_samples)
    purchase_history = np.random.exponential(100, n_samples)
    browsing_time = np.random.exponential(5, n_samples)
    cart_value = np.random.exponential(50, n_samples)
    device_type = np.random.choice([0, 1, 2], n_samples)
    country = np.random.choice([0, 1, 2], n_samples)

    X = pd.DataFrame({
        'age': age,
        'purchase_history': purchase_history,
        'browsing_time': browsing_time,
        'cart_value': cart_value,
        'device_type': device_type,
        'country': country
    })

    # 生成干预标志（模拟 A/B 测试）
    treatment = np.random.binomial(1, 0.5, n_samples)

    # 生成潜在结果 Y(0) 和 Y(1)
    # Y(0): 无干预时的购买概率（基准）
    Y0_prob = 1 / (1 + np.exp(-(
        -2 + 0.02 * age +
        0.01 * purchase_history +
        0.1 * browsing_time +
        0.02 * cart_value +
        0.1 * device_type
    )))
    Y0 = np.random.binomial(1, Y0_prob)

    # Y(1): 有干预时的购买概率（包含 uplift）
    # 某些用户对干预更敏感
    uplift_sensitivity = 0.3 * (purchase_history / 100) + 0.2 * (cart_value / 50)
    Y1_prob = Y0_prob + uplift_sensitivity
    Y1_prob = np.clip(Y1_prob, 0, 1)
    Y1 = np.random.binomial(1, Y1_prob)

    # 观测结果：根据实际干预情况选择
    outcome = np.where(treatment == 1, Y1, Y0)

    return X, treatment, outcome


def main():
    """主函数：演示 Uplift Model 的使用"""
    print("=" * 60)
    print("Uplift Modeling - 元学习框架测试")
    print("=" * 60)

    # 1. 生成模拟数据
    print("\n[1] 生成模拟数据...")
    X, treatment, outcome = generate_sample_data(n_samples=5000)
    print(f"   样本量: {len(X)}")
    print(f"   干预组: {sum(treatment)} ({sum(treatment)/len(treatment)*100:.1f}%)")
    print(f"   对照组: {len(treatment) - sum(treatment)} ({(len(treatment)-sum(treatment))/len(treatment)*100:.1f}%)")
    print(f"   整体转化率: {sum(outcome)/len(outcome)*100:.1f}%")

    # 2. 划分训练集和测试集
    print("\n[2] 划分训练集和测试集...")
    X_train, X_test, treatment_train, treatment_test, outcome_train, outcome_test = train_test_split(
        X, treatment, outcome, test_size=0.2, random_state=42
    )
    print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 3. 训练 T-Learner
    print("\n[3] 训练 T-Learner...")
    model_t = UpliftModel(method='tlearner')
    model_t.fit(X_train, treatment_train, outcome_train)
    uplift_t = model_t.predict(X_test)
    print(f"   T-Learner uplift 均值: {uplift_t.mean():.4f}")
    print(f"   T-Learner uplift 标准差: {uplift_t.std():.4f}")

    # 4. 训练 X-Learner
    print("\n[4] 训练 X-Learner...")
    model_x = UpliftModel(method='xlearner')
    model_x.fit(X_train, treatment_train, outcome_train)
    uplift_x = model_x.predict(X_test)
    print(f"   X-Learner uplift 均值: {uplift_x.mean():.4f}")
    print(f"   X-Learner uplift 标准差: {uplift_x.std():.4f}")

    # 5. 用户分群
    print("\n[5] 用户分群...")
    segments, scores = model_x.predict_user_segments(X_test)
    for seg in ['负_uplift', '低_uplift', '中_uplift', '高_uplift', '最高_uplift']:
        count = sum(segments == seg)
        print(f"   {seg}: {count} ({count/len(segments)*100:.1f}%)")

    # 6. 输出预测结果示例
    print("\n[6] 预测结果示例 (前10个样本):")
    print("-" * 60)
    result_df = pd.DataFrame({
        'age': X_test['age'].values[:10],
        'purchase_history': X_test['purchase_history'].values[:10],
        'uplift_score': uplift_x[:10],
        'segment': segments[:10]
    })
    print(result_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

    return model_x


if __name__ == '__main__':
    model = main()
