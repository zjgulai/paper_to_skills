# Skill Card: Uplift Modeling (元学习框架)

---

## ① 算法原理

### 核心思想
Uplift Modeling（提升建模）解决的核心问题是：**识别哪些用户最有可能因为某个干预（如促销、广告）而产生正向行为**。与传统的响应预测模型不同，uplift model 预测的是"干预带来的增量效果"，而非"干预后的绝对结果"。

### 数学直觉

**条件平均处理效应 (CATE)**：
$$\tau(x) = E[Y(1)|X=x] - E[Y(0)|X=x]$$

其中：
- Y(1) 是接受干预后的结果
- Y(0) 是未接受干预的结果
- x 是用户特征

**T-Learner 方法**：分别训练两个模型
- 模型 μ₁(x)：预测干预组结果
- 模型 μ₀(x)：预测对照组结果
- CATE 估计：τ̂(x) = μ̂₁(x) - μ̂₀(x)

**X-Learner 方法**（更高效）：
1. 阶段一：用 T-Learner 估计 τ₁(x) 和 τ₀(x)
2. 阶段二：计算 imputed treatment effect
   - 对干预组：τ₀(xᵢ) ≈ Yᵢ(1) - μ̂₀(xᵢ)
   - 对对照组：τ₁(xᵢ) ≈ μ̂₁(xᵢ) - Yᵢ(0)
3. 最终估计：τ̂(x) = τ₁(x) + e(x)·(τ₀(x) - τ₁(x))
   - e(x) 是倾向评分 P(T=1|X=x)

### 关键假设
- **SUTVA**：稳定单元处理值假设，用户间无干扰
- **条件独立假设**：给定特征 X，处理分配 T 与潜在结果 (Y(0), Y(1)) 条件独立
- **重叠假设**：对于所有 x，P(T=1|X=x) ∈ (0,1)，即每个用户都有被干预或不干预的可能性

---

## ② 母婴出海应用案例

### 场景一：广告投放归因优化

**业务问题**：
母婴出海电商在 Facebook/Google/TikTok 投放广告时，希望识别哪些用户群体最有可能因为广告点击而完成购买。传统方法是预测转化概率，但高转化概率用户可能本身就是高购买意愿用户，广告只是"锦上添花"。我们需要找到广告"增量转化"的用户，即如果不投广告就不会购买，但投了广告就会购买的用户。

**数据要求**：
- 用户特征：年龄、性别、国家、设备类型、浏览行为、加购行为、购买历史等
- 实验数据：A/B 测试数据或历史渐进式投放数据
- 标签：是否有曝光、是否点击、是否购买
- 数据量：建议至少 10,000 条样本（干预组和对照组各 5,000+）

**预期产出**：
- 每个用户的 uplift score：预测该用户看到广告后的增量购买概率
- 用户分群：高 uplift / 中 uplift / 低 uplift / 无 uplift / 负 uplift
- 投放策略：优先投放高 uplift 用户，减少对负 uplift 用户的投放

**业务价值**：
- 广告预算节省 20-30%（减少对无增量效果用户的投放）
- 转化率提升 15-25%（聚焦高增量效果用户）
- ROI 计算：假设月广告预算 50 万，通过优化可节省 10-15 万，同时提升 5-10 万销售额

---

### 场景二：新客促销效果评估

**业务问题**：
母婴出海电商对新用户发放首单优惠券（如满 100 减 20），但并非所有新用户都需要优惠券来促成首单。部分用户即使没有优惠券也会购买，给他们发券浪费了营销成本。需要识别哪些新用户是"优惠券敏感型"（有券才会下单），避免给"自然购买型"用户发券。

**数据要求**：
- 新用户特征：来源渠道、浏览页面数、加购商品数、加购金额、注册时间、设备类型
- 干预数据：历史发券数据（有券/无券）
- 标签：是否使用优惠券、是否完成首单、订单金额

**预期产出**：
- 每个新用户的优惠券敏感度评分
- 用户分群：
  - 自然购买型（负 uplift，发券反而降低购买意愿）
  - 券后必买型（高 uplift，券是成交关键）
  - 券无影响型（低 uplift，有没有券都会买）

**业务价值**：
- 优惠券成本降低 30-40%（减少对自然购买型的发放）
- 首单转化率维持或提升 5-10%
- ROI 计算：假设月发放优惠券成本 20 万，通过优化可节省 6-8 万

---

## ③ 代码模板

```python
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
```

---

## ④ 技能关联

### 前置技能
- **基础统计推断**：理解假设检验、置信区间、P值等概念
- **倾向评分分析**：掌握倾向评分的计算和应用场景
- **机器学习基础**：熟悉分类/回归模型原理（GradientBoosting, LogisticRegression）

### 延伸技能
- **因果森林 (Causal Forest)**：基于因果森林的 Uplift Modeling，适合大规模数据
- **Doubly Robust Estimation**：结合倾向评分和结果模型的双重稳健估计
- **A/B 实验设计**：掌握实验设计原理，与 Uplift Model 结合进行实验效果评估

### 可组合技能
- **广告投放归因**：与广告归因模型结合，识别广告的真实增量效果
- **用户生命周期价值 (LTV)**：结合 LTV 预测，优先对高 LTV 用户进行干预
- **促销效果评估**：与促销策略分析结合，评估不同促销力度的效果差异

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 广告投放归因 | 预算节省 20-30%（假设月预算 50 万，节省 10-15 万） | 开发 1-2 周，数据接入 1 周 | 10-15x |
| 优惠券优化 | 成本降低 30-40%（假设月成本 20 万，节省 6-8 万） | 开发 1 周，数据接入 1 周 | 6-8x |

### 实施难度
**评分：⭐⭐☆☆☆（2/5星）**

- 数据要求：需要 A/B 测试数据或准实验数据
- 技术门槛：中等，需要理解因果推断基本概念
- 工程复杂度：低，模型训练和预测均为标准 ML 流程
- 维护成本：低，模型可定期更新

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- 业务价值高：直接关联广告投放和促销成本优化
- 见效快：1-2 周可完成 POC，1 个月可上线
- 可落地性强：母婴出海业务场景明确
- 数据依赖：需要历史实验数据，初期数据量可能不足

### 评估依据
1. **广告投放归因**是母婴出海电商的核心痛点，广告成本占总营销成本的 40-60%
2. Uplift Modeling 有成熟的工程实现，技术风险低
3. 与现有 A/B 实验体系天然结合，可复用现有数据
4. 初期可从新用户投放归因切入，逐步扩展到全渠道
