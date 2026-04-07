# Skill Card: 智能归因 - 因果森林 (Causal Forest)

---

## ① 算法原理

### 核心思想
Causal Forest（因果森林）是 Uplift Modeling 的进阶方法，通过**随机森林的集成学习框架**直接估计异质性处理效应（CATE）。与传统 Uplift Modeling 的元学习方法（T/X-Learner）不同，因果森林将"处理效应异质性发现"本身作为优化目标，自动识别哪些用户特征维度对处理效应影响最大。

### 数学直觉

**条件平均处理效应（CATE）**:
$$\tau(x) = E[Y(1) - Y(0) | X = x]$$

**因果森林的核心创新**:
1. **双重分裂准则**：决策树节点分裂时，不仅考虑结果预测误差，还考虑**处理效应异质性最大化**
   - 分裂目标：$\max \hat{\Delta}(A) = \hat{\tau}_L^2 \cdot n_L + \hat{\tau}_R^2 \cdot n_R$
   - 其中 $\hat{\tau}_L, \hat{\tau}_R$ 是左右子树的平均处理效应

2. **诚实估计（Honest Estimation）**：将样本分为"结构构建"和"效应估计"两部分，避免过拟合
   - 结构样本：决定树的结构和分裂点
   - 估计样本：计算叶子节点的处理效应

3. **局部线性调整**：在叶子节点内进行线性回归调整，提高估计精度
   $$\hat{\tau}(x) = \hat{\tau}_{leaf} + \hat{\beta}^T (x - \bar{x}_{leaf})$$

### 关键假设
- **SUTVA**：稳定单元处理值假设
- **条件独立性**：给定特征 $X$，处理分配 $T$ 与潜在结果独立
- **重叠假设**：对所有 $x$，$0 < P(T=1|X=x) < 1$
- **一致性**：处理效应在不同子群体中是一致的

### 与 Uplift Modeling 的对比

| 维度 | Uplift Modeling (X-Learner) | Causal Forest |
|------|---------------------------|---------------|
| 核心方法 | 元学习框架（两阶段） | 集成学习（端到端） |
| 异质性发现 | 依赖人工特征工程 | 自动发现最优分裂维度 |
| 可解释性 | 中等（模型输出） | 高（分裂路径可解释） |
| 处理高维数据 | 需降维 | 天然支持 |
| 变量重要性 | 难以量化 | 可直接计算 |

---

## ② 母婴出海应用案例

### 场景一：多市场智能广告归因

**业务问题**：
我们在美国、加拿大、英国、德国同步投放吸奶器广告，不同市场的用户行为差异显著。美国妈妈注重性价比，德国妈妈注重品质认证，英国妈妈注重环保可持续。传统的统一归因模型无法捕捉这些市场差异，需要一种能**自动发现市场-用户特征交互效应**的方法，实现"千人千面"的投放归因。

**数据要求**：
- 用户特征：年龄、收入水平、是否新手妈妈、浏览行为、加购金额
- 市场特征：国家/地区、语言、时区
- 干预数据：Facebook/Instagram/TikTok 广告曝光记录
- 标签：是否购买、购买金额、购买时间
- 数据量：建议每个市场至少 5,000 样本

**预期产出**：
- **自动发现的用户分群**：因果森林自动识别高 uplift 用户特征组合（如"德国高收入新手妈妈"）
- **变量重要性排序**：哪些特征对广告效果影响最大（如国家 > 收入 > 是否新手妈妈）
- **细分策略**：
  - 美国：针对价格敏感型用户，突出性价比卖点
  - 德国：针对品质关注型用户，强调医疗认证
  - 英国：针对环保意识型用户，强调可持续材料

**业务价值**：
- 吸奶器广告预算月均 50 万，优化后预计：
  - 广告预算节省 25-35%（节省 12-17 万）
  - 跨市场转化率差异缩小（从 40% → 15%）
  - 找到 3-5 个高 uplift 细分人群，CTR 提升 30-50%

---

### 场景二：促销活动时段智能选择

**业务问题**：
我们通过邮件营销给北美用户发送吸奶器促销信息，但促销时机选择困难：新手妈妈可能在孕期就开始关注，而二胎妈妈可能在产后才购买。传统方法难以识别"促销时机敏感性"，我们需要找到**最佳促销触达时机**的用户特征。

**数据要求**：
- 用户特征：孕期阶段（备孕/孕中/产后）、购买历史、浏览时段偏好
- 干预数据：邮件发送时间（早上/中午/晚上）、促销力度
- 标签：是否打开邮件、是否点击、是否购买
- 数据量：建议 10,000+ 样本

**预期产出**：
- **时机敏感性预测**：每个用户的最佳触达时机
- **因果森林路径解释**：为什么某用户适合晚上发送（如"职场妈妈 + 晚间浏览习惯"）
- **自动发现规则**：
  - 孕早期用户：周末早晨发送效果最好
  - 产后 0-3 月：工作日午休时间发送效果最佳

**业务价值**：
- 邮件打开率从 15% 提升至 25-30%
- 转化率提升 20-35%
- 减少无效发送，降低邮件 fatigue

---

## ③ 代码模板

```python
"""
智能归因 - 因果森林 (Causal Forest)
用于母婴出海电商多市场智能广告归因和促销时机优化
基于 grf (Generalized Random Forests) 实现
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 注意：需要安装 grf 包
# pip install grf
# 或 R 的 grf 包通过 rpy2 调用


class CausalForestAttribution:
    """
    因果森林智能归因模型
    自动发现异质性处理效应，支持多市场场景
    """

    def __init__(self, n_trees=1000, min_node_size=5, max_depth=None):
        """
        初始化因果森林

        Args:
            n_trees: 树的数量（默认 1000）
            min_node_size: 叶子节点最小样本数（默认 5）
            max_depth: 最大深度（默认无限制）
        """
        self.n_trees = n_trees
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X, treatment, outcome, **kwargs):
        """
        训练因果森林

        Args:
            X: 特征矩阵 (DataFrame 或 np.array)
            treatment: 干预标志 (1=干预组, 0=对照组)
            outcome: 结果变量 (0/1 或连续值)
            **kwargs: 传递给底层实现的额外参数
        """
        try:
            from grf import CausalForest
            self.model = CausalForest(
                n_estimators=self.n_trees,
                min_samples_leaf=self.min_node_size,
                max_depth=self.max_depth,
                **kwargs
            )
            self.model.fit(X, treatment, outcome)
            self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
            return self
        except ImportError:
            print("Warning: grf package not found. Using sklearn GradientBoosting as fallback.")
            return self._fit_fallback(X, treatment, outcome)

    def _fit_fallback(self, X, treatment, outcome):
        """
        回退方案：使用 GradientBoosting 近似因果森林
        """
        from sklearn.ensemble import GradientBoostingRegressor

        # 使用 T-Learner 近似
        X_t = X[treatment == 1]
        X_c = X[treatment == 0]
        y_t = outcome[treatment == 1]
        y_c = outcome[treatment == 0]

        self.model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        self.model_c = GradientBoostingRegressor(n_estimators=100, max_depth=4)

        self.model_t.fit(X_t, y_t)
        self.model_c.fit(X_c, y_c)

        return self

    def predict(self, X):
        """
        预测 CATE (条件平均处理效应)

        Args:
            X: 特征矩阵

        Returns:
            cate: 每个样本的处理效应估计
        """
        if self.model is not None and hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            # 回退方案
            pred_t = self.model_t.predict(X)
            pred_c = self.model_c.predict(X)
            return pred_t - pred_c

    def get_variable_importance(self) -> pd.DataFrame:
        """
        获取变量重要性（仅 grf 支持）

        Returns:
            DataFrame: 特征重要性排序
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame()

        importances = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names or range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False)

    def explain_leaf_path(self, X_sample) -> dict:
        """
        解释单个样本的叶子路径（简化版）

        Args:
            X_sample: 单个样本

        Returns:
            路径解释
        """
        # 获取预测值
        cate = self.predict(X_sample.reshape(1, -1) if len(X_sample.shape) == 1 else X_sample)[0]

        return {
            'predicted_cate': cate,
            'interpretation': f"该用户的预测处理效应为 {cate:.4f}"
        }


# ==================== 母婴出海业务专用函数 ====================

def generate_multimarket_data(n_samples=10000, random_state=42):
    """
    生成多市场母婴电商模拟数据

    场景：吸奶器在北美/欧洲多市场投放
    """
    np.random.seed(random_state)

    # 市场分布
    market = np.random.choice(
        ['US', 'CA', 'UK', 'DE'],
        n_samples,
        p=[0.45, 0.20, 0.20, 0.15]
    )

    # 用户基础特征（不同市场略有差异）
    age = np.where(
        market == 'DE',
        np.random.normal(34, 4, n_samples),  # 德国妈妈年龄稍大
        np.random.normal(31, 5, n_samples)
    )
    age = np.clip(age, 25, 45)

    # 收入水平（市场差异明显）
    income_level = np.zeros(n_samples, dtype=int)
    for i, m in enumerate(['US', 'CA', 'UK', 'DE']):
        mask = market == m
        if m == 'US':
            income_level[mask] = np.random.choice([1,2,3,4], mask.sum(), p=[0.25, 0.35, 0.30, 0.10])
        elif m == 'CA':
            income_level[mask] = np.random.choice([1,2,3,4], mask.sum(), p=[0.20, 0.30, 0.35, 0.15])
        elif m == 'UK':
            income_level[mask] = np.random.choice([1,2,3,4], mask.sum(), p=[0.25, 0.35, 0.30, 0.10])
        else:  # DE
            income_level[mask] = np.random.choice([1,2,3,4], mask.sum(), p=[0.15, 0.25, 0.40, 0.20])

    # 是否新手妈妈
    is_first_time = np.random.binomial(1, 0.6, n_samples)

    # 浏览行为
    browsing_pages = np.random.poisson(5, n_samples) + 1
    time_on_site = np.random.exponential(5, n_samples)  # 分钟

    # 加购金额
    cart_value = np.random.exponential(80, n_samples)
    has_cart = np.random.binomial(1, 0.35, n_samples)
    cart_value = cart_value * has_cart

    X = pd.DataFrame({
        'age': age,
        'income_level': income_level,
        'is_first_time': is_first_time,
        'browsing_pages': browsing_pages,
        'time_on_site': time_on_site,
        'cart_value': cart_value,
        'market_US': (market == 'US').astype(int),
        'market_CA': (market == 'CA').astype(int),
        'market_UK': (market == 'UK').astype(int),
        'market_DE': (market == 'DE').astype(int)
    })

    # 生成干预（广告投放）
    # 不同市场的广告策略不同
    treatment_prob = np.where(market == 'US', 0.5,
                     np.where(market == 'DE', 0.4, 0.45))
    treatment = np.random.binomial(1, treatment_prob)

    # 生成潜在结果（市场异质性）
    base_prob = 0.05

    # 市场效应
    market_effect = np.where(market == 'US', 0.02,
                    np.where(market == 'DE', 0.03, 0.01))

    # 用户特征效应
    income_effect = 0.02 * (income_level - 2.5)
    first_time_effect = 0.04 * is_first_time
    cart_effect = 0.08 * (cart_value > 0)

    # 基础购买概率
    Y0_prob = base_prob + market_effect + income_effect + first_time_effect + cart_effect
    Y0_prob = np.clip(Y0_prob, 0.01, 0.4)
    Y0 = np.random.binomial(1, Y0_prob)

    # 广告效应（市场异质性）
    ad_effect = np.where(
        market == 'US',
        0.12 * (income_level == 2) + 0.08 * is_first_time,  # 美国：中等收入+新手敏感
        np.where(
            market == 'DE',
            0.10 * (income_level >= 3) + 0.06 * (time_on_site > 5),  # 德国：高收入+深度浏览
            0.09 * is_first_time + 0.05 * (cart_value > 50)  # 其他：新手+高加购
        )
    )

    Y1_prob = Y0_prob + ad_effect
    Y1_prob = np.clip(Y1_prob, 0.01, 0.6)
    Y1 = np.random.binomial(1, Y1_prob)

    outcome = np.where(treatment == 1, Y1, Y0)

    return X, treatment, outcome, market


def analyze_market_segments(model, X_test, market_test, treatment_test, outcome_test):
    """
    分析各市场的用户分群效果
    """
    cate_pred = model.predict(X_test)

    results = []
    for market in ['US', 'CA', 'UK', 'DE']:
        mask = market_test == market
        if mask.sum() == 0:
            continue

        market_cate = cate_pred[mask]

        # 计算市场内的高/中/低 uplift 用户占比
        q75, q25 = np.percentile(market_cate, [75, 25])

        results.append({
            'market': market,
            'avg_cate': market_cate.mean(),
            'high_uplift_pct': (market_cate > q75).mean() * 100,
            'low_uplift_pct': (market_cate < q25).mean() * 100,
            'cate_std': market_cate.std()
        })

    return pd.DataFrame(results)


def main():
    """主函数：演示因果森林在多市场归因中的应用"""
    print("=" * 70)
    print("母婴出海 - 因果森林智能归因")
    print("=" * 70)

    # 1. 生成多市场模拟数据
    print("\n[1] 生成多市场数据...")
    X, treatment, outcome, market = generate_multimarket_data(n_samples=10000)
    print(f"   总样本: {len(X)}")
    for m in ['US', 'CA', 'UK', 'DE']:
        count = (market == m).sum()
        print(f"   - {m}: {count} ({count/len(market)*100:.1f}%)")

    # 2. 划分训练/测试集
    print("\n[2] 划分训练/测试集...")
    indices = np.arange(len(X))
    train_idx, test_idx, _, _ = train_test_split(
        indices, outcome, test_size=0.3, random_state=42
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    treatment_train, treatment_test = treatment[train_idx], treatment[test_idx]
    outcome_train, outcome_test = outcome[train_idx], outcome[test_idx]
    market_test = market[test_idx]

    print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 3. 训练因果森林（回退方案）
    print("\n[3] 训练因果森林模型...")
    print("   (使用 GradientBoosting 近似因果森林)")
    model = CausalForestAttribution()
    model.fit(X_train.values, treatment_train, outcome_train)

    # 4. 预测 CATE
    print("\n[4] 预测处理效应...")
    cate_pred = model.predict(X_test.values)
    print(f"   平均 CATE: {cate_pred.mean():.4f}")
    print(f"   CATE 标准差: {cate_pred.std():.4f}")
    print(f"   CATE 范围: [{cate_pred.min():.4f}, {cate_pred.max():.4f}]")

    # 5. 变量重要性（简化版）
    print("\n[5] 变量重要性（基于特征相关性近似）...")
    importance = []
    for i, col in enumerate(X.columns):
        corr = np.corrcoef(X_test.iloc[:, i], cate_pred)[0, 1]
        importance.append((col, abs(corr)))
    importance.sort(key=lambda x: x[1], reverse=True)
    for col, imp in importance[:5]:
        print(f"   {col}: {imp:.4f}")

    # 6. 分市场分析
    print("\n[6] 分市场处理效应分析...")
    segment_analysis = analyze_market_segments(
        model, X_test.values, market_test, treatment_test, outcome_test
    )
    print(segment_analysis.to_string(index=False))

    # 7. 高 uplift 用户画像
    print("\n[7] 高 uplift 用户画像 (Top 20%)...")
    high_uplift_mask = cate_pred > np.percentile(cate_pred, 80)
    high_users = X_test[high_uplift_mask]
    print(f"   平均年龄: {high_users['age'].mean():.1f}")
    print(f"   平均收入: {high_users['income_level'].mean():.1f}")
    print(f"   新手妈妈占比: {high_users['is_first_time'].mean()*100:.1f}%")
    print(f"   美国市场占比: {high_users['market_US'].mean()*100:.1f}%")
    print(f"   德国市场占比: {high_users['market_DE'].mean()*100:.1f}%")

    # 8. 投放建议
    print("\n[8] 智能投放建议...")
    print("   - 美国市场：重点投放中等收入新手妈妈")
    print("   - 德国市场：重点投放高收入+深度浏览用户")
    print("   - 加拿大/英国：标准投放策略，关注新手妈妈")
    print("   - 高 uplift 用户优先分配预算")

    print("\n" + "=" * 70)
    print("智能归因分析完成！")
    print("=" * 70)

    return model


if __name__ == '__main__':
    model = main()
```

---

## ④ 技能关联

### 前置技能
- **Uplift Modeling (元学习框架)**：理解处理效应估计的基本概念
- **随机森林**：理解决策树和集成学习原理
- **因果推断基础**：理解条件独立性、重叠假设

### 延伸技能
- **多研究因果森林 (MCF)**：处理多数据源/多市场的异质性
- **变量重要性分析**：深入解释因果森林的决策路径
- **双重机器学习 (DML)**：结合因果森林与 Neyman 正交化

### 可组合技能
- **智能预测 (Doubly Robust)**：组合使用提高预测鲁棒性
- **LTV预测**：优先对高 LTV 潜力用户进行归因分析
- **多臂老虎机**：在线实验阶段动态调整投放策略

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 多市场广告归因 | 预算节省 25-35%（月预算 50 万，节省 12-17 万） | 开发 2-3 周 | 8-12x |
| 促销时机优化 | 转化率提升 20-35% | 开发 1-2 周 | 5-8x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要 A/B 测试数据，每个细分市场样本充足
- 技术门槛：中高，需理解随机森林和因果推断结合
- 工程复杂度：中，grf 包封装良好
- 维护成本：中，需定期重新训练模型

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- 业务价值极高：多市场投放是母婴出海的核心场景
- 可解释性强：变量重要性直接指导运营策略
- 技术前沿：2024-2025 年热点方法
- 可落地性强：有成熟开源实现（grf）

### 与 Uplift Modeling 的关系
因果森林是 Uplift Modeling 的**进阶版**，建议：
1. 先用 Uplift Modeling 建立基础归因能力
2. 当需要更精细的异质性发现时，升级到因果森林
3. 两个模型可以并行运行，相互验证

---

## 参考论文

- **Multi-Study Causal Forest (MCF)**: [arXiv:2502.02110](https://arxiv.org/abs/2502.02110) (2025)
- **Variable importance for causal forests**: Journal of Causal Inference (2025)
- **Generalized Random Forests**: Athey et al., Annals of Statistics (2019)
