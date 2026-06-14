---
title: Feature Engineering for E-Commerce Machine Learning
module: 12-ML基础
topic: feature-engineering

roadmap_phase: phase1
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: Feature Engineering

## ① 算法原理

**核心问题**：模型效果的上限由特征质量决定。同样的算法，好的特征 vs 差的特征，效果可能差3-5倍。特征工程是"把领域知识注入模型的艺术"。

**母婴电商的关键特征类别**：

| 类别 | 示例 | 用途 |
|------|------|------|
| **用户行为** | 浏览次数、加购次数、购买频次、浏览深度 | Churn/LTV/Uplift |
| **用户属性** | 注册时长、来源渠道、设备类型、国家 | 分群/冷启动 |
| **商品属性** | 品类、品牌、价格段、适用月龄 | 推荐/定价 |
| **时序特征** | RFM、生命周期阶段、距离上次购买天数 | 复购预测 |
| **交叉特征** | 用户价格敏感度 × 商品折扣力度 | Uplift/促销 |
| **文本特征** | 评论情感、搜索关键词 | VOC/NLP |

**核心操作**：

**1. 数值特征处理**
- **标准化**（Z-Score）：$(x - \mu) / \sigma$ — 适用于需要度量距离的模型（KNN、SVM、神经网络）
- **归一化**（Min-Max）：$(x - x_{min}) / (x_{max} - x_{min})$ — 适用于有界特征
- **对数变换**：$\log(x + 1)$ — 处理右偏分布（如消费金额、浏览次数）
- **分箱**（Binning）：将连续值离散化为区间 — 处理非线性关系、降低异常值影响

**2. 类别特征编码**
- **One-Hot**：高基数时维度爆炸（1000个SKU → 1000维）
- **Target Encoding**：用目标变量的均值编码，适用于高基数类别
- **Embedding**：用神经网络学习低维稠密表示，推荐系统标配
- **频率编码**：用出现频率编码，简单有效

**3. 时序特征工程**
- **滞后特征**：$x_{t-1}, x_{t-7}, x_{t-30}$ — 捕捉历史依赖
- **滚动统计**：7天均值、30天标准差 — 捕捉趋势和波动
- **差分**：$x_t - x_{t-1}$ — 捕捉变化率
- **时间特征**：星期几、是否周末、是否节假日 — 捕捉周期性

**4. 特征选择**
- **过滤法**：按相关性、互信息筛选
- **包裹法**：用模型效果评估特征子集（递归特征消除RFE）
- **嵌入法**：L1正则化自动稀疏化（Lasso）

**反直觉洞察**：
- 80%的特征贡献不到1%的效果提升——特征选择比特征构造更重要
- 交叉特征（A×B）的效果往往优于原始特征之和，但维度过高时需要控制
- "数据泄露"是特征工程中最危险的错误——用未来信息预测过去

---

## ② 母婴出海应用案例

### 场景：用户流失预测的特征工程

**业务问题**：预测哪些用户会在30天内流失，用于精准召回。

**原始数据**：用户ID、注册日期、订单记录、浏览记录

**特征构造**：

| 特征名 | 计算方式 | 业务含义 |
|--------|---------|---------|
| recency_days | 今天 - 最后购买日 | 多久没来了 |
| frequency_90d | 近90天订单数 | 购买活跃度 |
| monetary_total | 累计消费金额 | 用户价值 |
| avg_order_value | 总消费 / 订单数 | 客单价偏好 |
| browse_to_buy_ratio | 浏览次数 / 购买次数 | 购买决策效率 |
| category_diversity | 购买过的品类数 | 兴趣广度 |
| days_since_register | 注册天数 | 用户成熟度 |
| peak_hour_preference | 是否偏好夜间下单 | 用户画像 |
| country | 注册国家 | 地域差异 |
| device_type | 手机/电脑 | 渠道偏好 |

**特征处理**：
- 数值特征：对数变换（frequency, monetary）、标准化（全部）
- 类别特征：Target Encoding（country, device_type）
- 时序特征：滞后（last_month_orders）
- 交叉特征：recency × monetary（高价值+高流失风险 = 优先召回）

**效果对比**：
- 原始特征（仅RFM）：AUC = 0.72
- 工程后特征（15维）：AUC = 0.84

---

## ③ 代码模板

```python
"""
Feature Engineering — 特征工程工具箱
支持：数值处理、类别编码、时序特征、特征选择
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest


class FeatureEngineer:
    """特征工程器"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.target_means = {}

    def log_transform(self, df, cols, offset=1):
        """对数变换"""
        df = df.copy()
        for c in cols:
            df[f'{c}_log'] = np.log1p(df[c] + offset)
        return df

    def binning(self, df, col, n_bins=5, method='quantile'):
        """分箱"""
        df = df.copy()
        if method == 'quantile':
            df[f'{col}_bin'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        else:
            df[f'{col}_bin'] = pd.cut(df[col], bins=n_bins, labels=False)
        return df

    def target_encoding(self, df, cat_col, target_col, smoothing=10):
        """目标编码（带平滑）"""
        df = df.copy()
        global_mean = df[target_col].mean()

        stats = df.groupby(cat_col)[target_col].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)

        df[f'{cat_col}_te'] = df[cat_col].map(smoothed)
        self.target_means[cat_col] = smoothed.to_dict()
        return df

    def time_features(self, df, datetime_col):
        """时间特征提取"""
        df = df.copy()
        dt = pd.to_datetime(df[datetime_col])
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = dt.dt.is_month_start.astype(int)
        return df

    def rolling_features(self, df, group_col, value_col, windows=[7, 14, 30]):
        """滚动统计特征"""
        df = df.copy()
        df = df.sort_values(group_col)
        for w in windows:
            df[f'{value_col}_ma_{w}'] = df.groupby(group_col)[value_col].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f'{value_col}_std_{w}'] = df.groupby(group_col)[value_col].transform(lambda x: x.rolling(w, min_periods=1).std())
        return df

    def lag_features(self, df, group_col, value_col, lags=[1, 7, 30]):
        """滞后特征"""
        df = df.copy()
        df = df.sort_values([group_col, 'date'] if 'date' in df.columns else group_col)
        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df.groupby(group_col)[value_col].shift(lag)
        return df

    def select_features(self, X, y, k=10):
        """互信息特征选择"""
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(X, y)
        mask = selector.get_support()
        selected_cols = [c for c, m in zip(X.columns, mask) if m]
        return pd.DataFrame(X_new, columns=selected_cols), selected_cols

    def scale(self, df, cols, method='standard'):
        """标准化/归一化"""
        df = df.copy()
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
        self.scalers[method] = scaler
        return df


# 母婴电商用户特征工程示例
def build_user_features(orders_df, browsing_df, reference_date='2026-05-15'):
    """
    从订单和浏览数据构建用户特征

    Args:
        orders_df: DataFrame with [user_id, order_date, amount, category]
        browsing_df: DataFrame with [user_id, browse_date, page_type]
    """
    ref_date = pd.to_datetime(reference_date)

    # RFM
    rfm = orders_df.groupby('user_id').agg({
        'order_date': lambda x: (ref_date - pd.to_datetime(x.max())).days,
        'order_date': 'count',
        'amount': 'sum'
    }).rename(columns={'order_date': 'frequency', 'amount': 'monetary'})

    # 修正：重新计算recency
    rfm = orders_df.groupby('user_id').agg({
        'order_date': lambda x: (ref_date - pd.to_datetime(x.max())).days,
        'amount': ['count', 'sum']
    })
    rfm.columns = ['recency_days', 'frequency', 'monetary']

    # 品类多样性
    category_div = orders_df.groupby('user_id')['category'].nunique().rename('category_diversity')

    # 浏览行为
    browse_counts = browsing_df.groupby('user_id').size().rename('browse_count')

    # 合并
    features = pd.concat([rfm, category_div, browse_counts], axis=1).fillna(0)
    features['avg_order_value'] = features['monetary'] / features['frequency'].clip(lower=1)
    features['browse_to_buy'] = features['browse_count'] / features['frequency'].clip(lower=1)

    return features.reset_index()


if __name__ == '__main__':
    # 模拟数据
    np.random.seed(42)
    n_users = 1000

    orders = pd.DataFrame({
        'user_id': np.random.choice(range(n_users), 5000),
        'order_date': pd.date_range('2025-01-01', periods=5000, freq='h'),
        'amount': np.random.lognormal(4, 0.5, 5000),
        'category': np.random.choice(['奶粉', '纸尿裤', '辅食', '玩具'], 5000)
    })

    browsing = pd.DataFrame({
        'user_id': np.random.choice(range(n_users), 10000),
        'browse_date': pd.date_range('2025-01-01', periods=10000, freq='30min'),
        'page_type': np.random.choice(['详情页', '列表页', '购物车'], 10000)
    })

    features = build_user_features(orders, browsing)
    print(f"特征维度: {features.shape}")
    print(features.head())
```

---


## ④ 技能关联

### 前置技能
- 无（本 Skill 是基础入口卡）

### 延伸技能
- [Skill-Causal-Discovery-PC-Algorithm](../01-因果推断/[[Skill-Causal-Discovery-PC-Algorithm]].md) — 特征工程后用因果发现做变量筛选
- [Skill-Uplift-Modeling](../01-因果推断/[[Skill-Uplift-Modeling]].md) — 特征工程是 Uplift 建模的核心环节

### 可组合
- [Skill-Matrix-Factorization](../05-推荐系统/[[Skill-Matrix-Factorization]].md) — 推荐系统的隐因子也是特征工程的延伸
- [Skill-Customer-Churn-Prediction](../06-增长模型/[[Skill-Customer-Churn-Prediction]].md) — 流失模型严重依赖特征工程


- **可组合（延伸）**：[[Skill-Multilingual-NER-Universal-v2]] / [[Skill-Listing-Quality-Scoring]] / [[Skill-Deep-Learning-Churn-Prediction]] / [[Skill-RFM-Customer-Segmentation]]

## ⑤ 商业价值评估

- **ROI**：特征质量提升 → 模型效果提升30-50%，直接转化为业务收益
- **难度**：⭐⭐⭐☆☆（3/5）— 需要领域知识，不是纯技术问题
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 所有ML技能的前置基础，没有它就没有模型效果
