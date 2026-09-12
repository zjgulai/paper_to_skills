---
title: Feature Engineering for E-Commerce Machine Learning
module: 12-ML基础
topic: feature-engineering
status: stable
created: 2026-05-15
updated: 2026-09-12
# --- v2 溯源字段：仅覆盖本次 v2.1 增补节（①1b / ①b / ②场景2 / ③-2 / ⑤增补 / ⑥）---
# 原 v1 正文（①表格、②场景1、③主代码、⑤原 ROI）未溯源到任何论文，见 ② 的「增补说明」
paper_id: 2608.09162
paper: Tabular Numeric Stretch Transformation
venue: arXiv preprint
venue_tier: preprint
evidence_grade: A
verified_by: verify_skill_code.py（K1）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py + 人工抽检
supersedes:
related: Skill-Customer-Churn-Prediction.md, Skill-RFM-Customer-Segmentation.md, Skill-Uplift-Modeling.md
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

**1b.（v2.1 增补）把「数值特征变换」本身写成优化问题 —— Stretch Transformation**

上面四种处理（标准化 / 归一化 / log / 分箱）**都只看特征自己的分布，不看标签**。论文
《Tabular Numeric Stretch Transformation》(arXiv 2608.09162) 把这一步形式化为优化问题：找一个**保序**的
分段线性映射 s，把 x 打到 [0,1]，使**目标函数在变换后的空间里更平滑**——神经网络有频谱偏置（先学低频、
难学高频），而表格特征的边际信号天然凹凸不平，正好制造高频成分。

- **参数化**：把 $[x_{min}, x_{max}]$ 分成 T 个（分位数）箱，给第 t 箱分配宽度 $w_t$（$\sum_t w_t = 1$），
  箱内线性插值。于是问题只剩一句：**每一段该分多少宽度？** 宽度大 = 拉伸（该区域分辨率高），宽度小 = 压缩。
- **unsupervised stretch**（没有标签、或标签不可信时）：用 **minimax** —— 在所有「局部变化有界」的目标函数里
  最小化最坏情况的 Dirichlet 能量，闭式解是 **$w_t = 1/T$（均匀分配）**。两个理论联系：T 增大时它收敛到
  **经验 CDF 变换**；它与 **PLE**（Piecewise Linear Encoding）共享同一条分段线性几何——PLE 编码在 $\mathbb{R}^T$
  中的路径弧长恰好是 T × stretch(x)，两者只是同一流形的两种坐标写法，但 stretch **不扩维**（每特征内存 O(1)
  vs PLE 的 O(T)）。
- **supervised stretch**（有标签时）：令 $S_t$ = 第 t 箱内目标函数的**总变差**（$\sum |\Delta f_i|$），
  则 **$w_t^{\star} \propto S_t$**：目标变化快的区域给更多空间、平缓处压窄，效果是**把各箱斜率拉平**，
  于是变换后的目标函数更平滑。细箱极限（T = n）下它与 **target encoding** 接近（都是「用目标变化量决定变换」），
  而箱数 T 成了天然的正则化旋钮。
- **落地要点**：目标函数 f 必须用 **out-of-fold** 的核回归估计 $\hat{f}$（样本自己的标签不参与自己的 $\hat{f}$），
  否则标签会通过变换本身泄回训练集。目标变化可忽略、或特征取值过少时，回退到 identity / 均匀宽度。

**与上面四条的定位差异**（不是替代关系）：log / box-cox / quantile 是**无监督**的分布整形，改的是**特征**；
stretch 是**有监督**的平滑优化，改的是**特征与标签的联合形状**。论文的实验里，对「只调分布形状」的变换
（含两个 stretch 变体）统一在其后再接一步标准化。本卡 ③ 段给出 numpy 实现与 7 条断言，⑥ 段给出逐字出处。

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

## ①b 反例与适用边界（v2.1 增补 · 只增不改）

> 本段为 v2.1 新增，原正文一个字未删。段内每个带数字的论断都在 ⑥ 段有逐字出处。

**什么时候不要用这个算法**

1. **树模型不适用**。论文明确写着树模型对单调变换不变，而神经网络才需要预处理来稳定优化。stretch 是
   **保序**映射，对 GBDT / Random Forest 几乎不产生差异——把它加到 LightGBM 上属于白做。它是给
   **MLP / FT-Transformer / ResNet 这类神经网络**用的。
2. **它只作用于数值特征**。分类特征占主导的表收益有限；论文自己写的是「均衡组」结果**混杂、无单一方法占优**。
3. **别指望它解决特征交互**。当前变换是 **marginal**（逐特征独立）的，论文自承不优化联合/条件平滑，
   交互仍由下游模型自己学。

**已知的失败模式**

1. **「一致优于所有基线」是聚合口径，不是一个单元都不输**（最容易被误读，此处已回原文逐项核过）：
   - **基线清单（7 个）**：standardization（z-score）、Yeo-Johnson、quantile→Gaussian、min-max
     缩放、RobustScale+SmoothClip（RealMLP 的原生数值预处理）、PLE、PLE-T（文献里唯一的有监督数值变换，
     本文的主要有监督基线）。
   - **评测规模**：TALENT Tiny Benchmark 1 的 **38 个数据集**（26 分类 + 12 回归；原始 42 个里剔掉 4 个
     纯类别数据集）× **5 个神经网络架构**（FT-Transformer、MLP、MLP-PLR、RealMLP、ResNet）= **190 个
     dataset-model 单元**；每个配置先做 **100 次** Optuna 贝叶斯优化，再用 **15 个随机种子**在固定切分上评估
     （论文自己说明这 15 个种子只衡量初始化/批次顺序/优化波动，**不衡量换数据切分的方差**）。
   - **「一致优于」= 聚合指标最强**：逐 (dataset, model) min-max 归一化后的 Normalized Score，加上配对胜率
     （胜 = 差距 > 两者种子标准差中较大的那个；差距不超过该单元**中位种子标准差**的判平局、记 0.5）。
     回归侧的原始胜负记录最有说服力：**vs PLE 18-7、vs PLE-T 28-9、vs 标准化 14-4**。
   - **但它不是每个单元都赢**：论文 Table 1 的 `mlp` 一行 Overall Score 是 PLE（0.6835）高于 supervised
     stretch（0.6343）；`mlp_plr` 一行是 Unsupervised stretch（0.7347）最高、supervised stretch（0.6238）
     明显落后。分类侧论文自己的措辞只是 "remains in the top tier"（第一梯队），**没有**说每格都第一。
     选型时请按**贵司自己的骨干架构**看对应那一行，而不是看总榜。
2. **箱数 T 是唯一的正则化旋钮，调不对就退化**：T 小 → 保留原分布结构；T 大 → 逼近 CDF / target encoding，
   过度平滑会抹掉信号（论文引用了「降高频不等于提升效果」的既有结论）。论文把 T 交给 Optuna 一起搜（具体候选集见论文 Table 2，本卡不复述）。
3. **标签噪声**：supervised stretch 用标签设计变换，天然有被带偏的风险。论文的噪声实验显示它在最高噪声级
   （η = 0.5）平均排名反而最好，但那是 **4 个数据集 × 5 个模型 = 20 个组合上的平均排名**，不是每个数据集都成立；
   标签无法清洗时，退到 unsupervised stretch 更稳。
4. **工程回退**：PLE 这类扩维方法在大表上会因张量超内存而回退成标准化（论文称只影响「不到 5% 的 190 个组合」）
   ——扩维方案上线前要先量一遍显存/内存；stretch 本身不扩维，不存在这个问题。

**论文自己承认的局限**

1. 设计还能更好：可以用更平滑的**样条**替代分段线性，以及自适应分箱策略。
2. 变换是 **marginal** 的，**不是 interaction-aware** 的。
3. 理论分析限于小 σ 的 RBF 核近似，没有推广到更一般的情形。
4. 没有评测把 stretch 当作 TabPFN 推理期预处理集成的一个选项（论文明确说这超出本次受控架构研究的范围）。
5. **论文未报告**：任何电商/母婴口径的数据与收益、金额口径的 ROI、工程成本与线上时延。本卡 ⑤ 段因此
   只给公式、不给提升幅度。

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

**⚠️ 增补说明（v2.1 · 只标注、不删除原数据）**：上面「原始特征 / 工程后特征」的两个 AUC 值、
①段的倍数量级表述、⑤段的 ROI 百分比，都是**本卡 v1 的自述经验值，没有论文出处**——本次增补的论文
（arXiv 2608.09162）**未报告**任何电商场景指标。按 MasterPrompt-v2 的 R1/R2 规则，它们**不能当证据引用**，
请以贵司自己的 holdout 复测为准；本次新增的 ⑥ 段只托住 2608.09162 的数字，**不为这几条背书**。

---

### 场景 2（v2.1 增补）：把「手调 log / box-cox」换成一次可复核的宽度分配

- **业务问题**：复购 / 流失 / 加购预测里最贵的是重尾数值列（累计消费金额、客单价、加购转化率、浏览深度、
  距上次购买天数）。现在的做法是逐个试 log1p / sqrt / box-cox / 分位数分箱，靠离线指标挑；一轮迭代要重跑整套
  训练，而且**换品类最优变换就变**（奶粉、纸尿裤、童装的消费额分布完全不同）。目标动作：**下一轮模型迭代时，
  用 stretch 一次性替代这一轮手工搜索**，并把「每一段分了多少宽度」留成可复核的产物。
- **数据要求**：单表「数值特征 + 标签」（二分类或回归均可），行数 ≥ 数千（分位数分箱与 OOF 核回归都要样本量），
  数值列从几列到上百列都行；**必须按时间切分**训练 / 验证（例：2025-01~2025-09 训练、10~12 验证），且 OOF
  估计只在训练段内做——否则验证段标签会泄进变换。标签噪声大的场景优先用 unsupervised 变体。
- **数据可得性**：`企业内可得`。RFM / 行为 / 金额特征本来就在数仓里，额外只需一次 OOF 核回归（③ 段用 numpy
  实现，无额外依赖、无网络请求）。**不可得的部分**：论文用的 TALENT 基准与它的 190 单元评测协议是学术口径，
  不能直接搬来当业务基线。
- **预期产出**：（a）每个数值特征的**箱宽表 $w_t^{\star}$**，可直接落成 Python / SQL 里的分段映射函数；
  （b）「哪一段被拉伸」的解释——例如「累计消费额在中高价段的目标函数变化最快」，这本身就是一个价格带
  拐点（具体落在哪一段由数据算出来，本卡不预设），可以交给运营；（c）与现状（log / 分位数）在同一 holdout 上的对照。
- **业务价值**：收益形态是**省掉一轮手调实验 + 让变换可复核**，不是承诺提升幅度。ROI 公式与参数来源见 ⑤。

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

**③-2（v2.1 增补）Stretch 变换的可运行实现**（numpy only，无网络请求；7 条断言由 K1 的 L5 实跑）

- 对应论文：`quantile_edges` = §3.1 的分位数分箱；`widths_unsupervised` = 式 (10)；`bin_total_variation`
  = §3.4 的 $S_t$；`widths_supervised` = 式 (15)；`stretch_transform` = 式 (2)；`oof_kernel_estimate`
  = §3.4 的 out-of-fold 估计（K 折 + Nadaraya-Watson）；`dirichlet_energy` = 式 (7)；`energy_bound`
  = 式 (13)(14)；`ple_normalized_arc_length` = 式 (12)。
- ⚠️ 代码内所有数据由 `_make_demo_data` 合成，**不是论文数据**；打印出来的数值只用于验证实现正确性，
  **不可与论文的 38 数据集结果互相印证**。

```python
# ---------------------------------------------------------------------------
# 4. Stretch 变换（T3-18 / arXiv 2608.09162）：把数值特征变换写成宽度分配优化
#    unsupervised -> 每箱等宽（minimax 均匀化）；supervised -> 宽度 ∝ 箱内目标全变差
# ---------------------------------------------------------------------------
import numpy as np


def quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """分位数分箱边界 b_0..b_T（论文 §3.1 的默认分箱：每箱样本数近似相等）。"""
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.percentile(x, qs)
    edges = np.maximum.accumulate(edges)              # 保险：单调化
    edges[0], edges[-1] = float(x.min()), float(x.max())
    for i in range(1, len(edges)):                    # 重复取值 → 推开，保证 b_t > b_{t-1}
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """样本落在第几个箱（0-based）。"""
    idx = np.searchsorted(edges, np.asarray(x, dtype=float), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def widths_unsupervised(n_bins: int) -> np.ndarray:
    """无监督拉伸的闭式解：w_t* = 1/T（论文式 (10)）。"""
    return np.full(n_bins, 1.0 / n_bins)


def bin_total_variation(x: np.ndarray, f_hat: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """S_t = Σ_{i in I_t} |Δf_i|：第 t 箱内目标函数的总变差（论文 §3.4）。"""
    order = np.argsort(np.asarray(x, dtype=float), kind="stable")
    xs = np.asarray(x, dtype=float)[order]
    fs = np.asarray(f_hat, dtype=float)[order]
    df = np.abs(np.diff(fs))                          # |f(x_{i+1}) - f(x_i)|
    idx = bin_index(xs[:-1], edges)
    s = np.zeros(len(edges) - 1, dtype=float)
    np.add.at(s, idx, df)
    return s


def widths_supervised(x: np.ndarray, f_hat: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """有监督拉伸的闭式解：w_t* = S_t / Σ_u S_u（论文式 (15)）。

    边界情形：所有箱的 S_t 都为 0（目标与特征无关）→ 退化为均匀宽度，
    与论文 Appendix F 的说明一致（S_t 相近时分配回落到均匀）。
    """
    s = bin_total_variation(x, f_hat, edges)
    if s.sum() <= 1e-12:
        return widths_unsupervised(len(s))
    return s / s.sum()


def stretch_transform(x: np.ndarray, edges: np.ndarray, w: np.ndarray) -> np.ndarray:
    """式 (2)：分段线性单调映射 s(x) = c_{t-1} + (x-b_{t-1})/(b_t-b_{t-1}) * w_t，值域 [0,1]。

    它是严格递增的双射（论文 §3.1），所以顺序被保留、天然有界。
    """
    x = np.asarray(x, dtype=float)
    t = bin_index(x, edges)
    c = np.concatenate([[0.0], np.cumsum(w)])          # c_t = Σ_{i<=t} w_i
    b0, b1 = edges[t], edges[t + 1]
    frac = np.clip((x - b0) / np.maximum(b1 - b0, 1e-300), 0.0, 1.0)
    return np.clip(c[t] + frac * w[t], 0.0, 1.0)


def oof_kernel_estimate(x: np.ndarray, y: np.ndarray, n_folds: int = 5,
                        bandwidth=None) -> np.ndarray:
    """K 折 out-of-fold 的 Nadaraya-Watson 核回归估计 f̂(x_i)（论文 §3.4 / Appendix B.1）。

    关键点：第 k 折样本的估计只由**其它折**的标签算出 —— 样本自己的 y 从不参与
    自己的 f̂，因此不会把标签泄漏进变换本身。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="stable")
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % n_folds
    if bandwidth is None:
        bandwidth = 0.1 * max(float(x.max() - x.min()), 1e-12)
    est = np.empty(n, dtype=float)
    for k in range(n_folds):
        te, tr = folds == k, folds != k
        d = (x[te][:, None] - x[tr][None, :]) / max(bandwidth, 1e-12)
        wt = np.exp(-0.5 * d * d)
        s = wt.sum(axis=1)
        est[te] = np.where(s > 1e-12, (wt * y[tr]).sum(axis=1) / np.maximum(s, 1e-12),
                           y[tr].mean())
    return est


def dirichlet_energy(pos: np.ndarray, f: np.ndarray) -> float:
    """式 (7) 的离散 Dirichlet 能量 E = Σ ‖Δf_i‖² / Δy_i（pos = 变换后位置 y）。"""
    pos = np.asarray(pos, dtype=float)
    order = np.argsort(pos, kind="stable")
    dy = np.diff(pos[order])
    df = np.diff(np.asarray(f, dtype=float)[order])
    ok = dy > 1e-12
    return float(np.sum(df[ok] ** 2 / dy[ok]))


def energy_bound(s: np.ndarray, w: np.ndarray) -> float:
    """式 (13)(14) 的箱级下界 Σ_t S_t² / w_t（论文真正最小化的那个目标）。"""
    return float(np.sum(np.asarray(s, dtype=float) ** 2 / np.maximum(np.asarray(w, dtype=float), 1e-300)))


def ple_normalized_arc_length(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """PLE 编码在 R^T 中的路径弧长 L(x)/T（式 (12) 右侧的归一化弧长）。"""
    x = np.asarray(x, dtype=float)
    t = bin_index(x, edges)
    b0, b1 = edges[t], edges[t + 1]
    frac = np.clip((x - b0) / np.maximum(b1 - b0, 1e-300), 0.0, 1.0)
    return (t + frac) / (len(edges) - 1)


def _make_demo_data(n: int = 4000, seed: int = 0):
    """合成数据：目标函数在 x≈0.86 处有一个窄峰（局部剧烈变化），其余平缓。

    ⚠️ 这是**本卡合成的示例数据**，不是论文数据；论文用的是 TALENT 基准的 38 个数据集。
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    mu = lambda z: 6.0 * np.exp(-((z - 0.86) ** 2) / (2 * 0.012 ** 2)) + 0.25 * np.sin(2 * np.pi * z)
    y = mu(x) + rng.normal(0.0, 0.01, n)
    return x, y, mu


if __name__ == "__main__":
    x, y, mu = _make_demo_data()
    T = 12
    edges = quantile_edges(x, T)
    f_hat = oof_kernel_estimate(x, y, n_folds=5, bandwidth=0.05)
    w_u = widths_unsupervised(T)
    w_s = widths_supervised(x, f_hat, edges)
    y_u = stretch_transform(x, edges, w_u)
    y_s = stretch_transform(x, edges, w_s)
    s = bin_total_variation(x, f_hat, edges)
    print("== Stretch 变换（合成示例数据，非论文数据）==")
    print("箱内目标全变差 S_t (归一化):", np.round(s / s.mean(), 2))
    print("有监督宽度 w_t*:", np.round(w_s, 4), " 均匀宽度:", np.round(w_u, 4))
    print(f"下界 Σ S_t²/w_t: 均匀={energy_bound(s, w_u):.3f}  有监督={energy_bound(s, w_s):.3f}")
    print(f"离散 Dirichlet 能量: 均匀={dirichlet_energy(y_u, f_hat):.3f}  有监督={dirichlet_energy(y_s, f_hat):.3f}")
    print(f"PLE 弧长等价  max|(t+frac)/T - s_unsup(x)| = "
          f"{np.max(np.abs(ple_normalized_arc_length(x, edges) - y_u)):.2e}")


# ---------------------------------------------------------------------------
# 5. 测试（可执行断言）
# ---------------------------------------------------------------------------
def test_unsupervised_widths_are_uniform_and_normalised():
    w = widths_unsupervised(16)
    assert w.shape == (16,)
    assert np.allclose(w, 1.0 / 16)
    assert abs(w.sum() - 1.0) < 1e-12


def test_transform_is_monotone_and_maps_onto_unit_interval():
    x, _, _ = _make_demo_data(n=1500, seed=1)
    edges = quantile_edges(x, 10)
    w = widths_supervised(x, oof_kernel_estimate(x, x * 2.0, n_folds=5), edges)
    grid = np.linspace(x.min(), x.max(), 3000)
    s = stretch_transform(grid, edges, w)
    assert np.all(np.diff(s) >= -1e-12)          # 单调不减 → 顺序保留
    assert s.min() >= -1e-12 and s.max() <= 1.0 + 1e-12   # 有界输出
    assert abs(s[0] - 0.0) < 1e-9 and abs(s[-1] - 1.0) < 1e-9


def test_supervised_widths_follow_bin_total_variation():
    x, y, _ = _make_demo_data()
    edges = quantile_edges(x, 12)
    f_hat = oof_kernel_estimate(x, y, n_folds=5, bandwidth=0.05)
    s = bin_total_variation(x, f_hat, edges)
    w = widths_supervised(x, f_hat, edges)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w, s / s.sum())           # 式 (15)：w_t* ∝ S_t
    assert w.argmax() == s.argmax()              # 变化最快的箱拿到最大宽度
    assert w.max() > 5.0 * w.min()               # 非均匀分配（而非退化成均匀）


def test_supervised_stretch_reduces_the_objective_and_the_energy():
    x, y, _ = _make_demo_data()
    edges = quantile_edges(x, 12)
    f_hat = oof_kernel_estimate(x, y, n_folds=5, bandwidth=0.05)
    w_u, w_s = widths_unsupervised(12), widths_supervised(x, f_hat, edges)
    s = bin_total_variation(x, f_hat, edges)
    # 式 (14) 的箱级下界：w ∝ S 是它的最小点（Cauchy-Schwarz 取等）
    assert energy_bound(s, w_s) < energy_bound(s, w_u)
    # 真实离散能量（式 (7)）同样下降
    e_u = dirichlet_energy(stretch_transform(x, edges, w_u), f_hat)
    e_s = dirichlet_energy(stretch_transform(x, edges, w_s), f_hat)
    assert e_s < e_u


def test_ple_arc_length_equals_unsupervised_stretch():
    """式 (12)：L(x) = T · Unsupervised-Stretch(x) —— 两种编码几何等价。"""
    x, _, _ = _make_demo_data(n=2000, seed=3)
    edges = quantile_edges(x, 8)
    w_u = widths_unsupervised(8)
    assert np.allclose(stretch_transform(x, edges, w_u),
                       ple_normalized_arc_length(x, edges), atol=1e-12)


def test_oof_estimate_never_uses_the_sample_own_target():
    """泄漏检查：改掉第 k 折的标签，第 k 折自己的 f̂ 必须**完全不变**。"""
    x, y, _ = _make_demo_data(n=1200, seed=2)
    est1 = oof_kernel_estimate(x, y, n_folds=4)
    folds = np.empty(len(x), dtype=int)
    order = np.argsort(x, kind="stable")
    folds[order] = np.arange(len(x)) % 4
    y2 = y.copy()
    y2[folds == 0] += 100.0
    est2 = oof_kernel_estimate(x, y2, n_folds=4)
    assert np.allclose(est1[folds == 0], est2[folds == 0])       # 自己折不变 → 无泄漏
    assert not np.allclose(est1[folds == 1], est2[folds == 1])   # 其它折受影响


def test_flat_target_falls_back_to_uniform_widths():
    """特征与目标无关时 S_t 相近/为 0 → 分配回落到均匀（论文 Appendix F 的说明）。"""
    x = np.linspace(0.0, 1.0, 500)
    edges = quantile_edges(x, 5)
    assert np.allclose(widths_supervised(x, np.zeros_like(x), edges), 1.0 / 5)
```

---

## ④ 技能关联

- **前置**：数据清洗（缺失值处理、异常值检测）
- **延伸**：AutoML特征工程（自动特征构造）
- **可组合**：+ 所有ML技能 — 特征工程是所有预测模型的前置步骤

- **（v2.1 增补）可组合｜`Skill-Customer-Churn-Prediction.md`（06-增长模型）**：本卡 ② 的两个场景都是
  流失/复购模型的输入层。数据流：该卡负责标签定义与模型选择，本卡负责把它的数值输入列从「手调 log」
  换成可复核的宽度分配；组合时注意标签泄漏——OOF 估计必须只在该卡的时间切分**之内**做。
- **（v2.1 增补）可组合｜`Skill-RFM-Customer-Segmentation.md`（06-增长模型）**：RFM 三个分量（R/F/M）
  都是重尾数值列，正是 stretch 的典型对象；分箱宽度还能反过来解释「哪个 RFM 区间的人最不稳定」。
- **（v2.1 增补）前置/对照｜`Skill-Uplift-Modeling.md`（01-因果推断）**：uplift 模型对特征变换更敏感
  （要保的是效应差异而不是均值），supervised stretch 的 OOF 纪律在这里更关键；但**变换只能用训练段标签**，
  否则会把处理效应本身泄进特征。

---

## ⑤ 商业价值评估

- **ROI**：特征质量提升 → 模型效果提升30-50%，直接转化为业务收益
- **难度**：⭐⭐⭐☆☆（3/5）— 需要领域知识，不是纯技术问题
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 所有ML技能的前置基础，没有它就没有模型效果

**（v2.1 增补）stretch 变换的 ROI 口径**

ROI = (ΔM − C_impl) / C_impl，其中 ΔM = N_iter × H × (1 − r_manual)。

| 参数 | 含义 | 来源 |
|---|---|---|
| `N_iter` | 每年因「重选数值特征变换」而重跑的模型迭代次数 | 企业自估（可从实验记录数出来） |
| `H` | 单轮迭代的人力成本（特征工程 + 训练 + 复核） | 企业自有工时口径 |
| `r_manual` | 自动化后仍需人工介入的比例（审箱宽表、抽检单调性、处理回退） | 企业自估；论文未报告，本卡不预设取值 |
| `C_impl` | 一次性建设成本：本卡 ③ 的代码接入训练管线 + OOF 估计的算力 | 论文**未报告**任何成本量级，需企业自估 |

**收益侧不给数字**：论文报告的是 TALENT 基准上的归一化分数与胜负计数（学术口径、无金额），
**论文未报告**任何业务口径的提升幅度，本卡也不替你假设一个。

---

## ⑥ 原文引用（v2.1 增补 · 逐字核验）

> 出处底本：`paper2skills-vault/papers/12-ML基础/p2s-2026-0026/fulltext.md`（arXiv 2608.09162v1 的 HTML 全文）。
> 下列摘录全部由脚本从底本**逐字抽取**（命令行输出即复制粘贴），并经 `quote_check.py` 判 **VERBATIM**。
> 本卡正文里出现的论文数字，都能在下面对应到原文句。

**A. 「一致优于所有基线」的原始口径（含基线清单、评测规模、胜负记录）**

> 原文："Comprehensive experiments on 38 datasets from the TALENT benchmark demonstrate that supervised stretch consistently outperforms all baselines."
> 出处：2608.09162 §Abstract

> 原文："We empirically validate the framework on 38 datasets from the TALENT benchmark, where supervised stretch consistently outperforms all baselines, with the largest gains in regression tasks."
> 出处：2608.09162 §1 Introduction（Contributions）
> 原文："While tree-based methods naturally handle diverse distributions through split-based decisions, neural networks require careful preprocessing to achieve competitive performance [35, 23]."
> 出处：2608.09162 §1 Introduction
> 原文："We compare our proposed supervised and unsupervised stretch against seven established baselines: standardization (z-score normalization), Yeo-Johnson (YJ) power transformation [36], quantile transformation to a Gaussian distribution, min-max scaling to $[0,1]$, RobustScale+SmoothClip (RS-SC) as originally used in RealMLP [17], Piecewise Linear Encoding (PLE) [10], and PLE with Tree-based binning (PLE-T)."
> 出处：2608.09162 §4.1 Experimental Setup（基线清单）
> 原文："We conduct Bayesian optimization using Optuna [1] with 100 trials for each dataset-model-transformation combination, jointly tuning model and transformation hyperparameters (e.g., number of bins for stretch and PLE) using only the training and validation partitions."
> 出处：2608.09162 §4.1 Experimental Setup（协议）
> 原文："Each selected configuration is evaluated with 15 random seeds on the fixed split; these seeds measure initialization, minibatch-order, and optimization variability, not variability over alternative data splits."
> 出处：2608.09162 §4.1 Experimental Setup（协议）
> 原文："We use accuracy for classification and $R^{2}$ (clipped to $[0,1]$) for regression as primary metrics."
> 出处：2608.09162 §4.1 Experimental Setup（指标）

> 原文："We use the TALENT benchmark suite [35, 23], focusing on the Tiny Benchmark 1 collection of 26 classification and 12 regression datasets."
> 出处：2608.09162 §4.1 Experimental Setup（数据集构成：26 分类 + 12 回归）

> 原文："From the original TALENT Benchmark 1 collection of 42 datasets, we exclude four datasets that contain only categorical features (Amazon_employee_access, BNG(tic-tac-toe), led24, splice), yielding our final experimental suite of 38 datasets."
> 出处：2608.09162 §B.3 Experimental Setup Details（42 → 38 的剔除口径）

> 原文："We evaluate five representative neural architectures for tabular data: FT-Transformer (FTT) [9], standard MLP, MLP-PLR [10], RealMLP [17], and ResNet [14]. Experiments span 38 datasets, yielding 190 dataset-model combinations."
> 出处：2608.09162 §4.1 Experimental Setup（5 个架构 / 190 个单元）

**B. 与既有变换的理论联系（CDF / PLE / target encoding）**

> 原文："The win-rate analysis (Figure 3) further shows that it statistically outperforms every alternative under the pairwise criterion, with decisive regression win-loss records of 18-7 against PLE, 28-9 against PLE-T, and 14-4 against standardization (Appendix C)."
> 出处：2608.09162 §4.2 Results and Analysis（Finding 1）
> 原文："Unsupervised stretch consistently ranks as a strong runner-up: it outperforms all methods except its supervised counterpart under the aggregate pairwise comparison (Figure 3) and is especially competitive in classification (Figure 2, right), with classification win-loss records of 38–25 vs. PLE, 38–18 vs. PLE-T, and 26–13 vs. Standardization (Appendix C)."
> 出处：2608.09162 §4.2 Results and Analysis（Finding 2）
> 原文："More importantly, task type drives the largest differences: regression benefits more from advanced transformations than classification, with supervised stretch yielding the largest gains in continuous prediction tasks (Figure 2, middle)."
> 出处：2608.09162 §4.2 Results and Analysis（Finding 3）

**C. 方法与实现细节（闭式解、OOF、基线行分）**

> 原文："Balanced: results are more mixed across methods, with Supervised Stretch and PLE-T trading wins on different model-task pairs, and no single method dominating."
> 出处：2608.09162 §E Performance Breakdown by Feature Composition
> 原文："mlp 0.6343 0.6275 0.5600 0.6835 0.6182 0.5526 0.6653 0.6423 0.5790 mlp_plr 0.6238 0.7347 0.6037 0.4513 0.5763 0.5673 0.6091 0.6443 0.5802"
> 出处：2608.09162 §Table 1（Overall Score 行，按模型分行）
> 原文："mlp 0.7653 0.6173 0.7117 0.6897 0.5874 0.5561 0.7028 0.6203 0.6030 mlp_plr 0.6464 0.7525 0.7463 0.4372 0.5954 0.4796 0.5262 0.6635 0.4907"
> 出处：2608.09162 §Table 1（Reg. Score 行，按模型分行）
> 原文："With this uniform allocation, unsupervised stretch becomes a piecewise linear approximation of the empirical CDF transformation. As $T\to n$, our method converges to the exact empirical CDF, which maps each sample to its normalized rank."
> 出处：2608.09162 §3.3 Unsupervised Stretch（与经验 CDF 的关系）

> 原文："Thus, unsupervised stretch precisely parameterizes the PLE manifold by normalized arc length."
> 出处：2608.09162 §3.3 Unsupervised Stretch（与 PLE 的关系）
> 原文："This equivalence explains their similar empirical performance (Section 4.2), while unsupervised stretch offers significant computational advantages: $O(1)$ versus $O(T)$ memory per feature and no dimensional expansion."
> 出处：2608.09162 §3.3 Unsupervised Stretch
> 原文："Intuitively, in the transformed space, we allocate a larger space to the regions where the target function varies rapidly, effectively equalizing the slope magnitudes across bins and creating a smoother target function in the transformed space."
> 出处：2608.09162 §3.4 Supervised Stretch
> 原文："This is remarkably similar to applying min-max scaling to target encoding. Standard target encoding maps $x_{i}\mapsto f_{i}=\mathbb{E}[t|x=x_{i}]$, and if we scale these values to $[0,1]$, we get essentially the same transformation for monotonic targets."
> 出处：2608.09162 §3.4 Supervised Stretch（与 target encoding 的关系）
> 原文："The proposed supervised stretch offers a regularized, theoretically grounded approach to incorporating target information for numeric features, with the number of bins $T$ serving as a natural regularization parameter."
> 出处：2608.09162 §3.4 Supervised Stretch
> 原文："In practice, to prevent information leakage, we use $K$-fold cross-validation with adaptive Nadaraya-Watson kernel regression to obtain $\widehat{f}(x_{i})$ for each sample without using its own target value [25, 32] (details in Appendix B.1)."
> 出处：2608.09162 §3.4 Supervised Stretch（Out-of-fold estimation）

> 原文："We then average these ranks over the $4\times 5=20$ (dataset, model) combinations."
> 出处：2608.09162 §G.1 Noise Robustness（Experimental Setup：4 × 5 = 20）

**D. 边界、失败模式与论文自承局限**

> 原文："While tree-based models (e.g., Random Forests, Gradient Boosted Trees) are invariant to monotonic transformations [6], neural networks often require standardized inputs to stabilize optimization [19]."
> 出处：2608.09162 §2 Related Work
> 原文："At the highest noise level we test ($\eta=0.5$), it actually has the best mean rank among the five methods."
> 出处：2608.09162 §G.2 Noise Robustness Analysis（Results）
> 原文："In our experiments, if the resulting tensor size exceeds a memory limit, we use a fallback scheme that substitutes standardization for PLE on that specific dataset-model combination. This affects fewer than 5% of the 190 combinations and does not impact our proposed methods."
> 出处：2608.09162 §B.2 Computational Fallback Scheme for PLE
> 原文："Several fallback mechanisms ensure robustness: if a feature has too few unique values, or if the total target variation is negligible, we revert to simpler transformations (identity or unsupervised stretch)."
> 出处：2608.09162 §B.1 Out-of-Fold Kernel Regression
> 原文："Our framework’s design could be enhanced by exploring smoother, spline-based alternatives, and adaptive binning strategies."
> 出处：2608.09162 §5 Conclusion（Limitations and Future Work）
> 原文："The current transformation is marginal and therefore does not optimize a joint or conditional smoothness objective; the downstream model may learn feature interactions, but the preprocessing itself is not interaction-aware."
> 出处：2608.09162 §5 Conclusion（Limitations and Future Work）

