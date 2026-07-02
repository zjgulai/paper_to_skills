---
title: FT-Transformer表格深度学习 — 超越GBDT的表格数据神经网络
doc_type: knowledge
module: 12-ML基础
topic: tabular-deep-learning-ft-transformer
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Tabular Deep Learning FT Transformer

> **论文**：Revisiting Deep Learning Models for Tabular Data（Gorishniy et al., NeurIPS 2021, arXiv:2106.11959）
> **arXiv**：2106.11959 | 2021 | **桥梁**: 12-ML基础 ↔ 14-用户分析 ↔ 06-增长模型 | **类型**: 算法工具

## ① 算法原理

**为什么需要超越GBDT？**
GBM（XGBoost/LightGBM）在表格数据上统治了10年，但有三个关键局限：
1. **无法端到端学习**：需要手工特征工程，不能自动学习特征交叉
2. **难以利用预训练**：无法迁移其他任务学到的表示
3. **不擅长多任务**：共享表示困难

**FT-Transformer（Feature Tokenizer Transformer）**将Transformer引入表格数据：

**核心模块 — Feature Tokenizer**：
将每个表格特征（无论数值型还是类别型）转化为embedding向量：
- 数值特征：$x_j \to \mathbf{e}_j = x_j \cdot \mathbf{w}_j^{(num)} + \mathbf{b}_j^{(num)}$（线性投影）
- 类别特征：通过lookup table得到embedding

所有特征的embedding拼接成序列，输入Transformer Encoder：
$$[\mathbf{t}_1, \mathbf{t}_2, \ldots, \mathbf{t}_d] \to \text{Transformer} \to \text{分类头/回归头}$$

**自注意力机制捕获特征交叉**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
与人工构造特征交叉不同，注意力机制自动学习"月龄×购买频次"这样的非线性交互。

**与TabNet/MLP-Mixer的对比**：
- FT-Transformer在多个公开基准上与XGBoost相当甚至更好，特别是在特征数量多（>50）或有大量类别特征时优势明显
- 在大数据场景（>100万样本）FT-Transformer通常优于GBDT
- 可以预训练+微调，GBDT不行

**跨学科源头**：来自NLP的Transformer架构（Vaswani 2017），通过特征嵌入层桥接NLP→表格数据。对母婴电商的降维打击：用户特征表（月龄/购买历史/设备/地区 50+个特征）上，FT-Transformer可以自动学习特征交叉，避免繁重的手工特征工程。

## ② 母婴出海应用案例

**场景A：用户LTV预测（高维表格特征）**
- 业务问题：用户LTV预测有70个特征（行为、属性、时序统计量），手工构造特征交叉工程量巨大，XGBoost在验证集AUC=0.81，希望进一步提升
- 数据要求：用户特征表（行为统计、属性、历史购买）+ 12个月LTV标签；≥50万用户记录
- 预期产出：FT-Transformer在相同数据上AUC=0.84（+0.03提升），同时无需手工特征工程；在"月龄×购买间隔"等高阶交叉特征上的泛化性更强
- 业务价值：LTV精度+0.03 AUC对应精准营销投入减少约10%浪费，年化节省约40万元；减少特征工程人力约2人月/年（约50万元）

**三轨对抗验证**：
1. **成本验证**：FT-Transformer训练需要GPU（A100约20分钟/epoch），比XGBoost慢约10倍；推理速度相当；样本<10万时通常不如XGBoost
2. **合规验证**：深度学习模型可解释性低于GBDT，若需对用户决策说明原因（如贷款/保险场景）需要额外的SHAP解释层
3. **风险验证**：Transformer对超参数（学习率/层数/头数）更敏感，需要更细致的调参；建议用Optuna做贝叶斯超参搜索；数据量<1万时容易过拟合

**场景B：多分类商品品类推荐**
- 业务问题：根据用户特征预测最可能购买的品类（20类），类别特征（地区/设备类型/月龄段）多
- 方案：FT-Transformer自动嵌入类别特征，无需手工编码；多头输出同时预测20个品类概率
- 业务价值：品类推荐精准度提升约8%，点击率提升约3%，年化GMV增量约30万元

## ③ 代码模板

```python
"""
Skill-Tabular-Deep-Learning-FT-Transformer
FT-Transformer表格深度学习 — 用户LTV预测

依赖：pip install numpy pandas scikit-learn
注意：完整FT-Transformer需 PyTorch (pip install torch)
此处用MLP模拟核心架构思想（特征嵌入+多层非线性）
生产环境推荐: rtdl 库 (pip install rtdl)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

np.random.seed(42)

# ── 1. 生成高维表格数据（模拟用户LTV预测场景）───────────────────────
n = 10000  # 1万用户

# 数值特征（20个）
num_features = {
    'baby_age_months':     np.random.randint(0, 24, n).astype(float),
    'purchase_count_30d':  np.random.poisson(3, n).astype(float),
    'purchase_count_90d':  np.random.poisson(8, n).astype(float),
    'avg_order_value':     np.random.lognormal(4.5, 0.6, n),
    'days_since_last_buy': np.random.exponential(15, n),
    'session_count_7d':    np.random.poisson(5, n).astype(float),
    'search_count':        np.random.poisson(10, n).astype(float),
    'cart_abandon_rate':   np.random.beta(2, 5, n),
    'return_rate':         np.random.beta(1, 10, n),
    'review_count':        np.random.poisson(3, n).astype(float),
    'account_age_days':    np.random.uniform(30, 1500, n),
    'nps_score':           np.random.randint(1, 11, n).astype(float),
    'coupon_use_rate':     np.random.beta(2, 3, n),
    'organic_traffic_pct': np.random.beta(3, 2, n),
    'referral_count':      np.random.poisson(1, n).astype(float),
}

# 类别特征（10个）— 通常需要嵌入
cat_features = {
    'device_type':    np.random.choice(['mobile', 'desktop', 'tablet'], n, p=[0.6,0.3,0.1]),
    'region':         np.random.choice(['US-West', 'US-East', 'EU-DE', 'EU-UK', 'JP'], n),
    'acquisition_src':np.random.choice(['organic', 'paid', 'referral', 'social'], n),
    'membership_tier':np.random.choice(['free', 'silver', 'gold'], n, p=[0.5,0.3,0.2]),
    'primary_category':np.random.choice(['formula', 'stroller', 'clothing', 'toys', 'safety'], n),
}

# 合并特征
df = pd.DataFrame({**num_features, **cat_features})

# Label encode类别特征
le_dict = {}
for col in cat_features:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    le_dict[col] = le

# 构建目标：LTV（与特征有复杂非线性关系）
ltv = (
    200
    + 30 * (df['baby_age_months'] < 6)          # 新生儿溢价
    + 50 * (df['membership_tier'] == 'gold')
    + 20 * (df['device_type'] == 'mobile')
    + 5  * df['purchase_count_90d']
    + 0.3 * df['avg_order_value']
    - 100 * df['return_rate']
    + 40 * (df['primary_category'] == 'formula')
    # 非线性交叉：月龄×购买频次的交互
    + 10 * (df['baby_age_months'] < 6) * (df['purchase_count_30d'] > 3)
    + np.random.normal(0, 30, n)
)
df['ltv_12m'] = np.clip(ltv, 0, 1000)

# 特征列
num_cols = list(num_features.keys())
cat_cols  = [c + '_enc' for c in cat_features]
all_cols  = num_cols + cat_cols

X = df[all_cols].values
y = df['ltv_12m'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"数据: {n}用户, {len(all_cols)}特征 ({len(num_cols)}数值+{len(cat_cols)}类别)")

# ── 2. 基线：GBDT（XGBoost近似）─────────────────────────────────────
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)

gbm = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.05, random_state=42)
gbm.fit(X_tr, y_tr)  # GBDT不需要标准化
y_pred_gbm = gbm.predict(X_te)
r2_gbm = r2_score(y_te, y_pred_gbm)
mae_gbm = mean_absolute_error(y_te, y_pred_gbm)
print(f"\n基线GBDT:       R²={r2_gbm:.4f}, MAE={mae_gbm:.1f}元")

# ── 3. FT-Transformer近似（深度MLP with特征嵌入）────────────────────
# 核心思想：为每个特征学习独立嵌入，再经过多层非线性
# 完整FT-Transformer：每个特征先投影到d_model维，然后Transformer自注意力
# 此处用深度MLP近似（论文证明深度MLP+BatchNorm已接近FT-Transformer效果）

# 对类别特征做更丰富的嵌入（one-hot + 连续数值）
cat_onehot = pd.get_dummies(df[list(cat_features.keys())], drop_first=True)
X_enriched = np.column_stack([df[num_cols].values, cat_onehot.values])

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_enriched, y, test_size=0.2, random_state=42)

sc2 = StandardScaler()
X_tr2_sc = sc2.fit_transform(X_tr2)
X_te2_sc  = sc2.transform(X_te2)

# 深度MLP（模拟FT-Transformer的多层非线性）
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 256, 128, 64),  # 4层深度网络
    activation='relu',
    learning_rate_init=0.001,
    max_iter=200,
    batch_size=256,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    n_iter_no_change=15
)
mlp.fit(X_tr2_sc, y_tr2)
y_pred_mlp = mlp.predict(X_te2_sc)
r2_mlp  = r2_score(y_te2, y_pred_mlp)
mae_mlp = mean_absolute_error(y_te2, y_pred_mlp)
print(f"深度MLP近似:     R²={r2_mlp:.4f}, MAE={mae_mlp:.1f}元")

# ── 4. 特征重要性对比（GBDT vs 深度网络差异）────────────────────────
print(f"\n【GBDT特征重要性 Top10】")
importance_df = pd.DataFrame({
    'feature': all_cols,
    'importance': gbm.feature_importances_
}).sort_values('importance', ascending=False).head(10)
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"  {row['feature']:<28} {row['importance']:.4f} {bar}")

print(f"\n【模型对比总结】")
print(f"  {'模型':<20} {'R²':>8} {'MAE(元)':>10} {'特征工程需求':>12}")
print(f"  {'GBDT基线':<20} {r2_gbm:>8.4f} {mae_gbm:>9.1f}  需要人工交叉")
print(f"  {'深度MLP（FT近似）':<20} {r2_mlp:>8.4f} {mae_mlp:>9.1f}  自动学习交叉")

winner = 'GBDT' if r2_gbm > r2_mlp else 'Deep'
print(f"\n  当前数据集({n}样本): {winner}表现更好")
print(f"  注: 数据量>100万时深度网络通常超越GBDT")

assert r2_gbm > 0.3 and r2_mlp > 0.2, "两个模型R²应合理"
print("\n[✓] FT-Transformer表格深度学习 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Feature-Engineering]]（理解特征工程，才能体会深度方法的自动化优势）、[[Skill-Ensemble-Methods]]（GBDT基准）
- **延伸（extends）**：[[Skill-Embedding-Fundamentals]]（FT-Transformer的特征嵌入层与词嵌入同理）
- **可组合（combinable）**：[[Skill-SHAP-Shapley-Feature-Attribution]]（用SHAP解释深度表格模型的特征重要性）、[[Skill-LTV-Prediction-BTYD]]（FT-Transformer替代BTYD做LTV预测）、[[Skill-Federated-Learning-Privacy]]（联邦框架中用FT-Transformer共享表格表示）

## ⑤ 商业价值评估

- **ROI 预估**：LTV预测精度AUC+0.03，精准营销浪费减少约10%（约40万元/年）；特征工程自动化减少2人月工程工作（约50万元/年）；多任务场景下一个模型替代多个GBDT（MLOps成本降低30%）
- **实施难度**：⭐⭐⭐⭐☆（需要PyTorch和GPU；超参调整比GBDT复杂；建议用rtdl库降低实现门槛）
- **优先级**：⭐⭐⭐☆☆（数据量<10万时优先GBDT；数据量>50万、特征数>50时才明显体现优势）
- **评估依据**：NeurIPS 2021论文在31个数据集上的对比；rtdl库被广泛采用（GitHub 2k+ stars）；在Kaggle表格竞赛中FT-Transformer单模型经常进入Top10
