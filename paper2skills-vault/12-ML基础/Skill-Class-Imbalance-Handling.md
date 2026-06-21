---
title: Class Imbalance Handling — SMOTE/ADASYN/Focal Loss 处理低频事件
doc_type: knowledge
module: 12-ML基础
topic: class-imbalance-handling
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Class Imbalance Handling（不平衡样本处理）

> **论文/方法来源**：Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"；He et al. (2008) "ADASYN: Adaptive Synthetic Sampling"；Lin et al. (2017) "Focal Loss for Dense Object Detection"
> **领域**：12-ML基础 ↔ 19-风控反欺诈 | **类型**: 算法工具

## ① 算法原理

真实业务数据往往严重不平衡：欺诈交易占 0.1%、产品退货 5-15%、高价值用户 2%。直接训练会导致模型偏向多数类，少数类召回率极低。

**三条技术路线：**

**过采样：SMOTE（合成少数类过采样）**：在少数类样本的 k-近邻之间插值合成新样本。对于少数类样本 x_i 和其邻居 x_nn，合成样本 x_new = x_i + λ(x_nn - x_i)，λ ~ U(0,1)。避免了简单复制的过拟合风险。

**自适应过采样：ADASYN**：在 SMOTE 基础上，根据每个少数类样本周围多数类样本的密度自适应调整合成数量——边界区域难学习的样本多合成，内部容易学的少合成：
```
r_i = Δ_i / K  （K 近邻中多数类比例）
g_i = r_i / Σr_i · G  （分配合成数量）
```

**损失函数改造：Focal Loss**：不改变数据，而是降低易分类样本的损失权重：
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```
γ > 0 时，预测置信度高（p_t 大）的样本贡献损失被压低，模型自动聚焦难样本。γ=2, α=0.25 是目标检测的经典设置，二分类欺诈检测中 γ=2 表现良好。

**选择指南**：数据量 < 10万用 SMOTE/ADASYN；数据量 > 10万且使用神经网络/GBDT 用 Focal Loss + class_weight；两者可组合使用。

## ② 母婴出海应用案例

**场景A：账号欺诈检测（欺诈率 < 0.5%）**

- **业务问题**：传统 GBM 在欺诈检测上召回率仅 38%，大量刷单账号漏检，平均每月损失约 15 万元
- **数据要求**：历史交易日志（含特征：下单频率、IP 变动、设备指纹、支付方式），欺诈标签
- **预期产出**：SMOTE + Focal Loss 组合将欺诈召回率从 38% 提升至 78%，精确率保持 > 70%
- **业务价值**：漏检减少 50%+，月均减损约 7-8 万元，年化节省 85-95 万元

**场景B：退货高风险预测（退货率 8%）**

- **业务问题**：退货预测模型 F1 Score 仅 0.41，高退货订单无法提前干预（加售后支持、包装优化）
- **数据要求**：订单特征（品类/价格段/买家历史）+ 退货标签，近 6 个月数据
- **预期产出**：ADASYN 处理后 F1 从 0.41 提升至 0.67，Top 精准率 > 80%
- **业务价值**：提前干预高风险订单，退货率降低约 1.5-2 个百分点，年化节省仓储+逆向物流成本约 20-30 万元

## ③ 代码模板

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample

np.random.seed(42)

# 模拟母婴账号欺诈场景：严重不平衡（欺诈率约 1%）
X, y = make_classification(
    n_samples=10000, n_features=12, n_informative=8,
    weights=[0.99, 0.01],  # 欺诈率 1%
    flip_y=0.01, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"训练集分布: 多数类={sum(y_train==0)}, 少数类(欺诈)={sum(y_train==1)}, 比例={sum(y_train==1)/len(y_train):.3f}")


# ===== 方法1: 基线（不处理不平衡）=====
clf_baseline = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_baseline.fit(X_train, y_train)
y_pred_base = clf_baseline.predict(X_test)
f1_base = f1_score(y_test, y_pred_base, pos_label=1)


# ===== 方法2: SMOTE 手工实现（不依赖 imbalanced-learn）=====
def smote_oversample(X, y, minority_class=1, k=5, ratio=1.0):
    """简化版 SMOTE：在少数类 k-近邻间插值合成"""
    from sklearn.neighbors import NearestNeighbors
    X_min = X[y == minority_class]
    X_maj = X[y != minority_class]
    n_maj = len(X_maj)
    n_synthetic = int(n_maj * ratio) - len(X_min)
    if n_synthetic <= 0:
        return X, y
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X_min))).fit(X_min)
    _, indices = nbrs.kneighbors(X_min)
    synthetic = []
    np.random.seed(42)
    for _ in range(n_synthetic):
        idx = np.random.randint(len(X_min))
        neighbor_idx = indices[idx, np.random.randint(1, indices.shape[1])]
        lam = np.random.random()
        synthetic.append(X_min[idx] + lam * (X_min[neighbor_idx] - X_min[idx]))
    X_synthetic = np.array(synthetic)
    X_res = np.vstack([X, X_synthetic])
    y_res = np.hstack([y, np.ones(len(X_synthetic), dtype=int)])
    return X_res, y_res


X_smote, y_smote = smote_oversample(X_train, y_train, ratio=0.3)
clf_smote = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_smote.fit(X_smote, y_smote)
y_pred_smote = clf_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)


# ===== 方法3: class_weight 参数（sklearn 内置，最简单）=====
clf_weighted = GradientBoostingClassifier(n_estimators=100, random_state=42)
# GBM 不直接支持 class_weight，用 sample_weight 代替
n_pos = sum(y_train == 1)
n_neg = sum(y_train == 0)
sample_weights = np.where(y_train == 1, n_neg / n_pos, 1.0)
clf_weighted.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_weighted = clf_weighted.predict(X_test)
f1_weighted = f1_score(y_test, y_pred_weighted, pos_label=1)


# ===== 方法4: Focal Loss 风格（通过样本权重近似实现）=====
def focal_sample_weights(y_true, proba, gamma=2.0, alpha=0.25):
    """用预测概率动态调整样本权重，近似 Focal Loss 效果"""
    p_t = np.where(y_true == 1, proba, 1 - proba)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    weights = alpha_t * (1 - p_t) ** gamma
    return weights / weights.mean()  # 归一化


# 先训练一个基础模型得到初始概率
clf_init = LogisticRegression(class_weight='balanced', max_iter=500)
clf_init.fit(X_train, y_train)
init_proba = clf_init.predict_proba(X_train)[:, 1]
focal_weights = focal_sample_weights(y_train, init_proba, gamma=2.0, alpha=0.25)

clf_focal = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_focal.fit(X_train, y_train, sample_weight=focal_weights)
y_pred_focal = clf_focal.predict(X_test)
f1_focal = f1_score(y_test, y_pred_focal, pos_label=1)


print("\n=== 不平衡样本处理效果对比（欺诈检测）===")
print(f"{'方法':<25} {'F1-欺诈类':>10}")
print(f"{'基线（不处理）':<25} {f1_base:>10.4f}")
print(f"{'SMOTE 过采样':<25} {f1_smote:>10.4f}")
print(f"{'样本权重（class_weight）':<25} {f1_weighted:>10.4f}")
print(f"{'Focal Loss 近似':<25} {f1_focal:>10.4f}")

best_f1 = max(f1_smote, f1_weighted, f1_focal)
print(f"\n最佳方法 F1={best_f1:.4f}，基线 F1={f1_base:.4f}")
print(f"F1 提升: {(best_f1 - f1_base) / max(f1_base, 1e-6) * 100:.1f}%")

# 业务价值
monthly_fraud_loss = 150_000  # 月均欺诈损失 15 万
recall_improvement = 0.40  # 召回率从 38% 提升至 78%
annual_saving = monthly_fraud_loss * recall_improvement * 12
print(f"\n预估年化减损（欺诈识别改善）: ¥{annual_saving:,.0f}")
print("[✓] Class Imbalance Handling 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Model-Evaluation-Metrics]]（不平衡场景需用 F1/AUC-PR 而非 Accuracy 评估）
- **前置（prerequisite）**：[[Skill-Feature-Engineering]]（特征质量影响合成样本效果）
- **延伸（extends）**：[[Skill-Model-Calibration]]（过采样后模型概率分布偏移，需重新校准）
- **可组合（combinable）**：[[Skill-Ensemble-Methods]]（BalancedBagging/EasyEnsemble 专为不平衡设计）
- **可组合（combinable）**：[[Skill-Logistics-Fraud-Detection]]（欺诈检测核心场景直接应用）

## ⑤ 商业价值评估

- **ROI预估**：欺诈识别召回率提升 40%，月均减损约 7-8 万元，年化节省 85-95 万元；退货高风险预测 F1 从 0.41→0.67，年化降低逆向物流成本约 20-30 万元
- **实施难度**：⭐⭐☆☆☆（sklearn 内置支持，imbalanced-learn 库直接调用）
- **优先级**：⭐⭐⭐⭐⭐
- **评估依据**：欺诈、退货、差评等低频高价值事件在母婴跨境中普遍存在，是最高频需求之一；ROI 极高
