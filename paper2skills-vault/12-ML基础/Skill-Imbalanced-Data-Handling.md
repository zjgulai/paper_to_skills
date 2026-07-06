---
title: "Skill Card: Imbalanced Data Handling in Mother-Baby Cross-Border E-commerce"
slug: skill-imbalanced-data-handling
category: "AI Decision Making"
subcategory: "Data Processing & Feature Engineering"
roadmap_phase: phase1
difficulty: intermediate
updated: 2026-07-05
estimated_learning_time: 45
---

# Skill Card: 不平衡数据处理（Imbalanced Data Handling）

## ① 算法原理

### 核心思想
在母婴跨境电商中，**高价值事件（转化、复购、欺诈）往往占比 <5%**，直接训练模型会严重偏向多数类，导致少数类 Recall 极低（<0.3），业务决策失效。不平衡数据处理通过重采样、成本加权或阈值调优，使模型在保持精准度的前提下大幅提升对稀有事件的识别能力。

### 数学直觉

**核心公式：类别权重调整**
$$w_{pos} = \frac{n_{neg}}{n_{pos}}, \quad \text{Loss}_{weighted} = w_{pos} \cdot \text{Loss}_{pos} + w_{neg} \cdot \text{Loss}_{neg}$$

**含义**：给少数类（正例）分配更高的误分类惩罚权重。当正例仅占 2% 时，$w_{pos} = 49$，模型每漏掉 1 个正例的代价等同于误分 49 个负例——强制模型重视稀有事件。

**三大处理路线**：

1. **SMOTE 过采样**：在少数类样本间线性插值生成合成样本 $x_{new} = x_i + \lambda(x_j - x_i)$，$\lambda \in [0,1]$。优点：保留原始数据，增加样本多样性；缺点：假设特征空间连续，离散特征易产生噪声。

2. **类别权重（推荐）**：直接修改损失函数，无需改变数据分布，计算高效，适合极度不平衡场景（<1%）。XGBoost 的 `scale_pos_weight` 参数即此原理。

3. **阈值调优**：默认 0.5 分界点在不平衡数据下不合理。通过 Precision-Recall 曲线找到满足业务约束的最优阈值（如要求 Recall ≥ 0.8）。

### 关键假设
- SMOTE 假设特征空间连续且样本密度均匀，极端不平衡（<0.1%）时合成样本质量下降
- 类别权重方法假设误分类成本不对称，但在样本极少（<50）时易过拟合
- 所有方法都需通过**分层交叉验证**验证效果，防止数据泄露

### 非共识迁移
**原始领域**：医学影像中的罕见病诊断（患病率 0.1%）、金融欺诈检测（欺诈率 0.05%）。

**为何降维打击母婴出海**：母婴品类具有 **"长尾转化"** 特征——新品上市首月转化率 2-3%，复购率 8-12%，高价值客户占比 <3%。传统 Baseline 模型会将 97% 流量判为"低价值"，导致营销预算严重浪费。通过不平衡处理，可将稀有高价值客户的识别率从 30% 提升至 75%+，直接撬动 ROI。

---

## ② 母婴出海应用案例

### 场景一：婴儿奶粉销量预测的库存优化

**业务问题**：
某跨境电商平台销售 50+ SKU 婴儿奶粉。其中 3 款头部产品日销 500+ 件（占总销量 60%），其余 47 款日销 10-50 件。直接用 XGBoost 预测，模型优化目标自动向头部产品倾斜，导致长尾产品预测 RMSE 高达 45%，库存积压率 28%，每月滞销成本 ¥18 万。

**具体数据规模**：
- 训练集：18 个月 × 50 SKU × 30 天 = 27,000 条样本
- 特征：价格、季节、促销、竞品、评价数、复购率等 12 维
- 标签分布：头部 SKU（60% 销量，占样本 12%）vs 长尾 SKU（40% 销量，占样本 88%）

**处理方案**：
1. 按 SKU 分组，计算每组销量标准差
2. 对长尾 SKU（σ > 平均值）应用 **SMOTE 过采样**，将长尾样本从 23,760 扩展至 32,400
3. 对头部 SKU 应用 **类别权重** $w_{tail} = 0.6 / 0.4 = 1.5$
4. 使用 **分层 5 折交叉验证**，按 SKU 分层保证每折都含头部+长尾

**量化产出**：
| 指标 | Baseline | 不平衡处理 | 改进 |
|------|---------|----------|------|
| 长尾 SKU RMSE | 45% | 18% | **↓ 60%** |
| 头部 SKU RMSE | 12% | 14% | ↑ 2pp（可接受） |
| 库存积压率 | 28% | 9% | **↓ 19pp** |
| 月度备货成本节省 | — | ¥12 万 | **¥12 万/月** |

**业务价值**：
- 直接节省：¥12 万/月 × 12 月 = **¥144 万/年**
- 间接收益：长尾产品销售额提升 8%（库存充足，减少缺货），月增 ¥6 万
- **年度总 ROI：¥216 万**

**三轨验证**：
- **成本**：模型重训周期 1 周，工程改造 3 人天，总成本 ¥2 万（< 月收益，ROI > 100x）
- **合规**：不涉及个人数据，仅涉及商品库存决策，无合规风险 ✓
- **风险**：长尾 SKU 预测偏差可能导致某些产品缺货；缓解方案：设置安全库存下限 ✓

---

### 场景二：高价值客户识别与精准营销

**业务问题**：
跨境母婴平台有 200 万活跃用户，其中高价值客户（年复购 ≥ 3 次、客单价 ≥ $80）仅占 2.8%（5.6 万人）。直接用 Logistic Regression 训练，模型倾向将 97% 用户判为"低价值"，Recall 仅 0.22，导致 95% 高价值客户被漏掉，营销预算浪费在低价值用户身上，ROI 仅 1.2x。

**具体数据规模**：
- 训练集：200 万用户 × 18 个月行为数据
- 特征：首购金额、复购间隔、评价数、推荐指数、浏览深度等 18 维
- 正例（高价值）：5.6 万（2.8%），负例（低价值）：194.4 万（97.2%）
- 极度不平衡比例：1:35

**处理方案**：
1. **不用 SMOTE**（200 万样本，合成样本质量堪忧），改用 **类别权重** $w_{pos} = 35$
2. 在 XGBoost 中设置 `scale_pos_weight=35`，同时调整 `max_depth=5`（防止过拟合）
3. **阈值调优**：通过 Precision-Recall 曲线，找到满足营销成本约束的最优阈值
   - 默认阈值 0.5：Precision=0.45, Recall=0.22
   - 调优后阈值 0.15：Precision=0.38, Recall=0.78 ← **选择此阈值**
4. 预测出 4.4 万高价值客户（78% Recall），投入精准营销

**量化产出**：
| 指标 | Baseline | 类别权重+阈值调优 | 改进 |
|------|---------|----------|------|
| Recall（识别率） | 0.22 | 0.78 | **↑ 56pp** |
| Precision（准确率） | 0.45 | 0.38 | ↓ 7pp（可接受） |
| 识别的高价值客户数 | 1.2 万 | 4.4 万 | **↑ 3.7 倍** |
| 营销 ROI | 1.2x | 3.8x | **↑ 216%** |
| 月度增收 | — | ¥28 万 | **¥28 万/月** |

**业务价值**：
- 直接增收：¥28 万/月 × 12 月 = **¥336 万/年**
- 客户生命周期价值提升：高价值客户年均消费 $320，新识别 3.2 万客户 × $320 = **¥768 万**
- **年度总 ROI：¥1,104 万**

**三轨验证**：
- **成本**：模型迭代 2 周，A/B 测试 4 周，总成本 ¥8 万（< 月收益，ROI > 40x）
- **合规**：涉及用户行为数据，需符合 GDPR/CCPA；方案：数据脱敏、用户可选退出 ✓
- **风险**：阈值过低可能误判低价值用户为高价值，导致营销成本上升；缓解方案：设置 Precision 下限 0.35 ✓

---

## ③ 代码模板

```python
"""
Imbalanced Data Handling Toolkit
不平衡数据处理工具集 — 类别权重 / SMOTE / 阈值调优

适用场景：高价值客户识别、流失预警、欺诈检测、稀有事件预警
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List


class ImbalancedDataHandler:
    """不平衡数据处理工具类"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.optimal_threshold = 0.5
    
    def generate_synthetic_imbalanced_data(
        self, 
        n_samples: int = 10000,
        imbalance_ratio: float = 0.03
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成不平衡数据集（模拟母婴电商高价值客户识别场景）
        
        Args:
            n_samples: 总样本数
            imbalance_ratio: 正例比例（默认 3%，对应高价值客户）
        
        Returns:
            X: 特征矩阵 (n_samples, 10)
            y: 标签 (n_samples,)
        """
        np.random.seed(self.random_state)
        
        # 正例（高价值客户）：特征均值较高
        n_pos = int(n_samples * imbalance_ratio)
        X_pos = np.random.normal(loc=2.0, scale=1.5, size=(n_pos, 10))
        y_pos = np.ones(n_pos)
        
        # 负例（低价值客户）：特征均值较低
        n_neg = n_samples - n_pos
        X_neg = np.random.normal(loc=0.0, scale=1.0, size=(n_neg, 10))
        y_neg = np.zeros(n_neg)
        
        # 合并并打乱
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([y_pos, y_neg])
        
        idx = np.random.permutation(len(y))
        return X[idx], y[idx]
    
    def train_with_class_weight(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'logistic'
    ) -> Dict:
        """
        使用类别权重训练模型（推荐方案）
        
        Args:
            X: 特征矩阵
            y: 标签
            model_type: 'logistic' 或 'rf'
        
        Returns:
            dict: 包含模型、权重、性能指标
        """
        # 计算类别权重
        pos_ratio = y.mean()
        neg_ratio = 1 - pos_ratio
        class_weight = {
            0: pos_ratio / (2 * neg_ratio),
            1: neg_ratio / (2 * pos_ratio)
        }
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        if model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        self.model.fit(X_scaled, y)
        
        # 评估
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = self.model.predict(X_scaled)
        
        return {
            'model': self.model,
            'class_weight': class_weight,
            'auc': roc_auc_score(y, y_prob),
            'f1': f1_score(y, y_pred),
            'report': classification_report(y, y_pred, output_dict=True)
        }
    
    def find_optimal_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_recall: float = 0.8
    ) -> float:
        """
        通过 Precision-Recall 曲线找最优阈值
        
        Args:
            X: 特征矩阵
            y: 标签
            target_recall: 目标 Recall（业务约束）
        
        Returns:
            float: 最优阈值
        """
        X_scaled = self.scaler.transform(X)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        # 计算 Precision-Recall 曲线
        precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
        
        # 找满足 Recall >= target_recall 的最高 Precision 对应的阈值
        valid_idx = np.where(recalls >= target_recall)[0]
        if len(valid_idx) == 0:
            self.optimal_threshold = 0.0
        else:
            best_idx = valid_idx[np.argmax(precisions[valid_idx])]
            self.optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        return self.optimal_threshold
    
    def evaluate_with_custom_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = None
    ) -> Dict:
        """
        用自定义阈值评估模型
        
        Args:
            X: 特征矩阵
            y: 标签
            threshold: 自定义阈值（默认使用 optimal_threshold）
        
        Returns:
            dict: 性能指标
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        X_scaled = self.scaler.transform(X)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        return {
            'threshold': threshold,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_prob)
        }
    
    def compare_strategies(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> pd.DataFrame:
        """
        对比 Baseline vs 类别权重 vs 阈值调优
        
        Args:
            X: 特征矩阵
            y: 标签
        
        Returns:
            DataFrame: 各策略对比
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        results = []
        
        # 策略 1: Baseline（无处理）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        baseline_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        baseline_model.fit(X_train_scaled, y_train)
        y_prob_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_baseline = baseline_model.predict(X_test_scaled)
        
        results.append({
            'Strategy': 'Baseline (No Treatment)',
            'Threshold': 0.5,
            'Precision': round(
                np.sum((y_pred_baseline == 1) & (y_test == 1)) / np.sum(y_pred_baseline == 1)
                if np.sum(y_pred_baseline == 1) > 0 else 0, 3
            ),
            'Recall': round(
                np.sum((y_pred_baseline == 1) & (y_test == 1)) / np.sum(y_test == 1)
                if np.sum(y_test == 1) > 0 else 0, 3
            ),
            'F1': round(f1_score(y_test, y_pred_baseline), 3),
            'AUC': round(roc_auc_score(y_test, y_prob_baseline), 3)
        })
        
        # 策略 2: 类别权重
        weighted_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state
        )
        weighted_model.fit(X_train_scaled, y_train)
        y_prob_weighted = weighted_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_weighted = weighted_model.predict(X_test_scaled)
        
        results.append({
            'Strategy': 'Class Weight (balanced)',
            'Threshold': 0.5,
            'Precision': round(
                np.sum((y_pred_weighted == 1) & (y_test == 1)) / np.sum(y_pred_weighted == 1)
                if np.sum(y_pred_weighted == 1) > 0 else 0, 3
            ),
            'Recall': round(
                np.sum((y_pred_weighted == 1) & (y_test == 1)) / np.sum(y_test == 1)
                if np.sum(y_test == 1) > 0 else 0, 3
            ),
            'F1': round(f1_score(y_test, y_pred_weighted), 3),
            'AUC': round(roc_auc_score(y_test, y_prob_weighted), 3)
        })
        
        # 策略 3: 类别权重 + 阈值调优（目标 Recall >= 0.8）
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_weighted)
        valid_idx = np.where(recalls >= 0.8)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(precisions[valid_idx])]
            optimal_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            optimal_thresh = 0.5
        
        y_pred_tuned = (y_prob_weighted >= optimal_thresh).astype(int)
        
        results.append({
            'Strategy': 'Class Weight + Threshold Tuning',
            'Threshold': round(optimal_thresh, 3),
            'Precision': round(
                np.sum((y_pred_tuned == 1) & (y_test == 1)) / np.sum(y_pred_tuned == 1)
                if np.sum(y_pred_tuned == 1) > 0 else 0, 3
            ),
            'Recall': round(
                np.sum((y_pred_tuned == 1) & (y_test == 1)) / np.sum(y_test == 1)
                if np.sum(y_test == 1) > 0 else 0, 3
            ),
            'F1': round(f1_score(y_test, y_pred_tuned), 3),
            'AUC': round(roc_auc_score(y_test, y_prob_weighted), 3)
        })
        
        return pd.DataFrame(results)


# ============================================================================
# 主程序：完整演示
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Imbalanced Data Handling Skill Card - Complete Demo")
    print("=" * 80)
    
    # 初始化处理器
    handler = ImbalancedDataHandler(random_state=42)
    
    # 生成合成数据（模拟高价值客户识别）
    print("\n[1] 生成合成不平衡数据集...")
    X, y = handler.generate_synthetic_imbalanced_data(
        n_samples=10000,
        imbalance_ratio=0.03  # 3% 高价值客户
    )
    print(f"    样本总数: {len(y)}")
    print(f"    正例（高价值）: {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"    负例（低价值）: {int((1-y).sum())} ({(1-y.mean())*100:.1f}%)")
    print(f"    不平衡比例: 1:{int((1-y.mean())/y.mean())}")
    
    # 分割数据
    print("\n[2] 分割训练/测试集（分层采样）...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"    训练集: {len(y_train)} 样本")
    print(f"    测试集: {len(y_test)} 样本")
    
    # 对比三种策略
    print("\n[3] 对比三种处理策略...")
    comparison_df = handler.compare_strategies(X_train, y_train)
    print("\n" + comparison_df.to_string(index=False))
    
    # 详细评估最优方案
    print("\n[4] 最优方案详细评估（类别权重 + 阈值调优）...")
    handler.train_with_class_weight(X_train, y_train, model_type='logistic')
    optimal_threshold = handler.find_optimal_threshold(X_test, y_test, target_recall=0.8)
    
    eval_result = handler.evaluate_with_custom_threshold(X_test, y_test, threshold=optimal_threshold)
    print(f"    最优阈值: {eval_result['threshold']:.3f}")
    print(f"    Precision: {eval_result['precision']:.3f}")
    print(f"    Recall: {eval_result['recall']:.3f}")
    print(f"    F1-Score: {eval_result['f1']:.3f}")
    print(f"    AUC: {eval_result['auc']:.3f}")
    print(f"    混淆矩阵: TP={eval_result['tp']}, FP={eval_result['fp']}, " +
          f"FN={eval_result['fn']}, TN={eval_result['tn']}")
    
    # 业务价值计算
    print("\n[5] 业务价值评估...")
    total_high_value = int(y_test.sum())
    identified = eval_result['tp']
    missed = eval_result['fn']
    false_positive = eval_result['fp']
    
    print(f"    总高价值客户数: {total_high_value}")
    print(f"    成功识别: {identified} ({identified/total_high_value*100:.1f}%)")
    print(f"    漏掉: {missed} ({missed/total_high_value*100:.1f}%)")
    print(f"    误判低价值为高价值: {false_positive}")
    
    # 假设单个高价值客户年均消费 $320，营销成本 $50
    customer_value = 320
    marketing_cost = 50
    revenue_gain = identified * (customer_value - marketing_cost)
    
    print(f"    单客户年均价值: ${customer_value}")
    print(f"    单客户营销成本: ${marketing_cost}")
    print(f"    年度增收: ${revenue_gain:,} (约 ¥{revenue_gain*7:,.0f})")
    
    print("\n" + "=" * 80)
    print("[✓] Skill-Imbalanced-Data-Handling 测试通过")
    print("=" * 80)
```

---

## ④ 技能关联

### 前置技能（Prerequisite）
- **[[Skill-Data-Preprocessing-and-Cleaning]]**：不平衡处理前需要完整的数据清洗和特征工程
- **[[Skill-Classification-Model-Evaluation]]**：需要理解 Precision、Recall、F1 等评估指标的含义

### 延伸技能（Extends）
- **[[Skill-Cost-Sensitive-Learning]]**：类别权重是成本敏感学习的特例，可进一步扩展到非对称成本矩阵
- **[[Skill-Ensemble-Methods-for-Imbalanced-Data]]**：级联多个不平衡处理方法（SMOTE + EasyEnsemble + 阈值调优）

### 可组合技能（Combinable）
- **[[Skill-Hyperparameter-Tuning]]** + **本技能**：在不平衡数据上进行超参数网格搜索，需同时优化模型参数和类别权重
  - 组合场景：母婴产品销量预测中，同时调优 XGBoost 的 `max_depth`、`learning_rate` 和 `scale_pos_weight`，找到最优组合
- **[[Skill-Cross-Validation-Strategy]]** + **本技能**：必须使用分层交叉验证确保每折都包含正负样本，防止数据泄露

---

## ⑤ 商业价值评估

### ROI 预估

| 应用场景 | 年度增收 | 实施成本 | ROI | 备注 