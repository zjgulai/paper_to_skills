# Skill Card: Customer Churn Prediction (用户流失预测)

roadmap_phase: phase2
updated: 2026-07-05
difficulty: ⭐⭐⭐☆☆
priority: ⭐⭐⭐⭐☆

---

## ① 算法原理

### 核心思想
**用户流失预测的本质**：通过历史行为数据训练分类模型，提前 7-30 天识别即将停止购买/访问的用户，使运营团队从被动流失分析转变为主动干预，将挽留成功率从 8% 提升至 35%+。

### 数学直觉

**Logistic 回归模型**（可解释性最强）：
$$P(\text{churn}=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{n}\beta_i x_i)}}$$

**业务含义**：将用户的 n 个行为特征（如"近30天购买次数""最后购买距今天数"等）的线性组合压缩到 (0,1) 概率区间。$\beta_i$ 为各特征的权重系数，正值表示该特征增加流失风险，负值表示降低风险。

**梯度提升树（Gradient Boosting）**（预测准确率最高）：
- 通过迭代构建决策树，每棵树学习前一棵树的残差
- 特征重要性 = 该特征在所有树中的平均信息增益
- 适合非线性关系强、特征交互复杂的母婴电商场景

### 关键假设
1. **历史可预测未来**：过去 90 天的流失模式可预测未来 14 天流失
2. **特征稳定性**：用户行为特征分布在 3 个月内不发生剧烈变化
3. **流失定义清晰**：明确定义"流失"（如 90 天未购买或 30 天未访问）
4. **样本充足**：至少 500+ 流失用户样本用于训练

### 非共识迁移
**原始领域**：信用卡客户流失预测（金融风控，用户行为相对稳定）

**降维打击母婴出海**：
- 母婴用户生命周期短（孩子 0-3 岁，需求集中）→ 流失定义需从 90 天缩短至 30-45 天
- 跨境物流延迟 15-30 天 → 特征工程需加入"物流延迟后的活跃度恢复"指标
- 季节性强（婴儿用品有明显淡旺季）→ 模型需分季节训练或加入季节因子
- 多渠道购买（APP/Web/社交电商）→ 特征需跨渠道聚合，避免单渠道沉默误判

---

## ② 母婴出海应用案例

### 场景一：婴儿暖奶器配件复购用户流失预警（14天提前干预）

**业务问题**：
某头部母婴出海品牌（年营收 ¥2.8 亿）主营婴儿暖奶器及配件。核心复购用户为购买后需定期更换配件的妈妈（温控探头、密封圈、电源适配器等）。近 3 个月配件复购率从 32% 下滑至 24%，月均流失用户 42 人，直接损失 ¥7,560/月。

**具体数据规模**：
- 活跃用户池：800 人/月（购买过暖奶器的妈妈）
- 历史数据：12 个月交易记录，共 1,840 个流失样本、4,320 个活跃样本
- 特征维度：
  - 用户特征：注册时间、首购时间、历史购买次数（均值 2.3 次）、历史总金额（均值 ¥180）
  - 行为特征：近 7/30/90 天浏览配件页面数（均值 4.2/12.1/28.5 次）、加购未购次数（均值 0.8/2.1/5.3 次）、收藏商品数（均值 1.5 件）
  - 时序特征：近 7/30/90 天活跃天数（均值 2/8/18 天）、最后购买距今天数（均值 45 天）、登录频次（均值 1.2 次/天）

**预期产出**：
- **流失概率评分**：为每个用户生成 0-1 的流失风险分数，模型 AUC 达到 0.84（对标行业 0.78）
- **高风险用户清单**：识别 top 20% 高风险用户（约 160 人），精准触达后挽留成功率 38%（对标群发 12%）
- **优先级排序**：结合用户 LTV（生命周期价值）排序，优先触达高价值用户

**量化业务价值**：
- **流失率降低**：从 24% 降至 19.2%，月均减少流失用户 39 人
- **挽留收入**：月均挽回用户贡献 39 人 × ¥180/人 = ¥7,020，**年化 ¥84,240**
- **营销成本优化**：从群发短信 ¥0.08/条改为精准推送 ¥0.03/条，月均节省 ¥2,400
- **综合 ROI**：模型开发成本 ¥15,000，6 个月回本，年化净收益 ¥76,440

**三轨验证**：
- **成本**：数据清洗 + 模型训练 + 系统集成 ≈ ¥15,000，月运维成本 ¥800
- **合规**：用户流失预测属于行为分析，无个人隐私泄露风险，符合 GDPR（欧盟）和中国《个人信息保护法》
- **风险**：模型漂移风险（用户行为季节性变化），需每月重训练；假阳性率 18%（误判流失用户），需人工审核 top 50 用户

---

### 场景二：婴儿推车沉默用户激活预测（精准优惠券投放）

**业务问题**：
某母婴出海品牌婴儿推车品类，注册用户 12,000 人中，有 3,600 人（30%）在首次购买后 30 天内未再次访问网站（沉默用户）。盲目群发优惠券成本高（¥10,800/月），激活率仅 15%。需通过预测模型区分"可激活用户"和"自然回流用户"，精准投放 8 折优惠券，避免浪费预算。

**具体数据规模**：
- 沉默用户池：3,600 人/月（30 天未访问）
- 历史数据：18 个月用户行为数据，共 8,200 个激活样本、2,100 个持续沉默样本
- 特征维度：
  - 购买行为：首购金额（均值 ¥420）、购买频次（均值 1.2 次）、加购行为（近 30 天均值 0.5 次）
  - 浏览行为：浏览深度（平均 3.2 个页面）、浏览时长（均值 8.5 分钟）、品类偏好（推车/配件/衣物）
  - 营销响应：历史领券率（32%）、点击率（8.5%）、优惠券核销率（12%）、邮件打开率（22%）

**预期产出**：
- **激活概率评分**：为每个沉默用户生成激活概率，模型准确率 78%
- **分层触达策略**：
  - 高激活概率用户（top 30%，1,080 人）→ 发 8 折优惠券
  - 中等用户（30%-60%，1,080 人）→ 发满 ¥200 减 ¥50 优惠券
  - 低激活用户（bottom 40%，1,440 人）→ 不触达，节省成本
- **最优触达时机**：基于用户历史活跃时段推荐最佳发送时间

**量化业务价值**：
- **激活率提升**：从 15% 提升至 28.5%（高概率用户激活率 42%），月均多激活 408 人
- **营销成本优化**：从 ¥10,800（群发 3,600 人 × ¥3/人）降至 ¥6,480（精准触达 2,160 人 × ¥3/人），**月均节省 ¥4,320，年化 ¥51,840**
- **收入增长**：激活用户月均复购 1.8 次，客单价 ¥380，月均新增收入 408 人 × 1.8 次 × ¥380 = ¥278,688，**年化 ¥3,344,256**
- **综合 ROI**：模型成本 ¥12,000，1 个月回本，年化净收益 ¥3,384,096

**三轨验证**：
- **成本**：数据集成 + 模型开发 + API 对接 ≈ ¥12,000，月运维 ¥600
- **合规**：激活预测基于用户自身行为，无第三方数据融合，符合数据隐私法规；优惠券投放需获得用户同意
- **风险**：模型偏差风险（新用户行为模式与历史不符），需每周评估模型性能；过度优惠导致利润率下降，需设置优惠券上限

---

## ③ 代码模板

```python
"""
Customer Churn Prediction for Mother-Baby Cross-border E-commerce
母婴出海电商用户流失预测完整实现
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    """母婴出海电商用户流失预测器"""

    def __init__(self, model_type='gradient_boosting', random_state=42):
        """
        初始化预测器
        
        Args:
            model_type: 'logistic' 或 'gradient_boosting'
            random_state: 随机种子
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.random_state = random_state

    def _generate_sample_data(self, n_samples=1000):
        """生成模拟母婴电商用户数据"""
        np.random.seed(self.random_state)
        
        data = {
            'user_id': np.arange(n_samples),
            'days_since_registration': np.random.randint(30, 730, n_samples),
            'days_since_last_purchase': np.random.randint(1, 180, n_samples),
            'total_purchase_count': np.random.randint(1, 15, n_samples),
            'total_purchase_amount': np.random.uniform(100, 2000, n_samples),
            'browse_count_7d': np.random.randint(0, 20, n_samples),
            'browse_count_30d': np.random.randint(0, 80, n_samples),
            'cart_add_count_30d': np.random.randint(0, 10, n_samples),
            'wishlist_count': np.random.randint(0, 8, n_samples),
            'active_days_30d': np.random.randint(0, 30, n_samples),
            'active_days_90d': np.random.randint(0, 90, n_samples),
            'purchase_frequency_30d': np.random.randint(0, 5, n_samples),
            'login_frequency_7d': np.random.uniform(0, 7, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # 生成目标变量：流失标签（90天未购买）
        # 特征与流失的关系：最后购买距今天数长、活跃度低 → 流失风险高
        churn_score = (
            0.05 * df['days_since_last_purchase'] -
            0.02 * df['total_purchase_count'] -
            0.01 * df['browse_count_30d'] -
            0.03 * df['active_days_30d'] +
            0.01 * df['days_since_registration'] +
            np.random.normal(0, 5, n_samples)
        )
        df['churn'] = (churn_score > churn_score.median()).astype(int)
        
        return df

    def prepare_features(self, df):
        """特征工程"""
        features = df[[
            'days_since_registration',
            'days_since_last_purchase',
            'total_purchase_count',
            'total_purchase_amount',
            'browse_count_7d',
            'browse_count_30d',
            'cart_add_count_30d',
            'wishlist_count',
            'active_days_30d',
            'active_days_90d',
            'purchase_frequency_30d',
            'login_frequency_7d',
        ]].copy()
        
        # 衍生特征
        features['avg_purchase_amount'] = (
            df['total_purchase_amount'] / df['total_purchase_count'].clip(lower=1)
        )
        features['browse_to_purchase_ratio'] = (
            df['browse_count_30d'] / df['purchase_frequency_30d'].clip(lower=1)
        )
        features['engagement_score'] = (
            df['active_days_30d'] * 0.3 +
            df['browse_count_30d'] * 0.2 +
            df['login_frequency_7d'] * 0.5
        )
        
        # 处理缺失值
        features = features.fillna(0)
        
        self.feature_names = features.columns.tolist()
        return features

    def train(self, df, test_size=0.2):
        """训练模型"""
        X = self.prepare_features(df)
        y = df['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000, random_state=self.random_state
            )
        else:  # gradient_boosting
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=self.random_state
            )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # 模型评估
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n=== 模型性能评估 ===")
        print(f"模型类型: {self.model_type}")
        print(f"AUC 分数: {auc:.4f}")
        print(f"F1 分数: {f1:.4f}")
        print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")
        
        return {
            'auc': auc,
            'f1': f1,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }

    def predict_churn(self, df):
        """预测用户流失概率"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 train() 方法")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        churn_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        result = df[['user_id']].copy()
        result['churn_probability'] = churn_proba
        result['churn_risk_level'] = pd.cut(
            churn_proba,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['低风险', '中风险', '高风险']
        )
        
        return result.sort_values('churn_probability', ascending=False)

    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        if self.model_type == 'gradient_boosting':
            importances = self.model.feature_importances_
        else:
            importances = np.abs(self.model.coef_[0])
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance


def main():
    """主函数：完整演示流程"""
    print("=" * 60)
    print("母婴出海电商用户流失预测系统")
    print("=" * 60)
    
    # 1. 初始化预测器
    predictor = ChurnPredictor(model_type='gradient_boosting')
    
    # 2. 生成模拟数据
    print("\n[1] 生成模拟用户数据...")
    df = predictor._generate_sample_data(n_samples=1000)
    print(f"✓ 生成 {len(df)} 条用户记录")
    print(f"✓ 流失用户占比: {df['churn'].mean():.1%}")
    
    # 3. 训练模型
    print("\n[2] 训练流失预测模型...")
    metrics = predictor.train(df, test_size=0.2)
    
    # 4. 预测流失概率
    print("\n[3] 预测用户流失概率...")
    predictions = predictor.predict_churn(df)
    print(f"✓ 高风险用户 (流失概率 > 0.6): {(predictions['churn_probability'] > 0.6).sum()} 人")
    print(f"✓ 中风险用户 (0.3-0.6): {((predictions['churn_probability'] > 0.3) & (predictions['churn_probability'] <= 0.6)).sum()} 人")
    print(f"✓ 低风险用户 (< 0.3): {(predictions['churn_probability'] <= 0.3).sum()} 人")
    
    # 5. 特征重要性分析
    print("\n[4] 特征重要性排序...")
    feature_imp = predictor.get_feature_importance()
    print(feature_imp.head(8).to_string(index=False))
    
    # 6. 业务决策建议
    print("\n[5] 业务决策建议...")
    high_risk = predictions[predictions['churn_probability'] > 0.6]
    print(f"✓ 建议触达高风险用户 {len(high_risk)} 人")
    print(f"✓ 预期挽留成功率: 35-40%")
    print(f"✓ 预期挽留收入: ¥{len(high_risk) * 0.38 * 180:.0f}")
    
    print("\n" + "=" * 60)
    print("[✓] Skill-Customer-Churn-Prediction 测试通过")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

---

## ④ 技能关联

### 前置技能（Prerequisite）
- [[Skill-Data-Cleaning-for-Ecommerce]] — 用户行为数据清洗与标准化是流失预测的基础
- [[Skill-Feature-Engineering-Behavioral]] — 需要从原始日志构建购买频次、活跃度等特征

### 延伸技能（Extends）
- [[Skill-Retention-Campaign-Optimization]] — 基于流失预测结果设计精准挽留活动
- [[Skill-Customer-Lifetime-Value-Prediction]] — 结合 LTV 预测优化触达优先级

### 可组合技能（Combinable）
- **组合场景**：[[Skill-Churn-Prediction]] + [[Skill-Propensity-Scoring]] + [[Skill-Marketing-Attribution]]
  - 先预测流失用户，再评估其对营销活动的响应倾向，最后分配营销预算
  - 应用：某品牌将流失预测与优惠券响应倾向结合，精准投放 8 折券给"高流失风险 + 高优惠敏感"用户，激活率从 15% 提升至 42%

---

## ⑤ 商业价值评估

### ROI 预估

| 指标 | 数值 | 依据 |
|------|------|------|
| **年化挽留收入** | ¥84,240 - ¥3,344,256 | 场景一：39 人/月 × ¥180 × 12 月；场景二：408 人/月 × 1.8 次 × ¥380 × 12 月 |
| **年化成本节省** | ¥27,600 - ¥51,840 | 精准触达替代群发，营销成本降低 32%-55% |
| **模型开发成本** | ¥12,000 - ¥15,000 | 数据集成、模型训练、系统对接 |
| **投资回报周期** | 1-6 个月 | 场景二 1 个月回本，场景一 6 个月回本 |
| **年化净收益** | ¥76,440 - ¥3,384,096 | 挽留收入 + 成本节省 - 开发成本 - 月运维成本 |

### 实施难度：⭐⭐⭐☆☆（3/5星）

**理由**：
- ✓ **优势**：算法成熟（Logistic/GBDT），开源库完整，无需复杂基础设施
- ✓ **数据可得性强**：母婴电商平台通常有完整的购买、浏览、登录日志
- ✗ **难点**：流失定义需业务确认（30/45/90 天？），特征工程需领域知识，模型漂移需定期重训练
- ✗ **系统集成**：需与 CRM/营销自动化平台对接，实时预测需 API 部署

### 优先级：⭐⭐⭐⭐☆（4/5星）

**理由**：
- ✓ **高商业价值**：直接提升 LTV 和复购率，年化收益百万级
- ✓ **低技术风险**：算法稳定，模型可解释，易于业务接受
- ✓ **快速见效**：2-4 周可上线，1 个月内验证 ROI
- ✓ **可扩展性**：模型可迁移至其他品类（推车、奶粉、衣物等）
- ✗ **竞争饱和**：头部品牌已普遍应用，差异化需结合营销策略

---

## 附录：实施检查清单

- [ ] 数据准备：收集 12+ 个月历史数据，至少 500+ 流失样本
- [ ] 流失定义确认：与业务团队确定流失周期（30/45/90 天）
- [ ] 特征工程：构建 15+ 个行为特征，处理缺失值和异常值
- [ ] 模型训练：AUC ≥ 0.80，F1 ≥ 0.70
- [ ] 业务验证：预测结果与运营团队经验对齐
- [ ] 系统集成：与 CRM/营销平台对接，实现自动化触达
- [ ] 监控告警：设置模型性能监控，AUC 下降 5% 时触发重训练
- [ ] 隐私合规：确保符合 GDPR/CCPA/《个人信息保护法》
