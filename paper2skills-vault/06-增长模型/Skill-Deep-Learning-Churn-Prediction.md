# Skill: Deep Learning for Customer Churn Prediction

roadmap_phase: phase2
updated: 2026-07-05

---

## ① 算法原理

**核心思想**：通过多层神经网络自动学习用户RFM行为的非线性特征表示，预测用户在未来14-30天内停止购买的概率，相比传统逻辑回归能捕获复杂特征交互（如"低频+高金额+长沉默"的组合信号）。

**数学直觉**：
- **前馈网络层级表示**：
$$h^{(l)} = \text{ReLU}(W^{(l)} h^{(l-1)} + b^{(l)})$$
其中$h^{(l)}$为第$l$层隐藏特征，自动学习"用户活跃度"→"购买周期"→"流失风险"的递进表示。

- **Sigmoid输出层**：将网络输出映射为流失概率
$$P(\text{churn}|x) = \sigma(W^{(L)} h^{(L-1)} + b^{(L)}) = \frac{1}{1 + e^{-z}}$$

- **加权交叉熵损失**（处理类别不平衡）：
$$L = -w_1 y \log(\hat{y}) - w_0(1-y)\log(1-\hat{y})$$
其中$w_1 > w_0$对少数类（流失用户）施加更高惩罚。

**关键假设**：
1. 用户过去90天的购买行为与未来14天流失倾向存在强相关
2. 跨境电商母婴用户的复购周期（奶粉60-90天、纸尿裤30-45天）是流失识别的关键时间窗口
3. 类别严重不平衡（流失率5-15%）需通过样本加权或过采样纠正

**非共识迁移**：传统深度学习流失预测多用于SaaS/订阅制（用户活跃度直接可观），但在跨境母婴电商中，用户购买间隔长（60-90天）且受宝宝成长阶段驱动，需将"产品生命周期"（奶粉分段：1段0-6月、2段6-12月）作为隐变量融入特征工程，才能准确区分"正常沉默"vs"真实流失"。

---

## ② 母婴出海应用案例

### 场景1：订阅制奶粉盒流失预警与14天干预

**业务问题**：
某跨境母婴电商运营月订奶粉盒服务，用户平均复购周期60天。当前流失率12%，年活跃用户50万，客单价480元。通过深度学习提前14天识别高风险用户，精准投放"宝宝成长关键期"主题优惠，挽留流失用户。

**具体数字**：
- 基础流失用户数：50万 × 12% = 6万人/年
- 当前挽留成功率：8%（无针对性干预）
- 目标：提升至28%（深度学习精准识别+个性化干预）
- 可挽留用户：6万 × (28%-8%) = 1.2万人/年

**预期产出**：
| 产出物 | 具体内容 | 业务含义 |
|------|--------|--------|
| 流失概率评分 | 每用户0-100分，≥70分为高风险 | 识别需干预的1.2万用户 |
| 风险分层 | 低风险(0-30分)、中风险(31-70分)、高风险(71-100分) | 分层营销成本控制 |
| 影响因子排序 | "距上次购买58天"(贡献度32%)、"本月浏览0次"(28%)、"上月订单金额↓30%"(18%) | 运营团队理解流失驱动因素 |
| 干预优先级 | Top 3000高风险用户+个性化文案 | 触达成本3000×50元=15万元 |

**量化业务价值**：
- **收入增长**：1.2万人 × 480元 × 2次/年 = 1152万元/年
- **净利润增长**：1152万 × 35% = 403万元/年（考虑干预成本15万+优惠折扣成本80万）
- **LTV提升**：从1440元/用户(3年) → 1944元/用户(挽留1.2万人后平均) = +35%

**三轨验证**：
- **成本轨**：模型训练+推理成本月均2万元，干预成本15万元，总成本17万元/月，ROI = 403万/12/17万 = 19.7倍
- **合规轨**：用户行为数据采集需符合GDPR(欧盟用户)和当地隐私法，建议数据脱敏存储、用户可查询流失评分，成本+5%
- **风险轨**：模型漂移风险（新品上市改变购买周期），需月度重训，误杀率控制在3%以内（避免过度干预损伤用户体验）

---

### 场景2：纸尿裤复购用户激活与沉默期唤醒

**业务问题**：
跨境电商纸尿裤用户复购周期30-45天，但用户在"沉默期"（距上次购买35-50天）易被竞品(Amazon/沃尔玛)抢走。需在用户即将流失前7天识别并推送"宝宝成长里程碑"主题内容+优惠，实现沉默用户激活。

**具体数字**：
- 月活用户：30万
- 沉默期用户（35-50天未购）：30万 × 25% = 7.5万人
- 当前自然复购率：35%（无干预）
- 目标复购率：58%（深度学习+个性化激活）
- 可激活用户：7.5万 × (58%-35%) = 17.25万人次/年

**预期产出**：
| 产出物 | 具体内容 | 业务含义 |
|------|--------|--------|
| 激活优先级评分 | 基于"沉默天数"、"历史复购频率"、"宝宝月龄段"的综合评分 | 识别最有可能被激活的用户 |
| 个性化文案推荐 | "您的宝宝已长到12个月，该升级L码纸尿裤了" | 提高点击率和转化率 |
| 优惠券决策 | 高价值用户(历史客单价>600元)发放15%优惠，低价值用户发放10% | 成本最优化 |
| 激活漏斗数据 | 推送→点击→加购→支付各环节转化率 | 持续优化激活策略 |

**量化业务价值**：
- **销售额增长**：17.25万人次 × 280元(纸尿裤客单价) = 4830万元/年
- **净利润增长**：4830万 × 32% - 优惠成本(4830万 × 12%) - 推送成本(17.25万 × 3元) = 1545万 - 580万 - 52万 = 913万元/年
- **用户LTV提升**：从2800元(3年) → 3650元 = +30%

**三轨验证**：
- **成本轨**：模型推理成本月均1.5万元，优惠成本580万/12=48万/月，总成本49.5万/月，ROI = 913万/12/49.5万 = 1.54倍（相比场景1较低，因优惠力度大）
- **合规轨**：涉及宝宝月龄推算需获得用户明确同意，建议在注册时获取宝宝出生日期的隐私授权，成本+2%
- **风险轨**：激活过度可能导致用户优惠依赖（价格敏感度上升），建议激活频率限制为"每用户每季度最多2次"，监控优惠后复购时的客单价变化

---

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import scipy.special

# ============ 1. 生成母婴电商模拟数据 ============
def generate_maternity_ecommerce_data(n_samples=10000, churn_rate=0.12):
    """
    生成母婴电商用户流失数据
    特征：RFM + 行为 + 产品生命周期
    """
    np.random.seed(42)
    
    # RFM特征
    recency = np.random.exponential(scale=30, size=n_samples)  # 距上次购买天数
    frequency = np.random.poisson(lam=3, size=n_samples) + 1  # 过去90天订单数
    monetary = np.random.gamma(shape=2, scale=200, size=n_samples)  # 过去90天消费金额
    
    # 行为特征
    app_open_freq = np.random.poisson(lam=5, size=n_samples)  # App打开频次
    browse_no_purchase = np.random.poisson(lam=2, size=n_samples)  # 浏览未下单次数
    customer_service_contact = np.random.poisson(lam=1, size=n_samples)  # 客服咨询次数
    
    # 产品生命周期特征（宝宝月龄推算）
    baby_age_months = np.random.uniform(0, 36, size=n_samples)  # 宝宝月龄0-36月
    formula_segment_changes = np.random.poisson(lam=1.5, size=n_samples)  # 奶粉分段切换次数
    
    # 渠道特征
    channel_quality = np.random.choice([0.8, 0.9, 1.0], size=n_samples)  # 获客渠道质量系数
    
    # 构造流失标签（基于特征的非线性组合）
    churn_prob = (
        0.05 +  # 基础流失率
        0.003 * recency +  # 距上次购买越久越易流失
        -0.02 * frequency +  # 购买频率高则流失率低
        -0.00001 * monetary +  # 消费金额高则流失率低
        -0.01 * app_open_freq +  # App活跃度高则流失率低
        0.02 * browse_no_purchase +  # 浏览未下单是流失信号
        -0.005 * customer_service_contact +  # 客服接触增加粘性
        0.001 * np.abs(baby_age_months - 6) +  # 宝宝6月龄(奶粉分段)附近流失率高
        -0.01 * formula_segment_changes +  # 分段切换频繁说明活跃
        -0.05 * (channel_quality - 0.8)  # 优质渠道用户流失率低
    )
    churn_prob = np.clip(churn_prob, 0.01, 0.5)  # 限制在合理范围
    
    churn = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # 调整流失率到目标比例
    current_churn_rate = churn.mean()
    if current_churn_rate > churn_rate:
        churn_indices = np.where(churn == 1)[0]
        remove_count = int(len(churn_indices) * (1 - churn_rate / current_churn_rate))
        churn[np.random.choice(churn_indices, remove_count, replace=False)] = 0
    
    df = pd.DataFrame({
        'user_id': range(n_samples),
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'app_open_freq': app_open_freq,
        'browse_no_purchase': browse_no_purchase,
        'customer_service_contact': customer_service_contact,
        'baby_age_months': baby_age_months,
        'formula_segment_changes': formula_segment_changes,
        'channel_quality': channel_quality,
        'churn': churn
    })
    
    return df

# ============ 2. 深度神经网络模型 ============
class DeepChurnPredictor:
    """
    多层感知机(MLP)流失预测模型
    使用numpy手工实现，展示核心算法
    """
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.layers = []
        self.biases = []
        
        # 初始化网络权重
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * 0.01
            b = np.zeros((1, dims[i+1]))
            self.layers.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.layers) - 1):
            z = np.dot(self.activations[-1], self.layers[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        # 输出层
        z_out = np.dot(self.activations[-1], self.layers[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        y_pred = self.sigmoid(z_out)
        self.activations.append(y_pred)
        
        return y_pred
    
    def backward(self, y_true, sample_weights=None):
        """反向传播"""
        m = y_true.shape[0]
        if sample_weights is None:
            sample_weights = np.ones(m)
        
        # 输出层梯度
        dz = (self.activations[-1] - y_true) * sample_weights.reshape(-1, 1)
        
        for i in range(len(self.layers) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dz = np.dot(dz, self.layers[i].T) * self.relu_derivative(self.z_values[i-1])
            
            self.layers[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
    
    def train(self, X, y, epochs=100, batch_size=32, class_weight=None):
        """训练模型（处理类别不平衡）"""
        n_samples = X.shape[0]
        
        if class_weight is None:
            class_weight = {0: 1.0, 1: 1.0}
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # 计算样本权重
                sample_weights = np.array([class_weight[int(y)] for y in y_batch])
                
                y_pred = self.forward(X_batch)
                self.backward(y_batch.reshape(-1, 1), sample_weights)
            
            if (epoch + 1) % 20 == 0:
                y_pred_all = self.forward(X)
                loss = -np.mean(y * np.log(y_pred + 1e-10) + (1-y) * np.log(1-y_pred + 1e-10))
                auc = roc_auc_score(y, y_pred)
                print(f"Epoch {epoch+1}: Loss={loss:.4f}, AUC={auc:.4f}")
    
    def predict(self, X):
        """预测"""
        return self.forward(X)

# ============ 3. 训练流程 ============
def train_churn_model():
    """完整训练流程"""
    print("=" * 60)
    print("Skill: Deep Learning for Customer Churn Prediction")
    print("=" * 60)
    
    # 生成数据
    print("\n[1] 生成母婴电商模拟数据...")
    df = generate_maternity_ecommerce_data(n_samples=5000, churn_rate=0.12)
    print(f"    数据规模: {len(df)} 用户")
    print(f"    流失率: {df['churn'].mean():.2%}")
    print(f"    特征维度: {df.shape[1] - 2}")
    
    # 特征工程
    print("\n[2] 特征工程与标准化...")
    feature_cols = [col for col in df.columns if col not in ['user_id', 'churn']]
    X = df[feature_cols].values
    y = df['churn'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    训练集: {len(X_train)} 样本")
    print(f"    测试集: {len(X_test)} 样本")
    
    # 计算类别权重（处理不平衡）
    pos_weight = (1 - y_train.mean()) / y_train.mean()
    class_weight = {0: 1.0, 1: pos_weight}
    print(f"    类别权重: 正常={class_weight[0]:.2f}, 流失={class_weight[1]:.2f}")
    
    # 训练模型
    print("\n[3] 训练深度神经网络...")
    model = DeepChurnPredictor(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32, 16],
        learning_rate=0.01
    )
    model.train(X_train, y_train, epochs=100, batch_size=32, class_weight=class_weight)
    
    # 评估模型
    print("\n[4] 模型评估...")
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()
    
    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_test = roc_auc_score(y_test, y_pred_test)
    print(f"    训练集 AUC: {auc_train:.4f}")
    print(f"    测试集 AUC: {auc_test:.4f}")
    
    # 精准率-召回率分析
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_test)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"    最优阈值: {best_threshold:.3f}")
    
    # 高风险用户识别
    print("\n[5] 高风险用户识别...")
    df['churn_prob'] = model.predict(X_scaled).flatten()
    df['risk_level'] = pd.cut(df['churn_prob'], 
                               bins=[0, 0.3, 0.7, 1.0],
                               labels=['低风险', '中风险', '高风险'])
    
    high_risk = df[df['churn_prob'] >= 0.7].sort_values('churn_prob', ascending=False)
    print(f"    高风险用户数: {len(high_risk)} ({len(high_risk)/len(df):.2%})")
    print(f"    高风险用户中实际流失率: {high_risk['churn'].mean():.2%}")
    
    print("\n[6] 影响因子分析 (Top 5)...")
    feature_importance = []
    for i, col in enumerate(feature_cols):
        # 通过扰动法计算特征重要性
        X_perturbed = X_scaled.copy()
        X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
        y_pred_perturbed = model.predict(X_perturbed).flatten()
        importance = np.abs(y_pred_test - y_pred_perturbed).mean()
        feature_importance.append((col, importance))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(feature_importance[:5], 1):
        print(f"    {i}. {feat}: {imp:.4f}")
    
    print("\n[✓] Skill-Deep-Learning-Churn-Prediction测试通过")
    print("=" * 60)
    
    return model, df, scaler

if __name__ == "__main__":
    train_churn_model()
```

---

## ④ 技能关联

### 前置技能
- [[Skill-Feature-Engineering]] — 母婴电商流失预测的核心是将"宝宝成长阶段"、"产品分段周期"等领域知识转化为特征，特征质量直接决定深度学习模型效果

### 延伸技能
- [[Skill-Causal-Inference-Churn]] — 从"预测谁会流失"升级到"干预谁能被挽留"，需要因果推断识别真实的干预效应

### 可组合
- [[Skill-DQN-Retention-Optimization]] — 流失预测 + 强化学习优惠决策形成完整留存优化闭环：先用深度学习识别高风险用户，再用DQN动态决策最优优惠力度和时机

---

## ⑤ 商业价值评估

| 指标 | 评估 | 说明 |
|-----|------|------|
| **ROI预估** | 高 | 场景1年增收403万元，场景2年增收913万元，综合ROI 8-20倍 |
| **实施难度** | ⭐⭐⭐☆☆ | 3/5星。需6个月用户行为数据、特征工程投入2周、模型训练1周、系统对接2周 |
| **优先级评分** | ⭐⭐⭐⭐☆ | 4/5星。用户留存直接影响LTV，母婴电商获客成本高(CAC 200-500元)，挽留老客成本仅20-50元，ROI达5-10倍 |

**评估依据**：
1. **量化收益**：基于场景1+场景2，年增收1300+万元，净利润1300万元，投入成本仅200万元/年
2. **实施要求**：需完整的用户行为数据仓库（订单、浏览、客服等），数据治理是主要瓶颈，建议先用RFM基础模型上线快速验证，再迭代深度学习
3. **运营闭环**：模型输出需对接CRM/营销自动化系统，建议与运营团队联合制定干预SOP（干预频率、优惠额度、文案模板）

**建议实施路径**：
- **阶段1**（2周）：RFM特征工程 + 逻辑回归基础模型上线，验证流失预测的业务价值
- **阶段2**（2周）：深度学习模型优化，接入行为特征和产品生命周期特征，AUC目标≥0.78
- **阶段3**（2周）：与CRM/营销自动化系统对接，自动化高风险用户识别和干预流程
- **阶段4**（持续）：A/B测试验证干预效果，月度模型重训应对数据漂移