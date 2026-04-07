# Skill Card: LTV预测 - 零膨胀对数正态模型 (ZILN)

---

## ① 算法原理

### 核心思想
LTV (Customer Lifetime Value, 客户生命周期价值) 预测是增长模型的核心能力。传统回归方法难以处理 LTV 分布的两个典型特征：**零膨胀**（大量用户只购买一次）和**长尾分布**（少数高价值用户贡献大部分收入）。

ZILN (Zero-Inflated Lognormal) 模型通过**联合建模流失概率和购买金额**，有效解决了这两个挑战。它假设：
1. 用户以概率 $p$ 在首次购买后流失（成为"一次性购买者"）
2. 未流失用户的消费金额服从对数正态分布

### 数学直觉

**零膨胀对数正态分布**：

对于用户 $i$ 的 LTV $Y_i$：
$$
P(Y_i = 0) = p_i \\
P(Y_i = y | Y_i > 0) = \text{Lognormal}(y; \mu_i, \sigma_i^2)
$$

**联合损失函数**：

ZILN 模型同时优化两个目标：

$$
\mathcal{L} = \underbrace{-\sum_{i} [y_i > 0] \cdot \log(1 - p_i)}_{\text{留存概率损失}} + \underbrace{\sum_{i} [y_i > 0] \cdot \left[ \frac{(\log y_i - \mu_i)^2}{2\sigma^2} + \log(y_i \sigma \sqrt{2\pi}) \right]}_{\text{对数正态损失}}
$$

其中：
- $p_i$：用户 $i$ 成为一次性购买者的概率（通过 sigmoid 输出）
- $\mu_i, \sigma_i$：对数消费金额的均值和标准差（通过网络预测）
- $[\cdot]$：指示函数

**期望 LTV 计算**：

对于预测，ZILN 计算期望 LTV：
$$
E[Y_i] = (1 - p_i) \cdot \exp\left(\mu_i + \frac{\sigma_i^2}{2}\right)
$$

### 网络架构

```
输入特征 (用户行为、 demographics)
    ↓
共享神经网络层 (Dense + ReLU)
    ↓
┌──────────────┬──────────────┬──────────────┐
↓              ↓              ↓
Churn Head    μ Head         σ Head
(Sigmoid)     (Linear)       (Softplus)
    ↓              ↓              ↓
   p_i           μ_i            σ_i
    └──────────────┴──────────────┘
                   ↓
              期望 LTV
```

### 关键优势

| 挑战 | 传统方法 | ZILN 方案 |
|------|---------|----------|
| 零膨胀 | MSE 损失会被大量 0 值主导 | 分离建模流失和金额 |
| 长尾分布 | 均值回归对高值用户欠拟合 | 对数正态自然建模右偏 |
| 不确定性 | 仅输出点估计 | 同时输出均值和方差 |

---

## ② 母婴出海应用案例

### 场景一：吸奶器新客 LTV 预测

**业务问题**：
我们通过 Facebook/TikTok 广告获取了大量北美新客，但并非所有新客都有长期价值。部分用户只购买一次基础款吸奶器就流失了，而另一部分会复购配件、升级高端款，LTV 可达初购金额的 3-5 倍。我们需要在**首次购买时就预测用户的 LTV**，以决策是否值得继续投放广告维护关系。

**数据要求**：
- 用户画像：年龄、是否新手妈妈、收入水平、地域
- 首购行为：购买产品 SKU、客单价、是否使用优惠券
- 行为特征：注册到首购间隔、浏览页面数、加购次数
- 渠道特征：获客渠道、广告素材、落地页类型
- 标签：历史 LTV（6 个月/12 个月）、是否复购
- 数据量：建议至少 5,000+ 有完整 LTV 历史的用户

**预期产出**：
- **LTV 点估计**：每个新客的预测生命周期价值
- **流失概率**：该用户成为"一次性购买者"的概率
- **价值分层**：
  - 高 LTV 潜力（Top 20%）：重点运营，推送会员计划
  - 中 LTV 潜力（中间 50%）：标准运营，定期触达
  - 低 LTV 潜力（Bottom 30%）：降低维护成本

**业务价值**：
- 新客获取成本（CAC）约 $25-40，优化后预计：
  - 识别高 LTV 用户，针对性投放，ROI 提升 30-50%
  - 减少低 LTV 用户的过度营销，节省成本 20-30%
  - 精准分层运营，整体营销效率提升 25%

---

### 场景二：会员等级智能划分

**业务问题**：
我们的吸奶器品牌有一个会员体系（普通/银卡/金卡/钻石），但当前等级仅基于历史消费金额划分，属于"事后诸葛亮"。我们希望**基于预测的 LTV 进行等级划分**，让高潜力新客一加入就享受更好的服务，提升留存。

**数据要求**：
- 用户特征：同场景一
- 交互特征：客服咨询次数、App 使用频次、内容互动
- 社交特征：是否关注社交媒体、是否参与社群
- 标签：12 个月 LTV、会员活跃度
- 数据量：建议 3,000+ 用户

**预期产出**：
- **预测 LTV 等级**：新客注册时即预测其未来价值
- **升级概率**：该用户从普通会员升级到高等级会员的概率
- **动态等级调整**：基于实际行为与预测的偏差，动态调整等级

**业务价值**：
- 高潜力用户早期识别率提升 40%
- 会员满意度提升（高潜力用户获得更好体验）
- 会员体系运营成本优化，ROI 提升 20%

---

## ③ 代码模板

```python
"""
LTV预测 - 零膨胀对数正态模型 (ZILN)
用于母婴出海电商新客价值预测和会员等级划分
基于 Google Research 的 ZILN 实现
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ZILNModel(nn.Module):
    """
    Zero-Inflated Lognormal 神经网络模型
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        """
        初始化 ZILN 模型

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout 率
        """
        super(ZILNModel, self).__init__()

        # 构建共享网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # 三个输出头
        # 1. 流失概率 (churn probability)
        self.churn_head = nn.Linear(prev_dim, 1)

        # 2. 对数均值 (mu)
        self.mu_head = nn.Linear(prev_dim, 1)

        # 3. 对数标准差 (sigma) - 使用 softplus 保证正值
        self.sigma_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        """
        前向传播

        Returns:
            churn_prob: 流失概率 (batch_size, 1)
            mu: 对数均值 (batch_size, 1)
            sigma: 对数标准差 (batch_size, 1)
        """
        shared = self.shared_layers(x)

        churn_logits = self.churn_head(shared)
        churn_prob = torch.sigmoid(churn_logits)

        mu = self.mu_head(shared)
        sigma = torch.nn.functional.softplus(self.sigma_head(shared)) + 1e-6

        return churn_prob, mu, sigma

    def predict_ltv(self, x):
        """
        预测期望 LTV

        Returns:
            expected_ltv: 期望 LTV 值
        """
        churn_prob, mu, sigma = self.forward(x)

        # E[Y] = (1 - p) * exp(mu + sigma^2 / 2)
        expected_log_ltv = mu + 0.5 * (sigma ** 2)
        expected_ltv = (1 - churn_prob) * torch.exp(expected_log_ltv)

        return expected_ltv.squeeze()


class ZILNTrainer:
    """
    ZILN 模型训练器
    """

    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'val_loss': []}

    def ziln_loss(self, y_true, churn_prob, mu, sigma, eps=1e-6):
        """
        ZILN 损失函数

        Args:
            y_true: 真实 LTV 值 (batch_size,)
            churn_prob: 预测流失概率 (batch_size, 1)
            mu: 预测对数均值 (batch_size, 1)
            sigma: 预测对数标准差 (batch_size, 1)
        """
        y_true = y_true.unsqueeze(1) if len(y_true.shape) == 1 else y_true

        # 零值掩码
        positive_mask = (y_true > 0).float()

        # 1. 零膨胀部分损失: -log(p) for y=0, -log(1-p) for y>0
        # 使用数值稳定性技巧
        churn_prob = torch.clamp(churn_prob, eps, 1 - eps)
        zero_inflated_loss = -(
            (1 - positive_mask) * torch.log(churn_prob + eps) +
            positive_mask * torch.log(1 - churn_prob + eps)
        )

        # 2. 对数正态部分损失 (仅对 y > 0 计算)
        log_y = torch.log(y_true + eps)

        # 对数正态密度函数
        lognormal_loss = positive_mask * (
            torch.log(sigma + eps) +
            0.5 * ((log_y - mu) / (sigma + eps)) ** 2 +
            log_y  # Jacobian adjustment
        )

        # 总损失
        total_loss = torch.mean(zero_inflated_loss + lognormal_loss)

        return total_loss

    def train_epoch(self, dataloader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            churn_prob, mu, sigma = self.model(batch_x)
            loss = self.ziln_loss(batch_y, churn_prob, mu, sigma)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                churn_prob, mu, sigma = self.model(batch_x)
                loss = self.ziln_loss(batch_y, churn_prob, mu, sigma)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader, epochs=100, patience=10):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break

        return self


# ==================== 母婴出海业务专用函数 ====================

def generate_ltv_data(n_samples=5000, random_state=42):
    """
    生成母婴出海电商 LTV 模拟数据

    场景：吸奶器新客 LTV 预测
    """
    np.random.seed(random_state)

    # 用户画像特征
    age = np.random.normal(31, 5, n_samples)
    age = np.clip(age, 22, 45)

    is_first_time = np.random.binomial(1, 0.65, n_samples)
    income_level = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.35, 0.35, 0.1])

    # 首购行为特征
    first_order_value = np.random.lognormal(4.2, 0.4, n_samples)  # 约 $67 均值
    days_to_first_order = np.random.exponential(3, n_samples)
    pages_viewed = np.random.poisson(8, n_samples)
    used_coupon = np.random.binomial(1, 0.4, n_samples)

    # 渠道特征
    channel = np.random.choice(['facebook', 'tiktok', 'google', 'organic'], n_samples,
                               p=[0.4, 0.3, 0.2, 0.1])

    X = pd.DataFrame({
        'age': age,
        'is_first_time': is_first_time,
        'income_level': income_level,
        'first_order_value': first_order_value,
        'days_to_first_order': days_to_first_order,
        'pages_viewed': pages_viewed,
        'used_coupon': used_coupon,
        'channel_facebook': (channel == 'facebook').astype(int),
        'channel_tiktok': (channel == 'tiktok').astype(int),
        'channel_google': (channel == 'google').astype(int)
    })

    # 生成 LTV (零膨胀 + 对数正态)
    # 1. 流失概率 (哪些用户会成为一次性购买者)
    churn_logit = (
        -1.5 +
        0.5 * (1 - is_first_time) +  # 非新手妈妈更容易复购
        0.3 * (income_level - 2.5) +  # 高收入更可能复购
        0.02 * (first_order_value - 60) +  # 首购金额高更可能复购
        0.1 * (pages_viewed - 8) +  # 浏览多更可能复购
        0.2 * used_coupon  # 用优惠券的更容易流失（价格敏感）
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    is_churned = np.random.binomial(1, churn_prob)

    # 2. 对于未流失用户，生成对数正态的 LTV
    mu = (
        3.5 +
        0.2 * (income_level - 2.5) +
        0.01 * (first_order_value - 60) +
        0.05 * is_first_time
    )
    sigma = 0.6

    ltv = np.zeros(n_samples)
    non_churned = ~is_churned.astype(bool)
    ltv[non_churned] = np.random.lognormal(mu[non_churned], sigma, non_churned.sum())

    return X, ltv


def calculate_ltv_metrics(y_true, y_pred):
    """
    计算 LTV 预测评估指标
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # 基础指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 归一化 Gini 系数 (衡量排序能力)
    def normalized_gini(y_true, y_pred):
        """计算归一化 Gini 系数"""
        def gini(actual, pred):
            assert len(actual) == len(pred)
            all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
            all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
            totalLosses = all[:, 0].sum()
            giniSum = all[:, 0].cumsum().sum() / totalLosses
            giniSum -= (len(actual) + 1) / 2.
            return giniSum / len(actual)

        return gini(y_true, y_pred) / gini(y_true, y_true)

    gini_coef = normalized_gini(y_true, y_pred)

    # Top 10% 捕获率
    top_10_pct = int(0.1 * len(y_true))
    top_indices_true = np.argsort(y_true)[-top_10_pct:]
    top_indices_pred = np.argsort(y_pred)[-top_10_pct:]
    capture_rate = len(set(top_indices_true) & set(top_indices_pred)) / top_10_pct

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Normalized_Gini': gini_coef,
        'Top10_Capture_Rate': capture_rate
    }


def analyze_ltv_segments(y_true, y_pred, n_segments=5):
    """
    分析 LTV 分层效果
    """
    # 按预测 LTV 分箱
    segments = pd.qcut(y_pred, q=n_segments, labels=['Low', 'D-Low', 'Mid', 'D-High', 'High'])

    analysis = []
    for seg in ['Low', 'D-Low', 'Mid', 'D-High', 'High']:
        mask = segments == seg
        if mask.sum() > 0:
            analysis.append({
                'Segment': seg,
                'Count': mask.sum(),
                'Pred_LTV_Mean': y_pred[mask].mean(),
                'True_LTV_Mean': y_true[mask].mean(),
                'True_LTV_Sum': y_true[mask].sum()
            })

    return pd.DataFrame(analysis)


def main():
    """主函数：演示 ZILN 在母婴 LTV 预测中的应用"""
    print("=" * 70)
    print("母婴出海 - ZILN LTV 预测")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    X, ltv = generate_ltv_data(n_samples=5000)
    print(f"   总样本: {len(X)}")
    print(f"   零 LTV 用户: {(ltv == 0).sum()} ({(ltv == 0).mean()*100:.1f}%)")
    print(f"   平均 LTV: ${ltv.mean():.2f}")
    print(f"   LTV 中位数: ${np.median(ltv):.2f}")
    print(f"   LTV 最大值: ${ltv.max():.2f}")

    # 2. 数据预处理
    print("\n[2] 数据预处理...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, ltv, test_size=0.2, random_state=42
    )

    # 创建 PyTorch 数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # 3. 训练 ZILN 模型
    print("\n[3] 训练 ZILN 模型...")
    model = ZILNModel(input_dim=X.shape[1], hidden_dims=[64, 32], dropout_rate=0.2)
    trainer = ZILNTrainer(model, learning_rate=0.001)
    trainer.fit(train_loader, test_loader, epochs=100, patience=15)

    # 4. 预测
    print("\n[4] 预测 LTV...")
    model.eval()
    with torch.no_grad():
        y_pred = model.predict_ltv(torch.FloatTensor(X_test)).numpy()

    # 5. 评估指标
    print("\n[5] 模型评估...")
    metrics = calculate_ltv_metrics(y_test, y_pred)
    print(f"   MAE: ${metrics['MAE']:.2f}")
    print(f"   RMSE: ${metrics['RMSE']:.2f}")
    print(f"   归一化 Gini: {metrics['Normalized_Gini']:.3f}")
    print(f"   Top10% 捕获率: {metrics['Top10_Capture_Rate']*100:.1f}%")

    # 6. 分层分析
    print("\n[6] LTV 分层分析...")
    segment_analysis = analyze_ltv_segments(y_test, y_pred)
    print(segment_analysis.to_string(index=False))

    # 7. 业务建议
    print("\n[7] 业务建议...")
    print("   - High 分群: 预测高 LTV，投入高成本维护（VIP 服务、专属优惠）")
    print("   - D-High 分群: 预测中高 LTV，标准维护（定期邮件、会员权益）")
    print("   - Mid 分群: 预测中等 LTV，低成本触达（自动化营销）")
    print("   - D-Low/Low 分群: 预测低 LTV，降低维护成本（减少投放）")

    # 计算潜在收益
    high_value_count = (y_pred > np.percentile(y_pred, 80)).sum()
    high_value_ltv = y_test[y_pred > np.percentile(y_pred, 80)].mean()
    print(f"\n   高价值用户识别: {high_value_count} 人")
    print(f"   高价值用户平均 LTV: ${high_value_ltv:.2f}")
    print(f"   是整体平均的 {high_value_ltv / y_test.mean():.1f} 倍")

    print("\n" + "=" * 70)
    print("LTV 预测分析完成！")
    print("=" * 70)

    return model, scaler


if __name__ == '__main__':
    model, scaler = main()
```

---

## ④ 技能关联

### 前置技能
- **机器学习基础**：理解神经网络和损失函数
- **概率分布**：理解对数正态分布和零膨胀模型
- **用户增长分析**：熟悉 CAC、LTV、留存率等指标

### 延伸技能
- **HT-GNN (超图神经网络)**：结合图神经网络建模用户关系
- **不确定性估计**：使用 Monte Carlo Dropout 量化预测不确定性
- **动态 LTV 预测**：基于用户行为序列进行实时 LTV 更新

### 可组合技能
- **智能归因 (Causal Forest)**：识别高 LTV 用户的获客渠道
- **冷启动推荐**：针对高 LTV 潜力用户进行个性化推荐
- **优惠券优化**：基于 LTV 预测决定优惠券发放策略

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 新客 LTV 预测 | 营销 ROI 提升 30-50%，年增收 100-200 万 | 开发 2-3 周 | 15-25x |
| 会员等级智能划分 | 会员满意度提升，留存率 +10% | 开发 1-2 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要 6-12 个月的用户行为数据
- 技术门槛：中等，需理解概率分布和神经网络
- 工程复杂度：中，PyTorch 实现直观
- 维护成本：中，需定期重新训练模型

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- 业务价值极高：LTV 是增长模型的核心指标
- 方法成熟：Google Research 开源，业界广泛验证
- 效果可量化：直接关联营销 ROI
- 可扩展性强：可延伸至动态预测、不确定性估计

### 关键优势
与简单回归模型相比，ZILN 的**独特价值**：
1. **零值处理**：准确建模"一次性购买者"，不被大量 0 值干扰
2. **长尾建模**：对数正态分布自然拟合高价值用户的右偏分布
3. **双重输出**：同时预测"是否复购"和"复购金额"

---

## 参考论文

- **A Deep Probabilistic Model for Customer Lifetime Value Prediction**: [arXiv:1912.07753](https://arxiv.org/abs/1912.07753) (Google Research, 2019)
- **HT-GNN: Hyper-Temporal Graph Neural Network for LTV Prediction**: [arXiv:2601.13013](https://arxiv.org/abs/2601.13013) (Baidu, 2026)
- **Customer Lifetime Value Prediction with Uncertainty Estimation**: [arXiv:2411.15944](https://arxiv.org/abs/2411.15944) (2024)

## 开源资源

- **Google lifetime_value**: [github.com/google/lifetime_value](https://github.com/google/lifetime_value)
- **TensorFlow 实现**: 支持 ZILN 损失的完整实现和评估指标
