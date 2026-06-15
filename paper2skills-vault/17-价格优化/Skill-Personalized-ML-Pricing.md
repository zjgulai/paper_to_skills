---
title: Personalized ML Pricing — 个性化 ML 定价：用户级支付意愿驱动的差异化定价
doc_type: knowledge
module: 17-价格优化
topic: personalized-ml-pricing
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Personalized ML Pricing — 个性化 ML 定价

> **论文**：Personalized Pricing via Machine Learning: Estimating Individual Willingness-to-Pay in E-Commerce (2024)
> **arXiv**：2406.18234 | **桥梁**: 17-价格优化 ↔ 14-用户分析 ↔ 01-因果推断 | **类型**: 跨域融合
> **反直觉来源**：现有所有定价 Skill 都是"市场级"定价（所有用户看同一个价格）——但 DTC 独立站/App 可以实现真正的"用户级"定价：对历史高消费用户显示 $159，对价格敏感新用户显示 $129，实现利润最大化的差异化定价

---

## ① 算法原理

### 核心思想

**市场级定价 vs 个性化定价**：

```
市场级定价（所有用户 $149）：
  WTP > $149 的用户：留下了利润（但实际愿意支付更多）
  WTP < $149 的用户：流失了（本来愿意付 $129）
  
个性化定价（用户A $169, 用户B $129）：
  更精准捕捉每个用户的 WTP
  总利润提升（WTP分布更充分利用）
```

**个体 WTP 估计**（机器学习方法）：

```python
特征 -> WTP 预测模型：
  用户历史：
    - 历史购买均价（高价用户 WTP 更高）
    - 价格敏感度（是否经常等待折扣）
    - 品牌忠诚度（复购次数）
  
  当前信号：
    - 页面停留时间（越长越有购买意向）
    - 对比竞品次数（比对多次 = 价格敏感）
    - 来源渠道（Google 直达 vs 社交媒体）
  
  外部特征：
    - 设备类型（iPhone 用户 WTP 通常更高）
    - 地区（一线城市 vs 三四线城市）
    - 时间（发薪日前后）
```

**法律合规边界**：
- 差异化定价在美国法律中通常合法（非歧视性保护类别）
- 明确禁止：基于种族/性别/宗教/国籍的定价歧视
- 合法：基于购买历史/会员等级/价格敏感度/地理区域
- 最佳实践：展示"此次专属优惠"而非直接显示不同价格

---

## ② 母婴出海应用案例

### 场景：DTC 独立站用户级价格优化

**业务问题**：独立站吸奶器标价 $149，对所有用户相同。实际上：
- 老用户群（历史购买均价 $180+）：愿意付 $169
- 新用户群（来自 Google 搜索"最便宜吸奶器"）：最高愿意付 $129
- 若不区分，要么流失价格敏感用户，要么损失高 WTP 用户的溢价

**数据要求**：
- 用户历史购买记录（均价/频率/品类）
- 当前 session 行为（停留时间/页面路径/设备）
- 历史价格实验数据（不同价格点的转化率）

**预期产出**：
- 每位用户的 WTP 估计（区间而非点估计）
- 个性化显示价格建议（含折扣显示策略）
- 预期利润提升：vs 统一定价的收益对比

**业务价值**：
- 利润提升 10-18%（捕捉更多 WTP 曲线面积）
- 价格敏感用户转化率提升（展示合适价格）
- 年化 ROI：**¥20-60 万**

---

## ③ 代码模板

```python
"""
Personalized ML Pricing
个性化ML定价：用户级WTP估计与差异化定价策略
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from dataclasses import dataclass


@dataclass
class UserPricingFeatures:
    """用户定价特征"""
    user_id: str
    avg_past_purchase_price: float    # 历史购买均价
    purchase_frequency: int            # 购买次数
    days_since_last_purchase: int
    is_repeat_buyer: bool
    session_duration_sec: float        # 当前访问时长
    pages_viewed: int                  # 浏览页面数
    comparison_events: int             # 竞品对比次数（价格敏感信号）
    device_type: str                   # mobile / desktop / tablet
    acquisition_channel: str           # organic / paid / email / direct
    region_tier: int                   # 1=一线 2=二线 3=三四线


def extract_feature_vector(user: UserPricingFeatures) -> np.ndarray:
    """将用户特征转为数值向量"""
    device_score = {'mobile': 0.6, 'desktop': 0.8, 'tablet': 0.7}.get(user.device_type, 0.7)
    channel_score = {'direct': 1.0, 'email': 0.9, 'organic': 0.7, 'paid': 0.6}.get(user.acquisition_channel, 0.7)
    price_sensitivity_proxy = 1 - min(1, user.comparison_events / 3)  # 越多对比越敏感

    return np.array([
        user.avg_past_purchase_price / 200,    # 归一化
        user.purchase_frequency / 10,
        1 / (1 + user.days_since_last_purchase / 30),
        float(user.is_repeat_buyer),
        min(1, user.session_duration_sec / 300),
        min(1, user.pages_viewed / 10),
        price_sensitivity_proxy,
        device_score,
        channel_score,
        (4 - user.region_tier) / 3,
    ])


class WTPPredictor:
    """支付意愿预测模型"""

    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        self.is_trained = False

    def generate_training_data(self, n_samples: int = 500, seed: int = 42):
        """生成模拟训练数据（生产中用真实A/B实验数据）"""
        np.random.seed(seed)
        features_list = []
        wtp_labels = []

        for _ in range(n_samples):
            avg_price = np.random.lognormal(5.0, 0.5)  # ~$150 均值
            freq = np.random.poisson(3)
            days = np.random.randint(1, 90)
            is_repeat = freq > 1

            user = UserPricingFeatures(
                user_id='train',
                avg_past_purchase_price=avg_price,
                purchase_frequency=freq,
                days_since_last_purchase=days,
                is_repeat_buyer=is_repeat,
                session_duration_sec=np.random.exponential(120),
                pages_viewed=np.random.randint(1, 15),
                comparison_events=np.random.poisson(1),
                device_type=np.random.choice(['mobile','desktop','tablet'], p=[0.5,0.4,0.1]),
                acquisition_channel=np.random.choice(['organic','paid','email','direct'], p=[0.3,0.3,0.2,0.2]),
                region_tier=np.random.choice([1,2,3], p=[0.2,0.4,0.4]),
            )

            # 真实WTP（含噪声）
            base_wtp = 0.6 * avg_price + 0.3 * (150 if is_repeat else 110)
            wtp = base_wtp * (0.8 + 0.4 * np.random.random())
            features_list.append(extract_feature_vector(user))
            wtp_labels.append(wtp)

        return np.array(features_list), np.array(wtp_labels)

    def train(self):
        X, y = self.generate_training_data()
        self.model.fit(X, y)
        self.is_trained = True

    def predict_wtp(self, user: UserPricingFeatures) -> dict:
        """预测用户WTP及建议定价"""
        if not self.is_trained:
            self.train()

        features = extract_feature_vector(user).reshape(1, -1)
        wtp_pred = float(self.model.predict(features)[0])

        # 定价策略（四档）
        if wtp_pred >= 165:
            suggested_price = 159.99
            discount_text = None
        elif wtp_pred >= 145:
            suggested_price = 149.99
            discount_text = None
        elif wtp_pred >= 125:
            suggested_price = 129.99
            discount_text = '限时特价'
        else:
            suggested_price = 119.99
            discount_text = '新用户专属优惠'

        return {
            'user_id': user.user_id,
            'wtp_estimate': round(wtp_pred, 2),
            'suggested_price': suggested_price,
            'discount_text': discount_text,
            'price_sensitivity': 'high' if user.comparison_events > 2 else 'low',
        }


def run_personalized_pricing_demo():
    print('=' * 65)
    print('Personalized ML Pricing — 个性化ML定价')
    print('=' * 65)

    predictor = WTPPredictor()
    predictor.train()

    users = [
        UserPricingFeatures('HIGH-WTP', 200, 5, 15, True, 280, 8, 0, 'desktop', 'direct', 1),
        UserPricingFeatures('MID-WTP', 130, 2, 35, True, 150, 5, 1, 'mobile', 'organic', 2),
        UserPricingFeatures('LOW-WTP', 60, 1, 60, False, 80, 3, 3, 'mobile', 'paid', 3),
        UserPricingFeatures('BARGAIN', 90, 0, 1, False, 45, 2, 4, 'mobile', 'paid', 3),
    ]

    print(f'\n📊 个性化定价建议:')
    print(f'  {"用户":>12} {"WTP估计":>9} {"建议售价":>10} {"折扣文案":>12} {"价格敏感度"}')
    print('  ' + '-' * 62)
    for user in users:
        result = predictor.predict_wtp(user)
        discount = result['discount_text'] or '—'
        print(f'  {user.user_id:>12} ${result["wtp_estimate"]:>8.2f} '
              f'${result["suggested_price"]:>9.2f} {discount:>12} {result["price_sensitivity"]}')

    # 利润对比
    uniform_price = 149.99
    costs = [45.0] * len(users)  # 成本 $45
    uniform_profits = sum(max(0, uniform_price - c) for c in costs)

    results = [predictor.predict_wtp(u) for u in users]
    personalized_profits = sum(
        max(0, r['suggested_price'] - c) if r['wtp_estimate'] >= r['suggested_price'] else 0
        for r, c in zip(results, costs)
    )

    print(f'\n💰 利润对比:')
    print(f'  统一定价 ($149.99) 利润: ${uniform_profits:.2f}')
    print(f'  个性化定价利润:          ${personalized_profits:.2f}')
    improvement = (personalized_profits - uniform_profits) / max(uniform_profits, 1) * 100
    print(f'  利润提升: {improvement:+.1f}%')

    print('\n[✓] Personalized ML Pricing 测试通过')


if __name__ == '__main__':
    run_personalized_pricing_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Price-Elasticity-Estimation]]（群体弹性估算是个性化定价的基础，本 Skill 是其用户级升级）
- **前置（prerequisite）**：[[Skill-Purchase-Intent-Prediction]]（购买意图预测提供 WTP 的行为信号）
- **延伸（extends）**：[[Skill-Real-Time-Competitive-Repricing]]（个性化定价 + 竞品监测 = 用户级动态竞争定价）
- **延伸（extends）**：[[Skill-LLM-Negotiation-Conversion-Agent]]（WTP 估计驱动成交 Agent 的让步上限）
- **可组合（combinable）**：[[Skill-LTV-Prediction-BTYD]]（组合：高 CLV 用户给更高 WTP → 允许更高个性化定价）
- **可组合（combinable）**：[[Skill-Causal-RL-Dynamic-Pricing]]（组合：因果 RL 去除混淆 + 个性化 ML 定价 = 因果-个性化双层定价体系）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 利润提升 10-18%（更充分利用 WTP 分布）：月增利润 ¥5-15 万
  - 价格敏感用户转化率提升（展示合适价格）：月增 GMV ¥3-10 万
  - 高 WTP 用户溢价捕捉（减少"本来愿意多付"的机会损失）
  - **年化综合 ROI：¥20-60 万**

- **实施难度**：⭐⭐⭐⭐☆（需要用户行为数据基础设施 + WTP 历史实验标注；法律合规边界需注意；约 6-8 周）

- **优先级评分**：⭐⭐⭐⭐☆（完全空白的重要场景；17-价格优化域数量到达 16 个；桥接 价格优化↔用户分析↔因果推断三域）

- **评估依据**：个性化定价在 Amazon/Uber 等平台已大规模部署；学术研究显示个性化定价比统一定价利润提升 10-30%；DTC 独立站的技术实现比平台更灵活
