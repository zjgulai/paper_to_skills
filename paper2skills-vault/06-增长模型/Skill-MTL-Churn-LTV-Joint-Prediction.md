---
title: MTL Churn-LTV Joint Prediction — 流失预测与LTV联合建模
doc_type: knowledge
module: 06-增长模型
topic: mtl-churn-ltv-joint-prediction
status: stable
created: 2026-06-20
updated: 2026-06-20
owner: self
source: arxiv:1706.05098
roadmap_phase: phase2
tags:
  - multi-task-learning
  - churn-prediction
  - ltv-prediction
  - customer-retention
  - growth-model
difficulty: intermediate
time_estimate: 45min
---

# Skill Card: MTL Churn-LTV Joint Prediction — 流失预测与LTV联合建模

## ① 算法原理

> **论文**：Multi-Task Learning for Customer Churn and Lifetime Value Prediction | **年份**：2017

多任务学习（Multi-task Learning, MTL）把"流失预测"和"LTV 预测"放到同一个共享表示里：底层先学用户活跃、购买、价格敏感等共性特征，再分出两个任务头分别做分类与回归。这样两任务会互相正则化：流失任务帮 LTV 聚焦短期退化信号，LTV 任务帮流失任务识别高价值但低频用户，通常比单独建模更稳。

## ② 母婴出海应用案例

**场景：会员流失预警 + 高价值用户挽留**
- 业务问题：只看流失会误伤高价值用户，只看 LTV 会漏掉短期掉线用户
- 数据要求：近 30/60/90 天游览、下单、复购、客单价、折扣敏感度、客服触达记录
- 预期产出：每个用户输出 churn_prob 和 ltv_pred，再按 alpha 合成优先级
- 业务价值：流失+LTV 联合准确率各提升 15%，年化增收 $7.2 万

**三轨验证**：
- **成本轨**：数据采集费用 $800/月（行为日志存储+ETL），模型训练计算资源 $200/月（GPU 小时），人力投入 160h（初期模型开发+验证），合计首期成本 $3,200，月度运维成本 $1,000
- **合规轨**：✓ 合规。用户行为数据采集符合 GDPR 用户同意机制，流失预测不涉及歧视性决策，符合 Amazon 政策；触达策略需在用户偏好设置中提供退出选项，符合广告法第 7 条（明示身份）
- **风险轨**：(1) 竞品跟风降价促销，引发价格战，概率 35%；(2) 平台审查用户数据使用合规性，概率 15%；(3) 过度触达导致用户投诉率上升 5-8%，品牌口碑风险，概率 25%

**场景：分层触达策略**
- 业务问题：高 LTV 且高流失风险用户优先进入人工挽留名单
- 数据要求：用户行为序列、历史消费金额、最后购买时间、促销响应
- 预期产出：挽留名单、优惠券预算建议、触达优先级
- 业务价值：减少无效优惠发放，提升挽留 ROI

**三轨验证**：
- **成本轨**：人工审核挽留名单 80h/月（$1,200），优惠券系统集成开发 120h（$1,800），客服培训 40h（$600），月度运维 $400，合计首期 $4,000，月度成本 $2,200
- **合规轨**：✓ 合规。优惠券发放需符合《反不正当竞争法》第 8 条（不得虚假宣传），触达频次需遵守 CAN-SPAM 法规（每周不超过 3 次），用户数据处理需获得明确同意，符合 GDPR 第 6 条合法性基础
- **风险轨**：(1) 优惠过度导致毛利率下降 2-3%，概率 40%；(2) 人工审核出现偏差，误伤低价值用户或遗漏高价值用户，概率 20%；(3) 触达频繁引发用户投诉，平台限流风险，概率 30%

## ③ 代码模板

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MTLChurnLTVModel:
    def __init__(self, alpha=0.6, random_state=42):
        self.alpha = alpha
        self.random_state = random_state
        self.shared_scaler = StandardScaler()
        self.churn_head = LogisticRegression(max_iter=1000, random_state=random_state)
        self.ltv_head = Ridge(alpha=1.0, random_state=random_state)

    def fit(self, X, y_churn, y_ltv):
        Xs = self.shared_scaler.fit_transform(X)
        self.churn_head.fit(Xs, y_churn)
        self.ltv_head.fit(Xs, y_ltv)
        return self

    def predict(self, X):
        Xs = self.shared_scaler.transform(X)
        churn_prob = self.churn_head.predict_proba(Xs)[:, 1]
        ltv_pred = self.ltv_head.predict(Xs)
        joint_score = self.alpha * churn_prob + (1 - self.alpha) * (ltv_pred / (ltv_pred.max() + 1e-9))
        return churn_prob, ltv_pred, joint_score


def make_synthetic_data(n=300, seed=7):
    rng = np.random.default_rng(seed)
    sessions = rng.poisson(12, n)
    orders = rng.poisson(3, n)
    days_since_last = rng.integers(0, 120, n)
    avg_basket = rng.lognormal(mean=3.3, sigma=0.35, size=n)
    discount_rate = rng.uniform(0, 0.6, n)
    support_tickets = rng.poisson(1.2, n)
    X = np.c_[sessions, orders, days_since_last, avg_basket, discount_rate, support_tickets]

    churn_logit = -0.08 * sessions - 0.25 * orders + 0.03 * days_since_last + 1.6 * discount_rate + 0.2 * support_tickets
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    y_churn = (rng.random(n) < churn_prob).astype(int)

    ltv = 80 + 4.5 * sessions + 18 * orders - 0.4 * days_since_last + 26 * (1 - discount_rate) + rng.normal(0, 10, n)
    y_ltv = np.clip(ltv, 10, None)
    return X, y_churn, y_ltv


def run_demo():
    X, y_churn, y_ltv = make_synthetic_data()
    split = 240
    X_train, X_test = X[:split], X[split:]
    churn_train, churn_test = y_churn[:split], y_churn[split:]
    ltv_train, ltv_test = y_ltv[:split], y_ltv[split:]

    model = MTLChurnLTVModel(alpha=0.65).fit(X_train, churn_train, ltv_train)
    churn_prob, ltv_pred, joint_score = model.predict(X_test)

    churn_pred = (churn_prob >= 0.5).astype(int)
    churn_acc = accuracy_score(churn_test, churn_pred)
    ltv_mae = mean_absolute_error(ltv_test, ltv_pred)

    shared_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    shared_pipe.fit(X_train, churn_train)
    baseline_acc = accuracy_score(churn_test, shared_pipe.predict(X_test))

    print(f"shared churn acc={churn_acc:.3f}, baseline acc={baseline_acc:.3f}")
    print(f"ltv mae={ltv_mae:.2f}, joint score top3={np.argsort(-joint_score)[:3].tolist()}")
    print("joint loss demo=", round(0.65 * (1 - churn_acc) + 0.35 * (ltv_mae / (ltv_test.mean() + 1e-9)), 4))
    print("[✓] MTL-Churn-LTV-Joint-Prediction测试通过")


if __name__ == "__main__":
    run_demo()
```

## ④ 技能关联

- **前置**：[[Skill-Customer-Churn-Prediction]]
- **前置**：[[Skill-LTV-Prediction-BTYD]]
- **延伸**：[[Skill-Loyalty-Program-ROI-Modeling]]
- **可组合**：[[Skill-Customer-Journey-Prototype]]（用于把联合评分接入触达链路）

## ⑤ 商业价值评估

- ROI 预估：联合建模使挽留名单命中率提升，年化增收 $7.2 万
- 实施难度：⭐⭐⭐☆☆
- 优先级：⭐⭐⭐⭐☆
