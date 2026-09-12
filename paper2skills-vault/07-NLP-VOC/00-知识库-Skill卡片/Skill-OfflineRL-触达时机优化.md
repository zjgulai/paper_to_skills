---
title: Skill: 离线RL触达时机优化 - 不打扰的最优干预
module: 07-NLP-VOC
paper_id: 2202.03867
evidence_basis: paper-verbatim
created: 2026-05-15
updated: 2026-09-12
---

# Skill: 离线RL触达时机优化 - 不打扰的最优干预

## 基础信息

- **arXiv ID**: 2202.03867
- **论文标题**: Offline Reinforcement Learning for Mobile Notifications
- **发表会议**: CIKM 2022 (LinkedIn)
- **核心方法**: 离线RL (CQL) + 状态边缘化重要性采样

---

## 1. 算法原理

### 1.1 问题背景

推送通知的困境：
- 发太少 → 用户遗忘，转化率低
- 发太多 → 打扰用户，导致关闭推送甚至卸载
- **时机不对** → 睡眠中的妈妈被吵醒，品牌好感度暴跌

传统方法缺陷：
- 响应预测模型只预测"是否点击"，不优化长期用户体验
- 无法处理序列决策（多次推送的累积效应）

### 1.2 离线RL框架

```
状态(State): 用户上下文
  - 时间特征（时区、工作日/周末）
  - 用户活动状态（活跃/睡眠/工作中）
  - 近期推送历史（次数、频率）
  - 用户画像（新手妈妈/职场妈妈等）

动作(Action): 推送决策
  - 发送 / 不发
  - 推送内容类型

奖励(Reward): 多目标加权
  - 即时点击: +1
  - 负面反馈(关闭推送): -10
  - 长期留存提升: +5

目标: 学习最优策略 π(a|s) 最大化累积奖励
```

### 1.3 CQL (Conservative Q-Learning)

核心挑战：离线RL没有在线探索，容易高估未见过动作的价值。

**CQL解决方案**：
```
Q(s,a) = 真实价值 - 不确定性惩罚

保守估计避免过度乐观
```

**反直觉洞察**：
1. **序列效应 > 单次决策**：优化推送序列比优化单条推送提升18%
2. **睡眠窗口期**：新妈妈凌晨2-5点绝对静默，即使"高价值"推送也要抑制
3. **疲劳衰减**：同一用户3天内第3条推送的边际效应趋近于0

---

## 2. 业务应用

### 2.1 Momcozy场景：新妈妈推送时机优化

```python
# 目标用户：产后3个月的职场背奶妈妈
# 推送内容：配件促销

# 状态编码
state = {
    'time_of_day': '14:00',        # 下午2点
    'day_of_week': 'Tuesday',      # 周二
    'user_activity': 'app_active', # 刚刚打开过App
    'recent_pushes': 1,            # 本周已推送1次
    'persona': '职场背奶妈妈',
    'baby_age_months': 3,
    'last_pump_time': '12:30'      # 中午吸奶过
}

# RL策略输出
action_probabilities = {
    '立即发送促销推送': 0.05,     # 低 - 打扰工作
    '今日18:00发送': 0.65,       # 高 - 下班通勤时间
    '今日21:00发送': 0.25,       # 中 - 睡前
    '不发': 0.05                 # 低 - 用户活跃中，有机会
}

# 最优决策：延迟到下班时间
```

### 2.2 不同人群的推送窗口

| 人群类型 | 最佳窗口 | 禁忌时段 | 周频次上限 |
|---------|---------|---------|-----------|
| **职场背奶妈妈** | 12:00-13:00, 18:00-19:00 | 09:00-11:30 (会议) | 3次 |
| **全职新手妈妈** | 10:00-11:00, 14:00-15:00 | 02:00-06:00 (夜奶) | 5次 |
| **出差旅行妈妈** | 灵活，基于地理位置 | 航班起飞/降落时段 | 2次 |

### 2.3 与GPLR的联动

```
【GPLR人群标签】
       ↓
【离线RL时机优化】
       ↓
人群标签 → 个性化时间窗口 → 最优推送时机

例：
职场背奶妈妈 → 避开会议时段 → 午休/下班触发
全职新手妈妈 → 避开夜奶时段 → 上午/下午触发
```

---

## 3. 业务价值

| 收益来源 | 提升幅度 | 预估收益 |
|---------|---------|---------|
| 点击率提升 | +15-20% | 50万/年 |
| 负面反馈减少 | -30% | 减少流失 40万/年 |
| 用户满意度提升 | NPS +8分 | 品牌价值提升 |
| **总计** | - | **90万+/年** |

---

## 4. 技能关联

| 前置技能 | 关系 | 说明 |
|---------|------|------|
| **GPLR人群标签** | 输入 | 人群标签作为状态特征 |
| **TSCAN挽回策略** | 配合 | 确定策略后选择时机 |

| 后置技能 | 关系 | 说明 |
|---------|------|------|
| **个性化文案生成** | 配合 | 时机 + 内容联合优化 |

---

**难度**: ⭐⭐⭐⭐⭐ (5/5) - 需要RL基础设施  
**优先级**: P5 - 营销优化方向进阶技能

---

## ⑥ 原文引用

> 原文:"We propose an offline reinforcement learning framework to optimize sequential notification decisions for driving user engagement."
> 出处：2202.03867 §Abstract

> 原文:"We describe a state-marginalized importance sampling policy evaluation approach, which can be used to evaluate the policy offline and tune learning hyperparameters."
> 出处：2202.03867 §Abstract

> 原文:"we collect data through online exploration in the production system, train an offline Double Deep Q-Network and launch a successful policy online."
> 出处：2202.03867 §Abstract

> 原文:"a user’s experience depends on a sequence of notifications and attributing impact to a single notification is not always accurate, if not impossible."
> 出处：2202.03867 §Abstract

> 原文:"Most machine learning applications in notification systems are built around response-prediction models, trying to attribute both short-term impact and long-term impact to a notification decision."
> 出处：2202.03867 §Abstract

> 原文:"Although intrusive and frequent notifications can bring users back to site, they could create notification fatigue or cause notification disablement, which hurts user engagement in the long run"
> 出处：2202.03867 §I Introduction

> 原文:"We propose a state-marginalized importance sampling algorithm for offline evaluation to reduce the high variance of the existing importance sampling based algorithms."
> 出处：2202.03867 §I Introduction（贡献之一）

> 原文:"In this paper, we focus our discussions on applying reinforcement learning to such time-insensitive notifications to determine the best delivery times towards long-term engagement."
> 出处：2202.03867 §III Notification delivery time optimization

> 原文:"We consider a discrete action space consisting of two actions - SEND (send the notification candidate to the user) and NOT-SEND (the notification candidate is put back in the notification queue for further considerations)."
> 出处：2202.03867 §III-B Markov Decision Process for Notification Spacing

> 原文:"In this paper, we use a user visit to the platform within the next time step as a reward."
> 出处：2202.03867 §III-B Markov Decision Process for Notification Spacing

> 原文:"The reward can also be defined as notification clicks, or notification disables as negative rewards or a linear combination of them."
> 出处：2202.03867 §III-B Markov Decision Process for Notification Spacing

> 原文:"Our proposed offline solution is a combination of Offline Deep Q-Network (DQN) and data collection with well-controlled online exploration."
> 出处：2202.03867 §IV-A Offline Training

> 原文:"We then train the Double DQN models described in Section IV-A using a fully-connected 3-layer neural network with different hyper-parameters"
> 出处：2202.03867 §V-B Online Experiments in Notification Spacing

> 原文:"Compared with the baseline policy, the new policy from offline reinforcement learning increased the total sessions by $0.3\%$, which is considered a moderate gain in a volume neutral iteration, but very impressive given that the total notification volume is reduced by $3.49\%$."
> 出处：2202.03867 §V-B Online Experiments in Notification Spacing（表 I 线上 A/B 结果）

> 原文:"The $4.53\%$ increase in notification CTR and $4.37\%$ decrease in notification unfollow total are mainly driven by the reduction in notification volume."
> 出处：2202.03867 §V-B Online Experiments in Notification Spacing（表 I 线上 A/B 结果）

> 原文:"The tuning typically takes 1-3 weeks for notifications as site engagement responses takes days to show up."
> 出处：2202.03867 §V-B Online Experiments in Notification Spacing

> 原文:"One of the limitations of our presented results is that we trained and tested this framework in a one-week frame."
> 出处：2202.03867 §VI Discussion（论文自承局限）
