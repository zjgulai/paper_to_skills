---
title: Skill-DQN-Purchase-Prediction
module: 06-增长模型
topic: 借用 DQN 的经验回放思路做 LSTM 购买意图预测（监督式框架）
status: draft
created: 2026-05-15
updated: 2026-09-13
owner: self
source: ai
paper_id: 2506.17543
paper: Predicting E-commerce Purchase Behavior using a DQN-Inspired Deep Learning Model
evidence_basis: paper-verbatim
verified_by: quote_check.py (引文逐字核验 VERBATIM) + gate_check.py G2
verified_at: 2026-09-12
related: Skill-User-Lifecycle-STAN.md, Skill-Customer-Journey-Prototype.md
---

# Skill Card: DQN-Inspired Purchase Intent Prediction
# DQN深度强化学习购买意图预测

**论文来源**: Predicting E-commerce Purchase Behavior using a DQN-Inspired Deep Learning Model  
**arXiv ID**: [2506.17543](https://arxiv.org/abs/2506.17543)  
**发表日期**: 2025-06  
**作者**: Aditi Madhusudan Jain  
**适用领域**: 购买意图预测、AIPL的P阶段量化、实时转化评分

---

## ① 算法原理

### 核心思想
传统购买预测模型将问题视为静态分类任务。DQN-inspired方法把**单次用户会话**当作一个整体分析单元：用 LSTM 对会话内的行为序列建模，并借用 DQN 训练中的**经验回放**（把历史会话存池、训练时随机采样批次）来打破连续样本的相关性。模型输出的是「该会话最终成交」的概率，是**监督式二分类**，而不是 RL 闭环。

论文自述其借用边界（§V-A「Deep Q-Network Inspiration」，该节列举「Our model draws parallels to this approach in several ways」的三条）：
- **Sequential Decision Making**：像 DQN 处理状态/动作序列那样，处理会话内的用户交互序列（§V-A）
- **Value Prediction**：DQN 预测动作值，本模型预测「会话的价值」= 该会话成交的可能性（§V-A）
- **Experience Replay**：把历史会话存池、训练时随机采样，对应 DQN 的训练过程（§V-A）
- **Iterative Update**：像 DQN 迭代更新 Q 值那样，随处理更多会话迭代更新购买概率（§V-C）
- **明确不借用的部分**：论文在 §V-C 写 `While not directly applicable in our supervised setting`——epsilon-greedy 探索在这套监督式设定下**不直接适用**，论文只退化为「训练时给特征集偶尔注入随机噪声」。论文**没有**把营销干预当作 action、把转化当作 reward，**没有** Q 学习闭环（无 Bellman 更新、无 target network）。

### 数学直觉

**会话值估计**：  
V(s) = P(会话 s 最终成交) —— 论文用「会话的价值」类比 DQN 的状态值，而非动作值 Q(s,a)

**前向计算**（§III-C）：  
V(X) = σ( W₂ · ReLU(W₁ · LSTM₂(LSTM₁(X)) + b₁) + b₂ )

**经验回放**（§V-A / §V-C）：  
从回放池随机采样批次 → 降低连续训练样本之间的相关性，提升学习稳定性

**直观解释**：模型不学「该在什么时机对谁做什么干预」，它只学「这次会话看起来有多像会成交」。干预时机属于下游业务决策，论文本身不涉及。

### 关键假设
1. 用户行为序列具有时序依赖性（当前行为受历史影响）
2. 类别不平衡严重（购买样本远少于浏览样本）
3. 存在可学习的隐含状态转移规律

---

## ② 母婴出海应用案例

### 场景1：实时购买意向评分

**业务问题**  
母婴用户决策周期长，但转化窗口期短（如孕晚期囤货）。运营团队需要在正确时机推送正确优惠。传统规则（如"加购后24小时发券"）太粗糙，无法个性化识别高意向用户。

**数据要求**
- 用户会话特征（1,114维）： demographics、浏览历史、加购记录
- 行为序列：点击、浏览、搜索、加购、收藏的时间序列
- 标签：是否购买、购买金额

| 特征类别 | 示例字段 | 维度 |
|----------|----------|------|
| 用户画像 | 孕周、宝宝月龄、地域 | 10 |
| 行为统计 | 近7天浏览数、加购数 | 50 |
| 品类偏好 | 奶粉/尿布/辅食点击占比 | 20 |
| 时序编码 | 小时、星期、是否周末 | 10 |
| 行为序列 | 最近20步行为ID | 20 |
| 交叉特征 | 用户-品类交互 | 1000+ |

**预期产出**
- 实时购买概率评分（0-100%）
- AIPL的P阶段细分（Purchase-High/Med/Low/Interest）
- 分群后的干预优先级排序（下游业务决策，非论文输出）

**业务价值**（业务假设代入，非论文实测）
- 营销触达精准度提升 40%+
- 优惠券使用率提升 25%
- 无效触达减少 30%（降低用户疲劳）

**参考论文指标**: 论文自身表 V 为 AUC-ROC **0.6257**、总体准确率 87.62%；摘要另称 88% / AUC-ROC 0.88，与表 V 冲突（见 ⑤ 与 ⑥）

---

### 场景2：AIPL三技能联动决策

**业务问题**  
已有STAN识别生命周期阶段，Journey Prototype识别行为模式，但缺少**量化转化概率**的一环。需要三技能联动：阶段+模式+概率 = 完整画像。

**数据流**
```
用户行为 → STAN(阶段识别) → Interest
         → Journey Prototype(模式识别) → 跨渠道比价型
         → DQN Predictor(概率预测) → 72%购买概率
         
输出标签: "兴趣期-跨渠道比价型-高转化概率(72%)"
干预策略: "立即推送App专属优惠券，强调比价优势"
```

**预期产出**
- 三维度用户标签体系
- 分群转化概率基线
- 个性化干预ROI预估

**业务价值**
- 标签体系完整度从60%→95%
- 运营策略可解释性大幅提升
- A/B测试设计更精准（基于概率分层）

---

## ③ 代码模板

代码位置: `paper2skills-code/growth_model/dqn_purchase_prediction/model.py`

**架构对齐（严格按 §III-C）**：输入层 → 2 层 LSTM（64 / 32 单元）→ 每层后接 BatchNorm + Dropout(0.2) → Dense(16, ReLU) → Dense(1, sigmoid)。
底本**无 attention 组件**、**无 bidirectional**、**无 target network / Bellman 更新**（§III-C 只列 LSTM / BatchNorm / Dropout / Dense 四类）。

核心组件：
1. **ExperienceReplayBuffer**: 经验回放池（论文借用的 DQN 要素，§V-A）
2. **DQNInspiredPurchaseNet**: 上述监督式 LSTM 网络（**无 attention**）
3. **train / predict_proba**: Adam(lr=1e-3) + 二元交叉熵（§III-D），反频率类别权重
4. **aipl_stage**: 概率→AIPL 分群映射（业务侧约定，非论文指标）

```python
"""DQN-inspired 购买意图预测 —— 严格对齐 arXiv 2506.17543 v1 §III-C 的监督式实现"""
import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn

RANDOM_SEED = 42
SEQ_LENGTH = 20
INPUT_DIM = 32
# 论文 §III-C 的架构参数：2 层 LSTM(64,32) + Dropout 0.2 + Dense(16, ReLU) + Dense(1, sigmoid)
PAPER_ARCH = {"lstm_units": (64, 32), "dropout": 0.2, "dense_units": 16}


@dataclass
class Session:
    """一条会话样本（监督式：只有特征与购买标签，无 action / reward）"""
    features: torch.Tensor   # [seq_len, input_dim]
    purchased: int           # 0/1


class ExperienceReplayBuffer:
    """DQN-inspired 经验回放（§V-A）——池内元素是 (特征, 购买标签)，不是 RL 五元组"""

    def __init__(self, capacity: int = 2000, seed: int = RANDOM_SEED):
        self.buffer = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, session: Session) -> None:
        self.buffer.append(session)

    def sample(self, batch_size: int) -> list:
        n = min(batch_size, len(self.buffer))
        return self._rng.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNInspiredPurchaseNet(nn.Module):
    """监督式购买概率预测网络（架构严格按 §III-C）"""

    def __init__(self, input_dim: int = INPUT_DIM, dropout: float = PAPER_ARCH["dropout"]):
        super().__init__()
        l1, l2 = PAPER_ARCH["lstm_units"]

        self.lstm1 = nn.LSTM(input_dim, l1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(l1)
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(l1, l2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(l2)
        self.drop2 = nn.Dropout(dropout)

        self.dense = nn.Linear(l2, PAPER_ARCH["dense_units"])
        self.out = nn.Linear(PAPER_ARCH["dense_units"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F] -> 购买概率 [B]"""
        h1, _ = self.lstm1(x)
        h1 = self.drop1(self.bn1(h1.transpose(1, 2)).transpose(1, 2))
        h2, _ = self.lstm2(h1)
        h2 = self.drop2(self.bn2(h2.transpose(1, 2)).transpose(1, 2))
        return torch.sigmoid(self.out(torch.relu(self.dense(h2[:, -1, :])))).squeeze(-1)


def build_synthetic_sessions(n: int = 240, seed: int = RANDOM_SEED) -> list:
    """构造可复现的合成会话（仅用于验证代码可执行，不代表论文实验结果）"""
    g = torch.Generator().manual_seed(seed)
    sessions = []
    for _ in range(n):
        feat = torch.randn(SEQ_LENGTH, INPUT_DIM, generator=g)
        noise = torch.randn(1, generator=g).item()
        purchased = int(feat.mean().item() + 0.35 * noise > 0.0)
        sessions.append(Session(features=feat, purchased=purchased))
    return sessions


def train(model, sessions, epochs: int = 3, batch_size: int = 32, lr: float = 1e-3):
    """监督式训练：Adam(lr=1e-3) + 二元交叉熵 + 反频率类别权重（§III-D）"""
    replay = ExperienceReplayBuffer()
    for s in sessions:
        replay.push(s)

    pos = sum(s.purchased for s in sessions)
    neg = len(sessions) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []
    for _ in range(epochs):
        batch = replay.sample(batch_size)
        x = torch.stack([s.features for s in batch])
        y = torch.tensor([s.purchased for s in batch], dtype=torch.float32)

        optimizer.zero_grad()
        h = model.lstm1(x)[0]
        h = model.drop1(model.bn1(h.transpose(1, 2)).transpose(1, 2))
        h = model.lstm2(h)[0]
        h = model.drop2(model.bn2(h.transpose(1, 2)).transpose(1, 2))
        logits = model.out(torch.relu(model.dense(h[:, -1, :]))).squeeze(-1)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def predict_proba(model, sessions) -> torch.Tensor:
    model.eval()
    return model(torch.stack([s.features for s in sessions]))


def aipl_stage(p: float) -> str:
    """概率→AIPL 分群（业务侧阈值，非论文指标）"""
    if p >= 0.8:
        return "Purchase-High"
    if p >= 0.5:
        return "Purchase-Med"
    if p >= 0.3:
        return "Purchase-Low"
    return "Interest"


def test_can_run_dqn_purchase_prediction() -> None:
    """K1 断言：架构与 §III-C 逐项对齐 + 监督式训练可跑通"""
    torch.manual_seed(RANDOM_SEED)
    sessions = build_synthetic_sessions()

    assert PAPER_ARCH["lstm_units"] == (64, 32), "LSTM 单元数须为 64/32（§III-C）"
    assert PAPER_ARCH["dropout"] == 0.2, "dropout 须为 0.2（§III-C）"
    assert PAPER_ARCH["dense_units"] == 16, "Dense 须为 16 单元 + ReLU（§III-C）"

    model = DQNInspiredPurchaseNet()
    assert model.lstm1.hidden_size == 64 and model.lstm2.hidden_size == 32
    assert model.lstm1.bidirectional is False, "论文为单向 LSTM，无 bidirectional"
    assert not hasattr(model, "attention"), "论文架构无 attention 组件（§III-C）"

    losses = train(model, sessions)
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))

    probs = predict_proba(model, sessions[:16])
    assert probs.shape == (16,), f"输出应为每会话一个概率，实得 {tuple(probs.shape)}"
    assert bool(((probs >= 0) & (probs <= 1)).all()), "概率须落在 [0,1]"
    assert aipl_stage(0.9) == "Purchase-High"


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    sessions = build_synthetic_sessions()
    model = DQNInspiredPurchaseNet()
    losses = train(model, sessions)
    probs = predict_proba(model, sessions[:16])
    test_can_run_dqn_purchase_prediction()

    print("架构: 2×LSTM(64,32)+BN+Dropout(0.2) → Dense(16,ReLU) → Dense(1,sigmoid)")
    print(f"训练 3 轮，末轮 loss={losses[-1]:.4f}")
    print(f"示例概率 {[round(float(p), 3) for p in probs[:5]]} → 分群 {aipl_stage(float(probs[0]))}")


if __name__ == "__main__":
    main()
```

运行测试:
```bash
cd paper2skills-code/growth_model/dqn_purchase_prediction
python3 model.py
```

> ⚠️ 本模板用合成数据验证「架构与训练流程可执行」，**不复现论文的实验数值**（论文数值见 ⑥ 引文与 ⑤ 的实测口径）。
>
> ⚠️ **与仓库现有文件的已知漂移**：`paper2skills-code/growth_model/dqn_purchase_prediction/model.py` 目前实现的是
> `bidirectional=True` 的 LSTM + 自注意力层 + Q 值/目标网络 + Epsilon 衰减 —— 这些均**不在论文 §III-C 的架构里**。
> 本卡③以论文为准；该文件的同步修正需另开任务（本次仅改卡片）。

---

## ④ 技能关联

### 前置技能
- **Skill-User-Lifecycle-STAN**: 生命周期阶段作为特征输入
- **Skill-Customer-Churn-Prediction**: 用户行为序列处理基础
- **Skill-Time-Series-Forecasting**: 时序建模基础

### 延伸技能
- **Skill-Multi-Armed-Bandit**: 将预测概率用于动态定价/优惠券策略
- **Skill-Causal-Uplift-Modeling**: 评估干预的真实因果效应（预测vs实际）
- **Skill-Recommendation-System**: 高概率用户精准推荐

### 三技能联动（AIPL-VOC标签体系闭环）

| 技能 | 输出 | 作用 |
|------|------|------|
| **STAN** | Awareness/Interest/Purchase/Loyalty | 判断用户所处生命周期阶段 |
| **Journey Prototype** | App深度型/跨渠道比价型/线下体验型 | 识别用户行为模式 |
| **DQN Purchase** | 购买概率 0-100% | 量化当前转化可能性 |

**组合效果**：
```
输入: 用户U123的最新会话数据
↓
STAN: Interest期 (置信度85%)
Journey: 跨渠道比价型 (距离0.23)
DQN: 购买概率72% (置信度medium)
↓
组合标签: "兴趣期-跨渠道比价型-高转化(72%)"
↓
运营动作: "立即推送全渠道比价指南+限时优惠券"
预期转化: 72% → 85% (提升13个百分点)
```

---

## ⑤ 商业价值评估

### ROI预估

**实施成本**：
- 模型开发：2-3周（基于代码模板）
- 实时服务部署：1-2周
- **总计成本**：约25-35人天

**预期收益**（年化）：
- 营销触达精准度提升40% → 假设月营销费用100万，节约40万/月 = **480万/年**
- **模型判别力口径（论文实测表 V）**：本模型 AUC-ROC **0.6257**，低于论文同时评测的 Logistic Regression(0.8152) / Random Forest(0.8599) / XGBoost(0.8366) / 普通 LSTM(0.6928)；论文 §IX 自承「lower AUC-ROC score compared to some traditional methods」。摘要的 AUC-ROC 0.88 与表 V 冲突，**不可作为收益依据**
- **年化ROI**：480万 / 20万成本 = **24倍**（收益侧全部为业务假设代入，与论文指标的因果关系未经论文验证）

### 实施难度
3/5星

**依据**：
- 代码模板完整，包含经验回放、2 层 LSTM + BatchNorm + Dropout 等细节（无 attention）
- 需要实时计算基础设施（在线推理）
- 特征工程要求高（1,114维特征）
- ⚠️ 判别力实测偏低（AUC-ROC 0.6257），高精度只能靠在 0.9 阈值牺牲召回换取（精确率 0.9967 / 召回 0.2521）

### 优先级评分
3/5星

**依据**：
- **时效性**：2025年6月论文，方法较新
- **指标需打折**：摘要称 88% / AUC-ROC 0.88，但论文自身表 V 的 AUC-ROC 仅 0.6257，且**低于全部传统基线**；准确率 88% 只在 0.6–0.8 阈值区间成立
- **互补性好**：与已有两技能形成AIPL-VOC闭环
- **业务价值为假设代入**：收益数字与论文指标之间无已验证因果链

### 三技能组合实施建议

**阶段1**（2周）：STAN上线，输出生命周期阶段标签
**阶段2**（2周）：Journey Prototype上线，输出行为模式标签  
**阶段3**（2周）：DQN Purchase上线，输出购买概率评分
**阶段4**（2周）：三技能联动，构建完整用户画像和运营决策系统

**预期组合效果**：
- 用户画像完整度：95%+
- 运营策略精准度：+40%
- 营销ROI提升：+50%

---

## ⑥ 原文引用

> 原文："Our approach to predicting buying intent and product demand in e-commerce settings draws inspiration from Deep Q-Networks (DQN), a technique traditionally used in reinforcement learning. We adapt this concept to a supervised learning context, leveraging its ability to handle sequential data and make decisions based on complex patterns of user behavior."
> 出处：2506.17543 §V Methodology（PDF 第 5 页）

> 原文："Experience Replay: We implement a form of experience replay, storing and randomly sampling from past user sessions during training, mirroring the DQN training process."
> 出处：2506.17543 §V-A Deep Q-Network Inspiration（PDF 第 5 页）

> 原文："Epsilon-Greedy Exploration: While not directly applicable in our supervised setting, we implement a form of exploration by occasionally introducing random noise in our feature set during training, inspired by the epsilon-greedy strategy in DQNs."
> 出处：2506.17543 §V-C Training Process（PDF 第 6 页）——注意：论文**没有**把干预当作 action、把转化当作 reward 的 Q 学习闭环

> 原文："Value Prediction: While DQNs predict action-values, our model predicts the ”value” of a session in terms of its likelihood to result in a purchase."
> 出处：2506.17543 §V-A（PDF 第 5 页）

> 原文："We evaluate our model on a large-scale e-commerce dataset comprising over 885,000 user sessions, each characterized by 1,114 features."
> 出处：2506.17543 §Abstract（PDF 第 1 页）

> 原文："Through comprehensive experimentation with various classification thresholds, we show that our model achieves a balance between precision and recall, with an overall accuracy of 88% and an AUC-ROC score of 0.88."
> 出处：2506.17543 §Abstract（PDF 第 1 页）——摘要口径；与表 V 的 AUC-ROC 0.6257 **不一致**，见下

> 原文："Our approach demonstrates robust performance in handling the inherent class imbalance typical in e-commerce data, where purchase events are significantly less frequent than non-purchase events."
> 出处：2506.17543 §Abstract（PDF 第 1 页）

> 原文："To address the class imbalance inherent in e-commerce data (where purchase events are typically less frequent than view or cart events), we employed class weighting in the loss function. The weights were inversely proportional to the class frequencies in the training data."
> 出处：2506.17543 §III-D Training Process（PDF 第 3 页）

> 原文："We split the dataset into training (80%), validation (10%), and test (10%) sets, ensuring that all sessions from a single user were kept in the same set to prevent data leakage."
> 出处：2506.17543 §III-D Training Process（PDF 第 3 页）

> 原文："The model demonstrates strong overall accuracy (87.62%), as evident from Table III, indicating its general effectiveness in predicting user behavior."
> 出处：2506.17543 §VI-A Interpretation of Results（PDF 第 7 页）——原文 87.62%，比摘要的「88%」低

> 原文："The overall accuracy plateaus at 0.88 for thresholds 0.6 through 0.8."
> 出处：2506.17543 §VII-A Analysis of Threshold Impact（PDF 第 9 页）

> 原文："For purchase sessions, Table II shows high precision (0.90) but low recall (0.29)."
> 出处：2506.17543 §VI-A Interpretation of Results（PDF 第 7 页）

> 原文："At the highest threshold of 0.9, we observe an interesting phenomenon. The precision for purchase events reaches 1.00, meaning that when the model predicts a purchase at this threshold, it is always correct. However, this comes at a significant cost to recall, which drops to 0.25."
> 出处：2506.17543 §VII-A Analysis of Threshold Impact（PDF 第 9 页）

> 原文："| Model / Threshold | Accuracy | Precision | Recall | F1-Score | AUC-ROC |"
> 出处：2506.17543 §VII 表 V 表头（PDF 第 9 页）

> 原文："| Our Model (0.5) | 0.8762 | 0.8967 | 0.2921 | 0.4406 | 0.6257 |"
> 出处：2506.17543 §VII 表 V（PDF 第 9 页）——**本表 AUC-ROC 为 0.6257**，与摘要的 0.88 冲突

> 原文："| Logistic Regression | 0.8786 | 0.7793 | 0.3804 | 0.5112 | 0.8152 |"
> 出处：2506.17543 §VII 表 V（PDF 第 9 页）——传统基线判别力高于本模型

> 原文："| Random Forest | 0.8814 | 0.7797 | 0.4035 | 0.5318 | 0.8599 |"
> 出处：2506.17543 §VII 表 V（PDF 第 9 页）

> 原文："| XGBoost | 0.8819 | 0.7949 | 0.3940 | 0.5269 | 0.8366 |"
> 出处：2506.17543 §VII 表 V（PDF 第 9 页）

> 原文："| LSTM | 0.8813 | 0.7720 | 0.4098 | 0.5354 | 0.6928 |"
> 出处：2506.17543 §VII 表 V（PDF 第 9 页）

> 原文："| Our Model (0.9) | 0.8762 | 0.9967 | 0.2521 | 0.4016 | 0.6257 |"
> 出处：2506.17543 §VII 表 V（PDF 第 9 页）——高精度以牺牲召回为代价

> 原文："Our model architecture consists of an input layer accepting the preprocessed session data, followed by two LSTM layers with 64 and 32 units respectively. These LSTM layers are designed to capture the sequential nature of user interactions within a shopping session. Each LSTM layer is followed by a batch normalization layer and a dropout layer with a rate of 0.2 to prevent overfitting."
> 出处：2506.17543 §III-C Model Architecture（PDF 第 3 页）——**架构权威表述；全文无 attention 组件**

> 原文："The output of the LSTM layers is then fed into a dense layer with 16 units and ReLU activation, followed by a final dense layer with a single unit and sigmoid activation. This final layer outputs the probability of the session resulting in a purchase."
> 出处：2506.17543 §III-C Model Architecture（PDF 第 3 页）

> 原文："Firstly, the model’s lower AUC-ROC score compared to some traditional methods suggests that there may be room for improvement in its overall discriminative ability across different classification thresholds."
> 出处：2506.17543 §IX Limitations（PDF 第 10 页）——论文自承判别力弱于部分传统方法

> 原文："Our study utilizes the ”E-commerce Events History in Electronics Store” dataset, publicly available on Kaggle."
> 出处：2506.17543 §III-A Dataset（PDF 第 3 页）

> **口径提示（不改正文，仅记录）**：本卡 ② 里触达精准度、券使用率、无效触达三项提升幅度，以及 ⑤ 的全部费用金额、倍数与人天/周数，均为**业务假设代入**；
> 论文可核验的只有 §V/§VI/§VII 的模型指标（见上方 表 V 与 §VI-A、§VII-A 引文）。

---

## 附录：论文核心信息

| 项目 | 内容 |
|------|------|
| 论文标题 | Predicting E-commerce Purchase Behavior using a DQN-Inspired Deep Learning Model |
| 作者 | Aditi Madhusudan Jain |
| 发表 | arXiv 2025-06 |
| arXiv | 2506.17543 |
| 核心贡献 | 将 DQN 的**经验回放**引入监督式购买预测，结合 2 层 LSTM 时序建模 |
| 架构 | 2×LSTM(64,32) + BatchNorm + Dropout(0.2) → Dense(16,ReLU) → Dense(1,sigmoid)，**无 attention** |
| 数据集 | 885,000+用户会话，1,114个特征 |
| 实验结果 | 表 V：准确率 0.8762，AUC-ROC **0.6257**（低于 LR/RF/XGBoost/LSTM 基线）；摘要另称 88% / AUC-ROC 0.88，两者冲突 |
| 创新点 | 处理类别不平衡、适应性强、适合高维稀疏数据 |
