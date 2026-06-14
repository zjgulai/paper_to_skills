---
title: TRACE 跨会话点击流用户嵌入
doc_type: knowledge
module: 14-用户分析
topic: clickstream-user-embedding

roadmap_phase: phase2
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: arXiv:2409.12972
---

# Skill: TRACE — 跨多会话点击流用户嵌入 (Transformer-based Clickstream User Embedding)

> 主论文：**TRACE: Transformer-based user Representations from Attributed Clickstream Event sequences** · arXiv:2409.12972 (2024) · RecTour @ RecSys 2024
> 作者：William Black, Alexander Manlove, Jack Pennington, Andrea Marchini, Ercument Ilhan, Vilda Markeviciute（Expedia Group）
> 应用：旅行电商跨多会话页面级点击流建模，生成实时低维用户嵌入向量用于个性化推荐

---

## ① 算法原理

### 核心思想

传统序列推荐模型只看**单会话内的商品点击序列**，TRACE 的创新在于：把整个用户的**多会话页面浏览历史**（包括首页、搜索页、详情页、购物车、结账等各类页面，跨越数天甚至数周）打包成一条有序序列，送入轻量级 Transformer Encoder 学习**全局用户状态嵌入**。

关键区别：
- **粒度是页面（page view），不是商品（item）**：捕捉用户在网站上的完整轨迹，而非仅限购买路径上的商品点击
- **跨会话（multi-session）**：用时间间隔 T 切分会话边界，用两个独立的可学习位置编码分别标记"在第几个会话"和"在该会话第几次点击"，让模型学会区分"今天首次到访"和"三周后返回决策"的语义差异
- **多任务联合训练（MTL）**：同时预测 5 个未来行为信号（购买、跳出、搜索、浏览详情页、查看已下单），靠多目标对齐逼迫 backbone 学出泛化性更强的表示，而非只拟合单一稀疏目标
- **嵌入维度极低（d=32）**：训练完成后去掉 5 个任务头，只留 backbone 输出的 32 维向量作为用户嵌入，接入下游 XGBoost / 推荐召回等系统

### 数学直觉

**会话切分条件（公式 1）：**

```
t_Pj - t_P(j-1) ≤ T,  ∀j ∈ [1, N]
```
时间间隔超过阈值 T（通常几小时）则开启新会话。

**输入表示：**
每次页面访问事件 P 包含：
- 类别特征（页面名称、设备类型）→ 各自独立的 32 维可学习 Embedding
- 时间特征（相邻事件间隔 Δt、距最近事件的时间）→ 取对数后标准化
- 会话 ID → 标识该事件属于第 n 个最近会话（值为 n）

事件位置编码 m（第 m 个最近事件）和会话位置编码 n 各自通过独立线性层映射到 ℝ^D，加到特征向量上，形成**事件-会话双重位置编码**。

最终每次旅程 J 被编码为矩阵 M_J ∈ ℝ^{L×D}（L 为最大序列长度，D ≈ 几百维）。

**Transformer Encoder（单层）：**
- 8 头多头自注意力 + 前馈网络（中间层 128 维）
- Dropout + 残差连接 + LayerNorm
- Global Max Pooling → 压缩到序列无关的固定大小向量
- 共享 FFN → 输出 e_J ∈ ℝ^32（用户嵌入）

**多任务损失（公式 2）：**

```
L(J, y) = -∑_{k=1}^{5} [w_k · y_k · log(ŷ_k) + (1-y_k) · log(1-ŷ_k)]
```
- 每个任务 k 有独立的 class-weighted binary cross-entropy
- 权重 w_k = 1 / (正样本比例)，解决极度类不平衡（购买率极低）
- 5 个任务等权求和，防止稀疏任务被"淹没"

**5 个预测任务：**
| 任务代号 | 含义 | 典型正样本率 |
|---------|------|------------|
| PW2 | 2周内完成购买 | 很低（~1-3%） |
| BN5 | 未来5次页面访问内跳出 | 较高 |
| SRP | 本会话内发起搜索 | 中等 |
| PDP | 本会话内浏览商品详情页 | 中等 |
| VUO | 本会话内查看已下单订单 | 较低 |

### 关键假设

1. **用户意图连续性**：跨会话的历史浏览轨迹能反映用户的持续偏好与购买阶段，而非每次到访都是独立的
2. **页面导航序列有结构**：用户的浏览路径（首页→搜索→详情→结账）蕴含隐式的购买漏斗状态，可被 Transformer 自注意力机制捕捉
3. **多任务信号互补**：高频任务（搜索、浏览详情）的学习信号可迁移到低频任务（购买），MTL 共享 backbone 能获得比单任务更好的泛化
4. **短序列足够**：相比 NLP 长文本，电商点击序列较短，词汇表有限（~1000 页面名），单层 Transformer 即可充分学习

### 关键效果数字

以下为论文在真实 Expedia 旅行电商数据集（|J| > 5000 万）上的实验结果，以"相比只看最近一次页面的近视基线（Myopic Baseline）的提升幅度"表示：

| 模型 | AUROC 提升 | AUPRC 提升 | F1 提升 | Acc 提升 |
|------|-----------|-----------|--------|---------|
| **TRACE（本文）** | **+7.23%** | **+13.58%** | **+2.73%** | **+2.15%** |
| 单任务集成（ST Cohort） | +6.38% | +10.75% | +2.72% | +2.06% |
| 单任务嵌入平均（ST Agg） | +6.34% | +10.62% | +2.18% | +1.73% |
| 多任务 LSTM | +1.91% | −3.29% | −0.29% | +0.27% |
| Mini-GPT | +1.86% | −2.40% | −1.13% | −0.60% |

**TRACE 在购买预测任务（PW2）单项 AUROC 提升高达 +11.8%**，超越对应专门训练的单任务模型（+11.2%）。

**推理延迟**：单层 Transformer，Nvidia T4 GPU，均值 **27.5ms**，远低于实时系统 100ms 要求。

---

## ② 母婴出海应用案例

### 场景1：母婴 DTC 站点用户购买意图识别与页面布局优化

**业务问题**：
母婴出海跨境独立站（如婴儿推车、有机奶粉、儿童安全座椅品类）面临典型问题：用户决策周期长（备孕→孕期→育儿长达数年），单次会话转化率极低（通常 <1%），但通过多次访问才能判断哪些用户真正处于购买决策阶段。

具体痛点：
- 无法区分"随机浏览的新妈妈"与"已在比价、即将下单的精准用户"
- 首页、分类页、评测博客页的访问比例不明，无法调整流量引导策略
- 跨境物流页（Shipping Policy）和信任背书页（Certifications）的浏览顺序是否预示转化，缺乏数据支持
- 再营销广告投放预算有限，只能定向给高意图用户，但无法识别

**数据要求**：

| 字段 | 类型 | 示例 |
|------|------|------|
| user_id | string | "usr_abc123" |
| session_id | string | "sess_2026042001" |
| page_name | category | "homepage" / "product_detail" / "cart" / "checkout" / "blog_stroller_review" / "shipping_policy" |
| device_type | category | "mobile" / "desktop" / "tablet" |
| event_timestamp | datetime | "2026-04-20 14:32:05" |
| country | category | "US" / "CA" / "AU" |
| utm_source | category | "instagram" / "google" / "organic" |
| page_dwell_time_sec | float | 45.3 |

最低要求：每用户至少 3 次历史页面访问，覆盖至少 2 个会话；训练集需包含至少 10 万用户（含可观察到的购买事件）。

**预期产出**：
1. 每个活跃用户一个 32 维嵌入向量，每次页面访问后实时更新
2. 5 个预测标签的分数：购买意图分、跳出风险分、搜索意图分等
3. t-SNE 可视化揭示用户所处购买阶段（探索期 / 比较期 / 决策期 / 复购期）
4. 高意图用户（PW2 > 0.7）自动进入再营销受众包

**业务价值**：
- 定向再营销：精准识别高意图用户，将 Facebook/Google 再营销 CPO（每次转化成本）降低 **25-40%**（同品类实际案例经验值）
- 首页内容分流：根据用户所处阶段动态展示"新手妈妈入门指南"或"精选爆款直接购买"，预计提升主页 CTR **15-20%**
- 降低跳出率：BN5 高风险用户立即触发 exit-intent 弹窗（满减优惠券），预计减少跳出 **10-15%**
- 供应链预警：VUO（查看订单）高频用户群体突然激增可预警配送延迟风险

---

## ③ 代码模板

完整可运行的 Python 实现，模拟母婴电商点击流数据并训练 TRACE 模型：

```python
"""
TRACE 母婴电商点击流用户嵌入实现
arXiv: 2409.12972

环境依赖: pip install torch numpy scikit-learn matplotlib
可选依赖: pip install seaborn  # 更美观的可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
import random
import math

# ─────────────────────────────────────────────
# 1. 模拟母婴电商点击流数据生成
# ─────────────────────────────────────────────

# 母婴 DTC 站点页面类型
PAGE_NAMES = [
    "homepage",           # 首页
    "category_stroller",  # 品类页：婴儿推车
    "category_formula",   # 品类页：奶粉
    "category_carseat",   # 品类页：安全座椅
    "product_detail",     # 商品详情页（PDP）
    "blog_review",        # 博客/评测页
    "search_results",     # 搜索结果页（SRP）
    "cart",               # 购物车
    "checkout",           # 结账页
    "order_confirmation", # 订单确认页
    "shipping_policy",    # 配送政策页
    "certifications",     # 认证信任背书页
    "my_orders",          # 我的订单（VUO）
    "wishlist",           # 收藏夹
]

DEVICE_TYPES = ["mobile", "desktop", "tablet"]
UTM_SOURCES = ["instagram", "google", "organic", "email", "tiktok"]

# 购买漏斗阶段 → 影响页面访问概率
USER_STAGE_PROBS = {
    "explorer": {  # 探索期：大量浏览，不太购买
        "homepage": 0.20, "category_stroller": 0.15, "category_formula": 0.10,
        "category_carseat": 0.10, "product_detail": 0.15, "blog_review": 0.15,
        "search_results": 0.08, "cart": 0.03, "checkout": 0.01, 
        "order_confirmation": 0.00, "shipping_policy": 0.01, "certifications": 0.01,
        "my_orders": 0.00, "wishlist": 0.01,
    },
    "evaluator": {  # 比较期：聚焦商品详情、评测
        "homepage": 0.05, "category_stroller": 0.08, "category_formula": 0.05,
        "category_carseat": 0.05, "product_detail": 0.30, "blog_review": 0.20,
        "search_results": 0.12, "cart": 0.08, "checkout": 0.03,
        "order_confirmation": 0.00, "shipping_policy": 0.02, "certifications": 0.01,
        "my_orders": 0.00, "wishlist": 0.01,
    },
    "buyer": {  # 决策期：频繁访问购物车、结账
        "homepage": 0.03, "category_stroller": 0.03, "category_formula": 0.03,
        "category_carseat": 0.03, "product_detail": 0.20, "blog_review": 0.05,
        "search_results": 0.05, "cart": 0.25, "checkout": 0.20,
        "order_confirmation": 0.05, "shipping_policy": 0.05, "certifications": 0.02,
        "my_orders": 0.01, "wishlist": 0.00,
    },
    "returning": {  # 复购期：直接到我的订单页
        "homepage": 0.10, "category_stroller": 0.05, "category_formula": 0.10,
        "category_carseat": 0.05, "product_detail": 0.20, "blog_review": 0.05,
        "search_results": 0.08, "cart": 0.15, "checkout": 0.10,
        "order_confirmation": 0.03, "shipping_policy": 0.01, "certifications": 0.01,
        "my_orders": 0.07, "wishlist": 0.00,
    },
}


def simulate_user_journey(
    user_id: int,
    stage: str = "explorer",
    num_sessions: int = 3,
    pages_per_session: int = 5,
    start_ts: float = 1_700_000_000.0,
    session_gap_hours: float = 24.0,
    within_session_gap_minutes: float = 3.0,
) -> List[Dict]:
    """生成单个用户的多会话点击流"""
    events = []
    ts = start_ts
    probs = USER_STAGE_PROBS[stage]
    pages = list(probs.keys())
    weights = list(probs.values())
    device = random.choice(DEVICE_TYPES)
    utm = random.choice(UTM_SOURCES)

    for s_idx in range(num_sessions):
        n_pages = max(1, int(np.random.poisson(pages_per_session)))
        for p_idx in range(n_pages):
            page = random.choices(pages, weights=weights, k=1)[0]
            events.append({
                "user_id": user_id,
                "session_id": f"u{user_id}_s{s_idx}",
                "session_idx": s_idx,
                "page_name": page,
                "device_type": device,
                "utm_source": utm,
                "timestamp": ts,
            })
            ts += random.expovariate(1.0 / (within_session_gap_minutes * 60))
        # 会话间隔
        ts += session_gap_hours * 3600 * random.uniform(0.5, 2.0)

    return events


def generate_labels(events: List[Dict], stage: str) -> Dict[str, int]:
    """
    根据用户阶段和行为生成 5 个任务标签
    PW2: 2周内购买
    BN5: 下5次页面内跳出（不再访问深层页面）
    SRP: 会话内搜索
    PDP: 会话内浏览商品详情
    VUO: 会话内查看订单
    """
    pages_in_events = [e["page_name"] for e in events]
    
    pw2 = int(stage in ("buyer", "returning") and random.random() < 0.6)
    
    # BN5: 如果最近5次都是浅层页面（homepage/category/blog），判定为跳出风险高
    recent_5 = pages_in_events[-5:] if len(pages_in_events) >= 5 else pages_in_events
    shallow = {"homepage", "category_stroller", "category_formula", "category_carseat", "blog_review"}
    bn5 = int(all(p in shallow for p in recent_5))
    
    srp = int("search_results" in pages_in_events)
    pdp = int("product_detail" in pages_in_events)
    vuo = int("my_orders" in pages_in_events)

    return {"PW2": pw2, "BN5": bn5, "SRP": srp, "PDP": pdp, "VUO": vuo}


def build_dataset(
    n_users: int = 2000,
    max_seq_len: int = 64,
    seed: int = 42,
) -> Tuple[List, List, dict, dict]:
    """构建完整数据集"""
    random.seed(seed)
    np.random.seed(seed)

    stages = ["explorer", "evaluator", "buyer", "returning"]
    stage_weights = [0.4, 0.3, 0.2, 0.1]

    page_encoder = LabelEncoder().fit(PAGE_NAMES)
    device_encoder = LabelEncoder().fit(DEVICE_TYPES)

    all_sequences = []
    all_labels = []

    for uid in range(n_users):
        stage = random.choices(stages, weights=stage_weights, k=1)[0]
        n_sessions = random.randint(2, 6)
        events = simulate_user_journey(uid, stage, num_sessions=n_sessions)
        labels = generate_labels(events, stage)
        
        # 截取最近 max_seq_len 个事件
        events = events[-max_seq_len:]

        all_sequences.append((events, stage))
        all_labels.append(labels)

    return all_sequences, all_labels, {"page": page_encoder, "device": device_encoder}, {"stages": stages}


# ─────────────────────────────────────────────
# 2. PyTorch Dataset
# ─────────────────────────────────────────────

class ClickstreamDataset(Dataset):
    def __init__(
        self,
        sequences: List,
        labels: List[Dict],
        page_encoder: LabelEncoder,
        device_encoder: LabelEncoder,
        max_seq_len: int = 64,
    ):
        self.sequences = sequences
        self.labels = labels
        self.page_encoder = page_encoder
        self.device_encoder = device_encoder
        self.max_seq_len = max_seq_len
        self.n_pages = len(page_encoder.classes_)
        self.n_devices = len(device_encoder.classes_)

    def __len__(self):
        return len(self.sequences)

    def _encode_event(self, event: Dict, prev_ts: float, latest_ts: float) -> np.ndarray:
        """将单次页面访问事件编码为特征向量"""
        # 类别特征：独热索引（后续由 Embedding 层处理，这里只存索引）
        page_idx = self.page_encoder.transform([event["page_name"]])[0]
        device_idx = self.device_encoder.transform([event["device_type"]])[0]

        # 时间特征：时间间隔 + 距最新事件的时间（秒，取对数）
        delta_prev = max(event["timestamp"] - prev_ts, 1.0)
        delta_latest = max(latest_ts - event["timestamp"], 1.0)
        log_delta_prev = math.log(delta_prev)
        log_delta_latest = math.log(delta_latest)

        # 会话位置（第几个最近会话，值越大越老）
        session_idx = event["session_idx"]

        return np.array([page_idx, device_idx, log_delta_prev, log_delta_latest, session_idx], dtype=np.float32)

    def __getitem__(self, idx: int):
        events, stage = self.sequences[idx]
        label_dict = self.labels[idx]

        # 计算时间特征基准
        latest_ts = events[-1]["timestamp"] if events else 0.0
        
        # 编码序列
        features = []
        prev_ts = events[0]["timestamp"] if events else 0.0
        for ev in events:
            feat = self._encode_event(ev, prev_ts, latest_ts)
            features.append(feat)
            prev_ts = ev["timestamp"]

        # Padding 到 max_seq_len
        seq_len = len(features)
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            pad = np.zeros((pad_len, 5), dtype=np.float32)
            features = [pad[i] for i in range(pad_len)] + features  # 前面补 0

        feature_matrix = np.array(features, dtype=np.float32)  # (max_seq_len, 5)
        
        # 事件位置索引（距最近事件第 m 近，从后往前）
        event_positions = np.arange(self.max_seq_len - 1, -1, -1, dtype=np.float32)  # (max_seq_len,)
        
        # 会话位置索引（从 feature_matrix[:, 4] 取）
        session_positions = feature_matrix[:, 4]  # (max_seq_len,)

        # 标签
        labels = np.array(
            [label_dict["PW2"], label_dict["BN5"], label_dict["SRP"], label_dict["PDP"], label_dict["VUO"]],
            dtype=np.float32,
        )

        # 注意力 mask（0 位置为 padding）
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        mask[pad_len:] = True

        return {
            "features": torch.FloatTensor(feature_matrix),
            "event_pos": torch.LongTensor(event_positions.astype(np.int64)),
            "session_pos": torch.LongTensor(np.clip(session_positions, 0, 20).astype(np.int64)),
            "mask": torch.BoolTensor(mask),
            "labels": torch.FloatTensor(labels),
        }


# ─────────────────────────────────────────────
# 3. TRACE 模型架构
# ─────────────────────────────────────────────

class FeatureEncoder(nn.Module):
    """将原始事件特征编码为稠密向量"""
    def __init__(self, n_pages: int, n_devices: int, embed_dim: int = 32, feature_dim: int = 128):
        super().__init__()
        self.page_embed = nn.Embedding(n_pages + 1, embed_dim, padding_idx=0)
        self.device_embed = nn.Embedding(n_devices + 1, embed_dim, padding_idx=0)
        # 时间特征（2 维）投影
        self.time_proj = nn.Linear(2, embed_dim)
        # 合并投影
        self.proj = nn.Linear(embed_dim * 3, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (batch, seq_len, 5)
          - features[:,:,0]: page_idx
          - features[:,:,1]: device_idx
          - features[:,:,2:4]: time features
          - features[:,:,4]: session_idx (handled separately via position encoding)
        """
        page_idx = features[:, :, 0].long()
        device_idx = features[:, :, 1].long()
        time_feats = features[:, :, 2:4]  # (batch, seq, 2)

        page_emb = self.page_embed(page_idx)            # (batch, seq, 32)
        device_emb = self.device_embed(device_idx)      # (batch, seq, 32)
        time_emb = self.time_proj(time_feats)            # (batch, seq, 32)

        combined = torch.cat([page_emb, device_emb, time_emb], dim=-1)  # (batch, seq, 96)
        out = self.norm(torch.relu(self.proj(combined)))                  # (batch, seq, feature_dim)
        return out


class EventSessionPositionEncoding(nn.Module):
    """双重位置编码：事件级 + 会话级"""
    def __init__(self, max_event_pos: int = 128, max_session_pos: int = 32, dim: int = 128):
        super().__init__()
        self.event_embed = nn.Embedding(max_event_pos, dim)
        self.session_embed = nn.Embedding(max_session_pos, dim)

    def forward(self, event_pos: torch.Tensor, session_pos: torch.Tensor) -> torch.Tensor:
        """
        event_pos: (batch, seq_len) - 事件相对位置（0 = 最近）
        session_pos: (batch, seq_len) - 会话索引
        """
        event_enc = self.event_embed(event_pos)    # (batch, seq, dim)
        session_enc = self.session_embed(session_pos)  # (batch, seq, dim)
        return event_enc + session_enc


class TRACEModel(nn.Module):
    """
    TRACE: Transformer-based User Representations from Attributed Clickstream Event sequences
    
    轻量级单层 Transformer Encoder + 双重位置编码 + 5任务 MTL 头
    embedding维度 d=32
    """
    def __init__(
        self,
        n_pages: int,
        n_devices: int,
        feature_dim: int = 128,
        embed_dim: int = 32,
        n_heads: int = 8,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        n_tasks: int = 5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # 特征编码
        self.feature_encoder = FeatureEncoder(n_pages, n_devices, 32, feature_dim)

        # 双重位置编码
        self.pos_encoding = EventSessionPositionEncoding(
            max_event_pos=max_seq_len + 1,
            max_session_pos=33,
            dim=feature_dim,
        )

        # 单层 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 共享 FFN → 用户嵌入
        self.backbone_ffn = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.Sigmoid(),  # 论文中 shared dense layer 使用 sigmoid，确保嵌入有界
        )

        # 5 个任务头（简单 logistic regression 层，逼迫 backbone 学更多信息）
        self.task_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(n_tasks)
        ])

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embed" in name and p.dim() == 2:
                nn.init.normal_(p, mean=0, std=0.01)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_embedding(
        self,
        features: torch.Tensor,
        event_pos: torch.Tensor,
        session_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """获取用户嵌入向量 (batch, embed_dim)"""
        # 特征编码
        x = self.feature_encoder(features)              # (batch, seq, feature_dim)
        # 加入双重位置编码
        pos_enc = self.pos_encoding(event_pos, session_pos)
        x = x + pos_enc

        # Transformer（src_key_padding_mask: True 表示该位置是 padding，需要屏蔽）
        padding_mask = ~mask  # (batch, seq): True = padding, False = real
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (batch, seq, feature_dim)

        # Global Max Pooling（只对真实 token 做 pooling）
        # 将 padding 位置设为极小值再取 max
        mask_expanded = mask.unsqueeze(-1).float()      # (batch, seq, 1)
        x_masked = x * mask_expanded + (1 - mask_expanded) * (-1e9)
        x_pooled = x_masked.max(dim=1)[0]              # (batch, feature_dim)

        # 共享 FFN → embedding
        embedding = self.backbone_ffn(x_pooled)         # (batch, embed_dim)
        return embedding

    def forward(
        self,
        features: torch.Tensor,
        event_pos: torch.Tensor,
        session_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embeddings: (batch, embed_dim)
            logits: (batch, n_tasks)
        """
        embedding = self.get_embedding(features, event_pos, session_pos, mask)
        logits = torch.cat(
            [head(embedding) for head in self.task_heads], dim=-1
        )  # (batch, n_tasks)
        return embedding, logits


# ─────────────────────────────────────────────
# 4. 多任务训练
# ─────────────────────────────────────────────

def compute_class_weights(labels_list: List[Dict]) -> torch.Tensor:
    """计算每个任务的正样本类权重 w_k = 1 / positive_rate"""
    tasks = ["PW2", "BN5", "SRP", "PDP", "VUO"]
    weights = []
    n = len(labels_list)
    for task in tasks:
        pos_count = sum(lbl[task] for lbl in labels_list)
        pos_rate = max(pos_count / n, 0.01)  # 避免除零
        weights.append(1.0 / pos_rate)
    return torch.FloatTensor(weights)


def train_trace(
    n_users: int = 2000,
    max_seq_len: int = 64,
    batch_size: int = 64,
    n_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[TRACEModel, ClickstreamDataset, ClickstreamDataset]:
    """完整训练流程"""
    torch.manual_seed(seed)

    # 数据
    print("🛒 生成母婴电商点击流数据...")
    sequences, labels, encoders, _ = build_dataset(n_users, max_seq_len, seed)

    # 80/20 分割
    split = int(0.8 * len(sequences))
    train_seqs, val_seqs = sequences[:split], sequences[split:]
    train_lbls, val_lbls = labels[:split], labels[split:]

    page_enc = encoders["page"]
    device_enc = encoders["device"]

    train_ds = ClickstreamDataset(train_seqs, train_lbls, page_enc, device_enc, max_seq_len)
    val_ds = ClickstreamDataset(val_seqs, val_lbls, page_enc, device_enc, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 类权重
    class_weights = compute_class_weights(train_lbls).to(device)

    # 模型
    model = TRACEModel(
        n_pages=len(page_enc.classes_),
        n_devices=len(device_enc.classes_),
        feature_dim=128,
        embed_dim=32,
        n_heads=8,
        ffn_dim=128,
        max_seq_len=max_seq_len,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"🔧 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 训练集: {len(train_ds)} 用户, 验证集: {len(val_ds)} 用户")

    for epoch in range(1, n_epochs + 1):
        # ── 训练 ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            event_pos = batch["event_pos"].to(device)
            session_pos = batch["session_pos"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)  # (batch, 5)

            optimizer.zero_grad()
            _, logits = model(features, event_pos, session_pos, mask)  # (batch, 5)

            # 多任务加权 BCE 损失（公式 2）
            loss = 0.0
            for k in range(5):
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits[:, k], y[:, k], pos_weight=class_weights[k:k+1]
                )
                loss = loss + bce
            loss = loss / 5.0

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                event_pos = batch["event_pos"].to(device)
                session_pos = batch["session_pos"].to(device)
                mask = batch["mask"].to(device)
                y = batch["labels"].to(device)

                _, logits = model(features, event_pos, session_pos, mask)
                loss = sum(
                    nn.functional.binary_cross_entropy_with_logits(
                        logits[:, k], y[:, k], pos_weight=class_weights[k:k+1]
                    )
                    for k in range(5)
                ) / 5.0
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"  Epoch {epoch:2d}/{n_epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    print("✅ 训练完成！")
    return model, train_ds, val_ds


# ─────────────────────────────────────────────
# 5. 嵌入可视化（t-SNE / PCA）
# ─────────────────────────────────────────────

def visualize_embeddings(
    model: TRACEModel,
    dataset: ClickstreamDataset,
    n_samples: int = 400,
    method: str = "tsne",
    device: str = "cpu",
):
    """用 t-SNE 或 PCA 可视化用户嵌入，按购买阶段着色"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过可视化")
        return

    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    embeddings = []
    stage_labels = []
    pw2_labels = []

    STAGE_COLORS = {
        "explorer": "#4e79a7",
        "evaluator": "#f28e2b",
        "buyer": "#e15759",
        "returning": "#76b7b2",
    }

    with torch.no_grad():
        for i in indices:
            sample = dataset[i]
            feat = sample["features"].unsqueeze(0).to(device)
            epos = sample["event_pos"].unsqueeze(0).to(device)
            spos = sample["session_pos"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)

            emb = model.get_embedding(feat, epos, spos, mask)
            embeddings.append(emb.squeeze(0).cpu().numpy())
            stage_labels.append(dataset.sequences[i][1])
            pw2_labels.append(dataset.labels[i]["PW2"])

    embeddings = np.array(embeddings)

    # 降维
    if method == "tsne" and len(embeddings) > 50:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
        coords = reducer.fit_transform(embeddings)
        title = "TRACE 嵌入 t-SNE 投影（母婴电商用户）"
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        title = "TRACE 嵌入 PCA 投影（母婴电商用户）"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：按购买阶段着色
    ax1 = axes[0]
    for stage, color in STAGE_COLORS.items():
        mask_s = [i for i, s in enumerate(stage_labels) if s == stage]
        if mask_s:
            ax1.scatter(
                coords[mask_s, 0], coords[mask_s, 1],
                c=color, label=stage, alpha=0.7, s=20, edgecolors="none"
            )
    ax1.set_title(f"{title}\n（按用户购买阶段着色）", fontsize=11)
    ax1.legend(title="购买阶段", fontsize=8)
    ax1.set_xlabel("维度1"); ax1.set_ylabel("维度2")

    # 右图：按 PW2（2周内购买）标签着色
    ax2 = axes[1]
    colors2 = ["#aec7e8" if p == 0 else "#d62728" for p in pw2_labels]
    ax2.scatter(coords[:, 0], coords[:, 1], c=colors2, alpha=0.7, s=20, edgecolors="none")
    from matplotlib.patches import Patch
    handles = [Patch(color="#aec7e8", label="未购买"), Patch(color="#d62728", label="2周内购买")]
    ax2.legend(handles=handles, title="PW2 标签", fontsize=8)
    ax2.set_title(f"{title}\n（按2周内购买标签着色）", fontsize=11)
    ax2.set_xlabel("维度1"); ax2.set_ylabel("维度2")

    plt.tight_layout()
    output_path = "/tmp/trace_embeddings_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 嵌入可视化已保存到: {output_path}")
    return coords, stage_labels


# ─────────────────────────────────────────────
# 6. 测试用例
# ─────────────────────────────────────────────

def run_tests():
    """验证 TRACE Pipeline 的核心功能"""
    print("\n" + "="*60)
    print("🧪 运行测试用例")
    print("="*60)

    DEVICE = "cpu"
    MAX_SEQ = 32
    BATCH = 4

    # ── 测试1：数据生成 ──
    print("\n[Test 1] 数据生成")
    seqs, lbls, encs, _ = build_dataset(n_users=100, max_seq_len=MAX_SEQ, seed=0)
    assert len(seqs) == 100, "用户数量应为 100"
    assert len(lbls) == 100, "标签数量应为 100"
    tasks = ["PW2", "BN5", "SRP", "PDP", "VUO"]
    for t in tasks:
        assert t in lbls[0], f"标签中缺少任务 {t}"
        assert lbls[0][t] in (0, 1), f"任务 {t} 标签应为 0/1"
    print("  ✓ 100 个用户序列生成成功，5 个任务标签完整")

    # ── 测试2：Dataset & DataLoader ──
    print("\n[Test 2] Dataset 与 DataLoader")
    ds = ClickstreamDataset(seqs, lbls, encs["page"], encs["device"], MAX_SEQ)
    sample = ds[0]
    assert sample["features"].shape == (MAX_SEQ, 5), \
        f"特征矩阵形状应为 ({MAX_SEQ}, 5)，实际 {sample['features'].shape}"
    assert sample["labels"].shape == (5,), f"标签形状应为 (5,)，实际 {sample['labels'].shape}"
    assert sample["mask"].shape == (MAX_SEQ,), "mask 形状错误"
    print(f"  ✓ features: {sample['features'].shape}, labels: {sample['labels'].shape}")

    loader = DataLoader(ds, batch_size=BATCH)
    batch = next(iter(loader))
    assert batch["features"].shape == (BATCH, MAX_SEQ, 5), "batch features 形状错误"
    print(f"  ✓ DataLoader batch: {batch['features'].shape}")

    # ── 测试3：模型前向传播 ──
    print("\n[Test 3] 模型前向传播")
    model = TRACEModel(
        n_pages=len(encs["page"].classes_),
        n_devices=len(encs["device"].classes_),
        feature_dim=64,
        embed_dim=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=MAX_SEQ,
    ).to(DEVICE)

    feat = batch["features"].to(DEVICE)
    epos = batch["event_pos"].to(DEVICE)
    spos = batch["session_pos"].to(DEVICE)
    mask = batch["mask"].to(DEVICE)

    emb, logits = model(feat, epos, spos, mask)
    assert emb.shape == (BATCH, 32), f"嵌入维度应为 (batch, 32)，实际 {emb.shape}"
    assert logits.shape == (BATCH, 5), f"logits 应为 (batch, 5)，实际 {logits.shape}"
    assert emb.min() >= 0 and emb.max() <= 1, "Sigmoid 激活应保证嵌入在 [0,1] 内"
    print(f"  ✓ 嵌入: {emb.shape}, logits: {logits.shape}, 嵌入范围: [{emb.min():.3f}, {emb.max():.3f}]")

    # ── 测试4：损失计算 ──
    print("\n[Test 4] 多任务损失计算")
    y = batch["labels"].to(DEVICE)
    weights = compute_class_weights(lbls).to(DEVICE)
    total_loss = sum(
        nn.functional.binary_cross_entropy_with_logits(
            logits[:, k], y[:, k], pos_weight=weights[k:k+1]
        )
        for k in range(5)
    ) / 5.0
    assert not torch.isnan(total_loss), "损失不应为 NaN"
    assert total_loss.item() > 0, "损失应大于 0"
    print(f"  ✓ 多任务损失: {total_loss.item():.4f}")

    # ── 测试5：梯度流动 ──
    print("\n[Test 5] 梯度流动验证")
    total_loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0, "梯度应能正常反向传播"
    print(f"  ✓ 总梯度 L2 范数: {grad_norm:.4f}")

    # ── 测试6：嵌入一致性（相同输入应产生相同嵌入）──
    print("\n[Test 6] 嵌入推理一致性")
    model.eval()
    with torch.no_grad():
        emb1 = model.get_embedding(feat[:1], epos[:1], spos[:1], mask[:1])
        emb2 = model.get_embedding(feat[:1], epos[:1], spos[:1], mask[:1])
    assert torch.allclose(emb1, emb2), "相同输入应产生相同嵌入"
    print(f"  ✓ 确定性推理通过")

    print("\n" + "="*60)
    print("✅ 所有 6 个测试通过！")
    print("="*60)
    return True


# ─────────────────────────────────────────────
# 7. 主入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 运行测试
    run_tests()

    print("\n" + "="*60)
    print("🚀 开始完整训练（母婴电商 TRACE 示例）")
    print("="*60)

    # 快速演示训练（减少 epoch 以加速）
    model, train_ds, val_ds = train_trace(
        n_users=1000,
        max_seq_len=32,
        batch_size=32,
        n_epochs=5,
        lr=1e-3,
        device="cpu",
        seed=42,
    )

    # 嵌入可视化
    print("\n📊 生成嵌入可视化...")
    visualize_embeddings(model, val_ds, n_samples=300, method="pca", device="cpu")

    print("\n💡 如何将嵌入用于下游任务：")
    print("  1. 导出所有用户嵌入 → 存入特征平台（如 Redis/Feast）")
    print("  2. 用 XGBoost 在嵌入上训练购买预测分类器")
    print("  3. 实时更新：每次页面访问后触发 TRACE 推理（< 30ms），更新用户向量")
    print("  4. 相似用户检索：用余弦相似度在 32 维嵌入空间做 ANN 搜索")
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [[Skill-User-Funnel-Analysis]] | TRACE 的目标之一（PW2、BN5）正是漏斗分析的关键节点，需要先理解转化漏斗结构才能正确定义预测任务 |
| 前置 | [[Skill-Customer-Journey-Prototype]] | TRACE 处理的"旅程 Journey"概念与用户旅程地图高度对应，业务分析先行可指导选择哪些页面作为关键节点 |
| 延伸 | Skill-STAN-User-Lifecycle | TRACE 产出的用户嵌入可以作为 STAN 用户生命周期分类模型的输入特征，两者形成"嵌入表示→阶段分类"的 Pipeline |
| 延伸 | [[Skill-PersonaBot-RAG-Profiling]] | TRACE 学到的用户群簇（t-SNE 可视化揭示的 latent states）可以喂给 PersonaBot 生成对应 Persona 描述 |
| 组合 | [[Skill-Cohort-Retention-Analysis]] | TRACE 嵌入可以作为留存分析的协变量，识别哪些用户行为模式与 7 日/30 日留存率强相关 |

---

- **前置技能**：[[Skill-User-Funnel-Analysis]] | [[Skill-Customer-Journey-Prototype]]
- **延伸技能**：[[Skill-Cohort-Retention-Analysis]]
- **可组合技能**：[[Skill-RFM-User-Segmentation]] | [[Skill-Session-Intent-Shift]]
- **相关技能**：[[Skill-NonItem-Page-Path-Modeling]]
- **相关**：[[Skill-Ad-to-Behavior-Funnel]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| ROI 预估 | ⭐⭐⭐⭐☆ | Expedia 实测 AUROC 提升 7%，等效于再营销 CPO 降低 25-35%；母婴品类高客单价（$150-$500）使每1%的转化提升具有显著营收价值 |
| 实施难度 | ⭐⭐⭐☆☆ | 需要多会话点击流埋点（通常已有但未整合）、PyTorch 推理服务部署；单层 Transformer 推理 <30ms，工程友好；主要挑战在数据清洗和多会话拼接 |
| 优先级 | ⭐⭐⭐⭐☆ | 母婴用户决策周期长（2-8周），正是 TRACE 跨会话建模的最佳场景；再营销广告成本占 DTC 站 CAC 的 30-50%，精准识别意图的杠杆效应极高 |
| 数据门槛 | ⭐⭐⭐☆☆ | 需要至少 10 万活跃用户的历史点击流 + 可观察的购买事件；冷启动用户（历史 <3 次访问）需退化到规则引导 |
| 可复用性 | ⭐⭐⭐⭐⭐ | 架构与页面名称无关，更换页面词汇表即可迁移到任何垂类电商；多任务目标定义灵活，可替换为母婴垂类特有标签（如"添加收藏""开启比价模式"等）|

**推荐使用场景优先级：**
1. 🥇 **再营销受众分层**：区分"高意图买家"与"随机浏览者"，集中预算打高价值用户
2. 🥈 **首屏内容动态化**：根据嵌入预测的用户阶段，个性化展示探索型/决策型内容
3. 🥉 **跳出率干预**：BN5 风险用户实时触发优惠弹窗或推荐
4. 🏅 **商品排序个性化**：将 32 维嵌入与商品特征一起输入深度排序模型（如 DIN/DSSM）
