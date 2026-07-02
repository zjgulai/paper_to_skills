---
title: 图神经网络欺诈检测 — 关系图结构感知的电商欺诈识别
doc_type: knowledge
module: 19-风控反欺诈
topic: gnn-fraud-detection
status: stable
created: 2026-07-02
updated: 2026-07-02
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: GNN Fraud Detection

> **论文**：Fraud Detection under Multi-Relational Graph（Rao et al., KDD 2021, arXiv:2105.13433）+ Graph Neural Networks for Fraud Detection in E-Commerce（Zhang et al., WWW 2022）
> **arXiv**：2105.13433 | 2021 | **桥梁**: 19-风控反欺诈 ↔ 08-知识图谱 ↔ 04-供应链 | **类型**: 跨域融合

## ① 算法原理

**传统欺诈检测的致命弱点**：只看单个账号的行为特征（购买频次、账号年龄等），忽略了**账号之间的关系网络**。职业欺诈团伙的最大暴露点恰恰在关系网络上：
- 多个账号共享同一设备ID / IP段
- 收货地址在某几个"中转站"地址之间循环使用
- 账号注册时间、购买商品高度相似

**GNN（图神经网络）欺诈检测框架**：

**Step 1：构建多关系欺诈图**
$$G = \{V, E^{(1)}, E^{(2)}, ..., E^{(K)}\}$$
- 节点 $V$：账号/订单/设备/地址
- 边类型（多关系）：
  - $E^{device}$：共享设备ID的账号对
  - $E^{address}$：共享收货地址的账号对
  - $E^{ip}$：同一IP段注册的账号对
  - $E^{timing}$：注册时间窗口内的账号对

**Step 2：图注意力网络（GAT）聚合邻居信息**
$$h_v^{(k+1)} = \sigma\left(\sum_{r \in R} \sum_{u \in N_r(v)} \alpha_{r,vu} W_r h_u^{(k)}\right)$$
其中 $\alpha_{r,vu}$ 是关系 $r$ 下节点 $v$ 对邻居 $u$ 的注意力权重——自动学习不同关系的重要性。

**Step 3：标签传播**
即使只有少量标注欺诈节点（正样本稀少），GNN可以通过**图结构传播**发现"与已知欺诈账号高度连接的未知节点"——这是传统ML无法做到的。

**多关系图的关键优势**：
在单一关系图中，欺诈团伙可能伪造部分特征绕过检测，但在**多种关系同时考虑**下，几乎不可能在所有关系上都表现正常。

## ② 母婴出海应用案例

**场景A：刷单欺诈团伙图识别**
- 业务问题：出现大量"婴儿床"刷单订单（虚假好评），传统规则检测只发现了部分账号；已确认的欺诈账号中，有些与大量未被标记的账号共享设备/IP/地址，形成欺诈网络
- 数据要求：订单记录（含设备ID/收货地址/注册IP）+ 已标注欺诈账号（种子）+ 账号行为特征
- 预期产出：从12个已知欺诈账号出发，通过图传播发现完整欺诈团伙78个账号；精确率=89%（其中70个确认欺诈，8个误报），召回率从传统方法的15%提升至72%
- 业务价值：识别并封停完整欺诈团伙，减少刷单GMV浪费约50万元/年；防止虚假好评对搜索排名的污染，保护品牌的自然流量

**三轨对抗验证**：
1. **成本验证**：图构建可在标准数据库上完成（无需图数据库）；GNN训练在中等规模图（10万节点）上约30分钟（CPU）；更新图约每日一次即可
2. **合规验证**：分析账号关系是合法的风控行为；注意不可将"与欺诈账号共享IP"作为封号的唯一依据（会误伤共享网络的正常用户），需多因子确认
3. **风险验证**：图构建的边类型选择对结果影响显著；建议先做"边质量评估"（相同边类型下，欺诈账号的邻居中欺诈比例）；大图上GNN训练需要MiniBatch采样（GraphSAGE）

## ③ 代码模板

```python
"""
Skill-GNN-Fraud-Detection
图神经网络欺诈检测 — 多关系欺诈团伙图识别

依赖：pip install numpy pandas scikit-learn scipy
注意：完整GNN需 PyTorch Geometric；此处为图传播的简化实现
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from collections import defaultdict

np.random.seed(42)

# ── 1. 生成模拟欺诈账号图数据 ──────────────────────────────────────────
n_accounts = 5000

# 账号特征
account_age_days  = np.random.exponential(200, n_accounts)
purchase_count    = np.random.poisson(3, n_accounts).astype(float)
return_rate       = np.random.beta(1, 10, n_accounts)
device_id         = np.random.randint(0, 3000, n_accounts)  # 设备ID
address_cluster   = np.random.randint(0, 4000, n_accounts)  # 地址聚类
ip_segment        = np.random.randint(0, 5000, n_accounts)  # IP段

# 注入欺诈团伙（5个团伙，每团伙10-30个账号）
fraud_accounts = set()
gang_profiles  = []
for gang_id in range(5):
    gang_size = np.random.randint(10, 30)
    # 团伙共享特征
    shared_device  = np.random.randint(0, 100)  # 团伙使用的设备
    shared_address = np.random.randint(0, 100)  # 团伙使用的地址
    shared_ip      = np.random.randint(0, 100)  # 团伙使用的IP段

    gang_members = np.random.choice(n_accounts, gang_size, replace=False)
    fraud_accounts.update(gang_members)

    # 覆盖团伙成员的图特征
    for m in gang_members:
        if np.random.random() < 0.7:  # 70%概率共享设备
            device_id[m] = shared_device
        if np.random.random() < 0.6:  # 60%概率共享地址
            address_cluster[m] = shared_address
        if np.random.random() < 0.8:  # 80%概率同IP段
            ip_segment[m] = shared_ip
    gang_profiles.append(gang_members)

labels = np.zeros(n_accounts, dtype=int)
for gang in gang_profiles: labels[gang] = 1

print(f"账号数: {n_accounts} | 欺诈账号: {labels.sum()} ({labels.mean():.1%})")

# ── 2. 构建多关系欺诈图 ──────────────────────────────────────────────
def build_adjacency_from_shared_feature(feature_array, min_group_size=2, max_group_size=10):
    """
    根据共享特征值构建边（共享同一特征值 → 添加边）
    只构建合理规模的组（太大的组可能是公共特征，非欺诈）
    """
    groups = defaultdict(list)
    for i, val in enumerate(feature_array):
        groups[val].append(i)
    
    edges = []
    for members in groups.values():
        if min_group_size <= len(members) <= max_group_size:
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    edges.append((members[i], members[j]))
    return edges

edges_device  = build_adjacency_from_shared_feature(device_id,  max_group_size=8)
edges_address = build_adjacency_from_shared_feature(address_cluster, max_group_size=6)
edges_ip      = build_adjacency_from_shared_feature(ip_segment, max_group_size=12)

# 合并多关系边（去重）
all_edges = list(set(edges_device + edges_address + edges_ip))

# 构建邻接表
adj = defaultdict(set)
for u, v in all_edges:
    adj[u].add(v)
    adj[v].add(u)

avg_degree = np.mean([len(nb) for nb in adj.values()])
print(f"\n图结构: {len(all_edges)}条边 | 平均度数={avg_degree:.1f}")
print(f"  设备关系边: {len(edges_device)} | 地址关系边: {len(edges_address)} | IP关系边: {len(edges_ip)}")

# ── 3. 简化图神经传播（Label Propagation）────────────────────────────
# 用已知欺诈种子（10%）通过图传播扩散欺诈分数
def graph_label_propagation(adj, seed_labels, n_iterations=3, damping=0.7):
    """
    简化版标签传播（近似GNN聚合）
    seed_labels: 已知欺诈账号的标签（0或1）
    """
    n = len(seed_labels)
    scores = seed_labels.astype(float).copy()
    
    for iteration in range(n_iterations):
        new_scores = np.zeros(n)
        for node in range(n):
            neighbors = adj[node]
            if neighbors:
                neighbor_avg = np.mean([scores[nb] for nb in neighbors])
                new_scores[node] = damping * scores[node] + (1-damping) * neighbor_avg
            else:
                new_scores[node] = scores[node]
        scores = new_scores
    
    return scores

# 只用10%的已知欺诈账号作为种子
known_mask = np.zeros(n_accounts, dtype=bool)
known_fraud_idx = np.where(labels == 1)[0]
seed_size = max(1, int(len(known_fraud_idx) * 0.1))
known_mask[np.random.choice(known_fraud_idx, seed_size, replace=False)] = True

seed_labels = np.zeros(n_accounts)
seed_labels[known_mask] = 1.0

# 图传播
propagated_scores = graph_label_propagation(adj, seed_labels, n_iterations=3)

# ── 4. 融合图特征 + 行为特征（最终分类）─────────────────────────────
# 特征：行为特征 + 图传播分数 + 图度数特征
degree_arr = np.array([len(adj[i]) for i in range(n_accounts)])
X_with_graph = np.column_stack([
    account_age_days,
    purchase_count,
    return_rate,
    propagated_scores,          # 图传播欺诈分数
    degree_arr,                  # 图度数（连接越多越可疑）
])
X_without_graph = np.column_stack([
    account_age_days, purchase_count, return_rate
])

# 留出已知种子外的账号做评估
eval_mask = ~known_mask
X_eval   = X_with_graph[eval_mask]
X_eval_ng = X_without_graph[eval_mask]
y_eval   = labels[eval_mask]

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_eval, y_eval, test_size=0.3, stratify=y_eval, random_state=42)
X_tr_ng, X_te_ng, _, _ = train_test_split(X_eval_ng, y_eval, test_size=0.3, stratify=y_eval, random_state=42)

clf_with_gnn = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_with_gnn.fit(X_tr, y_tr)

clf_no_gnn = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_no_gnn.fit(X_tr_ng, y_tr)

auc_with = roc_auc_score(y_te, clf_with_gnn.predict_proba(X_te)[:,1])
auc_no   = roc_auc_score(y_te, clf_no_gnn.predict_proba(X_te_ng)[:,1])

print(f"\n【图特征效果对比】")
print(f"  无图特征  AUC: {auc_no:.4f}")
print(f"  含图传播  AUC: {auc_with:.4f}  (提升{auc_with-auc_no:+.4f})")

y_pred = clf_with_gnn.predict(X_te)
report = classification_report(y_te, y_pred, target_names=['正常','欺诈'], output_dict=True)
print(f"  欺诈召回率: {report['欺诈']['recall']:.2%}")
print(f"  欺诈精确率: {report['欺诈']['precision']:.2%}")
print(f"\n  图特征把AUC从{auc_no:.3f}提升到{auc_with:.3f} (+{(auc_with-auc_no)*100:.1f}pp)")
print(f"  关键原因: 欺诈团伙成员通过共享设备/地址/IP暴露了团伙关系")

assert auc_with > 0.6, f"含图特征的AUC过低: {auc_with:.4f}"
print("\n[✓] GNN欺诈检测 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-GNN-Foundations]]（图神经网络基础）、[[Skill-Review-Fraud-Detection]]（评论欺诈检测的配套场景）
- **延伸（extends）**：[[Skill-DS-DGA-GCN-Fake-Review-Group-Detection]]（专门用于虚假评论团伙GNN检测）
- **可组合（combinable）**：[[Skill-Class-Conditional-Generation-Augment]]（欺诈数据增强 + GNN检测双保险）、[[Skill-Identity-Fraud-Detection]]（身份欺诈检测中图关系的应用）、[[Skill-Multimodal-Fake-Review-Detection]]（多模态 + 图结构双维度欺诈检测）

## ⑤ 商业价值评估

- **ROI 预估**：欺诈团伙召回率从15%提升至72%（+57%），识别完整团伙而非零散账号；年化减少刷单损失约50万元；防止搜索排名污染保护自然流量价值约30万元；综合约80万元/年
- **实施难度**：⭐⭐⭐⭐☆（图构建简单，GNN训练需PyTorch Geometric；主要挑战是构建准确的关系图边）
- **优先级**：⭐⭐⭐⭐⭐（修复19-风控↔08-知识图谱弱桥梁；团伙欺诈是亚马逊卖家最大的威胁之一，传统方法无法识别）
- **评估依据**：KDD 2021 FRAUDRE是该方向最重要论文，引用量200+；阿里/京东/Amazon已在生产部署图神经欺诈检测；PyTorch Geometric已内置多种欺诈检测GNN实现
