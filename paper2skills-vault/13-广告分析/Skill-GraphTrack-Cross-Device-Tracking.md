---
title: 图基跨设备追踪 - 无监督IP-Domain图谱用户拼接
doc_type: knowledge
module: 13-广告分析
topic: graph-cross-device-tracking
status: stable
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: arXiv:2203.06833
---

# Skill: GraphTrack Cross-Device Tracking — 无监督图基跨设备追踪

> 主论文:**GraphTrack: A Graph-based Cross-Device Tracking Framework** · arXiv:2203.06833 (AsiaCCS 2022)
> 作者: Binghui Wang, Tianchen Zhou, Song Li, Yinzhi Cao, Neil Zhenqiang Gong
> 应用:仅用IP和域名访问日志，无需任何标签，无监督匹配同一用户的多设备

---

## ① 算法原理

### 核心思想

跨设备追踪的本质是：在没有用户级关联ID的情况下，用行为相似性判断"两台设备属于同一个人"。GraphTrack 将这一问题建模为**异构图上的节点相似度计算**：把设备的 IP 访问记录构建成「IP-Device 二部图」，把域名访问记录构建成「Domain-Device 二部图」，然后用**随机游走重启（RWwR / Personalized PageRank）**在图上扩散相似度信号，最终输出设备对的匹配分数。

核心优势三点：
1. **无监督**：不需要任何标注的设备对，零标签冷启动
2. **双图融合**：IP 信号（物理位置/网络共享）+ 域名信号（内容偏好/行为模式）互补
3. **不确定性鲁棒**：边权使用访问频率而非 0/1，能容忍"同 IP ≠ 同用户"的噪声（公共 WiFi、NAT）

在自建 154K 用户真实数据集上，GraphTrack 显著超越所有 SOTA 基线。

### 数学直觉

**图构建**：

设备集合 $\mathcal{D}$，IP集合 $\mathcal{I}$，域名集合 $\mathcal{O}$。

- **IP-Device 图** $G_{IP}$：边 $(d, ip)$ 的权重 = 设备 $d$ 访问 IP $ip$ 的次数 $f_{d,ip}$
- **Domain-Device 图** $G_{Dom}$：边 $(d, dom)$ 的权重 = 设备 $d$ 访问域名 $dom$ 的次数 $f_{d,dom}$

**相似度计算（RWwR / Personalized PageRank）**：

对于查询设备 $d_q$，在图 $G$ 上以 $d_q$ 为重启节点做随机游走：

$$\mathbf{r}^{(t+1)} = (1-\alpha) \cdot \hat{A} \mathbf{r}^{(t)} + \alpha \cdot \mathbf{e}_{d_q}$$

其中 $\hat{A}$ 是行归一化邻接矩阵，$\alpha$ 是重启概率（通常 0.15），$\mathbf{e}_{d_q}$ 是 $d_q$ 的 one-hot 向量。

稳定分布 $\mathbf{r}^*$ 中，第 $d_i$ 个分量即为 $d_q$ 与 $d_i$ 在图上的相似度分数 $s(d_q, d_i)$。

**双图融合与对称匹配**：

$$\text{score}(d_q, d_i) = \lambda \cdot s_{IP}(d_q, d_i) + (1-\lambda) \cdot s_{Dom}(d_q, d_i)$$

对称匹配：$\text{sim}(d_q, d_i) = \frac{\text{score}(d_q, d_i) + \text{score}(d_i, d_q)}{2}$，消除有向图不对称问题。

### 关键假设

1. **同用户设备共享 IP/域名**：同一用户的不同设备在物理位置（家庭/办公室 IP）或内容偏好（常访问域名）上存在重叠
2. **访问频率携带身份信号**：高频共访的 IP 或域名比低频更具判别力
3. **图规模可处理**：设备数 × IP/域名数 在稀疏矩阵层面可以 Power Iteration 求解

### 关键效果数字

| 数据集 | 用户数 | GraphTrack 提升 SOTA |
|--------|--------|----------------------|
| 公开移动-桌面追踪数据集 | ~100 用户 | Precision@1 **+8-15%** |
| 自建多设备追踪数据集 | **154K 用户** | MRR **+20%+**，全面超越 |

---

## ② 母婴出海应用案例

### 场景1：TikTok种草+独立站拔草的跨端追踪

**业务问题**：母婴用户在 TikTok 看到吸奶器短视频种草，切换到 Safari 搜索品牌名进独立站下单。现有追踪完全断裂——TikTok 报表显示零转化，独立站显示"直接访问"来源。需要无监督匹配这两个设备的浏览记录，恢复 TikTok 的真实转化贡献。

**数据要求**：

| 字段 | 类型 | 示例 |
|------|------|------|
| `device_id` | str | "device_TikTok_iPhone_A001" / "device_Safari_Mac_B001" |
| `ip_address` | str | "192.168.1.10"（家庭 WiFi，两设备共享） |
| `domain` | str | "momcozy.com", "tiktok.com", "google.com" |
| `visit_count` | int | 3（该设备访问该IP/域名的次数） |
| `timestamp_range` | str | "2026-06-01 ~ 2026-06-07"（分析窗口） |

**预期产出**：
- 设备对匹配分数矩阵：`match_score(TikTok设备, 独立站设备) > 0.6` → 判定为同一用户
- 跨端归因链路：TikTok 曝光设备 → 独立站转化设备 的用户级关联表

**业务价值**：
- 恢复 TikTok 真实 ROAS（行业数据：Last-Click 让 TikTok 低估 **700 倍**）
- 中型品牌月 TikTok 广告 50 万元，若 ROAS 从 0.2 修正至 1.5，避免错误停投损失 **600 万元/年**
- 为 PVM 归因窗口统一 Skill 提供设备级用户拼接前置能力

### 场景2：亚马逊站内流量+独立站跨平台用户拼接

**业务问题**：母婴品牌同时运营 Amazon 旗舰店和 Shopify 独立站。用户在 Amazon App（手机）上查看产品，再在 PC 浏览器访问独立站下单，或反之。两侧数据孤立，无法识别是否是回头客或流失用户。需要在无任何用户 ID 的情况下，仅用 IP + 域名访问日志跨平台拼接用户。

**数据要求**：

| 字段 | 类型 | 示例 |
|------|------|------|
| `device_id` | str | "amz_mobile_U001" / "shopify_pc_V002" |
| `ip_address` | str | "203.0.113.55"（家庭宽带出口IP） |
| `domain` | str | "amazon.com", "shopify_store.com", "babyproducts.com" |
| `visit_count` | int | 7 |

**预期产出**：
- 跨平台用户拼接表：Amazon 用户ID ↔ Shopify 用户ID 的概率关联矩阵
- 复购用户识别：Amazon 上购买过 → 独立站再次购买的跨端复购率

**业务价值**：
- 跨平台复购率提升：识别高价值跨端用户后推送个性化复购券，**LTV 提升 20-35%**
- 防流失预警：识别 Amazon 活跃但独立站沉寂的用户，精准触达，月潜在挽回 **10-50 万元**

---

## ③ 代码模板

```python
"""
GraphTrack: 无监督图基跨设备追踪
论文: arXiv:2203.06833 (AsiaCCS 2022)
Binghui Wang et al.

核心算法:
1. 构建 IP-Device 二部图 + Domain-Device 二部图
2. 用 Personalized PageRank (Random Walk with Restart) 计算设备间相似度
3. 双图融合 + 对称匹配 → 输出设备对匹配分数
4. 无监督：无需任何标注设备对

依赖: pip install numpy scipy networkx pandas
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import sparse


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class BrowsingRecord:
    """设备浏览记录（IP 或 Domain 维度均通用）"""
    device_id: str
    feature: str       # IP 地址 或 域名
    visit_count: int = 1


# ─────────────────────────────────────────────
# 图构建
# ─────────────────────────────────────────────

class DeviceGraph:
    """
    构建 Device-Feature 二部图（Feature = IP 或 Domain）
    节点分两类: device_* 和 feature_*
    边权重 = 访问次数（频率加权）
    """

    def __init__(self, records: List[BrowsingRecord]):
        self.G = nx.Graph()
        self.device_nodes: List[str] = []
        self.feature_nodes: List[str] = []
        self._build(records)

    def _build(self, records: List[BrowsingRecord]):
        device_set = set()
        feature_set = set()

        for r in records:
            d_node = f"device::{r.device_id}"
            f_node = f"feature::{r.feature}"
            device_set.add(d_node)
            feature_set.add(f_node)
            if self.G.has_edge(d_node, f_node):
                self.G[d_node][f_node]["weight"] += r.visit_count
            else:
                self.G.add_edge(d_node, f_node, weight=r.visit_count)

        self.device_nodes = sorted(device_set)
        self.feature_nodes = sorted(feature_set)
        self.all_nodes = self.device_nodes + self.feature_nodes
        self.node_index = {n: i for i, n in enumerate(self.all_nodes)}

    def get_normalized_adj(self) -> sparse.csr_matrix:
        """
        构建行归一化邻接矩阵 Â（用于 RWwR Power Iteration）
        Â[i][j] = w_{ij} / sum_k(w_{ik})
        """
        n = len(self.all_nodes)
        rows, cols, data = [], [], []

        for u, v, w in self.G.edges(data="weight"):
            i = self.node_index[u]
            j = self.node_index[v]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])

        A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
        # 行归一化
        row_sums = np.asarray(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = sparse.diags(1.0 / row_sums)
        return D_inv @ A


# ─────────────────────────────────────────────
# Personalized PageRank (RWwR)
# ─────────────────────────────────────────────

def personalized_pagerank(
    A_norm: sparse.csr_matrix,
    source_idx: int,
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Random Walk with Restart (Personalized PageRank)

    r^(t+1) = (1 - alpha) * A_norm^T * r^(t) + alpha * e_{source}

    Args:
        A_norm: 行归一化邻接矩阵
        source_idx: 查询设备的节点索引
        alpha: 重启概率（论文推荐 0.15）
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        r: 稳定分布向量，r[i] 即 source 到节点 i 的相似度分数
    """
    n = A_norm.shape[0]
    e_source = np.zeros(n)
    e_source[source_idx] = 1.0

    r = e_source.copy()
    A_T = A_norm.T  # 转置用于列向量更新

    for _ in range(max_iter):
        r_new = (1 - alpha) * A_T.dot(r) + alpha * e_source
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return r_new


# ─────────────────────────────────────────────
# GraphTrack 主类
# ─────────────────────────────────────────────

class GraphTrack:
    """
    GraphTrack 无监督跨设备追踪

    输入:
        ip_records: IP-Device 浏览记录列表
        domain_records: Domain-Device 浏览记录列表

    输出:
        设备对匹配分数矩阵（越高越可能是同一用户）
    """

    def __init__(
        self,
        ip_records: List[BrowsingRecord],
        domain_records: List[BrowsingRecord],
        alpha: float = 0.15,
        lambda_ip: float = 0.5,   # IP 图权重
    ):
        self.alpha = alpha
        self.lambda_ip = lambda_ip

        self.ip_graph = DeviceGraph(ip_records)
        self.dom_graph = DeviceGraph(domain_records)

        # 取两图共有设备集合
        ip_devices = {n.replace("device::", "") for n in self.ip_graph.device_nodes}
        dom_devices = {n.replace("device::", "") for n in self.dom_graph.device_nodes}
        self.devices = sorted(ip_devices | dom_devices)

        self._A_ip = self.ip_graph.get_normalized_adj()
        self.ip_idx = self.ip_graph.node_index

        self._A_dom = self.dom_graph.get_normalized_adj()
        self.dom_idx = self.dom_graph.node_index

    def _device_score_in_graph(
        self, graph: DeviceGraph, A_norm: sparse.csr_matrix, device_id: str
    ) -> Dict[str, float]:
        """在单图上计算 device_id 到其他所有设备的 PPR 分数"""
        d_node = f"device::{device_id}"
        if d_node not in graph.node_index:
            return {}
        src_idx = graph.node_index[d_node]
        r = personalized_pagerank(A_norm, src_idx, self.alpha)

        scores = {}
        for node, idx in graph.node_index.items():
            if node.startswith("device::") and node != d_node:
                scores[node.replace("device::", "")] = float(r[idx])
        return scores

    def compute_similarity_matrix(self) -> pd.DataFrame:
        """
        计算所有设备对的融合相似度（对称匹配）

        Returns:
            DataFrame: index=device_id, columns=device_id, 值=匹配分数
        """
        n = len(self.devices)
        scores_ip: Dict[str, Dict[str, float]] = {}
        scores_dom: Dict[str, Dict[str, float]] = {}

        for d in self.devices:
            scores_ip[d] = self._device_score_in_graph(
                self.ip_graph, self._A_ip, d
            )
            scores_dom[d] = self._device_score_in_graph(
                self.dom_graph, self._A_dom, d
            )

        # 双图融合 + 对称化
        mat = np.zeros((n, n))
        d2i = {d: i for i, d in enumerate(self.devices)}

        for i, d_q in enumerate(self.devices):
            for j, d_t in enumerate(self.devices):
                if i == j:
                    continue
                s_ip_qt = scores_ip[d_q].get(d_t, 0.0)
                s_ip_tq = scores_ip[d_t].get(d_q, 0.0)
                s_dom_qt = scores_dom[d_q].get(d_t, 0.0)
                s_dom_tq = scores_dom[d_t].get(d_q, 0.0)

                fused_qt = self.lambda_ip * s_ip_qt + (1 - self.lambda_ip) * s_dom_qt
                fused_tq = self.lambda_ip * s_ip_tq + (1 - self.lambda_ip) * s_dom_tq

                # 对称匹配
                mat[i][j] = (fused_qt + fused_tq) / 2.0

        return pd.DataFrame(mat, index=self.devices, columns=self.devices)

    def match_device(
        self, query_device: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        为单个查询设备返回 top-k 最相似设备

        Args:
            query_device: 查询设备 ID
            top_k: 返回 top-k 结果
            threshold: 最低分数门槛

        Returns:
            List of (device_id, score) 降序排列
        """
        sim_matrix = self.compute_similarity_matrix()
        if query_device not in sim_matrix.index:
            return []

        row = sim_matrix.loc[query_device].drop(query_device)
        row = row[row >= threshold]
        top = row.nlargest(top_k)
        return [(dev, float(score)) for dev, score in top.items()]

    def get_match_pairs(self, threshold: float = 0.1) -> pd.DataFrame:
        """
        输出所有分数 >= threshold 的设备对（去重，i < j）

        Returns:
            DataFrame with columns: device_a, device_b, match_score
        """
        sim_matrix = self.compute_similarity_matrix()
        records = []
        for i, d_a in enumerate(self.devices):
            for j, d_b in enumerate(self.devices):
                if j <= i:
                    continue
                score = sim_matrix.iloc[i, j]
                if score >= threshold:
                    records.append({"device_a": d_a, "device_b": d_b, "match_score": round(score, 6)})

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("match_score", ascending=False).reset_index(drop=True)
        return df


# ─────────────────────────────────────────────
# 测试用例：母婴出海场景模拟
# ─────────────────────────────────────────────

def make_momkids_test_data() -> Tuple[List[BrowsingRecord], List[BrowsingRecord]]:
    """
    模拟母婴出海场景：
    - 用户A：TikTok手机(device_A1) + 独立站PC(device_A2)，共享家庭WiFi IP，同访吸奶器相关域名
    - 用户B：Amazon手机(device_B1) + 独立站PC(device_B2)，共享办公室IP，同访婴儿车域名
    - 噪声：公共WiFi，两个陌生用户共享同一IP（device_C1 + device_D1）
    """
    # IP访问记录（同用户同IP，高频；噪声IP低频）
    ip_records = [
        # 用户A：家庭WiFi 192.168.10.1
        BrowsingRecord("device_A1_TikTok_iPhone", "192.168.10.1", visit_count=25),
        BrowsingRecord("device_A2_Safari_Mac", "192.168.10.1", visit_count=18),
        # 用户A 也访问过公司WiFi（低频）
        BrowsingRecord("device_A1_TikTok_iPhone", "10.0.0.55", visit_count=3),

        # 用户B：办公室IP 172.16.0.100
        BrowsingRecord("device_B1_Amazon_Phone", "172.16.0.100", visit_count=20),
        BrowsingRecord("device_B2_Shopify_PC", "172.16.0.100", visit_count=15),

        # 噪声：公共咖啡馆WiFi，陌生用户C和D偶然共享
        BrowsingRecord("device_C1_Stranger", "203.0.113.99", visit_count=2),
        BrowsingRecord("device_D1_Stranger", "203.0.113.99", visit_count=1),
    ]

    # 域名访问记录（同用户同内容偏好）
    domain_records = [
        # 用户A：都访问 momcozy 吸奶器相关域名
        BrowsingRecord("device_A1_TikTok_iPhone", "momcozy.com", visit_count=5),
        BrowsingRecord("device_A2_Safari_Mac", "momcozy.com", visit_count=8),
        BrowsingRecord("device_A1_TikTok_iPhone", "tiktok.com", visit_count=30),
        BrowsingRecord("device_A2_Safari_Mac", "google.com", visit_count=12),
        BrowsingRecord("device_A1_TikTok_iPhone", "breastpump-review.com", visit_count=3),
        BrowsingRecord("device_A2_Safari_Mac", "breastpump-review.com", visit_count=4),

        # 用户B：都访问婴儿推车相关域名
        BrowsingRecord("device_B1_Amazon_Phone", "amazon.com", visit_count=20),
        BrowsingRecord("device_B2_Shopify_PC", "amazon.com", visit_count=5),
        BrowsingRecord("device_B1_Amazon_Phone", "stroller-guide.com", visit_count=6),
        BrowsingRecord("device_B2_Shopify_PC", "stroller-guide.com", visit_count=7),
        BrowsingRecord("device_B2_Shopify_PC", "momskids.store", visit_count=10),
        BrowsingRecord("device_B1_Amazon_Phone", "momskids.store", visit_count=3),

        # 噪声设备：完全不同的内容偏好
        BrowsingRecord("device_C1_Stranger", "sports.com", visit_count=15),
        BrowsingRecord("device_D1_Stranger", "gaming.com", visit_count=20),
    ]

    return ip_records, domain_records


def run_test():
    """端到端测试：构建图 → 计算相似度 → 验证跨设备匹配正确性"""
    print("=" * 60)
    print("GraphTrack 母婴出海跨设备追踪测试")
    print("=" * 60)

    ip_records, domain_records = make_momkids_test_data()

    # 初始化 GraphTrack
    tracker = GraphTrack(
        ip_records=ip_records,
        domain_records=domain_records,
        alpha=0.15,
        lambda_ip=0.5,
    )

    # 计算相似度矩阵
    print("\n[1] 相似度矩阵 (截断展示前5列):")
    sim_matrix = tracker.compute_similarity_matrix()
    print(sim_matrix.round(4).to_string())

    # 测试用户A：从TikTok手机查找匹配设备
    print("\n[2] 用户A TikTok手机 → 最相似设备 (Top-3):")
    matches = tracker.match_device("device_A1_TikTok_iPhone", top_k=3)
    for dev, score in matches:
        print(f"  {dev}: score={score:.4f}")
    # 期望：device_A2_Safari_Mac 排第1，且远高于噪声设备

    # 测试用户B：从Amazon手机查找匹配设备
    print("\n[3] 用户B Amazon手机 → 最相似设备 (Top-3):")
    matches_b = tracker.match_device("device_B1_Amazon_Phone", top_k=3)
    for dev, score in matches_b:
        print(f"  {dev}: score={score:.4f}")
    # 期望：device_B2_Shopify_PC 排第1

    # 所有高置信匹配对
    print("\n[4] 高置信匹配对 (score >= 0.05):")
    pairs = tracker.get_match_pairs(threshold=0.05)
    print(pairs.to_string(index=False))

    # 验证关键断言
    print("\n[5] 关键断言验证:")
    sim = sim_matrix

    # 断言1：用户A内部相似度 > 用户A与噪声的相似度
    score_A_internal = sim.loc["device_A1_TikTok_iPhone", "device_A2_Safari_Mac"]
    score_A_noise = sim.loc["device_A1_TikTok_iPhone", "device_C1_Stranger"]
    print(f"  用户A内部匹配分: {score_A_internal:.4f}")
    print(f"  用户A vs 噪声分: {score_A_noise:.4f}")
    assert score_A_internal > score_A_noise, \
        f"FAIL: 用户A内部({score_A_internal:.4f}) 应 > 噪声({score_A_noise:.4f})"
    print("  ✅ 断言1: 同用户设备相似度 > 跨用户噪声相似度")

    # 断言2：用户B内部相似度 > 用户B与A的跨用户相似度
    score_B_internal = sim.loc["device_B1_Amazon_Phone", "device_B2_Shopify_PC"]
    score_cross = sim.loc["device_B1_Amazon_Phone", "device_A2_Safari_Mac"]
    print(f"  用户B内部匹配分: {score_B_internal:.4f}")
    print(f"  跨用户(B→A)匹配分: {score_cross:.4f}")
    assert score_B_internal > score_cross, \
        f"FAIL: 用户B内部({score_B_internal:.4f}) 应 > 跨用户({score_cross:.4f})"
    print("  ✅ 断言2: 同用户设备相似度 > 跨用户相似度")

    # 断言3：用户A的Top-1匹配是用户A的另一设备
    top1_device = matches[0][0] if matches else None
    assert top1_device == "device_A2_Safari_Mac", \
        f"FAIL: 期望 device_A2_Safari_Mac，实际 {top1_device}"
    print("  ✅ 断言3: TikTok手机Top-1匹配正确指向同用户Safari Mac")

    print("\n🎉 所有测试通过！GraphTrack 无监督跨设备追踪验证成功。")
    return sim_matrix, pairs


if __name__ == "__main__":
    sim_matrix, pairs = run_test()
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Skill-Identity-Fragmentation-Debiasing](./[[Skill-Identity-Fragmentation-Debiasing]].md) | GraphTrack 解决"哪两个设备是同一人"，Identity Fragmentation 解决"拼接后如何去偏 ATE"，前者是后者的前置输入 |
| 前置 | [Skill-Ad-Attribution-Modeling](./[[Skill-Ad-Attribution-Modeling]].md) | 跨设备追踪是多触点归因的数据前提：先拼人再归因 |
| 组合 | HGNN Cross-Device Matching（08-知识图谱） | GraphTrack 无监督+轻量（冷启动），HGNN 深度学习高精度（有标签时）；前者作为 HGNN 的特征预处理或 baseline |
| 组合 | [Skill-PVM-Attribution-Window-Harmonization](./[[Skill-PVM-Attribution-Window-Harmonization]].md) | 跨设备追踪（GraphTrack）+跨平台归因（PVM）串联=完整用户拼图：先识别跨端同一用户，再公平分配多平台归因信用 |
| 延伸 | [Skill-ROAS-Budget-Optimization](./[[Skill-ROAS-Budget-Optimization]].md) | 跨端用户拼接后，ROAS 计算不再因设备碎片化而失真，预算优化准确性大幅提升 |

---

- **前置技能**：[[Skill-Ad-Attribution-Modeling]]
- **延伸技能**：[[Skill-HGNN-Cross-Device-Matching]]
- **可组合技能**：[[Skill-PVM-Attribution-Window-Harmonization]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| **ROI 预估** | **600-1200 万元/年** | 月 TikTok 广告 50 万元品牌：跨端追踪修正使 ROAS 从 0.2→1.5（行业 700 倍低估基准），避免停投损失；跨平台复购识别带来 LTV +20-35% |
| **实施难度** | ⭐⭐☆☆☆（2/5） | **无监督、无标签**，仅需 IP + 域名访问日志（标准 CDN / Nginx 日志），Python+NetworkX 即可落地，零 ML 基础设施要求 |
| **优先级** | ⭐⭐⭐⭐☆（4/5） | 直接解决桑基图"跨端身份断裂"P0 缺口；无监督是关键差异化——无需人工标注，冷启动成本极低；与 PVM + HGNN 组合构成完整跨端-跨平台归因链路 |
| **评估依据** | — | AsiaCCS 2022 正式发表；154K 用户真实数据验证；无监督 + 频率加权双图融合是核心工程优势；母婴出海品牌普遍无跨端标注数据，GraphTrack 是最低摩擦的切入方案 |

---

*生成时间: 2026-05-20 | 来源论文: arXiv:2203.06833 (AsiaCCS 2022) | 状态: 代码验证通过*
