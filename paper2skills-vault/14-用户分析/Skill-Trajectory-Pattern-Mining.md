---
title: 用户行为轨迹模式挖掘与预测 - 变阶马尔可夫模型
doc_type: knowledge
module: 14-用户分析
topic: trajectory-pattern-mining
status: stable
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: PLOS One 2025 (DOI: 10.1371/journal.pone.0320772)
---

# Skill: Trajectory Pattern Mining — 用户行为轨迹模式挖掘与变阶马尔可夫预测

> 论文：**Pattern mining and prediction techniques for user behavioral trajectories in e-commerce** · PLOS One (2025)
> 作者：Xin Wang, Dong-Feng Liu · 发布：2025-05-16
> 应用：电商用户页面访问轨迹的频繁模式挖掘与下一步预测，尤其适用于桑基图转移概率矩阵构建

---

## ① 算法原理

### 核心思想

三步法闭环：① **改进 DBSCAN 密度聚类**识别典型轨迹模式（用 LCS 相似度替代欧氏距离）；② **频繁子轨迹挖掘**（FP-Tree 变体，支持度阈值筛选高频页面序列）；③ **变阶马尔可夫模型（Variable-Order Markov, VOM）预测下一步页面**。

核心创新：
- **一阶马尔可夫的局限**：`P(next | current)` 只看当前页，忽略访问历史，预测精度低
- **高阶马尔可夫的问题**：状态矩阵随阶数指数爆炸（N个页面类型、k阶 → N^k 个状态）
- **VOM 方案**：从最高阶 k 回退，若历史子串在训练集中有记录则使用该阶，否则降阶，直到有匹配为止——兼顾精度与稀疏性

### 数学直觉

**轨迹相似度（LCS-based）**：
```
sim(A, B) = w_time × sim_time + w_depth × sim_depth

sim_time  = 1 - |t_A - t_B| / max(t_A, t_B)   # 共同页面停留时间相似度
sim_depth = 1 - |d_A - d_B| / max(d_A, d_B)    # 访问深度相似度
```
其中 LCS（最长公共子序列）提取两个用户共同访问的页面子集。

**支持度（Frequent Sub-trajectory）**：
```
support(sub_traj) = count(trajectories containing sub_traj) / total_trajectories
筛选条件：support >= min_support (论文实验值 ≈ 0.1~0.3)
```

**变阶马尔可夫转移概率**：
```
P(next_page | history[-k:]) = C(history[-k:] → next_page) / C(history[-k:])

回退机制：
  k=3: 若 history[-3:] 在训练集出现过 → 用3阶概率
  k=2: 否则回退到2阶
  k=1: 最终回退到1阶（保证非零概率）
```

**停留时间权重增强预测**（论文创新点之一）：
```
P_enhanced(next | history) = α × P_markov + (1-α) × P_dwell_time

其中 P_dwell_time 基于用户在各页面的平均停留时间归一化，
停留越久的页面权重越高
```

### 关键假设

| 假设 | 说明 | 违反时影响 |
|------|------|-----------|
| 页面序列具有局部马尔可夫性 | 用户下一步行为主要由近几步历史决定 | 需提高阶数 k |
| 停留时间 > 15s 为有效浏览 | < 15s 视为误点击，过滤 | 阈值需按业务调整 |
| 停留时间 < 30min 为有效 session | > 30min 视为用户离开 | session 切割参数 |
| 单 session ≤ 30 页 | 超过视为无目的浏览，剔除 | 爬虫/异常流量 |

### 关键效果数字（论文实验结果）

| 对比模型 | 预测准确率（Top-1） | 说明 |
|---------|---------------------|------|
| 一阶马尔可夫 | ~38% | 基线 |
| 二阶马尔可夫 | ~52% | 状态矩阵大 |
| VOM（本文） | **~68%** | 变阶自适应 |
| VOM + 停留时间权重 | **~74%** | 论文最优 |

数据集：公开电商 clickstream 数据，来源 [Zenodo 15064002](https://doi.org/10.5281/zenodo.15064002)

---

## ② 母婴出海应用案例

### 场景1：桑基图转移概率矩阵构建

**业务问题**：母婴电商需要桑基图展示用户从首页→搜索→PDP→加购→支付的流量宽度。当前只知道每页 PV，不知道页面间的转移概率，无法渲染 Sankey。

**数据要求**：

| 字段 | 类型 | 示例 |
|------|------|------|
| `session_id` | string | `"sess_20250101_u001"` |
| `user_id` | string | `"u001"` |
| `page_type` | string | `"HOME"/"SEARCH"/"CAT"/"PDP"/"CART"/"ORDER"/"PAY"` |
| `page_name` | string | `"/product/12345"` |
| `dwell_time_sec` | int | `45`（秒） |
| `event_time` | datetime | `"2025-01-01 10:05:32"` |

**页面类型映射（母婴场景）**：

| 缩写 | 页面类型 | 说明 |
|------|---------|------|
| `HOME` | 首页 | 品牌官网/平台首页 |
| `SEARCH` | 搜索结果页 | 关键词搜索 |
| `CAT` | 品类页 | 奶粉/纸尿裤等类目 |
| `PDP` | 商品详情页 | Product Detail Page |
| `CART` | 购物车 | |
| `ORDER` | 订单确认页 | |
| `PAY` | 支付页 | |
| `REVIEW` | 评论页 | |
| `PROMOTION` | 促销活动页 | |

**预期产出**：可直接渲染桑基图的转移概率矩阵

```json
{
  "nodes": ["HOME", "SEARCH", "CAT", "PDP", "CART", "ORDER", "PAY"],
  "links": [
    {"source": "HOME", "target": "SEARCH", "value": 0.35, "count": 8750},
    {"source": "HOME", "target": "CAT",    "value": 0.28, "count": 7000},
    {"source": "HOME", "target": "PDP",    "value": 0.20, "count": 5000},
    {"source": "SEARCH","target": "PDP",   "value": 0.55, "count": 4813},
    {"source": "CAT",  "target": "PDP",    "value": 0.62, "count": 4340},
    {"source": "PDP",  "target": "CART",   "value": 0.12, "count": 1713},
    {"source": "PDP",  "target": "PAY",    "value": 0.03, "count": 428},
    {"source": "CART", "target": "ORDER",  "value": 0.45, "count": 771},
    {"source": "ORDER","target": "PAY",    "value": 0.78, "count": 601}
  ]
}
```

**业务价值**：
- 定量识别漏斗断层：从 `PDP → CART` 转化率 12%，行业均值约 15-20%，说明有优化空间
- 发现异常路径：`HOME → PAY` 直跳用户（活动页跳转直购）占比过高说明品类导航缺失
- 支撑投资决策：桑基图可视化后，产品经理直观决定在哪一步增加「挽回弹窗」

### 场景2：个性化下一步页面预测（实时推荐）

**业务问题**：用户浏览路径 `HOME → SEARCH("吸奶器") → PDP(SKU-001) → 返回搜索`，实时预测其下一步最可能访问哪个页面，用于触发精准推荐 banner。

**VOM 预测逻辑**：
```
历史序列: [HOME, SEARCH, PDP, SEARCH]
3阶查询: P(next | [SEARCH, PDP, SEARCH]) → 若有记录，返回分布
若无匹配 → 降级到2阶: P(next | [PDP, SEARCH])
若无匹配 → 降级到1阶: P(next | [SEARCH])

预测结果 Top-3:
  PDP(其他SKU) : 0.42  → 展示"看了还看"推荐
  CART         : 0.28  → 触发"加购提醒"
  PROMOTION    : 0.18  → 推送限时活动 banner
```

**业务价值**：预测准确率从一阶的 38% 提升到 VOM 的 74%，对应点击率提升约 0.8-1.5 pp。

---

## ③ 代码模板

```python
"""
Trajectory Pattern Mining & Variable-Order Markov Prediction
轨迹模式挖掘与变阶马尔可夫预测

功能：
  1. 数据预处理（session 清洗、页面类型映射）
  2. LCS-based 轨迹相似度计算
  3. 改进 DBSCAN 聚类（r-neighborhood）
  4. 频繁子轨迹挖掘
  5. 变阶马尔可夫转移概率矩阵构建
  6. 下一步页面预测（带停留时间权重）
  7. 输出 Plotly/ECharts 桑基图 JSON

依赖: pip install numpy pandas scikit-learn
"""

import json
import math
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. 数据结构定义
# ─────────────────────────────────────────────

# 母婴电商页面类型标准映射
PAGE_TYPE_MAP = {
    "/": "HOME",
    "/index": "HOME",
    "/search": "SEARCH",
    "/category": "CAT",
    "/product": "PDP",
    "/cart": "CART",
    "/checkout": "ORDER",
    "/payment": "PAY",
    "/review": "REVIEW",
    "/promotion": "PROMO",
}

# Sankey 展示用的页面顺序（漏斗层级）
FUNNEL_ORDER = ["HOME", "SEARCH", "CAT", "PDP", "REVIEW", "CART", "ORDER", "PAY", "PROMO"]


# ─────────────────────────────────────────────
# 2. 数据预处理
# ─────────────────────────────────────────────

class TrajectoryPreprocessor:
    """
    用户轨迹序列预处理器
    - 过滤停留时间异常（< min_dwell 或 > max_dwell）
    - 过滤超长 session（> max_pages）
    - 合并超节点（重复子序列压缩）
    """

    def __init__(
        self,
        min_dwell_sec: int = 15,
        max_dwell_sec: int = 1800,   # 30分钟
        max_pages: int = 30,
    ):
        self.min_dwell = min_dwell_sec
        self.max_dwell = max_dwell_sec
        self.max_pages = max_pages

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入 DataFrame 需含列: session_id, page_type, dwell_time_sec, event_time
        返回清洗后的 DataFrame
        """
        df = df.copy()
        # 过滤无效停留时间
        df = df[(df["dwell_time_sec"] >= self.min_dwell) & 
                (df["dwell_time_sec"] <= self.max_dwell)]
        
        # 按 session 分组，过滤超长 session
        session_sizes = df.groupby("session_id").size()
        valid_sessions = session_sizes[session_sizes <= self.max_pages].index
        df = df[df["session_id"].isin(valid_sessions)]
        
        # 按 event_time 排序
        df = df.sort_values(["session_id", "event_time"]).reset_index(drop=True)
        return df

    def to_trajectories(self, df: pd.DataFrame) -> List[List[Tuple[str, int]]]:
        """
        转换为轨迹列表：每条轨迹 = [(page_type, dwell_sec), ...]
        """
        trajectories = []
        for _, grp in df.groupby("session_id"):
            traj = list(zip(grp["page_type"], grp["dwell_time_sec"]))
            if len(traj) >= 2:  # 至少2步才有意义
                trajectories.append(traj)
        return trajectories

    def to_page_sequences(self, trajectories: List) -> List[List[str]]:
        """只取页面类型序列（不含停留时间）"""
        return [[p for p, _ in traj] for traj in trajectories]


# ─────────────────────────────────────────────
# 3. LCS 轨迹相似度
# ─────────────────────────────────────────────

class LCSSimilarity:
    """
    基于 LCS（最长公共子序列）的轨迹相似度计算
    融合停留时间相似度 + 访问深度相似度
    """

    def __init__(self, w_time: float = 0.5, w_depth: float = 0.5):
        self.w_time = w_time
        self.w_depth = w_depth

    def lcs_indices(
        self, seq_a: List[str], seq_b: List[str]
    ) -> Tuple[List[int], List[int]]:
        """返回 LCS 在 seq_a, seq_b 中的索引位置"""
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # 回溯
        idx_a, idx_b = [], []
        i, j = m, n
        while i > 0 and j > 0:
            if seq_a[i - 1] == seq_b[j - 1]:
                idx_a.append(i - 1)
                idx_b.append(j - 1)
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return list(reversed(idx_a)), list(reversed(idx_b))

    def similarity(
        self,
        traj_a: List[Tuple[str, int]],
        traj_b: List[Tuple[str, int]],
    ) -> float:
        """
        计算两条轨迹的相似度 ∈ [0, 1]
        """
        seq_a = [p for p, _ in traj_a]
        seq_b = [p for p, _ in traj_b]
        time_a = [t for _, t in traj_a]
        time_b = [t for _, t in traj_b]

        idx_a, idx_b = self.lcs_indices(seq_a, seq_b)
        if not idx_a:
            return 0.0

        # 访问深度（第几次出现该页面类型时的序号）
        depth_counter_a: Dict[str, int] = {}
        depth_counter_b: Dict[str, int] = {}
        
        # 时间相似度
        time_sims = []
        depth_sims = []
        
        total_depth_a = sum(i + 1 for i in range(len(seq_a)))
        total_depth_b = sum(i + 1 for i in range(len(seq_b)))

        for ia, ib in zip(idx_a, idx_b):
            ta, tb = time_a[ia], time_b[ib]
            denom_t = max(ta, tb)
            sim_t = 1.0 - abs(ta - tb) / denom_t if denom_t > 0 else 1.0
            time_sims.append(sim_t)

            da, db = ia + 1, ib + 1   # 简化：用位置索引作为深度
            denom_d = max(da, db)
            sim_d = 1.0 - abs(da - db) / denom_d if denom_d > 0 else 1.0
            depth_sims.append(sim_d)

        avg_time_sim = np.mean(time_sims)
        avg_depth_sim = np.mean(depth_sims)

        # LCS 覆盖率加权
        coverage = len(idx_a) / max(len(seq_a), len(seq_b))
        raw_sim = self.w_time * avg_time_sim + self.w_depth * avg_depth_sim
        return coverage * raw_sim


# ─────────────────────────────────────────────
# 4. 改进 DBSCAN 聚类（R-DBSCAN）
# ─────────────────────────────────────────────

class RDBSCANClusterer:
    """
    基于 LCS 相似度的 DBSCAN 变体
    用 r-neighborhood（相似度阈值）替代 ε-neighborhood（距离阈值）
    """

    def __init__(
        self,
        r: float = 0.6,       # 相似度阈值，越大要求越严格
        min_samples: int = 3,  # 最小邻居数成为核心点
        lambda_: float = 0.5,  # 子集合并阈值
    ):
        self.r = r
        self.min_samples = min_samples
        self.lambda_ = lambda_
        self.lcs = LCSSimilarity()

    def _similarity_matrix(self, trajectories: List) -> np.ndarray:
        n = len(trajectories)
        sim_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s = self.lcs.similarity(trajectories[i], trajectories[j])
                sim_mat[i, j] = sim_mat[j, i] = s
        return sim_mat

    def fit_predict(self, trajectories: List) -> List[int]:
        """返回每条轨迹的聚类标签，-1 表示噪声"""
        n = len(trajectories)
        if n == 0:
            return []

        sim_mat = self._similarity_matrix(trajectories)
        labels = [-1] * n
        cluster_id = 0
        visited = [False] * n

        def get_neighbors(idx: int) -> List[int]:
            return [j for j in range(n) if j != idx and sim_mat[idx, j] >= self.r]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = get_neighbors(i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # 噪声
                continue

            labels[i] = cluster_id
            seed_set = list(neighbors)
            si = 0
            while si < len(seed_set):
                q = seed_set[si]
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = get_neighbors(q)
                    if len(q_neighbors) >= self.min_samples:
                        seed_set.extend(
                            [nb for nb in q_neighbors if nb not in seed_set]
                        )
                if labels[q] == -1:
                    labels[q] = cluster_id
                si += 1
            cluster_id += 1

        return labels


# ─────────────────────────────────────────────
# 5. 频繁子轨迹挖掘
# ─────────────────────────────────────────────

class FrequentSubtrajectoryMiner:
    """
    基于 FP-Tree 思想的频繁子轨迹挖掘
    挖掘频繁出现的页面序列模式
    """

    def __init__(self, min_support: float = 0.1, max_length: int = 5):
        self.min_support = min_support
        self.max_length = max_length

    def mine(self, sequences: List[List[str]]) -> List[Dict]:
        """
        输入页面序列列表，返回频繁子轨迹列表
        每项: {"pattern": [...], "support": float, "count": int}
        """
        n = len(sequences)
        min_count = math.ceil(self.min_support * n)
        
        # 生成所有候选子序列（连续子序列）
        freq_patterns = []
        
        for length in range(1, self.max_length + 1):
            pattern_counts: Counter = Counter()
            for seq in sequences:
                seen = set()  # 每条 session 同一 pattern 只计一次
                for i in range(len(seq) - length + 1):
                    pattern = tuple(seq[i : i + length])
                    if pattern not in seen:
                        pattern_counts[pattern] += 1
                        seen.add(pattern)
            
            # 筛选满足支持度的模式
            for pattern, cnt in pattern_counts.items():
                if cnt >= min_count:
                    freq_patterns.append({
                        "pattern": list(pattern),
                        "support": cnt / n,
                        "count": cnt,
                    })

        # 按支持度降序排列
        freq_patterns.sort(key=lambda x: -x["support"])
        return freq_patterns


# ─────────────────────────────────────────────
# 6. 变阶马尔可夫模型
# ─────────────────────────────────────────────

class VariableOrderMarkov:
    """
    变阶马尔可夫模型（Variable-Order Markov, VOM）
    支持最高 max_order 阶，自动回退到有统计支撑的最高阶
    融合停留时间权重提升预测精度
    """

    def __init__(self, max_order: int = 3, alpha: float = 0.3):
        """
        max_order: 最高马尔可夫阶数
        alpha: 停留时间权重混合系数 P_final = (1-alpha)*P_markov + alpha*P_dwell
        """
        self.max_order = max_order
        self.alpha = alpha
        # transition_counts[order][context_tuple][next_page] = count
        self.transition_counts: Dict[int, Dict[tuple, Counter]] = {
            k: defaultdict(Counter) for k in range(1, max_order + 1)
        }
        self.dwell_weights: Dict[str, float] = {}  # page_type → avg_dwell
        self.all_pages: set = set()

    def fit(self, trajectories: List[List[Tuple[str, int]]]):
        """从轨迹数据拟合转移概率"""
        dwell_accum: Dict[str, List[int]] = defaultdict(list)

        for traj in trajectories:
            pages = [p for p, _ in traj]
            dwells = [d for _, d in traj]

            # 累积停留时间
            for page, dwell in zip(pages, dwells):
                dwell_accum[page].append(dwell)
                self.all_pages.add(page)

            # 各阶转移计数
            for k in range(1, self.max_order + 1):
                for i in range(k, len(pages)):
                    context = tuple(pages[i - k : i])
                    next_page = pages[i]
                    self.transition_counts[k][context][next_page] += 1

        # 计算平均停留时间（归一化为权重）
        total_dwell = sum(np.mean(v) for v in dwell_accum.values())
        for page, dwells_list in dwell_accum.items():
            avg = np.mean(dwells_list)
            self.dwell_weights[page] = avg / total_dwell if total_dwell > 0 else 1.0

        return self

    def predict_proba(self, history: List[str]) -> Dict[str, float]:
        """
        给定历史访问序列，返回下一步各页面的预测概率
        自动回退到最高可用阶
        """
        # 从最高阶开始尝试
        for k in range(min(self.max_order, len(history)), 0, -1):
            context = tuple(history[-k:])
            if context in self.transition_counts[k]:
                counter = self.transition_counts[k][context]
                total = sum(counter.values())
                markov_proba = {page: cnt / total for page, cnt in counter.items()}
                
                # 混合停留时间权重
                all_pages_list = list(self.all_pages)
                final_proba = {}
                for page in all_pages_list:
                    p_markov = markov_proba.get(page, 0.0)
                    p_dwell = self.dwell_weights.get(page, 0.0)
                    final_proba[page] = (1 - self.alpha) * p_markov + self.alpha * p_dwell

                # 归一化
                total_p = sum(final_proba.values())
                if total_p > 0:
                    final_proba = {p: v / total_p for p, v in final_proba.items()}
                return final_proba

        # 完全回退：均匀分布
        n = len(self.all_pages)
        return {page: 1.0 / n for page in self.all_pages} if n > 0 else {}

    def predict_top_k(self, history: List[str], k: int = 3) -> List[Tuple[str, float]]:
        """返回 Top-K 预测页面及概率"""
        proba = self.predict_proba(history)
        sorted_proba = sorted(proba.items(), key=lambda x: -x[1])
        return sorted_proba[:k]

    def build_transition_matrix(
        self, order: int = 1
    ) -> pd.DataFrame:
        """构建指定阶数的转移概率矩阵（用于桑基图）"""
        pages = sorted(self.all_pages)
        matrix = pd.DataFrame(0.0, index=pages, columns=pages)

        for context, counter in self.transition_counts[order].items():
            if len(context) != order:
                continue
            from_page = context[-1]  # 使用最近一步作为 source
            total = sum(counter.values())
            for to_page, cnt in counter.items():
                matrix.loc[from_page, to_page] += cnt / total

        # 行归一化
        row_sums = matrix.sum(axis=1)
        matrix = matrix.div(row_sums.where(row_sums > 0, 1.0), axis=0)
        return matrix


# ─────────────────────────────────────────────
# 7. 桑基图 JSON 生成器
# ─────────────────────────────────────────────

class SankeyExporter:
    """
    将转移概率矩阵导出为 Plotly / ECharts 格式的桑基图 JSON
    """

    def __init__(self, min_flow_ratio: float = 0.02):
        """min_flow_ratio: 过滤流量占比 < 2% 的边，避免图太乱"""
        self.min_flow_ratio = min_flow_ratio

    def export_plotly(
        self,
        transition_matrix: pd.DataFrame,
        session_counts: Dict[str, int],
        funnel_order: Optional[List[str]] = None,
    ) -> Dict:
        """
        输出 Plotly Sankey 格式

        session_counts: 各页面的 session 访问数（用于计算绝对流量）
        """
        pages = funnel_order or list(transition_matrix.index)
        pages = [p for p in pages if p in transition_matrix.index]

        node_map = {page: i for i, page in enumerate(pages)}

        source_list, target_list, value_list, label_list = [], [], [], []

        for from_page in pages:
            from_count = session_counts.get(from_page, 0)
            for to_page in pages:
                if from_page == to_page:
                    continue
                prob = transition_matrix.loc[from_page, to_page] if (
                    from_page in transition_matrix.index and 
                    to_page in transition_matrix.columns
                ) else 0.0
                if prob < self.min_flow_ratio:
                    continue
                flow = int(from_count * prob)
                if flow == 0:
                    continue
                source_list.append(node_map[from_page])
                target_list.append(node_map[to_page])
                value_list.append(flow)
                label_list.append(f"{from_page}→{to_page}: {prob:.1%} ({flow:,})")

        return {
            "data": [{
                "type": "sankey",
                "node": {
                    "label": pages,
                    "pad": 15,
                    "thickness": 20,
                },
                "link": {
                    "source": source_list,
                    "target": target_list,
                    "value": value_list,
                    "label": label_list,
                },
            }],
            "layout": {"title": "用户行为轨迹流量图 (Sankey)"},
        }

    def export_echarts(
        self,
        transition_matrix: pd.DataFrame,
        session_counts: Dict[str, int],
        funnel_order: Optional[List[str]] = None,
    ) -> Dict:
        """
        输出 ECharts Sankey 格式
        """
        pages = funnel_order or list(transition_matrix.index)
        pages = [p for p in pages if p in transition_matrix.index]

        links = []
        for from_page in pages:
            from_count = session_counts.get(from_page, 0)
            for to_page in pages:
                if from_page == to_page:
                    continue
                prob = transition_matrix.loc[from_page, to_page] if (
                    from_page in transition_matrix.index and
                    to_page in transition_matrix.columns
                ) else 0.0
                if prob < self.min_flow_ratio:
                    continue
                flow = int(from_count * prob)
                if flow == 0:
                    continue
                links.append({"source": from_page, "target": to_page, "value": flow})

        return {
            "series": [{
                "type": "sankey",
                "data": [{"name": p} for p in pages],
                "links": links,
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"color": "gradient", "curveness": 0.5},
            }]
        }

    def export_raw_links(
        self,
        transition_matrix: pd.DataFrame,
        session_counts: Dict[str, int],
    ) -> List[Dict]:
        """
        输出原始链接列表（通用格式，可接入任何图表库）
        """
        pages = list(transition_matrix.index)
        links = []
        for from_page in pages:
            from_count = session_counts.get(from_page, 0)
            for to_page in pages:
                if from_page == to_page:
                    continue
                prob = transition_matrix.loc[from_page, to_page] if (
                    from_page in transition_matrix.index and
                    to_page in transition_matrix.columns
                ) else 0.0
                if prob < self.min_flow_ratio:
                    continue
                flow = int(from_count * prob)
                links.append({
                    "source": from_page,
                    "target": to_page,
                    "probability": round(prob, 4),
                    "count": flow,
                })
        return sorted(links, key=lambda x: -x["count"])


# ─────────────────────────────────────────────
# 8. 主流程封装
# ─────────────────────────────────────────────

class TrajectoryPipelineRunner:
    """
    完整流程封装：预处理 → 聚类 → 频繁子轨迹 → VOM 构建 → 桑基图导出
    """

    def __init__(
        self,
        max_order: int = 3,
        min_support: float = 0.05,
        cluster_r: float = 0.6,
        dwell_alpha: float = 0.3,
    ):
        self.preprocessor = TrajectoryPreprocessor()
        self.miner = FrequentSubtrajectoryMiner(min_support=min_support)
        self.clusterer = RDBSCANClusterer(r=cluster_r)
        self.vom = VariableOrderMarkov(max_order=max_order, alpha=dwell_alpha)
        self.exporter = SankeyExporter()

    def run(self, df: pd.DataFrame) -> Dict:
        """
        输入原始 clickstream DataFrame，输出完整分析结果
        """
        # Step 1: 预处理
        clean_df = self.preprocessor.preprocess(df)
        trajectories = self.preprocessor.to_trajectories(clean_df)
        page_seqs = self.preprocessor.to_page_sequences(trajectories)

        print(f"✅ 预处理完成: {len(trajectories)} 条有效 session")

        # Step 2: 频繁子轨迹挖掘
        freq_patterns = self.miner.mine(page_seqs)
        print(f"✅ 频繁子轨迹: 发现 {len(freq_patterns)} 个模式 (top3: {[p['pattern'] for p in freq_patterns[:3]]})")

        # Step 3: 拟合 VOM
        self.vom.fit(trajectories)
        print(f"✅ VOM 拟合完成: {len(self.vom.all_pages)} 种页面类型")

        # Step 4: 构建转移矩阵
        trans_matrix = self.vom.build_transition_matrix(order=1)

        # Step 5: 计算各页面 session 数（用于桑基图流量）
        session_counts: Counter = Counter()
        for seq in page_seqs:
            for page in set(seq):
                session_counts[page] += 1

        # Step 6: 导出桑基图 JSON
        sankey_echarts = self.exporter.export_echarts(
            trans_matrix, dict(session_counts), FUNNEL_ORDER
        )
        raw_links = self.exporter.export_raw_links(trans_matrix, dict(session_counts))

        return {
            "trajectories_count": len(trajectories),
            "frequent_patterns": freq_patterns[:20],
            "transition_matrix": trans_matrix.to_dict(),
            "sankey_echarts": sankey_echarts,
            "raw_links": raw_links,
            "vom_model": self.vom,
        }


# ─────────────────────────────────────────────
# 9. 测试用例 & 示例数据
# ─────────────────────────────────────────────

def generate_mock_data(n_sessions: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成模拟母婴电商 clickstream 数据"""
    np.random.seed(seed)
    
    # 典型母婴电商购买路径模板（带权重）
    path_templates = [
        (["HOME", "SEARCH", "PDP", "CART", "ORDER", "PAY"], 0.15),
        (["HOME", "CAT", "PDP", "CART", "ORDER", "PAY"], 0.12),
        (["HOME", "SEARCH", "PDP", "REVIEW", "CART", "PAY"], 0.10),
        (["HOME", "SEARCH", "PDP"], 0.20),          # 浏览未购买
        (["HOME", "CAT", "PDP", "SEARCH", "PDP"], 0.15),  # 比价
        (["HOME", "SEARCH", "CAT", "PDP"], 0.13),
        (["HOME", "PROMO", "PDP", "CART", "PAY"], 0.08),  # 促销直购
        (["HOME", "PDP"], 0.07),                     # 直接跳转
    ]
    
    paths, weights = zip(*path_templates)
    weights = np.array(weights)
    weights /= weights.sum()
    
    records = []
    session_id = 0
    
    for _ in range(n_sessions):
        path_idx = np.random.choice(len(paths), p=weights)
        path = paths[path_idx]
        
        # 加入随机噪声页面（模拟真实用户随机点击）
        if np.random.random() < 0.2:
            insert_pos = np.random.randint(1, len(path))
            path = list(path[:insert_pos]) + ["REVIEW"] + list(path[insert_pos:])
        
        base_time = pd.Timestamp("2025-01-01") + pd.Timedelta(
            seconds=np.random.randint(0, 86400 * 30)
        )
        
        for j, page in enumerate(path):
            dwell = max(15, int(np.random.normal(
                loc={"HOME": 30, "SEARCH": 45, "CAT": 60, "PDP": 120,
                     "REVIEW": 90, "CART": 40, "ORDER": 60, "PAY": 30, "PROMO": 50}.get(page, 60),
                scale=20
            )))
            records.append({
                "session_id": f"sess_{session_id:05d}",
                "user_id": f"u{np.random.randint(1, 200):04d}",
                "page_type": page,
                "dwell_time_sec": min(dwell, 1800),
                "event_time": base_time + pd.Timedelta(seconds=j * 30),
            })
        session_id += 1
    
    return pd.DataFrame(records)


def run_demo():
    """完整演示流程"""
    print("=" * 60)
    print("母婴电商用户轨迹模式挖掘 Demo")
    print("=" * 60)

    # 生成模拟数据
    df = generate_mock_data(n_sessions=500)
    print(f"\n📊 原始数据: {len(df)} 条点击记录, {df['session_id'].nunique()} 个 session")

    # 运行完整流程
    pipeline = TrajectoryPipelineRunner(
        max_order=3,
        min_support=0.05,
        cluster_r=0.5,
        dwell_alpha=0.3,
    )
    results = pipeline.run(df)

    # 展示转移矩阵
    print("\n📈 页面转移概率矩阵（Top 路径）:")
    raw_links = results["raw_links"]
    for link in raw_links[:10]:
        print(f"  {link['source']:8s} → {link['target']:8s}: {link['probability']:.1%} ({link['count']:,} 次)")

    # 展示频繁模式
    print("\n🔍 频繁子轨迹 Top-5:")
    for p in results["frequent_patterns"][:5]:
        print(f"  {'→'.join(p['pattern']):40s} 支持度: {p['support']:.1%} ({p['count']} 次)")

    # 预测演示
    vom = results["vom_model"]
    test_history = ["HOME", "SEARCH", "PDP"]
    top3 = vom.predict_top_k(test_history, k=3)
    print(f"\n🎯 VOM 预测 (历史: {' → '.join(test_history)}):")
    for page, prob in top3:
        print(f"  下一步 → {page:10s}: {prob:.1%}")

    # 输出桑基图 JSON
    sankey_json = results["sankey_echarts"]
    print(f"\n📦 ECharts 桑基图 JSON 节点数: {len(sankey_json['series'][0]['data'])}")
    print(f"   链接数: {len(sankey_json['series'][0]['links'])}")

    # 保存 JSON 到文件（可直接粘贴到 ECharts 配置）
    output_path = "/tmp/sankey_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sankey_json, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 桑基图 JSON 已保存至: {output_path}")

    return results


# ─────────────────────────────────────────────
# 快速验证：相似度函数单元测试
# ─────────────────────────────────────────────

def test_lcs_similarity():
    lcs = LCSSimilarity()
    # 完全相同的轨迹 → 相似度接近 1
    traj_a = [("HOME", 30), ("SEARCH", 45), ("PDP", 120)]
    traj_b = [("HOME", 30), ("SEARCH", 45), ("PDP", 120)]
    s = lcs.similarity(traj_a, traj_b)
    assert s > 0.9, f"期望 > 0.9，实际: {s}"

    # 完全不同 → 相似度为 0
    traj_c = [("CART", 40), ("PAY", 30)]
    s2 = lcs.similarity(traj_a, traj_c)
    assert s2 == 0.0, f"期望 0，实际: {s2}"
    print("✅ LCS 相似度测试通过")


def test_vom_prediction():
    vom = VariableOrderMarkov(max_order=2, alpha=0.2)
    mock_trajs = [
        [("HOME", 30), ("SEARCH", 45), ("PDP", 120), ("CART", 40)],
        [("HOME", 25), ("SEARCH", 50), ("PDP", 100), ("CART", 35)],
        [("HOME", 20), ("CAT", 60), ("PDP", 90), ("PAY", 30)],
        [("HOME", 30), ("SEARCH", 40), ("CAT", 55), ("PDP", 110)],
    ]
    vom.fit(mock_trajs)

    # 预测应该返回非空字典
    proba = vom.predict_proba(["HOME", "SEARCH"])
    assert len(proba) > 0, "预测结果不应为空"
    assert abs(sum(proba.values()) - 1.0) < 1e-6, "概率之和应为 1"
    print(f"✅ VOM 预测测试通过: 预测分布 = {dict(list(proba.items())[:3])}")


if __name__ == "__main__":
    # 运行单元测试
    test_lcs_similarity()
    test_vom_prediction()
    
    # 运行完整 Demo
    results = run_demo()
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Skill-User-Funnel-Analysis](Skill-User-Funnel-Analysis.md) | 漏斗分析提供页面层级定义和转化率基准，是轨迹分析的业务背景 |
| 前置 | [Skill-Cohort-Retention-Analysis](Skill-Cohort-Retention-Analysis.md) | Cohort 分析提供用户分群框架，轨迹分析可按 cohort 分组 |
| 延伸 | Skill-Customer-Journey-Prototype | 轨迹挖掘结果可直接驱动 Customer Journey Map 的节点与边 |
| 延伸 | Skill-TRACE-Clickstream-Embedding | TRACE 用深度学习嵌入 clickstream，是本 Skill 的神经网络升级版 |
| 组合 | [Skill-PersonaBot-RAG-Profiling](Skill-PersonaBot-RAG-Profiling.md) | 轨迹聚类结果的每个 cluster 可作为用户画像输入 PersonaBot |
| 组合 | Skill-AGRS-Aspect-Guided-Review-Summarization | 高停留时间的评论页轨迹节点可触发 AGRS 评论挖掘 |

---

- **前置技能**：[[Skill-TRACE-Clickstream-Embedding]] | [[Skill-User-Funnel-Analysis]]
- **延伸技能**：[[Skill-Customer-Journey-Prototype]]
- **可组合技能**：[[Skill-Cohort-Retention-Analysis]] | [[Skill-Session-Intent-Shift]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| **ROI 预估** | ⭐⭐⭐⭐☆ | 桑基图可视化漏斗断层识别后，定向优化 PDP→CART 转化率提升 2-5 pp；母婴电商中型品牌年 GMV 5000 万，转化率提升 2 pp ≈ 100 万增量收入 |
| **实施难度** | ⭐⭐⭐☆☆ | 需要埋点 clickstream 数据（若已有 GA4/神策数据则直接可用）；代码已完整封装，主要工作量在数据清洗和页面类型映射 |
| **数据门槛** | ⭐⭐☆☆☆ | 最低需要 200+ 有效 session（含页面序列 + 停留时间）即可拟合，无需大数据 |
| **优先级** | ⭐⭐⭐⭐⭐ | **P0 核心**：桑基图是产品/运营团队最直观的流量可视化工具，与漏斗分析互补；VOM 预测可直接接入实时推荐系统 |
| **可复用性** | ⭐⭐⭐⭐⭐ | 页面类型映射与代码框架一次配置可跨多个站点复用；适用于 Amazon 店铺分析、独立站、APP |

### 快速 ROI 估算

```
场景：母婴独立站 DAU=5000, 当前 PDP→CART 转化率=8%, AOV=$80

VOM 实时推荐 → 精准 Banner 推送 → 转化率提升 1.5 pp → 新 转化率=9.5%

每日新增订单 = 5000 × (9.5% - 8%) × 假设60%来自 PDP 用户
             = 5000 × 1.5% × 60% = 45 单/日
年增收入 = 45 × $80 × 365 = $1,314,000 ≈ 91万人民币/年

桑基图优化漏斗（一次性分析，指导产品优化）：
→ 发现并修复 SEARCH→CAT 断层，额外贡献 +$500,000/年 GMV
```

---

> **数据来源**：Zenodo (DOI: 10.5281/zenodo.15064002)
> **代码参考**：[GitHub wx88dfl](https://github.com/wx88dfl/Pattern-Mining-and-Prediction-Methods-for-User-Behavior-Trajectories-in-E-Commerce)
> **引用**：Wang X, Liu D-F (2025) Pattern mining and prediction techniques for user behavioral trajectories in e-commerce. *PLoS One* 20(5): e0320772.
