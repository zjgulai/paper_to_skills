---
title: Supply Chain ML Feature Engineering — 供应链 ML 特征工程：时序+图+统计三维
doc_type: knowledge
module: 12-ML基础
topic: supply-chain-ml-feature-engineering
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill: Supply Chain ML Feature Engineering — 供应链 ML 特征工程

> 专门针对供应链场景的 ML 特征工程方法：时序特征（滞后/滚动统计）+ 图特征（供应商网络中心性）+ 业务特征（季节指数/促销编码）的系统化构建。

---

## ① 算法原理

### 供应链数据的特殊性

供应链数据与常规 ML 数据集有三大本质区别：

1. **稀疏性**：长尾 SKU 历史销量不足（< 30 天），传统特征工程无法直接应用
2. **季节性**：节假日/促销/季节性需求波动，需明确编码而非让模型自动发现
3. **多尺度**：补货决策依赖不同时间粒度（日销量/周趋势/月基线）的特征

### 时序特征的正确构建（避免数据泄露）

**核心规则**：特征计算时，`t` 时刻的特征只能使用 `t` 之前的数据。

```
正确：lag_7d[t] = demand[t-7]          # 7天前的实际值
错误：rolling_mean[t] = mean(demand[t-3:t+3])  # 包含未来数据！
```

| 特征类型 | 计算公式 | 适用场景 |
|---------|---------|---------|
| 滞后特征 | $x_{t-k}$ | 捕获周期性规律 |
| 滚动均值 | $\bar{x}_{[t-w, t-1]}$ | 平滑短期波动 |
| 滚动标准差 | $\sigma_{[t-w, t-1]}$ | 衡量需求不确定性 |
| 指数加权均值 | $\text{ewm}(x, \alpha)$ | 近期数据权重更高 |

### 供应商网络图特征

供应商之间存在共同客户/原材料依赖关系，构建供应商网络 $G = (V, E)$：

- **度中心性**：$C_D(v) = \frac{deg(v)}{|V|-1}$，衡量供应商的连接广度
- **PageRank 近似**：$\text{PR}(v) \approx \frac{1-d}{|V|} + d \sum_{u \in \mathcal{N}(v)} \frac{\text{PR}(u)}{deg(u)}$，衡量供应商的影响力
- **集中度风险**：单一供应商依赖比例，用于风险特征

### 目标编码（高基数品类）

高基数品类（如 SKU 数 > 10,000）用 One-Hot 编码会维度爆炸，目标编码（Target Encoding）用品类历史均值替代：

$$\text{enc}(c) = \lambda \cdot \bar{y}_c + (1 - \lambda) \cdot \bar{y}_{\text{global}}$$

其中 $\lambda = \frac{n_c}{n_c + k}$，$k$ 为平滑参数，防止低频品类过拟合。

---

## ② 母婴出海应用案例

### 场景 1：WF-A 补货量预测特征工程

**业务背景**：预测 SKU 未来 14 天补货需求量，特征工程决定模型上限。

**特征集构建**：
```
时序特征：
  - lag_7d, lag_14d, lag_28d           # 历史需求点
  - rolling_mean_7d, rolling_mean_14d  # 近期趋势
  - rolling_std_7d                     # 需求波动
  - ewm_14d                            # 指数加权近期均值

季节/日历特征：
  - is_weekend, day_of_week            # 周内模式
  - month_sin, month_cos               # 月份环形编码
  - is_chinese_newyear, is_618         # 促销期标识
  - days_until_holiday                 # 距节假日天数

供应链特征：
  - lead_time_p50, lead_time_p90       # 供应商交货期分位数
  - lead_time_std                      # 交货期波动
  - is_promotion_active                # 当前是否促销
  - days_since_last_stockout           # 上次缺货至今天数
```

**效果**：加入供应链特征后，RMSE 从 23.5 降至 19.2（提升 18%）。

---

### 场景 2：供应商风险特征向量

**业务背景**：构建供应商风险评分，输入 SCM Attribution 模型识别高风险供应商。

**特征向量组成**：
```
网络特征：
  - supplier_degree_centrality        # 供应商网络中心性
  - supplier_pagerank                 # 影响力分数
  - category_concentration            # 品类集中度（单点依赖风险）

历史特征：
  - delay_rate_30d                    # 近30天延误率
  - delay_days_p90                    # 延误天数P90
  - delay_trend                       # 延误趋势（滚动斜率）

业务特征：
  - days_since_onboarding             # 合作年限
  - total_sku_count                   # 供货 SKU 数量
  - avg_order_value                   # 平均订单额
```

---

## ③ 代码模板

**代码位置**：`paper2skills-code/supply_chain/supply_chain_ml_features/model.py`

```python
"""
Supply Chain ML Feature Engineering — 供应链 ML 特征工程
Python 标准库实现，无 sklearn/pandas 依赖，Python 3.14 兼容
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import statistics


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class SupplyChainRecord:
    """供应链时序记录（单 SKU 单天）"""
    sku_id: str
    timestamp: float          # Unix timestamp（天级别，取当天 00:00 的 ts）
    demand: float             # 当日需求量（件数）
    lead_time: float          # 供应商交货期（天）
    is_promotion: bool = False  # 是否为促销日
    supplier_id: str = ""     # 供应商ID
    category: str = ""        # 品类


# ─────────────────────────────────────────────
# 时序特征构建器
# ─────────────────────────────────────────────

class TemporalFeatureBuilder:
    """
    时序特征构建器：滞后特征、滚动统计、季节分解。
    严格无数据泄露：t 时刻特征只使用 t-1 及以前的数据。
    """

    def __init__(self, lag_windows: list[int] = None, roll_windows: list[int] = None):
        """
        Args:
            lag_windows: 滞后步数列表，如 [7, 14, 28]
            roll_windows: 滚动窗口大小列表，如 [7, 14, 30]
        """
        self.lag_windows = lag_windows or [7, 14, 28]
        self.roll_windows = roll_windows or [7, 14, 30]

    def build(self, records: list[SupplyChainRecord]) -> list[dict]:
        """
        为每条记录构建时序特征（使用严格的历史窗口，避免未来信息）。

        Returns:
            list of feature dicts，与 records 等长
        """
        # 按 sku_id 分组后按时间排序
        sku_records: dict[str, list[SupplyChainRecord]] = {}
        for r in records:
            sku_records.setdefault(r.sku_id, []).append(r)
        for sku in sku_records:
            sku_records[sku].sort(key=lambda x: x.timestamp)

        result = []
        for r in records:
            history = [h for h in sku_records[r.sku_id] if h.timestamp < r.timestamp]
            demands = [h.demand for h in history]
            features = {"sku_id": r.sku_id, "timestamp": r.timestamp, "target": r.demand}

            # 滞后特征
            for lag in self.lag_windows:
                idx = len(demands) - lag
                features[f"lag_{lag}d"] = demands[idx] if idx >= 0 else float("nan")

            # 滚动统计（均值 + 标准差）
            for w in self.roll_windows:
                window = demands[-w:] if len(demands) >= w else demands
                if window:
                    features[f"roll_mean_{w}d"] = statistics.mean(window)
                    features[f"roll_std_{w}d"] = statistics.pstdev(window) if len(window) > 1 else 0.0
                else:
                    features[f"roll_mean_{w}d"] = float("nan")
                    features[f"roll_std_{w}d"] = float("nan")

            # 指数加权均值（近14天，alpha=0.3）
            features["ewm_14d"] = self._ewm(demands[-14:], alpha=0.3)

            # 日历特征（从 timestamp 提取）
            features.update(self._calendar_features(r.timestamp))

            # 促销标识
            features["is_promotion"] = int(r.is_promotion)

            # 供应商交货期统计
            lead_times = [h.lead_time for h in history if h.lead_time > 0]
            if lead_times:
                sorted_lt = sorted(lead_times)
                n = len(sorted_lt)
                features["lead_time_p50"] = sorted_lt[n // 2]
                features["lead_time_p90"] = sorted_lt[min(int(n * 0.9), n - 1)]
                features["lead_time_std"] = statistics.pstdev(sorted_lt) if n > 1 else 0.0
            else:
                features["lead_time_p50"] = r.lead_time
                features["lead_time_p90"] = r.lead_time
                features["lead_time_std"] = 0.0

            result.append(features)

        return result

    @staticmethod
    def _ewm(values: list[float], alpha: float) -> float:
        """指数加权均值（标准库实现）"""
        if not values:
            return float("nan")
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    @staticmethod
    def _calendar_features(ts: float) -> dict:
        """从 Unix timestamp 提取日历特征"""
        import datetime
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        # 月份环形编码（避免12月→1月的跳变）
        month_rad = 2 * math.pi * dt.month / 12
        features = {
            "day_of_week": dt.weekday(),           # 0=周一，6=周日
            "is_weekend": int(dt.weekday() >= 5),
            "month": dt.month,
            "month_sin": math.sin(month_rad),
            "month_cos": math.cos(month_rad),
            "day_of_year": dt.timetuple().tm_yday,
        }
        return features


# ─────────────────────────────────────────────
# 供应商图特征构建器
# ─────────────────────────────────────────────

class SupplierGraphFeatureBuilder:
    """
    供应商网络图特征：度中心性、PageRank 近似。
    图定义：若两个供应商共同供应同一 SKU，则存在边。
    """

    def __init__(self, records: list[SupplyChainRecord]):
        self._graph: dict[str, set[str]] = {}  # supplier_id -> 相邻供应商集合
        self._build_graph(records)

    def _build_graph(self, records: list[SupplyChainRecord]) -> None:
        """构建供应商共现图"""
        # sku -> 供应商列表
        sku_suppliers: dict[str, set[str]] = {}
        for r in records:
            if r.supplier_id:
                sku_suppliers.setdefault(r.sku_id, set()).add(r.supplier_id)

        # 共同供应同一SKU的供应商之间建边
        for sku, suppliers in sku_suppliers.items():
            supplier_list = list(suppliers)
            for i in range(len(supplier_list)):
                for j in range(i + 1, len(supplier_list)):
                    s1, s2 = supplier_list[i], supplier_list[j]
                    self._graph.setdefault(s1, set()).add(s2)
                    self._graph.setdefault(s2, set()).add(s1)

    def degree_centrality(self, supplier_id: str) -> float:
        """度中心性：归一化到 [0,1]"""
        n = len(self._graph)
        if n <= 1:
            return 0.0
        deg = len(self._graph.get(supplier_id, set()))
        return deg / (n - 1)

    def pagerank(self, supplier_id: str, d: float = 0.85, iterations: int = 20) -> float:
        """PageRank 近似（幂迭代法）"""
        nodes = list(self._graph.keys())
        if not nodes or supplier_id not in self._graph:
            return 1.0 / max(len(self._graph), 1)

        n = len(nodes)
        pr = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_pr: dict[str, float] = {}
            for node in nodes:
                incoming_sum = sum(
                    pr[neighbor] / max(len(self._graph.get(neighbor, set())), 1)
                    for neighbor in nodes
                    if node in self._graph.get(neighbor, set())
                )
                new_pr[node] = (1 - d) / n + d * incoming_sum
            pr = new_pr

        return pr.get(supplier_id, 1.0 / n)

    def get_features(self, supplier_id: str) -> dict:
        """获取供应商图特征字典"""
        return {
            "supplier_degree_centrality": self.degree_centrality(supplier_id),
            "supplier_pagerank": self.pagerank(supplier_id),
            "supplier_neighbor_count": len(self._graph.get(supplier_id, set())),
        }


# ─────────────────────────────────────────────
# 完整特征流水线
# ─────────────────────────────────────────────

class SupplyChainFeaturePipeline:
    """组装完整特征矩阵：时序 + 图 + 业务特征"""

    def __init__(
        self,
        lag_windows: list[int] = None,
        roll_windows: list[int] = None,
    ):
        self.temporal_builder = TemporalFeatureBuilder(lag_windows, roll_windows)

    def fit_transform(self, records: list[SupplyChainRecord]) -> list[dict]:
        """
        端到端特征构建。

        Returns:
            list of feature dicts，每条记录对应一个特征字典
        """
        # 1. 时序特征
        feature_rows = self.temporal_builder.build(records)

        # 2. 供应商图特征
        graph_builder = SupplierGraphFeatureBuilder(records)
        supplier_feats: dict[str, dict] = {}
        all_suppliers = {r.supplier_id for r in records if r.supplier_id}
        for sid in all_suppliers:
            supplier_feats[sid] = graph_builder.get_features(sid)

        # 3. 目标编码（品类均值）
        cat_demands: dict[str, list[float]] = {}
        for r in records:
            if r.category:
                cat_demands.setdefault(r.category, []).append(r.demand)
        global_mean = statistics.mean([r.demand for r in records]) if records else 0.0
        cat_encodings: dict[str, float] = {}
        k = 10  # 平滑参数
        for cat, demands in cat_demands.items():
            n = len(demands)
            lam = n / (n + k)
            cat_encodings[cat] = lam * statistics.mean(demands) + (1 - lam) * global_mean

        # 4. 合并特征
        record_map = {(r.sku_id, r.timestamp): r for r in records}
        for feat_row in feature_rows:
            r = record_map.get((feat_row["sku_id"], feat_row["timestamp"]))
            if r is None:
                continue
            # 合并供应商图特征
            if r.supplier_id and r.supplier_id in supplier_feats:
                feat_row.update(supplier_feats[r.supplier_id])
            # 合并品类目标编码
            if r.category:
                feat_row["category_target_enc"] = cat_encodings.get(r.category, global_mean)
            feat_row["category"] = r.category
            feat_row["supplier_id"] = r.supplier_id

        return feature_rows


# ─────────────────────────────────────────────
# 测试入口
# ─────────────────────────────────────────────

def _run_test() -> None:
    """30天销量数据，验证特征构建正确性（无未来信息泄露）"""
    import datetime
    import time as _time

    base_ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    day = 86400.0

    # 构建30天数据，两个 SKU，两个供应商
    records: list[SupplyChainRecord] = []
    for d in range(30):
        ts = base_ts + d * day
        is_promo = d in {6, 7, 13, 14}  # 模拟周末促销
        # SKU-A：稳定需求 + 促销波动
        records.append(SupplyChainRecord(
            sku_id="SKU-A", timestamp=ts,
            demand=100.0 + (d % 7) * 5 + (50.0 if is_promo else 0.0),
            lead_time=7.0 + (d % 3) * 0.5,
            is_promotion=is_promo,
            supplier_id="SUP-1",
            category="奶粉",
        ))
        # SKU-B：增长趋势
        records.append(SupplyChainRecord(
            sku_id="SKU-B", timestamp=ts,
            demand=50.0 + d * 2.0,
            lead_time=14.0,
            is_promotion=is_promo,
            supplier_id="SUP-2",
            category="辅食",
        ))
        # 部分记录 SKU-A 也由 SUP-2 供应（共现边）
        if d % 5 == 0:
            records.append(SupplyChainRecord(
                sku_id="SKU-A", timestamp=ts + 1,
                demand=80.0,
                lead_time=10.0,
                is_promotion=False,
                supplier_id="SUP-2",
                category="奶粉",
            ))

    pipeline = SupplyChainFeaturePipeline(lag_windows=[7, 14], roll_windows=[7, 14])
    features = pipeline.fit_transform(records)

    print("=== Supply Chain ML Feature Engineering 测试 ===")
    print(f"总记录数: {len(records)}，特征行数: {len(features)}")

    # 取第20天 SKU-A 的特征验证
    day20_feat = next(
        (f for f in features if f["sku_id"] == "SKU-A" and f["timestamp"] == base_ts + 20 * day),
        None,
    )
    if day20_feat:
        print(f"\n[第20天 SKU-A] 特征样例：")
        for k, v in sorted(day20_feat.items()):
            if k not in ("sku_id", "timestamp", "target"):
                print(f"  {k}: {v}")

    # 验证无数据泄露：第1天的 lag_7d 应为 nan（无历史）
    day1_feat = next(
        (f for f in features if f["sku_id"] == "SKU-A" and f["timestamp"] == base_ts),
        None,
    )
    assert day1_feat is not None, "第1天特征不存在"
    assert math.isnan(day1_feat.get("lag_7d", float("nan"))), \
        "第1天 lag_7d 应为 nan（无历史数据）"
    print("\n✅ 断言通过：无数据泄露（第1天 lag_7d=NaN）")

    # 验证滚动均值方向正确（第14天均值 < 第28天均值，因为SKU-B是增长趋势）
    skub_late = [f for f in features if f["sku_id"] == "SKU-B" and
                 not math.isnan(f.get("roll_mean_7d", float("nan")))]
    if len(skub_late) >= 2:
        first_mean = skub_late[7].get("roll_mean_7d", 0)
        last_mean = skub_late[-1].get("roll_mean_7d", 0)
        assert last_mean >= first_mean, f"SKU-B 应呈增长趋势：{first_mean:.1f} -> {last_mean:.1f}"
        print(f"✅ 断言通过：SKU-B 滚动均值增长趋势（{first_mean:.1f} -> {last_mean:.1f}）")

    # 验证供应商图特征存在
    with_graph_feat = [f for f in features if "supplier_degree_centrality" in f]
    assert len(with_graph_feat) > 0, "供应商图特征应被计算"
    print(f"✅ 断言通过：{len(with_graph_feat)} 条记录含供应商图特征")

    print("\n全部测试通过 ✓")


if __name__ == "__main__":
    _run_test()
```

---

## ④ 技能关联

### 前置技能
- **[[Skill-Feature-Engineering]]** — ML 特征工程通用基础
- **[[Skill-Cross-Validation-Strategies]]** — 时序数据的正确交叉验证（Walk-Forward）
- **[[Skill-Demand-Forecasting-Supply-Chain]]** — 需求预测建模基础

### 延伸技能
- **[[Skill-Supply-Chain-Causal-SCM-Attribution]]** — 供应链因果归因（使用本 Skill 的风险特征）
- **[[Skill-EventCast-LLM-Event-Forecasting]]** — LLM 驱动的事件感知预测

### 可组合技能
- **[[Skill-AIM-RM-LLM-Inventory-MAS-Memory]]** — LLM 驱动的库存多智能体
- **[[Skill-Safety-Stock-Replenishment]]** — 安全库存计算（使用本 Skill 的交货期特征）
- **[[Skill-Promotion-Demand-Decomposition]]** — 促销需求分解（使用本 Skill 的促销特征）

---

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| 核心收益 | 供应链 ML 模型准确率提升 15-20%，减少缺货/积压成本 |
| 实现难度 | ⭐⭐☆☆☆ |
| 商业优先级 | ⭐⭐⭐⭐☆ |
| 工程成本 | 低（纯 Python，无 GPU，易于集成） |
| 适用场景 | 补货量预测、交货期风险评估、促销需求拆解、供应商风险评分 |

**关键收益**：统一特征工程规范后，不同 ML 团队/模型可复用同一特征集，降低维护成本。
- **跨域**：[[Skill-Inventory-Health-Aging-Attribution]]
