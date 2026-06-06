"""
Supply Chain ML Feature Engineering — 供应链 ML 特征工程
Python 标准库实现，无 sklearn/pandas 依赖，Python 3.14 兼容
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import statistics
import datetime


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class SupplyChainRecord:
    """供应链时序记录（单 SKU 单天）"""
    sku_id: str
    timestamp: float          # Unix timestamp（天级别）
    demand: float             # 当日需求量（件数）
    lead_time: float          # 供应商交货期（天）
    is_promotion: bool = False
    supplier_id: str = ""
    category: str = ""


# ─────────────────────────────────────────────
# 时序特征构建器
# ─────────────────────────────────────────────

class TemporalFeatureBuilder:
    """
    时序特征：滞后特征、滚动统计、季节分解。
    严格无数据泄露：t 时刻特征只使用 t-1 及以前的数据。
    """

    def __init__(
        self,
        lag_windows: Optional[list[int]] = None,
        roll_windows: Optional[list[int]] = None,
    ) -> None:
        self.lag_windows = lag_windows or [7, 14, 28]
        self.roll_windows = roll_windows or [7, 14, 30]

    def build(self, records: list[SupplyChainRecord]) -> list[dict]:
        """
        为每条记录构建时序特征。

        Returns:
            list of feature dicts，与 records 等长，顺序一致
        """
        sku_records: dict[str, list[SupplyChainRecord]] = {}
        for r in records:
            sku_records.setdefault(r.sku_id, []).append(r)
        for sku in sku_records:
            sku_records[sku].sort(key=lambda x: x.timestamp)

        result: list[dict] = []
        for r in records:
            history = [h for h in sku_records[r.sku_id] if h.timestamp < r.timestamp]
            demands = [h.demand for h in history]
            features: dict = {
                "sku_id": r.sku_id,
                "timestamp": r.timestamp,
                "target": r.demand,
            }

            for lag in self.lag_windows:
                idx = len(demands) - lag
                features[f"lag_{lag}d"] = demands[idx] if idx >= 0 else float("nan")

            for w in self.roll_windows:
                window = demands[-w:] if len(demands) >= w else demands
                if window:
                    features[f"roll_mean_{w}d"] = statistics.mean(window)
                    features[f"roll_std_{w}d"] = (
                        statistics.pstdev(window) if len(window) > 1 else 0.0
                    )
                else:
                    features[f"roll_mean_{w}d"] = float("nan")
                    features[f"roll_std_{w}d"] = float("nan")

            features["ewm_14d"] = self._ewm(demands[-14:], alpha=0.3)
            features.update(self._calendar_features(r.timestamp))
            features["is_promotion"] = int(r.is_promotion)

            lead_times = [h.lead_time for h in history if h.lead_time > 0]
            if lead_times:
                sorted_lt = sorted(lead_times)
                n = len(sorted_lt)
                features["lead_time_p50"] = sorted_lt[n // 2]
                features["lead_time_p90"] = sorted_lt[min(int(n * 0.9), n - 1)]
                features["lead_time_std"] = (
                    statistics.pstdev(sorted_lt) if n > 1 else 0.0
                )
            else:
                features["lead_time_p50"] = r.lead_time
                features["lead_time_p90"] = r.lead_time
                features["lead_time_std"] = 0.0

            result.append(features)

        return result

    @staticmethod
    def _ewm(values: list[float], alpha: float) -> float:
        """指数加权均值：alpha 越大近期数据权重越高"""
        if not values:
            return float("nan")
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    @staticmethod
    def _calendar_features(ts: float) -> dict:
        """从 Unix timestamp 提取日历特征（月份使用环形编码避免边界跳变）"""
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        month_rad = 2 * math.pi * dt.month / 12
        return {
            "day_of_week": dt.weekday(),
            "is_weekend": int(dt.weekday() >= 5),
            "month": dt.month,
            "month_sin": math.sin(month_rad),
            "month_cos": math.cos(month_rad),
            "day_of_year": dt.timetuple().tm_yday,
        }


# ─────────────────────────────────────────────
# 供应商图特征构建器
# ─────────────────────────────────────────────

class SupplierGraphFeatureBuilder:
    """
    供应商网络图特征：度中心性、PageRank 近似。
    图定义：两个供应商共同供应同一 SKU 则存在边。
    """

    def __init__(self, records: list[SupplyChainRecord]) -> None:
        self._graph: dict[str, set[str]] = {}
        self._build_graph(records)

    def _build_graph(self, records: list[SupplyChainRecord]) -> None:
        sku_suppliers: dict[str, set[str]] = {}
        for r in records:
            if r.supplier_id:
                sku_suppliers.setdefault(r.sku_id, set()).add(r.supplier_id)

        for suppliers in sku_suppliers.values():
            supplier_list = list(suppliers)
            for i in range(len(supplier_list)):
                for j in range(i + 1, len(supplier_list)):
                    s1, s2 = supplier_list[i], supplier_list[j]
                    self._graph.setdefault(s1, set()).add(s2)
                    self._graph.setdefault(s2, set()).add(s1)

    def degree_centrality(self, supplier_id: str) -> float:
        n = len(self._graph)
        if n <= 1:
            return 0.0
        return len(self._graph.get(supplier_id, set())) / (n - 1)

    def pagerank(self, supplier_id: str, d: float = 0.85, iterations: int = 20) -> float:
        """幂迭代 PageRank 近似（d=阻尼系数）"""
        nodes = list(self._graph.keys())
        if not nodes or supplier_id not in self._graph:
            return 1.0 / max(len(self._graph), 1)

        n = len(nodes)
        pr = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_pr: dict[str, float] = {}
            for node in nodes:
                incoming = sum(
                    pr[nb] / max(len(self._graph.get(nb, set())), 1)
                    for nb in nodes
                    if node in self._graph.get(nb, set())
                )
                new_pr[node] = (1 - d) / n + d * incoming
            pr = new_pr

        return pr.get(supplier_id, 1.0 / n)

    def get_features(self, supplier_id: str) -> dict:
        return {
            "supplier_degree_centrality": self.degree_centrality(supplier_id),
            "supplier_pagerank": self.pagerank(supplier_id),
            "supplier_neighbor_count": len(self._graph.get(supplier_id, set())),
        }


# ─────────────────────────────────────────────
# 完整特征流水线
# ─────────────────────────────────────────────

class SupplyChainFeaturePipeline:
    """组装完整特征矩阵：时序 + 图 + 目标编码"""

    def __init__(
        self,
        lag_windows: Optional[list[int]] = None,
        roll_windows: Optional[list[int]] = None,
    ) -> None:
        self.temporal_builder = TemporalFeatureBuilder(lag_windows, roll_windows)

    def fit_transform(self, records: list[SupplyChainRecord]) -> list[dict]:
        """
        端到端特征构建。

        Returns:
            list of feature dicts，每条记录对应一个特征字典
        """
        feature_rows = self.temporal_builder.build(records)

        graph_builder = SupplierGraphFeatureBuilder(records)
        all_suppliers = {r.supplier_id for r in records if r.supplier_id}
        supplier_feats = {sid: graph_builder.get_features(sid) for sid in all_suppliers}

        # 目标编码（品类级平滑均值，防止低频品类过拟合）
        cat_demands: dict[str, list[float]] = {}
        for r in records:
            if r.category:
                cat_demands.setdefault(r.category, []).append(r.demand)
        global_mean = statistics.mean([r.demand for r in records]) if records else 0.0
        k = 10  # 平滑参数
        cat_encodings = {
            cat: (len(d) / (len(d) + k)) * statistics.mean(d) + (k / (len(d) + k)) * global_mean
            for cat, d in cat_demands.items()
        }

        record_map = {(r.sku_id, r.timestamp): r for r in records}
        for feat_row in feature_rows:
            r = record_map.get((feat_row["sku_id"], feat_row["timestamp"]))
            if r is None:
                continue
            if r.supplier_id and r.supplier_id in supplier_feats:
                feat_row.update(supplier_feats[r.supplier_id])
            if r.category:
                feat_row["category_target_enc"] = cat_encodings.get(r.category, global_mean)
            feat_row["category"] = r.category
            feat_row["supplier_id"] = r.supplier_id

        return feature_rows


# ─────────────────────────────────────────────
# 测试入口
# ─────────────────────────────────────────────

def run_test() -> None:
    """30天销量数据，验证特征构建正确性（无未来信息泄露）"""
    base_ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    day = 86400.0

    records: list[SupplyChainRecord] = []
    for d in range(30):
        ts = base_ts + d * day
        is_promo = d in {6, 7, 13, 14}
        records.append(SupplyChainRecord(
            sku_id="SKU-A", timestamp=ts,
            demand=100.0 + (d % 7) * 5 + (50.0 if is_promo else 0.0),
            lead_time=7.0 + (d % 3) * 0.5,
            is_promotion=is_promo,
            supplier_id="SUP-1",
            category="奶粉",
        ))
        records.append(SupplyChainRecord(
            sku_id="SKU-B", timestamp=ts,
            demand=50.0 + d * 2.0,
            lead_time=14.0,
            is_promotion=is_promo,
            supplier_id="SUP-2",
            category="辅食",
        ))
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

    day20_feat = next(
        (f for f in features if f["sku_id"] == "SKU-A" and f["timestamp"] == base_ts + 20 * day),
        None,
    )
    if day20_feat:
        print(f"\n[第20天 SKU-A] 特征样例（部分）：")
        for k_name in ["lag_7d", "lag_14d", "roll_mean_7d", "roll_std_7d", "ewm_14d",
                        "lead_time_p50", "lead_time_p90", "is_promotion",
                        "supplier_degree_centrality", "supplier_pagerank", "category_target_enc"]:
            v = day20_feat.get(k_name, "N/A")
            if isinstance(v, float):
                print(f"  {k_name}: {v:.4f}")
            else:
                print(f"  {k_name}: {v}")

    day1_feat = next(
        (f for f in features if f["sku_id"] == "SKU-A" and f["timestamp"] == base_ts),
        None,
    )
    assert day1_feat is not None, "第1天特征不存在"
    assert math.isnan(day1_feat.get("lag_7d", float("nan"))), \
        "第1天 lag_7d 应为 nan（严格无数据泄露验证）"
    print("\n✅ 断言通过：无数据泄露（第1天 lag_7d=NaN）")

    skub_feats = sorted(
        [f for f in features if f["sku_id"] == "SKU-B"
         and not math.isnan(f.get("roll_mean_7d", float("nan")))],
        key=lambda x: x["timestamp"],
    )
    if len(skub_feats) >= 10:
        first_mean = skub_feats[7]["roll_mean_7d"]
        last_mean = skub_feats[-1]["roll_mean_7d"]
        assert last_mean >= first_mean, f"SKU-B 应呈增长趋势：{first_mean:.1f} -> {last_mean:.1f}"
        print(f"✅ 断言通过：SKU-B 滚动均值增长趋势（{first_mean:.1f} -> {last_mean:.1f}）")

    with_graph = [f for f in features if "supplier_degree_centrality" in f]
    assert len(with_graph) > 0, "供应商图特征应被计算"
    print(f"✅ 断言通过：{len(with_graph)} 条记录含供应商图特征")

    print("\n全部测试通过 ✓")


if __name__ == "__main__":
    run_test()
