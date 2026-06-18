---
title: 供应链实时特征存储架构 — Online/Offline分离+毫秒级读取的Zalando母婴SKU模式
doc_type: knowledge
module: 24-标签工程
topic: online-feature-store-sc-realtime
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 供应链实时特征存储架构

> **来源**：Zalando Engineering Blog 2025（Inventory Optimisation System：500万SKU特征管道）+ SageMaker Feature Store 设计文档 + Feast（开源特征存储）最佳实践
> **桥梁**：标签工程 ↔ 数据工程 ↔ Palantir OKB Layer | **类型**：数据工程+特征工程

## ① 算法原理

**特征存储是 Agent 实时决策的"弹药库"**：LLM/ML 模型做出决策需要大量上下文特征（当前库存/近7日销量/竞品价格/促销状态/供应商可用性），如果每次决策都实时计算，延迟 >5 秒无法接受；如果只用离线批量，特征会过期 24 小时。

**Online/Offline 分离架构**（Zalando 模式）：

```
离线存储 (Offline Store)           在线存储 (Online Store)
──────────────────────             ───────────────────────
S3 / Delta Lake                    DynamoDB / Redis
批量历史特征                        最新特征快照
训练数据 / 回测                     实时推理 <20ms
GB-TB 级别                         GB 级别（热数据）
每日/小时刷新                       秒级推送

             ↕ Feature Pipeline ↕
             (Spark / PySpark 计算)
```

**Zalando 500万SKU特征管道**（2 小时完成全量）：

```
Step 1: 数据预处理 (PySpark on Databricks)
  - 销售历史聚合（2.5年滑动窗口，7/14/28/90天）
  - 库存 & 可用性 JOIN
  - 定价 & 元数据关联

Step 2: 特征转换 (SageMaker Processing + Numba 加速)
  - Lag 特征（t-1/t-7/t-28）
  - 假期指示变量（节日前后窗口）
  - 标准化 & 类别编码

Step 3: 双写 (Online + Offline)
  - Offline: S3 Parquet（追加模式，保留历史）
  - Online: DynamoDB（覆盖写，最新版本）
  - 延迟: 每日批量 <2 小时（500万SKU）
```

**核心特征维度设计**（供应链决策适用）：

```python
SC_FEATURE_SCHEMA = {
    # 库存类特征（实时性高，Online为主）
    "inventory_features": [
        "current_stock", "in_transit_qty", "dos_current",
        "dos_forecast_7d", "reorder_flag", "overstock_flag"
    ],
    # 需求类特征（批量计算，Offline为主）
    "demand_features": [
        "sales_7d_avg", "sales_28d_avg", "sales_yoy_ratio",
        "promo_lift_factor", "seasonality_index", "trend_slope"
    ],
    # 供应类特征（日更新）
    "supply_features": [
        "supplier_lead_time_p50", "supplier_reliability_score",
        "inbound_eta_accuracy", "po_coverage_days"
    ],
    # 风险类特征（事件驱动更新）
    "risk_features": [
        "stockout_risk_score", "overstock_risk_score",
        "supplier_risk_score", "logistics_delay_prob"
    ]
}
```

## ② 母婴出海应用案例

**场景A：Agent 实时决策时的特征拉取**

补货 Agent 需要决策某 SKU 是否触发紧急补货，在线读取 10 维特征仅需 15ms，包括：当前库存(50件)、在途量(200件)、日均销量(25件)、供应商可靠性(87分)、当前促销状态(无)、缺货风险(68分)。

无特征存储时，Agent 需要分别查询 ERP/WMS/预测服务，总延迟 >3 秒；有了 Online Store，15ms 完成，Agent 可以实时决策而不阻塞。

**数据要求**：ERP 库存 API、WMS 在途数据、预测服务输出
**预期产出**：特征 Pipeline（每日全量刷新 <2 小时）+ Online Store（<20ms 读取）
**业务价值**：Agent 决策延迟从 >3 秒 → <100ms，支撑每日 100K+ 次实时决策

**场景B：模型训练的 Point-in-Time 特征重建**

补货模型需要用"历史某时刻的正确特征"训练（避免 label leakage），Offline Store 的追加模式保存了所有历史快照，支持 point-in-time join。

**数据要求**：Offline Store 历史特征快照
**预期产出**：无泄漏的训练数据集 + 回测准确的特征评估
**业务价值**：避免因特征泄漏导致的模型过拟合，实际部署准确率提升 15-30%

## ③ 代码模板

```python
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import math

@dataclass
class FeatureRecord:
    """特征记录（带时间戳）"""
    entity_id: str       # SKU ID
    features: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    version: int = 1

class SCOnlineFeatureStore:
    """
    供应链在线特征存储（内存实现）
    生产环境：替换为 DynamoDB / Redis 客户端调用
    
    核心设计：
    - <20ms 读取延迟
    - TTL 自动过期
    - 批量写入优化
    """
    
    def __init__(self, ttl_seconds: int = 86400):  # 默认24小时过期
        self._store: Dict[str, FeatureRecord] = {}
        self.ttl = ttl_seconds
        self.read_count = 0
        self.cache_hits = 0
    
    def write(self, entity_id: str, features: Dict[str, float]) -> bool:
        """单条写入（覆盖最新值）"""
        self._store[entity_id] = FeatureRecord(
            entity_id=entity_id,
            features=features.copy(),
            timestamp=time.time()
        )
        return True
    
    def batch_write(self, records: List[Dict]) -> Dict:
        """批量写入（优化吞吐）"""
        success, failed = 0, 0
        for rec in records:
            try:
                self.write(rec["entity_id"], rec["features"])
                success += 1
            except Exception:
                failed += 1
        return {"success": success, "failed": failed, "total": len(records)}
    
    def read(self, entity_id: str, 
              feature_names: Optional[List[str]] = None) -> Optional[Dict]:
        """在线读取（目标延迟 <20ms）"""
        self.read_count += 1
        record = self._store.get(entity_id)
        if record is None:
            return None
        
        # TTL 检查
        if time.time() - record.timestamp > self.ttl:
            del self._store[entity_id]
            return None
        
        self.cache_hits += 1
        if feature_names:
            return {k: record.features.get(k) for k in feature_names}
        return record.features.copy()
    
    def batch_read(self, entity_ids: List[str],
                    feature_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """批量读取"""
        results = {}
        for eid in entity_ids:
            feat = self.read(eid, feature_names)
            if feat is not None:
                results[eid] = feat
        return results
    
    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.read_count, 1)

class SCFeaturePipeline:
    """
    特征计算管道（Zalando 模式）
    
    输入: 原始业务数据（库存/销售/供应商）
    输出: 结构化特征向量 → 写入 Online/Offline Store
    """
    
    def __init__(self, online_store: SCOnlineFeatureStore):
        self.online_store = online_store
    
    def compute_inventory_features(self, sku: str, 
                                    stock: int, in_transit: int,
                                    daily_sales: List[float],
                                    reorder_point: int) -> Dict[str, float]:
        """计算库存类特征"""
        avg_sales_7d = sum(daily_sales[-7:]) / max(len(daily_sales[-7:]), 1)
        avg_sales_28d = sum(daily_sales[-28:]) / max(len(daily_sales[-28:]), 1)
        
        total_available = stock + in_transit
        dos = total_available / max(avg_sales_7d, 0.1)  # Days of Supply
        
        return {
            "current_stock": float(stock),
            "in_transit_qty": float(in_transit),
            "dos_current": round(dos, 1),
            "sales_7d_avg": round(avg_sales_7d, 2),
            "sales_28d_avg": round(avg_sales_28d, 2),
            "reorder_flag": float(stock < reorder_point),
            "overstock_flag": float(dos > 90),
            "trend_slope": self._compute_trend(daily_sales[-14:] if len(daily_sales) >= 14 else daily_sales),
        }
    
    def compute_risk_features(self, dos: float, supplier_reliability: float,
                               logistics_delay_prob: float) -> Dict[str, float]:
        """计算风险类特征"""
        # 缺货风险：DOS < 14天 + 供应商可靠性低
        stockout_risk = min(100, max(0,
            (14 - dos) * 5 + (100 - supplier_reliability) * 0.3
        ))
        # 呆滞风险：DOS > 90天
        overstock_risk = min(100, max(0, (dos - 90) * 0.5))
        
        return {
            "stockout_risk_score": round(stockout_risk, 1),
            "overstock_risk_score": round(overstock_risk, 1),
            "supplier_reliability_score": float(supplier_reliability),
            "logistics_delay_prob": float(logistics_delay_prob),
        }
    
    def _compute_trend(self, sales_series: List[float]) -> float:
        """简单线性趋势斜率（单位：件/天）"""
        n = len(sales_series)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(sales_series) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(sales_series))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return round(numerator / max(denominator, 0.001), 4)
    
    def run_full_pipeline(self, sku_data_list: List[Dict]) -> Dict:
        """
        全量特征计算管道（对应 Zalando 每日批量更新）
        
        Args:
            sku_data_list: [{sku, stock, in_transit, daily_sales, 
                             reorder_point, supplier_reliability, logistics_delay_prob}]
        """
        start_time = time.time()
        records = []
        
        for data in sku_data_list:
            sku = data["sku"]
            inv_features = self.compute_inventory_features(
                sku, data["stock"], data["in_transit"],
                data["daily_sales"], data["reorder_point"]
            )
            risk_features = self.compute_risk_features(
                inv_features["dos_current"],
                data.get("supplier_reliability", 80),
                data.get("logistics_delay_prob", 0.1)
            )
            all_features = {**inv_features, **risk_features}
            records.append({"entity_id": sku, "features": all_features})
        
        write_result = self.online_store.batch_write(records)
        elapsed = time.time() - start_time
        
        return {
            "skus_processed": len(sku_data_list),
            "write_result": write_result,
            "pipeline_time_seconds": round(elapsed, 3),
            "throughput_skus_per_sec": round(len(sku_data_list) / max(elapsed, 0.001))
        }


# ===== 测试用例 =====
def run_test():
    import random
    random.seed(42)
    
    store = SCOnlineFeatureStore(ttl_seconds=3600)
    pipeline = SCFeaturePipeline(store)
    
    # 生成 100 个 SKU 的测试数据
    sku_data = []
    for i in range(100):
        sku_data.append({
            "sku": f"BABY-SKU-{i:03d}",
            "stock": random.randint(20, 500),
            "in_transit": random.randint(0, 300),
            "daily_sales": [random.gauss(20, 5) for _ in range(30)],
            "reorder_point": 150,
            "supplier_reliability": random.uniform(60, 99),
            "logistics_delay_prob": random.uniform(0.05, 0.35)
        })
    
    # 运行管道
    result = pipeline.run_full_pipeline(sku_data)
    assert result["write_result"]["success"] == 100, "应成功写入100个SKU特征"
    print(f"  特征管道: {result['skus_processed']} SKUs, {result['pipeline_time_seconds']:.3f}s")
    print(f"  吞吐量: {result['throughput_skus_per_sec']:,} SKUs/秒")
    
    # 测试在线读取
    t0 = time.time()
    features = store.read("BABY-SKU-000", 
                          feature_names=["current_stock", "dos_current", "stockout_risk_score"])
    read_ms = (time.time() - t0) * 1000
    
    assert features is not None, "应能读到特征"
    assert "stockout_risk_score" in features, "应包含风险特征"
    assert read_ms < 100, f"在线读取应<100ms，实际{read_ms:.1f}ms"
    print(f"  在线读取延迟: {read_ms:.2f}ms (目标<20ms)")
    print(f"  特征样例: DOS={features['dos_current']:.1f}天, 风险={features['stockout_risk_score']:.1f}/100")
    
    # 测试批量读取
    batch_result = store.batch_read(["BABY-SKU-001", "BABY-SKU-002", "BABY-SKU-999"])
    assert len(batch_result) == 2, "BABY-SKU-999不存在，应返回2条"
    print(f"  批量读取: {len(batch_result)}/3 命中 (命中率{store.hit_rate:.0%})")
    
    # 测试高风险SKU识别
    high_risk = []
    for i in range(50):
        f = store.read(f"BABY-SKU-{i:03d}", ["stockout_risk_score", "dos_current"])
        if f and f.get("stockout_risk_score", 0) > 50:
            high_risk.append(f"BABY-SKU-{i:03d}")
    print(f"  高风险SKU识别: {len(high_risk)} 个需要紧急关注")
    
    print("\n[✓] Online-Feature-Store-SC 测试通过 — 批量计算+实时读取+风险识别就绪")

run_test()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supply-Chain-Data-Lineage-Tracking]] — 特征血缘追踪是特征质量的保障
- **前置（prerequisite）**：[[Skill-Tag-Schema-Engineering-Lifecycle]] — 特征 Schema 设计与标签工程同源
- **延伸（extends）**：[[Skill-Graph-OKB-Design-SC]] — OKB 图谱与特征存储互补（关系推理 vs 数值特征）
- **延伸（extends）**：[[Skill-Real-Time-Supply-Chain-Drift-Detection]] — 特征漂移检测是特征存储的质量监控层
- **可组合（combinable）**：[[Skill-SCPA-Autonomous-SC-Planning-Agent]] — SCPA Agent 从 Online Store 实时拉取特征做决策
- **可组合（combinable）**：[[Skill-LLM-SC-MultiAgent-Consensus-Replenishment]] — 多智能体需要实时特征作为决策输入

## ⑤ 商业价值评估

- **ROI 预估**：Zalando 案例：500万SKU特征管道 <2 小时（全量刷新），在线特征 10-20ms 读取；Agent 决策延迟从 >3 秒 → <100ms；模型避免特征泄漏准确率提升 15-30%
- **实施难度**：⭐⭐⭐⭐☆（DynamoDB/Redis 配置 + Spark 管道是主要工程量）
- **优先级**：⭐⭐⭐⭐☆（企业 AI 知识库的数值特征层，Agent 实时决策的前提基础设施）
- **企业AI知识库依赖**：高 — 特征存储即是 AI 知识库的实时数值层，所有 ML/LLM 模型依赖于此
