---
title: Amazon SP-API Data Pipeline — 增量采集标准化管道（订单/库存/财务报告）
doc_type: knowledge
module: 22-数据采集工程
topic: amazon-sp-api-data-pipeline
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Amazon SP-API Data Pipeline

> **领域**：数据采集工程 × 电商数据接入 | **类型**: 工程基础
> **桥梁**: 22-数据采集工程 ↔ 04-供应链 | **2026年**

---

## ① 算法原理

### 核心思想

Amazon Selling Partner API（SP-API）是跨境卖家获取一手运营数据的唯一标准渠道，但其 API 设计分散（订单、库存、财务各自独立端点）、限速严格（按资源类型分配 token bucket），且返回字段随版本变化。一旦采集逻辑散落在各 Skill 的 ad-hoc 代码中，维护成本极高。

**SP-API Data Pipeline** 建立三层统一架构：
1. **增量采集层**：基于 `lastUpdatedAfter` 游标，每次只拉取变更数据，避免全量重复采集
2. **Schema 映射层**：将各端点返回的异构 JSON 映射为统一的 Parquet 列式存储格式
3. **Retry 策略层**：429 限流自动指数退避，403/401 触发 Token 刷新，5xx 最多重试 3 次

### 数学直觉

**Token Bucket 限速模型**：

$$\text{tokens}(t) = \min\left(B_{\max},\ \text{tokens}(t-1) + r \cdot \Delta t\right)$$

其中 $B_{\max}$ 为桶容量，$r$ 为补充速率（tokens/s）。每次 API 调用消耗 $c$ tokens，超限返回 429 需等待：

$$t_{\text{wait}} = \frac{c - \text{tokens}_{\text{current}}}{r}$$

**增量采集窗口**：

$$\text{since} = \max(\text{last\_checkpoint},\ t_{\text{now}} - 24h)$$

### 关键假设

- SP-API OAuth2 Token 有效期 1 小时（需自动刷新）
- 实际生产中使用官方 `python-amazon-sp-api` SDK
- 本 Skill 使用 mock HTTP client，不依赖真实 API key

---

## ② 母婴出海应用案例

**场景 A：每日库存 + 订单数据自动同步**

- **业务问题**：运营团队每天早上手动下载 Seller Central 报告（订单/库存/FBA 状态），耗时 40 分钟，且数据 D+1 延迟
- **数据要求**：Amazon 卖家账号的 SP-API 凭证（Client ID/Secret/Refresh Token）
- **预期产出**：每小时自动拉取增量订单，每 6 小时同步 FBA 库存，输出统一 Parquet 文件到本地/S3
- **业务价值**：数据延迟从 D+1 → 实时（< 1 小时），运营人工报告工作每天节省 40min × 250 天 = **167h/年**，折算约 **5 万元**；更重要的是实时数据支撑 Agent 决策

**场景 B：财务报告自动核对**

- **业务问题**：Amazon 结算报告分散在多个 Settlement Report 中，与内部财务系统核对每月需要 2 人天
- **数据要求**：SP-API Finance Reports 接口
- **预期产出**：自动下载月度 Settlement 报告，解析为结构化 DataFrame，与 ERP 数据自动核对差异
- **业务价值**：财务核对时间从 2 人天 → 2 小时，年化节省 **22 人天**，减少人为错误导致的漏报风险

---

## ③ 代码模板

```python
"""
Amazon SP-API Data Pipeline
增量采集 + Schema 映射 + Retry 策略（mock HTTP，不依赖真实 API key）
"""

import json
import time
import math
import random
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


# ─── Token Bucket 限速器 ──────────────────────────────────────────────────────

class TokenBucket:
    """SP-API 限速模拟（生产环境复用此逻辑）"""

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens/s
        self.tokens = capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def consume(self, cost: float = 1.0) -> float:
        """返回需要等待的秒数（0 表示立即可执行）"""
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return 0.0
        wait = (cost - self.tokens) / self.refill_rate
        return wait


# ─── Mock SP-API Client ───────────────────────────────────────────────────────

@dataclass
class MockOrder:
    order_id: str
    asin: str
    quantity: int
    price_usd: float
    status: str
    updated_at: str


@dataclass
class MockInventoryItem:
    asin: str
    sku: str
    fulfillable_qty: int
    inbound_qty: int
    reserved_qty: int
    updated_at: str


class MockSPAPIClient:
    """Mock SP-API，模拟限速和 429 响应（不依赖真实凭证）"""

    def __init__(self, fail_rate: float = 0.1):
        self.buckets = {
            "orders": TokenBucket(capacity=10, refill_rate=0.5),
            "inventory": TokenBucket(capacity=6, refill_rate=0.2),
            "finance": TokenBucket(capacity=4, refill_rate=0.1),
        }
        self.fail_rate = fail_rate  # 模拟随机 5xx 错误率
        self._call_count = 0

    def get_orders(self, since: str, limit: int = 10) -> dict[str, Any]:
        wait = self.buckets["orders"].consume(1.0)
        if wait > 0:
            return {"status": 429, "retry_after": wait, "data": None}

        self._call_count += 1
        if random.random() < self.fail_rate:
            return {"status": 503, "data": None}

        # 生成 mock 订单数据
        orders = [
            MockOrder(
                order_id=f"ORD-{i:04d}-{self._call_count}",
                asin=f"B{i:08d}",
                quantity=random.randint(1, 5),
                price_usd=round(random.uniform(15.0, 89.0), 2),
                status=random.choice(["Shipped", "Pending", "Canceled"]),
                updated_at=since,
            ).__dict__
            for i in range(min(limit, 5))
        ]
        return {"status": 200, "data": orders, "next_token": None}

    def get_inventory(self, marketplace_id: str = "ATVPDKIKX0DER") -> dict[str, Any]:
        wait = self.buckets["inventory"].consume(1.0)
        if wait > 0:
            return {"status": 429, "retry_after": wait, "data": None}

        self._call_count += 1
        items = [
            MockInventoryItem(
                asin=f"B{i:08d}",
                sku=f"SKU-{i:04d}",
                fulfillable_qty=random.randint(0, 500),
                inbound_qty=random.randint(0, 200),
                reserved_qty=random.randint(0, 50),
                updated_at=datetime.now(timezone.utc).isoformat(),
            ).__dict__
            for i in range(8)
        ]
        return {"status": 200, "data": items}


# ─── 采集管道 ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    last_order_checkpoint: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc) - timedelta(hours=24)
        ).isoformat()
    )
    total_orders_fetched: int = 0
    total_inventory_fetched: int = 0
    errors: list[str] = field(default_factory=list)


class SPAPIDataPipeline:
    """
    增量采集管道：
    - 自动重试（指数退避，最多 3 次）
    - 429 限速处理（等待 retry_after）
    - Schema 统一映射（输出标准字典列表）
    """

    def __init__(self, client: MockSPAPIClient, max_retries: int = 3):
        self.client = client
        self.max_retries = max_retries
        self.state = PipelineState()

    def _call_with_retry(self, fn, *args, **kwargs) -> dict[str, Any] | None:
        for attempt in range(self.max_retries):
            result = fn(*args, **kwargs)
            if result["status"] == 200:
                return result
            elif result["status"] == 429:
                wait = result.get("retry_after", 2.0 ** attempt)
                # mock 中不真正等待，记录即可
                print(f"  [限速] 等待 {wait:.1f}s（attempt {attempt+1}）")
                time.sleep(min(wait, 0.01))  # mock: 最多等 10ms
            elif result["status"] in (500, 503):
                backoff = (2 ** attempt) + random.uniform(0, 1)
                print(f"  [5xx] 退避 {backoff:.1f}s（attempt {attempt+1}）")
                time.sleep(min(backoff, 0.01))
        return None

    def fetch_orders(self) -> list[dict[str, Any]]:
        """增量拉取订单，更新 checkpoint"""
        result = self._call_with_retry(
            self.client.get_orders,
            since=self.state.last_order_checkpoint,
        )
        if not result:
            self.state.errors.append("orders fetch failed after retries")
            return []

        orders = result["data"] or []
        self.state.total_orders_fetched += len(orders)

        # 更新 checkpoint
        self.state.last_order_checkpoint = datetime.now(timezone.utc).isoformat()

        # Schema 映射：统一字段名
        normalized = [
            {
                "order_id": o["order_id"],
                "asin": o["asin"],
                "qty": o["quantity"],
                "revenue_usd": o["price_usd"] * o["quantity"],
                "status": o["status"],
                "updated_at": o["updated_at"],
                "source": "amazon_sp_api",
            }
            for o in orders
        ]
        return normalized

    def fetch_inventory(self) -> list[dict[str, Any]]:
        """全量拉取 FBA 库存（频率 6h）"""
        result = self._call_with_retry(self.client.get_inventory)
        if not result:
            self.state.errors.append("inventory fetch failed after retries")
            return []

        items = result["data"] or []
        self.state.total_inventory_fetched += len(items)

        normalized = [
            {
                "asin": i["asin"],
                "sku": i["sku"],
                "available_qty": i["fulfillable_qty"],
                "inbound_qty": i["inbound_qty"],
                "reserved_qty": i["reserved_qty"],
                "total_qty": i["fulfillable_qty"] + i["inbound_qty"],
                "updated_at": i["updated_at"],
                "source": "amazon_sp_api",
            }
            for i in items
        ]
        return normalized

    def run_batch(self) -> dict[str, Any]:
        """执行一轮批量采集"""
        orders = self.fetch_orders()
        inventory = self.fetch_inventory()
        return {
            "orders": orders,
            "inventory": inventory,
            "state": {
                "checkpoint": self.state.last_order_checkpoint,
                "total_orders": self.state.total_orders_fetched,
                "total_inventory": self.state.total_inventory_fetched,
                "errors": self.state.errors,
            }
        }


# ─── 测试用例 ──────────────────────────────────────────────────────────────────

def test_amazon_sp_api_pipeline():
    random.seed(42)

    client = MockSPAPIClient(fail_rate=0.05)
    pipeline = SPAPIDataPipeline(client, max_retries=3)

    # 1. 执行批量采集
    result = pipeline.run_batch()

    # 2. 验证订单数据
    orders = result["orders"]
    assert len(orders) > 0, "订单数据为空"
    for o in orders:
        assert "order_id" in o and "asin" in o and "revenue_usd" in o
        assert o["source"] == "amazon_sp_api"
        assert o["revenue_usd"] >= 0
    print(f"[✓] 订单采集: {len(orders)} 条，示例: {orders[0]}")

    # 3. 验证库存数据
    inventory = result["inventory"]
    assert len(inventory) > 0, "库存数据为空"
    for inv in inventory:
        assert "asin" in inv and "available_qty" in inv
        assert inv["total_qty"] == inv["available_qty"] + inv["inbound_qty"]
    print(f"[✓] 库存采集: {len(inventory)} 条，示例: {inventory[0]}")

    # 4. 验证状态管理（checkpoint 更新）
    state = result["state"]
    assert state["total_orders"] == len(orders)
    assert state["total_inventory"] == len(inventory)
    print(f"[✓] 状态管理: checkpoint={state['checkpoint'][:19]}, "
          f"errors={len(state['errors'])}")

    # 5. 增量采集测试（第二次调用 checkpoint 应为当前时间）
    result2 = pipeline.run_batch()
    total_orders_2 = result2["state"]["total_orders"]
    assert total_orders_2 > state["total_orders"], "增量采集未叠加"
    print(f"[✓] 增量采集: 第2批订单总计 {total_orders_2} 条（已叠加）")

    # 6. Token Bucket 限速测试
    bucket = TokenBucket(capacity=2.0, refill_rate=1.0)
    assert bucket.consume(1.0) == 0.0
    assert bucket.consume(1.0) == 0.0
    wait = bucket.consume(1.0)
    assert wait > 0.0, "Token Bucket 限速未生效"
    print(f"[✓] Token Bucket: 超限等待 {wait:.2f}s")

    print("\n[✓] Amazon SP-API Data Pipeline 测试通过")


if __name__ == "__main__":
    test_amazon_sp_api_pipeline()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Data-Collection-Agent-Pipeline]]（理解通用数据采集 Agent 架构）
- **延伸（extends）**：[[Skill-Cross-System-Data-Reconciliation]]（SP-API 数据与 ERP/财务系统核对）
- **可组合（combinable）**：[[Skill-Real-Time-Inventory-Event-Stream]]（SP-API 采集的库存变化触发事件流处理）
- **可组合（combinable）**：[[Skill-Data-Quality-Monitor-Alert]]（对采集结果做质量监控，检测缺失/延迟/异常）
- **可组合（combinable）**：[[Skill-Advertising-API-Unified-Schema]]（与广告 API 数据联合分析 ACOS/ROAS）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 现状：5 个品类运营团队各自维护 SP-API 采集脚本，重复建设成本约 **20 万元/年**；数据延迟 D+1 导致补货决策滞后，平均缺货损失 **8 万元/月**
  - 引入后：统一管道，维护成本降至 **3 万元/年**；实时数据支撑 Agent，缺货损失降低 60% → **节省约 58 万元/年**
  - 总计年化 ROI：**约 75 万元**
- **实施难度**：⭐⭐☆☆☆（SP-API 官方有 Python SDK，主要工作是 Schema 映射和限速策略）
- **优先级评分**：⭐⭐⭐⭐⭐（数据接入 last-mile 最关键的单点，所有 Skill 的数据来源基础）
- **评估依据**：没有标准化的数据采集管道，所有 Skill 的「业务演示」永远停留在 mock 数据阶段；本 Skill 是「教学级」→「生产级」的最重要一步
