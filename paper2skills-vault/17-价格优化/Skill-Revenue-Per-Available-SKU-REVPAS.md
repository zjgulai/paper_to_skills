---
title: Revenue Per Available SKU（REVPAS）— 以可售库存单位衡量定价效率
doc_type: knowledge
module: 17-价格优化
topic: revenue-per-available-sku-revpas
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Revenue Per Available SKU（REVPAS）

> **核心**：把酒店 RevPAR 的思路迁移到 SKU 层，衡量“每个可售库存单位”贡献了多少收入，而不是只看 ROAS、销量或毛利率。

## ① 算法原理
REVPAS = 实际销售额 / 可售库存单位数，也可拆成 **均价 × 售罄率**。它把价格效率和库存效率放进同一指标：高价但卖不动、低价但卖得快，都可能在 REVPAS 上暴露问题。做法是先按 SKU 计算 REVPAS，再放入「价格效率 × 库存效率」矩阵：横轴看价格偏离目标带来的收入损失，纵轴看周转/售罄损失，优先定位左下角低 REVPAS SKU。关键假设是库存口径、统计周期和促销口径一致，否则指标会被短促噪声污染。

## ② 母婴出海应用案例
**场景A：奶瓶/吸奶器 SKU 定价复盘**
- 业务问题：同一品类里有些 SKU 看起来销量不错，但占用库存多、单件收入低
- 数据要求：SKU 维度的销量、销售额、可售库存、定价、促销标签
- 预期产出：REVPAS 排名、低效 SKU 清单、建议调价幅度
- 业务价值：把资源从“高销量低效率”SKU 迁移到更高收益 SKU

**场景B：FBA 备货与定价联动**
- 业务问题：补货后库存增加，但收入没有同步增长，导致仓储费抬升
- 数据要求：FBA 库存、断货天数、周销量、ASP、毛利
- 预期产出：REVPAS 低点预警、补货和降价联动建议
- 业务价值：减少滞销库存，提升每个可售库存单位的现金回报

## ③ 代码模板
```python
from dataclasses import dataclass
from typing import List


@dataclass
class SKURecord:
    sku: str
    revenue: float
    available_units: int
    sold_units: int
    target_revpAS: float = 0.0


def calculate_revpas(record: SKURecord) -> float:
    if record.available_units <= 0:
        return 0.0
    return record.revenue / record.available_units


def analyze_revpas(records: List[SKURecord]):
    rows = []
    for r in records:
        revpas = calculate_revpas(r)
        sell_through = r.sold_units / r.available_units if r.available_units else 0.0
        avg_price = r.revenue / r.sold_units if r.sold_units else 0.0
        rows.append({
            "sku": r.sku,
            "revpas": round(revpas, 2),
            "avg_price": round(avg_price, 2),
            "sell_through": round(sell_through, 3),
            "gap": round((r.target_revpAS or revpas) - revpas, 2),
        })
    rows.sort(key=lambda x: x["revpas"])
    return rows


def recommend_actions(rows):
    recs = []
    for row in rows:
        if row["gap"] > 0:
            recs.append(f"{row['sku']}: 优先提价/优化促销，目标 REVPAS 需提升 {row['gap']}")
        elif row["sell_through"] < 0.5:
            recs.append(f"{row['sku']}: 先做去库存，避免低周转继续稀释 REVPAS")
        else:
            recs.append(f"{row['sku']}: 维持现价，继续观察")
    return recs


def main():
    sample = [
        SKURecord("Bottle-A", 12000, 400, 240, 35),
        SKURecord("Pump-B", 9800, 500, 140, 28),
        SKURecord("Wipes-C", 7600, 300, 250, 22),
    ]
    rows = analyze_revpas(sample)
    recs = recommend_actions(rows)
    print("REVPAS ranking:")
    for row in rows:
        print(row)
    print("Recommendations:")
    for item in recs:
        print(item)
    assert rows[0]["sku"] == "Pump-B"
    print("[✓] REVPAS 测试通过")


if __name__ == "__main__":
    main()
```

## ④ 技能关联
- 前置：[[Skill-SKU-Level-PL-Dashboard]]
- 前置：[[Skill-FBA-Fee-Intelligence]]
- 延伸：[[Skill-Cross-Border-Price-Harmonization]]
- 可组合：[[Skill-Bundle-Pricing-Strategy]]（适合捆绑定价与库存清理联动）

## ⑤ 商业价值评估
- ROI 预估：识别低 REVPAS SKU 后做定价/库存优化，年化毛利提升约 $2.4 万
- 实施难度：⭐⭐⭐☆☆
- 优先级：⭐⭐⭐⭐☆
- 评估依据：指标直接连接收入和库存利用率，能快速定位低效 SKU
