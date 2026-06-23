---
title: Inventory Theft Warehouse Anomaly — 海外仓库存异常检测盗窃/错发/损耗识别
doc_type: knowledge
module: 19-风控反欺诈
topic: inventory-theft-warehouse-anomaly
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Inventory-Theft-Warehouse-Anomaly

## ① 算法原理（≤300字）

**核心问题**：母婴卖家使用海外第三方仓库（3PL）时，库存差异（Inventory Discrepancy）是常见风险——盗窃、错误出库、损耗（破损/过期）、记录错误都可能导致「账面库存 > 实际库存」。每次 FBA 补仓时才发现差异，损失已无法追溯。

**差异检测模型**：

**基线模型**：建立库存消耗速率基线：
$$E[\Delta_t] = \text{SalesOrders}_t + \text{FBA\_Shipped}_t - \text{Restocking}_t$$
预期库存变化量 = 销售出库 + FBA 发货 - 补货入库

**异常检测**：
$$\text{Discrepancy}_t = \text{Actual\_Inventory}_t - \hat{\text{Expected\_Inventory}}_t$$

当 $|\text{Discrepancy}_t| > k \cdot \sigma_{\text{hist}}$（历史正常波动的 k 倍）时触发告警。

**差异分类**：

| 差异模式 | 可能原因 | 处理建议 |
|---------|---------|---------|
| 持续小量缺少（-1至-3件/天）| 内部盗窃 | 视频监控 + 随机盘点 |
| 突发大量缺少（单次-50件+） | 错误出库 / 整批盗窃 | 紧急核查出库记录 |
| 定期规律缺少（每周五） | 系统性问题 | 核查对应班次员工 |
| 慢速损耗（月级） | 过期 / 破损 | 改善仓储条件 + 保质期管理 |

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某母婴卖家在美国海外仓存放奶粉（高价值品，约 $50/罐），3 个月内共 220 件库存差异（账面多于实际），发现规律是每周二、四各缺少约 8-12 件。经比对出库记录和运单，发现是某班次操作员系统性错误记录。

**数据要求**：WMS（仓库管理系统）库存日志、出库单、运单号，每日盘点记录。

**异常检测应用**：持续偏差检测在第 2 周触发告警，3 个月内追回损失，而非年底对账时才发现。

**量化产出**：及时发现 220 件 × $50 = $11,000 损失，年化通过改进追回并预防潜在盗损 **10-20 万元**。

## ③ 代码模板

```python
import numpy as np
from collections import defaultdict

def inventory_anomaly_detector(
    inventory_records: list,  # [{'date': int, 'actual': float, 'sales': float, 'shipped': float, 'restocked': float}]
    k: float = 2.5,
    warmup_days: int = 14
) -> dict:
    """
    库存异常检测
    """
    n = len(inventory_records)
    if n < warmup_days + 1:
        return {'error': '数据不足'}

    discrepancies = []
    expected_changes = []

    # 计算每天的预期变化和实际变化
    for i in range(1, n):
        prev = inventory_records[i - 1]
        curr = inventory_records[i]

        expected_change = -(curr['sales'] + curr['shipped']) + curr['restocked']
        actual_change = curr['actual'] - prev['actual']
        discrepancy = actual_change - expected_change

        discrepancies.append(discrepancy)
        expected_changes.append(expected_change)

    discrepancies = np.array(discrepancies)

    # 用 warmup 期建立基线
    baseline = discrepancies[:warmup_days]
    mu = np.mean(baseline)
    sigma = np.std(baseline) + 1e-8

    # 检测异常
    z_scores = (discrepancies - mu) / sigma
    anomalies = np.abs(z_scores) > k

    # 累积差异趋势（CUSUM 风格）
    cumulative_discrepancy = np.cumsum(discrepancies)

    # 模式分析：按日期（模 7）聚合，找周期性规律
    day_of_week_disc = defaultdict(list)
    for i, disc in enumerate(discrepancies):
        dow = (inventory_records[i + 1]['date']) % 7
        day_of_week_disc[dow].append(disc)

    weekly_pattern = {
        dow: {'mean': np.mean(vals), 'std': np.std(vals)}
        for dow, vals in day_of_week_disc.items()
    }

    # 找最异常的星期几（均值最负 = 系统性损耗）
    worst_dow = min(weekly_pattern.keys(), key=lambda d: weekly_pattern[d]['mean'])

    return {
        'discrepancies': discrepancies,
        'z_scores': z_scores,
        'anomaly_days': np.where(anomalies)[0].tolist(),
        'total_discrepancy': float(cumulative_discrepancy[-1]),
        'weekly_pattern': weekly_pattern,
        'suspect_day_of_week': worst_dow,
        'alert': len(np.where(anomalies)[0]) > 3 or abs(cumulative_discrepancy[-1]) > 50
    }

# 测试：模拟系统性损耗（每周二/四各损失8件）
np.random.seed(42)
n = 60
records = []
inventory = 1000.0

for day in range(n):
    sales = float(np.random.poisson(15))
    shipped = float(np.random.poisson(5))
    restocked = float(np.random.poisson(20)) if day % 7 == 0 else 0

    # 每周二（2）和四（4）额外损失
    theft = 8.0 if day % 7 in [2, 4] else 0

    actual_inventory = inventory - sales - shipped + restocked - theft
    records.append({
        'date': day,
        'actual': actual_inventory,
        'sales': sales,
        'shipped': shipped,
        'restocked': restocked
    })
    inventory = actual_inventory

result = inventory_anomaly_detector(records, k=2.0)
print(f"累积库存差异: {result['total_discrepancy']:.0f} 件")
print(f"异常天数: {len(result['anomaly_days'])} 天")
print(f"最可疑星期: 星期{result['suspect_day_of_week'] + 1}")
print(f"⚠️ 告警: {'是' if result['alert'] else '否'}")
assert result['alert'], "应触发库存异常告警"
assert result['total_discrepancy'] < -100, "应累积明显负差异"
print("[✓] Inventory-Theft-Warehouse-Anomaly 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Transaction-Anomaly-Detection]]
- 前置技能：[[Skill-Time-Series-Anomaly-Detection]]
- 延伸技能：[[Skill-Supply-Chain-Counterfeit-Detection]]
- 延伸技能：[[Skill-Anomaly-Detection-Foundation-Model]]
- 可组合：[[Skill-SKU-Warehouse-Footprint-Monitoring]]
- 可组合：[[Skill-Warehouse-Inbound-Quality-Accuracy-KPI]]

## ⑤ 商业价值评估

- **ROI量化**: 年化发现并预防盗损 10-20 万元
- **实施难度**: ⭐⭐（需要接入 WMS 系统，数据整合成本中等）
- **优先级**: ⭐⭐⭐⭐（高价值品品类（奶粉/电器）的必备监控）
