---
title: Intermittent Demand Croston TSB — 母婴长尾 SKU 间歇需求预测
doc_type: knowledge
module: 03-时间序列
topic: intermittent-demand-croston-tsb
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Intermittent-Demand-Croston-TSB

## ① 算法原理（≤300字）

**核心问题**：母婴长尾 SKU（如特定规格吸奶管配件）销量稀疏——大量时期销量为 0，偶发需求时量级波动大。ARIMA/Prophet 在零值序列上失效，而 Croston 方法专为间歇需求设计。

**Croston 方法**：将需求序列拆为两个独立过程：
- **需求量序列** $z$：仅包含非零需求时刻的量值
- **需求间隔序列** $q$：连续两次非零需求之间的间隔期数

两者分别用指数平滑更新：
$$\hat{z}_{t+1} = \alpha z_t + (1-\alpha)\hat{z}_t$$
$$\hat{q}_{t+1} = \alpha q_t + (1-\alpha)\hat{q}_t$$

预测值：$\hat{y} = \hat{z} / \hat{q}$

**TSB（Teunter-Syntetos-Babai）改进**：Croston 存在偏差，TSB 对需求间隔用发生概率 $p$ 替代直接建模：
$$\hat{p}_{t+1} = \alpha \cdot \mathbb{1}[y_t > 0] + (1-\alpha)\hat{p}_t$$
$$\hat{y} = \hat{p} \times \hat{z}$$

TSB 在需求模式转变时收敛更快（如 SKU 停售后预测快速归零）。

**关键假设**：需求间隔和需求量相互独立；指数平滑假设平稳性（无趋势/季节）。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某卖家有 400+ 个配件类 SKU（奶瓶密封圈、吸奶管接头），其中 70% 月均销量 < 5 件，传统补货公式长期缺货或过度库存。

**数据要求**：每 SKU 过去 24 个月日/周销量历史，前置期 30-45 天。

**TSB 应用**：识别每个 SKU 的需求频率和批次量，差异化补货触发点。需求频率 < 20% 的 SKU 转为「按需生产」模式，高频 SKU 维持安全库存。

**量化产出**：长尾 SKU 库存周转天数从 180 天压缩至 90 天，资金占用降低 50%，年化释放库存资金约 **20-30 万元**。

## ③ 代码模板

```python
import numpy as np

def croston_tsb(demand: np.ndarray, alpha: float = 0.2) -> dict:
    """
    TSB 间歇需求预测
    demand: 含零值的需求序列（如月度销量）
    alpha: 平滑系数
    返回: 预测值、需求概率、非零均值
    """
    n = len(demand)
    z = np.mean(demand[demand > 0]) if (demand > 0).any() else 1.0  # 非零需求均值
    p = (demand > 0).mean()  # 需求概率初始值

    z_hat = z
    p_hat = p
    forecasts = []

    for i in range(n):
        forecasts.append(p_hat * z_hat)
        if demand[i] > 0:
            z_hat = alpha * demand[i] + (1 - alpha) * z_hat
            p_hat = alpha * 1.0 + (1 - alpha) * p_hat
        else:
            p_hat = alpha * 0.0 + (1 - alpha) * p_hat

    next_forecast = p_hat * z_hat
    return {
        'forecasts': np.array(forecasts),
        'next_forecast': next_forecast,
        'demand_prob': p_hat,
        'demand_size': z_hat
    }

# 测试：模拟间歇需求序列
np.random.seed(42)
demand = np.array([0,0,3,0,0,0,5,0,2,0,0,4,0,0,0,6,0,0,3,0,0,0,0,2])
result = croston_tsb(demand, alpha=0.2)

assert len(result['forecasts']) == len(demand)
assert 0 < result['next_forecast'] < 10
assert 0 < result['demand_prob'] < 1
print(f"下期预测需求: {result['next_forecast']:.2f} 件/期")
print(f"需求发生概率: {result['demand_prob']:.1%}")
print(f"非零需求均值: {result['demand_size']:.2f} 件")
print("[✓] Intermittent-Demand-Croston-TSB 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Demand-Forecasting-Supply-Chain]]
- 前置技能：[[Skill-Time-Series-Forecasting]]
- 延伸技能：[[Skill-Safety-Stock-Replenishment]]
- 延伸技能：[[Skill-Long-Tail-SKU-Clearance-Optimization]]
- 可组合：[[Skill-Dynamic-Lot-Sizing-MOQ]]
- 可组合：[[Skill-Forecast-Driven-Inventory]]

## ⑤ 商业价值评估

- **ROI量化**: 长尾 SKU 库存资金占用降低 40-50%，年化释放 20-30 万元
- **实施难度**: ⭐⭐（纯 Python 无外部依赖，1 天可集成）
- **优先级**: ⭐⭐⭐⭐（400+ SKU 卖家立竿见影）
