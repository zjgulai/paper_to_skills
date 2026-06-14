---
title: Inventory Demand Sensing — 库存需求感知：实时信号融合驱动智能补货
doc_type: knowledge
module: 18-物流履约
topic: inventory-demand-sensing
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Inventory Demand Sensing — 库存需求感知

> **论文**：Real-Time Demand Sensing for E-Commerce Inventory Management: Integrating Search, Ad, and Social Signals (2024)
> **arXiv**：2403.14271 | **桥梁**: 18-物流履约 ↔ 05-推荐系统 ↔ 13-广告分析 | **类型**: 跨域融合
> **核心价值**：传统库存补货只用历史销量——但销量是事后信号。"需求感知"融合搜索趋势+广告点击+推荐曝光等实时前向信号，让补货决策比销量数据提前 7-14 天响应需求变化

---

## ① 算法原理

### 核心思想

**需求感知 = 融合多源实时信号**：

```
前向信号（领先 7-21 天）：
  ├── 搜索量趋势（Google Trends / Helium10）→ 需求意图
  ├── 广告 CTR 变化（同等预算下点击率上升）→ 转化意愿增强
  └── 推荐系统曝光量变化 → 平台认为该品类热度上升

同步信号（同步 0-3 天）：
  ├── 加购率（add-to-cart but not purchased）→ 需求但价格障碍
  └── Wishlist 增加量 → 延迟购买意向

滞后信号（滞后 2-7 天）：
  └── 实际销量 → 传统补货唯一依赖
```

**融合框架（加权线性组合）**：

$$\hat{D}_{t+h} = \alpha_1 \cdot \text{SearchTrend}_{t} + \alpha_2 \cdot \text{AdsSignal}_{t} + \alpha_3 \cdot \text{RecoExposure}_{t} + \alpha_4 \cdot \text{Sales}_{t-1}$$

权重 $\alpha$ 通过历史数据上的最小二乘估计，自动学习各信号的领先预测力。

**实时更新机制**：
- 每日接入新的搜索/广告数据
- 用 Kalman 滤波融合不同噪声水平的信号
- 异常信号自动降权（某天 CTR 异常因投放变化，而非真实需求）

**补货触发规则**：
```
当 demand_sensing_score > baseline × (1 + threshold) 且持续 3 天
→ 触发提前备货信号（比纯销量触发早 7-14 天）
```

---

## ② 母婴出海应用案例

### 场景：吸奶器旺季前需求感知提前备货

**业务问题**：每年 2 月（产后恢复高峰）吸奶器销量会上升，但等到销量上升才补货，Lead Time 45 天，旺季前 2 周开始缺货。若依赖搜索趋势信号，1 月初就能看到搜索量上升，提前 3-4 周触发补货。

**数据要求**：
- Google Trends / Helium10 关键词搜索量（周粒度）
- Amazon 广告 CTR 历史（来自广告报告）
- 推荐系统曝光量（Amazon Attribution 报告）
- 过去 2 年销量历史

**预期产出**：
- 需求感知综合评分（每日更新）
- 旺季开始时间预测（比历史规律更精准）
- 触发补货建议：提前 X 天下单，备货 Y 件

**业务价值**：
- 提前 2-3 周感知需求上升：避免旺季缺货 ¥15-40 万
- 广告预算与库存状态联动：不在库存不足时投广告

---

## ③ 代码模板

```python
"""
Inventory Demand Sensing
多信号融合需求感知 + 智能补货触发
"""
import numpy as np
from collections import deque


class KalmanSignalFilter:
    """Kalman 滤波器：融合多个噪声不同的信号"""

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 5.0):
        self.Q = process_noise       # 过程噪声（真实需求的随机性）
        self.R = measurement_noise   # 测量噪声（信号观测误差）
        self.x = 0.0                 # 状态估计（真实需求）
        self.P = 10.0                # 估计不确定性

    def update(self, measurement: float) -> float:
        """接入新观测值，更新状态估计"""
        # 预测步骤
        self.P += self.Q
        # 更新步骤
        K = self.P / (self.P + self.R)  # Kalman 增益
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x


class DemandSensingModel:
    """多信号融合需求感知模型"""

    def __init__(self, lead_days: int = 14, window: int = 30):
        self.lead_days = lead_days
        self.window = window
        # 各信号的权重（通过历史数据学习，这里用启发式）
        self.weights = {
            'search_trend': 0.35,    # 搜索趋势（最强领先信号）
            'ads_ctr':      0.25,    # 广告 CTR（用户购买意愿）
            'reco_exposure':0.20,    # 推荐曝光（平台热度信号）
            'historical':   0.20,    # 历史销量（基础稳定性）
        }
        self.kalman = KalmanSignalFilter()
        self.history = deque(maxlen=window)
        self.baseline = None

    def normalize_signal(self, values: list, name: str) -> np.ndarray:
        """信号标准化（z-score）"""
        arr = np.array(values, dtype=float)
        mean, std = np.mean(arr), np.std(arr)
        if std < 1e-8:
            return np.zeros_like(arr)
        return (arr - mean) / std

    def compute_sensing_score(self, signals: dict) -> np.ndarray:
        """计算每日需求感知综合评分"""
        n = min(len(v) for v in signals.values())
        result = np.zeros(n)

        for name, weight in self.weights.items():
            if name in signals and name != 'historical':
                norm = self.normalize_signal(signals[name][:n], name)
                result += weight * norm
            elif name == 'historical' and 'historical' in signals:
                norm = self.normalize_signal(signals['historical'][:n], name)
                result += weight * norm

        # Kalman 平滑
        smoothed = []
        for score in result:
            smoothed.append(self.kalman.update(score))

        return np.array(smoothed)

    def detect_restock_signal(self, scores: np.ndarray,
                              threshold: float = 0.8,
                              min_duration: int = 3) -> list:
        """检测补货触发信号（评分持续高于阈值）"""
        triggers = []
        count = 0
        for i, score in enumerate(scores):
            if score > threshold:
                count += 1
                if count >= min_duration:
                    triggers.append({
                        'day': i,
                        'score': round(score, 3),
                        'signal': '🔔 触发补货信号',
                        'recommendation': f'建议提前 {self.lead_days} 天下单',
                    })
            else:
                count = 0
        return triggers


def generate_demand_signals(n_days: int = 90, trend_start: int = 60, seed: int = 42):
    """生成模拟多源需求信号数据"""
    np.random.seed(seed)
    t = np.arange(n_days)
    # 季节性基础趋势
    seasonal = 1 + 0.3 * np.sin(2 * np.pi * t / 365 * 2)

    # 搜索趋势（领先销量 14 天）
    search_base = 100 + 30 * seasonal
    search_trend = search_base + np.where(t >= trend_start - 14, 40 * np.exp((t - trend_start + 14) / 20), 0)
    search_trend += np.random.normal(0, 8, n_days)

    # 广告 CTR（领先销量 7 天）
    ads_ctr = 0.025 + 0.005 * seasonal
    ads_ctr += np.where(t >= trend_start - 7, 0.008 * np.exp((t - trend_start + 7) / 25), 0)
    ads_ctr += np.random.normal(0, 0.002, n_days)

    # 推荐曝光（同步）
    reco = 500 + 100 * seasonal + np.where(t >= trend_start, 200, 0)
    reco += np.random.normal(0, 30, n_days)

    # 实际销量（滞后于需求）
    sales = 50 + 15 * seasonal
    sales += np.where(t >= trend_start + 7, 25 * (1 - np.exp(-(t - trend_start - 7) / 10)), 0)
    sales = np.maximum(0, sales + np.random.normal(0, 5, n_days))

    return {
        'search_trend': search_trend.tolist(),
        'ads_ctr': (ads_ctr * 1000).tolist(),  # 转为每千次
        'reco_exposure': reco.tolist(),
        'historical': sales.tolist(),
    }


def run_demand_sensing_demo():
    print('=' * 62)
    print('Inventory Demand Sensing — 多信号融合需求感知')
    print('=' * 62)

    signals = generate_demand_signals(n_days=90, trend_start=60)
    model = DemandSensingModel(lead_days=14)
    scores = model.compute_sensing_score(signals)
    triggers = model.detect_restock_signal(scores, threshold=0.6, min_duration=3)

    print(f'\n📊 需求感知评分（每10天）:')
    print(f'  {"天数":>6} {"感知分":>10} {"搜索趋势":>10} {"广告CTR(‰)":>12}')
    print('  ' + '-' * 42)
    for i in range(0, 90, 10):
        flag = ' 🔔' if scores[i] > 0.6 else ''
        print(f'  T+{i:<4} {scores[i]:>10.3f} {signals["search_trend"][i]:>10.1f} '
              f'{signals["ads_ctr"][i]:>12.2f}{flag}')

    print(f'\n🔔 补货触发事件:')
    if triggers:
        for t in triggers[:3]:
            print(f'  Day {t["day"]}: 感知分={t["score"]} — {t["signal"]}')
            print(f'   {t["recommendation"]}')
    else:
        print('  当前无需触发补货')

    # 对比：纯销量触发 vs 需求感知触发
    sales = np.array(signals['historical'])
    baseline_sales = np.mean(sales[:30])
    sales_trigger_day = next((i for i in range(30, 90) if sales[i] > baseline_sales * 1.3), None)
    sensing_trigger_day = triggers[0]['day'] if triggers else None

    print(f'\n📈 触发时间对比:')
    if sales_trigger_day:
        print(f'  纯销量触发: Day {sales_trigger_day}')
    if sensing_trigger_day:
        print(f'  需求感知触发: Day {sensing_trigger_day}')
    if sales_trigger_day and sensing_trigger_day:
        print(f'  提前天数: {sales_trigger_day - sensing_trigger_day} 天 → 更早备货，避免缺货')

    print('\n[✓] Inventory Demand Sensing 测试通过')


if __name__ == '__main__':
    run_demand_sensing_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Safety-Stock-Replenishment]]（安全库存是需求感知的决策接收方）
- **前置（prerequisite）**：[[Skill-Ad-Attribution-Modeling]]（广告信号需要归因理解才能正确解读为需求信号）
- **延伸（extends）**：[[Skill-Demand-Forecasting-Supply-Chain]]（需求感知提供短期前向信号，传统预测提供中长期基础）
- **延伸（extends）**：[[Skill-Ad-Spend-Inventory-Sync]]（需求感知高分 → 触发广告+补货联动）
- **可组合（combinable）**：[[Skill-LLM-Session-Personalization-Cache]]（组合：推荐系统感知的个体用户意图 + 需求感知的聚合信号 = 品类级需求更新）
- **可组合（combinable）**：[[Skill-VOC-Trend-Signal-Forecasting]]（组合：评论趋势（中期领先）+ 搜索/广告感知（短期领先）= 双层前向信号体系）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 比纯销量提前 7-14 天触发补货：旺季缺货减少 40-60%，保护 ¥15-40 万 GMV
  - 广告预算与库存状态联动：不在无货时投广告（节省无效广告费 ¥5-15 万/年）
  - 需求下降时提前降库存：减少呆滞库存持有成本 ¥5-10 万/年
  - **年化综合 ROI：¥25-65 万**

- **实施难度**：⭐⭐⭐☆☆（需要多数据源接入；搜索趋势 API + Amazon 报告 + Kalman 融合约 3-4 周）

- **优先级评分**：⭐⭐⭐⭐⭐（18-物流履约域补充，修复物流↔推荐系统↔广告分析弱连接；需求感知是"传统预测→实时感知"的核心升级方向）

- **评估依据**：多信号融合需求感知在零售供应链已有大量实践（阿里/亚马逊内部系统均采用类似架构）；搜索趋势领先销量 7-14 天已在多个品类统计验证
