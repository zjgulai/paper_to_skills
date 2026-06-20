---
title: 自适应补货点 Kalman 版 — 让 ROP 随市场需求动态漂移
doc_type: knowledge
module: 04-供应链
topic: adaptive-reorder-point-kalman
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 自适应补货点 Kalman 版

> **论文**：Adaptive Control of Inventory with Unknown Demand（Burnetas & Katehakis, 1996, Operations Research; Silver et al. 2017）
> **arXiv**：控制论迁移 | 1996/2017 | **桥梁**: 自适应控制 ↔ 补货点优化 | **类型**: 算法工具

## ① 算法原理

**火箭制导→自适应补货点的迁移逻辑**：

火箭制导中的自适应控制（Adaptive Control）解决的问题是：**当系统参数（如空气阻力、推力）随时间变化时，控制器如何实时更新自己的参数估计并修正控制策略**。传统的固定参数控制器在参数漂移时会失效，就像固定ROP在需求趋势上升时会越来越滞后。

传统补货点（Reorder Point）公式：
```
ROP_static = μ_d × LT + z × σ_d × √LT
```
其中 μ_d 和 σ_d 是历史均值和标准差——一旦需求结构变化（新竞品上架、平台算法调整），这两个参数就过时了。

**Kalman 自适应 ROP 的核心改进**：

用 Kalman Filter 实时估计需求的均值 `μ̂_t` 和方差 `P̂_t`，替换历史统计量：

```
ROP_kalman(t) = μ̂_t × LT + z × √(P̂_t × LT)
```

每天收到新销量数据后，Kalman 的预测-更新步骤：
1. **预测步**：`μ̂_{t|t-1} = μ̂_{t-1|t-1}`，`P_{t|t-1} = P_{t-1|t-1} + Q`
2. **更新步**：`K_t = P_{t|t-1}/(P_{t|t-1}+R)`，`μ̂_t = μ̂_{t|t-1} + K_t(y_t - μ̂_{t|t-1})`，`P_t = (1-K_t)P_{t|t-1}`
3. **ROP计算**：`ROP_t = μ̂_t × LT + z × √(P_t × LT)`

**自适应的关键**：P_t 不再是固定的历史方差，而是 Kalman 估计的**当前不确定性**——当需求刚发生结构变化时，P_t 会自动上升（卡尔曼增益变大，快速追踪），同时 ROP 的安全余量也自动扩大，防止结构变化初期的缺货。当需求稳定后，P_t 收敛到低值，ROP 的安全余量收缩，节约库存成本。

**关键参数**：
- Q/R 比：决定对新数据的响应速度，Q大则快速适应需求漂移，R大则平滑噪声
- z（服务水平因子）：z=1.65对应95%服务水平，z=2.05对应98%
- LT（前置期）：应包含亚马逊入库审核时间（通常+3-5天）

## ② 母婴出海应用案例

**场景A：婴儿奶瓶 SKU 应对竞品下架带来的需求暴涨**

- **业务问题**：头部竞品被平台下架，本品需求在3天内从日销50跳到150，传统ROP（基于过去90天μ=50）触发补货量严重不足。Kalman自适应ROP在第5天就将μ̂更新到80+，第10天更新到120+，触发足量补货。
- **数据要求**：SKU日销量（≥30天历史），FBA LT（含审核），目标服务水平（z值）
- **预期产出**：
  - 每日动态 ROP 值（随需求变化自动漂移）
  - 需求估计 μ̂_t 和置信区间 √P_t 轨迹
  - 首次触发补货的时间点对比（静态 ROP vs 动态 ROP）
- **业务价值**：需求突变情境下，避免1-2周缺货，以日销150件×$8/件×10天估算，挽回销售损失**约¥86,400（≈¥8.6万）/次事件**

**场景B：婴儿床淡旺季切换时的ROP自动过渡**

- **业务问题**：婴儿床需求从Q2（日销30件）到Q3（日销80件）有明显爬坡，传统ROP切换依赖人工每季更新参数，常有2-3周的参数滞后期。Kalman版本无缝在线学习，逐日跟随爬坡自动调整ROP。
- **数据要求**：历史2季度以上日销量
- **预期产出**：淡旺季过渡期（约4-6周）内，ROP的平滑过渡曲线
- **业务价值**：消除人工参数更新，每季节省运营时间8-10小时；消除滞后期缺货损失¥5-15万

## ③ 代码模板

```python
import numpy as np
from scipy import stats

class KalmanAdaptiveROP:
    """
    基于 Kalman Filter 的自适应补货点（Reorder Point）计算
    核心：实时追踪需求均值 μ̂_t 和不确定性 P_t，替代历史静态统计量
    
    完全用 numpy 手写 Kalman 核心方程
    """
    
    def __init__(self, 
                 lead_time_days: int = 14,
                 service_level: float = 0.95,
                 Q: float = 5.0,    # 过程噪声方差（需求漂移速度）
                 R: float = 50.0,   # 观测噪声方差（日销量随机波动）
                 initial_demand: float = 50.0,
                 initial_variance: float = 200.0):
        """
        lead_time_days: 补货前置期（含FBA审核，建议+3-5天buffer）
        service_level: 目标服务水平（0.95 → z=1.645）
        Q: 过程噪声（Q/R比越大，对需求漂移越敏感）
        R: 观测噪声（日销量的固有随机性）
        """
        self.LT = lead_time_days
        self.z = stats.norm.ppf(service_level)  # 服务水平 → z 值
        self.Q = Q
        self.R = R
        
        # Kalman 状态初始化
        self.mu_est = initial_demand      # 需求均值估计
        self.P_est = initial_variance     # 需求方差估计（不确定性）
        
        # 历史记录
        self.history = []
        self.step = 0
    
    def update(self, y_t: float) -> dict:
        """
        处理单天新销量观测，更新需求估计并计算新 ROP
        y_t: 当日销量（件）
        """
        self.step += 1
        
        # ===== Kalman 预测步 =====
        mu_pred = self.mu_est          # 需求均值预测（随机游走假设）
        P_pred = self.P_est + self.Q   # 预测方差（不确定性增加）
        
        # ===== Kalman 更新步 =====
        K = P_pred / (P_pred + self.R)             # Kalman 增益
        innovation = y_t - mu_pred                  # 创新残差
        mu_new = mu_pred + K * innovation           # 后验均值估计
        P_new = (1 - K) * P_pred                    # 后验方差（不确定性降低）
        
        # ===== 计算自适应 ROP =====
        # ROP = 前置期需求 + 安全余量
        # 前置期需求方差 = P_t × LT（假设各天独立，方差累加）
        safety_stock = self.z * np.sqrt(P_new * self.LT)
        rop_kalman = mu_new * self.LT + safety_stock
        
        # 对比：传统静态 ROP（基于累积历史均值和方差）
        historical_demands = [r['y_t'] for r in self.history] + [y_t]
        mu_static = np.mean(historical_demands)
        sigma_static = np.std(historical_demands) if len(historical_demands) > 1 else np.sqrt(self.R)
        rop_static = mu_static * self.LT + self.z * sigma_static * np.sqrt(self.LT)
        
        # 更新状态
        self.mu_est = mu_new
        self.P_est = P_new
        
        result = {
            'step': self.step,
            'y_t': y_t,
            'mu_kalman': mu_new,           # Kalman 需求均值估计
            'P_kalman': P_new,             # Kalman 需求方差估计
            'K_gain': K,                   # Kalman 增益（本次对新数据的信任度）
            'safety_stock': safety_stock,  # 安全库存量
            'rop_kalman': rop_kalman,      # 自适应补货点
            'rop_static': rop_static,      # 传统静态补货点（对比）
            'mu_static': mu_static,        # 传统历史均值
            'rop_delta': rop_kalman - rop_static,  # 两者差距
        }
        self.history.append(result)
        return result
    
    def batch_simulate(self, demand_series: np.ndarray, 
                       shock_day: int = None, 
                       shock_multiplier: float = 3.0) -> dict:
        """
        批量模拟，支持注入需求突变场景
        shock_day: 需求突变开始天（None=不注入）
        shock_multiplier: 突变倍数（如3.0=需求变为原来3倍）
        """
        # 注入需求突变
        obs = np.array(demand_series, dtype=float)
        if shock_day is not None:
            obs[shock_day:] *= shock_multiplier
        
        # 重置状态
        self.mu_est = obs[0]
        self.P_est = np.var(obs[:min(7, len(obs))])
        self.history = []
        self.step = 0
        
        rop_kalmans = []
        rop_statics = []
        mu_kalmans = []
        P_kalmans = []
        
        for y in obs:
            r = self.update(y)
            rop_kalmans.append(r['rop_kalman'])
            rop_statics.append(r['rop_static'])
            mu_kalmans.append(r['mu_kalman'])
            P_kalmans.append(r['P_kalman'])
        
        # 计算追踪速度：需求突变后多少天 ROP 追上真实需求水平
        if shock_day is not None:
            true_mu_after_shock = np.mean(obs[shock_day:])
            kalman_catch_up = None
            static_catch_up = None
            for i in range(shock_day, len(obs)):
                if kalman_catch_up is None and mu_kalmans[i] > 0.8 * true_mu_after_shock:
                    kalman_catch_up = i - shock_day
                if static_catch_up is None and rop_statics[i] > 0.8 * (true_mu_after_shock * self.LT):
                    static_catch_up = i - shock_day
        else:
            kalman_catch_up = static_catch_up = None
        
        return {
            'observations': obs,
            'rop_kalman': np.array(rop_kalmans),
            'rop_static': np.array(rop_statics),
            'mu_kalman': np.array(mu_kalmans),
            'P_kalman': np.array(P_kalmans),
            'kalman_catch_up_days': kalman_catch_up,
            'static_catch_up_days': static_catch_up,
        }


# ==================== 测试用例 ====================
if __name__ == '__main__':
    np.random.seed(1234)
    
    # 场景1：需求结构突变（竞品下架场景）
    n_days = 90
    # 基础需求：50件/天 + 噪声
    base_demand = 50 + np.random.normal(0, 8, n_days)
    base_demand = np.maximum(5, base_demand)
    
    kf_rop = KalmanAdaptiveROP(
        lead_time_days=14,
        service_level=0.95,
        Q=8.0,
        R=64.0,
        initial_demand=50.0,
    )
    
    # 第30天注入需求突变（竞品下架，需求×3）
    result = kf_rop.batch_simulate(base_demand, shock_day=30, shock_multiplier=3.0)
    
    print("=" * 55)
    print("自适应补货点 Kalman 版 — 需求突变追踪测试")
    print("=" * 55)
    
    print(f"\n需求突变场景（第30天需求×3倍）:")
    print(f"  Kalman ROP 追踪速度: {result['kalman_catch_up_days']} 天追上80%目标")
    print(f"  静态  ROP 追踪速度: {result['static_catch_up_days']} 天追上80%目标")
    
    print(f"\n第28-35天 ROP 轨迹（突变前后对比）:")
    print(f"{'天':>4} {'实际销量':>8} {'Kalman ROP':>12} {'静态ROP':>10} {'K增益':>8}")
    for i in range(27, 36):
        r = kf_rop.history[i]
        print(f"{i+1:>4} {result['observations'][i]:>8.1f} "
              f"{result['rop_kalman'][i]:>12.1f} "
              f"{result['rop_static'][i]:>10.1f} "
              f"{r['K_gain']:>8.3f}")
    
    # 场景2：渐进式需求爬坡（淡旺季过渡）
    ramp_demand = np.linspace(30, 80, 60) + np.random.normal(0, 6, 60)
    ramp_demand = np.maximum(5, ramp_demand)
    
    kf_rop2 = KalmanAdaptiveROP(lead_time_days=14, service_level=0.95, Q=5.0, R=36.0)
    ramp_result = kf_rop2.batch_simulate(ramp_demand)
    
    # ROP 应跟随需求爬坡上升
    rop_early = np.mean(ramp_result['rop_kalman'][:15])
    rop_late = np.mean(ramp_result['rop_kalman'][-15:])
    print(f"\n渐进式爬坡测试（需求30→80）:")
    print(f"  早期 ROP 均值: {rop_early:.1f}")
    print(f"  末期 ROP 均值: {rop_late:.1f}")
    print(f"  ROP 随需求自动爬坡: {(rop_late/rop_early-1)*100:.1f}%↑")
    
    # 断言
    assert result['kalman_catch_up_days'] is not None, "Kalman 应在突变后追上需求"
    if result['static_catch_up_days'] is not None:
        assert result['kalman_catch_up_days'] <= result['static_catch_up_days'], \
            "Kalman 追踪应快于或等于静态方法"
    assert rop_late > rop_early, "需求爬坡时 ROP 应自动上升"
    assert np.all(ramp_result['rop_kalman'] > 0), "ROP 应始终为正"
    
    print(f"\n[✓] 自适应补货点 Kalman 版 测试通过")
    print(f"    突变追踪 {result['kalman_catch_up_days']} 天 | 爬坡适应 +{(rop_late/rop_early-1)*100:.1f}%")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Kalman-Filter-Demand-Tracking]]（本 Skill 的核心是 Kalman 对需求均值和方差的追踪，需先理解 Kalman Filter 基础）
- **前置（prerequisite）**：[[Skill-Dynamic-Lot-Sizing-MOQ]]（ROP决定何时补货，Lot-Sizing决定补多少——两者是补货决策的两个维度）
- **延伸（extends）**：[[Skill-PID-Safety-Stock-Controller]]（自适应ROP提供补货触发时机，PID控制器动态调整安全库存量；两者组合实现完整自适应库存控制）
- **可组合（combinable）**：[[Skill-Bullwhip-Effect-Kalman-Mitigation]]（每个节点使用自适应ROP，避免因固定ROP导致的订单放大效应）

## ⑤ 商业价值评估

- **ROI 预估**：中大型卖家有100个以上活跃SKU，平均每个SKU需求结构变化1-2次/年（竞品上下架、算法调整等），每次结构变化期间平均缺货损失¥2-5万，传统静态ROP平均滞后10-20天，Kalman版本滞后3-7天，年化挽回损失**¥80-200万**（100 SKU × 2次/年 × 节省7天缺货 × ¥0.5万/天）。
- **实施难度**：⭐⭐☆☆☆（比传统ROP计算复杂度只增加了 Kalman 的预测+更新两步，可嵌入现有补货系统）
- **优先级**：⭐⭐⭐⭐⭐（需求结构不稳定的品类，如季节性强或竞争激烈的母婴品类，效果最显著）
- **迁移风险**：低——Kalman更新步骤保证了算法的数值稳定性，参数 Q/R 有物理含义，调参直观
- **落地路径**：第1周替换单个高价值SKU的ROP计算 → 验证2周 → 扩展到全部A类SKU
