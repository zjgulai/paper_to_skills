---
title: PID 安全库存控制器 — 将工业自动控制迁移到动态安全库存调整
doc_type: knowledge
module: 04-供应链
topic: pid-safety-stock-controller
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: PID 安全库存控制器

> **论文**：PID Control Applied to Inventory Management（Silver et al. 1998; Diehl & Jaeger, 2016, IJPDLM）
> **arXiv**：控制论迁移 | 1998/2016 | **桥梁**: 工业控制论 ↔ 安全库存管理 | **类型**: 跨域融合

## ① 算法原理

**火箭/工厂控制→安全库存调整的迁移逻辑**：

PID（比例-积分-微分）控制器是工业自动化的核心模块，原本用于**控制工厂温度/电机转速/飞行姿态**——目标是让系统状态稳定在设定值附近，当偏差出现时快速修正。

库存管理的本质也是一个**控制问题**：设定值 = 目标库存水平（通常是安全库存），当前状态 = 实际库存，误差 = 实际库存偏离目标的量，控制量 = 补货量/安全库存调整量。

**三项控制信号的业务含义**：

| 控制项 | 数学表达 | 工业含义 | 库存含义 |
|--------|---------|---------|---------|
| P（比例）| Kp × e(t) | 当前偏差 → 当前修正 | 当前库存偏离目标 → 当次补货调整 |
| I（积分）| Ki × ∫e(t)dt | 历史累积误差 → 消除稳态偏差 | 累积缺货次数 → 长期安全库存基线上调 |
| D（微分）| Kd × de(t)/dt | 误差变化速率 → 预防超调 | 需求加速度 → 提前应对需求突变 |

**安全库存 PID 公式**：

```
e(t) = 目标库存 - 当前库存
SS_adjustment(t) = Kp × e(t) + Ki × Σe(τ) + Kd × Δe(t)
SS_new = SS_base + SS_adjustment(t)
```

**关键优势**：传统安全库存 = z × σ_d × √(LT)，只用历史标准差，是静态固定值。PID控制器让安全库存随实时库存状态和需求变化**动态漂移**，在需求加速上升时提前增加安全库存，在连续缺货后系统性上调基线——这正是PID积分项消除"稳态误差"的用武之地。

**参数调优指引**（Ziegler-Nichols法则的库存版本）：
- Kp：设置为 1/（平均补货周期），避免过度反应
- Ki：设置为 Kp/（3×补货周期），积累足够历史才修正
- Kd：设置为 Kp × 0.5×补货周期，预测1-2步的需求加速

## ② 母婴出海应用案例

**场景A：婴儿安全座椅安全库存动态调整**

- **业务问题**：安全座椅Q3（返校季+节日备货季）需求剧增，固定安全系数z=1.65计算的安全库存在8-9月严重不足，缺货率高达15%；而Q1-Q2静默期过度备货占用资金。PID控制器可以在7月检测到需求加速时，自动将安全库存上调30-40%，并在Q4后缓慢归位。
- **数据要求**：日库存水平（FBA快照）、日销量、LT（Fulfillment Lead Time），历史缺货事件记录
- **预期产出**：
  - 每周动态安全库存推荐值
  - 三项控制信号分解（P/I/D各自贡献量）
  - 缺货率从15%降至5%以下
- **业务价值**：单品类年缺货损失¥50万，缺货率从15%→5%可挽回**¥33万/年**；同时Q1-Q2安全库存降低20%，减少资金占用¥10万

**场景B：纸尿裤旗舰款应对促销后需求反弹**

- **业务问题**：Prime Day后消费者购买透支，促销结束后2-3周需求低谷，卖家补货时按正常需求计划，导致低谷期积压。PID的D项（微分）检测到需求快速下降趋势，自动暂停安全库存上调，等需求回升再恢复。
- **数据要求**：日销量时序 + 促销期标记
- **预期产出**：促销前后的安全库存调整轨迹，积压减少量化
- **业务价值**：减少大促后积压¥15-25万/次大促

## ③ 代码模板

```python
import numpy as np
from collections import deque

class PIDSafetyStockController:
    """
    将 PID 控制论迁移到安全库存动态调整
    P: 当前库存偏差
    I: 历史累积缺货（稳态误差修正）
    D: 需求变化速率（预防超调）
    
    纯 numpy 实现，不依赖任何控制论库
    """
    
    def __init__(self, 
                 target_stock_days: float = 30.0,  # 目标库存覆盖天数
                 avg_daily_demand: float = 50.0,   # 初始需求估计
                 lead_time_days: int = 14,          # 补货前置期（天）
                 Kp: float = 0.15,                  # 比例增益
                 Ki: float = 0.03,                  # 积分增益
                 Kd: float = 2.0,                   # 微分增益
                 integral_window: int = 30,         # 积分历史窗口（天）
                 ss_min_days: float = 5.0,          # 安全库存最小值（天）
                 ss_max_days: float = 60.0):        # 安全库存最大值（天）
        
        self.target_stock_days = target_stock_days
        self.avg_demand = avg_daily_demand
        self.LT = lead_time_days
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.ss_min = ss_min_days * avg_daily_demand
        self.ss_max = ss_max_days * avg_daily_demand
        
        # 基础安全库存（传统公式：z × σ × √LT）
        self.ss_base = 1.65 * (avg_daily_demand * 0.3) * np.sqrt(lead_time_days)
        
        # 状态记录
        self.error_history = deque(maxlen=integral_window)
        self.prev_error = 0.0
        self.integral = 0.0
        
        self.history = []
    
    def compute_error(self, current_stock: float, current_demand: float) -> float:
        """误差 = 目标库存（天×日需求）- 当前实际库存"""
        target_abs = self.target_stock_days * current_demand
        return target_abs - current_stock
    
    def update(self, current_stock: float, current_demand: float, 
               dt: float = 1.0) -> dict:
        """
        单步PID更新
        current_stock: 当前FBA库存（件）
        current_demand: 当日销量（件）
        dt: 时间步长（默认1天）
        """
        # 计算误差
        error = self.compute_error(current_stock, current_demand)
        
        # === P 项：比例控制 ===
        P_term = self.Kp * error
        
        # === I 项：积分控制（消除稳态缺货偏差）===
        self.error_history.append(error)
        self.integral = sum(self.error_history) * dt
        I_term = self.Ki * self.integral
        
        # === D 项：微分控制（预测需求加速度）===
        derivative = (error - self.prev_error) / dt
        D_term = self.Kd * derivative
        self.prev_error = error
        
        # === PID 输出：安全库存调整量 ===
        ss_adjustment = P_term + I_term + D_term
        ss_new = np.clip(self.ss_base + ss_adjustment, self.ss_min, self.ss_max)
        
        # 更新内部状态
        self.ss_base = ss_new * 0.9 + self.ss_base * 0.1  # 平滑更新基础值
        
        result = {
            'current_stock': current_stock,
            'current_demand': current_demand,
            'error': error,
            'P_term': P_term,
            'I_term': I_term,
            'D_term': D_term,
            'ss_adjustment': ss_adjustment,
            'ss_recommended': ss_new,
            'ss_days': ss_new / max(current_demand, 1.0),  # 换算为覆盖天数
            'stock_warning': '⚠️ 缺货风险' if current_stock < ss_new else '✅ 库存健康',
        }
        self.history.append(result)
        return result
    
    def batch_simulate(self, stock_series: np.ndarray, 
                       demand_series: np.ndarray) -> dict:
        """批量历史模拟"""
        assert len(stock_series) == len(demand_series)
        n = len(stock_series)
        
        ss_recs = []
        errors = []
        P_terms, I_terms, D_terms = [], [], []
        
        for i in range(n):
            r = self.update(stock_series[i], demand_series[i])
            ss_recs.append(r['ss_recommended'])
            errors.append(r['error'])
            P_terms.append(r['P_term'])
            I_terms.append(r['I_term'])
            D_terms.append(r['D_term'])
        
        # 计算缺货率（当前库存 < 推荐安全库存）
        stockout_mask = stock_series < np.array(ss_recs)
        stockout_rate = np.mean(stockout_mask) * 100
        
        # 传统固定安全库存作为基准
        traditional_ss = 1.65 * np.std(demand_series) * np.sqrt(self.LT)
        traditional_stockout_rate = np.mean(stock_series < traditional_ss) * 100
        
        return {
            'ss_dynamic': np.array(ss_recs),
            'ss_traditional': traditional_ss,
            'errors': np.array(errors),
            'P_contribution': np.array(P_terms),
            'I_contribution': np.array(I_terms),
            'D_contribution': np.array(D_terms),
            'stockout_rate_pid': stockout_rate,
            'stockout_rate_traditional': traditional_stockout_rate,
            'improvement_pct': traditional_stockout_rate - stockout_rate,
        }


def pid_report(result: dict, n_days: int = 7) -> None:
    """输出PID控制分析报告"""
    print("=" * 55)
    print("PID 安全库存控制器报告")
    print("=" * 55)
    print(f"\n缺货率对比:")
    print(f"  传统固定安全库存: {result['stockout_rate_traditional']:.1f}%")
    print(f"  PID动态安全库存: {result['stockout_rate_pid']:.1f}%")
    print(f"  改善幅度: {result['improvement_pct']:+.1f}%")
    
    print(f"\n最近{n_days}天 PID 控制信号分解:")
    print(f"{'天':>4} {'P项':>8} {'I项':>8} {'D项':>8} {'SS推荐':>10}")
    for i in range(-n_days, 0):
        print(f"{i:>4} {result['P_contribution'][i]:>+8.1f} "
              f"{result['I_contribution'][i]:>+8.1f} "
              f"{result['D_contribution'][i]:>+8.1f} "
              f"{result['ss_dynamic'][i]:>10.1f}")


# ==================== 测试用例 ====================
if __name__ == '__main__':
    np.random.seed(2025)
    
    # 模拟婴儿安全座椅90天日销量和库存
    n_days = 90
    
    # 需求：Q3旺季（前30天基线50/天，中间30天旺季需求爬升到120/天，后30天回落到60/天）
    demand_base = np.concatenate([
        np.full(30, 50.0) + np.random.normal(0, 8, 30),
        np.linspace(50, 120, 30) + np.random.normal(0, 15, 30),
        np.linspace(120, 60, 30) + np.random.normal(0, 10, 30),
    ])
    demand = np.maximum(5, demand_base)
    
    # 库存：初始5000件，按固定传统补货策略消耗
    # （传统策略：ROP固定为需求均值×LT，补货量固定500件）
    LT = 14
    initial_stock = 5000.0
    stock = np.zeros(n_days)
    stock[0] = initial_stock
    pending_replenishment = []  # (到货天, 到货量)
    
    trad_ss = 1.65 * 20 * np.sqrt(LT)   # 传统固定安全库存
    
    for t in range(1, n_days):
        # 到货处理
        arrived = [qty for (day, qty) in pending_replenishment if day == t]
        stock[t] = max(0, stock[t-1] - demand[t-1]) + sum(arrived)
        pending_replenishment = [(d, q) for (d, q) in pending_replenishment if d > t]
        
        # 传统补货触发（固定ROP）
        if stock[t] < np.mean(demand[:t+1]) * LT + trad_ss:
            replen_qty = np.mean(demand[:t+1]) * (LT + 7)  # 补货至LT+7天覆盖
            pending_replenishment.append((t + LT, replen_qty))
    
    # PID 控制器模拟
    pid = PIDSafetyStockController(
        target_stock_days=20, 
        avg_daily_demand=np.mean(demand),
        lead_time_days=LT,
        Kp=0.10, Ki=0.02, Kd=3.0
    )
    batch_result = pid.batch_simulate(stock, demand)
    
    # 输出报告
    pid_report(batch_result, n_days=7)
    
    # 验证断言
    assert batch_result['stockout_rate_pid'] <= batch_result['stockout_rate_traditional'] + 5, \
        "PID缺货率不应远高于传统方法"
    
    # 验证PID各项控制信号非全零
    assert np.any(np.abs(batch_result['I_contribution']) > 0.1), "积分项应有非零贡献"
    assert np.any(np.abs(batch_result['D_contribution']) > 0.1), "微分项应有非零贡献"
    
    # 验证旺季期间安全库存自动上调
    q3_peak_ss = np.mean(batch_result['ss_dynamic'][40:60])     # 旺季
    q1_base_ss = np.mean(batch_result['ss_dynamic'][:20])       # 基线期
    print(f"\n旺季安全库存 vs 基线期: {q3_peak_ss:.0f} vs {q1_base_ss:.0f} 件")
    assert q3_peak_ss > q1_base_ss, "旺季期间PID应自动上调安全库存"
    
    print(f"\n[✓] PID 安全库存控制器 测试通过")
    print(f"    旺季SS自动上调 {(q3_peak_ss/q1_base_ss-1)*100:.0f}% | 缺货率改善 {batch_result['improvement_pct']:+.1f}%")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Dynamic-ABC-Stratification-Adaptive-Policy]]（ABC分级决定哪些SKU值得用PID控制，A类最优先）
- **前置（prerequisite）**：[[Skill-State-Space-Inventory-Signal-Smoothing]]（PID的输入误差信号应基于去季节化后的真实需求，而非原始噪声销量）
- **延伸（extends）**：[[Skill-Adaptive-Reorder-Point-Kalman]]（PID调整安全库存，Kalman调整补货点；二者协同构成完整的自适应库存控制系统）
- **可组合（combinable）**：[[Skill-Kalman-Filter-Demand-Tracking]]（Kalman提供需求真实状态估计 → 作为PID控制器的误差信号输入，消除噪声对PID的干扰）

## ⑤ 商业价值评估

- **ROI 预估**：以50个A类SKU、年销售额¥2000万为例，传统固定安全库存导致旺季缺货率8-12%（损失¥160-240万），PID动态控制缺货率压至3-4%，年化挽回损失**¥80-120万**；同时淡季安全库存降低25%，减少资金占用¥30万，合计**年化¥110-150万**。
- **实施难度**：⭐⭐☆☆☆（三个参数调优，有Ziegler-Nichols经验公式指导，无需复杂机器学习）
- **优先级**：⭐⭐⭐⭐⭐（高客单价A类SKU立竿见影，2周内可见缺货率变化，ROI清晰）
- **迁移风险**：低——PID的三项增益有明确物理含义，调参过程直观可控，出错时容易诊断
- **参数稳定性**：Kp/Ki/Kd每季度重新校准一次即可，不需要持续再训练
