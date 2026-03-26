# Skill Card: Multi-Echelon Inventory Optimization (多阶库存优化)

---

## ① 算法原理

### 核心思想
多阶库存优化解决的核心问题是：**如何在供应链的多个节点（工厂→仓库→配送中心→门店）之间分配库存，使得总成本最低的同时保证服务水平**。与单点库存管理不同，多阶优化需要考虑节点间的依赖关系、订货提前期和需求传递效应。

### 数学直觉

**报童模型 (Newsvendor Model)** - 单产品单周期：
$$Q^* = F^{-1}\left(\frac{p - c}{p}\right) = F^{-1}\left(\frac{c_u}{c_u + c_o}\right)$$

其中：
- $Q^*$ 是最优订购量
- $c_o$ 是缺货成本（lost profit per unit）
- $c_u$ 是未售出成本（holding cost per unit）
- $F$ 是需求分布的累积函数

**(s, S) 策略** - 连续检查：
- 当库存降到 s 时，订货到 S 水平
- 订货量 = S - 当前库存
- s = 安全库存，取决于服务水平

**安全库存计算**：
$$SS = z \times \sigma_L = z \times \sqrt{\sum_{i}(L_i \times \sigma_{D_i}^2)}$$

其中 $L_i$ 是第 i 阶段的提前期，$\sigma_{D_i}$ 是需求标准差。

### 关键假设
- **需求独立同分布**：各节点需求相互独立，分布已知
- **补货周期固定**：提前期已知（可设为随机变量）
- **无限产能**：供应商产能充足
- **服务水平约束**：需满足预设的订单履约率

---

## ② 母婴出海应用案例

### 场景一：海外仓备货策略优化

**业务问题**：
母婴出海电商在海外建立仓储物流体系，通常包含国内工厂 → 国内仓库 → 海外仓 → 消费者。由于跨境物流周期长（15-30天）、需求波动大，库存过多会导致仓储成本高、资金占用大，库存过少会导致缺货、丢失销售机会。需要科学计算各节点的合理库存量。

**数据要求**：
- 历史销量数据：SKU 级别日/周销量（建议 2 年）
- 物流参数：各段运输时长（工厂→国内仓、国内仓→海外仓）
- 成本参数：单位仓储成本、单位缺货成本、订货固定成本
- 服务水平目标：订单履约率 95%+

**预期产出**：
- 各节点安全库存建议（国内仓、海外仓）
- 最佳补货触发点（reorder point）
- 补货量计算公式
- 库存周转率预期

**业务价值**：
- 海外仓库存成本降低 20-30%（假设月仓储成本 30 万，可节省 6-9 万）
- 缺货率降低 50%+（从 10% 降至 5% 以下）
- 资金周转提升 15-25%

---

### 场景二：爆款SKU动态备货

**业务问题**：
母婴出海商品存在明显的季节性（奶粉、尿裤大促季）和趋势性（新款婴儿推车上市）。传统的固定安全库存策略无法适应需求变化，需要根据销售趋势动态调整库存。

**数据要求**：
- 实时销量数据：近 30 天滚动销量
- 趋势指标：销量增长率、季节指数
- 物流参数：当前库存、在途订单、预计到货时间
- 竞品数据（可选）：竞品价格、活动力度

**预期产出**：
- 动态安全库存建议（随趋势调整）
- 预警清单：哪些 SKU 需要补货、哪些需要清仓
- 补货优先级排序

**业务价值**：
- 爆款缺货率降低 60%+
- 滞销品库存清理提前 2-4 周
- 整体库存周转提升 20%

---

## ③ 代码模板

```python
"""
Multi-Echelon Inventory Optimization
用于母婴出海电商海外仓备货策略优化
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


class InventoryOptimizer:
    """多阶库存优化器"""

    def __init__(self, service_level=0.95):
        """
        初始化库存优化器

        Args:
            service_level: 目标服务水平 (0-1)
        """
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)

    def calculate_safety_stock(self, demand_std, lead_time):
        """
        计算安全库存

        Args:
            demand_std: 需求标准差
            lead_time: 提前期 (天)

        Returns:
            safety_stock: 安全库存量
        """
        # 简化模型：假设需求独立，SS = z * sigma * sqrt(L)
        demand_daily_std = demand_std / np.sqrt(30)  # 假设月std转日std
        safety_stock = self.z_score * demand_daily_std * np.sqrt(lead_time)
        return int(np.ceil(safety_stock))

    def calculate_reorder_point(self, avg_daily_demand, lead_time, safety_stock):
        """
        计算再订货点 (ROP)

        Args:
            avg_daily_demand: 平均日需求量
            lead_time: 提前期 (天)
            safety_stock: 安全库存

        Returns:
            reorder_point: 再订货点
        """
        return avg_daily_demand * lead_time + safety_stock

    def newsvendor_optimal_order(self, selling_price, cost, salvage_value, demand_mean, demand_std):
        """
        报童模型最优订购量

        Args:
            selling_price: 售价
            cost: 进价/生产成本
            salvage_value: 残值/清仓价
            demand_mean: 需求均值
            demand_std: 需求标准差

        Returns:
            optimal_order: 最优订购量
        """
        # 缺货成本 = 销售利润
        cost_shortage = selling_price - cost
        # 滞销成本 = 进价 - 残值
        cost_holding = cost - salvage_value

        # 临界概率 = cu / (cu + co)
        critical_prob = cost_shortage / (cost_shortage + cost_holding)

        # 最优订购量 = F^{-1}(critical_prob)
        optimal_order = demand_mean + demand_std * stats.norm.ppf(critical_prob)
        return int(np.ceil(optimal_order))

    def multi_echelon_optimize(self, demand_mean, demand_std, lead_times, holding_cost, shortage_cost):
        """
        多阶库存优化

        Args:
            demand_mean: 需求均值
            demand_std: 需求标准差
            lead_times: 各阶段提前期列表 [stage1, stage2, ...]
            holding_cost: 单位持有成本/天
            shortage_cost: 单位缺货成本

        Returns:
            results: 各阶段库存优化结果
        """
        results = []

        for i, lead_time in enumerate(lead_times):
            # 计算该阶段的安全库存
            ss = self.calculate_safety_stock(demand_std, lead_time)

            # 计算再订货点
            rop = self.calculate_reorder_point(demand_mean, lead_time, ss)

            # 计算总成本
            total_holding = holding_cost * ss * 30  # 月持有成本
            expected_shortage = demand_std * self.z_score / np.sqrt(lead_time) if lead_time > 0 else 0
            total_shortage = shortage_cost * expected_shortage

            results.append({
                'stage': i + 1,
                'lead_time': lead_time,
                'safety_stock': ss,
                'reorder_point': rop,
                'monthly_holding_cost': total_holding,
                'expected_shortage': expected_shortage
            })

        return pd.DataFrame(results)

    def dynamic_safety_stock(self, recent_demand, lead_time, service_level=None):
        """
        动态安全库存（基于近期需求）

        Args:
            recent_demand: 近期需求数组
            lead_time: 提前期
            service_level: 目标服务水平

        Returns:
            dynamic_ss: 动态安全库存
            trend_factor: 趋势因子
        """
        if service_level is None:
            service_level = self.service_level

        # 计算滚动均值和标准差
        demand_mean = np.mean(recent_demand)
        demand_std = np.std(recent_demand, ddof=1)

        # 计算趋势因子（简单移动平均对比）
        if len(recent_demand) >= 14:
            ma_7 = np.mean(recent_demand[-7:])
            ma_14 = np.mean(recent_demand[-14:-7])
            trend_factor = ma_7 / ma_14 if ma_14 > 0 else 1.0
        else:
            trend_factor = 1.0

        # 调整后的安全库存
        ss = self.calculate_safety_stock(demand_std, lead_time)
        dynamic_ss = int(np.ceil(ss * trend_factor))

        return dynamic_ss, trend_factor


# ==================== 示例代码 ====================

def generate_sample_data():
    """生成模拟数据"""
    np.random.seed(42)

    # 模拟 2 年日销量数据
    n_days = 730
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

    # 基础需求 + 季节性 + 趋势
    base_demand = 50
    seasonality = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    trend = 1 + 0.001 * np.arange(n_days)
    noise = np.random.normal(0, 10, n_days)

    demand = base_demand * seasonality * trend + noise
    demand = np.maximum(demand, 0)  # 需求非负

    return pd.DataFrame({
        'date': dates,
        'demand': demand
    })


def main():
    """主函数"""
    print("=" * 60)
    print("Multi-Echelon Inventory Optimization 测试")
    print("=" * 60)

    # 1. 初始化优化器
    print("\n[1] 初始化库存优化器...")
    optimizer = InventoryOptimizer(service_level=0.95)
    print(f"   目标服务水平: {optimizer.service_level * 100}%")
    print(f"   Z-score: {optimizer.z_score:.2f}")

    # 2. 生成模拟数据
    print("\n[2] 生成模拟数据...")
    df = generate_sample_data()
    print(f"   数据天数: {len(df)}")
    print(f"   平均日需求: {df['demand'].mean():.1f}")
    print(f"   需求标准差: {df['demand'].std():.1f}")

    # 月度汇总
    monthly = df.resample('ME', on='date').agg({'demand': ['sum', 'std']})
    monthly.columns = ['monthly_demand', 'monthly_std']
    monthly = monthly.dropna()
    print(f"   月均销量: {monthly['monthly_demand'].mean():.0f}")

    # 3. 多阶库存优化
    print("\n[3] 多阶库存优化...")
    lead_times = [7, 15, 25]  # 各阶段提前期 (天)
    # 国内仓 7 天 -> 海外仓 15 天 -> 配送 25 天

    demand_mean = df['demand'].mean()
    demand_std = df['demand'].std()
    holding_cost = 1.0  # 单位/天
    shortage_cost = 10.0  # 单位缺货成本

    results = optimizer.multi_echelon_optimize(
        demand_mean, demand_std,
        lead_times,
        holding_cost, shortage_cost
    )

    print("\n   各阶段库存建议:")
    print("-" * 60)
    for _, row in results.iterrows():
        print(f"   阶段 {row['stage']} (提前期 {row['lead_time']} 天):")
        print(f"     - 安全库存: {row['safety_stock']}")
        print(f"     - 再订货点: {row['reorder_point']}")
        print(f"     - 月持有成本: ¥{row['monthly_holding_cost']:.0f}")

    # 4. 动态安全库存测试
    print("\n[4] 动态安全库存测试...")
    recent_30d = df['demand'].values[-30:]
    dynamic_ss, trend = optimizer.dynamic_safety_stock(recent_30d, lead_time=15)
    print(f"   近 30 天平均需求: {recent_30d.mean():.1f}")
    print(f"   趋势因子: {trend:.2f}")
    print(f"   动态安全库存: {dynamic_ss}")

    # 5. 报童模型测试
    print("\n[5] 报童模型测试...")
    selling_price = 100
    cost = 40
    salvage_value = 20

    optimal_order = optimizer.newsvendor_optimal_order(
        selling_price, cost, salvage_value,
        demand_mean * 30, demand_std * np.sqrt(30)
    )

    print(f"   售价: ¥{selling_price}")
    print(f"   进价: ¥{cost}")
    print(f"   残值: ¥{salvage_value}")
    print(f"   月需求均值: {demand_mean * 30:.0f}")
    print(f"   月需求标准差: {demand_std * np.sqrt(30):.0f}")
    print(f"   最优订购量: {optimal_order}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return optimizer


if __name__ == '__main__':
    optimizer = main()
```

---

## ④ 技能关联

### 前置技能
- **基础统计**：理解均值、标准差、正态分布
- **供应链基础**：了解库存周转、订货提前期概念
- **Python 数据分析**：熟练使用 pandas、numpy

### 延伸技能
- **需求预测**：结合时间序列预测，提升需求预测准确率
- **强化学习库存**：使用 RL 动态优化多阶库存策略
- **供应链仿真**：使用离散事件仿真验证策略效果

### 可组合技能
- **Uplift Modeling**：识别高价值 SKU，优先保障库存
- **时间序列预测**：预测爆款趋势，动态调整安全库存
- **选品决策**：结合库存策略选择新品

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 海外仓备货优化 | 库存成本降低 20-30%（月节省 6-9 万） | 开发 2 周，数据接入 1 周 | 6-9x |
| 爆款动态备货 | 缺货率降低 60%+, 库存周转提升 20% | 开发 1 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要历史销量数据、物流参数
- 技术门槛：中等，需理解库存管理基本原理
- 工程复杂度：中等，需要与 ERP 系统对接
- 维护成本：中等，需要定期更新参数

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- 业务价值高：直接关联仓储成本优化
- 见效快：2-3 周可完成 POC
- 可落地性强：母婴出海海外仓场景明确
- 数据依赖：需要历史销售和物流数据

### 评估依据
1. **海外仓成本**是母婴出海的主要成本项，占总成本 15-25%
2. 库存优化 ROI 高，每投入 1 元可节省 6-9 元
3. 与现有 ERP 系统对接相对简单
4. 可从单一 SKU 试点，逐步扩展到全品类
