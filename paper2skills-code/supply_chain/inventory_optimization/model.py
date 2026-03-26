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
