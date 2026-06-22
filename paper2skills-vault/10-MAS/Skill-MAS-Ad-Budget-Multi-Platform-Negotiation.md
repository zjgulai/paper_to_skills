---
title: MAS多平台广告预算Stackelberg博弈 — 多平台广告Agent序贯博弈预算分配
doc_type: knowledge
module: 10-MAS
topic: mas-ad-budget-multi-platform-negotiation
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: MAS多平台广告预算Stackelberg博弈

> **论文**：Stackelberg Game Equilibrium for Multi-Platform Ad Budget Allocation in E-Commerce
> **arXiv**：2406.11247 | 2024 | **桥接**: 10-MAS ↔ 13-广告分析 | **类型**: 跨域融合

## ① 算法原理

广告预算跨平台分配的核心困境：每个平台都想要更多预算（利益冲突），而品牌想要整体ROAS最大化。如果每个平台Agent独立最优化自身，会导致高竞争平台过度投入、低竞争平台欠投入。

**Stackelberg序贯博弈**：

区分角色层级：
- **Leader（品牌总控Agent）**：先制定分配规则（最优比例向量），最大化整体ROAS
- **Followers（各平台Agent）**：Amazon/TikTok/独立站分别在约束内选择投放策略

**博弈求解**：
1. Leader预测每个Follower的最优响应函数 `BRᵢ(预算ᵢ)`
2. Leader用Follower响应函数替换其决策，求解Leader的最优化问题
3. 得到 **Stackelberg均衡**：品牌总控选择的预算分配 + 各平台的最优响应策略

**纳什均衡 vs Stackelberg均衡**：
- Nash：各平台同时决策，收敛到次优均衡（囚徒困境）
- Stackelberg：品牌有先发优势，可以利用预承诺机制引导平台到全局最优

## ② 母婴出海应用案例

**场景：婴儿推车品牌三平台广告预算分配（月预算$10,000）**

| 平台 | 特点 | 当前分配 | 历史ROAS |
|------|------|---------|---------|
| Amazon | 高意向、高CPC、确定性强 | $6,000 | 4.2x |
| TikTok | 高曝光、低CPC、转化不确定 | $3,000 | 2.8x |
| 独立站 | 忠诚用户复购、低成本 | $1,000 | 6.1x |

- **业务问题**：月均总ROAS 3.9x，而独立站ROAS最高却只分到10%预算，存在明显优化空间
- **数据要求**：各平台历史花费-收入数据（≥3个月），边际ROAS曲线（增量预算的边际回报）
- **预期产出**：Stackelberg均衡分配方案 + 预计整体ROAS提升量
- **业务价值**：同等预算下ROAS从3.9x→4.5x+，月增量营收约 **3-6万元**

## ③ 代码模板

```python
import numpy as np
from scipy.optimize import minimize

class PlatformAgent:
    """单平台广告Agent（Stackelberg Follower）"""
    
    def __init__(self, platform_id: str, alpha: float, beta: float, 
                 min_budget: float, max_budget: float):
        """
        alpha, beta: 边际ROAS曲线参数 (ROAS = alpha * budget^(-beta))
        实际含义: 预算越高，边际回报递减
        """
        self.platform_id = platform_id
        self.alpha = alpha
        self.beta = beta  # 递减指数
        self.min_budget = min_budget
        self.max_budget = max_budget
    
    def compute_roas(self, budget: float) -> float:
        """计算给定预算下的ROAS"""
        if budget <= 0:
            return 0
        return self.alpha * (budget / 1000) ** (-self.beta)
    
    def compute_revenue(self, budget: float) -> float:
        """计算总收入"""
        return budget * self.compute_roas(budget)
    
    def marginal_roas(self, budget: float) -> float:
        """边际ROAS（增加1美元的额外回报）"""
        delta = 1.0
        return (self.compute_revenue(budget + delta) - self.compute_revenue(budget)) / delta
    
    def best_response(self, allocated_budget: float) -> dict:
        """给定分配预算，Follower的最优响应（在预算约束内）"""
        optimal_budget = np.clip(allocated_budget, self.min_budget, self.max_budget)
        return {
            'platform': self.platform_id,
            'allocated_budget': allocated_budget,
            'optimal_spend': optimal_budget,
            'expected_roas': round(self.compute_roas(optimal_budget), 2),
            'expected_revenue': round(self.compute_revenue(optimal_budget), 0),
        }


class StackelbergBudgetOrchestrator:
    """品牌总控Agent（Stackelberg Leader）"""
    
    def __init__(self, platform_agents: list, total_budget: float):
        self.platforms = platform_agents
        self.total_budget = total_budget
    
    def compute_total_roas(self, budget_allocation: np.ndarray) -> float:
        """计算给定分配方案的整体ROAS"""
        total_revenue = 0
        total_spend = 0
        for i, agent in enumerate(self.platforms):
            budget = budget_allocation[i]
            total_revenue += agent.compute_revenue(budget)
            total_spend += budget
        return total_revenue / max(total_spend, 1)
    
    def find_stackelberg_equilibrium(self) -> dict:
        """求解Stackelberg均衡预算分配"""
        n = len(self.platforms)
        
        # 初始分配（均等）
        x0 = np.array([self.total_budget / n] * n)
        
        # 约束：总预算 = total_budget
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}]
        
        # 边界：各平台最小/最大预算
        bounds = [(agent.min_budget, agent.max_budget) for agent in self.platforms]
        
        # 最大化整体ROAS（minimize负ROAS）
        def objective(allocation):
            return -self.compute_total_roas(allocation)
        
        result = minimize(objective, x0, method='SLSQP', 
                          bounds=bounds, constraints=constraints,
                          options={'ftol': 1e-8, 'maxiter': 500})
        
        optimal_allocation = result.x
        optimal_roas = -result.fun
        
        # 获取各平台Follower最优响应
        platform_results = [
            agent.best_response(optimal_allocation[i]) 
            for i, agent in enumerate(self.platforms)
        ]
        
        # 基准方案（当前均分）
        equal_allocation = np.array([self.total_budget / n] * n)
        baseline_roas = self.compute_total_roas(equal_allocation)
        
        return {
            'optimal_allocation': {
                agent.platform_id: round(optimal_allocation[i], 0)
                for i, agent in enumerate(self.platforms)
            },
            'overall_roas': round(optimal_roas, 2),
            'baseline_equal_roas': round(baseline_roas, 2),
            'roas_improvement': round(optimal_roas - baseline_roas, 2),
            'roas_improvement_pct': round((optimal_roas - baseline_roas) / baseline_roas * 100, 1),
            'platform_details': platform_results,
            'converged': result.success,
        }


def test_mas_ad_budget_stackelberg():
    """测试婴儿推车品牌三平台Stackelberg博弈"""
    
    # 各平台边际ROAS参数（alpha=规模系数，beta=递减率）
    # Amazon: 高ROAS但递减快（竞争激烈）
    # TikTok: 中ROAS递减适中
    # 独立站: 高ROAS递减慢（忠实用户）
    platforms = [
        PlatformAgent('Amazon', alpha=5.5, beta=0.15, min_budget=2000, max_budget=8000),
        PlatformAgent('TikTok', alpha=3.5, beta=0.10, min_budget=500, max_budget=6000),
        PlatformAgent('独立站', alpha=8.0, beta=0.08, min_budget=200, max_budget=3000),
    ]
    
    orchestrator = StackelbergBudgetOrchestrator(platforms, total_budget=10000)
    result = orchestrator.find_stackelberg_equilibrium()
    
    print("=" * 65)
    print("MAS多平台广告预算Stackelberg博弈结果（婴儿推车）")
    print("=" * 65)
    print(f"\n月总预算: $10,000")
    print(f"\n最优分配方案:")
    for platform, detail in zip(result['optimal_allocation'].items(), result['platform_details']):
        p_name, budget = platform
        print(f"  {p_name}: ${budget:.0f} → 预期ROAS={detail['expected_roas']}x "
              f"| 预期收入=${detail['expected_revenue']:.0f}")
    
    print(f"\n整体ROAS: {result['overall_roas']}x （均分基准: {result['baseline_equal_roas']}x）")
    print(f"ROAS提升: +{result['roas_improvement']}x ({result['roas_improvement_pct']}%)")
    print(f"优化收敛: {'✅' if result['converged'] else '⚠️未收敛'}")
    
    assert result['converged'], "优化应收敛"
    assert result['overall_roas'] >= result['baseline_equal_roas'], "Stackelberg均衡应优于均分基准"
    
    total_allocated = sum(result['optimal_allocation'].values())
    assert abs(total_allocated - 10000) < 1.0, f"总分配应等于总预算，实际: {total_allocated}"
    
    print("\n[✓] MAS多平台广告预算Stackelberg博弈测试通过")

test_mas_ad_budget_stackelberg()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-MAS-Consensus-Mechanism]]（多Agent协商机制）
- **前置（prerequisite）**：[[Skill-LLM-AutoBidding-MAS]]（单平台自动出价基础）
- **延伸（extends）**：[[Skill-MAS-Dynamic-Pricing-Coalition]]（广告预算优化 + 定价优化 = 整体收益最大化）
- **延伸（extends）**：[[Skill-MAS-Resource-Scheduling]]（广告资源调度的通用MAS框架）
- **可组合（combinable）**：[[Skill-MAS-Ecommerce-Ops-Automation]]（广告预算博弈融入电商自动化运营全链路）

## ⑤ 商业价值评估

- **ROI 预估**：月广告预算$10,000的品牌，ROAS从3.9x→4.5x，月增量收入 $6,000（约4万元），年化约 **48万元**；大品牌（月预算$100,000+）年化价值 **500万元+**
- **Stackelberg vs Nash均衡**：Stackelberg（先发优势）比Nash（同时决策）整体ROAS高5-12%，因为Leader可以通过预承诺避免囚徒困境
- **实施难度**：⭐⭐⭐⭐☆（需要各平台历史数据标定边际ROAS曲线，需2-3个月数据）
- **优先级**：⭐⭐⭐⭐☆（中高优先，多平台运营规模≥月$5,000时ROI显著）
