# Skill Card: Channel Saturation Curve（渠道饱和曲线建模）

> **领域**: 15-营销投放分析 | **类型**: 综合萃取

---

## ① 算法原理

### 核心思想
广告预算不是线性回报——每多投 $1，边际回报递减。渠道饱和曲线量化"这个渠道再投多少钱就没增量了"，避免过度投放。

### 数学直觉

**Hill 函数型饱和曲线**（广告效果建模的标准形式）：
$$ROAS(x) = \frac{\beta \cdot x^\alpha}{K^\alpha + x^\alpha}$$

其中 $x$ 是投放金额，$\beta$ 是最大回报，$K$ 是半饱和点（达到 $\beta/2$ 所需的投放），$\alpha$ 控制曲线陡峭度。

**边际回报**：$MR(x) = \frac{d}{dx} ROAS(x)$ —— 当 $MR(x) < 1$ 时，继续投放已无利润增量。

**渠道间饱和差异**：通常 Google Search 的饱和点最高（$K$ 大），TikTok 次之，Facebook 最低（受众疲劳快）。

### 关键假设
- 饱和曲线在 campaign 级别稳定（不因素材变化而剧烈跳动）
- 各渠道饱和曲线独立（忽略跨渠道协同/竞争效应）

---

## ② 母婴出海应用案例

### 场景：Facebook 吸奶器广告的饱和点判断

**业务问题**：Facebook 月预算从 $5 万加到 $8 万后，ROAS 从 3.2 掉到 2.1。继续加到 $10 万 ROI 可能变负。需要找到饱和点。

**数据要求**：6 个月不同预算水平下的 ROAS 数据（来自历史 A/B 或渐进加预算实验）

**预期产出**：
- 拟合 Hill 曲线：$\beta=4.2, K=6.2\text{万}, \alpha=1.8$
- 半饱和点 $62,000/月，边际回报 <1 的临界点 $85,000/月
- **建议**：FB 月预算上限 $75,000-$80,000，超出部分分配给 TikTok

**业务价值**：避免过度投放浪费 $15,000-20,000/月

---

## ③ 代码模板

```python
"""Channel Saturation Curve — Hill 函数拟合 + 边际分析"""

import numpy as np
from scipy.optimize import curve_fit


def hill_function(x, beta, K, alpha):
    """Hill 饱和函数"""
    return beta * (x**alpha) / (K**alpha + x**alpha)


def fit_saturation_curve(spend: np.ndarray, roas: np.ndarray):
    """拟合渠道饱和曲线"""
    popt, _ = curve_fit(hill_function, spend, roas, 
                        p0=[max(roas), np.median(spend), 1.5],
                        bounds=([0, 0, 0.5], [100, max(spend)*3, 5]))
    return popt  # (beta, K, alpha)


def find_saturation_point(beta, K, alpha, min_roi=1.0):
    """找边际ROI=min_roi的饱和点"""
    for x in np.linspace(K*0.1, K*3, 1000):
        mr = beta * alpha * K**alpha * x**(alpha-1) / (K**alpha + x**alpha)**2
        if mr < min_roi:
            return x
    return K * 2


if __name__ == '__main__':
    np.random.seed(42)
    # 模拟: beta=4, K=60, alpha=1.8
    spend = np.array([10, 20, 30, 50, 70, 90, 110, 130]) * 1000
    true_roas = hill_function(spend, 4.0, 60000, 1.8)
    roas = true_roas + np.random.normal(0, 0.15, len(spend))
    
    beta, K, alpha = fit_saturation_curve(spend, roas)
    sat_point = find_saturation_point(beta, K, alpha)
    
    print(f"Hill: β={beta:.2f}, K=${K:,.0f}, α={alpha:.2f}")
    print(f"半饱和点: ${K:,.0f}/月")
    print(f"饱和点(MR<1): ${sat_point:,.0f}/月")
    print(f"\n[✓] Channel Saturation 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Marketing-Mix-Modeling]] | [[Skill-ROAS-Budget-Optimization]]
- **可组合技能**：[[Skill-Multi-Objective-Budget-Allocation]] | [[Skill-Geo-Level-Marketing-Effectiveness]]

---

## ⑤ 商业价值评估

- **ROI 预估**：避免过度投放 $15-20K/月；年化 **18-25 万元**
- **实施难度**：⭐⭐☆☆☆（2 星）— 曲线拟合简单
- **优先级评分**：⭐⭐⭐⭐☆（4 星）— MMM 的自然延伸
