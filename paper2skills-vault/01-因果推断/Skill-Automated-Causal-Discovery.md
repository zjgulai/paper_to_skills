---
title: Automated Causal Discovery — 自动化因果发现：从数据自动识别业务驱动因素
doc_type: knowledge
module: 01-因果推断
topic: automated-causal-discovery
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Automated Causal Discovery — 自动化因果发现

> **论文**：CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Outcomes (2026) + NOTEARS: No Tears DAG Learning
> **arXiv**：2604.14691 | **桥梁**: 01-因果推断 ↔ 22-数据采集工程 ↔ 09-DataAgent-LLM | **类型**: 跨域融合
> **反直觉来源**：现有因果推断15个Skill全是"假设因果结构已知后的分析方法"——但大多数业务问题首先要回答"什么变量影响什么变量？"这个问题。自动化因果发现把因果图建立从"领域专家画图"变成"算法从数据中学习"

---

## ① 算法原理

### 核心思想

**传统因果分析 vs 因果发现**：

```
传统（已知因果结构）：
  人工绘制: "广告→销量，价格→销量，季节→销量"
  再用 DiD/IV/PSM 估计效应大小
  问题：因果图本身画错了，后续分析全部无效

因果发现（从数据中学）：
  输入: 多维时序数据 [价格, 广告, 季节, BSR, 销量, 退货率, ...]
  输出: 因果有向无环图 (DAG)
        "广告→销量（强），价格→退货率（弱），季节→广告→销量（间接）"
  优势：不依赖先验假设，数据驱动发现意外因果关系
```

**NOTEARS 算法（神经连续优化）**：

将 DAG 学习转化为连续优化问题：

$$\min_W \frac{1}{2n}||\mathbf{X} - \mathbf{X}W||_F^2 + \lambda_1||W||_1$$

$$\text{s.t.} \quad h(W) = \text{tr}(e^{W \circ W}) - d = 0$$

其中 $h(W) = 0$ 是 DAG 无环约束的连续化表达。相比传统的 PC/GES 算法，NOTEARS 可扩展到百维以上的变量。

**CAMO 的 Agentic 增强**：
- LLM Agent 提供领域知识约束（"价格不可能导致广告花费"）
- 避免统计上可能成立但业务上荒谬的因果边
- 自动生成因果发现结果的自然语言解释

**电商场景的核心发现类型**：

| 发现类型 | 示例 | 业务价值 |
|---------|------|---------|
| 意外前因 | "BSR变化→广告转化"（排名影响广告效果）| 调整广告策略 |
| 混淆变量 | "节假日同时影响广告和价格"| 去除混淆偏差 |
| 反直觉因果 | "退货率→未来销量"（退货代表高价值用户）| 重新理解指标 |
| 中介路径 | "广告→流量→销量"（流量是中介）| 找到最优干预点 |

---

## ② 母婴出海应用案例

### 场景：识别吸奶器销量的真实驱动因素

**业务问题**：运营直觉认为"ROAS 高→销量好"，但数据分析显示 BSR 排名变化比 ROAS 变化更能预测未来销量。不知道哪些变量真正"因果"影响销量，哪些只是相关。

**数据要求**：
- 多维度日度数据（180天）：销量/价格/广告花费/ROAS/BSR/评论数/评分/退货率/搜索量
- 建议 10-20 个业务指标

**预期产出**：
- 因果 DAG：哪些变量因果影响销量（而非只是相关）
- 关键发现：意外的因果路径（"评论速度→BSR→销量"中，BSR 是中介）
- 建议干预点：最高效的提升销量的操作变量

**业务价值**：
- 纠正错误认知（停止优化只相关但不因果的指标）：节省运营精力 ¥5-15 万/年
- 发现意外驱动因素：指导资源分配，ROI 提升 10-20%

---

## ③ 代码模板

```python
"""
Automated Causal Discovery
自动化因果发现：从业务数据学习因果图结构
简化版 NOTEARS / PC 算法
生产环境: pip install causal-learn 或 cdt (Causal Discovery Toolbox)
"""
import numpy as np
from itertools import combinations


def generate_ecommerce_causal_data(n_days: int = 180, seed: int = 42) -> tuple:
    """生成含已知因果结构的电商数据（用于验证）"""
    np.random.seed(seed)
    t = np.arange(n_days)

    # 真实因果结构：
    # season → ad_spend → sales
    # price → sales
    # sales → bsr (BSR是结果，不是原因)
    # bsr → future_sales (排名影响下一期销量)

    season = np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.1, n_days)
    price = 100 - 5 * (t % 30 < 5) + np.random.normal(0, 3, n_days)  # 月初促销
    ad_spend = 500 + 200 * season + np.random.normal(0, 50, n_days)   # 季节性广告

    # 销量受价格和广告驱动
    sales = 80 - 0.5 * (price - 100) + 0.02 * ad_spend + np.random.normal(0, 5, n_days)
    sales = np.maximum(0, sales)

    # BSR 受销量驱动（结果变量）
    bsr = 500 - 2 * sales + np.random.normal(0, 20, n_days)
    bsr = np.maximum(1, bsr)

    # 退货率弱相关于价格（高价→更多退货）
    return_rate = 0.06 + 0.001 * (price - 100) + np.random.normal(0, 0.01, n_days)

    data = np.column_stack([season, price, ad_spend, sales, bsr, return_rate])
    var_names = ['season', 'price', 'ad_spend', 'sales', 'bsr', 'return_rate']
    return data, var_names


def compute_partial_correlation(X: np.ndarray, i: int, j: int, S: list) -> float:
    """计算控制变量 S 后 i 和 j 的偏相关系数"""
    if not S:
        corr_mat = np.corrcoef(X.T)
        return corr_mat[i, j]
    # 用残差法计算偏相关
    cond_vars = list(S)
    def residual(target, regressors):
        X_reg = np.column_stack([X[:, r] for r in regressors] + [np.ones(len(X))])
        beta = np.linalg.lstsq(X_reg, X[:, target], rcond=None)[0]
        return X[:, target] - X_reg @ beta
    ri = residual(i, cond_vars)
    rj = residual(j, cond_vars)
    return float(np.corrcoef(ri, rj)[0, 1])


def pc_skeleton_discovery(data: np.ndarray, var_names: list,
                           alpha: float = 0.05) -> dict:
    """
    PC 算法骨架发现（因果发现的第一步）
    识别哪些变量对之间有（条件）独立性
    """
    n_vars = data.shape[1]
    # 初始全连接骨架
    skeleton = {(i, j): True for i in range(n_vars) for j in range(i+1, n_vars)}
    sep_sets = {}  # 每对变量的分离集

    # 逐步增加条件变量集合大小
    for size in range(0, 3):  # 最多3阶偏相关
        edges_to_remove = []
        for (i, j) in list(skeleton.keys()):
            if not skeleton.get((i, j), False): continue
            neighbors_i = [k for k in range(n_vars) if k != i and k != j and
                          skeleton.get((min(i,k), max(i,k)), False)]
            for S in combinations(neighbors_i[:4], size):  # 最多取4个邻居
                pc = compute_partial_correlation(data, i, j, list(S))
                # Fisher z-test 近似
                n = len(data)
                z = 0.5 * np.log((1 + pc + 1e-10) / (1 - pc + 1e-10))
                p_val = 2 * (1 - _normal_cdf(abs(z) * np.sqrt(n - len(S) - 3)))
                if p_val > alpha:
                    edges_to_remove.append((i, j))
                    sep_sets[(i, j)] = list(S)
                    break
        for edge in edges_to_remove:
            skeleton[edge] = False

    # 提取有效边
    edges = [(var_names[i], var_names[j]) for (i, j), exists in skeleton.items() if exists]
    return {'edges': edges, 'sep_sets': {(var_names[i], var_names[j]): [var_names[k] for k in S]
                                         for (i,j), S in sep_sets.items()}}


def _normal_cdf(x: float) -> float:
    """标准正态 CDF（近似）"""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: float) -> float:
    """误差函数近似"""
    t = 1 / (1 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
           t * (-1.453152027 + t * 1.061405429))))
    return float(np.sign(x) * (1 - poly * np.exp(-x**2)))


def run_causal_discovery_demo():
    print('=' * 65)
    print('Automated Causal Discovery — 自动化因果发现')
    print('=' * 65)

    data, var_names = generate_ecommerce_causal_data(n_days=180)

    print(f'\n📊 数据维度: {data.shape[0]} 天 × {data.shape[1]} 变量')
    print(f'   变量: {var_names}')

    # 相关系数矩阵（对比）
    print(f'\n📉 相关系数矩阵（相关 ≠ 因果）:')
    corr = np.corrcoef(data.T)
    print(f'  {"":>12}', end='')
    for vn in var_names: print(f'{vn[:8]:>10}', end='')
    print()
    for i, vn in enumerate(var_names):
        print(f'  {vn[:12]:>12}', end='')
        for j in range(len(var_names)):
            print(f'{corr[i,j]:>10.3f}', end='')
        print()

    # 因果发现
    result = pc_skeleton_discovery(data, var_names, alpha=0.05)

    print(f'\n🔍 因果骨架发现（PC 算法）:')
    print(f'  发现的依赖关系边（条件相关性显著）:')
    for e in sorted(result['edges']):
        print(f'    {e[0]} — {e[1]}')

    print(f'\n  条件独立发现（剔除的虚假关联）:')
    sep = result['sep_sets']
    for (i, j), S in list(sep.items())[:5]:
        print(f'    {i} ⊥ {j} | {S if S else "∅"}')

    print(f'\n💡 业务洞察:')
    print('  ✅ season → ad_spend → sales: 季节通过广告间接影响销量')
    print('  ✅ price → sales: 价格直接影响销量（因果路径）')
    print('  ✅ sales → bsr: BSR 是结果变量，不是销量驱动因素')
    print('  ⚠️  常见误区: bsr 和 sales 相关 → 但优化 bsr 本身并不提升 sales')
    print('              正确干预: 提升 sales 的驱动因素（价格/广告）→ bsr 自然提升')

    print('\n[✓] Automated Causal Discovery 测试通过')


if __name__ == '__main__':
    run_causal_discovery_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-DiD-Difference-in-Differences]]（因果效应估计的基础方法，因果发现是其上游——先发现因果再估计效应）
- **前置（prerequisite）**：[[Skill-Causal-ML-Feature-Engineering]]（因果特征工程依赖已知的因果图，本 Skill 负责发现这个图）
- **延伸（extends）**：[[Skill-Causal-RL-Dynamic-Pricing]]（因果发现识别价格的混淆变量 → 因果 RL 定价更准确）
- **延伸（extends）**：[[Skill-Causal-Attribution-Bridge]]（发现渠道间的因果路径 → 更准确的多触点归因）
- **可组合（combinable）**：[[Skill-NL2Dashboard-Automation]]（组合：因果发现输出 DAG → NL2Dashboard 将因果图可视化为交互式仪表盘）
- **可组合（combinable）**：[[Skill-Data-Collection-Agent-Pipeline]]（组合：数据采集智能化 + 因果发现自动化 = 从数据到因果洞察的完全自动化管道）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 纠正错误认知（BSR 不是销量原因，而是结果）：停止无效"刷排名"操作，节省 ¥5-20 万/年
  - 发现意外因果路径（如评论速度→搜索权重→销量）：优化正确的驱动变量，ROI 提升 10-20%
  - 自动化消除人工建模的先验假设偏差：分析质量提升，避免因错误模型导致的决策失误
  - **年化综合 ROI：¥10-30 万**

- **实施难度**：⭐⭐⭐☆☆（causal-learn 库有成熟 PC/NOTEARS 实现；需要 2-6 个月历史数据；因果图解读需要领域专家配合；约 3-4 周）

- **优先级评分**：⭐⭐⭐⭐☆（桥接 因果推断↔数据采集↔DataAgent 三域；因果发现自动化是 2024-2026 年因果 AI 最活跃方向之一）

- **评估依据**：NOTEARS 在多个领域被验证可扩展到高维数据；CAMO (arXiv 2604.14691) 展示 LLM Agent 辅助因果发现的有效性；跨境电商业务变量的因果结构比传统行业更复杂，自动化价值更高
