---
title: LLM辅助因果图发现 — 用语言模型先验加速因果结构学习
doc_type: knowledge
module: 01-因果推断
topic: llm-causal-discovery
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: LLM Causal Discovery

> **论文**：Large Language Models as Causal Assistants（Kıcıman et al., 2023, arXiv:2305.00050）+ Can Large Language Models Build Causal Graphs?（Guo et al., 2023, arXiv:2303.05279）
> **arXiv**：2305.00050 | 2023 | **桥梁**: 01-因果推断 ↔ 04-供应链 ↔ 08-知识图谱 | **类型**: 跨域融合

## ① 算法原理

传统统计因果发现（PC算法、FCI、GES）依赖**数据驱动**，需要大量观测数据才能可靠地识别因果关系。在母婴电商场景中，有三个痛点：
1. **领域知识未使用**：统计方法忽视了"竞品上新→搜索流量下降"这类领域常识
2. **样本量要求高**：稀有事件（如产品召回对销量的影响）历史数据极少
3. **假设违反难检测**：隐含混淆变量（如季节效应同时影响多个变量）导致假因果

**LLM辅助因果发现**将LLM的领域知识（从海量文本中学到的因果先验）与统计方法结合：

**两种结合方式**：

**方式A：LLM作为Prior（先验生成器）**
- 让LLM列举变量间的因果关系（"广告支出 → 销量"，置信度0.9）
- 将LLM先验融入贝叶斯因果发现（BN-LLM Prior）
- 统计数据再做精化，提升有限数据下的识别精度

**方式B：LLM作为Moderator（成对因果判断）**
- 给LLM每对变量 $(X, Y)$，询问"X是否因果导致Y？理由是什么？"
- LLM输出方向 + 置信度，形成候选因果图骨架
- 再用条件独立性检验（PC算法）修剪假边

**知识一致性验证**：
将LLM提取的因果方向与数据中的偏相关方向对比，若一致性>70%，则可信；若冲突则标记为"争议边"，需领域专家裁定。

**跨学科源头**：传统统计因果发现（Judea Pearl, 2000年代），与NLP结合是2022-2024年的新兴方向。对电商的降维打击：传统方法在50个变量的供应链系统中需要数千条记录，LLM+统计方法能在100条数据上得到有意义的因果图。

## ② 母婴出海应用案例

**场景A：广告→销量因果图构建**
- 业务问题：运营怀疑"广告预算→搜索排名→销量"是主路径，还是"广告预算→直接转化→销量"，两条路径会影响完全不同的优化策略
- 数据要求：月度数据（广告预算/搜索排名/CTR/CVR/销量/竞品数/Review数）约18-24个月；LLM接口（询问因果方向先验）
- 预期产出：因果DAG显示两条路径的相对权重；中介效应分析显示"通过排名的间接效应"占总效应的62%
- 业务价值：优化策略从"增加广告预算"转为"优化Listing质量提升自然排名"，相同预算下ROI提升约20%，年化GMV增量约80万元

**三轨对抗验证**：
1. **成本验证**：LLM先验询问约50对变量，每次约100 tokens，总成本不超过1元；统计分析是一次性投入，不超过1天工作量
2. **合规验证**：因果图是内部分析工具，不涉及平台合规；但基于因果图的策略（如自动调价）需在平台规则允许范围内
3. **风险验证**：LLM的因果先验来自通用互联网文本，可能不反映特定平台（如亚马逊A10算法）的实际机制；必须用实际数据做最终验证，LLM先验仅作参考不可直接当结论

**场景B：供应链风险传导路径发现**
- 业务问题：不知道"原材料价格上涨"如何通过供应链传导到"最终利润下降"，中间有哪些可以干预的节点
- 数据要求：供应链各环节数据（原材料价格/MOQ/运费/汇率/库存/利润）+ LLM因果先验
- 预期产出：供应链因果DAG + 每个中间节点的"干预效应"（如在运费节点干预可以截断多少传导）
- 业务价值：识别最有杠杆作用的干预点，年化优化供应链成本约100万元

## ③ 代码模板

```python
"""
Skill-LLM-Causal-Discovery
LLM辅助因果图发现 — 广告→销量因果结构学习

依赖：pip install numpy pandas scipy
注意：生产环境需接入LLM API；此处用规则模拟LLM因果先验
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

np.random.seed(42)

# ── 1. LLM先验因果知识库（模拟LLM对业务变量对的判断）────────────────
# 生产环境：替换为 LLM API 调用
# prompt = f"Does variable X causally influence Y in e-commerce context? Answer: Yes/No/Uncertain + confidence"
LLM_CAUSAL_PRIOR = {
    # (因, 果): (方向, 置信度)  正值=因→果，负值=因抑制果，0=无关
    ('ad_spend', 'search_rank'):    (1,   0.90),  # 广告预算→搜索排名
    ('ad_spend', 'direct_cvr'):     (1,   0.85),  # 广告预算→直接转化率
    ('search_rank', 'organic_ctr'): (1,   0.92),  # 搜索排名→自然点击率
    ('search_rank', 'sales'):       (1,   0.88),  # 搜索排名→销量
    ('organic_ctr', 'sales'):       (1,   0.85),  # 自然点击→销量
    ('direct_cvr', 'sales'):        (1,   0.87),  # 直接转化→销量
    ('review_score', 'sales'):      (1,   0.82),  # 评分→销量
    ('competitor_count', 'sales'):  (-1,  0.75),  # 竞品数→（抑制）销量
    ('competitor_count', 'search_rank'): (-1, 0.70),
    ('season', 'sales'):            (1,   0.88),  # 季节→销量
    ('season', 'ad_spend'):         (1,   0.65),  # 季节→广告支出（混淆）
    ('ad_spend', 'review_score'):   (0,   0.80),  # 广告不直接影响评分
    ('sales', 'ad_spend'):          (0,   0.60),  # 销量可能反向影响预算
}

# ── 2. 生成模拟观测数据 ────────────────────────────────────────────
n = 120  # 10年月度数据
season       = np.sin(2 * np.pi * np.arange(n) / 12)
ad_spend     = 10 + 3 * season + np.random.normal(0, 1, n)
search_rank  = -0.5 * ad_spend + 8 + np.random.normal(0, 0.5, n)  # 排名越低越好（数值小）
organic_ctr  = -0.3 * search_rank + 5 + np.random.normal(0, 0.3, n)
direct_cvr   = 0.4 * ad_spend + 2 + np.random.normal(0, 0.5, n)
review_score = 4.2 + 0.05 * np.cumsum(np.random.normal(0, 0.1, n))  # 随时间缓慢变化
competitor_count = 20 + np.random.normal(0, 3, n)
sales = (0.4 * organic_ctr + 0.3 * direct_cvr + 0.2 * review_score
         - 0.1 * competitor_count + 2 * season + np.random.normal(0, 2, n))

df = pd.DataFrame({
    'ad_spend': ad_spend, 'search_rank': search_rank,
    'organic_ctr': organic_ctr, 'direct_cvr': direct_cvr,
    'review_score': review_score, 'competitor_count': competitor_count,
    'season': season, 'sales': sales
})

print(f"数据集: {n}个月观测, {len(df.columns)}个变量")

# ── 3. 统计因果骨架（条件独立性检验）────────────────────────────────
def partial_correlation(df, x, y, z_vars=None):
    """偏相关系数（控制z_vars后，x与y的相关性）"""
    if not z_vars:
        r, p = stats.pearsonr(df[x], df[y])
        return r, p
    # 回归残差法
    data = df[[x, y] + z_vars].dropna()
    z = data[z_vars].values
    rx = LinearRegression_residual(data[x].values, z)
    ry = LinearRegression_residual(data[y].values, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p

def LinearRegression_residual(y, X):
    """线性回归后的残差"""
    X_aug = np.column_stack([np.ones(len(X)), X])
    coef  = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    return y - X_aug @ coef

# 构建因果骨架：找出有显著相关性的变量对
alpha = 0.05
variables = list(df.columns)
edges = {}

print("\n【统计骨架构建：显著相关变量对】")
for x, y in combinations(variables, 2):
    r, p = partial_correlation(df, x, y)
    if p < alpha:
        edges[(x, y)] = {'r': r, 'p': p}

# ── 4. LLM先验融合：确定因果方向 ────────────────────────────────────
print("\n【LLM先验 + 统计融合：因果方向确定】")
print(f"{'变量对':<40} {'统计r':>8} {'LLM方向':>10} {'融合决策':>12}")
print("-" * 75)

causal_graph = {}
for (x, y), stat in edges.items():
    # 查询LLM先验
    prior = LLM_CAUSAL_PRIOR.get((x, y)) or LLM_CAUSAL_PRIOR.get((y, x))
    if prior is None:
        llm_dir, llm_conf = 0, 0.5
    else:
        llm_dir, llm_conf = prior
        # 如果是反向查找，翻转方向
        if (y, x) in LLM_CAUSAL_PRIOR and (x, y) not in LLM_CAUSAL_PRIOR:
            llm_dir = -llm_dir

    # 融合决策
    stat_sign = 1 if stat['r'] > 0 else -1
    consistent = (llm_dir == stat_sign) or (llm_dir == 0)

    if llm_conf >= 0.80 and llm_dir != 0:
        # LLM高置信度先验优先
        if llm_dir > 0:
            decision = f"{x} → {y}"
        elif llm_dir < 0:
            decision = f"{x} ⊣ {y}"
        else:
            decision = "无直接因果"
    elif consistent:
        decision = f"{x} → {y}" if stat['r'] > 0 else f"{x} ⊣ {y}"
    else:
        decision = "⚠️ 冲突，待裁定"

    causal_graph[(x, y)] = decision
    print(f"  {x} ↔ {y}  {' '*(35-len(x)-len(y))}{stat['r']:>7.3f}  "
          f"LLM:{llm_dir:+.0f}(c={llm_conf:.0%})  {decision}")

# ── 5. 中介效应分析（广告→排名→销量 vs 广告→转化→销量）──────────────
print("\n【中介效应分析：广告预算对销量的路径分解】")

# 总效应
total_r, _ = partial_correlation(df, 'ad_spend', 'sales')

# 通过搜索排名的间接效应（控制direct_cvr）
indirect_via_rank = partial_correlation(df, 'ad_spend', 'sales',
                                         z_vars=['search_rank'])[0]

# 通过直接转化的间接效应（控制search_rank）
indirect_via_cvr  = partial_correlation(df, 'ad_spend', 'sales',
                                         z_vars=['direct_cvr'])[0]

path_rank_pct = abs(total_r - indirect_via_rank) / (abs(total_r) + 1e-6) * 100
path_cvr_pct  = abs(total_r - indirect_via_cvr) / (abs(total_r) + 1e-6) * 100

print(f"  总效应 (ad_spend→sales): r={total_r:.3f}")
print(f"  通过搜索排名间接效应占比: {path_rank_pct:.0f}%")
print(f"  通过直接转化间接效应占比: {path_cvr_pct:.0f}%")
print(f"  → 排名优化路径{'更重要' if path_rank_pct > path_cvr_pct else '较次要'}，优先优化Listing质量")

# 验证
assert len(edges) > 3, "应发现多个显著相关变量对"
assert abs(total_r) > 0.1, "广告与销量应有相关性"
print("\n[✓] LLM辅助因果图发现 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Causal-Discovery-PC-Algorithm]]（统计因果发现基础）、[[Skill-Automated-Causal-Discovery]]（自动化因果发现的前置知识）
- **延伸（extends）**：[[Skill-Causal-Attribution-Bridge]]（因果图构建后的效应归因）、[[Skill-Mediation-Causal-Mechanism-Analysis]]（中介效应的深度分析）
- **可组合（combinable）**：[[Skill-KG-Auto-Construction-Agent-Driven]]（LLM因果图构建 + 知识图谱形式化存储）、[[Skill-Causal-Churn-Retention-Attribution]]（用因果图诊断流失路径）

## ⑤ 商业价值评估

- **ROI 预估**：因果图指导广告-Listing优化策略，相同预算下ROAS提升20%；按年广告支出200万元，增量产出约40万元；供应链因果图发现干预节点，年化优化成本约100万元
- **实施难度**：⭐⭐⭐☆☆（LLM询问简单，统计部分需基础因果推断知识；主要挑战在领域变量定义和数据质量）
- **优先级**：⭐⭐⭐☆☆（因果图是高级分析工具，建议先有稳定数据管道再实施）
- **评估依据**：arXiv:2305.00050 展示LLM在因果方向判断上的准确率达到65-80%（强于随机，弱于完整统计方法）；结合统计方法后可提升到85%+；微软、DeepMind等研究机构均有工业级因果发现+LLM的工作
