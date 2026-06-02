---
title: ClusterSC - 聚类合成控制法
doc_type: knowledge
module: 01-因果推断
topic: clustersc-synthetic-control
status: stable
created: 2026-05-19
updated: 2026-05-19
owner: self
source: human+ai
paper: arXiv:2503.21629
---

# Skill: ClusterSC — 聚类合成控制法

> 论文：**ClusterSC: Advancing Synthetic Control with Donor Clustering** (2025-03) · arXiv:2503.21629
>
> 方法论扩展：**Synthetic Control Method** (Abadie & Gardeazabal 2003)

---

## ① 算法原理

### 核心思想

合成控制法（Synthetic Control, SC）通过为目标单元"合成"一个反事实来估计因果效应，是评估地区级/城市级无法做 A/B 的大型干预（如区域广告投放、城市政策）的黄金标准。

传统 SC 的致命弱点：当供体池从几十个州扩展到数千个县/门店时，维度灾难（Curse of Dimensionality）导致权重过度拟合，合成反事实严重失真。

**ClusterSC** 在 SC 前插入一个聚类层，三步完成：
1. **供体聚类**：K-Means 对所有 donor 历史时序特征聚类，把数千供体降至 K 个簇代表。
2. **目标匹配**：找到与目标单元历史行为最相似的单一簇，提取其成员作为缩减供体池。
3. **合成权重**：在缩减池内用约束最小二乘（NNLS）求权重，预测反事实并计算处理效应。

### 数学直觉

**经典 SC 权重问题**：

$$\hat{W}^* = \arg\min_{W \geq 0,\ \mathbf{1}^T W=1} \|Y_{\text{pre}}^{\text{treated}} - W^T Y_{\text{pre}}^{\text{donors}}\|_2^2$$

当供体数 $N \gg T_{\text{pre}}$ 时，上式欠定，权重过度拟合。

**ClusterSC 解法**：令 $C_k$ 为最近簇的 donor 集合（$|C_k| \ll N$），将上式限定在 $C_k$ 内求解，使问题从 $N$ 维降至 $|C_k|$ 维，**理论上可证明预测误差上界更紧（tighter upper bound）**。

**误差上界改善**（论文定理）：

$$\text{MSE}_{\text{ClusterSC}} \leq \text{MSE}_{\text{SC-full}} \cdot f(\sigma^2_{\text{noise}},\ |C_k|,\ N)$$

当噪声方差 $\sigma^2$ 不可忽略时，$f < 1$，聚类减少了噪声引入量。

### 关键假设

1. **平行趋势**（干预前）：目标单元与某个供体簇共享相同的潜在趋势驱动因子。
2. **稳定单元干预值**（SUTVA）：干预单元的处理不影响供体单元的结果。
3. **供体充分**：至少存在一个簇，其历史时序与目标单元高度相关。
4. **干预后无泄漏**：目标单元干预效应不扩散到供体池（无溢出效应）。

---

## ② 母婴出海应用案例

### 场景一：TikTok 品牌大促在北美单一区域的增量归因

- **业务问题**：母婴品牌在黑五期间，仅对美国加州投放 50 万美元 TikTok 联动广告。加州销量上涨，但无法区分黑五自然增长和广告带来的增量。若直接拿 49 个州合成"伪加州"，因各州差异悬殊极易产生噪声权重。

- **数据要求**：
  - 面板格式：行=地理单元（县级，约 3000 个），列=月度销量（干预前≥12 个月）
  - 干预标记：`is_treated=1` 对加州，`is_treated=0` 对其余所有县
  - 建议特征：月度销量、月度访客数、历史季节性指数

- **ClusterSC 落地**：
  1. K-Means 对 2999 个供体县聚类（K=20），缩减至与加州行为同频的 30-80 个县
  2. NNLS 在缩减供体池内拟合合成加州的反事实销量曲线
  3. ATT = 干预后加州实际销量 - 合成反事实，即为 TikTok 广告纯增量

- **预期产出**：
  - 每月 TikTok 广告带来的增量销量（单位：件）
  - 增量 GMV（增量销量 × 客单价）
  - 50 万美元投放产生的 ROI = 增量 GMV / 50 万

- **业务价值**：为动辄百万美金的"不可分割型"品牌广告（OOH、TikTok TopView）提供高精度因果 ROI 核算，指导下季度媒介预算分配。按中型母婴品牌估算：发现无效渠道后优化媒介组合，年化节省 200-500 万元广告浪费。

---

### 场景二：独立站新市场开拓的地理实验归因

- **业务问题**：品牌在德国率先上线本地化独立站（含德语客服、本地支付），其他欧洲国家维持原版英文站。3 个月后德国销量大幅提升，但欧洲整体消费也在上涨，需要剥离本地化改版的纯增量效应。

- **数据要求**：
  - 面板：行=欧洲各国，列=周度 GMV（干预前≥26 周）
  - 目标单元：德国；供体池：其余 20-30 个欧洲国家
  - 注：此场景供体数量少（20-30），无需聚类，但验证了 ClusterSC 对小规模供体的兼容性（K=min(K,n_donors)）

- **ClusterSC 落地**：K-Means K=5（数据少时自动退化到 SC 模式），NNLS 拟合周度反事实 GMV；若改版前 RMSPE < 5%，可信度高。

- **预期产出**：本地化改版对德国 GMV 的周度纯增量，换算独立站本地化建设（工程+运营约 50 万元）的 3 年 ROI。

- **业务价值**：估算值为 300-800% ROI（参考行业本地化收益均值），支撑向法国/意大利复制的扩张决策，年化潜在增量 GMV 500-1500 万元。

---

## ③ 代码模板

```python
"""
ClusterSC 母婴品牌地区级广告增量归因
依赖: numpy, pandas, scipy, scikit-learn
运行: python model.py
"""
from paper2skills_code.causal_inference.synthetic_control_2025.model import (
    ClusterSC, simulate_geo_data
)

# ── 1. 准备面板数据 ────────────────────────────────────────────────────────
# df: 宽表，行=地理单元，列=时间期
# donors: donor 单元 id 列表
# treated: 目标单元 id 列表（支持多目标）
# pre_cols: 干预前时间列名
# post_cols: 干预后时间列名

df, donors, treated, pre_cols, post_cols = simulate_geo_data(
    n_donors=500,       # 模拟 500 个供体县
    n_treated=1,        # 1 个目标单元（加州）
    n_pre=12,           # 12 个干预前时间期（月）
    n_post=3,           # 3 个干预后时间期（黑五前后）
    treatment_effect=20.0,  # 模拟真实效应 20 单位销量
    seed=42,
)

# ── 2. 拟合 ClusterSC ─────────────────────────────────────────────────────
model = ClusterSC(
    n_clusters=15,  # 建议 K = sqrt(n_donors) 附近
    random_state=42,
)
model.fit(df, donors, treated, pre_cols)

# ── 3. 验证干预前拟合质量 ──────────────────────────────────────────────────
pre_fit = model.pre_treatment_fit(df, treated, pre_cols)
print(f"干预前 RMSPE: {pre_fit['rmspe'].iloc[0]:.4f}")
# RMSPE < 5 为佳；若 > 10，考虑增大 n_clusters 或增加干预前期数

# ── 4. 估计处理效应 ATT ────────────────────────────────────────────────────
att_df = model.estimate_att(df, treated, post_cols)
print(att_df[["period", "actual", "counterfactual", "att"]])
print(f"平均月度增量: {att_df['att'].mean():.2f} 件")

# ── 5. 业务解读 ───────────────────────────────────────────────────────────
avg_att = att_df["att"].mean()
unit_price = 200      # 客单价（元）
ad_spend = 500_000    # 广告投入（元）
incremental_gmv = avg_att * 3 * unit_price   # 3 个月增量 GMV
roi = incremental_gmv / ad_spend
print(f"3 个月增量 GMV: {incremental_gmv:,.0f} 元")
print(f"广告 ROI: {roi:.2f}x")
```

**输入输出定义**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | `pd.DataFrame` | 宽表，行=单元，列=时间期 |
| `donors` | `list[str]` | 供体单元 id 列表 |
| `treated` | `list[str]` | 目标单元 id 列表 |
| `pre_cols` | `list[str]` | 干预前时间列名（至少 8 期） |
| `post_cols` | `list[str]` | 干预后时间列名 |

| 输出 | 类型 | 说明 |
|------|------|------|
| `att_df` | `pd.DataFrame` | 列：unit, period, actual, counterfactual, att |
| `pre_fit` | `pd.DataFrame` | 列：unit, rmspe（干预前拟合质量） |

---

## ④ 技能关联

### 前置技能
- [Skill-DiD-Difference-in-Differences](./[[Skill-DiD-Difference-in-Differences]].md) — DiD 是 SC 的特殊情形（双向固定效应），是方法学前置
- [Skill-Intelligent-Attribution-Causal-Forest](./[[Skill-Intelligent-Attribution-Causal-Forest]].md) — 地区级归因场景的替代方法，可对比使用

### 延伸技能
- [Skill-DML-Cohort-Causal-Effect](./[[Skill-DML-Cohort-Causal-Effect]].md) — DML 处理个体级异质性，ClusterSC 处理地理级宏观归因，互补覆盖
- [Skill-Causal-Discovery-PC-Algorithm](./[[Skill-Causal-Discovery-PC-Algorithm]].md) — 在供体特征选择阶段，PC 算法可用于识别真正的混杂变量

### 可组合
- [Skill-DiD-Difference-in-Differences](./[[Skill-DiD-Difference-in-Differences]].md) — 安慰剂检验（Placebo Test）可与 DiD 共用，交叉验证 SC 结论
- [Skill-MMM-Marketing-Mix-Modeling](../15-营销投放分析/[[Skill-Marketing-Mix-Modeling]].md) — MMM 给出宏观媒介分配建议；ClusterSC 提供地理级因果验证，两者形成闭环

---

## ⑤ 商业价值评估

### ROI 预估

**场景一（区域广告归因）**：
- 识别无效区域广告后，优化 50 万元投放方向
- 假设媒介效率提升 20%，年化节省 = 50 万 × 12 个月 × 20% = **120 万元/年**（保守）
- 若指导全年 500 万预算：**年化节省 100-500 万元；ROI ≈ 50-250 倍**（建模成本约 2 人月）

**场景二（本地化扩张决策）**：
- 验证德国本地化改版 300-800% ROI 后，推进法/意复制
- 增量拓展带来年化 GMV 500-1500 万元，**建模成本约 5-10 万元，ROI ≈ 50-150 倍**

### 实施难度：⭐⭐⭐☆☆ (3/5)

- **易**：依赖 sklearn + scipy 标准库，无需 econml 等额外安装；代码结构清晰，2 周可上线
- **中**：面板数据清洗（对齐时间序列、处理缺失）是主要工程成本
- **难**：供体池同质性验证（需要业务 domain knowledge 排除不可比供体）；SUTVA 假设的业务确认

### 优先级评分：⭐⭐⭐⭐☆ (4/5)

**评估依据**：
1. **填补关键方法空白**：现有 Skill 库缺少地理级/宏观级因果归因工具，ClusterSC 直接填补 01-因果推断领域中大规模供体池场景缺口
2. **有理论保证**：论文提供误差上界数学证明，非经验做法，可信度高
3. **业务场景直接匹配**：母婴品牌区域广告投放归因是高频痛点，ROI 核算需求刚性
4. **工程门槛适中**：不依赖商业软件或 GPU，2-3 名数据科学家 2-4 周可完成生产部署
5. **减 1 星原因**：供体池同质性验证需要较强业务 domain knowledge，自动化程度相对较低
