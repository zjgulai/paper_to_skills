---
name: AB-Experimental-Design
description: 基于 Zhou et al. (2023) 的 A/B 实验设计基础工具包，覆盖连续/二分类指标样本量计算、相对提升 Delta method、统计功效与 MDE 回验、分层随机分配及 CUPED 方差缩减，为跨境电商业务实验提供严谨统计支撑。
paper: "Zhou et al. (2023). All about Sample-Size Calculations for A/B Testing. CIKM."
area: 02-A_B实验
---

# Skill: A/B 实验设计基础

## 一、算法原理

### 核心思想

A/B 测试的统计严谨性建立在**样本量规划**、**功效保证**和**方差控制**三大支柱上。本技能基于 Zhou et al. (2023) 的系统性综述，将学术界的最佳实践封装为可直接调用的 Python 工具包，解决电商实验中最常见的四类问题：

1. **该测多少用户？** — 连续/二分类指标的样本量计算
2. **能不能检测到这个效应？** — Power 分析与 MDE 回验
3. **分组是否公平？** — 分层随机分配消除基线偏差
4. **能不能更快出结论？** — CUPED 利用实验前数据缩减方差

### 数学直觉

**1. 连续型指标样本量 (两样本 t 检验)**

$$
n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \sigma^2 \cdot (1 + 1/r)}{\delta^2}
$$

其中 $\delta$ 为最小可检测效应 (MDE)，$\sigma$ 为指标标准差，$r$ 为治疗组/控制组样本量比。

**2. 二分类指标样本量 (比例检验)**

使用合并比例 (pooled proportion) 估计方差：

$$
p_{\text{pool}} = \frac{p_c + p_t}{2}, \quad n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot p_{\text{pool}}(1-p_{\text{pool}}) \cdot (1 + 1/r)}{\delta^2}
$$

**3. 相对提升 (Relative Lift) 的 Delta Method 修正**

当业务关注相对提升 $\delta_{\text{rel}} = (p_t - p_c) / p_c$ 时，分母中的控制组比例引入额外方差。Zhou et al. (2023) 提出使用 Delta method 调整方差因子：

$$
n_{\text{rel}} \approx \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot p_{\text{pool}}(1-p_{\text{pool}}) \cdot (1/p_c^2 + p_t^2/p_c^4)}{\delta_{\text{rel}}^2}
$$

相对提升设计通常比绝对提升**更保守**（需要更大样本量）。

**4. 统计功效 (Power) 与 MDE**

给定每组样本量 $n_t$，功效计算公式：

$$
\text{Power} = 1 - \Phi\left( Z_{1-\alpha/2} - \frac{\delta \sqrt{n_t}}{\sigma \sqrt{1+1/r}} \right)
$$

MDE（最小可检测效应）则是功效公式的反解：

$$
\text{MDE} = \frac{(Z_{1-\alpha/2} + Z_{1-\beta}) \cdot \sigma \cdot \sqrt{1+1/r}}{\sqrt{n_t}}
$$

**5. CUPED 方差缩减**

CUPED (Controlled-experiment Using Pre-Experiment Data) 通过实验前的同一指标 $x$ 对结果 $y$ 做线性调整：

$$
\theta = \frac{\text{Cov}(y, x)}{\text{Var}(x)}, \quad y_{\text{cuped}} = y - \theta \cdot (x - \bar{x})
$$

当历史指标与未来指标相关性高时，CUPED 可将方差降低 20%-50%，等效于样本量缩减 1.25-2 倍。

### 关键假设

- 样本独立同分布，治疗效果恒定 (ATE 框架)
- 对于二分类指标，正态近似在样本量足够大时成立
- CUPED 要求实验前协变量与实验结果存在稳定相关性
- 分层分配假设各 stratum 的处理概率恒定

---

## 二、业务应用

### 场景 1: Momcozy 落地页转化率优化实验

Momcozy 在 Amazon 详情页测试新版主图，基线转化率 2.5%，期望通过改版实现 10% 相对提升（即达到 2.75%）。

**应用流程**：
1. **样本量计算**：使用 `sample_size_binary` 计算绝对提升所需样本量，再用 `sample_size_relative_lift` 按相对提升做保守修正
2. **实验计划**：输入日流量（如 2000 UV/天），自动计算所需实验天数
3. **分层分配**：按 `国家 × 设备类型 × 新老用户` 做分层随机化，确保各层内治疗/控制组 1:1 平衡
4. **CUPED 加速**：利用实验前各用户的转化率作为协变量，缩减方差后提前 30% 时间得出结论

**预期效果**：
- 避免"样本量不足导致假阴性"（Type II 错误）
- 分层设计消除国家/设备差异带来的基线不平衡
- CUPED 在保持统计功效的前提下缩短实验周期

### 场景 2: 大促前快速评估实验可行性

季度大促前只有 3 周窗口期，需要快速判断某个功能改动是否值得上线测试。

**应用流程**：
1. 使用 `mde_calculator`：输入可用样本量（如每组 50,000），计算在该样本下能稳定检测到的最小效应
2. 与业务预期的提升幅度对比：如果 MDE > 预期提升，则实验无法给出有效结论，建议扩大流量或降低预期
3. 使用 Power 查询表快速评估不同相对提升下的检测概率

**决策价值**：
- 在投入研发资源前，用 5 分钟判断实验是否"测得出来"
- 避免浪费 3 周窗口期在"注定无法显著"的实验上

---

## 三、代码模板

### 文件结构

```
paper2skills-code/ab_testing/experimental_design/
├── __init__.py
└── design.py
```

### 核心模块说明

- `sample_size_continuous` / `sample_size_binary` / `sample_size_relative_lift`：三类指标的样本量计算
- `power_analysis` / `mde_calculator`：功效与最小可检测效应互算
- `stratified_allocation`：支持多维度分层随机分配
- `cuped_adjustment`：CUPED 方差缩减，输出原始提升、调整后提升、方差缩减比例
- `ABTestDesigner`：统一封装，输入业务参数直接输出完整实验计划
- `generate_momcozy_users`：Momcozy 母婴电商合成数据生成器

### 运行方式

```bash
cd paper2skills-code/ab_testing/experimental_design
python3 design.py
```

### 示例输出

```
============================================================
A/B 实验设计基础 - Momcozy 母婴电商场景演示
============================================================

【场景 1】落地页转化率优化实验设计
  基线转化率: 2.50%
  目标相对提升: 10%
  绝对提升 MDE: 0.0025
  按绝对提升计算样本量: 每组 64,200 人
  按相对提升计算样本量: 每组 70,941 人
  -> 相对提升设计更保守, 建议采用每组 70,941 人

  实验计划:
    控制组: 64,200 人
    治疗组: 64,200 人
    总计:   128,400 人
    日流量: 2,000 人/天
    预计时长: 65 天
    回验 Power: 80.00%

【场景 2】分层随机分配 (按国家 + 设备类型 + 新老用户)
  各分层内分配比例 (前 6 组):
assignment                         C      T
country device_type user_type              
CA      desktop     new        0.491  0.509
                    returning  0.551  0.449
        mobile      new        0.469  0.531
                    returning  0.518  0.482
DE      desktop     new        0.491  0.509
                    returning  0.507  0.493

【场景 3】CUPED 方差缩减演示
  原始估计提升量: 0.0019
  CUPED 调整后提升量: 0.0019
  方差缩减比例: 29.3%
  -> 等效于样本量缩减至原来的 1.4 倍

【场景 4】Power / MDE 快速查询表
  假设基线转化率 2.5%, 每组 50,000 用户:
    相对提升   5% -> Power = 23.9%
    相对提升   8% -> Power = 51.1%
    相对提升  10% -> Power = 69.6%
    相对提升  15% -> Power = 95.6%
```

---

## 四、技能关联

### 前置技能
- **Demand Forecasting**（时间序列需求预测）
  - 为 CUPED 提供实验前的历史指标基线，如历史转化率、历史 AOV
- **Uplift Modeling**（因果推断/ uplift 建模）
  - A/B 实验是 Uplift Modeling 获取训练数据的黄金标准来源；本技能确保实验设计的统计严谨性

### 延伸技能
- **Thompson Sampling / Multi-Armed Bandit**
  - MAB 和 Thompson Sampling 是在线探索优化算法，本技能提供其前置所需的实验设计基础（Power、MDE、样本量）
- **TJAP 跨市场品类组合定价**
  - 定价实验需要严谨的 A/B 测试设计来验证不同价格策略的真实因果效应

### 组合推荐
- `A/B Experimental Design + CUPED + Uplift Modeling`：构成"严谨实验 → 方差缩减 → 因果效应估计"的完整因果推断链路
- `A/B Experimental Design + Thompson Sampling`：离线实验设计保障基础统计能力，在线 MAB 动态优化探索效率

---

## 五、业务价值评估

### 价值量化
- **避免假阴性损失**：正确的样本量计算确保 80% 功效，避免"真实有效策略被误判为无效"而导致的收益损失
- **缩短实验周期**：CUPED 方差缩减 20%-50%，等效于将实验时间缩短 20%-50%，在 2000 UV/天的场景下，65 天的实验可缩短至 45-52 天
- **提升决策置信度**：分层随机化消除基线不平衡争议，减少实验结果的政治化解读

### 实施难度
⭐⭐⭐（3/5）
- 统计学概念（Power、MDE、Pooled Proportion）需要一定理解成本
- 代码本身即插即用，无需复杂基础设施
- CUPED 需要历史数据可用，数据工程成本取决于现有数据仓库成熟度

### 优先级评分
⭐⭐⭐⭐⭐（5/5）
- 所有后续 A/B 实验技能（MAB、Thompson Sampling、Uplift Modeling）的前置基石
- 直接填补 paper2skills 图谱中 02-A/B实验 领域的核心缺口
- 业务价值即时可量化，任何有流量实验的团队都能直接落地

---

## 六、论文基准

Zhou et al. (2023) 在论文中系统比较了多种样本量计算方法的偏差：

| 方法 | 相对偏差 (连续型) | 相对偏差 (二分类) |
|------|------------------|------------------|
| 正态近似 (本技能采用) | < 1% (n > 1000) | < 2% (p > 1%) |
| 忽略合并比例 | — | 5%-15% |
| 忽略 Delta method (相对提升) | — | 10%-30% |

本技能采用的公式在电商常见样本规模下（每组 > 10,000，转化率 > 1%）与精确检验的差异可忽略，同时保持计算简洁性。
