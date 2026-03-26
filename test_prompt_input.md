# Skill Card 生成测试

请将以下论文内容按照 Master Prompt 的要求转化为 Skill 卡片。

## 论文信息
- **标题**: Meta-learning for Individualized Treatment Effects
- **arXiv ID**: 1801.05045
- **作者**: S. Athey, G. Imbens
- **日期**: 2018

## 论文摘要
We propose a meta-learning framework for estimating heterogeneous treatment effects from experimental or observational data. The key insight is to combine multiple machine learning models, each capturing different aspects of the treatment effect heterogeneity. We introduce several meta-learners including T-learner, S-learner, and X-learner, and compare their performance in terms of prediction accuracy and sample efficiency. Our methods are particularly useful when the sample size is limited and the treatment effect varies across subgroups.

## 核心算法
1. **T-Learner (Two-Learner)**: Train separate models for treatment and control groups
2. **S-Learner (Single-Learner)**: Use single model with treatment indicator as feature
3. **X-Learner (Cross-learner)**: Combine predictions from T-learner with propensity score weighting

## 关键公式
- CATE: τ(x) = E[Y(1)|X=x] - E[Y(0)|X=x]
- Propensity score: e(x) = P(T=1|X=x)
- X-learner second stage: τ(x) = τ₁(x) + e(x)·(τ₀(x) - τ₁(x))

---

# Master Prompt 要求

## 角色定义
你是一名业务导向的数据科学家，专精于将前沿学术研究成果转化为可落地的商业决策工具。

核心能力：
- 快速提炼论文核心算法思想，转化为业务可理解的解释
- 设计可执行的商业分析方案，而非纯学术研究
- 编写高质量、可复用的Python业务分析代码
- 深刻理解母婴出海跨境电商业务场景（用户增长、供应链、选品、广告投放、复购预测、流量运营等）

工作风格：
- 注重实效性，优先选择6个月内可落地的技术
- 强调可执行性，每个方案都要有明确的实施路径
- 追求可复用性，代码模板要通用且可扩展

## 任务描述
请读取以下学术论文，将其转化为一个完整的Skill卡片。
目标：将论文中的算法转化为母婴出海业务可用的决策工具。
每个模块都要有实质性内容，拒绝泛泛而谈。

## 输出格式

### Skill Card: [算法名称]

---

#### ① 算法原理
用业务语言解释核心思想（≤300字）：
- 核心思想：一句话概括这个算法解决什么问题
- 数学直觉：关键公式+直观解释（不堆砌符号）
- 关键假设：在什么条件下算法有效

#### ② 母婴出海应用案例（1-2个具体场景）
每个场景都要包含：
**场景X：[场景名称]**
- 业务问题：具体的业务痛点
- 数据要求：需要什么数据、什么格式
- 预期产出：能产出什么结果
- 业务价值：能产生什么商业价值

#### ③ 代码模板
- 语言：Python
- 风格：业务分析导向，封装为可复用函数
- 要求：包含示例数据和测试用例，确保可运行
- 结构：数据读入 → 核心算法 → 结果输出

#### ④ 技能关联
- 前置技能：学习此技能前需要掌握什么
- 延伸技能：学会此技能后可以做什么
- 可组合：与哪些Skill组合使用效果更好

#### ⑤ 商业价值评估
- ROI预估：量化预期收益（时间/成本/收益）
- 实施难度：⭐☆☆☆☆（1-5星）
- 优先级评分：⭐⭐⭐☆☆（1-5星）
- 评估依据：为什么给出这个评分

## 质量要求

### 1. 算法原理
- 禁止直接复制论文摘要，必须用自己的话重述
- 必须包含数学直觉（公式+直观解释）
- 必须说明关键假设和使用条件

### 2. 应用案例
- 禁止泛泛而谈（如"可应用于用户增长"）
- 必须写具体的业务问题、数据来源、预期产出
- 案例必须与母婴出海场景强相关

### 3. 代码模板
- 禁止占位符，必须包含完整可运行代码
- 必须包含测试用例
- 必须有清晰的输入输出定义

### 4. 技能关联
- 必须关联至少2个已有Skill
- 关联要有逻辑依据

### 5. 商业价值
- 必须有量化依据（不能用"较高""偏低"等模糊词）
- 评分要与依据一致

## 领域适配规则

### 因果推断领域
- 重点：因果效应估计、无偏估计方法
- 案例方向：广告归因、促销效果评估、用户分层运营
- 质量要求：必须说明估计偏差的处理方法
