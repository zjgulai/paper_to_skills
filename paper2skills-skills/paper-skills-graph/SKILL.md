---
name: paper-skills-graph
description: This skill should be used when the user asks to "分析技能图谱", "skills graph 选题", "发现知识缺口", "推荐新选题", "skill 关联分析", "知识体系分析". Analyzes relationships between existing skills to identify knowledge gaps and recommend new paper topics.
version: 0.1.0
---

# paper-skills-graph

基于已有 Skill 卡片的关系图谱，分析知识缺口并推荐新论文选题。

## 概述

Skills Graph 是一个动态的知识图谱系统，通过解析现有 Skill 卡片中的技能关系（前置、延伸、可组合），构建技能间的依赖网络，从而：

1. **发现知识缺口** - 识别缺失的前置技能或孤立的技能节点
2. **推荐延伸方向** - 基于现有技能推荐自然的延伸学习路径
3. **优化选题策略** - 优先选择能填补关键知识空白的论文
4. **发现组合机会** - 识别可以组合多个技能的创新应用场景

## 触发方式

用户提及以下内容时触发：
- "分析技能图谱" 或 "skills graph"
- "发现知识缺口" 或 "知识缺口分析"
- "推荐新选题" 或 "基于图谱推荐选题"
- "skill 关联分析" 或 "技能关系分析"
- "知识体系分析" 或 "知识网络"
- "下一步该学什么" 或 "学习路径推荐"

## Skills Graph 构建方法

### 节点定义

每个 Skill 卡片是一个图谱节点，包含以下属性：

```yaml
skill_node:
  id: Skill-Name
  domain: 所属领域（因果推断/A_B实验/时间序列/供应链/推荐系统/增长模型/NLP-VOC）
  difficulty: 实施难度（1-5星）
  business_value: 业务价值（1-5星）
  status: 状态（已完成/进行中/计划中）
```

### 边定义

从 Skill 卡片的「技能关联」部分解析三种关系：

```yaml
edges:
  - type: prerequisite        # 前置技能（依赖）
    from: 当前技能
    to: 前置技能
    weight: 1.0

  - type: extension          # 延伸技能（进阶）
    from: 当前技能
    to: 延伸技能
    weight: 0.8

  - type: combinable         # 可组合技能（协同）
    from: 技能A
    to: 技能B
    weight: 0.6
```

### 图谱构建步骤

1. **扫描所有 Skill 卡片**
   ```bash
   # 扫描路径
   paper2skills-vault/*/Skill-*.md
   ```

2. **解析技能关系**
   - 读取每个 Skill 的「④ 技能关联」部分
   - 提取前置技能、延伸技能、可组合技能列表
   - 构建有向图（前置和延伸为有向边，组合为无向边）

3. **计算节点指标**
   - **入度** (In-degree): 有多少技能依赖此技能（基础重要性）
   - **出度** (Out-degree): 此技能延伸出多少技能（发展潜力）
   - **PageRank**: 技能在知识网络中的中心性
   - **聚类系数**: 技能所在社区的紧密程度

## 知识缺口分析方法

### 缺口类型定义

| 缺口类型 | 定义 | 识别方法 | 优先级 |
|---------|------|---------|--------|
| **前置缺口** | 某技能的前置技能不存在 | 边指向不存在的节点 | 高 |
| **延伸缺口** | 高价值技能无延伸技能 | 高业务价值节点出度=0 | 高 |
| **孤岛技能** | 无任何关联的孤立技能 | 度=0 | 中 |
| **桥梁缺口** | 两个社区间缺少连接 | 社区检测发现断开的模块 | 中 |
| **深度缺口** | 某领域的技能链过短 | 领域内最长路径 < 3 | 低 |

### 缺口分析算法

```python
def analyze_knowledge_gaps(skills_graph):
    gaps = []

    # 1. 前置缺口检测
    for edge in skills_graph.edges:
        if edge.type == 'prerequisite' and edge.target not in skills_graph.nodes:
            gaps.append({
                'type': 'missing_prerequisite',
                'skill': edge.source,
                'missing': edge.target,
                'priority': 'high'
            })

    # 2. 延伸缺口检测
    for node in skills_graph.nodes:
        if node.business_value >= 4 and node.out_degree == 0:
            gaps.append({
                'type': 'missing_extension',
                'skill': node.id,
                'priority': 'high'
            })

    # 3. 社区桥接缺口
    communities = detect_communities(skills_graph)
    for comm_a, comm_b in combinations(communities, 2):
        bridges = find_bridges(comm_a, comm_b, skills_graph)
        if len(bridges) == 0:
            gaps.append({
                'type': 'missing_bridge',
                'between': [comm_a.domain, comm_b.domain],
                'priority': 'medium'
            })

    return gaps
```

## 选题推荐逻辑

### 推荐策略

基于缺口类型，采用不同的选题策略：

#### 1. 前置缺口 → 基础技能优先

当发现某技能缺少前置基础时：
- **推荐方向**: 寻找该前置技能的基础论文
- **搜索关键词**: "基础理论 + 方法论"
- **筛选标准**: 优先选择教程式 (tutorial) 或基础原理论文

**示例**:
- 缺口: `Doubly Robust Estimation` 缺少前置 `Propensity Score Matching`
- 推荐: 搜索 `propensity score matching tutorial`

#### 2. 延伸缺口 → 前沿应用优先

当发现高价值技能无延伸时：
- **推荐方向**: 该技能的前沿应用或变种
- **搜索关键词**: "原技能 + 新应用场景 + 最新年份"
- **筛选标准**: 近2年论文，有实验验证

**示例**:
- 缺口: `Uplift Modeling` 缺少供应链领域的应用
- 推荐: 搜索 `uplift modeling inventory optimization`

#### 3. 桥梁缺口 → 跨领域融合优先

当发现两个领域缺少连接时：
- **推荐方向**: 跨领域方法迁移
- **搜索关键词**: "领域A方法 + 领域B问题"
- **筛选标准**: 有实际业务场景，可落地性强

**示例**:
- 缺口: 因果推断 与 推荐系统 之间桥梁不足
- 推荐: 搜索 `causal inference recommendation system debiased`

#### 4. 组合缺口 → 综合方案优先

当发现多个技能可以组合但无综合方案时：
- **推荐方向**: 多技能融合的系统方案
- **搜索关键词**: "技能A + 技能B + framework/system"
- **筛选标准**: 端到端解决方案，工程可落地

**示例**:
- 缺口: 缺乏 `Uplift Modeling + LTV预测` 的组合方案
- 推荐: 搜索 `uplift modeling customer lifetime value optimization`

### 选题评分模型

对每个候选选题进行综合评分：

```python
def score_topic_candidate(candidate, gaps, graph):
    score = 0

    # 缺口匹配度 (40%)
    if matches_gap(candidate, gaps):
        score += 40 * gap_priority_weight(gaps)

    # 业务价值 (30%)
    score += 30 * estimate_business_value(candidate)

    # 可落地性 (20%)
    score += 20 * estimate_feasibility(candidate)

    # 网络增益 (10%)
    # 新增此技能后，图谱连通性的提升程度
    score += 10 * estimate_network_gain(candidate, graph)

    return score
```

## 输出格式

### 图谱分析报告

```
# Skills Graph 分析报告

## 1. 图谱概览

- **节点总数**: X 个技能
- **边总数**: Y 条关系
- **领域分布**:
  - 因果推断: X 个
  - A/B实验: X 个
  - ...

## 2. 中心性分析

### 核心基础技能 (高入度)
| 排名 | 技能 | 入度 | 被依赖数 |
|-----|------|------|---------|
| 1 | 基础统计推断 | 5 | 被 5 个技能依赖 |
| 2 | ... | ... | ... |

### 潜力延伸技能 (高出度且高价值)
| 排名 | 技能 | 出度 | 业务价值 | 推荐延伸方向 |
|-----|------|------|---------|------------|
| 1 | Uplift Modeling | 0 | ⭐⭐⭐⭐⭐ | 因果森林、动态定价应用 |
| 2 | ... | ... | ... | ... |

## 3. 知识缺口

### 🔴 高优先级缺口

#### 缺口 1: 前置技能缺失
- **影响技能**: Skill-A
- **缺失前置**: Skill-B (Propensity Score Matching)
- **推荐选题**:
  - 论文 1: [标题] - arXiv:xxxx - 匹配度: 85%
  - 论文 2: [标题] - arXiv:xxxx - 匹配度: 78%

#### 缺口 2: 高价值技能无延伸
- **技能**: Uplift Modeling
- **业务价值**: ⭐⭐⭐⭐⭐
- **推荐延伸方向**:
  - 方向 1: Uplift + 动态定价 (搜索: `uplift dynamic pricing`)
  - 方向 2: Uplift + 推荐去偏 (搜索: `uplift debiased recommendation`)

### 🟡 中优先级缺口

#### 缺口 3: 领域间桥梁缺失
- **领域 A**: 因果推断
- **领域 B**: 推荐系统
- **推荐跨领域选题**:
  - 关键词: `causal recommendation`, `debiased recommendation`
  - 业务场景: 消除推荐系统的选择偏差

## 4. 推荐选题列表

| 优先级 | 选题 | 类型 | 缺口匹配 | 业务价值 | 搜索关键词 |
|-------|------|------|---------|---------|-----------|
| P0 | Propensity Score Matching 基础 | 前置填补 | 100% | ⭐⭐⭐⭐ | `propensity score tutorial` |
| P0 | Causal Forest for Uplift | 延伸拓展 | 95% | ⭐⭐⭐⭐⭐ | `causal forest uplift` |
| P1 | Causal Recommendation | 跨领域 | 80% | ⭐⭐⭐⭐⭐ | `causal recommendation debiased` |
| ... | ... | ... | ... | ... | ... |

## 5. 学习路径推荐

基于当前技能掌握情况，推荐学习顺序：

```
基础统计推断 → 倾向评分分析 → Uplift Modeling → 因果森林
                                    ↓
                              动态定价应用
```

## 6. 行动建议

1. **立即行动**: 优先填补高价值技能的前置缺口
2. **本周计划**: 基于延伸缺口搜索 3-5 篇候选论文
3. **本月目标**: 建立跨领域桥梁，完成 1 个跨领域 skill
```

## 使用示例

### 示例 1: 完整图谱分析

用户: "分析一下我们的技能图谱，看看有什么知识缺口"

执行:
1. 扫描所有 Skill-*.md 文件
2. 构建技能关系图谱
3. 运行缺口分析算法
4. 输出完整的图谱分析报告

### 示例 2: 特定技能延伸推荐

用户: "Uplift Modeling 技能接下来应该延伸什么方向？"

执行:
1. 定位 `Skill-Uplift-Modeling.md`
2. 分析其当前延伸技能列表
3. 基于其业务价值（⭐⭐⭐⭐⭐）推荐未开发的延伸方向
4. 输出延伸选题建议

### 示例 3: 跨领域发现

用户: "因果推断和推荐系统可以怎么结合？"

执行:
1. 分别提取两个领域的技能节点
2. 分析领域间的现有连接（边）
3. 识别缺失的桥梁连接
4. 推荐跨领域论文选题

## 与 paper-选题 的协作

`paper-skills-graph` 与 `paper-选题` 是互补关系：

| Skill | 触发场景 | 输出 |
|-------|---------|------|
| `paper-选题` | 用户有明确的业务需求或关键词 | 基于关键词的论文列表 |
| `paper-skills-graph` | 用户想系统性完善知识体系 | 基于知识缺口的选题推荐 |

**协作流程**:

```
用户: "我想系统性完善我们的技能体系"

→ paper-skills-graph: 分析图谱，发现缺口
  → 输出: 推荐填补缺口的关键词和方向

→ paper-选题: 基于推荐的关键词搜索论文
  → 输出: 具体的论文列表

→ paper-萃取: 将选中论文转化为 skill
  → 输出: 新的 Skill 卡片

→ paper-skills-graph: 更新图谱，验证缺口填补
  → 输出: 更新后的图谱状态
```

## 注意事项

- **定期更新**: 每次新增 skill 后，建议重新运行图谱分析
- **动态调整**: 业务优先级变化时，调整缺口优先级权重
- **人工判断**: 算法推荐的缺口需结合业务实际情况人工确认
- **避免过度连接**: 不是所有技能都需要强关联，保留合理的模块边界

## 演进计划

### Round 1: 基础图谱分析
- 解析技能关系，构建静态图谱
- 基础缺口检测算法

### Round 2: 智能推荐增强
- 集成 ArXiv API，自动搜索推荐选题
- 引入 LLM 辅助缺口描述和选题建议生成

### Round 3: 动态知识管理
- 追踪技能学习进度
- 基于业务目标动态调整推荐优先级
