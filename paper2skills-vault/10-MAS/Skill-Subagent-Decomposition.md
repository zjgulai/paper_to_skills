---
title: Subagent Decomposer — 复杂任务子智能体分解
doc_type: knowledge
module: 10-MAS
topic: subagent-decomposition
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill: Subagent Decomposer — 复杂任务子智能体分解

---

## ① 算法原理

### 核心思想

**Subagent Decomposer** 负责将复杂的 Task Blueprint 分解为可独立执行的子任务，并分配给专门的子 Agent。核心洞察：**复杂任务（如"生成全品类 VOC 周报"）无法由单个 Agent 高效完成，必须分解为并行/串行的子任务，每个子任务由最优技能的子 Agent 执行**。

分解的三个层次：

1. **横向分解（Parallel Decomposition）**：按数据维度拆分，子任务间无依赖，可并行执行
   - 例：按品类拆分（吸奶器/储奶袋/推车分别分析）

2. **纵向分解（Sequential Decomposition）**：按处理阶段拆分，子任务间有依赖，必须串行
   - 例：抽取 → 清洗 → 分析 → 报告

3. **混合分解（Hybrid Decomposition）**：横向 + 纵向结合，形成 DAG（有向无环图）
   - 例：先按品类并行抽取，再汇总做跨品类对比分析

### 分解策略

**基于 Skill Registry 的分解**：

```
Task Blueprint
    ↓
[分解器] — 分析任务类型和数据维度
    ↓
查询 Skill Registry:
  - 有哪些技能可以处理这个任务？
  - 每个技能的处理范围是什么？
    ↓
构建子任务 DAG:
  节点 = 子任务（技能 + 数据子集）
  边 = 依赖关系（数据依赖 / 控制依赖）
    ↓
调度执行:
  无依赖的节点并行启动
  有依赖的节点等待前置完成
```

### 关键假设

1. **任务可分解**：复杂任务可以拆分为独立的子任务
2. **子任务可分配**：每个子任务有对应的最优技能
3. **依赖可静态分析**：子任务间的依赖关系在分解时可确定
4. **并行有收益**：并行执行的总时间 < 串行执行的总时间

---

## ② 母婴出海应用案例

### 场景一：全品类 VOC 周报生成

**业务问题**：

生成一份覆盖全品类的 VOC 周报需要：抽取所有品类的评论实体/情感、汇总趋势、生成洞察、输出报告。数据量大（36万+评论），单 Agent 处理太慢。

**数据要求**：

- 按品类分区的评论数据
- 各品类的分析技能（抽取/情感/汇总）
- 报告模板

**预期产出**：

```
输入 Task:
  "生成本周全品类 VOC 周报（覆盖 8 大品类）"

Subagent Decomposer 分解:

  横向分解（按品类并行）:
    ├─ Subtask 1: 吸奶器品类分析
    ├─ Subtask 2: 储奶袋品类分析
    ├─ Subtask 3: 温奶器品类分析
    ├─ Subtask 4: 推车品类分析
    ├─ Subtask 5: 安全座椅品类分析
    ├─ Subtask 6: 洗护用品品类分析
    ├─ Subtask 7: 喂养 accessories 分析
    └─ Subtask 8: 其他品类分析

  每个子任务内部纵向分解:
    Step 1: InstructUIE — 抽取实体/关系/事件
    Step 2: ABSA — 方面级情感分析
    Step 3: TrendAnalyzer — 趋势变化检测

  汇总阶段（依赖所有子任务完成）:
    Step 4: CrossCategoryAnalyzer — 跨品类对比
    Step 5: ReportGenerator — 生成 Markdown 报告

执行 DAG:
  [T1..T8 并行执行] ──→ [CrossCategory] ──→ [ReportGen]

执行时间:
  串行: ~8h (单 Agent 处理全部)
  并行: ~1.2h (8 个子 Agent 并行 + 汇总 0.5h)
  加速比: 6.7x
```

**业务价值**：
- 周报生成从 8 小时缩短到 1.2 小时
- 各品类分析质量一致（专用 Agent 处理专用数据）
- 支持增量更新（只需重新分析变化的部分）

---

### 场景二：竞品深度对标分析

**业务问题**：

深度对标分析涉及多维度（价格、功能、用户评价、市场份额、渠道分布），每个维度需要不同的数据源和分析方法。

**数据要求**：

- 竞品产品数据（多平台）
- 用户评价数据
- 市场价格数据
- 渠道/市场份额数据

**预期产出**：

```
输入 Task:
  "深度对标 Spectra S1 vs Medela vs Elvie（全维度）"

分解结果:

  横向分解（按维度并行）:
    ├─ Subtask A: 产品规格对比
    │   技能: ProductSpecMatcher
    │   数据: 官方规格表
    │
    ├─ Subtask B: 用户评价对比
    │   技能: ReviewComparator (InstructUIE + ABSA)
    │   数据: Amazon/Trustpilot 评论
    │
    ├─ Subtask C: 价格竞争力分析
    │   技能: PriceAnalyzer
    │   数据: 多平台价格
    │
    └─ Subtask D: 市场份额估算
        技能: MarketShareEstimator
        数据: 销售排名/评论数/搜索量

  汇总阶段:
    Step E: SWOTAnalyzer — 综合各维度生成 SWOT
    Step F: StrategyRecommender — 输出策略建议

依赖关系:
  A ──┐
  B ──┼──→ E ──→ F
  C ──┘
  D ──┘
```

**业务价值**：
- 深度分析从 3-5 天缩短到 2-4 小时
- 各维度分析专业深度一致
- 分析框架可复用（换竞品只需替换数据）

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/subagent_decomposer/subagent_decomposer.py`

核心组件：
- `Subtask`: 子任务（ID、技能、数据范围、依赖、优先级）
- `TaskDAG`: 任务依赖图（节点=子任务，边=依赖关系）
- `Decomposer`: 分解器主类
  - `decompose_parallel`: 横向分解
  - `decompose_sequential`: 纵向分解
  - `decompose_hybrid`: 混合分解
  - `build_dag`: 构建依赖图
  - `get_execution_order`: 拓扑排序获取执行顺序
- `SubagentPool`: 子 Agent 池（管理并行执行）

运行方式：
```bash
cd paper2skills-code/mas/subagent_decomposer
python subagent_decomposer.py
```

生产环境建议：
1. 使用 Temporal / Celery 作为底层任务调度引擎
2. 实现子任务容错（失败重试、降级策略）
3. 监控 DAG 执行状态（哪些节点完成/失败/进行中）
4. 支持动态调整（执行中发现某子任务可进一步分解）
5. 结果缓存（相同输入的子任务直接复用结果）

---

## ④ 技能关联

### 前置技能
- **Skill Registry**：提供可用技能列表和依赖信息
- **DAG / 图论**：理解拓扑排序、关键路径

### 延伸技能
- **MapReduce**：大规模数据并行处理范式
- **Celery / Temporal**：分布式任务队列和工作流编排
- **Ray**：分布式计算框架

### 可组合技能
- **Skill Registry**：Decomposer 从 Registry 查询技能能力边界
- **MAS Orchestrator**：Orchestrator 按 DAG 调度子任务执行
- **ToT**：复杂分解策略本身可以用 ToT 搜索最优分解方式
- **AutoGen**：每个子任务对应一个 AutoGen Agent

---


- **可组合**：[[Skill-MAS-Orchestrator]] / [[Skill-ReAct-Reasoning-Acting]]

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 全品类周报生成 | 时间从 8h → 1.2h，加速 6.7x | 开发 2-3 周 | 15-25x |
| 竞品深度分析 | 时间从 3-5d → 2-4h | 开发 2 周 | 12-18x |
| 大规模数据处理 | 支持 10x 数据量而不增加时间 | 开发 2-3 周 | 10-15x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：中，需要任务定义和技能边界信息
- 技术门槛：中高，需要理解 DAG、拓扑排序、分布式执行
- 工程复杂度：高，涉及并行调度、容错、状态管理
- 维护成本：中，分解策略需要随业务变化调整

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **性能核心**：并行分解是 MAS 处理大规模任务的性能基础
- **可扩展性**：系统能力随子 Agent 数量线性扩展
- **通用模式**：MapReduce 模式已被业界广泛验证
- **业务刚需**：36万+评论的全量分析必须并行化

---

## 参考设计

1. **MapReduce: Simplified Data Processing on Large Clusters** (OSDI 2004)
   - Dean, J. & Ghemawat, S. (Google)
   - 核心思想：横向分解 + 并行 Map + 汇总 Reduce

2. **Temporal: Durable Execution**
   - 核心思想：工作流即代码，任务持久化、可重放、可观测

3. **Apache Airflow DAG Design**
   - 核心思想：有向无环图定义任务依赖，调度器按拓扑排序执行

---

## 在 MAS 工作流中的位置

```
[Task Blueprint]
    ↓
[Skill Registry] — 发现可用技能
    ↓
[Subagent Decomposer] ← 当前技能
    ├─ 横向分解: 按数据维度并行
    ├─ 纵向分解: 按处理阶段串行
    └─ 混合分解: DAG
    ↓
[执行 DAG]
    ↓
[MAS Orchestrator] — 调度执行
    ↓
[Subagent 1] [Subagent 2] ... [Subagent N] (并行)
    ↓
[结果汇总]
```
