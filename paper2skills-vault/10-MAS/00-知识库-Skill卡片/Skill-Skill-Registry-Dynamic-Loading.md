---
title: Skill Registry — 技能注册表与动态发现
doc_type: knowledge
module: 10-MAS
topic: skill-registry-dynamic-loading
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
---

# Skill: Skill Registry — 技能注册表与动态发现

---

## ① 算法原理

### 核心思想

**Skill Registry** 是 MAS 工作流的核心基础设施，负责管理所有可用技能的元数据、依赖关系和运行时状态。核心洞察：**一个可扩展的多 Agent 系统必须能动态发现、加载和组合技能，而不是硬编码固定流程**。

Skill Registry 的四个核心能力：

1. **技能注册（Registration）**：技能以声明式方式注册，包含元数据（名称、版本、输入/输出 Schema、依赖、质量阈值）
2. **技能发现（Discovery）**：根据 Task Blueprint 的需求，动态匹配最合适的技能组合
3. **依赖解析（Dependency Resolution）**：解析技能间的依赖关系，确保执行顺序正确
4. **版本管理（Versioning）**：支持技能的多版本共存和兼容性检查

### 架构设计

```
Task Blueprint 到达
    ↓
[需求解析] — 提取 required_skills + constraints
    ↓
[Skill Registry] ← 当前技能
    ├─ 查询: 哪些技能满足需求？
    ├─ 解析: 技能间依赖关系
    ├─ 验证: 输入/输出 Schema 兼容性
    └─ 排序: 拓扑排序确定执行顺序
    ↓
[执行计划] — 有序的技能调用序列
```

### 关键假设

1. **技能可声明式描述**：每个技能都能用元数据完整描述其能力
2. **需求可精确表达**：Task Blueprint 能清晰表达所需技能类型
3. **依赖可静态解析**：技能间的依赖关系在运行前可确定
4. **Schema 可兼容检查**：技能的输入/输出可以通过 Schema 验证兼容性

---

## ② 母婴出海应用案例

### 场景一：Task Blueprint 动态技能匹配

**业务问题**：

不同分析任务需要不同的技能组合。例如"抽取实体和情感"需要 InstructUIE + ABSA，"构建用户画像"需要 PERSONABOT + SoMeR。系统需要根据 Task Blueprint 自动匹配最合适的技能，而不是人工配置。

**数据要求**：

- Skill Registry 中注册的所有技能元数据
- Task Blueprint（任务类型、输入/输出 Schema、质量阈值）

**预期产出**：

```
输入 Task Blueprint:
  task_type: "EXTRACT"
  description: "抽取本周吸奶器评论中的实体和情感"
  required_skills: ["entity_extraction", "sentiment_analysis"]
  input_schema: {type: "raw_text", format: "string"}
  output_schema: {type: "structured", format: "json"}
  quality_threshold: 0.85

Skill Registry 匹配过程:
  1. 查询 "entity_extraction" 技能:
     - 候选1: InstructUIE (F1: 0.91, 支持多语言) ← 匹配
     - 候选2: BERT-CRF (F1: 0.87, 仅英文)
     - 候选3: LLM-ZeroShot (F1: 0.82, 通用)

  2. 查询 "sentiment_analysis" 技能:
     - 候选1: ABSA-BERT-MoE (F1: 0.89, 方面级情感) ← 匹配
     - 候选2: TextBlob (F1: 0.72, 文档级情感)

  3. 依赖解析:
     - InstructUIE 输出: entities[] → ABSA 输入: entities[]
     - Schema 兼容: ✅ (entities 格式一致)

  4. 执行计划:
     Step 1: InstructUIE — 抽取实体
     Step 2: ABSA — 对实体进行情感分析
     Step 3: 语义蓝图编译器 — 标准化输出

输出:
  matched_skills: [InstructUIE, ABSA-BERT-MoE]
  execution_plan: [Step1 → Step2 → Step3]
  estimated_quality: 0.90 (满足阈值 0.85)
```

**业务价值**：
- 零配置启动分析任务，系统自动匹配最优技能
- 新技能注册后立即可用，无需修改编排逻辑
- 技能效果可量化对比（F1、延迟、成本）

---

### 场景二：技能版本管理与回滚

**业务问题**：

技能迭代更新时（如 InstructUIE v2.0 → v2.1），新版本可能在某些场景下表现更差。需要支持多版本共存、A/B 测试、快速回滚。

**数据要求**：

- 技能版本历史
- 版本性能指标（F1、延迟、错误率）
- 回滚策略配置

**预期产出**：

```
技能版本管理:
  InstructUIE:
    v2.1 (current): F1=0.91, latency=120ms, 使用率 85%
    v2.0 (stable): F1=0.89, latency=100ms, 使用率 10%
    v1.9 (deprecated): F1=0.85, latency=80ms, 使用率 5%

质量监控告警:
  "InstructUIE v2.1 在 '德语评论' 场景下 F1 降至 0.72"

自动回滚:
  触发条件: 某场景 F1 < 0.80 连续 10 分钟
  回滚动作: 该场景自动切换至 v2.0
  通知: 发送告警给技能维护者
```

**业务价值**：
- 降低技能更新风险，确保生产稳定性
- A/B 测试驱动技能迭代决策
- 故障时自动回滚，减少人工干预

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/skill_registry/skill_registry.py`

核心组件：
- `SkillMetadata`: 技能元数据（名称、版本、Schema、依赖、指标）
- `SkillRegistry`: 注册表主类
  - `register`: 注册技能
  - `discover`: 根据需求发现技能
  - `resolve_dependencies`: 依赖解析 + 拓扑排序
  - `check_compatibility`: Schema 兼容性检查
  - `get_execution_plan`: 生成执行计划
- `VersionManager`: 版本管理（多版本、金丝雀、回滚）

运行方式：
```bash
cd paper2skills-code/mas/skill_registry
python skill_registry.py
```

生产环境建议：
1. 使用数据库（PostgreSQL/MongoDB）持久化技能元数据
2. 实现技能健康检查（定期探测技能可用性）
3. 集成监控（Prometheus metrics：匹配延迟、成功率、版本分布）
4. 支持技能热更新（无需重启 Registry）
5. 实现技能评分排序（综合考虑 F1、延迟、成本、稳定性）

---

## ④ 技能关联

### 前置技能
- **语义蓝图编译器**：生成 Task Blueprint，作为 Skill Registry 的输入
- **FastAPI / 微服务**：技能以独立服务形式部署和注册

### 延伸技能
- **Service Mesh**：技能间的服务发现与负载均衡
- **Feature Store**：技能共享的特征存储
- **Model Registry**：ML 模型的版本管理（与 Skill Registry 联动）

### 可组合技能
- **Subagent Decomposer**：Skill Registry 提供可用技能列表，Decomposer 决定如何组合
- **MAS Orchestrator**：Orchestrator 从 Registry 获取执行计划并调度
- **Self-Refine**：技能性能数据反馈到 Registry，驱动技能优先级调整

---


### 图谱链接
- [[Skill-Semantic-Blueprint-Compiler]]
- [[Skill-Subagent-Decomposition]]
- [[Skill-MAS-Orchestrator]]
- [[Skill-Agent-Memory-Learning]]
- [[Skill-AutoGen-Multi-Agent-Conversation]]
- [[Skill-MetaGPT-SOP-Driven-Collaboration]]

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 动态技能匹配 | 零配置启动分析任务，开发效率提升 50% | 开发 2-3 周 | 15-20x |
| 技能版本管理 | 降低更新风险，减少生产事故 70% | 开发 1-2 周 | 12-18x |
| 技能效果对比 | 数据驱动技能选型，避免拍脑袋决策 | 开发 1 周 | 10-15x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：中，需要技能元数据和性能指标
- 技术门槛：中，核心是依赖解析和 Schema 验证
- 工程复杂度：中，需要持久化和监控
- 维护成本：中低，技能注册是低频操作

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **基础设施地位**：是整个 MAS 工作流的"技能目录"，所有任务都从这里开始
- **可扩展性**：新技能即插即用，系统能力持续扩展
- **决策支持**：量化数据支持技能选型，避免经验主义
- **可落地性强**：2-3 周可完成核心功能

---

## 参考设计

1. **Microservices Service Discovery Pattern**
   - Netflix Eureka / Consul / Kubernetes Service Registry
   - 核心思想：服务声明式注册 + 客户端动态发现

2. **Apache Airflow DAG Dependency Resolution**
   - 核心思想：任务依赖图 + 拓扑排序确定执行顺序

3. **MLflow Model Registry**
   - 核心思想：模型版本管理 + 阶段转换（dev → staging → production）

---

## 在 MAS 工作流中的位置

```
[Task Blueprint 生成]
    ↓
[Skill Registry] ← 当前技能
    ├─ 发现: 匹配可用技能
    ├─ 解析: 依赖拓扑排序
    └─ 验证: Schema 兼容性
    ↓
[执行计划]
    ↓
[Subagent Decomposer]
    ↓
[MAS Orchestrator]
    ↓
[Agent 执行]
```
