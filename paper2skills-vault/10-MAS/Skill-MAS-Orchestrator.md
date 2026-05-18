---
title: MAS Orchestrator — 多智能体编排与调度
doc_type: knowledge
module: 10-MAS
topic: mas-orchestrator
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
---

# Skill: MAS Orchestrator — 多智能体编排与调度

---

## ① 算法原理

### 核心思想

**MAS Orchestrator** 是多 Agent 系统的"中枢神经系统"，负责协调多个子 Agent 的执行顺序、数据流转、状态同步和错误恢复。核心洞察：**分解后的子任务需要一个可靠的调度器来管理它们的生命周期——启动、监控、通信、容错、收尾**。

Orchestrator 的五大职责：

1. **生命周期管理**：启动子 Agent、监控执行状态、处理完成/失败事件
2. **数据流转**：管理子 Agent 间的输入/输出传递（消息总线）
3. **状态同步**：维护全局执行状态，支持断点续传和进度查询
4. **错误处理**：失败重试、降级策略、超时控制、死锁检测
5. **资源调度**：子 Agent 并发度控制、优先级调度、资源配额管理

### 执行模型

```
Orchestrator 执行循环:

初始化:
  加载 DAG
  初始化所有节点的状态为 PENDING

主循环:
  while 存在非终态节点:
    1. 扫描: 找出所有依赖已满足的 PENDING 节点
    2. 调度: 将这些节点加入执行队列
    3. 执行: 为每个节点启动对应的 Subagent
    4. 监控: 轮询运行中节点的状态
    5. 处理事件:
       - SUCCESS → 标记完成，触发下游节点
       - FAILURE → 根据策略重试或标记失败
       - TIMEOUT → 取消并触发超时处理
       - PROGRESS → 更新进度（用于长任务）

收尾:
  收集所有节点的输出
  执行汇总/归约操作
  返回最终结果
```

### 关键假设

1. **DAG 无环**：子任务依赖图必须是无环的
2. **状态可观测**：子 Agent 的执行状态可以被外部查询
3. **失败可恢复**：失败的子任务可以被重试或降级
4. **消息可靠传递**：子 Agent 间的数据传递可靠

---

## ② 母婴出海应用案例

### 场景一：全品类 VOC 分析流水线编排

**业务问题**：

全品类 VOC 分析涉及 8 个并行子任务（各品类分析）+ 2 个串行汇总任务。需要可靠地编排执行、处理部分失败、汇总结果。

**数据要求**：

- Subagent Decomposer 生成的执行 DAG
- 各子 Agent 的配置（技能、资源配额）
- 超时和重试策略配置

**预期产出**：

```
Orchestrator 执行视图:

时间线:
  T+0s  │ [启动] Orchestrator 加载 DAG
  T+1s  │ [调度] T1-T8 (8个品类分析) 依赖满足，全部启动
  T+1s  │   T1[吸奶器] RUNNING  (Agent: InstructUIE+ABSA)
  T+1s  │   T2[储奶袋] RUNNING
  T+1s  │   ...
  T+1s  │   T8[其他] RUNNING
        │
  T+45s │   T3[温奶器] SUCCESS (最快完成)
  T+52s │   T5[推车] SUCCESS
  T+61s │   T1[吸奶器] SUCCESS
  T+63s │   T2[储奶袋] SUCCESS
  T+68s │   ...
  T+72s │   T8[其他] SUCCESS (最慢完成)
        │
  T+73s │ [调度] T9[跨品类对比] 依赖满足，启动
  T+95s │   T9 SUCCESS
        │
  T+96s │ [调度] T10[报告生成] 依赖满足，启动
  T+110s│   T10 SUCCESS
        │
  T+110s│ [完成] 全部任务成功
        │   总耗时: 110s
        │   理论串行: ~500s
        │   加速比: 4.5x

错误处理示例:
  T+60s │   T4[安全座椅] FAILURE (API 超时)
  T+60s │ [重试] T4 第1次重试 (指数退避: 等待 5s)
  T+68s │   T4 SUCCESS (重试成功)
  T+68s │ [恢复] T9 解除阻塞，正常调度
```

**业务价值**：
- 复杂流水线可靠执行，无需人工监控
- 部分失败自动恢复，不影响整体进度
- 执行过程可视化，进度可查询

---

### 场景二：实时 VOC 预警流水线

**业务问题**：

需要实时监控评论数据流，当某个品类的负面率突增时，自动触发分析流水线（抽取 → 根因分析 → 预警生成 → 通知发送），要求在 5 分钟内完成。

**数据要求**：

- 实时评论数据流（Kafka / 消息队列）
- 预警触发规则
- 各阶段子 Agent 配置
- 通知渠道配置

**预期产出**：

```
实时预警流水线:

触发条件: "吸奶器品类负面率 15min 滑动窗口均值 > 20%"

编排执行:
  Stage 1 (0-30s): 数据获取
    ├─ 获取最近 1h 吸奶器评论 (500条)
    └─ 获取上一周期对比数据

  Stage 2 (30-90s): 并行分析
    ├─ InstructUIE Agent: 抽取实体/属性
    ├─ ABSA Agent: 方面级情感分析
    └─ Trend Agent: 时间序列异常检测

  Stage 3 (90-120s): 根因分析
    └─ RCA Agent: 综合各维度输出根因假设

  Stage 4 (120-180s): 预警生成与分发
    ├─ AlertGen Agent: 生成结构化预警报告
    ├─ 发送邮件给产品经理
    ├─ 发送 Slack 消息给运营团队
    └─ 写入告警数据库

SLA: 触发 → 通知 全程 ≤ 5min
实际: 平均 2.8min
成功率: 99.7%
```

**业务价值**：
- 质量问题从"事后发现"变为"实时预警"
- 响应时间从 1-2 天缩短到 3 分钟
- 预警准确率随反馈持续优化

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/mas_orchestrator/orchestrator.py`

核心组件：
- `TaskNode`: 任务节点（ID、状态、依赖、Agent 配置）
- `ExecutionDAG`: 执行图（加载 DAG、状态管理）
- `MessageBus`: 消息总线（子 Agent 间通信）
- `Orchestrator`: 编排器主类
  - `execute`: 执行完整 DAG
  - `schedule_ready_tasks`: 调度就绪任务
  - `handle_event`: 处理执行事件
  - `retry_task`: 失败重试
  - `collect_results`: 汇总结果

运行方式：
```bash
cd paper2skills-code/mas/mas_orchestrator
python orchestrator.py
```

生产环境建议：
1. 使用 Temporal / Airflow / Dagster 作为底层工作流引擎
2. 实现执行状态持久化（支持断点续传）
3. 集成监控告警（执行延迟、失败率、资源使用率）
4. 支持动态 DAG 修改（执行中调整流程）
5. 实现资源隔离（避免某个子 Agent 占用全部资源）

---

## ④ 技能关联

### 前置技能
- **Subagent Decomposer**：提供执行 DAG
- **分布式系统基础**：理解任务调度、消息队列、容错

### 延伸技能
- **Temporal**：持久化工作流引擎
- **Apache Airflow**：DAG 调度平台
- **Kubernetes Jobs**：容器化任务调度

### 可组合技能
- **Skill Registry**：Orchestrator 从 Registry 获取 Agent 配置
- **Subagent Decomposer**：Decomposer 生成 DAG，Orchestrator 执行 DAG
- **Self-Refine**：执行中的异常触发反思，优化后续执行策略
- **AutoGen**：每个子任务对应一个 AutoGen ConversableAgent

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 复杂流水线编排 | 全自动化执行，人工监控减少 90% | 开发 3-4 周 | 12-18x |
| 实时预警流水线 | 响应时间从天级缩短到分钟级 | 开发 2-3 周 | 20-30x |
| 批量数据处理 | 支持 10x 数据量，时间不变 | 开发 2 周 | 15-20x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：中，需要 DAG 定义和 Agent 配置
- 技术门槛：高，需要分布式系统经验
- 工程复杂度：高，涉及并发控制、容错、状态管理
- 维护成本：中，DAG 调整需要测试验证

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **系统核心**：没有 Orchestrator，多 Agent 系统就是一盘散沙
- **可靠性保障**：容错、重试、超时控制是生产必备
- **可观测性**：执行过程透明，便于调试和优化
- **业务刚需**：任何复杂分析都需要可靠的执行保障

---

## 参考设计

1. **Temporal: Durable Execution** (2021)
   - 核心思想：工作流代码持久化，故障后可从断点恢复
   - 适用：长周期、多步骤、需要可靠性的工作流

2. **Apache Airflow** (Apache)
   - 核心思想：Python 定义 DAG，调度器按依赖执行
   - 适用：数据管道、ETL、批处理

3. **AWS Step Functions** (Amazon)
   - 核心思想：状态机定义工作流，可视化编排
   - 适用：事件驱动、微服务编排

---

## 在 MAS 工作流中的位置

```
[Task Blueprint]
    ↓
[Skill Registry] — 技能发现
    ↓
[Subagent Decomposer] — 任务分解
    ↓
[MAS Orchestrator] ← 当前技能
    ├─ 调度就绪任务
    ├─ 监控执行状态
    ├─ 处理失败重试
    └─ 汇总最终结果
    ↓
[Agent 1] ←→ [Agent 2] ←→ [Agent 3] ...
    ↓
[结果汇总]
    ↓
[反馈 → 再训练 → 闭环]
```
