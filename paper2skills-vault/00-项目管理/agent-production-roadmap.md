# paper2skills Agent 生产化路线图 · 达尔文进化 50-Loop 完整规划

> 制定时间: 2026-06-01 | 版本: v1.0
> 战略目标: 将知识图谱作为 Agent 生产底座，通过 50 个达尔文进化 Loop 实现 Skills 自动进化

---

## 一、项目现状基线（2026-06-01）

```
Skill 总数:      257（22 个领域）
图谱边数:        3778  /  双向率: 39.6%
MAS 测试:        47/47 passed
断链:            0
最薄领域:        11-AI人文(1) / 21-合规决策(1) / 18-物流履约(4)
WF 覆盖率:       WF-A 95% / WF-B 90% / WF-C 85% / WF-D 80% / WF-E 85%
P0 生产 GAP:     10 个（Agent SLO / 持续学习 / 安全沙箱 / Registry 等）
```

---

## 二、达尔文进化框架

### 2.1 五大机制映射

| 达尔文概念 | paper2skills 映射 | 执行工具 |
|---|---|---|
| **变异 Mutation** | autoresearch 论文搜索 → 新 Skill 候选 | paper-选题 + paper-萃取 |
| **遗传 Heredity** | 高质量 Skill 4 维结构模板传承 | MasterPrompt structure_inheritance |
| **选择 Selection** | Fitness 门控（< 0.6 进入观察期）| skills_graph_analyzer + Fitness函数 |
| **竞争 Competition** | 同生态位 Skill 对比 F 分，低分标记冗余 | 人工确认后归档 |
| **生态位 Niche** | missing_bridge 热力图 = 空白生态位 | skills_graph_analyzer --gaps |

### 2.2 适应度函数（Fitness Function）

```
F = 0.30 × G(图谱连通度)     # 入度+出度之和，越高越好
  + 0.25 × C(WF覆盖贡献度)   # 填补的工作流缺口数量
  + 0.25 × U(业务使用率)      # MAS 日志中被调用次数
  + 0.15 × Q(卡片质量分)      # 4维审核评分均值
  + 0.05 × T(技术时效性)      # exp(-0.3 × (2026 - paper_year))

门控规则:
  F < 0.6 → 进入观察期（30天无调用则归档）
  F < 0.4 → 自动标记淘汰候选
  F ≥ 0.7 → 结构遗传（成为下一代模板基因）
```

### 2.3 防漂移机制

```
领域熵 H = -Σ p_d × log(p_d)
当 H < 0.7（某领域过度密集）→ 强制将下 3 个 Loop 锁定薄领域

稳定性检验（每 5 Loop 执行）:
  - 图谱密度（边/节点）不得下降
  - 双向率不得低于 35%
  - 否则暂停变异，专注孤立节点回填
```

---

## 三、Agent 生产底座架构

### 3.1 Skill → Agent Tool 可执行化路径

```
Layer 1: Skill Card (Markdown + model.py)
         ↓ [代码优先生成]
Layer 2: Skill Descriptor (JSON Schema)
         {name, version, inputs, outputs, fitness_score}
         ↓ [注册]
Layer 3: SkillRegistry (已有 mas/skills/)
         ↓ [封装]
Layer 4: @skill_tool Decorator（Python Callable）
         ↓ [保护]
Layer 5: 沙箱执行 (Sandlock/Progent)
         ↓ [监控]
Layer 6: SLO Dashboard (Agent-SRE + OpenTelemetry)
```

### 3.2 三层 SLI 体系

```
Layer 1 - 服务层（目标 99.5%）:
  SLI: Skill 调用成功率 / P95 延迟 < 5s

Layer 2 - 任务完成（目标 92%）:
  SLI: Agent Tool 正确选择率 / 工作流完整执行率

Layer 3 - 判断质量（目标 95%）:
  SLI: 母婴业务相关性 ≥ 4.0/5.0（LLM-as-Judge，5%采样）
       人工覆核协议率 > 90%
```

### 3.3 安全隔离三级

```
L0 只读 Skill（VOC 分析/报告生成）→ 无沙箱
L1 写操作 Skill（补货/广告出价）  → 金额上限 + HITL 审批
L2 高危 Skill（合规/合同）        → 独立 subprocess + 超时强杀
```

### 3.4 版本控制与热更新

```
格式: Skill-Name-v{major}.{minor}.{fitness}
例如: Skill-Flowr-Supply-Chain-MAS-v1.0.85

shadow_deploy(14天) → champion 切换（无人工干预）
rollback() → < 30s 热回滚到上一稳定版本
```

---

## 四、P0 阻塞 GAP（10个，必须填补才能进入生产）

| # | GAP | 阻塞原因 | 已找到论文 |
|---|---|---|---|
| 1 | **Agent SLO 管理** | 生产盲飞，无法量化健康度 | Microsoft agent-sre (2026) |
| 2 | **持续学习（在线进化）** | 50-loop 自进化的技术底座 | ATLAS(2511.01093) / AutoSkill(2603.01145) / CASCADE(2605.06702) |
| 3 | **Agent 安全沙箱** | L2 高危 Skill 无隔离 | Sandlock(2605.26298) / Progent(2504.11703) / AgentTrust(2605.04785) |
| 4 | **Agent Registry & Discovery** | 工具无法动态组合 | Agent Registry 2025-2026 |
| 5 | **Exception-Handling-Policy** | 无语义级异常分类 | LLM agent exception recovery 2025 |
| 6 | **Dynamic-DAG-Orchestration** | WF-D 无法动态调整 | Dynamic DAG workflow 2025 |
| 7 | **跨模态理解（Table/Chart/PDF）** | 合规文档无法解析 | Multimodal document agent 2025 |
| 8 | **Tool-Auto-Discovery** | 无法自动接入新工具 | Tool discovery MCP 2025 |
| 9 | **Cultural-Adaptation** | 跨境无文化适应层 | Cross-lingual cultural agent 2025 |
| 10 | **Recall Risk Prediction** | 合规决策仅 1 个 Skill | Product recall ML 2025 |

---

## 五、50 Loop 执行计划

### Loop 标准 SOP（每 Loop 约 2-4 小时）

```
输入:  gap_priority[n] + arxiv_search_keywords
处理:  paper-选题(30min) → paper-萃取(60min) → 验证(30min) → 图谱同步(15min)
输出:  Skill 卡片 + 代码 + fitness_score + 图谱更新
验收:  F ≥ 0.6 + 图谱边数净增 ≥ 3 + 断链 = 0
```

### Loop 1-10：P0 生产化基础设施

| Loop | Skill 目标 | 论文来源 | 预期 F |
|---|---|---|---|
| L1 | Skill-ATLAS-Continual-Learning | arXiv:2511.01093 | 0.91 |
| L2 | Skill-AutoSkill-Lifelong | arXiv:2603.01145 | 0.88 |
| L3 | Skill-CASCADE-Deployment-Learning | arXiv:2605.06702 | 0.86 |
| L4 | Skill-Sandlock-Agent-Sandbox | arXiv:2605.26298 | 0.89 |
| L5 | Skill-Progent-Privilege-Control | arXiv:2504.11703 | 0.87 |
| L6 | Skill-AgentTrust-Runtime-Safety | arXiv:2605.04785 | 0.85 |
| L7 | Skill-Agent-SLO-Manager | Microsoft agent-sre | 0.92 |
| L8 | Skill-Agent-Error-Budget | Microsoft agent-sre | 0.88 |
| L9 | Skill-Agent-Registry-Discovery | 搜索 2025-2026 | 0.85 |
| L10 | Skill-CapSeal-Secret-Mediation | arXiv:2604.16762 | 0.83 |

**里程碑 M1（Loop 10 后）**: P0 GAP 归零，安全沙箱可用，SLO 体系建立

### Loop 11-20：P1 薄领域 + 可靠性

| Loop | 方向 | 目标 |
|---|---|---|
| L11-13 | 跨模态理解（Table/Chart/PDF Agent）| 09-DataAgent 扩充 |
| L14-15 | Dynamic-DAG + Exception-Handling | 10-MAS + 16-智能体 |
| L16-18 | 合规决策扩充（+5 Skill）| 21-合规领域 1→6 |
| L19-20 | 物流履约扩充（+3 Skill）| 18-物流 4→7 |

**里程碑 M2（Loop 20 后）**: 全WF 覆盖率 ≥ 95%，最薄领域 Skill ≥ 5

### Loop 21-30：跨域桥梁密化

每个 Loop 消灭 3-4 对零跨域领域组合，优先：
- 因果推断 × DataAgent-LLM（Causal Data Science）
- 因果推断 × 20-AI视频（视频效果归因）
- A/B实验 × 推荐系统（实验驱动推荐）
- 营销 × 风控（营销欺诈联合分析）

**里程碑 M3（Loop 30 后）**: 零跨域对 < 30，图谱边数 > 4500

### Loop 31-40：竞争力 Skill + 自主进化能力

| Loop | Skill | 方向 |
|---|---|---|
| L31 | Tool-Auto-Discovery | 工具自动接入 |
| L32 | Cultural-Adaptation-Agent | 跨文化适应 |
| L33 | XSkill-Dual-Stream | arXiv:2603.12056，多模态自进化 |
| L34 | User-Profile-Long-Memory | 长期用户画像 |
| L35 | Cross-Session-Context | 跨会话上下文 |
| L36-38 | Fitness 自动评估器 | autoresearch 自动化 |
| L39-40 | 组合变异实验 | 2+2 Skill 融合 → 桥接 Skill |

**里程碑 M4（Loop 40 后）**: 双向率 ≥ 45%，autoresearch 自动化 80%

### Loop 41-50：图谱稳定 + 生产验证

| Loop | 内容 |
|---|---|
| L41-43 | 选择压力：淘汰 F < 0.4 Skill，人工审核 |
| L44-46 | 全局断链扫描 + 质量核查 |
| L47-48 | 生产就绪度评估（SLO 达标 / 安全测试 / 性能基准）|
| L49 | Agent Production Readiness Report |
| L50 | 首批 Skill 正式进入 MAS 工具链 + 自进化机制验证 |

**里程碑 M5（Loop 50 后）**: Agent Production Readiness = PASS

---

## 六、逃生阀（何时停止 autoresearch，切人工）

```
触发条件（满足任意一条）:
1. 连续 3 个 Loop F < 0.6 → arXiv 无合适论文，切换白皮书/工业报告
2. 图谱双向率连续下降 → 变异过激，专注关联回填
3. 任意 WF 覆盖率下降 > 5% → Skill 被场景淘汰，重评
4. 领域熵 H < 0.5 → 强制锁定薄领域 3 个 Loop
```

---

## 七、立即执行（本周）

### 本周 Loop 1-3 执行计划

**L1（今天）**: 萃取 ATLAS 持续学习 Skill
- 论文: arXiv:2511.01093
- 目标: Skill-ATLAS-Gradient-Free-Continual-Learning
- 领域: 16-智能体工程
- 技术要点: 双 Agent 架构（Teacher/Student）+ 持久学习记忆

**L2（明天）**: 萃取 Sandlock 安全沙箱 Skill
- 论文: arXiv:2605.26298
- 目标: Skill-Sandlock-Agent-Execution-Sandbox
- 领域: 16-智能体工程
- 技术要点: Landlock + seccomp-bpf，5ms 启动，MCP 集成

**L3（后天）**: 萃取 Agent-SLO-Manager Skill
- 来源: Microsoft agent-governance-toolkit
- 目标: Skill-Agent-SLO-Manager
- 领域: 16-智能体工程
- 技术要点: SLI/SLO/Error-Budget/Circuit-Breaker，OpenTelemetry

---

> **下一步行动**: 立即执行 L1（ATLAS 萃取），2-4小时完成
> **当前知识图谱**: 257 Skill / 3778 边 / 严重缺口=0
> **50-Loop 完成后预期**: ~310 Skill / 4500+ 边 / 双向率 45%+ / 生产就绪
