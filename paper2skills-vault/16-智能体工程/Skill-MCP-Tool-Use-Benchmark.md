---
title: MCP Tool Use 评估基准 — TFS/TEFS 双指标与干扰测试
doc_type: knowledge
module: 16-智能体工程
topic: mcp-tool-use-benchmark
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: MCP Tool Use 评估基准 — TFS/TEFS 双指标与干扰测试

---

## ① 算法原理

### 核心思想

**MCPAgentBench** (北京大学 + ZTE, 2026) 是首个专注于**工具选择与执行效率**的 MCP 评估基准。现有基准 (MCP-Universe, MCP-RADAR) 主要测正确性，忽略了一个关键问题：**模型能完成任务，但效率极低** —— 该并行时串行、该串行时并行、传过多参数、反复试错。

论文核心洞察：**任务完成率 ≠ 执行效率**。一个模型可能 TFS (完成率) 很高，但 TEFS (效率完成率) 很低，说明它在"暴力解题"而非"优雅解题"。

### 数据集构建

**四步流水线**:

| 步骤 | 输入 | 输出 | 方法 |
|------|------|------|------|
| 1. 原始数据收集 | MCP Market, HuggingFace, Infinity-Instruct | 9,714 servers, 20,000+ tools | 去重 |
| 2. 标注 | 原始 tools + tasks | 标准化标签集 | LLM 开放标注 → 人工整合 → LLM 约束标注 |
| 3. 匹配与筛选 | 标签匹配的 task-tool 对 | 178 个高质量测试用例 | 人工审核，确保唯一解 |
| 4. 代码生成 | 筛选后的 tool 定义 | 可执行的 mock 函数 | GPT-4o 生成 + 专家审核 |

**最终数据集**: 178 tasks, 覆盖 Daily (日常生活) 和 Professional (专业领域) 两个 domain。

### 任务分类体系

**按 Domain**: Daily (日常) vs Professional (专业)

**按调用复杂度** (Invocation Complexity):

| 类型 | 数量 | 描述 | 测试能力 |
|------|------|------|---------|
| **Single-Tool** | 60 (30×2) | 单次调用一个 tool | 基础 tool 选择 |
| **Dual Parallel** | 40 (20×2) | 两个独立 tool 并发调用 | 任务分解 + 并行规划 |
| **Dual Serial** | 40 (20×2) | 两个 tool 按顺序调用，第二步依赖第一步输出 | 多步推理 + 状态维护 |
| **Multi-Tool** | 38 (20+18) | 多步组合（并行+串行混合） | 复杂编排 |

### Distractor 设计

**核心创新**: 每个 task 提供候选工具列表 L (含 K=20/30 个 tools)，其中：
- **G**: 正确工具 (golden solution 所需)
- **F**: 干扰工具 (distractors，功能相似但不适用的 tools)

**"大海捞针"场景**: Agent 必须从含干扰项的列表中精确选择正确工具，测试**工具辨别能力**和**抗干扰鲁棒性**。

### 评估指标

**双核心指标**:

#### TFS (Task Finish Score) — 任务完成率

$$TFS = \frac{\sum_{i=1}^{N} |G_i| \cdot IsFinished(T_i)}{\sum_{i=1}^{N} |G_i|}$$

- $IsFinished(T_i) = 1$ iff Agent 调用的 tool 集合与 golden solution 完全相同
- 要求: tool name + parameters 完全匹配
- **不考虑执行顺序**

#### TEFS (Task Efficiency Finish Score) — 效率完成率

$$TEFS = \frac{\sum_{i=1}^{N} |G_i| \cdot IsEfficientlyFinished(T_i)}{\sum_{i=1}^{N} |G_i|}$$

- $IsEfficientlyFinished(T_i) = 1$ iff: (1) tool 集合匹配，且 (2) **serial/parallel 执行顺序与 golden solution 一致**
- **更严格的指标**: 不仅要做对，还要高效地做

**效率指标**:

| 指标 | 定义 | 用途 |
|------|------|------|
| **Token Efficiency** | TEFS 分数 / 1k output tokens | 评估 token 成本效益 |
| **Time Efficiency** | TEFS 分数 / 分钟执行时间 | 评估时间成本效益 |

### 关键实证结果

**11 个主流模型 avg@4 评估** (Figure 3):

| 模型 | TFS | TEFS | TEFS-TFS Gap | 策略特征 |
|------|-----|------|-------------|---------|
| **Claude Sonnet 4.5** | **71.6** | **57.7** | -13.9 | 激进并行，Dual Parallel TFS=TEFS |
| o3 | 66.0 | 37.5 | **-28.5** | 极端串行，Dual Parallel TEFS=0 |
| glm-4.6 | 65.1 | 54.4 | -10.7 | 均衡策略 |
| qwen3-235b (no-thinking) | ~60 | 51.8 | ~-8 | 无 thinking token，Token Efficiency 最高 |
| Gemini 3 Pro | 48.1 | 33.5 | -14.6 | 全面落后 |

**关键发现**:

1. **所有模型 TEFS < TFS** (gap 8-28.5 points): 当前模型优先"做对"而非"做高效"
2. **OpenAI 系列极端串行**: gpt-5, o3, o4-mini 在 Dual Parallel 任务上 TEFS=0，完全不会并行调用
3. **Claude Sonnet 4.5 激进并行**: 并行任务上 TFS=TEFS，但错误地将并行策略用于串行任务，导致 Dual Serial TEFS 异常下降
4. **模型规模 vs TEFS**: 总体正相关，但存在 dip (如 32B 可能优于 72B)
5. **Token Efficiency**: qwen3-235b (no-thinking) 最高，gpt-5 最低 (过多 thinking token 不转化为有效分数)

**Dual Parallel 困境** (Table 1 vs Table 2):
- TFS: Dual Parallel (77.9) > Dual Serial (70.0) — 逻辑上并行更简单
- TEFS: Dual Parallel (32.8) << Dual Serial (57.0) — 但模型普遍不会正确执行并行

### 关键假设

1. Golden solution 的 tool 调用顺序是"最优"的
2. Mock 函数能准确模拟真实 tool 行为
3. Distractor 工具足够相似以产生干扰
4. TEFS 的严格匹配能反映真实执行效率

### 关键挑战

- **沙箱依赖**: 基于 Autogen 的评估环境，迁移成本高
- **Mock 函数局限**: 自动生成的 stub 可能不完全匹配真实 API
- **Golden solution 主观性**: 人工标注的"唯一解"可能不是真正的唯一解
- **Domain 覆盖有限**: 仅 Daily + Professional，缺少特定垂直领域

---

## ② 母婴出海应用案例

### 场景一:客服 Agent Tool Use 能力评估

**业务问题**:

公司部署了多个客服 Agent (基于不同 LLM)，需要客观评估它们的 tool use 能力：
- 简单查询: 单次 tool call (订单查询)
- 复杂查询: 多步串行 (先查订单 → 再查物流)
- 批量查询: 并行调用 (同时查多个订单)

当前评估仅靠人工抽查，无法量化比较不同模型的能力差异。

**MCPAgentBench 落地方案**:

```
评估框架:

1. 构建内部测试集
   - 从客服历史对话提取 50+ 典型任务
   - 分类: Single / Dual Serial / Dual Parallel / Multi
   - 标注 golden solution (正确 tool + 参数 + 顺序)

2. 生成候选工具列表
   - 内部 MCP server: 订单、物流、尺码、合规、退换货等
   - 添加 distractor: 功能相似但不适用的 tools
   - K = 10-15 (根据实际 tool 数量调整)

3. 运行评估
   - 每个模型运行完整测试集
   - 记录: tool calls, parameters, execution order, tokens, time

4. 计算指标
   - TFS: 任务完成率
   - TEFS: 效率完成率
   - Token Efficiency: 成本效益
   - Time Efficiency: 响应速度
```

**预期结果对比**:

| 模型 | TFS | TEFS | Token Eff. | 适用场景 |
|------|-----|------|-----------|---------|
| Claude Sonnet 4.5 | 高 | 高 | 中 | 复杂查询，预算充足 |
| gpt-5 | 高 | 低 | 低 | 简单查询，不推荐复杂场景 |
| qwen3-235b | 中 | 中高 | **最高** | 成本敏感场景 |
| 内部 Hermes 4 70B | 待测 | 待测 | 待测 | 本地部署替代方案 |

**业务价值**:
- 客观选型: 用数据替代主观印象选择客服 Agent 基座
- 成本优化: Token Efficiency 直接对应 API 费用
- 能力定位: 识别模型在并行/串行调用上的短板

### 场景二:新 Tool 上线前的能力回归测试

**业务问题**:

新 tool (如 "跨境合规检查") 上线后，需要验证：
1. Agent 能否正确选择新 tool
2. 新 tool 是否会干扰已有 tool 的选择
3. 整体 TFS/TEFS 是否下降

**MCPAgentBench 落地方案**:

```
回归测试流程:

Baseline (上线前):
  - 运行完整测试集 → TFS_baseline, TEFS_baseline

After (上线后):
  - 添加新 tool 到候选列表
  - 添加含新 tool 的测试任务
  - 运行完整测试集 → TFS_after, TEFS_after

判定:
  - TFS 下降 > 3pp → 阻塞，检查 distractor 影响
  - TEFS 下降 > 5pp → 警告，检查执行策略退化
  - 新 tool 任务 TFS = 0 → 阻塞，tool 描述或实现有问题
```

**业务价值**:
- 预防性: 上线前发现 tool 干扰问题
- 量化: 明确新 tool 对整体系统的影响
- 自动化: 集成到 CI/CD，每次 tool 变更自动跑回归

---

## ③ 代码模板

代码位置: `paper2skills-code/llm_agent_engineering/mcp_tool_use_benchmark/mcp_agent_bench.py`

核心组件:

- `Task`: MCPAgentBench 任务定义 (domain + complexity + distractors)
- `ToolInvocation`: 单次 tool 调用记录
- `GoldenSolution`: 标准答案 (tool sequence + order)
- `TFSEvaluator`: TFS 计算 (set match)
- `TEFSEvaluator`: TEFS 计算 (set + order match)
- `DistractorGenerator`: 干扰工具生成
- `BenchmarkRunner`: 完整评估流程
- 母婴客服场景 demo

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/mcp_tool_use_benchmark
python3 mcp_agent_bench.py
```

生产环境建议:

1. **测试集构建**: 从业务历史提取真实任务，人工标注 golden solution
2. **Distractor 选择**: 功能相似度 > 0.6 但不适用的 tools
3. **评估频率**: 新模型上线前、季度模型选型时
4. **指标阈值**: TFS ≥ 70%, TEFS/TFS ratio ≥ 0.75
5. **沙箱隔离**: 使用 mock 函数避免调用真实 API
6. **结果追踪**: 建立 model × task type 的性能矩阵，长期追踪

---

## ④ 技能关联

### 前置技能

- **16-智能体工程 [[Skill-MCP-A2A-Protocol-Stack]]**(P1-4): MCP 协议基础
- **16-智能体工程 [[Skill-Tool-Description-Audit]]**(P2-8): Tool 描述质量影响 TFS/TEFS
- **16-智能体工程 [[Skill-Open-Source-Tool-Use-Model]]**(P2-6): Hermes 4 等模型可用本基准评估

### 延伸技能

- **16-智能体工程 [[Skill-SLM-Tool-Calling-Optimization]]**(P2-7): SLM 的 TEFS 通常更低，需要专门优化
- **16-智能体工程 [[Skill-Orchestration-Trace-RL]]**(P2-5): 可用 trace 数据优化并行/串行决策

### 可组合技能

- **02-A_B实验 [[Skill-AB-Test-Result-Interpretation]]**: 模型选型可用 A/B 测试验证
- **本项目 paper-选题 skill**: 本基准方法论可用于评估新论文的 tool use 能力声明

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 客服 Agent 选型 | 避免选错模型，节省试错成本 $10k+ | 测试集构建 1 周 + 标注 $1k | 10-20x |
| 新 Tool 回归测试 | 预防上线故障，减少客诉 | 集成 CI/CD 2 天 | 长期收益 |
| 并行策略优化 | Token 成本 -20-30% | 分析 + prompt 调优 1 周 | 5-10x |

### 实施难度

**评分: ⭐⭐⭐☆☆ (3/5 星)**

- 数据要求: 中，需要构建 domain-specific 测试集
- 技术门槛: 中，需要理解 MCP 协议 + 评估指标
- 工程复杂度: 中，沙箱环境 + mock 函数 + 指标计算
- 维护成本: 中低，测试集可复用，新增任务时增量更新

### 优先级评分

**评分: ⭐⭐⭐☆☆ (3/5 星)**

- **数据驱动选型**: 替代主观印象，用指标说话
- **预防故障**: 新 tool 上线前的回归测试
- **成本可见**: Token Efficiency 直接对应 API 账单
- **可复用**: 一次构建测试集，多次评估不同模型
- **限制**: 需要 domain-specific 适配，通用基准不完全适用

### 评估依据

1. **工业级验证**: 北京大学 + ZTE 出品，11 个主流模型实测
2. **指标设计精巧**: TFS/TEFS 双层指标区分"做对"和"高效做"
3. **干扰测试创新**: Distractor 设计真实反映"大海捞针"场景
4. **开源**: 代码和数据集已开源
5. **发现重要问题**: 揭示 OpenAI 模型并行调用能力缺失的行业级问题

---

## 参考论文

1. **MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use** (2026-01)
   - Liu, Liu, Dai, Yu, Yu, Yang, Han, Gao (Peking University + ZTE)
   - 核心贡献: 178 真实任务 + TFS/TEFS 双指标 + distractor 干扰测试 + 四阶调用复杂度
   - arxiv: [2512.24565](https://arxiv.org/abs/2512.24565)

## 相关基础

- **MCP**: Model Context Protocol (Anthropic)
- **MCP-Universe**: 现有 MCP 评估基准 (依赖真实远程 server)
- **MCP-RADAR**: 多维度 MCP 评估指标
- **ToolBench**: 16,000+ API 的 tool manipulation 基准
- **Autogen**: 微软开源的多 Agent 框架
- **ICC**: 评分者间一致性度量

---

## 与同领域 Skill 的对比

| 维度 | MCPAgentBench (本) | MCP-Universe | ToolBench |
|------|-------------------|-------------|-----------|
| 评估重点 | 效率 + 干扰鲁棒性 | 正确性 + 协议合规 | 功能覆盖 |
| 执行环境 | 本地沙箱 (mock) | 真实远程 server | 虚拟 API server |
| 效率指标 | TFS + TEFS + Token/Time | 无 | Pass Rate |
| Distractor | 有 | 无 | 无 |
| 任务类型 | 4 阶复杂度 | 未分类 | 6 类 |
| 稳定性 | 高 (本地) | 低 (依赖远程) | 中 |

**互补使用**:
- **开发阶段**: MCPAgentBench 本地快速迭代
- **生产验证**: MCP-Universe 真实环境最终确认
- **能力对比**: 两个基准一起跑，交叉验证
- **成本控制**: MCPAgentBench 的 Token Efficiency 指导模型选型
