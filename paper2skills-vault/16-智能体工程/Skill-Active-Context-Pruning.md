---
title: 仿生粘菌主动上下文剪枝 — Focus Agent 自主压缩架构
doc_type: knowledge
module: 16-智能体工程
topic: active-context-pruning
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: 主动上下文剪枝 — Focus 仿生粘菌自主压缩架构

---

## ① 算法原理

### 核心思想

**Focus** 借鉴 **Physarum polycephalum**(多头绒泡菌,俗称粘菌)的探索-收缩策略,把 LLM agent 从被动 "append-only" 模式升级为主动 "explore → compress → withdraw" 模式:

- **生物类比**:粘菌探索迷宫时不保留每条肌肉运动的轨迹,只保留"地图学习";它会主动撤回死胡同里的伪足,同时留下化学痕迹避免重复探索。
- **Agent 类比**:agent 不需要记住 10 分钟前 `ls -R` 输出的 50 行内容,只需要记住"配置文件不在 /src 目录里"。

### 与已有方案的差异

| 方案 | 控制方 | 时机 | 状态保留 |
|------|--------|------|---------|
| Append-only(默认) | 无 | 永不压缩 | 全部累积 |
| MemGPT | OS 控制 | 由 paging 触发 | 虚拟内存层级 |
| Reflexion | 任务间 | episodic memory | 反思在两次任务之间 |
| **Focus(本)** | **Agent 自主** | **任务内任意时刻** | **持久 Knowledge block** |

### 算法直觉:Focus Loop 与 Sawtooth Pattern

Focus 给 standard ReAct 加 2 个 primitive 工具:

1. **`start_focus(topic)`**:声明当前要调查什么(如"Debug DB connection"),打 checkpoint
2. **`complete_focus(summary)`**:生成结构化总结(尝试了什么 / 学到什么 / 结果),系统把 summary 写到顶部 Knowledge block,**删除** checkpoint 到当前的所有原始消息

context 大小由"单调递增"变成 **Sawtooth(锯齿)**:

```
context size
   ^
   |    /\      /\      /\
   |   /  \    /  \    /  \      <- 探索期 context 涨
   |  /    \__/    \__/    \__   <- complete_focus 后塌缩
   |________________________ time
            ↑ Knowledge block 持续累积关键学习
```

agent 自主决定何时 start/complete,**没有外部 timer 或启发式**强制压缩。

### 关键实证结果(SWE-bench Lite, Claude Haiku 4.5, N=5)

| 指标 | Baseline | Focus | Δ |
|------|---------|-------|---|
| 任务通过率 | 3/5 (60%) | 3/5 (60%) | **Same** |
| 总 token 消耗 | 14.92M | 11.53M | **−22.7%** |
| 平均/任务 token | 2.98M | 2.31M | −678K |
| 平均压缩次数 | 0 | 6.0 | +6 |
| 平均消息删除数 | 0 | 70.2 | +70 |

**单任务最佳节省**:matplotlib-26020 节省 **57%**(4.0M → 1.7M)。

### Aggressive Prompting 才是关键

论文最反直觉的发现:**被动 prompting 失败,激进 prompting 才成功**:

- 被动 prompting(只给 tool):2 次压缩/任务,**仅 6% 节省**,且准确率下降 60% → 80%
- 激进 prompting(强制 + 提醒):6 次压缩/任务,**22.7% 节省**,准确率不变

激进 prompt 设计要点:
1. **强制工作流**:"ALWAYS call start_focus before ANY exploration"
2. **周期性 reminder**:每 15 次工具调用注入 "REMINDER: You should call complete_focus"
3. **结构化阶段**:explicit 引导 explore → understand → implement → verify 4-6 个 focus 阶段

### 关键洞察

- **频次比力度重要**:频繁小压缩(每 10-15 次工具调用)优于偶尔大压缩。前者保留近期上下文同时弃掉过期 log,后者风险大。
- **认知税(Cognitive Tax)可被摊销**:每次压缩成本几百 token,但节省数千 token。任务越长(50+ tool calls)摊销越划算。
- **探索型任务收益最大**:扩展性强的代码导航任务获 50-57% 节省;迭代精修类任务(如 pylint-7080)可能 +110% 反向增加。
- **LLM 缺乏成本意识**:当前模型不会自发压缩,需 scaffold 把 compression 做成 first-class workflow。

### 关键挑战

- **过度压缩风险**:迭代精修任务可能丢掉关键上下文,迫使重新探索(pylint-7080 +110%)
- **压缩质量依赖 LLM**:summary 写得差会丢信息;summary 写得太长抵消节省
- **强 prompting 难泛化**:Haiku 4.5 的激进策略未在其他模型验证(论文 N=5 有限)

### 关键假设

1. Agent 能用自然语言写出准确的 sub-task summary
2. 任务有 explore/implement 区分(对纯迭代任务不适用)
3. LLM 服从激进 prompt 的指令性(否则回退到 2 次/任务的低效模式)

---

## ② 母婴出海应用案例

### 场景一:跨境客服 Agent 长会话压缩(降本)

**业务问题**:

跨境母婴客服 agent 需要处理长会话:客户多轮咨询(过敏 → 退货 → 物流 → 关税 → 售后)。每轮会调多个工具(查订单、查批次、查物流、查关税),最终 context 累积到 30k-80k token,主要是中间工具结果(批次明细、物流单步骤、关税计算表)。

如果用 Claude Haiku 4.5 处理 100k 工单/月:
- Baseline:每工单 50k token × 100k = 5B token/月 ≈ $500
- 但 50k token 里只有 5-10k 是关键 (客户身份、品牌、决策依据)

**Focus 落地方案**:

```
客服 Agent 工作流:

1. start_focus("查询客户订单 ORD1001 状态")
   ├─ tool_call: order_lookup(ORD1001)
   ├─ tool_result: 长输出(订单 + 物流 + 批次)
   ├─ tool_call: batch_check(BATCH4)
   └─ tool_result: 批次详情

2. complete_focus()
   Knowledge block 追加: "ORD1001 已发货, BATCH4 无召回, 客户 VIP 银卡"
   删除中间消息: 节省 5k token

3. start_focus("过敏退货决策")
   ├─ tool_call: check_allergy_severity("severe rash")
   ├─ tool_call: compliance_check("CN", "refund_full")
   └─ tool_call: refund_workflow_init(ORD1001)

4. complete_focus()
   Knowledge block 追加: "决定全额退款 + 召回 BATCH4"

5. 生成最终回复(只看 Knowledge block, context 极小)
```

**业务价值**:

- Token 消耗:-22.7% (按论文数据)= -$113/月,年化 -$1.4k
- 响应延迟:context 短 = TTFT 快 = 客户体验更好
- 准确率:论文实证 0 退步 (60% = 60%),内部测试需验证

### 场景二:商家自助选品 Agent 长探索压缩

**业务问题**:

母婴选品 agent 帮商家做新品决策:浏览 50+ 商品页 → 查 30+ 历史销量 → 对比 10+ 竞品 → 看法规 → 做财务模型 → 输出最终选品建议。整个过程 100+ 工具调用,context 容易爆 200k 上限。

**Focus 落地方案**:

```
选品 Agent 工作流(对应论文 explore-heavy 场景,预期 -50 ~ -57% 节省):

阶段 1: start_focus("筛选 0-1 岁辅食目标品类")
  → 浏览 50 个商品页, 抓取价格 / 销量 / 评分
  → complete_focus("候选 8 个: 米粉 3 / 果泥 3 / 蔬泥 2, 全部 4.5+ 评分")
  [删除 50 个商品页详细内容]

阶段 2: start_focus("评估前 5 候选的财务模型")
  → 查毛利 / 物流 / 关税 / 仓储
  → complete_focus("Top 2: 米粉 X (毛利 45%) + 果泥 Y (毛利 38%)")
  [删除中间财务计算]

阶段 3: start_focus("法规合规检查")
  → 查 8 个国家的婴幼儿食品标准
  → complete_focus("米粉 X 美国 + 加拿大可上, 欧盟需调配方")

阶段 4: 最终生成选品建议(只引用 Knowledge block, context 5-10k)
```

**业务价值**:

- Token 消耗:对应论文 matplotlib/sympy 的 -57% 节省,选品任务约 -50%
- 单次任务从 50k token → 25k token = 节省 $0.03/次
- 月度 5k 次选品任务 = -$150/月
- 准确率:对探索-实施清晰的任务,论文实证不下降
- 长任务可行性:不再因 context 限制中断(原来 200k 经常爆)

---

## ③ 代码模板

代码位置:`paper2skills-code/llm_agent_engineering/active_context_pruning/focus_agent.py`

核心组件:

- `ContextMessage`:单条消息(role / content / message_id / phase_id)
- `FocusPhase`:一个 explore-compress 段(start checkpoint → tool calls → complete summary)
- `KnowledgeBlock`:持久化的关键学习(append-only)
- `FocusAgent`:维护 context + knowledge,提供 `start_focus()` / `complete_focus()` 工具
- `FocusOrchestrator`:外部 ReAct loop + system reminder injection
- `aggressive_prompt_template`:论文激进策略 prompt 模板
- 母婴客服 demo:模拟过敏退货长会话,验证 sawtooth 模式 + token 节省

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/active_context_pruning
python3 focus_agent.py
```

生产环境建议:

1. 接 Claude Haiku 4.5 / GPT-5.2,启用 Anthropic Skills 风格的 tool 定义
2. **激进 prompt 是必须的**:passive prompting 节省只有 6%,系数要乘 4 倍
3. **System reminder 周期**:每 15 次 tool call 注入压缩提醒
4. **结构化阶段引导**:explicit list 出 explore → understand → implement → verify 阶段
5. **Trade-off 监控**:对迭代型任务(如代码精修)可禁用 Focus 或调低 compression 频次
6. 与 ACON / AgeMem 互补使用:Focus = intra-task,AgeMem = cross-task LTM

---

## ④ 技能关联

### 前置技能

- **16-智能体工程 [[Skill-Context-Compression]]**(ACON):被动外部压缩对比,Focus 是 agent-centric 版
- **16-智能体工程 [[Skill-Skill-Lifecycle-Design]]**(SoK):理解 Skill 作为 first-class workflow primitive

### 延伸技能

- **16-智能体工程 [[Skill-Memory-as-Action]]**(MemAct,待萃取 P2-3):把 memory 操作完全嵌入 policy 的 RL 训练版
- **16-智能体工程 [[Skill-Co-Evolutionary-Skill-Verification]]**(EvoSkills):演化 skill 时 context 累积过多用 Focus

### 可组合技能

- **16-智能体工程 [[Skill-Agentic-Memory-Management]]**(AgeMem):Focus 处理 intra-task,AgeMem 处理 inter-task LTM
- **16-智能体工程 [[Skill-MCP-A2A-Protocol-Stack]]**:Focus tools 注册到 MCP server,跨 agent 复用
- **本项目 paper-审核 skill**:对长 paper 萃取过程可应用 Focus 压缩中间笔记

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 跨境客服长会话压缩 | Token -22.7%, 年化 -$1.4k/100k 工单 | 工程 2-3 周 + prompt 迭代 1 周 | 8-12x |
| 商家选品 Agent 长探索 | Token -50%, 长任务 200k → 100k 内 | 工程 3-4 周 + 业务测试 | 10-15x |
| Skill 演化 / 自动萃取流程 | 演化轮次延长 2-3x, 收敛率 +20% | 工程 2 周(集成现有 EvoSkills) | 5-8x |

### 实施难度

**评分:⭐⭐⭐☆☆(3/5 星)**

- 数据要求:低,不需训练数据,纯 prompt 工程
- 技术门槛:中,要懂 ReAct loop + tool 系统 + LLM message manipulation
- 工程复杂度:中,主要是 message store + history pruning 的状态管理
- 维护成本:低,prompt 一次调好后基本不变

### 优先级评分

**评分:⭐⭐⭐⭐☆(4/5 星)**

- **立刻可落地**:不需要训练,纯架构 + prompt
- **直接降本**:22-57% token 节省直接转化为 API 成本节省
- **副作用低**:论文实证准确率 0 损失(对探索型任务)
- **泛化性强**:任何长会话 agent 都可套用
- **风险点**:迭代精修任务可能反向亏损,需要任务类型分流

### 评估依据

1. **A/B 实证清晰**:N=5 但 Δ −22.7% 显著,且配套 ablation(passive vs aggressive)
2. **极简实现**:只需 2 个新工具 + 1 个 message store,1-2 周可上线
3. **直接业务价值**:跨境客服长会话场景天然契合,Focus 是 first-class fit
4. **完整 receipt**:论文给出 prompt 模板 + 实测结果 + 失败 case study(pylint-7080)
5. **可叠加优化**:与 ACON / AgeMem / Skills 互补,不冲突

---

## 参考论文

1. **Focus: An Agent-Centric Approach to Active Context Pruning Inspired by Slime Mold** (2026-01)
   - Nikhil Verma
   - 核心贡献:agent-controlled intra-task context compression + slime mold analogy + sawtooth pattern + aggressive prompting strategy
   - arxiv:[2601.07190](https://arxiv.org/abs/2601.07190)

## 相关基础

- **MemGPT** (arxiv:2310.08560):OS-style virtual memory hierarchy
- **Voyager** (arxiv:2305.16291):reusable skill library reduces redundant exploration
- **Reflexion** (NeurIPS 2023):episodic memory between task attempts
- **StreamingLLM** (arxiv:2309.17453):attention sink for infinite context
- **Lost in the Middle** (arxiv:2307.03172):长 context 中信息检索退化

---

## 与同领域 Skill 的对比

| 维度 | Focus (本) | ACON (P1-2) | AgeMem (P1-3) |
|------|-----------|-------------|---------------|
| 压缩范围 | Intra-task | Intra/inter-task | Inter-task LTM |
| 控制方 | Agent 自主 | 外部 utility/CO loss | RL policy |
| 训练需要 | 无(纯 prompt) | 需要 trajectory data | RL 训练 |
| 压缩对象 | 历史消息 | History + Observation | LTM/STM 全量 |
| 实证节省 | -22.7% token | -26 ~ -54% memory | 准确率 +10pp |
| 落地周期 | 短(2-3 周) | 中(4-6 周) | 长(8-12 周) |

**互补使用**:
- **Intra-task 压缩**用 Focus(纯 prompt 工程)
- **训练时压缩 policy**用 ACON 蒸馏到小模型
- **跨任务持久记忆**用 AgeMem 三阶段 RL
- **三者叠加**:Focus 管单任务 + ACON 模型推理时压缩 + AgeMem 长周期 LTM
