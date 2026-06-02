---
title: MCP Tool 描述质量审核 — 六维 Smell 扫描与动态路由
doc_type: knowledge
module: 16-智能体工程
topic: tool-description-audit
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: MCP Tool 描述质量审核 — 六维 Smell 扫描与动态路由

---

## ① 算法原理

### 核心思想

Queen's University 2026 年的大规模实证研究揭示：**97.1% 的 MCP tool 描述至少含有一个 smell**，这些描述缺陷直接导致 FM 选错工具、传错参数或产生不必要的交互步骤。论文提出**六维评分 rubric + 动态组件路由**，在提升 agent 准确率 (+5.85pp) 的同时控制 token 开销 (+67.46% steps 的 trade-off)。

**研究规模**: 103 个 MCP servers, 856 个 tools

### Tool Description 的六组件模型

论文将 MCP tool 描述分解为六个互补组件：

| 组件 | 角色 | 作用 |
|------|------|------|
| **Purpose** | 需求规范 | 定义工具核心功能和身份 |
| **Guidelines** | 指令引导 | 何时、如何使用工具 |
| **Limitations** | 需求规范 | 已知约束、边界情况 |
| **Parameter Explanation** | 需求规范 | 参数语义和行为含义 |
| **Examples** | 指令引导 | 使用示例 |
| **Return Value** | 需求规范 | 返回值结构和语义 |

**双重性质**: tool 描述既是**需求规范**（定义工具行为），又是**prompt 指令**（引导 FM 推理）。

### 六类 Smell 定义

每类 smell 对应一个组件的 score < 3 (5-point Likert scale):

| Smell | 对应组件 | 患病率 | 含义 |
|-------|---------|--------|------|
| **Unclear Purpose** | Purpose | 56.0% | 工具目的不清晰或模糊 |
| **Missing Usage Guidelines** | Guidelines | 89.3% | 缺少何时/如何使用工具的指引 |
| **Unstated Limitations** | Limitations | 89.8% | 未声明已知约束和边界情况 |
| **Opaque Parameters** | Parameter Explanation | 84.3% | 参数含义或行为影响不透明 |
| **Underspecified or Incomplete** | Return Value | 79.1% | 返回值结构或语义不完整 |
| **Exemplar Issues** | Examples | 77.9% | 示例缺失、错误或不相关 |

**Smell-free 率**: 仅 2.9% 的 tool 描述在所有组件上达标。

### 评分 Rubric

**5-point Likert scale**，score 3 = minimum viable threshold:

| Score | 含义 |
|-------|------|
| 5 | 理想: 精确、完整、含行为细节 |
| 4 | 良好: 清晰、含关键信息 |
| 3 | **最低可行**: 基本信息存在 |
| 2 | 有缺陷: 模糊或不完整 |
| 1 | 缺失: 完全未提及 |

**Smell 判定**: score < 3 → 触发对应 smell。

### 增强效果 (RQ-2)

**全组件增强** (所有 6 个组件补全) 的效果：

| 指标 | 变化 | 含义 |
|------|------|------|
| Success Rate | **+5.85pp** (median) | 任务成功率提升 |
| Average Evaluator (AE) | **+15.12%** | 中间步骤完成质量提升 |
| Execution Steps (AS) | **+67.46%** (median) | 执行步数增加（token 开销） |
| Regression | **16.67%** | 增强后性能反而下降的情况 |

**Trade-off**: 语义完整性 ↔ token 效率。更丰富的描述提升准确率，但消耗更多 context window。

### 组件消融 (RQ-3)

**关键发现**:

1. **没有单一"黄金组合"**: 最佳组件组合因 domain-model 对而异
2. **Examples 可移除**: 移除 Examples 组件不会显著降低性能 (Cochran's Q test p > 0.20)
3. **Purpose + Guidelines 常足够**: 在 Finance 等 domain 中，仅这两个组件就超过全增强版
4. **Domain-specific pruning 可行**: 可为特定 domain 找到最小有效组件集

### 关键假设

1. MCP tool 描述的质量直接影响 FM tool 选择准确率
2. 5-point Likert 评分能有效区分描述质量等级
3. 增强效果可泛化到不同 domain 和 model
4. Domain-specific 组件组合可在保持性能的同时减少 token

### 关键挑战

- **增强可能引入回归**: 16.67% 的情况性能下降
- **Context window 膨胀**: 全增强增加 67% 执行步数
- **无通用最佳组合**: 需要 per-domain 调优
- **FM 评分一致性**: 不同 FM 对同一描述的评分可能不一致

---

## ② 母婴出海应用案例

### 场景一:内部 MCP Tool 描述质量审核

**业务问题**:

公司内部 MCP server 管理多个业务 tools（订单查询、物流追踪、尺码推荐、合规检查、退换货处理等）。随着工具数量增加，描述质量参差不齐：
- 56% 的工具目的不清晰
- 89% 缺少使用指南
- 客服 agent 经常选错工具或传错参数

**MCP Smell Scanner 落地方案**:

```
审核流程:

1. 扫描阶段
   - 输入: 所有 MCP tool 描述 (JSON schema + description)
   - 工具: FM-based Smell Scanner (六维评分)
   - 输出: 每 tool 的六维分数 + smell 报告

2. 分级阶段
   - P0 (紧急): score < 2 的组件 → 立即修复
   - P1 (重要): score = 2 的组件 → 本周修复
   - P2 (优化): score = 3 的组件 → 迭代改进

3. 增强阶段
   - 用 FM Augmentor 自动生成缺失组件
   - 人工审核后替换原描述

4. 验证阶段
   - A/B 测试: 原描述 vs 增强描述
   - 指标: tool 选择准确率、参数正确率、执行步数
```

**示例审核结果**:

| Tool | Purpose | Guidelines | Limitations | Params | Examples | Returns | Smells |
|------|---------|-----------|-------------|--------|----------|---------|--------|
| order_lookup | 4 | 2 | 1 | 3 | 1 | 2 | Missing Guidelines, Unstated Limits, Exemplar Issues |
| logistics_track | 3 | 3 | 2 | 4 | 2 | 3 | Unstated Limits, Exemplar Issues |
| size_recommend | 2 | 1 | 1 | 2 | 1 | 1 | 6 smells |

**业务价值**:
- 质量: tool 选择准确率 +5-10pp
- 成本: 减少错误调用导致的 API 费用
- 维护: 系统化审核替代人工抽查

### 场景二:动态 Tool 描述路由

**业务问题**:

不同业务场景对 tool 描述的需求不同：
- 高峰时段: 需要紧凑描述（减少 token，快速响应）
- 复杂查询: 需要完整描述（提高准确率）
- 新用户: 需要含示例的描述（降低学习成本）

**Tool Description Router 落地方案**:

```
运行时路由策略:

Compact 模式 (默认):
  - 组件: Purpose + Guidelines
  - 适用: 简单查询、高峰时段
  - Token: 最低
  - 准确率: 中上

Standard 模式:
  - 组件: Purpose + Guidelines + Limitations + Parameter Explanation
  - 适用: 标准业务查询
  - Token: 中等
  - 准确率: 高

Full 模式:
  - 组件: 全部 6 个组件
  - 适用: 复杂查询、新用户
  - Token: 最高
  - 准确率: 最高（但有 regression 风险）

路由逻辑:
  - query 复杂度评分 → 选择模式
  - 如果 Compact 模式失败 → 自动升级 Standard
  - 如果 Standard 失败 → 升级 Full
  - 记录每模式的成功率，动态调整路由权重
```

**业务价值**:
- 灵活性: 同一 tool 多套描述，按场景切换
- 成本: 默认 Compact 模式节省 40-60% token
- 准确率: 复杂查询自动切换 Full 模式保证质量

---

## ③ 代码模板

代码位置: `paper2skills-code/llm_agent_engineering/tool_description_audit/mcp_smell_scanner.py`

核心组件:

- `ToolDescription`: MCP tool 描述六组件模型
- `ScoringRubric`: 5-point Likert 六维评分
- `SmellScanner`: 扫描 tool 描述，输出 smell 报告
- `DescriptionAugmentor`: 基于规则/LLM 的描述增强
- `ToolDescriptionRouter`: 运行时动态选择描述版本
- 母婴客服 tool 审核 demo

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/tool_description_audit
python3 mcp_smell_scanner.py
```

生产环境建议:

1. **扫描频率**: 每次 MCP server 更新后自动扫描
2. **评分一致性**: 用多个 FM 交叉评分，取 median
3. **增强策略**: 优先修复 P0/P1，P2 可逐步迭代
4. **路由监控**: 记录每模式成功率，自动调优
5. **回归测试**: 增强后必须跑 A/B 测试验证
6. **Token 预算**: 设置每请求最大 token，超出时降级 Compact

---

## ④ 技能关联

### 前置技能

- **16-智能体工程 [[Skill-MCP-A2A-Protocol-Stack]]**(P1-4): MCP 协议基础，tool 描述格式
- **16-智能体工程 [[Skill-Open-Source-Tool-Use-Model]]**(P2-6): Hermes 4 的 tool_call 格式可直接用于审核

### 延伸技能

- **16-智能体工程 [[Skill-MCP-Tool-Use-Benchmark]]**(P3): MCPAgentBench 评估工具选择能力
- **16-智能体工程 [[Skill-SLM-Tool-Calling-Optimization]]**(P2-7): SLM 对 tool 描述质量更敏感

### 可组合技能

- **16-智能体工程 [[Skill-Auto-Skill-Synthesis]]**(P0-1): SkillForge 生成的 tool 描述可用本技能审核
- **16-智能体工程 [[Skill-Context-Compression]]**(P1-2): Compact 模式是 context compression 的一种形式
- **07-NLP-VOC VOC 标签体系**: 标签定义的质量审核可借鉴六维 rubric

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| Tool 描述质量审核 | 准确率 +5-10pp, 错误调用 -30% | 工程 2 周 + FM API $200 | 20-30x |
| 动态描述路由 | Token 开销 -40%, 延迟 -20% | 工程 1 周 | 15-20x |
| 新 tool 上线审核 | 上线缺陷 -50% | 集成到 CI/CD 1 天 | 长期收益 |

### 实施难度

**评分: ⭐⭐⭐☆☆ (3/5 星)**

- 数据要求: 低，直接扫描现有 MCP tool 描述
- 技术门槛: 中，需要理解 MCP 协议 + FM prompt engineering
- 工程复杂度: 中，六维评分 + 增强 + 路由三层架构
- 维护成本: 中低，FM 评分可自动化，只需定期校准

### 优先级评分

**评分: ⭐⭐⭐⭐☆ (4/5 星)**

- **普适性强**: 任何使用 MCP 的系统都受益
- **成本收益明确**: 准确率提升直接转化为业务价值
- **可渐进实施**: 从扫描开始，逐步加入增强和路由
- **预防性价值**: 在 tool 上线前发现描述缺陷
- **风险**: 增强可能引入回归，需要 A/B 测试验证

### 评估依据

1. **大规模实证**: 856 tools, 103 servers 的系统分析
2. **量化影响**: 明确给出 accuracy vs cost trade-off 数据
3. **可操作**: 提供 scanner + augmentor + router 完整工具链
4. **开源**: 复现包已发布 github.com/SAILResearch/mcp-tool-description-augmentation
5. **学术严谨**: ICC 评分一致性验证 + 统计显著性检验

---

## 参考论文

1. **Model Context Protocol (MCP) Tool Descriptions Are Smelly!** (2026-02)
   - Hasan, Li, Rajbahadur, Adams, Hassan (Queen's University, Canada)
   - 核心贡献: 六维评分 rubric + FM smell scanner + 组件消融 + 动态路由
   - arxiv: [2602.14878](https://arxiv.org/abs/2602.14878)
   - 复现包: [github.com/SAILResearch/mcp-tool-description-augmentation](https://github.com/SAILResearch/mcp-tool-description-augmentation)

## 相关基础

- **MCP**: Model Context Protocol (Anthropic) — tool 调用标准
- **Code Smells**: Fowler & Beck 提出的代码坏味道概念
- **Prompt Engineering**: tool 描述本质是 prompt 设计
- **MCP Universe**: MCP 基准测试框架
- **ICC (Intraclass Correlation)**: 评分者间一致性度量
- **Cochran's Q Test**: 多配对二分类结果比较

---

## 与同领域 Skill 的对比

| 维度 | MCP Smell Scanner (本) | 手动审核 | 自动文档生成 |
|------|----------------------|---------|-------------|
| 覆盖范围 | 六维结构化评分 | 主观、不一致 | 仅生成，不评估 |
| 可扩展性 | 自动扫描任意 MCP server | 人工成本高 | 高 |
| 精准度 | ICC 验证，多 FM 交叉 | 依赖专家经验 | 中等 |
| 可操作性 | 直接输出修复建议 | 需人工整理 | 需人工审核 |
| 成本 | FM API 调用费 | 人力成本 | 训练/推理成本 |

**互补使用**:
- **开发阶段**: Smell Scanner 自动审核 + 人工确认
- **上线前**: 跑 A/B 测试验证增强效果
- **运行时**: Tool Description Router 按场景切换描述版本
- **维护**: 定期重新扫描，检测描述腐化
