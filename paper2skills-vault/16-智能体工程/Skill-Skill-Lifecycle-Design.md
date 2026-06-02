---
title: SoK Agentic Skills — Agent Skill 全生命周期与方法论底座
doc_type: knowledge
module: 16-智能体工程
topic: skill-lifecycle-design
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
---

# Skill Card: SoK Agentic Skills — Skill 全生命周期与方法论底座

---

## ① 算法原理

### 核心思想

**SoK Agentic Skills**(Systematization of Knowledge)是 Agent Skill 领域第一篇综合 survey,把分散在 Voyager / CodeAct / Reflexion / Claude Skills / GPT Store / MCP 等系统中的"Skill 概念"统一为一个理论框架。它解决三个根本问题:

1. **Skill 到底是什么** —— 用 4 元组形式化定义,与 tool/plan/memory/prompt 区分开
2. **Skill 怎么演化** —— 用 7 阶段生命周期把所有现有系统映射到统一图谱
3. **Skill 怎么落地** —— 用 7 个设计模式 + Representation×Scope 双轴 taxonomy 系统化实现选择

### 数学直觉

**Skill 的 4 元组形式化**:

$$
S = (C, \pi, T, R)
$$

- $C: O \times G \rightarrow \{0, 1\}$ —— **applicability condition**(适用条件):observation × goal → 是否适用
- $\pi: O \times H \rightarrow A \cup \Sigma$ —— **executable policy**(执行策略):observation × history → action 或 子 skill 调用
- $T: O \times H \times G \rightarrow \{0, 1\}$ —— **termination condition**(终止条件):何时结束
- $R = (\text{name}, \text{params}, \text{returns})$ —— **callable interface**(可调用接口):程序化签名

**与 RL options framework 对应**:论文里的 $(C, \pi, T)$ 对应 Sutton 的 $(I, \pi, \beta)$,但增加了 $R$ 让 skill 可被外部 orchestrator 显式调用——这是 LLM Agent 与 RL options 的关键区别。

### 生命周期 7 阶段

```
Discovery → Practice/Refinement → Distillation → Storage 
   ↑                                                    ↓
Evaluation/Update ← Execution ← Retrieval/Composition
```

虚线反馈:Evaluation → Practice(失败时);Retrieval → Storage(索引失效);Execution → Discovery(运行时缺失)。

### 7 个设计模式

| # | 模式 | 代表 | 表征 | 优势 | 主要风险 |
|---|------|------|------|------|---------|
| P1 | **Metadata-Driven Disclosure** | Claude Code, LangChain, Semantic Kernel | NL+metadata | token 高效 | metadata 投毒 |
| P2 | **Code-as-Skill** | Voyager, CodeAct, SWE-agent | Code | 确定性,可测 | 代码注入 |
| P3 | **Workflow Enforcement** | TDD agents, LATS | NL+rules | 可审计 | rule bypass |
| P4 | **Self-Evolving Libraries** | Voyager, CRADLE, DEPS | Code+NL | 自适应 | 蒸馏污染 |
| P5 | **Hybrid NL+Code Macros** | Claude skills, ReAct | NL+code+refs | 灵活可读 | NL/code 边界歧义 |
| P6 | **Meta-Skills** | Self-Instruct, skill generators | NL/hybrid | 规模化生成 | 递归错误放大 |
| P7 | **Marketplace Distribution** | GPT Store, MCP servers, npm/pip | Any(packaged) | 生态增长 | 供应链攻击(ClawHavoc) |

### 关键实证发现

来自 SkillsBench 锚定研究:

- **curated skills:+16.2pp** 提升 agent pass rate(对照无 skill 基线)
- **self-generated skills: -1.3pp** 平均**下降**(零样本自生成会编码错误启发式)
- **小模型 + curated skills > 大模型 + 无 skill**:procedural memory 是"模型规模的部分替代"

### 安全洞察:ClawHavoc 案例

近 **1200 个恶意 skill** 渗透某主流 Agent marketplace,通过 skill payload 实施:

- API key / 加密货币钱包 / 浏览器凭证 大规模窃取
- prompt injection 借助 skill 描述文本
- 供应链攻击(P7 模式天然脆弱)

这是 Skill 时代的"npm typo-squatting + Log4Shell"组合。

### 关键假设

1. Skill 可以从 plan/tool/memory 中独立出来作为一等抽象
2. 4 元组形式化能覆盖现有所有 Skill 表征
3. 生命周期 7 阶段非严格线性,允许回流
4. Curated 与 self-generated 的边界由"verifiability + iteration"决定

---

## ② 母婴出海应用案例

### 场景一:用 4 元组重构本项目现有 Skill 库的质量

**业务问题**:

本项目 `paper2skills-vault/` 当前有 15+ 领域 80+ Skill 卡,但每张卡的"何时用、怎么用、何时停、怎么调用"散落在各模块中,没有统一的契约。新人 onboarding 或 Agent 自动选 skill 时,缺乏明确的 trigger 条件,误用率高。

**数据要求**:

- 现有 Skill 卡 80+ 篇 markdown(模块 ① 算法原理 + ② 应用案例)
- 实际调用日志(如果有)
- 错误案例(skill 误用、用错时机)

**预期产出**:

每张 Skill 卡新增统一"4 元组 header":

```yaml
skill_contract:
  C (when_to_use):
    - "用户咨询包含[品类]纸尿裤[尺码]咨询"
    - "可用历史会话 ≥ 2 turn"
  pi (policy):
    - "Step 1: 查询体重/月龄"
    - "Step 2: 调用 size_calculator"
    - "Step 3: 输出 SKU + 备选"
  T (terminate_when):
    - "用户确认推荐"
    - "或工具调用失败 2 次"
  R (interface):
    name: skill_diaper_size_consult
    params:
      baby_weight_kg: float
      baby_age_months: int
    returns: list[SKU]
```

**业务价值**:

- 误用率下降:Skill 选择从"凭感觉"到"check $C$",误用率预期降 40-60%
- 多 Agent 协作可能:有了 $R$,主 Agent 可程序化调用 sub-agent 的 skill
- 审核标准化:本项目 `paper-审核` 流程可以 check 4 元组完整性

---

### 场景二:本项目 Skill 库的 7 模式 + 安全审计

**业务问题**:

本项目跨境母婴的 Skill 库可能将来开放给商家/分销商(P7 marketplace 模式),需要在开放前完成安全审计,避免 ClawHavoc 式攻击。

**数据要求**:

- 现有所有 Skill 卡及关联 Python 代码模板
- 7 模式分类标准(论文 Table III)
- 供应链风险 checklist

**预期产出**:

```
Skill 库审计报告 (2026-05):

P1 Metadata-Driven (主体): 80+ Skill 卡均符合, 用 description 触发
  风险: 跨境多语言下 description 译错导致 wrong trigger
  缓解: 强制中英双语 description + 一致性校验

P2 Code-as-Skill (代码模板): 80+ Python 模板, 全部 deterministic
  风险: 模板调用第三方 API (如 BM25, scikit-learn) 的版本依赖
  缓解: 锁版本 + 沙箱执行

P5 Hybrid NL+Code: 本项目核心模式 (markdown skill 卡 + python 模板)
  风险: NL 描述与 Code 实现脱钩 (comment rot)
  缓解: paper-审核 流程加 "code-comment 一致性校验"

P7 Marketplace: 尚未开放, 但 GitHub repo 已对外
  风险: PR 投毒 (恶意 Skill 卡 + 隐藏 prompt injection)
  缓解: PR review 强制 require trust tier + 4 元组校验
```

**业务价值**:

- 安全前置:发现"description 多语言一致性"是高风险,提前缓解
- 模式自省:确认本项目主体是 P1+P2+P5 组合,与 Claude Code 风格一致
- 准备 P7 开放:在开放给商家前完成必要审计,避免重蹈 ClawHavoc 覆辙

---

## ③ 代码模板

代码位置:`paper2skills-code/llm_agent_engineering/skill_lifecycle_design/skill_contract.py`

核心组件:

- `SkillContract`:4 元组数据结构 $(C, \pi, T, R)$
- `LifecycleStage`:7 阶段枚举(Discovery → Update)
- `DesignPattern`:7 模式枚举
- `SkillRegistry`:本项目 Skill 库的中央注册器(支持 4 元组查询)
- `SkillAuditor`:7 模式分类 + 安全风险评估
- `SkillSelector`:基于 $C$(applicability) 选择 skill 的简单 retriever

运行方式:

```bash
cd paper2skills-code/llm_agent_engineering/skill_lifecycle_design
python skill_contract.py
```

生产环境建议:

1. 把 `SkillContract` 嵌入到本项目所有 `Skill-*.md` 的 frontmatter 中
2. CI 流程加入 `SkillAuditor.audit_all()`,阻止不符合 4 元组的 Skill 合入
3. `paper-审核` skill 升级,加入"模式分类 + 安全风险评分"
4. `paper-同步` 时,把 4 元组写入 sync_status.json 元数据
5. 长期建立 marketplace(P7) 时,实施 trust tier 分层

---

## ④ 技能关联

### 前置技能

- **基础**:对 Voyager/CodeAct/Claude Code 的基本理解
- **10-MAS [[Skill-Skill-Registry-Dynamic-Loading]]**:本项目已有的 skill 注册器基础

### 延伸技能

- **16-智能体工程 [[Skill-Auto-Skill-Synthesis]]**(SkillForge):Discovery + Distillation + Update 的具体实现
- **16-智能体工程 [[Skill-Co-Evolutionary-Skill-Verification]]**(EvoSkills,P2):验证机制
- **16-智能体工程 [[Skill-Tool-Description-Audit]]**(MCP Smelly):metadata 质量审计

### 可组合技能

- **本项目 paper-审核 skill**:加入 4 元组完整性 + 7 模式分类检查
- **本项目 paper-同步 skill**:同步 4 元组元数据到平台
- **10-MAS [[Skill-MAS-Orchestrator]]**:基于 $R$ 程序化调度 sub-skill

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 4 元组重构 80+ Skill 卡 | 误用率 -40-60%, Skill 调用质量提升 | 工程 4-6 周(批量重构) | 8-12x |
| 7 模式安全审计 | 提前发现 marketplace 模式 3-5 个高风险点 | 工程 2 周 + 安全 review | 15-20x(避免一次事故) |
| paper-审核 升级 | 自动化质量门禁, 减少人工审核 40% | 工程 2 周 | 6-10x |

### 实施难度

**评分:⭐⭐⭐☆☆(3/5 星)**

- 数据要求:低,主要是元数据建模
- 技术门槛:低,主要是文档工程 + Python 简单注册器
- 工程复杂度:中,涉及现有 80+ Skill 卡的批量重构
- 维护成本:低,4 元组一旦建立可长期使用

### 优先级评分

**评分:⭐⭐⭐⭐⭐(5/5 星)**

- **方法论底座**:为本项目所有 16+ 领域提供统一抽象,后续所有 Skill 卡受益
- **质量护城河**:与 SkillsBench 实证(curated +16.2pp vs self -1.3pp)形成对照,本项目 curated 路线得到背书
- **安全前置**:ClawHavoc 案例提示开放前必须做的事
- **可立即落地**:不需训练 / 不需 RL,纯文档 + 注册器工程

### 评估依据

1. **理论完备**:4 元组对接 RL options + ACT-R + STRIPS 经典 AI,代际继承性强
2. **实证锚定**:SkillsBench +16.2pp / -1.3pp 给出了 curated vs self-generated 量化对照
3. **安全案例真实**:ClawHavoc 1200 恶意 skill 提供具体威胁模型
4. **与项目契合**:本项目正是 P1+P2+P5 模式,survey 直接适用

---

## 参考论文

1. **SoK: Agentic Skills — Beyond Tool Use in LLM Agents** (2026-02)
   - Jiang, Y., Li, D., Deng, H., Ma, B., Wang, X., Wang, Q., Yu, G. — University of Technology Sydney + CSIRO Data61
   - 核心贡献:Skill 4 元组形式化 + 7 阶段生命周期 + 7 设计模式 + Representation×Scope taxonomy + ClawHavoc 安全分析
   - arxiv:[2602.20867](https://arxiv.org/abs/2602.20867)

## 相关基础

- **Voyager** (arxiv:2305.16291):Minecraft 自动 skill discovery
- **CodeAct** (arxiv:2402.01030):Code-as-skill 范式
- **Claude Code Skills**:P1 metadata-driven disclosure 工业实现
- **SkillsBench** (arxiv:2602.12670):curated vs self-generated 量化对照
- **Sutton et al. options framework**:RL 经典理论锚

---

## 与同领域 Skill 的对比

| 维度 | SoK Agentic Skills | SkillForge(P0-1) | EComStage(P0-3) |
|------|---------------------|--------------------|--------------------|
| 类型 | Survey + 方法论框架 | 工业落地系统 | 评估 benchmark |
| 时间维度 | 全生命周期 7 阶段 | Discovery → Update 闭环 | 单次评估 |
| 抽象层级 | 最高(定义 + taxonomy) | 中(框架 + pipeline) | 低(任务级) |
| 立即落地 | 中(需重构 80+ Skill 卡) | 是(可单场景试点) | 是(2 周可上线) |

**互补使用**:
- **先用 SoK** 建立本项目 Skill 库的统一 4 元组契约 + 安全审计框架
- **再用 SkillForge** 在每个领域具体实施 Discovery → Distillation → Update 闭环
- **同时用 EComStage** 量化评估每个 Skill 的能力,反馈到 Update 阶段
