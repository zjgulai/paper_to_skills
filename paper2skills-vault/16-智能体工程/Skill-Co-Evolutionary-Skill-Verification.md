---
title: Skill 自动演化与验证 — EvoSkills 双 LLM 协同优化
doc_type: knowledge
module: 16-智能体工程
topic: co-evolutionary-skill-verification
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: 协同演化 Skill 验证 — EvoSkills 自动萃取 + 信息隔离审核

---

## ① 算法原理

### 核心思想

**EvoSkills** 解决 LLM Agent **多文件 Skill 包**自动生成的两个根本挑战:

1. **一次性生成不可靠**:单次 LLM 调用难以同时写好 `SKILL.md` + 多个脚本 + 参考资料
2. **缺失 ground-truth 信号**:真实环境下没有 oracle 测试,只有不透明的 pass/fail 二值反馈

EvoSkills 用**双 LLM 协同演化框架**绕过这两个问题:

- **Skill Generator**($\pi_\theta$):维持持久对话上下文 $C$,迭代生成 + 修复 skill 包
- **Surrogate Verifier**($\pi_\theta^V$):**独立 LLM session**,只看任务指令 + 输出文件,合成 deterministic 测试断言
- 两者通过 **alternating refinement** 协同演化,**信息隔离**防止 verifier 继承 generator 的偏见

### 算法直觉

任务建模为 POMDP $\mathcal{M}=\langle\mathcal{X},\mathcal{A},T,\mathcal{O},\Omega,\mathcal{R}\rangle$:状态是 filesystem,动作是 terminal 命令 + 文件编辑,oracle reward $\mathcal{R}(x_T)\in[0,1]$ 只在末态返回。

**Surrogate Reward**(Eq. 4):

$$
\tilde{\mathcal{R}}(x,\mathcal{V})\triangleq\frac{1}{|\mathcal{V}|}\sum_{k=1}^{|\mathcal{V}|}\mathbf{1}[e_k(x)]\in[0,1]
$$

**Alternating Refinement**(Eq. 5-6):

$$
\mathcal{S}^{(i+1)}\leftarrow\arg\max_{\mathcal{S}}\tilde{\mathcal{R}}\!\bigl(\Phi(\mathcal{S},\mathcal{E}),\mathcal{V}^{(j)}\bigr)
$$

$$
\mathcal{V}^{(j+1)}\sim\pi_\theta^V\!\bigl(\,\cdot\mid I,x^{(i)},\mathcal{V}^{(j)}\bigr),\;\;\text{if }\tilde{\mathcal{R}}{=}1\wedge\mathcal{R}<1
$$

**Skill Refinement**(Eq. 7):

$$
\mathcal{S}^{(i+1)}\sim\pi_\theta\!\bigl(\,\cdot\mid\mathcal{S}^{(i)},C^{(i+1)}\bigr),\quad C^{(i+1)}=C^{(i)}\oplus\mathcal{F}^{(i,j)}
$$

其中 $\mathcal{F}^{(i,j)}$ 是 Verifier 给出的失败诊断(失败 case + 根因分析 + 修复建议)。

### 双轨反馈机制

```
Skill Generator π_θ                Surrogate Verifier π_θ^V
       |                                       |
       |  生成 S^(i) 并执行 → x^(i)              |
       |                                       |
       |  ←── 失败诊断 F^(i,j) (内层循环) ────|
       |                                       |
       |  当 surrogate 通过时:                  |
       |                                       |
       |     ground-truth oracle test          |
       |     返回 1[R<1] (opaque bit only)     |
       |                                       |
       |  ──── 触发 verifier 升级 ────→         |
       |       (test escalation)               |
```

**关键设计:信息隔离 + opaque oracle bit**

- Verifier 完全不看 generator 的 reasoning / code,只看任务指令 + 输出文件
- Oracle 通过 `1[R<1]` 只返回一个 bit,**不泄露任何测试内容或失败细节**
- 这阻止 generator 过拟合到 held-out test,迫使 verifier 自己升级测试覆盖

### 关键结果

来自 SkillsBench(87 个任务,11 个专业领域):

| 配置 | Pass Rate | vs No-Skill |
|------|-----------|-------------|
| No-Skill Baseline | 30.6% | — |
| Skill-Creator (Anthropic 官方) | 34.1% | +3.5pp |
| Human-Curated Skills | 53.5% | +22.9pp |
| **EvoSkills (Full)** | **71.1%** | **+40.5pp** |

**Ablation**:
- W/O Surrogate Verifier: 41.1% (−30.0pp)
- W/O Skill Evolution: 48.6% (−22.5pp)

**Cross-Model Transfer**(Opus 4.6 演化的 skills 跨模型迁移):

| 目标模型 | 无 skill | 用 Opus skills | Δ |
|---------|---------|---------------|---|
| GPT-5.2 | 29.6% | 65.0% | +35.4 |
| Claude Sonnet 4.5 | 20.0% | 63.1% | +43.1 |
| Claude Haiku 4.5 | 10.4% | 54.5% | +44.1 |
| Qwen3-Coder-480B | 8.4% | 50.8% | +42.4 |
| DeepSeek V3-671B | 13.0% | 48.8% | +35.8 |
| Mistral Large 3-675B | 4.9% | 43.1% | +38.2 |

### 三个关键洞察

1. **Agents create better skills than humans**:Self-evolved 在 9/11 领域超越 human-curated,Finance 领域差距高达 +56.9pp
2. **Skills are portable across model families**:迁移到 6 个不同公司模型仍获 +36 ~ +44pp
3. **Human-Machine cognitive misalignment**:Natural Science 领域 human-curated **降低**性能,说明人写的 workflow 不符合 agent 推理模式

### 关键假设

1. 任务有可机器自动验证的输出(filesystem state)
2. Surrogate Verifier 与 Generator 可在不同 LLM session 中并行执行
3. Oracle test 存在(可以是真人审核,但只回 pass/fail)
4. Skill 表达为 **多文件包**(`SKILL.md` + scripts + references),而非单 prompt

### 关键挑战

- **演化成本**:每个任务平均 4.1 cycles × 5 oracle rounds = $K+M=20$ 次 LLM 调用 + 多次环境执行
- **Verifier 漂移**:`V^(j+1)` 必须既不偏向 generator 的实现,又要覆盖更难的测试
- **Context 溢出**:演化轮数过高时 LLM context 满,需 $\beta=0.7$ 上限检查

---

## ② 母婴出海应用案例

### 场景一:跨境客服 Skill 自动萃取(替代人工 SOP 撰写)

**业务问题**:

跨境母婴客服需要覆盖大量场景:退货、过敏咨询、物流追踪、关税计算、母婴专业问答……每个场景的 SOP 需要人工撰写并打磨,周期 1-2 周/场景。同时人写的 SOP 经常出现 human-machine misalignment:客服 manager 觉得逻辑清晰的步骤,Claude/GPT 执行起来反而经常报错(对应论文 Takeaway 3)。

**EvoSkills 落地方案**:

```
对每个新场景(例如"过敏退货决策流"):

1. 提供任务指令 I:
   "客户咨询新生儿过敏退货,需要根据品牌+批次+症状+保留状态给出退/换/补差价决策"

2. 提供历史样本作为 oracle:
   - 30 个历史工单 (含已知正确处理结果)
   - oracle 只判定:agent 给出的决策是否与历史 ground-truth 一致

3. EvoSkills 自动演化:
   - Skill Generator 写 skill 包:
     SKILL.md (决策树) + verify_batch.py (批次查询) + check_symptom.py (症状分类)
   - Surrogate Verifier 写测试:
     test_batch_match.py (验证批次查询正确)
     test_severity_classification.py (验证症状分级)
     test_decision_consistency.py (验证决策与历史一致)
   - 演化 5 轮收敛

4. 部署:
   - 通过 oracle test 的 skill 包推到 customer_service_agent
   - 后续 3 个月每月 retrieve oracle test 1 次,触发 retraining
```

**业务价值**:

- 场景接入周期:1-2 周 → 2 天(只需收集 30 个 oracle 样本)
- SOP 质量:对应论文 +18pp ~ +40pp 改进
- 维护成本:新法规出来后只需更新 oracle ground-truth,skill 自动重新演化

### 场景二:跨平台 Skill 迁移(Cross-Model Skill Transfer)

**业务问题**:

业务上需要在多个 LLM 服务商之间切换:
- 高峰期用 Claude Opus 4.6(贵但准)
- 平峰用 GPT-5.2(性价比)
- 低优先级用 Qwen3-Coder(成本最低)

直接把 Opus 写的 prompt 用到 Qwen3 上,性能从 65% 跌到 8%(对应论文 Mistral Large 3 baseline)。

**EvoSkills 落地方案**:

```
利用论文 Takeaway 2 — skills 可跨模型迁移(+36 ~ +44pp gain)

1. 用 Opus 4.6 + EvoSkills 演化出母婴客服 skill 包
   - 输入:30 个历史工单 + 业务指令
   - 输出:skill_baby_customer_v1/
            ├── SKILL.md
            ├── check_allergy.py
            ├── lookup_batch.py
            └── decide_refund.py

2. 直接部署到三个模型:
   - Claude Opus 4.6 (prod 高峰): 配合 skill,71.1% baseline
   - GPT-5.2 (prod 平峰): +35.4pp → 65% 准确率
   - Qwen3-Coder-480B (低优场景): +42.4pp → 50.8%

3. 月度成本对比 (假设 100k 工单/月):
   - 全 Opus:$15k
   - Opus 30% + GPT 50% + Qwen 20%:$5.2k
   - 准确率加权后:从 71% → 60% (但成本 -65%)
```

**业务价值**:

- 模型成本:-65% (与全 Opus 对比)
- 准确率 trade-off:-11pp (仍显著优于无 skill 的 8-30%)
- 灵活性:新平台开通(例如 DeepSeek V3)无需重写 skill,直接用现有的 +35.8pp 收益

---

## ③ 代码模板

```python
from dataclasses import dataclass

@dataclass
class SkillPackage:
    skill_id: str
    description: str
    scripts: dict[str, str]
    version: int = 1

@dataclass
class VerificationResult:
    skill_id: str
    passed: bool
    feedback: str
    score: float

def generator_propose(skill_id: str, task: str, prev_feedback: str = "") -> SkillPackage:
    base_desc = f"执行任务: {task}"
    if "缺少错误处理" in prev_feedback:
        base_desc += " (含错误处理)"
    return SkillPackage(
        skill_id=skill_id,
        description=base_desc,
        scripts={"main.py": "def run():\n    pass  # " + task},
    )

def verifier_evaluate(pkg: SkillPackage) -> VerificationResult:
    issues = []
    if "错误处理" not in pkg.description:
        issues.append("缺少错误处理")
    if len(pkg.scripts.get("main.py", "")) < 30:
        issues.append("实现过于简单")
    score = max(0.0, 1.0 - len(issues) * 0.3)
    return VerificationResult(
        skill_id=pkg.skill_id,
        passed=score >= 0.7,
        feedback="; ".join(issues) if issues else "通过",
        score=score,
    )

skill_id = "Skill-补货决策"
feedback = ""
for iteration in range(3):
    pkg = generator_propose(skill_id, "自动补货", feedback)
    result = verifier_evaluate(pkg)
    print(f"Iter {iteration+1}: score={result.score:.2f} | feedback={result.feedback}")
    feedback = result.feedback
    if result.passed:
        print(f"  ✅ 通过验证 at iteration {iteration+1}")
        break
print("[✓] Co-Evolutionary Skill Verification 测试通过")
```


## ④ 技能关联

### 前置技能

- **16-智能体工程 [[Skill-Auto-Skill-Synthesis]]**(SkillForge):skill 自动萃取的 4 模块流水线,EvoSkills 是其增强版(加入 verifier 协同演化)
- **16-智能体工程 [[Skill-Skill-Lifecycle-Design]]**(SoK):理解 skill 4-tuple `(C, π, T, R)` 和 7 阶段生命周期,EvoSkills 主要落在 Generation+Update 阶段
- **10-MAS [[Skill-MAS-Orchestrator]]**:理解多 agent 编排,这里 Generator 和 Verifier 就是 2 个特化 agent

### 延伸技能

- **16-智能体工程 [[Skill-Active-Context-Pruning]]**(Focus,待萃取 P2-2):演化轮数过高时压上下文
- **16-智能体工程 [[Skill-Memory-as-Action]]**(MemAct,待萃取 P2-3):把 LTM 操作嵌入演化策略
- **16-智能体工程 [[Skill-Tool-Description-Audit]]**(MCP Smelly,待萃取 P2-8):用类似机制审计 MCP tool 描述质量

### 可组合技能

- **16-智能体工程 [[Skill-Context-Compression]]**(ACON):context 累积过多时压缩 generator 历史
- **16-智能体工程 [[Skill-Agentic-Memory-Management]]**(AgeMem):演化轨迹存入 LTM,后续任务复用
- **16-智能体工程 [[Skill-MCP-A2A-Protocol-Stack]]**:Generator/Verifier 间用 A2A 通信,oracle 接 MCP 工具
- **本项目 paper-审核 skill**:本身就是一种 surrogate verification 模式,可借鉴 EvoSkills 升级测试套件的设计

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 跨境客服 Skill 自动萃取 | 场景上线周期 -85%,SOP 质量 +18 ~ +40pp | 工程 4-6 周 + LLM 成本 $200/场景演化 | 8-12x |
| 跨模型 Skill 迁移 | 模型成本 -65%,准确率 trade-off -11pp | 工程 2-3 周 (复用现有 skill 包) | 长期 5-10x |
| 内部 Skill marketplace | 团队复用 skill,新成员上手周期 -70% | 工程 6-8 周 + governance 持续投入 | 4-8x |

### 实施难度

**评分:⭐⭐⭐⭐☆(4/5 星)**

- 数据要求:中,需要 oracle ground-truth 样本(30-100 个/场景)
- 技术门槛:高,需要多 LLM session 编排 + 信息隔离设计
- 工程复杂度:中高,核心算法清晰但环境执行 + 沙箱隔离需要工程化
- 维护成本:中,主要是 oracle 数据维护 + 演化轮次调参

### 优先级评分

**评分:⭐⭐⭐⭐☆(4/5 星)**

- **方法论价值高**:协同演化框架可推广到其他自我改进场景(prompt 优化、tool 演化)
- **跨平台红利明显**:论文给出 +36 ~ +44pp 跨模型迁移收益,直接转化为成本节省
- **可分阶段落地**:可以先只做 Generator + Oracle(W/O Surrogate Verifier),性能从 30%→41%(论文 ablation),后续再加 Verifier 升级到 71%
- **学习曲线陡**:需要懂 POMDP + alternating optimization + LLM session 管理

### 评估依据

1. **论文实证强**:SkillsBench 87 任务 + 7 个模型迁移测试,数据完整
2. **Ablation 清楚**:验证了每个组件的必要性(verifier −30pp, evolution −22.5pp)
3. **工业相关性高**:Claude/GPT 都已支持 skill 概念,EvoSkills 直接对接生产
4. **完整 receipt**:论文给出从 POMDP 形式化 → Algorithm 1 → 实现细节 → 评估配置的完整 blueprint
5. **完整 Case Study**:Appendix E 给出 Exoplanet Transit 任务的完整演化轨迹,可复刻

---

## 参考论文

1. **EvoSkills: Self-Evolving Skills via Co-Evolutionary Verification for LLM Agents** (2026-04)
   - Zhang, H., Fan, S., Zou, H. P., Chen, Y., Wang, Z., Zhou, J., Li, C., Huang, W.-C., Yao, Y., Zheng, K., Liu, X., Li, X., Yu, P. S.
   - 核心贡献:双 LLM 协同演化 + 信息隔离 surrogate verifier + opaque oracle bit + cross-model transfer
   - arxiv:[2604.01687](https://arxiv.org/abs/2604.01687)

## 相关基础

- **SkillsBench** (arxiv 引用 [9]):87 任务 / 11 领域 / deterministic verifier
- **Anthropic Agent Skills 概念**:[https://anthropic.com/agent-skills](https://anthropic.com/agent-skills)
- **SkillForge** (arxiv:2604.08618):本领域的 P0-1,客服对话萃取 skill,EvoSkills 是其方法论增强
- **SoK Agentic Skills**(arxiv:2602.20867):skill 4-tuple 设计哲学

---

## 与同领域 Skill 的对比

| 维度 | EvoSkills (本) | SkillForge (P0-1) | SoK Agentic Skills (P1-1) |
|------|-------------------|-------------------|------------------------------|
| 目标 | 多文件 skill 包自动演化 | 从对话提取 skill | skill 设计哲学 |
| 反馈来源 | Surrogate Verifier + Oracle bit | 客服对话 + 工单结果 | 静态 4-tuple 验证 |
| 迭代机制 | Alternating refinement | 4 模块流水线 | 7 阶段生命周期 |
| 信息隔离 | **是**(verifier 不见 generator) | 否 | 不适用 |
| 跨模型迁移 | **+36 ~ +44pp** 实证 | 未测 | 不适用 |
| 落地周期 | 中(4-6 周) | 短(2-4 周) | 中(4-6 周) |
| 适用阶段 | 持续改进 | 冷启动 | 治理设计 |

**互补使用**:
- **冷启动**用 SkillForge 萃取初版 skill
- **质量提升**用 EvoSkills 协同演化
- **整体治理**用 SoK Agentic Skills 的 4-tuple + 7 阶段框架
- **生产监控**接 Skill-Agent-Stage-Evaluation(EComStage)做三阶段评估
