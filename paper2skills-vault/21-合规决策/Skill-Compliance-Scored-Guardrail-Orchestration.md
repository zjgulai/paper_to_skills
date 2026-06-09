---
title: Compliance-Scored Guardrail Orchestration — 合规评分 Best-of-N 守护编排
doc_type: knowledge
module: 21-合规决策
topic: compliance-scored-guardrail-orchestration
status: verified
created: 2026-06-07
updated: 2026-06-07
owner: self
source: arxiv:2606.01513
roadmap_phase: phase1
---

# Skill-Compliance-Scored-Guardrail-Orchestration

---

## ① 算法原理

**核心思想**：把 LLM 自动生成的合规风险控制从“生成后人工看一眼”升级为同步的加权评分系统。系统并行生成多个候选输出，对每个候选运行 PII、内容安全、schema、领域规则和证据引用检查，计算合规得分；一旦最佳候选超过阈值就提前返回，否则进入人工复核。

**数学直觉**：

```text
C(y) = sum_j w_j * g_j(y) / sum_j w_j
```

`g_j(y)` 是规则 j 对候选输出 y 的通过程度，范围 `[0,1]`；`w_j` 是规则严重度权重。高权重规则（如 PII 泄漏、违规承诺、缺强制安全字段）对最终分数影响更大。输出不只包含文本，还包含 attempts、latency、规则失败原因和 config version，方便审计。

**关键假设**：

- 合规规则可以拆成可执行 checker，并能返回 `[0,1]` 分数。
- 自动发布阈值必须由业务和法务共同校准。
- 低于阈值不等于失败，而是转入 human review。
- 论文公开结果是 operational scenario comparison，不是随机 A/B test；业务迁移时应重新验证。

---

## ② 母婴出海应用案例

**场景 A：Listing 合规文案自动发布门控**

- **业务问题**：AI 自动生成 Amazon/TikTok Shop Listing 时，容易写出“guaranteed safe”“no certification needed”等违规承诺，也可能泄露供应商联系人或测试报告中的 PII。
- **数据要求**：候选 Listing 文案、目标市场、品类、供应商测试报告摘要、图片 OCR 文本、平台政策规则。
- **预期产出**：
  - `compliance_score`，如 0.92。
  - 命中规则：PII、schema、CPSC/FDA/EU GPSR 规则、证据引用。
  - 决策：`accepted` / `human_review_required`。
  - 审计元数据：attempts、threshold、violations。
- **业务价值**：把合规团队从逐条初筛转为复核低分项。若每日 300 条 AI 文案中 70% 自动通过，每条节省 4 分钟人工初筛，每月约节省 840 小时，同时降低违规上架概率。

**场景 B：召回/认证行动摘要的人工复核路由**

- **业务问题**：法规变更或召回事件进入系统后，需要快速生成“对哪些 SKU 有影响、下一步做什么”的行动摘要；但这类摘要不能遗漏证据来源，也不能把风险描述成确定结论。
- **数据要求**：CPSC/RAPEX 更新、SKU 类目映射、测试报告、历史召回记录、客服投诉摘要。
- **预期产出**：
  - 通过 Best-of-N 选择最高合规摘要。
  - 若缺证据引用、缺市场字段或含过度承诺，自动转人工。
  - 输出 reviewer 可读的失败原因列表。
- **业务价值**：缩短合规响应时间。对 CRITICAL/HIGH 事件，初稿从 2-4 小时压到 10-20 分钟，人工只处理低分或高危摘要。

---

## ③ 代码模板

**代码路径**：`paper2skills-code/compliance/compliance_scored_guardrail_orchestration/`

```python
from paper2skills_code.compliance.compliance_scored_guardrail_orchestration import (
    ComplianceScoredGuardrailOrchestrator,
    baby_compliance_guardrails,
)

orchestrator = ComplianceScoredGuardrailOrchestrator(
    baby_compliance_guardrails(),
    threshold=0.88,
)

result = orchestrator.select_best([
    {
        "title": "Listing draft",
        "body": "Based on evidence, route the US launch through CPSC safety review.",
        "evidence_summary": "Supplier provided battery report and US/EU target-market plan.",
        "recommended_action": "human_review_before_publish",
    }
])

print(result.as_dict())
```

本模板使用 Python 标准库实现以下能力：

- 加权 guardrail score。
- Best-of-N 候选选择。
- PII / moderation / schema / baby compliance domain-rule 检查。
- 低于阈值时路由到 `human_review_required`。
- 保留 attempts、violations、threshold、elapsed_ms。

已验证命令：

```bash
venv/bin/python -m pytest paper2skills-code/compliance/compliance_scored_guardrail_orchestration -q
```

结果：`3 passed`。

---

## ④ 技能关联

**前置技能**

- [[Skill-Amazon-ToS-Compliance-Guardrail]]：提供平台政策规则和禁语基线。
- [[Skill-Agent-Safety-Guardrails]]：提供 agent 输出安全与运行时拦截基础。

**可组合技能**

- [[Skill-Product-Safety-Testing-Requirements]]：把安全测试字段作为强制 schema 和 domain rule。
- [[Skill-Regulatory-Change-Monitoring]]：把法规变更摘要接入本 Skill 做输出门控。
- [[Skill-Category-Compliance-Prescan]]：新品类进入前生成风险摘要，再由本 Skill 评分后发布给选品团队。

**延伸技能**

- 多模态证据 OCR guardrail：把图片中的证书号、地址、Tracking ID 统一进入 PII/证据评分。
- 配置版本治理：每次 rule set 更新都带 config version，可回溯历史输出为何通过或被拒。
- Reviewer calibration：用人工复核标签反推各 guardrail 权重。

---

## ⑤ 商业价值评估

- **ROI 预估**：以每日 300 条 AI 生成 Listing/客服/合规摘要计算，若 70% 达到自动通过阈值、每条节省 4 分钟人工初筛，则每月节省约 840 小时；按 $25/hour 估算，约 $21,000/月。
- **风险降低**：PII、违规承诺、缺证据引用从人工抽检变成每次输出必检，适合高风险品类和多市场上架。
- **实施难度**：⭐⭐⭐☆☆。标准库版本可立即落地；生产版需要接入 OCR、DLP、moderation 和 trace。
- **优先级评分**：⭐⭐⭐⭐☆。当前合规域已有规则类 Skill，但缺少一个统一的“生成输出门控 + 审计元数据”编排层。
- **评估依据**：论文公开 readout 报告 5 次候选尝试、20 秒预算、91% compliance；业务侧收益主要来自减少人工初筛和减少违规输出，而非直接复用论文中的 payments win-rate。
