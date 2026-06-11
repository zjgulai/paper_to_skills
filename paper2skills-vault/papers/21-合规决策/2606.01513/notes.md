---
paper_id: "2606.01513"
title: "Compliance-Scored Best-of-N Guardrail Orchestration for Multimodal Document Generation in Payments Dispute Defense"
authors:
  - "Nataraj Agaram Sundar"
  - "Tejas Morabia"
source_url: "https://arxiv.org/abs/2606.01513"
pdf_path: "paper.pdf"
domain: "compliance"
decision: "selected"
selected_at: "2026-06-07"
selection_reason: "比 arXiv 搜索排名第一的 2606.03271 更贴近合规决策域；可直接转化为可验证的合规评分、Best-of-N 选择和人工复核路由代码模板。"
---

# Reading Notes

## 核心问题

企业级文档生成不是开放式写作，而是 schema、政策、隐私和延迟共同约束的生产系统。论文指出，传统做法常把 PII 检测、内容审核、格式校验作为独立后处理服务拼接，导致延迟叠加、规则解释分散、失败后重试成本高。

## 方法摘要

论文提出统一 guardrail orchestration layer：

- 多个生成 head 并行产生候选输出。
- 每个候选被一组加权 guardrail 评分：PII、moderation、schema、domain rules。
- 计算归一化 compliance score：`C(y)=sum(w_j*g_j(y))/sum(w_j)`。
- 若最佳候选达到阈值 `tau`，提前退出；否则继续尝试直到超时或转人工复核。
- 文本和图片证据先统一标准化，再进入同一个隐私/安全/合规评分面。
- 同步路径只保留 enforcement 和 scoring；telemetry、dashboard、policy config feedback 放到异步路径。

## 论文证据

- 公开 operational readout：典型运行 5 次候选尝试、20 秒请求预算、91% compliance。
- payments dispute defense aggregate readout：总体 variable cohorts 相比 controls 高 11.0 个百分点，95% CI [6.6, 15.5]，p < 0.001。
- adjusted item-not-received 子集高 7.5 个百分点，95% CI [0.2, 15.7]，p = 0.045。
- 论文明确说明这些是 operational scenario comparisons，不是随机 A/B test；不能把全部 lift 归因为 guardrail layer。
- Responsible-AI evidence review：770 generated-evidence reviews 和 70-case OCR slice。

## 对 paper2skills 的适配判断

保留论文的系统思想，不照搬 payments 业务：

- 原始场景：支付争议摘要和 evidence package。
- 迁移场景：母婴出海 Listing、合规说明、客服回复、召回/认证行动摘要。
- 关键价值：把“能否输出”从 LLM prompt 风格问题，变成可审计的合规分数、拒绝原因、人工复核路由和配置版本问题。

## 抽取边界

- PDF 已保存到 `paper.pdf`。
- 已使用 `pypdf` 从 PDF 抽取文本用于核验核心公式、算法伪代码和结果段。
- 代码模板实现的是论文核心算法骨架，不包含真实 OCR、外部 PII 服务或企业级 moderation 模型；这些在生产中应作为可插拔 guardrail checker 接入。
