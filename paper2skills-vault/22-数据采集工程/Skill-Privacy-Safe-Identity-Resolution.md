---
title: Privacy-Safe Identity Resolution — 隐私合规跨平台 ID 解析：多方对齐与差分隐私
doc_type: knowledge
module: 22-数据采集工程
topic: privacy-safe-identity-resolution

roadmap_phase: phase1
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
---

# Skill Card: Privacy-Safe Identity Resolution — 隐私合规跨平台 ID 解析：多方对齐与差分隐私

## ① 核心算法

**论文**：Sherpa.ai [2604.19219] + Cross-Domain SID [2606.01396]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：跨平台（Amazon+TikTok+独立站）用户 ID 打通，满足 GDPR/PIPL 合规

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/identity_resolution/model.py`

## ④ 技能关联

### 前置技能
- [[无（Layer 1）]]

### 延伸技能
- [[Skill-HGNN-Cross-Device-Matching]]
- [[Skill-CDA-Cookieless-Attribution]]

### 可组合技能
- [[Skill-Identity-Fragmentation-Debiasing]]
- [[Skill-Privacy-Preserving-Federated-Collection]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 冷启动用户 AUC +1.522%；多方 PSU 隐藏交集保护商业机密 |
| **实施难度** | ⭐⭐⭐⭐☆ |
| **优先级评分** | ⭐⭐⭐☆☆ |

## 论文来源

- 2604.19219 (Sherpa.ai PSU)
- 2606.01396 (Cross-Domain SID)
- 2604.16521 (CAMP)

---
## ⑥ Skill Relations
**延伸技能（Extends）**
- [[Skill-HGNN-Cross-Device-Matching]]

**可组合技能（Combinable）**
- [[Skill-Privacy-Preserving-Federated-Collection]]
- [[Skill-Identity-Fragmentation-Debiasing]]

