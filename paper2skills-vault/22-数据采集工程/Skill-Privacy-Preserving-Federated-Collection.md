---
title: Privacy-Preserving Federated Collection — 隐私保护联邦采集：差分隐私预算与联邦推荐
doc_type: knowledge
module: 22-数据采集工程
topic: privacy-preserving-federated-collection
status: stable
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Privacy-Preserving Federated Collection — 隐私保护联邦采集：差分隐私预算与联邦推荐

## ① 核心算法

**论文**：SF-UBM [2604.14833] + MFG-RegretNet [2603.28329]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：跨平台（Amazon+TikTok+独立站）数据联合建模，GDPR/PIPL 合规下不共享原始数据

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/federated_collection/model.py`

## ④ 技能关联

### 前置技能
- [[无（Layer 1）]]

### 延伸技能
- 无

### 可组合技能
- [[Skill-Privacy-Safe-Identity-Resolution]]
- [[Skill-Category-Compliance-Prescan]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 模型反演攻击成功率压至 4.8%；O(N²logN)→O(N) 计算复杂度；联邦推荐首解非重叠域 PPCDR |
| **实施难度** | ⭐⭐⭐⭐☆ |
| **优先级评分** | ⭐⭐☆☆☆ |

## 论文来源

- Informatica 2026
- 2604.14833 (SF-UBM)
- 2603.28329 (MFG-RegretNet)

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-Privacy-Safe-Identity-Resolution]]

**可组合技能（Combinable）**
- [[Skill-Category-Compliance-Prescan]]

