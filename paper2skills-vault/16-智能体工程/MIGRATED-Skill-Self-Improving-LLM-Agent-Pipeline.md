---
name: skill-self-improving-llm-agent-pipeline-migrated
description: 已迁出技能占位说明。当查找 Skill-Self-Improving-LLM-Agent-Pipeline 时使用本说明定位真实位置。
---

# ⚠️ Skill-Self-Improving-LLM-Agent-Pipeline — 已迁出

## 状态

**本仓库不再维护此 Skill 副本。**

## 历史信息

| 属性 | 内容 |
|------|------|
| 原 Skill 名 | `Skill-Self-Improving-LLM-Agent-Pipeline` |
| 生成时间 | 2026-05-06 |
| 原归属领域 | `07-NLP-VOC`（本仓库已迁出该领域） |
| 业务场景 | Momcozy 吸奶器 Amazon US 文案 CTR 自迭代（2.1% → 3.8%，+81%）；竞品情报自动萃取准确率自提升（62% → 89%，+44%） |
| 核心论文 | The AI Scientist (arXiv:2408.06292)、SEAL (NeurIPS 2025)、Self-Challenging Language Model Agents (NeurIPS 2025)、ETO (2024) |
| 审核评分 | 8.0/10（已通过） |

## 当前位置

该 Skill 已随 `07-NLP-VOC` 领域整体迁至独立仓库：

```
/Users/lute/project/ai_nlp_voc/
```

参见 [`CLAUDE.md`](../../CLAUDE.md) "NLP-VOC 子项目迁出说明" 章节，迁出 commit 为 `47b1dbf` (2026-05-17)。

## 与本领域（16-智能体工程）的差异

本领域已落地的同主题但不同切入点的 Skill：

- [Skill-Auto-Skill-Synthesis](Skill-Auto-Skill-Synthesis.md) — 自动 Skill 合成
- [Skill-Co-Evolutionary-Skill-Verification](Skill-Co-Evolutionary-Skill-Verification.md) — 协同演化与 Skill 验证
- [Skill-Orchestration-Trace-RL](Skill-Orchestration-Trace-RL.md) — 编排轨迹 RL

如需 NLP-VOC 业务场景下的"自迭代文案/情报"管线，请查阅 `ai_nlp_voc` 仓库。

## 不要做的事

- ❌ 不要在本仓库重新生成同名 Skill 文件
- ❌ 不要将 `ai_nlp_voc` 中的副本拷回本仓库
- ❌ 不要在 sync_status.json 中为此 Skill 添加新条目

如需在本领域萃取相关算法，请挑选不同切入点（例：偏向 Agent Skills 框架本身的"自迭代 Skill 注册-验证-淘汰"机制），并使用新的 Skill 名称。
