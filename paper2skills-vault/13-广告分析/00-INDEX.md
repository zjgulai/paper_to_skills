---
title: 13-广告分析技能索引
doc_type: index
module: 13-广告分析
status: stable
created: 2026-05-17
updated: 2026-05-20
owner: self
source: human+ai
---

# 13-广告分析 (Advertising Analytics) 技能索引

## 领域定位

聚焦广告投放的归因、意图分类、预算分配与 ROAS 优化. 与 15-营销投放分析 的区别:

- **13-广告分析**:面向单次投放的"事后归因 + 当下优化 + 意图入口"
- **15-营销投放分析**:面向长周期的"营销组合建模 + 因果效应"

> **2026-05-17 重大扩展**:WF-B 广告优化工作流 P0 阻塞缺口全覆盖,新增意图分类 + 跨平台归因统一两大核心能力.

## 已落地 Skill(9 个)

### 基础归因与预算(2)

| 技能 | 状态 | 业务场景 |
|------|------|---------|
| [Skill-Ad-Attribution-Modeling](./Skill-Ad-Attribution-Modeling.md) | 已萃取 P0 | 多触点 Shapley/Markov 归因 |
| [Skill-ROAS-Budget-Optimization](./Skill-ROAS-Budget-Optimization.md) | 已萃取 P0 | 边际 ROAS 均等化 + 预算分配优化 |

### WF-B P0 阻塞缺口(Sprint 2, 2026-05-17)

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Hierarchical-Search-Intent-Classification](./Skill-Hierarchical-Search-Intent-Classification.md) | 已萃取 P0 | arXiv:2403.06021 (Amazon WWW 2024) | 月龄敏感词分类 + 信息vs购买意图 |
| [Skill-PVM-Attribution-Window-Harmonization](./Skill-PVM-Attribution-Window-Harmonization.md) | 已萃取 P0 | arXiv:2511.22918 (NeurIPS 2025) | 跨平台归因窗口对齐 + 跨渠道 ROAS 矫正 |

### 跨设备用户匹配(Sprint 3, 2026-05-20)

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-HGNN-Cross-Device-Matching](./Skill-HGNN-Cross-Device-Matching.md) | 已萃取 P1 | arXiv:2304.03215 (NVIDIA 2023) | 层次GNN跨设备用户匹配 |
| [Skill-GraphTrack-Cross-Device-Tracking](./Skill-GraphTrack-Cross-Device-Tracking.md) | 已萃取 P1 | arXiv:2203.06833 | 无监督图基跨设备追踪 |

### 归因去偏与延迟转化(Round 4, 2026-05-20)

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-CABB-Cross-Category-Attribution](./Skill-CABB-Cross-Category-Attribution.md) | 已萃取 P1 | arXiv:2507.15113 (2025) | Click A Buy B 跨品类归因去偏 |
| [Skill-TRACE-Delayed-CVR](./Skill-TRACE-Delayed-CVR.md) | 已萃取 P1 | arXiv:2604.23197 (2025) | 不等14天窗口实时更新CVR |
| [Skill-TESLA-NetCVR-Cascade](./Skill-TESLA-NetCVR-Cascade.md) | 已萃取 P2 | arXiv:2601.19965 (Taobao 2025) | 扣除退款的净转化级联建模 |

## 闭环组合(2026-05-20 更新)

```
Hierarchical-Search-Intent (自动词意图分类)
  → 高 ROI 词识别(月龄/购买意图)
  → Ad-Attribution-Modeling (Shapley/Markov)
  → PVM (跨平台归因窗口统一)
  → HGNN + GraphTrack (跨设备用户匹配)
  → CABB (跨品类归因去偏)
  → TRACE Delayed CVR (不等窗口实时更新)
  → TESLA NetCVR (扣除退款的净转化)
  → ROAS-Budget-Optimization (跨渠道预算重分配)
```

## 规划 Skill 路线图(Sprint 3 P1 候选)

| 技能 | 优先级 | 论文/方法 | 业务场景 |
|------|--------|----------|---------|
| Skill-Negative-Keyword-Safe-Guard | P1 | Bayesian keyword performance 2024 | 否定词置信区间保护 |
| Skill-Creative-Fatigue-Detection | P1 | Survival analysis 2024 | 创意疲劳检测与切换 |
| Skill-Incremental-Attribution-Causal | P1 | Geo-Lift / Causal Forest | 区分 incremental 与 spurious 归因 |
| Skill-Frequency-Cap-Optimization | P2 | reach/frequency curve | 频次上限对 CTR 与转化的影响 |
| Skill-TikTok-Shop-Content-Attribution | P2 | Interest-based commerce 2025 | TikTok 兴趣电商归因 |
| Skill-View-Through-Conversion | P2 | VTC vs CTC | 曝光归因与点击归因融合 |

## 与其他领域的衔接

| 领域 | 衔接点 |
|------|--------|
| 01-因果推断 | 归因偏差的因果识别 |
| 15-营销投放分析 | MMM 渠道层 + DARA 跨渠道预算 |
| 14-用户分析 | 漏斗转化中的广告触点贡献 |
| 08-知识图谱 | 意图分类基于多语种 NER + Semantic Search |

## 统计数据

- 已萃取:**9**(2 基础 + 2 P0 缺口 + 5 桑基图流量转化)
- 规划待萃取:6
- 服务工作流:**WF-B 广告优化**
