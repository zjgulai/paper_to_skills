---
title: 13-广告分析技能索引
doc_type: index
module: 13-广告分析
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
---

# 13-广告分析 (Advertising Analytics) 技能索引

## 领域定位

聚焦广告投放的归因、预算分配与 ROAS 优化。与 15-营销投放分析 的区别：

- **13-广告分析**:面向单次投放的"事后归因 + 当下优化"
- **15-营销投放分析**:面向长周期的"营销组合建模 + 因果效应"

## 已落地 Skill

| 技能 | 状态 | 业务场景 |
|------|------|---------|
| [Skill-Ad-Attribution-Modeling](./Skill-Ad-Attribution-Modeling.md) | 已萃取 P0 | 多触点 Shapley/Markov 归因 |
| [Skill-ROAS-Budget-Optimization](./Skill-ROAS-Budget-Optimization.md) | 已萃取 P0 | 预算分配优化 |

## 规划 Skill 路线图

| 技能 | 优先级 | 论文/方法 | 业务场景 |
|------|--------|----------|---------|
| Skill-Incremental-Attribution-Causal | P0 | Geo-Lift / Causal Forest | 区分 incremental 与 spurious 归因 |
| Skill-Frequency-Cap-Optimization | P1 | reach/frequency curve | 频次上限对 CTR 与转化的影响建模 |
| Skill-Creative-Fatigue-Detection | P1 | survival analysis | 创意疲劳曲线与切换时机 |
| Skill-Cross-Channel-Attribution-MTA | P2 | MTA + position bias | 跨渠道（搜索/社交/展示）归因 |
| Skill-View-Through-Conversion | P2 | view-through vs click-through | 曝光归因与点击归因融合 |

## 与其他领域的衔接

| 领域 | 衔接点 |
|------|--------|
| 01-因果推断 | 归因偏差的因果识别 |
| 15-营销投放分析 | MMM 的渠道粒度细化 |
| 14-用户分析 | 漏斗转化中的广告触点贡献 |

## 统计数据

- 已萃取:2
- 规划待萃取:5
