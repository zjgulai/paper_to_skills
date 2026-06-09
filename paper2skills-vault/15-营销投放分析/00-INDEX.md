---
title: 15-营销投放分析技能索引
doc_type: index
module: 15-营销投放分析
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
---

# 15-营销投放分析 (Marketing Analytics) 技能索引

## 领域定位

长周期、跨渠道的营销组合建模与因果效应估计。与 13-广告分析 的区别：

- **15-营销投放分析**:长周期、跨渠道 MMM、促销效果因果识别
- **13-广告分析**:单次投放归因、预算分配优化

## 已落地 Skill

| 技能 | 状态 | 业务场景 |
|------|------|---------|
| [Skill-Marketing-Mix-Modeling](./Skill-Marketing-Mix-Modeling.md) | 已萃取 P0 | MMM 营销组合建模 |
| [Skill-Promotion-Effectiveness](./Skill-Promotion-Effectiveness.md) | 已萃取 P0 | 促销 DML 因果效应 |

## 规划 Skill 路线图

| 技能 | 优先级 | 论文/方法 | 业务场景 |
|------|--------|----------|---------|
| Skill-Geo-Lift-Experiment | P0 | Google Meridian / Geo Experiment | 区域实验设计与 lift 估计 |
| Skill-Synthetic-Control | P1 | Abadie synthetic control | 大促/品牌投放事件研究 |
| Skill-Difference-in-Differences | P1 | DiD + event study | 政策/价格变更前后影响 |
| Skill-Bayesian-MMM | P2 | LightweightMMM / PyMC | 贝叶斯 MMM 不确定性量化 |
| Skill-Saturation-Adstock-Tuning | P2 | Hill / Geometric decay | 饱和度与延滞效应参数估计 |

## 与其他领域的衔接

| 领域 | 衔接点 |
|------|--------|
| 01-因果推断 | DML/DiD/Synthetic Control 方法学共享 |
| 13-广告分析 | MMM 渠道层 → 单次归因 |
| 02-A_B实验 | Geo-Lift 与传统 A/B 互补 |

## 统计数据

- 已萃取:2
- 规划待萃取:5
