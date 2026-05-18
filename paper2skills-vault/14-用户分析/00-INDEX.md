---
title: 14-用户分析技能索引
doc_type: index
module: 14-用户分析
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
---

# 14-用户分析 (User Analytics) 技能索引

## 领域定位

聚焦用户行为数据的描述性分析与分群运营，与 06-增长模型 的预测建模互补：

- **14-用户分析**:Where are users now? — 漏斗/留存/分群的现状描述
- **06-增长模型**:What will they do next? — 流失/LTV/购买的预测建模

## 已落地 Skill

| 技能 | 状态 | 业务场景 |
|------|------|---------|
| [Skill-User-Funnel-Analysis](./Skill-User-Funnel-Analysis.md) | 已萃取 P0 | 漏斗转化分析 |
| [Skill-Cohort-Retention-Analysis](./Skill-Cohort-Retention-Analysis.md) | 已萃取 P0 | 队列留存分析 |

## 规划 Skill 路线图

| 技能 | 优先级 | 论文/方法 | 业务场景 |
|------|--------|----------|---------|
| Skill-RFM-Segmentation | P0 | 经典 RFM + K-means | 母婴用户分群运营 |
| Skill-User-Path-Analysis | P1 | sequence mining | 用户跳转路径与流失节点 |
| Skill-Repeat-Purchase-Behavior | P1 | inter-purchase time | 复购周期分析 |
| Skill-User-Lifetime-Stage-Detection | P2 | HMM / change point | 新生/孕产/婴幼儿不同阶段切换识别 |

## 与其他领域的衔接

| 领域 | 衔接点 |
|------|--------|
| 06-增长模型 | 现状分析 → 预测建模 |
| 05-推荐系统 | 用户分群 → 个性化推荐 |
| 13-广告分析 | 漏斗归因衔接 |

## 统计数据

- 已萃取:2
- 规划待萃取:4
