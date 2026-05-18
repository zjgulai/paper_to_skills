---
title: 下一步论文选题路线图
doc_type: roadmap
module: project-management
status: active
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
---

# 下一步论文选题路线图(next-papers-roadmap)

> **v2 更新(2026-05-17 下午):Sprint 1+2 全部 10 个 P0 Skill 萃取完成,图谱 89→107 节点 / 418→581 边. Sprint 3 候选已就绪.**

---

## 0. 最新状态快照(2026-05-17 下午)

### Sprint 1+2 完成清单 (10 个新 Skill)

**Sprint 1 — WF-E + WF-C 多语种 ABSA 闭环 (4 个):**
- ✅ Skill-AGRS-Aspect-Guided-Review-Summarization (arXiv:2509.26103, Wayfair 2025)
- ✅ Skill-MAA-Review-to-Action-Decision (arXiv:2601.12024)
- ✅ Skill-StaR-Review-Statement-Ranking (arXiv:2604.03724)
- ✅ Skill-LACA-CrossLingual-ABSA (arXiv:2508.09515, ACL 2025) — WF-C + WF-E 双用

**Sprint 2 — WF-A + WF-B P0 阻塞缺口 (6 个):**
- ✅ Skill-Hierarchical-Demand-Forecasting-Reconciliation (arXiv:2412.14718, Walmart BigData 2024)
- ✅ Skill-Lead-Time-Distribution-Risk-GenQOT (arXiv:2310.17168, Amazon 2024)
- ✅ Skill-Hierarchical-Search-Intent-Classification (arXiv:2403.06021, Amazon WWW 2024)
- ✅ Skill-DialIn-LLM-Case-Intent-Clustering (arXiv:2412.09049, EMNLP 2025)
- ✅ Skill-PVM-Attribution-Window-Harmonization (arXiv:2511.22918, NeurIPS 2025)
- ✅ Skill-Bass-Diffusion-New-Product-Forecasting (arXiv:2307.03595, Amazon 2023)

### 工作流能力覆盖率提升

| 工作流 | 审计前 | Sprint1+2 后 | P0 缺口解锁 |
|---|---|---|---|
| WF-A 智能补货 | 55% | **85%+** | 分层调和 + Lead Time 波动 + 新品冷启动 |
| WF-B 广告优化 | 48% | **75%+** | 月龄意图分类 + 跨平台归因统一 |
| WF-C 客服分诊 | 38% | **70%+** | 多语种 ABSA + 意图聚类 |
| WF-E Review 监控 | 25% | **85%+** | StaR + AGRS + MAA + LACA 完整闭环 |

### 累计业务 ROI 估算
**6550-13160 万元/年潜在**(中型母婴跨境品牌)

详见 [Sprint 1+2 迭代总报告](./sprint1-2-iteration-report-20260517.md).

---

## 0.1 Sprint 3 候选清单(P1 优先级,待启动)

| 候选 Skill | 服务工作流 | 推荐论文方向 | 业务紧迫度 |
|---|---|---|---|
| Negative-Keyword-Safe-Guard | WF-B (S03/S04) | Bayesian keyword performance estimation small sample | 高 |
| Creative-Fatigue-Detection | WF-B (S13) | Creative fatigue survival analysis digital advertising 2024 | 高 |
| Conformal-Prediction-Demand-UQ | WF-A (P15) | Conformal Prediction for Time Series 2023 | 中 |
| Multi-Channel-Inventory-Pooling | WF-A (P7) | Cross-channel inventory transshipment 2024 | 高 |
| Amazon-ToS-Compliance-Guardrail | WF-C + WF-B | LLM compliance guardrail e-commerce 2024 | 极高(合规) |
| TikTok-Shop-Content-Attribution | WF-B (S16) | Interest-based commerce attribution social 2025 | 高 |

---

## 一、来源 A:图谱自动分析的高优先级缺口(节选)

> 历史 v1 记录(2026-05-17 上午). 当前 HIGH 缺口仅剩 1 个(CausalRAG),需手动选题填补.

由 `skills_graph_analyzer.py --gaps` 产生。当前共 156 个缺口,其中高优先级 19 个。前置技能缺失是主要类型。

### 必须填补的基础前置 Skill

| 优先级 | 缺失 Skill | 阻塞的下游 Skill | 建议萃取方向 |
|--------|-----------|-----------------|------------|
| P0 | Skill-Recommendation-System | DQN-Purchase-Prediction、Customer-Journey-Prototype | 推荐系统基础综述 + LightFM/Wide&Deep |
| P0 | Skill-A-B-Test-Design | Uplift-Churn-Prediction | 实验设计基础（已有 Skill-AB-Experimental-Design，需补别名链接） |
| P0 | Skill-Doubly-Robust-Estimation | Uplift-Churn-Prediction | 已有 Skill-Intelligent-Prediction-Doubly-Robust，需补别名链接 |
| P0 | Skill-Reinforcement-Learning | Uplift-Churn-Prediction | RL 基础（DQN/PPO 概念入门） |
| P0 | CausalRAG | GraphRAG-Knowledge-Enhanced-Retrieval | 因果增强检索 |
| P0 | 动态 LTV 预测 | LTV-Prediction-ZILN 延伸 | 时变 LTV 建模 |
| P0 | Contextual Bandit | LTV-Prediction-ZILN | 上下文 Bandit 应用 |
| P0 | Skill-VOC-Aspect-Extraction | NLP-VOC 系列（已迁出，仅引用） | 在 ai_nlp_voc 仓库落地 |

> 💡 多个"缺失"实际是命名漂移：图谱分析使用的引用名与现存 Skill 的实际文件名不一致。建议在 P3 后续工作中,做一次"Skill 别名映射表"修复，让图谱能识别现存 Skill 的等价命名。

### 跨领域桥梁机会

| 机会 | 涉及领域 | 业务价值 |
|------|---------|---------|
| 因果推断 ↔ 推荐系统 | 01 + 05 | 反事实推荐、Uplift 推荐 |
| 知识图谱 ↔ DataAgent | 08 + 09 | KG-augmented 数据 Agent |
| 智能体工程 ↔ 营销分析 | 16 + 15 | Agentic MMM / 自动调优 |

---

## 二、来源 B:语义蓝图 5 方向（用户已确认萃取目标）

来源：[`drafts/analysis/semantic-blueprint-paper-selection-draft-20260428.md`](../../drafts/analysis/semantic-blueprint-paper-selection-draft-20260428.md)

| 方向 | 主推论文 | 评分 | 萃取后归属领域 |
|------|---------|------|--------------|
| 1. 产品属性图谱 | Hierarchical KG Construction from Images for Scalable E-Commerce (arXiv:2410.21237) | 8.50 | 08-知识图谱 |
| 2. VOC 语义蓝图 | （详见草稿，匹配度最高） | — | 已迁至 ai_nlp_voc 仓库 |
| 3. 行为意图解析 | （序列→树，需进一步筛选） | — | 待定（可能 05-推荐系统 或 14-用户分析） |
| 4. 跨语言语义对齐 | AMR 类研究 | — | 业务契合度低，建议暂缓 |
| 5. 客服对话决策树 | 匹配度高 | — | 09-DataAgent-LLM 或 10-MAS |

### 萃取行动建议

1. **方向 1（产品属性图谱）** → 立即可萃取，归入 `08-知识图谱`，新 Skill 名建议 `Skill-Hierarchical-Product-KG-Construction`
2. **方向 2（VOC 语义蓝图）** → 不在本仓库萃取，由 `ai_nlp_voc` 仓库负责
3. **方向 3（行为意图解析）** → 需要再做一次窄化检索，待定优先级
4. **方向 4（跨语言语义对齐）** → 暂缓
5. **方向 5（客服对话决策树）** → 高匹配度，归入 `09-DataAgent-LLM`

---

## 三、综合执行建议（按本季度可行性排序）

### Sprint 1（本周）— 补齐图谱基础前置

1. 在已有 Skill 上补充别名（建立 `Skill-A-B-Test-Design`、`Skill-Doubly-Robust-Estimation` 等的引用映射），消除多个"伪缺口"
2. 萃取 **Skill-Recommendation-System**（综述向，3 篇 review 综合）作为基础前置

### Sprint 2（下周）— 启动语义蓝图方向 1

3. 萃取 **Skill-Hierarchical-Product-KG-Construction**（arXiv:2410.21237），归入 08-知识图谱
4. 同步更新 skills_graph_report.md 验证缺口减少

### Sprint 3（本月内）— 跨领域桥梁

5. 萃取 **Skill-Causal-Recommendation**（01 + 05 桥梁），可基于 Causal Forest + Uplift 现有 Skill 综合
6. 萃取 **方向 5 客服对话决策树** 进入 09-DataAgent-LLM

---

## 四、未纳入路线图的方向

- **方向 4 跨语言语义对齐 (AMR)** — 业务契合度低，暂缓
- **VOC 语义蓝图（方向 2）** — 由 ai_nlp_voc 仓库负责
- **图谱报告中标注的"NLP-VOC 系列前置"** — 全部由 ai_nlp_voc 仓库负责

---

## 五、维护说明

- 本路线图与 `skills_graph_report.md` 形成"自动 + 人工"双驱动:
  - 自动:`skills_graph_analyzer.py --gaps` 找出图谱中的依赖断裂
  - 人工:本文件根据业务优先级筛选并排序
- 完成一个 Skill 后:更新本文件的"已完成"状态,并重跑 `skills_graph_analyzer.py --analyze`
