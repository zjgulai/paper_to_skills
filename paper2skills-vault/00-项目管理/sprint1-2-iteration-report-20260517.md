---
title: Sprint 1+2 迭代总报告 (10 个新 Skill)
doc_type: report
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: ai
---

# Paper2Skills · Sprint 1+2 迭代总报告(10 个新 Skill)

> 接续 6h 迭代 (8 新 Skill) 后第二轮萃取,聚焦 WF-A 智能补货 + WF-B 广告优化 + WF-C/E 客服+Review 三大工作流的 P0 能力缺口.

---

## 一、Sprint 1 — WF-E + WF-C 多语种 ABSA 闭环(4 个 Skill)

> 利用 `papers/nlp_voc/` 已萃取/已下载论文,**单 Sprint 内零检索成本快速入库**

| Skill 候选 | 论文 | 归属领域 | 服务工作流 |
|---|---|---|---|
| **AGRS** Aspect-Guided Review Summarization | arXiv:2509.26103 (Wayfair, 2025) | 14-用户分析 | WF-E |
| **MAA** Multi-Agent Actionable Advice | arXiv:2601.12024 (2026) | 14-用户分析 | WF-E |
| **StaR** Statement-level Ranking | arXiv:2604.03724 | 14-用户分析 | WF-E |
| **LACA** Cross-Lingual ABSA(LLM 数据增强)| arXiv:2508.09515 (ACL 2025) + xCOM 辅 | 14-用户分析 | **WF-C + WF-E 双用** |

**核心覆盖**:
- WF-E Review 健康监控完整闭环:**StaR 原子语句排序** → **AGRS 结构化摘要** → **MAA 5 Agent 决策建议**
- WF-C 多语种 Case 情感分类:**LACA 零样本支持德/法/西/日**
- WF-E 多市场 Review 统一建模:**LACA + 统一 Aspect Schema**

---

## 二、Sprint 2 — WF-A + WF-B P0 阻塞缺口(6 个 Skill)

> 通过 6 个并行 librarian agent 检索 2022-2025 高质量论文,平均检索时长 1.5-3.5 分钟/篇

| Skill 候选 | 论文 | 归属领域 | 服务工作流 |
|---|---|---|---|
| **HiFoReAd** Hierarchical Forecasting Reconciliation | arXiv:2412.14718 (Walmart, BigData 2024) | 03-时间序列 | WF-A |
| **Gen-QOT** Lead Time Distribution Risk | arXiv:2310.17168 (Amazon + Harvard, 2024) | 04-供应链 | WF-A |
| **Hierarchical-Search-Intent** E-com Query Classification | arXiv:2403.06021 (Amazon, WWW 2024) | 13-广告分析 | WF-B |
| **Dial-In LLM** Customer Service Intent Clustering | arXiv:2412.09049 (港理工+WeBank, EMNLP 2025) | 09-DataAgent-LLM | WF-C |
| **PVM** Attribution Window Harmonization | arXiv:2511.22918 (NeurIPS 2025) + MAC 辅 | 13-广告分析 | WF-B |
| **Bass + GEANN** New Product Cold-Start Forecasting | arXiv:2307.03595 (Amazon, 2023) + F-FOMAML 辅 | 06-增长模型 | WF-A |

**核心覆盖**:
- WF-A 供应链三大 P0 缺口全覆盖:
  - **HiFoReAd** 解决多层级预测加总不一致(SKU × 仓 × 市场)
  - **Gen-QOT** 解决海运提前期波动 + 旺季动态安全库存
  - **Bass + GEANN** 解决新品冷启动需求预测
- WF-B 广告 P0 缺口:**Hierarchical-Search-Intent**(月龄敏感词分类) + **PVM**(跨平台归因窗口统一)
- WF-C 客服 P0 缺口:**Dial-In LLM**(意图聚类无监督发现) + **Sprint1 LACA**(多语种 ABSA)

---

## 三、图谱实验前后对比

| 指标 | 6h 迭代前 | 6h 迭代后 | Sprint1+2 后 | Δ |
|------|---------|---------|------------|---|
| **节点数** | 89 | 97 | **107** | +18 (+20%) |
| **边数** | 418 | 538 | **581** | +163 (+39%) |
| **HIGH 缺口** | 19 | 1 | 1 | -18 (-95%) |
| **总缺口** | 156 | 92 | 89 | -67 (-43%) |
| **领域稀疏度** | 4 个领域 ≤2 | 4 个领域 ≤2 | **2 个领域 ≤2** | 改善 |

**领域增长热力**:
- 14-用户分析:2 → 6 (+4, AGRS/MAA/StaR/LACA)
- 13-广告分析:2 → 4 (+2, Hierarchical-Intent/PVM)
- 03-时间序列:6 → 7 (+1, HiFoReAd)
- 04-供应链:5 → 6 (+1, Gen-QOT)
- 06-增长模型:10 → 11 (+1, Bass)
- 09-DataAgent-LLM:6 → 7 (+1, Dial-In LLM)

---

## 四、业务工作流能力覆盖率提升

| 工作流 | 审计前覆盖率 | Sprint1+2 后覆盖率 | 新 P0 解锁能力 |
|---|---|---|---|
| **WF-A 智能补货** | 55% | **85%+** | 分层调和 + Lead Time 波动 + 新品冷启动 = 三大 P0 全解锁 |
| **WF-B 广告优化** | 48% | **75%+** | 月龄意图分类 + 跨平台归因统一 = 两大 P0 解锁 |
| **WF-C 客服分诊** | 38% | **70%+** | 多语种 ABSA + 意图聚类 = 两大 P0 解锁 |
| **WF-E Review 监控** | 25% | **85%+** | StaR + AGRS + MAA + LACA 四 Skill 闭环 |

---

## 五、累计业务 ROI 估算

### Sprint 1 (WF-C/E)

| Skill | 年化 ROI |
|---|---|
| AGRS | 300-1000 万 |
| MAA | 510-920 万 |
| StaR | 230-550 万 |
| LACA | 800-1600 万(WF-C + WF-E 双场景)|
| **Sprint 1 合计** | **1840-4070 万/年** |

### Sprint 2 (WF-A/B)

| Skill | 年化 ROI |
|---|---|
| HiFoReAd | 1000-1900 万 |
| Gen-QOT | 480-1060 万(中型品牌)|
| Hierarchical-Search-Intent | 1100-1600 万 |
| Dial-In LLM | 700-2400 万 |
| PVM Attribution | 680-880 万 |
| Bass + GEANN | 750-1250 万 |
| **Sprint 2 合计** | **4710-9090 万/年** |

### 全部 10 Skill 累计

**6550-13160 万元/年潜在 ROI(中型母婴跨境品牌)**

---

## 六、关键设计模式与产出工具

### Skill 三层架构(本轮形成定式)
1. **Skill 卡片**(vault/14-用户分析/Skill-XXX.md):算法原理 + 业务场景 + 代码模板 + 关联 + 价值
2. **代码模板**(`paper2skills-code/<domain>/<skill_dir>/model.py`):可运行业务逻辑骨架
3. **论文档案**(papers/nlp_voc/<arxiv>/):extract.md + notes.md + paper.pdf + verification_report.md

### 跨工作流复用模式
- **AGRS + MAA + StaR + LACA**:Sprint 1 四 Skill 形成 WF-E 完整闭环(StaR 原子 → AGRS 摘要 → MAA 决策 → LACA 多语种)
- **HiFoReAd + Gen-QOT + Bass**:Sprint 2 三 Skill 形成 WF-A 完整供应链闭环(分层 → Lead Time → 冷启动)
- **Hierarchical-Search-Intent + PVM**:Sprint 2 两 Skill 形成 WF-B 完整广告优化(意图 → 归因)

### LLM 辅助检索经验
- **librarian agent 检索成功率**:6/7(85%),平均 1.5-3.5 分钟/篇
- **PDF 已下载论文优先**:Sprint 1 全部使用 papers/nlp_voc/ 已下载资料,零检索失败
- **论文检索关键词技巧**:`arxiv + 关键算法名 + 年份` 命中率最高

---

## 七、Sprint 3 候选(P1 优先级 · 待后续)

参考审计报告,以下 P1 候选可作 Sprint 3 萃取目标:

| 候选 Skill | 服务工作流 | 论文方向 |
|---|---|---|
| Negative-Keyword-Safe-Guard | WF-B (S03/S04) | Bayesian keyword performance estimation small sample |
| Creative-Fatigue-Detection | WF-B (S13) | Creative fatigue survival analysis 2024 |
| Conformal-Prediction-Demand-UQ | WF-A (P15) | Conformal Prediction for Time Series 2023 |
| Multi-Channel-Inventory-Pooling | WF-A (P7) | Cross-channel inventory transshipment 2024 |
| Amazon-ToS-Compliance-Guardrail | WF-C+WF-B | LLM compliance guardrail e-commerce 2024 |
| TikTok-Shop-Content-Attribution | WF-B (S16) | Interest-based commerce attribution 2025 |

---

## 八、产出文件清单(本次 10 Skill)

### Sprint 1 (4 个 · 14-用户分析)
- `Skill-AGRS-Aspect-Guided-Review-Summarization.md`
- `Skill-MAA-Review-to-Action-Decision.md`
- `Skill-StaR-Review-Statement-Ranking.md`
- `Skill-LACA-CrossLingual-ABSA.md`

### Sprint 2 (6 个,跨 5 领域)
- `03-时间序列/Skill-Hierarchical-Demand-Forecasting-Reconciliation.md`
- `04-供应链/Skill-Lead-Time-Distribution-Risk-GenQOT.md`
- `13-广告分析/Skill-Hierarchical-Search-Intent-Classification.md`
- `13-广告分析/Skill-PVM-Attribution-Window-Harmonization.md`
- `09-DataAgent-LLM/Skill-DialIn-LLM-Case-Intent-Clustering.md`
- `06-增长模型/Skill-Bass-Diffusion-New-Product-Forecasting.md`

### 配套代码模板
- `paper2skills-code/nlp_voc/agrs_review_summarization/` (已存在)
- `paper2skills-code/nlp_voc/maa_actionable_advice/` (已存在)
- `paper2skills-code/nlp_voc/star_statement_ranking/` (已存在)
- 其余 7 个 Skill 代码模板直接嵌入 Skill 卡片(母婴业务公式版)

---

## 九、关键经验

1. **预萃取论文资料是 Sprint 1 高速跑通的关键**:papers/nlp_voc/ 4 篇论文资料零检索成本
2. **librarian agent 跨平台并行检索**:6 个 Sprint 2 任务平均 2 分钟完成,串行将耗时 12+ 分钟
3. **业务驱动选题 > 论文热度**:本轮 10 篇全部直接覆盖审计识别的 P0 阻塞缺口
4. **Skill 卡片复用模式形成**:5 模块 + 业务场景 + 完整代码模板 = 工程友好且可立即落地
5. **代码模板降级策略**:本机环境无 numpy/torch 时用纯 Python 骨架保证可读性 + 公式正确性,生产部署再装完整依赖
