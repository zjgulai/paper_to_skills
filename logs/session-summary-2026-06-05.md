---
title: Session 摘要 — MAS 2026 萃取 + 数据采集工程新领域建立
doc_type: session-summary
date: 2026-06-05
session_id: ses_16f294c5fffeloBYidiiN8FVS1
created: 2026-06-05
---

# Session 摘要：paper2skills MAS 扩充 + 数据采集新领域

## 一、本 Session 完成的工作

### 1. MAS 2026 专项萃取（9 个新 Skill）

**触发**：用户要求针对 MAS 多智能体系统继续萃取 2026 年论文，结合 skills graph 进行缺口分析。

**图谱起点**：287 节点 / 5,152 边 / MAS 领域 25 个 Skills

**执行路径**：按知识图谱填充逻辑（prerequisite 断链修复 → 孤立节点接入 → Hub 延伸链 → 跨域桥梁）排序萃取：

| Wave | Skill | 填补的图谱空白 | 核心论文 |
|------|-------|-------------|---------|
| W1 | `Skill-MAS-Dynamic-Trust` | 修复 AgentTrust prerequisite 断链 | DynaTrust [2603.15661] + A-Trust [2506.02546] + ECL [2601.21742] |
| W1 | `Skill-MAS-Testing-Verification` | 接通 MASEval 延伸链 | FLARE [2604.05289] + MAESTRO [2601.00481] |
| W2 | `Skill-MAS-Resource-Scheduling` | Orchestrator 生产化延伸 | HiveMind [2604.17111] + AgentRM [2603.13110] + MCPP [2605.06110] |
| W2 | `Skill-MAS-Consensus-Mechanism` | Multi-Agent-Debate 理论深化 | Aegean [2512.20184] + DySCo [2606.01828] + SAC [2605.09076] |
| W3 | `Skill-MAS-Scale-Management` | AgentRegistry 架构延伸 | MegaFlow [2601.07526] + MonoScale [2601.23219] + OrgAgent [2604.01020] |
| W3 | `Skill-LLM-AutoBidding-MAS` ★ | **mas ↔ advertising 首条跨域桥梁** | DARA [2601.14711, WWW'26] + LBM [2603.05134, WWW'26] |
| W4 | `Skill-MAS-Adversarial-Defense` | Dynamic-Trust 应用延伸 | GroupGuard [2603.13940] + FlowSteer [2605.11514] + Conjunctive [2604.16543] |
| W4 | `Skill-Cross-Org-Agent-Protocol` | G²CP+MCP-A2A 协议上层 | MPAC [2604.09744] + ACP [2602.15055] + IETF MACP |
| W5 | `Skill-MAS-Dynamic-KG-Collaboration` ★ | **mas ↔ knowledge_graph 跨域桥梁** | MemGraphRAG [2606.00610] + MAGE [2605.10064] |

**结果**：
- MAS Skills：25 → 34
- 新建跨域桥梁：2 条（mas↔advertising、mas↔KG）
- 修复 prerequisite 断链：2 条
- 所有代码测试：44/44 通过

---

### 2. 项目文档更新

更新了以下文件：
- `paper2skills-vault/README.md`：节点 302 / 边 5,390 / 新跨域桥梁
- `00-项目管理/项目总结.md`：新增 MAS 2026 专项章节
- `00-项目管理/进度追踪.md`：追加完整萃取记录
- `00-项目管理/知识图谱速查卡.md`：v4.0 更新领域表、组合推荐
- `paper-skills-graph/skills_graph_report.md`：追加变更摘要

**运行 skills_graph_analyzer 确认**：302 节点 / 5,390 边 / HIGH 缺口 = 0

---

### 3. 新领域建立：22-数据采集工程（13 个 Skill）

**触发**：用户提出增加数据采集方向选题，希望萃取 2026 年以来的相关算法，全部萃取。

**5 路并行搜索**后识别 18 篇 2026 年核心论文，新建 `22-数据采集工程` 领域。

**领域定位**：Layer 1 基础层，是所有分析 Skill 的上游数据前置。

| Skill | 核心论文 | 解锁的下游 Skill |
|-------|---------|----------------|
| `Skill-Document-Intelligence-Parsing` | dots.mocr [2603.13032] + Qianfan-OCR [2603.13398] + MinerU-Popo [2605.24973] | Supplier-Capacity-Planning, Multi-SKU-Procurement |
| `Skill-Procurement-Email-Extraction` | Contract2Plan [2601.06164] + ProUIE [2604.10633] | Dynamic-Lot-Sizing-MOQ |
| `Skill-LLM-Focused-Web-Crawling` | Webscraper [2603.29161] + W→K→W Pipeline [2602.24262] | Competitor-Intelligence, Listing-Quality-Scoring |
| `Skill-Adaptive-Crawl-Scheduling` | SB-CLASSIFIER [2602.11874, EDBT'26] + Neural Prioritisation [2506.16146] | — |
| `Skill-Web-Page-Change-Detection` | DiffSpot [2605.29615] + DOM Atomicity [2603.00476] | — |
| `Skill-Review-Dedup-Quality-Filter` | FOLD [2606.03001] + ResLPO [2601.07449] + Scalable ABSA [2602.21082] | AGRS, Review-Pain-Point-Mining |
| `Skill-Fake-Review-Detection` | JARVIS [2602.12941] + DS-DGA-GCN [2603.08332] + CAMERA [2605.20032] | — |
| `Skill-Clickstream-Persona-Pipeline` | SimPersona [2605.14205] + BIP [2604.22762] | TRACE-Clickstream, Trajectory-Pattern |
| `Skill-Privacy-Safe-Identity-Resolution` | Sherpa.ai PSU [2604.19219] + Cross-Domain SID [2606.01396] + CAMP [2604.16521] | HGNN-Cross-Device, CDA-Cookieless |
| `Skill-Ecommerce-Data-Quality-Assessment` | MESReduce [2603.08612] + MMPCBench [2601.19750] | Data-Drift-Detection |
| `Skill-Synthetic-Data-Ecommerce` | SIGIR'26 [2602.23620] + ICML'26 [2602.07298] + SCALR [2606.00282] | New-Product-Inventory-Coldstart |
| `Skill-Data-Provenance-Lineage` | Tracing Roots [2604.10480] + DEBUGLM [2603.17884] | — |
| `Skill-Privacy-Preserving-Federated-Collection` | SF-UBM [2604.14833] + MFG-RegretNet [2603.28329] | — |

**所有代码测试**：55/55 通过

---

## 二、项目当前状态（2026-06-05 EOD）

| 指标 | 数值 |
|------|------|
| 图谱节点（Skills）总数 | **315**（302 原有 + 13 新建） |
| 图谱边数（估算） | **5,500+** |
| 领域数 | **21**（新增 22-数据采集工程） |
| MAS 领域 Skills | **34** |
| 22-数据采集工程 Skills | **13**（全新） |
| HIGH 缺口 | **0** |
| 新建跨域桥梁 | **2**（mas↔advertising / mas↔KG） |
| 本轮代码测试 | **99/99 通过** |

---

## 三、关键文件路径

### Vault（Skill 卡片）
```
paper2skills-vault/
├── 10-MAS/                              # MAS 领域（34 Skills）
│   ├── Skill-MAS-Dynamic-Trust.md
│   ├── Skill-MAS-Testing-Verification.md
│   ├── Skill-MAS-Resource-Scheduling.md
│   ├── Skill-MAS-Consensus-Mechanism.md
│   ├── Skill-MAS-Scale-Management.md
│   ├── Skill-LLM-AutoBidding-MAS.md     ← mas↔advertising 桥梁
│   ├── Skill-MAS-Adversarial-Defense.md
│   ├── Skill-Cross-Org-Agent-Protocol.md
│   └── Skill-MAS-Dynamic-KG-Collaboration.md  ← mas↔KG 桥梁
└── 22-数据采集工程/                     # 新领域（13 Skills）
    ├── 00-INDEX.md
    ├── Skill-Document-Intelligence-Parsing.md
    ├── Skill-Procurement-Email-Extraction.md
    ├── Skill-LLM-Focused-Web-Crawling.md
    ├── Skill-Adaptive-Crawl-Scheduling.md
    ├── Skill-Web-Page-Change-Detection.md
    ├── Skill-Review-Dedup-Quality-Filter.md
    ├── Skill-Fake-Review-Detection.md
    ├── Skill-Clickstream-Persona-Pipeline.md
    ├── Skill-Privacy-Safe-Identity-Resolution.md
    ├── Skill-Ecommerce-Data-Quality-Assessment.md
    ├── Skill-Synthetic-Data-Ecommerce.md
    ├── Skill-Data-Provenance-Lineage.md
    └── Skill-Privacy-Preserving-Federated-Collection.md
```

### 代码（可运行验证）
```
paper2skills-code/
├── mas/
│   ├── dynamic_trust/model.py
│   ├── testing_verification/model.py
│   ├── resource_scheduling/model.py
│   ├── consensus_mechanism/model.py
│   ├── scale_management/model.py
│   ├── autobidding/model.py
│   ├── adversarial_defense/model.py
│   ├── cross_org_protocol/model.py
│   └── dynamic_kg/model.py
└── data_collection/
    ├── document_intelligence/model.py
    ├── procurement_email/model.py
    ├── web_crawling/model.py
    ├── adaptive_crawl/model.py
    ├── change_detection/model.py
    ├── review_dedup/model.py
    ├── fake_review_detection/model.py
    ├── clickstream_persona/model.py
    ├── identity_resolution/model.py
    ├── data_quality_assessment/model.py
    ├── synthetic_data/model.py
    ├── data_provenance/model.py
    └── federated_collection/model.py
```

### 选题计划文档
```
drafts/analysis/
├── mas-2026-paper-selection-plan-20260604.md    # MAS 原始选题（未排序版）
├── mas-2026-kg-path-plan-20260604.md            # MAS 图谱路径版（执行蓝本）
└── data-collection-2026-paper-selection-plan-20260605.md  # 数据采集选题
```

---

## 四、Skill 卡片格式规范（Codex 萃取参考）

每张 Skill 卡片遵循以下 5 段结构：

```markdown
## ① 算法原理
- 核心思想（业务语言，非论文摘要抄录）
- 关键算法/公式（数学直觉优先）
- 多论文对比表格（互补关系说明）

## ② 母婴出海应用案例（1-2 个具体场景）
- 场景：具体业务痛点 + 数据要求 + 预期产出 + 量化效果

## ③ 代码模板
- 路径：paper2skills-code/<领域>/<topic>/model.py
- 可运行 Python 代码，含核心类和测试用例

## ④ 技能关联
- 前置技能（prerequisite）
- 延伸技能（extends）
- 可组合技能（combinable）

## ⑤ 商业价值评估
- ROI 预估（量化）
- 实施难度（⭐ 评级）
- 优先级评分（⭐ 评级）
- 论文来源（arXiv ID）
```

**审核标准**：4 维度各 2.5 分，总分 ≥7/10 通过。

---

## 五、图谱结构关键规则（Codex 操作须知）

### Edge 类型
- `prerequisite`：必须掌握才能理解本 Skill（有向，强依赖）
- `extends`：本 Skill 的自然发展方向（有向）
- `combinable`：组合使用效果更好（双向）

### 新建 Skill 的图谱操作优先级
1. **prerequisite 断链修复最高优先**（阻塞下游）
2. 孤立节点接入（已存在但无连接）
3. Hub Skill 延伸链补强
4. 跨域桥梁建立（连通两个大领域）

### 领域编号
```
01 因果推断 | 02 A/B实验 | 03 时间序列 | 04 供应链 | 05 推荐系统
06 增长模型 | 08 知识图谱 | 09 DataAgent-LLM | 10 MAS | 11 AI人文
12 ML基础   | 13 广告分析 | 14 用户分析 | 15 营销投放 | 16 智能体工程
17 价格优化 | 18 物流履约 | 19 风控反欺诈 | 20 AI视频生成
21 合规决策 | 22 数据采集工程（本 Session 新建）
```

---

## 六、下一步建议

### 立即可做（论文已确认）
1. 运行 `python paper2skills-skills/paper-skills-graph/scripts/skills_graph_analyzer.py --analyze` 验证最新图谱状态
2. 更新 `00-项目管理/项目总结.md` 中 10-MAS 和 22-数据采集工程的 Skill 数

### 下一轮萃取候选
- `causal_inference ↔ advertising` 跨域桥梁（Causal MMM Bayesian）
- `time_series ↔ mas` 桥梁（时序感知 MAS）
- 21-合规决策领域扩充（Recall Risk Prediction）

---

*本摘要生成于 2026-06-05，覆盖 session ses_16f294c5fffeloBYidiiN8FVS1 的完整工作内容。*
