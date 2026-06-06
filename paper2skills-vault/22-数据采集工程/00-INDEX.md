---
title: 22-数据采集工程 技能索引
doc_type: index
module: 22-数据采集工程
status: active
created: 2026-06-05
updated: 2026-06-05
---

# 22-数据采集工程（Data Collection Engineering）技能索引

## 领域定位

**图谱层级**：Layer 1 基础层 — 所有分析 Skill 的上游数据前置

```
22-数据采集工程（获取原始数据）
    ↓ 提供干净的结构化数据
09-DataAgent-LLM（分析已有数据）
    ↓
其余所有分析领域（因果/推荐/增长/MAS...）
```

与 `09-DataAgent-LLM` 的区别：
- DataAgent = 分析**已有**数据（SQL/Dashboard/RCA）
- 数据采集工程 = **获取**原始数据（爬取/解析/清洗/合成）

## 技能分类

### A. 智能网页爬取与变化检测

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-LLM-Focused-Web-Crawling](./Skill-LLM-Focused-Web-Crawling.md) | 已萃取 | Webscraper + W→K→W (2026) | MLLM 导航动态 JS 页面，KG 驱动竞品发现 |
| [Skill-Adaptive-Crawl-Scheduling](./Skill-Adaptive-Crawl-Scheduling.md) | 已萃取 | SB-CLASSIFIER + Neural Prioritisation (2026) | Bandit 调度最大化采集效率，节省配额 |
| [Skill-Web-Page-Change-Detection](./Skill-Web-Page-Change-Detection.md) | 已萃取 | DiffSpot + DOM Atomicity (2026) | 仅在竞品页面变化时触发全量抓取 |

### B. 文档智能与非结构化数据解析

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Document-Intelligence-Parsing](./Skill-Document-Intelligence-Parsing.md) | 已萃取 | dots.mocr + Qianfan-OCR + MinerU-Popo (2026) | 解析供应商 PDF 报价单/发票/跨页表格 |
| [Skill-Procurement-Email-Extraction](./Skill-Procurement-Email-Extraction.md) | 已萃取 | Contract2Plan + ProUIE (2026) | 结构化提取邮件中的 MOQ/交期/价格梯度 |

### C. 评论 / UGC 采集质量

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Review-Dedup-Quality-Filter](./Skill-Review-Dedup-Quality-Filter.md) | 已萃取 | FOLD + ResLPO + Scalable ABSA (2026) | 多平台评论在线去重+质量排序 |
| [Skill-Fake-Review-Detection](./Skill-Fake-Review-Detection.md) | 已萃取 | JARVIS + DS-DGA-GCN + CAMERA (2026) | 刷评群组检测，过滤竞品恶意差评 |

### D. 点击流采集 / 用户身份解析

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Clickstream-Persona-Pipeline](./Skill-Clickstream-Persona-Pipeline.md) | 已萃取 | SimPersona + BIP (2026) | 原始点击流→离散 Persona，含 session 拼接 |
| [Skill-Privacy-Safe-Identity-Resolution](./Skill-Privacy-Safe-Identity-Resolution.md) | 已萃取 | Sherpa.ai + Cross-Domain SID (2026) | 跨平台 ID 解析，GDPR/PIPL 合规 |

### E. 数据质量 / 合成数据 / 血缘追踪

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Ecommerce-Data-Quality-Assessment](./Skill-Ecommerce-Data-Quality-Assessment.md) | 已萃取 | MESReduce + MMPCBench (2026) | 商品 catalog 错误检测与缺失模态补全 |
| [Skill-Synthetic-Data-Ecommerce](./Skill-Synthetic-Data-Ecommerce.md) | 已萃取 | SIGIR'26 + ICML'26 + SCALR (2026) | 新品冷启动合成数据，幂律 Scaling Law |
| [Skill-Data-Provenance-Lineage](./Skill-Data-Provenance-Lineage.md) | 已萃取 | Tracing Roots + DEBUGLM (2026) | AI 法规审计数据血缘追踪 |
| [Skill-Privacy-Preserving-Federated-Collection](./Skill-Privacy-Preserving-Federated-Collection.md) | 已萃取 | FL + MFG-RegretNet (2026) | 跨平台联邦采集，隐私预算商品化 |

## 统计数据

- 总技能数：13
- 已萃取：13 ✅
- 待萃取：0
- 解锁下游 Skill 数：~15 个（供应链/VOC/用户分析领域）

## 与下游领域的衔接

| 下游领域 | 衔接点 | 本领域提供什么 |
|---------|--------|-------------|
| 04-供应链 | Supplier-Capacity-Planning / Multi-SKU-Procurement | 供应商 PDF/邮件结构化数据 |
| 14-用户分析 | TRACE-Clickstream / Trajectory-Pattern | 原始点击流 + 用户身份拼接 |
| 07-VOC（已迁） | AGRS / Review-Pain-Point | 去重去噪后的高质量评论 |
| 13-广告分析 | Ad-Attribution / CDA-Cookieless | 跨设备 ID 解析数据 |
| 06-增长模型 | Cold-Start / Bass-Diffusion | 合成数据补充新品冷启动 |

---

> 最后更新: 2026-06-05 | 全部 13 个 Skills 萃取完毕，Wave 1-4 全部交付
