# PHASE3 批次 3A–3E 交付归档

> 本文件由 `CLAUDE.md` 移出（2026-09-13），原因是 CLAUDE.md 体积超过 workspace
> instruction 预算（65,536 bytes）被截断。**内容一字未改**，只是换了存放位置。
> CLAUDE.md 保留一句指针，需要逐卡明细时查这里。

---

### PHASE3 批次 3A 交付(4 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `13-广告分析/Skill-Cannibalization-Corrected-Attribution.md` | 2606.26690 | ADKDD 2026 (**workshop**) | PASS | 18/18 | ✅ | ✅ |
| `13-广告分析/Skill-Funnel-Causal-Coupon-Allocation.md` | 2608.11675 | CIKM 2026 | PASS | 28/28 | ✅ | ✅ |
| `13-广告分析/Skill-Causal-Budget-Allocation.md` | 2608.10182 | arXiv preprint | PASS | 38/38 | ✅ | ✅ |
| `06-增长模型/Skill-Seasonal-Aligned-Churn-Label.md` | 2608.18174 | arXiv preprint | PASS | 40/40 | ✅ | ✅ |

### PHASE3 批次 3B 交付(4 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `03-时间序列/Skill-Decision-Conditioned-Forecasting.md` | 2608.25871 | **KDD 2026** (CCF-A) | PASS | 47/47 | ✅ | ✅ |
| `03-时间序列/Skill-Shipping-Cost-Estimation.md` | 2607.16230 | arXiv preprint | PASS | 36/36 | ✅ | ✅ |
| `04-供应链/Skill-Supply-Network-Simulation.md` | 2607.09745 | WSC 2026 (CCF-B) | PASS | 38/38 | ✅ | ✅ |
| `04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md` | 2606.29366 | arXiv preprint | PASS | 41/41 | ✅ | ✅ |

### PHASE3 批次 3C 交付(3 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `14-用户分析/Skill-Incrementality-Measurement.md` | 2607.09608 | arXiv preprint | PASS | 31/31 | ✅ | ✅ |
| `00-电商Agent/Skill-Live-Catalog-Conversational-Rec.md` | 2608.27006 | RecSys'26 **Demo** | PASS | 37/37 | ✅ | ✅ |
| `00-电商Agent/Skill-Agentic-Catalog-Enrichment.md` | 2608.20844 | arXiv preprint | PASS | 45/45 | ✅ | ✅ |

### PHASE3 批次 3D 交付(5 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `16-智能体工程/Skill-Stateful-Skill-Runtime.md` | 2608.26263 | arXiv preprint | PASS | 50/50 | ✅ | ✅ |
| `10-MAS/Skill-Multi-Agent-Collaboration-Tax.md` | 2608.22152 | arXiv preprint | PASS | 55/55 | ✅ | ✅ |
| `10-MAS/Skill-Routed-Graph-Handoff.md` | 2608.25277 | arXiv preprint | PASS | 51/51 | ✅ | ✅ |
| `09-DataAgent-LLM/Skill-SQL-Agent-Access-Control.md` | 2607.22115 | arXiv preprint | PASS | 39/39 | ✅ | ✅ |
| `02-A_B实验/Skill-Persona-Based-AB-Simulation.md` | 2609.01038 | arXiv preprint | PASS | 41/41 | ✅ | ✅ |

### PHASE3 批次 3E 交付(增强既有卡,不新增)

| 论文 | 被增强的卡 | 引用块 | 说明 |
|------|-----------|--------|------|
| 2608.28978 | `08-知识图谱/Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md` | 0 → **18/18** | 负结果:「反例与适用边界」 |
| 2608.28978 | `16-智能体工程/Skill-Agentic-Memory-Management.md` | 0 → **17/17** | 同上 |
| 2608.09162 | `12-ML基础/Skill-Feature-Engineering.md` | 0 → **30/30** | 数值特征变换形式化为优化问题（摘要称「一致优于所有基线」，但论文自己的 Table 1 里 `mlp`/`mlp_plr` 两行 PLE 优于 stretch —— 反例已写进 1b）|
| 2608.10240 | `05-推荐系统/Skill-Cold-Start-Meta-Learning-PAM.md` | 0 → **32/32** | 顺序模态丢弃 |

**这 16 张(3A–3D)也是全库唯一的 G2 通过者** —— 存量 130 张的 G2 全红,根因是**引用块为 0**。

