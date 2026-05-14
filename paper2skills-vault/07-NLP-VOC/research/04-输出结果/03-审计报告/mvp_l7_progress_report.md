---
name: mvp-l7-progress-report
description: VOC 深度分析 MVP L7 部署+文档+培训阶段进度报告。部署确认（腾讯云生产已在线），SOP v1→v2 升级覆盖 L4-L6 全资产清单（13 dashboard / 42 chart / 10 dataset），新增 888KB 单文件 HTML 业务方培训手册（5 场景 / 8 FAQ / 6 内嵌截图 / Playwright 验收）。0 生产代码改动，纯文档交付。MVP 全部 7 个 L 阶段完成 · 13/13.5 工日 · 0 LLM 成本 · 0 schema 改动。
title: MVP L7 · 部署 + 文档 + 培训
doc_type: progress-report
module: voc-nlp
phase: mvp-l7
status: completed
created: 2026-05-14
updated: 2026-05-14
owner: self
source: ai
---

# MVP L7 · 部署 + 文档 + 培训 进度报告

> 落地日期：2026-05-14 · 上线分支：`feat/voc-deep-analysis-mvp` · 上一节点：[L6 D-Action 7 部门行动总队列](mvp_l6_progress_report.md)。

## 一、L7 范围决策（用户确认）

| 项 | 用户选择 | 工作量 |
|---|---|---|
| **部署** | **A · 部署已完成** · 仅更新文档 | 0d |
| **文档** | **A · 仅升级 SOP**（技术运维向） | ~0.5d |
| **培训** | **A · 静态 HTML 培训手册** | ~0.5d |

实际花费：~1 工日（与计划 1d 一致），0 LLM 成本，0 生产数据改动。

## 二、上线对象

### 2.1 Superset_BI_SOP.md · v1 → v2 升级

| 维度 | v1 (Phase 7) | v2 (L7) | Δ |
|---|---:|---:|---:|
| 行数 | 731 | **1008** | +277 |
| URL 主入口 | `localhost:8088` | **腾讯云 prod URL** | switched |
| Dashboard 文档清单 | 8 个 | **13 个** | +5 |
| Chart 文档清单 | 12 (mention) | **42** (完整 L4/L5/L6 清单) | +30 |
| Dataset 文档清单 | 6 | **10** | +4 |

**新增 §B 章节**（MVP L4-L6 资产清单 + 重建/回滚）：

- §B.1 资产规模对比 + 工程周期演进
- §B.2 4 个 MVP 新视图（mvp_l3_views.sql）
- §B.3 MVP L4 D-Health 全清单（dataset 7-10 / chart 13-19 / dashboard 9）
- §B.4 MVP L5 D-Diag 三专题全清单（chart 20-34 / dashboard 10/11/12 + 15 chart 详情表）
- §B.5 MVP L6 D-Action 双路径全清单（chart 35-42 / dashboard 13 + 优先级公式 + 关键过滤）
- §B.6 完整 MVP 重建顺序（10-15 分钟）
- §B.7 完整 MVP 回滚顺序（5-10 分钟）
- §B.8 6 个 L 阶段备份目录索引

### 2.2 VOC_业务方培训手册.html（v2 新增）

| 维度 | 值 |
|---|---|
| 路径 | `research/01-设计文档/07-操作指南/VOC_业务方培训手册.html` |
| 大小 | **888 KB**（含 6 张内嵌 JPEG 1200w q60） |
| 形式 | 单文件 HTML · Docusaurus 风格（左 TOC + 右滚动） |
| 受众 | 7 部门业务方（非技术决策层） |
| 自包含 | 0 外部 CDN · 邮件分发可直接打开 |
| 打印 | `@media print` 隐藏 TOC，正文铺满（A4 友好） |

**内容结构**：

| 节 | 内容 |
|---|---|
| 速记卡 | 4 KPI（364,569 / 42 chart / 13 dashboard / 7 部门） |
| §1 13 张看板导览 | 13 dashboard 卡片 + URL + 谁用 + 看什么 |
| §2 5 个典型场景 | 产品经理 / 客服 / 供应链 / 品控 / 老板 3 步走 |
| §3 过滤器使用 | 5 步走 + Overview 限制说明 |
| §4 FAQ | 8 个常见问答 |
| §5 求助流程 | 问题类型 × 找谁 × 必备信息 表 |

**6 张内嵌截图**：
- 产品中心 dashboard（dept dashboard 示范 with D-Action）
- D-Health（管理层 KPI 入口）
- D-Diag-Product / Service / Brand（3 专题）
- D-Action 总览（老板入口）

### 2.3 00-INDEX.md（操作指南索引）

- 新增 §A 技术运维 / §B 业务方培训 分组
- 按场景查表新增「重建 MVP L4-L6」「回滚 MVP L4-L6」「业务方第一次用看板」三个常见场景
- 关联文档增加 L4/L5/L6/L7 进度报告链接

## 三、验收（Playwright 端到端）

### 3.1 HTML 结构

| 维度 | 期望 | 实测 |
|---|---:|---:|
| h2 大章节 | 5 | **7** ✅ |
| 内嵌图片 | 6 | **6** ✅（所有 naturalWidth=1200px） |
| TOC 链接 | ≥10 | **14** ✅ |
| Dashboard 卡片 | 6 | **6** ✅ |
| Scenarios | 5 | **5** ✅ |
| FAQ | 8 | **8** ✅ |

### 3.2 交互

| 维度 | 实测 |
|---|---|
| 锚点跳转 | 点击 `#sc-5` → scrollY=8927 ✅ |
| Console errors | **0** ✅ |
| PageError | **0** ✅ |
| Print mode | TOC 隐藏 · 正文铺满 ✅ |

### 3.3 视觉抽查（multimodal-looker）

- [manual_top.png](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L7-20260514-173412/manual_top.png)：**ready for distribution**
- [manual_print_mode.png](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L7-20260514-173412/manual_print_mode.png)：**print-ready**

## 四、对存量资产的影响（0）

| 类型 | L7 前 | L7 后 | Δ |
|---|---:|---:|---:|
| Superset dashboard | 13 | 13 | 0 |
| Superset chart | 42 | 42 | 0 |
| Superset dataset | 10 | 10 | 0 |
| Postgres view | 10 | 10 | 0 |
| voc_review schema | — | — | 0 |
| dim_tag schema | — | — | 0 |
| 文档行数（SOP） | 731 | **1008** | +277 |
| 文档新增（HTML） | 0 | **1 × 888KB** | +1 |

**完全 0 生产代码改动 · 纯文档交付**。

## 五、回滚预案

```bash
# 1. 回滚 SOP
cp ~/.secrets/backups/voc-deep-analysis-mvp/L7-*/Superset_BI_SOP.md.pre_l7 \
   research/01-设计文档/07-操作指南/Superset_BI_SOP.md

# 2. 删除业务方手册
rm research/01-设计文档/07-操作指南/VOC_业务方培训手册.html

# 3. 还原 00-INDEX.md（git revert L7 commit）
```

## 六、备份索引

`~/.secrets/backups/voc-deep-analysis-mvp/L7-20260514-173412/`

| 文件 | 用途 |
|---|---|
| `MANIFEST.md` | L7 baseline 完整说明 |
| `Superset_BI_SOP.md.pre_l7` | v1 SOP 备份（731 行） |
| `Superset_BI_SOP.md.post_l7` | v2 SOP（1008 行） |
| `00-INDEX.md.post_l7` | 升级后索引 |
| `VOC_业务方培训手册.html` | 业务方培训手册（888 KB） |
| `shots_compressed/` | 6 张 1200w JPEG q60 截图源 |
| `manual_{top,middle,bottom,print_mode}.png` | Playwright 验证截图 |
| `playwright_l7_results.json` | 端到端验证结果 |

## 七、MVP 全部完成 🎉

| 阶段 | 状态 | 工日 | 累计 |
|---|:---:|---:|---:|
| L0 准备 | ✅ | 1.0 | 1.0 |
| L1 voc_review 扩展 | ✅ | 0.5 | 1.5 |
| L2 50 SAT 映射 | ✅ | 2.0 | 3.5 |
| L3 4 深度视图 | ✅ | 1.0 | 4.5 |
| L4 D-Health 看板 | ✅ | 2.0 | 6.5 |
| L5 D-Diag 3 专题 | ✅ | 4.0 | 10.5 |
| L6 D-Action 7 部门 | ✅ | 1.5 | 12.0 |
| **L7 部署+文档+培训** | **✅** | **1.0** | **13.0** |

**13.0 / 13.5 工日预算 · 提前 0.5 工日完成 · 0 LLM 成本 · 0 生产 schema 改动**。

### 总收益

| 资产 | 数量 |
|---|---:|
| 新 Superset dashboard | 5（id 9-13） |
| 新 Superset chart | 30（id 20-42 减去 L4 重建的 4 个） + 7（L4 13-19）= 30 净增 |
| 新 Postgres view | 4（mvp_l3_views） |
| 新 voc_review 列 | 2（country + ts_inferred） |
| 新 dim_tag 列 | 1（atomic_indicator_id） |
| 50 SAT 原子指标 | 已定义 + 映射到 267 tag |
| 总建设周期 | 13 工日（含 L0 准备） |
| LLM 成本 | 0（沿用现有数据 + 字典） |

### 关键发现

- **产品力 Top 3**：延迟 4046.9 / 尺码小 2843.8 / 使用体验差 2183.2 → 产品研发部「核心体验改良包」
- **服务力极端**：品质管理中心 78.71% 负向（US 90.09% / UK 76.24%）
- **品牌监测信号**：4 个竞品（Elvie/Tommee Tippee/Spectra/Philips Avent）100% 负向 → 竞品声量监测而非内部整改

## 八、文件清单（提交到仓库）

| 文件 | 说明 | 操作 |
|---|---|---|
| `research/01-设计文档/07-操作指南/Superset_BI_SOP.md` | v1 → v2 | modified |
| `research/01-设计文档/07-操作指南/VOC_业务方培训手册.html` | 业务方培训手册（888 KB） | new |
| `research/01-设计文档/07-操作指南/00-INDEX.md` | 索引升级 | modified |
| `research/04-输出结果/03-审计报告/mvp_l7_progress_report.md` | 本进度报告 | new |

## 九、后续可选（Phase 8 候选 · 已超出 MVP 范围）

| 优先级 | 方向 | 工日预估 |
|---|---|---:|
| 🟡 加固 | 改默认密码 + RBAC + HTTPS 证书检查 | 1 |
| 🟡 加固 | SSO 接入（钉钉/飞书） | 2 |
| 🟢 数据 | 真实 timestamp 入库 → 启用时间维度过滤 + 还原 line chart | 2 |
| 🟢 数据 | 月度 ETL cron + 自动刷新 | 1 |
| 🟢 体验 | 修复 Superset 4.1.1 饼图 dashboard-mode bug（升级 4.2+） | 1 |
| 🟢 体验 | 部门权限隔离（每部门只看自己 + 总览） | 1 |
| 🔵 业务 | 第二品牌迁移试点（1-2 天迁移成本验证） | 3-5 |
| 🔵 业务 | 与公司主数仓打通（与订单/退货 JOIN） | 5 |
