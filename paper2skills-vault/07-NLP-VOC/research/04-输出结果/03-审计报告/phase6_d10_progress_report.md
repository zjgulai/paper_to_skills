---
name: phase6-d10-progress-report
description: Phase 6 D10 BI 看板实质上线报告 — 7 部门 MAA+AGRS 周报（C）+ 静态 HTML 看板（A）双重交付，基于 D9 过滤后数据（Gate 7/7 + precision 0.896）。当审计 BI 看板从零到有、验证 C→A 路径交付效果时使用。
date: 2026-05-10
phase: phase6
day: D10
status: 🎉 C + A 双路径全部交付 — 7 部门周报 + 125KB 静态看板
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D10 进度报告 — BI 看板上线（C→A 路径）

> **总判定**：🟢 **C 路径（7 部门周报）+ A 路径（静态 HTML 看板）全部完成**。基于 D9 过滤后的 Gate 7/7 + precision 0.896 数据，产出 14 份真实周报（MAA + AGRS × 7 部门）+ 单文件离线看板（125 KB, Chart.js 可视化）。**BI spec §三 "上线节奏" 的 Phase 6 节点实质兑现**。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| **C.1** 7 部门 MAA 周报 | ✅ | 7 × MAA.md + 7 × MAA.json（2026-W19-d9/）|
| **C.2** 7 部门 AGRS 摘要 | ✅ | 7 × AGRS.md + 7 × AGRS.json |
| **C.3** bi_spec_validator 回归 | ✅ | 35 断言全过 |
| **A.1** HTML 看板设计 | ✅ | 单文件 + Chart.js CDN + 左侧 tab 导航 |
| **A.2** bi_dashboard_generator.py | ✅ | 从 14 JSON + 全量 jsonl 聚合 |
| **A.3** 浏览器实跑验证 | ✅ | 9/9 图表 + 8/8 tab + 68 SRAC 行 + 68 AGRS 分组 |

## 二、C 路径：7 部门真实周报

### 2.1 触发

Phase 5 收官后，BI spec §三 节奏要求 Phase 6 启动看板原型；D9 precision 0.896 的数据已就绪。

### 2.2 工程踩坑（1 处 bug 修复）

| 问题 | 现象 | 修复 |
|---|---|---|
| `maa_strategy_generator.py` 对 D9 数据抛 ValueError | `float("negative")` — D9 部分 labels 的 `sentiment_calibrated` 是字符串 | 新增 `_to_sentiment_float()` helper 支持 `"positive"/"negative"/"neutral"` → `[1,-1,0]` 映射 |

一次修复后 3 个失败部门（全球客服与体验中心/供应链中心/电商运营部）全部通过。

### 2.3 部门 MAA 输出样本（Top-3）

| 部门 | 总话题 | Top-1 标签 | SRAC 分差 |
|---|---:|---|---:|
| 产品中心/品线 | 62 | 质量感知 (20,192 hits) | **5.35** ✅ |
| 全球客服与体验中心 | 36 | 服务满意 (10,173 hits) | **5.42** ✅ |
| 品牌市场中心 | 34 | — | **5.03** ✅ |
| 供应链中心 | 39 | — | **5.27** ✅ |
| 电商运营部 | 34 | — | **5.57** ✅ |
| 品控部 | 14 | — | **6.46** ✅ |
| 质量与法规部 | 8 | 产品影响人身安全/健康状况 (646 hits) | **6.23** ✅ |

**所有 7 部门 SRAC spread ≥ 5**（D11 T11.1 QA 场景 1 Pass 标准）。

### 2.4 业务洞察样本（质量与法规部）

Top-1: `TAG_P1_014 产品影响人身安全/健康状况` — 646 命中
代表评论：
> "This bottle warmer is terrible. It heated a bottle of breastmilk to 140 degrees. No bottle warmer should be heating any bottle even close to that hot. Had we not checked the temp before giving the bottle to our baby..."

SRAC = **9.53 / 10**（最高严重度）。Phase 5 未曾直接暴露此类高危信号；BI 看板接入后产品安全合规部门可立即优先追踪。

## 三、A 路径：静态 HTML 看板

### 3.1 设计原则

| 原则 | 实现 |
|---|---|
| **零部署** | file:// 或简单 HTTP 皆可直开；无后端、无数据库 |
| **离线可用** | CSS 内联；Chart.js 通过 CDN（无 JS 时表格仍可读）|
| **自包含** | 单 HTML 文件 125 KB，涵盖全部数据 |
| **可扩展** | 生成器分离数据和模板，周度重跑即可更新 |

### 3.2 结构

```
+-------------------------------------+
| 左侧导航（220px）                    |
| - 📊 总览                            |
| - 全球客服与体验中心 / 产品中心/品线 / ...（7 部门）|
+-------------------------------------+
| 主内容区                              |
| 总览:                                 |
|   - 4 卡片: 评论数 / 有标签数 / 源 / 标签 |
|   - 数据源 Doughnut + 情感 Pie        |
|   - Top-20 标签表                      |
|   - NPS 分布表 + 情感分布表             |
| 部门页:                               |
|   - 5 卡片: Top 数 / 负 / 正 / 中 / 分差 |
|   - SRAC 横向柱状图                    |
|   - Top 10 SRAC 表                     |
|   - 10 AGRS 分组摘要（accordion）      |
+-------------------------------------+
```

### 3.3 浏览器实跑验证

启 Python HTTP 服务 → Playwright 加载 → 运行时断言：

| 断言 | 值 |
|---|---|
| Chart.js 加载 | `window.Chart === 'function'` ✅ |
| Canvas 注册 | **9/9** (2 overview + 7 dept SRAC) ✅ |
| Tab 页元素 | **8/8** 存在 ✅ |
| Tab 按钮 | **8/8** 存在 ✅ |
| 侧栏宽度 | 220px（设计一致）✅ |
| 卡片数量 | **39** (4 overview + 5×7 dept = 39) ✅ |
| 表格数量 | 11（overview 4 + 7 dept SRAC）✅ |
| SRAC 行 | **68**（7 部门 × 10 + 质量与法规部 8）✅ |
| AGRS 分组 | **68**（全部 7 × 10 左右）✅ |
| 产品中心/品线首行 | "质量感知 TAG_GEN_E003 8.59" ✅ 与 MAA JSON 一致 |
| Console error | 0（仅 favicon 404 非业务错误）|

### 3.4 Tab 切换实测

初始：`overview` 可见，其他 7 个隐藏。
点击 `产品中心/品线` tab 后：
- `tab-产品中心/品线` display=block ✅
- 其他 7 个 display=none ✅
- `tab-btn.active` 标记正确切换到 产品中心/品线

## 四、产出清单

| 文件 | 用途 | 大小 |
|---|---|---:|
| `04-输出结果/10-周报/2026-W19-d9/*.md` | 7 部门 MAA + 7 部门 AGRS = 14 MD | ~200 KB |
| `04-输出结果/10-周报/2026-W19-d9/*.json` | 14 结构化 JSON（供看板读取）| ~50 KB |
| `02-脚本工具/01-标签进化/maa_strategy_generator.py` | 新增 `_to_sentiment_float()` | 改 |
| `02-脚本工具/01-标签进化/bi_dashboard_generator.py` | 新建 HTML 看板生成器（~17K）| 新 |
| `04-输出结果/bi-dashboard/dashboard-2026-W19.html` | **静态看板（可直接浏览器打开）** | 125 KB |
| `04-输出结果/03-审计报告/phase6_d10_progress_report.md` | 本文档 | ~8 KB |

## 五、使用方式

### 5.1 看板打开

```bash
# 方式 1: file://
open paper2skills-vault/07-NLP-VOC/research/04-输出结果/bi-dashboard/dashboard-2026-W19.html

# 方式 2: 简单 HTTP 服务（规避部分浏览器 file:// 策略）
cd paper2skills-vault/07-NLP-VOC/research/04-输出结果/bi-dashboard
python3 -m http.server 8777
# 打开 http://localhost:8777/dashboard-2026-W19.html
```

### 5.2 更新频率

周度：生成新的 2026-Wxx 目录 → 跑 MAA + AGRS × 7 部门 → 生成器产出新 HTML。

### 5.3 接入月度 cron

可以把 `bi_dashboard_generator.py` 挂到 `monthly_evolution_cron.py` Step 8（bi_recompute），每月自动产出最新看板到 `bi-dashboard/dashboard-YYYY-MM.html`。

## 六、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | 看板数据是静态快照（周度更新）| 中 | 够用；未来升级 B 路径（Superset/Metabase）才需实时 |
| R2 | Chart.js 依赖 CDN | 低 | 离线 fallback：HTML 已有表格视图，无 JS 仍可读 |
| R3 | 无鉴权 / 公开数据 | 中 | 看板含 review 原文；分享前需脱敏或限内网 |
| R4 | 未接入 Proxy NPS 时序图 | 低 | D10 未要求；下一迭代 |

## 七、C→A→B 路径演进

```
C (本): 7 部门 Markdown 周报                  实用但不可视化
  ↓
A (本): 静态 HTML 看板（Chart.js）            轻量可视化
  ↓
B (未来): Superset / Metabase（B 路径）       可交互 BI（需部署）
```

D10 完成 C + A 两级。B 路径可作为 Phase 7 独立 Sprint（Docker 部署 + 数据仓库 + 账号权限），不阻塞当前成果使用。

## 八、Phase 5 + Phase 6 累计状态

```
Phase 5 D14 验收:    Gate 4/7 + QA-2 BLOCKED (部分收官)
  ↓
Phase 6 D1-D5:       Gate 7/7 PASS (数值)
  ↓
Phase 6 D6:          QA-2 unblocked
  ↓
Phase 6 D7-D9:       precision 0.639 → 0.896 (质量)
  ↓
Phase 6 D10 (本):    BI 看板实质上线（C + A 双交付）✅
```

**BI spec §三 "Phase 6 启动" 的"看板原型（飞书文档/单页）"节点真正兑现**。

## 九、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-10 15:50 | C.1 跑 7 部门 MAA（3/7 失败，发现 sentiment_calibrated 类型 bug）|
| 2026-05-10 15:55 | 修复 `_to_sentiment_float()` helper，7/7 通过 |
| 2026-05-10 16:01 | C.2 跑 7 部门 AGRS（全过）|
| 2026-05-10 16:05 | C.3 bi_spec_validator 35 断言 PASS |
| 2026-05-10 16:08 | A.1 设计 HTML 看板结构 |
| 2026-05-10 16:12 | A.2 `bi_dashboard_generator.py` 产出 125 KB 看板 |
| 2026-05-10 16:16 | A.3 Playwright 浏览器实跑 + 10+ 运行时断言全过 |
| 2026-05-10 16:25 | 本报告归档，BI 看板实质上线 |

## 十、一行总结

> Phase 6 D10 **C→A 双路径全部交付**：7 部门 MAA+AGRS 周报（14 MD+JSON, 基于 D9 precision 0.896 数据）+ 125 KB 静态 HTML 看板（9 Chart.js 图表 + 8 tab 导航 + 68 SRAC 行 + 68 AGRS 组，Playwright 浏览器实跑验证全过）。**BI spec §三 "Phase 6 启动看板原型" 实质兑现**，数据可用，可增量更新。
