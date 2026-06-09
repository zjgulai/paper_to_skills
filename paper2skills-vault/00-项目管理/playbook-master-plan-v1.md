---
name: playbook-master-plan-v1
description: paper2skills Playbook 90天主计划 v1.0，基于金字塔原理（结论先行+MECE+SCQA）综合分析输出。覆盖叙事架构修复、内容质量提升、Skills Graph延伸、B2B销售闭环四大支柱，4个双周Sprint共81小时。
---

# paper2skills Playbook · 90天主计划 v1.0

> 生成日期：2026-06-08｜基于金字塔原理三工具全量审计

## 战略结论（结论先行）

**在90天内，将 paper2skills 从"内容充足但缺乏销售力"的知识库，转型为「顶会论文→跨境运营决策」转化率可量化、Playbook 端到端可演示、B2B 商机可追踪的母婴出海智决策平台。**

---

## SCQA 叙事

**S（情境）** 350 Skills / 22域 / 8工作流 / 14手册 / CEO白皮书 / Agent广场，覆盖母婴跨境电商全价值链，5904条图谱边，年化潜在ROI估算逾1.2亿元。

**C（冲突）** 四层系统性缺陷：
- 内容层：68.6%（240/350）problem_solved 算法优先而非业务优先
- 叙事层：仅14%（50/350）Skill 具备完整S+C叙事结构
- 手册层：14本手册全部缺少"Before State"导致价值不可感知
- 销售层：漏斗在"下载PDF"后完全断裂，0个Lead capture点

**Q（问题）** 在不增加新基础设施依赖的前提下，哪些改动能在90天内让平台从「内容充足但无法被卖出」变为「可演示、可售卖、可扩展」？

**A（答案）** 按「叙事修复 → 内容重写 → 图谱延伸 → 销售闭环」四柱推进，共81小时工作量，分4个双周Sprint。

---

## MECE 四柱工作分解

### Pillar A — 叙事架构修复（Narrative Fix）

| 缺陷 | 修复方向 | Sprint |
|---|---|---|
| 白皮书缺S+C+Q | 首屏前加4-6行SCQA引导段 | S1 |
| 14本手册无Before State | 每本补2-3句量化痛点描述 | S2 |
| 8个工作流无度量步骤 | 每个WF加Measurement Step + ≥2 KPI | S2 |
| 首页主信息无WIIFM | 改为「帮母婴跨境卖家用AI每年多赚3000万」 | S1 |
| 四Tab MECE失效 | 「自由探索」改为「技术/算法研究者」 | S2 |
| 三阶段命名维度混用 | 统一为「时间段+战略意图」 | S2 |
| 竞争差异化未陈述 | 白皮书首屏加「顶会论文→跨境运营决策翻译」定位语 | S1 |

### Pillar B — 内容质量提升（Content Quality）

三层问题：
- Level 1（最差）：problem_solved ≈ algorithm_summary → 163个Skill
- Level 2（次差）：algorithm_summary开头无业务语境 → 77个Skill
- Level 3（尚可）：有语境但缺数字/冲突 → 60个Skill

改写模板：「[业务角色]面临[具体场景]，传统方法[为什么失效]，导致[可量化损失]。」

| 任务 | Sprint |
|---|---|
| Level 1 高ROI前80张批量重写 | S1 |
| Level 1 剩余83张+Level 2批量重写 | S2 |
| 高分Skill(≥8.0)50张场景数字化增强 | S2 |
| doctor新增problem_solved相似度检测规则 | S2 |

### Pillar C — Skills Graph延伸（Graph Extension）

最弱桥接域对：
- `17-价格优化 ↔ 21-合规决策`：仅2条边
- `01-因果推断 ↔ 18-物流履约`：仅2条边

最高溢出比域（严重供给不足）：
- `11-AI人文`：溢出比7.0，需扩充到15+个Skill
- `21-合规决策`：溢出比7.0，7个Skill，需扩充

5个最高价值新Skill（按预测图中心性排序）：
1. `Skill-Compliant-Dynamic-Pricing-Guard` — 合规-定价双约束，预测度60-80
2. `Skill-KOL-ROI-Causal-Attribution` — KOL效果因果归因，预测度55-70
3. `Skill-Cross-Border-Cash-Flow-Forecasting` — 跨境现金流，填补运营财务空白
4. `Skill-Post-Purchase-Email-Sequence-Optimizer` — 购后复购邮件，拉升WF-H
5. `Skill-Multimarket-Expansion-Readiness-Scorer` — 多市场拓展就绪度，四域枢纽

3个应新建域：
- `23-运营财务`（Amazon现金流/融资），完全空白
- `24-KOL社媒运营`（达人投放管理），严重分散
- `25-账户健康与平台合规运营`（账号健康/申诉），高频痛点

| 任务 | Sprint |
|---|---|
| WF-I智能体工程专属工作流YAML | S3 |
| 合规/关税独立Handbook合并 | S3 |
| 新建3个合规补充Skill | S3 |
| Skill-Compliant-Dynamic-Pricing-Guard | S3 |
| Skill-Post-Purchase-Email-Sequence-Optimizer | S3 |
| Skill roadmap_phase字段标注（全350张） | S3 |
| 16-智能体工程代码可运行性核查 | S3 |
| Skill-KOL-ROI-Causal-Attribution | S4 |
| 评估新域23/24/25 Go/No-Go | S4 |

### Pillar D — B2B销售闭环（Sales Enablement）

当前漏斗断裂点：访客→内容→下载PDF→[断裂]→无Lead

修复四节点：
1. 首页Primary CTA按钮（Calendly/Typeform）
2. 白皮书Lead Capture（Formspree，零后端）
3. ROI计算器独立页（纯JS，无后端）
4. B2B Demo路径文档化（6步Demo流）

| 任务 | Sprint |
|---|---|
| 首页Hero区Primary CTA | S1 |
| 白皮书Lead Capture表单(Formspree) | S1 |
| GitHub Pages CI修复（所有Sales任务前置） | S1 |
| ROI计算器静态页 roi-calculator.html | S4 |
| B2B Demo路径文档化 | S4 |
| agents.html页尾联系表单 | S4 |
| Playbook按Phase分组展示 | S4 |
| 8个工作流覆盖地图可视化 | S4 |

---

## 完整TODO清单（P0→P2，共38项）

### P0 — 阻塞核心体验（Sprint 1 全部完成）

- [ ] S1-1 [P0][S][Tech] GitHub Pages CI修复 → AC: Pages可访问，CI绿色
- [ ] S1-2 [P0][S][Narrative] 白皮书首屏加SCQA引导段(S+C+Q 4行) → AC: 有情境→冲突→问题三层铺垫
- [ ] S1-3 [P0][S][Narrative] 首页主信息改WIIFM价值主张 → AC: Lead文案含「多赚X万」价值承诺
- [ ] S1-4 [P0][S][Tech] 首页Hero区加Primary CTA按钮 → AC: 可见行动按钮链接Calendly
- [ ] S1-5 [P0][S][Tech] 白皮书Lead Capture表单(Formspree) → AC: Email采集，POST成功
- [ ] S1-6 [P0][S][Narrative] 竞品差异化定位语 → AC: 白皮书含核心定位+3条差异点
- [ ] S1-7 [P0][S][Tech] 识别163个problem_solved重复Skill脚本 → AC: 输出JSON清单
- [ ] S1-8 [P0][M][Content] 批量重写前80张高优先级Skill → AC: 每张含业务触发句+冲突描述
- [ ] S1-9 [P0][S][Content] pb-risk-defense「800万vs30万」视觉callout → AC: 独立对比卡片

### P1 — 中期叙事与内容质量（Sprint 2）

- [ ] S2-1 [P1][M][Content] 14本手册各补Before State(2-3句含量化指标) → AC: 每本有痛点描述段
- [ ] S2-2 [P1][M][Content] 8个工作流各加度量步骤(WF-A~H) → AC: 每个WF有Measurement Step+≥2 KPI
- [ ] S2-3 [P1][M][Content] 批量重写后83张Skill → AC: 全部problem_solved无算法术语开头
- [ ] S2-4 [P1][S][Tech] 首页第四Tab改名(→「技术/算法研究者」) → AC: 四Tab统一角色轴
- [ ] S2-5 [P1][S][Content] 白皮书三阶段命名统一维度 → AC: 三阶段用「时间+战略意图」命名
- [ ] S2-6 [P1][S][Content] pb-pricing-engine intro加AIGP A/B数据 → AC: 「定价是乘数」有即时数字支撑
- [ ] S2-7 [P1][S][Content] pb-agent-replace 核心Hook做视觉callout → AC: Hook是视觉焦点
- [ ] S2-8 [P1][M][Content] Playbook index加竞品对比表 → AC: 3列对比（咨询/SaaS/paper2skills）
- [ ] S2-9 [P1][S][Tech] doctor新增problem_solved相似度检测规则 → AC: 0.85余弦相似度阈值

### P2 — 图谱延伸与销售闭环（Sprint 3-4）

- [ ] S3-1 [P2][M][Content] WF-I智能体工程工作流YAML设计 → AC: 5步覆盖Agent全生命周期
- [ ] S3-2 [P2][S][Tech] WF-I HTML生成并加入导航 → AC: wf-i.html与其他WF格式统一
- [ ] S3-3 [P2][M][Content] 合规/关税独立Handbook合并 → AC: pb-compliance.html上线
- [ ] S3-4 [P2][M][Content] 新建3个合规补充Skill(CPSC/HTS/申诉) → AC: doctor检查通过
- [ ] S3-5 [P2][M][Content] 新建Skill-Compliant-Dynamic-Pricing-Guard → AC: 桥接17↔21域
- [ ] S3-6 [P2][M][Content] 新建Skill-Post-Purchase-Email-Sequence-Optimizer → AC: 归入WF-H
- [ ] S3-7 [P2][M][Tech] Skill roadmap_phase字段标注(全350张) → AC: JSON含Phase1/2/3
- [ ] S3-8 [P2][M][Tech] 16-智能体工程代码可运行性核查 → AC: 0个import_error
- [ ] S4-1 [P2][L][Tech] roi-calculator.html静态页面 → AC: 3输入→ROI区间+Skill推荐
- [ ] S4-2 [P2][M][Strategy] B2B Demo路径文档化 → AC: demo-flow.md 6步路径
- [ ] S4-3 [P2][S][Tech] agents.html页尾联系表单 → AC: 同Formspree endpoint
- [ ] S4-4 [P2][M][Tech] Playbook按Phase分组展示 → AC: 每本手册Skill按Phase1/2/3 tab
- [ ] S4-5 [P2][M][Content] 8个工作流覆盖地图可视化 → AC: Mermaid/SVG展示WF-A~I
- [ ] S4-6 [P2][M][Content] 新建Skill-KOL-ROI-Causal-Attribution → AC: 桥接15↔20↔01域
- [ ] S4-7 [P2][M][Strategy] 评估新域23/24/25 Go/No-Go → AC: 输出决策文档

---

## Sprint计划

| Sprint | 周期 | 核心产出 | 估计工时 |
|---|---|---|---|
| S1 | W1-2 | Pages上线+SCQA破局+Lead capture开启+80张Skill改写 | ~16.5h |
| S2 | W3-4 | 全站叙事质量达Demo级+83张Skill完成+8个WF度量步骤 | ~22h |
| S3 | W5-6 | WF-I上线+合规Handbook+5个新Skill+图谱Phase标注 | ~25.5h |
| S4 | W7-8 | ROI计算器+B2B Demo流+图谱桥梁Skill | ~17.5h |
| **合计** | **8周** | | **~81h** |

---

## 核心一句话价值主张（应作为白皮书首句）

> 我们是唯一一家把顶会ML论文（NeurIPS/KDD/ICML）系统性翻译为母婴跨境运营决策的平台——350个可落地Skill，覆盖从断货预警到AI定价的全链路，每个决策都有学术证明和ROI数字，这是任何咨询公司和SaaS工具都无法复制的能力。

---

## 注意事项

1. **GitHub Pages修复是所有Sales任务的硬性前置**（S1-1必须第一优先）
2. **163张Skill改写风险**：批量处理容易同质化，每域保留1张人工校对样本作为few-shot基准
3. **WF-I设计是Sprint 3的关键路径**：D1.1需Sprint 3 Day 1完成Review
4. **Lead capture积累50+条后**再评估是否接入轻量CRM
