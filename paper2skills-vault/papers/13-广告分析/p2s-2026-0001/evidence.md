# evidence.md — p2s-2026-0001

> 本文件是 `Skill-Cannibalization-Corrected-Attribution.md` 的**证据档案**。
> 卡片 ⑥ 段是给读者的引用；本文件是给审核者的**核验记录**（含工具、命令、结果、未采信项）。

## 1. 论文元数据

| 项 | 值 |
|---|---|
| paper_id（registry） | `p2s-2026-0001` |
| arXiv | `2606.26690` |
| 标题（原文） | Attributed, But Not Incremental: Cannibalization-Corrected Attribution for Large-Scale Advertising |
| 作者单位 | TikTok（多市场生产系统） |
| venue | **ADKDD 2026**（KDD workshop，2026-08-10，济州）⚠️ **非 KDD 主会** |
| venue_tier | `workshop` |
| 全文存档 | `paper2skills-vault/papers/13-广告分析/p2s-2026-0001/fulltext.md`（36,259 字符，来自 arXiv LaTeXML HTML v1） |
| 抓取工具 | `paper2skills-research/scripts/fetch_fulltext.py --batch 3A` |

## 2. 核验记录

| 门禁 | 命令 | 结果 |
|---|---|---|
| 引文逐字核验 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <卡片>` | ✅ **VERBATIM 18/18**，0 近似，0 伪造 |
| 引文核验器自检 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest` | ✅ 能区分真引文 / 伪造 / 拼接（见 §5） |
| K1 代码可执行 | `python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <卡片> --level 5` | ✅ **PASS**（3 块拼接为 1 单元，6 个 `test_*` 全绿） |
| K2 · G2 事实可溯源 | `python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <卡片> --only G2` | ✅ **passed: true**，红灯 0 |
| K2 · G3 业务可落地 | 同上（不加 `--only`） | ✅ 红灯 0 |

## 3. 引文清单（逐字，与卡片 ⑥ 段一一对应）

每条均已在全文底本中连续命中（`quote_check.py` 的最长连续匹配段 = 引文全长）。

> ⚠️ **本表是索引，不是权威副本。** `quote_check.py` 只把 `> 原文："..."` 形式的块当作引用块，
> 因此下表的表格单元格形式**不参与**自动核验。权威副本在卡片 ⑥ 段（那 18 条已逐字核验 VERBATIM）。
> 改引文请改卡片 ⑥ 段，再回来同步本表；只改本表不会被任何门禁发现。

| # | 章节 | 引文（逐字） |
|---|---|---|
| 1 | §1 | We refer to this gap between credited conversions and causal incremental conversions as the attribution–cannibalization mismatch |
| 2 | §1 | Production attribution is timely, granular, and continuously available, but observational by construction. |
| 3 | §3.1（式 2 前） | we define the cannibalization rate as the fraction of nominal paid-attributed conversions that is not truly incremental |
| 4 | §3.4 | In deployment, we use a generalized linear model for robustness and operational simplicity. |
| 5 | §3.4 | as Huber loss to reduce the influence of extreme daily experimental points that remain after input filtering |
| 6 | §3.4 | Negative cannibalization estimates are treated as diagnostic signals for potential organic spillover, channel complementarity, or missing attribution touchpoints |
| 7 | §3.5 | producing actionable fine-grained corrections while satisfying three operational constraints: aggregate consistency, feasibility, and locality |
| 8 | §4.1 | the same experiment is not used both for calibration and evaluation within the same model version |
| 9 | §4.2 | This section summarizes the overall performance across 18 rounds of channel-level A/B incrementality experiments, covering up to eight markets. |
| 10 | §4.2 | Device ML reduces normalized calibration error by 69.11% relative to Raw Attribution, but its signed-error distribution remains relatively wide across experiment slices. |
| 11 | §4.2 | ETDC+HCA achieves the best overall calibration, reducing normalized calibration error by 91.38%, with median signed error close to zero and a narrower interquartile range. |
| 12 | §4.2 表 1 | \| ETDC+HCA \| 0.09 \| 91.38% \| 0.60% \| [-8.11%, 7.24%] \| |
| 13 | §4.3 | Raw Attribution consistently overestimates incremental contribution in these slices, with signed relative errors ranging from 179% to 334%. |
| 14 | §4.3 | In contrast, ETDC+HCA stays close to experimental lift across all reported slices, with signed relative errors ranging from -7% to 10%. |
| 15 | §5 | the measured overall cannibalization rate subsequently decreased by approximately 15 percentage points |
| 16 | §6 Limitations | The framework depends on the quality and coverage of incrementality experiments. Sparse experiments, wide confidence intervals, incomplete treatment isolation, or unresolved pull-forward effects can propagate uncertainty into calibration. |
| 17 | §6 Limitations | Proxy-based extrapolation also requires monitored relevance and approximate exogeneity; product launches, seasonality, market shocks, or acquisition-channel shifts may weaken these assumptions and require recalibration. |
| 18 | §6 Limitations | fine-grained outputs should be interpreted as calibrated allocations rather than independently identified unit-level causal effects |

## 4. 与 registry 记录的核对

registry（`papers_registry.json`）对本文的 `decision_reason` 写着：
「归因蚕食校正：用增量实验当因果锚点校正归因，直击'站内品牌词+站外种草'归因虚高。
TikTok 多市场部署，蚕食率降约 15pp」

逐项核对：

| registry 断言 | 核验结论 |
|---|---|
| 用增量实验当因果锚点校正归因 | ✅ 属实，§3.3–§3.4 |
| TikTok 多市场部署 | ✅ 属实，§5「Deployed across multiple global TikTok markets」（摘要）；§4.2「covering up to eight markets」 |
| 蚕食率降约 15pp | ✅ 属实，§5 原文 `approximately 15 percentage points` |
| 归因虚高（attribution overstates） | ✅ 属实，§4.3 给出 179%–334% 的正向误差区间 |

**结论：registry 记录与原文一致，无需修正。**

## 5. 引用核验器为何可信（不是「加了个检查」而已）

`quote_check.py --selftest` 的四个用例，其中**第 3 个是开发过程中实测发现的漏洞**：

| 用例 | 结果 |
|---|---|
| 真引文（论文原句） | ✅ VERBATIM，连续度 1.0 |
| 纯伪造（术语对但句子不存在） | ✅ FABRICATED，连续度 0.124 |
| **拼接**（摘要句 + 引言句缝成一句） | ✅ FABRICATED，连续度 0.681，**n-gram 召回却是 1.0** |
| 真引文含排版差异 | ✅ VERBATIM |

第 3 行是关键：该核验器最初用「全文档 n-gram 覆盖率」判定，而**拼接引文的覆盖率是 1.0**
（每个碎片都能在文档某处找到）→ 拼接可以通过。
改为「最长**连续**匹配段占引文的比例」后才拦得住。
这正是 `Cited but Not Verified`(arXiv:2605.06635) 所指的失效模式：
**「有引用」与「引用为真」是两件事，前者可达 >94%，后者实测只有 39–77%。**

## 6. 未能核验 / 有意保留为定性的项

- **Table 1 中 Raw Attribution 的绝对误差量级**：论文以商业敏感为由隐去，
  原文注：`Raw Attribution signed-error distribution and absolute magnitudes are omitted for confidentiality.`
  → 卡片中**未引用任何 Raw Attribution 的绝对数字**，只引用其归一化指数 1.00 与相对误差区间。
- **ETDC 与 ETDC+HCA 的渠道级指标相同**：论文注：
  `ETDC-only and ETDC+HCA have identical channel-level metrics because HCA preserves calibrated channel totals; HCA is evaluated in Section 4.4.`
  → 卡片据此说明 HCA 的价值体现在**局部性**（§4.4 定性），而非渠道级误差指标。
- **§4.4 局部性验证只有图，无数字**：Figure 3 的纵轴为 `absolute scale anonymized`
  → 卡片以定性方式表述（「把变化集中在目标渠道、无关渠道保持稳定」），未编造数字。
- **贵司侧的 ROI 参数**：论文无法提供。卡片 ⑤ 段只给公式与参数来源，
  不填具体金额 —— 填入具体数字会违反 R1（凭空数字比没有数字更有害）。

## 7. 时效性（R4 检查）

- 本文为 **2026-06-25 首发**的预印本（arXiv v1），同月即为 ADKDD 2026 录用稿，
  不存在「顶刊新发表但方法陈旧」的问题（该问题实测占顶刊文章的 79%）。
- 首次公开距今 < 3 个月，符合 `venue-whitelist.md` 的时效基准。

## 8. 本卡引出的工具改进（已回写）

撰写本卡的过程暴露并修复了 5 个门禁缺陷，均已落入脚本：

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | G2 只判「有无出处」，不判「出处真假」 | 伪造一段带数字的引文即可让任意数字过闸 | 新增 G2b 逐字核验（`quote_check.py`） |
| 2 | 两个脚本各有一套引用正则，互不兼容（`> 原文："..."` 匹配不到） | 带合格引文的卡片反被判「无任何出处」——**假红灯** | 统一抽取器，两脚本共用 |
| 3 | `evidence.md` 的数字无条件全收 | 在 evidence.md 里裸写一句 `ROI 提升 42.7%` 即可洗白 | 只认「已核验引文内」或「带出处指针的行」 |
| 4 | `NOISE_PATTERNS` 含 `^\d{1,2}$` | **所有 1–2 位数字**被豁免，含 `提升 15%` | 删除，改用结构前缀规则识别章节号 |
| 5 | `NOISE_PATTERNS` 含 `^v?\d+\.\d+$` | **所有小数**被豁免，含 `92.2%` | 仅认显式版本号 `v1.2.3` |

> 教训与 K1 迭代一致：**门禁工具的可信度必须先于被门禁对象建立。**
> 这 5 条里有 3 条是「门禁自己放水」（假绿灯），1 条是「门禁自己误伤」（假红灯）——
> 假绿灯比假红灯危险得多，因为它让人以为已经达标。
