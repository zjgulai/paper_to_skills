# evidence.md — p2s-2026-0006

> 本文件是 `03-时间序列/Skill-Decision-Conditioned-Forecasting.md` 的**证据档案**。
> 卡片 ⑥ 段是给读者的引用；本文件是给审核者的**核验记录**（含工具、命令、结果、未采信项）。

## 1. 论文元数据

| 项 | 值 |
|---|---|
| paper_id（registry） | `p2s-2026-0006` |
| arXiv | `2608.25871` |
| 标题（原文） | CEDAR: Controlled and Event-Driven Demand Forecasting via Residual Decomposition |
| 作者署名单位 | 中国科学技术大学（人工智能与数据科学学院）、阿里巴巴集团、香港科技大学（广州） |
| venue | **KDD 2026**（第 32 届 ACM SIGKDD，2026-08-09–13，济州岛） |
| venue_tier | `CCF-A`（`venue-whitelist.md` §2.2 与 §3 已核实出现在 KDD 2026 proceedings） |
| 论文集 DOI | `10.1145/3770855.3818338`（fulltext 头部；ISBN 979-8-4007-2259-2/2026/08） |
| 全文存档 | `paper2skills-vault/papers/03-时间序列/p2s-2026-0006/fulltext.md`（70,222 字符，来自 arXiv LaTeXML HTML v1） |
| 数据来源 | Alibaba 1688（**国内 B2B 批发平台**，非跨境场景） |

### ⚠️ 迁移性声明（写在卡片之前就应当明确）

论文的**方法**与**上线环境**都在中国国内电商平台。迁移到母婴出海（Amazon / 独立站）时，
以下三件事**必须重新取证**，不能引用论文数字代替：

1. **动作空间**：论文只有「折扣 + 广告花费」两个动作。跨境场景还要处理头程时效、
   汇率、平台政策，这些在本卡里属于 ①b 明确的边界外事项。
2. **事件信号**：论文的事件源是小红书/抖音热搜 + 节假日；跨境场景需要换成
   海外社媒/搜索趋势 + 目的地国节假日日历，语义分布不同。
3. **月龄生命周期**：论文没有这一类特征（母婴品类的内生漂移）。
   卡片 ③ 的「月龄阶段」是**本卡新增的观测特征**，属有意偏离论文，已在卡片中标注。

## 2. 核验记录（四条命令，全部为实际执行结果）

| 门禁 | 命令 | 结果 |
|---|---|---|
| K1 代码可执行 | `python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <卡片> --level 5 --timeout 60` | ✅ **PASS**（3 块拼接为 1 单元；L4 脚本执行成功 + L5 的 7 个 `test_*` 全绿；ENV_BLOCKED 0 / ORPHAN_DEP 0 / FAIL 0） |
| G2b 引文逐字核验 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <卡片>` | ✅ **VERBATIM 47/47**，0 近似，0 伪造，0 拼接 |
| 引文核验器自检 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest` | ✅ 自检通过（能区分真引文 / 伪造 / 拼接；拼接用例召回 1.0 但连续度 0.681，被正确拦下） |
| K2 · G1/G2/G3 | `python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <卡片> --k1 <K1 JSON>` | ✅ **G1 passed=true（红灯 0）／G2 passed=true（红灯 0）／G3 passed=true（红灯 0）** |

> **G2 残留 2 条黄灯**（非阻塞，已人工确认）：来自 frontmatter 的
> `created: 2026-09-12` / `updated: 2026-09-12`，被数字扫描器识别为一般数字 `12`。
> 该数字是**日期分量**，不是事实断言；不改动日期以免破坏 frontmatter 语义。
> 卡片正文与 ⑥ 段中**没有任何**黄灯级或红灯级未溯源数字。

> **G1 为什么必须带 `--k1`**：`gate_check.py` 的 G1 只读 K1 产物，不带 `--k1` 时
> 一律判 `G1-NO-EVIDENCE` 红灯（「禁止凭人工判断放行」）。本卡先跑
> `verify_skill_code.py --json-out <临时文件>`，再把该文件喂给 `gate_check.py --k1`。

## 3. 引文清单（逐字，与卡片 ⑥ 段一一对应）

> ⚠️ **本表是索引，不是权威副本。** `quote_check.py` 只把 `> 原文："..."` 形式的块当作引用块，
> 下表的表格单元格形式**不参与**自动核验。权威副本在卡片 ⑥ 段（那 47 条已逐字核验 VERBATIM）。
> 改引文请改卡片 ⑥ 段，再回来同步本表；只改本表不会被任何门禁发现。

| # | 出处 | 引文（逐字） |
|---|---|---|
| 1 | 2608.25871 Abstract（PDF 第 1 页） | `merchants need to evaluate sales outcomes under future action sequences such as budget schedules, rather than passively predicting what happens next.` |
| 2 | §1 | `what would happen if I follow a particular budget schedule (and other actions) over the next weeks?` |
| 3 | §1 | `This shifts the goal from passive forecasting to decision-conditioned simulation` |
| 4 | 2608.25871 Abstract（PDF 第 1 页） | `This design suffers from autoregressive inertia and conflates endogenous market evolution with decision-induced transitions, leading to policy-insensitive rollouts and unreliable counterfactual analysis.` |
| 5 | 2608.25871 Abstract（PDF 第 1 页） | `we propose CEDAR (Controlled and Event-Driven Demand forecasting via Action-aware Residual decomposition), a two-stage framework for robust decision-conditioned simulation` |
| 6 | §1 | `If these exogenous shocks are not separated from action effects, the simulator will misattribute demand changes, leading to erroneous credit assignment and disastrous budget decisions.` |
| 7 | §3.2 式 | `which reflects the natural causal structure of merchant operations: historical system states inform merchant decisions, and these decisions subsequently drive state transitions.` |
| 8 | §3.3 式 | `where $f_{\theta}$ captures controllable dynamics and $\epsilon_{t}$ represents latent external perturbations. Stage I models the former, while Stage II estimates the latter.` |
| 9 | §3.3 Residual Prediction | `The hotspot embedding $\mathbf{h}_{t}$ is then combined with the item status embedding via a cross-attention module, enabling dynamic alignment between external events and product-level temporal patterns.` |
| 10 | §3.4.2 | `For instance, the demand for seasonal items, such as Christmas hats, is unlikely to surge during the Chinese New Year, despite the presence of a significant festival event.` |
| 11 | §3.4.1 | `Consequently, we do not include explicit timestamp embeddings in CEDAR.` |
| 12 | §3.4.3 Inference Phase | `to perform multi-step forecasting, this result is appended to the historical sequence and fed back into the model in an auto-regressive manner, enabling the stable simulation of future product trajectories over an extended horizon.` |
| 13 | 2608.25871 Abstract（PDF 第 1 页） | `comprising approximately 32 million product trajectories with paired state–action sequences and aligned event signals` |
| 14 | §4.1.1 | `This process yields approximately 32 million training samples.` |
| 15 | §4.1.1 | `We segment each trajectory into overlapping windows of 15 consecutive weeks.` |
| 16 | §4.1.1 | `The action vector contains two controllable variables: the 7-day average marketing discount (defined as the ratio between the discounted price and the original price) and the total advertising expenditure during the same period.` |
| 17 | §4.1.1 | `we further incorporate exogenous event signals derived from both on-platform trending search topics and off-platform public hotspots, along with a curated list of 26 major holidays.` |
| 18 | §4.1.1 | `At the finest granularity, the taxonomy contains 8,942 distinct subcategories.` |
| 19 | §4.1.1 | `we reserve the final window of 2025 as the test set and use all preceding windows for training, because random sampling may cause shortcut learning on already observed external shocks.` |
| 20 | §4.1.3 | `we consider two evaluation settings: (i) forecasting the next 10 weeks given the past 5 weeks of observations, and (ii) forecasting the next 5 weeks given the past 10 weeks of observations.` |
| 21 | §4.1.3 | `The hidden dimension of all models is uniformly set to 256.` |
| 22 | §4.1.3 | `we configure the number of attention heads and layers as $n_{\text{head}}=4$ and $n_{\text{layer}}=5$, respectively.` |
| 23 | §4.1.3 | `we adopt the BGE-zh-v1.5 model as our text encoder, with an embedding dimension of 1024.` |
| 24 | §4.2 | `at horizon next 5, CEDAR attains an MSE of $0.182$, significantly outperforming the strongest baseline PatchTST $0.424$ and PETFormer $0.434$, corresponding to relative improvements of $57.1\%$ and $58.1\%$, respectively.` |
| 25 | §4.2 | `For the next 10 horizon, CEDAR also achieves the lowest MSE $0.414$ and NMSE $0.189$, consistently outperforming all competing approaches.` |
| 26 | §4.2 | `Similar trends are observed under NMSE, where CEDAR reduces the error to $0.083$, yielding more than $56\%$ improvement over the best baseline.` |
| 27 | §4.2 | `classical TSF models such as Informer suffer from severe performance degradation, with MSE exceeding $30$ at next 10.` |
| 28 | §4.2 表 2 | `| Full Model | 0.414 | 0.182 | 0.132 | 0.0603 |` |
| 29 | §4.2 表 2 | `| w/o AIT Prediction | 0.499 | 0.264 | 0.181 | 0.0697 |` |
| 30 | §4.2 表 2 | `| Temporal shuffle | 0.527 | 0.274 | 0.194 | 0.0712 |` |
| 31 | §4.3 | `baselines that treat actions as exogenous covariates exhibit limited sensitivity to intervention signals` |
| 32 | §4.3 | `This ability enables stable and realistic multi-step rollouts under dynamically changing action plans, which is critical for budget planning and strategy exploration.` |
| 33 | §4.4 | `This variant performs worse than the aligned-event setting and even degrades relative to AIT-only in next-5 MSE, indicating that CEDAR benefits from temporally meaningful event-demand alignment rather than merely using event embeddings as generic auxiliary features.` |
| 34 | §4.4 | `we observe that incorporating the Residual Correction Module yields a modest improvement in MSE but leads to a substantial reduction in MAE.` |
| 35 | §4.5 | `it differs from our setting because it is organized at the store-family level and lacks merchant actions with explicit budget-planning semantics` |
| 36 | §4.5 | `Thus, this experiment mainly tests whether the event-aware residual decomposition transfers to a public retail forecasting scenario, rather than fully reproducing our counterfactual budget-planning task.` |
| 37 | §4.5 | `CEDAR reduces MSE from $0.6321$ to $0.5819$ compared with the strongest baseline PETFormer, and also achieves lower MAE and NMSE than both PatchTST and PETFormer.` |
| 38 | §4.1.1 | `As the largest domestic B2B wholesale platform in China, Alibaba 1688 provides a uniquely rich environment for studying decision-conditioned forecasting and budget planning.` |
| 39 | §4.1.1 | `we are actively working toward releasing a partially anonymized version to facilitate future research.` |
| 40 | §4.6 | `The initial deployment successfully engaged 239 cooperative merchants across 245 orders for the prediction service, facilitating a total transaction value of 8.77 million RMB with an initial repurchase rate of 61%.` |
| 41 | §4.6 | `merchants in the treatment group achieve a 13%(46,471 vs 41,228) increase in lifetime value (LTV) and a 15% improvement in store-level return on investment (ROI) on average.` |
| 42 | §4.6 | `The control group follows the existing production model based on a diffusion-based time series forecasting model with only total budgets, while the treatment group adopts CEDAR-driven budget planning and traffic allocation strategies.` |
| 43 | §4.6 | `Together, these capabilities significantly reduce ineffective ad spend and improve the alignment between budget allocation and true market demand.` |
| 44 | 2608.25871 Appendix B 式 (10) 前（PDF 第 8 页） | `This approximation relies on the assumption of Sequential Ignorability, which posits that potential outcomes $S(a)$ are conditionally independent of the current action given the history` |
| 45 | 2608.25871 Appendix C（表 4；PDF 第 9 页） | `Specifically, Stage I takes about 154 minutes to train, while Stage II adds another 80 minutes for learning the residual correction module. The total training time of CEDAR is therefore approximately 234 minutes, which is higher than PatchTST but substantially lower than PETFormer in our implementation.` |
| 46 | 2608.25871 Appendix D（PDF 第 9 页） | `all models exhibit larger errors as the horizon increases, which is expected due to autoregressive error accumulation.` |
| 47 | 2608.25871 Appendix D（表 5；PDF 第 9 页） | `At next-25, CEDAR obtains an MSE of 1.612, substantially lower than PatchTST 2.847 and PETFormer 2.561.` |

## 4. 卡片正文每个数字 → ⑥ 段出处对照

> 本表回答「正文里的这个数字，是哪一条引文支撑的」。R1 要求一个数字一条出处。
> 下表由脚本从卡片 ⑥ 段的 47 条引文里按**数字整词边界**查找生成（不是人工填的，也不是子串匹配）。
> ⚠️ 它比 G2 门禁更严：门禁只要求该数字串在引文里**出现过**，本表另外排除了 `0.132` 里命中 `32` 这类子串假象（左侧不得紧邻数字或小数点，右侧不得紧邻数字）。

| 正文数字 | 含义 | 对应 ⑥ 引文（#） |
|---|---|---|
| `32` | 数据集体量：32 million 产品轨迹 / 训练样本 | #13、#14 |
| `15` | 轨迹切窗长度（周） | #15、#41 |
| `7` | 动作向量的 7-day 折扣口径 | #16 |
| `26` | 精选节假日数 | #17 |
| `8,942` | 三级细分品类数 | #18 |
| `10` | 评测设定中的周数（past/future） | #20、#25、#27 |
| `5` | 评测设定中的周数（past/future） | #20、#22、#24、#33 |
| `256` | 统一隐藏维度 | #21 |
| `4` | 注意力头数 | #22 |
| `1024` | 文本编码器嵌入维度 | #23 |
| `0.182` | next 5 的 MSE | #24、#28 |
| `0.424` | PatchTST 的 next 5 MSE | #24 |
| `0.434` | PETFormer 的 next 5 MSE | #24 |
| `57.1` | next 5 的相对提升 | #24 |
| `58.1` | next 5 的相对提升 | #24 |
| `0.414` | next 10 的 MSE | #25、#28 |
| `0.189` | next 10 的 NMSE | #25 |
| `0.083` | NMSE | #26 |
| `56` | NMSE 的相对提升 | #26 |
| `30` | Informer 在 next 10 的 MSE 下界 | #27 |
| `0.132` | 完整模型 MAE（表 2，next 10） | #28 |
| `0.0603` | 完整模型 MAE（表 2，next 5） | #28 |
| `0.499` | w/o AIT Prediction 的 MSE（表 2） | #29 |
| `0.0697` | w/o AIT Prediction 的 MAE（表 2） | #29 |
| `0.527` | Temporal shuffle 的 MSE（表 2） | #30 |
| `0.0712` | Temporal shuffle 的 MAE（表 2） | #30 |
| `0.6321` | Kaggle Store Sales：最强基线的 MSE | #37 |
| `0.5819` | Kaggle Store Sales：CEDAR 的 MSE | #37 |
| `239` | 线上部署：合作商家数 | #40 |
| `245` | 线上部署：订单数 | #40 |
| `8.77` | 线上部署：交易额（million RMB） | #40 |
| `61` | 线上部署：初始复购率 | #40 |
| `13` | 线上 A/B：LTV 改善 | #41 |
| `46,471` | 线上 A/B：LTV 分子 | #41 |
| `41,228` | 线上 A/B：LTV 分母 | #41 |
| `15` | 线上 A/B：店铺 ROI 改善 | #15、#41 |
| `154` | Stage I 训练时长（分钟） | #45 |
| `80` | Stage II 训练时长（分钟） | #45 |
| `234` | 合计训练时长（分钟） | #45 |
| `1.612` | next-25 的 MSE | #47 |
| `2.847` | PatchTST 的 next-25 MSE | #47 |
| `2.561` | PETFormer 的 next-25 MSE | #47 |

**未被上表引用的引文编号**（纯定性结论或迁移性边界，不含正文数字）：#1、#2、#3、#4、#5、#6、#7、#8、#9、#10、#11、#12、#19、#31、#32、#34、#35、#36、#38、#39、#42、#43、#44、#46。

## 5. 与 registry 记录的核对

`papers_registry.json` 对本文的 `decision_reason` 写着：
「CEDAR：决策条件化仿真，可回答'给定预算排期与备货计划的销量'，KDD 2026 正式发表并用于真实预算规划」，
并标注 `data_availability: available`。逐项核对：

| registry 断言 | 核验结论 |
|---|---|
| 决策条件化仿真 | ✅ **属实**。Abstract 与 §1 明确把目标从被动预测改为 decision-conditioned simulation |
| 可回答「给定 Budget Schedule 的销量」 | ✅ **属实**。§1 原文即「what would happen if I follow a particular budget schedule (and other actions) over the next weeks?」 |
| 可回答「给定**备货计划**的销量」 | ❌ **不属实**。论文的动作向量只有两个变量（折扣、广告花费，§4.1.1），**不含补货/备货决策**；`inventory planning` 只在 §1 作为 TSF 的通用用途出现，`operational planning` 只作为下游任务被提及。卡片据此把「补货量」实现为**仿真的下游换算**（`plan_replenishment`），不声称论文能直接回答备货计划。**建议把 registry 的该措辞改为「给定预算排期（折扣 + 广告）的销量」** |
| KDD 2026 正式发表 | ✅ **属实**。fulltext 头部含完整 proceedings 信息（32nd ACM SIGKDD, V.2, Jeju）与 DOI；`venue-whitelist.md` §3 也已核实 |
| 用于真实预算规划 | ✅ **属实，但主体不是跨境场景**。Abstract 写「delivers practical gains for real-world budget planning」；§4.6 报告 2026-01-01 至 2026-01-30 在 1688 广告与营销优化系统的线上 A/B。**平台是阿里 1688（国内 B2B），商家是平台商家，不是母婴出海卖家**；卡片据此在 ⑤ 明确写「不是贵司的预期收益」 |
| `data_availability: available` | ⚠️ **口径需修正**。论文两次表述为「正在推进发布部分匿名版本」（§1、§4.1.1），**当前并不对外可得**，1688 数据为平台私有。若 registry 的 `available` 语义是「企业内可自建等价面板」，则本卡成立（自有店铺的状态-动作面板可得，见卡片 ② 数据可得性）；若语义是「论文数据集可用」，则**不成立**，建议改为 `partial` |

## 6. 未能核验 / 有意保留为定性的项

1. **论文的 MSE/MAE 不能换算成业务量级**。§4.1.1 做了**按三级子类别的 z-score 归一化**
   （`we perform category-wise normalization`），因此表 1/表 2/表 3 的误差都是**归一化尺度**，
   论文未给出任何还原到「件/单/元」的口径。→ 卡片**不写**「误差相当于多少件」这类换算，
   也不把论文误差与本卡合成输出的误差并列比较。
2. **表 1 的表格文本在 HTML→Markdown 转换中损坏**。均值与标准差被拼成 `0.4140.015` 这类不可直接引用的形式
   （原意是 `0.414 ± 0.015`）。→ 卡片只引用 §4.2 正文里表述完整的数值，**不引用表 1 的合并单元格**。
   表注本身完整可引（`All offline results are reported as mean $\pm$ standard deviation over five independent runs.`），
   但为控制 ⑥ 段长度未收录。
3. **线上 A/B 的统计口径未报告**。论文给出 239 商家 / 245 订单 / 8.77 million RMB / 61% 复购率，
   以及 LTV 与店铺 ROI 的改善幅度，但**未报告实验单元划分、样本量、显著性检验与置信区间**。
   → 卡片把这些数字当作**体量与方向**的证据，不当作严格效应量证据。
4. **事件信号的召回率/精度未报告**。附录 A 只描述了「LLM 先滤噪抽标签、再与日历事件合成一句话」的两阶段流程，
   未给提示词全文、未给抽取质量评估。→ 卡片把事件信号的自建成本写成「需补充」，不写「已可复现」。
5. **长 horizon 只报告了 MSE**。附录 D 只给 next-15/20/25 的 MSE，没有 MAE/NMSE，也没有
   「轨迹多样性」「rollout 稳定性」的量化指标（§4.4 的多样性结论是定性的）。
   → 卡片在 ①b 只定性引用「自回归误差累积」，未编造长 horizon 的其他指标。
6. **无代码与无超参搜索细节**。论文未提供官方实现，表 4 只给 wall-clock，
   §4.1.3 只给核心超参（隐藏维度、头数、层数、文本编码器）。
   → 卡片 ③ 明确写「不是论文的忠实复现」，并逐条列出替代实现方式。
7. **「两个完整年度」是本卡的工程建议，不是论文结论**。论文用的是 2024–2025 两年数据，
   但**没有论证**为什么需要两年。→ 卡片 ② 的该建议来自「大促效应与月龄生命周期会共线」这一推理，
   已在卡片中按建议口径表述，不冒充论文结论。
8. **本卡模板的系统性偏差是本地发现，论文未讨论**。合成实验显示多步 rollout 会放大动作响应
   （模型弹性明显高于真值），论文只承认「误差随 horizon 增大」。→ 卡片 ③ 把这一条明确标注为
   **本卡实测、论文没有讨论**，并据此拒绝把绝对仿真值直接用于备货决策。
9. **事件信号的覆盖度与延迟**。附录 A 提到数据源为「major social platforms (eg.Red notebook and Douyin)」，
   但未报告采集频率、延迟与覆盖率。→ 卡片 ② 只写「需自建采集 + LLM 归纳」。

## 7. 时效性（R4 检查）

- registry 记录的 `published` 为 **2026-08-26**，论文正文标注的会议时间为 **2026-08-09–13**，二者同月；
  论文为 KDD 2026 正式 proceedings 论文，不存在「顶刊新发表但方法陈旧」的问题（该问题实测占顶刊文章的 79%）。
- 论文含真实线上 A/B（§4.6），不是纯理论、不是 survey、不是 demo/workshop 短文。
- 论文正文未见任何针对 LLM/审稿人的提示注入内容；无撤稿标记。
- 首次公开距今 < 3 个月，符合 `venue-whitelist.md` 的时效基准。

## 8. 本卡交付物与复现命令

```bash
C=paper2skills-vault/03-时间序列/Skill-Decision-Conditioned-Forecasting.md
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" \
  --level 5 --timeout 60 --json-out /tmp/k1_cedar.json
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$C"
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" --k1 /tmp/k1_cedar.json
```
