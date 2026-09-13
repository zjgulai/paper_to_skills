> # ⛔ 本文件已作废（2026-09-13，PHASE5-Z1）—— **不是论文底本，不得充当证据源**
>
> 这是早期批次的**手写/合成提取稿**，不是从论文原文抓取的底本。
> **禁止**把其中的任何句子当作 `> 原文:"..."` 引文，也禁止据此判定卡片数字「有出处」。
>
> 已确认的缺陷（台账 `PHASE4-发现的内容缺陷.md` 丙类 C1/C2）：
> - **「论文摘要」是改写/合成文本**，不是逐字原文（TFT 的真实摘要措辞与本文件不同），
>   故**不得**当作 `> 原文` 引文使用。
> - 文件未给出任何 arXiv ID / DOI，**无法定位来源论文**。
>
> **下游保护**：`paper-审核/scripts/gate_check.py` 已把 `extract.md` 与 `_superseded/` 下的
> 文件列为**永不允许**充当证据源（`_is_forbidden_evidence_source`，自检用例 22 锁定）。
> 该判据对全库 15 份 `extract.md` **一律生效**，不只本文件。
>
> 如需把本批次论文纳入证据链，正确做法是**重新抓取真实底本**：
> `python3 paper2skills-research/scripts/fetch_fulltext.py --arxiv <id> --domain <域> --paper-id <p2s-id>`
> 抓完应落在 `papers/<域>/<paper_id>/fulltext.md`，与本文件无关。
>
> 保留本文件仅为**历史留痕**（记录早期批次曾如何产出提取稿），不删。

---

# 论文信息

## Paper 3: Time Series Forecasting with Transformer

### 论文信息
- **标题**: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **领域**: 时间序列 / 需求预测

### 核心算法
1. **LSTM/GRU**: 时序特征提取
2. **Attention Mechanism**: 注意力机制捕捉长期依赖
3. **Transformer**: 时序融合Transformer

### 关键公式
- Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
- Multi-head: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
- Temporal fusion: 结合历史编码和未来协变量

---

# 论文摘要

We introduce the Temporal Fusion Transformer (TFT), a novel attention-based architecture designed for multi-horizon forecasting with heterogeneous inputs. The key innovation is a multi-horizon forecasting model that combines gated residual networks, variable selection networks, and a self-attention mechanism to capture both temporal dependencies and static covariates. TFT achieves state-of-the-art performance on several benchmark datasets while maintaining interpretability through attention weight analysis.
