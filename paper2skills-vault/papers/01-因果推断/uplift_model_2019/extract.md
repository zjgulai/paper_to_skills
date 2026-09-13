> # ⛔ 本文件已作废（2026-09-13，PHASE5-Z1）—— **不是论文底本，不得充当证据源**
>
> 这是早期批次的**手写/合成提取稿**，不是从论文原文抓取的底本。
> **禁止**把其中的任何句子当作 `> 原文:"..."` 引文，也禁止据此判定卡片数字「有出处」。
>
> 已确认的缺陷（台账 `PHASE4-发现的内容缺陷.md` 丙类 C1/C2）：
> - **arXiv ID 错误**：文件写 `1801.05045` 声称是 Athey & Imbens 的 treatment effects 论文，
>   而该 ID 实为 **hep-th 物理论文**；作者与日期均系张冠李戴。
> - **摘要为合成文本**：`## 摘要 (用于测试)` 一节是**写出来的**，不是任何真实论文的摘要
>   （该段末尾自标「用于测试」，但这不足以防止它被当成论文事实引用）。
> - 同目录的 PDF `Meta-Learning_for_Individualized_Treatment_Effects.pdf` 与本题材相关，
>   但其准确的 arXiv ID **未经核实**，不要沿用本文件的 ID。
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

# Uplift Modeling 测试论文

## 论文信息
- **标题**: Meta-learning for Individualized Treatment Effects
- **arXiv ID**: 1801.05045
- **作者**: S. Athey, G. Imbens
- **日期**: 2018

## 摘要 (用于测试)
We propose a meta-learning framework for estimating heterogeneous treatment effects from experimental or observational data. The key insight is to combine multiple machine learning models, each capturing different aspects of the treatment effect heterogeneity. We introduce several meta-learners including T-learner, S-learner, and X-learner, and compare their performance in terms of prediction accuracy and sample efficiency. Our methods are particularly useful when the sample size is limited and the treatment effect varies across subgroups.

## 核心算法
1. **T-Learner (Two-Learner)**: Train separate models for treatment and control groups
2. **S-Learner (Single-Learner)**: Use single model with treatment indicator as feature
3. **X-Learner (Cross-learner)**: Combine predictions from T-learner with propensity score weighting

## 关键公式
- CATE: τ(x) = E[Y(1)|X=x] - E[Y(0)|X=x]
- Propensity score: e(x) = P(T=1|X=x)
- X-learner second stage: τ(x) = τ₁(x) + e(x)·(τ₀(x) - τ₁(x))
