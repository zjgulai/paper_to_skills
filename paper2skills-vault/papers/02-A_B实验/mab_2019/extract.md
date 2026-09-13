> # ⛔ 本文件已作废（2026-09-13，PHASE5-Z1）—— **不是论文底本，不得充当证据源**
>
> 这是早期批次的**手写/合成提取稿**，不是从论文原文抓取的底本。
> **禁止**把其中的任何句子当作 `> 原文:"..."` 引文，也禁止据此判定卡片数字「有出处」。
>
> 已确认的缺陷（台账 `PHASE4-发现的内容缺陷.md` 丙类 C1/C2）：
> - **「论文摘要」是 LLM 风格合成文本**，非任何真实摘要。最明显的破绽是这一句：
>   `Our methods are particularly effective for cross-border e-commerce where ad creative
>   testing is continuous and conversion data arrives sequentially.` ——
>   这是把**本项目自己的业务语境**写进了论文摘要里，真实论文不会这么写。
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

## Paper 4: Multi-Armed Bandit for Online Advertising

### 论文信息
- **标题**: Thompson Sampling for Multi-Armed Bandit Problems
- **领域**: A/B实验 / 在线学习

### 核心算法
1. **Epsilon-Greedy**: 探索概率 ε 的贪心策略
2. **UCB (Upper Confidence Bound)**: 上置信界算法
3. **Thompson Sampling**: 汤普森采样（贝叶斯方法）

### 关键公式
- UCB: a_t = argmax[ μ_i + sqrt(2 ln t / N_i) ]
- Thompson Sampling: θ_i ~ Beta(α_i, β_i)
- Expected regret: O(√(KT ln T))

---

# 论文摘要

We present a comprehensive study of multi-armed bandit algorithms for online advertising optimization. The key insight is that traditional A/B testing wastes resources on exploring inferior options, while bandit algorithms dynamically allocate traffic to better-performing ad variants. We compare epsilon-greedy, UCB, and Thompson Sampling, showing that Thompson Sampling achieves the best performance in most scenarios. Our methods are particularly effective for cross-border e-commerce where ad creative testing is continuous and conversion data arrives sequentially.
