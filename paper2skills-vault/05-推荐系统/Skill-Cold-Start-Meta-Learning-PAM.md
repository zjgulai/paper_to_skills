---
title: Popularity-Aware Meta-Learning for Cold-Start Recommendation
module: 05-推荐系统
topic: cold-start-meta-learning
status: stable
created: 2026-05-15
updated: 2026-09-12
# --- v2 溯源字段：仅覆盖本次 v2.1 增补节（①1b / ①b / ②场景2 / ③-2 / ⑤增补 / ⑥）---
# 原 v1 正文（①PAM 原理、②场景1、③主代码、⑤原 ROI）未溯源到任何论文，见 ② 的「增补说明」
paper_id: 2608.10240
paper: Sequential Modality Dropout for Robust Multi-Modal Sequential Recommendation
venue: CIKM 2026
venue_tier: CCF-B
evidence_grade: A
verified_by: verify_skill_code.py（K1）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py + 人工抽检
supersedes:
related: Skill-Cold-Start-Product-Recommendation.md, Skill-Matrix-Factorization.md, Skill-Semantic-ID-Retrieval-RPG.md
---

# Skill Card: Cold-Start Meta-Learning (PAM)

## ① 算法原理

**核心问题**：母婴品类SKU迭代快（奶粉按月龄分段、辅食按月添加），新品上架无历史交互数据，传统协同过滤无法推荐。冷启动是母婴电商的结构性痛点。

**传统方案缺陷**：
- 基于内容的推荐：只看商品属性，忽略用户偏好
- 热门兜底：新品永远打不过爆款
- 探索策略：随机曝光，转化率极低

**PAM 创新（KDD 2025, 快手）**：
按商品**流行度分层**构建元学习任务：
1. **分层策略**：将商品按历史交互量分为高/中/低流行度三层
2. **元学习**：每层视为独立任务，用MAML学习"如何快速适应新商品"
3. **流行度感知**：区分内容特征（商品属性）和行为特征（用户交互）在不同流行度下的权重
4. **数据增强+自监督**：专门为低流行度商品设计增强策略

**关键洞察**：高流行度商品的行为信号丰富，低流行度商品只能依赖内容信号。PAM让模型学会"根据流行度自动调整信号权重"——高流行度看行为，低流行度看内容。

**1b.（v2.1 增补）内容信号自己也会缺：Sequential Modality Dropout（SMD）**

上面这条洞察（「低流行度商品只能依赖内容信号」）有一个没说出口的前提：**内容信号本身是齐的**。
多模态序列推荐（用 CLIP 图向量 + BERT/Llama 文本向量，与 ID embedding 一起融合）在真实商品目录里
并不成立。论文《Sequential Modality Dropout for Robust Multi-Modal Sequential Recommendation》
（CIKM '26，arXiv 2608.10240）给出的目录实测是：**Toys & Games 有 34.3%、Beauty & Personal Care 有 48.3%
的商品没有文本描述**；在它重抓的四个 MISSRec Amazon 域上，**26% 到 41% 的商品没有图像**。母婴品类
（奶瓶、辅食、童装、孕妇装）与这两个类目高度相邻，缺图、缺文案是常态而不是例外。

论文的动作只有一个：**训练时在每个模态流（图像、文本）的融合点，按样本独立地以概率 p 把整条模态擦除**
——同一个用户序列里的所有 item 共用同一条掩码（因为真实缺失是「整批 / 整类目一起缺」，不是每个 item
独立随机缺）；这是**四行、架构无关**的改动（MM-SASRec / IISAN / MISSRec / fMRLRec 都插得上），
且**不做 $1/(1-p)$ 重标定**（目标是「对真正缺失的模态保持不变性」，不是方差缩减）。测试期不擦除。

为什么这条要写进冷启动卡：论文自己说，模态成块缺失**正是产生冷启动问题的同一种模式**。于是本卡的
信号分层从两段变成三段：

| 商品状态 | 主要可用信号 | 对应方法 |
|---|---|---|
| 高流行度 | 行为信号充足 | 常规协同过滤 / 序列模型 |
| 低流行度新品 | 行为稀疏 → 靠内容信号 | **本卡主体（PAM）** |
| **内容信号本身残缺** | 任一模态都可能整块缺失 | **本段补的 SMD（训练期模态擦除）** |

两者可叠加、不冲突：PAM 学「按流行度调信号权重」，SMD 学「任一模态单独在场也扛得住」。

**评估口径**（务必按论文口径报，否则数字会失真）：`retention R = 移除某模态后的 HR@10 ÷ 全模态 HR@10`；
`robustness gain G = R(θ_SMD) / R(θ_No SMD)`，两个 checkpoint 必须**同超参、同优化器、同种子**，
只差「有没有 SMD」这一项；缺失评测分两套协议——**Protocol 1** 整条模态全量移除（no_text / no_image / no_modal），
**Protocol 2** 每个 item 独立以 p_miss 置零（p_miss 从 0 扫到 0.95，取 5 个随机种子平均）。

---

## ①b 反例与适用边界（v2.1 增补 · 只增不改）

> 本段为 v2.1 新增，原正文一个字未删。段内每个带数字的论断都在 ⑥ 段有逐字出处。

**什么时候不要用这个算法**

1. **单模态系统用不上**。SMD 插在「多模态融合点」；只用行为 ID 的模型（MF、原版 SASRec、纯 ID 序列模型）
   没有模态流可擦，必须先有内容模态接入。
2. **缺模态率很低时收益接近 0**。论文自己给出的反例：fMRLRec 因为融合方式（concat + projection）本来就学到了
   近似冗余的模态表示，文本 retention 本来就有 94%，加 SMD 只提到 99%，robustness gain 是全场最小的一档。**先做一次目录缺失审计，
   再决定要不要改训练代码。**
3. **它不替代本卡主体的选型工作**。SMD 是训练期正则，解决不了「新品根本没有行为数据」——那是 PAM 的问题；
   两者是串联关系，不是二选一。

**已知的失败模式**

1. **全模态精度不是绝对无损，是权衡**。论文跑满 16 个 (backbone, dataset) 组合后汇总：**11 个**改善，
   平均峰值 HR@10 变化 **+0.8%**，但区间是 **−4.4% 到 +6.5%**；两个最大跌幅（IISAN/Scientific −4.4%、
   IISAN/Instruments −4.1%）**恰好出现在 retention 提升最大的地方**（3.2× 与 1.4×）。上线前要在自己的
   骨干上量一遍这个权衡，不能假设「白拿」。
2. **不要把 61% / 22% 当成普遍值**。这一对数字的条件是：**Scientific 域、MM-SASRec 骨干、5 个 seed 的均值**，
   在逐 item 缺失率扫描到 p_miss = 0.95 时取得。换域必须带条件复述。其他域的佐证是：**70% 逐 item 丢弃下
   SMD 保留 71–78% vs 无 SMD 的 46–53%**（Arts 与 Office；论文只画了 Scientific 的曲线，其余域以表格形式
   省略在附录，且 Office 的 IISAN/MM-SASRec 两点趋势一致）。
3. **缺模态率的测量口径差异极大，不要引用任何单一数字当基线**。论文重抓 Amazon Reviews 2023 得到的图像缺失率
   与 MISSRec 原表差距很大（原表 Scientific 26.75%、Instruments 63.12%、Arts 44.90%、Office 63.99%），
   而且论文的 Office 缺失率在下载时没记（标 n/a）。**先在自己的目录上做审计。**
4. **逐用户显著性检验有两格没过**。24 个检验里 22 个在 α = 0.05 显著；两个例外都是**全模态条件下的 HR@10
   比较**（Scientific 与 Instruments）——那里 SMD 把差不多同样多的用户从 miss 翻成 hit、又从 hit 翻成 miss，
   二值指标在总体层面抵消了。这两格的 NDCG@10 检验仍然显著。
5. **交叉重建损失在强骨干上是净亏**。MISSRec/Scientific 上加交叉重建损失只多 3 个点 retention（70% → 73%），
   却花掉 2.3% 的全模态 HR@10，在 MISSRec 自己报告的多数指标上是负收益。论文的结论很明确：
   **只在「融合简单 + 缺失严重」时才开**，默认不开。

**论文自己承认的局限**

1. 两个开放方向：把方法扩展到图像/文本之外的**音频、视频、结构化属性**；以及处理**模态损坏（corruption）
   而不是缺失（absence）**——本方法的前提是「模态整个不在」，不是「模态在但内容是坏的」。
2. **汇总的 16 个组合不是完整矩阵**：Arts 上 IISAN / MISSRec / fMRLRec 未跑，Office 上 MISSRec / fMRLRec
   因算力预算未跑，论文以「省版面」为由只给汇总不给完整表。
3. **论文未报告**：任何母婴 / 中文跨境目录的数据与效果、金额口径的 ROI、线上 A/B 结果（全部结论都是离线
   benchmark 的 HR@10 / NDCG@10 / retention）、以及加这四行之后的训练成本增量。本卡 ⑤ 段因此只给公式。
4. 唯一可复现性利好：代码与「支撑论文每个数字的 JSON 输出」都已开源（⑥ 段给了链接），
   这是本卡相比许多存量卡更容易做内部复现的地方。

---

## ② 母婴出海应用案例

### 场景：新品快速冷启动

**业务问题**：Momcozy每季度上架30+新品（新款吸奶器、新配件）。上架首周曝光转化率<0.5%，远低于成熟品的2.5%。

**PAM 应用**：
1. **分层**：
   - 高流行度：月交互>1000的SKU
   - 中流行度：月交互100-1000
   - 低流行度：月交互<100（主要是新品）
2. **元训练**：在现有SKU上训练，学习"从商品属性预测用户偏好"的初始化参数
3. **快速适应**：新品上架后，仅需少量交互（10-50次）即可微调出专属推荐模型

**预期产出**：
- 新品首周转化率：0.5% → 1.5%
- 新品达到成熟品转化率的时间：3个月 → 2周
- 长尾SKU总GMV占比：15% → 25%

**业务价值**：
- 加速新品验证：快速识别潜力爆款
- 降低库存风险：不好卖的新品及时止损
- 品类扩展：敢于尝试更多细分品类

**⚠️ 增补说明（v2.1 · 只标注、不删除原数据）**：上面「新品首周转化率」「达到成熟品转化率的时间」
「长尾 SKU 的 GMV 占比」与 ⑤ 段的 ROI 百分比，都是**本卡 v1 的自述经验值，没有论文出处**——本次增补的论文
（arXiv 2608.10240）**未报告**任何 GMV / 金额口径的数字，它的全部结论都是离线 benchmark 的 HR@10 / NDCG@10
与 retention。按 MasterPrompt-v2 的 R1/R2 规则，这些数字**不能当证据引用**；本次新增的 ⑥ 段只托住
2608.10240 的数字，**不为这几条背书**，请以贵司自己的埋点复测为准。

---

### 场景 2（v2.1 增补）：缺图 / 缺文案的 SKU 不该让推荐位塌掉

- **业务问题**：母婴跨境的商品图多来自供应商、卖点文案靠运营补，**新品上架时缺图或缺文案是常态**。
  一旦多模态内容特征接进了召回/排序，某个模态成批缺失（图床故障、某批 SKU 批量没写描述、供应商只给图不给字）
  就会让推荐位质量塌方——**而且塌得很隐蔽：全模态口径的离线指标照样好看，事故只在上线后暴露**。
  目标动作：**下一次重训时**在融合点加四行掩码，把「缺模态」从线上事故变成训练期就见过的情况；
  改完用 ③ 段的 R / G 表与逐用户配对检验来验收，而不是只看全模态指标。
- **数据要求**：序列推荐日志（user_id、item_id、timestamp；**按时间切分**，论文最长序列长度取 10）
  + 商品侧两路**冻结**内容向量（图像、文本）；再加一份**目录级缺失审计**（每个 SKU 有无图、有无文案、
  缺失是否按类目成块——这决定 p 该设多少，论文默认 p = 0.3）。评测要同时跑两套协议：
  整条模态移除（Protocol 1）与逐 item 随机缺失（Protocol 2）。
- **数据可得性**：`部分可得（需补充 X）`。序列日志与商品属性通常可得；**图像/文本 embedding 需要自建或调 API**
  （论文直接共用冻结的 MISSRec CLIP ViT-B/32，512 维 + 语言编码器特征）。**跨语种要额外补**：论文的实验
  全部是英文 Amazon 评论数据，**中文母婴标题上的编码器行为论文未报告**，换编码器就等于换了一个未验证的前提。
- **预期产出**：（a）一张按域按模态的 **R / G 表**（加 SMD 前 vs 后，两种缺失协议都给）；
  （b）逐用户配对显著性检验的 p 值（③ 段给了 McNemar 精确检验的实现，论文用它对 121k 用户做检验）；
  （c）**缺口评估**：你自己的缺模态率落在 retention 曲线的哪一段上，据此决定是否值得动训练代码。
- **业务价值**：形态是**用一次训练期正则，换掉一次「模态缺失导致推荐位塌方」的线上事故**。
  收益侧的论文口径是 retention 提升倍数（离线 benchmark），**不能直接折算成 GMV**，本卡不给金额结论；
  可量化的是「事故面的下降」与「重训时的人工成本节省」，二者都需企业自测。

---

## ③ 代码模板

```python
"""
Popularity-Aware Meta-Learning (PAM) for Cold-Start Recommendation
用于新品/新用户的快速冷启动推荐
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


class PAMModel(nn.Module):
    """流行度感知的元学习推荐模型"""

    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.content_proj = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, user_ids, item_ids, item_content=None):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)

        if item_content is not None:
            # 低流行度：融合内容特征
            i = self.content_proj(torch.cat([i, item_content], dim=-1))

        score = (u * i).sum(dim=-1)
        return torch.sigmoid(score)


def popularity_aware_meta_train(model, interactions, item_features,
                                popularity_thresholds=(1000, 100),
                                inner_lr=0.01, meta_lr=0.001, epochs=100):
    """
    PAM元训练

    Args:
        interactions: [(user, item, rating)] 列表
        item_features: 商品内容特征
        popularity_thresholds: (高, 低) 流行度阈值
    """
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # 按流行度分层
    item_counts = defaultdict(int)
    for u, i, r in interactions:
        item_counts[i] += 1

    high_pop = [i for i, c in item_counts.items() if c >= popularity_thresholds[0]]
    mid_pop = [i for i, c in item_counts.items()
               if popularity_thresholds[1] <= c < popularity_thresholds[0]]
    low_pop = [i for i, c in item_counts.items() if c < popularity_thresholds[1]]

    layers = {
        'high': high_pop,
        'mid': mid_pop,
        'low': low_pop
    }

    for epoch in range(epochs):
        meta_loss = 0

        for layer_name, items in layers.items():
            if len(items) < 10:
                continue

            # 采样该层的交互
            layer_interactions = [x for x in interactions if x[1] in items]
            if len(layer_interactions) < 5:
                continue

            # 内循环：快速适应
            fast_weights = {name: param.clone()
                           for name, param in model.named_parameters()}

            # 计算该层损失
            users = torch.LongTensor([x[0] for x in layer_interactions])
            items_t = torch.LongTensor([x[1] for x in layer_interactions])
            ratings = torch.FloatTensor([x[2] for x in layer_interactions])

            # 使用内容特征（低流行度层权重更高）
            content_weight = 0.3 if layer_name == 'high' else 0.7
            item_content = torch.randn(len(items_t), 64) * content_weight

            preds = model(users, items_t, item_content)
            loss = nn.BCELoss()(preds, ratings)

            meta_loss += loss

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Meta Loss: {meta_loss.item():.4f}")

    return model


def cold_start_adapt(model, new_item_id, new_item_features,
                     few_interactions, inner_steps=5, inner_lr=0.01):
    """
    新品快速适应：用少量交互微调模型
    """
    adapted_model = PAMModel(
        model.user_emb.num_embeddings,
        model.item_emb.num_embeddings,
        64
    )
    adapted_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)

    for step in range(inner_steps):
        users = torch.LongTensor([x[0] for x in few_interactions])
        items = torch.LongTensor([x[1] for x in few_interactions])
        ratings = torch.FloatTensor([x[2] for x in few_interactions])

        content = torch.FloatTensor([new_item_features] * len(few_interactions))

        preds = adapted_model(users, items, content)
        loss = nn.BCELoss()(preds, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapted_model


# 示例
if __name__ == '__main__':
    model = PAMModel(n_users=1000, n_items=500)
    # 元训练...
    # 新品适应...
    print("PAM模型初始化完成")
```

**③-2（v2.1 增补）SMD 的可运行实现**（只用 numpy + 标准库；6 条断言由 K1 的 L5 实跑）

- 对应论文：`smd_mask` = §2.2 式 (2) 的按样本伯努利掩码；`apply_smd` = 融合点处的 f~_m = m_b^{(m)} \cdot f_m
  （**按样本、不按 item**，且不做 $1/(1-p)$ 重标定）；`per_item_drop_mask` = §3.1 的 Protocol 2；
  `retention` / `robustness_gain` = 式 (4) 的 R 与 G；`mcnemar_exact_p` = Appendix D 的逐用户配对检验。
- 与框架的接口：本实现只做「掩码 + 度量」，`apply_smd` 接收形状 `(n_samples, seq_len, dim)` 的模态流张量，
  换成 torch/TF 张量即可直接接进 PAM / 任何融合层（掩码是 0/1，乘法语义与框架无关）。
- ⚠️ 代码内的数据（`_make_demo_streams`）与 ⑤ 段里的示意数字都是**合成的**，**不是论文数据**；
  论文的数字（61%/22%、1.0–3.2× 等）全部在 ⑥ 段的引文里，两者不可互相印证。

```python
# ---------------------------------------------------------------------------
# 4. Sequential Modality Dropout（T3-19 / arXiv 2608.10240）
#    训练期在**融合点**对每条模态流做按样本的伯努利擦除；测试期不擦除。
#    依赖：只用 numpy + 标准库（不引入任何框架，可直接套到 torch/TF 的张量上）。
# ---------------------------------------------------------------------------
import math

import numpy as np


def smd_mask(n_samples: int, n_modalities: int = 2, p: float = 0.3,
             rng=None) -> np.ndarray:
    """式 (2)：每个样本、每条模态独立抽 Bernoulli(1-p)；1=保留，0=整条历史擦除。

    返回形状 (n_samples, n_modalities) 的 0/1 掩码 —— **按样本、不按 item**：
    同一个用户序列里的所有 item 共用同一条模态掩码（论文 §2.2 "Why Per-Sample"）。
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p 必须在 [0,1] 内")
    rng = np.random.default_rng(0) if rng is None else rng
    return (rng.random((n_samples, n_modalities)) >= p).astype(np.float64)


def apply_smd(streams: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    """把掩码乘到融合点之前的每条模态流上：f̃_m = m_b^(m) · f_m(x_i)。

    streams[m] 形状 (n_samples, seq_len, dim)；mask 形状 (n_samples, n_modalities)。
    ⚠️ 论文强调：**不做 1/(1-p) 重标定**（与标准 dropout 不同），
    因为目标是「对真正缺失的模态保持不变性」，而不是方差缩减。
    """
    if len(streams) != mask.shape[1]:
        raise ValueError("模态数与掩码列数不一致")
    out = []
    for m, s in enumerate(streams):
        if s.shape[0] != mask.shape[0]:
            raise ValueError("样本数与掩码行数不一致")
        out.append(s * mask[:, m][:, None, None])     # 整条序列共用同一个掩码
    return out


def per_item_drop_mask(n_items: int, n_modalities: int, p_miss: float,
                       rng=None) -> np.ndarray:
    """Protocol 2：**每个 item 每条模态独立**以 p_miss 置零（真实目录的 item 级缺失）。

    与 smd_mask 的区别是评测口径：训练用按样本的整条擦除，评测用按 item 的随机缺失。
    """
    rng = np.random.default_rng(0) if rng is None else rng
    return (rng.random((n_items, n_modalities)) >= p_miss).astype(np.float64)


def per_sample_drop_mask(n_samples: int, n_modalities: int, p_miss: float,
                         rng=None) -> np.ndarray:
    """Protocol 1 的随机版本：整条模态**全量**移除（no_text / no_image / no_modal）。"""
    return (np.random.default_rng(0) if rng is None else rng).random(
        (n_samples, n_modalities)) >= p_miss


def retention(hr_missing: float, hr_full: float) -> float:
    """式 (4)：R(θ) = HR@10(θ; missing) / HR@10(θ; full)。

    hr_full<=0 时返回 nan（论文口径下 retention 无定义，不要瞎填 0）。
    """
    if hr_full <= 0:
        return float("nan")
    return float(hr_missing) / float(hr_full)


def robustness_gain(r_smd: float, r_no_smd: float) -> float:
    """式 (4)：G = R(θ_SMD) / R(θ_No SMD)（两个 checkpoint 必须同超参、同种子）。"""
    if r_no_smd <= 0:
        return float("inf")
    return float(r_smd) / float(r_no_smd)


def mcnemar_exact_p(b: int, c: int) -> float:
    """McNemar 精确检验（论文 Appendix D 用来做逐用户配对显著性检验）。

    b = 无 SMD 错、SMD 对的用户数；c = 反向翻转的用户数。双侧精确 p 值：
    p = 2 * Σ_{k=0}^{min(b,c)} C(b+c, k) * 0.5^(b+c)，上限截到 1.0。
    只依赖 math.comb，无需 scipy。
    """
    n = b + c
    if n == 0:
        return 1.0
    k_max = min(b, c)
    tail = sum(math.comb(n, k) for k in range(k_max + 1)) / float(2 ** n)
    return float(min(1.0, 2.0 * tail))


def _make_demo_streams(n_samples: int = 200, seq_len: int = 10, dim: int = 16,
                       seed: int = 0):
    """合成示例数据：一条「图像流」+ 一条「文本流」。⚠️ 非论文数据。"""
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(n_samples, seq_len, dim))
    txt = rng.normal(size=(n_samples, seq_len, dim))
    return img, txt


if __name__ == "__main__":
    img, txt = _make_demo_streams()
    p = 0.3
    mask = smd_mask(img.shape[0], 2, p, rng=np.random.default_rng(1))
    masked = apply_smd([img, txt], mask)

    print("== Sequential Modality Dropout（合成示例数据，非论文数据）==")
    print(f"训练掩码 p={p}: 经验擦除率 = {1 - mask.mean():.4f}"
          f" | 两条模态都被擦除的样本占比 = {np.mean(mask.sum(axis=1) == 0):.4f}")

    # 按样本 vs 按 item：同一个样本内所有 item 共用掩码
    shared = np.all(masked[0][0] == masked[0][0][0] * np.ones_like(masked[0][0]))
    print(f"序列内掩码一致（按样本擦除）: {shared}")

    item_mask = per_item_drop_mask(n_items=1000, n_modalities=2, p_miss=0.95,
                                  rng=np.random.default_rng(2))
    print(f"评测口径 Protocol 2（p_miss=0.95）: item×模态格子里被置零的比例 = "
          f"{1 - item_mask.mean():.4f}")

    # 式 (4) 的算术：这里代的是**示意值**，不是论文数值，请换成你自己的 HR@10
    hr_full, hr_no_txt_no_smd, hr_no_txt_smd = 5.00, 2.00, 4.00
    r_no = retention(hr_no_txt_no_smd, hr_full)
    r_yes = retention(hr_no_txt_smd, hr_full)
    print(f"示意：R(No SMD)={r_no:.2f}  R(SMD)={r_yes:.2f}  "
          f"G={robustness_gain(r_yes, r_no):.2f}×")

    # 逐用户配对检验（示意计数）
    print(f"McNemar 精确检验示意 b=180, c=60: p = {mcnemar_exact_p(180, 60):.3e}")


# ---------------------------------------------------------------------------
# 5. 测试（可执行断言）
# ---------------------------------------------------------------------------
def test_mask_rate_matches_p_and_p_zero_keeps_everything():
    m = smd_mask(200_000, 2, 0.3, rng=np.random.default_rng(7))
    assert m.shape == (200_000, 2)
    assert set(np.unique(m)).issubset({0.0, 1.0})
    assert abs(m.mean() - 0.7) < 5e-3                 # 保留率 ≈ 1-p
    assert np.allclose(smd_mask(100, 2, 0.0), 1.0)    # p=0 → 不擦除（测试期口径）


def test_mask_is_per_sample_not_per_item():
    img, txt = _make_demo_streams(n_samples=64, seq_len=10, dim=8)
    mask = smd_mask(64, 2, 0.5, rng=np.random.default_rng(3))
    mi, mt = apply_smd([img, txt], mask)
    # 整条用户历史被同一条掩码控制：被擦除的样本，序列内每个位置都是 0
    zero_img = np.all(mi == 0.0, axis=(1, 2))
    assert np.array_equal(zero_img, mask[:, 0] == 0)
    # 未被擦除的样本，数值保持原样（不做任何重标定）
    keep = mask[:, 0] == 1
    assert np.allclose(mi[keep], img[keep])
    # 对比：Protocol 2 是按 item 独立的，同一样本内保留/擦除会混合
    item_mask = per_item_drop_mask(1000, 2, 0.5, rng=np.random.default_rng(4))
    mixed = item_mask[:, 0]
    assert 0.45 < mixed.mean() < 0.55 and len(np.unique(mixed)) == 2


def test_masked_modality_is_zeroed_without_rescaling():
    """论文 §2.2：擦除即为 0，**不乘 1/(1-p)** —— 与标准 dropout 的约定不同。"""
    x = np.ones((4, 3, 2))
    mask = np.array([[1.0, 0.0]] * 4)
    out_i, out_t = apply_smd([x, x], mask)
    assert np.allclose(out_t, 0.0)                    # 文本模态整条归零
    assert np.allclose(out_i, 1.0)                    # 未擦除模态幅度不变（非 1/0.7）


def test_protocol_2_missing_rate_is_approximately_p_miss():
    m = per_item_drop_mask(50_000, 2, 0.95, rng=np.random.default_rng(5))
    assert abs((1 - m.mean()) - 0.95) < 5e-3


def test_retention_and_gain_follow_equation_4():
    assert abs(retention(4.0, 5.0) - 0.8) < 1e-12
    assert abs(retention(1.1, 6.18) - 1.1 / 6.18) < 1e-12
    assert abs(robustness_gain(0.8, 0.4) - 2.0) < 1e-12
    assert math.isnan(retention(1.0, 0.0))            # 全模态 HR=0 → 无定义
    assert robustness_gain(0.4, 0.0) == float("inf")


def test_mcnemar_exact_matches_hand_computed_values():
    assert abs(mcnemar_exact_p(9, 0) - 2.0 / 512.0) < 1e-12   # 2*C(9,0)/2^9
    assert mcnemar_exact_p(20, 20) > 0.99                      # 无差异 → p≈1
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(180, 60) < 1e-12                    # 大量单向翻转 → 显著
```

---

## ④ 技能关联

- **前置**：Matrix Factorization（理解协同过滤基础）
- **延伸**：Transfer Learning Cold-Start（跨品类迁移）
- **可组合**：+ Diversity Reranking（给新品曝光机会）

- **（v2.1 增补）同域强相关｜`Skill-Cold-Start-Product-Recommendation.md`（06-增长模型）**：那张卡解决
  「新品没有交互怎么被推出来」，本卡解决「新品连图/文案都不全怎么被推出来」。数据流：先按那张卡的分层
  确定低流行度集合，再对本卡 ② 场景 2 的目录审计结果决定这批 SKU 是否需要 SMD 训练期护栏。
- **（v2.1 增补）前置｜`Skill-Matrix-Factorization.md`（同目录）**：SMD 插在**融合点**，前提是骨干已经有
  内容模态接入（本卡 ③ 的 `content_proj` 就是这类融合层）；纯 ID 的矩阵分解没有模态流可擦，需先接入内容特征。
- **（v2.1 增补）延伸｜`Skill-Semantic-ID-Retrieval-RPG.md`（同目录）**：RPG 的语义 ID 由商品属性/文本/图片
  生成，**缺模态会直接污染 ID 生成**；两者作用点不同（RPG 改表示、SMD 改训练过程），可以叠加：
  先用 SMD 训出对缺模态鲁棒的编码/融合，再喂给语义 ID 生成。

---

## ⑤ 商业价值评估

- **ROI**：新品GMV提升50-100%，试错周期缩短60%
- **难度**：⭐⭐⭐☆☆（3/5）— 元学习概念门槛，但实现可模块化
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 母婴品类迭代快，冷启动是刚需痛点

**（v2.1 增补）SMD 的 ROI 口径**

ROI = (ΔR × E_missing × V_slot − C_impl) / C_impl，其中 ΔR = R(SMD) − R(No SMD)（**必须企业自测**）。

| 参数 | 含义 | 来源 |
|---|---|---|
| `R(·)` | 某个模态被移除后，HR@10 相对全模态的保留比例 | 用本卡 ③ 的 `retention` 口径在自家数据上测 |
| `E_missing` | 该模态在**曝光流量上**的缺失率（按曝光加权，不是按 SKU 计数） | 企业自建目录审计 + 埋点；论文只给了按商品的缺失率（26–41% 图像、Toys & Games 34.3% / Beauty 48.3% 文本），**不是按曝光加权**，不能直接代入 |
| `V_slot` | 该推荐位每单位 HR@10 对应的业务价值 | 企业自有口径（GMV/转化率标定） |
| `C_impl` | 一次性建设成本：内容 embedding 生成与存储 + 重训一轮 + 两套缺失协议的评测脚手架 | 论文**未报告**任何成本量级，需企业自估；改动本身是四行，但 **embedding 基建与评测协议才是主要成本** |

**收益侧不给金额结论**：论文全部是离线 benchmark 口径（HR@10 / NDCG@10 / retention），
**未报告**线上 A/B、GMV 或任何金额收益；本卡不替论文假设一个。

---

## ⑥ 原文引用（v2.1 增补 · 逐字核验）

> 出处底本：`paper2skills-vault/papers/05-推荐系统/p2s-2026-0023/fulltext.md`（arXiv 2608.10240v1 的 HTML 全文）。
> 下列摘录全部由脚本从底本**逐字抽取**（命令行输出即复制粘贴），并经 `quote_check.py` 判 **VERBATIM**。
> 本卡正文里出现的论文数字，都能在下面对应到原文句。**61% / 22% 这一对带条件的引文见 C 组。**

**A. venue 与「缺模态有多普遍」**

> 原文："Accepted at the 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 7–11, 2026, Rome, Italy. This is the authors’ preprint version."
> 出处：2608.10240 §标题注（venue）
> 原文："These methods are trained and evaluated on benchmarks in which every item carries every modality, but real product catalogs routinely violate this assumption (Fu et al., 2026): 34.3% of Toys & Games and 48.3% of Beauty & Personal Care items have no text description, and 26% to 41% of items have no image on the four MISSRec Amazon domains."
> 出处：2608.10240 §1 Introduction（真实目录的缺模态率）
> 原文："We used four Amazon domains following the MISSRec benchmark (Wang et al., 2023): Scientific, Instruments, Arts, and Office, spanning 4,385 to 25,986 items, with 26% to 41% image-missing rates measured on our re-downloaded catalog."
> 出处：2608.10240 §3.1 Setup（Datasets）
> 原文："Table 2 reports the per-domain catalog size, total number of interactions, and image-missing rate of our re-downloaded MISSRec catalog used in the main experiments; these numbers differ from the coverage figures in MISSRec’s original Table 1 (Scientific 26.75%, Pantry 93.65%, Instruments 63.12%, Arts 44.90%, Office 63.99%) because we re-downloaded images directly from the Amazon Reviews 2023 release rather than reusing the MISSRec-provided archives."
> 出处：2608.10240 §A Dataset Statistics（缺失率口径差异）

**B. 方法：按样本掩码、四行改动、不做重标定**

> 原文："We propose Sequential Modality Dropout (SMD): during training, each modality stream (image and text) is independently erased with probability $p$ for an entire user interaction history, so the model learns to predict the next item without relying on any single modality."
> 出处：2608.10240 §Abstract
> 原文："For each training sample, each modality stream (image and text) is independently zeroed with probability $p$, and the same mask applies to every item in the user’s chronological sequence."
> 出处：2608.10240 §1 Introduction（机制）
> 原文："The mask is per-sample, not per-item: all items in user $b$’s sequence share the same modality mask."
> 出处：2608.10240 §2.2 Modality Masking
> 原文："At test time the mask is not applied, except for the deterministic masks used in our robustness evaluation (Section 3.1)."
> 出处：2608.10240 §2.2 Modality Masking
> 原文："Its goal is invariance to a genuinely missing modality rather than variance reduction, so the full-modality input seen at test is simply the $p\!=\!0$ case the model already encountered during training, and no compensating scale is required."
> 出处：2608.10240 §2.2 Modality Masking（Relation to Standard Dropout）
> 原文："We use the per-sample mask instead because real missingness is whole-modality and structured by category or source rather than independent across items: a modality tends to go absent in a block, the same pattern that produces the cold-start problem (Wang et al., 2018; Ganhör et al., 2024), and in our data the text-missing rate varies sharply across categories (34.3% to 48.3%)."
> 出处：2608.10240 §2.2 Modality Masking（Why Per-Sample）
> 原文："The entire mechanism is the four-line modification at the fusion point of the host model shown in Figure 1."
> 出处：2608.10240 §2.2 Modality Masking（Implementation）
> 原文："The SMD module is the same $4$-line block in every host model; only the tensor names differ:"
> 出处：2608.10240 §F Implementation

**C. 评测口径与主结果（R / G、极端缺失、显著性）**

> 原文："We measure robustness by retention, the fraction of a model’s full-modality accuracy (HR@10) that survives when a modality is removed at test time."
> 出处：2608.10240 §Abstract
> 原文："For each backbone, we trained two checkpoints (one without SMD and one with SMD at $p\!=\!0.3$) using identical hyperparameters, optimizers, and seeds, sharing frozen MISSRec CLIP ViT-B/32 features (512-dim) across backbones; SMD is the only difference between the two."
> 出处：2608.10240 §3.1 Setup（Backbones，同超参配对）
> 原文："Across four backbones (MM-SASRec, IISAN, MISSRec, and fMRLRec) on four Amazon domains, SMD raises text retention by 1.0 to 3.2$\times$ at essentially no cost to full-modality accuracy; under an extreme 95% per-item missing rate, it retains 61% of HR@10 versus 22% without (a 2.8$\times$ improvement)."
> 出处：2608.10240 §Abstract
> 原文："On Amazon Scientific across the four backbones, SMD lifts HR@10 text retention from 18–94% to 56–99% (Table 1(a)); on MM-SASRec across four Amazon domains, it lifts text retention from 40–73% to 79–97% (Table 1(b))."
> 出处：2608.10240 §1 Introduction（RQ1 数字）
> 原文："the largest lifts are on IISAN (18% to 56%, 3.2$\times$) and MM-SASRec (40% to 83%, 2.1$\times$), while fMRLRec, already 94% retained because its concat-plus-projection fusion learns near-redundant modality embeddings, rises only to 99%."
> 出处：2608.10240 §3.2 RQ1（逐个骨干）
> 原文："Table 1(b) reports MM-SASRec when text and when images are removed across four domains: SMD lifts text retention from 40–73% to 79–97% and image retention from 67–89% to 87–96%, while matching or exceeding peak HR@10 on every dataset."
> 出处：2608.10240 §3.2 RQ1（逐域）
> 原文："Beyond these two slices, we ran all 16 (backbone, dataset) combinations and summarize them here, omitting the full table only to respect the page limit: 11 of the 16 improve, with robustness gain 1.0 to 3.2$\times$ and a mean peak-HR@10 change of +0.8% (range -4.4% to +6.5%). The two largest accuracy losses, IISAN/Scientific (-4.4%) and IISAN/Instruments (-4.1%), occur exactly where SMD delivers its largest retention gains (3.2$\times$ and 1.4$\times$)."
> 出处：2608.10240 §3.2 RQ1（16 组合汇总）
> 原文："SMD helps on every recommender we tested, regardless of how it fuses image and text; the biggest robustness gains coincide with the biggest full-modality accuracy drops, yet even those drops stay small (at most 4.4%)."
> 出处：2608.10240 §3.2 RQ1（Takeaway）
> 原文："Figure 2. Per-item missing-rate sweep on Scientific (MM-SASRec, mean of 5 seeds). At $p_{\mathrm{miss}}\!=\!0.95$, SMD retains 61% of HR@10 versus 22% without."
> 出处：2608.10240 §3.3 RQ2（Figure 2 图注，含域/骨干/seed 条件）
> 原文："The same pattern holds on Arts and Office: at 70% per-item drop SMD retains 71–78% versus 46–53% without SMD; we plot only the Scientific sweep (Figure 2) and omit the per-domain tables to respect the page limit."
> 出处：2608.10240 §3.3 RQ2（其他域的佐证）
> 原文："On a 0 to 95% per-item missing-rate sweep, MM-SASRec with SMD retains 61% of HR@10 at $p_{\mathrm{miss}}\!=\!0.95$ versus 22% for the unmodified model (Figure 2), and per-user paired tests on 121k users confirm the gains are statistically significant (unlikely to arise by chance)."
> 出处：2608.10240 §1 Introduction（RQ2 数字）
> 原文："To rule out user-level noise, we pair each user’s HR@10 and NDCG@10 between the No-SMD and SMD checkpoints across three datasets (121k users)."
> 出处：2608.10240 §3.3 RQ2（逐用户配对检验）
> 原文："Of the resulting 24 tests (12 (condition, dataset) pairs $\times$ 2 tests), 22 are significant at $\alpha=0.05$, meaning a gap this large is very unlikely if the two models were equivalent, with $p$-values from $4.3\times 10^{-3}$ to $1.0\times 10^{-233}$; the two exceptions are the full-modality McNemar tests on Scientific and Instruments, where SMD flips similar numbers of users each way."
> 出处：2608.10240 §3.3 RQ2（24 个检验，22 个显著）

**D. 可选的正交损失（交叉模态重建）**

> 原文："where we set $\lambda=0.01$ from preliminary runs; a larger $\lambda=0.1$ reduced full-modality accuracy."
> 出处：2608.10240 §2.3 Cross-Modal Reconstruction
> 原文："For simple additive backbones under severe text missingness, an auxiliary loss that trains the two modality projections to predict each other lifts text retention from 90% to 98% on Beauty & Personal Care (48% text-missing; Table 1(c))."
> 出处：2608.10240 §1 Introduction（RQ3 数字）
> 原文："On a strong dynamic-fusion backbone (MISSRec on Scientific) it adds only 3 points of text retention (70% to 73%) while costing 2.3% of full-modality HR@10, a net loss across most metrics MISSRec’s own results (Wang et al., 2023) report."
> 出处：2608.10240 §3.4 RQ3（强骨干上是净亏）
> 原文："The reconstruction loss helps only when fusion is simple and missingness is severe; SMD alone is the default, with the loss an opt-in enhancement."
> 出处：2608.10240 §3.4 RQ3（Takeaway）

**E. 论文自承局限与可复现性**

> 原文："Two directions remain open: extending beyond image and text to audio, video, or structured attributes, and handling modality corruption rather than absence."
> 出处：2608.10240 §4 Conclusion（开放方向）
> 原文："Arts runs are omitted for IISAN, MISSRec, and fMRLRec for the same compute-budget reason; the MM-SASRec/Arts cell is the $11$th cell in our main claim and appears in Table 1(b). Office runs are omitted for MISSRec and fMRLRec because their training cost on the larger Office catalog ($25{,}986$ items, $310$k interactions) is prohibitive at our compute budget; the trend holds on Office for IISAN ($34\%\!\to\!77\%$) and MM-SASRec ($50\%\!\to\!82\%$)."
> 出处：2608.10240 §B Full Plug-In Robustness Results（未跑的组合）
> 原文："Source code for SMD and the baseline plug-in modifications, training scripts, and the JSON outputs backing every number in this paper are released at https://github.com/guanqun-yang/SMD."
> 出处：2608.10240 §F Implementation（开源）

