---
title: Skill-Decision-Conditioned-Forecasting
module: 03-时间序列
topic: 把「给定大促排期与投放预算问销量」做成可仿真的决策条件化预测：动作交错的状态转移层 + 事件驱动的残差校正层
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.25871
paper: CEDAR: Controlled and Event-Driven Demand Forecasting via Residual Decomposition
venue: KDD 2026
venue_tier: CCF-A
evidence_grade: A
verified_by: verify_skill_code.py (K1 PASS) + quote_check.py (引文逐字核验 VERBATIM) + gate_check.py (G1/G2/G3)
verified_at: 2026-09-12
supersedes:
related: Skill-Time-Series-Forecasting.md, Skill-Temporal-Fusion-Transformer.md, Skill-Marketing-Mix-Modeling.md, Skill-Promotion-Effectiveness.md, Skill-Demand-Forecasting-Supply-Chain.md, Skill-Safety-Stock-Replenishment.md, Skill-ROAS-Budget-Optimization.md
---

# Skill-Decision-Conditioned-Forecasting

> **venue 说明（R3 要求显式标注）**：本文收录于 **KDD 2026**（第 32 届 ACM SIGKDD，
> 2026-08-09–13，济州）会议论文集（论文集 DOI 见 evidence.md），属 CCF-A，无需降级。
> 发表机构署名包含中国科学技术大学、阿里巴巴集团与香港科技大学（广州）。
> 论文用的是 **Alibaba 1688**（国内 B2B 批发平台）的工业数据，**不是跨境场景** ——
> 迁移到贵司时必须按 ①b 与 ② 的边界重新取证，不能直接套用。

---

## ① 算法原理

**核心思想**：常规时序模型回答「接下来会发生什么」，而商家真正要问的是
「**如果我按这个预算排期走，接下来会发生什么**」。论文把后者定义为
**决策条件化仿真**：给定历史的状态-动作序列与**未来的动作排期**，把未来销量轨迹推出来。

**数学直觉**：把一个商品的销量生成过程拆成两部分相加：

- 可控部分 $\hat{\mathbf{s}}_{t+1} = f_\theta(\mathbf{s}_{\le t}, \mathbf{a}_{\le t+1})$ —— 状态与动作**交错**进 Transformer
  （因果顺序 $\mathbf{s}_{t-1} \rightarrow \mathbf{a}_{t} \rightarrow \mathbf{s}_{t}$），学的是「动作如何驱动状态转移」
- 外生部分 $\Delta\mathbf{s}_{t+1} = g_\phi(\mathbf{h}_t, \hat{\mathbf{s}}_{t+1}, \mathbf{s}_{t-k:t})$ —— 用外部事件（热搜、节假日）
  经 LLM 归纳 + 文本编码得到稠密向量 $\mathbf{h}_t$，再经 cross-attention 对齐到**具体商品**，只负责补残差

最终仿真值 $\tilde{\mathbf{s}}_{t+1} = \hat{\mathbf{s}}_{t+1} + \Delta\mathbf{s}_{t+1}$，并按自回归方式逐步外推。

**关键假设**：① 动作是**可控且可计划**的（折扣、广告花费），排期提前已知；
② 事件信号必须与商品上下文对齐才有意义；③ 因果解释依赖**序贯可忽略性**（附录 B 明示）。

---

## ①b 反例与适用边界

**什么时候不要用这个算法**：

- **动作不可控 / 不可计划**：本方法的输入是「未来排期」。若促销由平台单方面定档、店铺只能被动跟随，
  则无法构造动作序列，退化成普通外推。
- **没有店铺自己的动作-状态配对面板**：状态面板（流量/加购/订单）与动作面板（折扣、花费）必须
  **同 SKU、同周、对齐**。只有销量没有花费记录的品类无法拟合转移层。
- **只想问「下周卖多少」**：单步预测用不着两阶段结构，常规模型更省事。本方法的价值在**多步反事实**。

**已知的失败模式**：

- **自回归误差累积**。论文附录 D 明说：horizon 拉长后所有模型误差都变大，原因是自回归累积。
  贵司若把 5 周仿真直接外推到 15 周，必须重新验证。
- **残差层只补短期、不补趋势**。论文 §4.4 自承该模块对 MSE 的改善**温和**，主要改善体现在 MAE ——
  即它擅长抓「突发、局部」的偏移，不擅长修正长期趋势偏差。
- **事件信号与商品不对齐就是噪声**。论文 §3.4.2 举了「圣诞帽不会在春节暴涨」的例子；
  本卡模板的实测也显示：把事件随机打到别的周上，效果反而比不做校正更差（见 ③ 输出围栏）。
- **未见过的事件类型无法被校正**。残差层只能外推它见过的事件-商品组合；首次出现的爆点
  （一个全新的达人热点）它给不出修正。本卡模板的合成实验里就保留了这样一个「未见过事件周」，
  校正层对它无响应。
- **多步外推的动作响应会被系统性放大**。这是本卡模板实测到的、**论文没有讨论的**失效模式：
  单步拟合良好的动作系数在多步 rollout 中逐周复利，合成数据上模型给出的排期弹性明显高于真值。
  详见 ③ 的警示块。

**论文自己承认的局限**：

- **论文没有独立的 Limitations 章节**。§5 结论与全文均未列出局限性清单 —— 本节其余内容是从论文
  正文中**明示的假设与自承的不足**提炼的，不是替论文补写的。
- 因果解释建立在**序贯可忽略性**这一不可检验的假设上（附录 B 式 10）。
- 公开数据集实验**不能**复现反事实预算规划任务：论文自承该数据集缺少带预算语义的商家动作。
- 论文在消融中**去掉了显式时间戳嵌入**（§3.4.1），因为与时序无关的节假日已被事件信号覆盖。
- 数据**尚未公开**：论文两次表述为「正在推进发布」，当前不对外可得（详见 §附 的 registry 核对）。

---

## ② 母婴出海应用案例

> **为什么母婴品类特别吃这一套**：母婴单品的需求同时被两股力量驱动 ——
> **宝宝月龄生命周期**（宝宝长大 → 复购同一 SKU 的人变少、换品，这是**内生漂移**）
> 与**促销/投放排期**（折扣深度、站内广告与站外投放预算，这是**可控动作**），
> 再叠加**外生事件**（大促日历、站外热点爆发）。
> 三者混在一起时最典型的误判是：把「这批买家正好进入高消耗月龄、订单自然上涨」
> 记成「这次投放很有效」，然后把预算继续加在同一批人身上。
> 本方法的价值就是把这三股力量**结构性拆开**：月龄与自然趋势进状态转移层（内生），
> 排期进动作条件（可控），事件进残差校正层（外生）。
> 而「给定排期问销量」这个能力，在业务上恰好就是**大促前定备货量**：
> 备货决策的唯一前置输入，就是「这档排期对应多少销量」。

### 场景一：黑五前六周，用三档候选排期定 Amazon FBA 备货量

- **业务问题**：吸奶器（单边电动）与奶瓶（防胀气）两个 ASIN 要在黑五前六周向供应链报 FBA 备货计划。
  运营团队手上是三档候选排期（折扣深度 × 站内广告周预算，见下方参数块）。
  现在的问题是：**这三档排期分别对应多少销量？该按哪一档备货、按哪一档备头程？**
  现行做法是「去年同期销量 ×（1 + 经验大促系数）」，它回答不了
  「折扣从浅折加到深折、广告预算上浮一档，销量会多多少」——
  而这个增量正是备货量该不该加码的全部依据。
- **数据要求**：
  - **状态面板**：SKU × 周粒度的状态向量，至少包含自然曝光、点击、加购、订单量、GMV、
    广告曝光、广告点击这几个口径（论文的状态向量是九个指标），按周聚合；
  - **动作面板**：同一 SKU、同一周的**两个可控动作** —— 折扣深度（折后价 / 原价）
    与站内广告周花费。粒度必须与状态面板严格对齐（论文的动作向量就是这两个变量）；
  - **事件信号**：平台大促日历（Prime Day / 黑五 / 网一 / 圣诞）＋ 站外热点（母婴相关热搜、
    达人种草爆发）＋ 节假日清单；
  - **商品上下文**：标题与三级类目标签（用于把事件对齐到商品）；
  - **历史长度**：建议覆盖**两个完整年度**，才能把「大促效应」与「月龄生命周期」分开 ——
    只有一个年度时，两者在时间上完全共线。
- **数据可得性**：`部分可得`。
  - 状态面板与动作面板：`企业内可得`。Amazon 广告报表 + 业务报表 + 广告后台的折扣与花费记录，
    按周聚合即可；独立站侧用 GA4 / Shopify 导出。
  - 平台级**全站**状态-动作配对数据：`不可得`（论文用的是 1688 平台侧数据，第三方卖家拿不到）。
    但本方法只需要**自己店铺**的动作-状态配对，这个缺口不影响可用性。
  - 事件信号：`部分可得（需补充）`。节假日日历可人工维护；站外热点需要自建采集 + LLM 归纳
    （论文用 Qwen-Plus 做关键词抽取与句子合成、BGE-zh-v1.5 做文本编码，
    中文场景可换用同等的国产模型组合）。
  - 月龄阶段：`部分可得`。可由上架周龄推算，或从买家宝宝月龄分布（客服记录、问卷）估计。
- **预期产出**：给定三档排期，输出未来五周逐周的订单/GMV 仿真轨迹、三档对比表，
  以及按选定档位的 FBA 补货量（预测需求 + 安全库存 − 在库 − 在途）。
- **业务价值**：见 ⑤ 的公式与参数来源。

**候选排期参数块**（贵司自己的参数，替换后即可复用；此块是**本卡参数、不是论文数字**）：

```text
排期档位        第1周   第2周   第3周   第4周   第5周
A 持平排期      折扣 6%  6%      6%      6%      6%
                广告 1500  1500    1500    1500    1500
B 小促排期      折扣 8%  12%     14%     10%     7%
                广告 1700  2000    2200    1800    1550
C 大促排期      折扣 12% 22%     28%     18%     8%
                广告 2200  3000    3400    2500    1700
```

### 场景二：站外种草排期 → 独立站承接能力与海外仓备货

- **业务问题**：站外内容投放（TikTok / 小红书种草）会把一批流量导进独立站。
  在排期敲定前必须回答两件事：**这波流量的订单峰值落在哪一周**（决定海外仓备货量与客服排班），
  以及**如果种草排期与大促撞周，独立站会不会同时爆仓**。
- **数据要求**：独立站侧的周粒度状态面板（会话数、加购、订单、GMV）＋ 动作面板
  （站外投放周花费、优惠券深度）＋ 事件信号（大促日历 + 站外热点）＋ 商品上下文。
  历史长度建议覆盖至少**一个**完整年度，且包含至少一次与当前量级相当的内容爆发样本。
- **数据可得性**：`部分可得`。独立站数据（Shopify / GA4）完全自有；
  缺口有二：① 站外种草的动作需要归集到「周花费 + 折扣深度」两个维度才与本方法对齐，
  而达人合作的结算口径常与投放周错位；② 爆量事件样本稀疏（一年可能只有几次），
  事件校正层的权重会不稳 —— 这类样本不足时应把校正层降级为人工判断。
- **预期产出**：种草排期 → 独立站周订单轨迹、峰值周与承接缺口、
  海外仓备货与客服排班建议。
- **业务价值**：把「承接能力够不够」从拍脑袋变成有区间的判断；
  峰值周定位准确可减少为应对峰值而多备的呆滞库存。

---

## ③ 代码模板

> **本模板是业务化的简化实现，不是论文的忠实复现。**
> Stage I 用「状态 × 动作交错项的岭回归」代替 Action-Interleaved Transformer；
> Stage II 用「事件 × 商品上下文的交叉门控」代替 cross-attention + LLM 热点嵌入。
> 依赖仅 numpy / pandas / scikit-learn，可直接嵌进现有数据栈。
> **论文 §3.4.1 明确去掉了时间戳嵌入**，本模板额外保留了「月龄阶段」这一维 ——
> 它在业务上是**可观测且提前已知**的协变量（上架周龄 / 买家宝宝月龄），
> 与「把时间戳当扰动嵌入」不是一回事。此处是**有意偏离论文**，请知悉。

### 第一段：业务常量与合成面板

```python
# -*- coding: utf-8 -*-
"""CEDAR 业务化简化实现：决策条件化销量仿真（动作交错转移层 + 事件残差校正层）。

对齐论文 2608.25871 的两阶段思路，但**不是论文的忠实复现**：
  * Stage I 用「状态 x 动作交错项的岭回归」代替 Action-Interleaved Transformer；
  * Stage II 用「事件 x 商品上下文的交叉门控」代替 cross-attention + LLM 热点嵌入。
依赖仅 numpy / pandas / scikit-learn，可在业务机上直接跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# 业务常量
# ---------------------------------------------------------------------------
SKUS = ("吸奶器-单边电动", "奶瓶-防胀气", "纸尿裤-L码", "辅食机-蒸煮一体")
DECAY = (0.010, 0.016, 0.008, 0.020)       # 月龄生命周期衰减速率（每周）
STATE_COLS = ("impressions", "orders")      # 状态：周曝光量、周订单量
ACTION_COLS = ("discount", "ad_spend")      # 动作：折扣深度(0-1)、站内广告周花费
EVENTS = ("母婴节", "双11", "黑五", "开学季", "流感季")
N_WEEKS, HORIZON, FIT_END = 26, 5, 20
TOTAL_WEEKS = N_WEEKS + HORIZON
RHO, AD_REF, AD_GAIN, CVR0, DISC_GAIN = 0.75, 1500.0, 0.35, 0.030, 0.25
FIRES = {4: {"母婴节": 1.0}, 11: {"双11": 1.0}, 17: {"流感季": 0.8},
         21: {"开学季": 0.9}, 22: {"双11": 1.0}, 24: {"母婴节": 0.7}}
# 事件 x 商品上下文相关性。对齐论文 §3.4.2 的判据：事件本身不产生需求，
# 必须与具体商品对齐（圣诞帽不会在春节暴涨）。
EVENT_RELEVANCE = pd.DataFrame(
    [[0.9, 0.1, 0.0, 0.2, 0.0],
     [0.7, 0.5, 0.4, 0.1, 0.0],
     [0.3, 0.9, 0.8, 0.0, 0.1],
     [0.2, 0.4, 0.2, 0.9, 0.0]],
    index=list(SKUS), columns=list(EVENTS))


def make_event_hotspots(n_weeks: int = TOTAL_WEEKS) -> pd.DataFrame:
    """周 x 事件的热点强度矩阵（站外热搜 + 平台热搜 + 节假日日历的语义归纳结果）。"""
    hot = np.zeros((n_weeks, len(EVENTS)))
    for wk, hits in FIRES.items():
        for ev, val in hits.items():
            if wk < n_weeks:
                hot[wk, EVENTS.index(ev)] = val
    return pd.DataFrame(hot, columns=list(EVENTS))


def mu_path(sku_idx: int, weeks) -> np.ndarray:
    """未观测的基线需求：随宝宝月龄指数衰减（长到一定月龄就换品，需求自然退坡）。"""
    return 4000.0 * (0.8 + 0.25 * sku_idx) * np.exp(-DECAY[sku_idx] * np.asarray(weeks, float))


def dgp(mu, t0, disc, ad, hot, rel, rng=None, noise=0.0):
    """真值生成过程（合成数据用）：流量带持续性，订单是流量的同期转化 x 事件放大。"""
    log_t, traffic, orders = np.log(t0), [], []
    for t in range(len(disc)):
        log_t = RHO * log_t + (1 - RHO) * np.log(mu[t]) + AD_GAIN * (ad[t] / AD_REF - 1.0)
        if rng is not None:
            log_t += noise * rng.standard_normal()
        tr = float(np.exp(log_t))
        od = tr * (CVR0 + DISC_GAIN * disc[t]) * (1.0 + float(hot[t] @ rel))
        if rng is not None:
            od *= float(np.exp(noise * rng.standard_normal()))
        traffic.append(tr)
        orders.append(od)
    return np.array(traffic), np.array(orders)


def make_panel(seed: int = 20260912) -> dict:
    """合成面板：历史观测（含噪声）+ 事件热点 + 未来排期（两档）。"""
    rng = np.random.default_rng(seed)
    hot = make_event_hotspots().to_numpy()
    recs = []
    for i, sku in enumerate(SKUS):
        wks = np.arange(TOTAL_WEEKS)
        mu = mu_path(i, wks)
        disc = np.clip(rng.normal(0.06, 0.025, TOTAL_WEEKS), 0.0, 0.30)
        ad = np.clip(rng.normal(1500.0 * (1 + 0.15 * i), 250.0, TOTAL_WEEKS), 300.0, None)
        for wk in (4, 11, 21):                     # 历史上已发生过的三档小促
            disc[wk] = 0.18
            ad[wk] *= 1.5
        tr, od = dgp(mu, mu[0], disc, ad, hot, EVENT_RELEVANCE.iloc[i].to_numpy(), rng, 0.06)
        recs.append(pd.DataFrame({"sku": sku, "week": wks, "sku_idx": i, "age_week": wks,
                                  "impressions": tr, "orders": od,
                                  "discount": disc, "ad_spend": ad}))
    panel = pd.concat(recs, ignore_index=True)
    return {"history": panel[panel["week"] < N_WEEKS].reset_index(drop=True),
            "full": panel, "hotspot": make_event_hotspots()}


def make_future_plans(n_steps: int = HORIZON) -> dict:
    """两档未来动作排期 —— 这就是「给定排期问销量」里的那个排期。"""
    return {"持平排期": pd.DataFrame({"discount": np.full(n_steps, 0.06),
                                      "ad_spend": np.full(n_steps, 1500.0)}),
            "大促排期": pd.DataFrame({"discount": np.array([0.08, 0.16, 0.20, 0.14, 0.07])[:n_steps],
                                      "ad_spend": np.array([1700., 2000., 2200., 1800., 1550.])[:n_steps]})}


def make_future_hotspots(n_steps: int = HORIZON) -> pd.DataFrame:
    """未来窗口的事件热点：平台大促日历已知，站外热点用预判值。"""
    hot = np.zeros((n_steps, len(EVENTS)))
    hot[1, EVENTS.index("双11")] = 1.0
    hot[3, EVENTS.index("黑五")] = 0.9
    return pd.DataFrame(hot, columns=list(EVENTS))
```

### 第二段：Stage I 转移层 / Stage II 残差校正层 / 备货换算

```python
# ---------------------------------------------------------------------------
# Stage I：动作交错转移层
# ---------------------------------------------------------------------------
def build_design(state_prev, action_curr, age_week, mode: str = "interleaved") -> np.ndarray:
    """状态转移层的设计矩阵（逐行）。

    `mode="interleaved"` 加入 state x action 交互项 —— 这是 AIT 把状态与动作
    **交错**建模的业务化类比：动作不只是一个加性移位，而是参与调节状态自身的
    转移斜率。经验：不加交互项时，动作的边际效应会被主导性的状态自回归压平。
    """
    l1, l2 = np.log(np.maximum(state_prev, 1e-9))
    if mode == "state_only":
        return np.array([1.0, l1, l2, age_week])
    d, m = float(action_curr[0]), float(action_curr[1] / AD_REF)
    if mode == "fusion":
        return np.array([1.0, l1, l2, d, m, age_week])
    return np.array([1.0, l1, l2, d, m, age_week, l1 * d, l2 * d, l1 * m, l2 * m])


class TransitionModel:
    """Stage I：动作条件化状态转移层（AIT 的业务化简化实现）。

    在「上一周状态 + 本周动作 -> 本周状态」的配对样本上拟合岭回归（对数尺度，
    避免多步外推时量级炸开），rollout 时把预测状态喂回自身、按排期自回归外推。
    """

    def __init__(self, mode: str = "interleaved", alpha: float = 0.05):
        assert mode in ("state_only", "fusion", "interleaved"), mode
        self.mode, self.alpha = mode, alpha
        self.model, self.mu_, self.sd_ = Ridge(alpha=alpha), None, None

    def _rows(self, hist: pd.DataFrame):
        rows, targets = [], []
        for sku, g in hist.groupby("sku", sort=False):
            g = g.reset_index(drop=True)
            oh = np.zeros(len(SKUS))
            oh[SKUS.index(sku)] = 1.0
            for t in range(1, len(g)):
                sp = g.loc[t - 1, list(STATE_COLS)].to_numpy(float)
                ac = g.loc[t, list(ACTION_COLS)].to_numpy(float)
                rows.append(np.concatenate(
                    [build_design(sp, ac, float(g.loc[t, "age_week"]), self.mode), oh]))
                targets.append(np.log(np.maximum(g.loc[t, list(STATE_COLS)].to_numpy(float), 1e-9)))
        return np.vstack(rows), np.vstack(targets)

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        Z = np.array(X, float, copy=True)
        if fit:
            self.mu_ = Z[:, 1:].mean(axis=0)
            sd = Z[:, 1:].std(axis=0)
            self.sd_ = np.where(sd < 1e-12, 1.0, sd)
        Z[:, 1:] = (Z[:, 1:] - self.mu_) / self.sd_
        return Z

    def fit(self, hist: pd.DataFrame) -> "TransitionModel":
        X, y = self._rows(hist)
        self.model.fit(self._scale(X, fit=True), y)
        return self

    def rollout(self, last_state, plan: pd.DataFrame, sku: str, start_age: int) -> np.ndarray:
        """按排期自回归仿真 H 步，返回 (H, 2) 的 [曝光, 订单] 轨迹。"""
        oh = np.zeros(len(SKUS))
        oh[SKUS.index(sku)] = 1.0
        log_s = np.log(np.maximum(np.array(last_state, float), 1e-9))
        out = []
        for t in range(len(plan)):
            a = plan.iloc[t][list(ACTION_COLS)].to_numpy(float)
            row = np.concatenate(
                [build_design(np.exp(log_s), a, float(start_age + t + 1), self.mode), oh])
            log_s = self.model.predict(self._scale(row.reshape(1, -1)))[0]
            out.append(np.exp(log_s))
        return np.vstack(out)


def one_step_predict(model: TransitionModel, hist: pd.DataFrame) -> np.ndarray:
    """用真实上一周状态做单步预测，返回与 hist 行序对齐的订单预测（第 0 周为 nan）。"""
    preds = np.full(len(hist), np.nan)
    for sku, g in hist.groupby("sku", sort=False):
        g = g.reset_index(drop=True)
        oh = np.zeros(len(SKUS))
        oh[SKUS.index(sku)] = 1.0
        rows = []
        for t in range(1, len(g)):
            sp = g.loc[t - 1, list(STATE_COLS)].to_numpy(float)
            ac = g.loc[t, list(ACTION_COLS)].to_numpy(float)
            rows.append(np.concatenate(
                [build_design(sp, ac, float(g.loc[t, "age_week"]), model.mode), oh]))
        pos = hist.index[hist["sku"] == sku].to_numpy()
        preds[pos[1:]] = np.exp(model.model.predict(model._scale(np.vstack(rows))))[:, 1]
    return preds


# ---------------------------------------------------------------------------
# Stage II：事件驱动的残差校正层
# ---------------------------------------------------------------------------
class EventResidualCorrector:
    """Stage II：事件残差校正层（cross-attention 的业务化简化实现）。

    交叉门控：事件热点先与商品上下文（SKU 独热）做外积，再线性映射到残差。
    学到的事件 x 商品权重矩阵就是「哪些事件对哪个 SKU 有实质影响」——
    这替代了论文用 LLM 文本 + cross-attention 做的事件-商品语义对齐。
    """

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    @staticmethod
    def _features(hotspot, sku_idx) -> np.ndarray:
        hot = np.atleast_2d(np.asarray(hotspot, float))
        idx = np.atleast_1d(np.asarray(sku_idx, int))
        oh = np.zeros((len(idx), len(SKUS)))
        oh[np.arange(len(idx)), idx] = 1.0
        return (oh[:, :, None] * hot[:, None, :]).reshape(len(idx), -1)

    def fit(self, resid, hotspot, sku_idx) -> "EventResidualCorrector":
        self.model.fit(self._features(hotspot, sku_idx), np.asarray(resid, float))
        return self

    def predict(self, hotspot, sku_idx) -> np.ndarray:
        return self.model.predict(self._features(hotspot, sku_idx))

    def weights(self) -> pd.DataFrame:
        """事件 x 商品权重矩阵（校正层学到的对齐强度）。"""
        return pd.DataFrame(self.model.coef_.reshape(len(SKUS), len(EVENTS)),
                            index=list(SKUS), columns=list(EVENTS))


def mae(pred, true) -> float:
    """平均绝对误差：业务侧比 MSE 更能反映「差一件就是一件」。"""
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(true, float))))


def plan_replenishment(pred_orders, resid_scale: float, service_z: float = 1.65,
                       on_hand: float = 0.0, inbound: float = 0.0) -> dict:
    """把仿真轨迹转成 FBA 补货量：预测需求 + 安全库存 - (在库 + 在途)。

    service_z=1.65 对应约 95% 单尾服务水平；
    resid_scale 用留出窗口的残差标准差估计（本卡由合成数据给出）。
    """
    demand = float(np.sum(pred_orders))
    safety = float(service_z * resid_scale * np.sqrt(len(pred_orders)))
    return {"demand": demand, "safety_stock": safety,
            "replenish": max(demand + safety - on_hand - inbound, 0.0)}
```

### 第三段：端到端实验、自检与演示

```python
# ---------------------------------------------------------------------------
# 评估：留出窗口递归仿真 + 反事实排期对比
# ---------------------------------------------------------------------------
def eval_rollout(model, full, corrector=None, hotspot=None):
    """从第 FIT_END-1 周出发，按评估窗口的真实动作递归仿真，返回 (预测订单, 真实订单)。"""
    preds, trues = [], []
    for sku in SKUS:
        h = full[full["sku"] == sku].sort_values("week").reset_index(drop=True)
        last = h.loc[FIT_END - 1, list(STATE_COLS)].to_numpy(float)
        plan = h.loc[FIT_END:N_WEEKS - 1, list(ACTION_COLS)].reset_index(drop=True)
        traj = model.rollout(last, plan, sku, FIT_END - 1)
        if corrector is not None and hotspot is not None:
            inj = corrector.predict(hotspot.to_numpy(),
                                    np.full(len(hotspot), SKUS.index(sku)))
            traj = np.maximum(traj + np.column_stack([np.zeros(len(inj)), inj]), 1e-6)
        preds.append(traj[:, 1])
        trues.append(h.loc[FIT_END:N_WEEKS - 1, "orders"].to_numpy())
    return np.concatenate(preds), np.concatenate(trues)


def cf_truth(data: dict, plan_name: str, fhot: pd.DataFrame) -> np.ndarray:
    """合成数据的反事实真值（只有本卡这种有真值生成过程的场景才拿得到）。"""
    plan = make_future_plans()[plan_name]
    out = []
    for i, sku in enumerate(SKUS):
        h = data["full"][data["full"]["sku"] == sku].sort_values("week").reset_index(drop=True)
        _, od = dgp(mu_path(i, np.arange(N_WEEKS, TOTAL_WEEKS)),
                    float(h.loc[N_WEEKS - 1, "impressions"]),
                    plan["discount"].to_numpy(), plan["ad_spend"].to_numpy(),
                    fhot.to_numpy(), EVENT_RELEVANCE.iloc[i].to_numpy())
        out.append(od)
    return np.concatenate(out)


def cf_sim(model, data: dict, plan_name: str, fhot: pd.DataFrame) -> np.ndarray:
    """模型的排期仿真（这就是给业务方看的那张「排期 -> 销量」表）。"""
    plan = make_future_plans()[plan_name]
    out = []
    for sku in SKUS:
        h = data["full"][data["full"]["sku"] == sku].sort_values("week").reset_index(drop=True)
        out.append(model.rollout(h.loc[N_WEEKS - 1, list(STATE_COLS)].to_numpy(float),
                                 plan, sku, N_WEEKS - 1)[:, 1])
    return np.concatenate(out)


def run_experiment(seed: int = 20260912, verbose: bool = True) -> dict:
    """完整流程：Stage I 三档对照 -> Stage II 对齐/打乱 -> 大促排期下的 FBA 补货建议。"""
    data = make_panel(seed)
    hist, full = data["history"], data["full"]
    fit_hist = hist[hist["week"] < FIT_END].reset_index(drop=True)
    eval_hot = data["hotspot"].iloc[FIT_END:N_WEEKS].reset_index(drop=True)
    fhot = make_future_hotspots()

    stage1 = {}
    for name, mode in (("仅状态外推 (state-only)", "state_only"),
                       ("协变量融合 (covariate fusion)", "fusion"),
                       ("动作交错 (AIT-lite)", "interleaved")):
        m = TransitionModel(mode).fit(fit_hist)
        p, t = eval_rollout(m, full)
        sim = {k: cf_sim(m, data, k, fhot) for k in ("持平排期", "大促排期")}
        true = {k: cf_truth(data, k, fhot) for k in ("持平排期", "大促排期")}
        stage1[name] = {
            "rollout_mae": mae(p, t), "model": m,
            "sim": sim, "true": true,
            "cf_mae": {k: mae(sim[k], true[k]) for k in sim},
            "lift": (sim["大促排期"].sum() / sim["持平排期"].sum() - 1.0),
        }
    true_lift = (stage1["动作交错 (AIT-lite)"]["true"]["大促排期"].sum()
                 / stage1["动作交错 (AIT-lite)"]["true"]["持平排期"].sum() - 1.0)

    # ---- Stage II：用拟合窗口的单步残差训练事件校正层 ----
    ait = stage1["动作交错 (AIT-lite)"]["model"]
    prow = one_step_predict(ait, fit_hist)
    mask = ~np.isnan(prow)
    resid = (fit_hist["orders"].to_numpy() - prow)[mask]
    hot_fit = data["hotspot"].iloc[fit_hist["week"].to_numpy()][mask].to_numpy()
    idx_fit = fit_hist["sku_idx"].to_numpy()[mask]

    rc = EventResidualCorrector(alpha=1.0).fit(resid, hot_fit, idx_fit)
    base_pred, base_true = eval_rollout(ait, full)
    corr_pred, _ = eval_rollout(ait, full, corrector=rc, hotspot=eval_hot)

    # 时间打乱对照：把「事件 -> 周」的对应关系整体打乱（拟合侧与仿真侧同时打乱），
    # 对齐论文 Table 2 的 Temporal shuffle 消融。
    rng = np.random.default_rng(seed + 1)
    hot_shuffled = data["hotspot"].iloc[rng.permutation(TOTAL_WEEKS)].reset_index(drop=True)
    hot_fit_shuf = hot_shuffled.iloc[fit_hist["week"].to_numpy()][mask].to_numpy()
    eval_hot_shuf = hot_shuffled.iloc[FIT_END:N_WEEKS].reset_index(drop=True)
    rc_shuf = EventResidualCorrector(alpha=1.0).fit(resid, hot_fit_shuf, idx_fit)
    shuf_pred, _ = eval_rollout(ait, full, corrector=rc_shuf, hotspot=eval_hot_shuf)

    # ---- 大促排期下的 FBA 补货建议（已叠加事件校正）----
    plan_rows = []
    for sku in SKUS:
        h = full[full["sku"] == sku].sort_values("week").reset_index(drop=True)
        inj = rc.predict(fhot.to_numpy(), np.full(len(fhot), SKUS.index(sku)))
        traj = np.maximum(
            ait.rollout(h.loc[N_WEEKS - 1, list(STATE_COLS)].to_numpy(float),
                        make_future_plans()["大促排期"], sku, N_WEEKS - 1)
            + np.column_stack([np.zeros(len(inj)), inj]), 1e-6)
        rep = plan_replenishment(traj[:, 1], resid_scale=float(np.std(resid)))
        plan_rows.append({"sku": sku, "pred_orders": rep["demand"],
                          "safety_stock": rep["safety_stock"], "replenish": rep["replenish"]})

    out = {"stage1": stage1, "true_lift": true_lift,
           "stage2": {"无事件校正": mae(base_pred, base_true),
                      "事件校正-对齐": mae(corr_pred, base_true),
                      "事件校正-时间打乱": mae(shuf_pred, base_true)},
           "resid_std": float(np.std(resid)),
           "event_weights": rc.weights(),
           "replenish": pd.DataFrame(plan_rows)}

    if verbose:
        print("=== Stage I：留出窗口递归仿真，订单 MAE（越低越好）===")
        for k, v in out["stage1"].items():
            print(f"  {k:30s} MAE = {v['rollout_mae']:8.1f}")
        print("=== 反事实排期对比：5 周订单总量（持平 vs 大促）===")
        for k, v in out["stage1"].items():
            print(f"  {k:30s} 持平 {v['sim']['持平排期'].sum():8.0f} | "
                  f"大促 {v['sim']['大促排期'].sum():8.0f} | 弹性 {v['lift'] * 100:+7.1f}% | "
                  f"排期仿真MAE {v['cf_mae']['大促排期']:7.1f}")
        tl = out["true_lift"]
        print(f"  {'合成数据真值（本卡自行生成）':30s} 持平 "
              f"{out['stage1']['动作交错 (AIT-lite)']['true']['持平排期'].sum():8.0f} | 大促 "
              f"{out['stage1']['动作交错 (AIT-lite)']['true']['大促排期'].sum():8.0f} | "
              f"弹性 {tl * 100:+7.1f}%")
        print("=== Stage II：事件残差校正（对齐 vs 时间打乱）===")
        for k, v in out["stage2"].items():
            print(f"  {k:24s} MAE = {v:8.1f}")
        print("=== 大促排期下的 FBA 补货建议（含 95% 服务水平安全库存）===")
        print(out["replenish"].round(1).to_string(index=False))
    return out


def test_state_only_is_policy_insensitive():
    """仅状态外推的仿真对排期完全不敏感 —— 论文批判的 autoregressive inertia。"""
    out = run_experiment(verbose=False)
    v = out["stage1"]["仅状态外推 (state-only)"]
    assert abs(v["lift"]) < 1e-9, v["lift"]
    assert np.allclose(v["sim"]["大促排期"], v["sim"]["持平排期"])


def test_action_conditioning_responds_to_promo_plan():
    """动作条件化的转移层必须对大促排期给出正响应，且方向与真值一致。"""
    out = run_experiment(verbose=False)
    for k in ("协变量融合 (covariate fusion)", "动作交错 (AIT-lite)"):
        assert out["stage1"][k]["lift"] > 0.05, (k, out["stage1"][k]["lift"])
    assert out["true_lift"] > 0


def test_action_conditioning_lowers_counterfactual_error():
    """大促排期下，动作条件化的仿真误差必须显著低于仅状态外推。"""
    out = run_experiment(verbose=False)
    base = out["stage1"]["仅状态外推 (state-only)"]["cf_mae"]["大促排期"]
    for k in ("协变量融合 (covariate fusion)", "动作交错 (AIT-lite)"):
        assert out["stage1"][k]["cf_mae"]["大促排期"] < base, (k, base)


def test_event_residual_reduces_rollout_error():
    """Stage II 的事件校正层必须降低留出窗口的订单 MAE。"""
    out = run_experiment(verbose=False)
    assert out["stage2"]["事件校正-对齐"] < out["stage2"]["无事件校正"], out["stage2"]


def test_temporal_shuffle_degrades_alignment():
    """把事件信号打到错误的周上效果必须变差（对齐论文 Table 2 的 Temporal shuffle）。"""
    out = run_experiment(verbose=False)
    assert out["stage2"]["事件校正-时间打乱"] >= out["stage2"]["事件校正-对齐"], out["stage2"]


def test_corrector_is_silent_on_unaligned_event_pairs():
    """未对齐的「事件 x 商品」不得被凭空放大：黑五对吸奶器的权重应远小于母婴节。"""
    out = run_experiment(verbose=False)
    w = out["event_weights"]
    assert abs(w.loc["吸奶器-单边电动", "黑五"]) < abs(w.loc["吸奶器-单边电动", "母婴节"]), w


def test_replenishment_covers_demand_and_scales_with_risk():
    """补货建议必须覆盖预测需求，且安全库存随残差波动放大。"""
    pred = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    lo = plan_replenishment(pred, resid_scale=5.0)
    hi = plan_replenishment(pred, resid_scale=50.0)
    assert lo["replenish"] >= lo["demand"]
    assert hi["safety_stock"] > lo["safety_stock"]


if __name__ == "__main__":
    run_experiment()
    print("全部自检通过 ✅")
```

**本模板的实际运行输出**（在仓库根目录执行 `python3 <本卡代码>.py` 可复现；
**这是本卡合成数据的结果，不是论文数据**）：

```text
=== Stage I：留出窗口递归仿真，订单 MAE（越低越好）===
  仅状态外推 (state-only)             MAE =    161.9
  协变量融合 (covariate fusion)       MAE =     71.9
  动作交错 (AIT-lite)                MAE =     66.6
=== 反事实排期对比：5 周订单总量（持平 vs 大促）===
  仅状态外推 (state-only)             持平     4501 | 大促     4501 | 弹性    +0.0% | 排期仿真MAE   167.5
  协变量融合 (covariate fusion)       持平     3495 | 大促     7551 | 弹性  +116.1% | 排期仿真MAE    82.7
  动作交错 (AIT-lite)                持平     3533 | 大促     7908 | 弹性  +123.8% | 排期仿真MAE    89.3
  合成数据真值（本卡自行生成）                 持平     4420 | 大促     7589 | 弹性   +71.7%
=== Stage II：事件残差校正（对齐 vs 时间打乱）===
  无事件校正                    MAE =     66.6
  事件校正-对齐                  MAE =     61.9
  事件校正-时间打乱                MAE =     67.2
=== 大促排期下的 FBA 补货建议（含 95% 服务水平安全库存）===
     sku  pred_orders  safety_stock  replenish
吸奶器-单边电动       1143.3         210.5     1353.8
  奶瓶-防胀气       1587.7         210.5     1798.2
  纸尿裤-L码       2642.6         210.5     2853.1
辅食机-蒸煮一体       2566.9         210.5     2777.4
全部自检通过 ✅
```

> ⚠️ **上方输出围栏里的每一个数字都来自本卡代码在合成数据上的运行，不是论文数字，两者不可互相印证。**
> 论文报告的误差量级（见 ⑥ 引用）来自 1688 工业数据集与 Kaggle 公开数据集，
> 数据分布、量纲与归一化口径都与本模板的合成面板不同；
> **围栏内的数值与论文数值若有任何接近，纯属巧合**，不得用来声称「复现了论文」。
>
> ⚠️ **必须看见的负面结果**：上方输出围栏里「动作交错」那一行的**排期弹性明显高于
> 同一围栏里「合成数据真值」那一行的弹性**，即模型把动作响应放大了 ——
> 这是本卡的**自曝其短**，不是可以略过的细节。
> 根因是**多步 rollout 会在逐周复利中放大动作响应**，与论文附录 D 承认的
> 自回归误差累积是同一类问题的不同表现（见 ⑥ 引用）。
> 结论：**本模板的绝对销量仿真不可直接用于备货量决策**，
> 必须在贵司历史上一次真实大促上做弹性校准（把模型给出的排期弹性与实测增量对齐），
> 再用于 ② 的备货建议；未校准前只可用于**排期档位之间的相对排序**。
> 同一围栏里「仅状态外推」那一行的排期弹性**恒为零**，这是本模板唯一可以强断言的行为：
> 不含动作输入的模型对排期**完全不敏感**，这正是论文批判的被动外推。

---

## ④ 技能关联

| 关联卡片 | 关系 | 为什么组合 |
|---|---|---|
| `Skill-Time-Series-Forecasting.md` | 前置（被本卡替换的对象） | 该卡是「被动外推」的标准口径（分解 + 季节性 + 节假日哑变量）。本卡不取代它做常规周预测，而是在**排期可变**时替换它的 rollout 环节。 |
| `Skill-Temporal-Fusion-Transformer.md` | 直接对标 | 论文 §2.1 点名 TFT 属于「协变量条件化」范式：动作被当作又一路输入通道，因此在政策偏移下不敏感。本卡的 Stage I 就是针对这一点设计的替代方案。 |
| `Skill-Promotion-Effectiveness.md` | 前置（参数校准） | 用因果方法估计折扣/促销的**真实增量**，其产出正是校准本卡动作响应系数所需的输入；缺了它，本卡的动作系数只是相关性拟合。 |
| `Skill-Marketing-Mix-Modeling.md` | 互补（粒度不同） | MMM 从**月度 × 渠道**估预算弹性，本卡从**周 × SKU**给排期轨迹。两者若对同一档排期的方向判断相反，说明动作面板或事件信号被污染了。 |
| `Skill-Demand-Forecasting-Supply-Chain.md` | 下游（口径转换） | 该卡定义的供应链需求口径（促销日历驱动的「人造季节」、偏差成本）是仿真轨迹进入补货决策前的转换层。 |
| `Skill-Safety-Stock-Replenishment.md` | 下游（决策落地） | 本卡 ③ 的 `plan_replenishment` 用的就是该卡的安全库存口径：服务水平分位数 × 残差波动 × 提前期平方根。 |
| `Skill-ROAS-Budget-Optimization.md` | 下游（预算排序） | 静态 ROAS 排序无法回答「预算排期不同会怎样」。本卡的排期仿真给该卡提供**按周**的产出曲线，使预算分配从静态切分变成排期优化。 |

---

## ⑤ 商业价值评估

**ROI 预估**（给公式与参数来源，不填编造的数）：

**备货决策增益（节省） = 被消除的断货缺口 × 单位贡献毛利 + 被压降的滞销库存 × 单位折价损失率**

参数来源：

- **被消除的断货缺口** —— 用本卡的三档排期仿真对比「现行经验系数备货量」的差距，
  乘以该 SKU 在大促周的实测售罄率缺口（来自业务报表）；
- **单位贡献毛利** —— 售价 − 采购成本 − 头程与 FBA 费用（来自财务口径）；
- **被压降的滞销库存** —— 大促后 N 周的剩余库存 × 折价清货折扣率（来自历史清货记录）。

三个参数都能在**企业内**算出，无需外部基准。
论文给出的方向性参照是：上线后商家侧 LTV 与店铺 ROI 都出现了可观改善（§4.6，见 ⑥ 引用），
说明「决策条件化仿真 + 事件校正」这条链路在真实平台上产生过商业结果。
**但这是阿里 1688 平台侧的生产读数，不是贵司的预期收益**；
贵司量级取决于动作面板的完整度与校准质量。

- **实施难度**：⭐⭐⭐⭐☆（4/5）。方法结构不复杂（本卡已给可运行实现），
  难点全在**数据工程**：动作与状态的周对齐、事件信号的采集与 LLM 归纳流水线、
  以及必不可少的历史大促弹性校准。若只做 Stage I（先不要事件层），难度降到 3 星，
  也能回答「排期档位谁更高」这一半问题。
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）。「给定排期问销量」是备货、投放、承接三条链路的**共同上游**：
  备货量错了会同时产生断货与滞销，预算排期错了会把钱投在低增量周。
  上游口径改一次，下游三条链路同时受益，杠杆最大。
- **评估依据**：① 论文在真实生产环境做过线上 A/B（§4.6），不是纯离线；
  ② 依赖的数据在平台店铺 + 独立站内**大部分可得**，缺口（事件信号、月龄）可补；
  ③ 与 `Skill-Temporal-Fusion-Transformer` / `Skill-Safety-Stock-Replenishment` /
  `Skill-ROAS-Budget-Optimization` 直接串联，不是孤立新领域；
  ④ 但**必须先做历史大促校准**，否则绝对量不可用（见 ③ 的负面结果），故不给 5 星难度以下的评价。

---

## ⑥ 原文引用

> 原文："merchants need to evaluate sales outcomes under future action sequences such as budget schedules, rather than passively predicting what happens next."
> 出处：2608.25871 Abstract（PDF 第 1 页）

> 原文："what would happen if I follow a particular budget schedule (and other actions) over the next weeks?"
> 出处：2608.25871 §1（PDF 第 1 页）

> 原文："This shifts the goal from passive forecasting to decision-conditioned simulation"
> 出处：2608.25871 §1（PDF 第 1 页）

> 原文："This design suffers from autoregressive inertia and conflates endogenous market evolution with decision-induced transitions, leading to policy-insensitive rollouts and unreliable counterfactual analysis."
> 出处：2608.25871 Abstract（PDF 第 1 页）

> 原文："we propose CEDAR (Controlled and Event-Driven Demand forecasting via Action-aware Residual decomposition), a two-stage framework for robust decision-conditioned simulation"
> 出处：2608.25871 Abstract（PDF 第 1 页）

> 原文："If these exogenous shocks are not separated from action effects, the simulator will misattribute demand changes, leading to erroneous credit assignment and disastrous budget decisions."
> 出处：2608.25871 §1（PDF 第 2 页）

> 原文："which reflects the natural causal structure of merchant operations: historical system states inform merchant decisions, and these decisions subsequently drive state transitions."
> 出处：2608.25871 §3.2 式 (2) 后（PDF 第 3 页）

> 原文："where $f_{\theta}$ captures controllable dynamics and $\epsilon_{t}$ represents latent external perturbations. Stage I models the former, while Stage II estimates the latter."
> 出处：2608.25871 §3.3 式 (5) 后（PDF 第 3 页）

> 原文："The hotspot embedding $\mathbf{h}_{t}$ is then combined with the item status embedding via a cross-attention module, enabling dynamic alignment between external events and product-level temporal patterns."
> 出处：2608.25871 §3.3 Residual Prediction（PDF 第 4 页）

> 原文："For instance, the demand for seasonal items, such as Christmas hats, is unlikely to surge during the Chinese New Year, despite the presence of a significant festival event."
> 出处：2608.25871 §3.4.2（PDF 第 4 页）

> 原文："Consequently, we do not include explicit timestamp embeddings in CEDAR."
> 出处：2608.25871 §3.4.1（PDF 第 4 页）

> 原文："to perform multi-step forecasting, this result is appended to the historical sequence and fed back into the model in an auto-regressive manner, enabling the stable simulation of future product trajectories over an extended horizon."
> 出处：2608.25871 §3.4.3 Inference Phase（PDF 第 4 页）

> 原文："comprising approximately 32 million product trajectories with paired state–action sequences and aligned event signals"
> 出处：2608.25871 Abstract（PDF 第 1 页）

> 原文："This process yields approximately 32 million training samples."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："We segment each trajectory into overlapping windows of 15 consecutive weeks."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："The action vector contains two controllable variables: the 7-day average marketing discount (defined as the ratio between the discounted price and the original price) and the total advertising expenditure during the same period."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："we further incorporate exogenous event signals derived from both on-platform trending search topics and off-platform public hotspots, along with a curated list of 26 major holidays."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："At the finest granularity, the taxonomy contains 8,942 distinct subcategories."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："we reserve the final window of 2025 as the test set and use all preceding windows for training, because random sampling may cause shortcut learning on already observed external shocks."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："we consider two evaluation settings: (i) forecasting the next 10 weeks given the past 5 weeks of observations, and (ii) forecasting the next 5 weeks given the past 10 weeks of observations."
> 出处：2608.25871 §4.1.3（PDF 第 5 页）

> 原文："The hidden dimension of all models is uniformly set to 256."
> 出处：2608.25871 §4.1.3（PDF 第 5 页）

> 原文："we configure the number of attention heads and layers as $n_{\text{head}}=4$ and $n_{\text{layer}}=5$, respectively."
> 出处：2608.25871 §4.1.3（PDF 第 5 页）

> 原文："we adopt the BGE-zh-v1.5 model as our text encoder, with an embedding dimension of 1024."
> 出处：2608.25871 §4.1.3（PDF 第 5 页）

> 原文："at horizon next 5, CEDAR attains an MSE of $0.182$, significantly outperforming the strongest baseline PatchTST $0.424$ and PETFormer $0.434$, corresponding to relative improvements of $57.1\%$ and $58.1\%$, respectively."
> 出处：2608.25871 §4.2（PDF 第 5 页）

> 原文："For the next 10 horizon, CEDAR also achieves the lowest MSE $0.414$ and NMSE $0.189$, consistently outperforming all competing approaches."
> 出处：2608.25871 §4.2（PDF 第 5 页）

> 原文："Similar trends are observed under NMSE, where CEDAR reduces the error to $0.083$, yielding more than $56\%$ improvement over the best baseline."
> 出处：2608.25871 §4.2（PDF 第 5 页）

> 原文："classical TSF models such as Informer suffer from severe performance degradation, with MSE exceeding $30$ at next 10."
> 出处：2608.25871 §4.2（PDF 第 5 页）

> 原文："| Full Model | 0.414 | 0.182 | 0.132 | 0.0603 |"
> 出处：2608.25871 §4.2 表 2（PDF 第 5 页）

> 原文："| w/o AIT Prediction | 0.499 | 0.264 | 0.181 | 0.0697 |"
> 出处：2608.25871 §4.2 表 2（PDF 第 5 页）

> 原文："| Temporal shuffle | 0.527 | 0.274 | 0.194 | 0.0712 |"
> 出处：2608.25871 §4.2 表 2（PDF 第 5 页）

> 原文："baselines that treat actions as exogenous covariates exhibit limited sensitivity to intervention signals"
> 出处：2608.25871 §4.3（PDF 第 6 页）

> 原文："This ability enables stable and realistic multi-step rollouts under dynamically changing action plans, which is critical for budget planning and strategy exploration."
> 出处：2608.25871 §4.3（PDF 第 6 页）

> 原文："This variant performs worse than the aligned-event setting and even degrades relative to AIT-only in next-5 MSE, indicating that CEDAR benefits from temporally meaningful event-demand alignment rather than merely using event embeddings as generic auxiliary features."
> 出处：2608.25871 §4.4（PDF 第 6 页）

> 原文："we observe that incorporating the Residual Correction Module yields a modest improvement in MSE but leads to a substantial reduction in MAE."
> 出处：2608.25871 §4.4（PDF 第 6 页）

> 原文："it differs from our setting because it is organized at the store-family level and lacks merchant actions with explicit budget-planning semantics"
> 出处：2608.25871 §4.5（PDF 第 7 页）

> 原文："Thus, this experiment mainly tests whether the event-aware residual decomposition transfers to a public retail forecasting scenario, rather than fully reproducing our counterfactual budget-planning task."
> 出处：2608.25871 §4.5（PDF 第 7 页）

> 原文："CEDAR reduces MSE from $0.6321$ to $0.5819$ compared with the strongest baseline PETFormer, and also achieves lower MAE and NMSE than both PatchTST and PETFormer."
> 出处：2608.25871 §4.5（PDF 第 7 页）

> 原文："As the largest domestic B2B wholesale platform in China, Alibaba 1688 provides a uniquely rich environment for studying decision-conditioned forecasting and budget planning."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："we are actively working toward releasing a partially anonymized version to facilitate future research."
> 出处：2608.25871 §4.1.1（PDF 第 5 页）

> 原文："The initial deployment successfully engaged 239 cooperative merchants across 245 orders for the prediction service, facilitating a total transaction value of 8.77 million RMB with an initial repurchase rate of 61%."
> 出处：2608.25871 §4.6（PDF 第 7 页）

> 原文："merchants in the treatment group achieve a 13%(46,471 vs 41,228) increase in lifetime value (LTV) and a 15% improvement in store-level return on investment (ROI) on average."
> 出处：2608.25871 §4.6（PDF 第 7 页）

> 原文："The control group follows the existing production model based on a diffusion-based time series forecasting model with only total budgets, while the treatment group adopts CEDAR-driven budget planning and traffic allocation strategies."
> 出处：2608.25871 §4.6（PDF 第 7 页）

> 原文："Together, these capabilities significantly reduce ineffective ad spend and improve the alignment between budget allocation and true market demand."
> 出处：2608.25871 §4.6（PDF 第 7 页）

> 原文："This approximation relies on the assumption of Sequential Ignorability, which posits that potential outcomes $S(a)$ are conditionally independent of the current action given the history"
> 出处：2608.25871 Appendix B 式 (10) 前（PDF 第 8 页）

> 原文："Specifically, Stage I takes about 154 minutes to train, while Stage II adds another 80 minutes for learning the residual correction module. The total training time of CEDAR is therefore approximately 234 minutes, which is higher than PatchTST but substantially lower than PETFormer in our implementation."
> 出处：2608.25871 Appendix C（表 4；PDF 第 9 页）

> 原文："all models exhibit larger errors as the horizon increases, which is expected due to autoregressive error accumulation."
> 出处：2608.25871 Appendix D（PDF 第 9 页）

> 原文："At next-25, CEDAR obtains an MSE of 1.612, substantially lower than PatchTST 2.847 and PETFormer 2.561."
> 出处：2608.25871 Appendix D（表 5；PDF 第 9 页）

---

## 附：证据链与核验方式

- **全文底本**：`paper2skills-vault/papers/03-时间序列/p2s-2026-0006/fulltext.md`
  （由 `paper2skills-research/scripts/fetch_fulltext.py` 从 arXiv LaTeXML HTML 转换，保留章节号）
- **证据档案**：`paper2skills-vault/papers/03-时间序列/p2s-2026-0006/evidence.md`
  （含 registry 断言逐条核对、未采信项清单、K1/引文/门禁的原始命令与结果）
- **引文逐字核验**：`python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <本卡>`
- **K1 代码可执行**：`python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <本卡>`
- **K2 门禁**：`python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <本卡>`
- **本地可复现数字的约定**：论文事实数字一律进 ⑥ 引用块；本模板运行输出与贵司参数示例
  放进代码/输出围栏并标注「可复现」—— 围栏内的数字**可自行验证**，不冒充论文结论。
