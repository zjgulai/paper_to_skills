---
title: Skill-Causal-Budget-Allocation
module: 13-广告分析
topic: 增量优先 + 全局硬约束的预算分配（因果 CATE → 神经 Bandit 探索 → 对偶 LP 影子价格）
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.10182
paper: From Prediction to Incrementality: Causal Optimization for Large-Scale Targeting and Recommendation
venue: arXiv preprint
venue_tier: preprint
venue_source: frontmatter-as-is
evidence_grade: A
verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py G2 passed + 人工抽检 3 处数字
verified_at: 2026-09-12
supersedes:
related: Skill-ROAS-Budget-Optimization.md, Skill-Ad-Attribution-Modeling.md, Skill-Uplift-Modeling.md, Skill-Multi-Armed-Bandit.md, Skill-Marketing-Mix-Modeling.md
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-05
l2_domain: 品牌与增长
l3_id: DOM-05-096
l3_business: 预算分配
l3_all: 预算分配 / 增量分析
l1_l2_l3: 业务运营/品牌与增长/预算分配
---

# Skill Card: 增量优先的约束预算分配（Causal Budget Allocation under Global Constraints）

**与同目录 `Skill-ROAS-Budget-Optimization.md` 的分工（两张卡不重复）**：那张卡解决「给定各渠道的
花费-收入曲线，如何让边际 ROAS 相等」——输入是历史 ROAS，是相关性口径；本卡解决「曲线的输入本身
不可信（平台归因天然高估），且分配必须同时满足广告位容量、独立站承接、现金三类硬约束」——输入是
因果增量，输出是每条约束的**影子价格**。前者是「怎么分」，本卡是「分给谁才算增量、以及到底是什么
在卡脖子」。

---

## ① 算法原理

**核心思想**：把「谁会响应」换成「谁会**因为这次投放才**响应」。用预测响应排序再按启发式分预算，
钱会流向「不投也会买」的人；本方法在全局硬约束下直接最大化因果增量：max Σ τ̃ x，
s.t. 守卫约束 Σ τ^{guardrail} x ≤ C、单用户频次 Σ_i x_{u,i} ≤ C_fcap、x ∈ {0,1}。

**数学直觉**：增量 τ(X) = E[Y(1) − Y(0) | X]。把 x 松弛成概率后用**平滑对偶分解**，对偶变量 λ
就是每条约束的**影子价格**：x*_{γ,u}(λ) = Π_{C_u}[ −(A_u^T λ + c_u) / γ ]，即每个用户只看
「增量目标 − 资源消耗 × 影子价格」再做一次投影；λ 沿 ∇g_γ(λ) = A x*_γ(λ) − b 做 Nesterov 加速
上升。某条约束被挤爆时它的 λ 上升，预算自动绕开这个瓶颈——这就是「哪类资源才是真瓶颈」的答案。
Bandit 层从后验采样 τ̃ 而不是把点估计直接交给分配器，使每个可行动作都有正概率入选，改善 overlap。

**关键假设**：论文 §2.1 的两个识别条件（无混淆 + 重叠）与 §2.2 式 (2) 的正性结论；不满足则增量
不可点识别，本卡只能当诊断框架用。

---

## ①b 反例与适用边界

**什么时候不要用这个算法**

1. **拿不到随机化或可信的「未触达」对照**。平台站内广告（Amazon SP/SB/SD）通常没有受众级随机化，
   无混淆假设不成立，τ 不可点识别。论文的线上结论是建立在 50/50 随机分臂之上的（⑥ Q24）。
2. **干预量级不够大**。对偶分解的收益来自「单轮变量数为 |U| × |I|、通用求解器不可解」这个规模；
   论文自己在离线实验里，约 400K 成员 × 5 个动作就直接用 OR-Tools 求解了，没有上自研求解器（⑥ Q16）。
3. **没有跨渠道共享资源**。论文明确指出：没有共享分配约束时，该策略等价于 Thompson sampling，
   此时引入 LP 层是多余的（⑥ Q33）。
4. **主指标回收窗口短于论文口径**。论文把处理窗设为 7 天、后续观察窗至少 30 天（Y 在 [D+7, R] 观测，
   ⑥ Q11/Q12），意味着不能在投放结束后几天就结算增量。

**已知的失败模式**（均有论文出处）

1. **实验臂投递量不匹配** → 把「策略更差」和「投得更少」混在一起，污染整个读数（⑥ Q13）。
2. **预期会看到的变化（不是故障）**：按增量口径分预算后，**总投放量通常会下降**——引擎会主动
   放过「本来就会买」的人，论文的离线实验里增量组的不推荐比例高于倾向组（⑥ Q39）。因此不能用
   「花费规模」或「触达人数」考核这条链路，必须看净收益。
3. **排除段用动态条件**（公司 / 地区 / 活跃度）→ 成员跨臂漂移且与平台活跃度相关，污染 ITT 对比（⑥ Q25）。
4. **两臂复用共享 segment 定义** → 控制组会通过被复用的子段静默继承处理组的发送决策（⑥ Q26）。
5. **探索的短期代价**。多轮模拟中 Bandit 版本在前若干轮不如 greedy 版本，约 50 次模型更新后才反超
   （⑥ Q19）；把它拿去跑短期 KPI 考核会被误杀。
6. **特征误用**。去掉 dense 特征后 outcome AUROC 从 0.857 降到 0.826，但 uplift AUUC 反而持平或更好
   （⑥ Q22）——这类特征主要是预后型（prognostic）而非效应修饰型，混进共享表示只会加方差；
   论文强调这仍只是 hypothesis，不是自动特征选择机制（⑥ Q23）。

**论文自己承认的局限**

1. 服务路径是拼装的：各模块可独立开关、可独立验证，但「调参负担与特征敏感性仍是当前共享表示模型的
   局限」（⑥ Q27）。
2. 商品 embedding 的冷启诊断只用了 1 个留出商品 + 采样 embedding，论文明确说这**不能**证明零样本
   上线有效（⑥ Q28）。
3. 线上 +7.20% 是**整条策略**的系统级结果：因果打分、探索策略、LP、withhold 规则相互耦合，论文明确
   指出不能把结果拆给单个组件（⑥ Q3）。**不要把它当成「LP 层单独值 7.20%」**。
4. **论文未讨论**：母婴 / 跨境电商场景、平台站内广告位与独立站承接之间的双边数据对齐问题；
   **论文未报告**任何工程成本、人力投入或求解器耗时量级。本卡在 ⑤ 自建 ROI 公式，并把必须由企业
   自测的参数单独标出。
5. registry 备注「无公开代码，需自研」→ 复现成本高于有开源实现的论文，落地前应先按 ③ 的模板做小规模
   自研验证，再决定是否上生产。

---

## ② 母婴出海应用案例

### 场景 1：月初把月度投放预算在「Amazon 站内广告位」与「站外种草 → 独立站」之间分配

- **业务问题**：母婴出海店铺（吸奶器、纸尿裤、奶瓶），投放主体是 Amazon 店铺，同时用站外内容种草
  把人引到独立站承接。**月初第 1 周**必须定下这个月的钱怎么分：站内 SP/SB/SD（品牌词防守位、
  竞品词位、优惠券 / Deal 位）多少，站外达人 / 内容位多少。现在的分法是看上周各位置 ROAS——站内
  品牌词 ROAS 常年最高，于是预算年年往品牌词堆；但搜品牌词的人本来就要买，而且站外种草带来的
  自然搜索还会被记到站内头上，钱实际投在了没有增量的地方（论文的离线单轮实验里，增量打分方法的平均收益高于倾向打分方法，⑥ Q38）。要换成按增量分，并且必须同时满足三类
  硬约束：① 站内各广告位的可达曝光 / 受众容量有上限；② 独立站落地页、客服排班、支付风控决定日
  承接上限；③ 站外种草到独立站成交的账期与退款率使当月可动用现金有上限。
- **数据要求**：站内与独立站两侧的设备 / 受众层级日志，**日粒度**：曝光、点击、加购、成交、金额、
  成本；必须含未触达对照（站内：品牌词 holdback 或 PSA 公益位；站外：按达人 / 内容位分组随机化）；
  历史长度**至少覆盖 1 个完整旺季周期**（Q4 黑五 + 春季），因为母婴品类有宝宝月龄生命周期，
  窗口太短会把生命周期效应误判成投放效果；两端身份对齐靠 邮箱 / 手机号 + UTM 与落地页参数，
  成交必须能区分站内单与独立站单。训练样本按论文 §5.1 的三段不重叠口径自己构造：上下文窗只看
  处理前信息，处理与结果取自其后的不重叠窗口（⑥ Q11/Q12）。
- **数据可得性**：`部分可得（需补充 X）`。独立站侧（Shopify + GA4 + 服务端埋点）数据可得；
  **Amazon 侧只有聚合广告报表，拿不到受众级随机化**，需要补充「品牌词 holdback 或地理 / 时间
  随机化」的实验设计。若两端都做不出对照，则本卡退化为「硬约束下的预算与容量诊断」，
  **不得声称增量因果**。
- **预期产出**：（a）一张「人群层 × 站内广告位 / 站外内容位」的预算分配表；（b）每条约束的影子
  价格 λ 与敏感性曲线，直接回答「这个月是站内曝光满了、独立站接不住、还是现金不够」；
  （c）用交付概率 p 折算的期望交付量（分配 ≠ 交付），对应独立站的日承接负载。
- **业务价值**：见 ⑤ 的 ROI 公式。本卡的钱来自两处：把预算从零增量位置挪走（增量口径替代归因口径），
  以及识别「买量超出承接能力」的过投部分。

### 场景 2：旺季前六周——加买量，还是先扩承接？

- **业务问题**：黑五前六周，站外种草要不要加预算？加流量会撞独立站日承接上限（落地页、在线客服、
  支付风控），转化反而掉；不加就错过「种草 → 独立站成交」的时滞窗口。团队过去靠「先加再调」。
- **数据要求与可得性**：`部分可得（需补充 X）`。轮次粒度（周 / 日）的**实际花费 D_t** 与 **LP 预测
  花费 A_t** 两侧都要有，用于估计实现比率 r̂_t（实际花费一侧可得，LP 预测一侧要先落地 ③ 的求解器）；
  交付概率需要独立站侧「进入落地页 → 下单」的链路埋点，以及按成员生命周期分段做 isotonic 校准
  （论文 §5.2 的做法）。
- **做法**：把独立站承接写成**区间约束**，并用交付概率把「分配」折成「期望交付」；再叠一层预算节奏
  控制器，把「计划花费 vs 实际花费」的偏差换算成下一轮的分配上限，避免计划与实际两张皮。
- **预期产出**：旺季六周的每周分配上限序列 + 承接能力缺口预警——当独立站承接约束的 λ 持续高于
  其他约束时，正确动作是**先加客服排班 / 加落地页承载**，而不是继续买流量。
- **业务价值**：把「过投」与「承接不足」分开定价。即使暂时不引入因果模型，**单是节奏控制器**就能
  让实际花费贴近承诺预算——论文引入它的原因正是「承诺预算的季度节奏」与「实验臂投递匹配」两个
  业务需求（⑥ Q14/Q13）。

---

## ③ 代码模板

- 依赖：**只需要 numpy**（不调用 OR-Tools / scipy，用「平滑对偶分解 + 影子价格上的 Nesterov 加速
  上升」自己解 LP 松弛），求解与投影都按用户解耦，因而能过 K1 的断网执行校验。
- 结构：合成示例数据 → 按用户投影 Π_{C_u} → 对偶求解器（输出分配与影子价格 λ）→ 可行性修复 →
  交付概率折算 → 预算节奏 + 实现比率反馈控制器 → 7 个断言测试 → 业务演示。
- 与论文的对应：`primal_from_lambda` 实现论文式 4；`solve_dual_allocation` 用论文 §2.3.1 给出的
  梯度 A x*_γ(λ) − b；`step_pacing` 实现论文式 14，`simulate_pacing` 实现论文式 13。
- ⚠️ 代码内所有数据均为**合成示例数据**（`make_synthetic_round`），不是论文数据，也不代表任何企业的
  真实预算。

```python
"""
增量优先的全局约束预算分配（causal + dual decomposition）
论文：2608.10182 §2.3 —— "Large-scale Allocation with Constraints" / §2.3.1 Dual Decomposition
业务映射：把一笔月度投放预算在「平台站内广告位」与「站外种草 → 独立站承接」之间分配，
        硬约束 = ①站内广告位可达曝光上限 ②独立站承接容量 ③当月可动用现金。

实现要点：不调用通用 LP 求解器，直接用「平滑对偶分解 + 对偶变量（影子价格）上的
Nesterov 加速上升」求解；每轮只需求解按用户解耦的投影子问题，因此只需 numpy。
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# 1. 合成示例数据（非论文数据；仅为让模板可运行）
# ---------------------------------------------------------------------------
ITEM_NAMES = ["站内SP品牌词位", "站内SP竞品词位", "站内Deal/优惠券位", "站外种草内容位"]
GUARD_NAMES = ["站内广告位曝光容量", "独立站承接容量", "当月现金"]


def make_synthetic_round(n_users: int = 240, n_items: int = 4, n_guard: int = 3,
                         seed: int = 7) -> dict:
    """生成一轮（一次分配周期）的合成数据。

    tau[u, i]  该用户在该广告位/内容位上的**增量**目标（可为负 = 本来就会买）
    A[k, u, i] 资源消耗系数（k 类全局约束，必须非负：曝光量 / 承接量 / 现金）
    b[k]       第 k 类全局约束的上限
    p[u, i]    交付概率（论文 (12) 式：assignment 不等于 delivery）
    fcap[u]    单用户频次上限（论文中的 C_fcap）
    """
    rng = np.random.default_rng(seed)
    # 「高活跃用户天然响应高」—— 预测模型会把他们排在前面，但他们本来就会买
    organic = rng.normal(0.0, 1.0, size=(n_users, n_items))
    warmth = rng.random((n_users, 1))
    tau = np.round(0.6 * organic - 1.4 * warmth + 0.40, 4)

    A = np.stack([
        rng.uniform(0.5, 1.5, size=(n_users, n_items)),   # 曝光容量消耗
        rng.uniform(0.2, 1.0, size=(n_users, n_items)),   # 独立站承接消耗
        rng.uniform(0.05, 0.30, size=(n_users, n_items)),  # 现金消耗
    ])[:n_guard]
    b = np.array([120.0, 70.0, 22.0])[:n_guard]
    p = np.clip(rng.beta(5.0, 2.0, size=(n_users, n_items)), 0.05, 0.99)
    fcap = np.full(n_users, 1.5)
    return {"tau": tau, "A": A, "b": b, "p": p, "fcap": fcap}


# ---------------------------------------------------------------------------
# 2. 按用户解耦的投影（论文 (4) 式中的 Π_{C_u}）
# ---------------------------------------------------------------------------
def project_per_user(v: np.ndarray, cap: float, n_bisect: int = 60) -> np.ndarray:
    """Π_{C_u}：把打分向量投影到 {0 <= x_i <= 1, sum_i x_i <= cap}。

    先裁剪到 [0,1]；若和超过频次上限，则对偶变量 mu >= 0 二分水位：
    x_i = clip(v_i - mu, 0, 1)，使 sum_i x_i = cap。排序实现是 O(I log I)，
    这里用二分（同样只依赖 numpy，且对 I 很小的一轮足够）。
    """
    x = np.clip(v, 0.0, 1.0)
    if cap <= 0.0:
        return np.zeros_like(v)
    if x.sum() <= cap:
        return x
    lo, hi = 0.0, float(np.max(v)) + 1.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        s = np.clip(v - mid, 0.0, 1.0).sum()
        if s > cap:
            lo = mid
        else:
            hi = mid
    return np.clip(v - hi, 0.0, 1.0)


def project_box_capped(v: np.ndarray, fcap: np.ndarray,
                       n_bisect: int = 50) -> np.ndarray:
    """project_per_user 的全用户向量化版本，逐用户结果一致（见 test_*）。"""
    v = np.atleast_2d(v)
    x = np.clip(v, 0.0, 1.0)
    over = x.sum(axis=1) > fcap
    if not over.any():
        return x
    lo = np.zeros(v.shape[0])
    hi = np.maximum(v.max(axis=1), 0.0) + 1.0     # mu 足够大时整行归零 → 必可行
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        s = np.clip(v - mid[:, None], 0.0, 1.0).sum(axis=1)
        too_big = s > fcap
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    return np.clip(v - hi[:, None], 0.0, 1.0)


def primal_from_lambda(tau: np.ndarray, A: np.ndarray, lam: np.ndarray,
                       fcap: np.ndarray, gamma: float) -> np.ndarray:
    """x*_{gamma,u}(lam) = Π_{C_u}[ (tau_u - A_u^T lam) / gamma ]。

    论文 (4) 式写作 Π_{C_u}[ -(A_u^T lam + c_u)/gamma ]（c 为最小化形式的目标系数）；
    本实现是最大化形式（c_u = -tau_u），两式等价。
    """
    adj = np.tensordot(A, lam, axes=(0, 0))      # (U, I) = A_u^T lam
    return project_box_capped((tau - adj) / gamma, fcap)


def dual_objective(tau: np.ndarray, A: np.ndarray, b: np.ndarray,
                   lam: np.ndarray, x: np.ndarray, gamma: float) -> float:
    """平滑对偶目标（本实现要**最大化**的那个）：h(lam) = -tau.x + lam.(A x - b)
    + gamma/2 * ||x||^2，取 x = x*_gamma(lam)。由 Danskin 定理，
    grad h(lam) = A x*_gamma(lam) - b —— 正是论文 §2.3.1 给出的梯度。
    """
    used = A.reshape(A.shape[0], -1) @ x.ravel()
    return float(-(tau * x).sum() + (lam * (used - b)).sum()
                 + 0.5 * gamma * float((x ** 2).sum()))


def solve_dual_allocation(tau: np.ndarray, A: np.ndarray, b: np.ndarray,
                          fcap: np.ndarray, gamma: float = 0.05,
                          n_iter: int = 400, step: float | None = None,
                          lam0: np.ndarray | None = None) -> dict:
    """对偶分解求解器：对偶变量上做 Nesterov 加速投影上升，lam >= 0。

    上升方向取论文 §2.3.1 写出的 gradient  A x*_gamma(lam) - b：
    某类约束被突破（用量 > 上限）时该分量为正 → 影子价格 lambda 抬高，
    下一轮分配自动绕开这条瓶颈。步长取 gamma / ||A||_2^2（Lipschitz 上界）。
    """
    K = A.shape[0]
    Aflat = A.reshape(K, -1)
    if step is None:
        spec = float(np.linalg.norm(Aflat, 2))
        step = gamma / max(spec ** 2, 1e-12)
    lam = np.zeros(K) if lam0 is None else np.asarray(lam0, dtype=float).copy()
    y = lam.copy()
    t = 1.0
    for _ in range(n_iter):
        x = primal_from_lambda(tau, A, y, fcap, gamma)
        grad = Aflat @ x.ravel() - b
        lam_new = np.maximum(0.0, y + step * grad)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = lam_new + ((t - 1.0) / t_new) * (lam_new - lam)
        lam, t = lam_new, t_new
    x = primal_from_lambda(tau, A, lam, fcap, gamma)
    return {"x": x, "lam": lam, "used": Aflat @ x.ravel(),
            "objective": float((tau * x).sum()), "step": step}


def repair_feasible(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """可行性修复：若某类约束被超出，按最大超出比例整体收缩。

    仅在 A >= 0 时成立（曝光/承接/现金这类「资源消耗」型约束）；
    若守卫指标可能为负，需改用逐约束的列生成式修复。
    """
    used = A.reshape(A.shape[0], -1) @ x.ravel()
    alpha = 1.0
    for k in range(len(b)):
        if used[k] > b[k] > 0:
            alpha = min(alpha, float(b[k] / used[k]))
    return x * alpha if alpha < 1.0 else x


def expected_delivery(x: np.ndarray, p: np.ndarray) -> float:
    """期望交付量（论文 (12)(5.2) 节：用 p_{u,i} 把「分配」折算成「期望交付」）。"""
    return float((p * x).sum())


# ---------------------------------------------------------------------------
# 3. 预算节奏 + 交付比率反馈控制器（论文 (13)(14) 式）
# ---------------------------------------------------------------------------
def step_pacing(cost_target: float, error: float, r_hat: float, kappa: float = 0.6,
                c_min: float = 0.0, c_max: float = 1e9, eps: float = 1e-9) -> float:
    """C_{t+1} = Π_[Cmin,Cmax]( C_t + kappa * e_t / max(r_hat_t, eps) )。

    error = S*_t - S_t（应为的累计花费 - 实际累计花费）。落后（error>0）→ 抬高
    下一轮的成本目标；超前（error<0）→ 压低。r_hat 是「LP 预测花费 → 实际花费」的
    平滑实现比率，除以它是为了把「目标差」换算成「分配量差」。
    """
    nxt = cost_target + kappa * error / max(r_hat, eps)
    return float(min(max(nxt, c_min), c_max))


def simulate_pacing(budget: float, n_rounds: int, realized_ratio: float = 0.8,
                    alpha: float = 0.5, kappa: float = 0.6, c_min: float = 0.0,
                    c_max: float = 1e9) -> dict:
    """按 (13)(14) 式跑一遍预算节奏：外层 S*_t = B q(t/H)，内层比率平滑。"""
    targets, spends, r_hat = [], [], 1.0
    cumulative = 0.0
    cost_target = budget / n_rounds
    for t in range(1, n_rounds + 1):
        should = budget * (t / n_rounds)                 # q(t/H) 取线性
        error = should - cumulative
        cost_target = step_pacing(cost_target, error, r_hat, kappa, c_min, c_max)
        assigned = cost_target
        realized = assigned * realized_ratio             # 实际只投出去一部分
        r = realized / assigned
        r_hat = alpha * r + (1.0 - alpha) * r_hat
        cumulative += realized
        targets.append(cost_target)
        spends.append(cumulative)
    return {"targets": targets, "cumulative_spend": spends, "r_hat": r_hat}
```

```python
# ---------------------------------------------------------------------------
# 4. 测试
# ---------------------------------------------------------------------------
def test_projection_respects_box_and_cap():
    rng = np.random.default_rng(0)
    v = rng.normal(0.0, 2.0, size=50)
    x = project_per_user(v, cap=3.0)
    assert np.all(x >= -1e-12) and np.all(x <= 1.0 + 1e-12)
    assert x.sum() <= 3.0 + 1e-9
    # 不超上限时投影应等于「裁剪到 [0,1]」
    y = project_per_user(v, cap=1e6)
    assert np.allclose(y, np.clip(v, 0.0, 1.0))
    # 一维与向量化版本必须一致
    V = rng.normal(0.0, 2.0, size=(20, 4))
    caps = np.full(20, 1.5)
    X = project_box_capped(V, caps)
    for u in range(20):
        assert np.allclose(X[u], project_per_user(V[u], caps[u]), atol=1e-6)


def test_allocation_is_feasible():
    data = make_synthetic_round()
    res = solve_dual_allocation(data["tau"], data["A"], data["b"], data["fcap"])
    x = repair_feasible(res["x"], data["A"], data["b"])
    used = data["A"].reshape(data["A"].shape[0], -1) @ x.ravel()
    assert np.all(used <= data["b"] * (1.0 + 1e-9))
    assert np.all(x >= -1e-12) and np.all(x <= 1.0 + 1e-12)
    assert np.all(x.sum(axis=1) <= data["fcap"] + 1e-9)
    assert res["lam"].shape == (data["A"].shape[0],)
    assert np.all(res["lam"] >= 0.0)


def test_dual_beats_naive_baselines():
    data = make_synthetic_round()
    tau, A, b, fcap = data["tau"], data["A"], data["b"], data["fcap"]
    res = solve_dual_allocation(tau, A, b, fcap)
    best = float((tau * repair_feasible(res["x"], A, b)).sum())
    # 基线 1：全量铺开（不可行，收缩到可行）
    flat = repair_feasible(np.ones_like(tau), A, b)
    # 基线 2：按增量打分贪心（忽略全局约束，收缩到可行）
    greedy = repair_feasible((tau > 0.0).astype(float), A, b)
    assert best >= float((tau * flat).sum()) + 1e-6
    assert best >= float((tau * greedy).sum()) + 1e-6


def test_tighter_capacity_raises_shadow_price():
    """收紧某类约束上限 → 它的影子价格必须上升（这才有业务解释力）。"""
    data = make_synthetic_round()
    tau, A, fcap = data["tau"], data["A"], data["fcap"]
    loose = solve_dual_allocation(tau, A, np.array([120.0, 70.0, 22.0]), fcap)
    tight = solve_dual_allocation(tau, A, np.array([60.0, 70.0, 22.0]), fcap)
    assert tight["lam"][0] > loose["lam"][0] + 1e-2
    assert tight["lam"][0] > tight["lam"][1]      # 被收紧的约束成为第一瓶颈


def test_dual_ascent_improves_the_dual_objective():
    """论文 §2.3.1 的加速上升确实在抬高对偶目标（并给出可行性修复前的违约量）。"""
    data = make_synthetic_round()
    tau, A, b, fcap = data["tau"], data["A"], data["b"], data["fcap"]
    lam0 = np.zeros(A.shape[0])
    x0 = primal_from_lambda(tau, A, lam0, fcap, 0.05)
    h0 = dual_objective(tau, A, b, lam0, x0, 0.05)
    res = solve_dual_allocation(tau, A, b, fcap)
    h1 = dual_objective(tau, A, b, res["lam"], res["x"], 0.05)
    assert h1 > h0
    # λ=0 时（完全不考虑约束）用量必然超出上限；λ 收敛后违约量应大幅收窄
    viol0 = float(np.max((A.reshape(A.shape[0], -1) @ x0.ravel()) - b))
    viol1 = float(np.max(res["used"] - b))
    assert viol0 > 0.0 and viol1 < 0.02 * viol0


def test_pacing_controller_direction_and_bounds():
    assert step_pacing(10.0, error=5.0, r_hat=0.8) > 10.0     # 落后 → 抬目标
    assert step_pacing(10.0, error=-5.0, r_hat=0.8) < 10.0    # 超前 → 压目标
    assert step_pacing(0.5, error=-100.0, r_hat=1.0, c_min=1.0) == 1.0
    assert step_pacing(10.0, error=1e6, r_hat=1.0, c_max=99.0) == 99.0
    sim = simulate_pacing(budget=1000.0, n_rounds=40, realized_ratio=0.8)
    assert len(sim["targets"]) == 40
    assert abs(sim["r_hat"] - 0.8) < 1e-6                      # 平滑后收敛到真实实现比率
    # 不做节奏控制时只能花掉 budget * 0.8；加上反馈控制后应逼近全额预算
    assert sim["cumulative_spend"][-1] > 1000.0 * 0.8 * 1.15
    assert abs(sim["cumulative_spend"][-1] - 1000.0) < 150.0


def test_expected_delivery_uses_delivery_probability():
    data = make_synthetic_round()
    x = np.ones_like(data["tau"]) * 0.5
    exp = expected_delivery(x, data["p"])
    assert abs(exp - float((data["p"] * x).sum())) < 1e-9
    assert exp < float(x.sum())                                # 交付概率 < 1 → 期望交付更少


# ---------------------------------------------------------------------------
# 5. 业务演示
# ---------------------------------------------------------------------------
def _business_demo() -> None:
    data = make_synthetic_round()
    tau, A, b, fcap, p = (data["tau"], data["A"], data["b"],
                          data["fcap"], data["p"])
    res = solve_dual_allocation(tau, A, b, fcap)
    x = repair_feasible(res["x"], A, b)

    print("== 一轮约束分配（合成示例数据，非论文数据）==")
    for i, name in enumerate(ITEM_NAMES[:A.shape[2]]):
        alloc = float(x[:, i].sum())
        print(f"  {name:16s} 分配量={alloc:8.1f}  期望交付={expected_delivery(x[:, i], p[:, i]):8.1f}")

    print("\n== 约束使用率与影子价格 lambda ==")
    used = A.reshape(A.shape[0], -1) @ x.ravel()
    for k, name in enumerate(GUARD_NAMES[:A.shape[0]]):
        print(f"  {name:16s} 使用 {used[k]:7.2f}/{b[k]:7.2f}   "
              f"lambda={res['lam'][k]:.4f}")

    print(f"\n总增量目标 = {float((tau * x).sum()):.3f}"
          f"  |  期望总交付 = {expected_delivery(x, p):.1f}")

    sim = simulate_pacing(budget=1000.0, n_rounds=40, realized_ratio=0.8)
    print("\n== 预算节奏 + 交付比率反馈（(13)(14) 式）==")
    print(f"  末轮成本目标 = {sim['targets'][-1]:.2f}"
          f"  |  累计实际花费 = {sim['cumulative_spend'][-1]:.2f}"
          f"  |  平滑实现比率 r_hat = {sim['r_hat']:.3f}")


if __name__ == "__main__":
    _business_demo()
```

---

## ④ 技能关联

- **前置｜`Skill-Uplift-Modeling.md`**（因果推断域）：提供 CATE / ITE 估计（S/T/X-Learner、DML）。
  数据流：本卡的 τ̃ 就来自这类估计量；论文用 DragonNet 变体替代多阶段 pipeline，两者可在同一份数据上
  用 AUUC 互验（论文 §4.4 与 §A.3 用的正是 CausalML 的 AUUC 口径）。
- **前置 / 对照｜`Skill-Ad-Attribution-Modeling.md`**（同目录）：Shapley / Markov 归因回答「哪些触点
  被记功」。逻辑依赖：归因输出是**相关性**权重，且平台归因天然高估，只能当先验或对照组，
  **不能**直接当 τ̃ 输入本卡的求解器（这正是 ①b 失败模式 5 的近亲）；正确组合是「归因给渠道权重 →
  增量实验给校准系数」。
- **同域对照｜`Skill-ROAS-Budget-Optimization.md`**（同目录）：那张卡在花费-收入曲线上做边际 ROAS
  均衡，输入是历史 ROAS；本卡把输入换成因果增量、把约束显式写进 LP，并把「各渠道边际 ROAS 相等」
  升级为「各约束的影子价格决定配置」。同一条曲线，一个用斜率当边际收益，一个用 λ 当资源价格。
- **延伸｜`Skill-Multi-Armed-Bandit.md`**（A_B 实验域）：本卡的探索层就是 neural Thompson sampling
  （用 last-layer 线性化拉普拉斯近似拿后验，论文 §2.2.1）；若暂时不上神经网络，退化为该卡的标准
  Thompson Sampling 仍然可用——论文也确认无共享约束时策略等价于 TS。实现细节可再参考
  `Skill-Thompson-Sampling-MAB.md`。
- **组合｜`Skill-Marketing-Mix-Modeling.md`**（营销投放分析域）：MMM 给的是渠道级、周粒度的宏观弹性，
  粒度对不上个体分配，但它的渠道弹性可以作为本卡约束影子价格的**先验**，用来给对偶变量做热启动
  （论文 §2.3.1 的热启动在稳定输入分布下能拿到当前最优解的 99% 以上，⑥ Q8）。

---

## ⑤ 商业价值评估

**ROI 公式**（本次不代入任何未经论文或企业数据支持的数字）：
ROI = (Δ增量毛利 − C_sys) / C_sys，其中 Δ增量毛利 = B × Δλ_incr × m。

| 参数 | 含义 | 来源 |
|---|---|---|
| `B` | 月度投放预算（站内 + 站外合计） | 企业自有财务口径 |
| `Δλ_incr` | 本策略相对现状的主指标增量率 | **必须企业自测**。论文给出的唯一跨域参考点是 LinkedIn Feed 八周线上 A/B 的端到端策略 +7.20%（p=0.041，95% CI [0.31%, 14.09%]，⑥ Q2/Q3）——注意它是**长期价值指标**而非母婴跨境 GMV / ROAS，且是整条策略的系统级结果（⑥ Q3），**只能当上界参考，不能当预测** |
| `m` | 增量 GMV 的贡献毛利率（扣平台佣金、头程 / FBA 履约、退货、支付等变动成本） | 企业自有财务口径 |
| `C_sys` | 一次性建设 + 运行成本：数据构造与因果模型训练、对偶求解器、实验脚手架 | **论文未报告任何成本量级**，需企业自估。注意论文 §5.4 明确指出，要在规模上跑通对照实验，需要把数百个 campaign 自动配置成预算拆分变体、冻结受众快照、审计 segment 血缘——人工不可行，这部分是本卡的主要实施成本 |

**保守结论**：把论文的 +7.20% 当作**上界**代入，`Δ增量毛利 = B × 7.20% × m`；由于母婴跨境的 `m`
明显低于 LinkedIn 这类自有流量场（要扣平台佣金、头程、履约与退货），实际 `Δλ_incr` 也需用企业自己的
holdback / 地理实验测量，因此**本卡不给收益金额结论**。

- **实施难度**：⭐⭐⭐⭐☆—— 因果模型与实验脚手架是主要成本；但 ③ 的求解器与节奏控制器
  可以独立先落地（只依赖 numpy），把难度拆成两段。
- **优先级**：⭐⭐⭐⭐☆—— 不是「立刻全量上」的卡，而是「旺季预算规划前必须先有影子价格视角」
  的卡。若企业当前连 holdback 都没有，优先级降到 3 星，先补实验能力。
- **评估依据**：收益侧的不确定性远大于工程侧（论文未给母婴场景任何数字），所以本卡的价值主张是
  **决策口径的修正**（增量替代归因、影子价格替代平均 ROAS），而不是承诺一个提升幅度。

---

## ⑥ 原文引用

**A. 线上结果与可外推性（含 registry 记录的 +7.20% 核实）**

> 原文："The end-to-end treatment policy delivered a statistically significant $+7.20\%$ lift in the primary long-term-value metric, demonstrating the feasibility of production-scale causal optimization under business constraints."
> 出处：2608.10182 §Abstract｜Q1

> 原文："Measured against long-term-value metrics over an eight-week period, the treatment arm achieved a statistically significant $+7.20\%$ lift ($p=0.041$, 95% CI: $[0.31\%,14.09\%]$)."
> 出处：2608.10182 §5.5 Results｜Q2

> 原文："The online result therefore measures the system-level impact of the complete production policy, while the offline studies examine individual mechanisms under controlled settings."
> 出处：2608.10182 §5.5 Results｜Q3

> 原文："Members were randomly assigned 50/50 to the two experiment arms."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q24

> 原文："We also distill production lessons on causal training-data construction and cost and delivery control, which were critical to successful deployment."
> 出处：2608.10182 §Abstract｜Q31

**B. 方法：因果估计 / 探索 / 对偶 LP**

> 原文："Incremental optimization requires estimating user-level treatment effects that quantify the expected lift from a targeting or recommendation action."
> 出处：2608.10182 §2.1 Incremental Modeling｜Q36

> 原文："Two standard assumptions identify $\tau(X)$ from observational data: unconfoundedness, $\{Y(0),Y(1)\}\perp\!\!\!\perp T\mid X$, and overlap, $0<e(X)<1$ for all $X$, where $e(X)=P(T=1\mid X)$ is the propensity score."
> 出处：2608.10182 §2.1 Incremental Modeling｜Q9

> 原文："Thus exploration guarantees positivity conditional on the feasible action."
> 出处：2608.10182 §2.2 Neural Bandit Exploration｜Q10

> 原文："We approximate the posterior over network parameters by a Gaussian centered at the MAP solution $\hat{\theta}_{\mathrm{MAP}}$ via the linearized Laplace approximation (LLA)"
> 出处：2608.10182 §2.2.1 Neural Thompson sampling via Laplace approximation｜Q37

> 原文："In the absence of shared allocation constraints, the resulting policy is equivalent to Thompson sampling: for each user, it selects the feasible action that maximizes the sampled incremental reward."
> 出处：2608.10182 §2.3 Large-scale Allocation with Constraints｜Q33

> 原文："Suppressing the round index, we relax $x_{u,i,t}$ to represent an action probability and solve the resulting large-scale problem using a smoothed dual-decomposition method (Basu et al., 2020)."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q5

> 原文："The solver then maximizes $g_{\gamma}$ over the $K$-dimensional dual with Nesterov-accelerated ascent, giving per-iteration cost linear in $|\mathcal{U}|\cdot|\mathcal{I}|$ versus $O((|\mathcal{U}||\mathcal{I}|)^{3.5})$ for interior-point methods."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q6

> 原文："The regularization $\gamma$ is picked as the largest value satisfying $\frac{\gamma\,\hat{x}^{T}\hat{x}}{2\,|c^{T}\hat{x}|}<10^{-3}$, so the ridge perturbation contributes $<0.1\%$ of the objective and the perturbed optimum is practically indistinguishable from the true LP optimum; when $A$ is ill-conditioned across constraint scales, we apply Jacobi row preconditioning."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q7

> 原文："At production scale, each round contains tens of millions of users and hundreds of items, so its batch has $|\mathcal{U}|\times|\mathcal{I}|$ variables and is intractable for general-purpose solvers."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q4

> 原文："In steady state, we warm-start the dual from the previous period’s $\lambda^{*}$, which under stable input distributions (KS-tested) achieves over 99% of the current optimum and also serves as an SLA fallback when the solver does not converge in time."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q8

> 原文："The framework also supports sequential context and multi-outcome, attribute-conditioned scoring through a Transformer encoder and outcome embeddings."
> 出处：2608.10182 §Abstract｜Q32

**C. 离线验证的规模与结果**

> 原文："We utilize the random-policy subset, synthetically mapping the original 34 products that were recommended to $>400$K users to 5 distinct actions that are relevant to the production incrementality use case: recommendation to one of four business lines or no-recommendation."
> 出处：2608.10182 §4.1 Dataset｜Q20

> 原文："The no-recommendation action is a key difference between incremental and non-incremental targeting."
> 出处：2608.10182 §4.1 Dataset｜Q34

> 原文："Finally, we assign a cost of $0.1 to each recommendation/targeting action to capture operational and bidding costs."
> 出处：2608.10182 §4.1 Dataset｜Q21

> 原文："We solve the offline simulation LP with Google OR-Tools, which is tractable at this scale (${\sim}400$K members, 5 actions) and convenient for reproduction; at full production scale we use the dual-decomposition method described in Section 2.3 instead."
> 出处：2608.10182 §4.2 Setup｜Q16

> 原文："We then simulate deployment over $T=200$ rounds using the prediction set as the environment: at each round, each method selects actions for the current batch, observes the realized rewards from its own recommendations, updates its training data accordingly, and incrementally updates the model before the next round."
> 出处：2608.10182 §4.2 Setup｜Q17

> 原文："Figure 2. Average cumulative reward and 95% confidence intervals after 200 rounds of feedback. The confidence intervals are calculated from 30 simulation runs."
> 出处：2608.10182 §4.3.2 Multi-turn evaluation（Figure 2 图注）｜Q18

> 原文："This reflects the short-term cost of exploration in exchange for long-term gains: after roughly 50 model updates, the Bandit Incremental Model begins to outperform both greedy variants."
> 出处：2608.10182 §4.3.2 Multi-turn evaluation｜Q19

> 原文："We tested 8 configurations on a fixed train/validation snapshot, each repeated 5 times with common hyperparameters."
> 出处：2608.10182 §4.4 Ablation Study｜Q15

> 原文："As shown in Table 2, incremental scores lead to higher rewards compared to propensity scores."
> 出处：2608.10182 §4.3.1 Single-turn evaluation｜Q38

> 原文："We also provide the corresponding send volumes in Table 3, where it can be seen that incremental targeting leads to a higher percentage of no-recommendations as the engine is able to identify members likely to convert organically."
> 出处：2608.10182 §4.3.1 Single-turn evaluation｜Q39

> 原文："Removing dense features is neutral or beneficial for uplift AUUC despite reducing outcome AUROC from 0.857 to 0.826 and minimally affecting treatment AUROC."
> 出处：2608.10182 §A.3 Uplift Results｜Q22

**D. 生产落地：数据构造、投递控制、实验脚手架**

> 原文："For each member, production samples $D=R-(W_{C}+W_{T}+W_{D})+U$, where $U\sim\operatorname{Uniform}\{0,\ldots,W_{D}-1\}$, $W_{D}=90$ days, $W_{T}=7$ days, and $W_{C}=30$ days. Thus $D\in[R-127,R-38]$."
> 出处：2608.10182 §5.1 Training-Data Construction for Causal Estimation｜Q11

> 原文："We set $T=1$ when at least one qualifying email send, on-platform impression, or video view occurs in $[D,D+7)$. We set $Y=1$ when the corresponding business-line or product-family conversion occurs in $[D+7,R]$, and $Y=0$ otherwise."
> 出处：2608.10182 §5.1 Training-Data Construction for Causal Estimation｜Q12

> 原文："Randomizing $D$ avoids a last-touch label, captures long-term action effects, and preserves variable-length histories."
> 出处：2608.10182 §5.1 Training-Data Construction for Causal Estimation｜Q35

> 原文："Let $B$ be the committed budget, $S_{t}$ the cumulative realized spend, and $q(t/H)$ the desired cumulative pacing curve over a horizon $H$, with $q(0)=0$ and $q(1)=1$."
> 出处：2608.10182 §5.2 Cost and Delivery Control｜Q14

> 原文："An early A/B test run without the above controls showed the treatment arm under-delivering: the causal policy withholds and reallocates sends, lowering treatment impressions relative to control. A raw total-bookings comparison then penalizes the treatment for delivering less rather than for choosing worse, confounding policy quality with delivery volume. Constraining treatment to the BAU impression and cost envelope equalizes delivery across arms and restores a clean, like-for-like read."
> 出处：2608.10182 §5.2 Cost and Delivery Control｜Q13

> 原文："A meaningful policy difference is that the treatment arm can withhold a send whenever the predicted incremental value is negative or no feasible positive-incremental option exists."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q30

> 原文："First, when out-of-scope campaigns are suppressed via an exclusion segment defined by dynamic criteria (company, locale, activity), members drift across arms in a way correlated with platform activity, contaminating the intent-to-treat contrast."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q25

> 原文："Second, when arms are assembled from shared segment definitions, a control audience can silently inherit a treatment send decision through a reused sub-segment."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q26

**E. 论文自承局限**

> 原文："Although the complete research architecture contains several losses and optional modules, the serving path is modular rather than a jointly tuned monolith: outcome embeddings are used only for multi-product scoring, LLA is applied after supervised training to the last layer, and the LP consumes exported scores independently of model training. This separation lets each module be disabled or validated without retraining the rest of the decision pipeline, while the remaining tuning burden and feature sensitivity are limitations of the current shared-representation model."
> 出处：2608.10182 §3.4 Complete Loss Function｜Q27

> 原文："Because this diagnostic uses one held-out product and sampled embeddings rather than prospective real launches, it does not establish zero-shot production effectiveness."
> 出处：2608.10182 §A.4 Outcome-Embedding Diagnostic｜Q28

> 原文："This remains a hypothesis rather than an automated feature-selection mechanism; architectures such as FlexTENet (Curth and van der Schaar, 2021) could explicitly separate the two subspaces."
> 出处：2608.10182 §A.3 Uplift Results｜Q23
