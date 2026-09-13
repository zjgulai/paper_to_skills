---
title: Skill-Multi-Warehouse-Allocation-LLM
module: 04-供应链
topic: 求解器验证的 MIP 公式生成与选择 —— 把运营的中文分仓要求变成多仓分仓模型（SD/SB/KI 三族 + 约束模块 + 罚函数松弛），并用求解器式校验守住「能跑 ≠ 算对」
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2606.29366
paper: Solver-Verified Formulation Generation and Selection for Multi-Warehouse Inventory Allocation Using Large Language Models
venue: arXiv preprint
venue_tier: preprint
evidence_grade: A
verified_by: verify_skill_code.py (K1 PASS) + quote_check.py (引文逐字核验 VERBATIM) + gate_check.py G1/G2/G3
verified_at: 2026-09-12
supersedes:
related: Skill-Safety-Stock-Replenishment.md, Skill-Demand-Forecasting-Supply-Chain.md, Skill-Multi-Echelon-Inventory.md, Skill-Two-Echelon-Inventory-DRL.md, Skill-Tool-Description-Audit.md, Skill-SQL-Agent-Text-to-SQL.md
---

# Skill-Multi-Warehouse-Allocation-LLM

> **venue 说明（R3 要求显式标注）**：全文首页页眉写着 `Journal: European Journal of Operational Research`
> （见 ⑥ 引用），但 `papers_registry.json` 里该论文的 `venue` 为空、`venue_tier: preprint`，
> 本次核验也**没有**在 EJOR 的正式录用信息中找到它。因此按 R3 一律记为 **arXiv preprint**，
> **不得**当作 EJOR 论文引用；页眉那行只说明作者当时的投稿目标。
>
> **registry 备注的核对结论（重要）**：registry 的 `decision_reason` 写「支持自然语言约束（'优先保 FBA 不断货'）」。
> 全文**从未出现 FBA、跨境、断货（stockout）等字样**（已 grep 全篇确认）。论文自己举的自然语言约束例子是
> Table 6 的 **`Minimum TID requirement` → `Enforce minimum TID`** 与 §4.4 的
> `ratio constraints, lower and upper allocation bounds, case-pack restrictions, group-level service requirements, and cardinality controls`。
> 也就是说：「优先保 FBA 不断货」是**业务侧的转译**，不是论文原话。本卡 ② 用运营原话做业务映射，
> 但在 ⑥ 只引论文自己的措辞，不把业务转译伪装成引文。
>
> **与 2607.25956 的关系**：registry 备注说两篇高度重叠。本卡**未**核验另一篇（按任务约束不抓取），
> 且**本文全文的参考文献列表里没有 2607.25956**（已 grep 确认），故本卡不对两篇关系作任何断言 ——
> 只把本文写成自包含的一页，合并时机由后续去重流程决定。

---

## ① 算法原理

**核心思想**：一批固定补货量 $R$ 要分给 $n$ 个仓（FBA 仓 + 自建海外仓），要求「各仓库存覆盖天数大致齐平」。
论文用 **TID**（Target Inventory Days＝可用库存 ÷ 预测日需求）当统一标尺，把「齐平」定义成
**各仓 TID 落在系统 TID 的带宽内**，再用**三个互补的 MIP 公式族**覆盖不同场景：
偏差最小化（SD）、软带宽（SB，越界量进目标函数）、背包式（KI，用 0-1 变量奖励「分给带宽内的仓」）。

**数学直觉**：系统级 TID $\tau_{\mathrm{all}}=\dfrac{\sum_k(I_k+x_k)}{\sum_k D_k}$ 在 $\sum_k x_k=R$ 下
**与决策无关**，于是目标库存量 $T_k=\tau_{\mathrm{all}}D_k$ 是常数，SD/SB 的目标对每个仓**可分离且凸** ——
这是本卡能用「逐台给边际代价最小的仓」这种贪心拿到**精确最优解**的原因（论文生产版用 SCIP 解 MILP，二者在 SD 族上等价）。
平衡率 Acc ＝ 带宽内各仓分到的量 ÷ 本批总量，是**量加权**的：一个已经齐平但没分到货的仓，对分子贡献为 0。

**关键假设**：单期、单 SKU、总量 $R$ 已由上游定死、$D_k>0$、所有业务规则都能写成线性/整数约束。

---

## ①b 反例与适用边界

### 什么时候**不要**让 LLM 生成这个模型

1. **提示里没有建模 schema 与求解器 API 契约时**。论文的高成功率是在
   `constrained PMC protocol`（把 MILP 建模与求解器 API 的 schema 放进训练/提示）下取得的（§6.2，见 ⑥ 引用）。
   缺了这个前提，那套可靠性数字**不能外推**到你自己的场景。
2. **约束类别没在训练集里出现过时**。论文对**未训练过的 5 个任务类别**做泛化测试，
   代码可执行是 5/5，但**方案合法只有 3/5**：`Fixed allocation ratio` 与 `Minimum TID requirement`
   两类都是「代码 ✓ / 方案 ✗」（Table 6，见 ⑥ 引用）。所以新约束一律**先人工建模 + 求解器校验**。
3. **超出单期分仓时**。论文自己把「多期与网络级规划」列为**未做**的开放方向（§8，见 ⑥ 引用）。
   要排多周滚动的补货节奏，本方法不覆盖。
4. **不可逆、事后无法补救的决策**（合规配额、清关/危险品限制、爆仓停收后的紧急改派）。
   即便代码可执行、求解器也报 feasible，也应保留人工闸门 —— 这一条是**本卡的判断**，不是论文结论。
5. **只想「省一个人」时**。本卡的真正交付物是**可复核的约束账**（被放宽了哪条、代价多少），
   不是「自动出一个数」。没有人工复核流程的组织，上了只会更快地算错。

### 已知失败模式：**能跑 ≠ 算对**（本卡最需要防的一条）

- 论文把负样本明确分成三类：`non-linear modeling hallucinations, wrong variable types, missing key constraints`
  （§5.1，见 ⑥ 引用）—— 它们**都能编译、多数能跑**，但模型本身是错的。
- 更硬的对照：代码语法准确率与执行成功率都接近满分（Table 4：SFT+KTO 的语法 100%、执行成功 99.9%），
  而**未训练任务的方案合法率只有 3/5**（§7.2）。两者放一起读，结论只有一个：
  **语法/执行通过率不能当正确性证据**。这就是本卡坚持「求解器式三项校验」的原因 ——
  可执行（explicit）、可行（守恒 $\sum x_k=R$ + 上下界）、解质量（平衡率）。
- **目标函数与业务指标不是一回事（本卡实测观察，非论文结论）**：SD 族最小化的是「按 $1/D_k$ 加权的偏差和」，
  而业务看的 Acc 是「带宽内量占比」。本卡代码里，某仓停收时 SD 族的最优解把余量全塞进需求最大的仓
  （加权偏差最便宜），平衡率随之塌掉；同一实例换 SB 族就回到全带内（两组读数见 ③ 的输出围栏）。
  所以**换目标函数＝换业务结论**，「公式族」不是学术装饰。

### 论文自己承认的局限

- 论文**没有独立的 Limitations 章节**，局限散见于 §6.2（结果受限于 constrained PMC protocol）与 §8 的三条开放方向。
- **未讨论**：跨境头程时效方差与清关不确定性、需求预测误差如何传导进分仓、仓储费/断货惩罚的金额标定、
  多期滚动与网络级调拨。这些在本卡里全部只能靠业务侧自建参数（见 ②/⑤），**不能**援引论文。

---

## ② 母婴出海应用案例

### 场景一：旺季前一批头程到港，FBA 两仓 + 海外仓两仓怎么分

- **业务问题**：吸奶器（双边电动款）一个批次到港，要在 **FBA-美西ONT8 / FBA-美东BWI2 /
  海外仓-长滩（服务独立站美西客群）/ 海外仓-新泽西（服务独立站美东客群）** 之间分配。
  FBA 侧断货会直接掉 Listing 权重与 BSR；海外仓侧压货按体积收仓储费。
  运营在群里给的要求长这样：**「宁可 A 仓多备也不能断货，B 仓仓储费太贵尽量别压货」** ——
  一句话里同时含下限意图与上限意图。人工拍脑袋的后果是：旺季前把量压给最省事的仓，
  结果一个仓断货、另一个仓付了整季仓储费。
- **数据要求**：SKU × 仓 × **日粒度**的在库 $I_k$；SKU × 仓的**预测日需求** $D_k$
  （≥8 周，含旺季抬升；FBA 侧与独立站侧分开预测再合并到仓）；本批总量 $R$；各仓上下限
  （FBA 补货限制/库容、海外仓托盘位或体积上限、安全库存下限）。
- **数据可得性**：`部分可得（需补 2 项）`。$I_k$ 与 $D_k$ 在 Amazon 卖家后台与独立站后台都能导出，
  母婴 SKU 的日粒度历史通常够 8 周以上；**缺口**是
  ① 海外仓的**可用托盘位/体积上限**（在 3PL 的 WMS 里，需接口或周报维护成一张表）；
  ② **FBA 入仓限制（补货限制/库容）** 会随季度变化，需运营每周更新。
- **预期产出**：整数分仓方案 + 逐仓 TID 与「偏紧/带内/压货」标记 + 平衡率读数 + **未满足/被放宽的约束清单**。
- **业务价值**：把「一批货怎么分」从经验变成可复核的账。可直接对照的业务量：
  `断货损失减少额 =（优化前后缺货天数差）× 该仓日均需求 × 单台毛利`；
  `滞销仓储费减少额 =（优化前后平均在库台数差 ÷ 箱规）× 单箱月仓储费 × 月数`。
  两个参数都能用贵司自己的财务与 3PL 报价单代入（见 ⑤）。

### 场景二：某仓临时停收/爆仓时的改派

- **业务问题**：**FBA-美东BWI2 突然停收**（爆仓或库容超限），或海外仓-新泽西因仓库改造停收一周。
  原方案里分给它的量必须在剩下的仓里重新分配。硬塞给 FBA-美西ONT8 会把美西 TID 顶到带外
  （压货 + 仓储费），而海外仓-长滩的容量又装不下全部。此时真正要回答的不是「解是多少」，
  而是**「该放松哪条约束、放松多少、代价是什么」**。
- **数据要求**：停收仓与停收窗口；各仓剩余可用容量；**在途头程的到港时间**（决定还来不来得及改派）；
  原批次方案与当时的约束表。
- **数据可得性**：`企业内可得`。停收信息来自平台与 3PL 的通知，属于运营日常；
  唯一要补的是把「停收」这件事**结构化**进约束表（本卡代码里就是一条 `max_qty = 0`）。
- **预期产出**：改派方案 + 被放宽的约束与其违规量 + 「必须人工裁决」清单（例如所有仓都装不下，
  只能减 $R$ 或走加急头程）。
- **业务价值**：把改派从「群里拍脑袋」变成「带代价标签的方案」；不可行时给出
  **放宽了哪条、代价多少**，而不是抛一个 infeasible 让人自己猜。

### 运营原话 → 约束模块的映射（本卡的核心落地动作）

| 运营原话（中文） | 结构化约束 | 论文里的模块 | 本卡代码字段 |
|---|---|---|---|
| 「宁可 A 仓多备也不能断货」 | $x_A \ge \alpha_A R$（比例下限） | OC1 | `min_ratio` |
| 「B 仓仓储费太贵尽量别压货」 | $x_B \le \beta_B R$（比例上限） | OC2 | `max_ratio` |
| 「这周 C 仓停收」 | $x_C \le 0$（数量上限的特例） | OC5 | `max_qty` |
| 「FBA 两个仓合计至少拿到保底份额」 | $\sum_{l\in\mathcal{S}_k} x_l \ge \eta_k R$（仓组最小份额） | OC8 | 需 MILP 求解器 |
| 「最多开三个仓，省得尾程拆单」 | $\sum_k s_k \le m$（开仓数上限） | OC9 | 需 MILP 求解器 |
| 「整箱发，别拆箱」 | $x_k = p_k q_k$（箱规整数倍） | OC7 | 需 MILP 求解器 |

> 上表的映射依据是论文 §4.4 与 Table 1（模块编号 OC1–OC9）；**论文没有给这些约束的任何业务话术**，
> 表中「运营原话」一列是贵司场景的转译。OC7/OC8/OC9 需要 0-1 变量或整数分解，本卡代码**未实现**，
> 真实部署请交给 MILP 求解器（论文生产版用 SCIP）。

---

## ③ 代码模板

> **定位**：这是业务化的**精简实现**，不是论文的忠实复现。三个必须说清的地方：
> ① 论文生产版用 **SCIP** 解 MILP（见 ⑥ 引用）；本卡为了「不 import 任何未安装的求解器」
> 自己实现了 **SD/SB 两族的精确解**（可分离凸 + 单守恒约束 + 箱式上下界 ⇒ 堆贪心即最优，
> 已用穷举在小算例上自证）；**带 0-1 变量的 KI 族本卡不支持，必须用真求解器**。
> ② 论文用 **Qwen3-8B** 后训练模型生成公式与代码（见 ⑥ 引用）；本卡不能调任何 LLM，
> 用一个小型**规则解析器**顶替「自然语言 → 约束」这一步，它**只**认中文里的三类片段。
> ③ 论文用 **LightGBM** 回归器给候选公式打分（见 ⑥ 引用）；本卡换成闭式岭回归 ——
> 它的职责只是「打分」，换什么回归器都不改变流程形状。

### 第一段：中文要求 → 结构化约束 → 整数上下界

```python
"""ORLA 业务模板：把运营的中文分仓要求变成 MIP 约束，再用「求解 + 校验」闭环守住正确性。

只依赖标准库 + numpy：
  * 论文生产版调用 SCIP 解 MILP；真实部署请换成正经的 MIP 求解器。
  * 自带的 `solve_separable_convex` 是一个**精确**算法，但适用面很窄：
    仅当「目标对每个仓可分离 + 只有一条守恒约束 + 箱式上下界」时成立 —— SD/SB 两族
    正好落在这一类；带 0-1 变量与大 M 的 KI 族**不能**用它，必须交给 MILP 求解器。
"""
from __future__ import annotations

import heapq
import itertools
import math
import re

import numpy as np

# --- 实例与业务常量（本卡合成数据，可复现；非论文数据）---------------------------
WAREHOUSES = ("FBA-美西ONT8", "FBA-美东BWI2", "海外仓-长滩", "海外仓-新泽西")
# 运营在群里用的简写；解析器要同时认简写与全名
ALIASES = {"A仓": "FBA-美西ONT8", "B仓": "海外仓-新泽西",
           "C仓": "FBA-美东BWI2", "D仓": "海外仓-长滩"}
# 运营没说具体百分比时使用的默认档（阈值应写进配置，别写死在代码里）
DEFAULT_MIN_RATIO = 0.20   # 「不能断货」默认给到 20% 的下限
DEFAULT_MAX_RATIO = 0.10   # 「仓储费太贵」默认压到 10% 的上限

DEMO = {
    "I": np.array([420.0, 180.0, 260.0, 90.0]),   # 各仓在库（台）
    "D": np.array([22.0, 14.0, 11.0, 8.0]),       # 各仓预测日需求（台/天）
    "R": 660,                                     # 本批次待分配总量（台）
    "ell": 0.8, "ubar": 1.2,                      # TID 平衡带控制参数（对齐论文 §3.1）
}


def parse_cn_allocation_request(text, warehouses=WAREHOUSES, aliases=None):
    """把运营的中文分仓要求解析成结构化约束。

    识别三类片段，对应论文 Table 1 的 OC1 / OC2 / OC5：
      「不能断货 / 宁可多备 / 优先保」 → 比例下限 α_k（OC1）
      「别压货 / 仓储费太贵 / 少压」   → 比例上限 β_k（OC2）
      「停收 / 限收 / 不收」           → 数量上限置 0（OC5）

    返回 (spec, unmatched)。unmatched 是**没能解析的片段**：宁可报出来让人补，
    也不静默丢掉 —— 静默丢约束正是「代码能跑但算错」的典型来源
    （论文 §5.1 把这类失败列为 negative instance：漏掉关键约束）。
    """
    aliases = dict(aliases or ALIASES)
    name_of = {w: w for w in warehouses}
    name_of.update(aliases)

    clauses = [c.strip() for c in re.split(r"[，。；,;\n]", text) if c.strip()]
    spec = {"min_ratio": {}, "max_ratio": {}, "min_qty": {}, "max_qty": {}}
    unmatched: list[str] = []

    lo_words = ("不能断货", "不断货", "优先保", "宁可多备", "多备", "保供", "别断")
    hi_words = ("别压货", "不压货", "仓储费", "少压", "尽量别", "清库", "停收", "限收", "不收")

    for cl in clauses:
        compact = cl.replace(" ", "")          # 运营会写「A 仓」，简写表里是「A仓」
        targets = [full for alias, full in name_of.items() if alias in compact]
        if not targets:
            unmatched.append(cl)
            continue
        is_lo = any(w in cl for w in lo_words)
        is_hi = any(w in cl for w in hi_words)
        pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|％|个点|成)", cl)
        val = float(pct.group(1)) / 100.0 if pct else None

        if any(w in cl for w in ("停收", "限收", "不收")):
            for w in targets:
                spec["max_qty"][w] = 0
        elif is_lo and not is_hi:
            for w in targets:
                spec["min_ratio"][w] = val if val is not None else DEFAULT_MIN_RATIO
        elif is_hi and not is_lo:
            for w in targets:
                spec["max_ratio"][w] = val if val is not None else DEFAULT_MAX_RATIO
        elif is_lo and is_hi:      # 同一句里既提下限又提上限 → 交给人复核，别猜
            unmatched.append(cl)
        else:
            unmatched.append(cl)
    return spec, unmatched


def spec_to_bounds(spec, R, n, warehouses=WAREHOUSES):
    """结构化约束 → 每个仓的整数上下界 [lo, hi]（OC1/OC2/OC4/OC5 的落地形式）。"""
    lo = np.zeros(n, dtype=int)
    hi = np.full(n, R, dtype=int)
    for i, w in enumerate(warehouses):
        if w in spec["min_ratio"]:
            lo[i] = max(lo[i], int(math.ceil(spec["min_ratio"][w] * R)))
        if w in spec["max_ratio"]:
            hi[i] = min(hi[i], int(math.floor(spec["max_ratio"][w] * R)))
        if w in spec["min_qty"]:
            lo[i] = max(lo[i], int(spec["min_qty"][w]))
        if w in spec["max_qty"]:
            hi[i] = min(hi[i], int(spec["max_qty"][w]))
    return lo, hi


def relax_if_infeasible(lo, hi, R):
    """可行性检测 + 罚函数式松弛（对齐论文 §4.5 的 detect–relax–resolve 环）。

    不可行的两个来源（与论文 §6.3 的两个算例同构）：
      ① Σlo > R —— 多个仓的下限互相打架（论文里是各仓组最小份额之和 > 1）
      ② Σhi < R —— 上限加起来也装不下这一批（论文里是数量上限/等式约束过紧）

    真实部署应由 MILP 一次解出：给硬约束配松弛变量 u、在目标里加罚 -ρ·Σu（论文 Table 3）。
    这里用同构的贪心松弛 —— 「下限最大者先让」是刻意的确定性规则（避免引入 ρ 这一层参数），
    并把**放宽了哪些约束、放宽了多少**原样报出来，供人工确认。
    """
    lo = np.array(lo, dtype=int).copy()
    hi = np.array(hi, dtype=int).copy()
    if np.any(lo > hi):
        raise ValueError("存在 lo > hi 的仓库：约束自相矛盾，需要人工裁决")
    slack_lo = np.zeros_like(lo)      # u^-：下限被放宽的幅度
    slack_hi = np.zeros_like(hi)      # u^+：上限被放宽的幅度
    log: list[str] = []

    while lo.sum() > R:               # ① 下限打架：先动下限最大的那个仓
        i = int(np.argmax(lo))
        cut = int(min(lo[i], lo.sum() - R))
        lo[i] -= cut
        slack_lo[i] += cut
        log.append(f"{WAREHOUSES[i]}: 下限放宽 {cut} 台")
    while hi.sum() < R:               # ② 上限不够：从余量最大的仓往上抬
        room = int(R) - int(hi.sum())
        i = int(np.argmax(np.asarray([R] * len(hi)) - hi))
        add = int(min(room, R - hi[i]))
        if add <= 0:
            raise RuntimeError("松弛未能恢复可行性")
        hi[i] += add
        slack_hi[i] += add
        log.append(f"{WAREHOUSES[i]}: 上限放宽 {add} 台")
    return lo, hi, {"u_minus": slack_lo, "u_plus": slack_hi}, log
```

### 第二段：MIP 的可分离精确求解 + 求解器式三项校验

```python
# --- 第二层：MIP 的可分离精确求解（SD / SB 两族；KI 族请用 MILP 求解器）------------
def system_tid(I, D, R):
    """系统级 TID：τ_all = Σ(I+x)/ΣD —— 因 Σx=R，它对决策是**常数**。

    这一步是整张卡的数学支点：τ_all 与 x 无关，于是 SD/SB 的目标对每个仓可分离，
    「每台给边际代价最小的仓」这种贪心才可能是精确的。
    """
    return float((np.sum(I) + R) / np.sum(D))


def target_inventory_levels(I, D, R):
    """目标库存量 T_k = τ_all · D_k（论文 §3.1）。"""
    return system_tid(I, D, R) * np.asarray(D, dtype=float)


def solve_separable_convex(n, R, lo, hi, marginal):
    """min Σ f_k(x_k)  s.t. Σx_k=R, lo≤x≤hi, x 整数。

    `marginal(k, x)` 返回第 x+1 台（即 f_k(x+1)-f_k(x)）的边际代价。
    凸性 ⇒ 边际代价随 x 非递减 ⇒ 「每台都分给当前边际代价最小的仓」即最优
    （可分离凸资源分配问题的标准结论）。用最小堆按需生成边际值，O(R·log n)。
    """
    lo = np.array(lo, dtype=int)
    hi = np.array(hi, dtype=int)
    R = int(R)
    if np.any(lo < 0) or np.any(lo > hi):
        raise ValueError("上下界不合法")
    if R < int(lo.sum()) or R > int(hi.sum()):
        raise ValueError(f"不可行：R={R} 不在 [{int(lo.sum())}, {int(hi.sum())}] 内")

    x = lo.copy()
    rem = R - int(x.sum())
    heap = [(float(marginal(k, x[k])), k) for k in range(n) if x[k] < hi[k]]
    heapq.heapify(heap)
    while rem > 0:
        if not heap:
            raise RuntimeError("可行域为空，贪心未能分配完")
        _, k = heapq.heappop(heap)
        x[k] += 1
        rem -= 1
        if x[k] < hi[k]:
            heapq.heappush(heap, (float(marginal(k, x[k])), k))
    return x


def sd_marginal(I, D, R):
    """SD/LSD 族（论文 §4.1）的边际代价：f_k(x)=|I_k+x-T_k|/D_k。"""
    I = np.asarray(I, float); D = np.asarray(D, float)
    T = target_inventory_levels(I, D, R)

    def marginal(k, x):
        return (abs(I[k] + x + 1 - T[k]) - abs(I[k] + x - T[k])) / D[k]
    return marginal


def band_marginal(I, D, R, ell, ubar):
    """SB 族（论文 §4.2）的边际代价：TID 落在 [ℓ·τ_all, ℓ̄·τ_all] 外才罚，罚的是越界幅度。"""
    I = np.asarray(I, float); D = np.asarray(D, float)
    tau = system_tid(I, D, R)
    lower, upper = ell * tau, ubar * tau

    def cost(k, x):
        tid = (I[k] + x) / D[k]
        return max(0.0, lower - tid) + max(0.0, tid - upper)

    def marginal(k, x):
        return cost(k, x + 1) - cost(k, x)
    return marginal


def baseline_proportional(R, D):
    """基线（对齐论文里的 incumbent 直觉）：按预测需求占比摊，不做任何建模。"""
    D = np.asarray(D, float)
    raw = R * D / D.sum()
    x = np.floor(raw).astype(int)
    x[int(np.argmax(raw - np.floor(raw)))] += int(R - x.sum())
    return x


# --- 第三层：求解器式校验（论文的 solver-as-correctness-filter）--------------------
def tid_accuracy(x, I, D, ell, ubar):
    """TID 平衡率（论文 §3.1 的 allocation accuracy / balance rate）。

    分子是「TID 落在带内」的仓所分到的量，分母是本批总量 —— 所以它是**量加权**的：
    一个已经齐平但没分到货的仓，对分子贡献为 0。
    """
    x = np.asarray(x, float); I = np.asarray(I, float); D = np.asarray(D, float)
    tau = (I + x).sum() / D.sum()
    tid = (I + x) / D
    in_band = (tid >= ell * tau) & (tid <= ubar * tau)
    total = float(x.sum())
    return float(x[in_band].sum() / total) if total > 0 else 0.0


def verify_plan(x, I, D, R, lo, hi, ell, ubar):
    """三项校验，对齐论文「求解器同时充当执行引擎与正确性过滤器」的定位。"""
    x = np.asarray(x, int)
    return {
        "executable": True,                              # 能算出来（比「能跑」弱一层）
        "conservation_ok": int(x.sum()) == int(R),       # Σx = R
        "bounds_ok": bool(np.all(x >= lo) and np.all(x <= hi)),
        "negative_ok": bool(np.all(x >= 0)),
        "accuracy": tid_accuracy(x, I, D, ell, ubar),
        "tid": (np.asarray(I, float) + x) / np.asarray(D, float),
    }
```

### 第三段：候选公式的选择 + 聚合后的可行性修复

```python
# --- 第四层：候选公式的选择与融合（论文 §5.2；LightGBM 用岭回归代替）---------------
class ScoreEstimator:
    """预测「每个候选公式在这个实例上的期望质量」并做 Top-K 加权融合。

    论文用 expert-specific LightGBM 回归器（§5.2）；本模板换成闭式岭回归 ——
    它的职责只是「打分」，换成什么回归器都不影响流程形状。
    """

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.w = None

    def fit(self, Phi, y):
        Phi = np.asarray(Phi, float); y = np.asarray(y, float)
        n, p = Phi.shape
        self.w = np.linalg.solve(Phi.T @ Phi + self.l2 * np.eye(p), Phi.T @ y)
        return self

    def predict(self, phi):
        return float(np.asarray(phi, float) @ self.w)

    @staticmethod
    def softmax_weights(scores, kappa=1.0):
        """论文 §5.2 的温度 softmax 权重 θ_ij。"""
        s = np.asarray(scores, float) / float(kappa)
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    def fuse(self, phi_row, plans, R, lo=None, hi=None, top_k=2, kappa=1.0):
        """Top-K 选择 → softmax 加权 → 聚合后修复可行性（论文 §5.2）。"""
        plans = np.asarray(plans, int)
        n = plans.shape[1]
        lo = np.zeros(n, int) if lo is None else np.asarray(lo, int)
        hi = np.full(n, int(R)) if hi is None else np.asarray(hi, int)
        scores = np.array([self.predict(phi) for phi in phi_row])
        order = np.argsort(-scores)[:top_k]
        theta = self.softmax_weights(scores[order], kappa=kappa)
        fused_raw = theta @ plans[order]
        fused = repair_plan(np.floor(fused_raw).astype(int), R, lo, hi, preference=fused_raw)
        return {"scores": scores, "selected": order, "weights": theta,
                "plan_raw": fused_raw, "plan": fused,
                "repaired": bool(np.any(fused != np.floor(fused_raw).astype(int)))}


def repair_plan(x, R, lo, hi, preference):
    """把聚合出来的分仓方案修回可行域：先截到上下界，再补齐/扣除守恒差额。

    论文 §5.2 只说 aggregation 后要「restore feasibility」；这里把这一步显式写出来 ——
    少了它，融合层会输出越界方案，而越界方案在下游就是「某个仓根本装不下」。
    """
    x = np.clip(np.asarray(x, int), lo, hi).astype(int)
    order = list(np.argsort(-np.asarray(preference, float)))
    diff = int(R) - int(x.sum())
    if diff > 0:
        for i in order:
            add = int(min(diff, hi[i] - x[i]))
            x[i] += add; diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        for i in reversed(order):
            cut = int(min(-diff, x[i] - lo[i]))
            x[i] -= cut; diff += cut
            if diff >= 0:
                break
    if int(x.sum()) != int(R):
        raise RuntimeError("修复后仍不满足 Σx=R：可行域与 R 不匹配，应先做松弛（见 §4.5）")
    return x
```

### 第四段：自检与演示

```python
# --- 自检 ----------------------------------------------------------------------
def _bruteforce_sd(I, D, R):
    """穷举全部整数分仓方案求 SD 目标最优值 —— 只用于小算例自证贪心的最优性。"""
    I = np.asarray(I, float); D = np.asarray(D, float)
    T = target_inventory_levels(I, D, R)
    n = len(I)
    best = math.inf
    for cuts in itertools.combinations(range(R + n - 1), n - 1):
        pts = (-1,) + cuts + (R + n - 1,)
        x = np.array([pts[i + 1] - pts[i] - 1 for i in range(n)])
        best = min(best, float(np.sum(np.abs(I + x - T) / D)))
    return best


def test_greedy_matches_bruteforce_on_small_instance():
    """自证：在可穷举的小算例上，堆贪心给出的 SD 目标与暴力最优值完全一致。"""
    I = np.array([5.0, 1.0, 3.0]); D = np.array([2.0, 1.0, 1.5]); R = 6
    x = solve_separable_convex(3, R, np.zeros(3, int), np.full(3, int(R)), sd_marginal(I, D, R))
    assert int(x.sum()) == R
    T = target_inventory_levels(I, D, R)
    greedy = float(np.sum(np.abs(I + x - T) / D))
    assert abs(greedy - _bruteforce_sd(I, D, R)) < 1e-9, (greedy, _bruteforce_sd(I, D, R))


def test_parser_maps_operator_sentence_to_bounds():
    """运营原话 → 约束：A 仓下限、B 仓上限，且两个仓都被正确识别。"""
    spec, unmatched = parse_cn_allocation_request("宁可 A 仓多备也不能断货，B 仓仓储费太贵尽量别压货")
    assert not unmatched, unmatched
    assert spec["min_ratio"]["FBA-美西ONT8"] > 0
    assert spec["max_ratio"]["海外仓-新泽西"] > 0
    lo, hi = spec_to_bounds(spec, 1000, len(WAREHOUSES))
    assert lo[0] == int(math.ceil(spec["min_ratio"]["FBA-美西ONT8"] * 1000)) > 0
    assert hi[3] == int(math.floor(spec["max_ratio"]["海外仓-新泽西"] * 1000)) < 1000


def test_stop_shipment_reroutes_and_conserves_total():
    """某仓临时停收：改派后该仓分到 0，总量与可行域都不变。"""
    spec, _ = parse_cn_allocation_request("FBA-美东BWI2 停收，本批全部改派其他仓")
    assert spec["max_qty"]["FBA-美东BWI2"] == 0
    lo, hi = spec_to_bounds(spec, DEMO["R"], len(WAREHOUSES))
    x = solve_separable_convex(len(WAREHOUSES), DEMO["R"], lo, hi,
                               sd_marginal(DEMO["I"], DEMO["D"], DEMO["R"]))
    assert x[1] == 0, x
    assert int(x.sum()) == DEMO["R"]


def test_infeasible_bounds_are_relaxed_and_reported():
    """下限之和超过总量 → 触发松弛，且放宽量与放宽对象都被显式报出。"""
    lo = np.array([40, 40, 10, 10]); hi = np.array([100, 100, 100, 100])
    lo2, hi2, slack, log = relax_if_infeasible(lo, hi, R=60)
    assert int(lo2.sum()) <= 60 and int(hi2.sum()) >= 60
    assert int(slack["u_minus"].sum()) > 0 and log, log
    x = solve_separable_convex(4, 60, lo2, hi2, sd_marginal(DEMO["I"], DEMO["D"], 60))
    assert int(x.sum()) == 60


def test_band_solution_reaches_full_accuracy_when_feasible():
    """可行时，SB 族的解应让每个仓都落在平衡带内（Acc=1.0）。"""
    I, D, R = DEMO["I"], DEMO["D"], DEMO["R"]
    x = solve_separable_convex(len(I), R, np.zeros(len(I), int), np.full(len(I), int(R)),
                               band_marginal(I, D, R, DEMO["ell"], DEMO["ubar"]))
    rep = verify_plan(x, I, D, R, np.zeros(len(I), int), np.full(len(I), int(R)),
                      DEMO["ell"], DEMO["ubar"])
    assert rep["conservation_ok"] and rep["bounds_ok"]
    assert abs(rep["accuracy"] - 1.0) < 1e-9, rep["accuracy"]


def test_fusion_restores_conservation():
    """融合（加权求和）会引入小数与守恒违约 → 修复后必须重新满足 Σx=R。"""
    rng = np.random.default_rng(7)
    Phi = rng.normal(size=(9, 4)); y = rng.normal(size=9)
    est = ScoreEstimator().fit(Phi, y)
    plans = np.array([[100, 120, 0, 40], [90, 130, 20, 60], [0, 200, 60, 40]])
    out = est.fuse(Phi[:3], plans, R=260, top_k=2, kappa=1.0)
    assert int(np.asarray(out["plan"]).sum()) == 260
    assert abs(float(np.sum(out["weights"])) - 1.0) < 1e-9


if __name__ == "__main__":
    I, D, R = DEMO["I"], DEMO["D"], DEMO["R"]
    ell, ubar = DEMO["ell"], DEMO["ubar"]
    n = len(WAREHOUSES)
    np.set_printoptions(precision=2, suppress=True)

    print("== 1. 运营原话 → 结构化约束（规则解析器代替 LLM）==")
    text = "宁可 A 仓多备也不能断货，B 仓仓储费太贵尽量别压货"
    spec, unmatched = parse_cn_allocation_request(text)
    lo, hi = spec_to_bounds(spec, R, n)
    print(f"  原话: {text}")
    print(f"  约束: {spec}")
    print(f"  未解析片段: {unmatched}")
    print(f"  整数上下界 lo={lo} hi={hi}")

    print("== 2. 分仓量：基线 vs SD 族 vs SB 族 ==")
    x_base = baseline_proportional(R, D)
    x_sd = solve_separable_convex(n, R, lo, hi, sd_marginal(I, D, R))
    x_sb = solve_separable_convex(n, R, lo, hi, band_marginal(I, D, R, ell, ubar))
    hdr = "  仓".ljust(18) + "在库".rjust(8) + "日需求".rjust(9)
    for tag in ("基线", "SD", "SB"):
        hdr += tag.rjust(9)
    print(hdr)
    for i, w in enumerate(WAREHOUSES):
        print(f"  {w}".ljust(18) + f"{I[i]:.0f}".rjust(8) + f"{D[i]:.1f}".rjust(9)
              + f"{x_base[i]:d}".rjust(9) + f"{x_sd[i]:d}".rjust(9) + f"{x_sb[i]:d}".rjust(9))
    print(f"  系统级 TID τ_all = {system_tid(I, D, R):.2f} 天"
          f"（平衡带 [{ell:.1f}τ, {ubar:.1f}τ] = [{ell * system_tid(I, D, R):.2f}, {ubar * system_tid(I, D, R):.2f}] 天）")
    for tag, x in (("基线（按需求占比）", x_base), ("SD 族", x_sd), ("SB 族", x_sb)):
        rep = verify_plan(x, I, D, R, np.zeros(n, int), np.full(n, int(R)), ell, ubar)
        print(f"  {tag.ljust(12)} 平衡率 Acc={rep['accuracy']:.4f}  守恒={rep['conservation_ok']}"
              f"  界内={rep['bounds_ok']}")
    tau = system_tid(I, D, R)
    for i, w in enumerate(WAREHOUSES):
        tid_sd = (I[i] + x_sd[i]) / D[i]
        flag = "偏紧(断货风险)" if tid_sd < ell * tau else ("压货(仓储费)" if tid_sd > ubar * tau else "带内")
        print(f"    {w.ljust(14)} SD 解后 TID={tid_sd:5.2f} 天 → {flag}")

    print("== 3. 求解器式校验（三项信号）==")
    rep = verify_plan(x_sd, I, D, R, lo, hi, ell, ubar)
    print(f"  可执行={rep['executable']}  可行(守恒+上下界)={rep['conservation_ok'] and rep['bounds_ok']}"
          f"  解质量 Acc={rep['accuracy']:.4f}")
    print(f"  逐仓 TID={rep['tid']}")

    print("== 4. 改派：FBA-美东BWI2 临时停收 ==")
    spec2, _ = parse_cn_allocation_request("FBA-美东BWI2 停收，本批全部改派其他仓")
    lo2, hi2 = spec_to_bounds(spec2, R, n)
    x2 = solve_separable_convex(n, R, lo2, hi2, sd_marginal(I, D, R))
    x2_sb = solve_separable_convex(n, R, lo2, hi2, band_marginal(I, D, R, ell, ubar))
    print(f"  新上下界 lo={lo2} hi={hi2}")
    print(f"  SD 族 分仓={x2}  合计={int(x2.sum())}  停收仓获得={int(x2[1])} 台"
          f"  Acc={tid_accuracy(x2, I, D, ell, ubar):.4f}")
    print(f"  SB 族 分仓={x2_sb}  合计={int(x2_sb.sum())}  停收仓获得={int(x2_sb[1])} 台"
          f"  Acc={tid_accuracy(x2_sb, I, D, ell, ubar):.4f}")

    print("== 5. 不可行 → 松弛 → 可解 ==")
    lo_bad = np.array([300, 300, 200, 200]); hi_bad = np.full(n, R)
    lo3, hi3, slack, log = relax_if_infeasible(lo_bad, hi_bad, R)
    print(f"  原下限 {lo_bad} 之和 {int(lo_bad.sum())} > R={R} → 不可行")
    for line in log:
        print(f"  松弛: {line}")
    x3 = solve_separable_convex(n, R, lo3, hi3, sd_marginal(I, D, R))
    print(f"  松弛后分仓={x3}  合计={int(x3.sum())}  下限违规量 u^-={slack['u_minus']}")

    print("== 6. 候选公式选择与融合（岭回归代替 LightGBM）==")
    rng = np.random.default_rng(11)
    Phi = rng.normal(size=(12, 3))
    quality = np.array([tid_accuracy(baseline_proportional(R, D), I, D, ell, ubar),
                        tid_accuracy(x_sd, I, D, ell, ubar),
                        tid_accuracy(x_sb, I, D, ell, ubar)])
    y = np.tile(quality, 4) + 0.01 * rng.normal(size=12)
    est = ScoreEstimator(l2=1.0).fit(Phi, y)
    plans = np.vstack([baseline_proportional(R, D), x_sd, x_sb])
    fused = est.fuse(Phi[:3], plans, R=R, lo=lo, hi=hi, top_k=2, kappa=0.5)
    print(f"  候选得分={np.round(fused['scores'], 4)}  选中={[int(i) for i in fused['selected']]}"
          f"  权重={np.round(fused['weights'], 4)}")
    print(f"  融合后（未修复）={np.round(fused['plan_raw'], 2)}  越界/守恒违约需修复={fused['repaired']}")
    print(f"  修复后分仓={fused['plan']}  合计={int(fused['plan'].sum())}")
    print(f"  融合后平衡率 Acc={tid_accuracy(fused['plan'], I, D, ell, ubar):.4f}")
    print("全部自检通过 ✅")
```

**本模板的实际运行输出**（`python3 <本卡代码>.py` 可复现；**这是合成实例的结果，不是论文数据**）：

```text
== 1. 运营原话 → 结构化约束（规则解析器代替 LLM）==
  原话: 宁可 A 仓多备也不能断货，B 仓仓储费太贵尽量别压货
  约束: {'min_ratio': {'FBA-美西ONT8': 0.2}, 'max_ratio': {'海外仓-新泽西': 0.1}, 'min_qty': {}, 'max_qty': {}}
  未解析片段: []
  整数上下界 lo=[132   0   0   0] hi=[660 660 660  66]
== 2. 分仓量：基线 vs SD 族 vs SB 族 ==
  仓                     在库      日需求       基线       SD       SB
  FBA-美西ONT8           420     22.0      264      302      352
  FBA-美东BWI2           180     14.0      168      230      242
  海外仓-长滩               260     11.0      132       62        0
  海外仓-新泽西               90      8.0       96       66       66
  系统级 TID τ_all = 29.27 天（平衡带 [0.8τ, 1.2τ] = [23.42, 35.13] 天）
  基线（按需求占比）    平衡率 Acc=0.6545  守恒=True  界内=True
  SD 族         平衡率 Acc=0.9000  守恒=True  界内=True
  SB 族         平衡率 Acc=0.9000  守恒=True  界内=True
    FBA-美西ONT8     SD 解后 TID=32.82 天 → 带内
    FBA-美东BWI2     SD 解后 TID=29.29 天 → 带内
    海外仓-长滩         SD 解后 TID=29.27 天 → 带内
    海外仓-新泽西        SD 解后 TID=19.50 天 → 偏紧(断货风险)
== 3. 求解器式校验（三项信号）==
  可执行=True  可行(守恒+上下界)=True  解质量 Acc=0.9000
  逐仓 TID=[32.82 29.29 29.27 19.5 ]
== 4. 改派：FBA-美东BWI2 临时停收 ==
  新上下界 lo=[0 0 0 0] hi=[660   0 660 660]
  SD 族 分仓=[454   0  62 144]  合计=660  停收仓获得=0 台  Acc=0.3121
  SB 族 分仓=[352   0 126 182]  合计=660  停收仓获得=0 台  Acc=1.0000
== 5. 不可行 → 松弛 → 可解 ==
  原下限 [300 300 200 200] 之和 1000 > R=660 → 不可行
  松弛: FBA-美西ONT8: 下限放宽 300 台
  松弛: FBA-美东BWI2: 下限放宽 40 台
  松弛后分仓=[  0 260 200 200]  合计=660  下限违规量 u^-=[300  40   0   0]
== 6. 候选公式选择与融合（岭回归代替 LightGBM）==
  候选得分=[-0.15  0.21 -0.37]  选中=[1, 0]  权重=[0.67 0.33]
  融合后（未修复）=[289.6  209.77  84.84  75.79]  越界/守恒违约需修复=True
  修复后分仓=[301 209  84  66]  合计=660
  融合后平衡率 Acc=0.9000
全部自检通过 ✅
```

> ⚠️ **不要把上表的平衡率/分仓量与论文 §7.1 报告的百分点增益混为一谈。** 上表全部来自本卡的合成实例，
> 只能说明「约束进得去、解出得来、校验拦得住」这套逻辑自洽；论文那几个百分点是京东生产批次上的读数
> （见 ⑥ 引用）。两者谁高谁低都不构成互相印证 —— 本卡**没有**复现论文实验，也**没有**贵司数据。
>
> ⚠️ 另一半更值得看的读数在 §4 行：同一个停收场景，SD 族与 SB 族给出的分仓方案完全不同
> （平衡率差了一个数量级）。这不是 bug，而是**目标函数不同 ⇒ 业务结论不同**的直接证据 ——
> 也正是论文要做「公式族 + 实例级选择」而不是只留一个模型的原因。

---

## ④ 技能关联

| 关联卡片 | 关系 | 为什么组合 |
|---|---|---|
| `Skill-Safety-Stock-Replenishment.md` | 前置（上游） | 本卡只解「$R$ 已定、怎么分到各仓」。$R$ 与各仓安全库存下限 $d_k$ 由它给出 —— 把 $d_k$ 直接填成本卡的 `min_qty`，就得到「不许把某仓压到安全库存以下」的硬约束。 |
| `Skill-Demand-Forecasting-Supply-Chain.md` | 前置（输入层） | TID 的分母是各仓**预测日需求** $D_k$，且要求 SKU × 仓粒度、≥8 周、含旺季抬升。预测偏差会一整条传导进分仓结论，所以必须先看它的分层预测与偏差成本。 |
| `Skill-Multi-Echelon-Inventory.md` | 边界（本卡不做） | 本卡是**单期、面向仓的分仓**；论文 §8 自承未扩展到多期与网络级。做「工厂→海外仓→FBA」多级联动或滚动补货，用多阶库存优化，不要用本卡硬凑。 |
| `Skill-Two-Echelon-Inventory-DRL.md` | 替代路线 | 同一问题的学习型解法：DRL 学「什么时候发多少」的策略、可带多期；本卡解单期 MIP、可解释、可审计。改派极其频繁、规则天天变时，前者的适应性更强；要「说得清为什么这么分」时用本卡。 |
| `Skill-Tool-Description-Audit.md` | 可组合（可靠性前提） | 论文的高生成成功率依赖把建模 schema 与求解器 API 契约写进提示（constrained PMC protocol）。给 LLM 的约束字段与工具描述一旦含糊，生成质量立刻退化 —— 上本卡前先用它审一遍字段定义。 |
| `Skill-SQL-Agent-Text-to-SQL.md` | 结构类比 + 可组合 | 同一个形状：「自然语言 → 可执行工件 → 必须校验」。分仓前用它把 $I_k$、$D_k$、仓容与停收事件从数仓里取出来，正好是本卡 ③ 的输入。 |

---

## ⑤ 商业价值评估

**ROI 预估**（给公式与参数来源，不直接给编造的数）：

分仓优化净收益 = 断货损失减少额 + 滞销仓储费减少额 + 改派加急成本减少额 − 系统建设与人工复核成本。
其中每一项的算法（可复现算式，代入贵司参数即可算）：

```text
断货损失减少额   =（优化前后缺货天数差）× 该仓日均需求 × 单台毛利
滞销仓储费减少额 =（优化前后平均在库台数差 ÷ 箱规）× 单箱月仓储费 × 月数
```

参数来源（全部在**企业内**可得，不需要外部基准）：
① **缺货天数差** —— 本卡输出逐仓 TID 与「偏紧/压货」标记，对照历史缺货记录即可算；
② **单台毛利** —— 贵司 SKU 财务表；③ **箱规与单箱月仓储费** —— 海外仓报价单 / FBA 仓储费率表；
④ **头程加急差价** —— 货代报价单；⑤ **复核成本** —— 每周人工核对约束表的人时。

论文给出的**方向性参照**（是京东生产系统的读数，**不是**贵司的预期收益）：
固定 KI 公式把 TID 平衡率整体拉高 3.4 个百分点、29 个生产批次里 26 个有改善；
加上公式选择后整体提升到 4.5 个百分点（§7.1，见 ⑥ 引用）。
但论文也如实报告了这个选择层的代价：26 个获益批次里有 7 个不如固定公式（§7.1，见 ⑥ 引用）——
所以**不要**把它当成「一定更好」的保证。

- **实施难度**：⭐⭐⭐⭐☆（4/5）。算法本身不难（本卡已给可运行的精确解 + 校验闭环），
  真正的成本在**数据工程**：SKU × 仓的日需求预测、仓容/箱规/入仓限制字段、停收事件的结构化，
  都要先补上；再叠一层「谁来看被放宽的约束」的运营流程。技术风险中、数据风险高。
- **优先级评分**：⭐⭐⭐⭐☆（4/5）。分仓直接决定断货率与仓储费两块现金流，
  且本卡产出的是**可复核的约束账**，比「再买一个预测模型」更容易验收。
  之所以不给满星：需求预测是它的上游 —— 预测偏了，分仓再优也只是把错误分得更均匀。
- **评估依据**：① 论文在生产批次上验证过（§7.1，非纯离线）；② 依赖的 $I_k$、$D_k$、$R$
  在平台店铺与独立站**大部分可得**，缺口是仓位与入仓限制两张表；
  ③ 与 04-供应链 的四张卡片直接串联，不是孤立新领域；
  ④ 上限约束：LLM 生成公式的可靠性有前提（`constrained PMC protocol`，§6.2），
  所以本卡主推 **「先用手写的 SD/SB 两族 + 求解器校验，再考虑让 LLM 去改约束」** 的推进顺序。

---

## ⑥ 原文引用

> 原文："Balance-oriented multi-warehouse inventory allocation is a recurring decision problem in large-scale e-commerce supply chains, in which a fixed replenishment quantity is distributed across warehouses to balance post-allocation inventory coverage while accounting for demand forecasts and heterogeneous allocation constraints."
> 出处：2606.29366 Abstract（fulltext.md L16）

> 原文："In practice, allocation requirements are often scenario-dependent and expressed in semi-structured or natural-language form rather than as ready-to-solve operations research (OR) formulations."
> 出处：2606.29366 Abstract（fulltext.md L16）

> 原文："The LLM component generates candidate formulations and executable solver code from textual or semi-structured specifications, while the solver provides verification signals for executability, feasibility, and solution quality."
> 出处：2606.29366 Abstract（fulltext.md L16）

> 原文："Experimental results on 29 production evaluation batches from JD.com show that the best single OR formulation improves allocation accuracy by 3.4 percentage points over the incumbent approach, while the full ORLA framework achieves a 4.5 percentage-point overall improvement and improves allocation accuracy in 26 of the 29 evaluation batches."
> 出处：2606.29366 Abstract（fulltext.md L16）

> 原文："We measure inventory coverage through Target Inventory Days (TID), defined as the ratio between available inventory and forecast demand, and use TID-based balance as the organizing principle of the optimization problem."
> 出处：2606.29366 §1 Introduction（fulltext.md L26）

> 原文："For a given stock keeping unit (SKU), the input consists of: (i) the on-hand inventory $I_{k}\geq 0$ at warehouse $k$, (ii) the forecasted daily demand $D_{k}>0$, and (iii) the total replenishment quantity $R\in\mathbb{Z}_{\geq 0}$ to be allocated across warehouses."
> 出处：2606.29366 §3.1 Problem Setup（fulltext.md L62）

> 原文："The decision is an integer allocation plan $x_{k}\in\mathbb{Z}_{\geq 0}$ for each warehouse $k$, satisfying the conservation constraint $\sum_{k=1}^{n}x_{k}=R$."
> 出处：2606.29366 §3.1 Problem Setup（fulltext.md L64）

> 原文："Based on these notions, the multi-warehouse allocation accuracy (or balance rate) is defined as"
> 出处：2606.29366 §3.1 Problem Setup（fulltext.md L78）

> 原文："In this section, three complementary mixed-integer formulations are developed, and we further enrich these base models through a modular constraint library and a penalty-based relaxation mechanism for instances in which strict allocation constraints render the original formulation infeasible."
> 出处：2606.29366 §4 A Family of OR Formulations（fulltext.md L100）

> 原文："real-world decision instances often involve heterogeneous operational rules, such as ratio constraints, lower and upper allocation bounds, case-pack restrictions, group-level service requirements, and cardinality controls."
> 出处：2606.29366 §4.4 Modular Heterogeneous Constraints（fulltext.md L204）

> 原文："A natural consequence of heterogeneous side constraints is that strict formulations may become infeasible."
> 出处：2606.29366 §4.5 Penalty-Based Relaxation（fulltext.md L294）

> 原文："A key design choice is that the solver serves both as an execution engine and as a correctness filter."
> 出处：2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L328）

> 原文："The hard-constraint model is solved first. If the solver reports infeasibility, the corresponding relaxation model is activated and re-solved to obtain a feasible allocation plan."
> 出处：2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L328）

> 原文："Our SFT dataset contains solver-verifiable PMC triples paired with structured prompts, covering the base formulation and 17 real-world constraint extensions."
> 出处：2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L332）

> 原文："Furthermore, we include targeted negative instances (e.g., non-linear modeling hallucinations, wrong variable types, missing key constraints) alongside positive instances to improve format robustness and error awareness."
> 出处：2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L332）

> 原文："If the generated code is executable, and its executed allocation matches the ground-truth plan, we label it as a positive sample."
> 出处：2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L336）

> 原文："Their outputs are then combined through score-aware weighting, and feasibility is restored when aggregation introduces mild violations."
> 出处：2606.29366 §5.2 Learning-Based Formulation-Selection（fulltext.md L340）

> 原文："For LLM training, we fine-tune the Qwen3-8B base model by minimizing the standard token-level cross-entropy loss under teacher forcing, with AdamW (Loshchilov and Hutter, 2019) as the optimizer and NEFTune (Jain et al., 2024) applied during training. The generated MILP problems are solved using SCIP."
> 出处：2606.29366 §6 Computational Evaluation（fulltext.md L358）

> 原文："We build PMC-style SFT data for all 18 formulations reported in Table 2, covering the base MIP formulation and its 17 variants."
> 出处：2606.29366 §6.1 Data Pools for Post-Training（fulltext.md L362）

> 原文："Negative multi-warehouse inventory allocation SFT samples are further constructed by injecting 6 common solver-script error types with approximately uniform frequency across categories."
> 出处：2606.29366 §6.1 Data Pools for Post-Training（fulltext.md L362）

> 原文："In the final mixture, the proportions of multi-warehouse inventory allocation positive samples, multi-warehouse inventory allocation negative samples, and classical MIP samples are 85%, 10%, and 5%, respectively."
> 出处：2606.29366 §6.1 Data Pools for Post-Training（fulltext.md L362）

> 原文："The evaluation is conducted on a held-out test set containing more than 10,000 real-world multi-warehouse allocation instances."
> 出处：2606.29366 §6.2 Code Generation Reliability（fulltext.md L368）

> 原文："| Code Syntax Accuracy | 99.9% | 100% |"
> 出处：2606.29366 §6.2 Code Generation Reliability；表 4 Code Syntax Accuracy 行（fulltext.md L374）

> 原文："| Code Execution Success Rate | 99% | 99.9% |"
> 出处：2606.29366 §6.2 Code Generation Reliability；表 4 Code Execution Success Rate 行（fulltext.md L376）

> 原文："| Code Execution Failure Rate | 1% | 0.1% |"
> 出处：2606.29366 §6.2 Code Generation Reliability；表 4 Code Execution Failure Rate 行（fulltext.md L378）

> 原文："On the held-out test set, the SFT+KTO variant attains 100% code syntax accuracy and reduces the code execution failure rate from 1.0% to 0.1% relative to the SFT-only variant."
> 出处：2606.29366 §6.2 Code Generation Reliability（fulltext.md L380）

> 原文："It is worth noting that these results are obtained under a constrained PMC protocol, where the MILP modeling and solver APIs schema are included during post-training."
> 出处：2606.29366 §6.2 Code Generation Reliability（fulltext.md L380）

> 原文："In this task, each predefined warehouse group is required to receive at least a minimum fraction of the total replenishment quantity."
> 出处：2606.29366 §6.3 Relaxation（fulltext.md L386）

> 原文："A typical conflicting case arises when the groups are disjoint and their required minimum shares satisfy $\eta_{1}+\eta_{2}>1$."
> 出处：2606.29366 §6.3 Relaxation（fulltext.md L386）

> 原文："In our task, we set $R=252$, $(D_{1},\ldots,D_{8})=(18.1,9.7,20.3,44.7,17.4,11.5,35.2,8.9)$, $(I_{1},\ldots,I_{8})=(33,13,34,102,28,21,66,14)$, $\eta_{1}=0.6$, $\eta_{2}=0.5$, $\underline{\ell}=0.8$, $\overline{\ell}=1.2$, $(\rho_{1},\rho_{2},\rho_{3})=\text{(200, 200, 10000)}$, and $M=1563$."
> 出处：2606.29366 §6.3 Relaxation；group minimum share 算例参数（fulltext.md L386）

> 原文："The fixed KI formulation improves 26 of the 29 production batches and achieves an overall improvement of 3.4 percentage points (pp). After applying the learning-based formulation-selection method, the overall improvement increases to 4.5 pp."
> 出处：2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L398）

> 原文："For example, in B19 and B20, the KI formulation improves accuracy by 0.8 and 2.9 pp, respectively, whereas the formulation-selection method increases the corresponding gains to 5.9 and 7.2 pp."
> 出处：2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）

> 原文："For all batches in which KI improves accuracy by more than 8 pp, the formulation-selection method achieves larger gains, with additional improvements over KI ranging from 0.2 to 1.6 pp."
> 出处：2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）

> 原文："Moreover, the number of batches with gains above 5 pp increases from 11 under KI to 17 under the formulation-selection method."
> 出处：2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）

> 原文："KI outperforms the formulation-selection method in 7 of the 26 improved batches, and only two of these cases show a gap of at least one percentage point."
> 出处：2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）

> 原文："For 5 task categories not included in post-training, we run inference with LLM and invoke the solver to verify the generated code, while checking the rationality of the resulting inventory allocation plans."
> 出处：2606.29366 §7.2 Generalization Evaluation（fulltext.md L440）

> 原文："Our LLM generates syntactically correct and executable solver code for all 5 scenarios. Moreover, for 3 of the 5 scenarios, the generated allocation plans are valid and satisfy all hard constraints."
> 出处：2606.29366 §7.2 Generalization Evaluation（fulltext.md L456）

> 原文："| Fixed allocation ratio | Enforce a fixed allocation ratio between two warehouses | ✓ | ✗ |"
> 出处：2606.29366 §7.2 Generalization Evaluation；表 6 Fixed allocation ratio 行（fulltext.md L452）

> 原文："| Minimum TID requirement | Enforce minimum TID | ✓ | ✗ |"
> 出处：2606.29366 §7.2 Generalization Evaluation；表 6 Minimum TID requirement 行（论文自己举的自然语言约束例子）（fulltext.md L454）

> 原文："Third, the framework can be extended from single-period allocation to multi-period and network-level planning."
> 出处：2606.29366 §8 Conclusion（fulltext.md L462）

> 原文："Journal: European Journal of Operational Research"
> 出处：2606.29366 首页页眉（投稿目标标注，非录用信息；fulltext.md L11）

---

## 附：证据链与核验方式

- **全文底本**：`paper2skills-vault/papers/04-供应链/p2s-2026-0008/fulltext.md`
  （由 `paper2skills-research/scripts/fetch_fulltext.py` 从 arXiv LaTeXML HTML 转换，保留章节号）
- **引文逐字核验**：`quote_check.py --card <本卡>` —— 每条 ⑥ 引用块回查全文底本，报告
  `VERBATIM` / `FUZZY` / `FABRICATED`
- **K1 代码可执行**：`verify_skill_code.py --card <本卡>`（四个 python 块按文档顺序拼成一个模块后执行 + pytest）
- **K2 门禁**：`gate_check.py --card <本卡> --k1 <K1 报告.json>`
  （G1 需要 K1 的 PASS 凭证，**没有凭证一律判红**，这是设计如此）
- **本地可复现数字的约定**：论文事实数字一律进 ⑥ 引用块；本模板的运行输出与按贵司参数代入的算式
  放进代码/输出围栏并标注「可复现」—— 围栏内的数字是**可自行验证**的，不冒充论文结论。

```text
# 复现命令（在本仓库根目录执行）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md
```
