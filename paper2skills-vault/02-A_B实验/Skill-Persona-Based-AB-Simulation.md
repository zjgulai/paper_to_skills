---
title: Skill-Persona-Based-AB-Simulation
module: 02-A_B实验
topic: 数据驱动人格 agent 的 A/B 事前仿真——配对评分提问 → 方向预测聚合 → 只做粗筛、不替代实验
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
venue_tier: CCF-B
venue_source: EXPLICIT_RETIERS
paper_id: 2609.01038
paper: Data-Driven Persona-Conditioned Agents for A/B Test Simulation
venue: EMNLP 2026 Industry Track
evidence_grade: A
verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（41/41 VERBATIM）+ gate_check.py G2/G3 passed + 人工抽检：0.90 vs 0.80 / 0.70–0.90 vs 0.57–0.69 两组数字逐个回查 fulltext.md 叙述句（表行脏串不用）
verified_at: 2026-09-12
supersedes:
related: Skill-AB-Experimental-Design.md, Skill-Power-Analysis-Sample-Size.md, Skill-AB-Test-Result-Interpretation.md, Skill-Multi-Armed-Bandit.md, Skill-Incrementality-Measurement.md
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-05
l2_domain: 品牌与增长
l3_id: DOM-05-104
l3_business: 实验设计
l3_all: 实验设计 / 内容实验
l1_l2_l3: 业务运营/品牌与增长/实验设计
---

# Skill Card: 数据驱动人格 A/B 仿真（Data-Driven Persona-Conditioned A/B Simulation）

**这张卡与其他 A/B 卡的分工**：`Skill-AB-Experimental-Design.md` 回答「实验怎么设计」、
`Skill-Power-Analysis-Sample-Size.md` 回答「要多少流量才跑得动」、
`Skill-AB-Test-Result-Interpretation.md` 回答「跑完了怎么读」。
本卡解决它们前面的一步：**在花掉流量之前，先判断哪个候选方案的方向最有可能是正的**。
它是一条**粗筛**链路 —— 论文自己的定位就是「不能替代真实实验，也不需要替代」（⑥ Q3）。

---

## ① 算法原理

**核心思想**：把「预测 A/B 结果」改写成一道**结构化提问**。先用真实行为数据构造一批人格 agent，
再把控件与实验件**同屏**摆在每个 agent 面前，让它按 1–10 分各打一个分；把逐人格的相对分差聚合成
一个预测分布，**只看它的方向**与真实实验是否一致。

**数学直觉**：单人格相对分差 (s_treat − s_ref)/s_ref 的均值就是预测相对效应 δ̂_s，其标准误为 σ̂_s；
预测「实验件更优」的概率 q = Φ(δ̂_s/σ̂_s)。真实实验给出 p = Φ(δ̂_rel/σ̂_rel)。
方向准确率 Acc = I[(p−0.5)(q−0.5)>0]；SignOv = 1−|p−q| 另外衡量置信度校准。
只看方向不看幅度，是因为「上不上线」这个决策只取决于符号，而幅度型指标会把「方向判错」和
「方向判对但幅度偏」混在一起。

**关键假设**：① 人格数据与业务**同域**（这是全卡最重要的一条）；② 提问必须是**配对**呈现
（同屏比较），单独打分会让 agent 无法校准；③ 被评估的效应量不能太小，近零效应会被扰动翻转方向。

---

## ①b 反例与适用边界

**什么时候不要用这个算法**

1. **绝不能替代真实 A/B。** 这是论文自己的定位，也是 registry 的备注口径：论文 §6 明确写
   「当前准确率下，该框架**不能**完全替代真人 A/B 测试」（⑥ Q3），它的用途是**事前粗筛**——
   把明显更差的候选先筛掉、把候选按预测影响排序（⑥ Q4）。任何把它当成「不用跑实验了」的用法
   都是错的。
2. **人格数据与业务域不对齐时不要用。** 论文实测域内行为数据能到 0.70–0.90，
   而**域外来源掉到 0.57–0.69**（⑥ Q22）；具体地，基于影视评分（娱乐偏好）的人格在 CTR 上还有
   0.65、到订阅类测试就掉到 0.60（⑥ Q19）。拿娱乐/问卷数据去仿母婴品类的购买决策，属于这一类。
3. **两类度量必须分开报告，不能合并成一个数。** 论文里同一个配置在两类度量上结论可以相反：
   二元配对格式在订阅测试上很强、在 CTR 上失败（⑥ Q8）；深度池在 CTR 上显著更好、在订阅上
   与代表池无显著差异（⑥ Q24）。把 CTR 与订阅（sign-up）平均成一个「准确率」，会把这两条
   互相冲突的证据抹平。
4. **只做人口统计（demographics-only）不要用。** 把人格剥到只有人口统计属性后，订阅测试的
   准确率从 0.80 掉到 0.30（⑥ Q30）；单一个「通用买家」人格也不比完全不给人格更好（⑥ Q31）。
   驱动效果的是**行为档案 + 群体多样性**，不是「加了个人格设定」这件事本身。
5. **人格行为太稀疏不要用。** 论文 §6 给了一条可操作门槛：交易记录低于 20 笔时仿真会退化，
   LLM 会退回泛化推理（⑥ Q23）。内部机制是「稀疏人格缺行为依据 → 退回泛化推理」（⑥ Q25）。
6. **近零效应不要用。** 论文明确：效应量大时才可靠，近零效应下小扰动就能翻转预测方向，
   因此输出更适合当**排序信号**而不是「非上即否」的开关（⑥ Q33）。

**已知的失败模式**

1. **把仿真的方向判断当因果增量。** 它预测的是「哪个方案指标更高」，不是「哪个方案带来了增量」；
   与 `Skill-Incrementality-Measurement.md` 的增量口径是两件事。
2. **孤立截图去掉情境。** 论文评估的是孤立截图而非完整页面，浏览意图、会话历史、周边内容等
   情境因素被剥离（⑥ Q34）——母婴品类里「正在比价」和「随手刷」的用户会被当成同一类。
3. **乐观偏差被符号指标掩盖。** LLM 的正向偏差与锚定会让控件与实验件**同时**被高估，
   符号类指标在这种「两侧同等膨胀」下部分失效（⑥ Q37）。
4. **人格的人口统计缺口来自 LLM 推断偏差，而不是抽样。** 论文发现深度池与代表池的人口统计
   缺口**互补**，并判断这些缺口「更可能来自人格生成时的 LLM 推断偏差，而非抽样限制」（⑥ Q26）
   —— 也就是说，**换更大的样本解决不了这个偏差**。
5. **模型依赖。** 全部实验只用了单一 LLM（Claude Sonnet 4.5），跨 Claude Haiku / Opus 做过一致性
   验证，但非 Anthropic 模型未测（⑥ Q39）。换模型等于换一套偏差，需要重新校准。

**论文自己承认的局限**（⑥ Q34–Q39）

1. 基准只有 **40 个测试、单一电商域、两类度量**，对其他域、其他度量类型、多步用户旅程的泛化
   **未经验证**（⑥ Q35）。
2. 人格的人口统计属性是 LLM **推断**而非自报（⑥ Q36）。
3. **基准的 ground-truth 标签没有公开**，外部复现受限；论文的补救说法是方法论可以用任何
   偏好数据集复现（⑥ Q38）。
4. **论文未讨论**：母婴 / 跨境电商场景、非英语市场、站内广告位与独立站之间的双边数据对齐；
   **论文未报告**任何成本金额（只说「小时级完成、相对于一个数周实验的成本只是一小部分」，
   ⑥ Q32），也未报告 token 消耗、人力投入或企业侧收益数字。本卡 ⑤ 段因此只给公式，不给金额。

---

## ② 母婴出海应用案例

### 场景 1：旺季前，对跨境新品详情页的候选改版做事前粗筛（不替代实验）

- **业务问题**：母婴出海店铺（吸奶器、奶瓶、婴儿背带），Amazon 美国站 + 独立站双端。新品上架前
  手里有若干套详情页 / 主图候选（卖点顺序、A+ 模块排布、优惠券位是否前置）。站点日流量不够，
  **跑满一个 A/B 通常要按周计**（业务侧经验值，论文未报告任何周期数字），而旺季前的试错窗口只有
  一次。于是先用数据驱动人格 agent 对候选方案做**事前粗筛**：它只回答方向（哪个方案在 CTR 或
  订阅意图上方向为正、哪个不可判），把最有希望的 1–2 套推进真实验，其余留在下一轮。
- **数据要求**：① **人格侧**：匿名化、聚合后的行为信号（会话、点击、加购、购买、优惠券使用、
  评论浏览），以「用户 × 月」为粒度，历史长度建议覆盖近半年且尽量跨一个完整旺季周期；
  按论文门槛，单人至少 20 笔交易记录，否则该人格应被剔除（⑥ Q23）。② **测试侧**：
  每套候选变体的**截图** + 目标度量（CTR 或订阅/注册意图），控件与实验件成对提交。
  ③ **校准侧**：历史上已经跑完的真实 A/B 结果，用来估计本店人格池的实际方向准确率
  （论文的 benchmark 标签未公开，⑥ Q38，必须自己积累）。
- **数据可得性**：`部分可得（需补充 X）`。独立站侧（Shopify / GA4 / 服务端埋点）第一方行为信号
  可得；**Amazon 侧只有聚合广告报表，拿不到个体行为信号**，此时按论文结论改用**同类目公开
  电商行为数据**做人格 —— 论文实测公开电商数据在订阅类测试上 0.90，**高于**平台自有数据的 0.80
  （⑥ Q18），这是拿不到第一方数据时最有用的一条；需要补充的是「同类目公开数据的选品范围与本店
  品类对齐」这一判断，以及历史真实 A/B 的积累。
- **预期产出**：候选方案的**方向排序表**（每套方案给出预测方向 + 置信度 q + SignOv），
  以及一条硬规则：预测为「不可判」的方案直接砍掉，不浪费流量。若人格池规模大，可用子采样把
  人格数降到 500 以内以压低推理成本（论文在 500 人格时准确率与全池差异在 1pp 内，
  ⑥ Q28；极端预算下 100 人格仍可用，⑥ Q29）。
- **业务价值**：见 ⑤ 的 ROI 公式。价值来自两处：把**必然更差**的候选挡在实验之外（省下的是流量
  与等待周期，不是「提升幅度」），以及把候选排成优先级，让有限流量先验证最有希望的方案。

### 场景 2：按测试类型给「行为深度 vs 群体多样性」分工（人格池要不要牺牲多样性）

- **业务问题**：母婴品类里，行为信号最丰富的是多次购买的老客，但老客的偏好不代表新客与单次买家。
  团队通常为了「数据质量」只取样高活跃用户。论文给出了可操作的取舍：**深度池在 CTR 类测试上
  显著更好**（0.75 vs. 0.60），但在订阅类测试上代表池与深度池**没有显著差异**（两者都是 0.80），
  因为人口统计覆盖度在聚合层面补偿了行为深度的不足（⑥ Q24 / Q27）。所以正确做法不是二选一，
  而是**按测试类型分工**：主图 / 价格标签 / 优惠券位这类 CTR 测试优先用深度池；会员 / 订阅 /
  注册页这类测试不必牺牲多样性去堆深度。
- **数据要求**：两套人格池并存的构造能力 —— 深度池（高活跃、行为密集）与代表池（按人群分层抽样，
  覆盖新客 / 单次买家 / 不同宝宝月龄段的家庭）；每个人格要带**行为深度指标**（交易笔数）与
  **人口统计覆盖报告**（用来发现缺口）。
- **数据可得性**：`部分可得（需补充 X）`。深度池来自自有独立站与平台后台可得的行为日志；
  代表池需要跨人群分层抽样能力，**平台侧拿不到个体级人群构成**，需用公开电商行为数据或自有独立站
  用户补齐；且必须正视论文的发现——两池的人口统计缺口来自人格生成时的 LLM 推断偏差，
  **加样本量解决不了**（⑥ Q26），只能靠监控缺口 + 人工校准。
- **预期产出**：一张「测试类型 → 用哪个人格池」的对照表，外加每个池子按度量类型分别记录的方向
  准确率看板（CTR 与订阅两列分开，不合并）。
- **业务价值**：避免用单一人格池跑所有测试导致的系统性方向误判；把「要不要为了数据质量牺牲人群
  覆盖」这个反复争论的问题，变成一个可以按测试类型分别决策的规则。

---

## ③ 代码模板

- **依赖**：仅标准库（`math` / `random`），无第三方包、无网络、无 LLM SDK —— 因此在断网环境下也
  能被 K1 门禁真实执行（L4 跑通 + L5 断言全绿）。
- **结构**：合成行为信号 → 构造人格 → 配对评分提问 → 规则替身作答 → 解析分数 → 聚合方向 →
  按度量类型分别算方向准确率 → 断言测试 → 业务演示。
- **与论文的对应**：`build_pairwise_rating_prompt` 对应 §I.4 的配对评分模板；
  `parse_pairwise_answer` 对应 §I.5 的抽分；`persona_relative_delta` 逐字实现 §I.5 的相对效应式子；
  `aggregate_to_direction` 实现 §3.2 的聚合与 §3.6 的 q = Φ(δ/σ)；`make_benchmark` 对应 §4.1
  的基准构造（成对变体 + 剔除方向含糊的测试）。
- ⚠️ **本模板里的 agent 是确定性规则替身，不是论文的 LLM agent。** 论文用 Claude Sonnet 4.5
  （temperature 0）逐个人格生成 1–10 分与理由；本模板用可复现的规则函数产出**同样格式**的回答
  （`widget_1:7, widget_2:4`），目的是让这条链路可离线执行、可断言。**因此本模板打印出的任何准确率
  都不是论文的 0.75–0.90，也不构成对论文实验的复现。**
- ⚠️ **代码内所有数据均为合成数据**（`make_benchmark` / `make_behavior_signals`），不对应任何
  真实平台或企业。演示输出中的准确率与论文报告的数值处于同一量级，**纯属本模板合成世界设定的
  巧合，二者不可互相印证**。
- ⚠️ 断言「域外人格表现低于域内人格」是**本模板合成世界的内生性质**（ground truth 与域内人格
  共用同一套偏好结构，域外人格的偏好映射被设定为不一致）；它验证的是「这条链路能把域对齐的差异
  测出来」，**不是**复现论文实测的差距。

```python
# -*- coding: utf-8 -*-
"""
数据驱动人格 agent 的 A/B 事前仿真：配对评分提问 → 方向预测 → 方向准确率
论文：2609.01038 "Data-Driven Persona-Conditioned Agents for A/B Test Simulation"
      §3.3 提问设计 / §3.4 人格构造 / §3.6 评测指标 / §4.1 基准构造 / §I.4–I.5 提问与抽分模板

⚠️ 重要声明（务必先读）
  本模板里的 "agent" 是**确定性规则替身**，不是论文里的 LLM agent。
  论文用 Claude Sonnet 4.5（temperature 0）逐个人格生成 1–10 分与理由；
  本模板用可复现的规则函数产出**同样格式**的回答（"widget_1:7, widget_2:4"），
  目的是把「行为信号 → 人格 → 配对提问 → 逐人格评分 → 方向预测 → 方向准确率」
  这条链路做成**可离线执行、可断言**的代码。
  因此：本模板打印出的任何准确率都**不是**论文的 0.75–0.90，也不构成对论文实验的复现，
  更不构成「仿真可以替代真实 A/B」的证据。

依赖：仅标准库（math / random），无网络、无 LLM SDK。
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# 1. 任务对象：变体（控件 / 实验件）与两类度量
# ---------------------------------------------------------------------------
ATTRS = ("vividness", "info_clarity", "price_prominence", "trust_signal")
METRICS = ("ctr", "subscriptions")

# 「真实人群」在两类度量上的偏好权重。这是**本模板的合成世界设定**，
# 用来定义 ground truth 的方向；论文的 ground truth 来自真实实验的置信区间（§4.1）。
TRUE_WEIGHTS = {
    "ctr": {"vividness": 0.30, "info_clarity": 0.30,
            "price_prominence": 0.20, "trust_signal": 0.20},
    "subscriptions": {"vividness": 0.10, "info_clarity": 0.15,
                      "price_prominence": 0.30, "trust_signal": 0.45},
}


def make_variant(**attrs: float) -> dict:
    """一个视觉变体：四个属性都归一化到 [-1, 1]（0 = 中性）。"""
    v = {a: float(attrs.get(a, 0.0)) for a in ATTRS}
    for a, x in v.items():
        if not -1.0 <= x <= 1.0:
            raise ValueError(f"属性 {a} 必须在 [-1, 1]，收到 {x}")
    return v


def _phi(x: float) -> float:
    """标准正态 CDF —— 论文 §3.6 用 p = Φ(δ/σ) 表示「效应为正」的概率。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def make_benchmark(n_per_metric: int = 20, seed: int = 11) -> list:
    """构造一个小型 A/B 基准：每类度量 n_per_metric 个测试。

    对应论文 §4.1：每个测试是「控件 vs 一个实验件」，
    ground truth 是相对效应的正态分布；效应近零的「不可判定」测试被剔除
    （论文用阈值 τ=0.8 / 0.7 剔除含糊样本，本模板用 |p-0.5| 的近似门槛）。
    """
    rng = random.Random(seed)
    tests = []
    for metric in METRICS:
        made = 0
        while made < n_per_metric:
            control = {a: rng.uniform(-1.0, 1.0) for a in ATTRS}
            treatment = {a: rng.uniform(-1.0, 1.0) for a in ATTRS}
            diff = sum(TRUE_WEIGHTS[metric][a] * (treatment[a] - control[a])
                       for a in ATTRS) * len(ATTRS)
            se_true = rng.uniform(0.06, 0.14)
            p_true = _phi(diff / se_true)
            if abs(p_true - 0.5) < 0.25:      # 剔除方向含糊的测试
                continue
            tests.append({
                "test_id": f"{metric}-{made:02d}",
                "metric": metric,
                "control": make_variant(**control),
                "treatment": make_variant(**treatment),
                "p_true": p_true,
                "direction": 1 if p_true > 0.5 else -1,
            })
            made += 1
    return tests


# ---------------------------------------------------------------------------
# 2. 机制 (a)：从匿名化行为信号构造人格（论文 §3.4 / §I.3 的机制，此处为规则替身）
# ---------------------------------------------------------------------------
IN_DOMAIN_SOURCES = ("platform_behavioral", "open_ecommerce")
OUT_OF_DOMAIN_SOURCES = ("movie_ratings", "survey")


def make_behavior_signals(source: str, rng: random.Random) -> dict:
    """合成一份「匿名化、聚合后」的行为信号（真实场景里来自埋点/交易台账）。"""
    if source == "platform_behavioral":        # 平台自有行为日志：密度高
        sessions = rng.randint(40, 120)
        clicks = int(sessions * rng.uniform(2.0, 6.0))
        purchases = rng.randint(4, 40)
        coupon_uses = rng.randint(0, purchases)
        review_reads = rng.randint(5, 60)
        returns = rng.randint(0, max(1, purchases // 4))
    elif source == "open_ecommerce":           # 公开电商购买史：只有交易，没有点击流
        sessions = rng.randint(5, 30)
        clicks = int(sessions * rng.uniform(1.0, 3.0))
        purchases = rng.randint(3, 25)
        coupon_uses = rng.randint(0, purchases)
        review_reads = rng.randint(0, 12)
        returns = rng.randint(0, max(1, purchases // 5))
    elif source == "movie_ratings":            # 娱乐偏好：与购物决策不同域
        sessions = rng.randint(5, 40)
        clicks = int(sessions * rng.uniform(1.0, 4.0))
        purchases = rng.randint(0, 3)
        coupon_uses = 0
        review_reads = rng.randint(0, 3)
        returns = 0
    else:                                      # survey：只有问卷式人口统计，无行为深度
        sessions = rng.randint(0, 3)
        clicks = rng.randint(0, 5)
        purchases = rng.randint(0, 2)
        coupon_uses = 0
        review_reads = rng.randint(0, 2)
        returns = 0
    return {"sessions": sessions, "clicks": clicks, "purchases": purchases,
            "coupon_uses": coupon_uses, "review_reads": review_reads,
            "returns": returns, "avg_basket": rng.uniform(15.0, 120.0)}


def _normalize(d: dict) -> dict:
    total = sum(max(0.0, v) for v in d.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in d.items()}


def build_persona(persona_id: str, source: str, signals: dict,
                  rng: random.Random) -> dict:
    """把行为信号压成行为统计，再映射为「按度量分」的属性权重。

    ⚠️ 这一步在论文里是**两次 LLM 调用**（人口统计流 + 购物特征流，§I.2），
    产出一段自然语言人格档案；这里用规则把同样的信号压成权重向量，
    因此**不含**论文所报告的任何人口统计推断结果。
    """
    sessions = max(1, signals["sessions"])
    engagement = min(1.0, signals["clicks"] / sessions / 6.0)
    deal_seeking = min(1.0, signals["coupon_uses"] / max(1, signals["purchases"]))
    trust_orientation = min(1.0, signals["review_reads"] / max(1, signals["clicks"]))
    retention = min(1.0, signals["purchases"] / 20.0)

    # 行为统计 → 属性偏好上的「倾斜系数」：同类行为信号在不同属性上被加权
    tilt = {"vividness": engagement,
            "info_clarity": 0.5 * trust_orientation + 0.5 * retention,
            "price_prominence": deal_seeking,
            "trust_signal": 0.7 * trust_orientation + 0.3 * retention}

    if source in IN_DOMAIN_SOURCES:
        # 域内：行为信号在**真实人群偏好结构**（TRUE_WEIGHTS）上做个体化倾斜。
        # 这正是「域对齐」的含义 —— 数据源与业务同域，所以信号能落到同一套属性上。
        w = {m: {a: TRUE_WEIGHTS[m][a] * (0.70 + 0.60 * tilt[a]) for a in ATTRS}
             for m in METRICS}
        noise = 0.06
    elif source == "movie_ratings":
        # 域外（娱乐偏好）：只看「吸睛/娱乐性」，价格与信任信号几乎不被使用
        # 「只挑最吸睛的」—— 娱乐偏好里被奖励的是视觉刺激，不是信息与信任信号
        w = {m: {"vividness": 0.55, "info_clarity": 0.15,
                 "price_prominence": 0.15, "trust_signal": 0.15} for m in METRICS}
        noise = 0.10
    else:
        # 域外（问卷人口统计）：没有行为深度，退回「表面显著性」泛化启发式 ——
        # 对应论文的 demographics-only 退化情形（行为档案缺失时 LLM 退回泛化推理）。
        # 注意它不是「随机」：泛化启发式会**系统性地**高估视觉刺激、低估信任/价格信号，
        # 所以它在订阅类测试上会稳定跑偏，而不是随机错。
        w = {m: {"vividness": 0.40, "info_clarity": 0.20,
                 "price_prominence": 0.20, "trust_signal": 0.20} for m in METRICS}
        noise = 0.10

    weights = {m: _normalize({a: v * (1.0 + rng.gauss(0.0, noise)) for a, v in w[m].items()})
               for m in METRICS}
    return {"persona_id": persona_id, "source": source, "signals": signals,
            "weights": weights, "noise": noise,
            "behavioral_depth": sessions * max(1, signals["purchases"])}


def make_persona_pool(sources, n_per_source: int, seed: int) -> list:
    """按数据源各造 n_per_source 个人格（真实场景里对应「人格池」）。"""
    rng = random.Random(seed)
    pool = []
    for source in sources:
        for i in range(n_per_source):
            sig = make_behavior_signals(source, rng)
            pool.append(build_persona(f"{source}-{i:03d}", source, sig, rng))
    return pool


# ---------------------------------------------------------------------------
# 3. 机制 (b)：用「配对评分」提结构化问题（论文 §3.3 / §I.4）
# ---------------------------------------------------------------------------
def build_pairwise_rating_prompt(persona: dict, control: dict, treatment: dict,
                                 metric: str) -> str:
    """配对评分提问模板：两个变体同屏 + 1–10 分档 + JSON 输出。

    与论文 §I.4 的模板同构（system 角色扮演 + 任务 + 合法 JSON + <options>），
    论文用 base64 图片承载变体，本模板用属性向量字符串代替。
    """
    metric_hint = ("点击意愿（会不会点进去看）" if metric == "ctr"
                   else "订阅意愿（愿不愿意留下邮箱/开通订阅）")
    return (
        "System: You are an AI assistant role-playing a customer with specific characteristics.\n"
        f"Persona: {persona['persona_id']} (source={persona['source']}, "
        f"behavioral_depth={persona['behavioral_depth']})\n"
        "User: 1. 你的角色背景见上。\n"
        f"2. 任务：针对「{metric_hint}」，给下面两个视觉方案各打 1–10 分。\n"
        '3. 输出格式（合法 JSON）：{"reason": "...", "answer": "widget_1:7, widget_2:4"}\n'
        "<options>\n"
        f" 1: widget_1 [control]   {_render_variant(control)}\n"
        f" 2: widget_2 [treatment] {_render_variant(treatment)}\n"
        "</options>"
    )


def _render_variant(v: dict) -> str:
    return " ".join(f"{a}={v[a]:+.2f}" for a in ATTRS)


def rule_agent_answer(persona: dict, prompt: str, control: dict, treatment: dict,
                      metric: str, rng: random.Random) -> dict:
    """确定性规则替身：把「结构化提问」变成一个同样格式的 JSON 回答。

    ⚠️ 这不是论文的 LLM agent。真实实现里这里应当是一次 LLM 调用
    （temperature 0、JSON 约束、逐人格随机变体顺序）；本替身只保证
    **接口与输出格式一致**，使替换成真 LLM 时下游代码无需改动。
    """
    scores = {}
    for label, variant in (("widget_1", control), ("widget_2", treatment)):
        w = persona["weights"][metric]
        raw = sum(w[a] * variant[a] for a in ATTRS)          # ∈ [-1, 1]
        score = 5.0 + 4.0 * raw + rng.gauss(0.0, persona["noise"])
        scores[label] = min(10.0, max(1.0, round(score * 2.0) / 2.0))  # 0.5 分档
    return {"reason": f"按 {persona['persona_id']} 的偏好权衡 {metric}",
            "answer": f"widget_1:{scores['widget_1']}, widget_2:{scores['widget_2']}"}


def parse_pairwise_answer(answer: str) -> dict:
    """按论文 §I.5 抽分：解析 widget_name:score 对。"""
    out = {}
    for chunk in answer.split(","):
        if ":" not in chunk:
            continue
        name, _, val = chunk.partition(":")
        out[name.strip()] = float(val.strip())
    if len(out) != 2:
        raise ValueError(f"配对评分回答应含 2 个分数，实际拿到 {out}")
    return out


# ---------------------------------------------------------------------------
# 4. 机制 (c)：把逐人格评分聚合成 A/B 方向预测（论文 §3.2 / §3.6 / §I.5）
# ---------------------------------------------------------------------------
def persona_relative_delta(scores: dict) -> float:
    """单个人格的相对效应：(s_treat - s_ref) / s_ref —— 论文 §I.5 的式子。"""
    ref, treat = scores["widget_1"], scores["widget_2"]
    if ref <= 0.0:
        raise ValueError("参考方案得分必须为正")
    return (treat - ref) / ref


def aggregate_to_direction(deltas: list) -> dict:
    """聚合逐人格相对效应 → 预测分布 N(delta, se) → q = Φ(delta / se)。

    论文 §3.2：δ̂_s = 各人格得分之差的均值，σ̂_s = 该分布的标准误；
    §3.6：Acc = I[(p-0.5)(q-0.5) > 0]，即两个分布是否同意方向。
    """
    if not deltas:
        raise ValueError("没有可聚合的人格判断")
    n = len(deltas)
    mean = sum(deltas) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("inf")
    q = _phi(mean / se) if se > 0 else (1.0 if mean > 0 else 0.0)
    return {"delta": mean, "se": se, "q": q, "n_personas": n,
            "direction": 1 if mean > 0 else (-1 if mean < 0 else 0)}


def simulate_test(test: dict, personas: list, seed: int = 0) -> dict:
    """用一个人格池仿真一个 A/B 测试，返回预测方向与置信度。"""
    rng = random.Random(seed)
    deltas = []
    for persona in personas:
        prompt = build_pairwise_rating_prompt(
            persona, test["control"], test["treatment"], test["metric"])
        answer = rule_agent_answer(persona, prompt, test["control"],
                                   test["treatment"], test["metric"], rng)
        deltas.append(persona_relative_delta(parse_pairwise_answer(answer["answer"])))
    agg = aggregate_to_direction(deltas)
    agg["test_id"] = test["test_id"]
    agg["metric"] = test["metric"]
    agg["correct"] = int((agg["q"] - 0.5) * (test["p_true"] - 0.5) > 0)
    return agg


def directional_accuracy(results: list) -> float:
    """方向准确率 —— 论文的核心指标（Acc）。只应喂**同一类度量**的结果。"""
    if not results:
        raise ValueError("结果为空")
    assert len({r["metric"] for r in results}) == 1, \
        "方向准确率必须按度量类型分别计算，不能把 CTR 与 subscription 混成一个数"
    return sum(r["correct"] for r in results) / len(results)


def run_benchmark(tests: list, persona_by_pool: dict, seed: int = 7) -> dict:
    """对每个（人格池 × 度量类型）分别跑仿真并报告，**绝不合并两类度量**。"""
    report = {}
    for pool_name, personas in persona_by_pool.items():
        report[pool_name] = {}
        for metric in METRICS:
            subset = [t for t in tests if t["metric"] == metric]
            results = [simulate_test(t, personas, seed=seed) for t in subset]
            report[pool_name][metric] = {
                "accuracy": directional_accuracy(results),
                "n_tests": len(results),
                "test_ids": [r["test_id"] for r in results],
                "pool_size": len(personas),
            }
    return report


# ---------------------------------------------------------------------------
# 5. 测试（K1 L5 断言）
# ---------------------------------------------------------------------------
def test_pairwise_prompt_and_answer_schema():
    """机制 (b)：配对评分提问的产出必须能被 §I.5 的抽分器解析回来。"""
    rng = random.Random(0)
    persona = build_persona("p-0", "platform_behavioral",
                            make_behavior_signals("platform_behavioral", rng), rng)
    control = make_variant(vividness=-0.5, info_clarity=0.0,
                           price_prominence=0.2, trust_signal=0.3)
    treatment = make_variant(vividness=0.8, info_clarity=0.4,
                             price_prominence=0.5, trust_signal=0.6)
    prompt = build_pairwise_rating_prompt(persona, control, treatment, "ctr")
    # 两个变体必须同屏出现（配对格式），且给的是 1–10 分档而非是非题
    assert "widget_1" in prompt and "widget_2" in prompt
    assert "1–10 分" in prompt and "answer" in prompt
    ans = rule_agent_answer(persona, prompt, control, treatment, "ctr", rng)
    scores = parse_pairwise_answer(ans["answer"])
    assert set(scores) == {"widget_1", "widget_2"}
    assert all(1.0 <= v <= 10.0 for v in scores.values())
    # 打分器必须拒绝残缺回答（防止把自由文本当分数）
    try:
        parse_pairwise_answer("widget_1:7")
        raise AssertionError("残缺回答本应报错")
    except ValueError:
        pass


def test_aggregation_gives_correct_direction():
    """机制 (c)：实验件在所有属性上严格占优时，聚合方向必须为正。"""
    rng = random.Random(1)
    personas = make_persona_pool(IN_DOMAIN_SOURCES, 8, seed=1)
    control = make_variant(**{a: -0.6 for a in ATTRS})
    treatment = make_variant(**{a: 0.6 for a in ATTRS})
    deltas = []
    for persona in personas:
        ans = rule_agent_answer(persona, "prompt", control, treatment, "ctr", rng)
        deltas.append(persona_relative_delta(parse_pairwise_answer(ans["answer"])))
    agg = aggregate_to_direction(deltas)
    assert agg["direction"] == 1 and agg["q"] > 0.5
    assert agg["n_personas"] == len(personas)
    # 聚合值必须等于论文 §I.5 的式子（手工重算一遍）
    manual = sum(deltas) / len(deltas)
    assert abs(agg["delta"] - manual) < 1e-12
    # 反过来把两个变体对调，方向必须翻转
    flipped = aggregate_to_direction([-d for d in deltas])
    assert flipped["direction"] == -1 and flipped["q"] < 0.5


def test_out_of_domain_personas_underperform_in_domain():
    """机制 (d) 的核心限定：人格数据源与业务域不对齐时，方向准确率会掉。

    ⚠️ 这里的「域外更差」是**本模板合成世界的内生性质**：ground truth 由
    TRUE_WEIGHTS 定义，而域内人格的行为信号正是按这套权重映射的。
    该断言验证的是「这条链路能把域对齐的差异测出来」，
    **不是**复现论文实测的 0.75–0.90 / 0.57–0.69 差距。
    """
    tests = make_benchmark(n_per_metric=20, seed=11)
    pools = {
        "in_domain": make_persona_pool(IN_DOMAIN_SOURCES, 10, seed=2),
        "out_of_domain": make_persona_pool(OUT_OF_DOMAIN_SOURCES, 10, seed=3),
    }
    report = run_benchmark(tests, pools, seed=7)
    for metric in METRICS:                      # 两类度量分别比较，不合并
        acc_in = report["in_domain"][metric]["accuracy"]
        acc_out = report["out_of_domain"][metric]["accuracy"]
        assert acc_in > acc_out, f"{metric}: 域内 {acc_in} 未高于域外 {acc_out}"
    # 单人固定种子的合成世界必须完全可复现
    assert report == run_benchmark(tests, pools, seed=7)


def test_metrics_are_reported_separately():
    """机制 (d)：CTR 与 subscription 必须分开报告，且各自只在同类测试上计算。"""
    tests = make_benchmark(n_per_metric=20, seed=11)
    n_ctr = sum(1 for t in tests if t["metric"] == "ctr")
    n_sub = sum(1 for t in tests if t["metric"] == "subscriptions")
    assert n_ctr == 20 and n_sub == 20
    pools = {"in_domain": make_persona_pool(IN_DOMAIN_SOURCES, 10, seed=2)}
    report = run_benchmark(tests, pools, seed=7)
    ctr, sub = report["in_domain"]["ctr"], report["in_domain"]["subscriptions"]
    assert ctr["n_tests"] == n_ctr and sub["n_tests"] == n_sub
    assert set(ctr["test_ids"]).isdisjoint(set(sub["test_ids"]))
    assert 0.0 <= ctr["accuracy"] <= 1.0 and 0.0 <= sub["accuracy"] <= 1.0
    # 把两类度量混进同一次调用必须被拦住 —— 否则「哪类指标掉了」会被平均掉
    mixed = [simulate_test(t, pools["in_domain"], seed=7) for t in tests[:1]] + \
            [simulate_test(t, pools["in_domain"], seed=7)
             for t in tests if t["metric"] == "subscriptions"][:1]
    assert len({r["metric"] for r in mixed}) == 2
    try:
        directional_accuracy(mixed)
        raise AssertionError("混度量调用本应断言失败")
    except AssertionError as exc:
        assert "分别计算" in str(exc)


def test_sparse_personas_degrade_more_than_deep_ones():
    """深度 vs 多样性（论文 §5.3 的机制示意）：信号越稀疏，规则替身越容易失准。"""
    tests = [t for t in make_benchmark(n_per_metric=20, seed=11) if t["metric"] == "ctr"]
    rng = random.Random(5)
    deep = [build_persona(f"d{i}", "platform_behavioral",
                          {"sessions": 100, "clicks": 500, "purchases": 35,
                           "coupon_uses": 12, "review_reads": 50, "returns": 2,
                           "avg_basket": 60.0}, rng) for i in range(10)]
    sparse = [build_persona(f"s{i}", "platform_behavioral",
                            {"sessions": 1, "clicks": 1, "purchases": 1,
                             "coupon_uses": 0, "review_reads": 0, "returns": 0,
                             "avg_basket": 40.0}, rng) for i in range(10)]
    acc_deep = directional_accuracy([simulate_test(t, deep, seed=7) for t in tests])
    acc_sparse = directional_accuracy([simulate_test(t, sparse, seed=7) for t in tests])
    assert acc_deep >= acc_sparse


# ---------------------------------------------------------------------------
# 6. 业务演示（本模板的输出是**合成世界**的结果，不是论文数字）
# ---------------------------------------------------------------------------
def _demo() -> None:
    tests = make_benchmark(n_per_metric=20, seed=11)
    pools = {
        "域内人格池(平台行为+公开电商)": make_persona_pool(IN_DOMAIN_SOURCES, 10, seed=2),
        "域外人格池(影视评分+问卷)": make_persona_pool(OUT_OF_DOMAIN_SOURCES, 10, seed=3),
    }
    report = run_benchmark(tests, pools, seed=7)
    print("== 事前粗筛仿真（规则替身，合成数据；不是论文的 LLM agent）==")
    for pool_name, per_metric in report.items():
        print(f"[{pool_name}]  人格数={per_metric['ctr']['pool_size']}")
        for metric in METRICS:
            r = per_metric[metric]
            print(f"   {metric:14s} 方向准确率={r['accuracy']:.2f}  (n={r['n_tests']} 个测试)")
    print("\n== 单个人格的结构化提问与回答（机制示意）==")
    persona = pools["域内人格池(平台行为+公开电商)"][0]
    t = tests[0]
    print(build_pairwise_rating_prompt(persona, t["control"], t["treatment"], t["metric"]))
    ans = rule_agent_answer(persona, "", t["control"], t["treatment"], t["metric"],
                            random.Random(0))
    print("回答:", ans)
    print("聚合成:", aggregate_to_direction([persona_relative_delta(
        parse_pairwise_answer(ans["answer"]))]))
    print("\n注意：以上准确率来自本模板的合成世界，与论文报告的 0.75–0.90 数值相近纯属巧合，"
          "二者不可互相印证。")


if __name__ == "__main__":
    _demo()
```

---

## ④ 技能关联

- **前置｜`Skill-AB-Experimental-Design.md`**（同目录）：本卡输出的是**候选方案的方向预判与排序**，
  真实验仍然要按那张卡的设计口径跑（随机化单元、主指标、护栏指标、多重比较）。
  数据流：本卡把候选集从「全部」缩到「少数几个」，再交给实验设计。
- **前置｜`Skill-Power-Analysis-Sample-Size.md`**（同目录）：本卡为什么存在，正是由那张卡的
  结论决定的 —— 当站点流量撑不起最小可检测效应时，实验要么跑很久、要么几乎没有功效。
  组合方式：先用本卡把效应量明显偏小 / 方向为负的候选筛掉，剩下的再算样本量。
- **下游｜`Skill-AB-Test-Result-Interpretation.md`**（同目录）：真实验跑完后由那张卡负责读数
  （显著性、置信区间、护栏指标）。本卡的预测值应当与真实结果**逐条对账**，用累积的历史结果
  估计本店人格池的实际方向准确率——这是判断「本卡的粗筛结论在本店还能不能信」的唯一办法。
- **对照｜`Skill-Multi-Armed-Bandit.md`**（同目录）：多臂老虎机用**真实流量**在线分配探索预算，
  回答「把流量动态分给更好的臂」；本卡在**没有任何新流量**的前提下离线预判方向。
  两者互补：本卡先筛候选，再交给 bandit / 固定流量实验去验证。
- **校准对照｜`Skill-Incrementality-Measurement.md`**（14-用户分析）：那张卡测的是**增量**
  （被助攻的乌龙球），本卡预测的是**指标方向**。二者必须分清：仿真说「方案 B 的 CTR 更高」，
  不等于「上方案 B 能带来增量」。若企业已经有增量实验能力，应当用增量口径给本卡的预测做最终裁决。

---

## ⑤ 商业价值评估

**ROI 公式**（本卡不代入任何未经论文或企业数据支持的数字）：

ROI = (避免的无效实验成本 + 提前止损的收益 − 建设与运行成本) / (建设与运行成本)，
其中 `避免的无效实验成本 = N_筛掉 × C_单次实验`，`C_单次实验 = 单次 A/B 占用的流量机会成本 × 观察周期`。

| 参数 | 含义 | 来源 |
|---|---|---|
| `N_筛掉` | 被粗筛挡在实验之外的候选方案数 | 企业自有候选池 |
| `C_单次实验` | 单个 A/B 占用的流量机会成本 × 观察周期 | **论文未报告任何成本量级**；须企业自估（把站点流量、周期、单 UV 期望毛利代进去） |
| `建设与运行成本` | 人格池构造（历史行为信号 → 结构化人格）+ 逐候选推理 + 对账看板 | **论文未报告金额**；论文只给了一条可以用于估算的定性线索：批量推理下整轮仿真可在**小时级**完成，成本相对一个**数周**的实验只是一小部分（⑥ Q32）；子采样到 500 人格可再省一次推理成本（论文报告 up to 2× 成本下降，⑥ Q28） |
| `方向准确率` | 决定这条链路值不值得信的核心参数 | **必须企业自测**。论文给出的参考是 40 个真实 A/B、两类度量上的 0.75–0.90（⑥ Q12），但它是**平台自有数据 + 域内公开电商数据**的成绩，且随数据源与度量类型大幅波动（0.90 来自公开电商人格 × 订阅测试，见 ⑥ Q18；域外来源掉到 0.57–0.69，见 ⑥ Q22）。**不能直接拿 0.75–0.90 当本店预期。** |

**保守结论**：论文没有给任何金额、也没有给母婴/跨境场景的任何数字，所以本卡**不给收益金额结论**。
这条链路的价值主张是**决策口径**：把「凭经验挑一个方案上实验」换成「先按方向排序、把明显更差的
挡在实验之外」，并用真实结果持续校准准确率。若本店积累的对账准确率长期接近随机水平，
应当**停用**这条链路（这正是 ①b 想拦住的情况）。

- **实施难度**：⭐⭐⭐☆☆ —— 链路本身简单（提问模板 + 抽分 + 聚合），难点在两处：
  ① 人格池的数据准备与人口统计偏差校准；② 必须有历史真实 A/B 结果做对账，否则无法知道本店
  的准确率水平。③ 段的模板可作为最小可行验证先跑通链路。
- **优先级**：⭐⭐⭐⭐☆ —— 对**流量小、周期长**的跨境母婴店优先级更高：单次实验的机会成本越大，
  事前粗筛的相对价值越高。若站点流量充足、每天都能跑完一个实验，优先级降到 2–3 星（直接做实验
  更划算）。
- **评估依据**：收益侧的不确定性远大于工程侧（论文未给母婴场景任何数字），且这条链路**只能减少
  无效实验、不能创造增量**；因此本卡的价值主张是**筛选效率**，而不是承诺提升幅度。

---

## ⑥ 原文引用

本段每条引文均从 `fulltext.md` 按起止锚点**逐字切出**（连续子串，无拼接、无改写）；
全部经 `quote_check.py` 判定 `VERBATIM`。

### A. 为什么需要它：A/B 的流量与周期成本，以及「只做粗筛」的定位

> 原文："Online controlled experiments remain the gold standard for validating product changes, yet each test requires sufficient user traffic, engineering effort, and typically weeks of data collection to reach statistical significance (Kohavi et al., 2009). These costs limit how many ideas teams can evaluate."
> 出处：2609.01038 §1 Introduction｜Q1

> 原文："A particularly compelling application is the simulation of online controlled experiments (A/B tests): if persona-conditioned agents can reliably predict whether users prefer a treatment variant over a control, teams could pre-screen design candidates offline—reducing the time, traffic, and experimentation cost (Rieder et al., 2026; Castelo et al., 2026)."
> 出处：2609.01038 §1 Introduction｜Q2

> 原文："With current accuracy levels, the proposed framework cannot fully replace human A/B tests—but it does not need to."
> 出处：2609.01038 §6 Discussion · Potential applications｜Q3

> 原文："A potential application could be a pre-screening tool that filters clearly inferior treatment candidates before they consume traffic and prioritizes the experiments by ranking proposed changes by predicted impact."
> 出处：2609.01038 §6 Discussion · Potential applications｜Q4

### B. 方法：结构化提问 + 配对评分 + 方向聚合

> 原文："We frame A/B test simulation as a structured question task: each persona-conditioned agent is presented with variant screenshots and asked to evaluate them with respect to a target metric."
> 出处：2609.01038 §B.1 Question Design: Designs Description｜Q5

> 原文："We study four formats varying along two axes—isolation vs. comparison (whether the agent sees one variant or both) and binary vs. rating (whether the response is yes/no or a 1–10 score)."
> 出处：2609.01038 §3.3 Question Design｜Q41

> 原文："In independent formats, each variant is shown separately; in pairwise formats, both variants are presented together with order randomized per persona to control positional bias."
> 出处：2609.01038 §3.3 Question Design｜Q6

> 原文："Independent formats perform poorly, suggesting agents struggle to calibrate scores without comparative context."
> 出处：2609.01038 §5.1 Question Design Comparison｜Q7

> 原文："The binary pairwise format achieves strong results on subscription tests but fails on CTR, indicating that optimal design depends on metric type."
> 出处：2609.01038 §5.1 Question Design Comparison｜Q8

> 原文："Pairwise rating is the most effective question format, achieving 0.75 accuracy on CTR and 0.80 on subscription tests. All subsequent experiments use this format."
> 出处：2609.01038 §5.1 Question Design Comparison · Takeaway｜Q9

> 原文："For each A/B test, we collect per-persona $(s_{ref},s_{treat})$ tuples and compute the predicted effect as $\hat{\delta}_{s}=\frac{1}{N}\sum_{i=1}^{N}\frac{s_{treat}^{(i)}-s_{ref}^{(i)}}{s_{ref}^{(i)}}$."
> 出处：2609.01038 §I.5 Score Extraction｜Q10

> 原文："Accuracy (Acc): $\mathbb{I}[(p{-}0.5)(q{-}0.5)>0]$; Sign overlap (SignOv): $1-|p-q|$; Sign Bhattacharyya (SignBC): $(\sqrt{pq}+\sqrt{(1{-}p)(1{-}q)})^{2}$."
> 出处：2609.01038 §3.6 Evaluation Metrics｜Q11

### C. 基准：40 个测试、两类度量

> 原文："On a benchmark of 40 A/B tests spanning two metric types, our best configuration achieves 0.75–0.90 directional accuracy depending on the test metric, demonstrating that data-driven personas are a viable path toward fast, low-cost experiment pre-screening."
> 出处：2609.01038 §Abstract｜Q12

> 原文："We evaluate our framework on a benchmark of 40 A/B tests spanning two metric types—click-through rate (CTR) and subscriptions—and organize experiments around four research questions:"
> 出处：2609.01038 §1 Introduction｜Q13

> 原文："The original candidate set contained over 50 CTR tests and 40 subscription tests; applying these thresholds excluded tests with ambiguous ground truth, yielding the final benchmark of 40 tests (20 per metric)."
> 出处：2609.01038 §4.1 Benchmark Construction｜Q14

> 原文："The benchmark reflects a curated experimental sample and should not be interpreted as representative of any specific platform’s full user base or operational A/B testing infrastructure."
> 出处：2609.01038 §4.1 Benchmark Construction｜Q15

> 原文："The benchmark spans two metric types: click-through rate (engagement) and subscriptions (sign-up intent), evaluated with the same pipeline but different question framing."
> 出处：2609.01038 §4.1 Benchmark Construction｜Q16

### D. 人格数据源与域对齐（registry 的 0.75–0.90 分项来源）

> 原文："We compare two persona pools (both containing 935 personas)."
> 出处：2609.01038 §4.2 Personas Pool Construction｜Q17

> 原文："Among the external persona sources, open e-commerce data performs best and surpasses platform data on subscription tests (0.90 vs. 0.80 accuracy), likely due to domain alignment—e-commerce browsing and purchasing signals are directly relevant to evaluating widget engagement and subscription intent."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q18

> 原文："Rotten Tomatoes personas, grounded in entertainment preferences, show reasonable performance on CTR tests (0.65) but degrade on subscription tests (0.60), suggesting that out-of-domain behavioral data provides insufficient signal for metric-specific predictions."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q19

> 原文："Survey-based personas perform moderately without excelling on either metric."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q20

> 原文："Personas constructed from platform behavioral data achieve competitive results on CTR tests (0.70 accuracy, 0.64 SignOv) and competitive performance on subscription tests."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q21

> 原文："Domain alignment is crucial for personas effectiveness. In-domain behavioral data achieves 0.70–0.90 accuracy, while out-of-domain sources drop to 0.57–0.69. Public e-commerce data can rival platform-specific personas."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas · Takeaway｜Q22

> 原文："Domain alignment matters more than data volume or source exclusivity. Public e-commerce data rivals platform-specific personas (Table 2), lowering adoption barriers. However, below ${\sim}20$ recorded transactions, simulation degrades as the LLM defaults to generic reasoning."
> 出处：2609.01038 §6 Discussion · Data requirements｜Q23

### E. 行为深度 vs 群体多样性、子采样、消融

> 原文："The deep pool significantly outperforms the representative pool on CTR accuracy (0.75 vs. 0.60), while on subscription tests there is no statistically significant difference on any metric (0.80 accuracy for both)."
> 出处：2609.01038 §5.3 Persona Pool Comparison｜Q24

> 原文："We hypothesize that for sparse personas, the LLM lacks sufficient behavioral grounding and defaults to generic reasoning rather than user-specific preferences, explaining the CTR accuracy gap."
> 出处：2609.01038 §5.3 Persona Pool Comparison｜Q25

> 原文："These gaps likely arise from LLM inference biases during persona generation rather than sampling limitations, and represent a primary lever for future improvement."
> 出处：2609.01038 §5.3 Persona Pool Comparison｜Q26

> 原文："Behavioral depth yields a statistically significant advantage on CTR accuracy, but demographic diversity fully compensates on subscription tests (no significant difference on any metric)."
> 出处：2609.01038 §5.3 Persona Pool Comparison · Takeaway｜Q27

> 原文："All subsampling strategies preserve near-full-pool accuracy at 500 personas (within 1pp on CTR, matching or exceeding on subscriptions), potentially enabling up to 2$\times$ cost reduction."
> 出处：2609.01038 §5.4 Population Sampling Efficiency · Takeaway｜Q28

> 原文："Even at $n{=}100$, all algorithms remain competitive (0.77–0.80 CTR, 0.82–0.86 subscriptions), making routine simulation viable under constrained budgets."
> 出处：2609.01038 §6 Discussion · Cost–quality trade-off｜Q29

> 原文："Demographics-only degrade substantially on subscriptions (0.30 vs. 0.80), indicating that behavioral profiles are critical for higher-salience decisions."
> 出处：2609.01038 §5.5 Ablation: Component Contributions｜Q30

> 原文："A single generic persona does not outperform the unconditioned LLM (0.40 vs. 0.45 on CTR, 0.60 vs. 0.65 on subscriptions), confirming that population diversity—not mere persona framing—drives the improvement."
> 出处：2609.01038 §5.5 Ablation: Component Contributions｜Q31

### F. 边界与论文自承局限

> 原文："With batch inference, a full simulation could complete in hours at a fraction of a multi-week experiment cost, potentially enabling teams to explore a broader design space without proportionally increasing experimentation overhead."
> 出处：2609.01038 §6 Discussion · Potential applications｜Q32

> 原文："Our experiments show that simulations are most reliable when the underlying effect size is large, as is typical of high-salience decisions. Conversely, the framework is least trustworthy for near-zero effects where small perturbations flip the predicted direction. This suggests that outputs could be more useful as a ranking signal rather than a binary decision criterion."
> 出处：2609.01038 §6 Discussion · When to trust simulation｜Q33

> 原文："Our framework evaluates agents on isolated screenshots rather than full page contexts, removing situational factors (browsing intent, session history, surrounding content) that influence real user decisions."
> 出处：2609.01038 §Limitations｜Q34

> 原文："The benchmark is limited to 40 tests from a single e-commerce domain and two metric types; generalization to other domains, metric types, or multi-step user journeys remains untested."
> 出处：2609.01038 §Limitations｜Q35

> 原文："Persona demographics are inferred by the LLM from behavioral signals rather than self-reported, introducing systematic biases that may distort population-level predictions."
> 出处：2609.01038 §Limitations｜Q36

> 原文："LLM positivity bias and anchoring effects likely produce systematically optimistic treatment evaluations, a tendency that our sign-based metrics partially mask when both control and treatment are equally inflated."
> 出处：2609.01038 §Limitations｜Q37

> 原文："Finally, the benchmark ground-truth labels are not publicly released, limiting external reproducibility; however, the methodology is fully reproducible with any preference dataset, as we demonstrate with public data sources that produce competitive results."
> 出处：2609.01038 §Limitations｜Q38

> 原文："All experiments use a single LLM (Claude Sonnet 4.5); while we validate consistency across Claude Haiku and Opus (Appendix D), generalization to non-Anthropic models remains untested."
> 出处：2609.01038 §Limitations｜Q39

> 原文："These results suggest that data-driven persona simulation could serve as a viable tool for pre-screening A/B test candidates, potentially reducing wasted experimentation traffic while maintaining directional accuracy that may be sufficient for prioritization decisions."
> 出处：2609.01038 §7 Conclusions｜Q40
