---
title: Skill-Stateful-Skill-Runtime
module: 16-智能体工程
topic: SKILL.state — 用「不可变 skill 规格 + 可变结构化执行状态」替代只增不减的对话历史，中间推理在状态更新通过校验后立即丢弃
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.26263
paper: SKILL.state: Scalable Long-Horizon Agent Skills
venue: EMNLP
venue_tier: top
evidence_grade: A
verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py（G2/G3 全绿）+ 人工抽检 5 处数字
supersedes:
related: Skill-Context-Compression.md, Skill-Active-Context-Pruning.md, Skill-Skill-Lifecycle-Design.md, Skill-ReAct-Reasoning-Acting.md, Skill-Agent-Memory-Learning.md
---

# Skill Card: 状态化的 Skill 运行时（SKILL.state — 显式执行状态替代对话历史）

**与同域两张「压缩卡」的分工（三张卡不重复）**：`Skill-Context-Compression.md` 与
`Skill-Active-Context-Pruning.md` 回答「上下文**已经**变长了怎么压」；本卡回答「它为什么一开始
就不该变长」——不是压缩历史，而是**取消历史**：把执行状态显式结构化，prompt 只由
(不可变规格, 当前状态, 最新观测) 三件东西组成。论文的预算对照实验正是冲着这个区别做的：
把压缩基线钉在**同样的** token 预算上，看它们是否还打得过结构化状态。

---

## ① 算法原理

**核心思想**：把「执行」从**对话流水账**改成**状态机**。对话式 runtime 每一步都把历史推理、
历史动作、历史观测重新喂给模型，让模型自己从文本里"还原"当前世界；SKILL.state 把「当前世界」
一次性写成结构化状态 Σ_t，模型每步只收到三样：不可变的 skill 规格 P、当前状态 Σ_t、最新观测
O_t —— 历史观测、历史动作、历史推理**一律不进 prompt**。

**数学直觉**：单步输入 A_t = (P, Σ_t, O_t)；模型产出 (R_t, ΔΣ_t, a_t)，R_t 是推理链、ΔΣ_t 是
状态补丁（JSON 字典）、a_t 是动作。提交规则是 Σ_{t+1} = Σ_t ⊕ ΔΣ_t，⊕ 是**带 null 删除语义的
字典合并**。推理链只在当步生成时存在，补丁通过校验后立即永久丢弃。于是单步 prompt 长度
|P_t| = O(|P|+|Σ|+|O|)，与步数 t 无关；累计 O(T)。对照之下，对话式 runtime 的 |C_t| = O(t)、
累计 O(T²)—— 差距不是"更省一点"，而是**复杂度阶不同**。

**关键假设**：执行状态必须能成为未来执行的**充分统计量** —— 凡是以后会影响动作的过去信息，
都必须在它刚出现时就被写进 Σ_t。这条假设不成立时，丢弃历史就是**有损**的（论文自承三种
失败场景，见 ①b）。第二个假设是：状态补丁的**校验权在确定性 runtime 手里**，不在模型手里。

---

## ①b 反例与适用边界

**什么时候不要用这个算法**

1. **没有固定 schema，且状态结构要在执行中发现**。论文自承的第一种失败场景（⑥ Q49）：
   如果连"哪些字段才构成状态"都得边跑边试，那么先写 schema 反而会把状态写窄。
2. **需要长程回溯推理**。某个早期观测当时看不出相关性、因而没写进 Σ_t，后面才发现它关键 ——
   这时历史已经不在 prompt 里了，丢掉的正好是救命信息（论文自承失败场景 2，⑥ Q49）。
3. **任务目标本身就是历史轨迹**。审计、调试溯源、解释"当时为什么这么判"这类任务里，
   交互历史是**交付物**而不是开销（论文自承失败场景 3，⑥ Q49）。正确做法是**并行写一份
   append-only 审计日志落库**——留痕但不喂 prompt。把两者混成一份，要么丢证据，要么把
   prompt 又撑大。
4. **执行步数很短**（一二十步以内）。省下的 token 抵不过 schema 设计 + 校验器 + 影子运行的
   工程量，属于过度设计。
5. **"状态"本身无界**。例如要保留用户每一句原话、每条差评原文、每份合规报告的全文 ——
   那是语料不是状态，硬塞进 Σ_t 只是把 O(T) 换成另一个 O(T)。

**已知的失败模式（均有论文出处）**

1. **小模型翻车在"状态更新"上，不在推理上**。论文在开放权重模型上把错误分成三类：
   过早覆盖/删除占 68%、schema 理解/类型强转占 20%、JSON 语法占 12%，并明确把根因归到
   **结构化输出遵从性**而非推理能力（⑥ Q39–Q43）。
2. **补丁语义必须"合并"而不是"整体覆盖"**。上一条里的"过早覆盖/删除"就是模型忘了把没改的键
   带上；如果运行时用整体替换实现，一次更新会静默抹掉全部其他状态。本卡 ③ 段的 `_merge`
   严格按"合并 + 值为 null 才删"实现，并用断言锁死这个语义。
3. **校验器缺失 = 状态可被污染**。论文的设计里 schema 所有权与校验都在确定性 runtime，
   坏补丁进不了 Σ_t，只触发 rollback-retry（⑥ Q12）。若把校验交给模型自评，本卡的核心安全
   属性直接消失——这一点在母婴这类"答错就是客诉"的场景里比省 token 更值钱。
4. **O(1) 是"与步数无关"，不是"与状态无关"**。Σ_t 会随 schema 内的键增多而变大；把几十个字段
   全塞进 schema，prompt 仍然会变大，只是不再随执行史增长。
5. **"结构化状态一定更省 prompt"是错的**。论文表 4 里，SKILL.state 在 τ-Bench Retail 上
   单步平均 prompt 长度**高于** ReAct 基线（⑥ Q37/Q38），它省下的是**随步数增长的那一部分**，
   累计口径才更低。所以不要用"单步 prompt 更短"当验收口径，要用"prompt 是否随步数持平"。
6. **观测文本本身会波动**。本卡代码里的 O(1) 是"给定同分布、等宽的观测"；真实系统里一条
   长日志就能把单步 prompt 顶上去，状态机制对此无能为力（能治它的是观测侧的裁剪）。

**论文自己承认的局限**

1. **只做了单 agent**。多 agent 共享一份执行状态会引入并发写，需要给 ⊕ 定义确定性的冲突消解
   语义，论文明确说这是单 agent 设置**没有检验**的（⑥ Q50）。
2. **依赖模型提出合法补丁**。对更小的开放权重模型，论文建议上语法约束解码来消除格式错误
   （⑥ Q12）；这等于承认"协议对了但模型写不对"是一个真实存在的落地风险。
3. **论文未报告任何货币成本、时延数字或工程人力**。摘要提到的 latency degradation 是**定性**
   描述（⑥ Q01），全文没有延迟或费用数字；本卡 ⑤ 因此只给公式与参数来源，不给金额结论。
4. **论文未报告**母婴 / 跨境电商场景的任何内容：它的公开基准是 Linux CTF 与零售/航空客服，
   与本卡的母婴出海场景只有机制上的同构，没有数据上的迁移证据。

---

## ② 母婴出海应用案例

### 场景 1：跨境售后 agent 跑长工单 —— 用显式状态机把 prompt 锁在常数大小

- **业务问题**：母婴出海店铺（吸奶器、纸尿裤、奶瓶）在 Amazon、独立站、TikTok Shop 三处接单，
  售后工单常常一跑就是几十轮：查订单 → 查物流 → 查批次 → 判断是否落在召回范围 → 给补偿方案 →
  等用户确认。对话式 agent 跑到几十轮以后，prompt 里塞满了早已失效的物流节点和已经作废的
  补偿提案，于是会出现最贵的一类错误：**拿一个政策已经变更、或商品已经下架的 SKU 的去回答**，
  或者把上一单的退货结论套到这一单。SKILL.state 的做法：把「这一单现在处于什么状态」写成
  结构化状态（工单号、订单号、SKU/ASIN、批次号、当前处理节点、已提出方案、用户已确认的事实、
  当前政策版本号），每轮只把 **{该状态 + 最新一条用户消息 + 不可变的售后政策 skill}** 喂给模型；
  中间判断链路在状态补丁通过校验后立即丢弃。
- **数据要求**：工单/订单/物流/退款主键与状态字段（工单 ID、订单 ID、SKU 或 ASIN、批次号、
  处理节点、已给方案、用户确认事实），**事件级**（每条用户消息、每条系统回调各一行），
  历史长度**≥90 天**（业务参数，非论文数字；跨境退换货周期长，太短会漏掉"已过售后期"这条分支）；另需一份
  **政策版本表**（政策生效/失效时间 + 适用 SKU 范围），因为它必须进 Σ_t 才能保证"用当前政策
  而不是历史政策"。
- **数据可得性**：`部分可得（需补充 X）`。工单与订单侧（Shopify / Amazon SP-API / 客服系统）
  可得；**政策版本表通常不存在** —— 售后政策往往散在帮助中心文案与运营群公告里，需要先补齐
  "政策 = 带生效时间的结构化记录"这一层。另外，三处渠道不做买家身份对齐时，同一用户会有三份
  互相污染的 Σ_t；建议先按邮箱/手机号 + 订单号做弱对齐再上状态机。
- **预期产出**：(a) 一套显式状态机（schema + 校验器 + 状态补丁协议）；(b) 每步 prompt 组成与
  大小的审计记录（可直接回答"prompt 是否随轮次持平"）；(c) 被 rollback 的补丁清单及原因。
- **业务价值**：见 ⑤ 的 ROI 公式。钱来自两处：① 长工单的 token 不再随轮次二次增长 —— 论文在
  Warehouse（T=100）上报告 Stateful 基线累计消耗 1,062,387 tokens、SKILL.state 只有 65,408
  tokens，即 **16.2× 的 token 缩减**（⑥ Q25）；② 把"用过期政策/旧库存作答"从"概率更低"变成
  "结构上不可能"——因为旧值根本不在 prompt 里。

### 场景 2：旺季前六周 —— 补货 / 上架 agent 的状态不许留在聊天记录里

- **业务问题**：黑五前六周，运营 agent 要跑一条几十步的流程：查销量 → 查库存 → 查头程在途 →
  算补货量 → 生成 FBA 补货单 → 核对 ASIN 合规（CE / CPSC / FDA 标签）→ 更新 Listing。
  对话式写法会把"第 3 步查到的库存"和"后面重新查到的库存"混在一起，模型经常按旧库存下单；
  或者忘了某个 ASIN 的合规结论已经变了。
- **数据要求**：SKU 日粒度销量与库存（**≥2 个完整旺季周期**，母婴有宝宝月龄生命周期，太短会
  把成长效应误判成季节效应）、在途头程批次与 ETA、FBA 库存快照、合规检查结论及其生效时间。
- **数据可得性**：`部分可得（需补充 X）`。销量 / 库存 / FBA 快照在卖家后台可得；
  **在途头程与合规结论**通常散在货代邮件与合规服务商报告里，需要先结构化成
  （批次号、ETD/ETA、合规项、结论、生效日）才能进 schema。拿不到就退化为"单域状态机"
  （只锁库存与 Listing 状态），仍可用，但防错范围变小。
- **预期产出**：每一步的状态补丁与校验结果（哪一步被 rollback、为什么），以及
  "prompt 大小 vs 步数"曲线；可直接作为旺季前 agent 稳定性验收的一部分。
- **业务价值**：把"按旧库存下单"这类错误变成结构上不可能。这部分**不依赖论文数字**，
  属于机制直接给出的性质；论文只提供"这套机制在长流程上不会劣化"的实验支撑。

---

## ③ 代码模板

- 依赖：**只用标准库**（`json` / `copy` / `random` / `dataclasses` / `typing`）。不联网、不调用
  任何 LLM SDK、不绘图 —— 因此能在 K1 的断网语义下真正跑通（L4 脚本执行 + L5 pytest 断言）。
- 结构：不可变 `SkillSpec`（P）→ `ExecutionState` + `_merge`（Σ_t 与 ⊕）→ `validate_patch`
  （确定性校验）→ `SkillStateRuntime`（Algorithm 1：构造 (P, Σ_t, O_t) → 校验 → 提交 → 丢弃推理）
  → `ConversationalRuntime`（论文 §5.1 的 ReAct 式基线对照）→ 构造数据 → 七个断言 → 业务演示。
- 与论文的对应：`build_prompt` = 式 (1)(2)；`Proposal` = 式 (3)；`ExecutionState.apply` /
  `_merge` = 式 (4) 的 ⊕（含 null 删除）与论文附录 A.4 的 `state_patch` 约定；
  `step()` 的"校验失败就不提交" = §7 的 rollback-retry；`ConversationalRuntime` = §5.1 的基线。
- ⚠️ 代码内所有 episode / 货架 / 商品编号均为**构造数据**（`make_warehouse_episode`），
  **不是论文数据**，也不代表任何企业真实流程；`_scripted_policy` 是**替代 LLM 的确定性脚本**，
  只为让模板离线可跑，不代表论文里模型的实际行为。
- 演示输出里的倍数、字符数、步数等均为**本地可复现数字**（跑一遍即可复核），与 ⑥ 段论文报告的
  16.2× 压缩比（⑥ Q25）是两个不同口径的测量，**数值不同、也不可互相印证**。

```python
"""
SKILL.state 运行时最小实现 —— 不可变 Skill 规格 + 结构化执行状态 + 每步只喂 (spec, state, observation)

论文：2608.26263 "SKILL.state: Scalable Long-Horizon Agent Skills"
对应：§3.1 执行状态与 schema、§3.2 推理与状态转移、Algorithm 1、§3.3 复杂度分析、§7 局限

与论文机制的逐条对应：
  1. `SkillSpec` 是 frozen dataclass，即论文的不可变过程性规格 P（整个生命周期只读）；
  2. `ExecutionState` 即结构化执行状态 Sigma_t，`apply()` 实现论文式 (4)
     Sigma_{t+1} = Sigma_t ⊕ DeltaSigma_t —— 字典合并 + null 删除语义（支持嵌套补丁）；
  3. `build_prompt()` 只组装 (P, Sigma_t, O_t) 三样，不追加任何历史 observation / 动作 / 推理；
  4. `step()` 先用确定性校验器验证 DeltaSigma_t，只有通过才提交状态并执行动作；
     模型给出的推理链 R_t 在提交后即被丢弃（运行时不保存、后续 prompt 不携带）；
  5. `ConversationalRuntime` 对应论文 §5.1 的 Prompt (ReAct-style) 基线，用于在同一次运行里
     把 O(1) 与 O(T^2) 的差距显式量出来。

⚠️ 本文件中的 episode、货架、商品编号全部是**构造数据**（`make_warehouse_episode`），
   不是论文数据，也不代表任何企业的真实流程。
   `_scripted_policy` 是**替代 LLM 的确定性脚本**（离线可跑，不依赖任何模型 SDK），
   它扮演"语言模型"的角色：读 prompt → 产出 (推理, 状态补丁, 动作)。
"""

import copy
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# 固定宽度占位符：让结构化状态在每一步的序列化长度严格相等（=> prompt 大小恒定）。
EMPTY_SLOT = "EMPTY__"      # 7 字符，与 "item_00" 等宽
IDLE_CODE = "IDLE_00"       # 7 字符，与 "SHIP_00" / "STOR_00" 等宽


# ---------------------------------------------------------------------------
# 1. 不可变 skill 规格 P（论文 §3：the immutable procedural specification）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkillSpec:
    """P：过程性规格。frozen=True 让运行时在类型层面就无法改写它。"""

    skill_id: str
    instructions: str
    state_schema: tuple
    action_space: tuple

    def prompt_block(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "instructions": self.instructions,
            "state_schema": list(self.state_schema),
            "action_space": list(self.action_space),
        }


# ---------------------------------------------------------------------------
# 2. 可变结构化执行状态 Sigma_t（论文 §3.1）
# ---------------------------------------------------------------------------
def _merge(old: Any, new: Any) -> Any:
    """嵌套字典合并；值为 None 表示删除该键（论文的 null-deletion semantics）。"""
    if isinstance(old, dict) and isinstance(new, dict):
        merged = copy.deepcopy(old)
        for key, value in new.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = _merge(merged.get(key), value)
        return merged
    return copy.deepcopy(new)


class ExecutionState:
    """Sigma_t：结构化、可变、面向未来执行的充分统计量（论文 §2.3、§3.1）。"""

    def __init__(self, schema: Sequence, values: dict = None) -> None:
        self.schema = tuple(schema)
        self._values = {}
        for key in self.schema:
            if values and key in values:
                self._values[key] = copy.deepcopy(values[key])

    def snapshot(self) -> dict:
        return copy.deepcopy(self._values)

    def apply(self, patch: dict) -> None:
        """Sigma_{t+1} = Sigma_t ⊕ DeltaSigma_t（论文式 (4)）。"""
        for key, value in patch.items():
            if value is None:
                self._values.pop(key, None)
            else:
                self._values[key] = _merge(self._values.get(key), value)


def validate_patch(patch: Any, schema: Sequence) -> tuple:
    """确定性校验 DeltaSigma_t —— 校验权在运行时，不在模型（论文 §7）。

    返回 (是否通过, 原因)。未通过时调用方必须**不提交**状态、也不执行动作。
    """
    if not isinstance(patch, dict):
        return False, "state_patch 不是 JSON 对象"
    unknown = sorted(k for k in patch if k not in set(schema))
    if unknown:
        return False, "state_patch 含 schema 外键: " + ",".join(unknown)
    try:
        json.dumps(patch, ensure_ascii=False)
    except (TypeError, ValueError):
        return False, "state_patch 值不可 JSON 序列化"
    return True, "ok"


# ---------------------------------------------------------------------------
# 3. 运行时（论文 Algorithm 1）
# ---------------------------------------------------------------------------
@dataclass
class Proposal:
    """模型单步产出 (R_t, DeltaSigma_t, a_t) —— 论文式 (3)。"""

    reasoning: str          # R_t：用完即弃
    patch: dict             # DeltaSigma_t
    action: str = None      # a_t


@dataclass
class StepOutcome:
    step: int
    prompt_chars: int
    committed: bool
    action: str
    reason: str


class SkillStateRuntime:
    """A_t = (P, Sigma_t, O_t)：每步只把这三样交给模型（论文式 (1)(2)）。"""

    def __init__(self, spec: SkillSpec, policy: Callable, initial_state: dict = None) -> None:
        self.spec = spec
        self.policy = policy
        self.state = ExecutionState(spec.state_schema, initial_state)
        self.n_steps = 0
        self.n_rollback = 0          # 校验失败触发的回滚-重试次数（论文 §7）

    def build_prompt(self, observation: str) -> str:
        """只组装 (P, Sigma_t, O_t)：没有历史 observation、没有历史动作、没有历史推理。"""
        return json.dumps(
            {
                "spec": self.spec.prompt_block(),
                "state": self.state.snapshot(),
                "observation": observation,
            },
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )

    def step(self, observation: str) -> StepOutcome:
        prompt = self.build_prompt(observation)
        proposal = self.policy(prompt)
        ok, reason = validate_patch(proposal.patch, self.spec.state_schema)
        if not ok:
            # 校验失败 → 不提交状态、不执行动作（rollback-retry cycle，论文 §7）
            self.n_rollback += 1
            return StepOutcome(self.n_steps, len(prompt), False, None, reason)
        self.state.apply(proposal.patch)     # 提交；proposal.reasoning 不进入任何持久结构
        self.n_steps += 1
        return StepOutcome(self.n_steps, len(prompt), True, proposal.action, "ok")

    def run(self, observations: Sequence) -> list:
        return [self.step(o) for o in observations]


class ConversationalRuntime:
    """对照基线：Prompt (ReAct-style)，把 observation / 推理 / 动作不断追加进 transcript（论文 §5.1）。

    保留它不是为了使用，而是为了让"prompt 随执行史增长"这件事在同一个脚本里可被量化。
    """

    _REASONING = ("The environment emitted an event; I should locate the affected item in my "
                  "inventory and decide whether to store it or ship it.")

    def __init__(self, spec: SkillSpec, initial_state: dict = None) -> None:
        self.spec = spec
        self.state = ExecutionState(spec.state_schema, initial_state)
        self.transcript = []

    def build_prompt(self, observation: str) -> str:
        self.transcript.append("Observation: " + observation)
        self.transcript.append("Reasoning: " + self._REASONING)
        self.transcript.append("Action: (recorded in transcript)")
        return json.dumps(
            {
                "instructions": self.spec.instructions,
                "history": list(self.transcript),
                "latest_observation": observation,
            },
            ensure_ascii=False,
        )

    def run(self, observations: Sequence) -> list:
        return [len(self.build_prompt(o)) for o in observations]


# ---------------------------------------------------------------------------
# 4. 构造数据 + 替代 LLM 的确定性脚本（离线可跑）
# ---------------------------------------------------------------------------
def make_warehouse_spec() -> SkillSpec:
    """域级 schema：整个域写一次，所有 episode 复用（论文 §3.1）。"""
    return SkillSpec(
        skill_id="warehouse-ops",
        instructions=("You operate a shelf-based warehouse. On a customer order, ship the item "
                      "from the shelf that currently holds it. On an arrival, store the item on "
                      "an empty shelf."),
        state_schema=("inventory", "last_action", "last_item"),
        action_space=("Store <item> <shelf>", "Ship <item> <shelf>", "Wait"),
    )


def make_warehouse_episode(n_steps: int, n_shelves: int = 20, seed: int = 7) -> tuple:
    """构造数据：一个货架仓储 episode（论文 §4.1 Environment 1 的同构简化版）。

    返回 (observations, initial_state)。货架数固定、id 与事件文本等宽  =>  prompt 长度恒定。
    """
    rng = random.Random(seed)
    shelves = ["shelf_%02d" % i for i in range(n_shelves)]
    items = ["item_%02d" % i for i in range(n_shelves)]
    inventory = {s: EMPTY_SLOT for s in shelves}
    for i in range(n_shelves // 2):
        inventory[shelves[i]] = items[i]
    occupied = [s for s in shelves if inventory[s] != EMPTY_SLOT]
    free = [s for s in shelves if inventory[s] == EMPTY_SLOT]

    observations = []
    for t in range(n_steps):
        take_from_shelf = (t % 2 == 0 and occupied) or not free
        if take_from_shelf:
            shelf = occupied.pop(0)
            item = inventory[shelf]
            observations.append("Customer ordered %s." % item)
            inventory[shelf] = EMPTY_SLOT
            free.append(shelf)
        else:
            shelf = free.pop(0)
            idle = [i for i in items if i not in set(inventory.values())]
            item = idle[rng.randrange(len(idle))]
            observations.append("Shipment arrived %s." % item)
            inventory[shelf] = item
            occupied.append(shelf)
    initial_state = {
        "inventory": {"shelf_%02d" % j: (items[j] if j < n_shelves // 2 else EMPTY_SLOT)
                      for j in range(n_shelves)},
        "last_action": IDLE_CODE,
        "last_item": EMPTY_SLOT,
    }
    return observations, initial_state


def _scripted_policy(prompt: str) -> Proposal:
    """替代 LLM 的确定性脚本：读 (P, Sigma_t, O_t)，产出 (推理, 状态补丁, 动作)。

    真实系统里这一步由语言模型完成；这里用规则脚本是为了让模板**离线可跑**。
    """
    obj = json.loads(prompt)
    inventory = obj["state"].get("inventory", {})
    observation = obj["observation"]

    if observation.startswith("Customer ordered "):
        item = observation[len("Customer ordered "):].rstrip(".")
        shelf = next((s for s, v in sorted(inventory.items()) if v == item), None)
        if shelf is None:
            return Proposal("REASONING: %s 不在状态里，等待。" % item, {}, "Wait")
        idx = item[-2:]
        return Proposal(
            "REASONING: %s 在 %s 上，生成 Ship 补丁并清空该货架。" % (item, shelf),
            {"inventory": {shelf: EMPTY_SLOT}, "last_action": "SHIP_" + idx, "last_item": item},
            "Ship %s %s" % (item, shelf),
        )

    if observation.startswith("Shipment arrived "):
        item = observation[len("Shipment arrived "):].rstrip(".")
        shelf = next((s for s, v in sorted(inventory.items()) if v == EMPTY_SLOT), None)
        if shelf is None:
            return Proposal("REASONING: 没有空货架放 %s，等待。" % item, {}, "Wait")
        idx = item[-2:]
        return Proposal(
            "REASONING: 把 %s 放到第一个空货架 %s。" % (item, shelf),
            {"inventory": {shelf: item}, "last_action": "STOR_" + idx, "last_item": item},
            "Store %s %s" % (item, shelf),
        )

    return Proposal("REASONING: 未知事件，等待。", {}, "Wait")
```

```python
# ===========================================================================
# 5. 可执行断言（K1 L5：pytest 必须全绿）
# ===========================================================================
def test_prompt_carries_only_spec_state_and_observation():
    """A_t = (P, Sigma_t, O_t)：prompt 里没有 history、也没有推理字段（论文式 (1)(2)）。"""
    spec = make_warehouse_spec()
    observations, initial = make_warehouse_episode(6)
    prompts = []

    def policy(prompt):
        prompts.append(prompt)
        return _scripted_policy(prompt)

    SkillStateRuntime(spec, policy, initial).run(observations)
    assert len(prompts) == 6
    for raw in prompts:
        obj = json.loads(raw)
        assert sorted(obj) == ["observation", "spec", "state"]
        assert sorted(obj["state"]) == sorted(spec.state_schema)
        assert "history" not in raw and "REASONING" not in raw


def test_prompt_size_does_not_grow_with_steps():
    """O(1)：prompt 大小与已执行步数 t 无关（论文 §3.3 式 (6)(7)）。"""
    spec = make_warehouse_spec()
    for n_steps in (10, 50, 200):
        observations, initial = make_warehouse_episode(n_steps)
        runtime = SkillStateRuntime(spec, _scripted_policy, initial)
        sizes = [outcome.prompt_chars for outcome in runtime.run(observations)]
        assert runtime.n_rollback == 0
        assert len(set(sizes)) == 1, "步数 %d 时 prompt 大小发生变化: %s" % (n_steps, sorted(set(sizes)))


def test_reasoning_is_discarded_after_validated_update():
    """R_t 用完即弃：任何一步的推理链都不出现在后续 prompt 里（论文 §3.2）。"""
    spec = make_warehouse_spec()
    observations, initial = make_warehouse_episode(30)
    prompts, reasonings = [], []

    def policy(prompt):
        prompts.append(prompt)
        proposal = _scripted_policy(prompt)
        reasonings.append(proposal.reasoning)
        return proposal

    SkillStateRuntime(spec, policy, initial).run(observations)
    assert len(prompts) == 30 and len(reasonings) == 30
    for i, reasoning in enumerate(reasonings):
        assert reasoning.startswith("REASONING:")
        for later_prompt in prompts[i + 1:]:
            assert reasoning not in later_prompt


def test_invalid_patch_is_not_committed():
    """校验失败 → 不提交状态、不执行动作（论文 §7 的 rollback-retry）。"""
    spec = make_warehouse_spec()
    observations, initial = make_warehouse_episode(3)

    def bad_key_policy(prompt):
        return Proposal("REASONING: 试着写一个 schema 外的键。",
                        {"warehouse_map": {"shelf_00": "item_00"}}, "Ship item_00 shelf_00")

    runtime = SkillStateRuntime(spec, bad_key_policy, initial)
    before = runtime.state.snapshot()
    outcome = runtime.step(observations[0])
    assert outcome.committed is False
    assert outcome.action is None
    assert runtime.n_rollback == 1
    assert runtime.n_steps == 0
    assert runtime.state.snapshot() == before          # 状态一个字节都没动

    def not_a_dict_policy(prompt):
        return Proposal("REASONING: 补丁不是 JSON 对象。", ["shelf_00"], "Wait")

    runtime2 = SkillStateRuntime(spec, not_a_dict_policy, initial)
    outcome2 = runtime2.step(observations[0])
    assert outcome2.committed is False and runtime2.n_rollback == 1
    assert runtime2.state.snapshot() == before

    ok, reason = validate_patch({"inventory": {"shelf_00": EMPTY_SLOT}}, spec.state_schema)
    assert ok and reason == "ok"


def test_merge_operator_deletes_on_null():
    """⊕：字典合并 + null 删除语义，支持论文附录 A.4 的嵌套补丁写法。"""
    spec = make_warehouse_spec()
    state = ExecutionState(
        spec.state_schema,
        {"inventory": {"shelf_00": "item_00", "shelf_01": "item_01"}, "last_item": "item_09"},
    )
    state.apply({"inventory": {"shelf_00": None}})        # 嵌套 null 删除
    assert state.snapshot()["inventory"] == {"shelf_01": "item_01"}
    state.apply({"inventory": {"shelf_02": "item_02"}})   # 嵌套新增不影响其他键
    assert state.snapshot()["inventory"]["shelf_01"] == "item_01"
    state.apply({"last_item": None})                      # 顶层 null 删除
    assert "last_item" not in state.snapshot()


def test_schema_authored_once_and_reused_across_episodes():
    """论文 §3.1：schema 每个域写一次，跨任务复用（InterCode CTF 是 100 题共用一套）。"""
    spec = make_warehouse_spec()
    assert spec.state_schema == ("inventory", "last_action", "last_item")
    for n_steps, seed in ((10, 1), (25, 2), (50, 3)):
        observations, initial = make_warehouse_episode(n_steps, seed=seed)
        runtime = SkillStateRuntime(spec, _scripted_policy, initial)
        runtime.run(observations)
        assert runtime.n_rollback == 0
        assert set(runtime.state.snapshot()) <= set(spec.state_schema)


def test_cumulative_tokens_linear_versus_quadratic():
    """论文 §3.3 式 (5)(7)：对话式累计 O(T^2)，SKILL.state 累计 O(T)。"""
    spec = make_warehouse_spec()
    observations, initial = make_warehouse_episode(200)
    outcomes = SkillStateRuntime(spec, _scripted_policy, initial).run(observations)
    state_sizes = [o.prompt_chars for o in outcomes]
    baseline_sizes = ConversationalRuntime(spec, initial).run(observations)
    assert max(state_sizes) - min(state_sizes) == 0
    assert state_sizes[-1] == state_sizes[9]
    assert baseline_sizes[-1] > 3 * baseline_sizes[9]          # 基线单步 prompt 随 t 增长
    assert sum(baseline_sizes) > 5 * sum(state_sizes)          # 累计口径差距被放大


# ===========================================================================
# 6. 业务演示（K1 L4：作为脚本直接跑通）
# ===========================================================================
def _business_demo():
    spec = make_warehouse_spec()
    observations, initial = make_warehouse_episode(100)
    runtime = SkillStateRuntime(spec, _scripted_policy, initial)
    outcomes = runtime.run(observations)
    sizes = [o.prompt_chars for o in outcomes]
    baseline = ConversationalRuntime(spec, initial).run(observations)

    print("== SKILL.state：100 步长流程（构造数据，非论文数据）==")
    print("  提交步数 = %d   回滚重试 = %d" % (sum(1 for o in outcomes if o.committed), runtime.n_rollback))
    print("  单步 prompt 字符数: 最小 %d / 最大 %d  -> 与步数无关（O(1)）" % (min(sizes), max(sizes)))
    print("  累计 prompt 字符数: %d" % sum(sizes))
    print("== 对照：同一条流程改用对话式 transcript（ReAct 式）==")
    print("  单步 prompt 字符数: 第 10 步 %d -> 第 100 步 %d" % (baseline[9], baseline[-1]))
    print("  累计 prompt 字符数: %d（%.1f 倍）" % (sum(baseline), sum(baseline) / sum(sizes)))
    snapshot = runtime.state.snapshot()
    inventory = snapshot["inventory"]
    occupied = sum(1 for v in inventory.values() if v != EMPTY_SLOT)
    print("== 最终结构化执行状态 ==")
    print("  在架商品 %d / 货架总数 %d   最后动作 = %s" % (occupied, len(inventory), snapshot["last_action"]))


if __name__ == "__main__":
    _business_demo()
```

---

## ④ 技能关联

- **同域对照｜`Skill-Context-Compression.md`**（16-智能体工程）：那张卡做的是"把已经变长的
  历史压短"（历史压缩 + 观测压缩 + 准则优化）。逻辑接口正好相反：本卡主张**先别让它变长**。
  两者可组合成两步——先用本卡把执行状态显式化，再对**状态之外**的内容（长观测、长工具返回）
  用那张卡的观测压缩；但不要再对"历史"做压缩，因为本卡已经把历史取消了。
- **同域对照｜`Skill-Active-Context-Pruning.md`**（16-智能体工程）：主动剪枝回答"哪一段上下文
  现在可以扔"。本卡把这个问题变成结构性问题：不是判断"哪段能扔"，而是**只允许状态进 prompt**，
  于是剪枝策略不再是模型的一次判断，而是运行时的不变式。
- **前置｜`Skill-Skill-Lifecycle-Design.md`**（16-智能体工程）：那张卡讨论 skill 的表示、组合与
  生命周期。本卡是它的执行侧补充：skill 被选中之后"怎么跑"。本卡的 `state_schema` 属于 skill
  规格的一部分——**schema 何时写、谁拥有、如何版本化**，应当在那张卡的生命周期框架里定。
- **基线对照｜`Skill-ReAct-Reasoning-Acting.md`**（10-MAS）：这张卡就是论文的
  **Prompt (ReAct-style) 基线**（⑥ Q03 描述的那种"每一步都追加"的 runtime）。两者不是替代关系
  而是分工：ReAct 的"推理-行动交替"在**短程**任务上仍然有效且实现成本低；本卡治的是它的长程
  退化（⑥ Q24）。选型判据很简单——**任务步数是常数级还是几十上百步**。
- **基线对照｜`Skill-Agent-Memory-Learning.md`**（10-MAS，MemGPT 系长期记忆 + 虚拟上下文）：
  这张卡对应论文的 **Memory (Summarization-style) 基线**（⑥ Q18）。差别在于：摘要式记忆仍在
  维护**对话语义**，未来决策依赖"对过去的文本重建"；本卡维护的是**显式世界状态**，未来决策
  只依赖当前状态。论文同时给出了两条路线的实测差距（⑥ Q26/Q33）。

---

## ⑤ 商业价值评估

**ROI 公式**（本节不代入任何未经论文或企业数据支持的数字；论文未报告任何货币成本）：

```
ROI = (ΔC_prompt + ΔE_error) / C_sys

ΔC_prompt = N_工单 × Σ_t [ S_hist(t) − S_state ] × p_token
            S_hist(t) = s_0 + k·t   （对话式：单步 prompt 随步数线性增长）
            S_state   = 常数        （状态化：单步 prompt 与步数无关）
ΔE_error  = N_错误 × c_错误        （过期政策/旧库存类错误 × 单次代价）
C_sys     = schema 设计 + 校验器开发 + 影子双跑（新旧两套 runtime 并行一段）成本
```

| 参数 | 含义 | 来源 |
|---|---|---|
| `N_工单` | 统计期内长工单条数（建议门槛：轮次超过该阈值才计入） | 企业客服系统自有口径 |
| `k` | 对话式 runtime **单步新增**的 prompt token（观测 + 推理 + 动作） | **必须企业自测**：把现有 agent 的单步 token 对轮次做线性拟合取斜率 |
| `s_0` | 对话式 runtime 的初始 prompt（系统提示 + 首轮） | 同上，拟合截距 |
| `S_state` | 状态化 runtime 的常数 prompt（skill 规格 + schema 大小 + 观测） | **本卡 ③ 段可现成测出**（演示里打印"最小/最大字符数"，两者相等即 O(1)） |
| `p_token` | 该模型的 token 单价（元/千 token） | 企业采购价 / 云厂商公开价 |
| `c_错误` | 一次"用过期政策/旧库存作答"的代价（补偿 + 客诉 + 差评处理 + 账号风险） | 企业自有口径 |
| `C_sys` | 一次性建设成本 | **论文未报告任何工程成本量级**，需企业自估 |

**能引用的唯一论文侧参考点**：论文报告在 Warehouse（T=100）同预算对照下，Stateful 基线累计
1,062,387 tokens、SKILL.state 65,408 tokens，即 16.2× 的 token 缩减（⑥ Q25）；在
InterCode CTF 上累计 token 比 ReAct 低 60.4%、比 Stateful 低 65.9%（⑥ Q30）。**这些是论文在
它自己的基准上的口径，不是母婴跨境的 ROI**；本卡不把它们换算成金额。

- **实施难度**：⭐⭐⭐☆☆ —— 机制本身简单（一个 schema、一个合并器、一个校验器），
  真正的成本在**两个地方**：schema 设计（要一次想清楚"未来会用到哪些字段"）与影子双跑
  （新旧两套 runtime 并行，确认状态化没有丢信息）。
- **优先级**：⭐⭐⭐⭐☆ —— 对"已经在跑长流程 agent"的团队是**优先级最高的一类改造**：
  它不是优化项，而是把"长流程必然退化"这件事从架构上移除。若当前 agent 普遍只在几步内结束，
  优先级降到 2 星。
- **评估依据**：收益侧最确定的部分是**结构性的**（prompt 不再随步数增长、旧值不再出现在
  prompt 里），这部分不依赖任何未经核验的数字；金额侧则完全依赖企业自己的 `k`、`p_token`、
  `c_错误` 三个参数，因此本卡不承诺任何提升幅度。

---

## ⑥ 原文引用

### A. 问题陈述：对话式 runtime 为什么会随长程退化

> 原文："Existing agent runtimes maintain execution by continually appending observations, actions, and intermediate reasoning traces to an ever-growing conversation history, causing latency degradation and context-poisoning failures over long horizons."
> 出处：2608.26263 §Abstract 摘要｜Q01

> 原文："Across diverse datasets, models, and execution environments, SKILL.state improves task accuracy while substantially reducing cumulative token consumption."
> 出处：2608.26263 §Abstract 摘要｜Q02

> 原文："Modern agent runtimes almost universally adopt a conversational execution model. At every execution step, the language model receives the original skill specification together with an ever-growing transcript of previous reasoning, actions, observations, and tool outputs (Yao et al., 2022; Mialon et al., 2023)."
> 出处：2608.26263 §1 Introduction｜Q03

> 原文："Prompt size grows with execution length, increasing token consumption and inference cost (Liu et al., 2024; Xiao et al., 2024a). Historical observations and obsolete reasoning remain embedded in the context long after they cease to be relevant, requiring the model to continually distinguish current facts from historical artifacts. Consequently, execution correctness increasingly depends on reconstructing state from accumulated textual history."
> 出处：2608.26263 §1 Introduction｜Q04

### B. 架构定义：每步只喂 (P, Σ_t, O_t)

> 原文："At execution step $t$, the language model receives only three inputs:"
> 出处：2608.26263 §1 Introduction｜Q05

> 原文："where $P$ is the immutable procedural specification, $\Sigma_{t}$ is the structured execution state at step $t$, and $O_{t}$ is the latest observation received from the environment. The language model never receives previous observations, previous actions, or previous reasoning traces."
> 出处：2608.26263 §3 SKILL.state｜Q06

> 原文："At each step, the runtime constructs a prompt from $(P,\Sigma_{t},O_{t})$, invokes the language model, deterministically validates the proposed state transition, updates the execution state, executes the selected action, and repeats the process using the updated state."
> 出处：2608.26263 §3 SKILL.state（Figure 1 执行循环）｜Q07

> 原文："Schemas are authored once per domain rather than per task; for example, across all 100 diverse challenge instances in the InterCode CTF benchmark, the agent reuses a single static 5-field schema (discovered_flags, tested_hypotheses, active_files, working_dir, cmd_summary)."
> 出处：2608.26263 §3.1 Execution State and Schema Authoring｜Q08

> 原文："After producing a validated state update, the intermediate reasoning trace is discarded while only the updated execution state is retained. Consequently, execution depends strictly on the current world state instead of replaying historical trajectories."
> 出处：2608.26263 §1 Introduction｜Q09

### C. 中间推理的丢弃机制与状态更新的校验

> 原文："Crucially, within-step multi-step reasoning is fully intact during generation to support complex deductive planning. However, once the state transition has been validated and applied, the reasoning trace $R_{t}$ is discarded permanently and never appears in subsequent prompts."
> 出处：2608.26263 §3.2 Reasoning and State Transitions｜Q10

> 原文："where $\oplus$ denotes the runtime’s dictionary merge operator with null-deletion semantics. This model projects transient reasoning into persistent structured state, allowing only information required for future execution to survive across interactions."
> 出处：2608.26263 §3.2 Reasoning and State Transitions（式 (4) 后）｜Q11

> 原文："Because schema ownership and validation reside in the deterministic runtime rather than the model, malformed outputs cannot corrupt persistent state $\Sigma_{t}$; an invalid patch triggers a rollback-retry cycle."
> 出处：2608.26263 §7 Limitations｜Q12

### D. O(1) prompt / O(T) 累计 token 的复杂度主张

> 原文："We propose SKILL.state, a runtime architecture that executes procedural skills through explicit structured execution state where intermediate reasoning is discarded after each step, proving a strictly bounded $\mathcal{O}(1)$ prompt footprint and $\mathcal{O}(T)$ cumulative token complexity."
> 出处：2608.26263 §1 Introduction（贡献 1）｜Q13

> 原文："Let $T$ denote the execution horizon. For conversational runtimes, prompt length grows with the accumulated interaction history, $|C_{t}|=\mathcal{O}(t)$, leading to cumulative token complexity:"
> 出处：2608.26263 §3.3 Complexity Analysis（式 (5) 前）｜Q14

> 原文："In contrast, SKILL.state maintains only the procedural specification, structured execution state, and latest observation:"
> 出处：2608.26263 §3.3 Complexity Analysis（式 (6) 前）｜Q15

> 原文："which is asymptotically bounded and independent of the number of previously executed turns $t$. Consequently, cumulative prompt complexity grows strictly linearly with the execution horizon:"
> 出处：2608.26263 §3.3 Complexity Analysis（式 (7) 前）｜Q16

> 原文："By discarding intermediate reasoning traces after each validated transition, SKILL.state maintains a bounded $\mathcal{O}(1)$ prompt footprint and scales linearly $\mathcal{O}(T)$ in cumulative tokens."
> 出处：2608.26263 §6 Conclusion｜Q17

### E. 对照组与「同预算」的压缩基线

> 原文："Memory (Summarization-style): Maintains a rolling 3-step conversational window alongside a periodically updated natural language summary of past interactions (Packer et al., 2023)."
> 出处：2608.26263 §5.1 Experimental Setup（Primary Runtime Paradigms）｜Q18

> 原文："Truncated (Sliding Window): Retains only the most recent interaction turns that fit within a fixed token budget."
> 出处：2608.26263 §5.1 Experimental Setup（Budget-Matched and Compression Controls）｜Q19

> 原文："Summary-capped: Strictly enforces a hard token ceiling on the natural language summary."
> 出处：2608.26263 §5.1 Experimental Setup（同组）｜Q20

> 原文："ReAct + LLMLingua (Jiang et al., 2023): Uses budget-aware small-model perplexity compression to prune tokens from the full history down to the target budget."
> 出处：2608.26263 §5.1 Experimental Setup（同组）｜Q21

> 原文："To determine whether SKILL.state’s performance gains stem merely from shorter prompts or from structured state representation, we evaluate budget-matched baselines on Warehouse ($T=100$, Gemini-3-Flash) pinned to the token budget of SKILL.state ($\sim$1,800 tokens)."
> 出处：2608.26263 §5.6 Experiment 5: Budget-Matched Controls and Statistical Compression｜Q22

### F. 结果数字（压缩比、准确率、公开基准）

> 原文："Results: As shown in Table 1, SKILL.state matches or exceeds baseline accuracy across all horizons while maintaining a flat prompt size ($\sim$1,736–1,905 tokens)."
> 出处：2608.26263 §5.2 Experiment 1: Long-Horizon Execution Scaling｜Q23

> 原文："In contrast, history-appending baselines suffer quadratic token accumulation $\mathcal{O}(T^{2})$."
> 出处：2608.26263 §5.2 Experiment 1（同段）｜Q24

> 原文："At $T=100$, the Stateful baseline consumes 1,062,387 tokens, whereas SKILL.state consumes only 65,408 tokens (a $16.2\times$ token reduction)."
> 出处：2608.26263 §5.2 Experiment 1（同段）｜Q25

> 原文："At $T=200$, SKILL.state maintains 0.94 accuracy consuming 122k tokens, while the Memory baseline inflates to 6.1M tokens."
> 出处：2608.26263 §5.2 Experiment 1（同段）｜Q26

> 原文："As shown in Table 2, the standard Prompt runtime degrades sharply from 0.68 at low noise down to 0.53 at high noise. In contrast, SKILL.state maintains robust task completion ($\geq 0.97$) across all noise levels because distractors are filtered out during state patch generation and never enter subsequent prompts."
> 出处：2608.26263 §5.3 Experiment 2: Context Corruption (Noise Robustness)｜Q27

> 原文："As shown in Table 3, history-based baselines hallucinate for 5 to 8 consecutive turns because obsolete facts in their prompt history overpower contradictory new observations. In sharp contrast, SKILL.state requires zero recovery steps: because its decisions depend on the current structured state, the state is updated immediately upon receiving the corrective alert."
> 出处：2608.26263 §5.4 Experiment 3: State Recovery｜Q28

> 原文："Results: As shown in Table 4, SKILL.state achieves the highest task completion rates across all three benchmarks while substantially cutting cumulative token consumption."
> 出处：2608.26263 §5.5 Experiment 4: Public Interactive Benchmarks｜Q29

> 原文："In InterCode CTF, maintaining explicit hypotheses and discovered flags in $\Sigma_{t}$ prevents the model from repeating failed commands, increasing pass@1 to 54.2% (+7.8 points over the strongest baseline and +12.4 points over Stateful) while cutting total tokens by 60.4% vs. ReAct and 65.9% vs. Stateful."
> 出处：2608.26263 §5.5 Experiment 4（同段）｜Q30

> 原文："In $\tau$-Bench Retail, SKILL.state leads with 58.3% pass rate at the lowest total token cost."
> 出处：2608.26263 §5.5 Experiment 4（同段）｜Q31

> 原文："In $\tau$-Bench Airline, where complex database responses cause baseline prompts to peak above 11,000 tokens/step, SKILL.state maintains a flat footprint of $\sim$2,800 tokens/step and achieves a 32.4% pass rate, saving 40.5% tokens vs. ReAct and 45.4% vs. Stateful."
> 出处：2608.26263 §5.5 Experiment 4（同段）｜Q32

> 原文："Sliding-window truncation drops to 0.18 because critical early inventory allocations are evicted. LLMLingua drops to 0.22 because statistical entropy filtering removes seemingly redundant slot identifiers that are semantically vital. In contrast, SKILL.state achieves 0.94 score, demonstrating that structured state maintenance preserves exact relational dependencies that statistical compressors destroy."
> 出处：2608.26263 §5.6 Experiment 5｜Q33

> 原文："| Full ReAct (Unbounded) | 0.84 | 36,362 | 1,245,413 |"
> 出处：2608.26263 §5.6 表 5（Budget $\sim$1,800 tokens 对照）｜Q34

> 原文："| Summary-capped | 0.52 | 1,840 | 63,400 |"
> 出处：2608.26263 §5.6 表 5｜Q35

> 原文："| SKILL.state (Structured) | 0.94 | 1,905 | 65,408 |"
> 出处：2608.26263 §5.6 表 5｜Q36

> 原文："| Prompt (ReAct) | 43.2% | 1,909 | 977k | 48.2% | 2,819 | 4.48M | 21.8% | 5,100 | 4.85M |"
> 出处：2608.26263 §5.5 表 4（InterCode CTF / τ-Bench Retail / τ-Bench Airline）｜Q37

> 原文："| SKILL.state | 54.2% | 813 | 387k | 58.3% | 3,325 | 3.47M | 32.4% | 2,800 | 2.88M |"
> 出处：2608.26263 §5.5 表 4｜Q38

### G. 失败模式、统计口径与模型设置

> 原文："On open-weight models (Gemma-4-31B at $T=100$, score 0.42), we analyze failure logs and categorize errors into three distinct modes:"
> 出处：2608.26263 §5.7 Error Taxonomy for Open-Weight Models｜Q39

> 原文："Premature State Overwrite / Deletion (68%): The model accidentally omits existing keys during state update rather than merging in-place."
> 出处：2608.26263 §5.7（失败模式 1）｜Q40

> 原文："Schema Comprehension / Type Coercion (20%): Inconsistencies between expected nested lists and dictionaries."
> 出处：2608.26263 §5.7（失败模式 2）｜Q41

> 原文："JSON Syntax / Formatting Slips (12%): Malformed JSON delimiters or trailing commas."
> 出处：2608.26263 §5.7（失败模式 3）｜Q42

> 原文："This error distribution shows that small-model degradation stems from structured output adherence rather than reasoning capacity, motivating constrained decoding in future runtime iterations."
> 出处：2608.26263 §5.7（同段）｜Q43

> 原文："Statistical Significance: All synthetic experiments are evaluated across 5 distinct procedural generator seeds. Results are reported as mean $\pm$ sample standard deviation. Differences between SKILL.state and baselines at extended horizons ($T\geq 50$) are statistically significant (paired $t$-test, $p<0.01$)."
> 出处：2608.26263 §5.1 Experimental Setup｜Q44

> 原文："Underlying Models: Evaluations are conducted across proprietary and open-weight models: Gemini-3-Flash, Gemma-4-31B-it, and Qwen-3-8B-it. Decoding is controlled at temperature $0.0$ and top-$p$ $1.0$ across all runs to ensure deterministic reproducibility."
> 出处：2608.26263 §5.1 Experimental Setup｜Q45

### H. 相关工作定位与论文自承局限

> 原文："Frameworks like LangGraph use auxiliary structured state to orchestrate workflows across agent nodes. However, these systems still rely on conversational transcripts as the primary reasoning substrate. SKILL.state replaces this substrate by discarding intermediate reasoning traces immediately after producing validated state transitions."
> 出处：2608.26263 §2.2 Memory Architectures for Long-Horizon Agents｜Q46

> 原文："Rather than attempting to process or compress extended conversational histories, SKILL.state prevents history accumulation entirely by maintaining the canonical execution state required for the next computation."
> 出处：2608.26263 §2.4 Context Management and Long-Context Reasoning｜Q47

> 原文："SKILL.state assumes that the execution state can be made a sufficient statistic for future execution: that everything in the past bearing on future actions can be projected into the structured state as soon as it becomes known."
> 出处：2608.26263 §7 Limitations｜Q48

> 原文："However, this assumption fails in three distinct settings: (1) when no fixed schema is known in advance and the relevant state structure must be discovered dynamically during execution; (2) when a correct state update depends on an earlier observation whose relevance was not recognized when first observed, and was therefore never committed to state; and (3) when the task objective is defined over the historical trajectory itself (e.g., auditing, debugging provenance, or explaining past actions), where interaction history is the target output rather than operational overhead."
> 出处：2608.26263 §7 Limitations（同段）｜Q49

> 原文："Our current implementation focuses on single-agent procedural execution. While the explicit state abstraction extends naturally to multi-agent systems—where a shared execution state acts as the central coordination substrate instead of exchanging quadratic conversational transcripts—multi-agent environments introduce concurrent writes, requiring deterministic conflict-resolution semantics in the merge operator $\oplus$ that our single-agent setting does not exercise."
> 出处：2608.26263 §7 Limitations｜Q50
