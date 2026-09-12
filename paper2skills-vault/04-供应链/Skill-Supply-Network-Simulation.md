---
title: Skill-Supply-Network-Simulation
module: 04-供应链
topic: 把供应链写成「节点 + 链路属性图」，用离散事件仿真回答多级备货、效期报废、单点中断与跨仓替代调拨的权衡
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2607.09745
paper: SupplyNetPy: An Open-Source Python Library for High-Fidelity Modeling and Simulation of Arbitrary Supply Chain and Inventory Networks
venue: Winter Simulation Conference 2026
venue_tier: CCF-B
evidence_grade: A
verified_by: verify_skill_code.py (K1 PASS) + quote_check.py (引文逐字核验 VERBATIM) + gate_check.py G1/G2/G3
supersedes:
related: Skill-Multi-Echelon-Inventory.md, Skill-Safety-Stock-Replenishment.md, Skill-Demand-Forecasting-Supply-Chain.md, Skill-Two-Echelon-Inventory-DRL.md, Skill-Time-Series-Forecasting.md
---

# Skill-Supply-Network-Simulation

> **venue 说明（R3 要求显式标注）**：本文是 **Winter Simulation Conference 2026（WSC 2026）** 的录用稿，
> 作者自述为 preprint（原文见 ⑥）。registry 的 `venue_tier` 记作 `second`，本卡按 venue 白名单口径归一化为
> `CCF-B`；registry 的 `note` 写的「有开源实现」经核对**属实**（包名、安装方式、仓库地址与许可证见 ⑥）。
> 本文不是 workshop 短文，也不是 findings——它是 WSC 正会论文，含完整的 Related Work、Architecture、
> Validation、Case Study 与 Conclusions 各节。
>
> **registry 的四项能力断言逐项核对（全部属实，逐字证据见 ⑥）**
>
> | registry 断言 | 核验结论 | 逐字证据所在 |
> |---|---|---|
> | 多级网络离散事件仿真 | ✅ 属实。网络是有向图，**可以不是树**（一个节点可挂多个上游）；库基于 SimPy 做事件驱动仿真；文档另有多级面包房网络算例 | §Abstract、§3.1、§4.3 |
> | 易腐库存 | ✅ 属实。per-unit 效期追踪 + FIFO + 报废成本记账；复现的案例研究正是"医院药房 + 有限保质期药品" | §1.1、§3.1（Inventory 条目）、§5.1 |
> | 节点中断 | ✅ 属实，且**节点与链路双通道**：节点有失效概率与可调用的中断/恢复时长，链路另有链路级失效概率 | §3.1（Node 条目）、§3.3 |
> | 随机提前期 | ✅ 属实。链路提前期写成 Python callable，可为确定性或随机；需求到达时刻与订货量同样是 callable，支持任意分布 | §3.1（Link 条目）、§3.3 |
>
> ⚠️ 一处需要业务方注意的口径差异：registry 的 `decision_reason` 说"落地成本最低"，
> 那是对**建模仿真这件事**的判断；卡片 ⑤ 的难度评分只针对"跑通一个网络模型"，
> 不含把贵司的批次效期台账与中断样本补齐的成本——那部分我按数据风险单列。

> **R4 判定：通过（不构成「纯 benchmark 报告 / demo paper」）。** 三条理由，逐条可复核：
>
> ① **贡献是建模能力，不是功能清单。** 论文把供应链抽象成「节点 + 链路属性」的有向图：三类节点
>    （源头 / 持货 / 需求）加结构规则，链路承载运输成本与提前期分布；补货策略与供应商选择策略
>    都是可继承的类。用户只描述结构与属性，事件推进、日志、节点级与网络级报表由库负责。
>    这是**可组合的建模原语**，而不是一段演示脚本——它把"建一个仿真模型"降级成"填两张表"。
> ② **有三类相互独立的验证，且给出可复核的数字。** 解析基准（报童问题、EOQ、安全库存）、
>    与商业工具 AnyLogistix 的**逐组件**数值比对、以及复现已发表的 WSC 2021 药房案例研究。
>    其中两处关键读数：EOQ 的仿真最优批量与分析解相差 3%，而该成本曲线上两者的成本差不到 0.1%
>    —— 论文自己用"曲线很平坦"解释了这 3% 为什么仍然算验证通过（原文见 ⑥）。
> ③ **有一项关于仿真本身的方法论结论。** 确定性边界下两个工具结果不一致的**原因**被定位到
>    DES 的「同时事件排序」与浮点精度，并给出了出现条件（需求间隔与提前期为整数倍、且缺货不补发）。
>    论文的态度是"没有唯一正确的同刻事件处理方案"——这是对离散事件仿真语义的陈述，
>    而不是产品介绍，也是本卡最值得带到业务侧的一条工程常识。
>
> **残余风险（一并说清）**：本文**没有提出新算法**，它的新意落在抽象与验证上；案例研究是**复现**而非新应用；
> 作者自述早期原型 InventOpt 已另行发表、更完整的讨论将出现在后续扩展版中。
> 因此本卡按**建模仿真工具卡**定位，不应被当作"新方法"卡片引用或用于方法学对比。

---

## ① 算法原理

**核心思想**：把供应链写成一张**有向图**——节点是供应链实体（供应商 / 制造商 / 分销商 / 零售商 / 需求点），
链路是节点之间的有向运输边；使用者只描述「结构 + 属性」（节点容量、补货策略、再订货点、中断的失效分布、
易腐品的保质期；链路的运输成本与提前期分布），库负责事件推进、事件日志与报表。它真正解决的问题不是
"算得更准"，而是**把模型变成可程序化生成的对象**：一段脚本可以批量生成并仿真成千上万个不同的网络配置，
用于设计空间探索、what-if 分析、机器学习训练数据生成与供应链数字孪生。

**数学直觉**：库存头寸 `IP = 在库 + 在途 − 欠交`；`IP` 跌到再订货点就补货（(s,S) 补到目标水位 `S`，
(R,Q) 补固定量 `Q`，定期评审每 `T` 个时间单位评审一次）。易腐品把库存拆成**批次**：每个单位带制造日与保质期，
发货按 FIFO，超期即出库并计**报废成本**（与持有成本、缺货成本并列记账）。中断是**节点在一段时间内不可收发**，
缺额沿未履约订单与缺货向下游自然传导——所以"一个海外仓倒下"会以丢单的形式出现在报表里，而不是以报错的形式。

**关键假设**：① 需求到达时刻、单笔订货量、链路提前期都写成可调用的抽样函数，因而支持任意分布
（泊松、正态、经验分布或自定义）；② 网络结构可以不是树——一个持货节点可以挂多个上游，
用可配置的供应商选择策略决定由谁履约；③ 库存运输被建模为**离散事件**，当前不支持连续物流（管道式链路）。

---

## ①b 反例与适用边界

**什么时候不要用这个仿真**：

- **只问单点、单策略的解析最优解。** 经典 (s,S)/(R,Q) 有闭式解或成熟解析近似时，用几千次重复去仿真
  是在"用模拟换一个本来就能算出来的数"。仿真的价值在解析模型写不出来的地方：多级 × 易腐 × 中断 × 跨仓替代
  同时出现时，解析式基本无从下手。
- **需要连续物流、多产品共享库存、车队与车辆路径。** 论文明确把这些列为当前不支持的项（原文见 ⑥）。
- **想把仿真输出当作"真实需求下的因果结论"。** 仿真给的是"给定输入分布下的系统行为"；输入分布本身仍要靠
  业务数据标定。标定错了，仿真只会把错误放大成一张看起来很精确的报表。

**已知的失败模式**：

- **同时事件与浮点精度造成的分叉**（论文自承，且给了具体条件）：当需求间隔与提前期互为整数倍、
  且采用"缺货不补发"（未满足的需求立即丢失）策略时，到货事件与需求事件可能落在同一时刻，
  谁先被处理取决于实现细节（本库用事件创建时分配的 ID，商业工具用进程注册顺序），事件日志从第一次碰撞起分叉，
  长跑中持续累积；非整数的时间间隔还会因浮点表示的累积误差造成同类偏差。论文的结论是：
  **没有唯一正确的同刻事件处理方案**，两个工具各按自己的语义都是对的；而真实供应链模型是随机的，
  用多次重复的均值估计会把这些边界差异平均掉。业务含义：**这类差异只在"确定性参数 + 整数周期"的对照实验里
  才会咬人**——而那恰恰是最容易做、也最常被用来做参数评审的实验形式。要避开它，就把对照实验也做成
  多次重复的随机配置，而不是只比一个确定性的数。
- **效期品上"多备一点"可能反而更差**（**本卡模板的模拟观测，不是论文结论**）：当头程与清关时间吃掉效期时，
  把备货档位往上抬会让更多库存在卖出前过期，可用量下降，满足率不升反降。这条现象在 ③ 的输出表里可复现
  （合成参数）。它提醒的是：效期品的"多备一点"和常规品的"多备一点"不是同一个动作。
- **把中断当成一次性事件**。中断期间未履约的订单、顺延的到货、以及替代调拨吃掉的兄弟仓库存，
  会在中断结束后继续影响一段时间；只统计中断窗口内的表现会低估影响，也看不到恢复期的二次缺货。

**论文自己承认的局限**（原文引用见 ⑥）：

- 当前版本**不支持**连续物流（管道式链路）、多产品共享库存、车队与车辆路径等高级物流功能；
- 库存运输被建模为离散事件（库存量本身可以是实数或离散）；
- 性能结论是**单机执行时间**下的标度律：执行时间随仿真时长线性增长、随网络规模接近线性，
  没有超线性的组件管理开销；并行/分布式仿真（PDES）仍是未来方向，因此"面向大规模网络的实时数字孪生"
  目前**没有**性能证据；
- 论文把「更完善的节点与链路中断建模 + 内建韧性指标」「车队与碳成本」「多产品共享库存与合并订单」
  「可扩展的实时仿真」「面向优化的元模型（图神经网络等）」全部列为**计划中**而非已完成；
- 论文没有给出中断、易腐这两个特性的**独立单点验证数字**（它们是通过案例研究复现间接验证的，见 ⑥）。

---

## ② 母婴出海应用案例

### 场景一：Amazon 平台店铺 + 独立站的双海外仓网络，要不要做「互备代发」

- **业务问题**：FBA 海外仓主要吃 Amazon 平台店铺的订单，第三方海外仓吃独立站（站外投放与内容种草引流来的）
  订单。旺季前要判断：若其中一个海外仓**爆仓 / 停收**，或头程与清关延误导致到不了货，连续数周不可收发，
  两个仓要不要在平时就配成可以互相代发？还是干脆为独立站的溢出订单在 FBA 仓预留一部分库存？
  这个决定必须**留出补货提前期**、在旺季前做完，做完之后就改不动了。
- **数据要求**：
  - **节点**：国内工厂 / FBA 海外仓 / 第三方海外仓的容量、初始在库、单位成本、补货策略参数
    （再订货点 `s` 与目标水位 `S`）、是否按批次管效期（相当于 `inventory_type=perishable` 的开关）；
  - **链路**：头程 + 清关的平均时长、波动（变异系数）、运输成本；
  - **需求**：Amazon 平台店铺与独立站各自的订单到达间隔与单笔订单量分布（日粒度即可，但要含旺季抬升）；
  - **中断**：历史上该仓不可用的天数，或"如果发生"的假设窗口。
  - **历史长度**：至少两个完整销售年度（至少覆盖一次旺季），有至少一次真实中断记录更好。
- **数据可得性**：`部分可得`。库存与订单数据企业内可得（ERP / WMS / 平台后台 / 独立站后台）；
  头程与清关的实际时长需要从物流商账单或报关记录回填（多数公司有，但常缺"报关排队"那一段）；
  **中断样本通常不足**——所以不要拟合中断分布，用「假设窗口 + 敏感性扫描」更诚实。
- **预期产出**：每个海外仓与网络级的满足率、丢单件数、报废量与报废成本、平均在库；
  以及「可替代调拨 / 不可替代」两种配置在同一个中断窗口下的对照表。
- **业务价值**：把"要不要双仓互备"从直觉判断变成有数字的对比。可量化的收益侧 =
  （不可替代配置的丢单件数 − 可替代配置的丢单件数）× 单件毛利，即互备动作**挽回的毛利**；
  成本侧是互备带来的额外持有成本（平均在库上升）。两者相减才是净收益，公式见 ⑤。

### 场景二：效期品（辅食 / 洗护 / 部分耗材）的备货档位与报废预算

- **业务问题**：辅食、洗护、部分耗材有保质期，而**头程 + 清关要占掉一段时间**，平台入库对剩余保质期还有门槛，
  于是真正能卖的窗口 = 出厂效期 − 运输与清关耗时 − 平台入库门槛。同一个 SKU 上因此出现两个方向相反的诉求：
  备少了断货（丢的是毛利），备多了效期先到、只能报废（丢的是货值）。更要命的是母婴品类的需求还有
  **月龄驱动的窗口**——同一罐辅食，宝宝月龄过了就是无效需求，容错比常规品类更低。
- **数据要求**：批次级的出厂日与到期日、到仓日、发货时是否真的按 FIFO 出库、报废与销毁记录、
  单件成本与售价、头程与清关时长分布。粒度：**批次级**（SKU 级不够，因为没有效期维度）；
  历史长度：至少两个完整年度，且要覆盖至少一次效期报废。
- **数据可得性**：`部分可得（需补批次效期台账）`。若 ERP 已有批次与效期字段，可直接导出；
  若只在 SKU 级管库存（无批次），需要先补批次台账——否则"到期报废"这条机制无法标定，
  仿真只能给方向、不能给金额。
- **预期产出**：在给定效期与提前期分布下，每个备货档位对应的「满足率 / 报废件数 / 报废成本」三列；
  以及「效期 × 提前期」组合的盈亏平衡表，回答"这个 SKU 到底该多备还是少备"。
- **业务价值**：丢单毛利与报废成本是同一把尺子上的两个方向，可以直接相加减；这把尺子能把
  "效期品要不要和常规品用同一套备货规则"变成一个**有答案的问题**，而不是两拨人各拿一个指标吵架。

---

## ③ 代码模板

> **本模板是自包含的教学实现，不是论文库的复现，也刻意不 import 它。**
> 论文的官方实现是可 pip 安装的 Python 包（包名、安装方式、仓库地址与许可证见 ⑥），
> 但**本机并未安装**；在卡片里凭空 `import` 一个不存在的包会被 K1 门禁判 `ORPHAN_DEP`（卡片缺陷）。
> 所以下面用标准库 `heapq` + `numpy` 重写同一组概念：图描述的网络、离散事件推进、批次制易腐库存、
> 节点中断窗口、随机提前期与跨仓替代调拨。
> **生产环境请直接用官方库**——它提供的是经过解析基准、商业工具与已发表案例研究三类验证的组件；
> 本模板只保证"概念闭环、可断言、可复现"。

### 第一段：批次制库存与节点 / 链路（对齐 §3.1 的组件划分）

```python
# -*- coding: utf-8 -*-
"""多级供应链离散事件仿真模板（对齐 2607.09745 的四项能力）。

⚠️ 本模板是**自包含的教学实现**，不是 SupplyNetPy 的复现：论文库 supplynetpy
   本机未安装，生产环境应直接使用官方实现（`pip install supplynetpy`，见卡片 ⑥）。
   这里用 heapq + numpy 重写同一组概念，把「多级网络 / 易腐库存 / 节点中断 /
   随机提前期」四件事变成本地可跑、可断言的最小闭环。

概念对齐：
  * 图描述网络：工厂（Supplier）→ 头程链路（Link）→ 海外仓（InventoryNode）→ 需求（Demand）
  * 离散事件推进：heapq 事件表（需求到达 / 到货 / 定期评审 / 中断起止）
  * 易腐库存：批次带到期时刻，FIFO 发放 + 到期报废与报废成本
  * 节点中断：中断窗口内不可发货、到货顺延，需求溢出到兄弟节点（替代调拨）
  * 随机提前期 / 随机需求：头程提前期抽样，需求到达间隔指数分布
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# 部件一：批次制库存（易腐 / 非易腐）
# ---------------------------------------------------------------------------
class Batch:
    """一个库存批次：到期时刻 + 数量。非易腐品的到期时刻记为正无穷。"""

    __slots__ = ("expiry", "qty")

    def __init__(self, expiry: float, qty: float):
        self.expiry = float(expiry)
        self.qty = float(qty)


class Inventory:
    """FIFO 批次库存。

    对齐论文 §3.1 对 Inventory 的描述：易腐品每个单位带制造日与保质期，
    过期单位按 FIFO 清除；报废成本与持有成本、缺货成本一起记账。
    """

    def __init__(self, perishable: bool = False, shelf_life: float = float("inf")):
        self.perishable = bool(perishable)
        self.shelf_life = float(shelf_life)
        self.batches: deque[Batch] = deque()
        self.writeoff_units = 0.0
        self.writeoff_cost = 0.0

    @property
    def on_hand(self) -> float:
        return float(sum(b.qty for b in self.batches))

    def receive(self, now: float, qty: float, made_at: float | None = None) -> None:
        """收货：批次到期时刻 = 制造时刻 + 保质期。

        关键业务含义：**头程 + 清关的时间直接吃掉可售期**——出厂日越早，
        到仓后剩下的效期越短（made_at 即下单/出厂时刻，默认等于到仓时刻）。
        """
        if qty <= 0:
            return
        start = now if made_at is None else float(made_at)
        expiry = start + self.shelf_life if self.perishable else float("inf")
        self.batches.append(Batch(expiry, float(qty)))

    def sweep(self, now: float, unit_cost: float = 0.0) -> float:
        """清理已到期批次并计入报废成本，返回本次报废量。"""
        if not self.perishable:
            return 0.0
        kept: deque[Batch] = deque()
        wasted = 0.0
        for b in self.batches:
            if b.expiry <= now:
                wasted += b.qty
            else:
                kept.append(b)
        self.batches = kept
        self.writeoff_units += wasted
        self.writeoff_cost += wasted * unit_cost
        return wasted

    def issue(self, now: float, qty: float, unit_cost: float = 0.0) -> float:
        """按 FIFO 发货，返回实际发运量（库存不足时小于请求量）。"""
        self.sweep(now, unit_cost)
        need = float(qty)
        while need > 1e-9 and self.batches:
            b = self.batches[0]
            take = min(b.qty, need)
            b.qty -= take
            need -= take
            if b.qty <= 1e-9:
                self.batches.popleft()
        return float(qty) - need
```

### 第二段：离散事件核心与网络装配（对齐 §3.1 的事件推进 + §3.2 的策略与供应商选择）

```python
# ---------------------------------------------------------------------------
# 部件二：节点与链路（对齐 §3.1 的 Node / InventoryNode / Link）
# ---------------------------------------------------------------------------
class StockNode:
    """持货节点（海外仓）。reorder_point / order_up_to 即 (s, S) 两个参数。

    downtime 给出一组中断窗口 (起, 止)；窗口内节点不可收货、不可发货。
    """

    def __init__(self, node_id: str, reorder_point: float = 0.0, order_up_to: float = 0.0,
                 initial_level: float = 0.0, unit_cost: float = 0.0,
                 perishable: bool = False, shelf_life: float = float("inf"),
                 review_period: float = 0.0,
                 downtime: tuple[tuple[float, float], ...] = ()):
        self.node_id = str(node_id)
        self.reorder_point = float(reorder_point)
        self.order_up_to = float(order_up_to)
        self.initial_level = float(initial_level)
        self.unit_cost = float(unit_cost)
        self.review_period = float(review_period)       # > 0 表示定期评审
        self.downtime = tuple(downtime)
        self.inv = Inventory(perishable=perishable, shelf_life=shelf_life)
        # --- 运行时状态 ---
        self.on_order = 0.0
        self.requested = 0.0
        self.served = 0.0
        self.lost = 0.0
        self.stockout_events = 0
        self.order_events = 0
        self.last_t = 0.0
        self.area = 0.0                                 # 在库量对时间的积分
        self.substitute_to: "StockNode | None" = None   # 兄弟仓（替代调拨对象）

    @property
    def inventory_position(self) -> float:
        return self.inv.on_hand + self.on_order

    def is_down(self, t: float) -> bool:
        """中断窗口内节点不可收发（爆仓 / 停收 / 头程口岸关闭）。"""
        return any(a <= t < b for a, b in self.downtime)

    def downtime_end(self, t: float) -> float:
        ends = [b for a, b in self.downtime if a <= t < b]
        return max(ends) if ends else t

    def record(self, t: float) -> None:
        """累计时间加权平均在库量。"""
        if t > self.last_t:
            self.area += self.inv.on_hand * (t - self.last_t)
            self.last_t = t

    def report(self, horizon: float) -> dict:
        self.record(horizon)
        denom = self.requested if self.requested > 0 else 1.0
        return {
            "node_id": self.node_id,
            "requested": self.requested,
            "served": self.served,
            "lost": self.lost,
            "fill_rate": self.served / denom,
            "writeoff_units": self.inv.writeoff_units,
            "writeoff_cost": self.inv.writeoff_cost,
            "avg_on_hand": self.area / horizon if horizon > 0 else 0.0,
            "stockout_events": self.stockout_events,
            "order_events": self.order_events,
        }


class DemandStream:
    """需求流：挂在某个持货节点上，间隔指数分布、单次订购量固定。"""

    def __init__(self, node_id: str, mean_interarrival: float, order_qty: float,
                 channel: str = ""):
        self.node_id = str(node_id)
        self.mean_interarrival = float(mean_interarrival)
        self.order_qty = float(order_qty)
        self.channel = channel


# ---------------------------------------------------------------------------
# 部件三：网络与离散事件仿真核心
# ---------------------------------------------------------------------------
class SupplyNetwork:
    """工厂 →（头程 + 清关，随机提前期）→ 海外仓 A / 海外仓 B → 需求。

    - 工厂为无限产能源（论文里的 infinite_supplier）
    - 两个海外仓互为**替代调拨**节点：主节点缺货或中断时需求溢出到兄弟节点
      （对应 §3.2 的 SelectAvailable：优先选择有货的上游/兄弟）
    """

    def __init__(self, nodes: dict[str, StockNode], demands: list[DemandStream],
                 lead_time_mean: float = 21.0, lead_time_cv: float = 0.45,
                 allow_substitution: bool = True, seed: int = 7):
        self.nodes = dict(nodes)
        for n in self.nodes.values():
            n.inv.receive(0.0, n.initial_level)
        self.demands = list(demands)
        self.allow_substitution = bool(allow_substitution)
        self.lead_time_mean = float(lead_time_mean)
        self.lead_time_cv = float(lead_time_cv)
        self._seq = itertools.count()
        self._events: list[tuple[float, int, str, tuple]] = []
        # 需求流与提前期各用独立随机流：保证「有中断 / 无中断」两场景的需求完全同分布
        self._rng_demand = np.random.default_rng(seed + 1)
        self._rng_lead = np.random.default_rng(seed + 2)

    # --- 事件表 ---
    def push(self, t: float, kind: str, payload: tuple) -> None:
        heapq.heappush(self._events, (float(t), next(self._seq), kind, payload))

    def draw_lead_time(self) -> float:
        """头程 + 清关提前期：均值 lead_time_mean、变异系数 lead_time_cv 的对数正态。"""
        mu = np.log(self.lead_time_mean / np.sqrt(1.0 + self.lead_time_cv ** 2))
        sigma = np.sqrt(np.log(1.0 + self.lead_time_cv ** 2))
        return float(self._rng_lead.lognormal(mu, sigma))

    # --- 策略 ---
    def maybe_order(self, t: float, node: StockNode) -> None:
        """连续评审的 (s, S)：库存头寸跌到 s 或以下就补到 S。"""
        if node.review_period > 0:
            return                                   # 定期评审走 review 事件
        ip = node.inventory_position
        if ip <= node.reorder_point:
            self.place_order(t, node, node.order_up_to - ip)

    def place_order(self, t: float, node: StockNode, qty: float) -> None:
        if qty <= 1e-9:
            return
        node.on_order += qty
        node.order_events += 1
        # 出厂时刻 = 下单时刻：到仓时剩余效期 = 保质期 - 头程/清关耗时
        self.push(t + self.draw_lead_time(), "arrival", (node.node_id, qty, t))

    # --- 事件处理 ---
    def on_demand(self, t: float, node_id: str, qty: float, mean_ia: float) -> None:
        node = self.nodes[node_id]
        for n in self.nodes.values():
            n.record(t)
        node.requested += qty
        got = 0.0
        if not node.is_down(t):
            got = node.inv.issue(t, qty, node.unit_cost)
        short = qty - got
        if short > 1e-9 and self.allow_substitution and node.substitute_to is not None:
            alt = node.substitute_to
            if not alt.is_down(t):
                got2 = alt.inv.issue(t, short, alt.unit_cost)
                got += got2
                short -= got2
                self.maybe_order(t, alt)
        node.served += got
        node.lost += short
        if short > 1e-9:
            node.stockout_events += 1
        self.maybe_order(t, node)
        self.push(t + float(self._rng_demand.exponential(mean_ia)),
                  "demand", (node_id, qty, mean_ia))

    def on_arrival(self, t: float, node_id: str, qty: float, made_at: float) -> None:
        node = self.nodes[node_id]
        for n in self.nodes.values():
            n.record(t)
        if node.is_down(t):
            # 中断期间不可收货：这批货顺延到恢复时刻（本模板取「顺延」而非「丢失」）
            self.push(node.downtime_end(t), "arrival", (node_id, qty, made_at))
            return
        node.on_order -= qty
        node.inv.receive(t, qty, made_at=made_at)
        self.maybe_order(t, node)

    def on_review(self, t: float, node_id: str) -> None:
        node = self.nodes[node_id]
        for n in self.nodes.values():
            n.record(t)
        if node.inventory_position <= node.reorder_point:
            self.place_order(t, node, node.order_up_to - node.inventory_position)
        self.push(t + node.review_period, "review", (node_id,))

    def run(self, horizon: float = 360.0) -> dict:
        for d in self.demands:
            self.push(float(self._rng_demand.exponential(d.mean_interarrival)),
                      "demand", (d.node_id, d.order_qty, d.mean_interarrival))
        for node in self.nodes.values():
            if node.review_period > 0:
                self.push(node.review_period, "review", (node.node_id,))
            node.record(0.0)
        while self._events:
            t, _, kind, payload = heapq.heappop(self._events)
            if t > horizon:
                break
            if kind == "demand":
                self.on_demand(t, *payload)
            elif kind == "arrival":
                self.on_arrival(t, *payload)
            elif kind == "review":
                self.on_review(t, *payload)
        for n in self.nodes.values():
            n.inv.sweep(horizon, n.unit_cost)        # 期末结存里过期的部分同样报废
        reports = {nid: n.report(horizon) for nid, n in self.nodes.items()}
        total_req = sum(r["requested"] for r in reports.values())
        total_served = sum(r["served"] for r in reports.values())
        return {
            "nodes": reports,
            "network": {
                "requested": total_req,
                "served": total_served,
                "fill_rate": total_served / total_req if total_req > 0 else 0.0,
                "lost": total_req - total_served,
                "writeoff_units": sum(r["writeoff_units"] for r in reports.values()),
                "writeoff_cost": sum(r["writeoff_cost"] for r in reports.values()),
            },
        }


# ---------------------------------------------------------------------------
# 场景装配：母婴出海（Amazon 平台店铺 + 独立站）的多级补货网络
# ---------------------------------------------------------------------------
def build_network(*, shelf_life: float = float("inf"), perishable: bool = False,
                  downtime: tuple[tuple[float, float], ...] = (),
                  allow_substitution: bool = True, horizon_days: float = 360.0,
                  review_period: float = 0.0, policy_scale: float = 1.0,
                  seed: int = 7) -> SupplyNetwork:
    """装配「国内工厂 → 头程 → FBA 海外仓 / 第三方海外仓 → 需求」四节点网络。

    shelf_life=inf 表示非易腐；downtime 给海外仓的中断窗口（爆仓/停收）；
    allow_substitution 控制单节点中断时是否允许跨仓替代调拨。
    """
    wh_a = StockNode("FBA海外仓", reorder_point=1300.0 * policy_scale,
                     order_up_to=2400.0 * policy_scale, initial_level=1200.0,
                     unit_cost=8.0, perishable=perishable, shelf_life=shelf_life,
                     review_period=review_period)
    wh_b = StockNode("第三方海外仓", reorder_point=1000.0 * policy_scale,
                     order_up_to=1900.0 * policy_scale, initial_level=900.0,
                     unit_cost=8.0, perishable=perishable, shelf_life=shelf_life,
                     review_period=review_period, downtime=downtime)
    if allow_substitution:
        wh_a.substitute_to = wh_b
        wh_b.substitute_to = wh_a
    demands = [
        DemandStream("FBA海外仓", mean_interarrival=1.0 / 3.5, order_qty=10.0,
                     channel="Amazon 平台店铺"),
        DemandStream("第三方海外仓", mean_interarrival=1.0 / 2.5, order_qty=10.0,
                     channel="独立站"),
    ]
    return SupplyNetwork({"FBA海外仓": wh_a, "第三方海外仓": wh_b}, demands,
                         lead_time_mean=21.0, lead_time_cv=0.45,
                         allow_substitution=allow_substitution, seed=seed)


def run_scenario(label: str, **kw) -> dict:
    horizon = float(kw.get("horizon_days", 360.0))
    net = build_network(**kw)
    out = net.run(horizon=horizon)
    out["label"] = label
    return out
```

### 第三段：自检与演示（把论文的四项能力各写成一个断言）

```python
# ---------------------------------------------------------------------------
# 自检：把论文的四项能力各写成一个断言
# ---------------------------------------------------------------------------
def test_disruption_lowers_fill_rate():
    """节点中断必须压低保单满足率（论文 §3.3：中断经未履约订单与缺货自然传导）。"""
    base = run_scenario("无中断", allow_substitution=True)
    hit = run_scenario("海外仓中断 60 天", allow_substitution=True,
                       downtime=((120.0, 180.0),))
    assert hit["network"]["fill_rate"] < base["network"]["fill_rate"], \
        (base["network"]["fill_rate"], hit["network"]["fill_rate"])
    assert hit["network"]["lost"] > base["network"]["lost"]


def test_substitution_recovers_fill_rate():
    """允许跨仓替代调拨时，中断造成的丢单必须少于不允许替代时。"""
    no_sub = run_scenario("中断 + 不可替代", allow_substitution=False,
                          downtime=((120.0, 180.0),))
    with_sub = run_scenario("中断 + 可替代调拨", allow_substitution=True,
                            downtime=((120.0, 180.0),))
    assert with_sub["network"]["fill_rate"] >= no_sub["network"]["fill_rate"]
    assert with_sub["network"]["lost"] <= no_sub["network"]["lost"]
    assert with_sub["nodes"]["第三方海外仓"]["lost"] < no_sub["nodes"]["第三方海外仓"]["lost"]


def test_perishable_stock_is_written_off():
    """易腐品（辅食/洗护类效期品）在长提前期下必然产生到期报废，非易腐品为 0。"""
    fresh = run_scenario("非易腐", perishable=False)
    shelf = run_scenario("保质期 90 天", perishable=True, shelf_life=90.0)
    assert fresh["network"]["writeoff_units"] == 0.0
    assert shelf["network"]["writeoff_units"] > 0.0
    assert shelf["network"]["writeoff_cost"] > 0.0


def test_fifo_issues_earliest_expiry_first():
    """FIFO：先到期的批次先出库（论文 §3.1 的 FIFO discipline）。"""
    inv = Inventory(perishable=True, shelf_life=10.0)
    inv.receive(0.0, 5.0)          # 到期 10
    inv.receive(3.0, 7.0)          # 到期 13
    assert inv.issue(4.0, 6.0) == 6.0
    assert len(inv.batches) == 1
    assert abs(inv.batches[0].qty - 6.0) < 1e-9, "第 2 批应剩 6 件"
    assert abs(inv.writeoff_units - 0.0) < 1e-9


def test_expired_units_are_removed_before_issue():
    """过期单位不得再发货，且必须计入报废而不是静默消失。"""
    inv = Inventory(perishable=True, shelf_life=5.0)
    inv.receive(0.0, 10.0)
    got = inv.issue(6.0, 4.0, unit_cost=3.0)     # 6 天时第 1 批已过期
    assert got == 0.0, "过期批次不得发运"
    assert abs(inv.writeoff_units - 10.0) < 1e-9
    assert abs(inv.writeoff_cost - 30.0) < 1e-9


def test_lead_time_sampling_matches_declared_mean():
    """随机提前期的抽样均值必须回到声明的均值（否则「随机」只是噪声）。"""
    net = build_network()
    draws = np.array([net.draw_lead_time() for _ in range(20000)])
    assert abs(draws.mean() / net.lead_time_mean - 1.0) < 0.03, draws.mean()
    assert draws.std() > 0.0, "提前期必须是随机的，不能退化成常数"


def test_reports_expose_network_level_aggregates():
    """网络级报表必须可加总到节点级（论文 §3.3 的 node + network 双级报表）。"""
    out = run_scenario("报表", perishable=True, shelf_life=120.0)
    nodes = out["nodes"]
    assert set(nodes) == {"FBA海外仓", "第三方海外仓"}
    assert abs(sum(r["requested"] for r in nodes.values())
               - out["network"]["requested"]) < 1e-6
    for r in nodes.values():
        assert 0.0 <= r["fill_rate"] <= 1.0
        assert r["avg_on_hand"] >= 0.0


if __name__ == "__main__":
    rows = [
        run_scenario("基准：非易腐 / 无中断 / 可替代"),
        run_scenario("易腐：效期 90 天", perishable=True, shelf_life=90.0),
        run_scenario("易腐：效期 60 天", perishable=True, shelf_life=60.0),
        run_scenario("易腐：效期 45 天", perishable=True, shelf_life=45.0),
        run_scenario("易腐：效期 45 天 + 备货档位 x1.3", perishable=True,
                     shelf_life=45.0, policy_scale=1.3),
        run_scenario("单仓中断 60 天（可替代调拨）", downtime=((120.0, 180.0),)),
        run_scenario("单仓中断 60 天（不可替代）", allow_substitution=False,
                     downtime=((120.0, 180.0),)),
    ]
    print("=== 多级补货网络仿真（合成参数，可复现；非论文数据）===")
    head = f"{'场景':38s}{'满足率':>9s}{'丢单件数':>10s}{'报废件数':>10s}{'报废成本':>10s}{'平均在库':>10s}"
    print(head)
    for r in rows:
        n = r["network"]
        avg = float(np.mean([x["avg_on_hand"] for x in r["nodes"].values()]))
        print(f"{r['label']:38s}{n['fill_rate']:>9.3f}{n['lost']:>10.0f}"
              f"{n['writeoff_units']:>10.0f}{n['writeoff_cost']:>10.0f}{avg:>10.0f}")
    down = run_scenario("单仓中断 60 天（可替代调拨）", downtime=((120.0, 180.0),))
    print("--- 中断期间需求向兄弟仓溢出（节点级丢单件数）---")
    for nid, rep in down["nodes"].items():
        print(f"  {nid:12s} 请求 {rep['requested']:8.0f}  履约 {rep['served']:8.0f}  "
              f"丢单 {rep['lost']:7.0f}  满足率 {rep['fill_rate']:.3f}")
    print("全部自检通过 ✅")
```

**本模板的实际运行输出**（`python3 <本卡代码>.py` 可复现；**合成参数下的模拟结果，不是论文数据**）：

```text
=== 多级补货网络仿真（合成参数，可复现；非论文数据）===
场景                                          满足率      丢单件数      报废件数      报废成本      平均在库
基准：非易腐 / 无中断 / 可替代                        1.000         0         0         0       986
易腐：效期 90 天                                1.000         0       900      7200       973
易腐：效期 60 天                                0.994       120      4930     39440       887
易腐：效期 45 天                                0.931      1460     12830    102640       749
易腐：效期 45 天 + 备货档位 x1.3                    0.853      3140     21630    173040      1025
单仓中断 60 天（可替代调拨）                          0.984       340         0         0       912
单仓中断 60 天（不可替代）                           0.910      1910         0         0       996
--- 中断期间需求向兄弟仓溢出（节点级丢单件数）---
  FBA海外仓       请求    12340  履约    12160  丢单     180  满足率 0.985
  第三方海外仓       请求     8960  履约     8800  丢单     160  满足率 0.982
全部自检通过 ✅
```

```text
上表关键对照（同一份脚本输出，可自行复现）：
  节点中断   ：网络满足率 1.000 → 0.984，丢单 0 → 340 件
  替代调拨   ：中断下丢单 1910 件（不可替代）→ 340 件（可替代）
  效期收紧   ：效期 90 → 45 天时，报废 900 → 12830 件，满足率 1.000 → 0.931
  备货档位   ：效期 45 天时把备货档位 ×1.3，报废 12830 → 21630 件，满足率 0.931 → 0.853
```

> ⚠️ **上面两个围栏里的所有数字都来自本模板的合成参数，与论文报告的任何数字无关，不可互相印证。**
> 论文报告的是它自己那套验证的数字（解析基准、商业工具逐组件比对、药房案例研究复现，见 ⑥）。
> 把上表当成"论文结论"或"贵司的预期收益"都是误读——它只说明这套最小实现**逻辑自洽**。

---

## ④ 技能关联

| 关联卡片 | 关系 | 为什么组合 |
|---|---|---|
| `Skill-Multi-Echelon-Inventory.md` | 前置（理论） | 多阶库存优化给出"多级网络的成本结构与服务水平"这套语言；本卡把它从解析模型搬到可执行的仿真里，从而能处理解析模型写不出的易腐、中断与替代调拨。 |
| `Skill-Safety-Stock-Replenishment.md` | 前置（参数） | 安全库存公式给出 (s,S) 的**初值**；把它当仿真起点，再用仿真修正"提前期不独立、需求不平稳、效期会吃掉在库"带来的偏差。 |
| `Skill-Demand-Forecasting-Supply-Chain.md` | 前置（输入） | 仿真的需求侧要的是"到达间隔 + 单笔量的分布"，不是一条点预测；需求预测卡片产出分布与促销日历，直接喂给本卡的需求节点。 |
| `Skill-Two-Echelon-Inventory-DRL.md` | 延伸（策略学习） | 该卡用强化学习求解两级库存的联合决策，需要一个**环境**；本卡的仿真网络可以充当这个环境（论文本身也把"为优化生成元模型与训练数据"列为设计动机）。 |
| `Skill-Time-Series-Forecasting.md` | 互补（反查标定） | 仿真结果与真实履约数据出现系统性背离时，第一个该怀疑的是需求分布标定；时序卡片负责回到历史数据里重估分布。 |

---

## ⑤ 商业价值评估

**ROI 预估**（给公式与参数来源，不直接给编造的数）：

```text
可挽回毛利 = (不可替代配置的丢单件数 − 可替代调拨配置的丢单件数) × 单件毛利
报废节省   = (原备货档位的报废件数 − 新备货档位的报废件数) × 单件成本
互备净收益 = 可挽回毛利 − 新增平均在库 × 单位持有成本
```

参数来源：① **丢单件数与报废件数**——本卡 ③ 的代码在贵司参数下直接输出（本地可复现，不依赖论文数字）；
② **单件毛利 / 单件成本 / 单位持有成本**——来自财务与 ERP，论文里没有，也不需要论文给；
③ **中断窗口**——用历史不可用天数，样本不足时做敏感性扫描（例如"如果停数周"），而不是强行拟合一个分布。

论文侧能提供的参照只有**验证强度**，不是收益：解析基准上，报童问题的最优订货量被仿真曲线复现
（扫描区间 10–200 上曲线峰值落在 Q≈110，与分析最优 Q*≈110 一致）；EOQ 的仿真最优批量与分析解相差 3%，
而该成本曲线上两者的成本差不到 0.1%，论文据此认为验证成立；安全库存的仿真值高于解析值，
论文把原因归给"负需求重抽样让有效需求偏高"；案例研究复现的是**已发表**的药房易腐库存结果。
**这些是论文的验证数字，不是贵司的预期收益。**

- **实施难度**：⭐⭐☆☆☆（2/5）。库可 pip 安装，模型是配置化的（描述节点与链路的字典 / 对象），
  代码门槛低；真正的成本在**数据侧**：批次效期台账与头程/清关时长的回填。技术风险低，数据风险中等。
- **优先级评分**：⭐⭐⭐⭐☆（4/5）。它改变的是**旺季前那一次备货与分仓决策**，而这个决策的重做周期以季度计；
  一旦跑通，同一套网络可以按月复用做滚动推演。没有给满星，是因为仿真本身不产生收益——
  收益来自它改变的决策，且需要业务方接受"用分布而不是用一个数来描述不确定性"。
- **评估依据**：① 论文有三类独立验证（解析 / 商业工具 / 已发表案例研究），不是"我们做了个库，欢迎试用"；
  ② 所需数据在贵司**大部分可得**，缺口是批次效期台账与中断样本，两者都有替代做法（补台账 / 敏感性扫描）；
  ③ 与已有卡片 `Skill-Safety-Stock-Replenishment` 和 `Skill-Multi-Echelon-Inventory` 直接串联，
  是"把公式变成沙盘"的一步，不是新领域。

---

## ⑥ 原文引用

### venue（作者自述）

> 原文："This is the author’s preprint of a paper accepted at Winter Simulation Conference, 2026."
> 出处：2607.09745 作者信息栏（作者自述）

### 能力声明与网络模型

> 原文："It supports multiple replenishment policies, perishable inventory, node disruptions, and stochastic demand and lead times."
> 出处：2607.09745 §Abstract

> 原文："The graph need not be a tree; an inventory node can be connected to multiple upstream manufacturers or suppliers, with a configurable supplier selection policy to choose among them."
> 出处：2607.09745 §3.1

> 原文："There is a dearth of well-maintained, well-documented open-source libraries specifically targeted for the discrete-event simulation (DES) of arbitrary SC networks."
> 出处：2607.09745 §2

> 原文："The library has been thoroughly validated through comparison with analytical benchmarks, against a commercial simulation tool (AnyLogistix) for unit-tests with deterministic configurations, and against published results from a case study."
> 出处：2607.09745 §1.1

### 易腐库存

> 原文："Perishable inventory is supported with per-unit expiry tracking using a first-in, first-out (FIFO) discipline and waste cost accounting, making the library attractive for use cases such as food SCs, milk distribution networks, cold chains for vaccines, and pharmaceutical SCs."
> 出处：2607.09745 §1.1

> 原文："For perishable inventory, each unit is tagged with its manufacture date and shelf life, and the expired items are removed using a FIFO discipline."
> 出处：2607.09745 §3.1（Inventory 类条目）

> 原文："The pharmacy stocks a perishable drug with a finite shelf life. Expired units are discarded and unmet demand is lost."
> 出处：2607.09745 §5.1（Problem description）

> 原文："The independent replication of this case study’s results confirms that SupplyNetPy correctly models perishable inventory with FIFO expiry, stochastic demand, probabilistic supply disruptions, and the associated cost structure."
> 出处：2607.09745 §5.1（Results）

### 节点中断

> 原文："Supports stochastic disruption modeling via a configurable failure probability (failure_p) and Python callables for disruption duration and recovery time."
> 出处：2607.09745 §3.1（Node 类条目）

> 原文："Nodes are configured with failure probability and callable disruption and recovery durations."
> 出处：2607.09745 §3.3

> 原文："There is no single universally correct scheme for simultaneous-event handling in DES, and both tools behave correctly according to their own documented semantics."
> 出处：2607.09745 §4.2

### 随机需求与提前期、策略与 API

> 原文："Demand arrival times, order quantities, and link lead times are all specified as Python callables, enabling any distribution (Poisson, normal, empirical, or user-defined)."
> 出处：2607.09745 §3.3

> 原文："Attributes include source node, sink node, transportation cost, and lead time (specified as a Python callable for deterministic or stochastic lead times)."
> 出处：2607.09745 §3.1（Link 类条目）

> 原文："When a node has multiple upstream suppliers, a supplier selection strategy determines which supplier fulfills each replenishment order."
> 出处：2607.09745 §3.2

### 安装方式、仓库与发布历史

> 原文："It is installable from the Python Package Index via pip install supplynetpy with detailed documentation, user guides and full examples at https://supplychainsimulation.github.io/SupplyNetPy, and source code on GitHub [8]."
> 出处：2607.09745 §1.1

> 原文："The full model configurations, tested parameter ranges, and numerical results of this validation study are available here: https://github.com/SupplyChainSimulation/SupplyNetPy/tree/main/validation."
> 出处：2607.09745 §4.2

> 原文："K. Lone (2024) SupplyNetPy github repository. Note: https://github.com/SupplyChainSimulation/SupplyNetPy"
> 出处：2607.09745 参考文献 [8]

> 原文："The library was subsequently expanded, improved, and thoroughly validated [10], and was released publicly on GitHub under an MIT license in 2025 as SupplyNetPy."
> 出处：2607.09745 §3.5

> 原文："SupplyNetPy has been under continuous development since 2022."
> 出处：2607.09745 §3.5

> 原文："the development of SupplyNet Web, a web-based GUI for SupplyNetPy, available at https://supply-net-web.vercel.app/."
> 出处：2607.09745 Acknowledgements

### 验证：解析基准

> 原文："We swept $Q$ from 10 to 200, ran 1,000 simulation replications at each value, and verified that the simulated profit curve peaks at $Q\approx 110$, matching the analytical optimum $Q^{*}\approx 110$."
> 出处：2607.09745 §4.1（Newsvendor problem 条目）

> 原文："At this optimum the mean simulated profit is $278.2$ with a 95% CI of $[273.5,282.8]$."
> 出处：2607.09745 §4.1（Newsvendor problem 条目）

> 原文："Over a long horizon (4,000 days), the simulated cost-minimizing lot size was approximately 1,010 units, within 3% of the analytical value."
> 出处：2607.09745 §4.1（Economic Order Quantity 条目）

> 原文："Because the EOQ total-cost curve is very flat near its optimum (the cost penalty at 1,010 versus 980 units is under 0.1%), this agreement validates the (R, Q) replenishment policy implementation and cost tracking."
> 出处：2607.09745 §4.1（Economic Order Quantity 条目）

> 原文："Over 100 simulation replications, the estimated safety stock level was 1,346.8 units (95% CI $[1{,}335.9,1{,}357.6]$), the average inventory level was 6,311.7 units (95% CI $[6{,}303.1,6{,}320.4]$), and the average order flow time was 13.81 days (95% CI $[13.79,13.83]$)."
> 出处：2607.09745 §4.1（Safety stock estimation 条目）

> 原文："The simulated safety stock and average inventory lie above their analytical values of 1,000 and 6,000 units, whereas the flow time lies below its analytical value of 2.4 weeks (16.8 days). These expected deviations arise because resampling negative demand realizations (an artifact of the normal approximation) makes the effective demand higher than the nominal normal demand assumed analytically, pushing the two inventory measures above and the flow time below their respective analytical values."
> 出处：2607.09745 §4.1（Safety stock estimation 条目）

### 验证：商业工具逐组件比对

> 原文："For most deterministic model configurations (non-stochastic demand and constant lead times), SupplyNetPy and AnyLogistix produced identical results across all tracked metrics for both single-echelon and two-echelon configurations over long simulations."
> 出处：2607.09745 §4.2（Results Summary）

> 原文："We found a few pathological configurations where results differed even though both models were deterministic and the implementation was correct."
> 出处：2607.09745 §4.2

> 原文："Each tool resolves such ties deterministically but differently: SupplyNetPy uses an event ID assigned at event creation time, whereas AnyLogistix uses the order in which processes register events."
> 出处：2607.09745 §4.2（A）

### 验证：已发表案例研究复现

> 原文："An exhaustive grid search over $(s,S)$ was performed, running 1,000 replications per parameter combination (chosen based on a convergence analysis of the standard error on mean daily cost)."
> 出处：2607.09745 §5.1（Implementation in SupplyNetPy 段）

> 原文："Further, the optimal region identified by SupplyNetPy is consistent with the original study."
> 出处：2607.09745 §5.1（Results 段）

### 性能

> 原文："Execution time grows linearly with simulation length, with tight median confidence intervals throughout."
> 出处：2607.09745 §5.2

> 原文："Scaling with $N$ is close to linear, and the per-event cost grows only gradually as $N$ increases, so SupplyNetPy introduces no super-linear overhead from component management or statistics collection."
> 出处：2607.09745 §5.2

> 原文："Replications are independent and were executed in parallel on a dual-socket Intel Xeon Gold 6148 server (two 20-core CPUs at 2.40 GHz, 80 logical cores) running 64-bit Linux."
> 出处：2607.09745 §5.2

### 论文自承的局限

> 原文："The current implementation does not yet support continuous material flows (e.g., pipelines as links), multi-product shared inventory, or advanced logistics features such as fleet management and vehicle routing."
> 出处：2607.09745 §1.1

> 原文："While the library currently supports both discrete and real-valued inventory, the inventory transport is modeled as discrete events."
> 出处：2607.09745 §6

> 原文："Key features include: (i) support for modeling perishable inventory, (ii) node disruption modeling for resilience assessment, (iii) several types of built-in replenishment policies and supplier selection strategies, all extensible via class inheritance, and (iv) a dual API enabling both rapid functional-style prototyping and deep customization."
> 出处：2607.09745 §6

---

## 附：证据链与核验方式

- **全文底本**：`paper2skills-vault/papers/04-供应链/p2s-2026-0009/fulltext.md`
  （由 `paper2skills-research/scripts/fetch_fulltext.py` 从 arXiv LaTeXML HTML 转换而来，保留章节号；
  该存档**没有页码锚点**，因此 ⑥ 的出处一律只写到章节 / 条目，不写"PDF 第 N 页"——不编造页码）
- **引文逐字核验**：`python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <本卡>`
  —— 每条 ⑥ 引用块都会回查全文底本，报告 `VERBATIM` / `FUZZY` / `FABRICATED`
- **K1 代码可执行**：`python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <本卡>`
- **K2 门禁**：`python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <本卡>`
- **数字的两类约定**：论文事实数字（报童 Q*≈110、EOQ 偏差 3% 与 0.1%、安全库存 1,346.8 对解析 1,000 等）
  一律进 ⑥ 引用块；本模板的运行输出与按贵司参数代入的算式放进代码 / 输出围栏并标注「可复现」。
  两类数字**没有出现在同一句话里**。

### 本卡未能核验、或有意保留为定性的项

1. **导入名与包名不一致**（论文内部不一致，本机无法实测）：§1.1 写的是 `pip install supplynetpy`（全小写），
   而 §3.4 的示例 Listing 写的是 `import SupplyNetPy.Components as scm`（混合大小写）。论文没有解释两者关系，
   本卡不猜；正式上手前请以官方文档的 API 指南为准。
2. **PyPI 与 GitHub 的可达性、版本与活跃度**：论文称包可从 Python Package Index 安装、源码在 GitHub 上、
   以 MIT 许可于 2025 年公开发布，但没有给出 PyPI 上的版本号或最近提交时间。
   本卡撰写时**未联网核实**这些页面（本卡不做网络请求），只逐字引用论文的说法。
3. **Table 1（工具对比表）的逐格内容**：该表在 HTML→Markdown 转换后列结构错位
   （表头列数与数据行的分隔符数不一致），无法可靠对齐列名与单元格。因此本卡**没有**引用 Table 1 的任何单元格，
   只引用正文里关于"开源替代品不支持任意网络、部分不支持仿真"的定性陈述。
4. **性能的绝对值**：论文只给了执行时间随网络规模与仿真时长的**标度律**（"close to linear"），
   没有给出单次仿真耗时、每秒事件数或内存占用的绝对值；本卡因此不引用任何性能绝对值，
   只在 ⑥ 保留其硬件配置以便读者判断适用边界。
5. **中断与易腐的独立验证数字**：论文对这两项特性的验证是**通过案例研究复现**间接给出的
   （成本曲面对比图），没有像 EOQ / 安全库存那样给出单点数字；本卡按此做定性表述。
6. **SupplyNet Web（网页 GUI）**：只出现在致谢中，论文没有对它做任何验证或说明，本卡只把它当作"存在"记录。
7. **registry 的四项能力断言**：逐项核对结论（全部属实）见同目录 `evidence.md`；
   registry 的 `venue_tier: second` → `CCF-B` 的归一化是本卡按白名单口径做的，不是论文的说法。
