---
title: Skill-Routed-Graph-Handoff
module: 10-MAS
topic: 多 Agent 交接格式的自适应选择——类型化依赖图 vs 自然语言，用轻量 router 只在该图化时图化
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.25277
paper: Routed Graph Handoff: Adaptive Format Selection for Multi-Agent LLM Delegation
venue: EMNLP 2026
venue_tier: top
evidence_grade: A
verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py G2 passed + 人工抽检 3 处数字
supersedes:
related: Skill-Subagent-Decomposition.md, Skill-MAS-Orchestrator.md, Skill-MetaGPT-SOP-Driven-Collaboration.md, Skill-AutoGen-Multi-Agent-Conversation.md
---

# Skill Card: 路由式图交接（Routed Graph Handoff, RGH）

**一句话定位**：现有框架只管「谁跟谁说话」（拓扑），不管「怎么说话」（格式）。本卡管格式：
把多 Agent 之间的交接从散文换成**类型化依赖图**，并用一次**轻量 router 调用**决定哪些委派值得图化。

**与同域 `Skill-Subagent-Decomposition.md` 的分工**：那张卡回答「要不要拆、怎么切」；
本卡回答「拆开之后，两个 agent 之间那句话该怎么写」。两张卡是流水线上的前后两道，
**先有拆分才有交接格式**，所以本卡默认你已经在拆 sub-agent。

---

## ① 算法原理

**核心思想**：多 Agent 的瓶颈常不是模型能力，而是**交接格式**——散文把执行顺序与前置条件留在
隐含处。Routed Graph Handoff 把每次委派编码成类型化依赖图（显式 depends_on / precondition）以消掉
错序，但不无条件使用：委派前用一次轻量 LLM 分类调用（约 155 token）判断任务的**计算模式**——
有序依赖链走图，迭代 / 条件 / 自由文本解释走自然语言，默认回落 NL。

**数学直觉**：结构-灵活性权衡。图等于给执行器一个**偏序约束**：正确解落在该偏序上时，错序概率
被消掉；任务需要环境反馈后回溯时，同一约束变成不可逃逸的刚性。因此 router 的目标不是最大化平均
收益，而是**消除负向回归**——只在依赖链上取图，宁可漏收益也不制造回退。

**关键假设**：① 图必须配 graph-aware executor prompt（命名节点类型、定义边语义、指定拓扑遍历），
论文把图与其解释指引当作**单一机制**；② 判据来自任务内容本身，不看 benchmark 身份；
③ 多 Agent 失败的主因确实是 agent 间错位（论文错因分类中占 76%）。

---

## ①b 反例与适用边界

**什么时候不要用这个算法**

1. **任务需要临场回溯 / 条件分支 / 自由文本解释时，绝不要图化。** 论文在 AppWorld 上实测
   graph-only 回退 −14.6 pp（CI [−22.8, −6.4]，152 paired trials）。原因不是图表达力不足，而是
   「图强制提前承诺计划」：环境一偏离，执行器无法逃脱。论文把 router 的首要功能明确定义为
   **regression prevention**——**本卡的核心不是「图更好」，而是「按任务的计算模式选格式」**。
2. **没有 graph-aware executor prompt 就不要上。** 同一份 JSON schema 交给标准 executor prompt，
   收益**严格为零**（论文原话："passing the same JSON to a standard executor prompt yields no gain"）；
   在 τ-retail 上，正是恢复该 prompt 才把 NGH 从低于 NL 抬到 +12.7 pp。论文明确把图与其解释指引
   当作**单一机制，而不是图本身**。落地时这张 prompt 是交付物的一部分，不是可选项。
3. **压缩比不要按 τ-retail 的 3.2× 做预算。** 那是单点值。加权平均只有 **2.1×**，其中
   BrowseComp 2.2×、τ-retail 3.2×、BFCL 2.0×、**AppWorld 仅 1.04×（几乎不省）**。
   按完整单次委派预算计（router 调用与 graph-aware executor prefill 都算进去），τ-retail 上是
   **461 vs 730 token，即 1.6×**。这才是可用于预算与 ROI 的口径；「2–3×」是交接 token 口径。
4. **依赖链占比低的任务流不要上这套机制。** 收益全部来自依赖链任务；论文的 router 在 AppWorld 上
   把 **89%** 的任务判给 NL——也就是说在这 89% 上你只多付了 router 的开销。先量自己的任务分布。
5. **不要指望 router 做实例级精细调度。** 论文的 oracle 分析显示还有 **8.6 pp** 余量，但那需要
   **执行期信号**（mid-trajectory 切换），论文把它留作 future work。现成方案只做到按任务类型。

**已知的失败模式**

1. **「按来源标签选格式」的 if-else 可能就够了。** 论文实测：一个更简单的、把 benchmark 标签映射到
   格式的非 LLM router 也能复现**按 benchmark 的聚合结果**，但它需要拿到 router 从来看不到的
   benchmark identity，且做不出 AppWorld 内部 11%/89% 的切分。反过来说——**如果你的任务流本来就是
   「按来源分类型」的，别上 LLM router**；它的价值被论文定位为「从任务内容做无标签泛化」。
2. **图在聚合类任务上确实有效，但 router 会误判。** AppWorld 的 aggregate 模式上图后 +6.7 pp
   （n=15），而 router 把 15 个非聚合任务也送进了 GRAPH，这 3.4 pp 的损失计在 router 误差里。
   误判方向很具体：**把「带部分序结构」的任务当成纯聚合**。
3. **τ-airline 上 graph-only 回退 −4.0 pp**，同样是 router 用同一条 prompt 把它救回来的。
4. **图会掩盖「灵活性」本身。** 论文的失败例子里，「自由文本建议需要灵活解释」这种要求，
   图根本表达不了——它不是没写好，而是**图这种表示法无法表达 "be flexible"**。

**论文自己承认的局限**

1. realized routing 是**按任务类型**而非按实例：依赖链 benchmark 上 100% 图、AppWorld 上 89% NL，
   粒度粗；细粒度实例级路由需要执行期信号，留作 future work。
2. schema 是**手工设计**的（在 47 条 τ-bench 轨迹上迭代得出），可能不能泛化到协调模式根本不同的
   领域（论文点名「开放式创作任务」）；自动化 schema 生成留作 future work。
3. 主结果只用了**一个 orchestrator backbone**（Claude Sonnet 4.5）；另用 GPT-5 mini 验证了方向一致、
   且做过跨厂商格式可移植性检查，但广泛的多模型复现留作 future work。
4. **论文未讨论**：母婴 / 跨境电商场景；把 token 口径换算成金额或工时；多团队共享同一份 schema 的
   治理与版本管理；中文任务描述下的 router 表现（论文的 router prompt 是英文的，也没有做多语言测试）。

---

## ② 母婴出海应用案例

### 场景 1：跨境运营 agent 的「选品分析 → 合规校验 → 定价」三段委派

- **业务问题**：旺季前的选品决策期，运营主 agent 把一款吸奶器 / 恒温调奶器的新品上架判断拆给三个
  sub-agent：**选品分析 agent**（拉 Amazon 榜单与评论、估需求量与价格带）、**合规校验 agent**
  （FDA 注册、CE、CPSC、CPC 证书、锂电池 UN38.3 报告）、**定价 agent**（头程 + 关税 + FBA 履约 +
  平台佣金 → 到手价与毛利）。现在的接法是自然语言交接，两个具体病症：① 每段交接写成长散文，
  三个 sub-agent 各吃一遍，交接本身吃掉可观 token；② 更致命的是**硬约束在传递中丢失**——
  合规 agent 给出的「必须有 CPC 证书 + UN38.3 报告」在下游被当成普通背景文本，定价 agent 据此
  算出了不含认证成本的毛利，甚至把未取证的 SKU 排进了广告位。这正是论文测到的失败模式：
  执行器误读顺序、丢前置条件、在含糊指令上打转，占多 Agent 失败的大头。
- **数据要求**：**编排层数据即可**，粒度是**单次委派**：（a）每个 sub-agent 的输入 / 输出契约
  （字段级 JSON schema）；（b）历史执行轨迹（每步工具调用、返回值、重试），JSONL 格式；
  （c）依赖关系标注——哪些步骤必须先于哪些步骤（选品 → 合规 → 定价 的 depends_on 边）；
  （d）失败归因标签。历史长度至少覆盖一个完整旺季周期（Q4 大促到次年春季）：母婴品类有宝宝月龄
  生命周期，窗口太短会把「这个品本身就是季节性爆款」误判成流程缺陷。
- **数据可得性**：`部分可得（需补充 X）`。自建编排的 trace 日志（LangGraph / 自研 orchestrator 的
  JSONL）通常已在手；**缺的是依赖关系标注与失败归因标签**——需要补建，但论文给了可照抄的确定性
  规则：从轨迹日志按「缺前置条件的工具报错 / 连续重试同一步骤 / 动作与声明的顺序矛盾」三类规则判定，
  **不依赖人工标注**，可从原始 JSONL 复现。平台侧（Amazon 榜单、广告报表）拿不到受众级日志，
  但**本卡不需要**——它只用编排层数据。
- **预期产出**：（a）一份「任务 → 交接格式」的路由判据表 + 每次委派的决策日志（谁被判成依赖链、
  谁回落 NL）；（b）类型化依赖图 schema（8 类节点 + 7 类边，含 depends_on）落到自有 sub-agent 的
  JSON 契约上；（c）graph-aware executor prompt 模板（交付物，不是可选项）；（d）合规硬约束以
  precondition / constraint 节点显式传递后，定价 agent 的输入里不再出现「会议上口头提过的认证要求」。
- **业务价值**：见 ⑤ 的 ROI 公式。本卡的钱来自三处：交接 token 变少、错序重试被消掉、
  以及**避免「无脑图化」带来的成功率回退**。

### 场景 2：客服 / 售后的「双形态任务流」——同一集群里两种格式必须共存

- **业务问题**：同一个客服 agent 集群里混着两类任务：一类是「查订单 → 查物流 → 生成状态回复」的
  **有序链**（适合图化：省 token，且不会把「生成回复」排在「查单」之前）；另一类是「客户要求退货 →
  判断是否在政策窗口内 → 是否有条件分支（已拆封 / 质量问题 / 跨境退运成本）→ 协商补偿」的
  **条件分支型**（适合 NL：需要临场解释与让步）。团队过去一刀切用自然语言，链式任务的 token 与
  错序重试都在浪费；看到「图交接更省」之后又一刀切改成图，结果退货协商类工单的解决率掉了——
  这就是 AppWorld 那个回退的业务版。
- **数据要求**：客服工单的编排轨迹 + 工单类型标签（是否含条件分支）；以及「政策窗口 / 品类例外规则」
  的结构化表示（用于把硬条件写成 constraint 节点）。粒度：单工单。历史长度：覆盖一个完整的
  大促后退货潮（活动结束后的一段高峰窗口），否则样本会被日常工单主导。
- **数据可得性**：`企业内可得`。工单系统 + agent trace 通常都有；政策规则的结构化需要一次人工梳理，
  属于一次性投入。
- **预期产出**：router 判据表在客服域上的落地版本；以及一份**「哪些工单类型必须走 NL」的白名单**。
- **业务价值**：把「退货协商类工单不要图化」写成显式规则，直接规避论文中 AppWorld 那一类回退。

---

## ③ 代码模板

- 依赖：**仅标准库**（`json` / `dataclasses` / `collections` / `typing`），不联网、不调用任何 LLM SDK，
  可在断网环境直接跑（本机以 K1 断网语义验证通过）。
- 结构：类型化依赖图 schema（节点类型 + 边语义 + 拓扑遍历）→ 两种交接格式下的执行结果 →
  规则替身 router → 业务演示 → 断言测试。
- ⚠️ **这是机制的确定性替身，不是论文方法的复现。** 论文的 router 是一次真实 LLM 分类调用
  （约 155 token），executor 是 Claude Sonnet 4.5 的真实推理；本文件的 router 是**规则替身**
  （关键词 + 结构特征），executor 是确定性模拟器。**下方所有成功率都是构造数据，与论文报告的
  pp 数不存在任何对应关系，数值接近纯属巧合，二者不可互相印证。** 本文件也**不模拟 token 数**——
  论文的 token 口径是真实测量的，模拟出来只会是编造的数字。
- 关键断言直接对应论文的三个事实：依赖链任务上图且 NL 在其上更差；去掉 graph-aware prompt 后
  收益归零（**严格为 0**，不是「略差」）；router 对非依赖链任务回落 NL。另加一条**防止替身比论文更
  乐观**的断言：规则替身必须复现论文 Appendix D 记录的误判方向（带部分序结构的任务被当成纯聚合），
  因此 routed 在构造样本上**也不是满分**——残余误差就是这个替身的 router 误差。
- ⚠️ 落地时请把 `route_format()` 换成一次真实的 LLM 分类调用（论文的 router prompt 见 ⑥ Q15），
  否则你得到的只是「关键词 if-else」，达不到论文说的「从任务内容做无标签泛化」。

```python
# -*- coding: utf-8 -*-
"""
Routed Graph Handoff —— 类型化依赖图交接 + 规则替身 router + graph-aware 执行

论文：2608.25277 "Routed Graph Handoff: Adaptive Format Selection for Multi-Agent LLM Delegation"
      §2.1 Native Graph Handoff Schema / §2.2 LLM Router / §2.1 Graph-aware execution
业务映射：跨境母婴运营 Agent 把「选品分析 → 合规校验 → 定价」交给三个 sub-agent，
         交接格式在「类型化依赖图」与「自然语言散文」之间按任务计算模式二选一。

⚠️ 这不是论文方法的复现，而是论文**机制**的确定性替身（mechanism stand-in）：
   1) 论文的 router 是一次真实 LLM 分类调用；本文件的 route_format() 是**规则替身**，
      只复现「按任务计算模式选格式」这一决策结构，没有复现论文 LLM 的语义判断能力。
   2) 论文的 executor 是 Claude Sonnet 4.5 真实推理；本文件用确定性模拟器替代。
   3) 下方所有成功率都是**构造数据**，与论文报告的 pp 数无任何对应关系，
      数值接近纯属巧合，二者不可互相印证。
   4) 本文件不模拟 token 数：论文的 token 口径是真实测量的，模拟只会产出编造的数字。

只依赖标准库，断网可跑。
"""

# 刻意不用 `from __future__ import annotations`：PEP 563 会把 dataclass 的注解变成字符串，
# 而 dataclasses 解析字符串注解时要回查 sys.modules —— 用 spec_from_file_location 加载
# （不注册进 sys.modules）的工具链会因此在 import 阶段报错。保持真注解对象，模块更稳。

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1) 类型化依赖图 schema（论文 §2.1：8 类节点 + 7 类边）
# ---------------------------------------------------------------------------
NODE_TYPES: Tuple[str, ...] = (
    "goal", "constraint", "entity", "action",
    "precondition", "postcondition", "tool_call", "tool_arg",
)
EDGE_RELATIONS: Tuple[str, ...] = (
    "requires", "targets", "blocks", "enables", "depends_on", "contradicts", "follows",
)
# 决定执行顺序的边语义：graph-aware executor 必须按这些边做拓扑遍历。
# 论文对 depends_on 的定义就是 "must complete before"。
ORDERING_RELATIONS: Tuple[str, ...] = ("depends_on", "requires", "follows")


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    value: str


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    relation: str


@dataclass
class DelegationGraph:
    """一次委派 = 一个有类型的 DAG（论文用 constrained decoding 保证合法 JSON）。"""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def validate(self) -> bool:
        ids = [n.id for n in self.nodes]
        assert len(ids) == len(set(ids)), "节点 id 必须唯一"
        for n in self.nodes:
            assert n.type in NODE_TYPES, "未定义的节点类型: %s" % n.type
        for e in self.edges:
            assert e.relation in EDGE_RELATIONS, "未定义的边语义: %s" % e.relation
            assert e.src in ids and e.dst in ids, "悬空边: %r" % (e,)
        return True

    def ordering_edges(self) -> List[GraphEdge]:
        return [e for e in self.edges if e.relation in ORDERING_RELATIONS]

    def topological_order(self) -> List[str]:
        """graph-aware executor 的核心动作：按 depends_on / requires / follows 拓扑遍历。

        论文 §2.1 要求接收方 prompt「specifies topological traversal」——
        依赖边没被拓扑遍历消费掉，等于图根本没被读懂（见 graph_executor）。
        """
        self.validate()
        indeg: Dict[str, int] = {n.id: 0 for n in self.nodes}
        adj: Dict[str, List[str]] = {n.id: [] for n in self.nodes}
        for e in self.ordering_edges():
            if e.dst not in adj[e.src]:
                adj[e.src].append(e.dst)
                indeg[e.dst] += 1
        ready = deque(sorted(i for i, d in indeg.items() if d == 0))
        order: List[str] = []
        while ready:
            cur = ready.popleft()
            order.append(cur)
            for nxt in sorted(adj[cur]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        assert len(order) == len(self.nodes), "存在环：依赖图不是 DAG"
        return order

    def to_json(self) -> str:
        """论文用 constrained decoding 直出合法 JSON；这里等价地做序列化。"""
        return json.dumps(
            {
                "nodes": [{"id": n.id, "type": n.type, "value": n.value}
                          for n in self.nodes],
                "edges": [{"src": e.src, "dst": e.dst, "relation": e.relation}
                          for e in self.edges],
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# 2) 任务模型与两种交接格式下的执行结果（构造数据）
# ---------------------------------------------------------------------------
# 落地到母婴出海时，analysis_pattern 只用于**事后分组统计**；
# router 看不到它（论文的核心主张之一：判据必须来自任务内容，不能来自标签）。
TASK_PATTERNS: Tuple[str, ...] = ("dependency_chain", "aggregate", "adaptive",
                                  "mixed_partial_ordering")


@dataclass
class Task:
    task_id: str
    description: str          # router 只能看这个（+ 结构特征）
    steps: List[str]
    nl_misorders: bool        # NL 交接下，执行器是否会错序 / 丢前置条件
    needs_backtracking: bool  # 环境反馈后是否需要回溯（图会锁死这种情况）
    analysis_pattern: str     # 仅用于分组统计，router 不可见


@dataclass
class Outcome:
    ok: bool
    detail: str
    visited: List[str] = field(default_factory=list)


def nl_executor(task: Task) -> Outcome:
    """自然语言交接：执行顺序与前置条件靠执行器从散文里自行推断。"""
    if task.nl_misorders:
        return Outcome(False, "错序 / 丢前置条件（NL 未显式表达依赖）", list(task.steps))
    return Outcome(True, "文本自洽，顺序被正确推断", list(task.steps))


def graph_executor(task: Task, graph: DelegationGraph,
                   graph_aware_prompt: bool) -> Outcome:
    """类型化图交接。

    graph_aware_prompt=False 对应论文的关键事实：同一份 JSON 交给标准 executor prompt，
    子 agent 把图当成不透明数据、读不出依赖结构，收益归零。这里如实建模为
    「行为退化成 NL」——增益**严格等于 0**，而不是「略差一点」。
    """
    if not graph_aware_prompt:
        base = nl_executor(task)
        return Outcome(base.ok,
                       "无 graph-aware prompt → 图被当作不透明数据，退化为 NL 行为："
                       + base.detail,
                       base.visited)
    order = graph.topological_order()
    if task.needs_backtracking:
        return Outcome(False, "刚性拓扑阻止回溯（图强制提前承诺计划）", order)
    return Outcome(True, "按拓扑序执行，依赖被强制满足", order)


def build_graph(task: Task) -> DelegationGraph:
    """把一次委派编码成类型化依赖图：goal + 逐步 tool_call + depends_on 链。

    业务上「合规认证要求」这类硬约束要用 constraint / precondition 节点显式传递——
    这正是论文所说的「把 NL 留在隐含处的东西显式化」。
    """
    nodes: List[GraphNode] = [GraphNode("g1", "goal", task.description)]
    edges: List[GraphEdge] = []
    prev = None
    for k, step in enumerate(task.steps, 1):
        nid = "t%d" % k
        nodes.append(GraphNode(nid, "tool_call", step))
        edges.append(GraphEdge("g1", nid, "targets"))
        if prev is not None:
            # depends_on = 必须在前一步完成之后才能执行
            edges.append(GraphEdge(prev, nid, "depends_on"))
        prev = nid
    nodes.append(GraphNode("c1", "constraint", "合规与预算硬约束必须在执行前满足"))
    edges.append(GraphEdge("c1", "t1", "requires"))
    return DelegationGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# 3) 规则替身 router（**不是**论文的 LLM router）
# ---------------------------------------------------------------------------
# 论文 router 的抽象模式（原文见 ⑥ Q15）：
#   有序子任务（聚合 / 多步查找 / 顺序 API 调用）→ GRAPH
#   迭代 / 条件 / 自由文本解释 / 自适应推理      → NL
# 且默认回落 NL（论文的第一个设计选择：零牺牲 NL 的胜场）。
#
# ⚠️ 下面是**关键词规则替身**，只复现决策结构，不具备语义判断能力。
#    真实落地请替换为一次 LLM 分类调用（temperature=0）。
ADAPTIVE_MARKERS: Tuple[str, ...] = (
    "条件", "视情况", "看情况", "如果", "自由文本", "解释", "协商", "迭代", "临场", "不一定",
)
ORDERED_MARKERS: Tuple[str, ...] = (
    "先", "然后", "再", "最后", "依次", "汇总结", "归并", "汇总", "按顺序", "逐步",
)


def route_format(task_description: str) -> str:
    """返回 'GRAPH' 或 'NL'。默认 NL —— 保守默认是论文的三个设计选择之一。"""
    if any(m in task_description for m in ADAPTIVE_MARKERS):
        return "NL"
    if any(m in task_description for m in ORDERED_MARKERS):
        return "GRAPH"
    return "NL"


# ---------------------------------------------------------------------------
# 4) 三种系统在同一批任务上的对照（论文 §3.3 的 router 消融）
# ---------------------------------------------------------------------------
STRATEGIES: Tuple[str, ...] = ("NL", "GRAPH", "GRAPH_NO_PROMPT", "ROUTED")


def run_suite(tasks: Sequence[Task], strategy: str) -> Dict[str, Dict[str, int]]:
    """跑一批任务，返回 {analysis_pattern: {n, ok}}。"""
    assert strategy in STRATEGIES, "未知策略: %s" % strategy
    stats: Dict[str, List[int]] = {}
    for t in tasks:
        g = build_graph(t)
        if strategy == "NL":
            out = nl_executor(t)
        elif strategy == "GRAPH":
            out = graph_executor(t, g, graph_aware_prompt=True)
        elif strategy == "GRAPH_NO_PROMPT":
            out = graph_executor(t, g, graph_aware_prompt=False)
        else:  # ROUTED：router 按任务内容选，再走对应执行器
            chosen = route_format(t.description)
            out = (graph_executor(t, g, graph_aware_prompt=True) if chosen == "GRAPH"
                   else nl_executor(t))
        bucket = stats.setdefault(t.analysis_pattern, [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if out.ok else 0
    return {k: {"n": v[0], "ok": v[1]} for k, v in stats.items()}


def make_suite() -> List[Task]:
    """构造样本（**不是论文数据**，仅用于让机制可执行、可断言）。

    - dependency_chain：选品分析 → 合规校验 → 定价，顺序不可换。
    - aggregate：抓取 → 归并 → 汇总，同样有序但没有自适应成分
      （对应论文 AppWorld 里唯一适合图化的 aggregate 模式）。
    - adaptive：退货协商 / 异常物流，需要条件分支与临场解释。
    - mixed_partial_ordering：描述里有「先…然后…」的顺序词，但任务实际需要条件回溯
      —— 这正是论文 Appendix D 记录的 router 误判方向（把带部分序结构的任务当成纯聚合）。
      刻意放进样本：一个从不误判的替身会比论文的 router 还乐观，那不是忠实的替身。
    """
    tasks: List[Task] = []
    for i in range(20):
        tasks.append(Task(
            task_id="chain-%02d" % i,
            description="先做选品分析，然后把认证要求交给合规校验，最后再定价并汇总毛利",
            steps=["选品分析", "合规校验", "定价"],
            nl_misorders=(i % 4 != 0),
            needs_backtracking=False,
            analysis_pattern="dependency_chain",
        ))
    for i in range(5):
        tasks.append(Task(
            task_id="agg-%02d" % i,
            description="依次抓取各站点销量，然后归并到 SKU 维度，最后汇总",
            steps=["抓取", "归并", "汇总"],
            nl_misorders=(i % 3 == 0),
            needs_backtracking=False,
            analysis_pattern="aggregate",
        ))
    for i in range(15):
        tasks.append(Task(
            task_id="adapt-%02d" % i,
            description="客户要求退货：需要看情况判断是否在政策窗口内，可能要走条件分支并协商补偿",
            steps=["判定政策", "协商补偿"],
            nl_misorders=False,
            needs_backtracking=True,
            analysis_pattern="adaptive",
        ))
    for i in range(5):
        tasks.append(Task(
            task_id="mixed-%02d" % i,
            description="先拉取各站点库存然后归并，若某站点接口报错则改走备用源并重新汇总",
            steps=["拉取库存", "归并", "异常处理"],
            nl_misorders=False,
            needs_backtracking=True,
            analysis_pattern="mixed_partial_ordering",
        ))
    return tasks


# ---------------------------------------------------------------------------
# 5) 测试：断言直接对应论文的关键事实（外加一条「替身不许比论文乐观」）
# ---------------------------------------------------------------------------
def test_schema_vocabulary_matches_paper():
    assert len(NODE_TYPES) == 8
    assert len(EDGE_RELATIONS) == 7
    assert "depends_on" in EDGE_RELATIONS


def test_topological_order_respects_depends_on():
    t = make_suite()[0]
    g = build_graph(t)
    assert g.validate() is True
    order = g.topological_order()
    assert order.index("t1") < order.index("t2") < order.index("t3")
    assert order.index("c1") < order.index("t1")   # requires 边：约束先于动作
    assert len(order) == len(g.nodes)
    assert json.loads(g.to_json())["nodes"][0]["type"] == "goal"


def test_schema_rejects_unknown_vocabulary():
    bad = DelegationGraph(nodes=[GraphNode("n1", "not_a_type", "x")])
    try:
        bad.validate()
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("未定义的节点类型必须被拒绝")


def test_dependency_chain_goes_graph_and_nl_is_worse():
    """断言 (i)：依赖链任务应由 router 判给 GRAPH，且 NL 在其上更差。"""
    chain = [t for t in make_suite() if t.analysis_pattern == "dependency_chain"]
    assert all(route_format(t.description) == "GRAPH" for t in chain)
    nl = run_suite(chain, "NL")["dependency_chain"]["ok"]
    gh = run_suite(chain, "GRAPH")["dependency_chain"]["ok"]
    assert gh > nl


def test_without_graph_aware_prompt_the_gain_vanishes():
    """断言 (ii)：缺 graph-aware executor prompt 时收益归零（论文的关键事实）。"""
    chain = [t for t in make_suite() if t.analysis_pattern == "dependency_chain"]
    nl = run_suite(chain, "NL")["dependency_chain"]["ok"]
    no_prompt = run_suite(chain, "GRAPH_NO_PROMPT")["dependency_chain"]["ok"]
    with_prompt = run_suite(chain, "GRAPH")["dependency_chain"]["ok"]
    assert no_prompt == nl          # 收益严格为 0：图没被读懂，等于没换格式
    assert with_prompt > no_prompt


def test_router_falls_back_to_nl_on_adaptive_tasks():
    """断言 (iii)：router 对非依赖链（自适应 / 条件分支）任务回落 NL。"""
    adapt = [t for t in make_suite() if t.analysis_pattern == "adaptive"]
    assert all(route_format(t.description) == "NL" for t in adapt)


def test_graph_only_regresses_on_adaptive_but_router_does_not():
    """论文 §3.3 的 router 消融：graph-only 在自适应任务上回退，router 消除回退。"""
    adapt = [t for t in make_suite() if t.analysis_pattern == "adaptive"]
    nl = run_suite(adapt, "NL")["adaptive"]["ok"]
    gh = run_suite(adapt, "GRAPH")["adaptive"]["ok"]
    routed = run_suite(adapt, "ROUTED")["adaptive"]["ok"]
    assert gh < nl
    assert routed >= nl


def test_rule_router_reproduces_the_known_misroute_direction():
    """替身不许比论文的 router 更乐观。

    论文 Appendix D 记录：router 会误判 15 个「带部分序结构」的任务（把多 App 任务当成
    纯聚合），这 3.4 pp 的损失计在 router 误差里。规则替身必须复现同一个失败方向，
    否则它就成了一个「零误判」的理想 router —— 那种东西不是本论文的机制。
    """
    mixed = [t for t in make_suite() if t.analysis_pattern == "mixed_partial_ordering"]
    assert mixed, "样本里必须含带部分序结构的任务"
    # 判据只看描述 → 命中顺序词即图化，于是这一类被错判成 GRAPH
    assert all(route_format(t.description) == "GRAPH" for t in mixed)
    nl = run_suite(mixed, "NL")["mixed_partial_ordering"]["ok"]
    routed = run_suite(mixed, "ROUTED")["mixed_partial_ordering"]["ok"]
    assert routed < nl          # 残余误差：router 也会把不该图化的任务图化
    assert nl > 0


# ---------------------------------------------------------------------------
# 6) 业务演示
# ---------------------------------------------------------------------------
def _business_demo() -> None:
    tasks = make_suite()
    print("== 构造样本：跨境母婴运营 agent 的四类委派 ==")
    for idx in (0, 20, 25, 40):      # chain-00 / agg-00 / adapt-00 / mixed-00
        t = tasks[idx]
        g = build_graph(t)
        print("  [%s] router=%s" % (t.task_id, route_format(t.description)))
        print("      NL成功=%s | GRAPH+prompt成功=%s | GRAPH无prompt成功=%s"
              % (nl_executor(t).ok,
                 graph_executor(t, g, graph_aware_prompt=True).ok,
                 graph_executor(t, g, graph_aware_prompt=False).ok))
        print("      拓扑序：%s" % " -> ".join(g.topological_order()))

    print("\n== 四种策略的构造样本成功率（非论文数据，不可与论文 pp 数对比）==")
    for strategy in STRATEGIES:
        stats = run_suite(tasks, strategy)
        n = sum(v["n"] for v in stats.values())
        ok = sum(v["ok"] for v in stats.values())
        print("  %-16s 总 %d/%d" % (strategy, ok, n))


if __name__ == "__main__":
    _business_demo()
```

---

## ④ 技能关联

- **前置｜`Skill-Subagent-Decomposition.md`**（同域）：那张卡回答「任务要不要拆、按什么切」，
  本卡回答「拆开之后两个 agent 之间那句话怎么写」。数据流是串联的：拆分决定**有几个 sub-agent、
  各自负责什么**，本卡的依赖图 schema 决定**它们之间的 depends_on 边怎么画**。没有拆分就没有交接。
- **同域｜`Skill-MAS-Orchestrator.md`**：编排器定义拓扑（谁跟谁说话），本卡补上格式（怎么说话）。
  论文对现有框架的批评正是这一点——拓扑是显式的，格式却被默认交给自然语言；
  所以本卡是编排器那层的**格式层补充**，不是替代。
- **对照｜`Skill-MetaGPT-SOP-Driven-Collaboration.md`**：SOP 驱动的协作是一种**静态结构**——
  流程写死在文档里，对所有任务一视同仁。本卡给出的反面证据是：同一种结构在自适应任务上会回退
  （AppWorld 的 graph-only 回退），所以静态结构必须配一个「什么时候不用它」的判据，
  这正是 router 的角色。两张卡组合起来 = 「稳定流程用 SOP，判据层用 router」。
- **对照｜`Skill-AutoGen-Multi-Agent-Conversation.md`**：AutoGen 式框架的默认交接就是自然语言会话。
  本卡不是要否定它，而是给出**「什么时候不该用自然语言」的可操作判据**（有序依赖链）与
  **「什么时候必须回到自然语言」的判据**（迭代 / 条件 / 自由文本解释）。
  实践上可以把本卡的 router 插在 AutoGen 的 handoff 之前，作为格式开关。
- **延伸｜`Skill-ReAct-Reasoning-Acting.md`**（同域）：ReAct 的思考-行动循环天然是「迭代型」，
  按论文的判据属于**必须回落 NL** 的那一类；若把 ReAct 子 agent 的委派强行图化，
  就等于复现论文里「刚性拓扑阻止回溯」的失败模式。这条关联是**负向**的——知道什么时候不组合，
  比知道什么时候组合更有价值。

---

## ⑤ 商业价值评估

**ROI 公式**（论文**只报告过一个金额**——router 单次分类调用约 0.0005 美元；其余成本量级论文全部未给，
故本卡只给公式与参数来源，不代入未经核实的数字）：

**节省 = (T_NL − T_graph_total) × N_daily × p_token**

| 参数 | 含义 | 来源 |
|---|---|---|
| `T_NL` | 每次交接的自然语言 token 数 | 企业自测：在自家 trace 里直接量 |
| `T_graph_total` | 图路径的**完整**单次委派 token（图交接 + router 调用 + graph-aware executor prefill） | 企业自测。论文在 τ-retail 上的全口径实测是 461 vs 730，即 1.6×；**不要用 3.2× 做预算**——3.2× 只是 τ-retail 的交接 token 单点值，加权平均只有 2.1×，而 AppWorld 只有 1.04× |
| `N_daily` | 每日委派次数 | 企业自有编排日志 |
| `p_token` | 单 token 价格（输入 / 输出不同价，建议分别乘） | 企业自有账单 |

第二块收益来自**错序重试的消除**：

**重试收益 = R_retry × C_step × F_chain**

`R_retry` 是依赖链失败时被省下的重试步数（论文报告依赖链失败可省 15–27 个重试步），`C_step` 是单步成本
（工具调用 + 该步 token），`F_chain` 是企业任务流里依赖链任务的失败频次。这三个参数**全部需要企业自测**。

第三块（最容易被忽略、也可能是最大的一块）是**避免回退**：论文的 AppWorld 实测说明，无脑图化会让
成功率掉 −14.6 pp；router 的价值就是把这块损失挡在门外。要把它折成钱，需要先量出自己任务流里
依赖链任务与自适应任务的比例——论文的 router 在 AppWorld 上把 89% 判给了 NL，提示这个比例
很可能远低于直觉。

- **实施难度**：⭐⭐⭐☆☆ —— schema、graph-aware prompt、router 判据表都可以先手工落地（论文的 schema
  本身就是手工设计的，没有自动生成）；真正的工程量在**改造 sub-agent 的输入契约**与**补建依赖标注 /
  失败归因**。前者是一次性投入，后者可以按论文的三条规则从既有 trace 里机械地跑出来。
- **优先级**：⭐⭐⭐⭐☆ —— 对已经在拆 sub-agent 的团队，这是「先做对、再省钱」的卡：
  先按 ①b 第 2 条把 graph-aware executor prompt 补上（否则一切收益为零），再谈压缩率与成本。
  如果团队还没拆 sub-agent，优先级降到 2 星，先做 `Skill-Subagent-Decomposition.md`。
- **评估依据**：收益侧的不确定性主要是「自己任务流里依赖链占比多少」，成本侧的不确定性主要是
  改造契约的工时。论文给的全是 token 与成功率的**相对口径**，唯一一个金额是 router 单次调用的
  0.0005 美元（且是论文所用供应商的价目，不能直接当自家单价），
  所以本卡**不给收益金额结论**，只给口径、判据与三条必须企业自测的参数。

---

## ⑥ 原文引用

**A. 核心主张、机制与规模**

> 原文："Multi-agent LLM systems coordinate through natural-language messages that consume 40–60% of their token budget."
> 出处：2608.25277 §Abstract｜Q1

> 原文："We propose Routed Graph Handoff, where a lightweight LLM router (155 tokens, 0.15% overhead) selects between a typed dependency graph and natural language for each delegation."
> 出处：2608.25277 §Abstract｜Q2

> 原文："On four benchmarks (1,050+ trajectories), the routed system matches or exceeds NL-only on every task: +12.7 pp on $\tau$-retail at 3.2$\times$ compression ($p{<}0.01$), +8.7 pp on BrowseComp at 2.2$\times$ compression ($p{<}0.05$), and parity on BFCL and AppWorld."
> 出处：2608.25277 §Abstract｜Q3

> 原文："Without the router, graph-only delegation regresses 14.6 pp on AppWorld; the router eliminates this at near-zero cost."
> 出处：2608.25277 §Abstract｜Q4

> 原文："A graph-aware executor prompt is required: the same schema without interpretation guidance yields no gain."
> 出处：2608.25277 §Abstract｜Q5

> 原文："An oracle analysis reveals 8.6 pp of additional headroom, motivating execution-time adaptive routing as future work."
> 出处：2608.25277 §Abstract｜Q6

> 原文："Error analysis on 345 multi-agent trajectories reveals that 76% of failures stem from inter-agent misalignment: the executor misinterprets ordering constraints, drops prerequisites, or loops on ambiguous instructions."
> 出处：2608.25277 §1 Introduction｜Q7

> 原文："We resolve this tradeoff with Routed Graph Handoff (Figure 1): a lightweight LLM router (${\sim}$155 tokens, 0.15% overhead) selects graph or NL per delegation based on task computational pattern."
> 出处：2608.25277 §1 Introduction｜Q9

**B. schema、router 与 graph-aware executor**

> 原文："Each delegation is encoded as a typed DAG with 8 node types (goal, constraint, entity, action, precondition, postcondition, tool_call, tool_arg) and 7 edge relations (requires, targets, blocks, enables, depends_on, contradicts, follows)."
> 出处：2608.25277 §2.1 Native Graph Handoff Schema｜Q10

> 原文："We design such a schema (8 node types, 7 edge relations) emitted via constrained decoding at ${\sim}$350 tokens per delegation (2$\times$ compression vs. NL)."
> 出处：2608.25277 §1 Introduction（schema 规模与压缩口径）｜Q8

> 原文："The schema was designed iteratively on 47 $\tau$-bench trajectories."
> 出处：2608.25277 §2.1 Native Graph Handoff Schema｜Q11

> 原文："This receiver-side instruction is essential: passing the same JSON to a standard executor prompt yields no gain, and on $\tau$-retail restoring it lifts NGH from below NL to +12.7 pp (Appendix E)."
> 出处：2608.25277 §2.1 Graph-aware execution is part of the interface｜Q12

> 原文："We therefore treat the typed graph and its interpretation guidance as a single mechanism, not the graph alone."
> 出处：2608.25277 §2.1 Graph-aware execution is part of the interface｜Q13

> 原文："Without this prompt (i.e., passing the JSON graph to a standard executor prompt), the sub-agent treats the graph as opaque data and fails to interpret the dependency structure."
> 出处：2608.25277 §Appendix E Graph-Aware Executor Prompt｜Q45

> 原文："Before each delegation, a single classification call (${\sim}$155 tokens total, $0.0005) decides whether to use graph or NL."
> 出处：2608.25277 §2.2 LLM Router｜Q14

> 原文："Pick GRAPH if the task requires deterministic answers that depend on ordered sub-tasks (aggregations, multi-step lookups, sequential API calls). Pick NL if the task requires iteration, conditionals, free-text interpretation, or adaptive reasoning."
> 出处：2608.25277 §2.2 LLM Router（router prompt 原文）｜Q15

> 原文："Conservative default: NL unless dependency-chain pattern is detected. This ensures zero NL wins are sacrificed."
> 出处：2608.25277 §2.2 LLM Router（保守默认）｜Q16

> 原文："Deterministic: temperature = 0, verified identical across 3 independent runs."
> 出处：2608.25277 §2.2 LLM Router（确定性）｜Q17

> 原文："The per-benchmark rates we report are therefore a post-hoc aggregate of these blind per-task decisions: 100% graph on BrowseComp/$\tau$-retail/BFCL; 11% graph / 89% NL on AppWorld; 2% graph on $\tau$-airline."
> 出处：2608.25277 §2.2 LLM Router（逐 benchmark 的 post-hoc 聚合）｜Q18

> 原文："That the same classifier splits AppWorld itself 11%/89% (which a fixed per-benchmark rule cannot do) confirms the decision is made per task, not per domain; the clustering by benchmark arises because within each of these benchmarks nearly every task shares the same better format."
> 出处：2608.25277 §2.2 LLM Router（per-task 而非 per-domain 的证据）｜Q19

**C. 主结果、规模与错因**

> 原文："We evaluate on four diverse multi-agent tasks: BrowseComp (Wei et al., 2025) (150 trials, long-horizon web search requiring multi-step evidence gathering), BFCL v3 (Patil et al., 2025) (600 trials, Berkeley Function Calling Leaderboard with complex API sequences), $\tau$-bench retail (Yao et al., 2025) (150 paired trials: 50 tasks $\times$ 3 seeds, multi-step customer service with tool calls), and AppWorld (Trivedi et al., 2024) (152 paired trials, multi-app tool use with conditional logic). Total: 1,052 trajectories."
> 出处：2608.25277 §3 Experiments（四个 benchmark / 1,052 trajectories）｜Q20

> 原文："Splits. Pinned 50 $\tau$-retail tasks $\times$ 3 seeds; BrowseComp 150; BFCL v3 600; AppWorld 152; $\tau$-airline 150. Total 1,052 trajectories (plus 150 $\tau$-airline for the router ablation)."
> 出处：2608.25277 §Appendix J Artifacts and Reproducibility（Splits）｜Q48

> 原文："*Table 1: Main results (task success / accuracy %). Routed matches or exceeds NL on all four benchmarks. $\tau$-retail: +12.7 pp (150 paired trials, $p{<}0.01$). BrowseComp: +8.7 pp, CI [+2.7, +14.7], $p{<}0.05$. AppWorld NGH-only regresses $-$14.6 pp; the router recovers parity.*"
> 出处：2608.25277 §3.1 Main Results（Table 1 表注）｜Q22

> 原文："The router’s primary function is regression prevention (Table 1). NGH delivers significant gains"
> 出处：2608.25277 §3.1 Main Results（router 的首要功能）｜Q23

> 原文："NGH delivers significant gains on dependency-chain tasks: +12.7 pp on $\tau$-retail (150 paired trials; $p{<}0.01$) and +8.7 pp on BrowseComp (CI [+2.7, +14.7]; $p{<}0.05$). Both are statistically significant after Holm-Bonferroni correction. However, NGH regresses sharply on AppWorld: $-$14.6 pp (CI [$-$22.8, $-$6.4])."
> 出处：2608.25277 §3.1 Main Results（显著性与 AppWorld 回退 CI）｜Q24

> 原文："By defaulting to NL on 89% of AppWorld tasks (those involving iteration, conditionals, or free-text interpretation), it recovers full parity (51.7% vs. 51.7%)."
> 出处：2608.25277 §3.1 Main Results（router 恢复 parity）｜Q25

> 原文："Some benchmarks are not natively multi-agent (BFCL, for instance, is function calling), but casting it this way tests whether the graph preserves complex API-sequence structure without harm; the parity we observe (75.4 vs. 75.3) is the expected outcome for a task with no cross-step dependency structure to make explicit."
> 出处：2608.25277 §3 Handoff harness（BFCL 打平）｜Q21

> 原文："The 76% inter-agent misalignment figure derives from an automated error taxonomy (MAST, Multi-Agent Systematic Taxonomy) applied to 345 $\tau$-bench trajectories across three protocols (single-agent, NL multi-agent, graph multi-agent; 115 tasks $\times$ 3 seeds each)."
> 出处：2608.25277 §Appendix C Misalignment Annotation Methodology（345 条轨迹口径）｜Q49

> 原文："Of all multi-agent failures, 76% are inter-agent misalignment: the executor misinterprets ordering, drops prerequisites, or enters retry loops from ambiguous instructions; this share is robust to the taxonomy’s thresholds (Appendix I)."
> 出处：2608.25277 §4 Analysis（76% 错位份额及阈值鲁棒性）｜Q34

> 原文："Classification is rule-based from trajectory logs (not human-annotated): a failure is “misalignment” if the trajectory contains (a) a tool call that returns an error due to missing prerequisites, (b) $\geq$3 consecutive retry steps on the same action, or (c) executor actions that contradict the delegation’s stated ordering."
> 出处：2608.25277 §Appendix C（错因分类是规则化、可复现的）｜Q46

**D. 效率与计费口径（本卡最容易被误读的一段）**

> 原文："Weighted across all trials, the routed system achieves 2.1$\times$ average handoff compression (BrowseComp 2.2$\times$, $\tau$-retail 3.2$\times$, BFCL 2.0$\times$, AppWorld 1.04$\times$)."
> 出处：2608.25277 §3.2 Efficiency（加权 2.1× 与四个 benchmark 的分项）｜Q26

> 原文："Compression and the 0.15% router overhead are measured over handoff tokens; accounting for the full per-delegation budget (the 155-token router call and the ${\sim}80$-token graph-aware executor prefill), the graph path still totals fewer tokens than NL (461 vs. 730 on $\tau$-retail, 1.6$\times$; Appendix H)."
> 出处：2608.25277 §3.2 Efficiency（全口径 461 vs 730 与 1.6×）｜Q27

> 原文："The 2.1$\times$ average compression and the 155-token (0.15%) router overhead reported in the main text are measured over handoff tokens."
> 出处：2608.25277 §Appendix H Total-Token and Latency Accounting（口径声明）｜Q47

> 原文："The graph’s dependency edges prevent executor spiraling (15–27 retry steps eliminated on dependency-chain failures)."
> 出处：2608.25277 §3.2 Efficiency（重试步数消除）｜Q28

> 原文："On AppWorld, the router preserves NL behavior, avoiding the 18% overhead that NGH-only incurs from failed graph executions."
> 出处：2608.25277 §3.2 Efficiency（AppWorld 上避免 18% 开销）｜Q29

**E. router 必要性、oracle 余量与替代方案**

> 原文："AppWorld: $-$14.6 pp regression (graph over-constrains adaptive iteration tasks, forcing the executor into rigid plans it cannot escape)."
> 出处：2608.25277 §3.3 Ablation: Router Necessity（AppWorld）｜Q30

> 原文："$\tau$-airline (150 additional trials): NGH-only regresses $-$4.0 pp; the router recovers parity by routing 98% to NL using the same prompt."
> 出处：2608.25277 §3.3 Ablation: Router Necessity（τ-airline）｜Q31

> 原文："A simpler non-LLM router that mapped benchmark label to format would reproduce this per-benchmark aggregate, but it would require the benchmark identity our router never sees and could not produce the within-AppWorld 11%/89% split; the LLM router’s value is exactly this label-free generalization from task content."
> 出处：2608.25277 §3.3 Ablation（更简单的非 LLM router 能做到什么、做不到什么）｜Q32

> 原文："Per-instance accuracy vs. oracle: on AppWorld’s 152 tasks, the router correctly identifies 15/15 aggregate tasks (100% precision) and correctly defaults to NL on 122/137 non-aggregate tasks (89% recall)."
> 出处：2608.25277 §Appendix D Router Decision Analysis（逐实例精度）｜Q43

> 原文："Oracle headroom decomposition: of the 8.6 pp gap between Routed (51.7%) and Oracle (60.3%), 5.2 pp comes from NGH rescues on tasks the router sends to NL, and 3.4 pp from NL rescues on tasks the router sends to GRAPH."
> 出处：2608.25277 §Appendix D（8.6 pp 余量的分解）｜Q44

> 原文："On aggregate tasks ($n$=15), NGH outperforms NL by +6.7 pp: edges enforce fetch-before-compute ordering (Table 3). On iterate ($n$=43) and conditional ($n$=11) tasks, NL outperforms by 7–18 pp: rigid edges prevent adaptive backtracking."
> 出处：2608.25277 §4 Analysis（按任务模式的正负号）｜Q35

> 原文："Complementarity is substantial: NGH rescues 9.9% of NL failures; NL rescues 19.7% of NGH failures. The oracle achieves 60.3% TSR (8.6 pp headroom)."
> 出处：2608.25277 §4 Analysis（互补性与 oracle 上界）｜Q36

> 原文："Routed NGH (+12.7 pp, 3.2$\times$) is the only zero-training protocol and achieves the highest TSR in the comparison, outperforming even trained compressors."
> 出处：2608.25277 §3.4 Protocol Comparison（唯一零训练协议拿到最高 TSR）｜Q33

> 原文："In the protocol comparison (Table 2), schema-unaware re-encodings that still hand the executor an explicit plan (TF-IDF and Predictive Delta) gain only +4.7 to +5.3 pp, against the typed graph’s +12.7 pp."
> 出处：2608.25277 §4 Isolating the typed-graph contribution｜Q37

**F. 可移植性与论文自承局限**

> 原文："Cross-vendor validation (Claude $\times$ Nova Pro) further shows 0% invalid JSON with 3.1–3.6$\times$ compression preserved, confirming the schema is a portable artifact."
> 出处：2608.25277 §4（跨厂商格式可移植性）｜Q38

> 原文："The direction of the effect (Routed $\geq$ NL on every family) is preserved across a different vendor and model family, consistent with the schema being a portable artifact rather than a Sonnet-specific behavior."
> 出处：2608.25277 §Appendix G Second Orchestrator Backbone｜Q50

> 原文："This adds accuracy portability to the earlier cross-vendor check (Claude $\times$ Nova Pro: 0% invalid JSON, 3.1–3.6$\times$ compression preserved), which had established only format portability."
> 出处：2608.25277 §3.1 Second orchestrator backbone｜Q51

> 原文："Main results use a single orchestrator backbone (Claude Sonnet 4.5); we additionally confirm accuracy portability on a second orchestrator (GPT-5 mini, Appendix G) and format portability across model families, but broad multi-model replication remains future work."
> 出处：2608.25277 §Limitations（单 backbone 与其补救）｜Q42

> 原文："The router is a single per-task classifier applied blind to benchmark identity, but on these benchmarks its decisions cluster by task type, so realized routing is coarse (100% graph on dependency-chain benchmarks; 89% NL on AppWorld) rather than fine-grained per-instance adaptation."
> 出处：2608.25277 §Limitations（routing 粒度粗）｜Q39

> 原文："Even so, the schema may not generalize to domains with fundamentally different coordination patterns (e.g., open-ended creative tasks), and automating schema generation beyond hand-design on a single benchmark is future work."
> 出处：2608.25277 §Limitations（schema 泛化性）｜Q40

> 原文："Finally, the graph-aware executor prompt is a necessary complement to the schema; systems integrating this approach must include interpretation guidance for the receiving agent."
> 出处：2608.25277 §Limitations（graph-aware prompt 是必要补充）｜Q41
