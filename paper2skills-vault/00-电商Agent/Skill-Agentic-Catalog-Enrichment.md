---
title: Skill-Agentic-Catalog-Enrichment
module: 00-电商Agent
topic: 多源证据接地的商品目录属性补全（Scout 取证 → Judge 裁决 → verify-before-write 写库门）
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.20844
paper: TRACE: Agentic Catalog Enrichment with Multi-source Evidence Grounding
venue: arXiv preprint
venue_tier: preprint
evidence_grade: A
verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py G2/G3 passed + 人工抽检 6 处数字回原文
supersedes:
related: Skill-KG-Auto-Construction-Agent-Driven.md, Skill-Multi-Agent-Debate.md, Skill-Agent-Stage-Evaluation.md, Skill-Dense-Retrieval-Ecommerce-Semantic-Search.md, Skill-Live-Catalog-Conversational-Rec.md
---

# Skill Card: 多源证据接地的目录属性补全（Agentic Catalog Enrichment, verify-before-write）

**这张卡解决什么**：商品目录的属性格大量空着（「适用月龄」「材质」「认证」「安全警示」），
买家筛不到、下游检索与推荐只能用粗粒度表示；人工补全在规模与增速下不可行（⑥ Q1 / Q2）。
本卡把属性补全做成一条**带证据的写库流水线**：多源取证 → 独立裁决 → 三种写库动作，
每一格都能回答「这个值是谁说的、凭什么写进去」。

---

## ① 算法原理

**核心思想**：把属性补全从「让模型猜一个值」改成「先取证、再裁决、后写库」。ScoutAgent 汇集三类证据
——卖家目录（文本与图片）、第三方商品数据 feed、**按身份匹配过**的网页搜索——只在证据确实指向
这一个 SKU 时才采纳，并把候选值连同证据一起打包（⑥ Q13 / Q9）。JudgeAgent 独立复核该候选是否被
**它自己的证据**支持，给出四值裁决；写库策略再把裁决映射为写入 / 阻断 / 转人工。它被放在服务路径上
当「先验后写」闸门，使发布质量对商品结构漂移不敏感（⑥ Q10 / Q12）。

**数学直觉**：候选是一个带证据的元组 c_a=(a, v_a, E_a, τ_a, q_a, z_a)——属性、提案值、支撑证据、
来源类型、自报置信度、抽取状态（extracted / not_found / not_applicable / ambiguous / conflict，⑥ Q11）。
发布动作**不是提案值的函数**，而是 (裁决 y_a, 置信度 q_a) 的函数：q_a 低于阈值 θ 或裁决为 FAIL → BLOCK；
UNCERTAIN → REVIEW；PASS 或 UNVERIFIED → WRITE（⑥ Q21）。把裁决与写库策略拆开，改发布口径不必动裁决器。

**关键假设**：证据必须与目标商品**同一身份**，否则主题相关但属于别的变体的页面会把错的属性写到这个
SKU 上（⑥ Q15 / Q16）；判官须采用比侦察更严的证据标准（⑥ Q19）。

---

## ①b 反例与适用边界

**什么时候不要用这个方法**

1. **没有可用的身份标识符**（无 ASIN / GTIN / 型号）。身份匹配会退化成「标题像不像」，而论文明确
   警告检索页可能「主题相关但讲的是别的商品或别的变体」（⑥ Q15）；此时网页证据应当整体禁用，
   只保留自有来源。
2. **属性必须由权威来源背书**（医疗器械注册号、法定安全警示、平台强制合规字段）。论文的判官只回答
   「证据是否支持这个值」，**不回答「这个来源是否权威」**（⑥ Q19）。这类属性仍应走官方渠道 / 人工复核，
   不要让 UNCERTAIN 之外的格自动过闸。
3. **自有数据源已经能可靠推断的属性**（标题里就写死的品牌、颜色、容量）。论文的动机恰恰是
   「部分属性值**无法**从自有数据源可靠推断，必须外采并对照具体商品核验」（⑥ Q6）；对能推断的属性上
   agent 只增加成本。
4. **没有人工标注集做校准**。判官支撑率**不是**人工验证的精确率（⑥ Q25）；若不能像论文那样在至少一类
   目上做人审校准（⑥ Q41），就没有可信的发布口径 —— 此时只能把它当**运营监控信号**用。
5. **单点一次性清洗、且允许多轮返工**。本方法的主要成本是「每 SKU 一次侦察调用 + 一次裁决调用」（⑥ Q44）
   以及人工复核队列；对几十个 SKU 的一次性清洗，直接人工更快。

**已知的失败模式**

1. **相关性 ≠ 身份**。检索返回的页面可能讲的是近亲商品或不同变体；论文的对策是在抽取前做身份匹配，
   失败则整页丢弃（⑥ Q15 / Q16）。跳过这一步，等于把别人的属性写到你的 SKU 上。
2. **视觉证据更噪**。论文把图片证据与文本证据**分成两层**，因为视觉抽取噪声更大，但仍要用图片覆盖
   「材质、认证标志、包装声明」这类视觉表达的属性（⑥ Q14）。
3. **更强 / 更贵的 backbone 可能让发布覆盖率变差**。在固定判官的前提下换 Scout 的 VLM：抽取覆盖率
   只涨了 0.5 个百分点，但发布覆盖率从 85.5% 掉到 81.9%，推理成本涨了超过 7 倍（⑥ Q31）；
   另外两个 backbone 的发布覆盖率更低，分别是 73.2% 与 63.5%（⑥ Q32）。**用「抽得多」考核会选错模型**，
   必须以「够不够格自动发布」为准。错误集中在证据密集型属性（件数、自由文本描述）（⑥ Q33）。
4. **图片单源候选会被判 UNVERIFIED，而论文口径下 UNVERIFIED 仍会写入**（⑥ Q21）。对母婴合规属性
   这是危险默认值，企业侧应把 θ 调高或改成「只有 PASS 自动写」。
5. **人工复核队列可能被撑满**。论文给了四值裁决的分类逻辑（⑥ Q20），但**论文未报告** UNCERTAIN 的占比，
   本卡因此不对该比例做任何估计，落地前必须自己先量。

**论文自己承认的局限**

1. Electronics 与 Home Improvement 两个垂类用的是 **judge-supported rate**，它是可扩展的运营指标，
   **不能替代人工验证的精确率**；JudgeAgent 只在 Grocery 与 Alcohol 上做过人工校准，向其他类目的迁移
   只有有限的人工评测（⑥ Q41）。
2. 主配置里 ScoutAgent 与 JudgeAgent **使用同一模型家族**，因此可能出现**相关性失效**
   （correlated failure modes）（⑥ Q42）。
3. 线上实验测的是「生成 → 校验 → 在 PDP 上展示」的**整条链路**效果，**不能**把结果拆给 JudgeAgent
   或写库门（⑥ Q43）。
4. **裁判循环（本卡补充指出，论文未自陈）**：因为四个垂类都缺乏穷尽人工标注，完整数据集是**用 JudgeAgent
   自己裁决的**（⑥ Q24）。用被评估者当裁判，judge-supported rate 会系统性偏高 —— 所以卡片里凡是
   「人工口径」的数字（如 98.2%）只在 Grocery 与 Alcohol 的 500 SKU / 2,497 对上成立（⑥ Q23 / Q26）。
5. **论文未讨论**：母婴 / 跨境电商场景、跨语言证据源、与平台（Amazon / 独立站）目录 API 的对接方式、
   多语言属性值的归一；**论文未报告**任何工程成本、单 SKU 延迟、人工复核工作量或人力投入量级。
   本卡 ⑤ 自建 ROI 公式，并把必须由企业自测的参数单独标出。

---

## ② 母婴出海应用案例

### 场景 1：把在售 SKU 的「适用月龄 / 材质 / 认证 / 安全警示」补进 Listing 结构化属性与筛选器

- **业务问题**：母婴出海店铺（吸奶器、奶瓶、纸尿裤、辅食餐具）在亚马逊与独立站同时在卖。卖家在
  标题与五点描述里写了「BPA Free」「适合六个月以上」，但**结构化属性格是空的**：站内筛选器捞不到，
  搜「婴儿奶瓶 六个月」召回差，家长收到货发现月龄不符就退货、给差评；材质与安全警示缺失还会带来
  合规风险 —— 论文把这类属性单列为「后果尤其严重」的一类（⑥ Q8）。上新节奏是**每周**都有新品，
  人工把「SKU × 目标属性」逐格补齐追不上增速（⑥ Q2）。要做的事：**每周三**用流水线跑一遍新增与
  存量缺口 SKU，周四人工只看 REVIEW 队列，周五随 Listing 更新一起发布。
- **数据要求**：① SKU 主数据（内部 SKU、ASIN、品牌、型号、标题、五点描述、主图与附图 URL），
  **日粒度全量快照**；② 类目属性模板（每个叶子类目的目标属性清单，如奶瓶：材质 / 容量 / 奶嘴流量 /
  适用月龄 / 是否 BPA-free / 消毒方式），模板需要版本号；③ 供应商 feed（出厂规格表）；
  ④ 证据源：品牌官网产品页与权威第三方页面的可检索副本；⑤ 一个**人工裁决过的小校准集**
  （几百个 SKU–属性对即可起步），用来量判官的误收 / 误拒。**无历史长度要求**（这是快照式补全任务，
  不是时序预测），但必须保留模板版本与每次运行的 provenance，便于回溯口径变更。
- **数据可得性**：`部分可得（需补充 X）`。自有 Listing、图片、供应商 feed、上架属性表**企业内部可得**；
  品牌官网 / 权威页面需要自建抓取与缓存（须遵守 robots 与平台条款）；**缺**的是论文所用的第三方商业
  数据 syndicator 授权源，以及人工标注的参考值 —— 两者都要补。跨站点（英 / 德 / 日）还需多语言证据源，
  否则非英语站点的候选值会大量落进 UNVERIFIED。
- **预期产出**：（a）属性补全表，每格含**提案值 + 证据列表（来源类型 + 定位符）+ 四值裁决 + 写库动作**；
  （b）人工只审 UNCERTAIN 那一队（论文口径：UNCERTAIN 连同证据链一起转人工，⑥ Q21）；
  （c）三个分开的看板：抽取覆盖率、判官支撑率、发布覆盖率 —— 论文把前两者与「人工验证精确率」
  明确分开报告（⑥ Q25）。
- **业务价值**：见 ⑤ 的 ROI 公式。钱的来源有两块：① 补上的属性让 PDP 信息更完整，
  论文的线上实验正是测这个动作（在商品详情页展示补全后的属性）对结账转化与缺错件率的影响（⑥ Q5 /
  Q40）；② 人工审核量从「全量」压到「只审分歧格」。

### 场景 2：旺季前六周，把合规敏感属性的审核从「全量人工」改成「机器提案 + 人工只审分歧」

- **业务问题**：黑五前**六周**要上三百个新品，其中吸奶器、安抚奶嘴、辅食餐具带材质与安全警示属性。
  漏写或写错会被平台下架，也会引发客诉。现在的做法是全部人工逐格核对，旺季前必然排队。
  要做的事：让流水线先出提案，人工只处理 UNCERTAIN 与所有「只有图片证据」的格。
- **数据要求与可得性**：`部分可得（需补充 X）`。需要一份「金标」小集（人工裁决过的 SKU–属性对，
  覆盖每个敏感属性的正反例）来校准判官，并按类目分别量误收率与误拒率；金标集企业内部可建，
  但要投入人工标注工时 —— 论文的做法就是在 Grocery 与 Alcohol 上做人审校准，并明确其他类目只是
  有限人工评测（⑥ Q41）。
- **做法**：把置信度阈值 θ **调高**（更保守），并把写库策略改成「只有 PASS 自动写、UNVERIFIED 也进人工」，
  偏离论文默认口径（⑥ Q21）；再对合规属性加一条硬规则：**必须至少有一条自有来源之外的证据**
  （品牌官网或权威页面），否则一律 REVIEW。这条硬规则是企业在论文之上自行加的，论文没有这项要求。
- **预期产出**：旺季六周的每日 REVIEW 队列 + 发布覆盖率曲线；当 REVIEW 队列持续超过人工处理能力时，
  正确动作是**先扩人工审单能力再放量**，而不是放宽 θ。
- **业务价值**：把「审核人工」从固定成本变成随缺口波动的变量；但**论文未报告**人工复核的占比或
  单格耗时，本卡不给节省人力的百分比，只给 ⑤ 的核算公式由企业代入自测值。

---

## ③ 代码模板

- 依赖：**只用标准库**（仅 `re`），不调用任何模型 SDK、不发网络请求，因此能在断网环境下
  通过 K1 的 L4 执行与 L5 断言校验。
- ⚠️ **重要声明**：ScoutAgent 与 JudgeAgent 在本模板里是**确定性规则替身（rule-based stub）**，
  **不是论文里的 LLM Agent**。它复现的是论文的**决策结构**（多源取证 → 身份过滤 → 归一 / 弃权 →
  对照证据裁决 → 写库门 → 保留 provenance），**不消费任何 LLM 能力**。因此本模板的运行结果
  **不能**用来印证论文报告的任何数字（98.2% / 90.4% / +0.48% 等都来自论文自己的 LLM 生产管线），
  两者数值无关，也不可互相印证。
- 与论文的对应：`scout_agent` 实现 §3.2（三源取证 + 身份匹配 + 归一 + 弃权）；`judge_agent` 实现 §3.3
  的四值裁决；`write_policy` 实现式 (3)（⑥ Q21）；`Candidate` 对应式 (2) 的候选元组（⑥ Q11）；
  `run_trace` 实现 Algorithm 1 的「未产出值的属性不进核验与写库门」（⑥ Q27）。
- 代码内所有数据均为**合成示例数据**，不代表论文数据，也不代表任何企业的真实商品。

```python
"""
TRACE 式商品目录属性补全：Scout 多源取证 → Judge 对照证据裁决 → 写库门（verify-before-write）
论文：2608.20844 §3.1 式 (2)(3) / §3.2 The ScoutAgent / §3.3 The JudgeAgent

⚠️ 声明：本模块用**确定性规则替身（rule-based stub）**实现 ScoutAgent 与 JudgeAgent，
   不是论文里的 LLM Agent，不调用任何模型、不发任何网络请求。
   它只复现论文的**决策结构**，因此本模块的输出**不能**用来印证论文报告的任何结果。
"""

from __future__ import annotations

import re

# --- 论文 §3.2：三类证据来源；图片被单列一层（视觉抽取噪声更大） ---
SRC_CATALOG = "catalog"        # 卖家目录：结构化字段与图片
SRC_SYNDICATED = "syndicated"  # 第三方商业数据 feed
SRC_WEB = "web"                # 身份匹配后的网页搜索
SRC_IMAGE = "image"            # 商品图片
TEXT_SOURCES = (SRC_CATALOG, SRC_SYNDICATED, SRC_WEB)

# --- 论文 §3.1：抽取状态 z_a ---
ST_EXTRACTED = "extracted"
ST_NOT_FOUND = "not_found"
ST_AMBIGUOUS = "ambiguous"
ST_CONFLICT = "conflict"

# --- 论文 §3.3：四值裁决 ---
PASS, FAIL, UNVERIFIED, UNCERTAIN = "PASS", "FAIL", "UNVERIFIED", "UNCERTAIN"
# --- 论文式 (3) / §3.1：三个写库动作 ---
WRITE, BLOCK, REVIEW = "WRITE", "BLOCK", "REVIEW"

_SRC_PRIORITY = {SRC_CATALOG: 0, SRC_SYNDICATED: 1, SRC_WEB: 2, SRC_IMAGE: 3}

# 良性写法差异的归一规则（论文 §3.2 举的例子：NiMH / nickel-metal hydride，60 Hz / 60Hz）
_NORM_RULES = (
    (r"\bnimh\b", "nickel-metal hydride"),
    (r"\bnickel[-\s]?metal\s+hydride\b", "nickel-metal hydride"),
    (r"\bbpa[-\s]?free\b", "bpa free"),
    (r"\bphthalate[-\s]?free\b", "phthalate free"),
    (r"\b(\d+)\s*hz\b", r"\1 hz"),
)


def normalize_value(raw: str) -> str:
    """把良性写法差异归一到同一表示，避免「同一个值被当成两个值」而误判冲突。"""
    s = str(raw).strip().lower()
    s = re.sub(r"\s+", " ", s)
    for pat, repl in _NORM_RULES:
        s = re.sub(pat, repl, s)
    return s


class Evidence:
    """一条支撑证据（论文 §3.1 的 E_a 与来源类型 τ_a，也是 provenance 的载体）。

    刻意用普通类而不是 dataclass：本模块要能被「按文件路径 exec、模块名未登记进
    sys.modules」的探针直接导入（K1 的 L3 import 探针就是这样跑的），
    而 Python 3.12+ 的 dataclass 在该场景下会因 `sys.modules[cls.__module__]` 为 None 报错。
    """

    __slots__ = ("source", "value", "locator", "identity_ok")

    def __init__(self, source: str, value: str, locator: str, identity_ok: bool = True):
        self.source = source
        self.value = value
        self.locator = locator        # 定位符：字段名 / feed id / 图片 id / 页面 URL
        self.identity_ok = identity_ok  # 是否通过身份匹配（网页证据专用，论文 §3.2）

    @property
    def normalized(self) -> str:
        return normalize_value(self.value)

    def __repr__(self) -> str:
        return (f"Evidence({self.source!r}, {self.value!r}, {self.locator!r}, "
                f"identity_ok={self.identity_ok})")


class Candidate:
    """论文式 (2)：c_a = (a, v_a, E_a, τ_a, q_a, z_a)，外加裁决、诊断子类型与写库动作。"""

    __slots__ = ("sku", "attribute", "value", "evidence", "confidence",
                 "status", "verdict", "subtype", "action")

    def __init__(self, sku: str, attribute: str, value=None, evidence=None,
                 confidence: float = 0.0, status: str = ST_NOT_FOUND,
                 verdict: str = "", subtype: str = "", action: str = ""):
        self.sku = sku
        self.attribute = attribute
        self.value = value
        self.evidence = list(evidence) if evidence else []
        self.confidence = confidence
        self.status = status
        self.verdict = verdict
        self.subtype = subtype
        self.action = action

    @property
    def sources(self) -> list:
        return sorted({e.source for e in self.evidence if e.identity_ok})


def identity_match(page: dict, sku: dict) -> bool:
    """论文 §3.2 Identity-grounded web retrieval 的规则替身。

    只有能确认「这一页讲的就是这个 SKU」时才采纳其证据：
      ① 页面声明的 asin / gtin 与目标一致；或
      ② 无标识符时，品牌与型号必须同时出现在页面标题里。
    否则整页丢弃（论文：Evidence from p is used only when ... otherwise, the page is discarded）。
    """
    page_id = str(page.get("asin") or page.get("gtin") or "").strip().upper()
    sku_id = str(sku.get("asin") or sku.get("gtin") or "").strip().upper()
    if page_id and sku_id:
        return page_id == sku_id
    title = normalize_value(page.get("title", ""))
    brand = normalize_value(sku.get("brand", ""))
    model = normalize_value(sku.get("model", ""))
    return bool(brand and model and brand in title and model in title)


def _owned_evidence(attribute: str, sku: dict) -> list:
    """自有来源：卖家目录结构化字段 / 第三方 feed / 图片声明。"""
    ev = []
    cat = sku.get("catalog", {}).get("attributes", {})
    if attribute in cat:
        ev.append(Evidence(SRC_CATALOG, cat[attribute], "catalog.attributes"))
    for rec in sku.get("syndicated", []):
        if attribute in rec.get("attributes", {}):
            ev.append(Evidence(SRC_SYNDICATED, rec["attributes"][attribute],
                               rec.get("feed_id", "syndicated")))
    for img in sku.get("images", []):
        if attribute in img.get("claims", {}):
            ev.append(Evidence(SRC_IMAGE, img["claims"][attribute],
                               img.get("image_id", "image")))
    return ev


def _web_evidence(attribute: str, sku: dict, search_pages: list) -> list:
    """网页搜索证据：先过身份匹配，未过的不采纳。"""
    ev = []
    for page in search_pages:
        raw = page.get("attributes", {}).get(attribute)
        if raw is None:
            continue
        ev.append(Evidence(SRC_WEB, raw, page.get("url", "search"),
                           identity_ok=identity_match(page, sku)))
    return ev


def _is_resolved(ev: list) -> bool:
    """自有来源是否已能给出一致的答案（论文 Algorithm 1：unresolved 的属性才去检索）。"""
    if not ev:
        return False
    return len({e.normalized for e in ev}) == 1


def _pick_by_priority(ev: list) -> str:
    return sorted(ev, key=lambda e: _SRC_PRIORITY.get(e.source, 9))[0].value


def scout_agent(sku: dict, attributes: list, search_pages: list) -> dict:
    """ScoutAgent 规则替身（论文 §3.2）：取证 → 身份过滤 → 归一 → 合并或弃权。

    返回 {属性: Candidate}。注意：**弃权是合法结果**（not_found / ambiguous / conflict），
    论文要求证据不足时弃权，而不是用背景知识补一个值。
    """
    out = {}
    for attr in attributes:
        ev = _owned_evidence(attr, sku)
        if not _is_resolved(ev):
            ev = ev + [e for e in _web_evidence(attr, sku, search_pages) if e.identity_ok]
        cand = Candidate(sku=sku.get("sku", "?"), attribute=attr, evidence=ev)
        if not ev:
            cand.status = ST_NOT_FOUND
            out[attr] = cand
            continue
        distinct = {e.normalized for e in ev}
        if len(distinct) > 1:
            cand.status = ST_CONFLICT
            cand.value = _pick_by_priority(ev)   # 仍给出提案值，但交给判官裁决
            # 注意：q_a 是**侦察端自报的置信度**，不是「证据是否一致」的度量。
            # 侦察端完全可能一边给出一个值、一边自报中等置信度 —— 这正是判官存在的理由。
            # 这里的 0.65 是本替身的规则常数（与论文的模型自报分数无关），
            # 取在默认阈值之上，是为了让「冲突 → 转人工」这条路径在端到端上真的被走到。
            cand.confidence = 0.65
        elif not any(e.source in TEXT_SOURCES for e in ev):
            cand.status = ST_AMBIGUOUS           # 只有图片证据：视觉抽取更噪
            cand.value = ev[0].value
            cand.confidence = 0.70
        else:
            cand.status = ST_EXTRACTED
            cand.value = ev[0].value
            kinds = {e.source for e in ev}
            cand.confidence = 0.95 if len(kinds) >= 2 else 0.80
        out[attr] = cand
    return out


def judge_agent(cand: Candidate) -> tuple:
    """JudgeAgent 规则替身（论文 §3.3）：只对照候选值**自己的支撑证据**核验。

    返回 (verdict, subtype)：
      PASS        有文本类证据支持该值，且没有任何同身份证据反对
      UNVERIFIED  未被反对，但支撑证据只有图片 —— 「无法直接确认」而非「发现错误」
      UNCERTAIN   存在同身份证据给出**不同**的值 —— 证据互相矛盾
      FAIL        没有任何证据支持该值（HALLUCINATION / UNSUPPORTED）
    注意 FAIL 与 UNCERTAIN 的区别：前者是「没证据（或被证据反驳且无支持）」，
    后者是「证据在互相打架」，论文把两者分开正是为了让写库策略区别对待。
    """
    if not cand.evidence or cand.value is None:
        return FAIL, "UNSUPPORTED"
    target = normalize_value(cand.value)
    support = [e for e in cand.evidence if e.identity_ok and e.normalized == target]
    oppose = [e for e in cand.evidence if e.identity_ok and e.normalized != target]
    if support and oppose:
        return UNCERTAIN, "CONFLICT"
    if not support:
        return FAIL, "HALLUCINATION"
    if not any(e.source in TEXT_SOURCES for e in support):
        return UNVERIFIED, "VISUAL_ONLY"
    return PASS, ""


def write_policy(verdict: str, confidence: float, theta: float = 0.6) -> str:
    """论文式 (3)：写库动作由裁决与自报置信度共同决定。

    BLOCK  if q_a < theta or y_a == FAIL
    REVIEW if y_a == UNCERTAIN
    WRITE  if y_a in {PASS, UNVERIFIED}
    把裁决与写库策略分开，意味着调整发布口径（例如把 theta 调高、或让 UNVERIFIED 也转人工）
    不需要改动判官本身。
    """
    if confidence < theta or verdict == FAIL:
        return BLOCK
    if verdict == UNCERTAIN:
        return REVIEW
    return WRITE


def run_trace(sku: dict, attributes: list, search_pages: list,
              theta: float = 0.6) -> tuple:
    """端到端：Scout → Judge → 写库门。

    未产出值的属性（not_found）**不进入**核验与写库门 —— 与论文一致：
    Unpopulated attributes are excluded because only proposed values enter the
    production verification and write gate。
    返回 (candidates, provenance 记录列表)。
    """
    cands = scout_agent(sku, attributes, search_pages)
    rows = []
    for attr in attributes:
        c = cands[attr]
        if c.status == ST_NOT_FOUND:
            continue
        c.verdict, c.subtype = judge_agent(c)
        c.action = write_policy(c.verdict, c.confidence, theta)
        rows.append({
            "sku": c.sku,
            "attribute": c.attribute,
            "value": c.value,
            "status": c.status,
            "verdict": c.verdict,
            "subtype": c.subtype,
            "confidence": round(c.confidence, 2),
            "action": c.action,
            # provenance：每个值都能回答「谁说的」
            "evidence": " | ".join(
                f"{e.source}:{e.value}@{e.locator}" for e in c.evidence if e.identity_ok),
        })
    return cands, rows


def coverage_stats(cands: dict) -> dict:
    """论文 §4.2 的三个分开的口径（不要合并成一个分数）。

    attribute_coverage   = 产出非空值的请求属性占比
    publication_coverage = 拿到写库动作 WRITE 的请求属性占比
    review_rate          = 转人工的占比
    blocked_rate         = 被阻断的占比
    not_gated_rate       = 没产出值、因而不进核验与写库门的占比（论文 §4.3 的口径）
    """
    total = len(cands) or 1
    filled = sum(1 for c in cands.values() if c.value)
    written = sum(1 for c in cands.values() if c.action == WRITE)
    review = sum(1 for c in cands.values() if c.action == REVIEW)
    blocked = sum(1 for c in cands.values() if c.action == BLOCK)
    not_gated = sum(1 for c in cands.values() if c.action == "")
    return {
        "requested": total,
        "attribute_coverage": round(filled / total, 4),
        "publication_coverage": round(written / total, 4),
        "review_rate": round(review / total, 4),
        "blocked_rate": round(blocked / total, 4),
        "not_gated_rate": round(not_gated / total, 4),
    }


# ---------------------------------------------------------------------------
# 合成示例数据（母婴品类；非论文数据、非任何企业真实数据）
# ---------------------------------------------------------------------------
ATTRS = ["适用月龄", "材质", "认证", "容量"]


def make_demo_sku() -> dict:
    """一个「自有来源只答了一半、检索页里既有同款也有别款」的合成 SKU。"""
    return {
        "sku": "DEMO-BABY-001",
        "asin": "B0DEMO0001",
        "brand": "DemoBaby",
        "model": "DB-240",
        "catalog": {"attributes": {"适用月龄": "6 个月以上"}},
        "syndicated": [{"feed_id": "feed-A", "attributes": {"认证": "FDA"}}],
        "images": [{"image_id": "img-1", "claims": {"容量": "240 ml"}}],
    }


def make_demo_search_pages() -> list:
    return [
        {   # 同款页面：asin 一致 → 证据可用
            "url": "https://brand.example/db-240",
            "asin": "B0DEMO0001",
            "title": "DemoBaby DB-240 奶瓶",
            "attributes": {"材质": "PP", "认证": "FDA", "适用月龄": "6 个月以上"},
        },
        {   # 变体页面：asin 不同 → 整页丢弃，不能用来补这个 SKU
            "url": "https://shop.example/db-240-xl",
            "asin": "B0OTHER0999",
            "title": "DemoBaby DB-240 XL 奶瓶",
            "attributes": {"材质": "tritan", "认证": "LFGB",
                           "适用月龄": "0-6 个月", "消毒方式": "蒸汽消毒"},
        },
        {   # 目标 SKU 的另一页：用于「自有来源与检索页互相矛盾」的场景
            "url": "https://retailer.example/b0demo0002",
            "asin": "B0DEMO0002",
            "title": "DemoBaby DB-300 奶瓶",
            "attributes": {"适用月龄": "0-6 个月"},
        },
    ]


def make_demo_sku_with_conflict() -> dict:
    """卖家目录与第三方 feed 对同一属性给出不同值 → 必须转人工而不是直接发布。"""
    return {
        "sku": "DEMO-BABY-002",
        "asin": "B0DEMO0002",
        "brand": "DemoBaby",
        "model": "DB-300",
        "catalog": {"attributes": {"适用月龄": "0-6 个月"}},
        "syndicated": [{"feed_id": "feed-A", "attributes": {"适用月龄": "6-12 个月"}}],
        "images": [],
    }


def make_demo_sku_multisource() -> dict:
    """两个独立来源用不同写法给出同一个值 → 归一后一致，可作为高置信度通过。"""
    return {
        "sku": "DEMO-BABY-003",
        "asin": "B0DEMO0003",
        "brand": "DemoBaby",
        "model": "DB-400",
        "catalog": {"attributes": {"材质": "BPA-Free"}},
        "syndicated": [{"feed_id": "feed-B", "attributes": {"材质": "bpa free"}}],
        "images": [],
    }
```

```python
# ---------------------------------------------------------------------------
# 测试：断言论文的核心机制真的成立（L5 由 pytest 执行）
# ---------------------------------------------------------------------------
def test_normalize_merges_benign_variants():
    """论文 §3.2：把 "NiMH"/"nickel-metal hydride"、"60Hz"/"60 Hz" 归一到同一表示。"""
    assert normalize_value("NiMH") == normalize_value("nickel-metal hydride")
    assert normalize_value("60Hz") == normalize_value("60 Hz")
    assert normalize_value("  BPA-Free ") == normalize_value("bpa free")
    # 不同值不能被归一掉
    assert normalize_value("PP") != normalize_value("tritan")


def test_identity_mismatch_discards_web_evidence():
    """变体页面（asin 不同）必须整页丢弃，不能用来补这个 SKU。"""
    sku, pages = make_demo_sku(), make_demo_search_pages()
    assert identity_match(pages[0], sku) is True
    assert identity_match(pages[1], sku) is False
    # 自有来源为空 → 触发检索；同款页给出 "PP"，变体页给出的 "tritan" 不得进入证据
    mat = scout_agent(sku, ["材质"], pages)["材质"]
    assert normalize_value(mat.value) == "pp"
    assert all("XL" not in e.locator for e in mat.evidence)
    # 只有变体页才有的属性 → 身份匹配过滤后无证据可用 → 必须弃权，而不是采纳别款的值
    dis = scout_agent(sku, ["消毒方式"], pages)["消毒方式"]
    assert dis.status == ST_NOT_FOUND and dis.value is None
    assert dis.evidence == []


def test_conflicting_evidence_goes_to_human_review_not_publish():
    """核心断言：证据互相矛盾时，判官判 UNCERTAIN，写库门走人工复核而不是发布。"""
    sku = make_demo_sku_with_conflict()
    cands, rows = run_trace(sku, ["适用月龄"], make_demo_search_pages())
    c = cands["适用月龄"]
    assert c.status == ST_CONFLICT
    assert c.verdict == UNCERTAIN and c.subtype == "CONFLICT"
    assert c.action == REVIEW
    assert c.action != WRITE          # 绝不自动发布
    assert rows[0]["action"] == REVIEW
    # 冲突时必须保留双方证据，人工才能裁决
    assert len({e.normalized for e in c.evidence}) >= 2
    # 若把置信度阈值抬到冲突候选的自报置信度之上，按论文式 (3) 的顺序会先被 BLOCK ——
    # 「转人工」与「阻断」都**不是**发布，这正是把裁决与写库策略拆开的意义。
    c2, _ = run_trace(sku, ["适用月龄"], make_demo_search_pages(), theta=0.9)
    assert c2["适用月龄"].action == BLOCK
    assert c2["适用月龄"].action != WRITE


def test_judge_fails_and_blocks_unsupported_value():
    """没有任何证据支持提案值 → FAIL → BLOCK（论文把「没证据」与「证据打架」分开）。"""
    bad = Candidate(sku="S", attribute="认证", value="LFGB",
                    evidence=[Evidence(SRC_WEB, "FDA", "https://x.example/p")],
                    confidence=0.9, status=ST_EXTRACTED)
    verdict, subtype = judge_agent(bad)
    assert verdict == FAIL and subtype == "HALLUCINATION"
    assert write_policy(verdict, bad.confidence) == BLOCK
    # 完全没证据的候选同样判 FAIL，而不是 UNVERIFIED
    empty = Candidate(sku="S", attribute="认证", value="FDA", evidence=[], confidence=0.9)
    assert judge_agent(empty)[0] == FAIL


def test_image_only_evidence_is_unverified_and_theta_decides_publication():
    """只有图片证据 → UNVERIFIED（「无法直接确认」）；是否发布由阈值 θ 决定。"""
    sku = make_demo_sku()
    cands, _ = run_trace(sku, ["容量"], make_demo_search_pages())
    c = cands["容量"]
    assert c.status == ST_AMBIGUOUS
    assert c.verdict == UNVERIFIED and c.subtype == "VISUAL_ONLY"
    assert c.action == WRITE                      # 论文默认口径：UNVERIFIED 且置信度达标 → 写入
    # 企业侧把阈值调高（更保守）后，同一候选必须被拦下
    assert write_policy(UNVERIFIED, c.confidence, theta=0.9) == BLOCK
    # 只有图片来源的候选，其来源类型集合里不含任何文本来源
    assert all(e.source not in TEXT_SOURCES for e in c.evidence)


def test_low_confidence_blocks_even_when_verdict_is_pass():
    """阈值 θ 是一道独立的闸门：裁决 PASS 但置信度不足，同样不得写入。"""
    sku = {"sku": "S2", "asin": "B0S2", "brand": "B", "model": "M",
           "catalog": {"attributes": {"材质": "PP"}}, "syndicated": [], "images": []}
    cands, _ = run_trace(sku, ["材质"], [], theta=0.9)
    c = cands["材质"]
    assert c.verdict == PASS
    assert c.confidence == 0.80 < 0.9
    assert c.action == BLOCK
    assert write_policy(PASS, 0.95, theta=0.9) == WRITE


def test_multi_source_agreement_publishes_with_full_provenance():
    """多个独立来源用不同写法给出同一个值 → 归一后一致 → PASS → WRITE，且带得出证据。"""
    sku = make_demo_sku_multisource()
    cands, rows = run_trace(sku, ["材质"], make_demo_search_pages())
    mat = cands["材质"]
    assert mat.status == ST_EXTRACTED and mat.verdict == PASS and mat.action == WRITE
    assert set(mat.sources) >= {SRC_CATALOG, SRC_SYNDICATED}   # 两个独立来源一致
    assert mat.confidence == 0.95                              # 多来源一致 → 更高自报置信度
    # 自有来源已经能给出一致答案时，不再触发网页检索（论文 Algorithm 1 的 unresolved 条件）
    assert all(e.source != SRC_WEB for e in mat.evidence)
    for r in rows:
        if r["action"] == WRITE:
            assert r["evidence"] and "@" in r["evidence"]


def test_unpopulated_attributes_skip_the_verification_gate():
    """未产出值的属性不进入核验与写库门（论文 §4.3 明确口径）。"""
    sku = {"sku": "S3", "asin": "B0S3", "brand": "B", "model": "M",
           "catalog": {}, "syndicated": [], "images": []}
    cands, rows = run_trace(sku, ["适用月龄", "材质"], [])
    assert rows == []
    assert all(c.status == ST_NOT_FOUND and c.action == "" for c in cands.values())
    st = coverage_stats(cands)
    assert st["attribute_coverage"] == 0.0 and st["publication_coverage"] == 0.0


def test_coverage_stats_are_reported_separately():
    """口径分开报：抽取覆盖率 / 发布覆盖率 / 转人工率 / 阻断率 / 未进门率，不合成单一分数。"""
    sku = make_demo_sku_with_conflict()
    cands, _ = run_trace(sku, ATTRS, make_demo_search_pages())
    st = coverage_stats(cands)
    assert st["requested"] == len(ATTRS)
    assert st["attribute_coverage"] == 0.25       # 四个属性里只有一个产出了非空值
    assert st["publication_coverage"] == 0.0      # 唯一候选是冲突 → 转人工，不发布
    assert st["review_rate"] == 0.25
    assert st["blocked_rate"] == 0.0
    assert st["not_gated_rate"] == 0.75           # 未产出值的属性不进核验与写库门
    # 产出了值的属性 = 发布 + 转人工 + 阻断；与「未进门」相加恰好是全部请求属性
    assert abs(st["publication_coverage"] + st["review_rate"] + st["blocked_rate"]
               - st["attribute_coverage"]) < 1e-9
    assert abs(st["attribute_coverage"] + st["not_gated_rate"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 业务演示
# ---------------------------------------------------------------------------
def _business_demo() -> None:
    sku, pages = make_demo_sku(), make_demo_search_pages()
    cands, rows = run_trace(sku, ATTRS, pages)
    print("== 一次目录补全运行（合成示例数据，非论文数据）==")
    for r in rows:
        print(f"  {r['attribute']:6s} -> {str(r['value']):14s} [{r['verdict']:<10s}]"
              f" q={r['confidence']:.2f}  {r['action']}")
        print(f"          证据: {r['evidence'] or '(无)'}")
    st = coverage_stats(cands)
    print("\n== 三个口径分开报（不要合成一个分数）==")
    print(f"  抽取覆盖率={st['attribute_coverage']:.2f}  "
          f"发布覆盖率={st['publication_coverage']:.2f}  "
          f"转人工率={st['review_rate']:.2f}  阻断率={st['blocked_rate']:.2f}")

    conflict = make_demo_sku_with_conflict()
    c2, rows2 = run_trace(conflict, ["适用月龄"], pages)
    print("\n== 证据矛盾时的处置（本卡的核心断言）==")
    for r in rows2:
        print(f"  {r['attribute']} -> {r['value']} [{r['verdict']}/{r['subtype']}] "
              f"→ {r['action']}（不自动发布）")


if __name__ == "__main__":
    _business_demo()
```

---

## ④ 技能关联

- **前置 / 上游｜`Skill-KG-Auto-Construction-Agent-Driven.md`**（08-知识图谱）：那张卡解决「怎么用
  LLM Agent 从商品描述里抽出属性、产出本体与图谱」；本卡解决它的下一跳——**抽出来的值凭什么写进目录**。
  数据流：本卡的**类目属性模板**可以直接复用该卡构建出的属性本体，而该卡的抽取结果可以作为本卡
  `catalog` 这一层证据进入取证环节。缺了本卡的裁决与写库门，图谱构建会被无证据的幻觉值污染。
- **同源方法论 / 对照｜`Skill-Multi-Agent-Debate.md`**（10-MAS）：那张卡讲的是多个 agent 独立推理后
  对抗式复核以打破单模型的思维退化；本卡把「生成者与核验者分离」用在生产写库上，但**判官只对照候选
  自己的证据**做核验，不做多方辩论。二者的差异本身有业务含义：辩论适合开放式推理题，写库门需要的是
  **可复现的证据标准**，而不是多数票。
- **配套度量｜`Skill-Agent-Stage-Evaluation.md`**（16-智能体工程）：那条卡的分阶段评测口径正好补上本卡
  最容易被做错的地方——把「抽得多」当成「发布得对」。本卡 ③ 的 `coverage_stats` 按论文口径把
  抽取覆盖率、发布覆盖率、转人工率**分开**输出，就是为了接到分阶段评测上。
- **下游｜`Skill-Dense-Retrieval-Ecommerce-Semantic-Search.md`**（08-知识图谱）：属性补全的直接受益方。
  数据流：本卡写进目录的结构化属性 → 变成该卡里生成模型可抽取的**结构化约束**（价格区间、评分要求
  同类的筛选项），检索与筛选的召回面因此变宽。没有结构化属性，语义检索只能靠标题与描述硬扛。
- **组合｜`Skill-Explainable-Recommendation.md`**（05-推荐系统）：补全后的属性会进入推荐模型的
  商品表示；若推荐侧需要解释「为什么推这个」，属性级 provenance 可以直接当解释素材（是哪个来源、
  哪条证据支持了这个属性）。这条组合是把本卡的可追溯性变成前台可见价值的方式。
- **同域互补｜`Skill-Live-Catalog-Conversational-Rec.md`**（同目录，00-电商Agent）：那张卡解决
  「目录条目频繁变动时，会话推荐怎么只处理增量」；本卡解决「目录里缺的属性怎么带着证据补上」。
  数据流是互补的：会话推荐依赖目录里的结构化属性做筛选与召回，而本卡的每一次写库动作都会改变
  目录版本，正好是增量刷新机制要消费的上游事件 —— 反过来说，若上游没有稳定性约束，
  本卡的批量写入会把下游的增量对账打爆，两张卡要一起设计。

---

## ⑤ 商业价值评估

**ROI 公式**（本次不代入任何论文未给、企业未测的数字）：

ROI =（Δ毛利 − C_sys）/ C_sys，其中 Δ毛利 = 月 GMV × ΔCVR × m + Δ缺错件率 × 月订单数 × c_fix。

| 参数 | 含义 | 来源 |
|---|---|---|
| `ΔCVR` | 展示补全属性后结账转化率的**相对**变化 | **必须企业自测**。论文线上随机实验（五周、90% 流量进处理组，⑥ Q36）报告的结账转化相对变化是 +0.48%，95% 置信区间 [+0.04%, +0.92%]，p=0.034（⑥ Q37 / Q45）；重度用户子群的相对变化是 +1.18%，95% 置信区间 [+0.24%, +2.12%]，p=0.014（⑥ Q38）。这是**整条链路 + PDP 展示**的端到端结果（⑥ Q43），只能当**上界参考**，不能当预测 |
| `m` | 增量 GMV 的贡献毛利率（扣平台佣金、头程与履约、退货、支付等变动成本） | 企业自有财务口径 |
| `Δ缺错件率` | 缺件 / 错件率的相对改善 | **必须企业自测**。同一次实验报告该指标的相对变化为 −1.08%，95% 置信区间 [−2.04%, −0.13%]，p=0.026（⑥ Q39），同样属于端到端结果，方向可参考、幅度不可直接搬 |
| `c_fix` | 单次缺件 / 错件的处理成本（补发、退运费、客服工时、店铺评分损失） | 企业自有财务口径 |
| `C_sys` | 一次性建设 + 运行成本：证据源接入与抓取维护、属性模板维护、判官校准的人工标注、复核队列人力、每次运行的模型调用 | **论文未报告任何成本或延迟量级**；论文只说明每个合格 SKU 会产生**一次侦察调用 + 一次裁决调用**（⑥ Q44），调用成本的量级必须企业自估 |

**口径提示（本卡主动标注，不要跳过）**：流传最广的头条数字「曝光加权补全覆盖率提升 **90.4%**」
**只出现在摘要**（⑥ Q4）；正文对同一指标写的是「**over 90%**」，并且**没有给分垂类的精确表**（⑥ Q35）。
本卡引用时一律写成「摘要报 90.4%，正文表述为 over 90%」，**不把它包装成正文表格里的精确值**。
同样地，生产规模数字是正文自己给的「**31 million SKUs**」（⑥ Q34），可以按正文引用。

**保守结论**：把 ⑥ Q37 的结账转化区间上界当作 `ΔCVR` 的**上界**代入即可得到收益的天花板；
由于母婴跨境要在平台佣金、头程与退货上再打折（`m` 明显低于论文作者所在平台的自有流量场），
且论文的线上实验只有五周、90% 流量进处理组（⑥ Q36），**本卡不给收益金额结论**。
成本侧另有一个容易被忽略的项：**人工复核队列**。论文未报告 UNCERTAIN 的占比，
本卡因此不对复核工作量做任何估计 —— 上线前必须先跑两周影子模式量出来。

- **实施难度**：⭐⭐⭐⭐☆—— 证据源接入与抓取维护、判官校准集标注、复核工作流是三块主要成本；
  但 ③ 的流水线骨架（取证 → 裁决 → 写库门 → provenance）可以先用规则替身跑通影子模式，
  把难度拆成「先跑通结构」与「再换成 LLM」两段。
- **优先级**：⭐⭐⭐⭐☆—— 对母婴出海是**高优先级**：属性缺失同时打中三处痛点（筛选召回、退货与差评、
  合规与安全警示）。若企业当前连 SKU 主数据与类目属性模板都不齐，优先级降到 3 星，先补模板与主数据。
- **评估依据**：收益侧的不确定性大于工程侧——论文能给的两个数都是**别人的类目、整条链路**的端到端结果；
  成本侧论文完全没给。所以本卡的价值主张是**决策链路的修正**：把「模型抽出来就写库」改成
  「有证据才能写库、证据打架就转人工」，让每一格属性都能审计。这一点不依赖论文的任何一个百分比成立。

---

## ⑥ 原文引用

**A. 摘要（头条数字；A3 是摘要与正文口径不一致的那一条）**

> 原文："On an offline human evaluation dataset, TRACE’s proposed attribute values were 98.2% accurate at 74.7% attribute coverage."
> 出处：2608.20844 §Abstract｜Q3

> 原文："Deployed in production on an industry-scale catalog, TRACE increased impression-weighted enrichment coverage across four business verticals by 90.4%."
> 出处：2608.20844 §Abstract｜Q4

> 原文："An online experiment subsequently showed that surfacing the enriched attributes on the product detail page increased checkout conversion by 0.48%."
> 出处：2608.20844 §Abstract｜Q5

**B. 问题与定位**

> 原文："Product catalogs underpin search, discovery, and recommendation in e-commerce, yet they are often attribute-sparse: the attributes shoppers and downstream systems rely on are either buried in unstructured content such as titles and images or missing from the catalog altogether."
> 出处：2608.20844 §1 Introduction｜Q1

> 原文："Manually enriching e-commerce catalogs is impractical given their scale and rapid growth."
> 出处：2608.20844 §Abstract｜Q2

> 原文："Some attribute values cannot be reliably inferred from owned data sources and must instead be sourced externally and verified against the exact product."
> 出处：2608.20844 §1 Introduction｜Q6

> 原文："Accuracy estimated from a point-in-time catalog audit may become less representative as the catalog’s product mix evolves."
> 出处：2608.20844 §1 Introduction｜Q7

> 原文："Missing or inaccurate attribute values can mislead shoppers and degrade fulfillment quality; for safety-sensitive attributes such as allergens or dietary restrictions, they can have especially serious consequences."
> 出处：2608.20844 §1 Introduction｜Q8

> 原文："We incorporate identity-matched search grounding to recover attribute values that cannot be reliably inferred from owned data sources."
> 出处：2608.20844 §1 Introduction｜Q9

> 原文："We place a JudgeAgent in the serving path as a verify-before-write gate, applying a consistent evidence standard to each proposed value and making publication quality less sensitive to shifts in the catalog’s product mix."
> 出处：2608.20844 §1 Introduction｜Q10

**C. 方法：ScoutAgent / JudgeAgent / 写库门**

> 原文："TRACE implements this process as a two-stage verify-before-write architecture. The ScoutAgent gathers evidence from multiple sources, verifies that externally retrieved evidence refers to the target product, and proposes grounded attribute values. The JudgeAgent then re-examines each candidate under a stricter verification policy and determines whether it is eligible for publication, should be blocked, or requires human review. Figure 1 shows the end-to-end workflow, and Algorithm 1 formalizes the procedure."
> 出处：2608.20844 §3.1 Overview｜Q12

> 原文："where $v_{a}$ is the proposed value, $E_{a}$ is its supporting evidence, $\tau_{a}$ records the evidence-source types, $q_{a}$ is a model-reported confidence score, and $z_{a}$ records the extraction status as one of extracted, not_found, not_applicable, ambiguous, or conflict."
> 出处：2608.20844 §3.1 Overview｜Q11

> 原文："The ScoutAgent gathers and reconciles evidence for each target attribute in two stages. It first considers readily available, product-linked information, including textual fields and product images from seller-provided catalog data and syndicated product data. When this information is insufficient to determine a reliable value for the target attribute, the ScoutAgent uses web search to gather additional evidence."
> 出处：2608.20844 §3.2 The ScoutAgent｜Q13

> 原文："This separation allows the ScoutAgent to prioritize textual evidence while still using images for attributes expressed visually, such as material, certification marks, and on-package claims."
> 出处：2608.20844 §3.2 The ScoutAgent｜Q14

> 原文："Web search may return pages that appear relevant to the query but refer to a different product or product variant."
> 出处：2608.20844 §3.2 Identity-grounded web retrieval｜Q15

> 原文："Evidence from $p$ is used only when the ScoutAgent determines that the page describes the target product; otherwise, the page is discarded."
> 出处：2608.20844 §3.2 Identity-grounded web retrieval｜Q16

> 原文："It maps benign variations to a common representation, such as “NiMH” and “nickel-metal hydride,” or “60 Hz” and “60Hz.”"
> 出处：2608.20844 §3.2 Evidence reconciliation and abstention｜Q17

> 原文："When the available evidence is insufficient, ambiguous, or conflicting, the ScoutAgent abstains rather than inferring a value from background knowledge."
> 出处：2608.20844 §3.2 Evidence reconciliation and abstention｜Q18

> 原文："It applies a stricter evidence policy than ScoutAgent, focusing on whether available evidence supports the proposed value for the target product."
> 出处：2608.20844 §3.3 The JudgeAgent｜Q19

> 原文："The distinction between UNVERIFIED and UNCERTAIN separates a lack of confirming evidence from active disagreement among the available evidence."
> 出处：2608.20844 §3.3 Verdict taxonomy｜Q20

> 原文："Candidates below the model-reported confidence threshold $\theta$ are blocked. Among the remaining candidates, those receiving PASS or UNVERIFIED are written, those receiving FAIL are blocked, and those receiving UNCERTAIN are routed to human review together with their evidence trail."
> 出处：2608.20844 §3.3 From verdict to catalog action（式 (3)）｜Q21

> 原文："Unpopulated attributes are excluded because only proposed values enter the production verification and write gate."
> 出处：2608.20844 §4.3 Evaluation Results｜Q27

> 原文："TRACE makes one ScoutAgent call and one JudgeAgent call per eligible SKU; each call returns a map of per-attribute outputs."
> 出处：2608.20844 §Appendix A Condensed Agent Prompt Templates｜Q44

**D. 离线评测**

> 原文："We evaluate TRACE on products from four business verticals: Grocery, Alcohol, Electronics, and Home Improvement. The number of distinct target attributes ranges from 11 in Grocery to 409 in Home Improvement."
> 出处：2608.20844 §4.1 Data Collection｜Q22

> 原文："This dataset contains 500 SKUs and 2,497 target SKU–attribute pairs. Human annotators established the reference attribute values, and a separate group of auditors reviewed the values proposed by the ScoutAgent."
> 出处：2608.20844 §4.1 Data Collection｜Q23

> 原文："This dataset contains 955 SKUs and 4,990 target SKU–attribute pairs. Because exhaustive human labeling was not available for these verticals, we use the JudgeAgent to adjudicate the complete dataset. The JudgeAgent results provide a scalable operational quality signal."
> 出处：2608.20844 §4.1 Data Collection｜Q24

> 原文："In particular, we refer to the fraction receiving PASS or UNVERIFIED as the judge-supported rate. This metric measures compliance with the JudgeAgent’s evidence policy and is not interpreted as human-validated accuracy."
> 出处：2608.20844 §4.2 Evaluation Metrics｜Q25

> 原文："Judge-supported rate is the fraction of all extracted values receiving a PASS or UNVERIFIED verdict; UNCERTAIN and invalid responses remain in the denominator. Publication coverage is the fraction of requested attributes receiving one of these two verdicts. Costs are normalized to Gemini 2.5 Flash."
> 出处：2608.20844 §4.4 Table 1 表注｜Q30

> 原文："On the fully human-labeled Grocery and Alcohol dataset, the ScoutAgent achieved 98.2% extraction accuracy at 74.7% attribute coverage."
> 出处：2608.20844 §4.3 Evaluation Results｜Q26

> 原文："Among values assigned PASS, 98.4% were confirmed correct by human reviewers. Of the disagreements between the JudgeAgent and human reviewers, 87.8% were false rejections — values assigned FAIL but judged correct by humans — whereas 12.2% were false acceptances."
> 出处：2608.20844 §4.3 Evaluation Results｜Q28

> 原文："On the Electronics and Home Improvement dataset, the ScoutAgent achieved 87.8% attribute coverage. Of the extracted values, 97.4% received a PASS or UNVERIFIED verdict from the JudgeAgent."
> 出处：2608.20844 §4.3 Evaluation Results｜Q29

> 原文："Although Gemini 3.5 Flash increases extraction coverage by $0.5$ percentage points, its lower judge-supported rate reduces publication coverage from 85.5% to 81.9%, while increasing inference cost by more than $7\times$."
> 出处：2608.20844 §4.4 LLM Backbone Comparison｜Q31

> 原文："GPT-5.4 and Claude Sonnet 5 achieve still lower publication coverage, at 73.2% and 63.5%, respectively."
> 出处：2608.20844 §4.4 LLM Backbone Comparison｜Q32

> 原文："Error analysis shows that the lower judge-supported rates of the alternative backbones arise primarily from unsupported or partial extractions rather than explicit contradictions or hallucinations. These errors are concentrated in evidence-intensive attributes, including unit count and free-text descriptions."
> 出处：2608.20844 §4.4 LLM Backbone Comparison｜Q33

**E. 生产部署与线上实验**

> 原文："We deployed TRACE in production and enriched 31 million SKUs across four business verticals."
> 出处：2608.20844 §5 Deployment｜Q34

> 原文："This increased impression-weighted enrichment coverage, defined as the share of customer impressions associated with product records carrying enriched attributes, by over 90% across these verticals."
> 出处：2608.20844 §5 Deployment｜Q35

> 原文："We evaluate this hypothesis through a five-week randomized A/B test with 90% of traffic assigned to treatment and 10% to holdout."
> 出处：2608.20844 §5.1 User Impact｜Q36

> 原文："Effects are reported as relative changes versus the control group, with 95% confidence intervals and $p$-values."
> 出处：2608.20844 §5.1 Table 2 表注｜Q45

> 原文："| Checkout conversion | +0.48% | [+0.04%, +0.92%] | 0.034 |"
> 出处：2608.20844 §5.1 Table 2（Checkout conversion）｜Q37

> 原文："| Checkout conversion (power users) | +1.18% | [+0.24%, +2.12%] | 0.014 |"
> 出处：2608.20844 §5.1 Table 2（Checkout conversion, power users）｜Q38

> 原文："| Missing/incorrect item rate | −1.08% | [−2.04%, −0.13%] | 0.026 |"
> 出处：2608.20844 §5.1 Table 2（Missing/incorrect item rate）｜Q39

> 原文："As shown in Table 2, the enriched PDP increased checkout conversion by $0.48\%$, with a larger $1.18\%$ increase among power users. It also reduced the missing/incorrect-item rate by $1.08\%$."
> 出处：2608.20844 §5.1 User Impact｜Q40

**F. 论文自承局限**

> 原文："The Electronics and Home Improvement results use judge-supported rate as a scalable operational metric rather than as a substitute for human-validated precision. The JudgeAgent was calibrated on the Grocery and Alcohol human audit, while its transfer to other categories has received more limited human evaluation."
> 出处：2608.20844 §6 Limitations｜Q41

> 原文："Moreover, because the ScoutAgent and JudgeAgent use models from the same family in the primary configuration, they may exhibit correlated failure modes."
> 出处：2608.20844 §6 Limitations｜Q42

> 原文："The online experiment evaluates the end-to-end effect of displaying enriched product pages. It therefore demonstrates the value of the deployed system as a whole, but does not isolate the contribution of the JudgeAgent or write-gating policy."
> 出处：2608.20844 §6 Limitations｜Q43
