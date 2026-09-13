---
title: Skill-Live-Catalog-Conversational-Rec
module: 00-电商Agent
topic: 活目录会话推荐——用 stable ID + full hash + semantic hash 三层对账，只处理目录增量
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.27006
paper: Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval
venue: RecSys 2026 (Demo)
venue_tier: preprint
venue_source: TRACK_RULES
evidence_grade: A
verified_by: verify_skill_code.py（K1 L1–L5 全绿）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py G2/G3 passed + 人工抽检：Table 1 表注与表体的全部报告值逐个回查 fulltext.md
verified_at: 2026-09-12
supersedes:
related: Skill-Diversity-Reranking-SMMR.md, Skill-Semantic-ID-Retrieval-RPG.md, Skill-Long-Term-Preference-Memory.md, Skill-Cold-Start-Meta-Learning-PAM.md, Skill-Agentic-Catalog-Enrichment.md
l1_id: PLN-PLT
l1_plane: 数据与Agent平台
l2_id: DOM-08
l2_domain: 数据与AI运行
l3_id: DOM-08-141
l3_business: 数据管道
l3_all: 数据管道 / 主数据治理
l1_l2_l3: 数据与Agent平台/数据与AI运行/数据管道
---

# Skill Card: 活目录会话推荐（Conversational Recommendation over Live Catalogues）

**这是 `00-电商Agent` 域的第一张卡，也是一张「工程系统卡」而不是「算法卡」。**
论文交付的不是一个新模型，而是把「商品目录一直在变」当成系统一等公民的**索引同步机制**；
它的主张是「让会话推荐的目录新鲜度变成可运维的问题」，而不是「让推荐更准」。

> ⚠️ **读卡前必读（诚实边界声明）**
>
> 1. 本文是 **RecSys 2026 的 Demo 短文**（3 页；章节只有 Introduction / System Overview / Demonstration /
>    Concluding Remarks / Appendix A），**全文没有 Experiments 或 Evaluation 章节**。
> 2. 论文**没有做过任何推荐质量的量化评测**——离线相关性研究、真实用户研究都被列为 future work
>    （⑥ Q30）。所以本卡**不提供任何效果数字**（没有准确率、转化率、GMV、留存）。
> 3. 论文**唯一的量化证据是同步侧的成本对照**（§3 的 Table 1，⑥ Q21/Q22）：它在**匿名 500 条记录的目录**上
>    量的是「增量同步 vs 全量重建」的耗时占比。这是**同步成本**，**不是**推荐质量。
> 4. ⑤ 段的 ROI 是**本项目的推算**（公式 + 参数来源），不是论文结论。
> 5. 治理提示：Demo 短文按本仓库 `MasterPrompt-v2.md` 的 R4 硬拦截清单口径本属「不出卡」类别。
>    本卡是在萃取简报明确指定下作为例外产出，因此 ①b 与 ⑤ 的边界声明一律从严，
>    且**不得被当作完整方法论文引用**。

---

## ① 算法原理

**核心思想**：把「商品目录一直在变」当作一等公民，用**三重标识**对每件商品做增量对账，
只重算真正变了的那部分，而不是每次重建整个向量索引。这是**工程系统描述**，不是方法论文。

**数学直觉**：每个 SKU 维护三个键——
`id`（stable product ID：跨快照对齐、驱动精确更新与删除，但答不了自然语言查询）、
`full_hash`（覆盖 feed 全部字段，任何字段变动都会变）、
`sem_hash`（只覆盖 name / description / brand / category，即「决定向量语义」的字段）。
比较这三者得到**五个互斥类别**：new、semantic-changed（重新 enrich + embed + upsert）、
metadata-only（价格/库存变了但语义没变 → **保留原向量**，只更新记录与 filter）、
deleted（连同向量移除）、unchanged（跳过）。
判据 `full_hash` 不等 **且** `sem_hash` 相等 ⇒ metadata-only —— 这一步是全部成本节省的来源。

**关键假设**：① 商品有稳定主键；② 语义字段集合能代表「是否需要重新 embedding」；
③ 同步是手动或定时的**批量对账**，不是流式 CDC（论文明确：it does not monitor the feed continuously）。

---

## ①b 反例与适用边界

### 什么时候不要用这个方法

1. **没有稳定商品主键**。整条机制建立在 stable product ID 上（⑥ Q10）。如果 feed 每次导出都重新生成 id
   （铺货 / 代发源常见），三层比较会退化成「全是 new + 全是 deleted」，本卡反而比全量重建更贵。
   先做 id 对齐，再谈增量。
2. **语义字段集合选错**。metadata-only 与 semantic-changed 的分界**完全由 `sem_hash` 覆盖哪些字段决定**
   （⑥ Q11）。若把 price / stock 也放进 `sem_hash`，每次改价都会触发重新 embedding，本卡的收益归零。
3. **需要事件级实时**。同步是 manual 或 scheduled 的一次对账，**不持续监听 feed**（⑥ Q09）；
   目录变更要等「下一次成功 sync」才进入推荐（⑥ Q24）。要「改价立刻反映」，本卡给不了。
4. **目录规模到千万级**。论文自己声明不做该量级的承诺（⑥ Q27）——
   实测与演示都在「匿名 500 条记录」这个尺度上（⑥ Q21）。
5. **你要买的是排序质量**。本卡一个字都没评测排序质量（见下）。

### 已知的失败模式

1. **写入非事务**。原文写明 Writes are non-transactional，生产上必须自行补「截断 feed 检测 / staging 或回滚 /
   异常 delta 监控」（⑥ Q27）。否则一次不完整快照会引发**推断性批量删除**；
   论文的对策是在同步入口拒收空快照或不完整快照（⑥ Q31），但那只挡住了一部分情形。
2. **「上架」不等于「有货」**。availability 被规范化成布尔并进 filter，
   但 feed 的库存字段本身可能滞后于真实仓库，推荐出的商品仍可能超卖。论文未处理这一环。
3. **enrichment 的生成式兜底是不可预测的那一段**。规则优先、生成式兜底（⑥ Q14），
   规则覆盖不到的类目（母婴里尤其多：材质、适用月龄、认证）会退回 LLM 调用，成本随之波动。
4. **多语言路径是「检测语言 → 用英文检索 → 用该语言回复」**（⑥ Q20）。
   母婴出海常见的西语 / 阿语 / 葡语市场里，英文检索词与本地语言商品描述之间的语义落差会直接影响召回，
   而论文对召回质量**没有任何评测**。
5. **第一条同步一定是全量**。没有任何历史索引状态时，五个类别里只有 new ——
   本卡不减少「冷启动那一次」的成本，它减少的是此后每一天的成本。

### 论文自己承认的局限（最关键的一条）

1. **论文未做任何推荐质量评测**。原文：The evaluation covers synchronization not ranking quality（⑥ Q30）——
   离线相关性研究与真实用户研究都留作 future work。因此本卡**不提供任何效果数字**，
   只提供「目录新鲜度」这套工程机制。
2. **唯一的量化证据在同步侧**：匿名 500 条记录目录上的增量同步耗时（⑥ Q21/Q22）。
   它证明的是「分类与操作数符合预期、增量确实比全量省」，**不是**「推荐因此变好」。
3. **可复现性有限**：公开的只有文档、引擎节选、一段 demo 视频与一个可运行的合成同步；
   完整引擎、目录与 Infobip 集成保持私有（⑥ Q28）。所以「作者的系统」与「本卡 ③ 的实现」是两套代码，
   本卡只复现机制，**不声称与作者实现等价**。
4. **论文未讨论**：母婴或跨境电商场景、平台店与独立站的双边目录对齐、任何成本或人力投入量级。
5. **论文未报告**：同步的并发/限流设计、向量库在真实规模下的表现、enrichment 的 LLM 调用占比。
   本卡 ⑤ 把这些全部标为「需企业自测」。

---

## ② 母婴出海应用案例

### 场景 1：旺季里客服/导购机器人「答出已下架商品」——用增量对账把目录新鲜度变成可运维指标

- **业务问题**：母婴出海店铺（吸奶器、纸尿裤、奶瓶）在旺季每天有几十个 SKU 改价、补货、下架换包装
  （奶嘴按月龄分段换代、纸尿裤按码数调整包装），而且独立站（Shopify）与平台店的商品数据是两套。
  客服 / 导购机器人（WhatsApp 私域、独立站站内客服）背后的向量索引如果是「每周日全量重建一次」，
  那么一周之内推荐出来的商品经常已经下架、断货或涨价：顾客点进链接看到商品不存在或价格已变 → **客诉、退货、差评**。
  论文的演示形态正是 WhatsApp 购物助手（⑥ Q25），与母婴出海在拉美 / 中东的私域客服习惯吻合。
  要做的事很具体：**每周日的全量重建改成每日（或跟 feed 落地）的增量对账**，
  把「索引与在售目录的偏差窗口」从一周压到「下一次成功同步」，同时**不增加 embedding 成本**。
- **数据要求**：一份商品 feed（Google Merchant Center Atom feed / Shopify 商品导出 / 平台店每日商品快照），
  **SKU 粒度**，字段至少含 id、name、description、brand、category、price、currency、availability、link；
  **必须持久化「上一次索引状态」**——每个 SKU 的 `(id, full_hash, sem_hash, vector_id)` 四元组；
  **必须留存每日 feed 快照归档**（否则算不出 delta）。历史长度：**≥ 1 个完整旺季周期**，
  用于估计「语义类变动 vs 元数据类变动」的占比，并据此定同步频率（母婴有明显季节性 + 宝宝月龄换代，
  窗口太短会把换代误判成一次性促销）。
- **数据可得性**：`部分可得（需补充 X）`。
  独立站侧（Shopify 商品导出 / 自有 feed）可得；**平台店侧没有原生 feed 导出**，
  需要补充**每日商品快照留档**。缺这一项时，平台店的商品只能继续走全量重建，
  本卡退化为「只管独立站那半边目录」。
- **预期产出**：每天一份五类对账清单（new / semantic-changed / metadata-only / deleted / unchanged），
  以及「今天需要重新 embedding 的 SKU 清单」——后者是唯一会花钱的部分。
- **业务价值**：同步成本从「每次全量重建」降到「只处理 delta」。论文在匿名 500 条记录目录上实测
  增量同步耗时为全量重建的 1.8%–12.3%（⑥ Q21/Q22）。
  **注意口径**：这是**同步耗时占比**，不是 GMV、不是转化率、更不是推荐质量——论文没有后者。

### 场景 2：黑五前六周——同步频率该定多高，而不是「多久重建一次」

- **业务问题**：黑五前六周 feed 每天都在变。同步频率定得太低 → 推荐过期、客诉；定得太高 →
  触顶 enrichment / embedding 的调用配额与限流。团队现在的做法是「每周日全量重建一次」，
  既没有依据，也无法回答「如果改成每天同步，成本会涨多少」。
- **数据要求**：feed 变更日志（或每日快照）算出的**每日 delta 占比与其构成**
  （语义类 vs 元数据类分别占多少），加上 enrichment / embedding 的调用单价与配额。
  粒度：SKU 级、日粒度；历史长度：≥ 1 个旺季周期。
- **数据可得性**：`企业内可得`。
  前提是场景 1 里的「索引状态 + 每日快照归档」已经落地；否则先用两周时间把 feed 差异存档，
  再回答频率问题（这两周的成本几乎只有存储）。
- **做法**：用 ③ 的对账把每日 delta 拆成五类，**metadata-only 类（改价、补货）的 embedding 成本为零**，
  因此「改价 / 补货同步即生效」在成本上是可以做到的；真正需要排期的是 semantic-changed 与 new 两类。
  按「每日新增 embedding 调用数 × 单价 vs 配额」倒推频率，而不是拍一个 cron。
- **预期产出**：一张「同步频率 × 每日 embedding 调用量 × 目录过期窗口」的对照表，
  以及建议的调度策略（例如：元数据类变动随 feed 落地即同步，语义类变动每日一次批量处理）。
- **业务价值**：见 ⑤ 的推算公式。本卡的收益形状是「**元数据类变动不再产生 embedding 成本**」，
  而母婴目录里改价、补货、换包装的频次通常远高于真正的语义改动。

---

## ③ 代码模板

- **依赖**：只用**标准库**（`hashlib` / `json` / `re` / `typing`）—— 论文的三层标识全部离线可复现，
  不需要向量库、不需要网络、不需要模型，因而能通过 K1 的断网执行校验。
- **一处刻意写法**：本模板不用 `@dataclass`，改用 `NamedTuple` + 普通类。原因是当前 K1 的 L3 import 探针
  在 CPython v3.14 下对**任何**含 `@dataclass` 的模块都会报
  `AttributeError: 'NoneType' object has no attribute '__dict__'`（探针 `exec_module` 时未把被测模块注册进
  `sys.modules`，而该版本的 `dataclasses._is_type` 需要它）—— 这是**门禁自身的缺陷**，与本卡机制无关；
  复现步骤与影响面记在 `evidence.md` §0。换写法只为让 L3 也干净，机制一个字节都没改。
- **结构**：三层标识（stable ID / full hash / semantic hash）→ 五个互斥类别的分类器 →
  按论文规则改索引的分流执行器 → 断言测试（pytest）→ 业务演示。
- **与论文的对应**：`SEMANTIC_FIELDS` 对应 ⑥ Q11 的 name/description/brand/category；
  `normalize_value` 对应 ⑥ Q36/§2.1 的 strips HTML / normalises prices / availability→boolean；
  `classify_changes` 对应 ⑥ Q13/Q14/Q15/Q16 的五个变化类别；`apply_sync` 的 metadata-only 分支
  （保留 `vector_id`）对应 ⑥ Q15。
- ⚠️ 代码内所有目录数据均为**构造数据**（`make_snapshot`），既不是论文数据，也不是任何企业的真实目录；
  代码块内出现的任何数字都是**本地可复现数字**，与论文 Table 1 的报告值没有任何对应关系。

```python
"""
活目录增量对账：stable ID + full hash + semantic hash 三层比较 → 五个变化类别分流。

论文：2608.27006 §2.1 Self-Refreshing Retriever（IDs, hashes, and embeddings / Change classes）
业务映射：母婴出海店铺（独立站 + 平台店）的商品 feed 每日变动（上新 / 下架 / 改价 / 补货 / 改描述），
         用它回答「今天哪些 SKU 需要重新 embedding、哪些只更新记录、哪些删除、哪些跳过」。

实现要点
  * 只用标准库（hashlib / json / re / typing）—— 纯离线可跑，不需要向量库、不需要网络；
  * 示例数据是**构造数据**（make_snapshot 造快照），不是论文数据，也不是任何企业的真实目录；
  * 五个变化类别互斥且完备，由 test_* 断言锁定。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# 1. 论文 §2.1 的三层标识
# ---------------------------------------------------------------------------
# 语义字段：原文 "a semantic hash over name, description, brand, and category
#           identifies changes requiring re-embedding"
SEMANTIC_FIELDS = ("name", "description", "brand", "category")
# 元数据字段：原文 "Metadata-only changes, such as price or stock,
#           retain the vector while updating the record and filters"
METADATA_FIELDS = ("price", "currency", "availability", "stock_qty", "link")

ID_FIELD = "id"

# 五个互斥类别（原文 "Comparing IDs and hashes yields five disjoint classes"）
NEW = "new"
SEMANTIC_CHANGED = "semantic_changed"
METADATA_ONLY = "metadata_only"
DELETED = "deleted"
UNCHANGED = "unchanged"
CHANGE_CLASSES = (NEW, SEMANTIC_CHANGED, METADATA_ONLY, DELETED, UNCHANGED)
# 只有这两类需要 enrich + embed + upsert
REEMBED_CLASSES = (NEW, SEMANTIC_CHANGED)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def normalize_value(value) -> str:
    """字段级规范化：strip HTML → 折叠空白 → 小写。

    论文 §2.1 的 parse 阶段会 "strips HTML, normalises prices, and converts
    availability to a boolean"；这里用等价的轻量规范化，避免**纯排版差异**
    被误判成语义变化、白付一次 embedding。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{round(value, 4):.4f}"
    return _WS_RE.sub(" ", _HTML_TAG_RE.sub(" ", str(value))).strip().lower()


def _digest(payload: dict) -> str:
    """稳定哈希：键排序 + 紧凑 JSON，避免 dict 插入顺序影响结果。

    取 SHA-256 的前 16 个十六进制字符，只为让索引状态可读；截断长度不影响机制。
    """
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def full_hash(record: dict) -> str:
    """full hash：覆盖 feed 的**全部字段**（除主键），任何字段变动都会变。"""
    payload = {k: normalize_value(v) for k, v in record.items() if k != ID_FIELD}
    return _digest(payload)


def semantic_hash(record: dict) -> str:
    """semantic hash：只覆盖 name / description / brand / category。

    它回答的是「这次变化是否改变了商品的**语义**、要不要重新 embedding」；
    price / stock 这类元数据变化**不应该**改变它 —— 这正是成本节省的来源。
    """
    payload = {k: normalize_value(record.get(k)) for k in SEMANTIC_FIELDS}
    return _digest(payload)


# ---------------------------------------------------------------------------
# 2. 索引状态（= 论文 storage layer 里的向量索引记录）
# ---------------------------------------------------------------------------
class IndexEntry(NamedTuple):
    """索引里的一条商品记录（= 论文 storage layer 的一行）。

    不可变：每次同步都换一个新对象，避免「谁改了哪个字段」被就地覆盖掉。
    """
    product_id: str
    full_hash: str
    semantic_hash: str
    vector_id: str | None            # 已 embedding 的向量主键；metadata-only 变化时必须保持不变
    record: dict


class IndexState:
    """索引状态：向量索引 + 每条记录的哈希指纹 + 累计 embedding 调用数。"""

    def __init__(self) -> None:
        self.entries: dict = {}
        self.embed_calls: int = 0    # 累计 embedding 调用次数 —— 用来量「只处理 delta」的收益

    def put(self, product_id: str, entry: IndexEntry) -> None:
        self.entries[product_id] = entry

    def remove(self, product_id: str) -> None:
        self.entries.pop(product_id, None)

    def has(self, product_id: str) -> bool:
        return product_id in self.entries


# ---------------------------------------------------------------------------
# 3. 三层比较 → 五个互斥类别
# ---------------------------------------------------------------------------
def classify_changes(index: IndexState, snapshot: dict) -> dict:
    """比较 (id, full_hash, semantic_hash) 三元组，返回 {类别: [product_id, ...]}。

    分流规则（论文 §2.1 Change classes）：
        主键没见过                              → new
        full_hash 相同                          → unchanged
        full_hash 不同 且 semantic_hash 不同     → semantic_changed（要重新 embedding）
        full_hash 不同 但 semantic_hash 相同     → metadata_only（保留向量，只更新记录/filter）
        索引里有、新快照里没有                    → deleted
    """
    plan = {cls: [] for cls in CHANGE_CLASSES}
    for product_id, record in snapshot.items():
        fh, sh = full_hash(record), semantic_hash(record)
        entry = index.entries.get(product_id)
        if entry is None:
            plan[NEW].append(product_id)
        elif entry.full_hash == fh:
            plan[UNCHANGED].append(product_id)
        elif entry.semantic_hash != sh:
            plan[SEMANTIC_CHANGED].append(product_id)
        else:
            plan[METADATA_ONLY].append(product_id)
    for product_id in index.entries:
        if product_id not in snapshot:
            plan[DELETED].append(product_id)
    for cls in CHANGE_CLASSES:
        plan[cls].sort()
    return plan


class SyncReport:
    """一次同步的执行结果（业务口径：哪些商品这次花了 embedding 的钱）。"""

    def __init__(self) -> None:
        self.enriched: int = 0           # 走过 enrichment 的商品数
        self.embedded: int = 0           # 发生过 embedding 的商品数
        self.metadata_updated: int = 0   # 只更新记录与 filter 的商品数
        self.deleted: int = 0
        self.skipped: int = 0
        self.embed_calls: int = 0
        self.reembed_ratio: float = 0.0  # 需重新 embedding 的商品占本次快照的比例

    def __repr__(self) -> str:
        return (f"SyncReport(embed_calls={self.embed_calls}, "
                f"metadata_updated={self.metadata_updated}, deleted={self.deleted}, "
                f"skipped={self.skipped}, reembed_ratio={self.reembed_ratio})")


def apply_sync(index: IndexState, snapshot: dict, plan: dict | None = None) -> SyncReport:
    """按论文 §2.1 的分流规则真正改索引。

    new / semantic_changed : enrich → embed → upsert
    metadata_only          : 保留 vector_id，只更新记录与 filter
    deleted                : 连同向量一起移除
    unchanged              : 什么都不做
    """
    plan = plan if plan is not None else classify_changes(index, snapshot)
    rep = SyncReport()
    for product_id in plan[NEW] + plan[SEMANTIC_CHANGED]:
        record = snapshot[product_id]
        rep.enriched += 1
        rep.embedded += 1
        rep.embed_calls += 1
        index.embed_calls += 1
        index.put(product_id, IndexEntry(
            product_id=product_id,
            full_hash=full_hash(record),
            semantic_hash=semantic_hash(record),
            vector_id=f"vec::{product_id}::{semantic_hash(record)}",
            record=dict(record),
        ))
    for product_id in plan[METADATA_ONLY]:
        entry = index.entries[product_id]
        rep.metadata_updated += 1
        # 关键：vector_id 原样保留，不调用 embedding
        index.put(product_id, IndexEntry(
            product_id=product_id,
            full_hash=full_hash(snapshot[product_id]),
            semantic_hash=entry.semantic_hash,
            vector_id=entry.vector_id,
            record=dict(snapshot[product_id]),
        ))
    for product_id in plan[DELETED]:
        rep.deleted += 1
        index.remove(product_id)
    rep.skipped = len(plan[UNCHANGED])
    total = len(snapshot)
    rep.reembed_ratio = round(rep.embed_calls / total, 4) if total else 0.0
    return rep


# ---------------------------------------------------------------------------
# 4. 构造数据（母婴 SKU 快照；非论文数据）
# ---------------------------------------------------------------------------
def make_snapshot() -> dict:
    """造一份母婴 SKU 快照：吸奶器 / 防溢乳垫 / 拉拉裤 / 奶瓶 各一。"""
    rows = [
        ("MB-001", "双边电动吸奶器 S1", "<p>可充电，低噪音</p>", "Lacta", "母婴/喂养/吸奶器", 89.9, True, 120),
        ("MB-002", "一次性防溢乳垫 100 片", "轻薄透气", "Lacta", "母婴/喂养/防溢乳垫", 12.5, True, 800),
        ("MB-003", "拉拉裤 L 码 4 包", "夜用加长", "BabyGo", "母婴/纸尿裤/拉拉裤", 39.9, True, 260),
        ("MB-004", "宽口径玻璃奶瓶 240ml", "耐高温", "BabyGo", "母婴/喂养/奶瓶", 15.0, True, 430),
    ]
    return {
        pid: {
            "id": pid, "name": name, "description": desc, "brand": brand,
            "category": category, "price": price, "currency": "USD",
            "availability": availability, "stock_qty": qty,
            "link": f"https://shop.example.com/p/{pid}",
        }
        for pid, name, desc, brand, category, price, availability, qty in rows
    }


# ---------------------------------------------------------------------------
# 5. 测试（K1 L5 断言）
# ---------------------------------------------------------------------------
def test_five_classes_are_disjoint_and_complete():
    """五个类别必须互斥且覆盖全部商品（论文：five disjoint classes）。"""
    index = IndexState()
    day1 = make_snapshot()
    apply_sync(index, day1)

    day2 = make_snapshot()
    day2["MB-001"]["price"] = 79.9                              # 只改价 → metadata-only
    day2["MB-003"]["description"] = "夜用加长，透气升级"          # 改描述 → semantic-changed
    del day2["MB-004"]                                          # 下架 → deleted
    day2["MB-005"] = {**day1["MB-001"], "id": "MB-005", "name": "单边电动吸奶器 S2"}  # 上新 → new

    plan = classify_changes(index, day2)
    seen = [pid for cls in CHANGE_CLASSES for pid in plan[cls]]
    assert len(seen) == len(set(seen)), "五个类别必须互斥"
    covered = set().union(*(set(plan[c]) for c in (NEW, SEMANTIC_CHANGED, METADATA_ONLY, UNCHANGED)))
    assert covered == set(day2), "四类必须覆盖新快照的全部商品"
    assert set(plan[DELETED]) == set(index.entries) - set(day2)
    assert plan[NEW] == ["MB-005"]
    assert plan[METADATA_ONLY] == ["MB-001"]
    assert plan[SEMANTIC_CHANGED] == ["MB-003"]
    assert plan[DELETED] == ["MB-004"]
    assert plan[UNCHANGED] == ["MB-002"]


def test_price_and_stock_change_keeps_vector_and_skips_embedding():
    """改价 / 补货 → 保留原向量、零 embedding 调用（本卡的成本主张）。"""
    index = IndexState()
    apply_sync(index, make_snapshot())
    vec_before = {pid: e.vector_id for pid, e in index.entries.items()}

    day2 = make_snapshot()
    day2["MB-001"]["price"] = 69.9
    day2["MB-001"]["stock_qty"] = 8
    rep = apply_sync(index, day2)

    assert rep.embed_calls == 0
    assert rep.metadata_updated == 1
    assert rep.skipped == 3
    assert index.entries["MB-001"].vector_id == vec_before["MB-001"]
    assert index.entries["MB-001"].record["price"] == 69.9


def test_semantic_hash_is_blind_to_metadata_fields():
    """semantic hash 对元数据字段不敏感，而 full hash 必须敏感。"""
    rec = make_snapshot()["MB-001"]
    h0 = semantic_hash(rec)
    for key, value in (("price", 1.0), ("stock_qty", 0), ("availability", False), ("currency", "EUR")):
        assert semantic_hash({**rec, key: value}) == h0, key
        assert full_hash({**rec, key: value}) != full_hash(rec), key


def test_each_semantic_field_change_triggers_reembedding():
    """name / description / brand / category 四个字段任一变化都要重新 embedding。"""
    for key in SEMANTIC_FIELDS:
        index = IndexState()
        apply_sync(index, make_snapshot())
        day2 = make_snapshot()
        day2["MB-002"] = {**day2["MB-002"], key: "改过的值"}
        plan = classify_changes(index, day2)
        assert plan[SEMANTIC_CHANGED] == ["MB-002"], key
        assert plan[NEW] == [] and plan[METADATA_ONLY] == [] and plan[DELETED] == []


def test_idempotent_second_sync_embeds_nothing():
    """同一份快照同步两次：第二次零 embedding（对账是幂等的）。"""
    index = IndexState()
    snap = make_snapshot()
    rep1 = apply_sync(index, snap)
    assert rep1.embed_calls == len(snap)          # 首次没有历史状态 → 全部是 new
    rep2 = apply_sync(index, snap)
    assert rep2.embed_calls == 0
    assert rep2.skipped == len(snap)


def test_html_and_whitespace_noise_does_not_trigger_reembedding():
    """feed 的排版噪声（HTML 包裹、首尾空白）不应被当成语义变化。"""
    index = IndexState()
    apply_sync(index, make_snapshot())
    noisy = make_snapshot()
    noisy["MB-002"]["description"] = "  <div>轻薄透气</div>  "
    plan = classify_changes(index, noisy)
    assert plan[SEMANTIC_CHANGED] == [] and plan[NEW] == []
    assert plan[UNCHANGED] == sorted(noisy)


def test_embedding_calls_scale_with_delta_not_catalogue_size():
    """embedding 调用次数与 delta 大小成正比，而不是与目录大小成正比。"""
    index = IndexState()
    apply_sync(index, make_snapshot())

    day2 = make_snapshot()
    day2["MB-001"]["price"] = 59.9                 # metadata-only → 0 次
    day2["MB-003"]["category"] = "母婴/纸尿裤/学步裤"  # semantic-changed → 1 次
    rep = apply_sync(index, day2)

    assert rep.embed_calls == 1
    assert rep.embed_calls < len(day2)
    assert rep.metadata_updated == 1 and rep.skipped == 2


def test_delete_removes_entry_and_new_gets_fresh_vector():
    """下架要连同向量移除；上新拿到的必须是新向量，不能复用旧商品的。"""
    index = IndexState()
    day1 = make_snapshot()
    apply_sync(index, day1)
    old_vec = index.entries["MB-004"].vector_id

    day2 = make_snapshot()
    del day2["MB-004"]
    day2["MB-006"] = {**day1["MB-004"], "id": "MB-006", "name": "宽口径 PPSU 奶瓶 240ml"}
    apply_sync(index, day2)

    assert not index.has("MB-004")
    assert index.has("MB-006")
    assert index.entries["MB-006"].vector_id != old_vec


# ---------------------------------------------------------------------------
# 6. 业务演示
# ---------------------------------------------------------------------------
def _business_demo() -> None:
    label = {
        NEW: "新增（要 enrich+embed）",
        SEMANTIC_CHANGED: "语义变化（要重新 embed）",
        METADATA_ONLY: "只有价格/库存变（保留向量）",
        DELETED: "下架（删除向量）",
        UNCHANGED: "无变化（跳过）",
    }
    print("== 活目录增量对账演示（构造数据，非论文数据）==")

    index = IndexState()
    day1 = make_snapshot()
    rep1 = apply_sync(index, day1)
    print(f"首次同步：embedding 调用 {rep1.embed_calls} 次 / {len(day1)} 个 SKU（等于全量重建）")

    day2 = make_snapshot()
    day2["MB-001"]["price"] = 76.0
    day2["MB-001"]["stock_qty"] = 5
    day2["MB-003"]["description"] = "夜用加长，新增透气升级"
    del day2["MB-004"]
    day2["MB-005"] = {**day1["MB-004"], "id": "MB-005", "name": "宽口径 PPSU 奶瓶 240ml"}

    plan = classify_changes(index, day2)
    for cls in CHANGE_CLASSES:
        print(f"  {label[cls]:24s} {len(plan[cls])} 个：{plan[cls]}")
    assert sum(len(plan[c]) for c in CHANGE_CLASSES) == len(day2) + len(plan[DELETED])

    rep2 = apply_sync(index, day2, plan)
    print(f"\n第二次同步：embedding 调用 {rep2.embed_calls} 次；"
          f"全量重建需要 {len(day2)} 次 —— 本卡只处理 delta")
    print(f"  需要重新 embedding 的占比 = {rep2.reembed_ratio:.1%}（本次，构造数据）")
    print(f"  仅元数据变化、零 embedding 成本的 SKU = {rep2.metadata_updated} 个")


if __name__ == "__main__":
    _business_demo()
```

---

## ④ 技能关联

- **组合（数据流的下游）｜`Skill-Diversity-Reranking-SMMR.md`**（05-推荐系统）：
  论文的推荐路径最后一步是「a greedy selector adds brand and category variety」（⑥ Q19）——
  一个贪心的多样性选择器。那张卡给的是 MMR / SMMR 里可调的 λ 折中写法，
  可以把论文一笔带过的 greedy selector 升级成可调 λ 的 MMR，用来控制「多品牌 / 多品类」的程度。
  依赖关系是**单向且具体**的：那张卡的候选池必须与在售目录一致，否则多样性重排会在**已经下架的候选**上做优化；
  本卡正好提供这份候选池的新鲜度保证。数据流：本卡的索引状态 → 那张卡的候选池 → 多样性重排。
- **延伸｜`Skill-Semantic-ID-Retrieval-RPG.md`**（05-推荐系统）：
  本卡的 `sem_hash` 覆盖 name / description / brand / category（⑥ Q11、⑥ Q19 里 query 与商品共享同一 embedding 空间），
  本质是**内容侧语义指纹**；那张卡把同一批属性编码成语义 ID 做生成式检索。
  两者可以组合：本卡判断「要不要重算」，那张卡决定「重算成什么 ID」。
- **组合｜`Skill-Long-Term-Preference-Memory.md`**（16-智能体工程）：
  论文的对话层里有 elicitor 子代理、用户 profile 与会话状态（⑥ Q18、⑥ Q37），
  而 profile 是**独立于目录**的存储。那张卡负责「记住用户偏好」，
  本卡负责「保证被偏好命中的商品还在卖」——两张卡的失效模式正好互补：
  偏好记错是推荐不相关，目录过期是推荐已下架。
- **触发点｜`Skill-Cold-Start-Meta-Learning-PAM.md`**（05-推荐系统）：
  新品上架的瞬间在目录侧就是本卡的 `new` 类别（⑥ Q13），而在推荐侧是零交互冷启动。
  组合方式：本卡的 new 清单直接作为那张卡的冷启动触发队列，
  避免「商品已上架但推荐侧还在等下一次全量重建」。
- **同域上下游｜`Skill-Agentic-Catalog-Enrichment.md`**（00-电商Agent，同批并行产出的另一张卡）：
  那张卡解决「属性空格怎么用多源证据补全并写库」（Scout 取证 → Judge 裁决 → verify-before-write），
  本卡解决「补完之后什么时候需要重算、哪些变化根本不用重算」。两张卡在管线上是**一条流水线的两段**：
  富化决定记录**长什么样**，本卡的对账决定**哪些记录才值得进富化** ——
  本卡的 `new` 与 `semantic-changed` 两类正是那张卡的输入队列，而 `metadata-only` 类**完全不进富化**
  （论文的分流规则，⑥ Q14/Q15）。
  ⚠️ 引用口径：本卡只核对了该卡的文件名、标题与主题（同为 00-电商Agent 域），**未逐段阅读其内容**，
  因此不对其结论负责；需要联合使用时请以该卡自身为准。

---

## ⑤ 商业价值评估

**ROI 公式**（**本项目的推算，不是论文结论**——论文没有报告任何业务指标、成本或人力数字）：

**ROI = (ΔC_sync + ΔC_risk) / C_build**

- **ΔC_sync = N_days × N_feed × (T_full − T_incr) × c_compute** —— 同步机时 / 调用成本节省，
  其中 `N_days × N_feed` 是年同步次数，`T_full − T_incr` 是单次全量重建与单次增量同步的耗时差。
- **ΔC_risk = N_stale × P_case × C_case** —— 因推荐过期商品而减少的客诉 / 退货损失，
  `N_stale` 是一年内「推荐了已下架或断货商品」的会话数。
- **C_build** = 一次性建设成本（对账器 + 索引状态存储 + 调度 + 监控看板）。

| 参数 | 含义 | 来源 |
|---|---|---|
| `T_full` / `T_incr` | 单次全量重建 / 单次增量同步耗时 | **必须企业自测**。论文在**匿名 500 条记录**目录上实测「增量同步耗时为全量重建的 1.8%–12.3%」（⑥ Q21/Q22）——**只能当方向性参考**：它量的是同步成本，且规模、enrichment 调用数、向量库与企业都不同 |
| `N_days × N_feed` | 年同步次数 = 每日同步次数 × feed 数（独立站 + 各平台店） | 企业调度配置 |
| `c_compute` | 单位机时与单次 embedding 调用成本 | 企业云账单（论文未报告） |
| `N_stale` | 一年内「推荐了已下架 / 断货商品」的会话数 | 企业客服工单系统口径 |
| `P_case`、`C_case` | 由过期推荐触发客诉的概率 / 单次客诉处理成本 | 企业客服成本口径（论文未报告） |
| `C_build` | 一次性建设成本 | 内部人日 × 单价 |

**保守结论**：本卡**不给收益金额结论**。可以确定的是收益的形状而不是大小：
`ΔC_sync` 在「改价 / 补货占多数、真语义改动占少数」的真实母婴目录里由 metadata-only 类主导，
而这一类**完全不调用 embedding**；`ΔC_risk` 的大小则取决于 feed 与仓库真实库存的一致性——
论文对后者**没有任何评测**，必须先自测 feed 的时效才能把它写进预算。

- **实施难度**：⭐⭐⭐☆☆ —— 对账与调度本身是工程活，不涉及模型训练；
  难度集中在「把索引状态持久化」与「非事务写入的补偿设计」（截断 feed 检测、staging / 回滚、异常 delta 监控，⑥ Q27）。
- **优先级**：⭐⭐⭐⭐☆ —— 会话导购 / 客服机器人一旦上线，目录新鲜度就是客诉的直接来源；
  **先有对账，再谈排序**。若企业连「每日 feed 快照归档」都还没有，优先级降到 3 星，先补这一项（成本几乎只有存储）。
- **评估依据**：收益侧论文零证据（⑥ Q30），工程侧论文给了完整机制描述与一个可运行的合成同步（⑥ Q28），
  所以这是一张「**机制可信、收益需自测**」的卡，而不是「拿来就有效果」的卡。

---

## ⑥ 原文引用

> 全部引文自 `paper2skills-vault/papers/00-电商Agent/p2s-2026-0014/fulltext.md`
> 逐字复制，未拼接、未改写；每条均由 `quote_check.py` 回查底本判为 VERBATIM。

### A. 问题：静态预索引的评测设定 vs 持续变化的真实目录（① 段立论）

> 原文："Conversational recommender systems based on large language models (LLMs) are usually evaluated on static, pre-indexed item collections, yet e-commerce catalogues change continuously as products are added or removed, repriced, and restocked."
> 出处：2608.27006 §Abstract

> 原文："Most large language model (LLM)-based conversational recommender systems (CRSs) are evaluated over fixed benchmark collections (He et al., 2023; Jannach et al., 2021), but in production the catalogue is a live object, continually updated."
> 出处：2608.27006 §1 Introduction

> 原文："Re-indexing the whole catalogue on every change is wasteful, yet letting the index drift degrades recommendations and surfaces out-of-stock or discontinued items."
> 出处：2608.27006 §1 Introduction

> 原文："Our emphasis is orthogonal to model quality."
> 出处：2608.27006 §1 Introduction（点明本文的重点不在模型质量）

> 原文："We treat catalogue freshness—keeping the index consistent with a live assortment—as the engineering problem that makes such systems production-viable, complementing work on adapting LLM recommenders to refreshed indices (He et al., 2025)."
> 出处：2608.27006 §1 Introduction

### B. self-refreshing retriever 的定义与「只处理增量」

> 原文："Its central component is a self-refreshing retriever that ingests a merchant product feed, enriches the records, and synchronizes them into a vector index."
> 出处：2608.27006 §Abstract

> 原文："On each run, per-item hashes identify which products are new, changed, deleted, or unchanged, so only the delta is processed rather than rebuilding the whole catalogue."
> 出处：2608.27006 §Abstract

> 原文："We demonstrate a conversational shopping assistant built around one contribution: a self-refreshing retriever that re-embeds only new or semantically changed products, keeping synchronization proportional to the changed subset."
> 出处：2608.27006 §1 Introduction（本文唯一贡献的表述）

> 原文："Each manual or scheduled run compares the latest catalogue snapshot with the index and applies only the difference; it does not monitor the feed continuously."
> 出处：2608.27006 §2.1 Self-Refreshing Retriever（**触发方式**：手动/定时，不是流式监听）

> 原文："Our proof of concept uses ChromaDB through a swappable VectorStore interface."
> 出处：2608.27006 §2 System Overview（PoC 用 ChromaDB，藏在可替换的 VectorStore 接口后面）

### C. 三种标识的分工：stable ID / full hash / semantic hash

> 原文："A stable product ID links snapshots and drives exact updates and deletions, but cannot answer natural-language queries."
> 出处：2608.27006 §2.1 IDs, hashes, and embeddings

> 原文："A full hash detects any feed-field change; a semantic hash over name, description, brand, and category identifies changes requiring re-embedding."
> 出处：2608.27006 §2.1 IDs, hashes, and embeddings（三种哈希的分工，本卡 ① 段的核心）

> 原文："The resulting vectors make products retrievable; the generative LLM is only an enrichment fallback."
> 出处：2608.27006 §2.1 IDs, hashes, and embeddings（生成式 LLM 只做 enrichment 兜底）

### D. 五个变化类别与分流规则

> 原文："Comparing IDs and hashes yields five disjoint classes."
> 出处：2608.27006 §2.1 Change classes

> 原文："New and semantically changed records are enriched, embedded, and upserted; enrichment resolves category paths and extracts attributes by rule, with generative fallback."
> 出处：2608.27006 §2.1 Change classes（new + semantic-changed 才走 enrich/embed/upsert）

> 原文："Metadata-only changes, such as price or stock, retain the vector while updating the record and filters."
> 出处：2608.27006 §2.1 Change classes（metadata-only 保留向量——成本节省的来源）

> 原文："Deleted records are removed, and unchanged records are skipped."
> 出处：2608.27006 §2.1 Change classes

> 原文："New and semantically changed items are enriched, embedded, and upserted. Metadata-only changes update the stored record while keeping its vector. Deleted items are removed, and unchanged items are skipped."
> 出处：2608.27006 §Figure 2 图注（Engine architecture，与 §2.1 的分流规则一致）

### E. LLM 只做意图分类与偏好 elicitation（registry 出入的关键证据）

> 原文："A controller-based dialogue layer consumes this index, using an LLM only for intent classification and preference elicitation while retrieval, reranking, and diversity selection run as dedicated functions."
> 出处：2608.27006 §Abstract（**注意：原文是 intent classification AND preference elicitation 两件事**）

> 原文："calling a generative model only for intent and elicitation"
> 出处：2608.27006 §Figure 2 图注（Engine architecture）

> 原文："The conversation pipeline follows an orchestrator-as-controller pattern (Yao et al., 2023; Schick et al., 2023; Huang et al., 2025): a generative model classifies messages into eight intents, composes replies, and uses an elicitor sub-agent to ask one to three clarifying questions when preferences are vague (Shimazu, 2001; Sun and Zhang, 2018)."
> 出处：2608.27006 §2.2 Conversational Pipeline（orchestrator-as-controller；LLM 分类八类意图 + elicitor 子代理）

> 原文："Recommendation uses content-based semantic retrieval: query and product text share one embedding space, metadata filters restrict candidates, an optional non-generative model reranks them (Yang and Chen, 2024; Kemper et al., 2024), and a greedy selector adds brand and category variety."
> 出处：2608.27006 §2.2 Conversational Pipeline（检索 / 过滤 / 重排 / 多样性是专用函数，不是 LLM）

> 原文："This generation-free path keeps cost predictable, although embedding and reranking still call external models (Kolb et al., 2025); the pipeline detects the user’s language, retrieves in English, and replies in that language."
> 出处：2608.27006 §2.2 Conversational Pipeline（无生成路径；**检测用户语言、用英文检索、用用户语言回复**）

> 原文："The engine has three subsystems (Appendix A): a catalogue pipeline that ingests and indexes products, a conversation pipeline that handles multi-turn dialogue, and a storage layer, written by the former and read by the latter, providing vector search, user profiles, and session state. This shared storage decouples catalogue synchronization from dialogue, allowing each pipeline to run independently. All generative, embedding, and reranking calls use a single proxy, making model choices configuration rather than code."
> 出处：2608.27006 §2 System Overview（三个子系统 + 共享存储解耦；所有模型调用走单一代理）

### F. 演示形态：WhatsApp 购物助手

> 原文："Our demonstration is a WhatsApp shopping assistant in which catalogue changes reach the recommendations after the next successful sync."
> 出处：2608.27006 §Abstract（目录变化在**下一次成功同步后**才到达推荐）

> 原文："Users access the live demonstration on any smartphone through WhatsApp, with no application to install (Figure 1)."
> 出处：2608.27006 §3 Demonstration

> 原文："Sessions may be anonymous or personalised from prior purchases; the assistant elicits preferences, searches the live catalogue, and returns diverse in-stock products with links, and each successful sync exposes catalogue changes."
> 出处：2608.27006 §3 Demonstration

### G. Table 1：论文唯一的量化证据（**同步成本**，不是推荐质量）

> 原文："Incremental synchronization of an anonymized 500-record catalogue (medians over three or five runs; full rebuild: 2.914 s)."
> 出处：2608.27006 §3 Demonstration（Table 1 表注：匿名 500 条记录目录、三次或五次运行取中位数、全量重建 2.914 s）

> 原文："| Change | E | Emb | M | D | Time (s) | Full (%) | | None | ✗ | ✗ | ✗ | ✗ | 0.053 | 1.8 | | Add product | ✓ | ✓ | ✗ | ✗ | 0.321 | 11.0 | | Price/stock | ✗ | ✗ | ✓ | ✗ | 0.072 | 2.5 | | Description/category | ✓ | ✓ | ✗ | ✗ | 0.357 | 12.3 | | Delete product | ✗ | ✗ | ✗ | ✓ | 0.062 | 2.1 |"
> 出处：2608.27006 §3 Demonstration（Table 1 表体，五行变化场景：无变化 / 新增 / 改价缺货 / 改描述类目 / 删除商品）

> 原文："Table 1 confirms the intended change classification: price or stock changes update only metadata, description or category changes re-enrich and re-embed the record, and all ID, hash, feed-field, and embedding-call checks passed."
> 出处：2608.27006 §3 Demonstration（Table 1 的结论句）

### H. 论文自承的边界、局限与可复现性

> 原文："Writes are non-transactional, so production requires truncated-feed detection, staging or rollback, and abnormal-delta monitoring; we make no tens-of-millions-scale claim."
> 出处：2608.27006 §3 Demonstration（**写入非事务**；论文明确不做千万级承诺）

> 原文："Documentation, engine excerpts, a demo video, and a runnable synthetic sync are public; the full engine, catalogue, and Infobip integration remain private."
> 出处：2608.27006 §3 Demonstration（**公开什么、保留什么**：完整引擎与目录私有）

> 原文："Synchronization rejects parser failures, invalid or duplicate IDs, and incompatible index versions, and rejects empty or incomplete snapshots to prevent inferred mass deletion; it prepares enrichment and embeddings before writing, then applies upserts and metadata updates before deletions."
> 出处：2608.27006 §3 Demonstration（同步入口的拒收规则与写入顺序）

> 原文："The prototype accepts Google Merchant Center Atom feeds directly; other catalogue sources or messaging channels require adapters to the product schema or engine API, while filtering, batching, consistency, and operations remain backend concerns behind the VectorStore interface."
> 出处：2608.27006 §3 Demonstration（原生支持 Google Merchant Center Atom feed；其他源要写 adapter）

> 原文："We presented a conversational shopping assistant whose self-refreshing retriever keeps a vector index consistent with a live catalogue, processing only new, changed, or removed products on each sync; a measured case study confirms the classification and operation counts behave as intended, and the assistant is deployed as a WhatsApp demo."
> 出处：2608.27006 §4 Concluding Remarks（论文对自己贡献的收束：a measured case study）

> 原文："The evaluation covers synchronization not ranking quality; an offline relevance study and a live user study of recommendation quality and cost remain future work, alongside freshness-aware retrieval and hybrid ranking."
> 出处：2608.27006 §4 Concluding Remarks（**本文最重要的局限**：评测只覆盖同步，不覆盖排序质量）

> 原文："A live chatbot, documentation, and a recorded walkthrough are available at [https://github.com/infobip/infobip-agentic-crs]."
> 出处：2608.27006 §Abstract（开源入口：live chatbot / 文档 / 录屏）
