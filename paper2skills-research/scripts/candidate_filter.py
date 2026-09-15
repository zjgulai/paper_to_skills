#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""candidate_filter.py — 三段式检索过滤（关键词库 v2 的可执行部分）

背景：v1（`关键词库.md`）只有正向词，实测污染严重 —— 04 域（供应链）27% 命中
「软件供应链 / SBOM / 包依赖」，14 域（用户分析）64% 命中医学影像与流行病学队列研究。
v2 因此引入负向词与约束词。

为什么过滤器是**独立脚本**而不是塞进 arXiv 查询串：
arXiv API 的 `search_query` 虽支持 ANDNOT，但把负向词写进查询串会让结果随
分词与转义细节漂移，出现「今天空集、明天有结果」的不可复现现象。
本脚本改为「收割阶段只捞全 → 过滤阶段本地判定」，规则可复现、可审计、可回归。

三段式语义：
    保留 = 命中正向词(由收割阶段保证) 且 未命中该域负向词 且 满足该域约束词

⚠️ 设计原则：**负向词按域生效**，因为同一个词在不同域含义相反：
    `trial`   —— 01/02 域是方法论词（randomized controlled trial），14 域是医学词
    `cohort`  —— 14 域是「同期群」，医学里是「队列研究」
    `ad`      —— 13 域是广告，医学里是 Alzheimer/ADHD

用法：
    python3 candidate_filter.py --report          # 只看污染统计，不写文件
    python3 candidate_filter.py --apply           # 过滤并写 arxiv_candidates_filtered.json
    python3 candidate_filter.py --selftest        # 自检：证明过滤器真的能拦下污染
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ⚠️ `P2S_REPO` 的语义**全仓只有一个**：**仓库根**（不是本文件所在的 `paper2skills-research/`）。
# 首版这里写成了「本文件往上两级」，而 `domains.py` 写的是「往上三级 = 仓库根」——
# **同一个环境变量在两个文件里是两个意思**，正是本次修复在查的那类缺陷本身。
# 发现它的是验收面新增的端到端用例（夹具上 `candidate_filter --check` 报「候选池读不到」）。
REPO = Path(os.environ.get("P2S_REPO") or Path(__file__).resolve().parents[2]).resolve()
ROOT = REPO / "paper2skills-research"
DATA = ROOT / "data"

# ⚠️ `domains.py` 是域名的**唯一事实源**，本文件不再另写一份域名表。
# 本次修复前正是「本文件写新名 / `arxiv_harvest.py` 写旧名」两套并存，
# 而 `filter_pool` 的 `if not domains: keep` 让认不出的标签**整篇放行** ——
# 141/1046 篇的负向词与约束词**一条都没生效，且没有任何读数**（详见 docstring）。
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import domains as _domains  # noqa: E402

# ---------------------------------------------------------------------------
# 域 → 领域目录名（**从唯一事实源派生**）
# ---------------------------------------------------------------------------
DOMAIN_DIR = {d: d for d in _domains.CANONICAL}


# ---------------------------------------------------------------------------
# 全局负向词（所有域一律生效）
# ---------------------------------------------------------------------------
GLOBAL_NEGATIVE = [
    # 软件供应链（04 域重灾区）
    "software supply chain", "sbom", "software bill of materials",
    "package detection", "dependency vulnerability", "model lineage", "model card",
    # 纯基础设施
    "kv cache", "serving throughput", "gpu kernel",
    # 医学（全局收敛，各域另有更严的清单）
    "clinical", "patient", "lesion", "mri", "eeg", "ecg", "fmri",
    "radiotherapy", "histopathology", "tumor",
]

# ---------------------------------------------------------------------------
# 按域负向词（**只在指定域生效**，理由见模块 docstring）
# ---------------------------------------------------------------------------
DOMAIN_NEGATIVE = {
    # ⚠️ 00 域本次**首次**进入本表。它此前压根不在 `DOMAIN_DIR` 里，于是它的
    # 37 篇被整篇放行。词表来源：该域自己池子里**实测点名的**域外论文（逐条可回指）。
    "00-电商Agent": ["catalyst", "chemistry", "molecul", "protein", "crystal"],
    "01-因果推断": ["clinical", "patient", "drug", "gene", "epidemiolog", "biomarker"],
    # ⚠️ 02 域刻意**不**含 "trial"：RCT 是方法论词
    "02-A_B实验": ["wet lab", "robot", "in vitro"],
    "03-时间序列": ["weather", "climate", "traffic flow", "wind power",
                    "load forecasting", "eeg", "ecg", "seismic"],
    "04-供应链": ["software supply chain", "sbom", "npm", "pypi", "malware",
                  "vulnerability", "repository", "compiler"],
    "05-推荐系统": ["movie", "music recommendation", "news recommendation",
                    "poi recommendation", "friend recommendation"],
    "06-增长模型": ["employee turnover", "student dropout", "attrition rate"],
    "07-NLP-VOC": ["clinical note", "hate speech", "fake news", "machine translation"],
    "08-知识图谱": ["molecular", "protein", "drug discovery", "biomedical",
                    "chemistry", "crystal"],
    "09-DataAgent-LLM": ["code generation benchmark", "gui agent", "web agent"],
    "10-MAS": ["multi-agent reinforcement learning"],
    "12-ML基础": ["speech recognition", "image classification", "object detection"],
    "13-广告分析": ["adversarial attack", "adhd", "alzheimer", "adversarial example"],
    "14-用户分析": ["clinical", "patient", "mri", "eeg", "fmri", "lesion",
                    "radiotherapy", "pet scan", "tumor", "cohort study",
                    "epidemiolog", "diagnosis"],
    "15-营销投放分析": ["health promotion", "academic promotion", "faculty"],
    "16-智能体工程": ["embodied agent", "robot manipulation", "gui agent"],
}

# ---------------------------------------------------------------------------
# 按域约束词：宽词必须与约束词之一共现才保留
# ---------------------------------------------------------------------------
# ⚠️ 07 / 09 / 15 三个域本次**首次被行使**（它们的标签此前与 harvest 不一致，
# 旧标签见 `domains.RETIRED`），一被行使就暴露：**它们的约束词不认自己收割查询的
# 定义性短语**。实测证据（逐篇可回指）：
#   `15-营销投放分析` 的查询第一条就是 `abs:"marketing mix model" OR abs:"media mix"`，
#     而约束词只有 `retail/e-commerce/sales/price/promotion` —— 于是 4 篇
#     Marketing Mix Modeling 论文被它**自己捞进来又扔出去**。
#   `07-NLP-VOC` 的查询是 `abs:"aspect-based sentiment" OR abs:"opinion mining"`，
#     约束词不认 `sentiment` —— 4 篇 ABSA 论文同病。
#   `09-DataAgent-LLM` 的查询是 `abs:"data analysis agent" OR abs:"text-to-SQL"`，
#     约束词不认 `text-to-sql` —— 14 篇 Text-to-SQL 论文同病。
# 修法是**把该域收割查询的定义性短语补进约束表**（其余 14 个域本来就是这样：
# `03` 的查询有 `demand forecasting` 而约束表有 `demand`；`04` 的查询有
# `inventory management` 而约束表有 `inventory`）。**没有发明新词**。
# ⚠️ 代价已登记、不掩盖：补 `text-to-sql` 会同时放进 4 篇「摘要里顺带提到
# text-to-SQL 而主题是别的」的论文（`Non-vacuous Generalization Bounds` /
# `Memory Reward Inflation` / `AuthentiCity` / `Knowing When Not to Reuse`）。
# 真命中与它们的区分点是**标题 vs 摘要的显著性**，而现有约束词规则表达不了这个
# 区分（改它会同时动到已判的 905 篇）⇒ **登记为独立工单，不在本批改规则**。
DOMAIN_CONSTRAINT = {
    "00-电商Agent": ["e-commerce", "ecommerce", "cross-border", "commerce",
                     "shopping", "assortment", "catalog", "merchant", "retail",
                     "product", "customer"],
    "01-因果推断": ["e-commerce", "retail", "advertis", "marketing", "pricing",
                    "subscription", "customer", "business"],
    "02-A_B实验": ["online", "platform", "marketplace", "advertis", "pricing",
                   "user", "e-commerce"],
    "03-时间序列": ["demand", "sales", "retail", "inventory", "e-commerce", "supply chain"],
    "04-供应链": ["goods", "physical", "retail", "warehouse", "fulfillment",
                  "inventory", "logistics", "order"],
    "05-推荐系统": ["e-commerce", "product", "item", "user", "session", "catalog"],
    "06-增长模型": ["subscription", "e-commerce", "retail", "customer", "platform"],
    "07-NLP-VOC": ["e-commerce", "product", "customer", "review", "feedback",
                   "aspect-based sentiment", "opinion mining"],
    "08-知识图谱": ["e-commerce", "product", "customer", "agent", "retrieval"],
    "09-DataAgent-LLM": ["analytics", "business", "enterprise", "database", "table",
                         "text-to-sql", "data analysis agent", "autonomous data science"],
    "12-ML基础": ["tabular", "business", "prediction", "dataset"],
    "13-广告分析": ["advertis", "marketing", "campaign", "budget", "roas"],
    "14-用户分析": ["e-commerce", "retail", "subscription", "churn", "user", "customer"],
    "15-营销投放分析": ["retail", "e-commerce", "sales", "price", "promotion",
                        "marketing mix", "media mix"],
    "16-智能体工程": ["llm", "agent", "tool", "context"],
}

# 强制要求摘要里出现电商信号的「重灾区」域：只靠负向词不足以分开
STRICT_ECOMMERCE_DOMAINS = {"14-用户分析", "04-供应链"}


# ---------------------------------------------------------------------------
# 约束词的「类」—— 由**实测域精度**导出，不是手写声明（PHASE6 P2b）
# ---------------------------------------------------------------------------
# 上一批（P2）把显著性假设的两个机械形式都推翻了，留下的交接是：
#
#     「先把表拆开，再谈显著性；在表拆开之前，任何位置规则都会被泛词短路。」
#
# 那是**对的诊断**（`table` 在 5% 处就能救活主题短语埋在 80% 处的论文），
# 但本批实测：**三个语法拆表判据全部失败**，逐条反例可复现 ——
#   · 「收割查询里被引号括起的短语」⇒ `table`（09 查询三 `abs:"table" AND …`）、
#     `agent` / `llm` / `tool` / `context`（16 的查询）**全都在引号里**；
#   · 「顶层 OR 分支的短语」⇒ 16 的 `abs:"agent skills" OR abs:"skill library" AND abs:"LLM"`
#     仍会把 `agent` 判成主题词；
#   · 「含空格的复合词」⇒ `text-to-sql` **没有空格**（漏掉最关键的那个词），
#     而 `supply chain` 有空格（误收）。
# ⇒ **语法判据问的是「这个词长什么样」，而需要的是「命中它意味着什么」。**
#
# 成立的是**实测**判据：某约束词对某域是**主题类**，当且仅当它的**域精度 > 0.5** ——
#
#     域精度(d, w) = 「摘要或题名命中 w 的论文」里「query_groups 解析后含 d」的比例
#
# 直觉：主题短语的语义角色就是「命中它 ⇒ 这篇属于这个域」；泛词的命中只是弱提示。
# 这不是同义反复 —— `table` 同样出自 09 自己的收割查询，精度却只有 **0.195**。
#
# ⚠️ **两个数、一个窗口，必须一起读**（全池普查 42 条逐条标注见
# `data/constraint-salience-labels.json`）：
#   · 污染召回 **18/26 → 26/26**（现行全部词的规则漏 8 条）；标出数 **68 → 42**；
#   · 纯污染率 **26.5% → 61.9%**；42 条**全部已标注**（现行那 68 条里有 43 条未标注）。
#   · 污染召回 100% 的窗口是 **[0.4, 0.5]**；>0.5 立刻掉到 65.4%。
#     上界由 `e-commerce`(00)=**0.521** 与 `database`(09)=**0.524** 顶住（各只高 0.02）。
#     ⇒ **窗口很窄，换池子必须重算**；域精度是**池子的性质**，不是词的性质。
# ⇒ 0.5 是唯一有非任意理由的值（「命中该词比不命中更可能属于本域」= 多头占优线），
#   同时是窗口内精度最高的一端。**其余阈值都是「碰巧能用」，本批不采纳。**
#
# ⚠️ 一条**实测有效但未采纳**的候选：词长 ≥7（36 条、召回仍 26/26、纯污染率 72.2%）。
#   它把 `item`（05 唯一的类词）整类移除 —— 省掉 5 条真命中误报，代价是该域**全部召回归零**。
#   「按字符串长度猜具体性」正是 #86/#104 那类代理指标，且它让读数好看，故**只登记不采纳**。
#
# ⚠️ 本节的产物**仍然不参与保留/丢弃**（`judge()` 一字未改）。理由与 P2 相同，再加一条：
#   42 条里有 **10 条**逐条确认过的真命中 ⇒ 拿它丢数据 = 24% 的已知静默损失。
#   它买到的是**分诊用的召回**（26/26 而非 18/26），不是判决权。
TOPIC_CLASS_MIN_PRECISION = 0.5


def domain_precision(items: list[dict]) -> dict:
    """每个 (域, 约束词) 的**域精度**。只从池子与收割标签算，不读任何手写表。

    返回 `{(domain, word): {"precision": float|None, "hits": int, "in_domain": int}}`。
    `hits == 0`（该词在池子里从没出现）⇒ precision 为 None，**不是 0** ——
    「没测到」与「测了是 0」是两件事（本仓库危险性排序 3 > 2 > 1 > 0）。
    """
    resolved = {}
    for it in items:
        resolved[_item_id(it)] = set(_domains.resolve_groups(it.get("query_groups") or [])["resolved"])
    out: dict = {}
    for dom, words in DOMAIN_CONSTRAINT.items():
        for w in words:
            hits = [k for k, (t, a, _m) in _blobs(items).items() if w in t or w in a]
            in_dom = sum(1 for k in hits if dom in resolved.get(k, ()))
            out[(dom, w)] = {
                "precision": (in_dom / len(hits)) if hits else None,
                "hits": len(hits), "in_domain": in_dom,
            }
    return out


def _blobs(items: list[dict]) -> dict:
    """`{id: (标题小写, 摘要小写, 元数据小写)}` —— 域精度与证据形态共用同一份切分。"""
    return {_item_id(i): ((i.get("title") or "").lower(),
                          (i.get("abstract") or "").lower(),
                          ((i.get("comment") or "") + " . " + (i.get("journal_ref") or "")).lower())
            for i in items}


def topic_class_words(domain: str, prec: dict, min_precision: float | None = None) -> list[str]:
    """该域的**主题类**约束词（精度 > 阈值的那些），按精度降序、同精度按词名。

    ⚠️ 该域可能**一个都没有**（实测 `02-A_B实验` 就是：`online`/`platform`/`marketplace`/
    `advertis`/`pricing`/`user` 全是泛词，最高 0.385）—— 那不是缺陷，是「该域的约束表
    里没有能代表主题的词」。**必须报出来**（`no_class`），不许静默为空。
    """
    thr = TOPIC_CLASS_MIN_PRECISION if min_precision is None else min_precision
    got = [(prec.get((domain, w), {}).get("precision") or 0.0, w)
           for w in DOMAIN_CONSTRAINT.get(domain, [])]
    return [w for p, w in sorted(got, key=lambda pw: (-pw[0], pw[1])) if p > thr]


# ---------------------------------------------------------------------------
# 约束词命中的「证据形态」—— PHASE6 P2
# ---------------------------------------------------------------------------
# 这一节**只记录，不判定**。它的存在理由是上一批登记的一条工单：
#
#   「补 `text-to-sql` 会同时放进 4 篇摘要里顺带提到它的论文。真命中与它们的
#     区分点是**标题 vs 摘要的显著性**，而现有约束词规则表达不了这个区分。」
#
# 本批把这条假设**做成了可测的形式**，结论是**假设不成立**（两个机械形式都错，
# 读数见 `reports/PHASE6-P2-约束词显著性与过滤器未接线.md`）：
#   · 字面版「约束词必须命中标题」⇒ 会砍掉 597 个「仅摘要」保留对里的 **326 个**；
#   · 深度版「只在摘要后半出现 ⇒ 顺带」⇒ 全池标 68 条，其中 62 条是**上下文词的
#     尾部出现**（`online`/`user`/`table`/`inventory` 落在摘要后段并不表示跑题），
#     同时还**漏掉 9 篇逐条确认过的真污染里的 3 篇**。**两个方向都错。**
#
# ⇒ 因此本批**不把这一节接进保留/丢弃**（`judge()` 一字未改，读数 772 不变）。
#   它做的是一件被反复证明更值钱的事：**把「看不见」变成「看得见、可复核」**。
#   本仓库危险性排序是 **3 > 2 > 1 > 0**，「没测到」比「测了是红的」更危险；
#   而「拿一个没标定的判据去丢数据」正是「没测到」的一种伪装。
#
# ⚠️ 顺序是有判据的，不是装饰：**题名自证必须前置于任何位置判定**。
#   证据来自本批自己的测量脚本 —— 首版按约束词表顺序「撞上第一个就 break」，
#   于是 `Listwise Selection for Text-to-SQL`（**题目就是** text-to-SQL）、
#   `Spider 2.0-AIFunc: Extending Real-World Text-to-SQL`、
#   `How Far Do On-Prem Open LLMs Get on Text-to-SQL?` 三篇**题目即主题**的论文
#   只因为先撞上表里的 `table` 就被标成了「顺带」。selftest 用这三个真实 id 锁死顺序。
T_SELF = "SELF_EVIDENT"              # 命中标题 ⇒ 自证（**前置**）
T_TOPICAL = "TOPICAL_WINDOW"         # 只在摘要，命中落在摘要前半（问题与贡献区）
T_INCIDENTAL = "INCIDENTAL_MENTION"  # 只在摘要，命中**全部**落在后半（评测与结果区）
T_META = "META_ONLY"                 # 只在 comment / journal_ref 命中（元数据，非论文自述）
T_NO_CONSTRAINT = "NO_CONSTRAINT"    # 该域不设约束词 ⇒ 本条对它们无从判定
T_NONE = "NONE_MATCHED"              # 一个约束词都没命中（此时 judge() 会判丢）

#: 摘要的**中点** —— 一个不需要拟合的自然分界：前半陈述问题与贡献，后半陈述
#: 实验设置、评测与结果。⚠️ **它不是从数据里选出来的阈值**，本批也没有用它做
#: 丢弃决定（见上）。如实登记：深度版规则若将来要升格为判据，必须先重新标定。
ABSTRACT_TOPICAL_CUT = 0.5


def constraint_evidence(item: dict, domain: str, class_words: list[str] | None = None) -> dict:
    """该 (论文, 域) 的约束词命中**证据**。只记录，不判定保留/丢弃。

    `class_words`（该域的**主题类**词，见 `topic_class_words()`）可选：
      · 传了 ⇒ 追加类感知读数 `class_hits` / `class_first_frac` / `late_class`；
      · 没传 ⇒ 这三个字段是 **None，不是 False** —— 「没提供类表 ⇒ 本条无从判定」
        与「判定为不晚」是两件事（同 `NO_CONSTRAINT` 的纪律，也是 #23 的形态）。
    **五态 tier 在任何情况下逐字节不变**（这是反向控制：新读数不许动老读数）。
    """
    cons = DOMAIN_CONSTRAINT.get(domain, [])
    ev = {"domain": domain, "tier": T_NO_CONSTRAINT, "words": [],
          "title_words": [], "abstract_words": [], "meta_words": [], "first_frac": None,
          "class_words": None, "class_hits": None, "class_first_frac": None,
          "late_class": None}
    if not cons:
        return ev
    t = (item.get("title") or "").lower()
    a = (item.get("abstract") or "").lower()
    meta = ((item.get("comment") or "") + " . " + (item.get("journal_ref") or "")).lower()

    # ⚠️ 题名自证**必须最先判**：反过来的写法会把「题目即主题」的论文标成顺带
    # （三篇真实 id 的复现见上面的编译期注释与 selftest）。
    t_hits = [c for c in cons if c in t]
    if t_hits:
        ev.update(tier=T_SELF, words=t_hits, title_words=t_hits)
        _add_class(ev, t, a, meta, class_words, title_hit=True)
        return ev
    a_hits = [c for c in cons if c in a]
    if a_hits:
        first = min(a.find(c) for c in a_hits)
        frac = first / max(1, len(a))
        ev.update(tier=(T_TOPICAL if frac < ABSTRACT_TOPICAL_CUT else T_INCIDENTAL),
                  words=a_hits, abstract_words=a_hits, first_frac=round(frac, 4))
        _add_class(ev, t, a, meta, class_words, title_hit=False)
        return ev
    m_hits = [c for c in cons if c in meta]
    if m_hits:
        ev.update(tier=T_META, words=m_hits, meta_words=m_hits)
        _add_class(ev, t, a, meta, class_words, title_hit=False)
        return ev
    ev.update(tier=T_NONE)
    _add_class(ev, t, a, meta, class_words, title_hit=False)
    return ev


def _add_class(ev: dict, t: str, a: str, meta: str,
               class_words: list[str] | None, title_hit: bool) -> None:
    """把**类感知**读数写进 `ev`。它只加字段，**一个已有字段都不动**。

    `late_class` 的定义（逐字与全池实测一致，42/42 已标注）：
      ① 该对**没有**任何约束词命中题名（题名自证优先 —— 否则会与 SELF_EVIDENT 打架）；
      ② 至少一个**类词**命中摘要；
      ③ 所有类词命中里**最浅**的那个落在摘要后半（≥ 50%）。
    ⚠️ 用「类词的最浅位置」而不是「全部词的最浅位置」正是本批的修复点：
    后者会被泛词短路（`table` 在 5% 处救活主题埋在 80% 处的论文，实测 3 篇）。
    """
    if class_words is None:
        return
    ev["class_words"] = list(class_words)
    a_hits = [c for c in class_words if c in a]
    ev["class_hits"] = a_hits
    if title_hit or not a_hits:
        ev["late_class"] = False
        return
    frac = min(a.find(c) for c in a_hits) / max(1, len(a))
    ev["class_first_frac"] = round(frac, 4)
    ev["late_class"] = frac >= ABSTRACT_TOPICAL_CUT


def _blob(item: dict) -> tuple[str, str]:
    """返回 (标题小写, 标题+摘要小写)。标题命中负向词 = 硬丢弃。"""
    t = (item.get("title") or "").lower()
    a = (item.get("abstract") or "").lower()
    c = (item.get("comment") or "").lower()
    j = (item.get("journal_ref") or "").lower()
    return t, f"{t} . {a} . {c} . {j}"


def judge(item: dict, domain: str) -> tuple[bool, str]:
    """判定一篇论文在给定域下是保留还是丢弃。返回 (keep, reason)。

    ⚠️ **PHASE6 P2 起，本函数的判定刻意不读 `constraint_evidence()`。**
    分级（题名自证 / 摘要主题区 / 摘要后段顺带）是一等读数，**不是判据** ——
    理由是它在全池上既不全也不准（62 假阳性 + 漏 3），拿它丢数据就是静默损失。
    这条「不许据此丢弃」由 selftest 的守卫判据 + 变异 M10 常驻守着。
    """
    title, blob = _blob(item)

    # ① 标题里的全局/本域负向词 → 硬丢弃（标题比摘要可信）
    for neg in GLOBAL_NEGATIVE + DOMAIN_NEGATIVE.get(domain, []):
        if neg in title:
            return False, f"标题否定词:{neg}"

    # ② 摘要里的本域负向词 → 丢弃
    for neg in DOMAIN_NEGATIVE.get(domain, []):
        if neg in blob:
            return False, f"摘要否定词:{neg}"

    # ③ 约束词：本域若配置了约束词，必须命中其一
    cons = DOMAIN_CONSTRAINT.get(domain, [])
    if cons and not any(c in blob for c in cons):
        return False, "未命中约束词"

    # ④ 重灾区域额外要求电商强信号（负向词挡不住同词异义）
    if domain in STRICT_ECOMMERCE_DOMAINS:
        strong = ("e-commerce", "ecommerce", "retail", "marketplace", "warehouse",
                  "fulfillment", "inventory", "subscription", "customer", "purchase")
        if not any(s in blob for s in strong):
            return False, "重灾区域缺少电商强信号"

    return True, "ok"


def filter_pool(items: list[dict]) -> dict:
    """对每篇论文，只要**存在任一**命中域判定为保留，就保留该论文。

    ⚠️ **三态，不是两态**（这是本次修复的核心）。旧版写的是：

        domains = [g for g in groups if g in DOMAIN_DIR]
        if not domains:
            kept.append(it)          # 不属于任何已知域，不擅自丢

    这一行把**两件完全不同的事**合并成了一个分支：
      ① 「这篇论文的域标签**我一个都不认识**」 ← **仪器瞎了**
      ② 「这篇论文真的没有域标签」             ← 数据如此
    两者都被**静默放行**，且**没有任何读数**。实测 1046 篇里 141 篇落在①上，
    它们的负向词与约束词一条都没生效，而报告上看起来一切正常。

    「不擅自丢」这个**处置**是对的（数据不能因为仪器瞎了就丢），但
    「不报」是错的 —— 本仓库危险性排序 **3 > 2 > 1 > 0**：
    **「没测到」比「测了是红的」更危险**（红会有人去修，没测到没人知道）。

    返回 dict（不再返回三元组 —— 三元组塞不下「没测到」这个一等输出）：
        kept / reasons / per_domain / judged / unresolved（标签计数）/
        unresolved_ids（论文 id，可复核）/ no_groups_ids /
        tiers（约束词命中的**证据形态**计数）· incidental_ids（顺带提及的论文，
        可复核）· evidence（逐 (论文,域) 的证据，进 `--apply` 产物）
        + P2b：class_table（逐域主题类词与实测精度，**每次现算**，不存盘）·
        late_class_ids（类感知「主题词只在摘要后段」的保留对）·
        no_class_ids（该域**没有**主题类词 ⇒ 这一类对它无从判定）
    """
    kept: list[dict] = []
    reasons: Counter = Counter()
    per_domain: Counter = Counter()
    unresolved: Counter = Counter()
    unresolved_ids: list[str] = []
    no_groups_ids: list[str] = []
    tiers: Counter = Counter()
    incidental_ids: list[dict] = []
    evidence: dict[str, list[dict]] = {}
    late_class_ids: list[dict] = []
    no_class_ids: list[str] = []
    judged = 0

    #: ⚠️ 类表**每次现算**，绝不存盘：把精度写进文件就是把它冻成一个会腐烂的豁免
    #: （#84 的形态）。代价是一次 O(域 × 词 × 池子) 的子串扫描，实测可忽略。
    prec = domain_precision(items)
    # ⚠️ 精度可以是 **None**（该词在池子里一次都没出现 ⇒ 「没测到」）。必须原样带出去、
    # 由报告说「n/a」，**不许在这里 round()** —— 本批的 M12 变异实测就是靠这一条把一个
    # 「阈值被改成负值就会崩」的脆弱点撞出来的：崩溃不是判红，它连「红在哪条判据上」
    # 都答不出来（本仓库已在 `--mutate` 里登记过这种「只让门禁崩掉」的形态）。
    class_table = {d: [{"word": w,
                        "precision": (round(prec[(d, w)]["precision"], 4)
                                      if prec[(d, w)]["precision"] is not None else None),
                        "hits": prec[(d, w)]["hits"]}
                       for w in topic_class_words(d, prec)]
                   for d in DOMAIN_CONSTRAINT}
    class_cache: dict[str, list[str]] = {}

    for it in items:
        groups = it.get("query_groups") or []
        r = _domains.resolve_groups(groups)

        # ① 仪器瞎了：标签一个都认不出来 —— **保留，但必须报**（不是静默通过）
        if not r["resolved"]:
            kept.append(it)
            if groups:
                for g in r["unknown"]:
                    unresolved[g] += 1
                unresolved_ids.append(_item_id(it))
            else:
                no_groups_ids.append(_item_id(it))
            continue

        # ② 正常判定
        judged += 1
        domains = r["resolved"]
        verdicts = [judge(it, d) for d in domains]
        # 证据**独立于判定**采集：只对真正把它留下来的那些域记。
        evs = []
        for d, v in zip(domains, verdicts):
            if not v[0]:
                continue
            evs.append(constraint_evidence(it, d, class_cache.setdefault(
                d, class_table.get(d) and [x["word"] for x in class_table[d]] or [])))
        if any(v[0] for v in verdicts):
            kept.append(it)
            for d, v in zip(domains, verdicts):
                if v[0]:
                    per_domain[d] += 1
            for e in evs:
                tiers[e["tier"]] += 1
                if e["tier"] == T_INCIDENTAL:
                    # ⚠️ 记 (论文, 域) **对**而不是只记 id：一篇论文可属多个域，
                    # 只记 id 时「这一条是哪个域的形态」无法回答，与标注集对账时
                    # 会把别的域的标注算进来（首版实测：68 条被数成「全部已标注」）。
                    incidental_ids.append({"arxiv_id": _item_id(it), "domain": e["domain"]})
                # ⚠️ 「该域没有主题类词」与「类词命中不晚」**分开计数** ——
                # 02-A_B实验 实测一个类词都没有（全是泛词），把这批算进
                # 「不晚」会让整域静默消失（本仓库危险性排序 3 > 2 > 1 > 0）。
                if e["class_words"] == []:
                    no_class_ids.append(_item_id(it))
                if e["late_class"]:
                    late_class_ids.append({"arxiv_id": _item_id(it), "domain": e["domain"],
                                           "class_hits": e["class_hits"],
                                           "first_frac": e["class_first_frac"]})
            if evs:
                evidence[_item_id(it)] = evs
        else:
            # ⚠️ 已登记（台账 #100，本批不改）：这里只记**首个域**的原因。
            # 一篇论文同时属于 3 个域而都被拒时，另两条原因不计入直方图。
            # 保留/丢弃的**结论**不受影响，受影响的只有「为什么」的归属。
            reasons[verdicts[0][1]] += 1

    return {
        "kept": kept, "reasons": reasons, "per_domain": per_domain,
        "judged": judged,
        "unresolved": unresolved, "unresolved_ids": unresolved_ids,
        "no_groups_ids": no_groups_ids,
        "tiers": tiers, "incidental_ids": incidental_ids, "evidence": evidence,
        "class_table": class_table, "late_class_ids": late_class_ids,
        "no_class_ids": no_class_ids,
    }


def _item_id(it: dict) -> str:
    return it.get("arxiv_id") or it.get("id") or it.get("url") or "<无 id>"


def selftest() -> int:
    """证明过滤器真的能拦下 v2 记录的两类真实污染，并证明**「没测到」会报**。"""
    ok = True

    # ⚠️ 第一条判据就是「域表覆盖了全部规范域」。
    # 少了它，M3（把 00-电商Agent 从域表里拿掉）**不会让 selftest 变红** ——
    # 实测就是这样：变异已生效、selftest 照样全绿，因为下面那些 `judge()` 用例
    # 是**直接点名域**的，根本不经过 `DOMAIN_DIR`。
    # 而 00 域不在域表里**正是本次修复的那个缺陷**（37 篇被静默放行）
    # ⇒ 这条判据必须有，否则修好的东西可以原样长回来。
    print("=== 判据：域表覆盖 ===")
    missing = [d for d in _domains.CANONICAL if d not in DOMAIN_DIR]
    good = not missing
    ok &= good
    print(f"  {'✅' if good else '❌'} 域表必须覆盖全部 {len(_domains.CANONICAL)} 个规范域"
          f"（域表 {len(DOMAIN_DIR)} 个）")
    if missing:
        print(f"       ↘ 缺失：{missing} —— 这些域的论文会被**静默整篇放行**")
    orphan = sorted(set(DOMAIN_DIR) - set(_domains.CANONICAL))
    good2 = not orphan
    ok &= good2
    print(f"  {'✅' if good2 else '❌'} 域表里不许有非规范域 → {orphan}")
    # 反向控制：词表不许对**规范域之外**的键配词（配了就是又长出一份事实源）
    extra = sorted((set(DOMAIN_NEGATIVE) | set(DOMAIN_CONSTRAINT)) - set(_domains.CANONICAL))
    good3 = not extra
    print(f"  {'✅' if good3 else '⚠️'} 负向/约束词表里的键必须都是规范域 → {extra}"
          f"{'' if good3 else '（`10-MAS`/`11-AI人文` 只在约束词表里缺项是允许的，'
                                '但键名本身必须是规范域）'}")

    cases = [
        # (域, 构造的论文, 期望保留?)
        ("04-供应链", {"title": "A Survey of Software Supply Chain Security",
                       "abstract": "we study sbom and package dependency vulnerability in npm"},
         False),
        ("04-供应链", {"title": "Multi-Echelon Inventory Optimization for Perishable Goods",
                       "abstract": "we optimize warehouse replenishment and fulfillment for retail goods"},
         True),
        ("14-用户分析", {"title": "Cohort Study of MRI Lesion Progression",
                       "abstract": "clinical patient cohort study with eeg and radiotherapy"},
         False),
        ("14-用户分析", {"title": "Cohort Retention Analysis for Subscription Commerce",
                       "abstract": "we analyze customer retention and repeat purchase in e-commerce"},
         True),
        # 同词异义：trial 在 02 域是方法论词，不得被误杀
        ("02-A_B实验", {"title": "Sequential Testing for Online Controlled Trials",
                        "abstract": "we design always-valid inference for platform pricing experiments"},
         True),
        # 约束词缺失 → 丢弃（气象预测冒充需求预测）
        ("03-时间序列", {"title": "Transformer Forecasting of Wind Power Generation",
                       "abstract": "we improve weather forecasting accuracy"},
         False),
        # 00 域首次进入词表：域外化学论文必须被拦（词来自该域池子里实测点名的论文）
        ("00-电商Agent", {"title": "Reaction-network reasoning for catalyst selectivity",
                          "abstract": "we discover novel catalyst architectures in chemistry"},
         False),
        ("00-电商Agent", {"title": "Agentic Shopping with Cross-border Catalog Enrichment",
                          "abstract": "we build a shopping agent over an e-commerce product catalog"},
         True),
        # 07/09/15 的约束词必须认**自己收割查询的定义性短语**（否则自己捞进来又扔出去）
        ("09-DataAgent-LLM", {"title": "AttnLink: Turning Attention into Schema Links for Text-to-SQL",
                              "abstract": "schema linking is a critical component of text-to-sql systems"},
         True),
        ("07-NLP-VOC", {"title": "Cross-lingual Aspect-Based Sentiment Analysis",
                        "abstract": "we transfer knowledge for aspect-based sentiment analysis"},
         True),
        ("15-营销投放分析", {"title": "Structural Estimation of Marketing Mix Model Parameters",
                            "abstract": "marketing mix models are widely used for budget allocation"},
         True),
        # ⚠️ 反向控制：补了定义性短语**不等于**这条域失去拦污能力 ——
        # 同域里一篇既无本域定义性短语、也无电商信号的论文，必须照旧被拦。
        ("09-DataAgent-LLM", {"title": "Non-vacuous Generalization Bounds for Reinforcement Learning",
                              "abstract": "we prove new bounds for verifiable-reward reinforcement learning"},
         False),
        ("07-NLP-VOC", {"title": "Molecular Docking with Graph Neural Networks",
                        "abstract": "we predict protein-ligand binding affinity for drug design"},
         False),
        ("15-营销投放分析", {"title": "A Randomized Trial of Health Promotion in Schools",
                            "abstract": "we evaluate a public health intervention programme"},
         False),
    ]
    for domain, item, expect in cases:
        got, why = judge(item, domain)
        flag = "✅" if got == expect else "❌"
        if got != expect:
            ok = False
        print(f"  {flag} [{domain}] keep={got} (期望 {expect})  {why}  ← {item['title'][:52]}")

    print("\n=== 判据：三态 ——「仪器瞎了」必须与「数据本来没有」分开，且都要报 ===")
    tri = [
        ("① 认识（规范名）",
         [{"arxiv_id": "a1", "title": "Cohort Retention in Subscription Commerce",
           "abstract": "we analyze customer retention and repeat purchase in e-commerce",
           "query_groups": ["14-用户分析"]}],
         {"judged": 1, "unresolved": 0, "no_groups": 0, "kept": 1}),
        # ⚠️ **这一条是本批改过口径的**：旧名以前会被静默折到规范名（那一版
        # 正是 141 篇被放行的成因之一），现在**必须走「仪器瞎了」**。
        # 理由：替漂移兜底 = 取消发现漂移的能力。折换只许在
        # `migrate_domain_labels.py --apply` 里显式做一次。
        ("② 退休名（`09-DataAgent`）⇒ **认不出**，不许静默折（那 141 篇的形态）",
         [{"arxiv_id": "a2", "title": "AttnLink: Schema Links for Text-to-SQL",
           "abstract": "schema linking for text-to-sql systems",
           "query_groups": ["09-DataAgent"]}],
         {"judged": 0, "unresolved": 1, "no_groups": 0, "kept": 1}),
        ("③ 仪器瞎了：标签认不出来 ⇒ 保留但要报（旧版这里静默通过）",
         [{"arxiv_id": "a3", "title": "Some Paper",
           "abstract": "whatever", "query_groups": ["09-DataAgentX"]}],
         {"judged": 0, "unresolved": 1, "no_groups": 0, "kept": 1}),
        ("④ 真的没有域标签 ⇒ 与③是**两件事**，分开计数",
         [{"arxiv_id": "a4", "title": "Some Paper", "abstract": "whatever"}],
         {"judged": 0, "unresolved": 0, "no_groups": 1, "kept": 1}),
        ("⑤ 同一个规范名出现两次 ⇒ 折成一条，不许判两次、也不许计数两次",
         [{"arxiv_id": "a5", "title": "AttnLink: Schema Links for Text-to-SQL",
           "abstract": "schema linking for text-to-sql systems",
           "query_groups": ["09-DataAgent-LLM", "09-DataAgent-LLM"]}],
         {"judged": 1, "unresolved": 0, "no_groups": 0, "kept": 1}),
    ]
    for desc, pool, expect in tri:
        r = filter_pool(pool)
        got = {"judged": r["judged"],
               "unresolved": sum(r["unresolved"].values()),
               "no_groups": len(r["no_groups_ids"]),
               "kept": len(r["kept"])}
        good = got == expect
        ok &= good
        print(f"  {'✅' if good else '❌'} {desc}")
        print(f"       judged={got['judged']} unresolved={got['unresolved']} "
              f"no_groups={got['no_groups']} kept={got['kept']}"
              f"{'' if good else f'（期望 {expect}）'}")
    # ③ 的反面：仪器瞎了**不等于**可以把它当通过 —— 标签名必须被点名报出来
    r3 = filter_pool([{"arxiv_id": "a3", "title": "x", "abstract": "y",
                       "query_groups": ["09-DataAgentX", "99-不存在的域"]}])
    good = set(r3["unresolved"]) == {"09-DataAgentX", "99-不存在的域"} and r3["unresolved_ids"] == ["a3"]
    ok &= good
    print(f"  {'✅' if good else '❌'} 反向控制：认不出的标签必须**逐个点名**、"
          f"论文 id 必须可复核 → {dict(r3['unresolved'])} {r3['unresolved_ids']}")
    # 退休名的报错必须**可行动**：规范名查得到
    r_ret = filter_pool([{"arxiv_id": "a2", "title": "x", "abstract": "y",
                          "query_groups": ["09-DataAgent"]}])
    hints = {g: _domains.RETIRED.get(g) for g in r_ret["unresolved"]}
    good = hints == {"09-DataAgent": "09-DataAgent-LLM"}
    ok &= good
    print(f"  {'✅' if good else '❌'} 退休名的报错必须**可行动**（查得到规范名）→ {hints}")

    # 反向控制：正常池子上「仪器瞎了」必须恰好为 0（否则上面那几条是恒真的）
    r_clean = filter_pool([{"arxiv_id": "c", "title": "Cohort Retention in Subscription Commerce",
                            "abstract": "customer retention in e-commerce",
                            "query_groups": ["14-用户分析"]}])
    good = not r_clean["unresolved"] and not r_clean["no_groups_ids"]
    ok &= good
    print(f"  {'✅' if good else '❌'} 反向控制：标签正常时「没测到」必须为 0 → "
          f"unresolved={dict(r_clean['unresolved'])} no_groups={r_clean['no_groups_ids']}")

    print("\n=== 判据：命中域与负向词的分歧矩阵仍然活着（负向词按域生效）===")
    r = filter_pool([{"arxiv_id": "d", "title": "Sequential Testing for Online Controlled Trials",
                      "abstract": "platform pricing experiments", "query_groups": ["02-A_B实验"]},
                     {"arxiv_id": "e", "title": "A Clinical Trial of Cohort Therapy",
                      "abstract": "clinical patient cohort study", "query_groups": ["14-用户分析"]}])
    good = (len(r["kept"]) == 1 and r["kept"][0]["arxiv_id"] == "d")
    ok &= good
    print(f"  {'✅' if good else '❌'} 同一个词（trial/cohort）在 02 是方法论词、在 14 是医学词 → "
          f"保留 {[x['arxiv_id'] for x in r['kept']]}")

    print("\n=== 判据：约束词命中的证据形态（PHASE6 P2；**读数，不判定**）===")
    # ⚠️ 判据 ① 是这一节存在的**主要理由**，不是顺手加的：
    # 本批的测量脚本首版按约束词表顺序「撞上第一个就 break」，于是
    # `2609.00834`（Listwise Selection for Text-to-SQL）、`2607.06229`（Spider 2.0-AIFunc:
    # Extending Real-World Text-to-SQL）、`2606.29733`（How Far Do On-Prem Open LLMs Get
    # on Text-to-SQL?）三篇**题目即主题**的论文，只因先撞上表里的 `table` 就被标成「顺带」。
    # 夹具复刻的正是那个形态（题名含 text-to-sql，而 table 只出现在摘要后段）。
    _filler = "a" * 60
    shape = [
        ("① 题名命中 ⇒ SELF_EVIDENT，且必须**压过**摘要后段的顺带命中"
         "（题名自证必须先于位置判定）",
         {"title": "Listwise Selection for Text-to-SQL",
          "abstract": _filler + " we report on the table benchmark"},
         "09-DataAgent-LLM", T_SELF),
        ("② 只在摘要、命中落在前半 ⇒ TOPICAL_WINDOW",
         {"title": "Simple and Effective Pipeline",
          "abstract": "we build table pipelines and then " + _filler},
         "09-DataAgent-LLM", T_TOPICAL),
        ("③ 只在摘要、命中全落在后半 ⇒ INCIDENTAL_MENTION",
         {"title": "A Study of Something Else Entirely",
          "abstract": _filler + " we report on the table benchmark"},
         "09-DataAgent-LLM", T_INCIDENTAL),
        ("④ 只在 comment / journal_ref 命中 ⇒ META_ONLY（元数据不是论文自述）",
         {"title": "A Study of Something Else Entirely",
          "abstract": "nothing relevant here at all", "comment": "accepted, table track"},
         "09-DataAgent-LLM", T_META),
        ("⑤ 该域不设约束词 ⇒ NO_CONSTRAINT（**不是**「没命中」）",
         {"title": "Anything", "abstract": "whatever"}, "10-MAS", T_NO_CONSTRAINT),
        ("⑥ 一个约束词都没命中 ⇒ NONE_MATCHED",
         {"title": "Anything", "abstract": "whatever"}, "09-DataAgent-LLM", T_NONE),
    ]
    for desc, item, dom, expect in shape:
        got = constraint_evidence(item, dom)["tier"]
        good = got == expect
        ok &= good
        print(f"  {'✅' if good else '❌'} {desc} → {got}"
              f"{'' if good else f'（期望 {expect}）'}")

    print("\n=== 判据：「顺带提及」**不许**改变保留/丢弃（守卫，防它被悄悄接进判据）===")
    # 这一段守的是一条**刻意的不作为**。本仓库 #10 的教训是「新增豁免必须同时写清
    # 什么情况下不许豁免」；反过来这里要写清的是：**新增「只标不判」，必须同时
    # 写清什么情况下不许不判** —— 否则下一个人会顺手把它接进丢弃路径，
    # 而按本批读数那会把 62 条真命中陆续变成静默损失。
    inc = {"arxiv_id": "inc1", "title": "A Study of Something Else Entirely",
           "abstract": _filler + " we report on the table benchmark",
           "query_groups": ["09-DataAgent-LLM"]}
    keep, why = judge(inc, "09-DataAgent-LLM")
    good = keep
    ok &= good
    print(f"  {'✅' if good else '❌'} 一篇「顺带提及」的论文**必须仍被保留**"
          f"（judge 不读证据形态）→ keep={keep} {why}")
    # 反面：守卫不许恒真 —— 同一形态的论文若另犯了别的规则，**照样要被丢**。
    inc_bad = {**inc, "abstract": _filler + " we report on the gui agent table"}
    keep_bad, why_bad = judge(inc_bad, "09-DataAgent-LLM")
    good = (not keep_bad) and "否定词" in why_bad
    ok &= good
    print(f"  {'✅' if good else '❌'} 反向控制：同形态但**另犯规则**的必须照样被丢"
          f"（证明上面那条不是恒真）→ keep={keep_bad} {why_bad}")

    print("\n=== 判据：真实事件回归 —— 题名自证优先，用的是真论文 id ===")
    # 夹具能证明「顺序错了会红」，但证明不了「真语料上确实是那三篇」。
    # 这里读校准集里点名的三条，要求它们当前**仍是** SELF_EVIDENT。
    try:
        lab = json.loads((DATA / "constraint-salience-labels.json").read_text(encoding="utf-8"))
        blk = lab.get("title_priority_regression") or {}
        reg = blk.get("ids") or []
        if blk.get("must_be") != T_SELF:
            ok = False
            print(f"  ❌ 回归块的 `must_be` 必须是 {T_SELF!r}，实际 {blk.get('must_be')!r}")
        if not reg:
            ok = False
            print("  ❌ 回归块里一个 id 都没有 —— 空壳回归等于没有回归")
        pool_items = json.loads((DATA / "arxiv_candidates.json").read_text(encoding="utf-8"))["items"]
        by_id = {_item_id(i): i for i in pool_items}
        for aid in reg:
            it = by_id.get(aid)
            if it is None:
                ok = False
                print(f"  ❌ 回归用例 {aid} 已不在候选池里 ⇒ 该用例失效，必须重挑")
                continue
            dom = _domains.resolve_groups(it.get("query_groups") or [])["resolved"]
            got = [constraint_evidence(it, d)["tier"] for d in dom]
            good = dom and all(g == T_SELF for g in got)
            ok &= good
            print(f"  {'✅' if good else '❌'} {aid} 必须是 SELF_EVIDENT → {got}"
                  f"  {it['title'][:56]}")
    except FileNotFoundError as exc:
        ok = False
        print(f"  ❌ 回归用例读不到：{exc} —— 「没东西可查」不等于「查过了没问题」")

    print("\n=== 判据：主题类词表（实测域精度，P2b；**读数，不判定**）===")
    # ① 三个**语法**拆表判据的反例必须被这条**实测**判据挡住：`table` 出自 09
    #    **自己的**收割查询（`abs:"table" AND abs:"agent" AND abs:"reasoning"`），
    #    语法上「在引号里、是收割词」，但实测精度只有 0.195 ⇒ **不许进类表**。
    #    这条是「把表拆开」这件事的核心断言，也是 M11/M12 两个变异的目标。
    prec = None
    try:
        pool_items = json.loads((DATA / "arxiv_candidates.json").read_text(encoding="utf-8"))["items"]
        prec = domain_precision(pool_items)
        c09 = topic_class_words("09-DataAgent-LLM", prec)
        for w, want in (("text-to-sql", True), ("table", False), ("agent", False)):
            got = w in c09
            good = got == want
            ok &= good
            p = prec.get(("09-DataAgent-LLM", w), {}).get("precision")
            print(f"  {'✅' if good else '❌'} 09 域类表{'必须' if want else '**不许**'}含 {w!r}"
                  f"（实测域精度 {p}）→ {'在' if got else '不在'}")
        good = bool(c09)
        ok &= good
        print(f"  {'✅' if good else '❌'} 09 域类表非空（空表会让整条读数静默变成 0）→ {c09}")
        # ② 「该域没有类词」必须**被报出来**，不能静默为空：02-A_B实验 整表泛词，
        #    实测最高精度仍不到阈值 —— 那不是「这个域很干净」，是「判据看不见」。
        c02 = topic_class_words("02-A_B实验", prec)
        best02 = max((prec[("02-A_B实验", w)]["precision"] or 0)
                     for w in DOMAIN_CONSTRAINT["02-A_B实验"])
        good = (c02 == []) and best02 < TOPIC_CLASS_MIN_PRECISION
        ok &= good
        print(f"  {'✅' if good else '❌'} 02 域整表泛词 ⇒ 类表为空（最高精度 {best02:.3f}）；"
              f"它必须落进 `no_class`，不许被当成「不晚」")
    except FileNotFoundError as exc:
        ok = False
        print(f"  ❌ 候选池读不到：{exc} —— 「没东西可查」不等于「查过了没问题」")

    print("\n=== 判据：类感知位置规则 —— 被泛词救活的三篇真污染必须重新可见 ===")
    # 这是 P2b 的**存在理由**。P2 实测：`Constrained Decoding for Diffusion LMs` /
    # `TTHE` / `EvolveNet` 三篇的主题短语 `text-to-sql` 埋在摘要 75%/78%/82%，
    # **却被同一个表里的泛词 `table`（5%/6%/31%）救回**，于是全词版规则漏掉它们。
    # 三条都是真 id、真语料 —— 夹具证明不了「真语料上就是它们」。
    c09 = topic_class_words("09-DataAgent-LLM", prec) if prec is not None else ["text-to-sql"]
    try:
        pool_items = json.loads((DATA / "arxiv_candidates.json").read_text(encoding="utf-8"))["items"]
        by_id = {_item_id(i): i for i in pool_items}
        for aid in ("2607.07026", "2607.08124", "2608.04968"):
            it = by_id.get(aid)
            if it is None:
                ok = False
                print(f"  ❌ 回归用例 {aid} 已不在候选池里 ⇒ 该用例失效，必须重挑")
                continue
            ev = constraint_evidence(it, "09-DataAgent-LLM", c09)
            good = ev["late_class"] is True and ev["tier"] == T_TOPICAL
            ok &= good
            print(f"  {'✅' if good else '❌'} {aid}：全词版判 {ev['tier']}（泛词抢跑），"
                  f"类感知 late_class={ev['late_class']} @{ev['class_first_frac']}"
                  f"  {it['title'][:42]}")
        early = by_id.get("2606.16878")
        if early is None:
            ok = False
            print("  ❌ 反向控制用例 2606.16878 不在池子里 ⇒ 该用例失效")
        else:
            ev = constraint_evidence(early, "15-营销投放分析",
                                     topic_class_words("15-营销投放分析", prec))
            good = (ev["late_class"] is False
                    and ev["class_first_frac"] is not None
                    and ev["class_first_frac"] < ABSTRACT_TOPICAL_CUT)
            ok &= good
            print(f"  {'✅' if good else '❌'} 反向控制：类词命中在摘要**前半**的真命中"
                  f"不许被标 late → late_class={ev['late_class']}"
                  f" @{ev['class_first_frac']}  {early['title'][:38]}")
    except FileNotFoundError as exc:
        ok = False
        print(f"  ❌ 回归用例读不到：{exc}")

    print("\n=== 判据：类感知读数**不许**改变保留/丢弃（守卫，同「顺带提及」那条）===")
    late = {"arxiv_id": "late1", "title": "A Study of Something Else Entirely",
            "abstract": _filler + " we report on the text-to-sql benchmark",
            "query_groups": ["09-DataAgent-LLM"]}
    ev = constraint_evidence(late, "09-DataAgent-LLM", c09)
    keep_late = judge(late, "09-DataAgent-LLM")[0]
    good = ev["late_class"] is True and keep_late
    ok &= good
    print(f"  {'✅' if good else '❌'} 一条 `late_class=True` 的论文**必须仍被保留**"
          f"（judge 不读类感知字段）→ late_class={ev['late_class']} keep={keep_late}")
    # 反向控制：同形态但另犯规则的，照样要丢 —— 证明上面那条不是恒真。
    late_bad = {**late, "abstract": _filler + " we report on the gui agent text-to-sql benchmark"}
    keep_bad, why_bad = judge(late_bad, "09-DataAgent-LLM")
    good = (not keep_bad) and "否定词" in why_bad
    ok &= good
    print(f"  {'✅' if good else '❌'} 反向控制：同形态但**另犯规则**的必须照样被丢 → "
          f"keep={keep_bad} {why_bad}")
    # ⚠️「没给类表」≠「判为不晚」：两者都返回 False 时，缺表会被读成干净。
    ev_none = constraint_evidence(late, "09-DataAgent-LLM")
    good = ev_none["late_class"] is None and ev_none["class_words"] is None
    ok &= good
    print(f"  {'✅' if good else '❌'} 不提供类表时 `late_class` 必须是 **None 而不是 False**"
          f"（「问不到」≠「不晚」，同 NO_CONSTRAINT 的纪律）→ {ev_none['late_class']!r}")

    print("✅ 自检通过：负向词/约束词/三态判定均按预期生效" if ok
          else "❌ 自检失败：过滤逻辑与关键词库 v2 的约定不符")
    return 0 if ok else 1



# ---------------------------------------------------------------------------
# 值变异：**改源文件 → 另起子进程跑真 CLI**，验证每条判据都会失败
# ---------------------------------------------------------------------------
# 为什么不是「在同一个进程里 patch 全局量」：本仓库记过两个坑 ——
#   ① 变异锚点文本在文件里出现两次 ⇒ 变异**没施上力**，却照样报「抓住了」；
#   ② 导入期缓存（`domains.CANONICAL_OF`）⇒ 常量改了而判据读的不是它。
# 这里因此做了三件事：
#   · 锚点在**变异表之前的源码正文**里必须**恰好出现一次**（本批实测：6/7 条首版
#     锚点因为被变异表自己抄了一遍而出现 2–3 次 —— 正是坑①，已由这条守卫抓住）；
#   · 改完另存临时目录（连同 `domains.py` 一起复制），以子进程跑 `--selftest`；
#   · 子进程必须 exit≠0，且**未变异时**必须 exit=0（反向控制放在最前面）。
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    ("M1 `filter_pool` 的①分支退回静默 keep（旧版行为：认不出就不报）",
     "candidate_filter.py",
     """            if groups:
                for g in r["unknown"]:
                    unresolved[g] += 1
                unresolved_ids.append(_item_id(it))
            else:
                no_groups_ids.append(_item_id(it))""",
     """            if groups:
                pass
            else:
                no_groups_ids.append(_item_id(it))""",
     "② 退休名"),
    ("M2 把「仪器瞎了」与「真的没有标签」合并成一件事（退回两态）",
     "candidate_filter.py",
     """            if groups:
                for g in r["unknown"]:
                    unresolved[g] += 1
                unresolved_ids.append(_item_id(it))
            else:
                no_groups_ids.append(_item_id(it))""",
     """            no_groups_ids.append(_item_id(it))""",
     "② 退休名"),
    ("M3 域名表退回旧口径（不含 00-电商Agent —— 那一域会被静默放行）",
     "candidate_filter.py",
     'DOMAIN_DIR = {d: d for d in _domains.CANONICAL}',
     'DOMAIN_DIR = {d: d for d in _domains.CANONICAL if d != "00-电商Agent"}',
     '域表必须覆盖全部 17 个规范域'),
    ("M4 抹掉 09 域的 `text-to-sql`（它自己捞进来的论文又被它扔掉）",
     "candidate_filter.py",
     '"text-to-sql", "data analysis agent", "autonomous data science"]',
     '"data analysis agent", "autonomous data science"]',
     '[09-DataAgent-LLM]'),
    ("M5 抹掉 15 域的 `marketing mix`（MMM 论文被自己的域扔掉）",
     "candidate_filter.py",
     '"marketing mix", "media mix"]',
     '"media mix"]',
     '[15-营销投放分析]'),
    ("M6 抹掉 07 域的 `aspect-based sentiment`（ABSA 论文被自己的域扔掉）",
     "candidate_filter.py",
     '"aspect-based sentiment", "opinion mining"]',
     '"opinion mining"]',
     '[07-NLP-VOC]'),
    ("M7 把旧名塞回解析路径（`resolve()` 重新认它）⇒ 自检必须报",
     "domains.py",
     '''    if label in canonical_set():
        return label, "canonical"
    return None, "unknown"''',
     '''    if label in canonical_set():
        return label, "canonical"
    if label in RETIRED:
        return RETIRED[label], "canonical"
    return None, "unknown"''',
     '② 退休名'),
    ("M8 退休名从「认不出」改成「猜一个规范名」⇒ 自检必须报"
     "（守的是「不许替漂移兜底」）",
     "domains.py",
     '''    if label in canonical_set():
        return label, "canonical"
    return None, "unknown"''',
     '''    if label in canonical_set():
        return label, "canonical"
    for _c in CANONICAL:
        if label and label[:2] == _c[:2]:
            return _c, "canonical"
    return None, "unknown"''',
     '② 退休名'),
    # --- PHASE6 P2：证据形态与守卫 ---
    ("M9 取消「题名自证前置」⇒ 题目即主题的论文会被摘要后段的泛词抢走 ⇒ 自检必须报",
     "candidate_filter.py",
     """    t_hits = [c for c in cons if c in t]
    if t_hits:""",
     """    t_hits = []
    if t_hits:""",
     "题名自证必须先于位置判定"),
    ("M10 把「顺带提及」接进丢弃路径 —— 本批**刻意不做**的那个动作，做了必须当场报",
     "candidate_filter.py",
     """    return True, "ok\"""",
     """    if constraint_evidence(item, domain)["tier"] == T_INCIDENTAL:
        return False, "顺带提及"
    return True, "ok\"""",
     "必须仍被保留"),
    # --- PHASE6 P2b：把表拆开 ---
    ("M11 类感知位置退回**全部约束词**（= P2b 修掉的那个缺陷本身）⇒ "
     "被泛词救活的 3 篇真污染必须重新消失 ⇒ 自检必须报",
     "candidate_filter.py",
     '    a_hits = [c for c in class_words if c in a]',
     '    a_hits = [c for c in DOMAIN_CONSTRAINT.get(ev["domain"], []) if c in a]',
     "泛词抢跑"),
    ("M12 类表改用**语法**判据（词长 ≥5）⇒ 09 自己的收割词 `table` 会被当成主题词 ⇒ "
     "自检必须报（这是本批逐条否决的三个语法判据之一）",
     "candidate_filter.py",
     "    return [w for p, w in sorted(got, key=lambda pw: (-pw[0], pw[1])) if p > thr]",
     "    return [w for p, w in sorted(got, key=lambda pw: (-pw[0], pw[1])) if len(w) >= 5]",
     "不许**含 'table'"),
]


def mutate() -> int:
    import subprocess
    import tempfile

    here = Path(__file__).resolve().parent
    srcs = {n: (here / n).read_text(encoding="utf-8")
            for n in ("candidate_filter.py", "domains.py")}
    ok = True
    print("=== 变异：改源文件 → 子进程跑真 CLI ===")

    # 反向控制**先跑**：未变异时 --selftest 必须 exit 0。
    # 不然「变异后会红」这句话没有意义（本来就红）。
    base = subprocess.run([sys.executable, str(here / "candidate_filter.py"), "--selftest"],
                          capture_output=True, text=True)
    good = base.returncode == 0
    ok &= good
    print(f"  {'✅' if good else '❌'} 反向控制：未变异时 `--selftest` 必须 exit 0 "
          f"（实测 exit={base.returncode}）")

    n_armed = n_fired = n_right = 0
    for desc, fname, old, new, expect in MUTATIONS:
        # ⚠️ 只在**变异表之前**的正文里数锚点：否则变异表会抄自己一遍，
        # 锚点计数变成 2–3，守卫反而变成误报源（首版就是这个形态）。
        body = srcs[fname].split("# 值变异：")[0]
        n_here = body.count(old)
        armed = n_here == 1
        n_armed += armed
        print(f"  {'✅' if armed else '❌'} {desc}")
        if not armed:
            print(f"       ↘ 锚点在 {fname} 正文里出现 {n_here} 次（必须恰好 1 次）"
                  f" ⇒ **没施上力**，下面的结果不算数")
            ok = False
            continue
        # ⚠️ 变异体放**临时目录**，但把仓库根用 `P2S_REPO` 钉住。
        # 两个失败版本都别再走一遍：
        #   ① 放 /tmp 且不钉仓库根 ⇒ 门禁**因为路径而不是判据**变色（domains 那边 8/8 假绿）；
        #   ② 放进本目录再跑**原文件名** ⇒ 跑的根本不是变异体（本次实测：8/8 全绿）。
        with tempfile.TemporaryDirectory() as td:
            for n, s in srcs.items():
                (Path(td) / n).write_text(s, encoding="utf-8")
            mut = Path(td) / fname
            mut.write_text(srcs[fname].replace(old, new, 1), encoding="utf-8")
            env = {**os.environ, "P2S_REPO": str(here.parents[1])}
            r = subprocess.run([sys.executable, str(Path(td) / "candidate_filter.py"),
                                "--selftest"], capture_output=True, text=True, env=env)
        fired = r.returncode != 0
        n_fired += fired
        first = next((ln.strip() for ln in r.stdout.splitlines()
                      if ln.strip().startswith("❌")), "")
        right = fired and expect in first
        n_right += right
        ok &= right
        print(f"       已生效于探针：是 · 变异体 exit={r.returncode} "
              f"{'（判红 ✅）' if fired else '（**仍然全绿 ⇒ 判据是摆设** ❌）'}")
        if fired and not right:
            print(f"       ↘ ⚠️ **红得不是地方**：首行 ❌ 里没有 {expect!r} ⇒ "
                  f"这个变异验证的不是那条判据")
        if first:
            print(f"       ↘ {first[:112]}")

    print(f"\n{'✅' if ok else '❌'} 变异 {n_fired}/{len(MUTATIONS)} 打红 · "
          f"{n_right}/{len(MUTATIONS)} **红在正确的判据上** · "
          f"{n_armed}/{len(MUTATIONS)} 已生效于探针（另 1 条反向控制）")
    return 0 if ok else 1


#: **冻结读数**（2026-09-14，P1 修复后现跑）。任何一项变了都要问「是池子变了、
#: 是词表变了，还是判据变了」—— 三者结论完全不同，所以**逐域分开对账**，
#: 不给一个总分（本仓库纪律：退出码不许合并成一个分数）。
BASELINE = {
    "count_before": 1046,
    "count": 772,
    "judged": 1046,
    "per_domain": {
        "00-电商Agent": 42, "01-因果推断": 5, "02-A_B实验": 63, "03-时间序列": 22,
        "04-供应链": 17, "05-推荐系统": 108, "06-增长模型": 4, "07-NLP-VOC": 9,
        "08-知识图谱": 60, "09-DataAgent-LLM": 91, "10-MAS": 75, "11-AI人文": 21,
        "12-ML基础": 67, "13-广告分析": 27, "14-用户分析": 3,
        "15-营销投放分析": 8, "16-智能体工程": 187,
    },
    # 修复前：判定 905 / 保留 776 / **静默放行 141**。
    # 修复后：判定 1046 / 保留 772 / 静默放行 0（−4 净额 = 新丢 6 − 救回 2）。
    "before_fix": {"judged": 905, "count": 776, "silently_passed": 141,
                   "newly_judged": 141, "newly_dropped": 6, "newly_rescued": 2},
    #: 约束词命中的**证据形态**（PHASE6 P2）。⚠️ 这一栏**不参与判定** ——
    #: 它进基线只为了一件事：让「形态分布变了」这个事实在 `--check` 里可见。
    #: 降为丢弃必须先过标定（见 `constraint_evidence()` 的编译期注释）。
    "tiers": {"SELF_EVIDENT": 386, "TOPICAL_WINDOW": 258, "NO_CONSTRAINT": 96,
              "INCIDENTAL_MENTION": 68, "META_ONLY": 1},
    "incidental_items": 68,
    #: P2b：**类感知**读数（同样不进判定）。`late_class_items` 是「主题类词只在
    #: 摘要后段」的保留对；`no_class_items` 是**该域一个主题类词都没有**的保留对
    #: —— 后者必须单独记，否则「7 个域判据看不见」会被读成「这 7 个域很干净」。
    "late_class_items": 42,
    "no_class_items": 367,
    "class_domains": 8,
}


SPEC_DOC = REPO / "paper2skills-vault" / "07-资源库" / "关键词库-v2.md"
#: 过滤产物。**判据 J3 对它的存在性双向判定** —— 见 `check_spec()`。
FILTERED_POOL = "arxiv_candidates_filtered.json"

#: 「读文件」的函数名。**只认这几个** —— 判据问的是「谁真的把它读进来」，
#: 不是在文件里找不到这几个字。加名字要连带加用例，别凭印象扩表。
_READ_FUNCS = ("open", "read_text", "read_bytes", "load", "loads")


def _reads_file(path: Path, needle: str) -> bool:
    """该脚本里是否**真的**把 `needle` 这个字符串传进了某个读函数。

    ⚠️ 用 AST 而不是文本邻近：首版是「`needle` 附近 ±400 字符里有没有 `read_text`」，
    于是 `run_phase6_gates.py` 里**那段构造假读者的字符串**被当成了真读者（假红）。
    邻近启发式问的是「附近像不像在读」，AST 问的是「**是不是在读**」。
    """
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if name not in _READ_FUNCS:
            continue
        # 搜整个调用子树：`json.loads(Path('x').read_text(...))` 里 needle 在**接收者**上，
        # 只搜 args 会漏（而漏 = 假绿，比假红危险）。
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and needle in sub.value:
                return True
    return False


def check_spec(spec_path: Path | None = None) -> int:
    """规格文档与代码**不得各说各话** —— 逐条验证本文档点名的接线点。

    退出码 **0 全过 / 1 判红 / 2 规格文档读不到 / 3 内部错误**。

    为什么要有这一条：`关键词库-v2.md` §0 一度写着「过滤阶段：`arxiv_harvest.py` 的
    `apply_negative_filter()` / `require_constraint()`」—— **这两个函数在全仓库里
    只出现一次，就是那一行**。它读起来像在引用现成代码，实际在描述一件从未存在过的事；
    而三个门禁（L18a–L18f）全绿，因为**它们没有一个读过这份文档**。

    ⇒ 与 #11（文档过期）、`registry_consistency`（声明 vs 实物）、#98（判据的动作与
    它声称的动作不是同一件事）同族。本条的机械形式是**双向**的：

      J1 规格块点名的每个文件必须真实存在；
      J2 规格块点名的每个 `name()` 必须在 `scripts/*.py` 里真实存在；
      J3 **接线判据，双向锁**：`wired=false` ⇒ 不许有脚本读过滤产物；
         `wired=true` ⇒ 必须有脚本读它。**任一方向改变都必须同时改文档**，
         否则本门禁判红。

    ⚠️ **判据只看 `` 里的规格块，不读散文。** 这不是偷懒，是本条自证的必要条件：
    首版按散文里的 `` `name()` `` 抓点名，于是**我写在订正段里的那两个历史错名
    被它当成了现行声明**（1 条假红 ×2，与台账 #94「判据把事故记录判成残留」同型）。
    文档必须能够**引用它正在纠正的错话**；把判据的适用范围显式划出来，
    比让判据去猜散文可靠。⇒ 规格块缺失或写坏 ⇒ **exit 2**（没测到 ≠ 通过）。
    """
    spec_path = spec_path or SPEC_DOC
    if not spec_path.is_file():
        print(f"🔴 规格文档读不到：{spec_path} —— 「没东西可查」不等于「查过了没问题」")
        return 2
    text = spec_path.read_text(encoding="utf-8")
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    if not scripts:
        print("🔴 scripts/ 下一个 .py 都没有 —— 输入没拿到，exit 2")
        return 2
    src_all = "\n".join(p.read_text(encoding="utf-8") for p in scripts)

    block = re.search(r"<!--\s*P2S-SPEC:pipeline(.*?)-->", text, re.S)
    if not block:
        print(f"🔴 规格文档里没有 `<!-- P2S-SPEC:pipeline ... -->` 规格块 —— "
              f"**判据没有可读的声明**（正文散文不算声明：文档必须能引用它纠正的错话）。"
              f"exit 2（没测到 ≠ 通过）")
        return 2
    errs, n_ok = [], 0
    # ⚠️ `wired=` **不是行首 token** —— 它跟在步骤行尾巴上（`2 过滤 candidate_filter.py  wired=false`）。
    # 首版写成 `line.startswith("wired=")` ⇒ 永远取不到 ⇒ `wired` 恒为 None ⇒
    # **整条 J3 一次都没执行，而屏幕上看起来一切正常**（本仓库 #99/#34 同族：
    # 「没测到」伪装成「通过」）。⇒ 现在**取不到就 exit 2**，并且端到端用例 ④b 专门喂
    # 一份**没有** `wired=` 的规格块，要求 exit 2。
    wm = re.findall(r"wired\s*=\s*([A-Za-z]+)", block.group(1))
    if len(wm) != 1:
        print(f"🔴 规格块里 `wired=` 出现 {len(wm)} 次（必须恰好 1 次）—— "
              f"接线状态读不到 ⇒ **判据无法执行**，exit 2（没测到 ≠ 通过）")
        return 2
    wired = wm[0].lower()
    steps = []
    for raw in block.group(1).strip().splitlines():
        line = raw.strip().lstrip("#").strip()
        if line:
            steps.append(line)
    if wired not in ("true", "false"):
        errs.append(f"J3 规格块的 `wired=` 必须是 true 或 false，实际 {wired!r}")

    # ---- J1：点名的文件必须存在；J2：点名的 `name()` 必须存在 ----
    for line in steps:
        for rel in re.findall(r"([A-Za-z0-9_./-]+\.py)", line):
            cand = [REPO / rel, ROOT / rel, ROOT / "scripts" / Path(rel).name]
            if any(c.is_file() for c in cand):
                n_ok += 1
            else:
                errs.append(f"J1 规格块点名了 `{rel}`，但仓内找不到这个文件")
        for fn in re.findall(r"([a-z_][a-z0-9_]*)\(\)", line):
            if re.search(rf"^\s*def {re.escape(fn)}\s*\(", src_all, re.M):
                n_ok += 1
            else:
                errs.append(f"J2 规格块点名了 `{fn}()`，但 scripts/ 下没有任何 `def {fn}(` "
                            f"—— **声明在引用一个不存在的函数**")
    if not steps:
        errs.append("J1/J2 规格块里一条流水线步骤都没写 —— 空块等于没声明")

    # ---- J3：接线判据，双向锁 ----
    # ⚠️ 「谁在读这个产物」用 **AST 判**，不用「产物名附近有没有 read_text」这种邻近启发式 ——
    # 首版就是邻近式（产物名 ±400 字符内有 `read_text`/`json.load`），当场判红了
    # **我自己刚写的端到端夹具**（`run_phase6_gates.py` 里那段构造假读者的字符串），
    # 而它根本不是读者。与 #79/#80「去归属语只查 `材料` 附近、不查词自己附近」同族。
    # AST 版只认**真的把该字符串传进了读函数**的位置。
    readers = sorted({p.name for p in scripts
                      if p.name != Path(__file__).name and _reads_file(p, FILTERED_POOL)})
    if wired == "false" and readers:
        errs.append(f"J3 文档声明**未接线**，但 {readers} 在读 {FILTERED_POOL} —— "
                    f"接线已发生 ⇒ **必须同时改文档**（否则下一个人读到的接线状态是假的）")
    elif wired == "true" and not readers:
        errs.append(f"J3 文档声明已接线，但没有任何脚本读 {FILTERED_POOL} "
                    f"—— 那就是「交付」被写成了「接线」（台账 #23/#71/#78 同族）")
    elif wired in ("true", "false"):
        n_ok += 1
        print(f"✅ J3 接线状态与代码一致："
              f"{'未接线（无脚本读过滤产物）' if wired == 'false' else '已接线'}"
              f"{f'（读它的脚本：{readers}）' if readers else ''}")

    print(f"规格块：{len(steps)} 条步骤 · 逐条核实通过 {n_ok} 项")
    for line in steps:
        print(f"    {line}")
    if errs:
        print(f"\n🔴 {len(errs)} 条与代码不符：")
        for e in errs:
            print(f"   - {e}")
        return 1
    print("✅ 规格块点名的接线点全部真实存在，且接线状态与代码一致")
    return 0


def check_labels(labels_path: Path | None = None, pool_path: Path | None = None) -> int:
    """校准集必须**仍然描述着当前池子** —— 否则它是一条会腐烂的过期豁免。

    退出码 **0 全过 / 1 判红（标注集与池子脱节，必须重标）/ 2 输入没拿到**。

    ⚠️ 为什么这条要单独存在：本仓库已经因为「过期豁免没人删」立过两次纪律
    （#84 绿门禁 stdout 根本不显示豁免提示；`card-identity-baseline` 带 `expires_when`）。
    标注集的腐烂形态尤其隐蔽 —— **它不会报错，它只会让下一个读它的人相信一份不再成立的账**。
    """
    labels_path = labels_path or DATA / "constraint-salience-labels.json"
    pool_path = pool_path or DATA / "arxiv_candidates.json"
    if not labels_path.is_file():
        print(f"🔴 标注集读不到：{labels_path} —— 「没东西可查」不等于「查过了没问题」")
        return 2
    if not pool_path.is_file():
        print(f"🔴 候选池读不到：{pool_path}")
        return 2
    doc = json.loads(labels_path.read_text(encoding="utf-8"))
    cases = doc.get("cases") or []
    if not cases:
        print("🔴 标注集为空 —— 同上，exit 2")
        return 2
    items = json.loads(pool_path.read_text(encoding="utf-8")).get("items", [])
    by_id = {_item_id(i): i for i in items}
    if not items:
        print("🔴 候选池为空 —— exit 2")
        return 2

    errs, warns = [], []
    tiers = Counter()
    # P2b：类表在**这个池子**上现算（域精度是池子的性质）—— 逐条核对 `class_late`。
    prec = domain_precision(items)
    cls = {d: topic_class_words(d, prec) for d in DOMAIN_CONSTRAINT}
    n_class_ok = 0
    for c in cases:
        aid, dom, word, lab = c.get("arxiv_id"), c.get("domain"), c.get("word"), c.get("label")
        tag = f"{aid}/{dom}/{word}"
        if lab not in ("POLLUTION", "BORDERLINE", "GENUINE"):
            errs.append(f"{tag}：label={lab!r} 不在枚举内")
        if not (c.get("why") or "").strip():
            errs.append(f"{tag}：`why` 是空的 —— 空壳条目等于没标注")
        if c.get("confidence") not in ("high", "medium", "low"):
            errs.append(f"{tag}：confidence={c.get('confidence')!r} 不在枚举内")
        it = by_id.get(aid)
        if it is None:
            errs.append(f"{tag}：这篇**已不在候选池里**（池子换了 ⇒ 标注集必须重标）")
            continue
        if dom not in _domains.resolve_groups(it.get("query_groups") or [])["resolved"]:
            errs.append(f"{tag}：它的 `query_groups` 解析后不含该域（标签被改过？）")
        if word not in DOMAIN_CONSTRAINT.get(dom, []):
            errs.append(f"{tag}：{word!r} 已不是 {dom} 的约束词（词表改了 ⇒ 该重标）")
        # ⚠️ 判据是「**该主题短语**只在摘要命中」，**不是**「所有约束词都不在标题」——
        # 后者会把 7 篇论文误判成脱节：它们题名里另有 `database`/`table` 等**别的**
        # 约束词（因而形态是 SELF_EVIDENT），而本条标注说的是 `text-to-sql` 只在摘要。
        # 这个区分由 `--check-labels` 首跑抓出来（首版把两个条件写成了一个）。
        blob_title = (it.get("title") or "").lower()
        blob_abs = (it.get("abstract") or "").lower()
        if word not in blob_abs:
            errs.append(f"{tag}：{word!r} 已不在摘要里（标注集的前提不成立）")
        if word in blob_title:
            errs.append(f"{tag}：{word!r} 现在**命中标题**了 —— 它不再是「仅摘要」形态，该重标")
        ev = constraint_evidence(it, dom, cls[dom])
        tiers[ev["tier"]] += 1
        if not judge(it, dom)[0]:
            errs.append(f"{tag}：它**当前被判丢**了 —— 标注集描述的是保留对")
        # ⚠️ 类感知读数必须逐条对得上 —— 这一栏是 P2b 交付的**全部理由**
        # （污染召回 18/26 → 26/26）。它漂了而没人知道，就等于交付了一个没人守的读数。
        if not isinstance(c.get("class_late"), bool):
            errs.append(f"{tag}：缺 `class_late`（布尔）—— 空壳条目等于没标注")
        elif c["class_late"] != ev["late_class"]:
            errs.append(f"{tag}：`class_late` 标注 {c['class_late']}，现算 {ev['late_class']}"
                        f"（类词命中 {ev['class_hits']}，最浅 {ev['class_first_frac']}）"
                        f" ⇒ 池子/词表变了，标注集必须重标")
        else:
            n_class_ok += 1
    got_counts = {k: sum(1 for c in cases if c.get("label") == k)
                  for k in ("POLLUTION", "BORDERLINE", "GENUINE")}
    if doc.get("counts") != got_counts:
        errs.append(f"`counts` 与逐条标注不符：文件写 {doc.get('counts')}，现算 {got_counts}")
    if not (doc.get("_expires_when") or "").strip():
        errs.append("缺 `_expires_when` —— 标注集必须写清什么时候失效（防过期豁免腐烂）")

    print(f"校准集 {len(cases)} 条 · 命中形态 {dict(tiers)}")
    print(f"  标注：{got_counts}")
    print(f"  其中 `class_late=True`（类感知标出的）"
          f"**{sum(1 for c in cases if c.get('class_late') is True)}** 条"
          f" · `class_late` 与现算相符 **{n_class_ok}/{len(cases)}**")
    _poll = sum(1 for c in cases if c.get("label") == "POLLUTION")
    _hit = sum(1 for c in cases if c.get("label") == "POLLUTION" and c.get("class_late"))
    print(f"  污染召回（类感知）：**{_hit}/{_poll}**"
          f"（全词版口径见 `--report` 的普查对照，数字现算不写死）")
    if warns:
        for w in warns:
            print(f"  ⚠️ {w}")
    if errs:
        print(f"\n🔴 {len(errs)} 条与当前池子/词表脱节 —— 标注集必须重标，不许当成还有效：")
        for e in errs:
            print(f"   - {e}")
        return 1
    print("✅ 校准集仍描述着当前池子（逐条 id / 域 / 词 / 形态 / 保留态五项相符）")
    return 0


def check(pool_path: Path | None = None, out_path: Path | None = None,
          baseline: dict | None = None, compare_baseline: bool = True) -> int:
    """当门禁用：池子必须可判、**「仪器瞎了」必须为 0**、读数必须等于基线。

    退出码 **0 全过 / 1 判红（读数漂了）/ 2 没测到（池子读不到，或域标签认不出来）
    / 3 内部错误**。

    ⚠️ `unresolved > 0` 走 **2 而不是 1**：它不是「这篇论文不合格」，是
    **仪器看不见它** —— 本仓库危险性排序 **3 > 2 > 1 > 0**，
    「没测到」比「测了是红的」更危险。旧版不但不报，还把它算进「保留」。
    """
    baseline = baseline if baseline is not None else BASELINE
    pool_path = pool_path or DATA / "arxiv_candidates.json"
    if not pool_path.is_file():
        print(f"🔴 候选池读不到：{pool_path} —— 「没东西可查」不等于「查过了没问题」")
        return 2
    src = json.loads(pool_path.read_text(encoding="utf-8"))
    items = src.get("items", src if isinstance(src, list) else [])
    if not items:
        print("🔴 候选池为空 —— 同上，exit 2")
        return 2

    r = filter_pool(items)
    n = len(items)
    n_unres = len(r["unresolved_ids"])
    reading = {
        "count_before": n,
        "count": len(r["kept"]),
        "judged": r["judged"],
        "unresolved_labels": dict(sorted(r["unresolved"].items())),
        "unresolved_items": n_unres,
        "no_groups_items": len(r["no_groups_ids"]),
        "per_domain": dict(sorted(r["per_domain"].items())),
        # ⚠️ 证据形态是**读数**，不进判定。放进来是为了让「形态分布变了」可见。
        "tiers": dict(sorted(r["tiers"].items())),
        "incidental_items": len(r["incidental_ids"]),
        # ⚠️ 类感知读数（P2b）。与 tiers 同栏：**不进判定**，漂了只打 🟡。
        "late_class_items": len(r["late_class_ids"]),
        "no_class_items": len(r["no_class_ids"]),
        "class_domains": sum(1 for ws in r["class_table"].values() if ws),
    }
    if out_path:
        out_path.write_text(json.dumps(reading, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    print(f"候选池 {n} 篇 → 判定 {r['judged']} · 保留 {len(r['kept'])} "
          f"（丢弃 {n - len(r['kept'])}）")
    if n_unres:
        print(f"\n🔴 **仪器瞎了：{n_unres} 篇的域标签认不出来** —— "
              f"这 {n_unres} 篇的负向词与约束词**一条都没生效**。"
              f"（不是「这些论文不合格」，是「判据看不见它们」）")
        for lab, c in sorted(r["unresolved"].items()):
            hint = _domains.RETIRED.get(lab)
            print(f"     {c:>4}  {lab!r}"
                  + (f"  ← 这是**退休的旧名**，规范名是 {hint!r}" if hint else ""))
        print(f"   论文 id（前 10）：{r['unresolved_ids'][:10]}")
        print("   ⇒ exit 2（没测到 ≠ 通过）。修法：把标签折到规范名"
              "跑 `migrate_domain_labels.py --apply` 折成规范名。\n"
              "   ⚠️ **不许**把退休名加回解析路径 —— 那会让下一次漂移重新变得看不见。")
        return 2
    if r["no_groups_ids"]:
        print(f"⚠️ {len(r['no_groups_ids'])} 篇真的没有 `query_groups`（与上面那类**是两件事**）："
              f"{r['no_groups_ids'][:10]}")

    # ⚠️ 证据形态**先于**基线比对被说出来：它是本批唯一新增的一等读数，
    # 但必须紧跟着那句「不进判定」，否则下一个人会以为它已经在过滤了。
    print(f"\n约束词命中形态（**读数，不进判定**）：{dict(sorted(r['tiers'].items()))}")
    print(f"  其中「只在摘要后段命中」的保留对 {len(r['incidental_ids'])} 条 —— "
          f"这是上一批登记的『顺带提及』那一类，现**可见、可复核**，"
          f"但**不据此丢弃**（判据未标定，理由见模块内编译期注释）。")
    print(f"  逐条名单：paper2skills-research/data/constraint-salience-labels.json")
    print(f"  （校准集；`--check-labels` 会在池子/词表变动后判红要求重标）")
    _blind = sorted(d for d, ws in r["class_table"].items() if not ws)
    print(f"\n类感知「主题词只在摘要后段」保留对 **{len(r['late_class_ids'])}** 条"
          f"（全词版 {len(r['incidental_ids'])} 条）· 有主题类词的域 "
          f"**{sum(1 for ws in r['class_table'].values() if ws)}/{len(r['class_table'])}**")
    print(f"  ⚠️ 无主题类词的 {len(_blind)} 个域（{_blind}）上这一栏**无从判定**："
          f"{len(r['no_class_ids'])} 个保留对落在那里 —— 「判据看不见」≠「干净」")

    if not compare_baseline:
        print(f"\n✅ 仪器瞎了 0（显式池子，**不对基线** —— 基线是这个冻结池子的性质）")
        return 0

    errs = []
    if n != baseline["count_before"]:
        errs.append(f"池子篇数 {n} ≠ 基线 {baseline['count_before']}（上游换了池子 ⇒ 基线要重取）")
    if len(r["kept"]) != baseline["count"]:
        errs.append(f"保留数 {len(r['kept'])} ≠ 基线 {baseline['count']}（改了词表或池子？）")
    if r["judged"] != baseline["judged"]:
        errs.append(f"判定过的篇数 {r['judged']} ≠ 基线 {baseline['judged']}")
    for d, c in baseline["per_domain"].items():
        if r["per_domain"].get(d, 0) != c:
            errs.append(f"域 {d} 保留数 {r['per_domain'].get(d, 0)} ≠ 基线 {c}")
    for d, c in r["per_domain"].items():
        if d not in baseline["per_domain"]:
            errs.append(f"域 {d} 有 {c} 篇保留，但它不在基线里（新域 ⇒ 显式登记）")
    # 证据形态：分栏比对，**与判定分开报**（改的是形态还是判定，结论完全不同）
    tier_errs = []
    for t, c in baseline.get("tiers", {}).items():
        if r["tiers"].get(t, 0) != c:
            tier_errs.append(f"形态 {t} 计数 {r['tiers'].get(t, 0)} ≠ 基线 {c}")
    for t, c in r["tiers"].items():
        if t not in baseline.get("tiers", {}):
            tier_errs.append(f"形态 {t} 有 {c} 条，但它不在基线里（新形态 ⇒ 显式登记）")
    if len(r["incidental_ids"]) != baseline.get("incidental_items", -1):
        tier_errs.append(f"「顺带提及」保留对 {len(r['incidental_ids'])} "
                         f"≠ 基线 {baseline.get('incidental_items')}")
    for key, cnt, name in (("late_class_items", len(r["late_class_ids"]), "类感知「主题词只在后段」"),
                           ("no_class_items", len(r["no_class_ids"]), "该域无主题类词"),
                           ("class_domains", sum(1 for ws in r["class_table"].values() if ws),
                            "有主题类词的域数")):
        if cnt != baseline.get(key, -1):
            tier_errs.append(f"{name} {cnt} ≠ 基线 {baseline.get(key)}")
    if tier_errs:
        print("\n🟡 证据形态与基线不符（**判定未变** —— 这一栏不是判据，是读数）：")
        for e in tier_errs:
            print(f"   - {e}")
    if errs:
        print("\n🔴 读数与基线不符：")
        for e in errs:
            print(f"   - {e}")
        return 1
    print(f"\n✅ 判定与基线逐项相等（判定 {baseline['judged']} · 保留 {baseline['count']} · "
          f"仪器瞎了 0 · {len(baseline['per_domain'])} 个域逐域相等）")
    # ⚠️ 形态漂了但判定没漂 ⇒ **不算判红**（否则一改词表就红成一片，人会去关掉它）。
    # 但它已经在上面的 🟡 里被点名 —— 「只登记不判」也比静默好。
    # 这里**不许**再说「逐项相等」：上面刚打过 🟡，两句并存会让读者以为形态也没漂。
    if tier_errs:
        print(f"   ⚠️ 但**证据形态有 {len(tier_errs)} 项漂移**（见上 🟡）—— 判定相等 ≠ 一切都相等")
    return 0


def _report_census(n_late: int, n_all: int, r: dict) -> None:
    """把两条规则在**全池普查标注集**上的读数并排打出来。**数字现算，不写死。**

    ⚠️ 为什么必须是现算：这两个数正是本条交付的**全部理由**（召回与纯度）。
    写死它们就等于「交付一个读数」，而本仓库已经四次栽在「交付 ≠ 接线」上
    （台账 #11 / #23 / #71 / #78 / #105）。标注集读不到 ⇒ **说出来**，不假装。
    """
    lp = DATA / "constraint-salience-labels.json"
    if not lp.is_file():
        print(f"  ⚠️ 普查标注集读不到（{lp}）⇒ 召回/纯度**无从报出**"
              f"（「没测到」不许当成「没问题」）")
        return
    lab = json.loads(lp.read_text(encoding="utf-8"))
    cases = lab.get("cases") or []
    by = {(c["arxiv_id"], c["domain"]): c for c in cases}
    def tally(keys):
        c = Counter(by[k]["label"] for k in keys if k in by)
        return c, len([k for k in keys if k not in by])

    # 全词版集合 = 五态里 INCIDENTAL 的那些 (论文,域)；类感知版 = late_class_ids。
    inc_keys = {(x["arxiv_id"], x["domain"]) for x in r["incidental_ids"]}
    late_keys = {(x["arxiv_id"], x["domain"]) for x in r["late_class_ids"]}
    poll = sum(1 for c in cases if c["label"] == "POLLUTION")
    a, a_un = tally(inc_keys)
    b, b_un = tally(late_keys)
    for tag, c, un, n in (("全词版（现行五态）", a, a_un, n_all),
                          ("类感知版（本批）", b, b_un, n_late)):
        print(f"  {tag}：标出 {n} 条 · 已标注 {n - un} 条 → 污染 {c['POLLUTION']} / "
              f"边界 {c['BORDERLINE']} / 真命中 {c['GENUINE']} · 未标注 {un}")
        if n:
            print(f"      污染召回 {c['POLLUTION']}/{poll}（{c['POLLUTION']/poll:.0%}） · "
                  f"纯污染率 {c['POLLUTION']/n:.1%}")
    print(f"  逐条 id / 上下文 / 理由：paper2skills-research/data/constraint-salience-labels.json")
    print(f"  ⚠️ 类感知版**覆盖率有限**：只有 {len([d for d in r['class_table'] if r['class_table'][d]])}"
          f"/{len(r['class_table'])} 个域有主题类词 ⇒ 它的「标出 0 条」不等于该域干净。")


def check_class(pool_path: Path | None = None, baseline: dict | None = None,
                compare_baseline: bool = True) -> int:
    """**类表与类感知读数**的独立核对（验收面 L19d）。

    退出码 **0 全过 / 1 判红 / 2 输入没拿到 / 3 内部错误**。

    为什么它自成一条门禁（而不是并进 `--check`）：`--check` 只回答「读数等于基线吗」，
    而基线是**我写下的数**。类表却是**每次现跑算出来的**，于是有一类假话它看不见 ——
    「把表冻成一个手写常量，值还刚好对得上」。#84（过期豁免腐烂）正是从这儿进来。
    本条的机械形式是四条，**每条都能失败**：

      J1 **独立复算**：用一份不复用 `domain_precision()` / `topic_class_words()` 的
         暴力实现重算整张类表，逐词比对 —— 手写表、陈旧表、错阈值都在这条上现形。
      J2 **盲区是一等输出**：类表为空的域名单与落在盲区的保留对必须**被报出来**；
         若**所有**域都为空（整条读数瞎了）⇒ 判红。「看不见」不许伪装成「干净」。
      J3 **结构性守卫**：`judge()` 的函数体里不许出现任何类感知字段名（AST 判定）。
         这不是重复 M10 —— M10 守**行为**（一条样本仍被保留），J3 守**源码**：
         下一个人把 `late_class` 写进别的分支时，行为用例未必挑得出来。
      J4 类感知计数与基线相等（仅默认池子 —— 基线是这个冻结池子的性质）。
    """
    baseline = baseline if baseline is not None else BASELINE
    pool_path = pool_path or DATA / "arxiv_candidates.json"
    if not pool_path.is_file():
        print(f"🔴 候选池读不到：{pool_path} —— 「没东西可查」不等于「查过了没问题」")
        return 2
    src = json.loads(pool_path.read_text(encoding="utf-8"))
    items = src.get("items", src if isinstance(src, list) else [])
    if not items:
        print("🔴 候选池为空 —— exit 2")
        return 2
    try:
        r = filter_pool(items)
    except Exception as exc:                                   # noqa: BLE001
        print(f"🔴 门禁内部错误（filter_pool 抛异常）：{exc!r}")
        return 3

    errs = []

    # --- J1 独立复算 -----------------------------------------------------------
    # ⚠️ 刻意**不 import、不调用** `domain_precision()` / `topic_class_words()`：
    # 复用被检对象等于让被判者给自己出卷。这里用最笨的循环重算一遍。
    brute: dict = {}
    for dom, words in DOMAIN_CONSTRAINT.items():
        rows = []
        for w in words:
            hits = [it for it in items
                    if w in ((it.get("title") or "") + " " + (it.get("abstract") or "")).lower()]
            in_dom = sum(1 for it in hits
                         if dom in _domains.resolve_groups(
                             it.get("query_groups") or [])["resolved"])
            rows.append(((in_dom / len(hits)) if hits else None, w))
        brute[dom] = [w for p, w in sorted(rows, key=lambda pw: (-(pw[0] or 0.0), pw[1]))
                      if (p or 0.0) > TOPIC_CLASS_MIN_PRECISION]
    for dom in DOMAIN_CONSTRAINT:
        got = [x["word"] for x in r["class_table"].get(dom, [])]
        if got != brute[dom]:
            errs.append(f"J1 {dom} 类表与独立复算不符：现算 {got} · 暴力复算 {brute[dom]}")
    print(f"J1 类表独立复算：{len(DOMAIN_CONSTRAINT)} 个域逐域比对 → "
          f"{'**不一致**' if any(e.startswith('J1') for e in errs) else '一致'}")

    # --- J2 盲区必须被报出来 ---------------------------------------------------
    blind = sorted(d for d, ws in r["class_table"].items() if not ws)
    n_no_class = len(r["no_class_ids"])
    print(f"J2 盲区：**{len(blind)}/{len(DOMAIN_CONSTRAINT)} 个域没有主题类词** → {blind}")
    print(f"   落在盲区里的保留对 **{n_no_class}** 条 —— 「判据看不见」≠「这些论文干净」")
    if len(blind) == len(DOMAIN_CONSTRAINT):
        errs.append("J2 **全部域都没有主题类词** ⇒ 类表整条是空的：这一定是判据瞎了"
                    "（阈值被改错、或 precision 恒为 None），不是「所有域都很干净」")

    # --- J3 结构性守卫：新读数不许进判定 ---------------------------------------
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "judge"), None)
    names = {"late_class", "class_words", "class_hits", "class_first_frac", "class_table"}
    used: list[str] = []
    if fn is None:
        errs.append("J3 找不到 `judge()` —— 门禁的判据锚点没了（改了函数名？）")
    else:
        # ⚠️ **三种写法都要认**：标识符（`late_class`）、属性（`ev.late_class`）、
        # 以及**字符串字面量**（`ev["late_class"]`）。首版只查前两种，于是
        # `constraint_evidence(item, domain)["late_class"]` 这种最自然的写法**溜过去了**
        # —— 负向控制实测 exit 0（假绿）。这正是 #2/#11/#14/#93/#107 那一族
        # 「判据只认一种写法」，只是这次写错的是本批新写的判据自己。
        doc = (fn.body[0].value if fn.body and isinstance(fn.body[0], ast.Expr)
               and isinstance(fn.body[0].value, ast.Constant) else None)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in names:
                used.append(n.id)
            elif isinstance(n, ast.Attribute) and n.attr in names:
                used.append(n.attr)
            elif isinstance(n, ast.Constant) and n is not doc \
                    and isinstance(n.value, str) and n.value in names:
                used.append(n.value)
            elif isinstance(n, ast.Constant) and n is not doc \
                    and isinstance(n.value, str) \
                    and any(f"\"{k}\"" in n.value or f"'{k}'" in n.value for k in names):
                used.append(n.value)
        used = sorted(set(used))
        if used:
            errs.append(f"J3 `judge()` 源码里出现了类感知字段 {used} —— "
                        f"**它只许是读数，不许进保留/丢弃**（见模块内编译期注释）")
    print(f"J3 `judge()` 源码：{'**出现**了 ' + str(used) if used else '不含类感知字段'}"
          f"（结构性「不许据此丢弃」）")

    # --- J4 基线 ---------------------------------------------------------------
    if compare_baseline:
        for key, cnt, name in (("late_class_items", len(r["late_class_ids"]),
                                "类感知「主题词只在后段」"),
                               ("no_class_items", n_no_class, "该域无主题类词"),
                               ("class_domains", sum(1 for ws in r["class_table"].values() if ws),
                                "有主题类词的域数")):
            if cnt != baseline.get(key, -1):
                errs.append(f"J4 {name} {cnt} ≠ 基线 {baseline.get(key)}")
    else:
        print("J4 跳过基线比对（显式池子 —— 基线是这个冻结池子的性质）")

    print(f"类感知「主题词只在后段」保留对 {len(r['late_class_ids'])} 条 · "
          f"有主题类词的域 {sum(1 for ws in r['class_table'].values() if ws)}/"
          f"{len(DOMAIN_CONSTRAINT)}")
    if errs:
        print(f"\n🔴 {len(errs)} 条不成立：")
        for e in errs:
            print(f"   - {e}")
        return 1
    print("✅ 类表可独立复算 · 盲区已报出 · `judge()` 不含类感知字段 · 计数等于基线")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="三段式检索过滤（关键词库 v2）")
    ap.add_argument("--pool", type=Path, default=DATA / "arxiv_candidates.json")
    ap.add_argument("--apply", action="store_true", help="过滤并写 arxiv_candidates_filtered.json")
    ap.add_argument("--report", action="store_true", help="只看统计，不写文件（默认行为）")
    ap.add_argument("--check", action="store_true",
                    help="当门禁用：仪器瞎了必须为 0，读数必须等于基线")
    ap.add_argument("--check-labels", action="store_true",
                    help="校准集必须仍描述着当前池子（池子/词表变动 ⇒ 判红要求重标）")
    ap.add_argument("--check-spec", action="store_true",
                    help="规格文档点名的接线点必须真实存在，接线状态必须与代码一致")
    ap.add_argument("--check-class", action="store_true",
                    help="主题类表可独立复算、盲区被报出、judge() 不含类感知字段")
    ap.add_argument("--spec", type=Path, default=SPEC_DOC)
    ap.add_argument("--labels", type=Path, default=DATA / "constraint-salience-labels.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.mutate:
        return mutate()
    if args.check_spec:
        return check_spec(args.spec)
    if args.check_labels:
        return check_labels(args.labels, args.pool)
    if args.check_class:
        # ⚠️ 与 `--check` 不同：这里**只看池子路径**，不看 `P2S_REPO`。
        # 理由是基线属于「这个池子」，而 `P2S_REPO` 只是「另一个 checkout」的代理 ——
        # 端到端夹具把 `P2S_REPO` 钉在本仓库上，J4 就会**静默跳过**，于是
        # 「计数漂了」这条判据在负向控制里**测不到**（本批实测：漏成假绿）。
        # 判据要能被测到，比判据的措辞更值钱。
        default_pool = (DATA / "arxiv_candidates.json").resolve()
        bl = BASELINE if Path(args.pool).resolve() == default_pool else None
        return check_class(args.pool, baseline=bl, compare_baseline=bl is not None)
    if args.check:
        # ⚠️ 基线是**这个冻结池子**的性质，不是判据的性质。所以只在读默认池子时对基线；
        # 显式 `--pool <别的池子>` 时只判「仪器瞎了没有」（否则夹具上必然报「篇数不符」，
        # 而那是一条**与判据无关**的红 —— 端到端用例里会看起来像「门禁有劲」）。
        # 基线只属于**本 checkout 的冻结池子**：`P2S_REPO` 指向别处时它没有意义。
        default_pool = (DATA / "arxiv_candidates.json").resolve()
        baseline = (BASELINE if (Path(args.pool).resolve() == default_pool
                                 and "P2S_REPO" not in os.environ) else None)
        return check(args.pool, args.json_out, baseline=baseline, compare_baseline=baseline is not None)

    if not args.pool.is_file():
        print(f"找不到候选池 {args.pool}")
        return 2
    src = json.loads(args.pool.read_text(encoding="utf-8"))
    items = src.get("items", src if isinstance(src, list) else [])
    if not items:
        print("候选池为空 —— 「没东西可查」不等于「查过了没问题」")
        return 2

    r = filter_pool(items)
    kept, reasons, per_domain = r["kept"], r["reasons"], r["per_domain"]
    n = len(items)
    print(f"候选池 {n} 篇 → 判定 {r['judged']} 篇 → 保留 {len(kept)} 篇"
          f"（丢弃 {n - len(kept)}，{100*(n-len(kept))/n:.1f}%）")

    # ⚠️ 「没测到」必须**先于**其它读数被说出来 —— 它是危险性最高的一类（3 > 2 > 1 > 0）。
    n_unres = len(r["unresolved_ids"])
    if n_unres:
        print(f"\n🔴 **仪器瞎了：{n_unres} 篇的域标签认不出来** —— "
              f"这 {n_unres} 篇的负向词与约束词**一条都没生效**，"
              f"而上面那个「保留数」把它们的**没测到**算成了**通过**。")
        for lab, c in sorted(r["unresolved"].items()):
            hint = _domains.RETIRED.get(lab)
            print(f"     {c:>4}  {lab!r}"
                  + (f"  ← 这是**退休的旧名**，规范名是 {hint!r}" if hint else ""))
        print(f"   论文 id（前 10）：{r['unresolved_ids'][:10]}")
    if r["no_groups_ids"]:
        print(f"⚠️ {len(r['no_groups_ids'])} 篇真的没有 `query_groups`"
              f"（与上一类**是两件事**，分开计数）：{r['no_groups_ids'][:10]}")

    print("\n丢弃原因 Top:")
    for rr, c in reasons.most_common(12):
        print(f"  {c:>4}  {rr}")
    print("\n各域保留数（**全部列出、不截断** —— 截断会让「哪个域变 0」看不见）：")
    for d in _domains.CANONICAL:
        print(f"  {per_domain.get(d, 0):>4}  {d}")

    print(f"\n约束词命中形态（**读数，不进判定**）：{dict(sorted(r['tiers'].items()))}")
    print(f"  「只在摘要后段命中」的保留对 {len(r['incidental_ids'])} 条；"
          f"其中去重论文 {len({x['arxiv_id'] for x in r['incidental_ids']})} 篇。"
          f"这一栏是上一批登记的『顺带提及』那一类 —— 现在看得见，但**不据此丢弃**。")
    if r["incidental_ids"]:
        print(f"  前 10 条 id：{sorted({x['arxiv_id'] for x in r['incidental_ids']})[:10]}")

    # --- P2b：把上面的读数**收紧到主题类词**（同样不进判定） -------------------
    n_late = len(r["late_class_ids"])
    blind = sorted(d for d, ws in r["class_table"].items() if not ws)
    print(f"\n主题类词（**实测域精度 > {TOPIC_CLASS_MIN_PRECISION}**，每次现算、不存盘）逐域清单：")
    for d in _domains.CANONICAL:
        ws = r["class_table"].get(d) or []
        if ws:
            print(f"  {d}： " + " · ".join(f"{x['word']}({x['precision']}，{x['hits']} 篇)"
                                          for x in ws))
    print(f"  ⚠️ **{len(blind)}/{len(r['class_table'])} 个域一个主题类词都没有**：{blind}")
    print(f"     它们的约束表整表都是泛词（实测最高精度也不到 {TOPIC_CLASS_MIN_PRECISION}）")
    print(f"     ⇒ 这些域上**类感知读数为空**（{len(r['no_class_ids'])} 个保留对落在这里）——")
    print(f"       是「判据看不见」，**不是**「这些论文干净」。")
    print(f"\n类感知「主题词只在摘要后段」保留对：**{n_late}** 条"
          f"（同一栏的全词版是 {len(r['incidental_ids'])} 条）")
    _report_census(n_late, len(r["incidental_ids"]), r)
    print(f"  ⚠️ 它是**分诊清单**，**不是判决** —— `judge()` 一字未读这两个新字段，")
    print(f"     且其中相当一部分是逐条确认过的真命中（读数见上）。")

    if args.apply:
        out = DATA / "arxiv_candidates_filtered.json"
        out.write_text(json.dumps(
            {"generated_at": src.get("generated_at"), "window": src.get("window"),
             "source_pool": str(args.pool.name), "count_before": n,
             "count": len(kept), "judged": r["judged"],
             # ⚠️ 没判定的条目 id 必须进产物：否则下游只看 count，
             # 会把「没测到」当成「通过」—— 这里是它唯一能被看见的地方。
             "unjudged_ids": r["unresolved_ids"] + r["no_groups_ids"],
             # ⚠️ 证据形态进产物：下游（打分/短名单/人工判定）据此知道
             # 「这篇是凭什么进这个域的」。**没有这一栏，9 篇污染与 19 篇真命中
             # 在文件里长得一模一样**（这正是本批要修的那个缺陷）。
             "constraint_evidence": r["evidence"],
             "incidental_ids": r["incidental_ids"],
             # P2b：类感知分诊清单 + 逐域类表（**这份是快照**，权威算法在脚本里现算；
             # 下游要用请重跑，别把这份快照当成词的性质 —— 它是**池子的性质**）。
             "late_class_ids": r["late_class_ids"],
             "no_class_ids": r["no_class_ids"],
             "topic_class_table": r["class_table"],
             "items": kept},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ {out}")
    return 0 if not n_unres else 2


if __name__ == "__main__":
    sys.exit(main())
