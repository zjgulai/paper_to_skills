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


def _blob(item: dict) -> tuple[str, str]:
    """返回 (标题小写, 标题+摘要小写)。标题命中负向词 = 硬丢弃。"""
    t = (item.get("title") or "").lower()
    a = (item.get("abstract") or "").lower()
    c = (item.get("comment") or "").lower()
    j = (item.get("journal_ref") or "").lower()
    return t, f"{t} . {a} . {c} . {j}"


def judge(item: dict, domain: str) -> tuple[bool, str]:
    """判定一篇论文在给定域下是保留还是丢弃。返回 (keep, reason)。"""
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
        unresolved_ids（论文 id，可复核）/ no_groups_ids
    """
    kept: list[dict] = []
    reasons: Counter = Counter()
    per_domain: Counter = Counter()
    unresolved: Counter = Counter()
    unresolved_ids: list[str] = []
    no_groups_ids: list[str] = []
    judged = 0

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
        if any(v[0] for v in verdicts):
            kept.append(it)
            for d, v in zip(domains, verdicts):
                if v[0]:
                    per_domain[d] += 1
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
}


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
    if errs:
        print("\n🔴 读数与基线不符：")
        for e in errs:
            print(f"   - {e}")
        return 1
    print(f"\n✅ 与基线逐项相等（判定 {baseline['judged']} · 保留 {baseline['count']} · "
          f"仪器瞎了 0 · {len(baseline['per_domain'])} 个域逐域相等）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="三段式检索过滤（关键词库 v2）")
    ap.add_argument("--pool", type=Path, default=DATA / "arxiv_candidates.json")
    ap.add_argument("--apply", action="store_true", help="过滤并写 arxiv_candidates_filtered.json")
    ap.add_argument("--report", action="store_true", help="只看统计，不写文件（默认行为）")
    ap.add_argument("--check", action="store_true",
                    help="当门禁用：仪器瞎了必须为 0，读数必须等于基线")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.mutate:
        return mutate()
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

    if args.apply:
        out = DATA / "arxiv_candidates_filtered.json"
        out.write_text(json.dumps(
            {"generated_at": src.get("generated_at"), "window": src.get("window"),
             "source_pool": str(args.pool.name), "count_before": n,
             "count": len(kept), "judged": r["judged"],
             # ⚠️ 没判定的条目 id 必须进产物：否则下游只看 count，
             # 会把「没测到」当成「通过」—— 这里是它唯一能被看见的地方。
             "unjudged_ids": r["unresolved_ids"] + r["no_groups_ids"],
             "items": kept},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ {out}")
    return 0 if not n_unres else 2


if __name__ == "__main__":
    sys.exit(main())
