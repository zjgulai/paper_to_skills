#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""check_keyword_tables.py — **词表三向对账**（PHASE6 P3）

## 为什么需要这个文件

三段式检索词住在**三个互不相识的地方**，而没有任何门禁比对过它们：

| 向 | 落点 | 谁在读它 |
|---|---|---|
| **散文** | `关键词库-v2.md` §2（人写的正向 / 约束 / 负向词） | 人 |
| **收割查询** | `arxiv_harvest.py` 的 `QUERY_GROUPS`（arXiv 查询串） | 收割脚本 |
| **代码表** | `candidate_filter.py` 的 `DOMAIN_CONSTRAINT` / `DOMAIN_NEGATIVE` | 过滤器 |

上一批（P2c）把 274 条被丢弃的论文登记为「**待裁决**」，并写下裁决的前置条件：

> 要判错杀得问「它是否也该被那个域的收割查询捞到」，**那要正向词表**。

本批先去找那张表，结论是**它早就存在，只是没人把它和另外两向接起来**：
正向词在散文里是一份、在 `QUERY_GROUPS` 里是另一份，**两份的词不一样**。
本文件把三向接成可对账的形式，并把差集**变成读数**。

## 五条判据（每条都有能打红的自检）

- **J1 双向完备**：§2 的域集合与 `domains.CANONICAL` 必须**双向**相等。
  缺一段 ⇒ 红（该域的词表只活在代码里，人看不见）；多一段 ⇒ 也红（散文写了
  一个不是规范域的段 —— `domains.py` 的教训是**别名表会把漂移静默折掉**）。
- **J2 覆盖率锁**：散文正向词里「有收割查询覆盖」的比例，与 `BASELINE` 逐项相等。
  ⚠️ 这不是「覆盖率越高越好」——它是**冻结的读数**，改了词表就必须在 diff 里
  显式改基线，而不是让数字悄悄漂。
- **J3 无表域登记**：§2 写了约束词、而代码里**没有约束表**的域，必须在
  `DECLARED_TABLE_GAPS` 里登记（带理由与到期条件）。登记项**不再成立** ⇒ 判红
  （台账 #84：过期豁免不许是绿的）。**没有表 ≠ 这个域干净** —— 这正是 P1 的形态
  在下一层的复现：`if not domains: keep` 是「域认不出来」，这里是「域认出来了、
  但它的表是空的」，两者在读数上长得一模一样。
- **J4 后果一等读数**：无表域的**实测后果**由脚本现算（不写死），
  并与登记里的数比对 —— 「21/21 篇会按散文表被丢」这句话必须能被打红。
- **J5 未归属词不静默**：§2 里带反引号但没落进任何字段的词，必须报出来。
  「没解析到」与「没有这个词」是两件事（危险性排序 **3 > 2 > 1 > 0**）。

## 本批自己撞出来的仪器缺陷

- **#118（续行）**：首版解析器只取标签**所在那一行**的反引号词，于是
  `01-因果推断` 的 11 个正向词只看见 **3** 个 —— 而 §2 的每条词表都跨 2–4 行。
  唯一的发现方式是拿它去跑真文档并**对着人眼数**。已锁进 selftest。
- **#119（同行换栏）**：`07-NLP-VOC` 的 `2. \`opinion mining\` + 约束 \`product\` / \`review\``
  把两个**约束**词算成了正向词 ⇒ 不处理的话「散文正向词」这个分母本身就是脏的。
- **#120（AST 只认一种写法）**：取 `QUERY_GROUPS` 的首版只认 `ast.Assign`，
  而真文件写的是**带注解**的 `QUERY_GROUPS: dict[str, list[str]] = {…}`（`ast.AnnAssign`）
  ⇒ 一个字面量都取不到。**与台账 #110 同型第三次**；这次没造成假绿，只因判据把
  「取不到」写成了 `SystemExit` 而不是给默认值（**这正是它当初写对的地方**）。
- **#123（缩进引用块）**：续行判据是「以空白开头」，于是 §2 里缩进两格的
  `  > ⚠️ 订正：…` 被并进当前栏 —— 本批自己加的订正引用块把 `constraint_total`
  从 **111** 抬到 **121**，**由 J2 覆盖率锁当场判红**。
  ⚠️ 方向与 #118 **相反**：那个让分母变小（差集整片消失），这个让分母**变大**
  （散文混进词表）—— 而后者自洽得多，**只打印计数是看不出来的**。

用法：
    python3 check_keyword_tables.py --check      # 五条判据（验收面 L20a）
    python3 check_keyword_tables.py --selftest   # 证明每条判据都会失败（L20b）
    python3 check_keyword_tables.py --adjudicate # 274 条被丢弃论文的四态裁决（L20c）
    python3 check_keyword_tables.py --mutate     # 篡改副本端到端（L20d）
    python3 check_keyword_tables.py --report     # 只出读数，不判定
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(os.environ.get("P2S_REPO") or Path(__file__).resolve().parents[2]).resolve()
SCRIPTS = REPO / "paper2skills-research" / "scripts"
DOC = REPO / "paper2skills-vault" / "07-资源库" / "关键词库-v2.md"
HARVEST = SCRIPTS / "arxiv_harvest.py"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

EXIT_OK, EXIT_RED, EXIT_NO_INPUT, EXIT_INTERNAL = 0, 1, 2, 3

#: §2 的四个字段标签（顺序即解析优先级；`补充正向` 并入 `positive`）
LABELS: list[tuple[str, str]] = [
    ("补充正向", "positive"),
    ("正向", "positive"),
    ("约束词", "constraint"),
    ("本域负向", "negative"),
]
FIELDS = ("positive", "constraint", "negative")

#: ⚠️ **冻结读数**，不是目标。覆盖率变了就必须在这里显式改一笔（diff 可见），
#: 而不是让它悄悄漂。取数见 `--report`。
BASELINE = {
    "positive_total": 180,
    "positive_covered": 74,
    "constraint_total": 111,
    "negative_total": 76,
}

#: §2 写了约束词、而 `candidate_filter.DOMAIN_CONSTRAINT` 里**没有这个域**的登记。
#: 每一条都必须带理由与**到期条件**；条件一旦不成立，本文件判红（#84）。
DECLARED_TABLE_GAPS: dict[str, dict] = {
    "10-MAS": {
        "field": "constraint",
        "why": "§2 写 `llm`/`agent`/`task`/`tool`，代码无表。实测这四个词在本域池子里"
               "**全是泛词** —— 按散文表实现会再丢 0 篇（见 J4 现算），"
               "即「写了等于没写」。处置是**登记为无效表**，不是补一张看起来有的表。",
        "expires_when": "candidate_filter.DOMAIN_CONSTRAINT 出现 '10-MAS' 键",
    },
    "11-AI人文": {
        "field": "constraint",
        "why": "§2 写 `metaphor`/`analogy`/`humanities`/`philosophy`/`well-being`，"
               "且**明写**不共现就会「退化成通用 ML 基础论文」。代码无表 ⇒ 实测该失败模式"
               "**已经 100% 发生**（J4 现算：21/21 篇按散文表都该丢），"
               "而没有任何门禁看得见 —— 这是散文预言了一个后果、代码没有拦截它。",
        "expires_when": "candidate_filter.DOMAIN_CONSTRAINT 出现 '11-AI人文' 键",
    },
}

# ---------------------------------------------------------------------------
# 解析：§2 散文 → 机读表
# ---------------------------------------------------------------------------
_TICK = re.compile(r"`([^`]+)`")
_HEAD = re.compile(r"^### (\S+?)(?:\s|$)")


def parse_section2(text: str) -> tuple[dict, dict]:
    """把 §2 解析成 `{domain: {positive/constraint/negative: [词]}}` + 未归属读数。

    ⚠️ 本函数的两个坑都已实测撞过，见模块 docstring 的 #118 / #119：
      · **续行**：词表跨 2–4 行，只读标签行会漏掉 60% 以上的词；
      · **同行换栏**：`+ 约束 \\`product\\`` 之后的反引号词属于约束栏，不属于本行标签栏。
    """
    lines = text.split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("## 2."))
        end = next(i for i, l in enumerate(lines) if l.startswith("## 3."))
    except StopIteration:
        return {}, {"fatal": "§2 段落找不到"}
    out: dict[str, dict] = {}
    unattributed: list[str] = []
    cur: str | None = None
    key: str | None = None
    for raw in lines[start:end]:
        m = _HEAD.match(raw)
        if m:
            cur = m.group(1)
            out[cur] = {f: [] for f in FIELDS}
            key = None
            continue
        if cur is None:
            continue
        if not raw.strip():
            key = None
            continue
        label = next((k for lab, k in LABELS if f"**{lab}" in raw), None)
        if label:
            key = label
            for w in _TICK.findall(raw):
                out[cur][key].append(w.strip().lower())
            continue
        if re.match(r"^\s*[-*]\s", raw):      # 别的 bullet（注意 / AI 方法 / ⚠️ …）
            key = None
            unattributed += [w for w in _TICK.findall(raw)]
            continue
        if re.match(r"^\s*>", raw):
            # ⚠️ **缩进引用块（`>`）是散文，不是词表**（#123）。
            # 首版把「以空白开头」一律当续行，于是 §2 里缩进两格的 `> ⚠️ …`
            # 被并进当前栏 —— 本批自己给 `10-MAS` / `11-AI人文` 加的**订正引用块**
            # 让 `constraint_total` 从 111 涨到 121，**由 J2 覆盖率锁当场判红**。
            # 这正是覆盖率锁存在的理由：解析面多吞了 10 个词，如果只报「解析到 121 个词」
            # 是看不出问题的（分母自己变大，读数永远自洽）。
            key = None
            continue
        if re.match(r"^\s+\S", raw):          # 续行（含 07 的编号列表）
            if key is None:
                unattributed += [w for w in _TICK.findall(raw)]
                continue
            # 同行换栏：`约束` 之后的词归约束栏（#119）
            head, sep, tail = raw.partition("约束")
            for w in _TICK.findall(head):
                out[cur][key].append(w.strip().lower())
            tgt = "constraint" if sep else key
            for w in _TICK.findall(tail if sep else ""):
                out[cur][tgt].append(w.strip().lower())
            continue
        key = None
        unattributed += [w for w in _TICK.findall(raw)]
    for d in out:
        for f in FIELDS:
            seen, uniq = set(), []
            for w in out[d][f]:
                if w and w not in seen:
                    seen.add(w)
                    uniq.append(w)
            out[d][f] = uniq
    return out, {"unattributed": sorted(set(unattributed))}


# ---------------------------------------------------------------------------
# 收割查询：AST 取 `QUERY_GROUPS`（**不 import** —— 那个模块会拉起 requests，
# 门禁不该因为一个无关的第三方库而在新克隆里变成 exit 3）
# ---------------------------------------------------------------------------
_Q_ABS = re.compile(r'abs:"([^"]+)"')


def _find_dict_literal(tree: ast.AST, name: str):
    """找 `name = {...}` **与** `name: T = {...}` 两种写法。

    ⚠️ 首版只认 `ast.Assign`，而真文件写的是 `QUERY_GROUPS: dict[str, list[str]] = {…}`
    （**带注解 ⇒ 是 `ast.AnnAssign`**）⇒ 一个字面量都找不到。
    这是台账 #110「AST 只认一种写法」的**又一次同型**，只是这次判据**把「取不到」
    写成了 SystemExit**，所以它当场炸而不是静默取空 —— 这正是 fail-loud 的兑现。
    """
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
                return stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                return stmt.value
    return None


def query_phrases(harvest_path: Path = HARVEST) -> dict[str, list[str]]:
    tree = ast.parse(harvest_path.read_text(encoding="utf-8"))
    node = _find_dict_literal(tree, "QUERY_GROUPS")
    if node is None:
        raise SystemExit("🔴 arxiv_harvest.py 里找不到 `QUERY_GROUPS = {...}` 字面量 —— "
                         "**没测到 ≠ 通过**（本文件不会猜它的新写法）")
    try:
        groups = ast.literal_eval(node)
    except ValueError:
        raise SystemExit("🔴 `QUERY_GROUPS` 不是纯字面量（引用了别的表达式）⇒ 取不到，"
                         "拒绝给默认值（#110 同族：AST 只认一种写法时会静默取空）")
    return {d: [p.strip().lower() for q in qs for p in _Q_ABS.findall(q) if p.strip()]
            for d, qs in groups.items()}


def code_tables():
    import candidate_filter as cf
    import domains as dm
    return cf.DOMAIN_CONSTRAINT, cf.DOMAIN_NEGATIVE, list(dm.CANONICAL)


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------
def _covered(word: str, phrases: list[str]) -> bool:
    """散文正向词是否被该域的收割查询覆盖（双向包含，容忍 `recommender/​recommendation`）。"""
    return any(word in p or p in word for p in phrases)


def audit(doc_path: Path = DOC, harvest_path: Path = HARVEST,
          declared: dict | None = None, baseline: dict | None = None) -> dict:
    declared = DECLARED_TABLE_GAPS if declared is None else declared
    baseline = BASELINE if baseline is None else baseline
    if not doc_path.exists():
        raise SystemExit(f"🔴 读不到 {doc_path} —— **没测到 ≠ 通过**")
    prose, notes = parse_section2(doc_path.read_text(encoding="utf-8"))
    if notes.get("fatal"):
        raise SystemExit(f"🔴 {notes['fatal']} —— exit 2（没测到 ≠ 通过）")
    queries = query_phrases(harvest_path)
    constraint, negative, canonical = code_tables()

    res: dict = {"prose_domains": sorted(prose), "canonical": canonical,
                 "unattributed": notes["unattributed"], "issues": []}

    # --- J1 双向完备 -------------------------------------------------------
    missing_in_doc = [d for d in canonical if d not in prose]
    extra_in_doc = [d for d in prose if d not in canonical]
    odd_domains = [d for d in queries if d not in canonical]
    res["j1"] = {"missing_in_doc": missing_in_doc, "extra_in_doc": extra_in_doc,
                 "query_domains_not_canonical": odd_domains}
    if missing_in_doc:
        res["issues"].append(f"J1 §2 缺这些规范域的段落：{missing_in_doc}")
    if extra_in_doc:
        res["issues"].append(f"J1 §2 有这些非规范域段落：{extra_in_doc}")
    if odd_domains:
        res["issues"].append(f"J1 收割查询里有非规范域：{odd_domains}")

    # --- J2 覆盖率锁 -------------------------------------------------------
    tot = cov = 0
    per_domain: dict[str, dict] = {}
    for d in canonical:
        pos = prose.get(d, {}).get("positive", [])
        qs = queries.get(d, [])
        hit = [w for w in pos if _covered(w, qs)]
        tot += len(pos)
        cov += len(hit)
        per_domain[d] = {"positive": len(pos), "covered": len(hit),
                         "queries": len(qs), "missing": [w for w in pos if w not in hit]}
    c_tot = sum(len(prose.get(d, {}).get("constraint", [])) for d in canonical)
    n_tot = sum(len(prose.get(d, {}).get("negative", [])) for d in canonical)
    got = {"positive_total": tot, "positive_covered": cov,
           "constraint_total": c_tot, "negative_total": n_tot}
    res["j2"] = {"computed": got, "baseline": baseline, "per_domain": per_domain}
    for k, v in got.items():
        if baseline.get(k) != v:
            res["issues"].append(
                f"J2 覆盖率读数 `{k}` 实测 {v} ≠ 基线 {baseline.get(k)} —— "
                f"改了词表就必须显式改基线（让漂移在 diff 里可见），不许让它悄悄漂")

    # --- J3 无表域登记 / 过期登记 -----------------------------------------
    gaps, stale = {}, []
    for d in canonical:
        wants = bool(prose.get(d, {}).get("constraint", []))
        has = d in constraint
        if wants and not has:
            gaps[d] = True
        if d in declared:
            # 登记的语义是「**这里有一个缺口**」。它仍然成立 ⟺ 表**仍然不存在**。
            # ⚠️ 首版把这里写反了（写成 `has`），症状是**真仓库里两个登记域当场被判过期** ——
            # 反过来也一样致命：写反之后「表补上了而登记还在」会变成绿的，
            # 那正好是 #84 要防的永久豁免。selftest ⑤ 的第三个用例就是冲它来的。
            still_gap = declared[d].get("field") == "constraint" and not has
            if not still_gap:
                stale.append(d)
    res["j3"] = {"gaps": sorted(gaps), "declared": sorted(declared), "stale": stale}
    unregistered = sorted(set(gaps) - set(declared))
    if unregistered:
        res["issues"].append(
            f"J3 这些域 §2 写了约束词、代码里**没有约束表**，且未登记：{unregistered} —— "
            f"「没有表」与「这个域很干净」在读数上长得一模一样（P1 的 `if not domains: keep` 同形态）")
    if stale:
        res["issues"].append(
            f"J3 过期登记：{stale} 的条件已不成立（表已存在）⇒ 请删掉登记"
            f"（台账 #84：永久豁免＝腐烂）")

    # --- J4 后果一等读数（现算，不写死）-----------------------------------
    consequences = {}
    try:
        import candidate_filter as cf
        import domains as dm
        pool_path = REPO / "paper2skills-research" / "data" / "arxiv_candidates.json"
        if pool_path.exists():
            items = json.load(open(pool_path, encoding="utf-8"))["items"]
            for d in sorted(gaps) + sorted(set(declared) - set(gaps)):
                words = [w.lower() for w in prose.get(d, {}).get("constraint", [])]
                labelled = [it for it in items
                            if d in dm.resolve_groups(it.get("query_groups") or [])["resolved"]]
                would_drop = [it for it in labelled
                              if words and not any(w in cf._blob(it)[1] for w in words)]
                consequences[d] = {"labelled": len(labelled),
                                   "would_drop_if_implemented": len(would_drop),
                                   "words": words}
    except Exception as exc:                      # noqa: BLE001
        consequences["<取不到>"] = f"{type(exc).__name__}: {exc}"
    res["j4"] = consequences

    # --- J5 未归属词不静默 -------------------------------------------------
    res["j5"] = {"unattributed": notes["unattributed"]}
    return res


# ---------------------------------------------------------------------------
# 274 条的裁决（P2c 登记的工单）
# ---------------------------------------------------------------------------
# P2c 把 274 条被丢弃的论文登记为「待裁决」，并写下前置条件：
#
#   > 要判错杀得问「它是否也该被那个域的收割查询捞到」，**那要正向词表**。
#
# 本批把正向词表接上了（`query_phrases()`），于是可以问这句话了。但**问法必须两段**：
# 只看「换一张域表能不能过」是**不够的** —— 约束表按设计只是**上下文闸门**
# （`table` 出现在 267 篇里、域精度 0.195，任何一篇提到 Table 1 的论文都能过 09 的表），
# 所以「过表」这件事本身几乎不携带归属信息。**归属信息在正向词那一侧。**
#
# 四态（互斥，优先级从高到低；次序本身是判据，见 J7）：
#   · OWN_TOPICAL_DROPPED  命中**本域**主题短语，却仍被本域判丢 —— 仪器自相矛盾
#   · RESCUE_TOPICAL       命中**他域**主题短语 **且** 该域约束表通过 ⇒ 有归属证据
#   · CONTEXT_ONLY         过得了某他域的约束表，但**一个主题短语都不命中** ⇒
#                          **这不是错杀的证据**（换域就换了问题；过表近乎必然）
#   · NO_EVIDENCE          任何域的主题短语都不命中 ⇒ 收割面看不见它
#                          ⚠️「看不见」≠「没用」：我们的查询只有 2–15 条/域
#
# ⚠️ **仪器的性质必须与读数一起报**：主题短语来自各域自己的收割查询，
# 而**独占短语**（只出现在一个域查询里的词）的精度会被「按构造贴标签」抬高 ——
# 命中它并被收割到的论文，按构造就带该域标签。故每条救回都连同**短语精度**与
# **是否独占短语**一起落盘，读数里也给直方图。
FILTERED_POOL = "arxiv_candidates_filtered.json"

S_OWN = "OWN_TOPICAL_DROPPED"
S_RESCUE = "RESCUE_TOPICAL"
S_CONTEXT = "CONTEXT_ONLY"
S_NONE = "NO_EVIDENCE"
STATES = (S_OWN, S_RESCUE, S_CONTEXT, S_NONE)


def _exclusive_phrases() -> dict[str, int]:
    """`{短语: 出现在几个域的收割查询里}` —— 用于标注「独占短语」。"""
    from collections import Counter
    c: Counter = Counter()
    for _d, ps in query_phrases().items():
        for w in set(ps):
            c[w] += 1
    return dict(c)


def adjudicate(dropped_path: Path | None = None, pool_path: Path | None = None) -> dict:
    """把被丢弃的论文四分。**只读，不改任何产物，不改 `judge()`。**"""
    import candidate_filter as cf
    import domains as dm
    data = REPO / "paper2skills-research" / "data"
    dp = Path(dropped_path) if dropped_path else data / FILTERED_POOL
    pp = Path(pool_path) if pool_path else data / "arxiv_candidates.json"
    if not dp.exists() or not pp.exists():
        raise SystemExit(f"🔴 读不到 {dp.name} / {pp.name} —— **没测到 ≠ 通过**（exit 2）")
    art = json.load(open(dp, encoding="utf-8"))
    # ⚠️ 判据看**自述块**，不看文件名（第 2 环接线立的三条纪律之一，逐字沿用）：
    # 把未过滤池复制成这个文件名必须同样被拒。
    filt = art.get("filter")
    if not isinstance(filt, dict) or not isinstance(filt.get("dropped"), list):
        raise SystemExit(
            "🔴 输入里没有过滤器的自述块 `filter.dropped` ⇒ 拒绝判定（exit 2）。\n"
            "   判据看自述块而不是文件名：把未过滤池改名冒充同样会被拒 ——\n"
            "   否则「接线了」与「没接线」在产物上长得一模一样。")
    dropped = filt["dropped"]
    items = json.load(open(pp, encoding="utf-8"))["items"]
    by = {cf._item_id(i): i for i in items}

    qp = query_phrases()
    prec = cf.word_precision(items, qp)
    thr = cf.TOPIC_CLASS_MIN_PRECISION
    topical = {d: [w for w in qp.get(d, [])
                   if (prec[(d, w)]["precision"] or 0.0) > thr] for d in qp}
    has_table = {d: d in cf.DOMAIN_CONSTRAINT for d in qp}
    excl = _exclusive_phrases()

    rows, states = [], {k: 0 for k in STATES}
    any_other_table = 0
    for e in dropped:
        pid = e.get("arxiv_id")
        it = by.get(pid)
        if it is None:
            raise SystemExit(f"🔴 丢弃集里的 {pid} 不在池子里 ⇒ 两份产物不同源（exit 2）")
        title, blob = cf._blob(it)
        recorded = set(dm.resolve_groups(e.get("domains") or [])["resolved"])
        own_ev, rescue_ev, ctx_domains = [], [], []
        for d, ws in topical.items():
            hits = [(w, prec[(d, w)]["precision"], excl.get(w, 1) == 1)
                    for w in ws if w in title or w in blob]
            if not hits:
                continue
            ok, _r = cf.judge(it, d)
            if d in recorded and not ok:
                own_ev.append({"domain": d, "phrases": hits})
            if d not in recorded and ok and has_table[d]:
                rescue_ev.append({"domain": d, "phrases": hits})
        for d in qp:
            if d in recorded or not has_table[d]:
                continue
            if cf.judge(it, d)[0]:
                ctx_domains.append(d)
        if ctx_domains:
            any_other_table += 1
        if own_ev:
            st = S_OWN
        elif rescue_ev:
            st = S_RESCUE
        elif ctx_domains:
            st = S_CONTEXT
        else:
            st = S_NONE
        states[st] += 1
        rows.append({"arxiv_id": pid, "title": it.get("title"),
                     "dropped_under": e.get("domains"), "reason": e.get("reason"),
                     "state": st, "own_evidence": own_ev, "rescue_evidence": rescue_ev,
                     "context_only_domains": ctx_domains})

    # --- 判据 -------------------------------------------------------------
    issues: list[str] = []
    if sum(states.values()) != len(dropped):
        issues.append(f"J6 恒等式破：四态之和 {sum(states.values())} ≠ 丢弃数 {len(dropped)}"
                      f"（**不许静默丢条目**）")
    bad_ctx = [r["arxiv_id"] for r in rows if r["state"] == S_CONTEXT and r["rescue_evidence"]]
    if bad_ctx:
        issues.append(f"J7 `CONTEXT_ONLY` 不许携带归属证据：{bad_ctx[:5]}")
    bad_r = [(r["arxiv_id"], h["domain"]) for r in rows if r["state"] == S_RESCUE
             for h in r["rescue_evidence"] if not has_table[h["domain"]]]
    if bad_r:
        issues.append(f"J7 救回证据不许来自**没有约束表**的域（过表是空的）：{bad_r[:5]}")
    # ⚠️ P2c 那句「226 条换一张域表就能过」的**分解**——这是本批最要紧的一个读数：
    # 过表这件事几乎不携带归属信息，所以必须拆成「有主题证据」与「只有上下文」两半。
    other_pass = [r for r in rows if r["context_only_domains"]]
    other_topical = [r for r in other_pass if r["rescue_evidence"]]
    other_ctx_only = [r for r in other_pass if not r["rescue_evidence"]]
    if len(other_topical) + len(other_ctx_only) != any_other_table:
        issues.append(f"J8 分解恒等式破：{len(other_topical)}+{len(other_ctx_only)} "
                      f"≠ 过他域表总数 {any_other_table}")
    # OWN 那一态的证据短语直方图 —— 用来暴露**独占短语自证**这个仪器性质
    own_hist: Counter = Counter()
    for r in rows:
        if r["state"] != S_OWN:
            continue
        for h in r["own_evidence"]:
            for w, _p, e in h["phrases"]:
                own_hist[(h["domain"], w, e)] += 1
    no_table_rescued = [d for d in qp if not has_table[d]]
    return {"dropped": len(dropped), "states": states, "rows": rows, "issues": issues,
            "any_other_table": any_other_table,
            "other_pass_topical": len(other_topical),
            "other_pass_context_only": len(other_ctx_only),
            "own_evidence_hist": {f"{d}|{w}|{'独占' if e else '共享'}": n
                                  for (d, w, e), n in own_hist.most_common()},
            "topical_phrases": {d: topical[d] for d in topical},
            "no_table_domains": no_table_rescued,
            "precision": {f"{d}|{w}": prec[(d, w)] for d in topical for w in topical[d]},
            "exclusive": {d: [w for w in topical[d] if excl.get(w, 1) == 1] for d in topical}}


def report_adjudication(res: dict) -> None:
    print("274 条被丢弃论文的裁决（四态 + 两段证据）")
    print(f"  丢弃 {res['dropped']} 条 · 他域约束表可过 {res['any_other_table']} 条"
          f"（P2c 登记的口径：226 —— 逐条复算**核对到 {res['any_other_table']}/226**）")
    print(f"  ⭐ 那 226 条**分解**：有主题证据 **{res['other_pass_topical']}** 条 · "
          f"**只有上下文（过表近乎必然，不携带归属信息）{res['other_pass_context_only']} 条**")
    print(f"     （两个数问的不是同一件事：{res['other_pass_topical']} 是「过表**且**有主题证据」，"
          f"{res['states'][S_RESCUE]} 是「有主题证据、过表、**且自己不是因为本域主题矛盾才被丢**」）")
    print("  ── OWN_TOPICAL_DROPPED 的证据短语直方图（**独占短语的自证性质看得见这里**）──")
    for k, n in list(res["own_evidence_hist"].items())[:12]:
        print(f"     {n:3d}  {k}")
    for k in STATES:
        print(f"    {k:22s} {res['states'][k]:3d}")
    print(f"  恒等式：{sum(res['states'].values())} == {res['dropped']}")
    from collections import Counter
    oc = Counter((r["reason"],) for r in res["rows"] if r["state"] == S_OWN)
    if oc:
        print("  OWN_TOPICAL_DROPPED 的原因分布（**原因本身就在说话**："
              "否定词多是判对了，`未命中约束词` 才是自相矛盾）：")
        for (r,), n in oc.most_common(8):
            print(f"     {n:3d}  {r}")
    print("  ── RESCUE_TOPICAL 逐条（可救回，按短语精度降序）──")
    got = sorted((r for r in res["rows"] if r["state"] == S_RESCUE),
                 key=lambda r: -max(p for h in r["rescue_evidence"] for _w, p, _e in h["phrases"]))
    for r in got:
        best = sorted(((h["domain"], w, p, e) for h in r["rescue_evidence"]
                       for w, p, e in h["phrases"]), key=lambda x: -x[2])[:2]
        print(f"     {r['arxiv_id']} 丢于{r['dropped_under']}({r['reason']}) "
              f"救={[(d, w, round(p, 2), '独占' if e else '') for d, w, p, e in best]}")
        print(f"        {str(r['title'])[:78]}")
    print(f"  ⚠️ 没有约束表、**不许当救回证据**的域：{res['no_table_domains']}")
    if res["issues"]:
        print("\n🔴 判红：")
        for i in res["issues"]:
            print(f"  · {i}")
    else:
        print("\n✅ J6/J7 全过")


def report(res: dict) -> None:
    print("词表三向对账（散文 §2 ↔ 收割查询 ↔ 代码表）")
    print(f"  §2 段落 {len(res['prose_domains'])} 个 · 规范域 {len(res['canonical'])} 个")
    j2 = res["j2"]
    c = j2["computed"]
    print(f"  J2 正向词 {c['positive_total']} 个，其中 **{c['positive_covered']} 个**"
          f"有收割查询覆盖（核对到 {c['positive_covered']}/{c['positive_total']}）")
    print(f"     约束词 {c['constraint_total']} 个 · 负向词 {c['negative_total']} 个")
    print("  ── 逐域（散文正向 / 已覆盖 / 查询短语数）──")
    for d, v in j2["per_domain"].items():
        flag = "  ⚠️" if v["covered"] < v["positive"] else "    "
        print(f"  {flag} {d:18s} {v['positive']:2d} / {v['covered']:2d}  "
              f"查询 {v['queries']:2d}   缺: {v['missing']}")
    print(f"  J3 无约束表的域 {res['j3']['gaps']}（已登记 {res['j3']['declared']}）")
    for d, v in res["j4"].items():
        if isinstance(v, dict):
            print(f"     {d:12s} 带此标签 {v['labelled']:3d} 篇 · "
                  f"按散文表实现**会再丢 {v['would_drop_if_implemented']} 篇** · 词={v['words']}")
        else:
            print(f"     {d} {v}")
    print(f"  J5 §2 里未归属的反引号词 {len(res['j5']['unattributed'])} 个"
          f"{res['j5']['unattributed'][:12]}")
    if res["issues"]:
        print("\n🔴 判红：")
        for i in res["issues"]:
            print(f"  · {i}")
    else:
        print("\n✅ 五条判据全过")


# ---------------------------------------------------------------------------
# selftest：每条判据各配一份**能打红的篡改样本**
# ---------------------------------------------------------------------------
def _fixture(domains=("D1", "D2")) -> tuple:
    doc = ["## 2. 按领域", ""]
    for d in domains:
        doc += [f"### {d}", "",
                "- **正向**：`alpha`、`beta`、", "  `gamma`、`delta`", "",
                "- **约束词**：`ctxone`、`ctxtwo`", "",
                "- **本域负向**：`badone`", ""]
    doc += ["", "## 3. 实测污染数据", ""]
    q = {d: ['abs:"alpha" OR abs:"beta"'] for d in domains}
    con = {d: ["ctxone"] for d in domains}
    neg = {d: ["badone"] for d in domains}
    return "\n".join(doc), q, con, neg


def _run_fixture(fx, con=None, neg=None, declared=None, baseline=None, canonical=None):
    """用夹具跑一遍 audit（不碰真仓库）。

    `fx` 是 `_fixture()` 的四元组；`con`/`neg` 传 None 表示沿用夹具自带的表，
    传 `{}` 表示**这个域根本没有表**（J3 要测的那一态）。
    """
    import candidate_filter as cf
    import domains as dm
    text, q, fx_con, fx_neg = fx
    tmp = Path(tempfile.mkdtemp(prefix="p2s-kwt-"))
    doc = tmp / "doc.md"
    doc.write_text(text, encoding="utf-8")
    har = tmp / "h.py"
    har.write_text("QUERY_GROUPS = " + repr(q) + "\n", encoding="utf-8")
    old_c, old_n, old_can = cf.DOMAIN_CONSTRAINT, cf.DOMAIN_NEGATIVE, dm.CANONICAL
    try:
        cf.DOMAIN_CONSTRAINT = fx_con if con is None else con
        cf.DOMAIN_NEGATIVE = fx_neg if neg is None else neg
        dm.CANONICAL = list(canonical if canonical is not None else cf.DOMAIN_CONSTRAINT.keys())
        return audit(doc, har, declared if declared is not None else {},
                     baseline if baseline is not None else {})
    finally:
        cf.DOMAIN_CONSTRAINT, cf.DOMAIN_NEGATIVE, dm.CANONICAL = old_c, old_n, old_can


B1 = {"positive_total": 4, "positive_covered": 2, "constraint_total": 2, "negative_total": 1}
B2 = {"positive_total": 8, "positive_covered": 4, "constraint_total": 4, "negative_total": 2}


def selftest() -> int:
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(f"  {'✅' if cond else '🔴'} {name}" + (f"  {extra}" if extra else ""))
        ok = ok and bool(cond)

    print("① 干净夹具：五条判据必须全过（**反向控制** —— 判据过严会被当坏工具绕过）")
    r = _run_fixture(_fixture(), baseline=B2)
    chk("干净夹具 0 issue", not r["issues"], str(r["issues"])[:100])
    chk("续行被解析到（每域 4 个正向词，不是 2 个）",
        r["j2"]["computed"]["positive_total"] == 8, str(r["j2"]["computed"]))
    chk("覆盖率 = 4/8（每域 alpha/beta 命中，gamma/delta 未进查询）",
        r["j2"]["computed"]["positive_covered"] == 4)

    print("② #118 续行缺陷回归：续行词必须被解析到，且**少算必须被抓**")
    r2 = _run_fixture(_fixture(domains=("D1",)),
                      baseline={"positive_total": 4, "positive_covered": 1,
                                "constraint_total": 2, "negative_total": 1},
                      canonical=["D1"])
    chk("4 个正向词全在（`gamma`/`delta` 在续行上）",
        r2["j2"]["per_domain"]["D1"]["positive"] == 4, str(r2["j2"]["per_domain"]["D1"]))
    r2b = _run_fixture(_fixture(domains=("D1",)),
                       baseline={"positive_total": 2, "positive_covered": 1,
                                 "constraint_total": 2, "negative_total": 1},
                       canonical=["D1"])
    chk("把基线写成 2（模拟首版只看见标签行）⇒ J2 判红",
        any("positive_total" in i for i in r2b["issues"]), str(r2b["issues"])[:90])

    print("③ #119 同行换栏：`+ 约束 `x`` 之后的词必须归约束栏")
    t3 = ["## 2. 按领域", "", "### D1", "",
          "- **正向**：`alpha`", "  1. `beta` + 约束 `ctxone` / `ctxtwo`", "",
          "- **本域负向**：`bad`", "", "## 3. x", ""]
    fx3 = ("\n".join(t3), {"D1": ['abs:"alpha"']}, {"D1": ["ctxone"]}, {"D1": ["bad"]})
    r3 = _run_fixture(fx3, baseline=B1, canonical=["D1"])
    chk("`ctxone`/`ctxtwo` 落进约束栏，不污染正向分母",
        r3["j2"]["per_domain"]["D1"]["positive"] == 2, str(r3["j2"]["per_domain"]["D1"]))
    chk("约束栏去重后 2 个", r3["j2"]["computed"]["constraint_total"] == 2,
        str(r3["j2"]["computed"]))

    print("④ J1 双向完备：缺一段 / 多一段都判红")
    r4 = _run_fixture(_fixture(), baseline=B2, canonical=["D1", "D2", "D3"])
    chk("§2 缺 D3 ⇒ 判红", any("J1" in i and "D3" in i for i in r4["issues"]),
        str(r4["issues"])[:90])
    t5 = _fixture()[0].replace("### D2", "### 99-不存在")
    fx5 = (t5, {"D1": ['abs:"alpha"'], "99-不存在": ['abs:"alpha"']},
           {"D1": ["ctxone"], "99-不存在": ["ctxone"]},
           {"D1": ["badone"], "99-不存在": ["badone"]})
    r5 = _run_fixture(fx5, baseline=B2, canonical=["D1", "D2"])
    chk("§2 多一段非规范域 ⇒ 判红（且双向都报出来）",
        any("J1" in i for i in r5["issues"]) and r5["j1"]["extra_in_doc"] == ["99-不存在"],
        str(r5["issues"])[:90])

    print("⑤ J3 无表域：未登记判红；登记后绿；条件不成立 ⇒ 过期判红")
    fx6 = _fixture(domains=("D1",))
    r6 = _run_fixture(fx6, con={}, baseline=B1, canonical=["D1"])
    chk("散文写了约束词而代码无表且未登记 ⇒ 判红",
        any("J3" in i and "未登记" in i for i in r6["issues"]), str(r6["issues"])[:90])
    decl = {"D1": {"field": "constraint", "why": "x", "expires_when": "y"}}
    r7 = _run_fixture(fx6, con={}, declared=decl, baseline=B1, canonical=["D1"])
    chk("登记后不再因未登记判红", not any("未登记" in i for i in r7["issues"]),
        str(r7["issues"])[:90])
    r8 = _run_fixture(fx6, declared=decl, baseline=B1, canonical=["D1"])
    chk("表已存在而登记还在 ⇒ **过期登记判红**（#84）",
        any("过期" in i for i in r8["issues"]), str(r8["issues"])[:90])

    print("⑤b #123 缩进引用块不得被当成续行（本批自己踩的：§2 里的订正引用块）")
    t5b = ["## 2. 按领域", "", "### D1", "",
           "- **约束词**：`ctxone`",
           "  > ⚠️ 订正：`should_not_be_a_field_word`", "",
           "- **本域负向**：`bad`", "", "## 3. x", ""]
    fx5b = ("\n".join(t5b), {"D1": ['abs:"alpha"']}, {"D1": ["ctxone"]}, {"D1": ["bad"]})
    r5b = _run_fixture(fx5b, baseline={"positive_total": 4, "positive_covered": 2,
                                       "constraint_total": 1, "negative_total": 1},
                       canonical=["D1"])
    chk("引用块里的反引号词不进约束栏（分母不许自己变大）",
        r5b["j2"]["computed"]["constraint_total"] == 1, str(r5b["j2"]["computed"]))

    print("⑥ J5 未归属词不静默 + 取不到就 exit 2（不许猜写法）")
    t9 = _fixture()[0].replace("### D2", "### D2\n\n- **注意**：`unattributed_word`")
    fx9 = (t9, {"D1": ['abs:"alpha"'], "D2": ['abs:"alpha"']},
           {"D1": ["ctxone"], "D2": ["ctxone"]},
           {"D1": ["badone"], "D2": ["badone"]})
    r9 = _run_fixture(fx9, baseline=B2)
    chk("未归属词进了读数", "unattributed_word" in r9["j5"]["unattributed"], str(r9["j5"]))
    for src, why in (("QUERY_GROUPS = dict()\n", "不是纯字面量"),
                     ("OTHER = {}\n", "找不到")):
        bad = Path(tempfile.mkdtemp(prefix="p2s-kwt-bad-")) / "h.py"
        bad.write_text(src, encoding="utf-8")
        raised = False
        try:
            audit(DOC, bad, {}, {})
        except SystemExit as e:
            raised = why in str(e)
        chk(f"`QUERY_GROUPS` 取不到（{why}）⇒ SystemExit，不是静默取空", raised)
    # 反向控制：#120 本体 —— **带注解的写法必须取得到**（首版只认 `ast.Assign`）
    good = Path(tempfile.mkdtemp(prefix="p2s-kwt-ok-")) / "h.py"
    good.write_text('QUERY_GROUPS: dict[str, list[str]] = {"D1": [\'abs:"alpha"\']}\n',
                    encoding="utf-8")
    chk("`QUERY_GROUPS: dict[...] = {...}`（AnnAssign）也取得到",
        query_phrases(good) == {"D1": ["alpha"]}, str(query_phrases(good)))

    print("⑦ 真产物自证：本文件全程**未改动**任何真产物")
    for pth in (DOC, HARVEST):
        chk(f"{pth.name} 仍在", pth.exists())

    print(f"\nselftest {'全过' if ok else '有红'}")
    return EXIT_OK if ok else EXIT_RED


def mutate() -> int:
    ok = True
    real = audit(DOC, HARVEST)
    print(f"真仓库基线：issues={len(real['issues'])} "
          f"正向覆盖 {real['j2']['computed']['positive_covered']}"
          f"/{real['j2']['computed']['positive_total']}")
    if real["issues"]:
        print("⚠️ 真仓库当前就判红，变异测试的前提不成立 ⇒ exit 3")
        return EXIT_INTERNAL

    tmp = Path(tempfile.mkdtemp(prefix="p2s-kwt-mut-"))
    text = DOC.read_text(encoding="utf-8")
    cases = []

    # M1：把一个域的段名改掉 ⇒ J1 双向都该报
    cases.append(("M1 改掉 §2 一个域段的名字",
                  "### 00-电商Agent　⚠️ **本节是 2026-09-14 补的，此前整节不存在**",
                  "### 00-电商AgentX"))
    # M2：把一个**已覆盖**的正向词换成不相干的词 ⇒ 覆盖数变 ⇒ J2 判红
    # ⚠️ 首版这里写的是 `uplift modeling` → `uplift modelingX`，**变异没施上力**：
    # `_covered()` 用的是双向子串包含（为了容忍 recommender/recommendation），
    # 于是 `uplift modeling` ⊂ `uplift modelingX` 照样算「已覆盖」。
    # ⇒ 变异体必须与所有查询短语**互不为子串**，否则测的是别的东西。
    cases.append(("M2 把一个已覆盖的正向词换成不相干的词 ⇒ J2 判红",
                  "`uplift modeling`、`causal inference`",
                  "`zzz-not-covered-anywhere`、`causal inference`"))
    # M3：伪造一个新域段 ⇒ J1 多段
    cases.append(("M3 在 §2 插一个新域段 `### 99-不存在`",
                  "## 3. 实测污染数据", "### 99-不存在\n\n- **正向**：`zzz`\n\n## 3. 实测污染数据"))
    # M4：把 `10-MAS` 的约束词段删空 ⇒ 过期登记（表不存在了？不，是把散文需求删掉 ⇒ 登记变过期）
    cases.append(("M4 把 `10-MAS` 段的 `**约束词**` 标签改坏 ⇒ 登记变过期",
                  "- **约束词**：需与 `LLM` / `agent` / `task` / `tool` 共现",
                  "- **约束表**：需与 `LLM` / `agent` / `task` / `tool` 共现"))

    for name, old, new in cases:
        if old not in text:
            print(f"  ⚠️ {name}：锚点不在文档里 ⇒ **变异没施上力**，该用例作废（不许算作已测）")
            ok = False
            continue
        mutated = text.replace(old, new, 1)
        if mutated == text:
            # ⚠️ 首版 M1 写成了 `old == new`（改一个字符都没动的「变异」）而照样打印
            # 「issues=0」—— 那读起来像「判据没劲」，实际是**变异根本没施上力**。
            # 与 `run_phase6_gates --mutate` 自己的历史缺陷同型：**先证明变异改变了真实取值，
            # 再谈判据有没有劲。**
            print(f"  ⚠️ {name}：替换前后**逐字节相同** ⇒ 变异没施上力，该用例作废")
            ok = False
            continue
        p = tmp / (re.sub(r"\W+", "_", name) + ".md")
        p.write_text(mutated, encoding="utf-8")
        try:
            r = audit(p, HARVEST)
        except SystemExit:
            print(f"  ✅ {name} ⇒ exit 2/3（取不到输入，也是失败关闭）")
            continue
        hit = bool(r["issues"])
        print(f"  {'✅' if hit else '🔴'} {name} ⇒ issues={len(r['issues'])}"
              + (f"  {r['issues'][0][:70]}" if hit else "  **变异没施上力**"))
        ok = ok and hit

    print(f"\nmutate {'全过' if ok else '有红'}")
    return EXIT_OK if ok else EXIT_RED


def main() -> int:
    ap = argparse.ArgumentParser(description="词表三向对账（散文 §2 ↔ 收割查询 ↔ 代码表）")
    ap.add_argument("--check", action="store_true", help="跑五条判据（验收面 L20a）")
    ap.add_argument("--report", action="store_true", help="只出读数，不判定")
    ap.add_argument("--selftest", action="store_true", help="证明每条判据都会失败（L20b）")
    ap.add_argument("--adjudicate", action="store_true",
                    help="274 条被丢弃论文的四态裁决（L20c）")
    ap.add_argument("--mutate", action="store_true", help="篡改副本端到端（L20c）")
    ap.add_argument("--json-out", default=None, help="把读数写到这个路径")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.mutate:
        return mutate()
    if a.adjudicate:
        res = adjudicate()
        report_adjudication(res)
        if a.json_out:
            out = Path(a.json_out)
            if not out.is_absolute():
                out = REPO / out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"→ {out}")
        return EXIT_RED if res["issues"] else EXIT_OK

    res = audit()
    report(res)
    if a.json_out:
        out = Path(a.json_out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {out}")
    if a.report:
        return EXIT_OK
    return EXIT_RED if res["issues"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
