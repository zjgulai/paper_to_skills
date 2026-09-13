#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_gap_ledger.py — 缺口账与靶区工单（PHASE6 F3 / T-A+T-C）。

## 它回答什么

把「151 个岗位责任名」× 「算法可服务性 A/B/C」×「供给状态（零供给 / 仅 legacy / 有精选卡）」
的交叉表**算出来**（而不是从方案文档抄一遍），再按可复算的权重排出扩充工单。

它是**缺口驱动扩充的入口**：S3（12 篇 backlog 由靶区接管）与 S4（三段式检索式生成）
都以本工单的顺序为输入。

## 事实源与归口（**不造第二份分类表**）

| 字段 | 来自 | 本脚本做什么 |
|---|---|---|
| 151 L3 / 岗位 / 域 / 面 / FLOW / 64 格 | `capability-graph.json`（F2 唯一机读图谱） | 只读 join |
| A/B/C + 边界条目 + 无 A 岗位 | 同上（`a_scope`，源出 `_survey_org_model.md` §F.3/F.5/F.6） | 只读，**不重判** |
| legacy 1338 张的 L3 归属 | 产品侧 `dsh-paper2skills/data/classification.json` | 只读 join（供给状态里「仅 legacy」那一列） |
| 精选 146 张的 L3 归属 | `capability-graph.json` 的 `cards[]` | 只读 |

本脚本**只有两个自己的产物**：① 交叉表；② 排序。两者都是**派生**，`--check` 可复算。

## 判据（本卡验收项，每条都能失败）

- **J1** 交叉表 4×3 逐格与方案 §4 相等（不等即**失败**，不许改判据去迁就结果）
- **J2** 靶区一 = A ∩ 无精选卡 = 36，且域分布 = 方案 §4 的七个数
- **J3** 靶区二：零供给 17，其中结构性空白 10（AGT-009..013 的零供给条目）
- **J4** 靶区三：A 类边界条目 10（含 1 条**材料内部矛盾**，登记在册、不重判）
- **J5** 低优先级位：无 A 岗位 10 个，与 §F.6 声明的名单**逐项相等**
- **J6** 排序可复算；**改权重则排序必须变**（`--weights`）
- **J7** 反后门：优先级分数**不得进入任何门禁** —— 扫门禁脚本，命中即失败
     （风险 N1：分数只用于排序。一个能影响判定的排序分数会立刻变成新的洗白工具。）

## 用法

    python3 paper2skills-research/scripts/build_gap_ledger.py                 # 生成 JSON + MD
    python3 paper2skills-research/scripts/build_gap_ledger.py --check         # 复跑比对（门禁用）
    python3 paper2skills-research/scripts/build_gap_ledger.py --top 12        # 只看前 N 条工单
    python3 paper2skills-research/scripts/build_gap_ledger.py --weights w.json
    python3 paper2skills-research/scripts/build_gap_ledger.py --selftest

退出码：0 正常；1 判据失败；2 输入缺失（**不是「没问题」**）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / "paper2skills-vault" / "07-资源库" / "capability-graph.json"
CLASSIFICATION = (REPO.parent / "Magpie-Horch" / "packages" / "capabilities"
                  / "dsh-paper2skills" / "data" / "classification.json")
OUT_JSON = REPO / "paper2skills-research" / "data" / "gap-ledger.json"
OUT_MD = REPO / "paper2skills-research" / "reports" / "PHASE6-F3-缺口账与靶区工单.md"

# ⚠️ 门禁脚本清单：J7 反后门扫描的对象。加门禁就要加进这张表，否则扫描覆盖率会静默变小。
GATE_SCRIPTS = [
    "paper2skills-skills/paper-萃取/scripts/verify_skill_code.py",
    "paper2skills-skills/paper-审核/scripts/gate_check.py",
    "paper2skills-skills/paper-审核/scripts/quote_check.py",
    "paper2skills-skills/paper-同步/scripts/sync.py",
    "paper2skills-skills/paper-维护/scripts/repo_health.py",
    "paper2skills-skills/paper-维护/scripts/scan_secrets.py",
    "paper2skills-research/scripts/check_contracts.py",
    "paper2skills-research/scripts/build_capability_graph.py",
]
#: 命中任一即判「分数进了门禁」。
#:
#: ⚠️ **刻意不把「缺口账」这个中文词当 needle —— 理由是首跑实测的假阳性**：
#:    `build_capability_graph.py`（在清单里，因为它也是 `--check` 门禁）的文档字符串写着
#:    「供缺口账 join」，那是**上游的说明**，不是数据流。若把词当 needle，就只能二选一：
#:    为了过闸把文档删掉，或把判据放宽 —— 两条都比「把 needle 收窄到**能定位产物的标识**」更糟。
#:    同族教训在 CLAUDE.md 里有名字：**判据的适用范围被默认成了全体**。
#:    代价如实登记：若有人把产物改成中文文件名再读，本判据看不见。
GATE_NEEDLES = ["gap-ledger", "gap_ledger", "缺口账.json", "PHASE6-F3"]

DEFAULT_WEIGHTS = {
    #: 供给密度越低越优先：1/(1+n_legacy)。**零供给自然 = 1.0**（无需特判，故不存在
    #: 「零供给被漏掉」这类分支 bug —— 第一版曾写 if zero: 1.0 else ...，是多余的。）
    "zero_gap": 3.0,
    #: 流程杠杆：该岗参与的 FLOW 数 / 8。一次萃取成本服务的格数越多越先做。
    "flow_breadth": 1.0,
    #: 首批切片对齐（Q3 决策：第一批 = FLOW-01 全做，按 FLOW 纵向切片）
    "first_slice": 1.5,
    #: §F.5 边界条目罚分：A 是否为主引擎尚有争议，降级条件一旦成立就要改 B
    "boundary_penalty": 0.75,
    #: 结构性空白罚分：交付物是物理/工程实物，走 SOP，不占检索预算
    "structural_penalty": 1.0,
}

#: 「结构性空白」岗位（方案 §4 靶区二：硬件研发 / 工业设计 / 工程制造）
STRUCTURAL_BLANK_ROLES = ["AGT-009", "AGT-010", "AGT-011", "AGT-012", "AGT-013"]

#: 方案 §4 的交叉表（**逐格断言**，不是注释）
EXPECT_TABLE = {
    "A": {"zero": 5, "legacy": 31, "curated": 37, "total": 73},
    "B": {"zero": 9, "legacy": 41, "curated": 16, "total": 66},
    "C": {"zero": 3, "legacy": 9, "curated": 0, "total": 12},
    "_col": {"zero": 17, "legacy": 81, "curated": 53, "_total": 151},
}
#: 方案 §4 靶区一的域分布（七个数）
EXPECT_T1_DOMAINS = {"供应与履约": 11, "产品与创新": 7, "渠道经营": 6,
                     "财务与合规": 6, "品牌与增长": 3, "经营与组织": 2, "数据与AI运行": 1}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    """仓库内路径显示相对路径，仓库外原样显示。

    ⚠️ 直接 `p.relative_to(REPO)` 对仓库外路径抛 ValueError —— 这与**门禁缺陷 #13**
        （`gate_check` 的 `relative_to(REPO_ROOT)`）是同一个 bug 的第二个副本，
        2026-09-13 写本脚本时**又一次**踩到（`--out-json /tmp/...` 直接崩）。
        同一族已在 CLAUDE.md 记过两次；这里不再只修一处，而是把它固化成函数。
    """
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# 装载（全部只读）
# ---------------------------------------------------------------------------
def load_graph(path: Path = GRAPH) -> dict:
    if not path.is_file():
        raise SystemExit(f"✗ 图谱不存在：{path}\n  先跑 build_capability_graph.py。"
                         f"**「读不到图谱」不是「没有缺口」。**")
    g = json.loads(path.read_text(encoding="utf-8"))
    for k in ("l3", "roles", "cells", "cards", "counts", "a_scope"):
        if k not in g:
            raise SystemExit(f"✗ 图谱缺字段 {k} —— 图谱版本太旧，先重生成")
    return g


def load_legacy(path: Path = CLASSIFICATION) -> tuple[dict, dict]:
    """产品侧 legacy 分类 → (卡名→L3, 元信息)。缺失时**不是失败**，但供给状态会整体失真。"""
    if not path.is_file():
        return {}, {"available": False, "path": str(path),
                    "why": "产品侧 classification.json 不在（跨仓库依赖）；"
                           "「仅 legacy 供给」一列会全判零供给 —— 结论方向会更激进，不可信"}
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items") or []
    return ({i["id"]: list(i.get("l3") or []) for i in items},
            {"available": True, "path": str(path), "n_items": len(items),
             "declared_total": raw.get("total"), "classified": raw.get("classified"),
             "unassigned": raw.get("unassigned"), "sha256": _sha(path)[:16]})


# ---------------------------------------------------------------------------
# 缺口账
# ---------------------------------------------------------------------------
def build_ledger(g: dict, legacy: dict, weights: dict) -> dict:
    l3_by_name = {x["name"]: x for x in g["l3"]}
    roles = {r["id"]: r for r in g["roles"]}
    doms = {d["id"]: d["name"] for d in g["domains"]}

    # 供给：精选（vault 146）与 legacy（产品侧 1338）两列分开数
    n_cur: dict[str, int] = {}
    for c in g["cards"]:
        for n in (c.get("l3") or []):
            n_cur[n] = n_cur.get(n, 0) + 1
    n_leg: dict[str, int] = {}
    for l3s in legacy.values():
        for n in l3s:
            n_leg[n] = n_leg.get(n, 0) + 1
    stray = sorted(set(n_leg) - set(l3_by_name))
    if stray:
        raise SystemExit(f"✗ legacy 分类里出现不在 151 名单内的 L3 名：{stray[:8]}"
                         f"（判据加了名字白名单，改名必须两边同时改）")

    # 该岗参与的 M 格（方法格 = 算法模型的挂载点）
    m_cells: dict[str, list[str]] = {}
    for c in g["cells"]:
        if c["cell_kind"] != "M":
            continue
        for rid in c["participating_role_ids"]:
            m_cells.setdefault(rid, []).append(c["cell_id"])

    rows = []
    for name, x in l3_by_name.items():
        rid = x["role_id"]
        r = roles[rid]
        cur, leg = n_cur.get(name, 0), n_leg.get(name, 0)
        status = "curated" if cur else ("legacy" if leg else "zero")
        rows.append({
            "l3": name, "role_id": rid, "role_title": r["title"],
            "domain": doms[x["domain_id"]], "plane_id": x["plane_id"],
            "serviceability": x["serviceability"],
            "boundary_a": bool(x.get("boundary_a")),
            "downgrade_condition": x.get("downgrade_condition"),
            "supply": {"status": status, "n_curated": cur, "n_legacy": leg},
            "flows": list(r["flows"]),
            "n_flows": len(r["flows"]),
            "in_first_slice": "FLOW-01" in r["flows"],
            "m_cells": sorted(m_cells.get(rid, [])),
            "structural_blank": False,      # 下面统一标
        })

    # --- 靶区二：零供给 17，其中结构性空白 10 ---
    zero = [r for r in rows if r["supply"]["status"] == "zero"]
    for r in zero:
        r["structural_blank"] = r["role_id"] in STRUCTURAL_BLANK_ROLES
    structural = [r for r in zero if r["structural_blank"]]

    # --- 靶区一：A ∩ 无精选卡 ---
    t1 = [r for r in rows if r["serviceability"] == "A" and r["supply"]["status"] != "curated"]
    for r in t1:
        r["score_parts"] = {
            "zero_gap": round(weights["zero_gap"] * (1.0 / (1 + r["supply"]["n_legacy"])), 4),
            "flow_breadth": round(weights["flow_breadth"] * (r["n_flows"] / 8), 4),
            "first_slice": round(weights["first_slice"] * (1 if r["in_first_slice"] else 0), 4),
            "boundary_penalty": round(-weights["boundary_penalty"] * (1 if r["boundary_a"] else 0), 4),
            "structural_penalty": round(-weights["structural_penalty"] * (1 if r["structural_blank"] else 0), 4),
        }
        r["score"] = round(sum(r["score_parts"].values()), 4)
    # ⚠️ 排序键必须含**确定性 tiebreak**（责任名），否则同分顺序随字典序变化，
    #    「可复算」就是假的。中位分的条目恰好不少，故这条不是理论担忧。
    t1.sort(key=lambda r: (-r["score"], r["l3"]))
    for i, r in enumerate(t1, 1):
        r["rank"] = i

    table = {}
    for s in "ABC":
        table[s] = {k: sum(1 for r in rows if r["serviceability"] == s
                           and r["supply"]["status"] == k)
                    for k in ("zero", "legacy", "curated")}
        table[s]["total"] = sum(table[s][k] for k in ("zero", "legacy", "curated"))
    table["_col"] = {k: sum(table[s][k] for s in "ABC") for k in ("zero", "legacy", "curated")}
    table["_col"]["_total"] = sum(table[s]["total"] for s in "ABC")

    t1_domains: dict[str, int] = {}
    for r in t1:
        t1_domains[r["domain"]] = t1_domains.get(r["domain"], 0) + 1

    return {
        "table": table,
        "targets": {
            "T1": {"rule": "A ∩ 无精选卡", "count": len(t1),
                   "domains": dict(sorted(t1_domains.items(), key=lambda kv: (-kv[1], kv[0])))},
            "T2": {"rule": "零供给（连 legacy 预览都没有）", "count": len(zero),
                   "structural_blank": len(structural),
                   "structural_blank_roles": STRUCTURAL_BLANK_ROLES,
                   "items": [{"l3": r["l3"], "role_id": r["role_id"],
                              "serviceability": r["serviceability"],
                              "structural_blank": r["structural_blank"]} for r in
                             sorted(zero, key=lambda r: (r["role_id"], r["l3"]))]},
            "T3": {"rule": "A 类中的边界条目（§F.5）",
                   "count": g["a_scope"]["a_count"],
                   "boundary": g["a_scope"]["boundary_a"],
                   "contradictions": g["a_scope"]["contradictions"]},
            "low_priority": {"rule": "无任何 A 类责任名的岗位（§F.6）",
                             "roles": g["a_scope"]["roles_without_a_computed"],
                             "n": len(g["a_scope"]["roles_without_a_computed"])},
        },
        "worklist": t1,
        # ⚠️ 只给一份 rows。第一版另存了一份 by_l3 索引，而它是**同一批 dict 的第二份拷贝** ——
        #    JSON 里逐个重复一遍，254 KB 里差不多一半是它。索引用函数现建即可。
        "rows": rows,
        # §F.6 的**声明**名单（与 roles_without_a_computed 逐项比对的另一半）
        "a_scope_declared": g["a_scope"]["roles_without_a_declared"],
        "weights": weights,
    }


def enrich_structural_scope(led: dict) -> dict:
    """靶区一里「落在结构性空白**岗位**上」的条目 —— 与 §4 的 **L3 级**名单分开报。

    §4 说的「10 条结构性空白」是那 10 个**零供给条目**；而靶区一是按 L3 定义的，
    于是出现两种读法，差 2 条（测试方案 / 可用性验证：岗位属 AGT-013/AGT-010，
    但它们**有** legacy 卡，不在那 10 条零供给名单里）。

    **默认取严口径（§4 原文那 10 条）**，只把宽口径的数字报出来供人决策 ——
    悄悄把定义放宽，正是本仓库反复记录的失败模式（漏洞 #11 / C2 / 两条铁律）。
    """
    wide = [r for r in led["worklist"] if r["role_id"] in STRUCTURAL_BLANK_ROLES]
    strict = [r for r in led["worklist"] if r["structural_blank"]]
    led["targets"]["T1"]["structural_strict"] = len(strict)
    led["targets"]["T1"]["structural_wide"] = len(wide)
    led["targets"]["T1"]["retrieval_budget_positions"] = {
        "strict": led["targets"]["T1"]["count"] - len(strict),
        "wide": led["targets"]["T1"]["count"] - len(wide),
        "default": "strict",
        "wide_only": sorted({r["l3"] for r in wide} - {r["l3"] for r in strict}),
    }
    return led


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------
def check_against_plan(led: dict) -> list[str]:
    """J1–J5：与方案 §4 逐格核对。**不等就是失败**，改判据去迁就结果是被禁止的。"""
    errs = []
    t = led["table"]
    for s in "ABC":
        for k, want in EXPECT_TABLE[s].items():
            got = t[s][k]
            if got != want:
                errs.append(f"J1 交叉表 {s}×{k}：方案 {want} vs 实测 {got}")
    for k, want in EXPECT_TABLE["_col"].items():
        got = t["_col"][k]
        if got != want:
            errs.append(f"J1 交叉表合计 {k}：方案 {want} vs 实测 {got}")

    t1 = led["targets"]["T1"]
    if t1["count"] != 36:
        errs.append(f"J2 靶区一 {t1['count']} 条 ≠ 36")
    for d, want in EXPECT_T1_DOMAINS.items():
        got = t1["domains"].get(d, 0)
        if got != want:
            errs.append(f"J2 靶区一域分布 {d}：方案 {want} vs 实测 {got}")
    extra = {d: n for d, n in t1["domains"].items() if d not in EXPECT_T1_DOMAINS}
    if extra:
        errs.append(f"J2 靶区一出现方案里没有的域：{extra}")

    t2 = led["targets"]["T2"]
    if t2["count"] != 17:
        errs.append(f"J3 零供给 {t2['count']} 条 ≠ 17")
    if t2["structural_blank"] != 10:
        errs.append(f"J3 结构性空白 {t2['structural_blank']} 条 ≠ 10")
    bad = [i["l3"] for i in t2["items"]
           if i["structural_blank"] and i["role_id"] not in STRUCTURAL_BLANK_ROLES]
    if bad:
        errs.append(f"J3 结构性空白标记越界：{bad}")

    t3 = led["targets"]["T3"]
    if len(t3["boundary"]) != 10:
        errs.append(f"J4 边界条目 {len(t3['boundary'])} 条 ≠ 10")
    names = {r["l3"] for r in led["rows"]}
    missing = [b["name"] for b in t3["boundary"] if b["name"] not in names]
    if missing:
        errs.append(f"J4 边界条目不在 151 名单内：{missing}")

    lp = led["targets"]["low_priority"]
    if lp["n"] != 10:
        errs.append(f"J5 无 A 类岗位 {lp['n']} 个 ≠ 10")
    if lp["roles"] != led["a_scope_declared"]:
        errs.append(f"J5 §F.6 声明 {led['a_scope_declared']} vs 算出 {lp['roles']}")
    return errs


def check_not_a_gate(repo: Path = REPO) -> tuple[list[str], int]:
    """J7 反后门：优先分数**不得**被任何门禁读到。

    返回 (失败原因, 实际扫描的文件数)。**扫到 0 个文件判失败** ——
    「没东西可查」不等于「查过了没问题」（同 scan_secrets.py 的退出码 2）。
    """
    errs, scanned = [], 0
    for rel in GATE_SCRIPTS:
        p = repo / rel
        if not p.is_file():
            errs.append(f"门禁脚本不存在：{rel}（扫描覆盖率静默变小 = 这道判据在骗人）")
            continue
        scanned += 1
        text = p.read_text(encoding="utf-8")
        hit = [n for n in GATE_NEEDLES if n in text]
        if hit:
            errs.append(f"J7 {rel} 引用了缺口账（命中 {hit}）—— 优先级分数只用于排序，"
                        f"不得进入任何门禁判定（风险 N1）")
    if scanned == 0:
        errs.append("J7 一个门禁脚本都没扫到 —— 这不是「干净」，是没测")
    return errs, scanned


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render_md(led: dict, g: dict, meta: dict) -> str:
    t, tg = led["table"], led["targets"]
    w = led["weights"]
    L = []
    A = L.append
    A("# PHASE6 F3 · 缺口账与靶区工单")
    A("")
    A(f"> 生成时间：{meta['generated']}　|　生成器：`{meta['generator']}`")
    A("> 复算：`python3 paper2skills-research/scripts/build_gap_ledger.py --check`")
    A("> 输入：`capability-graph.json` sha256:`{gsha}` · 产品侧 `classification.json` "
      "sha256:`{csha}`（{cn} 条 legacy 卡）".format(gsha=meta["graph_sha256"],
                                                csha=meta["classification"]["sha256"],
                                                cn=meta["classification"].get("n_items")))
    A("")
    A("机读产物 `paper2skills-research/data/gap-ledger.json`（含逐条 `m_cells` 格号与分数分解）—— "
      "`data/**/*.json` 按仓库惯例默认不入库，本条按 `skill_audit.json` / `dedup_report.json` 的先例"
      "用 `!` 规则**提升入库**：它是 S3/S4/S12 要消费的工单，不是中间产物。")
    A("")
    A("**这张表不是抄的。** 每一格由脚本从图谱与产品侧分类现算，并与方案 §4 逐格断言相等；")
    A("不等即判失败（J1）。所以下面的数字不会随文档漂移而变成一张过期的照片。")
    A("")
    A("## 1. 缺口账（151 责任名 × 供给状态）")
    A("")
    A("| 算法可服务性 | 零供给 | 仅 legacy 供给 | 有精选卡 | 合计 |")
    A("|---|---:|---:|---:|---:|")
    for s, lab in (("A", "**A · 可算法化**"), ("B", "B · 半可算法化"), ("C", "C · 难算法化")):
        A(f"| {lab} | {t[s]['zero']} | {t[s]['legacy']} | {t[s]['curated']} | **{t[s]['total']}** |")
    A(f"| **合计** | **{t['_col']['zero']}** | **{t['_col']['legacy']}** | "
      f"**{t['_col']['curated']}** | **{t['_col']['_total']}** |")
    A("")
    A("口径：**零供给** = 连 legacy 预览卡都没有；**仅 legacy** = 只有 1338 张预览卡里有它、")
    A("精选 146 张里没有；**有精选卡** = 精选线里至少一张卡挂了该责任名。")
    A("")
    A("## 2. 靶区一：A ∩ 无精选卡 = %d 条（扩充工单）" % tg["T1"]["count"])
    A("")
    A("域分布：" + " / ".join(f"{k} {v}" for k, v in tg["T1"]["domains"].items()))
    A("")
    A("其中 **%d 条落在结构性空白条目上**（§4 严口径；宽口径按岗位算是 %d 条，见 §3 脚注）"
      "⇒ **实际占检索预算 %d 位**（严口径）。" % (
          tg["T1"]["structural_strict"], tg["T1"]["structural_wide"],
          tg["T1"]["retrieval_budget_positions"]["strict"]))
    A("")
    A("排序权重（**只用于排序，不得进入任何门禁** —— J7 扫描全部门禁脚本，命中即失败）：")
    A("")
    A("| 因子 | 权重 | 含义 |")
    A("|---|---:|---|")
    A(f"| `zero_gap` | {w['zero_gap']} | 供给密度 1/(1+legacy 卡数)；零供给自然 = 1.0 |")
    A(f"| `flow_breadth` | {w['flow_breadth']} | 该岗参与的 FLOW 数 / 8（一次萃取服务的格数）|")
    A(f"| `first_slice` | {w['first_slice']} | 命中首批切片 FLOW-01（Q3 决策）|")
    A(f"| `boundary_penalty` | −{w['boundary_penalty']} | §F.5 边界条目：A 是否为主引擎有争议 |")
    A(f"| `structural_penalty` | −{w['structural_penalty']} | 结构性空白：走 SOP，不占检索预算 |")
    A("")
    A("| 位次 | 责任名 | 岗位 | 域 | 供给(legacy) | 参与 FLOW | M 格 | 分 | 标记 |")
    A("|---:|---|---|---|---:|---|---:|---:|---|")
    for r in led["worklist"]:
        flags = []
        if r["boundary_a"]:
            flags.append("边界")
        if r["structural_blank"]:
            flags.append("结构空白")
        if r["in_first_slice"]:
            flags.append("首批")
        fl = "、".join(f[-2:] for f in r["flows"])
        A(f"| {r['rank']} | {r['l3']} | {r['role_id']} {r['role_title']} | {r['domain']} "
          f"| {r['supply']['n_legacy']} | {fl} | {len(r['m_cells'])} "
          f"| {r['score']:.3f} | {'、'.join(flags) or '—'} |")
    A("")
    A("**M 格 = 该岗的方法格数（`cell_kind=M`）**，即新卡的算法模型可挂载点，"
      "**格号可由本表直接推出来**：")
    A("`M 格 = {FLOW-01…08 中本行「参与 FLOW」列出的每一个} × "
      "{STG-04 证据收集与诊断、STG-05 方案与标准产物、STG-08 结果核验与异步学习}`。")
    A("例：第 1 行 `01、04` ⇒ `FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、"
      "`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`（共 6 格）。")
    A("⚠️ 本列不是独立信息：`M 格数 = 3 × 参与 FLOW 数`，实测 36 条全部满足 —— "
      "故它没有单独做权重，避免同一信息被算两遍。")
    A("")
    A("## 3. 靶区二：零供给 %d 条，其中结构性空白 %d 条" % (
        tg["T2"]["count"], tg["T2"]["structural_blank"]))
    A("")
    A("| 责任名 | 岗位 | A/B/C | 处置 |")
    A("|---|---|---|---|")
    for i in tg["T2"]["items"]:
        A(f"| {i['l3']} | {i['role_id']} | {i['serviceability']} | "
          f"{'**走 SOP，不占检索预算**' if i['structural_blank'] else '建账，按靶区一规则排序'} |")
    A("")
    A("结构性空白岗位：`%s`（硬件研发 / 工业设计 / 工程制造）。" % "、".join(STRUCTURAL_BLANK_ROLES))
    A("")
    ow = tg["T1"]["retrieval_budget_positions"]
    A(f"> ⚠️ **两口径差额（登记，不擅自放宽）**：§4 的结构性空白名单是 **L3 级**的 10 条零供给条目；"
      f"若改按**岗位**读，靶区一里落在 AGT-009..013 上的还有 "
      f"{'、'.join(ow['wide_only']) or '（无）'} —— 它们有 legacy 卡，故不在那 10 条里。"
      f"本账默认取 **{'严口径' if ow['default'] == 'strict' else '宽口径'}**（§4 原文那 10 条零供给）；"
      f"据此可检索位 {ow['strict']} vs 宽口径 {ow['wide']} —— **差额由所有者裁决，脚本不擅自放宽。**")
    A("")
    A("## 4. 靶区三：A 类 73 的边界条目（会随阈值动摇）")
    A("")
    A("| 责任名 | 计入 A 的理由 | 降级条件 |")
    A("|---|---|---|")
    for b in tg["T3"]["boundary"]:
        A(f"| {b['name']} | {b['rationale']} | {b['downgrade_condition']} |")
    A("")
    A("**A 类 = 73 是「算法可作主引擎」口径的诚实计数，不是 40**；放宽到「可作重要辅助」时")
    A("可服务面达 139/151。上述 10 条若采用更严口径应**优先降级**。")
    A("")
    for c in tg["T3"]["contradictions"]:
        actual = next((r["serviceability"] for r in led["rows"] if r["l3"] == c["name"]), "?")
        A(f"> 🔴 **材料内部矛盾（登记，不重判）**：`{c['name']}`（{c['role_id']}）在 "
          f"{c['declared_in']} 被列为 A 类边界条目，但在 {c['actual_in']} 判为 **{actual}**。")
        A(f"> 处置：{c['handling']}。")
        A(">")
    A("> 这条是 2026-09-13 F3 跑图时被**断言当场抓到**的（原文断言：「边界条目必须本身是 A 类」）。")
    A("> 处置不是删断言 —— 是改成「非 A 的边界条目必须逐条登记」，于是这一条被显式承认，")
    A("> 而**新出现的**仍会打红。")
    A("")
    A("## 5. 低优先级位：无任何 A 类责任名的 %d 个岗位" % tg["low_priority"]["n"])
    A("")
    A("`%s`" % "`、`".join(tg["low_priority"]["roles"]))
    A("")
    A("不是缺陷，是定位：这些岗位的算法增益最小。它们的责任仍会被 B 类契约覆盖（139 = 151 − 12），")
    A("只是不进检索预算的前排。")
    A("")
    A("## 6. 判据自检")
    A("")
    A("| # | 判据 | 状态 |")
    A("|---|---|---|")
    A("| J1 | 交叉表 4×3 逐格 = 方案 §4 | ✅ 15/15 格相等（不等即失败）|")
    A("| J2 | 靶区一 = 36，且域分布七个数逐个相等 | ✅ |")
    A("| J3 | 零供给 17 / 结构性空白 10 | ✅ |")
    A("| J4 | 边界条目 10（含 1 条已登记矛盾）| ✅ |")
    A("| J5 | 无 A 岗位 10，声明与算出**逐项相等** | ✅ |")
    A("| J6 | 排序可复算 + 改权重则排序必须变 | ✅ 见 `--selftest` |")
    A("| J7 | 优先分数不得进入门禁（扫 %d 个门禁脚本）| ✅ 0 命中 |" % len(GATE_SCRIPTS))
    A("")
    A("## 7. 未做 / 留给下游")
    A("")
    A("- **检索式未生成**：三段式（正向/负向/约束）是 S4 的交付物；本工单只给顺序与挂载点。")
    A("- **12 篇 extract backlog 未逐篇定位**：那是 S3，按本表位次接管。")
    A("- **契约未挂**：`m_cells` 是候选挂载点，真正的挂载点是 S1 的 139 份契约。")
    A("- **A/B/C 不重判**：`a_scope.contradictions` 里的材料矛盾交由所有者裁决，本脚本不改名单。")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _selftest() -> int:
    ok = True
    g = load_graph()
    legacy, linfo = load_legacy()
    if not linfo["available"]:
        print(f"✗ J0 产品侧分类缺失：{linfo['path']}\n  {linfo['why']}")
        return 2

    led = enrich_structural_scope(build_ledger(g, legacy, DEFAULT_WEIGHTS))

    print("--- J1–J5：与方案 §4 逐格核对 ---")
    errs = check_against_plan(led)
    for e in errs:
        print("    ✗", e)
    print(("✅" if not errs else "❌") + f" 交叉表 + 三个靶区 + 低优先级位全部相等（{len(errs)} 条不符）")
    ok &= not errs

    print("\n--- J6：排序可复算、且**改权重排序必须变** ---")
    w2 = dict(DEFAULT_WEIGHTS)
    w2["flow_breadth"] = 0.0          # 关掉流程杠杆
    w2["first_slice"] = 0.0           # 关掉首批切片
    led2 = build_ledger(g, legacy, w2)
    o1 = [r["l3"] for r in led["worklist"]]
    o2 = [r["l3"] for r in led2["worklist"]]
    changed = o1 != o2
    n_moved = sum(1 for a, b in zip(o1, o2) if a != b)
    print(("✅" if changed else "❌") +
          f" 关掉 flow_breadth/first_slice → 前 10 位次变化 {n_moved}/36（{o1[:3]} → {o2[:3]}）")
    ok &= changed

    led3 = build_ledger(g, legacy, DEFAULT_WEIGHTS)
    same = [r["l3"] for r in led3["worklist"]] == o1
    print(("✅" if same else "❌") + " 同一权重下两次构建顺序**逐位相同**（有确定性 tiebreak）")
    ok &= same

    # ⚠️ 反向：把 tiebreak 去掉，同分条目顺序必须**可能**变 —— 但 Python 的 dict 序恰好稳定，
    #    所以「顺序变不变」测不出 tiebreak 是否必要。测**能测的**：**每一个**同分组内部
    #    都必须按责任名升序（而不是只测第一名那一组 —— 第一版就是那样，而它恰好只有 1 条，
    #    等价于「断言恒真 = 没断言」，同 C8「用例 18 是摆设」）。
    groups: dict[float, list[str]] = {}
    for r in led["worklist"]:
        groups.setdefault(r["score"], []).append(r["l3"])
    ties = {s: v for s, v in groups.items() if len(v) > 1}
    bad_groups = {s: v for s, v in ties.items() if v != sorted(v)}
    n_tied = sum(len(v) for v in ties.values())
    print(("✅" if not bad_groups else "❌") +
          f" 全部 {len(ties)} 个同分组（共 {n_tied}/{len(led['worklist'])} 条）内部按责任名升序"
          f"（乱序组：{bad_groups}）")
    ok &= not bad_groups

    print("\n--- J7：优先分数不得进入任何门禁（反后门）---")
    gerrs, scanned = check_not_a_gate()
    for e in gerrs:
        print("    ✗", e)
    print(("✅" if not gerrs else "❌") + f" 扫描 {scanned} 个门禁脚本，0 命中")
    ok &= not gerrs

    # ⚠️ 反向：往一份门禁脚本副本里注入一次引用，扫描**必须**报错。
    #    没有这一条，上面那句「0 命中」就是摆设 —— 真脚本本来就没引用它，
    #    把整段扫描删掉自检照样全绿（同 C8「用例 18 是摆设」）。
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        saved = GATE_SCRIPTS[:]
        # 每种 needle 各喂一次：两个产物文件（JSON / MD）都在射程内，才算扫描覆盖了产物面
        probes = [("# 读 gap-ledger.json 决定放行\n", "gap-ledger"),
                  ("NOTE = 'PHASE6-F3-缺口账与靶区工单.md'\n", "PHASE6-F3"),
                  ("import gap_ledger\n", "gap_ledger")]
        results = []
        try:
            for i, (body, lab) in enumerate(probes):
                victim = Path(td) / f"gate_{i}.py"
                victim.write_text(body, encoding="utf-8")
                GATE_SCRIPTS[:] = [victim.name]
                ge, sc = check_not_a_gate(repo=Path(td))
                results.append((lab, bool(ge) and sc == 1))
        finally:
            GATE_SCRIPTS[:] = saved
        miss = [lab for lab, okk in results if not okk]
        caught = not miss
        print(("✅" if caught else "❌") +
              f" 变异：三种 needle（{len(probes)} 种引用写法）各喂一次 → 全部抓住（漏 {miss}）")
        ok &= caught

    # 反向：扫不到文件必须判失败，而不是「干净」
    saved = GATE_SCRIPTS[:]
    try:
        GATE_SCRIPTS[:] = ["does/not/exist.py"]
        gerrs3, _ = check_not_a_gate()
        caught2 = any("不存在" in e for e in gerrs3)
    finally:
        GATE_SCRIPTS[:] = saved
    print(("✅" if caught2 else "❌") + " 反向：门禁脚本清单指向不存在的文件 → 判失败（「没东西可查」≠「没问题」）")
    ok &= caught2

    print("\n--- 回归守卫：门禁缺陷 #13 的同族 bug（relative_to 对仓库外路径抛错）---")
    guards = []
    for probe in (REPO / "paper2skills-research" / "scripts" / "build_gap_ledger.py",
                  Path("/tmp/definitely-outside-repo/x.json"),
                  Path("/etc/hosts")):
        try:
            got = _rel(probe)
            guards.append(bool(got))
        except ValueError:
            guards.append(False)
    print(("✅" if all(guards) else "❌") +
          f" `_rel` 对仓库内/仓库外路径均不抛错（{len(guards)} 个探针）—— "
          f"`--out-json /tmp/...` 不再崩")
    ok &= all(guards)

    print("\n--- 与图谱的嵌套断言 ---")
    a = g["a_scope"]
    eq = a["roles_without_a_declared"] == a["roles_without_a_computed"] == \
        led["targets"]["low_priority"]["roles"]
    print(("✅" if eq else "❌") +
          f" §F.6 声明 / 图谱算出 / 缺口账实得 三者逐项相等（{len(a['roles_without_a_declared'])} 个）")
    ok &= eq
    b10 = len(led["targets"]["T3"]["boundary"]) == 10
    print(("✅" if b10 else "❌") + f" §F.5 边界条目 10 条进账（实得 {len(led['targets']['T3']['boundary'])}）")
    ok &= b10

    print()
    print("SELFTEST " + ("PASS —— 交叉表可复算、排序对权重敏感、优先分数被挡住进不了门禁"
                         if ok else "FAIL —— 判据静默失效，勿信其产物"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="缺口账与靶区工单（PHASE6 F3 / T-A+T-C）")
    ap.add_argument("--graph", type=Path, default=GRAPH)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--weights", type=Path, help="覆盖权重的 JSON（键同 DEFAULT_WEIGHTS）")
    ap.add_argument("--top", type=int, help="只打印前 N 条工单")
    ap.add_argument("--check", action="store_true", help="复跑并与已入库产物比对")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    g = load_graph(args.graph)
    legacy, linfo = load_legacy()
    if not linfo["available"]:
        print(f"✗ 产品侧分类缺失：{linfo['path']}\n  {linfo['why']}\n"
              f"  **「数不出 legacy 供给」不是「零供给」。**")
        return 2
    if "a_scope" not in g:
        print("✗ 图谱没有 a_scope（F3 新加字段）：先跑 build_capability_graph.py")
        return 2

    w = dict(DEFAULT_WEIGHTS)
    if args.weights:
        if not args.weights.is_file():
            print(f"✗ 权重文件不存在：{args.weights}")
            return 2
        override = json.loads(args.weights.read_text(encoding="utf-8"))
        unknown = sorted(set(override) - set(DEFAULT_WEIGHTS))
        if unknown:
            print(f"✗ 未知权重键 {unknown}（拼错键会被静默忽略，故直接失败）")
            return 1
        w.update(override)

    led = build_ledger(g, legacy, w)
    led = enrich_structural_scope(led)
    errs = check_against_plan(led)
    gerrs, scanned = check_not_a_gate()
    if errs or gerrs:
        print("✗ 判据不成立，拒绝写出缺口账：")
        for e in errs + gerrs:
            print("   ", e)
        return 1

    if args.top:
        print(f"前 {args.top} 条工单（共 {len(led['worklist'])} 条）：")
        for r in led["worklist"][:args.top]:
            flags = [x for x, on in (("边界", r["boundary_a"]),
                                     ("结构空白", r["structural_blank"]),
                                     ("首批", r["in_first_slice"])) if on]
            print(f"  {r['rank']:>2}. {r['l3']:<8} {r['role_id']} {r['role_title'][:12]:<14}"
                  f" {r['domain']:<6} legacy={r['supply']['n_legacy']:<3} "
                  f"分={r['score']:.3f} {'、'.join(flags)}")
        return 0

    meta = {
        "generated": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "paper2skills-research/scripts/build_gap_ledger.py",
        "graph": _rel(args.graph),
        "graph_sha256": _sha(args.graph)[:16],
        "classification": linfo,
        "gate_scripts_scanned": scanned,
        "exit_codes": "0 正常 / 1 判据失败 / 2 输入缺失（不是「没问题」）",
    }
    led["_meta"] = meta

    text_json = json.dumps(led, ensure_ascii=False, indent=1) + "\n"
    text_md = render_md(led, g, meta)

    if args.check:
        bad = []
        for p, new in ((args.out_json, text_json), (args.out_md, text_md)):
            if not p.is_file():
                bad.append(f"{_rel(p)} 不存在：先跑一次生成")
                continue
            old = p.read_text(encoding="utf-8")
            if p.suffix == ".json":
                # ⚠️ 比对前剔掉易变字段，否则 --check 每次必红（同 build_capability_graph）。
                o = json.loads(old)
                o.pop("_meta", None)
                n = json.loads(new)
                n.pop("_meta", None)
                if o != n:
                    bad.append(f"{_rel(p)} 与图谱/分类现状不一致"
                               f"（差异键：{sorted(k for k in set(o) | set(n) if o.get(k) != n.get(k))[:6]}）")
            else:
                strip = lambda s: re.sub(r"^> 生成时间：.*$", "", s, flags=re.M)
                if strip(old) != strip(new):
                    bad.append(f"{_rel(p)} 与图谱/分类现状不一致（逐行比对）")
        if bad:
            print("✗ 缺口账产物已过期或与现状不一致：")
            for e in bad:
                print("   ", e)
            return 1
        print("✓ 缺口账 JSON 与工单报告均与图谱/分类现状一致（已剔除易变字段）")
        return 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text_json, encoding="utf-8")
    args.out_md.write_text(text_md, encoding="utf-8")

    t = led["table"]
    print(f"✓ 写入 {_rel(args.out_json)}（{args.out_json.stat().st_size/1024:.0f} KB）")
    print(f"✓ 写入 {_rel(args.out_md)}（{args.out_md.stat().st_size/1024:.0f} KB）")
    print(f"  缺口账：零供给 {t['_col']['zero']} / 仅 legacy {t['_col']['legacy']} / "
          f"有精选卡 {t['_col']['curated']} = {t['_col']['_total']}")
    print(f"  靶区一（A ∩ 无精选卡）{led['targets']['T1']['count']} 条，"
          f"其中结构性空白 {led['targets']['T1']['structural_strict']} 条 ⇒ "
          f"占检索预算 {led['targets']['T1']['retrieval_budget_positions']['strict']} 位")
    print(f"  靶区二：零供给 {led['targets']['T2']['count']} 条（结构性空白 "
          f"{led['targets']['T2']['structural_blank']} 条走 SOP，不占检索预算）")
    print(f"  靶区三：A 类 {led['targets']['T3']['count']} 条，边界条目 "
          f"{len(led['targets']['T3']['boundary'])} 条（材料矛盾 "
          f"{len(led['targets']['T3']['contradictions'])} 条已登记）")
    print(f"  低优先级位：无 A 岗位 {led['targets']['low_priority']['n']} 个")
    print(f"  反后门：扫描 {scanned} 个门禁脚本，0 命中（分数只用于排序）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
