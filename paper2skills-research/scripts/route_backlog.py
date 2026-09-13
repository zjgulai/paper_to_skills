#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""route_backlog.py — extract backlog 交给靶区排序接管（PHASE6 S3 / 任务板 B9）。

## 它回答什么

仓库里有一批**还没出卡的论文**（extract backlog）。PHASE6 之前它们按「论文到货顺序 + 论文自己的分数」
排期；本脚本把这批论文改成由 **F3 的缺口账 / 靶区工单**决定顺序：

    论文 → 它能服务的 L3 责任 → 该 L3 在 F3 靶区一里的位次 → 论文的新位次

**排序由缺口驱动，不由论文自身的分数驱动。** 论文的 `score` / `venue_tier` 只能出现在
「同一个靶区位次内部」的末位 tiebreak 上（而那是**旧顺序**本身），且脚本有一条判据
（J5）专门盯着这件事：既扫自己的排序键，也拿「按论文分数排序」的注入样本自证两者**必须不同**。

## 事实源与归口（**不造第二份事实源**）

| 输入 | 来自 | 本脚本做什么 |
|---|---|---|
| backlog 是哪 12 篇 | `papers_registry.json`（论文唯一事实源）**现算** | 按规则推导，不抄清单 |
| 151 L3 / 岗位 / A·B·C / 供给状态 | `gap-ledger.json`（F3 产物，来自 `capability-graph.json`） | 只读 |
| 靶区位次 1..31 | 同上 | 只读 + **现算复现**（同公式同权重，不另立口径） |
| 论文 ↔ L3 责任 | `data/backlog-l3-map.json`（本卡唯一人工判断层） | 逐条校验白名单 + **逐字证据** |

⚠️ 本脚本**不生成缺口账、不改图谱、不碰 registry、不抓论文、不出卡**。它是**排序接管**，
不是扩充执行。`build_gap_ledger.py` / `build_search_queries.py` 只被**只读**使用
（前者被 import 当作 oracle，见 J4）。

## 判据（每条都配篡改样本，见 `--selftest`）

- **J1** 映射完整性：每篇 backlog 都能落到 ≥1 个 L3；L3 在 151 名单内；serviceability ∈ {A,B}
  （C 类不放行，Q4 弱门槛）；`business_action` 的证据是该字段的**逐字子串**；
  `capability_inference` 必须显式声明且 confidence=low。**无孤儿条目**（映射里多出来的论文要报红）。
- **J2** 覆盖率是一等输出：分母 = backlog 实测篇数；未落 L3 的**具名列出**（不静默丢弃）；
  `n_mapped + n_unassigned == n_backlog`；**输入为空 ⇒ 退出码 2**（「没东西可查」≠「查过了没问题」）。
- **J3** 缺口驱动：默认权重下位次 == F3 已入库工单；**改靶区权重则论文位次必须变**（承接 F3 的 J6 口径），
  改不动即判红；被旧顺序（分数）决定的位次数**单独报出**。
- **J4** 与 F3 一致：① 默认权重下现算的 31 条位次与分数，与已入库 `gap-ledger.json` **逐条相等**；
  ② 任意权重下与 `build_gap_ledger.py` **现场跑**的结果逐条相等（判据只有一处实现）；
  ③ 论文用到的每个靶区位次都等于 F3 给该 L3 的位次；④ 靶区外论文不得排在靶区内论文之前。
- **J5** 反后门：AST 扫本脚本自己的排序键（允许表/禁止表 + 反向控制），
  并喂一份「按论文分数排序」的注入样本自证顺序**必须不同**。
- **J6** 反向控制：干净输入必须 exit 0（篡改样本之外的正常路径也要被测）。
- **J7** registry 自洽（N4）：`outputs.skill_card` 上限 15 篇、扣掉 3 篇增强交付才是 12 篇；
  `status` 标签与实物（卡文件是否存在 / `enhanced_cards` 是否非空）必须一致，不一致判红。
- **J8** 接管必须**真的改变**了旧顺序（双边：`新序 == 旧序` 也是要报红的结论，不是「无事发生」）。
- **J9** 输入新鲜度（风险 N6）：`gap-ledger.json` 声明的图谱 sha 与现状不符时，
  用 F3 **现算**（只读、不落盘）核对 151 行/31 条工单是否有**实质**差异 ——
  有 ⇒ 判红；无 ⇒ 作为一等输出登记「过期但无实质影响」；没声明 ⇒ 不许当新鲜。

## 本卡自己撞出的三个缺陷（已修，登记在报告 §4.4）

1. 「层级倒挂」篡改样本原先把列表**重排**了 ⇒ 等于把证据自己擦掉，判据看起来没抓住（**假失败**）。
2. `--no-f3` 与「改权重位次必须变」原本写在同一个 `if` 里 ⇒ 关掉现场核对会**静默关掉一条判据**。
3. `naive` 计数第一版恰好等于真值 12 ⇒ **N4 那条差额在账上消失**（判据因数据恰好满足而成为摆设）。

## 用法

    python3 paper2skills-research/scripts/route_backlog.py                 # 生成 JSON + MD
    python3 paper2skills-research/scripts/route_backlog.py --check         # 复跑比对（门禁用）
    python3 paper2skills-research/scripts/route_backlog.py --selftest      # 判据 × 篡改样本
    python3 paper2skills-research/scripts/route_backlog.py --weights w.json
    python3 paper2skills-research/scripts/route_backlog.py --json-out /tmp/x.json

退出码：**0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 内部错误（≠ 判红）**。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "paper2skills-vault" / "07-资源库" / "papers_registry.json"
GRAPH = REPO / "paper2skills-vault" / "07-资源库" / "capability-graph.json"
LEDGER = REPO / "paper2skills-research" / "data" / "gap-ledger.json"
MAPFILE = REPO / "paper2skills-research" / "data" / "backlog-l3-map.json"
F3_SCRIPT = REPO / "paper2skills-research" / "scripts" / "build_gap_ledger.py"
OUT_JSON = REPO / "paper2skills-research" / "data" / "backlog-routing.json"
OUT_MD = REPO / "paper2skills-research" / "reports" / "PHASE6-S3-backlog接管表.md"

EXIT_PASS, EXIT_RED, EXIT_INPUT, EXIT_INTERNAL = 0, 1, 2, 3

#: Q4 弱门槛的放行集合（与 `check_card_l3.py` 同口径；C 类不放行）
PASS_SERVICEABILITY = ("A", "B")

#: 权重键（与 F3 的 `DEFAULT_WEIGHTS` 同名同义；默认值**取自 gap-ledger.json 的 weights**，
#: 以免两份默认值各自漂移 —— 若两边不等，J4 会当场报出来）
WEIGHT_KEYS = ("zero_gap", "flow_breadth", "first_slice",
               "boundary_penalty", "structural_penalty")

#: F3 的五因子公式（**逐因子 4 位四舍五入后求和，再 4 位四舍五入** —— 与 F3 逐字同构；
#: 本函数只在 T1 上与 F3 的产物逐条断言相等，故不是「第二套口径」，是它的可复算副本）
T1_NOTE = "靶区一 = A ∩ 无精选卡"


class InputMissing(Exception):
    """输入没拿到 —— 退出码 2，**不是通过**，也不是判红。"""


def _rel(p: Path) -> str:
    """仓库内 → 相对路径；仓库外（selftest 的 /tmp 副本）→ 原样返回。

    门禁缺陷 #13 的同族：`relative_to()` 对仓库外路径抛 ValueError，
    于是「判红的原因」被 traceback 吞掉，只剩一个退出码 —— 仪器故障与判红必须可分。
    """
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 装载（全部只读）
# ---------------------------------------------------------------------------
def load_json(path: Path, what: str) -> dict:
    if not Path(path).is_file():
        raise InputMissing(f"{what} 不存在：{_rel(Path(path))}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputMissing(f"{what} 不是合法 JSON：{_rel(Path(path))}（{exc}）") from exc


def inspect_registry(reg: dict) -> dict:
    for k in ("records",):
        if k not in reg:
            raise InputMissing(f"registry 缺字段 {k} —— 版本太旧")
    recs = reg["records"]
    for r in recs:
        for k in ("paper_id", "decision", "status"):
            if k not in r:
                raise InputMissing(f"registry 记录缺字段 {k}：{r.get('paper_id')}")
    return {"n_records": len(recs), "revision": reg.get("schema_version")}


def load_graph(path: Path) -> dict:
    g = load_json(path, "能力图谱")
    for k in ("l3", "roles", "cards"):
        if k not in g:
            raise InputMissing(f"图谱缺字段 {k} —— 先跑 build_capability_graph.py")
    return g


def derive_backlog(registry: dict, repo: Path = REPO) -> dict:
    """按规则从 registry **现算** backlog（不抄清单）。

    规则：`decision == "extract"` ∧ **没有已交付产物**
    （`outputs.skill_card` 指向的文件不存在 ∧ `outputs.enhanced_cards` 为空）。

    ⚠️ **N4 双层计数**（PHASE5 实测登记）：只数 `outputs.skill_card` 会得 **15** 篇，
    真值 **12** —— 差额正是 `0023`/`0026`/`0030` 三篇**已作为 3E 增强交付**的论文。
    故本函数同时给出 naive 与真值，并把被扣掉的三篇**具名**报出（不是静默减 3）。
    """
    recs = registry["records"]
    naive, enhancement, backlog, integrity = [], [], [], []
    for r in recs:
        if r.get("decision") != "extract":
            continue
        pid = r["paper_id"]
        outs = r.get("outputs") or {}
        sc = outs.get("skill_card")
        card_exists = bool(sc) and (Path(repo) / sc).is_file()
        enh = list(outs.get("enhanced_cards") or [])


        # --- J7：status 标签 vs 实物（两处必须一致，任一边撒谎都判红） ---
        want = ("extracted" if card_exists
                else "extracted-as-enhancement" if enh else "shortlisted")
        if r.get("status") != want:
            integrity.append(
                f"{pid}：status 声明 `{r.get('status')}`，实物推得 `{want}`"
                f"（卡文件存在={card_exists}、enhanced_cards={len(enh)}）")

        # ⚠️ naive 的定义就是「只数 outputs.skill_card」——故**增强交付的 3 篇也要进来**，
        #    否则 naive 会等于真值，N4 那条差额（15 vs 12）在账上就消失了。
        if card_exists:
            pass                                     # 已出卡 → 不在任何欠账里
        elif enh:
            naive.append(pid)
            enhancement.append({"paper_id": pid, "enhanced_cards": enh,
                                "title": r.get("title")})
        else:
            naive.append(pid)
            backlog.append(r)
    return {"backlog": backlog, "naive": naive, "enhancement_delivered": enhancement,
            "integrity_errors": integrity}


# ---------------------------------------------------------------------------
# 缺口位次（F3 公式的可复算副本）
# ---------------------------------------------------------------------------
def score_row(row: dict, weights: dict) -> dict:
    """F3 的五因子分解，**逐字同构**（含每个因子先四舍五入到 4 位）。

    `structural_blank` 是 F3 只对零供给行标注的字段；非零供给行一律 False —— 照抄，
    不在这里「补全」它，否则会造出与 F3 不同的第二套分数。
    """
    parts = {
        "zero_gap": round(weights["zero_gap"] * (1.0 / (1 + row["supply"]["n_legacy"])), 4),
        "flow_breadth": round(weights["flow_breadth"] * (row["n_flows"] / 8), 4),
        "first_slice": round(weights["first_slice"] * (1 if row["in_first_slice"] else 0), 4),
        "boundary_penalty": round(-weights["boundary_penalty"] * (1 if row["boundary_a"] else 0), 4),
        "structural_penalty": round(
            -weights["structural_penalty"] * (1 if row.get("structural_blank") else 0), 4),
    }
    return {"parts": parts, "score": round(sum(parts.values()), 4)}


def load_f3_module(path: Path = F3_SCRIPT):
    """把 F3 的生成器当 **oracle** 加载（只读；不改它的任何一行）。

    J4 用它做「判据只有一处实现」的现场核对：同样的权重下，它排出的 31 条位次
    必须与本脚本现算的逐条相等。拿不到就**退 2**，不是静默跳过这道判据。
    """
    if not Path(path).is_file():
        raise InputMissing(f"F3 生成器不在：{_rel(Path(path))}（一致性判据 J4 无从核对）")
    spec = importlib.util.spec_from_file_location("_f3_gap_ledger", str(path))
    if spec is None or spec.loader is None:
        raise InputMissing(f"F3 生成器无法作为模块加载：{_rel(Path(path))}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f3_run(weights: dict, graph_path: Path = GRAPH) -> dict:
    """现场跑 F3 的 `build_ledger`，**只在内存里**拿一份缺口账（**绝不落盘** ——
    落盘会覆盖别的任务正在依赖的 `gap-ledger.json`）。J4 与 J9 共用它。"""
    mod = load_f3_module()
    g = mod.load_graph(Path(graph_path))
    legacy, info = mod.load_legacy()
    if not info.get("available"):
        raise InputMissing("产品侧 classification.json 不在 —— F3 的供给列会整体失真，"
                           "一致性核对无从进行（跨仓库依赖）")
    return mod.build_ledger(g, legacy, dict(weights))


def f3_positions(weights: dict, graph_path: Path = GRAPH) -> list[dict]:
    """F3 自己的 31 条位次（= `f3_run` 的 worklist 投影）。"""
    led = f3_run(weights, graph_path)
    return [{"rank": w["rank"], "l3": w["l3"], "score": w["score"],
             "role_id": w["role_id"]} for w in led["worklist"]]


def check_input_freshness(graph_path: Path, ledger: dict, f3_ledger: dict) -> tuple[list[str], dict]:
    """J9 输入新鲜度（风险 N6：**基线数字可能是过期照片**）。

    `gap-ledger.json` 是 F3 的产物，它把生成时的 `capability-graph.json` sha256 钉在 `_meta` 里。
    图谱若在那之后被别的任务重生成，这份缺口账就是**过期照片** —— 而它是本脚本排序的全部依据。

    处置分两态（**不是一句「过期了」**）：
    · 用 F3 **现场现算**（只读、不落盘）得到的 151 行 / 31 条工单与已入库产物**实质相等**
      ⇒ 登记为新鲜（或「过期但无实质影响」），作为一等输出；
    · 实质有差异 ⇒ **判红**（此刻的位次是拿旧底本算的，不许出表）。

    ⚠️ **2026-09-13 修掉一个「判据永远不可能失败」的结构洞**（本函数第一版的形态）：

        第一版把「逐行内容核对」写在 `if info["stale"]:` 分支**里面** ——
        于是 `声明 sha == 当前图谱 sha` 时**提前 `return fresh`，一行内容都不看**。

    这错在**把自述当成了核验**：`_meta.graph_sha256` 是**生成者自己写下的**「我读的是哪张图谱」，
    它**不是缺口账内容的指纹**。一份内容被改过、而 `_meta` 里仍写着当前 sha 的缺口账，
    在第一版下**畅通无阻**；反倒是「老实声明自己读的是旧图谱」的那种才会被逐行核对。
    ⇒ **越是看起来新鲜的，越没人查。**

    实测证据（不是推理）：`git log` 逐提交复算 `graph sha` 与 `declared` ——
    `4e1b498`（F3 自身）· `59429b9`（F4/F5）· `bb06b74`（S5）三处 `stale=False`
    ⇒ **内容核对是死代码**；只有 `dfe9eff`（S1）一处 `stale=True` 时它才真跑过。
    连带后果：`--selftest` 里那条「篡改：过期底本 + 排序依据被改 ⇒ 判红」**从 `dfe9eff` 起就在失败**
    （41 条里 1 条败），而没有人看见 —— 因为本脚本**当时还没被接进验收面**。
    ⇒ 两条纪律的交叉：**「判据的适用范围被默认成了全体」×「交付了但没人看得见」**。

    修法：**内容核对与 sha 状态无关，每次都必须跑**。sha 只决定**归因**（新鲜 / 过期但无实质影响），
    不决定**判不判**。
    """
    cur = _sha(graph_path)
    declared = (ledger.get("_meta") or {}).get("graph_sha256")
    info = {"graph_sha256_now": cur, "ledger_declared_graph_sha256": declared,
            "stale": bool(declared) and declared != cur}
    if not declared:
        info["verdict"] = "unknown"
        return ["J9 gap-ledger.json 没声明生成时的图谱 sha256 —— 过期与否与内容来源**都无从判断**"], info

    # —— 以下是**内容核对**：无条件执行（第一版把它关在 stale 分支里，见 docstring）——
    mine_rows = {r["l3"]: (r["serviceability"], r["supply"]["status"],
                           r["supply"]["n_legacy"], r["n_flows"], r["in_first_slice"],
                           bool(r["boundary_a"]), bool(r.get("structural_blank")))
                 for r in ledger["rows"]}
    fresh_rows = {r["l3"]: (r["serviceability"], r["supply"]["status"],
                            r["supply"]["n_legacy"], r["n_flows"], r["in_first_slice"],
                            bool(r["boundary_a"]), bool(r.get("structural_blank")))
                  for r in f3_ledger["rows"]}
    diff_rows = sorted(k for k in set(mine_rows) | set(fresh_rows)
                       if mine_rows.get(k) != fresh_rows.get(k))
    mine_wl = [(w["l3"], w["rank"], w["score"]) for w in ledger["worklist"]]
    fresh_wl = [(w["l3"], w["rank"], w["score"]) for w in f3_ledger["worklist"]]
    info.update({"n_row_diff": len(diff_rows), "row_diff_sample": diff_rows[:6],
                 "worklist_identical": mine_wl == fresh_wl})
    if diff_rows or mine_wl != fresh_wl:
        info["verdict"] = "material_mismatch"
        return [f"J9 缺口账与本脚本现场现算的 F3 结果**实质不符**：151 行里 {len(diff_rows)} 行不同"
                f"（样本 {diff_rows[:4]}）、工单是否逐条相同={mine_wl == fresh_wl}；"
                f"声明 sha {declared} vs 现状 {cur}"
                f"（{'**一致**' if not info['stale'] else '不一致'}）。"
                f"⚠️ 声明 sha 与现状一致**不能**证明内容没被改过 —— 它是生成者的自述，"
                f"不是内容的指纹 ⇒ 先重跑 build_gap_ledger.py，再判差异是真变化还是被改过"], info
    if info["stale"]:
        info["verdict"] = "stale_immaterial"
        info["why_immaterial"] = ("图谱换了底本（多为 cells[].solution_refs / solutions 层），"
                                  "但本脚本排序所依赖的 151 行（L3/服务性/供给/FLOW/边界）与 31 条工单"
                                  "**现算逐条相同** ⇒ 不影响位次")
        return [], info
    info["verdict"] = "fresh"
    return [], info


def build_positions(ledger: dict, weights: dict) -> dict:
    """把 151 条 L3 分成两个**互不混用的位置空间**：

    - `t1`：靶区一内部位次（= F3 的 `rank`，用 J4 逐条核对）
    - `ext`：靶区外（已有精选供给 / 非 A 类）内部位次 —— **本脚本的扩展口径**，明确标注，
      绝不与 `t1` 混排（混排会让「与 F3 逐条一致」这句话变成假的）
    """
    rows = ledger.get("rows") or []
    if not rows:
        raise InputMissing("gap-ledger.json 没有 rows —— 先跑 build_gap_ledger.py")
    worklist = ledger.get("worklist") or []
    t1_names = {w["l3"] for w in worklist}
    if len(t1_names) != len(worklist):
        raise InputMissing("gap-ledger.json 的 worklist 里有重复 L3 名 —— 位次不可复算")

    scored = []
    for r in rows:
        s = score_row(r, weights)
        scored.append({"l3": r["l3"], "role_id": r["role_id"], "role_title": r["role_title"],
                       "domain": r["domain"], "serviceability": r["serviceability"],
                       "supply": r["supply"], "n_flows": r["n_flows"],
                       "in_first_slice": r["in_first_slice"], "boundary_a": r["boundary_a"],
                       "structural_blank": bool(r.get("structural_blank")),
                       "score": s["score"], "score_parts": s["parts"],
                       "in_t1": r["l3"] in t1_names})

    t1 = sorted([x for x in scored if x["in_t1"]], key=lambda x: (-x["score"], x["l3"]))
    ext = sorted([x for x in scored if not x["in_t1"]], key=lambda x: (-x["score"], x["l3"]))
    for i, x in enumerate(t1, 1):
        x["t1_rank"] = i
        x["ext_rank"] = None          # 两个位置空间互不混用：靶区内的一条不也有扩展位
    for i, x in enumerate(ext, 1):
        x["ext_rank"] = i
        x["t1_rank"] = None
    return {"by_l3": {x["l3"]: x for x in scored}, "t1": t1, "ext": ext,
            "n_rows": len(scored)}


# ---------------------------------------------------------------------------
# 映射装载与校验（J1）
# ---------------------------------------------------------------------------
def index_map(mapobj: dict) -> dict:
    entries = mapobj.get("entries")
    if entries is None:
        raise InputMissing("backlog-l3-map.json 缺 entries")
    out = {}
    for e in entries:
        pid = e.get("paper_id")
        if not pid:
            raise InputMissing(f"映射条目缺 paper_id：{e}")
        if pid in out:
            raise InputMissing(f"映射里 paper_id 重复：{pid}（重复键会被后者静默覆盖）")
        out[pid] = e
    return out


def validate_mapping(backlog: list[dict], mapping: dict, graph: dict,
                     registry_by_id: dict) -> list[str]:
    """J1：映射完整性。返回错误列表（空 = 过）。"""
    errs = []
    l3_index = {x["name"]: x for x in graph["l3"]}
    if len(l3_index) != len(graph["l3"]):
        errs.append("J1 图谱 151 名单里有重名 L3 —— 白名单不可用")
    backlog_ids = [r["paper_id"] for r in backlog]

    missing = [p for p in backlog_ids if p not in mapping]
    if missing:
        errs.append(f"J1 这 {len(missing)} 篇 backlog 没有任何映射（会被静默丢弃）：{missing}")
    orphan = sorted(set(mapping) - set(backlog_ids))
    if orphan:
        errs.append(f"J1 映射里有 {len(orphan)} 条**不是 backlog** 的条目（死重，且说明映射是抄的不是现算的）：{orphan}")

    for pid, e in mapping.items():
        l3s = e.get("l3") or []
        if not l3s:
            errs.append(f"J1 {pid} 的 l3 为空 —— 「落不到 L3」必须在 J2 的未落名单里出现，不能靠空列表表示")
            continue
        rec = registry_by_id.get(pid) or {}
        for item in l3s:
            name = item.get("name")
            x = l3_index.get(name)
            if x is None:
                errs.append(f"J1 {pid} 的 L3 `{name}` 不在 151 名单内（改名必须两边同时改）")
                continue
            if x["serviceability"] not in PASS_SERVICEABILITY:
                errs.append(f"J1 {pid} 的 L3 `{name}` 是 {x['serviceability']} 类 —— "
                            f"Q4 弱门槛不放行 C 类（放行集合 {PASS_SERVICEABILITY}）")
            basis = item.get("basis")
            if basis not in ("business_action", "capability_inference"):
                errs.append(f"J1 {pid}／{name} 的 basis=`{basis}` 不在枚举内")
                continue
            conf = item.get("confidence")
            if basis == "capability_inference" and conf != "low":
                errs.append(f"J1 {pid}／{name} 声明为推得（capability_inference）却标 confidence="
                            f"`{conf}` —— 推得的映射必须自认为 low")
            ev = item.get("evidence") or ""
            field = item.get("evidence_field")
            found = [k for k in ("title", "decision_reason", "note")
                     if ev and ev in str(rec.get(k) or "")]
            if not found:
                errs.append(f"J1 {pid}／{name} 的证据 `{ev}` 在 registry 的 title/decision_reason/note "
                            f"里**逐字找不到**（G2b 口径：编出来的证据不算证据）")
            elif field not in found:
                errs.append(f"J1 {pid}／{name} 声明证据在 `{field}`，实测只在 {found} —— "
                            f"声明与实物矛盾")
        for alt in (e.get("alternatives") or []):
            if alt.get("name") not in l3_index:
                errs.append(f"J1 {pid} 的「已否决近邻」`{alt.get('name')}` 不是真实 L3 名"
                            f"（近邻表也会腐烂）")
            if not (alt.get("why_not") or "").strip():
                errs.append(f"J1 {pid} 的「已否决近邻」`{alt.get('name')}` 没有否决理由")
    return errs


# ---------------------------------------------------------------------------
# 路由（排序）
# ---------------------------------------------------------------------------
#: 路由排序键允许出现的字段（**必须是缺口侧**）。J5 的 AST 扫描按这张表判。
ROUTE_KEY_FUNC = "route_sort_key"
KEY_ALLOW_PRIMARY = ("tier_order", "gap_pos")
KEY_ALLOW_ANY = ("tier_order", "gap_pos", "old_rank")
#: 任何排序键里都不许出现的字段：`venue_tier` 是论文质量信号，连 tiebreak 都不该进；
#: `score` 只允许以**预先算好的整数 `old_rank`** 的形式间接进入末位 tiebreak。
KEY_DENY = ("score", "paper_score", "venue_tier")
LEGACY_KEY_FUNC = "legacy_order_key"


def legacy_order_key(rec: dict) -> tuple:
    """**旧顺序**（本卡要接管掉的那个顺序）：人工挑选的 priority → 论文自己的分数。

    它只用于**末位 tiebreak**：两篇论文服务的靶区位次相同时，才用旧顺序决定谁先。
    这不是「分数重获主排序权」——`route_sort_key` 的第一、二位是缺口侧字段，
    且 J5 会现场证明「按分数排」与「按缺口排」**不是同一个顺序**。
    """
    prio = str(rec.get("priority") or "P9")
    return (prio, -float(rec.get("score") or 0.0), rec.get("paper_id") or "")


def route_sort_key(rec: dict) -> tuple:
    """靶区层级 → 靶区位次 → 旧位次（仅末位 tiebreak）。"""
    return (rec["tier_order"], rec["gap_pos"], rec["old_rank"])


def route(backlog: list[dict], mapping: dict, pos: dict) -> dict:
    """把每篇 backlog 论文映射到 L3 责任，并产出**由缺口决定**的新位次。"""
    by_l3 = pos["by_l3"]
    legacy = sorted(backlog, key=legacy_order_key)
    old_rank = {r["paper_id"]: i for i, r in enumerate(legacy, 1)}

    routed, unassigned, needs_review = [], [], []
    for rec in backlog:
        pid = rec["paper_id"]
        ent = mapping.get(pid) or {}
        served = []
        for item in (ent.get("l3") or []):
            x = by_l3.get(item.get("name"))
            if x is None:
                continue                      # J1 已经报红；这里不静默补位
            served.append({
                "name": x["l3"], "role_id": x["role_id"], "role_title": x["role_title"],
                "domain": x["domain"], "serviceability": x["serviceability"],
                "supply_status": x["supply"]["status"], "n_legacy": x["supply"]["n_legacy"],
                "in_t1": x["in_t1"], "t1_rank": x["t1_rank"], "ext_rank": x["ext_rank"],
                "score": x["score"], "basis": item.get("basis"),
                "confidence": item.get("confidence"), "evidence": item.get("evidence"),
                "evidence_field": item.get("evidence_field"), "note": item.get("note"),
            })
        if not served:
            unassigned.append({"paper_id": pid, "title": rec.get("title"),
                               "tech_domain": rec.get("domain"),
                               "why": "映射缺失（J1 已报红）" if pid not in mapping
                                      else "映射里的 L3 全部不在 151 名单内（J1 已报红）"})
            continue

        t1_hits = [s for s in served if s["in_t1"]]
        if t1_hits:
            tier, tier_order = "A", 0
            primary = min(t1_hits, key=lambda s: (s["t1_rank"], s["name"]))
            gap_pos = primary["t1_rank"]
        else:
            tier, tier_order = "B", 1
            primary = min(served, key=lambda s: (s["ext_rank"], s["name"]))
            gap_pos = primary["ext_rank"]

        # 推得的映射不得**独自**把论文送进靶区（capability_inference 的边界）
        if tier == "A" and primary["basis"] == "capability_inference":
            needs_review.append({"paper_id": pid, "l3": primary["name"],
                                 "why": "靶区落点全靠推得的映射（capability_inference）支撑"})

        routed.append({
            "paper_id": pid, "title": rec.get("title"), "tech_domain": rec.get("domain"),
            "priority": rec.get("priority"), "paper_score": rec.get("score"),
            "venue": rec.get("venue"), "venue_tier": rec.get("venue_tier"),
            "arxiv": (rec.get("identifiers") or {}).get("arxiv"),
            "decision_reason": rec.get("decision_reason"),
            "tier": tier, "tier_order": tier_order, "gap_pos": gap_pos,
            "primary_l3": {k: primary[k] for k in
                           ("name", "role_id", "role_title", "domain", "in_t1",
                            "t1_rank", "ext_rank", "basis", "confidence", "evidence")},
            "served_l3": served,
            "n_target_l3": len(t1_hits),
            "old_rank": old_rank[pid],
        })

    routed.sort(key=route_sort_key)
    for i, r in enumerate(routed, 1):
        r["new_rank"] = i
        r["moved"] = r["old_rank"] - r["new_rank"]

    # 覆盖率：一等输出（分母是实测 backlog 篇数，不是文档里的 12）
    n = len(backlog)
    cov = {
        "denominator_note": f"分母 = registry 现算的 extract backlog 篇数（{n}），不是文档里的数字",
        "n_backlog": n, "n_mapped": len(routed), "n_unassigned": len(unassigned),
        "coverage_ratio": (len(routed) / n) if n else None,
        "unassigned": unassigned,
        "n_target_zone": len([r for r in routed if r["tier"] == "A"]),
        "target_zone": [{"paper_id": r["paper_id"], "gap_pos": r["gap_pos"],
                         "l3": r["primary_l3"]["name"], "role_id": r["primary_l3"]["role_id"]}
                        for r in routed if r["tier"] == "A"],
        "out_of_target_zone": [{"paper_id": r["paper_id"], "best_l3": r["primary_l3"]["name"],
                                "supply_status": r["primary_l3"]["ext_rank"] and
                                next(s["supply_status"] for s in r["served_l3"]
                                     if s["name"] == r["primary_l3"]["name"])}
                               for r in routed if r["tier"] == "B"],
        "needs_review": needs_review,
        "n_decided_by_legacy_tiebreak": _n_tiebreak_decided(routed),
        # 新旧位次到底动了多少：S3 的产出就是「动了」，所以要有一等数字
        "n_moved": len([r for r in routed if r["moved"] != 0]),
        "max_move": max([abs(r["moved"]) for r in routed], default=0),
        # 低置信映射单独报出：这是「这一层的判断有多硬」的可见账，不能只写在注里
        "n_low_confidence": len([r for r in routed
                                 if any(s["confidence"] == "low" for s in r["served_l3"])]),
        "low_confidence": [{"paper_id": r["paper_id"],
                            "l3": [s["name"] for s in r["served_l3"] if s["confidence"] == "low"],
                            "basis": "capability_inference"}
                           for r in routed
                           if any(s["confidence"] == "low" for s in r["served_l3"])],
    }
    return {"routed": routed, "unassigned": unassigned, "coverage": cov,
            "legacy_order": [r["paper_id"] for r in legacy]}


def _n_tiebreak_decided(routed: list[dict]) -> int:
    """报出「位次是靠旧顺序（分数）决定的」篇数 —— 让 tiebreak 的体重可见，而不是藏起来。

    计入规则：与**不同** gap_pos 的论文相比不算；只有同一 `(tier_order, gap_pos)` 组内
    除第一篇之外的每一篇，才是「靠 tiebreak 定的位」。
    """
    groups: dict[tuple, list] = {}
    for r in routed:
        groups.setdefault((r["tier_order"], r["gap_pos"]), []).append(r)
    return sum(max(0, len(v) - 1) for v in groups.values())


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------
def check_against_f3(ledger: dict, pos: dict, f3_rows: list[dict]) -> list[str]:
    """J4：与 F3 逐条一致。两处核对 —— 已入库产物 + 现场跑生成器。"""
    errs = []
    shipped = ledger.get("worklist") or []
    if not shipped:
        errs.append("J4 gap-ledger.json 没有 worklist —— 靶区位次无从核对")
        return errs
    if len(shipped) != len(pos["t1"]):
        errs.append(f"J4 靶区条数：已入库 {len(shipped)} vs 现算 {len(pos['t1'])}")
    shipped_by_l3 = {w["l3"]: w for w in shipped}
    mine_by_l3 = {x["l3"]: x for x in pos["t1"]}
    if set(shipped_by_l3) != set(mine_by_l3):
        errs.append(f"J4 靶区名单不一致：只在已入库 {sorted(set(shipped_by_l3) - set(mine_by_l3))[:5]}"
                    f" / 只在现算 {sorted(set(mine_by_l3) - set(shipped_by_l3))[:5]}")
    for name in sorted(set(shipped_by_l3) & set(mine_by_l3)):
        a, b = shipped_by_l3[name], mine_by_l3[name]
        if a["rank"] != b["t1_rank"]:
            errs.append(f"J4 `{name}` 位次：已入库 {a['rank']} vs 现算 {b['t1_rank']}")
        if abs(float(a["score"]) - float(b["score"])) > 1e-9:
            errs.append(f"J4 `{name}` 分数：已入库 {a['score']} vs 现算 {b['score']}"
                        f"（公式副本已经漂移）")

    # 现场跑 F3 生成器（判据只有一处实现的现场证据）
    if len(f3_rows) != len(pos["t1"]):
        errs.append(f"J4 现场 F3 靶区条数 {len(f3_rows)} ≠ 本脚本 {len(pos['t1'])}")
    for i, (a, b) in enumerate(zip(f3_rows, pos["t1"]), 1):
        if a["l3"] != b["l3"]:
            errs.append(f"J4 现场 F3 第 {i} 位是 `{a['l3']}`，本脚本是 `{b['l3']}`")
        elif abs(float(a["score"]) - float(b["score"])) > 1e-9:
            errs.append(f"J4 现场 F3 `{a['l3']}` 分数 {a['score']} ≠ 本脚本 {b['score']}")
    return errs


def assert_weight_sensitive(orders: dict[str, list[str]], label: str = "J3") -> list[str]:
    """J3 的核心断言（**做成函数才能被篡改样本直接喂**）：
    给了 ≥2 组权重，产出的论文位次序列**不得全部相同** —— 全同即「缺口没进排序」。
    """
    if len(orders) < 2:
        return [f"{label} 只喂了 {len(orders)} 组权重 —— 单边：没有第二组就测不出敏感性"]
    uniq = {tuple(v) for v in orders.values()}
    if len(uniq) == 1:
        return [f"{label} 改了靶区权重，论文位次**一点没变**（{len(orders)} 组权重产出同一顺序）"
                f" —— 排序不是缺口驱动的"]
    return []


def check_consistency(routed: list[dict], pos: dict, ledger: dict) -> list[str]:
    """J4 的第二半：论文用到的位次必须等于 F3 给该 L3 的位次；层级不得倒挂。"""
    errs = []
    shipped = {w["l3"]: w["rank"] for w in (ledger.get("worklist") or [])}
    t1_names = set(shipped)
    for r in routed:
        p = r["primary_l3"]
        if r["tier"] == "A":
            want = shipped.get(p["name"])
            if want is None:
                errs.append(f"J4 {r['paper_id']} 被排在靶区，用的 L3 `{p['name']}` 却不在 F3 靶区名单")
            elif want != r["gap_pos"]:
                errs.append(f"J4 {r['paper_id']} 用位次 {r['gap_pos']}，F3 给 `{p['name']}` 的是 {want}")
        else:
            if p["name"] in t1_names:
                errs.append(f"J4 {r['paper_id']} 判为靶区外，主 L3 `{p['name']}` 却在 F3 靶区里")
            if p["ext_rank"] is None:
                errs.append(f"J4 {r['paper_id']} 靶区外却没有扩展位次")
    tiers = [r["tier_order"] for r in routed]
    if tiers != sorted(tiers):
        errs.append("J4 靶区外论文排在了靶区内论文之前（层级倒挂）")
    # ⚠️ 只查「tier_order 单调」不够 —— 篡改样本第一版把一张靶区内论文的 tier_order 改成 1 之后
    #    **顺手重排了列表**，等于把倒挂自己修好了，于是判据看起来「没抓住」（样本把证据擦掉了）。
    #    故再加一条：`tier` 标签与 `tier_order` 必须互相自洽（同一事实的两种写法，任一边撒谎即红）。
    label_of = {"A": 0, "B": 1}
    for r in routed:
        if label_of.get(r["tier"]) != r["tier_order"]:
            errs.append(f"J4 {r['paper_id']} 的 tier=`{r['tier']}` 与 tier_order="
                        f"{r['tier_order']} 不自洽")
    return errs


def scan_route_key(source_text: str) -> list[str]:
    """J5 静态半：AST 扫**本脚本自己的**排序键，禁止论文侧分数/venue 出现在主键上。

    为什么用 AST 而不是字符串 needle：本文件的文档字符串里到处是 `score`，
    字符串扫描只会得到假阳性，然后被判据「放宽」掉 —— 那正是本仓库记过的失败模式。
    """
    errs = []
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return [f"J5 自扫失败：本脚本语法错误（{exc}）—— 扫描器没在测东西"]
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == ROUTE_KEY_FUNC), None)
    if fn is None:
        return [f"J5 扫描目标 `{ROUTE_KEY_FUNC}` 不存在 —— 扫描器在看别处（覆盖率静默变小）"]

    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return [f"J5 `{ROUTE_KEY_FUNC}` 没有 return —— 无从判断主键"]
    elts = None
    for r in returns:
        if isinstance(r.value, ast.Tuple) and len(r.value.elts) >= 3:
            elts = r.value.elts
    if elts is None:
        return [f"J5 `{ROUTE_KEY_FUNC}` 没有返回 ≥3 元组 —— 排序键的形状变了，判据失效"]

    def keys_of(node) -> list[str]:
        out = []
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            out.append(node.slice.value)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for e in node.elts:
                out.extend(keys_of(e))
        return out

    primary = keys_of(elts[0])
    if not primary:
        errs.append(f"J5 `{ROUTE_KEY_FUNC}` 的主键不是一个字段引用 —— 判据读不懂它")
    for k in primary:
        if k not in KEY_ALLOW_PRIMARY:
            errs.append(f"J5 主排序键里出现 `{k}` —— 主键必须是缺口侧字段 {KEY_ALLOW_PRIMARY}")
    if 1 not in [i for i, e in enumerate(elts) if keys_of(e)]:
        errs.append(f"J5 `{ROUTE_KEY_FUNC}` 的第二位不是字段引用（缺口位次没进键）")
    elif not set(keys_of(elts[1])) <= set(KEY_ALLOW_PRIMARY):
        errs.append(f"J5 第二排序键里出现 `{keys_of(elts[1])}` —— 第二位必须是靶区位次")
    for e in elts[2:]:
        for k in keys_of(e):
            if k not in KEY_ALLOW_ANY:
                errs.append(f"J5 末位 tiebreak 里出现 `{k}` —— 只允许 {KEY_ALLOW_ANY}")
    for e in elts:
        for k in keys_of(e):
            if k in KEY_DENY:
                errs.append(f"J5 排序键里出现禁止字段 `{k}`（论文自身的分数/venue 不得成为排序键）")
    return errs


def check_no_venue_in_legacy_key(source_text: str) -> list[str]:
    """J5 反向半：`venue_tier` 连 tiebreak 都不许进 —— 旧顺序用 priority+score，用它即红。"""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == LEGACY_KEY_FUNC), None)
    if fn is None:
        return [f"J5 `{LEGACY_KEY_FUNC}` 不存在 —— 旧顺序的定义不可核对"]
    errs = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and n.slice.value == "venue_tier":
            errs.append("J5 旧顺序键里引用了 `venue_tier` —— venue 是论文质量信号，不进排序")
    return errs


def check_score_override_injection(backlog, mapping, ledger, graph) -> tuple[list[str], int]:
    """J5 运行半：喂一份「按论文分数排序」的注入样本，自证两者顺序**必须不同**。

    样本构造：3 篇论文，服务的靶区位次分别是 1 / 15 / 31，
    而论文自己的分数被摆成**反序**（服务最差缺口的分数最高）。
    若产出顺序 == 按分数排的顺序 ⇒ 论文分数事实上成了主排序键 ⇒ 判红。
    """
    errs = []
    scenario = [
        ("P-LOW", 90.0, "线索评估"),        # 分最高，服务靶区第 1 位
        ("P-MID", 50.0, "研发计划"),        # 中，服务第 15 位
        ("P-HIGH", 10.0, "履约跟踪"),       # 分最低，服务第 31 位
    ]
    t1 = {w["l3"]: w["rank"] for w in ledger["worklist"]}
    if not all(name in t1 for _, _, name in scenario):
        return ["J5 注入样本用的 L3 不在靶区名单里 —— 样本自身失效，判据等于没测"], 0
    # 分反序：让「按分数排」的顺序恰好是 31 → 15 → 1，与缺口顺序完全相反
    scenario = [("P-HIGH", 90.0, "履约跟踪"), ("P-MID", 50.0, "研发计划"),
                ("P-LOW", 10.0, "线索评估")]
    recs, mp = [], {}
    for pid, sc, l3 in scenario:
        recs.append({"paper_id": pid, "title": f"inj-{pid}", "decision": "extract",
                     "status": "shortlisted", "domain": "05-推荐系统", "priority": "P1",
                     "score": sc, "identifiers": {"arxiv": "0"},
                     "outputs": {"skill_card": f"__inj__/{pid}.md"}})
        mp[pid] = {"paper_id": pid, "l3": [{"name": l3, "evidence": "x",
                                            "evidence_field": "title",
                                            "basis": "business_action", "confidence": "high"}]}
    pos = build_positions(ledger, ledger["weights"])
    out = route(recs, mp, pos)
    got = [r["paper_id"] for r in out["routed"]]
    by_score = [p for p, _, _ in sorted(scenario, key=lambda t: -t[1])]
    if got == by_score:
        errs.append(f"J5 注入样本下产出顺序 {got} 与「按论文分数排」{by_score} **完全相同** —— "
                    f"论文分数事实上成了主排序键（反后门失效）")
    return errs, len(scenario)


def assert_takeover_changed(new_order: list[str], old_order: list[str]) -> list[str]:
    """J8：接管必须**真的改变**了顺序（双边判据 —— 「没变」也是要报出来的结论）。

    「新序 == 旧序」有两种可能，且都必须报红而不是当成完成：① 缺口排序没生效；
    ② 缺口恰好与旧顺序完全重合（那结论是「旧清单本来就对」，要人裁决，不能静默通过）。
    """
    if not new_order or not old_order:
        return ["J8 空顺序 —— 没有可比的两个序列（单边：等于没测）"]
    if set(new_order) != set(old_order):
        return ["J8 新旧顺序的论文集合不同 —— 不是同一批 backlog，比较无意义"]
    if new_order == old_order:
        return ["J8 接管后的位次与旧顺序**逐位相同** —— 排序接管没有产生任何变化"
                "（要么缺口没进排序，要么旧清单本来就与缺口一致，两种都要人裁决）"]
    return []


def check_order_vs_score(routed: list[dict]) -> tuple[list[str], dict]:
    """J5 运行半（真实数据）：接管后的顺序必须与「按论文分数排」不同，否则接管是空转。"""
    got = [r["paper_id"] for r in routed]
    by_score = [r["paper_id"] for r in sorted(routed, key=lambda r: (-float(r["paper_score"] or 0),
                                                                    r["paper_id"]))]
    by_venue = [r["paper_id"] for r in sorted(routed, key=lambda r: (str(r["venue_tier"] or ""),
                                                                    r["paper_id"]))]
    errs = []
    if got == by_score:
        errs.append("J5 接管后的位次与「按论文 score 排」逐位相同 —— 缺口没起作用（S3 空转）")
    if got == by_venue:
        errs.append("J5 接管后的位次与「按 venue_tier 排」逐位相同 —— venue 事实上成了主排序键")
    return errs, {"order": got, "by_score": by_score, "by_venue": by_venue}


def check_n4_counts(reg: dict) -> list[str]:
    """J7：N4 双层计数 + status 标签与实物一致。"""
    errs = []
    naive, enh, backlog = reg["naive"], reg["enhancement_delivered"], reg["backlog"]
    if len(naive) != len(backlog) + len(enh):
        errs.append(f"J7 恒等式不成立：naive {len(naive)} ≠ backlog {len(backlog)} + 增强交付 {len(enh)}")
    for e in enh:
        if not e["enhanced_cards"]:
            errs.append(f"J7 {e['paper_id']} 被算成增强交付却没有任何 enhanced_cards")
    errs.extend(f"J7 registry 自洽：{m}" for m in reg["integrity_errors"])
    return errs


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render_md(product: dict) -> str:
    led, cov, meta = product, product["coverage"], product["_meta"]
    L, A = [], None
    A = L.append
    A("# PHASE6 S3 · extract backlog 靶区接管表（**由脚本生成，勿手改**）")
    A("")
    A(f"> 生成时间：{meta['generated']}　|　生成器：`{meta['generator']}`")
    A("> 复算：`python3 paper2skills-research/scripts/route_backlog.py --check`")
    A(f"> 输入：`papers_registry.json` sha256:`{meta['inputs']['registry']}` · "
      f"`gap-ledger.json` sha256:`{meta['inputs']['ledger']}` · "
      f"`backlog-l3-map.json` sha256:`{meta['inputs']['map']}` · "
      f"`capability-graph.json` sha256:`{meta['inputs']['graph']}`")
    A("")
    A(f"**backlog 是现算的，不是抄的。** 口径：registry 里 `decision=extract` ∧ 无已交付产物。"
      f"实测 **{meta['n_backlog']} 篇**（naive 只数 `skill_card` 得 {meta['n_naive']} 篇，"
      f"扣掉 {len(led['excluded_enhancement'])} 篇增强交付才对齐 —— 即 PHASE5 登记的 N4）。")
    A("")
    A("## 1. 逐篇定位与它服务的 L3 责任")
    A("")
    A("| # | paper_id | 技术域 | 标题 | 服务的 L3（岗位）| 逐字证据 | 依据字段 | 证据类别 |")
    A("|---:|---|---|---|---|---|---|---|")
    for r in led["routed"]:
        served = "、".join(f"{s['name']}（{s['role_id']}）" for s in r["served_l3"])
        ev = "、".join(f"「{s['evidence']}」" for s in r["served_l3"])
        fields = "、".join(sorted({s["evidence_field"] for s in r["served_l3"]}))
        kinds = "、".join(sorted({s["basis"] for s in r["served_l3"]}))
        A(f"| {r['new_rank']} | `{r['paper_id']}` | {r['tech_domain']} | {r['title']} "
          f"| {served} | {ev} | {fields} | {kinds} |")
    A("")
    A("## 2. 新旧位次对照（旧 = 人工 priority + 论文分数；新 = 靶区缺口）")
    A("")
    A("| 新位次 | 旧位次 | 变化 | paper_id | 分层 | 靶区位次 | 主 L3 → 岗位 | 供给 |")
    A("|---:|---:|---:|---|---|---:|---|---|")
    for r in led["routed"]:
        p = r["primary_l3"]
        gap = f"{p['t1_rank']}" if p["in_t1"] else f"靶区外 #{p['ext_rank']}"
        mv = f"{r['moved']:+d}" if r["moved"] else "0"
        A(f"| {r['new_rank']} | {r['old_rank']} | {mv} | `{r['paper_id']}` | "
          f"{'**靶区内**' if r['tier'] == 'A' else '靶区外'} | {gap} | "
          f"{p['name']} → {p['role_id']} {p['role_title']} | "
          f"{next(s['supply_status'] for s in r['served_l3'] if s['name'] == p['name'])} |")
    A("")
    A("分层口径：**靶区内** = 至少一条服务的 L3 落在 F3 靶区一（A ∩ 无精选卡）；"
      "**靶区外** = 服务的 L3 在精选线都已有供给（或不是 A 类）⇒ **不占检索预算位**，"
      "只作为「出卡后挂到哪个既有责任」的落点。")
    A("")
    A("## 3. 覆盖率（一等输出）")
    A("")
    A(f"- backlog 分母（实测）：**{cov['n_backlog']}**")
    A(f"- 落到 ≥1 个 L3 责任：**{cov['n_mapped']}/{cov['n_backlog']}**"
      f"（覆盖率 {cov['coverage_ratio']:.0%}）")
    A(f"- **未落到任何 L3 责任**：{cov['n_unassigned']} 篇"
      + ("（无）" if not cov["unassigned"] else "：" + "、".join(
          f"`{u['paper_id']}`（{u['why']}）" for u in cov["unassigned"])))
    A(f"- 落在**靶区内**（占检索预算位）：**{cov['n_target_zone']}/{cov['n_backlog']}**"
      + ("：" + "、".join(f"`{t['paper_id']}`→#{t['gap_pos']} {t['l3']}({t['role_id']})"
                          for t in cov["target_zone"]) if cov["target_zone"] else ""))
    A(f"- 落在**靶区外**（已有供给，具名登记、不静默丢弃）：{len(cov['out_of_target_zone'])} 篇"
      + ("：" + "、".join(f"`{o['paper_id']}`→{o['best_l3']}({o['supply_status']})"
                          for o in cov["out_of_target_zone"])
         if cov["out_of_target_zone"] else ""))
    A(f"- 靠**旧顺序（分数）**决定位次的篇数：{cov['n_decided_by_legacy_tiebreak']}"
      f"（只在靶区位次相同的组内生效）")
    A(f"- 低置信映射（`capability_inference`，原文未命名业务责任）："
      f"**{cov['n_low_confidence']}** 篇"
      + ("：" + "、".join(f"`{o['paper_id']}`（{'、'.join(o['l3'])}）"
                          for o in cov["low_confidence"]) if cov["low_confidence"] else "（无）"))
    A(f"- 新旧位次位移：**{cov['n_moved']}/{cov['n_backlog']}** 篇位次发生变化，"
      f"最大位移 {cov['max_move']} 位")
    if cov["needs_review"]:
        A("- ⚠️ needs_review（靶区落点全靠**推得**的映射支撑）："
          + "、".join(f"`{n['paper_id']}`／{n['l3']}" for n in cov["needs_review"]))
    A("")
    A("## 4. 被判据挡住的东西（反后门）")
    A("")
    A(f"- `venue_tier` 在排序键里出现 0 次；`score` 只以预计算整数 `old_rank` 进入**末位** tiebreak。")
    A(f"- 「按论文 score 排」的顺序：{'、'.join('`%s`' % p for p in product['probes']['by_score'])}")
    A(f"- 接管后的顺序：{'、'.join('`%s`' % p for p in product['probes']['order'])}")
    A(f"- 两者**不同** ⇒ 缺口确实在主导排序；`--selftest` 另有一份注入样本自证。")
    A("")
    A("## 5. 判据自检")
    A("")
    A("| # | 判据 | 状态 |")
    A("|---|---|---|")
    for row in product["criteria"]:
        A(f"| {row['id']} | {row['what']} | {'✅ ' if row['ok'] else '❌ '}{row['detail']} |")
    A("")
    A("## 6. 未做 / 留给下游")
    A("")
    A("- **不执行扩充**：本表只改「下一条该萃取哪篇」的顺序，不抓论文、不出卡（S3 的边界）。")
    A("- **没有逐篇实读全文**：映射依据是 registry 的 `decision_reason`／`title`（逐字取证），"
      "不是论文全文；`capability_inference` 那类已显式标出。")
    A("- **不改契约/图谱/缺口账**：`build_gap_ledger.py` 只被 import 当 oracle，未改动一行。")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 主流程（被 main 与 selftest 共用）
# ---------------------------------------------------------------------------
def compute(registry_path=REGISTRY, graph_path=GRAPH, ledger_path=LEDGER,
            map_path=MAPFILE, weights_override: dict | None = None,
            check_f3: bool = True) -> tuple[dict, list[str]]:
    """跑完整条链，返回 (产物, 判据错误)。**任何一处输入缺失都抛 InputMissing。**"""
    registry = load_json(registry_path, "论文 registry")
    inspect_registry(registry)
    graph = load_graph(graph_path)
    ledger = load_json(ledger_path, "缺口账")
    mapobj = load_json(map_path, "backlog→L3 映射")

    weights = {k: float(v) for k, v in (ledger.get("weights") or {}).items()}
    if set(weights) != set(WEIGHT_KEYS):
        raise InputMissing(f"gap-ledger.json 的 weights 键不是 {WEIGHT_KEYS}：{sorted(weights)}")
    if weights_override:
        unknown = sorted(set(weights_override) - set(WEIGHT_KEYS))
        if unknown:
            # 拼错键会被静默忽略 → 直接判红（不是退 2：输入拿到了，是调用方写错了）
            return {}, [f"J3 未知权重键 {unknown}（拼错的键会被静默忽略）"]
        weights.update({k: float(v) for k, v in weights_override.items()})

    reg = derive_backlog(registry)
    backlog = reg["backlog"]
    if not backlog:
        raise InputMissing("registry 现算的 extract backlog 为空 —— "
                           "「没东西可排」不是「排完了没问题」（同 scan_secrets 的退出码 2）")

    mapping = index_map(mapobj)
    registry_by_id = {r["paper_id"]: r for r in registry["records"]}
    errs: list[str] = []
    errs += validate_mapping(backlog, mapping, graph, registry_by_id)
    errs += check_n4_counts(reg)

    pos = build_positions(ledger, weights)
    out = route(backlog, mapping, pos)
    errs += check_consistency(out["routed"], pos, ledger)
    skipped: list[str] = []
    freshness: dict = {}
    if check_f3:
        f3_ledger = f3_run(weights, graph_path)
        errs += check_against_f3(ledger, pos,
                                 [{"rank": w["rank"], "l3": w["l3"], "score": w["score"],
                                   "role_id": w["role_id"]} for w in f3_ledger["worklist"]])
        fresh_errs, freshness = check_input_freshness(graph_path, ledger, f3_ledger)
        errs += fresh_errs
    else:
        # ⚠️ 跳过必须**可见**：静默少测一条判据，正是本仓库记过的「判据骗人」形态。
        skipped.append("J4-现场 + J9：现场跑 F3 生成器的核对与输入新鲜度核对被 --no-f3 跳过"
                       "（**这两条没测**）")
    # 权重敏感性：至少一组扰动必须改变论文位次（承接 F3 的 J6 口径）。
    # ⚠️ 它**不依赖 F3 的现场调用**，故不能一起被 --no-f3 关掉（第一版把它们写在同一个 if 里，
    #    于是 --no-f3 会顺手把「改权重位次必须变」这条判据也静默关掉 —— 单边判据的变体）。
    orders = {"default": [r["paper_id"] for r in out["routed"]]}
    for i, patch in enumerate(({"flow_breadth": 0.0, "first_slice": 0.0},
                               {"zero_gap": 0.0, "first_slice": 0.0},
                               {"boundary_penalty": 3.0})):
        w2 = dict(weights)
        w2.update(patch)
        pos2 = build_positions(ledger, w2)
        out2 = route(backlog, mapping, pos2)
        orders[f"perturb{i + 1}:{','.join(sorted(patch))}"] = [r["paper_id"] for r in out2["routed"]]
    errs += assert_weight_sensitive(orders)
    out["weight_probe"] = {k: v for k, v in orders.items()}
    sk_errs, probes = check_order_vs_score(out["routed"])
    errs += sk_errs
    errs += assert_takeover_changed([r["paper_id"] for r in out["routed"]], out["legacy_order"])

    product = {
        "backlog": [{"paper_id": r["paper_id"], "title": r.get("title"),
                     "tech_domain": r.get("domain"), "priority": r.get("priority"),
                     "score": r.get("score"), "venue": r.get("venue"),
                     "venue_tier": r.get("venue_tier"),
                     "arxiv": (r.get("identifiers") or {}).get("arxiv"),
                     "skill_card_planned": (r.get("outputs") or {}).get("skill_card")}
                    for r in backlog],
        "naive_including_enhancement": reg["naive"],
        "excluded_enhancement": reg["enhancement_delivered"],
        "routed": out["routed"], "coverage": out["coverage"],
        "legacy_order": out["legacy_order"],
        "probes": probes,
        "weights": weights,
        "skipped_checks": skipped,
        "input_freshness": freshness,
        "criteria": [],
        "_meta": {
            "generated": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
            "generator": "paper2skills-research/scripts/route_backlog.py",
            "n_backlog": len(backlog), "n_naive": len(reg["naive"]),
            "inputs": {"registry": _sha(registry_path), "graph": _sha(graph_path),
                       "ledger": _sha(ledger_path), "map": _sha(map_path)},
            "exit_codes": "0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 内部错误（≠ 判红）",
        },
    }
    if "weight_probe" in out:
        product["weight_probe"] = out["weight_probe"]
    return product, errs


def _strip_volatile(text: str) -> str:
    return re.sub(r"^> 生成时间：.*$", "", text, flags=re.M)


# ---------------------------------------------------------------------------
# 自检（每条判据配篡改样本 + 反向控制）
# ---------------------------------------------------------------------------
def _selftest() -> int:
    ok = True
    banner = lambda s: print("\n--- " + s + " ---")
    detail: list[tuple[str, bool, str]] = []

    def rec(cid: str, what: str, good: bool, note: str):
        nonlocal ok
        detail.append((cid, what, good, note))
        print(("✅" if good else "❌") + f" [{cid}] {what}：{note}")
        ok &= good

    banner("J0 反向控制：干净输入必须 exit 0")
    try:
        product, errs = compute()
    except InputMissing as exc:
        print(f"✗ J0 输入没拿到（退出码 2）：{exc}")
        return EXIT_INPUT
    rec("J6", "干净输入跑完判据", not errs,
        f"12 篇现算、判据错误 {len(errs)} 条" if not errs else f"错误：{errs[:3]}")

    banner("J1 映射完整性（篡改样本）")
    registry = load_json(REGISTRY, "registry")
    graph = load_graph(GRAPH)
    ledger = load_json(LEDGER, "缺口账")
    reg = derive_backlog(registry)
    mapping = index_map(load_json(MAPFILE, "映射"))
    by_id = {r["paper_id"]: r for r in registry["records"]}
    baseline = validate_mapping(reg["backlog"], mapping, graph, by_id)
    rec("J1", "干净映射", not baseline, f"{len(baseline)} 条错误")

    import copy
    cases = [
        ("编造证据（G2b）", lambda m: m["p2s-2026-0007"]["l3"][0].__setitem__("evidence", "耗材复购召回率提升"),
         "逐字找不到"),
        ("不存在的 L3 名", lambda m: m["p2s-2026-0007"]["l3"][0].__setitem__("name", "复购召回实验"),
         "不在 151 名单"),
        ("把 C 类 L3 当放行位", lambda m: m["p2s-2026-0007"]["l3"][0].__setitem__("name", "培训与招聘支持"),
         "C 类"),
        ("证据字段声明错位", lambda m: m["p2s-2026-0007"]["l3"][0].__setitem__("evidence_field", "title"),
         "声明与实物矛盾"),
        ("推得的映射标 high", lambda m: m["p2s-2026-0022"]["l3"][0].__setitem__("confidence", "high"),
         "confidence"),
        ("孤儿条目（非 backlog）", lambda m: m.__setitem__(
            "p2s-2026-0001", {"paper_id": "p2s-2026-0001",
                              "l3": [{"name": "复购实验", "evidence": "x",
                                      "evidence_field": "title",
                                      "basis": "business_action", "confidence": "high"}]}),
         "死重"),
        ("缺条目", lambda m: m.pop("p2s-2026-0034"), "没有任何映射"),
        ("近邻表无理由", lambda m: m["p2s-2026-0012"]["alternatives"][0].__setitem__("why_not", ""),
         "否决理由"),
    ]
    for label, mutate, expect in cases:
        m = copy.deepcopy(mapping)
        try:
            mutate(m)
        except Exception as exc:                     # noqa: BLE001
            rec("J1", label, False, f"样本自身构造失败：{exc}")
            continue
        got = validate_mapping(reg["backlog"], m, graph, by_id)
        rec("J1", label, bool(got) and any(expect in g for g in got),
            (got[0][:90] if got else "**没抓住**（判据失效）"))

    banner("J2 覆盖率 / 空输入（篡改样本）")
    cov = product["coverage"]
    rec("J2", "覆盖率恒等式", cov["n_mapped"] + cov["n_unassigned"] == cov["n_backlog"],
        f"{cov['n_mapped']}+{cov['n_unassigned']}=={cov['n_backlog']}")
    m2 = copy.deepcopy(mapping)
    m2.pop("p2s-2026-0032")
    pos = build_positions(ledger, {k: float(v) for k, v in ledger["weights"].items()})
    out2 = route(reg["backlog"], m2, pos)
    dropped = [u["paper_id"] for u in out2["unassigned"]]
    rec("J2", "缺映射的论文必须**具名**出现在未落名单（不静默丢弃）",
        dropped == ["p2s-2026-0032"], f"未落名单 = {dropped}")
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty.json"
        empty.write_text(json.dumps({"records": []}), encoding="utf-8")
        try:
            compute(registry_path=empty)
            rec("J2", "空 backlog → 退出码 2", False, "**居然跑通了**（空输入被当成通过）")
        except InputMissing as exc:
            rec("J2", "空 backlog → 退出码 2", True, str(exc)[:70])
        missing = Path(td) / "nope.json"
        try:
            compute(ledger_path=missing)
            rec("J2", "缺输入文件 → 退出码 2", False, "**居然跑通了**")
        except InputMissing as exc:
            rec("J2", "缺输入文件 → 退出码 2", True, str(exc)[:70])

    banner("J3 缺口驱动：改权重位次必须变（承接 F3 的 J6）")
    probe = product.get("weight_probe") or {}
    rec("J3", "真实数据：≥2 组权重产出不同位次",
        len({tuple(v) for v in probe.values()}) > 1,
        f"{len(probe)} 组权重 → {len({tuple(v) for v in probe.values()})} 种顺序；"
        f"默认 {probe.get('default', [])[:3]} → 扰动 "
        f"{[v[:3] for k, v in probe.items() if k != 'default'][:1]}")
    same = {"a": ["P1", "P2"], "b": ["P1", "P2"]}
    rec("J3", "篡改：两组权重同序", bool(assert_weight_sensitive(same)),
        (assert_weight_sensitive(same) or ["**没抓住**"])[0])
    rec("J3", "篡改：只喂一组权重（单边判据）", bool(assert_weight_sensitive({"a": ["P1"]})),
        (assert_weight_sensitive({"a": ["P1"]}) or ["**没抓住**"])[0])

    banner("J4 与 F3 逐条一致（篡改样本）")
    errs4 = check_against_f3(ledger, pos, f3_positions(ledger["weights"]))
    rec("J4", "已入库 + 现场 F3 双重核对", not errs4, f"{len(errs4)} 条不符")
    tampered = copy.deepcopy(ledger)
    wl = tampered["worklist"]
    wl[0], wl[5] = wl[5], wl[0]                      # 把第 1 位与第 6 位对调
    for i, w in enumerate(wl, 1):
        w["rank"] = i
    errs4b = check_against_f3(tampered, pos, f3_positions(ledger["weights"]))
    rec("J4", "篡改：靶区位次被对调", bool(errs4b), (errs4b[0][:90] if errs4b else "**没抓住**"))
    tampered2 = copy.deepcopy(ledger)
    tampered2["worklist"][0]["score"] = 99.0
    errs4c = check_against_f3(tampered2, pos, f3_positions(ledger["weights"]))
    rec("J4", "篡改：靶区分数被改", bool(errs4c), (errs4c[0][:90] if errs4c else "**没抓住**"))
    bad_tier = [dict(r) for r in out2["routed"]]
    if bad_tier and any(r["tier"] == "A" for r in bad_tier):
        # 把第一篇（靶区内）与末篇（靶区外）的层级标签对调，**保持列表顺序不变** ——
        # 上一版这里重排了列表，等于顺手把倒挂修好了，样本自己把证据擦掉。
        bad_tier[0]["tier_order"] = 1
        bad_tier[-1]["tier_order"] = 0
        errs4d = check_consistency(bad_tier, pos, ledger)
        rec("J4", "篡改：靶区内外层级倒挂", bool(errs4d),
            (errs4d[0][:90] if errs4d else "**没抓住**"))

    banner("J8 接管确实改变了顺序（篡改样本 + 反向控制）")
    same_order = product["legacy_order"]
    rec("J8", "真实数据：新序 ≠ 旧序", not assert_takeover_changed(
        [r["paper_id"] for r in product["routed"]], same_order),
        f"{product['coverage']['n_moved']}/{product['coverage']['n_backlog']} 篇位次移动，"
        f"最大位移 {product['coverage']['max_move']}")
    rec("J8", "篡改：新序 == 旧序（接管空转）", bool(assert_takeover_changed(same_order, same_order)),
        (assert_takeover_changed(same_order, same_order) or ["**没抓住**"])[0][:90])
    rec("J8", "篡改：两个序列成员不同（不可比）", bool(
        assert_takeover_changed(same_order[:-1], same_order)),
        (assert_takeover_changed(same_order[:-1], same_order) or ["**没抓住**"])[0][:90])
    rec("J8", "篡改：空序列（单边）", bool(assert_takeover_changed([], same_order)),
        (assert_takeover_changed([], same_order) or ["**没抓住**"])[0][:90])

    banner("J5 反后门：排序键 AST 扫描 + 分数注入样本")
    src = Path(__file__).read_text(encoding="utf-8")
    rec("J5", "自扫干净脚本", not (scan_route_key(src) + check_no_venue_in_legacy_key(src)),
        "0 命中")
    tampered_srcs = [
        ("主键换成 paper_score",
         src.replace('return (rec["tier_order"], rec["gap_pos"], rec["old_rank"])',
                     'return (rec["paper_score"], rec["gap_pos"], rec["old_rank"])'),
         "主排序键里出现 `paper_score`"),
        ("venue_tier 进第二位",
         src.replace('return (rec["tier_order"], rec["gap_pos"], rec["old_rank"])',
                     'return (rec["tier_order"], rec["venue_tier"], rec["old_rank"])'),
         "venue_tier"),
        ("末位 tiebreak 里塞 score",
         src.replace('return (rec["tier_order"], rec["gap_pos"], rec["old_rank"])',
                     'return (rec["tier_order"], rec["gap_pos"], rec["score"])'),
         "末位 tiebreak"),
        ("排序键函数改名（让扫描器看别处）",
         src.replace("def route_sort_key(", "def route_sort_key_renamed("),
         "扫描目标"),
    ]
    for label, tsrc, expect in tampered_srcs:
        got = scan_route_key(tsrc)
        rec("J5", f"篡改：{label}", bool(got) and any(expect in g for g in got),
            (got[0][:90] if got else "**没抓住**（判据失效）"))
    neg = src.replace('return (rec["tier_order"], rec["gap_pos"], rec["old_rank"])',
                      'return (rec["tier_order"], rec["gap_pos"], rec["old_rank"])')
    rec("J5", "反向控制：score 只在**另一个**函数里（旧顺序）→ 不得误报",
        not scan_route_key(neg), "旧顺序函数不被路由键扫描器误伤")
    inj_errs, n_inj = check_score_override_injection(reg["backlog"], mapping, ledger, graph)
    rec("J5", "注入样本：按论文分数排 vs 按缺口排", not inj_errs,
        f"{n_inj} 篇样本，两序不同" if not inj_errs else inj_errs[0][:90])
    rec("J5", "真实数据：接管序 ≠ 按 score 序 ≠ 按 venue 序",
        not check_order_vs_score(product["routed"])[0],
        f"score 序 {product['probes']['by_score'][:3]}… vs 接管序 {product['probes']['order'][:3]}…")

    banner("J7 registry 自洽（N4 双层计数）")
    rec("J7", "naive 15 = backlog 12 + 增强交付 3",
        len(product["naive_including_enhancement"]) == 15
        and len(product["backlog"]) == 12 and len(product["excluded_enhancement"]) == 3,
        f"naive={len(product['naive_including_enhancement'])} "
        f"backlog={len(product['backlog'])} "
        f"增强={[e['paper_id'] for e in product['excluded_enhancement']]}")
    fake = copy.deepcopy(reg)
    fake["integrity_errors"] = []
    fake_registry = copy.deepcopy(registry)
    for r in fake_registry["records"]:
        if r["paper_id"] == "p2s-2026-0001":
            r["status"] = "shortlisted"              # 卡明明在，标签却说没出
    fake = derive_backlog(fake_registry)
    rec("J7", "篡改：status 与实物矛盾", bool(check_n4_counts(fake)),
        (check_n4_counts(fake) or ["**没抓住**"])[0][:90])

    banner("J9 输入新鲜度（风险 N6：基线数字可能是过期照片）")
    f3_led = f3_run(ledger["weights"])
    f_errs, f_info = check_input_freshness(GRAPH, ledger, f3_led)
    rec("J9", "真实数据：缺口账声明的图谱 sha vs 现状",
        f_info["verdict"] in ("fresh", "stale_immaterial"),
        f"{f_info['ledger_declared_graph_sha256']} → {f_info['graph_sha256_now']}，"
        f"判为 {f_info['verdict']}"
        + (f"（{f_info.get('why_immaterial', '')[:40]}…）" if f_info.get("why_immaterial") else ""))
    tam = copy.deepcopy(ledger)
    for r in tam["rows"]:
        if r["l3"] == "复购实验":
            r["supply"]["n_legacy"] = 999          # 供给列被改 ⇒ 排序依据变了
    tam["_meta"]["graph_sha256"] = _sha(GRAPH)     # ⚠️ 显式把 sha 设成「与现状一致」
    tA_errs, tA_info = check_input_freshness(GRAPH, tam, f3_led)
    rec("J9", "篡改（结构洞）：sha 与现状**一致** + 排序依据被改 ⇒ 必须判红",
        bool(tA_errs) and tA_info["verdict"] == "material_mismatch",
        (tA_errs[0][:95] if tA_errs else "**没抓住**（内容被改而 sha 自述新鲜 ⇒ 照样出表）"
         f"；verdict={tA_info['verdict']}"))
    tamB = copy.deepcopy(ledger)
    for r in tamB["rows"]:
        if r["l3"] == "复购实验":
            r["supply"]["n_legacy"] = 999
    tamB["_meta"]["graph_sha256"] = "0" * 64       # sha 显式过期 + 内容被改
    t_errs, t_info = check_input_freshness(GRAPH, tamB, f3_led)
    rec("J9", "篡改：过期底本 + 排序依据被改 ⇒ 判红",
        bool(t_errs) and t_info["verdict"] == "material_mismatch",
        (t_errs[0][:95] if t_errs else "**没抓住**（过期账照样出表）"))
    tam2 = copy.deepcopy(ledger)
    tam2["_meta"].pop("graph_sha256", None)
    t2_errs, t2_info = check_input_freshness(GRAPH, tam2, f3_led)
    rec("J9", "篡改：抹掉生成时 sha ⇒ 不许当「新鲜」",
        bool(t2_errs) and t2_info["verdict"] == "unknown",
        (t2_errs[0][:95] if t2_errs else "**没抓住**"))
    tam3 = copy.deepcopy(ledger)
    tam3["_meta"]["graph_sha256"] = _sha(GRAPH)
    t3_errs, t3_info = check_input_freshness(GRAPH, tam3, f3_led)
    rec("J9", "反向控制：sha 一致时不得误报", (not t3_errs) and t3_info["verdict"] == "fresh",
        f"verdict={t3_info['verdict']}，错误 {len(t3_errs)} 条")

    banner("产物一致性（--check 用同一条链）")
    strip = _strip_volatile
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "p.json"
        m = Path(td) / "p.md"
        prod2, _ = compute()
        j.write_text(json.dumps(prod2, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        m.write_text(render_md(prod2), encoding="utf-8")
        same_json = json.loads(j.read_text(encoding="utf-8"))["coverage"] == product["coverage"]
        rec("J6", "同链两次构建产物一致", same_json, "coverage 逐字段相等")
        mut = json.loads(j.read_text(encoding="utf-8"))
        mut["coverage"]["n_mapped"] = 0
        rec("J6", "篡改：产物被改", mut["coverage"] != product["coverage"], "差异可检出")
        del strip

    n_bad = sum(1 for _, _, good, _ in detail if not good)
    print()
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} —— {len(detail)} 条，"
          f"{len(detail) - n_bad} 过 / {n_bad} 败；"
          f"每条判据都配了能打红的篡改样本与反向控制")
    return EXIT_PASS if ok else EXIT_RED


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="extract backlog 交给靶区排序接管（PHASE6 S3 / B9）")
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--graph", type=Path, default=GRAPH)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--map", type=Path, default=MAPFILE)
    ap.add_argument("--weights", type=Path, help="覆盖靶区权重（键同 F3 的 DEFAULT_WEIGHTS）")
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    ap.add_argument("--md-out", type=Path, default=OUT_MD)
    ap.add_argument("--no-f3", action="store_true",
                    help="跳过「现场跑 F3 生成器」的核对（默认不跳 —— 跳过即那条判据没测）")
    ap.add_argument("--check", action="store_true", help="复跑并与已入库产物比对")
    ap.add_argument("--selftest", action="store_true", help="判据 × 篡改样本")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    try:
        override = None
        if args.weights:
            if not args.weights.is_file():
                raise InputMissing(f"权重文件不存在：{args.weights}")
            override = json.loads(args.weights.read_text(encoding="utf-8"))
        product, errs = compute(args.registry, args.graph, args.ledger, args.map,
                                override, check_f3=not args.no_f3)
    except InputMissing as exc:
        print(f"✗ 输入没拿到（退出码 2，**不是通过**）：{exc}")
        return EXIT_INPUT

    if errs:
        print("✗ 判据不成立，拒绝写出接管表：")
        for e in errs:
            print("   ", e)
        return EXIT_RED

    src = Path(__file__).read_text(encoding="utf-8")
    errs_static = scan_route_key(src) + check_no_venue_in_legacy_key(src)
    if errs_static:
        print("✗ 反后门自扫不通过：")
        for e in errs_static:
            print("   ", e)
        return EXIT_RED

    skipped = product.get("skipped_checks") or []
    for s_ in skipped:
        print(f"⚠️  跳过：{s_}")
    freshness = product.get("input_freshness") or {}
    cov = product["coverage"]
    product["criteria"] = [
        {"id": "J1", "what": "映射完整性（白名单 + 逐字证据 + 无孤儿）", "ok": True,
         "detail": f"{cov['n_mapped']}/{cov['n_backlog']} 篇有映射，证据逐字命中"},
        {"id": "J2", "what": "覆盖率一等输出 / 未落具名 / 空输入退 2", "ok": True,
         "detail": f"落 L3 {cov['n_mapped']}/{cov['n_backlog']}，未落 {cov['n_unassigned']}，"
                   f"靶区内 {cov['n_target_zone']}"},
        {"id": "J3", "what": "改靶区权重位次必须变（承接 F3 J6）", "ok": True,
         "detail": f"{len(product.get('weight_probe', {}))} 组权重 → "
                   f"{len({tuple(v) for v in product.get('weight_probe', {}).values()})} 种顺序"},
        {"id": "J4", "what": "与 F3 逐条一致（已入库 + 现场生成器）", "ok": True,
         "detail": ("靶区 31 条逐条相等（已入库 + 现场生成器两处）" if not skipped
                    else "**只核了已入库产物**；现场生成器那半被 --no-f3 跳过（跳过即没测）")},
        {"id": "J5", "what": "反后门：排序键 AST 扫描 + 分数注入样本", "ok": True,
         "detail": "venue_tier 0 命中；接管序 ≠ 按 score 序"},
        {"id": "J6", "what": "反向控制：干净输入 exit 0", "ok": True, "detail": "本行即证据"},
        {"id": "J8", "what": "接管确实改变了旧顺序（双边）", "ok": True,
         "detail": f"{cov['n_moved']}/{cov['n_backlog']} 篇位次动了，最大位移 {cov['max_move']}"},
        {"id": "J3b", "what": "位次靠旧顺序 tiebreak 决定的篇数（可见账）", "ok": True,
         "detail": f"{cov['n_decided_by_legacy_tiebreak']} 篇；低置信映射 "
                   f"{cov.get('n_low_confidence', 0)} 篇"},
        {"id": "J9", "what": "输入新鲜度：缺口账声明的图谱 sha vs 现状（风险 N6）", "ok": True,
         "detail": ("图谱未变（fresh）" if freshness.get("verdict") == "fresh" else
                    f"图谱已变（{freshness.get('ledger_declared_graph_sha256')}→"
                    f"{freshness.get('graph_sha256_now')}），现算核对："
                    f"{freshness.get('verdict')}"
                    + (f"（{freshness.get('why_immaterial')}）"
                       if freshness.get("verdict") == "stale_immaterial" else ""))
                   if freshness else "未测（--no-f3）"},
        {"id": "J7", "what": "registry 自洽（N4：15 = 12 + 3）", "ok": True,
         "detail": f"naive {product['_meta']['n_naive']} − 增强交付 "
                   f"{len(product['excluded_enhancement'])} = {product['_meta']['n_backlog']}"},
    ]

    text_json = json.dumps(product, ensure_ascii=False, indent=1) + "\n"
    text_md = render_md(product)

    if args.check:
        bad = []
        for p, new in ((args.json_out, text_json), (args.md_out, text_md)):
            if not Path(p).is_file():
                bad.append(f"{_rel(Path(p))} 不存在：先跑一次生成")
                continue
            old = Path(p).read_text(encoding="utf-8")
            if Path(p).suffix == ".json":
                o = json.loads(old)
                o.pop("_meta", None)
                o.pop("criteria", None)
                n = json.loads(new)
                n.pop("_meta", None)
                n.pop("criteria", None)
                if o != n:
                    diff = sorted(k for k in set(o) | set(n) if o.get(k) != n.get(k))
                    bad.append(f"{_rel(Path(p))} 与 registry/缺口账现状不一致（差异键：{diff[:6]}）")
            else:
                if _strip_volatile(old) != _strip_volatile(new):
                    bad.append(f"{_rel(Path(p))} 与现状不一致（逐行比对）")
        if bad:
            print("✗ 接管表产物已过期或与现状不一致：")
            for e in bad:
                print("   ", e)
            return EXIT_RED
        print("✓ 接管表 JSON 与报告均与 registry / 缺口账 / 映射现状一致（已剔易变字段）")
        return EXIT_PASS

    for p in (args.json_out, args.md_out):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text_json, encoding="utf-8")
    args.md_out.write_text(text_md, encoding="utf-8")

    print(f"✓ 写入 {_rel(args.json_out)}（{args.json_out.stat().st_size/1024:.0f} KB）")
    print(f"✓ 写入 {_rel(args.md_out)}（{args.md_out.stat().st_size/1024:.0f} KB）")
    print(f"  backlog（现算）：{product['_meta']['n_backlog']} 篇"
          f"（naive 只数 skill_card 得 {product['_meta']['n_naive']} 篇，"
          f"扣 {len(product['excluded_enhancement'])} 篇增强交付 —— N4）")
    print(f"  覆盖率：落 L3 {cov['n_mapped']}/{cov['n_backlog']}，未落 {cov['n_unassigned']}；"
          f"靶区内 {cov['n_target_zone']}，靶区外 {len(cov['out_of_target_zone'])}")
    print(f"  靶区内（占检索预算位）：" + ("、".join(
        f"{t['paper_id']}→#{t['gap_pos']} {t['l3']}" for t in cov["target_zone"]) or "（无）"))
    print(f"  反后门：接管序 {product['probes']['order'][:4]}… ≠ 按 score 序 "
          f"{product['probes']['by_score'][:4]}…")
    if product.get("input_freshness", {}).get("stale"):
        fr = product["input_freshness"]
        print(f"  ⚠️ 输入新鲜度：缺口账声明的图谱 sha {fr['ledger_declared_graph_sha256']} "
              f"≠ 现状 {fr['graph_sha256_now']} ⇒ {fr['verdict']}"
              + (f"（{fr.get('why_immaterial')}）" if fr.get("verdict") == "stale_immaterial" else ""))
    print(f"  权重敏感性：{len({tuple(v) for v in product.get('weight_probe', {}).values()})} 种顺序"
          f"（{len(product.get('weight_probe', {}))} 组权重）")
    return EXIT_PASS


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InputMissing as _exc:                     # 兜底：任何一条路径漏了都退 2
        print(f"✗ 输入没拿到（退出码 2）：{_exc}")
        sys.exit(EXIT_INPUT)
    except Exception:                                # noqa: BLE001
        print("✗ 门禁内部错误（退出码 3，**不是判红**）：")
        traceback.print_exc()
        sys.exit(EXIT_INTERNAL)
