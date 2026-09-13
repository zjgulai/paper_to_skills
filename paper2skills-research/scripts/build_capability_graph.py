#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_capability_graph.py — 生成**唯一机读的五层能力图谱**（PHASE6 F2 / T-A）。

## 它是什么，以及它**不是**什么

它把「需求侧组织模型」与「供给侧 Skill 卡」接成一张可查询的图，输出
`paper2skills-vault/07-资源库/capability-graph.json`。

⚠️ **它不是第二份分类事实源。** 这条是本卡存在的理由之一（硬顺序约束：F2 先行，
否则 F4/F3/S1 会各自造第二份分类表）。分层与归口：

| 层 | 事实源在哪 | 本图做什么 |
|---|---|---|
| L1 面(4) / L2 域(8) / L3 责任名(151) | 材料 `organization-graph.json` + `role-catalog.json`；产品侧已派生一份 `dsh-paper2skills/data/taxonomy.json` | **join 并逐项断言相等**（`--check-taxonomy`），不重新定义 |
| 64 格（8 FLOW × 8 STG）与 `cell_kind` | 材料 `collaboration-graph.json` 的 `flow_stage_bindings` + `protocol.stages[].model_participation` | **本图首次落库**（此前没有任何地方有这 64 格） |
| A/B/C 算法可服务性 | `_survey_org_model.md` §F.3 的逐条判定（151 条） | 解析进图，供缺口账 join |
| A 类**边界**条目（10 条）与**无 A 类**岗位（10 个） | `_survey_org_model.md` §F.5 / §F.6 | 解析进图（`a_scope`），**并为 T-C 的阈值敏感性提供落点**；解析值与机器算出值**逐项断言相等** |
| 卡 ↔ L3 | **精选线**卡端事实源 `paper2skills-vault/07-资源库/card-classification.json`（F5 产物，146 张） | join 147→146 张；其中 93 张是从产品侧 `classification.json`（1338 条）**逐字继承**的，由 F5 的 J4 零漂移门禁守着 |
| 方案层（L3 方案域 8 份） | `paper2skills-vault/07-资源库/solutions/*.md`（S1 逐 FLOW 交付） | **派生**：读文档 frontmatter → `solutions[]` + 该 FLOW 8 格的 `solution_refs`；文档删了图自己归零，不需手改图 |

## 必须入库的两个判据（本卡验收项）

1. **MECE 性质**：每一层标明是**划分**还是**覆盖**，并给出主归属与重数。
   前两层（岗位/域、151 L3、64 格）是划分；20 SCN、方案、卡是覆盖。
   三个「覆盖」层的重数必须**算出来**而不是声称（见 `mece` 与 `counts`）。
2. **`cell_kind`**：64 格逐格标 M/R/D，**由 `model_participation` 机器导出**，
   映射表显式且**对未知取值直接失败**（不静默退回默认值 —— 本仓库的既有纪律）。
   ⚠️ 风险 N4：D/R 格出现算法模型即判错，故每格带 `algorithm_policy`。

## 三个 join 陷阱（`_survey_org_model.md` §H.1，全部显式处理）

1. **`plane_id` 只在 `organization-graph.json`**（`role-catalog.json` 只有 `group`）。
   只读 role-catalog 的实现会**静默丢掉平面维度**。本脚本从 organization-graph 取，
   并断言 50/50 与 `group` 的关系（两者**刻意错位**：AGT-040 域=财务与合规但面=业务运营）。
2. **`scenarios` 118 条 vs 传递闭包 701 条**（差 5.9 倍）。两列都给，各自命名清楚，
   并在 `mece` 里说明「选哪一种取决于问题，混用会让覆盖场景数虚高数倍」。
3. **AGT-012 ↔ SCN-020 的孤立绑定**：全量 50×118 里**仅此 1 条**无法由本岗位任一
   FLOW 的 `scenario_ids` 解释。记进 `join_traps`，不悄悄吞掉。

## 用法

    python3 paper2skills-research/scripts/build_capability_graph.py            # 生成
    python3 paper2skills-research/scripts/build_capability_graph.py --check    # 复跑比对（门禁用）
    python3 paper2skills-research/scripts/build_capability_graph.py --cell FLOW-01/STG-04
    python3 paper2skills-research/scripts/build_capability_graph.py --selftest
    python3 paper2skills-research/scripts/build_capability_graph.py --check-taxonomy

退出码：0 正常；1 判据失败（--check 有差异 / 划分断言不成立）；2 输入缺失（**不是「没问题」**）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAULT = REPO / "paper2skills-vault"
OUT_DEFAULT = VAULT / "07-资源库" / "capability-graph.json"
SURVEY = REPO / "paper2skills-research" / "reports" / "_survey_org_model.md"
TAXONOMY = Path("/Users/lute/project/Magpie-Horch/packages/capabilities/dsh-paper2skills/data/taxonomy.json")
# 卡 ↔ L3 的事实源：**精选线**的卡端落点（F5 产物）。产品侧那份 1338 条是另一个语料，
# 由 F5 的生成器逐条比对（J4 零漂移），本图只消费、不重判。
CARD_CLASSIFICATION = VAULT / "07-资源库" / "card-classification.json"
# L3 方案域的**源**（S1 逐 FLOW 交付的长文）。图里的 `solutions` 与 `cells[].solution_refs`
# 由它派生 —— 文档在、图上就有；文档删了、图自己会说「0 份」，不需要人手改图。
SOLUTIONS_DIR = VAULT / "07-资源库" / "solutions"
PRESET_DIR = Path(os.path.expanduser("~/.dsh/.agent-presets"))

MATERIAL_ROOT = Path(os.environ.get("AI_ORG_MATERIAL_ROOT", "/Users/lute/project/AI组织变革"))

# ---------------------------------------------------------------------------
# `cell_kind` 的导出判据（本卡验收项 ②）
# ---------------------------------------------------------------------------
# 左：`protocol.stages[].model_participation` 的**实际取值**（材料原样，不是我们造的词）。
# 右：(kind, 该格的模型权限说明)。
#
# ⚠️ 映射表必须**对未知取值直接失败**，不许有 `default` —— 材料加一个新阶段语义时，
#    静默归到某一类会让 40 个 D/R 格被误当方法格（风险 N4），而报告一切正常。
MODEL_PARTICIPATION_KIND: dict[str, tuple[str, str]] = {
    "none_required":
        ("D", "无模型参与"),
    "optional_bounded_classification":
        ("R", "模型只做**有界分类**；主判据是契约与规则"),
    "case_agent_initialized_after_acceptance":
        ("D", "模型在此仅被**装配**（选主岗位、钉 Bundle），不产出业务判定"),
    "primary_case_agent_with_on_demand_skills":
        ("M", "模型是**主引擎**（按需装配 Skills）"),
    "optional_read_only_semantic_evaluator_delegate":
        ("R", "模型只读评估语义；不得扩权、不得授权资产"),
    "case_agent_may_submit_action_intent_only":
        ("D", "模型只能**提交 Action Intent**；动作由策略闸与执行代理做"),
    "case_agent_summarizes_against_authoritative_receipts":
        ("M", "模型对**权威回执**做核验、关闭与异步学习"),
}

# 风险 N4 的机读形式：每类格允许什么。
ALGORITHM_POLICY = {
    "M": "算法模型在此接入（A/B 类责任名可挂方法）",
    "R": "契约与判据为主；模型只做有界分类，**不得作主引擎**",
    "D": "**无模型参与**；不消耗检索预算",
}

Q = "\n" + "=" * 78 + "\n"


# ---------------------------------------------------------------------------
# 读入
# ---------------------------------------------------------------------------
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_materials(root: Path) -> tuple[dict, dict, dict]:
    org = json.loads((root / "docs/04-organization/organization-graph.json").read_text(encoding="utf-8"))
    cat = json.loads((root / "docs/05-agents/role-catalog.json").read_text(encoding="utf-8"))
    coll = json.loads((root / "docs/07-orchestration/collaboration-graph.json").read_text(encoding="utf-8"))
    return org, cat, coll


def parse_serviceability(survey: Path) -> dict[tuple[str, str], str]:
    """从调查报告 §F.3 的逐条判定表解析 (AGT, 责任名) → A/B/C。

    ⚠️ 解析的是**已交付的判定结论**，不是重新判定一遍：
        A/B/C 是人的口径判断（阈值「算法可作主引擎」），重判会造出第二份名单。
        这里只做形式转换，并把解析条数当作**前提**断言（=151，少一条即失败）。
    """
    text = survey.read_text(encoding="utf-8")
    sec = text.split("### F.3")[1].split("### F.4")[0]
    out: dict[tuple[str, str], str] = {}
    for line in sec.splitlines():
        m = re.match(r"^\|\s*(AGT-\d{3})\s*\|[^|]*\|\s*([^|]+?)\s*\|\s*([ABC])\s*\|", line)
        if m:
            out[(m.group(1), m.group(2).strip())] = m.group(3)
    return out


def parse_a_boundary(survey: Path) -> list[dict]:
    """从 §F.5 解析 A 类中的**边界条目**（10 条）：责任名 + 计入 A 的理由 + 降级条件。

    ⚠️ 与 §F.3 同一条纪律：解析的是**已交付的口径判断**，不是重判。
        这 10 条正是 T-C（缺口账）要用的东西 —— 若把「A 类 73 条」当成铁板一块，
        下游就会把「A 是否为主引擎尚有争议」的条目和已定条目一视同仁地排进检索预算。
        **口径落在 §F.5 原文里有名字**，不是本脚本发明的。
    """
    text = survey.read_text(encoding="utf-8")
    sec = text.split("### F.5")[1].split("### F.6")[0]
    out: list[dict] = []
    for line in sec.splitlines():
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", line)
        if not m:
            continue
        name = m.group(1).strip()
        if name in ("责任名", "---") or name.startswith("-"):
            continue
        out.append({"name": name, "rationale": m.group(2).strip(),
                    "downgrade_condition": m.group(3).strip()})
    return out


def parse_roles_without_a(survey: Path) -> list[str]:
    """从 §F.6 解析**无任何 A 类责任名**的岗位（10 个）。用途同 §F.5：给排序一个已交付的口径锚点。"""
    text = survey.read_text(encoding="utf-8")
    sec = text.split("### F.6")[1].split("## G.")[0]
    return sorted(set(re.findall(r"^\|\s*(AGT-\d{3})\s*\|", sec, re.M)))


def _rel(p: Path) -> str:
    """仓库内相对路径；仓库外原样返回。

    门禁缺陷 #13（`relative_to(REPO)` 对仓库外路径抛 ValueError）的**同族第三次**出现：
    F3 修在 `build_gap_ledger.py`，本脚本当时没被扫到，而它有两处 `relative_to(REPO)`。
    改成统一走 `_rel()`，并配 `--selftest` 的 3 条探针（仓库内 / 仓库外 / 前缀相似）。
    """
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def load_card_l3() -> dict[str, list[str]]:
    """vault 卡名 → L3 列表（F5 的卡端事实源）。

    ⚠️ **缺失即 SystemExit，不返回空**：静默返回空会让 146 张卡全部退化成「无 L3」，
    而图上一切下游计数（`cards_with_l3`、格子的 `card_refs`、缺口账）都会跟着静默变错
    —— 「拿不到就炸，不要给默认值」。
    """
    if not CARD_CLASSIFICATION.is_file():
        raise SystemExit(f"✗ 缺少 {_rel(CARD_CLASSIFICATION)}（F5 的卡端分类事实源）"
                         f"\n  先跑：python3 paper2skills-research/scripts/build_card_classification.py --write")
    doc = json.loads(CARD_CLASSIFICATION.read_text(encoding="utf-8"))
    if doc.get("total") != len(doc.get("items", [])):
        raise SystemExit(f"✗ {_rel(CARD_CLASSIFICATION)} 的 total 与 items 条数对不上")
    return {i["id"]: list(i.get("l3") or []) for i in doc["items"]}


def load_solutions() -> list[dict]:
    """L3 方案域：读 `07-资源库/solutions/*.md` 的 frontmatter。

    **文档是源，图是派生。** 目录不存在 = 0 份（S1 未交付），这是**合法状态**而不是错误；
    但**文档存在却没有 `flow_id`** 即 SystemExit —— 那说明那份文档没资格当方案域，
    静默跳过会让它变成一份谁也查不到的孤儿（`--cell` 永远答「方案 0」而文档明明在）。

    ⚠️ 与 `load_card_l3` 的差别是有意的：那个的缺失会让 146 张卡静默退化，所以缺文件即炸；
    这个的缺失只意味着「那一层还没交付」，如实报 0 才对。
    """
    if not SOLUTIONS_DIR.is_dir():
        return []
    out = []
    for p in sorted(SOLUTIONS_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        fm = {}
        if m:
            for line in m.group(1).splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                k, _, v = line.partition(":")
                fm[k.strip()] = v.strip().strip('"').strip("'")
        flow = fm.get("flow_id")
        if not flow:
            raise SystemExit(f"✗ {_rel(p)} 有 frontmatter 但没有 `flow_id` —— "
                             f"方案域必须声明它属于哪条 FLOW（拿不到就炸，不给默认值）")
        # solution_id：优先取显式字段，否则用文件名的 `FLOW-NN-` 前缀（方案域是「一条 FLOW 一份」）。
        m2 = re.match(r"^(FLOW-\d{2})-", p.stem)
        out.append({
            "solution_id": fm.get("solution_id") or (m2.group(1) if m2 else p.stem),
            "flow_id": flow,
            "title": fm.get("title") or p.stem,
            "status": fm.get("status"),
            "path": _rel(p),
            # ⚠️ **刻意不记 `bytes`**：S1 期间实测 —— 记了它，只要方案域文档被编辑一次
            #    （哪怕只加一节正文），`--check` 就报「与现状不一致」，而图其实没变。
            #    那是把「内容变更」误报成「图漂移」。稳定性判据要盯**结构性字段**
            #    （flow_id / path / status），不是文件大小。
        })
    return out


def load_wiring() -> dict[str, list[str]]:
    """岗位 → 该岗 preset 已挂载的技能名（`x_lute.skills.subset`）。

    ⚠️ 这是**运行时**的家（`~/.dsh/.agent-presets`），不是仓库事实源。
       取不到就给 `null` 并把覆盖情况报出来 —— 「问不到」与「没接线」是两回事。
    """
    out: dict[str, list[str]] = {}
    if not PRESET_DIR.is_dir():
        return out
    for d in sorted(PRESET_DIR.glob("agt-*")):
        mp = d / "manifest.json"
        if not mp.is_file():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        subset = (((m.get("x_lute") or {}).get("skills") or {}).get("subset"))
        if isinstance(subset, list):
            out[str(m.get("id") or d.name).upper()] = sorted(subset)
    return out


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def build(root: Path = MATERIAL_ROOT) -> dict:
    org, cat, coll = load_materials(root)
    serve = parse_serviceability(SURVEY)
    boundary = parse_a_boundary(SURVEY)
    no_a_declared = parse_roles_without_a(SURVEY)
    card_l3 = load_card_l3()
    wiring = load_wiring()

    planes = {p["id"]: p for p in org["planes"]}
    domains = {d["id"]: d for d in org["domain_views"]}
    # ⚠️ join 陷阱 1：plane_id 只在 organization-graph.json。
    roles_org = {r["id"]: r for r in org["roles"]}
    roles_cat = {r["id"]: r for r in cat["roles"]}

    # --- 岗位层：平面从 org 取，域从 group 取，并断言两者独立 ---
    roles = []
    plane_domain_mismatch = []
    for rid in sorted(roles_cat):
        c, o = roles_cat[rid], roles_org.get(rid, {})
        plane_id = o.get("plane_id")
        # ⚠️ 字段名陷阱（2026-09-13 实测撞出，与漏洞 #11「判据只认一种字段名」同源）：
        #    `organization-graph.json` 里这个字段叫 **`domain_view_id`**，
        #    而产品侧 `taxonomy.json` 派生物里叫 **`domain_id`**。
        #    第一版按 `domain_id` 读 organization-graph，直接拿到 None → 当场失败。
        #    这次失败是**好事**（fail loud）；若当初写成 `.get("domain_id")` 再给个默认值，
        #    整张图的域维度会静默变成空。
        dom_id = o.get("domain_view_id") or o.get("domain_id")
        if plane_id not in planes or dom_id not in domains:
            raise SystemExit(f"✗ {rid} 的 plane/domain 在 organization-graph 里找不到：{plane_id}/{dom_id}")
        if c.get("group") != domains[dom_id]["name"]:
            plane_domain_mismatch.append({"role": rid, "group": c.get("group"),
                                          "domain": domains[dom_id]["name"]})
        roles.append({
            "id": rid, "alias": c.get("alias"), "title": c.get("title"),
            "plane_id": plane_id, "domain_id": dom_id,
            "group_field": c.get("group"),
            "l3": list(c.get("skills") or []),
            "flows": list(c.get("flows") or []),
            "scenarios_direct": list(c.get("scenarios") or []),
            "wired_skills": wiring.get(rid),
        })

    # --- L3 层：151 条，每条唯一属一岗（划分） ---
    l3 = []
    seen_l3: dict[str, str] = {}
    boundary_by_name = {b["name"]: b for b in boundary}
    for r in roles:
        for name in r["l3"]:
            if name in seen_l3:
                raise SystemExit(f"✗ L3 责任名重复：{name!r} 同属 {seen_l3[name]} 与 {r['id']}"
                                 f" —— 「151 条唯一属一岗」是划分级断言，不成立就不能出图")
            seen_l3[name] = r["id"]
            b = boundary_by_name.get(name)
            l3.append({
                "name": name, "role_id": r["id"],
                "plane_id": r["plane_id"], "domain_id": r["domain_id"],
                "serviceability": serve.get((r["id"], name)),
                # §F.5：边界条目 —— 计入 A，但「算法是否为主引擎」有争议（口径锚点，不重判）
                "boundary_a": b is not None,
                "boundary_rationale": (b or {}).get("rationale"),
                "downgrade_condition": (b or {}).get("downgrade_condition"),
            })

    # ⚠️ 解析值与机器算出值**两边都存**并断言相等 —— 这是「没有另造一份口径」的证据，
    #    也是唯一能发现「§F.5/F.6 写了一条、实际名单里没有」的判据。
    #    只存解析值 = 抄了一遍；只存算出值 = 丢了「这是人的口径判断」这个前提。
    a_names = {x["name"] for x in l3 if x["serviceability"] == "A"}
    roles_without_a_computed = sorted(r["id"] for r in roles
                                      if not any(x["name"] in a_names and x["role_id"] == r["id"]
                                                 for x in l3))
    for r in roles:
        r["a_count"] = sum(1 for x in l3 if x["role_id"] == r["id"] and x["serviceability"] == "A")

    # --- 场景层：直连 vs 传递闭包（join 陷阱 2） ---
    flow_scn = {f["id"]: list(f.get("scenario_ids") or []) for f in coll["flows"]}
    scn_ids = [s["id"] for s in coll["scenarios"]]
    scn_closure: dict[str, set[str]] = {}
    for r in roles:
        reach: set[str] = set()
        for fid in r["flows"]:
            reach |= set(flow_scn.get(fid, []))
        scn_closure[r["id"]] = reach

    scenarios = []
    for s in coll["scenarios"]:
        sid = s["id"]
        scenarios.append({
            "id": sid, "name": s.get("name") or s.get("title"),
            "eligible_flow_ids": list(s.get("eligible_flow_ids") or []),
            "roles_direct": sorted(r["id"] for r in roles if sid in r["scenarios_direct"]),
            "roles_closure": sorted(r["id"] for r in roles if sid in scn_closure[r["id"]]),
        })

    # --- 64 格：本图首次落库（唯一事实源） ---
    stage_kind: dict[str, tuple[str, str]] = {}
    for st in coll["protocol"]["stages"]:
        mp = st.get("model_participation")
        if mp not in MODEL_PARTICIPATION_KIND:
            raise SystemExit(
                f"✗ 未知的 model_participation={mp!r}（阶段 {st.get('name')}）。\n"
                f"  **故意不设默认值**：静默归类会把 D/R 格误当方法格（风险 N4），"
                f"而报告一切正常。请在此显式扩表。")
        # ⚠️ 同一族的第二个字段名陷阱：stage 对象的主键叫 `id`（不是 `stage_id`），
        #    而 `flow_stage_bindings` 里引用它时叫 `stage_id`。两处不能混用。
        stage_kind[st["id"]] = MODEL_PARTICIPATION_KIND[mp]

    stage_name = {st["id"]: st.get("name") for st in coll["protocol"]["stages"]}

    # --- 方案层：L3 方案域（S1 逐 FLOW 交付；**文档是源，本图是派生**）---
    # 读 `07-资源库/solutions/*.md` 的 frontmatter，再把该 FLOW 的 8 格 solution_refs 指向它。
    # 不许手改图里的 solutions —— 手改就繁殖出第二份事实源（R4 / 漏洞 #11 同型事故）。
    solutions = load_solutions()
    flow_ids = {f["id"] for f in coll["flows"]}
    for s in solutions:
        if s["flow_id"] not in flow_ids:
            raise SystemExit(f"✗ 方案域 {s['solution_id']} 的 flow_id={s['flow_id']!r} 不是合法 FLOW")

    cells = []
    for b in coll["flow_stage_bindings"]:
        sid = b["stage_id"]
        kind, why = stage_kind[sid]
        roles_in_flow = [r for r in roles if b["flow_id"] in r["flows"]]
        cells.append({
            "cell_id": f"{b['flow_id']}/{sid}",
            "flow_id": b["flow_id"], "stage_id": sid,
            "stage_name": stage_name.get(sid),
            "cell_kind": kind,
            "model_participation": next(st.get("model_participation")
                                        for st in coll["protocol"]["stages"] if st["id"] == sid),
            "cell_kind_basis": why,
            "algorithm_policy": ALGORITHM_POLICY[kind],
            "business_step": b.get("business_step"),
            "business_output_name": b.get("business_output_name"),
            "participating_role_ids": sorted(r["id"] for r in roles_in_flow),
            "solution_refs": sorted(s["solution_id"] for s in solutions
                                    if s["flow_id"] == b["flow_id"]),
            "card_refs": [],              # F5/S4 交付后回填
        })

    # --- 卡层：146 张，93 张有 L3（F5 欠 53） ---
    cards = []
    for p in sorted(VAULT.rglob("Skill-*.md")):
        if "_superseded" in p.parts:
            continue
        cards.append({
            "name": p.stem, "path": _rel(p),
            "domain_dir": p.parent.name,
            "l3": card_l3.get(p.stem),
        })

    counts = _counts(roles, l3, cells, scenarios, cards, solutions)
    graph = {
        "_meta": _meta(root, org, cat, coll),
        "mece": _mece(),
        "planes": [dict(planes[k], role_ids=sorted(r["id"] for r in roles if r["plane_id"] == k))
                   for k in sorted(planes)],
        "domains": [dict(domains[k], role_ids=sorted(r["id"] for r in roles if r["domain_id"] == k))
                    for k in sorted(domains)],
        "roles": roles,
        "l3": l3,
        "scenarios": scenarios,
        "cells": cells,
        "solutions": solutions,
        "cards": cards,
        "counts": counts,
        "a_scope": _a_scope(l3, boundary, no_a_declared, roles_without_a_computed),
        "join_traps": _join_traps(roles, org, plane_domain_mismatch, coll, counts),
        "open_items": _open_items(),
    }
    return graph


def _a_scope(l3, boundary, no_a_declared, no_a_computed) -> dict:
    """A 类口径的落点：阈值 + 边界条目 + 无 A 岗位。

    它回答的问题只有一个：**「A 类 73 条」里哪些是会随阈值动摇的。**
    下游（T-C 缺口账）据此把 A 当「优先匹配」而不是「已定名单」。
    """
    return {
        "threshold": "算法可作主引擎（若放宽到「算法可作重要辅助」，可服务面达 139/151）",
        "source": "_survey_org_model.md §F.1（标准）/ §F.5（边界条目）/ §F.6（无 A 岗位）",
        "a_count": sum(1 for x in l3 if x["serviceability"] == "A"),
        "boundary_a": [
            {"name": b["name"], "role_id": next(x["role_id"] for x in l3 if x["name"] == b["name"]),
             "rationale": b["rationale"], "downgrade_condition": b["downgrade_condition"]}
            for b in boundary],
        "contradictions": _a_scope_contradictions(l3, boundary),
        "roles_without_a_declared": no_a_declared,
        "roles_without_a_computed": no_a_computed,
        "note": "A 名单随阈值变，B 的 66 条不变 ⇒ 下游把 A 视为「优先匹配」、B 视为"
                "「需补外部证据后匹配」。**本条即 O-A 的落点**：阈值敏感性有名单可指，"
                "而不是一句形容词。",
    }


def _a_scope_contradictions(l3, boundary) -> list[dict]:
    """§F.5 说是 A、§F.3/F.4 却判 B 的条目 —— **登记，不重判**。

    ⚠️ 这条是 2026-09-13 F3 首次跑图时被**断言当场抓到**的（断言原文：
        「边界条目必须本身是 A 类」），实得 1 条：**产能调查**。
        §F.5 表头写「本次计入 A」，而降级条件栏又写「输入若全依赖供方声明，则降 B」，
        §F.3 逐条判定与 §F.4 的 B 清单（66 条）两处都把它列为 **B**。
        ⇒ 材料内部不一致，不是本脚本解析错。

    处置口径（与 JT-3 同型：**记进字段，不悄悄吞掉，也不替人重判**）：
        · A/B/C 一律以 **§F.3 逐条判定表**为准（它是唯一「岗位 + 责任名 + 理由」俱全的表，
          也是 F.2 唯一的 serviceability 事实源）；
        · §F.5 的边界标签保留，但**只能挂在真为 A 的条目上**；
        · 因此**产能调查不进靶区一**（它是 B），也就不会占检索预算 —— 这正是
          「判据的适用范围被默认成全体」的第四个变体：§F.5 写的是「A 类中的边界条目」，
          而它列的 10 条里实际只有 9 条是 A。
    """
    out = []
    for b in boundary:
        x = next((y for y in l3 if y["name"] == b["name"]), None)
        if x is not None and x["serviceability"] != "A":
            out.append({
                "name": b["name"], "role_id": x["role_id"],
                "declared_in": "§F.5（表头写「本次计入 A」）",
                "actual_in": f"§F.3 逐条判定 / §F.4 清单（均判 {x['serviceability']}）",
                "handling": f"以 §F.3 为准 ⇒ 按 {x['serviceability']} 处理，"
                            f"**不得**计入靶区一（它是按「A ∩ 无精选卡」定义的）",
            })
    return out


def _counts(roles, l3, cells, scenarios, cards, solutions) -> dict:
    kind = {}
    for c in cells:
        kind[c["cell_kind"]] = kind.get(c["cell_kind"], 0) + 1
    serve = {}
    for x in l3:
        k = x["serviceability"] or "UNPARSED"
        serve[k] = serve.get(k, 0) + 1
    return {
        "planes": 4, "domains": len({r["domain_id"] for r in roles}),
        "roles": len(roles), "l3": len(l3),
        "l3_unique_names": len({x["name"] for x in l3}),
        "l3_serviceability": serve,
        "l3_boundary_a": sum(1 for x in l3 if x.get("boundary_a")),
        "scenarios": len(scenarios),
        "cells": len(cells), "cells_by_kind": kind,
        # ⚠️ 两个口径**都**给：118 是直连声明，701 是传递闭包，差 5.9 倍（join 陷阱 2）
        "role_scenario_pairs_direct": sum(len(r["scenarios_direct"]) for r in roles),
        "role_scenario_pairs_closure": sum(
            len([s for s in scenarios if r["id"] in s["roles_closure"]]) for r in roles),
        "cards": len(cards),
        "cards_with_l3": sum(1 for c in cards if c["l3"]),
        "cards_unclassified": sorted(c["name"] for c in cards if not c["l3"]),
        "solutions": len(solutions),
        "solution_cells_covered": sum(1 for c in cells if c["solution_refs"]),
        "solution_flows_covered": sorted({s["flow_id"] for s in solutions}),
        "role_wiring_known": sum(1 for r in roles if r["wired_skills"] is not None),
        "roles_without_a": sum(1 for r in roles if r["a_count"] == 0),
    }


def _mece() -> dict:
    return {
        "_note": "每一层标明是**划分**还是**覆盖**。划分层可做加总，覆盖层**不可** —— "
                 "覆盖层的重数必须算出来（见 counts），不能声称。",
        "planes": {"property": "partition", "of": "50 岗位",
                   "check": "四平面 role_count 相加 = 50，并集去重 = 50，无重叠无遗漏"},
        "domains": {"property": "partition", "of": "50 岗位",
                    "check": "8 域 role_count 相加 = 50；域与面是**正交视角**，不是层级"},
        "l3": {"property": "partition", "of": "151 责任名",
               "check": "151 条全局唯一、每条唯一属一岗（重复即拒绝出图）",
               "primary_attribution": "每条 L3 的主归属 = 它唯一所属的岗位"},
        "cells": {"property": "partition", "of": "8 FLOW × 8 STG",
                  "check": "8×8=64，flow_stage_bindings 长度 64；每格唯一 (flow, stage)"},
        "scenarios": {"property": "cover", "of": "岗位",
                      "multiplicity": "一个 SCN 可被多岗位覆盖，且**两种口径并存**："
                                      "roles_direct（岗位直连声明，共 118 对）与 "
                                      "roles_closure（经 FLOW 传递闭包，共 701 对）。"
                                      "差 5.9 倍。下游必须选定一种并写明用的是哪一种。"},
        "solutions": {"property": "cover", "of": "64 格",
                      "multiplicity": "**一份方案域覆盖整条 FLOW 的 8 格**（不是一格一份）——"
                                      "8 份方案域 = 8 条 FLOW；S1 逐 FLOW 交付，"
                                      "已交付的 FLOW 其 8 格 solution_refs 全部指向它"},
        "cards": {"property": "cover", "of": "L3",
                  "multiplicity": "一卡可挂 1–3 个 L3；**146/146 张都有 L3**（F5 补齐）",
                  "check": "卡引用的 L3 必须都在 151 名单内（不在即拒绝出图）"},
    }


# `--check` 比对时**必须剔除**的易变字段。列在这里而不是散在代码里，
# 是为了让「什么算易变」成为一个可审阅的决定（剔多了会把真差异一起放过）。
VOLATILE_TOP_KEYS = ("generated",)


def _stable(g: dict) -> dict:
    """返回去掉易变字段的副本，供 --check 比对。"""
    out = json.loads(json.dumps(g, ensure_ascii=False))
    for k in VOLATILE_TOP_KEYS:
        out.get("_meta", {}).pop(k, None)
        out.pop(k, None)
    return out


def plane_domain_offdiagonal(roles) -> str:
    """面 × 域的**非对角**证据：从域推不出面，故 plane_id 必须 join。

    具体形态：单看「财务与合规」域，5 个岗位分别落在 业务运营 / 经营管理 / 独立控制 **三个**面；
    「经营与组织」域 5 个岗位落在 经营管理 / 独立控制 两个面。
    """
    from collections import defaultdict
    by: dict[str, set] = defaultdict(set)
    for r in roles:
        by[r["domain_id"]].add(r["plane_id"])
    split = {k: sorted(v) for k, v in by.items() if len(v) > 1}
    eg = "；".join(f"{k} 域跨 {len(v)} 个面 {v}" for k, v in sorted(split.items()))
    return f"{len(split)}/8 个域跨多个面 —— {eg}"


def _join_traps(roles, org, mismatch, coll, counts) -> list[dict]:
    # 陷阱 3：AGT-012 ↔ SCN-020 孤立绑定
    flow_scn = {f["id"]: list(f.get("scenario_ids") or []) for f in coll["flows"]}
    orphans = []
    for r in roles:
        reach: set[str] = set()
        for fid in r["flows"]:
            reach |= set(flow_scn.get(fid, []))
        for sid in r["scenarios_direct"]:
            if sid not in reach:
                orphans.append({"role_id": r["id"], "scenario_id": sid,
                                "role_flows": r["flows"]})
    return [
        {"id": "JT-1", "trap": "plane_id 只存在于 organization-graph.json",
         "why": "role-catalog.json 只带 group（=责任域）。只读 role-catalog 的实现会**静默丢掉平面维度**，"
                "而四平面正是本项目治理的第一视角。",
         "handling": "平面一律从 organization-graph 取；本图另留 group_field 以便对账。",
         # ⚠️ 证据要**证明**正交，不能用形容词。两条：
         #   ① `group` 与 organization-graph 的 `domain` 50/50 相等 ⇒ group 给的是**域**，不是面；
         #   ② 面×域是**非对角**的 ⇒ 从域推不出面。
         "evidence": f"① group vs organization-graph.domain 不一致 {len(mismatch)} 条（应为 0：两者同义）；"
                     f"② 面×域非对角：{plane_domain_offdiagonal(roles)}",
         "group_vs_domain_mismatch": len(mismatch),
         "plane_domain_offdiagonal": plane_domain_offdiagonal(roles)},
        # ⚠️ 数字**算出来**，不写死。第一版把调查报告的「118 / 701」抄进了文案，
        #    而本脚本实测是另一个值（见下）—— 抄进来的数字没有任何东西会发现它过期。
        {"id": "JT-2",
         "trap": f"scenarios 直连 {counts['role_scenario_pairs_direct']} 条 vs "
                 f"传递闭包 {counts['role_scenario_pairs_closure']} 条"
                 f"（差 {counts['role_scenario_pairs_closure']/max(counts['role_scenario_pairs_direct'],1):.2f} 倍）",
         "why": "两者**语义不同不是错误**：直连声明 vs「该岗位的流程覆盖到哪些场景」。混用会让"
                "某岗位的覆盖场景数虚高数倍。",
         "handling": "两个字段名分开、都给：roles[].scenarios_direct 与 scenarios[].roles_closure。",
         "evidence": f"counts.role_scenario_pairs_direct={counts['role_scenario_pairs_direct']} / "
                     f"_closure={counts['role_scenario_pairs_closure']}",
         "⚠️_订正": "调查报告 `_survey_org_model.md` §H 与方案 §3.1 记的是「118 vs 701（5.9 倍）」，"
                    "**该数字不成立**。2026-09-13 由本脚本实测订正：直连 127（role-catalog 逐岗相加，"
                    "去重后同值）、闭包 641（三种独立算法全得 641：按 flow.scenario_ids 并集 / "
                    "按 flow_scenario_bindings / 按 SCN.eligible_flow_ids ∩ role.flows）。"
                    "**结论方向不变**（SCN 仍是带重数的覆盖层），错的是量值。"},
        {"id": "JT-3", "trap": "AGT-012 ↔ SCN-020 的孤立绑定",
         "why": "SCN-020 的 eligible_flow_ids 只有 FLOW-06/08，而 AGT-012 的 flows=FLOW-02/07，"
                "两者完全不相交 —— 该岗位-场景对在 FLOW×SCN 兼容表里无路径可解释。",
         "handling": "记进本字段，不悄悄吞掉；按 SCN 归并论文卡时**不得**据此把该岗拉进数据/AI 运行场景。",
         "evidence": f"全量扫描 50 岗位 × 直连场景对，仅此 {len(orphans)} 条无路径",
         "orphans": orphans},
    ]


def _open_items() -> list[dict]:
    return [
        {"id": "O-SOL", "what": "方案层只交付了一部分 FLOW", "owner": "S1",
         "why": "`solutions` 与 `cells[].solution_refs` 由 `07-资源库/solutions/*.md` 派生。"
                "S1 按 FLOW 纵向切片交付：已交付的 FLOW 其 8 格 solution_refs 全部有值，"
                "未交付的仍为空数组 —— **空 = 还没做，不是缺陷**。"
                "契约层（139 份）与方案域分开计数：方案域 8 份，契约 139 份。"},
        {"id": "O-CARD", "what": "53 张精选卡无 L3 归属", "owner": "F5",
         "why": "146 张里 93 张能 join 到产品侧分类，另 53 张（含 PHASE3/4 新卡）待 F5"},
        {"id": "O-WIRE", "what": "岗位接线取自运行时 preset", "owner": "—",
         "why": "roles[].wired_skills 读 `~/.dsh/.agent-presets`；取不到时为 null，"
                "**「问不到」不等于「没接线」**"},
        {"id": "O-A", "what": "A/B/C 名单随阈值变化", "owner": "T-C",
         "why": "A 类阈值为「算法可作主引擎」；A 名单会随阈值变，B 的 66 条不变。"
                "下游把 A 视为「优先匹配」、B 视为「需补外部证据后匹配」",
         # ⚠️ 2026-09-13 F3：本条已从「一句形容词」变成**有名单可指**的字段。
         "handling": "落点在 `a_scope`：`boundary_a`（10 条边界条目，各带降级条件）"
                     "与 `roles_without_a_declared/_computed`（10 个岗位，两值必须相等）。"
                     "缺口账据此把边界条目降权排序，并把「A 是否为 73」变成断言。"},
    ]


def _meta(root: Path, org, cat, coll) -> dict:
    src = []
    for rel in ("docs/04-organization/organization-graph.json",
                "docs/05-agents/role-catalog.json",
                "docs/07-orchestration/collaboration-graph.json"):
        p = root / rel
        src.append({"path": str(p), "sha256": _sha(p)[:16], "bytes": p.stat().st_size})
    return {
        "what": "五层能力图谱：50 岗位 / 4 面 / 8 域 / 151 L3 / 20 SCN / 64 格（M/R/D）/ 方案 / 卡",
        "generated": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "paper2skills-research/scripts/build_capability_graph.py",
        "materials_root": str(root),
        "source_refs": src + [{"path": _rel(SURVEY),
                                "role": "A/B/C 判定（§F.3）+ A 类边界条目（§F.5）+ 无 A 岗位（§F.6）"},
                               {"path": _rel(CARD_CLASSIFICATION),
                                "role": "卡 ↔ L3 落点（F5）；其中 93 张逐字继承产品侧 "
                                        "classification.json，由 F5 的 J4 零漂移门禁守着"}],
        "l1_l2_l3_fact_source":
            "材料 organization-graph.json + role-catalog.json；产品侧已派生一份 "
            "dsh-paper2skills/data/taxonomy.json。本图**不重新定义** L1–L3，"
            "用 --check-taxonomy 与之逐项断言相等。",
        "cell_kind_basis":
            "由 collaboration-graph.json 的 protocol.stages[].model_participation 机器导出；"
            "映射表 MODEL_PARTICIPATION_KIND 对未知取值直接失败（无默认值）。",
        "version": 3,   # v3：方案层接上 L3 方案域源（S1 起，逐 FLOW 派生）
                        # v2：卡 ↔ L3 换到精选线事实源（F5），93 → 146 张
    }


# ---------------------------------------------------------------------------
# 守卫
# ---------------------------------------------------------------------------
def assert_partitions(g: dict) -> list[str]:
    """把 mece 里那些「声称」变成**断言**。不成立就返回失败原因（非空即判错）。"""
    errs = []
    c = g["counts"]
    if c["roles"] != 50:
        errs.append(f"岗位数 {c['roles']} ≠ 50")
    if sum(p["role_count"] for p in g["planes"]) != 50:
        errs.append("四平面 role_count 相加 ≠ 50")
    if len({rid for p in g["planes"] for rid in p["role_ids"]}) != 50:
        errs.append("四平面 role_ids 并集去重 ≠ 50（有重叠或遗漏）")
    if sum(d["role_count"] for d in g["domains"]) != 50:
        errs.append("8 域 role_count 相加 ≠ 50")
    if c["l3"] != 151 or c["l3_unique_names"] != 151:
        errs.append(f"L3 条数 {c['l3']} / 唯一 {c['l3_unique_names']} ≠ 151")
    if c["cells"] != 64:
        errs.append(f"格数 {c['cells']} ≠ 64")
    if c["cells_by_kind"] != {"M": 24, "R": 16, "D": 24}:
        errs.append(f"M/R/D 分型 {c['cells_by_kind']} ≠ 24/16/24")
    if c["scenarios"] != 20:
        errs.append(f"SCN 数 {c['scenarios']} ≠ 20")
    if c["l3_serviceability"].get("UNPARSED"):
        errs.append(f"{c['l3_serviceability']['UNPARSED']} 条 L3 没解析到 A/B/C")
    # ⚠️ A/B/C 总数是**下游靶区口径的锚点**（靶区一 = A ∩ 无精选卡）。锚点漂了必须当场红，
    #    否则缺口账会带着一个新分母继续算，而报告里看不出它换了口径。
    if {k: v for k, v in c["l3_serviceability"].items() if k in "ABC"} != {"A": 73, "B": 66, "C": 12}:
        errs.append(f"A/B/C 总数 {c['l3_serviceability']} ≠ 73/66/12"
                    f"（§F.3 变了 ⇒ 靶区口径必须重新推导，不能沿用旧工单）")
    # --- A 类口径（§F.5/F.6）：边界条目与无 A 岗位 ---
    if c["l3_boundary_a"] != 10:
        errs.append(f"§F.5 边界条目 {c['l3_boundary_a']} 条 ≠ 10（原文列了 10 条）")
    a_scope = g.get("a_scope") or {}
    # ⚠️ 这条断言 2026-09-13 首跑就抓到了材料内部矛盾（「产能调查」§F.5 记 A、§F.3/F.4 记 B）。
    #    **不要因为抓到就把它删掉或放宽成恒真** —— 改成：非 A 的边界条目必须**逐条登记在
    #    contradictions 里**。于是「已知的 1 条」被显式承认，而**新出现的**仍会打红。
    _contra = {c["name"] for c in a_scope.get("contradictions", [])}
    misrank = [x["name"] for x in a_scope.get("boundary_a", [])
               if next((y for y in g["l3"] if y["name"] == x["name"]), {}).get("serviceability") != "A"]
    unregistered = [n for n in misrank if n not in _contra]
    if unregistered:
        errs.append(f"边界条目不是 A 类且未登记：{unregistered}"
                    f"（若确属材料内部矛盾，请写进 a_scope.contradictions，不要删断言）")
    stale = [n for n in _contra if n not in misrank]
    if stale:
        errs.append(f"contradictions 里登记了 {stale}，但它们现在已是 A 类 —— 登记过期，请撤下")
    if c["roles_without_a"] != 10:
        errs.append(f"无 A 类岗位 {c['roles_without_a']} 个 ≠ 10")
    if a_scope.get("roles_without_a_declared") != a_scope.get("roles_without_a_computed"):
        errs.append(
            "§F.6 声明的无 A 岗位与按 §F.3 算出的不一致："
            f"仅声明 {sorted(set(a_scope.get('roles_without_a_declared', [])) - set(a_scope.get('roles_without_a_computed', [])))}；"
            f"仅算出 {sorted(set(a_scope.get('roles_without_a_computed', [])) - set(a_scope.get('roles_without_a_declared', [])))}")
    if len(a_scope.get("boundary_a", [])) != c["l3_boundary_a"]:
        errs.append(f"a_scope.boundary_a {len(a_scope.get('boundary_a', []))} 条 ≠ counts.l3_boundary_a "
                    f"{c['l3_boundary_a']}（表与计数对不上）")
    # 覆盖层：重数必须 > 基数（真覆盖，不是装出来的）
    if not c["role_scenario_pairs_closure"] > c["roles"]:
        errs.append("SCN 覆盖层没有重数 —— 它其实退化成了一一对应？")
    return errs


def check_taxonomy(g: dict, tax_path: Path) -> list[str]:
    """与产品侧 taxonomy.json 逐项比对 —— 证明本图**没有另造一份 L1–L3**。"""
    if not tax_path.is_file():
        return [f"产品侧 taxonomy 不存在：{tax_path}（无法证明「不另造事实源」）"]
    t = json.loads(tax_path.read_text(encoding="utf-8"))
    errs = []
    if len(t["planes"]) != len(g["planes"]):
        errs.append(f"面数不一致：taxonomy {len(t['planes'])} vs 本图 {len(g['planes'])}")
    if len(t["domains"]) != len(g["domains"]):
        errs.append(f"域数不一致：{len(t['domains'])} vs {len(g['domains'])}")
    a = {x["name"] for x in t["l3"]}
    b = {x["name"] for x in g["l3"]}
    if a != b:
        errs.append(f"L3 名单不一致：仅 taxonomy 有 {sorted(a - b)[:5]}；仅本图有 {sorted(b - a)[:5]}")
    # 逐条核对 L3 的岗位归属（这是最容易悄悄漂的一列）
    ta = {x["name"]: x["role_id"] for x in t["l3"]}
    tb = {x["name"]: x["role_id"] for x in g["l3"]}
    diff = [n for n in set(ta) & set(tb) if ta[n] != tb[n]]
    if diff:
        errs.append(f"L3 的岗位归属不一致 {len(diff)} 条：{diff[:5]}")
    tp = {x["id"]: x["role_count"] for x in t["planes"]}
    gp = {x["id"]: x["role_count"] for x in g["planes"]}
    if tp != gp:
        errs.append(f"面的 role_count 不一致：{tp} vs {gp}")
    return errs


# ---------------------------------------------------------------------------
# 查询：任取一格能机械回答「有无方案 / 有无卡 / 本岗是否接线」（本卡验收项）
# ---------------------------------------------------------------------------
def query_cell(g: dict, cell_id: str) -> dict:
    c = next((x for x in g["cells"] if x["cell_id"] == cell_id), None)
    if c is None:
        raise SystemExit(f"✗ 无此格：{cell_id}（格式 FLOW-NN/STG-NN；共 {len(g['cells'])} 格）")
    # 「有无卡」：该格参与岗位的 L3 上挂了哪些卡
    l3_of_roles = {}
    for x in g["l3"]:
        l3_of_roles.setdefault(x["role_id"], []).append(x["name"])
    reach_l3 = {n for rid in c["participating_role_ids"] for n in l3_of_roles.get(rid, [])}
    cards = [x["name"] for x in g["cards"] if x["l3"] and set(x["l3"]) & reach_l3]
    wired = [(r["id"], bool(r["wired_skills"]), len(r["wired_skills"] or []))
             for r in g["roles"] if r["id"] in c["participating_role_ids"]]
    return {
        "cell_id": c["cell_id"], "cell_kind": c["cell_kind"],
        "stage_name": c["stage_name"], "business_step": c["business_step"],
        "business_output_name": c["business_output_name"],
        "algorithm_policy": c["algorithm_policy"],
        "有无方案": {"count": len(c["solution_refs"]), "refs": c["solution_refs"],
                     "note": "0 = S1 未交付（不是缺陷）"},
        "有无卡": {"count": len(cards), "sample": cards[:8],
                   "note": "按「参与岗位的 L3 → 卡」反查；card_refs 待 F5/S4 显式回填"},
        "本岗是否接线": {"note": "wired=null 表示读不到运行时 preset（**问不到 ≠ 没接线**）",
                         "roles": [{"role_id": i, "wired": w, "wired_skill_count": n}
                                   for i, w, n in wired]},
        "reachable_l3_count": len(reach_l3),
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _selftest() -> int:
    ok = True
    # F5 的卡端事实源是**已入库的产物**：自检直接用它（更诚实），但缺了要炸得明白，
    # 否则后面每个变异块都会因为同一个缺文件而报 ❌，看起来像判据坏了。
    if not CARD_CLASSIFICATION.is_file():
        print(f"✗ 自检需要 {_rel(CARD_CLASSIFICATION)}（F5 产物）才能跑："
              f"先执行 build_card_classification.py --write")
        return 1
    # ⚠️ 在这里**保存一次**，供后面每个变异块共用。
    #    第一版只在第二个变异块里定义它，于是先跑的 M/R/D 变异块在 finally 里
    #    引用了尚未赋值的名字 → UnboundLocalError（**自检自己崩了**）。
    saved_builder = globals()["load_materials"]
    print("--- 判据 ②：cell_kind 由 model_participation 机器导出 ---")
    coll = json.loads((MATERIAL_ROOT / "docs/07-orchestration/collaboration-graph.json")
                      .read_text(encoding="utf-8"))
    mps = [st.get("model_participation") for st in coll["protocol"]["stages"]]
    unmapped = [m for m in mps if m not in MODEL_PARTICIPATION_KIND]
    print(("✅" if not unmapped else "❌") +
          f" 材料的 {len(mps)} 个 model_participation 取值全部在映射表内（未映射：{unmapped}）")
    ok &= not unmapped

    # ⚠️ 变异：把某个阶段的语义改成未知值，构建**必须**失败而不是静默归类
    saved = dict(MODEL_PARTICIPATION_KIND)
    try:
        MODEL_PARTICIPATION_KIND.pop("none_required", None)
        crashed = False
        try:
            build()
        except SystemExit as e:
            crashed = "未知的 model_participation" in str(e)
        print(("✅" if crashed else "❌") +
              " 变异：出现未映射取值时**直接失败**（无默认值 ⇒ D/R 格不会被静默当方法格）")
        ok &= crashed
    finally:
        MODEL_PARTICIPATION_KIND.clear()
        MODEL_PARTICIPATION_KIND.update(saved)

    print("\n--- 判据 ①：划分层必须真的划分 ---")
    g = build()
    errs = assert_partitions(g)
    print(("✅" if not errs else "❌") + f" 划分断言全过（{len(errs)} 条失败）")
    for e in errs:
        print("    ✗", e)
    ok &= not errs

    # ⚠️⚠️ **对每一条划分断言各喂一份被篡改的图**，逐条证明它会红。
    #    只测「真实数据下断言全过」是没用的：真数据本来就是对的，
    #    把任意一条断言删掉，自检照样全绿 —— 那就是**摆设**。
    #    （2026-09-13 变异测试先抓出 M3/M5，再抓出 M6，都是同一个病：
    #      **断言恒真 = 没断言**。故这里改成穷举式，新增断言必须同时加篡改样本。）
    def _tamper(mut) -> str | None:
        g2 = json.loads(json.dumps(g, ensure_ascii=False))
        mut(g2)
        e2 = assert_partitions(g2)
        return e2[0] if e2 else None

    tampers = [
        ("四平面 role_ids 并集去重",
         lambda x: x["planes"][0]["role_ids"].pop()),
        ("四平面 role_count 相加",
         lambda x: x["planes"][0].__setitem__("role_count", 4)),
        ("8 域 role_count 相加",
         lambda x: x["domains"][0].__setitem__("role_count", 4)),
        ("L3 条数",
         lambda x: x["counts"].__setitem__("l3", 150)),
        ("L3 唯一名数",
         lambda x: x["counts"].__setitem__("l3_unique_names", 150)),
        ("格数 64",
         lambda x: x["counts"].__setitem__("cells", 63)),
        ("M/R/D 分型",
         lambda x: x["counts"]["cells_by_kind"].__setitem__("M", 23)),
        ("SCN 数 20",
         lambda x: x["counts"].__setitem__("scenarios", 19)),
        ("A/B/C 未解析",
         lambda x: x["counts"]["l3_serviceability"].__setitem__("UNPARSED", 1)),
        ("覆盖层必须有重数",
         lambda x: x["counts"].__setitem__("role_scenario_pairs_closure", 1)),
        ("§F.5 边界条目数 10",
         lambda x: x["counts"].__setitem__("l3_boundary_a", 9)),
        ("边界条目本身必须是 A 类",
         lambda x: next(y for y in x["l3"] if y["name"] == "术语治理").__setitem__(
             "serviceability", "B")),
        ("contradictions 登记不得过期",
         lambda x: next(y for y in x["l3"] if y["name"] == "产能调查").__setitem__(
             "serviceability", "A")),
        ("A/B/C 总数锚点 73/66/12",
         lambda x: x["counts"]["l3_serviceability"].__setitem__("A", 72)),
        ("无 A 岗位数 10",
         lambda x: x["counts"].__setitem__("roles_without_a", 9)),
        ("§F.6 声明 vs 算出必须相等",
         lambda x: x["a_scope"].__setitem__("roles_without_a_declared",
                                            x["a_scope"]["roles_without_a_declared"] + ["AGT-001"])),
        ("a_scope 表与计数必须对得上",
         lambda x: x["a_scope"]["boundary_a"].pop()),
    ]
    miss = []
    for label, mut in tampers:
        if _tamper(mut) is None:
            miss.append(label)
    print(("✅" if not miss else "❌") +
          f" 对 {len(tampers)} 条划分断言各喂一份篡改图，全部被抓（漏 {miss}）")
    ok &= not miss

    # --- 判据 ③：方案层是**派生**的，不是手写进图的（S1 起） ---
    # 断言与变异各来一份：只断言「已交付的 FLOW 有 refs」是**恒真**的（除非有人删文件），
    # 所以必须再加一条「把源目录换空 ⇒ refs 必须清空」的注入式变异，
    # 否则这条判据证明不了「图跟着文档走」。
    print("\n--- 判据 ③：方案层由 solutions/*.md 派生 ---")
    sol = g["solutions"]
    bad_refs = []
    for s in sol:
        cells_of_flow = [c for c in g["cells"] if c["flow_id"] == s["flow_id"]]
        if len(cells_of_flow) != 8 or any(s["solution_id"] not in c["solution_refs"]
                                          for c in cells_of_flow):
            bad_refs.append(s["solution_id"])
    print(("✅" if not bad_refs else "❌") +
          f" {len(sol)} 份方案域各覆盖其 FLOW 的 8 格（未覆盖：{bad_refs}）")
    ok &= not bad_refs
    stray = [c["cell_id"] for c in g["cells"] if c["solution_refs"]
             and not any(s["solution_id"] in c["solution_refs"] for s in sol)]
    print(("✅" if not stray else "❌") +
          f" 反向：没有指向不存在方案的格（越界格：{stray}）")
    ok &= not stray

    saved_sol_dir = globals()["SOLUTIONS_DIR"]
    try:
        # 变异 A：源目录为空 ⇒ 全部 refs 清空，counts.solutions = 0（**图跟着文档走**）
        with tempfile.TemporaryDirectory() as td:
            globals()["SOLUTIONS_DIR"] = Path(td)
            g_empty = build()
        empty_ok = (g_empty["counts"]["solutions"] == 0
                    and all(not c["solution_refs"] for c in g_empty["cells"]))
        print(("✅" if empty_ok else "❌") +
              " 变异 A：把方案源目录换空 → 64 格 refs 全清空（证明是派生，不是缓存）")
        ok &= empty_ok

        # 变异 B：一份有 frontmatter 但没有 flow_id 的文档 ⇒ 必须失败，而不是静默跳过
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "FLOW-09-野方案.md").write_text(
                "---\ntitle: 野方案\n---\n\n没写 flow_id。\n", encoding="utf-8")
            globals()["SOLUTIONS_DIR"] = Path(td)
            crashed = False
            try:
                build()
            except SystemExit as e:
                crashed = "没有 `flow_id`" in str(e)
        print(("✅" if crashed else "❌") +
              " 变异 B：方案域缺 flow_id → 直接失败（静默跳过会让它成为谁也查不到的孤儿）")
        ok &= crashed
    finally:
        globals()["SOLUTIONS_DIR"] = saved_sol_dir

    # ⚠️ 变异：把 STG-04 的语义从 M 改成 D ⇒ 分型必须变成 16/16/32 并被抓到。
    #    没有这一条，「M/R/D 分型 ≠ 24/16/24」这句断言就是个**摆设** ——
    #    把整行删掉自检照样全绿。（2026-09-13 由变异测试实测抓出；
    #    与 CLAUDE.md 记的 C8「用例 18 是摆设」同型：**断言恒真 = 没断言**。）
    org_m, cat_m, coll_m = load_materials(MATERIAL_ROOT)
    globals()["load_materials"] = lambda root: (org_m, cat_m, coll_m)
    try:
        st04 = next(st for st in coll_m["protocol"]["stages"] if st["id"] == "STG-04")
        saved_mp = st04["model_participation"]
        st04["model_participation"] = "none_required"        # M → D
        g_mut = build()
        got = g_mut["counts"]["cells_by_kind"]
        caught = got != {"M": 24, "R": 16, "D": 24} and assert_partitions(g_mut) != []
        print(("✅" if caught else "❌") +
              f" 变异：STG-04 由 M 改 D → 分型变为 {got} 且划分断言报错（该断言不是摆设）")
        ok &= caught
        st04["model_participation"] = saved_mp
    finally:
        globals()["load_materials"] = saved_builder

    # ⚠️ 变异：把一个 L3 复制到第二个岗位（破坏划分），断言必须变红
    org, cat, coll2 = load_materials(MATERIAL_ROOT)
    victim = next(r for r in cat["roles"] if r["id"] == "AGT-002")
    victim["skills"] = [cat["roles"][0]["skills"][0]] + victim["skills"][1:]
    globals()["load_materials"] = lambda root: (org, cat, coll2)
    try:
        dup_caught = False
        try:
            build()
        except SystemExit as e:
            dup_caught = "L3 责任名重复" in str(e)
        print(("✅" if dup_caught else "❌") + " 变异：L3 重复归属 → 拒绝出图（划分断言是真的）")
        ok &= dup_caught
    finally:
        globals()["load_materials"] = saved_builder
        victim["skills"] = victim["skills"][1:]

    print("\n--- 验收：任取一格能机械回答三个问题 ---")
    q = query_cell(g, "FLOW-01/STG-04")
    three = all(k in q for k in ("有无方案", "有无卡", "本岗是否接线"))
    print(("✅" if three else "❌") +
          f" FLOW-01/STG-04 → kind={q['cell_kind']} · 方案 {q['有无方案']['count']} · "
          f"卡 {q['有无卡']['count']} · 参与岗位 {len(q['本岗是否接线']['roles'])}")
    ok &= three

    # ⚠️ 改一个节点，下游计数必须变（验收项原话）
    print("\n--- 验收：改一个节点，下游计数必须变 ---")
    base = g["counts"]
    org2, cat2, coll3 = load_materials(MATERIAL_ROOT)
    coll3["roles"] = None  # 占位，避免误用
    r0 = next(r for r in cat2["roles"] if r["id"] == "AGT-021")
    r0["flows"] = [f for f in r0["flows"] if f != "FLOW-01"]     # 把 021 从 FLOW-01 摘掉
    globals()["load_materials"] = lambda root: (org2, cat2, coll3)
    try:
        g2 = build()
        # FLOW-01 的四格参与岗位必须少一个（021 曾参与 FLOW-01）
        before = {c["cell_id"]: len(c["participating_role_ids"]) for c in g["cells"]
                  if c["flow_id"] == "FLOW-01"}
        after = {c["cell_id"]: len(c["participating_role_ids"]) for c in g2["cells"]
                 if c["flow_id"] == "FLOW-01"}
        changed = before != after
        print(("✅" if changed else "❌") +
              f" 摘掉 AGT-021 的 FLOW-01 → 该 FLOW 各格参与岗位数变化 {sorted(before.items())[:2]}"
              f" → {sorted(after.items())[:2]}")
        ok &= changed
    finally:
        globals()["load_materials"] = saved_builder
        r0["flows"] = r0["flows"] + ["FLOW-01"]

    print("\n--- 与产品侧 taxonomy 的一致性（证明没有另造事实源）---")
    tax_errs = check_taxonomy(g, TAXONOMY)
    print(("✅" if not tax_errs else "❌") + f" 逐项一致（{len(tax_errs)} 条不一致）")
    for e in tax_errs:
        print("    ✗", e)
    ok &= not tax_errs

    # ⚠️ 反向：把一份**被篡改的** taxonomy 喂进去，比对必须报错。
    #    没有这一条，上面那句「逐项一致」就是**摆设** —— 真 taxonomy 本来就是一致的，
    #    把整段比对删掉自检照样全绿（2026-09-13 由变异测试 M5 实测抓出）。
    # ⚠️ 这里**不再** `import tempfile`：函数内的局部 import 会让 `tempfile` 在整个函数里
    #    变成局部名，于是本函数**前面**任何一处 `tempfile.…` 都抛 UnboundLocalError
    #    —— 又一次「自检自己崩了」。模块级已经 import 过。
    with tempfile.TemporaryDirectory() as td:
        t = json.loads(TAXONOMY.read_text(encoding="utf-8"))
        victim_l3 = t["l3"][0]
        other = next(x for x in t["l3"] if x["role_id"] != victim_l3["role_id"])
        victim_l3["role_id"] = other["role_id"]          # 篡改归属
        tp = Path(td) / "taxonomy.json"
        tp.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
        caught = bool(check_taxonomy(g, tp))
        print(("✅" if caught else "❌") +
              f" 变异：篡改 taxonomy 里一条 L3 的岗位归属 → 比对报错（该比对不是摆设）")
        ok &= caught

    print("\n--- `_rel()`：仓库外路径不得抛 ValueError（门禁缺陷 #13 同族第三次）---")
    probes = [
        (REPO / "paper2skills-vault" / "07-资源库" / "capability-graph.json",
         "paper2skills-vault/07-资源库/capability-graph.json"),
        (Path("/tmp/p2s-f5/somewhere.json"), "/tmp/p2s-f5/somewhere.json"),
        (Path("/Users/lute/project/paper_to_skills-OTHER/x.json"),
         "/Users/lute/project/paper_to_skills-OTHER/x.json"),
    ]
    for path, want in probes:
        try:
            got = _rel(path)
        except ValueError as e:                       # 旧写法在这里抛
            got = f"ValueError: {e}"
        good = got == want
        print(("✅" if good else "❌") + f" {path} → {got}")
        ok &= good

    print("\n--- 三个 join 陷阱都在图里 ---")
    ids = {t["id"] for t in g["join_traps"]}
    jt = ids == {"JT-1", "JT-2", "JT-3"}
    print(("✅" if jt else "❌") + f" JT-1/2/3 全在（实得 {sorted(ids)}）")
    ok &= jt

    print()
    print("SELFTEST " + ("PASS —— 两个必库判据都由机器导出、划分断言可被破坏、三个验收问题可机械回答"
                        if ok else "FAIL —— 判据静默失效，勿信其产物"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="五层能力图谱生成器（PHASE6 F2 / T-A）")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--materials-root", type=Path, default=MATERIAL_ROOT)
    ap.add_argument("--check", action="store_true", help="重新生成并与已入库的图逐字节比对")
    ap.add_argument("--check-taxonomy", action="store_true", help="与产品侧 taxonomy.json 逐项比对")
    ap.add_argument("--cell", help="查询一格，如 FLOW-01/STG-04")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    for rel in ("docs/04-organization/organization-graph.json",
                "docs/05-agents/role-catalog.json",
                "docs/07-orchestration/collaboration-graph.json"):
        if not (args.materials_root / rel).is_file():
            print(f"✗ 材料缺失：{args.materials_root / rel}\n"
                  f"  用 --materials-root 指定，或设环境变量 AI_ORG_MATERIAL_ROOT。\n"
                  f"  **「读不到材料」不是「图谱没问题」。**")
            return 2

    g = build(args.materials_root)
    errs = assert_partitions(g)
    if errs:
        print("✗ 划分断言不成立，拒绝写出图谱：")
        for e in errs:
            print("   ", e)
        return 1

    if args.cell:
        print(json.dumps(query_cell(g, args.cell), ensure_ascii=False, indent=1))
        return 0

    if args.check_taxonomy:
        te = check_taxonomy(g, TAXONOMY)
        if te:
            print("✗ 与产品侧 taxonomy 不一致：")
            for e in te:
                print("   ", e)
            return 1
        print(f"✓ 与产品侧 taxonomy 逐项一致（{TAXONOMY}）")
        return 0

    text = json.dumps(g, ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not args.out.is_file():
            print(f"✗ {args.out} 不存在：先跑一次生成")
            return 1
        # ⚠️ 比对必须**剔掉易变字段**，否则永远不等 —— 第一版就是逐字节比，
        #    于是 --check 每次必红（`_meta.generated` 是当下时间）。
        #    那会让这道门禁被当成「坏掉的工具」而绕过，比没有门禁更糟。
        old = _stable(json.loads(args.out.read_text(encoding="utf-8")))
        new = _stable(g)
        if old != new:
            diff = sorted(set(old) | set(new))
            where = [k for k in diff if old.get(k) != new.get(k)]
            print(f"✗ {args.out} 与材料/卡片现状不一致（顶层差异：{where[:6]}）：重跑生成器")
            return 1
        print(f"✓ {args.out} 与材料一致（已剔除易变字段 {VOLATILE_TOP_KEYS}）")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    c = g["counts"]
    if not args.quiet:
        print(f"✓ 写入 {args.out}（{args.out.stat().st_size/1024:.0f} KB）")
        print(f"  面 {c['planes']} / 域 {c['domains']} / 岗位 {c['roles']} / L3 {c['l3']}"
              f" / SCN {c['scenarios']} / 格 {c['cells']} {c['cells_by_kind']}")
        print(f"  A/B/C = {c['l3_serviceability'].get('A')}/{c['l3_serviceability'].get('B')}"
              f"/{c['l3_serviceability'].get('C')}"
              f"（其中边界条目 {c['l3_boundary_a']} 条：会随阈值动摇；"
              f"无 A 类岗位 {c['roles_without_a']} 个）")
        print(f"  卡 {c['cards']} 张，其中 {c['cards_with_l3']} 张有 L3 归属"
              + (f"（欠 {c['cards'] - c['cards_with_l3']} 张）"
                 if c["cards_with_l3"] != c["cards"] else "（F5 已覆盖全部）"))
        print(f"  岗位-场景对：直连 {c['role_scenario_pairs_direct']} / "
              f"闭包 {c['role_scenario_pairs_closure']}（两口径都给，勿混用）")
        print(f"  岗位接线已知 {c['role_wiring_known']}/{c['roles']}"
              f"（读不到运行时时为 null —— 「问不到」≠「没接线」）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
