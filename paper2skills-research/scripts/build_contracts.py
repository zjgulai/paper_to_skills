#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""契约层（L4）生成器 —— 139 份供给契约的骨架与清单。

设计要点（与 PHASE6 F2 的 capability-graph.json 同一条纪律）：

1. **frontmatter 全部由图谱导出，人不许手改。**
   责任名/岗位/域/平面/服务性/flows/method_cells/rule_cells 都只有一处事实源；
   手写一份就会繁殖出第二份分类表（R4 与漏洞 #11 的同型事故）。
   `--check` 会重算一遍 frontmatter 并与文件逐字段比对，漂移即报错。

2. **模板类型由服务性唯一导出**：A→A、B→B、**C→无契约**。
   139 = 151 − 12（C 类是组织程序/人际协作/法定程序，给它们建契约只会制造形式主义）。

3. **cell_kind 决定契约能挂什么**：方法只许挂在 M 格（STG-04/05/08），
   R 格（STG-02/06）与 D 格（STG-01/03/07）只放判据（风险 N4）。

用法::

    python3 build_contracts.py --list                       # 139 行清单（stdout + JSON）
    python3 build_contracts.py --skeleton 依赖协调 容量管理      # 写骨架文件
    python3 build_contracts.py --all-skeletons              # 写全部 139 份骨架（S1 用）
    python3 build_contracts.py --check                      # frontmatter 漂移检测
    python3 build_contracts.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DEFAULT = REPO_ROOT / "paper2skills-vault/07-资源库/capability-graph.json"
CONTRACTS_DIR = REPO_ROOT / "paper2skills-vault/07-资源库/contracts"

# 由 capability-graph.json 实测导出；写死会腐烂，故在此只登记「应等于图谱」的期望值，
# 真值一律从图上读，读不到就炸（不给默认值 —— 默认值是静默错配的来源）。
M_STAGES = ("STG-04", "STG-05", "STG-08")
RD_STAGES = ("STG-01", "STG-02", "STG-03", "STG-06", "STG-07")

TODO_MARK = "<!-- 待撰写 -->"
UNWRITTEN_STATUS = "待卡"  # 骨架默认；撰写人按实际覆盖

A_HEADERS = (
    "## 1 方法来源（卡供方法）",
    "## 2 数据要求（五维）",
    "## 3 标定规则",
    "## 4 重标定触发条件",
    "## 5 适用 FLOW",
    "## 6 冻结与不许自动放行的情形",
)
B_HEADERS = (
    "## 1 算法可达部分",
    "## 2 不可达部分",
    "## 3 必须的外部证据与责任岗位",
    "## 4 该格何时必须冻结",
    "## 5 数据要求（五维）",
    "## 6 适用 FLOW",
)


def safe_rel(path: Path) -> str:
    """仓库外路径不得让脚本崩 —— 崩的若是**打印**这一步，人只会看到「写骨架失败」而实际已写。

    与门禁缺陷 #13（gate_check 的 relative_to 对仓库外路径抛 ValueError）同型。
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fail(msg: str) -> "None":
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_graph(path: Path) -> dict:
    if not path.exists():
        fail(f"图谱不存在：{path}")
    g = json.loads(path.read_text(encoding="utf-8"))
    for key in ("l3", "roles", "cells", "counts"):
        if key not in g:
            fail(f"图谱缺字段 {key} —— 判据只认一种字段名，拿不到就炸，不要给默认值")
    return g


class GraphIndex:
    """把图谱压成契约需要的索引。任何一次取不到都当场失败。"""

    def __init__(self, graph: dict):
        self.graph = graph
        self.roles = {r["id"]: r for r in graph["roles"]}
        self.l3 = {}
        for item in graph["l3"]:
            # 字段名陷阱：l3 记录用 role_id / domain_id / plane_id；缺一即炸。
            for f in ("name", "role_id", "domain_id", "plane_id", "serviceability"):
                if f not in item:
                    fail(f"l3 记录缺字段 {f}：{item}")
            if item["name"] in self.l3:
                fail(f"责任名重复：{item['name']} —— 151 条必须全局唯一")
            self.l3[item["name"]] = item
        self.cells = {c["cell_id"]: c for c in graph["cells"]}
        self.cell_kind = {c["cell_id"]: c["cell_kind"] for c in graph["cells"]}
        self.stage_of = {c["cell_id"]: c["stage_id"] for c in graph["cells"]}
        self.flow_of = {c["cell_id"]: c["flow_id"] for c in graph["cells"]}
        self.graph_digest = hashlib.sha256(
            json.dumps(graph, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    # ---- 派生物 -------------------------------------------------------
    def flows_of_role(self, role_id: str) -> list:
        r = self.roles.get(role_id) or fail(f"岗位不存在：{role_id}")
        if "flows" not in r:
            fail(f"岗位 {role_id} 缺 flows 字段")
        return sorted(r["flows"])

    def cells_of_role(self, role_id: str, kinds: tuple) -> list:
        """该岗位在其每个 FLOW 的哪些格上、且格类型命中 kinds。"""
        out = []
        for flow in self.flows_of_role(role_id):
            for stage in M_STAGES + RD_STAGES:
                cid = f"{flow}/{stage}"
                if cid not in self.cells:
                    fail(f"图谱缺格：{cid}（FLOW×STG 必须 64 格齐）")
                if self.cell_kind[cid] in kinds:
                    out.append(cid)
        # 先按 FLOW 再按 STG 号排序，保证同一岗位每次生成的顺序一致
        return sorted(out, key=lambda x: (x.split("/")[0], x.split("/")[1]))

    def template_of(self, serviceability: str) -> str:
        if serviceability == "A":
            return "A"
        if serviceability == "B":
            return "B"
        if serviceability == "C":
            fail("C 类不建契约（139 = 151 − 12）—— 调用方应先过滤")
        fail(f"未知服务性取值：{serviceability!r}（只认 A/B/C）")

    def ordered_responsibilities(self) -> list:
        """稳定的编号顺序：按岗位号、再按岗位内 l3 出现顺序（与材料一致）。"""
        order = []
        for rid in sorted(self.roles):
            role = self.roles[rid]
            for name in role.get("l3", []):
                order.append(name)
        missing = set(self.l3) - set(order)
        extra = [n for n in order if n not in self.l3]
        if missing or extra:
            fail(f"l3 与 roles[].l3 不一致：图上有而岗位上没有={sorted(missing)[:5]}，"
                 f"岗位上有而图上没有={extra[:5]}")
        return order

    def assignments(self) -> dict:
        """责任名 → 契约元数据（含 contract_id）。"""
        out = {}
        counters = {"A": 0, "B": 0}
        for name in self.ordered_responsibilities():
            item = self.l3[name]
            sv = item["serviceability"]
            if sv == "C":
                continue
            tpl = self.template_of(sv)
            counters[tpl] += 1
            rid = item["role_id"]
            out[name] = {
                "contract_id": f"CTR-{tpl}-{counters[tpl]:03d}",
                "template": tpl,
                "responsibility": name,
                "role_id": rid,
                "role_title": self.roles[rid]["title"],
                "domain_id": item["domain_id"],
                "plane_id": item["plane_id"],
                "serviceability": sv,
                "flows": self.flows_of_role(rid),
                "method_cells": self.cells_of_role(rid, ("M",)),
                "rule_cells": self.cells_of_role(rid, ("R", "D")),
            }
        return out


# ---------------------------------------------------------------------------
# frontmatter
# ---------------------------------------------------------------------------
def fm_block(meta: dict, cards=(), status=UNWRITTEN_STATUS, blocked_by=None) -> str:
    def yaml_list(xs):
        return "[" + ", ".join(xs) + "]"

    lines = [
        "---",
        f"contract_id: {meta['contract_id']}",
        f"template: {meta['template']}",
        f"responsibility: {meta['responsibility']}",
        f"role_id: {meta['role_id']}",
        f"role_title: {meta['role_title']}",
        f"domain_id: {meta['domain_id']}",
        f"plane_id: {meta['plane_id']}",
        f"serviceability: {meta['serviceability']}",
        f"flows: {yaml_list(meta['flows'])}",
        f"method_cells: {yaml_list(meta['method_cells'])}",
        f"rule_cells: {yaml_list(meta['rule_cells'])}",
        f"cards: {yaml_list(list(cards))}",
        f"status: {status}",
        f"blocked_by: {blocked_by if blocked_by else 'null'}",
        "---",
    ]
    return "\n".join(lines)


def skeleton_body(template: str) -> str:
    headers = A_HEADERS if template == "A" else B_HEADERS
    parts = []
    for h in headers:
        parts.append(f"{h}\n\n{TODO_MARK}\n")
    return "\n".join(parts)


def filename_for(meta: dict) -> str:
    tpl = meta["template"]
    return f"{tpl}/{meta['contract_id']}-{meta['responsibility']}.md"


def write_skeleton(idx: GraphIndex, names, outdir: Path, force=False) -> list:
    assignments = idx.assignments()
    written = []
    for name in names:
        if name not in assignments:
            if name in idx.l3:
                fail(f"{name} 是 C 类责任，不建契约（139 = 151 − 12）")
            fail(f"责任名不在图谱 151 内：{name!r}")
        meta = assignments[name]
        path = outdir / filename_for(meta)
        if path.exists() and not force:
            print(f"· 跳过已存在：{safe_rel(path)}（--force 可覆盖）")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            fm_block(meta) + "\n\n# " + meta["responsibility"] + " · 供给契约\n\n" + skeleton_body(meta["template"]),
            encoding="utf-8",
        )
        written.append(path)
        print(f"✅ {safe_rel(path)}")
    return written


# ---------------------------------------------------------------------------
# --list / --check / --selftest
# ---------------------------------------------------------------------------
def emit_list(idx: GraphIndex, json_out=None):
    assignments = idx.assignments()
    rows = sorted(assignments.values(), key=lambda m: m["contract_id"])
    n_a = sum(1 for m in rows if m["template"] == "A")
    n_b = sum(1 for m in rows if m["template"] == "B")
    print(f"契约清单：{len(rows)} 份（A {n_a} / B {n_b}）  ← 139 = 151 − 12(C 类)")
    print(f"{'contract_id':12} {'tpl':3} {'责任名':16} {'岗位':8} {'域':7} {'M格':4} {'R/D格':5} flows")
    for m in rows:
        print(f"{m['contract_id']:12} {m['template']:3} {m['responsibility']:16} "
              f"{m['role_id']:8} {m['domain_id']:7} {len(m['method_cells']):<4} "
              f"{len(m['rule_cells']):<5} {','.join(m['flows'])}")
    if json_out:
        payload = {
            "_meta": {
                "what": "139 份供给契约的清单（机器可读）",
                "generator": "build_contracts.py",
                "graph_digest": idx.graph_digest,
                "rule": "139 = 151 − 12；模板由 l3.serviceability 唯一导出",
            },
            "counts": {"total": len(rows), "A": n_a, "B": n_b},
            "contracts": rows,
        }
        Path(json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ {json_out}")
    return rows


FM_KEYS = ("contract_id", "template", "responsibility", "role_id", "role_title",
           "domain_id", "plane_id", "serviceability", "flows", "method_cells", "rule_cells")


def read_frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # 行尾 YAML 注释要剥掉：`flows: [A, B]  # 说明` 若不剥，列表判定会失败，
        # 于是 `norm_val` 退回整串，下游按**字符**逐个当元素遍历 —— 静默错配。
        # 判据：只在方括号已闭合（或不含方括号）时才剥，避免吃掉值里的 `#`。
        raw = line.split(":", 1)
        if len(raw) < 2:
            continue
        k, v = raw[0].strip(), raw[1]
        if v.count("[") == v.count("]"):
            v = re.sub(r"\s+#.*$", "", v)
        out[k] = v.strip()
    return out


def norm_val(v):
    v = (v or "").strip()
    if v.startswith("["):
        if not v.endswith("]"):
            # 拿不到就炸，不要给默认值 —— 否则调用方会把整串按字符遍历
            raise ValueError(f"frontmatter 列表字段格式不对（方括号未闭合）：{v!r}")
        inner = v[1:-1].strip()
        return [x.strip() for x in inner.split(",") if x.strip()]
    return v


def check(idx: GraphIndex, outdir: Path):
    """frontmatter 漂移检测：重算一遍，与文件逐字段比对。"""
    assignments = idx.assignments()
    files = sorted(p for p in outdir.rglob("CTR-*.md"))
    problems, checked, written, unwritten = [], 0, 0, 0
    for p in files:
        text = p.read_text(encoding="utf-8")
        fm = read_frontmatter(text)
        if not fm:
            problems.append((p, "无 frontmatter"))
            continue
        name = fm.get("responsibility", "")
        if name not in assignments:
            problems.append((p, f"责任名不在图谱内：{name!r}"))
            continue
        exp = assignments[name]
        for k in FM_KEYS:
            got, want = norm_val(fm.get(k)), exp[k]
            if got != want:
                problems.append((p, f"{k} 与图谱不一致：文件={got!r} 图谱={want!r}"))
        checked += 1
        if TODO_MARK in text:
            unwritten += 1
        else:
            written += 1
    total = len(assignments)
    cov = checked / total if total else 0.0
    print(f"契约文件 {len(files)} 个 · frontmatter 与图谱一致 {checked}/{total}（核对率 {cov:.1%}）")
    print(f"已撰写 {written} · 未撰写（含 {TODO_MARK}）{unwritten}")
    if cov < 0.9:
        print("⚠️ 核对率 < 90% ⇒ **不给「完整」结论**（「没东西可查」不等于「查过了没问题」）")
    for p, why in problems:
        print(f"🔴 {safe_rel(p)}: {why}")
    return 0 if not problems else 1


def selftest(idx: GraphIndex) -> int:
    """自证：划分、编号、M/R/D 分型、C 类排除，逐条可被篡改样本打红。"""
    fails, cases = [], 0
    assignments = idx.assignments()

    def expect(cond, label):
        nonlocal cases
        cases += 1
        if not cond:
            fails.append(label)

    # 1 总量：139 = 151 − 12
    sv = idx.graph["counts"]["l3_serviceability"]
    expect(len(assignments) == sv["A"] + sv["B"] == 139,
           f"契约总数应为 73+66=139，实得 {len(assignments)}")
    # 2 模板与类一一对应
    expect(all(m["template"] == m["serviceability"] for m in assignments.values()),
           "模板必须由 serviceability 唯一导出")
    # 3 计数与图谱逐类相等
    expect(sum(1 for m in assignments.values() if m["template"] == "A") == sv["A"], "A 计数应与图谱相等")
    expect(sum(1 for m in assignments.values() if m["template"] == "B") == sv["B"], "B 计数应与图谱相等")
    # 4 C 类零契约
    expect(all(idx.l3[n]["serviceability"] != "C" for n in assignments), "C 类不得有契约")
    # 5 责任名全局唯一且覆盖
    expect(len(assignments) == len(set(assignments)), "契约责任名不得重复")
    expect(set(assignments) == {n for n, i in idx.l3.items() if i["serviceability"] in "AB"},
           "契约必须恰好覆盖全部 A/B 类责任")
    # 6 contract_id 唯一且编号连续
    ids = sorted(m["contract_id"] for m in assignments.values())
    expect(len(ids) == len(set(ids)), "contract_id 不得重复")
    expect(ids == [f"CTR-A-{i:03d}" for i in range(1, sv["A"] + 1)] +
                  [f"CTR-B-{i:03d}" for i in range(1, sv["B"] + 1)], "contract_id 编号必须连续可复算")
    # 7 method_cells 全 M、rule_cells 全 R/D
    bad_m = [m["contract_id"] for m in assignments.values()
             if any(idx.cell_kind[c] != "M" for c in m["method_cells"])]
    expect(not bad_m, f"方法接入格必须全为 M 格，违反：{bad_m[:3]}")
    bad_r = [m["contract_id"] for m in assignments.values()
             if any(idx.cell_kind[c] not in ("R", "D") for c in m["rule_cells"])]
    expect(not bad_r, f"规则格必须全为 R/D 格，违反：{bad_r[:3]}")
    # 8 每份契约的格数 = flows × 8（3 M + 5 RD）
    bad_n = [m["contract_id"] for m in assignments.values()
             if len(m["method_cells"]) != 3 * len(m["flows"]) or len(m["rule_cells"]) != 5 * len(m["flows"])]
    expect(not bad_n, f"每份契约的格数应为 3×flows 与 5×flows，违反：{bad_n[:3]}")
    # 9 flows 必须等于岗位的 flows
    bad_f = [m["contract_id"] for m in assignments.values()
             if m["flows"] != idx.flows_of_role(m["role_id"])]
    expect(not bad_f, f"flows 必须等于该岗位的 flows，违反：{bad_f[:3]}")
    # 10 全局 M 格覆盖：至少一个 A 类责任能挂到每个 M 格
    touched = {c for m in assignments.values() if m["template"] == "A" for c in m["method_cells"]}
    expect(len(touched) == 24, f"24 个 M 格应全部可达（A 类），实得 {len(touched)}")

    # ---- 篡改样本：每条断言都要能被一份坏输入打红（否则是摆设断言）----
    def mut_total():
        a = idx.assignments()
        a.pop(next(iter(a)))
        return len(a) == 139

    expect(not mut_total(), "篡改样本①：删一份契约后总数断言必须失败")

    def mut_template():
        a = idx.assignments()
        k = next(iter(a))
        a[k] = dict(a[k], template="B" if a[k]["template"] == "A" else "A")
        return all(m["template"] == m["serviceability"] for m in a.values())

    expect(not mut_template(), "篡改样本②：翻转一份模板后「模板由服务性导出」必须失败")

    def mut_cell():
        a = idx.assignments()
        k = next(iter(a))
        a[k] = dict(a[k], method_cells=a[k]["method_cells"] + [a[k]["rule_cells"][0]])
        return not any(idx.cell_kind[c] != "M" for c in a[k]["method_cells"])

    expect(not mut_cell(), "篡改样本③：把 R 格塞进方法接入格必须失败")
    expect(not assignments[next(iter(assignments))]["template"].islower(),
           "篡改样本④：模板值必须是大写 A/B")

    print(f"自检：{cases - len(fails)}/{cases} 通过")
    for f in fails:
        print(f"🔴 {f}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="契约层（L4）生成器")
    ap.add_argument("--graph", default=str(GRAPH_DEFAULT))
    ap.add_argument("--outdir", default=str(CONTRACTS_DIR))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json-out")
    ap.add_argument("--skeleton", nargs="*")
    ap.add_argument("--all-skeletons", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    idx = GraphIndex(load_graph(Path(args.graph)))
    outdir = Path(args.outdir)

    if args.selftest:
        return selftest(idx)
    if args.check:
        return check(idx, outdir)
    if args.list:
        emit_list(idx, args.json_out)
        return 0
    if args.skeleton or args.all_skeletons:
        names = args.skeleton or []
        if args.all_skeletons:
            names = [n for n, i in idx.l3.items() if i["serviceability"] in "AB"]
        if not names:
            print("用法：--skeleton <责任名>… 或 --all-skeletons", file=sys.stderr)
            return 2
        write_skeleton(idx, names, outdir, force=args.force)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
