#!/usr/bin/env python3
"""registry_consistency.py — registry 门禁声明 vs 门禁实物 的一致性核对

## 为什么需要这个脚本

`papers_registry.json` 是「论文唯一事实源」，它的 `gates` 字段被下游当作事实读。
但它记的是**声明**，门禁产物记的是**实物**；两者会漂移。

本脚本的**存在理由本身就是一次实测教训**：第一版这个核对是临时写的 python，
用 `outputs.skill_card` 建索引去匹配门禁结果 —— 而 3 条「增强既有卡」型记录
**没有 `skill_card` 字段**（它们用的是 `enhanced_cards`）。于是那 3 条被
`if not rec: continue` **静默跳过**，脚本输出「无不一致」，看起来干净利落。
实际上它一条都没检查。

> 这正是本项目反复出现的失效模式：**「没东西可查」被当成「查过了没问题」**。
> 与 G2 旧基线（没有数字的卡自动通过）、quote_check 旧汇总（NO_QUOTES 并进通过）
> 是同一个错。所以本脚本把**覆盖率**作为一等输出：只声明「一致」不给分母的结论，
> 一律视为不可信。

用法：
    python3 registry_consistency.py                    # 控制台报告
    python3 registry_consistency.py --json-out <path>
    python3 registry_consistency.py --selftest         # 自检（含「覆盖率为 0 必须报警」）

退出码：0 = 无不一致；1 = 有不一致或覆盖率不足。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VAULT = REPO_ROOT / "paper2skills-vault"
REGISTRY = VAULT / "07-资源库" / "papers_registry.json"
GATES = VAULT / "07-资源库" / "gates"

# 覆盖率低于此值就拒绝给「一致」结论 —— 分母太小说明匹配逻辑错了，不是资产干净
MIN_COVERAGE = 0.9


def card_paths_of(rec: dict) -> list[str]:
    """一条 registry 记录指向的全部卡片（含两种形态）。

    ⚠️ 必须同时认 `outputs.skill_card`（新建卡）与 `outputs.enhanced_cards`
    （增强既有卡）。只认前者会静默丢掉所有增强记录 —— 本脚本诞生即因此。
    """
    o = rec.get("outputs") or {}
    out: list[str] = []
    if o.get("skill_card"):
        out.append(o["skill_card"])
    for c in o.get("enhanced_cards") or []:
        out.append(c)
    return out


def is_placeholder_path(p: str) -> bool:
    """是不是**占位路径**（计划要出卡但还没出）。

    实测 26 条 registry 记录的 `outputs.skill_card` 字面写着
    `paper2skills-vault/05-推荐系统/Skill-<方法名>.md` —— 尖括号里的
    `<方法名>` 从未被替换。这些记录的 `gates.evidence` 都是 `pending`。

    ⚠️ 为什么这是一个**独立**类别，不能和「路径腐烂」混为一谈：
    - 占位 = 计划中，尚未开工 → 无可核对，不该计入覆盖率分母；
    - 腐烂 = 声明已产出，但按路径找不到 → **是真缺陷**，必须报警。
    二者若混在一起，26 条占位会把覆盖率压到 44%，让「覆盖率不足」的警示
    变成常驻噪声，于是没人再看它。
    """
    return "<" in p or ">" in p


def load_gate_results() -> dict[str, dict]:
    """target 路径 → G2 结果。"""
    f = GATES / "gate_g2.json"
    if not f.is_file():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    return {r["target"]: r for r in data.get("results", [])}


def check() -> dict:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    gates = load_gate_results()
    if not gates:
        return {"error": "找不到 gate_g2.json —— 先跑 gate_check.py --all --outdir"}

    records = reg.get("records", [])
    with_card = [r for r in records if card_paths_of(r)]
    denom_records = len(with_card)
    findings: list[dict] = []
    checked = 0
    unresolved: list[dict] = []      # 路径腐烂：声明已产出但找不到
    planned = 0                      # 占位路径：计划中，未开工
    declared = 0                     # 分母 = 除占位外的全部卡片引用

    for rec in with_card:
        want = (rec.get("gates") or {}).get("evidence")
        for cp in card_paths_of(rec):
            if is_placeholder_path(cp):
                planned += 1
                continue
            declared += 1
            g = gates.get(cp)
            if g is None:
                unresolved.append({"paper_id": rec["paper_id"], "card": cp,
                                   "reason": "门禁产物里没有这张卡 —— 路径腐烂或已被删"})
                continue
            checked += 1
            reds = [f for f in g["findings"] if f["level"] == "RED"]
            n_red = len(reds)
            codes = sorted({f["code"] for f in reds})
            # 声明 pass 却实为红 → 不一致；声明 partial 允许有红灯（它已诚实登记）
            if want == "pass" and n_red:
                findings.append({
                    "paper_id": rec["paper_id"], "card": cp, "declared": want,
                    "actual_passed": g["passed"], "red": n_red, "codes": codes,
                    "verdict": "声明 pass 但门禁实为红",
                })
            elif want == "pass" and not g["passed"]:
                findings.append({
                    "paper_id": rec["paper_id"], "card": cp, "declared": want,
                    "actual_passed": g["passed"], "red": n_red, "codes": codes,
                    "verdict": "声明 pass 但门禁未通过（无红灯，可能是黄灯阻塞）",
                })

    coverage = checked / declared if declared else 0.0
    return {
        "registry_records": len(records),
        "records_with_cards": denom_records,
        "card_refs_total": planned + declared,
        "card_refs_planned_placeholder": planned,
        "card_refs_declared": declared,
        "card_refs_checked": checked,
        "coverage_pct": round(coverage * 100, 1),
        "inconsistencies": findings,
        "unresolved": unresolved,
        "coverage_ok": coverage >= MIN_COVERAGE,
    }


def selftest() -> int:
    """自检：证明本脚本在「覆盖率极低」时会报警，而不是给出干净的假结论。"""
    ok = True

    # 1｜两种卡片引用形态都要被认出来 —— 这是本脚本诞生的根因
    rec_new = {"outputs": {"skill_card": "a.md"}}
    rec_enh = {"outputs": {"enhanced_cards": ["b.md", "c.md"]}}
    rec_none = {"outputs": {}}
    got = (card_paths_of(rec_new), card_paths_of(rec_enh), card_paths_of(rec_none))
    want = (["a.md"], ["b.md", "c.md"], [])
    print(f"card_paths_of 三种形态: {got}")
    if got != want:
        print("❌ 未能识别 enhanced_cards 形态 —— 静默跳过的根因未修")
        ok = False
    else:
        print("✅ 新建卡与增强卡两种引用形态都能识别")

    # 2｜覆盖率门槛必须真的会拦住结论
    low = {"coverage_pct": 0.0, "coverage_ok": 0.0 >= MIN_COVERAGE}
    if low["coverage_ok"]:
        print("❌ 覆盖率为 0 时仍判 OK")
        ok = False
    else:
        print(f"✅ 覆盖率 0% 判为不可信（门槛 {MIN_COVERAGE:.0%}）")

    # 3｜对真实数据跑一遍并报告覆盖率（不判定失败，只展示分母）
    real = check()
    if "error" in real:
        print(f"⚠️  真实数据不可用：{real['error']}")
    else:
        print(f"真实数据：registry {real['registry_records']} 条 → "
              f"引用卡片 {real['card_refs_total']} 处，核对到 {real['card_refs_checked']} 处"
              f"（覆盖 {real['coverage_pct']}%）")

    print("✅ 自检通过" if ok else "❌ 自检失败")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="registry 门禁声明 vs 实物核对")
    ap.add_argument("--json-out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    r = check()
    if "error" in r:
        print(f"❌ {r['error']}")
        return 1

    print(f"registry 记录 {r['registry_records']} 条，其中引用卡片的 {r['records_with_cards']} 条")
    print(f"卡片引用 {r['card_refs_total']} 处："
          f"**占位（计划中，未开工）{r['card_refs_planned_placeholder']} 处**，"
          f"实际声明已产出 {r['card_refs_declared']} 处")
    print(f"已核对 {r['card_refs_checked']} / {r['card_refs_declared']}"
          f"　→ **覆盖率 {r['coverage_pct']}%**（分母已排除占位）")
    if not r["coverage_ok"]:
        print(f"⛔ 覆盖率低于 {MIN_COVERAGE:.0%} —— **本结论不可信**："
              f"不是「资产干净」，而是「大部分没被检查」")
    if r["unresolved"]:
        print(f"\n未核对的引用（{len(r['unresolved'])} 处）:")
        for u in r["unresolved"]:
            print(f"  - {u['paper_id']}: {u['card']} —— {u['reason']}")
    if r["inconsistencies"]:
        print(f"\n声明与实物不一致（{len(r['inconsistencies'])} 处）:")
        for f in r["inconsistencies"]:
            print(f"  ❌ {f['paper_id']}: {f['card']}")
            print(f"     {f['verdict']}｜红灯 {f['red']} 条 {f['codes'][:4]}")
    else:
        print("\n未发现不一致。")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"→ {args.json_out}")

    return 1 if (r["inconsistencies"] or not r["coverage_ok"]) else 0


if __name__ == "__main__":
    sys.exit(main())
