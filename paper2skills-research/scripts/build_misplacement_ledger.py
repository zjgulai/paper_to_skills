#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成错位清单的**输入**（`card-classification-inbox.json` 的 `misplacement_ledger` 段）。

一次性工具，不入流水线 —— 它把四处散落的散文判定合并成一份机读清单，
每条都带**可逐字复核**的依据：`path:line` + 该行的 `sha1[:12]`。

依据分五类（`groups` 词表）：

· `tech_domain_misplaced` 21 条 —— 取自 `facet_flags`（只加 `ref`，**不复制理由**，
  避免同一事实两份文本各自腐烂）
· `domain_misplaced` 3 条 —— 方案里登记的两类「卡在错的技术域，且实际业务另有归属」：
  `04-供应链/Skill-Monodense` 实为定价卡、`06-增长模型` 两张分别实为推荐卡与选品卡
· `not_baby_business` 2 条 —— 与母婴业务无关（内部技能推荐 / 本项目自己的元卡）
· `no_baby_anchor` 18 条 —— 调查报告 §5.3 的「② 段无母婴锚点」逐卡异常
· `no_baby_anchor_reproducible` 5 条 —— 上面 18 条里**机器可复现**的子集（品类词零命中）

用法::

    python3 build_misplacement_ledger.py            # 只打印，不改文件
    python3 build_misplacement_ledger.py --write    # 写进 inbox
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAULT = REPO / "paper2skills-vault"
CLASSIFICATION = VAULT / "07-资源库" / "card-classification.json"
INBOX = REPO / "paper2skills-research" / "data" / "card-classification-inbox.json"
SURVEY = REPO / "paper2skills-research" / "reports" / "_survey_card_inventory.md"

GROUPS = ["domain_misplaced", "not_baby_business", "no_baby_anchor",
          "no_baby_anchor_reproducible", "tech_domain_misplaced"]


def ev_line(rel_path: str, line: int) -> tuple[str, str]:
    """(证据串, 该行 sha1 前 12 位)。行号从 1 起，越界即抛（不静默）。"""
    text = (REPO / rel_path).read_text(encoding="utf-8").split("\n")
    if not (1 <= line <= len(text)):
        raise ValueError(f"{rel_path} 只有 {len(text)} 行，取不到第 {line} 行")
    return f"{rel_path}:{line}", hashlib.sha1(text[line - 1].strip().encode("utf-8")).hexdigest()[:12]


def survey_5_3_ids() -> list[str]:
    """调查报告 §5.3 的 18 条逐卡异常 → 卡 id（`Skill-<名>`）。"""
    sec = SURVEY.read_text(encoding="utf-8").split("### 5.3")[1].split("\n## ")[0]
    return [f"Skill-{n}" for n in re.findall(r"^- \*\*#\d+ `([^`]+)`", sec, re.M)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cc = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
    by = {i["id"]: i for i in cc["items"]}
    scan = cc["anchor_scan"]["cards"]
    entries: list[dict] = []

    def add(cid: str, group: str, basis: str, line: int, expected_l3=None,
            not_l3=None, ref=None) -> None:
        if cid not in by:
            raise SystemExit(f"❌ 错位登记指向不存在的卡：{cid}")
        rec = by[cid]
        e, h = ev_line(rec["path"], line)
        item = {"card_id": cid, "group": group, "basis": basis,
                "evidence": e, "basis_line_sha1_12": h, "path": rec["path"]}
        if expected_l3 is not None:
            item["expected_l3"] = expected_l3
        if not_l3 is not None:
            item["l3_not"] = not_l3
        if ref:
            item["ref"] = ref
        entries.append(item)

    # ---- ① 技术域错位：21 条，全部来自 facet_flags（只给 ref，不复制理由）----
    for f in cc["facet_flags"]:
        cid = f["id"]
        rec = by[cid]
        # 依据行 = 卡里自述它实际是什么的那一行（描述 / topic / module 三选一，取第一个存在者）
        text = (REPO / rec["path"]).read_text(encoding="utf-8").split("\n")
        ln = next((n for n, l in enumerate(text, 1)
                   if re.match(r"^(description|topic|module|title):", l)), 1)
        add(cid, "tech_domain_misplaced",
            f"技术域错位：声明 `{f['declared_tech_domain']}`，实际业务见 `ref` 登记",
            ln, ref=cid, expected_l3=rec["l3"])
        entries[-1]["declared_tech_domain"] = f["declared_tech_domain"]

    # ---- ② 域错位（卡在错的技术域，且实际业务另有归属）：方案 §登记的三条 ----
    add("Skill-Monodense-单品价格弹性估计", "domain_misplaced",
        "实为定价卡：卡自述产出用于「动态定价和促销决策」，L3 落 `价格敏感性`（渠道经营），"
        "而技术域是 04-供应链", 4)
    add("Skill-Cold-Start-Product-Recommendation", "domain_misplaced",
        "实为**推荐**卡：来源论文标题即 Large Language Model Simulator for Cold-Start "
        "Recommendation，技术域却是 06-增长模型", 11)
    add("Skill-New-Product-Opportunity-Mining", "domain_misplaced",
        "实为**选品**卡：来源论文是创业成功预测框架 SSFF，落 `市场机会评估`；"
        "调查报告 §5.3 明写「非本域核心动作，更接近选品/组合决策」，技术域却是 06-增长模型", 11)

    # ---- ③ 与母婴业务无关：2 条 ----
    add("Skill-Knowledge-Graph-for-Skills-Management", "not_baby_business",
        "② 段场景是**团队内部**新人技能/学习路径推荐，不是对外经营决策", 65)
    add("Skill-Skill-Lifecycle-Design", "not_baby_business",
        "元卡：讲的是重构**本项目自己的** Skill 库，不是母婴业务", 95)

    # ---- ④ §5.3 的 18 条「② 段无母婴锚点」 ----
    s53 = survey_5_3_ids()
    for cid in s53:
        if cid not in by:
            raise SystemExit(f"❌ §5.3 的卡不在分类表里：{cid}")
        rec = by[cid]
        text = (REPO / rec["path"]).read_text(encoding="utf-8").split("\n")
        ln = next((n for n, l in enumerate(text, 1) if re.match(r"^#{1,3}\s*②", l)), 1)
        add(cid, "no_baby_anchor",
            f"调查报告 §5.3 逐卡异常：② 段未锚定到母婴单品/具体设定（散文判定，"
            f"机器扫描 class={scan[cid]['class']}）", ln, expected_l3=rec["l3"])
        entries[-1]["machine_scan"] = {k: scan[cid][k]
                                       for k in ("class", "category_hits", "export_hits")}
    # ⑤ 其中**机器可复现**的子集（母婴品类词零命中）
    for e in [x for x in entries if x["group"] == "no_baby_anchor"]:
        if e["machine_scan"]["category_hits"] == 0:
            e2 = dict(e)
            e2["group"] = "no_baby_anchor_reproducible"
            e2["basis"] = ("同一张卡的另一档：母婴品类词**零命中**——这一档是"
                           "**机器可复现**的（可复算 `anchor_scan.cards[].category_hits == 0`）")
            entries.append(e2)

    ledger = {
        "_what": "错位清单（在册，不改正文）：把散落四处的散文判定合并成一份机读清单",
        "_rule": "**只登记不移动** —— 移动目录会断掉卡内 `related:` 与论文档案引用；"
                 "技术域只是筛选面（facet），主分类以 L3 为准（同 F4 的判定）",
        "_evidence": "每条给 `path:line` 与 `basis_line_sha1_12`（该行去空白后的 sha1 前 12 位）"
                     "⇒ 判定依据可**逐字复核**，不信散文。`tech_domain_misplaced` 只给 `ref`，"
                     "理由不复制（同一事实两份文本会各自腐烂）",
        "_counts": {},
        "groups": GROUPS,
        "entries": entries,
    }
    from collections import Counter
    ledger["_counts"] = dict(sorted(Counter(e["group"] for e in entries).items()))
    ledger["_counts"]["total"] = len(entries)
    ledger["_counts"]["distinct_cards"] = len({e["card_id"] for e in entries})

    print(json.dumps(ledger["_counts"], ensure_ascii=False))
    for e in entries[:3]:
        print(" ", json.dumps(e, ensure_ascii=False)[:160])
    print(f"… 共 {len(entries)} 条")

    if args.write:
        doc = json.loads(INBOX.read_text(encoding="utf-8"))
        doc["misplacement_ledger"] = ledger
        doc["_meta"]["misplacement_ledger"] = (
            "错位清单（在册不改正文）：21 条技术域错位（ref 指向 facet_flags）+ 3 条域错位 "
            "+ 2 条与母婴业务无关 + 18 条 §5.3 无锚点（其中 5 条机器可复现）。"
            "**只登记不移动**；每条带 path:line + 行 sha1 供逐字复核。")
        INBOX.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"✓ 写入 {INBOX.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
