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

## ⚠️ 锚点是**内容寻址**的，不是行号寻址的（2026-09-13 实测撞出并重做）

首版把 `path:line` 里的行号**写死**（`Skill-Cold-Start-*` 写 `:11`）。当天就有并发写入者
（S10）往同一张卡插了 `venue_tier`/`venue_source` 两行 **⇒ 第 11 行从 `paper:` 变成了
`venue_source:`，而我登记的 sha1 是**改行前**算的**。后果不是「锚点漂了」，而是：

  · 那条锚点在文件的**任何版本里都找不到**（git 证明该文件相对 HEAD 是 +10/−0 纯插入）；
  · 而门禁的文案说「依据行在整份文件里都找不到 ⇒ **内容确实变了**」——
    **它在说一件没发生的事**，下一个人会去查「谁改了这张卡」。

修法（两层，缺一不可）：
  1. **锚点由内容定位**：不再写死行号，改成「按字段名找那一行」或「按子串找那一行」，
     找到谁就是谁；行号只是**副产品**。
  2. **写盘即自验**：每条锚点写完**立刻回读该文件第 ln 行、重算 sha1、断言相等**；
     不等就 `SystemExit`。这是本仓库那条纪律的锚点版：
     **先证明这个值是从真实文件算出来的，再把它当依据用。**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAULT = REPO / "paper2skills-vault"
CLASSIFICATION = VAULT / "07-资源库" / "card-classification.json"
INBOX = REPO / "paper2skills-research" / "data" / "card-classification-inbox.json"
SURVEY = REPO / "paper2skills-research" / "reports" / "_survey_card_inventory.md"

GROUPS = ["domain_misplaced", "not_baby_business", "no_baby_anchor",
          "no_baby_anchor_reproducible", "tech_domain_misplaced"]


def sha12(line: str) -> str:
    return hashlib.sha1(line.strip().encode("utf-8")).hexdigest()[:12]


def locate(rel_path: str, *, field: str | None = None, contains: str | None = None,
           section: str | None = None) -> tuple[int, str]:
    r"""在文件里**按内容**定位那一行，返回 (行号, 该行 sha1[:12])。

    三种定位方式（同时给就是多重约束，必须同一行全部满足）：
      · `field`    —— 顶格字段名（`paper` → `^paper:`）
      · `section`  —— 标题行，匹配 `^#{1,4}\s*<section>`（如 `②`）
      · `contains` —— 该行必须含这个子串

    ⚠️ **命中必须唯一**，否则 `SystemExit`。不唯一时**绝不退化成「取第一行」**——
    首版就是「找不到就取第 1 行」，于是锚点指向 `---` 却带着一个别处算来的 sha1，
    在文件里任何版本都找不到（2026-09-13 实测事故）。
    本条断言当场抓过一次：用裸子串 `②` 定位 `Skill-New-Product-Opportunity-Mining`
    命中 **[105, 760]**（正文里还有一处 `②`）⇒ 必须收紧成标题行匹配。
    """
    lines = (REPO / rel_path).read_text(encoding="utf-8").split("\n")
    hits = []
    for n, ln in enumerate(lines, 1):
        if field is not None and not re.match(rf"^{re.escape(field)}[ \t]*:", ln):
            continue
        if section is not None and not re.match(rf"^#{{1,4}}\s*{re.escape(section)}", ln):
            continue
        if contains is not None and contains not in ln:
            continue
        hits.append(n)
    if len(hits) != 1:
        raise SystemExit(f"❌ 锚点定位不唯一：{rel_path} field={field!r} section={section!r} "
                         f"contains={contains!r} 命中 {hits} —— "
                         f"定位条件必须唯一，不许退化成「取第一行」")
    n = hits[0]
    return n, sha12(lines[n - 1])


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

    def add(cid: str, group: str, basis: str, *, field=None, contains=None, section=None,
            expected_l3=None, ref=None, declared_tech_domain=None) -> None:
        if cid not in by:
            raise SystemExit(f"❌ 错位登记指向不存在的卡：{cid}")
        rec = by[cid]
        ln, h = locate(rec["path"], field=field, contains=contains, section=section)
        item = {"card_id": cid, "group": group, "basis": basis,
                "evidence": f"{rec['path']}:{ln}", "basis_line_sha1_12": h,
                "path": rec["path"]}
        if expected_l3 is not None:
            item["expected_l3"] = expected_l3
        if ref:
            item["ref"] = ref
        if declared_tech_domain:
            item["declared_tech_domain"] = declared_tech_domain
        entries.append(item)

    # ---- ① 技术域错位：21 条，全部来自 facet_flags（只给 ref，不复制理由）----
    # 依据行 = 卡里**自述它实际是什么**的那一行：优先 `description`（内容最实），
    # 没有就退 `paper`，再没有退 `module`。三条都是**按字段名定位**，不写死行号。
    for f in cc["facet_flags"]:
        cid = f["id"]
        rec = by[cid]
        fm_keys = {ln.split(":", 1)[0].strip()
                   for ln in (REPO / rec["path"]).read_text(encoding="utf-8").split("\n")[:40]
                   if re.match(r"^[A-Za-z_]+[ \t]*:", ln)}
        field = next((k for k in ("description", "paper", "topic", "module") if k in fm_keys), None)
        if field is None:
            raise SystemExit(f"❌ {cid}：frontmatter 里找不到可当作依据的字段")
        add(cid, "tech_domain_misplaced",
            f"技术域错位：声明 `{f['declared_tech_domain']}`，实际业务见 `ref` 登记",
            field=field, ref=cid, expected_l3=rec["l3"],
            declared_tech_domain=f["declared_tech_domain"])

    # ---- ② 域错位：三条，锚点都指向 `description:`（卡自述产出）----
    add("Skill-Monodense-单品价格弹性估计", "domain_misplaced",
        "实为定价卡：卡自述产出用于「动态定价和促销决策」，L3 落 `价格敏感性`（渠道经营），"
        "而技术域是 04-供应链", field="description")
    add("Skill-Cold-Start-Product-Recommendation", "domain_misplaced",
        "实为**推荐**卡：来源论文标题即 Large Language Model Simulator for Cold-Start "
        "Recommendation，技术域却是 06-增长模型", field="paper")
    add("Skill-New-Product-Opportunity-Mining", "domain_misplaced",
        "实为**选品**卡：来源论文是创业成功预测框架 SSFF，落 `市场机会评估`；"
        "调查报告 §5.3 明写「非本域核心动作，更接近选品/组合决策」，技术域却是 06-增长模型",
        field="paper")

    # ---- ③ 与母婴业务无关：2 条 ----
    add("Skill-Knowledge-Graph-for-Skills-Management", "not_baby_business",
        "② 段场景是**团队内部**新人技能/学习路径推荐，不是对外经营决策",
        field="evidence_basis")
    add("Skill-Skill-Lifecycle-Design", "not_baby_business",
        "元卡：讲的是重构**本项目自己的** Skill 库，不是母婴业务",
        field="doc_type")

    # ---- ④ §5.3 的 18 条「② 段无母婴锚点」 ----
    for cid in survey_5_3_ids():
        if cid not in by:
            raise SystemExit(f"❌ §5.3 的卡不在分类表里：{cid}")
        rec = by[cid]
        add(cid, "no_baby_anchor",
            f"调查报告 §5.3 逐卡异常：② 段未锚定到母婴单品/具体设定（散文判定，"
            f"机器扫描 class={scan[cid]['class']}）",
            section="②", expected_l3=rec["l3"])
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

    # ---- ★ 写盘前**逐条回读自验**：这一条是本次事故的直接产物 ----
    # 「不许把没从文件里算出来的值当依据用」—— 断言而不是注释。
    for e in entries:
        ln = int(e["evidence"].rsplit(":", 1)[1])
        lines = (REPO / e["path"]).read_text(encoding="utf-8").split("\n")
        if not (1 <= ln <= len(lines)):
            raise SystemExit(f"❌ {e['card_id']}：锚点行号 {ln} 越界（文件共 {len(lines)} 行）")
        got = sha12(lines[ln - 1])
        if got != e["basis_line_sha1_12"]:
            raise SystemExit(
                f"❌ {e['card_id']}：我写下的锚点自己都验不过 —— "
                f"登记 {e['basis_line_sha1_12']} ≠ 第 {ln} 行现算 {got}")

    ledger = {
        "_what": "错位清单（在册，不改正文）：把散落四处的散文判定合并成一份机读清单",
        "_rule": "**只登记不移动** —— 移动目录会断掉卡内 `related:` 与论文档案引用；"
                 "技术域只是筛选面（facet），主分类以 L3 为准（同 F4 的判定）",
        "_evidence": "每条给 `path:line` 与 `basis_line_sha1_12`（该行去空白后的 sha1 前 12 位）"
                     "⇒ 判定依据可**逐字复核**，不信散文。`tech_domain_misplaced` 只给 `ref`，"
                     "理由不复制（同一事实两份文本会各自腐烂）",
        "_anchor_rule": "锚点**由内容定位**（按字段名 / 子串），行号只是副产品；"
                        "写盘前逐条回读该行重算 sha1 并断言相等 —— "
                        "写死行号会在并发写入者插入字段时静默失效（2026-09-13 实测事故）",
        "_counts": {},
        "groups": GROUPS,
        "entries": entries,
    }
    ledger["_counts"] = dict(sorted(Counter(e["group"] for e in entries).items()))
    ledger["_counts"]["total"] = len(entries)
    ledger["_counts"]["distinct_cards"] = len({e["card_id"] for e in entries})

    # 自验的第二层：**锚点的快路径命中率**（慢路径由门禁的回捞负责）
    hit = sum(1 for e in entries
              if sha12((REPO / e["path"]).read_text(encoding="utf-8")
                       .split("\n")[int(e["evidence"].rsplit(":", 1)[1]) - 1]) == e["basis_line_sha1_12"])
    # ---- ★ 仪器自证：定位器的**唯一性守卫**必须真的会拦 ----
    # 「不许退化成取第一行」这句话本身也得是一条能失败的判据（同 `check_card_l3.py`
    # 里「样板标题不得产生命中」的那条仪器自证）。
    # ⚠️ 这里用的两个定位条件是**实测在真实卡上分别不唯一 / 唯一的**，
    #    不是编出来的例子：`Skill-New-Product-Opportunity-Mining` 的正文里有两处 `②`，
    #    而 `Skill-Cold-Start-Product-Recommendation` 只有一个 `paper:`。
    over = by.get("Skill-New-Product-Opportunity-Mining")
    uniq = by.get("Skill-Cold-Start-Product-Recommendation")
    if over and uniq:
        raised = False
        try:
            locate(over["path"], contains="②")          # 子串 `②` 在正文里出现两次
        except SystemExit:
            raised = True
        if not raised:
            raise SystemExit("❌ 仪器自证失败：定位条件不唯一时**必须**报错，"
                             "不许退化成「取第一行」（这正是本次事故的成因）")
        ln_u, h_u = locate(uniq["path"], field="paper")
        lines_u = (REPO / uniq["path"]).read_text(encoding="utf-8").split("\n")
        if h_u != sha12(lines_u[ln_u - 1]):
            raise SystemExit("❌ 仪器自证失败：唯一定位出来的锚点与文件不符")
        print(f"仪器自证：定位不唯一 ⇒ 报错 ✅ · 唯一定位 ⇒ 锚点自洽 ✅"
              f"（{uniq['id']} @ 第 {ln_u} 行）")

    print(json.dumps(ledger["_counts"], ensure_ascii=False))
    print(f"锚点自验：{hit}/{len(entries)} 条按登记行号即可命中（写盘前逐条断言过）")
    for e in entries[:2]:
        print(" ", json.dumps(e, ensure_ascii=False)[:150])

    if args.write:
        doc = json.loads(INBOX.read_text(encoding="utf-8"))
        doc["misplacement_ledger"] = ledger
        doc["_meta"]["misplacement_ledger"] = (
            "错位清单（在册不改正文）：21 条技术域错位（ref 指向 facet_flags）+ 3 条域错位 "
            "+ 2 条与母婴业务无关 + 18 条 §5.3 无锚点（其中 5 条机器可复现）。"
            "**只登记不移动**；每条带 path:line + 行 sha1 供逐字复核。"
            "锚点由内容定位、写盘前逐条回读自验。")
        # ⚠️ 写盘之后**再验一遍**：验证的是**落盘后的字节**，不是内存里的对象
        INBOX.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        back = json.loads(INBOX.read_text(encoding="utf-8"))["misplacement_ledger"]["entries"]
        bad = [e["card_id"] for e in back
               if sha12((REPO / e["path"]).read_text(encoding="utf-8")
                        .split("\n")[int(e["evidence"].rsplit(":", 1)[1]) - 1]) != e["basis_line_sha1_12"]]
        if bad:
            raise SystemExit(f"❌ 落盘后自验失败：{bad}")
        print(f"✓ 写入 {INBOX.relative_to(REPO)}（落盘后 {len(back)} 条锚点全部自验通过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
