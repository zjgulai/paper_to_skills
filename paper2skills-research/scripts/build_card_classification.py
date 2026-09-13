#!/usr/bin/env python3
"""PHASE6 F5 · 精选线 146 张卡的分类落点（五层轴的卡端事实源）。

为什么另立一份而不用产品侧 `dsh-paper2skills/data/classification.json`：

  · 两者是**两个语料**。产品侧那份是 1338 张 legacy 预览卡的落点（由 25 个源域的分类
    子任务产出、`classify-check.mjs` → `merge-classification.mjs` 合并）；本文件是
    **精选线 146 张卡**的落点，其中 93 张与产品侧重叠、53 张（PHASE3/4 新卡 +
    NLP-VOC 迁入卡）产品侧**根本没有**。
  · 产品侧那份**不能手工扩写** —— 它是流水线产物，手写一条就再也没法用
    `merge-classification.mjs` 复现。而横跨两仓的「同一张卡两个落点」正是腐烂的温床。

因此本文件的定位是：**精选线语料**的卡端事实源，且对重叠的 93 张设一道**零漂移门禁**——
逐条断言与产品侧相等；**要改就必须显式声明**（inbox 里 `reclassify: true` + 理由），
不许静默分叉。判据（1–3 个 L3 / L1-L2 由首 L3 导出 / 跨面多挂允许 / 缺口认领三档）
一律 reuse `dsh-paper2skills/lib/taxonomy.js`，**本脚本不另立一套分类学**。

三份文件的分工（照产品侧 `classify-check.mjs` → `merge-classification.mjs` 的先例）：

    data/card-classification-inbox.json   人/AI 填的分类结论（**输入**，入库）
        ↓ --write
    07-资源库/card-classification.json    卡端事实源（**产物**，入库）
        ↓
    capability-graph.json 的 cards[].l3   图谱消费（F2 生成器读本文件）

用法：
    python3 build_card_classification.py --write          # 写产物（有错即拒写）
    python3 build_card_classification.py --check          # 门禁（退出码 0/1/2）
    python3 build_card_classification.py --excerpts DIR   # 导出待分类卡的题面（分批）
    python3 build_card_classification.py --selftest       # 判据自证（含篡改样本）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAULT = REPO / "paper2skills-vault"
GRAPH = VAULT / "07-资源库" / "capability-graph.json"
OUT = VAULT / "07-资源库" / "card-classification.json"
INBOX = REPO / "paper2skills-research" / "data" / "card-classification-inbox.json"

SURVEY = REPO / "paper2skills-research" / "reports" / "_survey_card_inventory.md"

PRODUCT = Path("/Users/lute/project/Magpie-Horch/packages/capabilities/dsh-paper2skills")
PRODUCT_CLASSIFICATION = PRODUCT / "data" / "classification.json"
PRODUCT_TAXONOMY = PRODUCT / "data" / "taxonomy.json"

# ① 技术域 facet：取自 vault 顶层数字目录（**不是** p.parent.name —— 嵌套卡的父目录名
#    会退化成 `00-知识库-Skill卡片`，把 07-NLP-VOC 与 10-MAS 的卡混成一个域）。
TECH_DOMAIN_RE = re.compile(r"^\d{2}-")

# 「② 应用案例有没有锚到母婴出海」这件事，调查报告 §5.3 是用**散文**判的（18 条），
# 分类子代理又是各判各的（批次 02 自述「若要求含跨境渠道要素，这张可复议」）。
# ⇒ 本脚本把它降成**两个可复算的计数**：母婴品类词 / 出海要素词。词表在这里，
#   改词表就是改判据，因此计数随产物一起落库、可被复算。
# ⚠️ 计数前必须 `body_lines()` 剥掉标题行 —— 每张卡都有 `## ② 母婴出海应用案例` 样板标题，
#   不剥就会把章节名当成卡的出海要素（首版实测：146 张里 139 张被误判成 both）。
CATEGORY_RE = re.compile(r"吸奶器|纸尿裤|婴儿|母婴|宝宝|奶瓶|Momcozy|momcozy|背奶|辅食|童装|孕妇")
EXPORT_RE = re.compile(r"Amazon|亚马逊|独立站|跨境|出海|FBA|头程|清关|Shopify|shopify|海外仓|"
                       r"ASIN|Listing|listing|美国站|德国站|站内广告|关税")
L3_MIN, L3_MAX = 1, 3
CONFIDENCE = ("high", "medium", "low")


def _rel(p: Path) -> str:
    """仓库内相对路径；仓库外（如 /tmp 下的篡改副本）原样返回。

    门禁缺陷 #13 的同族在 F3 第二次出现：`relative_to(REPO)` 对仓库外路径抛 ValueError。
    这里一开始就吃掉，且 `_rel` 的探针回归守卫已在图谱脚本里。
    """
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def load_product_classification(path: Path) -> dict[str, dict]:
    """产品侧落点。**连 confidence/note 一起取** —— 只取 l3 然后把 confidence 填成
    自己的默认值，等于把「另一个判定器给出的值」伪装成本判定器的结论。"""
    return {i["id"]: {"l3": list(i.get("l3") or []),
                      "confidence": i.get("confidence", ""),
                      "note": i.get("note", "")}
            for i in load_json(path)["items"]}


def load_l3_names(path: Path) -> list[str]:
    return [e["name"] for e in load_json(path)["l3"]]


def body_lines(text: str) -> str:
    """剥掉所有标题行再计数。

    ⚠️ **这一条是实测撞出来的，不是预防性设计。** 首版直接扫全文，结果 146 张里 139 张
    判成 `both` —— 因为**每一张卡都有 `## ② 母婴出海应用案例` 这个样板标题**，
    于是 `出海`/`母婴` 两个词在**每张卡**上都命中一次。量到的是**章节标题**，不是卡的业务内容。
    （与 CLAUDE.md 记的「判据用错了信息源」同族第四次：仪器指错了地方。）
    """
    return "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("#"))


def anchor_scan(cards: list[dict]) -> dict[str, dict]:
    """机械扫描：每张卡的「母婴品类词 / 出海要素词」命中数（**只算正文，不算标题**）。

    这是**代理量**，不是真相 —— 但它对 146 张卡用同一把尺子，而散文判定做不到。
    两类都命中才算「锚定到母婴**出海**」；只有品类命中 ⇒ `category-only`。
    """
    out = {}
    for c in cards:
        try:
            text = body_lines((REPO / c["path"]).read_text(encoding="utf-8"))
        except FileNotFoundError:
            out[c["id"]] = {"category_hits": None, "export_hits": None, "class": "unreadable"}
            continue
        cat = len(CATEGORY_RE.findall(text))
        exp = len(EXPORT_RE.findall(text))
        cls = ("both" if cat and exp else
               "category-only" if cat else
               "export-only" if exp else "neither")
        out[c["id"]] = {"category_hits": cat, "export_hits": exp, "class": cls}
    return out


def load_survey_ids() -> list[str]:
    """调查报告 §5.3 的 18 条逐卡异常清单（散文判定）→ 卡 id。"""
    if not SURVEY.is_file():
        return []
    sec = SURVEY.read_text(encoding="utf-8").split("### 5.3")[1].split("\n## ")[0]
    return [f"Skill-{n}" for n in re.findall(r"^- \*\*#\d+ `([^`]+)`", sec, re.M)]


def tech_domain_of(vault_rel_path: str) -> str:
    """卡 → 技术域 facet。取 `paper2skills-vault/<域>/…` 的第一段。"""
    parts = Path(vault_rel_path).parts
    if len(parts) < 2 or parts[0] != "paper2skills-vault":
        return "UNKNOWN"
    return parts[1]


def enumerate_cards(graph: dict) -> list[dict]:
    """图上登记的全部卡（F2 的 rglob 口径：`Skill-*.md`，排除 `_superseded`）。"""
    return sorted(({
        "id": c["name"], "path": c["path"],
        "domain_dir": c["domain_dir"], "tech_domain": tech_domain_of(c["path"]),
    } for c in graph["cards"]), key=lambda x: x["id"])


def load_inbox_doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_inbox(path: Path) -> tuple[dict[str, dict], list[str]]:
    """inbox 是人工/AI 填的输入：**重复 id 必须报错**（dict 会静默取最后一条）。"""
    if not path.exists():
        return {}, []
    items = load_json(path).get("items", [])
    ids = [i.get("id") for i in items]
    dups = sorted({k for k, v in Counter(ids).items() if v > 1})
    return {i["id"]: i for i in items if i.get("id")}, dups


def build(graph_path: Path = GRAPH,
          classification_path: Path = PRODUCT_CLASSIFICATION,
          taxonomy_path: Path = PRODUCT_TAXONOMY,
          inbox_path: Path = INBOX) -> tuple[dict, list[str]]:
    """返回 (产物, 错误列表)。错误非空即不可写、--check 判红。"""
    graph = load_json(graph_path)
    product = load_product_classification(classification_path)
    l3_set = set(load_l3_names(taxonomy_path))
    cards = enumerate_cards(graph)
    inbox, inbox_dups = load_inbox(inbox_path)
    inbox_doc = load_inbox_doc(inbox_path)

    errs: list[str] = []
    if inbox_dups:
        errs.append(f"J1 inbox 有重复 id（后一条会静默覆盖前一条）：{inbox_dups}")
    items = []
    for c in cards:
        pid = c["id"]
        inherited = product.get(pid)["l3"] if product.get(pid) else None
        got = inbox.get(pid)
        base = {"id": pid, "path": c["path"], "tech_domain": c["tech_domain"],
                "domain_dir": c["domain_dir"]}

        if got is not None:
            rec = bool(got.get("reclassify"))
            item = dict(base, l3=list(got.get("l3") or []),
                        source="reclassified" if rec else "new",
                        confidence=got.get("confidence", ""),
                        note=got.get("note", ""))
            if rec:
                item["inherited_l3"] = list(inherited)
                if inherited is None:
                    errs.append(f"J4 {pid}：inbox 声明 reclassify 但产品侧没有这条（无原值可对照）")
                elif list(inherited) == list(item["l3"]):
                    errs.append(f"J4 {pid}：声明 reclassify 却与产品侧取值相同（空声明）")
                if not (item["note"] or "").strip():
                    errs.append(f"J4 {pid}：reclassify 必须写理由")
            elif inherited is not None:
                errs.append(f"J4 {pid}：产品侧已有落点 {inherited}，inbox 却按 new 提交 "
                            f"—— 这正是不许的静默分叉；要改就写 reclassify: true + 理由")
            item["confidence_source"] = "f5-review"
            if item["confidence"] not in CONFIDENCE:
                errs.append(f"J2 {pid}：confidence「{item['confidence']}」不在 {CONFIDENCE}")
        elif inherited is not None:
            item = dict(base, l3=list(inherited), source="inherited",
                        inherited_from=_rel(classification_path),
                        confidence=product[pid]["confidence"] or "",
                        confidence_source="product-pipeline",
                        note=product[pid]["note"] or "（产品侧流水线的逐卡理由为空）")
        else:
            item = dict(base, l3=[], source="unclassified", confidence="", note="")
        items.append(item)

    # ---- 逐条判据（每条都能失败）----
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        errs.append(f"J1 卡 id 重复：{[k for k, v in Counter(ids).items() if v > 1]}")
    graph_ids = {c["name"] for c in graph["cards"]}
    if set(ids) != graph_ids:
        errs.append(f"J1 与图上卡集合不一致：缺 {sorted(graph_ids - set(ids))} "
                    f"多 {sorted(set(ids) - graph_ids)}")

    classified = [i for i in items if i["l3"]]
    blank = [i for i in items if not i["l3"]]
    for i in classified:
        if not (L3_MIN <= len(i["l3"]) <= L3_MAX):
            errs.append(f"J2 {i['id']}：L3 数 {len(i['l3'])} 超出 {L3_MIN}–{L3_MAX}")
        if len(set(i["l3"])) != len(i["l3"]):
            errs.append(f"J2 {i['id']}：L3 有重复项 {i['l3']}")
        for n in i["l3"]:
            if n not in l3_set:
                errs.append(f"J2 {i['id']}：L3「{n}」不在 taxonomy 的 151 条内（禁止改写/新造/简称）")
    for i in blank:
        if i["source"] != "unclassified" and not i.get("note"):
            errs.append(f"J3 {i['id']}：空 l3 必须带 note 说明理由（否则就是静默漏分）")

    # 零漂移：inherited 必须与产品侧逐字相等（reclassified 的原值由 inbox 分支已断言）
    for i in items:
        cur = product[i["id"]]["l3"] if i["id"] in product else None
        if i["source"] == "inherited":
            if cur is None:
                errs.append(f"J4 {i['id']}：声明 inherited 但产品侧没有这条")
            elif list(cur) != i["l3"]:
                errs.append(f"J4 {i['id']}：**漂移** —— 精选线 {i['l3']} ≠ 产品侧 {cur}"
                            f"（要改就显式写 reclassify: true + 理由）")
        elif i["source"] == "reclassified":
            if cur is not None and list(cur) != list(i.get("inherited_l3") or []):
                errs.append(f"J4 {i['id']}：inherited_l3 {i.get('inherited_l3')} "
                            f"≠ 产品侧现值 {cur}（原值被改过，登记失效）")

    # ---- 技术域错位登记（facet 只登记不移动）----
    flag_ids = {c["id"] for c in cards}
    tech_of = {c["id"]: c["tech_domain"] for c in cards}
    flags = list(inbox_doc.get("facet_flags", []))
    for f in flags:
        if f.get("id") not in flag_ids:
            errs.append(f"J7 错位登记指向不存在的卡：{f.get('id')}")
            continue
        if f.get("kind") not in ("tech_domain_misplaced", "meta_card"):
            errs.append(f"J7 {f['id']}：kind「{f.get('kind')}」不在词表内")
        if not (f.get("actual_business") or "").strip():
            errs.append(f"J7 {f['id']}：错位登记必须写它实际是什么业务")
        if not (f.get("source") or "").strip():
            errs.append(f"J7 {f['id']}：错位登记必须写来源（散文结论也要留痕）")
        if f.get("declared_tech_domain") != tech_of.get(f.get("id")):
            errs.append(f"J7 {f['id']}：登记的 declared_tech_domain"
                        f"「{f.get('declared_tech_domain')}」与实测技术域「{tech_of.get(f.get('id'))}」不符")

    # ---- 母婴锚点：机器扫描 vs 子代理意见（两者都留，分歧如实报）----
    scan = anchor_scan(cards)
    opinions = list(inbox_doc.get("anchor_opinions", []))
    for o in opinions:
        if o.get("id") not in flag_ids:
            errs.append(f"J8 锚点意见指向不存在的卡：{o.get('id')}")
        if not (o.get("criterion") or "").strip():
            errs.append(f"J8 {o.get('id')}：锚点意见必须写明用了什么口径")
    disagreements = []
    for o in opinions:
        s_ = scan.get(o["id"], {})
        if s_.get("class") in (None, "unreadable"):
            continue
        scan_ok = s_["class"] == "both"
        if bool(o.get("baby_anchor")) != scan_ok:
            disagreements.append({"id": o["id"], "opinion": bool(o.get("baby_anchor")),
                                  "opinion_source": o.get("source"),
                                  "scan": s_["class"],
                                  "category_hits": s_["category_hits"],
                                  "export_hits": s_["export_hits"]})
    survey_ids = load_survey_ids()
    survey_view = [{"id": i, **scan.get(i, {})} for i in survey_ids]
    # 调查报告 §5.3 的原话是「业务场景**未锚定到母婴单品/具体设定**」—— 那是散文判定。
    # 机器扫描只能回答一个更弱的问题：「正文里**有没有**母婴品类词」。
    # 故如实拆两档：零品类命中的（可复现）vs 有命中但判定仍为泛化（不可复现）。
    survey_none = [v["id"] for v in survey_view if v.get("category_hits") == 0]
    survey_some = [v["id"] for v in survey_view if (v.get("category_hits") or 0) > 0]

    # 技术域 facet
    domains = sorted({c["tech_domain"] for c in cards})
    for i in items:
        if i["tech_domain"] != tech_domain_of(i["path"]):
            errs.append(f"J5 {i['id']}：tech_domain {i['tech_domain']} 与路径推导不符")
        if not TECH_DOMAIN_RE.match(i["tech_domain"]):
            errs.append(f"J5 {i['id']}：tech_domain「{i['tech_domain']}」不是顶层数字域目录")
    if "00-知识库-Skill卡片" in domains:
        errs.append("J5 tech_domain 退化成 `00-知识库-Skill卡片` —— 该目录是 07-NLP-VOC/ 与 "
                    "10-MAS/ 的内层目录，不是技术域")

    n_unclassified = len(blank)
    if n_unclassified:
        errs.append(f"J6 未分类 {n_unclassified}/{len(items)} 张："
                    f"{[i['id'] for i in blank][:8]}{' …' if n_unclassified > 8 else ''}")

    unserved = sorted(l3_set - {n for i in classified for n in i["l3"]})
    doc = {
        "_meta": {
            "what": "精选线 146 张卡的分类落点（五层轴的卡端事实源）",
            "why_not_product_side":
                "产品侧 data/classification.json 是 1338 张 legacy 卡的流水线产物（不可手写）；"
                "本文件覆盖的是精选线 146 张，其中 53 张产品侧根本没有。两个语料，两个账。",
            "axis_fact_source": "dsh-paper2skills/lib/taxonomy.js（判据）+ data/taxonomy.json（151 条 L1–L3 名）",
            "inbox": _rel(inbox_path),
            "drift_gate": "source=inherited 的条目逐条断言与产品侧相等；要改必须在 inbox 里写 "
                          "reclassify:true + 理由（J4），静默分叉一律报红",
            "tech_domain_basis": "paper2skills-vault/<域>/… 的第一段目录名（facet，不参与主分类）",
            "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "generator": _rel(Path(__file__)),
            "source_refs": {
                "capability_graph": _rel(graph_path),
                "product_classification": _rel(classification_path),
                "product_taxonomy": _rel(taxonomy_path),
            },
            "version": "1.0",
        },
        "total": len(items),
        "classified": len(classified),
        "unclassified": n_unclassified,
        "by_source": dict(sorted(Counter(i["source"] for i in items).items())),
        "by_confidence": dict(sorted(Counter(
            f'{i["confidence"] or "(空)"}@{i.get("confidence_source", "?")}' for i in classified).items())),
        "by_tech_domain": dict(sorted(Counter(i["tech_domain"] for i in items).items())),
        "l3_coverage": {
            "l3_named": len({n for i in classified for n in i["l3"]}),
            "l3_total": len(l3_set),
            "l3_unserved": unserved,
            "l3_unserved_count": len(unserved),
        },
        "tech_domains": domains,
        "facet_flags": flags,
        "anchor_scan": {
            "_criterion": "母婴品类词 / 出海要素词 两类都命中才算「锚定到母婴出海」（both）；"
                          "只有品类命中 ⇒ category-only。词表在生成器里，与产物同源可复算。",
            "by_class": dict(sorted(Counter(v["class"] for v in scan.values()).items())),
            "cards": scan,
        },
        "anchor_opinions": opinions,
        "anchor_disagreements": disagreements,
        "survey_5_3": {
            "_what": "调查报告 §5.3 的 18 条逐卡异常（散文判定）经本次机械扫描后的取值",
            "count": len(survey_view),
            "by_class": dict(sorted(Counter(v.get("class") for v in survey_view).items())),
            "reproducible": {
                "_why": "§5.3 判的是「场景泛化/未落到母婴单品」（散文）；机器只能测「正文有无母婴品类词」。"
                        "两者只在**零命中**这一档上等价 ⇒ 拆开报，不把不可复现的部分说成已验证。",
                "zero_category_hits": survey_none,
                "has_category_hits": survey_some,
            },
            "cards": survey_view,
        },
        "reclassified": sorted(i["id"] for i in items if i["source"] == "reclassified"),
        "items": items,
    }
    return doc, errs


# --------------------------------------------------------------------------- #
# 题面导出：给分类用（不含任何结论，只给事实）
# --------------------------------------------------------------------------- #
def excerpts(outdir: Path, batch: int = 13, graph_path: Path = GRAPH,
             classification_path: Path = PRODUCT_CLASSIFICATION,
             inbox_path: Path = INBOX) -> list[Path]:
    graph = load_json(graph_path)
    product = load_product_classification(classification_path)
    inbox, _ = load_inbox(inbox_path)
    todo = [c for c in enumerate_cards(graph)
            if not product.get(c["id"]) and c["id"] not in inbox]
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for bi in range(0, len(todo), batch):
        chunk = todo[bi:bi + batch]
        lines = [f"# 待分类卡（第 {bi // batch + 1} 批，{len(chunk)} 张）", ""]
        for c in chunk:
            text = (REPO / c["path"]).read_text(encoding="utf-8")
            body = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
            lines += [f"## {c['id']}", f"（vault 路径：{c['path']} · 技术域：{c['tech_domain']}）", "",
                      body[:2600].strip(), "", "---", ""]
        p = outdir / f"batch-{bi // batch + 1:02d}.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p)
    return written


# --------------------------------------------------------------------------- #
# selftest：每条判据都配一份篡改样本（断言恒真 = 没断言）
# --------------------------------------------------------------------------- #
def _heading_not_counted() -> bool:
    """仪器自证：`## ② 母婴出海应用案例` 样板标题本身**不得**产生任何命中。"""
    head = "## ② 母婴出海应用案例\n\n### 场景：通用电商转化率提升\n"
    plain = body_lines(head)
    return not CATEGORY_RE.search(plain) and not EXPORT_RE.search(plain)


def selftest() -> int:
    import tempfile

    graph = load_json(GRAPH)
    product = load_product_classification(PRODUCT_CLASSIFICATION)
    l3_names = load_l3_names(PRODUCT_TAXONOMY)
    cards = enumerate_cards(graph)

    # 干净底：93 张继承 + 53 张按一份「完美的」inbox 提交
    clean_inbox = {"items": []}
    for c in cards:
        if product.get(c["id"]):
            continue
        clean_inbox["items"].append({"id": c["id"], "l3": ["质量分析"], "confidence": "medium",
                                     "note": "selftest 填位"})
    clean_inbox["items"].append({"id": "Skill-Monodense-单品价格弹性估计",
                                 "l3": ["价格敏感性", "组合设计"], "confidence": "medium",
                                 "reclassify": True,
                                 "note": "selftest：产品侧落点属供应链，卡内实为定价"})

    ok = bad = 0

    def run(graph_doc, inbox_doc):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            gp, cp, tp, ip = (tdp / n for n in ("g.json", "c.json", "t.json", "i.json"))
            gp.write_text(json.dumps(graph_doc), encoding="utf-8")
            cp.write_text(json.dumps({"items": [
                {"id": k, "l3": v["l3"], "confidence": v["confidence"], "note": v["note"]}
                for k, v in product.items()]}), encoding="utf-8")
            tp.write_text(json.dumps({"l3": [{"name": n} for n in l3_names]}), encoding="utf-8")
            ip.write_text(json.dumps(inbox_doc), encoding="utf-8")
            return build(gp, cp, tp, ip)

    def case(name, graph_doc, inbox_doc, expect):
        nonlocal ok, bad
        _, errs = run(graph_doc, inbox_doc)
        hit = any(expect in e for e in errs)
        print(f"  {'✅' if hit else '❌'} {name}"
              + ("" if hit else f"  ← 期望命中「{expect}」，实得 {errs[:2] or '0 错误'}"))
        if hit:
            ok += 1
        else:
            bad += 1

    def mutate_inbox(fn):
        d = json.loads(json.dumps(clean_inbox))
        fn(d)
        return d

    def mutate_graph(fn):
        d = json.loads(json.dumps(graph))
        fn(d)
        return d

    print("selftest（每条判据配一份篡改样本，必须逐条报红）")
    # 反向用例先跑：干净底必须 0 错误（否则下面的报红可能来自别的判据）
    _, clean_errs = run(graph, clean_inbox)
    print(f"  {'✅' if not clean_errs else '❌'} 反向用例：干净底必须 0 错误"
          + ("" if not clean_errs else f"  ← 实得 {clean_errs[:3]}"))
    if not clean_errs:
        ok += 1
    else:
        bad += 1

    MONO = "Skill-Monodense-单品价格弹性估计"

    def mono(**kw):
        def m(d):
            for it in d["items"]:
                if it["id"] == MONO:
                    it.update(kw)
        return m

    case("J4 静默分叉：改产品侧已有落点却没写 reclassify",
         graph, mutate_inbox(mono(reclassify=False, l3=["质量分析"])), "静默分叉")
    case("J4 空声明：reclassify 了但取值与产品侧相同",
         graph, mutate_inbox(mono(reclassify=True, l3=list(product[MONO]["l3"]))), "空声明")
    case("J4 reclassify 无理由",
         graph, mutate_inbox(mono(reclassify=True, l3=["质量分析"], note="")), "必须写理由")
    case("J1 inbox 重复 id 必须报错",
         graph, mutate_inbox(lambda d: d["items"].append(dict(
             [i for i in d["items"] if i["id"] == MONO][0]))), "重复 id")
    case("J2 L3 名不在 151 条内",
         graph, mutate_inbox(lambda d: d["items"].append(
             {"id": "Skill-Agentic-Catalog-Enrichment", "l3": ["不存在的责任名"],
              "confidence": "high"})),
         "不在 taxonomy")
    case("J2 L3 数量超上限",
         graph, mutate_inbox(lambda d: d["items"].append(
             {"id": "Skill-Agentic-Catalog-Enrichment",
              "l3": ["质量分析", "物流方案", "促销规划", "组合设计"], "confidence": "high"})),
         "超出 1–3")
    case("J2 confidence 取值非法",
         graph, mutate_inbox(lambda d: d["items"].append(
             {"id": "Skill-Agentic-Catalog-Enrichment", "l3": ["质量分析"],
              "confidence": "very-high"})),
         "confidence")
    case("J3 空 l3 无 note（静默漏分）",
         graph, mutate_inbox(lambda d: d["items"].append(
             {"id": "Skill-Agentic-Catalog-Enrichment", "l3": [], "confidence": "low"})),
         "必须带 note")
    case("J6 少填一张卡即报未分类",
         graph, mutate_inbox(lambda d: d["items"].pop(0)),
         "未分类")
    print("  " + ("✅" if _heading_not_counted() else "❌")
          + " 仪器自证：只有样板标题、正文无锚点词的卡必须判 neither")
    if _heading_not_counted():
        ok += 1
    else:
        bad += 1

    case("J5 tech_domain 退化成嵌套目录名",
         mutate_graph(lambda d: [c.update({"path": c["path"].replace("07-NLP-VOC", "00-知识库-Skill卡片")})
                                 for c in d["cards"] if "07-NLP-VOC" in c["path"]]),
         clean_inbox, "内层目录")

    print(f"\nselftest：{ok} 抓 / {bad} 漏")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--excerpts", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--batch", type=int, default=13)
    ap.add_argument("--graph", type=Path, default=GRAPH)
    ap.add_argument("--classification", type=Path, default=PRODUCT_CLASSIFICATION)
    ap.add_argument("--taxonomy", type=Path, default=PRODUCT_TAXONOMY)
    ap.add_argument("--inbox", type=Path, default=INBOX)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--out-json", type=Path, help="门禁结果另存 JSON（可指向仓库外）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.excerpts:
        for p in excerpts(args.excerpts, args.batch, args.graph, args.classification, args.inbox):
            print(f"  {p}  {p.stat().st_size} B")
        return 0

    doc, errs = build(args.graph, args.classification, args.taxonomy, args.inbox)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"ok": not errs, "errors": errs, "classified": doc["classified"],
             "total": doc["total"], "by_source": doc["by_source"],
             "generated": doc["_meta"]["generated"]}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    if args.check:
        print(f"卡 {doc['total']} · 已分类 {doc['classified']} · 未分类 {doc['unclassified']} "
              f"· 技术域 {len(doc['tech_domains'])} · L3 覆盖 {doc['l3_coverage']['l3_named']}"
              f"/{doc['l3_coverage']['l3_total']}")
        print(f"来源 {doc['by_source']} · 置信度 {doc['by_confidence']}")
        if errs:
            print(f"\n✗ {len(errs)} 项不合格：")
            for e in errs[:40]:
                print(f"  · {e}")
            if len(errs) > 40:
                print(f"  …（共 {len(errs)} 项）")
            return 1
        print("\n✅ 判据全过")
        return 0

    if errs:
        print(f"✗ 拒绝写入（{len(errs)} 项不合格）：")
        for e in errs[:40]:
            print(f"  · {e}")
        return 1
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()[:12]
    print(f"✓ 写入 {_rel(args.out)}（{args.out.stat().st_size // 1024} KB, sha256:{sha}）")
    print(f"  卡 {doc['total']} · 已分类 {doc['classified']} · 技术域 {len(doc['tech_domains'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
