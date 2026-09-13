#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 · 旧方案页收割器 —— 把 `playbook/solutions/` 的 19 个历史方案页收成 M 格素材。

## 收什么、丢什么

**收**（三族，每条都带 source + line + line_end + anchor 的四元锚点）：

| 族 | 源结构 | 内容 |
|---|---|---|
| `layers` | `.sd-layers > .sd-layer` | 分层架构：层号 / 层名 / 职责描述 |
| `roadmap` | `.sd-phases > .sd-phase` | 路线图：阶段名 / 时长 / 动作 / 预期产出 |
| `traps` | `.sd-traps > .sd-trap` | 架构陷阱：编号 / 陷阱名 / 说明全文 |

**丢**（`discarded.card_position_index`，显式登记、一等输出）：
19 页的「核心 Skill 索引」段 = `.sd-skill-chip` 锚点，**194 条 / 181 个唯一卡 id**。
它是**第四套平行分类**的产物（全文 grep `AGT-0` / `SCN-0` 零命中），且产品侧
`classification.json` 已是 1338 张装线卡的权威归属 ⇒ 收割它会造出第二份卡片归属事实源。

⚠️ 但它**被当作 join key 使用**（本文件里唯一可派生的连接），这一点在产物里如实登记为
`discarded.card_position_index.used_as_join_key = true` —— 丢弃的是「内容」，不是「连接」。

## 落到 64 格：全派生，不另造分类

    chip href ──► classification.json items[].id ──► items[].l3[]
              ──► capability-graph.json l3[].name ──► l3[].role_id ──► roles[].flows[]
              ──► cells[] where flow_id ∈ flows ──► (FLOW-0X, STG-0Y)，M 格 = cell_kind "M"

**每一跳都是已有事实源**，没有一格是新判的。STG 维度**不可从旧方案页派生**
（源页面没有任何阶段化字段）—— 故产物里 STG 是「由格展开」而来，并在报告里写死这条口径。

## 用法

    python3 paper2skills-research/scripts/harvest_legacy_solutions.py --run
    python3 paper2skills-research/scripts/harvest_legacy_solutions.py --check
    python3 paper2skills-research/scripts/harvest_legacy_solutions.py --selftest
    # 夹具/回归用（--run 与 --check 都认）：
    ... --solutions-dir <dir> --graph <g.json> --classification <c.json> \
        --out-json <o.json> --out-md <o.md>

退出码：**0 全过 / 1 判据红 / 2 输入没拿到（≠ 通过）/ 3 脚本内部错误（≠ 判红）**

## 本脚本的纪律（本仓库付过学费的坑）

1. **每条判据都要能失败**：`--selftest` 里 13 条判据各配一份**篡改样本**（改坏产物一个字段）。
2. **变异测试**：把判据守卫行改成 `if False:`，同一份篡改样本必须**由红转绿** ⇒ 证明该守卫
   是承重的，不是摆设（台账 #25：判据在 `main()` 里而自检只测库函数，删掉守卫照样全绿）。
3. **端到端**：至少一条用 `subprocess` 跑真 CLI + 夹具目录，不 import 库函数。
4. **输入拿不到 ⇒ exit 2**：目录不存在、图/分类缺、解析出 0 条。
5. **`--selftest` 不留临时文件**：全部写 `tempfile.mkdtemp()`，`finally` 删除，并在收尾**断言
   `scripts/` 目录跑前跑后无新增文件**。
6. **样板不得产生条目**：解析**限定到 `section.content`**；反向控制用**构造样本**证明这条
   限定不是恒真（去掉限定 ⇒ 侧边栏里的诱饵立刻被收成 1 条）。
"""
from __future__ import annotations

import argparse
import hashlib
import html as _html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ─────────────────────────── 路径与常量 ───────────────────────────

#: 仓库根。**必须可注入**：`--selftest` 把脚本的副本写进 tmp 再跑（变异测试），
#: 若只按 `__file__` 推，副本的 REPO 会变成 tmp 的祖父目录 ⇒ 源文件相对路径整体变形
#: ⇒ 变异体与原件**不是「只差一行」**，变异测试当场失去意义（本脚本首版实测撞到：
#: 变异体在 J0b/J7/J12 上因为路径口径不同而恒红）。
REPO = Path(os.environ.get("P2S_REPO_ROOT") or Path(__file__).resolve().parents[2])
SOLUTIONS_DIR_DEFAULT = REPO / "playbook" / "solutions"
GRAPH_DEFAULT = REPO / "paper2skills-vault" / "07-资源库" / "capability-graph.json"
OUT_JSON_DEFAULT = REPO / "paper2skills-research" / "data" / "legacy-solutions.json"
OUT_MD_DEFAULT = REPO / "paper2skills-research" / "reports" / "PHASE6-S2-旧方案页收割.md"

#: 产品侧装线卡分类（跨仓库只读依赖；本仓库 `build_gap_ledger.py` 已用同一个文件）。
#: 找不到 ⇒ exit 2（**不是**「跳过落格」）。
CLASSIFICATION_DEFAULT = (REPO.parent / "Magpie-Horch" / "packages" / "capabilities"
                          / "dsh-paper2skills" / "data" / "classification.json")
CLASSIFICATION_FALLBACK_GLOB = "*/packages/capabilities/dsh-paper2skills/data/classification.json"

FAMILIES = ("layers", "roadmap", "traps")

#: 进入 / 退出条件的显式标记词。**源 HTML 里没有这两个字段** —— 本表存在的唯一目的，
#: 是让「0 条」这个结论变成**算出来的**（探测器能命中构造样本，见 selftest 反向控制），
#: 而不是「我没找」。
ENTRY_MARKERS = ("进入条件", "前置条件", "准入条件", "启动条件", "触发条件", "入口条件")
EXIT_MARKERS = ("退出条件", "完成条件", "验收条件", "结项条件", "离场条件")

#: 解析作用域。**这一行是承重行**：改成 False ⇒ 侧边栏/`<style>`/脚本块都会进正文语料。
CONTENT_SCOPE_DEFAULT = True  # <<SCOPE>>

#: 逐字内容比对时**只比文本载荷、不比锚点**（锚点由 J2 单独判）——
#: 否则任何一条锚点篡改都会被两条判据同时抓住，变异测试就分不出谁在承重。
TEXT_FIELDS = {
    "layers": ("layer_no", "name", "desc"),
    "roadmap": ("phase_name", "duration", "action", "outcome"),
    "traps": ("trap_no", "title", "body"),
}

#: 判据表：(编号, 名称, 打红样本摘要)。数字与报告共用一个来源。
JUDGEMENTS = [
    ("J0a", "逐字内容漂移", "改 traps[0].title 一个字"),
    ("J0b", "源指纹漂移", "改 sources[0].sha256"),
    ("J1", "落格 join 完整性", "by_page[0].l3[0].role_id → AGT-999"),
    ("J2", "锚点合法性", "layers[0].line → 999999"),
    ("J3", "逐字回指", "layers[0].line_end → line（区间缩到一行）"),
    ("J4", "丢弃登记 + 泄漏扫描",
     "①往 harvest 注入 card_index 键；②把卡位标签塞进 layers[0].note（非文本载荷字段）"),
    ("J5", "计数现算一致", "counts.chips_entries_total 194 → 193"),
    ("J6", "报告数字同步", "改报告内嵌数字块 chips_entries_total"),
    ("J7", "页面划分完整", "从 by_page 删掉一页"),
    ("J8", "未 join 不静默", "清空 landing.missing.unjoined_chips"),
    ("J9", "M 格自洽", "by_m_cell[0].n_pages → 99"),
    ("J10", "进入/退出条件现算", "counts.roadmap_phases_with_labeled_entry → 3"),
    ("J11", "样板不产生条目", "counts.boilerplate_items_scoped → 5"),
    ("J12", "层数与 h2 声明自洽", "pages[0].layers_declared_depth → 99"),
    ("J13", "L3 落点自洽且满 151", "从 by_l3 删掉一条"),
]

#: J5 负责的 counts 键（其余键分别归 J10 / J11，避免两条判据抢同一个字段）。
J5_KEYS = (
    "source_files_scanned", "solution_pages", "index_pages",
    "layer_rows", "roadmap_phases", "trap_rows",
    "chips_entries_total", "chips_entries_joined", "chips_unique_total",
    "chips_unique_joined", "chips_unique_join_curated_146", "chips_unique_unjoined",
    "anchor_verbatim_ok", "judgements_declared", "mutations_declared",
    "tamper_samples_declared",
    "harvest_entries_total",
    "l3_landed", "l3_total", "roles_landed", "flows_landed",
    "m_cells_total", "m_cells_with_any_page",
)

#: 禁键名：收割产物里**不得**出现「卡位索引」这一族结构（丢弃登记所在处除外）。
FORBIDDEN_HARVEST_KEYS = ("card_index", "skill_index", "chips", "card_position_index")

NUMBERS_BEGIN = "<!-- LEGACY-SOLUTIONS-NUMBERS:BEGIN"
NUMBERS_END = "LEGACY-SOLUTIONS-NUMBERS:END -->"


class InputMissing(Exception):
    """输入没拿到 —— 调用方必须转成 exit 2，**不得**当成「没问题」。"""


class InternalError(Exception):
    """脚本内部错误 —— exit 3（≠ 判红）。"""


# ─────────────────────────── 小工具 ───────────────────────────

def sha256_16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
#: ⚠️ **只剥「真的像标签」的尖括号**：`<` 后必须紧跟 `/` 或**字母**。
#: 首版写成 `<[^>]+>`，而本语料里有裸的小于号（`接受率<40%时…`、`季节MAPE稳定在<12%`），
#: 于是 `<40%时…接受率。</p>` 被整段当成一个标签吃掉 —— 抽取出的字段没受影响（字段切片在 `</p>` 之前截断），
#: 但**行区间回查时**同一段被截成半句 ⇒ 判据 `J3` 在首跑报出 6 条「字段不在申报区间内」。
#: 同族先例：漏洞 #14（`\w` 把汉字算单词字符）、#18（`isalpha()` 把汉字算字母）。
RE_TAG = re.compile(r"</?[A-Za-z][^>]*>")


def strip_tags(s: str) -> str:
    return RE_TAG.sub("", RE_COMMENT.sub("", s))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(strip_tags(s))).strip()


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def now_cst() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S%z")


def resolve_classification(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise InputMissing(f"指定的 classification.json 不存在：{p}")
        return p
    if CLASSIFICATION_DEFAULT.is_file():
        return CLASSIFICATION_DEFAULT
    hits = sorted(REPO.parent.glob(CLASSIFICATION_FALLBACK_GLOB))
    if hits:
        return hits[0]
    raise InputMissing(
        "产品侧 classification.json 找不到 —— 落格链路（chip → 卡 → L3 → 岗位 → FLOW → 格）"
        f"整条断掉。预期位置：{CLASSIFICATION_DEFAULT}；可 `--classification` 指定。")


# ─────────────────────────── 解析 ───────────────────────────

RE_LAYER = re.compile(
    r'<div class="sd-layer">\s*'
    r'<div class="sd-layer-no"[^>]*>(?P<no>.*?)</div>\s*'
    r'<div class="sd-layer-body">\s*<strong>(?P<name>.*?)</strong>\s*'
    r'<span>(?P<desc>.*?)</span>', re.S)
RE_PHASE = re.compile(
    r'<span class="sd-phase-name"[^>]*>(?P<name>.*?)</span>\s*'
    r'<span class="sd-phase-dur"[^>]*>(?P<dur>.*?)</span>\s*</div>\s*'
    r'<div class="sd-phase-action">(?P<action>.*?)</div>\s*'
    r'<div class="sd-phase-roi">(?P<outcome>.*?)</div>', re.S)
RE_TRAP = re.compile(
    r'<span class="sd-trap-no">(?P<no>.*?)</span>\s*<div>\s*'
    r'<strong>(?P<title>.*?)</strong>\s*<p>(?P<body>.*?)</p>', re.S)
RE_CHIP = re.compile(r'<a class="sd-skill-chip" href="\.\./skills/(?P<cid>[^"]+)\.html">'
                     r'(?P<label>.*?)</a>')
RE_H2 = re.compile(r"<h2>(?P<t>.*?)</h2>", re.S)
RE_HERO_H1 = re.compile(r"<h1>(?P<t>.*?)</h1>", re.S)
RE_SUBTITLE = re.compile(r'<p class="sd-subtitle">(?P<t>.*?)</p>', re.S)
RE_CATEGORY = re.compile(r'<span class="sd-category">(?P<t>.*?)</span>', re.S)
RE_UPDATED = re.compile(r'<span class="sd-updated">(?P<t>.*?)</span>', re.S)
RE_ROI_BANNER = re.compile(r'<div class="sd-roi-banner">(?P<t>.*?)</div>', re.S)
RE_SUMMARY = re.compile(r'<p class="sd-summary">(?P<t>.*?)</p>', re.S)
RE_DEPTH_H2 = re.compile(r"^(\d+)\s*层架构设计$")
RE_INDEX_H2 = re.compile(r"^核心\s*Skill\s*索引（(\d+)\s*个）$")


def content_slice(text: str, scoped: bool = CONTENT_SCOPE_DEFAULT) -> tuple:
    """把页面切到 `section.content` 内，返回 `(正文, 切片起点偏移)`。

    ⚠️ **必须返回偏移量**：首版只返回正文，而 `_pos` 是**切片内**的下标，再拿去对**全文**算行号
    ⇒ 所有锚点行号整体前移，最大的那批正好落进侧边栏（39–42 行）——
    即「样板被当成正文」的坐标版。由判据 `J2` 在首跑时实测撞出（6 条 `[42,41]` 越界）。
    """
    if not scoped:
        return text, 0
    i = text.find('<section class="content">')
    if i < 0:
        return text, 0
    j = text.rfind("</section>")
    if j <= i:
        return text[i:], i
    return text[i:j], i


def parse_page(rel: str, text: str, scoped: bool = CONTENT_SCOPE_DEFAULT) -> dict:
    """解析单页。样板（header/sidebar/style/script）在 scoped=True 时天然被排除。"""
    body, base = content_slice(text, scoped)

    def anchor(selector: str, idx: int) -> str:
        return f"{selector}[{idx}]"

    layers = []
    for k, m in enumerate(RE_LAYER.finditer(body), start=1):
        layers.append({
            "page": rel,
            "index": k,
            "layer_no": norm(m.group("no")),
            "name": norm(m.group("name")),
            "desc": norm(m.group("desc")),
            "anchor": anchor("div.sd-layers > div.sd-layer", k),
            "_pos": base + m.start(),
            "_end": base + m.end(),
        })

    roadmap = []
    for k, m in enumerate(RE_PHASE.finditer(body), start=1):
        roadmap.append({
            "page": rel,
            "index": k,
            "phase_name": norm(m.group("name")),
            "duration": norm(m.group("dur")),
            "action": norm(m.group("action")),
            "outcome": norm(m.group("outcome")),
            "anchor": anchor("div.sd-phases > div.sd-phase", k),
            "_pos": base + m.start(),
            "_end": base + m.end(),
        })

    traps = []
    for k, m in enumerate(RE_TRAP.finditer(body), start=1):
        traps.append({
            "page": rel,
            "index": k,
            "trap_no": norm(m.group("no")),
            "title": norm(m.group("title")),
            "body": norm(m.group("body")),
            "anchor": anchor("div.sd-traps > div.sd-trap", k),
            "_pos": base + m.start(),
            "_end": base + m.end(),
        })

    chips = []
    for m in RE_CHIP.finditer(body):
        chips.append({"card_id": m.group("cid"), "label": norm(m.group("label"))})

    h2s = [norm(x) for x in RE_H2.findall(body)]
    depth = None
    for t in h2s:
        mm = RE_DEPTH_H2.match(t)
        if mm:
            depth = int(mm.group(1))
            break

    def one(rx):
        m = rx.search(body)
        return norm(m.group("t")) if m else None

    return {
        "rel": rel,
        "layers": layers,
        "roadmap": roadmap,
        "traps": traps,
        "chips": chips,
        "section_titles": h2s,
        "layers_declared_depth": depth,
        "hero_title": one(RE_HERO_H1),
        "subtitle": one(RE_SUBTITLE),
        "category": one(RE_CATEGORY),
        "updated": one(RE_UPDATED),
        "roi_banner": one(RE_ROI_BANNER),
        "summary": one(RE_SUMMARY),
    }


def count_items(text: str, scoped: bool = CONTENT_SCOPE_DEFAULT) -> int:
    p = parse_page("_probe_", text, scoped)
    return len(p["layers"]) + len(p["roadmap"]) + len(p["traps"])


def boilerplate_of(texts: list[str]) -> str:
    """语料的公共骨架 = 在**全部**页面里逐行相同的行（保持首现顺序）。

    这就是「样板」的定义 —— 不是手写的，是从语料算出来的。
    """
    if not texts:
        return ""
    others = [set(t.split("\n")) for t in texts[1:]]
    out = []
    for ln in texts[0].split("\n"):
        if ln.strip() and all(ln in o for o in others):
            out.append(ln)
    return "\n".join(out)


def boilerplate_h2_titles(per_page: list[dict]) -> list[str]:
    """在全部页面里都出现的 h2 标题 = 样板标题（**不得成为任何收割条目的名字**）。

    来由（同族事故）：上一轮曾把样板标题 `## ② 母婴出海应用案例` 当成卡的业务要素，
    146 张里 139 张误判。
    """
    if not per_page:
        return []
    common = set(per_page[0]["section_titles"])
    for p in per_page[1:]:
        common &= set(p["section_titles"])
    return sorted(common)


# ─────────────────────────── 落格（全派生） ───────────────────────────

def build_landing(per_page: list[dict], graph: dict, classi: dict) -> dict:
    items = {i["id"]: i for i in classi.get("items", []) if i.get("id")}
    l3_by_name = {x["name"]: x for x in graph["l3"]}
    role_by_id = {r["id"]: r for r in graph["roles"]}
    cells = graph["cells"]
    m_cells = [c for c in cells if c["cell_kind"] == "M"]
    curated = {c["name"] for c in graph.get("cards", [])}
    flows_all = sorted({c["flow_id"] for c in cells})
    stage_of = {c["stage_id"] for c in cells}

    by_page, unjoined_all = [], []
    for p in per_page:
        rel = p["rel"]
        chip_ids = sorted({c["card_id"] for c in p["chips"]})
        l3map, unjoined, n_curated = {}, [], 0
        for cid in chip_ids:
            it = items.get(cid)
            if it is None:
                unjoined.append(cid)
                continue
            if cid in curated:
                n_curated += 1
            for name in (it.get("l3") or []):
                if name in l3_by_name:
                    l3map.setdefault(name, set()).add(cid)
                else:
                    # 分类里出现不在 151 名单内的 L3 名 ⇒ 事实源之间打架，直接报内部错误
                    raise InternalError(
                        f"classification.json 的 L3 名不在图谱 151 名单内：{name!r}（卡 {cid}）")
        l3s = sorted(l3map)
        roles = sorted({l3_by_name[n]["role_id"] for n in l3s})
        flows = sorted({f for r in roles for f in (role_by_id[r].get("flows") or [])
                        if f in flows_all})
        mcells = sorted({c["cell_id"] for c in m_cells if c["flow_id"] in flows})
        entry = {
            "page": rel,
            "_curated_hits": sorted(c for c in chip_ids if c in curated),
            "chips_entries": len(p["chips"]),
            "chips_entries_joined": len(p["chips"]) - len(
                [c for c in p["chips"] if c["card_id"] in unjoined]),
            "chips_unique": len(chip_ids),
            "chips_unique_joined_classification": len(chip_ids) - len(unjoined),
            "chips_unique_joined_curated_146": n_curated,
            "chips_unjoined": sorted(unjoined),
            "l3": [{"name": n,
                    "role_id": l3_by_name[n]["role_id"],
                    "role_title": role_by_id[l3_by_name[n]["role_id"]]["title"],
                    "serviceability": l3_by_name[n]["serviceability"],
                    "plane_id": l3_by_name[n]["plane_id"],
                    "domain_id": l3_by_name[n]["domain_id"],
                    "via_cards": sorted(l3map[n])} for n in l3s],
            "role_ids": roles,
            "flow_ids": flows,
            "m_cells": mcells,
        }
        by_page.append(entry)
        unjoined_all.extend(unjoined)

    pages_with_any = [e["page"] for e in by_page if e["l3"]]
    pages_without_any = [e["page"] for e in by_page if not e["l3"]]

    by_l3 = []
    for name in sorted(l3_by_name):
        pages = sorted(e["page"] for e in by_page
                       if any(x["name"] == name for x in e["l3"]))
        by_l3.append({
            "l3": name,
            "role_id": l3_by_name[name]["role_id"],
            "serviceability": l3_by_name[name]["serviceability"],
            "n_pages": len(pages),
            "pages": pages,
        })

    by_m_cell = []
    for c in sorted(m_cells, key=lambda x: x["cell_id"]):
        pages = sorted(e["page"] for e in by_page if c["cell_id"] in e["m_cells"])
        l3s = sorted({x["name"] for e in by_page if c["cell_id"] in e["m_cells"]
                      for x in e["l3"]})
        by_m_cell.append({
            "cell_id": c["cell_id"],
            "flow_id": c["flow_id"],
            "stage_id": c["stage_id"],
            "stage_name": c["stage_name"],
            "business_step": c["business_step"],
            "n_pages": len(pages),
            "pages": pages,
            "n_l3": len(l3s),
            "l3": l3s,
        })

    uniq_all = sorted({c["card_id"] for p in per_page for c in p["chips"]})
    uq = {
        "chips_entries_total": sum(len(p["chips"]) for p in per_page),
        "chips_unique_total": len(uniq_all),
        "chips_unique_joined_classification": len(uniq_all) - len(set(unjoined_all)),
        "chips_unique_joined_curated_146": len(
            {c for e in by_page for c in e.get("_curated_hits", [])}),
        "chips_unique_unjoined": sorted(set(unjoined_all)),
    }
    for e in by_page:
        e.pop("_curated_hits", None)
    return {
        "uniqueness": uq,
        "join_chain": "chip href → classification.json items[].id → items[].l3[] → "
                      "capability-graph l3[].name → l3[].role_id → roles[].flows[] → "
                      "cells[] where flow_id ∈ flows（M 格 = cell_kind \"M\"）",
        "join_key_namespace": "classification.json items[].id",
        "stg_note": "STG 维度**不可从旧方案页派生**（源页面无任何阶段化字段）；"
                    "产物里的 STG 是 M 格展开出来的，不是从旧页面判出来的。",
        "by_page": by_page,
        "by_l3": by_l3,
        "by_m_cell": by_m_cell,
        "missing": {
            "unjoined_chips": sorted(set(unjoined_all)),
            "pages_without_any_join": pages_without_any,
            "pages_with_any_join": len(pages_with_any),
            "stages_covered": sorted(stage_of),
        },
    }


# ─────────────────────────── 构建产物 ───────────────────────────

def build(solutions_dir: Path, graph_path: Path, classi_path: Path) -> dict:
    if not solutions_dir.is_dir():
        raise InputMissing(f"方案页目录不存在：{solutions_dir}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    classi = json.loads(classi_path.read_text(encoding="utf-8"))

    all_html = sorted(solutions_dir.glob("*.html"))
    pages_paths = [p for p in all_html if p.name != "index.html"]
    index_paths = [p for p in all_html if p.name == "index.html"]
    if not pages_paths:
        raise InputMissing(f"{solutions_dir} 下没有任何方案页（sol-*.html）")
    if not index_paths:
        raise InputMissing(f"{solutions_dir} 下缺 index.html —— 声明的输入没到齐")

    src_root = REPO if str(solutions_dir).startswith(str(REPO)) else solutions_dir.parent
    sources, per_page = [], []
    for p in all_html:
        raw = p.read_bytes()
        text = raw.decode("utf-8")
        rel = str(p.relative_to(src_root)) if str(p).startswith(str(src_root)) else p.name
        scoped = parse_page(rel, text)   # 走 CONTENT_SCOPE_DEFAULT，**不写死**
        scoped["_text"] = text
        scoped["_bytes"] = len(raw)
        scoped["_sha"] = sha256_16(raw)
        scoped["_lines"] = text.count("\n") + 1
        sources.append({"path": rel, "kind": "index" if p.name == "index.html" else "solution",
                        "sha256": scoped["_sha"], "bytes": len(raw), "lines": scoped["_lines"]})
        per_page.append(scoped)

    sol_pages = [p for p in per_page if p["rel"].endswith(".html") and p["_bytes"] and p["chips"] is not None
                 and not p["rel"].endswith("index.html")]

    # 锚点补齐（行号只能在拿到全文后算）
    for p in per_page:
        text = p.pop("_text")
        for fam in FAMILIES:
            for it in p[fam]:
                it["source"] = p["rel"]
                it["line"] = line_of(text, it.pop("_pos"))
                # 块尾行取**匹配自身的结束位置**，不是「下一块起点-1」——
                # 后者让每族最后一条一路铺到文件末（实测 L102–L285），锚点等于没锚。
                it["line_end"] = line_of(text, max(it["_end"] - 1, 0))
                it.pop("_end")
        p["line_count"] = p.pop("_lines")
        p["sha256"] = p.pop("_sha")
        p["bytes"] = p.pop("_bytes")

    harvest = {fam: [it for p in sol_pages for it in p[fam]] for fam in FAMILIES}

    total_items = sum(len(harvest[f]) for f in FAMILIES)
    if total_items == 0 or not sol_pages:
        raise InputMissing(
            f"从 {len(all_html)} 个 HTML 里解析出 {total_items} 条收割条目 / {len(sol_pages)} 个方案页 "
            "—— 「拿到 0 条」不是「没有东西」，是输入没拿到")

    # 样板重建需要原文（`_text` 上面已 pop，按 rel 回读一次，与 sha256 同源）
    raw_texts = [_read_by_rel(solutions_dir, p["rel"]) for p in sol_pages]
    boiler = boilerplate_of(raw_texts)
    bp_scoped = count_items(boiler, True)
    bp_unscoped = count_items(boiler, False)
    bp_titles = boilerplate_h2_titles(sol_pages)

    landing = build_landing(sol_pages, graph, classi)

    chips_total = sum(len(p["chips"]) for p in sol_pages)
    chips_unique = len({c["card_id"] for p in sol_pages for c in p["chips"]})
    uq = landing["uniqueness"]
    depths = [p["layers_declared_depth"] for p in sol_pages if p["layers_declared_depth"]]
    hist = {}
    for p in sol_pages:
        for it in p["layers"]:
            hist[it["layer_no"]] = hist.get(it["layer_no"], 0) + 1
    n_entry = sum(1 for it in harvest["roadmap"]
                  if any(k in it["phase_name"] + it["action"] + it["outcome"] + it["duration"]
                         for k in ENTRY_MARKERS))
    n_exit = sum(1 for it in harvest["roadmap"]
                 if any(k in it["phase_name"] + it["action"] + it["outcome"] + it["duration"]
                        for k in EXIT_MARKERS))

    counts = {
        "source_files_scanned": len(sources),
        "solution_pages": len(sol_pages),
        "index_pages": len(index_paths),
        "layer_rows": len(harvest["layers"]),
        "roadmap_phases": len(harvest["roadmap"]),
        "trap_rows": len(harvest["traps"]),
        "harvest_entries_total": total_items,
        "layer_depth_declared_min": min(depths) if depths else None,
        "layer_depth_declared_max": max(depths) if depths else None,
        "layer_no_histogram": hist,
        "layers_h2_mismatch_pages": 0,   # 现算值在下方重填
        "anchor_verbatim_ok": 0,         # 现算值在下方重填
        "roadmap_phases_with_labeled_entry": n_entry,
        "roadmap_phases_with_labeled_exit": n_exit,
        "roadmap_phases_with_outcome_field": sum(1 for it in harvest["roadmap"] if it["outcome"]),
        "trap_pages": len({it["source"] for it in harvest["traps"]}),
        "trap_empty_section_pages": sorted(
            p["rel"] for p in sol_pages
            if any("陷阱" in t for t in p["section_titles"]) and not p["traps"]),
        "chips_entries_total": chips_total,
        "chips_entries_joined": sum(e["chips_entries_joined"] for e in landing["by_page"]),
        "chips_unique_total": chips_unique,
        "chips_unique_joined": uq["chips_unique_joined_classification"],
        "chips_unique_join_curated_146": uq["chips_unique_joined_curated_146"],
        "chips_unique_unjoined": len(uq["chips_unique_unjoined"]),
        "chips_unjoined_list": uq["chips_unique_unjoined"],
        "boilerplate_common_lines": boiler.count("\n") + 1 if boiler else 0,
        "boilerplate_common_lines_unique": len(set(boiler.split("\n"))) if boiler else 0,
        "boilerplate_items_scoped": bp_scoped,
        "boilerplate_items_unscoped": bp_unscoped,
        "boilerplate_h2_titles": len(bp_titles),
        "l3_landed": sum(1 for x in landing["by_l3"] if x["n_pages"] > 0),
        "l3_total": len(landing["by_l3"]),
        "roles_landed": len({r for e in landing["by_page"] for r in e["role_ids"]}),
        "flows_landed": len({f for e in landing["by_page"] for f in e["flow_ids"]}),
        "m_cells_total": len(landing["by_m_cell"]),
        "m_cells_with_any_page": sum(1 for x in landing["by_m_cell"] if x["n_pages"] > 0),
        "m_cell_min_pages": min((x["n_pages"] for x in landing["by_m_cell"]), default=0),
        "m_cell_max_pages": max((x["n_pages"] for x in landing["by_m_cell"]), default=0),
        "pages_without_any_join": len(landing["missing"]["pages_without_any_join"]),
        "judgements_declared": len(JUDGEMENTS),
        "mutations_declared": len(MUTATIONS),
        "tamper_samples_declared": len({m[2] for m in MUTATIONS if m[2] != "SCOPE"}),
    }
    counts["anchor_verbatim_ok"] = anchor_verbatim_hits(
        {"harvest": harvest, "sources": sources}, {"harvest": harvest},
        base_dir=solutions_dir)[0]
    counts["layers_h2_mismatch_pages"] = sum(
        1 for p in sol_pages
        if p["layers_declared_depth"] is not None
        and p["layers_declared_depth"] != len(p["layers"]))

    pages_meta = []
    for p in sol_pages:
        pages_meta.append({
            "file": p["rel"],
            "sha256": p["sha256"],
            "bytes": p["bytes"],
            "lines": p["line_count"],
            "title": p["hero_title"],
            "subtitle": p["subtitle"],
            "category": p["category"],
            "updated": p["updated"],
            "section_titles": p["section_titles"],
            "layers_declared_depth": p["layers_declared_depth"],
            "n_layers": len(p["layers"]),
            "n_roadmap": len(p["roadmap"]),
            "n_traps": len(p["traps"]),
            "n_chips": len(p["chips"]),
        })

    return {
        "_meta": {
            "what": "S2 · 19 个旧方案页的 M 格素材收割（L1–Ln 分层架构 / 路线图 / 三大陷阱）"
                    "＋ 194 卡位索引的丢弃登记；落格全走已有事实源派生，未另造分类。",
            "generated": now_cst(),
            "generator": "paper2skills-research/scripts/harvest_legacy_solutions.py",
            "sources": {
                "solutions_dir": str(solutions_dir),
                "graph": {"path": str(graph_path), "sha256": sha256_16(graph_path.read_bytes())},
                "classification": {"path": str(classi_path),
                                   "sha256": sha256_16(classi_path.read_bytes()),
                                   "n_items": len(classi.get("items", []))},
            },
            "exit_codes": "0 全过 / 1 判据红 / 2 输入没拿到（≠ 通过）/ 3 脚本内部错误（≠ 判红）",
            "judgements": [{"id": j[0], "name": j[1], "tamper": j[2]} for j in JUDGEMENTS],
        },
        "counts": counts,
        "sources": sources,
        "pages": pages_meta,
        "harvest": harvest,
        "landing": landing,
        "discarded": {
            "card_position_index": {
                "what": "19 个方案页的「核心 Skill 索引（N 个）」段 —— `.sd-skill-chip` 锚点构成的"
                        "「方案页卡位 → 卡」映射表",
                "n_entries": chips_total,
                "n_unique_card_ids": chips_unique,
                "pages_with_index": sum(1 for p in sol_pages if p["chips"]),
                "n_entries_joined": sum(e["chips_entries_joined"] for e in landing["by_page"]),
                "n_unique_joined": uq["chips_unique_joined_classification"],
                "n_unique_joined_curated_146": uq["chips_unique_joined_curated_146"],
                "used_as_join_key": True,
                "join_key_note": "**丢弃的是内容，不是连接**：181 个唯一卡 id 被当作 join key"
                                 "（chip href → classification.json items[].id），这是旧方案页"
                                 "通往 64 格体系**唯一可派生**的连接；但它本身不作为素材条目登记。",
                "unjoined": uq["chips_unique_unjoined"],
                "why_discarded": [
                    "它是**第四套平行分类**的产物：全 19 页 grep `AGT-0` / `SCN-0` 零命中，"
                    "卡位不与任何岗位 / 责任 / 场景 / 格绑定。",
                    "卡位映射是 2026-06 的历史快照，产品侧 `classification.json`（1338 张装线卡）"
                    "已是卡片归属的权威事实源；收割卡位会造出**第二份卡片归属事实源**。",
                    "本仓库纪律：**丢弃必须显式登记**，否则数字上看起来像「这 19 页本来就没那么多东西」。",
                ],
            }
        },
        "source_gaps": [
            {"gap": "路线图的「进入条件 / 退出条件」没有独立字段",
             "evidence": f"72 个 Phase 只有 阶段名 / 时长 / 动作 / 预期产出 四个字段；"
                         f"带显式进入条件标记的 = {n_entry} 条、显式退出条件标记的 = {n_exit} 条",
             "handling": "不编造。产物里用 `outcome`（预期产出）如实承载「退出证据」，"
                         "并在报告中写死「进入条件复算不出来」。"},
            {"gap": "分层架构的「边界 / 与其他层的关系」没有独立字段",
             "evidence": "`.sd-layer` 只有 层号 / 层名 / 描述 三元组；描述里不含 边界/不得/禁止 "
                         "一类边界语（实测 0 命中）",
             "handling": "不编造。只收三元组。"},
            {"gap": "1 个方案页的「三大架构陷阱」段是空的",
             "evidence": "sol-counterfactual-pricing.html 有 h2「三大架构陷阱」但 `.sd-trap` 计数 = 0 "
                         "⇒ 陷阱实收 54 = 18 页 × 3，而非 19 页 × 3",
             "handling": "登记为空段，不替它补 3 条。"},
        ],
        "not_harvested_this_round": [
            {"family": "hero 标题 / 副标题 / 分类 / 更新日", "n": len(sol_pages),
             "why": "本任务三族之外，未收（已解析进 pages[] 元数据，但未作为素材条目登记）"},
            {"family": "ROI 横幅（`.sd-roi-banner`）", "n": sum(1 for p in sol_pages if p["roi_banner"]),
             "why": "同上；且其中的 ROI 数字无论文/业务出处，按 G2 口径不宜作 M 格素材"},
            {"family": "摘要段（`.sd-summary`）", "n": sum(1 for p in sol_pages if p["summary"]),
             "why": "同上"},
            {"family": "核心 Skill 索引（`.sd-skill-chip`）", "n": chips_total,
             "why": "**显式丢弃**，见 discarded.card_position_index"},
        ],
    }


# ─────────────────────────── 判据 ───────────────────────────
# 每条判据的结构统一为：算出 bad 列表 → 一行带标记的守卫 → 返回。
# 守卫行必须**单独成行且带唯一标记**，`--selftest` 的变异测试按标记整行替换。

def load_sources(art: dict, base_dir=None) -> dict:
    """按产物记录的路径把源文件读回来（供 J2 / J3 现场重算）。

    回退序：仓库相对路径 → `_meta.sources.solutions_dir` 下的同名文件。
    ⚠️ 首版回退到 `SOLUTIONS_DIR_DEFAULT`（写死的真语料目录）⇒ **夹具环境下会去读真语料**，
    判据在夹具上恒红。回退目标必须取自产物自己记录的输入，不能写死。
    """
    base = Path(base_dir) if base_dir else Path(
        art.get("_meta", {}).get("sources", {}).get("solutions_dir", "."))
    smap = {}
    for s in art.get("sources", []):
        p = REPO / s["path"]
        if not p.is_file():
            p = base / Path(s["path"]).name
        smap[s["path"]] = p
    return smap


def _read_by_rel(solutions_dir: Path, rel: str) -> str:
    """按 rel 回读源文件：先试仓库相对路径，再退回 solutions_dir 下的同名文件。"""
    p = REPO / rel
    if not p.is_file():
        p = solutions_dir / Path(rel).name
    return p.read_text(encoding="utf-8")


def judge_content_drift(art: dict, live: dict) -> list:
    bad = []
    for fam in FAMILIES:
        a, b = art.get("harvest", {}).get(fam), live["harvest"][fam]
        if not isinstance(a, list) or len(a) != len(b):
            bad.append(f"{fam}: 条目数 {len(a) if isinstance(a, list) else 'N/A'} ≠ 现算 {len(b)}")
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            for f in TEXT_FIELDS[fam]:
                if x.get(f) != y.get(f):
                    bad.append(f"{fam}[{i}].{f}: {x.get(f)!r} ≠ 现算 {y.get(f)!r}")
    if bad:  # <<GUARD:J0A>>
        return bad
    return []


def judge_source_fingerprint(art: dict, live: dict) -> list:
    bad = []
    la = {s["path"]: s for s in art.get("sources", [])}
    lb = {s["path"]: s for s in live["sources"]}
    if set(la) != set(lb):
        bad.append(f"源文件集合不一致：{sorted(set(la) ^ set(lb))}")
    for k in sorted(set(la) & set(lb)):
        for f in ("sha256", "bytes", "lines", "kind"):
            if la[k].get(f) != lb[k].get(f):
                bad.append(f"sources[{k}].{f}: {la[k].get(f)!r} ≠ 现算 {lb[k].get(f)!r}")
    if bad:  # <<GUARD:J0B>>
        return bad
    return []


def judge_join(art: dict, graph: dict) -> list:
    """落格 join 完整性 —— **凡写进产物的 id 都必须能在图谱里 join 到**。

    ⚠️ 这条判据的守卫行是 `<<GUARD:J1>>`；把它改成 `if False` ⇒ 伪造 id 也能过（selftest 变异 3）。
    """
    bad = []
    l3_by_name = {x["name"]: x for x in graph["l3"]}
    role_ids = {r["id"] for r in graph["roles"]}
    flow_ids = {c["flow_id"] for c in graph["cells"]}
    stage_ids = {c["stage_id"] for c in graph["cells"]}
    cell_ids = {c["cell_id"] for c in graph["cells"]}
    for e in art["landing"]["by_page"]:
        for x in e["l3"]:
            n = x["name"]
            if n not in l3_by_name:
                bad.append(f"{e['page']}: l3 {n!r} 不在图谱 151 名单内")
                continue
            if x["role_id"] != l3_by_name[n]["role_id"]:
                bad.append(f"{e['page']}: l3 {n!r} 记的 role_id {x['role_id']} "
                           f"≠ 图谱归属 {l3_by_name[n]['role_id']}")
            if x["role_id"] not in role_ids:
                bad.append(f"{e['page']}: role_id {x['role_id']} 不在图谱 50 岗位内")
        for r in e["role_ids"]:
            if r not in role_ids:
                bad.append(f"{e['page']}: role_id {r} 不在图谱 50 岗位内")
        for f in e["flow_ids"]:
            if f not in flow_ids:
                bad.append(f"{e['page']}: flow_id {f} 不在图谱 8 条 FLOW 内")
        for c in e["m_cells"]:
            if c not in cell_ids:
                bad.append(f"{e['page']}: cell_id {c} 不在图谱 64 格内")
    for x in art["landing"]["by_m_cell"]:
        if x["cell_id"] not in cell_ids:
            bad.append(f"by_m_cell: {x['cell_id']} 不在图谱 64 格内")
        if x["flow_id"] not in flow_ids:
            bad.append(f"by_m_cell: {x['cell_id']} 的 flow_id {x['flow_id']} 不在图谱内")
        if x["stage_id"] not in stage_ids:
            bad.append(f"by_m_cell: {x['cell_id']} 的 stg_id {x['stage_id']} 不在图谱内")
        for n in x["l3"]:
            if n not in l3_by_name:
                bad.append(f"by_m_cell: {x['cell_id']} 的 l3 {n!r} 不在图谱 151 名单内")
    if bad:  # <<GUARD:J1>>
        return bad
    return []


def judge_anchor(art: dict) -> list:
    bad = []
    smap = load_sources(art)
    line_counts = {s["path"]: s["lines"] for s in art.get("sources", [])}
    for fam in FAMILIES:
        for i, it in enumerate(art["harvest"][fam]):
            src, ln, le, anc = it.get("source"), it.get("line"), it.get("line_end"), it.get("anchor")
            if src not in smap:
                bad.append(f"{fam}[{i}]: source {src!r} 不在 sources 列表里")
                continue
            if not isinstance(ln, int) or not isinstance(le, int):
                bad.append(f"{fam}[{i}]: line/line_end 不是整数")
                continue
            if ln < 1 or le < ln or le > line_counts.get(src, 0):
                bad.append(f"{fam}[{i}]: 行区间 [{ln},{le}] 越界（{src} 共 {line_counts.get(src)} 行）")
            if not anc or not re.match(r"^[a-zA-Z][\w.\-> ]*\[\d+\]$", anc):
                bad.append(f"{fam}[{i}]: anchor 形状不合法：{anc!r}")
    if bad:  # <<GUARD:J2>>
        return bad
    return []


def anchor_verbatim_hits(art: dict, live: dict, base_dir=None) -> tuple:
    """把**现场重算**的字，回查到**产物自己申报的** [line, line_end] 里。

    三者的分工必须互斥，否则任一篡改会被多条判据同时抓住，变异测试就分不出谁在承重：
    · `J0a` 管「产物里的字 == 源里的字」（**内容**被改）；
    · `J2` 管「行区间在文件范围内、anchor 形状合法」（**区间非法**）；
    · `J3`（本函数）管「产物指的那几行里，真的有源里的那段字」（**指错行**）。

    故：区间非法时**跳过**（那是 J2 的活）；比对基准是 live 的字段，不是产物的字段
    （否则内容篡改会同时打红 J0a 与 J3 —— 首版实测就是这三条扭在一起）。
    """
    lv = {(fam, it.get("source"), it.get("index")): it
          for fam in FAMILIES for it in live["harvest"][fam]}
    smap = load_sources(art, base_dir)
    ok, bad = 0, []
    for fam in FAMILIES:
        for i, it in enumerate(art["harvest"][fam]):
            ln, le, src = it.get("line"), it.get("line_end"), it.get("source")
            if not isinstance(ln, int) or not isinstance(le, int) or ln < 1 or le < ln:
                continue                      # 非法区间归 J2
            lp, fp = lv.get((fam, src, it.get("index"))), smap.get(src)
            if lp is None or fp is None:
                bad.append(f"{fam}[{i}]: 现场重算或源文件取不到（{src}）")
                continue
            lines = fp.read_text(encoding="utf-8").split("\n")
            if le > len(lines):
                continue                      # 越界也归 J2
            seg_norm = norm("\n".join(lines[ln - 1:le]))
            miss = [f for f in TEXT_FIELDS[fam]
                    if norm(lp.get(f) or "") and norm(lp[f]) not in seg_norm]
            if miss:
                bad.append(f"{fam}[{i}] @{Path(src).name}:{ln}-{le} 现场字段 {miss} "
                           f"不在申报的行区间内")
            else:
                ok += 1
    return ok, bad


def judge_anchor_verbatim(art: dict, live: dict) -> list:
    """逐条：区间合法者，现场重算的字必须落在产物申报的区间里。

    ⚠️ 本判据**不比对计数**（`counts.anchor_verbatim_ok` 归 `J5`）：
    首版在这里也比计数，于是「把 line 改成越界值」（J2 的活）会让本判据的计数同样对不上
    ⇒ 两条判据抢同一个篡改样本，变异 M2 的变体恒红、变异测试失效。
    """
    _, bad = anchor_verbatim_hits(art, live)
    if bad:  # <<GUARD:J3>>
        return bad
    return []


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}"
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def _walk_strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def judge_discard(art: dict, live: dict) -> list:
    """丢弃登记完好 + 泄漏扫描（**反向控制**：丢弃族不得以任何形式出现在 harvest 里）。"""
    bad = []
    reg = art.get("discarded", {}).get("card_position_index")
    if not isinstance(reg, dict):
        return ["discarded.card_position_index 缺失 —— 丢弃未登记（= 数字上看起来像「本来就没那么多东西」）"]
    for f in ("n_entries", "n_unique_card_ids", "pages_with_index",
              "n_entries_joined", "n_unique_joined", "n_unique_joined_curated_146"):
        if reg.get(f) != live["discarded"]["card_position_index"][f]:
            bad.append(f"discarded.card_position_index.{f}: {reg.get(f)!r} ≠ 现算 "
                       f"{live['discarded']['card_position_index'][f]!r}")
    chips = {c["card_id"] for p in live["harvest"]["_chips"] for c in p}
    labels = {c["label"] for p in live["harvest"]["_chips"] for c in p}
    # ⚠️ 禁键扫描必须遍历**字典键**：首版走 `_walk_strings`（只吐字符串值），
    # 于是 `harvest.card_index` 这个键名根本就不出现在扫描流里 ⇒ 注入卡位索引照样 exit 0。
    for k in _walk_keys(art.get("harvest", {})):
        if k.rsplit(".", 1)[-1] in FORBIDDEN_HARVEST_KEYS:
            bad.append(f"harvest 里出现禁键 {k} —— 卡位索引被收割进素材区")
    for k, v in _walk_strings(art.get("harvest", {})):
        if v in chips:
            bad.append(f"harvest{k} 的值是卡位 id {v!r} —— 卡位索引泄漏进素材区")
        elif v in labels:
            bad.append(f"harvest{k} 的值是卡位标签 {v!r} —— 卡位索引泄漏进素材区")
    if bad:  # <<GUARD:J4>>
        return bad
    return []


def judge_counts(art: dict, live: dict) -> list:
    bad = []
    for k in J5_KEYS:
        if art["counts"].get(k) != live["counts"][k]:
            bad.append(f"counts.{k}: {art['counts'].get(k)!r} ≠ 现算 {live['counts'][k]!r}")
    if bad:  # <<GUARD:J5>>
        return bad
    return []


def read_md_numbers(md_path: Path):
    if not md_path.is_file():
        return None, "报告文件不存在"
    txt = md_path.read_text(encoding="utf-8")
    m = re.search(re.escape(NUMBERS_BEGIN) + r"\n(.*?)\n" + re.escape(NUMBERS_END), txt, re.S)
    if not m:
        return None, "报告里找不到数字块"
    try:
        return json.loads(m.group(1)), None
    except Exception as e:  # noqa: BLE001
        return None, f"报告数字块不是合法 JSON：{e}"


def judge_report_numbers(art: dict, live: dict, md_path: Path) -> list:
    """报告内嵌数字块必须等于**现场重算的真值**（不是产物里那份，产物本身可能被改坏）。

    ⚠️ 首版拿产物当基准 ⇒ `counts.*` 的任何篡改都会**同时**触发本判据与对应判据，
    变异测试因此分不出谁在承重（实测 M5/M10/M11 三条同时失败）。
    """
    nums, err = read_md_numbers(md_path)
    if err:
        return [f"{md_path.name}: {err}"]
    bad = []
    for k in sorted(live["counts"]):
        if nums.get(k) != live["counts"][k]:
            bad.append(f"报告数字块 {k}: {nums.get(k)!r} ≠ 现算 {live['counts'][k]!r}")
    if bad:  # <<GUARD:J6>>
        return bad
    return []


def judge_partition(art: dict, live: dict) -> list:
    pages_live = {p["file"] for p in live["pages"]}
    got = [e["page"] for e in art["landing"]["by_page"]]
    got += art["landing"]["missing"]["pages_without_any_join"]
    dropped = sorted(pages_live - set(got))
    extra = sorted(set(got) - pages_live)
    dup = sorted({x for x in got if got.count(x) > 1})
    bad = []
    if dropped:
        bad.append(f"页面被静默丢弃：{dropped}")
    if extra:
        bad.append(f"页面凭空出现：{extra}")
    if dup:
        bad.append(f"页面重复入账：{dup}")
    if len(got) != len(pages_live):
        bad.append(f"划分不闭合：by_page+未落格 = {len(got)} ≠ 源页数 {len(pages_live)}")
    if bad:  # <<GUARD:J7>>
        return bad
    return []


def judge_unjoined(art: dict, live: dict) -> list:
    a = sorted(art["landing"]["missing"]["unjoined_chips"])
    b = sorted(live["landing"]["missing"]["unjoined_chips"])
    bad = []
    if a != b:
        bad.append(f"未 join 卡清单：{a} ≠ 现算 {b}")
    for e in art["landing"]["by_page"]:
        if e["chips_unique_joined_classification"] + len(e["chips_unjoined"]) != e["chips_unique"]:
            bad.append(f"{e['page']}: join 数 + 未 join 数 ≠ 唯一卡数")
        if e["chips_entries_joined"] + len(
                [c for c in e.get("chips_unjoined", [])]) > e["chips_entries"] + 1:
            bad.append(f"{e['page']}: 条目级 join 数超过条目数")
    total_unjoined = len({c for e in art["landing"]["by_page"] for c in e["chips_unjoined"]})
    if total_unjoined != art["counts"]["chips_unique_unjoined"]:
        bad.append(f"counts.chips_unique_unjoined {art['counts']['chips_unique_unjoined']} "
                   f"≠ 逐页登记 {total_unjoined}")
    # ⚠️ 与 `uniqueness` 的交叉核对**一律对现场真值**，不对产物自己的 `counts`：
    # 首版对 `counts.chips_entries_total`，于是「改 counts 里一个数」（J5 的活）
    # 会被本判据一并抓住，变异 M5 的变体恒红、变异测试失效。
    uq, uql = art["landing"].get("uniqueness", {}), live["landing"]["uniqueness"]
    for k in ("chips_entries_total", "chips_unique_total",
              "chips_unique_joined_classification", "chips_unique_joined_curated_146"):
        if uq.get(k) != uql.get(k):
            bad.append(f"landing.uniqueness.{k}: {uq.get(k)!r} ≠ 现算 {uql.get(k)!r}")
    if len(uq.get("chips_unique_unjoined") or []) != len(
            art["landing"]["missing"]["unjoined_chips"]):
        bad.append("landing.uniqueness.chips_unique_unjoined 与 missing.unjoined_chips 长度不符")
    if bad:  # <<GUARD:J8>>
        return bad
    return []


def judge_mcells(art: dict, graph: dict) -> list:
    bad = []
    want = sorted(c["cell_id"] for c in graph["cells"] if c["cell_kind"] == "M")
    got = sorted(x["cell_id"] for x in art["landing"]["by_m_cell"])
    if got != want:
        bad.append(f"M 格全集不一致：缺 {sorted(set(want) - set(got))} / 多 {sorted(set(got) - set(want))}")
    for x in art["landing"]["by_m_cell"]:
        if x["n_pages"] != len(x["pages"]):
            bad.append(f"{x['cell_id']}: n_pages {x['n_pages']} ≠ len(pages) {len(x['pages'])}")
        if x["n_l3"] != len(x["l3"]):
            bad.append(f"{x['cell_id']}: n_l3 {x['n_l3']} ≠ len(l3) {len(x['l3'])}")
    if bad:  # <<GUARD:J9>>
        return bad
    return []


def judge_conditions(art: dict, live: dict) -> list:
    bad = []
    for k in ("roadmap_phases_with_labeled_entry", "roadmap_phases_with_labeled_exit",
              "roadmap_phases_with_outcome_field"):
        if art["counts"].get(k) != live["counts"][k]:
            bad.append(f"counts.{k}: {art['counts'].get(k)!r} ≠ 现算 {live['counts'][k]!r}")
    if bad:  # <<GUARD:J10>>
        return bad
    return []


def judge_boilerplate(art: dict, live: dict, boiler: str, bp_titles: list) -> list:
    """样板不得产生任何收割条目。

    三重：(a) 语料公共骨架的**限定作用域**解析必须 0 条；(b) 声明值必须等于现算值；
    (c) 样板标题不得成为任何收割条目的名字。
    (a) 在真语料上是**结构性成立**的（样板里没有完整的三元组），故 (a) 的**可证伪性**由
    selftest 的构造样本负责（见 selftest 反向控制 #3）：去掉作用域限定，构造样本立刻产出条目。
    """
    bad = []
    if live["counts"]["boilerplate_items_scoped"] != 0:
        bad.append(f"样板骨架被解析出 {live['counts']['boilerplate_items_scoped']} 条 —— 收的是样板不是正文")
    for k in ("boilerplate_items_scoped", "boilerplate_items_unscoped",
              "boilerplate_common_lines", "boilerplate_h2_titles"):
        if art["counts"].get(k) != live["counts"][k]:
            bad.append(f"counts.{k}: {art['counts'].get(k)!r} ≠ 现算 {live['counts'][k]!r}")
    bset = set(bp_titles)
    for fam in FAMILIES:
        for i, it in enumerate(art["harvest"][fam]):
            for f in TEXT_FIELDS[fam]:
                v = it.get(f)
                if isinstance(v, str) and v and v in bset:
                    bad.append(f"{fam}[{i}].{f} 是样板标题 {v!r} —— 样板被当成了业务要素")
    if bad:  # <<GUARD:J11>>
        return bad
    return []


def judge_layers_depth(art: dict, live: dict) -> list:
    bad = []
    lv = {p["file"]: p for p in live["pages"]}
    for p in art["pages"]:
        if p["layers_declared_depth"] != p["n_layers"]:
            bad.append(f"{p['file']}: h2 声明 {p['layers_declared_depth']} 层，实收 {p['n_layers']} 层")
        q = lv.get(p["file"])
        if q is None:
            bad.append(f"{p['file']}: 源页不存在")
            continue
        for f in ("layers_declared_depth", "n_layers", "n_roadmap", "n_traps", "n_chips", "sha256"):
            if p.get(f) != q.get(f):
                bad.append(f"{p['file']}.{f}: {p.get(f)!r} ≠ 现算 {q.get(f)!r}")
    if bad:  # <<GUARD:J12>>
        return bad
    return []


def judge_by_l3(art: dict, live: dict) -> list:
    bad = []
    a = {x["l3"]: x for x in art["landing"]["by_l3"]}
    b = {x["l3"]: x for x in live["landing"]["by_l3"]}
    if set(a) != set(b):
        bad.append(f"by_l3 覆盖不全：缺 {sorted(set(b) - set(a))[:5]} / 多 {sorted(set(a) - set(b))[:5]}")
    for k in sorted(set(a) & set(b)):
        if a[k]["n_pages"] != len(a[k]["pages"]):
            bad.append(f"by_l3[{k}]: n_pages {a[k]['n_pages']} ≠ len(pages) {len(a[k]['pages'])}")
        if a[k]["n_pages"] != b[k]["n_pages"]:
            bad.append(f"by_l3[{k}].n_pages: {a[k]['n_pages']} ≠ 现算 {b[k]['n_pages']}")
        if a[k]["role_id"] != b[k]["role_id"]:
            bad.append(f"by_l3[{k}].role_id: {a[k]['role_id']} ≠ 现算 {b[k]['role_id']}")
    if bad:  # <<GUARD:J13>>
        return bad
    return []


def run_judgements(art: dict, live: dict, graph: dict, md_path: Path,
                   boiler: str, bp_titles: list) -> list:
    """跑全部判据，返回 (编号, 名称, 红条目) 列表。"""
    out = [
        ("J0a", "逐字内容漂移", judge_content_drift(art, live)),
        ("J0b", "源指纹漂移", judge_source_fingerprint(art, live)),
        ("J1", "落格 join 完整性", judge_join(art, graph)),
        ("J2", "锚点合法性", judge_anchor(art)),
        ("J3", "逐字回指（锚点指的地方真有这段字）", judge_anchor_verbatim(art, live)),
        ("J4", "丢弃登记 + 泄漏扫描", judge_discard(art, live)),
        ("J5", "计数现算一致", judge_counts(art, live)),
        ("J6", "报告数字同步", judge_report_numbers(art, live, md_path)),
        ("J7", "页面划分完整", judge_partition(art, live)),
        ("J8", "未 join 不静默", judge_unjoined(art, live)),
        ("J9", "M 格自洽", judge_mcells(art, graph)),
        ("J10", "进入/退出条件现算", judge_conditions(art, live)),
        ("J11", "样板不产生条目", judge_boilerplate(art, live, boiler, bp_titles)),
        ("J12", "层数与 h2 声明自洽", judge_layers_depth(art, live)),
        ("J13", "L3 落点自洽且满 151", judge_by_l3(art, live)),
    ]
    return out


# ─────────────────────────── 报告 ───────────────────────────

def render_md(art: dict, live: dict) -> str:
    c = art["counts"]
    L = []

    def A(s=""):
        L.append(s)

    A("# PHASE6 · S2 · 旧方案页收割（19 页 → 64 格体系）")
    A()
    A("> **本文件由 `paper2skills-research/scripts/harvest_legacy_solutions.py --run` 生成，手改会被 `--check` 判红。**")
    A("> 每个数字都由脚本现场算出；报告内嵌数字块必须等于**现场重算的真值**（判据 J6）——"
      "注意比对基准是现场真值，**不是产物里那一份**（产物本身可能已被改坏）。")
    A()
    A(NUMBERS_BEGIN)
    A(json.dumps(art["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    A(NUMBERS_END)
    A()

    A("## 1 结论")
    A()
    A(f"- 扫到 **{c['source_files_scanned']}** 个 HTML（{c['solution_pages']} 个方案页 + "
      f"{c['index_pages']} 个 index），收割 **{c['harvest_entries_total']}** 条 M 格素材：")
    A(f"  分层架构 **{c['layer_rows']}** 条 / 路线图 **{c['roadmap_phases']}** 条 / "
      f"架构陷阱 **{c['trap_rows']}** 条。")
    A(f"- **丢弃**：19 页的「核心 Skill 索引」= **{c['chips_entries_total']}** 条卡位"
      f"（{c['chips_unique_total']} 个唯一卡 id），显式登记在 `discarded.card_position_index`。")
    A(f"- **落格**（全派生）：**{c['chips_unique_joined']}/{c['chips_unique_total']}** 个唯一卡 id"
      f"（条目级 {c['chips_entries_joined']}/{c['chips_entries_total']}）能 join 到"
      f"产品侧 `classification.json`，覆盖 **{c['l3_landed']}/{c['l3_total']}** 个 L3、"
      f"**{c['roles_landed']}** 个岗位、**{c['flows_landed']}**/8 条 FLOW、"
      f"**{c['m_cells_with_any_page']}/{c['m_cells_total']}** 个 M 格。")
    A(f"- ⚠️ **M 格覆盖 {c['m_cells_with_any_page']}/{c['m_cells_total']} 这个数字没有区分度**："
      f"每个 M 格被 {c['m_cell_min_pages']}–{c['m_cell_max_pages']} 个方案页可达。"
      f"有区分度的是 **L3 层**（{c['l3_landed']}/{c['l3_total']}）。")
    A()

    A("## 2 复跑")
    A()
    A("```bash")
    A("python3 paper2skills-research/scripts/harvest_legacy_solutions.py --run       # 重建产物 + 本报告")
    A("python3 paper2skills-research/scripts/harvest_legacy_solutions.py --check     # 现场重算比对（已有产物不算数）")
    A("python3 paper2skills-research/scripts/harvest_legacy_solutions.py --selftest  # 判据表 + 篡改样本 + 变异测试 + 端到端")
    A("```")
    A()
    A("退出码：**0 全过 / 1 判据红 / 2 输入没拿到（≠ 通过）/ 3 脚本内部错误（≠ 判红）**")
    A()

    A(f"## 3 判据表（{c['judgements_declared']} 条判据 · {c['tamper_samples_declared']} 份篡改样本 · "
      f"{c['mutations_declared']} 份变异体）")
    A()
    A("「打红样本」= 把产物改坏一处；「变异体」= 把该判据的守卫行改成 `if False:`。")
    A("**判定式是成对的**：原版对篡改样本必须 `exit 1`，变异体对**同一份**样本必须 `exit 0` ——")
    A("只有同时成立，才证明「这条判据在承重」，而不是「恰好被别的判据兜住了」。")
    A()
    A("| 判据 | 这条标定了什么 | 打红样本 | 变异体 | 期望 原版/变体 |")
    A("|---|---|---|---|---|")
    muts = {m[2]: m for m in MUTATIONS}
    for jid, name, tamper in JUDGEMENTS:
        keys = {"J0a": "J0a", "J0b": "J0b", "J1": "J1", "J2": "J2", "J3": "J3", "J4": "J4",
                "J5": "J5", "J6": "J6", "J7": "J7", "J8": "J8", "J9": "J9",
                "J10": "J10", "J11": "J11", "J12": "J12", "J13": "J13"}.get(jid)
        mids = "、".join(m[0] for m in MUTATIONS if m[2] == keys)
        exp = "1 / 0"
        A(f"| `{jid}` | {name} | {tamper} | {mids or '—'} | {exp} |")
    A(f"| `MS` | （仪器）解析作用域限定 | 不动产物，改脚本 | MS | 0 / 1（反向） |")
    A()
    A("另有 **6 条端到端用例**（`subprocess` 跑真 CLI + 夹具，**不 import 库函数**："
      "干净夹具 `--run`/`--check` 各一条 + 四种「输入没拿到 ⇒ exit 2」）与 **3 条反向控制**"
      "（样板不产生条目 · 诱饵对照 · 进入/退出探测器能命中构造样本），全部由 `--selftest` 现场执行。")
    A("复跑 `--selftest` 的收尾会打印两行读数：`31 项用例 · 17/17 变异抓住` 与 "
      "`scripts/ 目录跑前跑后无新增文件`。")
    A()

    A("## 4 收割内容（每条都可逐字回指）")
    A()
    A("锚点四元组 = `source`（文件）+ `line`/`line_end`（行区间）+ `anchor`（可定位锚点）+ "
      "`verbatim`（逐字文本字段）。判据 `J0a` 保证内容与源逐字一致，`J2` 保证锚点合法。")
    A()
    A("### 4.1 分层架构（每层职责）")
    A()
    A("⚠️ **只有三元组**：层号 / 层名 / 描述。源 HTML 的 `.sd-layer` **没有边界字段、没有层间关系字段**"
      "（描述里 0 处边界语）。本报告不替它补。")
    A()
    A("| 页 | 层 | 层名 | 描述（逐字） | 锚点 |")
    A("|---|---|---|---|---|")
    for it in art["harvest"]["layers"]:
        f = Path(it["source"]).name
        A(f"| `{f}` | {it['layer_no']} | {it['name']} | {it['desc']} | L{it['line']} {it['anchor']} |")
    A()
    A("### 4.2 路线图（阶段划分）")
    A()
    A(f"⚠️ **进入条件 / 退出条件在源里没有独立字段**：{c['roadmap_phases']} 个 Phase 只有"
      f" 阶段名 / 时长 / 动作 / 预期产出。带显式进入条件标记的 = "
      f"**{c['roadmap_phases_with_labeled_entry']}** 条，显式退出条件标记 = "
      f"**{c['roadmap_phases_with_labeled_exit']}** 条（这两个数是**算出来的**，不是「我没找」）。"
      f"产物用 `outcome`（预期产出，{c['roadmap_phases_with_outcome_field']} 条）如实承载退出证据。")
    A()
    A("| 页 | 序 | 阶段 | 时长 | 动作（逐字） | 预期产出（逐字） | 行 |")
    A("|---|---|---|---|---|---|---|")
    for it in art["harvest"]["roadmap"]:
        f = Path(it["source"]).name
        A(f"| `{f}` | {it['index']} | {it['phase_name']} | {it['duration']} | "
          f"{it['action']} | {it['outcome']} | L{it['line']} |")
    A()
    A("### 4.3 架构陷阱")
    A()
    A(f"⚠️ 源页面标题写「三大架构陷阱」但 `sol-counterfactual-pricing.html` 的陷阱段是**空的**："
      f"实收 {c['trap_rows']} = **{c['trap_pages']} 页 × 3**，不是 19 × 3。不替它补。")
    A()
    A("| 页 | 序 | 陷阱 | 说明全文（逐字，含「为什么」与「怎么避免」） | 行 |")
    A("|---|---|---|---|---|")
    for it in art["harvest"]["traps"]:
        f = Path(it["source"]).name
        A(f"| `{f}` | {it['trap_no']} | {it['title']} | {it['body']} | L{it['line']} |")
    A()

    A("## 5 丢弃登记")
    A()
    d = art["discarded"]["card_position_index"]
    A(f"- **丢了什么**：{d['what']}")
    A(f"- **多少条**：**{d['n_entries']}** 条卡位（{d['n_unique_card_ids']} 个唯一卡 id），"
      f"分布在 {d['pages_with_index']} 个方案页。")
    A(f"- **能 join 多少**：条目级 **{d['n_entries_joined']}/{d['n_entries']}**；"
      f"唯一级 **{d['n_unique_joined']}/{d['n_unique_card_ids']}**；"
      f"其中能 join 到**精选线 146 张**的只有 **{d['n_unique_joined_curated_146']}** 个。")
    A("- **为什么丢**：")
    for w in d["why_discarded"]:
        A(f"  - {w}")
    A(f"- **丢弃但保留连接**：{d['join_key_note']}")
    A(f"  - join 到精选线 146 张的只有 **{d['n_unique_joined_curated_146']}** 个"
      f"（{d['n_unique_joined_curated_146']}/{d['n_unique_card_ids']} = "
      f"{100.0 * d['n_unique_joined_curated_146'] / max(d['n_unique_card_ids'], 1):.1f}%）"
      f"—— **这就是落格必须走装线卡而不是精选卡的原因**。")
    A(f"  - 未 join：`{d['unjoined']}`（**登记不猜**）。")
    A(f"- **反向控制**：判据 `J4` 扫 `harvest` 任意深度，卡位 id / 卡位标签 / 禁键名出现即判红"
      f"（篡改样本见第 3 节）。")
    A()

    A("## 6 落格（全派生，未另造分类）")
    A()
    A("```")
    A(art["landing"]["join_chain"])
    A("```")
    A()
    A(f"- 口径写死：{art['landing']['stg_note']}")
    A(f"- 未落格页 **{c['pages_without_any_join']}** 个"
      f"（名单：`{art['landing']['missing']['pages_without_any_join'] or '（无）'}`）——"
      f"「一个卡 id 都 join 不到」的页**有素材但无落点**，登记不兜底，也不塞进「未分类」。")
    A()
    A("### 6.1 M 格供给（24 格）")
    A()
    A("⚠️ **读这张表要小心**：`涉及 L3 数` 是「所有可达该格的方案页的 L3 并集」，")
    A("所以它跟 `可达页数` 一样**没有区分度**（一张页贡献它全部 9–18 个 L3）。")
    A("有区分度的是 §6.2 的**逐 L3 页数**。这两列如实列出，是为了让「没有区分度」这件事本身可复核。")
    A()
    A("| 格 | 阶段 | 可达页数 | 涉及 L3 数 |")
    A("|---|---|---|---|")
    for x in art["landing"]["by_m_cell"]:
        A(f"| `{x['cell_id']}` | {x['stage_name']} | {x['n_pages']} | {x['n_l3']} |")
    A()
    A("### 6.2 L3 供给（151 个，只列有页的）")
    A()
    A("| L3 | 岗位 | A/B/C | 页数 |")
    A("|---|---|---|---|")
    for x in art["landing"]["by_l3"]:
        if x["n_pages"]:
            A(f"| {x['l3']} | {x['role_id']} | {x['serviceability']} | {x['n_pages']} |")
    A()
    A(f"**未被任何旧方案页触及的 L3：{c['l3_total'] - c['l3_landed']} 个**（这就是 S2 的真实供给图）。")
    A()

    A("## 7 源缺口登记（不编造）")
    A()
    A("| 缺口 | 证据 | 处置 |")
    A("|---|---|---|")
    for g in art["source_gaps"]:
        A(f"| {g['gap']} | {g['evidence']} | {g['handling']} |")
    A()

    A("## 8 这一轮没做的事")
    A()
    A("| 族 | 条数 | 为什么没做 |")
    A("|---|---|---|")
    for x in art["not_harvested_this_round"]:
        A(f"| {x['family']} | {x['n']} | {x['why']} |")
    A()
    A("- **未改任何既有事实源**：`capability-graph.json` / `gap-ledger.json` / `contracts/**` / 任何 Skill 卡"
      "一个字都没动（本轮有并发写入者，动它们会造成写入冲突）。")
    A("- **未接入 `run_phase6_gates.py`**：新门禁没有挂到统一入口上（见第 9 节工单）。")
    A("- **未做语义对齐**：旧方案页与 8 条 FLOW 之间**没有**做人工语义映射 —— 那会造出第四套分类的第五个版本。")
    A("- **未处置 2 个 join 不到的卡 id**：只登记，未追查它们在装线语料里到底叫什么。")
    A("- **未收割 hero / ROI 横幅 / 摘要**（见本节表）。")
    A("- **未做「旧方案页 → 具体 STG」的判定**：源页面没有阶段化字段，任何把某个陷阱/某层挂到 "
      "STG-04 还是 STG-05 的动作都只能是新的语义判断 —— 本轮**只给到 FLOW 级闭包**。")
    A("- **未核验 `classification.json` 的 L3 归属本身对不对**：本轮把它当既有事实源只读 join，"
      "没有回头审它的 1327 条分类（那是 F5 的地盘）。")
    A()

    A("## 9 本轮实测撞出的**仪器缺陷**（判据抓到的，不是事后总结的）")
    A()
    A("这一节是本交付物最该留的部分：**每一条都是判据先报红，才被发现**。")
    A()
    A("| # | 缺陷 | 谁抓住的 | 后果（如果不修） |")
    A("|---|---|---|---|")
    A("| I1 | **`_pos` 是切片内下标，却拿去对全文算行号** | `J2` 首跑报 6 条 `[42,41]` 越界 | "
      "218 条锚点的行号整体前移，最大那批落进**侧边栏**（39–42 行）—— "
      "「把样板当正文」的坐标版；行号全错而条目内容全对，抽查极难发现 |")
    A("| I2 | **`strip_tags` 写成 `<[^>]+>`，而语料里有裸的小于号** | `J3` 首跑报 6 条「字段不在申报区间内」 | "
      "`接受率<40%时…`、`季节MAPE稳定在<12%` 被整段当标签吃掉。字段抽取侥幸没受影响"
      "（切片在 `</p>` 前截断），**行区间回查**却被截成半句 ⇒ 逐字回指这条判据会静默失效 |")
    A("| I3 | **锚点区间取「下一块起点 −1 / 文件末」** | 人工复查发现 `L102–L285` | 每族最后一条的区间一路铺到文件末，"
      "「可定位的锚点」名存实亡 |")
    A("| I4 | **`REPO` 由 `__file__` 推，而变异体跑在 tmp 里** | 变异矩阵首跑 12 项失败 | "
      "变异体的源文件相对路径整体变形 ⇒ 「只差一行」的前提不成立，变异测试当场失去意义 |")
    A("| I5 | **多条判据抢同一份篡改样本** | 变异 M2/M5/M8/M10/M11 变体恒红 | "
      "四组「判据 ∩ 判据」重叠（J0a∩J3、J5∩J6、J5∩J8、J2∩J3）与一条「夹具上空操作」的篡改样本 ⇒ "
      "变异分数虚高而实际分不出谁在承重。**这正是台账 #25 的同族形态** |")
    A("| I6 | **`counts` 被当成报告基准** | 同上 | 报告数字块本应与**现场真值**比，与产物比会把「产物被改坏」"
      "错报成「报告过期」 |")
    A()

    A("## 10 下游工单")
    A()
    A("0. 🔴 **`paper2skills-research/data/legacy-solutions.json` 当前被 `.gitignore` 挡住**"
      "（规则 `.gitignore:93` 的 `paper2skills-research/data/**/*.json`）——"
      "**不补 `!` 例外，这份产物就不会进 commit**。本轮**没有改 `.gitignore`**："
      "任务硬约束只许新增 3 个文件，且该文件正被并发写入者持续追加入库例外"
      "（`backlog-l3-map.json` / `backlog-routing.json` / `material-citations.json` / "
      "`search-queries.json` 都是同一形态），改它会造成写入冲突。")
    A("   - 建议加的行（理由同 `gap-ledger.json`：**它是一份账，不是中间态** —— "
      "218 条素材各带逐字锚点，落格链路跨仓库依赖产品侧 `classification.json`）：")
    A("     ```gitignore")
    A("     # S2 旧方案页收割产物（PHASE6 S2）：它是一份账，不是中间态 ——")
    A("     # 逐条记录 19 个方案页的分层架构/路线图/陷阱及其行级锚点，落格走跨仓库 join。")
    A("     # 判据=`scripts/harvest_legacy_solutions.py --check`。")
    A("     !paper2skills-research/data/legacy-solutions.json")
    A("     ```")
    A("   - 核验命令：`git check-ignore -v paper2skills-research/data/legacy-solutions.json`"
      "（补前 exit 0 且有输出 = 被忽略；补后应无输出）。")
    A("1. **台账编号**：#54–#57 已由 S1-W/W4 预留（材料引文核验族），故本轮的仪器缺陷建议编号 "
      "**#58–#63**（对应 I1–I6）。本轮**未改台账文件**（并发写入者正在改契约与卡片 frontmatter）。")
    A("2. **把本脚本接进 `run_phase6_gates.py`**（本轮不许改共享文件，故留作工单）；"
      "接的时候注意：`run_phase6_gates.py` 的汇总纪律是**退出码不许合并成分数**，"
      "而本脚本有 0/1/2/3 四态，`2` 必须与 `1` 分开报。")
    A("3. **2 个 join 不到的卡 id**：`Skill-Cross-Domain-Orthogonal-Signals` / "
      "`Skill-Uplift-Cannibalization-Modeling` —— 在 1338 装线语料里查无此 id，需确认是历史重命名还是从未上线。")
    A("4. **`sol-counterfactual-pricing.html` 的空陷阱段**：源页面标题声称「三大架构陷阱」而内容为空，"
      "需决定是补写还是降标题。")
    A("5. **路线图缺进入/退出条件**（72 条 Phase 全缺）：若要它们成为 M 格素材，需要在源侧补字段，"
      "而不是在收割侧猜。")
    A()

    A("## 11 口径与已知残留")
    A()
    A(f"- **「L1–L6 架构」是个不准确的转述**：实测每页 **{c['layer_depth_declared_min']}–"
      f"{c['layer_depth_declared_max']}** 层，最高到 **L7**；层号直方图 "
      f"`{json.dumps(c['layer_no_histogram'], ensure_ascii=False)}`。任务书里的「L1–L6」"
      f"应理解为「L1 起的分层架构」而非固定六层。")
    A(f"- **「194 卡位索引」复算成立**：实测 `.sd-skill-chip` 锚点 = **{c['chips_entries_total']}** 条"
      f"（{c['chips_unique_total']} 唯一；{c['chips_entries_joined']}/{c['chips_unique_joined']} 可 join），"
      f"与 194 一致。")
    A(f"- ⚠️ **两个分母不许混**：`entries`（194，含 13 个跨页重复）与 `unique`（181）是两个单位；"
      f"首版把条目级 join 数除以唯一数打过一次「192/181」的假账 —— 现两者分列。")
    A(f"- **样板反向控制**：语料公共骨架 {c['boilerplate_common_lines']} 行 / 样板 h2 标题 "
      f"{c['boilerplate_h2_titles']} 个；限定作用域解析产出 **{c['boilerplate_items_scoped']}** 条。"
      f"⚠️ 这个 0 在**真语料**上是**结构性成立**的（样板里没有完整的「层号+层名+描述」三元组），"
      f"所以它的可证伪性由 `--selftest` 的**构造样本**负责：把作用域限定去掉，构造样本立刻产出条目。")
    A(f"- **`--selftest` 不留临时文件**：全部写 `tempfile.mkdtemp()`，`finally` 删除，"
      f"并断言 `scripts/` 目录跑前跑后无新增文件。")
    A()
    return "\n".join(L)


# ─────────────────────────── CLI ───────────────────────────

def _prepare(args):
    sol = Path(args.solutions_dir).expanduser().resolve() if args.solutions_dir else SOLUTIONS_DIR_DEFAULT
    graph = Path(args.graph).expanduser().resolve() if args.graph else GRAPH_DEFAULT
    if not graph.is_file():
        raise InputMissing(f"图谱不存在：{graph}")
    classi = resolve_classification(args.classification)
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else OUT_JSON_DEFAULT
    out_md = Path(args.out_md).expanduser().resolve() if args.out_md else OUT_MD_DEFAULT
    return sol, graph, classi, out_json, out_md






def cmd_run(args) -> int:
    sol, graph, classi, out_json, out_md = _prepare(args)
    live = build(sol, graph_path=graph, classi_path=classi)
    live_g, boiler, bp_titles = _live_ids(sol, graph, classi)

    art = json.loads(json.dumps(live))
    # J4 需要卡位明细（**只在内存里给判据看，不落产物** —— 产物里出现卡位就是泄漏）
    live.setdefault("harvest", {})["_chips"] = _chips_of(sol, graph, classi)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_text = render_md(art, live)
    out_json.write_text(json.dumps(art, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")

    graph_obj = json.loads(graph.read_text(encoding="utf-8"))
    res = run_judgements(art, live, graph_obj, out_md, boiler, bp_titles)
    reds = [(j, n, b) for j, n, b in res if b]
    print(f"✓ 产物：{out_json}")
    print(f"✓ 报告：{out_md}")
    print(f"  收割 {art['counts']['harvest_entries_total']} 条 · 丢弃登记 "
          f"{art['counts']['chips_entries_total']} 条卡位 · 落格 {art['counts']['l3_landed']}/"
          f"{art['counts']['l3_total']} L3 / {art['counts']['m_cells_with_any_page']}/"
          f"{art['counts']['m_cells_total']} M 格")
    return _report(res, reds)


def _live_ids(sol, graph, classi):
    """重算「样板骨架」与「样板 h2 标题」—— 与 `build()` 内同一套函数，供 J11 现场比对。"""
    paths = sorted(p for p in sol.glob("*.html") if p.name != "index.html")
    texts = [p.read_text(encoding="utf-8") for p in paths]
    boiler = boilerplate_of(texts)
    per = [parse_page(p.name, p.read_text(encoding="utf-8"), True) for p in paths]
    return None, boiler, boilerplate_h2_titles(per)


def cmd_check(args) -> int:
    sol, graph, classi, out_json, out_md = _prepare(args)
    if not out_json.is_file():
        print(f"✗ 产物不存在：{out_json}（先跑 --run）")
        return 2
    art = json.loads(out_json.read_text(encoding="utf-8"))

    # 现场重算 —— 已有产物一律不算数
    live = build(sol, graph_path=graph, classi_path=classi)
    chk = _live_ids(sol, graph, classi)
    boiler, bp_titles = chk[1], chk[2]
    graph_obj = json.loads(graph.read_text(encoding="utf-8"))

    # J4 需要卡位明细（只在现场算，不落产物）
    live2 = json.loads(json.dumps(live))
    live2.setdefault("harvest", {})["_chips"] = _chips_of(sol, graph, classi)
    art2 = json.loads(json.dumps(art))
    res = run_judgements(art2, live2, graph_obj, out_md, boiler, bp_titles)
    reds = [(j, n, b) for j, n, b in res if b]
    c = live["counts"]
    print(f"· 现场重算：{c['solution_pages']} 页 / 收割 {c['harvest_entries_total']} 条 / "
          f"丢弃 {c['chips_entries_total']} 条卡位 / 落格 L3 {c['l3_landed']}/{c['l3_total']} · "
          f"M 格 {c['m_cells_with_any_page']}/{c['m_cells_total']}")
    print(f"· 样板反向控制：公共骨架 {c['boilerplate_common_lines']} 行 → 限定作用域收 "
          f"{c['boilerplate_items_scoped']} 条")
    return _report(res, reds)


def _chips_of(sol, graph, classi) -> list:
    out = []
    paths = sorted(p for p in sol.glob("*.html") if p.name != "index.html")
    for p in paths:
        out.append(parse_page(p.name, p.read_text(encoding="utf-8"), True)["chips"])
    return out


def _report(res: list, reds: list) -> int:
    print()
    for jid, name, bad in res:
        mark = "✅" if not bad else "🔴"
        print(f"  {mark} {jid} {name}" + (f" —— {len(bad)} 条" if bad else ""))
        for b in bad[:8]:
            print(f"       · {b}")
        if len(bad) > 8:
            print(f"       · …另 {len(bad) - 8} 条")
    if reds:
        print(f"\n🔴 判红 {len(reds)} 条判据 —— exit 1")
        return 1
    print("\n✅ 全部判据通过 —— exit 0")
    return 0


# ─────────────────────────── selftest ───────────────────────────

#: 变异表：把判据的守卫行整行换成 `if False:`，同一份篡改样本必须**由红转绿**。
#: 每条 = (编号, 被替换的整行, 替换成, 篡改样本编号, pristine 期望, mutant 期望)
#: 守卫行的标记。**不写字面量**：写成字面量会让同一串在脚本里出现两次（表里一次、守卫处一次），
#: 变异替换无法唯一定位 —— 本脚本首版就撞了这条，`assert src.count(line) == 1` 当场报出。
#: 现在由 `_marked_lines()` 从脚本自身扫出来。
GUARD_MARK_RE = re.compile(r"#\s*<<(GUARD:[A-Z0-9]+|SCOPE)>>\s*$")


def marked_lines(src: str | None = None) -> dict:
    src = src if src is not None else Path(__file__).read_text(encoding="utf-8")
    out = {}
    for ln in src.split("\n"):
        m = GUARD_MARK_RE.search(ln)
        if not m:
            continue
        key = m.group(1).split(":")[-1]
        if key in out:
            raise InternalError(f"守卫标记 <<{m.group(1)}>> 在脚本里出现多次，变异无法唯一定位")
        out[key] = ln
    return out


MUTATIONS = [
    ("M0a", "J0A", "J0a", 1, 0),
    ("M0b", "J0B", "J0b", 1, 0),
    ("M1", "J1", "J1", 1, 0),
    ("M2", "J2", "J2", 1, 0),
    ("M3", "J3", "J3", 1, 0),
    ("M4", "J4", "J4", 1, 0),
    ("M4b", "J4", "J4b", 1, 0),
    ("M5", "J5", "J5", 1, 0),
    ("M6", "J6", "J6", 1, 0),
    ("M7", "J7", "J7", 1, 0),
    ("M8", "J8", "J8", 1, 0),
    ("M9", "J9", "J9", 1, 0),
    ("M10", "J10", "J10", 1, 0),
    ("M11", "J11", "J11", 1, 0),
    ("M12", "J12", "J12", 1, 0),
    ("M13", "J13", "J13", 1, 0),
    ("MS", "SCOPE", "SCOPE", 0, 1),   # 去掉作用域限定 ⇒ 与已建产物漂移 ⇒ 由绿转红
]


def _tamper(art: dict, md_text: str, which: str):
    """返回 (art, md_text)。每次只改坏一处。"""
    import copy
    a = copy.deepcopy(art)
    md = md_text
    if which == "J0a":
        a["harvest"]["traps"][0]["title"] = a["harvest"]["traps"][0]["title"] + "（篡改）"
    elif which == "J0b":
        a["sources"][0]["sha256"] = "0" * 16
    elif which == "J1":
        a["landing"]["by_page"][0]["l3"][0]["role_id"] = "AGT-999"
    elif which == "J2":
        a["harvest"]["layers"][0]["line"] = 999999
    elif which == "J3":
        a["harvest"]["layers"][0]["line_end"] = a["harvest"]["layers"][0]["line"]
    elif which == "J4":
        a["harvest"]["card_index"] = [{"page": a["landing"]["by_page"][0]["page"],
                                       "card_id": "Skill-3D-Bin-Packing-Optimization",
                                       "label": "3D Bin Packing Optimization"}]
    elif which == "J4b":
        a["harvest"]["layers"][0]["note"] = "Skill-Fixture-Alpha"
    elif which == "J5":
        a["counts"]["chips_entries_total"] = a["counts"]["chips_entries_total"] - 1
    elif which == "J6":
        nums, err = read_md_numbers_from_text(md)
        nums["chips_entries_total"] = nums["chips_entries_total"] - 1
        md = _replace_md_numbers(md, nums)
    elif which == "J7":
        a["landing"]["by_page"].pop(0)
    elif which == "J8":
        a["landing"]["missing"]["unjoined_chips"] = []
    elif which == "J9":
        a["landing"]["by_m_cell"][0]["n_pages"] = 99
    elif which == "J10":
        a["counts"]["roadmap_phases_with_labeled_entry"] = 3
    elif which == "J11":
        a["counts"]["boilerplate_items_scoped"] = 5
    elif which == "J12":
        a["pages"][0]["layers_declared_depth"] = 99
    elif which == "J13":
        a["landing"]["by_l3"].pop(0)
    elif which == "SCOPE":
        pass
    else:
        raise InternalError(f"未知篡改样本 {which}")
    return a, md


def read_md_numbers_from_text(txt: str):
    m = re.search(re.escape(NUMBERS_BEGIN) + r"\n(.*?)\n" + re.escape(NUMBERS_END), txt, re.S)
    if not m:
        raise InternalError("数字块不存在")
    return json.loads(m.group(1)), None


def _replace_md_numbers(txt: str, nums: dict) -> str:
    new = json.dumps(nums, ensure_ascii=False, indent=2, sort_keys=True)
    return re.sub(re.escape(NUMBERS_BEGIN) + r"\n.*?\n" + re.escape(NUMBERS_END),
                  NUMBERS_BEGIN + "\n" + new + "\n" + NUMBERS_END, txt, flags=re.S)


# ── 夹具 ──

FIXTURE_GRAPH = {
    "l3": [
        {"name": "需求预测", "role_id": "AGT-101", "plane_id": "PLN-OPS",
         "domain_id": "DOM-03", "serviceability": "A", "boundary_a": False},
        {"name": "库存分层", "role_id": "AGT-102", "plane_id": "PLN-OPS",
         "domain_id": "DOM-03", "serviceability": "B", "boundary_a": False},
        {"name": "价格监测", "role_id": "AGT-103", "plane_id": "PLN-OPS",
         "domain_id": "DOM-01", "serviceability": "A", "boundary_a": False},
    ],
    "roles": [
        {"id": "AGT-101", "title": "需求计划", "flows": ["FLOW-01", "FLOW-03"]},
        {"id": "AGT-102", "title": "库存管理", "flows": ["FLOW-03"]},
        {"id": "AGT-103", "title": "定价运营", "flows": ["FLOW-01"]},
    ],
    "cells": [],
    "cards": [{"name": "Skill-Fixture-Alpha", "l3": ["需求预测"]}],
}
for _f in ("FLOW-01", "FLOW-03"):
    for _s in range(1, 9):
        FIXTURE_GRAPH["cells"].append({
            "cell_id": f"{_f}/STG-0{_s}", "flow_id": _f, "stage_id": f"STG-0{_s}",
            "stage_name": f"阶段{_s}", "cell_kind": "M" if _s in (4, 5) else "D",
            "business_step": f"步骤{_s}",
        })

FIXTURE_CLASSIFICATION = {
    "items": [
        {"id": "Skill-Fixture-Alpha", "l3": ["需求预测"]},
        {"id": "Skill-Fixture-Beta", "l3": ["库存分层"]},
        {"id": "Skill-Fixture-Gamma", "l3": ["价格监测"]},
    ]
}

FIXTURE_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title>
<style>.sd-layer{{display:flex}}.sd-trap{{display:flex}}.sd-skill-chip{{font-size:11px}}</style>
</head><body>
<header class="topbar"><a class="brand" href="../index.html">paper2skills</a></header>
<main class="layout">
  <aside class="sidebar" id="sidebar">
    <div class="sb-section"><p class="sb-label">决策工具</p>
      <div class="sb-links"><a href="../index.html"><span class="sbl-text">总览</span></a></div></div>
  </aside>
  <section class="content">
<div class="sd-hero"><h1>{title}</h1><p class="sd-subtitle">{title} 副标题</p>
  <span class="sd-category">夹具域</span><span class="sd-updated">更新于 2026-01-01</span></div>
<div class="sd-roi-banner"><span>[↑]</span> 夹具 ROI</div>
<p class="sd-summary">夹具摘要。</p>
<div class="sd-grid-2"><div class="sd-section"><h2>{depth}层架构设计</h2>
  <div class="sd-layers">
<div class="sd-layer">
  <div class="sd-layer-no" style="color:#000">L1</div>
  <div class="sd-layer-body"><strong>{title} 第一层</strong><span>夹具第一层职责</span></div>
</div>
<div class="sd-layer">
  <div class="sd-layer-no" style="color:#000">L2</div>
  <div class="sd-layer-body"><strong>{title} 第二层</strong><span>夹具第二层职责</span></div>
</div>
  </div></div>
<div class="sd-section"><h2>分阶段实施路线图</h2>
  <div class="sd-phases">
<div class="sd-phase">
  <div class="sd-phase-header" style="border-left:3px solid #000">
    <span class="sd-phase-name" style="color:#000">Phase 0 {title} 准备</span>
    <span class="sd-phase-dur">1-2 周</span>
  </div>
  <div class="sd-phase-action">夹具动作</div>
  <div class="sd-phase-roi">夹具产出</div>
</div>
  </div></div></div>
<div class="sd-section sd-full"><h2>[!] 三大架构陷阱</h2>
  <div class="sd-traps">
<div class="sd-trap"><span class="sd-trap-no">①</span><div><strong>{title} 陷阱一</strong>
  <p>夹具陷阱说明一</p></div></div>
<div class="sd-trap"><span class="sd-trap-no">②</span><div><strong>{title} 陷阱二</strong>
  <p>夹具陷阱说明二</p></div></div>
<div class="sd-trap"><span class="sd-trap-no">③</span><div><strong>{title} 陷阱三</strong>
  <p>夹具陷阱说明三</p></div></div>
  </div></div>
<div class="sd-section sd-full"><h2>核心 Skill 索引（2 个）</h2>
  <div class="sd-skills"><a class="sd-skill-chip" href="../skills/{card}.html">{card}</a><a class="sd-skill-chip" href="../skills/Skill-Fixture-Missing.html">Skill-Fixture-Missing</a></div>
</div>
<style>.sd-hero{{display:flex}}</style>
</section></main>
</body></html>
"""

FIXTURE_INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>方案库</title></head>
<body><h1>方案库</h1><a href="sol-fix-a.html">A</a><a href="sol-fix-b.html">B</a></body></html>
"""

DECOY_SIDEBAR = """
  <aside class="sidebar" id="sidebar">
    <div class="sd-layer">
      <div class="sd-layer-no" style="color:#000">L9</div>
      <div class="sd-layer-body"><strong>侧边栏诱饵层</strong><span>这行在样板区里</span></div>
    </div>
    <div class="sd-traps"><div class="sd-trap"><span class="sd-trap-no">⑨</span><div>
      <strong>侧边栏诱饵陷阱</strong><p>这行在样板区里</p></div></div></div>
  </aside>
"""

#: 反向控制的**构造样本**：诱饵在样板区（侧边栏），正文区是空的。
#: 必须带 `section.content` —— 否则 `content_slice` 找不到切片点、两种作用域返回同一段文本，
#: 「限定/不限定」的对照就恒等（首版实测 scoped=2 / unscoped=2）。
DECOY_DOC = ('<main class="layout">' + DECOY_SIDEBAR +
             '<section class="content"><p>夹具正文里没有可收割的结构</p></section></main>')


def _mk_fixture(root: Path, decoy: bool = False) -> dict:
    sol = root / "solutions"
    sol.mkdir(parents=True, exist_ok=True)
    pages = [("sol-fix-a.html", "夹具甲", 2, "Skill-Fixture-Alpha"),
             ("sol-fix-b.html", "夹具乙", 2, "Skill-Fixture-Beta")]
    for name, title, depth, card in pages:
        txt = FIXTURE_PAGE.format(title=title, depth=depth, card=card)
        if decoy:
            txt = txt.replace('<aside class="sidebar" id="sidebar">',
                              DECOY_SIDEBAR.strip())
        (sol / name).write_text(txt, encoding="utf-8")
    (sol / "index.html").write_text(FIXTURE_INDEX, encoding="utf-8")
    g = root / "graph.json"
    g.write_text(json.dumps(FIXTURE_GRAPH, ensure_ascii=False), encoding="utf-8")
    c = root / "classification.json"
    c.write_text(json.dumps(FIXTURE_CLASSIFICATION, ensure_ascii=False), encoding="utf-8")
    return {"sol": sol, "graph": g, "classi": c, "root": root}


def _run_cli(script: Path, mode: str, fx: dict, out_json: Path, out_md: Path):
    env = dict(os.environ, P2S_REPO_ROOT=str(REPO))
    cmd = [sys.executable, str(script), mode,
           "--solutions-dir", str(fx["sol"]), "--graph", str(fx["graph"]),
           "--classification", str(fx["classi"]),
           "--out-json", str(out_json), "--out-md", str(out_md)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def cmd_selftest(args) -> int:
    scripts_before = set(p.name for p in Path(__file__).parent.iterdir())
    src = Path(__file__).read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="harvest_selftest_"))
    results, failures = [], []
    try:
        # ── 用例 0：脚本自身变异表可定位（否则变异测试是摆设） ──
        guards = marked_lines(src)
        need = {m[1] for m in MUTATIONS}
        for tag in sorted(need):
            line = guards.get(tag)
            if line is None:
                failures.append(f"[定位] 守卫标记 <<{tag}>> 在脚本里找不到")
            elif src.count(line) != 1:
                failures.append(f"[定位] 守卫行 {tag} 在脚本里出现 {src.count(line)} 次（应为 1）")
        results.append(("定位", f"变异表 {len(need)} 条守卫全部唯一可定位", not failures))

        # ── 变体脚本 ──
        variants = {}
        for mid, tag, _t, _p, _m in MUTATIONS:
            line = guards[tag]
            if tag == "SCOPE":
                new = line.replace("True", "False") + " MUTATED"
            else:
                new = line.replace("if bad:", "if False:") + " MUTATED"
            if src.count(line) != 1:
                raise InternalError(f"变异目标 {tag} 不唯一（{src.count(line)} 次）")
            vp = tmp / f"mutant_{mid}.py"
            mutated = src.replace(line, new)
            if mutated == src:
                raise InternalError(f"变异 {mid} 没有改变源码 —— 变异没施上力")
            vp.write_text(mutated, encoding="utf-8")
            variants[mid] = vp

        # ── 端到端：真 CLI + 夹具 ──
        fx = _mk_fixture(tmp / "fx")
        oj, om = tmp / "fx_out.json", tmp / "fx_out.md"
        rc, out = _run_cli(Path(__file__), "--run", fx, oj, om)
        results.append(("E2E-1", "真 CLI --run + 夹具 → exit 0", rc == 0))
        if rc != 0:
            failures.append(f"[E2E-1] exit={rc}\n{out[-2500:]}")
        rc2, out2 = _run_cli(Path(__file__), "--check", fx, oj, om)
        results.append(("E2E-2", "真 CLI --check 干净夹具 → exit 0", rc2 == 0))
        if rc2 != 0:
            failures.append(f"[E2E-2] exit={rc2}\n{out2[-2500:]}")

        # ── 端到端：输入没拿到 ⇒ exit 2 ──
        for label, mutate in (
            ("目录不存在", lambda f: {**f, "sol": f["root"] / "nope"}),
            ("空目录", lambda f: {**f, "sol": (f["root"] / "empty")}),
            ("图缺失", lambda f: {**f, "graph": f["root"] / "nope.json"}),
            ("分类缺失", lambda f: {**f, "classi": f["root"] / "nope.json"}),
        ):
            f2 = dict(fx)
            if label == "空目录":
                (f2["root"] / "empty").mkdir(exist_ok=True)
            f2 = mutate(f2)
            rc3, out3 = _run_cli(Path(__file__), "--run", f2, tmp / "x.json", tmp / "x.md")
            ok = rc3 == 2
            results.append(("E2E-2x", f"输入没拿到（{label}）→ exit 2", ok))
            if not ok:
                failures.append(f"[E2E-2x {label}] exit={rc3}（应 2）\n{out3[-1200:]}")

        # ── 反向控制 #1：样板骨架不得产生条目（真语料） ──
        real_sol = SOLUTIONS_DIR_DEFAULT
        texts = [p.read_text(encoding="utf-8") for p in sorted(real_sol.glob("sol-*.html"))]
        boiler = boilerplate_of(texts)
        n_bp = count_items(boiler, True)
        results.append(("RC-1", f"真语料公共骨架 {boiler.count(chr(10)) + 1} 行 → 收割 {n_bp} 条（须 0）",
                        n_bp == 0))
        if n_bp != 0:
            failures.append(f"[RC-1] 样板骨架产出 {n_bp} 条")
        # 正向对照：同一份骨架里确实有诱饵（CSS 类名），不限定作用域就会误收 —— 用构造样本证明
        decoy_doc = DECOY_DOC
        n_decoy_scoped = count_items(decoy_doc, True)
        n_decoy_unscoped = count_items(decoy_doc, False)
        ok = (n_decoy_scoped == 0 and n_decoy_unscoped > 0)
        results.append(("RC-1b", f"构造诱饵：限定作用域 {n_decoy_scoped} 条 / 不限定 "
                                  f"{n_decoy_unscoped} 条（须 0 与 >0）", ok))
        if not ok:
            failures.append(f"[RC-1b] scoped={n_decoy_scoped} unscoped={n_decoy_unscoped}")

        # ── 反向控制 #2：进入/退出条件探测器能命中构造样本 ──
        probe = "Phase 0 准备 需具备 进入条件：数据齐备 退出条件：产出通过评审"
        hit_e = any(k in probe for k in ENTRY_MARKERS)
        hit_x = any(k in probe for k in EXIT_MARKERS)
        real_e = sum(1 for it in _all_phases(real_sol)
                     if any(k in it["phase_name"] + it["action"] + it["outcome"] + it["duration"]
                            for k in ENTRY_MARKERS))
        results.append(("RC-2", f"进入/退出探测器：构造样本命中 {hit_e}/{hit_x}，真语料 {real_e} 条",
                        hit_e and hit_x and real_e == 0))
        if not (hit_e and hit_x and real_e == 0):
            failures.append(f"[RC-2] hit_e={hit_e} hit_x={hit_x} real_e={real_e}")

        # ── 篡改 + 变异矩阵 ──
        base_json = json.loads(oj.read_text(encoding="utf-8"))
        base_md = om.read_text(encoding="utf-8")

        # 先验证：每一份篡改样本在**原版**下都必须被判红（exit 1）
        tamper_rc = {}
        for which in [m[2] for m in MUTATIONS if m[2] != "SCOPE"]:
            aj, am = _tamper(base_json, base_md, which)
            oj.write_text(json.dumps(aj, ensure_ascii=False), encoding="utf-8")
            om.write_text(am, encoding="utf-8")
            rc, _ = _run_cli(Path(__file__), "--check", fx, oj, om)
            tamper_rc[which] = rc
            if rc != 1:
                failures.append(f"[篡改 {which}] 原版 --check exit={rc}（应 1）")
        oj.write_text(json.dumps(base_json, ensure_ascii=False), encoding="utf-8")
        om.write_text(base_md, encoding="utf-8")
        ok_all = all(v == 1 for v in tamper_rc.values())
        results.append(("篡改", f"{len(tamper_rc)} 份篡改样本在原版下全部判红（exit 1）", ok_all))

        # 变异矩阵：反向（去掉作用域限定）单独处理
        rev = [m for m in MUTATIONS if m[1] == "SCOPE"]
        fwd = [m for m in MUTATIONS if m[1] != "SCOPE"]
        caught = 0
        for mid, tag, which, exp_p, exp_m in fwd:
            aj, am = _tamper(base_json, base_md, which)
            oj.write_text(json.dumps(aj, ensure_ascii=False), encoding="utf-8")
            om.write_text(am, encoding="utf-8")
            rc_p, _ = _run_cli(Path(__file__), "--check", fx, oj, om)
            rc_m, out_m = _run_cli(variants[mid], "--check", fx, oj, om)
            ok = (rc_p == exp_p and rc_m == exp_m)
            caught += ok
            results.append((f"变异 {mid}", f"守卫 {tag} 改坏：原版 exit {rc_p}（应 {exp_p}）/ "
                                           f"变体 exit {rc_m}（应 {exp_m}）", ok))
            if not ok:
                failures.append(f"[变异 {mid}] 守卫 {tag}: 原版 {rc_p} / 变体 {rc_m}（应 "
                                f"{exp_p}/{exp_m}）\n{out_m[-1200:]}")
        # 反向变异：去掉作用域限定后，用治具产物跑 --check ⇒ 重算与产物漂移 ⇒ 转红
        fx_decoy = _mk_fixture(tmp / "fx_decoy", decoy=True)
        dj, dm = tmp / "decoy.json", tmp / "decoy.md"
        rc_d, out_d = _run_cli(Path(__file__), "--run", fx_decoy, dj, dm)
        results.append(("E2E-3", "带侧边栏诱饵的夹具：原版 --run → exit 0（诱饵未被收）", rc_d == 0))
        if rc_d != 0:
            failures.append(f"[E2E-3] exit={rc_d}\n{out_d[-1500:]}")
        decoy_art = json.loads(dj.read_text(encoding="utf-8")) if rc_d == 0 else None
        if decoy_art:
            n = sum(1 for it in decoy_art["harvest"]["layers"] if "诱饵" in it["name"])
            ok = n == 0
            results.append(("RC-3", f"诱饵层未进产物（实收 {n} 条诱饵）", ok))
            if not ok:
                failures.append(f"[RC-3] 诱饵层被收 {n} 条")
        for mid, tag, which, exp_p, exp_m in rev:
            rc_p, _ = _run_cli(Path(__file__), "--check", fx_decoy, dj, dm)
            rc_m, out_m = _run_cli(variants[mid], "--check", fx_decoy, dj, dm)
            ok = (rc_p == exp_p and rc_m == exp_m)
            caught += ok
            results.append((f"变异 {mid}", f"作用域限定改坏：原版 exit {rc_p}（应 {exp_p}）/ "
                                           f"变体 exit {rc_m}（应 {exp_m}）", ok))
            if not ok:
                failures.append(f"[变异 {mid}] 作用域: 原版 {rc_p} / 变体 {rc_m}（应 {exp_p}/{exp_m}）\n"
                                f"{out_m[-1200:]}")

        # ── 真语料 --check 必须全绿（本交付物的实际状态） ──
        rp = subprocess.run([sys.executable, str(Path(__file__)), "--check"],
                            capture_output=True, text=True,
                            env=dict(os.environ, P2S_REPO_ROOT=str(REPO)))
        results.append(("真语料", f"--check 真语料 → exit {rp.returncode}", rp.returncode == 0))
        if rp.returncode != 0:
            failures.append(f"[真语料] exit={rp.returncode}\n{(rp.stdout + rp.stderr)[-2500:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 78)
    print("S2 旧方案页收割器 · selftest")
    print("=" * 78)
    for tag, desc, ok in results:
        print(f"  {'✅' if ok else '🔴'} [{tag}] {desc}")
    if failures:
        print("\n失败明细：")
        for f in failures:
            print(f"  · {f}")
        print(f"\n🔴 selftest 失败 {len(failures)} 项 —— exit 1")
        return 1

    scripts_after = set(p.name for p in Path(__file__).parent.iterdir())
    stray = sorted(scripts_after - scripts_before)
    if stray:
        print(f"\n🔴 scripts/ 下留下了临时文件：{stray} —— exit 1")
        return 1
    print(f"\n  ✅ [卫生] scripts/ 目录跑前跑后无新增文件（{len(scripts_before)} → {len(scripts_after)}）")
    n_mut = sum(1 for t, _, o in results if t.startswith("变异"))
    print(f"\n✅ selftest 全绿：{len(results)} 项用例 · {n_mut}/{n_mut} 变异抓住 —— exit 0")
    return 0


def _all_phases(sol_dir: Path) -> list:
    out = []
    for p in sorted(sol_dir.glob("sol-*.html")):
        out.extend(parse_page(p.name, p.read_text(encoding="utf-8"), True)["roadmap"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="S2 · 旧方案页收割（L1–Ln 架构 / 路线图 / 陷阱 → M 格素材）")
    ap.add_argument("--run", action="store_true", help="重建产物 JSON + 报告 MD")
    ap.add_argument("--check", action="store_true", help="现场重算并逐判据比对（已有产物不算数）")
    ap.add_argument("--selftest", action="store_true", help="判据表 + 篡改样本 + 变异测试 + 端到端")
    ap.add_argument("--solutions-dir", default=None)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--classification", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    modes = [m for m in ("run", "check", "selftest") if getattr(args, m)]
    if len(modes) != 1:
        ap.print_help()
        print("\n必须且只能指定 --run / --check / --selftest 之一")
        return 3

    try:
        if args.selftest:
            return cmd_selftest(args)
        if args.run:
            return cmd_run(args)
        return cmd_check(args)
    except InputMissing as e:
        print(f"✗ 输入没拿到（exit 2，**不是通过**）：{e}")
        return 2
    except InternalError as e:
        print(f"✗ 脚本内部错误（exit 3，**不是判红**）：{e}")
        return 3
    except FileNotFoundError as e:
        print(f"✗ 输入没拿到（exit 2，**不是通过**）：{e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
