#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE6 S5/B11 · p2s 换底：把产品侧八段预览页换成 vault 精选卡的完整卡底。

这条任务要接的断口
------------------
已装进产品的 1338 张 p2s 卡是**八段式预览页**（源站把代码截到 60 行，实测 457/1279
不可 `ast.parse`）；而 vault 精选线 146 张是**六段式完整卡**，过了 K1/K2 门禁、
有 2,087 条逐字引文。**实测：已安装的 1338 张里含 `> 原文:` 引用块的 = 0 条。**
换底就是把这个断口接上 —— 价值不在搬运文件，在**证据链第一次到达产品**。

三件事（Q6 已定口径）
--------------------
  · **93 张同 ID**：分类字段从旧 p2s 卡**继承** + 正文/溯源/代码从 vault 卡**取**
    + 全文落 `references/full-card.md`（复现已有 `references/implementation.py` 先例的形态）
  · **53 张 vault 独有**：按同规则新增
  · **1245 张**只有预览版：保持现状 + 标 `quality_tier: "preview"`

12 KB 是硬门禁，不是建议
------------------------
`MAX_SKILL_BYTES = 12*1024` 同时被 `assemble-skills.mjs` / `import-paper2skills.mjs` /
`verify-install.mjs` 三处强制。换底前 1338/1338 全过（实测 max 12211）；换底后必须**仍然全过**。
这条约束**就是**「全文必须走 `references/`」的原因，不是附带的工程细节。

而 vault 卡实测 p50 15155 字节、max 67355 —— **装不进** SKILL.md。所以本页必然是**投影**，
投影规则必须机械、可复算、且把「丢了什么」写清楚：

  1. **逐字引文一条不截断**（这是本次任务的载荷）。取完整卡里首尾 `> 原文:` 行之间的
     **连续区间**，整段原样内联 —— 它是完整卡的一个连续片段，可逐字节回查。
  2. **换底正文**取完整卡正文的**开头若干行**（逐行按原顺序，跳过代码围栏与引文区间），
     预算是剩余空间。这复现了既有 `references/implementation.py` 的形态
     （「卡面节选正是该文件的**开头**，逐行连续前缀」）。
  3. 旧 p2s 卡里 S2a 合成的四段（`输入 / 输出契约` / `执行步骤` / `边界与不做` / `技能关联`）
     能装下就**原样带过**；装不下就整体移入 `references/legacy-preview-card.md`。
  4. 旧卡**整份**逐字节留档到 `references/legacy-preview-card.md` —— 换底必须可回看、可回退。

由此**没有任何内容真的丢失**：SKILL.md 里没有的，都在同目录的两份 `references/` 里，
且路径与 sha256 都写进 frontmatter，`--check` 逐张核对。

源指纹（为什么必须做）
----------------------
vault 侧还有并发写入者（S10 写 `venue_tier`、S13 写 `l3_*`、其他会话改卡）。
若拷到一半源变了，交付的就是一份「不知道是哪一版」的快照 —— 那比失败更糟。
故：`--apply` 记下每张源卡的 sha256；`--check` **重算一遍**，凡漂移的逐条报出。

三件套纪律
----------
  `--check`   对**现场**重算（不读自己刚写的产物）
  `--selftest` 端到端跑真 CLI（不只测库函数）
  `--mutate`  变异测试，且**先证明变异改变了真实取值**

退出码：0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 门禁内部错误（≠ 判红）
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXIT_OK, EXIT_RED, EXIT_NO_INPUT, EXIT_INTERNAL = 0, 1, 2, 3

MAX_SKILL_BYTES = 12 * 1024

#: 给**装机后追加的 `p2s_*` 透传字段**留的余量。
#:
#: ⚠️ 实测撞出来的跨线耦合：12 KB 门禁是在**已装卡**上量的（`scripts/verify-install.mjs`），
#:    而装机链在 `import-paper2skills.mjs` 之后还会跑 `sync-card-facets.mjs --apply`，
#:    往 frontmatter 里追加 `p2s_code_level`（本包 145 张换底卡已自带另外四项 `p2s_*`，
#:    实测只多 42 字节；`p2s-monodense` 因取值短只多 28 字节）。
#:    ⇒ **staging ≤ 12288 推不出装机态 ≤ 12288**。首版没留余量、把 staging 顶到 12288，
#:    装机后 13 张卡变成 12290–12330，`verify-install` 当场判红。
#:    **48 = 实测装机追加上限 45 字节 + 3 字节余量。** 实测分布（1390 张，装机态 vs staging）：
#:    `+42` 1196 张 / `+28` 42 张 / `0` 33 张 / `+45` 5 张。⇒ 这是**观测上界 + 固定余量**，
#:    **不是算出来的**；`sync-card-facets.mjs` 若新增透传字段，这个数必须重测。
#:    权威判据仍是在**已装卡**上量的 `scripts/verify-install.mjs`；本常量只是 staging 侧的前置闸。
INSTALL_HEADROOM_BYTES = 48
def staging_budget() -> int:
    """换底自身的字节预算（staging 侧）。装机后仍须 ≤ MAX_SKILL_BYTES。

    ⚠️ 刻意写成**函数**而不是模块级派生常量：`--selftest` 用 `mock.patch` 改
    `MAX_SKILL_BYTES` 造预算夹具，模块级常量只在 import 时算一次、**不跟着变**
    ⇒ 三条预算用例当场变假（实测 9b/9c 双双失败）。与「用被测常量构造夹具 = 自指」同族。
    """
    return MAX_SKILL_BYTES - INSTALL_HEADROOM_BYTES

DEFAULT_P2S = "../magpie-horch/packages/capabilities/dsh-paper2skills"
DEFAULT_VAULT = "paper2skills-vault"

REBASE_VERSION = "s5-b11-v1"

RE_QUOTE = re.compile(r"^>\s*原文\s*[:：]")
RE_FENCE = re.compile(r"^\s*```")
RE_H2 = re.compile(r"^##\s")
RE_FM = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.S)
RE_FM_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$")

#: 旧 p2s 卡里由 S2a 合成的四段（不是从预览页抄来的，是产品自己的适配产物）
SYNTH_HEADS = ("输入 / 输出契约", "输入/输出契约", "执行步骤", "边界与不做", "技能关联")

#: 完整卡里**永不内联**的段落标题形态（它们各自有专门的落点）
NEVER_INLINE_H2 = re.compile(r"^##\s*(?:③|3[.、)]|三、)\s*代\s*码")


class NoInput(Exception):
    """输入没拿到 —— 转 exit 2。**不是「通过」。**"""


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def split_frontmatter(text: str) -> tuple[str, str]:
    """返回 (frontmatter 块含分隔线, 正文)。没有 frontmatter 时前者为空串。"""
    m = RE_FM.match(text)
    return (m.group(0), text[m.end():]) if m else ("", text)


def fm_fields(text: str) -> dict:
    block, _ = split_frontmatter(text)
    out = {}
    for line in block.split("\n"):
        m = RE_FM_LINE.match(line)
        if m:
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] == '"':
                v = v[1:-1]
            out[m.group(1)] = v
    return out


def jstr(v: str) -> str:
    """产品侧 frontmatter 的取值一律 JSON 引号化（import 校验强制）。"""
    return json.dumps(str(v), ensure_ascii=False)


def q(v: str) -> str:
    return '"' + str(v).replace('"', "'") + '"'


# --------------------------------------------------------------------------- #
# 完整卡的切片（全部按**行号**记录，便于 --check 逐字节复算）
# --------------------------------------------------------------------------- #
def fenced_line_set(lines: list[str]) -> set[int]:
    """围栏代码块占据的行号集合（含 ```` ``` ```` 两端的围栏行）。整块跳过，不留半截围栏。"""
    out, inside = set(), False
    for i, l in enumerate(lines):
        if RE_FENCE.match(l):
            out.add(i)
            inside = not inside
            continue
        if inside:
            out.add(i)
    return out


def quote_span(lines: list[str]) -> tuple[int, int] | None:
    """首尾 `> 原文:` 行之间的**连续**区间（首尾都是 `> 原文:` 行本身）。没有则 None。"""
    idx = [i for i, l in enumerate(lines) if RE_QUOTE.match(l)]
    return (idx[0], idx[-1]) if idx else None


def quote_units(lines):
    """把引文区间切成**整条**单位：每条从 `> 原文:` 行起，到下一个 `> 原文:` 行之前。

    为什么必须按「整条」取、不能按「行」取：按行取会在一条引文中间断开，读者看到的是半句 ——
    这正是 G2 漏洞 #15「引文被截成残句」的形态，而**残句仍然逐字存在于底本**，门禁看不出来。
    """
    starts = [i for i, l in enumerate(lines) if RE_QUOTE.match(l)]
    out = []
    for j, st in enumerate(starts):
        en = starts[j + 1] - 1 if j + 1 < len(starts) else None
        out.append((st, en))
    return out


def synth_block(body: str) -> tuple[str, int]:
    """切出旧卡里 S2a 合成的四段（正文里 `## <标题>` 起的整段）。返回 (文本, 段数)。"""
    lines = body.split("\n")
    keep, got, i = [], 0, 0
    while i < len(lines):
        if lines[i].startswith("## ") and any(
                lines[i][3:].strip().startswith(h) for h in SYNTH_HEADS):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("## "):
                j += 1
            keep.extend(lines[i:j])
            got += 1
            i = j
            continue
        i += 1
    return "\n".join(keep).strip("\n"), got


def old_code_pointer(old_text: str) -> tuple[str, str]:
    """从旧卡取 ⑦/⑧ 两段的**元信息**（不搬运旧文案）。返回 (⑦段, ⑧段) 原文。"""
    i7 = old_text.find("## ⑦")
    i8 = old_text.find("## ⑧", i7 + 1) if i7 >= 0 else -1
    s7 = old_text[i7:i8] if i7 >= 0 and i8 > i7 else ""
    s8 = old_text[i8:] if i8 >= 0 else ""
    # ⑧ 到下一个 `## `（合成的契约段）为止
    if s8:
        k = s8.find("\n## ", 4)
        if k > 0:
            s8 = s8[:k]
    return s7.strip("\n"), s8.strip("\n")


# --------------------------------------------------------------------------- #
# 渲染新卡
# --------------------------------------------------------------------------- #
def render_seven(recovery: dict | None, legacy_rel: str | None) -> str:
    """⑦ 段。判据与 `verify-install.mjs::checkCodeSection` **同口径**：
    有完整实现 → 必须指向 `references/implementation.py`（本卡不内联代码）；
    没有 → 必须出现「未附代码」字样。
    """
    rec = recovery or {}
    has_full = rec.get("tier") not in (None, "unrecovered") and (rec.get("lines") or 0) > 0
    L = ["## ⑦ 代码节选", ""]
    if has_full:
        L.append(f"> **本节的完整实现**在同目录的 `references/implementation.py`（{rec['lines']} 行）。")
        L.append("> 换底后本卡正文**不再内联代码** —— SKILL.md 有 12 KB 硬上限，"
                 "内联会挤掉逐字引文；引文是本次换底要送达产品的证据链。")
        if rec.get("tier") == "unverified":
            L.append("> 该卡的完整实现**未经交叉核对**（卡面无节选可校验）。")
        if legacy_rel:
            L.append(f"> 换底前的八段预览页（含当时的代码节选）留在 `{legacy_rel}`，可逐字回看。")
    else:
        L.append("（**本卡未附代码伴生文件** —— 产品侧完整实现索引里没有本卡；"
                 "代码原文见 `references/full-card.md` 的「代码模板」段。）")
    L.append("")
    return "\n".join(L)


def render_eight(vfm: dict, old8: str, vault_stem: str, full_rel: str, nq: int) -> str:
    """⑧ 段：**溯源从 vault 卡取**（这正是换底的一部分）。旧预览页的出处声明原样留档。"""
    L = ["## ⑧ 论文来源", ""]
    L.append(f"**本卡溯源换底自精选线** `{vault_stem}`（完整卡：`{full_rel}`）。")
    got = []
    for k, label in (("paper_id", "论文"), ("paper", "标题"), ("venue", "发表处"),
                     ("venue_tier", "venue 档位"), ("evidence_grade", "证据等级"),
                     ("evidence_basis", "证据基础"), ("related", "关联卡")):
        if vfm.get(k):
            got.append(f"- {label}：{vfm[k]}")
    if got:
        L.append("")
        L.extend(got)
    else:
        L.append("")
        L.append("- 精选卡 frontmatter 未记 `paper_id` —— 本卡**无一手论文出处**（如实登记，不补编）。")
    L.append("")
    L.append(f"- 逐字引文：{nq} 条，全部内联于上方「原文引用」段；一条不截断。")
    if old8:
        L.append("")
        L.append("**换底前八段预览页的出处声明（保留备查，不构成本卡的溯源结论）**：")
        L.append("")
        L.append("\n".join("> " + x if x.strip() else ">" for x in old8.split("\n")))
    L.append("")
    return "\n".join(L)


def build_card(*, slug, old_text, old_fields, vault_bytes, vault_rel, vault_stem,
               facets, quality_tier, recovery, add_new, synth_extra=None) -> dict:
    """算出新 SKILL.md 的全文与 record。**不写盘。**"""
    full = vault_bytes.decode("utf-8")
    vfm = fm_fields(full)
    _, body = split_frontmatter(full)
    vlines = body.split("\n")
    fences = fenced_line_set(vlines)
    qs = quote_span(vlines)
    nq_total = sum(1 for l in vlines[qs[0]:qs[1] + 1] if RE_QUOTE.match(l)) if qs else 0
    nq = nq_total

    title = (old_fields.get("title") if old_fields else None) or vfm.get("title") or slug
    description = (old_fields.get("description") if old_fields else "") or \
        f"触发词：{slug}。{vfm.get('title', '')}".strip()

    full_rel = "references/full-card.md"
    legacy_rel = "references/legacy-preview-card.md" if (old_text and not add_new) else None

    # ---- frontmatter -------------------------------------------------------
    fm = []
    fm.append(("name", slug))
    fm.append(("title", title))
    fm.append(("description", description))
    for k in ("l1_id", "l1_plane", "l2_id", "l2_domain", "l3_id", "l3_business",
              "l3_all", "l1_l2_l3"):
        if facets.get(k):
            fm.append((k, facets[k]))
    fm.append(("quality_tier", quality_tier))
    fm.append(("p2s_card_id", vault_stem))
    src_dom = (old_fields.get("p2s_src_domain") if old_fields else "") or \
        (vault_rel.split("/")[1] if len(vault_rel.split("/")) > 1 else "")
    fm.append(("p2s_src_domain", src_dom))
    # ⚠️ 溯源字段用 **`p2s_*` 这套既有命名**，不自造第二套。
    #    并发写入者已按 `venue→p2s_venue / venue_tier→p2s_venue_tier /
    #    evidence_grade→p2s_evidence_grade / paper_id→p2s_paper_id`
    #    把这四项从精选线透传进已装卡（`scripts/sync-card-facets.mjs`；实测 93 张已带）。
    #    若这里另写一套不带前缀的 `venue_tier`/`paper_id`，同一个事实就有两个字段名；
    #    而 `import-paper2skills.mjs` 整份重写已装卡、只保留 4 个 keep-key
    #    ⇒ 装机会把他们那套**静默抹掉**。同名同值 ⇒ 装机无损。
    for vk, fk in (("venue", "p2s_venue"), ("venue_tier", "p2s_venue_tier"),
                   ("evidence_grade", "p2s_evidence_grade"), ("paper_id", "p2s_paper_id")):
        if vfm.get(vk):
            fm.append((fk, vfm[vk]))
    # 以下才是**本次换底新产生的事实**（八段预览页里没有），故一律带 `rebase_` 前缀，与既有命名不撞。
    fm.append(("rebase_version", REBASE_VERSION))
    fm.append(("rebase_vault_card", vault_stem))
    fm.append(("rebase_vault_path", vault_rel))
    fm.append(("rebase_source_sha256", sha256_bytes(vault_bytes)))
    fm.append(("rebase_full_card", full_rel))
    fm.append(("rebase_full_card_sha256", sha256_bytes(vault_bytes)))
    fm.append(("rebase_full_card_bytes", str(len(vault_bytes))))
    fm.append(("rebase_full_card_lines", str(len(vlines))))
    if legacy_rel:
        fm.append(("rebase_legacy_card", legacy_rel))
        fm.append(("rebase_legacy_card_sha256", sha256_bytes(old_text.encode("utf-8"))))
    if old_fields:
        for k in ("user_summary", "user_try", "whenToUse", "workflow"):
            if old_fields.get(k):
                fm.append((k, old_fields[k]))
    fm.append(("enabled", "true"))
    fm.append(("disable-model-invocation", "true"))
    fm.append(("user-invocable", "true"))
    _QUOTE_KEYS = ("rebase_evidence_quotes", "rebase_evidence_quotes_total",
                   "rebase_evidence_quotes_complete")

    def fm_render(nq_in):
        """frontmatter 必须在引文条数**定稿之后**才渲染。

        ⚠️ 首版把 `fm_text` 在收缩循环之前就定死了 ⇒ 46 张卡的自报条数是**收缩前**的值
        （实测读数：`frontmatter 说内联 32 条，实物 18 条`）。这是门禁自己抓出来的缺陷。
        """
        base = [(k, v) for k, v in fm if k not in _QUOTE_KEYS]
        base.append(("rebase_evidence_quotes", str(nq_in)))
        base.append(("rebase_evidence_quotes_total", str(nq_total)))
        base.append(("rebase_evidence_quotes_complete", "true" if nq_in == nq_total else "false"))
        return "---\n" + "\n".join(f"{k}: {jstr(v)}" for k, v in base) + "\n---\n"

    # ---- 固定骨架与预算 ----------------------------------------------------
    synth, _n_synth = (synth_block(split_frontmatter(old_text)[1]) if old_text else ("", 0))
    if synth_extra:
        synth = (synth + "\n\n" + synth_extra).strip()
    seven = render_seven(recovery, legacy_rel)
    legacy_sha = sha256_bytes(old_text.encode("utf-8")) if (old_text and legacy_rel) else ""
    new8 = render_eight(vfm, old_code_pointer(old_text)[1] if old_text else "", vault_stem,
                        full_rel, nq_total)
    fixed_tail = "\n" + seven + "\n" + new8

    def compose(ex_pairs, synth_used, qspan, nq_in, cut):
        """按定稿的 nq 渲染全文 —— header 里写着内联条数，必须在 nq 定稿后再生成。"""
        header = [
            f"# {title}",
            "",
            f"> **底本**：本卡正文换底自语料 vault 的精选线卡 `{vault_stem}`"
            f"（完整卡：`{full_rel}`，sha256 `{sha256_bytes(vault_bytes)}`，"
            f"{len(vault_bytes)} 字节 / {len(vlines)} 行 / {nq_total} 条逐字引文）。",
            f"> 完整卡的**全部**内容都在那份文件里。本页内联：完整卡正文的开头逐行摘录，"
            f"以及{'全部' if nq_in == nq_total else f'前 {nq_in} 条（共 {nq_total} 条）'}逐字引文。",
        ]
        if legacy_rel:
            header.append(f"> 被换掉的八段预览页逐字节留档在 `{legacy_rel}`（sha256 `{legacy_sha}`），"
                          f"可回看、可回退。")
        header.append("")
        quote_used = "\n".join(vlines[qspan[0]:qspan[1] + 1]) if qspan else ""
        o = io.StringIO()
        o.write(fm_render(nq_in))
        o.write("\n".join(header) + "\n")
        o.write("## 换底正文（完整卡逐字摘录）\n\n")
        if ex_pairs:
            o.write("\n".join(l for _, l in ex_pairs) + "\n")
            o.write(f"\n（**换底正文在此截断** —— 完整卡正文共 {len(vlines)} 行，"
                    f"本页内联到第 {ex_pairs[-1][0] + 1} 行；其余见 `{full_rel}`。\n")
            o.write("  代码围栏已整块移入 `references/implementation.py`，引文整段见下节。）\n")
        else:
            o.write(f"（本卡字节预算不足以内联正文摘录；完整卡正文 {len(vlines)} 行全部见 `{full_rel}`。）\n")
        if not quote_used and nq_total:
            # ⚠️ 首版在这里**整段消失**：预算不够时连 `## 原文引用` 标题都不打印，
            #    读者无从知道这张卡本来有引文 —— 静默丢失，正是本次要修的缺陷形态。
            o.write("\n## 原文引用（逐字 · 内联 0 / 全 %d 条 —— **全部见 `%s`**）\n\n"
                    % (nq_total, full_rel))
            o.write("> ⚠️ 本卡字节预算装不下任何一条引文；完整卡的全部 %d 条逐字引文见 `%s`。\n" % (nq_total, full_rel))
        elif quote_used:
            if nq_in < nq_total:
                o.write("\n## 原文引用（逐字 · 内联 %d / 全 %d 条 —— **其余 %d 条见 `%s`**）\n\n"
                        % (nq_in, nq_total, nq_total - nq_in, full_rel))
                o.write("> ⚠️ SKILL.md 有 12 KB 硬门禁，本卡装不下全部 %d 条逐字引文。"
                        "本页按完整卡顺序内联**前 %d 条整条引文**（不在引文中间断开）；"
                        "其余 %d 条逐字原文见 `%s` 的「原文引用」段。\n\n"
                        % (nq_total, nq_in, nq_total - nq_in, full_rel))
            else:
                o.write("\n## 原文引用（逐字 · 全 %d 条 · 不截断）\n\n" % nq_total)
            o.write(quote_used.rstrip("\n") + "\n")
        if synth_used:
            o.write("\n" + synth_used.rstrip("\n") + "\n")
        elif synth and legacy_rel:
            o.write(f"\n> 旧预览页合成的「输入 / 输出契约 · 执行步骤 · 边界与不做 · 技能关联」四段"
                    f"因 12 KB 上限未内联，原文见 `{legacy_rel}`。\n")
        o.write(fixed_tail)
        return o.getvalue()

    def take_quotes(qb):
        tk, us, ct = [], 0, False
        for st, en in (quote_units(vlines) if qs else []):
            seg_end = en if en is not None else len(vlines) - 1
            seg = "\n".join(vlines[st:seg_end + 1])
            if us + len(seg.encode()) + 1 > qb:
                ct = True
                break
            tk.append((st, seg_end))
            us += len(seg.encode()) + 1
        return tk, us, (ct or len(tk) < nq_total)

    def take_excerpt(qb, qspan):
        qlines = set(range(qspan[0], qspan[1] + 1)) if qspan else set()
        ex, us = [], 0
        for i, l in enumerate(vlines):
            if i in fences or i in qlines:
                continue
            if NEVER_INLINE_H2.match(l) or l.startswith("## ⑦") or l.startswith("## ⑧"):
                break
            if us + len(l.encode()) + 1 > qb:
                break
            ex.append((i, l))
            us += len(l.encode()) + 1
        while ex and ex[-1][1].lstrip().startswith("|"):
            ex.pop()
        while ex and not ex[-1][1].strip():
            ex.pop()
        return ex

    # 首轮估算 → 再按**真实字节**收紧，直到 ≤ 门禁。
    # 引文是**最后**才动的（它是本次任务的载荷）；先砍摘录，再砍旧合成段，最后才减引文。
    fm_probe = len(fm_render(0).encode())
    tk, _u, cut = take_quotes(max(0, staging_budget() - fm_probe
                                 - len(fixed_tail.encode()) - len(synth.encode()) - 2000))
    qspan = (tk[0][0], tk[-1][1]) if tk else None
    quote_used = "\n".join(vlines[qspan[0]:qspan[1] + 1]) if qspan else ""
    ex = take_excerpt(max(0, staging_budget() - fm_probe - len(fixed_tail.encode())
                          - len(synth.encode()) - len(quote_used.encode()) - 400), qspan)
    text = compose(ex, synth, qspan, len(tk), cut)
    for _ in range(80):
        if len(text.encode()) <= staging_budget():
            break
        if ex:
            ex = ex[:max(0, len(ex) - max(1, len(ex) // 8))]
            text = compose(ex, synth, qspan, len(tk), cut)
            continue
        if synth:
            synth = ""
            text = compose(ex, synth, qspan, len(tk), cut)
            continue
        if len(tk) > 1:
            tk = tk[:-1]
            qspan = (tk[0][0], tk[-1][1])
            cut = True
            text = compose(ex, synth, qspan, len(tk), cut)
            continue
        break
    last_idx = ex[-1][0] if ex else -1
    over_budget = len(text.encode()) > MAX_SKILL_BYTES
    return {
        "text": text,
        "record": {
            "slug": slug, "vault_stem": vault_stem, "vault_path": vault_rel,
            "rebase_source_sha256": sha256_bytes(vault_bytes),
            "rebase_full_card_sha256": sha256_bytes(vault_bytes),
            "rebase_legacy_sha256": legacy_sha,
            "quality_tier": quality_tier,
            "evidence_quotes": len(tk),
            "evidence_quotes_total": nq_total,
            "evidence_quotes_complete": len(tk) == nq_total,
            "full_card_bytes": len(vault_bytes), "full_card_lines": len(vlines),
            "rebase_excerpt_span": f"0-{last_idx}" if last_idx >= 0 else "0--1",
            "rebase_quote_span": f"{qspan[0]}-{qspan[1]}" if qspan else "",
            "excerpt_emitted_lines": len(ex),
            "synth_dropped": synth == "",
            # 12 KB 是硬门禁。引文一条不截断 ⇒ 极少数卡可能连「引文 + 固定骨架」都装不下。
            # 此时**响亮失败**，绝不静默截引文（那正是本次要修的缺陷形态）。
            "over_budget": len(text.encode()) > staging_budget(),
            "bytes": len(text.encode()),
            "body_bytes": len(text.encode()) - len(fm_render(len(tk)).encode()),
            "kind": "new" if add_new else "rebase",
        },
    }


# --------------------------------------------------------------------------- #
# 读现场
# --------------------------------------------------------------------------- #
def load_world(p2s_root: Path, vault: Path) -> dict:
    if not p2s_root.is_dir():
        raise NoInput(f"p2s 包不存在：{p2s_root}")
    if not vault.is_dir():
        raise NoInput(f"vault 不存在：{vault}")
    staging = p2s_root / "staging"
    if not staging.is_dir():
        raise NoInput(f"staging 不存在：{staging}")
    cards = []
    for p in sorted(staging.rglob("SKILL.md")):
        if "backup" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        cards.append({"dir": p.parent, "path": p, "text": text, "fields": fm_fields(text)})
    if not cards:
        raise NoInput(f"扫到 0 张 p2s 卡：{staging}")
    vcards = {}
    for p in sorted(vault.rglob("Skill-*.md")):
        if ".git" in p.parts or "node_modules" in p.parts:
            continue
        vcards[p.stem] = p
    if not vcards:
        raise NoInput(f"扫到 0 张 vault 卡：{vault}")
    cls_path = p2s_root / "data" / "classification.json"
    if not cls_path.is_file():
        raise NoInput(f"classification.json 不存在：{cls_path}")
    cls = json.loads(cls_path.read_text(encoding="utf-8"))
    rcv_path = p2s_root / "data" / "code-recovery.json"
    recovery = json.loads(rcv_path.read_text(encoding="utf-8")).get("cards", {}) if rcv_path.is_file() else {}
    return {"staging": staging, "cards": cards, "vault_root": vault, "vcards": vcards,
            "cls": cls, "recovery": recovery, "p2s_root": p2s_root}


def card_id_of(c: dict) -> str:
    return c["fields"].get("p2s_card_id", "")


def load_selection_plan(p2s_root: Path) -> dict:
    """53 张精选线独有卡的计划（由 `scripts/append-selected-line.mjs` 产出）。"""
    p = p2s_root / "generated" / "selected-line-plan.json"
    if not p.is_file():
        raise NoInput(f"精选线接入计划不存在：{p}\n  先跑 node scripts/append-selected-line.mjs --plan-out <该路径>")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# --apply
# --------------------------------------------------------------------------- #
def do_apply(world: dict, plan: dict, out, dry: bool) -> dict:
    p2s_root = world["p2s_root"]
    written, records = [], []

    # 1) 93 张换底
    for c in world["cards"]:
        cid = card_id_of(c)
        if not cid or cid not in world["vcards"]:
            continue
        vp = world["vcards"][cid]
        vb = vp.read_bytes()
        facets = {k: c["fields"][k] for k in
                  ("l1_id", "l1_plane", "l2_id", "l2_domain", "l3_id", "l3_business",
                   "l3_all", "l1_l2_l3") if c["fields"].get(k)}
        r = build_card(slug=c["dir"].name, old_text=c["text"], old_fields=c["fields"],
                       vault_bytes=vb, vault_rel=str(vp.relative_to(world["vault_root"].parent)),
                       vault_stem=cid, facets=facets, quality_tier="curated",
                       recovery=world["recovery"].get(cid), add_new=False)
        records.append(r["record"])
        if not dry:
            c["path"].write_text(r["text"], encoding="utf-8")
            refs = c["dir"] / "references"
            refs.mkdir(exist_ok=True)
            (refs / "full-card.md").write_bytes(vb)
            (refs / "legacy-preview-card.md").write_bytes(c["text"].encode("utf-8"))
        written.append((c["dir"].name, r["record"]["bytes"]))

    # 2) 1245 张标 preview（最小手术：只在 frontmatter 里插一行）
    for c in world["cards"]:
        cid = card_id_of(c)
        if cid and cid in world["vcards"]:
            continue
        if c["fields"].get("quality_tier") == "preview":
            continue
        new = insert_fm_field(c["text"], "quality_tier", "preview")
        records.append({"slug": c["dir"].name, "kind": "preview", "quality_tier": "preview",
                        "bytes": len(new.encode()), "vault_stem": None})
        written.append((c["dir"].name, len(new.encode())))
        if not dry:
            c["path"].write_text(new, encoding="utf-8")

    # 3) 53 张新增
    cls_items = {it["id"]: it for it in world["cls"]["items"]}
    for item in plan["items"]:
        vp = world["vault_root"].parent / item["_provenance"]["vault_path"]
        if not vp.is_file():
            raise NoInput(f"精选线源卡不在：{vp}")
        vb = vp.read_bytes()
        facets = item["facets"]
        # 落到 L2 域目录（与 legacy 卡同一套目录名）
        dom = facets["l2_domain"]
        d = world["staging"] / dom / item["slug"]
        dummy_old = {"title": item["title"]}
        r = build_card(slug=item["slug"], old_text=None, old_fields=dummy_old,
                       vault_bytes=vb, vault_rel=item["_provenance"]["vault_path"],
                       vault_stem=item["id"], facets=facets, quality_tier="curated",
                       recovery=world["recovery"].get(item["id"]), add_new=True)
        records.append(r["record"])
        written.append((item["slug"], r["record"]["bytes"]))
        if not dry:
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(r["text"], encoding="utf-8")
            (d / "references").mkdir(exist_ok=True)
            (d / "references" / "full-card.md").write_bytes(vb)

    manifest = {"version": REBASE_VERSION, "generated": now_iso(),
                "counts": {"rebase": sum(1 for r in records if r["kind"] == "rebase"),
                           "new": sum(1 for r in records if r["kind"] == "new"),
                           "preview": sum(1 for r in records if r["kind"] == "preview")},
                "records": records}
    if not dry:
        mp = p2s_root / "generated" / "rebase-plan.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    over = [r for r in records if r.get("over_budget")]
    out.write(f"写入 {len(written)} 张卡（rebase {manifest['counts']['rebase']} / "
              f"new {manifest['counts']['new']} / preview {manifest['counts']['preview']}）"
              + ("（dry-run，未落盘）" if dry else "") + "\n")
    if over:
        out.write(f"\n🔴 {len(over)} 张换底后仍超 12 KB 硬门禁"
                  f"（引文不截断是硬约束，故这是**响亮失败**而不是静默截引文）：\n")
        for r in over[:10]:
            out.write(f"  · {r['slug']}: {r['bytes']} 字节（引文 {r['evidence_quotes']} 条）\n")
        manifest["over_budget"] = [r["slug"] for r in over]
    return manifest


def insert_fm_field(text: str, key: str, value: str) -> str:
    """在 frontmatter 里插一行（幂等：已有同名键则替换值）。"""
    block, body = split_frontmatter(text)
    if not block:
        raise NoInput("卡没有 frontmatter，无法插字段")
    lines = block.rstrip("\n").split("\n")
    for i, l in enumerate(lines):
        m = RE_FM_LINE.match(l)
        if m and m.group(1) == key:
            lines[i] = f"{key}: {jstr(value)}"
            return "\n".join(lines) + "\n" + body
    # 插在 `l1_l2_l3:` 之后；没有就插在最后一行 `---` 之前
    at = None
    for i, l in enumerate(lines):
        m = RE_FM_LINE.match(l)
        if m and m.group(1) in ("l1_l2_l3", "p2s_src_domain"):
            at = i
    if at is None:
        at = len(lines) - 2
    lines.insert(at + 1, f"{key}: {jstr(value)}")
    return "\n".join(lines) + "\n" + body


def now_iso() -> str:
    import datetime
    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# --check：对**现场**重算
# --------------------------------------------------------------------------- #
def do_check(world: dict, out, baseline: dict | None = None) -> tuple[int, dict]:
    problems, drift, stats = [], [], {}
    vcards = world["vcards"]
    cards = world["cards"]
    by_id = {}
    for c in cards:
        cid = card_id_of(c)
        if cid:
            by_id.setdefault(cid, []).append(c)

    rebased, preview, untagged = [], [], []
    for c in cards:
        qt = c["fields"].get("quality_tier")
        if c["fields"].get("rebase_vault_card"):
            rebased.append(c)
        elif qt == "preview":
            preview.append(c)
        else:
            untagged.append(c)
    stats.update(rebase=len(rebased), preview=len(preview), untagged=len(untagged),
                 staging_total=len(cards), vault_total=len(vcards))

    # C1 分母：底本与实物
    cls_ids = {it["id"] for it in world["cls"]["items"]}
    cls_slugs = {it["slug"] for it in world["cls"]["items"]}
    staging_slugs = {c["dir"].name for c in cards}
    if staging_slugs - cls_slugs:
        problems.append(("C1", f"{len(staging_slugs - cls_slugs)} 个 staging 目录不在分类底本里："
                               f"{sorted(staging_slugs - cls_slugs)[:5]}"))
    if len(cls_slugs) != len(staging_slugs):
        problems.append(("C1", f"底本 {len(cls_slugs)} 个 slug != staging 实物 {len(staging_slugs)} 个"))

    # C2 12 KB 硬门禁（**每张，不是抽样**）
    over = [(c["dir"].name, len(c["text"].encode())) for c in cards
            if len(c["text"].encode()) > staging_budget()]
    stats["over_12k"] = len(over)
    for s, b in over:
        problems.append(("C2", f"{s}: SKILL.md {b} 字节 > **staging 预算** {staging_budget()}"
                               f"（12KB 门禁 {MAX_SKILL_BYTES} − 装机追加余量 {INSTALL_HEADROOM_BYTES}）"
                               f" —— 装机后必然超限"))
    stats["max_staging_bytes"] = max((len(c["text"].encode()) for c in cards), default=0)

    # C3 quality_tier 覆盖与取值域
    dom_tiers = {"curated", "preview"}
    bad_tier = [(c["dir"].name, c["fields"].get("quality_tier")) for c in cards
                if c["fields"].get("quality_tier") not in dom_tiers]
    stats["quality_tier_covered"] = len(cards) - len(bad_tier)
    for s, v in bad_tier:
        problems.append(("C3", f"{s}: quality_tier「{v}」不在词表 {sorted(dom_tiers)} 内"))
    stats["quality_tier_curated"] = sum(1 for c in cards if c["fields"].get("quality_tier") == "curated")
    stats["quality_tier_preview"] = sum(1 for c in cards if c["fields"].get("quality_tier") == "preview")

    # C4 源指纹漂移 + 完整卡一致性 + 引文完整 + 摘录逐字节可复算
    stats["evidence_quotes_inline"] = 0
    stats["evidence_quotes_total"] = 0
    stats["cards_partial_evidence"] = 0
    stats["cards_complete_evidence"] = 0
    stats["vault_cards_not_rebased"] = len([s for s in vcards if s not in by_id])
    for c in rebased:
        slug = c["dir"].name
        cid = c["fields"].get("p2s_card_id")
        vp = vcards.get(cid)
        if not vp:
            problems.append(("C4", f"{slug}: p2s_card_id「{cid}」在 vault 里不存在"))
            continue
        vb = vp.read_bytes()
        cur = sha256_bytes(vb)
        rec_sha = c["fields"].get("rebase_source_sha256", "")
        if rec_sha != cur:
            drift.append((slug, cid, rec_sha, cur))
            problems.append(("C4", f"{slug}: **源卡漂移** —— 换底时记 {rec_sha[:12]}，现算 {cur[:12]}"
                                   f"（{cid}）"))
        full = c["dir"] / "references" / "full-card.md"
        if not full.is_file():
            problems.append(("C4", f"{slug}: references/full-card.md 不存在"))
            continue
        fb = full.read_bytes()
        if sha256_bytes(fb) != c["fields"].get("rebase_full_card_sha256"):
            problems.append(("C4", f"{slug}: full-card.md 的 sha256 与 frontmatter 不符"))
        if fb != vb:
            problems.append(("C4", f"{slug}: full-card.md 与 vault 源卡**逐字节不同**"))
        # 引文：SKILL.md 里的条数必须等于完整卡里的条数（证据链不许被截断）
        full_txt = fb.decode("utf-8")
        _, fbody = split_frontmatter(full_txt)
        n_full = sum(1 for l in fbody.split("\n") if RE_QUOTE.match(l))
        n_skill = sum(1 for l in c["text"].split("\n") if RE_QUOTE.match(l))
        stats["evidence_quotes_inline"] += n_skill
        stats["evidence_quotes_total"] += n_full
        # 12 KB 装不下 30–55 条引文 ⇒ 允许**有界**内联，但必须：① 条数自报准确（frontmatter 与实物一致）；
        # ② 不完整时**在正文里点名**；③ 一条都没内联时仍留下可见缺口（不许整段消失）。
        # 静默截断 = 假绿，正是本仓库纪律 ⑧ 要拦的东西。
        declared = int(c["fields"].get("rebase_evidence_quotes", "-1"))
        declared_total = int(c["fields"].get("rebase_evidence_quotes_total", "-1"))
        if declared != n_skill:
            problems.append(("C4", f"{slug}: frontmatter 说内联 {declared} 条，实物 {n_skill} 条"))
        if declared_total != n_full:
            problems.append(("C4", f"{slug}: frontmatter 说完整卡有 {declared_total} 条引文，实物 {n_full} 条"))
        if n_skill < n_full:
            stats["cards_partial_evidence"] += 1
            if n_skill == 0:
                problems.append(("C4", f"{slug}: 完整卡有 {n_full} 条引文，本页一条都没内联"
                                       f"（证据链完全没到达产品）"))
            # ⚠️ 判据必须**正面要求**那个缺口声明，不能只要求「标题里出现『全 N 条』」——
            #    首版就是这么写的，于是「悄悄删掉一条引文、标题原样不动」照样通过：
            #    标题写着「全 2 条 · 不截断」，实物只有 1 条，判据却看不出（假绿）。
            #    现在要求标题/声明里出现**精确的** `内联 k / 全 N 条`。
            marker = f"内联 {n_skill} / 全 {n_full} 条"
            if marker not in c["text"]:
                problems.append(("C4", f"{slug}: 引文只有 {n_skill}/{n_full} 条，正文却**没有**声明这个缺口"
                                       f"（缺 `{marker}`） —— 静默截断"))
        else:
            stats["cards_complete_evidence"] += 1
        # 内联引文必须是完整卡的一个**连续**片段，逐字节相同。
        # ⚠️ 判据要用**自报区间** `rebase_quote_span`，不能用整段引文区间 —— 有界内联时
        #    页内只有前缀，拿整段去比会 46 张全部假红（首版实测）。
        fbody_lines = fbody.split("\n")
        qs_full = quote_span(fbody_lines)
        rec_span = c["fields"].get("rebase_quote_span", "")
        if rec_span and "-" in rec_span:
            try:
                a_s, b_s = rec_span.split("-", 1)
                a_i, b_i = int(a_s), int(b_s)
            except ValueError:
                problems.append(("C4", f"{slug}: rebase_quote_span「{rec_span}」不成形"))
                a_i = b_i = -1
            if a_i >= 0 and b_i >= a_i:
                if qs_full and not (qs_full[0] <= a_i and b_i <= qs_full[1]):
                    problems.append(("C4", f"{slug}: 自报引文区间 {rec_span} 超出完整卡的引文区间"))
                want = "\n".join(fbody_lines[a_i:b_i + 1])
                if want not in c["text"]:
                    problems.append(("C4", f"{slug}: 内联引文不是完整卡的逐字节连续片段"
                                           f"（自报区间 {rec_span}）"))
        # 摘录必须是完整卡的可复算子序列
        lines = fbody_lines
        fences = fenced_line_set(lines)
        # 摘录的可复算同样要用**自报引文区间**（页内只内联了引文前缀）
        qlines = set(range(a_i, b_i + 1)) if (rec_span and a_i >= 0 and b_i >= a_i) else set()
        span = c["fields"].get("rebase_excerpt_span", "")
        if span and "-" in span:
            a, b = span.split("-", 1)
            # "0--1" 表示没有摘录
            if b != "-1" and b.lstrip("-").isdigit() and int(b) >= 0:
                idx = [i for i in range(0, int(b) + 1) if i not in fences and i not in qlines]
                want_ex = [lines[i] for i in idx]
                while want_ex and want_ex[-1].lstrip().startswith("|"):
                    want_ex.pop()
                while want_ex and not want_ex[-1].strip():
                    want_ex.pop()
                if want_ex:
                    if "\n".join(want_ex) not in c["text"]:
                        problems.append(("C4", f"{slug}: 摘录与完整卡第 0–{b} 行（去代码/去引文）"
                                               f"复算不上"))
        # legacy 留档
        if c["fields"].get("rebase_legacy_card"):
            lp = c["dir"] / c["fields"]["rebase_legacy_card"]
            if not lp.is_file():
                problems.append(("C4", f"{slug}: legacy 留档不存在（换底不可回看）"))
            elif sha256_bytes(lp.read_bytes()) != c["fields"].get("rebase_legacy_card_sha256"):
                problems.append(("C4", f"{slug}: legacy 留档 sha256 不符"))

    # C5 新增的 53 张：底本里必须有、目录必须有、分类字段必须与底本一致
    for c in rebased:
        cid = c["fields"].get("p2s_card_id")
        if cid in by_id and len(by_id[cid]) > 1:
            problems.append(("C5", f"{cid} 被 {len(by_id[cid])} 张 staging 卡同时认领"))
    # ⚠️ 底本里实测有 **11 条 `l3: []` / `facets: null`** 的「矩阵空白」卡 —— 空 L3 是**合法结论**
    #    （产品侧已有先例），不是缺陷；这里不能对 `facets` 直接下标（首版就撞了 TypeError，
    #    门禁自己炸掉 —— 那是 exit 3 的形态，不是「判红」）。它们单独计数，当作一等输出。
    blank_items = [it for it in world["cls"]["items"] if not it.get("facets")]
    stats["blank_l3_in_ledger"] = len(blank_items)
    for it in world["cls"]["items"]:
        f = it["facets"]
        if not f:
            continue
        hit = [c for c in cards if c["dir"].name == it["slug"]]
        if not hit:
            problems.append(("C5", f"底本条目 {it['slug']} 在 staging 里没有实物"))
            continue
        fld = hit[0]["fields"]
        for k in ("l1_id", "l2_id", "l3_business"):
            if fld.get(k) and fld[k] != f[k]:
                problems.append(("C5", f"{it['slug']}: frontmatter {k}「{fld[k]}」"
                                       f"与底本「{f[k]}」不一致"))

    # C6 未打标的卡
    for c in untagged:
        problems.append(("C6", f"{c['dir'].name}: 既没有 vault_card，也没有 quality_tier —— 没换到底"))

    kept, waived = [], []
    for code, msg in problems:
        (waived if baseline and any(
            w["code"] == code and w["slug"] in msg for w in baseline.get("items", [])) else kept).append(
            {"code": code, "slug": _slug_of(msg), "detail": msg})

    out.write("PHASE6 S5/B11 · p2s 换底 --check（对现场重算）\n")
    out.write(f"  staging 卡          {stats['staging_total']}\n")
    out.write(f"  已换底（有 vault_card） {stats['rebase']}\n")
    out.write(f"  quality_tier=preview  {stats['preview']}\n")
    out.write(f"  未打标              {stats['untagged']}\n")
    out.write(f"  vault 卡            {stats['vault_total']}\n")
    out.write(f"  其中未被认领（vault 独有） {stats['vault_cards_not_rebased']}\n")
    out.write(f"  超 staging 预算      {stats['over_12k']}"
              f"（预算 {staging_budget()} = 12KB − 装机余量 {INSTALL_HEADROOM_BYTES}；"
              f"最大卡 {stats.get('max_staging_bytes')} 字节）\n")
    out.write(f"  底本空 L3 条目（合法）  {stats.get('blank_l3_in_ledger', 0)}\n")
    out.write(f"  quality_tier 覆盖    {stats['quality_tier_covered']}/{stats['staging_total']}"
              f"（curated {stats['quality_tier_curated']} / preview {stats['quality_tier_preview']}）\n")
    out.write(f"  内联逐字引文          {stats['evidence_quotes_inline']}/{stats['evidence_quotes_total']} 条"
              f"（完整内联 {stats['cards_complete_evidence']} 张 / 有界内联 {stats['cards_partial_evidence']} 张）\n")
    if drift:
        out.write(f"\n🔴 源指纹漂移 {len(drift)} 条（vault 有并发写入者，换底中途源变了）：\n")
        for slug, cid, a, b in drift:
            out.write(f"  · {slug} ← {cid}：{a[:16]} → {b[:16]}\n")
    if kept:
        out.write(f"\n🔴 判红 {len(kept)} 条：\n")
        for p in kept[:40]:
            out.write(f"  · [{p['code']}] {p['detail']}\n")
        if len(kept) > 40:
            out.write(f"  …（共 {len(kept)} 条）\n")
    else:
        out.write("\n✅ 全部判据通过\n")
    if waived:
        out.write(f"\n⚠️ 已豁免 {len(waived)} 条（可见豁免，不是静默放行）：\n")
        for p in waived:
            out.write(f"  · 豁免 [{p['code']}] {p['detail']}\n")
        out.write(f"  到期条件：{baseline.get('expires_when', '（未写 —— 必须补）')}\n")
    stats["red"] = len(kept)
    return (EXIT_RED if kept else EXIT_OK), {"stats": stats, "problems": kept, "waived": waived,
                                             "drift": drift}


def _slug_of(msg: str) -> str:
    m = re.match(r"^([a-z0-9][a-z0-9-]*):", msg)
    return m.group(1) if m else "-"


# --------------------------------------------------------------------------- #
# CLI —— 路径只来自 argv/env，绝不硬编码
# --------------------------------------------------------------------------- #
def cli(argv: list[str] | None = None, *, _stdout=None, _stderr=None) -> int:
    out = _stdout or sys.stdout
    err = _stderr or sys.stderr
    ap = argparse.ArgumentParser(prog="rebase_p2s_cards.py")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--p2s-root", default=os.environ.get("P2S_PKG", DEFAULT_P2S))
    ap.add_argument("--vault", default=os.environ.get("P2S_VAULT", DEFAULT_VAULT))
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest(out)
    if a.mutate:
        return mutate(out, err)

    try:
        world = load_world(Path(a.p2s_root), Path(a.vault))
    except NoInput as e:
        err.write(f"✗ 输入没拿到：{e}\n  退出码 2 = 没测，不是通过。\n")
        return EXIT_NO_INPUT
    except Exception as e:
        err.write(f"✗ 门禁内部错误：{type(e).__name__}: {e}\n")
        return EXIT_INTERNAL

    if a.apply:
        try:
            plan = load_selection_plan(Path(a.p2s_root))
        except NoInput as e:
            err.write(f"✗ 输入没拿到：{e}\n")
            return EXIT_NO_INPUT
        m = do_apply(world, plan, out, dry=a.dry_run)
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        return EXIT_RED if m.get("over_budget") else EXIT_OK

    baseline = None
    if a.baseline:
        bp = Path(a.baseline)
        if not bp.is_file():
            err.write(f"✗ 输入没拿到：baseline 不存在 {bp}\n")
            return EXIT_NO_INPUT
        baseline = json.loads(bp.read_text(encoding="utf-8"))
    code, doc = do_check(world, out, baseline)
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return code


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
VAULT_SAMPLE = """---
title: Sample Card (样例)
venue_tier: CCF-A
paper_id: 2401.00001
evidence_grade: A
l1_id: PLN-OPS
---
# Skill Card: Sample

## ① 算法原理

核心思想是**分段线性**拟合。这里放一段够长的正文，用来占掉预算，
好让摘录在中途被截断，从而把「截断必须留痕」这条判据逼出来。

```python
def f(x):
    return x * 2
```

数学直觉：斜率的变化点就是临界点。

## ② 母婴出海应用案例

吸奶器品类的完播率临界点。

## ③ 代码模板

```python
print("hi")
```

## ⑥ 原文引用

> 原文:"This is a verbatim quote from the paper."
> 出处：2401.00001 §Abstract

> 原文:"Second verbatim quote, longer than the first one."
> 出处：2401.00001 §2

## ⑧ 论文来源

arXiv:2401.00001
"""

P2S_SAMPLE = """---
name: "p2s-sample-card"
title: "样例卡 — 用于自检"
description: "触发词：样例。何时不用：不是真的卡。安全边界：无。"
l1_id: "PLN-OPS"
l1_plane: "业务运营"
l2_id: "DOM-03"
l2_domain: "供应与履约"
l3_id: "DOM-03-047"
l3_business: "需求预测"
l3_all: "需求预测"
l1_l2_l3: "业务运营/供应与履约/需求预测"
p2s_card_id: "Skill-Sample-Card"
p2s_src_domain: "03-时间序列"
user_summary: "样例摘要。"
whenToUse: "样例时机。"
enabled: "true"
disable-model-invocation: "true"
user-invocable: "true"
---

# 样例卡 — 用于自检

## ① 解决的问题

预览页的问题陈述。

## ⑦ 代码节选

```python
print("preview")
```

## ⑧ 论文来源

（卡页此段未自动抽取。）

## 输入 / 输出契约

**输入**：历史日销量。

## 执行步骤

1. 收集数据。
"""


def _fixture(root: Path, *, vault_text=VAULT_SAMPLE, recovery=None) -> tuple[Path, Path]:
    p2s = root / "pkg"
    vault = root / "vault"
    d = p2s / "staging" / "供应与履约" / "p2s-sample-card"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(P2S_SAMPLE, encoding="utf-8")
    (d / "references").mkdir(exist_ok=True)
    (d / "references" / "implementation.py").write_text("# impl\nprint(1)\n", encoding="utf-8")
    vd = vault / "03-时间序列"
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "Skill-Sample-Card.md").write_text(vault_text, encoding="utf-8")
    (p2s / "data").mkdir(parents=True, exist_ok=True)
    (p2s / "data" / "classification.json").write_text(json.dumps({
        "version": "1.0", "total": 1, "classified": 1, "unclassified": 0,
        "items": [{"id": "Skill-Sample-Card", "slug": "p2s-sample-card", "title": "样例卡",
                   "src_domain": "03-时间序列", "l3": ["需求预测"],
                   "facets": {"l1_id": "PLN-OPS", "l1_plane": "业务运营", "l2_id": "DOM-03",
                              "l2_domain": "供应与履约", "l3_id": "DOM-03-047",
                              "l3_business": "需求预测", "l3_all": "需求预测",
                              "l1_l2_l3": "业务运营/供应与履约/需求预测"},
                   "planes": ["PLN-OPS"], "domains": ["DOM-03"], "cross_plane": False,
                   "confidence": "high", "fills_gap": [], "fills_gap_strong": [],
                   "fills_gap_adjacent": [], "fills_gap_dropped": [], "note": ""}]},
        ensure_ascii=False), encoding="utf-8")
    (p2s / "data" / "code-recovery.json").write_text(json.dumps(
        {"cards": {"Skill-Sample-Card": recovery or {"tier": "oracle", "lines": 2,
                                                     "cross_check": "prefix@1"}}},
        ensure_ascii=False), encoding="utf-8")
    (p2s / "generated").mkdir(parents=True, exist_ok=True)
    (p2s / "generated" / "selected-line-plan.json").write_text(json.dumps(
        {"items": [], "problems": [], "to_add": 0}, ensure_ascii=False), encoding="utf-8")
    return p2s, vault


def _run_cli(p2s: Path, vault: Path, *extra: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli(["--check", "--p2s-root", str(p2s), "--vault", str(vault), *extra],
               _stdout=buf, _stderr=buf)
    return code, buf.getvalue()


# --------------------------------------------------------------------------- #
# --selftest
# --------------------------------------------------------------------------- #
def selftest(out) -> int:
    cases = []
    tmp = Path(tempfile.mkdtemp(prefix="rebase-selftest-"))

    def case(name, got, want, note=""):
        cases.append((name, got == want, f"got={got!r} want={want!r} {note}"))

    # --- 用例 1：换底后 12 KB 全过 + 引文一条不丢 + full-card 与源逐字节同 ----
    p2s, vault = _fixture(tmp / "f1")
    code = cli(["--apply", "--p2s-root", str(p2s), "--vault", str(vault)],
               _stdout=io.StringIO(), _stderr=io.StringIO())
    case("1 apply exit 0", code, EXIT_OK)
    d = p2s / "staging" / "供应与履约" / "p2s-sample-card"
    txt = (d / "SKILL.md").read_text(encoding="utf-8")
    # ⚠️ 钉**字面 12288**，不许写成 MAX_SKILL_BYTES —— 后者是被测常量，变异它时夹具会跟着变，
    #    用例就对同一个改动前后都绿（M1 实测撞出：这条断言当时是自指的）。
    case("1 门禁常量就是 12288（硬门禁不是建议）", MAX_SKILL_BYTES, 12288)
    case("1 SKILL.md ≤ 12288", len(txt.encode()) <= 12288, True, f"{len(txt.encode())}")
    case("1 内联两条引文",
         sum(1 for l in txt.split("\n") if RE_QUOTE.match(l)), 2)
    case("1 full-card 与源逐字节同", (d / "references" / "full-card.md").read_bytes(),
         (vault / "03-时间序列" / "Skill-Sample-Card.md").read_bytes())
    case("1 legacy 留档就是旧卡", (d / "references" / "legacy-preview-card.md").read_text("utf-8"),
         P2S_SAMPLE)
    case("1 ⑦ 指向 implementation.py", "references/implementation.py" in txt, True)
    case("1 ⑦ 段不再内联代码围栏",
         "```" in txt.split("## ⑦")[1].split("## ⑧")[0], False)
    code, rep = _run_cli(p2s, vault)
    case("1 换底后 --check exit 0", code, EXIT_OK, rep[-200:])
    case("1 check 记到内联引文 2/2 条", "内联逐字引文          2/2 条" in rep, True)

    # --- 用例 2（篡改样本）：把内联引文删一条 ⇒ 必须报「证据链被截断」 -------
    d2 = tmp / "f2"
    shutil.copytree(tmp / "f1", d2)
    p2s2 = d2 / "pkg"
    sk = p2s2 / "staging" / "供应与履约" / "p2s-sample-card" / "SKILL.md"
    t2 = sk.read_text(encoding="utf-8")
    lines = [l for l in t2.split("\n")]
    for i, l in enumerate(lines):
        if RE_QUOTE.match(l):
            del lines[i:i + 2]
            break
    t2 = "\n".join(lines)
    # 强化篡改：连 frontmatter 的自报条数也一起改掉 —— 于是「条数不符」这条判据不会响，
    # **只剩**「不完整却没在正文里点名」这条能抓住它。这样用例测的才是那条判据本身，
    # 而不是被另一条判据顺手兜住（首版就是这样，`静默截断` 从未被触发）。
    t2 = re.sub(r'^rebase_evidence_quotes: "\d+"', 'rebase_evidence_quotes: "1"', t2, flags=re.M)
    sk.write_text(t2, encoding="utf-8")
    code, rep = _run_cli(p2s2, d2 / "vault")
    case("2 引文被删 ⇒ 判红", code, EXIT_RED, rep[-200:])
    case("2 报「静默截断」", "静默截断" in rep, True)

    # --- 用例 2b（另一条判据的篡改样本）：删引文但**不改** frontmatter 自报条数 -------
    d2b = tmp / "f2b"
    shutil.copytree(tmp / "f1", d2b)
    sk2b = d2b / "pkg" / "staging" / "供应与履约" / "p2s-sample-card" / "SKILL.md"
    ln = sk2b.read_text(encoding="utf-8").split("\n")
    for i, l in enumerate(ln):
        if RE_QUOTE.match(l):
            del ln[i:i + 2]
            break
    sk2b.write_text("\n".join(ln), encoding="utf-8")
    _c2b, rep2b = _run_cli(d2b / "pkg", d2b / "vault")
    case("2b 报 frontmatter 与实际条数不符", "frontmatter 说内联" in rep2b, True)

    # --- 用例 3（篡改样本）：改 SKILL.md 到 12KB 以上 ⇒ 必须报 C2 -----------
    d3 = tmp / "f3"
    shutil.copytree(tmp / "f1", d3)
    p2s3 = d3 / "pkg"
    sk3 = p2s3 / "staging" / "供应与履约" / "p2s-sample-card" / "SKILL.md"
    sk3.write_text(sk3.read_text(encoding="utf-8") + ("x" * (12288 + 50)), encoding="utf-8")
    code, rep = _run_cli(p2s3, d3 / "vault")
    case("3 超 12KB ⇒ 判红", code, EXIT_RED, rep[-200:])
    case("3 报 C2", "[C2]" in rep, True)

    # --- 用例 3b（装机余量的篡改样本）：staging 卡**正好 12288**也必须判红 ------
    # 因为 12 KB 门禁量的是**已装卡**，而装机链之后还会追 `p2s_*` 透传字段。
    # 去掉余量这条判据就形同虚设（M7 变异锁这条）。
    d3b = tmp / "f3b"
    shutil.copytree(tmp / "f1", d3b)
    sk3b = d3b / "pkg" / "staging" / "供应与履约" / "p2s-sample-card" / "SKILL.md"
    body3b = sk3b.read_text(encoding="utf-8")
    pad = 12288 - len(body3b.encode())
    case("3b 夹具恰好补到 12288", pad > 0, True, f"pad={pad}")
    sk3b.write_text(body3b + "x" * pad, encoding="utf-8")
    _c3b, rep3b = _run_cli(d3b / "pkg", d3b / "vault")
    case("3b staging 顶到 12288 也判红（要给装机追加留余量）", _c3b, EXIT_RED, rep3b[-160:])
    case("3b 报告写清预算 = 12KB − 余量", f"装机追加余量 {INSTALL_HEADROOM_BYTES}" in rep3b, True)

    # --- 用例 4（篡改样本）：源卡变了但没重换底 ⇒ 必须报源指纹漂移 ----------
    d4 = tmp / "f4"
    shutil.copytree(tmp / "f1", d4)
    v4 = d4 / "vault" / "03-时间序列" / "Skill-Sample-Card.md"
    v4.write_text(VAULT_SAMPLE.replace("Slope", "Slope") + "\n<!-- 并发写入者改了一笔 -->\n",
                  encoding="utf-8")
    code, rep = _run_cli(d4 / "pkg", d4 / "vault")
    case("4 源漂移 ⇒ 判红", code, EXIT_RED, rep[-200:])
    case("4 报源指纹漂移", "源指纹漂移" in rep, True)

    # --- 用例 5（篡改样本）：full-card.md 被改 ⇒ 必须报逐字节不同 -----------
    d5 = tmp / "f5"
    shutil.copytree(tmp / "f1", d5)
    fc = d5 / "pkg" / "staging" / "供应与履约" / "p2s-sample-card" / "references" / "full-card.md"
    fc.write_text(fc.read_text(encoding="utf-8").replace("Second", "SECOND"), encoding="utf-8")
    code, rep = _run_cli(d5 / "pkg", d5 / "vault")
    case("5 full-card 被改 ⇒ 判红", code, EXIT_RED, rep[-200:])
    case("5 报 sha256 不符", "sha256 与 frontmatter 不符" in rep, True)

    # --- 用例 6（I「不许静默放行」）：把 quality_tier 去掉 ⇒ 必须报 C3/C6 ---
    d6 = tmp / "f6"
    shutil.copytree(tmp / "f1", d6)
    sk6 = d6 / "pkg" / "staging" / "供应与履约" / "p2s-sample-card" / "SKILL.md"
    sk6.write_text(re.sub(r'^quality_tier:.*\n', '', sk6.read_text(encoding="utf-8"), flags=re.M),
                   encoding="utf-8")
    code, rep = _run_cli(d6 / "pkg", d6 / "vault")
    case("6 缺 quality_tier ⇒ 判红", code, EXIT_RED, rep[-200:])
    case("6 报 C3", "[C3]" in rep, True)

    # --- 用例 7：输入没拿到 ⇒ exit 2（**不是 0**） -------------------------
    code, rep = _run_cli(tmp / "nope", tmp / "nope2")
    case("7 输入没有 exit 2", code, EXIT_NO_INPUT, rep[-160:])

    # --- 用例 8：staging 空 ⇒ exit 2 --------------------------------------
    d8 = tmp / "f8"
    (d8 / "pkg" / "staging").mkdir(parents=True)
    (d8 / "vault").mkdir(parents=True)
    (d8 / "pkg" / "data").mkdir(parents=True)
    code, rep = _run_cli(d8 / "pkg", d8 / "vault")
    case("8 staging 空 exit 2", code, EXIT_NO_INPUT, rep[-160:])

    # --- 用例 9：12 KB 预算真的在承重（把上限调小，摘录必须变短、引文仍在）--
    small = _fixture(tmp / "f9")
    p9, v9 = small
    import unittest.mock as _mock
    _buf9 = io.StringIO()
    with _mock.patch.object(sys.modules[__name__], "MAX_SKILL_BYTES", 6500):
        code9 = cli(["--apply", "--p2s-root", str(p9), "--vault", str(v9)],
                    _stdout=_buf9, _stderr=_buf9)
    out9 = _buf9.getvalue()
    t9 = (p9 / "staging" / "供应与履约" / "p2s-sample-card" / "SKILL.md").read_text(encoding="utf-8")
    # 预算收紧后的**新契约**（不再要求「全部引文都在页内」—— 12 KB 装不下 30–55 条引文）：
    #   ① 文件仍 ≤ 上限；② 内联的引文逐条完整（不在引文中间断开）；③ 截断**被声明**
    case("9 预算收紧后文件仍 ≤ 上限", len(t9.encode()) <= 6500, True, f"{len(t9.encode())}")
    case("9 内联引文条数 = frontmatter 记的 rebase_evidence_quotes",
         sum(1 for l in t9.split("\n") if RE_QUOTE.match(l)),
         int(re.search(r'^rebase_evidence_quotes: "(\d+)"', t9, re.M).group(1)))
    case("9 frontmatter 记了总数与「完整/不完整」标志",
         re.search(r'^rebase_evidence_quotes_total: "(\d+)"', t9, re.M) is not None
         and re.search(r'^rebase_evidence_quotes_complete: "(true|false)"', t9, re.M) is not None, True)
    case("9 内联引文逐条完整（每条都在完整卡里逐字存在，没有半句）",
         all(l in VAULT_SAMPLE for l in t9.split("\n") if RE_QUOTE.match(l)), True)
    case("9 预算装得下时不许报超限", "仍超 12 KB 硬门禁" in out9, False)
    case("9 收缩后 --check 仍能过（有界内联是合规状态，不是红灯）",
         _run_cli(p9, v9)[0], EXIT_OK)

    # 9b 预算**不可行**（连固定骨架都装不下）⇒ 必须响亮失败，不许静默产出超限文件
    p9b, v9b = _fixture(tmp / "f9b")
    _b9b = io.StringIO()
    with _mock.patch.object(sys.modules[__name__], "MAX_SKILL_BYTES", 1500):
        code9b = cli(["--apply", "--p2s-root", str(p9b), "--vault", str(v9b)],
                     _stdout=_b9b, _stderr=_b9b)
    case("9b 预算不可行 ⇒ apply 判红", code9b, EXIT_RED, _b9b.getvalue()[-160:])
    case("9b 报告点名超限的卡", "仍超 12 KB 硬门禁" in _b9b.getvalue(), True)

    # 9c 引文一条都装不下时，`## 原文引用` 段落**不许整段消失**
    p9c, v9c = _fixture(tmp / "f9c")
    _b9c = io.StringIO()
    with _mock.patch.object(sys.modules[__name__], "MAX_SKILL_BYTES", 4000):
        cli(["--apply", "--p2s-root", str(p9c), "--vault", str(v9c)],
            _stdout=_b9c, _stderr=_b9c)
    t9c = (p9c / "staging" / "供应与履约" / "p2s-sample-card" / "SKILL.md").read_text("utf-8")
    case("9c 零条内联时仍留下可见的引文缺口声明",
         "## 原文引用" in t9c and "内联 0 / 全 2 条" in t9c, True)

    # --- 用例 10：豁免可见、带到期条件 ------------------------------------
    d10 = tmp / "f10"
    shutil.copytree(tmp / "f2", d10)  # f2 是「引文被删」的红卡
    bl = d10 / "baseline.json"
    bl.write_text(json.dumps({"id": "t", "expires_when": "重新换底后",
                              "items": [{"code": "C4", "slug": "p2s-sample-card"}]},
                             ensure_ascii=False), encoding="utf-8")
    code, rep = _run_cli(d10 / "pkg", d10 / "vault", "--baseline", str(bl))
    case("10 豁免后 exit 0", code, EXIT_OK, rep[-200:])
    # ⚠️ 不要断言「已豁免 1 条」—— 删一条引文会**同时**触发「引文截断」与
    #    「内联引文不是连续片段」两条 C4，条数是 2。断言的是「豁免块被打印」这件事本身。
    case("10 豁免可见（豁免块被打印、不是静默放行）",
         "已豁免" in rep and "可见豁免，不是静默放行" in rep, True)
    case("10 到期条件被打印", "到期条件：重新换底后" in rep, True)

    # --- 用例 11：`insert_fm_field` 幂等 ----------------------------------
    t11 = insert_fm_field(insert_fm_field(P2S_SAMPLE, "quality_tier", "preview"),
                          "quality_tier", "preview")
    case("11 插字段幂等",
         sum(1 for l in t11.split("\n") if l.startswith("quality_tier:")), 1)

    bad = [c for c in cases if not c[1]]
    out.write(f"rebase_p2s_cards --selftest：{len(cases) - len(bad)}/{len(cases)} 通过\n")
    for name, ok, note in cases:
        out.write(f"  {'✅' if ok else '❌'} {name}" + ("" if ok else f"   {note}") + "\n")
    if bad:
        out.write(f"✗ {len(bad)} 条用例失败\n")
        return EXIT_RED
    return EXIT_OK


# --------------------------------------------------------------------------- #
# --mutate
# --------------------------------------------------------------------------- #
MUTATIONS = [
    ("M1 12KB 上限不再承重（判据形同虚设）",
     "MAX_SKILL_BYTES = 12 * 1024",
     "MAX_SKILL_BYTES = 12 * 1024 * 100",
     "over_12k",
     "用例 3 应报红失败"),
    ("M2 引文完整性判据被摘掉（证据链可静默截断）",
     "            if marker not in c[\"text\"]:",
     "            if False:",
     "evidence_quotes_inline",
     "用例 2 应报红失败"),
    ("M7 装机余量清零（staging 顶到 12288，装机后必然超限）",
     "INSTALL_HEADROOM_BYTES = 48",
     "INSTALL_HEADROOM_BYTES = 0",
     "max_staging_bytes",
     "用例 3b 应报红失败"),
    ("M6 frontmatter 自报条数不核（自报与实际脱节也照样绿）",
     "        if declared != n_skill:",
     "        if False:",
     "evidence_quotes_inline",
     "用例 2b 应报红失败"),
    ("M3 源指纹漂移改成不报（并发写入者改源也照样绿）",
     "        if rec_sha != cur:",
     "        if False:",
     "源卡漂移",
     "用例 4 应报红失败"),
    ("M4 输入没拿到改成 exit 0（拿不到＝通过）",
     "        return EXIT_NO_INPUT\n    except Exception as e:",
     "        return EXIT_OK\n    except Exception as e:",
     "EXIT_NO_INPUT",
     "用例 7/8 应报红失败"),
    ("M5 quality_tier 词表判据被摘掉",
     "    for s, v in bad_tier:",
     "    for s, v in []:",
     "quality_tier",
     "用例 6 应报红失败"),
]


def _battery(root: Path) -> list[tuple[str, Path, Path]]:
    """探针电池：**每条判据配一份篡改夹具**。

    ⚠️ 首版只跑「干净夹具」⇒ 五条变异**全部**被判「没改变真实取值」而告负 ——
    因为干净夹具上判据从不触发，摘掉它读数当然不变。这不是变异没劲，是**探针没覆盖**。
    这与本仓库 `run_phase6_gates.py --mutate` 踩过的坑同型（变异体没跑起来，
    读数却像「用例是摆设」）；此处是它的镜像：变异体跑起来了，但探针没走到那条分支。
    """
    base = root / "base"
    p2s, vault = _fixture(base)
    d = p2s / "staging" / "供应与履约" / "p2s-sample-card"
    sk = d / "SKILL.md"

    def clone(name: str) -> tuple[Path, Path, Path]:
        dst = root / name
        shutil.copytree(base, dst)
        return dst / "pkg", dst / "vault", dst / "pkg" / "staging" / "供应与履约" / "p2s-sample-card"

    # 先让 base 完成一次换底，作为「干净」探针的起点
    cli(["--apply", "--p2s-root", str(p2s), "--vault", str(vault)],
        _stdout=io.StringIO(), _stderr=io.StringIO())
    probes = [("P1 干净（已换底）", p2s, vault)]
    _ = sk

    # P2 引文被删一条 ⇒ 证据链截断
    p2, v2, d2 = clone("p2")
    lines = d2.joinpath("SKILL.md").read_text(encoding="utf-8").split("\n")
    for i, l in enumerate(lines):
        if RE_QUOTE.match(l):
            del lines[i:i + 2]
            break
    d2.joinpath("SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    probes.append(("P2 引文被删", p2, v2))

    # P3 SKILL.md 超 12KB
    p3, v3, d3 = clone("p3")
    d3.joinpath("SKILL.md").write_text(
        d3.joinpath("SKILL.md").read_text(encoding="utf-8") + "x" * (MAX_SKILL_BYTES + 50),
        encoding="utf-8")
    probes.append(("P3 超 12KB", p3, v3))

    # P4 源卡在换底后被并发写入者改了
    p4, v4, _d4 = clone("p4")
    vf = v4 / "03-时间序列" / "Skill-Sample-Card.md"
    vf.write_text(vf.read_text(encoding="utf-8") + "\n<!-- 并发写入者改了一笔 -->\n",
                  encoding="utf-8")
    probes.append(("P4 源漂移", p4, v4))

    # P5 quality_tier 被抹掉
    p5, v5, d5 = clone("p5")
    d5.joinpath("SKILL.md").write_text(
        re.sub(r'^quality_tier:.*\n', '',
               d5.joinpath("SKILL.md").read_text(encoding="utf-8"), flags=re.M),
        encoding="utf-8")
    probes.append(("P5 缺 quality_tier", p5, v5))

    # P6 输入没拿到
    probes.append(("P6 缺输入", root / "nope-p2s", root / "nope-vault"))

    # P7 staging 卡顶到 12288（装机余量判据的**唯一**承重探针）
    # ⚠️ 没有它，`INSTALL_HEADROOM_BYTES = 0` 这条变异在六个探针上读数全不变 ——
    #    因为那些探针的卡都远在预算之下。变异测试实测把这一点打了出来。
    p7, v7, d7 = clone("p7")
    f7 = d7 / "SKILL.md"
    t7 = f7.read_text(encoding="utf-8")
    padding = 12288 - len(t7.encode())
    if padding > 0:
        f7.write_text(t7 + "x" * padding, encoding="utf-8")
    probes.append(("P7 staging 顶到 12288", p7, v7))
    return probes


def mutate(out, err) -> int:
    """把本脚本改坏，验证端到端用例真的有劲。先证变异生效，再谈判据有没有劲。"""
    src_path = Path(__file__).resolve()
    src = src_path.read_text(encoding="utf-8")
    code_part, marker, table_part = src.partition("\nMUTATIONS = [")
    assert marker, "找不到 MUTATIONS 表起点"

    tmp = Path(tempfile.mkdtemp(prefix="rebase-mutate-"))
    probes = _battery(tmp / "battery")

    def readings(script_py: Path, dest: Path) -> dict:
        rd = {}
        for pname, p2s, vault in probes:
            p = subprocess.run([sys.executable, str(script_py), "--check",
                                "--p2s-root", str(p2s), "--vault", str(vault)],
                               capture_output=True, text=True, cwd=str(dest))
            txt = p.stdout + p.stderr
            rd[pname] = (p.returncode, _readings(txt), _codes(txt))
        return rd

    base_readings = readings(src_path, tmp)
    out.write("变异测试基线读数（探针电池）：\n")
    for k, v in base_readings.items():
        out.write(f"  {k:16} exit={v[0]} 判红={v[1].get('判红')} 判据码={sorted(v[2])}\n")

    results = []
    for name, anchor, repl, obs, expect in MUTATIONS:
        n = code_part.count(anchor)
        if n != 1:
            results.append((name, False, False, f"锚点在代码区出现 {n} 次（必须恰好 1 次）"))
            continue
        variant = code_part.replace(anchor, repl, 1) + marker + table_part
        assert variant != src
        vp = tmp / (re.sub(r"\W+", "-", name)[:36] + ".py")
        vp.write_text(variant, encoding="utf-8")
        v_readings = readings(vp, tmp)
        diff = [k for k in base_readings if base_readings[k] != v_readings.get(k)]
        if not diff:
            results.append((name, False, False,
                            f"**变异没改变任何探针的真实取值**（{len(probes)} 个探针）"
                            f" ⇒ 这条变异不算数"))
            continue
        st = subprocess.run([sys.executable, str(vp), "--selftest"],
                            capture_output=True, text=True, cwd=str(tmp))
        sttxt = st.stdout + st.stderr
        caught = st.returncode != EXIT_OK and ("❌" in sttxt or "Traceback" in sttxt)
        fails = re.findall(r"❌ ([^\n]+?)(?:\s{2,}got=|$)", sttxt)[:3]
        results.append((name, True, caught,
                        f"变异已生效于探针 {diff}"
                        f"（exit {base_readings[diff[0]][0]}→{v_readings[diff[0]][0]}，"
                        f"判据码 {sorted(base_readings[diff[0]][2])}→{sorted(v_readings[diff[0]][2])}）；"
                        f"selftest exit={st.returncode}；失败项：{'；'.join(fails) or '（无）'}"))

    applied = sum(1 for _, ok, _, _ in results if ok)
    caught = sum(1 for _, ok, c, _ in results if ok and c)
    out.write(f"变异测试：{len(results)} 条 · 变异生效 {applied} · 抓住 {caught}\n")
    for name, ok, c, note in results:
        out.write(f"  {'✅' if (ok and c) else '❌'} {name}\n      {note}\n")
    if applied != len(results) or caught != len(results):
        out.write("✗ 有变异没生效或没被抓住 —— 先修变异，再谈判据\n")
        return EXIT_RED
    return EXIT_OK


def _codes(txt: str) -> set:
    return set(re.findall(r"\[([A-Z]\d+)\]", txt))


def _readings(txt: str) -> dict:
    d = {}
    for k in ("staging 卡", "已换底（有 vault_card）", "quality_tier=preview", "未打标",
              "超 staging 预算", "内联逐字引文", "quality_tier 覆盖"):
        m = re.search(re.escape(k) + r"\s+(\S+)", txt)
        if m:
            d[k] = m.group(1)
    m = re.search(r"判红 (\d+) 条", txt)
    d["判红"] = int(m.group(1)) if m else (0 if "✅ 全部判据通过" in txt else -1)
    return d


if __name__ == "__main__":
    try:
        sys.exit(cli())
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERNAL)
