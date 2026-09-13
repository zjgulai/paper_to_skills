#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE6 S5/B11 · 「同名同物」判定器 —— p2s 卡与 vault 精选卡是不是同一张。

背景（为什么需要它）
--------------------
产品侧 `dsh-paper2skills/staging/<域>/<slug>/SKILL.md` 有 1338 张卡，语料
`paper2skills-vault/<域>/Skill-*.md` 有 146 张精选卡。S5 换底的前提是
**「p2s 卡的 `p2s_card_id` 指向的那张 vault 卡，确实是同一张卡」**。
这个前提此前只被「人看了一遍」确认过 —— 那不算判据。

本模块把它变成可复跑的判定器：**脚本 + 退出码**，退出码沿用本仓库口径
    0 = 全过 / 1 = 判红 / 2 = 输入没拿到（≠ 通过）/ 3 = 门禁内部错误（≠ 判红）

六态（每张卡恰好落一态，恒等式进判据）
--------------------------------------
  SAME_KEY                   p2s_card_id 逐字等于某张 vault 卡的 stem      （两边都在）
  RENAMED_SAME               SAME_KEY 且 slug 与 vault 文件名不同型        （改名同物）
  P2S_ONLY_PREVIEW           p2s_card_id 不是任何 vault stem，且无近名关系 （只有预览版）
  VAULT_ONLY                 vault stem 没有被任何 p2s 卡认领               （精选线独有）
  SAME_NAME_DIFFERENT_THING  **名字像但不是同一张** —— 报红
  UNDECIDABLE                名字像，但判不出是不是同一张 —— 报红（逐条列出，不许丢）

判据 I1–I8（每条都能失败；`--selftest` 每条配一份**篡改样本**）
--------------------------------------------------------------
  I1 键等价      p2s_card_id 逐字命中 vault stem ⇒ SAME_KEY
  I2 改名同物    SAME_KEY 且 `p2s-`+norm(slug) ≠ norm(stem) ⇒ RENAMED_SAME（一等输出，不报红）
  I3 同名异物    p2s_card_id 不是 stem，但存在 stem V ⊊ p2s_card_id，**且 V 已被另一张 p2s 卡认领**
                 ⇒ 报红。判据的承重点是最后那半句：V 被别卡认领时，
                 「名字是超串」这件事**不是同一张**的证据，而是**抢了别人名字**的证据。
  I4 未决        同上但 V 未被认领 ⇒ 报红并逐条列出（「查不出」≠「没问题」）
  I5 vault 独有  vault stem 无人认领 ⇒ VAULT_ONLY（一等输出，不是错）
  I6 p2s 独有    无任何名字关系 ⇒ P2S_ONLY_PREVIEW（一等输出，不是错）
  I7 键唯一      重复 slug / 重复 p2s_card_id ⇒ 报红
  I8 恒等式      两边的计数恒等式成立 ⇒ 否则报红（防「静默丢卡」）

反后门纪律（本仓库用血换来的，逐条照做）
----------------------------------------
  · **拿不到输入 ⇒ exit 2**，绝不 exit 0（`--p2s-root`/`--vault` 指空即报）
  · **一个文件都没扫到 ⇒ exit 2**，不是「查过了没问题」
  · **`--check` 只对现场重算**，不读本脚本自己写过的任何产物
  · **豁免必须可见、带到期条件**：`--baseline <json>` 列出已知红项 + `expires_when`，
    报告里单列「已豁免 N 条」，且**转绿时提示请删除豁免**
  · **`cli()` 不硬编码任何路径**：路径一律由 argv / 环境变量给出（本仓库已因
    `cli()` 里写死原文件路径、或 `REPO = parents[2]` 导致变异体从未被跑过而误判两次）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
EXIT_OK = 0
EXIT_RED = 1
EXIT_NO_INPUT = 2
EXIT_INTERNAL = 3

STATE_SAME_KEY = "SAME_KEY"
STATE_RENAMED_SAME = "RENAMED_SAME"
STATE_P2S_ONLY = "P2S_ONLY_PREVIEW"
STATE_VAULT_ONLY = "VAULT_ONLY"
STATE_DIFFERENT = "SAME_NAME_DIFFERENT_THING"
STATE_UNDECIDABLE = "UNDECIDABLE"

RED_STATES = (STATE_DIFFERENT, STATE_UNDECIDABLE)

DEFAULT_P2S_ROOT = "../magpie-horch/packages/capabilities/dsh-paper2skills/staging"
DEFAULT_VAULT = "paper2skills-vault"

#: 名字近似判据的**唯一**尺度：去掉 `skill-` / `p2s-` 前缀后，只留 alnum 与汉字，小写。
#: 刻意不做模糊相似度（difflib / 编辑距离）—— 本仓库已因「阈值不是数据选出来的」翻过车，
#: 这里只用一条**有语义的**关系：**真子串**（`V` 是 `p2s_card_id` 去掉前缀后的真子串）。
_NORM_STRIP = re.compile(r"^(?:p2s-|skill-)+", re.I)
_NORM_KEEP = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


def norm_name(s: str) -> str:
    """把卡名归一成可比形态。只去前缀、大小写、非字母数字汉字。"""
    s = _NORM_STRIP.sub("", s or "")
    return _NORM_KEEP.sub("", s.lower())


def _fm_fields(text: str) -> dict:
    """取最简 frontmatter 的 `key: value` 标量（与产品侧 parseFrontmatter 同口径：不猜嵌套）。"""
    m = re.match(r"^---\r?\n(.*?)\r?\n---", text, re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1] in (" ", "\t") or line.lstrip().startswith("- "):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            v = v[1:-1]
        out[k.strip()] = v
    return out


# --------------------------------------------------------------------------- #
# 读入
# --------------------------------------------------------------------------- #
class NoInput(Exception):
    """输入没拿到 —— 由调用方转成 exit 2。**不是「通过」。**"""


def load_p2s(p2s_root: Path) -> list[dict]:
    """读产品侧 staging 下所有 `<slug>/SKILL.md`。空集 ⇒ NoInput。"""
    if not p2s_root.is_dir():
        raise NoInput(f"p2s staging 目录不存在：{p2s_root}")
    cards = []
    for p in sorted(p2s_root.rglob("SKILL.md")):
        if "backup" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        f = _fm_fields(text)
        cid = f.get("p2s_card_id")
        if not cid:
            cards.append({"slug": p.parent.name, "dir": str(p.parent), "path": str(p),
                          "card_id": None, "title": f.get("title", ""),
                          "l1_l2_l3": f.get("l1_l2_l3", ""), "no_id": True})
            continue
        cards.append({"slug": p.parent.name, "dir": str(p.parent), "path": str(p),
                      "card_id": cid, "title": f.get("title", ""),
                      "l1_l2_l3": f.get("l1_l2_l3", ""), "no_id": False})
    if not cards:
        raise NoInput(f"扫到 0 张 p2s 卡：{p2s_root}（「没东西可查」不是「查过了没问题」）")
    return cards


def load_vault(vault: Path) -> dict[str, str]:
    """vault 里所有 `Skill-*.md` 的 stem → 相对路径。空集 ⇒ NoInput。"""
    if not vault.is_dir():
        raise NoInput(f"vault 目录不存在：{vault}")
    out = {}
    for p in sorted(vault.rglob("Skill-*.md")):
        if ".git" in p.parts or "node_modules" in p.parts:
            continue
        out[p.stem] = str(p.relative_to(vault.parent))
    if not out:
        raise NoInput(f"扫到 0 张 vault 卡：{vault}")
    return out


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
def classify(p2s: list[dict], vault: dict[str, str]) -> dict:
    """跑 I1–I8。返回 {records, problems, counts}。

    判据的**颗粒度**刻意选在「一张卡」上，而不是「一对名字」上 —— 本仓库已发生过
    「按列判 vs 按行判，两次都是假的那一面」；一张卡恰落一态，恒等式（I8）才能承重。
    """
    problems: list[dict] = []
    stems = set(vault)
    by_norm: dict[str, list[str]] = {}
    for s in stems:
        by_norm.setdefault(norm_name(s), []).append(s)

    # I7 键唯一
    seen_slug, seen_id = {}, {}
    for c in p2s:
        slug = c["slug"]
        if slug in seen_slug:
            problems.append({"code": "I7", "kind": "DUP_SLUG", "slug": slug,
                             "detail": f"slug 重复：{seen_slug[slug]} 与 {c['path']}"})
        seen_slug[slug] = c["path"]
        cid = c.get("card_id")
        if cid:
            if cid in seen_id:
                problems.append({"code": "I7", "kind": "DUP_CARD_ID", "slug": slug,
                                 "detail": f"p2s_card_id 重复：{cid}（{seen_id[cid]} 与 {c['path']}）"})
            seen_id[cid] = c["path"]
        elif c.get("no_id"):
            problems.append({"code": "I7", "kind": "NO_CARD_ID", "slug": slug,
                             "detail": "卡里没有 p2s_card_id —— 无从建联"})

    # 先算「哪些 vault stem 被认领」—— I3 的承重半句要用它
    claimed_by: dict[str, str] = {}
    for c in p2s:
        cid = c.get("card_id")
        if cid and cid in stems:
            claimed_by.setdefault(cid, c["slug"])

    records = []
    for c in p2s:
        cid = c.get("card_id")
        rec = {"slug": c["slug"], "card_id": cid, "title": c.get("title", ""),
               "path": c["path"], "vault_path": None, "vault_stem": None,
               "substring_of": [], "substring_claimed_by": {}, "state": None}
        if cid and cid in stems:
            rec["state"] = STATE_SAME_KEY
            rec["vault_stem"] = cid
            rec["vault_path"] = vault[cid]
            if norm_name(c["slug"]) != norm_name(cid):
                rec["state"] = STATE_RENAMED_SAME
        else:
            # I3/I4：p2s_card_id 不是 stem。找「真子串」关系。
            cl = norm_name(cid or "")
            hits = [s for s in stems if norm_name(s) and norm_name(s) != cl
                    and norm_name(s) in cl]
            rec["substring_of"] = sorted(hits)
            rec["substring_claimed_by"] = {s: claimed_by[s] for s in hits if s in claimed_by}
            claimed_hits = sorted(s for s in hits if s in claimed_by)
            if claimed_hits:
                # **承重半句**：那个近名卡已被另一张 p2s 卡认领 ⇒ 本卡是在抢名，不是同一张
                # ⚠️ 用 `claimed_hits[0]` 而不是 `next(gen)`：后者在「判据被改坏」时抛
                #    StopIteration 直接崩掉，报告里只剩 traceback —— 变异测试实测撞出。
                rec["state"] = STATE_DIFFERENT
                problems.append({
                    "code": "I3", "kind": "SAME_NAME_DIFFERENT_THING", "slug": c["slug"],
                    "detail": (f"p2s_card_id「{cid}」在 vault 里不存在；"
                               f"近名卡「{claimed_hits[0]}」"
                               f"已被 p2s-{claimed_by[claimed_hits[0]]} 认领 —— "
                               f"名字是超串 ≠ 同一张卡"),
                })
            elif hits:
                rec["state"] = STATE_UNDECIDABLE
                problems.append({
                    "code": "I4", "kind": "UNDECIDABLE", "slug": c["slug"],
                    "detail": (f"p2s_card_id「{cid}」不是任何 vault stem，"
                               f"但与 {hits} 同型且无人认领 —— 判不出是不是同一张，不许丢弃"),
                })
            else:
                rec["state"] = STATE_P2S_ONLY
        records.append(rec)

    # I5 vault 独有
    vault_only = sorted(stems - set(claimed_by))
    for s in vault_only:
        records.append({"slug": None, "card_id": None, "title": "", "path": None,
                        "vault_path": vault[s], "vault_stem": s, "substring_of": [],
                        "substring_claimed_by": {}, "state": STATE_VAULT_ONLY})

    counts = {}
    for r in records:
        counts[r["state"]] = counts.get(r["state"], 0) + 1

    # I8 恒等式（两边的分母都必须闭合）
    lhs = (counts.get(STATE_SAME_KEY, 0) + counts.get(STATE_RENAMED_SAME, 0)
           + counts.get(STATE_DIFFERENT, 0) + counts.get(STATE_UNDECIDABLE, 0)
           + counts.get(STATE_P2S_ONLY, 0))
    if lhs != len(p2s):
        problems.append({"code": "I8", "kind": "P2S_SIDE_NOT_CLOSED", "slug": "-",
                         "detail": f"p2s 侧不闭合：{lhs} != {len(p2s)}"})
    rhs = (counts.get(STATE_SAME_KEY, 0) + counts.get(STATE_RENAMED_SAME, 0)
           + counts.get(STATE_VAULT_ONLY, 0) + counts.get(STATE_UNDECIDABLE, 0))
    if rhs != len(stems):
        problems.append({"code": "I8", "kind": "VAULT_SIDE_NOT_CLOSED", "slug": "-",
                         "detail": f"vault 侧不闭合：{rhs} != {len(stems)}"})
    # 改名同物必须是键等价的一个子集（不许凭空多出来）
    if counts.get(STATE_RENAMED_SAME, 0) > counts.get(STATE_SAME_KEY, 0):
        problems.append({"code": "I8", "kind": "RENAMED_NOT_SUBSET", "slug": "-",
                         "detail": "RENAMED_SAME 不是 SAME_KEY 的子集"})

    return {"records": records, "problems": problems, "counts": counts,
            "p2s_total": len(p2s), "vault_total": len(stems),
            "vault_only": vault_only,
            "claimed": claimed_by}


def apply_baseline(problems: list[dict], baseline: dict | None) -> tuple[list[dict], list[dict]]:
    """把已知红项按 `--baseline` 豁免。豁免**可见**（返回豁免名单）且**带到期条件**。"""
    if not baseline:
        return problems, []
    keys = set()
    for it in baseline.get("items", []):
        keys.add((it.get("code"), it.get("kind"), it.get("slug")))
    kept, waived = [], []
    for p in problems:
        if (p["code"], p["kind"], p.get("slug")) in keys:
            waived.append(p)
        else:
            kept.append(p)
    return kept, waived


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def render(result: dict, waived: list[dict], baseline: dict | None) -> str:
    c = result["counts"]
    L = []
    L.append("PHASE6 S5/B11 · 「同名同物」判定（p2s 卡 × vault 精选卡）")
    L.append(f"  p2s 卡       {result['p2s_total']}")
    L.append(f"  vault 卡     {result['vault_total']}")
    L.append("")
    L.append("【六态】")
    order = [STATE_SAME_KEY, STATE_RENAMED_SAME, STATE_P2S_ONLY, STATE_VAULT_ONLY,
             STATE_DIFFERENT, STATE_UNDECIDABLE]
    for s in order:
        L.append(f"  {s:26} {c.get(s, 0)}")
    L.append("")
    L.append("【两边都有的交集】")
    both = c.get(STATE_SAME_KEY, 0) + c.get(STATE_RENAMED_SAME, 0)
    L.append(f"  键等价（p2s_card_id 逐字 == vault stem）  {both}"
             f"   其中改名同物 {c.get(STATE_RENAMED_SAME, 0)}")
    L.append(f"  vault 独有（无 p2s 卡认领）              {c.get(STATE_VAULT_ONLY, 0)}")
    L.append(f"  只有预览版（无 vault 卡）                {c.get(STATE_P2S_ONLY, 0)}")
    L.append("")
    red = [p for p in result["problems"]]
    if red:
        L.append(f"🔴 判红 {len(red)} 条：")
        for p in red[:40]:
            L.append(f"  · [{p['code']}/{p['kind']}] {p.get('slug')} —— {p['detail']}")
        if len(red) > 40:
            L.append(f"  …（共 {len(red)} 条）")
    else:
        L.append("✅ 全部判据通过")
    if waived:
        L.append("")
        L.append(f"⚠️ 已豁免 {len(waived)} 条（**可见豁免，不是静默放行**）")
        for p in waived:
            L.append(f"  · 豁免 [{p['code']}/{p['kind']}] {p.get('slug')} —— {p['detail']}")
        L.append(f"  到期条件：{baseline.get('expires_when', '（未写 —— 必须补）')}")
        L.append("  ⇒ **本条一旦转绿，请从 baseline 里删掉它**（永久豁免＝台账 #5 那种腐烂）")
    return "\n".join(L)


def summarise(result: dict) -> dict:
    c = result["counts"]
    return {
        "p2s_total": result["p2s_total"],
        "vault_total": result["vault_total"],
        "counts": c,
        "same_key_incl_renamed": c.get(STATE_SAME_KEY, 0) + c.get(STATE_RENAMED_SAME, 0),
        "renamed_same": c.get(STATE_RENAMED_SAME, 0),
        "vault_only": c.get(STATE_VAULT_ONLY, 0),
        "p2s_only": c.get(STATE_P2S_ONLY, 0),
        "red": [p for p in result["problems"]],
    }


# --------------------------------------------------------------------------- #
# CLI —— **不硬编码任何路径**：路径只来自 argv / 环境变量
# --------------------------------------------------------------------------- #
def cli(argv: list[str] | None = None, *, _stdout=None, _stderr=None) -> int:
    out = _stdout or sys.stdout
    err = _stderr or sys.stderr
    ap = argparse.ArgumentParser(prog="check_card_identity.py", add_help=True)
    ap.add_argument("--check", action="store_true", help="对**现场**重算（默认动作）")
    ap.add_argument("--p2s-root", default=os.environ.get("P2S_STAGING", DEFAULT_P2S_ROOT))
    ap.add_argument("--vault", default=os.environ.get("P2S_VAULT", DEFAULT_VAULT))
    ap.add_argument("--baseline", default=None, help="可见豁免清单（json）")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--list-red", action="store_true", help="逐条打印判红项（默认只打前 40）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest(out)
    if a.mutate:
        return mutate(out, err)

    try:
        p2s = load_p2s(Path(a.p2s_root))
        vault = load_vault(Path(a.vault))
    except NoInput as e:
        err.write(f"✗ 输入没拿到：{e}\n  退出码 2 = 没测，不是通过。\n")
        return EXIT_NO_INPUT
    except Exception as e:  # pragma: no cover - 门禁自身出错与「判红」必须分开
        err.write(f"✗ 门禁内部错误：{type(e).__name__}: {e}\n")
        return EXIT_INTERNAL

    result = classify(p2s, vault)
    baseline = None
    if a.baseline:
        bp = Path(a.baseline)
        if not bp.is_file():
            err.write(f"✗ 输入没拿到：baseline 不存在 {bp}\n")
            return EXIT_NO_INPUT
        baseline = json.loads(bp.read_text(encoding="utf-8"))
    kept, waived = apply_baseline(result["problems"], baseline)
    result["problems"] = kept

    out.write(render(result, waived, baseline or {}) + "\n")
    if a.list_red:
        for p in kept:
            out.write(f"RED\t{p['code']}\t{p['kind']}\t{p.get('slug')}\t{p['detail']}\n")
    if a.json_out:
        doc = summarise(result)
        doc["waived"] = waived
        doc["baseline"] = baseline.get("id") if baseline else None
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return EXIT_RED if kept else EXIT_OK


# --------------------------------------------------------------------------- #
# 夹具（selftest / mutate 共用）：**不复用被测函数的任何判定逻辑**
# --------------------------------------------------------------------------- #
def _write_fixture(root: Path, *, p2s: list[tuple], vault: list[str]) -> tuple[Path, Path]:
    pr = root / "staging"
    vr = root / "vault"
    for slug, cid in p2s:
        d = pr / ("dom" if not slug.startswith("z") else "dom2") / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            "---\n"
            f'name: "{slug}"\n'
            f'p2s_card_id: "{cid}"\n'
            f'title: "t-{slug}"\n'
            "---\n\n# body\n", encoding="utf-8")
    for stem in vault:
        d = vr / "01-域"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.md").write_text(f"---\ntitle: t-{stem}\n---\n\n# {stem}\n", encoding="utf-8")
    return pr, vr


def _run_cli_in_process(p2s_root: Path, vault: Path, baseline: Path | None = None) -> tuple[int, str]:
    """端到端跑真 CLI（argparse → 读盘 → 判定 → 退出码），不是只调库函数。"""
    import io
    buf = io.StringIO()
    argv = ["--check", "--p2s-root", str(p2s_root), "--vault", str(vault)]
    if baseline:
        argv += ["--baseline", str(baseline)]
    code = cli(argv, _stdout=buf, _stderr=buf)
    return code, buf.getvalue()


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest(out) -> int:
    """每条判据一份**篡改样本**：样本必须被判红/判出对应态，否则该判据没在承重。

    ⚠️ 端到端：全部通过 `cli()` 跑真命令行，不 import 库函数当替身。
    """
    cases = []
    tmp = Path(tempfile.mkdtemp(prefix="cci-selftest-"))

    def case(name, got, want, note=""):
        cases.append((name, got == want, f"got={got!r} want={want!r} {note}"))

    # --- 用例 1：SAME_KEY + RENAMED_SAME -------------------------------------
    r = tmp / "c1"
    pr, vr = _write_fixture(r,
                            p2s=[("p2s-foo", "Skill-Foo"), ("p2s-bar-baz", "Skill-Bar")],
                            vault=["Skill-Foo", "Skill-Bar"])
    code, txt = _run_cli_in_process(pr, vr)
    case("c1 全过 exit 0", code, EXIT_OK, txt[-120:])
    case("c1 SAME_KEY=1", "SAME_KEY" in txt and re.search(r"SAME_KEY\s+1", txt) is not None, True)
    case("c1 RENAMED_SAME=1（p2s-bar-baz ↔ Skill-Bar）",
         re.search(r"RENAMED_SAME\s+1", txt) is not None, True)

    # --- 用例 2（I3 篡改样本）：同名异物必须报红 ----------------------------
    # 名字是超串，且近名卡已被**另一张** p2s 卡认领
    r = tmp / "c2"
    pr, vr = _write_fixture(r,
                            p2s=[("p2s-agent-time-series-forecasting", "Skill-Agent-Time-Series-Forecasting"),
                                 ("p2s-time-series-forecasting", "Skill-Time-Series-Forecasting")],
                            vault=["Skill-Time-Series-Forecasting"])
    code, txt = _run_cli_in_process(pr, vr)
    case("c2 同名异物判红 exit 1", code, EXIT_RED, txt[-160:])
    # ⚠️ 不许用 `"SAME_NAME_DIFFERENT_THING" in txt` —— 六态表**每次都打印全部六个状态名**，
    #    这种断言恒真。变异测试 M2 实测撞出：`"UNDECIDABLE" in txt` 在把未决改判成无事的
    #    变异体上照样为真。必须断言**计数**。
    case("c2 计数表里 SAME_NAME_DIFFERENT_THING=1",
         _extract_counts(txt).get(STATE_DIFFERENT), 1)
    # 承重断言：`Skill-Time-Series-Forecasting` **只**被它真正的那张卡认领，
    # 抢名的那张卡不许也落进 SAME_KEY（否则换底会把它接到别人的卡上）
    _c2 = classify(load_p2s(pr), load_vault(vr))
    _by_slug = {r["slug"]: r["state"] for r in _c2["records"] if r["slug"]}
    case("c2 抢名的卡不落 SAME_KEY",
         _by_slug.get("p2s-agent-time-series-forecasting"),
         STATE_DIFFERENT)
    case("c2 真卡仍落 SAME_KEY",
         _by_slug.get("p2s-time-series-forecasting"), STATE_SAME_KEY)
    case("c2 SAME_KEY 恰好 1 张（没有把别人的卡也算成同物）",
         _c2["counts"].get(STATE_SAME_KEY), 1)

    # --- 用例 3（I4 篡改样本）：近名但无人认领 ⇒ 未决也必须报红 ------------
    r = tmp / "c3"
    pr, vr = _write_fixture(r,
                            p2s=[("p2s-agent-time-series-forecasting", "Skill-Agent-Time-Series-Forecasting")],
                            vault=["Skill-Time-Series-Forecasting"])
    code, txt = _run_cli_in_process(pr, vr)
    case("c3 未决报红 exit 1", code, EXIT_RED, txt[-160:])
    case("c3 计数表里 UNDECIDABLE=1", _extract_counts(txt).get(STATE_UNDECIDABLE), 1)
    case("c3 未被改判成「只有预览版」", _extract_counts(txt).get(STATE_P2S_ONLY), 0)

    # --- 用例 4：只有预览版 / vault 独有，都是**一等输出**，不是错 ---------
    r = tmp / "c4"
    pr, vr = _write_fixture(r, p2s=[("p2s-unrelated-thing", "Skill-Unrelated-Thing")],
                            vault=["Skill-Only-In-Vault"])
    code, txt = _run_cli_in_process(pr, vr)
    case("c4 exit 0（独有不是错）", code, EXIT_OK, txt[-120:])
    case("c4 P2S_ONLY_PREVIEW=1", re.search(r"P2S_ONLY_PREVIEW\s+1", txt) is not None, True)
    case("c4 VAULT_ONLY=1", re.search(r"VAULT_ONLY\s+1", txt) is not None, True)

    # --- 用例 5：输入没拿到 ⇒ exit 2（**不是 0**） -------------------------
    code, txt = _run_cli_in_process(tmp / "nope", tmp / "nope2")
    case("c5 输入没有 exit 2", code, EXIT_NO_INPUT, txt[-120:])

    # --- 用例 6：vault 扫到 0 张 ⇒ exit 2 ----------------------------------
    r = tmp / "c6"
    pr, vr = _write_fixture(r, p2s=[("p2s-x", "Skill-X")], vault=[])
    code, txt = _run_cli_in_process(pr, vr)
    case("c6 vault 0 张 exit 2", code, EXIT_NO_INPUT, txt[-120:])

    # --- 用例 7（I7 篡改样本）：p2s_card_id 重复 --------------------------
    r = tmp / "c7"
    pr, vr = _write_fixture(r, p2s=[("p2s-a", "Skill-A"), ("p2s-b", "Skill-A")],
                            vault=["Skill-A"])
    code, txt = _run_cli_in_process(pr, vr)
    case("c7 重复 id 判红", code, EXIT_RED, txt[-160:])
    case("c7 报 DUP_CARD_ID", "DUP_CARD_ID" in txt, True)

    # --- 用例 8：没有 p2s_card_id 的卡不许静默略过 -------------------------
    r = tmp / "c8"
    pr, vr = _write_fixture(r, p2s=[("p2s-noid", None)], vault=["Skill-A"])
    d = pr / "dom" / "p2s-noid" / "SKILL.md"
    d.write_text('---\nname: "p2s-noid"\n---\n\n# x\n', encoding="utf-8")
    code, txt = _run_cli_in_process(pr, vr)
    case("c8 缺 id 判红", code, EXIT_RED, txt[-160:])
    case("c8 报 NO_CARD_ID", "NO_CARD_ID" in txt, True)

    # --- 用例 9（I8）：恒等式 —— 强行把一张卡的状态改掉，计数必须不闭合 ----
    p2s = [{"slug": "p2s-a", "card_id": "Skill-A", "path": "x", "title": "", "no_id": False}]
    vault = {"Skill-A": "v/Skill-A.md", "Skill-B": "v/Skill-B.md"}
    res = classify(p2s, vault)
    broken = dict(res["counts"])
    broken[STATE_VAULT_ONLY] = broken.get(STATE_VAULT_ONLY, 0) + 1
    closed = (broken.get(STATE_SAME_KEY, 0) + broken.get(STATE_RENAMED_SAME, 0)
              + broken.get(STATE_VAULT_ONLY, 0) + broken.get(STATE_UNDECIDABLE, 0)) == len(vault)
    case("c9 恒等式对篡改样本真的不闭合", closed, False)

    # --- 用例 10：豁免可见 + 到期条件必须打印 ------------------------------
    r = tmp / "c10"
    pr, vr = _write_fixture(r,
                            p2s=[("p2s-agent-time-series-forecasting", "Skill-Agent-Time-Series-Forecasting"),
                                 ("p2s-time-series-forecasting", "Skill-Time-Series-Forecasting")],
                            vault=["Skill-Time-Series-Forecasting"])
    bl = r / "baseline.json"
    bl.write_text(json.dumps({
        "id": "selftest-fixture",
        "expires_when": "该卡在 vault 侧的 id 被修正后",
        "items": [{"code": "I3", "kind": "SAME_NAME_DIFFERENT_THING",
                   "slug": "p2s-agent-time-series-forecasting"}]}, ensure_ascii=False), encoding="utf-8")
    code, txt = _run_cli_in_process(pr, vr, baseline=bl)
    case("c10 豁免后 exit 0", code, EXIT_OK, txt[-200:])
    case("c10 豁免可见", "已豁免 1 条" in txt, True)
    case("c10 到期条件被打印", "到期条件：该卡在 vault 侧的 id 被修正后" in txt, True)
    case("c10 提示转绿后删豁免", "请从 baseline 里删掉它" in txt, True)

    # --- 用例 11：豁免不是「全放行」—— 未登记的错项仍必须判红 --------------
    # `p2s-dup` 与 `p2s-dup2` 共用同一个 p2s_card_id ⇒ I7 报红，而 baseline 里没有它
    r = tmp / "c11"
    pr, vr = _write_fixture(r,
                            p2s=[("p2s-agent-time-series-forecasting", "Skill-Agent-Time-Series-Forecasting"),
                                 ("p2s-dup", "Skill-Duplicated-Xyz"),
                                 ("p2s-dup2", "Skill-Duplicated-Xyz"),
                                 ("p2s-time-series-forecasting", "Skill-Time-Series-Forecasting")],
                            vault=["Skill-Time-Series-Forecasting"])
    code, txt = _run_cli_in_process(pr, vr, baseline=bl)
    case("c11 豁免之外的项仍判红", code, EXIT_RED, txt[-200:])
    case("c11 豁免项确实被豁免了（不是整体失效）", "已豁免 1 条" in txt, True)
    case("c11 未豁免的 DUP_CARD_ID 仍在红名单里", "DUP_CARD_ID" in txt.split("已豁免")[0], True)

    # --- 用例 12：归一化本身 —— 只去前缀/大小写/符号，不做模糊 ------------
    case("c12 norm 去 p2s-/skill-/大小写/符号",
         norm_name("p2s-Agent_Time Series-Forecasting"),
         norm_name("Skill-Agent-Time-Series-Forecasting"))
    case("c12 norm 保留汉字", norm_name("Skill-Monodense-单品价格弹性估计"), "monodense单品价格弹性估计")
    case("c12 norm 不做模糊：编辑距离不同的名字必须不等",
         norm_name("Skill-Agent-Time-Series-Forecasting") == norm_name("Skill-Time-Series-Forecasting"),
         False)

    # --- 输出 --------------------------------------------------------------
    bad = [c for c in cases if not c[1]]
    out.write(f"check_card_identity --selftest：{len(cases) - len(bad)}/{len(cases)} 通过\n")
    for name, ok, note in cases:
        out.write(f"  {'✅' if ok else '❌'} {name}" + ("" if ok else f"   {note}") + "\n")
    if bad:
        out.write(f"✗ {len(bad)} 条用例失败\n")
        return EXIT_RED
    return EXIT_OK


# --------------------------------------------------------------------------- #
# mutate：先证明变异**真的改变了真实取值**，再谈判据有没有劲
# --------------------------------------------------------------------------- #
#: (名称, 原锚点, 替换文本, 期望「变异确实生效」的证据串, 期望行为)
#: 每条的 `changed_marker` 是**运行时真实取值**的差异证据 —— 不是「文件被改了」。
MUTATIONS = [
    ("M1 I3 的承重半句被拿掉（不再要求近名卡被别人认领）",
     "            if claimed_hits:",
     "            if True  or claimed_hits:",
     "SAME_NAME_DIFFERENT_THING",
     "同名异物用例应当从「判红 I3」变成别的东西 ⇒ 用例 c2 抓住"),
    ("M2 I4 未决改判成「无事」（把「查不出」当「没问题」）",
     "                rec[\"state\"] = STATE_UNDECIDABLE",
     "                rec[\"state\"] = STATE_P2S_ONLY",
     "UNDECIDABLE",
     "用例 c3 应报红失败"),
    ("M3 输入没拿到改成 exit 0（拿不到＝通过）",
     "        return EXIT_NO_INPUT\n    except Exception as e:",
     "        return EXIT_OK\n    except Exception as e:",
     "EXIT_NO_INPUT",
     "用例 c5/c6 应报红失败"),
    # ⚠️ 原本想把 `if lhs != len(p2s):` 直接改成 `if False:` —— 实测**变异不生效**：
    #    现场语料与六态夹具上 lhs 本来就等于 len(p2s)，恒等式从不触发，读数不变。
    #    「变异没改变真实取值」时判据是否承重无从判断，故换成**改坏求和式**：
    #    丢掉 P2S_ONLY 一项 ⇒ 现场语料当场多一条 I8 判红，可观测、可被 c4 抓住。
    ("M4 I8 恒等式的求和式被改坏（丢掉「只有预览版」一项）",
     "           + counts.get(STATE_P2S_ONLY, 0))",
     "           + 0)",
     "P2S_SIDE_NOT_CLOSED",
     "用例 c4 应报红失败（不闭合恒等式必须进退出码）"),
    ("M5 改名同物被静默当成普通 SAME_KEY（一等输出消失）",
     "                rec[\"state\"] = STATE_RENAMED_SAME",
     "                rec[\"state\"] = STATE_SAME_KEY",
     "RENAMED_SAME",
     "用例 c1 应报红失败"),
]


def mutate(out, err) -> int:
    """把本脚本改坏，验证端到端用例真的有劲。

    ⚠️ **先证变异生效**：每条先把变异体跑一遍真 CLI，diff 它与原版在**同一批探针**上的读数
    （退出码 + 六态计数）。读数没变的变异 = 变异没施上力，直接判为未抓住。

    探针刻意有**四个**，不只用现场语料 —— 现场语料实测 `UNDECIDABLE = 0`、输入都在，
    只看它的话「把未决改判成无事」「把 exit 2 改成 exit 0」这两条变异**读数不变**，
    于是会被误判成「用例是摆设」。探针覆盖六态夹具 / 缺输入 / 空 vault。

    ⚠️ 变异体放进临时目录后用**绝对路径**驱动探针（本仓库两次栽在
    `cli()` 硬编码原文件路径 / `REPO = parents[2]`：变异体根本没跑起来，读数却像「用例是摆设」）。

    ⚠️ 锚点计数**必须排除 MUTATIONS 表自身** —— 表里逐字写着锚点，首版
    `src.count(anchor)` 因此恒为 2，5 条变异全被判「锚点不唯一」。
    """
    src_path = Path(__file__).resolve()
    src = src_path.read_text(encoding="utf-8")
    # 变异只施在**代码区**：表自身也逐字含锚点，若不切开，count 恒 ≥2
    code_part, marker, table_part = src.partition("\nMUTATIONS = [")
    assert marker, "找不到 MUTATIONS 表的起点 —— 变异器自身坏了"

    here = Path.cwd()
    live_p2s = Path(os.environ.get("P2S_STAGING", here / DEFAULT_P2S_ROOT)).resolve()
    live_vault = Path(os.environ.get("P2S_VAULT", here / DEFAULT_VAULT)).resolve()

    tmp = Path(tempfile.mkdtemp(prefix="cci-mutate-"))
    # 六态夹具：五个都在 / 改名同物 / 抢名 / 未决 / 只有预览 / vault 独有
    fx = tmp / "fx"
    fx_p2s, fx_vault = _write_fixture(
        fx,
        p2s=[("p2s-agent-time-series-forecasting", "Skill-Agent-Time-Series-Forecasting"),
             ("p2s-time-series-forecasting", "Skill-Time-Series-Forecasting"),
             ("p2s-renamed-thing", "Skill-Original-Name"),
             ("p2s-conformal-ts-forecasting", "Skill-Conformal-TS-Forecasting"),
             ("p2s-my-free-standing-thing-x", "Skill-My-Free-Standing-Thing-X"),
             ("p2s-unrelated", "Skill-Unrelated")],
        vault=["Skill-Time-Series-Forecasting", "Skill-Original-Name",
               "Skill-Free-Standing-Thing", "Skill-Lonely"])
    probes = [
        ("现场语料", live_p2s, live_vault),
        ("六态夹具", fx_p2s, fx_vault),
        ("缺输入", tmp / "nope-p2s", tmp / "nope-vault"),
        ("空 vault", fx_p2s, tmp / "empty-vault-dir"),
    ]
    (tmp / "empty-vault-dir").mkdir(parents=True, exist_ok=True)

    def readings(script_py: Path, cwd: Path) -> dict:
        rd = {}
        for pname, pr, vr in probes:
            try:
                p = subprocess.run([sys.executable, str(script_py), "--check",
                                    "--p2s-root", str(pr), "--vault", str(vr)],
                                   capture_output=True, text=True, timeout=600, cwd=str(cwd))
                txt = p.stdout + p.stderr
                # ⚠️ 读数必须含**判红条数**，不能只取退出码 —— 现场语料本来就是红的，
                #    再叠一条 I8 判红时退出码 1→1 不变；首版因此把 M4 误判成「变异没生效」。
                rd[pname] = (p.returncode, _extract_counts(txt), _extract_red_count(txt))
            except subprocess.TimeoutExpired:
                rd[pname] = ("TIMEOUT", {}, -1)
        return rd

    base_readings = readings(src_path, here)
    out.write("变异测试基线读数：\n")
    for k, v in base_readings.items():
        out.write(f"  {k:8} exit={v[0]} 计数={v[1]}\n")

    results = []
    for name, anchor, repl, marker_of, expect in MUTATIONS:
        n = code_part.count(anchor)
        if n != 1:
            results.append((name, False, False,
                            f"锚点在**代码区**出现 {n} 次（必须恰好 1 次，否则变异施不上力）"))
            continue
        variant = code_part.replace(anchor, repl, 1) + marker + table_part
        assert variant != src, "变异没改变文本"
        vp = tmp / (re.sub(r"\W+", "-", name)[:40] + ".py")
        vp.write_text(variant, encoding="utf-8")
        v_readings = readings(vp, tmp)
        diff = [k for k in base_readings if base_readings[k] != v_readings.get(k)]
        if not diff:
            results.append((name, False, False,
                            f"**变异没改变任何探针的真实取值**（{list(base_readings)}）"
                            f" ⇒ 这条变异不算数，判据是否承重无从判断"))
            continue
        try:
            st = subprocess.run([sys.executable, str(vp), "--selftest"],
                                capture_output=True, text=True, timeout=600, cwd=str(tmp))
            stcode, sttxt = st.returncode, st.stdout + st.stderr
        except subprocess.TimeoutExpired:
            results.append((name, True, False, "变异体 selftest 超时（但变异已生效）"))
            continue
        caught = stcode != EXIT_OK and ("❌" in sttxt or "Traceback" in sttxt)
        detail = "；".join(re.findall(r"❌ ([^\n]+?)\s{2,}got=", sttxt)[:3]) or \
                 "；".join(re.findall(r"❌ ([^\n]+)", sttxt)[:3])
        results.append((name, True, caught,
                        f"变异已生效于探针 {diff}（exit {base_readings[diff[0]][0]}→{v_readings[diff[0]][0]}）；"
                        f"selftest exit={stcode}，失败项：{detail}"))

    caught_n = sum(1 for _, ok, c, _ in results if ok and c)
    applied_n = sum(1 for _, ok, _, _ in results if ok)
    out.write(f"变异测试：{len(results)} 条 · 变异生效 {applied_n} · 抓住 {caught_n}\n")
    for name, ok, caught, note in results:
        flag = "✅" if (ok and caught) else "❌"
        out.write(f"  {flag} {name}\n      {note}\n")
    if applied_n != len(results) or caught_n != len(results):
        out.write("✗ 有变异没生效或没被抓住 —— 先修变异，再谈判据\n")
        return EXIT_RED
    return EXIT_OK


def _extract_counts(txt: str) -> dict:
    """从报告的六态表里取计数。**只认计数行**，不认表头。"""
    d = {}
    # ⚠️ 状态名里有数字（`P2S_ONLY_PREVIEW`），首版写成 `[A-Z_]{4,}` ⇒ 该行**整条不被计数**，
    #    取到的是 None 而不是 0，于是「未被改判成只有预览版」这条断言恒假。
    for m in re.finditer(r"^  ([A-Z0-9_]{4,})\s+(\d+)\s*$", txt, re.M):
        d[m.group(1)] = int(m.group(2))
    return d


def _extract_red_count(txt: str) -> int:
    """取「🔴 判红 N 条」的 N；没有判红行时返回 0。"""
    m = re.search(r"判红 (\d+) 条", txt)
    if m:
        return int(m.group(1))
    return 0 if "✅ 全部判据通过" in txt else -1


if __name__ == "__main__":
    try:
        sys.exit(cli())
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERNAL)
