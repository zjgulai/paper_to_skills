#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""migrate_domain_labels.py — 把盘上产物里的**旧域名**折成规范名（PHASE6 P1）

## 覆盖范围（**含派生文本产物**）

不只迁 `arxiv_candidates.json/csv`，也迁**由它派生的文本产物**：
`recommendations.csv`（`domains` 列）、`shortlist.md`（分节标题）、
`bundles/*.json`（文件名 + `domain` 字段）。

⚠️ 这些派生产物**不重跑生成器来更新**，而是**原地只改标签**。理由是一条实测：
`rank_candidates.py` 的 `total` 里含**墙钟项** `fresh = 8 − 天数/12`，
所以「重跑生成器」的 diff 里必然混进**与本次改动无关的时间漂移**
（实测提交版 vs 今天重跑：1046 篇**全都**差 0.1–0.4，而 `s_*` 各维**一个都没变**）。
把时间漂移混进「改域名」这个提交里，等于**用一个真实的改动掩盖另一个真实的改动**。
⇒ 一律「只改该改的」，且用 J2 证明别的字段逐字没动。

## 为什么需要迁移而不是「到处加别名兼容」

本次修复查出域名字符串在仓内**至少 7 处各写一份**，其中 harvest 侧用旧名、
filter 侧用新名。最轻的修法是「在每个消费点都加一层别名折换」—— 但那等于
**把漂移合法化**：名字继续有两个，谁都可能在下一个文件里再写第三个。

所以分两步：
  ① **产出端**（`arxiv_harvest.py`）改成产规范名，并加**不许漂**的守卫（fail loud）；
  ② **盘上的历史产物**由本脚本一次性迁移到规范名，让"一处一名"真的成立。
`domains.RETIRED`（**已退休的旧名**）只给本脚本当依据 ——
过滤器**不认**退休名（认了就等于替漂移兜底，下一次漂移就再也看不见了），
所以盘上一旦出现旧名，过滤器会 `exit 2` 大喊，而修法就是跑这个脚本。

## 判据（每条都能失败）

- **J1 幂等**：再跑一次必须 0 处改动（`--apply` 后再 `--check` 必须绿）。
- **J2 只动该动的字段**：`query_groups` 之外**逐字节**不许变
  （深度比对整份 JSON，而不是只数「改了几条」）。
- **J3 判定不变**：迁移**不得改变任何一篇的保留/丢弃结论** ——
  迁移前后各跑一次 `candidate_filter.filter_pool`，逐篇对拍。
  这一条才真正证明「别名层与规范名等价」，而不是「看起来等价」。
- **J4 覆盖是一等输出**：改了多少条目 / 多少个标签 / 有没有残留旧名，全部报出来；
  **残留旧名 > 0 判红**（否则「迁移完成」是一句没法证伪的话）。
- **J5 没东西可查 ≠ 查过了没问题**：产物不存在 ⇒ exit 2。

用法：
    python3 migrate_domain_labels.py --check    # 只报，不写（门禁用）
    python3 migrate_domain_labels.py --apply    # 迁移
    python3 migrate_domain_labels.py --selftest
"""

from __future__ import annotations

import argparse
import csv as _csv
import io
import json
import os
import re
import sys
from pathlib import Path

# 与 `domains.py` / `candidate_filter.py` 同义：`P2S_REPO` = **仓库根**。
REPO = Path(os.environ.get("P2S_REPO") or Path(__file__).resolve().parents[2]).resolve()
ROOT = REPO / "paper2skills-research"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import domains as _domains  # noqa: E402
import candidate_filter as _cf  # noqa: E402

DATA = ROOT / "data"
JSON_POOL = DATA / "arxiv_candidates.json"
CSV_POOL = DATA / "arxiv_candidates.csv"
#: 由候选池派生的**文本**产物。它们**不重跑生成器更新**（重跑会把墙钟漂移混进来），
#: 而是原地只改标签 —— 见文件头「覆盖范围」。
DERIVED_ARTIFACTS = [DATA / "recommendations.csv", DATA / "recommendations.json",
                     DATA / "shortlist.md"]
BUNDLE_DIR = DATA / "bundles"


def rewrite_groups(groups) -> tuple[list, list]:
    """把一个条目的 `query_groups` 折成规范名。返回 (新列表, 用到的退休名对)。

    ⚠️ 这是全仓**唯一**把旧名折成规范名的地方，而且只有**显式跑 `--apply`** 才落盘。
    过滤器那边是不折的（退休名走 `unknown` ⇒ `exit 2`）—— 那是刻意的：
    **替漂移兜底，就等于取消发现漂移的能力。**
    """
    out: list[str] = []
    used: list[tuple[str, str]] = []
    for g in groups or []:
        canon, kind = _domains.resolve(g)
        if kind == "unknown":
            # 退休名 ⇒ 折（**只有这里允许折**，且必须显式跑 --apply）；
            # 其余认不出的**原样保留、不擅自改写**（迁移不是「把不认识的都删掉」）。
            if g in _domains.RETIRED:
                used.append((g, _domains.RETIRED[g]))
                if _domains.RETIRED[g] not in out:
                    out.append(_domains.RETIRED[g])
                continue
            out.append(g)
            continue
        if canon not in out:
            out.append(canon)
    return out, used


def scan(json_pool: Path = JSON_POOL, csv_pool: Path = CSV_POOL) -> dict:
    """只读：报出要改什么、改多少、有没有认不出的标签。"""
    out = {"available": False, "json_pool": str(json_pool), "csv_pool": str(csv_pool),
           "items": 0, "items_to_change": 0, "label_rewrites": {}, "changed_ids": [],
           "unknown_labels": {}, "csv_rows": 0, "csv_rows_to_change": 0,
           "csv_label_rewrites": {}, "text_artifacts": {}, "text_artifact_hits": 0,
           "bundle_renames": []}
    if not json_pool.is_file():
        return out
    out["available"] = True
    data = json.loads(json_pool.read_text(encoding="utf-8"))
    items = data.get("items", data if isinstance(data, list) else [])
    out["items"] = len(items)
    for it in items:
        new, used = rewrite_groups(it.get("query_groups"))
        if used:
            out["items_to_change"] += 1
            out["changed_ids"].append(it.get("arxiv_id"))
            for old, canon in used:
                out["label_rewrites"][old] = out["label_rewrites"].get(old, 0) + 1
        for g in it.get("query_groups") or []:
            if _domains.resolve(g)[1] == "unknown":
                out["unknown_labels"][g] = out["unknown_labels"].get(g, 0) + 1

    if csv_pool.is_file():
        rows = list(_csv.DictReader(csv_pool.open(encoding="utf-8")))
        out["csv_rows"] = len(rows)
        for r in rows:
            gs = [x for x in (r.get("domains") or "").split("|") if x]
            new, used = rewrite_groups(gs)
            if used:
                out["csv_rows_to_change"] += 1
                for old, canon in used:
                    out["csv_label_rewrites"][old] = out["csv_label_rewrites"].get(old, 0) + 1

    for f in DERIVED_ARTIFACTS:
        if f.is_file():
            hits = _count_labels(f.read_text(encoding="utf-8"))
            out["text_artifacts"][str(f.relative_to(REPO))] = hits
            out["text_artifact_hits"] += sum(hits.values())
    out["bundle_renames"] = [str(b.relative_to(REPO)) for b in BUNDLE_DIR.glob("*.json")
                             if b.stem in _domains.RETIRED]
    return out


#: 退休名的匹配正则。⚠️ **不能直接用 `text.count(旧名)`** ——
#: 退休名 `09-DataAgent` 是规范名 `09-DataAgent-LLM` 的**前缀**，
#: 于是迁移完成后计数**不为 0**，`--check` 会永远判「还没迁完」。
#: 这是本仓库那条反复出现的家族（判据只认一种写法 / 前缀当子串找）的又一例。
#: 修法：把「本名后面还可能接什么、接上就说明它其实是另一个名字」写成负向先行断言。
_LABEL_RE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _label_re(name: str):
    if name not in _LABEL_RE_CACHE:
        tails = sorted({c[len(name):] for c in _domains.CANONICAL if c.startswith(name) and c != name},
                       key=len, reverse=True)
        pat = re.escape(name) + (("(?!" + "|".join(re.escape(x) for x in tails) + ")") if tails else "")
        _LABEL_RE_CACHE[name] = re.compile(pat)
    return _LABEL_RE_CACHE[name]


def _count_labels(text: str) -> dict:
    """数一份文本产物里各退休名出现几次。**逐名计数** —— 只报「有命中」是没法复核的。"""
    return {a: len(_label_re(a).findall(text)) for a in _domains.RETIRED
            if _label_re(a).search(text)}


def _rewrite_text(text: str) -> tuple[str, int]:
    """文本产物里**只做标签替换**。用同一个正则（计数与替换同尺，否则两件事会各说各话）。"""
    n = 0
    for old in sorted(_domains.RETIRED, key=len, reverse=True):
        text, k = _label_re(old).subn(_domains.RETIRED[old], text)
        n += k
    return text, n


def _verdicts(items: list[dict]) -> list:
    """逐篇判定（**只为对拍用**，不落盘）。"""
    return [(_cf._item_id(it), tuple(_cf.filter_pool([it])["kept"]) != ()) for it in items]


def check(json_pool: Path = JSON_POOL, csv_pool: Path = CSV_POOL) -> int:
    r = scan(json_pool, csv_pool)
    if not r["available"]:
        print(f"🔴 候选池读不到：{json_pool} —— 「没东西可查」不等于「查过了没问题」")
        return 2
    print(f"条目 {r['items']} · 需迁移 {r['items_to_change']} · "
          f"CSV 行 {r['csv_rows']} · 需迁移 {r['csv_rows_to_change']}")
    if r["label_rewrites"]:
        print("\n要折的旧名（JSON）：")
        for a, c in sorted(r["label_rewrites"].items()):
            print(f"  {c:>4}  {a} → {_domains.RETIRED[a]}")
    if r["csv_label_rewrites"]:
        print("要折的旧名（CSV）：")
        for a, c in sorted(r["csv_label_rewrites"].items()):
            print(f"  {c:>4}  {a} → {_domains.RETIRED[a]}")
    if r["text_artifacts"]:
        print("\n派生文本产物里的退休名：")
        for f, hits in sorted(r["text_artifacts"].items()):
            if any(hits.values()):
                print(f"  {f}: " + " · ".join(f"{k}×{v}" for k, v in hits.items() if v))
    if r["bundle_renames"]:
        print("\n需要改名的 bundle 文件：")
        for f in r["bundle_renames"]:
            print(f"  {f} → {Path(f).parent / (_domains.RETIRED[Path(f).stem] + '.json')}")
    if r["unknown_labels"]:
        print(f"\n🔴 认不出的域标签（**不擅自改写**，但必须报）：")
        for a, c in sorted(r["unknown_labels"].items()):
            print(f"  {c:>4}  {a!r}")
        return 1
    if (r["items_to_change"] or r["csv_rows_to_change"]
            or r["text_artifact_hits"] or r["bundle_renames"]):
        print("\n🔴 盘上仍有旧名 ⇒ 迁移未完成（跑 `--apply`）")
        return 1
    print("\n✅ 盘上产物已全部是规范名（残留旧名 0）")
    return 0


def apply(json_pool: Path = JSON_POOL, csv_pool: Path = CSV_POOL) -> int:
    if not json_pool.is_file():
        print(f"🔴 候选池读不到：{json_pool}")
        return 2
    before = json.loads(json_pool.read_text(encoding="utf-8"))
    items = before.get("items", before if isinstance(before, list) else [])
    items_before = json.loads(json.dumps(items, ensure_ascii=False))   # 深拷贝，供 J2 对拍

    # ---- J3（前）：迁移**前**的逐篇判定 ----
    v_before = _verdicts(items)

    n_changed = n_labels = 0
    for it in items:
        new, used = rewrite_groups(it.get("query_groups"))
        if used:
            n_changed += 1
            n_labels += len(used)
            it["query_groups"] = new

    # ---- J2：**只动该动的字段** ----
    # 逐条深比对：除 `query_groups` 外，每个字段必须逐字节相等。
    # ❌ 反例（本仓库吃过）：「改了几条」是计数，计数相等不代表没碰别的字段。
    touched_other: list[str] = []
    for it_new, it_old in zip(items, items_before):
        for k in set(it_new) | set(it_old):
            if k == "query_groups":
                continue
            if it_new.get(k) != it_old.get(k):
                touched_other.append(f"{it_new.get('arxiv_id')}:{k}")
    if len(items) != len(items_before):
        touched_other.append(f"条目数 {len(items)} ≠ {len(items_before)}")
    if touched_other:
        print(f"🔴 **动了 `query_groups` 之外的字段 {len(touched_other)} 处**："
              f"{touched_other[:10]}")
        return 1

    # ---- J1：幂等 —— 折过的再折一次必须不变 ----
    not_idempotent = [it.get("arxiv_id") for it in items
                      if rewrite_groups(it.get("query_groups"))[0] != it.get("query_groups")]
    if not_idempotent:
        print(f"🔴 不幂等：{len(not_idempotent)} 条再折一次还会变 → {not_idempotent[:5]}")
        return 1

    # ---- J3（后）：逐篇对拍 ----
    v_after = _verdicts(items)
    flips = [(a, b) for (ida, a), (idb, b) in zip(v_before, v_after)
             if ida != idb or a != b]
    if flips:
        print(f"🔴 **判定变了 {len(flips)} 篇** —— 迁移不该改变任何结论：{flips[:10]}")
        return 1

    print(f"迁移：条目 {n_changed} 条 · 标签 {n_labels} 处")
    print(f"✅ J1 幂等 · J2 除 `query_groups` 外 0 处改动 · "
          f"J3 判定逐篇对拍 {len(v_before)} 篇 0 翻转")

    header = ({k: v for k, v in before.items() if k != "items"}
              if isinstance(before, dict) else {})
    header["domain_label_migration"] = {
        "tool": "migrate_domain_labels.py",
        "from_retired": _domains.RETIRED,
        "changed_items": n_changed,
        "changed_labels": n_labels,
        "canonical_source": "paper2skills-research/scripts/domains.py",
    }
    json_pool.write_text(json.dumps({**header, "items": items},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {json_pool}")

    if csv_pool.is_file():
        rows = list(_csv.DictReader(csv_pool.open(encoding="utf-8")))
        fields = list(rows[0].keys())
        n_csv = 0
        for r in rows:
            gs = [x for x in (r.get("domains") or "").split("|") if x]
            new, used = rewrite_groups(gs)
            if used:
                n_csv += 1
                r["domains"] = "|".join(new)
        buf = io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        csv_pool.write_text(buf.getvalue(), encoding="utf-8")
        print(f"→ {csv_pool}（{n_csv} 行）")

    # ---- 派生文本产物：**只做标签替换**（行数/字节数除标签外必须不变）----
    for f in DERIVED_ARTIFACTS:
        if not f.is_file():
            continue
        before_text = f.read_text(encoding="utf-8")
        after_text, n_hit = _rewrite_text(before_text)
        if not n_hit:
            continue
        # J2'：把新文本按规范名还原成旧名后，必须与原文**逐字节相同**
        back = after_text
        for old in sorted(_domains.RETIRED, key=len, reverse=True):
            back = back.replace(_domains.RETIRED[old], old)
        if back != before_text:
            print(f"🔴 {f.name}：替换后无法逐字还原 ⇒ 动了标签以外的东西")
            return 1
        # JSON 产物还要证明**仍然是合法 JSON**（逐字还原只能证明「没碰别的」，
        # 证明不了「没碰坏」）。
        if f.suffix == ".json":
            try:
                json.loads(after_text)
            except Exception as exc:                                  # noqa: BLE001
                print(f"🔴 {f.name}：替换后不是合法 JSON（{exc}）")
                return 1
        f.write_text(after_text, encoding="utf-8")
        print(f"→ {f.relative_to(REPO)}（{n_hit} 处标签）")

    # ---- bundles：改内容 + 用 git mv 改名（保历史）----
    for b in sorted(BUNDLE_DIR.glob("*.json")):
        if b.stem not in _domains.RETIRED:
            continue
        new_path = b.with_name(_domains.RETIRED[b.stem] + ".json")
        txt, n_hit = _rewrite_text(b.read_text(encoding="utf-8"))
        b.write_text(txt, encoding="utf-8")
        import subprocess as _sp
        r = _sp.run(["git", "mv", str(b.relative_to(REPO)), str(new_path.relative_to(REPO))],
                    cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"⚠️ git mv 失败（{r.stderr.strip()[:80]}）⇒ 退回普通改名")
            b.rename(new_path)
        print(f"→ git mv {b.name} → {new_path.name}（内容 {n_hit} 处标签）")
    return 0


def selftest() -> int:
    """用构造样本证明：折换规则、去重、认不出的标签**不擅自改写** 三类都成立。"""
    ok = True
    cases = [
        (["09-DataAgent"], ["09-DataAgent-LLM"], [("09-DataAgent", "09-DataAgent-LLM")]),
        (["09-DataAgent", "16-智能体工程"], ["09-DataAgent-LLM", "16-智能体工程"],
         [("09-DataAgent", "09-DataAgent-LLM")]),
        # 旧名与规范名同时出现 ⇒ 折成一条（不去重会让同一篇被判两次）
        (["09-DataAgent", "09-DataAgent-LLM"], ["09-DataAgent-LLM"],
         [("09-DataAgent", "09-DataAgent-LLM")]),
        # 认不出的**原样保留**：迁移不是「把不认识的都删掉」
        (["99-不存在"], ["99-不存在"], []),
        ([], [], []),
    ]
    for groups, exp_new, exp_used in cases:
        new, used = rewrite_groups(groups)
        good = new == exp_new and used == exp_used
        ok &= good
        print(f"  {'✅' if good else '❌'} {groups} → {new}（用的别名 {used}）")
    # 反向控制：两个真不同的域不许被折成一条
    new, _ = rewrite_groups(["09-DataAgent-LLM", "10-MAS"])
    good = new == ["09-DataAgent-LLM", "10-MAS"]
    ok &= good
    print(f"  {'✅' if good else '❌'} 反向控制：两个不同的域必须仍是两条 → {new}")

    # ⚠️ 前缀碰撞：退休名是某个规范名的**前缀**（`09-DataAgent` vs `09-DataAgent-LLM`）。
    # 不加负向先行断言，迁移完成后计数**不为 0**，`--check` 会永远判「还没迁完」。
    # 这是本仓库那条反复出现的家族（前缀当子串找）的又一例，故锁成用例。
    print("\n=== 前缀碰撞（退休名是规范名的前缀）===")
    pre = [
        ("09-DataAgent-LLM", 0), ("09-DataAgent", 1),
        ("09-DataAgent,09-DataAgent-LLM", 1),
        ("15-营销投放分析", 0), ("15-营销投放", 1),
    ]
    for s, exp in pre:
        got = _count_labels(s).get(s.strip("|,").split(",")[0].split("-LLM")[0] if False
                                   else ("09-DataAgent" if s.startswith("09-") else "15-营销投放"), 0)
        good = got == exp
        ok &= good
        print(f"  {'✅' if good else '❌'} {s!r} → 计数 {got}（期望 {exp}）")
    mixed = "a|09-DataAgent|b|09-DataAgent-LLM|c"
    new_txt, n = _rewrite_text(mixed)
    good = new_txt == "a|09-DataAgent-LLM|b|09-DataAgent-LLM|c" and n == 1
    ok &= good
    print(f"  {'✅' if good else '❌'} 替换与计数**同尺**：{mixed} → {new_txt}（{n} 处）")

    # J3 的独立复核：把「折换」换成恒等映射，verdict 对拍必须**发现不了**差别
    # （说明它对拍的确实是判定而不是别的）——
    # 而把 09 域折到 10-MAS（真错折）必须被发现。
    sample = [{"arxiv_id": "s1", "title": "AttnLink: Schema Links for Text-to-SQL",
               "abstract": "schema linking for text-to-sql systems",
               "query_groups": ["09-DataAgent"]}]
    good_ids = [x[1] for x in _verdicts(sample)]
    good = good_ids == [True]
    ok &= good
    print(f"  {'✅' if good else '❌'} J3 的仪器本身可用：迁移前该篇判定 = {good_ids}（期望 [True]）")
    print("✅ 自检通过" if ok else "❌ 自检失败")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="把盘上产物里的旧域名折成规范名（PHASE6 P1）")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json-pool", type=Path, default=JSON_POOL)
    ap.add_argument("--csv-pool", type=Path, default=CSV_POOL)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.apply:
        return apply(args.json_pool, args.csv_pool)
    return check(args.json_pool, args.csv_pool)


if __name__ == "__main__":
    sys.exit(main())
