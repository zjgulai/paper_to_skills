#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repo_health.py — 仓库体检（`paper-维护` 的执行体）

把「人工每隔一阵子发现一类脏数据」变成「一条命令跑完、可回归」的检查。

## 每一项检查都对应一次真实事故，不是想象出来的风险

| 检查 | 对应事故 |
|---|---|
| C1 重复卡片 | `07-NLP-VOC/` 下 26 组同名卡片各存两份 |
| C2 frontmatter 完整性 | 130 张卡只有 73 张有 frontmatter、6 张有 `paper:` 溯源字段 |
| C3 硬编码绝对路径 | 22 处 `/Users/pray/project/paper_to_skills` 失效路径 |
| C4 **代码围栏结构** | 9 处伪代码标成 ```python、1 张嵌套 ```、2 张围栏**未闭合** —— 这类缺陷在 K1 里表现为「语法错误」，很容易被误判成代码写错，实际是 markdown 结构问题 |
| C5 registry ↔ 卡片一致性 | registry 记的 `outputs.skill_card` 与磁盘实际不符 |
| C6 门禁汇总 | 需要一眼看到 K1/G1/G2/G3 通过率与最近一次运行时间 |
| C7 空目录 / `__pycache__` 入库 | Python 字节码与空目录混进版本控制 |

用法：
    python3 repo_health.py                 # 全量体检，人读报告
    python3 repo_health.py --json-out x.json
    python3 repo_health.py --only C4       # 只跑某一项
    python3 repo_health.py --selftest      # 自检：用构造样本证明检查真的会报警

退出码：0 = 无 CRITICAL；1 = 有 CRITICAL。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VAULT = REPO / "paper2skills-vault"
RESEARCH = REPO / "paper2skills-research"
REGISTRY = VAULT / "07-资源库" / "papers_registry.json"
GATES = VAULT / "07-资源库" / "gates"
K1_JSON = RESEARCH / "data" / "verification" / "k1_l5.json"

# 允许保留的历史绝对路径（有意保留，不算违规）
# 有意保留的历史绝对路径（见 CLAUDE.md「NLP-VOC 子项目迁出说明」与 T0-3 约定）：
#   paper2skills-code/nlp_voc/ 与 paper2skills-vault/07-NLP-VOC/ 是迁出子项目的
#   代码模板 + 论文档案镜像，其内部硬编码路径是**有意保留**的，不算失效路径。
#   早期白名单只写了 code 侧，导致 vault 侧 290+ 处被误报。
PATH_ALLOWLIST = ("paper2skills-code/nlp_voc/", "paper2skills-vault/07-NLP-VOC/",
                  "logs/", "evolve/", "01-MasterPrompt设计/",
                  "paper-维护/scripts/repo_health.py")
# frontmatter v2 必填字段
FM_REQUIRED = ("title", "module", "paper_id", "paper", "venue", "venue_tier", "evidence_grade")

# ⚠️ 只认**缩进 ≤3 空格**的围栏。
# 按 CommonMark，缩进 4 空格以上的 ``` 不是块级围栏。实测假阳性案例：
# `Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md` 第 257 行是一段
# **文档里演示 prompt 格式**的缩进示例（`    CONTENT: ```python`），
# 早期版本把它当真围栏，导致全文围栏奇偶错位，误报「围栏未闭合」。
# 这属于「门禁自己的 bug」，必须先修 —— 假红灯会让整份体检报告失去可信度。
FENCE_RE = re.compile(r"^( {0,3})(`{3,})([^\n]*)$", re.M)


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def cards() -> list[Path]:
    return sorted(p for p in VAULT.rglob("Skill-*.md")
                  if p.is_file() and "_superseded" not in p.parts)


def _fm(text: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------
def c1_duplicate_cards() -> dict:
    byname = defaultdict(list)
    for c in cards():
        byname[c.name].append(rel(c))
    dups = {k: v for k, v in byname.items() if len(v) > 1}
    return {"id": "C1", "name": "重复卡片", "critical": bool(dups),
            "detail": dups,
            "summary": f"{len(cards())} 张卡，{len(dups)} 组同名重复"}


def c2_frontmatter() -> dict:
    miss_all, miss_v2 = [], []
    for c in cards():
        fm = _fm(c.read_text(encoding="utf-8", errors="replace"))
        if not fm:
            miss_all.append(rel(c))
        else:
            lack = [k for k in FM_REQUIRED if not fm.get(k)]
            if lack:
                miss_v2.append({"card": rel(c), "missing": lack})
    return {"id": "C2", "name": "frontmatter 完整性", "critical": False,
            "detail": {"no_frontmatter": miss_all, "incomplete_v2": miss_v2},
            "summary": (f"无 frontmatter {len(miss_all)} 张；"
                        f"有 frontmatter 但缺 v2 必填字段 {len(miss_v2)} 张")}


def c3_hardcoded_paths() -> dict:
    bad = []
    pat = re.compile(r"/Users/(?:pray|lute)/project/\S+")
    for p in list(REPO.rglob("*.py")) + list(REPO.rglob("*.md")):
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        r = rel(p).replace("\\", "/")
        if any(a in r for a in PATH_ALLOWLIST):
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pat.search(line):
                    bad.append({"file": r, "line": i, "text": line.strip()[:110]})
        except OSError:
            continue
    return {"id": "C3", "name": "硬编码绝对路径", "critical": False,
            "detail": bad[:60], "total": len(bad),
            "summary": f"{len(bad)} 处可执行代码/文档里的绝对路径（历史归档已白名单豁免）"}


def c4_code_fences() -> dict:
    """检查 markdown 代码围栏的**结构**问题。

    ⚠️ 单独做这一项，是因为这类缺陷在 K1 里表现为「语法错误」：
    - 伪代码标成 ```python → K1 报 ast 语法错误（看着像代码写错）
    - 围栏未闭合 → 后面的正文被当成 Python 解析 → 报一堆莫名其妙的语法错误
    - 内层含 ``` 但外层只有 3 个反引号 → 代码块被截成两半，后半段**整个丢失**
    这三类共 12 处，本仓库全部踩过。
    """
    issues = []
    for c in cards():
        text = c.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        stack: list[tuple[int, int, str]] = []      # (行号, 反引号数, 语言)
        indented_fences = []
        for i, line in enumerate(lines, 1):
            m = FENCE_RE.match(line)
            if not m:
                continue
            ticks = len(m.group(2))
            lang = m.group(3).strip()
            if m.group(1):
                # 缩进 1-3 空格的围栏在列表项里合法，但会让「奇偶配对」这类
                # 朴素检查产生歧义。实测案例：Skill-DeepAnalyze 第 259 行是
                # **文档里演示 prompt 格式**的缩进示例，真正的代码块在第 422 行
                # 才闭合 —— 而 K1 能正确抽出并编译那段代码，说明卡片本身没问题。
                # 故**只用零缩进围栏做硬判定**，缩进的单独记为提示。
                indented_fences.append(i)
                continue
            if stack and ticks >= stack[-1][1]:
                opened_at, open_ticks, open_lang = stack.pop()
                # 内层围栏检测：块内出现了 >= 外层长度的围栏（正确写法是外层加长）
                body = "\n".join(lines[opened_at:i - 1])
                # 只把**零缩进**的内层围栏当作真嵌套（缩进的是列表项里的示例）
                inner = re.findall(r"^(`{3,})", body, re.M)
                if inner and max(len(x) for x in inner) >= open_ticks:
                    issues.append({
                        "card": rel(c), "type": "嵌套围栏外层未加长",
                        "line": opened_at, "lang": open_lang,
                        "note": f"块内有 {max(len(x) for x in inner)} 个反引号的围栏，"
                                f"外层只有 {open_ticks} 个 → 代码会被截断，后半段丢失",
                    })
            else:
                stack.append((i, ticks, lang))
                if lang.lower() in ("python", "py"):
                    # 伪代码启发式：python 块里出现 ASCII 框图字符
                    pass
        if indented_fences:
            issues.append({
                "card": rel(c), "type": "存在缩进围栏（提示，非缺陷）",
                "line": indented_fences[0], "lang": "",
                "note": f"第 {indented_fences} 行有缩进 1-3 空格的围栏，通常是列表项里"
                        f"演示代码格式的写法。已排除在硬判定之外，仅供人工确认。",
            })
        unclosed = [(ln, lg) for ln, _, lg in stack]
        if unclosed:
            issues.append({"card": rel(c), "type": "围栏未闭合",
                           "line": unclosed[0][0], "lang": unclosed[0][1],
                           "note": "此处之后的正文会被当作代码解析，K1 会报大量假语法错误"})
        # 伪代码标成 python：块内含框图字符或纯签名列表
        for m in re.finditer(r"^```(?:python|py)\s*$", text, re.M):   # 零缩进
            start = m.end()
            end = text.find("```", start)
            body = text[start:end] if end > 0 else ""
            if re.search(r"[│├└┌┐┘─┴┬┼]", body) and "def " not in body:
                issues.append({"card": rel(c), "type": "伪代码标成 python",
                               "line": text[:m.start()].count("\n") + 1, "lang": "python",
                               "note": "块内是 ASCII 框图，应标 ```text 或 ```pseudocode"})
    # ⚠️ critical 必须只看**真缺陷**：早期写成 `bool(issues)`，
    # 于是「存在缩进围栏」这类提示会让整份体检永远报 CRITICAL，
    # 而永远红的门禁等于没有门禁（会被直接忽略）。
    real_issues = [i for i in issues if i["type"] != "存在缩进围栏（提示，非缺陷）"]
    return {"id": "C4", "name": "代码围栏结构", "critical": bool(real_issues),
            "detail": issues[:40], "total": len(real_issues),
            "summary": (f"{len(real_issues)} 处围栏结构缺陷"
                        f"（未闭合 / 嵌套未加长 / 伪代码标错）"
                        + (f"；另有 {len(issues) - len(real_issues)} 条缩进围栏提示（非缺陷）"
                           if len(issues) > len(real_issues) else ""))}


def c5_registry_consistency() -> dict:
    if not REGISTRY.is_file():
        return {"id": "C5", "name": "registry 一致性", "critical": True,
                "detail": [], "summary": "registry 文件不存在"}
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    missing, extracted = [], 0
    by_paper = {}
    for c in cards():
        fm = _fm(c.read_text(encoding="utf-8", errors="replace"))
        if fm.get("paper_id"):
            by_paper.setdefault(fm["paper_id"], []).append(rel(c))
    for rec in data.get("records", []):
        out = (rec.get("outputs") or {}).get("skill_card") or ""
        if rec.get("status") == "extracted":
            extracted += 1
        if out and not (REPO / out).is_file() and "<" not in out:
            missing.append({"paper_id": rec["paper_id"], "expected": out})
    no_registry = [p for p in by_paper
                   if p not in {r.get("identifiers", {}).get("arxiv") for r in data.get("records", [])}]
    return {"id": "C5", "name": "registry 一致性", "critical": False,
            "detail": {"declared_but_missing": missing, "card_paper_not_in_registry": no_registry},
            "summary": (f"registry {len(data.get('records', []))} 条（extracted {extracted}）；"
                        f"声明存在但磁盘缺失 {len(missing)}；卡片 paper_id 不在 registry {len(no_registry)}")}


def c6_gate_summary() -> dict:
    out = {}
    for g in ("g1", "g2", "g3"):
        f = GATES / f"gate_{g}.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            out[g.upper()] = {"summary": d.get("summary", {}),
                              "generated_at": d.get("generated_at", "")}
    if K1_JSON.is_file():
        k = json.loads(K1_JSON.read_text(encoding="utf-8"))
        results = k.get("results", [])
        n = len(results)
        p = sum(1 for r in results if r.get("verdict") == "PASS")
        # ⚠️ K1 产物的时间戳在 `summary.generated_at`，不在顶层；
        #    早期只读顶层，导致误报「K1 产物缺生成时间戳」（是我的检查读错了，不是 K1 没写）。
        out["K1"] = {"units": n, "pass": p,
                     "rate_pct": round(p / n * 100, 1) if n else 0.0,
                     "generated_at": (k.get("summary") or {}).get(
                         "generated_at", k.get("generated_at", ""))}
    stale = []
    # ⚠️ 必须用 tz-aware 的 now：门禁产物的时间是 `2026-09-12T12:20:08+0800`（带时区），
    # 而 `datetime.now()` 是 naive，两者相减抛 TypeError，被 except 吞掉后
    # 全部误报成「时间戳不可解析」（实测 G1/G2/G3 三条假告警）。
    now = datetime.now().astimezone()
    for name, info in out.items():
        ts = info.get("generated_at") or ""
        if not ts:
            stale.append(f"{name} 产物缺生成时间戳")
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.astimezone()
            age = (now - dt).days
            if age > 7:
                stale.append(f"{name} 产物已 {age} 天未更新")
        except Exception as exc:
            stale.append(f"{name} 产物时间戳无法解析（{ts}）：{exc}")
    return {"id": "C6", "name": "门禁汇总", "critical": False,
            "detail": out, "stale": stale,
            "summary": "；".join(
                # ⚠️ 门禁产物的百分比在 v["summary"]["pass_rate_pct"]，
                #    而 K1 的在 v["rate_pct"] —— 早期只读顶层，四个门禁全显示成 "?"。
                f"{k} " + (f"{v['summary']['pass_rate_pct']}%"
                           if isinstance(v.get("summary"), dict)
                           and "pass_rate_pct" in v["summary"]
                           else (f"{v['rate_pct']}%" if "rate_pct" in v else "?"))
                for k, v in out.items()) or "无门禁产物"}


def c7_repo_hygiene() -> dict:
    """仓库卫生。

    ⚠️ 关键区分：**被 git 跟踪**的脏文件才是问题；本地存在但已被 .gitignore 忽略的
    不是问题。早期版本只数文件个数，于是每次跑都会报「6 个 .pyc」，
    而 `__pycache__/` 早已在 .gitignore 里 —— 这种「永远报同一件事」的检查
    会训练人忽略报告，比不检查更糟。
    """
    try:
        import subprocess
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                                 text=True, timeout=60).stdout.splitlines()
    except Exception:
        tracked = []
    junk_tracked = [t for t in tracked
                    if t.endswith(".pyc") or "__pycache__" in t or t.endswith(".pyo")]
    local_pyc = sum(1 for _ in REPO.rglob("*.pyc") if ".git" not in _.parts)
    empty = [rel(p) for p in VAULT.rglob("*")
             if p.is_dir() and not any(p.iterdir()) and ".git" not in p.parts]
    return {"id": "C7", "name": "仓库卫生", "critical": bool(junk_tracked),
            "detail": {"tracked_junk": junk_tracked[:10],
                       "empty_dirs": empty[:10],
                       "_local_only_not_tracked": f"{local_pyc} 个 .pyc 在本地但已被 .gitignore 忽略"},
            "total": len(junk_tracked),
            "summary": (f"被 git 跟踪的垃圾文件 {len(junk_tracked)} 个"
                        f"（本地另有 {local_pyc} 个 .pyc，已被 .gitignore 忽略，不算问题）；"
                        f"空目录 {len(empty)} 个")}


def c8_sections() -> dict:
    """检查卡片是否具备 v2 的六段结构。

    本项是「迁移欠账」的度量：存量卡片多为 v1 五段式甚至只有①②③，
    而 G2 要求 ⑥ 原文引用、G3 要求 ⑤ 商业价值 —— 缺段必然过不了门禁。
    实测：有 3 张卡只有 ①②③（代码块未闭合、且完全没有 ④⑤⑥）。
    """
    HAVE = {"①": r"^#+\s*①", "①b": r"^#+\s*①b", "②": r"^#+\s*②",
            "③": r"^#+\s*③", "④": r"^#+\s*④", "⑤": r"^#+\s*⑤", "⑥": r"^#+\s*⑥"}
    rows, no_six, only3 = [], [], []
    for c in cards():
        text = c.read_text(encoding="utf-8", errors="replace")
        have = [k for k, pat in HAVE.items() if re.search(pat, text, re.M)]
        missing = [k for k in HAVE if k not in have]
        rows.append({"card": rel(c), "missing": missing})
        if "⑥" not in have:
            no_six.append(rel(c))
        if set(have) <= {"①", "②", "③"}:
            only3.append(rel(c))
    return {"id": "C8", "name": "v2 段落完整性", "critical": False,
            "detail": {"missing_⑥": no_six[:20], "only_①②③": only3[:20],
                       "count_missing_six": len(no_six), "count_only3": len(only3)},
            "total": len(no_six),
            "summary": (f"缺 ⑥ 原文引用 {len(no_six)}/{len(rows)} 张"
                        f"（其中 {len(only3)} 张仅有 ①②③，是 v1 遗留）")}


CHECKS = {
    "C1": c1_duplicate_cards, "C2": c2_frontmatter, "C3": c3_hardcoded_paths,
    "C4": c4_code_fences, "C5": c5_registry_consistency, "C6": c6_gate_summary,
    "C7": c7_repo_hygiene,
    "C8": c8_sections,
}


def selftest() -> int:
    """自检：用构造样本证明 C1/C2/C4 真的会报警（防止检查静默失效）。"""
    import tempfile
    global VAULT, REPO
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "vault" / "01-域").mkdir(parents=True)
        (root / "vault" / "02-域").mkdir(parents=True)
        # C1 构造：同名卡片两份
        (root / "vault" / "01-域" / "Skill-Dup.md").write_text("---\ntitle: a\n---\n", encoding="utf-8")
        (root / "vault" / "02-域" / "Skill-Dup.md").write_text("---\ntitle: a\n---\n", encoding="utf-8")
        # C2 构造：无 frontmatter
        (root / "vault" / "01-域" / "Skill-NoFm.md").write_text("# 无 frontmatter\n", encoding="utf-8")
        # C4 构造：围栏未闭合 + 伪代码
        (root / "vault" / "01-域" / "Skill-BadFence.md").write_text(
            "---\ntitle: b\n---\n```python\nx = 1\n", encoding="utf-8")
        (root / "vault" / "01-域" / "Skill-Pseudo.md").write_text(
            "---\ntitle: c\n---\n```python\n┌───┐\n│ A │\n└───┘\n```\n", encoding="utf-8")

        VAULT, REPO = root / "vault", root
        r1, r2, r4 = c1_duplicate_cards(), c2_frontmatter(), c4_code_fences()
        checks = [
            ("C1 抓到重复卡片", bool(r1["detail"])),
            ("C2 抓到无 frontmatter", len(r2["detail"]["no_frontmatter"]) >= 1),
            ("C4 抓到未闭合围栏", any("未闭合" in i["type"] for i in r4["detail"])),
            ("C4 抓到伪代码标错", any("伪代码" in i["type"] for i in r4["detail"])),
        ]
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")
        ok &= passed
    print("✅ 自检通过：仓库体检的检查项真的会报警" if ok else "❌ 自检失败：检查静默失效")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="paper2skills 仓库体检")
    ap.add_argument("--only", action="append", choices=sorted(CHECKS), default=[])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    names = args.only or sorted(CHECKS)
    reports = [CHECKS[n]() for n in names]

    ts = datetime.now().isoformat(timespec="seconds")
    print(f"{'='*74}\npaper2skills 仓库体检 · {ts}\n{'='*74}")
    for r in reports:
        flag = "🔴" if r["critical"] else ("🟡" if r.get("total") else "✅")
        print(f"\n{flag} {r['id']} {r['name']}：{r['summary']}")
        d = r["detail"]
        if isinstance(d, dict):
            for k, v in d.items():
                if v:
                    print(f"     {k}: {json.dumps(v, ensure_ascii=False)[:220]}")
        elif d:
            for item in d[:8]:
                if isinstance(item, dict):
                    print(f"     {json.dumps(item, ensure_ascii=False)[:210]}")
                else:
                    print(f"     {item}")
            if len(d) > 8:
                print(f"     …另有 {len(d) - 8} 条")
    for r in reports:
        for s in r.get("stale", []):
            print(f"\n⚠️  {s}")

    crit = [r["id"] for r in reports if r["critical"]]
    print(f"\n{'='*74}")
    print(f"结论：{'❌ CRITICAL: ' + ', '.join(crit) if crit else '✅ 无 CRITICAL'}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"generated_at": ts, "reports": reports}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"→ {args.json_out}")
    return 1 if crit else 0


if __name__ == "__main__":
    sys.exit(main())
