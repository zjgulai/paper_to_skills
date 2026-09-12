#!/usr/bin/env python3
r"""verify_no_regression.py — 证明「补引文」这一步没有引入新的无出处断言

## 为什么需要这个脚本

给存量卡补引文时，一个很容易被忽略的风险是：**新增的内容本身也是正文**。
在 ⑥ 段导语里写「全文存档 54646 字符」「新增 12 条引文」，或在说明里写
「原本只有 3 张卡有引用」，这些数字会被 G2 当成**新的无出处断言** ——
于是「修门禁」的动作反而让门禁红灯变多。

`gate_check` 只报当前状态，它**答不了「是我的改动引入的，还是本来就有的」**。
本脚本用 git 取出改动前的版本，把**当前**每条红灯/黄灯的断言 token 在**旧文**
里回查：全部命中 ⇒ 证明没有新增。

## 判定

对每张卡：
1. 取 `HEAD` 版本（旧）与工作区版本（新）；
2. 分别跑 gate_check --only G2，得到 findings；
3. 收集**新** findings 的断言 token（从 message 的 `\`token\`` 里取，或整条消息）；
4. 在**旧文**里做空白归一化后回查；
5. 缺失的 token = **本次新增的无出处断言** → 报 REGRESSED。

⚠️ 反向也要查：**旧红新不红**是好事（修复），单独计数即可，不算问题。

用法：
    python3 verify_no_regression.py --card <card.md>          # 单卡
    python3 verify_no_regression.py --staged                  # 所有已改动卡片
    python3 verify_no_regression.py --card <c.md> --json
    python3 verify_no_regression.py --selftest

退出码：0 = 无回归；1 = 发现新增的无出处断言（或自检失败）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "paper2skills-skills" / "paper-审核" / "scripts" / "gate_check.py"

# 从 finding message 里取出被反引号包起来的断言 token
TOKEN_RE = re.compile(r"`([^`]+)`")
WS_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS_RE.sub("", s)


def load_gate_module():
    """按文件路径加载 gate_check（与 sync.py / quote_check 的加载方式一致）。

    ⚠️ 必须注册 sys.modules：CPython 3.14 下 `@dataclass` +
    `from __future__ import annotations` 会在 `dataclasses._is_type` 里查
    `sys.modules[cls.__module__]`，未注册则 AttributeError。
    这是本项目 K1 门禁 bug #5 的同源坑，已在此复用修法。
    """
    spec = importlib.util.spec_from_file_location("_p2s_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p2s_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def gate_findings(mod, text: str, path_for_rel: Path) -> list[dict]:
    """对一段文本跑 G2，返回 findings（写临时文件是为了让相对路径稳定）。"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / path_for_rel.name
        p.write_text(text, encoding="utf-8")
        r = mod.gate_g2(p, text)
        return [{"level": f.level, "code": f.code, "message": f.message}
                for f in r.findings]


def head_version(rel: str) -> str | None:
    proc = subprocess.run(["git", "show", f"HEAD:{rel}"],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def tokens_of(findings: list[dict]) -> set[str]:
    """findings 里出现的断言 token。取不到反引号 token 时退回整条消息。

    只关心「断言无出处」类，其余（引文伪造、结构问题）由 gate_check 直接管。
    """
    out: set[str] = set()
    for f in findings:
        if f["level"] not in ("RED", "YELLOW"):
            continue
        if f["code"] not in ("G2-UNSOURCED-METRIC", "G2-UNSOURCED-GENERAL",
                             "G2-SOURCED-STRUCTURAL-ONLY", "G2-CN-NUMERAL-CLAIM"):
            continue
        m = TOKEN_RE.search(f["message"])
        out.add(m.group(1) if m else f["message"])
    return out


def check_card(mod, rel: str) -> dict:
    """返回 {rel, old_red, new_red, fixed, regressed, added_tokens}。"""
    path = REPO_ROOT / rel
    new_text = path.read_text(encoding="utf-8", errors="replace")
    old_text = head_version(rel)
    if old_text is None:
        return {"card": rel, "status": "NO_HEAD_VERSION",
                "note": "该卡在 HEAD 里不存在（新增卡）—— 无回归可比"}

    old_f = gate_findings(mod, old_text, path)
    new_f = gate_findings(mod, new_text, path)
    old_tok, new_tok = tokens_of(old_f), tokens_of(new_f)

    old_norm = norm(old_text)
    regressed = sorted(t for t in new_tok - old_tok if norm(t) not in old_norm)
    fixed = sorted(old_tok - new_tok)

    return {
        "card": rel,
        "status": "REGRESSED" if regressed else "OK",
        "old_red": sum(1 for f in old_f if f["level"] == "RED"),
        "new_red": sum(1 for f in new_f if f["level"] == "RED"),
        "old_tokens": len(old_tok),
        "new_tokens": len(new_tok),
        "fixed_tokens": len(fixed),
        "regressed_tokens": len(regressed),
        "regressed": regressed[:20],
    }


def changed_cards() -> list[str]:
    """已改动（未提交）的 Skill 卡片相对路径。"""
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    out = []
    for line in proc.stdout.splitlines():
        rel = line[3:].strip().strip('"')
        if rel.startswith("paper2skills-vault/") and re.search(r"Skill-.*\.md$", rel):
            out.append(rel)
    return sorted(out)


def selftest() -> int:
    """自检：证明本脚本能抓到「新增的无出处断言」。

    构造一张卡：旧版无数字，新版在导语里塞一个带数字的新句子。
    新版该句子应被 G2 判无出处，且**旧文里找不到** → 必须报 REGRESSED。
    """
    ok = True
    mod = load_gate_module()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "Skill-Selftest.md"
        old = "---\ntitle: t\npaper_id: 2606.26690\n---\n\n结论是定性的。\n"
        new = (old + "\n本次新增了 **12** 条引文，覆盖 **85%** 的断言。\n")
        p.write_text(new, encoding="utf-8")
        new_f = gate_findings(mod, new, p)
        new_tok = tokens_of(new_f)
        hit = [t for t in new_tok if norm(t) not in norm(old)]
        print(f"新版 findings token: {sorted(new_tok)}")
        print(f"旧文里找不到的 token: {hit}")
        if not hit:
            print("❌ 新增的带数字断言没有产生「旧文找不到」的 token —— 检查失效")
            ok = False
        else:
            print("✅ 能识别新增的无出处断言（新增内容本身会被 G2 计入）")

    print("✅ 自检通过" if ok else "❌ 自检失败")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="证明补引文未引入新的无出处断言")
    ap.add_argument("--card")
    ap.add_argument("--staged", action="store_true",
                    help="检查所有已改动但未提交的卡片")
    ap.add_argument("--json-out")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.card:
        rels = [str(Path(args.card).resolve().relative_to(REPO_ROOT))]
    elif args.staged:
        rels = changed_cards()
        if not rels:
            print("没有已改动的卡片")
            return 0
    else:
        ap.error("需要 --card / --staged / --selftest")

    mod = load_gate_module()
    reports = [check_card(mod, r) for r in rels]

    n_bad = [r for r in reports if r.get("status") == "REGRESSED"]
    n_nohd = [r for r in reports if r.get("status") == "NO_HEAD_VERSION"]
    n_ok = [r for r in reports if r.get("status") == "OK"]

    if not args.quiet:
        for r in reports:
            if r["status"] == "NO_HEAD_VERSION":
                print(f"· {Path(r['card']).name}: {r['note']}")
                continue
            icon = "❌" if r["status"] == "REGRESSED" else "✅"
            print(f"{icon} {Path(r['card']).name}")
            print(f"     红灯 {r['old_red']} → {r['new_red']}｜"
                  f"无出处断言 {r['old_tokens']} → {r['new_tokens']}｜"
                  f"修掉 {r['fixed_tokens']}｜**新增 {r['regressed_tokens']}**")
            for t in r.get("regressed", []):
                print(f"       ⚠️ 新增无出处: `{t}`")

    print(f"\n共 {len(reports)} 张：{len(n_ok)} 无回归，{len(n_bad)} 有回归，"
          f"{len(n_nohd)} 无法比对（新增卡）")
    if n_bad:
        print("❌ 有卡片引入了新的无出处断言 —— 补引文时新增的正文数字也要给出处，"
              "或改写成中文数字/定性表述")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {args.json_out}")

    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
