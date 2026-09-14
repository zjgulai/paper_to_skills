#!/usr/bin/env python3
"""飞书残留门禁（2026-09-14 立，PHASE6 验收面 L17）。

为什么需要它
------------
`playbook/` 下 1496 个页面是**上游构建产物**，本仓库没有生成器。2026-09-14 按所有者要求
把注入进去的飞书前端集成剥掉了（`strip_feishu_from_playbook.py`）。但**剥掉不等于不会回来**：
上游重建、或下一次「按模板生成页面」都会把它们注回来，而那种回归**没有任何现有门禁看得见**
（K1/K2 看卡片，凭证扫描看密钥，契约门禁看契约）。

⇒ 这个门禁每跑一次就问一句：**飞书集成回来了吗？**

两条判据，判法完全不同
----------------------
**判据 A（判红）**：`A_IDENTIFIERS` 这 10 个**集成标识**在全库必须为 0。
    它们是**代码形态**（函数名、路由、CSS 类名、入口按钮文案），卡片正文不会写它们。

**判据 B（增长判红、收缩只报）**：`feishu|飞书` **裸词**普查，与
    `data/feishu-residue-baseline.json` 逐文件比对：
      · 新出现的文件 / 计数超过 baseline  ⇒ **判红**（「飞书又回来了」）
      · 计数低于 baseline                  ⇒ **不判红**，但打印「baseline 可收紧」，
                                            降到 0 的文件点名提示删条目
    裸词之所以不能一刀切判红：**它们是四类合法存量**（见 baseline 的 families），
    其中最多的一类是**卡片正文里的业务场景描述**（「告警推送（微信/飞书）」）——
    那是内容不是集成，删它就是改业务语义。⇒ **豁免必须可见、可收紧、带到期条件**
    （本仓库台账 #84 的纪律：过期豁免必须判红，不能只 printf）。

适用范围（如实写清，别当成绿灯）
--------------------------------
只看**已入库内容**（`git ls-files`）。「曾经 commit 过又删掉」的那种由
`check_history_secrets.py`（对象库）覆盖；「未跟踪但也未忽略」的由
`check_key_exposure.py` 覆盖。三者是三个不同的切片，缺一不可。

退出码：**0** 干净 · **1** 判红 · **2** 输入没拿到（≠ 通过）· **3** 门禁内部错误（≠ 判红）
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "paper2skills-research" / "data" / "feishu-residue-baseline.json"

# ---------------------------------------------------------------------------
# 判据 A：集成标识（**唯一一处定义**；strip_feishu_from_playbook.py 从这里 import，
# 保证「剥的人」与「判的人」看的是同一份清单 —— 风险 N2：判据只许有一处实现）
# ---------------------------------------------------------------------------
A_IDENTIFIERS = [
    "pbShareFeishu",  # share 函数
    "/api/feishu-callback",  # share 的后端路由
    "pb-feishu-bar",  # share 按钮所在外框的类名
    "_FEISHU_HOOK",  # 前端读配置里的 webhook（含 _FEISHU_HOOK_RPT）
    "feishuHook",  # 那个配置键
    "pushToFeishu",  # agents.html 的推送函数/调用
    "pushDetailToFeishu",  # agent-report.html 的推送函数
    "rpt-detail-btn-feishu",  # 「推送飞书」按钮的类名与 CSS
    "飞书登录",  # 登录入口按钮文案
    "sync_feishu",  # sync.py 的飞书同步目标
]

# ⚠️ 判据 A **不许**包含 `推送飞书` / `飞书` 这类裸词：卡片正文里合法存在
#    （「推送飞书提醒人工确认」），收进来就是造假红 —— 见 --selftest 用例 ⑤。

BARE = re.compile(r"feishu|飞书", re.I)


def tracked_files(root: Path) -> list[str]:
    """已入库文件清单（相对路径）。取不到 ⇒ 抛异常，由调用方判 exit 2。"""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git ls-files 失败（rc={out.returncode}）：{out.stderr.strip()[:200]}")
    return [f for f in out.stdout.split("\0") if f]


def scan(root: Path) -> tuple[dict[str, int], dict[str, list[str]]]:
    """返回 (裸词逐文件计数, 判据A 逐标识命中文件)。"""
    bare: dict[str, int] = {}
    ident: dict[str, list[str]] = {k: [] for k in A_IDENTIFIERS}
    for rel in tracked_files(root):
        p = root / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, PermissionError):
            continue
        n = len(BARE.findall(text))
        if n:
            bare[rel] = n
        for k in A_IDENTIFIERS:
            if k in text:
                ident[k].append(rel)
    return bare, {k: v for k, v in ident.items() if v}


def split_exempt(
    ident: dict[str, list[str]], exemptions: list[dict]
) -> tuple[dict[str, list[str]], list[tuple[str, str, str]]]:
    """把判据 A 的命中拆成 (判红的, 已豁免的)。

    豁免是**逐文件 × 逐标识**的，不是按目录一刀切 —— 台账 #78 的教训是
    「豁免的颗粒度就是判据的颗粒度」：整台仪器一起豁免，会让同族里**精确**的判据
    一起隐形。所以这里只认 (file, identifier) 这一对，且每次跑都打印出来。
    """
    allowed: dict[tuple[str, str], dict] = {}
    for e in exemptions:
        for i in e.get("identifiers", []):
            allowed[(e["file"], i)] = e
    red: dict[str, list[str]] = {}
    exempt_hits: list[tuple[str, str, str]] = []
    for ident_name, files in ident.items():
        for f in files:
            e = allowed.get((f, ident_name))
            if e is None:
                red.setdefault(ident_name, []).append(f)
            else:
                exempt_hits.append((f, ident_name, e.get("reason", "")))
    return red, exempt_hits


def compare(bare: dict[str, int], baseline: dict[str, int]) -> tuple[dict, dict, dict]:
    """返回 (新增或增长, 收缩, 持平)。三者都点名。"""
    grown, shrunk, same = {}, {}, {}
    for f, n in bare.items():
        b = baseline.get(f)
        if b is None:
            grown[f] = (n, None)
        elif n > b:
            grown[f] = (n, b)
        elif n < b:
            shrunk[f] = (n, b)
        else:
            same[f] = n
    for f, b in baseline.items():
        if f not in bare:  # baseline 里有、现在彻底没了
            shrunk[f] = (0, b)
    return grown, shrunk, same


def report(root: Path, baseline: dict, quiet: bool = False) -> int:
    bare, ident = scan(root)
    red_ident, exempt_hits = split_exempt(ident, baseline.get("identifier_exemptions", []))

    print(f"仓库 {root}")
    print(f"  已入库文件里有 feishu/飞书 的：{len(bare)} 个 · 共 {sum(bare.values())} 处")

    # ---- 判据 A ----
    a_red = bool(red_ident)
    print(f"\n判据 A（集成标识必须为 0）：{'❌ 判红' if a_red else '✅ 全 0'}（{len(A_IDENTIFIERS)} 个标识）")
    for k, files in red_ident.items():
        print(f"    ❌ {k} —— 命中 {len(files)} 个文件：{files[:4]}")
    if a_red:
        print("    ⇒ 修法：跑 paper2skills-research/scripts/strip_feishu_from_playbook.py（幂等）")
    # 豁免**每次跑都打印**（台账 #84：豁免只 printf 在绿门上根本看不见）
    for f, i, why in exempt_hits:
        print(f"    ⚠️  [已豁免] {i} ∈ {f}")
        print(f"        理由：{why}")

    # ---- 判据 B ----
    grown, shrunk, same = compare(bare, baseline.get("files", {}))
    print(f"\n判据 B（裸词普查 vs baseline {len(baseline.get('files', {}))} 条）：")
    print(f"    持平 {len(same)} · 收缩 {len(shrunk)} · 新增或增长 {len(grown)}")
    for f, (n, b) in sorted(grown.items(), key=lambda x: -x[1][0]):
        print(f"    ❌ {f}：{b if b is not None else '（baseline 无此文件）'} → {n}")
    for f, (n, b) in sorted(shrunk.items(), key=lambda x: x[1][1] - x[1][0], reverse=True):
        tip = "（已归零，请删 baseline 条目）" if n == 0 else "（baseline 可收紧）"
        print(f"    ⚠️  {f}：{b} → {n} {tip}")
    b_red = bool(grown)
    print(f"    ⇒ {'❌ 判红：飞书又长回来了' if b_red else '✅ 没有增长'}")

    if not quiet:
        print(f"\n  适用范围：只看已入库内容（{len(tracked_files(root))} 个文件）。"
              "对象库看 check_history_secrets.py，未跟踪未忽略的看 check_key_exposure.py。")
        if exempt_hits:
            print("  豁免是逐文件 × 逐标识的；本仓库台账 #78 的教训：豁免的颗粒度就是判据的颗粒度。")

    if a_red or b_red:
        return 1
    return 0


def selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool, str]] = []

    def ck(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    def make_tree(tmp: Path, files: dict[str, str]) -> Path:
        for rel, text in files.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=tmp, capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=tmp, capture_output=True)
        return tmp

    with tempfile.TemporaryDirectory() as td:
        # ⚠️ 自证**不许依赖真实 baseline**：baseline 还没建（或已被删）时，
        # 判据本身对不对仍然必须能问出来。
        # ① 反向控制：拿**真实仓库**跑，判据 A 必须为 0（仪器不许对真实语料报假红）
        #    注意要**带上豁免**再看 —— 记录类文件里出现标识名是取证叙述，不是集成代码。
        try:
            _real_ex = json.loads(BASELINE.read_text(encoding="utf-8")).get("identifier_exemptions", [])
        except FileNotFoundError:
            _real_ex = []
        bare, ident = scan(REPO)
        red_ident, exempt_real = split_exempt(ident, _real_ex)
        ck("① 真实仓库上判据 A 为 0（不误报）", not red_ident, f"判红={list(red_ident)}")
        ck("①a 真实仓库的判据 A 命中全部落在**逐条可见的豁免**里",
           len(exempt_real) == sum(len(v) for v in ident.values()),
           f"豁免 {len(exempt_real)} 条 / 原始命中 {sum(len(v) for v in ident.values())} 条")
        ck("①b 真实仓库裸词普查非空（证明它真的在读语料，不是空过）", sum(bare.values()) > 0,
           f"共 {sum(bare.values())} 处")

        # ② 判据 A 不许与仪器打架：scan_secrets.py 里的规则名是 `FEISHU_WEBHOOK`（大写），
        #    判据 A 里没有这一项 ⇒ 仪器本身不得被当成集成残留
        _low = [x.lower() for x in A_IDENTIFIERS]
        ck("② 判据 A 不含大写规则名 FEISHU_WEBHOOK、也不含裸词",
           "feishu_webhook" not in _low and "feishu" not in _low and "飞书" not in A_IDENTIFIERS,
           f"清单尾部={A_IDENTIFIERS[-3:]}")

        # ③ 正控制：注入一个集成标识 ⇒ 必须判红
        with tempfile.TemporaryDirectory() as t3:
            root3 = make_tree(Path(t3), {"a.html": "window.pbShareFeishu = function(){};\n"})
            _b, i3 = scan(root3)
            ck("③ 正控制：注入 pbShareFeishu ⇒ 判据 A 命中", "pbShareFeishu" in i3)

        # ④ 判据 B 增长判红（构造语料 + 一条人为增长）
        with tempfile.TemporaryDirectory() as t4:
            sample = [f"f{i}.md" for i in range(5)]
            files = {k: "" for k in sample}
            files["new.html"] = "飞书登录\n"  # 新文件
            root4 = make_tree(Path(t4), files)
            b4, _i4 = scan(root4)
            grown, shrunk, _s = compare(b4, {k: 1 for k in sample})
            ck("④ 判据 B：新出现的文件被判为「增长」", grown.get("new.html") == (1, None))
            ck("④b 判据 B：baseline 有而文件仍空 ⇒ 判为「收缩」而不是红",
               all(f in shrunk for f in sample) and not any(f in grown for f in sample))

        # ⑤ 反向控制：**只比计数不比文件**是不行的 —— 同一文件计数不变必须持平
        with tempfile.TemporaryDirectory() as t5:
            root5 = make_tree(Path(t5), {"c.md": "飞书 飞书\n"})
            b5, _ = scan(root5)
            g5, s5, sm5 = compare(b5, {"c.md": 2})
            ck("⑤ 同计数 ⇒ 持平（既不红也不报收紧）", not g5 and not s5 and sm5.get("c.md") == 2)

        # ⑦ 豁免的**颗粒度**：逐文件 × 逐标识，三个方向都要有劲
        ex = [{"file": "rec.md", "identifiers": ["feishuHook"], "reason": "事故归档里的取证叙述"}]
        red7, ok7 = split_exempt({"feishuHook": ["rec.md", "page.html"]}, ex)
        ck("⑦ 豁免只覆盖被点名的那个文件（同标识在别的文件里照旧判红）",
           ok7 == [("rec.md", "feishuHook", "事故归档里的取证叙述")] and red7 == {"feishuHook": ["page.html"]},
           f"判红={red7}")
        red8, ok8 = split_exempt({"feishuHook": ["rec.md"], "pbShareFeishu": ["rec.md"]}, ex)
        ck("⑦b 豁免只覆盖被点名的那个标识（同文件里别的标识照旧判红）",
           red8 == {"pbShareFeishu": ["rec.md"]} and len(ok8) == 1, f"判红={red8}")
        red9, _ok9 = split_exempt({"feishuHook": ["rec.md"]}, [])
        ck("⑦c 没有豁免条目时判红（豁免不许凭空生效）", red9 == {"feishuHook": ["rec.md"]})

        # ⑧ 输入没拿到必须是 exit 2，不是 0
        with tempfile.TemporaryDirectory() as t6:
            p6 = Path(t6)  # 不是 git 仓库
            try:
                tracked_files(p6)
                got = True
            except RuntimeError:
                got = False
            ck("⑧ 非 git 目录 ⇒ 抛错（调用方据此 exit 2，而不是当成干净）", not got)

    ok_n = sum(1 for _n, o, _d in checks if o)
    print(f"check_feishu_residue --selftest: {ok_n}/{len(checks)}")
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{detail}]" if detail else ""))
    return 0 if ok_n == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="飞书残留门禁（集成标识判红 + 裸词普查比对）")
    ap.add_argument("--check", action="store_true", help="跑判据（默认动作）")
    ap.add_argument("--selftest", action="store_true", help="构造样本自证")
    ap.add_argument("--update-baseline", action="store_true",
                    help="把当前裸词普查写成 baseline（**只应在有意变更存量后使用**）")
    ap.add_argument("--json-out", metavar="PATH", help="读数写成 JSON")
    ap.add_argument("--quiet", action="store_true", help="少打印（给验收面用）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not BASELINE.exists() and not args.update_baseline:
        print(f"❌ baseline 不存在：{BASELINE.relative_to(REPO)} —— 「没拿到输入」不等于「通过」")
        return 2

    try:
        bare, ident = scan(REPO)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 输入没拿到：{e}")
        return 2

    if not bare and not ident:
        print("❌ 一个文件都没扫到 —— 这不是干净，是没测")
        return 2

    if args.update_baseline:
        old = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        payload = {
            "_note": "飞书残留门禁的裸词 baseline。判据 A（集成标识）不需要 baseline —— 它恒为 0。",
            "why": "裸词不能一刀切判红：存量分四类合法来源，见 families。",
            "families": old.get("families", {}),
            "identifier_exemptions": old.get("identifier_exemptions", []),
            "expires_when": "当 playbook 卡片正文的飞书表述被上游改写、或 07-NLP-VOC 存档迁出后，"
                            "相应条目应删到 0；本文件归零时本门禁可退化为纯判据 A。",
            "files": dict(sorted(bare.items())),
        }
        BASELINE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"✅ baseline 已更新：{len(bare)} 条 → {BASELINE.relative_to(REPO)}")
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    rc = report(REPO, baseline, quiet=args.quiet)

    if args.json_out:
        grown, shrunk, _same = compare(bare, baseline.get("files", {}))
        red_ident, exempt_hits = split_exempt(ident, baseline.get("identifier_exemptions", []))
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "bare_files": len(bare),
                    "bare_total": sum(bare.values()),
                    "identifiers_red": red_ident,
                    "identifiers_exempt": [
                        {"file": f, "identifier": i, "reason": r} for f, i, r in exempt_hits
                    ],
                    "grown": {k: v for k, v in grown.items()},
                    "shrunk": {k: v for k, v in shrunk.items()},
                    "verdict": "red" if (red_ident or grown) else "green",
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"💥 门禁内部错误：{type(e).__name__}: {e}")
        sys.exit(3)
