#!/usr/bin/env python3
"""工作区 instruction 预算门禁 —— 「文件在磁盘上好好的，agent 读到的却是被砍过的那份」。

为什么需要这一道门（实测事件，不是理论担忧）
------------------------------------------------
2026-09-13，本会话在 `CLAUDE.md` 里加了一节（约 1.8 KB），随后 harness 报出：

    Workspace instruction budget 65536 bytes: omitted ~/.dsh/AGENTS.md, AGENTS.md;
    truncated CLAUDE.md from 66175 to 65244 bytes

即：**磁盘上三个文件都是完整的**，而 agent 实际读到的 `CLAUDE.md` 被**静默砍掉 931 字节**，
`AGENTS.md` 与 `~/.dsh/AGENTS.md` **整份没进上下文**。
被砍/被丢的是文件**尾部**——上一次同族事故砍掉的恰好是 `Skill Card Format` 那一节，
也就是 F5/S1 出卡要照的规格。**没有人会为此报错**：文件 diff 正常、门禁全绿、
唯一的症状是上下文里多了一行 marker。

判据的适用范围（先问「我这个仪器能看见什么」）
------------------------------------------------
本门禁**只在「agent 读到的是一份完整的、含全部工作区 instruction 文件的渲染」时判绿**。
它复刻 harness 的渲染与取舍算法（见下），不猜、不近似——渲染函数是逐字对齐的。

harness 的取舍顺序（`@deepseek-ai/dsh-agent-instructions/lib/index.js`）
------------------------------------------------------------------------
1. 渲染**全部**文件；若总字节 ≤ 预算 ⇒ `FULL`（无丢失）。
2. 否则**从最不具体的那个开始整份丢弃**（`~/.dsh/AGENTS.md` → 项目 `AGENTS.md` → …），
   保留一个后缀；若某个后缀能装下 ⇒ `OMITTED`（丢的是**整份文件**，marker 会点名）。
3. 再装不下 ⇒ 对**最后一个（最具体的）**文件做二分截断 ⇒ `TRUNCATED`（**文件内数据丢失**）。

三态与退出码
------------
`FULL`（0）/ `OMITTED`（**0，但一等输出**）/ `TRUNCATED`（**1**）/ 输入没拿到（2）/ 内部错误（3）。

⚠️ `OMITTED` 为什么不判红：整份丢弃是 harness **设计好的优先级**行为（更具体的指令优先），
不是数据损坏；而且 marker 会点名。但**它仍然是「文档写了、agent 没看到」**，
所以它必须作为**一等输出**逐文件报出字节数 —— 只警告不计数就是没有判据。

⚠️ `TRUNCATED` 为什么必须判红：它是**静默的**——同一份文件既「存在」又「不完整」，
任何 diff/门禁都看不出来。这正是本仓库反复栽的那一类（假绿）。

自证（`--selftest`）
--------------------
· 复刻真实事件：`(original=66175, included=65244)` 必须判「装得下」，
  而 `included=65245` 必须判「装不下」——**边界由渲染函数算出来，不是抄的常数**。
· 三态各自的夹具 + **反向控制**（干净夹具必须 exit 0，否则判据是恒红的摆设）。
· 把渲染函数改坏（`--mutate`）后用例必须抓住——先证明变异体真的跑了，再谈判据有没有劲。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DSH_HOME = Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh")))

# ---- harness 渲染常量（逐字复制自 dsh-agent-instructions/lib/index.js:111–117,130） ----
SYSTEM_REMINDER_OPEN = "<system-reminder>"
SYSTEM_REMINDER_CLOSE = "</system-reminder>"
WORKSPACE_CONTEXT_INTRO = (
    "The following workspace instructions may be relevant to your work. Use them as guidance "
    "when applicable. More specific instructions take precedence over broader ones. "
    "They do not override system, developer, or direct user instructions."
)


# 第二条渲染路径：文件**变更后的对账批**（`changedSectionText` + `intro: ""`）。
# ⚠️ 实测 2026-09-13 的那次截断走的就是这条：marker 里**没有** omitted 子句、section 头是
# 「Updated instructions from: …」。两条路径的包装字节不同（实测 292 vs 430），
# 混用会算出错的截断点 —— 所以两条都必须实现，并且**各自有用例**。
CHANGED_HEAD = (
    "{path}",
    "",
    "This file changed after it was loaded. Use the following content instead of the "
    "previously loaded instructions from this file.",
    "",
)


def section_text(display_path: str, content: str) -> str:
    return f"Instructions from: {display_path}\n\n{content}"


def changed_section_text(display_path: str, content: str) -> str:
    return "\n".join([f"Updated instructions from: {display_path}",
                      *CHANGED_HEAD[1:], content])


def marker_text(budget: int, omitted: list, truncated: list) -> str:
    if not omitted and not truncated:
        return ""
    parts = []
    if omitted:
        parts.append("omitted " + ", ".join(omitted))
    if truncated:
        parts.append("truncated " + ", ".join(
            f"{d} from {o} to {i} bytes" for d, o, i in truncated))
    return f"Workspace instruction budget {budget} bytes: {'; '.join(parts)}"


def build_instruction_text(budget: int, files: list, omitted: list, truncated: list,
                           style: str = "baseline") -> str:
    sec = section_text if style == "baseline" else changed_section_text
    intro = WORKSPACE_CONTEXT_INTRO if style == "baseline" else ""
    body = "\n\n".join(b for b in [
        marker_text(budget, omitted, truncated),
        intro,
        *[sec(dp, c) for dp, c in files],
    ] if b)
    body = body.replace(SYSTEM_REMINDER_CLOSE, "<\\/system-reminder>")
    return "\n".join([SYSTEM_REMINDER_OPEN, body, SYSTEM_REMINDER_CLOSE])


def nbytes(s: str) -> int:
    return len(s.encode("utf-8"))


def truncate_utf8(value: str, max_bytes: int) -> str:
    b = value.encode("utf-8")
    if len(b) <= max_bytes:
        return value
    end = max(0, int(max_bytes))
    while end > 0 and (b[end] & 0xC0) == 0x80:
        end -= 1
    return b[:end].decode("utf-8", errors="replace")


def render(files: list, budget: int, style: str = "baseline") -> dict:
    """复刻 renderInstructionContext：返回 {state, omitted, truncated, text_bytes}。"""
    full = build_instruction_text(budget, files, [], [], style)
    if nbytes(full) <= budget:
        return {"state": "FULL", "omitted": [], "truncated": [], "text_bytes": nbytes(full)}

    for start in range(1, len(files)):
        included = files[start:]
        omitted = [dp for dp, _ in files[:start]]
        suffix = build_instruction_text(budget, included, omitted, [], style)
        if nbytes(suffix) <= budget:
            return {"state": "OMITTED", "omitted": omitted, "truncated": [],
                    "text_bytes": nbytes(suffix), "included": [dp for dp, _ in included]}

    if not files:
        return {"state": "TRUNCATED", "omitted": [], "truncated": [], "text_bytes": 0}
    most = files[-1]
    omitted = [dp for dp, _ in files[:-1]]
    original = nbytes(most[1])
    low, high, best = 0, original, (0, "")
    while low <= high:
        mid = (low + high) // 2
        cand = truncate_utf8(most[1], mid)
        inc = nbytes(cand)
        text = build_instruction_text(budget, [(most[0], cand)], omitted,
                                      [(most[0], original, inc)], style)
        if nbytes(text) <= budget:
            best = (inc, cand)
            low = mid + 1
        else:
            high = mid - 1
    inc, _ = best
    return {"state": "TRUNCATED", "omitted": omitted,
            "truncated": [(most[0], original, inc)], "text_bytes": nbytes(
                build_instruction_text(budget, [(most[0], truncate_utf8(most[1], inc))],
                                       omitted, [(most[0], original, inc)], style))}


# ---- 预算从哪来（读不到 ⇒ exit 2，不许假设一个常数） --------------------------------

BUDGET_SOURCES = [
    ("env P2S_INSTRUCTION_BUDGET", None),
    ("~/.dsh/cordis.patch.yml", DSH_HOME / "cordis.patch.yml"),
    ("DSH_HOME/.agent-presets/*/agent.cordis.yml", DSH_HOME / ".agent-presets"),
]


def find_budget() -> tuple:
    env = os.environ.get("P2S_INSTRUCTION_BUDGET")
    if env and env.strip().isdigit():
        return int(env), BUDGET_SOURCES[0][0]
    p = DSH_HOME / "cordis.patch.yml"
    if p.is_file():
        m = re.search(r"^\s*maxBytes:\s*(\d+)", p.read_text(encoding="utf-8"), re.M)
        if m:
            return int(m.group(1)), str(p)
    presets = DSH_HOME / ".agent-presets"
    if presets.is_dir():
        for f in sorted(presets.glob("*/agent.cordis.yml")):
            m = re.search(r"^\s*maxBytes:\s*(\d+)", f.read_text(encoding="utf-8"), re.M)
            if m:
                return int(m.group(1)), str(f)
    # 出货的基座默认值（只读，优先级最低）
    for root in (Path("/Applications/DSH Desktop.app/Contents/Resources/app.asar.unpacked"),):
        f = root / "node_modules/@deepseek-ai/dsh-base/cordis.patch.yml"
        if f.is_file():
            m = re.search(r"^\s*maxBytes:\s*(\d+)", f.read_text(encoding="utf-8"), re.M)
            if m:
                return int(m.group(1)), f"{f}（出货默认值，非本机 profile）"
    return None, ""


def discover(workspace: Path) -> list:
    """候选文件与顺序 = harness 的发现顺序：用户级 → 项目根（AGENTS.md → CLAUDE.md）。"""
    out = []
    ug = DSH_HOME / "AGENTS.md"
    if ug.is_file():
        out.append(("~/.dsh/AGENTS.md", ug))
    for name in ("AGENTS.md", "CLAUDE.md"):
        f = workspace / name
        if f.is_file():
            out.append((name, f))
    return out


def run(workspace: Path, budget: int | None, budget_src: str, as_json: bool) -> int:
    if budget is None:
        print("❌ 输入没拿到：找不到 instruction 预算（maxBytes）。")
        print("   试过：env P2S_INSTRUCTION_BUDGET / ~/.dsh/cordis.patch.yml / "
              "~/.dsh/.agent-presets/*/agent.cordis.yml / 出货 dsh-base/cordis.patch.yml")
        print("   退出码 2 = 没测，不是通过。")
        return 2
    if not workspace.is_dir():
        print(f"❌ 输入没拿到：工作区不存在：{workspace}")
        return 2

    cands = discover(workspace)
    if not cands:
        print(f"❌ 输入没拿到：{workspace} 下没有任何工作区 instruction 文件（AGENTS.md / CLAUDE.md）")
        return 2
    files = []
    for dp, f in cands:
        try:
            files.append((dp, f.read_text(encoding="utf-8")))
        except OSError as e:
            print(f"❌ 输入没拿到：读不到 {dp}：{e}")
            return 2

    res = render(files, budget)
    sizes = {dp: nbytes(c) for dp, c in files}
    total = sum(sizes.values())

    print(f"工作区 instruction 预算门禁（预算 {budget} B，来自 {budget_src}）")
    for dp, _ in files:
        flag = ""
        if dp in res["omitted"]:
            flag = "  ← 整份被丢弃（agent 看不到）"
        if any(dp == t[0] for t in res["truncated"]):
            flag = "  ← 被截断（文件内数据丢失）"
        print(f"  {dp:<24} {sizes[dp]:>7} B{flag}")
    print(f"  {'合计（原始）':<24} {total:>7} B · 渲染后 {res['text_bytes']} B")

    if as_json:
        print(json.dumps({"budget": budget, "budget_source": budget_src, "state": res["state"],
                          "sizes": sizes, "total": total, "render_bytes": res["text_bytes"],
                          "omitted": res["omitted"],
                          "truncated": [{"file": t[0], "original": t[1], "included": t[2]}
                                        for t in res["truncated"]]},
                         ensure_ascii=False, indent=1))

    if res["state"] == "FULL":
        print("✅ 全部 instruction 文件完整进入上下文（无丢弃、无截断）")
        return 0
    if res["state"] == "OMITTED":
        print(f"⚠️ 状态 OMITTED：{len(res['omitted'])} 份**整份**未进上下文"
              f"（{'、'.join(res['omitted'])}）—— harness 的设计行为，marker 已点名，故不判红。")
        print("   但它仍是「文档写了、agent 没看到」：要消掉它就得让**全部**文件装得下。")
        return 0
    for f_, o, i in res["truncated"]:
        print(f"🔴 状态 TRUNCATED：{f_} 被静默截断，{o} → {i} B（丢 {o - i} B）。")
    print("   磁盘上的文件是完整的 ⇒ 任何 diff / 门禁都看不出来；只有上下文里多一行 marker。")
    return 1


def selftest() -> int:
    """用例集。每条都必须能失败；含反向控制与真实事件回归。"""
    cases, fails = [], 0

    def check(name, ok, detail=""):
        nonlocal fails
        cases.append((name, ok, detail))
        if not ok:
            fails += 1

    budget = 65536
    small = [("AGENTS.md", "a" * 100), ("CLAUDE.md", "b" * 100)]
    r = render(small, budget)
    check("干净夹具 ⇒ FULL（反向控制：判据不是恒红的摆设）", r["state"] == "FULL", r["state"])

    # ---- 真实事件回归（本文件存在的理由） --------------------------------------
    # harness 实报：`truncated CLAUDE.md from 66175 to 65244 bytes`
    # 该批走的是**变更对账**路径（files 只含变更的那一份 ⇒ marker 里没有 omitted 子句）。
    # 判据：复算出的 includedBytes 必须**逐字节相等**，且该值为最大可行值。
    r_ev = render([("CLAUDE.md", "z" * 66175)], budget, style="changed")
    check("真实事件：CLAUDE.md 66175 B 必须判 TRUNCATED", r_ev["state"] == "TRUNCATED", r_ev["state"])
    inc = r_ev["truncated"][0][2] if r_ev["truncated"] else -1
    check("真实事件：复算的截断点必须逐字节等于 harness 报的 65244（包装 292 B 同尺）",
          inc == 65244, f"实测 {inc}（差 {inc - 65244:+d} B）")
    t_in = build_instruction_text(budget, [("CLAUDE.md", "z" * inc)], [],
                                  [("CLAUDE.md", 66175, inc)], "changed")
    t_out = build_instruction_text(budget, [("CLAUDE.md", "z" * (inc + 1))], [],
                                   [("CLAUDE.md", 66175, inc + 1)], "changed")
    check("边界正证：content = included 时渲染 ≤ 预算", nbytes(t_in) <= budget, str(nbytes(t_in)))
    check("边界反证：content = included+1 时渲染 > 预算（否则 inc 不是最大可行值）",
          nbytes(t_out) > budget, str(nbytes(t_out)))
    check("两条渲染路径的包装不同（实测 292 vs 430）—— 混用会算出错的截断点",
          nbytes(build_instruction_text(budget, [("CLAUDE.md", "")], [], [], "changed"))
          != nbytes(build_instruction_text(budget, [("CLAUDE.md", "")], [], [], "baseline")), "")

    # 三态可区分
    chunky = [("AGENTS.md", "a" * 40000), ("CLAUDE.md", "b" * 40000)]
    rs = render(chunky, budget)
    check("两份合计超预算 ⇒ OMITTED（整份丢弃，不是截断）", rs["state"] == "OMITTED", rs["state"])
    check("OMITTED 必须点名丢了哪一份", rs["omitted"] == ["AGENTS.md"], str(rs["omitted"]))
    huge = [("CLAUDE.md", "b" * 200000)]
    rh = render(huge, budget)
    check("单份超预算 ⇒ TRUNCATED", rh["state"] == "TRUNCATED", rh["state"])

    # 预算读不到 ⇒ 2（不是 0）
    saved = os.environ.pop("P2S_INSTRUCTION_BUDGET", None)
    b, _ = find_budget()
    check("本机确实能读到预算（否则门禁每次都在「没测」）", b is not None, str(b))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "CLAUDE.md"
        p.write_text("x" * 10, encoding="utf-8")
        global DSH_HOME
        old = DSH_HOME
        DSH_HOME = Path(td) / "nope"
        code = run(Path(td), None, "", False)
        check("预算取不到 ⇒ exit 2（不是 0）", code == 2, str(code))
        code2 = run(Path(td) / "missing", b or 65536, "selftest", False)
        check("工作区不存在 ⇒ exit 2", code2 == 2, str(code2))
        DSH_HOME = old
    if saved is not None:
        os.environ["P2S_INSTRUCTION_BUDGET"] = saved

    # 端到端：真 CLI + 真夹具（台账 #25：判据在 main() 里而 selftest 只测库函数 ⇒ 假仪表）
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "CLAUDE.md").write_text("y" * 200000, encoding="utf-8")
        p = subprocess.run([sys.executable, __file__, "--workspace", str(td),
                            "--budget", "65536", "--no-home-agents"],
                           capture_output=True, text=True)
        check("端到端：超预算夹具 ⇒ exit 1", p.returncode == 1, f"{p.returncode} {p.stdout[-200:]}")
        check("端到端：输出里出现 TRUNCATED", "TRUNCATED" in p.stdout, "")
        (td / "CLAUDE.md").write_text("y" * 100, encoding="utf-8")
        p = subprocess.run([sys.executable, __file__, "--workspace", str(td),
                            "--budget", "65536", "--no-home-agents"],
                           capture_output=True, text=True)
        check("端到端反向控制：小夹具 ⇒ exit 0", p.returncode == 0, f"{p.returncode} {p.stdout[-200:]}")

    for name, ok, detail in cases:
        print(f"  {'✅' if ok else '🔴'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
    print(f"\n{'✅' if not fails else '🔴'} selftest {len(cases) - fails}/{len(cases)}")
    return 0 if not fails else 1


MUTANTS = [
    ("M1", "渲染不再算 marker 的字节（截断点算大 ⇒ 假绿）",
     'marker_text(budget, omitted, truncated),', 'marker_text(0, [], []),'),
    ("M2", "装不下时直接返回 FULL（门禁恒绿）",
     'if nbytes(full) <= budget:', 'if True:'),
    ("M3", "OMITTED 不再点名被丢的文件",
     'omitted = [dp for dp, _ in files[:start]]', 'omitted = []'),
    ("M4", "预算取不到时判 0 而不是 2",
     'if budget is None:', 'if False:'),
    ("M5", "TRUNCATED 分支返回 0（最危险的一处：静默丢失被放行）",
     '    print("   磁盘上的文件是完整的 ⇒ 任何 diff / 门禁都看不出来；只有上下文里多一行 marker。")\n    return 1',
     '    print("   磁盘上的文件是完整的 ⇒ 任何 diff / 门禁都看不出来；只有上下文里多一行 marker。")\n    return 0'),
]


def mutate() -> int:
    src = Path(__file__).read_text(encoding="utf-8")
    # ⚠️ 变异锚点必须只在**代码主体**里唯一 —— 变异表自己会把锚点原文抄一遍，
    # 若在整份源码上数出现次数，每条都会被误报「施不上力」（S11 踩过同一个坑）。
    SPLIT = "\nMUTANTS = ["
    head, tail = src.split(SPLIT, 1)
    ok = 0
    for mid, desc, old, new in MUTANTS:
        if head.count(old) != 1:
            print(f"  🔴 {mid} 变异锚点在代码主体里不唯一（出现 {head.count(old)} 次）：{desc}")
            continue
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "mutant.py"
            m.write_text(head.replace(old, new, 1) + SPLIT + tail, encoding="utf-8")
            # 先证明变异**真的施上了力**（文本确实变了），再证明它跑起来了，最后谈判据有没有劲
            if m.read_text(encoding="utf-8") == src:
                print(f"  🔴 {mid} 变异没施上力（写出的文件与原文一致）：{desc}")
                continue
            p = subprocess.run([sys.executable, str(m), "--selftest"],
                               capture_output=True, text=True)
        ran = p.returncode in (0, 1) and "selftest" in (p.stdout + p.stderr)
        caught = p.returncode != 0
        if not ran:
            print(f"  🔴 {mid} 变异体没跑起来（这条变异没施上力）：{desc}\n     {(p.stdout + p.stderr)[-300:]}")
        elif caught:
            ok += 1
            print(f"  ✅ {mid} 抓住：{desc}")
        else:
            print(f"  🔴 {mid} 漏网：{desc}")
    print(f"\n{'✅' if ok == len(MUTANTS) else '🔴'} 变异 {ok}/{len(MUTANTS)} 抓住")
    return 0 if ok == len(MUTANTS) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(REPO))
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--budget-source", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-home-agents", action="store_true",
                    help="selftest 用：不把 ~/.dsh/AGENTS.md 计入候选")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.mutate:
        return mutate()
    if a.no_home_agents:
        global DSH_HOME
        DSH_HOME = Path(tempfile.mkdtemp())  # 空目录 ⇒ 用户级 AGENTS.md 不存在
    budget, src = (a.budget, a.budget_source or "--budget") if a.budget else find_budget()
    return run(Path(a.workspace), budget, src, a.json)


if __name__ == "__main__":
    sys.exit(main())
