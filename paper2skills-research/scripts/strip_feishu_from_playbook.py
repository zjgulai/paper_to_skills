#!/usr/bin/env python3
"""把「注入式飞书前端集成」从 playbook/ 的生成页里剥掉（2026-09-14）。

背景
----
`playbook/` 下 1496 个页面是**上游构建产物**，本仓库没有生成器。每个页面都被注入
了一组飞书前端集成：

  T1  `<a id="p2s-login-btn" href="/auth/login">飞书登录</a>`       ×1496
  T2  `window.pbShareFeishu = function(...) { ... };`               ×1495
      （其中含 `fetch('/api/feishu-callback', {...})`）
  T3  `<button onclick="pbShareFeishu(...)">推送进度到飞书</button>`  ×37（手册页）
  T4  `class='pb-feishu-bar'`（T3 那个按钮所在的外框）               ×37
  T5  `const _FEISHU_HOOK = (... __PLAYBOOK_CONFIG__.feishuHook) || '';`  agents.html
  T6  `function pushToFeishu(entry) { ... }`                        agents.html
  T7  `pushToFeishu(entry);`（调用点）                                agents.html
  T8  `const _FEISHU_HOOK_RPT=...;`                                 agent-report.html
  T9  `function pushDetailToFeishu(){ ... }`                        agent-report.html
  T10 「推送飞书」详情按钮（含内联 SVG）                              agent-report.html
  T11 `.rpt-detail-btn-feishu{...}` / `:hover` 两条 CSS              agent-report.html

⚠️ 两件必须先说清的事
--------------------
1. **这是改构建产物，不是改源码。** 上游重建会把它们注回来 —— 所以本脚本的价值不在于
   「改完就干净了」，而在于**可复现**：重跑一次即恢复到同一状态（幂等）。配套门禁
   `check_feishu_residue.py`（验收面 L17）负责每次问一句「它回来了吗」。
2. **只动集成形态，不动卡片正文。** `assets/playbook-data.json` 与少数卡片正文里有
   「推送飞书提醒人工确认」「告警推送（微信/飞书）」这类**业务场景描述** —— 那是内容不是
   集成，本脚本一个字都不碰（它们由门禁的「只扫不判」那一族列名计数）。

`/auth/login`、`/auth/me`、`/auth/logout`、`/api/reports` 这些**非飞书**的服务端调用一律保留：
它们不属于飞书集成，删了会顺手弄坏别的功能。

用法
----
  python3 strip_feishu_from_playbook.py --check      # 只报不改；有残留 exit 1
  python3 strip_feishu_from_playbook.py              # 施改写（幂等，可反复跑）
  python3 strip_feishu_from_playbook.py --selftest   # 构造样本自证每个变换 + 反向控制

退出码：0 干净/已改完 · 1 --check 下仍有残留 · 2 输入没拿到（playbook 根不存在）· 3 内部错误
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAYBOOK = REPO / "playbook"


# ---------------------------------------------------------------------------
# 花括号配对：删「整块」必须靠配对，不能靠非贪婪正则
# ---------------------------------------------------------------------------
# ⚠️ 首版 T6/T9 用的是 `function X(...) {.*?^[ \t]*\}` 非贪婪正则，**切早了**：
# 两个函数体里都有关闭嵌套对象字面量的裸 `}`（`card: { ... }` 那一行），于是删除
# 在第 1 个裸 `}` 处停止，把函数体后半截（含真正的闭括号）留在了文件里 ⇒
# `agents.html` / `agent-report.html` 的内联脚本变成 `Unexpected token '}'`。
# 是「改写后重跑 JS 语法解析、与改写前逐文件对差」这一步把它抓出来的 ——
# 教训：**删块要配对，不要靠非贪婪；改完必须做一次与改前可比的结构校验。**
def _skip_string(text: str, i: int) -> int:
    """text[i] 是引号（' " `）时，返回越过整个字符串后的下标。"""
    q = text[i]
    i += 1
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == q:
            return i + 1
        if q == "`" and c == "$" and i + 1 < len(text) and text[i + 1] == "{":
            j, depth = i + 2, 1  # 模板里的 ${...}：按花括号配对跳过
            while j < len(text) and depth:
                if text[j] in "'\"`":
                    j = _skip_string(text, j)
                    continue
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            i = j
            continue
        i += 1
    return i


def block_span(text: str, start: int) -> int:
    """从 start（块首）起做花括号配对，返回整块之后的下标（吃掉尾随 `;`/空白/换行）。

    认不出配对时返回 -1 —— 调用方必须**放弃这一次替换**而不是猜（宁可留下残留，
    也不能悄悄切坏文件；残留由 --check 与门禁抓住）。
    """
    b = text.find("{", start)
    if b < 0:
        return -1
    depth = 0
    i = b
    while i < len(text):
        c = text[i]
        if c in "'\"`":
            i = _skip_string(text, i)
            continue
        if c == "/" and i + 1 < len(text):
            if text[i + 1] == "/":
                nl = text.find("\n", i)
                i = len(text) if nl < 0 else nl
                continue
            if text[i + 1] == "*":
                end = text.find("*/", i)
                i = len(text) if end < 0 else end + 2
                continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < len(text) and text[j] in "; \t":
                    j += 1
                if j < len(text) and text[j] == "\n":
                    j += 1
                return j
        i += 1
    return -1

# ---------------------------------------------------------------------------
# 变换表：一条 = 一种形态。顺序有意义（T7 必须在 T6 之前之后都安全，见 --selftest）
# ---------------------------------------------------------------------------
_TRANSFORMS = [
    (
        "T1_login_button",
        "「飞书登录」按钮",
        # ⚠️ id 必须用 `p2s-login-btn[^"]*` 而不是写死 `p2s-login-btn`：
        # diagnostic.html 用的是 `p2s-login-btn-d`。首版写死，于是这个文件被**静默跳过**
        # （--check 报 1495 而页面里实际有 1496 个）—— 与 G2 漏洞 #2/#11「判据只认一种
        # 写法」同型。反向控制见 --selftest ①b。
        re.compile(r'^[ \t]*<a id="p2s-login-btn[^"]*"[^>]*>飞书登录</a>[ \t]*\n', re.M),
        "drop_line",
    ),
    (
        "T2_pbsharefeishu_fn",
        "pbShareFeishu 函数（含 /api/feishu-callback）",
        re.compile(r"^[ \t]*window\.pbShareFeishu = function\(pbId, pbName, totalSteps\) \{", re.M),
        "drop_block_braced",
    ),
    (
        "T3_share_button",
        "「推送进度到飞书」按钮",
        re.compile(
            r"^[ \t]*<button onclick=\"pbShareFeishu\([^\n]*?>推送进度到飞书</button>[ \t]*\n",
            re.M,
        ),
        "drop_line",
    ),
    (
        "T4_feishu_bar_class",
        "pb-feishu-bar 类名（外框留在原地，只改名）",
        re.compile(r"pb-feishu-bar"),
        "rename",
    ),
    (
        "T5_feishu_hook_const",
        "_FEISHU_HOOK 常量（读 __PLAYBOOK_CONFIG__.feishuHook）",
        re.compile(r"^[ \t]*const _FEISHU_HOOK = \(typeof window.*?;[ \t]*\n", re.M | re.S),
        "drop_line",
    ),
    (
        "T6_pushtofeishu_fn",
        "pushToFeishu 函数",
        re.compile(r"^[ \t]*function pushToFeishu\(entry\) \{", re.M),
        "drop_block_braced",
    ),
    (
        "T7_pushtofeishu_call",
        "pushToFeishu 调用点（**任意实参形态**）",
        # ⚠️ 首版写死 `pushToFeishu(entry);`，于是 agents.html 里另一处
        # `pushToFeishu({id,name:...});` 被漏掉 —— **又一次「判据只认一种写法」**。
        # 这次是被本脚本自己的「标识普查」抓出来的（见 main() 的后置断言）。
        re.compile(r"^[ \t]*pushToFeishu\(.*\);[ \t]*\n", re.M),
        "drop_line",
    ),
    (
        "T8_feishu_hook_rpt",
        "_FEISHU_HOOK_RPT 常量",
        re.compile(r"^[ \t]*const _FEISHU_HOOK_RPT=.*?;[ \t]*\n", re.M | re.S),
        "drop_line",
    ),
    (
        "T9_pushdetail_fn",
        "pushDetailToFeishu 函数",
        re.compile(r"^[ \t]*function pushDetailToFeishu\(\)\{", re.M),
        "drop_block_braced",
    ),
    (
        "T10_detail_button",
        "「推送飞书」详情按钮（含内联 SVG）",
        re.compile(
            r"^[ \t]*<button[^>]*rpt-detail-btn-feishu[^>]*>\n.*?^[ \t]*</button>[ \t]*\n",
            re.M | re.S,
        ),
        "drop_block",
    ),
    (
        "T11_detail_btn_css",
        "CSS .rpt-detail-btn-feishu（含 :hover）",
        re.compile(r"^[ \t]*\.rpt-detail-btn-feishu(:hover)?\{[^}]*\}[ \t]*\n", re.M),
        "drop_line",
    ),
]
TRANSFORMS = [(k, d, rx) for (k, d, rx, _act) in _TRANSFORMS]
_ACTION = {k: act for (k, _d, _rx, act) in _TRANSFORMS}

RENAME_FROM, RENAME_TO = "pb-feishu-bar", "pb-progress-bar"

# 判据 A 的**唯一一处定义**在 check_feishu_residue.py —— 「剥的人」与「判的人」必须看
# 同一份清单，否则两边漂移时会出现「剥完了但门禁说没剥」或反之（风险 N2：判据只许一处实现）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_feishu_residue import A_IDENTIFIERS  # noqa: E402


def _page_files() -> list[Path]:
    if not PLAYBOOK.is_dir():
        raise FileNotFoundError(f"playbook 根不存在：{PLAYBOOK}")
    return sorted(PLAYBOOK.rglob("*.html"))


def strip_text(text: str) -> tuple[str, dict[str, int]]:
    """对一段 HTML 施全部变换，返回 (新文本, {变换名: 命中次数})。"""
    counts: dict[str, int] = {}
    out = text
    for key, _desc, rx in TRANSFORMS:
        action = _ACTION[key]
        if action == "rename":
            n = len(rx.findall(out))
            if n:
                out = rx.sub(RENAME_TO, out)
        elif action == "drop_block_braced":
            # 逐个锚点向后找配对闭括号；**从后往前**删，避免位移影响前面的锚点。
            spans: list[tuple[int, int]] = []
            for m in rx.finditer(out):
                end = block_span(out, m.start())
                if end < 0:
                    continue  # 认不出配对 ⇒ 这一次不动它（残留交给 --check / 门禁）
                spans.append((m.start(), end))
            for s, e in reversed(spans):
                out = out[:s] + out[e:]
            n = len(spans)
        else:
            out, n = rx.subn("", out)
        counts[key] = n
    return out, counts


def selftest() -> int:
    """构造样本自证：每个变换真的删掉了它声称删的东西，且不碰别的。"""
    checks: list[tuple[str, bool, str]] = []

    def ck(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # ---- ① 每种形态各一份最小样本 → 必须被清掉 ----
    samples = {
        "T1_login_button": '        <a id="p2s-login-btn" href="/auth/login" style="x">飞书登录</a>\n',
        "T2_pbsharefeishu_fn": (
            "  window.pbShareFeishu = function(pbId, pbName, totalSteps) {\n"
            "    const done = document.querySelectorAll('.pb-step-check:checked').length;\n"
            "    fetch('/api/feishu-callback', {\n"
            "      method: 'POST',\n"
            "      headers: {'Content-Type': 'application/json'},\n"
            "      body: JSON.stringify({action: {value: {action: 'confirm'}}})\n"
            "    }).then(() => {\n"
            "      const btn = event.target;\n"
            "      btn.textContent = '✅ 已推送';\n"
            "    }).catch(() => {});\n"
            "  };\n"
            "  const afterwards = 1;\n"
        ),
        "T3_share_button": (
            "  <button onclick=\"pbShareFeishu('pb-x','X手册',3)\" style='a'>推送进度到飞书</button>\n"
        ),
        "T4_feishu_bar_class": "<div class='pb-feishu-bar' style='a'>\n",
        "T5_feishu_hook_const": (
            "const _FEISHU_HOOK = (typeof window !== 'undefined' && window.__PLAYBOOK_CONFIG__"
            " && window.__PLAYBOOK_CONFIG__.feishuHook) || '';\n"
        ),
        "T6_pushtofeishu_fn": (
            "function pushToFeishu(entry) {\n"
            "  if (!_FEISHU_HOOK) return;\n"
            "  const body = JSON.stringify({\n"
            "    msg_type: 'interactive',\n"
            "    card: {\n"
            "      header: {title: {tag:'plain_text', content:'x'}},\n"
            "      elements: [\n"
            "        {tag:'div', text:{tag:'plain_text', content:'y'}}\n"
            "      ]\n"
            "    }\n"
            "  });\n"
            "  fetch(_FEISHU_HOOK, {method:'POST', headers:{'Content-Type':'application/json'}, body}).catch(()=>{});\n"
            "}\n"
            "const afterwards = 2;\n"
        ),
        "T7_pushtofeishu_call": "    pushToFeishu(entry);\n",        "T8_feishu_hook_rpt": "const _FEISHU_HOOK_RPT=typeof _FEISHU_HOOK!=='undefined'?_FEISHU_HOOK:'';\n",
        "T9_pushdetail_fn": (
            "function pushDetailToFeishu(){\n"
            "  const btn=document.getElementById('rpt-detail-feishu-btn');\n"
            "  const body=JSON.stringify({\n"
            "    msg_type:'interactive',\n"
            "    card:{\n"
            "      header:{title:{tag:'plain_text',content:'x'}},\n"
            "      elements:[{tag:'hr'}]\n"
            "    }\n"
            "  });\n"
            "  fetch(_FEISHU_HOOK_RPT,{method:'POST',body})\n"
            "    .then(()=>{if(btn){btn.innerHTML='✓ 已推送';};})\n"
            "    .catch(()=>{});\n"
            "}\n"
            "// ESC 关闭\n"
        ),
        "T10_detail_button": (
            '      <button class="rpt-detail-btn rpt-detail-btn-feishu" id="b" onclick="pushDetailToFeishu()">\n'
            '        <svg width="13"></svg>\n'
            "        推送飞书\n"
            "      </button>\n"
        ),
        "T11_detail_btn_css": ".rpt-detail-btn-feishu{background:#00b96b}\n",
    }
    for key, desc, _rx in TRANSFORMS:
        sample = samples[key]
        after, counts = strip_text(sample)
        hit = counts[key] >= 1
        # 该形态自己的标识必须消失
        ident = {
            "T1_login_button": "飞书登录",
            "T2_pbsharefeishu_fn": "pbShareFeishu",
            "T3_share_button": "推送进度到飞书",
            "T4_feishu_bar_class": "pb-feishu-bar",
            "T5_feishu_hook_const": "_FEISHU_HOOK",
            "T6_pushtofeishu_fn": "pushToFeishu",
            "T7_pushtofeishu_call": "pushToFeishu",
            "T8_feishu_hook_rpt": "_FEISHU_HOOK_RPT",
            "T9_pushdetail_fn": "pushDetailToFeishu",
            "T10_detail_button": "推送飞书",
            "T11_detail_btn_css": "rpt-detail-btn-feishu",
        }[key]
        # ⚠️ 「整块」还要证明**函数体没有剩渣**，且**块后面的代码还在** ——
        # 这正是首版非贪婪正则栽的地方：标识没了，但后半截函数体与真闭括号留下了。
        vanish = {
            "T2_pbsharefeishu_fn": ["/api/feishu-callback", "JSON.stringify", "已推送"],
            "T6_pushtofeishu_fn": ["JSON.stringify", "msg_type", "application/json"],
            "T9_pushdetail_fn": ["rpt-detail-feishu-btn", "msg_type", "已推送"],
            "T10_detail_button": ["<svg"],
        }.get(key, [])
        keep = {
            "T2_pbsharefeishu_fn": "const afterwards = 1;",
            "T6_pushtofeishu_fn": "const afterwards = 2;",
            "T9_pushdetail_fn": "// ESC 关闭",
        }.get(key, None)
        ok = hit and ident not in after and all(v not in after for v in vanish)
        if keep is not None:
            ok = ok and keep in after
        ck(
            f"① {key} 命中且整块清掉（含嵌套花括号不留残渣）",
            ok,
            f"命中={counts[key]} 残渣={[v for v in vanish if v in after]}"
            + ("" if keep is None else f" 后文保留={keep in after}"),
        )

    # ---- ①b 变体容错：id 带后缀的登录按钮也必须被清掉（首版漏过 diagnostic.html） ----
    variant = '        <a id="p2s-login-btn-d" href="/auth/login" style="x">飞书登录</a>\n'
    after_v, counts_v = strip_text(variant)
    ck(
        "①b 变体容错：p2s-login-btn-d 同样被清掉（防「判据只认一种写法」）",
        counts_v["T1_login_button"] == 1 and "飞书登录" not in after_v and after_v.strip() == "",
        f"命中={counts_v['T1_login_button']}",
    )

    # ---- ①c 调用点变体：任意实参形态都要认（首版只认 `(entry)`，漏了 {..} 那处） ----
    call_variant = (
        "    pushToFeishu({id,name:ADISP[id]||id,result:ans,ts:new Date().toLocaleString('zh-CN'),inputs:inp});\n"
        "    const keep = 1;\n"
    )
    after_cv, counts_cv = strip_text(call_variant)
    ck(
        "①c 调用点变体：pushToFeishu({...}) 同样被清掉，后一行保留",
        counts_cv["T7_pushtofeishu_call"] == 1
        and "pushToFeishu" not in after_cv
        and "const keep = 1;" in after_cv,
        f"命中={counts_cv['T7_pushtofeishu_call']}",
    )

    # ---- ② 反向控制：非飞书内容**一个字节都不许动** ----
    keep = (
        "  fetch('/auth/me', {credentials: 'include'}).then(function(r){ return r.json(); });\n"
        "  var loginNode = document.getElementById('p2s-login-btn');\n"
        "  <button class=\"rpt-detail-btn\" onclick=\"copyCurrentRpt()\">复制</button>\n"
        "  <li>预期产出：排名变化实时告警（微信/飞书）</li>\n"
    )
    after_keep, counts_keep = strip_text(keep)
    ck(
        "② 反向控制：非集成内容逐字节不变（含卡片正文里的「微信/飞书」）",
        after_keep == keep and sum(counts_keep.values()) == 0,
        f"变化={after_keep != keep} 命中={sum(counts_keep.values())}",
    )

    # ---- ③ T4 是**改名不是删除**：外框与进度显示必须留下 ----
    bar = "<div class='pb-feishu-bar' style='a'>\n  <span>执行进度 <strong id=\"pb-step-done-x\">0</strong> / 3 步</span>\n"
    after_bar, _ = strip_text(bar)
    ck(
        "③ T4 只改名：pb-feishu-bar→pb-progress-bar，进度 span 与其 id 保留",
        "pb-progress-bar" in after_bar
        and "pb-feishu-bar" not in after_bar
        and 'id="pb-step-done-x"' in after_bar,
    )

    # ---- ④ 幂等：对已改过的文本再跑一次，结果必须逐字节相同 ----
    once, _ = strip_text(samples["T2_pbsharefeishu_fn"] + samples["T1_login_button"])
    twice, counts_twice = strip_text(once)
    ck("④ 幂等：二次施改写零命中且逐字节相同", twice == once and sum(counts_twice.values()) == 0)

    # ---- ⑤ 判据清单里不许出现「卡片正文会用到的词」 ----
    # 这几个词在**卡片正文**里合法存在（业务场景描述），进判据 A 会造成假红。
    bad = [w for w in ("推送飞书", "飞书") if w in A_IDENTIFIERS]
    ck("⑤ 判据 A 不含卡片正文合法词（防假红）", not bad, f"混入={bad}")

    # ---- ⑥ 字符串/模板里的花括号不许骗过配对 ----
    tricky = (
        "function pushToFeishu(entry) {\n"
        "  const s = '} not the end {';\n"
        '  const t = "}{";\n'
        "  const u = `a ${ {b:1} } c`;\n"
        "}\n"
        "const afterwards = 3;\n"
    )
    after_t, counts_t = strip_text(tricky)
    ck(
        "⑥ 字符串/模板里的花括号不许骗过配对",
        counts_t["T6_pushtofeishu_fn"] == 1
        and "afterwards = 3" in after_t
        and "not the end" not in after_t,
        f"命中={counts_t['T6_pushtofeishu_fn']}",
    )

    # ---- ⑦ 配对认不出来时**宁可不动**，不许猜 ----
    broken = "function pushToFeishu(entry) {\n  const x = 1;\n"  # 没有闭括号
    after_b, counts_b = strip_text(broken)
    ck(
        "⑦ 无法配对时保留原文（宁可留残留给门禁抓，也不许悄悄切坏文件）",
        after_b == broken and counts_b["T6_pushtofeishu_fn"] == 0,
    )

    ok_n = sum(1 for _n, o, _d in checks if o)
    print(f"strip_feishu_from_playbook --selftest: {ok_n}/{len(checks)}")
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{detail}]" if detail else ""))
    return 0 if ok_n == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="剥离 playbook 生成页里的飞书前端集成")
    ap.add_argument("--check", action="store_true", help="只报不改（有残留 exit 1）")
    ap.add_argument("--selftest", action="store_true", help="构造样本自证")
    ap.add_argument("--json-out", metavar="PATH", help="把读数写成 JSON")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    try:
        files = _page_files()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 2

    totals = {k: 0 for k, _d, _rx in TRANSFORMS}
    touched: list[str] = []
    changed_bytes = 0

    for p in files:
        try:
            before = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"⚠️ 非 UTF-8，跳过：{p.relative_to(REPO)}")
            continue
        after, counts = strip_text(before)
        for k, n in counts.items():
            totals[k] += n
        if after != before:
            touched.append(str(p.relative_to(REPO)))
            changed_bytes += len(before.encode()) - len(after.encode())
            if not args.check:
                p.write_text(after, encoding="utf-8")

    residue = sum(totals.values())

    # ⚠️ 后置断言：**变换命中数只是仪器的视角，标识普查才是地面真值。**
    # 两者不一致 ⇒ 有形态没被认出来（本项目已三次踩到「判据只认一种写法」：
    # T1 漏 p2s-login-btn-d、T7 漏 pushToFeishu({...}) 的另一种实参形态）。
    census: dict[str, int] = {}
    for ident in A_IDENTIFIERS:
        n = 0
        for p in files:
            try:
                n += p.read_text(encoding="utf-8").count(ident)
            except UnicodeDecodeError:
                pass
        if n:
            census[ident] = n

    print(f"playbook 页面 {len(files)} 个 · 变动 {len(touched)} 个 · 命中合计 {residue}")
    for k, desc, _rx in TRANSFORMS:
        if totals[k]:
            print(f"  {k:<26} ×{totals[k]:<5} {desc}")
    if census:
        print("\n⚠️ 标识普查仍有残留（变换表没认出的形态）：")
        for ident, n in census.items():
            print(f"    {ident:<24} ×{n}")
        hits = [str(p.relative_to(REPO)) for p in files
                if any(ident in p.read_text(encoding="utf-8", errors="replace") for ident in census)]
        print(f"    涉及 {len(hits)} 个文件 —— 例：{hits[:3]}")
        print("    ⇒ 这说明变换表缺一种写法。**修变换表，不要手改文件。**")
    if args.check and (touched or census):
        print(f"\n❌ 仍有集成残留（{len(touched)} 个文件需改写 · 普查残留 {len(census)} 种标识）")
        print("   修法：直接跑本脚本（无 --check）施改写，然后重跑门禁。")
    elif args.check:
        print("\n✅ playbook 里没有集成残留")
    elif touched:
        print(f"\n✅ 已改写 {len(touched)} 个文件（净减 {changed_bytes} B）")
        if not census:
            print("   ✅ 标识普查确认：10 个集成标识全部归零")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "pages": len(files),
                    "touched": len(touched),
                    "hits": totals,
                    "residue": residue,
                    "census": census,
                    "mode": "check" if args.check else "apply",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # 普查有残留 ⇒ 无论 --check 还是已施改写，都判红（施加了却没收干净同样是失败）
    if census:
        return 1
    if args.check and touched:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"💥 内部错误：{type(e).__name__}: {e}")
        sys.exit(3)
