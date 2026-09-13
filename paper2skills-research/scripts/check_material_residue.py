#!/usr/bin/env python3
"""材料引文**残留**扫描（PHASE6 S1-W 收口期，2026-09-13，主控）

## 为什么要有这个脚本 —— 它诞生于一次自己的假绿

`check_material_citations.py`（门禁 L4e）现报 **exit 0「全部材料引文有据」**。
而直接 `grep` 同一批 139 份契约，实测仍有：

| 家族 | 实测残留 |
|---|---|
| `「季度经营策略」`（材料真词是「月度经营复盘」） | **49 处 / 40 份** |
| `材料 §F.3` / `材料 §F.5`（**§F.x 是我方综述 `_survey_org_model.md` 的章节号，材料里没有**） | **19 处 / 15 份** |
| `缺一项即 HOLD` 一族（声称材料定名的门禁名） | **10 处 / 10 份** |

⇒ **L4e 的绿不能读作「没有残留」。** 本脚本存在的唯一目的，是把这批残留
变成**看得见的数**，直到 L4e 自己把它们收进去。

## 为什么 L4e 看不见它们（机制，逐条实测）

`check_material_citations.py::extract()` 的**扫描单元是「行」**（`for ln_no, line in enumerate(...)`），
且判据三重收窄：

1. `材料` 必须与引号**同一行**；引号出现在 blockquote **续行**上 ⇒ `pos < 0` ⇒ **静默跳过**。
   实测 `CTR-A-048:39`：`> 「季度经营策略」＝1 个自然季度；阶段边界 STG-01…STG-08）；` —— 全行没有 `材料`。
2. `head` 还要按 `[。；\n]` 切成最后一段，**枚举里的 `；` 会把管辖它的头部切掉**。
   实测 `CTR-A-067:35`：`（a）业务处境或材料已给出的数（…Q11；经营节奏「月度经营复盘」＝1 个自然月、` ——
   二分号之后 `材料` 已不在同一段。
3. `run` 一旦含引号就 `continue`（那条守卫是为「一行多引号」加的）⇒
   **`材料已给出的数（A、B、C）` 这种枚举里，只有紧随材料的第一个引号会被判，后面全是盲区**。

⚠️ 这三条与台账 **#67** 同族，且与 **D27** 互为镜像：
D27 是「豁免只认**词**不认**话题**」⇒ 收紧到**条目块**；
本条是「声称只认**同行**不认**条目块**」⇒ 该放宽到**条目块**。
**同一条颗粒度教训，两个相反方向。** 本仓库已两次因颗粒度（列 vs 行 vs 块）翻面判决，
而两次都是**假**的那一面 —— 故本脚本**不动 L4e 的判定**，只把残留单独报出来。

## 本脚本的判据（故意比 L4e 宽，且**只报不判归属**）

扫描单元 = **blockquote 条目块**（连续 `>` 行；空 `>` 行分段）。
块内出现 `材料` ⇒ 该块内的 `「X」` 是**候选材料声称**，除非：
· `X` 之前、同一块内最近的**分句头**里有**去归属语**（`本契约`/`本项目`/`综述`/`二手来源`/`业务侧默认`/`行业惯例`/`决策 Q`/`不是材料`/`未给`），或
· `X` 本身在材料里**确实存在**（那就不是残留）。

另扫第二家族：`材料` 后跟章节号（`§X` / `§F.x` / `第X章`）—— 章节号一律可疑，
因为材料没有我们综述的章节号；命中即列出，由人复核。

## 退出码（沿用本仓库四态，**不许合并**）

`0` 无残留 · `1` 有残留 · `2` **输入没拿到**（材料根/契约目录不存在，或扫到 0 个文件）· `3` 仪器内部错误

⚠️ **`2` 不是 `0`**：没扫到 ≠ 干净。本仓库已有 `scan_secrets.py`（扫到 0 文件判失败）与
`run_phase6_gates.py`（一个门禁都没跑到 = exit 3）两条先例。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MATERIAL = Path("/Users/lute/project/AI组织变革")
DEFAULT_CONTRACTS = REPO / "paper2skills-vault" / "07-资源库" / "contracts"

OPEN_Q = "「『“‘"
CLOSE_Q = "」』”’"
QUOTE_RE = re.compile(f"[{OPEN_Q}]([^{CLOSE_Q}]{{1,40}})[{CLOSE_Q}]")

# 「材料」这个词本身
MATERIAL = "材料"

# 材料后跟章节号 —— 一律可疑（材料没有我们综述的 §F.x 章节号）
SECTION_RE = re.compile(r"材料\s*(?:§\s*[A-Za-z0-9.\-]+|第\s*[0-9一二三四五六七八九十]+\s*[章节])")

# 去归属语：出现在 X **之前**的同一分句头里 ⇒ 该 X 不是「材料声称」
DEATTRIB = (
    "本契约", "本项目", "本责任", "本行", "本方案", "方案域",
    "综述", "二手来源", "不是材料", "非材料", "材料未给", "材料没有",
    "业务侧默认", "行业惯例", "决策Q", "决策 Q", "开放事实",
)


def normalize(s: str) -> str:
    """NFKC + 去全部空白。

    ⚠️ 与 `check_material_citations.py::normalize()` 同尺（那两个脚本的结论要能对比）。
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def strip_punct(s: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s)


def load_corpus(root: Path) -> tuple[str, int]:
    """读材料语料（归一化后拼成一整串）。

    ⚠️ **必须排掉 `.git/`** —— 材料目录自己是个 git 仓库，历史 blob 里的词会被判「有据」
    （`check_material_citations.py` 首版就踩过这一条，见台账）。
    """
    if not root.exists():
        raise FileNotFoundError(f"材料根不存在：{root}")
    parts, n = [], 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue
        if p.suffix.lower() not in (".md", ".txt", ".json", ".yml", ".yaml", ".csv"):
            continue
        try:
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
            n += 1
        except OSError:
            continue
    if n == 0:
        raise FileNotFoundError(f"材料根下没读到任何文本文件：{root}")
    return normalize("\n".join(parts)), n


def blockquote_items(text: str):
    """把文件切成 blockquote 条目块：连续 `>` 行成一块；空 `>` 行分段。

    产出 `(起始行号, 块文本)`。块文本**保留原行**以便报行号，但把 `>` 标记与换行
    换成空格 —— 因为管辖关系跨行成立，断行不该斩断它。
    """
    lines = text.splitlines()
    items, buf, start = [], [], None
    for i, line in enumerate(lines, 1):
        s = line.lstrip()
        if s.startswith(">"):
            content = s[1:]
            if content.strip() == "":
                if buf:
                    items.append((start, " ".join(buf)))
                    buf, start = [], None
                continue
            if start is None:
                start = i
            buf.append(content.strip())
        else:
            if buf:
                items.append((start, " ".join(buf)))
                buf, start = [], None
    if buf:
        items.append((start, " ".join(buf)))
    return items


OPEN_P, CLOSE_P = "（(", "）)"


def attribution_spans(block: str):
    """给块内每一处 `材料` 算一个**归属跨度**：它到「包住它的那个括号」的右括号为止。

    为什么不是「整块」：块粒度过宽会把**同一个 blockquote 里无关的缺口描述**也算成材料声称。
    实测（本脚本首版就踩了）：整块口径把「这个商品该归哪个编码」「这份证书能不能过」
    「允许／禁止」这类**本契约自己的缺口问句**报成残留 ⇒ 78 处里大部分是假阳性。
    ⇒ 收紧到**括号跨度**：`材料已给出的数（A、B、C）` 里 A/B/C 全归材料；括号之外的引号不归。

    为什么不是「同一行」：`材料` 与枚举项常分处两行（L4e 的盲区即在此），
    而括号跨度**天然跨行**——这正是要修的颗粒度。

    ⚠️ 没有括号可依时（`材料` 后面直接跟引号），回退到「同一分句头」，与 L4e 同尺。
    """
    spans = []
    for m in re.finditer(re.escape(MATERIAL), block):
        i = m.start()
        # ⚠️ 关键是**材料之后**、很近处那个左括号 —— 不是「材料处已开着的括号」。
        #    实测反例：`材料已给出的数（出海历史…；…「月度经营复盘」…、「季度经营策略」…）`
        #    里 `材料` 所在位置**没有任何括号是开着的**（`（a）` 已在它之前闭合），
        #    按「已开着的括号」取 ⇒ 取不到 ⇒ 回退到同一分句 ⇒ 正好丢掉要抓的那两个引号。
        #    首版就是这么写的，实测把真缺陷丢了、只留下假阳性。
        nxt = None
        for k in range(i, min(len(block), i + 24)):
            if block[k] in OPEN_P:
                nxt = k
                break
            if block[k] in "。；\n":
                break
        if nxt is not None:
            d = 0
            for k in range(nxt, len(block)):
                if block[k] in OPEN_P:
                    d += 1
                elif block[k] in CLOSE_P:
                    d -= 1
                    if d == 0:
                        spans.append((i, k))
                        break
            else:
                spans.append((i, len(block)))
        else:
            tail = block[i:]
            stop = len(block)
            for mm in re.finditer(r"[。；\n]", tail):
                stop = i + mm.start()
                break
            spans.append((i, stop))
    return spans


def material_sections(root: Path) -> set[str]:
    """材料里**真实存在**的章节号集合（`E.2` / `F.3` / `1.2` …），从 Markdown 标题里抽。

    ⚠️ 为什么必须实测而不能「一律可疑」：材料《AI组织变革》确实有自己的编号章节。
    本脚本首版把 `材料 §X` 一律判可疑，那是**没问「仪器能不能看见」**就下结论 ——
    与 CLAUDE.md 铁律 1 同族。现在是**逐号对材料核实**：材料有 ⇒ 不算残留；材料没有 ⇒ 残留。
    """
    secs: set[str] = set()
    if not root.exists():
        raise FileNotFoundError(f"材料根不存在：{root}")
    for p in sorted(root.rglob("*.md")):
        if ".git" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = re.match(r"\s{0,3}#{1,6}\s*(?:§\s*)?([A-Za-z]?\d+(?:\.\d+)*)\b", line)
            if m:
                secs.add(m.group(1))
    return secs


def scan_file(path: Path, corpus: str, sections: set[str]):
    """返回本文件的两类残留：`term_residue`（待定标）与 `section_residue`（已对材料核实）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    term_res, sect_res = [], []
    # 家族二**扫全文，不限于 blockquote** —— `材料 §F.5` 大量出现在正文里
    # （实测 `CTR-A-018:143` 是正文、`CTR-A-020:26` 才是引文块）。只扫引文块会系统性少算。
    for i, line in enumerate(text.splitlines(), 1):
        for m in SECTION_RE.finditer(line):
            hit = m.group(0).strip()
            num = re.sub(r"^材料\s*(?:§\s*)?", "", hit).strip()
            if num in sections:
                continue          # 材料真有这一节 ⇒ 不是残留
            sect_res.append({"line": i, "hit": hit, "num": num})
    for ln, block in blockquote_items(text):
        if MATERIAL not in block:
            continue
        spans = attribution_spans(block)
        for qm in QUOTE_RE.finditer(block):
            term = qm.group(1).strip()
            if not strip_punct(term):
                continue
            covering = [s for s in spans if s[0] <= qm.start() < s[1]]
            if not covering:
                continue
            # 去归属语：只看 `材料` **紧邻**的窗口（前后各 12 字）。
            # ⚠️ 首版看的是「材料 → 引号」的整段，实测**把真缺陷丢了**：
            #    `材料已给出的数（…＝决策 Q11；…「月度经营复盘」…、「季度经营策略」…）`
            #    里 `决策 Q` 是**并列的枚举项**、不是对整段的去归属，却把整段判掉了。
            #    收紧到邻域后：`本契约把材料未命名的…` 仍被正确判为去归属（`本契约` 在材料前 6 字内），
            #    而 `决策 Q11`（距材料 18 字）不再误杀。
            lo = max(s[0] for s in covering)
            gov = normalize(block[max(0, lo - 12): lo + 12])
            if any(d in gov for d in DEATTRIB):
                continue
            if normalize(term) in corpus:
                continue
            term_res.append({"line": ln, "term": term})
    return term_res, sect_res


def collect(contracts: Path):
    files = sorted(list(contracts.glob("A/CTR-*.md")) + list(contracts.glob("B/CTR-*.md")))
    if not files:
        raise FileNotFoundError(f"契约目录里没扫到文件：{contracts}")
    return files


def run(contracts: Path, material: Path, quiet: bool = False) -> int:
    try:
        corpus, n_mat = load_corpus(material)
        sections = material_sections(material)
        files = collect(contracts)
    except FileNotFoundError as e:
        print(f"❌ {e} —— 「没拿到输入」不等于「通过」", file=sys.stderr)
        return 2

    per_file, term_counter, sect_counter = {}, {}, {}
    for p in files:
        t, s = scan_file(p, corpus, sections)
        if t or s:
            per_file[p.name] = {"term": t, "section": s}
        for x in t:
            term_counter.setdefault(x["term"], []).append(p.name)
        for x in s:
            sect_counter.setdefault(x["hit"], []).append(p.name)

    if not quiet:
        print(f"材料根：{material}（{n_mat} 个文件，已排 `.git/`）· 扫描 {len(files)} 份契约")
        print(f"扫描单元：**blockquote 条目块**（连续 `>` 行成块）—— 故意比 L4e 的「行」宽\n")
        if term_counter:
            print(f"❌ 家族一 · 块内声称「材料」而词在材料里查无实据："
                  f"**{sum(len(v) for v in term_counter.values())} 处 / "
                  f"{len({f for v in term_counter.values() for f in v})} 份 / {len(term_counter)} 个词**")
            for term, fs in sorted(term_counter.items(), key=lambda kv: -len(kv[1])):
                print(f"    · 「{term}」 ×{len(fs)} 份：{'、'.join(sorted(set(fs))[:8])}"
                      f"{' …' if len(set(fs)) > 8 else ''}")
        if sect_counter:
            print(f"\n❌ 家族二 · 声称「材料 §X / 第X章」："
                  f"**{sum(len(v) for v in sect_counter.values())} 处 / "
                  f"{len({f for v in sect_counter.values() for f in v})} 份**")
            for hit, fs in sorted(sect_counter.items(), key=lambda kv: -len(kv[1])):
                print(f"    · `{hit}` ×{len(fs)} 份")
        if not term_counter and not sect_counter:
            print("✅ 无残留（家族一、家族二均为 0）")

    total = sum(len(v) for v in term_counter.values()) + sum(len(v) for v in sect_counter.values())
    if not quiet:
        print(f"\n⇒ 残留合计 **{total} 处** —— exit {1 if total else 0}")
    return 1 if total else 0


# --------------------------------------------------------------------------- #
# 自检：每条判据都要有能打红的样本，且必须有**反向控制**（不许恒红）
# --------------------------------------------------------------------------- #

CLEAN_FIXTURE = """---
template_version: v2
---

> **取值来源纪律（全文适用）**：阈值来源只有三类 ——
> （a）材料已给出的数（经营节奏「月度经营复盘」＝1 个自然月）；
> （b）**业务侧默认**（本契约自定的门禁名，材料未命名）；
> （c）本契约 §3 算式的输出。
"""

# ① 续行形态：引号在 blockquote 续行上，与「材料」**不在同一行**
RESIDUE_CONT = """---
template_version: v2
---

> **取值来源纪律（全文适用）**：阈值来源只有三类 ——
> （a）业务处境或材料已给出的数（经营节奏「月度经营复盘」＝1 个自然月、
> 「季度经营策略」＝1 个自然季度）；
"""

# ② 枚举形态：材料之后有 `；` 与多个引号，后面那些是 L4e 的盲区
RESIDUE_ENUM = """---
template_version: v2
---

> **取值来源纪律（全文适用）**：来源只有三类 ——
> （a）材料已命名的节奏（「月度经营复盘」＝1 个自然月；「季度经营策略」＝1 个自然季度）；
"""

# ③ 章节号形态
RESIDUE_SECTION = """---
template_version: v2
---

> **本责任同时是材料 §F.5 的 A 类边界条目**，故降级条件明写在这里。
"""

# ④ 反向控制：引号在块内，但**去归属语**明说它不是材料给的 ⇒ 不得报
DEATTRIB_FIXTURE = """---
template_version: v2
---

> **取值来源纪律**：本契约把材料未命名的门禁落成数 ——「缺一项即 HOLD」是本契约的读法，
> 不是材料原文。其余阈值取自材料已给出的「月度经营复盘」。
"""


def selftest() -> int:
    import tempfile
    cases = []

    def scan_one(body: str, material_files: dict):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mroot = root / "material"
            mroot.mkdir()
            for name, txt in material_files.items():
                (mroot / name).write_text(txt, encoding="utf-8")
            croot = root / "contracts"
            (croot / "A").mkdir(parents=True)
            (croot / "A" / "CTR-A-999-夹具.md").write_text(body, encoding="utf-8")
            code = run(croot, mroot, quiet=True)
            return code

    # 材料里只有「月度经营复盘」，没有「季度经营策略」、没有 §F.5
    MAT = {"m.md": "经营节奏：月度经营复盘＝1 个自然月。门禁：接收门禁。"}

    cases.append(("① 干净夹具 ⇒ exit 0（反向控制，防恒红）", scan_one(CLEAN_FIXTURE, MAT) == 0))
    cases.append(("② 续行形态残留 ⇒ exit 1（L4e 看不见的那一类）",
                  scan_one(RESIDUE_CONT, MAT) == 1))
    cases.append(("③ 枚举形态残留 ⇒ exit 1", scan_one(RESIDUE_ENUM, MAT) == 1))
    cases.append(("④ `材料 §F.5` 章节号 ⇒ exit 1", scan_one(RESIDUE_SECTION, MAT) == 1))
    cases.append(("⑤ 去归属语在块内 ⇒ exit 0（不许把「本契约的读法」算成材料声称）",
                  scan_one(DEATTRIB_FIXTURE, MAT) == 0))

    # ⑥ 输入没拿到 ⇒ exit 2，**不是 0**
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "c" / "A").mkdir(parents=True)
        (root / "c" / "A" / "CTR-A-999-夹具.md").write_text(CLEAN_FIXTURE, encoding="utf-8")
        cases.append(("⑥ 材料根不存在 ⇒ exit 2（≠ 0）", run(root / "c", root / "nope", quiet=True) == 2))
        mroot = root / "m"
        mroot.mkdir()
        (mroot / "m.md").write_text("x", encoding="utf-8")
        empty = root / "empty"
        (empty / "A").mkdir(parents=True)
        cases.append(("⑦ 契约目录里 0 份 ⇒ exit 2（≠ 0）", run(empty, mroot, quiet=True) == 2))

    ok = True
    for label, passed in cases:
        print(f"  {'✅' if passed else '❌'} {label}")
        ok &= passed
    print(f"\n{'✅' if ok else '❌'} selftest {sum(1 for _, p in cases if p)}/{len(cases)}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="材料引文残留扫描（PHASE6 S1-W 收口期）")
    ap.add_argument("--material", default=str(DEFAULT_MATERIAL), help="材料根目录")
    ap.add_argument("--contracts", default=str(DEFAULT_CONTRACTS), help="契约目录")
    ap.add_argument("--json-out", default="", help="机读产物路径")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    code = run(Path(a.contracts), Path(a.material), quiet=a.quiet)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"exit": code}, ensure_ascii=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
