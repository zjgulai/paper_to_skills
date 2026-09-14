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

## `--family`：两个家族的定标状态不同，**判定必须分开**

| 家族 | 定标状态 | 能否进验收面 |
|---|---|---|
| **一 · 词**（块内声称「材料」而词查无实据） | **未定标**（24 个词里只有 `季度经营策略` 经独立确认是真阳性，其余大部分逐条看下来仍是假阳性） | ❌ **不许**（把一个未定标的仪器接进门禁 = 制造噪声，而噪声会被忽略 —— 台账 #67） |
| **二 · 章节号**（声称「材料 §X / 第X章」而材料没有该编号） | **已逐号对材料核实**（材料编号章节只有 `1–12` 与 `R01–R05`；`§E.`/`§F.` 是我方综述的章节号） | ✅ **已接**（`--family sections` ⇒ 验收面 **L4k**） |

`--family sections` 下**家族一照常扫描、但不参与退出码**，并**在输出里明写它未被判定**
（未判定的那一维必须看得见，否则「没测到」会被读成「测了是干净的」—— 本仓库台账 #23 同族）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

FAMILIES = ("all", "terms", "sections", "form")

# 同尺归一化：**引用**引文器的那一份实现，不另抄（台账 #79）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_material_citations import normalize as _citations_normalize  # noqa: E402

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
    """**必须与 `check_material_citations.py::normalize()` 同尺** —— 故直接**引用**它，不另抄一份。

    ⚠️ 2026-09-13 实测（台账 **#79**）：本函数原先只做「NFKC + 去空白」，
    而引文器的同门函数**还剥 markdown 强调符**（`*_`>#|~`）。两者被 docstring 声明为「同尺」，
    **实际不同尺** —— 后果是**家族一 20 个词里 13 个是假阳性**：
    材料里**逐字就有**，只是契约在引文里写了 `**复用效果和过期知识控制**`，
    去空白后带 `**` 的串当然不在语料里 ⇒ 被判「查无实据」。
    ⇒ 教训与 `check_material_citations.py` 记的那条同源（`str.translate({c: None})` 静默不做事）：
      **「同尺」不能靠注释声明，要有判据**（selftest 用例⑯直接断言两个函数相等）。
    """
    return _citations_normalize(s)



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
            # 去归属语**还要看词自己身边**（前后各 12 字）—— 2026-09-13 实测补上（台账 **#80**）。
            #
            # ⚠️ 为什么必须两处都看：`材料已给出的数（…）` 的**括号跨度**是个**语法容器**，
            #    它把整个枚举**都**算成了材料的声称 —— 而枚举里的每一项**自带归属**：
            #    实测 4 处假阳性全在这个形态里，且每处的归属就写在**词自己旁边**：
            #      · `独立站 holdback「没有、可以建、但需先建」＝**决策 Q10**；`   （×3 份）
            #      · `「…其余 20% 渠道构成未明」＝**开放事实 O1**）；`
            #    只查 `材料` 邻域 ⇒ 这 4 处都被判成「声称材料说了」。
            #    两处都查 = 「**容器说是材料的 ≠ 容器里每一项都是材料的**」。
            # ⚠️ 反向风险（本仓库亲手踩过，D27）：豁免只认词不认话题 ⇒ 假绿。
            #    故窗口**只有 12 字**、且配套 selftest ⑲/⑳ 一正一反：
            #    带 `＝决策 Q10` 的不报，**去掉那三个字后必须照旧报**。
            tstart, tend = qm.start(1), qm.end(1)
            near_term = normalize(block[max(0, tstart - 12): tend + 12])
            if any(d in near_term for d in DEATTRIB):
                continue
            if normalize(term) in corpus:
                continue
            term_res.append({"line": ln, "term": term})
    return term_res, sect_res


FORM_CLAUSE_B = re.compile(r"(?:^|[；。])\s*(?:>\s*)?（b）", re.MULTILINE)
FORM_CLAUSE_A = re.compile(r"(?:^|[；。])\s*(?:>\s*)?（a）", re.MULTILINE)
FORM_CLAUSE_ANY = re.compile(r"(?:^|[；。])\s*(?:>\s*)?（[a-d]）", re.MULTILINE)
FORM_NEG = re.compile(
    r"材料[^；。）\n]{0,14}?(?:没有|未给|未定名|不在|并非|只有|只到|只写|不由|不属|侧无|里没有)"
    r"|季度[^；。）\n]{0,12}?(?:不在材料|不由材料|并非材料|不属材料)")
FORM_ITEM = "＝1 个自然季度"
FORM_DEFAULT = "业务侧默认"


def section0_columns(text: str):
    """取 §0 的 (a) 栏与**「业务侧默认」那一子句**。

    返回 `(a 栏, 默认子句, §0 全文, 有没有 §0 块)`。

    ⚠️ 三处都不能想当然：
    · 不能用 `blockquote_items()` —— 它把块内各行并成一行，而分栏靠的正是这些标记；
    · **不能只认行首**的 `> （b）` —— 实测 139 份里有 **4 份**把
      `（a）…）；（b）业务侧默认…；（c）…` 全写在**同一行**（A-018 / B-017 / B-018 / B-054）；
    · **更不能假定「业务侧默认」那一栏一定叫 (b)** —— 实测有 **3 份**把它排在 (c)
      （`（b）材料已定名的规则与产物…；（c）业务侧默认…`，如 CTR-A-062）。
      按 (b) 硬取会把 A-062 判成「(b) 栏没收该值」，而它的账其实收得好好的。
    ⇒ 分栏按**子句标记**切，取「含 `业务侧默认` 的那一段」，不认字母。
    """
    blocks, cur = [], []
    for ln in text.splitlines():
        if ln.lstrip().startswith(">"):
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    for b in blocks:
        j = "\n".join(b)
        if "取值来源纪律" not in j:
            continue
        ms = list(FORM_CLAUSE_ANY.finditer(j))
        ma = FORM_CLAUSE_A.search(j)
        a_span = ""
        if ma:
            nxt = next((m.start() for m in ms if m.start() >= ma.end()), len(j))
            a_span = j[ma.end():nxt].strip()
        for k, m in enumerate(ms):
            seg = j[m.end():(ms[k + 1].start() if k + 1 < len(ms) else len(j))]
            if FORM_DEFAULT in seg:
                return a_span, seg.strip(), j, True
        return a_span, "", j, True
    return None, None, None, False


def _is_clause_marker(a: str, at: int) -> bool:
    """`a[at:]` 以 `（b）` 开头时，它到底是**子句标记**还是**指向 (b) 的指针**？

    判据只有一个字：**前一非空字符是不是 `；` 或 `。`**。
      · `…＝开放事实 O1）；（b）业务侧默认…`  ⇒ 是子句标记（实测 4 份这么写）
      · `…本档改记（b）**业务侧默认**；…`    ⇒ 是指针（前一字符是 `记`）

    ⚠️ **不许**把「行首」也算成子句标记。首版用了带 `^` 的正则，在只回看 1 个字符的切片上
    `^` 恒真，于是 `改记（b）` 被判成子句标记 ⇒ **指针一处都抓不到**（假绿方向）。
    这里连正则都不用，就是为了让这个判别只有一条规则、没有第二种解释。
    `a` 是多行拼起来的，指针前的实词一定在（`记`/`转`/`归`/`并`/`改`），不会落在行首。
    """
    i = at - 1
    while i >= 0 and a[i] in " \t":
        i -= 1
    return i >= 0 and a[i] in "；。"


def scan_form(path: Path) -> list[dict]:
    """第三族 · **归属形态**：一个值必须住在它自己那一栏里。

    四条判据**分开报**（问的是四个不同的问题）：

    · `pointer`   —— (a) 栏里有**指向 (b) 的指针**（`⇒ 改记（b）` 这类）。项搬到 (b) 之后，
                     这句话就成了一句空话（在 (b) 里说「改记 (b)」），而它又是**人写的、机器不验的**
                     交叉引用 ⇒ 判红。
                     ⚠️ 判法**不能**是「(a) 栏里找 `（b）` 子串」：实测 139 份里有 **4 份**把
                     `（a）…）；（b）业务侧默认…` 写在同一行，那是**子句标记**不是指针。
                     首版按子串判 ⇒ 那 4 份全成了假红。
    · `negative`  —— (a) 栏里有**负向材料归属语**（`材料没有季度档` 这类）。这一栏的抬头写着
                     「业务处境或材料已给出的数」，栏里的项却自己说材料没有 ⇒ **类目矛盾**：
                     读的人只看抬头。
    · `misplaced` —— §0 里**有** `＝1 个自然季度`，但它**不在**含 `业务侧默认` 的那一子句里
                     ⇒ 挂错了栏。实测 10 份：附录被插在 `…；（c）§3 算式的输出（由命名的数据系统直出）`
                     **尾巴上**，读起来这一项属于 (c)，而 (c) 是「算式直出」，与业务侧默认无关。
    · `unfiled`   —— 正文用了标着 `业务侧默认` 的 `1 个自然季度`，而 §0 **一个字都没提**这个值。
                    实测 1 份（CTR-A-054）：它既没搬错位置、也没留下指针 ⇒
                    前两次按关键词的机械对差**都没看见它**。

    ⚠️ `misplaced` 与 `unfiled` 必须分开：合成一条时，前半段的「挂错栏」会被后半段的
    「正文有没有用这个值」当成门禁 —— 实测 10 份里只有 5 份同时满足，另 5 份**因此漏判**。
    一个判据里塞两个必要条件，等于把判据的适用面悄悄缩到两者的交集上。

    另有第五种读数 `no-section0`（没有 §0 块）：**本族无从判定**，只报不判 ——
    「未判定 ≠ 干净」，故它必须可见（台账 #67）。
    """
    text = path.read_text(encoding="utf-8")
    a, default_seg, s0, has_s0 = section0_columns(text)
    if not has_s0:
        return [{"kind": "no-section0", "detail": "全文没有 §0「取值来源纪律」块"}]
    if a is None:
        return [{"kind": "no-section0", "detail": "有 §0 块但没有 (a) 栏"}]
    out: list[dict] = []
    for m in re.finditer(r"（b）", a):
        if not _is_clause_marker(a, m.start()):
            out.append({"kind": "pointer",
                        "detail": a[max(0, m.start() - 26):m.end() + 10].replace("\n", " ")})
    for m in FORM_NEG.finditer(a):
        out.append({"kind": "negative", "detail": m.group(0)})
    has_item = FORM_ITEM in (s0 or "")
    if has_item and not default_seg:
        # §0 里有这个值，却**没有「业务侧默认」这一栏** ⇒ 本族无从判定它该住哪
        # （不是「挂错了」—— 没有可挂的栏）。只报不判，与 no-section0 同类。
        return [{"kind": "no-default-clause",
                 "detail": "§0 里有「＝1 个自然季度」，但没有含「业务侧默认」的子句 ⇒ 无处可住，本族无从判定"}]
    if has_item and FORM_ITEM not in default_seg:
        out.append({"kind": "misplaced",
                    "detail": "§0 里有「＝1 个自然季度」，但不在含「业务侧默认」的那一子句里"})
    if not has_item:
        body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(">"))
        for para in re.split(r"\n\s*\n", body):
            if "1 个自然季度" in para and FORM_DEFAULT in para:
                out.append({"kind": "unfiled",
                            "detail": "正文有标着业务侧默认的「1 个自然季度」，而 §0 没有「＝1 个自然季度」"})
                break
    return out


def collect(contracts: Path):
    files = sorted(list(contracts.glob("A/CTR-*.md")) + list(contracts.glob("B/CTR-*.md")))
    if not files:
        raise FileNotFoundError(f"契约目录里没扫到文件：{contracts}")
    return files


def apply_baseline(
    term_counter: dict, baseline: dict | None
) -> tuple[dict, list[dict], list[dict]]:
    """按 `--baseline` 豁免**已知假阳性**。返回 (留下的, 被豁免的, **用不上了的**)。

    ⚠️ 第三项是关键。豁免的语义是「这一处**已逐处定标为假阳性**，不是待办」，
    所以一旦判据收窄到它不再报，这条豁免就**过期**了 —— 过期的豁免会腐烂成永久后门（台账 #5）。
    过期**判红而不是只打印提示**：验收面里绿门禁的 stdout **根本不显示**，
    在绿门禁里喊「请取消豁免」＝**没人看得见**。故提示必须走退出码。
    """
    if not baseline:
        return term_counter, [], []
    keys = {(it.get("term"), it.get("file")) for it in baseline.get("items", [])}
    kept: dict = {}
    waived: list[dict] = []
    matched: set = set()
    for term, fs in term_counter.items():
        for f in fs:
            if (term, f) in keys:
                waived.append({"term": term, "file": f})
                matched.add((term, f))
            else:
                kept.setdefault(term, []).append(f)
    stale = [it for it in baseline.get("items", []) if (it.get("term"), it.get("file")) not in matched]
    return kept, waived, stale


def run(contracts: Path, material: Path, quiet: bool = False, family: str = "all",
        baseline: dict | None = None, baseline_path: str = "") -> int:
    if family not in FAMILIES:
        raise SystemExit(f"未知 --family {family!r}，只认 {FAMILIES}")
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

    form_counter: dict[str, list[tuple[str, str]]] = {}
    not_judged: list[tuple[str, str]] = []      # 「本族无从判定」的读数（不参与退出码，但必须可见）
    NOT_JUDGED = ("no-section0", "no-default-clause")
    for p in files:
        for h in scan_form(p):
            if h["kind"] in NOT_JUDGED:
                not_judged.append((p.name, h["kind"]))
            else:
                form_counter.setdefault(h["kind"], []).append((p.name, h["detail"]))

    judge_terms = family in ("all", "terms")
    judge_sects = family in ("all", "sections")
    judge_form = family in ("all", "form")

    waived: list[dict] = []
    stale: list[dict] = []
    if judge_terms:
        term_counter, waived, stale = apply_baseline(term_counter, baseline)

    if not quiet:
        print(f"材料根：{material}（{n_mat} 个文件，已排 `.git/`）· 扫描 {len(files)} 份契约")
        print(f"扫描单元：**blockquote 条目块**（连续 `>` 行成块）—— 故意比 L4e 的「行」宽")
        if family != "all":
            names = {"terms": "家族一 · 词", "sections": "家族二 · 章节号", "form": "家族三 · 归属形态"}
            unjudged = "、".join(v for k, v in names.items() if k != family)
            print(f"⚠️ 本次只判 `--family {family}`；**{unjudged} 只扫不判**"
                  f"（不参与退出码 —— 未判定 ≠ 干净）")
        print()
        if term_counter:
            head = "❌" if judge_terms else "⚠️[未判]"
            print(f"{head} 家族一 · 块内声称「材料」而词在材料里查无实据："
                  f"**{sum(len(v) for v in term_counter.values())} 处 / "
                  f"{len({f for v in term_counter.values() for f in v})} 份 / {len(term_counter)} 个词**")
            for term, fs in sorted(term_counter.items(), key=lambda kv: -len(kv[1])):
                print(f"    · 「{term}」 ×{len(fs)} 份：{'、'.join(sorted(set(fs))[:8])}"
                      f"{' …' if len(set(fs)) > 8 else ''}")
        if sect_counter:
            head = "❌" if judge_sects else "⚠️[未判]"
            print(f"\n{head} 家族二 · 声称「材料 §X / 第X章」："
                  f"**{sum(len(v) for v in sect_counter.values())} 处 / "
                  f"{len({f for v in sect_counter.values() for f in v})} 份**")
            for hit, fs in sorted(sect_counter.items(), key=lambda kv: -len(kv[1])):
                print(f"    · `{hit}` ×{len(fs)} 份")
        if form_counter or not_judged:
            n_form = sum(len(v) for v in form_counter.values())
            head = ("❌" if n_form else "✅") if judge_form else "⚠️[未判]"
            print(f"\n{head} 家族三 · 归属形态（一个值必须住在它自己那一栏里）：**{n_form} 份**")
            LBL = {"pointer": "(a) 栏里有指向 (b) 的指针（项该搬家，不该留指针）",
                   "negative": "(a) 栏里有负向材料归属语（抬头写着「材料已给出的数」，项却说自己不是）",
                   "misplaced": "附录挂错了子句（§0 里有该值，却不在含「业务侧默认」的那一栏里）",
                   "unfiled": "该值完全没备案（正文用了业务侧默认的值，§0 一个字都没提）"}
            for kind, label in LBL.items():
                hits = form_counter.get(kind, [])
                if hits:
                    names_ = "、".join(n for n, _ in hits[:8])
                    print(f"    · {label}：**{len(hits)} 份**")
                    print(f"        {names_}{' …' if len(hits) > 8 else ''}")
                    print(f"        例：{hits[0][1][:110]}")
            for kind, why in (("no-section0", "没有 §0「取值来源纪律」块"),
                              ("no-default-clause", "有 §0 但没有含「业务侧默认」的子句")):
                names_ = sorted(n for n, k in not_judged if k == kind)
                if names_:
                    print(f"    ○ **读数**（不判）：{len(names_)} 份{why} —— 本族**无从判定**；未判定 ≠ 干净")
                    print(f"        {'、'.join(names_[:8])}{' …' if len(names_) > 8 else ''}")

        if judge_terms and judge_sects and not term_counter and not sect_counter and not waived \
                and (not judge_form or not form_counter):
            print("✅ 无残留（本次判定范围内的每一族均为 0）")

    if not quiet and waived:
        print(f"\n🟡 可见豁免（`--baseline {baseline_path or '(未传)'}`，共 {len(waived)} 处）：")
        for w in waived:
            print(f"    · 「{w['term']}」@{w['file']}")
        print(f"  到期条件：{baseline.get('expires_when', '（未写 —— 必须补）') if baseline else '（未写）'}")
        print("  ⇒ 这几处是**逐处定标为假阳性**（`材料` 在这里是普通名词），不是待办；"
              "判据一旦收窄到它们不再报，**请删掉 baseline 里的对应项**")

    n_term = sum(len(v) for v in term_counter.values())
    n_sect = sum(len(v) for v in sect_counter.values())
    n_form = sum(len(v) for v in form_counter.values())
    total = ((n_term if judge_terms else 0) + (n_sect if judge_sects else 0)
             + (n_form if judge_form else 0))
    if stale:
        print(f"\n❌ baseline 里有 {len(stale)} 条豁免**已经用不上了**（判据不再报它）——"
              f"过期的豁免会腐烂成永久后门（台账 #5）：", file=sys.stderr)
        for it in stale:
            print(f"    · 「{it.get('term')}」@{it.get('file')} —— 请从 baseline 删掉这一条", file=sys.stderr)
        return 1
    if not quiet:
        scope = "本次判定范围内" if family != "all" else ""
        print(f"\n⇒ {scope}残留合计 **{total} 处**"
              f"（另有可见豁免 {len(waived)} 处，不计入）—— exit {1 if total else 0}")
        if not judge_terms and n_term:
            print(f"   （另有家族一 {n_term} 处**未被判定** —— 见台账 #67 的 W-67c）")
        if not judge_sects and n_sect:
            print(f"   （另有家族二 {n_sect} 处**未被判定**）")
        if not judge_form and n_form:
            print(f"   （另有家族三 {n_form} 份**未被判定**）")
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

# ── 家族三 · 归属形态的夹具（W-67d）─────────────────────────────────────────
S0_HEAD = "> **取值来源纪律（全文适用）**：来源只有三类 ——\n> "

# ① 改动前的形态：(a) 栏里留着指向 (b) 的指针
FORM_POINTER = S0_HEAD + """（a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；经营节奏「月度经营复盘」＝1 个自然月（材料只到月度这一档 ⇒ 季度档记（b））；阶段边界 STG-01…STG-08）；
> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）；
> （c）§3 算式的输出。
"""

# ② **(a) 与 (b) 写在同一行** —— 那个 `（b）` 是**子句标记**不是指针。
#    实测 139 份里有 4 份是这个写法（A-018 / B-017 / B-018 / B-054）；判成指针就是**假红**。
FORM_CLAUSE_OK = S0_HEAD + """（a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；其余 20% 渠道构成未明＝开放事实 O1）；（b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）；（c）§3 算式的输出。
"""

# ③ 改动后的形态：(a) 栏干净，(b) 栏收了这个值
FORM_CLEAN = S0_HEAD + """（a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；经营节奏「月度经营复盘」＝1 个自然月；阶段边界 STG-01…STG-08）；
> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）——**「季度经营策略」＝1 个自然季度属本档**；
> （材料只到月度这一档）；替换条件 ＝ 〈某台账〉读满 2 个完整年度后定档；
> （c）§3 算式的输出。
"""

# ④ (b) 栏没收，而正文用了标着业务侧默认的该值
FORM_UNFILED = S0_HEAD + """（a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；阶段边界 STG-01…STG-08）；
> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）；
> （c）§3 算式的输出。

- **周期到期**：每 2 个完整年度重算基线；每 1 个自然季度（**业务侧默认**：材料只写了月度）复核一次。
"""

# ⑤ **精度反向控制**：正文里的 `1 个完整自然季度` 是**替换条件的触发语**，不是那个取值
#    ⇒ 不许把它读成「用了该值而 (b) 没收」。判据一旦写成「正文含 1 个…自然季度」，这里就会假红。
FORM_TRIGGER_ONLY = S0_HEAD + """（a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；阶段边界 STG-01…STG-08）；
> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）；
> （c）§3 算式的输出。

- 取值：复核覆盖率 ＝ 100%（**业务侧默认**）；**替换条件** ＝ 〈合规条款库〉按市场覆盖满 1 个完整自然季度后，改用实测分布重算。
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

    def scan_one(body: str, material_files: dict, family: str = "all", baseline: dict | None = None):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mroot = root / "material"
            mroot.mkdir()
            for name, txt in material_files.items():
                (mroot / name).write_text(txt, encoding="utf-8")
            croot = root / "contracts"
            (croot / "A").mkdir(parents=True)
            (croot / "A" / "CTR-A-999-夹具.md").write_text(body, encoding="utf-8")
            code = run(croot, mroot, quiet=True, family=family, baseline=baseline)
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

    # ⑧–⑪ `--family` 开关的**隔离性**：两个家族的定标状态不同，判定必须真的分开。
    #     ⚠️ 只测「开关能选」是没劲的 —— 必须证明**同一个夹具在两个家族下判决相反**。
    cases.append(("⑧ `--family sections` 对「只有家族一残留」的夹具 ⇒ exit 0（不越界判）",
                  scan_one(RESIDUE_CONT, MAT, family="sections") == 0))
    cases.append(("⑨ 同一夹具 `--family terms` ⇒ exit 1（隔离是双向的，不是恒绿）",
                  scan_one(RESIDUE_CONT, MAT, family="terms") == 1))
    cases.append(("⑩ `--family terms` 对「只有家族二残留」的夹具 ⇒ exit 0（反向隔离）",
                  scan_one(RESIDUE_SECTION, MAT, family="terms") == 0))
    cases.append(("⑪ `--family sections` 对「只有家族二残留」的夹具 ⇒ exit 1"
                  "（L4k 就是靠这一条有劲）",
                  scan_one(RESIDUE_SECTION, MAT, family="sections") == 1))
    # ⑫ 材料**真有**该章节号时不算残留（防止 `--family sections` 变成恒红）
    MAT_SEC = {"m.md": "# 1 总则\n\n## 1.2 运行边界\n\n材料原文：动作策略必须绑定具体范围。\n"}
    FIX_HIT = """---
template_version: v2
---

- 依据＝材料 §1.2 的运行边界。
"""
    cases.append(("⑫ 材料真有 `§1.2` ⇒ exit 0（逐号核实，不是一律可疑）",
                  scan_one(FIX_HIT, MAT_SEC, family="sections") == 0))


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

    # ⑯–⑱ **同尺**（台账 #79）：两族判据的归一化必须真的相等，且要**端到端**证明它有后果。
    #     ⚠️ 只断言 `normalize(x) == citations.normalize(x)` 是**弱**断言（两边可能是同一个错的实现）；
    #     故 ⑰/⑱ 用夹具走完整条链：**强调符包裹的真引文不得报**、**强调符包裹的假引文必须报**。
    import check_material_citations as _c

    cases.append(("⑯ 两个脚本的 `normalize()` 必须**逐字节相等**（同尺是判据，不是注释）",
                  all(normalize(x) == _c.normalize(x) for x in
                      ("**复用效果和过期知识控制**", "a b\tc", "ｆｕｌｌ", "`code`", "A|B", "> quote"))))
    # ⚠️ 夹具措辞有讲究：`材料` 前 12 字内**不许出现去归属语**（`本契约`/`本项目`/`决策 Q`…），
    #    否则会走 DEATTRIB 分支、夹具根本到不了判据 —— 首版夹具写的就是「本契约的数取自材料…」，
    #    于是 ⑱ 落空。**夹具自己也要先证明它落在被测判据上**（本仓库记过的「用例是摆设」同族）。
    EMPH_OK = """> **取值来源纪律**：阈值来源只有三类 —— 材料「**月度经营复盘**」＝1 个自然月。
"""
    EMPH_BAD = """> **取值来源纪律**：阈值来源只有三类 —— 材料「**季度经营策略**」＝1 个自然季度。
"""
    cases.append(("⑰ 强调符包裹的**真引文** ⇒ exit 0（原先被 `**` 判成「查无实据」）",
                  scan_one(EMPH_OK, MAT, family="terms") == 0))
    cases.append(("⑱ 反向控制：强调符包裹的**假引文** ⇒ 必须照旧 exit 1",
                  scan_one(EMPH_BAD, MAT, family="terms") == 1))

    # ⑲/⑳ 去归属语**看词自己身边**（台账 #80）：`材料已给出的数（…）` 的括号跨度是语法容器，
    #      容器里每一项**自带归属**。一正一反：带 `＝决策 Q10` 的不报，**去掉那三个字后必须报**。
    LABELED = """> **取值来源纪律**：来源只有三类 ——
> （a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；独立站 holdback「没有、可以建、但需先建」＝决策 Q10）；
"""
    UNLABELED = """> **取值来源纪律**：来源只有三类 ——
> （a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；独立站 holdback「没有、可以建、但需先建」）；
"""
    cases.append(("⑲ 容器里**自带归属**的枚举项（`＝决策 Q10`）⇒ exit 0，不得算成材料声称",
                  scan_one(LABELED, MAT, family="terms") == 0))
    cases.append(("⑳ 反向控制：**去掉**那句归属语后 ⇒ 必须照旧 exit 1（豁免不是全放行）",
                  scan_one(UNLABELED, MAT, family="terms") == 1))

    # ㉑–㉓ `--baseline`：可见豁免的**双向**控制（台账 #82 同族：豁免条款必须比拦截条款测得更严）
    BL = {"expires_when": "判据收窄到能区分「材料作普通名词」时删除", "items": [
        {"term": "没有、可以建、但需先建", "file": "CTR-A-999-夹具.md"}]}
    cases.append(("㉑ baseline 命中的残留 ⇒ exit 0（豁免生效，且不参与退出码）",
                  scan_one(UNLABELED, MAT, family="terms", baseline=BL) == 0))
    cases.append(("㉒ 反向控制：baseline 只豁免它点名的**那一处**，别的残留照旧判红",
                  scan_one(UNLABELED + "\n> 材料「另一个不存在的词」＝1。\n", MAT,
                           family="terms", baseline=BL) == 1))
    cases.append(("㉓ 反向控制：baseline 里有**用不上**的豁免 ⇒ 判红（过期豁免＝永久后门，台账 #5）",
                  scan_one(LABELED, MAT, family="terms",
                           baseline={"items": [{"term": "没有、可以建、但需先建",
                                               "file": "CTR-A-999-夹具.md"}]}) == 1))
    cases.append(("㉔ baseline 为空/未传 ⇒ 与不传时**逐字同判**（豁免不许改变默认行为）",
                  scan_one(UNLABELED, MAT, family="terms", baseline=None) == 1))
    # ㉕–㉛ 家族三 · 归属形态：三条判据都要能打红，且**两个方向的反向控制**都不能少
    cases.append(("㉕ 家族三：(a) 栏留指针 ⇒ exit 1", scan_one(FORM_POINTER, MAT, family="form") == 1))
    cases.append(("㉖ 家族三：`（a）…）；（b）业务侧默认` 写在同一行 ⇒ exit 0"
                  "（那个 `（b）` 是**子句标记**不是指针；实测 4 份这么写）",
                  scan_one(FORM_CLAUSE_OK, MAT, family="form") == 0))
    cases.append(("㉗ 家族三：改后的形态（(a) 干净 + (b) 收了该值）⇒ exit 0（防恒红）",
                  scan_one(FORM_CLEAN, MAT, family="form") == 0))
    cases.append(("㉘ 家族三：(b) 栏没收而正文用了该值 ⇒ exit 1",
                  scan_one(FORM_UNFILED, MAT, family="form") == 1))
    cases.append(("㉙ 家族三**精度**反向控制：`1 个完整自然季度` 是替换条件的触发语 ⇒ 不许读成该值",
                  scan_one(FORM_TRIGGER_ONLY, MAT, family="form") == 0))
    cases.append(("㉚ 家族三的隔离：同一「只有家族三残留」的夹具在 `--family terms` 下 ⇒ exit 0",
                  scan_one(FORM_POINTER, MAT, family="terms") == 0))
    cases.append(("㉛ 隔离是双向的：`--family form` 对「只有家族一残留」的夹具 ⇒ exit 0",
                  scan_one(RESIDUE_CONT, MAT, family="form") == 0))

    # ⚠️ **非空过守卫**：家族三的夹具必须先被真的解析成 (a)/(b) 两栏。
    #    首版夹具第一行漏了 `> ` ⇒ 整段没进 blockquote、判据**一次都没跑**，
    #    于是「干净夹具 exit 0」这类用例**空过**成假绿。判据有没有跑，要和判据的结论分开证。
    for _name, _fx in (("POINTER", FORM_POINTER), ("CLAUSE_OK", FORM_CLAUSE_OK),
                       ("CLEAN", FORM_CLEAN), ("UNFILED", FORM_UNFILED),
                       ("TRIGGER_ONLY", FORM_TRIGGER_ONLY)):
        _a, _b, _s0, _has = section0_columns(_fx)
        cases.append((f"㉜ 非空过守卫：夹具 {_name} 真的解析出了 (a)/(b) 两栏（否则上面的用例是假绿）",
                      _has and _a is not None and _b != ""))

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
    ap.add_argument("--family", default="all", choices=FAMILIES,
                    help="只判哪个家族（默认 all；验收面 L4k 用 sections、L4m 用 terms）")
    ap.add_argument("--baseline", default="",
                    help="可见豁免清单（json）；**显式传了但文件不存在 ⇒ exit 2**，不是「没有豁免」")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    baseline = None
    if a.baseline:
        bp = Path(a.baseline)
        if not bp.is_file():
            print(f"❌ 显式传了 --baseline 但文件不存在：{bp} —— "
                  f"「没拿到豁免清单」不等于「没有豁免」，按 exit 2 处理", file=sys.stderr)
            return 2
        baseline = json.loads(bp.read_text(encoding="utf-8"))
    code = run(Path(a.contracts), Path(a.material), quiet=a.quiet, family=a.family,
               baseline=baseline, baseline_path=a.baseline)
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps({"exit": code, "family": a.family}, ensure_ascii=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
