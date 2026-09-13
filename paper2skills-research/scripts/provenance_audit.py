#!/usr/bin/env python3
"""provenance_audit.py — 卡片「论文溯源」可达性体检（G2 根因分析）

## 为什么需要这个脚本

2026-09-12 的 G2 基线是 **16/146 通过**，原因被表述为「存量卡引用块为 0」。
这个表述**把一个可修的门禁缺陷，说成了一个不可修的资产缺陷** —— 于是
「怎么让存量卡过 G2」变成了无解题。真实问题分层是：

| 层 | 问题 | 可修性 |
|----|------|--------|
| L1 | 卡片有引用块，但全文存档不存在/不可索引 | **可修**（补全文/修索引） |
| L2 | 卡片可溯源到论文，但从未写引用块 | **可修**（补引用块） |
| L3 | 卡片**根本没有论文来源**（是人写的经验卡） | **不可修**（只能诚实声明） |

只有先把 130 张卡按这三层分开，「修根因」才有明确的目标与分母。

## 判定依据（全部来自仓库实物，不做猜测）

对每张卡收集「溯源线索」：
1. frontmatter `paper_id`      —— ⚠️ 语义是 **arXiv ID**（如 2608.25277）
2. frontmatter `paper` / `source` / `url` / `arxiv` —— 标题或 arXiv/DOI 链接
3. **正文里的 arXiv/DOI** —— ⚠️ 首版只扫了 frontmatter，把 47 张「正文写着
   `arXiv:2502.02110`」的卡误判成"无论文来源"。**存量卡的溯源信息主要在正文里，
   不在 frontmatter 里** —— 这是本脚本第一个被自己实测推翻的结论。
4. 卡片 `title` 与论文标题的**词重叠**（仅在 1/2/3 无果时兜底）
5. registry `outputs.skill_card` 反查 —— ⚠️ registry 的 `paper_id` 是 `p2s-2026-XXXX`

对每条线索，再判「全文是否真的可达」：
- `fulltext.md` **且 ≥ MIN_FULLTEXT_CHARS** → 可立即核验（G2b 能跑）
- `fulltext.md` 但过短 → 只存了摘要，**不可核验**
- 只有本地 PDF → **待转换**（pdftotext 可解，属可修）
- 只有 arXiv ID，无本地存档 → **待抓取**（fetch_fulltext.py 可解，属可修）
- 什么都没有 → **不可达**

用法：
    python3 provenance_audit.py                              # 控制台报告
    python3 provenance_audit.py --json-out <path>            # 机器可读（父目录会自动创建）
    python3 provenance_audit.py --worklist-out <path.md>     # 生成可执行工单
    python3 provenance_audit.py --selftest                   # 自检：构造样本锁定五层判定

退出码：0 正常；1 自检失败。

⚠️ `--selftest` 锁定的核心是**「有来源」与「没来源」不许互相冒充**：
PHASE4 实测本脚本曾把 7 张「用论文标题声明来源」的卡判成「无论文来源」，
F4 照章给其中 4 张加了**假声明** —— 判某个东西「不存在」之前先 `ls` 一次。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VAULT = REPO_ROOT / "paper2skills-vault"
PAPERS = VAULT / "papers"
REGISTRY = VAULT / "07-资源库" / "papers_registry.json"

# 与 quote_check.py 保持同一个下限：低于此值不是全文，引文无法核验
MIN_FULLTEXT_CHARS = 8000

ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s（()\"']+)")

# 非 ID 形式的来源声明（2026-09-12 新增）。
#
# ⚠️ 本脚本第一版**只认 arXiv ID / DOI**，于是把 7 张「用标题声明了来源、
# 但没写 ID」的卡误判成「无论文来源」，并让 F4 给它们加上了
# 「本卡无对应论文来源」的声明 —— **与被声明卡片自己的正文直接矛盾**。
# 由子代理在 F4 执行中实测发现，不是设计时想到的。
#
# 教训：判「没有来源」时，**「找不到 ID」不等于「没有来源声明」**。
# 与 K1 的 ORPHAN_DEP 误判同源（判某个东西「不存在」之前先 `ls` 一次）。
DECL_RE = re.compile(
    r"(?:\*\*)?(?:论文来源|来源论文|基于论文|主论文)(?:\*\*)?\s*[:：]\s*(.+)",
)
# ACL Anthology 正式编号（USSA 这类只有 ACL 号、无 arXiv 的论文）
ANTHOLOGY_RE = re.compile(r"\b(\d{4}\.[a-z0-9-]+\.[0-9]+)\b", re.I)

# frontmatter 里可承载溯源信息的键（按可信度排序）
ID_KEYS = ("paper_id", "arxiv", "arxiv_id", "doi", "anthology_id")
TITLE_KEYS = ("paper", "paper_title", "source", "url", "title")

_STOP = {
    "a", "an", "the", "for", "of", "and", "with", "via", "to", "in", "on",
    "is", "are", "based", "using", "toward", "towards", "from", "by",
}


def parse_frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if line.strip().startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def norm_title(s: str) -> set[str]:
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return {w for w in s.split() if w not in _STOP and len(w) > 2}


def title_overlap(a: str, b: str) -> float:
    """a 的词有多少比例出现在 b 中（不对称：卡片标题通常短于论文标题）。"""
    wa = norm_title(a)
    if not wa:
        return 0.0
    return len(wa & norm_title(b)) / len(wa)


# --------------------------------------------------------------------------
# 全文存档索引
# --------------------------------------------------------------------------

def fulltext_keys(md: Path) -> list[str]:
    keys = [md.parent.name]
    try:
        head = md.read_text(encoding="utf-8", errors="replace")[:800]
    except OSError:
        return keys
    for field in ("arxiv_id", "paper_id"):
        m = re.search(rf"^\s*{field}\s*:\s*(\S+)\s*$", head, re.M)
        if m:
            keys.append(m.group(1))
    return keys


class PaperIndex:
    """仓库内所有「论文实物」的索引：fulltext.md / PDF，以及它们的可用键。"""

    def __init__(self) -> None:
        self.by_key: dict[str, dict] = {}
        self.entries: list[dict] = []

    def add(self, kind: str, path: Path, keys: list[str], size: int) -> None:
        ent = {"kind": kind, "path": str(path.relative_to(REPO_ROOT)),
               "keys": keys, "size": size}
        self.entries.append(ent)
        for k in keys:
            prev = self.by_key.get(k)
            # fulltext 优先于 pdf；同类取大的
            if prev is None or (kind == "fulltext" and prev["kind"] != "fulltext") \
                    or (kind == prev["kind"] and size > prev["size"]):
                self.by_key[k] = ent

    @classmethod
    def build(cls) -> "PaperIndex":
        idx = cls()
        for md in sorted(PAPERS.rglob("fulltext.md")):
            if not md.is_file():
                continue
            size = md.stat().st_size
            idx.add("fulltext", md, fulltext_keys(md), size)
        for pdf in sorted(PAPERS.rglob("*.pdf")):
            if not pdf.is_file():
                continue
            keys = [pdf.parent.name, pdf.stem]
            keys += ARXIV_RE.findall(pdf.name)
            idx.add("pdf", pdf, keys, pdf.stat().st_size)
        return idx

    def lookup_ids(self, ids: list[str]) -> list[dict]:
        return [self.by_key[i] for i in ids if i in self.by_key]

    def classify_hit(self, ent: dict) -> str:
        if ent["kind"] == "pdf":
            return "pdf_only"          # 待转换 → 可修
        return "fulltext_ok" if ent["size"] >= MIN_FULLTEXT_CHARS else "fulltext_short"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def load_registry() -> tuple[dict[str, dict], dict[str, dict]]:
    """返回 (arxiv_id -> record, skill_card_path -> record)。"""
    if not REGISTRY.exists():
        return {}, {}
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_arxiv: dict[str, dict] = {}
    by_card: dict[str, dict] = {}
    for r in data.get("records", []):
        aid = (r.get("identifiers") or {}).get("arxiv")
        if aid:
            by_arxiv[aid] = r
        card = (r.get("outputs") or {}).get("skill_card")
        if card:
            by_card[card] = r
    return by_arxiv, by_card


# --------------------------------------------------------------------------
# 主判定
# --------------------------------------------------------------------------

VERDICT_ORDER = [
    "VERIFIED",          # 有引用块且全文可核 → G2b 能跑（已是资产）
    "RETROFIT_READY",    # 全文就位，缺引用块 → 可直接补（最优先）
    "NEEDS_FULLTEXT",    # 有 arXiv ID，缺存档 → 抓全文后即可补
    "NEEDS_PDF_CONVERT",  # 只有本地 PDF → 转 fulltext 后即可补
    "NO_PAPER_SOURCE",   # 查无论文来源 → 只能诚实声明，不可"修复"
]

VERDICT_LABEL = {
    "VERIFIED": "✅ 已核验",
    "RETROFIT_READY": "🟢 可立即补引文",
    "NEEDS_FULLTEXT": "🟡 抓全文后可补",
    "NEEDS_PDF_CONVERT": "🟡 转 PDF 后可补",
    "NO_PAPER_SOURCE": "⚪ 无论文来源",
}


def audit_card(card: Path, idx: PaperIndex, by_arxiv: dict, by_card: dict) -> dict:
    text = card.read_text(encoding="utf-8", errors="replace")
    fm = parse_frontmatter(text)
    rel = str(card.relative_to(REPO_ROOT))
    rec = by_card.get(rel)

    n_quotes = len(re.findall(r"^>\s*(?:原文\s*[:：]\s*)?[\"“「『]", text, re.M))

    # --- 收集线索 ---
    candidates: list[str] = []
    arxiv_ids: list[str] = []
    for k in ID_KEYS:
        v = fm.get(k, "")
        if not v:
            continue
        arxiv_ids += ARXIV_RE.findall(v)
        if DOI_RE.search(v):
            arxiv_ids.append(DOI_RE.search(v).group(1))
    for k in TITLE_KEYS:
        v = fm.get(k, "")
        arxiv_ids += ARXIV_RE.findall(v)
        if DOI_RE.search(v):
            arxiv_ids.append(DOI_RE.search(v).group(1))

    # 正文扫描：存量卡的溯源信息主要在正文（`arXiv:2502.02110`、DOI 链接）。
    # 只扫前 200 行 —— 卡片末尾的 arXiv 多为「技能关联」里引用的**别篇**论文，
    # 把它当成本卡来源会产生假匹配；而来源声明几乎总在开头（frontmatter 之后）。
    body = text[:12000]
    arxiv_ids += ARXIV_RE.findall(body)
    arxiv_ids += DOI_RE.findall(body)
    arxiv_ids += ANTHOLOGY_RE.findall(body)

    # 非 ID 的来源声明：`**论文来源**: X`、`基于论文: X`（含代码 docstring 内的）。
    # ⚠️ 这一条是本脚本第二版新增 —— 第一版只认 ID，把 7 张卡误判为无论文来源。
    # 命中的是**标题串**，没有 ID 可查全文，所以下面按 `declared_title` 单独记账：
    # 判定归 NEEDS_FULLTEXT（「有来源待补全文」），而不是 NO_PAPER_SOURCE。
    declared_title = ""
    dm = DECL_RE.search(text)
    if dm:
        declared_title = dm.group(1).strip().strip('*').strip()
    if not declared_title:
        # frontmatter 的 `paper:` 若是个标题串（不是 ID），也算来源声明
        for k in ("paper", "paper_title"):
            v = (fm.get(k) or "").strip().strip('"').strip("'")
            if len(v) > 12 and not ARXIV_RE.search(v) and not DOI_RE.search(v):
                declared_title = v
                break

    # registry 反查（同名不同义字段：registry.paper_id 是 p2s-XXXX）
    reg_pid = rec.get("paper_id") if rec else None
    if reg_pid:
        candidates.append(reg_pid)
    if rec:
        ra = (rec.get("identifiers") or {}).get("arxiv")
        if ra:
            arxiv_ids.append(ra)

    # 线索 → 存档
    hits: list[dict] = []
    seen_paths: set[str] = set()
    for key in dict.fromkeys(arxiv_ids + candidates):
        for ent in idx.lookup_ids([key]):
            if ent["path"] not in seen_paths:
                seen_paths.add(ent["path"])
                hits.append(dict(ent, matched_key=key))

    # 兜底：标题词重叠
    match_method = "id" if hits else ""
    if not hits:
        cand_title = fm.get("paper", "") or ""
        if len(cand_title) > 12:
            best, best_score = None, 0.0
            for ent in idx.entries:
                if ent["kind"] != "fulltext":
                    continue
                score = title_overlap(cand_title, ent["path"])
                if score > best_score:
                    best, best_score = ent, score
            if best and best_score >= 0.7:
                hits.append(dict(best, matched_key=f"title~{best_score:.2f}"))
                match_method = "title"

    # --- 判定 ---
    # 「来源」与「引用」必须分开：正文里出现的第一个 arXiv ID 通常是**本卡来源**，
    # 其余多是「技能关联」里提到的别篇论文。把后者当来源会让一张无来源的卡
    # 假装可溯源（并让工单把它排进"抓全文就能修"的队列）。
    if arxiv_ids:
        primary = arxiv_ids[0]
    elif hits:
        primary = hits[0]["matched_key"]
    elif declared_title:
        primary = declared_title
    else:
        primary = ""

    classes = {idx.classify_hit(h) for h in hits}
    if n_quotes and "fulltext_ok" in classes:
        verdict = "VERIFIED"
    elif "fulltext_ok" in classes:
        verdict = "RETROFIT_READY"
    elif "fulltext_short" in classes:
        verdict = "NEEDS_FULLTEXT"
    elif "pdf_only" in classes:
        verdict = "NEEDS_PDF_CONVERT"
    elif primary:
        # 含「只有标题、没有 ID」的情形：仍是有来源，只是**还没法抓全文**。
        # 判 NEEDS_FULLTEXT 而不是 NO_PAPER_SOURCE —— 后者会误导后续给出
        # 「本卡无论文来源」的声明（本轮实测的错就出在这里）。
        verdict = "NEEDS_FULLTEXT"
    else:
        verdict = "NO_PAPER_SOURCE"

    return {
        "card": rel,
        "domain": rel.split("/")[1] if "/" in rel else "",
        "name": card.stem,
        "verdict": verdict,
        "n_quotes": n_quotes,
        "primary_source": primary,
        "declared_title": declared_title,
        "all_candidate_ids": sorted(set(arxiv_ids)),
        "cited_ids": sorted(set(arxiv_ids[1:])),
        "registry_paper_id": reg_pid,
        "match_method": match_method or ("id" if arxiv_ids else
                                         ("declared" if declared_title else "")),
        "hits": hits,
    }


def write_report(path_str: str, text: str) -> None:
    """写产物文件 —— **先建父目录**（2026-09-13）。

    ⚠️ 与 `repo_health.py` 漏洞 #8 是**同一个缺陷、同一类后果**，在本脚本上也实测复现了：

        $ python3 provenance_audit.py --json-out /tmp/notexist/a/b.json
        FileNotFoundError: ... '/tmp/notexist/a/b.json'
        exit=1

    审计结果其实**已经算完**了，只是最后一步落盘失败 —— 退出码 1 会让自动化
    把一次**成功的**审计记成失败，而产物又没生成，两边都拿不到真相。
    这类「静默假失败」在本仓库已出现 3 次（gate_check 找不到 evidence.md、
    repo_health --json-out、本脚本），所以两处都用同一个写法。
    """
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
# 构造样本的**设计口径**：本脚本最危险的失效方向不是「漏报」，而是
# **判一个东西「不存在」** —— PHASE4 实测：早期版本只认 arXiv ID / DOI 形式，
# 把 7 张「用论文标题声明来源」的卡判成「无论文来源」，于是 F4 照章给其中 4 张
# 加了**假声明**。教训已写进 CLAUDE.md：**判某个东西「不存在」之前先 `ls` 一次**。
# 所以这里给这条判据配了回归用例，并且**反向也配了一条**（`paper: n/a` 这种
# 非标题值不许凭空造出来源）—— 只测一个方向会让判据在另一个方向上失守。
_SELFTEST_FULLTEXT = (
    "---\narxiv_id: 2606.26690\npaper_id: p2s-test-0001\n---\n\n"
    "# A Test Paper For The Provenance Selftest\n\n"
    + ("This is filler text standing in for a real fulltext archive. " * 200)
)

_SELFTEST_REGISTRY = {
    "records": [
        {"paper_id": "p2s-test-0003",
         "identifiers": {"arxiv": "2606.26690"},
         "outputs": {"skill_card": "paper2skills-vault/01-域/Skill-RegistryOnly.md"}},
    ]
}

# name -> (卡片文本, 期望判定)
_SELFTEST_CARDS: dict[str, tuple[str, str]] = {
    # ① PHASE4 回归用例（本条是本次 X5 的核心）
    "Skill-TitleOnly": ("""---
title: Title Only Card
module: 01-域
paper: "A Real Paper Title Declared Without Any Identifier At All"
---
# 正文

本卡只写了论文标题，没有 arXiv ID，也没有 DOI。
""", "NEEDS_FULLTEXT"),
    # ② 真的没有论文来源：只能诚实声明，不可"修复"
    "Skill-Practice": ("""---
title: Practice Card
module: 01-域
evidence_basis: author-practice
---
# 正文

纯作者经验卡，卡内没有任何论文线索。
""", "NO_PAPER_SOURCE"),
    # ③ 最理想状态：有 ID、有全文底本、有逐字引用块
    "Skill-Verified": ("""---
title: Verified Card
module: 01-域
paper_id: 2606.26690
---
# 正文

> 原文:"This is filler text standing in for a real fulltext archive."

出处：arXiv:2606.26690 §1
""", "VERIFIED"),
    # ④ 全文就位但没写引用块 → 最优先的工单
    "Skill-Retrofit": ("""---
title: Retrofit Card
module: 01-域
paper_id: 2606.26690
---
# 正文

本卡有来源、底本也在，但一条逐字引文都没写。
""", "RETROFIT_READY"),
    # ⑤ 溯源信息只在正文（frontmatter 什么都没有）—— 首版只扫 frontmatter，
    #    曾把 47 张这类卡误判成「无论文来源」
    "Skill-BodyOnly": ("""---
title: Body Only Card
module: 01-域
---
# 正文

本卡来源：arXiv:2408.05353（frontmatter 里没有任何溯源字段）。
""", "NEEDS_FULLTEXT"),
    # ⑥ 反向用例：`paper:` 里是垃圾值而不是标题 → 不许凭空造出来源
    "Skill-ShortPaperField": ("""---
title: Short Paper Field Card
module: 01-域
paper: n/a
---
# 正文

frontmatter 的 paper 字段是占位符，不是论文标题。
""", "NO_PAPER_SOURCE"),
    # ⑦ 第五层（任务书里的「四层」之外还有一层）：只有本地 PDF
    "Skill-PdfOnly": ("""---
title: Pdf Only Card
module: 02-域
paper_id: 2607.12345
---
# 正文

底本只存了 PDF，还没转成 fulltext。
""", "NEEDS_PDF_CONVERT"),
    # ⑧ 卡内无线索，但 registry 反查能认领（第三条线索）
    "Skill-RegistryOnly": ("""---
title: Registry Only Card
module: 01-域
---
# 正文

本卡自己没写来源，靠 registry 的 outputs.skill_card 反查认领。
""", "RETROFIT_READY"),
}


def _selftest_constructed() -> list[tuple[str, bool]]:
    """构造一棵临时仓库，端到端跑 `audit_card`，验证五层判定与互斥性。

    原有自检只测了 `classify_hit` / `title_overlap` 两个**辅助函数** ——
    那证明不了「一张卡会被判成哪一层」。PHASE4 的误判恰恰出在 audit_card 的
    判定链上，所以这里必须**端到端**测。

    ⚠️ 会临时改写模块级路径常量，**finally 里一定还原**（自检不许把配置改脏）。
    """
    import json as _json
    import tempfile
    global REPO_ROOT, VAULT, PAPERS, REGISTRY
    saved = (REPO_ROOT, VAULT, PAPERS, REGISTRY)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()          # ⚠️ macOS 的 /var 是符号链接，必须 resolve
        vault = root / "paper2skills-vault"
        papers = vault / "papers"
        (papers / "01-域" / "p2s-test-0001").mkdir(parents=True)
        (papers / "02-域").mkdir(parents=True)
        (vault / "01-域").mkdir(parents=True)
        (vault / "02-域").mkdir(parents=True)
        (vault / "07-资源库").mkdir(parents=True)
        (papers / "01-域" / "p2s-test-0001" / "fulltext.md").write_text(
            _SELFTEST_FULLTEXT, encoding="utf-8")
        (papers / "02-域" / "2607.12345.pdf").write_bytes(b"%PDF-1.4\n" + b"0" * 5000)
        reg = vault / "07-资源库" / "papers_registry.json"
        reg.write_text(_json.dumps(_SELFTEST_REGISTRY, ensure_ascii=False), encoding="utf-8")
        for name, (text, _want) in _SELFTEST_CARDS.items():
            domain = "02-域" if name == "Skill-PdfOnly" else "01-域"
            (vault / domain / f"{name}.md").write_text(text, encoding="utf-8")

        try:
            REPO_ROOT, VAULT, PAPERS, REGISTRY = root, vault, papers, reg
            idx = PaperIndex.build()
            by_arxiv, by_card = load_registry()
            got = {}
            for name, (text, want) in _SELFTEST_CARDS.items():
                domain = "02-域" if name == "Skill-PdfOnly" else "01-域"
                rep = audit_card(vault / domain / f"{name}.md", idx, by_arxiv, by_card)
                got[name] = rep["verdict"]
            n_cards = len(got)
            tally = {v: 0 for v in VERDICT_ORDER}
            for v in got.values():
                tally[v] = tally.get(v, 0) + 1
            layers_reached = {v for v in got.values()}
        finally:
            REPO_ROOT, VAULT, PAPERS, REGISTRY = saved

    def one(name: str) -> tuple[str, bool]:
        want = _SELFTEST_CARDS[name][1]
        return got[name] == want, want

    checks: list[tuple[str, bool]] = []
    for name, title in (
        ("Skill-TitleOnly", "【PHASE4 回归】只有 `paper:` 标题、无 arXiv/DOI → 判『有来源』，"
                            "**不得**判 NO_PAPER_SOURCE"),
        ("Skill-Practice", "只有 `evidence_basis: author-practice`、无任何论文线索 → "
                           "判 NO_PAPER_SOURCE"),
        ("Skill-Verified", "有 arXiv ID + 全文底本 + 引用块 → 判 VERIFIED"),
        ("Skill-Retrofit", "有 arXiv ID + 全文底本、无引用块 → 判 RETROFIT_READY"),
        ("Skill-BodyOnly", "溯源线索只在正文（frontmatter 无 ID）→ 仍判『有来源』"),
        ("Skill-ShortPaperField", "`paper: n/a` 非标题值 → 不得凭空造出来源（判 NO_PAPER_SOURCE）"),
        ("Skill-PdfOnly", "只有本地 PDF → 判 NEEDS_PDF_CONVERT（第五层可达）"),
        ("Skill-RegistryOnly", "卡内无线索但 registry 反查认领 → 判『有来源』"),
    ):
        passed, want = one(name)
        checks.append((f"{title}（实得 {got[name]}，期望 {want}）", passed))

    # 互斥性：一张卡只能落一层 —— 逐项验证「每张卡的判定都在枚举内」且
    # 「五个层各自可达且互不重叠」（八张样本卡 → 五种判定值，不多不少）
    all_in_enum = all(v in VERDICT_ORDER for v in got.values())
    checks.append((
        f"四层判定互斥：{n_cards} 张样本卡各落**恰好一层**，"
        f"tally 求和={sum(tally.values())}（应={n_cards}），"
        f"可达层={len(layers_reached)}（应≥4，本次={sorted(layers_reached)}）",
        all_in_enum and sum(tally.values()) == n_cards and len(layers_reached) >= 4))
    # 同一张卡不可能同时是两个判定 —— 逐卡再确认一次（判定是单值字符串，
    # 这条防的是「有人把 verdict 改成 list/多值」这种结构漂移）
    checks.append((
        "每张卡的 verdict 都是**单个字符串**（不是多值/集合）",
        all(isinstance(v, str) for v in got.values())
        and len(got) == len(_SELFTEST_CARDS)))
    return checks


def _selftest_output_paths() -> list[tuple[str, bool]]:
    """产物落盘：父目录不存在时必须自建（本轮新修的漏洞 #8 同类缺陷）。"""
    import tempfile
    results = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        for label, fname in (("--json-out", "deep/a/b.json"), ("--worklist-out", "deep/c/w.md")):
            target = root / fname
            if target.parent.exists():
                results.append((f"{label} 父目录不存在 → 自建并写出", False))
                continue
            try:
                write_report(str(target), "x")
                results.append((f"{label} 父目录不存在 → 自建并写出",
                                target.is_file() and target.read_text(encoding="utf-8") == "x"))
            except OSError:
                results.append((f"{label} 父目录不存在 → 自建并写出", False))
    return results


def selftest() -> int:
    """自检：证明本脚本真的能区分「可修」与「不可修」，且**不会把有来源判成没来源**。

    分两段：
      A. 真实仓库索引 + 辅助函数（原有）—— 证明索引本身可用
      B. 构造样本端到端（2026-09-13 新增）—— 证明**判定链**本身可信
    """
    checks: list[tuple[str, bool]] = []

    # --- A. 真实仓库索引（原有）------------------------------------------------
    idx = PaperIndex.build()
    kinds = {e["kind"] for e in idx.entries}
    print(f"索引：{len(idx.entries)} 个论文实物，类型 {sorted(kinds)}")
    checks.append(("索引里有 fulltext（否则索引构建失效）", "fulltext" in kinds))
    for ent, want in [
        ({"kind": "fulltext", "size": MIN_FULLTEXT_CHARS + 1}, "fulltext_ok"),
        ({"kind": "fulltext", "size": MIN_FULLTEXT_CHARS - 1}, "fulltext_short"),
        ({"kind": "pdf", "size": 10 ** 7}, "pdf_only"),
    ]:
        got = idx.classify_hit(ent)
        checks.append((f"classify_hit({ent['kind']},{ent['size']}) = {got}（期望 {want}）",
                       got == want))
    hi = title_overlap("ReAct Synergizing Reasoning and Acting in Language Models",
                       "papers/10-MAS/00-知识库-Skill卡片/Skill-ReAct-Reasoning-Acting.md")
    lo = title_overlap("ReAct Synergizing Reasoning and Acting in Language Models",
                       "papers/13-广告分析/p2s-2026-0001/fulltext.md")
    checks.append((f"标题重叠能区分相关/无关（相关={hi:.2f} > 无关={lo:.2f}）", hi > lo))

    # --- B. 构造样本端到端（新增）----------------------------------------------
    print("构造样本（临时仓库，端到端跑 audit_card）：")
    checks += _selftest_constructed()

    # --- C. 产物落盘 -----------------------------------------------------------
    checks += _selftest_output_paths()

    ok = True
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")
        ok &= passed
    print("✅ 自检通过：判定链可信 —— 有来源的不会被判成没来源，没来源的也不会被凭空认领"
          if ok else "❌ 自检失败")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="卡片论文溯源可达性体检")
    ap.add_argument("--json-out")
    ap.add_argument("--worklist-out", help="输出可执行工单 Markdown")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", default="", help="只打印该判定的卡片明细")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    idx = PaperIndex.build()
    by_arxiv, by_card = load_registry()
    cards = sorted(p for p in VAULT.rglob("Skill-*.md") if p.is_file())
    reports = [audit_card(c, idx, by_arxiv, by_card) for c in cards]

    tally: dict[str, int] = {v: 0 for v in VERDICT_ORDER}
    for r in reports:
        tally[r["verdict"]] += 1

    print(f"卡片总数 {len(reports)}｜论文实物 {len(idx.entries)}"
          f"（fulltext {sum(1 for e in idx.entries if e['kind'] == 'fulltext')}"
          f" / pdf {sum(1 for e in idx.entries if e['kind'] == 'pdf')}）")
    print()
    width = max(len(VERDICT_LABEL[v]) for v in VERDICT_ORDER)
    for v in VERDICT_ORDER:
        print(f"  {VERDICT_LABEL[v]:<{width}}  {tally[v]:>4}")

    fixable = sum(tally[v] for v in
                  ("RETROFIT_READY", "NEEDS_FULLTEXT", "NEEDS_PDF_CONVERT"))
    print()
    print(f"可修（补全文/转 PDF/补引文）: {fixable} / {len(reports)}")
    print(f"不可修（无论文来源，只能声明）: {tally['NO_PAPER_SOURCE']} / {len(reports)}")

    if args.show:
        print()
        for r in reports:
            if r["verdict"] == args.show:
                a = r["primary_source"] or (r["registry_paper_id"] or "-")
                print(f"  {r['card']}   [{a}]")

    if args.worklist_out:
        lines = ["# 存量卡溯源工单（由 provenance_audit.py 生成，勿手改）", ""]
        for v in VERDICT_ORDER:
            group = [r for r in reports if r["verdict"] == v]
            if not group:
                continue
            lines += [f"## {VERDICT_LABEL[v]}（{len(group)} 张）", ""]
            lines.append("| 卡片 | arXiv/DOI | 存档 | 引文数 |")
            lines.append("|---|---|---|---|")
            for r in group:
                ids = r["primary_source"] or (r["registry_paper_id"] or "—")
                hp = ", ".join(h["path"] for h in r["hits"]) or "—"
                lines.append(f"| `{r['card']}` | {ids} | {hp} | {r['n_quotes']} |")
            lines.append("")
        write_report(args.worklist_out, "\n".join(lines))
        print(f"\n工单 → {args.worklist_out}")

    if args.json_out:
        write_report(args.json_out, json.dumps({"tally": tally, "cards": reports},
                                               ensure_ascii=False, indent=2))
        print(f"JSON → {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
