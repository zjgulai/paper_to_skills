#!/usr/bin/env python3
"""quote_check.py — 引用块逐字核验（G2b）

## 为什么需要这个脚本

`gate_check.py` 的 G2 只回答了「**有没有**出处」，没有回答「出处**对不对**」。
它把卡片里所有引用块/evidence.md 中出现的数字汇聚成一个集合，再看正文数字是否
落在集合里 —— 这留下了一个致命后门：

    卡片正文写「ROI 提升 40%」
    引用块写   > 原文："我们观察到 ROI 提升 40%"
    但论文里根本没有这句话
    → G2 判 GREEN

也就是说，只要**伪造一段带数字的引文**，就能让任意数字过闸。这正是
`Cited but Not Verified`(arXiv:2605.06635) 实测「链接可用 >94%、事实一致性仅 39–77%」
所指的失效模式。

## 本脚本做什么

把每一条 `> 原文："..."` 摘录**逐字**回查论文全文存档
（`paper2skills-vault/papers/<domain>/<paper_id>/fulltext.md`）。

判定分三级：
- `VERBATIM`  —— 规范化后能在全文中找到完整子串（引文真实）
- `FUZZY`     —— 完整子串找不到，但 5-gram 覆盖率 ≥ 阈值（HTML→MD 转换、
                 连字符断行、上下标导致的轻微差异；**需人工复核**）
- `FABRICATED`—— 覆盖率低于阈值（**引文不存在于论文中**）

⚠️ 规范化**故意做得宽松**（大小写/空白/各类连字符与引号/常见 LaTeX 残留），
因为我们的目标是抓「编造的引文」，不是抓「空格不一致」——
过严会产生大量假阳性，而假阳性会让门禁被绕过（K1 迭代已证明这一点）。

用法：
    python3 quote_check.py --card <card.md>              # 单卡
    python3 quote_check.py --all                         # 全库（只报有引用块的卡）
    python3 quote_check.py --card <c.md> --json          # 机器可读
    python3 quote_check.py --selftest                    # 自检：证明它真的能抓到伪造引文

退出码：0 = 无 FABRICATED；1 = 存在 FABRICATED 或无法核验。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT = REPO_ROOT / "paper2skills-vault"
PAPERS_DIR = VAULT / "papers"

# 引用块：**同时**支持 v2 规范写法与旧库裸引用写法。
#
# ⚠️ 这里踩过一个坑：gate_check.py 原本用 `^\s*>\s*["“](.+?)["”]\s*$`，
# 只认「> 紧跟引号」；而 MasterPrompt-v2 要求写 `> 原文："..."`，
# 前缀是「原文：」→ 引号不紧跟 `>` → **一条都匹配不到**。
# 后果是带合格引文的卡片反而被判「无任何可追溯出处」（假红灯），
# 而假红灯会让人不再相信门禁。两个脚本必须共用同一个抽取器。
QUOTE_RE = re.compile(
    r"^>\s*(?:原文\s*[:：]\s*)?[\"“「『](.+?)[\"”」』]\s*$",
    re.M | re.S,
)
# 出处行：> 出处：2606.26690 §4.2（PDF 第 5 页）
SOURCE_RE = re.compile(r"^>\s*出处\s*[:：]\s*(.+)$", re.M)

# 判定阈值 —— 针对「最长**连续**匹配段占引文的比例」，而非全文档 n-gram 覆盖率。
#
# 为什么不用全文档 n-gram 覆盖率：实测它会被**拼接攻击**骗过。
# 把摘要句与引言句缝成一句（论文里不存在这句话），n-gram 召回仍是 1.0，
# 因为每个碎片都能在文档某处找到。而最长连续匹配段只有碎片那么长，
# 一眼就能看出是拼的。这个区分是实测出来的，不是设计偏好。
VERBATIM_RATIO = 0.995   # 几乎整句连续命中（仅容忍尾部标点差异）
FUZZY_RATIO = 0.80       # 有局部差异 → 需人工复核
# 拼接特征：n-gram 召回很高、但最长连续段很短
SPLICE_RECALL = 0.95
SPLICE_RUN = 0.75

NGRAM = 5

# 全文字符数下限：低于此值说明存档不是全文（可能只是摘要），
# 此时任何引文都「无法核验」，必须报 YELLOW 而不是 FABRICATED
MIN_FULLTEXT_CHARS = 8000

_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-",
    "\u2011": "-", "\u2012": "-", "\u00ad": "",   # soft hyphen
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    "\uff0c": ",", "\uff1a": ":", "\uff08": "(", "\uff09": ")",
    "\uff05": "%", "\u3002": ".", "\uff1b": ";",
}


def normalize(s: str) -> str:
    """规范化到「可比对」形式。只抹平排版差异，不抹平措辞差异。"""
    s = unicodedata.normalize("NFKC", s)
    for k, v in _PUNCT_MAP.items():
        s = s.replace(k, v)
    # LaTeX 残留：$...$、\cite{}、\textbf{} 等
    s = re.sub(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?", " ", s)
    s = s.replace("$", " ")
    # markdown 强调与反引号
    s = re.sub(r"[*_`~]+", "", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ngrams(s: str, n: int = NGRAM) -> set[str]:
    """按**字符**切 n-gram。

    不用词级 n-gram：中文没有空格分词，且论文里的复合术语（`counterfactual
    lift`）会被 tokenizer 切碎，字符级对中英混排最稳。
    """
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def ngram_recall(quote_norm: str, full_norm: str) -> float:
    """全文档 n-gram 召回（诊断用，**不用于判定**）。

    高召回 + 低连续度 = 拼接引文。单独看它会被拼接骗过。
    """
    q = ngrams(quote_norm)
    if not q:
        return 0.0
    f = ngrams(full_norm)
    return len(q & f) / len(q) if f else 0.0


def longest_match_run(quote_norm: str, full_norm: str) -> int:
    """引文中能在全文中找到的**最长连续子串**长度（精确，非近似）。

    用二分 + `in`（C 层子串搜索）：若存在长度 L 的匹配段，则长度 L-1 的
    必然也存在（取子段即可），所以对 L 单调，可以二分。
    每次判定是一次 C 层 substring 搜索，比 80k×200 的 DP 快几个数量级。
    """
    lo, hi = 0, len(quote_norm)
    if hi == 0:
        return 0
    if quote_norm in full_norm:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        found = False
        for i in range(len(quote_norm) - mid + 1):
            if quote_norm[i:i + mid] in full_norm:
                found = True
                break
        if found:
            lo = mid
        else:
            hi = mid - 1
    return lo


def check_one(quote: str, full_norm: str) -> tuple[str, float, float, bool]:
    """返回 (verdict, longest_run_ratio, ngram_recall, spliced_flag)。"""
    q = normalize(quote)
    if not q:
        return "FABRICATED", 0.0, 0.0, False

    run = longest_match_run(q, full_norm)
    ratio = run / len(q)
    recall = ngram_recall(q, full_norm)
    spliced = recall >= SPLICE_RECALL and ratio < SPLICE_RUN

    if ratio >= VERBATIM_RATIO:
        verdict = "VERBATIM"
    elif ratio >= FUZZY_RATIO:
        verdict = "FUZZY"
    else:
        verdict = "FABRICATED"
    return verdict, round(ratio, 3), round(recall, 3), spliced


def _index_keys(md: Path) -> list[str]:
    """一个全文存档的可用键。

    ⚠️ 目录名用的是 registry 的 `paper_id`（如 `p2s-2026-0001`），而卡片
    frontmatter 的 `paper_id` 是 arXiv ID（如 `2606.26690`）—— 两者不同名。
    所以必须同时索引：目录名 + 存档头部注释里的 arxiv_id / paper_id。
    漏掉这一步会让所有引文都判成「无法核验」，门禁形同虚设。
    """
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


def find_fulltext(paper_id: str, index: dict[str, Path] | None = None) -> Path | None:
    """按 paper_id（arXiv ID / DOI / registry id）找 fulltext.md。"""
    if not paper_id:
        return None
    pid = paper_id.strip()
    if index:
        hit = index.get(pid)
        if hit:
            return hit
    hits = [p for p in PAPERS_DIR.glob("*/fulltext.md") if p.is_file()]
    hits += [p for p in PAPERS_DIR.glob("*/*/fulltext.md") if p.is_file()]
    cands = [p for p in hits if pid in _index_keys(p)]
    if cands:
        # 命中多个时取内容最长的（最可能是完整存档）
        return max(cands, key=lambda p: p.stat().st_size)
    return None


def load_fulltext_index() -> dict[str, Path]:
    """paper_id / arxiv_id / 目录名 → fulltext.md 的索引（避免每张卡都 glob）。"""
    idx: dict[str, Path] = {}
    if not PAPERS_DIR.is_dir():
        return idx
    for p in PAPERS_DIR.glob("*/*/fulltext.md"):
        for k in _index_keys(p):
            prev = idx.get(k)
            if prev is None or p.stat().st_size > prev.stat().st_size:
                idx[k] = p
    return idx


def parse_frontmatter(text: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def extract_quotes(text: str) -> list[dict]:
    """抽出所有引用块及其紧跟的出处行。

    证据链也可能放在同目录 evidence.md 里，调用方负责合并。
    """
    quotes: list[dict] = []
    lines = text.split("\n")
    for m in QUOTE_RE.finditer(text):
        raw = m.group(1).strip()
        # 找出处：引用块之后 3 行内第一个 `> 出处：`
        tail = text[m.end():m.end() + 400]
        sm = SOURCE_RE.search(tail)
        quotes.append({
            "quote": raw,
            "source": sm.group(1).strip() if sm else "",
            "line": text[:m.start()].count("\n") + 1,
        })
    return quotes


def check_card(card: Path, index: dict[str, Path] | None = None) -> dict:
    text = card.read_text(encoding="utf-8", errors="replace")
    fm = parse_frontmatter(text)

    # 证据可以写在卡片里，也可以写在同目录 evidence.md 里
    blob = text
    ev = card.parent / "evidence.md"
    if ev.is_file():
        blob += "\n" + ev.read_text(encoding="utf-8", errors="replace")

    quotes = extract_quotes(blob)
    if not quotes:
        return {"card": str(card), "quotes": [], "verdict": "NO_QUOTES",
                "note": "卡片与 evidence.md 中都没有 `> 原文:\"...\"` 引用块"}

    # 决定核验底本：frontmatter.paper_id 优先，其次出处行里的 arXiv ID/DOI
    paper_id = fm.get("paper_id", "")
    ft = (index or {}).get(paper_id) or find_fulltext(paper_id)
    if ft is None:
        for q in quotes:
            m = re.search(r"(\d{4}\.\d{4,5}|10\.\d{4,9}/[^\s（(]+)", q["source"])
            if m:
                ft = (index or {}).get(m.group(1)) or find_fulltext(m.group(1))
                if ft:
                    paper_id = m.group(1)
                    break

    if ft is None:
        return {
            "card": str(card), "paper_id": paper_id, "verdict": "NO_FULLTEXT",
            "quotes": [dict(q, verdict="UNVERIFIABLE", longest_run_ratio=None,
                                    ngram_recall=None, spliced=None) for q in quotes],
            "note": f"未找到 {paper_id or '论文'} 的全文存档，无法核验引文",
        }

    full = ft.read_text(encoding="utf-8", errors="replace")
    full_norm = normalize(full)
    if len(full) < MIN_FULLTEXT_CHARS:
        return {
            "card": str(card), "paper_id": paper_id, "verdict": "NO_FULLTEXT",
            "fulltext": str(ft), "fulltext_chars": len(full),
            "quotes": [dict(q, verdict="UNVERIFIABLE", longest_run_ratio=None,
                                    ngram_recall=None, spliced=None) for q in quotes],
            "note": f"存档仅 {len(full)} 字符（疑似摘要而非全文），不足以核验引文",
        }

    results = []
    for q in quotes:
        verdict, ratio, recall, spliced = check_one(q["quote"], full_norm)
        results.append(dict(q, verdict=verdict, longest_run_ratio=ratio,
                            ngram_recall=recall, spliced=spliced))

    n_fab = sum(1 for r in results if r["verdict"] == "FABRICATED")
    n_fuzzy = sum(1 for r in results if r["verdict"] == "FUZZY")
    n_spliced = sum(1 for r in results if r.get("spliced"))
    if n_fab:
        verdict = "FABRICATED"
    elif n_fuzzy:
        verdict = "FUZZY"
    else:
        verdict = "VERBATIM"

    return {
        "card": str(card), "paper_id": paper_id, "fulltext": str(ft),
        "fulltext_chars": len(full), "verdict": verdict,
        "n_quotes": len(results), "n_verbatim": len(results) - n_fab - n_fuzzy,
        "n_fuzzy": n_fuzzy, "n_fabricated": n_fab, "n_spliced": n_spliced,
        "quotes": results,
    }


def collect_cards() -> list[Path]:
    cards = [p for p in VAULT.rglob("Skill-*.md") if p.is_file()]
    return sorted(p for p in cards if "_superseded" not in p.parts)


def selftest() -> int:
    """证明本脚本真的能抓到伪造引文 —— 门禁必须先自证可信。

    四个用例，其中「拼接」这一条是开发过程中**实测发现的漏洞**：
    最初用「全文档 n-gram 覆盖率」判定，把摘要句与引言句缝成一句时
    覆盖率仍是 1.0（每个碎片都能在文档某处找到）→ 拼接引文被放行。
    改为「最长连续匹配段占比」后该用例被正确拦截，故固化为回归测试。
    """
    ft = find_fulltext("2606.26690", load_fulltext_index())
    if ft is None:
        print("❌ 自检失败：找不到 2606.26690 的全文存档")
        return 1
    full_norm = normalize(ft.read_text(encoding="utf-8", errors="replace"))

    # 用例 1｜真引文：论文原句连续出现
    real = ("We refer to this gap between credited conversions and causal "
            "incremental conversions as the attribution–cannibalization mismatch")
    v_real, r_real, _, _ = check_one(real, full_norm)

    # 用例 2｜纯伪造：术语与句式都像论文，但原句不存在，还带一个诱人的数字
    fake = ("Our framework improves incremental ROAS by 42.7% across all "
            "deployed markets while reducing measurement cost by one third")
    v_fake, r_fake, _, _ = check_one(fake, full_norm)

    # 用例 3｜拼接：摘要句 + 引言句缝合。论文里不存在这句话，
    #        但每个碎片都是真的 —— 这是最难抓的一种，必须拦下。
    splice = ("We propose an experiment-calibrated attribution correction framework "
              "that combines the causal credibility of incrementality experiments")
    v_splice, r_splice, recall_splice, flag_splice = check_one(splice, full_norm)

    # 用例 4｜真引文但含公式/换行等排版差异
    para = ("Production attribution is timely, granular, and continuously available, "
            "but observational by construction.")
    v_para, r_para, _, _ = check_one(para, full_norm)

    ok = (
        v_real == "VERBATIM"
        and v_fake == "FABRICATED"
        and v_splice != "VERBATIM" and flag_splice        # 拼接必须被标出
        and v_para == "VERBATIM"
    )

    print(f"1 真引文     : {v_real:10s} 连续度={r_real}")
    print(f"2 纯伪造     : {v_fake:10s} 连续度={r_fake}   ← 必须 FABRICATED")
    print(f"3 拼接引文   : {v_splice:10s} 连续度={r_splice} 召回={recall_splice} "
          f"拼接标记={flag_splice}   ← 必须被拦下")
    print(f"4 排版差异   : {v_para:10s} 连续度={r_para}   ← 必须 VERBATIM")
    print("✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接" if ok
          else "❌ 自检失败：门禁无法区分真伪引文")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="引用块逐字核验（G2b）")
    ap.add_argument("--card")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--json-out")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    cards = [Path(args.card)] if args.card else (collect_cards() if args.all else [])
    if not cards:
        ap.error("需要 --card <path> / --all / --selftest")

    index = load_fulltext_index()
    reports = [check_card(c, index) for c in cards]
    # --all 时只报有引用块的卡，避免 130 条 NO_QUOTES 噪声
    if args.all and not args.json:
        reports = [r for r in reports if r["verdict"] != "NO_QUOTES"]

    n_fab_cards = [r for r in reports if r["verdict"] == "FABRICATED"]
    n_noft = [r for r in reports if r["verdict"] == "NO_FULLTEXT"]
    # ⚠️ NO_QUOTES 必须单列，不能并进「通过」。
    # 一张**没有任何引用块**的卡片此前会被算作「通过」—— 那是「没东西可查」，
    # 不是「出处为真」，正是本项目反复封堵的假绿灯类型（与 G2 基线里
    # 「没有数字的卡片自动通过」同源）。单卡模式下 --all 的过滤不生效，这条尤其容易被误读。
    n_noq = [r for r in reports if r["verdict"] == "NO_QUOTES"]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    elif not args.quiet:
        for r in reports:
            name = Path(r["card"]).name
            v = r["verdict"]
            icon = {"VERBATIM": "✅", "FUZZY": "🟡", "FABRICATED": "❌",
                    "NO_FULLTEXT": "⚪", "NO_QUOTES": "·"}.get(v, "?")
            extra = ""
            if "n_quotes" in r:
                parts = [f"{r['n_verbatim']}/{r['n_quotes']} 逐字"]
                if r["n_fuzzy"]:
                    parts.append(f"{r['n_fuzzy']} 近似")
                if r["n_fabricated"]:
                    parts.append(f"{r['n_fabricated']} 伪造")
                extra = "，".join(parts)
            print(f"{icon} {v:10s} {name}  {extra}")
            if v in ("FABRICATED", "FUZZY"):
                for q in r["quotes"]:
                    if q["verdict"] != "VERBATIM":
                        tag = " 拼接!" if q.get("spliced") else ""
                        print(f"      [{q['verdict']} 连续={q['longest_run_ratio']} "
                              f"召回={q['ngram_recall']}{tag}] "
                              f"L{q['line']}: {q['quote'][:80]}…")
            if v == "NO_FULLTEXT":
                print(f"      {r['note']}")

        print()
        n_pass = len(reports) - len(n_fab_cards) - len(n_noft) - len(n_noq)
        print(f"共 {len(reports)} 张卡："
              f"{n_pass} 通过（有引用块且逐字可核），"
              f"{len(n_fab_cards)} 含伪造引文，"
              f"{len(n_noft)} 无全文可核验，"
              f"{len(n_noq)} 无引用块（**不等于通过**）")
        if n_noq:
            print(f"      ↑ 这 {len(n_noq)} 张卡没有任何引用块（无 `> 原文:` 行），"
                  f"G2a 会判红；quote_check 对它们无话可说。")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    return 1 if (n_fab_cards or n_noft) else 0


if __name__ == "__main__":
    sys.exit(main())
