#!/usr/bin/env python3
"""fetch_fulltext.py — arXiv 全文抓取 + HTML→Markdown 转换（零第三方依赖）

PHASE 3 前置工具：萃取卡片前必须拿到**全文**，不能只看摘要。
本脚本解决三件事：

1. **抓全文**：优先 `arxiv.org/html/{id}v{n}`（LaTeXML 渲染版，保留章节号，
   便于 `> 原文:"..."` 引用块标注「出处 = §4.2」）；回退 `/abs/` 摘要页；
   再回退 PDF 下载（仅存档，不做解析）。
2. **转 Markdown**：不用 pandoc/bs4（环境不一定有），用 stdlib `html.parser`
   自己走一遍，保留 h1-h6 / p / li / table / 公式（alttext 或注释形式）。
   关键：**保留章节层级**，否则无法产出可核验的出处标注。
3. **存档**：落到 `paper2skills-vault/papers/<domain>/<paper_id>/fulltext.md`
   （+ `paper.pdf` 若有），与 `evidence.md` 同目录，形成「全文 ↔ 证据」闭环。

用法：
    python3 fetch_fulltext.py --arxiv 2606.26690 --domain 13-广告分析 --paper-id p2s-2026-0001
    python3 fetch_fulltext.py --batch 3A          # 批量读 registry 里 PHASE3 批次
    python3 fetch_fulltext.py --arxiv 2606.26690 --stdout | head -50   # 只打印不落盘

退出码：0 全部成功；1 有失败（逐条打印原因）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VAULT = REPO_ROOT / "paper2skills-vault"
PAPERS_DIR = VAULT / "papers"
REGISTRY = VAULT / "07-资源库" / "papers_registry.json"

UA = "paper2skills/1.0 (research; contact via repo)"
# arXiv 有软性限速：连续抓取需间隔，实测 3s 足够
THROTTLE_SECONDS = 3.0


# --------------------------------------------------------------------------
# HTML → Markdown
# --------------------------------------------------------------------------

# 这些标签直接丢弃（含内容）
_DROP_TAGS = {
    "script", "style", "noscript", "svg", "nav", "footer", "header",
    "button", "form", "select", "option", "iframe", "dialog",
}
# 这些标签自成一个块级换行
_BLOCK_TAGS = {
    "p", "div", "section", "article", "table", "tr", "ul", "ol", "dl",
    "figure", "figcaption", "blockquote", "pre", "hr", "br", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
}
# 自闭合标签：**绝不能压栈**，否则栈会与 endtag 错位
_VOID_TAGS = {
    "br", "hr", "img", "meta", "link", "input", "source", "col",
    "area", "base", "embed", "param", "track", "wbr",
}

# LaTeXML 的排版载体类名：命中即整块丢弃（含内容）。
# ⚠️ 不要把 ltx_bibliography / ltx_role_footnote 放进来：
#    脚注常含关键实现细节（"we use X for Y"），参考文献是交叉核验基线，
#    两者都属于可引用内容，丢弃会造成证据链缺口。
_SKIP_CLASSES = {
    "ltx_page_logo", "ltx_page_navbar", "ltx_pagination",
    "ltx_tag_item", "ltx_note_mark", "ltx_note_outer",
    "ds-site-footer-sep", "toggle-icon", "sr-only", "modal",
    "mobile-only", "desktop-only", "header-button", "form-control",
}

# 正文判定：必须在 <article> 里，否则抓到的全是 arXiv 站点导航
_ARTICLE_RE = re.compile(r"<article\b[^>]*>(.*?)</article>", re.DOTALL | re.IGNORECASE)

_ARXIV_NOISE = re.compile(
    r"(Report number:|License:|arXiv:\d{4}\.\d{4,5}v\d+\s*\[|"
    r"Submitted to|^References$|^Acknowledgements?$)",
    re.IGNORECASE,
)


class ArxivHTMLToMarkdown(HTMLParser):
    """把 arXiv LaTeXML HTML 转成保留章节层级的 Markdown。

    设计取舍（都是实测踩出来的）：
    - LaTeXML 把公式放在 `<math alttext="...">`，正文里真正可引用的是 alttext
      （LaTeX 源码），直接取 text 会得到一串无意义的单字符。
    - 表格 `<table>` 里 LaTeXML 会插入大量 `<span class="ltx_rule">` 之类的
      排版载体，转 Markdown 表格收益为负，改为「逐行文本 + 制表符分隔」。
    - figure 的 caption 要保留（论文里的数字大多在 caption 里），
      但图片本身无意义，丢弃 `<img>`。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._heading_level = 0
        self._in_math = False
        self._math_alt = ""
        self._in_table = False
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False
        # (tag, skipping) 栈。skipping 表示「该元素及其子树都不输出」。
        # 用栈而不是计数器：计数器在「skip 元素内含未标记的同名子元素」时会
        # 提前退出 skip 区（实测 LaTeXML 的 ltx_bibliography > div 就是这种结构）。
        self._stack: list[tuple[str, bool]] = []

    @property
    def _skipping(self) -> bool:
        return bool(self._stack) and self._stack[-1][1]

    # -- helpers ----------------------------------------------------------
    def _emit(self, text: str) -> None:
        if self._skipping:
            return
        self.out.append(text)

    def _newline(self, n: int = 1) -> None:
        if self._skipping:
            return
        tail = "".join(self.out[-3:])
        if tail.endswith("\n" * n):
            return
        self.out.append("\n" * n)

    # -- parser hooks -----------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        classes = set(d.get("class", "").split())
        parent_skipping = self._skipping
        skipping = parent_skipping or tag in _DROP_TAGS \
            or bool(classes & _SKIP_CLASSES)

        if tag not in _VOID_TAGS:
            self._stack.append((tag, skipping))
        if skipping:
            return

        if tag == "math":
            self._in_math = True
            self._math_alt = d.get("alttext", "")
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._newline(2)
            self._emit("#" * self._heading_level + " ")
        elif tag == "p":
            self._newline(2)
        elif tag == "li":
            self._newline()
            self._emit("- ")
        elif tag == "table":
            self._in_table = True
            self._newline(2)
        elif tag == "tr":
            self._row = []
            self._newline()
        elif tag in {"td", "th"}:
            self._in_cell = True
            self._cell = []
        elif tag == "figcaption":
            self._newline(2)
            self._emit("*")
        elif tag == "math":
            pass
        elif tag in _BLOCK_TAGS:
            self._newline()

    def handle_startendtag(self, tag: str,
                           attrs: list[tuple[str, str | None]]) -> None:
        """`<br/>` 这类自闭合写法：当成一次 starttag + 一次 endtag。"""
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        # 弹栈直到匹配（LaTeXML 存在未闭合/交错标签，不能假设严格配对）
        popped: tuple[str, bool] | None = None
        while self._stack:
            item = self._stack.pop()
            if item[0] == tag:
                popped = item
                break
        if popped is None:
            return
        if popped[1] or self._skipping:
            # 该元素本身就在 skip 区，或其父仍在 skip 区 → 不输出
            return

        if tag == "math":
            self._in_math = False
            if self._math_alt:
                self._emit(f"${self._math_alt}$")
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = 0
            self._newline(2)
        elif tag == "p":
            self._newline(2)
        elif tag == "li":
            self._newline()
        elif tag in {"td", "th"}:
            self._in_cell = False
            if self._in_table:
                self._row.append(" ".join("".join(self._cell).split()))
            else:
                self._emit("".join(self._cell))
            self._cell = []
        elif tag == "tr":
            if self._in_table and self._row:
                self._emit("| " + " | ".join(self._row) + " |")
                self._row = []
            self._newline()
        elif tag == "table":
            self._in_table = False
            self._newline(2)
        elif tag == "figcaption":
            self._emit("*")
            self._newline(2)
        elif tag in _BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._skipping:
            return
        if self._in_math:
            return  # 只取 alttext，不取渲染字符
        if self._in_cell:
            self._cell.append(data)
            return
        # LaTeXML 会在段落里塞大量换行/缩进，压缩掉
        text = data if "\n" not in data else data.replace("\n", " ")
        self._emit(text)

    def result(self) -> str:
        raw = "".join(self.out)
        return _clean(raw)


def _clean(text: str) -> str:
    """压缩空白、去掉脚注标记与导航残留，保留段落结构。"""
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines: list[str] = []
    for line in text.split("\n"):
        s = line.rstrip()
        if not s.strip():
            lines.append("")
            continue
        if _ARXIV_NOISE.search(s.strip()):
            continue
        lines.append(s)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def html_to_markdown(html: str) -> str:
    """只解析 <article> 正文。

    ⚠️ 必须切 <article>：arXiv 的 HTML 页把站点导航（"Report GitHub Issue"、
    "Why HTML?"、"Back to arXiv"、"Download PDF"）放在 article 之外，
    整页解析会把这些噪声顶到正文最前面，污染引用底本。
    """
    m = _ARTICLE_RE.search(html)
    body = m.group(1) if m else html
    p = ArxivHTMLToMarkdown()
    p.feed(body)
    return p.result()


# --------------------------------------------------------------------------
# 抓取
# --------------------------------------------------------------------------


def _curl(url: str, out: Path, max_time: int = 120) -> tuple[int, str]:
    """用 curl 而非 urllib：arXiv 对 python-urllib 的 UA 会 403。"""
    proc = subprocess.run(
        ["curl", "-sL", "--max-time", str(max_time), "-A", UA,
         "-w", "%{http_code}", "-o", str(out), url],
        capture_output=True, text=True,
    )
    code = proc.stdout.strip()[-3:] if proc.stdout.strip() else "000"
    try:
        return int(code), ""
    except ValueError:
        return 0, proc.stdout.strip()[:200]


def fetch(arxiv_id: str, max_time: int = 120, verbose: bool = True) -> dict:
    """返回 {ok, markdown, version, source, reason}。"""
    log = (lambda *a: print(*a, file=sys.stderr)) if verbose else (lambda *a: None)

    for version in (1, 2, 3):
        url = f"https://arxiv.org/html/{arxiv_id}v{version}"
        tmp = Path(f"/tmp/p2s_{arxiv_id}_v{version}.html")
        code, err = _curl(url, tmp, max_time)
        if code == 200 and tmp.exists() and tmp.stat().st_size > 20000:
            html = tmp.read_text(encoding="utf-8", errors="replace")
            # ⚠️ 200 不代表有全文：arXiv 对无 LaTeXML 的论文会返回兜底页。
            # 判据必须是**正文标志**，不能用 "ltx_page_title" 之类猜的类名
            # （实测该 class 在当前 LaTeXML 版本中不存在，会把好页面全部误杀）。
            has_body = ("ltx_para" in html or "ltx_p" in html) and _ARTICLE_RE.search(html)
            if "No HTML for this paper" in html or not has_body:
                log(f"  {arxiv_id}v{version}: 200 但无 LaTeXML 正文，跳过")
                continue
            md = html_to_markdown(html)
            if len(md) < 5000:
                log(f"  {arxiv_id}v{version}: 转换后仅 {len(md)} 字符，疑似失败，跳过")
                continue
            log(f"  {arxiv_id}v{version}: ✅ HTML {tmp.stat().st_size} → md {len(md)} 字符")
            return {"ok": True, "markdown": md, "version": version,
                    "source": url, "reason": ""}
        else:
            log(f"  {arxiv_id}v{version}: HTTP {code} {err}")
        time.sleep(THROTTLE_SECONDS)

    # 回退：摘要页（至少保住摘要 + 元数据，卡片可标 evidence_grade=abstract）
    tmp = Path(f"/tmp/p2s_{arxiv_id}_abs.html")
    code, _ = _curl(f"https://arxiv.org/abs/{arxiv_id}", tmp, 60)
    if code == 200 and tmp.exists():
        md = html_to_markdown(tmp.read_text(encoding="utf-8", errors="replace"))
        log(f"  {arxiv_id}: ⚠️ 回退到摘要页（{len(md)} 字符）")
        return {"ok": False, "markdown": md, "version": 0,
                "source": f"https://arxiv.org/abs/{arxiv_id}",
                "reason": "无 HTML 全文；仅摘要（evidence_grade 须降为 abstract）"}

    return {"ok": False, "markdown": "", "version": 0, "source": "",
            "reason": "HTML 与 abs 页均不可用"}


# --------------------------------------------------------------------------
# 落档
# --------------------------------------------------------------------------


def archive(arxiv_id: str, domain: str, paper_id: str, md: str,
            source: str, ok: bool) -> Path:
    outdir = PAPERS_DIR / domain / paper_id
    outdir.mkdir(parents=True, exist_ok=True)
    header = (
        f"<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py\n"
        f"     arxiv_id : {arxiv_id}\n"
        f"     paper_id : {paper_id}\n"
        f"     source   : {source}\n"
        f"     fulltext : {'是' if ok else '否（仅摘要）'}\n"
        f"     用途     : evidence.md 的 `> 原文:\"...\"` 引用块的出处核验底本\n"
        f"-->\n\n"
    )
    dst = outdir / "fulltext.md"
    dst.write_text(header + md, encoding="utf-8")
    return dst


BATCHES = {
    "3A": ["2606.26690", "2608.11675", "2608.10182", "2608.18174"],
    "3B": ["2607.16230", "2608.25871", "2607.09745", "2606.29366"],
    "3C": ["2607.09608", "2608.27006", "2608.20844"],
    "3D": ["2608.26263", "2608.22152", "2608.25277", "2607.22115", "2609.01038"],
    "3E": ["2608.28978", "2608.09162", "2608.10240"],
}


def _registry_lookup() -> dict[str, dict]:
    if not REGISTRY.exists():
        return {}
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for r in data.get("records", []):
        aid = (r.get("identifiers") or {}).get("arxiv")
        if aid:
            out[aid] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="arXiv 全文抓取 → Markdown 存档")
    ap.add_argument("--arxiv", action="append", default=[],
                    help="arXiv ID，可重复")
    ap.add_argument("--batch", choices=sorted(BATCHES),
                    help="批量抓取 PHASE3 批次")
    ap.add_argument("--domain", help="目标领域目录名，如 13-广告分析")
    ap.add_argument("--paper-id", help="registry paper_id，如 p2s-2026-0001")
    ap.add_argument("--stdout", action="store_true", help="只打印，不落盘")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ids = list(args.arxiv)
    if args.batch:
        ids += BATCHES[args.batch]
    if not ids:
        ap.error("需要 --arxiv 或 --batch")

    reg = _registry_lookup()
    failures: list[str] = []

    for aid in ids:
        rec = reg.get(aid, {})
        domain = args.domain or rec.get("domain") or "_unfiled"
        pid = args.paper_id if (args.paper_id and len(ids) == 1) \
            else rec.get("paper_id") or f"arxiv-{aid}"
        if not args.quiet:
            print(f"=== {aid} → {domain}/{pid} ===")
        res = fetch(aid, verbose=not args.quiet)

        if args.stdout:
            print(res["markdown"])
            continue

        if not res["markdown"]:
            failures.append(f"{aid}: {res['reason']}")
            continue

        dst = archive(aid, domain, pid, res["markdown"], res["source"], res["ok"])
        print(f"  → {dst.relative_to(REPO_ROOT)}  ({len(res['markdown'])} 字符)")
        if not res["ok"]:
            failures.append(f"{aid}: {res['reason']}")
        time.sleep(THROTTLE_SECONDS)

    if failures:
        print("\n失败/降级：")
        for f in failures:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
