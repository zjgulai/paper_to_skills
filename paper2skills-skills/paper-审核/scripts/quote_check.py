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

## 版本校验（2026-09-13 新增，对应缺陷 C5 / 甲类 A6）

⚠️ **逐字命中只证明「这句话存在于底本」，不证明「它来自卡片所声称的那个版本」。**

实测 A6（`Skill-Reflexion-Self-Improvement.md`）：卡片引的是 **NeurIPS 正式版**结论
（HumanEval 91% pass@1），而仓库底本是 **arXiv v1**（v1 只评测 AlfWorld 与 HotPotQA，
全文 `HumanEval` / `pass@1` **各 0 命中**）。引文本身逐字存在于 v1，于是本脚本报
VERBATIM —— **绿的是另一句话**。这是漏洞 #15（残句逐字存在于底本 → 报 VERBATIM）
那一类假绿灯的**变体**：核验器确实找到了字符串，只是找错了版本。

故本脚本在逐字判定之外，另有**一层独立的版本判定**，产出三个**新增**字段：

| 字段 | 含义 |
|------|------|
| `version_claim` | 卡片**声称**的版本：`v1` / `正式版(NeurIPS 2023)` / `""`（未声明） |
| `fulltext_version` | 底本**实际**版本：`v1`（读底本头部 `arxiv_version` / `version_id`，或从 `source:` URL 解析） |
| `version_verdict` | `MATCH` / `MISMATCH` / `UNKNOWN`（无从比较）/ `UNCLAIMED`（卡片未声明版本） |

**口径纪律（三条，缺一条就是放水）：**

1. **版本错配一律黄灯，绝不判红**，且必须与 `FABRICATED` **分开报** ——
   版本错配是「你可能引错了版本」，伪造是「这句话根本不存在」，两者性质不同。
   （与「三个门禁必须分开报」是同一条纪律。）
2. **`version_verdict` 是新增字段，`verdict` 一个字节都不动**。
   `gate_check.py` 读的是 `verdict` 与 `n_*` 计数；把版本错配混进 `verdict`
   会连带打挂 K2 门禁，也会把「黄」伪装成「红」。
3. **`unknown` 不是「一致」**。PDF 路径解不出 arXiv 版本号 → 只出 `UNKNOWN`，
   绝不给出版本结论。「没东西可比」≠「比过了没问题」。

用法：
    python3 quote_check.py --card <card.md>              # 单卡
    python3 quote_check.py --all                         # 全库（只报有引用块的卡）
    python3 quote_check.py --card <c.md> --json          # 机器可读
    python3 quote_check.py --selftest                    # 自检：证明它真的能抓到伪造引文

退出码：0 = 无 FABRICATED；1 = 存在 FABRICATED 或无法核验。
        ⚠️ 版本错配（黄灯）**不改变退出码** —— 它不阻塞，只提示。
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
#
# ⚠️⚠️ 第二个坑（2026-09-12 实测，同样是假红灯的子类）：
# 旧写法把开引号集 `["“「『]` 与闭引号集 `["”」』]` 写成两个**独立**字符类，
# 于是**任何闭引号都能闭合任何开引号**。后果：一张卡的证据声明行里若出现
# `「⑥ 原文引用」` 这样的中文书名号，`「` 会被当成开引号、后面的 `」` 闭合它，
# 而 `re.S` 让 `.+?` 跨行吞掉整段声明 —— 于是一张**完全没有引文**的卡
# 被抽出 1 条「引文」，再因找不到全文而被判 `NO_FULLTEXT`。
# 实测 6 张卡（AB-Experimental-Design / AGRS / MAA / StaR / VOC-Semantic-Blueprint /
# VOC-Proxy-NPS）全部中招，其中 5 张是我刚加的声明文本触发的。
#
# ⚠️⚠️ 第三个坑（2026-09-12 实测，漏洞 #15）：**引文内嵌双引号会被截成残句**。
# `"[^"\n]+"` 是非贪婪的，遇到内层引号就闭合。实测两条真实引文：
#     > 原文:"These results indicate that the LLM-based "Fuzzy" Random Forest model is …"
# 被抽成 42 字符的 `These results indicate that the LLM-based `，
# 而这条**残句确实逐字存在于底本**，于是 quote_check 报 VERBATIM ——
# **门禁核验的不是卡片声称的那句话，而是一段更短的碎片**，
# 且因为 `n_quotes` 与逐字命中都正常，报告里完全看不出来。
# 全库扫描：2,085 条引文中 4 条被截断（1 条已由子代理自行发现并改写）。
#
# 修法：ASCII 双引号分支改用 `"(.+)"`（**贪婪**），让它匹配到**行内最后一个**引号。
# 这样内层引号会被一并吞进引文，与卡片实际声称的文本一致；
# 另有「尾随内容检测」selftest 用例锁定该行为。
# 注：全角引号分支保持非贪婪 —— 它们不存在与 ASCII 混用的歧义。
QUOTE_RE = re.compile(
    r"^>[ \t]*(?:原文\s*[:：][ \t]*)?"
    r'(?:"(.+)"|“([^”\n]+)”|「([^」\n]+)」|『([^』\n]+)』)'
    r"[ \t]*$",
    re.M,
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

# ---------------------------------------------------------------------------
# 版本校验（缺陷 C5 / 甲类 A6）—— 与逐字判定**完全独立**的一层
# ---------------------------------------------------------------------------

# arXiv ID（可带版本后缀）。前置断言排除数字内部片段：
# `(?<![0-9])…(?![0-9])` 而不是 `\b` —— 因为 `2303.11366v1` 里
# `\b` 会在 `11366` 与 `v` 之间断开，`\b` 版本反而会漏掉带版本的写法。
ARXIV_VER_RE = re.compile(r"(?<![0-9])(\d{4}\.\d{4,5})(v\d+)(?![0-9])")
ARXIV_ANY_RE = re.compile(r"(?<![0-9])(\d{4}\.\d{4,5})(v\d+)?(?![0-9])")

# 底本头部里由 fetch_fulltext.py 落盘的版本字段
FT_VER_FIELD_RE = re.compile(r"^\s*(arxiv_version|version_id)\s*:\s*(\S+)\s*$", re.M)
FT_SOURCE_RE = re.compile(r"^\s*source\s*:\s*(\S.*?)\s*$", re.M)
FT_ARXIV_URL_VER_RE = re.compile(r"arxiv\.org/(?:html|pdf|abs)/(\d{4}\.\d{4,5})v(\d+)")

# 「预印本」标记：命中即认为卡片**没有**声称正式发表版。
#
# ⚠️ 这里踩过一个坑（2026-09-13 由**故意破坏实验**撞出）：
# 第一版把这条守卫套在 `FORMAL_VENUE_*` 的**命中片段**（如 `CIKM 2026`）上，
# 而那个片段按构造只可能含「会议名 + 年份」——**守卫永远不可能触发**。
# 把守卫改成 `if False:` 后自检**依然全绿**，即它是一段**死代码**。
# 死守卫比没有守卫更危险：它看起来像有保护。
# 现改为套在**整个 venue 字段 / 参考区条目**上，并由用例 6g 锁定其真实生效。
#
# ⚠️ 刻意**不含** `arxiv` 与 `底本未声明`：
#   - `arxiv`：`WSDM 2025, arXiv:2402.09176` 是**真错配**（甲类 A2），
#     整行判会让 arXiv 号把真阳性吃掉。而 `venue: arXiv preprint` 本来
#     就不会命中 `FORMAL_VENUE_NAME_RE`（名字表里没有 arXiv），无需这条兜底。
#   - `底本未声明`：`venue: ACL 2025 (底本未声明 venue)` 说的是**存档**的情况，
#     并不否定「卡片声称 ACL 2025」这一事实，不该被豁免。
VENUE_PREPRINT_RE = re.compile(
    r"preprint|under\s+review|submitted\s+to|in\s+preparation|working\s+paper|预印本",
    re.I,
)

# 「正式发表版」标记：会议 / 期刊名。
#
# ⚠️ **两套，不要合并**：
#   - `FORMAL_VENUE_RE`（**年份必需**）用于**正文/参考区散文扫描**。
#     年份是必需项，不是可选装饰：`WWW` 会命中网址里的 `www.`，
#     `ACL` / `POM` / `ISR` 这类缩写会命中普通英文单词。加了年份后
#     实测误报从「56 张」降到「25 张」。例：`Skill-Active-Context-Pruning`
#     的「相关基础」里写 `- **Reflexion** (NeurIPS 2023)`，
#     无年份约束时会被误读成「本卡声称 NeurIPS 2023 正式版」。
#   - `FORMAL_VENUE_NAME_RE`（**年份可选**）用于**结构化字段** `frontmatter.venue`。
#     该字段是人工策展的单一槽位、不掺散文，不会出现 `www.` 这类噪声；
#     实测 `venue: EMNLP`（无年份）是**真错配**，若也要求年份就漏掉了。
_VENUE_NAMES = (
    r"NeurIPS|NIPS|ICML|ICLR|KDD|SIGIR|WSDM|CIKM|ACL|EMNLP|NAACL|"
    r"AAAI|IJCAI|RecSys|CVPR|ICCV|ECCV|AISTATS|UAI|WSC|ADKDD|TKDD|TOIS|"
    r"SIGMOD|VLDB|WWW|The\s+Web\s+Conference|"
    r"Management\s+Science|Operations\s+Research|Marketing\s+Science|"
    r"MIS\s+Quarterly|Journal\s+of\s+Marketing(?:\s+Research)?|"
    r"Production\s+and\s+Operations\s+Management|"
    r"Information\s+Systems\s+Research|"
    r"Winter\s+Simulation\s+Conference"
)
FORMAL_VENUE_NAME_RE = re.compile(
    rf"\b(?:{_VENUE_NAMES})\b(?:[\s,（(]*(?:19|20)\d\d)?", re.I)
FORMAL_VENUE_RE = re.compile(
    rf"\b(?:{_VENUE_NAMES})\b[\s,（(]*(?:19|20)\d\d\b", re.I)

# 卡片里承载「本文出处声明」的区段标题
REF_SECTION_RE = re.compile(
    r"^#{1,4}[ \t]*(?:参考论文|参考资料|参考文献|参考来源|延伸阅读)[^\n]*$", re.M)

# 版本判定结果（**新增字段**，不参与 `verdict`）
V_MATCH = "MATCH"            # 声称版本 == 底本版本
V_MISMATCH = "MISMATCH"      # 声称版本 != 底本版本 → 黄灯（不是红灯）
V_UNKNOWN = "UNKNOWN"        # 底本版本无从比较（PDF / 头部无版本标识）
V_UNCLAIMED = "UNCLAIMED"    # 卡片未声明版本 → 无版本结论可下

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

    ⚠️ **版本键必须一起进索引**（2026-09-13，缺陷 C5）：
    `arxiv_version: v1` / `version_id: 2303.11366v1` 让底本可被**按版本定位**。
    否则「卡片写 `paper_id: 2303.11366v2`」既匹配不到 v1 存档（→ 假 NO_FULLTEXT），
    也无法暴露版本错配。注意这里同时给**基号**（`2303.11366`）与
    **带版本号**（`2303.11366v1`）两种键 —— 少任何一个都会制造假红灯/假绿灯。
    """
    keys = [md.parent.name]
    try:
        head = md.read_text(encoding="utf-8", errors="replace")[:1200]
    except OSError:
        return keys
    for field in ("arxiv_id", "paper_id"):
        m = re.search(rf"^\s*{field}\s*:\s*(\S+)\s*$", head, re.M)
        if m:
            keys.append(m.group(1))
    vid, _ = fulltext_version(md)
    if vid and vid != "unknown":
        keys.append(vid)
        base = ARXIV_ANY_RE.match(vid)
        if base:
            keys.append(base.group(1))
    return keys


def fulltext_version(md: Path) -> tuple[str, str]:
    """底本**实际**版本 → (version_id, version)，如 `("2303.11366v1", "v1")`。

    解析优先级（必须兼容**存量 93 个存档**，它们是在本字段加入之前落盘的）：
      1. 头部显式字段 `version_id:`（fetch_fulltext 新版会写）
      2. 头部显式字段 `arxiv_version:`
      3. 头部 `source:` URL 里的 `.../{id}v{n}`（**存量存档走这条**）
    解不出（PDF 路径 / 本地 PDF 转换）→ `("unknown", "unknown")`。

    ⚠️ 返回 `unknown` 时下游只出 `UNKNOWN`，**绝不给出版本结论** ——
    「没东西可比」不等于「比过了没问题」。
    """
    try:
        head = md.read_text(encoding="utf-8", errors="replace")[:1200]
    except OSError:
        return "unknown", "unknown"

    fields = {k: v for k, v in FT_VER_FIELD_RE.findall(head)}
    vid = fields.get("version_id", "")
    if vid and vid.lower() != "unknown" and ARXIV_VER_RE.search(vid):
        return vid, ARXIV_VER_RE.search(vid).group(2)
    m = FT_SOURCE_RE.search(head)
    src = m.group(1) if m else ""
    u = FT_ARXIV_URL_VER_RE.search(src)
    if u:
        return f"{u.group(1)}v{u.group(2)}", f"v{u.group(2)}"
    vlab = fields.get("arxiv_version", "")
    if vlab and vlab.lower() != "unknown":
        mid = ARXIV_ANY_RE.search(head)
        if mid and re.fullmatch(r"v\d+", vlab):
            return f"{mid.group(1)}{vlab}", vlab
    return "unknown", "unknown"


def _ref_section(text: str) -> str:
    """卡片尾部的「参考论文 / 参考资料 / 参考文献」区段（无则空串）。"""
    m = REF_SECTION_RE.search(text)
    return text[m.start():] if m else ""


def _first_ref_entry(ref_section: str) -> str:
    """参考区里的**第一条**条目。

    ⚠️ 只取第一条，是**实测校准**出来的，不是随意选择：
    参考区第 1 条是「本卡的来源论文」，第 2 条起（以及 `## 相关基础`）
    是延伸阅读。实测 `Skill-Active-Context-Pruning` 的
    `- **Reflexion** (NeurIPS 2023)` 出现在「相关基础」里，
    若扫全区段就会把它误当成「本卡声称 NeurIPS 2023 正式版」——
    实测误报从 25 张降到 12 张，且留下的全是真错配。
    """
    m = REF_SECTION_RE.search(ref_section)
    if not m:
        return ""
    body = ref_section[m.end():]
    out: list[str] = []
    started = False
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            if started:
                break
            continue
        if re.match(r"^#{1,6}[ \t]", s):
            break
        if re.match(r"^(?:[-*+][ \t]*)?\d{1,2}[.、)][ \t]", s):
            if started:
                break                    # 第 2 条起 → 第一条结束
            started = True
            out.append(s)
            continue
        if not started:
            if re.match(r"^[-*+][ \t]", s):   # 无编号的条目式参考区
                started = True
                out.append(s)
            continue
        out.append(s)
    return "\n".join(out)


def claimed_version(text: str, fm: dict, blob: str = "") -> tuple[str, str, str]:
    """卡片**声称**的版本 → (claim, kind, evidence)。

    `kind` 表示这条声称的**强度**，用于把「硬错配」与「待确认」分开报：
      - `declared` —— frontmatter `paper_version:` 显式声明（最强）
      - `explicit` —— `paper_id:` 或 `> 出处：` 行里带 `v{n}`（硬错配）
      - `venue`    —— 卡片声称论文**正式发表**在某会议/期刊（软：底本是 arXiv 快照）
      - `""`       —— 未声明版本 → 无版本结论可下（`UNCLAIMED`）

    ⚠️ **只扫 `> 出处：` 行，不全文扫 `v{n}`。** 实测 `Skill-NPS-Driver-Analysis`
    正文同时出现 `v1` 与 `v3`（讲的是两篇不同论文），全文扫会凭空造出一个
    「声称 v3」的假错配 —— 假红灯会让人不再相信门禁。
    """
    # 1) frontmatter 显式版本声明（本脚本新认的字段，卡片可选）
    pv = (fm.get("paper_version") or "").strip()
    if pv:
        if re.fullmatch(r"v\d+", pv):
            return pv, "declared", f"frontmatter paper_version: {pv}"
        if ARXIV_VER_RE.search(pv):
            m = ARXIV_VER_RE.search(pv)
            return m.group(2), "declared", f"frontmatter paper_version: {pv}"

    # 2) paper_id 带版本后缀
    pid = (fm.get("paper_id") or "").strip()
    m = ARXIV_VER_RE.search(pid)
    if m:
        return m.group(2), "explicit", f"frontmatter paper_id: {pid}"

    # 3) 出处行带版本后缀
    for line in SOURCE_RE.findall(blob or text):
        m = ARXIV_VER_RE.search(line)
        if m:
            return m.group(2), "explicit", f"出处行：{line.strip()[:80]}"

    # 4) 声称正式发表版：frontmatter venue，或参考区第 1 条
    #
    # ⚠️ 「预印本标记」判的是**整个 venue 字段 / 整个参考条目**（不是命中片段）
    # —— 详见 `VENUE_PREPRINT_RE` 上方关于「死守卫」的说明。
    # 另：`Skill-Cold-Start-Product-Recommendation` 参考区第 1 条是
    # `- **Large Language Model Simulator for Cold-Start Recommendation**, WSDM 2025, arXiv:2402.09176`
    # —— 它是**真错配**（甲类 A2：底本全文 `WSDM` 零命中）。判据要落在
    # 「卡片把 venue 声明成了什么」+「有没有自称预印本」，而不是「这行还写了别的什么」。
    venue = (fm.get("venue") or "").strip()
    if venue:
        vm = FORMAL_VENUE_NAME_RE.search(venue)
        if vm and not VENUE_PREPRINT_RE.search(venue):
            shown = " ".join(vm.group(0).split())
            return f"正式版({shown})", "venue", f"frontmatter venue: {venue}"

    entry = _first_ref_entry(text)
    if entry:
        vm = FORMAL_VENUE_RE.search(entry)
        if vm and not VENUE_PREPRINT_RE.search(entry):
            shown = " ".join(vm.group(0).split())
            return (f"正式版({shown})", "venue",
                    f"参考区第 1 条：{entry.split(chr(10))[0].strip()[:90]}")
    return "", "", ""


def version_check(text: str, fm: dict, blob: str, ft: Path | None) -> dict:
    """版本判定 —— 与逐字判定**完全独立**的一层，只产出**新增**字段。

    绝不动 `verdict`：`gate_check.py` 读的是 `verdict` 与 `n_*` 计数，
    把版本错配混进去会连带打挂 K2 门禁，也会把「黄」伪装成「红」。
    """
    claim, kind, evidence = claimed_version(text, fm, blob)
    if ft is None:
        vid, vlabel = "unknown", "unknown"
    else:
        vid, vlabel = fulltext_version(ft)

    out = {
        "version_claim": claim,
        "version_claim_kind": kind,
        "version_claim_evidence": evidence,
        "fulltext_version": vlabel,
        "fulltext_version_id": vid,
    }

    if not claim:
        out["version_verdict"] = V_UNCLAIMED
        out["version_note"] = (
            f"卡片未声明论文版本；底本版本={vlabel}。"
            f"**未声明不等于一致** —— 只是无版本结论可下。")
        return out

    if vlabel == "unknown":
        out["version_verdict"] = V_UNKNOWN
        out["version_note"] = (
            f"卡片声称 {claim}，但底本解不出 arXiv 版本号（PDF 路径 / 存档无版本标识）"
            f"→ **无从比较**，不给版本结论。")
        return out

    # ⚠️ `declared` 必须与 `explicit` 同权（2026-09-13 修）。
    # 原写 `if kind == "explicit" and claim == vlabel` —— 于是 frontmatter
    # `paper_version:` 这条路径**永远拿不到 MATCH**，会直落下面的 else，
    # 被贴上「卡片声称**正式发表版**」的标签判 MISMATCH：
    # **版本一致却报黄灯**。而 `claimed_version` 的文档明确把 declared
    # 称为「最强」的声明强度 —— 判据与文档自相矛盾。
    # 更糟的是本脚本自己印的处置建议就是「在 ⑥ 段显式声明底本版本
    # （frontmatter `paper_version:`）」—— 即**门禁在推荐一条会把门禁自己
    # 搞出假黄灯的路径**。假黄灯会让人不再相信版本层，与假红灯同害。
    if kind in ("explicit", "declared") and claim == vlabel:
        out["version_verdict"] = V_MATCH
        out["version_note"] = f"卡片声称 {claim}，底本 {vlabel} —— 一致。"
        return out

    if kind in ("explicit", "declared"):
        reason = (f"卡片声称 {claim}，底本是 {vlabel} —— **版本号不同**。")
    else:
        reason = (f"卡片声称**正式发表版**（{claim}），底本是 arXiv 预印本快照 "
                  f"{vid} —— 两者内容可能不同。")
    out["version_verdict"] = V_MISMATCH
    out["version_note"] = (
        f"{reason}⚠️ **版本错配不等于引文伪造**（未判红）：逐字命中只证明该句"
        f"存在于**本底本**，不证明它出自卡片所声称的版本。请补抓对应版本全文，"
        f"或在卡片 ⑥ 段显式声明底本版本。依据：{evidence}")
    return out


def find_fulltext(paper_id: str, index: dict[str, Path] | None = None) -> Path | None:
    """按 paper_id（arXiv ID / DOI / registry id）找 fulltext.md。

    ⚠️ 带版本号的 paper_id（`2303.11366v2`）必须能落到**基号**底本上
    （2026-09-13，缺陷 C5）：若直接放弃，卡片写 `paper_id: 2303.11366v2`
    而底本是 v1 时会报 `NO_FULLTEXT`（⚪ 无全文可核验）——
    那把一个**可判定的版本错配**变成了「无从核验」，是假绿灯。
    现改为：精确匹配失败 → 剥掉版本号再匹一次，交给 `version_check` 判版本。
    """
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
    if not cands:
        base = ARXIV_ANY_RE.match(pid)
        if base and base.group(1) != pid:
            cands = [p for p in hits if base.group(1) in _index_keys(p)]
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
        # 引号成对同型后用 4 个分组承载四种引号，取第一个非 None 的
        raw = next(g for g in m.groups() if g is not None).strip()
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
                "note": "卡片与 evidence.md 中都没有 `> 原文:\"...\"` 引用块",
                **version_check(text, fm, blob, None)}

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
            **version_check(text, fm, blob, None),
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
            **version_check(text, fm, blob, ft),
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
        # 版本校验是**独立一层**，只新增字段，不动 verdict（见模块文档）
        **version_check(text, fm, blob, ft),
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

    # 用例 5｜内嵌双引号不得被截成残句（漏洞 #15）
    # 实测原句：`…the LLM-based "Fuzzy" Random Forest model is …`
    # 旧正则抽成 42 字符的残句，而残句本身逐字存在于底本 → 报 VERBATIM，
    # 即**门禁核验的不是卡片声称的那句话**。这里锁定「抽出的必须与写入的一致」。
    inner = ('These results indicate that the LLM-based "Fuzzy" Random Forest model '
             'is a highly effective tool for predicting startup success.')
    line = f'> 原文:"{inner}"\n'
    got = extract_quotes(line)
    q_inner_ok = len(got) == 1 and got[0]["quote"] == inner
    print(f"5 内嵌双引号: 抽出 {len(got)} 条，长度 {len(got[0]['quote']) if got else 0}"
          f"（期望 {len(inner)}）{'✅' if q_inner_ok else '❌ 被截断'}")
    ok = ok and q_inner_ok

    # 用例 6｜**版本错配**：v1 底本 + 卡片声称正式版结论（缺陷 C5 / 甲类 A6）
    #
    # 为什么必须单独一条：逐字命中只证明「这句话存在于**本底本**」。
    # 实测 A6 的引文全都逐字存在于 v1，逐字层全绿；绿的是**另一句话** ——
    # 卡片真正引的是 NeurIPS 正式版（v1 里 `HumanEval`/`pass@1` 各 0 命中）。
    # 这条用例锁定「版本层必须能独立发现它」。
    #
    # ⚠️ 三向锁定，缺一条这个机制就是「见卡就报黄」：
    #   6a v1 底本 + 正式版声明      → 必须 MISMATCH
    #   6b v1 底本 + 未声明版本      → 必须 UNCLAIMED（不是 MISMATCH）
    #   6c v1 底本 + 声称 v1         → 必须 MATCH
    #   6d 无版本标识的底本 + 正式版 → 必须 UNKNOWN（「无从比较」≠「一致」）
    #   6e 版本标识必须**真的进了索引**（而不是只在报告里多印一行）
    # 6b/6c/6d/6e 就是「豁免条款必须比拦截条款测得更严」在本层的落地。
    v6 = _selftest_version_cases()

    print(f"1 真引文     : {v_real:10s} 连续度={r_real}")
    print(f"2 纯伪造     : {v_fake:10s} 连续度={r_fake}   ← 必须 FABRICATED")
    print(f"3 拼接引文   : {v_splice:10s} 连续度={r_splice} 召回={recall_splice} "
          f"拼接标记={flag_splice}   ← 必须被拦下")
    print(f"4 排版差异   : {v_para:10s} 连续度={r_para}   ← 必须 VERBATIM")
    print(f"6 版本错配   : {v6['summary']}")
    for line in v6["detail"]:
        print(f"      {line}")
    ok = ok and v6["ok"]
    print("✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接 / 版本错配" if ok
          else "❌ 自检失败：门禁无法区分真伪引文")
    return 0 if ok else 1


def _selftest_version_cases() -> dict:
    """用例 6 的四向构造样本 —— 全部走 `check_card()` 真实代码路径。

    ⚠️ 必须调用 `check_card()` 而不是直接调 `version_check()`：
    只有走完整路径才能证明「底本定位 → 版本解析 → 判定」这条链真的接通了。
    本仓库已发生过「第一版修复后门禁照样绿，因为它压根没找到文件」——
    验收标准是**修复真的生效**，不是**门禁还绿着**。
    """
    import tempfile

    v1_sentence = ("We observe success rates of 97% and 51%, respectively, and provide "
                   "a discussion on the emergent property of self-reflection.")
    filler = "The agent maintains an episodic memory buffer across trials. " * 200
    pool = Path(tempfile.mkdtemp(prefix="p2s_v6_"))

    def make_ft(name: str, header: str) -> Path:
        p = pool / name
        p.write_text(header + "\n# Reflexion: an autonomous agent with dynamic "
                              "memory and self-reflection\n\n" + filler +
                     "\n\n" + v1_sentence + "\n", encoding="utf-8")
        return p

    # 6/6c 用的底本：**新版头部**（显式 version_id）
    ft_new = make_ft("v1_new.md", (
        "<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py\n"
        "     arxiv_id : 2303.11366\n"
        "     paper_id : 2303.11366\n"
        "     source   : https://arxiv.org/html/2303.11366v1\n"
        "     fulltext : 是\n"
        "     arxiv_version : v1\n"
        "     version_id : 2303.11366v1\n"
        "-->\n"))
    # 6a/6b 用的底本：**存量 93 个存档的真实形态**（只有 source 行，无版本字段）
    ft_legacy = make_ft("v1_legacy.md", (
        "<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py\n"
        "     arxiv_id : 2303.11366\n"
        "     paper_id : 2303.11366\n"
        "     source   : https://arxiv.org/html/2303.11366v1\n"
        "     fulltext : 是\n"
        "-->\n"))
    # 6d 用的底本：PDF 路径 → 解不出版本号
    ft_pdf = make_ft("pdf.md", (
        "<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf\n"
        "     arxiv_id : 2303.11366\n"
        "     paper_id : 2303.11366\n"
        "     source   : papers/10-MAS/2303.11366/paper.pdf\n"
        "     fulltext : 是（本地 PDF 转换）\n"
        "     arxiv_version : unknown\n"
        "     version_id : unknown\n"
        "-->\n"))

    def make_card(name: str, pmid: str, ref_entry: str,
                  venue_line: str = "", extra_fm: str = "") -> Path:
        p = pool / name
        p.write_text(
            "---\n"
            f"title: 版本校验自检样本\n"
            f"paper_id: {pmid}\n"
            "paper: \"Reflexion: an autonomous agent with dynamic memory and self-reflection\"\n"
            "evidence_basis: paper-verbatim\n"
            + venue_line + extra_fm +
            "---\n\n"
            "## 参考论文\n\n" + ref_entry + "\n\n"
            "## ⑥ 原文引用\n\n"
            f"> 原文:\"{v1_sentence}\"\n"
            f"> 出处：{pmid.split('v')[0]} Abstract\n",
            encoding="utf-8")
        return p

    neurips = ("1. **Reflexion: Language Agents with Verbal Reinforcement Learning** "
               "(NeurIPS 2023)\n"
               "   - Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S.\n"
               "   - arXiv：2303.11366\n")
    plain = ("1. **Reflexion: an autonomous agent with dynamic memory and "
             "self-reflection** (2023)\n"
             "   - Shinn, N., Labash, B., Gopinath, A.\n"
             "   - arXiv：2303.11366\n")

    def run(card: Path, ft: Path) -> dict:
        # index 必须非空，否则 check_card 会回落到真实 PAPERS_DIR 去找底本。
        # ⚠️ 而且**必须把带版本号的键也钉进去**：`paper_id: 2303.11366v1` 的样本
        # 若只钉基号，会走 find_fulltext 的 glob 兜底、命中**真实仓库**里的
        # Reflexion 存档 —— 那样 6c 会因为「恰好也是 v1」而通过，
        # 测的是一个环境依赖的巧合，不是本脚本的逻辑。
        return check_card(card, {
            "2303.11366": ft,
            "2303.11366v1": ft,
            "2303.11366v2": ft,
        })

    r6a = run(make_card("c6a.md", "2303.11366", neurips), ft_legacy)
    r6b = run(make_card("c6b.md", "2303.11366", plain), ft_legacy)
    r6c = run(make_card("c6c.md", "2303.11366v1", neurips), ft_new)
    r6d = run(make_card("c6d.md", "2303.11366", neurips), ft_pdf)
    # 6f/6g 覆盖 **frontmatter `venue:`** 这条路径 —— 它贡献了全库 20 张错配里的 9 张，
    # 而 6a 走的是「参考区第 1 条」，**覆盖不到它**。
    # ⚠️ 这个洞是**故意破坏实验撞出来的**：把 frontmatter venue 分支改成 `if False:`
    # 后自检**依然全绿** —— 即「9/20 的真错配可以整条路径失效而无人发现」。
    # 自检没覆盖到的分支就是没有保护的分支。
    r6f = run(make_card("c6f.md", "2303.11366", plain, "venue: CIKM 2026\n"), ft_legacy)
    # 6g 反过来锁定**豁免条款**：自称预印本的卡**不得**被读成「声称正式版」。
    # 豁免判据必须比拦截判据测得更严。（这条同时是「死守卫」的回归网 ——
    # 第一版把预印本守卫套在只可能含「会议名+年份」的命中片段上，永远不触发。）
    r6g = run(make_card("c6g.md", "2303.11366", plain, "venue: arXiv preprint\n"), ft_legacy)
    r6h = run(make_card("c6h.md", "2303.11366", plain,
                        "venue: CIKM 2026 (preprint, under review)\n"), ft_legacy)
    # 6i：`底本未声明 venue` 说的是**存档**的情况，不否定「卡片声称 ACL 2025」，
    # 故**必须仍判错配**（这是全库真卡 `Skill-CrossLingual-Sentiment-Transfer` 的原字符串）。
    r6i = run(make_card("c6i.md", "2303.11366", plain,
                        "venue: ACL 2025 (底本未声明 venue)\n"), ft_legacy)
    # 6j：参考区条目的预印本豁免（与 6h 同构，但走的是**另一条**代码路径）。
    # ⚠️ 这条同样是**故意破坏实验**撞出来的：单独删掉参考区那条守卫时
    # 6a–6i **全绿** —— frontmatter 与参考区各有一份豁免判据，
    # 必须各有一条用例，否则删掉任一份都无人发现。
    ref_preprint = ("1. **SkillForge: Self-Evolving Agent Skills** "
                    "(CIKM 2026 preprint, under review)\n"
                    "   - Liu, X.\n   - arXiv：2303.11366\n")
    r6j = run(make_card("c6j.md", "2303.11366", ref_preprint), ft_legacy)
    # 6k/6l：frontmatter `paper_version:`（`kind="declared"`）这条路径。
    # ⚠️ 这对用例来自一个**实测撞出来的门禁 bug**（2026-09-13，由 Y5 子代理发现）：
    # `version_check` 原写 `if kind == "explicit" and claim == vlabel`，于是
    # declared 声明**永远拿不到 MATCH**，直落 else 被判 MISMATCH —— **版本一致却报黄灯**。
    # 而本脚本印给用户的处置建议恰恰就是「显式声明 `paper_version:`」。
    # **必须成对**：一致要 MATCH、不一致要 MISMATCH ——
    # 只测前者的话，把判据改成「恒 MATCH」也能全绿（那会让整个版本层失效）。
    r6k = run(make_card("c6k.md", "2303.11366", plain,
                        extra_fm="paper_version: v1\n"), ft_legacy)
    r6l = run(make_card("c6l.md", "2303.11366", plain,
                        extra_fm="paper_version: v3\n"), ft_legacy)

    # --- 6e：版本标识必须真的进了 `_index_keys`，且基号兜底必须能落到带版本档 ---
    # ⚠️ 这一条针对的是一个**真实踩过的坑**：本仓库曾出现「第一版修复后门禁照样绿，
    # 因为它压根没找到文件」。版本层同理 —— 报告里多印一行「声称=v2 底本=v1」
    # 不算修好，必须证明「卡片写 `paper_id: 2303.11366v2` 时真的能定位到 v1 档」。
    # 故这里**临时把 PAPERS_DIR 指向自建目录**（结束即还原），让 glob 路径也可控。
    global PAPERS_DIR
    saved_papers_dir = PAPERS_DIR
    pool2 = pool / "vault"
    arch_dir = pool2 / "99-自检" / "p2s-self-0001"
    arch_dir.mkdir(parents=True, exist_ok=True)
    ft_idx = arch_dir / "fulltext.md"
    ft_idx.write_text(
        "<!-- 自动生成 by fetch_fulltext.py\n"
        "     arxiv_id : 2303.11366\n"
        "     paper_id : p2s-self-0001\n"
        "     source   : https://arxiv.org/html/2303.11366v1\n"
        "     fulltext : 是\n"
        "-->\n\n" + filler + "\n\n" + v1_sentence + "\n", encoding="utf-8")
    keys = _index_keys(ft_idx)
    key_ok = "2303.11366v1" in keys and "2303.11366" in keys and "p2s-self-0001" in keys
    card_6e = make_card("c6e.md", "2303.11366v2", plain)
    try:
        PAPERS_DIR = pool2
        found = find_fulltext("2303.11366v2")
        r6e = check_card(card_6e)          # 不给 index → 必须走 glob + 基号兜底
    finally:
        PAPERS_DIR = saved_papers_dir
    fallback_ok = found == ft_idx and r6e.get("version_verdict") == V_MISMATCH

    checks = [
        ("6a v1底本+正式版声明", r6a, V_MISMATCH),
        ("6b v1底本+未声明版本", r6b, V_UNCLAIMED),
        ("6c v1底本+声称v1", r6c, V_MATCH),
        ("6d 无版本标识+正式版", r6d, V_UNKNOWN),
        ("6f venue字段声称正式版", r6f, V_MISMATCH),
        ("6g venue=预印本→不得误报", r6g, V_UNCLAIMED),
        ("6h venue自称preprint→豁免", r6h, V_UNCLAIMED),
        ("6i 底本未声明≠不声称正式版", r6i, V_MISMATCH),
        ("6j 参考区自称preprint→豁免", r6j, V_UNCLAIMED),
        ("6k declared 版本一致→必须 MATCH（不是黄灯）", r6k, V_MATCH),
        ("6l declared 版本不一致→必须 MISMATCH", r6l, V_MISMATCH),
    ]
    lines = []
    all_ok = True
    for label, rep, expect in checks:
        got = rep.get("version_verdict")
        # ⚠️ 6a 额外锁定：**逐字层必须仍是 VERBATIM** ——
        # 证明版本错配是「逐字绿 + 版本黄」的组合，而不是靠把引文判伪来实现。
        extra = ""
        if label.startswith("6a"):
            synced = rep.get("verdict") == "VERBATIM"
            all_ok = all_ok and synced
            extra = f"（逐字层={rep.get('verdict')}，期望 VERBATIM）"
        flag = "✅" if got == expect else f"❌ 期望 {expect}"
        all_ok = all_ok and got == expect
        lines.append(f"{label}: {got} {flag}{extra}")
    # ⚠️ 6l 还要锁**消息**，不能只锁 verdict：这个 bug 的**用户可见症状**正是消息错
    # —— 把「版本号不同」写成「卡片声称**正式发表版**」。只断 verdict 的话，
    # 把 else 分支的 kind 守卫删掉（declared 又会被贴回「正式发表版」标签）依然全绿，
    # 而那正是修复前用户读到的那句话。
    note6l = r6l.get("version_note") or ""
    note_ok = ("版本号不同" in note6l) and ("正式发表版" not in note6l)
    lines.append(f"6l 消息不得把「版本号不同」写成「声称正式发表版」: "
                 f"{'✅' if note_ok else '❌'}")
    all_ok = all_ok and note_ok
    lines.append(f"6e 版本键进索引+基号兜底: keys={keys} "
                 f"兜底命中={'是' if found == ft_idx else '否'} "
                 f"卡(paper_id=v2)判定={r6e.get('version_verdict')} "
                 f"{'✅' if (key_ok and fallback_ok) else '❌'}")
    all_ok = all_ok and key_ok and fallback_ok
    return {"ok": all_ok, "detail": lines,
            "summary": " / ".join(f"{l.split(':')[0]}={c[1].get('version_verdict')}"
                                  for l, c in zip((x[0] for x in checks), checks))
                       + f" / 6e={'OK' if key_ok and fallback_ok else 'FAIL'}"}



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
    # 版本错配：**黄灯，单列，且绝不并入任何「红」或「通过」** ——
    # 「引错版本」与「引文伪造」是两回事，合并会让最弱的那一维被平均掉。
    n_vmis = [r for r in reports if r.get("version_verdict") == V_MISMATCH]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    elif not args.quiet:
        for r in reports:
            name = Path(r["card"]).name
            v = r["verdict"]
            icon = {"VERBATIM": "✅", "FUZZY": "🟡", "FABRICATED": "❌",
                    "NO_FULLTEXT": "⚪", "NO_QUOTES": "·"}.get(v, "?")
            # 版本错配**降级显示**：逐字层是绿的，但整行带黄标
            vmark = " 🟡版本错配" if r.get("version_verdict") == V_MISMATCH else ""
            extra = ""
            if "n_quotes" in r:
                parts = [f"{r['n_verbatim']}/{r['n_quotes']} 逐字"]
                if r["n_fuzzy"]:
                    parts.append(f"{r['n_fuzzy']} 近似")
                if r["n_fabricated"]:
                    parts.append(f"{r['n_fabricated']} 伪造")
                extra = "，".join(parts)
            print(f"{icon} {v:10s} {name}  {extra}{vmark}")
            if r.get("version_verdict") == V_MISMATCH:
                print(f"      🟡 声称={r.get('version_claim') or '(未声明)'} "
                      f"｜ 底本={r.get('fulltext_version')}"
                      f"（{r.get('fulltext_version_id')}）"
                      f"｜ 依据={r.get('version_claim_evidence')}")
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

        # --- 版本错配清单（黄灯，与伪造**分开报**）-------------------------
        print()
        if n_vmis:
            print(f"🟡 版本错配 {len(n_vmis)} 张（**黄灯，不等于引文伪造**，需人工确认）：")
            for r in n_vmis:
                kind = {"explicit": "显式版本号", "venue": "声称正式版",
                        "declared": "显式声明"}.get(r.get("version_claim_kind"), "?")
                print(f"   - {Path(r['card']).name}")
                print(f"       声称版本 : {r.get('version_claim')}"
                      f"   ｜ 底本版本 : {r.get('fulltext_version')}"
                      f"（{r.get('fulltext_version_id')}）   ｜ 依据 : {kind}")
            print("   ↑ 逐字命中只证明该句存在于**本底本**，不证明它出自卡片声称的版本。"
                  "\n     处置：补抓对应版本全文，或在卡片 ⑥ 段显式声明底本版本"
                  "（frontmatter `paper_version:`）。")
        else:
            print("🟡 版本错配 0 张")
        n_vunknown = sum(1 for r in reports
                         if r.get("version_verdict") == V_UNKNOWN)
        n_vun = sum(1 for r in reports
                    if r.get("version_verdict") == V_UNCLAIMED)
        print(f"   （另：{n_vunknown} 张底本无版本标识 → 无从比较；"
              f"{n_vun} 张卡片未声明版本 → 无版本结论可下。"
              f"**两者都不等于「版本一致」**）")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    # 退出码**不含**版本错配：黄灯不阻塞（口径见模块文档「口径纪律」第 1 条）
    return 1 if (n_fab_cards or n_noft) else 0


if __name__ == "__main__":
    sys.exit(main())
