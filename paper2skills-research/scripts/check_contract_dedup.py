#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_contract_dedup.py —— PHASE6 契约层「同质化禁令」机检（F8-S1）

拦什么
------
撰写规范 `paper2skills-research/data/contracts/flow-01-common.md` §2.1 的同质化禁令：

> 同一句话在本 FLOW 的 54 份里重复出现 ⇒ 视为未撰写。

也就是说：一份契约的「缺值处置」如果与另一份的措辞逐字重合，那**不是**两份都写了，
而是**两份都没写**。本脚本把这条禁令变成可复核的退出码与 stdout。

判据（每条都能单独失败，编号 D1…D5）
------------------------------------
  D1  五维表「缺值处置」第三列（A 模板 §2 / B 模板 §5）**跨契约、同维**重复：
      两格的**最长公共窗口 ≥ n**（默认 18 字）⇒ 红。
  D2  B 模板 §2 的 `→ 替代路径：` 行**跨契约**重复：窗口 ≥ n ⇒ 红。
  D3  正文长行**跨契约**重复：窗口 ≥ n_body（默认 30 字）⇒ 红。
  D4  覆盖率：契约份数 / 格子数低于下限，或一个格都没扫到 ⇒ **exit 2**（不是通过）。
  D5  仪器自证：每次运行注入一组已知重复的活体探针（正探针 + 反探针），
      探针没被抓到 ⇒ **exit 3**（是检测器坏了，不是语料干净）。

算法（确定性，可复核；**不用模糊/语义相似度**）
-----------------------------------------------
「n 字滑窗」：把每格文本归一化后枚举全部 n 字窗口，窗口相同即视为重复；
两格共享任一 n 字窗口 ⟺ 两格最长公共子串 ≥ n（充要）。
报告里的「重复文本」取该对契约的**最长公共窗口**，由左极大窗口向右贪心扩展得到 ——
纯字符串比较，同输入必同输出。

归一化（只做两件事，都可关）
--------------------------
  · 去掉全部 Unicode 空白（空格是排版，不是「字」）
  · NFC 归一化（全角/半角形式不会因为编码形式不同而漏配）
**不**剥 Markdown 强调符、**不**做同义词替换 —— 那会把「可复核」变成「可解释」。

退出码约定（照本仓库规矩）
--------------------------
  0 = 全过 ／ 1 = 判红 ／ 2 = 输入没拿到（**≠ 通过**）／ 3 = 内部错误（**≠ 判红**）

用法
----
  python3 paper2skills-research/scripts/check_contract_dedup.py
  python3 paper2skills-research/scripts/check_contract_dedup.py --n 24 --json-out /tmp/d.json
  python3 paper2skills-research/scripts/check_contract_dedup.py --selftest
  python3 paper2skills-research/scripts/check_contract_dedup.py --expect-contracts 139 --only D1,D2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

DEFAULT_N = 18          # D1 / D2 的窗口长度（依据见 reports/PHASE6-F8-S1-跨契约同质化机检.md）
DEFAULT_N_BODY = 30     # D3 的窗口长度（正文更长、噪声更多，故取更大值）
DEFAULT_MIN_CONTRACTS = 1

DIMS = ["① 获取路径", "② 粒度", "③ 回溯深度", "④ 新鲜度", "⑤ 口径归属"]
DIM_KEY = {d.replace(" ", ""): d for d in DIMS}

EXIT_PASS = 0
EXIT_RED = 1
EXIT_NO_INPUT = 2
EXIT_INTERNAL = 3

# 判据编号 -> 人话
CRITERIA = {
    "D1": "五维表「缺值处置」第三列跨契约同维重复",
    "D2": "B 模板 §2「→ 替代路径：」行跨契约重复",
    "D3": "正文长行跨契约重复",
    "D4": "覆盖率（份数/格数低于下限，或一个格都没扫到）",
    "D5": "仪器自证（活体探针没被抓到）",
}


# --------------------------------------------------------------------------
# 归一化
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_ZW_RE = re.compile("[\u200b-\u200f\u202a-\u202e\ufeff]")


def normalize(text: str) -> str:
    """归一化：NFKC + 去空白 + 去 Markdown 强调符。

    两处都是**被实测逼出来的**，不是洁癖：

    · **去 `*`（强调符）**：首版只做 NFC，于是「把 `替换` 写成 `**替换**`」就能把
      18 字窗口切成两段 —— 检测器看不见了。这不是改措辞，是改排版。
      实测：整改第一批撰写人交回来的 3 份就是这种「加粗式改写」。
      强调符是**表现**，不是**用词**，不剥它等于留了一条假绿通路。
    · **NFKC 而不是 NFC**：`，`↔`,`、`（`↔`(`、`％`↔`%` 是同一句话的两种排版，
      只做 NFC 时换一套标点即可绕过。NFKC 把两侧归一，比较仍然对称。

    仍然**不**做同义词替换、不做语义相似度 —— 那会把「可复核」变成「可解释」。
    """
    t = unicodedata.normalize("NFKC", text)
    t = _ZW_RE.sub("", t)
    t = t.replace("*", "")
    return _WS_RE.sub("", t)


# --------------------------------------------------------------------------
# 契约解析
# --------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)


def split_frontmatter(text: str):
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end():]


def split_md_row(line: str):
    """按 `|` 切 Markdown 表格行，但**反引号内的 `|` 不算分隔符**。"""
    s = line.strip()
    if not s.startswith("|"):
        return None
    parts, buf, in_code = [], [], False
    for ch in s:
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    if len(parts) < 2:
        return None
    return [p.strip() for p in parts[1:-1]]


def extract_dims_cells(body: str, contract: str):
    """抽出五维表的「缺值处置」第三列。返回 [(dim, raw_cell, lineno)]。"""
    out = []
    for lineno, line in enumerate(body.splitlines(), 1):
        if not line.strip().startswith("|"):
            continue
        cells = split_md_row(line)
        if not cells or len(cells) < 3:
            continue
        key = cells[0].replace(" ", "")
        dim = DIM_KEY.get(key)
        if dim:
            out.append((contract, dim, cells[2], lineno))
    return out


_ALT_RE = re.compile(r"^\s*→\s*替代路径[：:]\s*(.*)$")


def extract_alt_paths(body: str, contract: str):
    """抽出 B 模板 §2 的 `→ 替代路径：` 行。返回 [(contract, '§2 → 替代路径', raw, lineno)]。"""
    out = []
    for lineno, line in enumerate(body.splitlines(), 1):
        m = _ALT_RE.match(line)
        if m:
            out.append((contract, "§2 → 替代路径", m.group(1).strip(), lineno))
    return out


def extract_body_lines(body: str, contract: str):
    """抽出正文长行（D3）。排除：围栏代码、表格行、替代路径行、标题行、纯符号行。"""
    out = []
    in_fence = False
    for lineno, line in enumerate(body.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not stripped or stripped.startswith("|") or stripped.startswith("#"):
            continue
        if stripped.startswith(">"):
            continue
        if _ALT_RE.match(line):
            continue
        # 「- 」/「1. 」等列表标记剥掉，只比内容
        content = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
        if len(content) < 10:
            continue
        out.append((contract, "正文", content, lineno))
    return out


def parse_contracts(root: Path, only_d1d2: bool):
    """读契约，返回 (parsed, unparsed, cells, altpaths, bodylines)。"""
    files = sorted(list(root.glob("A/CTR-*.md")) + list(root.glob("B/CTR-*.md")))
    cells, altpaths, bodylines = [], [], []
    unparsed = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            unparsed.append((f.name, f"读不了：{exc}"))
            continue
        fm, body = split_frontmatter(text)
        if fm is None:
            unparsed.append((f.name, "没有 frontmatter"))
            continue
        name = f.name
        found = extract_dims_cells(body, name)
        if len(found) != 5:
            unparsed.append((f.name, f"五维表只抽到 {len(found)}/5 行"))
        cells.extend(found)
        altpaths.extend(extract_alt_paths(body, name))
        if not only_d1d2:
            bodylines.extend(extract_body_lines(body, name))
    return files, unparsed, cells, altpaths, bodylines


# --------------------------------------------------------------------------
# 材料专名遮罩（**不是豁免，是修正判据的适用范围**）
# --------------------------------------------------------------------------
# 为什么需要它：判据问的是「同一句**处置**被抄了几遍」，但语料里有一批字符串
# **必须逐字重复** —— 材料的产物名、协同协议术语、AGT/FLOW/STG 编号、决策号、
# 数据源名。规范明写「不要自己造词」，所以把它们改掉才是违规。
# 不遮罩时判据会把它们算成同质化（实测：`StageAcceptanceRecord`、开放事实 O1
# 那两句都曾被判成重复对），整改方向就会被带偏成「把材料术语改写掉」。
#
# 遮罩口径**故意收得很窄**：只遮标识符与专名，不遮散文。
# 判断一个词该不该进表，问一句「契约里换个说法写它，算不算造词？」——算，才进表。
#
# 三道防倒灌（对应仓库纪律「新增豁免必须比拦截条款测得更严」）：
#   ① 遮罩量是一等输出：遮掉多少字、占比多少，都打印；吃掉语料 > MASK_BUDGET 判红
#      —— 遮太多说明白名单本身变成了后门。
#   ② selftest 反例：**含**专名的样板句仍必须判红（样板句里就有 `AGT-045`）。
#   ③ selftest 反例：只差专名的两格仍必须判绿。
# `--no-mask` 可关掉遮罩看原始数；报告里两个数并列。

MASK_CHAR = "〓"
MASK_BUDGET = 0.35   # 遮掉的字符占比上限；超了判红（白名单吃掉了语料）

MASK_PATTERNS = [
    r"`[^`\n]{1,80}`",                       # 行内代码（字段名、状态名、算式片段）
    r"〈[^〉\n]{1,60}〉",                      # 材料里的数据源/产物名（〈库存台账〉）
    r"AGT-\d{3}",                            # 岗位号
    r"FLOW-0\d",                             # FLOW 号
    r"STG-0\d",                              # 阶段号
    r"DOM-0\d",
    r"PLN-[A-Z]+",
    r"CTR-[AB]-\d{3}",
    r"p2s-[a-z0-9\-]+",
    r"CASE-PROTOCOL-\d",
    r"\bQ(?:9|10|11|12)\b",                  # 决策号
    r"\bD-026\b",
    r"\bO[12]\b",                            # 开放事实号
    r"\bD4\b",
    # 材料的产物名与协同协议术语（规范要求逐字引用，不许改写）
    r"经营信号记录", r"经营Case Charter", r"FLOW-0\d Context Manifest",
    r"经营证据与偏差诊断", r"联合经营行动包", r"FLOW-0\d Assurance Decision",
    r"动作尝试或无动作记录", r"经营关闭记录", r"Case Charter",
    r"Stage Acceptance Record", r"Execution Attempt", r"Case/Event Ledger",
    r"StageAcceptanceRecord", r"Assurance", r"Action Intent", r"NoActionRecord",
    r"Pre-Case Exception", r"Emergency Guard", r"Case Control",
    r"Execution Broker", r"Case Agent", r"业务规则字典", r"标签分配结果表",
    r"指标契约", r"主数据",
]
# ⚠️ 实测缺陷（2026-09-13，批次 09/15 两位撰写人独立撞出）：
# `normalize()` **先去空白**，而 `MASK_RE` 是在归一化**之后**跑的 ——
# 于是表里带空格的专名（`Action Intent` / `Case Control` / `Emergency Guard` /
# `Execution Attempt` / `Case/Event Ledger` …）**一个都匹配不到**，等价未遮罩。
# 只有 `Stage Acceptance Record` 侥幸还能工作，因为表里另有无空格孪生
# `StageAcceptanceRecord`。修法：**把模式也按同一口径去空白**，两边同尺。
# 另加 D5 活体自证（见 `mask_patterns_alive`）—— 这条不许再静默失效。
MASK_RE = re.compile("|".join(pat.replace(" ", "") for pat in MASK_PATTERNS))

# 人写的材料专名样本：D5 每次运行都拿它们验「遮罩模式真的活着」
MASK_TERM_SAMPLES = [
    "Action Intent", "Case Control", "Execution Broker", "Case Agent",
    "Pre-Case Exception", "Emergency Guard", "Stage Acceptance Record",
    "Execution Attempt", "Case/Event Ledger", "经营Case Charter",
    "业务规则字典", "标签分配结果表",
]


def mask_patterns_alive():
    """返回**归一化后已经匹配不到**的专名列表（应为空）。

    死模式不会报错、只会静默放行 —— 正是本仓库最怕的那类缺陷
    （「判据只写单边」/「仪器看不见却报干净」）。故把它挂进 D5。
    """
    return [t for t in MASK_TERM_SAMPLES if not MASK_RE.search(normalize(t))]


def mask_proper_nouns(text: str):
    """把材料专名/标识符替换成单个占位符。返回 (masked_text, 被遮字符数)。"""
    counter = [0]

    def _rep(m):
        counter[0] += len(m.group(0))
        return MASK_CHAR

    return MASK_RE.sub(_rep, text), counter[0]


# --------------------------------------------------------------------------
# 核心：n 字滑窗重复检测
# --------------------------------------------------------------------------

def keep_masks(texts):
    """每个字符位是否属于「撰写人自己的用词」（False = 落在材料专名里）。

    判据是「同一句**处置**被抄了几遍」，量的应当是**处置用词**，
    不是材料的专名。所以材料专名占掉的字符位不计入 n。
    """
    masks = []
    for t in texts:
        keep = [True] * len(t)
        for m in MASK_RE.finditer(t):
            for k in range(m.start(), m.end()):
                keep[k] = False
        masks.append(keep)
    return masks


def _all_kept(keep, j, n):
    return all(keep[j:j + n])


def shared_windows(texts, n, masks=None):
    """返回 {window: [(item_index, offset), ...]}，只保留出现于 ≥2 个不同 item 的窗口。

    `masks` 给出时，只有**整窗都落在撰写人自己的用词上**的窗口才算数 ——
    两个格共享的若只是 `StageAcceptanceRecord` 这类材料专名，不算「同一句话」。
    """
    idx = {}
    for i, t in enumerate(texts):
        L = len(t)
        if L < n:
            continue
        keep = masks[i] if masks else None
        for j in range(L - n + 1):
            if keep is not None and not _all_kept(keep, j, n):
                continue
            w = t[j:j + n]
            bucket = idx.get(w)
            if bucket is None:
                idx[w] = bucket = []
            bucket.append((i, j))
    return {w: occ for w, occ in idx.items() if len({i for i, _ in occ}) >= 2}


def duplicate_pairs(texts, n, masks=None):
    """返回 {(i, j)}：共享任一合格 n 字窗口的 item 对。

    同维限定由调用方在分组后自己保证（把不同维度的格放进不同的 texts 列表）。
    """
    pairs = set()
    for occ in shared_windows(texts, n, masks).values():
        ids = sorted({i for i, _ in occ})
        for a in range(len(ids)):
            ia = ids[a]
            for b in range(a + 1, len(ids)):
                pairs.add((ia, ids[b]))
    return pairs


def pair_lcs(texts, n, pairs, masks=None):
    """对每一对，求最长公共窗口（由左极大窗口向右贪心扩展）。返回 {(i,j): (len, text)}。

    正确性：任何长度 ≥ n 的合格公共子串都落在某个合格 n 字共享窗口里；
    该子串的极大扩展必是「左极大」的（否则还能往左扩，与极大矛盾）。
    """
    best = {}
    for w, occ in shared_windows(texts, n, masks).items():
        per = {}
        for i, j in occ:
            per.setdefault(i, j)
        ids = sorted(per)
        for a in range(len(ids)):
            i = ids[a]
            oi = per[i]
            ti = texts[i]
            for b in range(a + 1, len(ids)):
                j = ids[b]
                oj = per[j]
                tj = texts[j]
                # 左极大判定：左边相同**且**左移一格后仍是合格窗口 ⇒ 本窗口冗余。
                # ⚠️ 少了后半句就会全漏：两份文本逐字相同时，最左那个合格窗口的
                # 左邻字**也**相同（是材料专名的一部分），只判前半句会把每一格都 skip 掉，
                # 于是「两份完全一样」反而报绿。首版就是这么错的，靠 selftest 用例 1 抓住。
                if (oi > 0 and oj > 0 and ti[oi - 1] == tj[oj - 1]
                        and (masks is None or _all_kept(masks[i], oi - 1, n))):
                    continue
                li, lj = oi, oj
                while li > 0 and lj > 0 and ti[li - 1] == tj[lj - 1]:
                    li -= 1
                    lj -= 1
                ri, rj = oi + n, oj + n
                while ri < len(ti) and rj < len(tj) and ti[ri] == tj[rj]:
                    ri += 1
                    rj += 1
                length = ri - li
                cur = best.get((i, j))
                if cur is None or length > cur[0]:
                    best[(i, j)] = (length, ti[li:ri])
    return best


# --------------------------------------------------------------------------
# 判据执行
# --------------------------------------------------------------------------

def run_dedup(items, n, criterion, location_of=None, min_len=None, mask=True):
    """items: [(contract, location, raw_text, lineno)]（已归一化前的原文）。

    返回 (hits, coverage)。hits 是 list[dict]。
    """
    texts = []
    total_chars = 0
    masked_chars = 0
    for it in items:
        raw = normalize(it[2])
        total_chars += len(raw)
        if mask:
            _, m = mask_proper_nouns(raw)
            masked_chars += m
        texts.append(raw)
    masks = keep_masks(texts) if mask else None
    groups = {}
    for k, it in enumerate(items):
        groups.setdefault(group_key(it), []).append(k)

    hits = []
    pair_total = 0
    for loc, idxs in sorted(groups.items()):
        sub = [texts[k] for k in idxs]
        if len(sub) < 2:
            continue
        msub = [masks[k] for k in idxs] if masks else None
        pairs = duplicate_pairs(sub, n, msub)
        pair_total += len(pairs)
        lcs = pair_lcs(sub, n, pairs, msub)
        for (a, b), (length, text) in sorted(lcs.items()):
            ia, ib = idxs[a], idxs[b]
            A, B = items[ia], items[ib]
            if A[0] == B[0]:
                continue  # 同一份契约内部不算「跨契约同质化」
            hits.append({
                "criterion": criterion,
                "dimension": loc,
                "contract_a": A[0],
                "contract_b": B[0],
                "location_a": f"{A[1]}（{contract_section(A)} 第 {A[3]} 行）",
                "location_b": f"{B[1]}（{contract_section(B)} 第 {B[3]} 行）",
                "text": text,
                "window_length": length,
                "threshold_n": n,
                "emitted_min_len": min_len,
                "masked": bool(mask),
            })
    hits.sort(key=lambda h: (-h["window_length"], h["contract_a"], h["contract_b"]))
    coverage = {
        "items": len(items),
        "nonempty_items": sum(1 for t in texts if t),
        "groups": len(groups),
        "pairs_over_threshold": pair_total,
        "masked_chars": masked_chars,
        "total_chars": total_chars,
        "mask_ratio": round(masked_chars / total_chars, 4) if total_chars else 0.0,
    }
    return hits, coverage


def group_key(item):
    """同维限定：只在**同一维**内做跨契约比较。

    变异样本（selftest 用）：把它换成常数 ⇒ 「跨维同句」用例必须打红。
    """
    return item[1]


def coverage_problems(coverage, min_contracts, unparsed, enabled, n_parsed, n_files):
    """D4 的判据本体，单独成函数以便 selftest 做变异。

    返回 list[str]：空 = 覆盖到位；非空 = 输入没拿到（exit 2，**不是通过**）。
    """
    problems = []
    if n_parsed == 0:
        problems.append("一份契约都没解析出来")
    if n_parsed < min_contracts:
        problems.append(f"解析到 {n_parsed} 份 < 下限 {min_contracts} 份")
    if unparsed:
        problems.append(f"{len(unparsed)} 份契约解析不完整：" +
                        "；".join(f"{f}（{w}）" for f, w in unparsed[:5]) +
                        ("…" if len(unparsed) > 5 else ""))
    for cid in ("D1", "D2", "D3"):
        if cid not in enabled:
            continue
        sub = coverage.get(cid)
        if sub is None:
            continue
        if sub["nonempty_items"] == 0:
            problems.append(f"{cid} 一个可比的条目都没扫到（0 条）—— 不许据此判「没有重复」")
    if "D1" in enabled and coverage.get("D1", {}).get("groups", 0) < 2:
        problems.append("D1 只扫到 0/1 个维分组，凑不出「跨契约」这一层")
    return problems


def contract_section(item):
    return "§5" if item[0].startswith("CTR-B") else "§2"


# --------------------------------------------------------------------------
# 仪器自证（D5）：每次运行都跑的活体探针
# --------------------------------------------------------------------------

PROBE_BODY = "本探针句仅用于证明检测器能看见重复，长度足够且不会出现在真实语料里"


def instrument_selfcheck(n):
    """每次运行都跑的活体对照。

    正探针：两段文本共享一条 ≥ n 字的句子 ⇒ **必须**报重复。
    反探针：两段文本共享恰好 n-1 字（第 n 个字起分叉）⇒ **必须**不报。
    两个探针各自独立调用，互不干扰 —— 第一版把反探针写成「共享 43 字前缀 + 尾部不同」，
    正探针/反探针共用一个语料表，反探针自己就命中了，整条 D5 变成假红。

    返回 (ok, message)。
    """
    try:
        probe = PROBE_BODY * 2
        pos = [normalize("甲" * 40 + probe), normalize("乙" * 40 + probe)]
        pm = keep_masks(pos)
        if (0, 1) not in duplicate_pairs(pos, n, pm):
            return False, f"正探针（共享 ≥{n} 字）没有被抓到 —— 检测器坏了，不是在报干净"
        # 反探针：共享**恰好** n-1 字，第 n 个字起两边各走各的（尾部也全不同），
        # 因此最长公共窗口 = n-1 < n ⇒ 必须不报。
        neg = [normalize("丁" * (n - 1) + "戊" * (n + 20)),
               normalize("丁" * (n - 1) + "己" * (n + 20))]
        nm = keep_masks(neg)
        if (0, 1) in duplicate_pairs(neg, n, nm):
            return False, f"反探针（共享恰好 {n - 1} 字）被误判为重复 —— 阈值失效"
        dead = mask_patterns_alive()
        if dead:
            return False, f"专名遮罩有 {len(dead)} 条死模式（归一化后匹配不到）：{dead}"
        return True, (f"正探针命中（共享 ≥{n} 字）· 反探针未命中（共享恰好 {n - 1} 字）"
                      f"· 专名遮罩 {len(MASK_TERM_SAMPLES)}/{len(MASK_TERM_SAMPLES)} 条活着")
    except Exception as exc:  # noqa: BLE001
        return False, f"探针执行时异常：{exc}"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def build_report(args):
    root = Path(args.root)
    if not root.is_dir():
        return None, f"契约目录不存在：{root}"

    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    only_d1d2 = only <= {"D1", "D2"}

    files, unparsed, cells, altpaths, bodylines = parse_contracts(root, only_d1d2)

    report = {
        "root": str(root),
        "only": sorted(only),
        "n": args.n,
        "n_body": args.n_body,
        "coverage": {
            "contract_files_found": len(files),
            "contract_files_parsed": len(files) - len(unparsed),
            "unparsed": [{"file": f, "why": w} for f, w in unparsed],
            "dim_cells": len(cells),
            "dim_cells_nonempty": sum(1 for c in cells if normalize(c[2])),
            "dim_cells_per_dim": {d: sum(1 for c in cells if c[1] == d) for d in DIMS},
            "alt_path_lines": len(altpaths),
            "body_lines": len(bodylines),
            "files_covered": sorted({c[0] for c in cells}),
        },
        "criteria": [],
        "duplicates": [],
    }

    # ---- D5 仪器自证（先跑：看不见重复的仪器，说什么都不算） ----
    probe_ok, probe_msg = instrument_selfcheck(args.n)
    report["criteria"].append({
        "id": "D5", "name": CRITERIA["D5"],
        "status": "green" if probe_ok else "internal_error",
        "detail": probe_msg,
    })
    if not probe_ok:
        return report, None

    # ---- D4 覆盖率 ----
    n_parsed = report["coverage"]["contract_files_parsed"]

    # D1/D2/D3 的覆盖面必须先算出来，D4 才有东西可判（否则「0 条」会被当成「干净」）
    hits_by = {}
    use_mask = not getattr(args, "no_mask", False)
    if "D1" in only:
        hits_by["D1"] = run_dedup(cells, args.n, "D1", mask=use_mask)
        report["coverage"]["D1"] = hits_by["D1"][1]
    if "D2" in only:
        hits_by["D2"] = run_dedup(altpaths, args.n, "D2", mask=use_mask)
        report["coverage"]["D2"] = hits_by["D2"][1]
    if "D3" in only:
        hits_by["D3"] = run_dedup(bodylines, args.n_body, "D3", mask=use_mask)
        report["coverage"]["D3"] = hits_by["D3"][1]

    cov_red = coverage_problems(report["coverage"], args.min_contracts, unparsed,
                                only, n_parsed, len(files))
    # 防倒灌①：遮罩吃掉语料的比例超上限 ⇒ 判红（白名单变成了后门）
    for cid in ("D1", "D2", "D3"):
        sub = report["coverage"].get(cid)
        if sub and sub["mask_ratio"] > MASK_BUDGET:
            cov_red.append(f"{cid} 材料专名遮罩吃掉 {sub['mask_ratio']:.1%} 的字符 "
                           f"> 上限 {MASK_BUDGET:.0%} —— 白名单过宽，判据已被架空")
    report["criteria"].append({
        "id": "D4", "name": CRITERIA["D4"],
        "status": "red" if cov_red else "green",
        "detail": ("；".join(cov_red) if cov_red else
                   f"契约 {n_parsed}/{len(files)} 份、缺值处置格 "
                   f"{report['coverage']['dim_cells_nonempty']}/{len(cells)} 个、"
                   f"替代路径 {len(altpaths)} 行、正文长行 {len(bodylines)} 条"),
    })

    # ---- D1 / D2 / D3 的判据结论 ----
    for cid, label in (("D1", f"（扫 {report['coverage'].get('D1', {}).get('nonempty_items', 0)} "
                              f"个非空格 / {report['coverage'].get('D1', {}).get('groups', 0)} 维，n={args.n}）"),
                       ("D2", f"（扫 {report['coverage'].get('D2', {}).get('nonempty_items', 0)} "
                              f"条替代路径行，n={args.n}）"),
                       ("D3", f"（扫 {report['coverage'].get('D3', {}).get('nonempty_items', 0)} "
                              f"条正文长行，n_body={args.n_body}）")):
        if cid not in hits_by:
            continue
        hits, cov = hits_by[cid]
        report["criteria"].append({
            "id": cid, "name": CRITERIA[cid],
            "status": "red" if hits else "green",
            "detail": f"跨契约重复对 {cov['pairs_over_threshold']} 对 {label}",
        })
        report["duplicates"].extend(hits)

    return report, None


def render(report, stream=sys.stdout):
    w = stream.write
    w("=" * 78 + "\n")
    w("契约层「同质化禁令」机检 · check_contract_dedup.py\n")
    w("=" * 78 + "\n")
    cov = report["coverage"]
    w("【覆盖率（一等输出）】\n")
    w(f"  契约文件      : {cov['contract_files_parsed']}/{cov['contract_files_found']} 份解析成功")
    if cov["unparsed"]:
        for u in cov["unparsed"]:
            w(f"      ⚠ {u['file']}：{u['why']}")
    w(f"  缺值处置格    : {cov['dim_cells_nonempty']}/{cov['dim_cells']} 个非空")
    w(f"      " + " · ".join(f"{d} {n}" for d, n in cov["dim_cells_per_dim"].items()))
    w(f"  替代路径行    : {cov['alt_path_lines']} 行")
    w(f"  正文长行      : {cov['body_lines']} 条")
    for key in ("D1", "D2", "D3"):
        c = cov.get(key)
        if c:
            w(f"  {key} 遮罩        : 遮掉 {c['masked_chars']}/{c['total_chars']} 字 "
              f"（{c['mask_ratio']:.1%}，上限 {MASK_BUDGET:.0%}）")
    for key in ("D1", "D2", "D3"):
        c = cov.get(key)
        if c:
            w(f"  {key} 扫到窗口  : {c['items']} 个 item / {c['groups']} 组 / "
              f"超阈值对 {c['pairs_over_threshold']}")
    w("\n【判据】\n")
    for c in report["criteria"]:
        mark = {"green": "✅", "red": "🔴", "internal_error": "💥"}[c["status"]]
        w(f"  {mark} {c['id']} {c['name']}\n      {c['detail']}\n")
    w(f"\n【重复清单】共 {len(report['duplicates'])} 条跨契约重复\n")
    for h in report["duplicates"][:40]:
        w(f"  [{h['criterion']}] {h['window_length']:>3} 字 · {h['contract_a']} ↔ {h['contract_b']}\n"
          f"        {h['text'][:110]!r}\n")
    if len(report["duplicates"]) > 40:
        w(f"  …（另有 {len(report['duplicates']) - 40} 条，见 --json-out）\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="契约层「同质化禁令」机检")
    ap.add_argument("--root", default="paper2skills-vault/07-资源库/contracts")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"D1/D2 的 n 字滑窗长度（默认 {DEFAULT_N}）")
    ap.add_argument("--n-body", type=int, default=DEFAULT_N_BODY,
                    help=f"D3 的 n 字滑窗长度（默认 {DEFAULT_N_BODY}）")
    ap.add_argument("--only", default="D1,D2",
                    help="只跑这些判据，逗号分隔（默认 D1,D2；D3 见 --only D1,D2,D3）")
    ap.add_argument("--focus", default=None,
                    help="只**报告**涉及这些契约的重复（检测仍是全语料，不缩小比较范围）；"
                         "逗号分隔，可写文件名前缀如 CTR-A-003")
    ap.add_argument("--no-mask", action="store_true",
                    help="关掉材料专名遮罩，看原始数（不推荐；专名重复是规范要求的）")
    ap.add_argument("--min-contracts", type=int, default=DEFAULT_MIN_CONTRACTS,
                    help="D4 覆盖率下限：解析到的契约份数不低于此值，否则 exit 2")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest()

    try:
        report, err = build_report(args)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"💥 检测器内部错误：{type(exc).__name__}: {exc}\n")
        return EXIT_INTERNAL

    if err:
        sys.stderr.write(f"⛔ 输入没拿到：{err}\n（exit 2 ≠ 通过 —— 没扫到东西不等于东西没问题）\n")
        return EXIT_NO_INPUT

    if not args.quiet:
        render(report)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"机读清单已写出：{args.json_out}")

    if args.focus:
        keys = [k.strip() for k in args.focus.split(",") if k.strip()]
        report["duplicates"] = [
            h for h in report["duplicates"]
            if any(k in h["contract_a"] or k in h["contract_b"] for k in keys)]
        if not args.quiet:
            print(f"（--focus {args.focus}：只报告涉及这些契约的 {len(report['duplicates'])} 条；"
                  f"**判据结论仍是全语料的**）")

    by_id = {c["id"]: c for c in report["criteria"]}
    if by_id["D5"]["status"] == "internal_error":
        return EXIT_INTERNAL
    if by_id["D4"]["status"] == "red":
        return EXIT_NO_INPUT
    if any(by_id[k]["status"] == "red" for k in by_id if k in ("D1", "D2", "D3")):
        return EXIT_RED
    return EXIT_PASS


# --------------------------------------------------------------------------
# --selftest
# --------------------------------------------------------------------------

FM = """---
contract_id: {cid}
template: {tpl}
responsibility: 测试责任{tag}
role_id: AGT-0{num}
flows: [FLOW-01]
status: 可写
---
"""


def _mk(root: Path, section: str, cid: str, tpl: str, dims, alt=None, body=None):
    """造一份最小契约。dims: {维: 处置文本}；alt: [替代路径文本]；body: [正文行]。"""
    tag = cid[-1]
    text = FM.format(cid=cid, tpl=tpl, tag=tag, num=tag)
    text += f"\n# {section}\n\n"
    if tpl == "A":
        text += "## 2 数据要求（五维 · 三列）\n\n"
        text += "| 维 | 取值 | 缺值处置 |\n|---|---|---|\n"
        for d in DIMS:
            v = dims.get(d, ("**不可得**" if d == DIMS[0] else "非缺口"),)
            if isinstance(v, tuple):
                v = v[0]
            text += f"| {d} | {v} | {dims.get(d + '#处置', '')} |\n"
    else:
        text += "## 2 不可达部分\n\n"
        for a in (alt or []):
            text += f"- 不可达：这一段算法做不到\n  → 替代路径：{a}\n"
        text += "\n## 5 数据要求（五维 · 三列）\n\n"
        text += "| 维 | 取值 | 缺值处置 |\n|---|---|---|\n"
        for d in DIMS:
            text += f"| {d} | 非缺口 | {dims.get(d + '#处置', '')} |\n"
    if body:
        text += "\n## 6 适用 FLOW\n\n"
        for b in body:
            text += f"- {b}\n"
    (root / cid).mkdir(parents=True, exist_ok=True)
    p = root / cid / f"{cid}-{section}.md"
    p.write_text(text, encoding="utf-8")
    return p


def _naive_split_md_row(line: str):
    """变异样本用的坏切列器：按 `|` 直接切，反引号里的 `|` 也当分隔符。"""
    s = line.strip()
    if not s.startswith("|"):
        return None
    parts = s.split("|")
    return [p.strip() for p in parts[1:-1]]


def _corpus(tmp: Path, kind: str):
    """按名字造出测试语料目录，返回 root。"""
    root = tmp / kind
    (root / "A").mkdir(parents=True, exist_ok=True)
    (root / "B").mkdir(parents=True, exist_ok=True)
    return root


def _write(root: Path, sub: str, name: str, text: str):
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")


def _contract_a(cid: str, dims: dict, body=None) -> str:
    t = FM.format(cid=cid, tpl="A", tag=cid[-1], num=cid[-1])
    t += "\n## 2 数据要求（五维 · 三列）\n\n| 维 | 取值 | 缺值处置 |\n|---|---|---|\n"
    for d in DIMS:
        t += f"| {d} | 非缺口 | {dims.get(d, '')} |\n"
    if body:
        t += "\n## 6 适用 FLOW\n\n" + "".join(f"- {b}\n" for b in body)
    return t


def _contract_b(cid: str, dims: dict, alts=None, body=None) -> str:
    t = FM.format(cid=cid, tpl="B", tag=cid[-1], num=cid[-1])
    t += "\n## 2 不可达部分\n\n"
    for a in (alts or []):
        t += f"- 不可达：这一段算法到此为止\n  → 替代路径：{a}\n"
    t += "\n## 5 数据要求（五维 · 三列）\n\n| 维 | 取值 | 缺值处置 |\n|---|---|---|\n"
    for d in DIMS:
        t += f"| {d} | 非缺口 | {dims.get(d, '')} |\n"
    if body:
        t += "\n## 6 适用 FLOW\n\n" + "".join(f"- {b}\n" for b in body)
    return t


def run_selftest(argv=None):
    import shutil
    import tempfile

    BOILER = ("先按〈平台后台〉字段清单暂按口径并逐条标注「口径未签发」；"
              "契约签发后替换，替换前该口径只用于排序与内部判断")
    GOOD_1 = ("缺了「退货原因码到责任工位」的映射就算不出分诊准确率："
              "先按客服工单的自由文本关键词人工分诊一周的样本顶替，"
              "AGT-045 签发映射表后替换；替换前只出趋势不出责任归属")
    GOOD_2 = ("缺了广告后台的「归因窗」定义就算不出可比的 ROAS："
              "先按各站点后台默认归因窗（Amazon 7 日／独立站末次点击）顶替并标注口径不一致，"
              "AGT-045 签发统一归因窗后替换；替换前不做跨站点 ROAS 排名")

    tmp = Path(tempfile.mkdtemp(prefix="p2s-dedup-selftest-"))
    cases = []

    def case(name, builder, expect_exit, expect_red=None, expect_green=None, only="D1,D2,D3",
             n=None, n_body=None, min_contracts=2, desc="", sabotage_probe=False,
             no_mask=False):
        cases.append(dict(name=name, builder=builder, expect_exit=expect_exit,
                          expect_red=expect_red or [], expect_green=expect_green or [],
                          only=only, n=n, n_body=n_body,
                          min_contracts=min_contracts, desc=desc,
                          sabotage_probe=sabotage_probe, no_mask=no_mask))

    # ---- D1 正例：两格逐字相同 ----
    def b_d1_pos(root):
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d))
    case("D1 正例：两契约⑤处置逐字相同", b_d1_pos, EXIT_RED, expect_red=["D1"],
         only="D1", desc="n=18 窗口完全相同 ⇒ 必须判红")

    # ---- D1 反向控制：两格措辞不同但都合格 ----
    def b_d1_neg(root):
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", {DIMS[4]: GOOD_1}))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", {DIMS[4]: GOOD_2}))
    case("D1 反向控制：两份措辞不同且都合格", b_d1_neg, EXIT_PASS, expect_green=["D1"],
         only="D1",
         desc="内容都是「点名自己会缺的那维 + 先用什么 + 何时换」，无 ≥18 字重合 ⇒ 必须判绿")

    # ---- D1 变异样本：只改一个字，公共窗口降到 n-1 ----
    def b_d1_mut(root):
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        # 把样板句每 12 字插一个「改」⇒ 最长公共窗口 = 12 < 18
        broken = "改".join(BOILER[i:i + 12] for i in range(0, len(BOILER), 12))
        d2 = {x: "" for x in DIMS}
        d2[DIMS[4]] = broken
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d2))
    case("D1 变异：把样板句每 12 字打断一次 ⇒ 公共窗口 < 18", b_d1_mut, EXIT_PASS,
         expect_green=["D1"], only="D1",
         desc="最长公共窗口掉到 12 ⇒ 必须判绿（证明阈值真的在起作用）")
    cases[-1]["mutates"] = "D1 正例"

    # ---- D1 边界：恰好 n 字相同 ⇒ 红 ----
    def b_d1_edge_red(root):
        tail_a = "甲" * 12
        tail_b = "乙" * 12
        common = "共" * 18
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", {DIMS[4]: common + tail_a}))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", {DIMS[4]: common + tail_b}))
    case("D1 边界：恰好 18 字相同", b_d1_edge_red, EXIT_RED, expect_red=["D1"], only="D1",
         desc="公共窗口 = 18 = n ⇒ 判红（阈值含端点）")

    def b_d1_edge_green(root):
        common = "共" * 17
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", {DIMS[4]: common + "甲" * 12}))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", {DIMS[4]: common + "乙" * 12}))
    case("D1 边界：17 字相同（差一个字）", b_d1_edge_green, EXIT_PASS, expect_green=["D1"],
         only="D1",
         desc="公共窗口 = 17 < n ⇒ 判绿（与上一条配对，锁定阈值端点）")
    cases[-1]["mutates"] = "D1 边界：恰好 18 字相同"

    # ---- D1 同维限定：跨维相同不算 ----
    def b_d1_crossdim(root):
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        d2 = {x: "" for x in DIMS}
        d2[DIMS[3]] = BOILER
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d2))
    case("D1 同维限定：同句落在不同维不报", b_d1_crossdim, EXIT_PASS, expect_green=["D1"],
         only="D1", desc="D1 只在同一维内比较（⑤ 与 ⑤ 比），跨维重合不是本判据的对象")

    # ---- D1 空白归一 ----
    def b_d1_ws(root):
        """两份文本只差空格，且空格**每 8 字插一个** —— 故意插得比真实排版密。

        这样「不归一化」时会话的最长公共窗口只有 8 < 18，用例才真的能打红；
        若只按真实排版插一两个空格，最长公共窗口本来就 ≥18，
        这条用例会在「归一化被关掉」的变异下照样通过 —— 那它就是个摆设。
        """
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        spaced = "".join(ch + (" " if i % 8 == 7 else "") for i, ch in enumerate(BOILER))
        d2 = {x: "" for x in DIMS}
        d2[DIMS[4]] = spaced
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d2))
    case("D1 空白归一：每 8 字插一个空格仍判重复", b_d1_ws, EXIT_RED, expect_red=["D1"],
         only="D1", desc="空格是排版不是「字」⇒ 归一化后仍应抓到")

    # ---- D2 正例 / 反向控制 / 变异 ----
    # 注：这些用例只测 D2，但 D1 也开着 ⇒ 五维表必须填上互不相同的合格内容，
    #     否则 D4 会因为「D1 扫到 0 条」而 exit 2（那是对的，但会遮住本用例要测的东西）。
    def b_d2_pos(root):
        alt = ("由 AGT-014 在进入 STG-04 前提供验厂结论原件；"
               "拿不到时先用供应商自填产能表顶替并只作排序参考，不得进候选集")
        _write(root, "B", "CTR-B-101-甲.md", _contract_b("CTR-B-101", {DIMS[4]: GOOD_1}, [alt]))
        _write(root, "B", "CTR-B-102-乙.md", _contract_b("CTR-B-102", {DIMS[4]: GOOD_2}, [alt]))
    case("D2 正例：两条替代路径逐字相同", b_d2_pos, EXIT_RED, expect_red=["D2"], only="D2")

    def b_d2_neg(root):
        a1 = ("由 AGT-014 在进入 STG-04 前提供验厂结论原件；"
              "拿不到时先用供应商自填产能表顶替并只作排序参考，不得进候选集")
        a2 = ("由 AGT-043 在合同签署前核验资质原件有效期；"
              "缺失时该供应商整条不进候选集，不做任何降级顶替")
        _write(root, "B", "CTR-B-101-甲.md", _contract_b("CTR-B-101", {DIMS[4]: GOOD_1}, [a1]))
        _write(root, "B", "CTR-B-102-乙.md", _contract_b("CTR-B-102", {DIMS[4]: GOOD_2}, [a2]))
    case("D2 反向控制：两条替代路径各写各的责任岗位", b_d2_neg, EXIT_PASS,
         expect_green=["D2"], only="D2")

    def b_d2_mut(root):
        a1 = ("由 AGT-014 在进入 STG-04 前提供验厂结论原件；"
              "拿不到时先用供应商自填产能表顶替并只作排序参考，不得进候选集")
        a2 = "改".join(a1[i:i + 12] for i in range(0, len(a1), 12))
        _write(root, "B", "CTR-B-101-甲.md", _contract_b("CTR-B-101", {DIMS[4]: GOOD_1}, [a1]))
        _write(root, "B", "CTR-B-102-乙.md", _contract_b("CTR-B-102", {DIMS[4]: GOOD_2}, [a2]))
    case("D2 变异：把替代路径每 12 字打断一次", b_d2_mut, EXIT_PASS, expect_green=["D2"],
         only="D2")
    cases[-1]["mutates"] = "D2 正例"

    # ---- D3 正例 / 反向控制 / 变异 ----
    def b_d3_pos(root):
        sent = ("这条正文长句在多个契约里逐字重复出现，用来证明正文扫描器确实能看见重复，"
                "因此它的长度必须超过正文阈值")
        _write(root, "A", "CTR-A-101-甲.md",
               _contract_a("CTR-A-101", {DIMS[4]: GOOD_1}, body=[sent]))
        _write(root, "A", "CTR-A-102-乙.md",
               _contract_a("CTR-A-102", {DIMS[4]: GOOD_2}, body=[sent]))
    case("D3 正例：正文长句逐字相同", b_d3_pos, EXIT_RED, expect_red=["D3"], only="D3",
         n_body=30)

    def b_d3_neg(root):
        s1 = ("这条正文长句写的是甲责任自己的判据，与乙责任那句没有一处相同，"
              "用来做正文扫描的反向控制")
        s2 = ("这一句写的是完全另一码事：冻结条件、时限与证据清单都不同，"
              "措辞也不重合")
        _write(root, "A", "CTR-A-101-甲.md",
               _contract_a("CTR-A-101", {DIMS[4]: GOOD_1}, body=[s1]))
        _write(root, "A", "CTR-A-102-乙.md",
               _contract_a("CTR-A-102", {DIMS[4]: GOOD_2}, body=[s2]))
    case("D3 反向控制：正文两句各写各的", b_d3_neg, EXIT_PASS, expect_green=["D3"],
         only="D3", n_body=30)

    def b_d3_mut(root):
        sent = ("这条正文长句在多个契约里逐字重复出现，用来证明正文扫描器确实能看见重复，"
                "因此它的长度必须超过正文阈值")
        broken = "改".join(sent[i:i + 20] for i in range(0, len(sent), 20))
        _write(root, "A", "CTR-A-101-甲.md",
               _contract_a("CTR-A-101", {DIMS[4]: GOOD_1}, body=[sent]))
        _write(root, "A", "CTR-A-102-乙.md",
               _contract_a("CTR-A-102", {DIMS[4]: GOOD_2}, body=[broken]))
    case("D3 变异：把正文长句每 20 字打断一次", b_d3_mut, EXIT_PASS, expect_green=["D3"],
         only="D3", n_body=30)
    cases[-1]["mutates"] = "D3 正例"

    # ---- D4 覆盖率：空输入 ⇒ exit 2（不是通过） ----
    def b_empty(root):
        (root / "A").mkdir(parents=True, exist_ok=True)
        (root / "B").mkdir(parents=True, exist_ok=True)
    case("D4 空输入：一份契约都没有 ⇒ exit 2", b_empty, EXIT_NO_INPUT, expect_red=["D4"],
         desc="「没东西可查」不等于「查过了没问题」")

    # ---- D4 覆盖率：只有 1 份（无法构成「跨契约」） ⇒ exit 2 ----
    def b_single(root):
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = GOOD_1
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
    case("D4 单份契约：凑不出跨契约对 ⇒ exit 2", b_single, EXIT_NO_INPUT, expect_red=["D4"],
         min_contracts=2, desc="「一个格都没扫到对」不许报全部通过")

    # ---- D4 覆盖率：五维表全空 ⇒ exit 2 ----
    def b_allblank(root):
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", {}))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", {}))
    case("D4 五维表全空：一个非空格都没扫到 ⇒ exit 2", b_allblank, EXIT_NO_INPUT,
         expect_red=["D4"], desc="空表 + 「没有重复」是一句废话，必须 exit 2")

    # ---- D4 覆盖率下限 ----
    def b_below_min(root):
        d = {x: "" for x in DIMS}
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d))
    case("D4 覆盖率下限：2 份 < --min-contracts 5 ⇒ exit 2", b_below_min, EXIT_NO_INPUT,
         expect_red=["D4"], min_contracts=5)

    # ---- D4 解析失败：半份契约也算覆盖不到 ----
    def b_unparsed(root):
        d = {x: "" for x in DIMS}
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d))
        _write(root, "A", "CTR-A-103-丙.md", "# 没有 frontmatter 的残件\n")
    case("D4 解析不全：有一份抽不到五维表 ⇒ exit 2", b_unparsed, EXIT_NO_INPUT,
         expect_red=["D4"], desc="覆盖不到的东西不许算进「全过」")

    # ---- 表格行里的 `|` 不切错 ----
    def b_pipe_in_code(root):
        """A 的表第二列含反引号包裹的 `|`，B 的没有；两格的**处置列**逐字相同。

        朴素按 `|` 切列时 A 会多切出一列 ⇒ 取到的「处置」不是处置，D1 就看不见重复。
        """
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        t = FM.format(cid="CTR-A-101", tpl="A", tag="1", num="1")
        t += "\n## 2 数据要求（五维 · 三列）\n\n| 维 | 取值 | 缺值处置 |\n|---|---|---|\n"
        for x in DIMS:
            val = "自有埋点（口径 `a|b` 两套）" if x == DIMS[4] else "非缺口"
            t += f"| {x} | {val} | {d.get(x, '')} |\n"
        _write(root, "A", "CTR-A-101-甲.md", t)
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d))
    case("解析：单元格内反引号里的 | 不被当分隔符", b_pipe_in_code, EXIT_RED,
         expect_red=["D1"], only="D1", desc="切错列会把「取值」当「处置」比，判据就废了")

    # ---- 全绿基线：两份都合格 ----
    def b_all_green(root):
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", {DIMS[4]: GOOD_1}))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", {DIMS[4]: GOOD_2}))
        a1 = "由 AGT-014 在进入 STG-04 前提供验厂结论原件；拿不到时先用自填产能表顶替并只作排序参考"
        a2 = "由 AGT-043 在合同签署前核验资质原件有效期；缺失时该供应商整条不进候选集"
        _write(root, "B", "CTR-B-101-甲.md", _contract_b("CTR-B-101", {}, [a1]))
        _write(root, "B", "CTR-B-102-乙.md", _contract_b("CTR-B-102", {}, [a2]))
    case("基线：D1/D2/D3 全绿 ⇒ exit 0", b_all_green, EXIT_PASS,
         expect_green=["D1", "D2", "D3", "D4", "D5"])

    # ---- D4：某个判据一条都没扫到 ⇒ exit 2 ----
    def b_d2_zero_scan(root):
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", {DIMS[4]: GOOD_1}))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", {DIMS[4]: GOOD_2}))
    case("D4 判据级零覆盖：开着 D2 却一条替代路径都没有 ⇒ exit 2", b_d2_zero_scan,
         EXIT_NO_INPUT, expect_red=["D4"], only="D2",
         desc="「扫 0 条然后说没有重复」正是禁令要拦的那种假绿")
    cases[-1]["mutates"] = "基线：D1/D2/D3 全绿"

    # ---- D5 失败通路：仪器自证失败必须 exit 3（≠ 判红） ----
    def b_noop(root):
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", {DIMS[4]: GOOD_1}))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", {DIMS[4]: GOOD_2}))
    case("D5 失败通路：探针报错 ⇒ exit 3（≠ 判红）", b_noop, EXIT_INTERNAL,
         expect_red=["D5"], sabotage_probe=True,
         desc="检测器坏了与语料判红是两个退出码，不许糊成一个")

    def b_emphasis(root):
        """两份只差 Markdown 强调符（`**`），且强调符插得比真实排版密。

        没有这条时「把 `替换` 写成 `**替换**`」就能把 18 字窗口切成两段 ——
        实测整改第一批就有人这么交回来。加粗是排版，不是措辞。
        """
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        bolded = "".join(ch + ("**" if i % 8 == 7 else "") for i, ch in enumerate(BOILER))
        d2 = {x: "" for x in DIMS}
        d2[DIMS[4]] = bolded
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d2))
    case("归一：只差加粗标记（每 8 字插一个 **）仍判重复", b_emphasis, EXIT_RED,
         expect_red=["D1"], only="D1",
         desc="不剥强调符 ⇒ 加粗即绕过 ⇒ 假绿。这条用例就钉住它")
    cases[-1]["mutates"] = "D1 正例"

    def b_fullwidth(root):
        """两份只差全角/半角标点 ⇒ NFKC 后仍应判重复。"""
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        fw = (BOILER.replace("；", ";").replace("，", ",")
                    .replace("（", "(").replace("）", ")"))
        d2 = {x: "" for x in DIMS}
        d2[DIMS[4]] = fw
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d2))
    case("归一：只差全角/半角标点仍判重复", b_fullwidth, EXIT_RED, expect_red=["D1"],
         only="D1", desc="换一套标点不是换一句话 ⇒ NFKC 归一后必须仍抓到")
    cases[-1]["mutates"] = "D1 正例"

    # ---- 材料专名遮罩的三道防倒灌（新增豁免必须比拦截条款测得更严） ----
    def b_mask_boilerplate(root):
        """样板句里**含**专名（〈平台后台〉/`AGT-045`），遮罩不许因此把它放行。"""
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = BOILER
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d))
    case("遮罩①：含专名的样板句仍须判红", b_mask_boilerplate, EXIT_RED,
         expect_red=["D1"], only="D1",
         desc="样板句里就有 〈平台后台〉 与专名；遮罩若把它整句放行，判据就废了")

    def b_mask_only_nouns(root):
        """两格只共享一个材料专名（21 字），不是「同一句话」⇒ 判绿。"""
        # 两侧各垫足正文，让遮罩占比落在预算内（否则遮罩③ 的预算闸会先开火）
        d1 = {x: "" for x in DIMS}
        d1[DIMS[4]] = "甲" * 40 + "StageAcceptanceRecord" + "甲" * 40
        d2 = {x: "" for x in DIMS}
        d2[DIMS[4]] = "乙" * 40 + "StageAcceptanceRecord" + "乙" * 40
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d1))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d2))
    case("遮罩②：只共享材料专名 21 字 ⇒ 判绿", b_mask_only_nouns, EXIT_PASS,
         expect_green=["D1"], only="D1",
         desc="规范要求逐字引用材料术语；把它们判成同质化会把整改带偏成「改写术语」")

    def b_mask_budget(root):
        """两格几乎全是专名 ⇒ 遮罩吃掉语料 > 35% ⇒ 白名单过宽，必须判红。"""
        blob = "".join(f"`字段{k:02d}名称占位`" for k in range(12))
        d = {x: "" for x in DIMS}
        d[DIMS[4]] = blob
        _write(root, "A", "CTR-A-101-甲.md", _contract_a("CTR-A-101", d))
        _write(root, "A", "CTR-A-102-乙.md", _contract_a("CTR-A-102", d))
    case("遮罩③：遮罩吃掉语料超上限 ⇒ exit 2", b_mask_budget, EXIT_NO_INPUT,
         expect_red=["D4"], only="D1",
         desc="白名单一旦能吃掉大半语料，它就不再是修正适用范围，而是后门")

    # ---- 运行 ----
    print("=" * 78)
    print("check_contract_dedup.py --selftest")
    print("=" * 78)
    ok_count = 0
    fails = []

    def run_case(c, root):
        ns = argparse.Namespace(
            root=str(root), n=c["n"] or DEFAULT_N, n_body=c["n_body"] or DEFAULT_N_BODY,
            only=c["only"], min_contracts=c["min_contracts"], json_out=None,
            quiet=True, selftest=False, no_mask=c.get("no_mask", False))
        orig_probe = None
        if c.get("sabotage_probe"):
            orig_probe = globals()["instrument_selfcheck"]
            globals()["instrument_selfcheck"] = lambda n: (False, "（selftest 故意打坏的探针）")
        try:
            report, err = build_report(ns)
        finally:
            if orig_probe is not None:
                globals()["instrument_selfcheck"] = orig_probe
        if report is None:
            return EXIT_NO_INPUT, [], []
        by = {x["id"]: x for x in report["criteria"]}
        if by.get("D5", {}).get("status") == "internal_error":
            got_exit = EXIT_INTERNAL
        elif by.get("D4", {}).get("status") == "red":
            got_exit = EXIT_NO_INPUT
        elif any(by[k]["status"] == "red" for k in ("D1", "D2", "D3") if k in by):
            got_exit = EXIT_RED
        else:
            got_exit = EXIT_PASS
        got_red = [k for k in ("D1", "D2", "D3", "D4", "D5")
                   if by.get(k, {}).get("status") in ("red", "internal_error")]
        got_green = [k for k in ("D1", "D2", "D3", "D4", "D5")
                     if by.get(k, {}).get("status") == "green"]
        return got_exit, got_red, got_green

    def judge(c, got_exit, got_red, got_green):
        problems = []
        if got_exit != c["expect_exit"]:
            problems.append(f"退出码 {got_exit} ≠ 期望 {c['expect_exit']}")
        for k in c["expect_red"]:
            if k not in got_red:
                problems.append(f"{k} 应判红却是绿的")
        for k in c["expect_green"]:
            if k not in got_green:
                problems.append(f"{k} 应判绿却是红的")
        return problems

    try:
        for i, c in enumerate(cases, 1):
            root = _corpus(tmp, f"case{i:02d}")
            c["builder"](root)
            got_exit, got_red, got_green = run_case(c, root)
            problems = judge(c, got_exit, got_red, got_green)
            if problems:
                fails.append((c["name"], problems))
                print(f"  ❌ {i:>2}. {c['name']}")
                for p in problems:
                    print(f"        · {p}")
            else:
                ok_count += 1
                tag = f"  （篡改 {c['mutates']}）" if c.get("mutates") else ""
                print(f"  ✅ {i:>2}. {c['name']} → exit {got_exit}{tag}")

        # ------------------------------------------------------------------
        # 变异测试：把检测器**故意弄坏**，看用例能不能抓住。
        # 「每新增一条断言必须同时加一份篡改样本」—— 这里篡改的是检测器本身，
        # 一个变异只要被任何一个用例抓住就算「N/N 抓住」。
        # ------------------------------------------------------------------
        print("-" * 78)
        print("变异测试（把检测器弄坏，看还有没有用例抓得住）")
        _orig_split_md_row = globals()["split_md_row"]
        _orig_normalize = globals()["normalize"]
        _orig_dup_pairs = globals()["duplicate_pairs"]
        _orig_group_key = globals()["group_key"]
        _orig_coverage_problems = globals()["coverage_problems"]
        _orig_extract_alt = globals()["extract_alt_paths"]
        _orig_mask_budget = globals()["MASK_BUDGET"]
        _orig_keep_masks = globals()["keep_masks"]
        mutations = [
            ("按 `|` 直接切列（不保护反引号）",
             lambda: globals().__setitem__("split_md_row", _naive_split_md_row),
             lambda: globals().__setitem__("split_md_row", _orig_split_md_row)),
            ("归一化改成恒等（不去空白）",
             lambda: globals().__setitem__("normalize", lambda t: t),
             lambda: globals().__setitem__("normalize", _orig_normalize)),
            ("阈值用 n-1（把 17 字也算重复）",
             lambda: globals().__setitem__("duplicate_pairs",
                             lambda texts, n: _orig_dup_pairs(texts, max(n - 1, 1))),
             lambda: globals().__setitem__("duplicate_pairs", _orig_dup_pairs)),
            ("去掉同维限定（跨维一起比）",
             lambda: globals().__setitem__("group_key", lambda item: "ALL"),
             lambda: globals().__setitem__("group_key", _orig_group_key)),
            ("D4 覆盖率不判（0 条也放行）",
             lambda: globals().__setitem__("coverage_problems", lambda *a, **k: []),
             lambda: globals().__setitem__("coverage_problems", _orig_coverage_problems)),
            ("替代路径抽取器返回空（D2 变瞎）",
             lambda: globals().__setitem__("extract_alt_paths", lambda body, c: []),
             lambda: globals().__setitem__("extract_alt_paths", _orig_extract_alt)),
            ("归一化不剥 Markdown 强调符（加粗即绕过）",
             lambda: globals().__setitem__("normalize",
                                           lambda t: _WS_RE.sub("", _ZW_RE.sub("", t))),
             lambda: globals().__setitem__("normalize", _orig_normalize)),
            ("把专名遮罩整个关掉（专名重复也算同质化）",
             lambda: globals().__setitem__("keep_masks",
                                           lambda texts: [[True] * len(t) for t in texts]),
             lambda: globals().__setitem__("keep_masks", _orig_keep_masks)),
            ("专名遮罩模式还原成带空格（死模式）",
             lambda: globals().__setitem__(
                 "MASK_RE", re.compile("|".join(MASK_PATTERNS))),
             lambda: globals().__setitem__(
                 "MASK_RE", re.compile("|".join(pat.replace(" ", "") for pat in MASK_PATTERNS)))),
            ("遮罩预算放宽到 100%（白名单可以吃掉语料）",
             lambda: globals().__setitem__("MASK_BUDGET", 1.0),
             lambda: globals().__setitem__("MASK_BUDGET", _orig_mask_budget)),
            ("重复检测器永远返回空（整台仪器瞎掉）",
             lambda: globals().__setitem__("duplicate_pairs", lambda texts, n: set()),
             lambda: globals().__setitem__("duplicate_pairs", _orig_dup_pairs)),
        ]
        caught = 0
        for mi, (mname, apply_mut, revert) in enumerate(mutations, 1):
            apply_mut()
            try:
                catchers = []
                for i, c in enumerate(cases, 1):
                    root = _corpus(tmp, f"case{i:02d}")   # 目录已存在，沿用已写好的语料
                    got_exit, got_red, got_green = run_case(c, root)
                    if judge(c, got_exit, got_red, got_green):
                        catchers.append(i)
            finally:
                revert()
            if catchers:
                caught += 1
                print(f"  ✅ 变异 {mi}/{len(mutations)} · {mname} → 被用例 {catchers} 抓住")
            else:
                print(f"  ❌ 变异 {mi}/{len(mutations)} · {mname} → "
                      f"**没有任何用例抓住**（说明对应用例是摆设）")
        mutation_summary = f"{caught}/{len(mutations)}"
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"💥 selftest 自身异常：{type(exc).__name__}: {exc}")
        shutil.rmtree(tmp, ignore_errors=True)
        return EXIT_INTERNAL

    print("-" * 78)
    print(f"用例 {ok_count}/{len(cases)} 通过 · 篡改样本 "
          f"{len([c for c in cases if c.get('mutates')])} 个 · "
          f"检测器变异 {mutation_summary} 抓住")
    if fails:
        print(f"🔴 selftest 失败 {len(fails)} 例")
        shutil.rmtree(tmp, ignore_errors=True)
        return EXIT_RED
    if caught != len(mutations):
        print("🔴 有变异没被任何用例抓住 —— 用例是摆设")
        shutil.rmtree(tmp, ignore_errors=True)
        return EXIT_RED
    print("✅ selftest 全过（含反向控制、篡改样本与检测器变异）")
    shutil.rmtree(tmp, ignore_errors=True)
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
