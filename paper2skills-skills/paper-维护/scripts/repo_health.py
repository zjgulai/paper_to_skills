#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repo_health.py — 仓库体检（`paper-维护` 的执行体）

把「人工每隔一阵子发现一类脏数据」变成「一条命令跑完、可回归」的检查。

## 每一项检查都对应一次真实事故，不是想象出来的风险

| 检查 | 对应事故 |
|---|---|
| C1 重复卡片 | `07-NLP-VOC/` 下 26 组同名卡片各存两份 |
| C2 frontmatter 完整性 | 130 张卡只有 73 张有 frontmatter、6 张有 `paper:` 溯源字段。⚠️ 2026-09-13 修口径失真：曾把 48 张 `evidence_basis: author-practice` 卡（按设计就无来源字段）算成「缺 v2 字段」，报出 126 张的假欠账 |
| C3 硬编码绝对路径 | 22 处 `/Users/pray/project/paper_to_skills` 失效路径 |
| C4 **代码围栏结构** | 9 处伪代码标成 ```python、1 张嵌套 ```、2 张围栏**未闭合** —— 这类缺陷在 K1 里表现为「语法错误」，很容易被误判成代码写错，实际是 markdown 结构问题 |
| C5 registry ↔ 卡片一致性 | registry 记的 `outputs.skill_card` 与磁盘实际不符 |
| C6 门禁汇总 | 需要一眼看到 K1/G1/G2/G3 通过率与最近一次运行时间 |
| C7 空目录 / `__pycache__` 入库 | Python 字节码与空目录混进版本控制 |

用法：
    python3 repo_health.py                 # 全量体检，人读报告
    python3 repo_health.py --json-out x.json   # 父目录不存在会自动创建
    python3 repo_health.py --only C4       # 只跑某一项
    python3 repo_health.py --selftest      # 自检：用构造样本证明检查真的会报警，
                                           # 且 C2 的 author-practice 豁免没有把拦截一起关掉

退出码：0 = 无 CRITICAL；1 = 有 CRITICAL（含 C2 的「声明与实物矛盾」红线）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VAULT = REPO / "paper2skills-vault"
RESEARCH = REPO / "paper2skills-research"
REGISTRY = VAULT / "07-资源库" / "papers_registry.json"
GATES = VAULT / "07-资源库" / "gates"
K1_JSON = RESEARCH / "data" / "verification" / "k1_l5.json"

# 允许保留的历史绝对路径（有意保留，不算违规）
# 有意保留的历史绝对路径（见 CLAUDE.md「NLP-VOC 子项目迁出说明」与 T0-3 约定）：
#   paper2skills-code/nlp_voc/ 与 paper2skills-vault/07-NLP-VOC/ 是迁出子项目的
#   代码模板 + 论文档案镜像，其内部硬编码路径是**有意保留**的，不算失效路径。
#   早期白名单只写了 code 侧，导致 vault 侧 290+ 处被误报。
PATH_ALLOWLIST = ("paper2skills-code/nlp_voc/", "paper2skills-vault/07-NLP-VOC/",
                  "logs/", "evolve/", "01-MasterPrompt设计/",
                  "paper-维护/scripts/repo_health.py")
# frontmatter v2 必填字段。
# 口径来源：`paper2skills-vault/07-资源库/MasterPrompt-v2.md`
# 「前置：frontmatter（v2 必填，缺字段即视为未完成）」。
#
# ⚠️ 拆成两组不是洁癖，而是 C2 的全部修复所在（2026-09-13）：
#   · FM_STRUCTURAL_REQUIRED —— **任何卡**都必须有，与有没有论文来源无关
#   · FM_SOURCE_REQUIRED     —— 只有**有来源卡**必须有
# 声明 `evidence_basis: author-practice` 的卡**按设计就没有论文来源**（漏洞 #8/#10
# 建立的三类口径），对它们要求 paper_id/venue 是**判错了对象** ——
# 与 K1 的 ORPHAN_DEP 误判（判一个东西「不存在」之前先 `ls`）、
# 漏洞 #11（把尾部参考区当来源声明）完全同源。
#
# FM_REQUIRED 由两组拼出（而不是另写一份），所以「加一个必填字段」时
# 不可能只加进其中一份而让另一份静默腐烂。
FM_STRUCTURAL_REQUIRED = ("title", "module")
FM_SOURCE_REQUIRED = ("paper_id", "paper", "venue", "venue_tier", "evidence_grade")
FM_REQUIRED = FM_STRUCTURAL_REQUIRED + FM_SOURCE_REQUIRED     # 顺序与历史一致，勿调换

# --- C2 的豁免判据（必须与 gate_check.py 的 G2 证据基础分类同口径）--------------
# `evidence_basis` 取值：
#   paper-verbatim / paper-traceable / mixed → 有来源卡，来源字段必填
#   author-practice（含同义 practice / experience）→ 无论文来源，来源字段**豁免**
#
# ⚠️⚠️ 豁免本身是一次放水 —— 漏洞 #10 的教训：**加一行 frontmatter 就能全库免检**。
# 所以豁免必须同时带上「什么情况下不许豁免」，而且那一条要比拦截条款测得更严：
# 声明 `author-practice` 却在卡里写了 arXiv/DOI/论文标题 → 判**矛盾**、取消豁免、
# 并按有来源卡继续审查（与 gate_check.py 的 `G2-BASIS-CONTRADICTION` 同一判据、同一红线）。
AUTHOR_PRACTICE_BASIS = {"author-practice", "practice", "experience"}
ARXIV_ANY_RE = re.compile(r"\b(?:arXiv\s*[:：]?\s*)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.I)
DOI_ANY_RE = re.compile(r"\b10\.\d{4,9}/[^\s（()\[\]\"'<>]+")
# 尾部「参考论文 / 参考资料」区里的 arXiv ID 是**延伸阅读**，不是本卡的来源声明。
# ⚠️ 这段剥不剥是**会不会造假矛盾**的分水岭，不是可选的清洁动作 ——
# 2026-09-13 实测：48 张 author-practice 卡里 **8 张带尾部参考区，其中 4 张的
# 参考区含 arXiv/DOI**。不剥，这 4 张真·经验卡会被判成「声明与实物矛盾」。
# （词汇表必须含 `参考资料`：全库 43 张用 `参考论文`、另 3 张用 `参考资料`，见漏洞 #11）
_REF_SECTION_RE = re.compile(
    r"^#{1,4}\s*(参考论文|参考文献|参考资料|References|Bibliography|延伸阅读)\s*$",
    re.M | re.I,
)


def strip_reference_section(text: str) -> str:
    m = _REF_SECTION_RE.search(text)
    return text[:m.start()] if m else text


def card_declares_author_practice(fm: dict) -> bool:
    """卡片**声明**了「无论文来源」吗？（只读声明，不验真伪）"""
    basis = (fm.get("evidence_basis") or fm.get("provenance") or "").strip().lower()
    return basis in AUTHOR_PRACTICE_BASIS


def card_has_paper_source(text: str, fm: dict) -> bool:
    """卡片里**实际的**论文线索：有没有 arXiv ID / DOI / 论文标题字段？

    与 `gate_check.card_has_paper_source` 同判据（刻意保持逐条一致）。
    注意：frontmatter 的 `source: human+ai` 是**文档来源**（谁写的），不是论文来源 ——
    全库 67 张卡有该字段，把它当溯源字段会一次误判一大片。
    """
    if (fm.get("paper_id") or "").strip():
        return True
    for k in ("paper", "arxiv", "arxiv_id", "doi", "url"):
        v = (fm.get(k) or "").strip()
        if v and (ARXIV_ANY_RE.search(v) or DOI_ANY_RE.search(v)
                  or (k == "paper" and len(v) > 12)):
            return True
    body = strip_reference_section(text)
    return bool(ARXIV_ANY_RE.search(body) or DOI_ANY_RE.search(body))

# ⚠️ 只认**缩进 ≤3 空格**的围栏。
# 按 CommonMark，缩进 4 空格以上的 ``` 不是块级围栏。实测假阳性案例：
# `Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md` 第 257 行是一段
# **文档里演示 prompt 格式**的缩进示例（`    CONTENT: ```python`），
# 早期版本把它当真围栏，导致全文围栏奇偶错位，误报「围栏未闭合」。
# 这属于「门禁自己的 bug」，必须先修 —— 假红灯会让整份体检报告失去可信度。
FENCE_RE = re.compile(r"^( {0,3})(`{3,})([^\n]*)$", re.M)


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def cards() -> list[Path]:
    return sorted(p for p in VAULT.rglob("Skill-*.md")
                  if p.is_file() and "_superseded" not in p.parts)


def _fm(text: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------
def c1_duplicate_cards() -> dict:
    byname = defaultdict(list)
    for c in cards():
        byname[c.name].append(rel(c))
    dups = {k: v for k, v in byname.items() if len(v) > 1}
    return {"id": "C1", "name": "重复卡片", "critical": bool(dups),
            "detail": dups,
            "summary": f"{len(cards())} 张卡，{len(dups)} 组同名重复"}


def c2_frontmatter() -> dict:
    """frontmatter 完整性 —— **按口径分两类判**（2026-09-13 修口径失真）。

    ## 修的是什么

    C2 曾报「有 frontmatter 但缺 v2 必填字段 **126 张**」。逐张核对后
    **48 张是 `evidence_basis: author-practice` 卡** —— 它们按设计就不该有
    `paper_id` / `paper` / `venue` / `venue_tier`（漏洞 #8/#10 的三类口径）。
    把它们记成「有缺陷」有两个代价：
      ① 欠账表虚高 48 张（126 → 真值 78）；
      ② **把补救动作指向错误的方向** —— 经验卡的补救是「诚实声明」，
         不是「去找一篇论文」。与 K1 的 ORPHAN_DEP 误判同源。

    ## 现在的两个判定

    1. **有来源卡缺 v2 字段** → 真缺陷，继续报（`incomplete_v2`，并给出缺失字段名）
    2. **author-practice 卡缺来源字段** → 豁免，不报（`exempted_author_practice`，
       名单仍单列 —— 豁免必须**可审计**，否则等于静默放行）

    ## 豁免不许溢出的边界

    - 豁免**只管来源字段**。`title` / `module` 是结构性字段，author-practice 卡
      缺了照样报（实测全库 48 张都齐，但判据不能靠「今天恰好齐」）。
    - 声明与实物矛盾（卡里有 arXiv/DOI 却自称无论文来源）→ **取消豁免**，
      按有来源卡审查，并单列 `contradictions`（对应 gate_check 的红灯
      `G2-BASIS-CONTRADICTION`）。这一条是**反后门**：没有它，
      「加一行 `evidence_basis: author-practice`」就是全库免检令牌。
    - 尾部「参考论文」区先剥掉再判来源（漏洞 #11）—— 否则一张列了延伸阅读的
      纯经验卡会被判成矛盾（实测 4 张会中招）。

    ⚠️ 一张卡可以同时出现在 `incomplete_v2` 与 `contradictions` 里（矛盾卡按
    有来源卡审查，若它又缺字段就两边都在）。这两个名单是**清单不是分母**，
    不做相加，也不参与任何通过率。
    """
    miss_all, miss_v2, exempted, contradictions = [], [], [], []
    for c in cards():
        text = c.read_text(encoding="utf-8", errors="replace")
        fm = _fm(text)
        if not fm:
            miss_all.append(rel(c))
            continue
        basis = (fm.get("evidence_basis") or fm.get("provenance") or "").strip().lower()
        declared_practice = card_declares_author_practice(fm)
        has_source = card_has_paper_source(text, fm)
        if declared_practice and has_source:
            contradictions.append({
                "card": rel(c), "basis": basis,
                "note": "frontmatter 声明无论文来源，但卡内存在 arXiv/DOI/论文标题 —— "
                        "声明与实物矛盾，已取消豁免、按有来源卡审查"
                        "（同 gate_check 的 G2-BASIS-CONTRADICTION）",
            })
            declared_practice = False        # ← 反后门：矛盾即取消豁免
        if declared_practice:
            exempted.append({"card": rel(c), "basis": basis,
                             "exempted_fields": list(FM_SOURCE_REQUIRED)})
            lack = [k for k in FM_STRUCTURAL_REQUIRED if not fm.get(k)]
            if lack:
                # 豁免只管来源字段：结构性字段缺了仍算缺陷，不许跟着一起放行
                miss_v2.append({"card": rel(c), "missing": lack, "basis": basis,
                                "note": "来源字段已豁免；title/module 不豁免"})
            continue
        lack = [k for k in FM_REQUIRED if not fm.get(k)]
        if lack:
            miss_v2.append({"card": rel(c), "missing": lack,
                            "basis": basis or "(未声明)"})
    n_sourced_lack = sum(1 for e in miss_v2
                         if e.get("basis") not in AUTHOR_PRACTICE_BASIS)
    n_ap_structural = len(miss_v2) - n_sourced_lack
    by_field = {k: sum(1 for e in miss_v2 if k in e["missing"]
                       and e.get("basis") not in AUTHOR_PRACTICE_BASIS)
                for k in FM_SOURCE_REQUIRED}
    summary = (f"无 frontmatter {len(miss_all)} 张；"
               f"有来源卡缺 v2 必填字段 {n_sourced_lack} 张"
               f"（缺 " + " / ".join(f"{k} {v}" for k, v in by_field.items()) + "）")
    if exempted or n_ap_structural:
        summary += (f"；author-practice 卡豁免来源字段 {len(exempted)} 张")
        if n_ap_structural:
            summary += f"（其中缺结构性字段 {n_ap_structural} 张，仍计入缺陷）"
    if contradictions:
        summary += f"；⚠️ 声明矛盾 {len(contradictions)} 张（反后门红线）"
    return {"id": "C2", "name": "frontmatter 完整性",
            # 矛盾是**红线**：它意味着「免检令牌」正在被滥用，与 gate_check 的
            # G2-BASIS-CONTRADICTION 同级。实测 0 张，故不改变今日结论。
            "critical": bool(contradictions),
            "detail": {"no_frontmatter": miss_all,
                       "incomplete_v2": miss_v2,
                       "exempted_author_practice": exempted,
                       "contradictions": contradictions},
            # 历史版本没有 total，于是 C2 有 126 张欠账时仍显示 ✅（图标只看
            # critical/total）。加上 total 后它会如实显示 🟡。
            "total": len(miss_all) + n_sourced_lack + n_ap_structural + len(contradictions),
            "summary": summary}


def c3_hardcoded_paths() -> dict:
    bad = []
    pat = re.compile(r"/Users/(?:pray|lute)/project/\S+")
    for p in list(REPO.rglob("*.py")) + list(REPO.rglob("*.md")):
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        r = rel(p).replace("\\", "/")
        if any(a in r for a in PATH_ALLOWLIST):
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pat.search(line):
                    bad.append({"file": r, "line": i, "text": line.strip()[:110]})
        except OSError:
            continue
    return {"id": "C3", "name": "硬编码绝对路径", "critical": False,
            "detail": bad[:60], "total": len(bad),
            "summary": f"{len(bad)} 处可执行代码/文档里的绝对路径（历史归档已白名单豁免）"}


def c4_code_fences() -> dict:
    """检查 markdown 代码围栏的**结构**问题。

    ⚠️ 单独做这一项，是因为这类缺陷在 K1 里表现为「语法错误」：
    - 伪代码标成 ```python → K1 报 ast 语法错误（看着像代码写错）
    - 围栏未闭合 → 后面的正文被当成 Python 解析 → 报一堆莫名其妙的语法错误
    - 内层含 ``` 但外层只有 3 个反引号 → 代码块被截成两半，后半段**整个丢失**
    这三类共 12 处，本仓库全部踩过。
    """
    issues = []
    for c in cards():
        text = c.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        stack: list[tuple[int, int, str]] = []      # (行号, 反引号数, 语言)
        indented_fences = []
        for i, line in enumerate(lines, 1):
            m = FENCE_RE.match(line)
            if not m:
                continue
            ticks = len(m.group(2))
            lang = m.group(3).strip()
            if m.group(1):
                # 缩进 1-3 空格的围栏在列表项里合法，但会让「奇偶配对」这类
                # 朴素检查产生歧义。实测案例：Skill-DeepAnalyze 第 259 行是
                # **文档里演示 prompt 格式**的缩进示例，真正的代码块在第 422 行
                # 才闭合 —— 而 K1 能正确抽出并编译那段代码，说明卡片本身没问题。
                # 故**只用零缩进围栏做硬判定**，缩进的单独记为提示。
                indented_fences.append(i)
                continue
            if stack and ticks >= stack[-1][1]:
                opened_at, open_ticks, open_lang = stack.pop()
                # 内层围栏检测：块内出现了 >= 外层长度的围栏（正确写法是外层加长）
                body = "\n".join(lines[opened_at:i - 1])
                # 只把**零缩进**的内层围栏当作真嵌套（缩进的是列表项里的示例）
                inner = re.findall(r"^(`{3,})", body, re.M)
                if inner and max(len(x) for x in inner) >= open_ticks:
                    issues.append({
                        "card": rel(c), "type": "嵌套围栏外层未加长",
                        "line": opened_at, "lang": open_lang,
                        "note": f"块内有 {max(len(x) for x in inner)} 个反引号的围栏，"
                                f"外层只有 {open_ticks} 个 → 代码会被截断，后半段丢失",
                    })
            else:
                stack.append((i, ticks, lang))
                if lang.lower() in ("python", "py"):
                    # 伪代码启发式：python 块里出现 ASCII 框图字符
                    pass
        if indented_fences:
            issues.append({
                "card": rel(c), "type": "存在缩进围栏（提示，非缺陷）",
                "line": indented_fences[0], "lang": "",
                "note": f"第 {indented_fences} 行有缩进 1-3 空格的围栏，通常是列表项里"
                        f"演示代码格式的写法。已排除在硬判定之外，仅供人工确认。",
            })
        unclosed = [(ln, lg) for ln, _, lg in stack]
        if unclosed:
            issues.append({"card": rel(c), "type": "围栏未闭合",
                           "line": unclosed[0][0], "lang": unclosed[0][1],
                           "note": "此处之后的正文会被当作代码解析，K1 会报大量假语法错误"})
        # 伪代码标成 python：块内含框图字符或纯签名列表
        for m in re.finditer(r"^```(?:python|py)\s*$", text, re.M):   # 零缩进
            start = m.end()
            end = text.find("```", start)
            body = text[start:end] if end > 0 else ""
            if re.search(r"[│├└┌┐┘─┴┬┼]", body) and "def " not in body:
                issues.append({"card": rel(c), "type": "伪代码标成 python",
                               "line": text[:m.start()].count("\n") + 1, "lang": "python",
                               "note": "块内是 ASCII 框图，应标 ```text 或 ```pseudocode"})
    # ⚠️ critical 必须只看**真缺陷**：早期写成 `bool(issues)`，
    # 于是「存在缩进围栏」这类提示会让整份体检永远报 CRITICAL，
    # 而永远红的门禁等于没有门禁（会被直接忽略）。
    real_issues = [i for i in issues if i["type"] != "存在缩进围栏（提示，非缺陷）"]
    return {"id": "C4", "name": "代码围栏结构", "critical": bool(real_issues),
            "detail": issues[:40], "total": len(real_issues),
            "summary": (f"{len(real_issues)} 处围栏结构缺陷"
                        f"（未闭合 / 嵌套未加长 / 伪代码标错）"
                        + (f"；另有 {len(issues) - len(real_issues)} 条缩进围栏提示（非缺陷）"
                           if len(issues) > len(real_issues) else ""))}


def c5_registry_consistency() -> dict:
    if not REGISTRY.is_file():
        return {"id": "C5", "name": "registry 一致性", "critical": True,
                "detail": [], "summary": "registry 文件不存在"}
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    missing, extracted = [], 0
    by_paper = {}
    for c in cards():
        fm = _fm(c.read_text(encoding="utf-8", errors="replace"))
        if fm.get("paper_id"):
            by_paper.setdefault(fm["paper_id"], []).append(rel(c))
    for rec in data.get("records", []):
        out = (rec.get("outputs") or {}).get("skill_card") or ""
        if rec.get("status") == "extracted":
            extracted += 1
        if out and not (REPO / out).is_file() and "<" not in out:
            missing.append({"paper_id": rec["paper_id"], "expected": out})
    no_registry = [p for p in by_paper
                   if p not in {r.get("identifiers", {}).get("arxiv") for r in data.get("records", [])}]
    return {"id": "C5", "name": "registry 一致性", "critical": False,
            "detail": {"declared_but_missing": missing, "card_paper_not_in_registry": no_registry},
            "summary": (f"registry {len(data.get('records', []))} 条（extracted {extracted}）；"
                        f"声明存在但磁盘缺失 {len(missing)}；卡片 paper_id 不在 registry {len(no_registry)}")}


def c6_gate_summary() -> dict:
    out = {}
    for g in ("g1", "g2", "g3"):
        f = GATES / f"gate_{g}.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            out[g.upper()] = {"summary": d.get("summary", {}),
                              "generated_at": d.get("generated_at", "")}
    if K1_JSON.is_file():
        k = json.loads(K1_JSON.read_text(encoding="utf-8"))
        results = k.get("results", [])
        n = len(results)
        p = sum(1 for r in results if r.get("verdict") == "PASS")
        # ⚠️ K1 产物的时间戳在 `summary.generated_at`，不在顶层；
        #    早期只读顶层，导致误报「K1 产物缺生成时间戳」（是我的检查读错了，不是 K1 没写）。
        out["K1"] = {"units": n, "pass": p,
                     "rate_pct": round(p / n * 100, 1) if n else 0.0,
                     "generated_at": (k.get("summary") or {}).get(
                         "generated_at", k.get("generated_at", ""))}
    stale = []
    # ⚠️ 必须用 tz-aware 的 now：门禁产物的时间是 `2026-09-12T12:20:08+0800`（带时区），
    # 而 `datetime.now()` 是 naive，两者相减抛 TypeError，被 except 吞掉后
    # 全部误报成「时间戳不可解析」（实测 G1/G2/G3 三条假告警）。
    now = datetime.now().astimezone()
    for name, info in out.items():
        ts = info.get("generated_at") or ""
        if not ts:
            stale.append(f"{name} 产物缺生成时间戳")
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.astimezone()
            age = (now - dt).days
            if age > 7:
                stale.append(f"{name} 产物已 {age} 天未更新")
        except Exception as exc:
            stale.append(f"{name} 产物时间戳无法解析（{ts}）：{exc}")
    return {"id": "C6", "name": "门禁汇总", "critical": False,
            "detail": out, "stale": stale,
            "summary": "；".join(
                # ⚠️ 门禁产物的百分比在 v["summary"]["pass_rate_pct"]，
                #    而 K1 的在 v["rate_pct"] —— 早期只读顶层，四个门禁全显示成 "?"。
                f"{k} " + (f"{v['summary']['pass_rate_pct']}%"
                           if isinstance(v.get("summary"), dict)
                           and "pass_rate_pct" in v["summary"]
                           else (f"{v['rate_pct']}%" if "rate_pct" in v else "?"))
                for k, v in out.items()) or "无门禁产物"}


def c7_repo_hygiene() -> dict:
    """仓库卫生。

    ⚠️ 关键区分：**被 git 跟踪**的脏文件才是问题；本地存在但已被 .gitignore 忽略的
    不是问题。早期版本只数文件个数，于是每次跑都会报「6 个 .pyc」，
    而 `__pycache__/` 早已在 .gitignore 里 —— 这种「永远报同一件事」的检查
    会训练人忽略报告，比不检查更糟。
    """
    try:
        import subprocess
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                                 text=True, timeout=60).stdout.splitlines()
    except Exception:
        tracked = []
    junk_tracked = [t for t in tracked
                    if t.endswith(".pyc") or "__pycache__" in t or t.endswith(".pyo")]
    local_pyc = sum(1 for _ in REPO.rglob("*.pyc") if ".git" not in _.parts)
    empty = [rel(p) for p in VAULT.rglob("*")
             if p.is_dir() and not any(p.iterdir()) and ".git" not in p.parts]
    return {"id": "C7", "name": "仓库卫生", "critical": bool(junk_tracked),
            "detail": {"tracked_junk": junk_tracked[:10],
                       "empty_dirs": empty[:10],
                       "_local_only_not_tracked": f"{local_pyc} 个 .pyc 在本地但已被 .gitignore 忽略"},
            "total": len(junk_tracked),
            "summary": (f"被 git 跟踪的垃圾文件 {len(junk_tracked)} 个"
                        f"（本地另有 {local_pyc} 个 .pyc，已被 .gitignore 忽略，不算问题）；"
                        f"空目录 {len(empty)} 个")}


def c8_sections() -> dict:
    """检查卡片是否具备 v2 的六段结构。

    本项是「迁移欠账」的度量：存量卡片多为 v1 五段式甚至只有①②③，
    而 G2 要求 ⑥ 原文引用、G3 要求 ⑤ 商业价值 —— 缺段必然过不了门禁。
    实测：有 3 张卡只有 ①②③（代码块未闭合、且完全没有 ④⑤⑥）。
    """
    HAVE = {"①": r"^#+\s*①", "①b": r"^#+\s*①b", "②": r"^#+\s*②",
            "③": r"^#+\s*③", "④": r"^#+\s*④", "⑤": r"^#+\s*⑤", "⑥": r"^#+\s*⑥"}
    rows, no_six, only3 = [], [], []
    for c in cards():
        text = c.read_text(encoding="utf-8", errors="replace")
        have = [k for k, pat in HAVE.items() if re.search(pat, text, re.M)]
        missing = [k for k in HAVE if k not in have]
        rows.append({"card": rel(c), "missing": missing})
        if "⑥" not in have:
            no_six.append(rel(c))
        if set(have) <= {"①", "②", "③"}:
            only3.append(rel(c))
    return {"id": "C8", "name": "v2 段落完整性", "critical": False,
            "detail": {"missing_⑥": no_six[:20], "only_①②③": only3[:20],
                       "count_missing_six": len(no_six), "count_only3": len(only3)},
            "total": len(no_six),
            "summary": (f"缺 ⑥ 原文引用 {len(no_six)}/{len(rows)} 张"
                        f"（其中 {len(only3)} 张仅有 ①②③，是 v1 遗留）")}


# ---------------------------------------------------------------------------
# C10 验证声明的时间锚点（PHASE6 F1，2026-09-13）
# ---------------------------------------------------------------------------
# 卡片 frontmatter 的 `verified_by` 是一句**没有时间的断言**：
#     verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（…VERBATIM）
#                  + gate_check.py G2 passed + 人工抽检 3 处数字
# 读者无法分辨这是昨天验的还是半年前验的 —— 而两者含义完全不同。
#
# **实测（2026-09-13，PHASE6 F1 现场）**：25 张卡有 `verified_by`、**0 张有 `verified_at`**；
# 按当时权威门禁产物，其中 **9 张已实测为红**，而它们**当初是真的** ——
# PHASE5 的 C6（引用编号 `[8]` 洗白数字）与 C9（`A100` 洗白 `100`）把 G2a 改严了，
# 卡片并不知道。⇒ **声明腐烂时，卡片自己不会说话。**
#
# ⚠️ 本项**刻意不引入任何人工状态字段**（如 `verified_stale: true`）：
#    任何「由人写、能让检查闭嘴」的字段都是一个新后门（参见漏洞 #10 的教训）。
#    stale 与否**由测量导出**，卡片只记录「断言是在哪一天作出的」这个事实。

# 声明里的 token → 它实际支撑的门禁层。
_CLAIM_LAYER_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("G1",  (r"\bG1\b", r"\bK1\b", r"verify_skill_code")),
    ("G3",  (r"\bG3\b",)),
    ("G2",  (r"\bG2\b",)),
    ("G2b", (r"quote_check",)),
)

# ⚠️⚠️ **反后门（本项最容易做错的地方）**：
# `G2b`（逐字引文层）被指责的**只能**是引文层自己的红灯。
# 实测反例：9 张被判矛盾的卡**全部只因 `G2-UNSOURCED-METRIC` 变红**（G2a 层），
# 而它们的 `quote_check` 逐字核验至今仍是真的（`G2-QUOTE-VERBATIM` 绿）。
# 若把「G2 聚合红」直接算成「引文声明被推翻」，就会报 9 条**假矛盾** ——
# 与漏洞 #11「判据只认一种字段名」同族：**判据的适用范围被默认成了全体。**
QUOTE_LAYER_RED_CODES = frozenset({
    "G2-QUOTE-FABRICATED",     # 引文在底本里找不到 → 伪造
    "G2-QUOTE-TRUNCATED",      # 引文被截成残句（漏洞 #15）
    "G2-NO-EVIDENCE-CHAIN",    # 卡内既无引用块也无 evidence.md ⇒「VERBATIM」无从成立
})
QUOTE_LAYER_RED_PREFIX = "G2-QUOTE-"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _git_commit_dates() -> dict[str, dict[str, str]]:
    """一次 `git log` 拿到每个文件的首次/末次提交日。

    只调一次 git：逐卡调 `git log` 是 146 次进程启动，体检会从秒级退化到分钟级。
    """
    try:
        p = subprocess.run(
            ["git", "-c", "core.quotepath=false", "log",
             "--format=@@%cs", "--name-only", "--no-renames"],
            cwd=REPO, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return {}
    if p.returncode != 0:
        return {}
    out: dict[str, dict[str, str]] = {}
    cur = ""
    for line in p.stdout.splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
        elif line.strip() and cur:
            # git log 按时间倒序 ⇒ 首次见到的就是**末次**提交；持续覆盖得到**首次**
            e = out.setdefault(line.strip(), {"first": cur, "last": cur})
            e["first"] = cur
    return out


def _gate_lookup() -> dict[str, dict]:
    """卡片 rel 路径 → 三个门禁的当前判定（读权威产物，不重跑）。"""
    out: dict[str, dict] = {}
    for g in ("g1", "g2", "g3"):
        f = GATES / f"gate_{g}.json"
        if not f.is_file():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for r in d.get("results", []):
            e = out.setdefault(r.get("target", ""), {})
            e[g.upper()] = {
                "passed": r.get("passed"),
                "red_codes": [f2.get("code", "") for f2 in r.get("findings", [])
                              if f2.get("level") == "RED"],
            }
    return out


def c10_verified_claims(dates: dict | None = None,
                        gates: dict | None = None,
                        today: str | None = None) -> dict:
    """C10 验证声明的时间锚点与实测一致性。

    `dates` / `gates` / `today` 可注入 —— 自检靠它用构造样本喂假数据，
    否则「C10 真的会报警」这句话无法被证明（构造一个 git 仓库与时间机器都不现实）。
    """
    dates = _git_commit_dates() if dates is None else dates
    gates = _gate_lookup() if gates is None else gates
    today = datetime.now().astimezone().strftime("%Y-%m-%d") if today is None else today

    anchor_missing: list[dict] = []
    bad_anchor: list[dict] = []
    contradicted: list[dict] = []
    unspecific: list[dict] = []
    ok_cards: list[str] = []
    bad_seen: set[str] = set()
    contra_seen: set[str] = set()
    unspec_seen: set[str] = set()

    for c in cards():
        text = c.read_text(encoding="utf-8", errors="replace")
        fm = _fm(text)
        claim = (fm.get("verified_by") or "").strip()
        if not claim:
            continue
        relp = rel(c)
        vat = (fm.get("verified_at") or "").strip()

        # (a) 有声明、无锚点
        if not vat:
            anchor_missing.append({"card": relp, "claim": claim[:80]})
        elif not _DATE_RE.match(vat):
            bad_anchor.append({"card": relp, "verified_at": vat,
                               "why": "不是 YYYY-MM-DD 格式，无法比较"})
            bad_seen.add(relp)
        else:
            # (b) 声称了一个还没发生的验证
            if vat > today:
                bad_anchor.append({"card": relp, "verified_at": vat,
                                   "why": f"晚于今天（{today}）—— 该验证不可能已发生"})
                bad_seen.add(relp)
            # (c) 声称在卡片存在之前就验过
            d = dates.get(relp) or {}
            if d.get("first") and vat < d["first"]:
                bad_anchor.append({"card": relp, "verified_at": vat,
                                   "why": f"早于该卡首次提交（{d['first']}）—— 卡还不存在"})
                bad_seen.add(relp)

        # (d) 点名了 `gate_check` 却**没说它判了哪个门禁** ⇒ 不可复核（🟡，非矛盾）。
        # 实测命中 2 张：`… + gate_check.py + 人工抽检`（无 G2/G3 字样）。
        # ⚠️ 这里**不能**把它当成「声称了 G2」去报矛盾 —— 卡并没有断言 G2 通过，
        #    报矛盾就是我们自己造了一个它没做过的声明（同「假红灯」家族）。
        #    但它也确实不可复核，故单列一档，而不是静默丢弃。
        if re.search(r"gate_check", claim) and not re.search(r"\bG[123]\b", claim):
            unspecific.append({"card": relp, "claim": claim[:90],
                               "why": "点名 gate_check 但未写判了哪个门禁（G1/G2/G3），无法复核"})
            unspec_seen.add(relp)

        # (e) 声明点名的门禁**当前实测为红**
        hit = gates.get(relp) or {}
        for ly, toks in _CLAIM_LAYER_TOKENS:
            if not any(re.search(t, claim) for t in toks):
                continue
            if ly == "G2b":
                reds = [code for code in (hit.get("G2") or {}).get("red_codes", [])
                        if code in QUOTE_LAYER_RED_CODES
                        or code.startswith(QUOTE_LAYER_RED_PREFIX)]
                if reds:
                    contradicted.append({"card": relp, "layer": "G2b（逐字引文）",
                                         "red_codes": sorted(set(reds)),
                                         "verified_at": vat or "—"})
                    contra_seen.add(relp)
            else:
                g = hit.get(ly)
                if g and g.get("passed") is False:
                    contradicted.append({"card": relp, "layer": ly,
                                         "red_codes": sorted(set(g.get("red_codes", [])))[:4],
                                         "verified_at": vat or "—"})
                    contra_seen.add(relp)
        if vat and relp not in bad_seen and relp not in contra_seen and relp not in unspec_seen:
            ok_cards.append(relp)

    # ⚠️ total 必须是**去重后的张数**：一张卡可以同时「无锚点」且「矛盾」，
    #    早期写成四项相加，25 + 7 报成 32 —— 与本仓库反复出现的
    #    「分子分母口径不一致」同族（G2 漏洞 #8/#9 就是这个形态）。
    #    另：`consistent` 也曾把「不可复核」的卡算进去（7+2+18=27 > 25），
    #    现已互斥 —— 五档相加 == total（存在「既无锚点又矛盾」的重叠时，
    #    以去重后的 total 为准，故下方 summary 里 total 单独报）。
    total = len({e["card"] for e in anchor_missing} | bad_seen | contra_seen
                | unspec_seen | set(ok_cards))
    return {
        "id": "C10", "name": "验证声明时间锚点",
        # 矛盾是真红线（卡片在断言一件可测量为假的事）；
        # 缺锚点/不可复核是欠账，不是撒谎 —— 必须分开，否则红线会被欠账淹没。
        "critical": bool(bad_anchor or contradicted),
        "detail": {"total_with_claim": total,
                   "anchor_missing": anchor_missing,
                   "bad_anchor": bad_anchor,
                   "claim_contradicted": contradicted,
                   "claim_unspecific": unspecific,
                   "consistent": len(ok_cards)},
        "total": len(anchor_missing) + len(unspecific),
        "summary": (f"{total} 张有 verified_by：无锚点 {len(anchor_missing)} / "
                    f"锚点不实 {len(bad_seen)} / **声明与实测矛盾 {len(contra_seen)}** / "
                    f"点名工具但不可复核 {len(unspecific)} / 一致 {len(ok_cards)}"),
    }


CHECKS = {
    "C1": c1_duplicate_cards, "C2": c2_frontmatter, "C3": c3_hardcoded_paths,
    "C4": c4_code_fences, "C5": c5_registry_consistency, "C6": c6_gate_summary,
    "C7": c7_repo_hygiene,
    "C8": c8_sections,
    "C10": c10_verified_claims,
}


def write_json_report(path: Path, payload: dict) -> None:
    """写 JSON 产物 —— **先建父目录**（2026-09-13 修漏洞 #8）。

    ⚠️ 为什么这一行是必要的，而不是防御性编程：

    `Path.write_text` 在父目录不存在时抛 `FileNotFoundError`，而这一步在报告
    **全部打印完之后**才执行 —— 于是输出长这样：

        ...各检查项全部正常...
        结论：✅ 无 CRITICAL            ← 体检明明通过了
        FileNotFoundError: ...          ← 然后崩在这里
        exit=1

    **这个失败看起来像体检失败，实际体检是通过的。** 与漏洞 #6
    （`gate_check` 找不到 evidence.md 却照样绿）是同一类「静默假失败/假绿灯」，
    只是方向相反。后果：
      ① PHASE5 T5-4「周日体检 + 趋势追踪」拿不到可比较的历史文件；
      ② 退出码 1 会让自动化把一次**成功的**体检记成失败，久而久之没人再看退出码。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
# 构造样本的**设计口径**：每一个「豁免」都必须配一个「豁免不许溢出」的反例。
# 本仓库已封堵的 15 个漏洞里，有 2 个（#10 新口径变后门、#11 参考区被当来源）
# 正是「修一个问题时新开了一个洞」——所以自检里豁免条款的用例数 **多于** 拦截条款。
def _load_gate_check_for_crosscheck():
    """把 gate_check.py 作为模块加载，用于**交叉验证两份同口径判据**。

    返回 (模块 | None, 失败原因)。

    ⚠️ 必须先注册 `sys.modules` 再 `exec_module` —— 否则 CPython 3.14 下
    `@dataclass` + `from __future__ import annotations` 的模块会崩在
    `dataclasses._is_type`。这就是 CLAUDE.md「门禁自身的 bug #5」的最小复现，
    本脚本自己踩过一次（写分析脚本时），所以这里显式记下来。

    加载失败**不算自检失败**：gate_check 可能正被别人改（本仓库有并行修改约定），
    把别人的中间状态算成自己的红灯，会制造「假红灯比假绿灯更常见」的坏名声。
    """
    import importlib.util
    p = Path(__file__).resolve().parents[3] / "paper2skills-skills" / "paper-审核" / "scripts" / "gate_check.py"
    if not p.is_file():
        return None, f"未找到 {p}"
    try:
        spec = importlib.util.spec_from_file_location("_gc_crosscheck", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_gc_crosscheck"] = mod
        spec.loader.exec_module(mod)
        return mod, ""
    except Exception as exc:                                    # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _c2_samples() -> dict[str, tuple[str, dict]]:
    """C2 的构造样本（文本 + 已解析 frontmatter），供自检与交叉验证共用。"""
    def mk(text: str) -> tuple[str, dict]:
        return text, _fm(text)

    samples = {
        "有来源卡·字段齐全": mk(
            "---\ntitle: T\nmodule: 01-域\npaper_id: 2606.26690\n"
            "paper: A Real Paper Title\nvenue: KDD 2026\nvenue_tier: CCF-A\n"
            "evidence_grade: A\n---\n正文\n"),
        "有来源卡·缺 venue": mk(
            "---\ntitle: T\nmodule: 01-域\npaper_id: 2606.26690\n"
            "paper: A Real Paper Title\nvenue_tier: CCF-A\nevidence_grade: A\n---\n正文\n"),
        "经验卡·来源字段全缺": mk(
            "---\ntitle: T\nmodule: 01-域\nevidence_basis: author-practice\n---\n正文\n"),
        "经验卡·声明矛盾(带 paper_id)": mk(
            "---\ntitle: T\nmodule: 01-域\nevidence_basis: author-practice\n"
            "paper_id: 2606.26690\n---\n正文\n"),
        "经验卡·参考区有 arXiv(不算矛盾)": mk(
            "---\ntitle: T\nmodule: 01-域\nevidence_basis: author-practice\n---\n正文\n"
            "## 参考论文\n- Foo et al. arXiv:2408.05353\n"),
        "经验卡·正文有 DOI(矛盾)": mk(
            "---\ntitle: T\nmodule: 01-域\nevidence_basis: author-practice\n---\n"
            "正文，来源见 10.1287/mnsc.2022.02462\n"),
        "经验卡·缺结构性字段 title": mk(
            "---\nmodule: 01-域\nevidence_basis: author-practice\n---\n正文\n"),
        # ⚠️ 下面两例锁定一个**刻意做出的决定**，不是随手补的边界测试（2026-09-13）：
        #
        # 全库实测有 **11 张卡 frontmatter 没有 `title:`** —— 但其中只有 **2 张**改用
        # 了 Skill 格式的 `name:`/`description`，另 **9 张连标题字段都没有**（标题只写在
        # 正文 H1 里）。当时的登记文案写作「11 张卡用 `name:` 而非 `title:`」，**与实物不符**。
        #
        # 它看起来像漏洞 #11（「判据只认一种字段名 → 把另一种形态漏掉」），
        # 但**不是**：#11 里被漏掉的是**同一概念的另一种合规写法**（`参考资料`/`参考论文`），
        # 而 `name:`+`description` 是 **Claude Code Skill 的格式**（见 CLAUDE.md
        #「Skill File Format」），属于**另一种交付物**，vault 卡片的唯一格式权威
        #（`MasterPrompt-v2.md` 第 57 行）明写必填的是 `title:`。
        #
        # 所以处置是**修资产而不是放松判据**：给这 11 张补 `title:`（另 2 张的 `area:`
        # 改名 `module:`）。放松判据会让 C2 从此接受一个**没有格式依据**的替代字段 ——
        # 那正是「加一行 frontmatter 就能全库免检」（漏洞 #10）的同型风险。
        # 这两条用例就是防止后来者「为了让 11 张卡变绿」把判据改宽。
        "经验卡·用 name 代替 title（不算有 title）": mk(
            "---\nname: Some-Skill\ndescription: 该技能应在…时使用\nmodule: 01-域\n"
            "evidence_basis: author-practice\n---\n正文\n"),
        "有来源卡·用 area 代替 module（不算有 module）": mk(
            "---\ntitle: T\narea: 01-域\npaper_id: 2606.26690\n"
            "paper: A Real Paper Title\nvenue: KDD 2026\nvenue_tier: CCF-A\n"
            "evidence_grade: A\n---\n正文\n"),
        "经验卡·带文档来源 source:human+ai": mk(
            "---\ntitle: T\nmodule: 01-域\nevidence_basis: author-practice\n"
            "source: human+ai\n---\n正文\n"),
    }
    return samples


def _c10_isolated_noanchor(root: Path) -> bool:
    """在**只含一张无锚点卡**的隔离 vault 里跑 C10，断言 critical=False。

    隔离是必需的：调用方所在的 vault 里还有「锚点晚于今天」的样本，
    critical 会被它合理地抬起 —— 那样测到的就不是本用例想测的东西。
    """
    global VAULT
    saved = VAULT
    try:
        iso = root / "vault-iso-c10"
        (iso / "01-域").mkdir(parents=True, exist_ok=True)
        (iso / "01-域" / "Skill-Iso-NoAnchor.md").write_text(
            "---\ntitle: iso\nverified_by: gate_check.py G2 passed\n---\n\n# 样本\n",
            encoding="utf-8")
        VAULT = iso.resolve()
        rep = c10_verified_claims(dates={}, gates={}, today="2026-09-13")
        # 前提核对：确实抓到了那一张（否则「没报警」可能只是「没扫到」）
        got = len(rep["detail"]["anchor_missing"]) == 1
        return got and rep["critical"] is False
    finally:
        VAULT = saved


def selftest() -> int:
    """自检：用构造样本证明 C1/C2/C4 真的会报警，且**豁免不许溢出**。

    2026-09-13 扩充。新增用例集中在 C2 的两件事上：
      · 豁免真的生效（author-practice 卡不再被报）
      · 豁免**没有把拦截一起关掉**（缺 venue 的有来源卡照样报、矛盾照样报、
        结构性字段照样报、参考区不造假矛盾）
    """
    import tempfile
    global VAULT, REPO
    ok = True
    samples = _c2_samples()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "vault" / "01-域").mkdir(parents=True)
        (root / "vault" / "02-域").mkdir(parents=True)
        # C1 构造：同名卡片两份
        (root / "vault" / "01-域" / "Skill-Dup.md").write_text("---\ntitle: a\n---\n", encoding="utf-8")
        (root / "vault" / "02-域" / "Skill-Dup.md").write_text("---\ntitle: a\n---\n", encoding="utf-8")
        # C2 构造：无 frontmatter
        (root / "vault" / "01-域" / "Skill-NoFm.md").write_text("# 无 frontmatter\n", encoding="utf-8")
        # C2 构造：各口径样本各落一张卡（文件名与 samples 的键一一对应）
        for i, (name, (text, _fm_dict)) in enumerate(samples.items()):
            (root / "vault" / "02-域" / f"Skill-Sample{i}.md").write_text(text, encoding="utf-8")
        # C4 构造：围栏未闭合 + 伪代码
        (root / "vault" / "01-域" / "Skill-BadFence.md").write_text(
            "---\ntitle: b\n---\n```python\nx = 1\n", encoding="utf-8")
        (root / "vault" / "01-域" / "Skill-Pseudo.md").write_text(
            "---\ntitle: c\n---\n```python\n┌───┐\n│ A │\n└───┘\n```\n", encoding="utf-8")

        # ⚠️ 必须 resolve()：`rel()` 用 `p.resolve().relative_to(REPO)`，
        # 而 macOS 的临时目录是 `/var/folders/…` 的符号链接，resolve 后变成
        # `/private/var/folders/…` —— 不 resolve 会让 relative_to 抛 ValueError，
        # `rel()` 静默退回**绝对路径**，于是按文件名找样本全部落空。
        # （本自检第一版就是这么红的，见报告：自检真的会失败。）
        VAULT, REPO = (root / "vault").resolve(), root.resolve()
        r1, r2, r4 = c1_duplicate_cards(), c2_frontmatter(), c4_code_fences()

        miss_v2 = r2["detail"]["incomplete_v2"]
        exempt = {e["card"] for e in r2["detail"]["exempted_author_practice"]}
        contra = {e["card"] for e in r2["detail"]["contradictions"]}
        nofm = set(r2["detail"]["no_frontmatter"])

        def _card_for(key: str) -> str:
            """样本顺序 → 落盘文件名（samples 的插入顺序即编号顺序）。"""
            return f"vault/02-域/Skill-Sample{list(samples).index(key)}.md"

        def _reported(key: str) -> list[str] | None:
            c = _card_for(key)
            for e in miss_v2:
                if e["card"] == c:
                    return e["missing"]
            return None

        checks = [
            ("C1 抓到重复卡片", bool(r1["detail"])),
            ("C2 抓到无 frontmatter", len(nofm) == 1),
            # --- 拦截侧：真缺陷必须仍然被抓到 ---------------------------------
            ("C2 抓到『有来源卡缺 venue』（真缺陷未被豁免带走）",
             _reported("有来源卡·缺 venue") == ["venue"]),
            ("C2 不误报『有来源卡·字段齐全』",
             _reported("有来源卡·字段齐全") is None
             and _card_for("有来源卡·字段齐全") not in exempt),
            # --- 豁免侧：author-practice 卡的来源字段不再报 ----------------------
            ("C2 豁免『经验卡·来源字段全缺』的来源字段",
             _reported("经验卡·来源字段全缺") is None
             and _card_for("经验卡·来源字段全缺") in exempt),
            # --- 豁免不许溢出（四条反例，漏洞 #10 的纪律）-----------------------
            ("C2 豁免只管来源字段：经验卡缺 title 仍报",
             _reported("经验卡·缺结构性字段 title") == ["title"]),
            # --- 结构性字段的**格式权威**：不许用别的字段名顶替（2026-09-13）--------
            ("C2 不认 `name:` 顶替 `title`（Skill 格式不等于 vault v2 格式）",
             _reported("经验卡·用 name 代替 title（不算有 title）") == ["title"]),
            ("C2 不认 `area:` 顶替 `module`",
             _reported("有来源卡·用 area 代替 module（不算有 module）") == ["module"]),
            ("C2 抓到『声明 author-practice 但有 paper_id』的矛盾",
             _card_for("经验卡·声明矛盾(带 paper_id)") in contra
             and _card_for("经验卡·声明矛盾(带 paper_id)") not in exempt),
            ("C2 抓到『声明 author-practice 但正文有 DOI』的矛盾",
             _card_for("经验卡·正文有 DOI(矛盾)") in contra),
            ("C2 不把尾部参考区的 arXiv 当矛盾（漏洞 #11）",
             _card_for("经验卡·参考区有 arXiv(不算矛盾)") not in contra
             and _card_for("经验卡·参考区有 arXiv(不算矛盾)") in exempt),
            ("C2 不把文档来源 `source: human+ai` 当论文来源（否则经验卡会被判假矛盾）",
             _reported("经验卡·带文档来源 source:human+ai") is None
             and _card_for("经验卡·带文档来源 source:human+ai") in exempt
             and _card_for("经验卡·带文档来源 source:human+ai") not in contra
             and not card_has_paper_source(*samples["经验卡·带文档来源 source:human+ai"])),
            ("矛盾存在时 C2 判 critical（反后门红线是红的）", r2["critical"] is True),
            # --- C4 ----------------------------------------------------------
            ("C4 抓到未闭合围栏", any("未闭合" in i["type"] for i in r4["detail"])),
            ("C4 抓到伪代码标错", any("伪代码" in i["type"] for i in r4["detail"])),
            # --- X3(a)：--json-out 必须能自己建父目录 --------------------------
            ("--json-out 在父目录不存在时不崩、且文件可被 json.load 读回",
             _selftest_json_out(root)),
        ]

        # --- C10：验证声明的时间锚点（PHASE6 F1）---------------------------
        # 构造样本 + **注入**假 git 日期与假门禁产物 —— 两项都不可能真的造出来。
        # 样本设计为「每条判据各有其对应反例」，而不是一条大样本测全部：
        # 一条大样本全绿会掩盖「判据恒真」，而判据恒真是本仓库反复踩到的假绿灯形态。
        _GREEN = {"G1": {"passed": True, "red_codes": []},
                  "G2": {"passed": True, "red_codes": []},
                  "G3": {"passed": True, "red_codes": []}}
        _G2A_RED = {"G1": {"passed": True, "red_codes": []},
                    "G2": {"passed": False, "red_codes": ["G2-UNSOURCED-METRIC"]},
                    "G3": {"passed": True, "red_codes": []}}
        _G2B_RED = {"G1": {"passed": True, "red_codes": []},
                    "G2": {"passed": False, "red_codes": ["G2-QUOTE-FABRICATED"]},
                    "G3": {"passed": True, "red_codes": []}}

        def _mk(name: str, claim: str, at: str | None) -> str:
            body = f"---\ntitle: {name}\nverified_by: {claim}\n"
            if at is not None:
                body += f"verified_at: {at}\n"
            body += "---\n\n# 样本\n"
            (root / "vault" / "03-C10").mkdir(parents=True, exist_ok=True)
            (root / "vault" / "03-C10" / f"{name}.md").write_text(body, encoding="utf-8")
            return f"vault/03-C10/{name}.md"

        c_noanchor = _mk("Skill-C10-NoAnchor", "gate_check.py G2 passed", None)
        c_future = _mk("Skill-C10-Future", "gate_check.py G2 passed", "2099-01-01")
        c_predate = _mk("Skill-C10-Predate", "gate_check.py G2 passed", "2020-01-01")
        c_contra_g2 = _mk("Skill-C10-ContraG2", "gate_check.py G2 passed", "2026-09-12")
        c_quote_only = _mk("Skill-C10-QuoteOnly", "quote_check.py（逐字核验 VERBATIM）",
                           "2026-09-12")
        c_quote_fab = _mk("Skill-C10-QuoteFab", "quote_check.py（逐字核验 VERBATIM）",
                          "2026-09-12")
        c_clean = _mk("Skill-C10-Clean", "gate_check.py G2 passed + quote_check.py VERBATIM",
                      "2026-09-12")
        # 实测原型的复刻：点名 gate_check 但没写判了哪个门禁（真实命中 2 张）
        c_unspec = _mk("Skill-C10-Unspecific",
                       "verify_skill_code.py（K1）+ quote_check.py（…VERBATIM）+ gate_check.py + 人工抽检",
                       "2026-09-12")
        # 「无锚点」且「矛盾」同时成立 —— 专为 total 去重那条用例而设
        c_both = _mk("Skill-C10-Both", "gate_check.py G2 passed", None)

        r10 = c10_verified_claims(
            dates={c_predate: {"first": "2026-01-01", "last": "2026-01-01"},
                   c_future: {"first": "2026-01-01", "last": "2026-01-01"}},
            gates={c_contra_g2: _G2A_RED, c_quote_only: _G2A_RED,
                   c_quote_fab: _G2B_RED, c_clean: _GREEN, c_noanchor: _GREEN,
                   c_unspec: _G2A_RED, c_both: _G2A_RED},
            today="2026-09-13")
        d10 = r10["detail"]
        miss10 = {e["card"] for e in d10["anchor_missing"]}
        bad10 = {e["card"]: e["why"] for e in d10["bad_anchor"]}
        con10 = {e["card"]: e["layer"] for e in d10["claim_contradicted"]}
        uns10 = {e["card"] for e in d10["claim_unspecific"]}

        checks += [
            ("C10 抓到『有 verified_by 无 verified_at』且不误判为矛盾",
             c_noanchor in miss10 and c_noanchor not in con10),
            ("C10 抓到『锚点晚于今天』（声称了一个还没发生的验证）",
             "晚于今天" in bad10.get(c_future, "")),
            ("C10 抓到『锚点早于卡片首次提交』（卡还不存在就验过）",
             "早于该卡首次提交" in bad10.get(c_predate, "")),
            ("C10 抓到『声称 G2 而 G2 实测为红』",
             con10.get(c_contra_g2) == "G2"),
            # --- 反后门：判据的适用范围不许被默认成全体 ----------------------
            # 实测原型：9 张卡的 quote_check 至今为真，只因 G2a 变红 ⇒ **不得**报矛盾。
            ("C10 不把 G2a 的红算成引文声明的矛盾（否则报 9 条假矛盾）",
             c_quote_only not in con10),
            # 反向：引文层自己红了，就必须报 —— 否则上面那条会退化成「永不报」
            ("C10 抓到『声称 quote_check 而引文层伪造』",
             con10.get(c_quote_fab) == "G2b（逐字引文）"),
            # --- 第三档：点名了工具但没说判了什么 ⇒ 不可复核，且**不得**报成矛盾 ---
            # 卡并没有断言 G2 通过；报矛盾＝我们替它造了一个它没做过的声明。
            ("C10 抓到『点名 gate_check 未写门禁』并单列（不报成矛盾）",
             c_unspec in uns10 and c_unspec not in con10),
            ("C10 一致样本零告警（判据不是恒真）",
             c_clean not in miss10 and c_clean not in bad10 and c_clean not in con10
             and c_clean not in uns10 and d10["consistent"] >= 1),
            # total 必须去重：一张卡可同时「无锚点」且「矛盾」，不得相加。
            # ⚠️ 这里必须把**全部五个桶**都加进去。第一版漏了 `claim_unspecific`，
            #    于是加了该桶之后「朴素和」恰好等于去重值，用例当场报红 ——
            #    **是断言漏了一个桶，不是检查坏了**（与「判据没错、作用域错了」同族）。
            ("C10 total 去重（五项相加会把 9 张报成 %d）"
             % (len(d10["anchor_missing"]) + len(d10["bad_anchor"])
                + len(d10["claim_contradicted"]) + len(d10["claim_unspecific"])
                + d10["consistent"]),
             d10["total_with_claim"] == 9
             and (len(d10["anchor_missing"]) + len(d10["bad_anchor"])
                  + len(d10["claim_contradicted"]) + len(d10["claim_unspecific"])
                  + d10["consistent"])
             > d10["total_with_claim"]),
            # critical 语义：欠账不得把红线抬起来，否则红线会被欠账淹没
            # ⚠️ 必须在**只含这一张卡**的隔离目录里跑：第一版直接复用上面的 root，
            #    而那里还躺着 `Skill-C10-Future`（2099-01-01）⇒ critical 被它
            #    合理地抬成 True，用例报红。**是断言写错了，不是检查错了。**
            #    与「变异测试台复制到 /tmp 后 REPO_ROOT 解析错」同族：
            #    判据没错，**作用域**错了。
            ("C10 只有『无锚点』时 critical=False（欠账≠撒谎）", _c10_isolated_noanchor(root)),
            ("C10 存在矛盾时 critical=True（红线是真的）", r10["critical"] is True),
        ]

    # --- 交叉验证：C2 的豁免判据是否与 gate_check 同口径 -------------------
    gc, why = _load_gate_check_for_crosscheck()
    if gc is None:
        print(f"  ⚠️ 跳过与 gate_check 的交叉验证（{why}）—— 不算失败")
    else:
        diffs = []
        for name, (text, fm) in samples.items():
            mine = card_has_paper_source(text, fm)
            theirs = gc.card_has_paper_source(text, fm)
            if mine != theirs:
                diffs.append(f"{name}: 本脚本={mine} gate_check={theirs}")
            if card_declares_author_practice(fm) != (
                    (fm.get("evidence_basis") or fm.get("provenance") or ""
                     ).strip().lower() in gc.AUTHOR_PRACTICE_BASIS):
                diffs.append(f"{name}: author-practice 声明判据不一致")
        checks.append(("与 gate_check 的 G2 判据逐样本一致"
                       + (f"（{'; '.join(diffs)}）" if diffs else ""), not diffs))

    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")
        ok &= passed
    print("✅ 自检通过：仓库体检的检查项真的会报警，且豁免没有把拦截一起关掉"
          if ok else "❌ 自检失败：检查静默失效或豁免溢出")
    return 0 if ok else 1


def _selftest_json_out(root: Path) -> bool:
    """X3(a) 的回归网：写一个**父目录不存在**的 json_out，必须成功且可读回。"""
    target = root / "deep" / "not" / "there" / "repo_health.json"
    if target.parent.exists():
        return False                                  # 样本前提不成立 → 测不到东西
    try:
        write_json_report(target, {"reports": [{"id": "C2"}]})
    except OSError:
        return False
    try:
        return json.loads(target.read_text(encoding="utf-8"))["reports"][0]["id"] == "C2"
    except (OSError, ValueError, KeyError, IndexError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="paper2skills 仓库体检")
    ap.add_argument("--only", action="append", choices=sorted(CHECKS), default=[])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    names = args.only or sorted(CHECKS)
    reports = [CHECKS[n]() for n in names]

    ts = datetime.now().isoformat(timespec="seconds")
    print(f"{'='*74}\npaper2skills 仓库体检 · {ts}\n{'='*74}")
    for r in reports:
        flag = "🔴" if r["critical"] else ("🟡" if r.get("total") else "✅")
        print(f"\n{flag} {r['id']} {r['name']}：{r['summary']}")
        d = r["detail"]
        if isinstance(d, dict):
            for k, v in d.items():
                if v:
                    print(f"     {k}: {json.dumps(v, ensure_ascii=False)[:220]}")
        elif d:
            for item in d[:8]:
                if isinstance(item, dict):
                    print(f"     {json.dumps(item, ensure_ascii=False)[:210]}")
                else:
                    print(f"     {item}")
            if len(d) > 8:
                print(f"     …另有 {len(d) - 8} 条")
    for r in reports:
        for s in r.get("stale", []):
            print(f"\n⚠️  {s}")

    crit = [r["id"] for r in reports if r["critical"]]
    print(f"\n{'='*74}")
    print(f"结论：{'❌ CRITICAL: ' + ', '.join(crit) if crit else '✅ 无 CRITICAL'}")
    if args.json_out:
        write_json_report(args.json_out, {"generated_at": ts, "reports": reports})
        print(f"→ {args.json_out}")
    return 1 if crit else 0


if __name__ == "__main__":
    sys.exit(main())
