#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gate_check.py — paper2skills 三合一门禁（G1 代码 / G2 事实 / G3 业务）

设计依据（2026-09 调研，全部有一手数字）：
  * Cited but Not Verified (arXiv:2605.06635)：把「事实溯源」拆成三维后发现
    —— 链接可用 >94%、主题相关 >80%，而**事实一致性只有 39–77%**。
    且工具调用次数从 2 增到 150 时，事实一致性平均再降 42%（GPT-5.4 从 79%→17%）。
    结论：三个维度**必须分开报**，合成一个「可信度总分」会把最弱的那维平均掉。
  * ResearchCodeBench 附录 G：新研究代码失败中 58.6% 是**语义错误**（能跑但算错）。
  * AutoReproduce：无执行闭环时执行率仅 17.94% —— 所以 G1 必须是**执行凭证**，不是打分。
  * RQ-Bench「novelty mirage」：LLM 独立打分对「新颖性/有用性」系统性虚高，
    故 G2/G3 只做**机械可复核的计数**，不产出主观分数。

三个门禁各自产出独立 JSON，全绿才允许同步（见 paper-同步/scripts/sync.py）。

-------------------------------------------------------------------------------
G1 代码可执行  gate_g1.json   ← 读取 K1 (verify_skill_code.py) 的产物
   PASS  卡片代码至少一个单元真正跑通
   FAIL  无代码 / K1 判 FAIL 或 ORPHAN_DEP
   凭证  k1_full.json / 单卡 K1 输出

G2 事实可溯源  gate_g2.json
   做法  把卡片散文中带「度量语义」的数字逐一取出（代码块除外），
         要求它能在**证据链**中找到出处：
           ① 卡片内 `> 原文："..."` 引用块
           ② 同目录 evidence.md 的「数字 → 出处」表
           ③ frontmatter 的 paper/paper_id 指向的论文（须配 ① 或 ② 才算数）
   分级  RED    高价值断言（%/倍/pp/美元/ROI/提升/降低/准确率…）无出处 → 阻塞
         YELLOW 一般数字无出处 → 警告（不阻塞，但计入债务）
         GREEN  有出处
   ⚠️ 本门禁不做任何「这个数字对不对」的判断 —— 它只回答
      「这个数字有没有可点击的出处」。对错靠人工抽检（见 T5-3）。

G3 业务可落地  gate_g3.json
   检查  ① 业务场景段是否落到母婴出海的具体动作上（而非「提升效率」这类空话）
         ② 是否声明「数据要求 / 企业内是否可得」（T2-1 新增必填行）
         ③ ROI 是否有计算依据（公式或参数），而非只给一个数
         ④ 是否与 ≥2 张已有卡片建立关系
   依据  agentskills.io 官方点名的陷阱：不喂领域上下文 → 产出空泛通用流程。

用法
----
  python3 gate_check.py --all
  python3 gate_check.py --card ../../paper2skills-vault/13-广告分析/Skill-X.md
  python3 gate_check.py --all --k1 paper2skills-research/data/verification/k1_full.json
  python3 gate_check.py --all --outdir ../../paper2skills-vault/07-资源库/gates
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if os.environ.get("PAPER2SKILLS_ROOT"):
    REPO_ROOT = Path(os.environ["PAPER2SKILLS_ROOT"]).resolve()
VAULT = REPO_ROOT / "paper2skills-vault"

# ---------------------------------------------------------------------------
# G2：数字分类词典
# ---------------------------------------------------------------------------
# 高价值断言：带度量语义 —— 这类数字被写进卡片就是为了支撑结论，必须有出处
METRIC_UNITS = (
    "%", "％", "pp", "个百分点", "倍", "x", "X",
    "美元", "美金", "元", "万元", "亿元", "$",
    "ROI", "roi", "提升", "降低", "下降", "增长", "减少", "提高",
    "准确率", "召回", "精确率", "F1", "AUC", "RMSE", "MAE", "MAPE", "SMAPE",
    "转化率", "点击率", "CTR", "CVR", "留存", "复购", "流失",
    "GMV", "客单价", "毛利", "净利", "成本", "库存周转", "缺货率",
)
METRIC_WORDS = (
    "提升", "降低", "下降", "增长", "减少", "提高", "优于", "超过", "达到",
    "节省", "降低到", "从", "倍", "百分点",
)

# 低价值数字：结构/版本/年份/编号 —— 不要求出处
NOISE_PATTERNS = (
    # ⚠️ 这里原本是 r"^v?\d+\.\d+(\.\d+)?$"，本意是「版本号/章节号」，
    # 但它把**所有小数**一律豁免 —— 包括 `92.2%`、`0.55`、`3.14`。
    # 实测表现为：`误差下降 92.2%` 这种最典型的高价值断言直接不进 G2 检查。
    # 现在只认显式版本号（必须带 v），章节号改由「结构前缀」规则处理（见 is_noise）。
    r"^v\d+(\.\d+)+$",                # 版本号 v1.2.3
    r"^(19|20)\d{2}$",                # 年份
    # arXiv ID / DOI 是**标识符**，不是事实断言。不加这条会把每条
    # `> 出处：2606.26690 §4.2` 都报成「无出处的度量数字」（实测 4 条假红灯）。
    r"^\d{4}\.\d{4,5}(v\d+)?$",         # arXiv: 2606.26690
    r"^10\.\d{4,9}?$",               # DOI 前缀 10.1287
    # ⚠️ 这里原本还有 r"^\d{1,2}$"（当时的想法是「章节号/序号」）。
    # 但它会把**所有 1–2 位数字**一律豁免 —— 包括 `转化率提升 15%`、`留存 30%`，
    # 而百分比恰恰是最高价值、最需要出处的断言类型。实测这等于给
    # 「两位数以内的数字可以随便写」开了后门，故删除。
    # 章节号/序号由 `^v?\d+\.\d+` （如 4.2）与「无单位单字符」两条规则覆盖。
)

NUM_TOKEN_RE = re.compile(
    r"(?<![\w.])"
    r"(\d+(?:\.\d+)?)"
    r"\s*"
    r"(%|％|pp|个百分点|倍|美元|美金|万元|亿元|元|\$|天|周|月|年|小时|分钟|秒|ms|"
    r"人|单|张|条|个|次|GB|MB|KB|万|亿)?"
)

CODE_FENCE_RE = re.compile(r"^```.*?^```", re.M | re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
QUOTE_RE = re.compile(r"^>\s*(?:原文\s*[:：]\s*)?[\"“「『](.+?)[\"”」』]\s*$", re.M | re.S)  # 与 quote_check.py 保持一致

# ---------------------------------------------------------------------------
# G3：空话黑名单（业务场景段出现这些且无具体动作 → 判空泛）
# ---------------------------------------------------------------------------
VAGUE_PHRASES = (
    "提升效率", "提高效率", "优化流程", "赋能", "降本增效", "提升竞争力",
    "更好地", "有效地", "显著提升", "全面提升", "助力", "抓手", "闭环打法",
    "提升用户体验", "增强能力", "打造体系",
)
# 具体动作信号（母婴出海语境）
CONCRETE_SIGNALS = (
    "吸奶器", "纸尿裤", "奶瓶", "婴儿", "母婴", "奶粉", "辅食", "童装", "玩具",
    "Amazon", "亚马逊", "独立站", "Shopify", "TikTok", "Temu", "SHEIN",
    "海外仓", "头程", "尾程", "FBA", "清关", "关税", "CE", "FDA", "CPSC",
    "Listing", "ASIN", "SKU", "广告位", "竞价", "ROAS", "ACOS", "秒杀", "优惠券",
    "复购", "退货", "客诉", "差评", "选品", "测款", "补货", "备货", "断货",
)
DATA_AVAIL_RE = re.compile(
    r"(数据要求|数据可得|数据来源|所需数据|企业内可得|数据字段|输入数据)", re.M
)
ROI_FORMULA_RE = re.compile(r"(ROI|收益|回报|节省|增益).{0,80}?[=＝].{0,80}", re.S)
SKILL_REF_RE = re.compile(r"(Skill-[A-Za-z0-9\u4e00-\u9fff\-_]+)")


@dataclass
class Finding:
    level: str          # RED | YELLOW | GREEN | INFO
    code: str
    message: str
    evidence: str = ""


def rel_to_repo(p: Path) -> str:
    """安全的相对路径显示（卡片路径可能是相对 CWD 的）。"""
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


@dataclass
class GateResult:
    gate: str
    target: str
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, evidence: str = "") -> None:
        self.findings.append(Finding(level, code, message, evidence))


# ---------------------------------------------------------------------------
# 文本预处理
# ---------------------------------------------------------------------------
def strip_code(text: str) -> str:
    """去掉代码围栏与行内代码 —— 代码里的数字不是「事实断言」。"""
    t = CODE_FENCE_RE.sub("\n[[CODE]]\n", text)
    t = INLINE_CODE_RE.sub(" [[CODE]] ", t)
    return t


def parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _extract_quotes(text: str) -> list[str]:
    """抽取卡片中的引用块文本。

    ⚠️ 必须与 `quote_check.py` 用**同一个抽取器**。历史上这里另有一套正则
    （只认紧跟 `>` 的引号），而 v2 规范写的是 `> 原文："..."`，两者不匹配，
    导致「引文本身的判真」与「引文能否充当出处」两个环节看到的引用集合不同 ——
    合格引文被当成不存在，卡片被判「无出处」假红灯。
    现在统一：优先调用 quote_check 的抽取器，取不到才退回本地正则。
    """
    qc = _load_quote_check()
    if qc is not None:
        try:
            return [q["quote"] for q in qc.extract_quotes(text)]
        except Exception:
            pass
    return [m.group(1) for m in QUOTE_RE.finditer(text)]


def collect_evidence(card: Path, text: str,
                     trusted_quotes: set[str] | None = None) -> tuple[set[str], list[str]]:
    """收集证据链：卡片内引用块 + 同目录 evidence.md 的数字与出处。

    `trusted_quotes` 给出**允许充当出处**的引用文本集合（由 G2b 逐字核验产出）。
    传 None 表示不做甄别（离线调用或核验器不可用）；
    传集合时，**不在集合内的引用一律不计入出处** —— 这是防止
    「伪造引文洗白数字」的关键闸门。
    """
    nums: set[str] = set()
    sources: list[str] = []

    for src in _extract_quotes(text):
        if trusted_quotes is not None and src not in trusted_quotes:
            # 未通过逐字核验的引用不计入出处（G2b 已单独报警）
            continue
        sources.append(f"卡片引用块: {src[:80]}")
        for n in re.findall(r"\d+(?:\.\d+)?", src):
            nums.add(n)

    # 同目录 / 同名 evidence.md
    #
    # ⚠️ 这里曾经是一个洗白后门：evidence.md 里的数字**无条件**全收，
    # 于是「在 evidence.md 里裸写一句 `ROI 提升 42.7%`」就能让卡片正文的
    # 无出处数字过闸。现在只认两种：
    #   ① 位于**已逐字核验**的引用块内；
    #   ② 位于带显式出处指针的行（含 `出处` 或 `§`），用于引用表格/图号。
    for cand in (
        card.with_suffix(".evidence.md"),
        card.parent / "evidence.md",
        card.parent / f"{card.stem}.evidence.md",
    ):
        if cand.is_file():
            ev = cand.read_text(encoding="utf-8", errors="replace")
            sources.append(f"evidence.md: {cand.relative_to(REPO_ROOT)}")
            for src in _extract_quotes(ev):
                if trusted_quotes is not None and src not in trusted_quotes:
                    continue
                for n in re.findall(r"\d+(?:\.\d+)?", src):
                    nums.add(n)
            for line in ev.splitlines():
                if "出处" in line or "§" in line:
                    for n in re.findall(r"\d+(?:\.\d+)?", line):
                        nums.add(n)
    return nums, sources


# 结构前缀：紧邻数字左侧出现这些标记时，该数字是**指代**（章节/图表/公式号），
# 不是事实断言。用「前缀」而不是「数字形态」来判定，是因为 `4.2` 既可能是
# 「§4.2」也可能是「误差 4.2%」——只看数字本身无法区分，必须看上下文。
STRUCTURAL_PREFIX_RE = re.compile(
    r"(§|第|节|章|图|表|式|注|Section|Sec\.|Figure|Fig\.|Table|Eq\.?|Equation|Appendix)\s*$",
    re.IGNORECASE,
)


def is_noise(num: str, unit: str, prefix: str = "") -> bool:
    """判断数字是否为「结构性编号」而非事实断言。

    `prefix` 是数字左侧若干字符，用于识别 §4.2 / 图 3 / 式 (7) 这类指代。
    """
    if prefix and STRUCTURAL_PREFIX_RE.search(prefix):
        return True
    for pat in NOISE_PATTERNS:
        if re.match(pat, num):
            return True
    # 裸单个数字（无单位）几乎都是编号/序号/系数，不构成「事实断言」
    if not unit and len(num) <= 1:
        return True
    return False


def is_metric(context: str, num: str, unit: str) -> bool:
    """判断一个数字是否带「度量语义」—— 带则必须有出处。"""
    if unit and unit.strip() in ("%", "％", "pp", "个百分点", "倍", "美元", "美金",
                                 "$", "万元", "亿元"):
        return True
    window = context
    for w in METRIC_UNITS:
        # 单位词需与数字同句（±30 字符窗口由调用方保证）
        if w in window and w not in ("元",):
            return True
    if unit == "元" and re.search(r"(省|成本|价格|客单|毛利|净利|预算|花费)", window):
        return True
    return False


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------
def gate_g1(card: Path, text: str, k1_index: dict) -> GateResult:
    rel = rel_to_repo(card)
    g = GateResult("G1", rel, False)
    units = [v for k, v in k1_index.items() if k.startswith(str(card.name))
             or k.startswith(rel) or card.name in k]

    if not units:
        # 无 K1 记录 → 现场快速判定（L2 编译级别），并明确标注证据强度
        has_code = bool(re.search(r"^```(?:python|py)\s*$", text, re.M | re.I))
        if not has_code:
            g.add("RED", "G1-NO-CODE", "卡片不含 python 代码块")
            return g
        g.add("RED", "G1-NO-EVIDENCE",
              "无 K1 验证凭证 —— 请先运行 verify_skill_code.py，不得凭人工判断放行")
        return g

    for u in units:
        v = u.get("verdict")
        if v == "PASS":
            g.passed = True
            g.add("GREEN", "G1-EXEC-OK", f"{u['target']} 真实执行通过",
                  evidence=f"L4/L5 通过，耗时见 K1 报告")
        elif v == "ORPHAN_DEP":
            g.add("RED", "G1-ORPHAN",
                  f"{u['target']} 引用了仓库内不存在的本地模块: {u.get('orphan_deps')}")
        elif v == "FAIL":
            bad = next((s for s in u.get("stages", []) if s.get("passed") is False), None)
            g.add("RED", "G1-EXEC-FAIL",
                  f"{u['target']} 执行失败: {bad.get('detail') if bad else '未知'}")
        elif v == "ENV_BLOCKED":
            g.add("YELLOW", "G1-ENV",
                  f"{u['target']} 环境阻塞（缺依赖），未验证 —— 缺依赖不等于正确")
    g.metrics = {"units": len(units),
                 "pass": sum(1 for u in units if u.get("verdict") == "PASS")}
    return g


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------
# 证据语法行：`> 原文："..."` 与 `> 出处：...` 是**证据本身**，不是待取证的断言。
# 不剥离它们会有两个后果：①出处的 arXiv ID 被当成度量数字（假红灯）；
# ②引文里的数字被重复算成「正文断言」。
# 这不构成绕过：伪造的引文会在 G2b 被逐字核验判 FABRICATED（见 quote_check.py）。
EVIDENCE_LINE_RE = re.compile(r"^\s*>\s*(原文|出处)\s*[:：].*$", re.M)


def strip_evidence(text: str) -> str:
    return EVIDENCE_LINE_RE.sub("\n", text)


def gate_g2(card: Path, text: str) -> GateResult:
    rel = rel_to_repo(card)
    g = GateResult("G2", rel, True)
    # 先剥代码（代码里的数字不是事实断言），再剥证据语法行（引文是证据不是断言）
    prose = strip_evidence(strip_code(text))
    fm = parse_frontmatter(text)

    # --- G2b 先跑：逐字核验引用块 ---------------------------------------
    # ⚠️ 顺序很关键：必须**先**判定哪些引用是真的，**再**决定它们能否充当出处。
    # 否则伪造一段带数字的引文就能让任意数字过闸（见 quote_check.py 文档）。
    qc = _load_quote_check()
    qrep: dict = {}
    if qc is not None:
        try:
            qrep = qc.check_card(card)
        except Exception as exc:  # 核验器自身出错不能静默放行
            g.add("RED", "G2-QUOTE-CHECK-ERROR", f"引文核验器异常: {exc}")

    fabricated = [q for q in qrep.get("quotes", []) if q.get("verdict") == "FABRICATED"]
    fuzzy = [q for q in qrep.get("quotes", []) if q.get("verdict") == "FUZZY"]
    unverifiable = [q for q in qrep.get("quotes", []) if q.get("verdict") == "UNVERIFIABLE"]
    # 只有「逐字命中」与「无法核验」（历史卡无全文存档）两类才算出处；
    # 近似与伪造一律不算 —— 否则等于承认编造的引文可以当证据。
    if qc is None:
        # ⚠️ 核验器不可用时必须**退回不甄别**（None），不能退化成空集：
        # 空集 = 「一条引用都不可信」→ 会把全库证据链清空，
        # 表现为「忽然所有卡都红灯」，掩盖真实原因（核验器坏了 vs 卡片坏了）。
        trusted_quotes = None
        g.add("YELLOW", "G2-QUOTE-CHECK-UNAVAILABLE",
              "引文逐字核验器未加载，本轮未核验引文真伪 —— 结论不完整")
    else:
        trusted_quotes = {
            q["quote"] for q in qrep.get("quotes", [])
            if q.get("verdict") in ("VERBATIM", "UNVERIFIABLE")
        }

    ev_nums, ev_sources = collect_evidence(card, text, trusted_quotes=trusted_quotes)

    claimed: list[tuple[str, str]] = []
    for m in NUM_TOKEN_RE.finditer(prose):
        num, unit = m.group(1), (m.group(2) or "")
        prefix = prose[max(0, m.start() - 12):m.start()]
        if is_noise(num, unit, prefix):
            continue
        ctx = prose[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
        if is_metric(ctx, num, unit):
            claimed.append((f"{num}{unit}", ctx.strip()))

    red, yellow, green = [], [], []
    for token, ctx in claimed:
        bare = re.match(r"[\d.]+", token).group(0)
        if bare in ev_nums:
            green.append((token, ctx))
        else:
            red.append((token, ctx))

    # 一般数字（非度量）无出处 → 黄灯债务
    for m in NUM_TOKEN_RE.finditer(prose):
        num, unit = m.group(1), (m.group(2) or "")
        prefix = prose[max(0, m.start() - 12):m.start()]
        if is_noise(num, unit, prefix):
            continue
        ctx = prose[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
        if not is_metric(ctx, num, unit) and num not in ev_nums:
            yellow.append((f"{num}{unit}", ctx.strip()))

    # 去重：同一 (数字, 上下文) 只算一条 —— 避免重复计数虚增债务
    red = list(dict.fromkeys(red))
    yellow = list(dict.fromkeys(yellow))
    green = list(dict.fromkeys(green))

    g.metrics = {
        "metric_numbers": len(claimed),
        "sourced": len(green),
        "unsourced_metric": len(red),
        "unsourced_general": len(yellow),
        "evidence_sources": len(ev_sources),
        "has_paper_field": bool(fm.get("paper") or fm.get("paper_id")),
        "traceability_pct": round(len(green) / len(claimed) * 100, 1) if claimed else None,
        # --- G2b 引文逐字核验 ---
        "quotes_total": len(qrep.get("quotes", [])),
        "quotes_verbatim": qrep.get("n_verbatim"),
        "quotes_fuzzy": qrep.get("n_fuzzy"),
        "quotes_fabricated": qrep.get("n_fabricated"),
        "quotes_spliced": qrep.get("n_spliced"),
        "quotes_unverifiable": len(unverifiable),
        "quote_verdict": qrep.get("verdict"),
        "fulltext_archived": bool(qrep.get("fulltext")),
    }

    # --- G2b 报警 -------------------------------------------------------
    for q in fabricated[:10]:
        tag = "（拼接：论文里不存在这句话，碎片分别来自不同段落）" if q.get("spliced") else ""
        g.add("RED", "G2-QUOTE-FABRICATED",
              f"⚠️ 引文在论文全文中找不到{tag}",
              evidence=f"连续度={q.get('longest_run_ratio')} "
                       f"n-gram召回={q.get('ngram_recall')}｜{q['quote'][:100]}")
    if len(fabricated) > 10:
        g.add("RED", "G2-QUOTE-TRUNCATED",
              f"另有 {len(fabricated) - 10} 条引文无法在原文中找到（已截断展示）")
    for q in fuzzy[:10]:
        g.add("YELLOW", "G2-QUOTE-FUZZY",
              "引文与原文仅有部分连续匹配，需人工复核是否为改写",
              evidence=f"连续度={q.get('longest_run_ratio')}｜{q['quote'][:100]}")
    if unverifiable:
        g.add("YELLOW", "G2-QUOTE-UNVERIFIED",
              f"{len(unverifiable)} 条引文无法核验（未找到该论文的全文存档）—— "
              f"「无全文」不等于「引文为真」",
              evidence=f"paper_id={fm.get('paper_id', '(缺)')}")
    if qrep.get("verdict") == "VERBATIM" and qrep.get("n_quotes"):
        g.add("GREEN", "G2-QUOTE-VERBATIM",
              f"{qrep['n_quotes']} 条引文全部逐字命中论文全文")

    if not ev_sources:
        g.add("RED", "G2-NO-EVIDENCE-CHAIN",
              "卡片内既无 `> 原文：\"...\"` 引用块，也无 evidence.md —— 无任何可追溯的出处")
    for token, ctx in red[:40]:
        g.add("RED", "G2-UNSOURCED-METRIC",
              f"高价值断言 `{token}` 无出处", evidence=f"…{ctx}…")
    if len(red) > 40:
        g.add("RED", "G2-TRUNCATED", f"另有 {len(red) - 40} 条高价值断言无出处（已截断展示）")
    for token, ctx in yellow[:10]:
        g.add("YELLOW", "G2-UNSOURCED-GENERAL", f"一般数字 `{token}` 无出处",
              evidence=f"…{ctx}…")
    if green:
        g.add("GREEN", "G2-SOURCED", f"{len(green)} 个高价值断言有出处")

    # 只有「高价值断言全部有出处」**且**「无伪造引文」才算 G2 绿。
    # 必须按 findings 统一判定：早期版本写死 `len(red) == 0`，
    # 会在 G2b 追加红灯后仍报绿（门禁 bug，已修）。
    g.passed = not any(f.level == "RED" for f in g.findings)
    return g


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------
def gate_g3(card: Path, text: str) -> GateResult:
    rel = rel_to_repo(card)
    g = GateResult("G3", rel, True)
    prose = strip_code(text)

    concrete = sorted({s for s in CONCRETE_SIGNALS if s in prose})
    vague = sorted({s for s in VAGUE_PHRASES if s in prose})

    g.metrics = {
        "concrete_signals": len(concrete),
        "vague_phrases": len(vague),
        "has_data_requirement": bool(DATA_AVAIL_RE.search(prose)),
        "has_roi_basis": bool(ROI_FORMULA_RE.search(prose)),
        "skill_relations": len(set(SKILL_REF_RE.findall(text))),
    }

    if len(concrete) < 3:
        g.add("RED", "G3-NOT-CONCRETE",
              f"仅命中 {len(concrete)} 个母婴出海具体信号（需 ≥3）—— 场景过于泛化",
              evidence=f"命中: {concrete or '无'}")
    else:
        g.add("GREEN", "G3-CONCRETE", f"命中 {len(concrete)} 个具体业务信号: {', '.join(concrete[:8])}")

    if vague and len(vague) >= 3:
        g.add("RED", "G3-VAGUE",
              f"出现 {len(vague)} 处空泛表述（agentskills.io 点名的失败模式）",
              evidence=f"{', '.join(vague)}")
    elif vague:
        g.add("YELLOW", "G3-SOME-VAGUE", f"出现空泛表述: {', '.join(vague)}")

    if not g.metrics["has_data_requirement"]:
        g.add("RED", "G3-NO-DATA-REQ",
              "未声明「数据要求 / 企业内是否可得」（v2 模板必填行）")
    if not g.metrics["has_roi_basis"]:
        g.add("YELLOW", "G3-NO-ROI-BASIS", "ROI 缺少计算依据（公式或参数）")
    if g.metrics["skill_relations"] < 2:
        g.add("YELLOW", "G3-FEW-RELATIONS",
              f"仅关联 {g.metrics['skill_relations']} 张卡片（要求 ≥2）")

    g.passed = not any(f.level == "RED" for f in g.findings)
    return g


# ---------------------------------------------------------------------------
def collect_cards() -> list[Path]:
    return sorted(p for p in VAULT.rglob("Skill-*.md") if "_superseded" not in p.parts)


def load_k1(path: Path | None) -> dict:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["target"]: r for r in data.get("results", [])}


_QUOTE_CHECK_CACHE: list = []


def _load_quote_check():
    """加载同目录的 quote_check.py（G2b 逐字核验器）。

    用 importlib 按文件路径加载，而不是 `import quote_check`：
    脚本要能在任意 cwd 下被调用（CI、子代理、绝对路径直调），
    依赖 sys.path 会出现「本地能跑、别处 ImportError」。
    """
    if _QUOTE_CHECK_CACHE:
        return _QUOTE_CHECK_CACHE[0]
    mod = None
    try:
        import importlib.util
        p = Path(__file__).resolve().with_name("quote_check.py")
        if p.is_file():
            spec = importlib.util.spec_from_file_location("_p2s_quote_check", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
    except Exception:
        mod = None
    _QUOTE_CHECK_CACHE.append(mod)
    return mod


def summarise(gates: list[GateResult], name: str) -> dict:
    passed = sum(1 for g in gates if g.passed)
    reds = sum(1 for g in gates for f in g.findings if f.level == "RED")
    yellows = sum(1 for g in gates for f in g.findings if f.level == "YELLOW")
    return {
        "gate": name,
        "cards_checked": len(gates),
        "cards_passed": passed,
        "pass_rate_pct": round(passed / len(gates) * 100, 1) if gates else 0.0,
        "red_findings": reds,
        "yellow_findings": yellows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="paper2skills G1/G2/G3 门禁")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--card", type=Path)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--k1", type=Path, default=None, help="K1 报告 JSON（verify_skill_code.py 产出）")
    ap.add_argument("--outdir", type=Path, default=None, help="三份 gate_*.json 的输出目录")
    ap.add_argument("--only", choices=["G1", "G2", "G3"], default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cards = [args.card] if args.card else collect_cards()
    k1_index = load_k1(args.k1)
    if args.k1 and not k1_index:
        print(f"⚠️  K1 报告 {args.k1} 为空或不存在 —— G1 将全部判为「无凭证」")

    g1s, g2s, g3s = [], [], []
    for c in cards:
        text = c.read_text(encoding="utf-8", errors="replace")
        if args.only in (None, "G1"):
            g1s.append(gate_g1(c, text, k1_index))
        if args.only in (None, "G2"):
            g2s.append(gate_g2(c, text))
        if args.only in (None, "G3"):
            g3s.append(gate_g3(c, text))

    summaries = []
    for name, lst in (("G1", g1s), ("G2", g2s), ("G3", g3s)):
        if not lst:
            continue
        s = summarise(lst, name)
        summaries.append(s)
        if not args.quiet:
            print(f"\n{'='*70}\n{name} 门禁：{s['cards_passed']}/{s['cards_checked']} 通过 "
                  f"({s['pass_rate_pct']}%)  红灯 {s['red_findings']}  黄灯 {s['yellow_findings']}\n{'='*70}")
            for r in sorted(lst, key=lambda x: -sum(1 for f in x.findings if f.level == "RED"))[:5]:
                reds = [f for f in r.findings if f.level == "RED"]
                if reds:
                    print(f"  ❌ {r.target}")
                    for f in reds[:3]:
                        print(f"       [{f.code}] {f.message}")

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        for name, lst in (("g1", g1s), ("g2", g2s), ("g3", g3s)):
            if not lst:
                continue
            payload = {
                "gate": name.upper(),
                "generated_at": ts,
                "summary": next(s for s in summaries if s["gate"] == name.upper()),
                "results": [{**asdict(r), "findings": [asdict(f) for f in r.findings]} for r in lst],
            }
            # 单卡模式：文件名带卡片名，便于同步脚本精确查找
            fn = (f"gate_{name}_{cards[0].stem}.json" if args.card else f"gate_{name}.json")
            (args.outdir / fn).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
            print(f"→ {args.outdir / fn}")

    all_red = sum(s["red_findings"] for s in summaries)
    return 0 if all_red == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
