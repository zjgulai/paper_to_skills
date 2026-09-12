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
    r"^0\d+$",                        # 补零 ID 片段（如 p2s-2026-0001 的 0001）
    # ⚠️ 这里原本还有 r"^\d{1,2}$"（当时的想法是「章节号/序号」）。
    # 但它会把**所有 1–2 位数字**一律豁免 —— 包括 `转化率提升 15%`、`留存 30%`，
    # 而百分比恰恰是最高价值、最需要出处的断言类型。实测这等于给
    # 「两位数以内的数字可以随便写」开了后门，故删除。
    # 章节号/序号由 `^v?\d+\.\d+` （如 4.2）与「无单位单字符」两条规则覆盖。
)

NUM_TOKEN_RE = re.compile(
    # ⚠️ 前置断言**不能写成 `(?<![\w.])`**（2026-09-12 由子代理实测发现，漏洞 #14）：
    # Python 的 `\w` 把**汉字也算单词字符**，于是「紧跟在汉字之后的数字」全部不被扫描。
    # 这个误配在本仓库的后果是**系统性低估**：
    #     提升 35%         → ['35%']    ✅ 扫到
    #     准确率92.2%      → []         ❌ 漏掉（最典型的高价值断言形态）
    #     返工50%          → []         ❌
    #     落地率40%        → []         ❌
    #     转化率提升40%     → []         ❌
    #     ROI约7-10倍      → ['10倍']   ❌ 只扫到后一个
    # 而中文卡的绝大多数断言恰恰写成「名词+数字+单位」无空格形式 ——
    # 即**最需要检查的那一类，恰好是唯一被静默跳过的那一类**。
    # 修法：只把「拉丁字母/数字/下划线」算作 token 内部字符，汉字不再阻断。
    # 这样 `型号A200` 仍被跳过（A200 是单一标识符），而 `准确率92.2%` 会被扫到。
    r"(?<![0-9A-Za-z_])"
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
# G2：证据基础分类（决定「该修」还是「修不了」）
# ---------------------------------------------------------------------------
# `evidence_basis` 取值（写进 frontmatter，与 provenance_audit.py 的判定对齐）：
#   paper-verbatim   有论文来源，且已有逐字引文        → 按有来源卡审查
#   paper-traceable  有论文来源，尚未补逐字引文        → 按有来源卡审查（会红，且**应该**红）
#   author-practice  无论文来源（作者经验/行业实践）   → UNVERIFIABLE，不阻塞
#   mixed            多篇论文 + 经验混写               → 按有来源卡审查
#
# ⚠️ `author-practice` 是**待核验的声明**，不是免检令牌：
# 若卡片里同时存在 arXiv/DOI，判 G2-BASIS-CONTRADICTION 红灯（见 card_has_paper_source）。
# 否则「加一行 frontmatter」就成了全库洗白手段 —— 与本项目已封堵的 7 个后门同类。
AUTHOR_PRACTICE_BASIS = {"author-practice", "practice", "experience"}
ARXIV_ANY_RE = re.compile(r"\b(?:arXiv\s*[:：]?\s*)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.I)
DOI_ANY_RE = re.compile(r"\b10\.\d{4,9}/[^\s（()\[\]\"'<>]+")

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
    # PASS | FAIL | UNVERIFIABLE
    #
    # ⚠️ 为什么需要第三个结局（2026-09-12，G2 根因修复）：
    # 原设计只有「通过 / 红灯」两态，于是「这张卡**从设计上就没有论文来源**」
    # 被记成「这张卡有缺陷」—— 两者**补救动作相反**：前者要改判定口径，
    # 后者要改卡片。混为一谈会让 56 张人写的经验卡永远挂在红灯队列里，
    # 而基线数字里「通过率 11%」同时混进了两类完全不同的东西。
    # 与 K1 的 `ORPHAN_DEP` → `MIGRATED_DEP` 修正是同一个错，只是发生在另一层。
    outcome: str = "PASS"

    def add(self, level: str, code: str, message: str, evidence: str = "") -> None:
        self.findings.append(Finding(level, code, message, evidence))


# ---------------------------------------------------------------------------
# 文本预处理
# ---------------------------------------------------------------------------
def strip_code(text: str) -> str:
    """只去掉**代码围栏**，**保留行内代码**。

    ⚠️ 早期版本连行内代码一起去掉（`INLINE_CODE_RE`），实测形成一个洗白后门：
    把论文事实数字写成 `` `2–3 倍` `` 即可绕过 G2 —— 而反引号是写代码标识符的
    顺手习惯，作者未必有规避意图，但后果是该断言完全不进检查。
    代码围栏保留豁免（那是 MasterPrompt v2.1 明确的 B 类「本地可复现数字」约定），
    行内代码不再豁免。为此补两条标识符噪声规则（纯补零 ID、年份已覆盖）。
    """
    return CODE_FENCE_RE.sub("\n[[CODE]]\n", text)


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

    # 证据链 evidence.md 的定位。
    #
    # ⚠️ 这里曾经是一个洗白后门：evidence.md 里的数字**无条件**全收，
    # 于是「在 evidence.md 里裸写一句 `ROI 提升 42.7%`」就能让卡片正文的
    # 无出处数字过闸。现在只认两种：
    #   ① 位于**已逐字核验**的引用块内；
    #   ② 位于带显式出处指针的行（含 `出处` 或 `§`），用于引用表格/图号。
    #
    # ⚠️⚠️ 2026-09-12 修第二个缺陷：**门禁根本找不到 evidence.md**。
    # 原实现只在「卡片同目录」找，而 MasterPrompt-v2 与 registry 的约定是
    # `papers/<域>/<p2s-id>/evidence.md`。后果是 PHASE3 全部新卡的
    # evidence.md 对 G2 一点作用都没有 —— 上面那个「只认两种」的加固成了空转，
    # 而 G2 绿灯实际只靠卡片自身 ⑥ 段引文。
    # （由子代理在 p2s-2026-0018 上发现：它观察到 evidence_sources 全是「卡片引用块」。）
    # 现按 paper_id 反查 registry 的 outputs.evidence，并兜底扫 papers/*/<paper_id>/。
    ev_candidates: list[Path] = [
        card.with_suffix(".evidence.md"),
        card.parent / "evidence.md",
        card.parent / f"{card.stem}.evidence.md",
    ]
    # ⚠️ 注意 `paper_id` 在两个地方**语义不同**，这是本仓库的一个命名坑：
    #   卡片 frontmatter 的 `paper_id` = **arXiv ID**（如 2608.25277）
    #   registry 的 `paper_id`        = **项目内 ID**（如 p2s-2026-0018）
    # 第一版修复只按 registry 的 `paper_id` 匹配，于是全部落空、静默退回旧行为。
    # 现按三条线索依次认领：registry.paper_id / identifiers.arxiv / outputs.skill_card 路径。
    m_pid = re.search(r"^paper_id:\s*(\S+)", text, re.M)
    fm_pid = m_pid.group(1).strip().strip("\"'") if m_pid else ""
    if fm_pid:
        try:
            reg = json.loads((VAULT / "07-资源库" / "papers_registry.json")
                             .read_text(encoding="utf-8"))
            card_rel = str(card.resolve().relative_to(REPO_ROOT))
            rec_hit = None
            for rec in reg.get("records", []):
                if rec.get("paper_id") == fm_pid:
                    rec_hit = rec; break
                if (rec.get("identifiers") or {}).get("arxiv") == fm_pid:
                    rec_hit = rec; break
            if rec_hit is None:       # 再退一步：用 outputs.skill_card 反认（应对未回填的模板占位）
                for rec in reg.get("records", []):
                    sc = (rec.get("outputs") or {}).get("skill_card") or ""
                    if sc and "<" not in sc and sc == card_rel:
                        rec_hit = rec; break
            if rec_hit is not None:
                p_out = (rec_hit.get("outputs") or {}).get("evidence")
                if p_out and "<" not in p_out:
                    ev_candidates.insert(0, REPO_ROOT / p_out)
                # 兜底：用 registry 的项目内 ID 扫 papers/*/<p2s-id>/evidence.md
                p2s = rec_hit.get("paper_id")
                if p2s:
                    ev_candidates.extend(
                        sorted((VAULT / "papers").glob(f"*/{p2s}/evidence.md")))
        except Exception:
            pass                      # registry 不可读不该让门禁崩，退到路径兜底
        ev_candidates.extend(sorted((VAULT / "papers").glob(f"*/{fm_pid}/evidence.md")))

    seen_ev: set[Path] = set()
    for cand in ev_candidates:
        cand = Path(cand)
        if cand in seen_ev or not cand.is_file():
            continue
        seen_ev.add(cand)
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


# 中文数字量级断言：`两三倍`、`十五个百分点`、`三成`、`翻倍`。
# ⚠️ 这是一个**已知无法自动核验**的形式：NUM_TOKEN_RE 只认阿拉伯数字，
# 因此「论文说 two to more than three times」写成「两三倍」后，
# G2 完全看不到 —— 实测真实案例：卡片断言「离散度是对齐标签的两三倍」，
# 对应原文 "they leave two to more than three times the aligned label's dispersion"，
# 断言本身正确，但绕过了全部数字检查。
# 跨语言数字匹配无法机械化（引文是英文而断言是中文），所以这里**只报黄灯**，
# 要求人工确认该断言有对应引文 —— 目的是让豁免**可见**，而不是假装已覆盖。
CN_NUMERAL_CLAIM_RE = re.compile(
    r"([一二三四五六七八九十百千万两半]+\s*(?:倍|成|个百分点|pp))"
    r"|(翻[一二三四五六七八九十两]?倍)"
)

# 领域目录标签：`15-营销投放分析`、`13-广告分析`、`06-增长模型`。
# 这是仓库自身的命名约定，裸数字 15 不是任何事实断言 ——
# 不加这条会把「组合 Skill-Marketing-Mix-Modeling.md（15-营销投放分析）」
# 报成「高价值断言 15 无出处」（实测假红灯 2 条）。
DOMAIN_LABEL_RE = re.compile(r"^-\s*[\u4e00-\u9fff]")


def is_noise(num: str, unit: str, prefix: str = "", suffix: str = "") -> bool:
    """判断数字是否为「结构性编号」而非事实断言。

    `prefix` / `suffix` 是数字左右两侧的若干字符，用于识别
    §4.2 / 图 3 / 式 (7) / 15-营销投放分析 这类指代与标签。
    """
    if prefix and STRUCTURAL_PREFIX_RE.search(prefix):
        return True
    if suffix and DOMAIN_LABEL_RE.search(suffix):
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


def strip_frontmatter(text: str) -> str:
    """去掉 YAML frontmatter 块，再做数字扫描。

    ⚠️ 这是一个实测出来的**假黄灯源**（2026-09-12，由 F4 子代理发现）：
    frontmatter 里的 `created: 2026-05-15` / `updated: 2026-09-12` 会被
    NUM_TOKEN_RE 拆出 `15` 与 `12`，然后判成「一般数字无出处」。
    实测影响 **65 张卡、126 条黄灯** —— 全是**元数据**，不是断言。

    frontmatter 是**文档元数据**（标题/模块/时间/溯源字段），按定义不承载
    「事实断言」；把它计入 G2 断言集，会让黄灯数随「卡片填了多少元数据」变化，
    而不是随「内容有多少无出处数字」变化 —— 那正是本门禁最该避免的事。
    `paper_id` / `venue` 等字段只用于**分类**（card_has_paper_source），
    不参与断言计数，所以剥掉不影响溯源判定。
    """
    m = FRONTMATTER_RE.match(text)
    return text[m.end():] if m else text


def strip_evidence(text: str) -> str:
    return EVIDENCE_LINE_RE.sub("\n", text)


# 尾部「参考论文 / References」区：那里的 arXiv ID 是**延伸阅读**，不是本卡的来源声明。
# 实测：01-因果推断/Skill-Intelligent-Attribution-Causal-Forest 的正文里
# 「参考论文」出现在第 13853 字符（卡片总长 14069），若不去掉这段，
# 一张纯经验卡会因为列了几篇参考论文而被迫按「有来源卡」审查 —— 判错了对象。
#
# ⚠️ `参考资料` 必须在内（2026-09-12 由子代理实测发现）：全库 43 张卡用 `参考论文`，
# 但有 3 张用 `参考资料`。实测 `04-供应链/Skill-Two-Echelon-Inventory-DRL.md` 的
# `## 参考资料` 是**混合表** —— 第 1 条正是本卡来源论文（且正文 ③ 段 docstring 里
# 写着「基于论文：Stranieri & Stella (2022) …」），另两条才是延伸阅读。
# 漏掉这个词会让这类卡被判「有来源」（正确），但**理由**是错的；
# 更糟的是若有人据此给它们加 `author-practice`，会真的触发 BASIS-CONTRADICTION。
_REF_SECTION_RE = re.compile(
    r"^#{1,4}\s*(参考论文|参考文献|参考资料|References|Bibliography|延伸阅读)\s*$",
    re.M | re.I,
)


def strip_reference_section(text: str) -> str:
    """截掉尾部参考区（保留其余正文）。"""
    m = _REF_SECTION_RE.search(text)
    return text[:m.start()] if m else text


def card_has_paper_source(text: str, fm: dict) -> bool:
    """这张卡**有没有**把某篇论文声明为**自己的来源**？

    判据是「卡里存在 arXiv ID / DOI / 论文标题字段」——注意：
    - frontmatter 的 `source: human+ai` 是**文档来源**（谁写的）不是论文来源，
      不在此列。实测 67 张卡有 `source:` 字段，极易被误当溯源字段。
    - 尾部「参考论文」区里的 arXiv ID 是**延伸阅读**，先剥掉再判。

    ⚠️ 本函数**只回答「有没有论文可指」**，不回答「指得对不对」：
    `paper_id: 9999.99999`（不存在的 ID）也会返回 True。
    那是**该修的缺陷**，故意留给 G2 的红灯去抓 —— 若在这里顺手放行，
    就等于给「编一个 arXiv ID 来换免检」开了后门。
    """
    if (fm.get("paper_id") or "").strip():
        return True
    for k in ("paper", "arxiv", "arxiv_id", "doi", "url"):
        v = (fm.get(k) or "").strip()
        if v and (ARXIV_ANY_RE.search(v) or DOI_ANY_RE.search(v)
                  or (k == "paper" and len(v) > 12)):
            return True
    # 正文头部：存量卡的来源声明多在正文（`**论文来源**: ... (arXiv:2408.05353)`）
    body = strip_reference_section(text)
    return bool(ARXIV_ANY_RE.search(body) or DOI_ANY_RE.search(body))


def gate_g2(card: Path, text: str) -> GateResult:
    rel = rel_to_repo(card)
    g = GateResult("G2", rel, True)
    # 先剥 frontmatter（元数据不是断言），再剥代码（代码里的数字不是事实断言），
    # 最后剥证据语法行（引文是证据不是断言）
    prose = strip_evidence(strip_code(strip_frontmatter(text)))
    fm = parse_frontmatter(text)

    # --- 证据基础分类（决定本卡是「该修」还是「修不了」）-------------------
    # `evidence_basis: author-practice` = 作者经验卡，**从设计上就没有论文来源**。
    # 对它判「数字无出处」是**判错了对象** —— 补救动作是给它加声明，
    # 不是给它找论文。所以这类卡走 UNVERIFIABLE，且不产生阻塞红灯。
    basis = (fm.get("evidence_basis") or fm.get("provenance") or "").strip().lower()
    declared_practice = basis in AUTHOR_PRACTICE_BASIS
    has_source = card_has_paper_source(text, fm)

    if declared_practice and not has_source:
        g.outcome = "UNVERIFIABLE"
    elif declared_practice and has_source:
        # 声明与实物矛盾：卡里明明有 arXiv ID 却自称「无论文来源」。
        # 这**不能**静默采信声明 —— 否则加一行 frontmatter 就能全库免检。
        g.add("RED", "G2-BASIS-CONTRADICTION",
              f"frontmatter 声明 `evidence_basis: {basis}`（无论文来源），"
              f"但卡片内存在 arXiv/DOI/论文标题 —— 声明与实物矛盾，按有来源卡审查")
        has_source = True

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
        # ⚠️⚠️ 只有 **VERBATIM** 能充当出处（2026-09-12 修，漏洞 #13）。
        #
        # 原写法把 `UNVERIFIABLE` 也算作出处，理由是「历史卡无全文存档」——
        # 那是个善意豁免，但它**没有闸门**：`UNVERIFIABLE` 的成因恰恰是
        # 「找不到这篇论文的全文」，于是任何人只要写一段引文、配一个
        # 指不到底本的 `paper_id`，就能让任意数字变成"有出处"。
        #
        # 更糟的是它与 QUOTE_RE 的跨行 bug 叠加后形成**自引文洗白**：
        # 卡片自己的声明文字被解析成"引文"，卡片自己的数字就成了"有出处"。
        # 由子代理在 `Skill-AB-Experimental-Design` 上实测发现：
        # 一条 2777 字符的"引文"里 90% 是卡片自身正文，
        # metrics 显示 `sourced=20/26, traceability_pct=76.9` ——
        # **20 个数字的"出处"就是卡片自己**，红灯 28→6 是假象。
        #
        # 修法：`UNVERIFIABLE` 不再计入出处，只出一条黄灯（下面已有）。
        # 这与「门禁的豁免条款必须比拦截条款测得更严」是同一原则。
        trusted_quotes = {
            q["quote"] for q in qrep.get("quotes", [])
            if q.get("verdict") == "VERBATIM"
        }

    ev_nums, ev_sources = collect_evidence(card, text, trusted_quotes=trusted_quotes)

    claimed: list[tuple[str, str]] = []
    for m in NUM_TOKEN_RE.finditer(prose):
        num, unit = m.group(1), (m.group(2) or "")
        prefix = prose[max(0, m.start() - 12):m.start()]
        suffix = prose[m.end():m.end() + 12]
        if is_noise(num, unit, prefix, suffix):
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
        suffix = prose[m.end():m.end() + 12]
        if is_noise(num, unit, prefix, suffix):
            continue
        ctx = prose[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
        if not is_metric(ctx, num, unit) and num not in ev_nums:
            yellow.append((f"{num}{unit}", ctx.strip()))

    # --- G2c 出处「实质性」检查（高精度，专抓数字凑巧命中）-------------------------
    #
    # ⚠️ 这是 G2 最后一个已知漏洞：`ev_nums` 是**全局、不区分出处**的数字集合，
    # 因此不校验「某个断言由某条引文支撑」，只校验「这个数字在全库某处出现过」。
    # 实测反例：卡片断言「离散度是对齐标签的 2–3 倍」，而唯一含 `3` 的引文是
    # "the controlled model-skill comparison (Table 3) is single-organization" ——
    # 数字 3 来自**表号**，与该断言毫无关系，但 G2 判 GREEN。
    #
    # 修法（刻意做成高精度而非高召回）：只检查「该数字在引文里的**全部**出现位置
    # 是否都属于结构性语境」（表号/图号/公式号/章节号/样本量 10K）。
    # 若全是结构性的 → 该数字没有任何实质性出处 → 黄灯要求人工确认。
    # 不做语义相关性判断，因为引文多为英文而断言多为中文，跨语言词面重合度
    # 天然接近 0，强行相关会制造大量假阳性，而假阳性会让门禁失去威信。
    all_quote_texts = [q.get("quote", "") for q in qrep.get("quotes", [])
                       if q.get("verdict") in ("VERBATIM", "UNVERIFIABLE")]
    structural_only: list[tuple[str, str]] = []
    for token, ctx in claimed:
        bare = re.match(r"[\d.]+", token).group(0)
        occ_before: list[str] = []
        for qt in all_quote_texts:
            # 边界必须是 (?<![\d.]) 与 (?![\d.])：只看一侧会把小数 `3.4`
            # 里的 `3` 当成独立出现，于是「3」总能找到一堆假实质出处，
            # 实测导致本检查完全失效（合成用例该报却没报）。
            for mm in re.finditer(rf"(?<![\d.]){re.escape(bare)}(?![\d.])", qt):
                occ_before.append(qt[max(0, mm.start() - 14):mm.start()])
        if not occ_before:
            continue                      # 无出处的情况已由 G2-UNSOURCED-METRIC 覆盖
        substantive = [
            b for b in occ_before
            if not STRUCTURAL_PREFIX_RE.search(b)
            and not re.search(r"\d+\s*[KkMm]$", b)      # 10K / 20K 样本量
        ]
        if not substantive:
            structural_only.append((token, ctx))
    structural_only = list(dict.fromkeys(structural_only))

    # --- 中文数字量级断言：无法自动核验，报黄灯要求人工确认（见常量处说明）---
    cn_claims = list(dict.fromkeys(
        m.group(0) for m in CN_NUMERAL_CLAIM_RE.finditer(prose)))

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
        "sourced_structural_only": len(structural_only),
        "cn_numeral_claims": len(cn_claims),
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
        if g.outcome == "UNVERIFIABLE":
            # ⚠️ 这里不是「放行」，而是**换了问题**：本卡没有任何论文来源，
            # 「数字有没有出处」这一问题对它无定义。它的正确审查项是
            # 「有没有声明自己是经验卡」+「有没有伪装成论文结论」。
            g.add("INFO", "G2-UNVERIFIABLE-NO-SOURCE",
                  f"卡片声明 `evidence_basis: {basis}` 且无 arXiv/DOI —— "
                  f"按**作者经验卡**处理，{len(red) + len(yellow)} 处数字**无法核验**"
                  f"（非缺陷，但也不构成证据）",
                  evidence="若这些数字实际来自某篇论文，请补 paper_id 与 ⑥ 段引文，"
                           "本卡会自动转为有来源卡审查")
        else:
            g.add("RED", "G2-NO-EVIDENCE-CHAIN",
                  "卡片内既无 `> 原文：\"...\"` 引用块，也无 evidence.md —— 无任何可追溯的出处")
    # 无法核验的卡不再逐条报「数字无出处」：那是同一件事的 N 次重复，
    # 会把 56 张卡的红灯数（916 条）淹没真正的缺陷信号。
    report_unsourced = g.outcome != "UNVERIFIABLE"
    if report_unsourced:
        for token, ctx in red[:40]:
            g.add("RED", "G2-UNSOURCED-METRIC",
                  f"高价值断言 `{token}` 无出处", evidence=f"…{ctx}…")
        if len(red) > 40:
            g.add("RED", "G2-TRUNCATED", f"另有 {len(red) - 40} 条高价值断言无出处（已截断展示）")
    for token, ctx in yellow[:10]:
        g.add("YELLOW", "G2-UNSOURCED-GENERAL", f"一般数字 `{token}` 无出处",
              evidence=f"…{ctx}…")
    for tok in cn_claims[:15]:
        g.add("YELLOW", "G2-CN-NUMERAL-CLAIM",
              f"断言 `{tok}` 用中文数字表达量级，**无法自动核验**（引文通常为英文，"
              f"跨语言数字匹配不可机械化）—— 请人工确认它确实有对应引文",
              evidence="若该量级来自论文，建议改写成阿拉伯数字并在 ⑥ 段补引文；"
                       "若为定性描述，请确认措辞未夸大论文结论")
    for token, ctx in structural_only[:15]:
        g.add("YELLOW", "G2-SOURCED-STRUCTURAL-ONLY",
              f"断言 `{token}` 的数字只在引文的**结构性语境**（表号/图号/章节号/样本量）中出现，"
              f"未见实质性出处 —— 请人工确认该断言是否真的有引文支撑",
              evidence=f"…{ctx}…")
    if green:
        g.add("GREEN", "G2-SOURCED", f"{len(green)} 个高价值断言有出处")

    # 只有「高价值断言全部有出处」**且**「无伪造引文」才算 G2 绿。
    # 必须按 findings 统一判定：早期版本写死 `len(red) == 0`，
    # 会在 G2b 追加红灯后仍报绿（门禁 bug，已修）。
    has_red = any(f.level == "RED" for f in g.findings)
    g.passed = not has_red
    # --- 三态收敛：UNVERIFIABLE 只有当它**确实没有红灯**时才成立 -------------
    # ⚠️ 顺序不能反：先算 passed，再决定 outcome。
    # 若先无条件把 outcome 设成 UNVERIFIABLE，一张「声明无论文来源但引文伪造」
    # 的卡会被记成「无法核验」而不是「引文伪造」—— 那正是本门禁要防的事。
    if g.outcome == "UNVERIFIABLE" and has_red:
        g.outcome = "FAIL"
    elif g.outcome != "UNVERIFIABLE":
        g.outcome = "PASS" if g.passed else "FAIL"
    g.metrics["evidence_basis"] = basis or "(未声明)"
    g.metrics["has_paper_source"] = has_source
    g.metrics["outcome"] = g.outcome
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
    g.outcome = "PASS" if g.passed else "FAIL"
    return g


def selftest() -> int:
    """自证 G2 三态判定可信 —— 门禁必须先自证，再谈被门禁对象。

    五个用例，其中**用例 3 与 5 是防「新口径变成新后门」**：
    新增一个「不阻塞」的结局，本身就是一次放水风险。必须同时锁定
    「什么情况下不许给 UNVERIFIABLE」，否则加一行 frontmatter 就能全库免检。
    """
    import tempfile

    cases = [
        # (名称, frontmatter 追加行, 正文, 期望 outcome, 期望有红灯)
        ("1 有来源卡未声明", "paper_id: 2606.26690\n",
         "ROI 提升 **42.7%**，覆盖 2,300 万用户。\n> Skill-A\n> Skill-B",
         "FAIL", True),
        ("2 经验卡已声明", "evidence_basis: author-practice\n",
         "经验做法：先按品类分层再算 ROI，通常能省 **30%** 人力。\n> Skill-A\n> Skill-B",
         "UNVERIFIABLE", False),
        ("3 声明与实物矛盾", "evidence_basis: author-practice\npaper_id: 2606.26690\n",
         "ROI 提升 **42.7%**。\n> Skill-A\n> Skill-B",
         "FAIL", True),
        ("4 伪造引文不得因声明而免检",
         "evidence_basis: author-practice\n",
         'ROI 提升 **42.7%**。\n\n> 原文:"our framework improves incremental ROAS by 42.7% across all markets"\n> 出处：2606.26690 §1\n',
         "FAIL", True),
        ("5 有来源卡给足证据", "paper_id: 2606.26690\n",
         "定性结论，不带数字。\n\n"
         '> 原文:"Production attribution is timely, granular, and continuously '
         'available, but observational by construction."\n> 出处：2606.26690 §1\n',
         "PASS", False),
        # --- 用例 6/7 专防漏洞 #13（自引文洗白）----------------------------
        # 子代理实测：把声明写成以 `「` 开头的 `>` 行时，旧 QUOTE_RE 会跨行吞掉
        # 卡片正文，把卡片自己的数字变成"有出处"。用例 6 锁定「指不到底本的
        # 引文不算出处」；用例 7 锁定「中文闭引号不再跨行吞正文」。
        ("6 无全文的引文不得充当出处",
         "paper_id: 9999.99999\n",
         'ROI 提升 **42.7%**，覆盖 **2300万** 用户。\n\n'
         '> 原文:"our framework improves incremental ROAS by 42.7% And reaches 2300 万 users"\n'
         '> 出处：9999.99999 §1\n',
         "FAIL", True),
        ("7 中文闭引号不得跨行吞正文",
         "evidence_basis: author-practice\n",
         '# 卡\n\n'
         '> 「本行以中文开引号，闭引号后面还有字」。\n'
         '> 这一行是卡片自己的正文，**42.7%** 与 **2300万** 都是本卡的数字。\n',
         "UNVERIFIABLE", False),
    ]

    ok = True
    with tempfile.TemporaryDirectory() as td:
        for name, fmx, body, want_outcome, want_red in cases:
            p = Path(td) / "Skill-Selftest.md"
            p.write_text(f"---\ntitle: selftest\n{fmx}---\n\n{body}\n", encoding="utf-8")
            r = gate_g2(p, p.read_text(encoding="utf-8"))
            has_red = any(f.level == "RED" for f in r.findings)
            good = (r.outcome == want_outcome) and (has_red == want_red)
            ok = ok and good
            print(f"{'✅' if good else '❌'} {name}: outcome={r.outcome}"
                  f"（期望 {want_outcome}）红灯={has_red}（期望 {want_red}）")

    # 用例 2 的正确性还依赖一个前提：经验卡确实被判成「无论文来源」。
    # 若 card_has_paper_source 有缺陷（例如把 `source: human+ai` 当论文来源），
    # 用例 2 会退化成 FAIL 而被上面的断言抓到 —— 但把前提单独打印出来，
    # 能让人一眼看出失败是「判定错」还是「用例本身构造错了」。
    probe = "本卡为人写的经验总结。\n> Skill-A\n> Skill-B"
    print(f"   前提核对：无 arXiv/DOI 的正文判为无论文来源 = "
          f"{not card_has_paper_source(probe, {'source': 'human+ai'})}")

    print("✅ 自检通过：G2 三态互斥，且『无论文来源』不能靠声明洗白伪造引文" if ok
          else "❌ 自检失败：G2 三态判定不可信")
    return 0 if ok else 1


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
    """按 outcome 三态汇总。

    ⚠️ **绝不把 UNVERIFIABLE 并进分母算通过率**（2026-09-12）。
    与「Cited but Not Verified」的教训同源：合成一个数字会把最弱的那维平均掉。
    这里更糟 —— 它会把「无论文来源」平均成「通过」，于是全库通过率
    会因为**多了 56 张没法验的卡**而看起来变好。
    """
    counts = {"passed": 0, "failed": 0, "unverifiable": 0}
    for g in gates:
        if g.outcome == "UNVERIFIABLE":
            counts["unverifiable"] += 1
        elif g.passed:
            counts["passed"] += 1
        else:
            counts["failed"] += 1

    reds = sum(1 for g in gates for f in g.findings if f.level == "RED")
    yellows = sum(1 for g in gates for f in g.findings if f.level == "YELLOW")
    # 「可核验分母」= 排除无法核验的卡；通过率只在它之上计算
    verifiable = counts["passed"] + counts["failed"]
    out = {
        "gate": name,
        "cards_checked": len(gates),
        "cards_passed": counts["passed"],
        "cards_failed": counts["failed"],
        "cards_unverifiable": counts["unverifiable"],
        "verifiable_denominator": verifiable,
        "pass_rate_pct": round(counts["passed"] / verifiable * 100, 1) if verifiable else 0.0,
        "red_findings": reds,
        "yellow_findings": yellows,
    }
    # 兼容旧消费者（sync.py / 报告脚本）读的键名
    out["pass_rate_over_all_cards_pct"] = (
        round(counts["passed"] / len(gates) * 100, 1) if gates else 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="paper2skills G1/G2/G3 门禁")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--card", type=Path)
    g.add_argument("--all", action="store_true")
    g.add_argument("--selftest", action="store_true",
                   help="自证 G2 三态判定（含『声明不得洗白伪造引文』反例）")
    ap.add_argument("--k1", type=Path, default=None, help="K1 报告 JSON（verify_skill_code.py 产出）")
    ap.add_argument("--outdir", type=Path, default=None, help="三份 gate_*.json 的输出目录")
    ap.add_argument("--only", choices=["G1", "G2", "G3"], default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

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
            print(f"\n{'='*70}\n{name} 门禁：{s['cards_passed']}/{s['verifiable_denominator']} 通过 "
                  f"({s['pass_rate_pct']}%，可核验分母)"
                  f"　红灯 {s['red_findings']}　黄灯 {s['yellow_findings']}")
            if s["cards_unverifiable"]:
                print(f"         另有 {s['cards_unverifiable']} 张**无法核验**"
                      f"（无论文来源，不计入上述分母 —— 既不通过也不失败）")
            print("=" * 70)
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
