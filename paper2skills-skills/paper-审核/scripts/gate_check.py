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
    r"^v?\d+\.\d+(\.\d+)?$",          # 版本号 1.2.3
    r"^(19|20)\d{2}$",                # 年份
    r"^\d{1,2}$",                     # 个位数（章节号、序号）
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
QUOTE_RE = re.compile(r"^\s*>\s*[「『\"“](.+?)[」』\"”]\s*$", re.M)

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


def collect_evidence(card: Path, text: str) -> tuple[set[str], list[str]]:
    """收集证据链：卡片内引用块 + 同目录 evidence.md 的数字与出处。

    返回 (证据中出现的数字字符串集合, 出处说明列表)
    """
    nums: set[str] = set()
    sources: list[str] = []

    for m in QUOTE_RE.finditer(text):
        src = m.group(1)
        sources.append(f"卡片引用块: {src[:80]}")
        for n in re.findall(r"\d+(?:\.\d+)?", src):
            nums.add(n)

    # 同目录 / 同名 evidence.md
    for cand in (
        card.with_suffix(".evidence.md"),
        card.parent / "evidence.md",
        card.parent / f"{card.stem}.evidence.md",
    ):
        if cand.is_file():
            ev = cand.read_text(encoding="utf-8", errors="replace")
            sources.append(f"evidence.md: {cand.relative_to(REPO_ROOT)}")
            for n in re.findall(r"\d+(?:\.\d+)?", ev):
                nums.add(n)
    return nums, sources


def is_noise(num: str, unit: str) -> bool:
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
def gate_g2(card: Path, text: str) -> GateResult:
    rel = rel_to_repo(card)
    g = GateResult("G2", rel, True)
    prose = strip_code(text)
    ev_nums, ev_sources = collect_evidence(card, text)
    fm = parse_frontmatter(text)

    claimed: list[tuple[str, str]] = []
    for m in NUM_TOKEN_RE.finditer(prose):
        num, unit = m.group(1), (m.group(2) or "")
        if is_noise(num, unit):
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
        if is_noise(num, unit):
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
    }

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

    # 只有「高价值断言全部有出处」才算 G2 绿
    g.passed = len(red) == 0
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
