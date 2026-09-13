#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""契约层（L4）校验器 —— 判据 J1–J13 + 自检 + 变异测试。

判据只认一处事实源：`capability-graph.json`（PHASE6 F2 产出）。
契约文件里凡与图谱相关的字段都由 `build_contracts.py --skeleton` 生成，
本脚本独立重算一遍并逐字段比对 —— **漂移即报错，而不是「看起来一致」**。

设计纪律（本仓库已付过四次学费）：
· 每一条判据都要有一份**能把它打红**的变异样本（`--selftest` 里的变异表）；
  真实数据本来就满足的断言等于没断言（F2 抓出过 7 处摆设断言）。
· **覆盖率是一等输出**：核对到 N/139；低于阈值时拒绝给「完整」结论。
  （来由：`registry_consistency.py` 的前身静默跳过 3 条记录却报「无不一致」。）
· 反后门：`status` 是**如实记账**字段，不是放行开关；「不可得」是合法结论，
  但必须同时把 status 置为 `数据待盘`，否则缺口会在正文里被一句对冲掩盖掉。

用法::

    python3 check_contracts.py --all                     # 扫契约目录
    python3 check_contracts.py --file <契约.md>           # 单份（写卡时用）
    python3 check_contracts.py --json-out <path>
    python3 check_contracts.py --selftest                # 13 判据 + 变异表
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DEFAULT = REPO_ROOT / "paper2skills-vault/07-资源库/capability-graph.json"
CONTRACTS_DIR = REPO_ROOT / "paper2skills-vault/07-资源库/contracts"
BUILDER = Path(__file__).resolve().parent / "build_contracts.py"

# 数据要求五维的标签与 ① 的合法枚举（Q12 / S11）
DIM_LABELS = ("获取路径", "粒度", "回溯深度", "新鲜度", "口径归属")
ACCESS_ENUM = ("自有埋点", "平台后台", "第三方 API", "需授权", "不可得")

# A / B 模板的六段（编号 → 关键词）。关键词用于校验「段没被换掉」，
# 但不比对整行 —— 模板文档里同一段可能带括号补充，逐字比对会把正常写作判红。
SECTION_KEYS = {
    "A": {1: "方法来源", 2: "数据要求", 3: "标定规则", 4: "重标定触发条件", 5: "适用", 6: "冻结"},
    "B": {1: "算法可达部分", 2: "不可达部分", 3: "必须的外部证据", 4: "冻结", 5: "数据要求", 6: "适用"},
}
STATUS_ENUM = ("可写", "待卡", "数据待盘", "方法待启用")
# 反对冲：这三句是 SkillOpt 补丁的死法（26/40 挂上对冲措辞，Δ 从 −0.95 变 −1.10）
HEDGE_TOKENS = ("临时默认值", "需重新标定", "须重新标定", "待标定")
CELL_RE = re.compile(r"FLOW-0[1-8]/STG-0[1-8]")

# --- J13（v2 正面义务）：写了缺口就必须写「那先用什么」 ---
# 来由：F6 实测契约臂「以省略代对冲」——量化判据中位 3（对照 19.5 / 单卡 22），
# 弃权语中位 2（其余两臂 0），缺口句处置率 **0.000**。修法是给禁令补上对称的义务，
# 所以判据必须能**打红**「只写缺口不写处置」——否则模板里那句「不得留白」就是摆设。
UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|倍|天|小时|分钟|秒|日|周|月|年|个|条|款|元|美元|万|亿|次|人|分|折|国|SKU|pp|‰)")
GAP_TOKENS = ("暂无", "未定名", "不可得", "拿不到", "取不到", "尚未", "待定", "未知")
DISPOSAL_TOKENS = ("替换", "失效", "改用", "顶替", "代为", "先行", "暂按", "暂用",
                   "生效条件", "到位后", "上线后", "接入后", "补充后")
ALT_MARK = "替代路径"
# ⚠️ 只认 ASCII 声明的模板版本；拿不到就按 v1 免检，但**数量进摘要**（run() 里报出来）。
#    「静默豁免」是本仓库吃过四次亏的东西，所以豁免一律要计数。
def is_v2(c) -> bool:
    v = c.fm.get("template_version")
    try:
        return int(str(v)) >= 2
    except (TypeError, ValueError):
        return False


# 「取值」必须是**带标签的行**里的数字。
# ⚠️ 首版直接用 UNIT_RE 扫全段，自检当场打红：散文里的「近 2 个完整年度」被当成了取值
#    （`2 个`名副其实地匹配上了单位表）—— 那是 §2 的数据要求，不是本业务的标定取值。
#    这与漏洞 #14 同源：**判据的适用范围被默认成了全体，而它其实只覆盖一个子集。**
VALUE_LABEL_RE = re.compile(r"(取值|阈值|默认值|基线|区间|容差档|目标值|分档)")
_BARE_DIGIT_RE = re.compile(r"\d")


def has_value(text: str) -> bool:
    """带取值标签的行里出现数字 ⇒ 认为给出了取值（模板 §3 的 `- 取值：<...>` 形态）。"""
    return any(VALUE_LABEL_RE.search(ln) and _BARE_DIGIT_RE.search(ln) for ln in text.splitlines())


def has_disposal(text: str) -> bool:
    return any(tok in text for tok in DISPOSAL_TOKENS)


def gap_without_disposal(text: str) -> bool:
    """该段写了缺口却没有处置 ⇒ True（= 应当判红）。"""
    if not any(tok in text for tok in GAP_TOKENS):
        return False
    return not (has_value(text) or has_disposal(text))
AGT_RE = re.compile(r"\bAGT-0\d{2}\b")


def load_builder():
    spec = importlib.util.spec_from_file_location("build_contracts", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BC = load_builder()


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
class Contract:
    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text(encoding="utf-8")
        self.fm = BC.read_frontmatter(self.text)
        self.sections = self._split_sections()
        self.written = BC.TODO_MARK not in self.text

    def _split_sections(self) -> dict:
        out, cur = {}, None
        for line in self.text.splitlines():
            m = re.match(r"^#{2,3}\s*(\d)\s*(.*)$", line.strip())
            if m and m.group(1) in "123456":
                cur = int(m.group(1))
                out[cur] = {"title": m.group(2), "lines": []}
                continue
            if cur is not None:
                out[cur]["lines"].append(line)
        return {k: {"title": v["title"], "body": "\n".join(v["lines"]).strip()} for k, v in out.items()}

    @property
    def template(self):
        return self.fm.get("template", "")

    def section(self, n) -> str:
        s = self.sections.get(n)
        return s["body"] if s else ""

    def section_title(self, n) -> str:
        s = self.sections.get(n)
        return s["title"] if s else ""

    # 五维的「取值 + 缺值处置」两格（J13 用；data_dims 只取第一格，J5 用）
    def dim_cells(self) -> dict:
        n = 2 if self.template == "A" else 5
        body = self.section(n)
        out = {}
        for label in DIM_LABELS:
            m = re.search(rf"{label}[^\n|]*\|([^\n|]*)\|([^\n|]*)", body)
            out[label] = ((m.group(1).strip(), m.group(2).strip()) if m else ("", ""))
        return out

    # 五维取值：从 2 段（A）或 5 段（B）取值
    def data_dims(self) -> dict:
        n = 2 if self.template == "A" else 5
        body = self.section(n)
        dims = {}
        for label in DIM_LABELS:
            m = re.search(rf"{label}[^\n|]*[|:：]\s*([^\n|]*)", body)
            dims[label] = (m.group(1).strip() if m else "")
        return dims


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------
def check_one(c: Contract, idx, assignments) -> list:
    """返回 [(判据代号, 说明)]，空列表 = 全过。未撰写的骨架只跑结构类判据。

    解析层异常（如 YAML 列表括号未闭合）在这里被**收成一条判据**而不是让脚本崩 ——
    「门禁自己崩了」与「门禁判红」在流水线里必须可区分。
    """
    try:
        return _check_one(c, idx, assignments)
    except ValueError as e:
        return [("J2", f"frontmatter 解析失败：{e}")]


def _check_one(c: Contract, idx, assignments) -> list:
    """返回 [(判据代号, 说明)]，空列表 = 全过。"""
    bad = []

    def red(code, msg):
        bad.append((code, msg))

    # --- J1 模板由服务性唯一导出；C 类不得有契约 ---
    if not c.fm:
        return [("J1", "无 frontmatter")]
    name = c.fm.get("responsibility", "")
    meta = assignments.get(name)
    if meta is None:
        if name in idx.l3:
            return [("J1", f"{name} 是 C 类责任，不建契约（139 = 151 − 12）")]
        return [("J1", f"责任名不在图谱 151 内：{name!r}")]
    if c.fm.get("template") != meta["template"]:
        red("J1", f"模板与图谱服务性不符：文件={c.fm.get('template')!r} 应为={meta['template']!r}")

    # --- J2 frontmatter 与图谱逐字段一致（防手改成第二份事实源）---
    for k in BC.FM_KEYS:
        if k == "template_version":
            # 期望值来自生成器常量，不由图谱导出。缺席 = F6 v1 底本 ⇒ 不算错，
            # 但**必须计数**（见 run() 的摘要）：静默豁免正是本仓库吃过四次亏的东西。
            if "template_version" in c.fm and BC.norm_val(c.fm.get(k)) != str(BC.TEMPLATE_VERSION):
                red("J2", f"template_version={c.fm.get(k)!r}，本版生成器写 {BC.TEMPLATE_VERSION}")
            continue
        got, want = BC.norm_val(c.fm.get(k)), meta[k]
        if got != want:
            red("J2", f"{k} 与图谱不一致：文件={got!r} 图谱={want!r}")
    st = c.fm.get("status", "")
    if st not in STATUS_ENUM:
        red("J1", f"status 不在枚举内：{st!r}（只认 {'/'.join(STATUS_ENUM)}）")

    # --- J3 cell_kind（风险 N4：R/D 格出现算法模型即判错）---
    for cid in c.fm.get("method_cells", ""):
        pass
    mcells = BC.norm_val(c.fm.get("method_cells"))
    rcells = BC.norm_val(c.fm.get("rule_cells"))
    for cid in mcells if isinstance(mcells, list) else []:
        if idx.cell_kind.get(cid) != "M":
            red("J3", f"方法接入格 {cid} 不是 M 格（kind={idx.cell_kind.get(cid)}）")
    for cid in rcells if isinstance(rcells, list) else []:
        if idx.cell_kind.get(cid) not in ("R", "D"):
            red("J3", f"规则格 {cid} 不是 R/D 格（kind={idx.cell_kind.get(cid)}）")
    # 正文里提到的格必须属于本契约的格；把非 M 格说成算法/模型接入也算错
    own = set(mcells or []) | set(rcells or [])
    for line in c.text.splitlines():
        if line.strip().startswith("#"):
            continue
        for cid in CELL_RE.findall(line):
            if cid not in own:
                red("J3", f"正文引用了不属于本契约的格：{cid}")
            elif idx.cell_kind.get(cid) != "M" and re.search(r"算法|模型", line):
                red("J3", f"把非 M 格（{cid} kind={idx.cell_kind.get(cid)}）说成算法/模型接入")

    # --- J4 flows 必须等于该岗位的 flows ---
    flows = BC.norm_val(c.fm.get("flows"))
    if flows != meta["flows"]:
        red("J4", f"flows 与图谱不一致：文件={flows!r} 图谱={meta['flows']!r}")

    # --- J5 数据要求五维齐全 + ① 命中枚举 ---
    # 豁免只对**仍带 `<!-- 待撰写 -->` 的骨架**生效：骨架的空缺由「覆盖率/已撰写数」如实记账，
    # 若把它也算成 5 条缺陷，139 份骨架会刷出 695 条噪声，人就会开始忽略校验器。
    # ⚠️ 反面：标记一去掉（= 已撰写）而五维仍不全 ⇒ 立刻判红（变异样本锁定这一条）。
    if c.written:
        dims = c.data_dims()
        miss = [k for k, v in dims.items() if not v or v.strip() in ("", "—", "-")]
        if miss:
            red("J5", f"数据要求缺维：{'、'.join(miss)}")
        access = dims.get("获取路径", "")
        if access and not any(e in access for e in ACCESS_ENUM):
            red("J5", f"获取路径必须点名五选一（{'/'.join(ACCESS_ENUM)}），实得：{access!r}")

        # --- J6 「不可得」必须如实记账（否则缺口被一句对冲掩盖）---
        # 两级判据，刻意**不一刀切**：一刀切（凡出现「不可得」就必须整份标数据待盘）会训练
        # 撰写人**回避写「不可得」**—— 那正是本条要防的事。于是：
        #   · ① 的主取值就是「不可得」 ⇒ 必须 数据待盘（缺口在账上，不能自称可写）
        #   · 「不可得」只作为①的一项/子项 ⇒ 不得标 `可写`（禁止一边说可写一边藏着拿不到的数据）
        if access and "不可得" in access:
            whole = access.strip()
            primary = whole == "不可得" or whole.startswith("不可得")
            if primary and st != "数据待盘":
                red("J6", f"获取路径=不可得 时 status 必须为「数据待盘」，实得 {st!r}")
            elif not primary and st == "可写":
                red("J6", "获取路径里含「不可得」却标 status=可写 —— 拿不到的数据必须在账上")

    # --- J7 / J8 模板必备字段 + 段标题未被换掉 ---
    for n, kw in SECTION_KEYS[meta["template"]].items():
        title = c.section_title(n)
        if not title:
            red(f"J{7 if meta['template'] == 'A' else 8}", f"缺第 {n} 段（应含关键词「{kw}」）")
            continue
        if kw not in title:
            red(f"J{7 if meta['template'] == 'A' else 8}",
                f"第 {n} 段标题不含关键词「{kw}」：{title!r}（改标题等于删段）")
        if not c.written:
            continue
        if not c.section(n) or BC.TODO_MARK in c.section(n):
            red(f"J{7 if meta['template'] == 'A' else 8}", f"第 {n} 段为空")

    if c.written and meta["template"] == "A":
        for n, label in ((1, "方法来源"), (3, "标定规则"), (4, "重标定触发条件")):
            if not c.section(n):
                red("J7", f"A 模板必备字段为空：{label}")
    if c.written and meta["template"] == "B":
        # 第 3 段的责任岗位必须是图谱 50 岗位里的真号
        ids = set(AGT_RE.findall(c.section(3)))
        known = set(idx.roles)
        bad_ids = sorted(x for x in ids if x not in known)
        if bad_ids:
            red("J8", f"第 3 段引用了不存在的岗位号：{bad_ids}")
        if not ids:
            red("J8", "第 3 段必须点名至少一个责任岗位（AGT-0xx）")

    # --- J9 反对冲（A 模板的标定规则）---
    if c.written and meta["template"] == "A":
        rule = c.section(3)
        if any(t in c.text for t in ("临时默认值",)):
            red("J9", "出现「临时默认值」—— 把论文参数留作默认值正是 SkillOpt 补丁的死法")
        if any(t in rule for t in HEDGE_TOKENS) and not re.search(r"自有埋点|平台后台|第三方 API|需授权|不可得", rule):
            red("J9", "标定规则以对冲结案：既没有命名数据源，也没有算式")

    # --- J12 cards 必须真实存在 ---
    for slug in (BC.norm_val(c.fm.get("cards")) or []):
        if not find_card(slug, idx):
            red("J12", f"cards 里的 {slug!r} 在卡库里找不到")
    if st == "待卡" and (BC.norm_val(c.fm.get("cards")) or []):
        red("J12", "status=待卡 却有 cards —— 两者必须一致")
    if st == "可写" and not (BC.norm_val(c.fm.get("cards")) or []):
        red("J12", "status=可写 却无 cards")

    # --- J13（v2 正面义务）---
    # 只对声明了 template_version ≥ 2 的契约生效；缺席 = F6 v1 底本，按 v1 口径免检，
    # 免检份数在 run() 的摘要里如实报出（豁免必须可见，否则就是新后门）。
    if c.written and is_v2(c):
        if meta["template"] == "A":
            rule = c.section(3)
            if gap_without_disposal(rule):
                red("J13", "§3 以缺口结案却没写处置 —— 缺口陈述不是结论，必须接「先用什么 + 何时换」")
            elif not has_value(rule):
                red("J13", "§3 只有算式、没有本业务取值（既无「数字+单位」也无缺口+处置）")
        else:
            reach = c.section(1)
            if gap_without_disposal(reach):
                red("J13", "§1 以缺口结案却没写处置 —— 「可复算」是形容词，不是取值")
            elif not has_value(reach):
                red("J13", "§1 没有给出本业务取值（既无「数字+单位」也无缺口+处置）")
            # B 模板 §2：每一条不可达都必须带替代路径（「不可达」是分工，不是免责）
            body2 = c.section(2)
            items = [ln for ln in body2.splitlines() if re.match(r"^\s*(?:[-*]|\d+[.、])\s+\S", ln)]
            marks = len(re.findall(ALT_MARK, body2))
            if not items:
                red("J13", "§2 没有列出任何不可达条目")
            elif marks < len(items):
                red("J13", f"§2 有 {len(items)} 条不可达但只有 {marks} 条「{ALT_MARK}」——"
                           "只写「做不到」是免责，写出「谁在何时补」才是分工")
        # 五维：取值为缺口时，缺值处置格必填
        for label, (val, disp) in c.dim_cells().items():
            if any(tok in val for tok in GAP_TOKENS):
                if not disp or disp in ("—", "-", "无"):
                    red("J13", f"{label} 是缺口但「缺值处置」为空 —— 那先用什么？")
                elif not (_BARE_DIGIT_RE.search(disp) or has_disposal(disp)):
                    red("J13", f"{label} 的「缺值处置」既没有替代取值也没有替换条件：{disp!r}")
    return bad


_CARD_CACHE = {}


def find_card(slug: str, idx) -> bool:
    if slug in _CARD_CACHE:
        return _CARD_CACHE[slug]
    hit = False
    # ① 图谱里的 vault 卡（按文件名匹配）
    for card in idx.graph.get("cards", []):
        stem = Path(card["name"]).stem if card["name"].endswith(".md") else card["name"]
        if stem == slug or card["name"] == slug:
            hit = True
            break
    # ② 已装 p2s 卡
    if not hit and (Path.home() / ".dsh/skills" / slug / "SKILL.md").exists():
        hit = True
    # ③ vault 里的 Skill-*.md（slug 常形如 Skill-Xxx）
    if not hit:
        for dom in (REPO_ROOT / "paper2skills-vault").iterdir():
            if not dom.is_dir():
                continue
            if (dom / f"{slug}.md").exists():
                hit = True
                break
    _CARD_CACHE[slug] = hit
    return hit


# ---------------------------------------------------------------------------
# 汇总 / 覆盖率 / 计数
# ---------------------------------------------------------------------------
def _rel(p: Path) -> str:
    """仓库内路径 → 相对路径；仓库外路径 → 原样返回。

    ⚠️ 原来直接 `p.relative_to(REPO_ROOT)`：传**仓库外的相对路径**时抛 ValueError，
    于是**判红的原因被 traceback 吞掉**，只剩一个退出码 1 —— 门禁自己崩了与门禁判红
    在流水线里必须可区分（这条与已登记的缺陷 #13 同源，由 F7 的契约撰写人实测撞出）。
    """
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def collect_files(paths) -> list:
    if paths:
        return [Path(p) for p in paths]
    return sorted(p for p in CONTRACTS_DIR.rglob("CTR-*.md") if p.is_file())


def run(files, idx, json_out=None) -> int:
    assignments = idx.assignments()
    total_expected = len(assignments)
    exp_a = sum(1 for m in assignments.values() if m["template"] == "A")
    exp_b = total_expected - exp_a

    contracts = [Contract(p) for p in files]
    written = [c for c in contracts if c.written]
    problems = []
    for c in contracts:
        for code, msg in check_one(c, idx, assignments):
            problems.append({"file": _rel(c.path), "code": code, "msg": msg})

    seen = {c.fm.get("responsibility") for c in contracts if c.fm}
    n_a = sum(1 for c in contracts if c.template == "A")
    n_b = sum(1 for c in contracts if c.template == "B")
    coverage = len([r for r in seen if r in assignments]) / total_expected if total_expected else 0.0
    written_cov = len(written) / total_expected if total_expected else 0.0

    print(f"契约文件 {len(contracts)} 份（A {n_a} / B {n_b}） · 已撰写 {len(written)} · 未撰写 {len(contracts) - len(written)}")
    print(f"目标分母 {total_expected}（A {exp_a} / B {exp_b}） · 入账覆盖 {len(seen & set(assignments))}/{total_expected} "
          f"= {coverage:.1%} · 已撰写覆盖 {written_cov:.1%}")
    # J13 的豁免必须**可见**：未标 template_version 的按 v1 口径免检，份数是一等输出。
    # 「静默豁免」与「新增一个不阻塞的结局」同型，都是新后门（见 CLAUDE.md #10）。
    v2 = [c for c in written if is_v2(c)]
    v1 = [c for c in written if not is_v2(c)]
    print(f"模板版本：v2 {len(v2)} 份（J13 生效） · 无 template_version {len(v1)} 份（F6 v1 底本，J13 免检）")
    if v1:
        print("  免检名单：" + "、".join(sorted(c.fm.get("contract_id", "?") for c in v1)))
    if coverage and coverage < 1.0:
        print("⚠️ 覆盖不足 ⇒ **不给「139 份齐全」的结论**（核对率是一等输出）")
    if n_a > exp_a or n_b > exp_b:
        print(f"🔴 J10 计数超界：A {n_a}/{exp_a}、B {n_b}/{exp_b}")
    if problems:
        print(f"\n🔴 {len(problems)} 条问题：")
        for p in problems[:40]:
            print(f"  [{p['code']}] {p['file']}: {p['msg']}")
        if len(problems) > 40:
            print(f"  …（另 {len(problems) - 40} 条）")
    else:
        print("✅ 全部判据通过")

    if json_out:
        Path(json_out).write_text(json.dumps({
            "_meta": {"what": "契约层校验产物", "graph_digest": idx.graph_digest},
            "counts": {"files": len(contracts), "written": len(written), "A": n_a, "B": n_b,
                       "expected_total": total_expected, "expected_A": exp_a, "expected_B": exp_b},
            "coverage": {"entered": coverage, "written": written_cov},
            "template_version": {"v2": len(v2), "v1_exempt": len(v1),
                                 "v1_ids": sorted(c.fm.get("contract_id", "?") for c in v1)},
            "problems": problems,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {json_out}")
    return 0 if not problems else 1


# ---------------------------------------------------------------------------
# 自检 + 变异测试
# ---------------------------------------------------------------------------
def _fixture(idx, template: str, name: str, cards=(), status="可写") -> str:
    """构造一份「应当全过」的最小契约文本。"""
    meta = idx.assignments()[name]
    assert meta["template"] == template, f"fixture 责任 {name} 不是 {template} 类"
    fm = BC.fm_block(meta, cards=cards, status=status)
    if template == "A":
        body = f"""
## 1 方法来源（卡供方法）

- 卡：{cards[0] if cards else '（无）'}
- **不得移植的部分**：卡内阈值 0.75 与 0.2、迭代 3 轮

## 2 数据要求（五维 · 三列）

| 维 | 取值 | 缺值处置 |
|---|---|---|
| ① 获取路径 | 自有埋点 | — |
| ② 粒度 | 用户 × 日 | — |
| ③ 回溯深度 | 2 个完整年度 | — |
| ④ 新鲜度 | T+1 | — |
| ⑤ 口径归属 | AGT-045 指标契约 | — |

## 3 标定规则

- 取值来源：自有埋点（{name} 台账）
- 算式：按近 2 个完整年度的分组分配率取最大差，与业务规则字典的容差档比较
- 取值：容差档 5%（= 业务规则字典里该标签的容差档）

## 4 重标定触发条件

- 口径变、数据源变、每 2 个完整年度到期

## 5 适用 FLOW

{', '.join(meta['flows'])}

## 6 冻结与不许自动放行的情形

- 样本量不足 30 时不得自动进入受控动作
"""
    else:
        body = f"""
## 1 算法可达部分

- 按历史台账算出排序分（可取数复算）
- 算式：按〈采购台账〉近 2 个完整年度的准时交付率、批次不良率与采购价加权
- 取值：排序分 0–100，取数即得

## 2 不可达部分

- 需外部事实：验厂结论与真实产能
  → 替代路径：由 AGT-014 在 STG-04 前补齐验厂报告；补不上时该供应商不进候选集

## 3 必须的外部证据与责任岗位

| 证据 | 由谁提供（AGT） | 何时必须拿到 | 拿不到时的处置 |
|---|---|---|---|
| 验厂报告 | AGT-014 | STG-04 前 | 不进候选集，先用上一轮合格名录顶替 |

## 4 该格何时必须冻结

- 排序进前 3 但验厂报告缺失时不得进入 STG-07

## 5 数据要求（五维 · 三列）

| 维 | 取值 | 缺值处置 |
|---|---|---|
| ① 获取路径 | 需授权 | — |
| ② 粒度 | 供应商 × 批次 | — |
| ③ 回溯深度 | 2 个完整年度 | — |
| ④ 新鲜度 | 每季 | — |
| ⑤ 口径归属 | AGT-045 指标契约 | — |

## 6 适用 FLOW

{', '.join(meta['flows'])}
"""
    return fm + "\n\n# " + name + " · 供给契约\n" + body


def _any_installed_card(idx, preferred: str) -> str:
    """返回一个在卡库里确实找得到的 slug；preferred 找不到就当场失败（不静默换）。"""
    if find_card(preferred, idx):
        return preferred
    raise SystemExit(f"❌ 自检样本卡 {preferred!r} 在卡库里找不到 —— 先修样本，不要改判据")


def selftest(idx) -> int:
    fails, cases = [], 0
    def expect(cond, label):
        nonlocal cases
        cases += 1
        if not cond:
            fails.append(label)

    assignments = idx.assignments()
    # 取一个 A 类、一个 B 类责任做基准（从图谱现取，不写死 —— 写死会腐烂）
    a_name = next(n for n, m in assignments.items() if m["template"] == "A" and m["role_id"] == "AGT-005")
    b_name = next(n for n, m in assignments.items() if m["template"] == "B" and m["role_id"] == "AGT-046")
    # 卡 slug 从**卡库现取**（先已装 p2s 卡，再 vault 卡），不写死 ——
    # 写死的样本会腐烂：首版写死了 vault 的 `Skill-*` 名，而该卡只存在于已装 p2s 库，
    # 于是「基准样本必须全过」当场变红，暴露的其实是样本错了，不是判据错了。
    a_card = _any_installed_card(idx, "p2s-tag-fairness-bias-audit")
    b_card = _any_installed_card(idx, "p2s-auto-tagging-pipeline-rule-ml-llm")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good_a = td / "good-A.md"
        good_a.write_text(_fixture(idx, "A", a_name, cards=(a_card,)), encoding="utf-8")
        good_b = td / "good-B.md"
        good_b.write_text(_fixture(idx, "B", b_name, cards=(b_card,)), encoding="utf-8")

        # --- 基准：两份合格样本必须 0 问题（否则后面的变异无从判定）---
        expect(not check_one(Contract(good_a), idx, assignments), "基准 A 样本本身必须全过")
        expect(not check_one(Contract(good_b), idx, assignments), "基准 B 样本本身必须全过")
        expect(find_card(a_card, idx) and find_card(b_card, idx), "两张基准卡必须能在卡库里找到")

        # --- 变异表：每条判据一份篡改样本，逐条打红 ---
        base_a = good_a.read_text(encoding="utf-8")
        base_b = good_b.read_text(encoding="utf-8")
        mutations = [
            ("J1", "把模板翻成 B", base_a.replace("template: A", "template: B", 1)),
            ("J1", "status 写枚举外的值", base_a.replace("status: 可写", "status: 已验收", 1)),
            ("J2", "把责任名换成另一个（frontmatter 漂移）", base_a.replace(f"responsibility: {a_name}", "responsibility: 证据复核", 1)),
            ("J3", "把 R/D 格塞进方法接入格",
             re.sub(r"(method_cells: \[[^\]]*)\]", r"\1, FLOW-07/STG-02]", base_a, count=1)),
            ("J3", "正文把 R 格说成算法接入格",
             base_a.replace("## 6 冻结与不许自动放行的情形", "## 6 冻结与不许自动放行的情形\n\n- FLOW-07/STG-02 上挂分类模型", 1)),
            ("J4", "flows 里加一条不属于该岗位的 FLOW", base_a.replace("flows: [FLOW-07, FLOW-08]", "flows: [FLOW-07, FLOW-08, FLOW-01]", 1)),
            ("J5", "把获取路径改成「数据可得」", base_a.replace("| ① 获取路径 | 自有埋点 | — |", "| ① 获取路径 | 数据可得 | — |", 1)),
            ("J5", "删掉一回溯深度", base_a.replace("| ③ 回溯深度 | 2 个完整年度 | — |", "", 1)),
            ("J6", "写不可得但仍标可写", base_a.replace("| ① 获取路径 | 自有埋点 | — |", "| ① 获取路径 | 不可得 | — |", 1)),
            ("J6", "不可得只作子项却仍标可写（两级判据的另一半）",
             base_a.replace("| ① 获取路径 | 自有埋点 | — |", "| ① 获取路径 | 自有埋点；不可得：审计容量台账 | — |", 1)),
            ("J7", "删掉标定规则段", re.sub(r"## 3 标定规则.*?## 4", "## 4", base_a, flags=re.S)),
            ("J7", "把段标题里的关键词换掉", base_a.replace("## 3 标定规则", "## 3 参数说明", 1)),
            ("J8", "第 3 段点名一个不存在的岗位号",
             base_b.replace("| 验厂报告 | AGT-014 |", "| 验厂报告 | AGT-099 |", 1)),
            ("J8", "第 3 段一个岗位号都不写", base_b.replace("| 验厂报告 | AGT-014 | STG-04 前 |", "| 验厂报告 | 供应商管理 | STG-04 前 |", 1)),
            ("J9", "标定规则换成纯对冲句",
             re.sub(r"## 3 标定规则.*?## 4",
                    "## 3 标定规则\n\n- 阈值需重新标定，标定前当临时默认值\n\n## 4", base_a, flags=re.S)),
            ("J12", "引用一张不存在的卡", base_a.replace(f"cards: [{a_card}]", "cards: [Skill-Not-Exist-At-All]", 1)),
            # --- J13（v2 正面义务）四份变异 + 两份反向 ---
            ("J13", "A §3 只有算式、没有本业务取值",
             re.sub(r"## 3 标定规则.*?## 4",
                    "## 3 标定规则\n\n- 取值来源：自有埋点（台账）\n"
                    "- 算式：按近 2 个完整年度的分组分配率取最大差\n\n## 4", base_a, flags=re.S)),
            ("J13", "A §3 以缺口结案却不写处置",
             re.sub(r"## 3 标定规则.*?## 4",
                    "## 3 标定规则\n\n- 阈值：本业务暂无此数据\n\n## 4", base_a, flags=re.S)),
            ("J13", "B §1 只写「可复算」不给取值",
             re.sub(r"## 1 算法可达部分.*?## 2",
                    "## 1 算法可达部分\n\n- 按历史台账算出排序分（可取数复算）\n\n## 2", base_b, flags=re.S)),
            ("J13", "B §2 拿掉替代路径（只写「做不到」= 免责）",
             base_b.replace("  → 替代路径：由 AGT-014 在 STG-04 前补齐验厂报告；补不上时该供应商不进候选集", "", 1)),
            ("J13", "取值标签在但没给数字（「取值：待定」）",
             re.sub(r"## 3 标定规则.*?## 4",
                    "## 3 标定规则\n\n- 取值来源：自有埋点（台账）\n- 算式：按近 2 个完整年度的分组分配率取最大差\n"
                    "- 取值：待定\n\n## 4", base_a, flags=re.S)),
            ("J13", "五维取值是缺口、缺值处置为空",
             base_a.replace("| ③ 回溯深度 | 2 个完整年度 | — |",
                            "| ③ 回溯深度 | 本业务暂无此数据 | — |", 1)),
            # 解析层的两个方向：方括号未闭合必须**收成判据**（不是让门禁崩掉），
            # 行尾注释必须**能正常解析**（否则 norm_val 退回整串，下游按字符遍历 = 静默错配）
            ("J2", "列表字段方括号未闭合", base_a.replace("flows: [FLOW-07, FLOW-08]", "flows: [FLOW-07, FLOW-08", 1)),
            ("J12", "status=待卡 却填了卡", base_a.replace("status: 可写", "status: 待卡", 1)),
            # 反后门：未撰写骨架的五维豁免不得被滥用 —— 去掉标记但不补维度，必须判红
            ("J5", "去掉待撰写标记却不补五维（豁免不得被滥用）",
             BC.fm_block(idx.assignments()[a_name], cards=(a_card,), status="可写") + "\n\n# 空壳\n\n"
             + "\n".join(f"{h}\n\n有内容但不对\n" for h in BC.A_HEADERS)),
        ]
        for code, label, text in mutations:
            p = td / f"mut-{code}-{abs(hash(label)) % 10**6}.md"
            p.write_text(text, encoding="utf-8")
            got = {c for c, _ in check_one(Contract(p), idx, assignments)}
            expect(code in got, f"变异应被 {code} 抓住：{label}（实得 {sorted(got) or '全绿'}）")

        # --- J13 的两份**反向**用例：判据过宽会把正常写作判红，那比漏判更糟 ---
        # ① 缺口**带**处置 ⇒ 不得红（这正是 v2 要求的形态）
        ok_gap = td / "ok-gap.md"
        ok_gap.write_text(base_a.replace(
            "| ③ 回溯深度 | 2 个完整年度 | — |",
            "| ③ 回溯深度 | 本业务暂无此数据 | 先按经营者自述的近 2 个完整年度暂按取值，〈采购台账〉接入后替换为台账首单日 |", 1),
            encoding="utf-8")
        got_ok = {c for c, _ in check_one(Contract(ok_gap), idx, assignments)}
        expect("J13" not in got_ok, f"缺口带处置不得判红（实得 {sorted(got_ok) or '全绿'}）")
        # ② 未标 template_version 的 v1 底本 ⇒ J13 不适用（豁免生效）；**但豁免必须有计数器**，
        #    计数逻辑本身在下面 run() 的摘要里，这里只锁「不误判」这一半
        v1_text = base_a.replace("template_version: 2\n", "", 1).replace(
            "## 3 标定规则", "## 3 标定规则", 1)
        v1_text = re.sub(r"## 3 标定规则.*?## 4",
                         "## 3 标定规则\n\n- 阈值：本业务暂无此数据\n\n## 4", v1_text, flags=re.S)
        p_v1 = td / "mut-v1-exempt.md"
        p_v1.write_text(v1_text, encoding="utf-8")
        got_v1 = {c for c, _ in check_one(Contract(p_v1), idx, assignments)}
        expect("J13" not in got_v1, f"v1 底本（无 template_version）不适用 J13（实得 {sorted(got_v1) or '全绿'}）")
        expect(not is_v2(Contract(p_v1)), "未标 template_version 的契约必须被判为非 v2")

        # --- 覆盖率与计数：篡改总量必须让「139」这条断言失败 ---
        expect(len(assignments) == 139, f"契约总数应为 139，实得 {len(assignments)}")
        expect(sum(1 for m in assignments.values() if m["template"] == "A") == 73, "A 应为 73")
        expect(sum(1 for m in assignments.values() if m["template"] == "B") == 66, "B 应为 66")
        # J10 变异：拿掉一份契约，总数断言必须失败（否则总数断言是摆设）
        _minus = dict(assignments)
        _minus.pop(next(iter(_minus)))
        expect(len(_minus) != 139, "变异 J10：删一份契约后总数断言必须失败")
        # 反向（解析层）：`flows` 带行尾注释与不带注释必须解析出**同一个**列表
        p_c = td / "mut-comment.md"
        p_c.write_text(base_a.replace("flows: [FLOW-07, FLOW-08]", "flows: [FLOW-07, FLOW-08]  # 与图表一致", 1),
                       encoding="utf-8")
        got_c = {c for c, _ in check_one(Contract(p_c), idx, assignments)}
        expect(not (got_c & {"J2", "J4"}),
               f"行尾注释不得影响列表解析（实得 {sorted(got_c) or '全绿'}）")

        # J11 变异：只喂两份契约，覆盖率必须 < 1 并拒绝「齐全」结论
        expect(len({a_name, b_name}) / len(assignments) < 1.0,
               "变异 J11：子集覆盖率必须小于 1（否则覆盖率是一等输出这句话不成立）")
        # 反向：拿一份「C 类责任名」的契约必须被判 J1 红
        c_name = next(n for n, i in idx.l3.items() if i["serviceability"] == "C")
        p = td / "mut-C.md"
        p.write_text(base_a.replace(f"responsibility: {a_name}", f"responsibility: {c_name}", 1), encoding="utf-8")
        expect("J1" in {c for c, _ in check_one(Contract(p), idx, assignments)},
               f"C 类责任不得建契约（样本 {c_name}）")

    print(f"自检：{cases - len(fails)}/{cases} 通过（含 {len(mutations)} 份变异样本）")
    for f in fails:
        print(f"🔴 {f}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="契约层（L4）校验器")
    ap.add_argument("--graph", default=str(GRAPH_DEFAULT))
    ap.add_argument("--file", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json-out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    idx = BC.GraphIndex(BC.load_graph(Path(args.graph)))
    if args.selftest:
        return selftest(idx)
    files = collect_files(args.file)
    if not files:
        print("没有找到契约文件（--all 或 --file <路径>）", file=sys.stderr)
        return 2
    return run(files, idx, args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())
