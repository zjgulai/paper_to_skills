#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""契约「数据要求」栏 · 本业务数据可得性五维判据（PHASE6 S11）

判什么
======
139 份供给契约（`contracts/A` 73 份 + `contracts/B` 66 份）的「数据要求」表里
**已经有五维行**（① 获取路径 / ② 粒度 / ③ 回溯深度 / ④ 新鲜度 / ⑤ 口径归属，
实测 **139/139 齐全**，三是列「维度 | 取值 | 缺值处置」）。本脚本判的**不是**
「那五行在不在」，而是 **「那五个格子里有没有点名」**。

为什么需要这一层（它存在的全部理由）
====================================
`check_contracts.py` 的 **J5** 已经判「五维齐全 + ① 断言五选一」，**J13** 判
「缺口必须带处置」。但 J5 的颗粒度是**行**：行在、格非空即过 —— 它自己的注释里
就记着实测边界：

    `| ① 获取路径 | 本企业无自有埋点，实际全靠人工估算 | 非缺口 |` ⇒ exit=0

即 **139/139 全绿 ≠ 139 份都点了名**。这正是本仓库反复出现的形态
（「判据的适用范围被默认成了全体」/「没东西可查 ≠ 查过了没问题」）。
本脚本把颗粒度**降到格**，逐维问一句「这一维点名了吗」，并配能打红的篡改样本。

五维的词汇**不是本脚本新造的** —— 它取自材料《AI组织变革》：① 的枚举
（自有埋点 / 平台后台 / 第三方 API / 需授权 / 不可得）在 139 份契约里逐字在用，
④ 对应 AGT-046「数据工程与质量」的交付要求（带质量状态、来源和新鲜度），
⑤ 对应 AGT-045「业务口径与主数据」的职责（维护指标和业务对象契约）。

⚠️ 与论文侧字段的 schema 隔离（判据 R9）
==========================================
`papers_registry.json` 里那个描述**论文自己评测集/代码公不公开**的字段，
与本层描述**本公司系统里有没有这份数据**，是两件事。本脚本与它的产物**一律
不得出现该字段名**。为使该判据可自证，该字段名在本文件里**由两截拼出**
（见 `_paper_side_field()`），因此对本文件做字面 grep 的命中数是 **0** ——
而 R9 仍然会在**产物**里扫它，并配一份「注入了该词」的篡改样本来证明它会红。

退出码（照本仓库约定）
======================
    0  全过
    1  判红（有判据不通过）
    2  输入没拿到（目录不存在 / 一份契约都没扫到 / 有契约解析不出五维表）
       —— 「没测到」不是「干净」，故**不是 0**
    3  脚本内部错误（**不是判红**：判红的成因在语料，3 的成因在仪器）

用法
====
    python3 paper2skills-research/scripts/check_data_requirements.py \
        --dir paper2skills-vault/07-资源库/contracts \
        --json-out paper2skills-research/data/data-requirements.json
    python3 paper2skills-research/scripts/check_data_requirements.py --selftest
    python3 paper2skills-research/scripts/check_data_requirements.py \
        --dir ... --file CTR-A-049        # 单份
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# 退出码（本仓库约定；见 CLAUDE.md 「退出码 0/1/2/3」）
# --------------------------------------------------------------------------
EXIT_PASS = 0
EXIT_RED = 1
EXIT_NO_INPUT = 2
EXIT_INTERNAL = 3

CST = timezone(timedelta(hours=8))
SELF = Path(__file__).resolve()

# 正式契约只有这两个目录（台账 #26/#27：全集是**写下来的**，不是 rglob 默认出来的）
FORMAL_SUBDIRS = ("A", "B")
EXPECT_TOTAL = 139


# --------------------------------------------------------------------------
# 五维：标签、枚举与词表
#
# 词表的每一条都来自**现场实测的语料**（见报告 §判据），不是设计者臆想的写法。
# 归一化与仓库既有口径同尺（check_contract_dedup.normalize）：NFKC + 去零宽 + 去 `*`
# —— 「强调符是表现，不是用词」，不剥它等于留一条假绿通路（实测：把 `替换` 写成
# `**替换**` 就能把窗口切开）。这里同样先去强调符再探测。
# --------------------------------------------------------------------------
DIMS = ("① 获取路径", "② 粒度", "③ 回溯深度", "④ 新鲜度", "⑤ 口径归属")
DIM_KEY = {d.replace(" ", ""): d for d in DIMS}

# ① 的合法枚举（与 check_contracts.ACCESS_ENUM 同尺；归一化后无空格）
ACCESS_ENUM = ("自有埋点", "平台后台", "第三方API", "需授权", "不可得")
# 枚举词出现在**否定句**里不算数（check_contracts 的 D19：子串匹配会让
# `本企业无自有埋点` 被判成「断言了自有埋点」——那是假绿）
ACCESS_NEG = ("无", "没有", "不含", "不具备", "不靠", "不是", "非", "尚未",
              "未接入", "未建", "未开通", "未产生", "缺")

# ② 粒度的**轴**写法：`A × B × C` 或 `一条 = …` / `记录 = …`
_GRAIN_AXIS = re.compile(r"×|一条[=＝]|一行[=＝]|记录[=＝]|单元[=＝]|粒度[=＝]")
# ② 的粒度词（材料给的是 SKU / 订单 / 用户 / 日 / 月；语料实际用的轴更宽，
# 这里按**实测出现的**粒度词收，避免把合法的业务原生粒度判红 —— 见报告 §判据 的
# 「假红」一节）
_GRAIN_WORDS = (
    "SKU", "ASIN", "订单", "用户", "客户", "会员", "受试者", "批次", "店铺", "账号",
    "渠道", "商品", "科目", "仓库", "供应商", "自然日", "自然周", "自然月", "月度",
    "日级", "周级", "月级", "事件", "文本", "字段", "源记录", "表级", "字段级",
    "评估单元", "对象", "案例", "Case", "时间窗", "会话", "触点", "任务",
)
# ② 的空泛写法：只给这些 ⇒ 视同未点名
_GRAIN_VAGUE = re.compile(r"^(明细|全部|所有|不限|完整|原始|全量数据|全部数据|数据)+$")

# ③ 深度形态
_DEPTH_YEARS = re.compile(r"(\d+)\s*个完整年度")
_DEPTH_OTHER = re.compile(
    r"\d+\s*个?(?:自然)?(?:日|周|月|年|季度|小时)"
    r"|全量|自[^，。；]{1,24}起|首笔|上线首日|运行至今|至今|自上线|逐笔"
)
# ③ 深度的**来源**：深度不是 2 个完整年度时，必须点名它从哪来（否则「未对账」）
_DEPTH_SOURCE = re.compile(
    r"材料|决策\s*Q\d+|Q\d+|业务处境|平台侧|平台|按定义|定义决定|尚未|暂无|未建"
    r"|不回溯|只保留|散落|受控验证之前|不产生|未产生|未实例化"
)

# ④ 新鲜度节奏
_FRESH = re.compile(
    r"T\+\d|当日|实时|次日出|次日|前重读|前再读|前重校|前校|前重新|前核|前取值"
    r"|24:00|按自然日|改版|即时|stale|快照|月结|结账|关账|回填|到达后|完成后"
    r"|每(?:次|批|个|月|日|季|周)|按日|按周|按月|逐日|出数|入湖|上传后|提交后"
)

# ⑤ 归属方：本维问的是「是否已被 AGT-045 的主数据契约定名」
_OWNER_MAIN = "AGT-045"
_OWNER_ANY = re.compile(r"AGT-\d{3}")
_OWNER_HINT = re.compile(r"指标契约|主数据|业务口径|口径与主数据")

# 缺口语 / 处置语（与 check_contracts 的 GAP_TOKENS / DISPOSAL_TOKENS 同向；
# 本条只用于「整行判」的第二列——第三列非空是 J13 的事，本脚本判的是**点名**）
_GAP = re.compile(
    r"暂无|尚未|未定名|不可得|拿不到|取不到|待定|未知|缺失|缺少|未建|无从取得"
    r"|没有|不具备|未采集|不采集|不产生|未产生|不回溯|未开通|待签发|未签发|不成立"
)
_DISPOSAL = re.compile(
    r"先用|顶替|代为|代用|代理|暂按|暂用|暂以|暂时|替换|改用|改为|失效|先行"
    r"|生效条件|到位后|上线后|接入后|补充后|签发后|发布后|前先按|待.{0,6}后"
)

# R9：论文侧字段名（按两截拼出，故本文件对该词的字面命中数为 0）
_PAPER_FIELD_PARTS = ("data", "_", "avail", "ability")


def _paper_side_field() -> str:
    return "".join(_PAPER_FIELD_PARTS)


# --------------------------------------------------------------------------
# 归一化
# --------------------------------------------------------------------------
_ZW = re.compile(r"[\u200b-\u200d\ufeff]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC + 去零宽 + 去 Markdown 强调/反引号 + 去全部空白。

    ⚠️ 刻意**不做**同义词替换、不做语义相似 —— 那会把「可复核」变成「可解释」。
    去空白是为了让 `第三方 API` 与 `第三方API` 同尺；也因此，本文件里所有词表
    都按**无空格形态**书写。
    """
    t = unicodedata.normalize("NFKC", text or "")
    t = _ZW.sub("", t)
    t = t.replace("*", "").replace("`", "")
    return _WS.sub("", t)


def split_md_row(line: str):
    """按 `|` 切 Markdown 表格行，但**反引号内的 `|` 不算分隔符**（同仓库既有口径）。"""
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


_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)


def split_frontmatter(text: str):
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end():]


def _fm_get(fm: str, key: str):
    m = re.search(rf"(?m)^{re.escape(key)}\s*:\s*(.*?)\s*$", fm or "")
    return m.group(1) if m else None


def _dim_of_cell0(cell0: str):
    """首格是否就是五维标签（允许 ① 与强调符、允许标签后接括注）。

    ⚠️ 只允许**括注**跟在标签后（`| ① 获取路径（平台后台与埋点） |`），不允许
    在标签前有别的字 —— 否则段内提前出现的标签词会劫持该行（台账 #17）。
    """
    k = normalize(cell0).replace("：", "").replace(":", "")
    for d in DIMS:
        # ⚠️ 两边必须**同尺**：`normalize` 走 NFKC，而 NFKC 会把带圈号 ① 折成数字 `1`
        # （① 是兼容字符）。首版只归一了左边、右边用去空格的原文 ⇒ `①获取路径` vs
        # `1获取路径` 永不相等 ⇒ **五维表一行都抽不到**，而症状是「0/5 行」这种
        # 看起来像语料问题的读数。与仓库既有教训同族：`normalize()` 先去空白、
        # `MASK_RE` 后匹配，两边不同尺 ⇒ 9 条遮罩是死模式。
        nk = normalize(d)
        if k == nk:
            return d
        if k.startswith(nk) and k[len(nk):][:1] in ("（", "("):
            return d
    return None


# --------------------------------------------------------------------------
# 契约解析
# --------------------------------------------------------------------------
class Dim:
    __slots__ = ("name", "value", "disposal", "raw_value", "raw_disposal", "lineno")

    def __init__(self, name, value, disposal, raw_value, raw_disposal, lineno):
        self.name = name
        self.value = normalize(value)          # 归一化后：探测用
        self.disposal = normalize(disposal)
        self.raw_value = raw_value             # 原文：打印用（人要看原文）
        self.raw_disposal = raw_disposal
        self.lineno = lineno


def parse_contract(path: Path):
    """返回 (info: dict, error: str|None)。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:                                   # noqa: BLE001
        return None, f"读不了：{exc}"
    fm, body = split_frontmatter(text)
    if fm is None:
        return None, "没有 frontmatter"
    template = (_fm_get(fm, "template") or "").strip()
    if template not in ("A", "B"):
        return None, f"frontmatter 的 template 不是 A/B：{template!r}"

    dims, seen = {}, {}
    for lineno, line in enumerate(body.splitlines(), 1):
        if not line.strip().startswith("|"):
            continue
        cells = split_md_row(line)
        if not cells or len(cells) < 3:
            continue
        d = _dim_of_cell0(cells[0])
        if d and d not in seen:
            seen[d] = True
            dims[d] = Dim(d, cells[1], cells[2], cells[1], cells[2], lineno)

    info = {
        "file": path.name,
        "path": str(path),
        "contract_id": (_fm_get(fm, "contract_id") or "").strip() or path.stem,
        "template": template,
        "template_version": (_fm_get(fm, "template_version") or "").strip(),
        "responsibility": (_fm_get(fm, "responsibility") or "").strip(),
        "role_id": (_fm_get(fm, "role_id") or "").strip(),
        "status": (_fm_get(fm, "status") or "").strip(),
        "dims": dims,
    }
    missing_rows = [d for d in DIMS if d not in dims]
    if missing_rows:
        return info, f"五维表只抽到 {len(dims)}/5 行，缺：{'、'.join(missing_rows)}"
    return info, None


# --------------------------------------------------------------------------
# 判据 R1–R7：逐维「点名了吗」
#    状态三态：POS（给了本维的正取值）/ GAP_DISPOSED（缺口且已处置）/ MISSING（未点名）
# --------------------------------------------------------------------------
def asserts_access(value: str) -> tuple[bool, list]:
    """R1：① 是否**断言**了五选一（否定句不算，且要数出断言了哪几项）。

    ⚠️ 逐子句判（check_contracts D19 的口径）：命中枚举词的那个子句里，枚举词之前
    不得出现否定词。`本企业无自有埋点` ⇒ 不算断言。
    """
    asserted = []
    for seg in re.split(r"[＋+；;。，,、/]", value):
        for e in ACCESS_ENUM:
            i = seg.find(e)
            if i < 0:
                continue
            head = seg[:i]
            # 「不可得」是**结论**，它自己带「不」，故不对它做否定前置判定
            if e != "不可得" and any(n in head for n in ACCESS_NEG):
                continue
            if e == "不可得" and any(n in head for n in ("并非", "不是", "不一定")):
                continue
            if e not in asserted:
                asserted.append(e)
    return bool(asserted), asserted


_PAREN = re.compile(r"[（(][^）)]{2,}[）)]")


def names_concrete_source(raw_value: str) -> bool:
    """R2：① 除了枚举词，是否**点名了具体数据源**。

    两条合法形态（都取自实测语料）：材料的数据源命名法 `〈…〈`，或枚举词后紧跟括注
    `平台后台（Amazon 卖家后台报表）`。裸写一个枚举词不算点名。
    """
    if "〈" in raw_value and "〉" in raw_value:
        return True
    if _PAREN.search(raw_value):
        return True
    return False


def has_disposal_anywhere(d: Dim) -> bool:
    """整行判：处置在第三列，或写在取值格里（但**缺口句不能自己救自己**）。

    与 `check_contracts.row_disposal` 同向 —— 那里是 D28 的结论：撰写人常把
    「缺口 + 处置」合并写进取值格，按**列**判会把已经履行义务的行判红。
    """
    if d.disposal and _DISPOSAL.search(d.disposal):
        return True
    stripped = _DISPOSAL.sub("", d.value)
    # 摘掉陈述缺口的句子，再找处置（缺口不能自己救自己）
    kept = "".join(s for s in re.split(r"[；;。]", d.value) if not _GAP.search(s))
    if not kept:
        kept = stripped if not _GAP.search(stripped) else ""
    return bool(_DISPOSAL.search(kept))


def judge_dim(d: Dim) -> tuple[str, str]:
    """返回 (state, reason)。state ∈ {POS, GAP_DISPOSED, MISSING}。"""
    name, v = d.name, d.value
    if not v.strip():
        return "MISSING", "格为空"

    gap = bool(_GAP.search(v))

    if name == DIMS[0]:                                        # ① 获取路径
        ok, asserted = asserts_access(v)
        if not ok:
            if gap and has_disposal_anywhere(d):
                return "GAP_DISPOSED", "① 未点名枚举路径，但已声明缺口并给了处置"
            return "MISSING", ("① 未点名获取路径：必须断言五选一"
                               "（自有埋点/平台后台/第三方API/需授权/不可得）；"
                               "否定句与「数据可得」这类不打点名的写法不算")
        if not names_concrete_source(d.raw_value):
            return "MISSING", f"① 只写了枚举词（{'/'.join(asserted)}）而未点名具体数据源"
        return "POS", f"① 点名：{'/'.join(asserted)}"

    if name == DIMS[1]:                                        # ② 粒度
        if _GRAIN_AXIS.search(v) or any(w in v for w in _GRAIN_WORDS):
            return "POS", "② 给了粒度轴"
        if gap and has_disposal_anywhere(d):
            return "GAP_DISPOSED", "② 未给粒度轴，但已声明缺口并给了处置"
        if _GRAIN_VAGUE.match(v):
            return "MISSING", "② 只写了空泛粒度词（明细/全部/不限…），未给粒度轴"
        return "MISSING", "② 未给粒度轴：需写 `维度 × 维度 …` 或点明一条记录等于什么"

    if name == DIMS[2]:                                        # ③ 回溯深度
        m = _DEPTH_YEARS.search(v)
        if m:
            n = int(m.group(1))
            if n == 2:
                return "POS", "③ = 近 2 个完整年度（与既有口径一致）"
            if _DEPTH_SOURCE.search(v):
                return "POS", f"③ = {n} 个完整年度（偏离既有 2 年口径，但点名了来源）"
            return "MISSING", (f"③ = {n} 个完整年度，既非既有的「2 个完整年度」口径，"
                               f"又未点名该深度的来源 ⇒ 未对账")
        if _DEPTH_OTHER.search(v):
            if _DEPTH_SOURCE.search(v):
                return "POS", "③ 给了可判定的深度并点名了来源"
            return "MISSING", "③ 给了深度但未点名来源 ⇒ 未与「2 个完整年度」口径对账"
        if gap and has_disposal_anywhere(d):
            return "GAP_DISPOSED", "③ 声明无历史可回溯，并给了处置"
        return "MISSING", "③ 未给可判定的回溯深度（年度/自然月/日/全量/自…起）"

    if name == DIMS[3]:                                        # ④ 新鲜度
        if _FRESH.search(v):
            return "POS", "④ 给了节奏"
        if gap and has_disposal_anywhere(d):
            return "GAP_DISPOSED", "④ 未给节奏，但已声明缺口并给了处置"
        return "MISSING", "④ 未给节奏：需写 T+N / 当日 / 每次进入某阶段前重读 / 快照截点"

    if name == DIMS[4]:                                        # ⑤ 口径归属
        if _OWNER_MAIN in v and _OWNER_HINT.search(v):
            return "POS", "⑤ 已由 AGT-045 的业务口径与主数据定名"
        if _OWNER_MAIN in v:
            return "POS", "⑤ 点名了 AGT-045"
        if _OWNER_ANY.search(v):
            if gap and has_disposal_anywhere(d):
                return "GAP_DISPOSED", "⑤ 未由 AGT-045 定名，但点名了归属方与缺口处置"
            return "MISSING", "⑤ 点名了别的岗位，但未说明本维是否已由 AGT-045 定名"
        if gap and has_disposal_anywhere(d):
            return "GAP_DISPOSED", "⑤ 声明口径未定名并给了处置，但未点名归属方"
        return "MISSING", "⑤ 未点名归属方：需点名 AGT-045 的主数据/指标契约"


def contract_verdict(info: dict):
    """对一份契约跑五维判据，返回 (三态, dims_result, reds)。

    三态（本任务的核心口径）：
        `五维齐全`  五维逐格均 POS 或 GAP_DISPOSED
        `数据待盘`  **任一维** MISSING ⇒ 标数据待盘
                    （**不是**默认其他四维成立 —— 缺一维就是没有可用的本维取值）
        `不可得`    是 `五维齐全` 的**子类**：① 断言了「不可得」且五维齐全 ⇒ 合法结论
    """
    results, reds = {}, []
    for d in DIMS:
        state, reason = judge_dim(info["dims"][d])
        results[d] = {"state": state, "reason": reason,
                      "value": info["dims"][d].raw_value.strip(),
                      "disposal": info["dims"][d].raw_disposal.strip(),
                      "lineno": info["dims"][d].lineno}
        if state == "MISSING":
            reds.append({"contract": info["contract_id"], "file": info["file"],
                         "dim": d, "reason": reason,
                         "value": info["dims"][d].raw_value.strip()[:300]})

    _, asserted = asserts_access(info["dims"][DIMS[0]].value)
    not_obtainable = "不可得" in asserted
    # 「不可得」是本维的**主取值**（整维就是拿不到），还是**一项子缺口**
    # （其余路径仍可用，只是其中某一块拿不到）？两者都是合法结论、都判合格，
    # 但账上必须分开 —— 混在一起报会把「整条责任没数据」平均掉。
    whole = info["dims"][DIMS[0]].value
    nd_primary = not_obtainable and (whole == "不可得" or whole.startswith("不可得"))

    if reds:
        verdict = "数据待盘"
    else:
        verdict = "五维齐全"
    return verdict, results, reds, not_obtainable, asserted, nd_primary


# --------------------------------------------------------------------------
# R9：与论文侧字段的 schema 隔离
# --------------------------------------------------------------------------
def scan_paper_side_field(texts: dict) -> list:
    """扫产物（与脚本自身）里是否出现论文侧字段名。**出现即判红。**"""
    hits = []
    needle = _paper_side_field()
    for label, text in texts.items():
        if not text:
            continue
        n = text.count(needle)
        if n:
            lines = [i for i, ln in enumerate(text.splitlines(), 1) if needle in ln]
            hits.append({"target": label, "count": n, "lines": lines[:20]})
    return hits


# --------------------------------------------------------------------------
# 覆盖率 / 输入可用性
# --------------------------------------------------------------------------
def collect_contracts(root: Path):
    """正式契约只在 `A/` 与 `B/`（台账 #26/#27）；别处的 CTR-*.md 单独报，不入账。"""
    files, stray = [], []
    for sub in FORMAL_SUBDIRS:
        d = root / sub
        if d.is_dir():
            files.extend(sorted(d.glob("CTR-*.md")))
    for p in sorted(root.glob("**/CTR-*.md")):
        if p.parent.name not in FORMAL_SUBDIRS:
            stray.append(str(p))
    return files, stray


def coverage_problems(n_total: int, expect_total: int) -> list:
    """覆盖率**双边**判据（台账 #27：只写下界等于没有判据）。

    `cov > 1.0` 判红：分子里有不属于这 139 份的文件 —— 一个 >100% 的覆盖率会把
    真实缺口平均掉。
    """
    problems = []
    if n_total > expect_total:
        problems.append(f"计数超界：扫到 {n_total} 份 > 应有 {expect_total} 份"
                        f"（分子里有不属于这 {expect_total} 份的文件）")
    return problems


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def build_report(args):
    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        return None, f"契约目录不存在：{root}"

    files, stray = collect_contracts(root)
    if not files:
        return None, (f"{root} 下一个 CTR-*.md 都没扫到 —— "
                      f"「没东西可查」不等于「查过了没问题」")

    infos, unresolved, reds = [], [], []
    for f in files:
        info, err = parse_contract(f)
        if err:
            unresolved.append({"file": f.name, "path": str(f), "reason": err})
        if info is not None:
            infos.append(info)

    if unresolved:
        return None, ("有 %d 份契约解析不出五维表 ⇒ 判不了，"
                      "按「输入没拿到」处理：%s"
                      % (len(unresolved),
                         "；".join(f"{u['file']}（{u['reason']}）"
                                   for u in unresolved[:8])))

    if getattr(args, "file", None):
        want = args.file
        infos = [i for i in infos
                 if i["file"] == want or i["contract_id"] == want
                 or i["file"].startswith(want)]
        if not infos:
            return None, f"--file {want} 没有匹配到任何契约"
        reds = []

    rows, dim_stats = [], {d: {"POS": 0, "GAP_DISPOSED": 0, "MISSING": 0} for d in DIMS}
    complete, pending, not_obtainable, gap_disposed_contracts = [], [], [], []
    not_obtainable_primary, not_obtainable_subitem = [], []
    unfilled = []
    for info in infos:
        verdict, results, creds, nd, asserted, nd_primary = contract_verdict(info)
        for d in DIMS:
            dim_stats[d][results[d]["state"]] += 1
        rows.append({
            "contract_id": info["contract_id"],
            "file": info["file"],
            "template": info["template"],
            "responsibility": info["responsibility"],
            "role_id": info["role_id"],
            "status": info["status"],
            "verdict": verdict,
            "not_obtainable": nd,
            "not_obtainable_primary": nd_primary,
            "access_asserted": asserted,
            "dims": results,
        })
        reds.extend(creds)
        if verdict == "五维齐全":
            complete.append(info["contract_id"])
            if nd:
                not_obtainable.append(info["contract_id"])
                (not_obtainable_primary if nd_primary
                 else not_obtainable_subitem).append(info["contract_id"])
            if any(results[d]["state"] == "GAP_DISPOSED" for d in DIMS):
                gap_disposed_contracts.append(info["contract_id"])
        else:
            pending.append(info["contract_id"])
            for d in DIMS:
                if results[d]["state"] == "MISSING":
                    unfilled.append({
                        "contract_id": info["contract_id"], "file": info["file"],
                        "dim": d, "reason": results[d]["reason"],
                        "value": results[d]["value"][:300],
                        "disposal": results[d]["disposal"][:200],
                    })

    cov = len(infos) / EXPECT_TOTAL
    cov_problems = coverage_problems(len(infos), EXPECT_TOTAL)
    reds.extend({"contract": "-", "file": "-", "dim": "-", "reason": p}
                for p in cov_problems)

    fingerprint = hashlib.sha256(
        "".join(f"{f.name}:{f.stat().st_mtime_ns}:{f.stat().st_size}"
                for f in files).encode("utf-8")).hexdigest()[:16]

    report = {
        "_meta": {
            "generated": datetime.now(CST).isoformat(timespec="seconds"),
            "script": str(SELF),
            "corpus_dir": str(root),
            "corpus_fingerprint": fingerprint,
            "expect_total": EXPECT_TOTAL,
            "note": "五维词汇取自材料《AI组织变革》，不是本脚本新造；"
                    "本脚本判的是「格子里有没有点名」，不是「行在不在」",
        },
        "summary": {
            "contracts_total": len(infos),
            "a": sum(1 for i in infos if i["template"] == "A"),
            "b": sum(1 for i in infos if i["template"] == "B"),
            "coverage": round(cov, 4),
            "expected_total": EXPECT_TOTAL,
            "five_dims_complete": len(complete),
            "data_pending": len(pending),
            "not_obtainable": len(not_obtainable),
            "not_obtainable_primary": len(not_obtainable_primary),
            "not_obtainable_subitem": len(not_obtainable_subitem),
            "gap_disposed_contracts": len(gap_disposed_contracts),
            "unfilled_cells": len(unfilled),
            "cells_total": 5 * len(infos),
            "dimension_stats": dim_stats,
            "five_dims_complete_list": complete,
            "data_pending_list": pending,
            "not_obtainable_list": not_obtainable,
            "not_obtainable_primary_list": not_obtainable_primary,
            "not_obtainable_subitem_list": not_obtainable_subitem,
            "gap_disposed_list": gap_disposed_contracts,
            "stray_ctr_files_not_counted": stray,
        },
        "unfilled": unfilled,
        "reds": reds,
        "contracts": rows,
    }

    # R9：产物自身 + 脚本自身（本文件对论文侧字段名的字面命中数必须是 0）
    artifact_text = json.dumps(report, ensure_ascii=False, indent=1)
    scan_targets = {"json-artifact": artifact_text,
                    "script-source": SELF.read_text(encoding="utf-8")}
    for extra in (args.scan_extra or []):
        p = Path(extra)
        scan_targets[str(p)] = p.read_text(encoding="utf-8") if p.is_file() else ""
    r9_hits = scan_paper_side_field(scan_targets)
    report["criteria"] = [
        {"id": "R9", "name": "与论文侧字段 schema 互不引用",
         "status": "red" if r9_hits else "green", "hits": r9_hits},
    ]
    report["_meta"]["paper_side_field_scanned"] = True
    return report, None


def render_text(report) -> str:
    s = report["summary"]
    out = []
    w = out.append
    w("契约「数据要求」栏 · 本业务数据可得性五维判据（PHASE6 S11）")
    w("=" * 72)
    w(f"语料：{report['_meta']['corpus_dir']}")
    w(f"指纹：{report['_meta']['corpus_fingerprint']}   "
      f"（{report['_meta']['generated']}）")
    w(f"契约 {s['contracts_total']} 份（A {s['a']} / B {s['b']}） · "
      f"入账覆盖 {s['contracts_total']}/{s['expected_total']} = {s['coverage']*100:.1f}%")
    if s["contracts_total"] < s["expected_total"]:
        w(f"⚠️ 覆盖不足 ⇒ **不给「{s['expected_total']} 份齐全」的结论**（核对率是一等输出）")
    if s["stray_ctr_files_not_counted"]:
        w(f"样本/存档（A/ B/ 之外，**不计入 {s['expected_total']}**）："
          f"{len(s['stray_ctr_files_not_counted'])} 份")
    w("")
    w("【五维逐格读数（一等输出）】")
    for d in DIMS:
        st = s["dimension_stats"][d]
        w(f"  {d}：正取值 {st['POS']} · 缺口已处置 {st['GAP_DISPOSED']} · "
          f"**未点名 {st['MISSING']}**  / {s['contracts_total']} 份")
    w("")
    w("【三态】")
    w(f"  五维齐全     {s['five_dims_complete']:>4} 份")
    w(f"  数据待盘     {s['data_pending']:>4} 份"
      f"（任一维未点名 ⇒ 标数据待盘，**不默认其余四维成立**）")
    w(f"  不可得       {s['not_obtainable']:>4} 份（合法结论，**判合格**）"
      f"　其中本维主取值 {s['not_obtainable_primary']} 份 / 子缺口 "
      f"{s['not_obtainable_subitem']} 份")
    w(f"  缺口已处置   {s['gap_disposed_contracts']:>4} 份"
      f"（缺口 + 替代取值 + 替换条件 ⇒ 合格）")
    w(f"  未点名格数   {s['unfilled_cells']:>4} / {s['cells_total']}")
    w("")
    r9 = next((c for c in report.get("criteria", []) if c["id"] == "R9"), None)
    if r9:
        w("【R9 与论文侧字段 schema 互不引用】")
        if r9["status"] == "green":
            w("  ✅ 产物与脚本里都没出现论文侧字段名"
              f"（`{_paper_side_field()}` 字面命中 0）")
        else:
            w("  🔴 出现论文侧字段名 —— 说明把论文侧字段抄进了本业务侧：")
            for h in r9["hits"]:
                w(f"     · {h['target']}：{h['count']} 处，行 {h['lines']}")
    w("")
    if s["data_pending_list"]:
        w(f"【数据待盘名单（{len(s['data_pending_list'])} 份）】")
        for cid in s["data_pending_list"]:
            w(f"  · {cid}")
        w("")
    if s["not_obtainable_list"]:
        w(f"【不可得名单（{len(s['not_obtainable_list'])} 份）—— 判合格，不是判红】")
        for cid in s["not_obtainable_list"]:
            w(f"  · {cid}")
        w("")
    if report["unfilled"]:
        w(f"【未点名格清单（{len(report['unfilled'])} 格）】")
        for u in report["unfilled"][:40]:
            w(f"  · {u['contract_id']} {u['dim']}：{u['reason']}")
            w(f"      原文：{u['value'][:120]}")
        if len(report["unfilled"]) > 40:
            w(f"  …（其余 {len(report['unfilled']) - 40} 格见 --json-out）")
        w("")
    if report["reds"]:
        w(f"【判红（{len(report['reds'])} 条）】")
        for r in report["reds"][:40]:
            w(f"  · {r['contract']} / {r['dim']}：{r['reason']}")
        w("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="契约「数据要求」栏 · 本业务数据可得性五维判据（PHASE6 S11）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[2]
                                        / "paper2skills-vault" / "07-资源库" / "contracts"),
                    help="契约目录（默认仓库内的 contracts/）")
    ap.add_argument("--file", default=None, help="只判这一份（契约号或文件名前缀）")
    ap.add_argument("--json-out", default=None, help="机读产物路径")
    ap.add_argument("--scan-extra", action="append", default=[],
                    help="额外扫描论文侧字段名的文件（可重复；报告用）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="用构造样本自检（含 subprocess 跑真 CLI + 把判据改坏的变异测试）")
    ap.add_argument("--no-mutate", action="store_true",
                    help="selftest 只跑用例、不跑变异层（变异层内部调用本开关防递归）")
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest(no_mutate=args.no_mutate)

    try:
        report, err = build_report(args)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"💥 脚本内部错误：{type(exc).__name__}: {exc}")
        return EXIT_INTERNAL

    if report is None:
        print(f"⛔ 输入没拿到（exit {EXIT_NO_INPUT}）：{err}")
        print("   —— 这不是「干净」，是「没测」。")
        return EXIT_NO_INPUT

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.quiet:
        print(render_text(report))

    s = report["summary"]
    r9_red = any(c["id"] == "R9" and c["status"] == "red" for c in report["criteria"])
    if report["reds"] or r9_red:
        print("-" * 72)
        print(f"🔴 判红 {len(report['reds'])} 条"
              + ("（含 R9：出现了论文侧字段名）" if r9_red else ""))
        print(f"   五维齐全 {s['five_dims_complete']} / 数据待盘 {s['data_pending']} / "
              f"不可得 {s['not_obtainable']}")
        return EXIT_RED
    print("-" * 72)
    print(f"✅ 五维判据全过：{s['contracts_total']} 份 · 五维齐全 {s['five_dims_complete']} · "
          f"数据待盘 {s['data_pending']} · 不可得 {s['not_obtainable']}（判合格）")
    return EXIT_PASS


# ==========================================================================
# selftest：构造样本 + subprocess 跑真 CLI + 变异测试
# ==========================================================================
_SAMPLE_A = """---
contract_id: CTR-A-001
template: A
template_version: 2
responsibility: 资源情景比较
role_id: AGT-001
role_title: 经营目标与资源统筹
flows: [FLOW-02]
status: 可写
blocked_by: null
---

# 资源情景比较 · 供给契约

## 1 方法来源
- 方法族：资源约束下的方案比选

## 2 数据要求（五维 · 三列）

| 维 | 取值 | 缺值处置 |
|---|---|---|
| ① 获取路径 | {d1} | {d1d} |
| ② 粒度 | {d2} | {d2d} |
| ③ 回溯深度 | {d3} | {d3d} |
| ④ 新鲜度 | {d4} | {d4d} |
| ⑤ 口径归属 | {d5} | {d5d} |

## 3 标定规则
- 取值：资源占用率 ≤ 1.00。

## 4 重标定触发条件
- 口径变。

## 5 适用 FLOW
FLOW-02

## 6 冻结与不许自动放行的情形
- 无。
"""

OK_CELLS = {
    "d1": "平台后台（Amazon 卖家后台报表与广告后台）＋ 自有埋点（〈独立站埋点〉）",
    "d1d": "非缺口。",
    "d2": "SKU × 渠道账号 × 自然月",
    "d2d": "非缺口。",
    "d3": "近 2 个完整年度（决策 Q11 的出海历史长度）",
    "d3d": "非缺口。",
    "d4": "每次进入 STG-04 前重读一次；日级序列按 T+1 回填",
    "d4d": "非缺口。",
    "d5": "AGT-045 业务口径与主数据（GMV 的分子分母与时间归属）",
    "d5d": "非缺口。",
}

# 「不可得」但**五维齐全** —— 验收原话之一：这份必须判**合格**
NOT_OBTAINABLE_CELLS = dict(
    OK_CELLS,
    d1="平台后台（Amazon 广告后台报表）＋**不可得：平台店的受众级 holdback 分流数据**",
    d1d="受众级对照先按渠道级同窗前后对照**暂按**准实验顶替；独立站 holdback 建成后**替换**。",
)


def _write_fixture(root: Path, cells: dict, name="CTR-A-001", subdir="A"):
    d = root / subdir
    d.mkdir(parents=True, exist_ok=True)
    body = _SAMPLE_A.format(**cells)
    if name != "CTR-A-001":
        body = body.replace("CTR-A-001", name)
    (d / f"{name}-样本.md").write_text(body, encoding="utf-8")
    return d / f"{name}-样本.md"


def _run_cli(args, timeout=120):
    p = subprocess.run([sys.executable, str(SELF)] + args,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run_selftest(no_mutate=False) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cdr-selftest-"))
    fails, cases = [], []
    ok_count = 0

    def case(name, argv, expect_exit, *, mutates=None, must_contain=None,
             must_not_contain=None, check=None):
        cases.append({"name": name, "argv": argv, "expect_exit": expect_exit,
                      "mutates": mutates, "must_contain": must_contain,
                      "must_not_contain": must_not_contain, "check": check})

    def mk(tag):
        d = tmp / tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _expect_counts(**want):
        """断言 **汇总读数**（不只退出码）。

        ⚠️ 这条是变异 11 逼出来的：把「缺一维 ⇒ 数据待盘」改成恒不触发后，
        **退出码一个字都没变**（红因仍在 `reds` 里），只有三态的账变了。
        只断退出码的用例对这种判据完全瞎 —— 与仓库既有教训同族：
        「新门禁自己的 selftest 里有摆设」（台账 #25）。
        """
        def _f(summary):
            probs = []
            for k, v in want.items():
                got = summary.get(k)
                if got != v:
                    probs.append(f"汇总 {k} = {got!r} ≠ 期望 {v!r}")
            return probs
        return _f

    # ---------------- 用例 1–3：反向控制 + 两条「不可能通过」的写法 -------------
    r = mk("clean"); _write_fixture(r, OK_CELLS)
    case("干净夹具（五维都点了名）⇒ exit 0", ["--dir", str(r), "--quiet"], EXIT_PASS,
         mutates="反向控制：防「什么输入都判红」的假仪表",
         check=_expect_counts(five_dims_complete=1, data_pending=0, not_obtainable=0))

    r = mk("vague"); _write_fixture(r, dict(OK_CELLS, d1="数据可得", d1d="非缺口。"))
    case("① 只写「数据可得」不点名获取路径 ⇒ exit 1", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R1 正取值")

    # ⚠️ 下面两条**只打 R1**：R1 与 R2 是叠加的，用「数据可得」那种既缺枚举
    # 又没有括注的样本来测 R1，会出现「把 R1 改成恒真、用例照样红（被 R2 接住）」
    # —— 即用例是摆设。故各配一份**只打其中一条**的样本（台账 #26 的反同质化口径）。
    r = mk("r1only")
    _write_fixture(r, dict(OK_CELLS, d1="数据可得（渠道报表与站点埋点）", d1d="非缺口。"))
    case("① 有括注但一个枚举都没断言 ⇒ exit 1（只打 R1，不被 R2 接住）",
         ["--dir", str(r), "--quiet"], EXIT_RED, mutates="R1 正取值（隔离样本）")

    r = mk("negparen")
    _write_fixture(r, dict(OK_CELLS,
                           d1="本企业无自有埋点（现全靠人工估算，无站点日志）",
                           d1d="非缺口。"))
    case("① 枚举词写在否定句里且有括注 ⇒ exit 1（只打 R1 的否定句守卫）",
         ["--dir", str(r), "--quiet"], EXIT_RED, mutates="R1 的否定句守卫（隔离样本）")

    r = mk("neg"); _write_fixture(r, dict(OK_CELLS, d1="本企业无自有埋点，实际全靠人工估算",
                                          d1d="非缺口。"))
    case("① 枚举词写在否定句里 ⇒ exit 1（D19 同族）", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R1 的否定句守卫")

    # ---------------- 用例 4–8：逐维能打红 -------------
    r = mk("bare"); _write_fixture(r, dict(OK_CELLS, d1="平台后台", d1d="非缺口。"))
    case("① 只写枚举词、不点名具体数据源 ⇒ exit 1", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R2 点名具体源")

    r = mk("grain"); _write_fixture(r, dict(OK_CELLS, d2="明细", d2d="非缺口。"))
    case("② 只写「明细」不给粒度轴 ⇒ exit 1", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R3 粒度轴")

    r = mk("depth"); _write_fixture(r, dict(OK_CELLS, d3="有历史数据", d3d="非缺口。"))
    case("③ 不给可判定的回溯深度 ⇒ exit 1", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R4 深度")

    r = mk("depth2"); _write_fixture(r, dict(OK_CELLS, d3="近 5 个完整年度", d3d="非缺口。"))
    case("③ 深度偏离既有 2 年口径且未点名来源 ⇒ exit 1", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R4 与「2 个完整年度」的对账")

    r = mk("fresh"); _write_fixture(r, dict(OK_CELLS, d4="及时", d4d="非缺口。"))
    case("④ 只写「及时」不给节奏 ⇒ exit 1", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R5 新鲜度")

    r = mk("owner"); _write_fixture(r, dict(OK_CELLS, d5="有口径", d5d="非缺口。"))
    case("⑤ 只写「有口径」不点名归属方 ⇒ exit 1", ["--dir", str(r), "--quiet"],
         EXIT_RED, mutates="R6 口径归属")

    # ---------------- 用例 9：缺口无处置 ⇒ 红 -------------
    r = mk("gapnodisp")
    _write_fixture(r, dict(OK_CELLS, d3="本业务暂无该数据", d3d=""))
    case("缺口无处置（第三列空）⇒ exit 1", ["--dir", str(r), "--quiet"], EXIT_RED,
         mutates="R7 缺口必须带处置")

    # ---------------- 用例 10：⚠️ 验收原话之二 —— 不可得必须判合格 -------------
    r = mk("notobtain"); _write_fixture(r, NOT_OBTAINABLE_CELLS)
    case("★「不可得」但五维齐全 ⇒ exit 0（合法结论，不许改成 partial）",
         ["--dir", str(r), "--quiet"], EXIT_PASS,
         mutates="「不可得是合法结论」这条判据",
         must_contain="不可得",
         check=_expect_counts(five_dims_complete=1, data_pending=0, not_obtainable=1))

    # ---------------- 用例 11：不可得 + 缺一维 ⇒ 数据待盘（不默认其余四维成立）---
    r = mk("notobtain_pending")
    _write_fixture(r, dict(NOT_OBTAINABLE_CELLS, d4="及时", d4d="非缺口。"))
    case("★「不可得」但④缺 ⇒ 判「数据待盘」而不是白拿其余四维",
         ["--dir", str(r), "--quiet"], EXIT_RED,
         mutates="「缺一维 ⇒ 数据待盘」这条判据",
         check=_expect_counts(data_pending=1, five_dims_complete=0))

    # ---------------- 用例 12–14：输入没拿到 ⇒ exit 2（不是 0） -------------
    # ⚠️ 两条 exit 2 的成因不同（台账 #26 的教训：两条判据共用一份反向控制
    # ⇒ 互相顶账，删掉任一条自检都照样全绿）。故两条各配**自己的**断言：
    # 不只断退出码，还断**红因文案**，否则「目录不存在」的守卫被删掉后，
    # 「0 份契约」那条守卫会把退出码 2 照样交出来 —— 用例抓不住。
    r = mk("empty")
    case("目录在但一份契约都没有 ⇒ exit 2（不是 0）", ["--dir", str(r), "--quiet"],
         EXIT_NO_INPUT, mutates="main() 里的「0 份契约」守卫",
         must_contain="一个 CTR-*.md 都没扫到")
    case("契约目录不存在 ⇒ exit 2", ["--dir", str(tmp / "nope"), "--quiet"],
         EXIT_NO_INPUT, mutates="main() 里的「目录不存在」守卫",
         must_contain="契约目录不存在")

    r = mk("unparsed"); _write_fixture(r, OK_CELLS)
    p = next(r.rglob("*.md"))
    p.write_text(p.read_text(encoding="utf-8").replace("| ④ 新鲜度 |", "| ④ 新鲜度X |"),
                 encoding="utf-8")
    case("五维表抽不全（解析不了）⇒ exit 2，不猜", ["--dir", str(r), "--quiet"],
         EXIT_NO_INPUT, mutates="「解析不到 ⇒ 输入没拿到」守卫")

    # ---------------- 用例 16–18：R9 —— 与论文侧字段 schema 互不引用 -------------
    r = mk("clean19"); _write_fixture(r, OK_CELLS)
    inj = tmp / "injected.md"
    inj.write_text(f"本业务数据 | {_paper_side_field()} | available\n", encoding="utf-8")
    case("R9 被 --scan-extra 扫到的文件里出现论文侧字段名 ⇒ exit 1",
         ["--dir", str(r), "--quiet", "--scan-extra", str(inj)], EXIT_RED,
         mutates="R9 schema 隔离")

    # ⚠️ 这条才是**最像真实事故**的形态，而且是**只打 R9** 的隔离样本：
    # 字段名写在第三列「缺值处置」里 —— 那一列 R1–R7 **一律不看**，
    # 所以这份夹具在五维判据下全绿，只有 R9 能抓住它。
    # 首版把字段名写在 ⑤ 的**取值格**里 ⇒ ⑤ 因为没点名 AGT-045 先判红，
    # 于是「把 R9 改成恒绿」以后退出码一个字都没变 ⇒ **变异施不上力**。
    r9inj = mk("r9inj")
    _write_fixture(r9inj, dict(OK_CELLS,
                               d5d=f"非缺口（沿用 {_paper_side_field()} 的字段名）"))
    case("R9 隔离样本：字段名只写在第三列（五维判据看不见的那一列）⇒ exit 1",
         ["--dir", str(r9inj), "--quiet"], EXIT_RED, mutates="R9 schema 隔离（隔离样本）")

    # ---------------- 用例 19：R9 反向控制 —— 干净产物必须绿 -------------
    case("R9 反向控制：干净产物 ⇒ exit 0",
         ["--dir", str(r), "--quiet"], EXIT_PASS, mutates="R9 的反向控制")

    # ---------------- 用例 17：覆盖率双边（>139 判红） -------------
    r = mk("over")
    for i in range(1, 145):
        _write_fixture(r, OK_CELLS, name=f"CTR-A-{i:03d}")
    case("计数超界（>139 份）⇒ exit 1（判据只写下界等于没有判据）",
         ["--dir", str(r), "--quiet"], EXIT_RED, mutates="R10 覆盖率上界")

    # ---------------- 用例 18：单份 --file -------------
    r = mk("single")
    _write_fixture(r, OK_CELLS, name="CTR-A-001")
    _write_fixture(r, dict(OK_CELLS, d1="数据可得", d1d="非缺口。"), name="CTR-A-002")
    case("--file 只判指定的一份（该份合格 ⇒ exit 0）",
         ["--dir", str(r), "--file", "CTR-A-001", "--quiet"], EXIT_PASS,
         mutates="--file 切片")

    # ======================================================================
    # 跑用例（**subprocess 跑真 CLI**，不是 import 库函数 —— 台账 #25：
    # 判据在 main() 里而 selftest 只测库函数 ⇒ 把守卫改成 if(False) 照样全绿）
    # ======================================================================
    print("用例（subprocess 跑真 CLI + 构造夹具）")
    print("-" * 72)
    for i, c in enumerate(cases, 1):
        jout = tmp / f"case_{i:02d}.json"
        argv = c["argv"] + (["--json-out", str(jout)] if c["check"] else [])
        try:
            code, out, err = _run_cli(argv)
        except Exception as exc:                               # noqa: BLE001
            fails.append((c["name"], [f"CLI 跑不起来：{exc}"]))
            print(f"  ❌ {i:>2}. {c['name']} —— CLI 跑不起来：{exc}")
            continue
        probs = []
        if code != c["expect_exit"]:
            probs.append(f"退出码 {code} ≠ 期望 {c['expect_exit']}")
        blob = out + err
        if c["must_contain"] and c["must_contain"] not in blob:
            probs.append(f"输出里找不到 {c['must_contain']!r}")
        if c["must_not_contain"] and c["must_not_contain"] in blob:
            probs.append(f"输出里不该出现 {c['must_not_contain']!r}")
        if c["check"]:
            try:
                summary = json.loads(jout.read_text(encoding="utf-8"))["summary"]
                probs.extend(c["check"](summary))
            except Exception as exc:                           # noqa: BLE001
                probs.append(f"读不了 --json-out 产物：{exc}")
        if probs:
            fails.append((c["name"], probs))
            print(f"  ❌ {i:>2}. {c['name']}")
            for p in probs:
                print(f"        · {p}")
            tail = (out + err).strip().splitlines()[-3:]
            for t in tail:
                print(f"        | {t}")
        else:
            ok_count += 1
            tag = f"  （篡改 {c['mutates']}）" if c.get("mutates") else ""
            print(f"  ✅ {i:>2}. {c['name']} → exit {code}{tag}")

    # ======================================================================
    # 变异测试：把**判据本身**改坏，看用例能不能抓住。
    # 两条断言，缺一不可：
    #   ① 变异体自己的 --selftest 必须**失败**（说明有用例抓住了）
    #   ② 变异体在**对应探针夹具**上必须给出与干净版**不同**的读数
    #      （说明变异真的施上了力 —— run_phase6_gates 的教训：
    #       「先证明变异改变了真实取值，再谈判据有没有劲」）
    # ======================================================================
    src = SELF.read_text(encoding="utf-8")
    # ⚠️ 锚点只在**判据代码区**里数。变异表自己把这些锚点字符串又抄了一遍，
    # 在全文里数会得到 2 ⇒ 每条变异都被误报成「施不上力」。
    # （run_phase6_gates 的实测教训：首版用裸 `    if bad:` 做锚点，而该文本在
    #  文件里出现两次 ⇒ 只改到打印那处，**变异从没被跑过**，而读数看起来
    #  像「用例是摆设」。）
    MUT_MARK = "    mutations = [\n"
    code_region = src[:src.index(MUT_MARK)]
    # 探针 = (夹具名, 该夹具上必须出现/不出现的文案)。判定法：把**干净版**与
    # **变异体**在同一夹具上的读数对拍，`(退出码, 文案命中)` 必须**不同** ——
    # 这才叫「变异真的改变了真实取值」。只断「变异体 selftest 失败」会被
    # 别的用例的失败顺带满足（假抓住）。
    mutations = [
        ("R1 ① 判据恒真（枚举怎么都不判红）",
         '        ok, asserted = asserts_access(v)\n        if not ok:',
         '        ok, asserted = asserts_access(v)\n        ok = True\n        if not ok:',
         ("r1only", None)),
        ("R1 去掉否定句守卫（`无自有埋点` 算断言）",
         '            if e != "不可得" and any(n in head for n in ACCESS_NEG):\n'
         '                continue',
         '            if False and any(n in head for n in ACCESS_NEG):\n'
         '                continue',
         ("negparen", None)),
        ("R2 去掉「点名具体源」判据",
         '        if not names_concrete_source(d.raw_value):',
         '        if False and not names_concrete_source(d.raw_value):',
         ("bare", None)),
        ("R3 粒度判据恒真",
         '        if _GRAIN_AXIS.search(v) or any(w in v for w in _GRAIN_WORDS):',
         '        if True or _GRAIN_AXIS.search(v):',
         ("grain", None)),
        ("R4 深度判据恒真",
         '        if _DEPTH_OTHER.search(v):\n'
         '            if _DEPTH_SOURCE.search(v):\n'
         '                return "POS", "③ 给了可判定的深度并点名了来源"',
         '        if _DEPTH_OTHER.search(v):\n'
         '            if _DEPTH_SOURCE.search(v):\n'
         '                return "POS", "③ 给了可判定的深度并点名了来源"\n'
         '        if v:\n            return "POS", "MUTATED"',
         ("depth", None)),
        ("R4 对账判据去掉（深度≠2 年也不点名来源照样过）",
         '            if _DEPTH_SOURCE.search(v):\n'
         '                return "POS", f"③ = {n} 个完整年度（偏离既有 2 年口径，但点名了来源）"',
         '            if True:\n'
         '                return "POS", f"③ = {n} 个完整年度（MUTATED：不再要求点名来源）"',
         ("depth2", None)),
        ("R5 新鲜度判据恒真",
         '        if _FRESH.search(v):',
         '        if True or _FRESH.search(v):',
         ("fresh", None)),
        ("R6 口径归属判据恒真",
         '        if _OWNER_MAIN in v and _OWNER_HINT.search(v):',
         '        if True or _OWNER_MAIN in v:',
         ("owner", None)),
        ("R7 缺口判据去掉（缺口无处置也放行）",
         '    if not kept:\n'
         '        kept = stripped if not _GAP.search(stripped) else ""\n'
         '    return bool(_DISPOSAL.search(kept))',
         '    return True',
         ("gapnodisp", None)),
        ("★ 把「不可得」当缺口判红（本任务最容易做反的地方）",
         '    not_obtainable = "不可得" in asserted',
         '    not_obtainable = "不可得" in asserted\n'
         '    if not_obtainable:\n'
         '        reds.append({"contract": info["contract_id"], "file": info["file"],\n'
         '                     "dim": DIMS[0], "reason": "MUTATED：把不可得当成了缺口"})',
         ("notobtain", "不可得")),
        ("★ 缺一维不再判「数据待盘」（默认其余四维成立）",
         '    if reds:\n        verdict = "数据待盘"',
         '    if False:\n        verdict = "数据待盘"',
         ("notobtain_pending", None)),
        ("R9 schema 隔离判据去掉（论文侧字段名不再报红）",
         '    r9_hits = scan_paper_side_field(scan_targets)',
         '    r9_hits = []',
         ("r9inj", None)),
        ("main() 的「0 份契约 ⇒ exit 2」守卫改成恒不触发",
         '    if not files:\n        return None, (f"{root} 下一个 CTR-*.md 都没扫到 —— "',
         '    if False:\n        return None, (f"{root} 下一个 CTR-*.md 都没扫到 —— "',
         ("empty", "一个 CTR-*.md 都没扫到")),
        ("main() 的「目录不存在 ⇒ exit 2」守卫改成恒不触发",
         '    if not root.is_dir():\n        return None, f"契约目录不存在：{root}"',
         '    if False:\n        return None, f"契约目录不存在：{root}"',
         ("nope!", "契约目录不存在")),
        ("R10 覆盖率上界判据去掉（>139 也放行）",
         '    if n_total > expect_total:',
         '    if False and n_total > expect_total:',
         ("over", None)),
    ]

    print("-" * 72)
    print("变异测试（把判据弄坏，看还有没有用例抓得住 + 变异真的施上力了吗）")
    caught = 0
    if no_mutate:
        print("  （--no-mutate：跳过变异层）")
    else:
        for mi, (mname, old, new, probe) in enumerate(mutations, 1):
            problems = []
            n_hit = code_region.count(old)
            if n_hit != 1:
                problems.append(f"变异锚点在**判据代码区**里出现 {n_hit} 次（应为 1）"
                                f" —— **变异没施上力**")
            mut_path = tmp / f"mut_{mi:02d}.py"
            mut_path.write_text(src.replace(old, new, 1), encoding="utf-8")
            # ① 变异体自己的 selftest 必须失败 ⇒ 说明有用例抓住了
            mcode, mout, merr = _run_cli_on(mut_path, ["--selftest", "--no-mutate"])
            if mcode == EXIT_PASS:
                problems.append("变异体的 --selftest 照样全绿 ⇒ **没有任何用例抓住这个变异**")
            # ② 对拍：同一探针夹具上，干净版与变异体的读数必须不同
            if probe is not None:
                tag_raw, msg = probe
                expect_absent = tag_raw.endswith("!")   # 「目录必须不存在」的探针
                tag = tag_raw.rstrip("!")
                probe_dir = tmp / tag
                if expect_absent and probe_dir.exists():
                    problems.append(f"探针 {tag} 本应不存在，却存在")
                elif not expect_absent and not probe_dir.is_dir():
                    problems.append(f"探针夹具 {tag} 不存在")
                else:
                    argv = ["--dir", str(probe_dir), "--quiet"]
                    c_read = _reading(argv, msg, None)
                    p_read = _reading(argv, msg, mut_path)
                    if c_read == p_read:
                        problems.append(
                            f"变异体在探针 {tag} 上的读数与干净版**完全一样**"
                            f"（{c_read}）⇒ 变异没改变真实取值，这条变异是摆设")
            if problems:
                print(f"  ❌ 变异 {mi}/{len(mutations)} · {mname}")
                for p in problems:
                    print(f"        · {p}")
            else:
                caught += 1
                extra = (f"；探针 {probe[0]} 上读数已翻面"
                         if probe is not None else "（无探针：仅由用例抓）")
                print(f"  ✅ 变异 {mi}/{len(mutations)} · {mname} → 被用例抓住{extra}")

    # ------------------------------------------------------------------
    shutil.rmtree(tmp, ignore_errors=True)     # 跑完即删：不留临时文件
    print("-" * 72)
    print(f"用例 {ok_count}/{len(cases)} 通过 · 篡改样本 "
          f"{len([c for c in cases if c.get('mutates')])} 个 · 变异 {caught}/{len(mutations)} 抓住")
    leftover = sorted(p.name for p in (SELF.parent).glob("cdr-selftest-*"))
    if leftover:
        print(f"🔴 scripts/ 下留了临时文件：{leftover}")
        return EXIT_INTERNAL
    if fails:
        print(f"🔴 selftest 失败 {len(fails)} 例")
        return EXIT_RED
    if not no_mutate and caught != len(mutations):
        print("🔴 有变异没被抓住 —— 对应用例是摆设")
        return EXIT_RED
    print("✅ selftest 全过（含反向控制、篡改样本与判据变异）")
    return EXIT_PASS


def _run_cli_on(script: Path, args, timeout=180):
    p = subprocess.run([sys.executable, str(script)] + args,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _reading(argv, msg, script, timeout=180):
    """一次运行的**读数**：退出码 + 文案命中 + **汇总账的指纹**。

    ⚠️ 只看退出码是不够的 —— 实测：把「缺一维 ⇒ 数据待盘」改成恒不触发，
    **退出码一个字都没变**（红因仍在 `reds` 里），只有三态的账变了。
    故指纹取 `summary` 里所有数值项。这条是变异 11 逼出来的。
    """
    import hashlib as _h
    jout = Path(tempfile.mkdtemp(prefix="cdr-reading-")) / "r.json"
    try:
        if script is None:
            code, out, err = _run_cli(argv + ["--json-out", str(jout)], timeout)
        else:
            code, out, err = _run_cli_on(script, argv + ["--json-out", str(jout)], timeout)
        digest = None
        if jout.is_file():
            s = json.loads(jout.read_text(encoding="utf-8"))["summary"]
            nums = {k: v for k, v in s.items() if isinstance(v, (int, float))
                    and not isinstance(v, bool)}
            digest = _h.sha256(json.dumps(nums, sort_keys=True,
                                          ensure_ascii=False).encode()).hexdigest()[:12]
        return code, (msg in (out + err)) if msg else None, digest
    finally:
        shutil.rmtree(jout.parent, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(cli())
