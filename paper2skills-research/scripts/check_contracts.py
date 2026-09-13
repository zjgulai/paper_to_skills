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

退出码：0 全过 / 1 有判据红 / 2 输入没拿到（**不是通过**）/ 3 门禁内部错误（**不是判红**）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DEFAULT = REPO_ROOT / "paper2skills-vault/07-资源库/capability-graph.json"
CONTRACTS_DIR = REPO_ROOT / "paper2skills-vault/07-资源库/contracts"
BUILDER = Path(__file__).resolve().parent / "build_contracts.py"

# 数据要求五维的标签与 ① 的合法枚举（Q12 / S11）
DIM_LABELS = ("获取路径", "粒度", "回溯深度", "新鲜度", "口径归属")

# 五维表行的定位（门禁缺陷 #17）：**标签必须锚定到表行行首**。详见 Contract.dim_cells 的注释。
def _dim_row(label: str):
    """五维表行：**行首 + 标签**，标签后允许括注/补语，但不得跨过第一个 `|`。

    ⚠️ 收紧到「首格必须恰好等于标签」会**新造一种假红**（2026-09-13 由 S1 撰写人实测报出）：
    写成「| ① 获取路径（平台后台与埋点） | … |」时首格不是纯标签 ⇒ 静默读成空值 ⇒ J5 报缺维。
    故允许标签后接任意非 `|` 字符 —— 关键是**锚在行首**（真问题从来不是括注，是段内提前出现的标签词）。

    ⚠️ D13（2026-09-13 由 S1 撰写人实测撞出，**假红**）：首格写成 `| **① 获取路径** |` —— 内容
    一字未改、只是加粗 —— 原正则读成空值 ⇒ J5 报「数据要求缺维：获取路径」。全角标点同族。
    修法：容忍 `**` 与标签前后的全角标点；**「锚在行首」这一条不许动**，否则 #17 的原假绿会复现。
    """
    return re.compile(
        rf"(?m)^\s*\|\s*(?:\*\*)?\s*[①②③④⑤]?\s*[：:、.]?\s*(?:\*\*)?\s*"
        rf"{re.escape(label)}[^|\n]*\|")

ACCESS_ENUM = ("自有埋点", "平台后台", "第三方 API", "需授权", "不可得")
# D19 的否定语：出现在枚举词**之前**（同一子句内）即视为该枚举词只是被提到，不是被断言
_ACCESS_NEG = ("无", "没有", "不含", "不具备", "不靠", "不是", "非", "尚未", "未接入", "未建")


def _asserts_access(access: str) -> bool:
    """① 是否**断言**了五选一中的某一项（D19）。

    ⚠️ D19（2026-09-13 S1 撰写人实测撞出，**假绿**）：原实现是
    `any(e in access for e in ACCESS_ENUM)` —— **子串匹配**，枚举词出现在**否定句**里也算数。
    实测：`| ① 获取路径 | 本企业无自有埋点，实际全靠人工估算 | 非缺口 |` ⇒ exit=0。
    这与 #17 同族、方向相反：#17 是「标签被子串劫持」，本条是「**枚举词被否定句劫持**」。
    修法：逐子句判定 —— 命中枚举词的那个子句里，枚举词之前不得出现否定词。
    ⚠️ 反向控制：`自有埋点（独立站埋点事件流）` 这类正常写法必须仍然判绿（selftest 锁定）。
    """
    for seg in re.split(r"[＋+；;。，,、/]", access):
        for e in ACCESS_ENUM:
            i = seg.find(e)
            if i < 0:
                continue
            if not any(n in seg[:i] for n in _ACCESS_NEG):
                return True
    return False

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
GAP_TOKENS = ("暂无", "未定名", "不可得", "拿不到", "取不到", "尚未", "待定", "未知",
              # --- D6（2026-09-13 S1 撰写人实测撞出，**假绿**）---
              # 原词表漏词 ⇒ 缺口句「不在词表里就整条不检查」。实测：B §5 ③ 写成
              # 「本业务没有该数据」+ 第三列留空 ⇒ exit=0（J13 整个跳过这一维）。
              # 补词，但 ⚠️ **刻意不补单字「缺」** —— 它会命中「缺口账」「缺口编号」
              # 这类正常表述，把非缺口行判成缺口行（那是假红，比漏判更贵）。
              "缺失", "缺少", "未建", "无从取得", "没有", "不具备", "未采集", "不采集")
DISPOSAL_TOKENS = ("替换", "失效", "改用", "顶替", "代为", "先行", "暂按", "暂用",
                   "生效条件", "到位后", "上线后", "接入后", "补充后",
                   # --- D26 桶③（主控回测，1 条但最讽刺）---
                   # A-049 §3 原句：`在独立站 holdback 建成前，**不可得** 的是受众级随机对照，
                   # **替代取值**为同店同渠道的准实验窗。` —— 处置**就在同一句里**，
                   # 但本表收的是「替换」，**不收「替代」**。而 B 模板强制要求的写法字面就是
                   # `→ 替代路径：` —— **判据不认它自己模板规定的词**。
                   "替代",
                   # B-022：`本责任暂无候选方法卡，这就是缺口账里的一条扩充工单，不是
                   # 「这条责任没有方法」。` —— 正当的缺口陈述，处置就是已登记进缺口账/工单。
                   "已登记", "登记进", "工单", "扩充工单", "挂账",
                   # B-034：`源字段：…；缺失时退到〈平台后台〉折扣码与联盟链接的兑换数。`
                   # —— 处置在场（「退到」就是回退写法），只是词表不认「退」这一族。
                   "退到", "回退", "退化为")
ALT_MARK = "替代路径"
# D1：替代路径必须是**独立的箭头行**（不是正文里提一句「替代路径」就算数）
ALT_LINE_RE = re.compile(r"^\s*(?:→|->|=>)\s*" + re.escape(ALT_MARK))


def _item_blocks(body: str) -> list:
    """把段切成「条目块」：一条条目行 + 它的后续续行（含缩进的箭头行）。

    ⚠️ 条目行正则与取值锚定**共用 `ITEM_LINE_RE`/`BULLET_SRC`**（D7）：
    原实现写死 `^\\s*(?:[-*]|\\d+[.、])\\s+\\S`，不认 `1)` / `（1）` / `①` ——
    实测 B §2 六条写成 `1) 不可达①…` 且六条箭头一条不少，却报
    「§2 没有列出任何不可达条目」（**假红**）。
    """
    blocks, cur = [], None
    for ln in body.splitlines():
        if ITEM_LINE_RE.match(ln):
            if cur is not None:
                blocks.append(cur)
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
    if cur is not None:
        blocks.append(cur)
    return blocks
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

# --- 结构前缀：项目符 / 编号 / 圈号 / 加粗 / T1·S3 前缀 ---
# ⚠️ D7 与 D2 **共用这一套定义**（两处各写一份必然漂移 —— 撰写人实测：§2 条目正则不认
#    `1)` / `（1）` / `①` 时，同一批前缀在取值锚定里却认得，于是「修了一处漏一处」）。
BULLET_SRC = r"(?:[-*•]|\d+[.、)]|（\d+）|\(\d+\)|[①-⑳])"
# 条目行：项目符 + 非空白内容（后面**不再强制**要有空格 —— `（1）不可达` 是合法中文编号写法）
ITEM_LINE_RE = re.compile(rf"^\s*{BULLET_SRC}\s*\S")
# 结构前缀**完整匹配**：标签以前只允许 空白 / 项目符 / `**` / 编号 / `[TS]\d` / 空白
_STRUCT_PREFIX_FULL_RE = re.compile(
    rf"\s*(?:{BULLET_SRC})?\s*(?:\*\*)?\s*(?:[TS]\d)?\s*(?:\*\*)?\s*$")

# --- 交叉引用 / 转指成分：判「这一行自己有没有数字」之前必须先剥掉 ---
# D4（条款号）、D11（格号）、D18/D20（转指短语与版本号）是**同一个根因**：
#   数字的**归属**没有被约束 —— 只要行里有任意一个数字就算给了取值，
#   于是「见 §4 第 2 条」「按 FLOW-08/STG-06 判定」「（第 2 版口径）」都能冒充本业务取值。
XREF_RES = (
    re.compile(r"FLOW-0[1-8]\s*/\s*STG-0[1-8]"),   # 格号（D11）
    re.compile(r"§\s*\d+"),                         # 条款号（D4）
    re.compile(r"第\s*\d+\s*[条款项节版]"),          # 第 N 条 / 第 N 节 / 第 N 版（D4+D20）
)
# 转指短语（D18）。长的写在前面，避免 `见上` 抢先匹配掉 `见上文`。
POINTER_RE = re.compile(r"见本契约|见上文|见上表|见本节|见本表|见下|详见|参见|同上|见\s*§|见(?=\s*§)")


def _strip_pointers(s: str) -> str:
    """把**转指成分**连同它指向的东西一起剥掉。

    判据（D18 实测）：`- 阈值：见上文的数据要求（近 2 个完整年度）` 里的 `2` 属于 §2 的
    数据要求，不是 §3 的本业务取值。若只剥掉「见上文」四个字，那个 `2` 会留下来继续冒充取值。
    故剥到**本层结束**：指针在括号里就剥到配对的右括号，否则剥到下一个句读（，；。）或行尾。

    反向控制（务必保住）：`- 取值：容差档 5%（口径见 §2）` —— 指针在括注里，
    只剥括注，`5%` 必须留下（这是最容易误杀的一条，selftest 单独测）。
    """
    out, pos = s, 0
    while True:
        m = POINTER_RE.search(out, pos)
        if not m:
            return out
        depth = 0
        for ch in out[:m.start()]:
            if ch in "（(":
                depth += 1
            elif ch in "）)":
                depth = max(0, depth - 1)
        if depth > 0:                      # 指针在括号里 ⇒ 剥到配对的右括号
            d, i = depth, m.end()
            while i < len(out):
                if out[i] in "（(":
                    d += 1
                elif out[i] in "）)":
                    d -= 1
                    if d == 0:
                        i += 1
                        break
                i += 1
            end = i
        else:                              # 否则剥到下一个句读（或行尾）
            m2 = re.search(r"[，,；;。]", out[m.end():])
            end = m.end() + m2.start() + 1 if m2 else len(out)
        out = out[:m.start()] + out[end:]
        pos = m.start()


def _has_own_digits(s: str) -> bool:
    """剥掉交叉引用与转指成分之后，这一行**自己**还剩不剩数字。"""
    for r in XREF_RES:
        s = r.sub("", s)
    return bool(_BARE_DIGIT_RE.search(_strip_pointers(s)))


# 标签后到冒号之间只允许 `**` 与空白（D2 的判定条件，由父代理写死）
_LABEL_COLON_RE = re.compile(r"[\s*]*[：:]")


def _value_label_span(ln: str, m):
    """标签出现位置 → 冒号结束位置；不是「取值标签」则返回 None。

    ⚠️ D2（2026-09-13 S1 撰写人实测撞出，**假绿**）：原实现是 `VALUE_LABEL_RE.search(ln)`
    —— **子串**匹配，标签出现在行内任何位置都算。实测最小复现：A §3 只写「算式（无数字）」
    ＋ 一行 `- **不得移植的部分**：卡内异常阈值 0.75 与 0.2、迭代 3 轮、样本 300 条` ⇒ 全绿；
    删掉那一行即判红。**而 A 模板存在的理由正是拦「参数移植」** —— 禁令行反倒成了取值的凭证。

    修法（两段式，缺一不可）：
      ① 标签必须**锚定到结构前缀之后**（行首空白 / 项目符 / `**` / 编号 / `[TS]\\d`）；
      ② 标签后（中间只允许 `**` 与空白）必须**紧跟 `：` 或 `:`**。
    ② 是父代理后续写死的，它一次堵住两个现行绕过样本：
      `- 取值域：对齐〈采集台账〉（覆盖近 2 个完整年度）`（标签后是「域」不是冒号）
      `- 降级比例不设目标值，只用 T3 的成本线`（「目标值」在行中，锚定不成立）。
    ⚠️ **不许改用「关键词黑名单」**（禁「不得/禁止」之类）：实测它会误杀
    `- T2 取值：**容差档 = 0**（D 不得超过 A 一个单位）` 这类真取值行，全库误杀 9 份。
    锚定 + 冒号本身就能挡住禁令行（它的标签前是 `- **不得移植的部分**：卡内异常`，不是结构前缀）。

    D24 例外（同一处代码的另一端）：`- 取值来源与取值：…` 里第二个「取值」不在结构前缀之后，
    按 ①② 会被整行跳过 ⇒ 把一份**真给了取值**的契约判红。故补一条：
    标签到冒号之间的**标签短语**里若还含另一个取值标签词，则不算声明行，照常按冒号后的数字判。
    """
    if not _STRUCT_PREFIX_FULL_RE.match(ln[:m.start()]):
        return None                                     # ① 锚定
    seg = ln[m.end():]
    colon = _LABEL_COLON_RE.match(seg)
    if colon:
        return m.end() + colon.end()                    # ② 紧跟冒号
    head, sep, _ = seg.partition("：") if "：" in seg else seg.partition(":")
    if sep and VALUE_LABEL_RE.search(head):
        return m.end() + len(head) + 1                  # 合并标签行（D24）
    return None


def has_value(text: str) -> bool:
    """带取值标签的行里出现**本行的**数字 ⇒ 认为给出了取值（模板 §3 的 `- 取值：<...>` 形态）。

    ⚠️ **「取值来源」不是取值**（门禁缺陷 #18，2026-09-13 由三位 S1 撰写人各自实测撞出）。
    标签集里有「取值」，而 `- 取值来源：自有埋点 × AGT-045 指标契约（近 2 个完整年度）`
    这一行同时满足「含标签」与「含数字」⇒ **一份整段没有业务取值的契约照样过 J13**。
    实测探针：把一份合格契约 §3 的取值行全删、只留一行「取值来源」⇒ exit=0（假绿灯）。
    这与 #16/#18 同型：**判据的适用范围被默认成了全体，而它其实只覆盖一个子集**。
    ⚠️ D3 补强：原修法只排除了紧跟的「来源」二字，于是 `- 取值口径：按近 2 个完整年度算`
    仍能从右侧漏过（唯一的数字「2」来自 §2 的数据要求）。现由 `_value_label_span` 的
    「标签后必须紧跟冒号」一次性覆盖 来源/口径/说明/依据/出处 全部声明形态。

    ⚠️ D4/D11/D18/D20：数字一律取**冒号之后**的部分，且先剥交叉引用与转指成分 ——
    否则 `- 取值：见 §4 第 2 条`、`- 阈值：按 FLOW-08/STG-06 判定`、
    `- 取值：见上文的数据要求（近 2 个完整年度）` 都能拿别人的数字顶账。
    """
    for ln in text.splitlines():
        for m in VALUE_LABEL_RE.finditer(ln):
            colon_end = _value_label_span(ln, m)
            if colon_end is None:
                continue
            if _has_own_digits(ln[colon_end:]):
                return True
    return False


def has_disposal(text: str) -> bool:
    return any(tok in text for tok in DISPOSAL_TOKENS)


def _sentences(text: str) -> list:
    """把段落切成句子：先按行，再按句读（。；;!?！？）。"""
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        for part in re.split(r"(?<=[。；;!?！？])", ln):
            if part.strip():
                out.append(part.strip())
    return out


def _has_disposal_here(s: str) -> bool:
    return _has_own_digits(s) or has_disposal(s)


# --- D26 桶①：缺口词的**语境判据** ---
# ⚠️ D6 补词后全库实测撞出 24 份假红（2026-09-13 主控回测）。三组「词在、意思不在」：
#      · 名词短语：`缺失率`（指标名）/`缺失档`/`缺失值`/`缺失字段`/`待定主体`/`未定名渠道`
#      · 疑问构词：`有没有依据` / `还没有本业务的成…`（`没有` 只是「有没有」的一部分）
#      · 引号枚举值：``返回「未知」或「无终态」的对象``（`未知` 是枚举值，不是缺口）
#      · 条件从句：`许可缺失或已撤回的，一律不出营销触达`（条件，不是本段缺口陈述）
#    与 #17（五维标签子串劫持）、#19（「模型」二字误杀）、D10（审查句误判）**同一族**，
#    已是第 N 次复发：**判据的适用范围被默认成了全体，而它其实只覆盖一个子集。**
#    代价是照原样修下去，撰写人会被迫把「缺失率」改成「缺损率」—— 判据要测的东西
#    （缺口有没有处置）一个字都没测到，门禁缺陷又以「大家都学会绕路」的形式被吞掉（台账 #5）。
_GAP_NOUN_HEADS = ("率", "档", "值", "字段", "数", "项", "清单", "记录", "表", "度", "量",
                   "主体", "渠道", "原因", "部分", "环节", "判定", "情形", "情况", "状态",
                   "台账", "标记", "标签",
                   # 复合名词：`缺失工具数` 是产物里的指标名（B-059），不是缺口陈述
                   "工具数",
                   # 定语用法：`不得把缺失的库存读数当作零`（A-043）—— 形容词，不是谓语
                   "的")
_GAP_QUESTION_PREV = ("有", "是")
_GAP_COND_HEADS = ("若", "如果", "如", "当", "一旦", "倘若", "假若")


def _inside_short_quote(sent: str, i: int, tok: str) -> bool:
    """缺口词是否整体落在一段**短引号片段**里（枚举值 / 标签名，不是缺口陈述）。"""
    for op, cl in (("「", "」"), ("『", "』"), ("`", "`"), ("“", "”")):
        p = sent.rfind(op, 0, i)
        if p < 0:
            continue
        q = sent.find(cl, i + len(tok))
        if q < 0 or q - p > 12:            # 太长的引号片段视为正常陈述
            continue
        if not re.search(r"[。；;]", sent[p:q]):
            return True
    return False


def _is_gap_declaration(sent: str, tok: str, i: int) -> bool:
    """该出现位置是否**真的在陈述本段的缺口**（D26）。"""
    before, after = sent[:i], sent[i + len(tok):]
    if before[-1:] in _GAP_QUESTION_PREV and tok in ("没有", "缺少", "缺失"):
        return False                                   # 有没有 / 是没有
    if after.startswith(_GAP_NOUN_HEADS):
        return False                                   # 缺失率 / 待定主体 / 未定名渠道
    if _inside_short_quote(sent, i, tok):
        return False                                   # 「未知」这类枚举值
    clause = re.split(r"[，,；;。]", sent[:i])[-1]
    if clause.lstrip().startswith(_GAP_COND_HEADS):
        return False                                   # 若…尚未签发：…
    # 「X 尚未…时，<后果>」「X 缺失或已撤回的，<后果>」—— 词在条件从句里
    tail = sent[i + len(tok):]
    if re.search(r"^[^，。；]{0,24}(时|的)[，,]", tail) and re.search(
            r"[，,]\s*(一律|均|则|才|不|只能|须|应|当|即|该)", tail):
        return False
    return True


def has_gap_declaration(text: str) -> bool:
    """文本里是否存在**真的在陈述缺口**的位置（D26 桶① 的语境判据在此收口）。

    ⚠️ 逐个出现位置判定，不是「每词只看第一处」—— 否则 `缺失率` 在前、`本业务暂无此数据`
    在后时，后者会被前者的名词短语形态挡掉。
    """
    for sent in _sentences(text):
        for tok in GAP_TOKENS:
            start = 0
            while True:
                i = sent.find(tok, start)
                if i < 0:
                    break
                start = i + len(tok)
                if _is_gap_declaration(sent, tok, i):
                    return True
    return False


def gaps_without_disposal(text: str) -> list:
    """返回**没有任何处置**的缺口单元（空列表 = 该段缺口义务已履行）。

    判据的**颗粒度是「条目块」**（条目行 + 其后续缩进续行），与 B §2 的替代路径逐条配对同颗粒度
    —— 同一份文档里两种颗粒度才是真问题（D26 桶②）。

    ⚠️ D27（2026-09-13 主控反向控制实测撞出，**假绿**）：上一步「邻句/邻块豁免」只认**词**、
    不认**话题**，于是任意一句带「替换/改用/顶替」的**无关样板话**都能救活缺口句。实测最小复现
    （B-040 §1 末尾插一行真缺口）：
      `- 本业务的季节性先验来源暂无。`      ← 命中句，本块无处置
      `- 换卡时只替换方法实现，其余要求不变。` ← 邻块，只因含「替换」就发了豁免 ⇒ exit=0
    而且这不是个例：**60 / 66 份 B 契约的 §1 末句都是「换卡时只替换方法实现…」这一族样板**
    ⇒ §1 的最后一条缺口句在 60/66 份上被自动豁免，**J13 在 B 模板上的覆盖面被自己的样板话
    砍掉了尾部一格**。
    修法（主控指定方向）：豁免收紧到**同一条目块** —— 处置必须与缺口**同块**才算数。
    ⚠️ **不许改用「样板话黑名单」**：那是把这一次的样板硬编码，下一份换个写法立刻复发。
    """
    blocks = _item_blocks(text)
    units = ["\n".join(b) for b in blocks] if blocks else _sentences(text)
    return [u for u in units if has_gap_declaration(u) and not _has_disposal_here(u)]


def row_disposal(val: str, disp: str) -> bool:
    """五维一行是否给出了**可判定的处置**（缺口句不算它自己的处置）。

    ⚠️ D28（2026-09-13 实测，**假红**）：D22 的原修法是「缺口在任一列 ⇒ **另一列**必须给处置」，
    这是**按列**判的。撰写人常把「缺口 + 处置」**合并写进取值格**、第三列干脆不写（实测
    B-032/033/034/036/039/040 六份、8 格如此，且 ④ 行里直接写着 `缺口：…处置：…` 两个字面标签）
    —— 按列判会把**已经履行了义务**的行判红。
    修法：看**整行**；但先把「陈述缺口的那些句子」摘掉再找处置，否则 D22 的原复现
    （缺口写进处置列）会被自己的话救活。
    """
    if _has_disposal_here(disp):
        return True
    kept = [s for s in _sentences(val) if not has_gap_declaration(s)]
    return _has_disposal_here("\n".join(kept))


def gap_without_disposal(text: str) -> bool:
    """该段是否存在**没有任何处置**的缺口 ⇒ True（= 应当判红）。

    ⚠️ D21/D23（2026-09-13 S1 撰写人实测撞出，**假绿**，同一处代码）：
    原实现是**段级**短路 —— `has_value(text)` 为真即整段放行。实测最小复现：
    §3 ＝ `- 取值：容差档 5%。` ＋ `- 本业务暂无季节性先验来源。` ⇒ exit=0 ——
    **段内任意一处取值解除了全段的缺口义务**，与 J13「缺口陈述不是结论」的立意正好相反。
    对照：B 模板 §2 的替代路径是**按条**计数的，A 模板这一层原先没有同等颗粒度。

    ⚠️ D26 桶②（主控回测）：先改成「±1 句窗口」制造了 15 条假红 —— 撰写人的真实写法是
    「缺口句 → 定性说明句 → 处置句」，处置被中间那句顶出窗口（B-040 §1 逐句
    [15] 尚未签发 → [16] 不由本契约另造 → [17] 签发后替换）。**不敢简单放大窗口**：
    那会让邻句的**裸数字**救活缺口句，正是 D23 复现要防的。故改为按**条目块**判定。
    """
    return bool(gaps_without_disposal(text))
# --- J3 的「说成算法接入」判据（门禁缺陷 #19） ---
# ⚠️ 原实现是 `re.search(r"算法|模型", line)` —— **逐行字符串匹配**：同行只要出现「模型」二字，
#    该 R/D 格就判红。实测误杀三类正常写法：
#      ① 业务措辞「模型候选 + 待签」；② 引用材料原文（如「模型不得选参」）；
#      ③ **禁令句本身**（「该格只放判据，不接入算法」）。
#    被误杀的撰写人各自改词绕开（B-004 把「模型候选」改成「预筛候选」）——
#    正是缺陷 #5 记过的那种消化方式：**门禁缺陷以「大家都学会绕路」的形式被吞掉**。
#    修法：要求**接入语义**（算法/模型 与 接入/挂载 同句且相近），并放行含否定词的禁令句。
MOUNT_RE = re.compile(
    # ⚠️ 间隔里排除**并列分隔符**（＋ + ｜ |）：实测 A-072 §3 的
    #    `…（Case/Event Ledger、Agent 运行台账）＋ 平台后台（模型服务与云资源用量…）`
    #    里「运行台账）＋ 平台后台（模型服务」正好落进 12 字窗 ⇒ 补上「上线|运行」后
    #    这一行成了新增命中（父代理实测 0 行，本机复测 1 行）。跨并列项的「运行…模型」
    #    不是同一个接入短语，故把分隔符挡在窗外。
    r"(?:算法|模型)[^。；;\n＋+|]{0,12}(?:接入|挂载|上挂|上线|运行)"
    # ⚠️ D8（2026-09-13 S1 撰写人实测撞出，**假绿**）：第一分支有「上线|运行」，第二分支漏了。
    #    实测：`- FLOW-08/STG-02 上线模型做自动分档。` ⇒ exit=0；`运行算法出分档。` 同样 exit=0；
    #    而 `模型上线` / `挂载算法` 都判红 —— 同义两种语序两个结论。此处补齐，让两分支对称。
    r"|(?:接入|挂载|上挂|上线|运行)[^。；;\n＋+|]{0,12}(?:算法|模型)")
# 否定/限定词：出现即视为**禁令句或说明句**，不是「把该格当接入点」
NEG_TOKENS = ("不得", "不许", "禁止", "不接", "不挂", "不放", "不含", "不算", "不作",
              "不参与", "不涉及", "不再", "只放", "只写", "仅放", "仅作", "非 M", "不是 M",
              # D10：同义两判 —— 撰写人写「**不校验**模型上线所需的资源齐备性」判红，
              # 换成表里已有的「不接入」却判绿。补齐这组「不是本格在挂」的否定语。
              "不校验", "不核对", "不检查", "不审核", "不评估", "不确认")
# D10：审查/说明语。这些词出现即说明该句在**核对他人的接入是否合规**，没有断言本格挂模型。
REVIEW_TOKENS = ("核对", "审查", "检查", "复核", "确认", "评估", "校验", "审核")


def _claims_model_at_cell(line: str) -> bool:
    """该行是否在**断言**这个格上接了算法/模型（而不是在写禁令、提候选或引原文）。

    ⚠️ D9（2026-09-13 S1 撰写人实测撞出，**假绿**）：原实现是**整行**判定否定词。
    实测：`- FLOW-02/STG-02 不得改参；该格已接入算法模型做有界分类` ⇒ exit=0 ——
    前半句的「不得」豁免了后半句的**真接入断言**。删掉「不得改参」即 exit=1。
    修法：按 `；;/。` 切句，**逐句**判定 —— 某句命中 MOUNT_RE 且该句不含否定词，才判红。

    ⚠️ D10（同一处代码，方向相反，**假红**）：原实现把审查句误判成接入断言。实测三例：
      `接收门禁须核对本次诊断所用方法与版本号，以及外接模型的接入合规性。` ⇒ 判红（应绿）
      `接收门禁核的是…，**不校验**模型上线所需的资源齐备性。` ⇒ 判红（换「不接入」就判绿）
      `- FLOW-04/STG-06 的接收门禁核对算法上线前的版本号。` ⇒ 判红（「算法＋上线」相距 ≤12 字）
    修法：该句含 REVIEW_TOKENS 且审查词出现在**接入短语之前** ⇒ 视为审查/说明句。
    ⚠️ 放宽必须比拦截测得更严：位置条件正是为此而设 —— 若审查词在接入短语**之后**，
    豁免不成立，否则 `已接入算法模型做评估。` 这类真断言会被整片放掉（selftest 反例锁定）。
    """
    for sent in re.split(r"[；;。\n]", line):
        m = MOUNT_RE.search(sent)
        if not m:
            continue
        if any(t in sent for t in NEG_TOKENS):
            continue
        if any(t in sent[:m.start()] for t in REVIEW_TOKENS):
            continue
        return True
    return False


AGT_RE = re.compile(r"\bAGT-0\d{2}\b")

# --- J9 的两半（三个缺陷共用同一段代码）---
SOURCE_RE = re.compile(r"自有埋点|平台后台|第三方 API|需授权|不可得")
# 「声明行」：`取值来源：` / `取值口径：` … —— 说的是数从哪来，不是本段真的用了哪个源
_DECL_LABEL_RE = re.compile(
    r"(?:取值|阈值|默认值|基线|区间|容差档|目标值|分档)\s*(?:来源|口径|说明|依据|出处)")


def _hedge_sentences(text: str) -> list:
    """文本里**真的在陈述对冲**的句子（含对冲词、且不是禁令/引用句）。

    ⚠️ D12（2026-09-13 S1 撰写人实测撞出，**假红**）：原实现是 `any(t in c.text …)` 的
    **全文子串**扫描 —— 与台账 #19（J3 误杀禁令句）同族，J3 早已加了 NEG_TOKENS，J9 一直没加。
    实测：在契约 §4 加一行 `- 标定规则不得以「临时默认值」结案。` ⇒ exit=1 `[J9]`。
    **撰写人越把禁令写清楚越红。**
    修法：逐句判定，命中否定词的句子（禁令/引用）不计。
    ⚠️ 反向控制：`- 标定前先取临时默认值 0.5。` 必须仍然判红（selftest 单独锁定）。
    """
    return [s for s in _sentences(text)
            if any(t in s for t in HEDGE_TOKENS) and not any(t in s for t in NEG_TOKENS)]


def _names_data_source(text: str) -> bool:
    """该段是否**真的**命名了数据源（D16）。

    ⚠️ D16（2026-09-13 S1 撰写人实测撞出，**假绿**）：原实现是段级子串匹配，于是
    `- 取值来源：自有埋点（评估台账）` 这一**声明行自己**就把数据源放进了本段 ——
    「对冲句 + 声明行」自带豁免。实测 §3 只写两行
    （`取值来源：自有埋点（评估台账）` ＋ `取值：阈值待标定，标定前先按 0.05`）⇒ exit=0，
    即规范 §2 明令禁止的「以待标定结案」整份过闸（数字 0.05 又让 J13 成立，双重失效）。
    修法：声明行不算「命名数据源」。豁免本身有正当理由（确实给了命名源的契约不该红），
    故反向控制是「声明行之外真的用了该源」的样本，见 selftest。
    """
    for ln in text.splitlines():
        if SOURCE_RE.search(ln) and not _DECL_LABEL_RE.search(ln):
            return True
    return False


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
    #
    # ⚠️ **标签必须锚定到表行行首**（门禁缺陷 #17，2026-09-13 由三位 S1 撰写人各自撞出）。
    #    原实现是 `re.search(rf"{label}[^\n|]*\|…")` —— 标签被当成**普通子串**，
    #    段内任何一处提前出现的标签词都会劫持该维：实测把 ① 的处置格写成「日**粒度**不足」，
    #    ② 的取值就被读成空串，J5 报「数据要求缺维：粒度」**假红**，撰写人只能改措辞绕开。
    #    与 CLAUDE.md 漏洞 #11/#14 同型（**判据的适用范围被默认成全体，实际只覆盖一个子集**），
    #    且危险方向是双向的 —— 把没值的维写成有值同样容易。
    #    修法：只认「以 `|` 开头、且第一格恰好是该维标签（可带 ①②③④⑤ 序号）」的表行。
    def dim_cells(self) -> dict:
        n = 2 if self.template == "A" else 5
        body = self.section(n)
        out = {}
        for label in DIM_LABELS:
            m = _dim_row(label).search(body)
            if not m:
                out[label] = ("", "")
                continue
            cells = body[m.end():].split("\n", 1)[0]
            parts = [x.strip() for x in cells.split("|")]
            out[label] = (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")
        return out

    # 五维取值：从 2 段（A）或 5 段（B）取值
    def data_dims(self) -> dict:
        n = 2 if self.template == "A" else 5
        body = self.section(n)
        dims = {}
        for label in DIM_LABELS:
            m = _dim_row(label).search(body)
            if not m:
                dims[label] = ""
                continue
            dims[label] = body[m.end():].split("\n", 1)[0].split("|")[0].strip()
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
            elif idx.cell_kind.get(cid) != "M" and _claims_model_at_cell(line):
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
        if access and not _asserts_access(access):
            red("J5", f"获取路径必须**断言**五选一（{'/'.join(ACCESS_ENUM)}），"
                      f"写成否定句不算（D19）：{access!r}")

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

    # --- J9 反对冲（A 的标定规则 §3 / B 的算法可达部分 §1）---
    # ⚠️ D15（2026-09-13 S1 撰写人实测撞出，**假绿**）：原实现整条判据关在
    #    `meta["template"] == "A"` 分支里 ⇒ **66 份 B 契约上一条判据都没有**。
    #    实测：把某 B 契约 §1 换成 `- 取值：净增量阈值待标定，标定前先按大于 0` ⇒ exit=0。
    #    修法：对 B 也生效。**落在哪一段是量出来的，不是照搬 A**：B 的取值与算式写在
    #    §1「算法可达部分」（§2 不可达 / §3 外部证据 / §5 数据要求都不承载本业务取值），
    #    故 B 取 §1。全库实测：B 的 66 份在**任何段**里 HEDGE_TOKENS 命中都是 0 份，
    #    所以这条补的是覆盖面，今天新增判红 0 份 —— 不会制造假红。
    #    全文的「临时默认值」硬禁令两边都扫（它对 A/B 同样是死法）。
    if c.written:
        rule = c.section(3 if meta["template"] == "A" else 1)
        if any("临时默认值" in s for s in _hedge_sentences(c.text)):
            red("J9", "出现「临时默认值」—— 把论文参数留作默认值正是 SkillOpt 补丁的死法")
        if _hedge_sentences(rule) and not _names_data_source(rule):
            red("J9", "标定规则以对冲结案：既没有命名数据源，也没有算式")

    # --- J12 cards 必须真实存在（D17：先问「我用的信息源能看见它吗」）---
    for slug in (BC.norm_val(c.fm.get("cards")) or []):
        state, src = card_sources(idx).resolve(slug)
        if state == "found":
            continue
        if state == "unverifiable":
            # run() 会据此 exit 2。**不在这里判红** ——「我没能力查」不是「它不存在」。
            continue
        red("J12", f"cards 里的 {slug!r} 在 {src} 里找不到")
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
            # ⚠️ D1（2026-09-13 S1 撰写人实测撞出，**假绿**）：原实现是
            #    `items`＝§2 里 `- ` 起头行数、`marks`＝`re.findall("替代路径", body2)` 的
            #    **出现次数**，只比 `marks < len(items)` —— 是**总额顶账**，不是逐条配对。
            #    两个实测最小复现：
            #      ① 两条 bullet，删掉其中一条的箭头行、在**另一条**正文里补写
            #         「（替代路径另见下条）」⇒ 计数 2=2 ⇒ exit=0；
            #      ② 一条 bullet 正文写「本条没有替代路径可写」⇒ 照样算一个 mark ⇒ exit=0。
            #    修法：**逐条配对**。把 §2 切成「条目块」（条目行 + 其后续续行），
            #    **每个块自身**必须含一行形如 `^\s*(?:→|->|=>)\s*替代路径` 的箭头行。
            #    ⚠️ 必须允许续行：撰写人普遍写成条目行下面缩进一行放箭头
            #    （全库实测 459 行「替代路径」里 387 行正是这种独立箭头行）。
            body2 = c.section(2)
            blocks = _item_blocks(body2)
            if not blocks:
                red("J13", "§2 没有列出任何不可达条目")
            else:
                lacking = [b[0].strip()[:32] for b in blocks
                           if not any(ALT_LINE_RE.match(ln) for ln in b)]
                if lacking:
                    red("J13", f"§2 有 {len(blocks)} 条不可达，其中 {len(lacking)} 条没有独立的"
                               f"「{ALT_MARK}」箭头行 —— 只写「做不到」是免责，"
                               f"写出「谁在何时补」才是分工：{'；'.join(lacking[:3])}")
        # 五维：取值为缺口时，缺值处置格必填
        # ⚠️ D5：判据与 §3 同一套 —— 剥掉交叉引用后再判数字，否则
        #    `见 §2 第 3 条` / `暂无，详见上表 3 处` 这类**条款号与页内计数**冒充处置。
        # ⚠️ D22（2026-09-13 S1 撰写人实测撞出，**假绿**）：原实现只扫「取值」列
        #    （`any(tok in val …)`）。撰写人把缺口陈述写进**处置列**是常见写法，实测
        #    ① 行第三列写成 `**缺口在本维**：〈海关与合规条款库〉尚未接入。` ⇒ exit=0 ——
        #    缺口声明与「先用什么＋何时换」的义务**一起失效**。修法：两列都扫。
        for label, (val, disp) in c.dim_cells().items():
            # D26 桶①：缺口词必须通过语境判据（「非缺口」开头的行是撰写人的显式否认，
            # A-033 的处置格正是 `非缺口。若跨渠道统一口径尚未签发：…` —— 条件从句，不是缺口）
            if not (has_gap_declaration(val) or has_gap_declaration(disp)):
                continue
            # D28：处置按**整行**判（合并单元格的写法必须放行），但缺口句不算自己的处置
            if not row_disposal(val, disp):
                red("J13", f"{label} 声明了缺口，但整行没有可判定的处置"
                           f"（既无替代取值也无替换条件）—— 取值={val[:36]!r} 处置={disp[:36]!r}")
    return bad


_CARD_CACHE = {}

# --- J12 的卡片信息源（D17）---
# ⚠️ 原实现只有三条分支，其中「已装线」写死成 `Path.home()/".dsh/skills/<slug>/SKILL.md"`。
#    2026-09-13 实测：**125/139 份已撰写契约的 cards 至少有一个 slug 只靠这一条 HOME
#    分支才判绿**（图谱只含精选线 146 张，而已装线 1338 张的事实源不在这里）。
#    ⇒ 换机 / CI / S12 独立核对器里**整片转红**，而契约本身没错。
#    这与台账 #23「读空却打出漂亮的账」是同一枚硬币的两面：**判据用错了信息源**。
# 修法：把「卡片目录这件事有几个来源」写下来，逐源可测，并把结局拆成**三态**：
#   found        —— 某个来源认了它（并记下是哪一个）
#   absent       —— 所有**枚举型**来源都可用，且都不认它（= 真的不存在 ⇒ 判红）
#   unverifiable —— 有枚举型来源不可用 ⇒ **「查不到」与「没能力查」是两件事**，
#                   由 run() 按退出码 2 处理（拿不到输入 ≠ 判红 ≠ 通过）
# 枚举型来源 = 能列出整个命名空间的两处：图谱 cards[]（精选线）与产品侧 classification.json
# （已装线）。路径与 `build_contract_workpack.py` 的常量同源，**不另造第二份映射表**。
PRODUCT_CLASSIFICATION = Path("/Users/lute/project/Magpie-Horch/packages/capabilities/"
                              "dsh-paper2skills/data/classification.json")
INSTALLED_SKILLS_DIR = Path.home() / ".dsh" / "skills"
VAULT_DIR = REPO_ROOT / "paper2skills-vault"


def _load_product_slugs(path: Path):
    """产品侧 classification.json → slug 全集。返回 (slugs, available)。"""
    if not path.is_file():
        return set(), False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set(), False
    out = set()

    def walk(node):
        if isinstance(node, dict):
            for k in ("slug", "name", "id", "card_slug"):
                v = node.get(k)
                if isinstance(v, str) and "/" not in v:
                    out.add(v)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return out, True


class CardSources:
    """J12 的四个信息源，**逐源可测、逐源报告可用性**。

    构造参数留了注入点：selftest 靠它们分别拿掉某一个来源，证明
    「来源少了 ⇒ 说不清」与「来源都在 ⇒ 说它不存在」是两条不同的结论。
    """

    NAMES = ("graph.cards", "product.classification", "~/.dsh/skills", "vault")

    def __init__(self, idx, *, product_path=None, installed_dir=None, vault_dir=None):
        self.idx = idx
        self.product_path = Path(product_path) if product_path else PRODUCT_CLASSIFICATION
        self.installed_dir = Path(installed_dir) if installed_dir else INSTALLED_SKILLS_DIR
        self.vault_dir = Path(vault_dir) if vault_dir else VAULT_DIR
        self.product_slugs, self.product_available = _load_product_slugs(self.product_path)
        self.graph_available = bool(idx.graph.get("cards"))
        self.installed_available = self.installed_dir.is_dir()
        self.vault_available = self.vault_dir.is_dir()

    def resolve(self, slug: str):
        """→ (state, source)；state ∈ {found, absent, unverifiable}。"""
        # ① 图谱 cards[]（精选线 146 张，按 id / 卡名）
        for card in self.idx.graph.get("cards", []):
            name = card["name"]
            stem = Path(name).stem if name.endswith(".md") else name
            if stem == slug or name == slug:
                return "found", "graph.cards"
        # ② 产品侧 classification.json（已装线 1338 张的事实源）
        if self.product_available and slug in self.product_slugs:
            return "found", "product.classification"
        # ③ 实际安装位置（**只是佐证**，不再是唯一入口）
        if self.installed_available and (self.installed_dir / slug / "SKILL.md").exists():
            return "found", "~/.dsh/skills"
        # ④ vault 域目录顶层（slug 常形如 Skill-Xxx）
        if self.vault_available:
            for dom in self.vault_dir.iterdir():
                if dom.is_dir() and (dom / f"{slug}.md").exists():
                    return "found", "vault"
        # 都没认。能不能断定它**不存在**？取决于枚举型来源是否齐全。
        missing = [n for n, ok in (("graph.cards", self.graph_available),
                                   ("product.classification", self.product_available)) if not ok]
        if missing:
            return "unverifiable", "、" + "、".join(missing) + " 不可用"
        return "absent", "graph.cards+product.classification"

    def describe(self) -> str:
        return " · ".join(
            f"{n}={'可用' if ok else '不可用'}" for n, ok in
            (("graph.cards", self.graph_available),
             ("product.classification", self.product_available),
             ("~/.dsh/skills", self.installed_available),
             ("vault", self.vault_available)))


def find_card(slug: str, idx) -> bool:
    """兼容入口：`slug 被某个可用来源认定` 即为真（unverifiable 不算真，也不算假）。"""
    if slug in _CARD_CACHE:
        return _CARD_CACHE[slug]
    hit = card_sources(idx).resolve(slug)[0] == "found"
    _CARD_CACHE[slug] = hit
    return hit


_SOURCES = {}


def card_sources(idx) -> CardSources:
    """按 idx 缓存一份信息源清单（同一轮 run 里四个来源的可用性不会变）。"""
    key = id(idx)
    if key not in _SOURCES:
        _SOURCES[key] = CardSources(idx)
    return _SOURCES[key]


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
    """139 份契约的**正式**位置只有 `contracts/A/` 与 `contracts/B/` 两处。"""
    if paths:
        return [Path(p) for p in paths]
    return sorted(p for d in (CONTRACTS_DIR / "A", CONTRACTS_DIR / "B")
                  for p in d.glob("CTR-*.md") if p.is_file())


def sample_files() -> list:
    """`A/` `B/` 之外的 `CTR-*.md`（F6/F7 试点样本、v1 底本存档）—— **不计入 139**。

    ⚠️ 首版把 `CONTRACTS_DIR.rglob("CTR-*.md")` 当全集，于是 `contracts/v2/B/` 里 3 份
    F7 试点样本被算成 139 的成员。症状只在**文件总数超过 139 之后**才显形：
    `build_contracts.py --check` 打出「核对率 102.2%」（分子 142 > 分母 139）却照样通过；
    `check_contracts.py --all` 打出「B 69/66」的 🔴 却不计入退出码。
    在此之前它一直「看起来正常」—— 与 CLAUDE.md 记的「v1 底本留原地会与 v2 同 id
    重复入账」是同一条教训的第二次命中：**全集的定义必须写下来，否则它就是默认值。**
    """
    keep = {p.resolve() for p in collect_files([])}
    return sorted(p for p in CONTRACTS_DIR.rglob("CTR-*.md")
                  if p.is_file() and p.resolve() not in keep)


def count_problems(contracts, exp_a: int, exp_b: int) -> list:
    """J10：**计数超界**与**同一条责任两份正式契约** —— 两条判据各自独立。

    抽成纯函数是为了能被单独喂样本：首版把两条判据揉在 `run()` 里且只用一个反向控制，
    结果**任何一条都能替另一条顶账**（实测：只测「重复」时，去掉「超界」照样 exit 1；
    只测「超界」时，去掉「重复」照样 exit 1 ⇒ 两条都成了摆设）。

    ⚠️ 更要紧的是首版把超界只 `print` 一行 🔴 就完事，紧接着照样打「✅ 全部判据通过」
    并 exit 0 —— 任何用退出码判绿的消费者（撰写人 / sync / CI）都会把超界当成通过。
    **颜色不是判据，退出码才是。**
    """
    n_a = sum(1 for c in contracts if c.template == "A")
    n_b = sum(1 for c in contracts if c.template == "B")
    out = []
    if n_a > exp_a or n_b > exp_b:
        out.append({"file": "—", "code": "J10",
                    "msg": f"计数超界：A {n_a}/{exp_a}、B {n_b}/{exp_b}"})
    resp = Counter(c.fm.get("responsibility") for c in contracts if c.fm)
    for r, n in sorted(resp.items()):
        if r and n > 1:
            out.append({"file": "—", "code": "J10",
                        "msg": f"同一条责任有 {n} 份正式契约：{r} ⇒ "
                               f"判哪一份由文件顺序决定（静默二选一）"})
    return out


def run(files, idx, json_out=None) -> int:
    assignments = idx.assignments()
    total_expected = len(assignments)
    exp_a = sum(1 for m in assignments.values() if m["template"] == "A")
    exp_b = total_expected - exp_a

    contracts = [Contract(p) for p in files]
    written = [c for c in contracts if c.written]
    # --- D17：先分清「卡不存在」与「我没能力查卡」，再谈判红 ---
    #                            **「读空却打出漂亮的账」的反面**（台账 #23）。
    # 若某个枚举型来源（图谱 cards[] / 产品侧 classification.json）不可用，而 slug 在所有
    # 可用来源里都找不到，那么**结论不可判定** ⇒ 走退出码 2（拿不到输入），
    # 既不判红（契约没错），也不判绿（没查过）。
    src = card_sources(idx)
    blocked = []
    for c in contracts:
        for slug in (BC.norm_val(c.fm.get("cards")) or []):
            state, why = src.resolve(slug)
            if state == "unverifiable":
                blocked.append((_rel(c.path), slug, why))
    if blocked:
        print(f"❌ 输入没拿到（退出码 2，**不是判红**）：J12 无法判定 {len(blocked)} 个卡 slug —— "
              f"信息源可用性：{src.describe()}", file=sys.stderr)
        for f_, slug, why in blocked[:10]:
            print(f"   {f_}: {slug!r}（{why}）", file=sys.stderr)
        if len(blocked) > 10:
            print(f"   …（另 {len(blocked) - 10} 个）", file=sys.stderr)
        print("   ⇒「这个 slug 不存在」与「我没有能力查这个 slug」是两件事；"
              "后者请先补齐信息源，**不要据此改契约**。", file=sys.stderr)
        return 2
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
    samples = sample_files()
    print(f"样本/存档（A/ B/ 之外，**不计入 139**）：{len(samples)} 份"
          + (" —— " + "、".join(_rel(p) for p in samples) if samples else ""))
    # J10：计数超界与**同一条责任被建两次**都必须计入退出码。
    # ⚠️ 首版这条只 print 一行 🔴 就完事，紧接着照样打「✅ 全部判据通过」并 exit 0 ——
    #    任何用退出码判绿的消费者（撰写人 / sync / CI）都会把超界当成通过。
    #    「🔴 后面跟 ✅」本身就是一种假绿：**颜色不是判据，退出码才是。**
    # J10：计数超界 + 同一条责任两份正式契约（判据在 count_problems 里，可被单独喂样本）
    problems.extend(count_problems(contracts, exp_a, exp_b))
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

        def cc_check(path):
            return check_one(Contract(path), idx, assignments)

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
            # --- #18：J13 的「取值」不得被**来源行**形式满足（2026-09-13 S1 三位撰写人实测撞出）---
            ("J13", "#18 只留「取值来源」行（数字在来源行里）就想过闸",
             re.sub(r"## 3 标定规则.*?## 4",
                    "## 3 标定规则\n\n- 取值来源：自有埋点 × AGT-045 指标契约（近 2 个完整年度）\n\n## 4",
                    base_a, flags=re.S)),
            # --- #19：J3 不得因「同行出现『模型』二字」就判红 R/D 格 ---
            ("J3", "#19 把 R 格说成接入点（应判红）",
             base_a.replace("## 6 冻结与不许自动放行的情形",
                            "## 6 冻结与不许自动放行的情形\n\n- FLOW-07/STG-02 接入算法模型做有界分类", 1)),
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
        # ③ #17 反向：① 的处置格里出现「粒度」二字，**不得**把 ② 的取值读成空串（原实现会假红）
        probe17 = td / "ok-dim-hijack.md"
        probe17.write_text(base_a.replace(
            "| ① 获取路径 | 自有埋点 | — |",
            "| ① 获取路径 | 自有埋点 | 非缺口。若某源只有日粒度汇总：先用日粒度顶替并标「粒度不足」 |", 1),
            encoding="utf-8")
        got17 = {c for c, _ in check_one(Contract(probe17), idx, assignments)}
        expect("J5" not in got17, f"#17 反向：① 处置格里的维度词不得劫持 ② 的取值（实得 {sorted(got17) or '全绿'}）")
        # ③b #17 反向之二：首格写成「① 获取路径（平台后台与埋点）」不得判红
        probe17b = td / "ok-dim-paren.md"
        probe17b.write_text(base_a.replace(
            "| ① 获取路径 | 自有埋点 | — |",
            "| ① 获取路径（平台后台与独立站埋点） | 自有埋点 | 非缺口 |", 1), encoding="utf-8")
        got17b = {c for c, _ in cc_check(probe17b)}
        expect("J5" not in got17b, f"#17 反向之二：维标签带括注不得判红（实得 {sorted(got17b) or '全绿'}）")

        # ④ #19 反向：禁令句与「模型候选」措辞都不得被判成「把 R/D 格说成算法接入」
        for label, extra in (("#19 反向：禁令句",
                              "- FLOW-07/STG-02 只放判据与门禁，**不接入**算法模型"),
                             ("#19 反向：业务措辞「模型候选」",
                              "- FLOW-07/STG-02 的结论栏写「模型候选 + 待签」，不含算法接入")):
            probe19 = td / f"ok-j3-{abs(hash(label)) % 10**6}.md"
            probe19.write_text(base_a.replace(
                "## 6 冻结与不许自动放行的情形",
                "## 6 冻结与不许自动放行的情形\n\n" + extra, 1), encoding="utf-8")
            got19 = {c for c, _ in check_one(Contract(probe19), idx, assignments)}
            expect("J3" not in got19, f"{label} 不得判红（实得 {sorted(got19) or '全绿'}）")

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

        # =====================================================================
        # J13 / J3 / J9 修复批（D1–D26，2026-09-13）—— 每条修复**正反两份**样本
        # ⚠️ 纪律：没有两条判据共用同一份反向控制（本仓库实测过「共用 ⇒ 删掉任一条
        #    自检照样全绿」的互相顶账）。下面每组反例都只服务它自己那一条。
        # =====================================================================
        import contextlib
        import io

        def codes_of(text, tag):
            p = td / f"probe-{tag}-{abs(hash(text)) % 10**6}.md"
            p.write_text(text, encoding="utf-8")
            return {c for c, _ in check_one(Contract(p), idx, assignments)}

        def a_sec3(body):
            return re.sub(r"## 3 标定规则.*?## 4",
                          "## 3 标定规则\n\n" + body + "\n\n## 4", base_a, flags=re.S)

        def b_sec1(body):
            return re.sub(r"## 1 算法可达部分.*?## 2",
                          "## 1 算法可达部分\n\n" + body + "\n\n## 2", base_b, flags=re.S)

        def b_sec2(body):
            return re.sub(r"## 2 不可达部分.*?## 3",
                          "## 2 不可达部分\n\n" + body + "\n\n## 3", base_b, flags=re.S)

        def a_sec6(line):
            return base_a.replace("## 6 冻结与不许自动放行的情形",
                                  "## 6 冻结与不许自动放行的情形\n\n" + line, 1)

        def full(text):                      # 把某一段整体换掉（D1 用）
            return text

        def a_dim(val, disp="—"):
            return base_a.replace("| ③ 回溯深度 | 2 个完整年度 | — |",
                                  f"| ③ 回溯深度 | {val} | {disp} |", 1)

        # ---- D1：B §2 替代路径必须**逐条配对**（总额顶账不算）----
        expect("J13" in codes_of(b_sec2(
            "- 需外部事实：A\n  → 替代路径：由 AGT-014 补齐 A\n"
            "- 需外部事实：B（替代路径另见上条）\n"), "D1a"),
            "D1：另一条正文里提一句「替代路径」不得替缺箭头那条顶账")
        expect("J13" in codes_of(b_sec2(
            "- 需外部事实：A\n  → 替代路径：由 AGT-014 补齐 A\n"
            "- 需外部事实：B\n  本条没有替代路径可写\n"), "D1b"),
            "D1：一条写「本条没有替代路径可写」不得被当成一个 mark")
        expect("J13" in codes_of(b_sec2(
            "- 需外部事实：A\n  → 替代路径：由 AGT-014 补齐 A\n- 需外部事实：B\n"), "D1c"),
            "D1 反向控制（专用）：真缺箭头的那一条必须仍然判红")
        expect("J13" not in codes_of(b_sec2(
            "- 需外部事实：A 与 B\n  → 替代路径：由 AGT-014 在 STG-04 前补齐；补不上时先按上一轮名录顶替\n"),
            "D1d"),
            "D1 反向控制（专用）：条目行 + 缩进续行放箭头是撰写人的普遍写法，不得误杀")

        # ---- D7：§2 条目正则必须认合法编号（假红）----
        expect("J13" not in codes_of(b_sec2("\n".join(
            f"{i}) 不可达{'①②③④⑤⑥'[i-1]}：需外部事实 X{i}\n  → 替代路径：由 AGT-014 补齐"
            for i in range(1, 7))), "D7a"),
            "D7 反向控制（专用之②）：`1)` 编号六条且六条箭头齐全 ⇒ 不得再报「§2 没有列出任何不可达条目」")
        expect("J13" not in codes_of(b_sec2(
            "（1）不可达甲：需外部事实 X\n  → 替代路径：由 AGT-014 补齐\n"
            "① 不可达乙：需外部事实 Y\n  → 替代路径：由 AGT-014 补齐\n"
            "• 不可达丙：需外部事实 Z\n  → 替代路径：由 AGT-014 补齐\n"), "D7b"),
            "D7：全角括号/圈号/圆点三种项目符都必须被认")
        expect("J13" in codes_of(b_sec2(
            "1) 不可达甲：需外部事实 X\n2) 不可达乙：需外部事实 Y\n  → 替代路径：由 AGT-014 补齐\n"), "D7c"),
            "D7 反向控制（专用）：认了编号 ≠ 放掉「缺箭头」")

        # ---- D2/D3/D4/D11/D18/D20/D24：取值标签与数字归属 ----
        expect("J13" in codes_of(a_sec3(
            "- 算式：按分组分配率取最大差\n"
            "- **不得移植的部分**：卡内异常阈值 0.75 与 0.2、迭代 3 轮、样本 300 条\n"), "D2a"),
            "D2：禁令行里的「阈值」不得冒充本业务取值（锚定 + 冒号两条一起）")
        expect("J13" not in codes_of(a_sec3(
            "- 算式：按分组分配率取最大差\n- T2 取值：**容差档 = 0**（D 不得超过 A 一个单位）\n"), "D2b"),
            "D2 反向控制（专用）：`- T2 取值：**容差档 = 0**（…不得…）` 是真取值行，"
            "**不许用「含否定词就跳过」的黑名单**（实测那会误杀 9 份契约）")
        expect("J13" in codes_of(a_sec3("- 取值域：对齐〈采集台账〉（覆盖近 2 个完整年度）\n"), "D2c"),
            "D2（父代理写死的条件②）：标签后是「域」不是冒号 ⇒ 不算取值")
        expect("J13" in codes_of(a_sec3("- 降级比例不设目标值，只用 T3 的成本线\n"), "D2d"),
            "D2（父代理写死的条件①）：「目标值」在行中，锚定不成立 ⇒ 不算取值")
        # ⚠️ 这一条**专为锚定而设**：光有「紧跟冒号」挡不住它（该标签后面确实就是冒号），
        #    只有「标签必须落在行首结构前缀之后」才挡得住 —— 否则变异 M3a（删掉锚定）
        #    在自检里会漏网（实测过）。
        expect("J13" in codes_of(a_sec3(
            "- 算式：按分组分配率取最大差\n"
            "- 说明：卡面阈值：0.75 与 0.2（本业务不采用，只登记为卡面数）\n"), "D2e"),
            "D2：锚定与冒号是**两条独立条件** —— 标签在行中（前缀是 `- 说明：卡面`）且后面"
            "紧跟冒号时，必须仍然不算取值")
        expect("J13" in codes_of(a_sec3(
            "- 取值来源：自有埋点 × AGT-045 指标契约\n- 取值口径：按近 2 个完整年度算\n"), "D3a"),
            "D3：「取值来源/取值口径」两行不是取值（唯一数字来自年度叙述）")
        expect("J13" not in codes_of(a_sec3(
            "- 取值来源：自有埋点\n- 取值：容差档 5%\n"), "D3b"),
            "D3 反向控制（专用）：声明行之后真有取值行 ⇒ 不得判红")
        expect("J13" in codes_of(a_sec3("- 取值：见 §4 第 2 条\n"), "D4a"),
            "D4：裸条款号不得冒充取值")
        expect("J13" in codes_of(a_sec3(
            "- 取值来源：自有埋点\n"
            "- 阈值：按 FLOW-08/STG-06 的接收门禁判定；本业务暂无独立取值，接入后替换。\n"), "D11a"),
            "D11：数字全部来自格号 FLOW-08/STG-06 ⇒ 不得冒充取值")
        expect("J13" in codes_of(a_sec3(
            "- 取值来源：自有埋点\n"
            "- 阈值：按该阶段接收门禁判定；本业务暂无独立取值，接入后替换。\n"), "D11b"),
            "D11 配对对照（专用）：同一句话只把格号换成文字后必须同样判红 —— "
            "红绿差别只能来自格号里的数字，不能来自措辞")
        expect("J13" in codes_of(a_sec3("- 阈值：见上文的数据要求（近 2 个完整年度）\n"), "D18a"),
            "D18：转指行必须整行判掉（「2 个」是 §2 的，不是 §3 的）")
        expect("J13" not in codes_of(a_sec3("- 取值：容差档 5%（口径见 §2）\n"), "D18b"),
            "D18 反向控制（专用）：真给了取值、括注里写「口径见 §2」⇒ 必须仍然判绿（最易误杀）")
        expect("J13" in codes_of(a_sec3("- 阈值：本业务暂无此数据（第 2 版口径）\n"), "D20a"),
            "D20：多加「（第 2 版口径）」四个字不得从判红变判绿")
        expect("J13" not in codes_of(a_sec3(
            "- 取值来源与取值：自有埋点（Agent 运行日志）× AGT-045 指标契约；"
            "Service 可用性 ≥ 99%、Task 完成率 ≥ 95%\n"), "D24a"),
            "D24 反向控制（专用）：`取值来源与取值：` 合并标签行**真给了取值**，不得整行跳过判红")
        expect("J13" in codes_of(a_sec3("- 取值来源：自有埋点 × AGT-045 指标契约\n"), "D24b"),
            "D24 反向控制（专用之二）：纯声明行（#18）仍然不算取值 —— 放宽与拦截不许互相顶账")

        # ---- D5/D6/D22：五维缺口与处置 ----
        expect("J13" in codes_of(a_dim("本业务暂无此数据", "见 §2 第 3 条"), "D5a"),
            "D5：缺值处置格写条款号不得冒充处置")
        expect("J13" in codes_of(a_dim("本业务暂无此数据", "暂无，详见上表 3 处"), "D5b"),
            "D5：处置格写「详见上表 3 处」不得冒充处置")
        expect("J13" not in codes_of(a_dim("本业务暂无此数据", "先用 5% 顶替"), "D5c"),
            "D5 反向控制（专用）：处置格真给了替代取值 ⇒ 不得判红")
        expect("J13" in codes_of(base_b.replace(
            "| ① 获取路径 | 需授权 | — |",
            "| ① 获取路径 | 需授权 | **缺口在本维**：〈海关与合规条款库〉尚未接入。 |", 1), "D22a"),
            "D22：缺口陈述写进**处置列**不得整格免检（原实现只看取值列）")
        expect("J13" not in codes_of(base_b.replace(
            "| ① 获取路径 | 需授权 | — |",
            "| ① 获取路径 | 需授权 | 先按上一版条款库暂按，新库接入后替换 |", 1), "D22b"),
            "D22 反向控制（专用）：处置列真的给了处置 ⇒ 不得判红")
        expect("J13" in codes_of(base_b.replace(
            "| ③ 回溯深度 | 2 个完整年度 | — |",
            "| ③ 回溯深度 | 本业务没有该数据 |  |", 1), "D6a"),
            "D6：「本业务没有该数据」必须被认成缺口（原词表漏词 ⇒ 整维跳过）")
        expect("J13" not in codes_of(base_b.replace(
            "| ③ 回溯深度 | 2 个完整年度 | — |",
            "| ③ 回溯深度 | 按缺口账登记的 3 处 | 先用 5% 顶替 |", 1), "D6b"),
            "D6 反向控制（专用）：**不得补单字「缺」** —— 「缺口账」是正常表述，不是缺口声明")

        # ---- D21/D23/D27：缺口义务是**逐条（同块）**的，不是段级、也不是邻块 ----
        expect("J13" in codes_of(a_sec3("- 取值：容差档 5%。\n- 本业务暂无季节性先验来源。\n"), "D23a"),
            "D23：段内任意一处取值不得解除全段的缺口义务（D21 的段级短路）")
        expect("J13" not in codes_of(a_sec3(
            "- 取值：容差档 5%。\n"
            "- 本业务暂无季节性先验来源。\n"
            "  替换条件：〈采购台账〉接入后改用实测先验。\n"), "D23b"),
            "D23 反向控制（专用）：处置写在同一**条目块**内（条目行 + 缩进续行，"
            "这正是 B-040 §1 的真实写法）⇒ 不得判红")
        # D27（2026-09-13 主控反向控制实测撞出，**潜伏的洞**）
        expect("J13" in codes_of(a_sec3(
            "- 取值：容差档 5%。\n"
            "- 本业务暂无季节性先验来源。\n"
            "- 替换条件：〈采购台账〉接入后改用实测先验。\n"), "D27a"),
            "D27：处置写在**另一条 bullet** 里不算 —— 邻块豁免会让任意一条带「替换/改用」的"
            "条目救活缺口条（原先的 ±1 句/±1 块窗口）")
        expect("J13" in codes_of(b_sec1(
            "- 取值：排序分 0–100，取数即得\n"
            "- 本业务的季节性先验来源暂无。\n"
            "- 换卡时只替换方法实现，其余要求不变。\n"), "D27b"),
            "★ D27 反向控制（主控指定）：`换卡时只替换方法实现，其余要求不变。` 这条**与缺口"
            "毫无关系的样板话**不得发出豁免 —— 实测 60/66 份 B 契约的 §1 末句都是这一族样板")

        # ---- D28：五维「缺口 + 处置」合并写进取值格（假红）----
        expect("J13" not in codes_of(base_b.replace(
            "| ① 获取路径 | 需授权 | — |",
            "| ① 获取路径 | 需授权；平台侧消息回指不到本司节点时取不到分子分母。"
            "故先按〈Case Charter〉导出平台侧消息并注明「未落本司留痕」；"
            "等该渠道写入 Case/Event Ledger 之后，改用本司自存的答复留痕顶掉它。 |",
            1), "D28a"),
            "D28 反向控制（专用）：缺口与处置**合并写在同一个取值格**、第三列留空 —— "
            "实测 B-032/033/034/036/039/040 六份 8 格是这个写法，已履行义务，不得判红")
        expect("J13" in codes_of(base_b.replace(
            "| ① 获取路径 | 需授权 | — |",
            "| ① 获取路径 | 需授权；平台侧消息回指不到本司节点时取不到分子分母。 |", 1), "D28b"),
            "D28 反向控制（专用之二）：合并格但**只有缺口没有处置** ⇒ 必须仍然判红")

        # ---- D26 桶①：缺口词的**语境判据**（词在、意思不在）----
        # 每个探针都先给一行真取值，让 J13 **只可能**因缺口义务而红 —— 否则探针会
        # 被「§3 只有算式、没有本业务取值」那条分支抓住，测的东西就串了。
        _V = "- 取值：容差档 5%\n"
        for tag, body, why in (
                ("D26a", _V + "- 算式：`缺失率 = 当期未产生记录的期望批次 ÷ 当期期望批次数`\n",
                 "「缺失率」是指标名，不是缺口陈述"),
                ("D26b", _V + "- 边界：本规则的输出是「外部在说什么、承诺有没有被兑现」。\n",
                 "「有没有」是疑问构词，不是缺口"),
                ("D26c", _V + "- 算式：`未决数 = count(按 idempotency_key 查外部状态返回「未知」的对象)`\n",
                 "「未知」是引号里的枚举值，不是缺口"),
                ("D26d", _V + "- 判据：许可缺失或已撤回的，一律不出营销触达。\n",
                 "「许可缺失…的，一律…」是条件从句，不是本段的缺口陈述")):
            expect("J13" not in codes_of(a_sec3(body), tag),
                   f"D26 桶① 反向控制（专用）：{why} ⇒ 不得判红")
        expect("J13" in codes_of(a_sec3(_V + "- 本业务暂无季节性先验来源。\n"), "D26e"),
            "D26 桶① 反向控制（专用之二）：**真缺口无处置**必须仍然判红 —— "
            "语境判据不是把词删掉，不许把真缺口一起放走")
        # D26 桶③：判据必须认它自己模板规定的词
        expect("J13" not in codes_of(a_sec3(
            _V +
            "- 边界：独立站 holdback 建成前，**不可得** 的是受众级随机对照，"
            "**替代取值**为同店同渠道的准实验窗。\n"), "D26f"),
            "D26 桶③：处置就在同一句里（「替代取值为…」）⇒ 不得判红；"
            "B 模板强制要求的写法字面就是「→ 替代路径：」")

        # ---- D8/D9/D10：J3 ----
        expect("J3" in codes_of(a_sec6(f"- FLOW-08/STG-02 上线模型做自动分档。"), "D8a"),
            "D8：第二分支必须与第一分支对称（「上线模型」不得漏）")
        expect("J3" in codes_of(a_sec6(f"- FLOW-08/STG-02 运行算法出分档。"), "D8b"),
            "D8：「运行算法」同族")
        expect("J3" in codes_of(a_sec6(
            f"- FLOW-08/STG-02 不得改参；该格已接入算法模型做有界分类"), "D9a"),
            "D9：前半句的否定词不得豁免后半句的真接入断言（逐句判定）")
        expect("J3" not in codes_of(a_sec6(
            f"- FLOW-08/STG-02 不得改参；该格只放判据，不接入算法模型"), "D9b"),
            "D9 反向控制（专用）：两句都否定 ⇒ 不得判红")
        for tag, sent, why in (
                ("D10a", f"- FLOW-08/STG-06 接收门禁须核对本次诊断所用方法与版本号，以及外接模型的接入合规性。",
                 "这是核对他人的接入是否合规"),
                ("D10b", f"- FLOW-08/STG-06 接收门禁核的是可证伪性与资源来源，**不校验**模型上线所需的资源齐备性。",
                 "「不校验」与「不接入」同义，不得两判"),
                ("D10c", "- FLOW-08/STG-06 的接收门禁核对算法上线前的版本号。",
                 "「算法＋上线」相距 ≤12 字就命中，但该句只谈核对版本号")):
            expect("J3" not in codes_of(a_sec6(sent), tag),
                   f"D10 反向控制（专用）：{why} ⇒ 不得判红")
        expect("J3" in codes_of(a_sec6(f"- FLOW-08/STG-02 已接入算法模型做有界分类。"), "D10d"),
            "★ D10 放宽的反向控制（父代理指定）：真正的接入断言必须仍然判红")
        expect("J3" in codes_of(a_sec6(f"- FLOW-08/STG-02 已接入算法模型做评估。"), "D10e"),
            "★ D10 放宽的反向控制（本实现附加）：审查词在接入短语**之后**不构成豁免，"
            "否则「已接入算法模型做评估」会被整片放掉")
        expect("J3" in codes_of(a_sec6(
            f"- FLOW-08/STG-02 已接入算法模型做有界分类；门禁另需核对版本号。"), "D10f"),
            "★ D10 放宽的反向控制（父代理指定之二）：同句后半出现「核对」不得救活前半的真断言")

        # ---- D12/D15/D16：J9 ----
        expect("J9" not in codes_of(a_sec6("- 标定规则不得以「临时默认值」结案。"), "D12a"),
            "D12：禁令句不得被判红（撰写人越把禁令写清楚越红是荒谬的）")
        expect("J9" in codes_of(a_sec3("- 标定前先取临时默认值 0.5。\n"), "D12b"),
            "★ D12 反向控制（专用）：真的用了临时默认值必须仍然判红")
        expect("J9" in codes_of(b_sec1("- 取值：净增量阈值待标定，标定前先按大于 0\n"), "D15a"),
            "D15：J9 必须对 B 生效（原实现关在 A 分支里 ⇒ 66 份 B 契约零判据）")
        expect("J9" not in codes_of(b_sec1(
            "- 取值：按〈平台后台〉近 2 个完整年度的准时交付率加权，排序分 0–100\n"
            "- 说明：换仓或口径变更后须重新标定一次\n"), "D15b"),
            "★ D15 反向控制（专用）：B §1 有对冲句但**真的用了命名源** ⇒ 不得判红")
        expect("J9" in codes_of(a_sec3(
            "- 取值来源：自有埋点（评估台账）\n- 取值：阈值待标定，标定前先按 0.05\n"), "D16a"),
            "D16：「取值来源：」这一声明行不得自己给对冲句发豁免")
        expect("J9" not in codes_of(a_sec3(
            "- 取值来源：自有埋点（评估台账）\n"
            "- 算式：按自有埋点台账近 2 个完整年度的分配率取最大差\n"
            "- 取值：容差档 5%（换仓后须重新标定一次）\n"), "D16b"),
            "★ D16 反向控制（专用）：声明行之外**真的用了**该命名源 ⇒ 豁免的正当理由仍成立"
            "（与 D15b 是两份不同样本，两条不互相顶账）")

        # ---- D13：五维标签加粗/全角（假红）----
        expect("J5" not in codes_of(base_a.replace(
            "| ① 获取路径 | 自有埋点 | — |", "| **① 获取路径** | 自有埋点 | — |", 1), "D13a"),
            "D13：首格加粗是排版改动，内容一字未改 ⇒ 不得报「缺维」")
        expect("J5" not in codes_of(base_a.replace(
            "| ① 获取路径 | 自有埋点 | — |", "| **获取路径** | 自有埋点 | — |", 1), "D13b"),
            "D13：只加粗、不带圈号同族")

        # ---- D19：① 的枚举词必须是被**断言**的那个取值 ----
        expect("J5" in codes_of(base_a.replace(
            "| ① 获取路径 | 自有埋点 | — |",
            "| ① 获取路径 | 本企业无自有埋点，实际全靠人工估算 | 非缺口 |", 1), "D19a"),
            "D19：枚举词出现在否定句里不得算「五选一」")
        expect("J5" not in codes_of(base_a.replace(
            "| ① 获取路径 | 自有埋点 | — |", "| ① 获取路径 | 自有埋点（独立站埋点事件流） | 非缺口 |", 1),
            "D19b"), "D19 反向控制（专用）：正常写法必须仍然判绿")
        expect("J5" not in codes_of(base_a.replace(
            "| ① 获取路径 | 自有埋点 | — |",
            "| ① 获取路径 | `平台后台`（卖家后台报表）＋`自有埋点`（独立站事件流）；"
            "**下游读取事实**在本业务尚未登记 | 先用日粒度顶替 |", 1), "D19c"),
            "D19 反向控制（专用之二）：多源并列 + 尾句否定 —— 全库 14 份是这个形态，不得误杀")

        # ---- D17：J12 的四个信息源，逐源可测 + 三态可分辨 ----
        _empty = td / "empty-home"
        _empty.mkdir(exist_ok=True)
        _ghost = "p2s-this-slug-does-not-exist-anywhere"
        _real = card_sources(idx)
        expect(_real.resolve(a_card)[0] == "found",
               "D17：真实 slug 在正常环境必须 found")
        expect(_real.resolve(_ghost)[0] == "absent",
               "D17 反向控制（专用之二）：编造的 slug 必须判红（不得把「查不到」一律放行）")
        _nohome = CardSources(idx, installed_dir=_empty)
        expect(_nohome.resolve(a_card)[0] == "found",
               "D17：拿掉 ~/.dsh/skills 后真实 slug 仍须 found —— 原实现全库 125/139 份"
               "只靠这条 HOME 分支才判绿，换机/CI 会整片转红")
        expect(_nohome.resolve(_ghost)[0] == "absent",
               "D17 反向控制（专用）：拿掉 ~/.dsh/skills 后编造 slug 仍须判红")
        _blind = CardSources(idx, product_path=_empty / "nope.json", installed_dir=_empty)
        _blind.graph_available = False
        expect(_blind.resolve(_ghost)[0] == "unverifiable",
               "D17：「这个 slug 不存在」与「我没有能力查这个 slug」必须分开 —— "
               "枚举型来源不可用时判 unverifiable，**不得静默降级成「找不到」判红**")
        _inst = td / "skills" / "p2s-injected-only-here"
        _inst.mkdir(parents=True, exist_ok=True)
        (_inst / "SKILL.md").write_text("x", encoding="utf-8")
        expect(CardSources(idx, installed_dir=td / "skills",
                           product_path=_empty / "nope.json").resolve("p2s-injected-only-here")
               == ("found", "~/.dsh/skills"),
               "D17：安装目录分支必须能被**单独**测到")
        _pj = td / "cls.json"
        _pj.write_text(json.dumps({"cards": [{"slug": "p2s-injected-catalog"}]}), encoding="utf-8")
        expect(CardSources(idx, product_path=_pj, installed_dir=_empty).resolve("p2s-injected-catalog")
               == ("found", "product.classification"),
               "D17：产品侧目录分支必须能被**单独**测到")
        _pb = td / "probe-D17-run.md"
        _pb.write_text(base_a.replace(f"cards: [{a_card}]", f"cards: [{_ghost}]", 1), encoding="utf-8")
        _SOURCES[id(idx)] = _blind
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _rc17 = run([_pb], idx)
        finally:
            _SOURCES.pop(id(idx), None)
        expect(_rc17 == 2,
               f"D17：信息源不足以判定时 run() 必须 exit 2（拿不到输入 ≠ 判红 ≠ 通过），实得 {_rc17}")

        # ---- D25：已评估并**否决**的改动（登记不修）----
        expect("J13" not in codes_of(a_sec3("- 取值：按〈平台后台〉近 2 个完整年度的 P90 分位取值\n"),
                                     "D25"),
               "D25（父代理否决修改）：`P90` 是本批合法的业务取值写法，"
               "剥分位标记会误杀全批 —— 本条锁定「不修」这个决定")

        # --- 覆盖率与计数：篡改总量必须让「139」这条断言失败 ---
        expect(len(assignments) == 139, f"契约总数应为 139，实得 {len(assignments)}")
        expect(sum(1 for m in assignments.values() if m["template"] == "A") == 73, "A 应为 73")
        expect(sum(1 for m in assignments.values() if m["template"] == "B") == 66, "B 应为 66")
        # J10 变异：拿掉一份契约，总数断言必须失败（否则总数断言是摆设）
        _minus = dict(assignments)
        _minus.pop(next(iter(_minus)))
        expect(len(_minus) != 139, "变异 J10：删一份契约后总数断言必须失败")

        # --- #26：全集的定义（只收 A/ B/）与「超界/重复必须计入退出码」 ---
        import contextlib
        import io
        real = collect_files([])
        expect(len(real) == len(assignments),
               f"正式契约集合应恰为 {len(assignments)} 份，实得 {len(real)} "
               f"（>139 说明又把 A/ B/ 之外的样本算进来了）")
        expect(all(p.parent.name in ("A", "B") for p in real),
               "collect_files 只能收 contracts/A 与 contracts/B —— 收到别处说明仍在 rglob 扫全集")
        expect(not ({p.resolve() for p in real} & {p.resolve() for p in sample_files()}),
               "样本集合与正式集合不得有交集")
        # 反向控制：复制一份**已撰写**契约为第二条同名文件 ⇒ run() 必须 exit 1（而不是打一行 🔴 就绿）
        _w = next(p for p in real if BC.TODO_MARK not in p.read_text(encoding="utf-8"))
        dup = td / "CTR-A-998-复本.md"
        dup.write_text(_w.read_text(encoding="utf-8"), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as _buf:
            _rc = run(real + [dup], idx)
        expect(_rc == 1,
               "J10 变异（#26）：重复责任 + A 超界时 run() 必须 exit 1；"
               f"实得 {_rc} —— 「🔴 后面跟 ✅」是假绿，颜色不是判据")
        expect("J10" in _buf.getvalue(), "J10 的报错必须出现在输出里（否则读者看不到红因）")

        # 两条 J10 判据必须**各自**能被单独打红 —— 否则它们互相顶账（实测首版正是如此：
        # 只测「重复」时去掉「超界」照样 exit 1，只测「超界」时去掉「重复」照样 exit 1）。
        class _C:
            def __init__(self, resp, tpl):
                self.fm = {"responsibility": resp}
                self.template = tpl

        base_ok = ([_C(f"R{i}", "A") for i in range(73)] + [_C(f"S{i}", "B") for i in range(66)])
        expect(count_problems(base_ok, 73, 66) == [], "干净输入不得报 J10")
        over_only = base_ok + [_C("R-新的责任名", "A")]
        msgs = [p["msg"] for p in count_problems(over_only, 73, 66)]
        expect(any("超界" in m for m in msgs) and not any("份正式契约" in m for m in msgs),
               f"反向①：A 多一份且责任名不重复 ⇒ 只许报超界，实得 {msgs}")
        # 计数**恰好不超界**时，重复责任仍须被抓（这份样本把超界那条彻底隔离掉）
        dup_only = ([_C(f"R{i}", "A") for i in range(72)] + [_C("R0", "A")]
                    + [_C(f"S{i}", "B") for i in range(66)])
        msgs2 = [p["msg"] for p in count_problems(dup_only, 73, 66)]
        expect(any("份正式契约" in m for m in msgs2) and not any("超界" in m for m in msgs2),
               f"反向②：计数不超界时重复责任仍须单独被抓（两条判据不得互相顶账），实得 {msgs2}")
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
    # 显式 --file 给出来的路径必须存在：**输入没拿到 ≠ 通过**，也**不是判红**（0/1/2/3 四态）。
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        print(f"❌ 输入没拿到（退出码 2，不是判红）：{missing}", file=sys.stderr)
        return 2
    # ⚠️ **门禁自己崩了 ≠ 判红**（门禁缺陷 #21，2026-09-13 由 S1 撰写人实测撞出：
    #    脚本被并发改到一半时 `NameError: _dim_row`，退出码与「真的判红」同为 1，
    #    只有 stderr 的 traceback 能区分 —— 同事会把仪器故障误读成自己写错了）。
    #    本仓库对这条早有纪律（`_rel()` 与 J2 的注释都写着），这里把它落到**退出码**上：
    #    0 全过 / 1 有判据红 / 2 输入没拿到 / **3 门禁内部错误（不是判红）**。
    try:
        return run(files, idx, args.json_out)
    except Exception:                      # noqa: BLE001 —— 故意兜底，见上
        import traceback
        print("❌ 门禁内部错误（退出码 3 —— **这不是判红**，是校验器自己崩了，"
              "请把下面的 traceback 当作仪器缺陷上报）：", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
