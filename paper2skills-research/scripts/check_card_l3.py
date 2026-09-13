#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE6 S13 · 入卡弱门槛校验器（Q4 口径）+ 卡 frontmatter 的 L3 落盘。

## 这道门要回答什么

Q4 的入卡门槛取**弱**：卡的 L3 归属存在于 151 名单内，**且**该 L3 的算法可服务性为
**A 或 B**，即放行；C 类不放行。**放弃的三项**（认领制 / strong·adjacent·dropped 三档 /
逐字证据）**不在此实现** —— 它们是 Q4 明确放弃的强门槛，重新引入等于改口径。

弱≠松。弱体现在**没有额外举证义务**，不体现在「判据可以只有一个方向」：

· J1  卡 ↔ 图：每张卡的每个 L3 都在 `capability-graph.json` 的 151 名单内（不在即红）
· J2  闸门：**至少一个** L3 的 serviceability ∈ {A,B}；**一个都没有**即红，报责任名 + 责任岗位
      ⚠️ 判据写的是「至少一个 A/B」而**不是**「没有 C」—— 全 C 的卡必须红，
      而既有仓库里就有一类卡**只挂 C**（若写成「无 C 即过」，那种卡会静默全绿）。
· J3  两处逐张相等：卡 frontmatter 的 `l3_*` 与 `card-classification.json` 的 `l3[]` 逐卡相等
· J4  **同一张卡两处不同 L3**：`graph.cards[].l3` vs `card-classification.json.items[].l3`
· J5  账本自洽：图的 A/B/C 计数 + 名单 = 151、名单唯一、卡路径唯一（两处都查）
· J6  **覆盖率是一等输出且双边判**：卡 146/146、文件名 146/146，且**不得超界**
      （来由：`build_contracts.py --check` 曾打出核对率 **102.2%** 却照样通过 ——
      只写下界的覆盖率等于没有覆盖率）
· J7  现场重算 <-> 产物台账：`card-classification.json` 的 `l3_coverage` 必须等于现算值
· J8  落盘格式：写入 frontmatter 的字段名/取值形态**必须与已装线 p2s 卡同形**
      （`lib/taxonomy.js::resolveFacets` 的那 7 个字段），且 `l3_id` 由 taxonomy 现取
· J9  空 L3：`l3: []` 的卡不得被静默跳过，必须具名报出并可判红
· J10 输入没拿到 = 退出码 2（**不是通过**）：图/分类/税表读不到、卡文件一个都读不到
· J11 错位清单（在册不改正文）：`misplacement_ledger` 的每一条都能落到真实卡与真实依据

## 与产品侧的关系

字段形态**不是本脚本发明的**，是照抄 `dsh-paper2skills` 已装 1338 张卡的
`lib/taxonomy.js::resolveFacets()`：`l1_id / l1_plane / l2_id / l2_domain / l3_id /
l3_business / l3_all / l1_l2_l3`，其中 L1/L2 由**首个 L3** 唯一导出，`l3_id` 的序号
取自 `data/taxonomy.json` 的 `no`（**判据只有一处**：本脚本不自定义序号）。

用法::

    python3 check_card_l3.py --check               # 比对现状，不改任何文件（0/1/2/3）
    python3 check_card_l3.py --apply               # 把 L3 写进 146 张卡 frontmatter
    python3 check_card_l3.py --apply --dry-run     # 只说会改哪些卡
    python3 check_card_l3.py --selftest            # 11 判据 × 篡改样本（含反向控制）
    python3 check_card_l3.py --check --json-out <path>

退出码：**0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 门禁内部错误（≠ 判红）**
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# 仓库根。默认由 `__file__` 推出；**允许环境变量覆盖** —— 定向变异会把脚本副本
# 放到临时目录里跑，那时 `parents[2]` 指向临时目录而不是仓库，**每份变体都会因为
# 「图不存在」而崩溃**，而「崩溃」会被变异测试误读成「判据抓到了」——首版就是这么
# 拿到 17/17 的**假绿**。这与 check_contracts.py 缺陷 #21 同型：
# **门禁自己崩了 ≠ 判红**，而变体跑错仓库正是制造这种混淆的又一条路径。
REPO = Path(os.environ.get("P2S_L3_REPO_ROOT") or Path(__file__).resolve().parents[2])
SCRIPT = Path(__file__).resolve()
VAULT = REPO / "paper2skills-vault"
GRAPH_DEFAULT = VAULT / "07-资源库" / "capability-graph.json"
CLASSIFICATION_DEFAULT = VAULT / "07-资源库" / "card-classification.json"
PRODUCT_TAXONOMY = Path("/Users/lute/project/Magpie-Horch/packages/capabilities/"
                       "dsh-paper2skills/data/taxonomy.json")

# 弱门槛：放行集合。C 类不放行（Q2：139 = 151 − 12）。
PASS_SERVICEABILITY = ("A", "B")

# 与已装线 p2s 卡**同形**的 7 个字段（顺序即 resolveFacets 的返回顺序，勿调换）。
# ⚠️ 字段名不许自创：`dsh-paper2skills` 的 1338 张卡写的就是这 7 个，
#    另起一套（如 `l3`、`l3_primary`）会让同一张卡在两个语料里有两种写法。
L3_FIELDS = ("l1_id", "l1_plane", "l2_id", "l2_domain", "l3_id", "l3_business", "l3_all",
             "l1_l2_l3")

# 本脚本**拥有的**字段与**别人拥有的**字段，必须分开写死。
# 来由（2026-09-13 实操）：146 张卡的 frontmatter 同时有两个写入者 —— 本卡补 `l3_*`，
# S10 补 `venue_tier`。把两者混在一个常量里，就没有任何地方能回答
# 「我这次的写入有没有踩到别人的字段」。
MY_FIELDS = L3_FIELDS
FOREIGN_FIELDS = ("venue_tier",)

FRONT_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.S)
# `l3_all` 的连接符 —— 产品侧 `resolveFacets` 写的是 `' / '`，逐字节照抄。
L3_ALL_SEP = " / "
TAXONOMY_WITNESS = "dsh-paper2skills/lib/taxonomy.js::resolveFacets"


class InputMissing(Exception):
    """输入没拿到 —— 退出码 2，**不是通过**，也**不是判红**。"""


def _rel(p: Path) -> str:
    """仓库内 → 相对路径；仓库外（如 selftest 的 /tmp 副本）→ 原样返回。

    门禁缺陷 #13 的同族：`relative_to()` 对仓库外路径抛 ValueError，
    于是「判红的原因」被 traceback 吞掉，只剩一个退出码 —— 仪器故障与判红必须可分。
    """
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def load_json(p: Path, what: str) -> dict:
    p = Path(p)
    if not p.is_file():
        raise InputMissing(f"{what} 不存在：{p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InputMissing(f"{what} 不是合法 JSON：{p}（{e}）") from e


# --------------------------------------------------------------------------- #
# 事实源
# --------------------------------------------------------------------------- #
class Sources:
    """把「图 / 分类 / 税表」三份输入一次装齐，**逐项可失败**。"""

    def __init__(self, graph_path: Path, classification_path: Path, taxonomy_path: Path,
                 card_paths: list[Path] | None = None, vault_root: Path | None = None):
        self.graph_path = Path(graph_path)
        self.classification_path = Path(classification_path)
        self.taxonomy_path = Path(taxonomy_path)
        # 卡库实物根：**显式入参**，默认 vault。J6 的超界判据靠它 ——
        # 用「登记表条数」当分子是看不见「卡库里多出一张没人登记的卡」的。
        self.vault_root = Path(vault_root) if vault_root is not None else None

        g = load_json(self.graph_path, "能力图谱")
        c = load_json(self.classification_path, "卡分类落点")
        t = load_json(self.taxonomy_path, "产品侧 taxonomy")

        self.l3 = g.get("l3")
        if not isinstance(self.l3, list) or not self.l3:
            raise InputMissing(f"{self.graph_path} 没有 l3 列表")
        self.l3_by_name = {e["name"]: e for e in self.l3}
        self.plane_name = {p["id"]: p["name"] for p in g.get("planes", [])}
        self.domain_name = {d["id"]: d["name"] for d in g.get("domains", [])}
        # 岗位标题同样来自图（报红时要能点名「找谁」）；图里没有就留空，**不编**
        self.role_titles = {r["id"]: r.get("title", "") for r in g.get("roles", [])
                            if isinstance(r, dict) and "id" in r}

        # 税表：`l3_id` 的序号只此一处（判据 N2：每层判据只能有一处）
        t_l3 = t.get("l3")
        if not isinstance(t_l3, list) or not t_l3:
            raise InputMissing(f"{self.taxonomy_path} 没有 l3 列表")
        self.tax_by_name = {e["name"]: e for e in t_l3}
        self.tax_order = [e["name"] for e in t_l3]

        self.items = c.get("items")
        if not isinstance(self.items, list) or not self.items:
            raise InputMissing(f"{self.classification_path} 没有 items 列表")
        self.items_by_id = {i["id"]: i for i in self.items if isinstance(i, dict) and "id" in i}
        # 分类文件自己的台账（J7 要比对它）
        self.declared_coverage = c.get("l3_coverage", {})
        self.declared_total = c.get("total")
        self.ledger = c.get("misplacement_ledger")

        graph_cards = g.get("cards") or []
        self.graph_cards = {x["name"]: x for x in graph_cards}

        # 显式给了 --card 就只核那些（写卡时用）；否则按分类文件登记的路径收
        if card_paths:
            self.card_files = [Path(p) for p in card_paths]
        else:
            self.card_files = [REPO / i["path"] for i in self.items]
        if self.vault_root is None and self.classification_path == CLASSIFICATION_DEFAULT:
            self.vault_root = VAULT

    # ---- 派生 ----
    def serviceability(self, name: str) -> str | None:
        e = self.l3_by_name.get(name)
        return e["serviceability"] if e else None

    def responsible(self, name: str) -> dict:
        """责任名 → 责任岗位（报红时必须有它，否则读者不知道该找谁）。"""
        e = self.l3_by_name.get(name) or {}
        rid = e.get("role_id", "?")
        return {"l3": name, "role_id": rid,
                "role_title": self.role_title(rid),
                "serviceability": e.get("serviceability"),
                "plane": self.plane_name.get(e.get("plane_id", ""), ""),
                "domain": self.domain_name.get(e.get("domain_id", ""), "")}

    def role_title(self, rid: str) -> str:
        return self.role_titles.get(rid, "")

    def resolve_facets(self, l3_names: list[str]) -> dict | None:
        """**与产品侧同形**的 frontmatter 字段。

        判据只有一处：L1/L2 由**首个 L3** 导出（`resolveFacets`），`l3_id` 的序号取
        taxonomy 的 `no`。取不到就返回 None —— **不猜、不填默认值**。
        """
        if not l3_names:
            return None
        first = self.tax_by_name.get(l3_names[0])
        if first is None:
            return None
        g_first = self.l3_by_name.get(l3_names[0])
        if g_first is None:
            return None
        plane_id = g_first.get("plane_id", "")
        domain_id = g_first.get("domain_id", "")
        return {
            "l1_id": plane_id,
            "l1_plane": self.plane_name.get(plane_id, ""),
            "l2_id": domain_id,
            "l2_domain": self.domain_name.get(domain_id, ""),
            "l3_id": f"{domain_id}-{str(first['no']).zfill(3)}",
            "l3_business": first["name"],
            "l3_all": L3_ALL_SEP.join(l3_names),
            "l1_l2_l3": (f"{self.plane_name.get(plane_id, '')}/"
                         f"{self.domain_name.get(domain_id, '')}/{first['name']}"),
        }


# --------------------------------------------------------------------------- #
# frontmatter 读写
# --------------------------------------------------------------------------- #
def split_frontmatter(text: str) -> tuple[str, str] | None:
    """切成 (frontmatter 内容, 余下正文)。没有 frontmatter 返回 None。"""
    m = FRONT_RE.match(text)
    if not m:
        return None
    return m.group(1), text[m.end():]


def _l3_key(line: str) -> str | None:
    """这一行是不是**本语料的分类字段行**？是就返回字段名（只认顶格）。

    读（`read_l3_fields`）与写（`splice_l3`）**共用这一处判据** —— 各写一套必然漂移。
    两个坑都是实测撞出来的，不是预防性设计：

    1. **必须顶格**。`related:` 之类的块里会有 `  - l1_id: PLN-OPS` 这种 YAML 序列项，
       它长得像分类字段，其实是**别的字段的列表元素**。首版实测 2 张卡各有一条，
       卡被误判成「已有部分分类字段」，于是写盘器走 `rewritten` 分支而非 `inserted`。
       与漏洞 #12（frontmatter 元数据被当成事实断言）同族：**读错了对象**。

    2. **不能按 `startswith("l3_")` 收，也不能用 `str.isspace()`/`isalpha()` 判缩进**。
       本语料 8 个字段里 `l1_id / l1_plane / l2_id / l2_domain / l1_l2_l3`
       **不以 `l3_` 开头**（`l1_l2_l3` 只是以它**结尾**）。按前缀收的后果实测：
       写进去的 8 个字段里 5 个当场「不存在」⇒ 写盘器判定缺字段 ⇒
       **每次 `--apply` 都重写一遍（不幂等）**，`--check --after-apply` 永远判红。
       —— 与 CLAUDE.md 漏洞 #11「判据只认一种字段名」是**同一条**。
       缩进判据只看 ASCII 空格/制表符：绝不用 `str.isspace()`/`str.isalpha()`
       （漏洞 #18 的纪律：汉字也满足 `isalpha()`，用它会把判据原地推翻且静默）。
    """
    if line[:1] in (" ", "\t") or ":" not in line:
        return None
    k = line.split(":", 1)[0].strip()
    # 白名单**并上** `l3_` 前缀：只用前缀会漏掉本语料 5 个字段（不以 `l3_` 开头），
    # 只用白名单会漏掉自造字段（J8 就永远收不到样本）。并集才把两个方向都堵上。
    return k if (k in L3_FIELDS or k.startswith("l3_")) else None


def read_l3_fields(fm_text: str) -> dict:
    """读 frontmatter 里**本语料的分类字段**（8 个同形字段之外的 `l3_*` 也收）。

    取值只做 `.strip().strip('"').strip("'")`，**与 `gate_check.parse_frontmatter`
    和 `repo_health._fm` 同口径** —— 自造一套更聪明的解析器会让「门禁看到的」
    与「本脚本看到的」不是同一个值。

    ⚠️ 本函数只负责**取值**；「哪些行算分类字段行」由 `_l3_key` 一处说了算。
    首版把这条判据在本函数里另写了一遍（用 `startswith("l3_")`），当场自检打红：
    本语料 8 个字段里 `l1_id / l1_plane / l2_id / l2_domain / l1_l2_l3`
    **不以 `l3_` 开头**，于是「写进去的 8 个字段」里 5 个当场读不出来
    ⇒ 写盘器判定「缺字段」⇒ 每次 `--apply` 都重写（不幂等）。
    **判据只能有一处**（风险 N2）；这条与 CLAUDE.md 漏洞 #11（判据只认一种字段名）同源。
    """
    out = {}
    for line in fm_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        k = _l3_key(line)
        if k is not None:
            out[k] = line.split(":", 1)[1].strip().strip('"').strip("'")
    return out


def l3_block(facets: dict) -> list[str]:
    """生成要插入 frontmatter 的字段行（**裸值**，与全库 146 张卡的 module: 同形）。"""
    return [f"{k}: {facets[k]}" for k in L3_FIELDS]


def splice_l3(text: str, facets: dict) -> tuple[str, str]:
    """把 L3 字段写进 frontmatter，返回 (新文本, 动作)。

    三态（**互斥**，selftest 各配一份样本）：
      · `inserted` —— frontmatter 里原本一个 l3_* 字段都没有 ⇒ 在**最后一个已有字段行**
        之后整块插入。插在末尾而不是开头，是为了不扰动已有的 `title:` 首行习惯。
      · `replaced` —— 已有**全部**字段且取值一致 ⇒ 原样返回（幂等，重复 `--apply` 不漂移）。
      · `rewritten` —— 已有**部分或取值不同**的字段 ⇒ 逐行换成新块（不叠加、不留旧值）。

    ⚠️ **不动正文一个字**：只在 frontmatter 块内做行替换/插入，`text[m.end():]` 全程不变。
    """
    parts = split_frontmatter(text)
    if parts is None:
        raise ValueError("无 frontmatter")
    fm_text, body = parts
    lines = fm_text.split("\n")
    # 「什么算分类字段行」只有一处判据（`_l3_key`），读/写共用 —— 各写一套必然漂移
    keep = [(i, ln) for i, ln in enumerate(lines) if _l3_key(ln) is None]
    existing = read_l3_fields(fm_text)
    want = {k: facets[k] for k in L3_FIELDS}
    if existing == want:
        return text, "idempotent"
    if not existing:
        anchored = keep and keep[-1][0] is not None
        at = (keep[-1][0] + 1) if anchored else len(lines)
        new_lines = lines[:at] + l3_block(facets) + lines[at:]
        action = "inserted"
    else:
        # 已有部分字段：先摘掉旧行，再插回**原首行位置**（保持位置稳定）
        first_idx = min(i for i, ln in enumerate(lines) if _l3_key(ln) is not None)
        before = [ln for i, ln in keep if i < first_idx]
        after = [ln for i, ln in keep if i >= first_idx]
        new_lines = before + l3_block(facets) + after
        action = "rewritten"
    new_fm = "\n".join(new_lines)
    return f"---\n{new_fm}\n---\n{body}", action


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #
def gate_pass(serviceabilities) -> bool:
    """**弱门槛本身**：至少一个 L3 的可服务性 ∈ {A, B} 即放行（Q4）。

    抽成纯函数是为了能被单独喂样本 —— 定向变异实测：把判据写成 `!= "C"`
    （即「不在名单里 / 取值未知」的 L3 也算通过）时，**真实数据上两种写法结果完全一样**
    （146 张卡的 L3 全是 A/B），于是任何基于真实数据的断言都抓不到它。
    只有在「取值未知」这种输入上，两种写法才分得开 —— 而那种输入必须被构造出来。
    这与 check_contracts.py 把 `count_problems` 抽出来的理由同源：
    **两条判据不得互相顶账，也不得只靠真实数据碰巧覆盖。**
    """
    return any(s in PASS_SERVICEABILITY for s in serviceabilities)


def _p(code: str, card: str, msg: str, **extra) -> dict:
    return {"code": code, "card": card, "msg": msg, **extra}


# 期望的 L3 条数。写成常量而不是字面量，是为了让「151」只有一处（判据 N2）。
EXPECT_L3 = 151


def l3_count_problems(l3: list) -> list:
    """J5 的**长度判据**，抽成纯函数以便被单独喂样本。

    为什么要抽（实测教训）：原先它是 `audit()` 里的一行 `if n_l3 != 151`，
    自检样本撞上的是**同一条 J5 的另一半**（「图谱与 taxonomy 名单不一致」）
    ⇒ 定向变异把这一行整行删掉，自检**照样全绿** —— 真实数据本来就满足的断言
    等于没断言。抽成纯函数 + 直接喂长度样本，这条才真的被测到。
    （与 `check_contracts.py::count_problems` 的抽法同源：两条判据不得互相顶账。）
    """
    n = len(l3) if isinstance(l3, list) else 0
    if n != EXPECT_L3:
        return [_p("J5", "—", f"图谱 l3 名单不是 {EXPECT_L3} 条，实得 {n}")]
    return []


def audit(src: Sources, files: list[Path] | None = None,
          allow_missing_fields: bool = True) -> dict:
    """核心判定。返回一份**完整账**（覆盖率、逐卡记录、问题列表）。

    `allow_missing_fields=False` 时，frontmatter 缺 l3_* 字段**本身**就是判红 ——
    `--apply` 之后必须用这个口径复核，否则「写了字段」这件事没法被证明
    （默认 True 是为了让 `--apply` **之前**的 `--check` 能如实报出「还没写」，
    而不是把 146 张卡全判红）。
    """
    files = files if files is not None else src.card_files
    problems: list[dict] = []
    records: list[dict] = []

    # ---- J5 账本自洽（先判它：后面的覆盖率分母靠它）----
    svc = Counter(e.get("serviceability") for e in src.l3)
    n_l3 = len(src.l3)
    names = [e.get("name") for e in src.l3]
    dup_names = sorted(n for n, k in Counter(names).items() if k > 1)
    if dup_names:
        problems.append(_p("J5", "—", f"图谱 l3 名单有重复名：{dup_names}"))
    if len(src.l3_by_name) != n_l3:
        problems.append(_p("J5", "—", f"图谱 l3 名单唯一性不成立：{len(src.l3_by_name)}/{n_l3}"))
    problems.extend(l3_count_problems(src.l3))
    if svc.get("A", 0) + svc.get("B", 0) + svc.get("C", 0) != n_l3:
        problems.append(_p("J5", "—", f"A/B/C 计数之和 {dict(svc)} ≠ 名单 {n_l3}"
                                     f"（有取值不在词表内的 serviceability）"))
    # ⚠️ **比集合，不比顺序** —— 这是一条实测撞出来的**假红修正**。
    # 首版写的是 `src.tax_order != names`（逐位相等），当场打红：
    # 「图谱与 taxonomy 的 L3 名单/顺序不一致：仅图谱 []、仅税表 []（顺序差异 False）」
    # —— **两个「仅」都是空的，却报了不一致**。真因是两边的 **L3 排序不同**
    # （首个分叉在第 14 位：图谱 `抽样审计` vs 税表 `资金预测`），
    # 而 `|图谱| = |税表| = 151`、`set` 完全相等、两边都无重名。
    # 顺序**不承载本门禁用到的任何事实**：
    #   · 名字合法性 → 用 `l3_by_name` 按名查（与顺序无关）
    #   · `l3_id` 序号 → 用 taxonomy 的 `no` **按名取**（与顺序无关）
    #   而位置序号的正确性另有**更强的**判据兜着：`compare_with_installed_card()`
    #   拿一张**真实已装线 p2s 卡**逐字段比对（`--selftest` 里锁定）。
    # 判据的适用范围被默认成了全体（顺序 ≠ 身份）—— 与漏洞 #11/#14/#18 同族。
    # 顺序差异**如实报出但不判红**（登记不重判）。
    if src.tax_order:
        only_graph = [n for n in names if n not in set(src.tax_order)]
        only_tax = [n for n in src.tax_order if n not in set(names)]
        if only_graph or only_tax:
            problems.append(_p("J5", "—",
                               f"图谱与 taxonomy 的 L3 **名单集合**不一致：仅图谱 {only_graph[:5]}、"
                               f"仅税表 {only_tax[:5]}"))
        elif src.tax_order != names:
            # 找首个分叉；`next(..., None)` 而不是裸 `next()` —— 裸 next 在
            # 「长度不同但 zip 前缀全同」时会抛 StopIteration，把**打印一句提示**
            # 变成**门禁崩溃**（门禁自己崩了 ≠ 判红，见 check_contracts.py 缺陷 #21）。
            k = next((i for i, (a, b) in enumerate(zip(names, src.tax_order)) if a != b), None)
            print(f"ℹ️ L3 排序在两处不同（{len(names)} 条，首个分叉在第 "
                  f"{'—' if k is None else k + 1} 位："
                  f"{'' if k is None else f'图谱 {names[k]!r} / 税表 {src.tax_order[k]!r}'}）—— **不判红**："
                  f"本门禁只按名取序号，顺序不承载任何事实；"
                  f"位置序号的正确性由「与已装线卡逐字段比对」兜底")
    paths = [i.get("path") for i in src.items]
    dup_paths = sorted(p for p, k in Counter(paths).items() if k > 1)
    if dup_paths:
        problems.append(_p("J5", "—", f"分类文件里同一路径登记了多次：{dup_paths}"))
    ids = [i.get("id") for i in src.items]
    dup_ids = sorted(i for i, k in Counter(ids).items() if k > 1)
    if dup_ids:
        problems.append(_p("J5", "—", f"分类文件里重复卡 id：{dup_ids}"))

    # ---- J4 两处 L3 不同 ----
    graph_names = set(src.graph_cards)
    class_names = {i["id"] for i in src.items}
    if graph_names != class_names:
        problems.append(_p("J4", "—",
                           f"图的 cards 与分类文件的卡集合不一致：仅图 {sorted(graph_names - class_names)[:5]}、"
                           f"仅分类 {sorted(class_names - graph_names)[:5]}"))
    for cid in sorted(graph_names & class_names):
        a = list(src.graph_cards[cid].get("l3") or [])
        b = list(src.items_by_id[cid].get("l3") or [])
        if a != b:
            problems.append(_p("J4", cid,
                               f"**同一张卡两处不同 L3**：图 {a} ≠ 分类文件 {b}", graph=a, classification=b))

    # ---- 逐卡 ----
    n_read = 0
    n_pass = 0
    n_has_fields = 0
    n_partial = 0
    seen_files = 0
    for i in src.items:
        cid = i["id"]
        l3 = i.get("l3")
        l3_names = list(l3) if isinstance(l3, list) else []
        p = REPO / i["path"]
        rec = {"id": cid, "path": _rel(p), "l3": l3_names,
               "serviceability": [src.serviceability(n) for n in l3_names]}

        # J9：空 L3 / 非法取值都不得静默跳过。
        # ⚠️ 「不是列表」这一支是**定向变异**逼出来的：只判空列表时，把 J9 整行删掉，
        #    程序会在下游（`for n in l3_names`）崩掉 —— 退出码同样非 0，于是变异测试
        #    把「崩了」误读成「判据抓到了」。**崩掉 ≠ 判红**，判据得自己说话。
        if not isinstance(l3, list):
            problems.append(_p("J9", cid,
                               f"l3 取值不是列表（实为 {type(l3).__name__}: {l3!r}）"
                               f"—— 归属必须是一个 L3 名列表"))
            rec["verdict"] = "bad-l3-type"
            records.append(rec)
            continue
        if not l3_names:
            problems.append(_p("J9", cid, "l3 为空 —— 空归属不得静默跳过（要么补分类，要么显式登记）"))
            rec["verdict"] = "empty-l3"
            records.append(rec)
            continue

        # J1：名在 151 名单内
        unknown = [n for n in l3_names if n not in src.l3_by_name]
        for n in unknown:
            problems.append(_p("J1", cid, f"L3「{n}」不在图谱 151 名单内（不在名单内即红）"))
        rec["unknown_l3"] = unknown

        # J2：至少一个 A/B
        # 放行判定走**唯一一处**判据（纯函数），不在这里另写一遍条件
        svc_list = [(n, src.serviceability(n)) for n in l3_names]
        ok_names = [n for n, s in svc_list if s in PASS_SERVICEABILITY]
        if not gate_pass([s for _, s in svc_list]):
            resp = [src.responsible(n) for n in l3_names]
            detail = "；".join(f"{r['l3']}（{r['serviceability']}｜{r['role_id']} "
                               f"{r['role_title']}｜{r['plane']}/{r['domain']}）" for r in resp)
            problems.append(_p("J2", cid,
                               f"全部 L3 都不是 A/B 类 ⇒ 弱门槛不放行。责任名与岗位：{detail}",
                               responsibilities=resp))
            rec["verdict"] = "blocked"
        else:
            n_pass += 1
            rec["verdict"] = "pass"
            rec["passing_l3"] = ok_names

        # J3 / J8：frontmatter 与分类文件逐张相等、字段同形
        if not p.is_file():
            problems.append(_p("J3", cid, f"卡文件不存在：{_rel(p)}"))
            rec["verdict"] = "file-missing"
            records.append(rec)
            continue
        n_read += 1
        seen_files += 1
        text = p.read_text(encoding="utf-8")
        parts = split_frontmatter(text)
        if parts is None:
            problems.append(_p("J8", cid, "卡没有 frontmatter —— 无处落 L3 字段"))
            rec["frontmatter"] = "absent"
            records.append(rec)
            continue
        got = read_l3_fields(parts[0])
        rec["frontmatter_fields"] = sorted(got)
        want = src.resolve_facets(l3_names)
        if want is None:
            problems.append(_p("J8", cid, f"taxonomy 里取不到首 L3「{l3_names[0]}」的序号 ⇒ "
                                          f"无法生成 l3_id（不猜、不填默认值）"))
            records.append(rec)
            continue
        rec["expected"] = want
        rec["declared"] = got
        if not got:
            if not allow_missing_fields:
                problems.append(_p("J3", cid, "frontmatter 没有 l3_* 字段（--apply 后必须补齐）"))
            rec["frontmatter_drift"] = "absent"
            records.append(rec)
            continue
        n_has_fields += 1
        missing = [k for k in L3_FIELDS if k not in got]
        extra = [k for k in got if k not in L3_FIELDS]
        diff = {k: (got[k], want[k]) for k in L3_FIELDS if k in got and got[k] != want[k]}
        if missing or extra or diff:
            if missing:
                n_partial += 1
            if missing or diff:
                problems.append(_p("J3", cid,
                                   f"**frontmatter 与分类文件不等**：缺 {missing}、"
                                   f"取值不同 {diff}", missing=missing, diff=diff))
            if extra:
                problems.append(_p("J8", cid, f"出现非本语料字段名的 l3_* 字段：{extra}"
                                              f"（只认 {list(L3_FIELDS)}）"))
            rec["frontmatter_drift"] = "partial" if missing else "value"
        else:
            rec["frontmatter_drift"] = "none"
        records.append(rec)

    # ---- J6 覆盖率：**双边**判 ----
    n_items = len(src.items)
    n_files_exist = sum(1 for i in src.items if (REPO / i["path"]).is_file())
    cov_card = seen_files / n_items if n_items else 0.0
    cov_file = n_files_exist / n_items if n_items else 0.0
    # **分子该数什么**：分母是「卡库现在的实物」（`Skill-*.md`，剔 `_superseded`），
    # 与 `repo_health.cards()` / `gate_check.collect_cards()` / `quote_check.collect_cards()`
    # **同一口径**（三处都用 `VAULT.rglob("Skill-*.md")`）—— 用登记表的条数当分子，
    # 会把「卡库里多出一张没人登记的卡」这种超界情况整个看不见。
    # ⚠️ 这条不是洁癖：`build_contracts.py --check` 曾打出核对率 **102.2%**（142 > 139）
    #    却照样通过 —— **覆盖率只写下界等于没有覆盖率**（CLAUDE.md 已立为纪律）。
    if src.vault_root is not None:
        n_vault = sum(1 for p in src.vault_root.rglob("Skill-*.md")
                      if p.is_file() and "_superseded" not in p.parts)
    else:
        n_vault = n_files_exist
    cov_vault = n_vault / n_items if n_items else 0.0
    if n_vault > n_items:
        problems.append(_p("J6", "—",
                           f"**覆盖率超界**：卡库实物 {n_vault} 张 > 登记分母 {n_items} 张 "
                           f"= {cov_vault:.1%} —— 有 {n_vault - n_items} 张卡没进登记表却被算进分子，"
                           f"任何「全库已落 L3」的结论都不成立"))
    if seen_files > n_items or n_files_exist > n_items:
        problems.append(_p("J6", "—",
                           f"核对到的卡 {seen_files} / 物理存在 {n_files_exist} > 分母 {n_items}"))
    if cov_card < 1.0:
        problems.append(_p("J6", "—", f"覆盖率不足：核对到 {seen_files}/{n_items} = {cov_card:.1%}"
                                      f"（<100% 时拒绝给「全库已落 L3」的结论）"))

    # ---- J7 现场重算 <-> 产物台账 ----
    named = {n for i in src.items for n in (i.get("l3") or [])}
    live_named = len(named & set(src.l3_by_name))
    declared_named = src.declared_coverage.get("l3_named") if isinstance(
        src.declared_coverage, dict) else None
    if src.declared_total is not None and src.declared_total != n_items:
        problems.append(_p("J7", "—", f"分类文件自报 total={src.declared_total}，"
                                     f"实测 items={n_items}"))
    if declared_named is not None and declared_named != live_named:
        problems.append(_p("J7", "—", f"分类文件自报 l3_named={declared_named}，"
                                     f"现场重算={live_named}（产物台账与实物不符）"))

    # ---- J11 错位清单 ----
    ledger_problems = audit_ledger(src, records)
    problems.extend(ledger_problems)

    return {
        "problems": problems,
        "records": records,
        "coverage": {
            "cards_entered": seen_files, "cards_physical": n_files_exist,
            "cards_total": n_items,
            "cards_ratio": cov_card, "files_ratio": cov_file,
            "vault_physical": n_vault, "vault_ratio": cov_vault,
            "l3_named_live": live_named, "l3_named_declared": declared_named,
            "l3_total": len(src.l3_by_name),
        },
        "counts": {
            "items": n_items, "read": n_read, "pass": n_pass,
            "with_fields": n_has_fields, "partial_fields": n_partial,
            "serviceability": dict(sorted(svc.items())),
        },
    }


def audit_ledger(src: Sources, records: list[dict]) -> list[dict]:
    """J11：错位清单的每一条都必须落地 —— 卡存在、组别在词表内、**依据可复核**。

    「依据可复核」不是形容词：每条要么给 `evidence:`（`path:line` 或字段名），
    要么给 `ref:`（指向 `facet_flags` 的 id），两者至少有其一；
    且 `path:line` 必须真的指到那一行 —— **先打开文件再判它里面有没有**。
    """
    out: list[dict] = []
    if src.ledger is None:
        out.append(_p("J11", "—", "分类文件缺 misplacement_ledger 段（错位清单必须有机读落点）"))
        return out
    entries = src.ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        out.append(_p("J11", "—", "misplacement_ledger.entries 为空 —— 「没有错位」的结论"
                                 "必须先问「我的仪器能看见它吗」"))
        return out
    by_id = {i["id"]: i for i in src.items}
    rec_by_id = {r["id"]: r for r in records}
    groups = set(src.ledger.get("groups") or [])
    for e in entries:
        cid = e.get("card_id")
        if cid not in by_id:
            out.append(_p("J11", str(cid), "错位登记指向不存在的卡"))
            continue
        if e.get("group") not in groups:
            out.append(_p("J11", cid, f"group「{e.get('group')}」不在词表 {sorted(groups)} 内"))
        if not (e.get("basis") or "").strip():
            out.append(_p("J11", cid, "错位登记必须写判定依据（散文结论也要留痕）"))
        if not (e.get("evidence") or e.get("ref")):
            out.append(_p("J11", cid, "错位登记必须给 `evidence`（path:line 或字段名）"
                                      "或 `ref`（指向 facet_flags 的 id）之一 —— 否则不可复核"))
        ev = e.get("evidence") or ""
        m = re.match(r"^(?P<p>[^:]+):(?P<line>\d+)$", ev)
        if m:
            # ⚠️ 先打开文件再判它里面有没有（「某个东西不存在」先问仪器看不看得见）
            fp = VAULT.parent / m.group("p")
            if not fp.is_file():
                out.append(_p("J11", cid, f"evidence 指向的文件不存在：{m.group('p')}"))
            else:
                # ⚠️ `is_file()` 通过之后仍可能读不到（悬空软链 / 权限 / 竞态）。
                #    首版在这里裸读 ⇒ 门禁**崩成 traceback**，而变异测试把「崩了」
                #    误读成「判据抓到了」—— J11 的悬空 evidence 检查其实没被测到。
                #    **门禁自己崩了 ≠ 判红**（check_contracts.py 缺陷 #21），
                #    所以这里把读失败也收成一条判据。
                try:
                    lines = fp.read_text(encoding="utf-8").split("\n")
                # ⚠️ 顺序有意义：`UnicodeDecodeError` **必须**排在前面，
                #    这样把 read 退回裸写（不带 try）的变异会因解码失败**崩掉**，
                #    而「删掉 `is_file()` 检查」的变异仍被 OSError 接住 ——
                #    两条分支各自的变异才落在不同的可观测结果上，不会互相顶账。
                except (UnicodeDecodeError, OSError) as exc:
                    out.append(_p("J11", cid,
                                  f"evidence 指向的文件读不到：{m.group('p')}（{exc}）"))
                    lines = None
                if lines is not None:
                    ln = int(m.group("line"))
                    if not (1 <= ln <= len(lines)):
                        out.append(_p("J11", cid, f"evidence 行号 {ln} 越界（文件共 {len(lines)} 行）"))
        if e.get("path") and e["path"] != by_id[cid]["path"]:
            out.append(_p("J11", cid, f"登记的 path {e['path']} 与分类文件 {by_id[cid]['path']} 不符"))
        rec = rec_by_id.get(cid)
        if e.get("expected_l3") is not None and rec is not None and rec.get("l3") != e["expected_l3"]:
            out.append(_p("J11", cid, f"登记的 expected_l3 {e['expected_l3']} 与实测 {rec.get('l3')} 不符"))
    return out


# --------------------------------------------------------------------------- #
# --apply / --check
# --------------------------------------------------------------------------- #
def apply_all(src: Sources, dry_run: bool = False, verify_foreign: tuple[str, ...] = MY_FIELDS,
               rescan_every: int = 20, verbose: bool = True) -> dict:
    """把 L3 写进卡片 frontmatter。**只加字段，不改正文**。

    ## 与别的写入者共存（2026-09-13 实操教训，S10 同时在补 `venue_tier`）

    本机实测：146 张卡的 frontmatter 当时有**两个并发写入者**（本卡补 `l3_*`，S10 补
    `venue_tier`）。因此本函数守四条，任何一条都不能省：

    1. **逐卡 read-modify-write**：每张卡在**写入前一刻**才从磁盘读。绝不先把 146 张
       读进内存、算完再整体写回 —— 那样会把并发写入者在这段窗口里落的字段**静默抹掉**。
    2. **只增不改**：`splice_l3` 只插/换 `l3_*` 那 8 行，其余字段与正文**逐字节不动**，
       并在写盘前**自证**正文未变（变了就抛，不写）。
    3. **每 N 张重扫一次外部字段**：`verify_foreign`（默认 `venue_tier`）的计数掉下去就
       立刻报出是哪些卡 —— 丢失要**点名**，不许静默重写。
    4. **写盘后即时复验**：每写完一张，立刻重读该卡，确认 `l3_*` 在、且写前存在的外部
       字段**还在**。任一条不满足就记进 `verification_failed` 并从退出码上判红。

    参数 `verify_foreign` 收的是**别的写入者拥有的字段名**。默认值写在模块常量
    `MY_FIELDS` 里，不会与 `L3_FIELDS` 混。
    """
    changed, idem, skipped, verify_failed = [], [], [], []
    foreign_lost: list[dict] = []
    # ⚠️ 外部字段的计数**一律现扫**（`count_foreign_fields` 每张卡重读磁盘），
    #    **绝不在循环里累加一个自己维护的计数器**。首版就是这么错的：
    #    计数器被每张卡 +1（一趟下来 = 卡片数 × 命中数），与「现场重扫」的口径不同，
    #    于是每 20 张就报一次「240 → 146」的**假掉数**（实测 6 次假红）。
    #    两种口径混用 = 判据自己制造假红 —— 与漏洞 #14（判据适用范围被默认成全体）同族。
    baseline = count_foreign_fields(src, verify_foreign)

    for n, i in enumerate(src.items, 1):
        cid = i["id"]
        l3_names = list(i.get("l3") or [])
        p = REPO / i["path"]
        if not p.is_file():
            skipped.append({"id": cid, "why": "卡文件不存在"})
            continue
        facets = src.resolve_facets(l3_names)
        if facets is None:
            skipped.append({"id": cid, "why": f"取不到 facets（首 L3「{l3_names[0] if l3_names else ''}」）"})
            continue

        # ① 写入前一刻才读盘（不是循环外缓存的副本）
        text = p.read_text(encoding="utf-8")
        before_foreign = {f: has_top_field(text, f) for f in verify_foreign}
        try:
            new_text, action = splice_l3(text, facets)
        except ValueError as e:
            skipped.append({"id": cid, "why": str(e)})
            continue

        # ② 自证：正文逐字节不变（否则「只加 frontmatter」这句话就是假的）
        parts_before, parts_after = split_frontmatter(text), split_frontmatter(new_text)
        if parts_before is None or parts_after is None or parts_before[1] != parts_after[1]:
            raise RuntimeError(f"内部错误：{cid} 的正文被改动了 —— splice_l3 必须只动 frontmatter")
        # ②b 自证：外部写入者的字段在**新文本**里必须原样还在
        for f in verify_foreign:
            if before_foreign[f] and not has_top_field(new_text, f):
                raise RuntimeError(f"内部错误：{cid} 的 {f} 会在写入中丢失 —— 拒绝写盘")

        if action == "idempotent":
            idem.append(cid)
        else:
            changed.append({"id": cid, "action": action, "added": l3_block(facets)})
            if not dry_run:
                p.write_text(new_text, encoding="utf-8")

        # ④ 写盘后即时复验（dry-run 时只验内存里的新文本）
        check_text = new_text if dry_run else p.read_text(encoding="utf-8")
        fm = split_frontmatter(check_text)
        got = read_l3_fields(fm[0]) if fm else {}
        miss = [k for k in L3_FIELDS if k not in got]
        lost = [f for f in verify_foreign if before_foreign[f] and not has_top_field(check_text, f)]
        if miss or lost:
            verify_failed.append({"id": cid, "missing_l3": miss, "lost_foreign": lost})

        # ③ 每 rescan_every 张重扫一次全库外部字段；**低于基线**才叫掉数，并点名
        if verbose and rescan_every and n % rescan_every == 0:
            live = count_foreign_fields(src, verify_foreign)
            for f in verify_foreign:
                if live[f] < baseline[f]:
                    foreign_lost.append({"at": n, "field": f,
                                         "baseline": baseline[f], "after": live[f],
                                         "ids": lost_ids(src, verify_foreign, f)})
            print(f"  … 已处理 {n}/{len(src.items)} · 外部字段现算："
                  + "、".join(f"{k}={v}/{len(src.items)}" for k, v in live.items()))

    return {"changed": changed, "idempotent": idem, "skipped": skipped,
            "verify_failed": verify_failed, "foreign_baseline": baseline,
            "foreign_lost": foreign_lost, "dry_run": dry_run}


def has_top_field(text: str, name: str) -> bool:
    """卡里有没有**顶格**的 `name:` 字段（不看别人的 YAML 序列项）。"""
    parts = split_frontmatter(text)
    if parts is None:
        return False
    return re.search(rf"(?m)^{re.escape(name)}[ \t]*:", parts[0]) is not None


def count_foreign_fields(src: Sources, names) -> dict:
    """现场重扫：全库每张卡的外部字段命中数（**现算，不缓存**）。"""
    out = {}
    for name in names:
        n = 0
        for i in src.items:
            fp = REPO / i["path"]
            if fp.is_file() and has_top_field(fp.read_text(encoding="utf-8"), name):
                n += 1
        out[name] = n
    return out


def foreign_coverage(src: Sources, names) -> dict:
    """外部字段覆盖率（分子/分母都给，避免「只报百分比」）。"""
    total = len(src.items)
    counted = count_foreign_fields(src, names)
    return {f: {"hit": c, "total": total, "ratio": (c / total if total else 0.0)}
            for f, c in counted.items()}


def lost_ids(src: Sources, names, field: str) -> list[str]:
    return [i["id"] for i in src.items if not has_top_field(
        (REPO / i["path"]).read_text(encoding="utf-8"), field)]


def run_check(src: Sources, json_out: Path | None = None,
              after_apply: bool = False) -> int:
    res = audit(src, allow_missing_fields=after_apply)
    cov, cnt, probs = res["coverage"], res["counts"], res["problems"]

    print(f"卡 {cnt['items']} 张 · 核对到 {cov['cards_entered']}/{cov['cards_total']} "
          f"= {cov['cards_ratio']:.1%} · 卡库实物 {cov['vault_physical']}/{cov['cards_total']} "
          f"= {cov['vault_ratio']:.1%}")
    print(f"放行 {cnt['pass']} / 拦截 {cnt['items'] - cnt['pass']} "
          f"· frontmatter 已有字段 {cnt['with_fields']}（其中部分字段 {cnt['partial_fields']}）")
    print(f"图谱 serviceability {cnt['serviceability']} · L3 现场命名 "
          f"{cov['l3_named_live']}/{cov['l3_total']}（产物自报 {cov['l3_named_declared']}）")
    if cov["cards_ratio"] < 1.0:
        print("⚠️ 覆盖率不足 ⇒ **不给「146 张全部落 L3」的结论**（覆盖率是一等输出）")
    if cov["vault_ratio"] > 1.0:
        print(f"⚠️ 覆盖率超界 {cov['vault_ratio']:.1%} ⇒ 有卡没进登记表却被算进分子")
    if probs:
        print(f"\n🔴 {len(probs)} 条问题：")
        for p in probs[:40]:
            print(f"  [{p['code']}] {p['card']}: {p['msg']}")
        if len(probs) > 40:
            print(f"  …（另 {len(probs) - 40} 条）")
    else:
        print("✅ 判据全过")

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps({
            "_meta": {"what": "S13 入卡弱门槛校验产物（Q4）",
                      "gate": "L3 ∈ 151 且 serviceability ∈ {A,B} 即放行",
                      "graph": _rel(src.graph_path),
                      "classification": _rel(src.classification_path),
                      "taxonomy": _rel(src.taxonomy_path),
                      "field_witness": TAXONOMY_WITNESS,
                      "fields": list(L3_FIELDS)},
            "counts": cnt, "coverage": cov,
            "problems": probs,
            "cards": [{k: v for k, v in r.items() if k != "expected"} for r in res["records"]],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {json_out}")
    return 0 if not probs else 1


# --------------------------------------------------------------------------- #
# selftest：每条判据配一份**能把它打红**的篡改样本
# --------------------------------------------------------------------------- #
def selftest() -> int:
    """每条判据配一份**能把它打红**的样本；干净底必须先全绿。

    纪律（本仓库用真金白银换来的）：
      · 真实数据本来就满足的断言 = 没断言（F2 曾抓出 7 处摆设断言）；
      · 反向控制先跑：干净底不绿，下面的报红就无从归因；
      · 两个方向都要有：既要有「该红」，也要有「不该红」（否则判据退化成恒真）。
    """
    import tempfile

    cases, mutations, fails = 0, 0, []

    def expect(cond, label):
        nonlocal cases
        cases += 1
        if not cond:
            fails.append(label)
        print(f"  {'✅' if cond else '❌'} {label}")

    print("selftest（每判据一份篡改样本；真实数据本来就满足的断言等于没断言）")

    real_graph = load_json(GRAPH_DEFAULT, "能力图谱")
    real_class = load_json(CLASSIFICATION_DEFAULT, "卡分类落点")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cards_dir = td / "cards"
        cards_dir.mkdir()

        def clone(o):
            return json.loads(json.dumps(o))

        def write_src(g, c, t) -> tuple[Path, Path, Path]:
            gp, cp, tp = td / "g.json", td / "c.json", td / "t.json"
            gp.write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
            cp.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
            tp.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
            return gp, cp, tp

        # ---- 税表：序号取自真实图谱名单的顺序（**与产品侧 taxonomy 同序**）----
        tax = {"l3": [{"level": "L3", "no": n + 1, "name": e["name"],
                       "plane_id": e["plane_id"], "domain_id": e["domain_id"]}
                      for n, e in enumerate(real_graph["l3"])]}

        real_text = {i["id"]: repo_card_text(i["path"]) for i in real_class["items"]}
        # 真实底本（无 l3 字段）与真 facets 都需要一个 Sources 实例
        probe = Sources(*write_src(real_graph, real_class, tax))
        real_facets = {i["id"]: probe.resolve_facets(i["l3"]) for i in real_class["items"]}
        assert all(real_facets.values()), "基准样本取不到 facets —— 先修样本，不要改判据"

        # ⚠️ 夹具的「开工态」必须**自己造**，不能假设磁盘上的卡还没有字段 ——
        #    S13 `--apply` 落盘之后，磁盘上的 146 张卡**已经带上了** l3_* 字段，
        #    于是「摸掉字段后必须判红」这类用例的基准被悄悄换掉了：实测 2 条断言
        #    由红转绿（`--apply 后口径` 与「写盘器幂等」），**不是判据坏了，是夹具过期了**。
        #    （同族教训：真实数据本来就满足的断言等于没断言 —— 这次反过来，
        #     真实数据**已被自己改过**，于是夹具不再是它声称的那个形态。）
        def strip_l3(text: str) -> str:
            """把卡还原成「还没落 L3 字段」的开工态（只摘 frontmatter 里那 8 行）。"""
            parts = split_frontmatter(text)
            if parts is None:
                return text
            kept = [ln for ln in parts[0].split("\n") if _l3_key(ln) is None]
            return f"---\n" + "\n".join(kept) + f"\n---\n" + parts[1]

        # 夹具基准：一律以**开工态**为准（磁盘现状可能与它不同，见上）
        raw_text = {cid: strip_l3(real_text[cid]) for cid in real_text}
        assert all("l3_id" not in (split_frontmatter(x)[0] or "") for x in raw_text.values()), \
            "夹具开工态仍带 l3_ 字段 —— 先修夹具，不要改判据"

        # 干净底：146 张卡都补上 l3_* 字段（用开工态文本 splice，改的只是 frontmatter）
        clean_text = {cid: splice_l3(raw_text[cid], real_facets[cid])[0] for cid in real_text}

        def materialize(class_doc, texts) -> list[Path]:
            out = []
            for i in class_doc["items"]:
                fp = cards_dir / (i["id"] + ".md")
                fp.write_text(texts[i["id"]], encoding="utf-8")
                out.append(fp)
            return out

        def retarget(class_doc, files) -> dict:
            """把 items[].path 指到临时副本；**返回深拷贝，绝不改调用方的对象**。"""
            d = clone(class_doc)
            for i, fp in zip(d["items"], files):
                i["path"] = str(fp)
            return d

        # 干净底的分类文件（路径已指向临时卡）
        # ⚠️ 干净底必须**带一份合法的 misplacement_ledger**，否则 J11 会报「缺段」，
        #    下面每一条断言都会在同一个红因上「通过」—— 判别力就被这一个红因顶账了。
        clean_class = retarget(real_class, materialize(real_class, clean_text))
        clean_class["misplacement_ledger"] = {
            "groups": ["tech_domain_misplaced", "meta_card"],
            "entries": [{"card_id": real_class["items"][0]["id"],
                         "group": "meta_card",
                         "basis": "selftest 干净底用的占位依据",
                         "evidence": f"{real_class['items'][0]['path']}:1"}],
        }

        def variant(mut_class=None, mut_graph=None, mut_tax=None,
                    mutate_card=None, texts=None) -> dict:
            """造一份变体并跑一遍 audit。返回 audit 结果。"""
            nonlocal mutations
            mutations += 1
            tx = dict(texts if texts is not None else clean_text)
            if mutate_card is not None:
                k, fn = mutate_card
                tx[k] = fn(tx[k])
            cd = clone(mut_class if mut_class is not None else clean_class)
            files = materialize(cd, tx)
            cd = retarget(cd, files)
            g = clone(mut_graph if mut_graph is not None else real_graph)
            gp, cp, tp = write_src(g, cd, clone(mut_tax if mut_tax is not None else tax))
            sv = Sources(gp, cp, tp)
            return audit(sv)

        def codes(res):
            return {p["code"] for p in res["problems"]}

        def msgs(res, code):
            return [p["msg"] for p in res["problems"] if p["code"] == code]

        def case(label, code, **kw):
            res = variant(**kw)
            got = msgs(res, code)
            hit = bool(got)
            print(f"  {'✅' if hit else '❌'} [{code}] {label}"
                  + ("" if hit else f"  ← 实得 {sorted(codes(res)) or '全绿'}"))
            if not hit:
                fails.append(f"[{code}] 未被打红：{label}")
            nonlocal cases
            cases += 1
            return res, got

        # ---------------------------------------------------------------- #
        # 反向控制先跑：干净底必须 0 问题（否则下面的报红无从归因）
        # ---------------------------------------------------------------- #
        base = variant()
        expect(not base["problems"],
               f"反向控制：干净底必须 0 问题（实得 {base['problems'][:2]}）")
        expect(base["coverage"]["cards_ratio"] == 1.0,
               f"干净底覆盖率必须恰为 1.0（实得 {base['coverage']['cards_ratio']}）")
        expect(base["counts"]["pass"] == len(real_class["items"]),
               f"干净底必须 {len(real_class['items'])} 张全放行（实得 {base['counts']['pass']}）")

        # J2 的**真实数据控制组**（这一条是定向变异逼出来的）：
        # 把判据反转成「没有 C 就过」时，之前所有样本照样全绿 —— 因为干净底里
        # 根本没有一张「A/B 被误放行」的卡会因此**反过来**出错。
        # 生产数据现在 146/146 全过弱门槛，所以正向断言（「本该放行的确实放行了」）
        # 只能落在**真实分类文件**上：把它喂进 audit()，逐张核对裁决。
        def card_says(text, names):
            """把卡 frontmatter 的分类字段改成与 names 一致（用于隔离 J2/J1 之外的判据）。

            ⚠️ 取不到 facets（例如故意喂一个不在名单里的 L3）时**必须原样返回文本**，
            不许让基准构造器抛异常 —— 首版在这里直接 `[0]`，于是「卡挂非法 L3」这条
            变异样本把整个 selftest **打成 traceback**（退出码是 1，但一条断言摘要都打不出来），
            真实失败被仪器崩溃吞掉。这正是 `check_contracts.py` 缺陷 #21 的同型：
            **门禁自己崩了 ≠ 判红**。样本构造器也得守这条。
            """
            f = probe.resolve_facets(list(names))
            return splice_l3(text, f)[0] if f else text

        src_v = probe
        real_verdicts = audit(probe)
        cleared = {r["id"] for r in real_verdicts["records"] if r.get("verdict") == "pass"}
        expect(len(cleared) == len(real_class["items"]),
               f"J2 真实数据控制组：{len(real_class['items'])} 张精选卡的 L3 全属 A/B "
               f"⇒ 必须**全部**放行（实得 {len(cleared)}）")
        expect(not [p for p in real_verdicts["problems"] if p["code"] in ("J1", "J2")],
               f"J2 真实数据控制组：不得对任何一张卡报 J1/J2（实得 "
               f"{[p['msg'] for p in real_verdicts['problems'] if p['code'] in ('J1', 'J2')][:2]}）")
        # ⚠️ 上面两条控制组只证明「真数据都过」，**证不了判据的判准**：
        #    把判据反转成「没有 C 就过」时它们照样全绿（定向变异实测逃逸）。
        #    所以再加一条等号断言 —— 它把「放行集合」与「全部 L3 都在 A/B 的卡集合」
        #    钉在一起：判准一旦偏移（多放行 / 少放行 / 放行集合变成别的口径）立刻不等。
        svc_of = {n: src_v.serviceability(n) for n in src_v.l3_by_name}
        has_l3 = [r for r in real_verdicts["records"] if r.get("l3")]

        def _would_pass(r) -> bool:
            return all(svc_of.get(n) in PASS_SERVICEABILITY for n in r["l3"])

        # 判据写的是「**至少一个** L3 ∈ A/B」。两个方向必须各自钉住，
        # 否则判据偏一点就看不出来（两条都是定向变异实测逼出来的）：
        #   ① 含于：放行的卡必须**真的**有一个 L3 ∈ A/B  —— 挡住「放进了不该放的」；
        #      `!= "C"` 这种写法会把「L3 名不在名单里」的卡也放进 ok_names，
        #      而那种卡**不在** want_pass 里 ⇒ 这条立刻不等。
        #   ② 划分：pass / blocked / 其它 必须**恰好铺满**有 L3 的卡 —— 挡住「把该放的拦了」，
        #      也让 ① 不至于退化成一句恒真的包含关系。
        want_pass = {r["id"] for r in has_l3 if _would_pass(r)}
        cleared_all = {r["id"] for r in has_l3 if r.get("verdict") == "pass"}
        blocked = {r["id"] for r in has_l3 if r.get("verdict") == "blocked"}
        spurious = sorted(cleared_all - want_pass)
        expect(not spurious,
               f"J2 含于：放行的卡必须真的有一个 L3 ∈ A/B（越界放行 {len(spurious)} 张："
               f"{spurious[:3]}）")
        expect(cleared_all | blocked == {r['id'] for r in has_l3},
               f"J2 划分：pass ∪ blocked 必须恰好等于有 L3 的 {len(has_l3)} 张卡"
               f"（pass {len(cleared_all)} + blocked {len(blocked)} = "
               f"{len(cleared_all | blocked)}）")
        expect(not (cleared_all & blocked), "J2 划分：pass 与 blocked 不得相交")

        # 反向：把真实分类里**首 L3 全部换成 C** ⇒ 必须一张都不放行（判据的另一半）。
        # 用独立目录造，避免与 J6 的卡库实物计数互相干扰。
        all_c_first = sorted({e["name"] for e in real_graph["l3"] if e["serviceability"] == "C"})
        cd_c = clone(real_class)
        for n, it in enumerate(cd_c["items"]):
            it["l3"] = [all_c_first[n % len(all_c_first)]]
        gc = clone(real_graph)
        for cc, it in zip(gc["cards"], cd_c["items"]):
            cc["l3"] = list(it["l3"])
        c_dir = td / "vault-allc"
        (c_dir / "01-域").mkdir(parents=True)
        for it in cd_c["items"]:
            (c_dir / "01-域" / (it["id"] + ".md")).write_text(
                card_says(clean_text[it["id"]], it["l3"]), encoding="utf-8")
        gc_p, cc_p, tc_p = write_src(
            gc, retarget(cd_c, [c_dir / "01-域" / (it["id"] + ".md") for it in cd_c["items"]]), tax)
        rr_c = audit(Sources(gc_p, cc_p, tc_p, vault_root=c_dir))
        expect(rr_c["counts"]["pass"] == 0,
               f"J2 反向：全部落 C 类时放行数必须为 0（实得 {rr_c['counts']['pass']}）")

        ids = [i["id"] for i in real_class["items"]]
        c_first = ids[0]
        c_name = next(e["name"] for e in real_graph["l3"] if e["serviceability"] == "C")
        c_role = next(e["role_id"] for e in real_graph["l3"] if e["name"] == c_name)
        only_c = [e["name"] for e in real_graph["l3"] if e["serviceability"] == "C"][:2]

        def set_l3(cd, cid, names):
            for it in cd["items"]:
                if it["id"] == cid:
                    it["l3"] = list(names)
            return cd


        # ---------------------------------------------------------------- #
        # J1 名不在 151 名单内
        # ---------------------------------------------------------------- #
        res, _ = case("卡挂了一个不在 151 名单内的 L3",
                      "J1", mut_class=set_l3(clone(real_class), c_first, ["这不是一个责任名"]))

        # J2 至少一个 A/B（C 类不放行），且必须报责任名与责任岗位
        res2, m2 = case("一张卡只挂 C 类 ⇒ 弱门槛不放行",
                        "J2",
                        mut_class=set_l3(clone(real_class), c_first, [c_name]),
                        texts=dict(clean_text, **{c_first: card_says(clean_text[c_first], [c_name])}))
        expect(bool(m2) and c_name in m2[0] and c_role in m2[0],
               f"J2 必须报出责任名「{c_name}」与责任岗位「{c_role}」（实得 {m2[:1]}）")

        # J2 反向：判据写的是「至少一个 A/B」，**不是**「没有 C」。
        # 若写成后者，一张「只有 C」的卡会静默全绿 —— 这正是本仓库旧漏洞的形态。
        res3, _ = case("反向：只挂 C 的卡必须红（判据不是「无 C 即过」）",
                       "J2",
                       mut_class=set_l3(clone(real_class), c_first, only_c),
                       texts=dict(clean_text, **{c_first: card_says(clean_text[c_first], only_c)}))
        # 再加一分：A/C 混合（有 A 就该放行）—— 证明判据不误杀
        mix = next(e["name"] for e in real_graph["l3"] if e["serviceability"] == "A")
        res3b = variant(mut_class=set_l3(clone(real_class), c_first, [mix, c_name]),
                        texts=dict(clean_text, **{c_first: card_says(clean_text[c_first], [mix, c_name])}))
        expect(not [p for p in res3b["problems"] if p["code"] in ("J1", "J2")],
               f"反向：A+C 混合的卡必须放行（A 在即过），实得 "
               f"{[p['msg'] for p in res3b['problems'] if p['code'] in ('J1', 'J2')] or '无 J1/J2'}")

        # ---------------------------------------------------------------- #
        # J3 frontmatter 与分类文件逐张相等
        # ---------------------------------------------------------------- #
        case("frontmatter 的 l3_business 与分类文件不同（取值漂移）",
             "J3", mutate_card=(c_first, lambda t: re.sub(
                 r"(?m)^l3_business: .*$", "l3_business: 质量分析", t, count=1)))
        case("删掉 frontmatter 的一个分类字段（缺字段）",
             "J3", mutate_card=(c_first, lambda t: re.sub(
                 r"(?m)^l2_domain: .*\n", "", t, count=1)))
        case("l3_all 的连接符被换掉（`/` 无空格）",
             "J3", mutate_card=(c_first, lambda t: re.sub(
                 r"(?m)^l3_all: (.*)$", lambda m: "l3_all: " + m.group(1).replace(" / ", "/"), t, count=1)))

        # J3 的 --apply 后口径：缺字段本身即红（否则「写了字段」无法被证明）
        cd_no = clone(clean_class)
        files_no = materialize(cd_no, dict(clean_text, **{c_first: raw_text[c_first]}))
        sv_no = Sources(*write_src(real_graph, retarget(cd_no, files_no), tax))
        rr_no = audit(sv_no, allow_missing_fields=False)
        expect(any(p["code"] == "J3" and "没有 l3_* 字段" in p["msg"] for p in rr_no["problems"]),
               f"J3 --apply 后口径：卡缺分类字段本身即红（实得 "
               f"{[p['msg'] for p in rr_no['problems']] or '全绿'}）")
        expect(not any(p["code"] == "J3" for p in base["problems"]),
               "J3 反向：已写全字段的卡不得报 J3")

        # ---------------------------------------------------------------- #
        # J4 同一张卡两处不同 L3
        # ---------------------------------------------------------------- #
        g4 = clone(real_graph)
        g4["cards"][0]["l3"] = ["质量分析"]
        case("同一张卡被图上与分类文件赋予不同 L3", "J4", mut_graph=g4)
        expect(not msgs(base, "J4"),
               "J4 反向：两处一致时不得报（否则它是个恒真的摆设）")

        # ---------------------------------------------------------------- #
        # J5 账本自洽
        # ---------------------------------------------------------------- #
        # ⚠️ 长度判据**必须直接喂纯函数**，不能靠 `variant()` 造整图样本：
        #    「少一条」的整图样本会同时撞上同一条 J5 的**另一半**（「图谱与 taxonomy 名单不一致」）
        #    ⇒ 长度断言被它顶账，把 `if n_l3 != 151` 整行删掉自检照样全绿
        #    （首版定向变异实测：17 份里 1 份逃逸，逃的就是它）。
        #    两条判据各自独立 —— 这是在 check_contracts.py 的 J10 上学到的同一课。
        expect(l3_count_problems(real_graph["l3"]) == [],
               "J5 长度判据：151 条时不得报（反向控制）")
        expect(any("不是 151 条" in p["msg"] for p in l3_count_problems(real_graph["l3"][:-1])),
               "J5 长度判据：150 条必须报「不是 151 条」（下界方向）")
        expect(any("不是 151 条" in p["msg"]
                   for p in l3_count_problems(real_graph["l3"] + [real_graph["l3"][0]])),
               "J5 长度判据：152 条也必须报（上界方向 —— 只写一边等于没有判据）")
        expect(any("不是 151 条" in p["msg"] for p in l3_count_problems("不是列表")),
               "J5 长度判据：取值不是列表也必须报（不得崩）")

        # ⚠️ 上面四条只测纯函数；**把 audit() 里那次调用整行删掉，它们照样全绿**
        #    （定向变异实测逃逸）。所以必须再经 audit() 走一遍生产路径：
        #    造一份「名单 150 条、名字集合与税表完全相同」的图 ——
        #    既不撞「与税表不一致」，也不撞重名，只剩长度这一条可说。
        g5d = clone(real_graph)
        g5d["l3"] = [dict(g5d["l3"][0])] + g5d["l3"][1:-1]   # 152 → 150，集合不变、无重名
        r5d = variant(mut_graph=g5d)
        expect(any("不是 151 条" in m for m in msgs(r5d, "J5")),
               f"J5 长度判据经 audit()：名单 150 条必须报「不是 151 条」"
               f"（实得 {msgs(r5d, 'J5')[:2] or '全绿'}）")

        # J5 的**集合**判据（顺序不判红，只剩这一条）：
        # ⚠️ 另配一条**反向**样本 —— 「顺序不同但集合相同」**不得**报红。
        #    首版正是把顺序当身份用，于是在真实数据上打出一条**假红**
        #    （「仅图谱 []、仅税表 []」却不一致）。判据的适用范围被默认成全体，
        #    与漏洞 #11/#14/#18 同族。
        g5o = clone(real_graph)
        g5o["l3"] = list(reversed(g5o["l3"]))
        r5o = variant(mut_graph=g5o)
        expect(not msgs(r5o, "J5"),
               f"J5 反向：**顺序不同但集合相同**必须放行（实得 {msgs(r5o, 'J5')[:1] or '无 J5'}）")
        g5m = clone(real_graph)
        g5m["l3"][0] = dict(g5m["l3"][0], name="这是一个不在税表里的责任名")
        r5m = variant(mut_graph=g5m)
        expect(any("名单集合" in m for m in msgs(r5m, "J5")),
               f"J5 集合判据：名字集合不一致必须报（实得 {msgs(r5m, 'J5')[:1] or '全绿'}）")

        g5b = clone(real_graph)
        g5b["l3"][0]["serviceability"] = "D"
        case("serviceability 取值不在 A/B/C 词表内", "J5", mut_graph=g5b)

        g5c = clone(real_graph)
        g5c["l3"].append(dict(g5c["l3"][0]))
        case("图谱 l3 名单出现重名", "J5", mut_graph=g5c)

        cd5 = clone(clean_class)
        cd5["items"].append(clone(cd5["items"][0]))
        case("分类文件里重复卡 id（同一张卡登记两次）", "J5", mut_class=cd5)

        # 同一路径登记两次 —— ⚠️ **必须绕开 `variant()`**：它里面的 `retarget()` 会按顺序
        # 把每个 item 的 path 改写成它自己的临时副本，顺手把重复抹平（首版正是这样漏掉的）。
        # 这类「样本构造器自己修好了被测缺陷」的假绿，只有把样本造在它上游才看得见。
        cd5c = clone(clean_class)
        cd5c["items"][1]["path"] = cd5c["items"][0]["path"]
        gp5, cp5, tp5 = write_src(real_graph, cd5c, clone(tax))
        rr5 = audit(Sources(gp5, cp5, tp5))
        expect(any(p["code"] == "J5" and "同一路径登记了多次" in p["msg"] for p in rr5["problems"]),
               f"J5 分类文件里同一路径登记两次（实得 "
               f"{[p['msg'] for p in rr5['problems'] if p['code'] == 'J5'] or '全绿'}）")

        # ---------------------------------------------------------------- #
        # J6 覆盖率（**双边**）
        # ---------------------------------------------------------------- #
        cd6 = clone(clean_class)
        cd6["items"] = cd6["items"][:-1]
        g6 = clone(real_graph)
        g6["cards"] = g6["cards"][:-1]
        # 卡库实物仍是 146 张（`vault_root` 指向放满 146 张的目录）⇒ 分子 146 > 分母 145
        gp6, cp6, tp6 = write_src(g6, retarget(cd6, materialize(cd6, clean_text)), tax)
        rr6 = audit(Sources(gp6, cp6, tp6, vault_root=cards_dir))
        expect(any(p["code"] == "J6" and "超界" in p["msg"] for p in rr6["problems"]),
               f"J6 超界：登记分母 145 而卡库实物 146 ⇒ 分子>分母必须判红"
               f"（实得 {[p['msg'] for p in rr6['problems'] if p['code'] == 'J6'] or '全绿'}）")

        # **下界**：分子（真实存在的卡文件）少于分母 —— 删掉一个卡文件。
        # 这一条与上面那条是**两个方向**：合起来才能证明覆盖率不是单边判据。
        miss_dir = td / "vault-under"
        (miss_dir / "01-域").mkdir(parents=True)
        for i in real_class["items"]:
            (miss_dir / "01-域" / (i["id"] + ".md")).write_text(clean_text[i["id"]], encoding="utf-8")
        (miss_dir / "01-域" / (ids[-1] + ".md")).unlink()
        cd_un = clone(clean_class)
        f_un = [miss_dir / "01-域" / (i["id"] + ".md") for i in cd_un["items"]]
        gp_u, cp_u, tp_u = write_src(real_graph, retarget(cd_un, f_un), tax)
        rr_un = audit(Sources(gp_u, cp_u, tp_u, vault_root=miss_dir))
        expect(any(p["code"] == "J6" and "覆盖率不足" in p["msg"] for p in rr_un["problems"]),
               f"J6 下界：少一个卡文件 ⇒ 覆盖率 {rr_un['coverage']['cards_ratio']:.1%} 必须判红"
               f"（实得 {[p['msg'] for p in rr_un['problems'] if p['code'] == 'J6'] or '全绿'}）")
        expect(rr_un["coverage"]["cards_ratio"] < 1.0,
               "J6 下界的读数必须真的小于 1.0（否则这条断言是摆设）")

        # 超界：卡库实物多于登记分母。构造方式是**在卡目录里多放一张没人登记的卡**，
        # 而分母仍是 146 ⇒ 分子 147 > 分母 146 —— 这正是「核对率 102.2% 却通过」的形态。
        over_dir = td / "vault-over"
        (over_dir / "01-域").mkdir(parents=True)
        for i in real_class["items"]:
            (over_dir / "01-域" / (i["id"] + ".md")).write_text(clean_text[i["id"]], encoding="utf-8")
        (over_dir / "01-域" / "Skill-Unregistered-Extra.md").write_text(
            "---\ntitle: 没进登记表的一张卡\n---\n", encoding="utf-8")
        cd_over = clone(clean_class)
        f_over = [over_dir / "01-域" / (i["id"] + ".md") for i in cd_over["items"]]
        gp, cp, tp = write_src(real_graph, retarget(cd_over, f_over), tax)
        sv_over = Sources(gp, cp, tp, vault_root=over_dir)
        rr_over = audit(sv_over)
        expect(any(p["code"] == "J6" and "超界" in p["msg"] for p in rr_over["problems"]),
               f"J6 上界：卡库实物 147 > 登记分母 146 必须判红 —— 只写下界的覆盖率"
               f"等于没有覆盖率（实得 {[p['msg'] for p in rr_over['problems']] or '全绿'}）")
        expect(not msgs(base, "J6"), "J6 反向：干净底不得报覆盖率问题")

        # ---------------------------------------------------------------- #
        # J7 产物台账 <-> 现场重算
        # ---------------------------------------------------------------- #
        cd7 = clone(clean_class)
        cd7["l3_coverage"] = dict(cd7.get("l3_coverage") or {}, l3_named=999)
        case("分类文件自报 l3_named=999 与现算不符", "J7", mut_class=cd7)
        cd7b = clone(clean_class)
        cd7b["total"] = 999
        case("分类文件自报 total=999 与实测不符", "J7", mut_class=cd7b)

        # ---------------------------------------------------------------- #
        # J8 字段名与取值形态（与已装线同形）+ 不猜 l3_id
        # ---------------------------------------------------------------- #
        case("出现自造字段名 l3_primary",
             "J8", mutate_card=(c_first, lambda t: re.sub(
                 r"(?m)^l3_business: ", "l3_primary: 自造字段\nl3_business: ", t, count=1)))
        # 税表缺**本卡首 L3** 那一条 ⇒ 序号取不到 ⇒ 必须拒绝生成（不猜、不填默认值）。
        # ⚠️ 刻意只摘掉首 L3 那一条而不是整段切尾巴：切尾巴会先撞 J5 的名单比对，
        #    于是这条断言在**另一个红因**上「通过」—— 判别力被顶账了（首版正是如此）。
        tax_miss = {"l3": [e for e in tax["l3"] if e["name"] != real_class["items"][0]["l3"][0]]}
        res8, m8 = case("taxonomy 缺本卡首 L3 ⇒ 不得猜 l3_id（拒绝生成）",
                        "J8", mut_tax=tax_miss)
        expect(all("不猜" in m or "序号" in m for m in m8),
               f"J8 的报错必须说明「不猜序号」（实得 {m8[:1]}）")
        ok, detail = compare_with_installed_card(probe)
        expect(ok, f"与已装线 p2s 卡同形（字段名 + `' / '` 连接符 + l3_id 形态）：{detail}")

        # ---------------------------------------------------------------- #
        # J9 空 / 非法 L3 不得静默跳过
        # ---------------------------------------------------------------- #
        cd9 = clone(clean_class)
        g9 = clone(real_graph)
        set_l3(cd9, c_first, [])
        for cc in g9["cards"]:
            if cc["name"] == c_first:
                cc["l3"] = []
        case("空 l3 的卡必须具名报出（不得静默跳过）", "J9", mut_class=cd9, mut_graph=g9)

        # ⚠️ 第二份样本是**删掉 J9 那一行**的定向变异逼出来的：
        #    只留「空列表」样本时，删掉 J9 后程序会在 `rec["l3"]` / 循环里崩掉（退出码 1），
        #    于是变异测试把「崩了」误读成「抓到了」—— 而 J9 这条判据其实没被测到。
        #    「l3 取值不是列表」才会让**判据本身**（而不是它的下游）显形。
        cd9b = clone(clean_class)
        g9b = clone(real_graph)
        for it in cd9b["items"]:
            if it["id"] == c_first:
                it["l3"] = "实验设计"          # 字符串而不是列表
        for cc in g9b["cards"]:
            if cc["name"] == c_first:
                cc["l3"] = "实验设计"
        case("l3 取值不是列表（字符串）也必须具名报出", "J9", mut_class=cd9b, mut_graph=g9b)

        # ---------------------------------------------------------------- #
        # J10 输入没拿到 —— 必须是退出码 2，**不是通过**
        # ---------------------------------------------------------------- #
        # ⚠️ 必须**直接**断言 `load_json`，不能只经 `Sources`：经 Sources 时，
        #    把 `load_json` 的 raise 改成 return {} 之后，后面的 `g.get(...)` 会在
        #    另一个判据上崩掉 ⇒ 变异测试把「崩了」当成「抓到了」，而 J10 本身没被测到
        #    （首版变异测试实测：删掉 raise 后 selftest 照样全绿）。
        for label, fn in (("图", lambda: load_json(td / "nope.json", "能力图谱")),
                          ("税表", lambda: load_json(td / "nope-tax.json", "产品侧 taxonomy"))):
            try:
                fn()
                raised = False
            except InputMissing:
                raised = True
            expect(raised, f"J10 {label}不存在 ⇒ `load_json` 必须抛 InputMissing"
                           f"（main 转 2；「没东西可查 ≠ 查过了没问题」）")
        # 空文件（存在但没内容）同样算「没拿到」
        empty = td / "empty.json"
        empty.write_text("", encoding="utf-8")
        try:
            load_json(empty, "空产物")
            raised_empty = False
        except InputMissing:
            raised_empty = True
        expect(raised_empty, "J10 空文件也必须抛 InputMissing（存在 ≠ 有内容）")
        try:
            Sources(gp, cp, td / "nope-tax.json")
            got_missing2 = False
        except InputMissing:
            got_missing2 = True
        expect(got_missing2, "J10 端到端：税表不存在时 Sources 也必须抛 InputMissing")

        # ---------------------------------------------------------------- #
        # J11 错位清单
        # ---------------------------------------------------------------- #
        bad_ledgers = [
            ("指向不存在的卡",
             {"groups": ["tech_domain_misplaced"],
              "entries": [{"card_id": "Skill-Not-Exist", "group": "tech_domain_misplaced",
                           "basis": "x", "evidence": "x.md:1"}]}, "不存在的卡"),
            # ⚠️ 下面两份样本**各自只认自己那条消息文本**，不能只问「有没有 J11」：
            #    「文件不存在」与「文件存在但读不到」是**两条分支**，只问「有没有 J11」时，
            #    删掉前者的 `is_file()` 检查后裸读会 OSError、被后者接住 ⇒ J11 照样出红
            #    ⇒ 变异测试把「另一条分支抓到了」误当成「这条被抓到了」（实测逃逸）。
            #    **两条分支各自独立**，同 check_contracts.py J10 的教训。
            ("指向的文件不存在（应报「不存在」）",
             {"groups": ["tech_domain_misplaced"],
              "entries": [{"card_id": c_first, "group": "tech_domain_misplaced",
                           "basis": "x", "evidence": "paper2skills-vault/没有这张卡.md:1"}]},
             "文件不存在"),
            ("指向的文件存在但读不出来（非 UTF-8，应报「读不到」）",
             {"groups": ["tech_domain_misplaced"],
              "entries": [{"card_id": c_first, "group": "tech_domain_misplaced",
                           "basis": "x",
                           "evidence": "paper2skills-research/data/_s13-unreadable.md:1"}]},
             "读不到"),
            ("evidence 行号越界",
             {"groups": ["tech_domain_misplaced"],
              "entries": [{"card_id": c_first, "group": "tech_domain_misplaced",
                           "basis": "x", "evidence": f"{real_class['items'][0]['path']}:999999"}]}, None),
            ("basis 为空（不可复核）",
             {"groups": ["tech_domain_misplaced"],
              "entries": [{"card_id": c_first, "group": "tech_domain_misplaced",
                           "basis": "", "evidence": ""}]}, None),
            ("group 不在词表内",
             {"groups": ["tech_domain_misplaced"],
              "entries": [{"card_id": c_first, "group": "随便编的组",
                           "basis": "x", "evidence": f"{real_class['items'][0]['path']}:1"}]}, None),
            ("entries 为空（「没有错位」必须先问仪器看不看得见）",
             {"groups": ["tech_domain_misplaced"], "entries": []}, "entries 为空"),
        ]
        unread = REPO / "paper2skills-research/data/_s13-unreadable.md"
        unread.write_bytes(b"---\ntitle: \xff\xfe not utf-8\n---\n")
        try:
            for label, led, want in bad_ledgers:
                cd11 = clone(clean_class)
                cd11["misplacement_ledger"] = led
                res11 = variant(mut_class=cd11)
                hit11 = [m for m in msgs(res11, "J11") if want is None or want in m]
                print(f"  {'✅' if hit11 else '❌'} [J11] 错位登记{label}"
                      + ("" if hit11 else f"  ← 期望含「{want}」，实得 "
                         f"{msgs(res11, 'J11')[:2] or sorted(codes(res11)) or '全绿'}"))
                if hit11:
                    cases += 1
                else:
                    cases += 1
                    fails.append(f"[J11] 未被打红：错位登记{label}")
        finally:
            unread.unlink(missing_ok=True)

        cd11n = clone(clean_class)
        cd11n.pop("misplacement_ledger", None)
        case("分类文件缺 misplacement_ledger 段", "J11", mut_class=cd11n)

        # ---------------------------------------------------------------- #
        # 写盘器的三态 + 「只加字段不改正文」的仪器自证
        # ---------------------------------------------------------------- #
        t0 = raw_text[c_first]
        f0 = real_facets[c_first]
        once, act1 = splice_l3(t0, f0)
        twice, act2 = splice_l3(once, f0)
        expect(act1 in ("inserted", "rewritten") and act2 == "idempotent" and once == twice,
               f"写盘器幂等：第二次 --apply 必须原样返回（实得 {act1} → {act2}）")
        expect(split_frontmatter(once)[1] == split_frontmatter(t0)[1],
               "写盘器不得改动正文一个字（只在 frontmatter 块内插/换行）")
        # 只认**顶格且在 L3_FIELDS 内**的字段（`l3_` 前缀下还可能有别的键，不算数）
        got_w = read_l3_fields(split_frontmatter(once)[0])
        expect([k for k in L3_FIELDS if k not in got_w] == [],
               f"写盘器必须写全 {len(L3_FIELDS)} 个字段（实得缺 "
               f"{[k for k in L3_FIELDS if k not in got_w]}）")
        expect(all(re.search(rf"(?m)^{k}: ", once.split("---")[1]) for k in L3_FIELDS),
               "写进去的字段名必须**顶格**逐个出现在 frontmatter 块内")

        # 部分字段 ⇒ rewritten（不叠加、不留旧值）
        partial = re.sub(r"(?m)^l2_domain: .*\n", "", once, count=1)
        fixed, act3 = splice_l3(partial, f0)
        expect(act3 == "rewritten"
               and [k for k in L3_FIELDS if k not in read_l3_fields(split_frontmatter(fixed)[0])] == []
               and len(re.findall(r"(?m)^l1_id: ", fixed)) == 1,
               f"已有部分字段 ⇒ rewritten 且不叠加（实得 {act3}，l1_id 顶格出现 "
               f"{len(re.findall(r'(?m)^l1_id: ', fixed))} 次）")

        # 无 frontmatter ⇒ 拒绝（不猜、不硬插）
        try:
            splice_l3("# 无 frontmatter 的卡\n", f0)
            refused = False
        except ValueError:
            refused = True
        expect(refused, "写盘器对无 frontmatter 的卡必须拒绝，而不是硬插")

        # ---------------------------------------------------------------- #
        # **端到端 CLI**：子进程跑真 CLI + 夹具（不是库函数级）
        # ---------------------------------------------------------------- #
        # 来由（台账 #25）：判据长在 `main()` 里、而 selftest 只测库函数时，
        # 把 main() 里的守卫改成 `if False:` 照样全绿 —— 退出码契约（0/1/2/3）
        # 必须由**子进程真的跑一遍**来锁。所以下面每条都断言**退出码**，
        # 而不是断言「某个函数返回了什么」。
        def cli(*argv, cwd=REPO) -> tuple[int, str]:
            proc = subprocess.run([sys.executable, str(SCRIPT), *argv],
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", cwd=str(cwd), timeout=300)
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

        # ⚠️ 端到端夹具必须带一份**合法 ledger**：真实 `card-classification.json` 在
        #    S13 落库前还没有 `misplacement_ledger` 段，直接拿它当夹具 ⇒ 每条端到端用例
        #    都会因为 J11 判红 —— 而那是夹具的问题，不是被测行为的问题
        #    （「红因来自夹具」会让整套端到端断言的判别力互相顶账）。
        e2e_class = clone(clean_class)
        e2e_gp, e2e_cp, e2e_tp = write_src(real_graph, e2e_class, tax)
        base_argv = ["--graph", str(e2e_gp), "--classification", str(e2e_cp),
                     "--taxonomy", str(e2e_tp)]

        rc, out = cli(*base_argv, "--check", "--after-apply")
        expect(rc == 0, f"[端到端] 干净夹具 --check --after-apply 必须 exit 0（实得 {rc}）："
                        f"{out.strip().splitlines()[-1][:110] if out.strip() else '无输出'}")

        # 开工资态（146 张都没字段）：默认口径下不得判红 —— 这是「还没做」不是「做错了」
        rc, out = cli(*base_argv, "--check")
        expect(rc == 0, f"[端到端] 无字段的开工态 --check 必须 exit 0（实得 {rc}）")

        # 判红：把一张卡的字段改错
        mut_dir = td / "cli-判红"
        mut_dir.mkdir()
        for i in clean_class["items"]:
            (mut_dir / (i["id"] + ".md")).write_text(clean_text[i["id"]], encoding="utf-8")
        bad_id = clean_class["items"][0]["id"]
        (mut_dir / (bad_id + ".md")).write_text(
            re.sub(r"(?m)^l3_business: .*$", "l3_business: 质量分析",
                   clean_text[bad_id], count=1), encoding="utf-8")
        gp2, cp2, tp2 = write_src(
            real_graph, retarget(clean_class, [mut_dir / (i["id"] + ".md")
                                              for i in clean_class["items"]]), tax)
        rc, out = cli("--graph", str(gp2), "--classification", str(cp2),
                      "--taxonomy", str(tp2), "--check", "--after-apply")
        expect(rc == 1, f"[端到端] 字段取值漂移 ⇒ 必须 exit 1（实得 {rc}）")

        # 退出码 2：输入没拿到 —— **不是通过**
        rc, out = cli("--graph", str(td / "没有这个文件.json"),
                      "--classification", str(e2e_cp), "--taxonomy", str(e2e_tp), "--check")
        expect(rc == 2, f"[端到端] 图文件不存在 ⇒ 必须 exit 2（≠ 通过，实得 {rc}）")

        # 空卡库 ⇒ 也是 2（「0 张全过」是最危险的假绿）
        empty_class = clone(clean_class)
        empty_class["items"] = []
        ec = td / "empty-class.json"
        ec.write_text(json.dumps(empty_class, ensure_ascii=False), encoding="utf-8")
        rc, out = cli("--graph", str(e2e_gp), "--classification", str(ec),
                      "--taxonomy", str(e2e_tp), "--check")
        expect(rc == 2, f"[端到端] 卡库为空 ⇒ 必须 exit 2，不得打印「0 张全过」（实得 {rc}）")

        # --apply 真写盘 + 幂等 + 不踩外部字段 + 写后复验
        ap_dir = td / "cli-apply"
        ap_dir.mkdir()
        for i in real_class["items"]:
            body = clean_text[i["id"]]
            # 先塞一个**外部写入者**的字段（模拟 S10 的 venue_tier）
            body = re.sub(r"(?m)^module: ", "venue_tier: CCF-B\nmodule: ", body, count=1)
            (ap_dir / (i["id"] + ".md")).write_text(body, encoding="utf-8")
        ap_class = clone(real_class)
        ap_class["misplacement_ledger"] = e2e_class["misplacement_ledger"]
        gp3, cp3, tp3 = write_src(
            real_graph, retarget(ap_class, [ap_dir / (i["id"] + ".md")
                                            for i in real_class["items"]]), tax)
        rc, out = cli("--graph", str(gp3), "--classification", str(cp3),
                      "--taxonomy", str(tp3), "--apply",
                      "--verify-foreign", "venue_tier")
        expect(rc == 0, f"[端到端] --apply 后必须 exit 0（实得 {rc}）：{out[-200:]}")
        expect("venue_tier：146/146" in out,
               f"[端到端] --apply 后必须打出现算的 venue_tier 覆盖率 146/146"
               f"（这是「没抹掉别人字段」的物证）："
               f"{[l for l in out.splitlines() if 'venue_tier' in l][:1]}")
        wrote = [(ap_dir / (i["id"] + ".md")).read_text(encoding="utf-8")
                 for i in real_class["items"]]
        expect(all(re.search(rf"(?m)^{k}: " if False else rf"(?m)^{k}[ \t]*:", w)
                   for k in L3_FIELDS for w in wrote),
               "[端到端] --apply 后每张卡都必须带全 8 个字段")
        expect(all(re.search(r"(?m)^venue_tier[ 	]*:", w) for w in wrote),
               "[端到端] --apply 必须保留每张卡原有的 venue_tier")
        rc2, out2 = cli("--graph", str(gp3), "--classification", str(cp3),
                        "--taxonomy", str(tp3), "--apply",
                        "--verify-foreign", "venue_tier")
        expect(rc2 == 0 and "已写入 0 张" in out2,
               f"[端到端] 第二次 --apply 必须幂等（已写入 0 张，实得 exit {rc2}）："
               f"{[l for l in out2.splitlines() if '写入' in l][:1]}")
        # 幂等之后再验：值没变
        expect([(ap_dir / (i["id"] + ".md")).read_text(encoding="utf-8")
                for i in real_class["items"]] == wrote,
               "[端到端] 第二次 --apply 不得改动任何字节")

        # 外部字段真被抹掉时必须判红（**这条判据自身要能失败**）
        # 构造：把 `has_top_field` 之外的路走一遍 —— 直接给一份「写前有、写后没了」的
        # 场景不容易造，故用**反向探针**：喂一个不存在的外部字段名，覆盖率应为 0/146，
        # 且这不影响退出码（否则「没这个字段」会被误读成「字段丢了」）。
        rc, out = cli("--graph", str(gp3), "--classification", str(cp3),
                      "--taxonomy", str(tp3), "--apply",
                      "--verify-foreign", "真的不存在的字段名")
        expect(rc == 0 and "真的不存在的字段名：0/146" in out,
               f"[端到端] 反向探针：外部字段不存在 ⇒ 覆盖率 0/146 且**不**判红"
               f"（实得 exit {rc}）：{[l for l in out.splitlines() if '真的不存在' in l][:1]}")

        # ---------------------------------------------------------------- #
        # 仪器自证：scripts/ 下不得留任何变异样本
        # ---------------------------------------------------------------- #
        left = sorted((SCRIPT.parent).glob("_s13_mutant_*"))
        expect(not left, f"仪器自证：scripts/ 下不得留下变异样本（实得 {[p.name for p in left]}）")

    print(f"\nselftest：{cases - len(fails)}/{cases} 断言通过（含 {mutations} 份变异样本）")
    for f in fails:
        print(f"🔴 {f}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------- #
# 定向变异：逐条打残判据的**实现**，selftest 必须变红
# --------------------------------------------------------------------------- #
# 纪律来源（本仓库已付过学费）：**真实数据本来就满足的断言等于没断言**。
# `--selftest` 全绿只说明「样本被前面的用例抓到了」，**不说明每条判据真的在判** ——
# 若某条判据的实现被整行删掉而 selftest 照样全绿，那条就是摆设（F2 曾一次抓出 7 处）。
# 本模式就是那个「整行删掉」的动作，而且它**写盘**（子进程得从文件跑），
# 所以变体一律落在 `tempfile.mkdtemp()` 里，跑完即删 ——
# `--selftest` 末尾另有一条仪器自证：`scripts/` 下不得留下任何 `_s13_mutant_*`。
MUTANTS = [
    ("J1", "不再检查 L3 名是否在 151 名单内",
     r"unknown = \[n for n in l3_names if n not in src\.l3_by_name\]", "unknown = []"),
    ("J2", "弱门槛判据反转成「没有 C 就过」（即不在名单/取值未知也算通过）",
     r"return any\(s in PASS_SERVICEABILITY for s in serviceabilities\)",
     'return any(s != "C" for s in serviceabilities)'),
    ("J2b", "弱门槛彻底去掉（每张卡都放行）",
     r"return any\(s in PASS_SERVICEABILITY for s in serviceabilities\)", "return True"),
    ("J3", "不再比对 frontmatter 与分类文件的取值差",
     r"diff = \{k: \(got\[k\], want\[k\]\) for k in L3_FIELDS if k in got and got\[k\] != want\[k\]\}",
     "diff = {}"),
    ("J3b", "不再检查缺字段",
     r"missing = \[k for k in L3_FIELDS if k not in got\]", "missing = []"),
    ("J4", "不再比对图与分类文件的 L3", r"if a != b:", "if False:"),
    ("J5", "不再校验 A/B/C 计数之和 = 名单条数",
     r'if svc\.get\("A", 0\) \+ svc\.get\("B", 0\) \+ svc\.get\("C", 0\) != n_l3:', "if False:"),
    ("J5b", "不再校验名单是否恰为 151 条",
     r"problems\.extend\(l3_count_problems\(src\.l3\)\)", "pass  # 打残：长度判据整行删掉"),
    ("J5c", "长度判据的上界失效（152 条也放行）—— 只写单边等于没有判据",
     r"    if n != EXPECT_L3:", "    if n < EXPECT_L3:"),
    ("J6", "覆盖率下界失效（<100% 也不报）", r"if cov_card < 1\.0:", "if False:"),
    ("J6b", "覆盖率上界失效（超界也不报）—— 这正是 102.2% 假绿的形态",
     r"if n_vault > n_items:", "if False:"),
    ("J7", "不再比对产物台账与现场重算",
     r"if declared_named is not None and declared_named != live_named:", "if False:"),
    ("J8", "不再检查自造字段名", r"extra = \[k for k in got if k not in L3_FIELDS\]", "extra = []"),
    ("J8b", "税表取不到序号时静默填默认值（而不是拒绝生成）",
     r"        if first is None:\n            return None",
     '        if first is None:\n            return {"l1_id": "", "l1_plane": "", "l2_id": "",'
     ' "l2_domain": "", "l3_id": "DOM-00-000", "l3_business": "", "l3_all": "", "l1_l2_l3": ""}'),
    ("J9", "空 / 非法 L3 静默跳过（不报红）",
     r'problems\.append\(_p\("J9", cid, "l3 为空 —— 空归属不得静默跳过（要么补分类，要么显式登记）"\)\)',
     'pass'),
    ("J10", "输入没拿到时不抛异常（退化成「通过」）",
     r'    if not p\.is_file\(\):\n        raise InputMissing\(f"\{what\} 不存在：\{p\}"\)',
     '    if not p.is_file():\n        return {}'),
    ("J11", "错位清单不再校验（整段跳过）",
     r"    ledger_problems = audit_ledger\(src, records\)", "    ledger_problems = []"),
    ("J11b", "错位清单的悬空 evidence 不再打开文件核实",
     r"            if not fp\.is_file\(\):", "            if False:"),
    ("J11c", "evidence 读不到时不再收成判据（裸读 ⇒ 门禁崩掉而不是判红）",
     r"except \(UnicodeDecodeError, OSError\) as exc:", "except ZeroDivisionError as exc:"),
]


def mutation_selftest() -> int:
    """把每条判据的实现打残一次，selftest 必须变红；否则那条判据是摆设。

    变体写在 `tempfile.mkdtemp()` 里，**跑完即删**；本模式前后都自证
    `scripts/` 下没有残留 —— 「不许留垃圾」也得是一条能失败的判据，不能只是约定。
    """
    import tempfile

    src_text = SCRIPT.read_text(encoding="utf-8")
    print("定向变异：逐条打残判据实现，selftest 必须变红（否则那条判据是摆设）")
    if sorted(SCRIPT.parent.glob("_s13_mutant_*")):
        print("❌ 开工前 scripts/ 下就有残留变异样本 —— 先清理", file=sys.stderr)
        return 3
    ok = bad = crashed_n = 0
    try:
        with tempfile.TemporaryDirectory(prefix="s13-mut-") as td:
            td = Path(td)
            # ⚠️ 变体必须放在**与真脚本同深度**的目录里：脚本用
            #    `Path(__file__).resolve().parents[2]` 推仓库根，放浅一层就会推到
            #    `/private/var/folders/...`，于是**每份变体都因「图不存在」而崩溃**，
            #    首版就是这么拿到 17/17「全被抓」的**假绿**（崩溃被误读成抓到）。
            fake = td / "paper2skills-research" / "scripts"
            fake.mkdir(parents=True)
            for code, label, pat, repl in MUTANTS:
                new_text, n = re.subn(pat, repl.replace("\\", "\\\\"), src_text, count=1)
                if n != 1:
                    print(f"  ❌ [{code}] 打残失败：正则没命中（{label}）—— 样本本身失效")
                    bad += 1
                    continue
                mp = fake / f"_s13_mutant_{code}.py"
                mp.write_text(new_text, encoding="utf-8")
                env = dict(os.environ, P2S_L3_REPO_ROOT=str(REPO))
                proc = subprocess.run([sys.executable, str(mp), "--selftest"],
                                      capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", cwd=str(REPO), env=env, timeout=1800)
                out = (proc.stdout or "") + (proc.stderr or "")
                fails = re.findall(r"^\U0001f534 (.+)$", out, re.M)
                crashed = "Traceback (most recent call last)" in out and not fails
                caught = proc.returncode != 0
                note = (f"{len(fails)} 条断言落空" if fails
                        else ("**崩成 traceback**" if crashed else "摘要未打印"))
                head = ("；".join(fails[:2]) if fails
                        else (out.strip().splitlines()[-1][:110] if out.strip() else "无输出"))
                print(f"  {'✅' if caught else '❌'} [{code}] {label}"
                      + (f"  → selftest 变红（{note}，退出码 {proc.returncode}）：{head}" if caught
                         else "  → **selftest 照样全绿：这条判据没有被测到**"))
                if caught and not crashed:
                    ok += 1
                elif crashed:
                    crashed_n += 1
                else:
                    bad += 1
    except Exception:                           # noqa: BLE001
        import traceback
        print("❌ 门禁内部错误（退出码 3 —— **这不是判红**）：", file=sys.stderr)
        traceback.print_exc()
        return 3
    # 「不许留垃圾」也得是一条**能失败**的判据，不能只是约定
    left = sorted(SCRIPT.parent.glob("_s13_mutant_*"))
    if left:
        print(f"❌ 残留变异样本未清理：{[p.name for p in left]}", file=sys.stderr)
        return 3
    print(f"\n定向变异：{ok}/{ok + bad} 由**断言落空**抓到（判据自身被测到）"
          + (f"；另 {crashed_n} 份只让门禁崩掉（算抓到，不计入判据覆盖率）" if crashed_n else "")
          + "；scripts/ 下残留 0 份")
    return 0 if bad == 0 else 1


def repo_card_text(rel_path: str) -> str:
    return (REPO / rel_path).read_text(encoding="utf-8")


def compare_with_installed_card(src: Sources) -> tuple[bool, str]:
    """把 `resolve_facets` 的输出与**真实已装线 p2s 卡**的 frontmatter 逐字节对形状。

    这条是「不要在 vault 侧自创一套字段」的**机器证明**：
    产品侧 1338 张卡里随便取一张真实存在的，它的 `l3_id` 必须命中
    `^DOM-\\d{2}-\\d{3}$`、`l3_all` 必须用 `' / '` 连接、字段名必须完全一致。
    取不到真实卡时返回 False 并说清为什么 —— **不静默跳过**。
    """
    staged = Path("/Users/lute/project/Magpie-Horch/packages/capabilities/"
                  "dsh-paper2skills/staging")
    if not staged.is_dir():
        return False, f"找不到已装线卡目录 {staged}（无法比对 ⇒ 不是「同形」，是没测）"
    sample = next(iter(sorted(staged.rglob("SKILL.md"))), None)
    if sample is None:
        return False, f"{staged} 下没有 SKILL.md"
    fm = split_frontmatter(sample.read_text(encoding="utf-8"))
    if fm is None:
        return False, f"样本卡无 frontmatter：{sample}"
    got = read_l3_fields(fm[0])
    missing = [k for k in L3_FIELDS if k not in got]
    if missing:
        return False, f"样本卡缺字段 {missing}（本语料的字段名与它不同形）"
    inst = dict(got)
    # 拿样本卡的 l3_all 反推一份 facets，逐字段比形状
    names = [n.strip() for n in inst["l3_all"].split("/")]
    mine = src.resolve_facets(names)
    if mine is None:
        return False, f"样本卡的 l3_all「{inst['l3_all']}」在本仓库税表里取不到"
    bad = {k: (mine[k], inst[k]) for k in L3_FIELDS if mine[k] != inst[k]}
    if bad:
        return False, f"逐字段比对不一致：{bad}"
    if not re.fullmatch(r"DOM-\d{2}-\d{3}", inst["l3_id"]):
        return False, f"l3_id 形态「{inst['l3_id']}」不符合 DOM-\\d{{2}}-\\d{{3}}"
    return True, f"{_rel(sample)} → {inst['l3_id']} · {inst['l3_all']}"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="S13 入卡弱门槛校验器（Q4）")
    ap.add_argument("--graph", type=Path, default=GRAPH_DEFAULT)
    ap.add_argument("--classification", type=Path, default=CLASSIFICATION_DEFAULT)
    ap.add_argument("--taxonomy", type=Path, default=PRODUCT_TAXONOMY)
    ap.add_argument("--card", action="append", default=[], help="只核这些卡（写卡时用）")
    ap.add_argument("--check", action="store_true", help="比对现状，不改任何文件")
    ap.add_argument("--apply", action="store_true", help="把 L3 写进卡片 frontmatter")
    ap.add_argument("--dry-run", action="store_true", help="与 --apply 同，但不落盘")
    ap.add_argument("--after-apply", action="store_true",
                    help="按「字段必须已在」的口径复核（--apply 之后用它）")
    ap.add_argument("--json-out", type=Path, help="门禁结果另存 JSON")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-foreign", action="append", default=list(FOREIGN_FIELDS),
                    help="别的写入者拥有的字段名（写入前后必须原样还在）；可重复")
    ap.add_argument("--mutation-test", action="store_true",
                    help="定向变异：逐条打残判据实现，selftest 必须变红（临时目录跑完即删）")
    args = ap.parse_args()

    if args.mutation_test:
        return mutation_selftest()

    if args.selftest:
        return selftest()

    try:
        src = Sources(args.graph, args.classification, args.taxonomy,
                      [Path(p) for p in args.card] or None)
        # 分类文件登记的卡，一个都读不到 ⇒ 输入没拿到（≠ 通过，≠ 判红）
        if not src.card_files:
            raise InputMissing("卡文件列表为空（分类文件登记 0 条，或用 --card 指定）")
        existing = [p for p in src.card_files if p.is_file()]
        if not existing:
            raise InputMissing(
                f"{len(src.card_files)} 个卡路径一个都读不到，例如 {src.card_files[0]}")
    except InputMissing as e:
        print(f"❌ 输入没拿到（退出码 2，**这不是通过**）：{e}", file=sys.stderr)
        return 2

    if args.apply or args.dry_run:
        try:
            res = apply_all(src, dry_run=args.dry_run,
                            verify_foreign=tuple(args.verify_foreign))
        except Exception:                       # noqa: BLE001 —— 兜底见下
            import traceback
            print("❌ 门禁内部错误（退出码 3 —— **这不是判红**）：", file=sys.stderr)
            traceback.print_exc()
            return 3
        verb = "将写入" if args.dry_run else "已写入"
        print(f"{verb} {len(res['changed'])} 张 · 幂等跳过 {len(res['idempotent'])} 张 "
              f"· 跳过 {len(res['skipped'])} 张")
        for c in res["changed"][:5]:
            print(f"  · {c['id']} [{c['action']}] +{len(c['added'])} 字段：{c['added'][:3]}…")
        for s_ in res["skipped"][:10]:
            print(f"  ⚠️ 跳过 {s_['id']}：{s_['why']}")
        # 外部写入者的字段：**现算**覆盖率（这是「没有抹掉别人字段」的物证）
        cov_f = foreign_coverage(src, tuple(args.verify_foreign))
        for f, c in cov_f.items():
            print(f"  外部字段 {f}：{c['hit']}/{c['total']} = {c['ratio']:.1%}（现算）")
        bad = False
        if res["verify_failed"]:
            bad = True
            print(f"\n🔴 写盘后复验失败 {len(res['verify_failed'])} 张：")
            for v in res["verify_failed"][:20]:
                print(f"  · {v['id']}：缺 l3 {v['missing_l3']} · 丢失外部字段 {v['lost_foreign']}")
        if res["foreign_lost"]:
            bad = True
            print(f"\n🔴 重扫发现外部字段掉数 {len(res['foreign_lost'])} 次：")
            for v in res["foreign_lost"][:10]:
                print(f"  · 第 {v['at']} 张时 {v['field']}：基线 {v['baseline']} → "
                      f"现扫 {v['after']}，涉及 {v['ids'][:5]}")
        if bad:
            return 1
        if args.dry_run:
            return 0
        # 落盘后**立刻**用「字段必须已在」的口径复核（不靠「我以为写好了」）
        try:
            return run_check(src, args.json_out, after_apply=True)
        except Exception:                       # noqa: BLE001
            import traceback
            print("❌ 门禁内部错误（退出码 3 —— **这不是判红**）：", file=sys.stderr)
            traceback.print_exc()
            return 3

    if not args.check:
        ap.print_help()
        return 0
    try:
        return run_check(src, args.json_out, after_apply=args.after_apply)
    except Exception:                           # noqa: BLE001 —— 故意兜底，见下
        import traceback
        print("❌ 门禁内部错误（退出码 3 —— **这不是判红**，是校验器自己崩了，"
              "请把下面的 traceback 当作仪器缺陷上报）：", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
