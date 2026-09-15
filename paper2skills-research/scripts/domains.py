#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""domains.py — **业务域名的唯一事实源**（PHASE6 P1）

## 为什么需要这个文件

域名（`09-DataAgent-LLM` 这样的中文串）在本次修复前**至少 7 处各写一份**，
而且**互不相同**：

| 落点 | 用的名字 |
|---|---|
| `arxiv_harvest.QUERY_GROUPS` | 旧名（`07-VOC舆情` / `09-DataAgent` / `15-营销投放`） |
| `candidate_filter.DOMAIN_DIR` | 新名（`07-NLP-VOC` / `09-DataAgent-LLM` / `15-营销投放分析`） |
| `build_registry` / `rank_candidates` / `make_bundles` | 旧名 |
| `build_venue_tiers.DOMAIN_LINE` | 新名 |
| `build_search_queries.harvest_domain_of` | 一张**手写的桥接表**（第三份映射） |

后果不是「看起来不整齐」。后果是 **`candidate_filter.filter_pool` 里那一行
`if not domains: keep`** —— 标签认不出来的论文被**整篇放行**，
该域的负向词与约束词**一条都不生效**，而且**没有任何读数**。
实测：1046 篇候选池里 **141 篇**（89+37+9+6）落在这一支上，
其中 `00-电商Agent` 37 篇是因为 filter 里**压根没有这个域**，
另外 104 篇是因为三个域**被改了名而 filter 只认新名**。

> 这是本仓库那条铁律的又一例：**判「某个东西不存在」之前，先问仪器能不能看见它。**
> 差别在于这次「仪器」是过滤器自己，而它**静默地**看不见。

## canonical 以什么为准

**vault 顶层目录名**。理由：那是实物 —— 卡住在那里，`capability-graph.json`
的域、`card-classification.json` 的域都据它命名。`--check` 会拿实际目录
**双向**对账（注册了却没有目录 ⇒ 红；有目录却没注册 ⇒ 也红）。

## 别名表是**迁移垫片**，不是长期契约

**曾经的做法是加一张别名表把旧名静默折到 canonical**，迁移做完后立刻发现它错：
别名表一归零，**下一次漂移就再也不会被发现**（新写的旧名会被悄悄折掉）。
⇒ `RETIRED` 现在**不参与解析**，只用来把报错说得可行动；
旧名走 `unknown` 分支，让过滤器 `exit 2` 大喊「仪器瞎了」。
`--check` 的判据是**盘上不许出现退休名**（出现 ⇒ 判红），不是「别名用量归零」。

用法：
    python3 domains.py --check       # 与 vault 目录双向对账 + 别名使用计数
    python3 domains.py --selftest    # 证明每一条判据都会失败
    python3 domains.py --mutate      # 把本文件的判据改坏，验证 selftest 抓得住
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ⚠️ 仓库根**可显式指定**（`P2S_REPO`），默认仍是从本文件往上三级。
# 加这个口子不是为测试开的后门，而是两个真实需求的同一个解：
#   ① 变异实验要把**改坏的 domains.py 放到别处跑**（放仓库里会留残骸；
#      放 /tmp 又会因 `parents[2]` 指错而让门禁**因为一件无关的事**变红 ——
#      首版就这么假绿了 8/8）；
#   ② 「拿本脚本检查另一个 checkout」本来就是合理用法。
# 判据仍然只认 `paper2skills-vault` 的实际目录，指错了会**报错**而不是猜。
REPO = Path(os.environ.get("P2S_REPO") or Path(__file__).resolve().parents[2]).resolve()
VAULT = REPO / "paper2skills-vault"
POOL = REPO / "paper2skills-research" / "data" / "arxiv_candidates.json"

# ---------------------------------------------------------------------------
# 唯一事实源
# ---------------------------------------------------------------------------

#: 规范域名（17 个）。顺序 = 目录序 = 报告里的展示序，不另排一次。
CANONICAL: tuple[str, ...] = (
    "00-电商Agent",
    "01-因果推断",
    "02-A_B实验",
    "03-时间序列",
    "04-供应链",
    "05-推荐系统",
    "06-增长模型",
    "07-NLP-VOC",
    "08-知识图谱",
    "09-DataAgent-LLM",
    "10-MAS",
    "11-AI人文",
    "12-ML基础",
    "13-广告分析",
    "14-用户分析",
    "15-营销投放分析",
    "16-智能体工程",
)

#: **已退休的旧名 → 规范名**。⚠️ 这张表**不参与解析**（`resolve()` 不看它）——
#: 这是刻意的，理由见下面 `resolve()` 的注释。
#: 它只做两件事：① 给报错信息一句可行动的提示；② 给一次性迁移脚本当依据。
RETIRED: dict[str, str] = {
    "07-VOC舆情": "07-NLP-VOC",
    "09-DataAgent": "09-DataAgent-LLM",
    "15-营销投放": "15-营销投放分析",
}

#: vault 顶层里**形如域但不是业务域**的目录。必须逐条写明理由 ——
#: 否则「有目录没注册」这条判据会被这个例外悄悄吃掉。
NOT_A_DOMAIN: dict[str, str] = {
    "00-项目管理": "项目管理目录，不是业务域（与 `00-电商Agent` 同前缀，禁用前缀规则识别）",
    "07-资源库": "资源库（registry/契约/图谱的落点），不是业务域",
}

_DOMAIN_DIR_RE = re.compile(r"^\d\d-[^\s/]+$")

# ---------------------------------------------------------------------------
# 三态解析
# ---------------------------------------------------------------------------

def canonical_set() -> frozenset[str]:
    """**每次都从 `CANONICAL` 现算**，不缓存成模块级常量。

    ⚠️ 首版这里写的是 `CANONICAL_OF: dict = {d: d for d in CANONICAL}` ——
    一个**导入期**算好的常量。于是 `--mutate` 的 M2（从 `CANONICAL` 里抹掉
    `09-DataAgent-LLM`）**改了常量却没有改判据读到的值**：变异体报「✅ 抓住了」
    是靠别的原因，而那条判据在真实世界里有没有劲**根本没被证明**。
    —— 这正是本仓库记过的那个坑：**先证明变异改变了真实取值，再谈判据有没有劲。**
    """
    return frozenset(CANONICAL)


def resolve(label: str) -> tuple[str | None, str]:
    """把一个域标签折成规范名。返回 `(canonical | None, kind)`。

    返回 `(规范名 | None, kind)`，kind ∈ {`canonical`, `unknown`}。

    ⚠️ **这里一度还有一个 `alias` 态**：旧名经一张别名表**静默折到**规范名。
    它是为「盘上历史产物还带旧名」而加的迁移垫片。迁移做完后立刻暴露它是错的 ——
    别名表一归零，**下一次标签漂移就再也不会被发现**：新写的旧名会被悄悄折掉，
    而不是像现在这样让整台过滤器 `exit 2` 大喊「仪器瞎了」。
    ⇒ **「兼容」与「发现」是同一枚硬币的两面**：你替它兼容，你就不会再发现它。
    旧名现在走 `unknown` 分支（**响**），`RETIRED` 表只用来把报错说得可行动。
    """
    if label in canonical_set():
        return label, "canonical"
    return None, "unknown"


def resolve_groups(groups) -> dict:
    """解析一组域标签（一个条目的 `query_groups`）。

    返回 `{"resolved": [...], "retired": [旧名...], "unknown": [...]}`。
    退休名**进 `unknown`**（要被判红），`retired` 只是给它额外记一笔以便报错可行动。
    `resolved` 保序去重 —— 一篇论文同时带 `09-DataAgent` 与 `09-DataAgent-LLM`
    时必须折成**一条**，否则同一个域会被判定两次、计数也会重。
    """
    resolved: list[str] = []
    retired: list[str] = []
    unknown: list[str] = []
    for g in groups or []:
        canon, kind = resolve(g)
        if kind == "unknown":
            if g not in unknown:
                unknown.append(g)
            if g in RETIRED and g not in retired:
                retired.append(g)
            continue
        if canon not in resolved:
            resolved.append(canon)
    return {"resolved": resolved, "retired": retired, "unknown": unknown}


# ---------------------------------------------------------------------------
# 与 vault 实物对账
# ---------------------------------------------------------------------------

def vault_domains() -> list[str]:
    """vault 顶层里**形如域**的目录名（不排除例外，调用方自己减）。"""
    if not VAULT.is_dir():
        return []
    return sorted(p.name for p in VAULT.iterdir()
                  if p.is_dir() and _DOMAIN_DIR_RE.match(p.name))


def check(errors: list[str] | None = None, pool: Path = POOL) -> list[str]:
    """对账。返回错误列表（空 = 全过）。**不做任何修补**。"""
    errs: list[str] = []
    actual = vault_domains()
    if not actual:                 # 由 `main()` 折成 exit 2；`check()` 这里只报事实
        return [f"vault 目录读不到或没有形如域的目录：{VAULT}"]

    # ① 双向对账。只查一个方向＝只挡住「多注册」，挡不住「有目录没注册」。
    missing_dir = [d for d in CANONICAL if d not in actual]
    unregistered = [d for d in actual
                    if d not in canonical_set() and d not in NOT_A_DOMAIN]
    if missing_dir:
        errs.append(f"注册了却在 vault 里没有目录：{missing_dir} —— "
                    f"域名的 canonical 以 vault 目录为准，目录不在则它不是域")
    if unregistered:
        errs.append(f"vault 里有形如域的目录却没注册，也不在 NOT_A_DOMAIN 里："
                    f"{unregistered} —— 新域必须显式登记（否则它会像 "
                    f"`00-电商Agent` 那样被过滤器静默放行）")
    # 例外本身也要能被证伪：写了理由的例外若目录已不存在，它就是过期豁免。
    stale_exc = [d for d in NOT_A_DOMAIN if d not in actual]
    if stale_exc:
        errs.append(f"NOT_A_DOMAIN 里的目录已不存在：{stale_exc} —— 过期例外必须删")

    # ② 退休名登记表自身的形状。指到一个非规范名 ⇒ 报错提示会把下一班人带沟里。
    for old, new in RETIRED.items():
        # ⚠️ 顺序要紧：`old == new` 必须**先判**。
        # 首版把它放在最后，而恒等映射必然同时满足「new 不是规范名」，
        # 于是它**永远轮不到执行 = 一条死判据**（由 M6「红得不是地方」发现）。
        if old == new:
            errs.append(f"退休名 {old!r} 指向自己 —— 恒等映射什么也没说明")
        elif old in canonical_set():
            errs.append(f"退休名 {old!r} 本身是规范名 —— 那不是退休名，是重复登记")
        elif new not in canonical_set():
            errs.append(f"退休名 {old!r} → {new!r}，但 {new!r} 不是规范名")

    # ③ **盘上不许出现退休名**。这是本文件唯一一条「往外看」的判据，
    # 也是「一处一名」真正成立的地方 —— 只登记不检查＝一句没法证伪的话。
    usage = retired_usage(pool)
    if usage["available"]:
        for name, c in sorted(usage["per_retired"].items()):
            if c:
                errs.append(
                    f"盘上仍有 {c} 处使用退休名 {name!r}（规范名 {RETIRED[name]!r}）"
                    f" —— 跑 `migrate_domain_labels.py --apply`；"
                    f"**不许把它加回解析路径**：那会让下一次漂移重新变得看不见")

    if errors is not None:
        errors.extend(errs)
    return errs


def retired_usage(pool: Path = POOL) -> dict:
    """统计**退休名**在盘上产物里的使用次数 —— 应当恒为 0，不为 0 就判红。

    读不到池子 ⇒ `available=False`（**不是 0 次**：「没读」与「没用到」不是一回事）。
    """
    out = {"available": False, "path": str(pool), "per_retired": {a: 0 for a in RETIRED},
           "n_items_with_retired": 0, "n_items_total": 0}
    if not pool.is_file():
        return out
    data = json.loads(pool.read_text(encoding="utf-8"))
    items = data.get("items", data if isinstance(data, list) else [])
    out["available"] = True
    out["n_items_total"] = len(items)
    for it in items:
        hit = [g for g in (it.get("query_groups") or []) if g in RETIRED]
        if hit:
            out["n_items_with_retired"] += 1
            for g in set(hit):
                out["per_retired"][g] += 1
    return out


# ---------------------------------------------------------------------------
# 自检 + 变异
# ---------------------------------------------------------------------------

#: 每条判据配一份**篡改样本**。没有篡改样本的断言＝恒真断言＝没断言
#: （本仓库 F2 实测记过这条：M3/M5/M6 把断言整行删掉，自检照样全绿）。
#:
#: 做法：**改源文件 → 另起子进程跑真 CLI `--selftest`**，必须 exit≠0。
#: ⚠️ 首版是「在同一个进程里 `globals().update()` 打补丁，然后比对 `check()` 的返回」——
#: 两个毛病都撞上了：
#:   ① 只对拍 `check()` 的返回值 ⇒ **判据在 `selftest()` 里的那几条（③ 的夹具用例）
#:      改坏了也发现不了**（本仓库 S11 记过同型：「变异只对拍退出码，改坏后退出码一字未变」）；
#:   ② M2 改了 `CANONICAL` 而判据读的是导入期缓存的 `CANONICAL_OF` ⇒
#:      **变异没施上力却报「抓住了」**（已由 `canonical_set()` 的注释登记）。
#: 现在两条都堵住：子进程跑真 CLI，且锚点在**变异表之前的正文**里必须恰好出现一次。
MUTATIONS: list[tuple[str, str, str, str]] = [
    ("M1 注册一个 vault 里没有的域 ⇒ ① 必须报",
     '    "00-电商Agent",\n    "01-因果推断",',
     '    "99-不存在的域",\n    "00-电商Agent",\n    "01-因果推断",',
     "注册了却在 vault 里没有目录"),
    ("M2 抹掉一个注册域 ⇒ ① 的「有目录没注册」必须报",
     '    "09-DataAgent-LLM",\n',
     '',
     "却没注册"),
    ("M3 把例外表清空 ⇒ 「有目录没注册」必须改报 00-项目管理/07-资源库",
     'NOT_A_DOMAIN: dict[str, str] = {\n'
     '    "00-项目管理": "项目管理目录，不是业务域（与 `00-电商Agent` 同前缀，禁用前缀规则识别）",\n'
     '    "07-资源库": "资源库（registry/契约/图谱的落点），不是业务域",\n'
     '}', 
     'NOT_A_DOMAIN: dict[str, str] = {}',
     "00-项目管理"),
    ("M4 例外表里塞一个不存在的目录 ⇒ 过期豁免必须报",
     '    "07-资源库": "资源库（registry/契约/图谱的落点），不是业务域",\n',
     '    "07-资源库": "资源库（registry/契约/图谱的落点），不是业务域",\n'
     '    "99-幽灵目录": "构造的过期例外",\n',
     "已不存在"),
    ("M5 退休名指向非规范名 ⇒ ② 必须报（报错提示会把下一班人带沟里）",
     '    "07-VOC舆情": "07-NLP-VOC",\n',
     '    "07-VOC舆情": "07-不存在的规范名",\n',
     "不是规范名"),
    ("M6 退休名指向自己 ⇒ ② 必须报（恒等映射什么也没说明）",
     '    "09-DataAgent": "09-DataAgent-LLM",\n',
     '    "09-DataAgent": "09-DataAgent",\n',
     "指向自己"),
    ("M7 把一个规范名登记成退休名 ⇒ ② 必须报（重复登记）",
     '    "15-营销投放": "15-营销投放分析",\n',
     '    "15-营销投放": "15-营销投放分析",\n    "10-MAS": "11-AI人文",\n',
     "本身是规范名"),
    ("M8 让 `retired_usage` 认不出退休名 ⇒ ③ 的夹具用例必须报"
     "（守的是「盘上真的出现旧名时，还有没有人看得见」）",
     '        hit = [g for g in (it.get("query_groups") or []) if g in RETIRED]',
     '        hit = []',
     "夹具里放 1 处退休名"),
]


def mutate() -> int:
    """**改源文件 → 在原目录里另起子进程跑真 CLI `--selftest`**，验证每条判据都会失败。

    ⚠️ 首版把变异体放 `/tmp` 跑，于是 `REPO = parents[2]` 指到了临时目录的上级，
    `check()` 一律报「vault 目录读不到」，**8 个变异体全绿/全红都不是因为那条判据**，
    而屏幕上看起来是 8/8 ✅。现在两件事一起做：
      · 子进程带 `P2S_REPO=<真仓库根>`（否则门禁会因为路径而不是判据变色）；
      · 每条变异带一个 `expect`：**首行 ❌ 里必须出现指定字样**。
    **红得不是地方 = 没验到**（这条比「有没有红」重要）。
    """
    import subprocess
    import tempfile

    src_path = Path(__file__).resolve()
    src = src_path.read_text(encoding="utf-8")
    body = src.split("MUTATIONS: list[")[0]     # ⚠️ 只在正文里数锚点（变异表会抄自己）
    ok = True
    print("=== 变异：改源文件 → 子进程跑真 CLI `--selftest`（仓库根经 P2S_REPO 钉住）===")

    base = subprocess.run([sys.executable, str(src_path), "--selftest"],
                          capture_output=True, text=True)
    good = base.returncode == 0
    ok &= good
    print(f"  {'✅' if good else '❌'} 反向控制：未变异时 `--selftest` 必须 exit 0 "
          f"（实测 exit={base.returncode}）")

    n_armed = n_fired = n_right = 0
    for desc, old, new, expect in MUTATIONS:
        n_here = body.count(old)
        armed = n_here == 1
        n_armed += armed
        print(f"  {'✅' if armed else '❌'} {desc}")
        if not armed:
            print(f"       ↘ 锚点在正文里出现 {n_here} 次（必须恰好 1 次）"
                  f" ⇒ **没施上力**，下面的结果不算数")
            ok = False
            continue
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "domains.py"
            mut.write_text(src.replace(old, new, 1), encoding="utf-8")
            env = {**os.environ, "P2S_REPO": str(REPO)}
            r = subprocess.run([sys.executable, str(mut), "--selftest"],
                               capture_output=True, text=True, env=env)
        fired = r.returncode != 0
        n_fired += fired
        first = next((ln.strip() for ln in r.stdout.splitlines()
                      if ln.strip().startswith("❌")), "")
        right = fired and expect in first
        n_right += right
        ok &= right
        print(f"       已生效于探针：是 · 变异体 exit={r.returncode} "
              f"{'（判红 ✅）' if fired else '（**仍然全绿 ⇒ 判据是摆设** ❌）'}")
        if fired and not right:
            print(f"       ↘ ⚠️ **红得不是地方**：首行 ❌ 里没有 {expect!r} ⇒ "
                  f"这个变异验证的不是那条判据")
        if first:
            print(f"       ↘ {first[:112]}")

    print(f"\n{'✅' if ok else '❌'} 变异 {n_fired}/{len(MUTATIONS)} 打红 · "
          f"{n_right}/{len(MUTATIONS)} **红在正确的判据上** · "
          f"{n_armed}/{len(MUTATIONS)} 已生效于探针（另 1 条反向控制）")
    return 0 if ok else 1


def selftest(mutations: bool = False) -> int:
    ok = True
    print("=== 判据（在真实 vault 上）===")
    errs = check()
    if errs:
        ok = False
        for e in errs:
            print(f"  ❌ {e}")
    else:
        print(f"  ✅ 双向对账通过：{len(CANONICAL)} 个规范域 ↔ "
              f"{len(vault_domains())} 个 vault 目录（例外 {len(NOT_A_DOMAIN)} 个，各有理由）")

    print("\n=== 三态解析（`unknown` 必须与 `alias` 分开）===")
    cases = [
        ("09-DataAgent-LLM", "09-DataAgent-LLM", "canonical"),
        ("00-电商Agent", "00-电商Agent", "canonical"),
        ("09-DataAgent-LLMs", None, "unknown"),
        ("", None, "unknown"),
        ("09-dataagent-llm", None, "unknown"),
        # ⚠️ **退休名必须判 `unknown`，不许判 `canonical`**（本文件改过这一点）：
        # 认得它 ⇒ 静默放行 ⇒ 下一次漂移看不见。这三条是那个决定的锁。
        ("09-DataAgent", None, "unknown"),
        ("07-VOC舆情", None, "unknown"),
        ("15-营销投放", None, "unknown"),
    ]
    for label, exp_canon, exp_kind in cases:
        got_canon, got_kind = resolve(label)
        good = (got_canon, got_kind) == (exp_canon, exp_kind)
        ok &= good
        print(f"  {'✅' if good else '❌'} {label!r:<20} → {got_canon!r} / {got_kind}"
              f"（期望 {exp_canon!r} / {exp_kind}）")

    print("\n=== 组解析：去重 + 三类分开 ===")
    r = resolve_groups(["09-DataAgent-LLM", "09-DataAgent-LLM", "16-智能体工程", "09-DataAgentX"])
    good = (r["resolved"] == ["09-DataAgent-LLM", "16-智能体工程"]
            and r["unknown"] == ["09-DataAgentX"]
            and r["retired"] == [])
    ok &= good
    print(f"  {'✅' if good else '❌'} 同名重复只留一条；退休名与认不出的名分开；"
          f"resolved={r['resolved']} unknown={r['unknown']} retired={r['retired']}")
    # 反向控制：真的不同的两个域**不得**被折成一条（否则去重就成了「都并成一个」）
    r2 = resolve_groups(["09-DataAgent-LLM", "10-MAS"])
    good2 = r2["resolved"] == ["09-DataAgent-LLM", "10-MAS"]
    ok &= good2
    print(f"  {'✅' if good2 else '❌'} 反向控制：两个不同的域必须仍是两条 → {r2['resolved']}")

    print("\n=== ③ 盘上不许出现退休名（夹具；本判据靠这一条才有劲）===")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        fx = Path(td) / "pool.json"
        fx.write_text(json.dumps({"items": [
            {"arxiv_id": "f1", "query_groups": ["09-DataAgent"]},
            {"arxiv_id": "f2", "query_groups": ["09-DataAgent-LLM"]}]},
            ensure_ascii=False), encoding="utf-8")
        u = retired_usage(fx)
        good = u["per_retired"].get("09-DataAgent") == 1 and u["n_items_with_retired"] == 1
        ok &= good
        print(f"  {'✅' if good else '❌'} 夹具里放 1 处退休名 ⇒ 计数器必须数到 1 → "
              f"{u['per_retired']}")
        e = check(pool=fx)
        good2 = any("盘上仍有" in x and "09-DataAgent" in x for x in e)
        ok &= good2
        print(f"  {'✅' if good2 else '❌'} 同一夹具 ⇒ `check()` 必须判红并点名 → "
              f"{[x for x in e if '盘上仍有' in x][:1]}")
    # 反向控制：真池子上（已迁移）③ 必须一条都不报
    e_clean = check()
    good3 = not any("盘上仍有" in x for x in e_clean)
    ok &= good3
    print(f"  {'✅' if good3 else '❌'} 反向控制：真池子（已迁移）上 ③ 必须不报 → "
          f"{[x for x in e_clean if '盘上仍有' in x]}")

    if not mutations:
        print(f"\n{'✅ 自检通过' if ok else '❌ 自检失败'}"
              f"（--mutate 可验证每条判据都会失败）")
        return 0 if ok else 1

    if mutations:
        return mutate()
    print(f"\n{'✅ 自检通过' if ok else '❌ 自检失败'}"
          f"（--mutate 端到端验证每条判据都会失败）")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="业务域名唯一事实源（PHASE6 P1）")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest or args.mutate:
        # `--mutate` **蕴含** `--selftest`：变异的意义就是「跑自检看它红不红」，
        # 单独给一个不跑自检的 `--mutate` 只会让人以为验证过了。
        return selftest(mutations=args.mutate)

    # ⚠️ 「vault 目录读不到」是**没测到**，不是**判红** —— 本仓库危险性排序 3 > 2 > 1 > 0。
    # 首版让它混在 `check()` 的错误列表里一起走 exit 1，等于把「仪器没拿到输入」
    # 报成了「资产有问题」（与 `check_instruction_budget` 的 `预算取不到 ⇒ exit 2` 同尺）。
    if not vault_domains():
        print(f"🔴 vault 目录读不到或没有形如域的目录：{VAULT}\n"
              f"   —— 「没东西可查」不等于「查过了没问题」⇒ exit 2（不是 0，也不是 1）")
        return 2
    errs = check()
    usage = retired_usage()
    if args.json:
        print(json.dumps({"canonical": list(CANONICAL), "retired": RETIRED,
                          "not_a_domain": NOT_A_DOMAIN,
                          "vault_domains": vault_domains(),
                          "retired_usage": usage, "errors": errs},
                         ensure_ascii=False, indent=1))
        return 0 if not errs else 1

    print(f"规范域 {len(CANONICAL)} 个 · vault 目录 {len(vault_domains())} 个 · "
          f"例外 {len(NOT_A_DOMAIN)} 个 · 退休名 {len(RETIRED)} 条")
    print("\n退休名在盘上的使用次数（**应当恒为 0**；不为 0 就判红）：")
    if not usage["available"]:
        print(f"  ⚠️ 候选池读不到（{usage['path']}）—— 「没读」不是「0 次」")
    else:
        for a, n in usage["per_retired"].items():
            print(f"  {n:>4}  {a} → {RETIRED[a]}")
        if usage["n_items_with_retired"] == 0:
            print("  ✅ 0 处 —— 「一处一名」成立")

    if errs:
        print("\n🔴 对账失败：")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("\n✅ 域名对账通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
