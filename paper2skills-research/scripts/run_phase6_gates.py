#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE6 验收面 —— 一条命令跑完所有门禁，**逐条分开报**，绝不合并成一个分数。

为什么需要它（不是"顺手做个 runner"）：

PHASE6 到收口期已经有 11 个门禁脚本，分散在三条不同命令形态里
（`--check` / `--selftest` / 无参数即判）。**没有任何一处能回答"整个 PHASE6 现在是什么状态"** ——
每个数字都只活在产它的那一份报告里，而要复核就得把 11 条命令逐个背出来。
本仓库已经吃过这个亏：CLAUDE.md 里记着"三个门禁必须分开报"，
而"分开报"的前提是**有一个地方能把它们放在一起报**。

三条硬纪律（都来自本仓库已发生的实测事故）：

① **退出码不许合并成一个分数。** 结果表逐门禁给码，汇总只给"各类各几条"。
   仓库既有口径：0 全过 / 1 判红 / 2 **输入没拿到（≠ 通过）** / 3 门禁内部错误（≠ 判红）。
   危险性排序是 **3 > 2 > 1 > 0** ——
   「没测到」比「测了是红的」更危险，因为红会有人去修，没测到没人知道。

② **一个门禁都没跑到 = exit 3，不是 exit 0。**
   与 `scan_secrets.py`（扫到 0 个文件判失败）同源。空列表不是"干净"。

③ **豁免必须可见、必须带到期条件。** 未完工的门禁可以声明 `waived`，
   但 runner 会把豁免逐条打出来并**在它转绿时报"可以取消豁免"** ——
   否则豁免会像台账 #5 记的那样，以"大家都学会绕路"的形式永久化。

用法：
    python3 paper2skills-research/scripts/run_phase6_gates.py            # 全跑
    python3 paper2skills-research/scripts/run_phase6_gates.py --fast     # 跳过 selftest 类
    python3 paper2skills-research/scripts/run_phase6_gates.py --only L4d
    python3 paper2skills-research/scripts/run_phase6_gates.py --selftest # 证明 runner 自己会红
    python3 paper2skills-research/scripts/run_phase6_gates.py --json-out X.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# 材料根：S1 材料引文核验器与门禁 G-L4-c 都用它。取不到 ⇒ exit 2，不许退回空值。
MATERIAL = os.environ.get("P2S_MATERIAL_DIR", "/Users/lute/project/AI组织变革")


@dataclass
class Gate:
    gid: str
    title: str
    argv: list
    kind: str = "check"          # check | selftest
    # 未完工门禁的可见豁免：必须写清为什么、以及取消条件（"到期条件"）。
    waived: str = ""
    waive_until: str = ""
    cwd: str = ""
    env: dict = field(default_factory=dict)
    timeout: int = 900


def _p(*parts: str) -> str:
    return str(HERE.joinpath(*parts))


GATES: list[Gate] = [
    # --- 工作区 instruction 预算（本轮新增；实测撑爆过一次）---
    # ⚠️ 这条门禁守的是**所有其它门禁都看不见的一类丢失**：磁盘上的 `CLAUDE.md` 是完整的，
    # 而 agent 实际读到的被 harness **静默截断**，`AGENTS.md` / `~/.dsh/AGENTS.md` 整份没进上下文。
    # 实测事件：`truncated CLAUDE.md from 66175 to 65244 bytes`（被砍掉的是文件尾部）。
    # 同一个洞本仓库已经踩过第二次（上一次砍掉的恰好是 `Skill Card Format` 那一节）。
    Gate("L1a", "工作区 instruction 预算：全部指令文件完整进入上下文（无丢弃/无截断）",
         ["check_instruction_budget.py"]),
    Gate("L1b", "工作区 instruction 预算：自检（含真实截断事件回归 + 边界正反证）",
         ["check_instruction_budget.py", "--selftest"], kind="selftest"),
    Gate("L1c", "工作区 instruction 预算：变异测试（渲染/退出码改坏必须被抓）",
         ["check_instruction_budget.py", "--mutate"], kind="selftest"),
    # --- 图谱层（F2）---
    Gate("L2a", "五层能力图谱：图与材料一致", ["build_capability_graph.py", "--check"]),
    Gate("L2b", "五层能力图谱：L1–L3 与产品侧 taxonomy 逐项相等（证明没造第二份分类表）",
         ["build_capability_graph.py", "--check-taxonomy"]),
    Gate("L2c", "五层能力图谱：自检", ["build_capability_graph.py", "--selftest"], kind="selftest"),
    # --- 缺口账（F3）---
    Gate("L3a", "缺口账与靶区工单：与图谱/分类现状一致", ["build_gap_ledger.py", "--check"]),
    Gate("L3b", "缺口账：自检", ["build_gap_ledger.py", "--selftest"], kind="selftest"),
    # --- 分类轴（F4/F5）---
    Gate("L3c", "卡端分类：146 张卡逐项与产品侧一致", ["build_card_classification.py", "--check"]),
    Gate("L3d", "卡端分类：自检", ["build_card_classification.py", "--selftest"], kind="selftest"),
    # ⚠️ #60：分类层的依据行锚点是**位置锚**，S13 给 146 张卡插 frontmatter 后它整片报红。
    # 这条变异测试守的就是「位置漂移 ≠ 内容改变」这区分 —— 没有它，下次同样的插入会再红一次。
    Gate("L3e", "卡端分类：变异测试（位置锚 vs 内容改变）",
         ["build_card_classification.py", "--mutate"], kind="selftest"),
    # --- 契约生成与作业包（F6/F7/S1）---
    Gate("L4a", "契约生成器：底本与实物一致", ["build_contracts.py", "--check"]),
    Gate("L4b", "契约作业包：批次与材料摘要一致", ["build_contract_workpack.py", "--check"]),
    # --- 契约层判据（J1–J13）---
    Gate("L4c", "契约层判据 J1–J13（139 份全量）", ["check_contracts.py", "--all"]),
    Gate("L4d", "契约层判据：自检 + 变异样本", ["check_contracts.py", "--selftest"], kind="selftest"),
    # --- 材料引文（独立门禁，**不与 J1–J13 同号** —— 它是 S1 收口后才立的引用纪律）---
    Gate("L4e", "材料引文：契约声称的『材料「X」』必须真的在材料里",
         ["check_material_citations.py", "--material", MATERIAL]),
    Gate("L4f", "材料引文：自检", ["check_material_citations.py", "--selftest"], kind="selftest"),
    # --- 跨契约同质化（S1 收口期新增）---
    Gate("L4g", "契约层：跨文件同质化机检", ["check_contract_dedup.py"]),
    Gate("L4h", "契约层：同质化检测器自检", ["check_contract_dedup.py", "--selftest"], kind="selftest"),
    # --- 契约「数据要求」栏五维判据（S11）---
    # ⚠️ **不与 J1–J13 同号**：J5 已判「五维齐全 + ① 断言枚举」，但它的颗粒度是**行**
    # （J5 自己的注释里就记着实测边界：① 写「本企业无自有埋点，实际全靠人工估算」而 exit=0）。
    # S11 把颗粒度降到**格**，只问「这一维点名了吗」，**不替换** J5/J6/J13。
    # 本仓库已两次因颗粒度（列 vs 行 vs 块）翻面判决，故两者并存、分开报。
    Gate("L4i", "契约数据要求：五维逐格点名（与 J5 的行颗粒度分开）",
         ["check_data_requirements.py", "--dir", str(REPO / "paper2skills-vault" / "07-资源库" / "contracts")]),
    Gate("L4j", "契约数据要求：自检 + 变异（含「不可得是合法结论」反向控制）",
         ["check_data_requirements.py", "--selftest"], kind="selftest"),
    # --- 入卡门槛（S13）---
    Gate("L5a", "入卡弱门槛：146 张卡的 L3 归属可机读", ["check_card_l3.py", "--check"]),
    Gate("L5b", "入卡弱门槛：自检", ["check_card_l3.py", "--selftest"], kind="selftest"),
    # --- 缺口驱动检索式（S4）---
    Gate("L6a", "缺口驱动检索式：与工单/图谱一致", ["build_search_queries.py", "--check"]),
    Gate("L6b", "缺口驱动检索式：自检 + 价值变异", ["build_search_queries.py", "--selftest"],
         kind="selftest"),
    # --- 旧方案页收割（S2）---
    # ⚠️ 这条门禁守的是**负知识的落点**：19 个旧方案页是第四套平行分类（不绑任何 AGT/SCN），
    # 其「三大架构陷阱」不可能从 64 格骨架推出来。J13 里「87/151 L3 有落点、64 个 L3 无人触及」
    # 才是缺口图；而「24/24 M 格都有落点」**没有区分度**（每格 14–19 页可达），故不得当供给结论用。
    Gate("L7a", "旧方案页收割：M 格素材逐条可回指源 HTML", ["harvest_legacy_solutions.py", "--check"]),
    Gate("L7b", "旧方案页收割：判据自检 + 变异（含 194 卡位丢弃登记）",
         ["harvest_legacy_solutions.py", "--selftest"], kind="selftest"),
    # --- 「同名同物」判定（S5 换底的前提；2026-09-13 由主控接线）---
    # ⚠️ 这条门禁交付时就是 **exit 1**，而它**没被接进验收面** —— 于是验收面报「全绿」
    #    而一个已交付的判据在报红。这正是本验收面存在的理由：「没测到」比「测了是红的」更危险。
    # 8 条 I3 的处置是**可见豁免**（不是修）：它们不是缺陷，是两条语料线里真实存在的近名卡，
    # 每条的 p2s_card_id 都在 legacy 线（`playbook/domains/*.html`）里有同名条目 ——
    # 抽样四条逐条复核过。「vault 里不存在」= 判据只看得见精选线 146 张这一个库。
    # 豁免清单带 `expires_when`，且**转绿时会提示删除**（永久豁免＝台账 #5 那种腐烂）。
    Gate("L8a", "同名同物：p2s 卡 ↔ 精选卡的六态判定（8 条 I3 走可见豁免，非豁免项仍判红）",
         ["check_card_identity.py", "--check",
          "--baseline", _p("..", "data", "card-identity-baseline.json")]),
    Gate("L8b", "同名同物：自检（含「豁免不是全放行」的隔离用例）",
         ["check_card_identity.py", "--selftest"], kind="selftest"),
    Gate("L8c", "同名同物：变异测试（判据改坏必须被抓，且须证明变异真的生效）",
         ["check_card_identity.py", "--mutate"], kind="selftest"),
    # --- extract backlog 接管（S3；2026-09-13 由主控接线，**接上时它就是红的**）---
    # ⚠️ 接线时的实况：`--check` **exit 1**。根因不是判据坏了，是**产物是 S5 换底前的照片**
    #    —— 产品侧分类 1338 → 1390（S5 推的 52 张），逐 L3 供给数跟着变。
    #    **决策部分逐条相同**（12 条接管序、L3 落点、档位、占检索预算的 3 条全不变），
    #    变的只有供给读数（9 条 `n_legacy` **全部上移 +1..+5**，与「只增不减」互为佐证）。
    # ⚠️ 同一次检查还撞出 J9 一个「判据永远不可能失败」的结构洞：第一版把逐行内容核对
    #    关在 `stale` 分支里 ⇒ sha 自述新鲜时**一行都不看**。已修（见 route_backlog.py 的 docstring），
    #    并在 L9b 里补上结构洞专用用例；`git log` 逐提交复算证明该洞在 4 个提交里的 3 个上是活的。
    Gate("L9a", "extract backlog 接管：与 registry/缺口账/映射一致（含 J9 输入新鲜度内容核对）",
         ["route_backlog.py", "--check"]),
    Gate("L9b", "extract backlog 接管：自检（J9 含「sha 自述新鲜但内容被改」的结构洞用例）",
         ["route_backlog.py", "--selftest"], kind="selftest"),
    # --- p2s 换底（S5）---
    Gate("L10a", "p2s 换底：两语料收编后卡片 / references / staging 预算一致",
         ["rebase_p2s_cards.py", "--check"]),
    Gate("L10b", "p2s 换底：自检", ["rebase_p2s_cards.py", "--selftest"], kind="selftest"),
    Gate("L10c", "p2s 换底：变异测试", ["rebase_p2s_cards.py", "--mutate"], kind="selftest"),
    # --- 独立站 holdback（S9）---
    # ⚠️ `--check` 是 0，但本脚本的**主运行按设计是 exit 2**（无 `--traffic` 时「输入没拿到」
    #    —— 独立站流量在本仓库确实没有事实源）。验收面只取 `--check`：产物与现算一致与否是判据，
    #    「流量拿到了没有」不是。
    Gate("L11a", "独立站 holdback：分流口径与逐条责任功效产物与现算一致",
         ["holdback_power.py", "--check"]),
    Gate("L11b", "独立站 holdback：自检", ["holdback_power.py", "--selftest"], kind="selftest"),
    Gate("L11c", "独立站 holdback：变异测试", ["holdback_power.py", "--mutate"], kind="selftest"),
    # --- venue 分层（S10）---
    Gate("L12a", "venue 三词表统一：146 张卡 venue_tier 全量回填且与规范表逐条一致",
         ["build_venue_tiers.py", "--check"]),
    Gate("L12b", "venue 三词表统一：自检 + 变异（默认带变异，`--no-mutations` 才是快路径）",
         ["build_venue_tiers.py", "--selftest"], kind="selftest"),
    # --- K2/G3 卡侧门禁（PHASE6 从两侧改动了它，故它必须进验收面）---
    # ⚠️ 为什么一个**既有**门禁要进 PHASE6 的验收面：本阶段的产出物**从两侧**改动了它 ——
    #    ① 146 张卡新增了 `l3_*` / `venue_tier` / `venue_source` 等 frontmatter 字段；
    #    ② 主控据此改了 `gate_g3` 的扫描口径（台账 #70：元数据不得当业务场景信号）。
    #    一个「被本阶段改动过的判据」若不跑，就等于本阶段对自己最相关的那道门没有任何读数。
    # ⚠️ 这里只接 `--selftest`（0.6s，确定性）：`--all` 是**报告生成器**（写着红灯也退 1，
    #    那不是「门禁判红」而是「库里确实有红卡」），接进来会把报告与判据两种退出码语义混在一个数里。
    #    判据是否可信 = selftest；库里有几张红卡 = 报告。**两者分开**。
    #    ⚠️ 路径口径：`_p()` 是相对**本脚本所在目录**（`paper2skills-research/scripts`）拼接的，
    #    这是验收面里第一条跨目录的门禁，所以要显式走 `../..`。首版写成了
    #    `paper-审核/scripts/gate_check.py` ⇒ 解析到一个不存在的路径 ⇒ 汇总报 **exit=2「输入没拿到」**。
    #    **这正是三态设计该有的样子**：不是 0（假绿），也不是 1（假红）。
    Gate("L13a", "卡侧 G3 判据：自检（含「元数据不得当业务场景信号」的隔离用例 + 反向控制）",
         [_p("..", "..", "paper2skills-skills", "paper-审核", "scripts", "gate_check.py"),
          "--selftest"], kind="selftest"),
    # --- 凭证暴露面（S8 报告 §四 的待办；2026-09-13 由主控落地）---
    # ⚠️ 它补的是 `scan_secrets.py` **结构性看不见**的那一类：那边**只扫已入库文件**，
    #    因此「未被跟踪但**也未被忽略**」的私钥（一次 `git add -A` 即入库）不在它的适用范围里 ——
    #    S8 实测的那条真暴露面正是这一态，而 `scan_secrets.py` 对 DDDD.pem / ai_video.pem
    #    **一把都查不到**（不是失灵，是它们从未入库）。
    # ⚠️ **只接 `--check`（扫本仓库工作区，确定性）与 `--selftest`**；
    #    `--sweep-roots`（广扫任意目录，如 ~/project）**刻意不进验收面** ——
    #    它随环境漂移、跨机器结果不同，是**一等读数**不是判决（默认恒 exit 0，`--strict` 才判红）。
    #    把两者混起来，就会造出一条「结果取决于谁在哪台机器上跑」的门禁。
    Gate("L14a", "凭证暴露面：工作区里没有被跟踪、也没有「未被忽略」的密钥（补 scan_secrets 的盲区）",
         ["check_key_exposure.py", "--check"]),
    Gate("L14b", "凭证暴露面：自检（五态 + 公开证书白名单反向控制 + 三份「门禁改坏」变异）",
         ["check_key_exposure.py", "--selftest"], kind="selftest"),

    # --- 凭证扫描的另外两条缝（2026-09-13 覆盖审计的第二批发现）---
    # ⚠️ `scan_secrets.py` 是本仓库 CLAUDE.md 里写明的「推送前必跑」门禁，**却从来没进过验收面**。
    #    而这个事实有代价：它当时**正红着**（3 条，由 check_key_exposure.py 的固件字面量造成），
    #    是在覆盖审计中被人肉发现的，不是被判据发现的 —— 与台账 #23（check_card_identity
    #    交付时即 exit 1 而无人接线）**同型，而且是同一条纪律在两天内第二次被违反**。
    Gate("L15a", "推送前凭证扫描：已入库文件内容（CLAUDE.md 的「推送前必跑」，此前从未接线）",
         [_p("..", "..", "paper2skills-skills", "paper-维护", "scripts", "scan_secrets.py")]),
    Gate("L15b", "推送前凭证扫描：自检（含「扫描器自身必须干净」元级用例 + 墓碑反向控制）",
         [_p("..", "..", "paper2skills-skills", "paper-维护", "scripts", "scan_secrets.py"),
          "--selftest"], kind="selftest"),
    # ⚠️ L16a 现在**预期判红**：判据 C 抓到一个**实测仍然有效**的飞书 webhook
    #    （公开仓库历史里可一行还原）。活凭证不许豁免 ⇒ 这条红要留到飞书后台轮换完成。
    #    红在这里是对的：它对应一个真实的、有行动主的未闭环项（任务板 B14 / S8）。
    Gate("L16a", "凭证历史扫描：对象库内容 + 路径史 + 已知暴露值还原（第三条缝）",
         [_p("..", "..", "paper2skills-skills", "paper-维护", "scripts",
             "check_history_secrets.py"), "--check"]),
    Gate("L16b", "凭证历史扫描：自检（含「判据 C 抓得住而判据 B 抓不到」+ live 不许豁免反向控制）",
         [_p("..", "..", "paper2skills-skills", "paper-维护", "scripts",
             "check_history_secrets.py"), "--selftest"], kind="selftest"),
    Gate("L16c", "凭证历史扫描：变异测试（5 份端到端，各要求「先证明变异改变了真实取值」）",
         [_p("..", "..", "paper2skills-skills", "paper-维护", "scripts",
             "check_history_secrets.py"), "--mutate"], kind="selftest"),
]

SEV = {0: 0, 1: 1, 2: 2, 3: 3}


def run_gate(g: Gate) -> dict:
    argv = [sys.executable, _p(g.argv[0])] + list(g.argv[1:])
    env = {**os.environ, **g.env}
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=g.cwd or str(REPO), env=env,
                           capture_output=True, text=True, timeout=g.timeout)
        code, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        code, out = 3, f"❌ 超时（{g.timeout}s）"
    except FileNotFoundError as e:
        code, out = 2, f"❌ 跑不起来：{e}"
    return {"id": g.gid, "title": g.title, "code": code, "seconds": round(time.time() - t0, 1),
            "tail": "\n".join(out.strip().splitlines()[-3:]),
            "waived": bool(g.waived), "waive_until": g.waive_until}


def summarise(results: list, skipped: list) -> int:
    """逐类分开报；返回集合退出码（3 > 2 > 1 > 0）。"""
    if not results:
        print("❌ 一个门禁都没跑到 —— 「没测」不是「通过」，也不是「判红」。exit 3。", file=sys.stderr)
        return 3
    by = {0: [], 1: [], 2: [], 3: []}
    for r in results:
        by.setdefault(r["code"], []).append(r)
    print()
    print("=" * 78)
    print(f"门禁 {len(results)} 条（跳过 {len(skipped)}）· "
          f"✅ 全过 {len(by[0])} · 🔴 判红 {len(by[1])} · "
          f"❓ 输入没拿到 {len(by[2])} · 💥 门禁内部错误 {len(by[3])}")
    print("⚠️ 三类分开列，不合并成一个「PHASE6 得分」—— 合并会把最弱的那一维平均掉。")
    for c, label in ((3, "💥 门禁内部错误（先修仪器，不是被检对象）"),
                     (2, "❓ 输入没拿到（≠ 通过）"),
                     (1, "🔴 判红"),
                     (0, "✅ 全过")):
        for r in by[c]:
            mark = "🟡豁免" if r["waived"] and c != 0 else "  "
            print(f"  {mark} [{r['id']}] {r['title']}  → exit={r['code']} ({r['seconds']}s)")
    waived_red = [r for r in results if r["waived"] and r["code"] != 0]
    waived_green = [r for r in results if r["waived"] and r["code"] == 0]
    if waived_red:
        print("\n🟡 **可见豁免（未完工，不计入退出码）** —— 逐条带到期条件：")
        for r in waived_red:
            print(f"    [{r['id']}] {r['title']}")
            print(f"         到期条件：{r['waive_until']}")
    if waived_green:
        print("\n⚠️ **以下豁免已经转绿 —— 请取消豁免**（永久豁免是本仓库台账 #5 记的那种腐烂）：")
        for r in waived_green:
            print(f"    [{r['id']}] {r['title']}")
    hard = [r for r in results if not r["waived"]]
    worst = max((SEV.get(r["code"], 3) for r in hard), default=0)
    print(f"\n⇒ 集合退出码 exit={worst}（只由**未豁免**的 {len(hard)} 条门禁决定）")
    return worst


# --------------------------------------------------------------------------
# runner 自己的自检：证明它**会红**、会区分「没测到」、会拒绝空列表
# --------------------------------------------------------------------------
def e2e_material_gate(checker: Path | None = None) -> tuple:
    """把材料引文门禁端到端跑一遍（真 CLI + 真材料 + 构造契约夹具）。

    ⚠️ 这一条是 W3 的验收本体：台账 #25 的教训是
    「判据在 main() 里而 selftest 只测库函数 ⇒ 把守卫改成 if(false) 照样全绿」。
    故这里**不 import 库函数**，一律 `subprocess` 跑真 CLI。
    `checker` 可指向被篡改的副本，用来证明这三条用例**真的有劲**（见 mutate()）。
    返回 (用例结果列表, 明细)。
    """
    checker = checker or Path(_p("check_material_citations.py"))
    cases, detail = [], []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def fixture(name: str, body: str) -> Path:
            d = tmp / name
            (d / "A").mkdir(parents=True)
            (d / "A" / "CTR-A-999-夹具.md").write_text(body, encoding="utf-8")
            return d

        head = ("---\ntemplate_version: v2\n---\n\n# 夹具\n\n"
                "## §1 可复算性\n- 取值：容差档 5%。\n\n## §3 标定\n- 取值：点击率 3.2%。\n")
        # ① 注入自造词并**声称来自材料** ⇒ 必须 exit 1
        d1 = fixture("injected", head + "\n- 经营节奏（材料「季度经营策略」的经营节奏）。\n")
        # ② 干净夹具（引用材料真有的词）⇒ 必须 exit 0
        d2 = fixture("clean", head + "\n- 经营节奏（材料「月度经营复盘」＝1 个自然月）。\n")
        # ③ 材料根指错 ⇒ 必须 exit 2（不是 0、也不是 1）
        d3 = fixture("clean2", head + "填充。\n")

        def cli(d, material=MATERIAL):
            p = subprocess.run([sys.executable, str(checker),
                                "--material", material, "--dir", str(d)],
                               capture_output=True, text=True, timeout=300)
            return p.returncode, (p.stdout or "") + (p.stderr or "")

        for label, d, want, mat in (
                ("① 注入 `材料「季度经营策略」` ⇒ exit 1", d1, 1, MATERIAL),
                ("② 干净夹具（材料真有的词）⇒ exit 0", d2, 0, MATERIAL),
                ("③ 材料根不存在 ⇒ exit 2（≠ 0）", d3, 2, str(tmp / "nope"))):
            got, out = cli(d, mat)
            cases.append((label, got == want))
            detail.append({"case": label, "want": want, "got": got,
                           "head": out.strip().splitlines()[0] if out.strip() else ""})
    return cases, detail


def selftest() -> int:
    """runner 自检 —— 三条判据，每条都能失败。"""
    cases = []

    # ① 端到端：材料引文门禁真的会红（这是 W3 的验收原话）
    mat_cases, mat_detail = e2e_material_gate()
    cases += mat_cases

    # ② runner 会把「红」传播出来（用一个故意 exit 1 的假门禁）
    fake = Gate("FAKE-RED", "假门禁（应判红）", ["run_phase6_gates.py", "--list"])
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.py"
        bad.write_text("import sys; sys.exit(1)\n", encoding="utf-8")
        missing = Path(td) / "missing.py"
        zero = Path(td) / "zero.py"
        zero.write_text("import sys; sys.exit(0)\n", encoding="utf-8")

        g_red = Gate("FAKE-RED", "假门禁 exit 1", [str(bad)])
        g_missing = Gate("FAKE-MISSING", "假门禁 文件不存在", [str(missing)])
        g_ok = Gate("FAKE-OK", "假门禁 exit 0", [str(zero)])

        import contextlib, io
        def run_quiet(gates):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                res = [run_gate(g) for g in gates]
                code = summarise(res, [])
            return code, buf.getvalue()

        c_red, o_red = run_quiet([g_red])
        cases.append(("④ 门禁判红 ⇒ 集合码 exit 1", c_red == 1))
        c_miss, o_miss = run_quiet([g_missing])
        cases.append(("⑤ 门禁跑不起来 ⇒ 集合码 exit 2，**不得是 0**", c_miss == 2))
        c_ok, _ = run_quiet([g_ok])
        cases.append(("⑥ 门禁全过 ⇒ 集合码 exit 0", c_ok == 0))
        c_mix, o_mix = run_quiet([g_ok, g_red, g_missing])
        cases.append(("⑦ 混合 ⇒ 集合码取最危（2 > 1 > 0）", c_mix == 2))
        # ⑧ 空列表 ⇒ exit 3（「一个都没扫到」不是干净）
        c_empty, _ = run_quiet([])
        cases.append(("⑧ 一个门禁都没跑到 ⇒ exit 3，不得是 0", c_empty == 3))
        # ⑨ 豁免可见：未完工且红时，**标题与到期条件都必须被打印出来**
        #    ⚠️ 首版这里写的是 `... in o_red or True` —— 恒真，等于没有断言。
        #    本仓库已抓到过同一形态三次（build_capability_graph 的 M3/M5/M6、
        #    gate_check 的「用例 18 是摆设」）。断言恒真 = 没断言。
        g_wr0 = Gate("FAKE-WR0", "假门禁 未完工且红", [str(bad)],
                     waived="测试豁免", waive_until="测试到期条件XYZ")
        _, o_wr0 = run_quiet([g_ok, g_wr0])
        cases.append(("⑨ 豁免的标题与到期条件都必须被打出来",
                      "可见豁免" in o_wr0 and "FAKE-WR0" in o_wr0
                      and "测试到期条件XYZ" in o_wr0))
        g_w = Gate("FAKE-W", "假门禁 未完工", [str(zero)], waived="测试豁免",
                   waive_until="测试到期条件")
        c_w, o_w = run_quiet([g_ok, g_w])
        cases.append(("⑩ 已转绿的豁免必须提示「取消豁免」", "请取消豁免" in o_w))
        c_wr, o_wr = run_quiet([g_ok, Gate("FAKE-WR", "假门禁 未完工且红", [str(bad)],
                                           waived="测试豁免", waive_until="测试到期条件")])
        cases.append(("⑪ 豁免的门禁不计入集合退出码", c_wr == 0))

    print("run_phase6_gates · selftest")
    for label, ok in cases:
        print(f"  {'✅' if ok else '❌'} {label}")
    n_ok = sum(1 for _, ok in cases if ok)
    print(f"\n{n_ok}/{len(cases)} 通过")
    if mat_detail:
        print("材料引文端到端明细：")
        for d in mat_detail:
            print(f"  want={d['want']} got={d['got']}  {d['case']}")
            if d["head"]:
                print(f"      {d['head'][:100]}")
    return 0 if n_ok == len(cases) else 1


def mutate() -> int:
    """变异测试：把**被检门禁**改坏，看 runner 的端到端用例抓不抓得住。

    纪律来源：本仓库已抓到 4 次「断言恒真 = 没断言」/「用例是摆设」。
    一条用例有没有劲，只能靠**把被测对象改坏**来回答。
    每个变异指明「应当由哪条用例抓住」；抓不住即判红。
    """
    src = Path(_p("check_material_citations.py")).read_text(encoding="utf-8")

    # ⚠️ 锚点必须**唯一且落在退出码分支上**。首版用裸 `    if bad:\n` 做锚点，
    #    而该文本在文件里出现两次（打印分支在前、退出码分支在后），`replace(..., 1)`
    #    只改到**打印**那处 ⇒ 变异没施上力，却被读成「用例是摆设」。
    #    这与本仓库记的「先证明变异改变了真实取值，再谈判据」是同一条纪律
    #    （S4 的缺陷 ⑥：变异样本没施上力，看起来像判据无效）。
    EXIT1 = '    if bad:\n        print("\\n⇒ exit 1：材料引文有问题'
    MUTANTS = [
        ("M1 恒真放行：判红分支 `if bad:` → `if False:`（问题照打，但不再判红）",
         EXIT1, EXIT1.replace("    if bad:", "    if False:", 1),
         "① 注入 `材料「季度经营策略」` ⇒ exit 1"),
        ("M2 材料根取不到也放行：材料根不存在分支 `return 2` → `return 0`",
         '「没拿到输入」不等于「通过」", file=sys.stderr)\n        return 2',
         '「没拿到输入」不等于「通过」", file=sys.stderr)\n        return 0',
         "③ 材料根不存在 ⇒ exit 2（≠ 0）"),
        ("M3 恒红：判红分支 `if bad:` → `if True:`（干净输入也判红）",
         EXIT1, EXIT1.replace("    if bad:", "    if True:", 1),
         "② 干净夹具（材料真有的词）⇒ exit 0"),
    ]

    n_ok = 0
    print("run_phase6_gates · 变异测试（改坏被检门禁，看用例抓不抓得住）")
    for label, old, new, catcher in MUTANTS:
        if old not in src:
            print(f"  ❌ {label} —— 变异施不上力（锚点文本找不到，说明上游改了代码）")
            continue
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "check_material_citations_mutant.py"
            mp.write_text(src.replace(old, new, 1), encoding="utf-8")
            cases, _ = e2e_material_gate(checker=mp)
            hit = dict(cases)
            caught = hit.get(catcher) is False
            print(f"  {'✅' if caught else '❌'} {label}")
            print(f"        应由「{catcher}」抓住 —— "
                  f"{'抓住了' if caught else '**没抓住**（该用例是摆设）'}")
            if caught:
                n_ok += 1
    print(f"\n{n_ok}/{len(MUTANTS)} 抓住")
    return 0 if n_ok == len(MUTANTS) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑某个门禁 id（可逗号分隔）")
    ap.add_argument("--fast", action="store_true", help="跳过 kind=selftest 的门禁")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true",
                    help="把被检门禁改坏，检验 runner 的端到端用例有没有劲")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.mutate:
        return mutate()

    gates = list(GATES)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        gates = [g for g in gates if g.gid in want]
        if not gates:
            print(f"❌ --only {args.only} 没匹配到任何门禁", file=sys.stderr)
            return 2
    skipped = []
    if args.fast:
        skipped = [g.gid for g in gates if g.kind == "selftest"]
        gates = [g for g in gates if g.kind != "selftest"]

    if args.list:
        for g in gates:
            w = " 🟡豁免" if g.waived else ""
            print(f"{g.gid:6s} [{g.kind:8s}] {' '.join(g.argv)}{w}")
        return 0

    if not Path(MATERIAL).is_dir():
        print(f"⚠️ 材料根取不到：{MATERIAL} —— G-L4e 会判 exit 2（这是「没测到」，不是「通过」）",
              file=sys.stderr)

    results = []
    for g in gates:
        print(f"… [{g.gid}] {g.title}", flush=True)
        r = run_gate(g)
        results.append(r)
        icon = {0: "✅", 1: "🔴", 2: "❓", 3: "💥"}.get(r["code"], "?")
        print(f"  {icon} exit={r['code']} ({r['seconds']}s)")
        if r["code"] != 0 and r["tail"]:
            for ln in r["tail"].splitlines():
                print(f"      {ln[:160]}")

    code = summarise(results, skipped)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"gates": results, "skipped": skipped, "suite_exit": code},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
