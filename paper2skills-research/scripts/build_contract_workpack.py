#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""契约撰写作业包生成器（S1）—— 按 FLOW 切片，把「写一份契约要用的全部输入」算出来。

为什么要有这个脚本，而不是让撰写人自己去各文件里捞：

1. **批次规模必须算出来，不能手写。** S1 任务卡里写的「FLOW-01 = 39 份（25 可写 / 14 待卡）」
   实测复算不出来（真值 54 份）。手写的批次数字就是一张过期照片（风险 N6）。
2. **卡候选必须来自唯一映射。** 「这条责任有哪些方法卡」有两个语料：
   精选线（vault 146 张）与已装线（`dsh-paper2skills/data/classification.json` 的 1338 张）。
   两线靠 `id`（`Skill-*`）建联，**实测 93 张重合**——正是 F5 逐字继承的那 93 张。
   本脚本把两线并起来给出候选，并把「哪一线」标出来，避免撰写人各引各的。
3. **判据只有一处。** frontmatter / 格 / flows 全部由图谱导出，本脚本只**读**它们；
   卡片候选来自产品侧 classification.json（+ 精选线兜底），不在本脚本里另造一份映射表。

用法::

    python3 build_contract_workpack.py --flow FLOW-01 --batches 6
    python3 build_contract_workpack.py --flow FLOW-01 --check      # 作业包与图谱/卡库自洽吗
    python3 build_contract_workpack.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DEFAULT = REPO_ROOT / "paper2skills-vault/07-资源库/capability-graph.json"
CONTRACTS_DIR = REPO_ROOT / "paper2skills-vault/07-资源库/contracts"
VAULT_CLASSIFICATION = REPO_ROOT / "paper2skills-vault/07-资源库/card-classification.json"
SURVEY = REPO_ROOT / "paper2skills-research/reports/_survey_org_model.md"
PRODUCT_CLASSIFICATION = Path(
    "/Users/lute/project/Magpie-Horch/packages/capabilities/dsh-paper2skills/data/classification.json"
)
INSTALLED_SKILLS = Path.home() / ".dsh/skills"
OUT_DIR_DEFAULT = REPO_ROOT / "paper2skills-research/data/contracts"
BUILDER = Path(__file__).resolve().parent / "build_contracts.py"

# 数据要求五维的 ① —— 唯一合法枚举（与 check_contracts.py 的 ACCESS_ENUM 同源）
ACCESS_ENUM = ("自有埋点", "平台后台", "第三方 API", "需授权", "不可得")

STAGE_ORDER = [f"STG-{i:02d}" for i in range(1, 9)]


def fail(msg: str):
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_builder():
    spec = importlib.util.spec_from_file_location("build_contracts", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BC = load_builder()


def load_json(path: Path, what: str):
    if not path.exists():
        fail(f"{what}不存在：{path} —— 拿不到就炸，不要给默认值")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 材料：§F.3 判定理由 / §F.5 边界条目
# ---------------------------------------------------------------------------
F3_ROW_RE = re.compile(r"^\|\s*(AGT-\d{3})\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|")
F5_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|([^|]*)\|([^|]*)\|")


def parse_survey() -> dict:
    """返回 {责任名: {'class': 'A', 'reason': '…'}} 与 {'boundary': {责任名: (计入A的理由, 降级条件)}}。"""
    if not SURVEY.exists():
        fail(f"调查报告不存在：{SURVEY}")
    text = SURVEY.read_text(encoding="utf-8")
    reason, boundary = {}, {}
    section = None
    for line in text.splitlines():
        if line.startswith("### F."):
            section = line[4:7]
            continue
        if section == "F.3":
            m = F3_ROW_RE.match(line)
            if m:
                _agt, _title, name, cls, why = (x.strip() for x in m.groups())
                if cls in ("A", "B"):
                    reason[name] = {"class": cls, "reason": why}
        elif section == "F.5":
            m = F5_ROW_RE.match(line)
            if m and m.group(1).strip() not in ("责任名",) and not m.group(1).startswith("---"):
                name, why, downgrade = (x.strip() for x in m.groups())
                if name and name != "责任名":
                    boundary[name] = {"counted_as_a_because": why, "downgrade_condition": downgrade}
    if len(reason) != 139:
        fail(f"§F.3 应解析出 139 条 A/B 判定理由，实得 {len(reason)} —— 解析规则与材料形态不符时不许放行")
    return {"reason": reason, "boundary": boundary}


# ---------------------------------------------------------------------------
# 卡候选：精选线 ∪ 已装线
# ---------------------------------------------------------------------------
def card_index(fulltext_ids=None) -> dict:
    """card_index(fulltext_ids) —— `fulltext_ids` 只供自检注入坏输入用（默认从精选线现取）。

    把「哪些卡有全文」做成**可注入**的参数，是为了让「有全文卡」这条判据能被一份坏输入打红；
    若写成模块级常量，自检只能对着真实数据断言，而真实数据本来就满足它 ⇒ 摆设断言。
    """
    prod = load_json(PRODUCT_CLASSIFICATION, "产品侧 classification.json")
    vault = load_json(VAULT_CLASSIFICATION, "精选线 card-classification.json")
    vault_by_id = {x["id"]: x for x in vault["items"]}
    if fulltext_ids is not None:
        vault_by_id = {k: v for k, v in vault_by_id.items() if k in fulltext_ids}
    by_l3: dict = {}

    def add(name, kind, slug, card_id, src_domain, title, path, has_fulltext, l3_all):
        by_l3.setdefault(name, []).append({
            "line": kind,                    # paid=已装线 / curated=仅精选线
            "slug": slug,                    # 已装 slug（J12 ② 认它）
            "card_id": card_id,              # Skill-*（J12 ①/③ 认它）
            "src_domain": src_domain,
            "title": title,
            "path": path,
            "has_fulltext": has_fulltext,    # 是否有全文卡（否则只是 legacy 预览）
            "l3_all": l3_all,
        })

    for it in prod["items"]:
        vid = it.get("id")
        v = vault_by_id.get(vid)
        for name in (it.get("l3") or []):
            add(name, "paid", it.get("slug"), vid, it.get("src_domain"),
                it.get("title", ""), str(INSTALLED_SKILLS / str(it.get("slug")) / "SKILL.md"),
                bool(v), it.get("facets", {}).get("l3_all"))
    # 精选线独有（尚未换底进产品线的 53 张）
    prod_ids = {it.get("id") for it in prod["items"]}
    for v in vault["items"]:
        if v["id"] in prod_ids:
            continue
        for name in (v.get("l3") or []):
            add(name, "curated", None, v["id"], v.get("tech_domain", ""),
                v["id"], v["path"], True, None)
    return by_l3


def installed_card_exists(slug: str) -> bool:
    return (INSTALLED_SKILLS / slug / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# 作业包
# ---------------------------------------------------------------------------
def build(flow: str, batches: int, out_dir: Path, write: bool = True) -> dict:
    graph = BC.load_graph(GRAPH_DEFAULT)
    idx = BC.GraphIndex(graph)
    survey = parse_survey()
    cards = card_index()
    assignments = idx.assignments()
    names = idx.names_of_flow(flow)

    # FLOW 的 8 格（业务步骤与标准产物名 —— 撰写时逐字引用，别自己造词）
    stages = []
    for c in graph["cells"]:
        if c["flow_id"] != flow:
            continue
        stages.append({
            "cell_id": c["cell_id"], "stage_id": c["stage_id"], "stage_name": c["stage_name"],
            "cell_kind": c["cell_kind"], "business_step": c["business_step"],
            "business_output_name": c["business_output_name"], "algorithm_policy": c["algorithm_policy"],
        })
    stages.sort(key=lambda s: s["stage_id"])
    if len(stages) != 8:
        fail(f"{flow} 应有 8 格，实得 {len(stages)}")

    # FLOW 的贡献岗位（= 该 FLOW 的契约所属岗位集合，逐格参与岗位的并集）
    role_ids = sorted({rid for c in graph["cells"] if c["flow_id"] == flow
                       for rid in c["participating_role_ids"]})
    roles = [{"role_id": r["id"], "alias": r.get("alias"), "title": r["title"],
              "domain_id": r["domain_id"]} for r in graph["roles"] if r["id"] in role_ids]
    batch_role_ids = sorted({assignments[n]["role_id"] for n in names})
    if set(batch_role_ids) != set(role_ids):
        fail(f"{flow} 批次岗位 {batch_role_ids} 与逐格参与岗位 {role_ids} 不一致 —— "
             "两处口径不同就必须先解决，不许各写各的")

    contracts = []
    for name in names:
        meta = assignments[name]
        cand = cards.get(name, [])
        paid = [c for c in cand if c["line"] == "paid"]
        curated_only = [c for c in cand if c["line"] == "curated"]
        status = "可写" if cand else "待卡"
        cells = []
        for cid in meta["method_cells"] + meta["rule_cells"]:
            c = next(x for x in graph["cells"] if x["cell_id"] == cid)
            cells.append({"cell_id": cid, "kind": c["cell_kind"], "stage_id": c["stage_id"],
                          "stage_name": c["stage_name"], "business_step": c["business_step"],
                          "business_output_name": c["business_output_name"]})
        contracts.append({
            **{k: meta[k] for k in ("contract_id", "template", "responsibility", "role_id",
                                    "role_title", "domain_id", "plane_id", "serviceability",
                                    "flows", "method_cells", "rule_cells")},
            "file": f"{meta['template']}/{meta['contract_id']}-{name}.md",
            "status": status,
            "rationale_class": survey["reason"][name]["class"],
            "rationale": survey["reason"][name]["reason"],
            "boundary_note": survey["boundary"].get(name),
            "cells": cells,
            "card_candidates": paid + curated_only,
            "cards_paid": len(paid),
            "cards_fulltext": sum(1 for c in cand if c["has_fulltext"]),
        })

    # 分批：按 contract_id 稳定排序后轮转分配，保证同批内模板混合、规模均匀
    contracts.sort(key=lambda c: c["contract_id"])
    groups = [[] for _ in range(batches)]
    for i, c in enumerate(contracts):
        groups[i % batches].append(c)

    pack = {
        "_meta": {
            "what": f"{flow} 的契约撰写作业包（S1 第一批口径）",
            "generator": "build_contract_workpack.py",
            "graph_digest": idx.graph_digest,
            "batch_rule": "批次 = 「能力贡献岗位的 A/B 类责任」，由图谱算出（不手写）",
            "card_rule": ("候选卡 = 已装线(classification.json 1338 张) ∪ 精选线(card-classification.json 146 张)，"
                          "按 id 建联；cards 优先写已装 slug（S12 消费口认它）"),
            "counts": {"total": len(contracts),
                       "A": sum(1 for c in contracts if c["template"] == "A"),
                       "B": sum(1 for c in contracts if c["template"] == "B"),
                       "可写": sum(1 for c in contracts if c["status"] == "可写"),
                       "待卡": sum(1 for c in contracts if c["status"] == "待卡")},
        },
        "flow": flow,
        "stages": stages,
        "roles": roles,
        "batches": [{"batch": i + 1, "contracts": [c["contract_id"] for c in g]} for i, g in enumerate(groups)],
        "contracts": contracts,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{flow.lower()}-workpack.json").write_text(
            json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
        common = render_common(pack, survey, graph)
        (out_dir / f"{flow.lower()}-common.md").write_text(common, encoding="utf-8")
        for i, g in enumerate(groups):
            (out_dir / f"{flow.lower()}-batch-{i + 1:02d}.md").write_text(
                render_batch(pack, g, i + 1), encoding="utf-8")
        print(f"✅ 作业包 → {out_dir}/{flow.lower()}-workpack.json")
        print(f"✅ 公共材料 → {out_dir}/{flow.lower()}-common.md")
        for i in range(batches):
            print(f"✅ 批次 {i + 1:02d} → {out_dir}/{flow.lower()}-batch-{i + 1:02d}.md"
                  f"（{len(groups[i])} 份）")
    print(f"\n{flow}：{pack['_meta']['counts']['total']} 份契约"
          f"（A {pack['_meta']['counts']['A']} / B {pack['_meta']['counts']['B']}）· "
          f"可写 {pack['_meta']['counts']['可写']} / 待卡 {pack['_meta']['counts']['待卡']}")
    return pack


def render_common(pack: dict, survey: dict, graph: dict) -> str:
    stages = "\n".join(
        f"| {s['stage_id']} | {s['stage_name']} | {s['cell_kind']} | {s['business_step']} | "
        f"{s['business_output_name']} |" for s in pack["stages"])
    roles = "、".join(f"{r['role_id']} {r['alias']}·{r['title']}" for r in pack["roles"])
    boundary = "\n".join(
        f"| {n} | {v['counted_as_a_because']} | {v['downgrade_condition']} |"
        for n, v in sorted(survey["boundary"].items()))
    return f"""# {pack['flow']} 契约撰写 · 公共材料（所有批次共用）

> 本文件由 `build_contract_workpack.py` 生成，**不要手改**（改了就与图谱脱钩）。
> 逐份契约的规格在 `{pack['flow'].lower()}-batch-NN.md`。

## 0 先读这四份（按顺序）

1. `paper2skills-vault/07-资源库/contracts/撰写规范.md` —— **作业规程**（v2），硬规则 R1–R10 在那里。
2. `paper2skills-vault/07-资源库/contracts/TEMPLATE-{'A' if True else 'A'}-标定契约.md` 与
   `TEMPLATE-B-完整性契约.md` —— 六段结构与每段的义务。
3. 一份**已写好**的同模板样例（照它的密度写，不要更短）：
   - B 模板：`paper2skills-vault/07-资源库/contracts/v2/B/CTR-B-006-证据复核.md`
   - A 模板：`paper2skills-research/reports/PHASE6-F6F7-契约v1底本存档/A/CTR-A-003-依赖协调.md`
     （⚠️ 这是 **v1** 底本，用于看**密度与取材方式**；它的形态是「只有禁令没有义务」，
     正是 F6 实测判负的那一版 —— **结构照 v2 模板，不要照它**）
4. 你自己的骨架文件（已生成，frontmatter 一个字符都不要动）。

## 1 这个 FLOW 是什么

- **{pack['flow']}** 的能力贡献岗位 {len(pack['roles'])} 个：{roles}
- 8 个阶段（格类型：M = 算法模型接入点，R/D = 只放判据与契约，**不得**说成算法接入）：

| STG | 阶段 | 格 | 业务步骤 | 标准产物（**逐字引用这个名字**） |
|---|---|---|---|---|
{stages}

- `M` 格是该责任**方法接入**的地方；`R/D` 格只放判据。
  ⚠️ 正文里凡出现 `FLOW-0x/STG-0y`，**必须属于你自己 frontmatter 里的 `method_cells` / `rule_cells`**，
  且**不得**把 R/D 格写成「算法接入」——校验器 J3 会打红。

## 2 材料词汇（不要自己造词）

写正文时凡是提到产物、规则、记录，**一律用材料里的原名**：

- 阶段产物：经营信号记录 · 经营Case Charter · FLOW-01 Context Manifest · 经营证据与偏差诊断 ·
  联合经营行动包 · FLOW-01 Assurance Decision · 动作尝试或无动作记录 · 经营关闭记录
- 协同协议八阶段（COLLABORATION-GRAPH / CASE-PROTOCOL-8）：阶段接收记录、Assurance 接收门禁、
  受控动作、Action Intent、NoActionRecord、Pre-Case Exception、Emergency Guard（D-026 紧急保护通道）
- 岗位：Case Control（模型外建单）、Execution Broker（执行）、Case Agent（主岗位）
- 口径与主数据：AGT-045 指标契约与主数据、AGT-046 的质量状态/来源/新鲜度、
  `业务规则字典`、`标签分配结果表`
- 通用工单规则（PB-001 原文）：每项记录**统计窗和新鲜度**；**过期库存信息不能支撑加预算决定**；
  缺关键输入时进入等待或自治异常，**不把缺失值当作零**；已成功动作**不得随整单重跑**

## 2.1 已定决策与业务处境数字（**可直接引用，引时注明出处**）

> 两位试点撰写人都报了同一条：这些数只在方案文档里，没随作业包下发 —— 而缺口处置几乎全靠它们。
> 它们**属于取值来源白名单的第 ① 类**（业务处境/材料已给出的数），可以**直接引用**，
> 但每条引用都要写明是哪一个决策号。

| # | 已定内容 | 对本 FLOW 的直接后果 |
|---|---|---|
| **Q9** | 增量测量的主体＝**自有独立站为主** | FLOW-01 的实例主岗位按渠道四选一：Amazon→AGT-021；**独立站→AGT-023**；其他平台→AGT-024；B2B/零售→AGT-025 |
| **Q10** | 独立站 holdback **没有、可以建，但需先建** | **6 条** A 类契约标 `方法待启用`：增量分析 · 广告实验 · 预算分配 · 投放诊断 · 实验设计 · 内容实验。**不得**写成「先假设有 holdback」 |
| **Q11** | 出海历史＝**2 个完整年度** | 所有「回溯深度」维的默认取值；季节因子可从自身估形状，但**须与品类/行业先验对账**（先验来源＝开放事实 O2） |
| **Q12** | 数据判据＝**本业务数据可得性五维**，与 registry 的 `data_availability` **分开** | 五维见 §3；**不得**把论文的数据可得性抄进来（类别错误） |
| **D4** | 精选线作前台主供给，legacy 降为宽底 + `quality_tier` | 契约 `cards` 优先写**已装 slug**；未换底的 53 张精选卡先按卡名引 |
| **Q2/Q3** | 139 = 151 − 12；按 FLOW 纵向切片 | 本 FLOW 的批次规模由图谱算出（本文件已给） |
| **O1** | 材料自承「**其余 20% 渠道构成未明**」（Amazon 65% / 独立站 15% 之外） | 契约的「适用渠道」只开 4 个值；遇到未定名渠道**按缺口写处置**，不要编第五个渠道 |
| **O2** | **品类/行业季节性先验从哪来**未定（Q11 的 2 个年度不足以稳健分离趋势/季节） | 需求预测 · 生命周期分析 · 促销规划 · 库存分层 · 补货模拟 的标定规则遇到它**如实写缺口 + 处置**，不要编季节性参数 |
| — | 「Amazon 65% / 独立站 15%」是 **GMV 构成**，不是**投放构成** | 不得用它推「该在哪个渠道做增量实验」——那是类别错误 |

### ⚠️ 同质化禁令（试点实测：50 份会写成同一句话）

「先用独立站埋点顶替 + 满 2 个完整年度后替换」这句话对**每一条**责任都成立，
于是它极容易出现在 50 份契约的第三列里 —— 那样的第三列等于没写。要求：

1. 缺值处置必须点名**该责任实际会缺的那一维**（不是每份都缺同一维），并说清**缺了它这条责任算不出什么**。
2. 兜底口径只许用在**确实没有更具体来源**时；有更具体来源（该责任的台账/后台/契约）就写它。
3. 同一句话在 54 份里重复出现 ⇒ 视为未撰写。

## 3 数据源：只能从这些名字里选（§2 ① 必须是五选一）
| ① 取值 | 本业务对应的命名系统（举例，可再具体到字段） |
|---|---|
| `自有埋点` | 独立站埋点、Case/Event Ledger（Case Charter / Stage Acceptance Record / Execution Attempt） |
| `平台后台` | Amazon 卖家后台报表、广告后台、库存台账、账号健康与政策通知 |
| `第三方 API` | 物流承运商轨迹接口、汇率与税费接口 |
| `需授权` | 财务系统对账表、采购台账、供应商产能档案、海关与合规条款库 |
| `不可得` | **合法结论**；写了它 `status` 必须为 `数据待盘`（J6） |

⚠️ **禁止**：写「数据可得」而不点名上面五个之一（视同未填）；拿**论文里的**数据可得性
（registry 的 `data_available`）当本业务的取值 —— 那是**类别错误**（论文数据公不公开 ≠ 本司拿不拿得到）。

## 4 取值只能来自这三类（其余一律不许出现数字）

1. **业务处境里已经给出的**数字（如「近 2 个完整年度」「Amazon 65% / 独立站 15%」）；
2. **材料里已定名的**规则 / 产物 / 岗位（引用其名，不引用其值）；
3. **命名的数据系统**（写成算式：「按〈源〉的〈字段〉算〈量〉」）。

❌ 三种禁止：把卡里的论文参数写成本业务取值；凭空给一个说不出来源的数字；
以「待标定 / 需重新标定 / **临时默认值**」结案（J9 打红）。
✅ 两种必须：**给出取值或区间**（只写算式不给数 ⇒ J13 打红）；**缺口必须带处置**
（「先用什么 + 什么条件下换成什么」）。

### 4.1 三个卡侧陷阱（试点实测，会写错的那种）

1. **`cards` 只写 slug / 卡名，绝不写路径。** 校验器 `find_card` 认三种键：图谱 `cards[]` 里的
   卡名、`~/.dsh/skills/<slug>/SKILL.md`、`paper2skills-vault/<域>/<slug>.md`（**只扫域目录顶层**）。
   精选卡可能在**子目录**里（如 `07-NLP-VOC/00-知识库-Skill卡片/Skill-*.md`）—— 那种情况下
   能过闸靠的是图谱命中，**把路径当 slug 写会直接判红**。
2. **不要把两张卡的数字混成一份取值。** 每张卡的数字属于它自己；`cards` 里列了多张，
   §1 的清单就要能看出哪条来自哪张。
3. **同一责任的不同卡可能互相冲突**（比如一张说「无需历史数据」、另一张的算式恰恰要用历史序列）。
   冲突要么说明为什么选它，要么写进「不得移植」清单 —— 不要沉默地二选一。

## 5 §F.5 的边界条目（若你的责任在表内，正文明写一句它的降级条件）
| 责任名 | 计入 A 的理由 | 降级条件 |
|---|---|---|
{boundary}

## 6 写完必须自验（不许跳过）

```sh
python3 paper2skills-research/scripts/check_contracts.py --file <你写的契约路径>
```

退出码 **0 = 过**，1 = 有判据红（按报错改），2 = 输入没拿到（**不是通过**）。
写完把**实际退出码**与红因（若有）如实回报。
"""


def render_batch(pack: dict, group: list, n: int) -> str:
    parts = [f"# {pack['flow']} 契约撰写 · 批次 {n:02d}（{len(group)} 份）\n",
             f"> 公共材料见 `{pack['flow'].lower()}-common.md`（**先读它**）。"
             f"本批 {sum(1 for c in group if c['template'] == 'A')} 份 A 模板 / "
             f"{sum(1 for c in group if c['template'] == 'B')} 份 B 模板。\n"]
    for c in group:
        cand = c["card_candidates"]
        paid = [x for x in cand if x["line"] == "paid" and x["has_fulltext"]]
        paid_prev = [x for x in cand if x["line"] == "paid" and not x["has_fulltext"]]
        curated = [x for x in cand if x["line"] == "curated"]

        def fmt(xs, limit):
            return "\n".join(
                f"    - `{x['slug'] or x['card_id']}`"
                + (f" · 源卡号 `{x['card_id']}`" if x["slug"] and x["card_id"] else "")
                + f" · {x['src_domain']} · {x['title'][:60]}"
                + (f"\n      - 全文卡：`{x['path']}`" if x["has_fulltext"] and x["slug"]
                   else (f"\n      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 "
                         f"`references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」"
                         f"或「待人工判定」）→ `{x['path']}`" if x["slug"]
                         else f"\n      - 全文卡：`{x['path']}`"))
                for x in xs[:limit]) or "    （无）"

        parts.append(f"""
---

## {c['contract_id']} · {c['responsibility']}

| 项 | 值 |
|---|---|
| 模板 | **{c['template']}**（由算法可服务性 `{c['serviceability']}` 唯一导出） |
| 岗位 | `{c['role_id']}` {c['role_title']} |
| 域 / 面 | `{c['domain_id']}` / `{c['plane_id']}` |
| flows | {', '.join(c['flows'])} |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/{c['file']}` |
| `status` 初值 | **{c['status']}**（{"有候选卡" if c['status'] == '可写' else "无候选卡 —— 这就是扩充工单"}） |

**本责任为什么是 {c['rationale_class']} 类（材料 §F.3 原文，别改判）**：{c['rationale']}
{('- **§F.5 边界条目**：' + c['boundary_note']['counted_as_a_because'] + ' ⇒ 降级条件：' + c['boundary_note']['downgrade_condition']) if c['boundary_note'] else ''}

**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
{chr(10).join(f"| {x['cell_id']} | {x['kind']} | {x['stage_name']} | {x['business_step']} | {x['business_output_name']} |" for x in c['cells'])}

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 {len(paid)} 张
{fmt(paid, 40)}
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 {len(paid_prev)} 张
{fmt(paid_prev, 40)}
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 {len(curated)} 张
{fmt(curated, 40)}

> 候选总数 {len(cand)}。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。
""")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# --check / --selftest
# ---------------------------------------------------------------------------
def check(pack: dict, out_dir: Path) -> int:
    """作业包自洽：批次覆盖、卡候选可解析、status 与候选一致、格号属于本契约。"""
    graph = BC.load_graph(GRAPH_DEFAULT)
    idx = BC.GraphIndex(graph)
    problems, cases = [], 0

    def expect(cond, msg):
        nonlocal cases
        cases += 1
        if not cond:
            problems.append(msg)

    names = idx.names_of_flow(pack["flow"])
    expect(len(pack["contracts"]) == len(names),
           f"作业包契约数 {len(pack['contracts'])} ≠ 图谱批次 {len(names)}")
    flat = [cid for b in pack["batches"] for cid in b["contracts"]]
    expect(sorted(flat) == sorted(c["contract_id"] for c in pack["contracts"]),
           "分批必须恰好覆盖全部契约，不重不漏")
    for c in pack["contracts"]:
        own = set(c["method_cells"]) | set(c["rule_cells"])
        expect({x["cell_id"] for x in c["cells"]} == own,
               f"{c['contract_id']} 的格清单与 frontmatter 不一致")
        expect((c["status"] == "可写") == bool(c["card_candidates"]),
               f"{c['contract_id']} 的 status 与卡候选不一致")
        for cand in c["card_candidates"]:
            if cand["slug"]:
                expect(installed_card_exists(cand["slug"]),
                       f"{c['contract_id']} 候选 slug 未安装：{cand['slug']}")
            else:
                expect((REPO_ROOT / cand["path"]).exists(),
                       f"{c['contract_id']} 候选精选卡路径不存在：{cand['path']}")
        skel = CONTRACTS_DIR / c["file"]
        expect(skel.exists(), f"{c['contract_id']} 骨架文件不存在：{c['file']}")
        if skel.exists():
            fm = BC.read_frontmatter(skel.read_text(encoding="utf-8"))
            expect(BC.norm_val(fm.get("responsibility")) == c["responsibility"],
                   f"{c['contract_id']} 骨架的责任名与作业包不一致")
            expect(BC.norm_val(fm.get("method_cells")) == c["method_cells"],
                   f"{c['contract_id']} 骨架的 M 格与作业包不一致")
    for p in problems:
        print(f"🔴 {p}")
    print(f"作业包自洽：{cases - len(problems)}/{cases} 通过")
    return 0 if not problems else 1


def selftest() -> int:
    """自证：卡候选的判据能被篡改样本打红（否则是摆设断言）。"""
    fails, cases = [], 0

    def expect(cond, label):
        nonlocal cases
        cases += 1
        if not cond:
            fails.append(label)

    cards = card_index()
    all_names = [n for n in cards]
    expect(len(all_names) > 100, f"卡索引应覆盖上百条责任名，实得 {len(all_names)}")
    expect(any(c["line"] == "curated" for cs in cards.values() for c in cs),
           "应存在「仅精选线」的候选（53 张未换底卡）")
    expect(any(not c["has_fulltext"] for cs in cards.values() for c in cs),
           "应存在「只有 legacy 预览」的候选 —— 否则全文卡判据恒真")
    expect(all(c["slug"] for cs in cards.values() for c in cs if c["line"] == "paid"),
           "已装线候选必须有 slug")

    def mut_join():
        """篡改样本①：注入「精选线一张都没有」的坏输入，**已装线**的全文卡判据必须翻转为 False。

        ⚠️ 首版这条写成了「两线 id 有交集吗」—— 那是**对真实数据本来就成立**的断言，
        自检当场打红（与 F2 抓出的 7 处摆设断言同型）。改成注入式后才真的能失败。
        ⚠️ 第二版仍红：`has_fulltext` 对**仅精选线**的候选恒为真（它们本来就有全文），
        于是一起断言等于在测那条恒真分支。判据必须收窄到它真正想测的那一线。
        """
        return any(c["has_fulltext"] for cs in card_index(fulltext_ids=set()).values()
                   for c in cs if c["line"] == "paid")

    expect(not mut_join(), "篡改样本①：注入空全文集后，已装线的「有全文卡」仍为真 ⇒ 判据是摆设")

    # 篡改样本②：批次规模判据 —— 摘掉一个岗位的该 FLOW，规模必须变
    graph = BC.load_graph(GRAPH_DEFAULT)
    flow = "FLOW-01"
    n0 = len(BC.GraphIndex(graph).names_of_flow(flow))
    g2 = json.loads(json.dumps(graph))
    for r in g2["roles"]:
        if r["id"] == "AGT-032":
            r["flows"] = [f for f in r["flows"] if f != flow]
    expect(len(BC.GraphIndex(g2).names_of_flow(flow)) != n0,
           "篡改样本②：摘掉 AGT-032 的 FLOW-01 后批次规模未变 ⇒ 规模判据未被测到")

    print(f"自检：{cases - len(fails)}/{cases} 通过")
    for f in fails:
        print(f"🔴 {f}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="契约撰写作业包生成器")
    ap.add_argument("--flow", default="FLOW-01")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    pack = build(args.flow, args.batches, Path(args.out_dir), write=not args.no_write)
    if args.check:
        return check(pack, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
