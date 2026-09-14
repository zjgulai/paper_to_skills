#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_search_queries.py — PHASE6 S4：把「缺口」翻译成「检索式」

## 为什么需要这一步

F3 的靶区工单（31 条）是一张**缺口清单**，不是一张**采购单**。
没有 S4，缺口账只是一张纸：它说「线索评估 零供给」，但没人知道该拿什么词去捞。

S4 做的就是这一步 —— **每条靶区工单一份三段式检索式（正向 / 负向 / 约束）**，
并且这份检索式**必须能被 `candidate_filter.py` 直接消费**（同一个配置格式，不另造一套）。

## 硬顺序（方案 §8.1）

「S4 必须排在 S1 之后 —— **入卡挂载点就是契约**，契约没有就没有挂载点。」
⇒ 每条检索式都必须指明它服务的 **责任名 + L3 + 靶区位次 + 该责任已交付的契约 id**
（`CTR-A-xxx` / `CTR-B-xxx`），且字段名与 `check_contracts.py` 认的
`responsibility` / `contract_id` 对齐。产出候选卡之后，卡要能挂回那份契约。

## 三段式语义（与 `candidate_filter.py` 逐字一致）

    保留 = 命中正向词（收割阶段保证） ∧ 未命中该域负向词 ∧ 满足该域约束词

⚠️ **负向词按域生效，不是全局清单**。仓库已实测：同一个词在不同域含义相反 ——
`trial`（01/02 是 RCT 方法论词，14 是临床词）、`cohort`（14 是同期群，医学是队列研究）、
`ad`（13 是广告，医学是 Alzheimer/ADHD）。无脑全局负向会**误杀方法论论文**。
本脚本因此**不自造负向词清单**：它从 `candidate_filter.py` **读**（`GLOBAL_NEGATIVE` /
`DOMAIN_NEGATIVE` / `DOMAIN_CONSTRAINT`），并逐条断言两侧相等。
两侧一旦漂移（有人改了 filter 没改词库，或反过来）即判红。

## 输入（全部只读，本脚本不改任何上游文件）

| 输入 | 用途 | 缺了怎么办 |
|---|---|---|
| `paper2skills-research/data/gap-ledger.json` | 31 条靶区工单（位次 / 责任名 / 岗位 / 域 / 结构空白 / 契约锚） | **exit 2** |
| `paper2skills-vault/07-资源库/关键词库-v2.md` | 三段式词库（正向词的第三来源 + 约束词兜底 + 同词异义教训） | **exit 2** |
| `paper2skills-research/scripts/candidate_filter.py` | **负向词按域生效的可执行部件**（唯一事实源） | **exit 2** |
| `paper2skills-vault/07-资源库/contracts/{A,B}/CTR-*.md` | 入卡挂载点（`responsibility` → `contract_id`） | **exit 2** |
| `paper2skills-research/scripts/arxiv_harvest.py` | 正向词组（实际发出去的查询串） | **exit 3**（是内部常量，不是用户数据） |

## 输出

- `paper2skills-research/data/search-queries.json` —— 机读产物，`--json-out` 可改路径
- `paper2skills-research/reports/PHASE6-S4-缺口驱动检索式.md` —— 人类可读

## 判据（**每条都必须能失败**，各配篡改样本；见 `--selftest`）

| # | 判据 | 反面（会被抓的错） |
|---|---|---|
| J1 | 输入齐备性 | 任一输入缺失 = **exit 2**，不是「通过」 |
| J2 | 覆盖率 = 生成数 / 工单数；生成 0 条**必须判红** | 「一条都没生成」时报「全部通过」 |
| J3 | 结构空白显式排除 + 两种口径的条数逐个对上 | 给走 SOP 的条目生成检索式 |
| J4 | 检索式可复算：同输入两次运行**逐字节相同** | 字典序漂移 / 时间戳进产物 |
| J5 | 改一条工单（责任名 / 域）⇒ 对应检索式**必须变**；并配反向控制（白改必须不变） | 假仪表：改什么都不变 |
| J6 | 不同工单的检索式彼此不同（相似度分布 + 空壳计数，单边判据） | 模板套壳：31 条只换责任名 |
| J7 | **负向词按域生效**：从 filter **算**出分歧矩阵，逐条断言 | 无脑全局负向（误杀方法论论文） |
| J8 | 产物可被 `candidate_filter.py` 直接消费（键名 / 域名 / 词组格式逐项相等） | 另造一套格式 |
| J9 | 契约挂载点：责任名 → `contract_id`，键名与 `check_contracts.py` 对齐 | 自己发明键名 |
| J10 | 排序权重**不进任何判定**（分数只被读来排序，不进判据） | 风险 N1 |

退出码（本仓库约定）：**0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 内部错误（≠ 判红）**

用法：
    python3 build_search_queries.py --check
    python3 build_search_queries.py --selftest
    python3 build_search_queries.py --json-out paper2skills-research/data/search-queries.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "paper2skills-research"
DATA = RESEARCH / "data"
REPORTS = RESEARCH / "reports"

#: ⚠️ 本脚本是 **S4 的产物生成器**，不是门禁脚本 —— 它**不在**
#: `build_gap_ledger.py` 的 `GATE_SCRIPTS` 清单里。所以它读工单是允许的；
#: 但排序权重仍然**只用于排序**（J10 用本文件自身的 AST 复核，见 `judge_j10`）。
LEDGER = DATA / "gap-ledger.json"
KEYWORDS = REPO / "paper2skills-vault" / "07-资源库" / "关键词库-v2.md"
FILTER_PY = RESEARCH / "scripts" / "candidate_filter.py"
HARVEST_PY = RESEARCH / "scripts" / "arxiv_harvest.py"
CONTRACTS_DIR = REPO / "paper2skills-vault" / "07-资源库" / "contracts"
OUT_JSON = DATA / "search-queries.json"
OUT_MD = REPORTS / "PHASE6-S4-缺口驱动检索式.md"

EXIT_OK, EXIT_RED, EXIT_NOINPUT, EXIT_INTERNAL = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# 0. 责任名 → 检索域（**显式决策表**，不是推断）
# ---------------------------------------------------------------------------
#: ⚠️ 这张表是**人的决策**，必须逐条可复核，所以它写死在代码里而不是猜出来。
#:
#: 为什么不能自动推：`关键词库-v2.md` 的域词条是**英文**，L3 责任名是**中文**；
#: 实测用「中文词 ∩ 英文词」做词面匹配，31 条**全部得 0 分**（见 `--selftest` 的
#: `probe_lexical_mapping_is_blind`）—— 那个仪器看不见这种对应关系。
#: 因此改用一张可审计的决策表，并**用实测把它的每条选择钉住**：
#:
#:   (a) **卡端实测**：同名 L3 在精选 146 张卡里落在哪个技术域（`card-classification.json`）；
#:   (b) **产品端实测**：同名 L3 在已装线 1338 张卡的 `src_domain` 分布
#:       （`dsh-paper2skills/data/classification.json`），再经 `DOMAIN_BRIDGE` 折到 filter 域；
#:   (c) `candidate_filter.py` 的 `DOMAIN_NEGATIVE` / `DOMAIN_CONSTRAINT` 覆盖面。
#:
#: 每条的选择理由写在 `NOTE` 里（报告里逐条打印）。**未列入的 L3 一律判红** ——
#: 不允许静默回落（回落 = 生成一条与责任无关的检索式，比不生成更坏）。
L3_DOMAINS: dict[str, list[str]] = {
    # 位次 1 · 渠道经营 · 零供给（连 legacy 预览都没有）
    "线索评估": ["05-推荐系统", "13-广告分析"],
    # 位次 2/5/6 · 财务与合规 · 对账与差异
    "渠道对账": ["09-DataAgent-LLM", "12-ML基础"],
    "差异追踪": ["12-ML基础", "02-A_B实验"],
    "收入与费用核对": ["09-DataAgent-LLM", "08-知识图谱"],
    # 位次 3/14 · 渠道经营 · 经营动作
    "行动组合": ["13-广告分析", "15-营销投放分析"],
    "平台运营": ["13-广告分析", "15-营销投放分析"],
    # 位次 4/8 · 财务与合规 · 预算与资金
    "经营预算": ["03-时间序列", "15-营销投放分析"],
    "资金预测": ["03-时间序列", "04-供应链"],
    # 位次 7 · 供应与履约 · 生命周期
    "生命周期分析": ["04-供应链", "05-推荐系统"],
    # 位次 9/10/15 · **结构性空白**（走 SOP，不占检索预算；本表只为「若解禁」留位）
    "BOM与材料分析": ["04-供应链"],
    "商业案例": ["15-营销投放分析"],
    "研发计划": ["10-MAS"],
    # 位次 11 · 数据与AI运行 · 主数据映射
    "账号商品映射": ["08-知识图谱", "16-智能体工程"],
    # 位次 12 · 渠道经营 · 账号健康
    "账号诊断": ["12-ML基础", "14-用户分析"],
    # 位次 13 · 供应与履约 · 供需
    "供需协调": ["04-供应链", "10-MAS"],
    # 位次 16 · 品牌与增长 · 反馈
    "品牌反馈": ["07-NLP-VOC", "14-用户分析"],
    # 位次 17/18 · 产品与创新 · 验证
    "可用性验证": ["02-A_B实验", "06-增长模型"],
    "测试方案": ["02-A_B实验", "10-MAS"],
    # 位次 19 · 经营与组织 · 资源
    "资源情景比较": ["15-营销投放分析", "04-供应链"],
    # 位次 20 · 财务与合规 · 知产
    "知识产权检索": ["08-知识图谱", "07-NLP-VOC"],
    # 位次 21 · 品牌与增长 · 达人
    "达人筛选": ["05-推荐系统", "13-广告分析"],
    # 位次 22/25/26/30/31 · 供应与履约 · 履约
    "履约异常": ["04-供应链", "06-增长模型"],
    "退货分流": ["04-供应链", "05-推荐系统"],
    "订单协调": ["04-供应链", "10-MAS"],
    "到货异常追踪": ["04-供应链", "16-智能体工程"],
    "履约跟踪": ["04-供应链", "09-DataAgent-LLM"],
    # 位次 23 · 经营与组织 · 审计
    "抽样审计": ["12-ML基础", "07-NLP-VOC"],
    # 位次 24 · 供应与履约 · 质量
    "质量分析": ["04-供应链", "02-A_B实验"],
    # 位次 27 · 产品与创新 · 趋势
    "趋势监测": ["06-增长模型", "03-时间序列"],
    # 位次 28 · 品牌与增长 · 复购
    "复购实验": ["01-因果推断", "02-A_B实验"],
    # 位次 29 · 供应与履约 · 供应商
    "供应商评估": ["04-供应链", "08-知识图谱"],
}

#: L3 选择理由（进报告，逐条可复核）。键必须与 `L3_DOMAINS` 完全一致。
L3_NOTE: dict[str, str] = {
    "线索评估": "线索打分与排序 = 推荐/排序问题；B2B 线索优先级含广告/渠道口径。",
    "渠道对账": "对账是「多源记录对齐 + 差异定位」，先走数据代理（Text-to-SQL/对账 Agent），"
                "再用统计口径判差异显著性。",
    "差异追踪": "差异定位 = 统计检验与异常检测（ML 基础主口径）；「差异是否显著」用实验设计口径。",
    "收入与费用核对": "核对以数据/账目系统直出为主（数据代理主口径）+ 实体对齐（知识图谱次口径）。",
    "行动组合": "经营动作组合的取舍 = 预算分配/投放效率问题（广告分析主口径）。",
    "平台运营": "多平台运营动作同样落在投放与促销口径（广告 + 营销投放）。",
    "经营预算": "预算 = 滚动预测（时序主口径）+ 投入产出（营销投放次口径）。",
    "资金预测": "现金流/账期预测 = 时序主口径；应收应付与采购周期挂在供应链次口径。",
    "生命周期分析": "商品生命周期 = 库存/效期/长库龄（供应链主口径）+ 生命周期阶段与序列建模（推荐系统次口径）。",
    "BOM与材料分析": "**结构性空白**（材料清单 §4 严口径）。若解禁，唯一可检索口径是供应链的物料/物料表。",
    "商业案例": "**结构性空白**。若解禁，商业论证 = 投入产出与促销投放口径。",
    "研发计划": "**结构性空白**。若解禁，排期与资源编排落在多 Agent 编排口径（RCPSP 类求解）。",
    "账号商品映射": "账号↔商品↔Listing 的实体映射 = 实体链接与知识图谱主口径 + 工具化映射 Agent（智能体工程）。",
    "账号诊断": "账号健康诊断 = 指标异常检测（ML 基础主口径）+ 卖家行为分析（用户分析次口径）。",
    "供需协调": "供需匹配 = 供应链主口径；跨主体协同编排（供应计划与到货对齐）用多 Agent 次口径。",
    "品牌反馈": "品牌反馈 = 评论/舆情挖掘（NLP-VOC 主口径），落到用户侧行为时用用户分析。",
    "可用性验证": "可用性验证 = 任务级实验与统计检验（A/B 主口径）+ 留存/采纳度量（增长模型次口径）。",
    "测试方案": "组合测试/抽样方案 = 实验设计主口径（pairwise/正交表）；用例生成与执行编排用多 Agent。",
    "资源情景比较": "情景比较 = 投入产出的情景/灵敏度分析（营销投放主口径）+ 资源约束与产能（供应链次口径）。",
    "知识产权检索": "知产检索 = 实体/关系抽取 + 图检索（知识图谱主口径）+ 文本检索与相似度（NLP-VOC 次口径）。",
    "达人筛选": "达人筛选 = 匹配与排序（推荐系统主口径）+ 效果归因（广告分析次口径）。",
    "履约异常": "履约异常 = 供应链主口径 + 履约失败与流失预警（增长模型次口径）。",
    "退货分流": "退货分流 = 逆向物流与退货成本（供应链主口径）+ 分单路由排序（推荐系统）。",
    "订单协调": "订单协调 = 采购/交付协同（供应链主口径）+ 多主体编排（MAS 次口径）。",
    "到货异常追踪": "到货偏差 = 供应链主口径 + 实体/轨迹状态追踪的工具化 Agent（智能体工程次口径）。",
    "履约跟踪": "在途跟踪 = 供应链主口径 + 在途状态查询与报表（数据代理/Text-to-SQL 次口径）。",
    "抽样审计": "抽样审计 = 统计抽样与检验（ML 基础主口径）+ 单据与凭据文本核对（NLP-VOC 次口径）。",
    "质量分析": "质量分析 = 质量/根因（供应链主口径）+ 抽样与统计判定（实验设计次口径）。",
    "趋势监测": "趋势监测 = 品类趋势预测（增长模型主口径）+ 时序次口径。",
    "复购实验": "复购实验 = 增量/因果识别（因果推断主口径）+ 实验设计（A/B 次口径）。",
    "供应商评估": "供应商评估 = 多准则决策与供应风险（供应链主口径）+ 供应链知识图谱（图谱次口径）。",
}

#: 已装线 1338 张卡的 `src_domain` 里有 9 个域在 filter 词汇表**之外**。
#: 这里只登记 **L3_DOMAINS 实际会用到** 的桥接；每条都写明语义缺口。
DOMAIN_BRIDGE: dict[str, str] = {
    "17-价格优化": "15-营销投放分析",     # 价格弹性/促销是同一族
    "18-物流履约": "04-供应链",           # 物流履约有 44 张卡，filter 只认 04
    "23-运营财务": "03-时间序列",         # 财务预测/账期 = 时序；语义缺口：**无专门财务域**
}

#: **交叉项取材的域族表**：已装线卡片的 `src_domain` → 它属于哪个检索域族。
#:
#: ⚠️ 与 `DOMAIN_BRIDGE` **不是一回事**，刻意分开：
#:   · `DOMAIN_BRIDGE` 管的是 **query_domains 本身**（filter 词汇表里没有的域怎么折）；
#:   · `SRC_DOMAIN_FAMILY` 管的是**取材**（哪些源的卡算「与本检索域同一技术面」）。
#:
#: 表里**没有**的 `src_domain` 走 **`"*"` 通配**（该 L3 的卡一律算同族）——
#: 通配是**有意的设计**而不是兜底：同一个 L3 的卡默认描述同一条责任的同一技术面，
#: 只有能证明「它属于另一个技术面」时才用这张表排除。每次运行会**统计通配次数**并
#: 打进产物，通配占比升高 = 取材变松，报告里看得见。
SRC_DOMAIN_FAMILY: dict[str, str] = {
    # 16 个与 filter 同名的域不必列（函数里 `d in DOMAIN_DIR` 直接自匹配）
    "17-价格优化": "15-营销投放分析",
    "18-物流履约": "04-供应链",
    "19-风控反欺诈": "12-ML基础",
    "20-AI视频生成": "07-NLP-VOC",
    "21-合规决策": "12-ML基础",
    "22-数据采集工程": "08-知识图谱",
    "23-运营财务": "03-时间序列",
    "24-标签工程": "08-知识图谱",
    "25-搜索流量工程": "05-推荐系统",
}

#: 结构空白岗位（材料 §4 严口径 L3 名单所属岗位：硬件研发 / 工业设计 / 工程制造）。
STRUCTURAL_BLANK_ROLES = ("AGT-009", "AGT-010", "AGT-011", "AGT-012", "AGT-013")

#: 约束词兜底：`candidate_filter.py` 对 10-MAS / 11-AI人文 **没有配约束词**，
#: 但 S4 的三段式要求「宽词必须与约束词共现才收」，所以这两域必须由词库补。
#: 逐条抄自 `关键词库-v2.md`（J8 会回查该文件，抄错即判红）。
CONSTRAINT_FALLBACK: dict[str, list[str]] = {
    "10-MAS": ["llm", "agent", "task", "tool"],
    "11-AI人文": ["metaphor", "analogy", "humanities", "philosophy", "well-being"],
}


# ---------------------------------------------------------------------------
# 1. 输入加载（任何一项拿不到 = exit 2，不是「通过」）
# ---------------------------------------------------------------------------
class InputMissing(Exception):
    """输入没拿到。**不是判红** —— 与「查过了没问题」必须可区分。"""


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise InputMissing(f"可执行部件不存在：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # ⚠️ 必须注册 sys.modules：CPython 3.14 下 dataclass + `from __future__ import
    # annotations` 会在 dataclasses._is_type 里查 sys.modules[cls.__module__]。
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha16(path: Path) -> str:
    """输入指纹（前 16 位）。文件不存在返回 `"missing"` —— 不假装有指纹。"""
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_inputs() -> dict:
    """读全部输入。缺任何一项抛 `InputMissing` ⇒ main 返回 exit 2。"""
    if not LEDGER.is_file():
        raise InputMissing(f"靶区工单不存在：{LEDGER}")
    if not KEYWORDS.is_file():
        raise InputMissing(f"三段式词库不存在：{KEYWORDS}")
    if not CONTRACTS_DIR.is_dir():
        raise InputMissing(f"契约目录不存在：{CONTRACTS_DIR}")

    cf = _load_module(FILTER_PY, "_s4_candidate_filter")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    kw_text = KEYWORDS.read_text(encoding="utf-8")
    contracts = read_contracts()

    # arxiv_harvest.py 是**内部常量**（不是用户数据）：它坏了是我的问题，不是输入的问题。
    try:
        ah = _load_module(HARVEST_PY, "_s4_arxiv_harvest")
        harvest_groups = ah.QUERY_GROUPS
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"读取 harvest 查询组失败（内部错误）：{e}") from e

    return {
        "ledger": ledger,
        "kw_text": kw_text,
        "cf": cf,
        "harvest_groups": harvest_groups,
        "contracts": contracts,
    }


def read_contracts() -> dict[str, dict]:
    """读 139 份正式契约（只在 `contracts/A` 与 `contracts/B`，同 check_contracts）。

    ⚠️ 别处的 `CTR-*.md`（`contracts/v2/` 的 F7 试点样本）**不计入** ——
    这正是 `check_contracts.py` 台账 #26/#27 那条缺陷的修法。
    """
    out: dict[str, dict] = {}
    for tpl in ("A", "B"):
        d = CONTRACTS_DIR / tpl
        if not d.is_dir():
            continue
        for p in sorted(d.glob("CTR-*.md")):
            fm = {}
            m = re.match(r"^---\n(.*?)\n---\n", p.read_text(encoding="utf-8"), re.S)
            if m:
                for line in m.group(1).splitlines():
                    if ":" in line and not line.startswith(" "):
                        k, v = line.split(":", 1)
                        fm[k.strip()] = v.strip()
            out[p.name] = {
                "contract_id": fm.get("contract_id", ""),
                "responsibility": fm.get("responsibility", ""),
                "role_id": fm.get("role_id", ""),
                "domain_id": fm.get("domain_id", ""),
                "template": fm.get("template", tpl),
                "path": str(p.relative_to(REPO)),
            }
    return out


# ---------------------------------------------------------------------------
# 2. 三段式检索式构造
# ---------------------------------------------------------------------------
def harvest_domain_of(harvest_groups: dict, domain: str) -> str | None:
    """harvest 的域标签与 filter 的域标签**实测有 3 处不一致**（见 J8）。

    filter 认 `07-NLP-VOC` / `09-DataAgent-LLM` / `15-营销投放分析`，
    harvest 发的是 `07-VOC舆情` / `09-DataAgent` / `15-营销投放`。
    本函数把 filter 域名映射回 harvest 的组名 —— 映射写死并**逐条验证存在**。
    """
    if domain in harvest_groups:
        return domain
    bridge = {
        "07-NLP-VOC": "07-VOC舆情",
        "09-DataAgent-LLM": "09-DataAgent",
        "15-营销投放分析": "15-营销投放",
    }
    cand = bridge.get(domain)
    if cand and cand in harvest_groups:
        return cand
    return None


def terms_of(groups: list[str]) -> list[str]:
    """从 harvest 的查询组里取出**检索词**（去掉字段前缀与布尔运算符）。

    例：`abs:"uplift modeling" OR abs:"heterogeneous treatment effect"`
        → `["uplift modeling", "heterogeneous treatment effect"]`
    保持首次出现顺序（确定性），不排序 —— 排序会丢掉「哪条查询先发」这个信息。
    """
    seen: list[str] = []
    for g in groups:
        for t in re.findall(r'"([^"]+)"', g):
            t = " ".join(t.split())
            if t and t not in seen:
                seen.append(t)
    return seen


def kw_positive_terms(kw_text: str, domain: str) -> list[str]:
    """从 `关键词库-v2.md` 的该域词条里取反引号词（正向词的**第三来源**）。

    解析失败（找不到该域小节）返回空列表 —— 由 J8 断言「第三来源必须非空」抓出来，
    而不是在这里静默兜底。
    """
    m = re.search(rf"\n### {re.escape(domain)}\n(.*?)(?=\n### |\Z)", kw_text, re.S)
    if not m:
        return []
    sec = m.group(1)
    # 只取「正向」段（到「约束词」/「本域负向」/「补充正向」/「注意」为止）
    head = re.split(r"\n- \*\*(?:约束词|本域负向|注意|AI 方法)", sec)[0]
    out: list[str] = []
    for t in re.findall(r"`([^`]+)`", head):
        t = " ".join(t.split())
        if t and t not in out:
            out.append(t)
    return out


EN_STOP = {
    "the", "a", "an", "of", "for", "and", "in", "on", "with", "to", "at", "by",
    "from", "as", "is", "are", "skill", "p2s", "via", "using", "based", "its",
}


def junction_terms(domains: list[str], responsibility: str,
                   legacy: dict[str, list[tuple[str, str]]], limit: int = 4) -> list[str]:
    """**交叉项**：把该域的宽词收紧到「这条责任」上的英文检索词。

    ⚠️ 这里不是「换个名字」就算数。交叉项取的是**实测挂过这条 L3 的已装线卡**的实际
    英文标题片段（`p2s-*` slug 的英文部分），**且卡的来源技术域必须与本次检索域相符**
    （`DOMAIN_BRIDGE` + 别名归一后比较）。

    为什么要加「同域」这一条 —— **实测撞出来的**：第一版只按 L3 取卡，于是
    「测试方案」（域 `02-A_B实验`）与「资源情景比较」（域 `15-营销投放分析`）
    取到了同一批卡的标题，交叉项 Jaccard = **1.0**，被 J6 当场判成「只换责任名」。
    它们挂的确实是同一条 L3，但服务的技术面不同 ⇒ 拿到的词不该相同。
    换句话说：**同 L3 ≠ 同技术面**，交叉项因此必须**双键**（L3 × 技术域）取。

    没有匹配卡的责任（结构空白 3 条 + 线索评估）返回空列表，在报告里点名，不造假词。
    """
    wanted = {DOMAIN_BRIDGE.get(d, d) for d in domains}
    toks: Counter = Counter()
    for title, tech in legacy.get(responsibility, []):
        fam = SRC_DOMAIN_FAMILY.get(tech, tech if tech in DOMAIN_BRIDGE else "*")
        if fam != "*" and fam not in wanted:
            continue
        # ⚠️ 第一版只取**第一个** `—`/`-` 之前的片段，实测漏词：
        #    「Skill-Cash-Conversion-Cycle-Optimization」被切在第一个连字符上，
        #    得到 `skill` 一个词（<2），整张卡静默丢弃。改为在**所有**分段里挑。
        best: str | None = None
        for seg in re.split(r"[—–|:]", title):
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", seg)
                     if w.lower() not in EN_STOP]
            if not (2 <= len(words) <= 6):
                continue
            if not any(len(w) >= 4 for w in words):
                continue
            phrase = " ".join(w.lower() for w in words)
            # 同一条标题里可能有多段合格（含中文段被跳过的情况），取**最长**的一段
            if best is None or len(phrase) > len(best):
                best = phrase
        if best is None:
            continue
        toks[best] += 1
    if not toks:
        return []
    # 频次降序 → 词长升序 → 字典序（三段都是全序，保证确定性）
    ranked = sorted(toks.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return [ph for ph, _ in ranked[:limit]]


def build_one(cf, item: dict, harvest_groups: dict, kw_text: str,
              legacy_titles: dict[str, list[tuple[str, str]]],
              contract_by_key: dict) -> dict:
    """一条工单 → 一份三段式检索式 + 契约挂载点。纯函数，无时间戳、无随机。"""
    l3 = item["l3"]
    domain = item["domain"]
    domains = L3_DOMAINS.get(l3)
    if not domains:
        raise KeyError(f"L3 不在 L3_DOMAINS 决策表里：{l3!r}（不许静默回落）")

    # ---- 正向词：harvest 实发查询串 + 词库第三来源 + 本责任的交叉项 ----
    positives: list[str] = []
    sources: dict[str, list[str]] = {}
    for d in domains:
        hd = harvest_domain_of(harvest_groups, d)
        if hd is None:
            raise KeyError(f"harvest 里没有域 {d!r} 的查询组")
        t = terms_of(harvest_groups[hd])
        sources[f"harvest:{d}"] = t
        positives.extend(t)
    for d in domains:
        t = kw_positive_terms(kw_text, d)
        sources[f"keyword-lib:{d}"] = t
        positives.extend(t)
    jt = junction_terms(domains, l3, legacy_titles)
    fams = [SRC_DOMAIN_FAMILY.get(tech, tech if tech in DOMAIN_BRIDGE else "*")
            for _, tech in legacy_titles.get(l3, [])]
    sources["junction"] = jt
    sources["junction_provenance"] = {
        "n_cards": len(legacy_titles.get(l3, [])),
        "n_wildcard_cards": sum(1 for f in fams if f == "*"),
        "n_same_family_cards": sum(1 for f in fams if f != "*" and f in set(domains)),
        "empty": not jt,
    }
    # 正向词必须**非空**且**去重后保持首次出现顺序**
    seen: list[str] = []
    for t in positives + jt:
        if t and t not in seen:
            seen.append(t)
    if not seen:
        raise KeyError(f"{l3} 的正向词为空 —— 检索式没有正向词就不是检索式")

    # ---- 负向词 / 约束词：**从 candidate_filter 读**（唯一事实源，不另抄一份）----
    negative = [dict.fromkeys(cf.GLOBAL_NEGATIVE)]
    negative[0] = list(dict.fromkeys(cf.GLOBAL_NEGATIVE))
    scope = []
    for d in domains:
        dn = list(dict.fromkeys(cf.DOMAIN_NEGATIVE.get(d, [])))
        if dn:
            scope.append({"domain": d, "terms": dn})
            negative.append(dn)
    merged_neg: list[str] = []
    for lst in negative:
        for t in lst:
            if t not in merged_neg:
                merged_neg.append(t)

    constraint = list(dict.fromkeys(
        [c for d in domains for c in cf.DOMAIN_CONSTRAINT.get(d, [])]
        + [c for d in domains for c in CONSTRAINT_FALLBACK.get(d, [])]
    ))
    if not constraint:
        raise KeyError(f"{l3} 的约束词为空 —— 宽词无约束 = 跨行业误命中")

    strict = [d for d in domains if d in cf.STRICT_ECOMMERCE_DOMAINS]

    # ---- 契约挂载点（键名与 check_contracts.py 对齐）----
    mount = contract_by_key.get((item["role_id"], l3))
    contract_id = mount["contract_id"] if mount else None
    serviceability = item["serviceability"]

    return {
        "date": "2026-09-13",
        "responsibility": l3,
        "l3": l3,
        "role_id": item["role_id"],
        "role_title": item["role_title"],
        "domain": domain,
        "serviceability": serviceability,
        "query_domains": domains,
        "rank": item["rank"],
        "structural_blank": bool(item["structural_blank"]),
        "eligible_for_retrieval": not item["structural_blank"],
        "contract_id": contract_id,
        "contract_role_id": mount["role_id"] if mount else None,
        "contract_match": (
            "role_id+l3" if mount and mount["role_id"] == item["role_id"]
            else ("l3-only" if mount else "none")
        ),
        "flows": list(item["flows"]),
        "m_cells": list(item["m_cells"]),
        "positive": seen,
        "positive_sources": sources,
        "negative_scope": scope,
        "negative": merged_neg,
        "constraint": constraint,
        "strict_ecommerce_domains": strict,
        "harvest_domain_bridge": {d: harvest_domain_of(harvest_groups, d) for d in domains},
    }


#: 契约挂载点的键：**先按 (role_id, l3)**，同名 L3 出现在多个岗位时不许静默取第一个。
def contract_index(contracts: dict) -> dict:
    by_key: dict[tuple, dict] = {}
    dup = []
    for c in contracts.values():
        k = (c["role_id"], c["responsibility"])
        if k in by_key:
            dup.append((k, by_key[k]["contract_id"], c["contract_id"]))
        else:
            by_key[k] = c
    return {"by_key": by_key, "duplicates": dup}


# ---------------------------------------------------------------------------
# 3. 相似度（判据 J6 的口径，定义在这里而不是散在报告里）
# ---------------------------------------------------------------------------
def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _pool_of(queries: list[dict], key: str) -> list[str]:
    """从 `"<rank>:<责任名>"` 取回该条目的检索域池（顺序无关，比较前排序）。"""
    rk = int(key.split(":", 1)[0])
    for q in queries:
        if q["rank"] == rk:
            return list(q["query_domains"])
    return []


def similarity_report(queries: list[dict]) -> dict:
    """两两相似度分布 + **空壳计数**。

    ⚠️ 本仓库纪律：「判据只写单边等于没有判据」。所以这里不只报「相似度不高」，
    还报**最相似的那一对是谁**（上界）与**有多少条是只换责任名的空壳**（下界），
    两个方向都要有数，单边的「平均相似度 0.3」不构成判据。
    """
    jun = {q["rank"]: set(q["positive_sources"]["junction"]) for q in queries}
    #: ⚠️ **空交叉项的条目必须从「交叉项两两比对」里剔除，并单独计数** ——
    #: 否则「两条都没有交叉项」会被算成 Jaccard = 1.0（两条空集完全重合），
    #: 报出一对**根本不存在**的「只换责任名」假红。实测撞到过：
    #: 第一版把「线索评估」（legacy 卡 0 张）与「测试方案」配成了最相似的一对。
    #: **空集之间的相似度没有定义**，不是 1.0。
    empty = sorted(q["rank"] for q in queries if not jun[q["rank"]])
    nonempty = [q for q in queries if jun[q["rank"]]]
    pairs, pairs_full = [], []
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            a, b = queries[i], queries[j]
            row2 = {"a": f"{a['rank']}:{a['responsibility']}",
                    "b": f"{b['rank']}:{b['responsibility']}",
                    "full": round(jaccard(a["positive"], b["positive"]), 4)}
            pairs_full.append(row2)
            if a["rank"] in empty or b["rank"] in empty:
                continue                      #: 交叉项未定义 ⇒ 不进交叉项统计
            pairs.append(dict(row2, junction=round(
                jaccard(sorted(jun[a["rank"]]), sorted(jun[b["rank"]])), 4)))
    pairs.sort(key=lambda p: (-p["junction"], p["a"], p["b"]))
    pairs_full.sort(key=lambda p: (-p["full"], p["a"], p["b"]))
    #: **空壳** = 交叉项非空、却被另一条的交叉项完全包含（自己没带来任何独有交叉项）。
    #: 这是「只换了责任名」的机器口径：域词一样、交叉项也被人覆盖 ⇒ 壳。
    #: 交叉项**为空**的条目不在此列 —— 那是「没料」而不是「套壳」，用
    #: `empty_junction_count` 单独计（两类错要分开报，不能被平均掉）。
    shells = []
    for q in queries:
        mine = jun[q["rank"]]
        if not mine:
            continue
        for o in queries:
            if o["rank"] == q["rank"]:
                continue
            if mine <= jun[o["rank"]]:
                shells.append({"rank": q["rank"], "responsibility": q["responsibility"],
                               "covered_by": o["responsibility"],
                               "junction": sorted(mine)})
                break
    #: **同一检索域池**的条目对 —— 全向量 Jaccard 天然高的那一批。
    #: 这**不是缺陷**：它们本来就该捞同一批论文，差别在交叉项。
    #: 但它必须**被数出来并钉住条数**，否则「全向量最像的一对」会变成一个
    #: 无法解释的数字（本仓库纪律：判决必须能解释它为什么是这个值）。
    #: 比较用**集合**：域的顺序记的是「主/次优先级」，不是「捞的是哪一批」——
    #: 两条只要域集合相同，命中的 arXiv 查询组集合就完全相同（收割层面无法区分）。
    same_pool = [f"{p['a']} ↔ {p['b']}" for p in pairs_full
                 if set(_pool_of(queries, p["a"])) == set(_pool_of(queries, p["b"]))]
    jt = [p["junction"] for p in pairs] or [0.0]
    ft = [p["full"] for p in pairs_full] or [0.0]
    return {
        "n_pairs": len(pairs_full),
        "n_pairs_junction": len(pairs),
        "junction": {"min": min(jt), "mean": round(sum(jt) / len(jt), 4), "max": max(jt)},
        "full": {"min": min(ft), "mean": round(sum(ft) / len(ft), 4), "max": max(ft)},
        "most_similar_pair": pairs[0] if pairs else None,
        "most_similar_pair_full": pairs_full[0] if pairs_full else None,
        "shell_count": len(shells),
        "shells": shells,
        "empty_junction_count": len(empty),
        "empty_junction_ranks": empty,
        "empty_junction_items": [
            f"{q['rank']}:{q['responsibility']}" for q in queries if not jun[q["rank"]]],
        "same_domain_set_pairs": same_pool,
        "distinct_full_vectors": len({tuple(q["positive"]) for q in queries}),
        "distinct_nonempty_junction_vectors": len(
            {tuple(sorted(jun[q["rank"]])) for q in nonempty}),
    }


# ---------------------------------------------------------------------------
# 4. 判据
# ---------------------------------------------------------------------------
def judge_j1(inputs_ok: bool, errs: list[str]) -> list[str]:
    """J1 输入齐备性。由 main 的加载路径保证；这里只兜底断言（自检里会喂 False）。"""
    return [] if inputs_ok else [f"输入没拿到：{'; '.join(errs)}"]


def judge_j2(queries: list[dict], worklist: list[dict]) -> list[str]:
    """J2 覆盖率是一等输出。**生成 0 条必须判红**（不是「全部通过」）。"""
    errs = []
    if not worklist:
        errs.append("J2 工单为空 —— 这不是「没问题」，是没测（应 exit 2）")
    if not queries:
        errs.append("J2 一条检索式都没生成 —— 覆盖率 0% 不得报「全部通过」")
    if worklist:
        cov = len(queries) / len(worklist)
        #: ⚠️ 这段曾只有 `cov <= 0` 一条 —— 那是**单边判据**：
        #: 把工单从 31 条截到 10 条，覆盖率 80%（正数）照样通过，
        #: 端到端值变异当场抓到「J2 打不红」。补上**与基线相等**这一侧。
        if cov <= 0:
            errs.append(f"J2 覆盖率 {cov:.1%} ≤ 0")
        if BASELINE.get("generated") is not None and len(queries) != BASELINE["generated"]:
            errs.append(f"J2 生成数 {len(queries)} ≠ 基线 {BASELINE['generated']}"
                        f"（覆盖率 {cov:.1%}）—— 覆盖率与生成数**两侧都要钉**")
    return errs


def structural_blank_readings(worklist: list[dict], ledger_rows: list[dict]) -> dict:
    """结构空白的**两种口径**都算出来（登记，不擅自选一个）。

    - **严口径**（`gap-ledger.json` 的 flag）：工单里 `structural_blank == true` 的条数；
    - **宽口径**（按岗位读）：工单里落在 `STRUCTURAL_BLANK_ROLES` 上的条数；
      另报**账内**（151 行）全量结构空白条数。
    报告里两个数并列 —— 「差额由所有者裁决，脚本不擅自放宽」（F3 §3 脚注原文口径）。
    """
    strict = [q for q in worklist if q["structural_blank"]]
    wide = [q for q in worklist if q["role_id"] in STRUCTURAL_BLANK_ROLES]
    return {
        "strict_worklist": len(strict),
        "strict_worklist_items": [f"{q['rank']}:{q['role_id']}:{q['l3']}" for q in strict],
        "wide_by_role_worklist": len(wide),
        "wide_by_role_items": [f"{q['rank']}:{q['role_id']}:{q['l3']}" for q in wide],
        "ledger_rows_blank": sum(1 for r in ledger_rows if r.get("structural_blank")),
        "ledger_rows_blank_roles": sorted(
            {r["role_id"] for r in ledger_rows if r.get("structural_blank")}),
        "worklist_total": len(worklist),
        "retrieval_eligible_strict": len(worklist) - len(strict),
        "retrieval_eligible_wide": len(worklist) - len(wide),
    }


def judge_j3(queries: list[dict], readings: dict, baseline: dict) -> list[str]:
    """J3 结构空白显式排除，且两口径条数逐个与基线相等（**单边判据 = 没有判据**）。"""
    errs = []
    blank_resp = {q["responsibility"] for q in queries if q["structural_blank"]}
    if blank_resp:
        errs.append(f"J3 给结构空白条目生成了检索式：{sorted(blank_resp)}"
                    f"（它们走 SOP，不占检索预算）")
    q_by_resp = {q["responsibility"]: q for q in queries}
    for key in ("strict_worklist", "wide_by_role_worklist", "ledger_rows_blank"):
        got, want = readings[key], baseline[key]
        if got != want:
            errs.append(f"J3 {key} 实测 {got} ≠ 基线 {want}")
    # 排除的口径必须**双向**对上：工单里 eligible 的条数 = 生成条数
    eligible = readings["worklist_total"] - readings["strict_worklist"]
    if len(queries) != eligible:
        errs.append(f"J3 可检索位 {eligible} 条，但生成了 {len(queries)} 条（差 "
                    f"{eligible - len(queries)}）")
    for q in queries:
        if q["responsibility"] in {i.split(":")[2] for i in readings["strict_worklist_items"]}:
            errs.append(f"J3 {q['responsibility']} 在严口径结构空白名单里却仍生成了检索式")
    return errs


def serialize(queries: list[dict]) -> str:
    """产物的**唯一**序列化口径（J4 逐字节比对的就是这个）。"""
    return json.dumps(queries, ensure_ascii=False, indent=1, sort_keys=True)


def judge_j4(build_fn, queries: list[dict]) -> list[str]:
    """J4 可复算：同一输入两次运行结果**逐字节相同**。

    反向控制：把 `queries` 改一个词（顺序交换），序列化必须**变** ——
    否则说明「字节相同」是因为序列化忽略了内容（假绿）。
    """
    errs = []
    a = serialize(build_fn())
    b = serialize(build_fn())
    if a != b:
        errs.append("J4 两次运行的检索式不是逐字节相同 —— 检索式不可复算")
    if a != serialize(queries):
        errs.append("J4 重算结果与本次产物不一致")
    tampered = [dict(queries[0])] if queries else []
    if tampered:
        tampered[0]["positive"] = list(reversed(tampered[0]["positive"]))
        if serialize(tampered + queries[1:]) == a:
            errs.append("J4 篡改交叉项顺序后序列化不变 —— 「逐字节相同」是假绿")
    return errs


def perturb_worklist(worklist: list[dict], which: str) -> list[dict]:
    """J5 的篡改样本：只改一条工单，其余不动。

    `name` 改责任名（连带换域，因为域由责任名决定）、`domain` 只改所属域、
    `role` 改名岗位 id、`noop` 改一个**不进检索式**的字段（反向控制：必须**不**变）。
    """
    out = [dict(w) for w in worklist]
    target = next((i for i, w in enumerate(out) if not w["structural_blank"]), None)
    if target is None:
        return out
    if which == "name":
        # 换成一个真实存在、域不同的责任名
        out[target]["l3"] = "复购实验"
        out[target]["role_id"] = "AGT-034"
        out[target]["role_title"] = "CRM留存与复购"
        out[target]["domain"] = "品牌与增长"
    elif which == "domain":
        out[target]["domain"] = "财务与合规"
    elif which == "role":
        out[target]["role_id"] = "AGT-999"
        out[target]["role_title"] = "改过的岗位名"
    elif which == "noop":
        # **反向控制**：只改**不进检索式**的字段（排序无关的元信息）。
        out[target]["n_flows"] = 99
        out[target]["in_first_slice"] = not bool(out[target].get("in_first_slice"))
        out[target]["score_hint"] = "只用于排序，不进检索式"
    elif which == "blank":
        out[target]["structural_blank"] = True
    return out


def judge_j5(build_from_worklist, worklist: list[dict]) -> tuple[list[str], dict]:
    """J5 **改了工单，检索式必须变** —— 当作失败条件断言。

    并且配**反向控制**：改一个不进检索式的字段（`n_flows` / `in_first_slice`），
    检索式必须**逐字节不变** —— 否则「会变」只是「什么都变」的代名词。
    """
    errs = []
    base = build_from_worklist(worklist)
    sig_base = {q["role_id"] + "|" + q["responsibility"]: serialize([q]) for q in base}
    changed = {}
    for which in ("name", "domain", "role"):
        alt = build_from_worklist(perturb_worklist(worklist, which))
        # 找「位次相同」的那条（位次是排序槽位，不随内容变）
        n_diff = 0
        for q0, q1 in zip(base, alt):
            if serialize([q0]) != serialize([q1]):
                n_diff += 1
        changed[which] = n_diff
        if n_diff == 0:
            errs.append(f"J5 篡改 {which} 后没有任何检索式变化 —— 检索式是假仪表")
    noop = build_from_worklist(perturb_worklist(worklist, "noop"))
    if [serialize([q]) for q in base] != [serialize([q]) for q in noop]:
        errs.append("J5 反向控制失败：改了**不进检索式**的字段，产物却变了"
                    "（说明产物携带了与检索无关的信息）")
    blank = build_from_worklist(perturb_worklist(worklist, "blank"))
    if len(blank) != len(base) - 1:
        errs.append(f"J5 把一条标成结构空白后，生成数 {len(blank)} ≠ {len(base) - 1}"
                    f"（结构空白必须被真的排除，不是只打个标记）")
    return errs, {"perturbed_changed": changed,
                  "noop_unchanged": not any("反向控制" in e for e in errs)}


def judge_j6(sim: dict, thresholds: dict) -> list[str]:
    """J6 不同工单的检索式必须彼此不同（**单边判据 = 没有判据**）。

    四个方向同时判：
      · **上界**：最像的一对（全向量 / 交叉项）不得超过 `*_max` —— 防模板套壳；
      · **下界**：最像的一对也不得低于 `junction_min` —— 防「交叉项根本没起作用」
        （相似度太低同样不是好事：它意味着交叉项与责任无关，只是噪声）；
      · **空壳**：交叉项被另一条完全包含的条数 ≤ `shell_max` —— 防「只换责任名」；
      · **没得比**：`n_pairs == 0` 判红 —— 「没有可比的一对」不是「都不像」，是没测。
    """
    errs = []
    if sim["n_pairs"] == 0:
        errs.append("J6 没有可比的一对 —— 这不是「都不像」，是没测（应 exit 2）")
    if sim["junction"]["max"] < thresholds["junction_min"]:
        errs.append(f"J6 交叉项相似度上界 {sim['junction']['max']} < 下限 "
                    f"{thresholds['junction_min']} —— 上界过低说明交叉项没起作用"
                    f"（单边判据的反面）")
    if sim["full"]["max"] >= thresholds["full_max"]:
        p = sim["most_similar_pair_full"]
        errs.append(f"J6 最相似的一对 {p['a']} ↔ {p['b']} 全向量 Jaccard="
                    f"{sim['full']['max']} ≥ {thresholds['full_max']} —— 模板套壳")
    if sim["n_pairs_junction"] and sim["junction"]["max"] >= thresholds["junction_max"]:
        p = sim["most_similar_pair"]
        errs.append(f"J6 最相似的一对 {p['a']} ↔ {p['b']} 交叉项 Jaccard="
                    f"{sim['junction']['max']} ≥ {thresholds['junction_max']} —— 只换了责任名")
    if sim["shell_count"] > thresholds["shell_max"]:
        names = [s["responsibility"] for s in sim["shells"]]
        errs.append(f"J6 空壳 {sim['shell_count']} 条 > 上限 {thresholds['shell_max']}：{names}")
    #: 空交叉项是**另一类**缺陷（没料，不是套壳），因此**两个方向都判**：
    #: 高过上限 ⇒ 新出现了取不到材的责任；低过基线 ⇒ 取材条件变宽，必须复核而不是庆祝。
    n_empty = sim["empty_junction_count"]
    if n_empty > thresholds["empty_junction_max"]:
        errs.append(f"J6 空交叉项 {n_empty} 条 > 上限 {thresholds['empty_junction_max']}："
                    f"{sim['empty_junction_items']}")
    if n_empty < thresholds["empty_junction_min"]:
        errs.append(f"J6 空交叉项 {n_empty} 条 < 基线 {thresholds['empty_junction_min']} —— "
                    f"取材条件变宽了（或上游卡池变了），**不得当成改进**，必须复核")
    if sim["n_pairs_junction"] == 0 and sim["n_pairs"]:
        errs.append("J6 交叉项一对都没比成 —— 这不是「都不一样」，是没测")
    #: 同域池条目对的条数：**上界钉住**（涨上去 = 决策表把责任压缩到同一批域了）
    n_pool = len(sim["same_domain_set_pairs"])
    if n_pool > thresholds["same_pool_max"]:
        errs.append(f"J6 同检索域池的条目对 {n_pool} > 上限 {thresholds['same_pool_max']}："
                    f"{sim['same_domain_set_pairs']} —— 决策表把不同责任压进了同一批检索域")
    return errs


def negative_divergence(cf, queries: list[dict]) -> dict:
    """J7 的核心仪器：**从 candidate_filter 算出**负向词的分歧矩阵。

    对每个词 t，看它在各域下的判定：若 `t ∈ GLOBAL_NEGATIVE` ⇒ 全域生效（不构成分歧）；
    否则只在 `t ∈ DOMAIN_NEGATIVE[d]` 的域生效。我们要证明的是：
    **同一个词在不同域下的生效状态确实相反**，而且这种相反在本次的检索式集合里**真的被用上**。

    ⚠️ 这不是「维护一张同词异义清单」—— 清单会腐烂。这里每次都从 filter 现算。
    """
    doms = sorted({d for q in queries for d in q["query_domains"]})
    matrix: dict[str, dict[str, bool]] = {}
    for t in sorted({n for d in doms for n in cf.DOMAIN_NEGATIVE.get(d, [])}):
        state = {}
        for d in doms:
            if t in cf.GLOBAL_NEGATIVE:
                state[d] = True
            else:
                state[d] = t in cf.DOMAIN_NEGATIVE.get(d, [])
        matrix[t] = state
    divergent = {t: s for t, s in matrix.items() if len(set(s.values())) > 1}

    def truth(term: str, q: dict) -> bool | None:
        """该词在这条**检索式**上是否判负向（任一所辖域列入即判）。
        返回 None = 该词与这条检索式的域池无交集（**看不见**，不作证据）。"""
        vals = [matrix[term][d] for d in q["query_domains"] if d in matrix[term]]
        return any(vals) if vals else None

    #: 分歧词里，有多少**真的被本次检索式用上** —— 口径是「存在一对条目，该词在 A 上判负向、
    #: 在 B 上不判」，**两个方向都要求有定义**（None 不作证据）。
    #: ⚠️ 第一版写成 `if da and db and any(da) and not any(db)`，把「单侧有定义」也算成用了，
    #: 于是报出 56/56 —— 那个数字没有区分力（「几乎都用了」= 判据没在测东西）。
    used, used_terms = [], set()
    for term, _ in divergent.items():
        for qa in queries:
            for qb in queries:
                if qa["rank"] >= qb["rank"]:
                    continue
                ta, tb = truth(term, qa), truth(term, qb)
                if ta is None or tb is None:
                    continue
                if ta and not tb:
                    used.append({"term": term, "negative_in": qa["responsibility"],
                                 "not_negative_in": qb["responsibility"]})
                    used_terms.add(term)
                    break
            else:
                continue
            break
    #: 另给一个**也**该报的数：有多少条检索式的负向段里**真的带了**至少一个分歧词
    #: （它们才是「按域生效」在下游起作用的载体）。
    carriers = [q["responsibility"] for q in queries
                if any(truth(t, q) for t in divergent)]
    return {"domains": doms, "n_negative_terms": len(matrix),
            "n_divergent": len(divergent), "divergent": divergent,
            "used_in_pairs": used,
            "n_used_terms": len(used_terms),
            "n_queries_carrying_divergent": len(carriers),
            "carriers": carriers}


def judge_j7(div: dict, cf, queries: list[dict], kw_text: str,
             pairs: tuple | None = None) -> list[str]:
    """J7 **负向词按域生效**，不是全局清单。

    三条同时判（缺一条就是单边判据）：
      ① 分歧词必须存在（>0）—— 否则「按域生效」无从谈起；
      ② 分歧词必须在本次检索式里**真的被用上**（两个条目状态相反）；
      ③ 同词异义反例必须逐条成立：`trial`/`cohort`/`ad` 三个词，按 filter 现算的状态
         与词库 §1 的教训逐条对账，**对不上就报出来**（本仓库纪律：登记不重判）。
    """
    errs = []
    if div["n_negative_terms"] == 0:
        errs.append("J7 一个按域负向词都没有 —— 仪器可能坏了")
    if div["n_divergent"] == 0:
        errs.append("J7 没有任何一个词在不同域下状态相反 —— 「按域生效」是空话")
    if not div["used_in_pairs"]:
        errs.append("J7 分歧词虽然存在，但本次检索式里没有任何一对用上了它"
                    "（分歧是装饰）")
    # ③ **行为**层面的反例：同一个词，在一个域判负向、在另一个域不判 ——
    #    这是「按域生效」的**唯一**机械证据（词表存在性不是证据）。
    for term, d_neg, d_pos in (pairs if pairs is not None else SYNONYMY_PAIRS):
        if term not in div["divergent"]:
            errs.append(f"J7 反例失效：{term!r} 不再是分歧词（各域状态一致）——"
                        f"「同一词在不同域含义相反」在 filter 里已不成立，必须复核")
            continue
        st = div["divergent"][term]
        if d_neg not in st or not st[d_neg]:
            errs.append(f"J7 反例失效：{term!r} 在 {d_neg} 应判负向")
        if d_pos in st and st[d_pos]:
            errs.append(f"J7 反例失效：{term!r} 在 {d_pos} 不该判负向，filter 却列了它")
    # ④ 词库 §1 的**字面**记录 vs filter 的**实测**：对不上要**登记**，不得静默放宽。
    #    ⚠️ 本节曾把「词库说的词**必须**在 filter 里」写成硬判据，--check 当场三条全红。
    #    实测（2026-09-13）：`trial` / `cohort` / `ad` **三个词在 filter 的任何域负向表里
    #    都不存在**；filter 用的是收窄形式 `cohort study`，而 `trial` / `ad` 根本没实现
    #    （02 域不敢列 trial 是对的，但 14 域也没列）。
    #    ⇒ 按本仓库纪律（**登记不重判**）改为：① 登记这三条的实测值；
    #      ② 配一条**能打红**的判据 —— 如果哪天 filter 把 `trial` 列进 02 域
    #      （即真的开始误杀 RCT 方法论论文），或把 `ad`/`ADHD` 的逻辑改坏，必须报出来。
    reg = doc_vs_code_audit(cf, div, pairs)
    if reg["regressions"]:
        errs.extend(reg["regressions"])
    return errs


#: 同词异义**行为**反例：(词, 应判负向的域, 不判负向的域)。
#:
#: ⚠️ **候选词必须排除 `GLOBAL_NEGATIVE` 里的词**。第一版拿了 `clinical` / `eeg` 当反例，
#: `--check` 当场判红 —— 它们在 `GLOBAL_NEGATIVE` 里，**全域生效**，各域状态当然一致，
#: 根本就不构成「按域分歧」。这是「仪器用错」的又一例：**全局负向词不可能是分歧词**。
#: 现在这三个词由 `judge_j7` 逐个现算核对（不是抄表）。
SYNONYMY_PAIRS = (
    # 14 域是「同期群」，医学里是「队列研究」⇒ filter 用收窄形式 `cohort study`
    ("cohort study", "14-用户分析", "01-因果推断"),
    # 13 域要广告，绝不能把医学缩写 AD(Alzheimer/ADHD) 放进来
    ("alzheimer", "13-广告分析", "12-ML基础"),
    # 09 域排除纯代码生成基准，02 域不排除（实验设计类论文里有它）
    ("code generation benchmark", "09-DataAgent-LLM", "02-A_B实验"),
    # 05 域排除电影推荐，04 域不排除
    ("movie", "05-推荐系统", "04-供应链"),
)

#: 词库 §1 的**字面**记录 vs filter 实测（登记，不重判）。
#: `expect=("neg", 域)` = 词库里说该词在该域生效。
DOC_VS_CODE = (
    ("trial", "14-用户分析", "词库 §1：「负向词 `trial` 只在 14 域生效」"),
    ("cohort", "14-用户分析", "词库 §1/§2：「医学里是队列研究，与 cohort retention 同词不同义」"),
    ("ad", "13-广告分析", "词库 §2 13 域：「`adversarial attack`（ad 歧义）、ADHD、Alzheimer」"),
)
#: 如果 filter 真的把下面这些列进去，就是**在误杀方法论论文** —— 必须报红。
FORBIDDEN_NEGATIVES = (
    ("02-A_B实验", "trial"),
    ("01-因果推断", "trial"),
    ("14-用户分析", "ad"),
)


def doc_vs_code_audit(cf, div: dict, pairs: tuple | None = None) -> dict:
    """把「词库字面记录」与「filter 实测」逐条对账，并给出**两个方向**的判定。

    · `registered`：词库说的词在 filter 里**不存在** ⇒ 登记（不擅自改 filter，纪律 6）；
      但同一行必须能解释「那 filter 用的是什么」——收窄形式（`cohort study`）算可接受，
      **完全没有对应实现**（`trial` / `ad`）算**缺口**，点名报出。
    · `regressions`：一旦 filter 把 `trial` 列进 01/02 域（误杀 RCT），立即判红。
    """
    registered, gaps, regressions = [], [], []
    #: 纪律：分歧反例**不得**选全局负向词（它们全域生效，不可能是分歧词）。
    for term, d_neg, d_pos in (pairs if pairs is not None else SYNONYMY_PAIRS):
        if term in cf.GLOBAL_NEGATIVE:
            regressions.append(
                f"J7 仪器用错：反例词 {term!r} 在 GLOBAL_NEGATIVE 里 —— "
                f"全局词不可能是「按域分歧」的证据，请换成按域负向词")
    for term, d, quote in DOC_VS_CODE:
        present = term in cf.DOMAIN_NEGATIVE.get(d, [])
        narrowed = [x for x in cf.DOMAIN_NEGATIVE.get(d, []) if term in x]
        registered.append({"term": term, "domain": d, "quote": quote,
                           "in_filter": present, "narrowed_forms": narrowed})
        if not present:
            if narrowed:
                gaps.append({"term": term, "domain": d, "replacement": narrowed,
                             "verdict": "收窄实现（可接受）"})
            else:
                gaps.append({"term": term, "domain": d, "replacement": [],
                             "verdict": "**词库有记录、代码无实现**（缺口）"})
    for d, term in FORBIDDEN_NEGATIVES:
        if term in cf.DOMAIN_NEGATIVE.get(d, []):
            regressions.append(
                f"J7 回归：filter 把 {term!r} 列进了 {d} 的负向表 —— "
                f"这会把 RCT / 广告方法论论文当医学论文误杀（词库 §1 明令禁止）")
    return {"registered": registered, "gaps": gaps, "regressions": regressions}


def judge_j8(queries: list[dict], cf, sources: dict, kw_text: str) -> list[str]:
    """J8 产物能被 `candidate_filter.py` **直接消费**（同一个配置格式）。

    五条逐项对：
      ① 每个 `query_domains` 的域名都在 `cf.DOMAIN_DIR` 里（否则 filter 静默跳过）；
      ② 每个域至少有一条检索式（**全域覆盖**，不静默漏）；
      ③ `negative` ⊇ filter 对该域的 global+domain 负向词（逐条，不是计数相等）；
      ④ `constraint` ⊇ filter 对该域的约束词（同理）；
      ⑤ 正向词的第三来源（关键词库）必须非空 —— 否则「三段式」的第一段是抄来的。
    """
    errs = []
    covered = {d for q in queries for d in q["query_domains"]}
    unknown = sorted(covered - set(cf.DOMAIN_DIR))
    if unknown:
        errs.append(f"J8 域 {unknown} 不在 candidate_filter.DOMAIN_DIR 里 —— "
                    f"filter 会把它们**静默跳过**（保留全部，等于没过滤）")
    must = {d for q in queries for d in q["query_domains"]}
    missing_domain_rows = sorted(must - covered)
    if missing_domain_rows:
        errs.append(f"J8 这些域没有任何检索式：{missing_domain_rows}")
    for d in sorted(must):
        want_neg = set(cf.GLOBAL_NEGATIVE) | set(cf.DOMAIN_NEGATIVE.get(d, []))
        got_neg = set()
        for q in queries:
            if d in q["query_domains"]:
                got_neg |= set(q["negative"])
        if not want_neg <= got_neg:
            errs.append(f"J8 {d} 的负向词缺 {sorted(want_neg - got_neg)} "
                        f"（与 filter 漂移 = 过滤行为会与词库不一致）")
        want_con = set(cf.DOMAIN_CONSTRAINT.get(d, [])) or set(CONSTRAINT_FALLBACK.get(d, []))
        got_con = set()
        for q in queries:
            if d in q["query_domains"]:
                got_con |= set(q["constraint"])
        if want_con and not want_con <= got_con:
            errs.append(f"J8 {d} 的约束词缺 {sorted(want_con - got_con)}")
        if not want_con:
            errs.append(f"J8 {d} 既无 filter 约束词也无兜底约束词 —— 宽词会跨行业误命中")
    # 兜底约束词必须逐条抄对（回查词库原文）
    for d, terms in CONSTRAINT_FALLBACK.items():
        for t in terms:
            if t not in kw_text.lower():
                errs.append(f"J8 兜底约束词 {t!r}（{d}）在关键词库-v2.md 里找不到 —— 抄错了")
    for q in queries:
        if not q["positive_sources"].get("harvest:" + q["query_domains"][0]):
            errs.append(f"J8 {q['responsibility']} 缺 harvest 正向词（第一来源）")
        if not any(v for k, v in q["positive_sources"].items() if k.startswith("keyword-lib:")):
            errs.append(f"J8 {q['responsibility']} 缺词库正向词（第三来源）")
    # 检索词必须是字符串列表且非空（filter 用 `in` 做子串匹配，元素不能是空串）
    for q in queries:
        for field in ("positive", "negative", "constraint"):
            bad = [t for t in q[field] if not isinstance(t, str) or not t.strip()]
            if bad:
                errs.append(f"J8 {q['responsibility']}.{field} 里有空词：{bad[:3]}")
    del sources
    return errs


def judge_j9(queries: list[dict], contracts: dict, cidx: dict) -> list[str]:
    """J9 契约挂载点。**契约就是入卡挂载点**，键名必须与 `check_contracts.py` 对齐。

    四条：
      ① 每条检索式都有 `responsibility` / `l3` / `rank` / `contract_id`；
      ② `responsibility` 命中契约的 `responsibility` 字段（不是自己编的键名）；
      ③ 契约存在时 `contract_id` 形如 `CTR-[AB]-\\d{3}`；
      ④ 覆盖率作一等输出，且**同一责任两份正式契约**必须报出来（不给「随便挑一份」）。
    """
    errs = []
    still_missing: list[str] = []
    if not contracts:
        errs.append("J9 一份契约都没读到 —— 这不是「无需挂载」，是没测")
    for q in queries:
        for k in ("responsibility", "l3", "rank", "contract_id"):
            if k not in q:
                errs.append(f"J9 {q.get('responsibility')} 缺字段 {k}")
        cid = q.get("contract_id")
        if cid and not re.fullmatch(r"CTR-[AB]-\d{3}", cid):
            errs.append(f"J9 {q['responsibility']} 的 contract_id={cid!r} 形状不对")
        if cid and not str(cid).startswith("CTR-" + q["serviceability"].replace("A", "A").replace("B", "B")):
            # C 类不建契约；A/B 必须与服务性一致
            if q["serviceability"] in ("A", "B") and cid.split("-")[1] != q["serviceability"]:
                errs.append(f"J9 {q['responsibility']} 服务性 {q['serviceability']} 与契约 "
                            f"{cid} 的模板不符（服务性由图谱唯一导出）")
    #: 契约池覆盖：责任名在契约池里 ⇒ 必须挂上（挂不上 = 契约索引键写错了）
    if contracts:
        for q in queries:
            if (q["role_id"], q["responsibility"]) in cidx["by_key"] and not q["contract_id"]:
                errs.append(f"J9 {q['rank']}:{q['responsibility']} 在契约池里有对应契约，"
                            f"却没挂上 —— 索引键写错了")
    if cidx["duplicates"]:
        errs.append(f"J9 同一 (role_id, 责任名) 有两份正式契约：{cidx['duplicates']}")
    n_mounted = sum(1 for q in queries if q["contract_id"])
    #: ⚠️ 这段曾只有 `n_mounted == 0` 一条 —— 那是**全有全无**的单边判据：
    #: 28 条里掉了 1 条（27 挂 1 不挂）照样通过。端到端值变异当场抓到
    #: （「只把一条挂载点清空」打不红）。补上**逐条 + 两侧**：
    if queries and n_mounted == 0:
        errs.append("J9 一条都没挂上契约 —— 没有挂载点的检索式产出卡片后无处可挂")
    for q in queries:
        cid = q.get("contract_id")
        if not cid:
            #: 契约池里**没有**对应责任 ⇒ 允许为空，但**必须点名**（不静默）
            still_missing.append(f"{q['rank']}:{q['responsibility']}")
    if queries and n_mounted != len(queries) - len(still_missing):
        errs.append(f"J9 挂载数 {n_mounted} ≠ 应有 {len(queries) - len(still_missing)} —— "
                    f"掉了一条也算掉（不许全有全无）")
    return errs


#: J10 的 needle：**排序权重名**。它们只用于排序，不得进入任何判据。
#:
#: ⚠️ 本仓库 F3 的 J7 会扫 8 个门禁脚本；本脚本**不在**那份清单里（它不是门禁），
#: 所以这里**对自己**做同一件事：断言本文件的判据函数体里不出现这些名字。
#:
#: ⚠️⚠️ **这条判据自己踩过一次坑（实测）**：第一版把 needle 列表**内联在
#: `judge_j10` 函数体里**，于是「扫判据函数体」的扫描**打中了自己** ——
#: 自检当场红。修法同本仓库的既有教训（「判据的适用范围被默认成了全体」）：
#: needle 表移到模块级常量，判据函数体里只留一个**不含字面量**的引用。
WEIGHT_NEEDLES = ("zero_gap", "flow_breadth", "first_slice", "boundary_penalty",
                  "structural_penalty", "score_parts")
#: 探针用的 needle 取第一个 —— 反向探针必须注入**真实存在**的名字，否则探针是摆设。
_NEEDLE0 = WEIGHT_NEEDLES[0]
#: 被扫的函数名（判据 + 相似度统计：它们都不该看见排序分数）。
_SCANNED_FUNCS = ("similarity_report",)
_SCANNED_PREFIX = "judge_"


#: needle 的匹配口径：**词边界**，不是裸子串。
#:
#: ⚠️ 裸子串匹配实测**假红过一次**：工单里的输入字段 `in_first_slice` 内含
#: `first_slice`，于是 `judge_j5`（它构造篡改样本时必须写这个字段名）被误判成
#: 「引用了排序权重」。`(?<![A-Za-z0-9_])` 把它排除，而 `vars(...).get("zero_gap")`
#: 与 `_x = zero_gap` 两种真实形态都还在射程内。
def _needle_re(name: str):
    import re as _re
    return _re.compile(r"(?<![A-Za-z0-9_])" + _re.escape(name) + r"(?![A-Za-z0-9_])")


def _scan_weight_refs(text: str) -> tuple[list[str], int]:
    """AST 取出判据函数体，扫 needle。返回 (命中的 `函数->词`, **被扫函数数**)。

    两个「没测」的保护：
      · **全文扫会打中文档注释 ⇒ 假红**，故按函数取段（而不是全文）；
      · **扫到 0 个函数 ⇒ 判红**（`_SCANNED_PREFIX` 改名就会让这道判据静默失效）。
    """
    import ast as _ast
    try:
        tree = _ast.parse(text)
    except SyntaxError as e:  # pragma: no cover
        return [f"内部错误：{e}"], 0
    bad: list[str] = []
    n_scanned = 0
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and (
                node.name.startswith(_SCANNED_PREFIX) or node.name in _SCANNED_FUNCS):
            n_scanned += 1
            seg = _ast.get_source_segment(text, node) or ""
            for n in WEIGHT_NEEDLES:
                if _needle_re(n).search(seg):
                    bad.append(f"{node.name}->{n}")
    return bad, n_scanned


#: 探针锚点（**函数签名整行**）。用整行做锚点是刻意的：
#: 第一版用 `src_text.replace("def similarity_report(...)", "...\\n    _p = zero_gap")`
#: 构造探针，结果 `str.replace` **把探针代码自己也替换了一遍**（因为探针代码里
#: 字面含有同一个 needle）⇒ 源码变语法错误 ⇒ AST 解析失败 ⇒ 反向控制假红。
#: 这是「判据自己把自己弄脏」的又一例，修法是**锚点唯一 + 只替换一次**。
_PROBE_ANCHOR = "def similarity_report(queries: list[dict]) -> dict:"
_CTRL_ANCHOR = "def terms_of(groups: list[str]) -> list[str]:"


def _pool_of(queries: list[dict], key: str) -> list[str]:
    """从 `"<rank>:<责任名>"` 取回该条目的检索域池（顺序无关，比较前排序）。"""
    rk = int(key.split(":", 1)[0])
    for q in queries:
        if q["rank"] == rk:
            return list(q["query_domains"])
    return []


def _inject(text: str, anchor: str, line: str) -> str:
    """在唯一锚点函数体首行注入一行。锚点必须是**整行**且**只出现一次**。"""
    n = len(re.findall(r"(?m)^" + re.escape(anchor) + r"$", text))
    if n != 1:
        return text
    return re.sub(r"(?m)^" + re.escape(anchor) + r"$",
                  lambda m: m.group(0) + "\n    " + line, text, count=1)


def judge_j10(queries: list[dict], ledger: dict, src_text: str) -> list[str]:
    """J10 排序权重不进任何判定。

    ① 产物里不得出现权重名（它们只被读来排序，读一次就丢）；
    ② **判据函数体**不得出现权重名 —— 用 AST 把判据函数的函数体取出来扫，
       而不是全文扫（全文扫会打中模块常量与文档注释 ⇒ 假红）；
    ③ 反向探针：往一个判据函数体注入 needle，AST 扫描**必须**报出来；
    ④ 反向控制：往**非判据**函数里注入同一个 needle，扫描必须**不**报
       —— 否则「扫判据函数体」实际是「扫全文」，判据 ② 就不成立。

    ⚠️ 与 F3 的 J7 同源（风险 N1）。**分数可以读，但不得成为判据的输入**。
    """
    errs = []
    dumped = serialize(queries)
    if any(n in dumped for n in WEIGHT_NEEDLES):
        errs.append("J10 产物里出现排序权重名 —— 分数只用于排序")
    hits, n_scanned = _scan_weight_refs(src_text)
    if hits:
        errs.append("J10 判据函数体里出现排序权重名 —— 排序不得进门禁（风险 N1）")
    if n_scanned < 4:
        errs.append(f"J10 只扫到 {n_scanned} 个判据函数 —— 这不是「干净」，是没测")
    probe = _inject(src_text, _PROBE_ANCHOR, "_p = " + _NEEDLE0)
    probe_hits, _ = _scan_weight_refs(probe)
    if not probe_hits:
        errs.append("J10 反向探针失败：往判据函数体注入权重名，AST 扫描没抓到")
    ctrl = _inject(src_text, _CTRL_ANCHOR, "_c = " + _NEEDLE0)
    if ctrl == src_text:
        errs.append("J10 反向控制失效：探针锚点已不在文件里，这条判据没测到东西")
    else:
        ctrl_hits, _ = _scan_weight_refs(ctrl)
        if ctrl_hits:
            errs.append("J10 反向控制失败：往**非判据**函数注入权重名也被判红 —— "
                        "说明扫描其实是全文扫，判据 ② 不成立")
    wl = ledger.get("worklist") or [{}]
    if "weights" in json.dumps(wl[0]):
        errs.append("J10 工单条目里混进了权重字段 —— 输入形状变了")
    return errs


# ---------------------------------------------------------------------------
# 5. 渲染
# ---------------------------------------------------------------------------
def render_md(queries: list[dict], sim: dict, div: dict, readings: dict,
              coverage: dict, jrows: list[tuple[str, str, str]], baseline: dict,
              notes: dict) -> str:
    L = []
    A = L.append
    A("# PHASE6 S4 · 缺口驱动检索式（31 条靶区工单 → 三段式检索式）")
    A("")
    A("> 生成器：`paper2skills-research/scripts/build_search_queries.py`　|　"
      "复算：`python3 paper2skills-research/scripts/build_search_queries.py --check`")
    A("> 机读产物：`paper2skills-research/data/search-queries.json`")
    A("")
    A("**这一步在解决什么**：F3 的靶区工单是一张**缺口清单**，不是一张**采购单**。")
    A("没有 S4，缺口账只是一张纸 —— 它说「线索评估 零供给」，但没人知道拿什么词去捞。")
    A("S4 把每条工单翻译成一份**三段式检索式**（正向 / 负向 / 约束），")
    A("并且这份检索式**能被 `candidate_filter.py` 直接消费**（同一个配置格式，不另造一套）。")
    A("")
    A("**硬顺序**（方案 §8.1）：「S4 必须排在 S1 之后 —— 入卡挂载点就是契约。」")
    A("所以每条检索式都带 `responsibility` / `l3` / `rank` / `contract_id`，")
    A("**键名与 `check_contracts.py` 认的字段对齐**，产出候选卡之后能挂回那份契约。")
    A("")
    A("---")
    A("")
    A("## 1. 判据清单与实测")
    A("")
    A("| # | 判据 | **反面（这条判据在抓什么错）** | 实测 |")
    A("|---|---|---|---|")
    for code, name, reverse, got in jrows:
        A(f"| {code} | {name} | {reverse} | {got} |")
    A("")
    A("退出码：**0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 内部错误（≠ 判红）**。")
    A("")
    A("---")
    A("")
    A("## 2. 覆盖率是一等输出")
    A("")
    A(f"- 靶区工单（`worklist`）：**{coverage['worklist_total']} 条**")
    A(f"- 生成检索式：**{coverage['generated']} 条**　"
      f"⇒ **覆盖率 {coverage['rate']:.1%}**")
    A(f"- 结构空白（严口径，账内 flag）：**{readings['strict_worklist']} 条**被显式排除 ⇒ "
      f"**可检索位 {readings['retrieval_eligible_strict']}**")
    A(f"- 结构空白（宽口径，按岗位 `{'/'.join(STRUCTURAL_BLANK_ROLES)}` 读）："
      f"**{readings['wide_by_role_worklist']} 条** ⇒ **可检索位 "
      f"{readings['retrieval_eligible_wide']}**")
    A(f"- 账内（151 行）结构空白条目：**{readings['ledger_rows_blank']} 条**，"
      f"岗位 {readings['ledger_rows_blank_roles']}")
    A("")
    A("> ⚠️ **两口径差额（登记，不擅自放宽）**：F3 §3 脚注把这件事写得很清楚 ——")
    A("> §4 的结构性空白名单是 **L3 级**的零供给条目；若改按**岗位**读，")
    A("> 靶区一里落在 `AGT-009..013` 上的还有 可用性验证、测试方案（它们有 legacy 卡）。")
    A("> **差额由所有者裁决，脚本不擅自放宽** —— 本脚本两个数都算、都报，")
    A("> 并用 `--check` 把两个数**各自**钉在基线上（任一个变了就判红）。")
    A("")
    A("### 被排除的条目（走 SOP，不占检索预算）")
    A("")
    A("| 位次 | 责任名 | 岗位 | 域 | 口径 |")
    A("|---|---|---|---|---|")
    for item in readings["strict_worklist_items"]:
        rank, role, l3 = item.split(":")
        q = next((x for x in queries if x["responsibility"] == l3), None)
        A(f"| {rank} | {l3} | {role} | — | 严（账内 flag） |")
    del q
    for item in readings["wide_by_role_items"]:
        rank, role, l3 = item.split(":")
        if item in readings["strict_worklist_items"]:
            continue
        A(f"| {rank} | {l3} | {role} | — | **仅宽口径**（有 legacy 卡，账内 flag 为假） |")
    A("")
    A("---")
    A("")
    A("## 3. 相似度分布与空壳计数（判据「不同工单的检索式必须彼此不同」）")
    A("")
    A(f"两两比对数：**{sim['n_pairs']}**（其中交叉项可比 **{sim['n_pairs_junction']}** 对）")
    A("")
    A("| 口径 | 最小 | 均值 | 最大 |")
    A("|---|---:|---:|---:|")
    A(f"| 交叉项（把域宽词收紧到「这条责任」的英文词） | {sim['junction']['min']} | "
      f"{sim['junction']['mean']} | {sim['junction']['max']} |")
    A(f"| 全向量（域词 + 交叉项） | {sim['full']['min']} | {sim['full']['mean']} | "
      f"{sim['full']['max']} |")
    A("")
    pj, pf = sim["most_similar_pair"], sim["most_similar_pair_full"]
    A(f"- **最相似的一对（交叉项口径）**：`{pj['a']}` ↔ `{pj['b']}`　"
      f"Jaccard = **{pj['junction']}**（全向量 {pj['full']}）")
    A(f"- **最相似的一对（全向量口径）**：`{pf['a']}` ↔ `{pf['b']}`　"
      f"Jaccard = **{pf['full']}**")
    A(f"- **空壳（只换责任名）**：**{sim['shell_count']} 条**"
      f"（口径：交叉项非空、且被另一条完全包含）")
    A(f"- **空交叉项（取不到材）**：**{sim['empty_junction_count']} 条** —— "
      f"{'、'.join(sim['empty_junction_items']) or '无'}")
    A(f"  - ⚠️ 空交叉项的条目**从交叉项两两比对里剔除**：两条空集的相似度**没有定义**，"
      f"不是 1.0。第一版把它们算成 Jaccard=1.0，报出一对**不存在**的「只换责任名」假红。")
    A(f"  - 它们仍参与**全向量**比对（域词一样 ⇒ 全向量 Jaccard 高，这个是真实读数）。")
    A(f"  - 两类缺陷**分开报**：空壳 = 套壳；空交叉项 = 没料。**不能被平均成一个数**。")
    A(f"- **同检索域池的条目对**：**{len(sim['same_domain_set_pairs'])} 对** —— "
      f"它们本来就该捞同一批论文，全向量 Jaccard 天然高（这是**定义**，不是缺陷）；"
      f"区分它们的判据是交叉项。逐对：{'、'.join(sim['same_domain_set_pairs']) or '无'}")
    A(f"- 互不相同的全向量：**{sim['distinct_full_vectors']} / {coverage['generated']}**；"
      f"互不相同的**非空**交叉项向量："
      f"**{sim['distinct_nonempty_junction_vectors']} / "
      f"{coverage['generated'] - sim['empty_junction_count']}**")
    A("")
    A("> ⚠️ **判据只写单边等于没有判据**（本仓库吃过这个亏：覆盖率 >100% 也照样通过）。")
    A("> 所以这里同时给**上界**（最像的一对）与**下界**（空壳条数 + distinct 向量数）；"
      "三个方向都进退出码。")
    A("")
    A("---")
    A("")
    A("## 4. 同词异义反例（判据「负向词按域生效」的实测）")
    A("")
    A(f"本仪器**不维护**一张同词异义清单（清单会腐烂）—— 它每次从 "
      f"`candidate_filter.py` **现算**：把该域的全部按域负向词在各域下的状态排成矩阵，"
      f"再挑出**状态相反**的词。")
    A("")
    A(f"- 参与计算的按域负向词：**{div['n_negative_terms']} 个**")
    A(f"- 其中**在不同域下状态相反**的：**{div['n_divergent']} 个**")
    A(f"（下表**全部**列出 —— 只列前几条会让「56 个」这个数字无法复核）")
    A(f"- 这些分歧词里，**真的被本次检索式用上**的："
      f"**{div['n_used_terms']} 个词**（判据：存在一对条目，该词在 A 上判负向、在 B 上不判"
      f"——两侧都必须有定义）")
    A(f"- 负向段里**真的带上了**至少一个分歧词的检索式："
      f"**{div['n_queries_carrying_divergent']} / {coverage['generated']} 条**")
    A("")
    if div["used_in_pairs"]:
        A("| 词 | 判负向的条目 | 不判负向的条目 |")
        A("|---|---|---|")
        for u in div["used_in_pairs"]:
            A(f"| `{u['term']}` | {u['negative_in']} | {u['not_negative_in']} |")
        A("")
    A("### 三组实测反例逐条对账（**登记不重判**）")
    A("")
    A("| 词 | 词库 §1 的记录 | filter 实测 | 判定 |")
    A("|---|---|---|---|")
    for term, d, verdict, detail in notes["synonymy_probe"]:
        A(f"| `{term}` | {d} | {detail} | {verdict} |")
    A("")
    A("---")
    A("")
    A("## 5. 责任名 → 检索域（**显式决策表**，逐条可复核）")
    A("")
    A("为什么不能自动推：`关键词库-v2.md` 的域词条是**英文**，L3 责任名是**中文**。")
    A("实测用「中文词 ∩ 英文词」做词面匹配，**31 条全部得 0 分** ——")
    A("那个仪器看不见这种对应关系（同本仓库铁律：**判某个东西不存在之前，先问仪器能不能看见它**）。")
    A("")
    A("| 位次 | 责任名 | 检索域（主 / 次） | 选择理由 |")
    A("|---:|---|---|---|")
    for q in sorted(queries, key=lambda x: x["rank"]):
        doms = " / ".join(f"`{d}`" for d in q["query_domains"])
        A(f"| {q['rank']} | {q['responsibility']} | {doms} | {L3_NOTE.get(q['responsibility'], '—')} |")
    A("")
    A("**已装线 `src_domain` → filter 域的桥接**（9 个域在 filter 词汇表之外，"
      "本表只登记实际用到的 3 条）：")
    A("")
    A("| 源域 | 折到 | 语义缺口 |")
    A("|---|---|---|")
    for k, v in DOMAIN_BRIDGE.items():
        A(f"| `{k}` | `{v}` | 登记，不擅自放宽 |")
    A("")
    A("---")
    A("")
    A("## 6. 检索式全文（逐条）")
    A("")
    A("每条包含：**正向词**（三段式的第一段）×  **负向词**（第二段，按域生效）× "
      "**约束词**（第三段）+ 契约挂载点。")
    A("")
    for q in sorted(queries, key=lambda x: x["rank"]):
        A(f"### 位次 {q['rank']} · {q['responsibility']} · {q['role_title']}（{q['domain']}）")
        A("")
        A(f"- 服务性：`{q['serviceability']}`　|　检索域："
          f"{'、'.join('`' + d + '`' for d in q['query_domains'])}"
          + (f"　|　重灾区域：{'、'.join('`' + d + '`' for d in q['strict_ecommerce_domains'])}"
             if q["strict_ecommerce_domains"] else ""))
        A(f"- **入卡挂载点**：`responsibility: {q['responsibility']}`　"
          f"`contract_id: {q['contract_id'] or '（无正式契约）'}`　"
          f"（匹配 {q['contract_match']}）")
        A(f"- 服务格（M 格，{len(q['m_cells'])} 个）："
          f"{'、'.join('`' + c + '`' for c in q['m_cells'])}")
        A(f"- **正向词**（{len(q['positive'])}）：{'、'.join('`' + t + '`' for t in q['positive'])}")
        if q["positive_sources"]["junction"]:
            A(f"  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）："
              f"{'、'.join('`' + t + '`' for t in q['positive_sources']['junction'])}")
        else:
            A("  - ⚠️ **无交叉项**（该责任无 legacy 卡可取材）—— 见 §7 未做项")
        A(f"- **负向词**（{len(q['negative'])}，全局 + 本域）："
          f"{'、'.join('`' + t + '`' for t in q['negative'])}")
        for sc in q["negative_scope"]:
            A(f"  - 仅 `{sc['domain']}` 生效：{'、'.join('`' + t + '`' for t in sc['terms'])}")
        A(f"- **约束词**（{len(q['constraint'])}）："
          f"{'、'.join('`' + t + '`' for t in q['constraint'])}")
        A("")
    A("---")
    A("")
    A("## 6b. 端到端变异（**每条判据都要配篡改样本**）")
    A("")
    A("本仓库的纪律原话：**每新增一条断言，必须同时加一份篡改样本** ——")
    A("曾有 4 处断言因为真实数据本来就满足它，把断言整行删掉自检照样全绿（= 没断言）。")
    A("所以这里做**两层**证明，`--selftest`（单元级）与 `--mutate`（端到端级）：")
    A("")
    m = notes["mutations"]
    A(f"### 层一 · 单元级 `--selftest`：**20 用例 · 26 变异样本**")
    A("")
    A("把判据函数喂给合成输入，证明**那个函数**会报警。这是必要条件，不是充分条件 ——")
    A("它证明不了「判据在真实数据上会失败」。")
    A("")
    A(f"### 层二 · 端到端 `--mutate`：在**真实数据**上逐条弄坏判据")
    A("")
    A(f"**值变异**（改变一个真实取值让它违反判据）：**{m['n_fired']}/{m['n_mutations']} 判红**，"
      f"覆盖判据 **{len(m['judges_covered'])}/10**")
    A("")
    A("| 判据 | 值变异 | 打红 | 触发的判据原文 |")
    A("|---|---|---|---|")
    for code in ("J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"):
        for i, x in enumerate(m["by_judge"].get(code, [])):
            icon = "✅" if x["fired"] else "❌"
            trig = (x["trigger"] or "**没打红**").replace("|", "\\|")
            A(f"| {code if i == 0 else ''} | {x['desc']} | {icon} | {trig} |")
    A("")
    A("**删除型变异**（把判据整段删掉/清空）：**2/5 判红**。")
    A("")
    A("> ⚠️ **这两个数必须分开报，不能平均成一个「变异 N/N」。**")
    A("> 删除型里未判红的那 3 条（J2 覆盖率、J7 同词异义反例、J8 域名合法性）")
    A("> **不构成「判据有效」的证据** —— 真实数据本来就满足它们，删掉当然还是绿的。")
    A("> 它们的证据在**值变异**那一栏（三条都打红了）。**把两类混在一起报，")
    A("> 就等于用「删了还是绿」冒充「判据没问题」。**")
    A("")
    A("### 这一节自己抓到的东西（**登记，都是实测**）")
    A("")
    A("| # | 撞出的缺陷 | 形态 | 修法 |")
    A("|---|---|---|---|")
    A("| 1 | J6 把两条**空交叉项**算成 Jaccard=1.0 | 假红（报出一对不存在的「只换责任名」） | "
      "空集之间相似度**无定义**；空交叉项条目从交叉项比对里剔除，单独计数 |")
    A("| 2 | J9 只判「全挂上 / 一条都没挂」 | **单边判据**（28 条里掉 1 条照样过） | "
      "补「挂载数 = 应有数」这一侧 + 索引键核查 |")
    A("| 3 | J10 的 needle 表**内联在判据函数体里** | 自证假红（扫描打中了自己） | "
      "needle 移到模块常量，函数体只留不含字面量的引用 |")
    A("| 4 | J10 用**裸子串**匹配 needle | 假红（输入字段名 `in_first_slice` 含 `first_slice`） | "
      "改词边界正则 `(?<![A-Za-z0-9_])…(?![A-Za-z0-9_])` |")
    A("| 5 | J7 拿 `clinical`/`eeg` 当「同词异义」反例 | **仪器用错**（它们在 `GLOBAL_NEGATIVE` 里，"
      "全域生效，根本不可能是分歧词） | 反例改用按域负向词，并加「全局词不得冒充分歧反例」的断言 |")
    A("| 6 | 变异本身**没施上力**（截短到 28 条时尾部正好都是结构空白 ⇒ 生成数没变） | "
      "假阴性（看起来像「判据无效」，其实是变异没生效） | 先证明变异改变了真实取值，再谈判据 |")
    A("| 7 | `judge_j4` 拿**同一个对象**跟自己比 | 假绿（`build` 级变异永远打不红） | "
      "「重算」必须真的重算（`lambda: _rebuild(...)`） |")
    A("")
    A("---")
    A("")
    A("## 7. 未做 / 继承接项")
    A("")
    for line in notes["todo"]:
        A(f"- {line}")
    A("")
    A("---")
    A("")
    A("## 8. 基线与回归")
    A("")
    A("```json")
    A(json.dumps(baseline, ensure_ascii=False, indent=1))
    A("```")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------
BASELINE = {
    #: 两口径基线：**任一个变了就判红**（登记不重判 —— 变的原因要在报告里写清）
    "strict_worklist": 3,
    "wide_by_role_worklist": 5,
    "ledger_rows_blank": 9,
    "worklist_total": 31,
    "generated": 28,
    "min_junction_min": 0.05,   #: 交叉项**下界**的下限：太低说明交叉项没起作用
    "min_full_min": 0.05,       #: 全向量**下界**的下限：太低说明两条检索式几乎不相交
    #: 全向量上界的上限。**取 0.95 而不是 0.80，理由是可解释性**：
    #: 正向量 = 域词（占绝大多数）+ 交叉项（1–4 个）。**共享同一检索域池**的两条，
    #: 域词几乎逐词相同 ⇒ 全向量 Jaccard 天然就在 0.85–0.93。实测最像的一对
    #: （渠道对账 ↔ 差异追踪，同为 `09-DataAgent-LLM`+`12-ML基础`）是 0.9167，
    #: 而它们的**交叉项**完全不同（`agentic etl` vs `inventory theft warehouse anomaly`）。
    #: ⇒ 把阈值定在 0.80 会稳定误报「模板套壳」，而真正该抓的是「近乎逐字复制」；
    #: 区分同域池与套壳的判据是**交叉项 Jaccard**（阈值 0.80）与**空壳计数**（0）。
    "max_full_max": 0.95,
    "max_junction_max": 0.80,
    "shell_max": 0,             #: 空壳（只换责任名）上限
    #: 空交叉项（取不到材）—— **两个方向都钉**：多了是新增缺口，少了是取材变松。
    #: 基线 = 实测 2 条（线索评估：同名 legacy 卡 0 张；测试方案：唯一同名卡标题无可用英文段）。
    "empty_junction_max": 2,
    "empty_junction_min": 2,
    #: 同检索域池的条目对上限（全向量相似度天然高的那一批；上界钉住防决策表退化）
    "same_pool_max": 6,
    "n_divergent_terms_min": 3, #: 分歧负向词条数下限
    "n_divergent_used_min": 50,      #: 被**真实配对**用上的分歧词种数下限（实测 56/56）
    "n_divergent_carriers_min": 28,  #: 负向段带分歧词的检索式条数下限（实测 28/28）
    "n_contracts": 139,
}


def thresholds_from_baseline(b: dict) -> dict:
    return {"full_max": b["max_full_max"], "junction_max": b["max_junction_max"],
            "junction_min": b["min_junction_min"], "full_min": b["min_full_min"],
            "shell_max": b["shell_max"],
            "empty_junction_max": b["empty_junction_max"],
            "empty_junction_min": b["empty_junction_min"],
            "same_pool_max": b["same_pool_max"]}


def build_queries(inputs: dict, worklist: list[dict]) -> list[dict]:
    """把工单 → 检索式。纯函数（J4 依赖它两次调用逐字节相同）。"""
    ledger = inputs["ledger"]
    legacy_titles: dict[str, list[str]] = defaultdict(list)
    src = DATA.parent / ".." / ".." / "Magpie-Horch"
    del src
    legacy_titles.update(inputs["legacy_titles"])
    cidx = inputs["cidx"]
    out = []
    for item in sorted(worklist, key=lambda x: x["rank"]):
        if item["structural_blank"]:
            continue
        out.append(build_one(inputs["cf"], item, inputs["harvest_groups"],
                             inputs["kw_text"], legacy_titles, cidx["by_key"]))
    del ledger
    return out


def load_legacy_titles() -> dict[str, list[tuple[str, str]]]:
    """已装线 1338 张卡里，**同名 L3** 的卡的 (标题, 来源技术域) —— 交叉项的词源。

    ⚠️ 技术域是**双键的一半**：同一 L3 会挂在多个技术面下（实测「测试方案」与
    「资源情景比较」同 L3 但不同技术面）⇒ 只按 L3 取卡会让两个责任的交叉项撞成同一个。
    """
    p = Path("/Users/lute/project/Magpie-Horch/packages/capabilities/dsh-paper2skills"
             "/data/classification.json")
    if not p.is_file():
        return {}
    prod = json.loads(p.read_text(encoding="utf-8")).get("items", [])
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for it in prod:
        for l3 in (it.get("l3") or []):
            if len(out[l3]) < 120:
                out[l3].append((it.get("title", ""), it.get("src_domain", "")))
    return dict(out)


def synonymy_probe(cf, div: dict) -> tuple[list[tuple[str, str, str, str]], list[dict]]:
    """同词异义三反例的逐条对账，返回 (报告表行, 登记表)。

    **判定来自 instrument，不是来自手写**：`cohort study` 这种有实现的词，
    判定取自 J7 现算的分歧矩阵（`div["divergent"]`）；`trial` / `ad` 这种
    词库有记录、代码无实现的，判定取自 `doc_vs_code_audit`，并**明确写成登记**。
    """
    rows = []
    audit = doc_vs_code_audit(cf, div)
    by_term = {x["term"]: x for x in audit["registered"]}
    checks = [
        ("trial", "词库 §1：「`trial` 只在 14 域生效」；01/02 域是 RCT 方法论词，**不得**列负向",
         "14-用户分析", "01-因果推断"),
        ("cohort", "词库 §1：「医学里是队列研究，与 cohort retention 同词不同义」"
                   "（故 filter 用收窄形式 `cohort study`）",
         "14-用户分析", "01-因果推断"),
        ("ad", "词库 §2 13 域：「`adversarial attack`（ad 歧义）、ADHD、Alzheimer」"
               "（故 filter 用收窄形式 `adversarial attack`）",
         "13-广告分析", "02-A_B实验"),
    ]
    for term, rec, d_neg, d_pos in checks:
        detail = []
        st = div["divergent"].get(term)
        if st:
            detail.append(f"分歧矩阵：{d_neg}={'负向' if st.get(d_neg) else '不判'}、"
                          f"{d_pos}={'负向' if st.get(d_pos) else '不判'}")
        aud = by_term.get(term)
        if aud is None:
            # 词库记录里的词在 filter 里以收窄形式存在 ⇒ 从收窄形式上取实测
            forms = [x for x in cf.DOMAIN_NEGATIVE.get(d_neg, []) if term in x]
            detail.append(f"{d_neg} 负向表={forms or '**无对应实现**'}")
            verdict = ("✅ 以收窄形式实现（口子收紧，方向正确）" if forms
                       else "🔴 **词库有记录、代码无实现**（登记，不擅自改 filter）")
        else:
            forms = aud["narrowed_forms"]
            detail.append(f"{d_neg} 负向表={forms or '**无对应实现**'}")
            verdict = ("✅ 以收窄形式实现" if forms
                       else "🔴 **词库有记录、代码无实现**（登记，不擅自改 filter）")
        rows.append((term, rec, verdict, "；".join(detail)))
    return rows, audit["gaps"]


def run(as_json: bool = False, json_out: Path = OUT_JSON, md_out: Path = OUT_MD,
        quiet: bool = False) -> tuple[int, dict]:
    """返回 (exit_code, summary)。`as_json` 时把产物写到 json_out。"""
    inputs = load_inputs()
    ledger = inputs["ledger"]
    worklist = ledger["worklist"]
    inputs["legacy_titles"] = load_legacy_titles()
    inputs["cidx"] = contract_index(inputs["contracts"])

    queries = build_queries(inputs, worklist)
    readings = structural_blank_readings(worklist, ledger["rows"])
    sim = similarity_report(queries)
    div = negative_divergence(inputs["cf"], queries)
    coverage = {
        "worklist_total": len(worklist),
        "generated": len(queries),
        "rate": (len(queries) / len(worklist)) if worklist else 0.0,
        "excluded_structural_blank": readings["strict_worklist"],
        "retrieval_eligible_strict": readings["retrieval_eligible_strict"],
        "retrieval_eligible_wide": readings["retrieval_eligible_wide"],
    }
    src_text = Path(__file__).read_text(encoding="utf-8")

    judge_errors: dict[str, list[str]] = {}
    judge_errors["J1"] = judge_j1(True, [])
    judge_errors["J2"] = judge_j2(queries, worklist)
    judge_errors["J3"] = judge_j3(queries, readings, BASELINE)
    judge_errors["J4"] = judge_j4(lambda: build_queries(inputs, worklist), queries)
    e5, j5info = judge_j5(lambda wl: build_queries(inputs, wl), worklist)
    judge_errors["J5"] = e5
    judge_errors["J6"] = judge_j6(sim, thresholds_from_baseline(BASELINE))
    judge_errors["J7"] = judge_j7(div, inputs["cf"], queries, inputs["kw_text"])
    judge_errors["J8"] = judge_j8(queries, inputs["cf"], {}, inputs["kw_text"])
    judge_errors["J9"] = judge_j9(queries, inputs["contracts"], inputs["cidx"])
    judge_errors["J10"] = judge_j10(queries, ledger, src_text)

    # 基线下限：分歧词条数
    if div["n_divergent"] < BASELINE["n_divergent_terms_min"]:
        judge_errors["J7"].append(
            f"J7 分歧负向词只有 {div['n_divergent']} 个 < 下限 "
            f"{BASELINE['n_divergent_terms_min']}")
    if div["n_used_terms"] < BASELINE["n_divergent_used_min"]:
        judge_errors["J7"].append(
            f"J7 被真实配对用上的**分歧词**只有 {div['n_used_terms']} 个 < 下限 "
            f"{BASELINE['n_divergent_used_min']}")
    if div["n_queries_carrying_divergent"] < BASELINE["n_divergent_carriers_min"]:
        judge_errors["J7"].append(
            f"J7 负向段里带分歧词的检索式只有 {div['n_queries_carrying_divergent']} 条 "
            f"< 下限 {BASELINE['n_divergent_carriers_min']} —— "
            f"「按域生效」在产物里没有载体")

    #: 端到端变异（值变异）在 --check 里也算一遍 —— 报告必须能回答
    #: 「你说每条判据都会失败，证据呢」。它只做**值变异**（不 spawn 子进程），开销可忽略。
    #: 图谱漂移判据的两侧指纹 —— 在 dict **之前**算好：报告正文是一个列表字面量，
    #: 里面**放不了赋值语句**（首版把 `x = ...` 写进列表 ⇒ SyntaxError）。
    _graph_sha_now = _sha16(REPO / 'paper2skills-vault' / '07-资源库' / 'capability-graph.json')
    _ledger_sha = ledger['_meta'].get('graph_sha256')
    _drift = _graph_sha_now != _ledger_sha

    muts = mutation_table(make_collector(inputs, src_text, worklist), len(queries))
    if muts["judges_uncovered"]:
        judge_errors["J1"].append(
            f"以下判据没有端到端值变异覆盖：{muts['judges_uncovered']} —— "
            f"没被变异证明过的判据 = 没证明过它会失败")
    if muts["n_fired"] != muts["n_mutations"]:
        judge_errors["J1"].append(
            f"端到端值变异 {muts['n_fired']}/{muts['n_mutations']} 打红 —— "
            f"没打红的那条说明对应判据在真实数据上是摆设")

    notes = {
        "mutations": muts,
        "synonymy_probe": synonymy_probe(inputs["cf"], div)[0],
        "todo": [
            "**只生成检索式，不执行检索** —— 本任务（S4）的交付物止于「缺口 → 检索式」。"
            "真正发查询、收割、打分属下一步。",
            "🔴 **`arxiv_harvest.py` 的域标签与 `candidate_filter.py` 有 4 处实测不一致**"
            "（`07-VOC舆情` vs `07-NLP-VOC`、`09-DataAgent` vs `09-DataAgent-LLM`、"
            "`15-营销投放` vs `15-营销投放分析`、`00-电商Agent` 在 filter 里**根本没有**）："
            "实测候选池 1046 篇里 **141 篇**（89 + 37 + 9 + 6）的 `query_groups` **全都**"
            "不在 `DOMAIN_DIR` 里，会被 `filter_pool` 的 `if not domains: keep` 分支"
            "**整篇放行**（负向词与约束词都不生效）。"
            "⚠️ 本数字**首次报成了 113** —— 漏算了 `00-电商Agent` 那 37 篇；"
            "「漏算」的原因是当时只看了「filter 里有同名域」的三对，没查「filter 里压根没有的域」。"
            "**这正是本仓库那条铁律的又一例**：判「某个东西不存在」之前，先问仪器能不能看见它。"
            "本任务用 `harvest_domain_of` 在**本侧**折好这 3 对，`00-电商Agent` 不参与本次检索域，"
            "**不动上游脚本**（纪律 6），已登录。",
            "**结构空白那 3 条不给检索式**（走 SOP）。已装线里与它们同名 L3 的卡为 0，"
            "因此连交叉项都取不到材 —— 这不是本脚本的缺陷，是「结构性空白」的定义。",
            "**线索评估（位次 1）有 legacy 卡为 0**，但账里 `structural_blank=false` ⇒ "
            "它**进入检索预算**、却**没有交叉项**。已在 §6 逐条标注 ⚠️，请所有者裁决"
            "（补齐它的 legacy 归类，或接受它只有域词）。",
            "**契约挂载点覆盖率**见 J9 —— `responsibility` 与 `contract_id` 的键名**照抄** "
            "`check_contracts.py`，没有自造键名。",
            "**排序权重不进任何判定**：J10 用 AST 扫本文件的 `judge_*` 函数体，"
            "并配反向探针（往判据函数体注入 `zero_gap`，扫描必须报出来）。",
            "🔴 **登记（不重判）：`关键词库-v2.md` §1 与 `candidate_filter.py` 实测不符** —— "
            "词库说「负向词 `trial` 只在 14 域生效」，实测 `trial` **在 filter 的任何域的负向表里"
            "都不存在**（`14-用户分析` 用的是收窄形式 `cohort study` / `pet scan`，但没有 `trial`；"
            "`02-A_B实验` 不列 `trial` 是对的，但 14 域也没列）。`ad` 同理（用 `adversarial attack` / "
            "`adhd` / `alzheimer` 收窄实现）。按纪律**只登记不重判、不擅自改 filter**；"
            "同时配了一条能打红的判据：**若哪天 filter 真把 `trial` 列进 01/02 域（即开始误杀 RCT "
            "方法论论文）必须报红**（`FORBIDDEN_NEGATIVES`，selftest 里有对应变异）。",
            # ⚠️ 这一段**必须条件化**（`_drift` 在 dict 之前算好 —— 列表里放不了语句）。
            # 首版是**冻结的一句话**：不管指纹是否相同都写「上游图谱被改动过 ⇒ 工单可能已滞后」，
            # 于是**上游漂移被消除之后，这句话本身变成了假话** ——
            # 实测：重生成缺口账后两侧指纹都成了 `ecda4970…`，报告却仍写着
            # 「记的是 ecda49…，而现在是 ecda49… ⇒ 工单可能已滞后」（自相矛盾），
            # 并仍断言「`build_gap_ledger.py --check` 现在确实判红」（此时已绿）。
            # ⇒ **「登记」不是「刻石」**：一条登记的成立条件变了，它就得跟着变。
            # 这正是本仓库反复记的那一族（#23/#24：账落后于实物；#88：把「本次没命中」当「过期」）。
            (
                "🔴 **登记（不重判）：上游图谱在本任务执行期间被改动过** —— "
                f"工单账 `_meta.graph_sha256` 记的是 `{_ledger_sha}`，"
                f"而 `capability-graph.json` 现在的 sha256 前 16 位是 `{_graph_sha_now}`"
                "（文件 mtime 也晚于工单生成时间）。⇒ **工单可能已滞后于图谱**。"
                "本任务（S4）只消费工单、不重建它（纪律 6：不改缺口账生成器），"
                "故只登记：下一次跑 F3 的 `--check` 会由**它自己的**判据报出来。"
                "产物的 `_input_fingerprints` 已把两侧指纹都记下，消费方可自查。"
                "实测确认：`build_gap_ledger.py --check` 现在确实判红"
                "（红在「工单与图谱/分类现状不一致」），而那是**上游改动**造成的，不是本脚本改的。"
                if _drift else
                f"✅ **无漂移（本条前身是一条真实漂移登记，现已消除）** —— 工单账与图谱的指纹"
                f"**两侧相同**（均 `{_graph_sha_now}`）⇒ 工单不再滞后于图谱，"
                "`build_gap_ledger.py --check` 已转绿（由门禁 **L3a** 判）。"
                "⚠️ 本行的**前身**（2026-09-13）登记过一次真实漂移"
                "（`be36dd1d197def0e` → `ecda4970bbb7b56b`，因图谱新增 `closed_items`），"
                "那次确实由 F3 自己的 `--check` 报了出来、并已重生成。"
                "**保留这句是为了说明「漂移是被消除的，不是被忽略的」** —— "
                "而不是继续声称「仍可能滞后」。"
            ),
        ],
    }

    #: 判据表带**反面**列（「这条判据在抓什么错」）—— 只写名字的判据表是一张装饰。
    #: (判据代号, 判据, **反面：这条判据在抓什么错**, 实测)。三元组做成四元组，
    #: 是为了让报告里的判据表**自带反面** —— 只有名字的判据表是装饰。
    jrows = [
        ("J1", "输入齐备性",
         "缺输入却报「通过」——「没东西可查」被当成「查过了没问题」",
         "✅ 4/4 输入到位；缺任一 ⇒ exit 2（合成环境下实测 exit 2）"
         if not judge_errors["J1"] else "🔴 " + judge_errors["J1"][0]),
        ("J2", "覆盖率是一等输出：生成数 / 工单数，生成 0 条判红",
         "一条都没生成却报「全部通过」",
         f"✅ {coverage['generated']}/{coverage['worklist_total']}"
         f"（{coverage['rate']:.1%}）；空输入 ⇒ exit 2 已实测"
         if not judge_errors["J2"] else "🔴"),
        ("J3", "结构空白显式排除 + **两口径**逐个对基线",
         "给「走 SOP、不占检索预算」的条目生成检索式；或只对一种口径成立就报过",
         f"✅ 严 {readings['strict_worklist']} / 宽 {readings['wide_by_role_worklist']} / "
         f"账内 {readings['ledger_rows_blank']}（三个数各自与基线相等）"
         if not judge_errors["J3"] else "🔴"),
        ("J4", "可复算：同输入两次运行逐字节相同",
         "字典序/时间戳进产物 ⇒ 检索式不可复算；或**改了内容仍报「逐字节相同」**（假绿，"
         "以「反向控制：交换词序必须变」兜住）",
         "✅ 逐字节相同（md5 两次一致）+ 反向控制生效"
         if not judge_errors["J4"] else "🔴"),
        ("J5", "**改了工单，检索式必须变**（当作失败条件断言）",
         "假仪表：改责任名/域/岗位，输出岿然不动（等于所有工单共用一张模板）；"
         "反面用「改无关字段必须**不**变」兜住「什么都变」",
         f"✅ 责任名 {j5info['perturbed_changed'].get('name')} 条 / "
         f"域 {j5info['perturbed_changed'].get('domain')} 条 / "
         f"岗位 {j5info['perturbed_changed'].get('role')} 条发生变化；"
         f"改无关字段 0 条变化"
         if not judge_errors["J5"] else "🔴"),
        ("J6", "不同工单的检索式彼此不同（相似度分布 + 空壳）",
         "模板套壳（31 条只换责任名）。四个方向都判：全向量上界、交叉项上界、"
         "空壳计数、**空交叉项计数（两个方向）**、同域池条目对上限",
         f"✅ 交叉项 Jaccard max {sim['junction']['max']}（均 "
         f"{sim['junction']['mean']}）／全向量 max {sim['full']['max']}；"
         f"空壳 {sim['shell_count']} 条；空交叉项 {sim['empty_junction_count']} 条；"
         f"同域池条目对 {len(sim['same_domain_set_pairs'])} 对"
         if not judge_errors["J6"] else "🔴"),
        ("J7", "**负向词按域生效**，不是全局清单",
         "无脑全局负向 ⇒ 误杀方法论论文（`trial`/`cohort`/`ad`）；"
         "或「按域」只是文档说法、代码里其实全域生效",
         f"✅ 从 filter **现算**分歧矩阵：{div['n_divergent']} 个词在不同域状态相反，"
         f"{div['n_used_terms']} 个被**真实配对**用上，"
         f"载体 {div['n_queries_carrying_divergent']} 条"
         if not judge_errors["J7"] else "🔴"),
        ("J8", "产物能被 `candidate_filter.py` **直接消费**（同一个配置格式）",
         "另造一套格式 ⇒ 生成完还要人肉翻译，检索式形同废纸；"
         "或域名不在 `DOMAIN_DIR` 里 ⇒ filter **静默跳过**（保留全部 = 没过滤）",
         "✅ 域名全在 `DOMAIN_DIR`；负向词/约束词与 filter 逐项相等（3 域走 "
         "`harvest_domain_of` 桥接，已登记）" if not judge_errors["J8"] else "🔴"),
        ("J9", "契约挂载点：`responsibility` / `contract_id` 与 "
         "`check_contracts.py` 键名对齐",
         "自己发明键名 ⇒ 卡产出后无处可挂；或一份没挂上却报过；"
         "或同一责任两份正式契约时静默挑一份",
         f"✅ {sum(1 for q in queries if q['contract_id'])}/{len(queries)} 条挂上契约"
         f"（契约池 {len(inputs['contracts'])} 份）" if not judge_errors["J9"] else "🔴"),
        ("J10", "排序权重**不进任何判定**（风险 N1）",
         "把 `zero_gap`/`score` 之类排序分当成放行条件 ⇒ 分数决定谁过闸；"
         "反面：全文扫会打中文档注释（假红），故按 AST 取判据函数体",
         "✅ AST 扫 11 个判据函数、0 命中；反向探针注入即被抓；"
         "反向控制（注入非判据函数**不**报）生效"
         if not judge_errors["J10"] else "🔴"),
    ]

    baseline = dict(BASELINE)
    baseline["measured"] = {
        "generated": len(queries),
        "junction_max": sim["junction"]["max"],
        "full_max": sim["full"]["max"],
        "shell_count": sim["shell_count"],
        "n_divergent_terms": div["n_divergent"],
        "n_divergent_used": div["n_used_terms"],
        "n_divergent_carriers": div["n_queries_carrying_divergent"],
        "n_contracts": len(inputs["contracts"]),
        "mounted": sum(1 for q in queries if q["contract_id"]),
    }

    if as_json:
        payload = {
            "_what": "PHASE6 S4 缺口驱动检索式 · 机读产物。"
                     "三段式（正向/负向/约束）供 candidate_filter.py 直接消费。",
            "_generator": "paper2skills-research/scripts/build_search_queries.py",
            "_inputs": {
                "worklist": str(LEDGER.relative_to(REPO)),
                "keyword_lib": str(KEYWORDS.relative_to(REPO)),
                "filter": str(FILTER_PY.relative_to(REPO)),
                "contracts": str(CONTRACTS_DIR.relative_to(REPO)),
            },
            "_determinism": "同一输入两次运行逐字节相同（J4）；无时间戳、无随机。",
            #: ⚠️ **输入指纹**：上游任一输入变了，消费方必须能看出来。
            #: 实测（本任务执行期间）：`capability-graph.json` 在我开工后被另一个任务改过
            #: （sha256 `7ede8be1…` → `be36dd1d…`），而工单是按旧图谱生成的 ⇒
            #: 工单本身可能已经滞后于图谱。**登记不重判**：工单是 F3 的产物，重建它不属于 S4。
            "_input_fingerprints": {
                "gap_ledger_sha256_16": _sha16(LEDGER),
                "gap_ledger_generated": ledger["_meta"].get("generated"),
                "gap_ledger_graph_sha256_16": ledger["_meta"].get("graph_sha256"),
                "capability_graph_sha256_16_now": _sha16(
                    REPO / "paper2skills-vault" / "07-资源库" / "capability-graph.json"),
                "keyword_lib_sha256_16": _sha16(KEYWORDS),
                "candidate_filter_sha256_16": _sha16(FILTER_PY),
                "contracts_dir": str(CONTRACTS_DIR.relative_to(REPO)),
                "n_contracts": len(inputs["contracts"]),
            },
            "_filter_format": {
                "_what": "与 candidate_filter.py 的 DOMAIN_NEGATIVE / DOMAIN_CONSTRAINT "
                         "同键名、同语义；域名取自 DOMAIN_DIR。",
                "GLOBAL_NEGATIVE": list(dict.fromkeys(inputs["cf"].GLOBAL_NEGATIVE)),
                "DOMAIN_NEGATIVE": {k: list(v) for k, v in
                                    sorted(inputs["cf"].DOMAIN_NEGATIVE.items())},
                "DOMAIN_CONSTRAINT": {k: list(v) for k, v in
                                      sorted(inputs["cf"].DOMAIN_CONSTRAINT.items())},
                "CONSTRAINT_FALLBACK": CONSTRAINT_FALLBACK,
                "STRICT_ECOMMERCE_DOMAINS": sorted(inputs["cf"].STRICT_ECOMMERCE_DOMAINS),
            },
            "coverage": coverage,
            "structural_blank": readings,
            "similarity": sim,
            "negative_divergence": div,
            "contract_index": {
                "n_contracts": len(inputs["contracts"]),
                "n_mounted": sum(1 for q in queries if q["contract_id"]),
                "unknown_responsibilities": sorted(
                    {q["responsibility"] for q in queries if not q["contract_id"]}),
            },
            "queries": queries,
            "baseline": baseline,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                       sort_keys=True), encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_md(queries, sim, div, readings, coverage, jrows, baseline,
                                notes), encoding="utf-8")

    if not quiet:
        print("=" * 78)
        print("PHASE6 S4 · 缺口驱动检索式")
        print("=" * 78)
        print(f"靶区工单 {coverage['worklist_total']} 条 → 生成 {coverage['generated']} 条"
              f"（覆盖率 {coverage['rate']:.1%}）"
              f"｜结构空白排除 {readings['strict_worklist']} 条（严口径）")
        print()
        for code, name, _rev, got in jrows:
            print(f"  {code:<4} {name}")
            print(f"       {got}")
        print()
        bad = {k: v for k, v in judge_errors.items() if v}
        if bad:
            print("🔴 判红：")
            for k, v in bad.items():
                for line in v:
                    print(f"   [{k}] {line}")
        else:
            print("✅ 全部判据通过（10/10）")
        print()
        print(f"→ {json_out}")
        print(f"→ {md_out}")

    summary = {"coverage": coverage, "judge_errors": judge_errors, "similarity": sim,
               "divergence": {"n_divergent": div["n_divergent"],
                              "n_used_terms": div["n_used_terms"],
                              "carriers": div["n_queries_carrying_divergent"]},
               "readings": readings, "mutations": mutation_summary(inputs, worklist)}
    return (EXIT_RED if any(judge_errors.values()) else EXIT_OK), summary


# ---------------------------------------------------------------------------
# 6b. 端到端变异：**在真实数据上**证明每条判据会打红
# ---------------------------------------------------------------------------
#: ⚠️ 为什么需要**两种**变异（这是实测撞出来的）：
#:
#: `--selftest` 的变异是**单元级**的 —— 它们把判据函数喂给合成输入，证明那个函数会报错。
#: 但那还不够：本仓库有过 4 处断言「因为真实数据处理上本来就满足它，把断言整行删掉自检
#: 照样全绿」。所以还要来一遍**端到端变异**：在真实输入上把判据弄坏，跑完整的 `--check`，
#: 看退出码是不是真的从 0 变成 1。
#:
#: 端到端变异分两类，**必须分开报**（实测数据说明为什么）：
#:
#: · **删除型**（把判据整段删掉/恒真化）：只有当真实数据**违反过**它时才会打红。
#:   实测 9 条里只有 3 条打红 —— 另外 6 条的真实数据本来满足判据，
#:   删掉它当然还是绿的。**这正是「断言是摆设」的定义**，所以这类变异的漏检要**如实报**，
#:   不能拿它当「判据有效」的证据。
#: · **值变异**（改变一个**真实取值**让它违反判据）：无论真实数据本来怎样都必须打红。
#:   这才是「这条判据会失败」的**硬证据**。
#:
#: 两类都跑、分开计数 —— 只跑一类等于只做了一半的证明。
#: 每条判据**至少一条值变异**。`mutation_coverage()` 会断言「10 条判据都被覆盖到」——
#: 漏掉一条判据的变异 = 那条判据没有被端到端证明过。
VALUE_MUTATIONS = [
    # (代号, 说明, 施力点, 施加函数)
    #: 口径：工单 31 条、结构空白 3 条 ⇒ 应生成 28 条。
    #: 截到 10 条（其中 2 条结构空白）⇒ 生成 8 条，而「工单 10 − 空白 2 = 8」自洽，
    #: 所以这条变异抓的是 **J2/J3 与基线**的差（基线钉的是 31/3/28）。
    #: 第一版写成截到 28 条，结果尾部 3 条正好是结构空白 ⇒ 生成数仍是 28，
    #: 变异**没有改变任何真实取值** ⇒ 没打红。这是「变异样本本身没施上力」的典型：
    #: **变异不生效 ≠ 判据无效**，必须先把变异改到真的施上力再谈判据。
    ("J2", "把工单截短成 10 条 ⇒ 生成数 8 ≠ 基线 28（覆盖率读数随之改变）",
     "worklist", lambda o: o[:10]),
    ("J3", "把工单里 2 条标成结构空白 ⇒ 严口径 5 ≠ 基线 3",
     "worklist", lambda o: [dict(w, structural_blank=True) if w["rank"] in (1, 2) else w
                            for w in o]),
    ("J3", "把全部结构空白标记清掉 ⇒ 生成 31 条 ≠ 可检索位 28",
     "worklist", lambda o: [dict(w, structural_blank=False) for w in o]),
    #: J4 的变异必须**施加在「两次构建」上**，不能施加在「一次构建的产物」上。
    #: 第一版用 `query` 级变异给产物加时间戳 —— 但那个时间戳是**同一个对象**，
    #: 判据拿它跟自己比，永远相等，所以打不红。这又是一次「变异没施上力」。
    #: 现在用 `build` 级：每次构建现场取一个时间戳 ⇒ 两次构建天然不同 ⇒ 必须打红。
    ("J4", "让每次构建都现场取一个时间戳 ⇒ 两次运行不再逐字节相同",
     "build", lambda qs: [dict(q, burned_at=__import__("time").time()) for q in qs]),
    #: J5 的变异必须**只改一条**（判据的口径是「改一条 ⇒ 对应检索式要变」）；
    #: 第一版把所有条都改成同一个责任名，结果 L3_DOMAINS 里查得到它，
    #: 而且**改完前后仍能被 domain 变异抓住**，读数混在一起。改成只动位次 1 与 2。
    ("J5", "把位次 1/2 两条工单都改写成同一责任名+同一域 ⇒ 对应检索式应随之变化",
     "worklist", lambda o: [dict(w, l3="渠道对账", role_id="AGT-040",
                                role_title="改过", domain="财务与合规")
                            if w["rank"] in (1, 2) else w for w in o]),
    ("J6", "把所有交叉项统一成同一个词 ⇒ 交叉项 Jaccard = 1.0（模板套壳）",
     "query", lambda o: [dict(q, positive_sources=dict(
         q["positive_sources"], junction=["same shell term"])) for q in o]),
    ("J6", "把所有正向词统一成同一个列表 ⇒ 全向量 Jaccard = 1.0",
     "query", lambda o: [dict(q, positive=["one", "same", "vector"]) for q in o]),
    ("J7", "把 4 组同词异义反例换成全局负向词（仪器用错）",
     "pairs", lambda: (("clinical", "01-因果推断", "02-A_B实验"),
                         ("patient", "01-因果推断", "02-A_B实验"),
                         ("mri", "14-用户分析", "01-因果推断"),
                         ("eeg", "14-用户分析", "01-因果推断"))),
    ("J8", "把一个检索域换成 candidate_filter 不认识的域名 ⇒ filter 会静默跳过",
     "query", lambda o: [dict(q, query_domains=[q["query_domains"][0], "99-不存在的域"])
                         if q["rank"] == 1 else q for q in o]),
    #: J1 的变异走 `j1` 施力点：**把「输入齐备」这个事实改成假**。
    ("J1", "把「输入齐备」改成假 ⇒ 输入没拿到必须判红（不是「通过」）",
     "j1", None),
    #: J9 的两侧：一份都没挂上（全空）与**只掉一条**（覆盖率不是全有全无）。
    ("J9", "把**全部**契约挂载点清空 ⇒ 一条都没挂上",
     "query", lambda o: [dict(q, contract_id=None) for q in o]),
    ("J9", "只把**一条**契约挂载点清空 ⇒ 掉了一条也算掉（不许全有全无）",
     "query", lambda o: [dict(q, contract_id=None) if q["rank"] == 1 else q for q in o]),
    ("J10", "把工单的 weights 塞进产物 ⇒ 排序分数进了产物",
     "query", lambda o: [dict(q, weights={"zero_gap": 3.0}) if q["rank"] == 1 else q
                         for q in o]),
]


def _rebuild(inputs: dict, worklist: list[dict], muts: dict) -> list[dict]:
    """在**已被变异过的**工单上重建检索式，再按层级施加 query / build 级变异。

    三档施力点（必须分清，因为它们的**证据强度**不同）：
      · `worklist` —— 改输入（工单），跑完整生成链；
      · `query`    —— 改**一次**构建的产物（能测「产物里有没有违规内容」）；
      · `build`    —— 改**每一次**构建的输出（只有它能测「两次运行是否相同」，
                      因为 `query` 级的产物是同一个对象，拿它跟自己比永远相等）。
    """
    qs = build_queries(inputs, worklist)
    for m in muts.get("query", []):
        qs = m(qs)
    for m in muts.get("build", []):
        qs = m(build_queries(inputs, worklist))
    return qs


def mutation_table(collect, n_base: int) -> dict:
    """不打印，只算出「每条判据被哪条值变异打红」——供报告直接引用。"""
    per: dict[str, list[dict]] = {}
    for code, desc, site, fn in VALUE_MUTATIONS:
        muts = {"worklist": [fn]} if site == "worklist" else \
               {"query": [fn]} if site == "query" else \
               {"build": [fn]} if site == "build" else \
               {"j1": True} if site == "j1" else \
               {"pairs": fn() if callable(fn) else fn}
        try:
            got = collect(muts)
            err = list(got.get(code) or [])
        except Exception as e:  # noqa: BLE001
            err = [f"变异触发内部错误：{type(e).__name__}: {e}"]
        if any("变异触发内部错误" in x for x in err):
            err = []
        per.setdefault(code, []).append(
            {"desc": desc, "fired": bool(err), "trigger": err[0] if err else None})
    covered = sorted(per)
    return {
        "n_base": n_base,
        "by_judge": per,
        "n_mutations": len(VALUE_MUTATIONS),
        "n_fired": sum(1 for v in per.values() for x in v if x["fired"]),
        "judges_covered": covered,
        "judges_uncovered": [c for c in ("J1", "J2", "J3", "J4", "J5", "J6",
                                         "J7", "J8", "J9", "J10") if c not in covered],
    }


def make_collector(inputs: dict, src_text: str, baseline_worklist: list[dict]):
    """把「在给定变异下跑一遍**全部判据**」做成可复用的闭包。

    ⚠️ `--check` 与 `--mutate` **共用同一份实现**是刻意的：两边各写一套判据调用会漂移，
    而「`--mutate` 证明的是 `--check` 的判据」这句话就不成立了。本仓库有过
    「生成器与门禁各有一份判据」的坑（S12 把它修成了单一实现），这里照做。

    ⚠️ `judge_j4` 的「重算」必须真的重算（`lambda: _rebuild(inputs, wl, muts)`），
    **不能**写成 `judge_j4(lambda: qs, qs)` —— 那是拿同一个对象跟自己比，永远相等，
    于是 `build` 级的「时间戳」变异打不红。实测撞到过。
    """
    ledger = inputs["ledger"]

    def collect(muts: dict | None = None) -> dict:
        """按变异规格跑一遍全部判据。`muts` 的键 = 施力点。"""
        muts = muts or {}
        wl = baseline_worklist
        for m in muts.get("worklist", []):
            wl = m(wl)
        qs = _rebuild(inputs, wl, muts)
        readings = structural_blank_readings(wl, ledger["rows"])
        sim = similarity_report(qs)
        div = negative_divergence(inputs["cf"], qs)
        return {
            "J1": judge_j1(not muts.get("j1"), ["合成环境：靶区工单不存在"]),
            "J2": judge_j2(qs, wl),
            "J3": judge_j3(qs, readings, BASELINE),
            "J4": judge_j4(lambda: _rebuild(inputs, wl, muts), qs),
            "J5": judge_j5(lambda wl2: build_queries(inputs, wl2), wl)[0],
            "J6": judge_j6(sim, thresholds_from_baseline(BASELINE)),
            "J7": judge_j7(div, inputs["cf"], qs, inputs["kw_text"],
                           pairs=muts.get("pairs")),
            "J8": judge_j8(qs, inputs["cf"], {}, inputs["kw_text"]),
            "J9": judge_j9(qs, inputs["contracts"], inputs["cidx"]),
            "J10": judge_j10(qs, ledger, src_text),
        }

    return collect


def run_mutations() -> int:
    """端到端变异：在真实数据上把判据弄坏，逐条看退出码是否变成 1。"""
    inputs = load_inputs()
    inputs["legacy_titles"] = load_legacy_titles()
    inputs["cidx"] = contract_index(inputs["contracts"])
    ledger = inputs["ledger"]
    worklist = ledger["worklist"]
    src_text = Path(__file__).read_text(encoding="utf-8")

    base_q = build_queries(inputs, worklist)
    collect = make_collector(inputs, src_text, worklist)
    base = collect()
    return _report_mutations(collect, base, len(base_q), src_text)


def _report_mutations(collect, base: dict, n_base: int, src_text: str) -> int:
    print("=" * 78)
    print("PHASE6 S4 · 端到端变异（真实数据）")
    print("=" * 78)
    print(f"基线：生成 {n_base} 条；判红判据 {[k for k, v in base.items() if v] or '无'}")
    print()
    print("（A）**值变异** —— 改变一个真实取值让它违反判据，**必须**打红")
    print("-" * 78)
    a_ran = a_red = 0
    for code, desc, site, fn in VALUE_MUTATIONS:
        a_ran += 1
        try:
            muts = {"worklist": [fn]} if site == "worklist" else \
                   {"query": [fn]} if site == "query" else \
                   {"build": [fn]} if site == "build" else \
                   {"j1": True} if site == "j1" else \
                   {"pairs": fn() if callable(fn) else fn}
            got = collect(muts)
            #: ⚠️ 「变异触发内部错误」**不算判据打红** —— 那是脚本崩了，不是判据抓到了。
            #: 本仓库的退出码约定里，这两件事必须可区分（3 ≠ 1）。
            if got.get(code) and any("变异触发内部错误" in x for x in got[code]):
                got = {k: ([] if k == code else v) for k, v in got.items()}
        except Exception as e:  # noqa: BLE001
            got = {code: [f"变异触发内部错误：{type(e).__name__}: {e}"]}
        fired = bool(got.get(code))
        a_red += fired
        print(f"  {'✅' if fired else '❌'} [{code}] {desc}")
        if not fired:
            print(f"       ⚠️ 值变异没有让 {code} 打红 —— 这条判据在真实数据上是摆设")
        elif got.get(code):
            print(f"       触发：{got[code][0][:110]}")
    print("-" * 78)
    print(f"  值变异判红 {a_red}/{a_ran}"
          f"{'（全部判据在真实数据上都会失败 ✅）' if a_red == a_ran else '（有判据是摆设 ❌）'}")
    covered = sorted({c for c, *_ in VALUE_MUTATIONS})
    print(f"  覆盖判据 {len(covered)}/10：{covered}")
    miss = [c for c in ("J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10")
            if c not in covered]
    if miss:
        print(f"  ❌ 以下判据没有端到端值变异：{miss}")
    print()
    print("（B）**删除型变异** —— 把判据整段删掉（证明「删了它就会放行假绿」）")
    print("-" * 78)
    dels = [
        ("J2", "覆盖率判据恒真", 'if not queries:\n        errs.append("J2 一条检索式都没生成',
         'if False:\n        errs.append("J2 一条检索式都没生成'),
        ("J3", "不再排除结构空白", 'if item["structural_blank"]:\n            continue',
         'if False:\n            continue'),
        ("J8", "域名合法性检查删掉", 'unknown = sorted(covered - set(cf.DOMAIN_DIR))',
         'unknown = []'),
        ("J7", "同词异义反例检查整段删掉",
         'for term, d_neg, d_pos in (pairs if pairs is not None else SYNONYMY_PAIRS):\n'
         '        if term not in div["divergent"]:',
         'for term, d_neg, d_pos in ():\n        if term not in div["divergent"]:'),
        ("J10", "权重 needle 表清空",
         'WEIGHT_NEEDLES = ("zero_gap", "flow_breadth", "first_slice", "boundary_penalty",',
         'WEIGHT_NEEDLES = ("a", "b", "c", "d",'),
    ]
    b_ran = b_red = 0
    for code, desc, old, new in dels:
        if old not in src_text:
            print(f"  ⚠️ [{code}] {desc}：锚点未命中（这条没跑，**不算通过**）")
            continue
        b_ran += 1
        clone = src_text.replace(old, new, 1)
        with tempfile.TemporaryDirectory() as td:
            P = Path(__file__).parent / "build_search_queries.__mutprobe__.py"
            try:
                P.write_text(clone, encoding="utf-8")
                r = subprocess.run(
                    [sys.executable, str(P), "--check", "--quiet",
                     "--json-out", str(Path(td) / "q.json"), "--md-out", str(Path(td) / "q.md")],
                    capture_output=True, text=True)
            finally:
                P.unlink(missing_ok=True)
        fired = r.returncode == 1
        b_red += fired
        print(f"  {'✅' if fired else '⚠️'} [{code}] {desc}")
        print(f"       退出码 {r.returncode}"
              + ("（删了判据就放行 ⇒ 该判据**在真实数据上确实在起作用**）" if fired
                 else "（删了还是绿 ⇒ 真实数据本来满足它，**这条删除变异不构成证据**，"
                      "证据在上面的值变异里）"))
    print("-" * 78)
    print(f"  删除型判红 {b_red}/{b_ran} —— "
          f"未判红的条目**如实登记**，不得当成「判据有效」的证据")
    print()
    ok = (a_red == a_ran) and not miss and (b_ran == len(dels)) and not any(base.values())
    print(f"✅ 端到端变异通过：10 条判据在真实数据上都会失败"
          f"（值变异 {a_red}/{a_ran}，判据覆盖 {len(covered)}/10），基线本身全绿"
          if ok else "❌ 端到端变异未通过")
    if b_red < b_ran:
        print(f"   登记：删除型变异 {b_red}/{b_ran} 判红 —— 未判红的那几条"
              f"**不构成「判据有效」的证据**（真实数据本来满足它们），"
              f"证据是上面的值变异。这类数字**不得被平均成一个「变异 N/N」**。")
    return EXIT_OK if ok else EXIT_RED


# ---------------------------------------------------------------------------
# 7. 自检：**每条判据都要配一份篡改样本**
# ---------------------------------------------------------------------------
def _mini_inputs():
    """自检用的**合成**输入（不读真实数据，证明判据本身会报警）。"""
    cf = _load_module(FILTER_PY, "_s4_cf_selftest")
    groups = {
        "04-供应链": ['abs:"inventory management" OR abs:"supply chain"'],
        "03-时间序列": ['abs:"demand forecasting" OR abs:"sales prediction"'],
        "14-用户分析": ['abs:"cohort retention" OR abs:"funnel analysis"'],
        "01-因果推断": ['abs:"uplift modeling" OR abs:"causal inference"'],
        "02-A_B实验": ['abs:"A/B testing" OR abs:"experiment design"'],
        "12-ML基础": ['abs:"feature engineering" OR abs:"calibration"'],
        "13-广告分析": ['abs:"attribution" OR abs:"marketing mix model"'],
        "05-推荐系统": ['abs:"recommendation system" OR abs:"collaborative filtering"'],
        "06-增长模型": ['abs:"churn prediction" OR abs:"customer lifetime value"'],
        "07-NLP-VOC": ['abs:"product review summarization"'],
        "08-知识图谱": ['abs:"knowledge graph" OR abs:"entity linking"'],
        "09-DataAgent-LLM": ['abs:"text-to-SQL" OR abs:"data analysis agent"'],
        "10-MAS": ['abs:"multi-agent system" OR abs:"agent orchestration"'],
        "15-营销投放分析": ['abs:"promotion effectiveness" OR abs:"price promotion"'],
    }
    kw = KEYWORDS.read_text(encoding="utf-8")
    #: (标题, 来源技术域) —— 域要与 L3_DOMAINS 里给 A/B 的域对上，否则交叉项为空
    legacy = {"A": [("Alpha Beta Inventory Optimization — x", "04-供应链")],
              "B": [("Gamma Delta Demand Forecast — y", "03-时间序列")]}
    return cf, groups, kw, legacy


def _mini_worklist() -> list[dict]:
    return [
        {"l3": "A", "role_id": "AGT-001", "role_title": "岗位一", "domain": "D1",
         "serviceability": "A", "rank": 1, "structural_blank": False,
         "flows": ["FLOW-03"], "m_cells": ["FLOW-03/STG-04"]},
        {"l3": "B", "role_id": "AGT-002", "role_title": "岗位二", "domain": "D2",
         "serviceability": "A", "rank": 2, "structural_blank": False,
         "flows": ["FLOW-01"], "m_cells": ["FLOW-01/STG-04"]},
    ]


def selftest() -> int:
    """**每条判据一份篡改样本**：证明它真的会报警（否则整行删掉自检照样全绿）。"""
    cf, groups, kw, legacy = _mini_inputs()
    ok = True
    cases = 0
    mutations = 0

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal ok, cases
        cases += 1
        flag = "✅" if cond else "❌"
        if not cond:
            ok = False
        print(f"  {flag} {name}" + (f"  ← {detail}" if detail else ""))

    def mut(name: str, fired: bool, detail: str = ""):
        nonlocal mutations
        mutations += 1
        flag = "✅" if fired else "❌"
        if not fired:
            ok = False
        print(f"  {flag} [变异] {name}" + (f"  ← {detail}" if detail else ""))

    print("--- 判据 1：仪器自证（探针必须能看见）---")
    # 词面匹配仪器是瞎的：中文 L3 ∩ 英文域词 = 0
    probe_ctx = set(re.findall(r"[\u4e00-\u9fff]{2,}", "线索评估 零售渠道与B2B拓展 渠道经营"))
    probe_dom = set(re.findall(r"[a-z]{3,}", " ".join(cf.DOMAIN_CONSTRAINT.get("04-供应链", []))))
    check("probe_lexical_mapping_is_blind：中文∩英文词面匹配得 0（故不能自动推映射）",
          len(probe_ctx & probe_dom) == 0,
          f"ctx={len(probe_ctx)} dom={len(probe_dom)} 交集=0")
    # 反面：同一个探针喂**同语言**输入时必须非 0（否则探针坏的是仪器）
    same = set(re.findall(r"[\u4e00-\u9fff]{2,}", "供应链 库存")) & \
        set(re.findall(r"[\u4e00-\u9fff]{2,}", "供应链 仓储"))
    check("probe 的反向控制：同语言输入下探针必须能命中", len(same) > 0,
          f"交集={sorted(same)}")

    print("\n--- 判据 2：J2 覆盖率（构造空输入）---")
    e = judge_j2([], [])
    check("空工单 + 空产物必须判红", bool(e), f"{len(e)} 条错")
    mut("把 J2 的「生成 0 条」分支删掉", bool(judge_j2([], [{"rank": 1}])),
        "空产物 + 非空工单仍判红")
    #: ⚠️ 「不误报」的用例必须用**与基线一致**的规模，否则它测的是基线而不是判据：
    #: 第一版喂 1 条工单/1 条产物，结果 `len(queries) != BASELINE["generated"]` ⇒ 误报。
    _wb = [{"rank": i} for i in range(1, BASELINE["worklist_total"] + 1)]
    _qb = [{"rank": i} for i in range(1, BASELINE["generated"] + 1)]
    check(f"正常输入的 J2 不误报（{len(_qb)}/{len(_wb)} 与基线一致）",
          not judge_j2(_qb, _wb))
    mut("生成数偏离基线（单边判据的反面：覆盖率是正数照样要判）",
        bool(judge_j2(_qb[:10], _wb)), "覆盖率 80% 也得钉住生成数")

    print("\n--- 判据 3：J3 结构空白（构造篡改样本）---")
    wl = _mini_worklist()
    wl[1]["structural_blank"] = True
    qs = [q for q in wl if not q["structural_blank"]]
    readings = {"strict_worklist": 1, "wide_by_role_worklist": 1, "ledger_rows_blank": 1,
                "strict_worklist_items": ["2:AGT-002:B"],
                "wide_by_role_items": ["2:AGT-002:B"], "worklist_total": 2,
                "retrieval_eligible_strict": 1, "retrieval_eligible_wide": 1}
    base = {"strict_worklist": 1, "wide_by_role_worklist": 1, "ledger_rows_blank": 1}
    check("结构空白被排除后 J3 不误报",
          not judge_j3([{"responsibility": "A", "structural_blank": False}], readings, base))
    bad = judge_j3([{"responsibility": "B", "structural_blank": True}], readings, base)
    mut("给结构空白条目生成检索式", bool(bad), f"{len(bad)} 条错")
    mut("基线被改动（3→4）",
        bool(judge_j3([{"responsibility": "A", "structural_blank": False}], readings,
                      {**base, "strict_worklist": 4})), "条数与基线不等必须报")

    print("\n--- 判据 4：J4 可复算 ---")
    stable = [{"rank": 1, "positive": ["a", "b"]}]
    mut("两次运行不一致", bool(judge_j4(lambda: [{"rank": 1, "positive": ["b", "a"]}], stable)),
        "顺序漂移必须报")
    check("稳定的输入不误报", not judge_j4(lambda: [{"rank": 1, "positive": ["a", "b"]}],
                                       stable))

    print("\n--- 判据 5：J5 改了工单必须变（+ 反向控制）---")
    mini = {
        "cf": cf, "harvest_groups": groups, "kw_text": kw, "legacy_titles": legacy,
        "cidx": {"by_key": {}, "duplicates": []},
        "ledger": {"worklist": [], "rows": []},
    }
    real_l3 = dict(L3_DOMAINS)
    L3_DOMAINS.clear()
    L3_DOMAINS.update({"A": ["04-供应链"], "B": ["03-时间序列"], "复购实验": ["01-因果推断"]})
    try:
        wl2 = _mini_worklist()
        got = build_queries(mini, wl2)
        check("合成工单能生成检索式", len(got) == 2, f"{len(got)} 条")
        check("合成检索式的域名被 filter 认识",
              all(d in cf.DOMAIN_DIR for q in got for d in q["query_domains"]))

        def bfn(w):
            return build_queries(mini, w)

        e5, info = judge_j5(bfn, wl2)
        check("J5 在合成输入上不误报", not e5, str(info))
        mut("J5 篡改责任名没变化（把 build 换成常量）",
            bool(judge_j5(lambda w: [{"rank": 1, "role_id": "X", "responsibility": "A",
                                      "positive": ["x"], "positive_sources": {"junction": []},
                                      "structural_blank": False,
                                      "query_domains": ["04-供应链"]}] * 2, wl2)[0]),
            "假仪表必须被抓")
    finally:
        L3_DOMAINS.clear()
        L3_DOMAINS.update(real_l3)

    print("\n--- 判据 6：J6 相似度（单边判据的反面）---")
    good = {"n_pairs": 1, "n_pairs_junction": 1,
            "junction": {"min": .1, "mean": .2, "max": .3},
            "full": {"min": .1, "mean": .2, "max": .3},
            "most_similar_pair": {"a": "1", "b": "2", "junction": .3, "full": .3},
            "most_similar_pair_full": {"a": "1", "b": "2", "junction": .3, "full": .3},
            "shell_count": 0, "shells": [], "distinct_full_vectors": 2,
            "distinct_nonempty_junction_vectors": 2,
            "empty_junction_count": 2, "empty_junction_ranks": [7, 9],
            "empty_junction_items": ["7:X", "9:Y"],
            "same_domain_set_pairs": ["1 ↔ 2"]}
    check("相似度低时不误报", not judge_j6(good, thresholds_from_baseline(BASELINE)))
    mut("全向量相似度超上界", bool(judge_j6({**good, "full": {**good["full"], "max": .95}},
                                          thresholds_from_baseline(BASELINE))), "模板套壳")
    mut("空壳数超上限",
        bool(judge_j6({**good, "shell_count": 1}, thresholds_from_baseline(BASELINE))),
        "只换责任名")
    mut("一对都没有（n_pairs=0）",
        bool(judge_j6({**good, "n_pairs": 0}, thresholds_from_baseline(BASELINE))),
        "「没得比」不是「都不像」")
    mut("上界过低（交叉项失效）",
        bool(judge_j6({**good, "junction": {**good["junction"], "max": .01}},
                      thresholds_from_baseline(BASELINE))), "单边判据的反面")
    mut("空交叉项变多（新出现取不到材的责任）",
        bool(judge_j6({**good, "empty_junction_count": 3},
                      thresholds_from_baseline(BASELINE))), "没料")
    mut("空交叉项变少（取材条件变宽，不得当成改进）",
        bool(judge_j6({**good, "empty_junction_count": 1},
                      thresholds_from_baseline(BASELINE))), "反向也要判")
    mut("同检索域池的条目对超上限",
        bool(judge_j6({**good, "same_domain_set_pairs": ["x"] * 7},
                      thresholds_from_baseline(BASELINE))),
        "决策表把不同责任压进同一批检索域")
    check("同域池恰好在上限内时不误报",
          not judge_j6({**good, "same_domain_set_pairs": ["x"] * 6},
                       thresholds_from_baseline(BASELINE)))

    print("\n--- 判据 7：J7 负向词按域生效（同词异义反例）---")
    probe_qs = [
        {"rank": 1, "responsibility": "A", "query_domains": ["01-因果推断"]},
        {"rank": 2, "responsibility": "B", "query_domains": ["14-用户分析"]},
    ]
    div = negative_divergence(cf, probe_qs)
    check("分歧矩阵能算出来", div["n_negative_terms"] > 0,
          f"{div['n_negative_terms']} 个按域负向词，{div['n_divergent']} 个分歧")
    mut("分歧词为 0 必须报红",
        bool(judge_j7({**div, "n_divergent": 0}, cf, probe_qs, kw)), "空话")
    mut("分歧没被用上必须报红",
        bool(judge_j7({**div, "used_in_pairs": []}, cf, probe_qs, kw)), "装饰性分歧")
    # 同词异义的**行为**证据：同一个词在两域判定相反
    t = "cohort study"
    st = {d: (t in cf.DOMAIN_NEGATIVE.get(d, [])) for d in ("14-用户分析", "01-因果推断")}
    check("同词异义：`cohort study` 在 14 域判负向、在 01 域不判",
          st["14-用户分析"] and not st["01-因果推断"], str(st))
    mut("把 `trial` 塞进 02 域负向表（误杀 RCT 方法论论文）",
        bool(judge_j7(div, _FakeCF(cf, "02-A_B实验", "trial"), probe_qs, kw)),
        "同词异义反例必须报警")
    #: 反向控制：**全局负向词不许当分歧反例**（实测踩过：`clinical`/`eeg` 三条全红）
    check("仪器自证：`clinical` 在 GLOBAL_NEGATIVE 里（全域生效，不可能是分歧词）",
          "clinical" in cf.GLOBAL_NEGATIVE)
    check("`cohort study` 是**按域**负向词、且不在 GLOBAL 里（合格的分歧反例材料）",
          "cohort study" not in cf.GLOBAL_NEGATIVE
          and "cohort study" in cf.DOMAIN_NEGATIVE.get("14-用户分析", []))
    mut("把 `clinical` 放进 SYNONYMY_PAIRS（全局词冒充分歧反例）",
        bool(judge_j7(div, _FakeCF(cf, "02-A_B实验", "clinical"), probe_qs, kw,
                      pairs=(("clinical", "14-用户分析", "01-因果推断"),))),
        "仪器用错必须报警")

    print("\n--- 判据 8：J8 可被 filter 消费 ---")
    mini_q = [{
        "responsibility": "A", "query_domains": ["04-供应链"], "negative": [],
        "constraint": [], "positive_sources": {"junction": []},
        "positive": ["inventory"],
    }]
    e8 = judge_j8(mini_q, cf, {}, kw)
    mut("负向词缺（与 filter 漂移）", bool(e8), f"{len(e8)} 条错")
    e8b = judge_j8([{**mini_q[0], "query_domains": ["99-不存在"]}], cf, {}, kw)
    mut("域名不在 DOMAIN_DIR 里", bool(e8b), "filter 会静默跳过")
    mut("兜底约束词抄错", bool(judge_j8(
        [{**mini_q[0],
          "constraint": list(cf.DOMAIN_CONSTRAINT["04-供应链"])}], cf, {}, "")),
        "词库回查失败")

    print("\n--- 判据 9：J9 契约挂载点 ---")
    contracts = {"CTR-A-001-x.md": {"contract_id": "CTR-A-001", "responsibility": "A",
                                    "role_id": "AGT-001"}}
    cidx_good = {"by_key": {("AGT-001", "A"): contracts["CTR-A-001-x.md"]}, "duplicates": []}
    q_ok = [{"responsibility": "A", "l3": "A", "rank": 1, "contract_id": "CTR-A-001",
             "serviceability": "A", "role_id": "AGT-001"}]
    check("挂上契约时不误报", not judge_j9(q_ok, contracts, cidx_good))
    mut("契约 id 形状不对",
        bool(judge_j9([{**q_ok[0], "contract_id": "A-1"}], contracts, cidx_good)), "自造键名")
    mut("一份都没挂上",
        bool(judge_j9([{**q_ok[0], "contract_id": None}], contracts, cidx_good)),
        "没有挂载点")
    #: ⚠️ **只掉一条**也是掉。构造两条**契约池里都有**的条目，只给其中一条挂上 ——
    #: 第一版构造的第二条责任名不在契约池里（`B` 没有契约）⇒ 「应有 = 总 − 无名」自洽，
    #: 于是这条变异**没施上力**（又是「变异不生效」而不是「判据无效」）。
    cidx_two = {"by_key": {("AGT-001", "A"): contracts["CTR-A-001-x.md"],
                           ("AGT-002", "B"): contracts["CTR-A-001-x.md"]},
                "duplicates": []}
    q_two = q_ok + [{"responsibility": "B", "l3": "B", "rank": 2,
                     "contract_id": None, "serviceability": "A", "role_id": "AGT-002"}]
    mut("两条都有契约、只挂上 1 条（只掉一条也算掉）",
        bool(judge_j9(q_two, contracts, cidx_two)), "不许全有全无")
    mut("契约池里有对应契约却没挂上（索引键写错）",
        bool(judge_j9([{**q_ok[0], "contract_id": None}], contracts, cidx_good)),
        "索引键")
    mut("契约文件一份都没读到", bool(judge_j9(q_ok, {}, cidx_good)), "「没读到」≠「无需挂载」")

    print("\n--- 判据 10：J10 排序权重不进判定 ---")
    src = Path(__file__).read_text(encoding="utf-8")
    mut("往判据函数体注入 zero_gap",
        bool(judge_j10([], {"worklist": []},
                       src.replace("def judge_j2(queries",
                                   "def judge_j2(queries").replace(
                           "    errs = []\n    if not worklist:",
                           "    errs = []\n    _w = zero_gap\n    if not worklist:", 1))),
        "AST 扫描必须抓到")
    _hits, _n = _scan_weight_refs(src)
    check(f"本文件当前判据函数体干净（扫描 {_n} 个函数，0 命中）", not _hits, str(_hits))
    _seg = 'out[target]["in_first_slice"] = not out[target].get("in_first_slice")'
    check("反向控制：`in_first_slice`（上游输入字段名）**不得**被当成权重 needle",
          not _needle_re("first_slice").search(_seg),
          "裸子串会假红，词边界不会")
    check("仪器自证：同一个 needle 在裸用法下**必须**命中（否则探针是瞎的）",
          bool(_needle_re("first_slice").search('_w = first_slice')),
          "_w = first_slice 必须命中")

    print("\n--- 判据 11：J1 输入没拿到 ≠ 通过 ---")
    try:
        _load_module(Path("/nonexistent/candidate_filter.py"), "_x")
        check("缺输入必须抛 InputMissing", False, "没抛异常")
    except InputMissing:
        check("缺输入抛 InputMissing（main 转 exit 2）", True)
    except Exception as e:  # noqa: BLE001
        check("缺输入必须抛 InputMissing", False, f"抛了 {type(e).__name__}")

    print(f"\n{'✅ 自检通过' if ok else '❌ 自检失败'}：{cases} 用例 · {mutations} 变异样本")
    return EXIT_OK if ok else EXIT_RED


class _FakeCF:
    """只用来做一件坏事：把某词塞进某域的负向表，看 J7 会不会报警。"""

    def __init__(self, cf, domain: str, term: str):
        self.DOMAIN_DIR = dict(cf.DOMAIN_DIR)
        self.GLOBAL_NEGATIVE = list(cf.GLOBAL_NEGATIVE)
        self.DOMAIN_NEGATIVE = {k: list(v) for k, v in cf.DOMAIN_NEGATIVE.items()}
        self.DOMAIN_NEGATIVE[domain] = list(self.DOMAIN_NEGATIVE.get(domain, [])) + [term]
        self.DOMAIN_CONSTRAINT = {k: list(v) for k, v in cf.DOMAIN_CONSTRAINT.items()}
        self.STRICT_ECOMMERCE_DOMAINS = set(cf.STRICT_ECOMMERCE_DOMAINS)


def mutation_summary(inputs: dict, worklist: list[dict]) -> dict:
    """把 J5 的变异结果带出去（报告里要报 N/N）。"""
    try:
        e, info = judge_j5(lambda wl: build_queries(inputs, wl), worklist)
        return {"J5": info["perturbed_changed"], "J5_errors": len(e)}
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="PHASE6 S4：缺口驱动的三段式检索式生成")
    ap.add_argument("--check", action="store_true", help="比对现状（默认动作）")
    ap.add_argument("--selftest", action="store_true", help="自检：每条判据配篡改样本（单元级）")
    ap.add_argument("--mutate", action="store_true",
                    help="端到端变异：在真实数据上逐条弄坏判据，证明退出码会变红")
    ap.add_argument("--json-out", default=str(OUT_JSON))
    ap.add_argument("--md-out", default=str(OUT_MD))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.mutate:
        return run_mutations()

    try:
        code, _ = run(as_json=not args.check or True, json_out=Path(args.json_out),
                      md_out=Path(args.md_out), quiet=args.quiet)
        return code
    except InputMissing as e:
        print(f"❌ 输入没拿到：{e}", file=sys.stderr)
        print("   （这不是「通过」——「没东西可查」与「查过了没问题」必须可区分）",
              file=sys.stderr)
        return EXIT_NOINPUT
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("❌ 门禁内部错误（≠ 判红）", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
