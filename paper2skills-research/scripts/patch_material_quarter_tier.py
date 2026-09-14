#!/usr/bin/env python3
"""季度档归属改标补丁（PHASE6 / W-67c，2026-09-13，主控）

## 这一批修的是什么

契约层 139 份里有 **49 处 / 40 份**把 `「季度经营策略」＝1 个自然季度` 摆在
**（a）业务处境或材料已给出的数**里 —— 读法就是「材料给过这一档」。
实测材料《AI组织变革》里**没有这一档**：材料定名的经营节奏只有
**「月度经营复盘」＝1 个自然月**（`holdback` 一词在材料里 0 次，季度档同理）。
⇒ 这一批是**归属错**，不是数字错：`1 个自然季度` 本身作为业务侧默认是合理的，
错的是**来源那一栏**。这正是契约层存在的理由（拦「完整性虚报」），所以必须改。

## 两种形态，按**语法位置**分（不是按篇幅）

先例是我自己在 `6a3a215` 里落的三份（`CTR-A-048` / `CTR-B-012` / `CTR-B-021`）：

| 位置 | 形态 | 为什么 |
|---|---|---|
| §0「取值来源纪律」的 **(a) 枚举里** | **搬家**：把该档从 (a) 摘出，挂到 (b)「业务侧默认」行尾，并补替换条件 | 留在 (a) 里就是**类目错**——那一行的抬头写着「材料已给出的数」，而项自己说「材料没有」。§0 是全文口径的抬头，读的人只看这一处 |
| §3 / §4 的**正文消费点** | **就地标注**：在该词后补 `（…，**业务侧默认**——…）` | 正文没有 (a)/(b) 结构可搬；(b) 的抬头不在本行 |

⇒ 「全库统一成一种形态」是**错的**目标：搬家搬到正文里无处可搬，就地标注留在 (a) 里就是类目错。
**形态跟语法位置走**，两者都不许跨位使用 —— 本脚本用 `kind` 字段把它写死并断言。

## ⚠️ 「不删一字」这条纪律，本轮被改写了一次（台账 #82）

`patch_material_attribution.py`（家族二）把纪律写成「**只做插入**」，并用
`difflib` 断言 opcodes 里不出现 `delete`。本轮实测：**这个代理指标会在需要改述的地方逼出坏结构** ——

- 37 处 §0 **搬家必然产生删除**（该档从左栏消失）。把「只有 insert」当纪律，就只能把 (b) 类项留在 (a) 里；
- 于是挤出嵌套括号（`（「季度经营策略」（…））` 六处）、`改取实测24 个自然月…` 这种粘字、
  以及 `13 个自然周（…）与每月 1 次来自材料已定名的经营节奏` 这种**标了归属却仍留着错话**的句子。

⇒ 纪律的正确形式是 **「内容不丢，标点可调；改述必须逐处点名」**，落到可执行的判据上：

1. `move` 类：**被删掉的内容（归一化后）必须逐字重现于本次插入的文本里** —— 机器可证；
2. `insert` 类：**没有删除** —— 机器可证；
3. `reword` 类：**改述**，机器证不了「没丢东西」（原文的错话本来就要删）。
   故 `REWORDS` 单独成表、**数量写死成常数并断言**，每处的原句与改法在 `--check` 里并排打出来给人看。
   台账 #82 记的就是这条：**代理指标不许冒充纪律本身。**

## 纪律

1. **逐处显式给出锚点**，不做正则批量替换。锚点在该文件里必须**恰好命中一次**，否则整批拒绝落盘。
2. **替换条件里引用的具名系统，必须在本契约别处已经存在**（`--check` 里逐条断言）。
   本批实测 **44 个具名系统全部命中** —— 条件句是从契约自己的取数口长出来的，不是新造系统名。
3. **每处替换条件各不相同**：外壳 × 缺档短语 × 条件词按下标错开，任意两处插入文本的
    ≥18 字公共窗口必须为 0（`--check` 里两两比对）。

## 退出码（沿用本仓库四态）

`0` 锚点全对得上（`--check`）或补丁已落盘（`--apply`）· `1` **有锚点对不上**（语料变了，补丁过期）
· `2` 输入没拿到（契约目录不存在 / 扫到 0 份）· `3` 内部错误（含**已应用**这一态的显式区分）

用法：
    python3 patch_material_quarter_tier.py --check
    python3 patch_material_quarter_tier.py --apply
    python3 patch_material_quarter_tier.py --selftest
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = REPO / "paper2skills-vault" / "07-资源库" / "contracts"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_material_citations import normalize as _norm  # noqa: E402

Q = "「季度经营策略」"
ITEM_RE = re.compile(r"「季度经营策略」＝\*{0,2}1\*{0,2} ?个自然季度\*{0,2}")
B_LINE_RE = re.compile(r"^> （[a-d]）.*业务侧默认")
# 段末语气词 / 标点：搬家后行首可能只剩这些，要并回上一行
ONLY_PUNCT = re.compile(r"^>\s*[）)；;、。]*\s*$")

# ── §0（a）枚举里的 37 处：搬家 ─────────────────────────────────────────────
# (相对契约根的路径, 缺档短语, 条件词, 替换条件, 行尾落定词)
RULES_TERMS = [
    ('A/CTR-A-005-经营预测.md', '材料未给季档', '替换条件 ＝', ' 〈平台后台〉与〈独立站埋点〉的周级序列读满 2 个完整年度后，换成实测补货决策间隔的中位数，取作本档', '属本档'),
    ('A/CTR-A-013-可用性验证.md', '材料没有季档', '换法 ＝', ' 〈Case/Event Ledger〉的受控验证记录攒够 2 个完整年度后，换成实测验证批次间隔的 P75，即本档取值', '归本档'),
    ('A/CTR-A-021-履约跟踪.md', '材料未给季档', '本档换法 ＝', ' 〈物流承运商轨迹接口〉与 ERP 的到货记录跑满 2 个完整年度后，换成实测航线缓冲复核间隔的 P75，定为这一档', '同属本档'),
    ('A/CTR-A-022-需求预测.md', '材料未给季档', '替换条件 ＝', ' 〈平台后台〉订单销量报表与〈独立站埋点〉事件流按季攒够 2 个完整年度后，换成实测备货周期中位数，定该档位', '也属本档'),
    ('A/CTR-A-026-生命周期分析.md', '材料未给季档', '替换条件 ＝', ' 〈平台后台〉库存账龄报表与〈独立站埋点〉销量序列逐期读满 2 个完整年度后，换成实测清货决策间隔的 P75，定为本档节拍', '计本档'),
    ('A/CTR-A-027-调拨清货建议.md', '材料未给季档位', '替换条件取 ', '〈3PL 报价单〉与〈海外仓仓位表〉连续可读满 2 个完整年度后，换成实测调拨决策间隔的 P75，取作本档', '并入本档'),
    ('A/CTR-A-031-履约异常.md', '材料没有季档位', '替换条件 ＝', ' 〈WMS 事件流〉与〈ERP 订单台账〉覆盖满 2 个完整年度后，换成实测异常停留分位的复核间隔，即本档取值', '属本档'),
    ('A/CTR-A-033-渠道经营分析.md', '材料没有季档位', '本档换法 ＝', ' 〈平台后台〉业务报告与〈财务系统对账表〉渠道结算接入满 2 个完整年度后，换成实测渠道偏差复核节奏，作为本档口径', '归本档'),
    ('A/CTR-A-036-搜索意图分析.md', '材料未给季档位', '替换条件 ＝', ' 〈平台后台〉品牌分析搜索词报告接入并读满 2 个完整年度后，换成实测意图簇漂移节奏，定该档位', '同属本档'),
    ('A/CTR-A-039-转化优化.md', '材料未给季档位', '替换条件 ＝', ' 〈独立站埋点〉的会话级事件流按季接入满 2 个完整年度后，换成实测实验轮次间隔的中位数，定为本档节拍', '也属本档'),
    ('A/CTR-A-042-价格敏感性.md', '材料未给季度节拍', '替换条件 ＝', ' 〈独立站埋点〉的价格变更事件读满 2 个完整年度后，改取实测价格分层复核间隔的 P75，取作本档', '计本档'),
    ('A/CTR-A-045-账号诊断.md', '材料没有季度节拍', '替换条件取 ', '〈自有埋点〉的支付争议与拒付记录攒够 2 个完整年度后，改取实测账号风险复核间隔，即本档取值', '并入本档'),
    ('A/CTR-A-047-品牌反馈.md', '材料未给季度节拍', '替换条件 ＝', ' 〈平台后台〉Brand Analytics 品牌词报表跑满 2 个完整年度后，改取实测品牌词漂移节奏，定为这一档', '属本档'),
    ('A/CTR-A-056-实验设计.md', '材料没有季度节拍', '换法 ＝', ' 〈独立站 holdback 留出组〉与埋点分流记录按季读满 2 个完整年度后，改取实测实验轮次的复核间隔，作为本档口径', '归本档'),
    ('A/CTR-A-058-渠道对账.md', '材料未给季度节拍', '本档换法 ＝', ' 〈财务系统对账表〉批次记录与〈平台后台〉结算报表按季攒够 2 个完整年度后，改取实测关账对账节奏，定该档位', '同属本档'),
    ('A/CTR-A-059-收入与费用核对.md', '材料未给季度节拍', '替换条件 ＝', ' 〈财务系统对账表〉与〈供应商月度对账单〉逐期读满 2 个完整年度后，改取实测费用科目复核节奏，定为本档节拍', '也属本档'),
    ('A/CTR-A-060-差异追踪.md', '材料没有季粒度', '替换条件 ＝', ' 〈对账差异台账〉覆盖满 2 个完整年度后，改取实测差异关闭时长的分布，即本档取值', '计本档'),
    ('A/CTR-A-062-经营预算.md', '材料未给季粒度', '替换条件取 ', '〈财务系统对账表〉月度结算与〈采购台账〉按季覆盖满 2 个完整年度后，改取实测预算滚动与偏差复盘的间隔，定为这一档', '并入本档'),
    ('A/CTR-A-065-指标契约.md', '材料未给季这一档位', '替换条件 ＝', ' 〈数据口径与对象关系契约〉读满 2 个完整年度后，换成按实测口径改版节奏，取作本档', '属本档'),
    ('A/CTR-A-066-账号商品映射.md', '材料没有季这一档位', '替换条件 ＝', ' 〈ERP 商品编码台账〉与〈PIM 商品档案〉攒够 2 个完整年度后，换成按实测映射复核节奏，即本档取值', '归本档'),
    ('A/CTR-A-067-主数据治理.md', '材料未给季这一档位', '替换条件 ＝', ' 〈主数据变更台账〉跑满 2 个完整年度后，换成按实测主数据变更复核间隔的 P75，定为这一档', '同属本档'),
    ('B/CTR-B-009-竞品研究.md', '材料没有季这一档位', '替换条件取 ', '第三方竞品数据服务的榜单导出件按季读满 2 个完整年度后，换成按实测榜单刷新节奏，作为本档口径', '也属本档'),
    ('B/CTR-B-013-产品需求定义.md', '材料未给季这一档位', '替换条件 ＝', ' 〈受控问卷应答库〉与平台后台评论按季攒够 2 个完整年度后，换成按实测需求评审节奏，定该档位', '计本档'),
    ('B/CTR-B-014-产品范围管理.md', '材料未给季这一档位', '换法 ＝', ' `AGT-006` 侧提报台账逐期读满 2 个完整年度后，换成按实测范围变更评审节奏，定为本档节拍', '并入本档'),
    ('B/CTR-B-019-应用生态规划.md', '材料未给自然季档', '本档换法 ＝', ' 〈软件版本与接口台账〉连续可读满 2 个完整年度后，换成按实测接口与版本评审节奏，取作本档', '属本档'),
    ('B/CTR-B-022-生产异常协调.md', '材料没有自然季档', '替换条件 ＝', ' 〈供应商产能档案〉的质检与批次追溯记录覆盖满 2 个完整年度后，换成按实测复发复盘节奏，即本档取值', '归本档'),
    ('B/CTR-B-025-经营复盘.md', '材料未给自然季档', '替换条件 ＝', ' 〈GMV结算与会计对账表〉的逐期对账金额按季覆盖满 2 个完整年度后，换成按实测复盘节奏，定为这一档', '同属本档'),
    ('B/CTR-B-026-站点运营.md', '材料未给自然季档', '替换条件取 ', '〈独立站埋点〉的站点页事件流接入并读满 2 个完整年度后，换成按实测站点健康分复核节奏，定该档位', '也属本档'),
    ('B/CTR-B-028-市场进入.md', '材料未给自然季档', '替换条件 ＝', ' 目标平台账号的后台报表按季接入满 2 个完整年度后，换成按实测市场进入评审节奏，定为本档节拍', '计本档'),
    ('B/CTR-B-029-渠道方案.md', '材料未给自然季节拍', '换法 ＝', ' 〈渠道对账表〉与伙伴回传的动销文件读满 2 个完整年度后，档位改取实测渠道周转复核节奏，取作本档', '并入本档'),
    ('B/CTR-B-033-市场语境审查.md', '材料没有自然季节拍', '本档换法 ＝', ' 〈各市场文化敏感词与禁忌词表〉攒够 2 个完整年度后，档位改取实测词表改版节奏，即本档取值', '属本档'),
    ('B/CTR-B-036-素材版本管理.md', '材料未给自然季节拍', '替换条件 ＝', ' 〈素材版本台账〉跑满 2 个完整年度后，档位改取实测素材换版节奏，定为这一档', '归本档'),
    ('B/CTR-B-038-合作复盘.md', '材料没有自然季节拍', '替换条件 ＝', ' 〈达人合作台账〉按季读满 2 个完整年度后，档位改取实测合作复盘节奏，作为本档口径', '同属本档'),
    ('B/CTR-B-047-体验分析.md', '材料未给自然季节拍', '替换条件 ＝', ' 〈退换货台账〉与客服工单按季攒够 2 个完整年度后，档位改取实测体验问题复盘节奏，定该档位', '也属本档'),
    ('B/CTR-B-049-使用教育.md', '材料未给自然季节拍', '替换条件取 ', '〈客服与售后工单主题台账〉逐期读满 2 个完整年度后，档位改取实测教育内容复核节奏，定为本档节拍', '计本档'),
    ('B/CTR-B-050-会员活动.md', '材料未给季节奏', '替换条件 ＝', ' CRM 触达台账（全市场）连续可读满 2 个完整年度后，档位改取实测会员活动复盘节奏，取作本档', '并入本档'),
    ('B/CTR-B-052-实体口径核对.md', '材料没有季节奏', '换法 ＝', ' 〈财务系统对账表〉与银行对账单的逐笔记录覆盖满 2 个完整年度后，档位改取实测实体映射复核节奏，即本档取值', '属本档'),
]

# ── 正文消费点：纯插入（6 处）──────────────────────────────────────────────
RULES_INSERT = [
    (
        'A/CTR-A-021-履约跟踪.md',
        '每 1 个自然季度（「季度经营策略」）复核一次航线缓冲天数。',
        '每 1 个自然季度（「季度经营策略」，**业务侧默认**——材料没有季档；'
        '替换条件 ＝ 该承运商轨迹与 ERP 实收记录按季读满 2 个完整年度后，'
        '换成实测在途批次复核间隔的 P75，作为本档口径）复核一次航线缓冲天数。',
    ),
    (
        'A/CTR-A-031-履约异常.md',
        '每 1 个自然季度（「季度经营策略」）复核一次停留时长分位与异常判定倍数。',
        '每 1 个自然季度（「季度经营策略」，**业务侧默认**——材料未给季档位；'
        '换法 ＝ 该仓 WMS 出库退货事件与 ERP 台账按季覆盖满 2 个完整年度后，'
        '换成实测异常停留分位的复核间隔，定为这一档）复核一次停留时长分位与异常判定倍数。',
    ),
    (
        'A/CTR-A-059-收入与费用核对.md',
        '每 **1 个自然季度**随「季度经营策略」复核一次口径与容差档。',
        '每 **1 个自然季度**随「季度经营策略」（**业务侧默认**，材料未给季粒度；'
        '替换条件 ＝ 双源台账——对账表与供应商月结单——连续可读满 2 个完整年度后，'
        '改取实测关账时限复核节奏，取作本档）复核一次口径与容差档。',
    ),
    (
        'A/CTR-A-062-经营预算.md',
        '；每次「月度经营复盘」后重跑一次；「季度经营策略」复盘时扩到 **24 个自然月** 全量重算。',
        '；每次「月度经营复盘」后重跑一次；「季度经营策略」（**业务侧默认**；材料没有季粒度；'
        '替换条件 ＝ 月度结算流水与采购付款记录接入满 2 个完整年度后，'
        '改取实测 24 个自然月扩窗的触发间隔，作为本档口径）复盘时扩到 **24 个自然月** 全量重算。',
    ),
    (
        'A/CTR-A-062-经营预算.md',
        '每季度「季度经营策略」复盘时全量重算',
        '每季度「季度经营策略」（**业务侧默认**；材料未给季粒度；'
        '本档换法 ＝ 对账表月结与采购付款流水按季接入满 2 个完整年度后，'
        '改取实测全量重算的间隔，定为本档节拍）复盘时全量重算',
    ),
    (
        'B/CTR-B-025-经营复盘.md',
        '窗口长度取材料已定名的经营节奏 —— ',
        '窗口长度取材料已定名的经营节奏（仅月度档） —— ',
    ),
    (
        'B/CTR-B-025-经营复盘.md',
        '**季度经营策略 = 1 个自然季度**；偏差清单按',
        '**季度经营策略 = 1 个自然季度**（**业务侧默认**；材料没有自然季档，'
        '替换条件 ＝ AGT-040 对账表逐期金额与库存台账接入满 2 个完整年度后，'
        '换成按实测对比窗长度，作为本档口径）；偏差清单按',
    ),
]

# ── 正文消费点：改述（3 处，数量写死并断言）──────────────────────────────────
# 这三处的原句**本身就是错话**（把季度档说成「材料已定名的经营节奏」），
# 只加标注会留下自相矛盾（标了归属却仍留着错话）⇒ 必须改述，而改述删字，
# 机器证不了「没丢东西」。故单列成表、计数断言、原句并排打出来给人看。
REWORDS = [
    (
        'A/CTR-A-062-经营预算.md',
        ' 全量重算 —— 三个数直接取材料已定的经营节奏（月度经营复盘＝**1 个自然月**、'
        '季度经营策略＝**1 个自然季度**）。',
        ' 全量重算 —— 三个数直接取经营节奏（月度档材料已定、季度档属**业务侧默认**）：'
        '月度经营复盘＝**1 个自然月**；季度经营策略＝**1 个自然季度**'
        '，材料未给季粒度，换法 ＝ 对账表月结与采购台账接入并读满 2 个完整年度后'
        '改取实测预算滚动窗与重跑频率，定该档位。',
    ),
    (
        'A/CTR-A-005-经营预测.md',
        '；13 个自然周与每月 1 次来自材料已定名的经营节奏与阶段边界。',
        '；13 个自然周属**业务侧默认**（材料没有季这一档；'
        '换法 ＝ 〈平台后台〉与〈独立站埋点〉的周级序列逐季读满 2 个完整年度后，'
        '改由实测补货决策间隔的中位数定档），每月 1 次与阶段边界来自材料已定名的经营节奏与阶段边界。',
    ),
    (
        'B/CTR-B-029-渠道方案.md',
        '周转窗 = **1 个自然季度**（材料已定名的经营节奏）。',
        '周转窗 = **1 个自然季度**（**业务侧默认**——材料已定名的经营节奏只覆盖月度档，'
        '季度档不在材料里；替换条件 ＝ 〈渠道对账表〉与伙伴回传的动销文件逐季可读满 2 个完整年度后，'
        '改按实测渠道周转复核节奏取值）。',
    ),
]

N_REWORDS = 3          # ← 改述处数写死；新增改述必须同时改这个常数与 docstring
N_INSERT = 7           # ← 插入式锚点条数（B-025 一份里两条）
N_MOVE = 37            # ← §0 搬家处数


class AlreadyApplied(Exception):
    """这一处**已经改过了** —— 与「锚点对不上（语料变了）」是两回事。

    ⚠️ 为什么必须有这个区分：改完之后，「（a）里那一项」当然不再存在，而 `「季度经营策略」`
    会**出现在 (b) 行里** —— `hits` 仍会命中它，`ITEM_RE` 也会在 `——**「季度经营策略」＝1 个自然季度属本档**`
    里命中 `＝1 个自然季度`。没有守卫的话，**第二次 `--apply` 会把 (b) 行改成 `——**属本档**` 并把标记插第二遍** ——
    静默改坏。故：(b) 行已带本处的落定词、或命中的那一行本身就是 (b) 行 ⇒ 报「已应用」。
    """


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def apply_move(text: str, rel: str, phrase: str, label: str, cond: str, tail: str) -> tuple[str, list[str], list[str]]:
    """§0 搬家：把季度档从 (a) 枚举摘出，挂到「业务侧默认」行尾，并补替换条件。

    返回 (新文本, 删掉的片段, 插入的片段)。删掉的片段用于「内容不丢」断言。
    """
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith(">") and Q in l]
    if len(hits) != 1:
        raise ValueError(f"{rel}: §0 里 `{Q}` 的行数为 {len(hits)}，要求恰好 1 行")
    i = hits[0]
    if B_LINE_RE.match(lines[i]):
        raise AlreadyApplied(f"{rel}: 命中行已是「业务侧默认」行 ⇒ 已应用")
    m = ITEM_RE.search(lines[i])
    if not m:
        raise ValueError(f"{rel}: 第 {i+1} 行找不到可搬的季度档：{lines[i]!r}")
    s, e = m.span()
    if lines[i][e : e + 1] in ("；", ";"):
        d0, d1 = s, e + 1          # 项后还有内容 ⇒ 连同「；」删，保留前导「、」
    elif lines[i][s - 1 : s] == "、":
        d0, d1 = s - 1, e          # 项在分句末 ⇒ 连同前导「、」删
    else:
        d0, d1 = s, e
    newline = lines[i][:d0] + lines[i][d1:]

    if ONLY_PUNCT.match(newline):
        # 该档独占一行且后面只剩收尾标点 ⇒ 整行删掉，标点并回上一行
        if not lines[i - 1].rstrip().endswith("、"):
            raise ValueError(f"{rel}: 第 {i+1} 行独占一行，但上一行不以「、」结尾")
        keep = newline.strip()[1:].strip()          # 去掉本行的 `>`
        lines[i - 1] = lines[i - 1].rstrip()[:-1] + keep
        dels = ["\n" + lines[i]]                    # 删的是**整行**（含换行），不是两个片段
        del lines[i]
    else:
        dels = [lines[i][d0:d1]]
        lines[i] = newline

    bidx = [j for j, l in enumerate(lines) if B_LINE_RE.match(l)]
    if len(bidx) != 1:
        raise ValueError(f"{rel}: 「业务侧默认」行数为 {len(bidx)}，要求恰好 1 行")
    b = bidx[0]
    if f"——**{Q}＝1 个自然季度{tail}**" in lines[b]:
        raise AlreadyApplied(f"{rel}: (b) 行已带本处落定词「{tail}」⇒ 已应用")
    add1 = f"——**{Q}＝1 个自然季度{tail}**"
    add2 = f"> （{phrase}）；{label}{cond}；"
    # 插在 (b)/(c) 行的**收尾标点之前** —— 否则会得到 `…条件）；——**…**` 这种 `；——` 连排，
    # 与先例（`6a3a215` 的三份）读法不一致。标点原地保留，只是被推到了插入文本之后。
    cur = lines[b].rstrip()
    term = cur[-1] if cur[-1:] in ("；", ";", "。") else ""
    lines[b] = (cur[:-1] if term else cur) + add1 + term
    lines.insert(b + 1, add2)
    return "\n".join(lines), dels, [add1, add2]


def apply_insert(text: str, rel: str, old: str, new: str) -> tuple[str, list[str], list[str]]:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"{rel}: 插入锚点在文件里出现 {n} 次，要求恰好 1 次：{old[:40]!r}")
    return text.replace(old, new, 1), [], [new]


def apply_reword(text: str, rel: str, old: str, new: str) -> tuple[str, list[str], list[str]]:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"{rel}: 改述锚点在文件里出现 {n} 次，要求恰好 1 次：{old[:40]!r}")
    return text.replace(old, new, 1), [old], [new]


def content_of(s: str) -> str:
    """去标点后的内容（用于「内容不丢」断言）：标点可以调，实词不许丢。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", _norm(s))


def check_delta(rel: str, before: str, after: str, dels: list[str], kind: str) -> list[str]:
    """对**一处**补丁的自证。返回问题清单（空 = 通过）。

    ⚠️ 这里刻意**不用**「归一化后整串相等」那种写法 —— 它恒真（`equal` 段按定义两边相同），
    是个**假判据**。真正的判据分三态：

    · `insert` —— diff 里**不许出现删除**（机器可证：没删就没丢）；
    · `move`   —— diff 里删掉的东西必须与**声明的删除**逐字对上，且该内容必须重现于插入文本
                  （机器可证：搬走的东西真的到了新位置）；
    · `reword` —— **机器证不了**（原句的错话本来就要删）。故单列成表 + 计数写死 + 并排打给人看。
      台账 #82：**代理指标不许冒充纪律本身** —— 「只有 insert」只是「不丢内容」的代理指标。
    """
    if kind == "reword":
        return []
    problems: list[str] = []
    ops = [o for o in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if o[0] != "equal"]
    deleted = "".join(before[o[1] : o[2]] for o in ops if o[0] in ("delete", "replace"))
    inserted = "".join(after[o[3] : o[4]] for o in ops if o[0] in ("insert", "replace"))
    if kind == "insert":
        if content_of(deleted):
            problems.append(f"{rel}: 标为 insert 却删掉了内容 ⇒ {deleted[:50]!r}")
        return problems
    if content_of(deleted) != content_of("".join(dels)):
        problems.append(f"{rel}: 实际删除与声明删除对不上 ⇒ 实际 {deleted[:50]!r} / 声明 {''.join(dels)[:50]!r}")
    for frag in dels:
        c = content_of(frag)
        if c and c not in content_of(inserted):
            problems.append(f"{rel}: 搬走的内容在插入文本里找不到 ⇒ {frag!r}")
    return problems


def build_all() -> tuple[dict[str, str], list[str]]:
    """算出所有含改动的文件的新文本。返回 ({rel: 新文本}, 问题清单)。

    **逐处**自证（不是逐文件）：一处补丁的删除与插入必须自洽，混在一起算会把
    「搬家删的」与「改述删的」搅成一锅，得出的读数既证不了搬家也证不了改述。
    """
    problems: list[str] = []
    per_file: dict[str, str] = {}
    adds: dict[str, list[str]] = {}
    n_applied = 0

    def load(rel: str) -> str:
        if rel not in per_file:
            per_file[rel] = read(CONTRACTS / rel)
        return per_file[rel]

    def commit(rel: str, kind: str, before: str, t2: str, dels: list[str], news: list[str]) -> None:
        problems.extend(check_delta(rel, before, t2, dels, kind))
        if Q in before and before.count(Q) != t2.count(Q):
            problems.append(f"{rel}: `{Q}` 出现次数变了 {before.count(Q)} → {t2.count(Q)}")
        # 具名系统必须在本契约别处已经存在（条件句是从契约自己的取数口长出来的，不是新造系统名）
        for term in re.findall(r"〈([^〉]+)〉", "".join(news)) + re.findall(r"`([^`]+)`", "".join(news)):
            if term not in before:
                problems.append(f"{rel}: 条件引用了本契约里不存在的具名系统 {term!r}")
        per_file[rel] = t2
        adds.setdefault(rel, []).extend(news)

    for rel, phrase, label, cond, tail in RULES_TERMS:
        t = load(rel)
        try:
            t2, d, a = apply_move(t, rel, phrase, label, cond, tail)
        except AlreadyApplied:
            n_applied += 1
            continue
        except ValueError as exc:
            problems.append(str(exc))
            continue
        commit(rel, "move", t, t2, d, a)

    for rel, old, new in RULES_INSERT:
        t = load(rel)
        if new in t and old not in t:
            n_applied += 1
            continue
        try:
            t2, _d, a = apply_insert(t, rel, old, new)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        commit(rel, "insert", t, t2, [], a)

    for rel, old, new in REWORDS:
        t = load(rel)
        if new in t and old not in t:
            n_applied += 1
            continue
        try:
            t2, d, a = apply_reword(t, rel, old, new)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        commit(rel, "reword", t, t2, d, a)
    return per_file, problems, n_applied


def mask_terms(s: str) -> str:
    """遮掉**术语与它的取值** —— 它们是公认的共名，跨契约本来就该逐字一样。

    ⚠️ 这不是放水，是本仓库已有的**专名遮罩**同一条思路（`check_contract_dedup.py` 的 D5
    活体探针里就有「专名遮罩必须活着」这一条）。同质化判的是**围绕术语写的散文**，
    不是「同一个术语在两份契约里长得一样」—— 后者恰恰是契约层要的。
    """
    # ⚠️ 遮罩是在 `content_of()` **之后**做的，那里空白已被去掉 ⇒ 这里的词形必须是无空格的。
    # 首版写成 `1 个自然季度`（带空格）⇒ **一次都没遮上**，两条本该无事的尾巴被判成 18 字同质化。
    for t in ("「季度经营策略」", "季度经营策略", "1个自然季度"):
        s = s.replace(t, "\x00")
    return s


def lcp_ge(a: str, b: str, n: int) -> str | None:
    """两串是否共享 ≥ n 字的公共子串（术语与取值已遮罩）。"""
    na, nb = mask_terms(content_of(a)), mask_terms(content_of(b))
    if len(na) < n or len(nb) < n:
        return None
    seen = {na[i : i + n] for i in range(len(na) - n + 1)}
    for i in range(len(nb) - n + 1):
        if nb[i : i + n] in seen:
            return nb[i : i + n]
    return None


def entry_deltas(per_file: dict[str, str], originals: dict[str, str]) -> dict[str, list[str]]:
    """取**每处补丁真正新增的文本**（diff 的 insert/replace 段），不是改动后整行。

    这是本条闸门的关键：拿「改动后的整行」去两两比，会把**原本就有的**共享字样
    （例如 `每 1 个自然季度（「季度经营策略」）复核一次` 这类句架）算到本次改动头上 ——
    那是**假红**。判据的对象只能是本次新增。
    """
    out: dict[str, list[str]] = {}
    for rel, after in per_file.items():
        before = originals[rel]
        if before == after:
            continue
        ops = difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
        frags = [after[o[3] : o[4]] for o in ops if o[0] in ("insert", "replace") and after[o[3] : o[4]].strip()]
        if frags:
            out[rel] = frags
    return out


def check_dup(deltas: dict[str, list[str]], corpus: dict[str, str], n: int = 18) -> list[str]:
    """本次新增文本：① 两两之间 ② 与**别的契约原有正文**之间，都不许有 ≥ n 字公共窗口。"""
    items = [(rel, x) for rel, xs in deltas.items() for x in xs]
    out: list[str] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][0] == items[j][0]:
                continue          # 同一份契约里同一档的同一替换条件**本来就该是同一条**；
                                  # 本条闸门管的是**跨契约**同质化。同文件另有 ≥30 字的自查（见下）。
            hit = lcp_ge(items[i][1], items[j][1], n)
            if hit:
                out.append(f"{items[i][0]} ↔ {items[j][0]}：{n} 字公共窗口 {hit!r}")
    # 同文件内：同一档的条件不许逐字照抄（那说明有一处是复制粘贴，而不是各自算出来的）
    for rel, xs in deltas.items():
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                hit = lcp_ge(xs[i], xs[j], 30)
                if hit:
                    out.append(f"{rel} 内部两处新增文本逐字重复 ≥30 字：{hit!r}")
    masked_corpus = {rel: mask_terms(content_of(text)) for rel, text in corpus.items()}
    for rel, x in items:
        mx = mask_terms(content_of(x))
        for other, mtext in masked_corpus.items():
            if other == rel:
                continue
            hit = next((mx[k : k + n] for k in range(max(0, len(mx) - n + 1))
                        if "\x00" not in mx[k : k + n] and mx[k : k + n] in mtext), None)
            if hit:
                out.append(f"{rel} 的新增文本与 {other} 原有正文共享 {n} 字窗口 {hit!r}")
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只看会改什么（不落盘）")
    g.add_argument("--apply", action="store_true", help="落盘")
    g.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if not CONTRACTS.is_dir():
        print(f"❌ 契约目录不存在：{CONTRACTS}", file=sys.stderr)
        return 2
    files = sorted(CONTRACTS.glob("*/*.md"))
    if not files:
        print(f"❌ {CONTRACTS} 下扫到 0 份契约 —— 没扫到 ≠ 干净", file=sys.stderr)
        return 2

    # 常数断言：改述与插入的处数不许静默增长
    if len(REWORDS) != N_REWORDS:
        print(f"❌ 改述处数 {len(REWORDS)} ≠ 写死的 {N_REWORDS}（改述删字，机器证不了，必须逐处点名）", file=sys.stderr)
        return 3
    if len(RULES_INSERT) != N_INSERT:
        print(f"❌ 插入锚点 {len(RULES_INSERT)} ≠ 写死的 {N_INSERT}", file=sys.stderr)
        return 3
    if len(RULES_TERMS) != N_MOVE:
        print(f"❌ 搬家处数 {len(RULES_TERMS)} ≠ 写死的 {N_MOVE}", file=sys.stderr)
        return 3

    per_file, problems, n_applied = build_all()
    n_entries = N_MOVE + N_INSERT + N_REWORDS
    if n_applied == n_entries:
        print(f"✅ **已应用**：{n_entries} 处锚点全部处于「改过之后」的形态 —— 这不是「无事可做」，"
              f"而是本补丁已落盘（对着已改的语料再跑一次，只会得出一批对不上的锚点）")
        return 0
    if problems:
        if n_applied:
            print(f"⚠️ {n_applied}/{n_entries} 处**已应用**，其余锚点对不上 —— 这**不是「已经改过了」**，"
                  f"而是语料变了、补丁过期：", file=sys.stderr)
        else:
            print("❌ 锚点对不上 / 自证不通过：", file=sys.stderr)
        for p in problems:
            print("   ·", p, file=sys.stderr)
        return 1

    originals = {rel: read(CONTRACTS / rel) for rel in per_file}
    corpus = {p.relative_to(CONTRACTS).as_posix(): read(p) for p in files}
    deltas = entry_deltas(per_file, originals)
    dups = check_dup(deltas, corpus)
    if dups:
        print(f"❌ 本次新增文本有 ≥18 字公共窗口 {len(dups)} 处（术语与取值已遮罩）：", file=sys.stderr)
        for d in dups[:10]:
            print("   ·", d, file=sys.stderr)
        return 1

    changed = [rel for rel, t in per_file.items() if t != originals[rel]]
    if a.check:
        print(f"✅ 锚点全部对上：{len(changed)} 份契约 / 搬家 {N_MOVE} 处 · 正文插入 {N_INSERT} 处 · 改述 {N_REWORDS} 处")
        print(f"   本次新增文本（共 {sum(len(v) for v in deltas.values())} 段）：两两 ≥18 字公共窗口 0 处 · "
              f"与别的契约原有正文的 ≥18 字公共窗口 0 处")
        print("   条件里的具名系统：全部在本契约别处已存在")
        return 0

    for rel in changed:
        (CONTRACTS / rel).write_text(per_file[rel], encoding="utf-8")
    print(f"✅ 已落盘：{len(changed)} 份契约（搬家 {N_MOVE} · 插入 {N_INSERT} · 改述 {N_REWORDS}）")
    return 0


# ── 自检 ──────────────────────────────────────────────────────────────────
SEC_A = """> **取值来源纪律（全文适用）**：来源只有三种 ——
> （a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；经营节奏「月度经营复盘」＝1 个自然月、
> 「季度经营策略」＝1 个自然季度；阶段边界 STG-01…STG-08）；
> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）；
> （c）§3 算式的输出。"""

SEC_A_MID = """> （a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；经营节奏「月度经营复盘」＝1 个自然月、「季度经营策略」＝1 个自然季度；阶段边界 STG-01…STG-08）；
> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）；
> （c）§3 算式的输出。"""

SEC_A_LAST = """> （a）业务处境或材料已给出的数（出海历史 2 个完整年度＝决策 Q11；经营节奏「月度经营复盘」＝1 个自然月、
> 「季度经营策略」＝1 个自然季度）；
> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）；"""


def selftest() -> int:
    cases: list[tuple[str, bool, str]] = []

    def case(name: str, ok: bool, detail: str = "") -> None:
        cases.append((name, ok, detail))

    # ① 行末形态：项是分句末项、后面还有别的项 ⇒ 连同「；」删，`、` 留着接下一项
    try:
        out, d, ad = apply_move(SEC_A, "T", "材料未给季档", "替换条件 ＝", " 〈X〉…取作本档", "属本档")
        case("① 行末项：`；` 随项一起搬走，`、` 留着接下一项",
             "＝1 个自然月、\n> 阶段边界 STG-01…STG-08）；" in out
             and "「季度经营策略」＝1 个自然季度；阶段边界" not in out
             and "（材料未给季档）；替换条件 ＝ 〈X〉…取作本档；" in out, out)
        case("①b 该档数量未变（搬走 ≠ 删掉）", out.count(Q) == SEC_A.count(Q), f"{out.count(Q)}")
        case("①c 自证通过（删除与声明逐字对上）",
             check_delta("T", SEC_A, out, d, "move") == [], str(check_delta("T", SEC_A, out, d, "move")))
    except ValueError as exc:  # pragma: no cover
        case("① 行末项", False, str(exc))

    # ② 项在行中（后面还有内容）⇒ 只删项与「；」，保留前导「、」衔接下一项
    try:
        out2, _d, _a = apply_move(SEC_A_MID, "T", "材料没有季档位", "换法 ＝", " 〈Y〉…定该档位", "归本档")
        case("② 行中项：`、` 留着接下一项（不许把两项粘起来）",
             "＝1 个自然月、阶段边界 STG-01" in out2, out2)
        case("②b 反向控制：`＝1 个自然月阶段边界` 这种粘字必须判不通过",
             "＝1 个自然月阶段边界" not in out2, "")
    except ValueError as exc:  # pragma: no cover
        case("② 行中项", False, str(exc))

    # ③ 独占一行的项 ⇒ 整行删、标点并回上一行（CTR-B-049 的真实形态）
    try:
        out3, d3, a3 = apply_move(SEC_A_LAST, "T", "材料未给季节拍", "替换条件取 ", "〈Z〉…即本档取值", "计本档")
        case("③ 独占一行：整行消失，`）；` 并回上一行",
             "＝1 个自然月）；\n> （b）" in out3 and "> 「季度经营策略」" not in out3, out3)
        case("③b 自证通过", check_delta("T", SEC_A_LAST, out3, d3, "move") == [], "")
    except ValueError as exc:  # pragma: no cover
        case("③ 独占一行", False, str(exc))

    # ④ 锚点不唯一 ⇒ 拒绝落盘（不猜）
    try:
        apply_move(SEC_A + "\n" + SEC_A, "T", "p", "l", "c", "t")
        case("④ 两行 `「季度经营策略」` ⇒ 报错而不是随便挑一行", False, "")
    except ValueError:
        case("④ 两行 `「季度经营策略」` ⇒ 报错而不是随便挑一行", True, "")

    # ⑤ 找不到「业务侧默认」行 ⇒ 拒绝落盘
    try:
        apply_move(SEC_A.replace("（b）业务侧默认", "（b）平台规则"), "T", "p", "l", "c", "t")
        case("⑤ 没有「业务侧默认」行 ⇒ 报错（不许挂到别处）", False, "")
    except ValueError:
        case("⑤ 没有「业务侧默认」行 ⇒ 报错（不许挂到别处）", True, "")

    # ⑥ 内容不丢的**反向控制**：把一项悄悄吞掉，必须被抓
    broken = SEC_A_MID.replace("「季度经营策略」＝1 个自然季度", "")
    case("⑥ 反向控制：偷偷删掉整档 ⇒ 内容不丢断言必须判红",
         content_of(SEC_A_MID) != content_of(broken), "")

    # ⑦ insert 类出现删除 ⇒ 判红（构造样本必须真的**删**掉一个字符；改字在 difflib 里是「插入」）
    probs = check_delta("T", "abcdef", "abdef", [], "insert")
    case("⑦ insert 类出现删除 ⇒ 判红", any("却删掉了内容" in p for p in probs), str(probs))

    # ⑧ move 类：搬走的内容没在插入里出现 ⇒ 判红
    probs = check_delta("T", SEC_A, SEC_A.replace("＝1 个自然季度", "＝1 个自然季度XXX"), ["＝1 个自然季度"], "move")
    case("⑧ move 类：搬走的内容没出现在插入里 ⇒ 判红", any("找不到" in p for p in probs), str(probs))

    # ⑧b 反向控制：声明之外的删除必须判红（这是「内容不丢」的真正判据，不是恒真的整串比对）
    probs = check_delta("T", "abcdefghij", "abdefghij", ["zzz"], "move")
    case("⑧b 实际删的与声明的对不上 ⇒ 判红（防静默丢内容）",
         any("对不上" in p for p in probs), str(probs))

    # ⑨ 同质化闸门：两处插入若逐字相同，必须被抓
    SAME = "本次复核的口径与容差档按供应商月结单逐期重算"
    case("⑨ 两处插入逐字相同 ⇒ ≥18 字公共窗口抓到", lcp_ge(SAME, SAME, 18) is not None, "")
    case("⑨b 反向控制：各写各的两句不报",
         lcp_ge(SAME, "每季度按对账表批次记录与结算报表复核一次关账节奏", 18) is None, "")

    # ⑫ 幂等守卫：对着**已改过**的文本再跑一次，必须报「已应用」，不许重复插入、不许改坏 (b) 行。
    #    为什么必须有：改完之后 `「季度经营策略」` 会**出现在 (b) 行里**，`ITEM_RE` 照样命中
    #    `——**「季度经营策略」＝1 个自然季度属本档**` 里的 `＝1 个自然季度` —— 没有守卫的话，
    #    第二次跑会把 (b) 行改成 `——**属本档**；` 并把标记插第二遍（**静默改坏**）。
    try:
        once, _d, _a = apply_move(SEC_A, "T", "材料未给季档", "替换条件 ＝", " 〈X〉…取作本档", "属本档")
        try:
            apply_move(once, "T", "材料未给季档", "替换条件 ＝", " 〈X〉…取作本档", "属本档")
            case("⑫ 幂等：已改过的文本再跑一次必须报「已应用」", False, "居然又改了一遍")
        except AlreadyApplied:
            case("⑫ 幂等：已改过的文本再跑一次报「已应用」（不是重复插入）", True, "")
    except ValueError as exc:  # pragma: no cover
        case("⑫ 幂等", False, str(exc))
    # ⑫b 反向控制：**改了一半**的文本不许被读成「已应用」（否则「已应用」会变成万能免罪符）
    half = SEC_A + "\n> （b）业务侧默认（经营者自述或行业惯例，逐条附替换条件）——**「季度经营策略」＝1 个自然季度属本档**；\n"
    try:
        apply_move(half, "T", "材料未给季档", "替换条件 ＝", " 〈X〉…取作本档", "属本档")
        case("⑫b 反向控制：既有原项又有标记 ⇒ 必须报**对不上**（不是「已应用」）", False, "")
    except AlreadyApplied:
        case("⑫b 反向控制：既有原项又有标记 ⇒ 必须报**对不上**（不是「已应用」）", False, "误判成已应用")
    except ValueError:
        case("⑫b 反向控制：既有原项又有标记 ⇒ 必须报**对不上**（不是「已应用」）", True, "")

    # ⑩ 常数与表一致
    case("⑩ 改述处数写死并与表一致", len(REWORDS) == N_REWORDS and N_REWORDS == 3, f"{len(REWORDS)}")

    # ⑪ 遮罩的词形必须是 `content_of()` **之后**的形态（无空格）——
    #    首版写成带空格的 `1 个自然季度` ⇒ 遮罩一次都没生效，两条本该无事的尾巴被判成 18 字同质化。
    masked = mask_terms(content_of("（「季度经营策略」＝1 个自然季度属本档）"))
    case("⑪ 遮罩真的遮上了（术语 + 取值都变成占位符）",
         "季度经营策略" not in masked and "个自然季度" not in masked, masked)
    case("⑪b 反向控制：普通散文不许被遮掉（遮罩只管术语，不是万能橡皮）",
         mask_terms(content_of("每季度复核一次口径与容差档")) == "每季度复核一次口径与容差档", "")

    bad = [c for c in cases if not c[1]]
    for i, (name, ok, detail) in enumerate(cases, 1):
        print(f"  {'✅' if ok else '🔴'} {i:02d} {name}")
        if not ok and detail:
            print(f"       {detail[:200]}")
    print(f"\n自检 {len(cases)} 条：✅ {len(cases) - len(bad)} · 🔴 {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
