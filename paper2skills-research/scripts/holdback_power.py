#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdback_power.py —— PHASE6 S9「独立站 holdback」样本量/功效计算器 + 判据门禁

它回答什么
----------
决策 Q9/Q10 已定：增量测量以**自有独立站**为主，独立站**没有 holdback 流量、可以建**。
本脚本把「可以建」推进到「可判定」：对 FLOW-01 的 **6 条 A 类责任**
（增量分析 · 广告实验 · 预算分配 · 投放诊断 · 实验设计 · 内容实验）逐条算出

    按该条契约自带的 α / 功效 / MDE，在 90 个自然日内达成功效所需的**日均分流单元下限 N***，
    以及**反向口径**：给定真实流量时能检出的最小相对效应 r̂。

达不到 ⇒ 明确输出「**该条用准实验**」，**不硬凑数字**。

⚠️ 本脚本不产生任何流量数字。独立站流量在本仓库**没有事实源**（见 --traffic 的纪律）。
   没有带来源的实测流量时，退出码是 **2（输入没拿到 ≠ 通过）**，不是 0。

判据（每条都能单独失败，编号 J1…J11）
------------------------------------
  J1  分流参数自洽：各层占比 ∈ (0,1)、池内比例和为 1、对照臂占比偏离契约值时必须写明理由
  J2  幂等键完备：每层须有 unit 字段 + salt + 确定性哈希 + 模数；缺一即红
  J3  **层匹配（方法学判据）**：责任的推断层级必须等于其被分配分流层的单元层级。
      会话级随机化承载用户级效应 ⇒ 红（这是方法学错误，不是参数选择）
  J4  **契约事实源活体**：每次运行现读 6 份契约文件，抽 frontmatter 的 `responsibility`
      与正文的 α / 功效 / MDE 字面，与内置标定逐条断言相等。读不到 ⇒ exit 2；抽不出 ⇒ exit 3
  J5  **正向功效的已知答案**：两组比例 5% vs 5.5%、α=0.05 双侧、power=0.80
      ⇒ 每臂样本量必须落在公认区间 [29000, 34000]（教科书算式）
  J6  正反互逆：n ← MDE 与 MDE ← n 必须互为逆运算（相对误差 < 1e-9）
  J7  **流量输入必须有来源**：--traffic 文件的 `source` 为空 ⇒ exit 2。
      「编一个看起来合理的流量数字然后当实测用」是本仓库最重的一类错，此判据让它跑不进来
  J8  **声明路由 == 计算路由**：逐条比对，不等即判红（两边都能抓：假绿与假红）
  J9  geo 层可检测效应：r_min(m, CV) 与契约迁移阈值（20%）比对；做不到就判准实验
  J10 覆盖：6 条责任一条不少、一条不多，且责任名与契约 frontmatter 相符
  J11 停止条件可判据：每条必须含**数值阈值**（形容词不是判据）；提前终止必须带序贯边界

退出码约定（照本仓库规矩）
--------------------------
  0 = 全过 ／ 1 = 判红 ／ 2 = 输入没拿到（**≠ 通过**）／ 3 = 内部错误（**≠ 判红**）

用法
----
    python3 paper2skills-research/scripts/holdback_power.py --selftest
    python3 paper2skills-research/scripts/holdback_power.py --mutate
    python3 paper2skills-research/scripts/holdback_power.py --assume \\
        --json-out paper2skills-research/data/holdback-design.json
    python3 paper2skills-research/scripts/holdback_power.py --traffic <实测.json> ...
    python3 paper2skills-research/scripts/holdback_power.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

ROUTE_RANDOMIZED = "randomized_holdback"
ROUTE_QUASI = "quasi_experiment"
ROUTE_CORROBORATION = "holdback_corroboration_only"

EXIT_OK = 0
EXIT_RED = 1
EXIT_NOINPUT = 2
EXIT_INTERNAL = 3

REPO = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO / "paper2skills-vault" / "07-资源库" / "contracts" / "A"
DEFAULT_ARTIFACT = REPO / "paper2skills-research" / "data" / "holdback-design.json"

CYCLE_DAYS = 90          # 契约 CTR-A-056 T6 原文：「在 90 个自然日内可检出 T2 的 MDE」
MIN_WINDOW_DAYS = 14     # 契约 CTR-A-057/A-048/A-051/A-056 一致的观测窗下限

# 契约 CTR-A-051 T4 的迁移阈值：候选臂高出主臂 ≥ 20% 才值得挪预算
GEO_ACTIONABLE_REL = 0.20

# ---------------------------------------------------------------- 内置标定表
# 说明：α / 功效 / MDE 三项**不在这里写死**，每次运行从契约文件正文现抽（J4）。
# 本表只写「去哪抽、抽什么形状」以及契约没有数值出口的那几条的结构事实。
CONTRACT_SPECS: list[dict] = [
    {
        "contract_id": "CTR-A-057", "responsibility": "增量分析", "role_id": "AGT-035",
        "metric": "增量转化（处理组实测值 − 对照组实测值）",
        "result_type": "rate", "baseline_key": "site_conversion_rate",
        "inference_level": "user", "result_denominator": "unit",
        "inference_level_note": "契约 CTR-A-057 §2 ②：分流单元＝独立站访客标识",
        "layer_id": "L-USER",
        "mde_pattern": r"相对提升\s*(\d+)\s*%", "mde_expected_pct": 5,
        "alpha_pattern": r"α\s*=\s*(0\.05)",
        "power_pattern": r"1\s*-\s*β\s*=\s*(0\.80)",
        "branches": {"feasible": "randomized_holdback", "infeasible": "quasi_experiment"},
        "note": "对照臂占前台流量 10%（§3 业务侧默认）＋ 层内 1:1（同处）",
    },
    {
        "contract_id": "CTR-A-056", "responsibility": "实验设计", "role_id": "AGT-035",
        "metric": "实验主指标（OEC，由 Case Charter 指定）",
        "result_type": "rate", "baseline_key": "site_conversion_rate",
        "inference_level": "user", "result_denominator": "unit",
        "inference_level_note": "契约 CTR-A-056 §2 ②：分流单元＝独立站访客标识（不能访客级时退到账号级）",
        "layer_id": "L-USER",
        "mde_pattern": r"MDE\s*=\s*相对提升\s*(\d+)\s*%", "mde_expected_pct": 10,
        "alpha_pattern": r"α\s*=\s*(0\.05)",
        "power_pattern": r"1\s*-\s*β\s*=\s*(0\.80)",
        "branches": {"feasible": "randomized_holdback", "infeasible": "quasi_experiment"},
        "note": "T6 留出比例取分流单元 5%（上限 10%），且须满足 90 天内可检出 T2 的 MDE",
    },
    {
        "contract_id": "CTR-A-048", "responsibility": "内容实验", "role_id": "AGT-035",
        "metric": "内容点击率 content_click ÷ content_impression",
        "result_type": "rate_impression", "baseline_key": "content_ctr",
        "inference_level": "user", "result_denominator": "impression",
        "inference_level_note": "契约 CTR-A-048 §2 ② 明写分流单元＝独立站访客标识 ⇒ 效应层级是访客级；"
                                "曝光只是结果的分母，不是随机化单元", "layer_id": "L-USER",
        "cells": 4, "cells_note": "2^k 设计，k 上限 2（§2 ③ 观测不足时的因素数上限）⇒ 4 格",
        "mde_pattern": r"相对提升\s*(\d+)\s*%", "mde_expected_pct": 5,
        "alpha_pattern": r"α\s*=\s*(0\.05)",
        "power_pattern": r"1\s*-\s*β\s*=\s*(0\.80)",
        "branches": {"feasible": "randomized_holdback", "infeasible": "quasi_experiment"},
        "note": "分母是曝光不是访客；替代口径为站内随机化对照（§3 T5）",
    },
    {
        "contract_id": "CTR-A-051", "responsibility": "广告实验", "role_id": "AGT-021",
        "metric": "增量 ROAS = 增量 GMV ÷ 该实验组广告花费",
        "result_type": "cluster_continuous", "baseline_key": None,
        "inference_level": "geo", "layer_id": "L-GEO",
        "min_clusters": 2, "min_exposure_per_arm": 1000,
        "branches": {"feasible": "randomized_holdback", "infeasible": "quasi_experiment"},
        "note": "geo/switchback 要求对照单元在实验期内保持不投放（T5）",
    },
    {
        "contract_id": "CTR-A-049", "responsibility": "投放诊断", "role_id": "AGT-021",
        "metric": "ROAS 与归因份额差（规则口径份额 − 数据驱动份额）",
        "result_type": "structural", "baseline_key": None,
        "inference_level": "channel", "layer_id": "L-CHANNEL",
        "branches": {"feasible": "holdback_corroboration_only", "infeasible": "holdback_corroboration_only"},
        "note": "契约自带准实验窗（暂停/关闭投放前后各 14 个自然日）；holdback 只作佐证读数",
    },
    {
        "contract_id": "CTR-A-050", "responsibility": "预算分配", "role_id": "AGT-021",
        "metric": "各渠道边际回报（每多花一元带来的成交额增量）",
        "result_type": "structural", "baseline_key": None,
        "inference_level": "channel", "layer_id": "L-CHANNEL",
        "branches": {"feasible": "holdback_corroboration_only", "infeasible": "holdback_corroboration_only"},
        "note": "识别靠响应曲线与三类硬约束；holdback 建成后被解释变量由名义成交额换成增量成交额（T6）",
    },
]

# ---------------------------------------------------------------- 分流口径（设计默认值）
DEFAULT_SPLIT = {
    "program": "独立站 holdback 分流程序",
    "version": "v1",
    "layers": [
        {
            "id": "L-USER", "unit_field": "visitor_id", "unit_level": "user",
            "randomizable": True, "nesting_rule": None,
            "label": "独立站访客标识（用户级）",
            "share_of_frontend_traffic": 0.20,
            "pool_allocation": {"treatment": 0.5, "control": 0.5},
            "effective_control_share_of_frontend": 0.10,
            "allocation_rationale": (
                "契约 CTR-A-057 §3 T2 同时写了两条：对照臂占独立站前台流量 10%（业务侧默认）"
                "与层内处理/对照 1:1。同时满足两条的唯一读法是「池 = 前台流量 20%，池内 1:1」；"
                "另一读法（100% 前台直接 90/10）违反层内 1:1。两者在同样 n 下的前台流量代价由本脚本现算并列。"
            ),
            "stratify_keys": ["currency_or_country", "device_type", "new_or_returning"],
            "stratify_source": "契约 CTR-A-057 §3 T2「分层键从〈独立站埋点〉可判定的字段中选」",
            "idempotency": {
                "key_fields": ["visitor_id", "salt"], "salt": "holdback-v1",
                "hash": "sha256", "modulus": 10000, "deterministic": True,
                "recompute_rule": "arm = int(sha256(salt + '|' + visitor_id)[:8], 16) % modulus；同一单元永不重抽",
            },
            "attribution_compat": {
                "unit_grain": "与 AGT-045 指标契约的「分流单元」同粒度",
                "ledger": "Case/Event Ledger 的 Execution Attempt 回执逐条落分流臂",
                "quality_fields": "每条记录随带 AGT-046 的质量状态、来源与新鲜度",
                "attribution_excluded": True,
                "exclusion_note": "对照臂须在平台归因报表口径外单独成列，否则「零处理」会被平台归因销量掩盖",
            },
        },
        {
            "id": "L-GEO", "unit_field": "geo_region", "unit_level": "geo",
            "randomizable": True, "nesting_rule": None,
            "label": "地理区隔（站点/地区）",
            "share_of_frontend_traffic": None,
            "pool_allocation": {"treatment": 0.5, "control": 0.5},
            "stratify_keys": ["country", "currency"],
            "idempotency": {
                "key_fields": ["geo_region", "salt"], "salt": "holdback-geo-v1",
                "hash": "sha256", "modulus": 10000, "deterministic": True,
                "recompute_rule": "区隔级整体分配；同一区隔在实验期内不得换臂",
            },
            "attribution_compat": {
                "unit_grain": "geo 级（区隔内全部流量同臂）",
                "ledger": "广告实验台账逐区隔落臂",
                "attribution_excluded": True,
                "exclusion_note": "对照区隔在实验期内保持不投放，平台侧归因不得用于判定",
            },
        },
        {
            "id": "L-SESSION", "unit_field": "session_id", "unit_level": "session",
            "randomizable": True, "nesting_rule": None,
            "label": "会话级（仅站内即时交互）",
            "share_of_frontend_traffic": None,
            "pool_allocation": {"treatment": 0.5, "control": 0.5},
            "stratify_keys": ["device_type"],
            "idempotency": {
                "key_fields": ["session_id", "salt"], "salt": "holdback-sess-v1",
                "hash": "sha256", "modulus": 10000, "deterministic": True,
                "recompute_rule": "会话级哈希；跨会话不复用",
            },
            "attribution_compat": {"unit_grain": "会话级", "ledger": "站内事件流", "attribution_excluded": True},
            "admissible_only_for": "站内即时交互（点击/停留/加购等当次会话可闭合的行为）",
            "not_admissible_for": [
                "跨窗增量（观测窗 ≥ 14 个自然日）",
                "用户级 LTV / 复购 / 退款闭合类结果",
                "内容实验的格分配（契约 CTR-A-048 §2 ② 要求分流单元 = 独立站访客标识）",
            ],
        },
        {
            "id": "L-CHANNEL", "unit_field": "channel", "unit_level": "channel",
            "randomizable": False, "nesting_rule": None,
            "label": "渠道层（不随机化）",
            "share_of_frontend_traffic": None,
            "pool_allocation": {},
            "stratify_keys": [],
            "idempotency": {
                "key_fields": ["channel", "salt"], "salt": "holdback-channel-v1",
                "hash": "sha256", "modulus": 10000, "deterministic": True,
                "recompute_rule": "渠道层只做口径归属登记，不产生分流臂",
            },
            "attribution_compat": {"unit_grain": "渠道", "ledger": "渠道经营台账",
                                   "attribution_excluded": False},
            "randomizable": False,
            "why_not_randomizable": "渠道层没有可保持整块不投放/不花钱的对照单元："
                                    "减预算本身就是被检验的动作，随机化它等于把处理定义成对照",
        },
    ],
    "unassigned_layer_policy": "责任声明的 layer_id 不在 layers 里 ⇒ 判红，不得默认落到 L-USER",
    "routing_declarations": {
        "增量分析": {"if_feasible": "randomized_holdback", "if_infeasible": "quasi_experiment"},
        "实验设计": {"if_feasible": "randomized_holdback", "if_infeasible": "quasi_experiment"},
        "内容实验": {"if_feasible": "randomized_holdback", "if_infeasible": "quasi_experiment"},
        "广告实验": {"if_feasible": "randomized_holdback", "if_infeasible": "quasi_experiment"},
        "投放诊断": {"if_feasible": "holdback_corroboration_only",
                     "if_infeasible": "holdback_corroboration_only"},
        "预算分配": {"if_feasible": "holdback_corroboration_only",
                     "if_infeasible": "holdback_corroboration_only"},
    },
    "routing_declaration_policy": (
        "本块是**预登记**：建成前就把两条分支都写死，由实测流量决定走哪条。"
        "缺任何一条声明 ⇒ 判红（未声明即未标定）；声明与现算不一致 ⇒ 判红（J8）。"
    ),
}

# ---------------------------------------------------------------- 基线参数化（无事实源 ⇒ 只给网格）
ASSUMPTION_SET = {
    "note": "以下每一个数都**没有事实来源**（source 为空）。它们只用来把算式参数化，不得当作实测值引用。",
    "traffic": {
        "measurement_id": "independent_site_traffic",
        "source": None,
        "searched": [
            "playbook/（渲染快照：assets/*.json、*.html）",
            "paper2skills-vault/07-资源库/（*.md、*.json）",
            "paper2skills-research/data/（全部 JSON）",
            "/Users/lute/project/AI组织变革/（材料《AI组织变革》，docs/ 全量）",
        ],
        "found": (
            "无任何「日均访客/会话/曝光」读数。材料 docs/01-discovery/FACTS.md 的"
            "「尚无事实依据的内容」一节把**订单量**一并列为未知，并明写"
            "「不得从年营收或店铺数估算这些指标后再当作企业事实」。"
        ),
        "not_convertible": (
            "F-002（年规模约 100 亿人民币，口径待核对）与 F-004（独立站 15%）是 **GMV 构成**，"
            "不是访客数；从 GMV 推访客需要客单价与订单量，两者均无事实源 ⇒ 本脚本拒绝换算。"
        ),
    },
    "baseline_grid": {
        "site_conversion_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "content_ctr": [0.005, 0.01, 0.02, 0.03, 0.05],
    },
    "baseline_default": {"site_conversion_rate": 0.02, "content_ctr": 0.02,
                         "source": None, "label": "示例点，来源为空"},
    "traffic_grid": {"L-USER": [1000, 5000, 10000, 50000], "L-GEO": [2, 5, 20, 100],
                     "daily_content_impressions": [10000, 50000, 200000, 1000000]},
    "geo_cluster_cv_grid": [0.05, 0.10, 0.20, 0.30],
}

# ---------------------------------------------------------------- 污染防护（每条可判据）
POLLUTION_CONTROLS: list[dict] = [
    {
        "id": "P1", "name": "跨渠道曝光泄漏",
        "criterion": "对照单元在观测窗内的站外触点计数",
        "threshold": {"field": "external_touch_count", "must_equal": 0},
        "measure": "contamination_rate = 受站外触达的对照单元数 ÷ 对照单元总数",
        "ledger": "holdback_unit_ledger.external_touches（平台后台触点回传 + 广告后台受众投放日志）",
        "falsifier": "任一对照单元 external_touch_count > 0 ⇒ contamination_rate 必须现算并登记；>0 时该窗只出 ITT 口径并标「对照非零处理」",
        "why": "独立站 holdback 挡不住 Amazon/Meta 侧对同一批人的再触达，「零处理」是设计意图不是事实",
    },
    {
        "id": "P2", "name": "再营销受众重叠",
        "criterion": "受众包导出集合 ∩ holdback 单元集合",
        "threshold": {"intersection_size": 0},
        "measure": "overlap_count = |audience_export ∩ holdback_units|",
        "ledger": "受众包导出清单（Customer Match / lookalike seed / 平台受众包）+ holdback 分配台账",
        "falsifier": "交集非空 ⇒ 该窗作废并重新分流；每次受众推送前跑一次 exclusion 校验并留痕",
        "why": "把对照臂当种子灌进再营销，等于亲手把对照变成处理臂",
    },
    {
        "id": "P3", "name": "自然流量污染",
        "criterion": "对照臂自然访问量 > 0，且分流后结构同分布",
        "threshold": {"natural_sessions_min": 1, "balance_test": "预登记两比例检验 α=0.05"},
        "measure": "对照臂自然会话数；分流后新老访客比 / 设备构成的层间差异",
        "ledger": "独立站埋点事件流（source 维）+ 分流台账",
        "falsifier": "对照臂自然会话数 = 0（流量被截断，不是 holdback）或分流后构成差异显著 ⇒ 分流实现有缺陷，窗作废",
        "why": "站内 holdback 不阻断自然访问与自然转化；且 SEO/口碑外溢会让对照臂也受益 ⇒ 只能判 ITT",
    },
    {
        "id": "P4", "name": "平台算法学习期",
        "criterion": "预算/出价/素材变更后的冷却天数与单次幅度",
        "threshold": {"cooldown_days": 7, "max_single_change_pct": 20},
        "measure": "变更日 → 判定日的天数；单次调整相对当前月预算的百分比",
        "ledger": "Case/Event Ledger 的 Execution Attempt 时间戳 + 预算调整记录",
        "falsifier": "变更后不足 7 个自然日的读数被判为效果，或单次调整 > 20% ⇒ 该读数标 in_learning 并从判定中剔除",
        "why": "契约 CTR-A-050 §3 T3：单次调整超过 20% 会触发平台重新学习，短期效果反而更差；"
               "CTR-A-051 T4 的淘汰线用的正是「近 7 个自然日」窗口",
    },
]

# ---------------------------------------------------------------- 停止条件与提前终止
STOPPING_RULES: list[dict] = [
    {"id": "S1", "kind": "安全停止", "criterion": "可售天数 < 在途预计到货天数（可售天数按近 14 个自然日日均销量算）",
     "source": "CTR-A-049 §3 T4", "action": "不得加预算"},
    {"id": "S2", "kind": "输入过期", "criterion": "花费/库存/现金三份快照非同窗，或任一侧过期 > 1 个自然日",
     "source": "CTR-A-049 §6 / CTR-A-050 §6", "action": "该 Case 在接收门禁判 HOLD，转人工复核"},
    {"id": "S3", "kind": "劣臂淘汰", "criterion": "该臂近 7 个自然日的增量 ROAS 后验中位数 < 1.0",
     "source": "CTR-A-051 §3 T4", "action": "淘汰该臂，预算迁回主臂"},
    {"id": "S4", "kind": "提前终止-无效", "criterion": "预置无效边界：在信息比例 t 处条件功效 < 0.20",
     "source": "本设计补充（source 为空，assumption）", "action": "停止并关闭为 NoEffect，不得改判为「方向正确」"},
    {"id": "S5", "kind": "提前终止-有效", "criterion": "仅在预置信息时点按 O'Brien–Fleming 型 α 消耗读出；终期回到 α = 0.05",
     "source": "CTR-A-056 §3 T3", "action": "达边界才允许固化；禁止按固定 α 逐日检验"},
    {"id": "S6", "kind": "提前终止-污染", "criterion": "contamination_rate > 0.05，或对照臂出现系统性站外触达",
     "source": "本设计补充（source 为空，assumption）", "action": "立即停止并作废该窗，重新分流"},
    {"id": "S7", "kind": "观测窗下限", "criterion": "观测窗 < 14 个自然日，或退款/取消尚未闭合",
     "source": "CTR-A-057 §3/§6、CTR-A-048 §3、CTR-A-051 §3、CTR-A-056 §3",
     "action": "不得写入关闭记录"},
    {"id": "S8", "kind": "幂等与单元占用", "criterion": "同一分流单元当周内已进入另一未回填实验（CTR-A-056 为 1 个自然月）；同渠道并行实验 > 3 个",
     "source": "CTR-A-057 §6、CTR-A-056 §3 T7", "action": "不得发起新分流，进等待队列，不降级为「无对照观察」"},
]

# 层级粗细：数越大越粗。随机化单元必须与效应单元同级，或更细且带嵌套规则。
LAYER_FINENESS = {"impression": 0, "session": 1, "user": 2, "geo": 3, "channel": 4}

ADJECTIVE_BLACKLIST = ["明显", "显著改善", "差不多", "大致", "感觉", "看起来", "良好", "较好", "合理即可"]


# ================================================================ 统计核心
def z(p: float) -> float:
    return NormalDist().inv_cdf(p)


def n_per_arm_two_proportion(p_bar: float, rel_mde: float, alpha: float, power: float,
                             ratio: float = 1.0, average_rate: bool = True) -> dict:
    """两组比例检验的每臂样本量（契约 CTR-A-057/A-056/A-048 §3 的算式，含不等分配推广）。

    ratio = n_处理 / n_对照。契约原文写作 n = 2(z+z)²p̄(1−p̄)/δ²，是 ratio=1 的特例。
    已知答案：p̄=5%、相对 10%（5% vs 5.5%）、α=.05 双侧、power=.80 ⇒ 每臂 31,234（教科书口径）。
    """
    if not (0 < p_bar < 1):
        raise ValueError(f"基线率越界：{p_bar}")
    if rel_mde <= 0:
        raise ValueError(f"相对 MDE 必须为正：{rel_mde}")
    za, zb = z(1 - alpha / 2), z(power)
    p1 = p_bar
    p2 = p_bar * (1 + rel_mde)
    p_use = (p1 + p2) / 2 if average_rate else p_bar
    delta = p2 - p1
    n_control = (za + zb) ** 2 * p_use * (1 - p_use) * (1 + 1.0 / ratio) / delta ** 2
    n_treat = n_control * ratio
    return {
        "n_control": math.ceil(n_control), "n_treatment": math.ceil(n_treat),
        "n_control_exact": n_control, "n_treatment_exact": n_treat,
        "n_total_exact": n_control + n_treat,
        "n_per_arm_equal": math.ceil(n_control) if ratio == 1.0 else None,
        "n_total": math.ceil(n_control) + math.ceil(n_treat),
        "delta_abs": delta, "p_bar": p_bar, "rel_mde": rel_mde,
        "alpha": alpha, "power": power, "ratio": ratio,
        "form": "average_rate" if average_rate else "control_rate_only",
    }


def mde_from_n_two_proportion(n_total: int, p_bar: float, alpha: float, power: float,
                              ratio: float = 1.0, average_rate: bool = True) -> float:
    """反向口径：给定总样本量，反解可检出的**相对**最小效应（契约 CTR-A-056 §1 点名的反算口径）。"""
    za, zb = z(1 - alpha / 2), z(power)
    n_control = n_total / (1.0 + ratio)
    # p_use 依赖 p2 = p̄(1+r)，用两次不动点迭代收敛（r 越小越快）
    r = 0.05
    for _ in range(60):
        p2 = p_bar * (1 + r)
        p_use = (p_bar + p2) / 2 if average_rate else p_bar
        delta = (za + zb) * math.sqrt(p_use * (1 - p_use) * (1 + 1.0 / ratio) / n_control)
        r_new = delta / p_bar
        if abs(r_new - r) < 1e-12:
            r = r_new
            break
        r = r_new
    return r


def detectable_rel_effect_cluster(clusters_per_arm: int, cv: float, alpha: float, power: float) -> float:
    """区隔级（整块）随机化的可检测相对效应：r = (z_{1−α/2}+z_β)·CV·sqrt(2/m)。"""
    if clusters_per_arm < 1:
        raise ValueError("区隔数至少为 1")
    return (z(1 - alpha / 2) + z(power)) * cv * math.sqrt(2.0 / clusters_per_arm)


def clusters_needed(cv: float, rel_mde: float, alpha: float, power: float) -> float:
    return 2.0 * ((z(1 - alpha / 2) + z(power)) * cv / rel_mde) ** 2


# ================================================================ 契约现读（J4）
def _norm(text: str) -> str:
    return (text.replace("**", "").replace("−", "-").replace("–", "-")
                .replace("＝", "=").replace("\u3000", " "))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


CALIB_START = "## 3 标定规则"
CALIB_END = "## 4 重标定触发条件"


def calibration_section(raw: str) -> str | None:
    """只取 §3「标定规则」这一段。

    ⚠️ **必须排除 §1**：§1 的「不得移植的部分」逐条列了**卡面示例数字**
    （例如 CTR-A-048 §1 第 1 条写「期望相对提升 10%」），而该契约 §3 的 MDE 是 5%。
    把 §1 纳入抽取，A-048 会同时命中 10 与 5 两种取值 —— 那是**把不许移植的卡面数字
    当成了本业务取值**，正是本契约层要拦的那一类错。排除是载荷，不是洁癖（见 selftest）。
    """
    i = raw.find(CALIB_START)
    j = raw.find(CALIB_END)
    if i < 0 or j < 0 or j <= i:
        return None
    return raw[i:j]


def read_contracts(contracts_dir: Path) -> tuple[dict, list[str]]:
    """现读 6 份契约，抽 frontmatter responsibility 与 α / 功效 / MDE 字面。

    返回 (facts_by_contract_id, errors)。errors 非空 ⇒ 调用方决定 exit 2 或 3。
    """
    facts: dict[str, dict] = {}
    errors: list[str] = []
    for spec in CONTRACT_SPECS:
        cid = spec["contract_id"]
        hits = sorted(contracts_dir.glob(f"{cid}-*.md"))
        if len(hits) != 1:
            errors.append(f"MISSING::{cid}::契约文件命中 {len(hits)} 个（应为 1）")
            continue
        path = hits[0]
        raw = path.read_text(encoding="utf-8")
        sec = calibration_section(raw)
        if sec is None:
            errors.append(f"NOSECTION::{cid}::找不到 §3 标定规则 或 §4 重标定触发条件的边界")
            continue
        text = _norm(sec)
        m_resp = re.search(r"^responsibility:\s*(.+?)\s*$", raw, re.M)
        got_resp = m_resp.group(1).strip() if m_resp else None
        rec = {"contract_id": cid, "path": str(path.relative_to(REPO)),
               "sha256": sha256_file(path), "responsibility": got_resp,
               "calibration_section": "§3 标定规则（§1 卡面数字已排除）"}
        if got_resp != spec["responsibility"]:
            errors.append(f"RESP_MISMATCH::{cid}::{got_resp!r} != {spec['responsibility']!r}")
        for field, key in (("alpha", "alpha_pattern"), ("power", "power_pattern"), ("mde", "mde_pattern")):
            pat = spec.get(key)
            if pat is None:
                rec[field] = None
                continue
            found = re.findall(pat, text)
            uniq = sorted(set(found))
            if len(uniq) != 1:
                errors.append(f"PARSE::{cid}::{field}::命中 {len(uniq)} 种取值 {uniq}（应为 1）")
                rec[field] = None
            else:
                rec[field] = uniq[0]
        if spec.get("mde_expected_pct") is not None and rec["mde"] is not None:
            if int(rec["mde"]) != spec["mde_expected_pct"]:
                errors.append(f"MDE_CONFLICT::{cid}::{rec['mde']} != 内置标定 {spec['mde_expected_pct']}")
        facts[cid] = rec
    return facts, errors


# ================================================================ 判定
def judge_design(split: dict, facts: dict, traffic: dict | None,
                 baseline: dict, geo_cv: float | None) -> dict:
    """产出 J1…J11 与逐条责任判定。traffic=None ⇒ 只出 N* 与分支预备，不出「已达功效」。"""
    out: dict = {"judgements": {}, "responsibilities": [], "blocking": [], "warnings": []}
    J = out["judgements"]

    # ---- J1 分流参数自洽
    j1: list[str] = []
    layers = {L["id"]: L for L in split["layers"]}
    for L in split["layers"]:
        alloc = L.get("pool_allocation") or {}
        s = sum(alloc.values())
        if alloc and abs(s - 1.0) > 1e-9:
            j1.append(f"{L['id']} 池内比例和 = {s}（应为 1）")
        for k, v in alloc.items():
            if not (0 < v < 1):
                j1.append(f"{L['id']}.{k} = {v} 越界 (0,1)")
        share = L.get("share_of_frontend_traffic")
        if share is not None and not (0 < share <= 1):
            j1.append(f"{L['id']} share_of_frontend_traffic = {share} 越界 (0,1]")
        ctrl = L.get("effective_control_share_of_frontend")
        if ctrl is not None and not (0 < ctrl < 1):
            j1.append(f"{L['id']} effective_control_share_of_frontend = {ctrl} 越界 (0,1)")
        if L["id"] == "L-USER" and ctrl is not None and abs(ctrl - 0.10) > 1e-9:
            if not (L.get("allocation_rationale") or "").strip():
                j1.append("L-USER 对照臂占比偏离契约值 10%，但 allocation_rationale 为空")
    J["J1_split_self_consistent"] = {"pass": not j1, "violations": j1}

    # ---- J2 幂等键完备
    j2: list[str] = []
    for L in split["layers"]:
        idem = L.get("idempotency") or {}
        kf = idem.get("key_fields") or []
        if not kf:
            j2.append(f"{L['id']} 无 key_fields")
        if L["unit_field"] not in kf:
            j2.append(f"{L['id']} key_fields 未含单元字段 {L['unit_field']}")
        if "salt" not in kf or not idem.get("salt"):
            j2.append(f"{L['id']} 无 salt（不可复现）")
        if idem.get("hash") not in ("sha256", "sha1", "md5"):
            j2.append(f"{L['id']} hash 未声明")
        if idem.get("deterministic") is not True:
            j2.append(f"{L['id']} deterministic 未置 True")
        if not isinstance(idem.get("modulus"), int) or idem.get("modulus", 0) <= 0:
            j2.append(f"{L['id']} modulus 非法")
    J["J2_idempotency"] = {"pass": not j2, "violations": j2}

    # ---- J3 层匹配（方法学判据） + J10 覆盖
    #   规则：随机化层级必须**等于**效应层级；
    #   随机化更**粗** ⇒ 效应在更粗单元上不可识别（红）；
    #   随机化更**细** ⇒ 只在该层声明了可判据的嵌套规则
    #   （同一更粗单元在一次实验内不得跨臂）时才允许，否则同一用户两臂都在（红）。
    seen: list[str] = []
    j3: list[str] = []
    j10: list[str] = []
    for spec in CONTRACT_SPECS:
        seen.append(spec["responsibility"])
        L = layers.get(spec["layer_id"])
        if L is None:
            j3.append(f"{spec['responsibility']} 声明的 layer_id={spec['layer_id']} 不在分流层里"
                      f"（不得默认落到 L-USER）")
            continue
        eff, rnd = spec["inference_level"], L["unit_level"]
        if eff == rnd:
            continue
        if eff not in LAYER_FINENESS or rnd not in LAYER_FINENESS:
            j3.append(f"{spec['responsibility']}：未知层级 效应={eff} / 随机化={rnd}")
            continue
        if LAYER_FINENESS[rnd] > LAYER_FINENESS[eff]:
            j3.append(f"{spec['responsibility']}：随机化层级 {rnd} 比效应层级 {eff} **更粗**"
                      f" ⇒ 该效应在更粗单元上不可识别（方法学错误，不是参数选择）")
        else:
            nest = (L.get("nesting_rule") or "").strip()
            if not nest or not re.search(r"\d", nest):
                j3.append(f"{spec['responsibility']}：随机化层级 {rnd} 比效应层级 {eff} 更细，"
                          f"但 {L['id']} 未声明「同一 {eff} 单元在一次实验内不得跨臂」的可判据嵌套规则"
                          f" ⇒ 同一 {eff} 可能在两臂都出现（方法学错误：会话级随机化测不了用户级效应）")
        if L.get("randomizable") is False:
            j3.append(f"{spec['responsibility']} 被挂到不可随机化的层 {L['id']}"
                      f"（{L.get('why_not_randomizable','未说明')}）")
    if len(seen) != 6 or len(set(seen)) != 6:
        j10.append(f"责任条数 = {len(seen)}（去重 {len(set(seen))}），应为 6")
    for cid, rec in facts.items():
        exp = next(s["responsibility"] for s in CONTRACT_SPECS if s["contract_id"] == cid)
        if rec.get("responsibility") != exp:
            j10.append(f"{cid} frontmatter responsibility = {rec.get('responsibility')!r} ≠ {exp!r}")
    J["J3_layer_match"] = {"pass": not j3, "violations": j3}
    J["J10_coverage"] = {"pass": not j10, "violations": j10}

    # ---- J4 契约事实源活体（facts 由 read_contracts 填；此处只判其完整性）
    j4: list[str] = []
    for spec in CONTRACT_SPECS:
        rec = facts.get(spec["contract_id"])
        if rec is None:
            j4.append(f"{spec['contract_id']} 未读到")
            continue
        for f, key in (("alpha", "alpha_pattern"), ("power", "power_pattern")):
            if spec.get(key) is None:
                continue  # 该条契约本就不是功效设计（无 α/β 出口）⇒ 不要求，也不误报
            if rec.get(f) is None:
                j4.append(f"{spec['contract_id']}.{f} 抽取失败")
        if spec.get("mde_pattern") and rec.get("mde") is None:
            j4.append(f"{spec['contract_id']}.mde 抽取失败")
        if not rec.get("sha256"):
            j4.append(f"{spec['contract_id']} 无内容指纹")
    J["J4_contract_source"] = {"pass": not j4, "violations": j4}

    # ---- J5 正向功效已知答案
    known = n_per_arm_two_proportion(0.05, 0.10, 0.05, 0.80, average_rate=True)
    kn = known["n_per_arm_equal"]
    J5_LOW, J5_HIGH = 29000, 34000
    J["J5_known_answer"] = {
        "pass": J5_LOW <= kn <= J5_HIGH,
        "n_per_arm": kn, "accepted_range": [J5_LOW, J5_HIGH],
        "case": "p̄=5%、相对 MDE=10%（5% vs 5.5%）、α=0.05 双侧、power=0.80",
    }

    # ---- J6 正反互逆
    r_back = mde_from_n_two_proportion(known["n_total_exact"], 0.05, 0.05, 0.80)
    r_back_rounded = mde_from_n_two_proportion(known["n_total"], 0.05, 0.05, 0.80)
    J["J6_inverse"] = {
        "pass": abs(r_back - 0.10) < 1e-9 and abs(r_back_rounded - 0.10) < 1e-4,
        "rel_mde_inverted_exact": r_back, "abs_err_exact": abs(r_back - 0.10),
        "rel_mde_inverted_rounded_n": r_back_rounded, "abs_err_rounded": abs(r_back_rounded - 0.10),
        "note": "取整后的 n 反解会有 1/(2n) 量级的偏差（此处 ~1.6e-6），故两条容差分开给",
    }

    # ---- J9 geo 层可检测效应（数值是**发现**；声明与该数值不符才是判红 —— 同 J8 的两边口径）
    decl = split.get("routing_declarations", {})
    if geo_cv is None:
        J["J9_geo_consistency"] = {
            "pass": True, "violations": [], "pending": ["geo CV 未提供 ⇒ 该条只出假设网格，不出判定"],
            "clusters_per_arm": 2, "cv": None, "computed_routing": None,
        }
    else:
        m = next(s.get("min_clusters", 2) for s in CONTRACT_SPECS if s["contract_id"] == "CTR-A-051")
        rmin = detectable_rel_effect_cluster(m, geo_cv, 0.05, 0.80)
        cv_max = GEO_ACTIONABLE_REL / ((z(0.975) + z(0.80)) * math.sqrt(2.0 / m))
        ok = rmin <= GEO_ACTIONABLE_REL
        d = decl.get("广告实验") or {}
        declared = d.get("if_feasible") if ok else d.get("if_infeasible")
        violations = []
        if not d:
            violations.append("广告实验 未在 split.routing_declarations 里声明路由（未声明即未标定）")
        elif declared != (ROUTE_RANDOMIZED if ok else ROUTE_QUASI):
            violations.append(f"广告实验 声明 {declared!r}，但区隔数 {m}、CV={geo_cv} 下"
                              f"可检测相对效应 {rmin:.1%} "
                              f"{'≤' if ok else '>'} 迁移阈值 {GEO_ACTIONABLE_REL:.0%}"
                              f" ⇒ 应走 {ROUTE_RANDOMIZED if ok else ROUTE_QUASI}")
        J["J9_geo_consistency"] = {
            "pass": not violations, "violations": violations,
            "clusters_per_arm": m, "cv": geo_cv,
            "detectable_rel_effect": rmin, "actionable_rel_effect": GEO_ACTIONABLE_REL,
            "cv_upper_bound_for_actionable": cv_max,
            "computed_routing": ROUTE_RANDOMIZED if ok else ROUTE_QUASI,
            "declared_routing": declared,
            "finding": (f"区隔级随机化在 {m} 个区隔、CV={geo_cv} 下只能检出 {rmin:.1%} 的相对效应；"
                        f"要承载 {GEO_ACTIONABLE_REL:.0%} 的迁移阈值须 CV ≤ {cv_max:.2%}"),
        }

    # ---- J11 停止条件可判据
    j11: list[str] = []
    for s in STOPPING_RULES:
        if not re.search(r"\d", s["criterion"]):
            j11.append(f"{s['id']} 判据无数值阈值：{s['criterion']}")
        for w in ADJECTIVE_BLACKLIST:
            if w in s["criterion"]:
                j11.append(f"{s['id']} 判据含形容词「{w}」")
    if not any(s["kind"] == "提前终止-有效" and "序贯" in s["criterion"] + s["action"]
               or s["id"] == "S5" for s in STOPPING_RULES):
        j11.append("缺序贯/提前读出边界（S5）")
    J["J11_stopping_falsifiable"] = {"pass": not j11, "violations": j11}

    # ---- 逐条责任
    for spec in CONTRACT_SPECS:
        rec = facts.get(spec["contract_id"]) or {}
        alpha = float(rec["alpha"]) if rec.get("alpha") else None
        power = float(rec["power"]) if rec.get("power") else None
        row: dict = {
            "contract_id": spec["contract_id"], "responsibility": spec["responsibility"],
            "role_id": spec["role_id"], "metric": spec["metric"],
            "result_type": spec["result_type"],
            "inference_level": spec["inference_level"], "layer_id": spec["layer_id"],
            "alpha": alpha, "power": power,
            "alpha_source": (f"{rec.get('path','?')} §3" if alpha is not None
                             else "契约未给 α/β（该条不是功效设计）"),
            "result_denominator": spec.get("result_denominator"),
            "inference_level_note": spec.get("inference_level_note"),
            "note": spec["note"],
        }
        rtype = spec["result_type"]

        d = (decl.get(spec["responsibility"]) or {})
        if rtype == "structural":
            row.update({
                "declared_routing": d.get("if_infeasible"),
                "computed_routing": ROUTE_CORROBORATION,
                "verdict": "USE_QUASI_EXPERIMENT",
                "reason": "契约自身把识别口径定为准实验/响应曲线；holdback 只替换被解释变量或提供佐证读数，"
                          "不承载该条的判定量 ⇒ 该条用准实验（holdback 仅佐证）",
                "n_star_daily_units": None, "mde_rel": None,
            })
        elif rtype == "cluster_continuous":
            j9 = J["J9_geo_consistency"]
            ok = bool(j9.get("pass")) and j9.get("computed_routing") == ROUTE_RANDOMIZED
            row.update({
                "mde_rel": None,
                "min_clusters": spec.get("min_clusters"), "min_exposure_per_arm": spec.get("min_exposure_per_arm"),
                "detectable_rel_effect": j9.get("detectable_rel_effect"),
                "cv_upper_bound_for_actionable": j9.get("cv_upper_bound_for_actionable"),
                "declared_routing": d.get("if_feasible") if ok else d.get("if_infeasible"),
                "computed_routing": ROUTE_RANDOMIZED if ok else ROUTE_QUASI,
                "verdict": "RANDOMIZED_OK" if ok else "USE_QUASI_EXPERIMENT",
                "reason": ("区隔级随机化在给定 CV 下可检出 ≤ 迁移阈值"
                           if ok else
                           (f"区隔级随机化在 {j9.get('clusters_per_arm')} 个区隔、CV={j9.get('cv')} 下"
                            f"可检测相对效应仅 {j9['detectable_rel_effect']:.1%} "
                            f"(> 迁移阈值 {GEO_ACTIONABLE_REL:.0%}) ⇒ 该条用准实验/切换实验"
                            if isinstance(j9.get("detectable_rel_effect"), float) else "geo CV 未提供")),
                "n_star_daily_units": None,
            })
        else:
            mde_rel = spec["mde_expected_pct"] / 100.0
            p_bar = baseline[spec["baseline_key"]]
            cells = spec.get("cells", 1)
            calc = n_per_arm_two_proportion(p_bar, mde_rel, alpha, power)
            n_total = calc["n_total"] if cells == 1 else math.ceil(calc["n_control"]) * cells
            # 内容实验的分母是曝光（不受池占比限制）；其余按 L-USER 的池占比折算前台流量
            L = layers[spec["layer_id"]]
            pool_share = L.get("share_of_frontend_traffic") or 1.0
            daily_needed_pool = n_total / CYCLE_DAYS
            # ⚠️ 内容实验的分母是**曝光**（契约 CTR-A-048 §3 T3 的「日均可分配曝光数」），
            #    与前台会话不同量纲 ⇒ 不再按池占比折算。首版对两者都除了 pool_share，把 14,009 印成了 70,047。
            daily_needed_frontend = (daily_needed_pool if rtype == "rate_impression"
                                     else daily_needed_pool / pool_share)
            row.update({
                "mde_rel": mde_rel,
                "mde_source": f"{rec.get('path','?')} §3（相对提升 {spec['mde_expected_pct']}%）",
                "baseline_key": spec["baseline_key"], "baseline_value": p_bar,
                "baseline_source": baseline.get("source"),
                "cells": cells,
                "n_per_arm": calc["n_control"], "n_total_units": n_total,
                "pool_share_of_frontend": pool_share,
                "n_star_daily_units_pool": math.ceil(daily_needed_pool),
                "n_star_daily_units": math.ceil(daily_needed_frontend),
                "cycle_days": CYCLE_DAYS,
                "declared_routing_infeasible": d.get("if_infeasible"),
                "declared_routing_feasible": d.get("if_feasible"),
            })
            if traffic is None:
                row.update({"declared_routing": None, "computed_routing": None,
                            "verdict": "TRAFFIC_INPUT_MISSING",
                            "reason": "无带来源的实测流量 ⇒ 只出 N* 与分支预备，不出「已达功效」"})
            else:
                key = "daily_content_impressions" if rtype == "rate_impression" else spec["layer_id"]
                actual = traffic["daily_eligible_units"].get(key)
                if actual is None:
                    row.update({"declared_routing": None, "computed_routing": None,
                                "verdict": "TRAFFIC_INPUT_MISSING",
                                "reason": f"实测流量缺字段 {key}"})
                else:
                    # 用户级/曝光级口径不同：L-USER 的实验只发生在池内（池 = 前台流量 × pool_share），
                    # 内容实验的分母是曝光，与前台会话不同量纲，不再折算。
                    divisor = actual if rtype == "rate_impression" else actual * pool_share
                    days = n_total / divisor
                    r_hat = mde_from_n_two_proportion(n_total, p_bar, alpha, power)
                    feasible = days <= CYCLE_DAYS
                    routing = ROUTE_RANDOMIZED if feasible else ROUTE_QUASI
                    row.update({
                        "actual_daily_units": actual, "actual_key": key,
                        "effective_daily_experiment_units": divisor,
                        "days_to_power": days, "feasible_within_cycle": feasible,
                        "detectable_rel_effect_at_actual": r_hat,
                        # ⚠️ declared 取**设计文件的预登记分支**，computed 取**现算**。
                        #    首版把两者都写成 routing ⇒ J8 恒真（自检 ⑧d 当场抓到）。
                        "declared_routing": (d.get("if_feasible") if feasible else d.get("if_infeasible")),
                        "computed_routing": routing,
                        "verdict": "RANDOMIZED_OK" if feasible else "USE_QUASI_EXPERIMENT",
                        "reason": (f"所需 {days:.1f} 天 ≤ 90 天 ⇒ 随机化 holdback 可承载"
                                   if feasible else
                                   f"所需 {days:.1f} 天 > 90 天（缺 {days - CYCLE_DAYS:.1f} 天）⇒ 该条用准实验"),
                    })
        out["responsibilities"].append(row)

    # ---- J7 流量来源（由调用方在 IO 层判，这里只登记结论位）
    # ---- J8 声明 == 计算（两边都能抓：假绿与假红）
    #      「未定」与「不符」必须分开：流量没拿到时算不出 computed，
    #      那是**待定**（由 J7 判 exit 2），不是声明错了。混在一起会把「没测到」记成「测了是红的」。
    j8: list[str] = []
    j8_pending: list[str] = []
    for row in out["responsibilities"]:
        name = row["responsibility"]
        if name not in decl:
            j8.append(f"{name}：设计文件未声明路由（未声明即未标定）")
            continue
        d, c = row.get("declared_routing"), row.get("computed_routing")
        if d is None or c is None:
            j8_pending.append(f"{name}：预登记 {decl[name]['if_feasible']} / "
                              f"{decl[name]['if_infeasible']}，计算值待实测流量定")
            continue
        if d != c:
            j8.append(f"{name}：声明 {d} ≠ 计算 {c}（预登记 {decl[name]}）")
    extra = sorted(set(decl) - {r["responsibility"] for r in out["responsibilities"]})
    for e in extra:
        j8.append(f"设计文件声明了未知责任 {e}（超过 6 条）")
    J["J8_route_match"] = {"pass": not j8, "violations": j8, "pending": j8_pending}
    return out


# ================================================================ IO
def load_split(path: str | None) -> tuple[dict, str]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_SPLIT)), "builtin"
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8")), str(p)


def load_traffic(path: str) -> tuple[dict, str | None]:
    """返回 (traffic, error)。error 非空 ⇒ exit 2。"""
    p = Path(path)
    if not p.is_file():
        return {}, f"流量文件不存在：{p}"
    try:
        t = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {}, f"流量文件不是合法 JSON：{e}"
    src = (t.get("source") or "").strip()
    if not src:
        return t, "流量文件 `source` 为空 —— 无来源的流量数字按本仓库纪律不接受（编造的数字是最重的一类错）"
    if not isinstance(t.get("daily_eligible_units"), dict) or not t["daily_eligible_units"]:
        return t, "流量文件缺 `daily_eligible_units`"
    if not isinstance(t.get("baseline"), dict) or not t["baseline"]:
        return t, "流量文件缺 `baseline`"
    if not (t["baseline"].get("source") or "").strip():
        return t, "流量文件的 `baseline.source` 为空（基线率同样必须有来源）"
    return t, None


def split_cost_comparison(split: dict, facts: dict) -> dict:
    """两种分流读法在**同样 n** 下的前台流量代价（现算，不抄文档）。

    读法 A（本设计）：池 = 前台 20%，池内 1:1 ⇒ 对照 = 前台 10%，同时满足「层内 1:1」。
    读法 B：100% 前台直接 90/10 ⇒ 层内不是 1:1（违反 CTR-A-057 §3 T2），但在同样 n 下省前台流量。
    """
    L = next(l for l in split["layers"] if l["id"] == "L-USER")
    pool = L.get("share_of_frontend_traffic") or 1.0
    ctrl = L.get("effective_control_share_of_frontend") or 0.5
    ratio_b = (1.0 - ctrl) / ctrl
    rows = {}
    for spec in CONTRACT_SPECS:
        if spec["result_type"] not in ("rate",):
            continue
        p_bar = ASSUMPTION_SET["baseline_default"][spec["baseline_key"]]
        a = n_per_arm_two_proportion(p_bar, spec["mde_expected_pct"] / 100.0, 0.05, 0.80)
        b = n_per_arm_two_proportion(p_bar, spec["mde_expected_pct"] / 100.0, 0.05, 0.80, ratio=ratio_b)
        front_a = a["n_total"] / pool
        front_b = b["n_total"]          # 100% 前台参与随机化
        rows[spec["responsibility"]] = {
            "pool_20_1to1": {"pool_units": a["n_total"], "frontend_units": math.ceil(front_a)},
            "direct_90_10": {"pool_units": b["n_total"], "frontend_units": math.ceil(front_b),
                             "ratio_treatment_over_control": ratio_b},
            "frontend_ratio_direct_over_pool": front_b / front_a,
        }
    return {
        "note": "读法 A 满足契约两条（对照 10% + 层内 1:1）；读法 B 违反层内 1:1 但在同样 n 下更省前台流量。"
                "两个数都由本脚本现算，随基线 p̄ 变动（此处取示例点，来源为空）。",
        "by_responsibility": rows,
    }


def sensitivity(split: dict, facts: dict, geo_cv_grid: list) -> dict:
    """纯参数化：把「所需流量」与「可检测效应」写成基线率/流量的函数，不引用任何实测值。"""
    out = {"baseline_grid": ASSUMPTION_SET["baseline_grid"], "traffic_grid": ASSUMPTION_SET["traffic_grid"],
           "geo_cv_grid": geo_cv_grid, "rows": []}
    L = next(l for l in split["layers"] if l["id"] == "L-USER")
    pool = L.get("share_of_frontend_traffic") or 1.0
    for spec in CONTRACT_SPECS:
        if spec["result_type"] not in ("rate", "rate_impression"):
            continue
        cells = spec.get("cells", 1)
        for p_bar in ASSUMPTION_SET["baseline_grid"][spec["baseline_key"]]:
            calc = n_per_arm_two_proportion(p_bar, spec["mde_expected_pct"] / 100.0, 0.05, 0.80)
            n_total = math.ceil(calc["n_control"]) * cells if cells > 1 else calc["n_total"]
            key = "daily_content_impressions" if spec["result_type"] == "rate_impression" else "L-USER"
            row = {"responsibility": spec["responsibility"], "baseline": p_bar, "cells": cells,
                   "n_total_units": n_total,
                   "n_star_daily_units_pool": math.ceil(n_total / CYCLE_DAYS),
                   "n_star_daily_frontend_units": (
                       math.ceil(n_total / CYCLE_DAYS) if spec["result_type"] == "rate_impression"
                       else math.ceil(n_total / CYCLE_DAYS / pool)),
                   "traffic_key": key, "days_at_grid": {}}
            for t in ASSUMPTION_SET["traffic_grid"][key]:
                div = t if spec["result_type"] == "rate_impression" else t * pool
                row["days_at_grid"][str(t)] = round(n_total / div, 1)
            out["rows"].append(row)
    return out


def build_artifact(split, split_src, facts, traffic, traffic_err, judge, mode, baseline, geo_cv) -> dict:
    return {
        "schema": "holdback-design/v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "generator": "paper2skills-research/scripts/holdback_power.py",
        "mode": mode,
        "provenance": {
            "traffic": {
                "status": "measured" if (traffic and not traffic_err) else
                          ("rejected" if traffic_err and traffic else "absent"),
                "source": (traffic or {}).get("source"),
                "error": traffic_err,
                "searched_for_fact_source": ASSUMPTION_SET["traffic"]["searched"],
                "found": ASSUMPTION_SET["traffic"]["found"],
                "not_convertible": ASSUMPTION_SET["traffic"]["not_convertible"],
            },
            "baseline": {"value": baseline, "source": baseline.get("source")},
            "split_source": split_src,
            "contracts": [{"contract_id": c, "path": r.get("path"), "sha256": r.get("sha256"),
                           "responsibility": r.get("responsibility"),
                           "alpha": r.get("alpha"), "power": r.get("power"), "mde_pct": r.get("mde")}
                          for c, r in sorted(facts.items())],
        },
        "assumption_set": ASSUMPTION_SET,
        "split": split,
        "pollution_controls": POLLUTION_CONTROLS,
        "stopping_rules": STOPPING_RULES,
        "responsibilities": judge["responsibilities"],
        "judgements": judge["judgements"],
        "summary": summarise(judge),
        "split_cost_comparison": split_cost_comparison(split, facts),
        "sensitivity": sensitivity(split, facts, ASSUMPTION_SET["geo_cluster_cv_grid"]),
    }


def summarise(judge) -> dict:
    rows = judge["responsibilities"]
    tally: dict[str, int] = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    return {
        "n_responsibilities": len(rows),
        "verdict_tally": tally,
        "quasi_list": [r["responsibility"] for r in rows if r["verdict"] == "USE_QUASI_EXPERIMENT"],
        "randomized_list": [r["responsibility"] for r in rows if r["verdict"] == "RANDOMIZED_OK"],
        "missing_traffic_list": [r["responsibility"] for r in rows if r["verdict"] == "TRAFFIC_INPUT_MISSING"],
        "n_judgements_failed": sum(1 for v in judge["judgements"].values() if not v.get("pass")),
    }


def emit(judge, artifact, json_out, exit_code, reason) -> int:
    if json_out:
        p = Path(json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n📄 机读产物 → {p.relative_to(REPO) if str(p).startswith(str(REPO)) else p}")
    print("\n" + "=" * 78)
    for k, v in judge["judgements"].items():
        mark = "✅" if v.get("pass") else "🔴"
        print(f"  {mark} {k}")
        for viol in v.get("violations", []):
            print(f"       · {viol}")
        for pend in v.get("pending", []):
            print(f"       ○ 待定：{pend}")
    print("-" * 78)
    hdr = f"{'责任':<8}{'MDE':>6}{'所需日均单元':>13}{'单位':>8}{'实测':>10}  结论"
    print(hdr)
    for r in judge["responsibilities"]:
        mde = f"{r['mde_rel']*100:.0f}%" if r.get("mde_rel") else "—"
        nstar = r.get("n_star_daily_units")
        star = f"{nstar:,}" if isinstance(nstar, int) else "—"
        unit = {"rate_impression": "曝光", "rate": "前台会话"}.get(r.get("result_type"), "")
        act = r.get("actual_daily_units")
        act_s = f"{act:,}" if isinstance(act, int) else "—"
        print(f"{r['responsibility']:<8}{mde:>6}{star:>13}{unit:>8}{act_s:>10}  {r['verdict']}")
    print("-" * 78)
    s = artifact["summary"]
    print(f"  判定合计：{s['verdict_tally']}")
    if s["quasi_list"]:
        print(f"  ⇒ 该条用准实验：{' · '.join(s['quasi_list'])}")
    if s["missing_traffic_list"]:
        print(f"  ⇒ 缺实测流量、暂不能判：{' · '.join(s['missing_traffic_list'])}")
    print("=" * 78)
    print(f"退出码 {exit_code} —— {reason}")
    return exit_code


# ================================================================ 主流程
def run(args) -> int:
    contracts_dir = Path(args.contracts_dir)
    if not contracts_dir.is_dir():
        print(f"❌ 契约目录不存在：{contracts_dir} —— 没拿到输入 ≠ 通过", file=sys.stderr)
        return EXIT_NOINPUT

    try:
        split, split_src = load_split(args.design)
    except FileNotFoundError as e:
        print(f"❌ 分流参数文件不存在：{e}", file=sys.stderr)
        return EXIT_NOINPUT
    except json.JSONDecodeError as e:
        print(f"❌ 分流参数文件不是合法 JSON：{e}", file=sys.stderr)
        return EXIT_NOINPUT

    facts, errors = read_contracts(contracts_dir)
    if errors:
        missing = [e for e in errors if e.startswith("MISSING::")]
        print(f"❌ 契约读取失败（{len(errors)} 条）：", file=sys.stderr)
        for e in errors:
            print("   -", e, file=sys.stderr)
        return EXIT_NOINPUT if missing else EXIT_INTERNAL

    traffic, traffic_err = None, None
    mode = "assumed"
    if args.traffic:
        traffic, traffic_err = load_traffic(args.traffic)
        mode = "measured"
        if traffic_err:
            print(f"❌ 流量输入被拒：{traffic_err}", file=sys.stderr)
            traffic = None
    elif args.assume:
        mode = "assumed"
    else:
        traffic_err = ("未提供 --traffic。独立站流量在本仓库没有事实源（搜索范围见产物 provenance."
                       "traffic.searched）⇒ 退出码 2，不是 0。"
                       "要用假设网格跑演示请显式加 --assume（仍为 exit 2）。")

    if traffic is not None:
        baseline = dict(traffic["baseline"])
        baseline.setdefault("source", traffic["baseline"].get("source"))
        geo_cv = traffic.get("geo_cluster_cv")
    else:
        baseline = dict(ASSUMPTION_SET["baseline_default"])
        baseline["source"] = None
        geo_cv = None

    judge = judge_design(split, facts, traffic, baseline, geo_cv)
    artifact = build_artifact(split, split_src, facts, traffic, traffic_err, judge, mode, baseline, geo_cv)

    # ---- J7 流量来源判据（IO 层）
    src_ok = bool(traffic) and not traffic_err and bool((traffic.get("source") or "").strip()) \
             and bool((traffic.get("baseline", {}).get("source") or "").strip())
    judge["judgements"]["J7_traffic_provenance"] = {
        "pass": src_ok,
        "source": (traffic or {}).get("source"),
        "violations": [] if src_ok else [traffic_err or "流量来源为空"],
    }
    artifact["judgements"] = judge["judgements"]
    artifact["summary"] = summarise(judge)

    hard_fail = [k for k, v in judge["judgements"].items() if not v.get("pass")]
    if not src_ok:
        artifact["exit"] = {"code": EXIT_NOINPUT, "reason": "输入没拿到（≠ 通过）"}
        return emit(judge, artifact, args.json_out, EXIT_NOINPUT,
                    "输入没拿到：无带来源的实测流量 ⇒ 不是通过；逐条判定见上表")
    if hard_fail:
        artifact["exit"] = {"code": EXIT_RED, "reason": f"判据未过：{hard_fail}"}
        return emit(judge, artifact, args.json_out, EXIT_RED, f"判红：{', '.join(hard_fail)}")
    artifact["exit"] = {"code": EXIT_OK, "reason": "全过"}
    return emit(judge, artifact, args.json_out, EXIT_OK, "全过（含声明路由 == 计算路由）")


# ================================================================ 自检
def _run_cli(argv, timeout=180):
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv],
                          capture_output=True, text=True, timeout=timeout)


def selftest() -> int:
    scripts_dir = Path(__file__).resolve().parent
    before = sorted(os.listdir(scripts_dir))
    cases: list[tuple[str, bool]] = []
    tmp = Path(tempfile.mkdtemp(prefix="holdback-selftest-"))
    try:
        # ---- J5/J6 已知答案与互逆
        k = n_per_arm_two_proportion(0.05, 0.10, 0.05, 0.80)
        cases.append(("① 已知答案 5%→5.5% ⇒ 每臂 ∈[29000,34000]",
                      29000 <= k["n_per_arm_equal"] <= 34000))
        cases.append(("①b 控制组口径（p̄ 不取均值）落在 [28000,32000]",
                      28000 <= n_per_arm_two_proportion(0.05, 0.10, 0.05, 0.80,
                                                        average_rate=False)["n_per_arm_equal"] <= 32000))
        r_exact = mde_from_n_two_proportion(k["n_total_exact"], 0.05, 0.05, 0.80)
        r_round = mde_from_n_two_proportion(k["n_total"], 0.05, 0.05, 0.80)
        cases.append(("② 正反互逆（实数 n）|Δ| < 1e-9", abs(r_exact - 0.10) < 1e-9))
        cases.append(("②a 取整后反解偏差 < 1e-4（1/(2n) 量级）", abs(r_round - 0.10) < 1e-4))
        cases.append(("②b 篡改样本：把 n 砍半 ⇒ 反解 MDE 必须变大（判据会失败）",
                      mde_from_n_two_proportion(k["n_total"] // 2, 0.05, 0.05, 0.80) > 0.10))
        cases.append(("③ 不等分配 90/10 的总量 > 1:1 的总量",
                      n_per_arm_two_proportion(0.02, 0.05, 0.05, 0.80, ratio=9)["n_total"]
                      > n_per_arm_two_proportion(0.02, 0.05, 0.05, 0.80, ratio=1)["n_total"]))
        cases.append(("④ 区隔级可检测效应 r(m=2,CV=.30) = 84.05% ±0.5pp",
                      abs(detectable_rel_effect_cluster(2, 0.30, 0.05, 0.80) - 0.8405) < 0.005))
        cv_max = GEO_ACTIONABLE_REL / ((z(0.975) + z(0.80)) * math.sqrt(2.0 / 2))
        cases.append(("④b CV 上限（2 区隔承载 20% 迁移阈值）= 7.14% ±0.2pp", abs(cv_max - 0.0714) < 0.002))

        # ---- ⑤ 蒙特卡洛仿真复核闭式功效（独立于公式）
        try:
            import random as _rnd
            R = _rnd.Random(20260913)
            n_arm = k["n_per_arm_equal"]
            rej = 0
            reps = 3000
            for _ in range(reps):
                # ⚠️ 必须用实例 R，不能用 _rnd.binomialvariate（那是全局 RNG ⇒ 自检不可复现）
                a = R.binomialvariate(n_arm, 0.05)
                b = R.binomialvariate(n_arm, 0.055)
                p_pool = (a + b) / (2 * n_arm)
                se = math.sqrt(p_pool * (1 - p_pool) * 2 / n_arm)
                if se > 0 and abs(b - a) / n_arm / se > 1.959963984540054:
                    rej += 1
            emp = rej / reps
            R2 = _rnd.Random(20260913)
            reps2, rej2 = 300, 0
            for _ in range(reps2):
                a = R2.binomialvariate(n_arm, 0.05)
                b = R2.binomialvariate(n_arm, 0.055)
                p_pool = (a + b) / (2 * n_arm)
                se = math.sqrt(p_pool * (1 - p_pool) * 2 / n_arm)
                if se > 0 and abs(b - a) / n_arm / se > 1.959963984540054:
                    rej2 += 1
            emp2 = rej2 / reps2
            cases.append((f"⑤ 仿真功效 {emp:.3f} ∈ [0.765,0.835]（闭式 0.80）", 0.765 <= emp <= 0.835))
            cases.append((f"⑤b 仿真可复现：同种子独立重跑 300 次得 {emp2:.4f}（写死期望值）",
                          emp2 == 0.7966666666666666))
        except AttributeError:
            cases.append(("⑤ 仿真复核：python<3.12 无 binomialvariate ⇒ 判为**没测到**", False))

        # ---- ⑥ 契约现读
        facts, errs = read_contracts(CONTRACTS_DIR)
        cases.append(("⑥ 6 份契约全部读到且 α/β/MDE 抽取成功", not errs and len(facts) == 6))
        cases.append(("⑥b MDE 与内置标定相符（增量分析 5% / 实验设计 10% / 内容实验 5%）",
                      facts["CTR-A-057"]["mde"] == "5" and facts["CTR-A-056"]["mde"] == "10"
                      and facts["CTR-A-048"]["mde"] == "5"))
        cases.append(("⑥c 契约读不到 ⇒ 报错（拿空目录试）",
                      len(read_contracts(tmp / "no-such-dir")[1]) == 6))
        # ⑥d §1 排除是**载荷**：全文抽 MDE 会命中两种取值（卡面 10% 与本业务 5%）
        raw48 = next(CONTRACTS_DIR.glob("CTR-A-048-*.md")).read_text(encoding="utf-8")
        full = sorted(set(re.findall(r"相对提升\s*(\d+)\s*%", _norm(raw48))))
        only3 = sorted(set(re.findall(r"相对提升\s*(\d+)\s*%", _norm(calibration_section(raw48)))))
        cases.append((f"⑥d §1 卡面数字确实会污染抽取（全文命中 {full} / §3 命中 {only3}）⇒ 排除有载荷",
                      len(full) > 1 and only3 == ["5"]))
        cases.append(("⑥e calibration_section 边界缺失时返回 None（不静默取全文）",
                      calibration_section("## 3 标定规则\n没有 §4") is None))

        # ---- ⑥f 分流读法代价（现算）
        cc = split_cost_comparison(DEFAULT_SPLIT, facts)
        inc = cc["by_responsibility"]["增量分析"]
        cases.append((f"⑥f 读法 B/A 前台流量比 = {inc['frontend_ratio_direct_over_pool']:.3f}"
                      f" ∈ (0,1)（直接 90/10 更省前台流量，但违反层内 1:1）",
                      0 < inc["frontend_ratio_direct_over_pool"] < 1))
        sens = sensitivity(DEFAULT_SPLIT, facts, ASSUMPTION_SET["geo_cluster_cv_grid"])
        cases.append((f"⑥g 敏感性网格 {len(sens['rows'])} 行 · 每行天数随流量单调不增",
                      len(sens["rows"]) >= 10 and all(
                          list(r["days_at_grid"].values()) == sorted(r["days_at_grid"].values(), reverse=True)
                          for r in sens["rows"])))

        # ---- ⑦ 判据能失败：篡改分流设计
        bad = json.loads(json.dumps(DEFAULT_SPLIT))
        for L in bad["layers"]:
            if L["id"] == "L-USER":
                L["pool_allocation"] = {"treatment": 1.5, "control": 1.5}
        j = judge_design(bad, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.30)
        cases.append(("⑦ 篡改①：池内比例和=3.0 ⇒ J1 判红", not j["judgements"]["J1_split_self_consistent"]["pass"]))

        bad2 = json.loads(json.dumps(DEFAULT_SPLIT))
        for L in bad2["layers"]:
            if L["id"] == "L-USER":
                L["idempotency"]["key_fields"] = ["visitor_id"]
                L["idempotency"]["salt"] = ""
        j2 = judge_design(bad2, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.30)
        cases.append(("⑦b 篡改②：幂等键去 salt ⇒ J2 判红", not j2["judgements"]["J2_idempotency"]["pass"]))

        bad3 = json.loads(json.dumps(DEFAULT_SPLIT))
        for L in bad3["layers"]:
            if L["id"] == "L-USER":
                L["unit_level"] = "session"
        j3 = judge_design(bad3, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.30)
        cases.append(("⑦c 篡改③：把用户级效应挂到会话级分流 ⇒ J3 判红（方法学）",
                      not j3["judgements"]["J3_layer_match"]["pass"]))

        bad4 = json.loads(json.dumps(DEFAULT_SPLIT))
        bad4["layers"] = [L for L in bad4["layers"] if L["id"] != "L-GEO"]
        j4 = judge_design(bad4, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.30)
        cases.append(("⑦d 篡改④：删掉 L-GEO 层 ⇒ J3 判红（不得默认落到 L-USER）",
                      not j4["judgements"]["J3_layer_match"]["pass"]))

        # ---- ⑧ 声明 == 计算（J8 两边都能抓）
        good_traffic = {"source": "夹具（构造样本，非实测）", "observed_at": "2026-09-13", "window_days": 90,
                        "daily_eligible_units": {"L-USER": 18000, "L-GEO": 4, "L-SESSION": 40000,
                                                 "daily_content_impressions": 60000},
                        "geo_cluster_cv": 0.30,
                        "baseline": {"site_conversion_rate": 0.02, "content_ctr": 0.02,
                                     "source": "夹具（构造样本，非实测）"}}
        j8a = judge_design(DEFAULT_SPLIT, facts, good_traffic,
                           dict(good_traffic["baseline"]), 0.30)
        routes_ok = all(r["declared_routing"] == r["computed_routing"]
                        for r in j8a["responsibilities"] if r["declared_routing"])
        cases.append(("⑧ 声明路由 == 计算路由（正常路径）", routes_ok))
        inc = next(r for r in j8a["responsibilities"] if r["responsibility"] == "增量分析")
        cases.append(("⑧b 篡改样本：18,000/日 ⇒ 增量分析所需天数 > 90 ⇒ 判准实验",
                      inc["verdict"] == "USE_QUASI_EXPERIMENT" and inc["days_to_power"] > 90))
        low_traffic = json.loads(json.dumps(good_traffic))
        low_traffic["daily_eligible_units"]["L-USER"] = 40000
        j8b = judge_design(DEFAULT_SPLIT, facts, low_traffic, dict(low_traffic["baseline"]), 0.30)
        inc2 = next(r for r in j8b["responsibilities"] if r["responsibility"] == "增量分析")
        cases.append(("⑧c 反向控制：40,000/日 ⇒ 增量分析转为可随机化（判据不是恒红）",
                      inc2["verdict"] == "RANDOMIZED_OK" and inc2["days_to_power"] <= 90))

        # ---- ⑧d J8 两边都能抓
        flipsplit2 = json.loads(json.dumps(DEFAULT_SPLIT))
        flipsplit2["routing_declarations"]["增量分析"] = {"if_feasible": "randomized_holdback",
                                                       "if_infeasible": "randomized_holdback"}
        jf = judge_design(flipsplit2, facts, good_traffic, dict(good_traffic["baseline"]), 0.30)
        cases.append(("⑧d 篡改样本：把增量分析的不可行分支也声明成随机化 ⇒ J8 判红",
                      not jf["judgements"]["J8_route_match"]["pass"]))
        delsplit = json.loads(json.dumps(DEFAULT_SPLIT))
        del delsplit["routing_declarations"]["内容实验"]
        jd = judge_design(delsplit, facts, good_traffic, dict(good_traffic["baseline"]), 0.30)
        cases.append(("⑧e 篡改样本：删掉内容实验的声明 ⇒ J8 判红（未声明即未标定）",
                      not jd["judgements"]["J8_route_match"]["pass"]))

        # ---- ⑨ J9 geo
        j9bad = judge_design(DEFAULT_SPLIT, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.30)
        j9good = judge_design(DEFAULT_SPLIT, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.05)
        cases.append(("⑨ CV=0.30、声明准实验 ⇒ J9 一致性通过（数值 84% 作为发现登记）",
                      j9bad["judgements"]["J9_geo_consistency"]["pass"]
                      and j9bad["judgements"]["J9_geo_consistency"]["computed_routing"] == ROUTE_QUASI))
        cases.append(("⑨b CV=0.05、声明可随机化 ⇒ J9 一致性通过（判据不是恒红）",
                      j9good["judgements"]["J9_geo_consistency"]["pass"]
                      and j9good["judgements"]["J9_geo_consistency"]["computed_routing"] == ROUTE_RANDOMIZED))
        flipsplit = json.loads(json.dumps(DEFAULT_SPLIT))
        flipsplit["routing_declarations"]["广告实验"] = {"if_feasible": "quasi_experiment",
                                                       "if_infeasible": "randomized_holdback"}
        j9flip = judge_design(flipsplit, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.05)
        cases.append(("⑨c 篡改样本：把 geo 两分支对调 ⇒ J9 判红（声明与物理不符）",
                      not j9flip["judgements"]["J9_geo_consistency"]["pass"]))
        nosplit = json.loads(json.dumps(DEFAULT_SPLIT))
        del nosplit["routing_declarations"]["广告实验"]
        j9none = judge_design(nosplit, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.05)
        cases.append(("⑨d 篡改样本：删掉声明 ⇒ J9 判红（未声明即未标定）",
                      not j9none["judgements"]["J9_geo_consistency"]["pass"]))

        # ---- ⑩ J11 停止条件
        j11 = judge_design(DEFAULT_SPLIT, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.30)
        cases.append(("⑩ 停止条件全部可判据（含数值阈值 + 序贯边界）",
                      j11["judgements"]["J11_stopping_falsifiable"]["pass"]))
        _save = STOPPING_RULES[0]["criterion"]
        STOPPING_RULES[0]["criterion"] = "效果明显改善时就停止"
        j11b = judge_design(DEFAULT_SPLIT, facts, None, dict(ASSUMPTION_SET["baseline_default"]), 0.30)
        STOPPING_RULES[0]["criterion"] = _save
        cases.append(("⑩b 篡改样本：判据改成「效果明显改善时停止」⇒ J11 判红",
                      not j11b["judgements"]["J11_stopping_falsifiable"]["pass"]))

        # ---- ⑪ 端到端：真 CLI + 夹具（subprocess，不 import 库函数）
        #  (a) 无 --traffic ⇒ exit 2
        r = _run_cli(["--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪a 端到端：不给流量 ⇒ exit 2（实得 {r.returncode}）", r.returncode == 2))
        #  (b) --assume ⇒ 仍 exit 2（假设不是实测）
        r = _run_cli(["--assume", "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪b 端到端：--assume ⇒ 仍 exit 2（实得 {r.returncode}）", r.returncode == 2))
        cases.append(("⑪b2 --assume 的 stdout 必须出现「该条用准实验」",
                      "该条用准实验" in r.stdout))
        #  (c) 流量文件 source 为空 ⇒ exit 2，且不得是 0
        noSrc = tmp / "traffic-nosource.json"
        t = json.loads(json.dumps(good_traffic))
        t["source"] = ""
        noSrc.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
        r = _run_cli(["--traffic", str(noSrc), "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪c 端到端：流量无来源 ⇒ exit 2 且 ≠ 0（实得 {r.returncode}）",
                      r.returncode == 2))
        #  (d) 带来源夹具 + 18,000/日 ⇒ 可跑通、增量分析判准实验 ⇒ exit 0（判据全过）
        tf = tmp / "traffic-good.json"
        tf.write_text(json.dumps(good_traffic, ensure_ascii=False), encoding="utf-8")
        jo = tmp / "artifact.json"
        r = _run_cli(["--traffic", str(tf), "--json-out", str(jo),
                      "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪d 端到端：带来源夹具 ⇒ exit 0（实得 {r.returncode}）", r.returncode == 0))
        cases.append(("⑪d2 端到端产物落盘且含 6 条责任判定",
                      jo.is_file() and len(json.loads(jo.read_text())["responsibilities"]) == 6))
        #  (e) 篡改流量：把 source 留着但把数值改到极小 ⇒ 判定应翻面
        t2 = json.loads(json.dumps(good_traffic))
        t2["daily_eligible_units"]["L-USER"] = 100
        tf2 = tmp / "traffic-tiny.json"
        tf2.write_text(json.dumps(t2, ensure_ascii=False), encoding="utf-8")
        r = _run_cli(["--traffic", str(tf2), "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪e 端到端篡改：100/日 ⇒ 三条随机化责任全判准实验（exit {r.returncode}）",
                      r.returncode == 0 and "增量分析" in r.stdout and r.stdout.count("USE_QUASI_EXPERIMENT") >= 3))
        #  (f) 契约目录不存在 ⇒ exit 2
        r = _run_cli(["--traffic", str(tf), "--contracts-dir", str(tmp / "nope")])
        cases.append((f"⑪f 端到端：契约目录不存在 ⇒ exit 2（实得 {r.returncode}）", r.returncode == 2))
        #  (g) **判红路径**：设计夹具把「增量分析」的不可行分支声明成随机化
        #      ⇒ 真 CLI 必须 exit 1（证明判红能走通，不是只在库里能红）
        flipdesign = json.loads(json.dumps(DEFAULT_SPLIT))
        flipdesign["routing_declarations"]["增量分析"] = {"if_feasible": "randomized_holdback",
                                                       "if_infeasible": "randomized_holdback"}
        df = tmp / "design-flip.json"
        df.write_text(json.dumps(flipdesign, ensure_ascii=False), encoding="utf-8")
        r = _run_cli(["--traffic", str(tf), "--design", str(df), "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪g 端到端判红：夹具把不可行分支声明成随机化 ⇒ exit 1（实得 {r.returncode}）",
                      r.returncode == 1))
        cases.append(("⑪g2 判红输出必须点名 J8 与具体责任", "J8_route_match" in r.stdout and "增量分析" in r.stdout))
        #  (h) 设计文件里层被改坏（删掉 salt）⇒ 真 CLI exit 1
        badsalt = json.loads(json.dumps(DEFAULT_SPLIT))
        for L in badsalt["layers"]:
            if L["id"] == "L-USER":
                L["idempotency"]["key_fields"] = ["visitor_id"]
        df2 = tmp / "design-nosalt.json"
        df2.write_text(json.dumps(badsalt, ensure_ascii=False), encoding="utf-8")
        r = _run_cli(["--traffic", str(tf), "--design", str(df2), "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪h 端到端判红：夹具删掉幂等 salt ⇒ exit 1（实得 {r.returncode}）", r.returncode == 1))

        # ---- ⑪i --check：一致 ⇒ 0 ／ 缺产物 ⇒ 2 ／ 产物陈旧 ⇒ 1
        art = tmp / "artifact-check.json"
        r0 = _run_cli(["--assume", "--json-out", str(art), "--contracts-dir", str(CONTRACTS_DIR)])
        r1 = _run_cli(["--check", "--assume", "--json-out", str(art), "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪i 端到端 --check：产物与现算一致 ⇒ 0（实得 {r1.returncode}）", r1.returncode == 0))
        r2 = _run_cli(["--check", "--assume", "--json-out", str(tmp / "missing.json"),
                       "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪j 端到端 --check：产物不存在 ⇒ 2 且 ≠ 0/3（实得 {r2.returncode}）", r2.returncode == 2))
        stale = json.loads(art.read_text(encoding="utf-8"))
        for rr in stale["responsibilities"]:
            if rr["responsibility"] == "增量分析":
                rr["n_star_daily_units"] = 12345
        art2 = tmp / "artifact-stale.json"
        art2.write_text(json.dumps(stale, ensure_ascii=False), encoding="utf-8")
        r3 = _run_cli(["--check", "--assume", "--json-out", str(art2), "--contracts-dir", str(CONTRACTS_DIR)])
        cases.append((f"⑪k 端到端 --check：篡改产物里的 N* ⇒ 1（实得 {r3.returncode}）", r3.returncode == 1))

        # ---- ⑫ 变异测试：把关键判据改坏，自检必须抓到
        mut = mutate_check(tmp)
        cases.append((f"⑫ 变异测试 {mut['caught']}/{mut['total']} 抓住", mut["caught"] == mut["total"]))
        for name, ok, note in mut["detail"]:
            cases.append((f"⑫·{name} ⇒ {note}", ok))

        # ---- ⑬ 自检不留临时文件
        after = sorted(os.listdir(scripts_dir))
        cases.append(("⑬ --selftest 未在 scripts/ 下留文件", before == after))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 78)
    print("holdback_power.py --selftest")
    print("=" * 78)
    npass = 0
    for name, ok in cases:
        print(f"  {'✅' if ok else '🔴'} {name}")
        npass += bool(ok)
    print("-" * 78)
    print(f"{npass}/{len(cases)} 用例通过")
    if npass != len(cases):
        print("❌ 自检未全过")
        return EXIT_RED
    print("✅ 自检全过")
    return EXIT_OK


# ---------------------------------------------------------------- 变异测试
MUTATIONS = [
    ("M1 路由比对放水", '        if d != c:\n            j8.append(',
     '        if False:\n            j8.append('),
    ("M2 层匹配判据放水", '        eff, rnd = spec["inference_level"], L["unit_level"]',
     '        eff, rnd = L["unit_level"], L["unit_level"]'),
    ("M3 已知答案区间放宽", "    J5_LOW, J5_HIGH = 29000, 34000",
     "    J5_LOW, J5_HIGH = 0, 10**9"),
    ("M4 幂等键判据放水", '        if L["unit_field"] not in kf:',
     '        if False:'),
    ("M5 流量来源判据放水", '    if not src:\n        return t, "流量文件 `source` 为空',
     '    if False:\n        return t, "流量文件 `source` 为空'),
    ("M6 覆盖率判据放水", '    if len(seen) != 6 or len(set(seen)) != 6:',
     '    if False:'),
]


def mutate_check(tmp: Path) -> dict:
    """把本脚本的关键判据改坏，跑变异体的 --selftest，确认非零退出。

    两条纪律（照 run_phase6_gates.py 的教训）：
      ① 先证明变异**改变了真实取值**（锚点唯一 + 变异体字节确实不同）；
      ② 变异体必须被**真跑**（subprocess 指向变异体文件本身，不指向原文件）。
    """
    src_path = Path(__file__).resolve()
    src = src_path.read_text(encoding="utf-8")
    # ⚠️ 锚点字符串本身也写在 MUTATIONS 表里 ⇒ 必须在**表之前**的正文里计数，
    #    否则每个锚点都命中 2 次、变异全部「施不上力」（首版实测即如此）。
    head = src.split("# ---------------------------------------------------------------- 变异测试")[0]
    detail, caught = [], 0
    for name, old, new in MUTATIONS:
        n = head.count(old)
        if n != 1:
            detail.append((name, False, f"锚点命中 {n} 次（应为 1）—— 变异没施上力"))
            continue
        mutant_src = src.replace(old, new, 1)
        if mutant_src == src:
            detail.append((name, False, "变异体与原文件逐字节相同"))
            continue
        mdir = tmp / f"mut-{name.split()[0]}"
        mdir.mkdir(parents=True, exist_ok=True)
        mfile = mdir / src_path.name
        mfile.write_text(mutant_src, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(mfile), "--selftest"],
                               capture_output=True, text=True, timeout=600)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        ok = rc != 0
        detail.append((name, ok, f"变异体 --selftest 退出码 {rc}"))
        caught += ok
    return {"total": len(MUTATIONS), "caught": caught, "detail": detail}


def mutate_main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="holdback-mutate-"))
    try:
        res = mutate_check(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("=" * 78)
    print("holdback_power.py --mutate（把判据改坏，看自检报不报）")
    print("=" * 78)
    for name, ok, note in res["detail"]:
        print(f"  {'✅' if ok else '🔴'} {name} —— {note}")
    print("-" * 78)
    print(f"{res['caught']}/{res['total']} 变异被抓住")
    return EXIT_OK if res["caught"] == res["total"] else EXIT_RED


# ================================================================ CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="独立站 holdback 样本量/功效计算器与判据门禁")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--check", action="store_true", help="核对磁盘产物 vs 现算")
    ap.add_argument("--design", default=None, help="分流参数 JSON（缺省用内置默认）")
    ap.add_argument("--traffic", default=None,
                    help="独立站流量实测 JSON；必须含非空 source，否则 exit 2")
    ap.add_argument("--assume", action="store_true",
                    help="显式进入假设网格模式（来源为空，仍为 exit 2）")
    ap.add_argument("--contracts-dir", default=str(CONTRACTS_DIR))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.mutate:
        return mutate_main()

    try:
        if args.check:
            if not args.json_out:
                args.json_out = str(DEFAULT_ARTIFACT)
            if not Path(args.json_out).is_file():
                print(f"❌ 机读产物不存在：{args.json_out} —— 没拿到输入 ≠ 通过（exit 2）", file=sys.stderr)
                return EXIT_NOINPUT
            tmp = Path(tempfile.mkdtemp(prefix="holdback-check-"))
            fresh = tmp / "fresh.json"
            try:
                argv = ["--json-out", str(fresh), "--contracts-dir", args.contracts_dir]
                if args.traffic:
                    argv += ["--traffic", args.traffic]
                if args.assume:
                    argv += ["--assume"]
                if args.design:
                    argv += ["--design", args.design]
                r = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv],
                                   capture_output=True, text=True, timeout=300)
                if not fresh.is_file():
                    print(f"❌ --check 内部错误：现算未产出文件（rc={r.returncode}）\n{r.stderr}",
                          file=sys.stderr)
                    return EXIT_INTERNAL
                old = json.loads(Path(args.json_out).read_text(encoding="utf-8"))
                new = json.loads(fresh.read_text(encoding="utf-8"))
                diffs = []
                oldc = {c["contract_id"]: c for c in old.get("provenance", {}).get("contracts", [])}
                newc = {c["contract_id"]: c for c in new["provenance"]["contracts"]}
                for cid in sorted(set(oldc) | set(newc)):
                    if oldc.get(cid, {}).get("sha256") != newc.get(cid, {}).get("sha256"):
                        diffs.append(f"{cid} 契约内容指纹变了（产物陈旧）")
                oldr = {r_["responsibility"]: r_ for r_ in old.get("responsibilities", [])}
                newr = {r_["responsibility"]: r_ for r_ in new["responsibilities"]}
                for nm in sorted(set(oldr) | set(newr)):
                    a, b = oldr.get(nm, {}), newr.get(nm, {})
                    for f in ("verdict", "computed_routing", "n_star_daily_units"):
                        if a.get(f) != b.get(f):
                            diffs.append(f"{nm}.{f}：产物 {a.get(f)!r} ≠ 现算 {b.get(f)!r}")
                print(f"核对率：契约 {len(newc)}/{len(newc)} · 责任 {len(newr)}/{len(newr)}")
                if len(newr) != 6:
                    print(f"🔴 责任条数 {len(newr)} ≠ 6（覆盖率不足不是通过）")
                    return EXIT_NOINPUT
                if diffs:
                    print(f"🔴 产物与现算不一致（{len(diffs)} 条）：")
                    for d in diffs:
                        print("   -", d)
                    return EXIT_RED
                print("✅ 产物与现算一致")
                return EXIT_OK
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return run(args)
    except Exception:
        print("❌ 脚本内部错误（≠ 判红）：", file=sys.stderr)
        traceback.print_exc()
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
