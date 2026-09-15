#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 papers_registry.json —— paper2skills 的唯一事实源（初版）

数据来源：
  - data/recommendations.json      arXiv 1046 篇候选的评分结果
  - data/journal_candidates.json   Crossref 1751 篇期刊候选
  - 本轮四个领域子评审 + 顶刊子评审的 P0/P1 结论（人工录入 DECISIONS）

产出：
  paper2skills-vault/07-资源库/papers_registry.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper2skills-research" / "data"
OUT = ROOT / "paper2skills-vault" / "07-资源库" / "papers_registry.json"

# 领域 → code 目录
CODE_DIR = {
    "01-因果推断": "causal_inference", "02-A_B实验": "ab_testing", "03-时间序列": "time_series",
    "04-供应链": "supply_chain", "05-推荐系统": "recommendation", "06-增长模型": "growth_model",
    "07-NLP-VOC": "nlp_voc", "08-知识图谱": "knowledge_graph", "09-DataAgent-LLM": "data_agent_llm",
    "10-MAS": "mas", "11-AI人文": "ai_humanities", "12-ML基础": "ml_fundamentals",
    "13-广告分析": "advertising", "14-用户分析": "user_analytics", "15-营销投放分析": "marketing",
    "16-智能体工程": "llm_agent_engineering", "00-电商Agent": "ecommerce_agent",
}


# ---------------------------------------------------------------------------
# 域名守卫（PHASE6 P1）：本文件用到的每个域标签都必须是**规范名**
# ---------------------------------------------------------------------------
# 本仓库实测过一次「同一件事 7 处各写一份名字」造成的静默失效：
# `arxiv_harvest` 写旧名、`candidate_filter` 写新名 ⇒ 1046 篇候选里 141 篇
# 的负向词与约束词**一条都没生效**（且没有任何读数）。此处照 `arxiv_harvest.py`
# 的办法，在**自己这一侧** fail loud：认不出的名字当场炸，不留给下游去猜。
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import domains as _domains  # noqa: E402


def _assert_canonical(labels, where: str) -> None:
    bad = sorted(set(labels) - set(_domains.CANONICAL))
    if bad:
        hint = {b: _domains.RETIRED[b] for b in bad if b in _domains.RETIRED}
        raise SystemExit(
            f"🔴 {where} 里出现非规范域名：{bad}\n"
            + (f"   其中这些是**已退休的旧名**，规范名是：{hint}\n" if hint else "")
            + f"   规范名以 `domains.py` 的 CANONICAL 为准（{len(_domains.CANONICAL)} 个）。")

_assert_canonical(CODE_DIR, "build_registry.CODE_DIR")

# ⚠️⚠️ **DECISIONS 的守卫必须写在 DECISIONS 定义之后。**
# 它一度写在这一行（第 58 行），而 `DECISIONS` 定义在第 62 行 ——
# 模块级前向引用 ⇒ **`NameError` at import ⇒ 本脚本从被加守卫的那天起就无法运行**
# （台账 #113：加于 P1 `4dfa9f2`，此后每一次「跑一下 build_registry」都会当场炸）。
# 它没被发现，是因为**没有任何门禁跑过它**，而 `papers_registry.json` 是**已入库的产物**，
# 所有下游读的是产物不是生成器 —— 与 #105（过滤器无消费者）同型：
# **「交付」不是「接线」，而「接线」也不是「跑得起来」。**
# ⇒ 结构性修法见 `run_phase6_gates.py` 的 **L19e**：流水线脚本必须能被 `import`
#   （import 会执行模块级代码，正是这条会炸的地方）。
#
# 本轮经原文核对（arXiv API / Crossref）后的决策表
# 字段: arxiv_id, domain, priority, decision, reason, data_availability, venue_verified
DECISIONS: list[dict] = [
    # ---------- P0：本轮最强推荐 ----------
    dict(a="2606.26690", d="13-广告分析", p="P0", dec="extract", tier="workshop",
         reason="归因蚕食校正：用增量实验当因果锚点校正归因，直击'站内品牌词+站外种草'归因虚高。TikTok 多市场部署，蚕食率降约 15pp",
         data="available", note="ADKDD 2026（KDD workshop，非主会），须标注"),
    dict(a="2608.11675", d="13-广告分析", p="P0", dec="extract", tier="top",
         reason="FunnelCausalNet：GMV=转化×客单 双头 uplift + 多档券分配 + 拉格朗日预算分配器，直接对应券额度决策",
         data="partial", note="CIKM 2026；假设 RCT，平台卖家需退化为观察数据"),
    dict(a="2608.10182", d="13-广告分析", p="P0", dec="extract", tier="preprint",
         reason="从预测转向增量：因果网络+贝叶斯 bandit+对偶 LP 做全局约束下的受限分配，线上主指标 +7.20%",
         data="available", note="无 venue 标注；无公开代码，需自研"),
    dict(a="2608.18174", d="06-增长模型", p="P0", dec="extract", tier="preprint",
         reason="季节性误报修正：相邻窗口标签把季节当衰退，生产系统 1/3 行动清单是误报；含复现包，难度⭐1",
         data="need_2y", note="需 ≥2 个完整年度月度面板；出海不足 2 年需用品类季节性先验替代"),
    dict(a="2607.16230", d="03-时间序列", p="P0", dec="extract", tier="preprint",
         reason="RouteCost：运费成本预估四段式（需求预测→费率卡→残差→装箱合并推断），直连毛利率与定价",
         data="available", note="无 venue；摘要未说明代码"),
    dict(a="2608.25871", d="03-时间序列", p="P0", dec="extract", tier="top",
         reason="CEDAR：决策条件化仿真，可回答'给定预算排期与备货计划的销量'，KDD 2026 正式发表并用于真实预算规划",
         data="available", note="KDD 2026 proceedings 已核实（journal_ref 完整）"),
    dict(a="2607.12714", d="05-推荐系统", p="P0", dec="extract", tier="top",
         reason="SAM：建模'购买=意图终止'+从购买历史学个性化补货节奏，专治购后冗余推荐，即耗材复购召回",
         data="available", note="SIGIR 2026 proceedings 已核实"),
    dict(a="2606.29366", d="04-供应链", p="P0", dec="extract", tier="preprint",
         reason="ORLA：求解器验证的 MIP 建模生成与选择，做多仓库存分配；支持自然语言约束（'优先保 FBA 不断货'）",
         data="available", note="与 2607.25956 高度重叠，建议合并为一张卡"),
    dict(a="2607.09745", d="04-供应链", p="P0", dec="extract", tier="second",
         reason="SupplyNetPy：开源 Python 库，多级网络离散事件仿真，支持易腐库存/节点中断/随机提前期；落地成本最低",
         data="available", note="Winter Simulation Conference 2026；**有开源实现**"),
    dict(a="2607.09608", d="14-用户分析", p="P0", dec="extract", tier="preprint",
         reason="被助攻的乌龙球：站外种草引发增量但被下游 marketplace 记账，ROAS 系统性失真；跨境最贵的测量缺口",
         data="conditional", note="需受众级随机化：自建独立站可行，纯平台站内卖家须降级为诊断框架"),
    dict(a="2608.20844", d="00-电商Agent", p="P0", dec="extract", tier="preprint",
         reason="TRACE：Scout/Judge 双 agent + 多源证据接地做目录属性补全，生产四个垂类曝光加权覆盖率 +90.4%，结账转化 +0.48%",
         data="available", note="无 venue；数字来自完整摘要，萃取前须回原文核对"),
    dict(a="2607.22518", d="05-推荐系统", p="P0", dec="extract", tier="top",
         reason="PinEqualizer：Pinterest 全漏斗内容探索与去偏的**生产系统**，新品冷启动的评估骨架",
         data="partial", note="KDD 2026 已核实；平台侧系统，我方只能借'全漏斗+去偏+短长期度量'结构"),
    dict(a="2609.01038", d="02-A_B实验", p="P0", dec="extract", tier="top",
         reason="人格条件 A/B 仿真：40 个真实 A/B 上方向准确率 0.75-0.90，用于实验**事前粗筛**（母婴流量小、周期长）",
         data="available", note="EMNLP 2026 Industry Track 已核实；只能当粗筛器，不能替代实验"),
    dict(a="2608.27006", d="00-电商Agent", p="P0", dec="extract", tier="top",
         reason="自刷新检索的会话式推荐：逐项 hash 做目录增量对账，LLM 只做意图分类，检索/重排走专用函数",
         data="available", note="RecSys 2026，**3 页短文（Demo）**，标注时勿写成完整方法论文"),
    dict(a="2609.04540", d="12-ML基础", p="P0", dec="extract", tier="preprint",
         reason="Mitra-v2：合成数据预训练的表格基础模型，**本周可试**；选品打分表/广告报表/库存表的免调参基线",
         data="available", note="模型权重与代码在 HuggingFace（autogluon/mitra-finetune）"),
    dict(a="2608.26263", d="16-智能体工程", p="P0", dec="extract", tier="top",
         reason="SKILL.state：不可变 Skill 规格 + 可变结构化执行状态，prompt 不随步数增长 → Skill 运行时该做状态机而非 chat loop",
         data="available", note="摘要数字截断，须回原文核实压缩比"),
    dict(a="2608.22152", d="10-MAS", p="P0", dec="extract", tier="top",
         reason="Collaboration Tax：32 任务×11 模型量化'多 Agent 协作税'与四阶段失败级联 → **什么时候不该拆多 Agent**",
         data="available", note="EMNLP 2026 主会；现有 16 张卡全在讲'怎么搭'，缺'该不该搭'"),
    dict(a="2608.25277", d="10-MAS", p="P0", dec="extract", tier="top",
         reason="Routed Graph Handoff：NL 交接吞 40-60% token，155-token router 选类型化依赖图；τ-retail +12.7pp @3.2× 压缩",
         data="available", note="EMNLP 2026；须配 graph-aware executor prompt，否则失效"),
    dict(a="2607.22115", d="09-DataAgent-LLM", p="P0", dec="extract", tier="second",
         reason="RBAC Text-to-SQL 基准：度量越权率与过度拒答率 → 数据 Agent 上线门禁（客服看本店/运营看全店）",
         data="available", note="SIGMOD 2027 已录用（未召开），标注时写'已录用'"),
    # ---------- P1 ----------
    dict(a="2609.08725", d="02-A_B实验", p="P1", dec="extract", tier="workshop",
         reason="BAFF：RTB 的 A/B 中实验组与对照组共享日志会污染训练数据；给出 (k,l) 过滤器族与三阶段测量协议",
         data="available", note="OARS Workshop @RecSys 2026 poster，**非主会**"),
    dict(a="2608.25871x", d="", p="", dec="", tier="", reason="", data="", note=""),  # 占位保持结构
    dict(a="2607.24779", d="13-广告分析", p="P1", dec="extract", tier="top",
         reason="HOBA：三层竞价智能体（LLM 调超参/SARSA 选专家/专家池执行），在线学习只在'选哪个专家'上，探索风险可控",
         data="available", note="KDD 2026 Ads Track，线上目标成本 +3.6%"),
    dict(a="2608.20844b", d="", p="", dec="", tier="", reason="", data="", note=""),
    dict(a="2608.19735", d="05-推荐系统", p="P1", dec="extract", tier="preprint",
         reason="RecPFN：合成点击流预训练的先验拟合网络，推理零权重更新；**代码公开**（SAP-samples/tabular-ai-recpfn）",
         data="available", note="venue 标注可疑（仅页数），须复核"),
    dict(a="2608.10240", d="05-推荐系统", p="P1", dec="extract", tier="top",
         reason="Sequential Modality Dropout：四行代码让推荐模型在缺图/缺文案下仍可用；95% 单品缺失率下 HR@10 保留 61% vs 22%",
         data="available", note="CIKM 2026；摘要含代码链接，摘要截断导致误判为无代码"),
    dict(a="2609.04862", d="14-用户分析", p="P1", dec="extract", tier="top",
         reason="PTDG：按商品特性动态调整多任务漏斗依赖强度，治深层稀疏目标（复购/订阅）的信号侵蚀",
         data="available", note="CIKM 2026 已核实"),
    dict(a="2608.18849", d="12-ML基础", p="P1", dec="extract", tier="preprint",
         reason="GEAR：两阶段蒸馏把表格基础模型压成 CPU 可跑的 MLP/树，解决'效果好但线上跑不动'",
         data="available", note="与 2608.10837 TACTICL 同类，建议合并"),
    dict(a="2608.09162", d="12-ML基础", p="P1", dec="extract", tier="preprint",
         reason="Tabular Numeric Stretch Transformation：把数值特征变换形式化为优化问题，优于手调 log/box-cox",
         data="available", note="增强现有 Skill-Feature-Engineering"),
    dict(a="2606.31474", d="12-ML基础", p="P1", dec="watch", tier="workshop",
         reason="TabPATE：表格 ICL 的成员推断攻击与 DP 防御 → 客户数据进上下文的合规红线（GDPR/CCPA）",
         data="available", note="ICML 2026 FMSD workshop，非主会"),
    dict(a="2608.01708", d="16-智能体工程", p="P1", dec="extract", tier="top",
         reason="PGMem：把画像信号绑到支持/推翻它的事件，检索按证据效度排序 → 母婴 1-3 年跨周期画像的证据链",
         data="available", note="EMNLP 2026 主会"),
    dict(a="2608.29128", d="16-智能体工程", p="P1", dec="watch", tier="workshop",
         reason="APIFlow-Bench：长链路依赖 API 的 7 项工程能力分解 + mock 金丝雀溯源 → 上线验收模板",
         data="available", note="NeurIPS 2026 IAEval workshop **在投**，不可当已发表引用"),
    dict(a="2608.28978", d="16-智能体工程", p="P1", dec="extract", tier="top",
         reason="**负结果**（性价比最高的一页纸）：知识图谱记忆在 LongMemEval 上 token F1 0.417 < 平铺向量 0.468，劝退早期投图记忆",
         data="available", note="作为 Skill-GraphRAG 与 Agentic-Memory-Management 的'边界/反例'小节"),
    dict(a="2609.01519", d="10-MAS", p="P1", dec="watch", tier="workshop",
         reason="构念效度审计：多轮 agent 仿真的护栏效应从 +87.4 缩到 +7.2/−13.9，生成残差占变异 49.9% → agent 仿真的可信边界",
         data="available", note="Trust-AI-Eval workshop **投稿未录用**，仅可作方法学引述"),
    dict(a="2608.00558", d="10-MAS", p="P1", dec="extract", tier="preprint",
         reason="AiFlow：token 原生反应式编排 + 有界背压（Node Guardian 强制队列上界/取消传播/重试）→ 自建多 Agent 运行时的契约",
         data="available", note="有公开实现（GitHub ModelEngine-G…）"),
    dict(a="2608.22830", d="09-DataAgent-LLM", p="P1", dec="extract", tier="workshop",
         reason="Beyond the Harness：从生产 SQL 反解 query-DAG，把老师傅的历史查询蒸馏为可复用上下文 → 跨境口径（GMV 是否含退款）沉淀",
         data="available", note="COLM 2026 workshop；与 BIRD-History 同题，建议合并"),
    dict(a="2608.00426", d="10-MAS", p="P1", dec="extract", tier="preprint",
         reason="MAPLE-Guard：一条毒记忆可写入一次、反复检索、晋升为共享记忆并污染未见攻击的 agent → 记忆安全基线",
         data="available", note="27 页，无 venue"),
    dict(a="2607.28956", d="00-电商Agent", p="P1", dec="watch", tier="preprint",
         reason="MerchantBench：365 天订单级卖家侧仿真（98,843 条真实商品）→ 自研选品/运营 Agent 的外部评测床",
         data="unavailable", note="benchmark 非方法；底层数据不公开，只能用其环境与协议"),
    dict(a="2608.08621", d="00-电商Agent", p="P1", dec="watch", tier="preprint",
         reason="Business Arena：唯一明确'跨境店铺'的 Agent 沙盒（基于真实 Alibaba.com 采购数据）",
         data="unavailable", note="benchmark；数据不公开，借评测协议"),
    dict(a="2609.11190", d="00-电商Agent", p="P1", dec="watch", tier="second",
         reason="Agentic Share-of-Search：把'品牌在 AI 购物助手答案里的可见度份额'做成可测新 KPI（39% 恢复信号，CI 30.0-48.8%）",
         data="conditional", note="DSI 年会非正式 venue；作者自述为可行性原型，先当观测不当 KPI"),
    # ---------- 顶刊（Crossref 路线，需读全文定级） ----------
    dict(a="10.1287/mnsc.2024.03192", d="13-广告分析", p="P1", dec="watch", tier="top",
         reason="Algorithmic Pricing, Price Wars, and Tacit Collusion: Evidence from E-Commerce（Management Science 2026-07-09）— 算法定价的合谋风险，跨境多店铺调价的合规红线",
         data="unavailable", note="顶刊独占主题（arXiv 不会覆盖）；需订阅读全文"),
    dict(a="10.1287/mksc.2024.0296", d="15-营销投放分析", p="P1", dec="watch", tier="top",
         reason="Large Language Models for Market Research: A Data-Augmentation Approach（Marketing Science 2026-07）— LLM 做市场研究的统计保证，可用于选品前的需求验证",
         data="available", note="顶刊独占；需读全文确认可否复现"),
    dict(a="10.1287/isre.2023.0456", d="15-营销投放分析", p="P1", dec="watch", tier="top",
         reason="Impact of the Invisibles: Personalized Pricing on Platform with Anonymous Users（ISR 2026-08-24）— 个性化定价的隐私与反噬",
         data="partial", note="顶刊独占；DOI 需核对"),
    dict(a="10.1287/mnsc.2023.05123", d="04-供应链", p="P1", dec="watch", tier="top",
         reason="Learning in Lost-Sales Inventory Systems with Stochastic Lead Times and Random Supply（Management Science 2026-09-03）— 缺货即丢单 + 随机交期，正是跨境备货的真实结构",
         data="available", note="顶刊独占；与 2609.08071（LLM 库存策略）互补"),
    dict(a="10.1287/mnsc.2024.01034", d="04-供应链", p="P1", dec="watch", tier="top",
         reason="From Trees to Treewidth: Inventory Management in Complex Supply Chain Networks（Management Science 2026-09-10）— 复杂网络结构的库存管理可计算性",
         data="partial", note="顶刊独占；理论性较强，需读全文判断可落地性"),
    dict(a="10.1287/mnsc.2023.04987", d="02-A_B实验", p="P1", dec="watch", tier="top",
         reason="Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments（Management Science 2026-07）— 组合实验（多元素同时变）的因果估计，对应创意/主图/文案组合爆炸",
         data="available", note="顶刊独占；与 2609.01038 事件预筛互补"),
    dict(a="10.1287/mnsc.2024.00567", d="02-A_B实验", p="P1", dec="watch", tier="top",
         reason="Fast Selection from Multiple Treatments: A Sequential Method（ISR 2026-08-27）— 多处理序贯选择，直接对应'多档券/多版本素材'的快速择优",
         data="available", note="顶刊独占；需读全文"),
    dict(a="10.1287/mnsc.2023.04712", d="05-推荐系统", p="P2", dec="watch", tier="top",
         reason="Scalable Bundle Recommendations: A Large-Scale Field Experiment（Management Science 2026-07-15）— 组合推荐的大规模田野实验，对应配件搭售",
         data="available", note="顶刊独占；实测证据强"),
]

PLACEHOLDER = {"2608.25871x", "2608.20844b"}

# 域名守卫的**第二条**：`DECISIONS` 与 `PLACEHOLDER` 都已定义完毕，这里才是它合法的位置。
#
# ⚠️ 两条判据，缺一它就有后门：
#   J-a  **真正会被使用的**条目（不在 PLACEHOLDER 里的）域名必须是规范名；
#   J-b  域名为空的行**必须恰好是**已声明的 PLACEHOLDER。
# 少了 J-b，`d=""` 就是一个可以随手扩大的豁免面（「先留空，回头再填」永远不会被抓）；
# 少了 J-a，空域名会一路混进事实源 —— 那正是 P1 要拦的「认不出的名字静默通过」。
# ⚠️ 原来那条守卫把 **J-a 的适用范围写成了整张表**：表里两条占位行 `d=""`，
# 于是即使修好前向引用，脚本**仍然跑不起来**（本批实测：修完 NameError 立刻撞上这条）。
# 两条缺陷叠在一起 —— 前向引用让脚本根本到不了这里，所以第二条**从来没有被人看见过**。
_used = [d for d in DECISIONS if d["a"] not in PLACEHOLDER]
_assert_canonical([d["d"] for d in _used], "build_registry.DECISIONS 的 `d` 字段")
_empty = {d["a"] for d in DECISIONS if not d["d"]}
if _empty != PLACEHOLDER:
    raise SystemExit(
        "🔴 `DECISIONS` 里「域名为空」的行与已声明的 `PLACEHOLDER` 不是同一批：\n"
        f"   域名为空：{sorted(_empty)}\n"
        f"   PLACEHOLDER：{sorted(PLACEHOLDER)}\n"
        "   ⇒ 空域名是一个**豁免面**，它必须与显式声明的占位集合逐项相等，"
        "不许顺手扩大（漏洞 #11 同族：判据的适用范围被默认成了全体）。")



def load_scoring_set() -> dict:
    """打分集（**已被三段式过滤器过滤**，PHASE6 #105 接线后）。

    ⚠️ 与 `load_full_pool()` 分开是**必须的**，不是冗余：打分集回答「要不要选它」，
    全量池回答「它是什么」。把两者合并会让「过滤器丢了它」变成「它不存在」——
    而 `DECISIONS` 是**人的决策记录**，不该被一个筛选器作废（详见 `enrich()`）。
    """
    p = DATA / "recommendations.json"
    if not p.is_file():
        raise SystemExit(f"🔴 打分集读不到：{p} —— 先跑 rank_candidates.py（它读过滤产物）")
    return {i["arxiv_id"]: i for i in json.loads(p.read_text(encoding="utf-8"))["items"]}


def load_full_pool() -> dict:
    """全量候选池（未过滤）。**只用于补充元数据，不用于入选判定。**"""
    p = DATA / "arxiv_candidates.json"
    if not p.is_file():
        raise SystemExit(f"🔴 全量候选池读不到：{p}")
    return {i["arxiv_id"]: i for i in json.loads(p.read_text(encoding="utf-8"))["items"]}


def enrich(a: str, scored: dict, pool: dict) -> tuple[dict, str]:
    """给一条已决策论文取元数据。返回 (src, provenance)。

    provenance 三态（**这是一等输出，不是调试信息**）：
      `scored`     —— 在打分集里（正常路径）；
      `pool_only`  —— **被过滤器丢了，但论文确实存在** ⇒ 用全量池补元数据。
                      这一态是本条存在的**全部理由**：`papers_registry.json` 是
                      唯一事实源，一篇已 `decision: extract` 的论文不该因为
                      一个**筛选器**没选它，就在事实源里退化成 `(待补：见 note)`、
                      `url` 变空、`venue` 变空、`score` 变 `None`。
                      ⚠️ 首版（#105 接线前）这里是 `(src or {}).get(...)` —— 三态被
                      压成一态，**降级是静默的**。
      `missing`    —— 两处都找不到 ⇒ 下面的 `main()` 必须**点名报出**，不许静默填占位符。
    """
    if a in scored:
        return scored[a], "scored"
    if a in pool:
        # 全量池没有 `venue`/`score`（那是打分层算出来的）⇒ 用**同一个打分器**补，
        # 而不是在这里另写一份口径（漏洞 #11 同族：同一件事两处各写一份）。
        import rank_candidates as _rc
        return _rc.score(pool[a]), "pool_only"
    return {}, "missing"


def build_registry() -> tuple[dict, dict]:
    """**纯计算**：造出 registry 对象并返回，不碰盘。写盘由 `main()` 决定。"""
    scored = load_scoring_set()
    pool = load_full_pool()
    jrn = {i["doi"]: i for i in json.loads((DATA / "journal_candidates.json").read_text(encoding="utf-8"))["items"]}

    provenance = {"scored": [], "pool_only": [], "missing": []}
    records, n = [], 0
    for d in DECISIONS:
        if d["a"] in PLACEHOLDER:
            continue
        n += 1
        pid = f"p2s-2026-{n:04d}"
        is_journal = d["a"].startswith("10.")
        if is_journal:
            src, prov = jrn.get(d["a"]), ("scored" if d["a"] in jrn else "missing")
        else:
            src, prov = enrich(d["a"], scored, pool)
        provenance[prov].append({"a": d["a"], "paper_id": pid, "domain": d["d"],
                                 "decision": d["dec"]})
        src = src or {}
        rec_item = {
            "paper_id": pid,
            "identifiers": ({"doi": d["a"]} if is_journal else {"arxiv": d["a"]}),
            "title": src.get("title", "(待补：见 note)"),
            "url": src.get("url", ""),
            "published": (src.get("published", "") or "")[:10] if not is_journal else src.get("published", ""),
            "journal": src.get("journal", "") or "" if is_journal else "",
            "venue": src.get("venue", "") or src.get("journal", "") or "",
            "venue_tier": d["tier"],
            "domain": d["d"],
            "priority": d["p"],
            "decision": d["dec"],
            "decision_reason": d["reason"],
            "data_availability": d["data"],
            "note": d["note"],
            "score": src.get("score", None),
            "source_route": "crossref" if is_journal else "arxiv",
            "outputs": {
                "skill_card": f"paper2skills-vault/{d['d']}/Skill-<方法名>.md" if d["d"] else "",
                "code_dir": f"paper2skills-code/{CODE_DIR.get(d['d'], '?')}/<algo>" if d["d"] else "",
                "evidence": f"paper2skills-vault/papers/{d['d']}/{pid}/evidence.md" if d["d"] else "",
            },
            "gates": {"code": "pending", "evidence": "pending", "business": "pending"},
            "status": "shortlisted" if d["dec"] in ("extract", "watch") else "new",
        }
        records.append(rec_item)

    registry = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": {"start": "2026-06-12", "end": "2026-09-12"},
        "sources": {
            # ⚠️ `candidates` 与 `scored` **是两个数**，接线（#105）之后不再相等：
            #    candidates = 收割到的全量池；scored = 过了三段式过滤器、真的进打分的那些。
            # 写死一个数会让「过滤器到底有没有生效」在事实源里看不见。
            "arxiv": {"queries": 49, "candidates": len(pool), "scored": len(scored)},
            "crossref": {"journals": 28, "candidates": 1751, "biz_hits": 783},
        },
        "stats": {
            "total": len(records),
            "extract": sum(1 for r in records if r["decision"] == "extract"),
            "watch": sum(1 for r in records if r["decision"] == "watch"),
            "by_priority": {p: sum(1 for r in records if r["priority"] == p) for p in ("P0", "P1", "P2")},
        },
        "records": records,
    }
    print(json.dumps(registry["stats"], ensure_ascii=False, indent=2))

    # --- 元数据来源账（**一等输出**） ----------------------------------------
    # 「谁被过滤器丢了」必须看得见。否则接线（#105）的后果是：
    # 274 篇候选**静默**离开打分集，而其中恰好有一篇已是 `decision: extract`
    # （p2s-2026-0007 / 2607.12714：收割查询把它挂在 04-供应链 下，它一条约束词都不命中，
    #  于是被丢；而人的决策把它放在 05-推荐系统）。没有这一栏，事实源里那一条会
    # 从「有 title/url/venue/score」**静默退化成** `(待补：见 note)`。
    print(f"\n元数据来源（scored {len(provenance['scored'])} / "
          f"pool_only {len(provenance['pool_only'])} / missing {len(provenance['missing'])}）：")
    for x in provenance["pool_only"]:
        print(f"  ⚠️ pool_only  {x['paper_id']}  {x['a']}  {x['domain']}  {x['decision']}"
              f"  ← **被三段式过滤器丢弃，但论文存在**：元数据取自全量池，"
              f"丢弃原因见 data/arxiv_candidates_filtered.json 的 filter.dropped_ids 反查")
    for x in provenance["missing"]:
        print(f"  🔴 missing    {x['paper_id']}  {x['a']}  {x['domain']}  {x['decision']}"
              f"  ← 打分集与全量池**都没有它** ⇒ 事实源里这一条会退化成占位符")
    if provenance["missing"]:
        print(f"\n🔴 {len(provenance['missing'])} 条决策论文在候选池里查无此篇 —— "
              f"「查不到」不等于「没影响」：请先核实标识符，别让占位符入库。")
    if provenance["pool_only"]:
        print("\n⚠️ 上列条目**不在打分集里**（过滤器未选它），元数据走全量池补齐。"
              "\n   这不是错误，但它意味着：**过滤器的判定与人的决策在这些条目上不一致** ——"
              "\n   逐条记在册，别静默。")
    print("\nP0 萃取队列:")
    for r in records:
        if r["priority"] == "P0":
            print(f"  {r['paper_id']} [{r['domain']}] {r['title'][:64]}")

    return registry, provenance


# ---------------------------------------------------------------------------
# ⚠️⚠️ 覆盖守卫（台账 #114）：**本脚本不是 `papers_registry.json` 的生成器**
# ---------------------------------------------------------------------------
# 盘上那份是**唯一事实源**，PHASE3–PHASE5 期间由人逐条补过：venue_tier 校正 31 条、
# 已交付卡的 `status/gates/outputs` 回指 19 条、`note` 口径修正 9 条、
# **DOI 更正 8 条**、R4 的 `venue_track` 降级 7 条…… 只有 **6/45** 条能被本脚本原样复现。
#
# 本脚本的真实身份是**一次性初始化器**：DECISIONS 停在「初版建档」那一刻的快照。
# ⇒ 默认**只做只读对账、绝不写盘**。写盘必须 `--force-reinit --reason "<理由>"`。
#
# 这一条之所以必须是默认行为，而不是一句文档警告：
# **「跑不起来」掩盖了「跑起来会毁数据」** —— #113 让本脚本在模块级就 `NameError`，
# 于是两天里没有任何人知道它一旦能跑会发生什么。第一版复核我还把它读成了
# 「生成器与盘上一致」（见 #115：脚本在写盘前就炸了，我比的其实是同一份文件）。
# 门禁的价值在于**把「如果它跑了会怎样」变成每次都能看见的一行读数**。
HUMAN_FIELDS = ("note", "status", "gates", "outputs", "venue_tier", "venue_track",
                "venue_track_note", "decision_reason", "data_availability",
                "title", "url", "journal", "venue", "published", "score", "identifiers")


def overwrite_account(fresh: dict) -> tuple[int, dict, list]:
    """盘上那份若被本次生成覆盖，会改动多少条记录、哪些字段。**纯计算，不写盘。**"""
    if not OUT.is_file():
        return 0, {}, []
    disk = json.loads(OUT.read_text(encoding="utf-8"))
    kd = {r["paper_id"]: r for r in disk.get("records", [])}
    kf = {r["paper_id"]: r for r in fresh["records"]}
    fields, changed = {}, []
    for pid in sorted(set(kd) & set(kf)):
        diff = [f for f in HUMAN_FIELDS if kd[pid].get(f) != kf[pid].get(f)]
        if diff:
            changed.append(pid)
            for f in diff:
                fields[f] = fields.get(f, 0) + 1
    return len(kf) - len(kd), fields, changed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="初版建档（**一次性初始化器**，默认只对账、不写盘）")
    ap.add_argument("--force-reinit", action="store_true",
                    help="真的覆盖 papers_registry.json（会毁掉人工核对过的字段，见覆盖账）")
    ap.add_argument("--reason", default="",
                    help="与 --force-reinit 同用：写清为什么可以覆盖（必填，留痕）")
    args = ap.parse_args()
    if args.force_reinit and not args.reason.strip():
        print("🔴 `--force-reinit` 必须与 `--reason \"<理由>\"` 同用 —— "
              "覆盖唯一事实源这件事必须留痕（三条扫描与推送纪律同款）。")
        return 2

    registry, provenance = build_registry()

    print(f"\n=== 覆盖账（本脚本 vs 盘上 {OUT.name}）===")
    delta, fields, changed = overwrite_account(registry)
    if not changed and delta == 0:
        print("盘上那份与本次生成逐字段相同 —— 覆盖是安全的。")
    else:
        same = 45 - len(changed) if len(changed) <= 45 else 0
        print(f"⚠️ 覆盖会改动 **{len(changed)}/45** 条记录（逐字段完全相同的只有 {same} 条），"
              f"逐字段受影响记录数：")
        for f, c in sorted(fields.items(), key=lambda x: -x[1]):
            print(f"     {c:>3}  {f}")
        print(f"   受影响 paper_id（全部）：{changed}")
        print("   ⇒ 这些字段是**人逐条核对过的**（venue_tier 校正 / 已交付卡回指 / "
              "note 口径修正 / DOI 更正 / R4 降级）。本脚本是初版快照，不是生成器。")

    if provenance["missing"]:
        print(f"\n🔴 {len(provenance['missing'])} 条决策论文在候选池里查无此篇 ⇒ "
              f"本脚本的输出**不能**直接入库（那 8 条会退化成占位符）。"
              f"先核实标识符。")
        return 1

    if not args.force_reinit:
        print(f"\n⏸️  **未写盘**（默认行为）。盘上 {OUT.name} 原样保留。")
        print("   要覆盖：`--force-reinit --reason \"<理由>\"`。"
              "先读完上面的覆盖账，再决定值不值。")
        return 0
    if changed:
        print(f"\n🔴 拒绝覆盖：本脚本会改动 {len(changed)}/45 条**人工核对过**的记录，"
              f"而 `--reason` 没有对它们逐类给出处置。\n"
              f"   `--reason` 收到的是：{args.reason!r}\n"
              f"   ⇒ 一次性初始化器**不允许**在事实源已经长出手工内容之后被重跑；"
              f"要把某条决策搬进事实源，就**手工改那一条**（并留下可复核的理由），"
              f"不要用整表重写去覆盖 45 条里的 39 条。")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已写 -> {OUT}（理由：{args.reason}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
