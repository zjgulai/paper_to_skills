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
_assert_canonical([d["d"] for d in DECISIONS], "build_registry.DECISIONS 的 `d` 字段")

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


def main() -> None:
    rec = {i["arxiv_id"]: i for i in json.loads((DATA / "recommendations.json").read_text(encoding="utf-8"))["items"]}
    jrn = {i["doi"]: i for i in json.loads((DATA / "journal_candidates.json").read_text(encoding="utf-8"))["items"]}

    records, n = [], 0
    for d in DECISIONS:
        if d["a"] in PLACEHOLDER:
            continue
        n += 1
        pid = f"p2s-2026-{n:04d}"
        is_journal = d["a"].startswith("10.")
        src = jrn.get(d["a"]) if is_journal else rec.get(d["a"])
        rec_item = {
            "paper_id": pid,
            "identifiers": ({"doi": d["a"]} if is_journal else {"arxiv": d["a"]}),
            "title": (src or {}).get("title", "(待补：见 note)"),
            "url": (src or {}).get("url", ""),
            "published": (src or {}).get("published", "")[:10] if not is_journal else (src or {}).get("published", ""),
            "journal": (src or {}).get("journal", "") if is_journal else "",
            "venue": (src or {}).get("venue", "") or (src or {}).get("journal", ""),
            "venue_tier": d["tier"],
            "domain": d["d"],
            "priority": d["p"],
            "decision": d["dec"],
            "decision_reason": d["reason"],
            "data_availability": d["data"],
            "note": d["note"],
            "score": (src or {}).get("score", None),
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
            "arxiv": {"queries": 49, "candidates": 1046},
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(registry["stats"], ensure_ascii=False, indent=2))
    print(f"\n-> {OUT}")
    print("\nP0 萃取队列:")
    for r in records:
        if r["priority"] == "P0":
            print(f"  {r['paper_id']} [{r['domain']}] {r['title'][:64]}")


if __name__ == "__main__":
    main()
