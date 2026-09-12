#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""candidate_filter.py — 三段式检索过滤（关键词库 v2 的可执行部分）

背景：v1（`关键词库.md`）只有正向词，实测污染严重 —— 04 域（供应链）27% 命中
「软件供应链 / SBOM / 包依赖」，14 域（用户分析）64% 命中医学影像与流行病学队列研究。
v2 因此引入负向词与约束词。

为什么过滤器是**独立脚本**而不是塞进 arXiv 查询串：
arXiv API 的 `search_query` 虽支持 ANDNOT，但把负向词写进查询串会让结果随
分词与转义细节漂移，出现「今天空集、明天有结果」的不可复现现象。
本脚本改为「收割阶段只捞全 → 过滤阶段本地判定」，规则可复现、可审计、可回归。

三段式语义：
    保留 = 命中正向词(由收割阶段保证) 且 未命中该域负向词 且 满足该域约束词

⚠️ 设计原则：**负向词按域生效**，因为同一个词在不同域含义相反：
    `trial`   —— 01/02 域是方法论词（randomized controlled trial），14 域是医学词
    `cohort`  —— 14 域是「同期群」，医学里是「队列研究」
    `ad`      —— 13 域是广告，医学里是 Alzheimer/ADHD

用法：
    python3 candidate_filter.py --report          # 只看污染统计，不写文件
    python3 candidate_filter.py --apply           # 过滤并写 arxiv_candidates_filtered.json
    python3 candidate_filter.py --selftest        # 自检：证明过滤器真的能拦下污染
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# 域 → 领域目录名（与 query_groups 里的中文域标签对应）
# ---------------------------------------------------------------------------
DOMAIN_DIR = {
    "01-因果推断": "01-因果推断", "02-A_B实验": "02-A_B实验", "03-时间序列": "03-时间序列",
    "04-供应链": "04-供应链", "05-推荐系统": "05-推荐系统", "06-增长模型": "06-增长模型",
    "07-NLP-VOC": "07-NLP-VOC", "08-知识图谱": "08-知识图谱",
    "09-DataAgent-LLM": "09-DataAgent-LLM", "10-MAS": "10-MAS", "11-AI人文": "11-AI人文",
    "12-ML基础": "12-ML基础", "13-广告分析": "13-广告分析", "14-用户分析": "14-用户分析",
    "15-营销投放分析": "15-营销投放分析", "16-智能体工程": "16-智能体工程",
}

# ---------------------------------------------------------------------------
# 全局负向词（所有域一律生效）
# ---------------------------------------------------------------------------
GLOBAL_NEGATIVE = [
    # 软件供应链（04 域重灾区）
    "software supply chain", "sbom", "software bill of materials",
    "package detection", "dependency vulnerability", "model lineage", "model card",
    # 纯基础设施
    "kv cache", "serving throughput", "gpu kernel",
    # 医学（全局收敛，各域另有更严的清单）
    "clinical", "patient", "lesion", "mri", "eeg", "ecg", "fmri",
    "radiotherapy", "histopathology", "tumor",
]

# ---------------------------------------------------------------------------
# 按域负向词（**只在指定域生效**，理由见模块 docstring）
# ---------------------------------------------------------------------------
DOMAIN_NEGATIVE = {
    "01-因果推断": ["clinical", "patient", "drug", "gene", "epidemiolog", "biomarker"],
    # ⚠️ 02 域刻意**不**含 "trial"：RCT 是方法论词
    "02-A_B实验": ["wet lab", "robot", "in vitro"],
    "03-时间序列": ["weather", "climate", "traffic flow", "wind power",
                    "load forecasting", "eeg", "ecg", "seismic"],
    "04-供应链": ["software supply chain", "sbom", "npm", "pypi", "malware",
                  "vulnerability", "repository", "compiler"],
    "05-推荐系统": ["movie", "music recommendation", "news recommendation",
                    "poi recommendation", "friend recommendation"],
    "06-增长模型": ["employee turnover", "student dropout", "attrition rate"],
    "07-NLP-VOC": ["clinical note", "hate speech", "fake news", "machine translation"],
    "08-知识图谱": ["molecular", "protein", "drug discovery", "biomedical",
                    "chemistry", "crystal"],
    "09-DataAgent-LLM": ["code generation benchmark", "gui agent", "web agent"],
    "10-MAS": ["multi-agent reinforcement learning"],
    "12-ML基础": ["speech recognition", "image classification", "object detection"],
    "13-广告分析": ["adversarial attack", "adhd", "alzheimer", "adversarial example"],
    "14-用户分析": ["clinical", "patient", "mri", "eeg", "fmri", "lesion",
                    "radiotherapy", "pet scan", "tumor", "cohort study",
                    "epidemiolog", "diagnosis"],
    "15-营销投放分析": ["health promotion", "academic promotion", "faculty"],
    "16-智能体工程": ["embodied agent", "robot manipulation", "gui agent"],
}

# ---------------------------------------------------------------------------
# 按域约束词：宽词必须与约束词之一共现才保留
# ---------------------------------------------------------------------------
DOMAIN_CONSTRAINT = {
    "01-因果推断": ["e-commerce", "retail", "advertis", "marketing", "pricing",
                    "subscription", "customer", "business"],
    "02-A_B实验": ["online", "platform", "marketplace", "advertis", "pricing",
                   "user", "e-commerce"],
    "03-时间序列": ["demand", "sales", "retail", "inventory", "e-commerce", "supply chain"],
    "04-供应链": ["goods", "physical", "retail", "warehouse", "fulfillment",
                  "inventory", "logistics", "order"],
    "05-推荐系统": ["e-commerce", "product", "item", "user", "session", "catalog"],
    "06-增长模型": ["subscription", "e-commerce", "retail", "customer", "platform"],
    "07-NLP-VOC": ["e-commerce", "product", "customer", "review", "feedback"],
    "08-知识图谱": ["e-commerce", "product", "customer", "agent", "retrieval"],
    "09-DataAgent-LLM": ["analytics", "business", "enterprise", "database", "table"],
    "12-ML基础": ["tabular", "business", "prediction", "dataset"],
    "13-广告分析": ["advertis", "marketing", "campaign", "budget", "roas"],
    "14-用户分析": ["e-commerce", "retail", "subscription", "churn", "user", "customer"],
    "15-营销投放分析": ["retail", "e-commerce", "sales", "price", "promotion"],
    "16-智能体工程": ["llm", "agent", "tool", "context"],
}

# 强制要求摘要里出现电商信号的「重灾区」域：只靠负向词不足以分开
STRICT_ECOMMERCE_DOMAINS = {"14-用户分析", "04-供应链"}


def _blob(item: dict) -> tuple[str, str]:
    """返回 (标题小写, 标题+摘要小写)。标题命中负向词 = 硬丢弃。"""
    t = (item.get("title") or "").lower()
    a = (item.get("abstract") or "").lower()
    c = (item.get("comment") or "").lower()
    j = (item.get("journal_ref") or "").lower()
    return t, f"{t} . {a} . {c} . {j}"


def judge(item: dict, domain: str) -> tuple[bool, str]:
    """判定一篇论文在给定域下是保留还是丢弃。返回 (keep, reason)。"""
    title, blob = _blob(item)

    # ① 标题里的全局/本域负向词 → 硬丢弃（标题比摘要可信）
    for neg in GLOBAL_NEGATIVE + DOMAIN_NEGATIVE.get(domain, []):
        if neg in title:
            return False, f"标题否定词:{neg}"

    # ② 摘要里的本域负向词 → 丢弃
    for neg in DOMAIN_NEGATIVE.get(domain, []):
        if neg in blob:
            return False, f"摘要否定词:{neg}"

    # ③ 约束词：本域若配置了约束词，必须命中其一
    cons = DOMAIN_CONSTRAINT.get(domain, [])
    if cons and not any(c in blob for c in cons):
        return False, "未命中约束词"

    # ④ 重灾区域额外要求电商强信号（负向词挡不住同词异义）
    if domain in STRICT_ECOMMERCE_DOMAINS:
        strong = ("e-commerce", "ecommerce", "retail", "marketplace", "warehouse",
                  "fulfillment", "inventory", "subscription", "customer", "purchase")
        if not any(s in blob for s in strong):
            return False, "重灾区域缺少电商强信号"

    return True, "ok"


def filter_pool(items: list[dict]) -> tuple[list[dict], Counter, Counter]:
    """对每篇论文，只要**存在任一**命中域判定为保留，就保留该论文。

    返回 (保留列表, 丢弃原因计数, 各域保留数)
    """
    kept, reasons, per_domain = [], Counter(), Counter()
    for it in items:
        groups = it.get("query_groups") or []
        domains = [g for g in groups if g in DOMAIN_DIR]
        if not domains:
            kept.append(it)          # 不属于任何已知域，不擅自丢
            continue
        verdicts = [judge(it, d) for d in domains]
        keep = any(v[0] for v in verdicts)
        if keep:
            kept.append(it)
            for d, v in zip(domains, verdicts):
                if v[0]:
                    per_domain[d] += 1
        else:
            reasons[verdicts[0][1]] += 1
    return kept, reasons, per_domain


def selftest() -> int:
    """证明过滤器真的能拦下 v2 记录的两类真实污染。"""
    cases = [
        # (域, 构造的论文, 期望保留?)
        ("04-供应链", {"title": "A Survey of Software Supply Chain Security",
                       "abstract": "we study sbom and package dependency vulnerability in npm"},
         False),
        ("04-供应链", {"title": "Multi-Echelon Inventory Optimization for Perishable Goods",
                       "abstract": "we optimize warehouse replenishment and fulfillment for retail goods"},
         True),
        ("14-用户分析", {"title": "Cohort Study of MRI Lesion Progression",
                       "abstract": "clinical patient cohort study with eeg and radiotherapy"},
         False),
        ("14-用户分析", {"title": "Cohort Retention Analysis for Subscription Commerce",
                       "abstract": "we analyze customer retention and repeat purchase in e-commerce"},
         True),
        # 同词异义：trial 在 02 域是方法论词，不得被误杀
        ("02-A_B实验", {"title": "Sequential Testing for Online Controlled Trials",
                        "abstract": "we design always-valid inference for platform pricing experiments"},
         True),
        # 约束词缺失 → 丢弃（气象预测冒充需求预测）
        ("03-时间序列", {"title": "Transformer Forecasting of Wind Power Generation",
                       "abstract": "we improve weather forecasting accuracy"},
         False),
    ]
    ok = True
    for domain, item, expect in cases:
        got, why = judge(item, domain)
        flag = "✅" if got == expect else "❌"
        if got != expect:
            ok = False
        print(f"  {flag} [{domain}] keep={got} (期望 {expect})  {why}  ← {item['title'][:52]}")
    print("✅ 自检通过：负向词与约束词均按预期生效" if ok
          else "❌ 自检失败：过滤逻辑与关键词库 v2 的约定不符")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="三段式检索过滤（关键词库 v2）")
    ap.add_argument("--pool", type=Path, default=DATA / "arxiv_candidates.json")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not args.pool.is_file():
        print(f"找不到候选池 {args.pool}")
        return 1
    src = json.loads(args.pool.read_text(encoding="utf-8"))
    items = src.get("items", src if isinstance(src, list) else [])
    if not items:
        print("候选池为空")
        return 1

    kept, reasons, per_domain = filter_pool(items)
    n = len(items)
    print(f"候选池 {n} 篇 → 保留 {len(kept)} 篇（丢弃 {n - len(kept)}，{100*(n-len(kept))/n:.1f}%）")
    print("\n丢弃原因 Top:")
    for r, c in reasons.most_common(12):
        print(f"  {c:>4}  {r}")
    print("\n各域保留数 Top:")
    for d, c in per_domain.most_common(20):
        print(f"  {c:>4}  {d}")

    if args.apply:
        out = DATA / "arxiv_candidates_filtered.json"
        out.write_text(json.dumps(
            {"generated_at": src.get("generated_at"), "window": src.get("window"),
             "source_pool": str(args.pool.name), "count_before": n,
             "count": len(kept), "items": kept},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
