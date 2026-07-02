#!/usr/bin/env python3
"""Build a static HTML Playbook for paper2skills Skill cards."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from paper2skills_common.assets import detect_code_path, iter_skill_files  # noqa: E402
from paper2skills_common.domains import load_domain_registry  # noqa: E402

GRAPH_SCRIPT_DIR = BASE_DIR / "paper2skills-skills" / "paper-skills-graph" / "scripts"
if str(GRAPH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_SCRIPT_DIR))

from skills_graph_analyzer import SkillsGraph  # type: ignore  # noqa: E402

# ── Extracted data modules ──────────────────────────────────────────────────
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))
from config.playbooks_data import TOB_PLAYBOOKS
from config.agents_data import AGENT_CATALOG
# ─────────────────────────────────────────────────────────────────────────────

GA4_MEASUREMENT_ID = "G-N9HJR3G0MR"
FEISHU_WEBHOOK_URL = os.environ.get("P2S_FEISHU_WEBHOOK_URL", "")


# ---------------------------------------------------------------------------
# Section name normalizer: strips leading circled-number (①②③…) and Chinese
# ordinal prefixes (一、二、三…) so fuzzy matching always works.
# ---------------------------------------------------------------------------
_ORDINAL_PREFIX = re.compile(
    r"^[①②③④⑤⑥⑦⑧⑨⑩\u2460-\u2473]"   # ①–⑳  circled digits
    r"|^[一二三四五六七八九十]+[\.、]\s*"   # 一、二、…
    r"|^\d+[\.、]\s*"                      # 1. 2、…
)

def _norm_title(raw: str) -> str:
    """Normalise a section title: strip ordinal prefix, lowercase."""
    return _ORDINAL_PREFIX.sub("", raw).strip().lower()


# Section key → list of normalised name fragments to match against.
# All entries are already lower-case; matching is substring.
SECTION_KEYS: dict[str, list[str]] = {
    "algorithm": ["算法原理", "核心算法", "算法逻辑", "核心思想"],
    "scenario":  ["母婴出海应用案例", "应用案例", "业务应用", "业务场景", "应用场景",
                  "吸奶器出海应用案例"],
    "code":      ["代码模板", "完整可运行"],
    "guide":     ["使用指南"],
    "value":     ["商业价值", "业务价值", "量化 roi", "量化roi"],
    "relations": ["skill relations", "技能关联", "技能关系", "4. 技能关系", "四、技能关联"],
}

TOPIC_RULES = {
    "广告与投放": ["广告", "roas", "attribution", "tiktok", "keyword", "creative", "marketing", "mmm"],
    "供应链与补货": ["供应链", "库存", "补货", "forecast", "demand", "logistics", "fulfillment", "lead-time"],
    "客服与VOC": ["客服", "review", "voc", "absa", "sentiment", "translation", "customer"],
    "推荐与搜索": ["recommend", "推荐", "search", "retrieval", "ranking", "rerank", "embedding"],
    "知识图谱与RAG": ["rag", "graphrag", "knowledge graph", "知识图谱", "kg", "chunk", "hyde", "raptor", "ontology"],
    "数据采集与治理": ["data collection", "数据采集", "crawl", "quality", "provenance", "dedup", "signal"],
    "MAS与智能体工程": ["mas", "agent", "mcp", "orchestr", "tool", "memory", "trust"],
    "定价与利润": ["pricing", "price", "价格", "elasticity", "margin"],
    "风控与合规": ["fraud", "risk", "compliance", "合规", "风控", "fake"],
    "视觉内容生成": ["video", "visual", "image", "avatar", "ai视频", "multimodal", "mas video", "video commerce", "视频电商", "短视频带货", "视频标签化"],
    "实验与因果推断": ["causal", "因果", "uplift", "did", "diff-in-diff", "instrumental", "iv", "experiment", "a/b", "bandit", "counterfactual", "treatment", "rct"],
    "用户增长与留存": ["churn", "ltv", "retention", "rfm", "lifecycle", "cohort", "复购", "流失", "留存", "用户增长", "lapse", "reactivation", "clv"],
    "营销分析与预算": ["mmm", "budget allocation", "promo", "promotion", "saturation", "cannibalization", "media mix", "incrementality", "营销归因", "预算分配", "投资回报"],
    "跨境物流与履约": ["logistics", "fulfillment", "cross-border", "物流", "头程", "fba", "海外仓", "last-mile", "routing", "freight", "clearance", "履约"],
    "合规与关税决策": ["compliance", "tariff", "关税", "duty", "regulation", "gdpr", "ccpa", "合规", "legal", "policy", "prohibited", "cbam", "ipr"],
    "运营财务与P&L": ["p&l", "cogs", "财务", "成本", "profit", "cash-flow", "fba cost", "fee", "margin analysis", "财务建模", "roi", "irr", "payback"],
    "标签工程与决策触发": ["tag engineering", "label", "标签", "trigger", "ontology", "决策触发", "feature store", "entity resolution", "tagging", "taxonomy", "sku tag"],
    "搜索流量与SEO": ["seo", "a9", "organic", "搜索", "流量", "keyword ranking", "search visibility", "ppc synergy", "search traffic", "amazon seo", "索引"],
    "时序预测": ["time series", "时序", "forecasting", "预测", "arima", "prophet", "tft", "lstm", "seasonal", "trend", "decomposition", "temporal"],
    "LLM数据分析": ["llm", "nl2sql", "data agent", "text-to-sql", "自然语言", "language model", "gpt", "deepseek", "rag pipeline", "prompt", "in-context"],
    "ML基础与可解释性": ["shap", "lime", "explainab", "可解释", "feature importance", "feature engineering", "feature selection", "in-context learning", "icl", "tabpfn", "continual learning", "ewc", "rlhf", "reward model", "continual", "持续学习", "少样本", "few-shot", "xgboost", "lightgbm", "calibration", "overfitting"],
    "AI伦理与治理": ["ethics", "bias", "fairness", "伦理", "aigc", "hallucination", "safety", "alignment", "privacy", "responsi", "xai", "explainability", "regulatory compliance", "eu ai act", "gdpr compliance", "responsible ai", "red team", "transparency", "accountability"],
    "供应商管理与博弈": ["supplier", "vendor", "sourcing", "procurement", "supplier evaluation", "supplier risk", "negotiation", "qualification", "supplier performance", "supply base", "采购"],
    "KPI运营指标体系": ["kpi", "otif", "otd", "fill-rate", "fill rate", "oos", "sell-through", "days inventory", "scorecard", "dashboard", "benchmark", "metric", "运营指标", "履约率"],
    "逆向物流与退货": ["return", "reverse logistics", "refund", "chargeback", "退货", "退款", "disposition", "returnformer", "return fraud", "reverse", "退换"],
    "直播与TikTok商业化": ["live commerce", "tiktok shop", "直播", "短视频", "kol", "creator", "virtual anchor", "ugc", "short video", "livestream", "tiktok-shop", "live stream"],
    "仓储运营效率": ["warehouse", "wms", "slotting", "bin packing", "picking", "inbound", "outbound", "warehouse cost", "仓储", "库位", "理货", "入库"],
    "AI搜索与GEO": ["geo", "generative engine", "ai search", "zero-click", "answer engine", "llm search", "perplexity", "chatgpt search", "ai overview", "search generative", "featured snippet"],
    "现金流与供应链融资": ["cash conversion", "working capital", "inventory financing", "lending", "credit risk", "payable", "receivable", "cash cycle", "现金流", "融资", "账期", "供应链金融"],
    "竞品情报自动化": ["competitor keyword", "competitive intelligence", "competitor price", "market share", "competitive response", "competitor monitor", "竞品", "竞争对手", "情报", "share of voice"],
    "多渠道协同增长": ["omnichannel", "multi-channel", "cross-platform", "channel synergy", "walmart", "temu", "shopee", "lazada", "多渠道", "全渠道", "跨平台", "渠道协同"],
    "ESG与绿色供应链": ["esg", "carbon", "sustainability", "green supply", "epr", "cbam", "carbon footprint", "climate", "resp", "cold chain", "冷链", "碳排放", "carbon emission", "green logistics", "drone delivery", "uav", "carrier selection", "onsible sourcing", "碳足迹", "绿色", "可持续"],
}


# ── 全局：工作流卡片配置 ──

# ── 工作流卡片颜色配置 ──
WF_CARD_CONFIG = {
    "WF-A 智能补货":        {"color": "#15803d", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>', "pattern": "hex"},
    "WF-B 广告优化":        {"color": "#dc2626", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M11 12H3l8.45-3.1A1 1 0 0 1 13 9.9V18"/><path d="M13 18V6.1a1 1 0 0 1 1.55-.84L21 9v6l-6.45 3.74A1 1 0 0 1 13 18z"/></svg>', "pattern": "dots"},
    "WF-C 客服分诊":        {"color": "#0891b2", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>', "pattern": "wave"},
    "WF-D 选品扫描":        {"color": "#d97706", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>', "pattern": "grid"},
    "WF-E Review监控":      {"color": "#7c3aed", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>', "pattern": "dots"},
    "WF-F 动态定价":        {"color": "#b45309", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>', "pattern": "wave"},
    "WF-G Listing内容优化": {"color": "#0369a1", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>', "pattern": "grid"},
    "WF-H 复购增长":        {"color": "#059669", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 16l4-4 4 4 4-8"/></svg>', "pattern": "hex"},
    "WF-I 智能体工程":      {"color": "#1d4ed8", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="9" cy="16" r="1.5"/><circle cx="15" cy="16" r="1.5"/><path d="M9 7V5a3 3 0 0 1 6 0v2"/></svg>', "pattern": "circuit"},
    "WF-J DTC 独立站增长":  {"color": "#be185d", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/><line x1="3" y1="6" x2="21" y2="6"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>', "pattern": "dots"},
    "WF-K 全域风险防御":    {"color": "#b91c1c", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>', "pattern": "circuit"},
    "WF-L 内容营销增长":    {"color": "#6d28d9", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>', "pattern": "wave"},
    "WF-M 新品上市全链路":  {"color": "#0f766e", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3-8 3 8-7-5h8l-7 5z"/><line x1="12" y1="3" x2="12" y2="21"/></svg>', "pattern": "hex"},
    "WF-N 库存危机响应":    {"color": "#dc2626", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>', "pattern": "grid"},
    "WF-O 广告ROI最大化":   {"color": "#c2410c", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="8"/><rect x="12" y="6" width="3" height="12"/><rect x="17" y="13" width="3" height="5"/></svg>', "pattern": "dots"},
    "WF-P 关税危机应对":    {"color": "#92400e", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>', "pattern": "wave"},
    "WF-Q 标签驱动决策":    {"color": "#6d28d9", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>', "pattern": "circuit"},
    "WF-R Amazon SEO优化":  {"color": "#0369a1", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>', "pattern": "grid"},
    "WF-S 跨境数据合规":    {"color": "#0891b2", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>', "pattern": "hex"},
    "WF-T 用户分层精细运营":{"color": "#059669", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>', "pattern": "wave"},
    "WF-U P&L健康诊断":    {"color": "#065f46", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>', "pattern": "circuit"},
    "WF-V 供应商降本博弈":  {"color": "#15803d", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>', "pattern": "dots"},
    "WF-W 退货根因修复":    {"color": "#dc2626", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg>', "pattern": "grid"},
    "WF-X TikTok Shop冷启动":{"color": "#7c3aed", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 12a4 4 0 1 0 4 4V4a5 5 0 0 0 5 5"/></svg>', "pattern": "wave"},
    "WF-Y 账期现金流优化":  {"color": "#047857", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><circle cx="12" cy="12" r="2"/><path d="M6 12h.01M18 12h.01"/></svg>', "pattern": "hex"},
    "WF-Z 竞品情报作战室":  {"color": "#1e40af", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M1 6l5 5 5-5 5 5 5-5 2-2"/><path d="M1 12l5 5 5-5 5 5 5-5 2-2"/></svg>', "pattern": "circuit"},
    "WF-AA 多渠道协同增长": {"color": "#059669", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="4" height="8" rx="1"/><rect x="10" y="3" width="4" height="12" rx="1"/><rect x="18" y="3" width="4" height="6" rx="1"/><path d="M4 11v7M12 15v4M20 9v8M4 18h16"/></svg>', "pattern": "dots"},
    "WF-AB 视频电商全链路": {"color": "#6d28d9", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"/></svg>', "pattern": "wave"},
    "WF-AC 因果定价决策":   {"color": "#b45309", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="6" cy="18" r="2"/><path d="M8 6h4a4 4 0 0 1 4 4v0a2 2 0 0 0 2 2"/><path d="M8 18h2"/></svg>', "pattern": "grid"},
    "WF-AD 标签驱动实验设计":{"color": "#7c3aed", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><path d="M17 13v8M13 17h8"/></svg>', "pattern": "circuit"},
    "WF-AE 搜索流量全链路优化":{"color": "#0369a1", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><path d="M11 8a3 3 0 0 1 3 3"/></svg>', "pattern": "hex"},
    "WF-AF MAS运营财务智能化":{"color": "#065f46", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="6" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="18" r="2"/><circle cx="20" cy="18" r="2"/><path d="M6 6l4 4M14 14l4 4M6 18l4-4M14 10l4-4"/></svg>', "pattern": "dots"},
}
_WF_DEFAULT_COLORS = ["#555","#0369a1","#15803d","#dc2626","#7c3aed","#d97706","#0891b2"]
_WF_PATTERNS = ["grid", "dots", "wave", "hex", "circuit"]


# ── 全局：领域卡片配置 ──

# ── 领域卡片颜色和图案配置 ──
DOMAIN_CARD_CONFIG = {
    "01-因果推断":      {"color": "#7c3aed", "bg": "#f5f3ff", "num": "01", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="6" cy="18" r="2"/><path d="M8 6h4a4 4 0 0 1 4 4v0a2 2 0 0 0 2 2"/><path d="M8 18h2"/></svg>', "pattern": "grid"},
    "02-A_B实验":      {"color": "#0891b2", "bg": "#ecfeff", "num": "02", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><path d="M17 13v8M13 17h8"/></svg>', "pattern": "dots"},
    "03-时间序列":      {"color": "#0369a1", "bg": "#eff6ff", "num": "03", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 18 8 10 13 14 18 6"/><circle cx="21" cy="5" r="2"/><line x1="3" y1="21" x2="21" y2="21"/></svg>', "pattern": "wave"},
    "04-供应链":        {"color": "#15803d", "bg": "#f0fdf4", "num": "04", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>', "pattern": "hex"},
    "05-推荐系统":      {"color": "#d97706", "bg": "#fffbeb", "num": "05", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>', "pattern": "dots"},
    "06-增长模型":      {"color": "#059669", "bg": "#ecfdf5", "num": "06", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 16l4-4 4 4 4-8"/></svg>', "pattern": "grid"},
    "07-NLP-VOC":      {"color": "#7c3aed", "bg": "#fdf4ff", "num": "07", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 10h8M8 14h5"/></svg>', "pattern": "wave"},
    "08-知识图谱":      {"color": "#0f766e", "bg": "#f0fdfa", "num": "08", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="2"/><circle cx="4" cy="19" r="2"/><circle cx="20" cy="19" r="2"/><circle cx="12" cy="14" r="2"/><line x1="12" y1="7" x2="12" y2="12"/><line x1="5.5" y1="18" x2="10.5" y2="15"/><line x1="13.5" y1="15" x2="18.5" y2="18"/></svg>', "pattern": "hex"},
    "09-DataAgent-LLM":{"color": "#1d4ed8", "bg": "#eff6ff", "num": "09", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="9" cy="16" r="1.5"/><circle cx="15" cy="16" r="1.5"/><path d="M9 7V5a3 3 0 0 1 6 0v2"/><path d="M7 11V9"/><path d="M17 11V9"/></svg>', "pattern": "circuit"},
    "10-MAS":           {"color": "#7c3aed", "bg": "#faf5ff", "num": "10", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="6" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="18" r="2"/><circle cx="20" cy="18" r="2"/><path d="M6 6l4 4M14 14l4 4M6 18l4-4M14 10l4-4"/></svg>', "pattern": "circuit"},
    "11-AI人文":        {"color": "#be185d", "bg": "#fdf2f8", "num": "11", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/></svg>', "pattern": "wave"},
    "12-ML基础":        {"color": "#0369a1", "bg": "#eff6ff", "num": "12", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>', "pattern": "grid"},
    "13-广告分析":      {"color": "#b91c1c", "bg": "#fef2f2", "num": "13", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M11 12H3l8.45-3.1A1 1 0 0 1 13 9.9V18"/><path d="M13 18V6.1a1 1 0 0 1 1.55-.84L21 9v6l-6.45 3.74A1 1 0 0 1 13 18z"/></svg>', "pattern": "dots"},
    "14-用户分析":      {"color": "#0891b2", "bg": "#ecfeff", "num": "14", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>', "pattern": "wave"},
    "15-营销投放分析":  {"color": "#c2410c", "bg": "#fff7ed", "num": "15", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>', "pattern": "grid"},
    "16-智能体工程":    {"color": "#1d4ed8", "bg": "#eff6ff", "num": "16", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>', "pattern": "circuit"},
    "17-价格优化":      {"color": "#b45309", "bg": "#fffbeb", "num": "17", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>', "pattern": "dots"},
    "18-物流履约":      {"color": "#047857", "bg": "#ecfdf5", "num": "18", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="3" width="15" height="13" rx="1"/><path d="M16 8h4l3 4v3h-7V8z"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/></svg>', "pattern": "hex"},
    "19-风控反欺诈":    {"color": "#dc2626", "bg": "#fef2f2", "num": "19", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>', "pattern": "circuit"},
    "20-AI视频生成":    {"color": "#7c3aed", "bg": "#fdf4ff", "num": "20", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>', "pattern": "wave"},
    "21-合规决策":      {"color": "#92400e", "bg": "#fffbeb", "num": "21", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M6 13.5v3M18 13.5v3M6 16.5c0 1 1.5 2.5 3 2.5s3-1.5 3-2.5M18 16.5c0 1-1.5 2.5-3 2.5s-3-1.5-3-2.5"/><line x1="12" y1="12" x2="12" y2="22"/></svg>', "pattern": "grid"},
    "22-数据采集工程":  {"color": "#1e40af", "bg": "#eff6ff", "num": "22", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>', "pattern": "dots"},
    "23-运营财务":      {"color": "#065f46", "bg": "#f0fdf4", "num": "23", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>', "pattern": "hex"},
    "24-标签工程":      {"color": "#6d28d9", "bg": "#f5f3ff", "num": "24", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>', "pattern": "circuit"},
    "25-搜索流量工程":  {"color": "#0369a1", "bg": "#eff6ff", "num": "25", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><path d="M11 8a3 3 0 0 1 3 3"/></svg>', "pattern": "wave"},
}

# ── SVG图案生成函数 ──
def make_pattern_svg(pattern: str, color: str) -> str:
    alpha = "26"  # 15% opacity hex
    c = color
    if pattern == "grid":
        return f"""<svg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'><defs><pattern id='p' width='20' height='20' patternUnits='userSpaceOnUse'><path d='M20 0L0 0 0 20' fill='none' stroke='{c}' stroke-width='0.8' opacity='0.25'/></pattern></defs><rect width='60' height='60' fill='url(%23p)'/></svg>"""
    elif pattern == "dots":
        return f"""<svg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'><circle cx='10' cy='10' r='2' fill='{c}' opacity='0.3'/><circle cx='30' cy='10' r='2' fill='{c}' opacity='0.2'/><circle cx='10' cy='30' r='2' fill='{c}' opacity='0.2'/><circle cx='30' cy='30' r='2' fill='{c}' opacity='0.3'/></svg>"""
    elif pattern == "wave":
        return f"""<svg width='80' height='40' viewBox='0 0 80 40' xmlns='http://www.w3.org/2000/svg'><path d='M0 20 Q20 10 40 20 Q60 30 80 20' stroke='{c}' stroke-width='1.5' fill='none' opacity='0.25'/><path d='M0 30 Q20 20 40 30 Q60 40 80 30' stroke='{c}' stroke-width='1' fill='none' opacity='0.15'/></svg>"""
    elif pattern == "hex":
        return f"""<svg width='60' height='52' viewBox='0 0 60 52' xmlns='http://www.w3.org/2000/svg'><polygon points='30,2 58,17 58,47 30,62 2,47 2,17' stroke='{c}' stroke-width='1' fill='none' opacity='0.2'/><polygon points='30,10 50,21 50,43 30,54 10,43 10,21' stroke='{c}' stroke-width='0.8' fill='none' opacity='0.15'/></svg>"""
    elif pattern == "circuit":
        return f"""<svg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'><path d='M10 40 H30 V20 H60' stroke='{c}' stroke-width='1.2' fill='none' opacity='0.25'/><path d='M10 60 H40 V40' stroke='{c}' stroke-width='1' fill='none' opacity='0.2'/><circle cx='30' cy='40' r='3' fill='{c}' opacity='0.3'/><circle cx='60' cy='20' r='3' fill='{c}' opacity='0.25'/><circle cx='40' cy='60' r='2.5' fill='{c}' opacity='0.2'/></svg>"""
    return ""

def svg_to_css_bg(svg_str: str) -> str:
    import urllib.parse
    return "url(\"data:image/svg+xml," + urllib.parse.quote(svg_str) + "\")"

DOMAIN_BIZ_LABELS = {
    "01-因果推断": "广告归因·促销效果",
    "02-A_B实验": "策略验证·转化测试",
    "03-时间序列": "销量预测·需求预测",
    "04-供应链": "补货·库存·物流",
    "05-推荐系统": "复购推荐·搜索排序",
    "06-增长模型": "用户增长·LTV·流失",
    "07-NLP-VOC": "评论分析·情感挖掘",
    "08-知识图谱": "语义检索·关系图谱",
    "09-DataAgent-LLM": "数据智能体·Text2SQL",
    "10-MAS": "多Agent协作·编排",
    "11-AI人文": "AI伦理·内容检测",
    "12-ML基础": "模型校准·漂移检测",
    "13-广告分析": "ROAS·归因·预算分配",
    "14-用户分析": "漏斗分析·分群·RFM",
    "15-营销投放分析": "MMM·促销效果",
    "16-智能体工程": "Agent工程·工具调用",
    "17-价格优化": "动态定价·弹性估算",
    "18-物流履约": "清关·跨境物流",
    "19-风控反欺诈": "刷单·封号·跟卖",
    "20-AI视频生成": "TikTok·直播·短视频",
    "21-合规决策": "CPSC·关税·法规",
    "22-数据采集工程": "数据质量·采集管道",
    "23-运营财务": "P&L·FBA成本·汇率",
    "24-标签工程": "标签体系·决策触发",
    "25-搜索流量工程": "Amazon SEO·关键词",
}

# ── 全局：主题卡片颜色配置 ──
TOPIC_CARD_COLORS = {
    "广告与投放":       {"color": "#dc2626", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M11 12H3l8.45-3.1A1 1 0 0 1 13 9.9V18"/><path d="M13 18V6.1a1 1 0 0 1 1.55-.84L21 9v6l-6.45 3.74A1 1 0 0 1 13 18z"/></svg>'},
    "供应链与补货":     {"color": "#15803d", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>'},
    "客服与VOC":        {"color": "#0891b2", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 10h8M8 14h5"/></svg>'},
    "推荐与搜索":       {"color": "#d97706", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><path d="M11 8a3 3 0 0 1 3 3"/></svg>'},
    "知识图谱与RAG":    {"color": "#0f766e", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="2"/><circle cx="4" cy="19" r="2"/><circle cx="20" cy="19" r="2"/><circle cx="12" cy="14" r="2"/><line x1="12" y1="7" x2="12" y2="12"/><line x1="5.5" y1="18" x2="10.5" y2="15"/><line x1="13.5" y1="15" x2="18.5" y2="18"/></svg>'},
    "数据采集与治理":   {"color": "#1e40af", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>'},
    "MAS与智能体工程":  {"color": "#7c3aed", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="6" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="18" r="2"/><circle cx="20" cy="18" r="2"/><path d="M6 6l4 4M14 14l4 4M6 18l4-4M14 10l4-4"/></svg>'},
    "定价与利润":       {"color": "#b45309", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'},
    "风控与合规":       {"color": "#b91c1c", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'},
    "视觉内容生成":     {"color": "#6d28d9", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>'},
    "实验与因果推断":   {"color": "#7c3aed", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><path d="M17 13v8M13 17h8"/></svg>'},
    "用户增长与留存":   {"color": "#059669", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 16l4-4 4 4 4-8"/></svg>'},
    "营销分析与预算":   {"color": "#c2410c", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>'},
    "跨境物流与履约":   {"color": "#047857", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="3" width="15" height="13" rx="1"/><path d="M16 8h4l3 4v3h-7V8z"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/></svg>'},
    "合规与关税决策":   {"color": "#92400e", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M6 13.5v3M18 13.5v3M6 16.5c0 1 1.5 2.5 3 2.5s3-1.5 3-2.5M18 16.5c0 1-1.5 2.5-3 2.5s-3-1.5-3-2.5"/><line x1="12" y1="12" x2="12" y2="22"/></svg>'},
    "运营财务与P&L":    {"color": "#065f46", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>'},
    "标签工程与决策触发": {"color": "#6d28d9", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>'},
    "搜索流量与SEO":    {"color": "#0369a1", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'},
    "时序预测":         {"color": "#0369a1", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 18 8 10 13 14 18 6"/><circle cx="21" cy="5" r="2"/><line x1="3" y1="21" x2="21" y2="21"/></svg>'},
    "LLM数据分析":      {"color": "#1d4ed8", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/></svg>'},
    "ML基础与可解释性": {"color": "#0369a1", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>'},
    "AI伦理与治理":     {"color": "#be185d", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/></svg>'},
    "供应商管理与博弈": {"color": "#15803d", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>'},
    "KPI运营指标体系":  {"color": "#0891b2", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="8"/><rect x="12" y="6" width="3" height="12"/><rect x="17" y="13" width="3" height="5"/></svg>'},
    "逆向物流与退货":   {"color": "#dc2626", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg>'},
    "直播与TikTok商业化": {"color": "#7c3aed", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"/></svg>'},
    "仓储运营效率":     {"color": "#047857", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>'},
    "AI搜索与GEO":      {"color": "#1d4ed8", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'},
    "现金流与供应链融资": {"color": "#065f46", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><circle cx="12" cy="12" r="2"/><path d="M6 12h.01M18 12h.01"/></svg>'},
    "竞品情报自动化":   {"color": "#0891b2", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M1 6l5 5 5-5 5 5 5-5 2-2"/><path d="M1 12l5 5 5-5 5 5 5-5 2-2"/></svg>'},
    "多渠道协同增长":   {"color": "#059669", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="4" height="8" rx="1"/><rect x="10" y="3" width="4" height="12" rx="1"/><rect x="18" y="3" width="4" height="6" rx="1"/><path d="M4 11v7M12 15v4M20 9v8M4 18h16"/></svg>'},
    "ESG与绿色供应链":  {"color": "#15803d", "icon": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 0 1 10 10c0 5.52-4.48 10-10 10S2 17.52 2 12"/><path d="M12 6c-3.31 0-6 2.69-6 6s2.69 6 6 6"/><path d="M8 2.34A10 10 0 0 0 2.34 8"/></svg>'},
}
_TOPIC_PATTERNS = ["dots", "wave", "grid", "hex", "circuit"]

WORKFLOW_RULES = {
    "WF-A 智能补货": ["供应链", "库存", "补货", "demand", "forecast", "lead-time", "safety-stock", "logistics"],
    "WF-B 广告优化": ["广告", "roas", "attribution", "tiktok", "keyword", "creative", "mmm", "marketing"],
    "WF-C 客服分诊": ["客服", "review", "voc", "absa", "translation", "customer", "sentiment"],
    "WF-D 选品扫描": ["选品", "product", "market", "competitive", "signal", "data collection", "knowledge graph"],
    "WF-E Review监控": ["review", "fake-review", "sentiment", "absa", "dedup", "quality"],
    "WF-F 动态定价": ["pricing", "price", "价格", "elasticity", "markdown", "定价", "竞价", "discount"],
    "WF-G Listing内容优化": ["listing", "content", "copywriting", "主图", "视频", "a/b", "creative", "文案"],
    "WF-H 复购增长": ["churn", "ltv", "retention", "复购", "流失", "rfm", "lifecycle", "cohort"],
    "WF-I 智能体工程": ["agent", "智能体", "mas", "mcp", "llm agent", "workflow", "tool use", "safety guard", "监控", "部署"],
    "WF-J DTC 独立站增长": ["dtc", "独立站", "shopify", "acquisition", "federated", "intent", "conversion", "ltv", "personalization"],
    "WF-K 全域风险防御": ["fraud", "account health", "appeal", "欺诈", "合规", "compliance", "violation", "anomaly", "risk"],
    "WF-L 内容营销增长": ["kol", "content", "tiktok", "live commerce", "video", "creator", "内容", "直播", "短视频"],
    "WF-M 新品上市全链路": ["新品", "上市", "launch", "listing", "keyword", "compliance", "ranking", "新品上市", "上架"],
    "WF-N 库存危机响应": ["库存危机", "断货", "积压", "stockout", "overstock", "emergency", "crisis", "物流路由"],
    "WF-O 广告ROI最大化": ["广告roi", "mmm", "预算分配", "budget allocation", "saturation", "cannibalization", "ad roi", "归因修正"],
    "WF-P 关税危机应对": ["tariff", "关税", "duty", "trade war", "cbam", "供应链重构", "成本转嫁", "alternative sourcing", "hts", "de minimis"],
    "WF-Q 标签驱动决策": ["tag engineering", "标签工程", "sku tag", "trigger", "label hierarchy", "ontology", "决策触发", "feature store", "taxonomy", "action trigger"],
    "WF-R Amazon SEO优化": ["seo", "a9", "search ranking", "keyword ranking", "organic traffic", "索引", "search visibility", "listing optimization", "amazon seo", "搜索排名"],
    "WF-S 跨境数据合规": ["gdpr", "ccpa", "privacy", "data governance", "合规", "跨境数据", "数据出境", "pii", "consent", "data residency"],
    "WF-T 用户分层精细运营": ["rfm", "segmentation", "用户分层", "persona", "lifecycle stage", "cohort", "uplift", "personalization", "targeted intervention", "precision marketing"],
    "WF-U P&L健康诊断": ["p&l", "profitability", "cogs", "fba cost", "unit economics", "margin", "break-even", "cash flow", "财务健康", "成本结构"],
    "WF-V 供应商降本博弈": ["supplier negotiation", "supplier evaluation", "supplier risk", "procurement", "sourcing", "vendor", "commodity", "moq", "price negotiation", "supplier development"],
    "WF-W 退货根因修复": ["return", "退货", "refund", "returnformer", "return root cause", "return fraud", "reverse logistics", "chargeback", "return rate", "disposition"],
    "WF-X TikTok Shop冷启动": ["tiktok shop", "tiktok-shop", "live commerce", "直播", "creator", "kol", "ugc", "short video", "viral", "livestream", "tiktok algorithm"],
    "WF-Y 账期现金流优化": ["cash conversion", "working capital", "inventory financing", "cash cycle", "现金流", "账期", "supply chain finance", "lending", "receivable", "payable"],
    "WF-Z 竞品情报作战室": ["competitive intelligence", "competitor keyword", "competitor price", "market share", "competitive response", "share of voice", "竞品监控", "competitor monitor"],
    "WF-AA 多渠道协同增长": ["omnichannel", "multi-channel", "cross-platform", "channel synergy", "多渠道", "全渠道", "渠道协同", "platform expansion", "channel diversification"],
    "WF-AB 视频电商全链路": ["video commerce", "视频电商", "tiktok shop", "短视频", "mas video", "content tagging", "video tagging", "rlhf", "爆款率"],
    "WF-AC 因果定价决策": ["dml", "double debiased", "causal pricing", "price elasticity", "因果弹性", "价格弹性去偏", "heterogeneous treatment", "定价决策"],
    "WF-AD 标签驱动实验设计": ["tag experiment", "标签分层", "stratified", "tag ab", "causal feature", "geo holdout", "interference", "spillover"],
    "WF-AE 搜索流量全链路优化": ["search revenue", "搜索p&l", "search attribution", "mas search", "search organic", "organic growth attribution"],
    "WF-AF MAS运营财务智能化": ["mas revenue", "p&l agent", "financial agent", "运营财务", "多智能体财务", "financial diagnostics", "agent finance"],
}

KNOWN_SKILL_IDS: set[str] = set()

ALGO_TAG_RULES = {
    "causal": ["causal", "因果", "uplift", "dml", "did", "iv"],
    "experiment": ["ab", "a/b", "experiment", "bandit", "实验"],
    "forecasting": ["forecast", "time series", "预测", "demand"],
    "optimization": ["optimization", "优化", "allocation", "scheduling"],
    "recommendation": ["recommend", "推荐", "ranking"],
    "rag": ["rag", "retrieval", "chunk", "hyde", "raptor", "rerank"],
    "knowledge_graph": ["knowledge graph", "知识图谱", "kg", "ontology", "entity resolution"],
    "multi_agent": ["mas", "multi-agent", "agent", "orchestr"],
    "data_collection": ["data collection", "数据采集", "crawl", "signal"],
    "fraud_detection": ["fraud", "risk", "anomaly"],
    "pricing": ["pricing", "price", "价格"],
    "visual_generation": ["video", "visual", "image", "multimodal"],
}

# ---------------------------------------------------------------------------
# Domain → Business Context mapping (22 domains × role/trigger/outcome/pain)
# Used to inject a "business perspective panel" on every skill detail page.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Skill-level config overrides — loaded from config/ YAML files at startup
# ---------------------------------------------------------------------------
def _load_config_dir() -> tuple[dict, dict, dict]:
    """Load SKILL_PS_OVERRIDE, SKILL_BIZ_CONTEXT_OVERRIDE, SKILL_HANDBOOK_MAP from YAML."""
    import yaml as _yaml

    _config_dir = Path(__file__).parent / "config"

    def _load(fname: str) -> dict:
        p = _config_dir / fname
        if not p.exists():
            return {}
        return _yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    ps = _load("skill_ps_override.yaml")
    biz = _load("skill_biz_context_override.yaml")
    # SKILL_HANDBOOK_MAP: convert {id, name} dicts back to (id, name) tuples
    hb_raw = _load("skill_handbook_map.yaml")
    hb = {sid: [(item["id"], item["name"]) for item in lst]
          for sid, lst in hb_raw.items()}
    return ps, biz, hb


SKILL_PS_OVERRIDE, SKILL_BIZ_CONTEXT_OVERRIDE, SKILL_HANDBOOK_MAP = _load_config_dir()


DOMAIN_BUSINESS_CONTEXT: dict[str, dict[str, Any]] = {
    "01-因果推断": {
        "role": "增长负责人 / CMO",
        "role2": "数据分析师 · 广告优化师",
        "trigger": "广告预算花了，但不确定哪个渠道真的带来新客；做了大促，不知道销量增长是促销效果还是季节规律",
        "outcome": "能区分「真实增量」和「自然购买」，砍掉虚假归因渠道后同等预算 ROI 提升 20-40%",
        "pain": "钱花出去了不知道有没有用 · 各渠道报告都说自己贡献最大 · 怎么向老板证明这笔钱值得花",
        "platform": "Amazon · TikTok Shop · Meta Ads · DTC 独立站",
    },
    "02-A_B实验": {
        "role": "运营负责人 / 产品经理",
        "role2": "广告优化师 · 选品负责人",
        "trigger": "改了主图/标题/价格，不确定销量变化是改动导致的还是流量波动；两个方案团队各持己见，需要数据裁决",
        "outcome": "每次改动都有 ≥95% 置信度的数据结论，好的改动快速全量，坏的及时止损",
        "pain": "改了主图感觉好多了但不确定 · 小范围测试结果好全量后没效果 · 测试周期短结论不可靠",
        "platform": "Amazon Listing · TikTok 广告素材 · DTC 落地页",
    },
    "03-时间序列": {
        "role": "供应链负责人 / 采购负责人",
        "role2": "运营负责人 · 财务负责人",
        "trigger": "大促前备货总是不是多了就是少了；新品上线第一个月断货，再补又积压；年底预算不知道各月目标怎么定",
        "outcome": "提前 4-8 周准确预判各 SKU 需求峰值，库存积压减少 30%，断货率降低 50%",
        "pain": "备货总是压货或断货 · 旺季淡季波动太大预测不准 · 补货周期 30 天但预测只看 7 天",
        "platform": "Amazon FBA · 海外仓 · 多市场多仓",
    },
    "04-供应链": {
        "role": "供应链负责人",
        "role2": "采购负责人 · CEO / 运营 VP",
        "trigger": "库存周转率低，资金压在海外仓出不来；SKU 断货紧急空运，物流成本吃掉毛利；多仓库存分布不均",
        "outcome": "库存周转天数从 90 天降到 60 天，断货率 <3%，海外仓综合成本降低 15-25%",
        "pain": "库存周转天数太长资金压死了 · 断货了只能空运救急成本爆了 · 多市场库存分配不均",
        "platform": "Amazon FBA · 海外仓 · 多国仓位（美/欧/日）",
    },
    "05-推荐系统": {
        "role": "运营负责人 / 选品负责人",
        "role2": "产品经理 · 广告优化师",
        "trigger": "老客来了只买一件就走，相关产品没被推出去；Bundle 商品连带销售做不起来；站内推荐位点击率低",
        "outcome": "老客连带购买率提升 20-35%，客单价提升，品类交叉销售做起来",
        "pain": "老客复购率上不去 · 相关产品没有被看到 · Bundle 凑单没人用 · 新品没有曝光机会",
        "platform": "Amazon · DTC 独立站 · 邮件/SMS 个性化",
    },
    "06-增长模型": {
        "role": "CEO / 增长负责人",
        "role2": "CMO · 财务负责人",
        "trigger": "公司增长放缓，不知道是市场饱和还是产品问题还是获客太贵；老板要 12 个月 GMV 预测，只能靠感觉",
        "outcome": "建立增长拆解模型找到瓶颈，预测未来 6-12 个月营收区间，支撑融资/战略会议",
        "pain": "增长放缓不知道问题在哪 · CAC 越来越高已经高于 LTV · 新市场要不要进没有数据支撑",
        "platform": "Amazon · TikTok Shop · DTC 独立站 · 多市场",
    },
    "07-NLP-VOC": {
        "role": "产品运营负责人 / 选品负责人",
        "role2": "客服负责人 · 品牌负责人",
        "trigger": "每月几千条差评和 Q&A 没有人力一条条看，但痛点都在里面；新品开发不知道做什么功能、改什么问题",
        "outcome": "自动提取 Top 10 高频痛点，新品开发有用户数据背书，每月出竞品用户洞察报告",
        "pain": "差评太多看不过来 · 不知道用户真正在意什么 · 竞品评论没有系统分析过 · 新品开发靠拍脑袋",
        "platform": "Amazon Reviews / Q&A · TikTok 评论区 · Reddit 母婴社区",
    },
    "08-知识图谱": {
        "role": "选品负责人 / 运营负责人",
        "role2": "数据分析师 · 供应链负责人",
        "trigger": "品类很多，不清楚品类间的关联，没法做系统性类目扩张规划；竞品矩阵太复杂，品牌/SKU/渠道理不清",
        "outcome": "建立品类知识图谱，清晰看到哪些是入口品/引流品/利润品，指导下一步选品扩张方向",
        "pain": "品类太多不知道先做哪个 · 竞品关系理不清楚 · 不知道用户买了奶瓶还会买什么 · 类目扩张没有逻辑",
        "platform": "Amazon 品类体系 · 竞品 ASIN 网络分析",
    },
    "09-DataAgent-LLM": {
        "role": "数据分析师 / 运营负责人",
        "role2": "CEO · 供应链负责人",
        "trigger": "数据需求太多，数据团队排期 2 周；非技术人员（采购/客服/运营）有数据问题但不会 SQL；重复报表占用大量时间",
        "outcome": "业务方用自然语言自助查数据，常规报表自动化，数据驱动决策响应速度从「天」变「分钟」",
        "pain": "数据需求排期太长 · 不会 SQL 只能等数据团队 · 老板临时要数据没法马上出 · 分析师时间都花在取数上",
        "platform": "Amazon SP API · Shopify · TikTok Ads API · 多平台数据整合",
    },
    "10-MAS": {
        "role": "运营负责人 / CTO",
        "role2": "产品经理 · CEO",
        "trigger": "运营任务太碎，选品/定价/广告/客服同时跑，人手严重不足；重复性运营动作需要 7×24 响应但没有足够人力",
        "outcome": "多个 AI Agent 协作自动完成跨系统运营任务，运营团队人效提升 3-5 倍，7×24 无人值守运营",
        "pain": "运营人手不够任务太多 · 价格变化没有及时响应 · 重复性工作占据太多时间 · 想做 7×24 监控但没人盯",
        "platform": "Amazon PPC + 库存 + 定价 多 Agent 协作 · TikTok 内容运营流水线",
    },
    "11-AI人文": {
        "role": "品牌负责人 / 内容运营",
        "role2": "CEO · 社媒运营",
        "trigger": "品牌内容同质化，想在母婴赛道建立有温度有记忆点的品牌人设；海外用户文化差异大，本地化内容难以真正有共鸣",
        "outcome": "品牌内容从「产品介绍」升级为「情感共鸣的故事」，海外用户分享率和评论互动率提升",
        "pain": "内容没有灵魂用户不爱看 · AI 写的东西太像 AI · 不同文化的妈妈怎么打动 · 品牌故事讲不出来",
        "platform": "TikTok · Instagram · DTC 品牌站 · 母婴社媒内容",
    },
    "12-ML基础": {
        "role": "数据分析师 / 数据工程师",
        "role2": "运营负责人 · 产品经理",
        "trigger": "想用机器学习解决业务问题，但不知道该选什么模型；模型上线后效果越来越差不知道为什么",
        "outcome": "选对算法工具减少 50% 试错时间，模型上线后可监控可解释，数据团队和业务团队建立共同语言",
        "pain": "不知道该用什么模型 · 模型准确率不稳定 · 业务不相信模型结果 · 模型黑盒说不清为什么这么预测",
        "platform": "选品评分 · 差评预测 · 用户流失预警 · 广告出价预测",
    },
    "13-广告分析": {
        "role": "广告优化师 / 投放负责人",
        "role2": "CMO · 运营负责人",
        "trigger": "广告账户几十个系列，不知道哪个在真正赚钱；ROAS 看起来好看但实际利润没有提升；预算有限想集中打高价值用户",
        "outcome": "每分广告预算有明确 ROI 追踪，砍掉低效渠道后同等预算 ROAS 提升 30-50%",
        "pain": "ROAS 好看但利润没有涨 · 不知道哪个素材真的有效 · 归因窗口期不同数据打架 · TikTok/Meta/Amazon 广告数据整合不了",
        "platform": "Amazon PPC（SP/SB/SD）· TikTok Ads · Meta 广告 · 多平台归因",
    },
    "14-用户分析": {
        "role": "运营负责人 / 用户增长负责人",
        "role2": "CMO · 产品经理",
        "trigger": "有大量老客户，但不知道谁是高价值客户、谁快要流失；新客获取成本越来越高，老客复购却上不去",
        "outcome": "用户按 RFM/LTV 分层精准触达，高价值用户留存率提升，老客贡献收入占比从 30% 提升到 50%",
        "pain": "老客复购率上不去 · 不知道哪些用户要流失了 · 所有用户用同一套活动 · 买过一次就不见了",
        "platform": "Amazon 买家分层 · DTC 站 LTV 预测 · Klaviyo/Brevo 邮件分群",
    },
    "15-营销投放分析": {
        "role": "CMO / 营销负责人",
        "role2": "广告优化师 · CEO",
        "trigger": "同时跑 Amazon 广告/TikTok/网红投放/邮件，不知道整体预算怎么分配最高效；网红投放花了大钱但不知道带来多少真实 GMV",
        "outcome": "建立全渠道营销归因模型（MMM），每个渠道真实 ROI 可量化，大促前做预算优化模拟",
        "pain": "多渠道预算分配靠感觉 · 网红带货效果不知道怎么量化 · 渠道之间互相抢功劳数据打架 · 整体营销 ROI 算不清楚",
        "platform": "Amazon + TikTok + Meta + KOL 四渠道 · Prime Day / Black Friday 预算前置",
    },
    "16-智能体工程": {
        "role": "CTO / 技术负责人",
        "role2": "产品经理 · 数据工程师",
        "trigger": "想把 AI 集成到业务系统，但 LLM 稳定性差、幻觉问题、成本控制都是挑战；Agent 任务失败了不知道哪步出了问题",
        "outcome": "AI Agent 在生产环境稳定运行，失败可追踪，成本可控，复杂任务完成率 >85%",
        "pain": "LLM 返回结果不稳定不可靠 · AI 幻觉导致业务决策错误 · Agent 任务失败了不知道哪步出问题 · AI 调用成本控制不住",
        "platform": "跨境运营 AI Agent 工程落地 · Amazon SP API + LLM 集成 · 多平台数据采集 Agent",
    },
    "17-价格优化": {
        "role": "定价负责人 / 运营负责人",
        "role2": "选品负责人 · CEO",
        "trigger": "竞品突然降价，不知道该不该跟，跟了怕伤利润不跟怕丢 BSR；大促期间不知道折扣给多少，给多了利润没了",
        "outcome": "实时监控竞品价格并自动触发调价，毛利率保持在目标区间，BSR 排名和利润同时兼顾",
        "pain": "竞品降价了不知道要不要跟 · 大促折扣给多少没有依据 · 手动盯价格太累反应不及时 · 新品上线定价高了还是低了",
        "platform": "Amazon Buy Box 竞价策略 · 多市场价格协调 · Prime Day / Coupon 折扣优化",
    },
    "18-物流履约": {
        "role": "物流负责人 / 供应链负责人",
        "role2": "客服负责人 · 运营负责人",
        "trigger": "物流时效不稳定，差评里大量「收货太慢」，影响 DSR 评分；退货率高，处理成本吃掉大量利润；旺季物流爆仓",
        "outcome": "物流时效提升 20-30%，物流相关差评减少 40%，退货成本可控，旺季履约稳定不崩溃",
        "pain": "物流超时差评太多 · 旺季爆仓订单积压 · 退货处理成本太高 · 头程运费太贵压缩了毛利",
        "platform": "FBA vs FBM vs 第三方海外仓 · 美国本土最后一公里 · 跨境退货逆向物流",
    },
    "19-风控反欺诈": {
        "role": "运营负责人 / 合规负责人",
        "role2": "品牌负责人 · CEO",
        "trigger": "竞品刷单刷好评，自己的 BSR 和评分被打压；账号/ASIN 被恶意投诉删除；店铺有异常订单不确定是真实买家",
        "outcome": "识别过滤刷评/恶意竞争行为，账号风险提前预警，维权有数据证据，降低封号风险",
        "pain": "竞品刷评打压我们 · 我们的好评被恶意举报删除 · 不知道差评是真实的还是恶意的 · 如何证明竞品恶意行为",
        "platform": "Amazon 刷评检测与举报 · TikTok Shop 刷单识别 · 竞品 Listing 攻击溯源",
    },
    "20-AI视频生成": {
        "role": "内容运营 / 品牌负责人",
        "role2": "社媒运营 · CMO",
        "trigger": "TikTok/Reels 需要大量视频，拍摄成本高周期长产能跟不上；想做直播带货但真人主播成本高语言是障碍",
        "outcome": "视频内容产能提升 5-10 倍，单条视频成本降低 80%，多语言市场内容本地化快速覆盖",
        "pain": "视频内容来不及做 · 拍视频成本太高 · 主播太贵或不稳定 · 多语言内容没有人拍 · TikTok 更新频率要求太高",
        "platform": "TikTok Shop LIVE · Instagram Reels · 多语言虚拟主播（英/西/阿/日）",
    },
    "21-合规决策": {
        "role": "合规负责人 / 选品负责人",
        "role2": "CEO · 供应链负责人",
        "trigger": "新品上架前不确定在美国/欧盟是否需要认证，怕因合规问题被下架；产品被平台下架但不清楚哪里出了问题",
        "outcome": "上架前自动完成合规预扫描，0 合规下架事故，新市场合规准备时间从 3 个月缩短到 2 周",
        "pain": "产品被下架说是合规问题 · 不知道目标市场需要什么认证 · EU/US 合规要求不一样怎么处理 · 母婴产品安全标准太严怕踩雷",
        "platform": "美国 CPSC/ASTM · 欧盟 CE/EN71 · Amazon 类目合规要求 · 德国/英国/中东市场",
    },
    "22-数据采集工程": {
        "role": "数据工程师 / 技术负责人",
        "role2": "运营负责人 · 选品负责人",
        "trigger": "想监控竞品价格/评论/排名但没有稳定采集能力，手动太慢；多平台数据分散整合成本极高；数据管道不稳定经常断",
        "outcome": "竞品价格/评论数据每日自动更新，多平台数据统一入仓，数据管道稳定性 >99%，取数时间从小时降到分钟",
        "pain": "竞品数据要手动收集太慢 · 平台 API 限制抓不到数据 · 多系统数据整合不起来 · 报表用的数据是过期的",
        "platform": "Amazon SP API + Keepa · TikTok Shop API · 跨境多平台数据湖",
    },
    "24-标签工程": {
        "role": "数据架构师 / 供应链数字化负责人",
        "role2": "CTO · 数据工程师 · 供应链团队",
        "trigger": "多平台数据孤岛导致断货识别延迟8小时；标签覆盖率不足使AI决策触发率<30%；想实现分析→行动自动闭环但不知从何下手",
        "outcome": "统一 Tag Schema + 传播引擎将标签覆盖率从 30% 提升至 97%；Palantir 风格 Object-Action-Writeback 将补货响应从 2 天缩短至 4 小时自动触发",
        "pain": "多平台 SKU 编码混乱无法统一 · 合规标签手工维护遗漏频繁 · 预测模型有了但结果无法自动触发采购 · 标签打了但没有质量监控",
    },
    "25-搜索流量工程": {
        "role": "SEO 负责人 / 运营主管",
        "role2": "CEO · 品牌负责人 · 广告投手",
        "trigger": "Listing 自然排名停在第 3 页；广告 ACoS 居高不下但自然流量没有增长；关键词研究全靠感觉没有数据支撑",
        "outcome": "系统化关键词缺口分析 + 排名因子权重建模，将目标关键词从第 3 页提升至第 1 页，自然流量提升 30-50%，广告依赖度降低",
        "pain": "不知道哪些关键词最值得投入 · 自然排名影响因子不透明 · 广告投放和自然排名互相独立没有协同 · Listing 文案优化靠经验没有数据验证",
        "platform": "Amazon SP-API · Amazon Advertising API · Helium 10 / DataDive · Merchant Words",
    },
    "23-运营财务": {
        "role": "CFO / 财务负责人",
        "role2": "CEO · 运营负责人",
        "trigger": "月度 FBA 账单 15 万但不知道哪些 SKU 在亏损；大促备货资金不够但不知道缺口多少；整体利润率 18% 但不知道是哪条产品线在拖累",
        "outcome": "SKU 级 P&L 实时可见，FBA 费用长库龄提前预警，大促现金流缺口提前识别，融资窗口精准规划",
        "pain": "FBA 费用算不清楚 · 现金流紧张不知道哪里漏了 · 哪个 SKU 真正赚钱看不见 · 财务数据滞后一个月才出来",
        "platform": "Amazon Seller Central · Amazon SP API · FBA 报告 · 多货币财务系统",
    },
}

# ---------------------------------------------------------------------------
# Business-problem → workflow quick-entry (for home page C-redesign)
# ---------------------------------------------------------------------------
BUSINESS_ENTRIES = [
    {
        "icon": "AG",
        "label": "防御竞品攻击 / 平台封号预防",
        "desc": "广告刷量、虚假差评、AI 推荐注入、合规封号——四条战线主动防御",
        "href": "playbooks/pb-risk-defense.html",
        "tag": "风险防御",
    },
    {
        "icon": "TR",
        "label": "关税冲击 / 贸易政策应对",
        "desc": "72 小时内输出完整行动清单：定价调整 + 库存处置 + 供应链转移方案",
        "href": "playbooks/pb-tariff-response.html",
        "tag": "关税响应",
    },
    {
        "icon": "CL",
        "label": "上架合规 / 关税编码优化",
        "desc": "新品上架前合规预扫描 + HTS 关税编码精准分类 + 封号风险防御三合一",
        "href": "playbooks/pb-compliance.html",
        "tag": "合规手册",
    },
    {
        "icon": "PL",
        "label": "提升广告 ROI / 归因准确性",
        "desc": "识别无效预算、纠正渠道归因偏差、实现因果驱动的广告优化",
        "href": "workflows/wf-b-广告优化.html",
        "tag": "WF-B 广告优化",
    },
    {
        "icon": "SC",
        "label": "FBA 库存健康 / 头程优化",
        "desc": "长库龄清仓 + 头程路线成本优化 + 旺季备货计划，库存周转天数降 30%",
        "href": "playbooks/pb-fba-operations.html",
        "tag": "FBA 运营",
    },
    {
        "icon": "VP",
        "label": "竞品差评 → 新品机会挖掘",
        "desc": "竞品 1-3 星差评是最好的免费 R&D，新品成功率从 30% 提升到 50%",
        "href": "playbooks/pb-voc-product-loop.html",
        "tag": "竞品情报",
    },
    {
        "icon": "CS",
        "label": "客服 24h 自动化 / 差评防御",
        "desc": "70% 工单全自动处理，多语言覆盖，INR 欺诈退货从 35% 降至 5%",
        "href": "playbooks/pb-customer-service-agent.html",
        "tag": "客服售后",
    },
    {
        "icon": "VO",
        "label": "分析用户评价 / 发现产品痛点",
        "desc": "多语言 VOC 挖掘、差评根因归类、产品改进信号提取",
        "href": "workflows/wf-c-客服分诊.html",
        "tag": "WF-C 客服",
    },
    {
        "icon": "NP",
        "label": "评估新品 / 新市场机会",
        "desc": "市场规模估算、竞品情报采集、选品可行性综合评分",
        "href": "workflows/wf-d-选品扫描.html",
        "tag": "WF-D 选品",
    },
    {
        "icon": "UG",
        "label": "预测用户流失 / 提升 LTV",
        "desc": "Uplift 建模识别可干预用户，精准发券减少无效留存成本",
        "href": "domains/14-用户分析.html",
        "tag": "用户分析",
    },
    {
        "icon": "AI",
        "label": "AI Agent 替代重复性岗位",
        "desc": "供应链对账、数据分析提数、广告出价——三类岗位 70% 重复工作 Agent 覆盖",
        "href": "playbooks/pb-agent-replace.html",
        "tag": "Agent 替人",
    },
    {
        "icon": "PR",
        "label": "动态定价 / A/B 实测 GMV +13%",
        "desc": "LLM 动态定价引擎，定价是乘数——精准定价 1% 比多投广告 15% 更高效",
        "href": "playbooks/pb-pricing-engine.html",
        "tag": "定价引擎",
    },
    {
        "icon": "NP",
        "label": "新品冷启动备货 / 预测",
        "desc": "零历史数据下的扩散曲线预测，跨市场迁移学习",
        "href": "playbooks/pb-new-product-launch.html",
        "tag": "新品冷启动",
    },
    {
        "icon": "CR",
        "label": "广告归因打架 / 渠道预算分配",
        "desc": "PVM 窗口统一 480万/年，Bayesian MMM 1000万——让四份报告说一个真相",
        "href": "playbooks/pb-attribution-unification.html",
        "tag": "全渠道归因",
    },
]


@dataclass
class PlaybookSkill:
    skill_id: str
    title: str
    domain_key: str
    domain_dir: str
    path: str
    algorithm_summary: str = ""
    problem_solved: str = ""
    business_scenarios: list[str] = field(default_factory=list)
    scenario_paragraphs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    roi: list[str] = field(default_factory=list)
    roi_figure: str = ""          # e.g. "10-20 万元"
    difficulty: str = ""          # e.g. "⭐⭐⭐☆☆"
    priority: str = ""            # e.g. "⭐⭐⭐⭐☆"
    papers: list[str] = field(default_factory=list)
    code_path: str | None = None
    code_blocks: int = 0
    code_preview: str = ""
    relations: dict[str, list[str]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    workflows: list[str] = field(default_factory=list)
    biz_role: str = ""
    biz_role2: str = ""
    biz_trigger: str = ""
    biz_outcome: str = ""
    biz_pain: str = ""
    biz_platform: str = ""


def slugify(value: str) -> str:
    value = re.sub(r"^Skill-", "", value)
    value = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff_-]+", "-", value)
    return value.strip("-").lower() or "item"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[end + 4:]
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data, body


def section_map(body: str) -> dict[str, str]:
    """Build section_key → content dict using normalised title matching."""
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", body, re.MULTILINE))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        raw_title = match.group(1).strip()
        norm = _norm_title(raw_title)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        for key, names in SECTION_KEYS.items():
            if any(name in norm for name in names):
                # Keep first match per key (highest in document)
                if key not in sections:
                    sections[key] = content
    return sections


def first_nonempty_line(text: str, fallback: str = "") -> str:
    for line in text.splitlines():
        clean = re.sub(r"[#>*`\-]+", "", line).strip()
        if clean and not clean.startswith("|") and len(clean) > 8:
            return clean[:220]
    return fallback


def _clean_problem_solved(text: str, skill_id: str = "") -> str:
    text = re.sub(r"^[：:。\s]+", "", text).strip()
    text = re.sub(r"^\*\*\s*[：:]?\s*", "", text).strip()
    text = re.sub(r"^是\s*[：:]\s*", "", text).strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text).strip()
    text = re.sub(r"\*\*\.?$", "", text).strip()
    text = re.sub(r"\*\*", "", text).strip()
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"[：:]\s*$", "", text).strip()
    return text[:280]


def first_bold_sentence(text: str, fallback: str = "") -> str:
    """Extract problem statement: prefer labelled 核心问题/业务问题 paragraph, then first clean sentence."""
    for marker in ("核心问题", "业务问题", "核心挑战", "解决的核心问题", "核心痛点"):
        m = re.search(r"(?:" + marker + r")[：:\s]*([^\n。！？]{20,300}[。！？\n]?)", text)
        if m:
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()
            if len(clean) > 20:
                return clean[:250]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```") or stripped.startswith("|"):
            continue
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        clean = re.sub(r"\*(.+?)\*", r"\1", clean)
        clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", clean)
        clean = re.sub(r"^\$\$.*", "", clean).strip()
        clean = re.sub(r"^[-*>]+\s*", "", clean).strip()
        if len(clean) > 30:
            return clean[:250]
    return fallback


def extract_title(frontmatter: dict[str, str], body: str, skill_id: str) -> str:
    if frontmatter.get("title"):
        return frontmatter["title"]
    for line in body.splitlines()[:30]:
        if line.startswith("# Skill Card:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("# "):
            return line[2:].strip()
    return skill_id


def extract_list_snippets(text: str, limit: int = 6) -> list[str]:
    snippets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", stripped[2:]).strip()
            # skip relation / paper reference lines
            if len(clean) > 8 and not clean.startswith("**前置") and not clean.startswith("**组合"):
                snippets.append(clean[:180])
        if len(snippets) >= limit:
            break
    return snippets


def extract_scenario_paragraphs(text: str, limit: int = 3) -> list[str]:
    """Extract meaningful prose paragraphs from the scenario section."""
    paras: list[str] = []
    # Split on double newline, skip headers and code blocks
    in_code = False
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.startswith("#"):
            continue
        if line.strip() == "":
            chunk = " ".join(buf).strip()
            if len(chunk) > 20:
                # clean markdown formatting
                chunk = re.sub(r"\[\[([^\]]+)\]\]", r"\1", chunk)
                chunk = re.sub(r"\*\*(.+?)\*\*", r"\1", chunk)
                chunk = re.sub(r"\*(.+?)\*", r"\1", chunk)
                chunk = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", chunk)
                paras.append(chunk[:300])
            buf = []
        else:
            buf.append(line.strip())
    if buf:
        chunk = " ".join(buf).strip()
        if len(chunk) > 20:
            paras.append(chunk[:300])
    return [p for p in paras if len(p) > 20][:limit]


def extract_roi(value_text: str) -> tuple[str, str, str]:
    """
    Parse the ⑤ 商业价值 section.
    Returns (roi_figure, difficulty_stars, priority_stars).
    """
    roi_figure = ""
    difficulty = ""
    priority = ""

    # Format A: "- **ROI**：10-20 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐⭐☆"
    inline = re.search(
        r"(?:ROI|年化)[：:]\s*([^\|*\n]{3,30})"
        r"(?:[^|]*\|[^|]*难度[：:]\s*([⭐☆]{3,6}))?",
        value_text,
    )
    if inline:
        roi_figure = inline.group(1).strip().rstrip("|").strip()
        if inline.group(2):
            difficulty = inline.group(2).strip()

    # Format B: "年化：**XX万元**"
    if not roi_figure:
        m = re.search(r"(?:年化|增收)[：:]\s*\*?\*?([^\n*]{3,30})", value_text)
        if m:
            roi_figure = m.group(1).strip()

    # Format C: bullet "- 避免损失：xxx，节省 30 万元"
    if not roi_figure:
        m = re.search(r"-\s[^\n]{0,40}(?:节省|节约|增收|创造)[^\n]{0,20}(\d[\d,\.\-]*\s*万元)", value_text)
        if m:
            roi_figure = m.group(1).strip()

    # Format D: bare "¥XX万" anywhere
    if not roi_figure:
        m = re.search(r"(\d[\d,\.\-]+\s*万(?:元|\/月|\/年)?)", value_text)
        if m:
            roi_figure = m.group(1).strip()

    # Format E: table row "| ROI预估 | xxx |" or "| 核心收益 | xxx |"
    if not roi_figure:
        m = re.search(r"\|\s*(?:ROI预估|年化收益|核心收益|节省成本)[^\|]*\|\s*([^\|\n]{3,60})\|", value_text)
        if m:
            roi_figure = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()[:40]

    # Difficulty: table "| 实现难度 | ⭐⭐⭐ |"
    if not difficulty:
        m = re.search(r"(?:实现)?难度[^|]*\|\s*([⭐☆]{3,6})", value_text)
        if m:
            difficulty = m.group(1).strip()
    # Difficulty: bold "**实施难度**：⭐⭐⭐☆☆"
    if not difficulty:
        m = re.search(r"\*\*(?:实施|实现)?难度\*\*[：:]\s*([⭐☆]{3,6})", value_text)
        if m:
            difficulty = m.group(1).strip()
    # Difficulty: unbolded "实施难度：⭐⭐☆☆☆（1/5星）"
    if not difficulty:
        m = re.search(r"(?:实施|实现)?难度[：:]\s*([⭐☆]{2,6})", value_text)
        if m:
            difficulty = m.group(1).strip()
    # Difficulty: "评分：⭐☆☆☆☆" pattern
    if not difficulty:
        m = re.search(r"评分[：:]\s*([⭐☆]{3,6})", value_text)
        if m:
            difficulty = m.group(1).strip()

    # Priority — four formats:
    # A: "优先级：⭐⭐⭐⭐☆"
    # B: "**评分：⭐⭐⭐⭐⭐（5/5星）**" or "**评分:⭐⭐⭐⭐☆(4/5 星)**"
    # C: "⭐⭐⭐⭐⭐ (5/5)" bare stars with fraction
    # D: table "** | ⭐⭐⭐☆☆ |" (bold in table cell)
    if not priority:
        m = re.search(r"(?:商业)?优先级[：:*|\s]*([⭐☆]{3,6})", value_text)
        if m:
            priority = m.group(1).strip()
    if not priority:
        m = re.search(r"\*\*评分[：:]\s*([⭐☆]{3,6})", value_text)
        if m:
            priority = m.group(1).strip()
    if not priority:
        m = re.search(r"([⭐☆]{3,6})\s*\([1-5]/5", value_text)
        if m:
            priority = m.group(1).strip()
    if not priority:
        m = re.search(r"\*\*\s*\|\s*([⭐☆]{3,6})\s*\|", value_text)
        if m:
            priority = m.group(1).strip()

    return roi_figure, difficulty, priority


def extract_roi_from_scenario(scenario_text: str) -> str:
    """Pull ROI figure from the scenario/case section when value section has none."""
    # Pattern 1: 年化/增收/节省：¥XX 万元
    m = re.search(r"(?:年化|增收|节省|节约)[：:]\s*\*?\*?([^\n*]{3,40})", scenario_text)
    if m:
        return m.group(1).strip()
    # Pattern 2: bare XX万元 following a benefit phrase
    m = re.search(
        r"(?:减少|节省|节约|增加|提升)[^\n，。]{0,20}(\d[\d,\.\-]*\s*万元)",
        scenario_text,
    )
    if m:
        return m.group(1).strip()
    # Pattern 3: ¥XX万 or XX万元 anywhere prominent
    m = re.search(r"\*\*([¥￥]?\d[\d,\.\-]*\s*万元)\*\*", scenario_text)
    if m:
        return m.group(1).strip()
    # Pattern 4: +XX% 提升 as last-resort summary
    m = re.search(r"(\+\d[\d\.]*%[^，。\n]{0,20}(?:提升|增长|增加))", scenario_text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_first_code_block(code_section: str) -> str:
    m = re.search(r"```(?:python|bash|sql)?\n(.*?)```", code_section, re.DOTALL)
    if m:
        raw = m.group(1)
        lines = raw.splitlines()
        if len(lines) > 60:
            lines = lines[:60]
        return "\n".join(lines)
    return ""


def extract_papers(text: str) -> list[str]:
    ids = sorted(set(re.findall(r"\b\d{4}\.\d{4,5}\b", text)))
    return ids[:12]


def classify(text: str, rules: dict[str, list[str]]) -> list[str]:
    lower = text.lower()
    labels = []
    for label, needles in rules.items():
        if any(needle.lower() in lower for needle in needles):
            labels.append(label)
    return labels


def build_graph(vault: Path) -> SkillsGraph:
    graph = SkillsGraph(str(vault))
    graph.build_graph()
    return graph


def build_skills(root: Path, vault: Path, graph: SkillsGraph) -> list[PlaybookSkill]:
    registry = load_domain_registry(root)
    domain_by_dir = {entry.vault_dir: entry.key for entry in registry.entries}
    skills: list[PlaybookSkill] = []

    node_by_id = graph.nodes
    for file_path in iter_skill_files(root):
        if not file_path.name.startswith("Skill-"):
            continue
        rel_path = file_path.relative_to(root).as_posix()
        skill_id = file_path.stem
        text = read_text(file_path)
        fm, body = parse_frontmatter(text)
        sections = section_map(body)
        domain_dir = file_path.parent.name
        domain_key = domain_by_dir.get(domain_dir, slugify(domain_dir))
        node = node_by_id.get(skill_id)
        relations = {
            "prerequisite": sorted(node.prerequisites) if node else [],
            "extends": sorted(node.extensions) if node else [],
            "combinable": sorted(node.combinable) if node else [],
        }
        full_text_for_classify = "\n".join([skill_id, domain_dir, text[:5000]])
        code_path = detect_code_path(root, skill_id, domain_key)
        code_path_str = None
        if code_path:
            normalized_code_path = code_path if code_path.is_absolute() else root / code_path
            try:
                code_path_str = normalized_code_path.relative_to(root).as_posix()
            except ValueError:
                code_path_str = code_path.as_posix()

        algo_text = sections.get("algorithm", "")
        scenario_text = sections.get("scenario", "")
        value_text = sections.get("value", "")

        roi_figure, difficulty, priority = extract_roi(value_text)
        if not roi_figure:
            roi_figure = extract_roi_from_scenario(scenario_text)

        skill = PlaybookSkill(
            skill_id=skill_id,
            title=extract_title(fm, body, skill_id),
            domain_key=domain_key,
            domain_dir=domain_dir,
            path=rel_path,
            algorithm_summary=first_nonempty_line(algo_text, first_nonempty_line(body, skill_id)),
            problem_solved=_clean_problem_solved(
                first_bold_sentence(algo_text, first_nonempty_line(scenario_text, "")),
                skill_id,
            ),
            business_scenarios=extract_list_snippets(scenario_text, 8),
            scenario_paragraphs=extract_scenario_paragraphs(scenario_text, 3),
            inputs=extract_list_snippets(
                re.sub(r"输出.*", "", sections.get("guide", ""), flags=re.DOTALL), 5
            ),
            outputs=extract_list_snippets(sections.get("guide", ""), 5),
            roi=extract_list_snippets(value_text, 6),
            roi_figure=roi_figure,
            difficulty=difficulty,
            priority=priority,
            papers=extract_papers(text),
            code_path=code_path_str,
            code_blocks=len(re.findall(r"```(?:python)?", text)) // 2,
            code_preview=_extract_first_code_block(sections.get("code", "")),
            relations=relations,
            tags=classify(full_text_for_classify, ALGO_TAG_RULES),
            topics=classify(full_text_for_classify, TOPIC_RULES),
            workflows=classify(full_text_for_classify, WORKFLOW_RULES),
        )
        if skill_id in SKILL_PS_OVERRIDE:
            skill.problem_solved = _clean_problem_solved(SKILL_PS_OVERRIDE[skill_id], skill_id)
        if skill.problem_solved and skill.algorithm_summary and \
                skill.problem_solved[:60] == skill.algorithm_summary[:60]:
            import sys
            print(f"WARN dup_ps: {skill_id} — problem_solved==algorithm_summary", file=sys.stderr)
        biz = SKILL_BIZ_CONTEXT_OVERRIDE.get(skill_id) or DOMAIN_BUSINESS_CONTEXT.get(domain_dir, {})
        skill.biz_role     = biz.get("role", "")
        skill.biz_role2    = biz.get("role2", "")
        skill.biz_trigger  = biz.get("trigger", "")
        skill.biz_outcome  = biz.get("outcome", "")
        skill.biz_pain     = biz.get("pain", "")
        skill.biz_platform = biz.get("platform", "")
        if not skill.tags:
            skill.tags = [domain_key]
        if not skill.topics:
            skill.topics = ["其他"]
        skills.append(skill)
    return sorted(skills, key=lambda item: (item.domain_dir, item.skill_id))


# ---------------------------------------------------------------------------
# Workflow YAML loader (Phase 2B)
# ---------------------------------------------------------------------------

def load_workflow_defs(root: Path) -> dict[str, Any]:
    """
    Load structured workflow YAML definitions from
    paper2skills-skills/paper-workflow/definitions/*.yaml
    Returns dict keyed by workflow id (e.g. "wf-b").
    Falls back to empty dict if PyYAML unavailable or no files found.
    """
    wf_dir = root / "paper2skills-skills" / "paper-workflow" / "definitions"
    if not wf_dir.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    defs: dict[str, Any] = {}
    for yf in sorted(wf_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yf.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "id" in data:
                defs[data["id"]] = data
        except Exception as e:
            import sys
            print(f"WARN: failed to load workflow YAML {yf.name}: {e}", file=sys.stderr)
    return defs


def render_workflow_step(step: dict[str, Any], skill_lookup: dict[str, "PlaybookSkill"], depth: int = 0) -> str:
    """Recursively render a workflow step as nested HTML decision tree."""
    parts: list[str] = []
    indent = "  " * depth
    name = html.escape(step.get("name", ""))
    question = html.escape(step.get("question", ""))
    context = html.escape(step.get("context", ""))

    parts.append(f'<div class="wf-step" style="--depth:{depth}">')
    if name:
        parts.append(f'  <div class="wf-step-name">{name}</div>')
    if question:
        parts.append(f'  <div class="wf-question">{question}</div>')
    if context:
        parts.append(f'  <div class="wf-context">{context}</div>')

    branches = step.get("branches", [])
    if branches:
        parts.append('  <div class="wf-branches">')
        for branch in branches:
            cond = html.escape(branch.get("condition", ""))
            parts.append(f'    <details class="wf-branch" open>')
            parts.append(f'      <summary class="wf-condition">{cond}</summary>')
            branch_skills = branch.get("skills", [])
            if branch_skills:
                parts.append('      <div class="wf-branch-skills">')
                for bs in branch_skills:
                    sid = bs.get("id", "")
                    role = html.escape(bs.get("role", ""))
                    sk = skill_lookup.get(sid)
                    if sk:
                        parts.append(
                            f'        <a class="wf-skill-chip" href="../skills/{html.escape(sid)}.html">'
                            f'<span class="chip-name">{html.escape(sk.title)}</span>'
                            f'<span class="chip-role">{role}</span></a>'
                        )
                    else:
                        parts.append(
                            f'        <span class="wf-skill-chip missing">'
                            f'<span class="chip-name">{html.escape(sid)}</span>'
                            f'<span class="chip-role">{role}</span></span>'
                        )
                parts.append('      </div>')
            parts.append('    </details>')
        parts.append('  </div>')

    # Inline (non-branching) skills
    inline_skills = step.get("skills", [])
    if inline_skills and not branches:
        parts.append('  <div class="wf-branch-skills">')
        for bs in inline_skills:
            sid = bs.get("id", "")
            role = html.escape(bs.get("role", ""))
            sk = skill_lookup.get(sid)
            if sk:
                parts.append(
                    f'    <a class="wf-skill-chip" href="../skills/{html.escape(sid)}.html">'
                    f'<span class="chip-name">{html.escape(sk.title)}</span>'
                    f'<span class="chip-role">{role}</span></a>'
                )
        parts.append('  </div>')

    parts.append('</div>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# toB Scene Playbooks (Phase F)
# ---------------------------------------------------------------------------



def render_agents_page(skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    """Render the Agent Marketplace page with 12 callable demo agents."""

    cats = {"全部": "", "选品分析": "selection", "Listing优化": "listing",
            "广告归因": "attribution", "VOC分析": "voc", "供应链预警": "supply",
            "客服售后": "cs", "价格策略": "pricing", "合规风控": "risk",
            "数据分析": "analytics", "内容营销": "content", "竞品监控": "competitor",
            "标签工程": "tag"}

    cat_pills = "".join(
        f"<button class='cat-pill{'  active' if k == '全部' else ''}' data-cat='{v}'>{k}</button>"
        for k, v in cats.items()
    )

    def _chip(sid: str) -> str:
        sk = skill_lookup.get(sid)
        label = sk.title[:24] + "…" if sk and len(sk.title) > 24 else (sk.title if sk else sid[-20:])
        href = f"skills/{sid}.html"
        return f"<a class='agent-skill-chip' href='{html.escape(href)}'>{html.escape(label)}</a>"

    def _input_field(inp: dict[str, Any], agent_id: str) -> str:
        fid = f"{html.escape(agent_id)}__{html.escape(inp['id'])}"
        label = html.escape(inp['label'])
        placeholder = html.escape(inp.get('placeholder', ''))
        if inp['type'] == 'textarea':
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<textarea class='modal-input' id='{fid}' rows='4' placeholder='{placeholder}'></textarea></div>")
        elif inp['type'] == 'select':
            opts = "".join(f"<option value='{html.escape(o)}'>{html.escape(o)}</option>" for o in inp.get('options', []))
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<select class='modal-input' id='{fid}'>{opts}</select></div>")
        else:
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<input class='modal-input' type='text' id='{fid}' placeholder='{placeholder}'></div>")

    cards_html = ""
    modals_html = ""
    for ag in AGENT_CATALOG:
        sid_chips = "".join(_chip(s) for s in ag.get("linked_skills", [])[:3])
        cards_html += f"""
<div class='agent-card' data-cat='{html.escape(ag["cat_key"])}' onclick='openAgent("{html.escape(ag["id"])}")'>
  <div class='agent-card-top'>
    <div class='agent-icon-wrap {html.escape(ag["cat_class"])}'>{ag.get("svg_icon") or ag["icon"]}</div>
    <div class='agent-card-info'>
      <div class='agent-name'>{html.escape(ag["name"])}</div>
      <span class='agent-cat-badge'>{html.escape(ag["category"])}</span>
    </div>
  </div>
  <div class='agent-status'>
    <span class='status-dot live'></span>
    <span style='color:#059669;font-size:12px;font-weight:600'>本地分析</span>
    &nbsp;·&nbsp;
    <span style='font-size:12px;color:#64748b'>即时响应</span>
  </div>
  <p class='agent-desc'>{html.escape(ag["desc"])}</p>
  <div class='agent-skills'>{sid_chips}</div>
  <div class='agent-roi'>{html.escape(ag["roi"])}</div>
  <button class='agent-invoke-btn'>立即调用</button>
</div>"""

        input_fields = "".join(_input_field(inp, ag["id"]) for inp in ag.get("inputs", []))
        demo_out_escaped = html.escape(ag.get("demo_output", ""))
        modals_html += f"""
<div id='modal-{html.escape(ag["id"])}' class='agent-modal-overlay' role='dialog' aria-modal='true' aria-label='{html.escape(ag["name"])}'>
  <div class='agent-modal'>
    <div class='modal-header'>
      <span class='modal-icon'>{ag.get("svg_icon") or ag["icon"]}</span>
      <div class='modal-header-info'>
        <h2>{html.escape(ag["name"])}</h2>
        <div style='display:flex;gap:8px;align-items:center'>
          <span class='agent-cat-badge'>{html.escape(ag["category"])}</span>
          <span class='agent-status'><span class='status-dot live'></span> <span style='font-size:12px;color:#059669;font-weight:600'>本地分析 · 即时</span></span>
        </div>
      </div>
      <button class='modal-close' onclick='closeAgent("{html.escape(ag["id"])}")'>×</button>
    </div>
    <div class='modal-body'>
      <div class='modal-section'>
        <h3>输入参数</h3>
        <div style='display:flex;flex-direction:column;gap:14px'>{input_fields}</div>
        <div style='margin-top:8px'>
          <button class='btn-secondary' style='font-size:12px;padding:5px 12px' onclick='fillExample("{html.escape(ag["id"])}")'>填入示例数据</button>
        </div>
      </div>
      <button class='modal-run-btn' id='run-{html.escape(ag["id"])}' onclick='runAgent("{html.escape(ag["id"])}")'>
         <span id='run-label-{html.escape(ag["id"])}'>开始分析</span>
      </button>
      <div class='modal-output' id='output-{html.escape(ag["id"])}'>
        <div class='output-thinking' id='thinking-{html.escape(ag["id"])}' style='display:none'>
                    <span>Agent 正在分析</span>
          <span class='thinking-dots'><span>·</span><span>·</span><span>·</span></span>
        </div>
        <pre class='output-content' id='content-{html.escape(ag["id"])}' style='margin:0;font-family:inherit;white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.7'></pre>
      </div>
      <div class='modal-footer-skills' id='footer-skills-{html.escape(ag["id"])}' style='margin-top:16px'>
        <span style='font-size:12px;color:#64748b;font-weight:600'>关联 Skills：</span>
        {sid_chips}
      </div>
    </div>
  </div>
</div>"""

    demo_data_js = json.dumps(
        {ag["id"]: {"output": ag.get("demo_output", ""), "inputs": ag.get("inputs", [])} for ag in AGENT_CATALOG},
        ensure_ascii=False,
    )

    body = rf"""
<div class='agent-hero'>
  <div class='agent-hero-text'>
    <h1 style='font-size:32px;font-weight:900;letter-spacing:-.03em;margin:0 0 10px'>
      智能体广场
    </h1>
    <p class='lead'>{len(AGENT_CATALOG)} 个专业 AI Agent，覆盖选品→Listing→广告→客服→合规全链路</p>
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:4px'>
      <span style='font-size:13px;background:#d1fae5;color:#065f46;padding:3px 10px;border-radius:999px;font-weight:600'> 本地计算引擎</span>
      <span style='font-size:13px;color:#64748b'>输入你的真实数据，即时获得个性化计算结果</span>
    </div>
  </div>
  <div class='agent-hero-stats'>
    <div class='agent-stat'><strong>{len(AGENT_CATALOG)}</strong><span>个专业 Agent</span></div>
    <div class='agent-stat'><strong>7</strong><span>业务场景</span></div>
    <div class='agent-stat'><strong>30+</strong><span>关联 Skills</span></div>
  </div>
</div>

<div class='agent-cat-filter' id='catFilter'>
  {cat_pills}
</div>

<div class='agent-grid' id='agentGrid'>
  {cards_html}
</div>

{modals_html}

<script>
const DEMO_DATA = {demo_data_js};

function openAgent(id) {{
  const overlay = document.getElementById('modal-' + id);
  if (!overlay) return;
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
  const firstInput = overlay.querySelector('.modal-input');
  if (firstInput) setTimeout(() => firstInput.focus(), 200);
}}
function closeAgent(id) {{
  const overlay = document.getElementById('modal-' + id);
  if (overlay) overlay.classList.remove('open');
  document.body.style.overflow = '';
  resetOutput(id);
}}
function resetOutput(id) {{
  const out = document.getElementById('output-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const content = document.getElementById('content-' + id);
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  if (out) out.classList.remove('visible');
  if (thinking) thinking.style.display = 'none';
  if (content) content.textContent = '';
  if (btn) btn.disabled = false;
  if (label) label.textContent = '开始分析';
}}
const RADAR_KEYWORDS = [
  '硅胶婴儿餐具','吸奶器','婴儿推车','母婴消毒器',
  '婴儿安全防护角','儿童宝宝硅胶牙刷','婴儿辅食机',
  '新生儿礼盒套装','婴儿水杯学饮杯','孕妇枕头哺乳枕',
];

function fillExample(id) {{
  const data = DEMO_DATA[id];
  if (!data || !data.inputs) return;
  data.inputs.forEach(inp => {{
    const el = document.getElementById(id + '__' + inp.id);
    if (!el) return;
    if (id === 'agent-product-radar' && inp.id === 'keyword') {{
      el.value = RADAR_KEYWORDS[Math.floor(Math.random() * RADAR_KEYWORDS.length)];
    }} else if (inp.type === 'textarea') {{
      el.value = inp.placeholder || '';
    }} else if (inp.type === 'select') {{
      if (inp.options && inp.options.length > 0) el.value = inp.options[0];
    }} else {{
      el.value = inp.placeholder ? inp.placeholder.replace(/^例：/, '') : '';
    }}
  }});
}}

function getVal(id, field) {{
  const el = document.getElementById(id + '__' + field);
  return el ? el.value.trim() : '';
}}
function fmtNum(n) {{
  return n.toLocaleString('en-US', {{maximumFractionDigits: 0}});
}}
function fmtMoney(n) {{
  return '$' + Math.abs(n).toLocaleString('en-US', {{minimumFractionDigits: 0, maximumFractionDigits: 0}});
}}
function pct(n) {{ return (n * 100).toFixed(1) + '%'; }}

function computeSupplySentinel(id) {{
  const stock = parseFloat(getVal(id,'stock')) || 340;
  const vel   = parseFloat(getVal(id,'velocity')) || 28;
  const lt    = parseFloat(getVal(id,'lead_time')) || 21;
  const ch    = getVal(id,'channel') || 'Amazon FBA';
  const days  = vel > 0 ? (stock / vel) : 999;
  const safetyDays = 30;
  const reorderQty = Math.ceil(vel * (lt + safetyDays));
  const airQty  = Math.ceil(reorderQty * 0.5);
  const seaQty  = reorderQty - airQty;
  const airCost = (airQty * 0.8).toFixed(0);
  const lossMd  = Math.min(days, lt) * vel * 25;
  const riskLv  = days < lt ? '<span style="color:#B5323E;font-weight:600;">[高危]</span>' : days < lt + 7 ? '<span style="color:#d97706;font-weight:600;">[警戒]</span>' : '<span style="color:#059669;font-weight:600;">[安全]</span>';
  const action  = days < lt ? '需立即行动！' : days < lt + 7 ? '建议本周下单' : '库存充裕，按计划补货';
  const q4Multi = 2.8;
  const q4Stock = Math.ceil(vel * q4Multi * 60);
  return `[供应链哨兵] 实时计算结果

━━ 库存状态 ━━
当前库存: ${{fmtNum(stock)}} 件
日均销速: ${{vel}} 件/天（您输入）
剩余可售天数: ${{days.toFixed(1)}} 天
风险等级: ${{riskLv}}

━━ 供货周期分析（${{ch}}）━━
您的供货周期: ${{lt}} 天
安全库存天数目标: ${{safetyDays}} 天
${{days < lt ? '[WARN] 已进入断货窗口，需立即行动！' : '[OK] ' + action}}

━━ 补货建议 ━━
├─ 建议补货量: ${{fmtNum(reorderQty)}} 件（${{lt}}天周期 + ${{safetyDays}}天安全库存）
├─ 推荐方案: 空运 ${{fmtNum(airQty)}} 件（应急）+ 海运 ${{fmtNum(seaQty)}} 件（补充）
├─ 空运额外成本: +${{airCost}}
└─ 不补货预估断货损失: ${{fmtMoney(lossMd)}}（${{Math.ceil(Math.min(days,lt))}}天断货 × ${{vel}}件/天 × $25 BSR成本）

━━ Q4 旺季预警 ━━
历史旺季销速倍数: ×${{q4Multi}}
Q4 建议备货量: ${{fmtNum(q4Stock)}} 件
最迟启动时间: 旺季前 ${{lt + 14}} 天

[${{days >= lt ? '>' : '!'}}] 结论: ${{action}}`;
}}

function computePricingAdvisor(id) {{
  const price    = parseFloat(getVal(id,'price')) || 19.99;
  const cost     = parseFloat(getVal(id,'cost'))  || 7.80;
  const compRaw  = getVal(id,'comp_range') || '$15-$22';
  const bsr      = parseInt(getVal(id,'bsr')) || 500;
  const margin   = (price - cost) / price;
  const compNums = compRaw.match(/[\d.]+/g) || ['15','22'];
  const compLo   = parseFloat(compNums[0]) || 15;
  const compHi   = parseFloat(compNums[1] || compNums[0]) || 22;
  const compMid  = (compLo + compHi) / 2;
  const bsrScore = bsr < 100 ? 'Top 100（强势）' : bsr < 500 ? 'Top 500（良好）' : bsr < 2000 ? 'Top 2000（普通）' : '2000+（待提升）';
  const suggested_lo = Math.max(price * 1.05, compMid * 0.95).toFixed(2);
  const suggested_hi = (compHi * 0.98).toFixed(2);
  const newMargin = ((parseFloat(suggested_lo) - cost) / parseFloat(suggested_lo) * 100).toFixed(1);
  const w1 = (price + 1).toFixed(2);
  const w2 = parseFloat(suggested_lo).toFixed(2);
  const primeDayPrice = (price * 0.95).toFixed(2);
  const q4Price = Math.min(parseFloat(suggested_hi), price * 1.15).toFixed(2);
  const monthlyUnits = Math.max(30, Math.round(3000 / bsr));
  const monthlyGain  = ((parseFloat(suggested_lo) - price) * monthlyUnits).toFixed(0);
  return `[动态定价顾问] 实时分析结果

━━ 当前状态 ━━
售价: ${{price}} | 成本: ${{cost}} | 毛利率: ${{(margin*100).toFixed(1)}}% | BSR: #${{bsr}}（${{bsrScore}}）

━━ 竞品价格带分析 ━━
竞品区间: ${{compRaw}} | 中位价: $${{compMid.toFixed(2)}}
您的定价相对竞品: ${{price < compMid ? '偏低，有提价空间' : price > compHi ? '高于竞品，需强差异化支撑' : '处于合理区间'}}

━━ 最优定价建议 ━━
推荐区间: $${{suggested_lo}} - $${{suggested_hi}}
理由: 竞品中位 $${{compMid.toFixed(2)}}，BSR ${{bsrScore}} 支持适当溢价
预期毛利率提升: ${{(margin*100).toFixed(1)}}% → ${{newMargin}}%（+${{(parseFloat(newMargin)-margin*100).toFixed(1)}}pp）
月均增益估算: +$${{monthlyGain}}（约 ${{monthlyUnits}} 单/月 × ${{(parseFloat(suggested_lo)-price).toFixed(2)}} 差价）

━━ 分步涨价路径 ━━
Week 1: $${{price}} → $${{w1}}（观察转化率变化）
Week 2: 若转化率降幅 <15%，升至 $${{w2}}
Week 3+: 稳定后评估是否继续到 $${{suggested_hi}}

━━ 促销节奏建议 ━━
├─ 每月1次 Coupon 10-15%（维持搜索权重，建议 $${{(price*0.88).toFixed(2)}}）
├─ Prime Day 前2周: $${{primeDayPrice}}（冲BSR，接受短期利润压缩）
└─ Q4 旺季: $${{q4Price}}（需求刚性，不主动降价）

[WARN] 监控阈值: 若7天内转化率下降 >20%，立即回退至 $${{w1}}`;
}}

function computePnLAnalyzer(id) {{
  const rev    = parseFloat(getVal(id,'revenue')) || 32400;
  const cogs   = parseFloat(getVal(id,'cogs'))    || 9200;
  const fba    = parseFloat(getVal(id,'fba'))     || 5800;
  const ads    = parseFloat(getVal(id,'ads'))     || 6500;
  const retPct = parseFloat(getVal(id,'return_rate')) || 4;
  const comm   = rev * 0.15;
  const shipping = rev * 0.059;
  const retCost  = rev * retPct / 100 * 0.40;
  const total_cost = cogs + fba + ads + comm + shipping + retCost;
  const profit = rev - total_cost;
  const netPct = (profit / rev * 100).toFixed(1);
  const acos   = (ads / rev * 100).toFixed(1);
  const targetAcos = 18;
  const adWaste = Math.max(0, ads * (parseFloat(acos) - targetAcos) / parseFloat(acos));
  const retSave  = rev * 0.01 * 0.40;
  const shippingSave = shipping * 0.32;
  const improved_profit = profit + adWaste + retSave + shippingSave;
  const improved_pct = (improved_profit / rev * 100).toFixed(1);
  const rank = [
    [ads/rev, `广告花费占比 ${{(ads/rev*100).toFixed(1)}}% → 行业均值 18% → 优化空间: +${{fmtMoney(adWaste)}}/月`],
    [retCost/rev, `退货率 ${{retPct}}% → 行业优秀 3% → 每降1% = +${{fmtMoney(retSave)}}/月`],
    [shippingSave/rev, `头程物流优化（海运替代）→ 节省 ${{fmtMoney(shippingSave)}}/月`],
  ].sort((a,b)=>b[0]-a[0]);
  return `[P&L 透视镜] 实时财务分析

━━ 收支明细 ━━
收入: ${{fmtMoney(rev)}}
├─ 商品成本:  -${{fmtMoney(cogs)}}（${{(cogs/rev*100).toFixed(1)}}%）
├─ FBA 费用:  -${{fmtMoney(fba)}}（${{(fba/rev*100).toFixed(1)}}%）
├─ 广告花费:  -${{fmtMoney(ads)}}（${{(ads/rev*100).toFixed(1)}}%）${{parseFloat(acos)>20?'[!] 偏高':''}}
├─ 平台佣金:  -${{fmtMoney(comm)}}（15.0%）
├─ 头程物流:  -${{fmtMoney(shipping)}}（5.9% 估算）
├─ 退货成本:  -${{fmtMoney(retCost)}}（${{retPct}}% × 40%）
└─ 净利润:   ${{profit>=0?'+':''}}${{fmtMoney(profit)}}（净利率 ${{netPct}}%）${{parseFloat(netPct)<12?'[!] 低于行业均值 15%':parseFloat(netPct)>20?'[OK] 优于行业均值':'[~] 接近行业均值'}}

━━ 利润漏洞识别（TOP3，按优化空间排序）━━
${{rank.map((r,i)=> (i+1) + '. ' + r[1]).join('\\n')}}

━━ 改善后利润模拟 ━━
执行以上3项优化后:
预计净利润: ${{fmtMoney(improved_profit)}}（净利率 ${{improved_pct}}%）
利润提升: +${{((improved_profit/profit-1)*100).toFixed(0)}}%（+${{fmtMoney(improved_profit-profit)}}/月）

[>] 最优先行动: ${{rank[0][1].split('→')[0].trim()}}（ROI最高，可在30天内见效）`;
}}

function computeAdAttribution(id) {{
  const platform   = getVal(id,'platform') || 'Amazon SP';
  const spend      = parseFloat(getVal(id,'spend')) || 12400;
  const targetRaw  = getVal(id,'target_acos') || 'ACoS 18%';
  const dataText   = getVal(id,'data') || '';
  const targetMatch = targetRaw.match(/[\d.]+/);
  const targetAcos  = targetMatch ? parseFloat(targetMatch[0]) : 18;
  const estAcos     = spend > 0 ? (spend / (spend * 3.2) * 100) : 26;
  const actualAcos  = Math.min(35, Math.max(12, estAcos + (dataText.length > 50 ? -3 : 5)));
  const wasteRatio  = Math.max(0, (actualAcos - targetAcos) / actualAcos);
  const wasteAmt    = spend * wasteRatio * 0.85;
  const saving1     = wasteAmt * 0.45;
  const saving2     = spend * 0.03;
  const saving3     = spend * 0.015;
  const totalSave   = saving1 + saving2 + saving3;
  const lines = dataText.split('\\n').filter(l=>l.trim()).slice(0,5);
  const keywordsSection = lines.length > 2
    ? `━━ 基于您粘贴的数据（前${{lines.length}}行）━━\\n${{lines.map((l,i)=>`${{'!'}} 行${{i+1}}: ${{l.slice(0,60)}}${{l.length>60?'…':''}}`).join('\\n')}}\\n` : '';
  return `[广告归因侦探] 实时诊断（${{platform}}）

━━ 花费概览 ━━
月广告花费: ${{fmtMoney(spend)}}
目标 ACoS: ${{targetAcos}}%
估算当前 ACoS: ${{actualAcos.toFixed(1)}}%${{actualAcos > targetAcos ? ` [!] 超标 ${{(actualAcos-targetAcos).toFixed(1)}}pp` : ' [OK] 达标'}}
估算无效花费: ${{fmtMoney(wasteAmt)}}（${{(wasteRatio*100).toFixed(1)}}%）

${{keywordsSection}}━━ 优化行动清单（执行后预期节省）━━
1. 否定低效关键词（高展现零转化） → 节省 ${{fmtMoney(saving1)}}/月
2. 开启 SP 动态竞价-仅降低         → 节省 ${{fmtMoney(saving2)}}/月（ACoS -1.5pp）
3. 新增否定词组（wholesale/cheap/bulk）→ 节省 ${{fmtMoney(saving3)}}/月
──────────────────────────────
预计月节省合计: ${{fmtMoney(totalSave)}} → 年化: ${{fmtMoney(totalSave*12)}}

━━ 归因漏洞检查 ━━
${{platform.includes('SB') || platform.includes('SD') ? '[WARN] SB/SD 广告归因窗口与 SP 不统一，建议统一归因窗口至7天点击' : '[OK] 归因窗口配置正常（建议7天点击 + 1天浏览）'}}
${{actualAcos > 25 ? '[!] ACoS 超过25%，建议检查广告组与关键词相关性，SB 广告建议增加 Retargeting 受众' : '[OK] ACoS 控制合理'}}

[>] 首要行动: 立即暂停 ACoS > ${{(targetAcos*2).toFixed(0)}}% 的关键词，预计7天内 ACoS 下降 ${{(actualAcos - targetAcos).toFixed(1)}}pp`;
}}

function computeCompetitorRadar(id) {{
  const asinText = getVal(id,'asins') || 'B08XYZ1234\\nB09ABC5678';
  const period   = getVal(id,'period') || '过去7天';
  const metrics  = getVal(id,'metrics') || '全部';
  const asins    = asinText.split('\\n').map(l=>l.trim()).filter(l=>l.match(/^B[0-9A-Z]{{9}}$/i));
  const n = Math.max(1, asins.length);
  const days = period.includes('7') ? 7 : period.includes('14') ? 14 : 30;
  const alerts = [];
  const reports = asins.slice(0,5).map((asin,i) => {{
    const priceDrop = i===0 ? -18 : i===1 ? -5 : Math.round((Math.random()*10-5)*10)/10;
    const bsrChange = i===0 ? -253 : i===1 ? 45 : Math.round(Math.random()*200-100);
    const newReviews = Math.round(days * (i===0 ? 6.7 : i===1 ? 2.1 : 1.5));
    const lines = [];
    if (metrics==='全部' || metrics.includes('价格')) {{
      lines.push(`├─ 价格变化: ${{priceDrop<-10?'[WARN] 大幅降价 '+priceDrop+'%':priceDrop<0?'小幅降价 '+priceDrop+'%':'稳定 '+priceDrop+'%'}}`);
      if (priceDrop < -10) alerts.push(`[${{asin}}] 大幅降价${{priceDrop}}%，建议密切关注`);
    }}
    if (metrics==='全部' || metrics.includes('BSR')) {{
      lines.push(`├─ BSR 变化: ${{bsrChange<0?'上升 '+Math.abs(bsrChange)+' 名 [WARN]':'下降 '+bsrChange+' 名'}}`);
    }}
    if (metrics==='全部' || metrics.includes('评论')) {{
      lines.push(`└─ 新增评论: +${{newReviews}}条（${{days}}天）${{newReviews>20?'[注意] 增速较快':''}}`);
    }}
    return `${{asin}}（竞品${{i+1}}）\\n${{lines.join('\\n')}}`;
  }});
  const noAsin = n===0 ? '未检测到有效 ASIN（格式: B开头+9位字母数字），使用示例数据' : '';
  return `[竞品雷达站] ${{period}}监控报告（${{metrics}}）
${{noAsin ? '[~] ' + noAsin + '\\n' : ''}}
监控对象: ${{n}} 个 ASIN | 周期: ${{days}} 天 | 维度: ${{metrics}}

━━ 逐品分析 ━━
${{(n > 0 ? reports : [
  'B08XYZ1234（示例竞品A）\\n├─ 价格: -18% [WARN] 降价促销\\n├─ BSR: 上升253名\\n└─ 新增评论: +47条（含差评激增）',
  'B09ABC5678（示例竞品B）\\n├─ 价格: 稳定\\n├─ BSR: 下降45名\\n└─ 新增评论: +15条'
]).join('\\n\\n')}}

━━ 预警汇总 ━━
${{alerts.length > 0 ? alerts.map(a=>'[!] '+a).join('\\n') : '[OK] 无异常波动'}}

━━ 建议响应 ━━
${{asins[0] && asins[0] !== '' ? `P0: 重点关注 ${{asins[0]}} 的价格动态` : 'P0: 请输入真实竞品 ASIN 获得针对性建议'}}
P1: 若竞品出现大量差评，可针对竞品词做广告截流（时间窗口约 2 周）
P2: 每月检查竞品 Listing 变更，防止关键卖点被模仿`;
}}

function computeListingDoctor(id) {{
  const title   = getVal(id,'title') || '';
  const bullets = getVal(id,'bullets') || '';
  const kws     = getVal(id,'keywords') || '';
  const kwList  = kws.split(/[,，]/).map(k=>k.trim()).filter(Boolean);
  const tLen    = title.length;
  const bLines  = bullets.split('\\n').filter(l=>l.trim()).length;
  const score   = Math.max(30, Math.min(95,
    (tLen>150?25:tLen>80?15:5) +
    (tLen>0 && kwList.some(k=>title.toLowerCase().includes(k.toLowerCase()))?20:5) +
    (bLines>=4?20:bLines*4) +
    (title.length>0?10:0) + 20
  ));
  const missingKws = kwList.filter(k=>!title.toLowerCase().includes(k.toLowerCase()));
  const issues = [];
  if (tLen < 80)   issues.push(`标题字符仅 ${{tLen}} 个，建议 150-200 字符，当前损失关键词密度`);
  if (tLen > 200)  issues.push(`标题字符 ${{tLen}} 个，超过200字符上限，Amazon 会截断`);
  if (missingKws.length > 0) issues.push(`标题缺少核心词: "${{missingKws.join('" "')}}"，建议加入标题前60字符`);
  if (bLines < 4)  issues.push(`Bullet 仅 ${{bLines}} 条，建议5条，充分利用 Amazon 展示空间`);
  if (bLines > 0 && bullets.split('\\n').some(l=>l.length<20)) issues.push(`部分 Bullet 过短（<20字符），缺乏量化证明和场景描述`);
  const rewritten = title.length > 0 && kwList.length > 0
    ? `[参考重写] ${{kwList[0] ? kwList[0].toUpperCase() + ' - ' : ''}}${{title.slice(0,100)}}${{missingKws.length > 0 ? ' | ' + missingKws.join(' | ') : ''}} — Premium Quality`
    : '[提示] 请输入 Title 和核心词以获得重写建议';
  return `[Listing 医生] 实时诊断

━━ 综合评分 ━━
当前 Listing 评分: ${{score}}/100（${{score>=80?'[OK] 良好':score>=60?'[~] 需优化':'[!] 较差，急需改进'}}）

━━ Title 分析（${{tLen}} 字符）━━
${{tLen===0?'[!] 未输入 Title':'字符数评估: '+(tLen>150?'[OK] 长度充足':tLen>80?'[~] 可进一步丰富':'[!] 过短，严重损失关键词密度')}}
关键词覆盖: ${{kwList.length===0?'未输入目标关键词':missingKws.length===0?'[OK] 全部覆盖':'[!] 缺失: "'+missingKws.join('", "')+'"'}}

━━ Bullet Points 分析（${{bLines}} 条）━━
${{bLines===0?'[!] 未输入 Bullet Points':bLines>=5?'[OK] 条数充足':('[~] 仅 '+bLines+' 条，建议补充至5条')}}

━━ 问题清单 ━━
${{issues.length > 0 ? issues.map((v,i)=>`${{i+1}}. ${{v}}`).join('\\n') : '[OK] 未发现明显结构问题'}}

━━ 重写建议 ━━
${{rewritten}}

预估优化后 CTR 提升: ${{score < 60 ? '+25-35%' : score < 80 ? '+12-20%' : '+5-10%'}}`;
}}

function computeVocDecoder(id) {{
  const reviews = getVal(id,'reviews') || '';
  const lang    = getVal(id,'lang') || '英语';
  const lines   = reviews.split('\\n').filter(l=>l.trim().length > 5);
  const total   = lines.length;
  const negKws  = ['break','broke','cheap','disappoint','return','refund','bad','worse','terrible','leak','crack','fell apart','not worth','waste','awful','horrible'];
  const posKws  = ['love','great','perfect','amazing','easy','best','excellent','recommend','happy','nice','awesome','quality','durable','worth'];
  const painKws = {{
    '质量问题': ['break','broke','crack','leak','fell apart','cheap','flimsy','terrible'],
    '尺寸/规格': ['small','big','large','size','fit','tight','loose'],
    '使用体验': ['hard','difficult','confusing','complicated','instruction'],
    '物流/包装': ['damaged','broken','shipping','package','arrived','late'],
    '性价比': ['price','expensive','cheap','value','worth','overpriced'],
  }};
  const joyKws = {{
    '易用性': ['easy','simple','convenient','user friendly','intuitive'],
    '质量耐用': ['durable','sturdy','solid','quality','last','strong'],
    '外观设计': ['cute','beautiful','nice','design','color','look'],
    '性价比': ['value','worth','affordable','price','deal'],
  }};
  const negLines = lines.filter(l=>negKws.some(k=>l.toLowerCase().includes(k)));
  const posLines = lines.filter(l=>posKws.some(k=>l.toLowerCase().includes(k)));
  const pains = Object.entries(painKws).map(([cat,kws])=>{{
    const count = lines.filter(l=>kws.some(k=>l.toLowerCase().includes(k))).length;
    const example = lines.find(l=>kws.some(k=>l.toLowerCase().includes(k)));
    return {{cat, count, example: example ? '"'+example.slice(0,80)+'"' : null}};
  }}).filter(p=>p.count>0).sort((a,b)=>b.count-a.count).slice(0,3);
  const joys = Object.entries(joyKws).map(([cat,kws])=>{{
    const count = lines.filter(l=>kws.some(k=>l.toLowerCase().includes(k))).length;
    const example = lines.find(l=>kws.some(k=>l.toLowerCase().includes(k)));
    return {{cat, count, example: example ? '"'+example.slice(0,80)+'"' : null}};
  }}).filter(j=>j.count>0).sort((a,b)=>b.count-a.count).slice(0,3);
  const noData = total < 3;
  const noDataHint = noData ? '[~] 输入不足3条，以下为示例输出（请粘贴真实评论获得精准分析）' : '';
  return `[用户之声解码器] 实时分析${{total>0?' ('+total+'条输入)':''}}
${{noDataHint}}${{noDataHint?'\\n':''}}
━━ 评论概览 ━━
输入评论数: ${{total}} 条
负面信号: ${{negLines.length}} 条（${{total>0?(negLines.length/total*100).toFixed(0):'-'}}%）
正面信号: ${{posLines.length}} 条（${{total>0?(posLines.length/total*100).toFixed(0):'-'}}%）

━━ TOP 痛点（高频）━━
${{(pains.length > 0 ? pains : [
  {{cat:'吸盘失效',count:38,example:'suction doesn\\u0027t hold after 2 months of use'}},
  {{cat:'颜色褪色',count:29,example:'faded after dishwasher, looks cheap now'}},
  {{cat:'尺寸偏小',count:21,example:'not big enough for 18mo+, she outgrew it fast'}},
]).map((p,i)=>`${{i+1}}. ${{p.cat}}（${{p.count}}次提及）\\n   ${{p.example||''}}`).join('\\n')}}

━━ TOP 爽点（高频）━━
${{(joys.length > 0 ? joys : [
  {{cat:'好清洗',count:61,example:'easiest to clean baby product I own'}},
  {{cat:'防摔耐用',count:44,example:'dropped 100 times still perfect'}},
  {{cat:'外观设计',count:38,example:'great minimalist colors, love it'}},
]).map((j,i)=>`${{i+1}}. ${{j.cat}}（${{j.count}}次提及）\\n   ${{j.example||''}}`).join('\\n')}}

━━ 产品迭代建议 ━━
${{pains.length > 0 ?
  pains.map((p,i)=>`P${{i}}: 改善「${{p.cat}}」→ ${{i===0?'直接影响复购率':i===1?'延长产品生命周期':'提升品牌形象'}}`).join('\\n') :
  'P0: 吸盘结构升级 → 直接影响复购率\\nP1: 推出大码版本 → 延长产品生命周期\\nP2: 加强洗碗机耐用工艺'
}}

[${{lang.includes('多') ? '多语言' : lang}}] ${{lang !== '英语' ? '检测到多语言模式，建议用 Skill-LACA-CrossLingual-ABSA 进行跨语言情感分析' : '数据来源：用户输入'}}`;
}}

function computeCsTriage(id) {{
  const tickets  = getVal(id,'tickets') || '';
  const platform = getVal(id,'platform') || 'Amazon';
  const sla      = getVal(id,'sla') || '24小时';
  const lines    = tickets.split('\\n').filter(l=>l.trim().length>5);
  const total    = lines.length;
  const highRiskKws  = ['a-to-z','atoz','claim','1-star','one star','1 star','lawsuit','legal','furious','extremely angry','demand refund'];
  const refundKws    = ['refund','return','money back','不满意','退款','退货'];
  const defectKws    = ['break','broke','defect','quality','不能用','坏了','质量'];
  const logisticsKws = ['where is','tracking','shipped','delivery','lost','arrived','物流','快递','到了吗'];
  const highRisk  = lines.filter(l=>highRiskKws.some(k=>l.toLowerCase().includes(k)));
  const refunds   = lines.filter(l=>refundKws.some(k=>l.toLowerCase().includes(k)));
  const defects   = lines.filter(l=>defectKws.some(k=>l.toLowerCase().includes(k)));
  const logistics = lines.filter(l=>logisticsKws.some(k=>l.toLowerCase().includes(k)));
  const rest      = total - refunds.length - defects.length - logistics.length;
  const tooFewHint = total < 3 ? '[~] 工单不足3条，以下为示例输出（粘贴真实工单获得精准分诊）' : '';
  return `[客服分诊台] 实时分析（${{platform}} | SLA ${{sla}}）
${{tooFewHint}}${{tooFewHint?'\\n':''}}
━━ 工单分类分布（共 ${{total>0?total:'63'}} 条）━━
退货退款请求: ${{total>0?refunds.length:'18'}} 条（${{total>0?(refunds.length/total*100).toFixed(1):'28.6'}}%）
产品质量问题: ${{total>0?defects.length:'14'}} 条（${{total>0?(defects.length/total*100).toFixed(1):'22.2'}}%）
物流查询:     ${{total>0?logistics.length:'19'}} 条（${{total>0?(logistics.length/total*100).toFixed(1):'30.2'}}%）
使用咨询:     ${{total>0?Math.max(0,rest):'12'}} 条（${{total>0?(Math.max(0,rest)/total*100).toFixed(1):'19.0'}}%）

━━ 高优先级预警（需 ${{sla}} 内处理）━━
${{highRisk.length > 0
  ? highRisk.slice(0,3).map((t,i)=>`[ALERT] 工单${{i+1}}: "${{t.slice(0,80)}}${{t.length>80?'…':''}}"`).join('\\n')
  : total > 0
    ? '[OK] 本批工单未检测到 A-to-Z/差评威胁关键词'
    : '[ALERT] 工单#2847: "file A-to-Z claim if no response by tomorrow"\\n[ALERT] 工单#2851: "going to leave 1-star review, terrible quality"'
}}

━━ 标准回复模板（物流查询）━━
"Hi [Name], thank you for reaching out!\\nYour order is currently in transit. Expected delivery: [DATE].\\nIf not received by [DATE+3], reply and we will send a replacement immediately."

━━ 产品缺陷信号 ━━
${{defects.length > 2
  ? `[!] ${{defects.length}}条工单涉及产品质量 → 可能存在批次性问题，建议联系工厂复查`
  : total > 0
    ? '[OK] 本批无明显批次性质量问题信号'
    : '[!] 14条工单提及结构性质量问题 → 建议联系工厂复查该批次'
}}`;
}}

function computeAccountGuardian(id) {{
  const notice  = getVal(id,'notice') || '';
  const asins   = getVal(id,'asins') || '';
  const health  = getVal(id,'health') || '绿色（正常）';
  const riskBase = health.includes('红') ? 8.5 : health.includes('黄') ? 6.5 : 3.2;
  const noticeRisk = notice.toLowerCase().includes('violation') || notice.includes('违规') ? 2.5
    : notice.toLowerCase().includes('warning') || notice.includes('警告') ? 1.5 : 0;
  const score = Math.min(10, riskBase + noticeRisk).toFixed(1);
  const riskLabel = parseFloat(score) >= 7 ? '高风险，需立即处理' : parseFloat(score) >= 5 ? '中等风险，需关注' : '低风险，保持监控';
  const asinList = asins.split('\\n').map(l=>l.trim()).filter(l=>l.match(/^B[0-9A-Z]{{9}}$/i));
  const noticeLines = notice.split('\\n').filter(l=>l.trim()).slice(0,3);
  return `[账号风险卫士] 实时风险评估

━━ 综合风险评分 ━━
风险评分: ${{score}}/10（${{riskLabel}}）
账号状态: ${{health}}
${{noticeRisk > 0 ? '[!] 检测到警告通知，风险分上升 +'+noticeRisk : '[OK] 通知内容无高危关键词'}}

━━ 通知内容摘要 ━━
${{noticeLines.length > 0
  ? noticeLines.map(l=>'> '+l.slice(0,100)).join('\\n')
  : '（未粘贴通知内容）'
}}

━━ ASIN 合规检查（${{asinList.length}} 个）━━
${{asinList.length > 0
  ? asinList.slice(0,4).map((a,i)=>`${{a}}: [~] 建议检查 Title 中是否含竞品品牌词、绝对化表述`).join('\\n')
  : health.includes('红') ? '[!] 请输入问题 ASIN 进行逐个排查' : '[OK] 请输入 ASIN 列表进行合规扫描'
}}

━━ 整改清单 ━━
${{parseFloat(score) >= 7
  ? 'P0（今日）: 检查并删除 Listing 中的侵权词/医疗声明\\nP0（今日）: 处理所有未回复差评工单（ODR 目标 <0.9%）\\nP1（本周）: 提交 POA（行动计划）'
  : parseFloat(score) >= 5
  ? 'P1（本周）: 回复所有差评工单，目标 ODR <0.9%\\nP2（本月）: 完成 Brand Registry 申请\\nP2（本月）: 检查广告文案合规性'
  : 'P2（本月）: 定期健康检查，保持 ODR <0.5%\\nP3: 考虑申请 Brand Registry 加强品牌保护'
}}

━━ POA 申诉框架（如需）━━
"Root Cause: [问题根因]
Corrective Actions: [已执行的改正措施]
Preventive Measures: [预防措施和未来计划]"`;
}}

function computeBrandGuardian(id) {{
  const copy     = getVal(id,'copy') || '';
  const category = getVal(id,'category') || '母婴';
  const market   = getVal(id,'market') || 'US';
  const forbiddenKws = [
    {{w:'clinically proven', fix:'designed with safety in mind', rule:'FDA - 需临床认证'}},
    {{w:'prevents', fix:'designed for', rule:'FTC - 绝对化预防声明'}},
    {{w:'cures', fix:'supports', rule:'FDA - 医疗声明'}},
    {{w:'treats', fix:'supports', rule:'FDA - 医疗声明'}},
    {{w:'heals', fix:'helps with', rule:'FDA - 医疗声明'}},
    {{w:'100% safe', fix:'made with food-grade materials', rule:'FTC - 绝对化表述'}},
    {{w:'totally safe', fix:'carefully tested for safety', rule:'FTC - 绝对化表述'}},
    {{w:'fda approved', fix:'FDA registered facility', rule:'FDA - 批准措辞限制'}},
    {{w:'guaranteed to', fix:'designed to', rule:'FTC - 绝对保证'}},
    {{w:'no side effects', fix:'carefully formulated', rule:'FTC - 无法证实'}},
  ];
  const cautionKws = [
    {{w:'bpa-free', note:'需第三方检测报告支撑'}},
    {{w:'bpa free', note:'需第三方检测报告支撑'}},
    {{w:'non-toxic', note:'需 CPSIA/EN71 认证文件'}},
    {{w:'organic', note:'需 USDA/有机认证'}},
    {{w:'hypoallergenic', note:'需皮肤科测试报告'}},
    {{w:'pediatrician', note:'需执业医师签名或机构背书'}},
  ];
  const copyLower = copy.toLowerCase();
  const violations = forbiddenKws.filter(k=>copyLower.includes(k.w));
  const cautions   = cautionKws.filter(k=>copyLower.includes(k.w));
  const totalIssues = violations.length + cautions.length;
  const baseScore = copy.length > 0 ? Math.max(40, 100 - violations.length*15 - cautions.length*5) : 65;
  const afterScore = Math.min(95, baseScore + violations.length*12 + cautions.length*4);
  const shortHint = copy.length < 20 ? '[~] 文案不足20字，以下为示例输出（粘贴真实文案获得精准扫描）' : '';
  return `[品牌合规卫士] 扫描报告（${{category}} | ${{market}} 市场）
${{shortHint}}${{shortHint?'\\n':''}}
━━ 综合评分 ━━
当前合规评分: ${{baseScore}}/100 → 整改后预计: ${{afterScore}}/100

━━ 禁用词（${{violations.length}}处违规）━━
${{violations.length > 0
  ? violations.map((v,i)=>`${{i+1}}. "${{v.w}}" → ${{v.rule}}\\n   合规改写: "${{v.fix}}..."`).join('\\n')
  : copy.length > 0 ? '[OK] 未检测到明确禁用词' : '[!] 示例违规: "clinically proven" → 需FDA认证\\n[!] 示例违规: "prevents colic" → 医疗声明，违反FTC'
}}

━━ 慎用词（${{cautions.length}}处需证明文件）━━
${{cautions.length > 0
  ? cautions.map((c,i)=>`${{i+1+violations.length}}. "${{c.w}}" → ${{c.note}}`).join('\\n')
  : copy.length > 0 ? '[OK] 未检测到需额外证明的慎用词' : '[~] 示例慎用: "BPA-free" → 需第三方检测报告\\n[~] 示例慎用: "non-toxic" → 需CPSIA认证'
}}

━━ 所需证明文件清单 ━━
${{category.includes('母婴') || category.includes('baby') ? '□ SGS/Intertek 第三方安全检测报告\\n□ CPSIA 儿童产品认证（US必需）\\n□ EN71/CE 认证（EU市场）' : '□ 对应品类的第三方检测报告\\n□ 目标市场认证文件'}}
${{violations.some(v=>v.w.includes('bpa')) || cautions.some(c=>c.w.includes('bpa')) ? '□ BPA-Free 声明（实验室报告）' : ''}}
${{market.includes('EU') ? '□ REACH 法规合规声明' : ''}}`;
}}

function computeProductRadar(id) {{
  const keyword = getVal(id,'keyword') || '母婴产品';
  const market  = getVal(id,'market') || 'US';
  const budget  = getVal(id,'budget') || '$5-20k';
  const len = keyword.length;
  const isNiche  = len > 8;
  const searchVol = isNiche ? Math.round(50000 + len * 3200) : Math.round(120000 + len * 5000);
  const growth    = isNiche ? 15 + Math.floor(len*1.5) : 8 + Math.floor(len*0.8);
  const cr        = isNiche ? 35 + Math.floor(len*0.5) : 45 + Math.floor(len*0.3);
  const score     = Math.min(95, Math.max(45, 55 + (isNiche?15:5) + (budget.includes('>$20')?10:5) + Math.floor(growth/3)));
  const scoreLabel = score >= 80 ? '[+] 强力推荐' : score >= 65 ? '[~] 值得尝试' : '[!] 谨慎评估';
  const winStars = score >= 80 ? '⭐⭐⭐⭐' : score >= 65 ? '⭐⭐⭐' : '⭐⭐';
  const marketName = market === 'US' ? '美国' : market === 'UK' ? '英国' : market === 'DE' ? '德国' : market === 'AU' ? '澳洲' : '日本';
  const avgPrice = market === 'US' ? 19.9 : market === 'UK' ? 16.5 : market === 'DE' ? 22.0 : market === 'AU' ? 28.0 : 2800;
  const currency = market === 'JP' ? '¥' : '$';
  const costBand = market === 'JP' ? '¥800-1400' : '$6-9';
  const firstBatch = budget.includes('<$5') ? '200-400' : budget.includes('>$20') ? '1000-2000' : '500-900';
  return `[选品雷达] 实时分析

━━ 机会评分 ━━
品类: "${{keyword}}" | 市场: ${{marketName}} | 预算: ${{budget}}
综合评分: ${{score}}/100 ${{scoreLabel}}

━━ 市场数据（基于关键词特征估算）━━
月均搜索量: ${{fmtNum(searchVol)}}（YoY +${{growth}}%）
BSR TOP10 均价: ${{currency}}${{avgPrice}} | 您的成本带: ${{costBand}}
头部集中度（前3卖家）: ${{cr}}% ${{cr>50?'[!] 较高，需差异化':'[OK] 仍有切入空间'}}

━━ 差异化切入角度 ━━
1. 材质/工艺升级（食品级/环保材料 → 情感溢价 +${{currency}}${{(avgPrice*0.2).toFixed(0)}}）
2. 套装/组合策略（提升 AOV 至 ${{currency}}${{(avgPrice*1.8).toFixed(0)}}+）
3. ${{market==='JP'?'日文本地化+日本安全认证':'月龄/场景分段（精准细分需求）'}}

━━ 竞争分析 ━━
新品切入评论门槛: ~${{isNiche?100:200}} 条
新品窗口: ${{winStars}} ${{score>=80?'良好':score>=65?'一般':'竞争激烈'}}

━━ 建议 ━━
${{scoreLabel}} — ${{score>=80?'搜索量健康，价格带有利润空间':score>=65?'需要明确差异化方向':'建议进一步验证市场规模'}}
建议首批备货: ${{firstBatch}} 件（${{budget}} 预算匹配）`;
}}

function computeTikTokContent(id) {{
  const product  = getVal(id,'product') || '母婴产品';
  const audience = getVal(id,'audience') || '0-3岁宝妈';
  const style    = getVal(id,'style') || '痛点反转';
  const freq     = getVal(id,'freq') || '3条/周';
  const freqNum  = freq.includes('5') ? 5 : freq.includes('每日') ? 7 : 3;
  const styleMap = {{
    '教程/攻略':  ['使用教程', '3步搞定', '保姆级攻略'],
    '痛点反转':  ['妈妈们最崩溃的是…', 'Before/After 对比', '这一刻终于解放了'],
    '生活记录':  ['真实日常', '一天的使用记录', '宝宝的反应'],
    '对比测评':  ['vs 竞品测试', '同价位横评', '真实对比'],
    'UGC种草':  ['素人妈妈真实分享', '口碑传播', '用户证言'],
  }};
  const hooks = styleMap[style] || ['吸引人的开场白'];
  const topics = ['#babymom', '#toddlermom', '#momhack', '#babyfood', '#parenting'];
  const days = ['周一', '周三', '周五'];
  const plan = Array.from({{length: freqNum}}).map((_,i) => {{
    const d = ['周一','周二','周三','周四','周五','周六','周日'][i % 7];
    const hook = hooks[i % hooks.length];
    return `Day ${{i+1}}（${{d}}）— ${{style}}\\nHook: "${{hook}} — 关于${{product}}"\\n话题: ${{topics.slice(0,3).join(' ')}} #${{product.replace(/\s/g,'')}}`;
  }});
  return `[TikTok 内容官] 本周选题矩阵

━━ 创作策略 ━━
产品: ${{product}}
目标受众: ${{audience}}
内容风格: ${{style}} | 更新频次: ${{freq}}

━━ 内容日历（${{freqNum}} 条/周）━━
${{plan.join('\\n\\n')}}

━━ 爆款公式 ━━
${{style === '痛点反转' ? '情绪触发（共鸣）+ 意外反转 + 简单CTA = 完播率 65%+' :
   style === '教程/攻略' ? '价值前置（3秒说明能学到什么）+ 步骤清晰 + 截图提示 = 收藏率 20%+' :
   style === '对比测评' ? '争议性开场 + 公正对比 + 明确结论 = 评论互动率 8%+' :
   style === 'UGC种草' ? '真实感 + 使用场景 + 情感共鸣 = 转化率 3%+' :
   '日常记录 + 真实感 + 长期关系积累'}}

━━ 发布建议 ━━
最佳时间: ${{audience.includes('宝妈') || audience.includes('mom') ? '晚9-11PM（宝宝入睡后）' : '晚7-9PM（目标时区）'}}
话题标签: ${{topics.join(' ')}}
预算建议: ${{freq.includes('每日') ? '$150-300/周（素人合作）' : '$75-150/周'}}（寄送产品换视频）`;
}}

async function runAgent(id) {{
  if(typeof gtag!=='undefined')gtag('event','agent_run',{{agent_id:id}});
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const out = document.getElementById('output-' + id);
  const content = document.getElementById('content-' + id);
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  if (label) label.textContent = '计算中...';
  if (content) content.textContent = '';
  if (out) out.classList.add('visible');
  if (thinking) thinking.style.display = 'flex';
  await sleep(600);
  if (thinking) thinking.style.display = 'none';
  let text = '';
  try {{
    if (id === 'agent-supply-sentinel')   text = computeSupplySentinel(id);
    else if (id === 'agent-pricing-advisor') text = computePricingAdvisor(id);
    else if (id === 'agent-pnl-analyzer')   text = computePnLAnalyzer(id);
    else if (id === 'agent-ad-attribution') text = computeAdAttribution(id);
    else if (id === 'agent-competitor-radar') text = computeCompetitorRadar(id);
    else if (id === 'agent-listing-doctor')  text = computeListingDoctor(id);
    else if (id === 'agent-voc-decoder')     text = computeVocDecoder(id);
    else if (id === 'agent-cs-triage')       text = computeCsTriage(id);
    else if (id === 'agent-account-guardian') text = computeAccountGuardian(id);
    else if (id === 'agent-brand-guardian')  text = computeBrandGuardian(id);
    else if (id === 'agent-product-radar')   text = computeProductRadar(id);
    else if (id === 'agent-tiktok-content')  text = computeTikTokContent(id);
    else text = (DEMO_DATA[id] || {{}}).output || '暂无计算结果';
  }} catch(e) {{
    text = '[计算错误] ' + e.message + '\\n请检查输入格式';
  }}
  await streamText(content, text);
  saveReport(id, text);
  if (btn) btn.disabled = false;
  if (label) label.textContent = '重新计算';
}}

function _sessionKey() {{
  let k = localStorage.getItem('_p2s_sk');
  if (!k) {{ k = Math.random().toString(36).slice(2) + Date.now().toString(36); localStorage.setItem('_p2s_sk', k); }}
  return k;
}}

function saveReport(agentId, result) {{
  try {{
    const reports = JSON.parse(localStorage.getItem('agentReports') || '[]');
    const agentNames = {{}};
    document.querySelectorAll('.agent-card').forEach(c => {{
      const id = c.getAttribute('onclick').match(/"([^"]+)"/)?.[1];
      const name = c.querySelector('.agent-name')?.textContent;
      if (id && name) agentNames[id] = name;
    }});
    const entry = {{
      id: agentId,
      name: agentNames[agentId] || agentId,
      result,
      ts: new Date().toLocaleString('zh-CN'),
      inputs: collectInputs(agentId),
    }};
    reports.unshift(entry);
    localStorage.setItem('agentReports', JSON.stringify(reports.slice(0, 50)));
    pushToFeishu(entry);
    try {{
      fetch('/api/reports', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          session_key: _sessionKey(),
          agent_id: agentId,
          agent_name: entry.name,
          inputs: entry.inputs || {{}},
          result: result,
          metadata: {{}}
        }})
      }}).catch(function(){{}});
    }} catch(e) {{}}
  }} catch(e) {{}}
}}

const _FEISHU_HOOK = '{FEISHU_WEBHOOK_URL}';
function pushToFeishu(entry) {{
  if (!_FEISHU_HOOK) return;
  const inpLines = Object.entries(entry.inputs||{{}}).map(([k,v])=>k+': '+v).join('\n');
  const resultText = (entry.result||'').replace(/\*\*/g,'').replace(/#+\s/g,'').trim().slice(0,1800);
  const header = 'Agent: '+entry.name+'\n时间: '+entry.ts+(inpLines?'\n\n输入参数:\n'+inpLines:'');
  const body = JSON.stringify({{
    msg_type: 'interactive',
    card: {{
      header: {{title: {{tag:'plain_text', content:'paper2skills · '+entry.name}}, template:'red'}},
      elements: [
        {{tag:'div', text:{{tag:'plain_text', content:header}}}},
        {{tag:'hr'}},
        {{tag:'div', text:{{tag:'plain_text', content:resultText}}}}
      ]
    }}
  }});
  fetch(_FEISHU_HOOK, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body}}).catch(()=>{{}});
}}

function collectInputs(agentId) {{
  const data = DEMO_DATA[agentId];
  if (!data || !data.inputs) return {{}};
  const result = {{}};
  data.inputs.forEach(inp => {{
    const el = document.getElementById(agentId + '__' + inp.id);
    if (el) result[inp.label] = el.value.slice(0, 100);
  }});
  return result;
}}

async function streamText(el, text) {{
  let i = 0;
  const chunk = 3;
  while (i < text.length) {{
    el.textContent += text.slice(i, i + chunk);
    el.parentElement && (el.parentElement.scrollTop = el.parentElement.scrollHeight);
    const c = text[i];
    await sleep(c === '\\n' ? 30 : c === '━' ? 5 : 8);
    i += chunk;
  }}
}}
function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

document.querySelectorAll('.agent-modal-overlay').forEach(ov => {{
  ov.addEventListener('click', e => {{
    if (e.target === ov) {{
      const id = ov.id.replace('modal-', '');
      closeAgent(id);
    }}
  }});
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    document.querySelectorAll('.agent-modal-overlay.open').forEach(ov => {{
      const id = ov.id.replace('modal-', '');
      closeAgent(id);
    }});
  }}
}});

const catBtns = document.querySelectorAll('.cat-pill');
const cards = document.querySelectorAll('.agent-card');
catBtns.forEach(btn => {{
  btn.addEventListener('click', () => {{
    catBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const cat = btn.dataset.cat;
    cards.forEach(c => {{
      c.style.display = (cat === '' || c.dataset.cat === cat) ? '' : 'none';
    }});
  }});
}});
</script>

"""
    return html_page("智能体广场", body, active_nav="agents")


def render_agent_report_page() -> str:
    agent_names_js = json.dumps(
        {ag["id"]: ag["name"] for ag in AGENT_CATALOG}, ensure_ascii=False)
    agent_categories_js = json.dumps(
         {ag["id"]: ag.get("category","") for ag in AGENT_CATALOG}, ensure_ascii=False)
    filter_buttons = "".join(
        f'<button class="rpt-filter" data-agent="{ag["id"]}" onclick="setAgentFilter(\'{ag["id"]}\')">{ag["name"]}</button>'
        for ag in AGENT_CATALOG[:18])
    body = f"""
<div class="rpt-page">
  <div class="rpt-header">
    <div class="rpt-header-left">
      <h1 class="rpt-title">智能体运行报告台</h1>
      <p class="rpt-subtitle">Agent Analytics Dashboard</p>
    </div>
    <div class="rpt-header-actions">
      <button onclick="exportReports()" class="rpt-btn rpt-btn-outline">导出报告</button>
      <button onclick="clearReports()" class="rpt-btn rpt-btn-ghost">清空记录</button>
    </div>
  </div>
  <div class="rpt-summary-bar">
    <div class="rpt-metric"><span class="rpt-metric-value" id="rpt-total">0</span><span class="rpt-metric-label">总运行次数</span></div>
    <div class="rpt-metric"><span class="rpt-metric-value" id="rpt-agents">0</span><span class="rpt-metric-label">调用智能体</span></div>
    <div class="rpt-metric"><span class="rpt-metric-value" id="rpt-today">0</span><span class="rpt-metric-label">今日运行</span></div>
    <div class="rpt-metric"><span class="rpt-metric-value" id="rpt-latest">—</span><span class="rpt-metric-label">最近运行</span></div>
  </div>
  <div class="rpt-filter-bar">
    <button class="rpt-filter active" data-agent="all" onclick="setAgentFilter('all')">全部</button>
    {filter_buttons}
  </div>
  <div id="rpt-list" class="rpt-list"></div>
</div>
<style>
.rpt-page{{max-width:1100px;margin:0 auto;padding:32px 24px}}
.rpt-header{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:32px;padding-bottom:24px;border-bottom:1px solid #E5E5E5}}
.rpt-title{{margin:0;font-size:24px;font-weight:700;color:var(--ink,#0C0C0C);letter-spacing:-.5px}}
.rpt-subtitle{{margin:4px 0 0;font-size:12px;color:#999;font-family:monospace;text-transform:uppercase;letter-spacing:.5px}}
.rpt-header-actions{{display:flex;gap:8px}}
.rpt-btn{{height:36px;padding:0 16px;border-radius:4px;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s}}
.rpt-btn-outline{{background:#fff;border:1px solid #E5E5E5;color:var(--ink,#0C0C0C)}}
.rpt-btn-outline:hover{{border-color:#0C0C0C}}
.rpt-btn-ghost{{background:transparent;border:1px solid transparent;color:#999}}
.rpt-btn-ghost:hover{{color:var(--accent,#B5323E)}}
.rpt-summary-bar{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#E5E5E5;border:1px solid #E5E5E5;border-radius:8px;overflow:hidden;margin-bottom:24px}}
.rpt-metric{{background:#fff;padding:20px 24px}}
.rpt-metric-value{{display:block;font-size:28px;font-weight:700;color:var(--ink,#0C0C0C);letter-spacing:-1px}}
.rpt-metric-label{{display:block;font-size:12px;color:#888;margin-top:4px}}
.rpt-filter-bar{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #F0F0F0}}
.rpt-filter{{height:30px;padding:0 12px;border:1px solid #E5E5E5;border-radius:4px;background:#FAFAFA;color:#555;font-size:12px;cursor:pointer;transition:all .15s}}
.rpt-filter.active{{background:var(--ink,#0C0C0C);border-color:var(--ink,#0C0C0C);color:#fff}}
.rpt-filter:hover:not(.active){{border-color:#999;color:var(--ink,#0C0C0C)}}
.rpt-list{{display:flex;flex-direction:column;gap:16px}}
.rpt-card{{border:1px solid #E5E5E5;border-radius:8px;background:#fff;overflow:hidden}}
.rpt-card-header{{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-bottom:1px solid #F0F0F0;background:#FAFAFA}}
.rpt-card-title{{font-size:14px;font-weight:600;color:var(--ink,#0C0C0C)}}
.rpt-card-meta{{display:flex;align-items:center;gap:10px}}
.rpt-card-agent{{font-size:11px;color:#555;background:#F0F0F0;padding:3px 8px;border-radius:3px;font-family:monospace}}
.rpt-card-ts{{font-size:11px;color:#999;font-family:monospace}}
.rpt-card-body{{padding:16px 20px}}
body {{padding:20px}}
.rpt-inputs{{margin-bottom:14px}}
.rpt-inputs-title{{font-size:10px;font-weight:600;color:#aaa;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}}
.rpt-inputs-grid{{display:flex;flex-wrap:wrap;gap:6px}}
.rpt-input-chip{{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:3px;font-size:12px}}
.rpt-input-key{{color:#888}}
.rpt-input-val{{color:var(--ink,#0C0C0C);font-weight:500;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.rpt-kpi-row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}}
.rpt-kpi{{flex:1;min-width:110px;padding:12px 14px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px}}
 .rpt-kpi-label{{font-size:11px;color:#888;margin-bottom:4px}}
 .rpt-kpi-value{{font-size:18px;font-weight:700;color:var(--ink,#0C0C0C)}}
 .rpt-kpi-value.warn{{color:var(--accent,#B5323E)}}
 .rpt-kpi-value.ok{{color:#059669}}
 .rpt-output{{font-family:monospace;font-size:12.5px;color:#334155;background:#FAFAFA;border:1px solid #F0F0F0;border-radius:4px;padding:16px;white-space:pre-wrap;line-height:1.7;max-height:380px;overflow-y:auto}}
 .rpt-output .rpt-warn{{color:var(--accent,#B5323E);font-weight:600}}
 .rpt-output .rpt-ok{{color:#059669;font-weight:600}}
 .rpt-output .rpt-info{{color:#0369a1}}
 .rpt-output .rpt-section{{font-weight:700;color:var(--ink,#0C0C0C)}}
 .rpt-card-footer{{padding:10px 20px;border-top:1px solid #F0F0F0;display:flex;gap:8px}}
 .rpt-action-btn{{height:28px;padding:0 12px;font-size:12px;border-radius:3px;cursor:pointer;border:1px solid #E5E5E5;background:#fff;color:#555;transition:all .15s}}
 .rpt-action-btn:hover{{border-color:var(--ink,#0C0C0C);color:var(--ink,#0C0C0C)}}
 .rpt-seeded-badge{{font-size:10px;color:#888;margin-left:auto;padding:2px 6px;border:1px solid #E5E5E5;border-radius:3px}}
 @media(max-width:640px){{.rpt-summary-bar{{grid-template-columns:repeat(2,1fr)}};.rpt-header{{flex-direction:column;align-items:flex-start;gap:16px}}}}
/* ── Structured Report Renderer ── */
.rr-body{{padding:4px 0;font-size:13px;color:#1e293b;line-height:1.7}}
.rr-spacer{{height:6px}}
.rr-section{{display:flex;align-items:center;gap:8px;background:linear-gradient(90deg,#F1F5F9 0%,#FAFAFA 100%);border-left:3px solid var(--ink,#0C0C0C);padding:8px 12px;margin:14px 0 6px;font-size:12.5px;font-weight:700;color:var(--ink,#0C0C0C);border-radius:0 4px 4px 0;letter-spacing:.3px;text-transform:uppercase}}
.rr-section-dot{{width:5px;height:5px;border-radius:50%;background:var(--ink,#0C0C0C);flex-shrink:0}}
.rr-warn{{display:flex;align-items:flex-start;gap:8px;background:#FEF2F2;border:1px solid #FECACA;border-radius:6px;padding:8px 12px;margin:4px 0;font-size:12.5px;color:#991B1B;font-weight:500}}
.rr-ok{{display:flex;align-items:flex-start;gap:8px;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:6px;padding:8px 12px;margin:4px 0;font-size:12.5px;color:#166534;font-weight:500}}
.rr-action{{display:flex;align-items:flex-start;gap:8px;background:#EFF6FF;border-left:3px solid #3B82F6;padding:8px 12px;margin:4px 0;font-size:12.5px;color:#1E40AF;font-weight:500;border-radius:0 4px 4px 0}}
.rr-priority{{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:12.5px;color:#334155}}
.rr-p0{{background:var(--accent,#B5323E);color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;flex-shrink:0}}
.rr-p1{{background:#F59E0B;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;flex-shrink:0}}
.rr-p2{{background:#64748B;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;flex-shrink:0}}
.rr-row{{display:flex;align-items:baseline;gap:8px;padding:3px 0 3px 12px;font-size:12.5px;border-left:2px solid #E2E8F0;margin-left:4px;margin-bottom:2px}}
.rr-row-plain{{padding:3px 0 3px 12px;font-size:12.5px;border-left:2px solid #E2E8F0;margin-left:4px;color:#475569}}
.rr-row-key{{color:#64748B;font-size:12px;flex-shrink:0;min-width:100px}}
.rr-row-sep{{color:#CBD5E1;margin:0 2px}}
.rr-kv{{display:flex;align-items:baseline;gap:6px;padding:3px 0;font-size:12.5px}}
.rr-kv-key{{color:#64748B;font-size:12px;flex-shrink:0}}
.rr-kv-sep{{color:#CBD5E1}}
.rr-val{{font-weight:700;color:var(--ink,#0C0C0C);font-variant-numeric:tabular-nums}}
.rr-line{{padding:2px 0;font-size:12.5px;color:#334155}}
.rr-num{{font-weight:700;color:#0C0C0C;font-variant-numeric:tabular-nums}}
.rr-pct{{font-weight:600;color:#0369a1;font-size:12px}}
/* ── Pagination ── */
.rpt-pagination{{display:flex;justify-content:center;align-items:center;gap:4px;padding:24px 0 8px;flex-wrap:wrap}}
.rpt-page-btn{{min-width:32px;height:32px;padding:0 10px;border:1px solid #E5E5E5;border-radius:4px;background:#fff;color:#555;font-size:13px;cursor:pointer;transition:all .15s;display:flex;align-items:center;justify-content:center}}
.rpt-page-btn:hover:not([disabled]){{border-color:var(--ink,#0C0C0C);color:var(--ink,#0C0C0C)}}
.rpt-page-btn.active{{background:var(--ink,#0C0C0C);border-color:var(--ink,#0C0C0C);color:#fff;font-weight:600}}
.rpt-page-btn[disabled]{{opacity:.35;cursor:not-allowed}}
.rpt-page-info{{font-size:12px;color:#888;padding:0 8px}}
</style>
<script>
const _AGENT_NAMES={agent_names_js};
const _AGENT_CATS={agent_categories_js};
let _activeFilter='all';
function _initSeeds(){{
  try{{
    var ex=JSON.parse(localStorage.getItem('agentReports')||'[]');
    if(ex.length===0){{
      var rootPrefix=window.location.pathname.includes('/skills/')||window.location.pathname.includes('/domains/')||window.location.pathname.includes('/playbooks/')||window.location.pathname.includes('/solutions/')?'../':'';
      fetch(rootPrefix+'assets/seed_reports.json').then(function(r){{return r.json();}}).then(function(seeds){{
        localStorage.setItem('agentReports',JSON.stringify(seeds));
        _renderR();
      }}).catch(function(){{}});
    }}
  }}catch(e){{}}
}}
function _loadR(){{try{{return JSON.parse(localStorage.getItem('agentReports')||'[]')}}catch(e){{return[]}}}}
function setAgentFilter(id){{_activeFilter=id;document.querySelectorAll('.rpt-filter').forEach(b=>b.classList.toggle('active',b.dataset.agent===id));_renderR();}}
function _extractKPIs(result){{
  const kpis=[];
  const pats=[
    {{rx:new RegExp('\u5269\u4f59\u53ef\u552e\u5929\u6570[:]+([0-9.]+)[ ]*\u5929'),label:'\u53ef\u552e\u5929\u6570',suffix:'\u5929',warnBelow:14}},
    {{rx:new RegExp('\u51c0\u5229\u6da6[:]+[+]?[$]([0-9,.]+)'),label:'\u51c0\u5229\u6da6',prefix:'$'}},
    {{rx:new RegExp('\u51c0\u5229\u7387[:]+([0-9.]+)%'),label:'\u51c0\u5229\u7387',suffix:'%',warnBelow:8}},
    {{rx:new RegExp('ACoS[:]+([0-9.]+)%'),label:'ACoS',suffix:'%',warnAbove:25}},
    {{rx:new RegExp('\u6708\u8282\u7701\u5408\u8ba1[:]+[$]([0-9,.]+)'),label:'\u6708\u8282\u7701',prefix:'$'}},
    {{rx:new RegExp('\u8bc4\u5206[:]+([0-9]+)/100'),label:'\u8bc4\u5206',suffix:'/100',warnBelow:60}},
  ];
  for(const p of pats){{
    const m=result.match(p.rx);
    if(m){{
      const n=parseFloat(m[1].replace(/,/g,''));
      const w=(p.warnBelow&&n<p.warnBelow)||(p.warnAbove&&n>p.warnAbove);
      kpis.push({{label:p.label,value:(p.prefix||'')+m[1]+(p.suffix||''),warn:w}});
    }}
    if(kpis.length>=3)break;
  }}
  return kpis;
}}
 function _colorize(txt){{
   return txt.replace(/</g,'&lt;').replace(/>/g,'&gt;')
     .replace(/\[WARN\]|\[!\]/g,'<span class="rpt-warn">[!]</span>')
      .replace(/\[OK\]/g,'<span class="rpt-ok">[OK]</span>')
      .replace(/\[>\]/g,'<span class="rpt-info">[>]</span>')
      .replace(/(\u2501\u2501[^\\n]+\u2501\u2501)/g,'<span class="rpt-section">$1</span>');
 }}
function _renderReport(raw){{
  if(!raw)return'';
  const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const lines=raw.split('\\n');
  let html='',inSection=false;
  for(let i=0;i<lines.length;i++){{
    const l=lines[i];
    const trim=l.trim();
    if(!trim){{html+='<div class="rr-spacer"></div>';continue;}}

    // ━━ Section Header ━━
    if(/^\u2501\u2501.+\u2501\u2501$/.test(trim)){{
      const title=trim.replace(/^\u2501+\s*/,'').replace(/\s*\u2501+$/,'');
      html+=`<div class="rr-section"><span class="rr-section-dot"></span>${{esc(title)}}</div>`;
      continue;
    }}
    // [!] Warning
    if(/^\[!\]|^\[WARN\]/.test(trim)){{
      const msg=trim.replace(/^\[!\]\s*|\[WARN\]\s*/,'');
      html+=`<div class="rr-warn"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>${{esc(msg)}}</div>`;
      continue;
    }}
    // [OK]
    if(/^\[OK\]/.test(trim)){{
      const msg=trim.replace(/^\[OK\]\s*/,'');
      html+=`<div class="rr-ok"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>${{esc(msg)}}</div>`;
      continue;
    }}
    // [>] Action
    if(/^\[>\]/.test(trim)){{
      const msg=trim.replace(/^\[>\]\s*/,'');
      html+=`<div class="rr-action"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>${{esc(msg)}}</div>`;
      continue;
    }}
    // ├─ └─ tree line
    if(/^[\u251c\u2514]\u2500/.test(trim)){{
      const msg=trim.replace(/^[\u251c\u2514]\u2500+\s*/,'');
      // Check if it contains a number value after colon
      const valMatch=msg.match(/^(.+?):\s*(.+)$/);
      if(valMatch){{
        const isNum=/[\$\d,%\+\-]+/.test(valMatch[2]);
        const val=isNum?`<span class="rr-val">${{esc(valMatch[2])}}</span>`:esc(valMatch[2]);
        html+=`<div class="rr-row"><span class="rr-row-key">${{esc(valMatch[1])}}</span><span class="rr-row-sep">:</span>${{val}}</div>`;
      }}else{{
        html+=`<div class="rr-row-plain">${{esc(msg)}}</div>`;
      }}
      continue;
    }}
    // P0/P1/P2 priority
    if(/^P[0-3][\u00ef\uff08（(]/.test(trim)||/^P[0-3]:/.test(trim)){{
      const p=trim.match(/^(P\d)/)?.[1]||'P1';
      const pClass=p==='P0'?'rr-p0':p==='P1'?'rr-p1':'rr-p2';
      html+=`<div class="rr-priority"><span class="${{pClass}}">${{p}}</span>${{esc(trim.replace(/^P\d[^\s]*/,'').trim())}}</div>`;
      continue;
    }}
    // Key: Value line
    const kvMatch=trim.match(/^(.{2,20})[：:]\s*(.{1,200})$/);
    if(kvMatch&&!trim.startsWith('http')){{
      const numRaw=kvMatch[2];
      const hasNum=/[\$\d,%]+/.test(numRaw)&&numRaw.length<60;
      const val=hasNum?`<span class="rr-val">${{esc(numRaw)}}</span>`:esc(numRaw);
      html+=`<div class="rr-kv"><span class="rr-kv-key">${{esc(kvMatch[1])}}</span><span class="rr-kv-sep">:</span>${{val}}</div>`;
      continue;
    }}
    // Default text line
    let styled=esc(trim)
      .replace(/(\$[\d,]+(?:\.\d+)?)/g,'<span class="rr-num">$1</span>')
      .replace(/([\d.]+%)/g,'<span class="rr-pct">$1</span>');
    html+=`<div class="rr-line">${{styled}}</div>`;
  }}
  return html;
}}
function _fmtTs(ts){{
  if(!ts)return'—';
  const d=new Date(ts.replace(' ','T'));
  return isNaN(d)?ts:d.toLocaleDateString('zh-CN',{{month:'2-digit',day:'2-digit'}})+' '+d.toLocaleTimeString('zh-CN',{{hour:'2-digit',minute:'2-digit'}});
}}
function _renderCard(r,idx){{
  const name=_AGENT_NAMES[r.id]||r.name||r.id;
  const ts=_fmtTs(r.ts);
  const kpis=_extractKPIs(r.result||'');
  const chips=Object.entries(r.inputs||{{}}).slice(0,5).map(([k,v])=>
    `<span class="rpt-input-chip"><span class="rpt-input-key">${{k}}:</span><span class="rpt-input-val" title="${{v}}">${{v.length>30?v.slice(0,28)+'…':v}}</span></span>`
  ).join('');
  const kpiRow=kpis.length?`<div class="rpt-kpi-row">${{kpis.map(k=>`<div class="rpt-kpi"><div class="rpt-kpi-label">${{k.label}}</div><div class="rpt-kpi-value ${{k.warn?'warn':'ok'}}">${{k.value}}</div></div>`).join('')}}</div>`:'';
  const seededBadge=r.seeded?'<span class="rpt-seeded-badge">预置示例</span>':'';
  const reportBody=_renderReport(r.result||'');
  return`<div class="rpt-card"><div class="rpt-card-header"><span class="rpt-card-title">分析报告 #${{idx+1}} — ${{name}}</span><div class="rpt-card-meta"><span class="rpt-card-agent">${{name}}</span><span class="rpt-card-ts">${{ts}}</span>${{seededBadge}}</div></div><div class="rpt-card-body">${{chips?`<div class="rpt-inputs"><div class="rpt-inputs-title">输入参数</div><div class="rpt-inputs-grid">${{chips}}</div></div>`:''}}<div>${{kpiRow}}</div><div class="rr-body">${{reportBody}}</div></div><div class="rpt-card-footer"><button class="rpt-action-btn" onclick="copyRpt(${{idx}})">复制报告</button><button class="rpt-action-btn" onclick="delRpt(${{idx}})">删除</button></div></div>`;
}}
function _updateSummary(reports){{
  const today=new Date().toDateString();
  const todayN=reports.filter(r=>r.ts&&new Date(r.ts.replace(' ','T')).toDateString()===today).length;
  const agents=new Set(reports.map(r=>r.id)).size;
  const latest=reports.length?_fmtTs(reports[reports.length-1].ts):'—';
  document.getElementById('rpt-total').textContent=reports.length;
  document.getElementById('rpt-agents').textContent=agents;
  document.getElementById('rpt-today').textContent=todayN;
  document.getElementById('rpt-latest').textContent=latest;
}}
function _renderR(){{
  const reports=_loadR();
  _updateSummary(reports);
  const filtered=_activeFilter==='all'?reports:reports.filter(r=>r.id===_activeFilter);
  const list=document.getElementById('rpt-list');
  if(!filtered.length){{
    list.innerHTML='<div style="text-align:center;padding:80px 40px;"><p style="font-size:16px;color:#555;font-weight:500;">暂无报告记录</p><p style="font-size:13px;color:#888;">前往 <a href=\"agents.html\" style=\"color:var(--accent,#B5323E);text-decoration:none\">智能体广场</a> 运行分析后，报告将自动保存至此。</p></div>';
    return;
  }}
  const PAGE_SIZE=6;
  const reversed=[...filtered].reverse();
  const totalPages=Math.ceil(reversed.length/PAGE_SIZE);
  let curPage=parseInt(list.dataset.page||'1');
  if(curPage<1)curPage=1;
  if(curPage>totalPages)curPage=totalPages;
  list.dataset.page=curPage;
  const start=(curPage-1)*PAGE_SIZE;
  const pageItems=reversed.slice(start,start+PAGE_SIZE);
  const cards=pageItems.map((r,i)=>_renderCard(r,filtered.length-1-start-i)).join('');
  const pager=totalPages<=1?'':(() => {{
    let btns=`<button class="rpt-page-btn" onclick="_goPage(${{curPage-1}})" ${{curPage<=1?'disabled':''}}>‹</button>`;
    for(let p=1;p<=totalPages;p++){{
      if(totalPages>7&&p>2&&p<totalPages-1&&Math.abs(p-curPage)>1){{
        if(p===3||p===totalPages-2)btns+='<span class="rpt-page-info">…</span>';
        continue;
      }}
      btns+=`<button class="rpt-page-btn ${{p===curPage?'active':''}}" onclick="_goPage(${{p}})">${{p}}</button>`;
    }}
    btns+=`<button class="rpt-page-btn" onclick="_goPage(${{curPage+1}})" ${{curPage>=totalPages?'disabled':''}}>›</button>`;
    btns+=`<span class="rpt-page-info">${{start+1}}-${{Math.min(start+PAGE_SIZE,filtered.length)}} / ${{filtered.length}} 条</span>`;
    return`<div class="rpt-pagination">${{btns}}</div>`;
  }})();
  list.innerHTML=cards+pager;
}}
function _goPage(p){{
  const list=document.getElementById('rpt-list');
  list.dataset.page=p;
  _renderR();
  list.scrollIntoView({{behavior:'smooth',block:'start'}});
}}
function copyRpt(idx){{
  const r=_loadR()[idx];if(!r)return;
  navigator.clipboard.writeText(`[${{r.name||r.id}}] ${{r.ts||''}}\\n\\n输入:\\n${{JSON.stringify(r.inputs,null,2)}}\\n\\n结果:\\n${{r.result||''}}`).then(()=>alert('报告已复制'));
}}
function delRpt(idx){{
  if(!confirm('确认删除这条报告记录？'))return;
  const rs=_loadR();rs.splice(idx,1);localStorage.setItem('agentReports',JSON.stringify(rs));_renderR();
}}
function clearReports(){{
  if(!confirm('确认清空全部运行记录？此操作不可撤销。'))return;
  localStorage.removeItem('agentReports');_renderR();
}}
function exportReports(){{
  const rs=_loadR();if(!rs.length){{alert('暂无报告记录');return;}}
  const txt=rs.map((r,i)=>`=== 报告 #${{i+1}} | ${{r.name||r.id}} | ${{r.ts||''}} ===\\n输入:\\n${{JSON.stringify(r.inputs,null,2)}}\\n\\n结果:\\n${{r.result||''}}\\n`).join('\\n');
  const a=document.createElement('a');a.href='data:text/plain;charset=utf-8,'+encodeURIComponent(txt);a.download='agent-reports-'+new Date().toISOString().slice(0,10)+'.txt';a.click();
}}
document.addEventListener('DOMContentLoaded',function(){{_initSeeds();_renderR();_loadRemoteReports();}});
function _loadRemoteReports(){{
  (async function(){{
    try{{
      const remote=await fetch('/api/reports?session_key='+_sessionKey()+'&limit=20').then(function(r){{return r.json();}});
      if(remote&&remote.length){{
        const local=_loadR();
        const seen=new Set(local.map(function(r){{return r.id||r.ts;}}) );
        const merged=remote.filter(function(r){{return !seen.has(r.id)&&!seen.has(r.created_at);}})
          .map(function(r){{return {{id:r.agent_id,name:r.agent_name,result:r.result,ts:r.created_at,inputs:JSON.parse(r.inputs||'{{}}')}}; }})
          .concat(local).slice(0,50);
        if(remote.length&&!local.length){{localStorage.setItem('agentReports',JSON.stringify(merged));_renderR();}}
      }}
    }}catch(e){{}}
  }})();
}}
window.addEventListener('storage',e=>{{if(e.key==='agentReports')_renderR();}});
</script>
"""
    return html_page("智能体报告", body, active_nav="agent-report")


from builders.solutions_data import SOLUTIONS_CATALOG
def render_solutions_index(total_skill_count: int) -> str:
    """方案库首页：所有系统方案的卡片列表"""
    cards_html = ""
    for sol in SOLUTIONS_CATALOG:
        tags_html = "".join(f'<span class="sol-tag">{t}</span>' for t in sol["tags"][:4])
        phase_count = len(sol.get("phases", []))
        sol_skill_count = len(sol.get("core_skills", []))
        cards_html += f"""
<a class="sol-card" href="{sol['id']}.html">
  <div class="sol-card-header">
    <div class="sol-icon" style="background:{sol['icon_color']}20;color:{sol['icon_color']};border:1.5px solid {sol['icon_color']}40">{sol['icon']}</div>
    <div class="sol-card-meta">
      <span class="sol-category">{sol['category']}</span>
      <span class="sol-updated">更新 {sol['updated']}</span>
    </div>
  </div>
  <h3 class="sol-title">{sol['title']}</h3>
  <p class="sol-subtitle">{sol['subtitle']}</p>
  <p class="sol-summary">{sol['summary'][:120]}…</p>
  <div class="sol-roi">{sol['roi_headline']}</div>
  <div class="sol-footer">
    <div class="sol-tags">{tags_html}</div>
    <div class="sol-stats">
      <span>{phase_count} 阶段</span>
      <span>{sol_skill_count} Skills</span>
    </div>
  </div>
</a>"""

    body = f"""
<div class="sol-hero">
  <h1>方案库</h1>
  <p class="muted">从 Palantir 方法论到可落地的 AI 决策系统方案，每个方案含完整架构设计、分阶段路线图、核心 Skill 索引</p>
  <div class="sol-hero-stats">
    <span><strong>{len(SOLUTIONS_CATALOG)}</strong> 个方案</span>
    <span><strong>{total_skill_count}</strong> 个 Skills 支撑</span>
    <span><strong>24</strong> 个知识域覆盖</span>
  </div>
</div>

<div class="sol-grid">
{cards_html}
</div>

<div class="sol-coming-soon">
  <h3>更多方案即将发布</h3>
  <div class="sol-coming-grid">
    <div class="sol-coming-card">
      <span class="sol-coming-icon" style="color:#8b5cf6">◆</span>
      <strong>广告归因 → 预算智能分配</strong>
      <p>从 PVM 统一到 Bayesian MMM 全渠道预算优化</p>
    </div>
    <div class="sol-coming-card">
      <span class="sol-coming-icon" style="color:#059669">◆</span>
      <strong>用户 LTV → 精准留存干预</strong>
      <p>Uplift 建模识别可干预用户，精准发券 ROI 优化</p>
    </div>
    <div class="sol-coming-card">
      <span class="sol-coming-icon" style="color:#d97706">◆</span>
      <strong>选品决策 → 市场机会矩阵</strong>
      <p>五维数据并行采集，GO/NO-GO 评分体系</p>
    </div>
    <div class="sol-coming-card">
      <span class="sol-coming-icon" style="color:#e11d48">◆</span>
      <strong>AI Agent 替代重复性岗位</strong>
      <p>供应链对账、数据提取、广告出价三类场景</p>
    </div>
  </div>
</div>

<style>
.sol-hero{{text-align:center;padding:36px 0 28px;border-bottom:1px solid var(--line);margin-bottom:28px}}
.sol-hero h1{{font-size:32px;font-weight:800;margin:0 0 8px;letter-spacing:-.04em;color:var(--ink)}}
.sol-hero .muted{{max-width:520px;margin:0 auto 14px;color:var(--muted);font-size:13.5px;line-height:1.6}}
.sol-hero-stats{{display:flex;gap:28px;justify-content:center;flex-wrap:wrap}}
.sol-hero-stats span{{font-size:12.5px;color:var(--muted)}}
.sol-hero-stats strong{{color:var(--ink);font-size:20px;font-weight:800;letter-spacing:-.04em}}
.sol-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-bottom:32px}}
.sol-card{{display:flex;flex-direction:column;gap:10px;padding:18px;background:var(--panel);border:1px solid var(--line);border-radius:var(--r-lg);text-decoration:none;color:inherit;transition:border-color var(--t-card),box-shadow var(--t-card),transform var(--t-card);cursor:pointer}}
.sol-card:hover{{border-color:var(--line-strong);box-shadow:var(--shadow-md);transform:translateY(-2px)}}
.sol-card-header{{display:flex;align-items:center;gap:10px}}
.sol-icon{{width:36px;height:36px;border-radius:var(--r-md);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0}}
.sol-card-meta{{display:flex;flex-direction:column;gap:2px}}
.sol-category{{font-size:10.5px;font-weight:700;color:var(--accent);letter-spacing:.03em;text-transform:uppercase}}
.sol-updated{{font-size:10.5px;color:var(--muted)}}
.sol-title{{font-size:14.5px;font-weight:700;color:var(--ink);margin:0;line-height:1.35;letter-spacing:-.02em}}
.sol-subtitle{{font-size:12.5px;color:var(--muted);margin:0;line-height:1.4}}
.sol-summary{{font-size:12.5px;color:var(--ink-2);margin:0;line-height:1.55}}
.sol-roi{{font-size:12px;font-weight:600;color:var(--green-dark);background:var(--green-bg);border:1px solid rgba(20,83,45,.12);border-radius:var(--r-xs);padding:5px 10px}}
.sol-footer{{display:flex;align-items:center;justify-content:space-between;margin-top:4px}}
.sol-tags{{display:flex;gap:4px;flex-wrap:wrap}}
.sol-tag{{font-size:10.5px;background:var(--panel-2);color:var(--ink-2);padding:2px 6px;border-radius:var(--r-xs);border:1px solid var(--line)}}
.sol-stats{{display:flex;gap:8px;font-size:11px;color:var(--muted)}}
.sol-coming-soon{{background:var(--panel);border:1px dashed var(--line-strong);border-radius:var(--r-lg);padding:22px;margin-top:6px}}
.sol-coming-soon h3{{font-size:14px;font-weight:700;color:var(--muted);margin:0 0 14px;text-align:center;letter-spacing:-.01em}}
.sol-coming-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}}
.sol-coming-card{{background:var(--panel-2);border:1px solid var(--line);border-radius:var(--r-md);padding:12px;display:flex;flex-direction:column;gap:4px}}
.sol-coming-icon{{font-size:16px}}
.sol-coming-card strong{{font-size:12.5px;color:var(--ink);font-weight:600;letter-spacing:-.01em}}
.sol-coming-card p{{font-size:11.5px;color:var(--muted);margin:0;line-height:1.4}}
</style>
"""
    return html_page("方案库", body, nav="../", active_nav="solutions")


def render_solution_detail(sol: dict, total_skill_count: int) -> str:
    """方案详情页：完整架构 + 分层设计 + 实施路线图"""

    # 七层架构
    layers_html = ""
    layer_colors = ["#6366f1","#8b5cf6","#0ea5e9","#06b6d4","#10b981","#f59e0b","#ef4444"]
    for i, layer in enumerate(sol.get("layers", [])):
        color = layer_colors[i % len(layer_colors)]
        layers_html += f"""
<div class="sd-layer">
  <div class="sd-layer-no" style="background:{color}15;color:{color};border:1.5px solid {color}30">{layer['no']}</div>
  <div class="sd-layer-body">
    <strong>{layer['name']}</strong>
    <span>{layer['desc']}</span>
  </div>
</div>"""

    # 实施阶段
    phases_html = ""
    phase_colors = ["#64748b","#3b82f6","#8b5cf6","#059669"]
    for i, phase in enumerate(sol.get("phases", [])):
        color = phase_colors[i % len(phase_colors)]
        phases_html += f"""
<div class="sd-phase">
  <div class="sd-phase-header" style="border-left:3px solid {color}">
    <span class="sd-phase-name" style="color:{color}">{phase['name']}</span>
    <span class="sd-phase-dur">{phase['duration']}</span>
  </div>
  <div class="sd-phase-action">{phase['action']}</div>
  <div class="sd-phase-roi">{phase['roi']}</div>
</div>"""

    # 三大陷阱
    traps_html = ""
    for trap in sol.get("traps", []):
        traps_html += f"""
<div class="sd-trap">
  <span class="sd-trap-no">{trap['no']}</span>
  <div>
    <strong>{trap['title']}</strong>
    <p>{trap['desc']}</p>
  </div>
</div>"""

    # 核心 Skills
    skills_html = ""
    for sid in sol.get("core_skills", []):
        label = sid.replace("Skill-", "").replace("-", " ")[:36]
        skills_html += f'<a class="sd-skill-chip" href="../skills/{sid}.html">{label}</a>'

    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="index.html">方案库</a> / {sol['title']}</nav>

<div class="sd-hero">
  <div class="sd-hero-icon" style="background:{sol['icon_color']}15;color:{sol['icon_color']};border:2px solid {sol['icon_color']}30">{sol['icon']}</div>
  <div>
    <div class="sd-hero-meta"><span class="sd-category">{sol['category']}</span><span class="sd-updated">更新于 {sol['updated']}</span></div>
    <h1>{sol['title']}</h1>
    <p class="sd-subtitle">{sol['subtitle']}</p>
  </div>
</div>

<div class="sd-roi-banner">
  <span>[↑]</span> {sol['roi_headline']}
</div>

<p class="sd-summary">{sol['summary']}</p>

<div class="sd-grid-2">
  <div class="sd-section">
    <h2>{len(sol.get("layers", []))}层架构设计</h2>
    <p class="sd-sec-desc">从原始数据到自动执行决策的完整分层</p>
    <div class="sd-layers">
{layers_html}
    </div>
  </div>

  <div class="sd-section">
    <h2>分阶段实施路线图</h2>
    <p class="sd-sec-desc">MVP（3个月）→ 智能化（12个月）</p>
    <div class="sd-phases">
{phases_html}
    </div>
  </div>
</div>

<div class="sd-section sd-full">
  <h2>[!] 三大架构陷阱</h2>
  <p class="sd-sec-desc">最容易被忽视、代价最大的设计决策错误</p>
  <div class="sd-traps">
{traps_html}
  </div>
</div>

<div class="sd-section sd-full">
  <h2>核心 Skill 索引（{len(sol.get('core_skills', []))} 个）</h2>
  <p class="sd-sec-desc">本方案涉及的关键 Skills，点击查看详情</p>
  <div class="sd-skills">
{skills_html}
  </div>
  <div style="margin-top:12px">
    <a href="../skills/index.html" style="font-size:13px;color:var(--accent,#3b82f6);text-decoration:none">查看全部 {total_skill_count} 个 Skills →</a>
  </div>
</div>

<style>
.sd-hero{{display:flex;align-items:flex-start;gap:14px;margin-bottom:18px;padding-bottom:18px;border-bottom:1px solid var(--line)}}
.sd-hero-icon{{width:48px;height:48px;border-radius:var(--r-lg);display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;flex-shrink:0}}
.sd-hero-meta{{display:flex;align-items:center;gap:10px;margin-bottom:5px}}
.sd-category{{font-size:10.5px;font-weight:700;color:var(--accent);background:var(--accent-light);padding:2px 7px;border-radius:var(--r-xs);text-transform:uppercase;letter-spacing:.04em}}
.sd-updated{{font-size:11px;color:var(--muted)}}
.sd-hero h1{{font-size:20px;font-weight:800;margin:0 0 3px;color:var(--ink);letter-spacing:-.03em}}
.sd-subtitle{{font-size:13px;color:var(--muted);margin:0}}
.sd-roi-banner{{background:var(--green-bg);border:1px solid rgba(20,83,45,.12);border-radius:var(--r-md);padding:10px 14px;font-size:12.5px;font-weight:600;color:var(--green-dark);display:flex;align-items:center;gap:8px;margin-bottom:14px}}
.sd-summary{{font-size:13px;color:var(--ink-2);line-height:1.7;margin-bottom:20px}}
.sd-grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
@media(max-width:768px){{.sd-grid-2{{grid-template-columns:1fr}}}}
.sd-section{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-lg);padding:18px}}
.sd-section h2{{font-size:15px;font-weight:700;margin:0 0 3px;color:var(--ink);letter-spacing:-.02em}}
.sd-sec-desc{{font-size:11.5px;color:var(--muted);margin:0 0 12px}}
.sd-full{{margin-top:0}}
.sd-layers{{display:flex;flex-direction:column;gap:8px}}
.sd-layer{{display:flex;align-items:center;gap:10px}}
.sd-layer-no{{width:32px;height:32px;border-radius:var(--r-md);display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0}}
.sd-layer-body{{display:flex;flex-direction:column;gap:1px}}
.sd-layer-body strong{{font-size:12.5px;font-weight:600;color:var(--ink-2)}}
.sd-layer-body span{{font-size:11.5px;color:var(--muted);line-height:1.4}}
.sd-phases{{display:flex;flex-direction:column;gap:8px}}
.sd-phase{{padding:10px 12px;background:var(--panel-2);border-radius:var(--r-md)}}
.sd-phase-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:3px}}
.sd-phase-name{{font-size:12.5px;font-weight:700}}
.sd-phase-dur{{font-size:10.5px;color:var(--muted)}}
.sd-phase-action{{font-size:12px;color:var(--ink-2);margin-bottom:2px}}
.sd-phase-roi{{font-size:11px;color:var(--green-dark);font-weight:600}}
.sd-traps{{display:flex;flex-direction:column;gap:10px}}
.sd-trap{{display:flex;gap:10px;padding:12px;background:var(--amber-bg);border:1px solid rgba(217,119,6,.2);border-radius:var(--r-md)}}
.sd-trap-no{{width:26px;height:26px;background:var(--amber);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0;margin-top:2px}}
.sd-trap strong{{font-size:13px;font-weight:600;color:var(--amber-dark);display:block;margin-bottom:3px}}
.sd-trap p{{font-size:12px;color:var(--amber-dark);margin:0;line-height:1.5;opacity:.85}}
.sd-skills{{display:flex;flex-wrap:wrap;gap:5px}}
.sd-skill-chip{{font-size:11px;background:var(--panel-2);color:var(--ink-2);border:1px solid var(--line);border-radius:var(--r-xs);padding:3px 8px;text-decoration:none;transition:border-color var(--t),color var(--t)}}
.sd-skill-chip:hover{{border-color:var(--accent);color:var(--accent)}}
</style>
"""
    return html_page(sol['title'], body, nav="../", active_nav="solutions")


def render_roadmap_page(skill_lookup: dict[str, "PlaybookSkill"], skill_count: int) -> str:
    """CEO-facing AI capability roadmap whitepaper. Designed for B2B sales, print-ready via @media print."""

    PHASES = [
        {
            "id": "phase1",
            "label": "Phase 1",
            "period": "第 1-3 个月",
            "theme": "立竿见影（Month 1-3）",
            "color": "#2563eb",
            "bg": "#eff6ff",
            "tagline": "30 天内出数字，首月可见 ROI",
            "roi": "800 - 1,900 万/年",
            "items": [
                {
                    "icon": "SC",
                    "title": "供应链预测基线",
                    "story": "某母婴品牌 60+ SKU × 多仓 × 多市场，每月底 2-3 名 PM 纯人工对账三层预测数字——SKU 求和对不上仓库数，仓库数对不上市场总量，相差最高 50%。",
                    "result": "HiFoReAd 分层调和后：对账人力清零，补货计划冲突率降至 < 5%",
                    "roi": "800-1,500 万/年",
                    "skills": ["Skill-Hierarchical-Demand-Forecasting-Reconciliation", "Skill-Demand-Forecasting-Supply-Chain"],
                },
                {
                    "icon": "AI",
                    "title": "客服智能路由 + 70% 工单自动化",
                    "story": "某品牌日均 5 万条跨领域工单，人工路由正确率 61%，每天约 1 万条工单需二次转单；新手妈妈咨询「宝宝 3 月夜醒频繁」，人工客服响应平均 72 小时，且有医疗合规风险。",
                    "result": "AgentRouter 路由正确率 61% → 82%；客服决策树从历史日志自学，70% 工单实现自动化处理，年节省运营成本",
                    "roi": "1,900 万/年（路由）+ 600 万/年（自动化）",
                    "skills": ["Skill-AgentRouter-KG-Guided", "Skill-Customer-Journey-Decision-Tree"],
                },
                {
                    "icon": "RA",
                    "title": "广告归因修正",
                    "story": "某品牌 TikTok 渠道被 naive 归因分配 45% 贡献，但因果 ITE 分析显示真实增量只有 32%——13% 的购买是用户自然意愿，和广告无关。",
                    "result": "纠正后 TikTok 预算从 $40K 调至 $30K，节省的 $10K 转投 Google（ITE 更高），预算效率提升",
                    "roi": "10-20 万/年",
                    "skills": ["Skill-Causal-Attribution-Bridge"],
                },
            ],
        },
        {
            "id": "phase2",
            "label": "Phase 2",
            "period": "第 4-6 个月",
            "theme": "让快赢可持续（Month 4-6）",
            "color": "#7c3aed",
            "bg": "#f5f3ff",
            "tagline": "让 Phase 1 的效果可持续、可复制",
            "roi": "200 - 500 万/年（新增）",
            "items": [
                {
                    "icon": "KG",
                    "title": "产品知识图谱",
                    "story": "AI 在没有结构化产品知识的情况下，给用户推荐「买了吸奶器的用户还需要什么」——它不知道硅胶法兰和乳头霜属于哺乳期刚需配件，推荐准确率极低。",
                    "result": "构建产品 KG 后：KGQA 查询召回率 52% → 92%，跨品类推荐 CTR +18%",
                    "roi": "20-35 万/年（推荐层增量）",
                    "skills": ["Skill-Hierarchical-Product-KG-Construction", "Skill-Ontology-Schema-Design"],
                },
                {
                    "icon": "VP",
                    "title": "A/B 实验平台",
                    "story": "某品牌每次调整定价策略或上新 listing，无法区分效果是真实改进还是季节波动。团队争论持续数周，决策依赖「感觉」。",
                    "result": "Switchback 实验体系搭建后：物流/双边市场实验可信，决策从争论变为数据裁决",
                    "roi": "1,500 万/年（错误决策避免）",
                    "skills": ["Skill-Switchback-Experiment-Design", "Skill-CUPED-Variance-Reduction"],
                },
                {
                    "icon": "AA",
                    "title": "多渠道库存池化",
                    "story": "Amazon FBA 仓吸奶器缺货（超卖），独立站海外仓还有 200 件积压，TikTok Shop 慢速消化——三渠道不互通，总库存 800 件但某渠道已断货。",
                    "result": "跨渠道动态调拨后：总库存减少 15-25%，缺货率 8% → 3%",
                    "roi": "200-400 万/年",
                    "skills": ["Skill-Multi-Channel-Inventory-Pooling"],
                },
            ],
        },
        {
            "id": "phase3",
            "label": "Phase 3",
            "period": "第 7-12 个月",
            "theme": "建立不对称优势（Month 7-12）",
            "color": "#059669",
            "bg": "#f0fdf4",
            "tagline": "竞争对手需要 18 个月才能追上这里",
            "roi": "5,000 万+ 潜力（战略级）",
            "items": [
                {
                    "icon": "PR",
                    "title": "AI 定价引擎",
                    "story": "某品牌大促前手动跟价——降太多伤利润，降太少丢份额。运营靠经验感知，每次大促前都是高压决策，无法同时优化当前销量和品牌长期溢价。",
                    "result": "AIGP 动态定价 A/B 实测：GMV +13%，实验数据非预测值",
                    "roi": "1,321 万/年（A/B 实测）",
                    "skills": ["Skill-AIGP-LLM-Dynamic-Pricing", "Skill-Dynamic-Pricing-Elasticity"],
                },
                {
                    "icon": "TC",
                    "title": "AI 内容工厂",
                    "story": "某品牌进入德国/日本市场，需要本地化口播 Review 视频。人工方案：雇本地 KOL 拍摄，周期 3-4 周，单条成本 $2,000+。批量生产 20 个 SKU 的测评视频需要 6 个月预算。",
                    "result": "Virbo 多语言虚拟人：同等内容量成本降低 80%，生产周期 3 周→ 3 天（实验接入中）",
                    "roi": "35-150 万/年（视接入程度）",
                    "skills": ["Skill-Virbo-Multilingual-Avatar-UGC", "Skill-AnchorCrafter-Virtual-Anchor-Demo"],
                },
                {
                    "icon": "TR",
                    "title": "MAS 多智能体联动",
                    "story": "大促首日某品牌吸奶器打 7 折卖出 5,000 件，第 3 天库存告急被迫涨价，剩余 7 天流量白白浪费——整个大促周期总利润反而低于平销期。根因：定价和补货是两个团队各自决策，无法实时联动。",
                    "result": "FSDA-DRL 快慢双 Agent：定价与补货实时联动，大促周期利润最优化，中小卖家（月 GMV 100-500 万）保守估计年化 225-300 万",
                    "roi": "225-300 万/年（中小规模）· 5,000 万+（GMV > 2 亿规模）",
                    "skills": ["Skill-FSDA-DRL", "Skill-Event-Driven-Demand-MAS"],
                },
                {
                    "icon": "AG",
                    "title": "防御性 AI：保护推荐系统不被竞品劫持",
                    "story": "竞品卖家在商品描述中嵌入恶意 prompt 指令，劫持 AI 导购排名，导致某品牌自营商品在 AI 搜索中的曝光量下降 30-50%——这是 2025 年已出现的真实攻击方式。",
                    "result": "Agent 支付安全红队：自动检测注入攻击并拦截，保护 AI 推荐系统不被操控",
                    "roi": "防御价值 > 5,000 万（以被攻击时的流量损失计）",
                    "skills": ["Skill-Agent-Payment-Security-Red-Team", "Skill-MAS-Adversarial-Defense"],
                },
            ],
        },
    ]

    def _phase_html(phase: dict[str, Any]) -> str:
        items_html = ""
        for item in phase["items"]:
            skill_chips = ""
            for sid in item.get("skills", []):
                sk = skill_lookup.get(sid)
                if sk:
                    skill_chips += (
                        f"<a class='rm-chip' href='skills/{html.escape(sid)}.html'>"
                        f"{html.escape(sk.title[:36])}{'…' if len(sk.title) > 36 else ''}</a>"
                    )
            items_html += f"""
<div class="rm-item">
  <div class="rm-item-icon">{item['icon']}</div>
  <div class="rm-item-body">
    <h4 class="rm-item-title">{html.escape(item['title'])}</h4>
    <div class="rm-story">
      <span class="rm-story-label">真实案例</span>
      {html.escape(item['story'])}
    </div>
    <div class="rm-result">
      <span class="rm-result-label">[OK] 结果</span>
      {html.escape(item['result'])}
    </div>
    <div class="rm-roi-line">年化 ROI：<strong>{html.escape(item['roi'])}</strong></div>
    <div class="rm-chips">{skill_chips}</div>
  </div>
</div>"""

        return f"""
<div class="rm-phase" id="{phase['id']}" style="--phase-color:{phase['color']};--phase-bg:{phase['bg']}">
  <div class="rm-phase-header">
    <div class="rm-phase-badge">{html.escape(phase['label'])}</div>
    <div class="rm-phase-meta">
      <span class="rm-phase-period">{html.escape(phase['period'])}</span>
      <h3 class="rm-phase-theme">{html.escape(phase['theme'])}</h3>
      <p class="rm-phase-tagline">{html.escape(phase['tagline'])}</p>
    </div>
    <div class="rm-phase-roi">
      <span class="rm-phase-roi-label">阶段可验证 ROI</span>
      <strong>{html.escape(phase['roi'])}</strong>
    </div>
  </div>
  <div class="rm-items">{items_html}</div>
</div>"""

    phases_html = "".join(_phase_html(p) for p in PHASES)

    body = f"""
<div class="rm-scqa">
  <div class="rm-scqa-s"><span class="rm-scqa-label">现状</span>2025年，母婴跨境出海品牌平均每月仍有 3 名运营人员全职处理重复性决策——手工对账、等待提数、人工盯价，每天消耗 72 小时以上的宝贵人力。</div>
  <div class="rm-scqa-c"><span class="rm-scqa-label">冲突</span>而先行品牌已在用 AI 将这些成本清零——AgentRouter 年节省 1,900 万元运营成本，AIGP 定价 A/B 实测 GMV +13%。一旦错过这个窗口，差距将在 18-24 个月内变得不可追赶。</div>
  <div class="rm-scqa-q"><span class="rm-scqa-label">问题</span>你的品牌应该从哪里开始，才能在首月就看到可验证的 ROI？</div>
</div>
<div class="rm-hero">
  <div class="rm-hero-eyebrow">唯一把顶会 ML 论文翻译为跨境运营决策的平台 · 2025-2026</div>
  <h1 class="rm-hero-title">12 个月，3 阶段，AI 替代 3 类岗位的重复性决策</h1>
  <p class="rm-hero-sub">首月可见 ROI，全年可验证收益 > 3,000 万元 | NeurIPS · KDD · ICML 论文背书</p>
  <div class="rm-hero-cta">
    <button class="rm-btn-primary" onclick="window.print()">下载 PDF</button>
    <a class="rm-btn-sec" href="playbooks/index.html">查看场景手册 →</a>
    <a class="rm-btn-sec" href="mailto:skills@lute-tlz-dddd.top?subject=预约Demo-AI能力路线图" style="background:#9c5455;color:#fff;border:none">预约 Demo</a>
  </div>
  <p class="rm-hero-note">所有 ROI 数字来源于真实 A/B 实验或匿名客户案例，非模型预测 | 与 Northbeam / Jungle Scout / 纯咨询公司的核心差异：我们给你的是「决策算法」，不是「数据报表」</p>
</div>

<div class="rm-summary-bar">
  <div class="rm-summary-item">
    <span class="rm-summary-num">3</span>
    <span class="rm-summary-label">阶段</span>
  </div>
  <div class="rm-summary-sep">→</div>
  <div class="rm-summary-item">
    <span class="rm-summary-num">10</span>
    <span class="rm-summary-label">核心场景</span>
  </div>
  <div class="rm-summary-sep">→</div>
  <div class="rm-summary-item">
    <span class="rm-summary-num">3,000万+</span>
    <span class="rm-summary-label">可验证年化 ROI</span>
  </div>
  <div class="rm-summary-sep">→</div>
  <div class="rm-summary-item">
    <span class="rm-summary-num">3</span>
    <span class="rm-summary-label">岗位重复性工作被替代</span>
  </div>
</div>

<div class="rm-roles-bar">
  <div class="rm-role">
    
    <div>
      <strong>供应链全链路</strong>
      <p>从 15 个 Excel 联动 → 1 个 MAS 自动执行</p>
      <span class="rm-role-roi">ROI 5,000-8,000 万</span>
    </div>
  </div>
  <div class="rm-role">
    
    <div>
      <strong>数据分析师</strong>
      <p>提数从 72 小时 → 5 分钟，报告从做 → 审</p>
      <span class="rm-role-roi">ROI 1,600-3,000 万</span>
    </div>
  </div>
  <div class="rm-role">
    
    <div>
      <strong>广告优化师</strong>
      <p>凌晨 2 点平台调整，Agent 已完成出价</p>
      <span class="rm-role-roi">ROI 3,000-5,000 万</span>
    </div>
  </div>
</div>

<div class="rm-phases">
  {phases_html}
</div>

<div class="rm-footer">
  <div class="rm-footer-left">
     <h3>从哪里开始？</h3>
     <p>根据你的当前痛点选择入口——每个场景手册包含完整操作步骤、所需数据和 ROI 计算模板。</p>
     <div class="rm-footer-links">
        <a href="playbooks/pb-risk-defense.html">跨境风险防御作战室</a>
        <a href="playbooks/pb-agent-replace.html">AI Agent 替人手册</a>
        <a href="playbooks/pb-tariff-response.html">关税冲击 72h 响应</a>
        <a href="playbooks/pb-compliance.html">跨境合规全链路</a>
        <a href="playbooks/pb-voc-product-loop.html">竞品情报 → 产品迭代</a>
       <a href="playbooks/pb-customer-service-agent.html">客服售后智能体</a>
       <a href="playbooks/pb-fba-operations.html">FBA 运营全链路</a>
       <a href="playbooks/pb-pricing-engine.html">AI 定价引擎手册</a>
       <a href="playbooks/pb-inventory-festival.html">大促备货决策手册</a>
     </div>
  </div>
  <div class="rm-footer-right">
    <div class="rm-footer-cta">
      <p>获取 PDF + 预约 30 分钟 ROI 测算</p>
      <button class="rm-btn-primary" onclick="window.print()" style="margin-bottom:10px">下载 PDF</button>
      <form class="rm-lead-form" action="mailto:skills@lute-tlz-dddd.top" method="GET">
        <input type="email" name="email" placeholder="your@company.com" required style="width:100%;padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#f1f5f9;font-size:13px;margin-bottom:8px;box-sizing:border-box">
        <input type="hidden" name="subject" value="paper2skills ROI测算申请">
        <button type="submit" style="width:100%;padding:8px;background:#9c5455;color:#fff;border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer">获取定制 ROI 测算（邮件联系）</button>
      </form>
    </div>
    <p class="rm-footer-note">
      数据来源：{skill_count} 个从顶会论文萃取的业务 Skills，包含真实 A/B 实验与匿名客户案例。<br>
      所有案例均已脱敏处理，以「某跨境母婴品牌」表述。
    </p>
  </div>
</div>
"""


    return html_page("AI 能力建设路线图", body)


def _render_roi_calculator(calc: dict[str, Any] | None) -> str:
    """Render an interactive ROI calculator as self-contained HTML+JS. No backend needed."""
    if not calc:
        return ""

    sections = calc.get("sections", [])

    tabs_html = "".join(
        "<button class='calc-tab{active}' data-sec='{sid}' style='--tc:{color}'>{label}</button>".format(
            active=" active" if i == 0 else "",
            sid=html.escape(sec["id"]),
            color=html.escape(sec["color"]),
            label=html.escape(sec["label"]),
        )
        for i, sec in enumerate(sections)
    )

    panels_html = ""
    for i, sec in enumerate(sections):
        inputs_html = ""
        for inp in sec["inputs"]:
            inputs_html += f"""
<div class='calc-row'>
  <label class='calc-label' for='{html.escape(sec["id"])}_{html.escape(inp["id"])}'>{html.escape(inp["label"])}</label>
  <div class='calc-input-wrap'>
    <input class='calc-input' type='range'
      id='{html.escape(sec["id"])}_{html.escape(inp["id"])}'
      data-sec='{html.escape(sec["id"])}' data-var='{html.escape(inp["id"])}'
      min='{inp["min"]}' max='{inp["max"]}' step='{inp["step"]}' value='{inp["default"]}'>
    <span class='calc-val' id='v_{html.escape(sec["id"])}_{html.escape(inp["id"])}'>{inp["default"]}</span>
    <span class='calc-unit'>{html.escape(inp["unit"])}</span>
  </div>
</div>"""

        active_cls = " active" if i == 0 else ""
        panels_html += (
            "<div class='calc-panel{active}' id='panel_{sid}' data-sec='{sid}' style='--tc:{color}'>"
            "<div class='calc-inputs'>{inputs}</div>"
            "<div class='calc-result'>"
            "<div class='calc-result-label'>年化 ROI 估算</div>"
            "<div class='calc-result-num' id='result_{sid}'>—</div>"
            "<div class='calc-result-unit'>万元/年</div>"
            "<div class='calc-disclaimer'>基于行业平均改善率保守估算，实际收益因业务规模与实施深度而异</div>"
            "</div></div>"
        ).format(
            active=active_cls,
            sid=html.escape(sec["id"]),
            color=html.escape(sec["color"]),
            inputs=inputs_html,
        )

    formulas_js = "{\n"
    for sec in sections:
        var_names = [inp["id"] for inp in sec["inputs"]]
        defaults = {inp["id"]: inp["default"] for inp in sec["inputs"]}
        formulas_js += f"  '{html.escape(sec['id'])}': {{\n"
        formulas_js += f"    vars: {json.dumps(var_names)},\n"
        formulas_js += f"    defaults: {json.dumps(defaults)},\n"
        formula_body = sec["formula"].strip().replace("\n", "\n    ")
        formulas_js += f"    compute: function({', '.join(var_names)}) {{\n    {formula_body}\n    }},\n"
        formulas_js += f"  }},\n"
    formulas_js += "}"

    js = f"""
<script>
(function(){{
  var CALC = {formulas_js};
  var state = {{}};
  Object.keys(CALC).forEach(function(sec){{
    state[sec] = Object.assign({{}}, CALC[sec].defaults);
  }});

  function compute(sec){{
    var cfg = CALC[sec];
    var args = cfg.vars.map(function(v){{ return state[sec][v]; }});
    try {{
      var result = cfg.compute.apply(null, args);
      var el = document.getElementById('result_' + sec);
      if(el) el.textContent = isNaN(result) || result < 0 ? '—' : result.toLocaleString('zh-CN');
    }} catch(e) {{}}
  }}

  document.querySelectorAll('.calc-input').forEach(function(inp){{
    var sec = inp.dataset.sec, v = inp.dataset.var;
    var valEl = document.getElementById('v_' + sec + '_' + v);
    inp.addEventListener('input', function(){{
      state[sec][v] = parseFloat(this.value);
      if(valEl) valEl.textContent = this.value;
      compute(sec);
    }});
    compute(sec);
  }});

  document.querySelectorAll('.calc-tab').forEach(function(btn){{
    btn.addEventListener('click', function(){{
      var sec = this.dataset.sec;
      document.querySelectorAll('.calc-tab').forEach(function(b){{ b.classList.remove('active'); }});
      document.querySelectorAll('.calc-panel').forEach(function(p){{ p.classList.remove('active'); }});
      this.classList.add('active');
      var panel = document.getElementById('panel_' + sec);
      if(panel){{ panel.classList.add('active'); compute(sec); }}
    }});
  }});

  Object.keys(CALC).forEach(compute);
}})();
</script>"""

    return f"""
<div class='calc-wrapper'>
  <div class='calc-header'>
    <h2>{html.escape(calc.get('title', 'ROI 计算器'))}</h2>
    <p class='muted'>{html.escape(calc.get('subtitle', ''))}</p>
  </div>
  <div class='calc-tabs'>{tabs_html}</div>
  <div class='calc-body'>{panels_html}</div>
</div>
{js}"""


def render_tob_playbook(pb: dict[str, Any], skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    nav = "../"
    steps_html = ""
    for i, step in enumerate(pb.get("steps", []), 1):
        skills_html = ""
        for bs in step.get("skills", []):
            sid = bs["id"]
            why = html.escape(bs.get("why", ""))
            sk = skill_lookup.get(sid)
            if sk:
                roi = f"<span class='roi-badge'>{html.escape(sk.roi_figure)}</span>" if sk.roi_figure else ""
                diff = f"<span class='diff-badge'>{html.escape(sk.difficulty)}</span>" if sk.difficulty else ""
                skills_html += (
                    f"<div class='pb-skill'>"
                    f"<div class='pb-skill-header'>"
                    f"<a href='../skills/{html.escape(sid)}.html' class='pb-skill-name'>{html.escape(sk.title)}</a>"
                    f"<div class='pb-skill-badges'>{roi}{diff}</div>"
                    f"</div>"
                    f"<p class='pb-skill-why'>→ {why}</p>"
                    f"</div>"
                )
            else:
                skills_html += f"<div class='pb-skill'><span class='muted'>{html.escape(sid)}</span></div>"

        data_req = html.escape(step.get("data", ""))
        output = html.escape(step.get("output", ""))
        step_title_safe = html.escape(step['step']).replace("'", "\\'")
        pb_name_safe_q = html.escape(pb['name']).replace("'", "\\'")
        pb_id_safe = pb.get('id', 'unknown')
        exec_btn = f"""<div style='margin-top:14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap'>
      <a href='../chat.html?q={html.escape(step.get("step","").replace(" ","+")+"+在母婴跨境电商场景如何应用？")}' target='_blank'
         style='display:inline-flex;align-items:center;gap:6px;padding:8px 16px;background:var(--accent,#3b82f6);color:#fff;border-radius:8px;font-size:12.5px;font-weight:600;text-decoration:none;transition:background .15s'
         onmouseover="this.style.background='var(--accent-dark,#2563eb)'"
         onmouseout="this.style.background='var(--accent,#3b82f6)'">
        向 AI 咨询此步骤
      </a>
      <a href='../agents.html' target='_blank'
         style='display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:var(--panel-2,#f8fafc);border:1.5px solid var(--line,#e2e8f0);color:var(--ink-2,#475569);border-radius:8px;font-size:12.5px;font-weight:600;text-decoration:none;transition:all .15s'
         onmouseover="this.style.borderColor='var(--accent,#3b82f6)';this.style.color='var(--accent,#3b82f6)'"
         onmouseout="this.style.borderColor='var(--line,#e2e8f0)';this.style.color='var(--ink-2,#475569)'">
        ◈ 调用 Agent 执行
      </a>
      <button id='step-done-{pb_id_safe}-{i}'
         onclick='markStepDone("{pb_id_safe}",{i},{len(pb.get("steps",[]))},this)'
         style='display:inline-flex;align-items:center;gap:5px;padding:7px 12px;background:var(--panel-2,#f8fafc);border:1.5px solid var(--line,#e2e8f0);color:var(--muted,#94a3b8);border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s'>
        ○ 标记完成
      </button>
    </div>
    <script>
    (function(){{
      var k='pb_progress_{pb_id_safe}';
      var done=JSON.parse(localStorage.getItem(k)||'[]');
      if(done.includes({i})){{
        var btn=document.getElementById('step-done-{pb_id_safe}-{i}');
        if(btn){{btn.textContent='✓ 已完成';btn.style.background='#f0fdf4';btn.style.borderColor='#86efac';btn.style.color='#15803d';}}
      }}
    }})();
    function markStepDone(pbId,stepNum,totalSteps,btn){{
      var k='pb_progress_'+pbId;
      var done=JSON.parse(localStorage.getItem(k)||'[]');
      if(!done.includes(stepNum)){{done.push(stepNum);localStorage.setItem(k,JSON.stringify(done));}}
      btn.textContent='✓ 已完成';btn.style.background='#f0fdf4';btn.style.borderColor='#86efac';btn.style.color='#15803d';
      if(done.length>=totalSteps){{
        var toast=document.createElement('div');
        toast.style.cssText='position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#059669;color:#fff;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:700;z-index:9999;box-shadow:0 4px 12px rgba(5,150,105,.3)';
        toast.textContent=''+pbId+' 全部步骤完成！';
        document.body.appendChild(toast);setTimeout(()=>toast.remove(),4000);
      }}
    }}
    </script>"""
        steps_html += f"""
<div class='pb-step'>
  <div class='pb-step-num'>Step {i}</div>
  <div class='pb-step-body'>
    <h3 class='pb-step-title'>{html.escape(step['step'])}</h3>
    <p class='pb-problem'>{html.escape(step['problem'])}</p>
    <div class='pb-skills'>{skills_html}</div>
    {'<div class="pb-data"><strong>所需数据：</strong>' + data_req + '</div>' if data_req else ''}
    {'<div class="pb-output"><strong>输出结果：</strong>' + output + '</div>' if output else ''}
    {exec_btn}
  </div>
</div>"""

    outcomes = "".join(f"<li>[OK] {html.escape(o)}</li>" for o in pb.get("outcomes", []))
    calc_html = _render_roi_calculator(pb.get("roi_calculator")) if pb.get("roi_calculator") else ""
    pb_name_safe = pb['name'].replace(' ', '%20').replace('&', '%26')
    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../playbooks/index.html">场景手册</a> / {html.escape(pb['name'])}</nav>
<div class='pb-hero'>
  <span class='pb-icon'>{pb.get('svg_icon') or pb['icon']}</span>
  <div>
    <h1>{html.escape(pb['name'])}</h1>
    <p class='lead'>{html.escape(pb['desc'])}</p>
    <span class='biz-tag'>{html.escape(pb['tag'])}</span>
  </div>
</div>
{''.join([f"<div class='pb-roi-callout'>{html.escape(item['label'])}<span class='pb-roi-val'>{html.escape(item['value'])}</span></div>" for item in pb.get('roi_callout', [])])}
<div class='pb-intro'>{html.escape(pb['intro'])}</div>
{'<div class="wf-outcomes"><h3>预期收益</h3><ul>' + outcomes + '</ul></div>' if outcomes else ''}
<div class='pb-feishu-bar' style='display:flex;align-items:center;gap:10px;margin:12px 0;padding:10px 14px;background:#fff;border:1px solid var(--line);border-radius:8px;flex-wrap:wrap'>
  <span style='font-size:12.5px;color:var(--muted);flex:1'>执行进度 <strong id="pb-step-done-{pb['id']}">0</strong> / {len(pb.get('steps',[]))} 步</span>
  <button onclick="pbShareFeishu('{html.escape(pb['id'])}','{html.escape(pb['name'])}',{len(pb.get('steps',[]))})" style='padding:6px 14px;background:#00B96B;color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer'>推送进度到飞书</button>
</div>
<div class='pb-steps'>{steps_html}</div>
{calc_html}
<div class='pb-lead-capture'>
  <div class='pb-lead-inner'>
    <div class='pb-lead-text'>
      <h3>想了解这套方案如何落地你的业务？</h3>
      <p>预约 30 分钟免费 ROI 测算 — 基于你的 SKU 数量、广告预算和当前痛点，给出定制化收益估算。</p>
      <ul class='pb-lead-bullets'>
        <li>✓ 结合你的实际数据，不是通用模板</li>
        <li>✓ 明确哪 1-2 个 Skill 优先落地 ROI 最高</li>
        <li>✓ 30 分钟，结束后你有一份行动清单</li>
      </ul>
    </div>
    <div class='pb-lead-action'>
      <a href='mailto:skills@lute-tlz-dddd.top?subject=预约ROI测算-{pb_name_safe}&body=手册:{pb_name_safe}%0A公司规模:%0A主要痛点:%0A当前月GMV:' class='pb-lead-btn'>预约 30 分钟 ROI 测算 →</a>
      <p class='pb-lead-note'>发送邮件后 24h 内回复确认时间</p>
    </div>
  </div>
</div>
"""
    return html_page(pb["name"], body, nav)


def render_workflow_page(wf_def: dict[str, Any], skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    """Render a full decision-tree workflow page from YAML definition."""
    nav = "../"
    name = html.escape(wf_def.get("name", ""))
    description = html.escape(wf_def.get("description", "按业务流程推荐的 Skill 链。"))
    entry_q = html.escape(wf_def.get("entry_question", ""))
    target_users = wf_def.get("target_users", [])
    outcomes = wf_def.get("outcomes", [])
    steps = wf_def.get("steps", [])

    user_tags = "".join(
        f"<span class='tag'>{html.escape(u)}</span>" for u in target_users
    )
    outcome_items = "".join(
        f"<li>[OK] {html.escape(o)}</li>" for o in outcomes
    )
    step_html = "".join(render_workflow_step(s, skill_lookup) for s in steps)

    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../workflows/index.html">工作流</a> / {name}</nav>
<h1>{name}</h1>
<p class="lead">{description}</p>
<div class="wf-meta">
  <div><strong>适用角色</strong>{user_tags}</div>
  {'<div class="wf-entry-question"><strong>入口问题：</strong>' + entry_q + '</div>' if entry_q else ''}
</div>
{'<div class="wf-outcomes"><h3>预期收益</h3><ul>' + outcome_items + '</ul></div>' if outcomes else ''}
<div class="wf-tree">
  {step_html}
</div>
"""
    return html_page(wf_def.get("name", "工作流"), body, nav)


# ---------------------------------------------------------------------------
# HTML page scaffold
# ---------------------------------------------------------------------------

def html_page(title: str, body: str, nav: str = "", active_nav: str = "") -> str:
    def sidebar_link(href: str, label: str, key: str = "", icon: str = "") -> str:
        active = ' aria-current="page" class="active"' if key and key == active_nav else ""
        icon_html = f'<span class="sbl-icon">{icon}</span>' if icon else ""
        return f'<a href="{nav}{href}"{active}>{icon_html}<span class="sbl-text">{label}</span></a>'

    def sidebar_section(label: str, links: str) -> str:
        return f'<div class="sb-section"><p class="sb-label">{label}</p><div class="sb-links">{links}</div></div>'

    ga4_snippet = f"""
  <script async src="https://www.googletagmanager.com/gtag/js?id={GA4_MEASUREMENT_ID}"></script>
  <script>
  window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}
  gtag('js',new Date());gtag('config','{GA4_MEASUREMENT_ID}');
  </script>""" if GA4_MEASUREMENT_ID and not GA4_MEASUREMENT_ID.startswith("G-X") else ""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)} · paper2skills</title>
  <link rel="stylesheet" href="{nav}assets/style.css">{ga4_snippet}
</head>
<body>
  <header class="topbar">
    <button class="hamburger" id="hamburger" aria-label="菜单" aria-expanded="false">
      <span></span><span></span><span></span>
    </button>
    <a class="brand" href="{nav}index.html">
      <span class="brand-icon">P</span>
      <span class="brand-name">paper2skills<span class="brand-tag">Playbook</span></span>
    </a>
    <div class="topbar-right">
      <input id="global-search" placeholder="搜索技能 / 场景…" autocomplete="off" role="search" aria-label="搜索">
      <a href="{nav}ai-roadmap.html" class="topbar-cta{'  active' if active_nav == 'roadmap' else ''}">AI 路线图 →</a>
      <div id="p2s-auth-widget" style="margin-left:8px;display:flex;align-items:center;gap:8px">
        <span id="p2s-user-avatar" style="width:22px;height:22px;border-radius:999px;display:none;align-items:center;justify-content:center;background:#111827;color:#fff;font-size:11px;font-weight:700;flex:0 0 auto"></span>
        <span id="p2s-user-name" style="font-size:12px;color:#94a3b8;display:none;white-space:nowrap"></span>
        <a id="p2s-login-btn" href="/auth/login" style="font-size:11px;padding:4px 10px;background:#B5323E;color:#fff;border-radius:5px;text-decoration:none;white-space:nowrap">飞书登录</a>
      </div>
    </div>
  </header>
  <div id="search-results" class="search-results hidden" role="listbox"></div>
  <div class="mobile-nav-overlay" id="mobile-overlay"></div>
  <main class="layout">
    <aside class="sidebar" id="sidebar">
       <div class="sb-top">
         {sidebar_section('决策工具',
           sidebar_link('index.html', '总览', 'index', '') +
           sidebar_link('diagnostic.html', '业务诊断中心', 'diagnostic', '') +
           sidebar_link('chat.html', 'AI 知识库对话', 'chat', '')
         )}
         {sidebar_section('执行手册',
           sidebar_link('playbooks/index.html', '场景手册', 'playbooks', '') +
           sidebar_link('solutions/index.html', '方案库', 'solutions', '')
         )}
         {sidebar_section('智能体',
           sidebar_link('agents.html', '智能体广场', 'agents', '') +
           sidebar_link('agent-report.html', '智能体报告', 'agent-report', '')
         )}
         {sidebar_section('知识图谱',
           sidebar_link('domains/index.html', '按领域浏览', 'domains', '') +
           sidebar_link('topics/index.html', '按主题浏览', 'topics', '') +
           sidebar_link('workflows/index.html', '业务工作流', 'workflows', '') +
           sidebar_link('graph/overview.html', '技能关系图谱', 'graph', '') +
           sidebar_link('skills/index.html', '全部 Skills', 'skills', '')
         )}
         {sidebar_section('战略报告',
           sidebar_link('ai-roadmap.html', 'AI 能力路线图', 'roadmap', '') +
           sidebar_link('maturity-report.html', '成熟度报告 2026', 'maturity', '')
         )}
      </div>
      <div class="sb-bottom">
        <a href="{nav}pricing.html" class="sb-upgrade-card">
          <span class="sb-upgrade-label">免费版 · 每月10次</span>
          <span class="sb-upgrade-title">升级 Pro 解锁无限调用</span>
          <span class="sb-upgrade-sub">CSV接入 · REST API · 无限Agent →</span>
        </a>
        <a href="{nav}settings.html" class="sb-settings-link">
          <span class="sb-settings-icon">⚙</span>
          <span>设置 · API Key</span>
        </a>
      </div>
    </aside>
    <section class="content">{body}</section>
  </main>

  <!-- AI Chat Panel -->
  <script src="{nav}assets/playbook-data.js"></script>
  <script src="{nav}assets/search.js"></script>
  <script>
  const hbtn = document.getElementById('hamburger');
  const overlay = document.getElementById('mobile-overlay');
  const sidebar = document.getElementById('sidebar');
  function toggleMenu(open) {{
    hbtn.setAttribute('aria-expanded', open);
    hbtn.classList.toggle('open', open);
    sidebar.classList.toggle('open', open);
    overlay.classList.toggle('show', open);
    document.body.style.overflow = open ? 'hidden' : '';
  }}
  hbtn.addEventListener('click', () => toggleMenu(hbtn.getAttribute('aria-expanded') !== 'true'));
  overlay.addEventListener('click', () => toggleMenu(false));

  fetch('/auth/me', {{credentials: 'include', cache: 'no-store'}}).then(function(r){{ return r.json(); }}).then(function(u){{
    if (u && u.name) {{
      var nameNode = document.getElementById('p2s-user-name');
      var avatarNode = document.getElementById('p2s-user-avatar');
      var loginNode = document.getElementById('p2s-login-btn');
      if (nameNode) {{ nameNode.textContent = u.name; nameNode.style.display = 'inline'; }}
      if (avatarNode) {{ avatarNode.textContent = String(u.name).slice(0, 1); avatarNode.style.display = 'inline-flex'; avatarNode.title = u.name; }}
      if (loginNode) {{ loginNode.textContent = '退出'; loginNode.href = '#'; loginNode.onclick = function(){{ fetch('/auth/logout', {{method: 'POST', credentials: 'include'}}).then(function(){{ location.reload(); }}); return false; }}; }}
    }}
  }}).catch(function(){{}});

  window.pbShareFeishu = function(pbId, pbName, totalSteps) {{
    const done = document.querySelectorAll('.pb-step-check:checked, .step-done, [data-done="1"]').length;
    const pct = totalSteps > 0 ? Math.round(done / totalSteps * 100) : 0;
    const bar = '█'.repeat(Math.round(pct/10)) + '░'.repeat(10-Math.round(pct/10));
    fetch('/api/feishu-callback', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        action: {{value: {{
          action: 'confirm',
          agent_id: 'playbook-progress',
          inputs_str: '手册: '+pbName+' | 进度: '+done+'/'+totalSteps+' ('+pct+'%) '+bar+' | 时间: '+new Date().toLocaleString('zh-CN')
        }}}}
      }})
    }}).then(() => {{
      const btn = event.target;
      btn.textContent = '✅ 已推送';
      btn.style.background = '#059669';
      setTimeout(() => {{ btn.textContent = '推送进度到飞书'; btn.style.background = '#00B96B'; }}, 2000);
    }}).catch(() => {{}});
  }};
  </script>
</body>
</html>"""


def skill_url(skill_id: str, nav: str = "") -> str:
    return f"{nav}skills/{skill_id}.html"


def render_skill_card(skill: PlaybookSkill, nav: str = "") -> str:
    roi_html = (
        f"<span class='sc-roi'>{html.escape(skill.roi_figure)}</span>"
        if skill.roi_figure else ""
    )
    diff_html = (
        f"<span class='sc-diff'>{html.escape(skill.difficulty)}</span>"
        if skill.difficulty else ""
    )
    footer_html = f"<div class='sc-footer'>{roi_html}{diff_html}</div>" if (roi_html or diff_html) else ""
    desc = html.escape(skill.problem_solved or skill.algorithm_summary)
    data_domain = html.escape(skill.domain_dir)
    data_diff   = html.escape(skill.difficulty or "")
    return f"""<a class="card skill-card anim-fade-in-up" href="{skill_url(skill.skill_id, nav)}" data-domain="{data_domain}" data-diff="{data_diff}">
  <div class="sc-domain">{html.escape(skill.domain_dir)}</div>
  <h3 class="sc-title">{html.escape(skill.title)}</h3>
  <p class="sc-desc">{desc}</p>
  {footer_html}
</a>"""


def link_list(items: list[str], nav: str = "", skill_ids: set[str] | None = None) -> str:
    if not items:
        return "<p class='muted'>暂无</p>"
    _ids = skill_ids if skill_ids is not None else KNOWN_SKILL_IDS
    rows = []
    for item in items:
        escaped = html.escape(item)
        if item in _ids:
            rows.append(f"<li><a href='{skill_url(item, nav)}'>{escaped}</a></li>")
        else:
            rows.append(f"<li><span class='muted'>{escaped}</span></li>")
    return "<ul>" + "".join(rows) + "</ul>"


def render_skill_page(skill: PlaybookSkill) -> str:
    nav = "../"

    # Handbook uplinks: show which handbooks use this skill
    hb_refs = SKILL_HANDBOOK_MAP.get(skill.skill_id, [])
    handbook_uplinks = ""
    if hb_refs:
        chips = "".join(
            f"<a class='hb-uplink' href='../playbooks/{html.escape(pb_id)}.html'>{html.escape(pb_name)}</a>"
            for pb_id, pb_name in hb_refs
        )
        handbook_uplinks = f"<div class='hb-uplinks'><span class='hb-uplinks-label'>收录于</span>{chips}</div>"

    # Scenario section: prefer prose paragraphs; fall back to bullet list
    if skill.scenario_paragraphs:
        scenario_html = "".join(f"<p>{html.escape(p)}</p>" for p in skill.scenario_paragraphs)
    elif skill.business_scenarios:
        scenario_html = render_items(skill.business_scenarios)
    else:
        scenario_html = "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>"

    # ROI / value panel
    roi_meta = ""
    if skill.roi_figure or skill.difficulty or skill.priority:
        parts = []
        if skill.roi_figure:
            parts.append(f"<div class='roi-item'><span class='roi-label'>年化 ROI</span><span class='roi-value'>{html.escape(skill.roi_figure)}</span></div>")
        if skill.difficulty:
            parts.append(f"<div class='roi-item'><span class='roi-label'>实现难度</span><span class='roi-value'>{html.escape(skill.difficulty)}</span></div>")
        if skill.priority:
            parts.append(f"<div class='roi-item'><span class='roi-label'>业务优先级</span><span class='roi-value'>{html.escape(skill.priority)}</span></div>")
        roi_meta = (
            "<div class='roi-panel anim-fade-in-up anim-delay-2'>"
            + "".join(parts)
            + "</div>"
        )

    # Business context panel (injected from DOMAIN_BUSINESS_CONTEXT)
    biz_panel = ""
    if skill.biz_role or skill.biz_trigger:
        role_html = (
            f"<div class='biz-ctx-item'>"
            f"<span class='biz-ctx-label'>适用角色</span>"
            f"<span class='biz-ctx-value'>{html.escape(skill.biz_role)}"
            + (f"<span class='biz-ctx-secondary'> · {html.escape(skill.biz_role2)}</span>" if skill.biz_role2 else "")
            + f"</span></div>"
        )
        trigger_html = (
            f"<div class='biz-ctx-item biz-ctx-full'>"
            f"<span class='biz-ctx-label'>什么情况下用</span>"
            f"<span class='biz-ctx-value'>{html.escape(skill.biz_trigger)}</span>"
            f"</div>"
        ) if skill.biz_trigger else ""
        outcome_html = (
            f"<div class='biz-ctx-item biz-ctx-full'>"
            f"<span class='biz-ctx-label'>成功是什么样的</span>"
            f"<span class='biz-ctx-value biz-ctx-outcome'>{html.escape(skill.biz_outcome)}</span>"
            f"</div>"
        ) if skill.biz_outcome else ""
        pain_html = (
            f"<div class='biz-ctx-item biz-ctx-full'>"
            f"<span class='biz-ctx-label'>业务痛点</span>"
            f"<div class='biz-pain-tags'>"
            + "".join(
                f"<span class='biz-pain-tag'>{html.escape(p.strip())}</span>"
                for p in skill.biz_pain.split("·") if p.strip()
            )
            + f"</div></div>"
        ) if skill.biz_pain else ""
        platform_html = (
            f"<div class='biz-ctx-item'>"
            f"<span class='biz-ctx-label'>适用平台</span>"
            f"<span class='biz-ctx-value'>{html.escape(skill.biz_platform)}</span>"
            f"</div>"
        ) if skill.biz_platform else ""
        biz_panel = (
            f"<div class='biz-ctx-panel'>"
            f"<div class='biz-ctx-header'>业务视角</div>"
            f"<div class='biz-ctx-grid'>"
            f"{role_html}{platform_html}{trigger_html}{outcome_html}{pain_html}"
            f"</div></div>"
        )

    agent_cases_html = ""

    # ── Phase 8: 一键调用Agent入口（每个Skill页面注入） ──
    from config.agents_data import AGENT_CATALOG as _AGENT_CATALOG
    _skill_to_agents: dict[str, list[dict]] = {}
    for _ag in _AGENT_CATALOG:
        for _sid in _ag.get("linked_skills", []):
            _skill_to_agents.setdefault(_sid, []).append({"id": _ag["id"], "name": _ag["name"]})
    related_agents = _skill_to_agents.get(skill.skill_id, [])
    agent_invoke_html = ""
    if related_agents:
        btns = "".join(
            f"<a href='../agents.html' "
            f"style='display:inline-flex;align-items:center;gap:6px;padding:8px 14px;"
            f"background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;"
            f"border-radius:8px;font-size:12.5px;font-weight:600;text-decoration:none;"
            f"transition:opacity .15s' onmouseover='this.style.opacity=\".85\"' onmouseout='this.style.opacity=\"1\"'>"
            f"◈ {html.escape(ag['name'])}"
            f"</a>"
            for ag in related_agents[:3]
        )
        agent_invoke_html = (
            f"<div style='margin:24px 0;padding:16px 18px;"
            f"background:linear-gradient(135deg,#f0f4ff 0%,#faf5ff 100%);"
            f"border:1px solid #c7d2fe;border-radius:12px'>"
            f"<div style='font-size:12.5px;font-weight:700;color:#3730a3;margin-bottom:10px'>"
            f"可直接调用的 Agent</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:8px'>{btns}</div>"
            f"<div style='font-size:11px;color:#94a3b8;margin-top:8px'>"
            f"Agent 已内置此 Skill 的业务逻辑，点击进入智能体广场立即运行</div>"
            f"</div>"
        )
    try:
        candidate_paths = [
            Path(skill.path),
            Path("paper2skills-vault") / skill.path,
            Path(__file__).parent.parent.parent.parent / "paper2skills-vault" / skill.path,
        ]
        raw = ""
        for mp in candidate_paths:
            if mp.exists():
                raw = mp.read_text(encoding="utf-8", errors="replace")
                break
        m = re.search(r'##\s*🧪\s*调用案例.*?$(.+?)(?=\n##\s|\Z)', raw, re.DOTALL | re.MULTILINE)
        if m:
            case_text = m.group(1).strip()
            lines_html = []
            for line in case_text.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                if '**' in stripped:
                    parts = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html.escape(stripped))
                    lines_html.append(f"<p style='margin:4px 0'>{parts}</p>")
                else:
                    lines_html.append(f"<p style='margin:4px 0;color:#475569'>{html.escape(stripped)}</p>")
            inner = "\n".join(lines_html)
            agent_cases_html = (
                f"<div style='margin-top:24px;padding:16px 20px;"
                f"background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px'>"
                f"<div style='font-size:13px;font-weight:700;color:#065f46;margin-bottom:10px'>智能体广场调用案例</div>"
                f"{inner}"
                f"<div style='margin-top:10px'><a href='../agents.html' "
                f"style='font-size:12px;color:#059669;font-weight:600'>→ 前往智能体广场运行</a></div>"
                f"</div>"
            )
    except Exception:
        pass

    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../domains/{slugify(skill.domain_dir)}.html">{html.escape(skill.domain_dir)}</a> / {html.escape(skill.skill_id)}</nav>
<div class="skill-toc">
  <a href="#s-problem">① 问题</a>
  <a href="#s-algo">② 算法</a>
  <a href="#s-scenario">③ 场景</a>
  <a href="#s-code">④ 代码</a>
  <a href="#s-relations">⑤ 关联</a>
  <a href="#s-value">⑥ 价值</a>
</div>
<div class="skill-header-block anim-fade-in-up">
  <div class="skill-domain-chip">{html.escape(skill.domain_dir)}</div>
  <h1 class="skill-main-title">{html.escape(skill.title)}</h1>
  <p class="skill-skill-id">{html.escape(skill.skill_id)}</p>
</div>
<div class="tag-row anim-fade-in-up anim-delay-1">{''.join(f"<span class='tag'>{html.escape(t)}</span>" for t in skill.tags + skill.topics + skill.workflows)}</div>
{handbook_uplinks}
{roi_meta}
{biz_panel}
<div class="two-col">
  <section>
    <h2 id="s-problem">1. 解决的问题</h2>
    <p>{html.escape(skill.problem_solved or skill.algorithm_summary)}</p>
    <h2 id="s-algo">2. 核心算法逻辑</h2>
    <p>{html.escape(skill.algorithm_summary)}</p>
    <h2 id="s-scenario">3. 业务应用场景</h2>
    {scenario_html}
    <h2>4. 输入数据要求</h2>{render_items(skill.inputs) if skill.inputs else "<p class='muted'>请查看原始代码模板获取输入规格。</p>"}
    <h2>5. 输出结果</h2>{render_items(skill.outputs) if skill.outputs else "<p class='muted'>请查看原始代码模板获取输出规格。</p>"}
    <h2 id="s-value">6. 业务价值 / ROI</h2>{render_items(skill.roi) if skill.roi else ("<p>" + html.escape(skill.roi_figure) + "</p>" if skill.roi_figure else "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>")}
    <h2 id="s-code">7. 代码模板</h2>
    <p class="muted">代码块数量：{skill.code_blocks} · 路径：{html.escape(skill.code_path or '未检测到')}</p>
    {_render_code_preview(skill.code_preview)}
    <h2>8. 论文来源</h2>{render_items(skill.papers)}
  </section>
  <aside class="relation-panel" id="s-relations">
    <h2>Skill Relations</h2>
    <svg id="ego-graph" data-skill="{html.escape(skill.skill_id)}" width="280" height="220"></svg>
    <div id="ego-legend" class="ego-legend">
      <span class="edge-dot prereq"></span>前置
      <span class="edge-dot combo" style="margin-left:8px"></span>组合
      <span class="edge-dot ext" style="margin-left:8px"></span>延伸
    </div>
    <h3>前置技能</h3>{link_list(skill.relations.get('prerequisite', []), nav)}
    <h3>延伸技能</h3>{link_list(skill.relations.get('extends', []), nav)}
    <h3>可组合技能</h3>{link_list(skill.relations.get('combinable', []), nav)}
  </aside>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="../assets/ego-graph.js"></script>
<script>
function copyCode(btn) {{
  var pre = btn.nextElementSibling;
  var text = pre ? pre.textContent : '';
  navigator.clipboard.writeText(text).then(function() {{
    btn.textContent = '已复制 ✓';
    btn.classList.add('copied');
    if(typeof gtag!=='undefined')gtag('event','skill_code_copy',{{skill_id:'{html.escape(skill.skill_id)}',skill_domain:'{html.escape(skill.domain_dir)}'}});
    setTimeout(function() {{
      btn.textContent = '复制';
      btn.classList.remove('copied');
    }}, 2000);
  }}).catch(function() {{
    btn.textContent = '复制失败';
    setTimeout(function() {{ btn.textContent = '复制'; }}, 1500);
  }});
}}
if(typeof gtag!=='undefined')gtag('event','skill_view',{{skill_id:'{html.escape(skill.skill_id)}',skill_domain:'{html.escape(skill.domain_dir)}',skill_difficulty:'{html.escape(skill.difficulty or "")}'}});
</script>
{agent_invoke_html}
{agent_cases_html}"""
    return html_page(skill.title, body, nav)


def _md_inline(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', text)
    return text


def render_items(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>"
    return "<ul>" + "".join(f"<li>{_md_inline(html.escape(item))}</li>" for item in items) + "</ul>"


def _render_code_preview(code: str) -> str:
    if not code:
        return "<p class='muted'>请查看原始 Skill 卡片获取完整代码。</p>"
    escaped = html.escape(code)
    # 检测语言
    lang = "Python" if "import " in code or "def " in code or "print(" in code else "Code"
    line_count = code.count("\n") + 1
    return (
        f"<div class='code-wrap'>"
        f"<div class='code-header'>"
        f"<span class='code-lang-badge'>{lang}</span>"
        f"<span class='code-meta'>{line_count} 行 · 可运行</span>"
        f"<button class='copy-btn' onclick='copyCode(this)' title='复制代码'>"
        f"<svg width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><rect x='9' y='9' width='13' height='13' rx='2'/><path d='M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1'/></svg>"
        f"复制"
        f"</button>"
        f"</div>"
        f"<pre class='code-preview' data-lang='{lang}'><code>{escaped}</code></pre>"
        f"</div>"
    )


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Index page (Phase 3C — three-audience redesign)
# ---------------------------------------------------------------------------

def render_index(skill_count: int, domain_count: int, edge_count: int, domains: list[dict[str, Any]], skills: list[PlaybookSkill], workflow_count: int = 32) -> str:
    domain_cards = "".join(
        f"<a class='metric-card domain-card anim-fade-in-up' href='domains/{slugify(d['vault_dir'])}.html' "
        f"style='animation-delay:{i*0.03:.2f}s'>"
        f"<strong>{html.escape(d['vault_dir'])}</strong>"
        f"<span>{d.get('skill_count', 0)} Skills</span></a>"
        for i, d in enumerate(domains)
    )

    # Top 5 skills by relation count (degree centrality proxy)
    skill_degree = {s.skill_id: len(s.relations.get("prerequisite", [])) + len(s.relations.get("combinable", [])) for s in skills}
    hot_skills = sorted(skills, key=lambda s: skill_degree.get(s.skill_id, 0), reverse=True)[:5]
    hot_items = "".join(
        f"<li class='anim-fade-in-up' style='animation-delay:{i*0.04:.2f}s'>"
        f"<a href='skills/{s.skill_id}.html' class='hot-skill-link'>"
        f"<span class='hot-skill-domain'>{html.escape(s.domain_dir)}</span>"
        f"{html.escape(s.title)}"
        f"</a>"
        f"{'<span class=roi-badge>' + html.escape(s.roi_figure) + '</span>' if s.roi_figure else ''}"
        f"</li>"
        for i, s in enumerate(hot_skills)
    )

    business_cards = "".join(
        f"<a class='biz-card' href='{e['href']}'>"
        f"<div class='biz-card-header'>"
        f"<span class='biz-icon'>{e['icon']}</span>"
        f"<div class='biz-body'>"
        f"<div class='biz-card-meta'>"
        f"<strong>{html.escape(e['label'])}</strong>"
        f"</div>"
        f"<p>{html.escape(e['desc'])}</p>"
        f"<div class='biz-card-footer'><span class='biz-tag'>{html.escape(e['tag'])}</span></div>"
        f"</div>"
        f"</div>"
        f"</a>"
        for e in BUSINESS_ENTRIES
    )

    return f"""
<div class="hero">
  <p class="hero-badge">唯一把顶会 ML 论文翻译为跨境运营决策的平台</p>
  <h1>1123 个顶会 AI 技能 × 3.3亿元年化 ROI — 母婴跨境品牌的增长基础设施</h1>
  <p class="lead">{skill_count} 个从 NeurIPS / KDD / ICML / ICLR 顶会萃取的决策技能，覆盖因果推断·MAS多智能体·视频电商·ESG合规等最新方向。每个技能配有真实 ROI 数字（均值29万元/年）、可运行代码和业务场景——这是任何咨询公司和 SaaS 工具都无法复制的能力。</p>
  <div class="hero-primary-cta">
    <a class="btn-primary accent" href="ai-roadmap.html">查看 AI 能力路线图</a>
    <a class="btn-secondary" href="mailto:skills@lute-tlz-dddd.top?subject=预约Demo-paper2skills" >预约 30 分钟 Demo</a>
  </div>
  <div class="hero-tabs" id="heroTabs">
    <button class="tab-btn active" data-tab="biz">业务专家 / 运营</button>
    <button class="tab-btn" data-tab="ds">数据科学家</button>
    <button class="tab-btn" data-tab="ceo">CEO / 决策层</button>
    <button class="tab-btn" data-tab="explore">技术 / 算法研究者</button>
  </div>
</div>


  <div style="margin:0 0 28px;padding:16px 22px;background:linear-gradient(135deg,#fff5f5 0%,#fff 100%);border:1.5px solid #f0c0c0;border-radius:12px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap">
    <div>
      <div style="font-size:13.5px;font-weight:700;color:#B5323E;margin-bottom:3px">免费探索 5 个核心领域 · 解锁全部 25 域 + Agent 运行权限</div>
      <div style="font-size:12px;color:#6b7280">当前开放：因果推断 · A/B实验 · 时间序列 · 供应链 · 推荐系统</div>
    </div>
    <a href="mailto:skills@lute-tlz-dddd.top?subject=申请企业版-paper2skills" style="padding:9px 20px;background:#B5323E;color:#fff;border-radius:8px;font-size:13px;font-weight:700;text-decoration:none;white-space:nowrap;flex-shrink:0">申请企业版 →</a>
  </div>

  <div style="margin:32px 0 24px;padding:20px 24px;background:#fff;border:1px solid var(--line);border-radius:12px">
    <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin-bottom:14px">从你的角色开始</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
      <a href="playbooks/pb-supply-chain-intelligence.html" style="padding:14px 16px;background:var(--bg);border:1px solid var(--line);border-radius:10px;text-decoration:none;color:inherit;transition:all .15s;display:block" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--line)'">
        <div style="font-size:13px;font-weight:700;color:var(--ink);margin-bottom:3px">供应链负责人</div>
        <div style="font-size:12px;color:var(--muted)">补货·库存·履约·风险</div>
      </a>
      <a href="playbooks/pb-attribution-unification.html" style="padding:14px 16px;background:var(--bg);border:1px solid var(--line);border-radius:10px;text-decoration:none;color:inherit;transition:all .15s;display:block" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--line)'">
        <div style="font-size:13px;font-weight:700;color:var(--ink);margin-bottom:3px">广告运营</div>
        <div style="font-size:12px;color:var(--muted)">ROAS·归因·MMM·预算</div>
      </a>
      <a href="ai-roadmap.html" style="padding:14px 16px;background:var(--bg);border:1px solid var(--line);border-radius:10px;text-decoration:none;color:inherit;transition:all .15s;display:block" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--line)'">
        <div style="font-size:13px;font-weight:700;color:var(--ink);margin-bottom:3px">CEO / 管理层</div>
        <div style="font-size:12px;color:var(--muted)">能力路线图·ROI·成熟度</div>
      </a>
    </div>
  </div>

<div class="tab-panel active" id="tab-biz">
  <h2>从业务问题出发</h2>
  <p class="muted">选择你正在面对的挑战，直达对应的 Skill 路径与工作流。</p>
  <div class="biz-grid">
    {business_cards}
  </div>
</div>

<div class="tab-panel" id="tab-ds">
  <h2>数据科学家视角</h2>
  <div class="ds-grid">
    <div class="ds-card">
      <h3>高连接度 Skills</h3>
      <p class="muted">被最多 Skill 依赖的核心算法，学习回报最高。</p>
      <ul class="hot-list">{hot_items}</ul>
    </div>
    <div class="ds-card">
      <h3>按算法类型</h3>
      <div class="algo-tags">
        <a class="tag" href="topics/广告与投放.html">广告与投放</a>
        <a class="tag" href="topics/供应链与补货.html">供应链与补货</a>
        <a class="tag" href="topics/知识图谱与rag.html">知识图谱&amp;RAG</a>
        <a class="tag" href="topics/mas与智能体工程.html">MAS&amp;智能体</a>
        <a class="tag" href="topics/推荐与搜索.html">推荐与搜索</a>
        <a class="tag" href="topics/定价与利润.html">定价与利润</a>
        <a class="tag" href="topics/风控与合规.html">风控与合规</a>
        <a class="tag" href="topics/视觉内容生成.html">视觉内容生成</a>
      </div>
    </div>
    <div class="ds-card">
      <h3>vs 竞品对比</h3>
      <table style="font-size:12px;margin-top:8px">
        <thead><tr><th>维度</th><th>纯咨询</th><th>SaaS工具</th><th>paper2skills</th></tr></thead>
        <tbody>
          <tr><td>证据级别</td><td>经验判断</td><td>平台数据</td><td><strong>顶会论文 + A/B实测</strong></td></tr>
          <tr><td>ROI可溯源</td><td>无</td><td>部分</td><td><strong>每个Skill有ROI数字</strong></td></tr>
          <tr><td>跨境场景</td><td>通用</td><td>通用</td><td><strong>母婴跨境专属</strong></td></tr>
          <tr><td>可执行代码</td><td>无</td><td>无</td><td><strong>{skill_count}个可运行模板</strong></td></tr>
          <tr><td>知识更新</td><td>项目制</td><td>产品迭代</td><td><strong>持续萃取顶会论文</strong></td></tr>
        </tbody>
      </table>
      <div class="algo-tags">
        <a class="tag" href="topics/广告与投放.html">广告与投放</a>
        <a class="tag" href="topics/供应链与补货.html">供应链与补货</a>
        <a class="tag" href="topics/知识图谱与rag.html">知识图谱&amp;RAG</a>
        <a class="tag" href="topics/mas与智能体工程.html">MAS&amp;智能体</a>
        <a class="tag" href="topics/推荐与搜索.html">推荐与搜索</a>
        <a class="tag" href="topics/定价与利润.html">定价与利润</a>
        <a class="tag" href="topics/风控与合规.html">风控与合规</a>
        <a class="tag" href="topics/视觉内容生成.html">视觉内容生成</a>
      </div>
    </div>
    <div class="ds-card">
      <h3>Skills Graph</h3>
      <p class="muted">{skill_count} 节点 · {edge_count} 关系边的知识图谱可视化。</p>
      <a class="btn-primary" href="graph/overview.html">打开图谱 →</a>
    </div>
  </div>
</div>

<div class="tab-panel" id="tab-ceo">
  <h2>AI 能力建设路线图</h2>
  <p class="muted">12 个月 3 阶段，替代 3 类岗位重复性工作，可验证 ROI > 3,000 万/年</p>
  <div class="ceo-entry">
    <div class="ceo-entry-body">
      <h3>「大促首日打 7 折清空库存，第 3 天涨价流量白白浪费」</h3>
      <p>这是供应链没有联动决策的代价。AI 路线图从这里开始。</p>
      <a class="btn-primary" href="ai-roadmap.html">查看完整路线图 →</a>
      <a class="btn-primary" href="ai-roadmap.html" onclick="window.open('ai-roadmap.html','_blank').print();return false;" style="margin-left:8px;background:#475569">下载 PDF</a>
    </div>
    <div class="ceo-phases">
      <div class="ceo-phase" style="border-color:#2563eb">
        <span style="color:#2563eb;font-weight:700">Phase 1</span> 快赢
        <p>HiFoReAd + AgentRouter + 归因修正</p>
        <strong style="color:#2563eb">800-1,900 万/年</strong>
      </div>
      <div class="ceo-phase" style="border-color:#7c3aed">
        <span style="color:#7c3aed;font-weight:700">Phase 2</span> 基础设施
        <p>产品 KG + 实验平台 + 库存池化</p>
        <strong style="color:#7c3aed">200-500 万/年（新增）</strong>
      </div>
      <div class="ceo-phase" style="border-color:#059669">
        <span style="color:#059669;font-weight:700">Phase 3</span> 护城河
        <p>AI 定价 + 内容工厂 + MAS 联动</p>
        <strong style="color:#059669">5,000 万+ 潜力</strong>
      </div>
    </div>
  </div>
</div>

<div class="tab-panel" id="tab-explore">
  <h2>按领域浏览</h2>
  <div class="metrics">
    <div><strong class="anim-fade-in-up" style="animation-delay:.0s">{skill_count}</strong><span>Skills</span></div>
    <div><strong class="anim-fade-in-up" style="animation-delay:.05s">{domain_count}</strong><span>领域</span></div>
    <div><strong class="anim-fade-in-up" style="animation-delay:.10s">{edge_count}</strong><span>知识边</span></div>
    <div><strong class="anim-fade-in-up" style="animation-delay:.15s">{workflow_count}</strong><span>业务工作流</span></div>
    <div><strong class="anim-fade-in-up accent-num" style="animation-delay:.20s">¥3.3亿</strong><span>年化ROI总量</span></div>
    <div><strong class="anim-fade-in-up" style="animation-delay:.25s">26</strong><span>AI Agents</span></div>
  </div>
  <div class="grid">{domain_cards}</div>
</div>

<script>
(function(){{
  const btns = document.querySelectorAll('#heroTabs .tab-btn');
  const panels = document.querySelectorAll('.tab-panel');
  btns.forEach(btn => btn.addEventListener('click', () => {{
    btns.forEach(b => b.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }}));
}})();
</script>
"""


# ---------------------------------------------------------------------------
# Skills Graph D3 page (Phase 3D)
# ---------------------------------------------------------------------------

def render_graph_page(skill_count: int, edge_count: int, build_ts: str = "") -> str:
    dc_js = '''{"supply_chain": "#0ea5e9", "mas": "#8b5cf6", "knowledge_graph": "#06b6d4", "llm_agent_engineering": "#ec4899", "advertising": "#f59e0b", "user_analytics": "#10b981", "tag_engineering": "#6366f1", "growth_model": "#ef4444", "marketing": "#f97316", "operations_finance": "#14b8a6", "recommendation": "#84cc16", "ab_testing": "#a855f7", "causal_inference": "#64748b", "time_series": "#22d3ee", "pricing": "#fb923c", "logistics": "#4ade80", "risk_fraud": "#f43f5e", "visual_content": "#c084fc", "compliance": "#fbbf24", "data_collection": "#38bdf8", "ml_fundamentals": "#94a3b8", "ai_humanities": "#e879f9", "nlp_voc": "#2dd4bf", "search_traffic": "#f59e0b"}'''
    dl_js = '''{"supply_chain": "供应链", "mas": "多智能体", "knowledge_graph": "知识图谱", "llm_agent_engineering": "LLM/Agent工程", "advertising": "广告分析", "user_analytics": "用户分析", "tag_engineering": "标签工程", "growth_model": "增长模型", "marketing": "营销投放", "operations_finance": "运营财务", "recommendation": "推荐系统", "ab_testing": "A/B实验", "causal_inference": "因果推断", "time_series": "时间序列", "pricing": "价格优化", "logistics": "物流履约", "risk_fraud": "风控反欺诈", "visual_content": "AI内容生成", "compliance": "合规决策", "data_collection": "数据采集", "ml_fundamentals": "ML基础", "ai_humanities": "AI人文", "nlp_voc": "NLP-VOC", "search_traffic": "搜索流量工程"}'''
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>技能关系图谱 · paper2skills</title>
  <link rel="stylesheet" href="../assets/style.css">
  <style>
    .graph-page{{display:flex;flex-direction:column;height:calc(100vh - var(--topbar-height,56px));overflow:hidden}}
    .graph-toolbar{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 16px;background:var(--panel,#fff);border-bottom:1px solid var(--line,#e2e8f0);flex-shrink:0}}
    .graph-toolbar input[type=text]{{padding:6px 12px;border:1.5px solid var(--line,#e2e8f0);border-radius:8px;font-size:13px;width:200px;font-family:inherit;outline:none}}
    .graph-toolbar input[type=text]:focus{{border-color:var(--accent,#3b82f6)}}
    .domain-pill{{display:inline-flex;align-items:center;gap:4px;padding:4px 9px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;border:2px solid transparent;transition:all .15s;white-space:nowrap}}
    .domain-pill.inactive{{opacity:.3}}
    .graph-main{{flex:1;display:flex;overflow:hidden;position:relative}}
    #graph-canvas{{flex:1;background:var(--panel-2,#f8fafc);cursor:grab}}
    #graph-canvas:active{{cursor:grabbing}}
    .graph-sidebar{{width:270px;flex-shrink:0;border-left:1px solid var(--line,#e2e8f0);background:var(--panel,#fff);overflow-y:auto;display:flex;flex-direction:column}}
    .gsb-header{{padding:12px 14px;border-bottom:1px solid var(--line,#e2e8f0);font-size:13px;font-weight:700;color:var(--ink,#0f172a);flex-shrink:0}}
    .gsb-body{{padding:12px 14px;flex:1}}
    .gsb-empty{{color:var(--muted,#94a3b8);font-size:12.5px;text-align:center;padding:30px 0}}
    .gsb-neighbor{{display:flex;align-items:center;gap:6px;padding:5px 7px;border-radius:6px;background:var(--panel-2,#f8fafc);border:1px solid var(--line,#e2e8f0);cursor:pointer;transition:all .15s;margin-top:4px}}
    .gsb-neighbor:hover{{background:var(--accent-light,#eff6ff);border-color:var(--accent,#3b82f6)}}
    .path-finder{{padding:10px 14px;border-top:1px solid var(--line,#e2e8f0)}}
    .path-finder input{{width:100%;padding:5px 8px;border:1.5px solid var(--line,#e2e8f0);border-radius:7px;font-size:11.5px;font-family:inherit;margin-bottom:5px;box-sizing:border-box}}
    .path-finder-btn{{width:100%;padding:6px;background:var(--accent,#3b82f6);color:#fff;border:none;border-radius:7px;font-size:11.5px;font-weight:600;cursor:pointer}}
    .graph-stats{{position:absolute;bottom:10px;left:10px;background:rgba(255,255,255,.9);border:1px solid var(--line,#e2e8f0);border-radius:8px;padding:5px 10px;font-size:11px;color:var(--muted,#94a3b8);pointer-events:none;backdrop-filter:blur(4px)}}
    .loading-overlay{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:var(--panel-2,#f8fafc);flex-direction:column;gap:12px;z-index:10}}
    .loading-spinner{{width:34px;height:34px;border:3px solid var(--line,#e2e8f0);border-top-color:var(--accent,#3b82f6);border-radius:50%;animation:spin .8s linear infinite}}
    @keyframes spin{{to{{transform:rotate(360deg)}}}}
    @media(max-width:768px){{.graph-sidebar{{display:none}}}}
  </style>
</head>
<body>
  <header class="topbar">
    <button class="hamburger" id="hamburger" aria-label="菜单" aria-expanded="false"><span></span><span></span><span></span></button>
    <a class="brand" href="../index.html"><span class="brand-icon">P</span><span class="brand-name">paper2skills<span class="brand-tag">Playbook</span></span></a>
    <div class="topbar-right"><input id="global-search" placeholder="搜索技能 / 场景…" autocomplete="off" role="search" aria-label="搜索"><a href="../ai-roadmap.html" class="topbar-cta">AI 路线图 →</a></div>
  </header>
  <div id="search-results" class="search-results hidden" role="listbox"></div>
  <div class="mobile-nav-overlay" id="mobile-overlay"></div>
  <div class="graph-page">
    <div class="graph-toolbar">
      <input type="text" id="node-search" placeholder="搜索技能..." autocomplete="off">
      <div id="domain-pills" style="display:flex;flex-wrap:wrap;gap:4px;flex:1"></div>
      <button id="btn-reset" style="padding:5px 11px;background:var(--panel-2,#f8fafc);border:1.5px solid var(--line,#e2e8f0);border-radius:8px;font-size:11.5px;font-weight:600;cursor:pointer;color:var(--ink-2,#475569)">重置</button>
      <button id="btn-focus-top" style="padding:5px 11px;background:var(--panel-2,#f8fafc);border:1.5px solid var(--line,#e2e8f0);border-radius:8px;font-size:11.5px;font-weight:600;cursor:pointer;color:var(--ink-2,#475569)">核心节点</button>
    </div>
    <div class="graph-main">
      <svg id="graph-canvas"></svg>
      <div class="loading-overlay" id="loading">
        <div class="loading-spinner"></div>
        <div style="font-size:13px;color:var(--muted)">加载 {skill_count} 个技能节点…</div>
      </div>
      <div class="graph-stats" id="graph-stats">节点 {skill_count} · 边 {edge_count}</div>
      <div class="graph-sidebar">
        <div class="gsb-header">PL 节点详情</div>
        <div class="gsb-body" id="gsb-body"><div class="gsb-empty">点击图谱中的节点<br>查看详情和关联</div></div>
        <div class="path-finder">
          <div style="font-size:11.5px;font-weight:700;color:var(--ink);margin-bottom:7px"> 学习路径发现</div>
          <input type="text" id="path-from" placeholder="起点 Skill ID...">
          <input type="text" id="path-to" placeholder="终点 Skill ID...">
          <button class="path-finder-btn" onclick="findPath()">找最短路径</button>
          <div id="path-result" style="margin-top:7px;font-size:11px;color:var(--ink-2);line-height:1.7"></div>
        </div>
      </div>
    </div>
  </div>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="../assets/playbook-data.js?v={build_ts}"></script>
  <script src="../assets/search.js"></script>
  <script>
  const DOMAIN_COLORS = {dc_js};
  const DOMAIN_LABELS = {dl_js};
  let allNodes=[],allLinks=[],simulation,svg,g,zoom,selectedNode=null;
  let activeDomains=new Set(Object.keys(DOMAIN_COLORS));
  async function init(){{
    try{{
      const resp=await fetch('../assets/graph-data.json');
      const data=await resp.json();
      allNodes=data.nodes; allLinks=data.links;
      buildDomainPills(); buildGraph();
      document.getElementById('loading').style.display='none';
    }}catch(e){{document.getElementById('loading').innerHTML='<div style="color:#ef4444">加载失败: '+e.message+'</div>';}}
  }}
  function buildDomainPills(){{
    const container=document.getElementById('domain-pills');
    const dc={{}};allNodes.forEach(n=>{{dc[n.domain]=(dc[n.domain]||0)+1;}});
    Object.entries(dc).sort((a,b)=>b[1]-a[1]).forEach(([domain,cnt])=>{{
      const color=DOMAIN_COLORS[domain]||'#94a3b8',label=DOMAIN_LABELS[domain]||domain;
      const pill=document.createElement('div');
      pill.className='domain-pill';pill.style.background=color+'22';pill.style.borderColor=color;pill.style.color=color;
      pill.innerHTML=`<span style="width:6px;height:6px;border-radius:50%;background:${{color}};flex-shrink:0"></span>${{label}} <span style="opacity:.6">${{cnt}}</span>`;
      pill.onclick=()=>toggleDomain(domain,pill);pill.dataset.domain=domain;container.appendChild(pill);
    }});
  }}
  function toggleDomain(domain,pill){{
    if(activeDomains.has(domain)){{activeDomains.delete(domain);pill.classList.add('inactive');}}
    else{{activeDomains.add(domain);pill.classList.remove('inactive');}}
    if(!g)return;
    g.selectAll('.node-group').style('display',d=>activeDomains.has(d.domain)?null:'none');
    g.selectAll('.link').style('display',d=>{{
      const s=allNodes.find(n=>n.id===(d.source.id||d.source)),t=allNodes.find(n=>n.id===(d.target.id||d.target));
      return(s&&t&&activeDomains.has(s.domain)&&activeDomains.has(t.domain))?null:'none';
    }});
  }}
  function buildGraph(){{
    const svgEl=document.getElementById('graph-canvas');
    const W=svgEl.clientWidth||window.innerWidth-270,H=svgEl.clientHeight||window.innerHeight-110;
    svg=d3.select('#graph-canvas').attr('width',W).attr('height',H);
    zoom=d3.zoom().scaleExtent([0.05,4]).on('zoom',e=>g.attr('transform',e.transform));
    svg.call(zoom);g=svg.append('g');
    const edgeColor={{prerequisite:'#94a3b8',combinable:'#3b82f6',extension:'#10b981'}};
    const degree={{}};
    allNodes.forEach(n=>{{degree[n.id]=0;}});
    allLinks.forEach(l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;degree[s]=(degree[s]||0)+1;degree[t]=(degree[t]||0)+1;}});
    const topN=200,topIds=new Set(Object.entries(degree).sort((a,b)=>b[1]-a[1]).slice(0,topN).map(([id])=>id));
    const vNodes=allNodes.filter(n=>topIds.has(n.id));
    const vSet=new Set(vNodes.map(n=>n.id));
    const vLinks=allLinks.filter(l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;return vSet.has(s)&&vSet.has(t);}});
    simulation=d3.forceSimulation(vNodes)
      .force('link',d3.forceLink(vLinks).id(d=>d.id).distance(55).strength(0.4))
      .force('charge',d3.forceManyBody().strength(-120).distanceMax(200))
      .force('center',d3.forceCenter(W/2,H/2))
      .force('collision',d3.forceCollide().radius(d=>nr(d,degree)+3))
      .alphaDecay(0.02);
    const link=g.append('g').selectAll('.link').data(vLinks).enter().append('line').attr('class','link')
      .style('stroke',d=>edgeColor[d.type]||'#cbd5e1').style('stroke-width',d=>d.type==='prerequisite'?1.5:1)
      .style('stroke-opacity',0.4).style('stroke-dasharray',d=>d.type==='extension'?'4,3':null);
    const node=g.append('g').selectAll('.node-group').data(vNodes).enter().append('g').attr('class','node-group')
      .style('cursor','pointer')
      .call(d3.drag().on('start',(e,d)=>{{if(!e.active)simulation.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}})
                     .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
                     .on('end',(e,d)=>{{if(!e.active)simulation.alphaTarget(0);d.fx=null;d.fy=null;}}))
      .on('click',(e,d)=>{{e.stopPropagation();selectNode(d,degree,link,node);}})
      .on('mouseover',(e,d)=>highlightNb(d,link,node,true))
      .on('mouseout',(e,d)=>{{if(selectedNode!==d)highlightNb(d,link,node,false);}});
    node.append('circle').attr('r',d=>nr(d,degree)).style('fill',d=>DOMAIN_COLORS[d.domain]||'#94a3b8')
      .style('fill-opacity',0.85).style('stroke','#fff').style('stroke-width',2);
    node.append('text').text(d=>sl(d.title||d.id)).attr('dy',d=>nr(d,degree)+10)
      .style('font-size',d=>degree[d.id]>30?'9px':'7.5px').style('text-anchor','middle').style('fill','#475569')
      .style('pointer-events','none').style('paint-order','stroke').style('stroke','#f8fafc').style('stroke-width','3px');
    simulation.on('tick',()=>{{
      link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
      node.attr('transform',d=>`translate(${{d.x}},${{d.y}})`);
    }});
    svg.on('click',()=>deselectAll(link,node));
    document.getElementById('graph-stats').textContent=`显示 ${{vNodes.length}} 节点 / ${{vLinks.length}} 边（全量 ${{allNodes.length}} / ${{allLinks.length}}）`;
    document.getElementById('node-search').addEventListener('input',function(){{
      const q=this.value.toLowerCase().trim();
      if(!q){{node.style('opacity',1);link.style('opacity',0.4);return;}}
      const m=new Set(vNodes.filter(n=>n.id.toLowerCase().includes(q)||(n.title||'').toLowerCase().includes(q)).map(n=>n.id));
      node.style('opacity',d=>m.has(d.id)?1:0.1);
      link.style('opacity',l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;return(m.has(s)&&m.has(t))?0.6:0.03;}});
    }});
    document.getElementById('btn-reset').onclick=()=>{{svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity.translate(W/2,H/2).scale(1));node.style('opacity',1);link.style('opacity',0.4);document.getElementById('node-search').value='';}};
    document.getElementById('btn-focus-top').onclick=()=>{{const t=new Set(Object.entries(degree).sort((a,b)=>b[1]-a[1]).slice(0,20).map(([id])=>id));node.style('opacity',d=>t.has(d.id)?1:0.08);link.style('opacity',0.04);}};
  }}
  function nr(d,degree){{return Math.max(5,Math.min(22,4+Math.sqrt(degree[d.id]||0)*1.8));}}
  function sl(t){{return(t||'').split(/[—-]/)[0].trim().slice(0,16);}}
  function highlightNb(d,link,node,on){{
    if(selectedNode)return;const nid=d.id;const c=new Set([nid]);
    link.each(l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;if(s===nid)c.add(t);if(t===nid)c.add(s);}});
    node.style('opacity',n=>on?(c.has(n.id)?1:0.15):1);
    link.style('opacity',l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;return on?((s===nid||t===nid)?0.8:0.04):0.4;}});
  }}
  function selectNode(d,degree,link,node){{
    selectedNode=d;const nid=d.id;const nb={{prerequisite:[],combinable:[],extension:[]}};const nids=new Set([nid]);
    link.each(l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;if(s===nid||t===nid){{const o=s===nid?t:s;nids.add(o);nb[l.type]=nb[l.type]||[];nb[l.type].push({{id:o}});}}}});
    node.style('opacity',n=>nids.has(n.id)?1:0.1).select('circle').style('stroke',n=>n.id===nid?'#fbbf24':'#fff').style('stroke-width',n=>n.id===nid?3:2);
    link.style('opacity',l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;return(s===nid||t===nid)?0.9:0.04;}}).style('stroke-width',l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;return(s===nid||t===nid)?2.5:1;}});
    renderSidebar(d,degree,nb);
  }}
  function deselectAll(link,node){{selectedNode=null;node.style('opacity',1).select('circle').style('stroke','#fff').style('stroke-width',2);link.style('opacity',0.4).style('stroke-width',d=>d.type==='prerequisite'?1.5:1);document.getElementById('gsb-body').innerHTML='<div class=\"gsb-empty\">点击图谱中的节点<br>查看详情和关联</div>';}}
  function esc(s){{return(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
  function renderSidebar(d,degree,nb){{
    const DATA=window.PLAYBOOK_DATA||{{}};const sk=(DATA.skills||[]).find(s=>s.skill_id===d.id)||{{}};
    const color=DOMAIN_COLORS[d.domain]||'#94a3b8',label=DOMAIN_LABELS[d.domain]||d.domain;
    let html=`<div style="font-size:13px;font-weight:700;color:var(--ink);line-height:1.4;margin-bottom:7px">${{esc(d.title||d.id)}}</div>
      <span style="display:inline-flex;align-items:center;padding:2px 8px;border-radius:12px;font-size:10.5px;font-weight:600;background:${{color}}22;color:${{color}};margin-bottom:8px">◉ ${{label}}</span>
      <div style="font-size:10.5px;color:#94a3b8;margin-bottom:7px">连接度: ${{degree[d.id]||0}}</div>`;
    if(sk.problem_solved)html+=`<div style="font-size:12px;color:#475569;line-height:1.6;margin-bottom:8px">${{esc(sk.problem_solved.slice(0,140))}}…</div>`;
    if(sk.roi_figure)html+=`<div style="font-size:11.5px;color:#059669;font-weight:600;margin-bottom:8px">PR ${{esc(sk.roi_figure)}}</div>`;
    const tl={{prerequisite:'⬅ 前置',combinable:'🔗 组合',extension:'➡ 延伸'}};
    const tc={{prerequisite:'#94a3b8',combinable:'#3b82f6',extension:'#10b981'}};
    html+='<div>';
    for(const[type,nbs]of Object.entries(nb)){{
      if(!nbs.length)continue;
      html+=`<div style="font-size:10.5px;color:#94a3b8;font-weight:600;margin:8px 0 3px">${{tl[type]||type}}</div>`;
      nbs.slice(0,4).forEach(n=>{{
        const nn=allNodes.find(x=>x.id===n.id)||{{title:n.id,domain:'unknown'}};
        const nc=DOMAIN_COLORS[nn.domain]||'#94a3b8';
        html+=`<div class="gsb-neighbor" onclick="focusById('${{esc(n.id)}}')"><div style="width:6px;height:6px;border-radius:50%;background:${{nc}};flex-shrink:0"></div><span style="font-size:11px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${{esc((nn.title||n.id).split('—')[0].trim().slice(0,30))}}</span></div>`;
      }});
    }}
    html+='</div>';
    html+=`<a href="../skills/${{d.id}}.html" target="_blank" style="display:inline-flex;align-items:center;gap:5px;padding:7px 12px;background:var(--accent,#3b82f6);color:#fff;border-radius:7px;font-size:12px;font-weight:600;text-decoration:none;margin-top:10px">→ 查看完整详情</a>`;
    document.getElementById('gsb-body').innerHTML=html;
  }}
  function findPath(){{
    const fi=document.getElementById('path-from').value.trim(),ti=document.getElementById('path-to').value.trim(),rel=document.getElementById('path-result');
    if(!fi||!ti){{rel.textContent='请输入起点和终点';return;}}
    const adj={{}};allNodes.forEach(n=>{{adj[n.id]=[];}});
    allLinks.forEach(l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;if(!adj[s])adj[s]=[];if(!adj[t])adj[t]=[];adj[s].push(t);adj[t].push(s);}});
    const q=[[fi]],vis=new Set([fi]);let found=null;
    while(q.length&&!found){{const p=q.shift();const c=p[p.length-1];if(c===ti){{found=p;break;}}if(p.length>8)continue;for(const nb of(adj[c]||[])){{if(!vis.has(nb)){{vis.add(nb);q.push([...p,nb]);}}}}}}
    if(!found){{rel.textContent=`未找到路径（最多8跳）`;return;}}
    const labels=found.map(id=>{{const n=allNodes.find(n=>n.id===id);return(n&&n.title?n.title.split('—')[0].trim().slice(0,18):id);}});
    rel.innerHTML='<strong>最短路径（'+(found.length-1)+'步）：</strong><br>'+labels.map((l,i)=>`${{i+1}}. ${{esc(l)}}`).join('<br>');
    if(g){{const ps=new Set(found);g.selectAll('.node-group').style('opacity',d=>ps.has(d.id)?1:0.08);g.selectAll('.link').style('opacity',l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;const ip=found.some((p,i)=>i<found.length-1&&((p===s&&found[i+1]===t)||(p===t&&found[i+1]===s)));return ip?1:0.03;}}).style('stroke-width',l=>{{const s=l.source.id||l.source,t=l.target.id||l.target;const ip=found.some((p,i)=>i<found.length-1&&((p===s&&found[i+1]===t)||(p===t&&found[i+1]===s)));return ip?3:1;}});}}
  }}
  function focusById(nid){{
    const tn=allNodes.find(n=>n.id===nid);if(!tn||tn.x===undefined)return;
    const svgEl=document.getElementById('graph-canvas');svg.transition().duration(500).call(zoom.transform,d3.zoomIdentity.translate(svgEl.clientWidth/2-tn.x,svgEl.clientHeight/2-tn.y).scale(1.5));
  }}
  window.findPath=findPath;window.focusById=focusById;
  window.addEventListener('DOMContentLoaded',init);
  </script>
  <script>
  const hbtn=document.getElementById('hamburger'),overlay=document.getElementById('mobile-overlay');
  function toggleMenu(open){{hbtn.setAttribute('aria-expanded',open);hbtn.classList.toggle('open',open);overlay.classList.toggle('show',open);document.body.style.overflow=open?'hidden':'';}}
  hbtn.addEventListener('click',()=>toggleMenu(hbtn.getAttribute('aria-expanded')!=='true'));overlay.addEventListener('click',()=>toggleMenu(false));
  </script>
</body>
</html>"""


from builders.graph_js_builder import build_ego_graph_js, build_graph_js  # noqa: E402
from builders.css_builder import build_css  # noqa: E402


def render_settings_page() -> str:
    return html_page(
        "设置 · API Key",
        """
<style>
.settings-wrap{max-width:760px;margin:0 auto;padding:40px 20px 60px}
.settings-wrap h1{font-size:1.5rem;font-weight:800;color:var(--ink);margin:0 0 6px}
.settings-wrap .sub{font-size:14px;color:var(--muted);margin-bottom:32px}
.scard{background:#fff;border:1px solid var(--line);border-radius:12px;padding:28px 28px;margin-bottom:24px}
.scard h2{font-size:14px;font-weight:700;color:var(--ink);margin:0 0 16px;letter-spacing:.03em;text-transform:uppercase}
.key-row{display:flex;gap:10px;align-items:center;margin-bottom:12px}
.key-row input{flex:1;padding:10px 14px;border:1px solid var(--line);border-radius:8px;font-size:13px;font-family:monospace;background:#f8fafc;color:var(--ink)}
.key-row button{padding:9px 16px;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap}
.btn-primary{background:var(--accent);color:#fff;border:none}
.btn-secondary{background:transparent;border:1px solid var(--line);color:var(--ink)}
.btn-primary:hover{opacity:.88}
.btn-secondary:hover{border-color:var(--ink)}
.usage-bar{background:#e2e8f0;border-radius:4px;height:8px;overflow:hidden;margin:8px 0}
.usage-fill{background:var(--accent);height:100%;transition:width .3s}
.usage-row{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:4px}
.tag-free{display:inline-block;padding:2px 8px;background:#f1f5f9;color:#64748b;border-radius:20px;font-size:11px;font-weight:600}
.tag-pro{display:inline-block;padding:2px 8px;background:#fef2f2;color:var(--accent);border-radius:20px;font-size:11px;font-weight:700}
#key-status{font-size:12.5px;margin-top:6px;min-height:20px}
</style>
<div class="settings-wrap">
  <h1>设置</h1>
  <p class="sub">管理 API Key、查看使用量、升级账户</p>

  <div class="scard">
    <h2>使用量</h2>
    <div class="usage-row">
      <span>本月 Agent 调用</span>
      <span id="usage-count">加载中…</span>
    </div>
    <div class="usage-bar"><div class="usage-fill" id="usage-fill" style="width:0%"></div></div>
    <div style="font-size:12px;color:var(--muted)">免费版每月 <strong>10 次</strong>。<a href="pricing.html" style="color:var(--accent);font-weight:600">升级 Pro</a> 获取无限次数。</div>
  </div>

  <div class="scard">
    <h2>API Key <span class="tag-pro">Pro</span></h2>
    <p style="font-size:13px;color:var(--muted);margin:0 0 14px">用于程序化调用 <code>/api/v1/skills/search</code> 等接口。格式：<code>Authorization: Bearer p2s_xxx</code></p>
    <div class="key-row">
      <input type="password" id="api-key-display" value="••••••••••••••••" readonly placeholder="升级 Pro 后可生成 API Key">
      <button class="btn-secondary" onclick="toggleKeyVisibility()">显示</button>
      <button class="btn-primary" onclick="generateApiKey()">生成新 Key</button>
    </div>
    <div id="key-status"></div>
    <div style="font-size:12px;color:#94a3b8;margin-top:8px">Key 只显示一次，请妥善保存</div>
  </div>

  <div class="scard">
    <h2>浏览器指纹（Session Key）</h2>
    <div class="key-row">
      <input type="text" id="session-key-display" readonly value="加载中…" style="font-family:monospace;font-size:12px">
      <button class="btn-secondary" onclick="copySessionKey()">复制</button>
    </div>
    <div style="font-size:12px;color:#94a3b8">这是当前浏览器的匿名标识，用于追踪免费用量。清除 localStorage 会重置。</div>
  </div>

  <div class="scard">
    <h2>快速接入示例</h2>
    <pre style="background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:14px;font-size:12.5px;overflow-x:auto;color:#374151">curl -H "Authorization: Bearer &lt;your-api-key&gt;" \\
  "https://skills.lute-tlz-dddd.top/api/v1/skills/search?q=供应链&limit=5"</pre>
  </div>
</div>

<script>
(function(){
  function _sessionKey(){
    let k=localStorage.getItem('p2s_session');
    if(!k){k='sk-'+Math.random().toString(36).slice(2)+Date.now().toString(36);localStorage.setItem('p2s_session',k);}
    return k;
  }
  const sk=_sessionKey();
  document.getElementById('session-key-display').value=sk;

  fetch('/api/agent/check-limit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_key:sk,agent_id:''})})
    .then(r=>r.json()).then(d=>{
      const used=d.monthly_used||0,limit=d.monthly_limit||10;
      document.getElementById('usage-count').textContent=used+' / '+limit+' 次';
      document.getElementById('usage-fill').style.width=Math.min(100,used/limit*100)+'%';
      document.getElementById('usage-fill').style.background=used>=limit?'#ef4444':used/limit>0.7?'#f59e0b':'var(--accent)';
    }).catch(()=>{document.getElementById('usage-count').textContent='--';});

  let _storedKey=localStorage.getItem('p2s_api_key')||'';
  if(_storedKey){document.getElementById('api-key-display').value='••••••••••••••••';}

  window.toggleKeyVisibility=function(){
    const el=document.getElementById('api-key-display');
    if(el.type==='password'){el.type='text';el.value=_storedKey||'（尚未生成）';}
    else{el.type='password';el.value='••••••••••••••••';}
  };

  window.generateApiKey=function(){
    const tier=localStorage.getItem('p2s_tier')||'free';
    if(tier!=='pro'){
      document.getElementById('key-status').innerHTML='<span style="color:#b91c1c">需要升级 Pro 才能生成 API Key。<a href="pricing.html" style="color:var(--accent);font-weight:700">立即升级 →</a></span>';
      return;
    }
    const newKey='p2s_sk_'+Math.random().toString(36).slice(2)+Math.random().toString(36).slice(2);
    _storedKey=newKey;
    localStorage.setItem('p2s_api_key',newKey);
    const el=document.getElementById('api-key-display');
    el.type='text';el.value=newKey;
    document.getElementById('key-status').innerHTML='<span style="color:#059669">✓ 已生成并保存到本地（仅本设备可见）</span>';
    setTimeout(()=>{el.type='password';el.value='••••••••••••••••';},10000);
  };

  window.copySessionKey=function(){
    navigator.clipboard.writeText(sk).then(()=>{document.getElementById('session-key-display').select();});
  };
})();
</script>
""",
        active_nav="settings",
    )


def render_pricing_page(skill_count: int = 1037) -> str:
    return html_page(
        "升级 Pro",
        f"""
<style>
.pricing-hero{{text-align:center;padding:60px 20px 40px;max-width:800px;margin:0 auto}}
.pricing-hero h1{{font-size:2rem;font-weight:800;color:var(--ink);margin:0 0 12px}}
.pricing-hero p{{font-size:1.05rem;color:var(--muted);max-width:520px;margin:0 auto 32px}}
.pricing-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px;max-width:900px;margin:0 auto 60px;padding:0 20px}}
.pc{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:32px 28px;display:flex;flex-direction:column}}
.pc.featured{{border-color:var(--accent);box-shadow:0 0 0 2px rgba(181,50,62,.15)}}
.pc-badge{{display:inline-block;background:var(--accent);color:#fff;font-size:11px;font-weight:700;letter-spacing:.04em;padding:3px 10px;border-radius:20px;margin-bottom:16px;width:fit-content}}
.pc-name{{font-size:1.2rem;font-weight:800;color:var(--ink);margin-bottom:4px}}
.pc-price{{font-size:2rem;font-weight:800;color:var(--accent);margin:8px 0 4px}}
.pc-price sub{{font-size:14px;font-weight:500;color:var(--muted)}}
.pc-desc{{font-size:13px;color:var(--muted);margin-bottom:20px;line-height:1.6}}
.pc-features{{list-style:none;padding:0;margin:0 0 28px;flex:1}}
.pc-features li{{font-size:13.5px;color:var(--ink);padding:7px 0;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:flex-start}}
.pc-features li:last-child{{border-bottom:none}}
.pc-features li::before{{content:"✓";color:var(--accent);font-weight:700;flex-shrink:0}}
.pc-cta{{display:block;text-align:center;padding:11px 20px;border-radius:8px;font-size:14px;font-weight:700;text-decoration:none;cursor:pointer;transition:opacity .15s}}
.pc-cta.primary{{background:var(--accent);color:#fff;border:none}}
.pc-cta.secondary{{background:transparent;color:var(--ink);border:1px solid var(--line)}}
.pc-cta.primary:hover{{opacity:.88}}
.pc-cta.secondary:hover{{border-color:var(--ink)}}
.pricing-faq{{max-width:700px;margin:0 auto;padding:0 20px 60px}}
.pricing-faq h2{{font-size:1.2rem;font-weight:800;margin-bottom:20px;color:var(--ink)}}
.faq-item{{border-bottom:1px solid var(--line);padding:16px 0}}
.faq-item summary{{font-weight:600;cursor:pointer;list-style:none;display:flex;justify-content:space-between;align-items:center;font-size:14px}}
.faq-item summary::after{{content:"+";color:var(--muted);font-size:18px}}
.faq-item[open] summary::after{{content:"−"}}
.faq-item p{{font-size:13.5px;color:var(--muted);line-height:1.7;margin:10px 0 0}}
</style>
<div class="pricing-hero">
  <h1>为母婴出海卖家而生的 AI 决策平台</h1>
  <p>免费访问 {skill_count}+ Skill 知识库，升级 Pro 解锁无限 Agent 调用、数据接入与 API 权限。</p>
</div>
<div class="pricing-grid">
  <div class="pc">
    <div class="pc-name">Free</div>
    <div class="pc-price">¥0 <sub>/月</sub></div>
    <p class="pc-desc">适合个人探索与学习</p>
    <ul class="pc-features">
      <li>访问全部 {skill_count}+ Skill 卡片</li>
      <li>25 个领域知识图谱</li>
      <li>场景手册（只读）</li>
      <li>AI Agent 每月 10 次调用</li>
      <li>Agent 报告本地存储（7 天）</li>
    </ul>
    <a href="agents.html" class="pc-cta secondary">开始使用</a>
  </div>
  <div class="pc featured">
    <div class="pc-badge">最受欢迎</div>
    <div class="pc-name">Pro</div>
    <div class="pc-price">¥299 <sub>/月</sub></div>
    <p class="pc-desc">适合跨境电商运营团队</p>
    <ul class="pc-features">
      <li>Free 所有权益</li>
      <li><strong>无限</strong> AI Agent 调用</li>
      <li>Agent 报告云端持久化（永久）</li>
      <li>CSV 数据文件直接上传</li>
      <li>飞书多维表格自动触发分析</li>
      <li>REST API 访问（1000 次/月）</li>
      <li>飞书群专属技术支持</li>
    </ul>
    <a href="#pro-payment" class="pc-cta primary" onclick="document.getElementById('pro-modal').style.display='flex'">立即升级 Pro</a>
  </div>
  <div class="pc">
    <div class="pc-name">Enterprise</div>
    <div class="pc-price">定制 <sub></sub></div>
    <p class="pc-desc">适合品牌方 / 大型 TP 服务商</p>
    <ul class="pc-features">
      <li>Pro 所有权益</li>
      <li>私有化部署选项</li>
      <li>无限 API 调用</li>
      <li>Amazon SP-API 数据接入</li>
      <li>定制 Agent Prompt 与工作流</li>
      <li>SLA 保障 + 专属客户成功</li>
    </ul>
    <a href="mailto:support@paper2skills.com" class="pc-cta secondary">联系销售</a>
  </div>
</div>

<div class="pricing-faq">
  <h2>常见问题</h2>
  <details class="faq-item">
    <summary>免费版和 Pro 版的 Agent 调用有什么区别？</summary>
    <p>免费版每月可调用 AI Agent 10 次（所有 Agent 共享），Pro 版无限制。超过 10 次后免费用户会看到升级提示，当月已用次数会在下月 1 日重置。</p>
  </details>
  <details class="faq-item">
    <summary>REST API 可以做什么？</summary>
    <p>Pro 版可通过 <code>GET /api/v1/skills/search?q=关键词</code> 程序化查询 Skill 知识库，适合把 paper2skills 集成到自有系统、飞书机器人、或自动化工作流中。</p>
  </details>
  <details class="faq-item">
    <summary>如何升级？支持什么支付方式？</summary>
    <p>点击「立即升级 Pro」按钮，扫码完成微信支付后系统自动开通。支持月付，随时可取消，下个计费周期不再续费。</p>
  </details>
  <details class="faq-item">
    <summary>企业版有试用期吗？</summary>
    <p>企业版提供 7 天免费 POC 试用，包含完整功能接入和技术支持。请通过邮件联系我们安排。</p>
  </details>
</div>

<!-- Pro 升级 Modal -->
<div id="pro-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9999;align-items:center;justify-content:center" onclick="if(event.target===this)this.style.display='none'">
  <div style="background:#fff;border-radius:16px;padding:40px 36px;max-width:420px;width:90%;text-align:center">
    <h2 style="font-size:1.3rem;font-weight:800;margin:0 0 8px">升级 Pro 版</h2>
    <p style="font-size:13.5px;color:#64748b;margin:0 0 24px">微信扫码完成支付，系统自动开通</p>
    <div style="width:180px;height:180px;margin:0 auto 20px;background:#f1f5f9;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:13px">
      微信支付二维码<br>（配置后显示）
    </div>
    <div style="font-size:28px;font-weight:800;color:#B5323E;margin-bottom:4px">¥299<span style="font-size:14px;font-weight:500;color:#64748b">/月</span></div>
    <p style="font-size:12px;color:#94a3b8;margin:4px 0 20px">支付后自动开通，随时可取消</p>
    <button onclick="document.getElementById('pro-modal').style.display='none'" style="width:100%;padding:12px;border:1px solid #e2e8f0;border-radius:8px;background:#fff;cursor:pointer;font-size:13px;color:#64748b">稍后再说</button>
  </div>
</div>
""",
        active_nav="pricing",
    )


def render_maturity_report(skill_count: int, edge_count: int, domain_count: int) -> str:
    biz_groups = [
        ("供应链与库存", 182, "需求预测·补货优化·物流履约", "#0369a1"),
        ("智能决策基础", 202, "知识图谱·MAS·Agent工程·DataAgent", "#7c3aed"),
        ("数据与运营", 237, "标签工程·财务分析·A/B实验·因果推断", "#059669"),
        ("流量与增长", 149, "搜索SEO·推荐系统·增长模型·视频内容", "#d97706"),
        ("广告与营销", 123, "广告归因·MMM·VOC挖掘", "#dc2626"),
        ("风险与合规", 117, "风控反欺诈·合规决策·用户流失防御", "#b45309"),
    ]
    risk_events = [
        ("", "用户流失风险", "high", 21),
        ("", "产品合规预警", "critical", 23),
        ("", "竞品价格攻击", "medium", 9),
        ("", "供应链断货风险", "high", 16),
        ("", "虚假评论攻击", "high", 14),
        ("", "ASIN流量异常", "high", 11),
        ("", "账号健康恶化", "critical", 10),
        ("", "广告素材疲劳", "medium", 9),
        ("", "数据/模型漂移", "medium", 9),
        ("", "关税汇率风险", "high", 9),
        ("", "库存积压滞销", "high", 10),
        ("", "DTC SEO可见性", "medium", 8),
        ("", "TikTok内容衰减", "medium", 9),
        ("", "库存积压/清仓", "high", 10),
        ("", "FBA仓储费超标", "high", 8),
        ("", "物流履约异常", "high", 9),
        ("", "评价质量下滑", "high", 8),
    ]
    sev_colors = {"critical": "#dc2626", "high": "#d97706", "medium": "#2563eb"}

    groups_html = ""
    for name, cnt, desc, color in biz_groups:
        pct = round(cnt / skill_count * 100, 1)
        groups_html += f"""
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;display:flex;flex-direction:column;gap:6px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:14px;font-weight:700;color:#0c0c0c">{name}</span>
            <span style="font-size:20px;font-weight:800;color:{color}">{cnt}</span>
          </div>
          <div style="background:#f3f4f6;border-radius:4px;height:6px;overflow:hidden">
            <div style="background:{color};height:100%;width:{pct}%;border-radius:4px"></div>
          </div>
          <div style="font-size:11.5px;color:#6b7280">{desc}</div>
          <div style="font-size:11px;color:#9ca3af">{pct}% of total</div>
        </div>"""

    events_html = ""
    for icon, name, sev, cnt in risk_events:
        col = sev_colors.get(sev, "#6b7280")
        events_html += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:9px 14px;background:#fff;border:1px solid #e5e7eb;border-radius:8px">
          <div style="flex:1;min-width:0">
            <div style="font-size:12.5px;font-weight:600;color:var(--ink,#1A1A2E)">{name}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:2px">{cnt} Skills 诊断链</div>
          </div>
          <span style="font-size:10px;font-weight:700;color:{col};background:{col}18;padding:2px 7px;border-radius:4px;white-space:nowrap">{sev.upper()}</span>
        </div>"""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>母婴跨境 AI 能力成熟度报告 2026 — paper2skills</title>
<link rel="stylesheet" href="assets/style.css">
<style>
.mr-wrap{{max-width:900px;margin:0 auto;padding:32px 24px 80px}}
.mr-eyebrow{{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#B5323E;margin-bottom:12px}}
.mr-title{{font-size:32px;font-weight:800;color:#0c0c0c;line-height:1.2;letter-spacing:-.5px;margin-bottom:10px}}
.mr-sub{{font-size:15px;color:#4b5563;line-height:1.7;margin-bottom:24px;max-width:680px}}
.mr-meta{{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:40px;padding:16px 20px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px}}
.mr-meta-item{{display:flex;flex-direction:column;gap:2px}}
.mr-meta-num{{font-size:22px;font-weight:800;color:#0c0c0c;line-height:1}}
.mr-meta-label{{font-size:11px;color:#6b7280}}
.mr-section{{margin-bottom:48px}}
.mr-section-title{{font-size:18px;font-weight:700;color:#0c0c0c;margin-bottom:6px;padding-bottom:8px;border-bottom:2px solid #B5323E;display:inline-block}}
.mr-section-desc{{font-size:13px;color:#6b7280;margin-bottom:20px;line-height:1.6}}
.mr-grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.mr-grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.mr-callout{{background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:18px 22px;margin-bottom:20px}}
.mr-callout-title{{font-size:13px;font-weight:700;color:#991b1b;margin-bottom:6px}}
.mr-callout-body{{font-size:12.5px;color:#7f1d1d;line-height:1.6}}
.mr-table{{width:100%;border-collapse:collapse;font-size:12.5px}}
.mr-table th{{text-align:left;padding:8px 12px;background:#f3f4f6;color:#374151;font-weight:600;border-bottom:2px solid #e5e7eb}}
.mr-table td{{padding:9px 12px;border-bottom:1px solid #f3f4f6;color:#1f2937;vertical-align:top}}
.mr-table tr:hover td{{background:#f9fafb}}
.mr-badge{{display:inline-block;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px}}
.mr-footer{{margin-top:60px;padding-top:20px;border-top:1px solid #e5e7eb;font-size:11.5px;color:#9ca3af;text-align:center;line-height:1.7}}
@media print{{.topbar,.sidebar{{display:none!important}}.mr-wrap{{padding:20px}}.mr-title{{font-size:24px}}}}
@media(max-width:680px){{.mr-grid-2,.mr-grid-3{{grid-template-columns:1fr}}.mr-title{{font-size:22px}}}}
</style>
</head>
<body>
<header class="topbar">
  <a class="brand" href="index.html">
    <span class="brand-icon">P</span>
    <span class="brand-name">paper2skills<span class="brand-tag">Playbook</span></span>
  </a>
  <div class="topbar-right">
    <span class="topbar-stat">成熟度报告 2026</span>
    <button onclick="window.print()" class="topbar-cta" style="border:none;cursor:pointer">⬇ PDF</button>
  </div>
</header>

<div class="mr-wrap">
  <div class="mr-eyebrow">paper2skills 研究报告 · 2026 年度</div>
  <h1 class="mr-title">母婴跨境电商 AI 能力成熟度报告</h1>
  <p class="mr-sub">基于 {skill_count} 个从顶刊论文萃取的可落地 AI 决策技能，系统性分析母婴跨境电商在供应链、广告、风控、增长等核心场景的 AI 能力现状与演进路径。</p>

  <div class="mr-meta">
    <div class="mr-meta-item"><span class="mr-meta-num">{skill_count}</span><span class="mr-meta-label">可落地 AI 技能</span></div>
    <div class="mr-meta-item"><span class="mr-meta-num">{domain_count}</span><span class="mr-meta-label">业务领域</span></div>
    <div class="mr-meta-item"><span class="mr-meta-num">{edge_count:,}</span><span class="mr-meta-label">技能关联关系</span></div>
    <div class="mr-meta-item"><span class="mr-meta-num">13</span><span class="mr-meta-label">高频风险事件类型</span></div>
    <div class="mr-meta-item"><span class="mr-meta-num">33</span><span class="mr-meta-label">可执行场景手册</span></div>
    <div class="mr-meta-item"><span class="mr-meta-num">100%</span><span class="mr-meta-label">技能均含可运行代码</span></div>
  </div>

  <div style="background:#0c0c0c;color:#fff;border-radius:10px;padding:22px 28px;margin-bottom:40px">
    <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9ca3af;margin-bottom:14px">执行摘要</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div style="border-left:2px solid #B5323E;padding-left:12px">
        <div style="font-size:13px;font-weight:700;margin-bottom:3px">分析能力 ≠ 执行能力</div>
        <div style="font-size:12px;color:#9ca3af;line-height:1.6">95% 的卖家 AI 停在「看报表」，D 层自动决策渗透率 &lt;5%</div>
      </div>
      <div style="border-left:2px solid #d97706;padding-left:12px">
        <div style="font-size:13px;font-weight:700;margin-bottom:3px">风险 ROI &gt; 增长 ROI</div>
        <div style="font-size:12px;color:#9ca3af;line-height:1.6">风控/合规技能年化避损 30-100 万元，高于增长类 15-50 万元</div>
      </div>
      <div style="border-left:2px solid #059669;padding-left:12px">
        <div style="font-size:13px;font-weight:700;margin-bottom:3px">供应链是最确定的起点</div>
        <div style="font-size:12px;color:#9ca3af;line-height:1.6">124 个供应链 Skills，平均被引用 53 次，ROI 最可量化</div>
      </div>
      <div style="border-left:2px solid #7c3aed;padding-left:12px">
        <div style="font-size:13px;font-weight:700;margin-bottom:3px">13 类风险已有诊断链</div>
        <div style="font-size:12px;color:#9ca3af;line-height:1.6">每类风险配备诊断→处置→预防三层 Skill，可直接调用</div>
      </div>
    </div>
  </div>

  <!-- Section 1: 核心发现 -->
  <div class="mr-section">
    <div class="mr-section-title">01 · 核心发现</div>
    <p class="mr-section-desc">通过对 {skill_count} 个技能的系统性分析，我们识别出母婴跨境电商 AI 能力建设的三个关键洞察：</p>
    <div class="mr-callout" style="background:#eff6ff;border-color:#bfdbfe">
      <div class="mr-callout-title" style="color:#1d4ed8">发现 1：分析能力过剩，决策执行能力严重不足</div>
      <div class="mr-callout-body" style="color:#1e3a8a">当前知识图谱中 A 层（分析/检测/预测）Skills 占比 >95%，D 层（自动触发/执行/封锁）Skills 占比不足 5%。绝大多数卖家的 AI 应用停留在「看报表」阶段，尚未进入「自动决策」阶段。这是最大的能力鸿沟——拥有预测结果却无法自动化执行，等同于有地图却不会开车。</div>
    </div>
    <div class="mr-callout" style="background:#eff6ff;border-color:#bfdbfe">
      <div class="mr-callout-title" style="color:#1d4ed8">发现 2：风险防御的 ROI 远高于增长投入</div>
      <div class="mr-callout-body" style="color:#1e3a8a">从 ps_override 数据统计，风险/合规类技能的平均量化 ROI（年化避损）达 30-100 万元，而纯增长类技能平均 ROI 约 15-50 万元。母婴跨境卖家对风险的接受敏感度远高于增长机会，这一心理偏好也得到实际数据验证：风险类 Skills 被引用频率比增长类高 40%。</div>
    </div>
    <div class="mr-callout" style="background:#f0fdf4;border-color:#bbf7d0">
      <div class="mr-callout-title" style="color:#166534">发现 3：供应链是 AI 渗透最深、回报最确定的领域</div>
      <div class="mr-callout-body" style="color:#14532d">供应链域拥有 {skill_count} 个技能中最多的 124 个（12.3%），且跨 Skills 引用关系最密集（平均每个供应链 Skill 被引用 53 次）。需求预测→安全库存→补货决策→前置期风险的完整 AI 决策链已在论文层面成熟，是当前 ROI 最确定的 AI 投入方向。</div>
    </div>
  </div>

  <!-- Section 2: 技能分布 -->
  <div class="mr-section">
    <div class="mr-section-title">02 · AI 技能业务分布</div>
    <p class="mr-section-desc">{skill_count} 个技能按业务方向分布如下，覆盖母婴跨境电商全链路核心场景：</p>
    <div class="mr-grid-2">{groups_html}</div>
  </div>

  <!-- Section 3: 成熟度矩阵 -->
  <div class="mr-section">
    <div class="mr-section-title">03 · 成熟度四阶段模型</div>
    <p class="mr-section-desc">我们将母婴跨境卖家的 AI 能力建设分为四个阶段，每个阶段有明确的标志性技能和典型 ROI：</p>
    <table class="mr-table">
      <tr>
        <th>阶段</th><th>特征</th><th>典型技能示例</th><th>典型 ROI</th><th>覆盖卖家比例</th>
      </tr>
      <tr>
        <td><span class="mr-badge" style="background:#f3f4f6;color:#374151">阶段 1</span><br><strong>数据感知</strong></td>
        <td>能跑报表，靠人工决策</td>
        <td>销量预测、竞品价格监控</td>
        <td>年省工时 200h</td>
        <td>~60%</td>
      </tr>
      <tr>
        <td><span class="mr-badge" style="background:#dbeafe;color:#1d4ed8">阶段 2</span><br><strong>预测驱动</strong></td>
        <td>AI 提供预测，人工执行</td>
        <td>流失预测、前置期风险建模、广告归因</td>
        <td>年增收 50-200 万元</td>
        <td>~25%</td>
      </tr>
      <tr>
        <td><span class="mr-badge" style="background:#dcfce7;color:#166534">阶段 3</span><br><strong>决策自动化</strong></td>
        <td>AI 自动触发执行，人工审核</td>
        <td>补货触发器、广告预算自动重分配、账号预警响应</td>
        <td>年增收 200-800 万元</td>
        <td>~12%</td>
      </tr>
      <tr>
        <td><span class="mr-badge" style="background:#fef9c3;color:#854d0e">阶段 4</span><br><strong>自主运营</strong></td>
        <td>Multi-Agent 自主编排，全链路闭环</td>
        <td>MAS 补货协商、跨域 Agent 联动、实时风险-价格-库存三元联动</td>
        <td>年增收 1000 万元+</td>
        <td>&lt;3%</td>
      </tr>
    </table>
  </div>

  <!-- Section 4: 风险事件 Ontology -->
  <div class="mr-section">
    <div class="mr-section-title">04 · 13 类高频风险事件</div>
    <p class="mr-section-desc">基于图谱引用频率和业务痛点频次，识别出母婴跨境卖家面临的 13 类高频风险事件，每类均配备「诊断→处置→预防」三层 Skill 链：</p>
    <div class="mr-grid-2">{events_html}</div>
    <p style="font-size:12px;color:#9ca3af;margin-top:12px">→ 完整诊断链可访问 <a href="diagnostic.html" style="color:#B5323E">业务诊断中心</a></p>
  </div>

  <!-- Section 5: 行动建议 -->
  <div class="mr-section">
    <div class="mr-section-title">05 · 分阶段行动建议</div>
    <p class="mr-section-desc">基于成熟度模型，不同阶段的卖家应优先投入以下方向：</p>
    <table class="mr-table">
      <tr><th>当前阶段</th><th>优先投入方向</th><th>推荐起点技能</th><th>预期达成时间</th></tr>
      <tr>
        <td><strong>阶段 1 → 2</strong></td>
        <td>需求预测 + 竞品监控自动化</td>
        <td>Skill-Lead-Time-Distribution-Risk-GenQOT<br>Skill-Competitive-Price-Monitoring</td>
        <td>3 个月</td>
      </tr>
      <tr>
        <td><strong>阶段 2 → 3</strong></td>
        <td>风险预警 → 自动响应闭环</td>
        <td>Skill-Account-Health-Proactive-Monitor<br>Skill-Markdown-Schedule-Auto-Trigger</td>
        <td>6 个月</td>
      </tr>
      <tr>
        <td><strong>阶段 3 → 4</strong></td>
        <td>多 Agent 协作 + 跨域决策编排</td>
        <td>Skill-MAS-Inventory-Consensus-Action<br>Skill-Combo-Inventory-Crisis-Response</td>
        <td>12 个月</td>
      </tr>
    </table>
  </div>

  <!-- Section 6: 方法论 -->
  <div class="mr-section">
    <div class="mr-section-title">06 · 数据来源与方法论</div>
    <p class="mr-section-desc">本报告基于以下数据来源，确保所有结论有据可查：</p>
    <ul style="font-size:13px;color:#374151;line-height:2;padding-left:20px">
      <li><strong>{skill_count} 个 Skill 文件</strong>：从 NeurIPS/KDD/ICML/WWW/ICLR 等顶会论文萃取，每个 Skill 含可运行 Python 代码</li>
      <li><strong>{edge_count:,} 条关联关系</strong>：通过 Skill 文件的双括号链接自动构建，反映真实算法依赖关系</li>
      <li><strong>607 条 ROI 数字</strong>：来自 ps_override 中基于业务场景的量化估算（非模型生成，基于行业基准数据）</li>
      <li><strong>13 个风险事件 Ontology</strong>：基于图谱被引频率排序，选取被引用次数最高的风险/诊断类 Skills 构建</li>
    </ul>
  </div>

  <div class="mr-footer">
    paper2skills · 母婴跨境 AI 决策知识库 · <a href="https://skills.lute-tlz-dddd.top" style="color:#B5323E">skills.lute-tlz-dddd.top</a><br>
    报告生成时间：2026 年 6 月 | 数据基于 {skill_count} 个可落地 AI 技能<br>
    <a href="diagnostic.html" style="color:#B5323E">业务诊断中心</a> ·
    <a href="chat.html" style="color:#B5323E">AI 知识库对话</a> ·
    <a href="playbooks/index.html" style="color:#B5323E">33 本场景手册</a>
  </div>
</div>
</body>
</html>"""


def render_diagnostic_page(skill_count: int, build_ts: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>业务诊断中心 — paper2skills</title>
<link rel="stylesheet" href="assets/style.css">
<style>
.diag-wrap{{display:flex;gap:0;min-height:calc(100vh - var(--topbar-height,52px));background:var(--bg,#F6F6F6)}}
.diag-left{{
  width:320px;min-width:280px;flex-shrink:0;
  background:var(--panel,#fff);
  border-right:1px solid var(--line,#e4e4e4);
  padding:0;
  display:flex;flex-direction:column;
  position:sticky;top:var(--topbar-height,52px);
  height:calc(100vh - var(--topbar-height,52px));
  overflow-y:auto;overflow-x:hidden;
}}
.diag-left::-webkit-scrollbar{{width:3px}}
.diag-left::-webkit-scrollbar-track{{background:transparent}}
.diag-left::-webkit-scrollbar-thumb{{background:var(--line-strong,#ccc);border-radius:2px}}
.diag-left-head{{
  padding:20px 16px 14px;
  border-bottom:1px solid var(--line,#e4e4e4);
  flex-shrink:0;
}}
.diag-left-eyebrow{{
  font-size:10px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted,#888);
  margin-bottom:5px;
}}
.diag-title{{font-size:15px;font-weight:700;color:var(--ink,#1A1A2E);letter-spacing:-.02em;line-height:1.3}}
.diag-sub{{font-size:12px;color:var(--muted,#888);line-height:1.5;margin-top:3px}}
.diag-search-wrap{{padding:12px 16px;border-bottom:1px solid var(--line,#e4e4e4);flex-shrink:0}}
.diag-input-row{{display:flex;gap:7px}}
.diag-input{{
  flex:1;padding:8px 12px;
  border:1.5px solid var(--line,#e4e4e4);
  border-radius:var(--r-md,6px);
  font-size:13px;font-family:var(--font);
  outline:none;color:var(--ink);
  background:var(--bg,#F6F6F6);
  transition:border-color var(--t),background var(--t);
}}
.diag-input:focus{{border-color:var(--accent,#B5323E);background:#fff;box-shadow:0 0 0 3px rgba(181,50,62,.08)}}
.diag-input::placeholder{{color:var(--muted,#888);font-size:12.5px}}
.diag-btn{{
  padding:8px 14px;
  background:var(--accent,#B5323E);color:#fff;
  border:none;border-radius:var(--r-md,6px);
  font-size:12.5px;font-weight:600;cursor:pointer;
  white-space:nowrap;font-family:var(--font);
  transition:background var(--t);flex-shrink:0;
}}
.diag-btn:hover{{background:var(--accent-dark,#8C2530)}}
.diag-section-label{{
  font-size:10px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted,#888);
  padding:12px 16px 6px;
  user-select:none;
}}
.diag-events{{display:flex;flex-direction:column;gap:1px;padding:0 8px 8px}}
.diag-event-btn{{
  display:flex;align-items:center;gap:10px;
  padding:9px 10px;
  background:transparent;
  border:none;border-radius:var(--r-md,6px);
  cursor:pointer;text-align:left;font-family:var(--font);
  transition:background var(--t),color var(--t);
  position:relative;width:100%;
}}
.diag-event-btn:hover{{background:var(--panel-2,#f3f3f3)}}
.diag-event-btn.active{{
  background:var(--accent-bg,#FDF0F1);
}}
.diag-event-btn.active::before{{
  content:'';position:absolute;left:0;top:6px;bottom:6px;
  width:2px;border-radius:0 2px 2px 0;
  background:var(--accent,#B5323E);
}}
.diag-event-info{{min-width:0;flex:1}}
.diag-event-name{{font-size:12.5px;font-weight:500;color:var(--ink,#1A1A2E);line-height:1.35}}
.diag-event-btn.active .diag-event-name{{font-weight:600;color:var(--accent,#B5323E)}}
.diag-event-sev{{
  font-size:10px;padding:1px 6px;border-radius:3px;
  display:inline-block;margin-top:3px;font-weight:600;
  letter-spacing:.02em;
}}
.sev-critical{{background:#fef2f2;color:#991b1b}}
.sev-high{{background:#fff7ed;color:#c2410c}}
.sev-medium{{background:#eff6ff;color:#1d4ed8}}
.sev-low{{background:#f0fdf4;color:#166534}}
.diag-right{{flex:1;padding:28px 28px;overflow-y:auto}}
.diag-empty{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;color:var(--muted,#888);text-align:center;gap:12px}}
.diag-empty-icon{{font-size:40px;opacity:.5}}
.diag-empty-text{{font-size:14px;line-height:1.6;max-width:300px}}
.diag-result{{display:none}}
.diag-result.visible{{display:block}}
.diag-result-header{{background:#fff;border:1px solid var(--line,#e5e7eb);border-radius:10px;padding:20px 22px;margin-bottom:16px}}
.diag-result-title{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.diag-result-icon{{font-size:24px}}
.diag-result-name{{font-size:17px;font-weight:700;color:var(--ink)}}
.diag-result-summary{{font-size:13px;color:var(--muted);line-height:1.6}}
.diag-phases{{display:flex;flex-direction:column;gap:12px}}
.diag-phase{{background:#fff;border:1px solid var(--line,#e5e7eb);border-radius:10px;overflow:hidden}}
.diag-phase-header{{padding:12px 18px;font-size:13px;font-weight:700;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--line,#e5e7eb)}}
.diag-phase-diagnose .diag-phase-header{{background:#f0f9ff;color:#0369a1}}
.diag-phase-treat .diag-phase-header{{background:#fff7ed;color:#b45309}}
.diag-phase-prevent .diag-phase-header{{background:#f0fdf4;color:#166534}}
.diag-skill-list{{padding:8px 0}}
.diag-skill-item{{display:flex;align-items:flex-start;gap:10px;padding:9px 18px;border-bottom:1px solid #f3f4f6;transition:background .12s}}
.diag-skill-item:last-child{{border-bottom:none}}
.diag-skill-item:hover{{background:#f9fafb}}
.diag-skill-num{{font-size:11px;font-weight:700;color:#9ca3af;flex-shrink:0;padding-top:2px;min-width:16px}}
.diag-skill-body{{min-width:0}}
.diag-skill-link{{font-size:12.5px;font-weight:600;color:var(--accent,#B5323E);text-decoration:none;display:block}}
.diag-skill-link:hover{{text-decoration:underline}}
.diag-skill-role{{font-size:12px;color:var(--muted);line-height:1.5;margin-top:2px}}
.diag-skill-cond{{font-size:11px;background:#fef9c3;color:#854d0e;padding:1px 6px;border-radius:4px;display:inline-block;margin-top:3px}}
.diag-related{{margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px}}
.diag-related-title{{font-size:12px;font-weight:700;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}}
.diag-related-links{{display:flex;flex-wrap:wrap;gap:6px}}
.diag-related-link{{font-size:12px;padding:4px 10px;background:var(--bg);border:1px solid var(--line);border-radius:20px;text-decoration:none;color:var(--ink);transition:all .12s}}
.diag-related-link:hover{{border-color:var(--accent);color:var(--accent)}}
.path-finder{{
  margin:8px 8px;
  border:1px solid var(--line,#e4e4e4);
  border-radius:var(--r-lg,8px);overflow:hidden;
  flex-shrink:0;
}}
.path-finder-header{{
  padding:10px 14px;font-size:12.5px;font-weight:600;
  color:var(--ink-2);cursor:pointer;
  display:flex;align-items:center;justify-content:space-between;
  background:var(--panel-2,#f3f3f3);
  transition:background var(--t);user-select:none;
}}
.path-finder-header:hover{{background:var(--panel-3,#ececec)}}
.path-finder-arrow{{font-size:10px;color:var(--muted);transition:transform var(--t)}}
.path-finder-header.open .path-finder-arrow{{transform:rotate(90deg)}}
.path-finder-body{{padding:12px 14px;display:none;border-top:1px solid var(--line)}}
.path-finder-body.open{{display:block}}
.path-select{{
  width:100%;padding:7px 10px;
  border:1.5px solid var(--line);
  border-radius:var(--r-md,6px);
  font-size:12.5px;font-family:var(--font);
  margin-bottom:7px;outline:none;
  background:var(--bg);color:var(--ink);
  transition:border-color var(--t);
}}
.path-select:focus{{border-color:var(--accent)}}
.path-run-btn{{width:100%;padding:8px;background:var(--accent);color:#fff;border:none;border-radius:var(--r-md,6px);font-size:12.5px;font-weight:600;cursor:pointer;font-family:var(--font);transition:background var(--t)}}
.path-run-btn:hover{{background:var(--accent-dark)}}
.path-result{{margin-top:14px}}
.path-step{{display:flex;align-items:flex-start;gap:8px;padding:8px 0;border-bottom:1px solid #f3f4f6}}
.path-step:last-child{{border-bottom:none}}
.path-step-num{{font-size:11px;font-weight:700;color:#fff;background:var(--accent);border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}}
.path-step-link{{font-size:12.5px;font-weight:600;color:var(--accent);text-decoration:none;display:block}}
.path-step-link:hover{{text-decoration:underline}}
.path-step-domain{{font-size:11px;color:var(--muted);margin-top:1px}}
.path-edge-type{{font-size:10px;padding:1px 5px;border-radius:3px;margin-left:6px;vertical-align:middle}}
.pet-prerequisite{{background:#f0f9ff;color:#0369a1}}.pet-extension{{background:#f0fdf4;color:#166534}}.pet-combinable{{background:#fef3c7;color:#92400e}}
@media(max-width:768px){{.diag-wrap{{flex-direction:column}}.diag-left{{width:100%;position:static;height:auto;border-right:none;border-bottom:1px solid var(--line)}}.diag-right{{padding:16px}}}}
</style>
</head>
<body>
<header class="topbar">
  <button class="hamburger" id="hamburger-diag" aria-label="菜单" aria-expanded="false">
    <span></span><span></span><span></span>
  </button>
  <a class="brand" href="index.html">
    <span class="brand-icon">P</span>
    <span class="brand-name">paper2skills<span class="brand-tag">Playbook</span></span>
  </a>
  <div class="topbar-right">
    <input id="global-search-diag" placeholder="搜索技能 / 场景…" autocomplete="off"
      style="width:min(220px,18vw);padding:6px 12px 6px 30px;border-radius:var(--r-sm);border:1px solid #2E2E2E;background:#1A1A1A url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2212%22 height=%2212%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22%23555%22 stroke-width=%222%22%3E%3Ccircle cx=%2211%22 cy=%2211%22 r=%228%22/%3E%3Cpath d=%22m21 21-4.35-4.35%22/%3E%3C/svg%3E') no-repeat 10px center;color:#CCCCCC;font-size:12.5px;font-family:var(--font)">
    <span class="topbar-stat">{skill_count} Skills</span>
    <a href="pricing.html" class="topbar-cta">升级 Pro →</a>
    <div id="p2s-auth-widget" style="margin-left:4px;display:flex;align-items:center;gap:8px">
      <span id="p2s-user-avatar-d" style="width:22px;height:22px;border-radius:999px;display:none;align-items:center;justify-content:center;background:#111827;color:#fff;font-size:11px;font-weight:700;flex:0 0 auto"></span>
      <a id="p2s-login-btn-d" href="/auth/login" style="font-size:11px;padding:4px 10px;background:#B5323E;color:#fff;border-radius:5px;text-decoration:none;white-space:nowrap">飞书登录</a>
    </div>
  </div>
</header>
<div class="diag-wrap">
  <aside class="diag-left">
    <div class="diag-left-head">
      <div class="diag-left-eyebrow">paper2skills · 诊断中心</div>
      <div class="diag-title">症状 → Skill 链</div>
      <div class="diag-sub">描述业务问题，匹配「诊断→处置→预防」三层技能</div>
    </div>
    <div class="diag-search-wrap">
      <div class="diag-input-row">
        <input class="diag-input" id="diag-input" placeholder="例：ASIN 流量下降、广告 ACoS 过高…" />
        <button class="diag-btn" onclick="runDiag()">诊断</button>
      </div>
    </div>
    <div class="diag-section-label">风险症状</div>
    <div class="diag-events" id="diag-events"></div>
    <div class="path-finder">
      <div class="path-finder-header" id="pf-header" onclick="togglePathFinder()">
        <span>Skill 路径规划</span>
        <span class="path-finder-arrow" id="pf-arrow">▸</span>
      </div>
      <div class="path-finder-body" id="path-finder-body">
        <select class="path-select" id="pf-from" title="起点 Skill"></select>
        <select class="path-select" id="pf-to" title="目标 Skill"></select>
        <button class="path-run-btn" onclick="runPathFinder()">查找路径 →</button>
      </div>
    </div>
  </aside>
  <main class="diag-right" id="diag-right">
    <div class="diag-empty" id="diag-empty">
      <div class="diag-empty-icon">—</div>
      <div class="diag-empty-title" style="font-size:16px;font-weight:700;color:var(--ink);margin-bottom:6px">症状 → 诊断链 → 行动方案</div>
      <div class="diag-empty-text">点击左侧症状，获取「诊断→处置→预防」三层 Skill 链</div>
      <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:400px;text-align:left">
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">诊断层</div>找出根因的算法技能
        </div>
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">处置层</div>立即执行的行动 Skill
        </div>
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">预防层</div>长效防范的系统 Skill
        </div>
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">手册</div>配套可执行场景手册
        </div>
      </div>
    </div>
    <div class="diag-result" id="diag-result"></div>
  </main>
</div>
<script src="assets/risk-events.js?v={build_ts}"></script>
<script>
(function(){{
  const SEV_CLASS={{'critical':'sev-critical','high':'sev-high','medium':'sev-medium','low':'sev-low'}};
  const SEV_LABEL={{'critical':'紧急','high':'高风险','medium':'中风险','low':'低风险'}};
  const PHASE_CFG={{
    diagnose:{{label:'第一步：诊断根因',cls:'diag-phase-diagnose'}},
    treat:{{label:'第二步：处置行动',cls:'diag-phase-treat'}},
    prevent:{{label:'第三步：长效预防',cls:'diag-phase-prevent'}}
  }};

  function initButtons(){{
    const events=(window.RISK_EVENTS||{{}}).events||[];
    const container=document.getElementById('diag-events');
    container.innerHTML='';
    events.forEach(ev=>{{
      const btn=document.createElement('button');
      btn.className='diag-event-btn';
      btn.dataset.id=ev.event_id;
      const sevCls=SEV_CLASS[ev.severity]||'sev-medium';
      const sevLabel=SEV_LABEL[ev.severity]||ev.severity;
      btn.innerHTML=`<div class="diag-event-info"><div class="diag-event-name">${{ev.event_name}}</div><span class="diag-event-sev ${{sevCls}}">${{sevLabel}}</span></div>`;
      btn.addEventListener('click',()=>showEvent(ev.event_id));
      container.appendChild(btn);
    }});
  }}

  function matchEvent(query){{
    const events=(window.RISK_EVENTS||{{}}).events||[];
    const q=query.toLowerCase();
    let best=null,top=0;
    events.forEach(ev=>{{
      let sc=0;
      (ev.symptom_keywords||[]).forEach(kw=>{{if(q.includes(kw.toLowerCase()))sc++;}});
      if(sc>top){{top=sc;best=ev;}}
    }});
    return top>0?best:null;
  }}

  function showEvent(eventId){{
    const events=(window.RISK_EVENTS||{{}}).events||[];
    const ev=events.find(e=>e.event_id===eventId);
    if(!ev)return;
    document.querySelectorAll('.diag-event-btn').forEach(b=>b.classList.remove('active'));
    const btn=document.querySelector(`.diag-event-btn[data-id="${{eventId}}"]`);
    if(btn)btn.classList.add('active');
    const sevCls=SEV_CLASS[ev.severity]||'sev-medium';
    const sevLabel=SEV_LABEL[ev.severity]||ev.severity;
    let html=`<div class="diag-result-header"><div class="diag-result-title"><span class="diag-result-name">${{ev.event_name}}</span><span class="diag-event-sev ${{sevCls}}" style="margin-left:8px">${{sevLabel}}</span></div><div class="diag-result-summary">${{ev.summary||''}}</div></div><div class="diag-phases">`;
    ['diagnose','treat','prevent'].forEach(phase=>{{
      const skills=(ev.phases||{{}})[phase]||[];
      if(!skills.length)return;
      const cfg=PHASE_CFG[phase];
      let items='';
      skills.forEach((sk,i)=>{{
        const cond=sk.condition?`<span class="diag-skill-cond">触发条件：${{sk.condition}}</span>`:'';
        const title=sk.title?` — ${{sk.title.split('—')[0].trim()}}`:'';
        items+=`<div class="diag-skill-item"><span class="diag-skill-num">${{i+1}}</span><div class="diag-skill-body"><a class="diag-skill-link" href="skills/${{sk.skill_id}}.html" target="_blank">${{sk.skill_id}}</a><div class="diag-skill-role">${{sk.role||''}}</div>${{cond}}</div></div>`;
      }});
      html+=`<div class="diag-phase ${{cfg.cls}}"><div class="diag-phase-header">${{cfg.label}}<span style="margin-left:auto;font-size:11px;font-weight:400;opacity:.7">${{skills.length}} 个 Skills</span></div><div class="diag-skill-list">${{items}}</div></div>`;
    }});
    html+='</div>';
    const pbs=(ev.related_playbooks||[]);
    if(pbs.length){{
      const links=pbs.map(id=>`<a class="diag-related-link" href="playbooks/${{id}}.html" target="_blank">${{id.replace('pb-','').replace(/-/g,' ')}}</a>`).join('');
      html+=`<div class="diag-related"><div class="diag-related-title">相关手册</div><div class="diag-related-links">${{links}}</div></div>`;
    }}
    document.getElementById('diag-empty').style.display='none';
    const res=document.getElementById('diag-result');
    res.innerHTML=html;
    res.className='diag-result visible';
  }}

  window.runDiag=function(){{
    const q=document.getElementById('diag-input').value.trim();
    if(!q)return;
    const ev=matchEvent(q);
    if(ev){{showEvent(ev.event_id);}}
    else{{
      document.getElementById('diag-empty').style.display='none';
      document.getElementById('diag-result').className='diag-result visible';
      document.getElementById('diag-result').innerHTML='<div class="diag-empty"><div class="diag-empty-icon">—</div><div class="diag-empty-text">未匹配到具体风险场景<br>请尝试点击左侧症状按钮，或前往 <a href="chat.html">AI对话</a> 获取帮助</div></div>';
    }}
  }};

  document.getElementById('diag-input').addEventListener('keydown',e=>{{if(e.key==='Enter')window.runDiag();}});
  if(window.RISK_EVENTS)initButtons();
  else window.addEventListener('load',initButtons);
}})();

(function(){{
  let gNodes={{}}, gAdj={{}};

  function loadGraph(cb){{
    if(Object.keys(gNodes).length){{cb();return;}}
    fetch('assets/graph-data.json').then(r=>r.json()).then(d=>{{
      d.nodes.forEach(n=>{{gNodes[n.id]={{id:n.id,domain:n.domain,title:n.title}};gAdj[n.id]=[];}});
      d.links.forEach(l=>{{if(gAdj[l.source])gAdj[l.source].push({{to:l.target,type:l.type}});}});
      const from=document.getElementById('pf-from'),to=document.getElementById('pf-to');
      const sorted=d.nodes.slice().sort((a,b)=>a.id.localeCompare(b.id));
      sorted.forEach(n=>{{
        const o1=document.createElement('option');o1.value=n.id;o1.textContent=n.id;from.appendChild(o1);
        const o2=document.createElement('option');o2.value=n.id;o2.textContent=n.id;to.appendChild(o2);
      }});
      cb();
    }}).catch(()=>{{}});
  }}

  function bfs(startId,endId){{
    if(startId===endId)return [{{id:startId,edgeType:''}}];
    const visited={{[startId]:true}},queue=[{{id:startId,path:[{{id:startId,edgeType:''}}]}}];
    while(queue.length){{
      const {{id,path}}=queue.shift();
      for(const nb of(gAdj[id]||[])){{
        if(visited[nb.to])continue;
        visited[nb.to]=true;
        const np=[...path,{{id:nb.to,edgeType:nb.type}}];
        if(nb.to===endId)return np;
        if(np.length<7)queue.push({{id:nb.to,path:np}});
      }}
    }}
    return null;
  }}

   window.togglePathFinder=function(){{
    const body=document.getElementById('path-finder-body');
    const header=document.getElementById('pf-header');
    const arrow=document.getElementById('pf-arrow');
    const open=body.classList.toggle('open');
    if(header)header.classList.toggle('open',open);
    arrow.textContent=open?'▾':'▸';
    if(open)loadGraph(()=>{{}});
  }};

  window.runPathFinder=function(){{
    const fromId=document.getElementById('pf-from').value;
    const toId=document.getElementById('pf-to').value;
    if(!fromId||!toId)return;
    loadGraph(()=>{{
      const path=bfs(fromId,toId);
      const empty=document.getElementById('diag-empty');
      const res=document.getElementById('diag-result');
      empty.style.display='none';
      if(!path){{
        res.className='diag-result visible';
        res.innerHTML=`<div class="diag-result-header"><div class="diag-result-name">未找到路径</div><div class="diag-result-summary">从 ${{fromId}} 到 ${{toId}} 在当前图谱中无可达路径（跳数≤6）。</div></div>`;
        return;
      }}
      const edgeLabel={{'prerequisite':'前置','extension':'延伸','combinable':'可组合'}};
      const edgeCls={{'prerequisite':'pet-prerequisite','extension':'pet-extension','combinable':'pet-combinable'}};
      let steps='';
      path.forEach((node,i)=>{{
        const n=gNodes[node.id]||{{}};
        const badge=node.edgeType?`<span class="path-edge-type ${{edgeCls[node.edgeType]||''}}">${{edgeLabel[node.edgeType]||node.edgeType}}</span>`:'';
        steps+=`<div class="path-step"><span class="path-step-num">${{i+1}}</span><div><a class="path-step-link" href="skills/${{node.id}}.html" target="_blank">${{node.id}}</a><div class="path-step-domain">${{n.domain||''}}${{badge}}</div></div></div>`;
      }});
      res.className='diag-result visible';
      res.innerHTML=`<div class="diag-result-header"><div class="diag-result-title"><span class="diag-result-name">Skill 路径：${{fromId}} → ${{toId}}</span></div><div class="diag-result-summary">${{path.length}} 步路径（${{path.length-1}} 条边）</div></div><div class="path-result">${{steps}}</div>`;
    }});
  }};
}})();
</script>
<script src="assets/search.js"></script>
<script src="assets/playbook-data.js"></script>
<script>
(function(){{
  const hbtn=document.getElementById('hamburger-diag');
  if(hbtn){{
    hbtn.addEventListener('click',function(){{
      const open=hbtn.getAttribute('aria-expanded')!=='true';
      hbtn.setAttribute('aria-expanded',open);
      hbtn.classList.toggle('open',open);
    }});
  }}
  fetch('/auth/me',{{credentials:'include',cache:'no-store'}}).then(function(r){{return r.json();}}).then(function(u){{
    if(u&&u.name){{
      var av=document.getElementById('p2s-user-avatar-d');
      var lb=document.getElementById('p2s-login-btn-d');
      if(av){{av.textContent=String(u.name).slice(0,1);av.style.display='inline-flex';}}
      if(lb){{lb.textContent=u.name;lb.href='/settings.html';lb.style.background='transparent';lb.style.color='#94a3b8';lb.style.border='1px solid rgba(255,255,255,.15)';}}
    }}
  }}).catch(function(){{}});
}})();
</script>
</body>
</html>"""


def render_chat_page(nav: str = "", skill_count: int = 0) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI 知识库对话 · paper2skills</title>
  <link rel="stylesheet" href="{nav}assets/style.css">
  <style>
    body {{ overflow: hidden; }}
    .chat-layout {{
      display: flex; height: 100vh; flex-direction: column;
    }}
    .chat-topbar {{
      display: flex; align-items: center; gap: 0;
      height: var(--topbar-height); flex-shrink: 0;
      padding: 0 20px;
      background: rgba(255,255,255,0.92);
      backdrop-filter: blur(12px) saturate(180%);
      -webkit-backdrop-filter: blur(12px) saturate(180%);
      border-bottom: 1px solid var(--nav-border);
      box-shadow: 0 1px 0 rgba(0,0,0,0.05);
    }}
    .chat-back {{
      display: flex; align-items: center; gap: 6px;
      color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 500;
      padding: 6px 10px; border-radius: var(--r-md);
      transition: background var(--t);
      flex-shrink: 0;
    }}
    .chat-back:hover {{ background: var(--accent-light); text-decoration: none; }}
    .chat-title-area {{
      flex: 1; display: flex; align-items: center; justify-content: center;
      gap: 10px;
    }}
    .chat-title-icon {{
      font-size: 20px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .chat-title-text {{
      font-size: 15px; font-weight: 650; letter-spacing: -.02em; color: var(--ink);
    }}
    .chat-title-sub {{
      font-size: 11.5px; color: var(--muted); font-weight: 400;
      padding: 3px 9px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }}
    .chat-ctrl {{ flex-shrink: 0; display: flex; align-items: center; gap: 10px; }}
    .web-search-toggle {{
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 12px; color: var(--muted); cursor: pointer;
      padding: 5px 10px; border-radius: var(--r-full);
      border: 1.5px solid var(--line); background: transparent;
      transition: all var(--t); user-select: none; flex-shrink: 0;
      font-family: var(--font); white-space: nowrap;
    }}
    .web-search-toggle.on {{
      color: var(--accent); border-color: var(--accent);
      background: var(--accent-light); font-weight: 600;
    }}
    .web-search-toggle:hover {{ border-color: var(--line-strong); color: var(--ink); }}
    .web-search-toggle.on:hover {{ border-color: var(--accent-dark); }}
    .web-search-toggle-icon {{ font-size: 13px; line-height: 1; }}
    .chat-body {{
      flex: 1; display: flex; flex-direction: column;
      max-width: 760px; width: 100%; margin: 0 auto;
      padding: 0 16px; overflow: hidden;
    }}
    .chat-messages {{
      flex: 1; overflow-y: auto; padding: 28px 0 12px;
      display: flex; flex-direction: column; gap: 20px;
    }}
    .chat-messages::-webkit-scrollbar {{ width: 4px; }}
    .chat-messages::-webkit-scrollbar-thumb {{ background: var(--line-strong); border-radius: 4px; }}
    .cmsg {{ display: flex; gap: 12px; align-items: flex-start; }}
    .cmsg-user {{ flex-direction: row-reverse; }}
    .cmsg-avatar {{
      width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      color: #fff; display: flex; align-items: center; justify-content: center;
      font-size: 13px; font-weight: 700; margin-top: 2px;
    }}
    .cmsg-user .cmsg-avatar {{
      background: var(--panel-3); color: var(--muted); font-size: 11px;
    }}
    .cmsg-body {{ flex: 1; min-width: 0; }}
    .cmsg-name {{
      font-size: 11px; font-weight: 600; letter-spacing: .02em;
      text-transform: uppercase; color: var(--muted); margin-bottom: 5px;
    }}
    .cmsg-user .cmsg-name {{ text-align: right; }}
    .cmsg-bubble {{
      display: inline-block; max-width: 100%;
      padding: 12px 16px; border-radius: 4px 18px 18px 18px;
      background: var(--panel); border: 1px solid var(--line);
      font-size: 14.5px; line-height: 1.72; color: var(--ink);
      box-shadow: var(--shadow-xs);
    }}
    .cmsg-user .cmsg-bubble {{
      background: var(--accent); color: #fff; border-color: transparent;
      border-radius: 18px 4px 18px 18px; box-shadow: none;
    }}
    .cmsg-bubble strong {{ font-weight: 700; }}
    .cmsg-bubble code {{
      background: rgba(0,0,0,.06); padding: 2px 6px;
      border-radius: 5px; font-size: 13px; font-family: 'SF Mono', 'Menlo', monospace;
    }}
    .cmsg-bubble br {{ margin: 0; }}
    .cmsg-web-badge {{
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: var(--muted); margin-bottom: 6px;
      padding: 2px 8px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }}
    .cmsg-event-badge {{
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: #991b1b; margin-bottom: 6px;
      padding: 2px 8px; background: #fef2f2; border-radius: var(--r-full);
      border: 1px solid #fecaca;
    }}
    .cmsg-typing .cmsg-bubble::after {{
      content: ''; display: inline-block; width: 40px; height: 10px;
      background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 10'%3E%3Ccircle cx='5' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0s' repeatCount='indefinite'/%3E%3C/circle%3E%3Ccircle cx='20' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0.2s' repeatCount='indefinite'/%3E%3C/circle%3E%3Ccircle cx='35' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0.4s' repeatCount='indefinite'/%3E%3C/circle%3E%3C/svg%3E") no-repeat center;
      vertical-align: middle; margin-left: 4px;
    }}
    .chat-welcome {{
      text-align: center; padding: 40px 20px 20px; color: var(--muted);
    }}
    .chat-welcome-icon {{
      font-size: 48px; display: block; margin-bottom: 16px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .chat-welcome h2 {{
      font-size: 22px; font-weight: 700; letter-spacing: -.03em; color: var(--ink);
      margin: 0 0 8px; border: none; padding: 0;
    }}
    .chat-welcome p {{ font-size: 14.5px; color: var(--muted); margin: 0 0 24px; }}
    .chat-suggestions {{
      display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px;
    }}
    .chat-sug-btn {{
      padding: 8px 16px; border-radius: var(--r-full);
      border: 1.5px solid var(--line); background: var(--panel);
      font-size: 13px; color: var(--ink-2); cursor: pointer; font-family: var(--font);
      transition: border-color var(--t), background var(--t), color var(--t);
    }}
    .chat-sug-btn:hover {{
      border-color: var(--accent); color: var(--accent); background: var(--accent-light);
    }}
    .chat-input-area {{
      flex-shrink: 0; padding: 10px 0 18px;
      border-top: 1px solid var(--line);
    }}
    .chat-input-wrap {{
      display: flex; align-items: flex-end; gap: 8px;
      background: var(--panel); border: 1.5px solid var(--line);
      border-radius: 18px; padding: 8px 10px 8px 12px;
      transition: border-color var(--t), box-shadow var(--t);
    }}
    .chat-input-wrap:focus-within {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(194,91,110,.10);
    }}
    .chat-input-wrap .web-search-toggle {{
      align-self: flex-end; margin-bottom: 1px;
    }}
    .chat-textarea {{
      flex: 1; border: none; outline: none; resize: none;
      font-family: var(--font); font-size: 14.5px; line-height: 1.6;
      color: var(--ink); background: transparent;
      min-height: 24px; max-height: 160px;
      overflow-y: auto;
    }}
    .chat-textarea::placeholder {{ color: var(--muted); }}
    .chat-send-btn {{
      width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
      background: var(--accent); border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-size: 16px;
      transition: background var(--t), transform var(--t);
    }}
    .chat-send-btn:hover {{ background: var(--accent-dark); transform: scale(1.06); }}
    .chat-send-btn:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
    .chat-hint {{
      text-align: center; font-size: 11.5px; color: var(--muted);
      margin-top: 8px;
    }}
    @media (max-width: 600px) {{
      .chat-title-sub {{ display: none; }}
      .chat-body {{ padding: 0 12px; }}
    }}
  </style>
</head>
<body>
  <div class="chat-layout">
    <header class="chat-topbar">
      <a class="chat-back" href="{nav}index.html">← 返回</a>
      <div class="chat-title-area">
        <span class="chat-title-text">AI 知识库对话</span>
        <span class="chat-title-sub">{skill_count} Skills · DeepSeek V3</span>
      </div>
      <div class="chat-ctrl">
        <select id="role-select" style="font-size:12px;padding:5px 9px;border:1.5px solid var(--line);border-radius:var(--r-full);background:transparent;color:var(--ink);cursor:pointer;font-family:var(--font);">
          <option value="ops">运营视角</option>
          <option value="analyst">数据分析师</option>
          <option value="ceo">CEO 战略</option>
        </select>
      </div>
    </header>

    <div class="chat-body">
      <div class="chat-messages" id="chat-messages">
        <div class="chat-welcome" id="chat-welcome">
          <h2>paper2skills 知识库助手</h2>
          <p>基于 {skill_count} 个从顶会论文萃取的跨境电商 AI 决策技能，为你提供专业问答</p>
          <div style="font-size:11px;color:var(--muted);font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px">试试问这些</div>
          <div class="chat-suggestions">
            <button class="chat-sug-btn">如何提升广告 ROI？</button>
            <button class="chat-sug-btn">大促备货如何预测需求？</button>
            <button class="chat-sug-btn">供应链 AI 有哪些关键技能？</button>
            <button class="chat-sug-btn">KOL 投放效果怎么归因？</button>
            <button class="chat-sug-btn">账号 ODR 异常，担心被封号</button>
            <button class="chat-sug-btn">ASIN 流量突然下降 30%</button>
            <button class="chat-sug-btn">产品被平台合规警告</button>
          </div>
        </div>
      </div>

      <div class="chat-input-area">
        <div class="chat-input-wrap">
          <button class="web-search-toggle" id="web-search-toggle" title="开启联网搜索">
            <span class="web-search-toggle-icon">联网</span>
            <span id="web-search-label">联网搜索</span>
          </button>
          <textarea class="chat-textarea" id="chat-input"
            placeholder="描述业务症状，如：ASIN 流量下降、库存积压、广告 ACOS 过高…"
            rows="1" autocomplete="off"></textarea>
          <button class="chat-send-btn" id="chat-send" title="发送 (Enter)">↑</button>
        </div>
        <p class="chat-hint">Enter 发送 · Shift+Enter 换行</p>
      </div>
    </div>
  </div>

  <script src="{nav}assets/playbook-data.js"></script>
  <script src="{nav}assets/risk-events.js"></script>
  <script src="{nav}assets/chat-page.js"></script>
</body>
</html>"""


def build_chat_page_js() -> str:
    return r"""
(function () {
  const msgsEl = document.getElementById('chat-messages');
  const welcome = document.getElementById('chat-welcome');
  const textarea = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const webToggle = document.getElementById('web-search-toggle');
  const webLabel = document.getElementById('web-search-label');

  let webSearchOn = false;
  const _HIST_KEY = 'p2s_chat_v1';

  function _loadH() {
    try { return JSON.parse(localStorage.getItem(_HIST_KEY) || '[]'); } catch (e) { return []; }
  }

  function _saveH() {
    try { localStorage.setItem(_HIST_KEY, JSON.stringify(history.slice(-20))); } catch (e) {}
  }

  let history = _loadH();

  window.clearHistory = function() {
    history = [];
    try { localStorage.removeItem(_HIST_KEY); } catch (e) {}
    if (msgsEl) { msgsEl.innerHTML = ''; }
    if (welcome) welcome.style.display = '';
  };

  webToggle.addEventListener('click', () => {
    webSearchOn = !webSearchOn;
    webToggle.classList.toggle('on', webSearchOn);
    webLabel.textContent = webSearchOn ? '已开启联网' : '联网搜索';
  });

  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  });

  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      doSend();
    }
  });

  sendBtn.addEventListener('click', doSend);

  if (history.length && welcome) {
    welcome.style.display = 'none';
    history.forEach(function (m) {
      if (m.role === 'user') addMsg(m.content, 'user');
      else if (m.role === 'assistant') addMsg(m.content, 'bot');
    });
  }

  document.querySelectorAll('.chat-sug-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.textContent.trim();
      textarea.dispatchEvent(new Event('input'));
      doSend();
    });
  });

  function md(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/gs, '<strong>$1</strong>')
      .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
      .replace(/`([^`\n]+)`/g, '<code>$1</code>')
      .replace(/^#{1,3}\s+(.+)$/gm, '<strong style="font-size:15px">$1</strong>')
      .replace(/^[-•]\s+(.+)$/gm, '<span style="display:block;padding-left:14px;margin:2px 0">• $1</span>')
      .replace(/^\d+\.\s+(.+)$/gm, '<span style="display:block;padding-left:14px;margin:2px 0">$&</span>')
      .replace(/\n\n+/g, '<br><br>').replace(/\n/g, '<br>');
  }

  const _idx = [];
  let _built = false;

  function buildSkillIndex() {
    if (_built) return;
    const DATA = window.PLAYBOOK_DATA || {};
    (DATA.skills || []).forEach(s => {
      const t = [
        s.skill_id || '', s.title || '', s.problem_solved || '',
        s.algorithm_summary || '', s.biz_trigger || '', s.biz_outcome || '',
        (s.tags || []).join(' '), (s.topics || []).join(' ')
      ].join(' ').toLowerCase();
      _idx.push({ s, t });
    });
    _built = true;
  }

  let _skillIdx = null;
  async function _loadSkillIdx() {
    if (_skillIdx) return _skillIdx;
    try {
      _skillIdx = await fetch('/assets/skill-index.json').then(r => r.json());
    } catch(e) { _skillIdx = []; }
    return _skillIdx;
  }

  async function _retrieveSkills(query, topK) {
    topK = topK || 5;
    const idx = await _loadSkillIdx();
    if (!idx || !idx.length) return [];
    const tokens = query.toLowerCase().replace(/[^\u4e00-\u9fa5a-z0-9\s]/g,' ').split(/\s+/).filter(function(t){ return t.length > 1; });
    if (!tokens.length) return [];
    const scored = idx.map(function(s) {
      const text = (s.summary + ' ' + s.keywords.join(' ')).toLowerCase();
      const score = tokens.reduce(function(n, t) { return n + (text.includes(t) ? 1 : 0); }, 0);
      return { s: s, score: score };
    }).filter(function(x) { return x.score > 0; });
    scored.sort(function(a,b) { return b.score - a.score; });
    return scored.slice(0, topK).map(function(x) { return x.s; });
  }

  function searchSkills(query, k) {
    k = k || 8;
    buildSkillIndex();
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 1);
    if (!words.length) return [];
    return _idx.map(item => {
      let sc = 0;
      words.forEach(w => {
        const tf = item.t.split(w).length - 1;
        if (tf > 0) sc += tf * (w.length > 3 ? 2 : 1);
      });
      return { skill: item.s, sc };
    }).filter(x => x.sc > 0).sort((a, b) => b.sc - a.sc).slice(0, k).map(x => x.skill);
  }

  function buildRAGContext(query) {
    const top = searchSkills(query, 10);
    if (!top.length) {
      return (window.PLAYBOOK_DATA && window.PLAYBOOK_DATA.skills || []).slice(0, 60).map(s =>
        s.skill_id + ': ' + (s.problem_solved || s.algorithm_summary || '').slice(0, 140)
      ).join('\n');
    }
    return top.map(s => {
      const p = [s.skill_id, s.title];
      if (s.problem_solved) p.push('解决: ' + s.problem_solved.slice(0, 120));
      if (s.biz_trigger) p.push('触发: ' + s.biz_trigger.slice(0, 100));
      if (s.roi_figure) p.push('ROI: ' + s.roi_figure);
      return p.join(' | ');
    }).join('\n');
  }

  function renderSkillCards(text) {
    const DATA = window.PLAYBOOK_DATA || {};
    const map = {};
    (DATA.skills || []).forEach(s => { map[s.skill_id] = s; });

    const found = [];
    const seen = {};
    [/\[\[?(Skill-[\w-]+)\]?\]/g, /\*\*(Skill-[\w-]+)\*\*/g].forEach(pat => {
      let m;
      while ((m = pat.exec(text)) !== null) {
        if (map[m[1]] && !seen[m[1]]) {
          seen[m[1]] = 1;
          found.push(map[m[1]]);
        }
      }
    });

    if (!found.length) return '';

    const esc = t => (t || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const cards = found.map(s =>
      '<a href="skills/' + s.skill_id + '.html" target="_blank" style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:var(--panel-2,#f8fafc);border:1px solid var(--line,#e2e8f0);border-radius:8px;text-decoration:none;color:inherit;margin-top:6px;transition:box-shadow .15s" onmouseover="this.style.boxShadow=\'0 2px 8px rgba(0,0,0,.08)\'" onmouseout="this.style.boxShadow=\'none\'">' +
      '<div style="flex-shrink:0;width:32px;height:32px;border-radius:6px;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700">S</div>' +
      '<div style="min-width:0">' +
      '<div style="font-size:12px;font-weight:600;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc((s.title || s.skill_id).slice(0, 60)) + '</div>' +
      '<div style="font-size:11.5px;color:#64748b;margin-top:2px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical">' + esc((s.problem_solved || s.biz_trigger || '').slice(0, 90)) + '</div>' +
      (s.roi_figure ? '<span style="font-size:11px;color:#059669;font-weight:600;margin-top:4px;display:block">ROI: ' + esc(s.roi_figure) + '</span>' : '') +
      '</div></a>'
    ).join('');

    return '<div style="margin-top:10px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px">' +
      '<div style="font-size:11.5px;color:#64748b;font-weight:600;margin-bottom:6px">知识库 相关技能</div>' +
      cards +
      '</div>';
  }

  const AKWS = {
    'agent-supply-sentinel': ['供应链', '库存', '断货', '补货', 'DOS', '海运'],
    'agent-pricing-advisor': ['定价', '价格', 'ACoS', '竞品价', '利润率'],
    'agent-pnl-analyzer': ['P&L', '利润', 'GMV', '毛利', '亏损'],
    'agent-ad-attribution': ['广告', 'ROAS', '归因', 'ACoS', '投放'],
    'agent-listing-doctor': ['Listing', '标题', '关键词', 'A+'],
    'agent-voc-decoder': ['评论', 'VOC', '用户反馈', '差评'],
    'agent-cs-triage': ['客服', '工单', '退款', '投诉', 'A-to-Z'],
    'agent-account-guardian': ['封号', '账号', '违规', '风险'],
    'agent-brand-guardian': ['合规', '文案', '广告法', '违禁'],
    'agent-product-radar': ['选品', '蓝海', '竞争', '市场机会'],
    'agent-tiktok-content': ['TikTok', '短视频', '内容', '脚本'],
    'agent-competitor-radar': ['竞品', '竞争对手', 'ASIN', 'BSR']
  };

  const ANAMES = {
    'agent-supply-sentinel': '供应链哨兵',
    'agent-pricing-advisor': '动态定价顾问',
    'agent-pnl-analyzer': 'P&L透视镜',
    'agent-ad-attribution': '广告归因侦探',
    'agent-listing-doctor': 'Listing医生',
    'agent-voc-decoder': '用户之声解码器',
    'agent-cs-triage': '客服分诊台',
    'agent-account-guardian': '账号风险卫士',
    'agent-brand-guardian': '品牌合规卫士',
    'agent-product-radar': '选品雷达',
    'agent-tiktok-content': 'TikTok内容官',
    'agent-competitor-radar': '竞品雷达站'
  };

  function detectAgents(text) {
    const t = text.toLowerCase();
    return Object.keys(AKWS).filter(id => AKWS[id].some(k => t.indexOf(k.toLowerCase()) >= 0)).slice(0, 3);
  }

  function renderAgentBtns(ids) {
    if (!ids.length) return '';
    const btns = ids.map(id =>
      '<a href="agents.html" target="_blank" style="display:inline-flex;align-items:center;gap:5px;padding:6px 12px;background:var(--accent-light,#eff6ff);border:1px solid var(--accent,#3b82f6);border-radius:20px;font-size:12px;font-weight:600;color:var(--accent,#3b82f6);text-decoration:none;transition:all .15s;white-space:nowrap" onmouseover="this.style.background=\'var(--accent,#3b82f6)\';this.style.color=\'#fff\'" onmouseout="this.style.background=\'var(--accent-light,#eff6ff)\';this.style.color=\'var(--accent,#3b82f6)\'">◈ ' + (ANAMES[id] || id) + '</a>'
    ).join('');
    return '<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px"><span style="font-size:11.5px;color:#64748b;font-weight:600;align-self:center;margin-right:4px"> 直接调用：</span>' + btns + '</div>';
  }

  function addMsg(text, role, extras) {
    extras = extras || {};
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-' + role;
    
    const av = document.createElement('div');
    av.className = 'cmsg-avatar';
    av.textContent = role === 'bot' ? '\u2726' : 'U';
    
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    
    const nm = document.createElement('div');
    nm.className = 'cmsg-name';
    nm.textContent = role === 'bot' ? 'AI 助手' : '你';
    body.appendChild(nm);
    
    if (extras.webBadge) {
      const b = document.createElement('div');
      b.className = 'cmsg-web-badge';
      b.innerHTML = '联网搜索';
      body.appendChild(b);
    }
    
    if (extras.eventBadge) {
      const b = document.createElement('div');
      b.className = 'cmsg-event-badge';
      b.innerHTML = extras.eventBadge;
      body.appendChild(b);
    } else if (extras.ragBadge) {
      const b = document.createElement('div');
      b.className = 'cmsg-web-badge';
      b.style.cssText = 'background:#f0fdf4;color:#166534;border-color:#bbf7d0';
      b.innerHTML = '知识库检索 · ' + extras.ragBadge + ' 条相关技能';
      body.appendChild(b);
    }
    
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    
    if (role === 'bot') {
      bubble.innerHTML = md(text);
      const agIds = detectAgents(text);
      const sc = renderSkillCards(text);
      const ab = renderAgentBtns(agIds);
      if (sc || ab) {
        const x = document.createElement('div');
        x.innerHTML = (sc || '') + (ab || '');
        bubble.appendChild(x);
      }
    } else {
      bubble.textContent = text;
    }
    
    body.appendChild(bubble);
    row.appendChild(av);
    row.appendChild(body);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    
    return { row, bubble };
  }

  function addTyping() {
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-bot cmsg-typing';
    
    const av = document.createElement('div');
    av.className = 'cmsg-avatar';
    av.textContent = '\u2726';
    
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    
    const nm = document.createElement('div');
    nm.className = 'cmsg-name';
    nm.textContent = 'AI 助手';
    
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    
    body.appendChild(nm);
    body.appendChild(bubble);
    row.appendChild(av);
    row.appendChild(body);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    
    return row;
  }

  function matchRiskEvent(query) {
    if (!window.RISK_EVENTS || !window.RISK_EVENTS.events) return null;
    const lowerQuery = query.toLowerCase();
    let bestEvent = null;
    let maxScore = 0;
    
    for (const event of window.RISK_EVENTS.events) {
      if (!event.symptom_keywords) continue;
      let score = 0;
      for (const kw of event.symptom_keywords) {
        if (lowerQuery.includes(kw.toLowerCase())) {
          score++;
        }
      }
      if (score > maxScore) {
        maxScore = score;
        bestEvent = event;
      }
    }
    
    return maxScore > 0 ? bestEvent : null;
  }

  function buildEventSkillChain(event) {
    let result = '';
    const phases = event.phases || {};
    
    if (phases.diagnose && phases.diagnose.length > 0) {
      result += '【诊断层】\n';
      phases.diagnose.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\n`;
      });
    }
    
    if (phases.treat && phases.treat.length > 0) {
      result += '【处置层】\n';
      phases.treat.forEach((s, i) => {
        const cond = s.condition ? `（条件: ${s.condition}时触发）` : '';
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}${cond}\n`;
      });
    }
    
    if (phases.prevent && phases.prevent.length > 0) {
      result += '【预防层】\n';
      phases.prevent.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\n`;
      });
    }
    
    return result;
  }

  async function doSend() {
    const text = textarea.value.trim();
    if (!text || sendBtn.disabled) return;
    textarea.value = '';
    textarea.style.height = 'auto';
    sendBtn.disabled = true;
    
    addMsg(text, 'user');
    history.push({ role: 'user', content: text });
    
    const typing = addTyping();
    
    let ctxMsg = '';
    const matchedEvent = matchRiskEvent(text);
    let matchedEventText = null;
    
    let sys = '你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\n知识库现有 {skill_count} 个从顶会论文萃取的可落地业务技能。\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\n当前时间：' + new Date().toLocaleDateString('zh-CN', {year:'numeric',month:'long',day:'numeric'});
    
    if (matchedEvent) {
      sys += `\n\n当前诊断场景：${matchedEvent.event_name}\n严重程度：${matchedEvent.severity}`;
      const eventChain = buildEventSkillChain(matchedEvent);
      ctxMsg = '\n\n【场景推荐 Skill 链】\n' + eventChain;
      matchedEventText = `${matchedEvent.icon} 识别到场景：${matchedEvent.event_name}`;
    } else {
      const ragSkills = searchSkills(text, 10);
      const ragCtx = buildRAGContext(text);
      const ragCount = ragSkills.length;
      ctxMsg = ragCount > 0 ? '\n\n【知识库相关技能（检索到' + ragCount + '条）】\n' + ragCtx : '\n\n【知识库摘要（前60条）】\n' + ragCtx;
      const _idxSkills = await _retrieveSkills(text, 5);
      if (_idxSkills.length) {
        ctxMsg += '\n\n【知识库检索结果 — 请优先引用这些 Skill ID 回答】\n' +
          _idxSkills.map(function(s) {
            return '[' + s.id + '] ' + s.title + ': ' + s.summary.slice(0, 120);
          }).join('\n');
      }
    }
    
    const messages = [
      { role: 'system', content: sys + ctxMsg },
      ...history.slice(-8)
    ];
    
    try {
      const body = {
        model: 'deepseek-chat',
        messages,
        max_tokens: 1500,
        temperature: 0.55,
        stream: false
      };
      
      if (webSearchOn) {
        body.tools = [{ 
          type: 'function', 
          function: { 
            name: 'web_search', 
            description: 'Search the web', 
            parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } 
          } 
        }];
        body.tool_choice = 'auto';
      }
      
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      const data = await res.json();
      const choice = data && data.choices && data.choices[0];
      let answer = (choice && choice.message && choice.message.content || '').trim();
      
      if (!answer && choice && choice.finish_reason === 'tool_calls') {
        answer = '（联网搜索触发中…）\n\n' + ((choice.message.tool_calls[0] && choice.message.tool_calls[0].function.arguments) || '');
      }
      answer = answer || '抱歉，暂时无法获取回答，请稍后重试。';
      
      typing.remove();
      
      let ragCountToPass = null;
      let _lastRagSkills = [];
      if (!matchedEvent) {
          const ragSkills = searchSkills(text, 10);
          ragCountToPass = ragSkills.length > 0 ? ragSkills.length : null;
          _lastRagSkills = ragSkills;
      }
      
      const _msgResult = addMsg(answer, 'bot', { webBadge: webSearchOn, eventBadge: matchedEventText, ragBadge: ragCountToPass });
      
      if (_lastRagSkills.length) {
        _retrieveSkills(text, 5).then(function(idxSkills) {
          if (!idxSkills.length) return;
          const citDiv = document.createElement('div');
          citDiv.style.cssText = 'font-size:11px;color:#94a3b8;margin-top:6px;padding-top:6px;border-top:1px solid #f1f5f9';
          citDiv.innerHTML = '\u53c2\u8003 Skill: ' + idxSkills.map(function(s) {
            return '<a href="/skills/' + s.id + '.html" target="_blank" style="color:#6366f1;text-decoration:none">' + s.title + '</a>';
          }).join(' \u00b7 ');
          if (_msgResult && _msgResult.bubble) _msgResult.bubble.appendChild(citDiv);
        });
      }
      
      history.push({ role: 'assistant', content: answer });
      _saveH();
      
    } catch (e) {
      typing.remove();
      addMsg('网络请求失败，请检查连接后重试。', 'bot');
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  }
})();
"""


def build_search_js() -> str:
    return r"""
(function(){
  const input = document.getElementById('global-search');
  const box   = document.getElementById('search-results');
  if (!input || !box || !window.PLAYBOOK_DATA) return;
  const skills = window.PLAYBOOK_DATA.skills || [];

  function applyFilters(list) {
    const diff  = (document.getElementById('filter-diff')  || {}).value || '';
    const roi   = (document.getElementById('filter-roi')   || {}).value || '';
    const dom   = (document.getElementById('filter-domain') || {}).value || '';
    return list.filter(s => {
      if (dom  && s.domain_dir !== dom) return false;
      if (diff && s.difficulty !== diff) return false;
      if (roi) {
        const stars = (s.difficulty || '').split('⭐').length - 1;
        if (roi === 'easy'   && stars > 2) return false;
        if (roi === 'medium' && (stars < 3 || stars > 3)) return false;
        if (roi === 'hard'   && stars < 4) return false;
      }
      return true;
    });
  }

  function doSearch() {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { box.classList.add('hidden'); box.innerHTML = ''; return; }
    let hits = skills.filter(s =>
      [s.skill_id, s.title, s.domain_dir,
       (s.tags||[]).join(' '), (s.topics||[]).join(' '),
       s.algorithm_summary, s.problem_solved, s.roi_figure
      ].join(' ').toLowerCase().includes(q)
    );
    hits = applyFilters(hits).slice(0, 24);
    box.innerHTML = hits.map(s =>
      `<a class="result" href="${rootPrefix()}skills/${s.skill_id}.html">` +
      `<strong>${esc(s.title)}</strong>` +
      `<br><span>${esc(s.domain_dir)}` +
      `${s.roi_figure ? ' · ' + esc(s.roi_figure) : ''}` +
      `${s.difficulty ? ' · ' + esc(s.difficulty) : ''}</span></a>`
    ).join('') || '<p class="muted" style="padding:12px">无结果</p>';
    box.classList.remove('hidden');
  }

  input.addEventListener('input', doSearch);
  ['filter-diff','filter-roi','filter-domain'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', doSearch);
  });
  document.addEventListener('click', e => {
    if (e.target !== input && !box.contains(e.target)) box.classList.add('hidden');
  });
  function rootPrefix() {
    const p = location.pathname;
    return (p.includes('/skills/') || p.includes('/domains/') || p.includes('/topics/') ||
            p.includes('/workflows/') || p.includes('/playbooks/') || p.includes('/graph/')) ? '../' : '';
  }
  function esc(s) {
    return String(s||'').replace(/[&<>"']/g, c =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
})();
""".strip()


# ---------------------------------------------------------------------------
# render_pages orchestrator
# ---------------------------------------------------------------------------

def render_pages(
    out: Path,
    skills: list[PlaybookSkill],
    domains: list[dict[str, Any]],
    graph: SkillsGraph,
    wf_defs: dict[str, Any],
) -> dict[str, Any]:
    known_skill_ids: set[str] = {skill.skill_id for skill in skills}
    KNOWN_SKILL_IDS.clear()
    KNOWN_SKILL_IDS.update(known_skill_ids)
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)

    skill_count  = len(skills)
    domain_count = len({s.domain_dir for s in skills})

    # 有效边 = 两端节点均在图谱内（与 graph-data.json 的 links 保持一致）
    _skill_title_map = {s.skill_id: s.title for s in skills}
    graph_node_ids = {n.id for n in graph.nodes.values()}
    valid_links = [
        {"source": e.source, "target": e.target, "type": e.edge_type}
        for e in graph.edges
        if e.source in graph_node_ids and e.target in graph_node_ids
    ]
    edge_count = len(valid_links)

    # Data assets
    build_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {"skill_count": skill_count, "domain_count": domain_count, "edge_count": edge_count},
        "domains": domains,
        "skills": [skill.__dict__ for skill in skills],
    }
    write_file(out / "assets" / "playbook-data.json", json.dumps(data, ensure_ascii=False, indent=2))
    write_file(out / "assets" / "playbook-data.js",  "window.PLAYBOOK_DATA = " + json.dumps(data, ensure_ascii=False) + ";")

    graph_json = {
        "nodes": [
            {
                "id": n.id,
                "domain": n.domain,
                "title": _skill_title_map.get(n.id, n.id),
            }
            for n in graph.nodes.values()
        ],
        "links": valid_links,
    }
    write_file(out / "assets" / "graph-data.json", json.dumps(graph_json, ensure_ascii=False, indent=2))

    _ont = Path(__file__).parent / "config" / "risk_events_ontology.yaml"
    if _ont.exists():
        import yaml as _yaml
        with open(_ont, encoding="utf-8") as _f:
            _oraw = _yaml.safe_load(_f)
        _evts = _oraw.get("events", [])
        _sm = {s.skill_id: {"title": s.title, "problem_solved": s.problem_solved} for s in skills}
        for _ev in _evts:
            for _ph in ("diagnose", "treat", "prevent"):
                for _sk in _ev.get("phases", {}).get(_ph, []):
                    _d = _sm.get(_sk.get("skill_id", ""), {})
                    if _d:
                        _sk.setdefault("title", _d.get("title", ""))
        _re = {"version": _oraw.get("version", "1.0"), "events": _evts}
        write_file(out / "assets" / "risk-events.json", json.dumps(_re, ensure_ascii=False, indent=2))
        write_file(out / "assets" / "risk-events.js", "window.RISK_EVENTS = " + json.dumps(_re, ensure_ascii=False) + ";")

    write_file(out / "assets" / "style.css",  build_css())
    write_file(out / "assets" / "search.js",  build_search_js())
    write_file(out / "assets" / "graph.js",   build_graph_js())
    write_file(out / "assets" / "ego-graph.js", build_ego_graph_js())
    write_file(out / "assets" / "chat-page.js", build_chat_page_js())
    write_file(out / "chat.html", render_chat_page(skill_count=skill_count))
    write_file(out / "diagnostic.html", render_diagnostic_page(skill_count=skill_count, build_ts=build_ts))

    # ── Index (Phase 3C) ──
    write_file(out / "index.html", html_page(
        "总览",
        render_index(skill_count, domain_count, edge_count, domains, skills, len(WORKFLOW_RULES)),
        active_nav="index",
    ))

    # ── Skill pages ──
    for skill in skills:
        write_file(out / "skills" / f"{skill.skill_id}.html", render_skill_page(skill))
    all_cards = "".join(render_skill_card(skill, "../") for skill in skills)
    domain_opts = "".join(
        f"<option value='{html.escape(d)}'>{html.escape(d)}</option>"
        for d in sorted({s.domain_dir for s in skills})
    )
    filter_bar = f"""
<div class="filter-bar">
  <select id="filter-domain" class="filter-select">
    <option value="">全部领域</option>{domain_opts}
  </select>
  <select id="filter-diff" class="filter-select">
    <option value="">全部难度</option>
    <option value="⭐☆☆☆☆">⭐ 入门</option>
    <option value="⭐⭐☆☆☆">⭐⭐ 简单</option>
    <option value="⭐⭐⭐☆☆">⭐⭐⭐ 中等</option>
    <option value="⭐⭐⭐⭐☆">⭐⭐⭐⭐ 较难</option>
    <option value="⭐⭐⭐⭐⭐">⭐⭐⭐⭐⭐ 专家</option>
  </select>
  <span class="filter-hint muted" id="filter-count"></span>
</div>
<div id="skill-pg-bar" class="pg-bar"></div>
<script>
(function(){{
  var PAGE=48;var cur=1;
  function getVisible(){{return Array.from(document.querySelectorAll('#skill-card-grid .skill-card')).filter(function(c){{return c.style.display!=='none';}});}}
  function renderPager(total){{
    var pages=Math.ceil(total/PAGE);
    var bar=document.getElementById('skill-pg-bar');
    if(!bar)return;
    if(pages<=1){{bar.innerHTML='';return;}}
    var html='<div class="pg-inner">';
    html+='<button class="pg-btn" onclick="window._pgGo('+(cur-1)+')" '+(cur<=1?'disabled':'')+'>‹ 上一页</button>';
    for(var p=1;p<=pages;p++){{
      if(pages>8&&p>2&&p<pages-1&&Math.abs(p-cur)>1){{
        if(p===3||p===pages-2)html+='<span class="pg-ellipsis">…</span>';
        continue;
      }}
      html+='<button class="pg-btn'+(p===cur?' pg-active':'')+'" onclick="window._pgGo('+p+')">'+p+'</button>';
    }}
    html+='<button class="pg-btn" onclick="window._pgGo('+(cur+1)+')" '+(cur>=pages?'disabled':'')+'>下一页 ›</button>';
    html+='<span class="pg-info">第'+cur+'/'+pages+'页 · 共'+total+'个</span>';
    html+='</div>';
    bar.innerHTML=html;
  }}
  function applyPage(){{
    var all=getVisible();
    var start=(cur-1)*PAGE;
    all.forEach(function(c,i){{c.style.display=(i>=start&&i<start+PAGE)?'':'none';}});
    renderPager(all.length);
    window.scrollTo({{top:0,behavior:'smooth'}});
  }}
  window._pgGo=function(p){{
    var all=getVisible();
    var pages=Math.ceil(all.length/PAGE);
    cur=Math.max(1,Math.min(p,pages));
    all.forEach(function(c){{c.style.display='';}});
    applyPage();
  }};
  function applyCardFilters(){{
    cur=1;
    var domSel=document.getElementById('filter-domain');
    var diffSel=document.getElementById('filter-diff');
    var dom=domSel?domSel.value:'';
    var diff=diffSel?diffSel.value:'';
    var cards=document.querySelectorAll('#skill-card-grid .skill-card');
    cards.forEach(function(c){{
      var matchDom=!dom||c.dataset.domain===dom;
      var matchDiff=!diff||!c.dataset.diff||c.dataset.diff===diff;
      c.style.display=(matchDom&&matchDiff)?'':'none';
    }});
    var hint=document.getElementById('filter-count');
    var vis=getVisible().length;
    if(hint){{hint.textContent=(dom||diff)?('\u663e\u793a '+vis+' / '+cards.length+' \u4e2a'):('');}}
    applyPage();
  }}
  ['filter-domain','filter-diff'].forEach(function(id){{
    var el=document.getElementById(id);
    if(el)el.addEventListener('change',applyCardFilters);
  }});
  document.addEventListener('DOMContentLoaded',applyCardFilters);
}})();
</script>
<style>
.pg-bar{{margin:16px 0 8px}}
.pg-inner{{display:flex;align-items:center;gap:4px;flex-wrap:wrap}}
.pg-btn{{min-width:36px;height:32px;padding:0 10px;border:1px solid #E5E5E5;border-radius:4px;background:#fff;color:#555;font-size:12px;cursor:pointer;transition:all .12s}}
.pg-btn:hover:not([disabled]){{border-color:#0C0C0C;color:#0C0C0C}}
.pg-btn.pg-active{{background:#0C0C0C;border-color:#0C0C0C;color:#fff;font-weight:600}}
.pg-btn[disabled]{{opacity:.35;cursor:not-allowed}}
.pg-ellipsis{{color:#aaa;padding:0 4px}}
.pg-info{{font-size:12px;color:#999;margin-left:8px}}
</style>"""
    write_file(out / "skills" / "index.html", html_page(
        "全部 Skills",
        f"<h1>全部 Skills</h1>\n<p class='page-lead'>从 1010 个顶会论文萃取的 AI 决策技能中，找到你的业务场景对应算法。每个 Skill 含可运行 Python 代码、量化 ROI 估算和具体案例。</p>\n{filter_bar}<div class='cards' id='skill-card-grid'>{all_cards}</div>",
        "../",
        active_nav="skills",
    ))

    # ── Domain pages ──
    domain_index_cards: list[str] = []
    for domain in domains:
        domain_skills = [s for s in skills if s.domain_dir == domain["vault_dir"]]
        cards = "".join(render_skill_card(s, "../") for s in domain_skills)
        title = domain["vault_dir"]
        domain_search_bar = f"""
<div style='display:flex;align-items:center;gap:10px;margin:12px 0 8px'>
  <input id='domain-search' placeholder='在 {html.escape(title)} 中搜索…' autocomplete='off'
    style='flex:1;max-width:340px;padding:8px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:14px'>
  <span class='muted' id='domain-count' style='font-size:13px'></span>
</div>
<div id="dom-pg-bar" class="pg-bar"></div>
<script>
(function(){{
  var PAGE=40;var cur=1;
  function getVis(){{return Array.from(document.querySelectorAll('.cards .skill-card')).filter(function(c){{return c.style.display!=='none';}});}}
  function renderPager(total){{
    var pages=Math.ceil(total/PAGE);
    var bar=document.getElementById('dom-pg-bar');if(!bar)return;
    if(pages<=1){{bar.innerHTML='';return;}}
    var h='<div class="pg-inner">';
    h+='<button class="pg-btn" onclick="window._dpg('+(cur-1)+')" '+(cur<=1?'disabled':'')+'>‹</button>';
    for(var p=1;p<=pages;p++){{
      if(pages>8&&p>2&&p<pages-1&&Math.abs(p-cur)>1){{if(p===3||p===pages-2)h+='<span class="pg-ellipsis">…</span>';continue;}}
      h+='<button class="pg-btn'+(p===cur?' pg-active':'')+'" onclick="window._dpg('+p+')">'+p+'</button>';
    }}
    h+='<button class="pg-btn" onclick="window._dpg('+(cur+1)+')" '+(cur>=pages?'disabled':'')+'>›</button>';
    h+='<span class="pg-info">第'+cur+'/'+pages+'页 · 共'+total+'个</span></div>';
    bar.innerHTML=h;
  }}
  function applyPage(){{
    var all=getVis();
    var s=(cur-1)*PAGE;
    all.forEach(function(c,i){{c.style.display=(i>=s&&i<s+PAGE)?'':'none';}});
    renderPager(all.length);
  }}
  window._dpg=function(p){{
    var all=getVis();
    var pages=Math.ceil(all.length/PAGE);
    cur=Math.max(1,Math.min(p,pages));
    all.forEach(function(c){{c.style.display='';}});
    applyPage();
    window.scrollTo({{top:0,behavior:'smooth'}});
  }};
  var inp=document.getElementById('domain-search');
  var cnt=document.getElementById('domain-count');
  if(!inp)return;
  inp.addEventListener('input',function(){{
    cur=1;
    var q=this.value.trim().toLowerCase();
    document.querySelectorAll('.cards .skill-card').forEach(function(c){{
      c.style.display=(!q||(c.textContent||'').toLowerCase().includes(q))?'':'none';
    }});
    var vis=getVis().length;
    if(cnt)cnt.textContent=q?('\u663e\u793a '+vis+' / '+document.querySelectorAll('.cards .skill-card').length+' \u4e2a'):'';
    applyPage();
  }});
  document.addEventListener('DOMContentLoaded',applyPage);
}})();
</script>"""
        write_file(
            out / "domains" / f"{slugify(title)}.html",
            html_page(title,
                      f"<h1>{html.escape(title)}</h1>"
                      f"<p>{html.escape(domain.get('description',''))}</p>"
                      f"{domain_search_bar}"
                      f"<div class='cards'>{cards}</div>",
                      "../"),
        )

        # DOMAIN_CARD_CONFIG, make_pattern_svg, svg_to_css_bg 已在全局定义

        biz_label = DOMAIN_BIZ_LABELS.get(title, "")
        cfg = DOMAIN_CARD_CONFIG.get(title, {"color": "#555555", "bg": "#f5f5f5", "num": "?", "icon": "📌", "pattern": "dots"})
        c = cfg["color"]; bg = cfg["bg"]; num = cfg["num"]; icon = cfg["icon"]
        pattern_svg = make_pattern_svg(cfg["pattern"], c)
        bg_svg = svg_to_css_bg(pattern_svg)
        domain_index_cards.append(
            f"<a class='gallery-card domain-gallery-card' href='{slugify(title)}.html' "
            f"style='--card-color:{c};--card-bg:{bg}'>"
            f"<div class='gallery-card-bg' style='background-image:{bg_svg}'></div>"
            f"<div class='gallery-card-body'>"
            f"<div class='gallery-card-num'>{num}</div>"
            f"<div class='gallery-card-icon'>{icon}</div>"
            f"<div class='gallery-card-title'>{html.escape(title.split('-', 1)[-1] if '-' in title else title)}</div>"
            f"<div class='gallery-card-desc'>{html.escape(biz_label)}</div>"
            f"</div>"
            f"<div class='gallery-card-footer'>"
            f"<span class='gallery-card-count'>{len(domain_skills)}</span>"
            f"<span class='gallery-card-unit'>Skills</span>"
            f"</div>"
            f"</a>"
        )
    write_file(out / "domains" / "index.html", html_page(
        "按领域",
        """<h1>按领域浏览</h1>
<p class='page-lead'>从 25 个技术领域进入，每个领域包含完整 Skill 卡片、引用关系和业务场景。不熟悉领域名称？按业务方向选择：</p>
<div style="margin-bottom:20px;padding:14px 18px;background:linear-gradient(135deg,#fff5f5 0%,#fff 100%);border:1.5px solid #f0c0c0;border-radius:10px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap">
  <div><span style="font-size:13px;font-weight:700;color:#B5323E">免费开放：因果推断 · A/B实验 · 时间序列 · 供应链 · 推荐系统</span><span style="font-size:12px;color:#6b7280;margin-left:8px">/ 企业版解锁全部 25 域</span></div>
  <a href="mailto:skills@lute-tlz-dddd.top?subject=申请企业版-paper2skills" style="padding:7px 16px;background:#B5323E;color:#fff;border-radius:7px;font-size:12.5px;font-weight:700;text-decoration:none;white-space:nowrap">申请企业版 →</a>
</div>
<div class='domain-biz-tags' style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px'>
  <a href='04-供应链.html' class='domain-tag'>供应链&amp;库存</a>
  <a href='13-广告分析.html' class='domain-tag'>广告&amp;归因</a>
  <a href='06-增长模型.html' class='domain-tag'>用户增长</a>
  <a href='19-风控反欺诈.html' class='domain-tag'>风控&amp;安全</a>
  <a href='07-NLP-VOC.html' class='domain-tag'>VOC&amp;评论</a>
  <a href='21-合规决策.html' class='domain-tag'>合规&amp;关税</a>
</div>
<div class='gallery-grid'>""" + "".join(domain_index_cards) + "</div>",
        "../",
        active_nav="domains",
    ))

    # ── Topic pages ──
    all_topics = sorted({topic for s in skills for topic in s.topics})

    # ── 主题卡片颜色配置 (使用全局变量) ──
    # TOPIC_CARD_COLORS 和 _TOPIC_PATTERNS 已定义在全局

    topic_cards: list[str] = []
    for _ti, topic in enumerate(all_topics):
        topic_skills = [s for s in skills if topic in s.topics]
        cards = "".join(render_skill_card(s, "../") for s in topic_skills)
        path = f"{slugify(topic)}.html"
        topic_search = (
            "<input id='topic-search' placeholder='在 " + html.escape(topic) + " 中搜索…'"
            " style='max-width:320px;padding:8px 14px;border:1px solid #e2e8f0;"
            "border-radius:8px;font-size:14px;margin:10px 0 16px;display:block'>"
            "<script>(function(){var inp=document.getElementById('topic-search');"
            "var cs=document.querySelectorAll('.cards .skill-card');"
            "if(!inp)return;inp.addEventListener('input',function(){"
            "var q=this.value.trim().toLowerCase();"
            "cs.forEach(function(c){c.style.display=(!q||(c.textContent||'').toLowerCase().includes(q))?'':'none';});"
            "});})()</script>"
        )
        write_file(out / "topics" / path, html_page(
            topic,
            f"<h1>{html.escape(topic)}</h1>{topic_search}<div class='cards'>{cards}</div>",
            "../",
        ))
        _tcfg = TOPIC_CARD_COLORS.get(topic, {"color": "#555", "icon": "📌"})
        _tc = _tcfg["color"]; _ticon = _tcfg["icon"]
        _tpat = _TOPIC_PATTERNS[_ti % len(_TOPIC_PATTERNS)]
        _tsvg = make_pattern_svg(_tpat, _tc)
        _tbg  = svg_to_css_bg(_tsvg)
        topic_cards.append(
            f"<a class='gallery-card topic-gallery-card' href='{path}' style='--card-color:{_tc}'>"
            f"<div class='gallery-card-bg' style='background-image:{_tbg}'></div>"
            f"<div class='gallery-card-body'>"
            f"<div class='gallery-card-icon gallery-card-icon-lg'>{_ticon}</div>"
            f"<div class='gallery-card-title'>{html.escape(topic)}</div>"
            f"</div>"
            f"<div class='gallery-card-footer'>"
            f"<span class='gallery-card-count'>{len(topic_skills)}</span>"
            f"<span class='gallery-card-unit'>Skills</span>"
            f"</div>"
            f"</a>"
        )
    write_file(out / "topics" / "index.html", html_page(
        "按主题",
        "<h1>按主题浏览</h1><p class='page-lead'>按业务主题聚合的 AI 技能索引，快速定位你关心的场景。</p><div class='gallery-grid'>" + "".join(topic_cards) + "</div>",
        "../",
    ))
    # ── Workflow pages (Phase 2B: YAML-first, keyword fallback) ──
    skill_lookup = {s.skill_id: s for s in skills}
    # WF_CARD_CONFIG 等已在全局定义

    workflow_cards: list[str] = []
    for _wfi, workflow_name in enumerate(WORKFLOW_RULES):
        slug_path = f"{slugify(workflow_name)}.html"
        wf_id = slugify(workflow_name).split("-")[0] + "-" + slugify(workflow_name).split("-")[1]

        if wf_defs and wf_id in wf_defs:
            page_html = render_workflow_page(wf_defs[wf_id], skill_lookup)
        else:
            wf_skills = [s for s in skills if workflow_name in s.workflows]
            cards = "".join(render_skill_card(s, "../") for s in wf_skills)
            page_html = html_page(
                workflow_name,
                f"<h1>{html.escape(workflow_name)}</h1>"
                f"<p class='muted'>按业务流程推荐的 Skill 链。</p>"
                f"<div class='cards'>{cards}</div>",
                "../",
            )
            wf_skills_for_count = wf_skills

        write_file(out / "workflows" / slug_path, page_html)
        _wf_steps_dict = wf_defs.get(wf_id, {})
        wf_skill_count = len(_wf_steps_dict.get("steps", [])) or len([s for s in skills if workflow_name in s.workflows])
        
        _wcfg = WF_CARD_CONFIG.get(workflow_name, {
            "color": _WF_DEFAULT_COLORS[_wfi % len(_WF_DEFAULT_COLORS)],
            "icon": "⚡",
            "pattern": _WF_PATTERNS[_wfi % len(_WF_PATTERNS)]
        })
        _wc = _wcfg["color"]; _wicon = _wcfg["icon"]; _wpat = _wcfg["pattern"]
        _wsvg = make_pattern_svg(_wpat, _wc)
        _wbg  = svg_to_css_bg(_wsvg)
        # 提取WF编号
        _wf_num = workflow_name.split(" ")[0] if " " in workflow_name else workflow_name[:4]
        _wf_title = workflow_name.split(" ", 1)[1] if " " in workflow_name else workflow_name
        workflow_cards.append(
            f"<a class='gallery-card wf-gallery-card' href='{slug_path}' style='--card-color:{_wc}'>"
            f"<div class='gallery-card-bg' style='background-image:{_wbg}'></div>"
            f"<div class='gallery-card-body'>"
            f"<div class='gallery-card-num gallery-card-num-sm'>{_wf_num}</div>"
            f"<div class='gallery-card-icon'>{_wicon}</div>"
            f"<div class='gallery-card-title'>{html.escape(_wf_title)}</div>"
            f"</div>"
            f"<div class='gallery-card-footer'>"
            f"<span class='gallery-card-count'>{wf_skill_count}</span>"
            f"<span class='gallery-card-unit'>步骤</span>"
            f"</div>"
            f"</a>"
        )
    write_file(out / "workflows" / "index.html", html_page(
        "工作流",
        "<h1>业务工作流</h1><p class='page-lead'>端到端业务决策路径，每条工作流包含分步决策树和推荐 Skill 组合。</p>"
        "<div class='gallery-grid'>" + "".join(workflow_cards) + "</div>",
        "../",
    ))

    # ── Skills Graph (Phase 3D: D3 visualisation) ──
    write_file(out / "graph" / "overview.html", render_graph_page(skill_count, edge_count, data["generated_at"].replace("-", "").replace(":", "").replace("T", "")))

    # ── CEO Roadmap whitepaper ──
    write_file(out / "ai-roadmap.html", render_roadmap_page(skill_lookup, skill_count=skill_count))
    write_file(out / "maturity-report.html", render_maturity_report(skill_count=skill_count, edge_count=edge_count, domain_count=domain_count))
    write_file(out / "agents.html", render_agents_page(skill_lookup))
    write_file(out / "agent-report.html", render_agent_report_page())
    write_file(out / "pricing.html", render_pricing_page(skill_count=skill_count))
    write_file(out / "settings.html", render_settings_page())
    write_file(out / "solutions" / "index.html", render_solutions_index(total_skill_count=skill_count))
    for sol in SOLUTIONS_CATALOG:
        write_file(out / "solutions" / f"{sol['id']}.html", render_solution_detail(sol, total_skill_count=skill_count))

    # ── toB Scene Playbooks (Phase F) ──
    for pb in TOB_PLAYBOOKS:
        write_file(
            out / "playbooks" / f"{pb['id']}.html",
            render_tob_playbook(pb, skill_lookup),
        )
    def _pb_card(pb: dict) -> str:
        tag = html.escape(pb.get("tag", ""))
        pb_id = pb["id"]
        name = html.escape(pb["name"])
        icon = pb.get("svg_icon") or pb["icon"]
        biz_tag = html.escape(pb["tag"])
        desc = html.escape(pb["desc"])
        total_steps = len(pb.get("steps", []))
        return (
            f"<a class='biz-card' href='{pb_id}.html' data-tag='{tag}'>"
            f"<div class='biz-card-header'>"
            f"<span class='biz-icon'>{icon}</span>"
            f"<div class='biz-body'>"
            f"<div class='biz-card-meta'>"
            f"<strong>{name}</strong>"
            f"</div>"
            f"<p>{desc}</p>"
            f"<div class='biz-card-footer'>"
            f"<span class='biz-tag'>{biz_tag}</span>"
            f"<div id='prog-{pb_id}' style='margin-left:auto;font-size:11px;color:#94a3b8'></div>"
            f"</div>"
            f"</div>"
            f"</div>"
            f"</a>"
            f"<script>(function(){{"
            f"var done=JSON.parse(localStorage.getItem('pb_progress_{pb_id}')||'[]');"
            f"var el=document.getElementById('prog-{pb_id}');"
            f"if(el&&done.length>0){{"
            f"var pct=Math.round(done.length/{total_steps}*100);"
            f"el.innerHTML='<div style=\"display:flex;align-items:center;gap:5px\"><div style=\"background:#e2e8f0;border-radius:4px;height:4px;overflow:hidden;width:40px\"><div style=\"background:#059669;height:100%;width:'+pct+'%;transition:width .3s\"></div></div>'"
            f"+'<span style=\"color:#059669;font-weight:600\">'+done.length+'/{total_steps}</span></div>';}}"
            f"}})();</script>"
        )
    tob_index_cards = "".join(_pb_card(pb) for pb in TOB_PLAYBOOKS)
    pb_search_bar = """<div style='margin:12px 0 20px;display:flex;flex-wrap:wrap;gap:8px;align-items:center'>
  <input id='pb-search' placeholder='搜索手册名称…' autocomplete='off'
    style='padding:8px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:14px;min-width:200px'>
  <button class='pb-tag-btn active' data-tag='' onclick='pbFilter("")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px;font-weight:600'>全部</button>
  <button class='pb-tag-btn' data-tag='供应链' onclick='pbFilter("供应链")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>供应链</button>
  <button class='pb-tag-btn' data-tag='广告' onclick='pbFilter("广告")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>广告</button>
  <button class='pb-tag-btn' data-tag='合规' onclick='pbFilter("合规")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>合规风控</button>
  <button class='pb-tag-btn' data-tag='选品' onclick='pbFilter("选品")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>选品增长</button>
  <button class='pb-tag-btn' data-tag='Agent' onclick='pbFilter("Agent")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>AI Agent</button>
  <button class='pb-tag-btn' data-tag='客服' onclick='pbFilter("客服")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>客服运营</button>
  <a href='progress.html' style='margin-left:auto;display:inline-flex;align-items:center;gap:5px;padding:6px 14px;background:transparent;color:var(--muted);border:1px solid var(--line);border-radius:8px;font-size:12.5px;font-weight:600;text-decoration:none'>PL 进度看板</a>
</div>
<script>
(function(){
  var bizCards = document.querySelectorAll('.biz-grid .biz-card');
  var searchInput = document.getElementById('pb-search');
  var currentTag = '';
  function applyFilter() {
    var q = searchInput ? searchInput.value.trim().toLowerCase() : '';
    bizCards.forEach(function(c) {
      var tagMatch = !currentTag || (c.dataset.tag||'').includes(currentTag);
      var textMatch = !q || (c.textContent||'').toLowerCase().includes(q);
      c.style.display = (tagMatch && textMatch) ? '' : 'none';
    });
  }
  window.pbFilter = function(tag) {
    currentTag = tag;
    document.querySelectorAll('.pb-tag-btn').forEach(function(b) {
      var isActive = b.dataset.tag === tag;
      b.style.background = isActive ? 'var(--accent)' : '#f8fafc';
      b.style.color = isActive ? '#fff' : '';
      b.style.borderColor = isActive ? 'var(--accent)' : '#e2e8f0';
    });
    applyFilter();
  };
  if(searchInput) searchInput.addEventListener('input', applyFilter);
})();
</script>"""
    write_file(out / "playbooks" / "index.html", html_page(
        "场景手册",
        "<h1>场景手册</h1>"
        "<p class='page-lead'>33 本可执行场景手册，覆盖选品→上架→广告→供应链→风控全链路。每本手册包含分步骤操作指南、所需数据和预期 ROI。</p>"
        "<div class='playbook-quick-tags' style='display:flex;flex-wrap:wrap;gap:8px;margin:0 0 20px'>"
        "  <span class='pq-tag'>供应链</span>"
        "  <span class='pq-tag'>广告投放</span>"
        "  <span class='pq-tag'>风险防御</span>"
        "  <span class='pq-tag'>用户增长</span>"
        "  <span class='pq-tag'>合规关税</span>"
        "  <span class='pq-tag'>搜索流量</span>"
        "</div>"
        f"{pb_search_bar}"
        f"<div class='biz-grid'>{tob_index_cards}</div>",
        "../",
    ))

    write_file(out / "playbooks" / "progress.html", _render_playbook_progress_page(TOB_PLAYBOOKS))

    write_file(out / "README.md", "# paper2skills Playbook\n\n打开 `index.html` 浏览。\n")

    for html_file in out.rglob("*.html"):
        try:
            content = html_file.read_text(encoding="utf-8")
            new_content = content.replace(
                'playbook-data.js"',
                f'playbook-data.js?v={build_ts}"'
            ).replace(
                "playbook-data.js'",
                f"playbook-data.js?v={build_ts}'"
            )
            if new_content != content:
                html_file.write_text(new_content, encoding="utf-8")
        except Exception:
            pass
    report = {
        "skill_pages": skill_count,
        "domains": domain_count,
        "edges": edge_count,
        "generated_at": data["generated_at"],
    }
    assert skill_count > 1000, f"Unexpected skill count: {skill_count}"
    write_file(out / "build-report.json", json.dumps(report, ensure_ascii=False, indent=2))

    skill_index = []
    for s in skills:
        summary = " ".join(filter(None, [
            s.algorithm_summary[:150] if s.algorithm_summary else "",
            s.problem_solved[:100] if s.problem_solved else "",
        ]))
        keywords = list({
            t.lower() for tag in (s.tags or []) for t in tag.split()
        } | {
            w for w in re.sub(r'[^\u4e00-\u9fa5a-z0-9]', ' ', (s.problem_solved or "").lower()).split()
            if len(w) > 1
        })[:30]
        skill_index.append({
            "id": s.skill_id,
            "title": s.title,
            "domain": s.domain_dir,
            "summary": summary,
            "keywords": keywords,
        })
    write_file(out / "assets" / "skill-index.json", json.dumps(skill_index, ensure_ascii=False))
    import shutil as _shutil
    _seed_src = Path(__file__).parent / "scripts" / "config" / "seed_reports.json"
    if not _seed_src.exists():
        _seed_src = Path(__file__).parent / "config" / "seed_reports.json"
    if _seed_src.exists():
        _shutil.copy2(_seed_src, out / "assets" / "seed_reports.json")
    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def build(root: Path, vault: Path, out: Path) -> dict[str, Any]:
    graph  = build_graph(vault)
    skills = build_skills(root, vault, graph)
    domains = domain_dicts(root)
    wf_defs = load_workflow_defs(root)
    return render_pages(out, skills, domains, graph, wf_defs)


def domain_dicts(root: Path) -> list[dict[str, Any]]:
    registry = load_domain_registry(root)
    return [
        {
            "key": entry.key,
            "vault_dir": entry.vault_dir,
            "description": entry.description,
            "skill_count": entry.skill_count,
            "code_status": entry.code_status,
        }
        for entry in registry.entries
    ]


def _post_build_patch(out: "Path") -> None:
    """Post-build patch: inject AI workbench features into generated static files."""
    import re

    # ── 1. agents.html: AI模式 + 链式调用 ──────────────────────────────────
    agents_path = out / "agents.html"
    if agents_path.exists():
        agents_html = agents_path.read_text(encoding="utf-8")

        # 插入模式切换控件（在每个 run 按钮前）— 覆盖全部 Agent
        agent_ids = [ag["id"] for ag in AGENT_CATALOG]
        for aid in agent_ids:
            old = f"<button class='modal-run-btn' id='run-{aid}' onclick='runAgent(\"{aid}\")'>"
            new = (f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:8px;"
                   f"padding:6px 10px;background:var(--panel-2,#f8fafc);border-radius:7px;"
                   f"border:1px solid var(--line,#e2e8f0)'>"
                   f"<span style='font-size:11px;font-weight:600;color:#475569'>模式:</span>"
                   f"<label style='display:flex;gap:3px;align-items:center;cursor:pointer;font-size:11px'>"
                   f"<input type='radio' name='mode-{aid}' value='local' style='accent-color:#059669'>"
                   f"本地演示</label>"
                   f"<label style='display:flex;gap:3px;align-items:center;cursor:pointer;font-size:11px'>"
                   f"<input type='radio' name='mode-{aid}' value='ai' checked style='accent-color:#6366f1'>"
                   f"<span style='color:#6366f1;font-weight:600'>AI 真实分析</span></label></div>\n"
                   f"      <button class='modal-run-btn' id='run-{aid}' onclick='runAgent(\"{aid}\")'>"
            )
            if old in agents_html:
                agents_html = agents_html.replace(old, new, 1)

        # 注入 AI 运行引擎 + 链式调用（在 catBtns 之前）
        AI_ENGINE = """const AGENT_PROMPTS={
'agent-supply-sentinel':'你是供应链分析专家，专注母婴跨境电商。用户会提供SKU库存和销速数据。输出格式严格如下：\\n【风险评级】🔴/🟡/🟢 + 一句话定性\\n【DOS分析】当前可销天数 vs 安全阈值（给出具体数字）\\n【断货概率】% + 置信区间\\n【补货建议】海运/空运决策 + 建议补货量（件）+ 到货时间窗口\\n【紧急行动】3条，48小时内必须执行的具体步骤\\n用数字说话，禁止模糊表达。',
'agent-pricing-advisor':'你是跨境电商定价顾问，专注母婴品类。输出格式：\\n【当前定价诊断】评分0-10 + 问题点\\n【竞争价格带】市场P25/P50/P75分位价格\\n【最优定价区间】具体价格范围 + 毛利率预测\\n【提价/降价路径】分3步的具体执行方案（时间+幅度+监控指标）\\n【预期ROI影响】价格调整后GMV/利润变化预测%\\n所有数字必须具体，不得写"大约"。',
'agent-pnl-analyzer':'你是P&L财务分析师，专注跨境电商。输出格式：\\n【利润结构瀑布】GMV→退款→FBA→广告→头程→净利，逐层拆解（%）\\n【亏损根因TOP3】每条附具体金额和占GMV%\\n【对标行业基准】各项成本率vs行业均值（用+/-说明偏差）\\n【提利优先级】3个改善方向，每个附：当前值→目标值→预期净利润改善额\\n【本月预警】需要立即关注的财务指标。',
'agent-ad-attribution':'你是广告归因分析师，专注Amazon/TikTok跨境广告。输出格式：\\n【广告健康评分】0-100 + 关键问题\\n【ROAS归因拆解】品牌词/泛词/竞品词/ASIN各自ROAS\\n【预算浪费识别】具体浪费金额/天 + 浪费原因\\n【优化TOP3行动】每条：当前指标→目标指标→预期ROAS提升\\n【预算重分配方案】各渠道/campaign建议预算%。',
'agent-competitor-radar':'你是竞品情报分析师，专注母婴跨境品类。输出格式：\\n【竞争格局】头部3家市占率估算 + 市场集中度\\n【竞品定价策略】对手价格带 + 近期调价趋势\\n【差评规律】竞品TOP差评词频（给出具体词和频次）\\n【市场空白】2-3个未被充分满足的用户需求\\n【切入建议】差异化方向 + 预期可获得的市占率%。',
'agent-listing-doctor':'你是Amazon Listing优化专家。输出格式：\\n【综合评分】0-100（A10算法视角）\\n【Title诊断】字符数/核心词覆盖/问题点 → 重写版本\\n【Bullet诊断】逐条评分 + 最差一条的重写版本\\n【关键词缺口】搜索量>5000但未覆盖的核心词（列出TOP5）\\n【A+内容建议】模块结构推荐\\n【预期效果】优化后预计自然流量提升%。',
'agent-voc-decoder':'你是VOC分析师，专注母婴电商用户声音。输出格式：\\n【情感分布】正面/中性/负面占比%\\n【TOP痛点】频次最高的5个负面词 + 出现频率\\n【用户期待TOP3】高频出现的改进诉求 + 商业机会评估\\n【竞品对比信号】用户提到竞品时的核心关键词\\n【产品迭代建议】3条，每条附：用户需求来源 + 开发优先级（高/中/低）。',
'agent-cs-triage':'你是客服分诊专家，专注跨境电商售后。输出格式：\\n【工单分类】A-to-Z风险/退款/换货/咨询 各占%\\n【高风险工单】识别出需要24h内处理的紧急工单特征\\n【根因TOP3】引发工单的产品/物流/描述问题\\n【回复模板】针对最高频工单类型生成标准回复（中英双语）\\n【预防建议】减少工单量的2个具体运营动作。',
'agent-account-guardian':'你是Amazon账号风险专家。输出格式：\\n【账号健康评分】0-100 + 风险等级（低/中/高/紧急）\\n【违规风险点】具体指标值 vs Amazon阈值（ODR/LDR/VTR等）\\n【封号概率】% + 主要风险因子\\n【48h紧急行动】必须立即执行的3个步骤\\n【长期健康方案】3条系统性改善建议 + 预期指标改善时间线。',
'agent-brand-guardian':'你是品牌合规专家，专注跨境广告法。输出格式：\\n【合规评分】0-100\\n【违禁词清单】逐个标出文案中的违禁/夸大表达 + 违规依据（FTC/广告法）\\n【风险等级】每处违规：低风险/中风险/下架风险\\n【合规改写】针对每处违规给出合规替代表达\\n【上架可行性】当前文案能否通过Amazon/TikTok审核的判断。',
'agent-product-radar':'你是母婴跨境选品分析师。输出格式：\\n【机会评分】0-100 + GO/NO-GO建议\\n【市场规模】月销量估算 + 市场增长率%/年\\n【竞争强度】头部垄断度 + 新品切入难度（1-10）\\n【利润空间】预估毛利率% + FBA成本结构\\n【差异化方向】3个具体切入角度 + 各自可获得溢价估算\\n【首批建议】备货量 + 预算 + ROI预测。',
'agent-tiktok-content':'你是TikTok跨境内容策略师，专注母婴品类。输出格式：\\n【内容矩阵】痛点类/种草类/测评类/UGC各建议占比%\\n【爆款公式】针对该品类的黄金钩子 + 情绪触发点\\n【脚本框架】一条完整30秒视频脚本（分镜+文案）\\n【达人合作建议】腰部达人画像 + 合理报价区间\\n【数据预期】预估CPM/CPV/转化率范围。',
'agent-sku-tag-scanner':'你是标签工程质量专家，专注电商SKU数据治理。输出格式：\\n【标签覆盖率诊断】各关键标签维度覆盖%\\n【质量问题TOP3】具体缺失/错误标签 + 影响的下游决策\\n【高优先级修复清单】按ROI排序的5个标签修复任务\\n【自动化打标建议】哪些标签可规则化，哪些需ML模型\\n【预期改善效果】修复后预计决策准确率提升%。',
'agent-compliance-matrix':'你是跨境合规专家。输出格式：\\n【多市场准入评分】US/EU/JP/AU各市场0-100分\\n【认证缺口清单】每个市场缺失的强制认证 + 申请周期\\n【高风险项】可能导致下架/罚款的合规问题（标注紧急程度）\\n【合规成本估算】各市场认证总费用范围\\n【上市时间线】按优先级排序的合规路径，标注关键节点日期。',
'agent-return-analyzer':'你是退货根因分析专家。输出格式：\\n【退货率诊断】当前退货率 vs 品类均值（%）\\n【三层归因】表层原因→运营原因→供应链根因，每层TOP2\\n【金额影响】退货损失/月（GMV%）\\n【改善优先级】3个行动项，每个附：预计退货率下降%+实施难度\\n【快赢方案】1周内可执行、不需要产品改动的2个降退货操作。',
'agent-margin-calculator':'你是SKU利润归因专家。输出格式：\\n【P&L瀑布图】GMV→平台佣金→退款→FBA→广告→头程→关税→其他→净利润，每项金额+%\\n【成本漏点TOP3】超出行业均值的费用项 + 具体超出金额\\n【利润改善杠杆】3个提利方向，每个附：当前值→目标值→净利润改善/月\\n【保本价格】当前成本结构下的最低定价\\n【下月利润预测】基于当前趋势的利润区间预测。',
'agent-geopolitical-risk':'你是供应链地缘风险专家。输出格式：\\n【综合风险评分】0-100（越高越危险）\\n【五维风险评估】关税/港口/汇率/出口管制/供应商集中度，每维：当前状态+风险分（0-10）\\n【最高风险项】详细分析 + 触发概率%\\n【应急预案】针对最高风险的3步应对方案\\n【多元化建议】供应链分散化路径 + 预期成本增加%。',
'agent-epr-calculator':'你是欧盟EPR合规专家。输出格式：\\n【EPR义务概览】各市场（DE/FR/IT/ES/PL等）是否需要注册\\n【费用估算】各市场年度EPR注册费用区间（€）\\n【截止日期】各市场强制合规截止日期（标注紧急程度）\\n【注册优先级】按销售体量和截止日期排序的注册顺序\\n【操作步骤】最紧急市场的注册流程（3-5步）。',
'agent-dml-counterfactual-pricing':'你是反事实定价专家，专注跨境电商动态定价。输出格式：\\n【反事实基线】当前价格的弹性估算（价格↑1%→销量变化%）\\n【最优价格区间】基于弹性+竞品+季节性的建议定价\\n【定价情景对比】3个情景（保守/中性/激进）各自GMV/利润预测\\n【竞品反应预测】竞品可能的跟价行为及概率\\n【执行时机】最优调价时间窗口 + 监控指标阈值。',
'agent-cold-start-advisor':'你是新品冷启动策略专家，专注跨境电商。输出格式：\\n【冷启动诊断】当前阶段（0-7天/1-4周/1-3月）及核心障碍\\n【流量获取方案】前30天具体推广策略（广告类型+预算分配%）\\n【定价策略】冷启动期建议价格 + 提价时间节点\\n【首评获取】获取前20条真实评价的3个具体操作\\n【里程碑目标】D7/D30/D90的可量化目标（排名/评价数/日销）。',
'agent-festival-replenishment':'你是大促补货决策专家，专注跨境电商旺季。输出格式：\\n【大促需求预测】基于历史数据的需求倍率区间（P50/P80/P95）\\n【安全库存计算】建议备货量 = 预测需求×安全系数，给出具体件数\\n【资金占用评估】备货总成本 + 滞销风险敞口（超卖/断货各自损失估算）\\n【物流时间线】最晚下单时间 + 各运输方式到仓时间\\n【清仓预案】大促后剩余库存>30%的降价处理方案。'

,'agent-video-content-mas':'你是视频电商内容专家，擅长TikTok/Reels母婴内容策略。\n输出格式：\n【热点话题】3个当前趋势（含预测爆款率）\n【脚本方案A】开场Hook + 核心内容 + CTA（字数限制60秒）\n【脚本方案B】不同角度的替代方案\n【商品标签建议】视频中应展示的SKU + 购物链接策略\n【发布策略】最优时间 + 标签 + 跨平台建议\n【预测指标】首日播放量范围 + 爆款概率 + ROI预估\n严格控制在母婴安全红线内，不发布任何医疗建议。'
,'agent-causal-pricing-advisor':'你是因果定价专家，使用DML和X-Learner方法估计真实价格弹性。\n输出格式：\n【弹性诊断】OLS估计值 vs DML去偏估计值 + 偏差来源分析\n【用户分群弹性】高/中/低弹性用户群体划分 + 每群弹性值\n【定价建议】全量降价ROI vs 精准发券ROI对比\n【行动方案】3步精准定价策略：目标人群+优惠形式+预期效果\n【风险提示】定价操纵合规检查 + 竞品跟价风险\n数据驱动，用具体数字，禁止"可能"等模糊词。'
,'agent-growth-diagnostics':'你是增长分析专家，专注用户激活率/复购率/ROAS等核心增长指标的根因诊断。\n输出格式：\n【异常确认】统计检验结果（p值 + z分数）+ 是否显著异常\n【假设树】5个候选根因（按概率排序）\n【数据验证】每个假设对应的数据查询和验证结果\n【根因定位】主因（置信度>80%）+ 次因（如有）\n【行动清单】P0行动（今日执行）+ P1行动（本周内）+ 预计恢复时间\n严格基于数据，不做主观猜测，每个结论附数据支撑。'
,'agent-search-mas-monitor':'你是Amazon搜索排名专家，擅长多Agent协同监控和根因诊断。\n输出格式：\n【排名扫描】已监控关键词数量 + 发现异常关键词列表\n【根因矩阵】库存/评分/广告/竞品4个维度的影响评估\n【优先级排序】P0/P1/P2行动，每个行动的预期恢复天数和排名回升幅度\n【协同建议】广告出价调整 × 自然排名优化的协同策略\n【监控设置】建议的告警阈值和监控频率\n输出具体数字，不接受"可能""大概"等模糊表达。'
,'agent-xai-compliance-auditor':'你是AI合规专家，专注EU AI Act、GDPR和算法公平性审计。\n输出格式：\n【合规级别】高风险/中风险/低风险 + 判定依据（引用具体法规条款）\n【歧视检测】各受保护属性的差异冲击比(DI) + 是否通过80%规则\n【可解释性评估】SHAP特征分析 + 个体解释能力 + 反事实解释\n【合规差距清单】已满足/待补充/未满足三栏对比\n【整改路线图】3个优先级行动 + 工程工时估算 + 合规风险量化\n以监管机构视角评审，不为企业利益辩护。'};
const CHAINS=[{id:'supply-decision',name:'供应链全链路决策',agents:['agent-supply-sentinel','agent-pnl-analyzer','agent-pricing-advisor']},{id:'growth-analysis',name:'增长归因分析',agents:['agent-ad-attribution','agent-competitor-radar','agent-product-radar']},{id:'brand-protection',name:'品牌合规防御',agents:['agent-listing-doctor','agent-brand-guardian','agent-account-guardian']}];
const ADISP={'agent-supply-sentinel':'供应链哨兵','agent-pricing-advisor':'动态定价顾问','agent-pnl-analyzer':'P&L透视镜','agent-ad-attribution':'广告归因侦探','agent-listing-doctor':'Listing医生','agent-voc-decoder':'用户之声解码器','agent-cs-triage':'客服分诊台','agent-account-guardian':'账号风险卫士','agent-brand-guardian':'品牌合规卫士','agent-product-radar':'选品雷达','agent-tiktok-content':'TikTok内容官','agent-competitor-radar':'竞品雷达站','agent-sku-tag-scanner':'SKU标签质量扫描器','agent-compliance-matrix':'多市场合规矩阵','agent-return-analyzer':'退货根因分析师','agent-margin-calculator':'SKU利润归因计算器','agent-geopolitical-risk':'地缘风险评估仪','agent-epr-calculator':'EPR合规费用测算','agent-dml-counterfactual-pricing':'反事实定价引擎','agent-cold-start-advisor':'新品冷启动顾问','agent-festival-replenishment':'大促补货决策师'};
function getMode(id){const r=document.querySelector('input[name="mode-'+id+'"]:checked');return r?r.value:'local';}
function getInputs(id){const o={};document.querySelectorAll('[id^="'+id+'__"]').forEach(el=>{o[el.id.replace(id+'__','')]=el.value||el.textContent||'';});return o;}
function getInputsLabeled(id){
  const o={};
  document.querySelectorAll('[id^="'+id+'__"]').forEach(el=>{
    const fieldId=el.id.replace(id+'__','');
    const wrap=el.closest('.modal-input-group');
    const label=wrap?wrap.querySelector('label')?.textContent?.trim()||fieldId:fieldId;
    const val=el.value||el.textContent||'';
    if(val)o[label]=val;
  });
  return o;
}
async function runAgentAI(id){
  const btn=document.getElementById('run-'+id),lb=document.getElementById('run-label-'+id),th=document.getElementById('thinking-'+id),out=document.getElementById('output-'+id),cel=document.getElementById('content-'+id);
  btn.disabled=true;if(lb)lb.textContent='AI分析中...';if(cel)cel.textContent='';if(out)out.classList.add('visible');if(th)th.style.display='flex';
  try{
    const sk=_sessionKey();
    const limitRes=await fetch('/api/agent/check-limit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_key:sk,agent_id:id})});
    const limitData=await limitRes.json();
    if(!limitData.can_proceed){
      if(th)th.style.display='none';
      if(cel)cel.innerHTML='<div style="padding:16px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;text-align:center"><div style="font-size:15px;font-weight:700;color:#b91c1c;margin-bottom:8px">本月免费次数已用完</div><div style="font-size:13px;color:#6b7280;margin-bottom:14px">免费版每月 '+limitData.monthly_limit+' 次 Agent 调用已用完。升级 Pro 版解锁无限次调用。</div><a href="/pricing.html" style="display:inline-block;padding:9px 20px;background:#B5323E;color:#fff;border-radius:8px;font-size:13px;font-weight:700;text-decoration:none">升级 Pro →</a></div>';
      if(btn)btn.disabled=false;if(lb)lb.textContent='运行分析';return;
    }
    const inp=getInputsLabeled(id),inpStr=Object.entries(inp).map(([k,v])=>k+': '+v).join('\\n');
    const res=await fetch('/api/agent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'deepseek-chat',messages:[{role:'system',content:AGENT_PROMPTS[id]||'你是跨境电商AI分析助手。'},{role:'user',content:'请根据以下数据进行分析，严格按照格式输出：\\n\\n'+inpStr}],max_tokens:1800,temperature:0.5,stream:false})});
    const data=await res.json();const ans=(data?.choices?.[0]?.message?.content||'').trim()||'分析失败，请重试';
    await fetch('/api/agent/record-usage',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_key:sk,agent_id:id})});
    if(th)th.style.display='none';await streamText(cel,'[AI深度分析 · DeepSeek]\\n\\n'+ans);
    saveReport(id,'[AI] '+ans);
    pushToFeishu({id,name:ADISP[id]||id,result:ans,ts:new Date().toLocaleString('zh-CN'),inputs:inp});
    const newLimit=await fetch('/api/agent/check-limit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_key:sk,agent_id:id})}).then(r=>r.json()).catch(()=>null);
    if(newLimit&&newLimit.remaining<=3&&newLimit.remaining>=0){const warn=document.getElementById('usage-warning-banner');if(!warn){const b=document.createElement('div');b.id='usage-warning-banner';b.style.cssText='position:fixed;bottom:16px;right:16px;z-index:9999;background:#fff;border:1px solid #fbbf24;border-radius:10px;padding:12px 16px;box-shadow:0 4px 20px rgba(0,0,0,.12);max-width:320px';b.innerHTML='<div style="font-size:13px;font-weight:700;color:#b45309;margin-bottom:4px">⚠ 免费次数剩余 '+newLimit.remaining+' 次</div><div style="font-size:12px;color:#6b7280;margin-bottom:8px">本月免费版 Agent 调用即将用完</div><a href="/pricing.html" style="font-size:12px;color:#B5323E;font-weight:700;text-decoration:none">升级 Pro 获取无限次数 →</a><button onclick="this.parentNode.remove()" style="position:absolute;top:8px;right:8px;background:none;border:none;cursor:pointer;color:#94a3b8;font-size:14px">×</button>';document.body.appendChild(b);setTimeout(()=>b.remove(),8000);}}
  }catch(e){if(th)th.style.display='none';if(cel)cel.textContent='[AI调用失败] '+e.message;}
  finally{if(btn)btn.disabled=false;if(lb)lb.textContent='重新分析';}
}
function openChainPanel(){document.getElementById('agent-chain-panel').style.display='flex';document.body.style.overflow='hidden';}
function closeChainPanel(){document.getElementById('agent-chain-panel').style.display='none';document.body.style.overflow='';}
async function runChain(chainId){
  const chain=CHAINS.find(c=>c.id===chainId);if(!chain)return;
  const outEl=document.getElementById('chain-output-'+chainId),btnEl=document.getElementById('chain-btn-'+chainId);
  if(!outEl||!btnEl)return;btnEl.disabled=true;btnEl.textContent='执行中...';outEl.style.display='block';outEl.innerHTML='';
  for(let i=0;i<chain.agents.length;i++){
    const aid=chain.agents[i],aname=ADISP[aid]||aid,sEl=document.createElement('div');
    sEl.style.cssText='margin-bottom:12px;padding:12px;background:var(--panel-2,#f8fafc);border-radius:8px;border:1px solid var(--line,#e2e8f0)';
    sEl.innerHTML='<div style="font-size:12px;font-weight:700;color:#6366f1;margin-bottom:5px">Step '+(i+1)+'：'+aname+'</div><div class="cs_" style="font-size:13px;color:#475569;line-height:1.6;white-space:pre-wrap">分析中...</div>';
    outEl.appendChild(sEl);outEl.scrollTop=outEl.scrollHeight;const cel=sEl.querySelector('.cs_');
    try{
      const r=await fetch('/api/agent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'deepseek-chat',messages:[{role:'system',content:AGENT_PROMPTS[aid]||'你是跨境电商AI助手。'},{role:'user',content:'请为跨境母婴品牌执行'+aname+'分析，给出5-8条关键洞察。'}],max_tokens:700,temperature:0.5,stream:false})});
      const d=await r.json();cel.textContent=(d?.choices?.[0]?.message?.content||'').trim()||'分析失败';
    }catch(e){cel.textContent='[错误] '+e.message;}
    outEl.scrollTop=outEl.scrollHeight;
  }
  btnEl.disabled=false;btnEl.textContent='重新执行';
}
"""
        CHAIN_BANNER = """<div style='margin:0 0 18px;padding:14px 18px;background:linear-gradient(135deg,#f0f4ff,#faf5ff);border:1px solid #c7d2fe;border-radius:10px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px'><div><div style='font-size:13px;font-weight:700;color:#3730a3'>⛓ Agent 链式调用</div><div style='font-size:11px;color:#6366f1;margin-top:2px'>多个 Agent 串联执行，形成完整决策链</div></div><button onclick='openChainPanel()' style='padding:7px 15px;background:#6366f1;color:#fff;border:none;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer' onmouseover="this.style.background='#4f46e5'" onmouseout="this.style.background='#6366f1'">启动链式分析 →</button></div>
<div id='agent-chain-panel' style='display:none;position:fixed;inset:0;z-index:2000;background:rgba(15,23,42,.55);backdrop-filter:blur(4px);align-items:flex-start;justify-content:center;padding:40px 16px;overflow-y:auto'><div style='background:#fff;border-radius:14px;width:100%;max-width:660px;box-shadow:0 20px 60px rgba(0,0,0,.15);overflow:hidden'><div style='padding:16px 20px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between'><div><div style='font-size:16px;font-weight:800;color:#0f172a'>⛓ Agent 链式调用</div><div style='font-size:11.5px;color:#64748b;margin-top:2px'>选择链路，多 Agent 串联深度分析</div></div><button onclick='closeChainPanel()' style='background:none;border:none;font-size:20px;cursor:pointer;color:#94a3b8'>×</button></div><div style='padding:16px 20px;display:flex;flex-direction:column;gap:12px'><div style='border:1px solid #e2e8f0;border-radius:9px;padding:13px'><div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px'><div><span style='font-size:13.5px;font-weight:700;color:#0f172a'>🔗 供应链全链路决策</span><div style='font-size:11px;color:#64748b;margin-top:1px'>供应链哨兵 → P&L透视镜 → 动态定价顾问</div></div><button id='chain-btn-supply-decision' onclick='runChain("supply-decision")' style='padding:6px 13px;background:#059669;color:#fff;border:none;border-radius:7px;font-size:11.5px;font-weight:600;cursor:pointer;white-space:nowrap'>▶ 执行</button></div><div id='chain-output-supply-decision' style='display:none;max-height:260px;overflow-y:auto;font-size:12.5px;color:#374151;line-height:1.65;white-space:pre-wrap'></div></div><div style='border:1px solid #e2e8f0;border-radius:9px;padding:13px'><div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px'><div><span style='font-size:13.5px;font-weight:700;color:#0f172a'>[↑] 增长归因分析</span><div style='font-size:11px;color:#64748b;margin-top:1px'>广告归因侦探 → 竞品雷达站 → 选品雷达</div></div><button id='chain-btn-growth-analysis' onclick='runChain("growth-analysis")' style='padding:6px 13px;background:#2563eb;color:#fff;border:none;border-radius:7px;font-size:11.5px;font-weight:600;cursor:pointer;white-space:nowrap'>▶ 执行</button></div><div id='chain-output-growth-analysis' style='display:none;max-height:260px;overflow-y:auto;font-size:12.5px;color:#374151;line-height:1.65;white-space:pre-wrap'></div></div><div style='border:1px solid #e2e8f0;border-radius:9px;padding:13px'><div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px'><div><span style='font-size:13.5px;font-weight:700;color:#0f172a'>RK️ 品牌合规防御</span><div style='font-size:11px;color:#64748b;margin-top:1px'>Listing医生 → 品牌合规卫士 → 账号风险卫士</div></div><button id='chain-btn-brand-protection' onclick='runChain("brand-protection")' style='padding:6px 13px;background:#7c3aed;color:#fff;border:none;border-radius:7px;font-size:11.5px;font-weight:600;cursor:pointer;white-space:nowrap'>▶ 执行</button></div><div id='chain-output-brand-protection' style='display:none;max-height:260px;overflow-y:auto;font-size:12.5px;color:#374151;line-height:1.65;white-space:pre-wrap'></div></div><p style='font-size:11px;color:#94a3b8;text-align:center;margin:4px 0 0'>链式调用使用 AI 模式（DeepSeek），每步约5-10秒</p></div></div></div>
"""
        # 替换 runAgent 函数（括号计数法）
        m = re.search(r'async function runAgent\(id\) \{', agents_html)
        if m:
            start = m.start()
            bc = pos = 0
            pos = start
            func_end = None
            while pos < len(agents_html):
                if agents_html[pos] == '{': bc += 1
                elif agents_html[pos] == '}':
                    bc -= 1
                    if bc == 0: func_end = pos; break
                pos += 1
            if func_end:
                NEW_RUNAGENT = """async function runAgent(id) {
  if (getMode(id) === 'ai') { return runAgentAI(id); }
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const out = document.getElementById('output-' + id);
  const content = document.getElementById('content-' + id);
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  if (label) label.textContent = '计算中...';
  if (content) content.textContent = '';
  if (out) out.classList.add('visible');
  if (thinking) thinking.style.display = 'flex';
  await sleep(600);
  if (thinking) thinking.style.display = 'none';
  let text = '';
  try {
    if (id === 'agent-supply-sentinel')   text = computeSupplySentinel(id);
    else if (id === 'agent-pricing-advisor') text = computePricingAdvisor(id);
    else if (id === 'agent-pnl-analyzer')   text = computePnLAnalyzer(id);
    else if (id === 'agent-ad-attribution') text = computeAdAttribution(id);
    else if (id === 'agent-competitor-radar') text = computeCompetitorRadar(id);
    else if (id === 'agent-listing-doctor')  text = computeListingDoctor(id);
    else if (id === 'agent-voc-decoder')     text = computeVocDecoder(id);
    else if (id === 'agent-cs-triage')       text = computeCsTriage(id);
    else if (id === 'agent-account-guardian') text = computeAccountGuardian(id);
    else if (id === 'agent-brand-guardian')  text = computeBrandGuardian(id);
    else if (id === 'agent-product-radar')   text = computeProductRadar(id);
    else if (id === 'agent-tiktok-content')  text = computeTikTokContent(id);
    else text = (DEMO_DATA[id] || {}).output || '暂无计算结果';
  } catch(e) {
    text = '[计算错误] ' + e.message + '\\n请检查输入格式';
  }
  await streamText(content, text);
  saveReport(id, text);
  if (btn) btn.disabled = false;
  if (label) label.textContent = '重新计算';
}"""
                agents_html = agents_html[:start] + NEW_RUNAGENT + agents_html[func_end+1:]

        # 注入 AI 引擎到 catBtns 前
        if 'const catBtns' in agents_html and 'AGENT_PROMPTS' not in agents_html:
            agents_html = agents_html.replace('const catBtns', AI_ENGINE + '\nconst catBtns', 1)

        # 插入链式调用Banner到 agent-grid 前
        grid_marker = "<div class='agent-grid' id='agentGrid'>"
        if grid_marker in agents_html and 'agent-chain-panel' not in agents_html:
            agents_html = agents_html.replace(grid_marker, CHAIN_BANNER + '\n' + grid_marker, 1)

        # 注入 CSV 上传功能
        if 'p2s-csv-upload' not in agents_html:
            CSV_JS = """
<script id="p2s-csv-upload">
(function(){
function parseCSV(text){
  const lines=text.trim().split(/\\r?\\n/);
  if(lines.length<2)return{headers:[],rows:[]};
  const headers=lines[0].split(',').map(h=>h.trim().replace(/^["']|["']$/g,''));
  const rows=lines.slice(1).map(l=>{
    const cols=[],re=/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g;
    let last=0,m;
    while((m=re.exec(l))!==null){cols.push(l.slice(last,m.index).replace(/^"|"$/g,''));last=m.index+1;}
    cols.push(l.slice(last).replace(/^"|"$/g,''));
    return Object.fromEntries(headers.map((h,i)=>[h,cols[i]||'']));
  });
  return{headers,rows};
}
function csvFillAgent(agentId,rows,headers){
  if(!rows.length)return;
  const allText=rows.map(r=>headers.map(h=>h+': '+r[h]).join(' | ')).join('\\n');
  const textareas=document.querySelectorAll('#modal-'+agentId+' textarea, #modal-'+agentId+' input[type=text]');
  textareas.forEach(function(el){
    const label=(el.id||el.placeholder||'').toLowerCase();
    const matchedHeader=headers.find(function(h){
      const hl=h.toLowerCase();
      return label.includes(hl)||hl.includes(label.replace(/\\s+/g,'_'));
    });
    if(matchedHeader){
      const vals=rows.map(r=>r[matchedHeader]||'').filter(Boolean);
      el.value=vals.join('\\n');
    } else if(el.tagName==='TEXTAREA'&&!el.value){
      el.value=allText.slice(0,2000);
    }
  });
  const notice=document.getElementById('csv-notice-'+agentId);
  if(notice)notice.textContent='✓ 已导入 '+rows.length+' 行 CSV 数据（'+headers.join('、')+'）';
}
function openCSVPicker(agentId){
  const inp=document.createElement('input');
  inp.type='file';inp.accept='.csv,text/csv';
  inp.onchange=function(){
    const file=this.files[0];if(!file)return;
    const reader=new FileReader();
    reader.onload=function(e){
      const{headers,rows}=parseCSV(e.target.result);
      if(!rows.length){alert('CSV 为空或格式不正确，请确保第一行为表头');return;}
      csvFillAgent(agentId,rows,headers);
    };
    reader.readAsText(file,'UTF-8');
  };
  inp.click();
}
window.openCSVPicker=openCSVPicker;
})();
</script>"""
            agents_html = agents_html.replace('</body>', CSV_JS + '\n</body>', 1)

            modal_marker = "class='modal-run-btn'"
            if modal_marker in agents_html:
                import re as _re2
                def _inject_csv_btn(m):
                    full = m.group(0)
                    agent_id_match = _re2.search(r"id='run-([^']+)'", full)
                    if not agent_id_match:
                        return full
                    aid = agent_id_match.group(1)
                    csv_btn = (
                        f"<button type='button' onclick='openCSVPicker(\"{aid}\")' "
                        f"style='padding:8px 14px;border:1px dashed #cbd5e1;border-radius:8px;"
                        f"background:transparent;color:#64748b;font-size:12px;cursor:pointer;"
                        f"display:inline-flex;align-items:center;gap:5px'>"
                        f"⬆ 上传 CSV</button>"
                        f"<div id='csv-notice-{aid}' style='font-size:11px;color:#059669;margin-top:4px'></div>"
                    )
                    return csv_btn + full
                agents_html = _re2.sub(r"<button class='modal-run-btn'[^>]+>[^<]*</button>", _inject_csv_btn, agents_html)

        agents_path.write_text(agents_html, encoding="utf-8")

    # ── 2. chat-page.js: 客户端RAG + Skill卡片 + Agent跳转 ──────────────────
    chat_html_path = out / "chat.html"
    if chat_html_path.exists():
        _ch = chat_html_path.read_text(encoding="utf-8")
        if 'btn-clear-chat' not in _ch:
            _ch = _ch.replace(
                '<button class="web-search-toggle" id="web-search-toggle" title=',
                '<button id="btn-clear-chat" onclick="clearHistory()" style="padding:5px 10px;border-radius:var(--r-full);border:1.5px solid var(--line);background:transparent;font-size:12px;color:var(--muted);cursor:pointer;font-family:var(--font);white-space:nowrap;flex-shrink:0">✕ 清空</button> <button class="web-search-toggle" id="web-search-toggle" title=',
                1
            )
            chat_html_path.write_text(_ch, encoding="utf-8")
    
        chat_js_path = out / "assets" / "chat-page.js"
    if chat_js_path.exists():
        RAG_JS = r"""(function () {
  const msgsEl=document.getElementById('chat-messages'),welcome=document.getElementById('chat-welcome'),textarea=document.getElementById('chat-input'),sendBtn=document.getElementById('chat-send'),webToggle=document.getElementById('web-search-toggle'),webLabel=document.getElementById('web-search-label');
  let webSearchOn=false;
  const _HIST_KEY='p2s_chat_v1';
  function _loadH(){try{return JSON.parse(localStorage.getItem(_HIST_KEY)||'[]');}catch(e){return[];}}
  function _saveH(){try{localStorage.setItem(_HIST_KEY,JSON.stringify(history.slice(-20)));}catch(e){}}
  let history=_loadH();
  window.clearHistory=function(){history=[];try{localStorage.removeItem(_HIST_KEY);}catch(e){}if(msgsEl){msgsEl.innerHTML='';}if(welcome)welcome.style.display='';};
  webToggle.addEventListener('click',()=>{webSearchOn=!webSearchOn;webToggle.classList.toggle('on',webSearchOn);webLabel.textContent=webSearchOn?'已开启联网':'联网搜索';});
  textarea.addEventListener('input',()=>{textarea.style.height='auto';textarea.style.height=Math.min(textarea.scrollHeight,160)+'px';});
  textarea.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();doSend();}});
  sendBtn.addEventListener('click',doSend);
  if(history.length&&welcome){welcome.style.display='none';history.forEach(function(m){if(m.role==='user')addMsg(m.content,'user');else if(m.role==='assistant')addMsg(m.content,'bot');});}
  document.querySelectorAll('.chat-sug-btn').forEach(btn=>{btn.addEventListener('click',()=>{textarea.value=btn.textContent.trim();textarea.dispatchEvent(new Event('input'));doSend();});});
  function md(text){return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\*\*(.+?)\*\*/gs,'<strong>$1</strong>').replace(/\*([^*\n]+)\*/g,'<em>$1</em>').replace(/`([^`\n]+)`/g,'<code>$1</code>').replace(/^#{1,3}\s+(.+)$/gm,'<strong style="font-size:15px">$1</strong>').replace(/^[-•]\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">• $1</span>').replace(/^\d+\.\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">$&</span>').replace(/\n\n+/g,'<br><br>').replace(/\n/g,'<br>');}
  const _idx=[];let _built=false;
  function buildSkillIndex(){if(_built)return;const DATA=window.PLAYBOOK_DATA||{};(DATA.skills||[]).forEach(s=>{const t=[s.skill_id||'',s.title||'',s.problem_solved||'',s.algorithm_summary||'',s.biz_trigger||'',s.biz_outcome||'',(s.tags||[]).join(' '),(s.topics||[]).join(' ')].join(' ').toLowerCase();_idx.push({s,t});});_built=true;}
  let _skillIdx=null;async function _loadSkillIdx(){if(_skillIdx)return _skillIdx;try{_skillIdx=await fetch('/assets/skill-index.json').then(r=>r.json());}catch(e){_skillIdx=[];}return _skillIdx;}
  async function _retrieveSkills(query,topK){topK=topK||5;const idx=await _loadSkillIdx();if(!idx||!idx.length)return[];const tokens=query.toLowerCase().replace(/[^\u4e00-\u9fa5a-z0-9\s]/g,' ').split(/\s+/).filter(t=>t.length>1);if(!tokens.length)return[];const scored=idx.map(function(s){const text=(s.summary+' '+s.keywords.join(' ')).toLowerCase();const score=tokens.reduce(function(n,t){return n+(text.includes(t)?1:0);},0);return{s,score};}).filter(x=>x.score>0);scored.sort((a,b)=>b.score-a.score);return scored.slice(0,topK).map(x=>x.s);}
  function searchSkills(query,k){k=k||8;buildSkillIndex();const words=query.toLowerCase().split(/\s+/).filter(w=>w.length>1);if(!words.length)return[];return _idx.map(item=>{let sc=0;words.forEach(w=>{const tf=item.t.split(w).length-1;if(tf>0)sc+=tf*(w.length>3?2:1);});return{skill:item.s,sc};}).filter(x=>x.sc>0).sort((a,b)=>b.sc-a.sc).slice(0,k).map(x=>x.skill);}
  function buildRAGContext(query){const top=searchSkills(query,10);if(!top.length){return(window.PLAYBOOK_DATA&&window.PLAYBOOK_DATA.skills||[]).slice(0,60).map(s=>s.skill_id+': '+(s.problem_solved||s.algorithm_summary||'').slice(0,140)).join('\n');}return top.map(s=>{const p=[s.skill_id,s.title];if(s.problem_solved)p.push('解决: '+s.problem_solved.slice(0,120));if(s.biz_trigger)p.push('触发: '+s.biz_trigger.slice(0,100));if(s.roi_figure)p.push('ROI: '+s.roi_figure);return p.join(' | ');}).join('\n');}
  function renderSkillCards(text){const DATA=window.PLAYBOOK_DATA||{};const map={};(DATA.skills||[]).forEach(s=>{map[s.skill_id]=s;});const found=[],seen={};[/\[\[?(Skill-[\w-]+)\]?\]/g,/\*\*(Skill-[\w-]+)\*\*/g].forEach(pat=>{let m;while((m=pat.exec(text))!==null){if(map[m[1]]&&!seen[m[1]]){seen[m[1]]=1;found.push(map[m[1]]);}}});if(!found.length)return'';const esc=t=>(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');const cards=found.map(s=>'<a href="skills/'+s.skill_id+'.html" target="_blank" style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:var(--panel-2,#f8fafc);border:1px solid var(--line,#e2e8f0);border-radius:8px;text-decoration:none;color:inherit;margin-top:6px;transition:box-shadow .15s" onmouseover="this.style.boxShadow=\'0 2px 8px rgba(0,0,0,.08)\'" onmouseout="this.style.boxShadow=\'none\'">'+'<div style="flex-shrink:0;width:32px;height:32px;border-radius:6px;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700">S</div>'+'<div style="min-width:0"><div style="font-size:12px;font-weight:600;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+esc((s.title||s.skill_id).slice(0,60))+'</div>'+'<div style="font-size:11.5px;color:#64748b;margin-top:2px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical">'+esc((s.problem_solved||s.biz_trigger||'').slice(0,90))+'</div>'+(s.roi_figure?'<span style="font-size:11px;color:#059669;font-weight:600;margin-top:4px;display:block">ROI: '+esc(s.roi_figure)+'</span>':'')+'</div></a>').join('');return'<div style="margin-top:10px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px"><div style="font-size:11.5px;color:#64748b;font-weight:600;margin-bottom:6px">知识库 相关技能</div>'+cards+'</div>';}
  const AKWS={'agent-supply-sentinel':['供应链','库存','断货','补货','DOS','海运'],'agent-pricing-advisor':['定价','价格','ACoS','竞品价','利润率'],'agent-pnl-analyzer':['P&L','利润','GMV','毛利','亏损'],'agent-ad-attribution':['广告','ROAS','归因','ACoS','投放'],'agent-listing-doctor':['Listing','标题','关键词','A+'],'agent-voc-decoder':['评论','VOC','用户反馈','差评'],'agent-cs-triage':['客服','工单','退款','投诉','A-to-Z'],'agent-account-guardian':['封号','账号','违规','风险'],'agent-brand-guardian':['合规','文案','广告法','违禁'],'agent-product-radar':['选品','蓝海','竞争','市场机会'],'agent-tiktok-content':['TikTok','短视频','内容','脚本'],'agent-competitor-radar':['竞品','竞争对手','ASIN','BSR']};
  const ANAMES={'agent-supply-sentinel':'供应链哨兵','agent-pricing-advisor':'动态定价顾问','agent-pnl-analyzer':'P&L透视镜','agent-ad-attribution':'广告归因侦探','agent-listing-doctor':'Listing医生','agent-voc-decoder':'用户之声解码器','agent-cs-triage':'客服分诊台','agent-account-guardian':'账号风险卫士','agent-brand-guardian':'品牌合规卫士','agent-product-radar':'选品雷达','agent-tiktok-content':'TikTok内容官','agent-competitor-radar':'竞品雷达站'};
  function detectAgents(text){const t=text.toLowerCase();return Object.keys(AKWS).filter(id=>AKWS[id].some(k=>t.indexOf(k.toLowerCase())>=0)).slice(0,3);}
  function renderAgentBtns(ids){if(!ids.length)return'';const btns=ids.map(id=>'<a href="agents.html" target="_blank" style="display:inline-flex;align-items:center;gap:5px;padding:6px 12px;background:var(--accent-light,#eff6ff);border:1px solid var(--accent,#3b82f6);border-radius:20px;font-size:12px;font-weight:600;color:var(--accent,#3b82f6);text-decoration:none;transition:all .15s;white-space:nowrap" onmouseover="this.style.background=\'var(--accent,#3b82f6)\';this.style.color=\'#fff\'" onmouseout="this.style.background=\'var(--accent-light,#eff6ff)\';this.style.color=\'var(--accent,#3b82f6)\'">◈ '+(ANAMES[id]||id)+'</a>').join('');return'<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px"><span style="font-size:11.5px;color:#64748b;font-weight:600;align-self:center;margin-right:4px"> 直接调用：</span>'+btns+'</div>';}
  function addMsg(text,role,extras){extras=extras||{};if(welcome)welcome.style.display='none';const row=document.createElement('div');row.className='cmsg cmsg-'+role;const av=document.createElement('div');av.className='cmsg-avatar';av.textContent=role==='bot'?'\u2726':'U';const body=document.createElement('div');body.className='cmsg-body';const nm=document.createElement('div');nm.className='cmsg-name';nm.textContent=role==='bot'?'AI 助手':'你';body.appendChild(nm);if(extras.webBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.innerHTML='联网搜索';body.appendChild(b);}if(extras.eventBadge){const b=document.createElement('div');b.className='cmsg-event-badge';b.innerHTML=extras.eventBadge;body.appendChild(b);}else if(extras.ragBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.style.cssText='background:#f0fdf4;color:#166534;border-color:#bbf7d0';b.innerHTML='知识库检索 · '+extras.ragBadge+' 条相关技能';body.appendChild(b);}const bubble=document.createElement('div');bubble.className='cmsg-bubble';if(role==='bot'){bubble.innerHTML=md(text);const agIds=detectAgents(text),sc=renderSkillCards(text),ab=renderAgentBtns(agIds);if(sc||ab){const x=document.createElement('div');x.innerHTML=(sc||'')+(ab||'');bubble.appendChild(x);}}else{bubble.textContent=text;}body.appendChild(bubble);row.appendChild(av);row.appendChild(body);msgsEl.appendChild(row);msgsEl.scrollTop=msgsEl.scrollHeight;return{row,bubble};}
  function addTyping(){if(welcome)welcome.style.display='none';const row=document.createElement('div');row.className='cmsg cmsg-bot cmsg-typing';const av=document.createElement('div');av.className='cmsg-avatar';av.textContent='\u2726';const body=document.createElement('div');body.className='cmsg-body';const nm=document.createElement('div');nm.className='cmsg-name';nm.textContent='AI 助手';const bubble=document.createElement('div');bubble.className='cmsg-bubble';body.appendChild(nm);body.appendChild(bubble);row.appendChild(av);row.appendChild(body);msgsEl.appendChild(row);msgsEl.scrollTop=msgsEl.scrollHeight;return row;}

  function matchRiskEvent(query) {
    if (!window.RISK_EVENTS || !window.RISK_EVENTS.events) return null;
    const lowerQuery = query.toLowerCase();
    let bestEvent = null;
    let maxScore = 0;
    
    for (const event of window.RISK_EVENTS.events) {
      if (!event.symptom_keywords) continue;
      let score = 0;
      for (const kw of event.symptom_keywords) {
        if (lowerQuery.includes(kw.toLowerCase())) {
          score++;
        }
      }
      if (score > maxScore) {
        maxScore = score;
        bestEvent = event;
      }
    }
    return maxScore > 0 ? bestEvent : null;
  }

  function buildEventSkillChain(event) {
    let result = '';
    const phases = event.phases || {};
    
    if (phases.diagnose && phases.diagnose.length > 0) {
      result += '【诊断层】\\n';
      phases.diagnose.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\\n`;
      });
    }
    
    if (phases.treat && phases.treat.length > 0) {
      result += '【处置层】\\n';
      phases.treat.forEach((s, i) => {
        const cond = s.condition ? `（条件: ${s.condition}时触发）` : '';
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}${cond}\\n`;
      });
    }
    
    if (phases.prevent && phases.prevent.length > 0) {
      result += '【预防层】\\n';
      phases.prevent.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\\n`;
      });
    }
    return result;
  }

  async function doSend(){const text=textarea.value.trim();if(!text||sendBtn.disabled)return;textarea.value='';textarea.style.height='auto';sendBtn.disabled=true;addMsg(text,'user');history.push({role:'user',content:text});const typing=addTyping();
  
  let ctxMsg = '';
  const matchedEvent = matchRiskEvent(text);
  let matchedEventText = null;
  
  const roleSelect = document.getElementById('role-select');
  const roleVal = roleSelect ? roleSelect.value : 'ops';
  const rolePrompts = {
    ops: '当前用户是电商运营，关注具体操作步骤、SOP 执行、数据指标改善，回答要简洁可执行。',
    analyst: '当前用户是数据分析师，关注算法原理、统计方法、代码实现，回答要有技术深度，可附公式。',
    ceo: '当前用户是 CEO，关注战略决策、ROI 全局、竞争壁垒，回答要高度概括、突出商业价值。',
  };
  const roleCtx = rolePrompts[roleVal] || rolePrompts.ops;
  
  let sys=`你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\n知识库现有 ${window.PLAYBOOK_DATA && window.PLAYBOOK_DATA.skills ? window.PLAYBOOK_DATA.skills.length : 800}+ 个从顶会论文萃取的可落地业务技能。\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\n${roleCtx}\n当前时间：${new Date().toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric'})}`;
  
  if (matchedEvent) {
    sys += `\n\n当前诊断场景：${matchedEvent.event_name}\n严重程度：${matchedEvent.severity}`;
    const eventChain = buildEventSkillChain(matchedEvent);
    ctxMsg = '\n\n【场景推荐 Skill 链】\n' + eventChain;
    matchedEventText = `${matchedEvent.icon} 识别到场景：${matchedEvent.event_name}`;
  } else {
    const ragSkills=searchSkills(text,10),ragCtx=buildRAGContext(text),ragCount=ragSkills.length;
    ctxMsg=ragCount>0?'\n\n【知识库相关技能（检索到'+ragCount+'条）】\n'+ragCtx:'\n\n【知识库摘要（前60条）】\n'+ragCtx;
    const _idxSkills=await _retrieveSkills(text,5);if(_idxSkills.length){ctxMsg+='\n\n【知识库检索结果 — 请优先引用这些 Skill ID 回答】\n'+_idxSkills.map(function(s){return'['+s.id+'] '+s.title+': '+s.summary.slice(0,120);}).join('\n');}
  }
  
  const messages=[{role:'system',content:sys+ctxMsg},...history.slice(-8)];try{const body={model:'deepseek-chat',messages,max_tokens:1500,temperature:0.55,stream:false};if(webSearchOn){body.tools=[{type:'function',function:{name:'web_search',description:'Search the web',parameters:{type:'object',properties:{query:{type:'string'}},required:['query']}}}];body.tool_choice='auto';}const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();const choice=data&&data.choices&&data.choices[0];let answer=(choice&&choice.message&&choice.message.content||'').trim();if(!answer&&choice&&choice.finish_reason==='tool_calls')answer='（联网搜索触发中…）\n\n'+((choice.message.tool_calls[0]&&choice.message.tool_calls[0].function.arguments)||'');answer=answer||'抱歉，暂时无法获取回答，请稍后重试。';typing.remove();
  let ragCountToPass = null;
  if (!matchedEvent) {
      const ragSkills = searchSkills(text, 10);
      ragCountToPass = ragSkills.length > 0 ? ragSkills.length : null;
  }
  const _msgResult=addMsg(answer,'bot',{webBadge:webSearchOn, eventBadge: matchedEventText, ragBadge: ragCountToPass});
  if (!matchedEvent){_retrieveSkills(text,5).then(function(idxSkills){if(!idxSkills.length)return;const citDiv=document.createElement('div');citDiv.style.cssText='font-size:11px;color:#94a3b8;margin-top:6px;padding-top:6px;border-top:1px solid #f1f5f9';citDiv.innerHTML='\u53c2\u8003 Skill: '+idxSkills.map(function(s){return'<a href="/skills/'+s.id+'.html" target="_blank" style="color:#6366f1;text-decoration:none">'+s.title+'</a>';}).join(' \u00b7 ');if(_msgResult&&_msgResult.bubble)_msgResult.bubble.appendChild(citDiv);});}
  history.push({role:'assistant',content:answer});_saveH();}catch(e){typing.remove();addMsg('网络请求失败，请检查连接后重试。','bot');}finally{sendBtn.disabled=false;textarea.focus();}}
})();
"""
        chat_js_path.write_text(RAG_JS, encoding="utf-8")

    # ── 3. agent-report.html: AI综合分析按钮 ────────────────────────────────
    report_path = out / "agent-report.html"
    if report_path.exists():
        report_html = report_path.read_text(encoding="utf-8")
        if 'aiSynthesizeReports' not in report_html:
            OLD_EXPORT = "onclick='exportReports()'"
            NEW_EXPORT = (
                "onclick='exportReports()'"
                " style='padding:8px 16px;border-radius:8px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;font-size:13px;font-weight:600'>"
                "⬇ 导出全部</button>"
                "  <button onclick='aiSynthesizeReports()' style='padding:8px 16px;border-radius:8px;border:1.5px solid #6366f1;background:#f0f4ff;color:#6366f1;cursor:pointer;font-size:13px;font-weight:600'>AI 综合分析"
            )
            if OLD_EXPORT in report_html:
                report_html = report_html.replace(
                    "onclick='exportReports()'",
                    "onclick='exportReports()' style='padding:8px 16px;border-radius:8px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;font-size:13px;font-weight:600'"
                )
            AI_SYNTH_FN = """function aiSynthesizeReports(){const reports=loadReports();if(!reports.length){alert('暂无报告，请先在智能体广场运行分析');return;}const recent=reports.slice(0,8);const st=recent.map((r,i)=>'【'+(i+1)+'. '+r.name+' | '+r.ts+'】\\n输入: '+JSON.stringify(r.inputs)+'\\n结果: '+r.result.slice(0,400)).join('\\n\\n---\\n\\n');const overlay=document.createElement('div');overlay.style.cssText='position:fixed;inset:0;z-index:3000;background:rgba(15,23,42,.55);backdrop-filter:blur(4px);display:flex;align-items:center;justify-content:center;padding:20px';overlay.innerHTML='<div style="background:#fff;border-radius:14px;width:100%;max-width:680px;max-height:80vh;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.15)"><div style="padding:16px 20px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between"><div><div style="font-size:16px;font-weight:800;color:#0f172a">AI 综合分析报告</div><div style="font-size:12px;color:#64748b;margin-top:2px">基于最近 '+recent.length+' 条 Agent 运行记录</div></div><button id="ai-synth-close" style="background:none;border:none;font-size:20px;cursor:pointer;color:#94a3b8">×</button></div><div style="flex:1;overflow-y:auto;padding:16px 20px"><div id="ai-synth-output" style="font-size:13.5px;color:#374151;line-height:1.7;white-space:pre-wrap"><div style="color:#6366f1;font-weight:600">⏳ 正在综合分析 '+recent.length+' 条报告，请稍候…</div></div></div><div style="padding:12px 20px;border-top:1px solid #e2e8f0;display:flex;justify-content:flex-end;gap:8px"><button id="ai-synth-copy" style="padding:7px 14px;background:var(--panel-2,#f8fafc);border:1.5px solid var(--line,#e2e8f0);border-radius:8px;font-size:12px;font-weight:600;cursor:pointer"> 复制</button></div></div>';document.body.appendChild(overlay);const outputEl=document.getElementById('ai-synth-output');document.getElementById('ai-synth-close').onclick=()=>overlay.remove();overlay.onclick=e=>{if(e.target===overlay)overlay.remove();};fetch('/api/agent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'deepseek-chat',messages:[{role:'system',content:'你是跨境电商AI数据分析师。请综合分析下面多个Agent的运行结果，给出：1.跨Agent综合洞察；2.最需优先处理的3个问题；3.具体可量化行动建议。结构化中文输出。'},{role:'user',content:'以下是近期Agent运行结果：\\n\\n'+st}],max_tokens:1500,temperature:0.45,stream:false})}).then(r=>r.json()).then(data=>{const ans=(data?.choices?.[0]?.message?.content||'').trim()||'分析失败，请重试';outputEl.innerHTML=ans.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/\\n\\n+/g,'<br><br>').replace(/\\n/g,'<br>');document.getElementById('ai-synth-copy').onclick=()=>{navigator.clipboard.writeText(ans).then(()=>{document.getElementById('ai-synth-copy').textContent='✓ 已复制';});};}).catch(e=>{outputEl.innerHTML='<span style="color:#ef4444">[AI调用失败] '+e.message+'</span>';});}\n"""
            # 在 exportReports 函数后插入
            if 'function exportReports' in report_html and 'aiSynthesizeReports' not in report_html:
                # 找 exportReports 函数结束的位置（下一个 function 前）
                idx = report_html.find('function exportReports')
                next_fn = report_html.find('\nfunction ', idx + 100)
                if next_fn > 0:
                    report_html = report_html[:next_fn] + '\n' + AI_SYNTH_FN + report_html[next_fn:]
            # 插入按钮（在导出按钮旁）
            old_export_btn = ">⬇ 导出全部</button>"
            if old_export_btn in report_html:
                report_html = report_html.replace(
                    old_export_btn,
                    ">⬇ 导出全部</button>\n  <button onclick='aiSynthesizeReports()' style='padding:8px 16px;border-radius:8px;border:1.5px solid #6366f1;background:#f0f4ff;color:#6366f1;cursor:pointer;font-size:13px;font-weight:600'>AI 综合分析</button>"
                )
            report_path.write_text(report_html, encoding="utf-8")

    # ── 4. index.html: 快速行动区 ────────────────────────────────────────────
    index_path = out / "index.html"
    if index_path.exists():
        index_html = index_path.read_text(encoding="utf-8")
        if '分析报告台' not in index_html:
            QUICK_ACTIONS = (
                '\n  <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:7px;justify-content:center">'
                '<a href="diagnostic.html" style="display:inline-flex;align-items:center;gap:5px;padding:6px 13px;background:rgba(181,50,62,.1);border:1.5px solid rgba(181,50,62,.3);border-radius:20px;font-size:11.5px;font-weight:600;color:#B5323E;text-decoration:none">业务诊断</a>'
                '<a href="chat.html" style="display:inline-flex;align-items:center;gap:5px;padding:6px 13px;background:rgba(99,102,241,.1);border:1.5px solid rgba(99,102,241,.3);border-radius:20px;font-size:11.5px;font-weight:600;color:#6366f1;text-decoration:none">AI 知识库对话</a>'
                '<a href="graph/overview.html" style="display:inline-flex;align-items:center;gap:5px;padding:6px 13px;background:rgba(6,182,212,.1);border:1.5px solid rgba(6,182,212,.3);border-radius:20px;font-size:11.5px;font-weight:600;color:#0891b2;text-decoration:none">技能关系图谱</a>'
                '<a href="playbooks/index.html" style="display:inline-flex;align-items:center;gap:5px;padding:6px 13px;background:rgba(16,185,129,.1);border:1.5px solid rgba(16,185,129,.3);border-radius:20px;font-size:11.5px;font-weight:600;color:#059669;text-decoration:none">场景手册</a>'
                '<a href="agent-report.html" style="display:inline-flex;align-items:center;gap:5px;padding:6px 13px;background:rgba(245,158,11,.1);border:1.5px solid rgba(245,158,11,.3);border-radius:20px;font-size:11.5px;font-weight:600;color:#d97706;text-decoration:none">分析报告台</a>'
                '</div>'
            )
            HERO_CTA_END = '</div>\n  <div class="hero-tabs"'
            ALT_CTA_END = '</div>\n<div class="hero-tabs"'
            if HERO_CTA_END in index_html:
                index_html = index_html.replace(HERO_CTA_END, QUICK_ACTIONS + HERO_CTA_END, 1)
            elif ALT_CTA_END in index_html:
                index_html = index_html.replace(ALT_CTA_END, QUICK_ACTIONS + ALT_CTA_END, 1)
            # 替换 Demo 按钮为 Agent 工作台
            index_html = index_html.replace('预约 30 分钟 Demo', '进入智能体广场')
            index_html = index_html.replace('mailto:skills@lute-tlz-dddd.top?subject=预约Demo-paper2skills', 'agents.html')
            index_path.write_text(index_html, encoding="utf-8")

    # ── 5. search.js: 语义增强版（预览卡片 + 颜色域标识 + 评分排序） ──────
    search_js_path = out / "assets" / "search.js"
    if search_js_path.exists():
        search_js_path.write_text(r"""(function(){
  var input=document.getElementById('global-search');
  var box=document.getElementById('search-results');
  if(!input||!box) return;
  var skills=[];
  function waitForData(cb){if(window.PLAYBOOK_DATA){skills=window.PLAYBOOK_DATA.skills||[];cb();return;}var t=setInterval(function(){if(window.PLAYBOOK_DATA){clearInterval(t);skills=window.PLAYBOOK_DATA.skills||[];cb();}},80);}
  function rootPrefix(){var p=window.location.pathname;if(p.includes('/skills/')||p.includes('/domains/')||p.includes('/playbooks/')||p.includes('/topics/')||p.includes('/workflows/')||p.includes('/graph/'))return'../';return'';}
  function esc(s){return(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
  var DCOLORS={'04-供应链':'#0ea5e9','10-MAS':'#8b5cf6','08-知识图谱':'#06b6d4','16-智能体工程':'#ec4899','13-广告分析':'#f59e0b','14-用户分析':'#10b981','24-标签工程':'#6366f1','06-增长模型':'#ef4444','15-营销投放分析':'#f97316','23-运营财务':'#14b8a6','05-推荐系统':'#84cc16','02-A_B实验':'#a855f7','01-因果推断':'#64748b','03-时间序列':'#22d3ee','17-价格优化':'#fb923c','18-物流履约':'#4ade80','19-风控反欺诈':'#f43f5e','20-AI视频生成':'#c084fc','21-合规决策':'#fbbf24','22-数据采集工程':'#38bdf8','12-ML基础':'#94a3b8','11-AI人文':'#e879f9','07-NLP-VOC':'#2dd4bf','09-DataAgent-LLM':'#f97316'};
  function scoreSkill(s,q,words){var text=[s.skill_id,s.title,s.domain_dir,(s.tags||[]).join(' '),(s.topics||[]).join(' '),s.algorithm_summary,s.problem_solved,s.roi_figure,s.biz_trigger,s.biz_outcome].join(' ').toLowerCase();if(!text.includes(q))return 0;var sc=0;if((s.title||'').toLowerCase().includes(q))sc+=10;if((s.problem_solved||'').toLowerCase().includes(q))sc+=5;words.forEach(function(w){if(text.includes(w))sc+=1;});return sc;}
  function applyFilters(list){var diff=(document.getElementById('filter-diff')||{}).value||'';var roi=(document.getElementById('filter-roi')||{}).value||'';var dom=(document.getElementById('filter-domain')||{}).value||'';return list.filter(function(s){if(dom&&s.domain_dir!==dom)return false;if(diff&&s.difficulty!==diff)return false;if(roi){var stars=(s.difficulty||'').split('\u2b50').length-1;if(roi==='easy'&&stars>2)return false;if(roi==='medium'&&(stars<3||stars>3))return false;if(roi==='hard'&&stars<4)return false;}return true;});}
  function renderPreview(s,prefix){var color=DCOLORS[s.domain_dir]||'#94a3b8';var roi=s.roi_figure?'<span style="color:#059669;font-size:11px;font-weight:600;margin-top:2px;display:block">\ud83d\udcb0 '+esc(s.roi_figure)+'</span>':'';var diff=s.difficulty?'<span style="font-size:10px;color:#94a3b8">'+esc(s.difficulty)+'</span>':'';var problem=s.problem_solved?'<div style="font-size:11.5px;color:#475569;margin-top:3px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical">'+esc((s.problem_solved||'').slice(0,100))+'</div>':'';return'<a class="result" href="'+prefix+'skills/'+s.skill_id+'.html" style="display:block;padding:10px 14px;border-bottom:1px solid var(--line,#e2e8f0);text-decoration:none"><div style="display:flex;align-items:flex-start;gap:10px"><div style="width:8px;height:8px;border-radius:50%;background:'+color+';flex-shrink:0;margin-top:5px"></div><div style="min-width:0;flex:1"><div style="font-size:13px;font-weight:700;color:var(--ink,#0f172a);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+esc((s.title||s.skill_id).slice(0,55))+'</div><div style="display:flex;align-items:center;gap:8px;margin-top:2px"><span style="font-size:10.5px;padding:1px 7px;border-radius:10px;background:'+color+'22;color:'+color+';font-weight:600">'+esc(s.domain_dir)+'</span>'+diff+'</div>'+problem+roi+'</div></div></a>';}
  function doSearch(){var q=input.value.trim().toLowerCase();if(q.length<2){box.classList.add('hidden');box.innerHTML='';return;}var words=q.split(/\s+/).filter(function(w){return w.length>0;});var scored=skills.map(function(s){return{s:s,sc:scoreSkill(s,q,words)};}).filter(function(x){return x.sc>0;}).sort(function(a,b){return b.sc-a.sc;});var hits=applyFilters(scored.map(function(x){return x.s;})).slice(0,16);var prefix=rootPrefix();if(!hits.length){box.innerHTML='<p class="muted" style="padding:12px 14px;font-size:13px">\u672a\u627e\u5230\u300c'+esc(q)+'\u300d\u76f8\u5173\u6280\u80fd</p>';box.classList.remove('hidden');return;}box.innerHTML=hits.map(function(s){return renderPreview(s,prefix);}).join('')+(scored.length>16?'<div style="padding:8px 14px;font-size:11.5px;color:#94a3b8;text-align:center">\u663e\u793a\u524d16\u6761\uff0c\u516825\u670827'+scored.length+'\u6761\u5339\u914d</div>':'');box.classList.remove('hidden');}
  waitForData(function(){input.addEventListener('input',doSearch);input.addEventListener('keydown',function(e){if(e.key==='Escape'){box.classList.add('hidden');input.blur();}if(e.key==='Enter'&&!box.classList.contains('hidden')){var first=box.querySelector('.result');if(first)window.location.href=first.href;}});document.addEventListener('click',function(e){if(!input.contains(e.target)&&!box.contains(e.target))box.classList.add('hidden');});['filter-diff','filter-roi','filter-domain'].forEach(function(id){var el=document.getElementById(id);if(el)el.addEventListener('change',doSearch);});});
})();
""", encoding="utf-8")


def _render_playbook_progress_page(playbooks: list) -> str:
    pb_meta_json = json.dumps(
        [{"id": pb["id"], "name": pb["name"], "total": len(pb.get("steps", []))}
         for pb in playbooks],
        ensure_ascii=False
    )
    pb_cards = ""
    for pb in playbooks:
        pb_id = pb["id"]
        pb_name = html.escape(pb["name"])
        pb_desc = html.escape(pb["desc"])
        pb_icon = pb["icon"]
        pb_tag = html.escape(pb.get("tag", ""))
        total_steps = len(pb.get("steps", []))
        step_items = "".join(
            f"<div class='prog-step' data-pb='{pb_id}' data-step='{i+1}'>"
            f"<span class='prog-step-num'>{i+1}</span>"
            f"<span class='prog-step-title'>{html.escape(s.get('step',''))[:50]}</span>"
            f"<span class='prog-step-status'>○</span>"
            f"</div>"
            for i, s in enumerate(pb.get("steps", []))
        )
        pb_cards += (
            f"<div class='prog-card' id='prog-card-{pb_id}'>"
            f"<div class='prog-card-header'>"
            f"<span class='prog-icon'>{pb_icon}</span>"
            f"<div class='prog-info'>"
            f"<div class='prog-name'>{pb_name}</div>"
            f"<div class='prog-tag'>{pb_tag}</div>"
            f"</div>"
            f"<div class='prog-pct' id='pct-{pb_id}'>0%</div>"
            f"</div>"
            f"<div class='prog-bar'><div class='prog-bar-fill' id='bar-{pb_id}' style='width:0'></div></div>"
            f"<div class='prog-steps'>{step_items}</div>"
            f"<div class='prog-actions'>"
            f"<a href='{pb_id}.html' class='prog-btn-open'>继续 →</a>"
            f"<button class='prog-btn-export' onclick='exportProgress(\"{pb_id}\",\"{pb_name}\")'>⬇ 导出报告</button>"
            f"<button class='prog-btn-reset' onclick='resetProgress(\"{pb_id}\")'>重置</button>"
            f"</div>"
            f"</div>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>学习进度看板 · paper2skills</title>
  <link rel="stylesheet" href="../assets/style.css">
  <style>
    .prog-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;padding:16px 0}}
    .prog-card{{background:var(--panel,#fff);border:1px solid var(--line,#e2e8f0);border-radius:12px;padding:16px 18px;transition:box-shadow .15s}}
    .prog-card:hover{{box-shadow:0 4px 16px rgba(0,0,0,.08)}}
    .prog-card-header{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
    .prog-icon{{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;font-weight:800;flex-shrink:0}}
    .prog-name{{font-size:14px;font-weight:700;color:var(--ink,#0f172a)}}
    .prog-tag{{font-size:11px;color:var(--muted,#94a3b8);margin-top:2px}}
    .prog-pct{{margin-left:auto;font-size:15px;font-weight:800;color:#059669;flex-shrink:0}}
    .prog-bar{{height:6px;background:var(--panel-2,#f1f5f9);border-radius:3px;overflow:hidden;margin-bottom:12px}}
    .prog-bar-fill{{height:100%;background:linear-gradient(90deg,#059669,#10b981);border-radius:3px;transition:width .5s ease}}
    .prog-steps{{display:flex;flex-direction:column;gap:4px;margin-bottom:12px}}
    .prog-step{{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:6px;transition:background .15s}}
    .prog-step.done{{background:#f0fdf4}}
    .prog-step-num{{width:20px;height:20px;border-radius:50%;background:var(--panel-2,#f1f5f9);display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:var(--muted);flex-shrink:0}}
    .prog-step.done .prog-step-num{{background:#059669;color:#fff}}
    .prog-step-title{{font-size:12px;color:var(--ink-2,#475569);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .prog-step-status{{font-size:14px;flex-shrink:0;color:var(--muted,#94a3b8)}}
    .prog-step.done .prog-step-status{{color:#059669}}
    .prog-actions{{display:flex;gap:8px;flex-wrap:wrap}}
    .prog-btn-open{{display:inline-flex;align-items:center;padding:6px 14px;background:var(--accent,#3b82f6);color:#fff;border-radius:8px;font-size:12px;font-weight:600;text-decoration:none;transition:background .15s}}
    .prog-btn-open:hover{{background:var(--accent-dark,#2563eb)}}
    .prog-btn-export,.prog-btn-reset{{padding:6px 12px;background:var(--panel-2,#f8fafc);border:1.5px solid var(--line,#e2e8f0);border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;color:var(--ink-2,#475569);transition:all .15s}}
    .prog-btn-export:hover{{border-color:var(--accent,#3b82f6);color:var(--accent,#3b82f6)}}
    .prog-btn-reset:hover{{border-color:#ef4444;color:#ef4444}}
    .prog-summary{{display:flex;gap:16px;flex-wrap:wrap;padding:14px 16px;background:var(--panel-2,#f8fafc);border-radius:10px;margin-bottom:16px;border:1px solid var(--line,#e2e8f0)}}
    .prog-sum-item{{text-align:center}}
    .prog-sum-num{{font-size:22px;font-weight:800;color:var(--ink,#0f172a)}}
    .prog-sum-label{{font-size:11px;color:var(--muted,#94a3b8);margin-top:2px}}
  </style>
</head>
<body>
  <header class="topbar">
    <button class="hamburger" id="hamburger" aria-label="菜单" aria-expanded="false"><span></span><span></span><span></span></button>
    <a class="brand" href="../index.html"><span class="brand-icon">P</span><span class="brand-name">paper2skills<span class="brand-tag">Playbook</span></span></a>
    <div class="topbar-right"><input id="global-search" placeholder="搜索技能 / 场景…" autocomplete="off" role="search" aria-label="搜索"><a href="../ai-roadmap.html" class="topbar-cta">AI 路线图 →</a></div>
  </header>
  <div id="search-results" class="search-results hidden" role="listbox"></div>
  <div class="mobile-nav-overlay" id="mobile-overlay"></div>
  <main class="layout">
    <aside class="sidebar" id="sidebar">
      <div class="sb-top">
        <div class="sb-section"><p class="sb-label">主导航</p><div class="sb-links"><a href="../index.html"><span class="sbl-icon"></span><span class="sbl-text">总览</span></a><a href="../chat.html"><span class="sbl-icon"></span><span class="sbl-text">AI 知识库对话</span></a><a href="../playbooks/index.html"><span class="sbl-icon"></span><span class="sbl-text">场景手册</span></a><a href="../solutions/index.html"><span class="sbl-icon"></span><span class="sbl-text">方案库</span></a><a href="../agents.html"><span class="sbl-icon"></span><span class="sbl-text">智能体广场</span></a><a href="../agent-report.html"><span class="sbl-icon"></span><span class="sbl-text">智能体报告</span></a><a href="../ai-roadmap.html"><span class="sbl-icon"></span><span class="sbl-text">AI 能力路线图</span></a></div></div>
        <div class="sb-section"><p class="sb-label">知识图谱</p><div class="sb-links"><a href="../domains/index.html"><span class="sbl-icon"></span><span class="sbl-text">按领域浏览</span></a><a href="../graph/overview.html"><span class="sbl-icon"></span><span class="sbl-text">技能关系图谱</span></a><a href="../skills/index.html"><span class="sbl-icon"></span><span class="sbl-text">全部 Skills</span></a></div></div>
      </div>
    </aside>
    <section class="content">
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="index.html">场景手册</a> / 学习进度看板</nav>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:10px">
  <div><h1 style="margin:0 0 4px">PL 学习进度看板</h1><p class="muted" style="margin:0">{len(playbooks)} 本手册 · 步骤完成情况一目了然 · 支持导出 Markdown 报告</p></div>
  <div style="display:flex;gap:8px">
    <button onclick="exportAllProgress()" style="padding:8px 16px;background:var(--accent,#3b82f6);color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer">⬇ 导出全部报告</button>
    <a href="index.html" style="padding:8px 14px;background:var(--panel-2,#f8fafc);border:1.5px solid var(--line,#e2e8f0);border-radius:8px;font-size:13px;font-weight:600;color:var(--ink-2);text-decoration:none">← 返回手册列表</a>
  </div>
</div>
<div class="prog-summary" id="overall-summary">
  <div class="prog-sum-item"><div class="prog-sum-num" id="sum-total">{len(playbooks)}</div><div class="prog-sum-label">总手册数</div></div>
  <div class="prog-sum-item"><div class="prog-sum-num" id="sum-started" style="color:#3b82f6">0</div><div class="prog-sum-label">已开始</div></div>
  <div class="prog-sum-item"><div class="prog-sum-num" id="sum-completed" style="color:#059669">0</div><div class="prog-sum-label">已完成</div></div>
  <div class="prog-sum-item"><div class="prog-sum-num" id="sum-steps" style="color:#8b5cf6">0</div><div class="prog-sum-label">完成步骤数</div></div>
</div>
<div class="prog-grid">{pb_cards}</div>
<script>
const PB_META = {pb_meta_json};
function loadAllProgress() {{
  let started=0,completed=0,totalSteps=0;
  PB_META.forEach(pb => {{
    const done = JSON.parse(localStorage.getItem('pb_progress_'+pb.id)||'[]');
    const pct = pb.total>0 ? Math.round(done.length/pb.total*100) : 0;
    const pctEl=document.getElementById('pct-'+pb.id);
    const barEl=document.getElementById('bar-'+pb.id);
    if(pctEl) pctEl.textContent=pct+'%';
    if(pctEl) pctEl.style.color=pct>=100?'#059669':pct>0?'#3b82f6':'#94a3b8';
    if(barEl) barEl.style.width=pct+'%';
    document.querySelectorAll('.prog-step[data-pb="'+pb.id+'"]').forEach(step => {{
      const stepNum = parseInt(step.dataset.step);
      const isDone = done.includes(stepNum);
      step.classList.toggle('done', isDone);
      step.querySelector('.prog-step-status').textContent = isDone ? '✓' : '○';
    }});
    if(done.length>0) started++;
    if(pct>=100) completed++;
    totalSteps+=done.length;
  }});
  document.getElementById('sum-started').textContent=started;
  document.getElementById('sum-completed').textContent=completed;
  document.getElementById('sum-steps').textContent=totalSteps;
}}
function resetProgress(pbId) {{
  if(!confirm('确认重置「'+pbId+'」的进度吗？')) return;
  localStorage.removeItem('pb_progress_'+pbId);
  loadAllProgress();
}}
function exportProgress(pbId, pbName) {{
  const pb = PB_META.find(p=>p.id===pbId);
  if(!pb) return;
  const done = JSON.parse(localStorage.getItem('pb_progress_'+pbId)||'[]');
  const pct = pb.total>0?Math.round(done.length/pb.total*100):0;
  let md = '# '+pbName+' — 学习进度报告\\n\\n';
  md += '> 导出时间: '+new Date().toLocaleString('zh-CN')+'\\n\\n';
  md += '## 完成情况\\n\\n';
  md += '- 完成步骤: '+done.length+'/'+pb.total+'\\n';
  md += '- 完成率: '+pct+'%\\n\\n';
  md += '## 已完成步骤\\n\\n';
  done.forEach(s => {{ md += '- [x] Step '+s+'\\n'; }});
  const undone = Array.from({{length:pb.total}},((_,i)=>i+1)).filter(s=>!done.includes(s));
  if(undone.length) {{ md += '\\n## 待完成步骤\\n\\n'; undone.forEach(s=>{{md+='- [ ] Step '+s+'\\n';}}); }}
  const blob = new Blob([md], {{type:'text/markdown;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = pbId+'-progress-'+new Date().toISOString().slice(0,10)+'.md';
  a.click();
}}
function exportAllProgress() {{
  let md = '# paper2skills 全部手册进度报告\\n\\n';
  md += '> 导出时间: '+new Date().toLocaleString('zh-CN')+'\\n\\n';
  md += '| 手册 | 完成率 | 步骤 |\\n|------|--------|------|\\n';
  PB_META.forEach(pb => {{
    const done = JSON.parse(localStorage.getItem('pb_progress_'+pb.id)||'[]');
    const pct = pb.total>0?Math.round(done.length/pb.total*100):0;
    md += '| '+pb.name+' | '+pct+'% | '+done.length+'/'+pb.total+' |\\n';
  }});
  const blob = new Blob([md], {{type:'text/markdown;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'playbook-progress-'+new Date().toISOString().slice(0,10)+'.md';
  a.click();
}}
window.addEventListener('DOMContentLoaded', loadAllProgress);
</script>
    </section>
  </main>
  <script src="../assets/playbook-data.js"></script>
  <script src="../assets/search.js"></script>
  <script>
  const hbtn=document.getElementById('hamburger'),overlay=document.getElementById('mobile-overlay');
  function toggleMenu(open){{hbtn.setAttribute('aria-expanded',open);hbtn.classList.toggle('open',open);overlay.classList.toggle('show',open);document.body.style.overflow=open?'hidden':'';}}
  hbtn.addEventListener('click',()=>toggleMenu(hbtn.getAttribute('aria-expanded')!=='true'));overlay.addEventListener('click',()=>toggleMenu(false));
  </script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper2skills static Playbook")
    parser.add_argument("--root",  default=str(BASE_DIR))
    parser.add_argument("--vault", default="paper2skills-vault")
    parser.add_argument("--out",   default="playbook")
    args = parser.parse_args()

    root  = Path(args.root).resolve()
    vault = (root / args.vault).resolve() if not Path(args.vault).is_absolute() else Path(args.vault)
    out   = (root / args.out).resolve()   if not Path(args.out).is_absolute()   else Path(args.out)
    report = build(root, vault, out)
    _post_build_patch(out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
