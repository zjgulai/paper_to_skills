#!/usr/bin/env python3
"""Build a static HTML Playbook for paper2skills Skill cards."""

from __future__ import annotations

import argparse
import html
import json
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
    "relations": ["skill relations", "技能关联", "技能关系", "技能关系"],
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
    "视觉内容生成": ["video", "visual", "image", "avatar", "ai视频", "multimodal"],
}

WORKFLOW_RULES = {
    "WF-A 智能补货": ["供应链", "库存", "补货", "demand", "forecast", "lead-time", "safety-stock", "logistics"],
    "WF-B 广告优化": ["广告", "roas", "attribution", "tiktok", "keyword", "creative", "mmm", "marketing"],
    "WF-C 客服分诊": ["客服", "review", "voc", "absa", "translation", "customer", "sentiment"],
    "WF-D 选品扫描": ["选品", "product", "market", "competitive", "signal", "data collection", "knowledge graph"],
    "WF-E Review监控": ["review", "fake-review", "sentiment", "absa", "dedup", "quality"],
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
# Business-problem → workflow quick-entry (for home page C-redesign)
# ---------------------------------------------------------------------------
BUSINESS_ENTRIES = [
    {
        "icon": "🛡️",
        "label": "防御竞品攻击 / 平台封号预防",
        "desc": "广告刷量、虚假差评、AI 推荐注入、合规封号——四条战线主动防御",
        "href": "playbooks/pb-risk-defense.html",
        "tag": "风险防御",
    },
    {
        "icon": "⚡",
        "label": "关税冲击 / 贸易政策应对",
        "desc": "72 小时内输出完整行动清单：定价调整 + 库存处置 + 供应链转移方案",
        "href": "playbooks/pb-tariff-response.html",
        "tag": "关税响应",
    },
    {
        "icon": "📈",
        "label": "提升广告 ROI / 归因准确性",
        "desc": "识别无效预算、纠正渠道归因偏差、实现因果驱动的广告优化",
        "href": "workflows/wf-b-广告优化.html",
        "tag": "WF-B 广告优化",
    },
    {
        "icon": "📦",
        "label": "FBA 库存健康 / 头程优化",
        "desc": "长库龄清仓 + 头程路线成本优化 + 旺季备货计划，库存周转天数降 30%",
        "href": "playbooks/pb-fba-operations.html",
        "tag": "FBA 运营",
    },
    {
        "icon": "🔬",
        "label": "竞品差评 → 新品机会挖掘",
        "desc": "竞品 1-3 星差评是最好的免费 R&D，新品成功率从 30% 提升到 50%",
        "href": "playbooks/pb-voc-product-loop.html",
        "tag": "竞品情报",
    },
    {
        "icon": "🎧",
        "label": "客服 24h 自动化 / 差评防御",
        "desc": "70% 工单全自动处理，多语言覆盖，INR 欺诈退货从 35% 降至 5%",
        "href": "playbooks/pb-customer-service-agent.html",
        "tag": "客服售后",
    },
    {
        "icon": "💬",
        "label": "分析用户评价 / 发现产品痛点",
        "desc": "多语言 VOC 挖掘、差评根因归类、产品改进信号提取",
        "href": "workflows/wf-c-客服分诊.html",
        "tag": "WF-C 客服",
    },
    {
        "icon": "🔍",
        "label": "评估新品 / 新市场机会",
        "desc": "市场规模估算、竞品情报采集、选品可行性综合评分",
        "href": "workflows/wf-d-选品扫描.html",
        "tag": "WF-D 选品",
    },
    {
        "icon": "👤",
        "label": "预测用户流失 / 提升 LTV",
        "desc": "Uplift 建模识别可干预用户，精准发券减少无效留存成本",
        "href": "domains/14-用户分析.html",
        "tag": "用户分析",
    },
    {
        "icon": "🤖",
        "label": "AI Agent 替代重复性岗位",
        "desc": "供应链对账、数据分析提数、广告出价——三类岗位 70% 重复工作 Agent 覆盖",
        "href": "playbooks/pb-agent-replace.html",
        "tag": "Agent 替人",
    },
    {
        "icon": "🏷️",
        "label": "动态定价 / A/B 实测 GMV +13%",
        "desc": "LLM 动态定价引擎，定价是乘数——精准定价 1% 比多投广告 15% 更高效",
        "href": "playbooks/pb-pricing-engine.html",
        "tag": "定价引擎",
    },
    {
        "icon": "🚀",
        "label": "新品冷启动备货 / 预测",
        "desc": "零历史数据下的扩散曲线预测，跨市场迁移学习",
        "href": "playbooks/pb-new-product-launch.html",
        "tag": "新品冷启动",
    },
]


@dataclass
class PlaybookSkill:
    skill_id: str
    title: str
    domain_key: str
    domain_dir: str
    path: str
    status: str = "unknown"
    topic: str = ""
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
    source_excerpt: str = ""


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


def first_bold_sentence(text: str, fallback: str = "") -> str:
    """Extract problem statement: prefer '核心问题/业务问题' labelled sentence, then first bold."""
    for marker in ("核心问题", "业务问题", "核心挑战", "解决的核心问题"):
        m = re.search(
            r"(?:" + marker + r")[：:\s]*([^\n。]{15,180})",
            text,
        )
        if m:
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()
            if len(clean) > 15:
                return clean[:200]
    m = re.search(r"\*\*(.{15,150}?)\*\*", text)
    if m:
        return m.group(1).strip()
    return first_nonempty_line(text, fallback)


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
            status=fm.get("status", "unknown"),
            topic=fm.get("topic", ""),
            algorithm_summary=first_nonempty_line(algo_text, first_nonempty_line(body, skill_id)),
            problem_solved=first_bold_sentence(algo_text, first_nonempty_line(scenario_text, "")),
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
            source_excerpt=algo_text[:1200] or body[:1200],
        )
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
        except Exception:
            pass
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
        parts.append(f'  <div class="wf-question">❓ {question}</div>')
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

TOB_PLAYBOOKS: list[dict[str, Any]] = [
    {
        "id": "pb-tiktok-shop",
        "icon": "🎯",
        "name": "TikTok Shop 运营决策手册",
        "tag": "广告 · 内容 · 归因",
        "desc": "从内容归因到智能竞价的 TikTok Shop 全链路数据决策指南",
        "intro": "TikTok Shop 运营面临三大核心挑战：内容归因不准（到底是哪条视频带来的转化）、"
                 "竞价策略不清（出价多少 ROAS 最优）、冷启动阶段缺数据。本手册提供从第 1 天就能用的决策路径。",
        "steps": [
            {
                "step": "Step 1 — 内容归因（上线第 1 周）",
                "problem": "视频带货归因不准：直播、短视频、商品卡三类流量的真实增量贡献各是多少？",
                "skills": [
                    {"id": "Skill-Causal-Attribution-Bridge", "why": "将 naive 归因（点击→购买）替换为因果 ITE，识别「无论有没有这条视频都会买」的用户"},
                    {"id": "Skill-TikTok-Shop-Content-Attribution", "why": "TikTok 场域专用的内容-转化路径建模"},
                ],
                "data": "需要：用户曝光日志（impression）、点击日志、订单数据，至少 7 天",
                "output": "每条内容的真实增量 ITE 值，用于下周预算分配",
            },
            {
                "step": "Step 2 — 受众精准化（第 2-4 周）",
                "problem": "如何找到「这个视频对谁最有效」，而不是全量推送？",
                "skills": [
                    {"id": "Skill-Uplift-Modeling", "why": "识别可说服者（Persuadables），避免对必然购买者和无法说服者浪费预算"},
                    {"id": "Skill-Guardrailed-Uplift-Targeting", "why": "加入预算护栏约束，自动输出最优干预名单"},
                ],
                "data": "需要：A/B 实验数据（发/未发内容对照组）或历史随机分组数据，≥5000 用户",
                "output": "按 CATE 排序的用户分群，精准定向投放",
            },
            {
                "step": "Step 3 — 智能竞价（第 4 周起）",
                "problem": "如何在有限预算下最大化 ROAS，同时避免渠道过度饱和？",
                "skills": [
                    {"id": "Skill-LLM-AutoBidding-MAS", "why": "LLM 驱动的层次化竞价，支持多目标（GMV、ROAS、品牌曝光）动态平衡"},
                    {"id": "Skill-Channel-Saturation-Curve", "why": "识别投放饱和拐点，防止边际 ROAS 持续下滑"},
                    {"id": "Skill-Creative-Fatigue-Detection", "why": "监控创意疲劳，自动触发素材轮换"},
                ],
                "data": "需要：每日投放日志、ROAS 数据，实时流可选",
                "output": "自动竞价策略 + 创意轮换提醒",
            },
        ],
        "outcomes": ["内容归因准确率 +30%，预算错配减少", "可说服用户定向精准率 +25%", "竞价 ROAS 提升 15-25%"],
    },
    {
        "id": "pb-inventory-festival",
        "icon": "📦",
        "name": "大促备货决策手册",
        "tag": "供应链 · 补货 · 预测",
        "desc": "双十一 / Prime Day / Black Friday 前 8 周的库存决策完整路线图",
        "intro": "大促备货最大的风险：备多了积压，备少了断货。两种错误的年化损失都可能超过 300 万。"
                 "本手册提供从 T-8 周到大促当日的分阶段决策节点。",
        "steps": [
            {
                "step": "T-8 周 — 需求预测基线",
                "problem": "大促期间需求是平日的 3-10 倍，历史数据如何泛化到极端场景？",
                "skills": [
                    {"id": "Skill-Demand-Forecasting-Supply-Chain", "why": "结合促销日历、竞品价格、渠道库存的供应链专用预测"},
                    {"id": "Skill-Promotion-Demand-Decomposition", "why": "将大促需求拆解为：基础需求 + 促销增量 + 渠道转移"},
                    {"id": "Skill-Conformal-Prediction-Demand-UQ", "why": "输出置信区间（P10/P50/P90），为保守/激进备货方案提供概率支撑"},
                ],
                "data": "需要：近 2 年同类大促历史销售、促销方案草案、竞品去年大促价格",
                "output": "SKU 级需求预测 + 90% 置信区间",
            },
            {
                "step": "T-4 周 — 补货策略锁定",
                "problem": "如何在 MOQ 约束和海外仓容积限制下，制定最优补货计划？",
                "skills": [
                    {"id": "Skill-Safety-Stock-Replenishment", "why": "动态安全库存计算，考虑供应商交期波动"},
                    {"id": "Skill-Dynamic-Lot-Sizing-MOQ", "why": "最小起订量约束下的动态批量优化，避免超额采购"},
                    {"id": "Skill-Multi-Channel-Inventory-Pooling", "why": "跨 Amazon/独立站/TikTok 动态调拨，防止 A 仓过剩 B 仓断货"},
                ],
                "data": "需要：供应商 lead time 分布、各仓容积上限、各渠道历史分单比例",
                "output": "分渠道补货计划 + 调拨策略",
            },
            {
                "step": "大促实时 — 库存健康监控",
                "problem": "大促期间如何实时发现异常并快速响应？",
                "skills": [
                    {"id": "Skill-Inventory-Health-Aging-Attribution", "why": "实时 FSN 分级监控，识别哪些 SKU 正在快速耗尽"},
                    {"id": "Skill-Argos-Agentic-Anomaly-Detection", "why": "Agentic 异常检测，自动预警库存骤降"},
                ],
                "data": "需要：实时库存快照（每小时）、销售流水",
                "output": "实时库存预警 + 自动调拨建议",
            },
        ],
        "outcomes": ["需求预测 MAPE 降低 20-35%", "库存积压减少 15-25%", "断货率从 8% → 3%", "年化节省 50-200 万元"],
    },
    {
        "id": "pb-new-product-launch",
        "icon": "🚀",
        "name": "新品冷启动手册",
        "tag": "选品 · 预测 · 增长",
        "desc": "零历史数据下的新品上市全链路决策：从选品验证到首批备货到冷启动推广",
        "intro": "母婴跨境每年 20-30 款新品上市，前 8 周零销售记录，人工拍脑袋备货。"
                 "本手册用数据替代直觉，从选品验证开始就建立可量化的决策依据。",
        "steps": [
            {
                "step": "上市前 12 周 — 选品验证",
                "problem": "这个品在目标市场有多大空间？竞争格局如何？",
                "skills": [
                    {"id": "Skill-Market-Size-Estimation", "why": "TAM/SAM/SOM 双路径估算 + Monte Carlo 置信区间"},
                    {"id": "Skill-Cross-Market-Product-Transfer", "why": "预测国内爆品在海外的适配性，避免负迁移"},
                    {"id": "Skill-Category-Compliance-Prescan", "why": "目标市场合规预扫描（FDA/CE），提前发现上市障碍"},
                ],
                "data": "需要：类似品近 1 年 Amazon BSR、Google Trends、同类竞品价格带",
                "output": "市场空间评分 + GO/NO-GO 建议",
            },
            {
                "step": "上市前 8 周 — 首批备货预测",
                "problem": "没有历史数据，首批备多少货？",
                "skills": [
                    {"id": "Skill-Bass-Diffusion-New-Product-Forecasting", "why": "Bass 扩散模型 + 相似品参数迁移，输出 8 周扩散曲线"},
                    {"id": "Skill-Cold-Start-Product-Recommendation", "why": "LLM 模拟用户行为，预测冷启动期的需求信号"},
                ],
                "data": "需要：3 个以上相似品的历史销售曲线、产品定价方案",
                "output": "首批备货建议量（P25/P50/P75 三档）",
            },
            {
                "step": "上市后 1-4 周 — 冷启动加速",
                "problem": "如何在数据稀疏期快速学习并调整投放策略？",
                "skills": [
                    {"id": "Skill-BCCB-Causal-Bandits", "why": "预算约束因果 Bandit，Day 1 就开始在线学习，无需等待历史数据"},
                    {"id": "Skill-Thompson-Sampling-MAB", "why": "MAB 动态分配流量，快速识别高转化渠道"},
                    {"id": "Skill-Category-Trend-Forecasting", "why": "实时监测品类趋势，判断新品是否踩上上升风口"},
                ],
                "data": "需要：实时用户行为流（曝光→点击→加购→转化）",
                "output": "最优渠道分配 + 动态竞价策略",
            },
        ],
        "outcomes": ["首批备货准确率提升，积压/断货损失减少 60%", "冷启动学习周期从 4 周压缩至 1 周", "选品 GO/NO-GO 决策有数据支撑"],
    },
    {
        "id": "pb-user-growth",
        "icon": "👤",
        "name": "用户增长决策手册",
        "tag": "LTV · 流失 · 分层运营",
        "desc": "从用户价值分层到精准干预的全链路用户增长决策指南",
        "intro": "母婴跨境用户运营的核心矛盾：大多数营销动作（发券、触达）对「无论如何都会复购」的用户是纯浪费，"
                 "对「无论如何都不会复购」的用户是无效投入。真正的增长在于找准「可被干预改变」的那 10-20%。",
        "steps": [
            {
                "step": "Step 1 — 用户价值分层",
                "problem": "谁是你的高价值用户？谁即将流失？谁从未真正激活？",
                "skills": [
                    {"id": "Skill-RFM-Customer-Segmentation", "why": "用 R/F/M 三维把用户分成 8 类（冠军→流失→沉睡），每类对应不同运营策略"},
                    {"id": "Skill-LTV-Prediction-ZILN", "why": "零膨胀对数正态模型预测用户生命周期价值，识别潜力高价值用户"},
                    {"id": "Skill-User-Lifecycle-STAN", "why": "时空注意力网络建模用户生命周期阶段（新客→成熟→衰退），判断当前阶段"},
                ],
                "data": "需要：用户历史订单（近 1 年）、注册时间、品类购买记录",
                "output": "用户价值分层标签 + 各层用户规模与贡献占比",
            },
            {
                "step": "Step 2 — 流失预警与精准干预",
                "problem": "哪些用户会流失？发券对谁真正有效？",
                "skills": [
                    {"id": "Skill-Uplift-Churn-Prediction", "why": "Uplift 流失预测：识别「可说服者」，避免对必然流失和必然留存的用户浪费资源"},
                    {"id": "Skill-Guardrailed-CATE-NBA", "why": "带预算护栏的最优行动决策：在预算约束下输出最优干预名单"},
                    {"id": "Skill-Customer-Churn-Prediction", "why": "深度学习流失预测，捕捉复杂的行为序列模式"},
                ],
                "data": "需要：历史 A/B 干预数据（发/未发券对照）或 RCT 数据，≥ 5000 用户",
                "output": "可干预用户名单（按 CATE 排序）+ 干预方式推荐（券面值/文案/渠道）",
            },
            {
                "step": "Step 3 — 复购周期与 LTV 提升",
                "problem": "如何在正确的时机触达用户，提升复购率？",
                "skills": [
                    {"id": "Skill-Cohort-Retention-Analysis", "why": "队列留存分析，找出复购的关键时间窗口（如首单后 14 天是黄金窗口）"},
                    {"id": "Skill-Long-Term-Preference-Memory", "why": "长期偏好记忆模型，捕捉用户跨品类的兴趣演变"},
                    {"id": "Skill-User-Profile-Long-Memory", "why": "用户长记忆画像，支持个性化触达内容生成"},
                ],
                "data": "需要：用户行为序列（浏览/加购/购买）、触达记录与响应结果",
                "output": "个人化复购提醒时机 + 触达内容模板",
            },
            {
                "step": "Step 4 — 新用户冷启动推荐",
                "problem": "新用户没有历史数据，如何实现个性化推荐？",
                "skills": [
                    {"id": "Skill-Cold-Start-Product-Recommendation", "why": "LLM 模拟新用户行为，生成合成交互数据填补冷启动空白"},
                    {"id": "Skill-Cold-Start-Meta-Learning-PAM", "why": "元学习框架，用少量交互数据快速适配新用户偏好"},
                ],
                "data": "需要：注册信息、首次浏览 session 行为（即使只有 3-5 次点击）",
                "output": "新用户首屏个性化推荐列表",
            },
        ],
        "outcomes": [
            "可说服用户识别精准率 +25%，优惠券 ROI 提升 3-5x",
            "复购率提升 10-15%，LTV 增加",
            "新用户首单转化率提升 15-20%",
            "年化节省无效促销成本 25-50 万元",
        ],
    },
    {
        "id": "pb-data-foundation",
        "icon": "🏗️",
        "name": "数据治理基础手册",
        "tag": "数据质量 · KG · Agent",
        "desc": "中小跨境电商团队从零建立 AI 可用数据基础设施的分阶段路线图",
        "intro": "AI 决策的上限是数据质量。大多数中小团队在部署 AI 时遭遇的「效果差」，根因不是模型不好，"
                 "而是数据没有准备好。本手册提供从数据采集、清洗到知识图谱的分阶段建设路线。",
        "steps": [
            {
                "step": "Step 1 — 数据采集与质量基线",
                "problem": "数据从哪来？质量怎么保证？",
                "skills": [
                    {"id": "Skill-Data-Collection-Agent-Pipeline", "why": "Agent 驱动的自动化数据采集流水线，覆盖 Amazon/社媒/竞品多源"},
                    {"id": "Skill-Ecommerce-Data-Quality-Assessment", "why": "电商数据质量综合评估，建立数据质量基线（完整性/准确性/时效性）"},
                    {"id": "Skill-Data-Provenance-Lineage", "why": "数据血缘追踪，知道每条数据从哪来、经过什么处理"},
                ],
                "data": "需要：现有数据源清单（ERP/平台API/手工Excel）",
                "output": "数据源地图 + 质量评分报告 + 优先修复清单",
            },
            {
                "step": "Step 2 — 数据清洗与治理",
                "problem": "如何系统性消除脏数据、孤岛数据、重复数据？",
                "skills": [
                    {"id": "Skill-Review-Dedup-Quality-Filter", "why": "评论/工单去重与质量过滤，建立标准化文本数据集"},
                    {"id": "Skill-Entity-Resolution-KG-Dedup", "why": "跨系统实体解析去重（同一 SKU 在不同系统有不同编码）"},
                    {"id": "Skill-Data-Drift-Detection", "why": "数据漂移检测，发现数据分布变化（如季节性导致的分布偏移）"},
                ],
                "data": "需要：各系统导出的原始数据文件",
                "output": "清洗后的标准化数据集 + 实体映射表",
            },
            {
                "step": "Step 3 — 产品知识图谱构建",
                "problem": "如何把产品目录、用户行为、竞品关系结构化为 AI 可查询的知识库？",
                "skills": [
                    {"id": "Skill-Ontology-Schema-Design", "why": "母婴电商本体设计（品牌→系列→产品→成分→适用年龄），是 KG 的地图"},
                    {"id": "Skill-Hierarchical-Product-KG-Construction", "why": "层次化产品知识图谱自动构建"},
                    {"id": "Skill-KG-Incremental-Update", "why": "图谱增量更新，新品上架/下架自动同步，不用每次全量重建"},
                ],
                "data": "需要：产品目录（SKU/类目/属性）、历史订单中的同购关系",
                "output": "可查询的产品知识图谱（支持 KGQA + GraphRAG）",
            },
            {
                "step": "Step 4 — AI 可用数据接口",
                "problem": "如何让 AI Agent 能直接查询业务数据，而不依赖数据团队提数？",
                "skills": [
                    {"id": "Skill-SQL-Agent-Text-to-SQL", "why": "自然语言转 SQL，业务同学直接用中文查数据库"},
                    {"id": "Skill-NL2Dashboard-Automation", "why": "自然语言→仪表盘自动生成，运营自助分析无需等排期"},
                    {"id": "Skill-RAG-Enhanced-Data-Analysis", "why": "RAG 增强数据分析，结合知识库回答「为什么」类问题"},
                ],
                "data": "需要：已完成 Step 1-3 的数据基础设施",
                "output": "可供 AI Agent 调用的数据查询接口 + 自助分析工具",
            },
        ],
        "outcomes": [
            "数据质量分从基线提升，AI 模型效果直接受益",
            "运营自助分析比例从 20% → 80%，数据团队提数工作量 -60%",
            "知识图谱建成后 KGQA 查询召回率 > 90%",
            "新品上架数据同步从 2 天 → 实时",
        ],
    },
    {
        "id": "pb-agent-replace",
        "icon": "🤖",
        "name": "AI Agent 替人手册",
        "tag": "供应链 · 数据分析 · 广告优化",
        "desc": "三类岗位的核心重复性工作，逐步交给 AI Agent 执行，释放人力专注高价值决策",
        "intro": "AI 替代的不是岗位，是岗位里的「重复性决策」——每天都要做、做法固定、但规模超出人力上限的那部分。"
                 "本手册从供应链出发，提供三条独立的替代路径，每条路径都可以单独落地。",
        "steps": [
            {
                "step": "Chapter 1：供应链全链路 Agent（主角）",
                "problem": "15 个 Excel 联动地狱：SKU×仓库×市场三层预测加总不一致，月底 2-3 天纯人工对账；大促定价和补货是两个团队各自决策，无法实时联动",
                "skills": [
                    {"id": "Skill-Hierarchical-Demand-Forecasting-Reconciliation", "why": "分层预测调和，数学保证 SKU→仓库→市场三层一致，月底对账人力清零（ROI 800-1500万）"},
                    {"id": "Skill-Event-Driven-Demand-MAS", "why": "事件感知补货 MAS：大促信号触发自动备货，告别「排期跑批已来不及」（ROI 5000万）"},
                    {"id": "Skill-FSDA-DRL", "why": "快慢双 Agent 联动定价与补货：大促全周期利润最优，替代供应链+运营协调会（ROI 5000万）"},
                    {"id": "Skill-Lead-Time-Distribution-Risk-GenQOT", "why": "海运延误动态安全库存：苏伊士/巴拿马运河异常时自动预警并提前采购（ROI 200-500万）"},
                    {"id": "Skill-Multi-Channel-Inventory-Pooling", "why": "跨渠道动态调拨：Amazon缺货+独立站积压同时存在时自动平衡（ROI 200-400万）"},
                ],
                "data": "需要：60天+销售历史、多仓库存快照、大促日历、供应商交货记录",
                "output": "自动化补货计划 + 大促定价策略 + 海运风险预警 + 跨渠道调拨指令",
            },
            {
                "step": "Chapter 2：数据分析师 Agent",
                "problem": "提数需求排期 2-3 天，分析师 80% 时间在取数不在分析；GMV 异常时人工排查根因需要 1-2 小时，节假日无人监控",
                "skills": [
                    {"id": "Skill-SQL-Agent-Text-to-SQL", "why": "自然语言→SQL：业务同学直接用中文查数据库，消灭 BI 提数排期"},
                    {"id": "Skill-NL2Dashboard-Automation", "why": "自然语言→仪表盘自动生成，运营自助分析，节省 BI 开发人力"},
                    {"id": "Skill-DeepAnalyze-Autonomous-Data-Science-Agent", "why": "自主数据科学 Agent：多平台数据周报 4-6 小时人工 → 5 分钟 Agent 自动生成（ROI 20万）"},
                    {"id": "Skill-ProRCA-Business-Analysis", "why": "因果图根因分析：GMV 暴跌 1-2 小时人工排查 → 0.5 秒锁定根因路径（ROI 100万）"},
                ],
                "data": "需要：数据仓库连接权限、历史报表模板、异常阈值定义",
                "output": "自助查询接口 + 自动周报 + 实时异常根因诊断",
            },
            {
                "step": "Chapter 3：广告优化师 Agent",
                "problem": "人工出价精度上限是工作时长：凌晨 2 点平台出新流量包，人在睡觉；新品冷启动 3-5 周无数据，靠经验猜测出价",
                "skills": [
                    {"id": "Skill-Negative-Keyword-Safe-Guard", "why": "贝叶斯小样本负关键词过滤：无关消耗从 18% → 3.2%，自动替代人工整词/否词"},
                    {"id": "Skill-Creative-Fatigue-Detection", "why": "广告素材疲劳检测：CTR/CVR 持续衰减时自动触发素材更新信号，替代人工监控"},
                    {"id": "Skill-DARA-Agentic-MMM-Optimizer", "why": "LLM+RL 双阶段广告预算分配：自动日预算分配，冷启动 ROAS 提升 15-30%（ROI 360-720万）"},
                    {"id": "Skill-Identified-Bayesian-MMM", "why": "贝叶斯 MMM 归因：告诉你广告钱真正浪费在哪里，1000万 ROI"},
                ],
                "data": "需要：各平台广告 API 数据、历史出价记录、素材表现数据",
                "output": "自动出价策略 + 素材更换信号 + 预算分配建议 + 归因报告",
            },
        ],
        "outcomes": [
            "供应链人工对账工作量从 2-3 天/月 → 接近零",
            "数据分析提数响应从 2-3 天 → 5 分钟",
            "广告无关消耗从 18% → 3.2%，冷启动 ROAS +15-30%",
            "三类岗位重复性工作 Agent 覆盖率 > 70%，人力聚焦高价值决策",
        ],
        "roi_calculator": {
            "title": "计算你的 ROI",
            "subtitle": "填入你的业务数字，实时看 AI Agent 的年化收益估算",
            "sections": [
                {
                    "id": "sc",
                    "label": "📦 供应链",
                    "color": "#2563eb",
                    "inputs": [
                        {"id": "sku_count",    "label": "管理 SKU 数量",        "unit": "个",    "default": 60,      "min": 1,    "max": 5000,   "step": 10},
                        {"id": "stockout_rate","label": "当前断货率",            "unit": "%",     "default": 8,       "min": 0,    "max": 50,     "step": 0.5},
                        {"id": "monthly_gmv",  "label": "月均 GMV",             "unit": "万元",  "default": 200,     "min": 10,   "max": 50000,  "step": 10},
                        {"id": "overstock_pct","label": "库存积压占总库存比例", "unit": "%",     "default": 20,      "min": 0,    "max": 80,     "step": 1},
                        {"id": "pm_days",      "label": "月底对账人力",         "unit": "人天/月","default": 3,      "min": 0,    "max": 30,     "step": 0.5},
                        {"id": "pm_cost",      "label": "人力日均成本",         "unit": "元/天", "default": 800,     "min": 200,  "max": 5000,   "step": 100},
                    ],
                    "formula": """
                        const stockout_loss = (stockout_rate/100) * monthly_gmv * 12 * 0.4;
                        const overstock_cost = (overstock_pct/100) * monthly_gmv * 2 * 0.15;
                        const labor_save = pm_days * pm_cost * 12 / 10000;
                        const improvement_stockout = stockout_loss * 0.55;
                        const improvement_overstock = overstock_cost * 0.25;
                        return Math.round(improvement_stockout + improvement_overstock + labor_save);
                    """,
                    "items": [
                        {"label": "断货损失减少（断货率 8%→3.5%）", "key": "improvement_stockout"},
                        {"label": "库存积压成本降低（25%）",         "key": "improvement_overstock"},
                        {"label": "对账人力节省（年化）",             "key": "labor_save"},
                    ],
                },
                {
                    "id": "da",
                    "label": "📊 数据分析师",
                    "color": "#7c3aed",
                    "inputs": [
                        {"id": "analyst_count",   "label": "数据分析师人数",      "unit": "人",    "default": 3,    "min": 1,   "max": 50,    "step": 1},
                        {"id": "analyst_salary",  "label": "分析师年均成本",      "unit": "万元/人","default": 40,  "min": 10,  "max": 200,   "step": 5},
                        {"id": "fetch_hours_pct", "label": "取数占工作时间比例",  "unit": "%",     "default": 60,   "min": 10,  "max": 90,    "step": 5},
                        {"id": "anomaly_per_month","label": "月均 GMV 异常事件",  "unit": "次",    "default": 2,    "min": 0,   "max": 50,    "step": 1},
                        {"id": "anomaly_cost",    "label": "平均每次异常损失",    "unit": "万元",  "default": 5,    "min": 1,   "max": 500,   "step": 1},
                    ],
                    "formula": """
                        const labor_save = analyst_count * analyst_salary * (fetch_hours_pct/100) * 0.7;
                        const anomaly_save = anomaly_per_month * anomaly_cost * 12 * 0.6;
                        return Math.round(labor_save + anomaly_save);
                    """,
                    "items": [
                        {"label": "取数人力节省（覆盖 70% 取数工作）", "key": "labor_save"},
                        {"label": "异常响应加速带来的损失减少",         "key": "anomaly_save"},
                    ],
                },
                {
                    "id": "ad",
                    "label": "📈 广告优化师",
                    "color": "#059669",
                    "inputs": [
                        {"id": "monthly_adspend", "label": "月广告投放额",         "unit": "万元",  "default": 50,    "min": 1,   "max": 10000, "step": 5},
                        {"id": "wasted_pct",      "label": "估算无效消耗比例",     "unit": "%",     "default": 18,    "min": 0,   "max": 60,    "step": 1},
                        {"id": "cold_start_sku",  "label": "月均新品冷启动 SKU 数","unit": "个",    "default": 3,     "min": 0,   "max": 100,   "step": 1},
                        {"id": "cold_start_spend","label": "每 SKU 冷启动广告费",  "unit": "万元",  "default": 5,     "min": 0,   "max": 200,   "step": 1},
                    ],
                    "formula": """
                        const keyword_save = monthly_adspend * 12 * ((wasted_pct - 3.2) / 100) * 0.8;
                        const roas_lift = cold_start_sku * cold_start_spend * 12 * 0.20;
                        return Math.round(Math.max(0, keyword_save) + roas_lift);
                    """,
                    "items": [
                        {"label": "负关键词过滤：无关消耗从 " + "wasted_pct" + "% → 3.2%", "key": "keyword_save"},
                        {"label": "冷启动 ROAS 提升 20%（保守估算）",                        "key": "roas_lift"},
                    ],
                },
            ],
        },
    },
    {
        "id": "pb-content-factory",
        "icon": "🎬",
        "name": "AI 内容工厂手册",
        "tag": "素材采集 · 视频生成 · 投放归因",
        "desc": "从人工内容团队到 AI 批量生产的 4 步迁移路线图，部分能力今日可用，视频生成接入中",
        "intro": "内容工厂的核心矛盾：进入多语言市场需要本地化内容，但人工拍摄成本高（单条 $2000+）、周期长（3-4周）、"
                 "无法批量。本手册提供分步落地路径——今天可用的部分立刻带来 ROI，实验阶段的部分提供未来路线图。",
        "steps": [
            {
                "step": "Step 1 【今日可用】素材智能采集",
                "problem": "内容创作前需要竞品素材、用户 VOC、爆品视频——人工采集耗时且遗漏率高",
                "skills": [
                    {"id": "Skill-Visual-Data-Collection", "why": "电商图文视频批量采集，为 AI 生成构建原材料库（ROI 380万）"},
                    {"id": "Skill-Review-Dedup-Quality-Filter", "why": "多平台评论去重净化，提取高质量用户 VOC 作为内容脚本素材（ROI 10-50万）"},
                ],
                "data": "需要：目标平台账号、竞品 ASIN/URL 列表",
                "output": "结构化素材库（图/视频/UGC 评论）",
            },
            {
                "step": "Step 2 【实验接入中】AI 批量内容生成",
                "problem": "进入德/日/韩市场需要本地化视频，人工方案成本 $2000+/条、周期 3-4 周",
                "skills": [
                    {"id": "Skill-AnchorCrafter-Virtual-Anchor-Demo", "why": "虚拟主播带货视频生成，HOI 交互保持商品真实感（ROI 50-100万）— 实验接入中"},
                    {"id": "Skill-DAWN-Talking-Head-Review", "why": "AI 口播 Review 视频，批量生成不同语言/形象的真人测评风格（ROI 30-60万）— 实验接入中"},
                    {"id": "Skill-Virbo-Multilingual-Avatar-UGC", "why": "多语言虚拟人 UGC 批量生产，100+ 语言 TTS + 对口型（ROI 35-60万）— 实验接入中"},
                    {"id": "Skill-Aquarius-Brand-Video-Generation", "why": "品牌营销视频生成，2B 参数模型，多主题批量（ROI 80-150万）— 实验接入中"},
                ],
                "data": "需要：品牌素材包（Logo/色调/产品图）、脚本模板、目标语言列表",
                "output": "批量本地化营销视频（预计 Q3 完整上线）",
            },
            {
                "step": "Step 3 【今日可用】内容质量评估",
                "problem": "AI 生成视频质量参差不齐，商品保真度和品牌一致性无客观评分标准",
                "skills": [
                    {"id": "Skill-E-Commerce-Video-Benchmark", "why": "电商域专用 Benchmark，PCF/LTP/MN 三维度量化评估，驱动工具选型与质检 SOP"},
                    {"id": "Skill-Creative-Fatigue-Detection", "why": "素材疲劳生命周期监测，生存分析识别 CTR/CVR 衰减，触发素材更新信号"},
                ],
                "data": "需要：已生成视频文件、历史广告表现数据",
                "output": "视频质量评分报告 + 素材更换优先级",
            },
            {
                "step": "Step 4 【今日可用】投放归因闭环",
                "problem": "不知道哪类内容真正带来了转化，内容创作无数据反馈，靠感觉迭代",
                "skills": [
                    {"id": "Skill-TikTok-Shop-Content-Attribution", "why": "TikTok Shop 短视频带货因果归因：识别哪类内容元素效率最高，指导下期制作"},
                    {"id": "Skill-DARA-Agentic-MMM-Optimizer", "why": "内容分发预算自动分配到最优渠道/时段（ROI 360-720万）"},
                ],
                "data": "需要：TikTok 广告数据 API、用户行为序列（曝光→点击→转化）",
                "output": "内容效果归因报告 + 下期制作方向 + 渠道预算自动分配",
            },
        ],
        "outcomes": [
            "Step 1+3+4 今天就能带来 ROI，Step 2 是未来的乘数",
            "本地化视频成本降低 80%，生产周期 3 周 → 3 天（Step 2 接入后）",
            "内容创作从「靠感觉迭代」升级为「数据驱动选题」",
            "素材库规模 10x，多语言市场同步覆盖",
        ],
    },
    {
        "id": "pb-pricing-engine",
        "icon": "💰",
        "name": "AI 定价引擎手册",
        "tag": "竞品监控 · 弹性估算 · 动态定价",
        "desc": "定价是乘数，广告是加法——A/B 实测 GMV +13%，定价科学化比多投广告更高效",
        "intro": "大多数跨境品牌的定价策略是「跟感觉」+「盯竞品手动调」。这里有两个系统性漏洞："
                 "① 你不知道哪些用户对价格敏感、哪些无论如何都会买；② 你不知道降价 $1 能多卖多少件。"
                 "本手册提供从竞品监控到动态定价的完整飞轮，每步都有可量化 ROI。",
        "steps": [
            {
                "step": "Step 1：竞品价格实时感知",
                "problem": "竞品降价后 47 分钟才响应，Buy Box 已丢失——每 30 分钟延迟损失 GMV 约 $1,600",
                "skills": [
                    {"id": "Skill-Price-Signal-Collection", "why": "竞品价格信号实时采集，Buy Box 获得率 41%→79%，响应延迟从 47 分钟降至 18 分钟（ROI 73.2万）"},
                    {"id": "Skill-Competitive-Price-Monitoring", "why": "因果竞争响应模型：量化「不跟降损失多少」，驱动有依据的响应决策（ROI 5-60万）"},
                ],
                "data": "需要：竞品 ASIN 列表、自身历史定价与销量数据",
                "output": "实时竞品价格监控面板 + 响应建议",
            },
            {
                "step": "Step 2：需求弹性估算",
                "problem": "「这个 SKU 降 $2 能多卖多少件」——没有弹性数据，定价靠猜",
                "skills": [
                    {"id": "Skill-Dynamic-Pricing-Elasticity", "why": "需求价格弹性估计 + 最优价格公式 P*=ε/(ε+1)·MC，利润率 +8-12%（ROI 50万）"},
                    {"id": "Skill-DML-Cohort-Causal-Effect", "why": "DML 分群弹性差异：7-12 月龄用户对价格敏感度是 0-3 月龄的 2.3 倍，精准定价（ROI 1500-2500万）"},
                ],
                "data": "需要：历史价格变动记录（至少 6 次调价）、同期销量数据",
                "output": "各 SKU 价格弹性曲线 + 最优定价区间",
            },
            {
                "step": "Step 3：动态定价执行",
                "problem": "人工定价最大缺陷：只能优化当前销量，无法同时考虑长期品牌溢价和库存健康",
                "skills": [
                    {"id": "Skill-AIGP-LLM-Dynamic-Pricing", "why": "LLM 跨周期 GMV 对齐定价，A/B 实测 GMV +13%，这是真实实验数据不是预测（ROI 1321万）"},
                    {"id": "Skill-Markdown-Optimization", "why": "清仓折扣优化：库存生命周期内最大化总回收价值，清仓多回收 15-40%（ROI 20-50万）"},
                    {"id": "Skill-Bundle-Pricing-Strategy", "why": "捆绑定价提升 AOV：吸奶器+法兰配件组合定价，配件复购率 +25%（ROI 10-15万）"},
                ],
                "data": "需要：SKU 成本结构、库存水位、竞品定价实时数据",
                "output": "动态定价策略 + 每日价格执行建议",
            },
            {
                "step": "Step 4：定价效果度量与迭代",
                "problem": "调价后「效果好不好」靠感觉判断，无法区分是定价作用还是季节波动",
                "skills": [
                    {"id": "Skill-DiD-Difference-in-Differences", "why": "双重差分估计调价因果效应：区分真实影响与自然波动（ROI 50万）"},
                    {"id": "Skill-Identified-Bayesian-MMM", "why": "分离价格效应与广告效应：告诉 CMO 预算浪费在哪里（ROI 1000万）"},
                    {"id": "Skill-Causal-Cohort-Analysis", "why": "促销长期 LTV 追踪：6 个月后用户复购是否真的提升了？（ROI 200-500万）"},
                ],
                "data": "需要：调价前后各 30 天销售数据、广告投放数据",
                "output": "调价效果因果报告 + 下一轮定价优化方向",
            },
        ],
        "outcomes": [
            "Buy Box 获得率从 41% 恢复至 79%，月均 GMV +$32,000",
            "AIGP 动态定价 A/B 实测 GMV +13%（真实实验，非预测）",
            "清仓效率提升，资金回收速度 +15-40%",
            "「定价是乘数，广告是加法」——精准定价 1% 比广告多投 15% 更高效",
        ],
    },
    {
        "id": "pb-risk-defense",
        "icon": "🛡️",
        "name": "跨境风险防御作战室",
        "tag": "欺诈反制 · 合规预警 · 品牌保护",
        "desc": "竞品用 AI 攻击你的广告、评分和排名——你需要 AI 来守门。封号预防 800 万/年 vs 防御投入 30 万",
        "intro": "跨境电商的竞争已经进入「AI 对抗」阶段。竞品在三条战线同时进攻："
                 "① 广告刷量消耗你的预算；② 虚假差评拉低你的评分；③ AI 注入攻击你的推荐排名。"
                 "与此同时，平台合规政策每季度更新，一次封号损失 30-80 万，一次召回损失 500 万+。"
                 "本手册提供从欺诈信号采集到合规预警的完整防御体系。",
        "steps": [
            {
                "step": "Step 1：欺诈信号采集与基线建立",
                "problem": "不知道自己正在被攻击——竞品刷评、广告 IVT、刷单行为悄无声息地侵蚀 ROI，发现时已损失数周",
                "skills": [
                    {"id": "Skill-Fraud-Signal-Collection",
                     "why": "主动采集刷单行为、虚假评论、异常流量信号，建立欺诈监控基线，同时向平台举报竞品（ROI 48万）"},
                    {"id": "Skill-Transaction-Anomaly-Detection",
                     "why": "Isolation Forest 检测异常交易模式（订单金额/IP/支付方式异常组合），拦截盗刷订单（ROI 3-8万/月）"},
                ],
                "data": "需要：订单流水、广告点击日志、评论数据（近 90 天）",
                "output": "欺诈信号基线报告 + 异常事件告警规则",
            },
            {
                "step": "Step 2：广告 IVT 实时过滤",
                "problem": "月广告 30 万，8% 是无效点击 = 2.4 万/月浪费。Bot 点击、竞品恶意点击无法靠平台自动过滤完全解决",
                "skills": [
                    {"id": "Skill-Click-Fraud-Detection",
                     "why": "时序异常 + 行为模式识别 IVT 攻击，向 FB/Google 申请退款，月均挽回 6-15 万"},
                    {"id": "Skill-Identity-Fraud-Detection",
                     "why": "设备+行为+网络三重验证账号欺诈，识别刷单账号防止 Amazon 卖家账户关联封号"},
                ],
                "data": "需要：广告平台 API（点击日志、IP、设备指纹）",
                "output": "IVT 过滤规则 + 月度退款申请报告",
            },
            {
                "step": "Step 3：Listing 评论生态保护",
                "problem": "新品上架 6 小时内被刷评团伙集中攻击；或竞品 ChatGPT 批量生成高质量虚假好评拉高自己",
                "skills": [
                    {"id": "Skill-Review-Fraud-Detection",
                     "why": "评论者-产品-评分关系图检测刷评团伙，向 Amazon 举报并保护自身 listing（ROI 5-15万/月）"},
                    {"id": "Skill-DS-DGA-GCN-Fake-Review-Group",
                     "why": "动态图 GCN 检测冷启动新品上架 6 小时内的刷评冲击，防止新品评分被操控"},
                    {"id": "Skill-FraudSquad-LLM-Review-Detection",
                     "why": "LM 嵌入 + 门控图变换器检测 ChatGPT 生成的高质量虚假好评（2025 年最新攻击方式）"},
                    {"id": "Skill-AIGC-Content-Detection",
                     "why": "鉴别 AI 生成内容，保护自身 Review 生态可信度，防止 VOC 分析被污染"},
                ],
                "data": "需要：Listing 评论历史、评论者行为序列",
                "output": "刷评团伙举报名单 + 评论质量评分 + 告警规则",
            },
            {
                "step": "Step 4：AI 推荐排名防御",
                "problem": "竞品在商品描述中嵌入恶意 Prompt 指令，劫持 AI 导购排名，某品牌自营商品曝光量下降 30-50%（2025 年真实攻击）",
                "skills": [
                    {"id": "Skill-Agent-Payment-Security-Red-Team",
                     "why": "检测 Prompt 注入攻击，保护 AI 推荐系统不被竞品操控（防御价值 > 5000万）"},
                    {"id": "Skill-MAS-Adversarial-Defense",
                     "why": "多智能体对抗防御，应对竞品协同攻击（多个假账号+多个被操控 listing 联合操作）"},
                ],
                "data": "需要：竞品 ASIN 列表、自身 AI 搜索排名监控数据",
                "output": "注入攻击检测告警 + 竞品操控行为报告",
            },
            {
                "step": "Step 5：合规预警与封号防御",
                "problem": "平台政策每季度更新，人工合规检查平均滞后 90 天，一次封号损失 30-80 万 GMV + BSR 排名恢复 2-6 周",
                "skills": [
                    {"id": "Skill-Regulatory-Change-Monitoring",
                     "why": "监管机构（FDA/CPSC/EU GPS）法规更新自动映射到受影响 SKU，提前 90 天预警"},
                    {"id": "Skill-Cross-Border-Compliance-Framework",
                     "why": "US+EU+UK 三维合规矩阵自动映射，新市场进入合规核查从 3 个月压缩到 2 周"},
                    {"id": "Skill-Product-Safety-Testing-Requirements",
                     "why": "品类×市场安全测试需求自动生成，选品阶段前置合规成本估算，避免选错品"},
                    {"id": "Skill-Compliance-Scored-Guardrail-Orchestration",
                     "why": "AI 生成 Listing 文案的合规门控（Best-of-N 评分），防止 ChatGPT 写出违规声明被 Amazon 下架"},
                ],
                "data": "需要：全部在售 SKU 信息、目标销售市场列表",
                "output": "合规风险评分矩阵 + 高风险 SKU 预警 + Listing 文案合规检测",
            },
            {
                "step": "Step 6：召回风险预测与供应链尽调",
                "problem": "被动等到消费者集中投诉才发现产品安全问题，主动召回 vs 被动召回成本相差 10 倍",
                "skills": [
                    {"id": "Skill-Consumer-Complaint-Recall-Prediction",
                     "why": "投诉信号驱动的召回风险预测，提前 12 个月预警，主动召回比被动召回节省 80% 成本"},
                    {"id": "Skill-Supply-Chain-Due-Diligence",
                     "why": "供应商劳工+环境+产品三维合规评估（德国供应链法 LkSG2023），满足欧洲 B2B 买家准入要求"},
                ],
                "data": "需要：客服工单历史、产品投诉数据、供应商信息",
                "output": "召回风险评分 + 供应商合规评级报告",
            },
        ],
        "outcomes": [
            "广告 IVT 从 8% 降至 2%，月均挽回 6-15 万",
            "竞品刷评攻击检测率 > 90%，Listing 评分保护",
            "合规检查从滞后 90 天→实时预警，封号风险降低 70%",
            "主动召回比被动召回节省 80% 成本，一次预防 = 500 万保障",
            "AI 推荐注入攻击防御：保护自然流量不被操控",
        ],
    },
    {
        "id": "pb-tariff-response",
        "icon": "⚡",
        "name": "关税冲击 72h 响应手册",
        "tag": "关税应对 · 定价重估 · 供应链转移",
        "desc": "关税涨 10 个点 = 利润腰斩。你有 72 小时决定怎么做——AI 给你完整行动清单",
        "intro": "2025 年跨境电商面临的最大不确定性：关税政策随时变动，汇率剧烈波动，每一次外部冲击都要求在"
                 "72 小时内同时回答五个问题：哪些 SKU 受影响？价格怎么调？库存怎么处置？"
                 "广告要不要暂停？供应链何时转移？没有 AI 的团队靠开会讨论，有 AI 的团队已经在执行了。",
        "steps": [
            {
                "step": "Step 1 【触发后 0-4h】冲击量化",
                "problem": "关税变动后，不知道哪些 SKU 利润率归零、哪些还有空间——凭感觉操作容易误伤优质品",
                "skills": [
                    {"id": "Skill-DML-Cohort-Causal-Effect",
                     "why": "DML 双机器学习分群估计：不同市场/用户群对关税引发的价格变化弹性差异（ROI 1500-2500万）"},
                    {"id": "Skill-Supply-Chain-Causal-SCM-Attribution",
                     "why": "供应链因果 SCM 根因归因：区分「关税直接影响」vs「市场自然波动」，避免把季节性下滑误归因于关税"},
                ],
                "data": "需要：全 SKU 成本结构（含关税比例）、近 6 个月销售数据、竞品价格",
                "output": "SKU 级利润率冲击矩阵（三种关税假设场景）",
            },
            {
                "step": "Step 2 【4-24h】定价响应决策",
                "problem": "吸收多少关税、转嫁多少给消费者——不同 SKU 的价格弹性差异巨大，统一处理必然错误",
                "skills": [
                    {"id": "Skill-Cross-Border-Price-Harmonization",
                     "why": "跨境价格协调：US/EU/JP 三市场同时调价时防止价差过大引发 Amazon 最低价政策违规（ROI 8-15万）"},
                    {"id": "Skill-AIGP-LLM-Dynamic-Pricing",
                     "why": "LLM 跨周期定价优化：考虑关税冲击期间品牌溢价保护 vs 短期销量最大化的权衡（A/B 实测 GMV +13%）"},
                ],
                "data": "需要：各 SKU 价格弹性估算、竞品实时价格监控数据",
                "output": "SKU 级定价调整方案（吸收 / 部分转嫁 / 全转嫁 三档建议）",
            },
            {
                "step": "Step 3 【24-48h】库存与广告决策",
                "problem": "现有库存是清仓套现还是维价等待？广告是暂停还是继续投？两个决策相互影响",
                "skills": [
                    {"id": "Skill-Markdown-Optimization",
                     "why": "库存生命周期清仓定价优化：关税冲击后高速清库存，多回收 15-40%（ROI 20-50万）"},
                    {"id": "Skill-Channel-Saturation-Curve",
                     "why": "渠道饱和曲线：价格波动期广告效率评估，判断是否应暂停广告等价格稳定后再恢复（ROI 18-25万）"},
                ],
                "data": "需要：各 SKU 库存水位、广告当前 ACOS/TACOS、历史清仓速度",
                "output": "库存处置方案（清仓/维价/转渠道）+ 广告预算调整建议",
            },
            {
                "step": "Step 4 【48-72h】供应链转移可行性评估",
                "problem": "「把订单转移到越南工厂」说起来容易，但不知道实际需要多长时间、成本差多少、合规认证能否复用",
                "skills": [
                    {"id": "Skill-Supplier-Capacity-Planning",
                     "why": "供应商产能规划：评估备选工厂（越南/墨西哥/印尼）的实际产能上限和爬坡周期"},
                    {"id": "Skill-Lead-Time-Distribution-Risk-GenQOT",
                     "why": "转移供应商后的交期分布重建：新工厂 lead time 从均值 30 天→不确定分布，动态安全库存防断货（ROI 200-500万/年）"},
                ],
                "data": "需要：备选供应商资质清单、现有认证（CPSC/CE）可复用性评估",
                "output": "供应链转移可行性报告（时间轴 + 成本差 + 合规风险）",
            },
        ],
        "outcomes": [
            "72 小时内输出完整行动清单，替代 3 天团队讨论",
            "定价响应精准化：弹性低的 SKU 维价，弹性高的提前清仓",
            "供应链转移决策有数据支撑：不靠直觉，知道转移到哪里、什么时候、成本差多少",
            "关税冲击期间广告预算不盲目暂停，基于饱和曲线做有依据的调整",
            "一次关税冲击响应提速 = 保护 3-6 个月 BSR 排名稳定",
        ],
    },
    {
        "id": "pb-voc-product-loop",
        "icon": "🔬",
        "name": "竞品情报→产品迭代加速器",
        "tag": "VOC挖掘 · 痛点归因 · 新品机会",
        "desc": "新品从洞察到上架 18 个月 → 6 个月。竞品差评是你最好的免费 R&D 数据",
        "intro": "你花 18 个月开发的新品，竞品 3 个月前就上了类似款，价格还低 30%。差距不在执行力，在情报速度。"
                 "90% 的跨境卖家只分析自家评论，而竞品的 1-3 星差评才是最高密度的产品洞察来源——"
                 "用户在告诉你「市场缺什么」，你只需要系统性地听。",
        "steps": [
            {
                "step": "Step 1：竞品差评多语言采集与净化",
                "problem": "手动分析竞品评论：人工处理 1 万条评论需 2-3 周，且只能做英语市场，德语/日语市场的洞察完全缺失",
                "skills": [
                    {"id": "Skill-Review-Pain-Point-Mining",
                     "why": "无监督竞品差评痛点挖掘，自动聚类「漏液」「噪音大」「难清洗」等产品缺陷维度（ROI 50-100万）"},
                    {"id": "Skill-Multilingual-NER-Universal-v2",
                     "why": "22 种语言命名实体识别，从德语/日语评论抽取「品牌/产品/症状」实体，覆盖非英语市场洞察"},
                    {"id": "Skill-Cultural-Data-Collection",
                     "why": "跨文化 UGC 采集：量化「美国妈妈要便利」vs「日本妈妈要安全」的消费偏好差异（ROI 280万）"},
                ],
                "data": "需要：竞品 ASIN 列表（1-3 星差评）、目标市场语言范围",
                "output": "多语言痛点矩阵（按功能维度聚类，按频次排序）",
            },
            {
                "step": "Step 2：痛点归因与产品差距识别",
                "problem": "知道「用户说漏液」还不够——漏液是哪个具体设计特征导致的？竞品有哪些功能是你没有的？",
                "skills": [
                    {"id": "Skill-AGRS-Aspect-Guided-Review-Summarization",
                     "why": "方面引导评论摘要：将「漏液投诉」归因到「密封圈设计/材质/安装方式」具体特征（ROI 1.5万/月）"},
                    {"id": "Skill-LACA-CrossLingual-ABSA",
                     "why": "跨语言方面级情感分析：同一功能在不同市场的情感极性对比，识别市场特有痛点（ROI 300-600万）"},
                ],
                "data": "需要：Step 1 输出的痛点矩阵、产品规格说明书",
                "output": "「功能差距矩阵」：竞品有/你没有的功能列表 + 各市场用户优先级排序",
            },
            {
                "step": "Step 3：洞察转化为产品需求",
                "problem": "从「用户说什么」到「供应商要改什么」之间有巨大鸿沟——数据团队的报告无法直接交给工厂",
                "skills": [
                    {"id": "Skill-StaR-Review-Statement-Ranking",
                     "why": "评论声明重要性排序：找出「最影响购买决策的 5 个改进点」，聚焦有限的产品迭代资源（ROI 80-150万/年）"},
                    {"id": "Skill-MAA-Review-to-Action-Decision",
                     "why": "多 Agent 评论→行动建议：自动生成供应商沟通文档（改良规格 + 测试要求），从洞察直达执行（ROI 510-920万/年）"},
                ],
                "data": "需要：Step 2 的功能差距矩阵、现有产品规格",
                "output": "可直接交工厂的「产品改良规格书」+ 优先级排序的改进清单",
            },
            {
                "step": "Step 4：新品上线后效果因果追踪",
                "problem": "改良版新品上架后，不知道销量提升是来自产品改进还是自然市场增长——下次无法复制成功",
                "skills": [
                    {"id": "Skill-DiD-Difference-in-Differences",
                     "why": "双重差分：对比改良品 vs 未改版品在同期的销量变化，量化产品迭代的真实因果效应（ROI 50万）"},
                ],
                "data": "需要：改良品和对照品的销售数据（上架前后各 30 天）",
                "output": "产品迭代 ROI 归因报告 + 下一轮迭代优先级建议",
            },
        ],
        "outcomes": [
            "竞品差评分析从 2-3 周人工 → 实时自动，覆盖 22 种语言",
            "新品开发周期从 18 个月压缩到 6 个月（情报速度提升 3x）",
            "新品成功率从 30% 提升到 50%，年增量 GMV 400万+",
            "产品迭代有数据追踪，每次改动的 ROI 可量化",
            "跨市场文化差异洞察：知道日本和德国要什么，不用靠猜",
        ],
    },
    {
        "id": "pb-customer-service-agent",
        "icon": "🎧",
        "name": "客服售后智能体手册",
        "tag": "多语言客服 · 退货优化 · 差评防御",
        "desc": "跨境客服三大成本：人力、时效、差评——AI 覆盖 70% 工单，差评率从 3% 降至 1.5%",
        "intro": "跨境客服有三个独特难点：① 时区跨度让响应时效极难保证（德国用户凌晨投诉，中国团队白天才看到）；"
                 "② 语言门槛让 80% 的卖家只能处理英语工单，德语/日语差评无人响应；"
                 "③ 平台惩罚机制让每一条差评都可能拉低 BSR 和转化率。"
                 "本手册提供从工单分流到退货闭环的完整 AI 客服体系，不需要人工值守即可 24h 响应。",
        "steps": [
            {
                "step": "Step 1：工单意图自动分类与路由",
                "problem": "日均 500 条工单，40% 是重复问题（物流时效/使用方法/退货流程）；人工分流每条耗时 3-5 分钟，高峰期积压严重",
                "skills": [
                    {"id": "Skill-DialIn-LLM-Case-Intent-Clustering",
                     "why": "无监督层次化意图聚类：自动发现客服意图树（退款/换货/咨询/投诉），无需人工标注，70% 工单自动路由（ROI 200-400万）"},
                    {"id": "Skill-Customer-Journey-Decision-Tree",
                     "why": "从历史日志自学决策树：70% 标准工单完全自动化处理，客服人力节省，释放人力专注高价值案例（ROI 600万）"},
                ],
                "data": "需要：近 6 个月客服工单历史（含工单文本、处理结果、处理时长）",
                "output": "意图分类体系 + 自动路由规则 + 工单自动回复模板",
            },
            {
                "step": "Step 2：多语言实时响应",
                "problem": "德语/日语差评无人响应，导致 Amazon 账号健康分下降；人工翻译后回复不够专业，文化语气不对",
                "skills": [
                    {"id": "Skill-Multilingual-Customer-Service-Translation",
                     "why": "多语言客服自动翻译与回复生成：覆盖德/日/法/西/葡语，A-to-Z 投诉文化适配回复，响应从 48h 压缩到 2h"},
                    {"id": "Skill-Emotional-AI-Customer-Care",
                     "why": "情感感知客服：高压场景（召回恐慌/宝宝安全问题）识别情绪强度，ANGRY/FRIGHTENED 时自动升级人工，避免 AI 误判激化矛盾"},
                ],
                "data": "需要：目标市场语言列表、历史回复模板、升级阈值配置",
                "output": "多语言 24h 自动回复 + 情绪升级告警",
            },
            {
                "step": "Step 3：差评根因归因与主动干预",
                "problem": "差评出现才处理是被动模式——研究表明 70% 的差评根因是可预防的（说明书不清晰/预期管理失败/物流时效误判）",
                "skills": [
                    {"id": "Skill-AGRS-Aspect-Guided-Review-Summarization",
                     "why": "方面引导评论摘要：将「宝宝用了皮肤发红」聚类到「材质/成分」问题，驱动产品改进而非仅回复差评（ROI 1.5万/月）"},
                    {"id": "Skill-Review-Pain-Point-Mining",
                     "why": "差评痛点挖掘：识别高频重复投诉点（「说明书看不懂」「充电口设计差」），生成预防性改进优先级（ROI 50-100万）"},
                    {"id": "Skill-LACA-CrossLingual-ABSA",
                     "why": "跨语言方面级情感分析：德语/日语差评的情感极性识别，跨市场差评根因对比（ROI 300-600万）"},
                ],
                "data": "需要：全部市场评论数据（含评分、语言、日期）",
                "output": "差评根因矩阵 + 高频问题改进清单 + 主动干预触发规则",
            },
            {
                "step": "Step 4：退货预测与欺诈拦截",
                "problem": "退货率 12%，其中约 35% 的 PayPal/信用卡纠纷（Chargeback）是欺诈性退货（INR 欺诈），人工无法区分",
                "skills": [
                    {"id": "Skill-Returns-Reverse-Logistics",
                     "why": "退货概率预测（XGBoost，按品类/价格/历史退货率）+ 退货处理路径优化（FBA退货 vs 海外仓 vs 销毁），年化 ROI 6-10万"},
                    {"id": "Skill-Logistics-Fraud-Detection",
                     "why": "物流链路欺诈检测：虚假收货/地址篡改/刷单物流识别，INR 欺诈从 35% 降至 5% 以下，月均挽回 $3,200+"},
                ],
                "data": "需要：订单数据、物流轨迹、历史退货记录、支付纠纷记录",
                "output": "退货风险评分 + 欺诈退货拦截规则 + 退货处理路径建议",
            },
        ],
        "outcomes": [
            "70% 标准工单全自动处理，客服人力节省 60%（ROI 600-800万/年）",
            "多语言覆盖：德/日/法/西/葡，响应时效从 48h → 2h",
            "差评率从 3% 降至 1.5%，对应转化率提升约 8-12%",
            "INR 欺诈退货从 35% 降至 5%，月均挽回 $3,200+",
            "差评根因转化为产品改进清单，形成「客服→R&D」反馈闭环",
        ],
    },
    {
        "id": "pb-fba-operations",
        "icon": "📦",
        "name": "FBA 运营全链路手册",
        "tag": "库存健康 · 头程优化 · 旺季备货",
        "desc": "FBA 仓储成本占 GMV 8-15%，库存周转天数中位数 95 天 vs 行业标杆 60 天——差距就是现金",
        "intro": "FBA 是跨境卖家最大的「看不见的成本中心」。长库龄费、移仓费、超量仓储费、头程成本……"
                 "每一项单独看都不大，加总起来可能吃掉全年利润的一半。"
                 "大多数卖家的 FBA 决策靠感觉和经验，本手册提供数据驱动的全链路优化：从头程路线选择，"
                 "到在库健康监控，到旺季备货计划，覆盖 FBA 运营的每一个成本节点。",
        "steps": [
            {
                "step": "Step 1：库存健康诊断与长库龄清仓",
                "problem": "长库龄（>180 天）SKU 每月额外收费，但「应该清仓还是降价促销还是转移到海外仓」没有系统性决策框架",
                "skills": [
                    {"id": "Skill-Inventory-Health-Aging-Attribution",
                     "why": "业务指标驱动的库存预测：按库存健康分层，识别「高风险滞销品」并给出清仓 vs 维价 vs 转仓的量化建议"},
                    {"id": "Skill-Markdown-Optimization",
                     "why": "清仓折扣优化：长库龄 SKU 在剩余库存寿命内最大化回收价值，多回收 15-40%（ROI 20-50万）"},
                ],
                "data": "需要：全 SKU 库存年龄报告、销售速度（Sales Velocity）、FBA 仓储费率",
                "output": "SKU 库存健康评分 + 清仓/维价/转仓优先级清单",
            },
            {
                "step": "Step 2：需求预测驱动补货计划",
                "problem": "月底补货计划靠拍脑袋：要么缺货（BSR 排名跌落，广告 ACOS 飙升），要么积压（长库龄费开始计算）",
                "skills": [
                    {"id": "Skill-Hierarchical-Demand-Forecasting-Reconciliation",
                     "why": "分层预测调和：SKU→ASIN→市场三层预测数学保证一致，月度补货计划自动生成（ROI 800-1500万）"},
                    {"id": "Skill-Promotion-Logistics-Surge-Forecast",
                     "why": "大促物流爆仓预测：Prime Day/黑五前 3-7 天预测 FBA 入库峰值，提前锁定仓位防爆仓（ROI 20-40万）"},
                ],
                "data": "需要：历史销售数据（90天+）、大促日历、当前库存水位",
                "output": "月度 FBA 补货计划 + 大促前置入库时间表",
            },
            {
                "step": "Step 3：头程路线成本优化",
                "problem": "海运 vs 空运 vs 铁运的选择靠货代报价，不知道「多花 X 美元空运」是否值得——没有时效×成本的系统性对比框架",
                "skills": [
                    {"id": "Skill-Cross-Border-Logistics-Routing",
                     "why": "多式联运帕累托最优路径：成本/时效/碳排放三目标同时优化，识别「当前库存水位下空运是否合算」（ROI 30-50万）"},
                    {"id": "Skill-Lead-Time-Distribution-Risk-GenQOT",
                     "why": "头程交期分布建模：海运延误不是均值问题而是分布问题，动态安全库存防止「准时到港但仍然断货」（ROI 200-500万/年）"},
                ],
                "data": "需要：历史头程时效数据（按货代/路线/季节）、当前各 SKU 安全库存",
                "output": "头程路线推荐（当前最优选择 + 成本差分析）+ 动态安全库存设定",
            },
            {
                "step": "Step 4：多渠道库存调拨",
                "problem": "Amazon FBA 库存充足，但独立站缺货；或者 FBA 某仓库积压，另一仓库缺货——渠道间库存无法实时平衡",
                "skills": [
                    {"id": "Skill-Multi-Channel-Inventory-Pooling",
                     "why": "跨渠道动态调拨：Amazon FBA + 独立站 + TikTok Shop 库存实时平衡，防止单渠道缺货同时另一渠道积压（ROI 200-400万）"},
                    {"id": "Skill-Returns-Reverse-Logistics",
                     "why": "退货库存再利用：FBA 退货品质检后按「可售/翻新/销毁」三路分流，退货库存回收率提升（ROI 6-10万）"},
                ],
                "data": "需要：各渠道实时库存、销售速度、调拨成本矩阵",
                "output": "跨渠道库存调拨指令 + 退货品分流建议",
            },
        ],
        "outcomes": [
            "库存周转天数从 95 天降至 65 天，释放压占资金 200-400万",
            "长库龄费支出减少 60%（提前清仓 + 补货节奏优化）",
            "头程成本优化 15-25%（路线选择 + 安全库存精准设定）",
            "大促前不爆仓：Prime Day/黑五提前 7 天完成入库",
            "跨渠道缺货率从 8% → 2%，保护 BSR 排名稳定",
        ],
    },
]


def render_roadmap_page(skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    """CEO-facing AI capability roadmap whitepaper. Designed for B2B sales, print-ready via @media print."""

    PHASES = [
        {
            "id": "phase1",
            "label": "Phase 1",
            "period": "第 1-3 个月",
            "theme": "快赢",
            "color": "#2563eb",
            "bg": "#eff6ff",
            "tagline": "30 天内出数字，首月可见 ROI",
            "roi": "800 - 1,900 万/年",
            "items": [
                {
                    "icon": "📦",
                    "title": "供应链预测基线",
                    "story": "某母婴品牌 60+ SKU × 多仓 × 多市场，每月底 2-3 名 PM 纯人工对账三层预测数字——SKU 求和对不上仓库数，仓库数对不上市场总量，相差最高 50%。",
                    "result": "HiFoReAd 分层调和后：对账人力清零，补货计划冲突率降至 < 5%",
                    "roi": "800-1,500 万/年",
                    "skills": ["Skill-Hierarchical-Demand-Forecasting-Reconciliation", "Skill-Demand-Forecasting-Supply-Chain"],
                },
                {
                    "icon": "🤖",
                    "title": "客服智能路由 + 70% 工单自动化",
                    "story": "某品牌日均 5 万条跨领域工单，人工路由正确率 61%，每天约 1 万条工单需二次转单；新手妈妈咨询「宝宝 3 月夜醒频繁」，人工客服响应平均 72 小时，且有医疗合规风险。",
                    "result": "AgentRouter 路由正确率 61% → 82%；客服决策树从历史日志自学，70% 工单实现自动化处理，年节省运营成本",
                    "roi": "1,900 万/年（路由）+ 600 万/年（自动化）",
                    "skills": ["Skill-AgentRouter-KG-Guided", "Skill-Customer-Journey-Decision-Tree"],
                },
                {
                    "icon": "🎯",
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
            "theme": "数据基础设施",
            "color": "#7c3aed",
            "bg": "#f5f3ff",
            "tagline": "让 Phase 1 的效果可持续、可复制",
            "roi": "200 - 500 万/年（新增）",
            "items": [
                {
                    "icon": "🗄️",
                    "title": "产品知识图谱",
                    "story": "AI 在没有结构化产品知识的情况下，给用户推荐「买了吸奶器的用户还需要什么」——它不知道硅胶法兰和乳头霜属于哺乳期刚需配件，推荐准确率极低。",
                    "result": "构建产品 KG 后：KGQA 查询召回率 52% → 92%，跨品类推荐 CTR +18%",
                    "roi": "20-35 万/年（推荐层增量）",
                    "skills": ["Skill-Hierarchical-Product-KG-Construction", "Skill-Ontology-Schema-Design"],
                },
                {
                    "icon": "🔬",
                    "title": "A/B 实验平台",
                    "story": "某品牌每次调整定价策略或上新 listing，无法区分效果是真实改进还是季节波动。团队争论持续数周，决策依赖「感觉」。",
                    "result": "Switchback 实验体系搭建后：物流/双边市场实验可信，决策从争论变为数据裁决",
                    "roi": "1,500 万/年（错误决策避免）",
                    "skills": ["Skill-Switchback-Experiment-Design", "Skill-CUPED-Variance-Reduction"],
                },
                {
                    "icon": "📊",
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
            "theme": "差异化护城河",
            "color": "#059669",
            "bg": "#f0fdf4",
            "tagline": "竞争对手需要 18 个月才能追上这里",
            "roi": "5,000 万+ 潜力（战略级）",
            "items": [
                {
                    "icon": "🏷️",
                    "title": "AI 定价引擎",
                    "story": "某品牌大促前手动跟价——降太多伤利润，降太少丢份额。运营靠经验感知，每次大促前都是高压决策，无法同时优化当前销量和品牌长期溢价。",
                    "result": "AIGP 动态定价 A/B 实测：GMV +13%，实验数据非预测值",
                    "roi": "1,321 万/年（A/B 实测）",
                    "skills": ["Skill-AIGP-LLM-Dynamic-Pricing", "Skill-Dynamic-Pricing-Elasticity"],
                },
                {
                    "icon": "🎬",
                    "title": "AI 内容工厂",
                    "story": "某品牌进入德国/日本市场，需要本地化口播 Review 视频。人工方案：雇本地 KOL 拍摄，周期 3-4 周，单条成本 $2,000+。批量生产 20 个 SKU 的测评视频需要 6 个月预算。",
                    "result": "Virbo 多语言虚拟人：同等内容量成本降低 80%，生产周期 3 周→ 3 天（实验接入中）",
                    "roi": "35-150 万/年（视接入程度）",
                    "skills": ["Skill-Virbo-Multilingual-Avatar-UGC", "Skill-AnchorCrafter-Virtual-Anchor-Demo"],
                },
                {
                    "icon": "⚡",
                    "title": "MAS 多智能体联动",
                    "story": "大促首日某品牌吸奶器打 7 折卖出 5,000 件，第 3 天库存告急被迫涨价，剩余 7 天流量白白浪费——整个大促周期总利润反而低于平销期。根因：定价和补货是两个团队各自决策，无法实时联动。",
                    "result": "FSDA-DRL 快慢双 Agent：定价与补货实时联动，大促周期利润最优化",
                    "roi": "5,000 万/年（规模依赖）",
                    "skills": ["Skill-FSDA-DRL", "Skill-Event-Driven-Demand-MAS"],
                },
                {
                    "icon": "🛡️",
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
      <span class="rm-result-label">✅ 结果</span>
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
<div class="rm-hero">
  <div class="rm-hero-eyebrow">AI 能力建设路线图 · 2025-2026</div>
  <h1 class="rm-hero-title">12 个月，让 AI 替代 3 类岗位的重复性决策</h1>
  <p class="rm-hero-sub">分三阶段部署，首月可见 ROI，全年可验证收益 > 3,000 万元</p>
  <div class="rm-hero-cta">
    <button class="rm-btn-primary" onclick="window.print()">⬇ 下载 PDF</button>
    <a class="rm-btn-sec" href="playbooks/index.html">查看场景手册 →</a>
  </div>
  <p class="rm-hero-note">所有 ROI 数字来源于真实 A/B 实验或匿名客户案例，非模型预测</p>
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
    <span class="rm-role-icon">📦</span>
    <div>
      <strong>供应链全链路</strong>
      <p>从 15 个 Excel 联动 → 1 个 MAS 自动执行</p>
      <span class="rm-role-roi">ROI 5,000-8,000 万</span>
    </div>
  </div>
  <div class="rm-role">
    <span class="rm-role-icon">📊</span>
    <div>
      <strong>数据分析师</strong>
      <p>提数从 72 小时 → 5 分钟，报告从做 → 审</p>
      <span class="rm-role-roi">ROI 1,600-3,000 万</span>
    </div>
  </div>
  <div class="rm-role">
    <span class="rm-role-icon">📈</span>
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
       <a href="playbooks/pb-risk-defense.html">🛡️ 跨境风险防御作战室</a>
       <a href="playbooks/pb-agent-replace.html">🤖 AI Agent 替人手册</a>
       <a href="playbooks/pb-tariff-response.html">⚡ 关税冲击 72h 响应</a>
       <a href="playbooks/pb-voc-product-loop.html">🔬 竞品情报→产品迭代</a>
       <a href="playbooks/pb-customer-service-agent.html">🎧 客服售后智能体</a>
       <a href="playbooks/pb-fba-operations.html">📦 FBA 运营全链路</a>
       <a href="playbooks/pb-pricing-engine.html">💰 AI 定价引擎手册</a>
       <a href="playbooks/pb-inventory-festival.html">🎯 大促备货决策手册</a>
     </div>
  </div>
  <div class="rm-footer-right">
    <div class="rm-footer-cta">
      <p>获取完整 PDF 版本</p>
      <button class="rm-btn-primary" onclick="window.print()">⬇ 下载 PDF</button>
    </div>
    <p class="rm-footer-note">
      数据来源：348 个从顶会论文萃取的业务 Skills，包含真实 A/B 实验与匿名客户案例。<br>
      所有案例均已脱敏处理，以「某跨境母婴品牌」表述。
    </p>
  </div>
</div>
"""

    rm_css = """
<style>
.rm-hero{text-align:center;padding:60px 40px 40px;max-width:860px;margin:0 auto}
.rm-hero-eyebrow{font-size:13px;font-weight:600;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.rm-hero-title{font-size:36px;font-weight:800;line-height:1.2;margin:0 0 16px;color:#0f172a}
.rm-hero-sub{font-size:18px;color:#475569;margin:0 0 28px}
.rm-hero-cta{display:flex;gap:12px;justify-content:center;margin-bottom:14px}
.rm-hero-note{font-size:12px;color:var(--muted)}
.rm-btn-primary{padding:12px 28px;background:#2563eb;color:#fff;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block}
.rm-btn-primary:hover{background:#1d4ed8}
.rm-btn-sec{padding:12px 28px;background:#f1f5f9;color:#334155;border-radius:10px;font-size:15px;font-weight:600;text-decoration:none;display:inline-block}

.rm-summary-bar{display:flex;align-items:center;justify-content:center;gap:0;background:#0f172a;color:#fff;padding:24px 40px;margin:0 -36px;flex-wrap:wrap;gap:8px}
.rm-summary-item{text-align:center;padding:0 24px}
.rm-summary-num{display:block;font-size:28px;font-weight:800;color:#60a5fa}
.rm-summary-label{font-size:12px;color:#94a3b8}
.rm-summary-sep{font-size:24px;color:#334155;padding:0 8px}

.rm-roles-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0;padding:0}
.rm-role{display:flex;gap:16px;align-items:flex-start;background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:20px}
.rm-role-icon{font-size:32px;flex-shrink:0}
.rm-role strong{display:block;font-size:15px;margin-bottom:4px}
.rm-role p{margin:0 0 8px;font-size:13px;color:#64748b}
.rm-role-roi{font-size:13px;font-weight:700;color:#2563eb;background:#eff6ff;padding:3px 10px;border-radius:999px}

.rm-phases{display:flex;flex-direction:column;gap:24px;margin:32px 0}
.rm-phase{border-radius:16px;overflow:hidden;border:1px solid #e2e8f0}
.rm-phase-header{display:flex;align-items:flex-start;gap:20px;padding:24px 28px;background:var(--phase-bg);border-bottom:1px solid #e2e8f0}
.rm-phase-badge{width:72px;height:72px;border-radius:50%;background:var(--phase-color);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;flex-shrink:0;text-align:center;line-height:1.2}
.rm-phase-meta{flex:1}
.rm-phase-period{font-size:12px;font-weight:600;color:var(--phase-color);text-transform:uppercase;letter-spacing:.06em}
.rm-phase-theme{margin:4px 0;font-size:22px;font-weight:800}
.rm-phase-tagline{margin:0;font-size:14px;color:#64748b}
.rm-phase-roi{text-align:right;flex-shrink:0}
.rm-phase-roi-label{display:block;font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}
.rm-phase-roi strong{font-size:18px;font-weight:800;color:var(--phase-color)}

.rm-items{display:flex;flex-direction:column;gap:0}
.rm-item{display:flex;gap:20px;padding:24px 28px;border-bottom:1px solid #f1f5f9}
.rm-item:last-child{border-bottom:none}
.rm-item-icon{font-size:28px;flex-shrink:0;width:40px;text-align:center;margin-top:2px}
.rm-item-body{flex:1}
.rm-item-title{margin:0 0 12px;font-size:16px;font-weight:700}
.rm-story{background:#fefce8;border-left:3px solid #f59e0b;padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;color:#78350f;margin-bottom:8px}
.rm-story-label,.rm-result-label{font-weight:700;margin-right:6px}
.rm-result{background:#f0fdf4;border-left:3px solid #10b981;padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;color:#14532d;margin-bottom:8px}
.rm-roi-line{font-size:13px;margin-bottom:8px;color:#374151}
.rm-chips{display:flex;flex-wrap:wrap;gap:6px}
.rm-chip{font-size:11px;background:#eff6ff;color:#1e40af;padding:3px 10px;border-radius:999px;text-decoration:none;border:1px solid #bfdbfe}
.rm-chip:hover{background:#dbeafe}

.rm-footer{display:grid;grid-template-columns:1fr 340px;gap:40px;margin-top:40px;padding:36px;background:#0f172a;border-radius:16px;color:#e2e8f0}
.rm-footer h3{margin:0 0 10px;font-size:18px;color:#fff}
.rm-footer p{font-size:14px;color:#94a3b8;margin:0 0 16px}
.rm-footer-links{display:flex;flex-direction:column;gap:8px}
.rm-footer-links a{color:#60a5fa;text-decoration:none;font-size:14px}
.rm-footer-links a:hover{text-decoration:underline}
.rm-footer-right{display:flex;flex-direction:column;justify-content:space-between}
.rm-footer-cta{text-align:center;background:#1e293b;border-radius:12px;padding:24px;margin-bottom:16px}
.rm-footer-cta p{color:#94a3b8;font-size:13px;margin-bottom:12px}
.rm-footer-note{font-size:11px;color:#475569;line-height:1.6}

@media print {
  .topbar,.sidebar,.rm-hero-cta,.rm-footer-cta button{display:none!important}
  body{background:#fff}
  .content{padding:0!important;max-width:100%!important}
  .rm-summary-bar{margin:0!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .rm-phase,.rm-footer{break-inside:avoid}
  .rm-phases{gap:16px}
  @page{margin:20mm 15mm;size:A4}
}
@media(max-width:900px){
  .rm-roles-bar{grid-template-columns:1fr}
  .rm-footer{grid-template-columns:1fr}
  .rm-phase-header{flex-wrap:wrap}
}
</style>"""

    full_body = rm_css + body
    return html_page("AI 能力建设路线图", full_body)


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
                roi = f"<span class='roi-badge'>💰 {html.escape(sk.roi_figure)}</span>" if sk.roi_figure else ""
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
        steps_html += f"""
<div class='pb-step'>
  <div class='pb-step-num'>Step {i}</div>
  <div class='pb-step-body'>
    <h3 class='pb-step-title'>{html.escape(step['step'])}</h3>
    <p class='pb-problem'>❓ {html.escape(step['problem'])}</p>
    <div class='pb-skills'>{skills_html}</div>
    {'<div class="pb-data"><strong>所需数据：</strong>' + data_req + '</div>' if data_req else ''}
    {'<div class="pb-output"><strong>输出结果：</strong>' + output + '</div>' if output else ''}
  </div>
</div>"""

    outcomes = "".join(f"<li>✅ {html.escape(o)}</li>" for o in pb.get("outcomes", []))
    calc_html = _render_roi_calculator(pb.get("roi_calculator")) if pb.get("roi_calculator") else ""
    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../playbooks/index.html">场景手册</a> / {html.escape(pb['name'])}</nav>
<div class='pb-hero'>
  <span class='pb-icon'>{pb['icon']}</span>
  <div>
    <h1>{html.escape(pb['name'])}</h1>
    <p class='lead'>{html.escape(pb['desc'])}</p>
    <span class='biz-tag'>{html.escape(pb['tag'])}</span>
  </div>
</div>
<div class='pb-intro'>{html.escape(pb['intro'])}</div>
{'<div class="wf-outcomes"><h3>预期收益</h3><ul>' + outcomes + '</ul></div>' if outcomes else ''}
<div class='pb-steps'>{steps_html}</div>
{calc_html}
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
        f"<li>✅ {html.escape(o)}</li>" for o in outcomes
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

def html_page(title: str, body: str, nav: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)} · paper2skills Playbook</title>
  <link rel="stylesheet" href="{nav}assets/style.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="{nav}index.html">paper2skills Playbook</a>
    <nav class="topnav">
      <a href="{nav}index.html">首页</a>
      <a href="{nav}domains/index.html">领域</a>
      <a href="{nav}topics/index.html">主题</a>
      <a href="{nav}workflows/index.html">工作流</a>
      <a href="{nav}playbooks/index.html">场景手册</a>
      <a href="{nav}ai-roadmap.html" style="color:#f59e0b;font-weight:700">🗺 AI 路线图</a>
      <a href="{nav}graph/overview.html">图谱</a>
      <a href="{nav}skills/index.html">全部</a>
    </nav>
    <input id="global-search" placeholder="搜索 Skill / 场景 / 算法..." autocomplete="off">
  </header>
  <div id="search-results" class="search-results hidden"></div>
  <main class="layout">
    <aside class="sidebar">
      <a href="{nav}index.html">总览</a>
      <a href="{nav}ai-roadmap.html">🗺 AI 能力路线图</a>
      <a href="{nav}domains/index.html">按领域</a>
      <a href="{nav}topics/index.html">按主题</a>
      <a href="{nav}workflows/index.html">工作流</a>
      <a href="{nav}playbooks/index.html">场景手册</a>
      <a href="{nav}graph/overview.html">Skills Graph</a>
      <a href="{nav}skills/index.html">全部 Skills</a>
    </aside>
    <section class="content">{body}</section>
  </main>
  <script src="{nav}assets/playbook-data.js"></script>
  <script src="{nav}assets/search.js"></script>
</body>
</html>"""


def skill_url(skill_id: str, nav: str = "") -> str:
    return f"{nav}skills/{skill_id}.html"


def render_skill_card(skill: PlaybookSkill, nav: str = "") -> str:
    tags = "".join(f"<span class='tag'>{html.escape(tag)}</span>" for tag in skill.tags)
    topics = "".join(f"<span class='tag topic'>{html.escape(t)}</span>" for t in skill.topics)
    roi_badge = (
        f"<span class='roi-badge'>💰 {html.escape(skill.roi_figure)}</span>"
        if skill.roi_figure else ""
    )
    diff_badge = (
        f"<span class='diff-badge'>{html.escape(skill.difficulty)}</span>"
        if skill.difficulty else ""
    )
    return f"""<article class="card skill-card">
  <h3><a href="{skill_url(skill.skill_id, nav)}">{html.escape(skill.title)}</a></h3>
  <p class="muted">{html.escape(skill.domain_dir)}</p>
  <p>{html.escape(skill.problem_solved or skill.algorithm_summary)}</p>
  <div class="card-badges">{roi_badge}{diff_badge}</div>
  <div>{tags}{topics}</div>
</article>"""


def link_list(items: list[str], nav: str = "") -> str:
    if not items:
        return "<p class='muted'>暂无</p>"
    rows = []
    for item in items:
        escaped = html.escape(item)
        if item in KNOWN_SKILL_IDS:
            rows.append(f"<li><a href='{skill_url(item, nav)}'>{escaped}</a></li>")
        else:
            rows.append(f"<li><span class='muted'>{escaped}</span></li>")
    return "<ul>" + "".join(rows) + "</ul>"


def render_skill_page(skill: PlaybookSkill) -> str:
    nav = "../"

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
            parts.append(f"<div class='roi-item'><span class='roi-label'>年化 ROI</span><span class='roi-value'>💰 {html.escape(skill.roi_figure)}</span></div>")
        if skill.difficulty:
            parts.append(f"<div class='roi-item'><span class='roi-label'>实现难度</span><span class='roi-value'>{html.escape(skill.difficulty)}</span></div>")
        if skill.priority:
            parts.append(f"<div class='roi-item'><span class='roi-label'>业务优先级</span><span class='roi-value'>{html.escape(skill.priority)}</span></div>")
        roi_meta = "<div class='roi-panel'>" + "".join(parts) + "</div>"

    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../domains/{slugify(skill.domain_dir)}.html">{html.escape(skill.domain_dir)}</a> / {html.escape(skill.skill_id)}</nav>
<h1>{html.escape(skill.title)}</h1>
<p class="muted">{html.escape(skill.skill_id)} · {html.escape(skill.domain_dir)}</p>
<div class="tag-row">{''.join(f"<span class='tag'>{html.escape(t)}</span>" for t in skill.tags + skill.topics + skill.workflows)}</div>
{roi_meta}
<div class="two-col">
  <section>
    <h2>1. 解决的问题</h2>
    <p>{html.escape(skill.problem_solved or skill.algorithm_summary)}</p>
    <h2>2. 核心算法逻辑</h2>
    <p>{html.escape(skill.algorithm_summary)}</p>
    <h2>3. 业务应用场景</h2>
    {scenario_html}
    <h2>4. 输入数据要求</h2>{render_items(skill.inputs) if skill.inputs else "<p class='muted'>请查看原始代码模板获取输入规格。</p>"}
    <h2>5. 输出结果</h2>{render_items(skill.outputs) if skill.outputs else "<p class='muted'>请查看原始代码模板获取输出规格。</p>"}
    <h2>6. 业务价值 / ROI</h2>{render_items(skill.roi) if skill.roi else ("<p>" + html.escape(skill.roi_figure) + "</p>" if skill.roi_figure else "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>")}
    <h2>7. 代码模板</h2>
    <p class="muted">代码块数量：{skill.code_blocks} · 路径：{html.escape(skill.code_path or '未检测到')}</p>
    {_render_code_preview(skill.code_preview)}
    <h2>8. 论文来源</h2>{render_items(skill.papers)}
  </section>
  <aside class="relation-panel">
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
<script src="../assets/ego-graph.js"></script>"""
    return html_page(skill.title, body, nav)


def render_items(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>"
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def _render_code_preview(code: str) -> str:
    if not code:
        return "<p class='muted'>请查看原始 Skill 卡片获取完整代码。</p>"
    escaped = html.escape(code)
    return f"<pre class='code-preview'><code>{escaped}</code></pre>"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Index page (Phase 3C — three-audience redesign)
# ---------------------------------------------------------------------------

def render_index(skill_count: int, domain_count: int, edge_count: int, domains: list[dict[str, Any]], skills: list[PlaybookSkill]) -> str:
    domain_cards = "".join(
        f"<a class='metric-card domain-card' href='domains/{slugify(d['vault_dir'])}.html'>"
        f"<strong>{html.escape(d['vault_dir'])}</strong>"
        f"<span>{d.get('skill_count', 0)} Skills</span></a>"
        for d in domains
    )

    # Top 5 skills by relation count (degree centrality proxy)
    skill_degree = {s.skill_id: len(s.relations.get("prerequisite", [])) + len(s.relations.get("combinable", [])) for s in skills}
    hot_skills = sorted(skills, key=lambda s: skill_degree.get(s.skill_id, 0), reverse=True)[:5]
    hot_items = "".join(
        f"<li><a href='skills/{s.skill_id}.html'>{html.escape(s.title)}</a>"
        f"{'<span class=roi-badge>💰 ' + html.escape(s.roi_figure) + '</span>' if s.roi_figure else ''}</li>"
        for s in hot_skills
    )

    business_cards = "".join(
        f"<a class='biz-card' href='{e['href']}'>"
        f"<span class='biz-icon'>{e['icon']}</span>"
        f"<div class='biz-body'>"
        f"<strong>{html.escape(e['label'])}</strong>"
        f"<p>{html.escape(e['desc'])}</p>"
        f"</div>"
        f"<span class='biz-tag'>{html.escape(e['tag'])}</span>"
        f"</a>"
        for e in BUSINESS_ENTRIES
    )

    return f"""
<div class="hero">
  <h1>paper2skills Playbook</h1>
  <p class="lead">面向母婴跨境电商的 AI 决策技能库：348 个从顶会论文萃取的可落地 Skills，覆盖广告、供应链、用户增长、智能体工程等 22 个领域。</p>
  <div class="hero-tabs" id="heroTabs">
    <button class="tab-btn active" data-tab="biz">🎯 业务专家 / 运营</button>
    <button class="tab-btn" data-tab="ds">📊 数据科学家</button>
    <button class="tab-btn" data-tab="ceo">🗺 CEO / 决策层</button>
    <button class="tab-btn" data-tab="explore">📂 自由探索</button>
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
      <h3>🔥 高连接度 Skills</h3>
      <p class="muted">被最多 Skill 依赖的核心算法，学习回报最高。</p>
      <ul class="hot-list">{hot_items}</ul>
    </div>
    <div class="ds-card">
      <h3>📐 按算法类型</h3>
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
      <h3>🕸️ Skills Graph</h3>
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
      <a class="btn-primary" href="ai-roadmap.html" onclick="window.open('ai-roadmap.html','_blank').print();return false;" style="margin-left:8px;background:#475569">⬇ 下载 PDF</a>
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
    <div><strong>{skill_count}</strong><span>Skills</span></div>
    <div><strong>{domain_count}</strong><span>领域</span></div>
    <div><strong>{edge_count}</strong><span>关系边</span></div>
    <div><strong>5</strong><span>工作流</span></div>
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

def render_graph_page(skill_count: int, edge_count: int) -> str:
    body = f"""
<h1>Skills Graph</h1>
<p class="muted">节点 {skill_count} · 边 {edge_count}　　点击节点查看详情，悬停高亮邻居，滚轮缩放。</p>
<div class="graph-controls">
  <label><input type="checkbox" id="cb-prerequisite" checked> <span class="edge-dot prereq"></span> 前置 (prerequisite)</label>
  <label><input type="checkbox" id="cb-combinable" checked> <span class="edge-dot combo"></span> 可组合 (combinable)</label>
  <label><input type="checkbox" id="cb-extension"> <span class="edge-dot ext"></span> 延伸 (extension)</label>
  <input id="graph-search" placeholder="搜索节点..." style="margin-left:16px;padding:6px 10px;border:1px solid #e5e7eb;border-radius:8px;width:200px">
</div>
<div id="graph-info" class="graph-info hidden">
  <button id="graph-info-close" style="float:right;background:none;border:none;cursor:pointer;font-size:18px">✕</button>
  <h3 id="gi-title"></h3>
  <p id="gi-domain" class="muted"></p>
  <p id="gi-summary"></p>
  <a id="gi-link" href="#" class="btn-primary" target="_self">查看详情 →</a>
</div>
<svg id="graph-svg"></svg>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="../assets/graph.js"></script>
"""
    return html_page("Skills Graph", body, "../")


def build_ego_graph_js() -> str:
    """Ego graph for Skill detail pages: renders 1-hop neighbourhood in the relation panel."""
    return r"""
(function () {
  const svg = document.getElementById('ego-graph');
  if (!svg || typeof d3 === 'undefined') return;
  const centerId = svg.dataset.skill;
  if (!centerId) return;

  const W = +svg.getAttribute('width')  || 280;
  const H = +svg.getAttribute('height') || 220;

  function load(cb) {
    if (window._EGO_DATA) { cb(window._EGO_DATA); return; }
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '../assets/graph-data.json');
    xhr.onload = () => {
      try { window._EGO_DATA = JSON.parse(xhr.responseText); cb(window._EGO_DATA); }
      catch (e) { cb(null); }
    };
    xhr.onerror = () => cb(null);
    xhr.send();
  }

  load(function (raw) {
    if (!raw) return;

    const edgeCfg = {
      prerequisite: '#3b82f6',
      combinable:   '#10b981',
      extension:    '#f59e0b',
    };

    const neighborIds = new Set([centerId]);
    const egoLinks = raw.links.filter(l => {
      if (l.source === centerId || l.target === centerId) {
        neighborIds.add(l.source);
        neighborIds.add(l.target);
        return true;
      }
      return false;
    });

    if (neighborIds.size <= 1) {
      d3.select(svg).append('text')
        .attr('x', W / 2).attr('y', H / 2)
        .attr('text-anchor', 'middle').attr('fill', '#9ca3af').attr('font-size', 12)
        .text('无关联 Skill');
      return;
    }

    const egoNodes = raw.nodes
      .filter(n => neighborIds.has(n.id))
      .map(n => ({ ...n }));

    const sel = d3.select(svg).attr('viewBox', `0 0 ${W} ${H}`);

    const sim = d3.forceSimulation(egoNodes)
      .force('link', d3.forceLink(egoLinks.map(l => ({ ...l }))).id(d => d.id).distance(65).strength(0.6))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(18));

    const linkEl = sel.append('g').selectAll('line')
      .data(egoLinks)
      .join('line')
      .attr('stroke', d => edgeCfg[d.type] || '#94a3b8')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.7);

    const nodeEl = sel.append('g').selectAll('g')
      .data(egoNodes)
      .join('g')
      .attr('cursor', d => d.id === centerId ? 'default' : 'pointer')
      .on('click', (e, d) => {
        if (d.id !== centerId) window.location.href = `${d.id}.html`;
      });

    nodeEl.append('circle')
      .attr('r', d => d.id === centerId ? 10 : 7)
      .attr('fill', d => d.id === centerId ? '#2563eb' : '#7c3aed')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('fill-opacity', d => d.id === centerId ? 1 : 0.75);

    nodeEl.append('text')
      .attr('dy', d => d.id === centerId ? -13 : -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', d => d.id === centerId ? 11 : 9)
      .attr('fill', d => d.id === centerId ? '#1e40af' : '#374151')
      .attr('font-weight', d => d.id === centerId ? '700' : '400')
      .text(d => {
        const label = d.id.replace(/^Skill-/, '').replace(/-/g, ' ');
        return label.length > 18 ? label.slice(0, 17) + '…' : label;
      });

    nodeEl.append('title').text(d => d.id);

    sim.on('tick', () => {
      linkEl
        .attr('x1', d => Math.max(10, Math.min(W - 10, d.source.x)))
        .attr('y1', d => Math.max(10, Math.min(H - 10, d.source.y)))
        .attr('x2', d => Math.max(10, Math.min(W - 10, d.target.x)))
        .attr('y2', d => Math.max(10, Math.min(H - 10, d.target.y)));
      nodeEl.attr('transform', d =>
        `translate(${Math.max(10, Math.min(W - 10, d.x))},${Math.max(12, Math.min(H - 8, d.y))})`
      );
    });

    sim.stop();
    for (let i = 0; i < 120; i++) sim.tick();
    sim.on('tick', () => {
      linkEl
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeEl.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    sim.restart();
  });
})();
""".strip()


def build_graph_js() -> str:
    """Return the D3 force graph JS bundle."""
    return r"""
(function () {
  const DATA = window.PLAYBOOK_DATA || {};
  const skills = DATA.skills || [];
  const skillMap = {};
  skills.forEach(s => { skillMap[s.skill_id] = s; });

  // Load graph-data.json via XHR (works for file:// and http://)
  function loadGraphData(cb) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '../assets/graph-data.json');
    xhr.onload = () => {
      try { cb(JSON.parse(xhr.responseText)); } catch (e) { cb(null); }
    };
    xhr.onerror = () => cb(null);
    xhr.send();
  }

  loadGraphData(function (raw) {
    if (!raw) { document.getElementById('graph-svg').insertAdjacentHTML('beforebegin', '<p class="muted">无法加载图谱数据。</p>'); return; }

    const nodes = raw.nodes.map(n => ({ ...n }));
    const links = raw.links.map(l => ({ ...l }));

    // Domain colour palette (Tableau-10 extended)
    const domains = [...new Set(nodes.map(n => n.domain))].sort();
    const colour = d3.scaleOrdinal(d3.schemeTableau10.concat(d3.schemePastel1)).domain(domains);

    // Degree map for node sizing
    const degree = {};
    links.forEach(l => {
      degree[l.source] = (degree[l.source] || 0) + 1;
      degree[l.target] = (degree[l.target] || 0) + 1;
    });
    const maxDeg = Math.max(...Object.values(degree), 1);
    const rScale = d3.scaleSqrt().domain([0, maxDeg]).range([4, 14]);

    const svg = d3.select('#graph-svg');
    const W = svg.node().parentElement.clientWidth || 1100;
    const H = Math.max(600, window.innerHeight - 240);
    svg.attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`);

    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom().scaleExtent([0.1, 6]).on('zoom', e => g.attr('transform', e.transform)));

    // Edge type → display config
    const edgeCfg = {
      prerequisite: { stroke: '#3b82f6', dasharray: null, width: 1.5 },
      combinable:   { stroke: '#10b981', dasharray: '5,3',   width: 1 },
      extension:    { stroke: '#f59e0b', dasharray: '2,4',   width: 1 },
    };

    // Visibility state
    const visible = { prerequisite: true, combinable: true, extension: false };
    document.querySelectorAll('.graph-controls input[type=checkbox]').forEach(cb => {
      cb.addEventListener('change', () => {
        visible[cb.id.replace('cb-', '')] = cb.checked;
        updateEdgeVisibility();
      });
    });

    function updateEdgeVisibility() {
      linkEl.style('display', d => visible[d.type] ? null : 'none');
    }

    // Simulation — only prerequisite + combinable edges by default for perf
    const activeLinks = links.filter(l => l.type === 'prerequisite' || l.type === 'combinable');
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(activeLinks).id(d => d.id).distance(60).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide().radius(d => rScale(degree[d.id] || 0) + 4));

    // Draw all edges (extension hidden initially)
    const linkEl = g.append('g').selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => (edgeCfg[d.type] || edgeCfg.prerequisite).stroke)
      .attr('stroke-width', d => (edgeCfg[d.type] || edgeCfg.prerequisite).width)
      .attr('stroke-dasharray', d => (edgeCfg[d.type] || edgeCfg.prerequisite).dasharray)
      .attr('stroke-opacity', 0.5)
      .style('display', d => visible[d.type] ? null : 'none');

    // Draw nodes
    const nodeEl = g.append('g').selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => rScale(degree[d.id] || 0))
      .attr('fill', d => colour(d.domain))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Hover: highlight 1-hop neighbourhood
    const neighborSet = new Set();
    nodeEl
      .on('mouseover', (e, d) => {
        neighborSet.clear();
        neighborSet.add(d.id);
        links.forEach(l => {
          const src = typeof l.source === 'object' ? l.source.id : l.source;
          const tgt = typeof l.target === 'object' ? l.target.id : l.target;
          if (src === d.id || tgt === d.id) { neighborSet.add(src); neighborSet.add(tgt); }
        });
        nodeEl.attr('opacity', n => neighborSet.has(n.id) ? 1 : 0.15);
        linkEl.attr('stroke-opacity', l => {
          const src = typeof l.source === 'object' ? l.source.id : l.source;
          const tgt = typeof l.target === 'object' ? l.target.id : l.target;
          return (neighborSet.has(src) && neighborSet.has(tgt)) ? 0.8 : 0.05;
        });
      })
      .on('mouseout', () => {
        nodeEl.attr('opacity', 1);
        linkEl.attr('stroke-opacity', 0.5);
      })
      .on('click', (e, d) => showInfo(d));

    // Tick
    sim.on('tick', () => {
      linkEl
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeEl.attr('cx', d => d.x).attr('cy', d => d.y);
    });

    // Info panel
    const infoPanel = document.getElementById('graph-info');
    document.getElementById('graph-info-close').addEventListener('click', () => infoPanel.classList.add('hidden'));

    function showInfo(d) {
      const sk = skillMap[d.id];
      document.getElementById('gi-title').textContent = sk ? sk.title : d.id;
      document.getElementById('gi-domain').textContent = d.domain || '';
      document.getElementById('gi-summary').textContent = sk ? (sk.problem_solved || sk.algorithm_summary || '') : '';
      const link = document.getElementById('gi-link');
      link.href = `../skills/${d.id}.html`;
      infoPanel.classList.remove('hidden');
    }

    // Search
    document.getElementById('graph-search').addEventListener('input', function () {
      const q = this.value.trim().toLowerCase();
      if (!q) { nodeEl.attr('opacity', 1); return; }
      nodeEl.attr('opacity', d => (d.id.toLowerCase().includes(q) || (skillMap[d.id] && skillMap[d.id].title.toLowerCase().includes(q))) ? 1 : 0.1);
    });

    updateEdgeVisibility();
  });
})();
"""


# ---------------------------------------------------------------------------
# CSS + JS assets (Phase 3C/D additions merged)
# ---------------------------------------------------------------------------

def build_css() -> str:
    return """
:root {
  --bg: #f7f7f4;
  --panel: #fff;
  --ink: #202124;
  --muted: #667085;
  --line: #e5e7eb;
  --accent: #2563eb;
  --accent2: #7c3aed;
  --tag: #eef2ff;
  --green: #10b981;
  --amber: #f59e0b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.6 -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif; }

/* ── Top bar ── */
.topbar {
  position: sticky; top: 0; z-index: 10;
  display: flex; gap: 16px; align-items: center;
  padding: 12px 22px; background: #111827; color: #fff;
}
.brand { color: #fff; text-decoration: none; font-weight: 700; font-size: 16px; white-space: nowrap; }
.topnav { display: flex; gap: 4px; }
.topnav a { color: #9ca3af; text-decoration: none; font-size: 13px; padding: 4px 10px; border-radius: 6px; }
.topnav a:hover { background: #1f2937; color: #fff; }
#global-search {
  margin-left: auto; width: min(500px, 40vw); padding: 8px 12px;
  border-radius: 10px; border: 0; font-size: 14px;
}

/* ── Layout ── */
.layout { display: grid; grid-template-columns: 200px 1fr; min-height: calc(100vh - 52px); }
.sidebar { padding: 24px 14px; border-right: 1px solid var(--line); background: #fff; }
.sidebar a { display: block; color: #374151; text-decoration: none; padding: 7px 10px; border-radius: 8px; font-size: 14px; }
.sidebar a:hover { background: #f3f4f6; }
.content { padding: 28px 36px; max-width: 1400px; }

/* ── Hero / tabs ── */
.hero { margin-bottom: 8px; }
.hero h1 { margin: 0 0 8px; font-size: 28px; }
.lead { font-size: 16px; color: var(--muted); margin: 0 0 20px; }
.hero-tabs { display: flex; gap: 8px; margin-bottom: 24px; }
.tab-btn {
  padding: 8px 18px; border: 1.5px solid var(--line); background: var(--panel);
  border-radius: 999px; font-size: 14px; cursor: pointer; color: var(--muted);
  transition: all .15s;
}
.tab-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ── Business entry cards ── */
.biz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; margin: 16px 0; }
.biz-card {
  display: flex; gap: 14px; align-items: flex-start;
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 16px; text-decoration: none; color: var(--ink);
  transition: box-shadow .15s, transform .1s;
}
.biz-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.08); transform: translateY(-2px); }
.biz-icon { font-size: 26px; flex-shrink: 0; line-height: 1; }
.biz-body strong { display: block; font-size: 15px; margin-bottom: 4px; }
.biz-body p { margin: 0; font-size: 13px; color: var(--muted); }
.biz-tag { margin-left: auto; flex-shrink: 0; font-size: 11px; background: var(--tag); color: #3730a3; padding: 3px 8px; border-radius: 999px; align-self: center; }

/* ── DS panel ── */
.ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin: 16px 0; }
.ds-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }
.ds-card h3 { margin: 0 0 8px; font-size: 15px; }
.hot-list { padding-left: 18px; margin: 8px 0 0; }
.hot-list li { margin-bottom: 6px; font-size: 14px; }
.hot-list a { color: var(--accent); text-decoration: none; }
.algo-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.algo-tags .tag { text-decoration: none; }
.ceo-entry { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 20px 0; align-items: start; }
.ceo-entry-body h3 { margin: 0 0 8px; font-size: 18px; }
.ceo-entry-body p { color: var(--muted); margin: 0 0 16px; font-size: 14px; }
.ceo-phases { display: flex; flex-direction: column; gap: 10px; }
.ceo-phase { background: var(--panel); border-left: 4px solid; border-radius: 0 10px 10px 0; padding: 12px 16px; font-size: 13px; }
.ceo-phase p { margin: 4px 0; color: var(--muted); }
.btn-primary {
  display: inline-block; margin-top: 12px; padding: 8px 16px;
  background: var(--accent); color: #fff; border-radius: 8px; text-decoration: none; font-size: 14px;
}

/* ── Metrics ── */
.metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 14px; margin: 16px 0; }
.metrics > div { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }
.metrics strong { display: block; font-size: 30px; font-weight: 700; }
.metrics span { color: var(--muted); font-size: 13px; }

/* ── Domain / topic grids ── */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 14px; margin: 16px 0; }
.metric-card, .domain-card {
  display: block; background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 16px; text-decoration: none; color: var(--ink);
  transition: box-shadow .15s;
}
.metric-card:hover, .domain-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.07); }
.metric-card strong { display: block; font-weight: 600; }
.metric-card span { color: var(--muted); font-size: 13px; }

/* ── Skill cards ── */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; margin: 16px 0; }
.skill-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; }
.skill-card h3 { margin: 0 0 6px; font-size: 15px; }
.skill-card h3 a { color: var(--accent); text-decoration: none; }
.skill-card h3 a:hover { text-decoration: underline; }
.skill-card p { margin: 4px 0; font-size: 14px; color: #374151; }
.skill-card .muted { color: var(--muted); font-size: 13px; }
.card-badges { display: flex; gap: 8px; margin: 6px 0; flex-wrap: wrap; }
.roi-badge { font-size: 12px; background: #ecfdf5; color: #065f46; padding: 2px 8px; border-radius: 999px; }
.diff-badge { font-size: 12px; background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 999px; }

/* ── Tags ── */
.tag { display: inline-block; margin: 3px 5px 0 0; padding: 3px 8px; background: var(--tag); border-radius: 999px; font-size: 12px; color: #3730a3; text-decoration: none; }
.tag.topic { background: #ecfdf5; color: #047857; }
.tag-row { margin: 8px 0 16px; }

/* ── Skill detail page ── */
.breadcrumbs { color: var(--muted); margin-bottom: 12px; font-size: 13px; }
.breadcrumbs a { color: var(--accent); text-decoration: none; }
.two-col { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 24px; margin-top: 20px; }
.relation-panel { position: sticky; top: 70px; align-self: start; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; }
.relation-panel h2 { margin: 0 0 10px; font-size: 15px; }
.relation-panel h3 { margin: 14px 0 6px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.relation-panel ul { padding-left: 18px; margin: 0; }
#ego-graph { display: block; border-radius: 8px; background: #f8fafc; border: 1px solid var(--line); margin-bottom: 6px; }
.ego-legend { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 4px; margin-bottom: 12px; }
.roi-panel { display: flex; gap: 16px; flex-wrap: wrap; background: #f8fafc; border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; margin: 12px 0 20px; }
.roi-item { display: flex; flex-direction: column; }
.roi-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.roi-value { font-size: 16px; font-weight: 600; margin-top: 2px; }
.muted { color: var(--muted); }

/* ── Search dropdown ── */
.search-results { position: absolute; top: 52px; left: 0; right: 0; z-index: 100; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,.12); max-height: 440px; overflow-y: auto; margin: 0 22px; }
.search-results.hidden { display: none; }
.search-results .result { display: block; padding: 10px 16px; text-decoration: none; color: var(--ink); border-bottom: 1px solid var(--line); font-size: 14px; }
.search-results .result:hover { background: #f9fafb; }
.search-results .result:last-child { border-bottom: none; }

/* ── Workflow tree ── */
.wf-meta { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 12px 0; }
.wf-entry-question { font-size: 15px; font-weight: 500; }
.wf-outcomes { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 14px 18px; margin: 16px 0; }
.wf-outcomes h3 { margin: 0 0 8px; font-size: 14px; color: #166534; }
.wf-outcomes ul { margin: 0; padding-left: 18px; }
.wf-outcomes li { font-size: 14px; margin-bottom: 4px; }
.wf-tree { margin-top: 24px; }
.wf-step {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 20px; margin-bottom: 12px;
  border-left: 4px solid var(--accent);
}
.wf-step-name { font-weight: 700; font-size: 15px; margin-bottom: 6px; }
.wf-question { font-size: 15px; margin: 6px 0; color: #1d4ed8; }
.wf-context { font-size: 13px; color: var(--muted); margin-bottom: 10px; }
.wf-branches { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
.wf-branch { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.wf-branch > summary { padding: 10px 14px; cursor: pointer; font-size: 14px; font-weight: 500; background: #f9fafb; list-style: none; }
.wf-branch > summary::before { content: "▶ "; font-size: 11px; color: var(--muted); }
.wf-branch[open] > summary::before { content: "▼ "; }
.wf-condition { color: var(--ink); }
.wf-branch-skills { display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 14px; }
.wf-skill-chip {
  display: flex; flex-direction: column; background: var(--tag); border-radius: 10px;
  padding: 8px 12px; text-decoration: none; color: var(--ink); min-width: 160px;
  transition: box-shadow .1s;
}
.wf-skill-chip:hover { box-shadow: 0 2px 8px rgba(37,99,235,.15); }
.wf-skill-chip.missing { opacity: .5; cursor: default; }
.chip-name { font-size: 13px; font-weight: 600; color: #1e40af; }
.chip-role { font-size: 12px; color: var(--muted); margin-top: 2px; }

/* ── Graph page ── */
#graph-svg { width: 100%; display: block; background: #fafafa; border-radius: 12px; border: 1px solid var(--line); }
.graph-controls { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; font-size: 14px; }
.graph-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.edge-dot { width: 14px; height: 4px; border-radius: 2px; display: inline-block; }
.edge-dot.prereq { background: #3b82f6; }
.edge-dot.combo { background: #10b981; }
.edge-dot.ext { background: #f59e0b; }
.graph-info {
  position: fixed; top: 80px; right: 24px; z-index: 20;
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 20px; width: 280px; box-shadow: 0 8px 32px rgba(0,0,0,.12);
}
.graph-info.hidden { display: none; }
.graph-info h3 { margin: 0 28px 8px 0; font-size: 15px; }
.graph-info p { margin: 4px 0; font-size: 13px; }

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); }
th { background: #f9fafb; font-weight: 600; }

/* ── Code preview (Phase H) ── */
.code-preview {
  background: #0f172a; color: #e2e8f0;
  border-radius: 10px; padding: 16px 18px; overflow-x: auto;
  font-size: 13px; line-height: 1.55; font-family: 'JetBrains Mono', 'Fira Code', Menlo, monospace;
  max-height: 400px; overflow-y: auto;
  margin: 10px 0;
  white-space: pre;
}

/* ── Filter bar (Phase I) ── */
.filter-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 16px 0 8px; }
.filter-select {
  padding: 7px 12px; border: 1px solid var(--line); border-radius: 8px;
  font-size: 14px; background: var(--panel); color: var(--ink); cursor: pointer;
}
.filter-select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.filter-hint { font-size: 13px; }

/* ── toB Playbook pages (Phase F) ── */
.pb-hero { display: flex; gap: 20px; align-items: flex-start; margin-bottom: 16px; }
.pb-icon { font-size: 48px; flex-shrink: 0; line-height: 1; margin-top: 4px; }
.pb-intro { background: #f8fafc; border-left: 4px solid var(--accent); border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 16px 0; font-size: 15px; color: #374151; }
.pb-steps { margin-top: 24px; display: flex; flex-direction: column; gap: 16px; }
.pb-step { display: flex; gap: 20px; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }
.pb-step-num { width: 48px; height: 48px; border-radius: 50%; background: var(--accent); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.pb-step-body { flex: 1; min-width: 0; }
.pb-step-title { margin: 0 0 8px; font-size: 16px; }
.pb-problem { font-size: 14px; color: #1d4ed8; margin: 0 0 12px; }
.pb-skills { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; }
.pb-skill { background: var(--tag); border-radius: 10px; padding: 10px 14px; }
.pb-skill-header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.pb-skill-name { font-weight: 600; font-size: 14px; color: var(--accent); text-decoration: none; }
.pb-skill-name:hover { text-decoration: underline; }
.pb-skill-badges { display: flex; gap: 6px; }
.pb-skill-why { margin: 4px 0 0; font-size: 13px; color: var(--muted); }
.pb-data, .pb-output { font-size: 13px; margin-top: 8px; background: #f9fafb; border-radius: 6px; padding: 8px 12px; }

/* ── ROI Calculator ── */
.calc-wrapper { margin: 40px 0 0; background: var(--panel); border: 2px solid var(--line); border-radius: 20px; overflow: hidden; }
.calc-header { padding: 28px 32px 20px; border-bottom: 1px solid var(--line); }
.calc-header h2 { margin: 0 0 6px; font-size: 22px; }
.calc-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--line); background: #f8fafc; }
.calc-tab { flex: 1; padding: 14px 8px; border: none; background: none; cursor: pointer; font-size: 14px; font-weight: 500; color: var(--muted); border-bottom: 3px solid transparent; transition: all .15s; }
.calc-tab:hover { color: var(--ink); background: #f1f5f9; }
.calc-tab.active { color: var(--tc); border-bottom-color: var(--tc); background: #fff; font-weight: 700; }
.calc-body { padding: 0; }
.calc-panel { display: none; grid-template-columns: 1fr 280px; gap: 0; }
.calc-panel.active { display: grid; }
.calc-inputs { padding: 28px 32px; display: flex; flex-direction: column; gap: 20px; }
.calc-row { display: flex; flex-direction: column; gap: 6px; }
.calc-label { font-size: 14px; font-weight: 500; color: #374151; }
.calc-input-wrap { display: flex; align-items: center; gap: 10px; }
.calc-input { flex: 1; accent-color: var(--tc); height: 6px; cursor: pointer; }
.calc-val { font-size: 16px; font-weight: 700; color: var(--tc); min-width: 52px; text-align: right; }
.calc-unit { font-size: 12px; color: var(--muted); min-width: 52px; }
.calc-result { background: linear-gradient(135deg, var(--tc) 0%, color-mix(in srgb, var(--tc) 70%, #000) 100%); padding: 40px 28px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
.calc-result-label { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: rgba(255,255,255,.7); margin-bottom: 12px; }
.calc-result-num { font-size: 52px; font-weight: 900; color: #fff; line-height: 1; font-variant-numeric: tabular-nums; }
.calc-result-unit { font-size: 16px; color: rgba(255,255,255,.8); margin-top: 6px; }
.calc-disclaimer { font-size: 11px; color: rgba(255,255,255,.5); margin-top: 20px; line-height: 1.5; max-width: 200px; }
@media(max-width:700px){ .calc-panel.active{grid-template-columns:1fr} .calc-result{padding:28px} .calc-result-num{font-size:38px} }
""".strip()


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
      `${s.roi_figure ? ' · 💰 ' + esc(s.roi_figure) : ''}` +
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
    global KNOWN_SKILL_IDS
    KNOWN_SKILL_IDS = {skill.skill_id for skill in skills}
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)

    skill_count  = len(skills)
    edge_count   = len(graph.edges)
    domain_count = len({s.domain_dir for s in skills})

    # Data assets
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {"skill_count": skill_count, "domain_count": domain_count, "edge_count": edge_count},
        "domains": domains,
        "skills": [skill.__dict__ for skill in skills],
    }
    write_file(out / "assets" / "playbook-data.json", json.dumps(data, ensure_ascii=False, indent=2))
    write_file(out / "assets" / "playbook-data.js",  "window.PLAYBOOK_DATA = " + json.dumps(data, ensure_ascii=False) + ";")

    graph_json = {
        "nodes": [{"id": n.id, "domain": n.domain, "title": n.id} for n in graph.nodes.values()],
        "links": [{"source": e.source, "target": e.target, "type": e.edge_type} for e in graph.edges],
    }
    write_file(out / "assets" / "graph-data.json", json.dumps(graph_json, ensure_ascii=False, indent=2))
    write_file(out / "assets" / "style.css",  build_css())
    write_file(out / "assets" / "search.js",  build_search_js())
    write_file(out / "assets" / "graph.js",   build_graph_js())
    write_file(out / "assets" / "ego-graph.js", build_ego_graph_js())

    # ── Index (Phase 3C) ──
    write_file(out / "index.html", html_page(
        "总览",
        render_index(skill_count, domain_count, edge_count, domains, skills),
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
  <span class="filter-hint muted">过滤结合搜索框使用</span>
</div>"""
    write_file(out / "skills" / "index.html", html_page(
        "全部 Skills",
        f"<h1>全部 Skills</h1>{filter_bar}<div class='cards' id='skill-card-grid'>{all_cards}</div>",
        "../",
    ))

    # ── Domain pages ──
    domain_index_cards: list[str] = []
    for domain in domains:
        domain_skills = [s for s in skills if s.domain_dir == domain["vault_dir"]]
        cards = "".join(render_skill_card(s, "../") for s in domain_skills)
        title = domain["vault_dir"]
        write_file(
            out / "domains" / f"{slugify(title)}.html",
            html_page(title,
                      f"<h1>{html.escape(title)}</h1>"
                      f"<p>{html.escape(domain.get('description',''))}</p>"
                      f"<div class='cards'>{cards}</div>",
                      "../"),
        )
        domain_index_cards.append(
            f"<a class='metric-card domain-card' href='{slugify(title)}.html'>"
            f"<strong>{html.escape(title)}</strong>"
            f"<span>{len(domain_skills)} Skills</span></a>"
        )
    write_file(out / "domains" / "index.html", html_page(
        "按领域",
        "<h1>按领域</h1><div class='grid'>" + "".join(domain_index_cards) + "</div>",
        "../",
    ))

    # ── Topic pages ──
    all_topics = sorted({topic for s in skills for topic in s.topics})
    topic_cards: list[str] = []
    for topic in all_topics:
        topic_skills = [s for s in skills if topic in s.topics]
        cards = "".join(render_skill_card(s, "../") for s in topic_skills)
        path = f"{slugify(topic)}.html"
        write_file(out / "topics" / path, html_page(
            topic,
            f"<h1>{html.escape(topic)}</h1><div class='cards'>{cards}</div>",
            "../",
        ))
        topic_cards.append(f"<a class='metric-card' href='{path}'>{html.escape(topic)}<span>{len(topic_skills)} Skills</span></a>")
    write_file(out / "topics" / "index.html", html_page(
        "按主题",
        "<h1>按主题</h1><div class='grid'>" + "".join(topic_cards) + "</div>",
        "../",
    ))

    # ── Workflow pages (Phase 2B: YAML-first, keyword fallback) ──
    skill_lookup = {s.skill_id: s for s in skills}
    workflow_cards: list[str] = []
    for workflow_name in WORKFLOW_RULES:
        slug_path = f"{slugify(workflow_name)}.html"
        wf_id = slugify(workflow_name).split("-")[0] + "-" + slugify(workflow_name).split("-")[1]  # e.g. "wf-a"

        # Check if a structured YAML definition exists
        if wf_defs and wf_id in wf_defs:
            page_html = render_workflow_page(wf_defs[wf_id], skill_lookup)
        else:
            # Fallback: keyword-matched skill list (original behaviour)
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
        wf_skill_count = len(wf_defs.get(wf_id, {}).get("steps", [])) or len([s for s in skills if workflow_name in s.workflows])
        workflow_cards.append(
            f"<a class='metric-card' href='{slug_path}'>"
            f"<strong>{html.escape(workflow_name)}</strong>"
            f"<span>{wf_skill_count} 步骤/Skills</span></a>"
        )
    write_file(out / "workflows" / "index.html", html_page(
        "工作流",
        "<h1>工作流</h1><p class='muted'>端到端业务决策路径，每条工作流包含分步决策树和推荐 Skill 组合。</p>"
        "<div class='grid'>" + "".join(workflow_cards) + "</div>",
        "../",
    ))

    # ── Skills Graph (Phase 3D: D3 visualisation) ──
    write_file(out / "graph" / "overview.html", render_graph_page(skill_count, edge_count))

    # ── CEO Roadmap whitepaper ──
    write_file(out / "ai-roadmap.html", render_roadmap_page(skill_lookup))

    # ── toB Scene Playbooks (Phase F) ──
    for pb in TOB_PLAYBOOKS:
        write_file(
            out / "playbooks" / f"{pb['id']}.html",
            render_tob_playbook(pb, skill_lookup),
        )
    tob_index_cards = "".join(
        f"<a class='biz-card' href='{pb['id']}.html'>"
        f"<span class='biz-icon'>{pb['icon']}</span>"
        f"<div class='biz-body'><strong>{html.escape(pb['name'])}</strong>"
        f"<p>{html.escape(pb['desc'])}</p></div>"
        f"<span class='biz-tag'>{html.escape(pb['tag'])}</span></a>"
        for pb in TOB_PLAYBOOKS
    )
    write_file(out / "playbooks" / "index.html", html_page(
        "场景手册",
        "<h1>场景手册</h1>"
        "<p class='muted'>针对运营部门的开箱即用决策指南，每本手册包含完整操作步骤、所需数据和预期收益。</p>"
        f"<div class='biz-grid'>{tob_index_cards}</div>",
        "../",
    ))

    write_file(out / "README.md", "# paper2skills Playbook\n\n打开 `index.html` 浏览。\n")
    report = {
        "skill_pages": skill_count,
        "domains": domain_count,
        "edges": edge_count,
        "generated_at": data["generated_at"],
    }
    write_file(out / "build-report.json", json.dumps(report, ensure_ascii=False, indent=2))
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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
