"""Candidate queue generation for incremental paper selection."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from paper2skills_common.domains import load_domain_registry, project_root_from


DEFAULT_QUEUE_PATH = Path("paper2skills-vault/00-项目管理/paper_candidate_queue.json")

DOMAIN_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compliance", ("compliance", "policy", "regulatory", "tos", "risk", "合规", "政策")),
    ("causal_inference", ("causal", "uplift", "counterfactual", "did", "dml", "因果")),
    ("ab_testing", ("ab", "a/b", "bandit", "experiment", "实验")),
    ("time_series", ("forecast", "time series", "temporal", "demand", "预测")),
    ("supply_chain", ("inventory", "supply", "procurement", "replenishment", "库存", "供应")),
    ("recommendation", ("recommend", "ranking", "retrieval", "推荐", "排序")),
    ("knowledge_graph", ("graph", "kg", "gnn", "knowledge", "图谱")),
    ("data_agent_llm", ("dataagent", "dashboard", "sql", "analysis", "agentic anomaly")),
    ("mas", ("multi-agent", "mas", "orchestration", "consensus")),
    ("advertising", ("ad", "advertising", "roas", "attribution", "keyword", "广告")),
    ("user_analytics", ("funnel", "cohort", "clickstream", "user", "用户")),
    ("marketing", ("marketing", "mmm", "promotion", "campaign", "营销")),
    ("pricing", ("pricing", "price", "elasticity", "定价")),
    ("logistics", ("logistics", "delivery", "routing", "returns", "物流")),
    ("risk_fraud", ("fraud", "anomaly", "transaction", "风控", "欺诈")),
    ("ml_fundamentals", ("feature", "model evaluation", "drift", "validation")),
    ("llm_agent_engineering", ("mcp", "tool", "context", "guardrail", "agent engineering")),
)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-")
    return value.upper()[:48] or "TOPIC"


def infer_candidate_domain(text: str, known_domains: Iterable[str]) -> str:
    """Infer a domain from a topic string without silently defaulting to one domain."""
    known = set(known_domains)
    lowered = text.lower()
    for domain, hints in DOMAIN_HINTS:
        if domain in known and any(hint.lower() in lowered for hint in hints):
            return domain
    return "unknown"


def score_candidate(
    *,
    gap_priority: str = "P2",
    business_value: int = 3,
    paper_quality: int = 0,
    implementability: int = 3,
    recency: int = 3,
    code_available: bool = False,
) -> int:
    """Score a candidate with transparent, auditable weights."""
    priority_weight = {"P0": 100, "P1": 72, "P2": 44}.get(gap_priority.upper(), 30)
    score = priority_weight
    score += max(0, min(business_value, 5)) * 6
    score += max(0, min(paper_quality, 10)) * 3
    score += max(0, min(implementability, 5)) * 5
    score += max(0, min(recency, 5)) * 3
    if code_available:
        score += 12
    return score


def _keywords_from_gap(gap: dict[str, Any]) -> str:
    if gap.get("type") == "missing_prerequisite":
        skill = gap.get("missing_skill", "")
        return skill.replace("Skill-", "").replace("-", " ") + " ecommerce implementation"
    if gap.get("type") == "missing_bridge":
        return f"{gap.get('domain_a', '')} {gap.get('domain_b', '')} cross-domain ecommerce"
    if gap.get("type") == "missing_extension":
        skill = gap.get("skill", "")
        return skill.replace("Skill-", "").replace("-", " ") + " advanced applications"
    if gap.get("type") == "isolated_skill":
        skill = gap.get("skill", "")
        return skill.replace("Skill-", "").replace("-", " ") + " related methods"
    if gap.get("type") == "thin_domain":
        return f"{gap.get('domain', '')} ecommerce applied machine learning"
    if gap.get("type") == "domain_review":
        return f"{gap.get('domain', '')} ecommerce guardrail evaluation"
    return str(gap.get("description") or gap.get("topic") or "")


def candidate_from_gap(root: str | Path | None, gap: dict[str, Any], index: int) -> dict[str, Any]:
    project_root = project_root_from(Path(root) if root is not None else None)
    registry = load_domain_registry(project_root)
    priority = str(gap.get("priority", "P2")).upper()
    gap_type = str(gap.get("type", "unknown_gap"))
    text = " ".join(str(gap.get(key, "")) for key in ("missing_skill", "skill", "domain_a", "domain_b", "description"))
    domain = gap.get("domain") or gap.get("domain_a") or infer_candidate_domain(text, registry.known_keys())
    keywords = _keywords_from_gap(gap)
    topic = gap.get("missing_skill") or gap.get("skill") or gap.get("topic") or gap.get("description") or gap_type

    return {
        "topic_id": f"GAP-{priority}-{index:03d}-{_slug(gap_type)}",
        "source": "skills_graph_gaps",
        "domain": domain,
        "gap_type": gap_type,
        "topic": str(topic),
        "keywords": keywords.strip(),
        "paper_url": gap.get("paper_url"),
        "arxiv_id": gap.get("arxiv_id"),
        "score": score_candidate(gap_priority=priority, business_value=int(gap.get("business_value", 3) or 3)),
        "decision": gap.get("decision", "pending"),
        "reason": gap.get("description") or f"Generated from {gap_type}",
        "gap_ref": gap,
    }


def _candidate_from_roadmap(root: Path, item: dict[str, Any], index: int) -> dict[str, Any]:
    registry = load_domain_registry(root)
    text = " ".join(str(value) for value in item.values())
    domain = item.get("domain") or infer_candidate_domain(text, registry.known_keys())
    priority = str(item.get("priority", "P1")).upper()
    topic = item.get("skill") or item.get("topic") or item.get("title") or text
    return {
        "topic_id": f"ROADMAP-{priority}-{index:03d}-{_slug(str(topic))}",
        "source": "roadmap",
        "domain": domain,
        "gap_type": item.get("gap_type", "roadmap_candidate"),
        "topic": str(topic),
        "keywords": str(item.get("keywords") or item.get("paper_direction") or topic),
        "paper_url": item.get("paper_url"),
        "arxiv_id": item.get("arxiv_id"),
        "score": score_candidate(
            gap_priority=priority,
            business_value=int(item.get("business_value", 4) or 4),
            implementability=int(item.get("implementability", 3) or 3),
        ),
        "decision": item.get("decision", "pending"),
        "reason": item.get("reason", "Roadmap item carried into auditable candidate queue"),
        "roadmap_ref": item,
    }


def _candidate_from_user_priority(root: Path, item: dict[str, Any], index: int) -> dict[str, Any]:
    registry = load_domain_registry(root)
    priority = str(item.get("priority", "P0")).upper()
    topic = item.get("topic") or item.get("skill") or item.get("keywords") or "manual priority"
    domain = item.get("domain") or infer_candidate_domain(str(topic), registry.known_keys())
    return {
        "topic_id": f"USER-{priority}-{index:03d}-{_slug(str(topic))}",
        "source": "user_priority",
        "domain": domain,
        "gap_type": item.get("gap_type", "user_priority"),
        "topic": str(topic),
        "keywords": str(item.get("keywords") or topic),
        "paper_url": item.get("paper_url"),
        "arxiv_id": item.get("arxiv_id"),
        "score": score_candidate(gap_priority=priority, business_value=int(item.get("business_value", 5) or 5)),
        "decision": item.get("decision", "pending"),
        "reason": item.get("reason", "Explicit user priority"),
        "user_ref": item,
    }


def _candidate_skill_ref(candidate: dict[str, Any]) -> str | None:
    for key in ("skill_id", "topic"):
        value = candidate.get(key)
        if isinstance(value, str) and value.startswith("Skill-"):
            return value[:-3] if value.endswith(".md") else value
    for ref_key in ("roadmap_ref", "gap_ref", "user_ref"):
        ref = candidate.get(ref_key)
        if not isinstance(ref, dict):
            continue
        value = ref.get("skill") or ref.get("missing_skill")
        if isinstance(value, str) and value.startswith("Skill-"):
            return value[:-3] if value.endswith(".md") else value
    return None


def _load_skill_aliases(root: Path) -> dict[str, str]:
    alias_file = root / "paper2skills-vault" / "07-资源库" / "skill-aliases.json"
    if not alias_file.exists():
        return {}
    try:
        data = json.loads(alias_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    aliases: dict[str, str] = {}
    nested = data.get("aliases") if isinstance(data, dict) else None
    if isinstance(nested, dict):
        aliases.update({str(key): str(value) for key, value in nested.items() if isinstance(value, str)})
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith("_") or key == "aliases" or not isinstance(value, str):
                continue
            aliases[str(key)] = value
    return aliases


def _mark_existing_assets(root: Path, candidates: list[dict[str, Any]]) -> None:
    from paper2skills_common.assets import build_asset_inventory

    existing = {item["skill_id"] for item in build_asset_inventory(root)}
    aliases = _load_skill_aliases(root)
    for candidate in candidates:
        skill_ref = _candidate_skill_ref(candidate)
        resolved = aliases.get(skill_ref or "", skill_ref)
        if resolved == "__MIGRATED_TO_AI_NLP_VOC__":
            candidate["decision"] = "external_migrated"
            candidate["score"] = min(int(candidate["score"]), 5)
            candidate["resolved_existing_skill"] = resolved
            candidate["reason"] = f"{skill_ref} migrated out of this repository; retained for audit, not pending extraction."
        elif resolved == "__GENUINE_GAP__":
            continue
        elif resolved and resolved in existing:
            candidate["decision"] = "already_exists"
            candidate["score"] = min(int(candidate["score"]), 10)
            candidate["resolved_existing_skill"] = resolved
            candidate["reason"] = f"Existing vault Skill detected: {resolved}; retained for audit, not pending extraction."


def parse_roadmap_candidates(root: str | Path | None = None, paths: list[Path] | None = None) -> list[dict[str, Any]]:
    """Extract lightweight candidate rows from roadmap markdown tables."""
    project_root = project_root_from(Path(root) if root is not None else None)
    if paths is None:
        paths = [
            project_root / "paper2skills-vault" / "00-项目管理" / "next-papers-roadmap.md",
            project_root / "paper2skills-vault" / "00-项目管理" / "next-papers-roadmap-v2.md",
        ]

    items: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|") or "---" in line or "Skill" not in line:
                continue
            cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3 or cells[0] in {"#", "候选 Skill", "Date", "日期"}:
                continue
            if any("Skill 名称" in cell or "缺失 Skill" in cell or "候选 Skill" in cell for cell in cells):
                continue
            skill = next((cell for cell in cells if re.search(r"\bSkill-[A-Za-z0-9]", cell)), "")
            if not skill:
                continue
            items.append(
                {
                    "skill": skill.replace("[[", "").replace("]]", ""),
                    "paper_direction": cells[2] if len(cells) > 2 else "",
                    "priority": "P1",
                    "reason": f"Parsed from {path.relative_to(project_root)}",
                }
            )
    return items


def build_candidate_queue(
    root: str | Path | None = None,
    *,
    graph_gaps: list[dict[str, Any]] | None = None,
    roadmap_items: list[dict[str, Any]] | None = None,
    user_priorities: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    project_root = project_root_from(Path(root) if root is not None else None)
    candidates: list[dict[str, Any]] = []

    for index, gap in enumerate(graph_gaps or [], 1):
        candidates.append(candidate_from_gap(project_root, gap, index))
    for index, item in enumerate(roadmap_items if roadmap_items is not None else parse_roadmap_candidates(project_root), 1):
        candidates.append(_candidate_from_roadmap(project_root, item, index))
    for index, item in enumerate(user_priorities or [], 1):
        candidates.append(_candidate_from_user_priority(project_root, item, index))

    _mark_existing_assets(project_root, candidates)
    candidates.sort(key=lambda item: (-int(item["score"]), item["topic_id"]))
    if limit is not None:
        candidates = _limit_with_source_floor(candidates, limit)

    return {
        "_README": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "paper2skills_common.candidates",
            "note": "Derived candidate queue. CLAUDE.md remains the human source of truth; keep decisions auditable here.",
        },
        "summary": {
            "total_candidates": len(candidates),
            "pending": sum(1 for item in candidates if item["decision"] == "pending"),
            "selected": sum(1 for item in candidates if item["decision"] == "selected"),
            "already_exists": sum(1 for item in candidates if item["decision"] == "already_exists"),
            "external_migrated": sum(1 for item in candidates if item["decision"] == "external_migrated"),
            "sources": sorted({item["source"] for item in candidates}),
        },
        "candidates": candidates,
    }


def _limit_with_source_floor(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    sources = sorted({item["source"] for item in candidates})
    for source in sources:
        source_top = next((item for item in candidates if item["source"] == source), None)
        if source_top is not None and source_top["topic_id"] not in selected_ids:
            selected.append(source_top)
            selected_ids.add(source_top["topic_id"])

    for item in candidates:
        if len(selected) >= limit:
            break
        if item["topic_id"] in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item["topic_id"])

    selected.sort(key=lambda item: (-int(item["score"]), item["topic_id"]))
    return selected[:limit]


def write_candidate_queue(document: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def carry_forward_candidate_search(document: dict[str, Any], existing_document: dict[str, Any]) -> dict[str, Any]:
    carry_fields = (
        "paper_search",
        "decision",
        "workflow_status",
        "verification_status",
        "skill_id",
        "arxiv_id",
        "paper_url",
        "selected_paper",
        "selected_outputs",
        "review_notes",
    )
    existing_by_topic = {
        item.get("topic_id"): item
        for item in existing_document.get("candidates", [])
        if item.get("topic_id")
    }
    for candidate in document.get("candidates", []):
        topic_id = candidate.get("topic_id")
        existing = existing_by_topic.get(topic_id)
        if not existing:
            continue
        for field in carry_fields:
            if field in existing:
                candidate[field] = existing[field]
    if "summary" in document:
        candidates = document.get("candidates", [])
        document["summary"]["pending"] = sum(1 for item in candidates if item.get("decision") == "pending")
        document["summary"]["selected"] = sum(1 for item in candidates if item.get("decision") == "selected")
        document["summary"]["already_exists"] = sum(1 for item in candidates if item.get("decision") == "already_exists")
        document["summary"]["external_migrated"] = sum(1 for item in candidates if item.get("decision") == "external_migrated")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper2skills paper candidate queue")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = project_root_from(args.root)
    document = build_candidate_queue(project_root, limit=args.limit)
    output = args.output or (project_root / DEFAULT_QUEUE_PATH)
    if output.exists():
        try:
            existing_document = json.loads(output.read_text(encoding="utf-8"))
            document = carry_forward_candidate_search(document, existing_document)
        except json.JSONDecodeError:
            pass
    print(json.dumps(document["summary"], ensure_ascii=False, indent=2))
    if args.dry_run:
        print(json.dumps(document["candidates"][:5], ensure_ascii=False, indent=2))
        return 0
    write_candidate_queue(document, output)
    print(f"已写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
