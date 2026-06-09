"""Paper search helpers for auditable incremental candidate selection."""

from __future__ import annotations

import copy
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from paper2skills_common.candidates import DEFAULT_QUEUE_PATH
from paper2skills_common.domains import project_root_from


DEFAULT_SEARCH_RUN_DIR = Path("paper2skills-vault/00-项目管理/paper_search_runs")
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

STOPWORDS = {
    "and",
    "cross",
    "domain",
    "e",
    "with",
    "for",
    "the",
}
GENERIC_QUERY_TOKENS = {"commerce", "ecommerce", "marketplace", "retail"}
TOKEN_ALIASES = {
    "risk_fraud": ("risk", "fraud"),
    "data_agent_llm": ("data", "agent", "llm"),
    "visual_content": ("visual", "content", "image"),
    "mas": ("multi", "agent"),
}
DOMAIN_BRIDGE_ALIASES = {
    "advertising": ("advertising", "ads", "ctr", "campaign"),
    "compliance": ("compliance", "guardrail", "policy", "risk"),
    "data_agent_llm": ("data", "agent", "llm", "analytics"),
    "logistics": ("logistics", "delivery", "fulfillment", "routing"),
    "marketing": ("marketing", "promotion", "campaign", "customer"),
    "mas": ("multi-agent", "multi", "agent", "orchestration"),
    "pricing": ("pricing", "price", "dynamic"),
    "recommendation": ("recommendation", "recommender", "ranking", "retrieval"),
    "risk_fraud": ("risk", "fraud", "anomaly"),
    "visual_content": ("visual", "image", "video", "content"),
}
EXPERIMENT_SIGNALS = ("experiment", "benchmark", "ablation", "evaluation", "dataset", "real-world")
IMPLEMENTATION_SIGNALS = ("implementation", "code", "open-source", "github", "framework", "system")
BUSINESS_SIGNALS = ("e-commerce", "ecommerce", "retail", "marketplace", "advertising", "recommendation")
LOW_VALUE_SIGNALS = ("survey", "review", "position paper", "tutorial", "purely theoretical")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def _candidate_tokens(candidate: dict[str, Any]) -> list[str]:
    raw_parts = [
        str(candidate.get("domain") or ""),
        str(candidate.get("keywords") or ""),
    ]
    tokens: list[str] = []
    for part in raw_parts:
        for alias, replacements in TOKEN_ALIASES.items():
            part = part.replace(alias, " ".join(replacements))
        for token in re.split(r"[^A-Za-z0-9]+", part.lower()):
            if len(token) < 3 or token in STOPWORDS:
                continue
            tokens.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        deduped.append(token)
        seen.add(token)
    return sorted(deduped, key=lambda token: (token in GENERIC_QUERY_TOKENS, deduped.index(token)))


def build_arxiv_query(candidate: dict[str, Any], *, max_terms: int = 4) -> str:
    tokens = _candidate_tokens(candidate)[:max_terms]
    if not tokens:
        raise ValueError(f"Cannot build arXiv query for candidate {candidate.get('topic_id')}: no searchable tokens")
    return " AND ".join(f"all:{token}" for token in tokens)


def _entry_text(entry: ET.Element, path: str) -> str:
    found = entry.find(path, ATOM_NS)
    return _normalize_text(found.text or "") if found is not None else ""


def _entry_links(entry: ET.Element) -> tuple[str | None, str | None]:
    abs_url: str | None = None
    pdf_url: str | None = None
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href")
        if not href:
            continue
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = href
        elif link.attrib.get("rel") == "alternate":
            abs_url = href
    return abs_url, pdf_url


def parse_arxiv_feed(payload: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = _entry_text(entry, "atom:id")
        arxiv_id = entry_id.rstrip("/").rsplit("/", 1)[-1] if entry_id else ""
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        url, pdf_url = _entry_links(entry)
        categories = [category.attrib.get("term", "") for category in entry.findall("atom:category", ATOM_NS)]
        authors = [_entry_text(author, "atom:name") for author in entry.findall("atom:author", ATOM_NS)]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": _entry_text(entry, "atom:title"),
                "abstract": _entry_text(entry, "atom:summary"),
                "authors": [author for author in authors if author],
                "published": _entry_text(entry, "atom:published"),
                "updated": _entry_text(entry, "atom:updated"),
                "categories": [category for category in categories if category],
                "url": url or entry_id,
                "pdf_url": pdf_url,
            }
        )
    return papers


def fetch_arxiv_results(query: str, *, max_results: int = 5, timeout: int = 30) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(
        f"{ARXIV_API_URL}?{params}",
        headers={"User-Agent": "paper2skills/0.1 (https://github.com/paper2skills)"},
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return parse_arxiv_feed(response.read())
        except Exception as exc:  # pragma: no cover - network boundary
            last_error = exc
            if attempt == 0:
                time.sleep(3)
    raise last_error if last_error is not None else RuntimeError("arXiv request failed")


def _published_year(paper: dict[str, Any]) -> int | None:
    published = str(paper.get("published") or paper.get("updated") or "")
    match = re.match(r"(\d{4})", published)
    return int(match.group(1)) if match else None


def _bridge_endpoint_hits(candidate: dict[str, Any], text: str) -> list[tuple[str, bool]]:
    if candidate.get("gap_type") != "missing_bridge":
        return []
    gap_ref = candidate.get("gap_ref") if isinstance(candidate.get("gap_ref"), dict) else {}
    domains = [gap_ref.get("domain_a"), gap_ref.get("domain_b")]
    hits: list[tuple[str, bool]] = []
    for domain in domains:
        if not domain:
            continue
        aliases = DOMAIN_BRIDGE_ALIASES.get(str(domain), (str(domain),))
        hits.append((str(domain), any(alias in text for alias in aliases)))
    return hits


def score_paper_for_candidate(candidate: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    tokens = _candidate_tokens(candidate)
    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    text = f"{title} {abstract}"

    token_hits = [token for token in tokens if token in text]
    title_hits = [token for token in tokens if token in title]
    score = 20 + len(token_hits) * 8 + len(title_hits) * 4
    reasons = [f"keyword_hits={len(token_hits)}", f"title_hits={len(title_hits)}"]

    year = _published_year(paper)
    current_year = datetime.now(timezone.utc).year
    if year is not None:
        age = max(0, current_year - year)
        recency_bonus = 15 if age <= 1 else 10 if age <= 2 else 5 if age <= 4 else 0
        score += recency_bonus
        reasons.append(f"recency_bonus={recency_bonus}")

    experiment_hits = [signal for signal in EXPERIMENT_SIGNALS if signal in text]
    implementation_hits = [signal for signal in IMPLEMENTATION_SIGNALS if signal in text]
    business_hits = [signal for signal in BUSINESS_SIGNALS if signal in text]
    score += min(12, len(experiment_hits) * 4)
    score += min(10, len(implementation_hits) * 5)
    score += min(8, len(business_hits) * 2)
    reasons.append(f"experiment_hits={len(experiment_hits)}")
    reasons.append(f"implementation_hits={len(implementation_hits)}")
    reasons.append(f"business_hits={len(business_hits)}")

    low_value_hits = [signal for signal in LOW_VALUE_SIGNALS if signal in text]
    if low_value_hits:
        score -= min(20, len(low_value_hits) * 10)
        reasons.append(f"low_value_penalty={len(low_value_hits)}")

    bridge_hits = _bridge_endpoint_hits(candidate, text)
    missing_bridge_endpoints = [domain for domain, hit in bridge_hits if not hit]
    if missing_bridge_endpoints:
        score -= min(45, len(missing_bridge_endpoints) * 25)
        reasons.append(f"missing_bridge_endpoints={','.join(missing_bridge_endpoints)}")

    score = max(0, min(100, score))
    if score >= 60:
        recommendation = "candidate"
    elif score >= 40:
        recommendation = "watchlist"
    else:
        recommendation = "skip"

    return {
        "search_score": score,
        "recommendation": recommendation,
        "score_reasons": reasons,
    }


def rank_papers_for_candidate(candidate: dict[str, Any], papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for paper in papers:
        scored = copy.deepcopy(paper)
        scored.update(score_paper_for_candidate(candidate, paper))
        ranked.append(scored)
    ranked.sort(key=lambda item: (int(item["search_score"]), str(item.get("published") or "")), reverse=True)
    return ranked


def build_search_run(
    queue: dict[str, Any],
    *,
    fetcher: Callable[[str, int], list[dict[str, Any]]] | None = None,
    limit: int | None = None,
    max_results_per_candidate: int = 5,
    sleep_seconds: float = 3.0,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pending = [item for item in queue.get("candidates", []) if item.get("decision") == "pending"]
    if limit is not None:
        pending = pending[:limit]
    fetch = fetcher or (lambda query, max_results: fetch_arxiv_results(query, max_results=max_results))

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(pending):
        query = build_arxiv_query(candidate)
        attempted_queries = [query]
        try:
            papers = fetch(query, max_results_per_candidate)
            if not papers:
                broad_query = build_arxiv_query(candidate, max_terms=2)
                if broad_query != query:
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    attempted_queries.append(broad_query)
                    papers = fetch(broad_query, max_results_per_candidate)
            ranked = rank_papers_for_candidate(candidate, papers)
            error = None
        except Exception as exc:  # pragma: no cover - network boundary
            ranked = []
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "topic_id": candidate["topic_id"],
                "domain": candidate.get("domain"),
                "gap_type": candidate.get("gap_type"),
                "keywords": candidate.get("keywords"),
                "search_query": query,
                "attempted_queries": attempted_queries,
                "error": error,
                "papers": ranked,
            }
        )
        if sleep_seconds > 0 and index < len(pending) - 1:
            time.sleep(sleep_seconds)

    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "paper2skills_common.paper_search",
        "source": "arxiv",
        "summary": {
            "searched_candidates": len(results),
            "papers_found": sum(len(item["papers"]) for item in results),
            "candidate_papers": sum(
                1 for item in results for paper in item["papers"] if paper["recommendation"] == "candidate"
            ),
            "errors": sum(1 for item in results if item["error"]),
        },
        "results": results,
    }


def enrich_queue_with_search(queue: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    enriched = copy.deepcopy(queue)
    by_topic = {item["topic_id"]: item for item in run.get("results", [])}
    for candidate in enriched.get("candidates", []):
        result = by_topic.get(candidate.get("topic_id"))
        if result is None:
            continue
        existing_search = candidate.get("paper_search") if isinstance(candidate.get("paper_search"), dict) else None
        if result.get("error") and not result.get("papers") and existing_search and existing_search.get("top_papers"):
            preserved = copy.deepcopy(existing_search)
            preserved["last_successful_run_id"] = preserved.get("last_successful_run_id") or preserved.get("last_run_id")
            preserved["last_error_run_id"] = run["run_id"]
            preserved["last_error"] = result.get("error")
            preserved["last_error_attempted_queries"] = result.get("attempted_queries")
            candidate["paper_search"] = preserved
            continue
        candidate["paper_search"] = {
            "last_run_id": run["run_id"],
            "source": run.get("source", "arxiv"),
            "search_query": result.get("search_query"),
            "attempted_queries": result.get("attempted_queries"),
            "error": result.get("error"),
            "top_papers": [
                {
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "url": paper.get("url"),
                    "published": paper.get("published"),
                    "search_score": paper.get("search_score"),
                    "recommendation": paper.get("recommendation"),
                    "score_reasons": paper.get("score_reasons"),
                }
                for paper in result.get("papers", [])[:5]
            ],
        }
    return enriched


def load_candidate_queue(root: str | Path | None = None, queue_path: Path | None = None) -> dict[str, Any]:
    project_root = project_root_from(Path(root) if root is not None else None)
    path = queue_path or (project_root / DEFAULT_QUEUE_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_document(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def write_search_run(root: str | Path | None, run: dict[str, Any], output_dir: Path | None = None) -> Path:
    project_root = project_root_from(Path(root) if root is not None else None)
    directory = output_dir or (project_root / DEFAULT_SEARCH_RUN_DIR)
    output = directory / f"{run['run_id']}.json"
    write_json_document(run, output)
    return output
