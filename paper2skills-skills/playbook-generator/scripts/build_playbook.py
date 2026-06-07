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


SECTION_KEYS = {
    "algorithm": ["算法原理"],
    "scenario": ["应用案例", "母婴出海应用案例", "业务应用"],
    "code": ["代码模板"],
    "guide": ["使用指南"],
    "value": ["业务价值", "业务价值评估"],
    "relations": ["Skill Relations", "技能关联"],
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
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    roi: list[str] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)
    code_path: str | None = None
    code_blocks: int = 0
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
    body = text[end + 4 :]
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data, body


def section_map(body: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", body, re.MULTILINE))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        for key, names in SECTION_KEYS.items():
            if any(name.lower() in title.lower() for name in names):
                sections[key] = content
    return sections


def first_nonempty_line(text: str, fallback: str = "") -> str:
    for line in text.splitlines():
        clean = re.sub(r"[#>*`\-]+", "", line).strip()
        if clean and not clean.startswith("|") and len(clean) > 8:
            return clean[:220]
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
            if len(clean) > 8:
                snippets.append(clean[:180])
        if len(snippets) >= limit:
            break
    return snippets


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
        skill = PlaybookSkill(
            skill_id=skill_id,
            title=extract_title(fm, body, skill_id),
            domain_key=domain_key,
            domain_dir=domain_dir,
            path=rel_path,
            status=fm.get("status", "unknown"),
            topic=fm.get("topic", ""),
            algorithm_summary=first_nonempty_line(sections.get("algorithm", ""), first_nonempty_line(body, skill_id)),
            problem_solved=first_nonempty_line(sections.get("scenario", ""), ""),
            business_scenarios=extract_list_snippets(sections.get("scenario", ""), 8),
            inputs=extract_list_snippets(re.sub(r"输出.*", "", sections.get("guide", ""), flags=re.DOTALL), 5),
            outputs=extract_list_snippets(sections.get("guide", ""), 5),
            roi=extract_list_snippets(sections.get("value", ""), 6),
            papers=extract_papers(text),
            code_path=code_path_str,
            code_blocks=len(re.findall(r"```(?:python)?", text)) // 2,
            relations=relations,
            tags=classify(full_text_for_classify, ALGO_TAG_RULES),
            topics=classify(full_text_for_classify, TOPIC_RULES),
            workflows=classify(full_text_for_classify, WORKFLOW_RULES),
            source_excerpt=sections.get("algorithm", body)[:1200],
        )
        if not skill.tags:
            skill.tags = [domain_key]
        if not skill.topics:
            skill.topics = ["其他"]
        skills.append(skill)
    return sorted(skills, key=lambda item: (item.domain_dir, item.skill_id))


def html_page(title: str, body: str, nav: str = "") -> str:
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
  <title>{html.escape(title)} · paper2skills Playbook</title>
  <link rel=\"stylesheet\" href=\"{nav}assets/style.css\">
</head>
<body>
  <header class=\"topbar\">
    <a class=\"brand\" href=\"{nav}index.html\">paper2skills Playbook</a>
    <input id=\"global-search\" placeholder=\"搜索 Skill / 场景 / 算法...\" autocomplete=\"off\">
  </header>
  <div id=\"search-results\" class=\"search-results hidden\"></div>
  <main class=\"layout\">
    <aside class=\"sidebar\">
      <a href=\"{nav}index.html\">总览</a>
      <a href=\"{nav}domains/index.html\">按领域</a>
      <a href=\"{nav}topics/index.html\">按主题</a>
      <a href=\"{nav}workflows/index.html\">工作流</a>
      <a href=\"{nav}graph/overview.html\">Skills Graph</a>
      <a href=\"{nav}skills/index.html\">全部 Skills</a>
    </aside>
    <section class=\"content\">{body}</section>
  </main>
  <script src=\"{nav}assets/playbook-data.js\"></script>
  <script src=\"{nav}assets/search.js\"></script>
</body>
</html>"""


def skill_url(skill_id: str, nav: str = "") -> str:
    return f"{nav}skills/{skill_id}.html"


def render_skill_card(skill: PlaybookSkill, nav: str = "") -> str:
    tags = "".join(f"<span class='tag'>{html.escape(tag)}</span>" for tag in skill.tags)
    topics = "".join(f"<span class='tag topic'>{html.escape(t)}</span>" for t in skill.topics)
    return f"""<article class=\"card skill-card\">
  <h3><a href=\"{skill_url(skill.skill_id, nav)}\">{html.escape(skill.title)}</a></h3>
  <p class=\"muted\">{html.escape(skill.domain_dir)} · {html.escape(skill.skill_id)}</p>
  <p>{html.escape(skill.algorithm_summary)}</p>
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
    body = f"""
<nav class=\"breadcrumbs\"><a href=\"../index.html\">首页</a> / <a href=\"../domains/{slugify(skill.domain_dir)}.html\">{html.escape(skill.domain_dir)}</a> / {html.escape(skill.skill_id)}</nav>
<h1>{html.escape(skill.title)}</h1>
<p class=\"muted\">{html.escape(skill.skill_id)} · {html.escape(skill.domain_dir)} · status={html.escape(skill.status)}</p>
<div class=\"tag-row\">{''.join(f"<span class='tag'>{html.escape(t)}</span>" for t in skill.tags + skill.topics + skill.workflows)}</div>
<div class=\"two-col\">
  <section>
    <h2>1. 解决的问题</h2>
    <p>{html.escape(skill.problem_solved or skill.algorithm_summary)}</p>
    <h2>2. 核心算法逻辑</h2>
    <p>{html.escape(skill.algorithm_summary)}</p>
    <h2>3. 输入</h2>{render_items(skill.inputs)}
    <h2>4. 输出</h2>{render_items(skill.outputs)}
    <h2>5. 业务场景</h2>{render_items(skill.business_scenarios)}
    <h2>6. 业务价值 / ROI</h2>{render_items(skill.roi)}
    <h2>7. 代码模板</h2>
    <p>代码块数量：{skill.code_blocks}</p>
    <p>代码路径：{html.escape(skill.code_path or '未检测到独立代码路径')}</p>
    <h2>8. 论文来源</h2>{render_items(skill.papers)}
  </section>
  <aside class=\"relation-panel\">
    <h2>Skill Relations</h2>
    <h3>前置技能</h3>{link_list(skill.relations.get('prerequisite', []), nav)}
    <h3>延伸技能</h3>{link_list(skill.relations.get('extends', []), nav)}
    <h3>可组合技能</h3>{link_list(skill.relations.get('combinable', []), nav)}
  </aside>
</div>"""
    return html_page(skill.title, body, nav)


def render_items(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>"
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_pages(out: Path, skills: list[PlaybookSkill], domains: list[dict[str, Any]], graph: SkillsGraph) -> dict[str, Any]:
    global KNOWN_SKILL_IDS
    KNOWN_SKILL_IDS = {skill.skill_id for skill in skills}
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)

    skill_count = len(skills)
    edge_count = len(graph.edges)
    domain_count = len({s.domain_dir for s in skills})

    # Data assets
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {"skill_count": skill_count, "domain_count": domain_count, "edge_count": edge_count},
        "domains": domains,
        "skills": [skill.__dict__ for skill in skills],
    }
    write_file(out / "assets" / "playbook-data.json", json.dumps(data, ensure_ascii=False, indent=2))
    write_file(out / "assets" / "playbook-data.js", "window.PLAYBOOK_DATA = " + json.dumps(data, ensure_ascii=False) + ";")

    graph_json = {
        "nodes": [{"id": node.id, "domain": node.domain, "title": node.id} for node in graph.nodes.values()],
        "links": [{"source": e.source, "target": e.target, "type": e.edge_type} for e in graph.edges],
    }
    write_file(out / "assets" / "graph-data.json", json.dumps(graph_json, ensure_ascii=False, indent=2))

    write_assets(out)

    # Index
    domain_cards = "".join(
        f"<a class='metric-card' href='domains/{slugify(d['vault_dir'])}.html'><strong>{html.escape(d['vault_dir'])}</strong><span>{d.get('skill_count', 0)} Skills</span></a>"
        for d in domains
    )
    body = f"""
<h1>paper2skills Playbook</h1>
<p class=\"lead\">面向母婴跨境电商的 Skill 使用手册：按领域、业务问题、工作流和 Skills Graph 浏览。</p>
<div class=\"metrics\">
  <div><strong>{skill_count}</strong><span>Skills</span></div>
  <div><strong>{domain_count}</strong><span>领域</span></div>
  <div><strong>{edge_count}</strong><span>关系边</span></div>
  <div><strong>0</strong><span>P0/P1 阻塞</span></div>
</div>
<h2>按领域浏览</h2><div class=\"grid\">{domain_cards}</div>
<h2>推荐入口</h2>
<div class=\"grid\">
  <a class='metric-card' href='topics/知识图谱与rag.html'>知识图谱与 RAG</a>
  <a class='metric-card' href='workflows/wf-b-广告优化.html'>WF-B 广告优化</a>
  <a class='metric-card' href='workflows/wf-d-选品扫描.html'>WF-D 选品扫描</a>
  <a class='metric-card' href='graph/overview.html'>Skills Graph</a>
</div>"""
    write_file(out / "index.html", html_page("总览", body))

    # Skills pages and index
    for skill in skills:
        write_file(out / "skills" / f"{skill.skill_id}.html", render_skill_page(skill))
    all_cards = "".join(render_skill_card(skill, "../") for skill in skills)
    write_file(out / "skills" / "index.html", html_page("全部 Skills", f"<h1>全部 Skills</h1><div class='cards'>{all_cards}</div>", "../"))

    # Domain pages
    domain_index_cards = []
    for domain in domains:
        domain_skills = [s for s in skills if s.domain_dir == domain["vault_dir"]]
        cards = "".join(render_skill_card(s, "../") for s in domain_skills)
        title = domain["vault_dir"]
        write_file(out / "domains" / f"{slugify(title)}.html", html_page(title, f"<h1>{html.escape(title)}</h1><p>{html.escape(domain.get('description',''))}</p><div class='cards'>{cards}</div>", "../"))
        domain_index_cards.append(f"<a class='metric-card' href='{slugify(title)}.html'>{html.escape(title)}<span>{len(domain_skills)} Skills</span></a>")
    write_file(out / "domains" / "index.html", html_page("按领域", "<h1>按领域</h1><div class='grid'>" + "".join(domain_index_cards) + "</div>", "../"))

    # Topic pages
    all_topics = sorted({topic for s in skills for topic in s.topics})
    topic_cards = []
    for topic in all_topics:
        topic_skills = [s for s in skills if topic in s.topics]
        cards = "".join(render_skill_card(s, "../") for s in topic_skills)
        path = f"{slugify(topic)}.html"
        write_file(out / "topics" / path, html_page(topic, f"<h1>{html.escape(topic)}</h1><div class='cards'>{cards}</div>", "../"))
        topic_cards.append(f"<a class='metric-card' href='{path}'>{html.escape(topic)}<span>{len(topic_skills)} Skills</span></a>")
    write_file(out / "topics" / "index.html", html_page("按主题", "<h1>按主题</h1><div class='grid'>" + "".join(topic_cards) + "</div>", "../"))

    # Workflow pages
    workflow_cards = []
    for workflow in WORKFLOW_RULES:
        wf_skills = [s for s in skills if workflow in s.workflows]
        cards = "".join(render_skill_card(s, "../") for s in wf_skills)
        path = f"{slugify(workflow)}.html"
        write_file(out / "workflows" / path, html_page(workflow, f"<h1>{html.escape(workflow)}</h1><p>按业务流程推荐的 Skill 链。</p><div class='cards'>{cards}</div>", "../"))
        workflow_cards.append(f"<a class='metric-card' href='{path}'>{html.escape(workflow)}<span>{len(wf_skills)} Skills</span></a>")
    write_file(out / "workflows" / "index.html", html_page("工作流", "<h1>工作流</h1><div class='grid'>" + "".join(workflow_cards) + "</div>", "../"))

    # Graph overview
    skill_ids = {skill.skill_id for skill in skills}
    def graph_cell(skill_id: str) -> str:
        escaped = html.escape(skill_id)
        if skill_id in skill_ids:
            return f"<a href='../skills/{escaped}.html'>{escaped}</a>"
        return f"<span class='muted'>{escaped}</span>"

    relation_rows = "".join(
        f"<tr><td>{graph_cell(e.source)}</td><td>{html.escape(e.edge_type)}</td><td>{graph_cell(e.target)}</td></tr>"
        for e in graph.edges[:2000]
    )
    graph_body = f"<h1>Skills Graph</h1><p>节点 {skill_count} · 边 {edge_count}</p><table><thead><tr><th>Source</th><th>Type</th><th>Target</th></tr></thead><tbody>{relation_rows}</tbody></table>"
    write_file(out / "graph" / "overview.html", html_page("Skills Graph", graph_body, "../"))

    write_file(out / "README.md", "# paper2skills Playbook\n\n打开 `index.html` 浏览。\n")
    report = {"skill_pages": skill_count, "domains": domain_count, "edges": edge_count, "generated_at": data["generated_at"]}
    write_file(out / "build-report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def write_assets(out: Path) -> None:
    css = """
:root{--bg:#f7f7f4;--panel:#fff;--ink:#202124;--muted:#667085;--line:#e5e7eb;--accent:#2563eb;--tag:#eef2ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif}.topbar{position:sticky;top:0;z-index:5;display:flex;gap:24px;align-items:center;padding:14px 22px;background:#111827;color:#fff}.brand{color:#fff;text-decoration:none;font-weight:700}#global-search{width:min(680px,55vw);padding:10px 12px;border-radius:10px;border:0}.layout{display:grid;grid-template-columns:220px 1fr;min-height:calc(100vh - 58px)}.sidebar{padding:24px 18px;border-right:1px solid var(--line);background:#fff}.sidebar a{display:block;color:#374151;text-decoration:none;padding:8px 10px;border-radius:8px}.sidebar a:hover{background:#f3f4f6}.content{padding:32px;max-width:1280px}.lead{font-size:18px;color:var(--muted)}.metrics,.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px;margin:18px 0}.metrics>div,.metric-card,.card{display:block;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;text-decoration:none;color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.03)}.metrics strong{display:block;font-size:32px}.metrics span,.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}.skill-card h3{margin-top:0}.skill-card a{color:var(--accent);text-decoration:none}.tag{display:inline-block;margin:4px 6px 0 0;padding:3px 8px;background:var(--tag);border-radius:999px;font-size:12px;color:#3730a3}.tag.topic{background:#ecfdf5;color:#047857}.breadcrumbs{color:var(--muted);margin-bottom:12px}.breadcrumbs a{color:var(--accent)}.two-col{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:28px}.relation-panel{position:sticky;top:86px;align-self:start;background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px}table{width:100%;border-collapse:collapse;background:#fff}th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left}.search-results{position:fixed;top:56px;left:260px;right:32px;max-height:65vh;overflow:auto;background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px;z-index:10;box-shadow:0 12px 30px rgba(0,0,0,.12)}.search-results.hidden{display:none}.result{display:block;padding:10px;border-bottom:1px solid var(--line);text-decoration:none;color:var(--ink)}.result:hover{background:#f9fafb}@media(max-width:900px){.layout{grid-template-columns:1fr}.sidebar{position:static;border-right:0}.two-col{grid-template-columns:1fr}#global-search{width:100%}}
"""
    js = """
(function(){
  const input=document.getElementById('global-search');
  const box=document.getElementById('search-results');
  if(!input||!box||!window.PLAYBOOK_DATA)return;
  const skills=window.PLAYBOOK_DATA.skills||[];
  input.addEventListener('input',()=>{
    const q=input.value.trim().toLowerCase();
    if(q.length<2){box.classList.add('hidden');box.innerHTML='';return;}
    const hits=skills.filter(s=>[s.skill_id,s.title,s.domain_dir,(s.tags||[]).join(' '),(s.topics||[]).join(' '),s.algorithm_summary,s.problem_solved].join(' ').toLowerCase().includes(q)).slice(0,20);
    box.innerHTML=hits.map(s=>`<a class="result" href="${rootPrefix()}skills/${s.skill_id}.html"><strong>${escapeHtml(s.title)}</strong><br><span>${escapeHtml(s.domain_dir)} · ${escapeHtml(s.skill_id)}</span></a>`).join('')||'<p class="muted">无结果</p>';
    box.classList.remove('hidden');
  });
  document.addEventListener('click',e=>{if(e.target!==input&&!box.contains(e.target))box.classList.add('hidden')});
  function rootPrefix(){const path=location.pathname; if(path.includes('/skills/')||path.includes('/domains/')||path.includes('/topics/')||path.includes('/workflows/')||path.includes('/graph/'))return '../'; return '';}
  function escapeHtml(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
})();
"""
    write_file(out / "assets" / "style.css", css.strip())
    write_file(out / "assets" / "search.js", js.strip())


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


def build(root: Path, vault: Path, out: Path) -> dict[str, Any]:
    graph = build_graph(vault)
    skills = build_skills(root, vault, graph)
    domains = domain_dicts(root)
    return render_pages(out, skills, domains, graph)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper2skills static Playbook")
    parser.add_argument("--root", default=str(BASE_DIR))
    parser.add_argument("--vault", default="paper2skills-vault")
    parser.add_argument("--out", default="playbook")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    vault = (root / args.vault).resolve() if not Path(args.vault).is_absolute() else Path(args.vault)
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    report = build(root, vault, out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
