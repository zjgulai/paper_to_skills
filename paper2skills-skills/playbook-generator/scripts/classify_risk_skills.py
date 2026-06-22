"""
classify_risk_skills.py
将未归类的风险/诊断类 Skills 自动分配到 risk_events_ontology.yaml 的对应事件中。

用法:
  python3 classify_risk_skills.py --dry-run   # 预览分类结果
  python3 classify_risk_skills.py             # 写入 ontology.yaml
"""
import argparse
import os
import re
import yaml
from pathlib import Path

VAULT = Path("paper2skills-vault")
ONTOLOGY_PATH = Path("paper2skills-skills/playbook-generator/scripts/config/risk_events_ontology.yaml")

EVENT_KW_MAP = {
    "customer-churn-risk": [
        "churn", "retention", "repurchase", "cohort", "ltv-acq", "abandon",
        "uplift-churn", "uplift-intervention", "high-value-customer",
        "guardrailed-uplift", "rfm-segment", "rfm-campaign",
    ],
    "compliance-prescan-alert": [
        "compliance", "cpsc", "regulatory", "hts-tariff", "gpsr", "fda",
        "allergen", "recall", "vat", "gst", "patent", "contract", "baby-food",
        "infant-formula", "product-safety", "supply-chain-due",
        "gcc-cpc", "category-compliance", "listing-compliance",
    ],
    "competitor-price-attack": [
        "price-monitor", "competitive-price", "buybox", "repricing",
        "elasticity", "nash", "signaling-game", "mixed-strategy",
        "price-war", "real-time-competitive",
    ],
    "supply-chain-lead-time-risk": [
        "lead-time", "inventory-health", "replenish", "stockout", "supplier-risk",
        "safety-stock", "cvar-inventory", "supply-chain-kpi", "ito-three",
        "healthy-inventory", "customs-clearance", "shipment-risk",
        "real-time-supply-chain", "sc-resilience", "supply-chain-finance",
        "geopolitical-risk-tag",
    ],
    "fake-review-attack": [
        "review-fraud", "fake-review", "vine", "promoguardian", "review-velocity",
        "fraudsquad", "ds-dga", "crypto-anomaly", "review-defense",
        "ai-fake-review", "aigc-content", "ai-generated-content",
    ],
    "asin-traffic-drop": [
        "traffic", "keyword-cannibalization", "index-health", "listing-health",
        "creative-fatigue", "ad-fraud", "click-fraud", "seo",
        "brand-defense-search", "negative-keyword",
    ],
    "account-health-degradation": [
        "account-health", "account-association", "account-fingerprint",
        "amazon-account-appeal", "multi-account", "odr",
        "identity-fraud", "platform-policy",
    ],
    "ad-creative-fatigue": [
        "creative-fatigue", "mmm-budget", "channel-budget", "channel-saturation",
        "bayesian-mmm", "dara-agentic", "content-roi-budget",
        "amazon-tos", "negative-keyword-safe",
    ],
    "data-model-drift": [
        "data-drift", "concept-drift", "model-performance", "anomaly-detection-foundation",
        "data-quality-monitor", "argos", "conformal-risk", "experiment-data-quality",
        "forecast-bias",
    ],
    "tariff-fx-risk": [
        "montecarlo-tariff", "hts-code-risk", "hts-agentic", "customs-compliance",
        "cross-border-price", "supply-chain-total-cost", "commodity-futures",
    ],
}

PHASE_HINTS = {
    "diagnose": ["detection", "monitor", "audit", "health", "diagnos", "tracking", "scan", "prescan", "signal", "assessment", "scoring"],
    "treat": ["trigger", "dispatcher", "executor", "gate", "auto-adjuster", "repair", "resolver", "response", "recovery", "action", "reallocation"],
    "prevent": ["prevention", "guardrail", "defense", "protection", "isolation", "forecast", "strategy", "planning", "framework"],
}

RISK_KWS = [
    "risk","fraud","anomaly","monitor","alert","detect","defense","guard","diagnos",
    "health","warning","churn","compliance","protect","crisis","trigger","dispatcher",
    "executor","gate","hijack","clearance","drift","fatigue","velocity","tos","guardrail",
    "appeal","fingerprint","association","saturation","cannibalization","bias",
]


def is_risk_skill(skill_id: str) -> bool:
    sl = skill_id.lower()
    return any(kw in sl for kw in RISK_KWS)


def classify_event(skill_id: str) -> tuple[str, str]:
    sl = skill_id.lower()
    best_event, best_score = None, 0
    for event_id, kws in EVENT_KW_MAP.items():
        score = sum(kw in sl for kw in kws)
        if score > best_score:
            best_score, best_event = score, event_id
    if best_score == 0:
        return "", ""
    best_phase = "diagnose"
    for phase, hints in PHASE_HINTS.items():
        if any(h in sl for h in hints):
            best_phase = phase
            break
    return best_event, best_phase


def get_skill_role(skill_id: str, vault: Path) -> str:
    for d in vault.iterdir():
        if not d.is_dir():
            continue
        fp = d / f"{skill_id}.md"
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    if line.startswith("title:"):
                        title = line.strip()[6:].strip()
                        parts = title.split("—", 1)
                        return parts[1].strip()[:80] if len(parts) > 1 else title[:80]
    return skill_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent.parent
    vault = root / "paper2skills-vault"
    ontology_path = root / str(ONTOLOGY_PATH)

    with open(ontology_path, encoding="utf-8") as f:
        ontology = yaml.safe_load(f)

    already_mapped: set[str] = set()
    event_index: dict[str, dict] = {}
    for ev in ontology["events"]:
        event_index[ev["event_id"]] = ev
        for phase in ("diagnose", "treat", "prevent"):
            for sk in ev.get("phases", {}).get(phase, []):
                already_mapped.add(sk["skill_id"])

    to_add: list[tuple[str, str, str, str]] = []
    for d in sorted(vault.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not (f.name.startswith("Skill-") and f.name.endswith(".md")):
                continue
            sid = f.name[:-3]
            if sid in already_mapped or not is_risk_skill(sid):
                continue
            event_id, phase = classify_event(sid)
            if not event_id:
                continue
            role = get_skill_role(sid, vault)
            to_add.append((sid, event_id, phase, role))

    print(f"未归类风险 Skills: {len(to_add)} 个")
    by_event: dict[str, list] = {}
    for sid, ev, ph, role in to_add:
        by_event.setdefault(ev, []).append((sid, ph, role))

    for ev_id, items in sorted(by_event.items()):
        print(f"\n  [{ev_id}] +{len(items)} 个:")
        for sid, ph, role in items[:3]:
            print(f"    {ph:10s}  {sid}")
        if len(items) > 3:
            print(f"    ... (+{len(items)-3} more)")

    if args.dry_run:
        print("\n--dry-run 模式，不写入文件")
        return

    for sid, event_id, phase, role in to_add:
        ev = event_index[event_id]
        phases = ev.setdefault("phases", {})
        phase_list = phases.setdefault(phase, [])
        existing_ids = {sk["skill_id"] for sk in phase_list}
        if sid not in existing_ids:
            seq = max((sk.get("sequence", 0) for sk in phase_list), default=0) + 1
            entry: dict = {"skill_id": sid, "role": role}
            if phase == "diagnose":
                entry["sequence"] = seq
            phase_list.append(entry)

    with open(ontology_path, "w", encoding="utf-8") as f:
        yaml.dump(ontology, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n✅ 写入完成，{len(to_add)} 个 Skills 已归类入 ontology")


if __name__ == "__main__":
    main()
