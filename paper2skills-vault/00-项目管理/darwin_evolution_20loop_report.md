# paper2skills Darwin Evolution 20-Loop Report

- Run ID: `20260606T005350`
- Dry run: `False`
- Mechanisms: mutation, heredity, selection, competition, niche
- Autoresearch mutation sources: roadmap, skills_graph_gaps
- Graph state: 320 nodes / 5501 edges / density 17.191
- Selection gates: inherit=19, observe=1, reject=0

Autoresearch mutation uses graph gaps and roadmap-derived paper topics; Darwin selection keeps all decisions auditable.

## 20 Loop Table

| Loop | Candidate ID | Candidate | Domain | Fitness | Gate | Action |
|---|---|---|---|---:|---|---|
| L01 | GAP-P2-008-DOMAIN-REVIEW | compliance 已纳入图谱健康检查，当前 6 个 Skill，无 P0/P1 断链 | compliance | 0.514 | observe | domain_health_monitoring |
| L02 | GAP-P2-009-MISSING-BRIDGE | ab_testing 与 data_collection 之间缺少桥梁连接 | ab_testing | 0.714 | inherit | autoresearch_bridge_skill |
| L03 | GAP-P2-010-MISSING-BRIDGE | ai_humanities 与 data_collection 之间缺少桥梁连接 | ai_humanities | 0.714 | inherit | autoresearch_bridge_skill |
| L04 | GAP-P2-011-MISSING-BRIDGE | causal_inference 与 data_collection 之间缺少桥梁连接 | causal_inference | 0.714 | inherit | autoresearch_bridge_skill |
| L05 | GAP-P2-012-MISSING-BRIDGE | data_agent_llm 与 data_collection 之间缺少桥梁连接 | data_agent_llm | 0.714 | inherit | autoresearch_bridge_skill |
| L06 | GAP-P2-013-MISSING-BRIDGE | knowledge_graph 与 data_collection 之间缺少桥梁连接 | knowledge_graph | 0.714 | inherit | autoresearch_bridge_skill |
| L07 | GAP-P2-014-MISSING-BRIDGE | logistics 与 data_collection 之间缺少桥梁连接 | logistics | 0.714 | inherit | autoresearch_bridge_skill |
| L08 | GAP-P2-015-MISSING-BRIDGE | logistics 与 risk_fraud 之间缺少桥梁连接 | logistics | 0.714 | inherit | autoresearch_bridge_skill |
| L09 | GAP-P2-016-MISSING-BRIDGE | logistics 与 visual_content 之间缺少桥梁连接 | logistics | 0.714 | inherit | autoresearch_bridge_skill |
| L10 | GAP-P2-017-MISSING-BRIDGE | marketing 与 data_collection 之间缺少桥梁连接 | marketing | 0.714 | inherit | autoresearch_bridge_skill |
| L11 | GAP-P2-018-MISSING-BRIDGE | marketing 与 logistics 之间缺少桥梁连接 | marketing | 0.714 | inherit | autoresearch_bridge_skill |
| L12 | GAP-P2-019-MISSING-BRIDGE | pricing 与 data_collection 之间缺少桥梁连接 | pricing | 0.714 | inherit | autoresearch_bridge_skill |
| L13 | GAP-P2-020-MISSING-BRIDGE | recommendation 与 advertising 之间缺少桥梁连接 | recommendation | 0.714 | inherit | autoresearch_bridge_skill |
| L14 | GAP-P2-021-MISSING-BRIDGE | recommendation 与 data_agent_llm 之间缺少桥梁连接 | recommendation | 0.714 | inherit | autoresearch_bridge_skill |
| L15 | GAP-P2-022-MISSING-BRIDGE | recommendation 与 data_collection 之间缺少桥梁连接 | recommendation | 0.714 | inherit | autoresearch_bridge_skill |
| L16 | GAP-P2-023-MISSING-BRIDGE | recommendation 与 marketing 之间缺少桥梁连接 | recommendation | 0.714 | inherit | autoresearch_bridge_skill |
| L17 | GAP-P2-024-MISSING-BRIDGE | recommendation 与 mas 之间缺少桥梁连接 | recommendation | 0.714 | inherit | autoresearch_bridge_skill |
| L18 | GAP-P2-025-MISSING-BRIDGE | recommendation 与 pricing 之间缺少桥梁连接 | recommendation | 0.714 | inherit | autoresearch_bridge_skill |
| L19 | GAP-P2-026-MISSING-BRIDGE | risk_fraud 与 data_collection 之间缺少桥梁连接 | risk_fraud | 0.714 | inherit | autoresearch_bridge_skill |
| L20 | GAP-P2-027-MISSING-BRIDGE | time_series 与 data_collection 之间缺少桥梁连接 | time_series | 0.714 | inherit | autoresearch_bridge_skill |

## Stability Checkpoints

| Loop | Nodes | Edges | Density | P0 | P1 | P2 | Mutation Paused |
|---|---:|---:|---:|---:|---:|---:|---|
| L05 | 320 | 5501 | 17.191 | 2 | 0 | 26 | True |
| L10 | 320 | 5501 | 17.191 | 2 | 0 | 26 | True |
| L15 | 320 | 5501 | 17.191 | 2 | 0 | 26 | True |
| L20 | 320 | 5501 | 17.191 | 2 | 0 | 26 | True |

## Escape Valves

- Triggered: `True`
- Consecutive low fitness loops: `1`
- Policy: `switch_to_manual_or_relation_backfill`
