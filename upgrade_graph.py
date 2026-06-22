import re
from pathlib import Path

def patch_graph_analyzer():
    p = Path('paper2skills-skills/paper-skills-graph/scripts/skills_graph_analyzer.py')
    code = p.read_text()
    
    # Check if we already have biz_confidence
    if "biz_confidence: float =" not in code:
        code = code.replace("    weight: float = 1.0\n", "    weight: float = 1.0\n    domain_relevance: float = 1.0\n    biz_confidence: float = 1.0\n")

    # In build_graph, we need to modify edge creation
    if "edge.domain_relevance =" not in code:
        # Find where edges are created
        replacement = """
                        edge = SkillEdge(
                            source=source_id,
                            target=target_id,
                            edge_type=rel_type,
                            weight=1.5 if is_bridge else 1.0,
                            domain_relevance=2.5 if is_bridge else 0.8,
                            biz_confidence=1.0
                        )
                        self.edges.append(edge)
"""
        code = re.sub(r'self\.edges\.append\(SkillEdge\(\s*source=source_id,\s*target=target_id,\s*edge_type=rel_type,\s*weight=1\.5 if is_bridge else 1\.0\s*\)\)', replacement, code, flags=re.DOTALL)
        
    p.write_text(code)

patch_graph_analyzer()
print("Upgraded Graph Analyzer for Alpha schema.")
