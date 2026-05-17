"""
Knowledge Graph Relation Completion — CBLiP-inspired implementation
Skill: Skill-KG-Relation-Completion-CBLiP.md
"""

import numpy as np
from collections import defaultdict


class SimpleKG:
    """简化版知识图谱"""

    def __init__(self):
        self.entities = {}
        self.relations = defaultdict(list)
        self.entity_embeddings = {}

    def add_entity(self, eid, etype, name):
        self.entities[eid] = {'type': etype, 'name': name}

    def add_relation(self, head, rel, tail):
        self.relations[(head, rel)].append(tail)

    def get_neighbors(self, eid):
        return [(r, t) for (h, r), tails in self.relations.items() if h == eid for t in tails]


class CBLiPScorer:
    """CBLiP-inspired relation completion scorer"""

    def __init__(self, kg, dim=64):
        self.kg = kg
        np.random.seed(42)
        for eid in kg.entities:
            kg.entity_embeddings[eid] = np.random.randn(dim)

    def score_triple(self, head, rel, tail):
        if head not in self.kg.entity_embeddings or tail not in self.kg.entity_embeddings:
            return 0.0
        h_emb = self.kg.entity_embeddings[head]
        t_emb = self.kg.entity_embeddings[tail]
        base = -np.linalg.norm(h_emb - t_emb)

        h_n = self.kg.get_neighbors(head)
        t_n = self.kg.get_neighbors(tail)
        h_types = set(r for r, _ in h_n)
        t_types = set(r for r, _ in t_n)
        sim = len(h_types & t_types) / max(len(h_types), len(t_types), 1)
        return base + 0.5 * sim

    def predict_tail(self, head, rel, candidates=None, top_k=5):
        if candidates is None:
            candidates = [e for e, info in self.kg.entities.items() if info['type'] != self.kg.entities[head]['type']]
        scores = [(t, self.score_triple(head, rel, t)) for t in candidates]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


if __name__ == '__main__':
    kg = SimpleKG()
    kg.add_entity('b_aptamil', 'brand', 'Aptamil')
    kg.add_entity('p_fm1', 'product', 'Aptamil 3段')
    kg.add_entity('p_fm2', 'product', 'Aptamil 2段')
    kg.add_entity('i_dha', 'ingredient', 'DHA')
    kg.add_entity('i_probiotic', 'ingredient', '益生菌')

    kg.add_relation('b_aptamil', 'produces', 'p_fm1')
    kg.add_relation('b_aptamil', 'produces', 'p_fm2')
    kg.add_relation('p_fm1', 'contains', 'i_dha')
    kg.add_relation('p_fm1', 'contains', 'i_probiotic')
    kg.add_relation('p_fm2', 'contains', 'i_dha')

    scorer = CBLiPScorer(kg)
    print("预测 p_fm2 的 contains 关系:")
    for tail, score in scorer.predict_tail('p_fm2', 'contains', ['i_dha', 'i_probiotic']):
        print(f"  {kg.entities[tail]['name']}: {score:.3f}")
