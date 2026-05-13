"""
iReFeed: interconnected ReFeed Pipeline
基于论文: Enhancing User-Feedback Driven Requirements Prioritization (arXiv:2603.28677)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class UserFeedback:
    """单条用户反馈"""
    text: str
    sentiment_neg: float = 0.0
    sentiment_pos: float = 0.0
    intention: float = 0.0


@dataclass
class FeedbackCluster:
    """主题簇"""
    cluster_id: int
    feedbacks: List[UserFeedback] = field(default_factory=list)
    top_keywords: List[str] = field(default_factory=list)

    @property
    def aggregated_score(self) -> float:
        return sum(f.sentiment_neg + f.sentiment_pos + f.intention for f in self.feedbacks)


@dataclass
class CandidateRequirement:
    """候选需求"""
    req_id: str
    name: str
    description: str
    estimated_cost: float
    stakeholder_value: float = 0.0


class iReFeedPrioritizer:
    """iReFeed 需求优先级排序器"""

    def __init__(self, n_topics: int = 5, pop_size: int = 80, generations: int = 60):
        self.n_topics = n_topics
        self.pop_size = pop_size
        self.generations = generations
        self.clusters: List[FeedbackCluster] = []
        self.requires_map: Dict[str, Set[str]] = defaultdict(set)
        self.d_values: Dict[str, float] = {}

    def fit_topics(self, feedbacks: List[UserFeedback]) -> List[FeedbackCluster]:
        """LDA 主题建模"""
        feedback_texts = [fb.text for fb in feedbacks]
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
        doc_term_matrix = vectorizer.fit_transform(feedback_texts)
        lda = LatentDirichletAllocation(
            n_components=self.n_topics, random_state=42,
            learning_method="online", max_iter=20,
        )
        doc_topic_dist = lda.fit_transform(doc_term_matrix)
        feature_names = vectorizer.get_feature_names_out()

        clusters = []
        for t in range(self.n_topics):
            top_indices = np.argsort(lda.components_[t])[-10:]
            keywords = [feature_names[i] for i in reversed(top_indices)]
            clusters.append(FeedbackCluster(cluster_id=t, top_keywords=keywords))

        for idx, dist in enumerate(doc_topic_dist):
            dominant_topic = int(np.argmax(dist))
            clusters[dominant_topic].feedbacks.append(feedbacks[idx])

        self.clusters = [c for c in clusters if len(c.feedbacks) > 0]
        return self.clusters

    def compute_priorities(
        self, requirements: List[CandidateRequirement], feedback_texts: List[str]
    ) -> Dict[str, float]:
        """簇级反馈关联优先级 P_r"""
        tfidf = TfidfVectorizer(max_df=0.95, min_df=1, stop_words="english")
        all_texts = feedback_texts + [r.description for r in requirements]
        tfidf_matrix = tfidf.fit_transform(all_texts)
        req_vectors = tfidf_matrix[len(feedback_texts):]

        priorities: Dict[str, float] = {}
        for i, req in enumerate(requirements):
            req_vec = req_vectors[i].reshape(1, -1)
            cluster_scores = []
            req_words = set(req.name.lower().split() + req.description.lower().split())

            for cluster in self.clusters:
                if len(cluster.feedbacks) == 0:
                    continue
                cluster_indices = []
                for fb in cluster.feedbacks:
                    try:
                        idx = feedback_texts.index(fb.text)
                        cluster_indices.append(idx)
                    except ValueError:
                        continue
                if len(cluster_indices) == 0:
                    continue

                cluster_vectors = tfidf_matrix[cluster_indices]
                sims = cosine_similarity(req_vec, cluster_vectors).flatten()
                alpha = 1.0  # 生产环境可按簇内需求相似度计算

                score = 0.0
                for j, cidx in enumerate(cluster_indices):
                    fb = cluster.feedbacks[j]
                    sentiment_score = fb.sentiment_neg + fb.sentiment_pos + fb.intention
                    sim = float(sims[j])
                    if sim < 1e-3:
                        fb_words = set(fb.text.lower().split())
                        overlap = len(req_words & fb_words)
                        sim = overlap / max(len(req_words), 1)
                    score += alpha * sim * sentiment_score

                score /= len(cluster_indices)
                cluster_scores.append(score)

            priorities[req.req_id] = max(cluster_scores) if cluster_scores else 0.5
        return priorities

    def discover_dependencies(self, requirements: List[CandidateRequirement]) -> Dict[str, Set[str]]:
        """模拟 LLM 发现 requires 关系"""
        name_to_id = {r.name: r.req_id for r in requirements}
        self.requires_map = defaultdict(set)
        patterns = [
            ("app", ["bluetooth", "connectivity", "data"]),
            ("smart", ["sensor", "bluetooth"]),
            ("fast charge", ["type-c", "usb-c", "charging port"]),
            ("memory", ["multi-level", "adjustable", "mode"]),
            ("silent", ["motor", "noise reduction"]),
        ]
        for req in requirements:
            req_lower = req.name.lower()
            for trigger, deps in patterns:
                if trigger in req_lower:
                    for dep_keyword in deps:
                        for name, rid in name_to_id.items():
                            if rid == req.req_id:
                                continue
                            if dep_keyword in name.lower():
                                self.requires_map[req.req_id].add(rid)
        return dict(self.requires_map)

    def compute_d_values(self, requirements: List[CandidateRequirement]) -> Dict[str, float]:
        """D-value_i = count_i / |D|"""
        self.d_values = {r.req_id: 0.0 for r in requirements}
        if not self.requires_map:
            self.discover_dependencies(requirements)
        total_relations = sum(len(deps) for deps in self.requires_map.values())
        if total_relations == 0:
            return self.d_values
        count: Dict[str, int] = defaultdict(int)
        for deps in self.requires_map.values():
            for dep in deps:
                count[dep] += 1
        for req in requirements:
            self.d_values[req.req_id] = count[req.req_id] / total_relations
        return self.d_values

    def optimize_selection(
        self, requirements: List[CandidateRequirement], priorities: Dict[str, float], budget: float
    ) -> Tuple[List[CandidateRequirement], np.ndarray]:
        """NSGA-II 三目标优化"""
        self.compute_d_values(requirements)
        n = len(requirements)
        priority_vec = np.array([priorities[r.req_id] for r in requirements])
        cost_vec = np.array([r.estimated_cost for r in requirements])
        d_vec = np.array([self.d_values[r.req_id] for r in requirements])

        idx_map = {r.req_id: i for i, r in enumerate(requirements)}
        req_idx_requires = {
            idx_map[k]: {idx_map[v] for v in vals if v in idx_map}
            for k, vals in self.requires_map.items() if k in idx_map
        }

        nsga2 = _LightweightNSGA2(pop_size=self.pop_size, generations=self.generations)
        best_x = nsga2.optimize(n, priority_vec, cost_vec, d_vec, budget, req_idx_requires)
        best_x = self._repair_budget(best_x, cost_vec, budget, req_idx_requires)
        selected = [requirements[i] for i, x in enumerate(best_x) if x]
        return selected, best_x

    def _repair_budget(
        self, x: np.ndarray, cost_vec: np.ndarray, budget: float, requires_map: Dict[int, Set[int]]
    ) -> np.ndarray:
        x = x.copy()
        required_by: Dict[int, Set[int]] = defaultdict(set)
        for i, deps in requires_map.items():
            for j in deps:
                required_by[j].add(i)
        while np.sum(cost_vec * x) > budget:
            selected = set(int(i) for i in np.where(x)[0])
            removable = [i for i in selected if not (required_by.get(i, set()) & selected)]
            if not removable:
                break
            worst = max(removable, key=lambda i: cost_vec[i])
            x[worst] = False
        return x


class _LightweightNSGA2:
    """轻量级 NSGA-II 实现 (纯 numpy)"""

    def __init__(self, pop_size: int = 80, generations: int = 60, seed: int = 42):
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        import random
        random.seed(seed)
        np.random.seed(seed)

    def optimize(
        self, n: int, priority_vec: np.ndarray, cost_vec: np.ndarray,
        d_vec: np.ndarray, budget: float, requires_map: Dict[int, Set[int]]
    ) -> np.ndarray:
        pop = [np.random.randint(0, 2, size=n).astype(bool) for _ in range(self.pop_size)]
        pop = [self._repair(p, requires_map) for p in pop]

        for _ in range(self.generations):
            objs = np.array([self._evaluate(p, priority_vec, cost_vec, d_vec, budget) for p in pop])
            fronts = self._non_dominated_sort(objs)
            cd = self._crowding_distance(objs, fronts)

            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament_selection(pop, fronts, cd)
                p2 = self._tournament_selection(pop, fronts, cd)
                c1, c2 = self._crossover(p1, p2)
                c1, c2 = self._mutate(c1), self._mutate(c2)
                c1, c2 = self._repair(c1, requires_map), self._repair(c2, requires_map)
                offspring.extend([c1, c2])
            offspring = offspring[:self.pop_size]

            combined = pop + offspring
            combined_objs = np.array([self._evaluate(p, priority_vec, cost_vec, d_vec, budget) for p in combined])
            combined_fronts = self._non_dominated_sort(combined_objs)
            combined_cd = self._crowding_distance(combined_objs, combined_fronts)
            pop = self._environmental_selection(combined, combined_fronts, combined_cd)

        final_objs = np.array([self._evaluate(p, priority_vec, cost_vec, d_vec, budget) for p in pop])
        final_fronts = self._non_dominated_sort(final_objs)
        front0 = [i for i, f in enumerate(final_fronts) if f == 0]
        best_idx = min(front0, key=lambda i: final_objs[i, 0] + final_objs[i, 2])
        return pop[best_idx]

    def _evaluate(self, x, priority_vec, cost_vec, d_vec, budget):
        f1 = -np.sum(priority_vec * x)
        f2 = np.sum(cost_vec * x)
        f3 = -np.sum(d_vec * x)
        penalty = max(0.0, np.sum(cost_vec * x) - budget) * 10.0
        f2 += penalty
        return np.array([f1, f2, f3])

    def _repair(self, x, requires_map):
        x = x.copy()
        changed = True
        while changed:
            changed = False
            for i, selected in enumerate(x):
                if selected and i in requires_map:
                    for j in requires_map[i]:
                        if not x[j]:
                            x[j] = True
                            changed = True
        return x

    def _tournament_selection(self, pop, fronts, cd, tournament_size=2):
        import random
        best = random.choice(range(len(pop)))
        for _ in range(tournament_size - 1):
            contender = random.choice(range(len(pop)))
            if fronts[contender] < fronts[best]:
                best = contender
            elif fronts[contender] == fronts[best] and cd[contender] > cd[best]:
                best = contender
        from copy import deepcopy
        return deepcopy(pop[best])

    def _crossover(self, p1, p2):
        import random
        if random.random() > self.crossover_prob:
            from copy import deepcopy
            return deepcopy(p1), deepcopy(p2)
        point = random.randint(1, len(p1) - 1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def _mutate(self, x):
        import random
        x = x.copy()
        for i in range(len(x)):
            if random.random() < self.mutation_prob:
                x[i] = not x[i]
        return x

    def _non_dominated_sort(self, objs):
        n = len(objs)
        fronts = np.full(n, -1, dtype=int)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [set() for _ in range(n)]
        front1 = []
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                    dominated_solutions[p].add(q)
                elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                fronts[p] = 0
                front1.append(p)
        i = 0
        current_front = front1
        while current_front:
            next_front = []
            for p in current_front:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        fronts[q] = i + 1
                        next_front.append(q)
            i += 1
            current_front = next_front
        return fronts

    def _crowding_distance(self, objs, fronts):
        n = len(objs)
        cd = np.zeros(n)
        max_front = fronts.max()
        for f in range(max_front + 1):
            indices = np.where(fronts == f)[0]
            if len(indices) <= 2:
                cd[indices] = np.inf
                continue
            front_objs = objs[indices]
            distances = np.zeros(len(indices))
            for m in range(objs.shape[1]):
                sorted_idx = np.argsort(front_objs[:, m])
                distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
                denom = front_objs[sorted_idx[-1], m] - front_objs[sorted_idx[0], m]
                if denom > 1e-9:
                    for k in range(1, len(indices) - 1):
                        distances[sorted_idx[k]] += (
                            front_objs[sorted_idx[k + 1], m] - front_objs[sorted_idx[k - 1], m]
                        ) / denom
            cd[indices] = distances
        return cd

    def _environmental_selection(self, combined, fronts, cd):
        selected = []
        front_idx = 0
        while len(selected) < self.pop_size:
            indices = np.where(fronts == front_idx)[0]
            if len(indices) == 0:
                front_idx += 1
                continue
            if len(selected) + len(indices) <= self.pop_size:
                selected.extend([combined[i] for i in indices])
            else:
                remaining = self.pop_size - len(selected)
                sorted_indices = indices[np.argsort(-cd[indices])]
                selected.extend([combined[i] for i in sorted_indices[:remaining]])
            front_idx += 1
        return selected[:self.pop_size]


def generate_test_data():
    """生成Momcozy季度需求排序测试数据"""
    feedback_templates = [
        # (text, neg, pos, intention) - sentiment scores for meaningful priority calc
        ("The motor noise is too loud when pumping at night, wakes up my baby.", 0.85, 0.05, 0.70),
        ("Wish it had Type-C charging, much more convenient.", 0.20, 0.60, 0.80),
        ("App connection is unreliable, keeps disconnecting.", 0.80, 0.05, 0.65),
        ("Love the silent mode but wish it was quieter.", 0.40, 0.50, 0.55),
        ("Need a portable travel case for pumping on the go.", 0.10, 0.30, 0.75),
        ("Memory function would be great, save my preferred settings.", 0.05, 0.70, 0.60),
        ("Bluetooth connection drops frequently, very frustrating.", 0.75, 0.05, 0.50),
        ("Fast charging would be a game changer for working moms.", 0.10, 0.80, 0.85),
        ("The noise reduction works well but could be better.", 0.30, 0.60, 0.45),
        ("Compact design is nice but need more accessories.", 0.15, 0.55, 0.40),
    ]
    feedbacks = [
        UserFeedback(text=t, sentiment_neg=neg, sentiment_pos=pos, intention=intent)
        for t, neg, pos, intent in feedback_templates
        for _ in range(100)  # 每条模板复制100次 = 1000条评论
    ]

    requirements = [
        CandidateRequirement("R1", "Silent Motor Upgrade", "Upgrade motor to reduce noise below 35dB", 50.0),
        CandidateRequirement("R2", "Type-C Interface", "Replace charging port with Type-C", 20.0),
        CandidateRequirement("R3", "Bluetooth Module", "Upgrade Bluetooth 5.0 module for stable connection", 30.0),
        CandidateRequirement("R4", "Smart App Control", "Mobile app for remote control and monitoring", 80.0),
        CandidateRequirement("R5", "Fast Charge Support", "Support 30min fast charge to 80%", 40.0),
        CandidateRequirement("R6", "Memory Mode", "Save and recall user preferred settings", 25.0),
        CandidateRequirement("R7", "Portable Travel Case", "Compact travel case with accessories", 15.0),
        CandidateRequirement("R8", "Noise Reduction", "Active noise cancellation technology", 60.0),
    ]
    return feedbacks, requirements


if __name__ == "__main__":
    print("=" * 60)
    print("iReFeed: 需求优先级排序 - Momcozy 测试案例")
    print("=" * 60)

    feedbacks, requirements = generate_test_data()
    budget = 120.0

    prioritizer = iReFeedPrioritizer(n_topics=3, pop_size=50, generations=30)

    # Step 1: Topic modeling
    clusters = prioritizer.fit_topics(feedbacks)
    print(f"\n[1] 主题建模完成: 发现 {len(clusters)} 个有效主题簇")
    for c in clusters:
        print(f"    簇 {c.cluster_id}: {len(c.feedbacks)} 条反馈, 关键词: {c.top_keywords[:5]}")

    # Step 2: Compute priorities
    feedback_texts = [fb.text for fb in feedbacks]
    priorities = prioritizer.compute_priorities(requirements, feedback_texts)
    print(f"\n[2] 需求优先级计算完成:")
    sorted_reqs = sorted(requirements, key=lambda r: priorities[r.req_id], reverse=True)
    for r in sorted_reqs:
        print(f"    {r.name}: P={priorities[r.req_id]:.3f}, Cost={r.estimated_cost}")

    # Step 3: Discover dependencies
    deps = prioritizer.discover_dependencies(requirements)
    print(f"\n[3] 依赖关系发现完成:")
    for req_id, dep_set in deps.items():
        if dep_set:
            req_name = next(r.name for r in requirements if r.req_id == req_id)
            dep_names = [next(r.name for r in requirements if r.req_id == d) for d in dep_set]
            print(f"    {req_name} requires: {dep_names}")

    # Step 4: D-values
    d_values = prioritizer.compute_d_values(requirements)
    print(f"\n[4] D-value 计算完成 (被依赖度):")
    for r in sorted(requirements, key=lambda r: d_values[r.req_id], reverse=True):
        if d_values[r.req_id] > 0:
            print(f"    {r.name}: D={d_values[r.req_id]:.3f}")

    # Step 5: NSGA-II optimization
    selected, selection_vec = prioritizer.optimize_selection(requirements, priorities, budget)
    total_cost = sum(r.estimated_cost for r in selected)
    print(f"\n[5] NSGA-II 优化完成 (预算={budget}):")
    print(f"    选中 {len(selected)}/{len(requirements)} 个需求, 总成本={total_cost:.1f}")
    for r in selected:
        print(f"    + {r.name} (成本={r.estimated_cost}, P={priorities[r.req_id]:.3f})")

    print(f"\n{'=' * 60}")
    print("测试通过: iReFeed pipeline 端到端运行成功")
    print(f"{'=' * 60}")
