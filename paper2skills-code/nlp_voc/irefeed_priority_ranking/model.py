"""
iReFeed: interconnected ReFeed Pipeline
基于论文: Enhancing User-Feedback Driven Requirements Prioritization (arXiv:2603.28677)

核心流程:
1. Topic Modeling (LDA) 将用户反馈聚类为主题簇
2. 将候选需求映射到主题簇，计算簇级反馈优先级
3. 引入簇内聚性增强因子 (coherence factor)
4. 用模板模拟 LLM 发现需求间 "requires" 关系，计算 D-value
5. NSGA-II 三目标优化求解 Next Release Problem
"""

from __future__ import annotations

import random
from copy import deepcopy
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
    sentiment_neg: float = 0.0  # 负面情感强度 [0,1]
    sentiment_pos: float = 0.0  # 正面情感强度 [0,1]
    intention: float = 0.0       # 功能请求意图强度 [0,1]


@dataclass
class FeedbackCluster:
    """主题簇 (Topic Cluster)"""
    cluster_id: int
    feedbacks: List[UserFeedback] = field(default_factory=list)
    top_keywords: List[str] = field(default_factory=list)

    @property
    def aggregated_score(self) -> float:
        """簇内反馈的情感+意图总和"""
        return sum(
            f.sentiment_neg + f.sentiment_pos + f.intention
            for f in self.feedbacks
        )


@dataclass
class CandidateRequirement:
    """候选需求/功能"""
    req_id: str
    name: str
    description: str
    estimated_cost: float
    stakeholder_value: float = 0.0


class TopicModeler:
    """LDA 主题建模器"""

    def __init__(self, n_topics: int = 5, random_state: int = 42):
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer: CountVectorizer | None = None
        self.lda: LatentDirichletAllocation | None = None

    def fit(self, feedback_texts: List[str]) -> List[FeedbackCluster]:
        """对用户反馈进行 LDA 主题聚类"""
        self.vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
        doc_term_matrix = self.vectorizer.fit_transform(feedback_texts)

        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            learning_method="online",
            max_iter=20,
        )
        doc_topic_dist = self.lda.fit_transform(doc_term_matrix)

        feature_names = self.vectorizer.get_feature_names_out()
        clusters = []
        for t in range(self.n_topics):
            top_indices = np.argsort(self.lda.components_[t])[-10:]
            keywords = [feature_names[i] for i in reversed(top_indices)]
            clusters.append(FeedbackCluster(cluster_id=t, top_keywords=keywords))

        # 将每条反馈分配到主导主题
        for idx, dist in enumerate(doc_topic_dist):
            dominant_topic = int(np.argmax(dist))
            clusters[dominant_topic].feedbacks.append(
                UserFeedback(text=feedback_texts[idx])
            )

        # 过滤空簇
        clusters = [c for c in clusters if len(c.feedbacks) > 0]
        return clusters


class iReFeedPrioritizer:
    """
    iReFeed 需求优先级排序器

    实现论文核心公式:
    - 簇级反馈关联
    - 簇内聚性增强因子 alpha(F_C)
    - D-value 依赖价值
    - NSGA-II 三目标优化
    """

    def __init__(self, n_topics: int = 5, pop_size: int = 80, generations: int = 60):
        self.n_topics = n_topics
        self.pop_size = pop_size
        self.generations = generations
        self.topic_modeler = TopicModeler(n_topics=n_topics)
        self.clusters: List[FeedbackCluster] = []
        self.requires_map: Dict[str, Set[str]] = defaultdict(set)
        self.d_values: Dict[str, float] = {}

    def fit_topics(self, feedback_texts: List[str]) -> List[FeedbackCluster]:
        """步骤1: 对用户反馈进行主题建模"""
        self.clusters = self.topic_modeler.fit(feedback_texts)
        return self.clusters

    def compute_priorities(
        self,
        requirements: List[CandidateRequirement],
        feedback_texts: List[str],
    ) -> Dict[str, float]:
        """
        步骤2: 将候选需求映射到主题簇并计算优先级 P_r
        """
        if not self.clusters:
            self.fit_topics(feedback_texts)

        # TF-IDF 向量化反馈文本和需求描述
        tfidf = TfidfVectorizer(max_df=0.95, min_df=1, stop_words="english")
        all_texts = feedback_texts + [r.description for r in requirements]
        tfidf_matrix = tfidf.fit_transform(all_texts)

        req_vectors = tfidf_matrix[len(feedback_texts):]

        priorities: Dict[str, float] = {}
        for i, req in enumerate(requirements):
            req_vec = req_vectors[i].reshape(1, -1)

            # 计算该需求与每个簇的相似度和优先级贡献
            cluster_scores = []
            for cluster in self.clusters:
                if len(cluster.feedbacks) == 0:
                    continue

                # 提取簇内反馈的向量
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

                # 簇内聚性因子 alpha(F_C)
                # 简化: 模板中固定为 1.0; 生产环境应计算该簇内需求描述的 pairwise similarity
                alpha = 1.0

                # P_r 计算: sum[ alpha * sim(r, F_C) * (neg + pos + int) ] / |F_C|
                score = 0.0
                req_words = set(req.name.lower().split() + req.description.lower().split())
                for j, cidx in enumerate(cluster_indices):
                    fb = cluster.feedbacks[j]
                    sentiment_score = fb.sentiment_neg + fb.sentiment_pos + fb.intention
                    # TF-IDF cosine similarity + keyword overlap fallback
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

    def discover_dependencies(
        self, requirements: List[CandidateRequirement]
    ) -> Dict[str, Set[str]]:
        """
        步骤3: 模拟 LLM 发现需求间 "requires" 关系
        实际部署中应调用 LLM API (如 ChatGPT) 进行关系抽取
        """
        req_names = {r.req_id: r.name for r in requirements}
        name_to_id = {r.name: r.req_id for r in requirements}

        self.requires_map = defaultdict(set)

        # 基于关键词模式的规则推断 (模拟 LLM 输出)
        dependency_patterns = [
            ("app", ["bluetooth", "wifi", "connectivity", "data"]),
            ("smart", ["sensor", "bluetooth", "connectivity"]),
            ("fast charge", ["type-c", "usb-c", "charging port"]),
            ("memory", ["multi-level", "adjustable", "mode"]),
            ("silent", ["motor", "noise reduction", "pump"]),
            ("portable", ["battery", "lightweight", "compact"]),
        ]

        for req in requirements:
            req_lower = req.name.lower()
            for trigger, deps in dependency_patterns:
                if trigger in req_lower:
                    for dep_keyword in deps:
                        for name, rid in name_to_id.items():
                            if rid == req.req_id:
                                continue
                            if dep_keyword in name.lower():
                                self.requires_map[req.req_id].add(rid)

        return dict(self.requires_map)

    def compute_d_values(
        self, requirements: List[CandidateRequirement]
    ) -> Dict[str, float]:
        """
        步骤4: 计算依赖价值 D-value
        D-value_i = count_i / |D|
        其中 count_i 是需求 i 作为 "requires" 右侧出现的次数
        """
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
        self,
        requirements: List[CandidateRequirement],
        priorities: Dict[str, float],
        budget: float,
    ) -> Tuple[List[CandidateRequirement], np.ndarray]:
        """
        步骤5: NSGA-II 三目标优化求解 Next Release Problem

        目标:
        1. 最大化利益相关者价值 (sum priority_i * x_i)
        2. 最小化开发成本 (sum cost_i * x_i)
        3. 最大化依赖价值 D-value (sum d_i * x_i)

        约束:
        - 总成本 <= budget
        - 依赖满足: 若选 i 且 i requires j, 则必须选 j
        """
        self.compute_d_values(requirements)

        req_ids = [r.req_id for r in requirements]
        n = len(requments := requirements)
        priority_vec = np.array([priorities[r.req_id] for r in requirements])
        cost_vec = np.array([r.estimated_cost for r in requirements])
        d_vec = np.array([self.d_values[r.req_id] for r in requirements])

        # 建立索引映射
        idx_map = {r.req_id: i for i, r in enumerate(requirements)}
        req_idx_requires: Dict[int, Set[int]] = {
            idx_map[k]: {idx_map[v] for v in vals if v in idx_map}
            for k, vals in self.requires_map.items()
            if k in idx_map
        }

        nsga2 = _LightweightNSGA2(pop_size=self.pop_size, generations=self.generations)
        best_x = nsga2.optimize(
            n=n,
            priority_vec=priority_vec,
            cost_vec=cost_vec,
            d_vec=d_vec,
            budget=budget,
            requires_map=req_idx_requires,
        )

        # 后处理: 若超预算, 贪心移除高成本且不被依赖的项目
        best_x = self._repair_budget(best_x, cost_vec, budget, req_idx_requires)

        selected = [requirements[i] for i, x in enumerate(best_x) if x]
        return selected, best_x

    def _repair_budget(
        self,
        x: np.ndarray,
        cost_vec: np.ndarray,
        budget: float,
        requires_map: Dict[int, Set[int]],
    ) -> np.ndarray:
        """若超预算, 移除不被其他选中项依赖的最高成本项"""
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
    """
    轻量级 NSGA-II 实现 (纯 numpy)
    支持二元决策变量、依赖修复、预算约束
    """

    def __init__(self, pop_size: int = 80, generations: int = 60, seed: int = 42):
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        random.seed(seed)
        np.random.seed(seed)

    def optimize(
        self,
        n: int,
        priority_vec: np.ndarray,
        cost_vec: np.ndarray,
        d_vec: np.ndarray,
        budget: float,
        requires_map: Dict[int, Set[int]],
    ) -> np.ndarray:
        # 初始化种群
        pop = [np.random.randint(0, 2, size=n).astype(bool) for _ in range(self.pop_size)]
        pop = [self._repair(p, requires_map) for p in pop]

        for gen in range(self.generations):
            # 评估当前种群
            objs = np.array([self._evaluate(p, priority_vec, cost_vec, d_vec, budget) for p in pop])
            fronts = self._non_dominated_sort(objs)
            cd = self._crowding_distance(objs, fronts)

            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament_selection(pop, fronts, cd)
                p2 = self._tournament_selection(pop, fronts, cd)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                c1 = self._repair(c1, requires_map)
                c2 = self._repair(c2, requires_map)
                offspring.extend([c1, c2])

            offspring = offspring[: self.pop_size]

            # 合并并环境选择
            combined = pop + offspring
            combined_objs = np.array(
                [self._evaluate(p, priority_vec, cost_vec, d_vec, budget) for p in combined]
            )
            combined_fronts = self._non_dominated_sort(combined_objs)
            combined_cd = self._crowding_distance(combined_objs, combined_fronts)
            pop = self._environmental_selection(combined, combined_fronts, combined_cd)

        # 返回最终帕累托前沿中价值最高的解
        final_objs = np.array([self._evaluate(p, priority_vec, cost_vec, d_vec, budget) for p in pop])
        final_fronts = self._non_dominated_sort(final_objs)

        # 选择第一前沿中综合得分最好的解 (f1 最小即价值最大, f2 最小即成本最低, f3 最小即 d-value 最大)
        front0 = [i for i, f in enumerate(final_fronts) if f == 0]
        best_idx = min(front0, key=lambda i: final_objs[i, 0] + final_objs[i, 2])
        return pop[best_idx]

    def _evaluate(
        self,
        x: np.ndarray,
        priority_vec: np.ndarray,
        cost_vec: np.ndarray,
        d_vec: np.ndarray,
        budget: float,
    ) -> np.ndarray:
        f1 = -np.sum(priority_vec * x)  # 最大化价值 -> 最小化负值
        f2 = np.sum(cost_vec * x)        # 最小化成本
        f3 = -np.sum(d_vec * x)          # 最大化 D-value -> 最小化负值

        # 预算惩罚 (硬约束通过 repair 尽量满足, 但仍有软惩罚)
        cost = np.sum(cost_vec * x)
        penalty = max(0.0, cost - budget) * 10.0
        f2 += penalty

        return np.array([f1, f2, f3])

    def _repair(self, x: np.ndarray, requires_map: Dict[int, Set[int]]) -> np.ndarray:
        """修复依赖约束: 若选 i, 则必须选所有 i requires 的项"""
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

    def _tournament_selection(
        self,
        pop: List[np.ndarray],
        fronts: np.ndarray,
        cd: np.ndarray,
        tournament_size: int = 2,
    ) -> np.ndarray:
        best = random.choice(range(len(pop)))
        for _ in range(tournament_size - 1):
            contender = random.choice(range(len(pop)))
            if fronts[contender] < fronts[best]:
                best = contender
            elif fronts[contender] == fronts[best] and cd[contender] > cd[best]:
                best = contender
        return deepcopy(pop[best])

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.crossover_prob:
            return deepcopy(p1), deepcopy(p2)
        point = random.randint(1, len(p1) - 1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for i in range(len(x)):
            if random.random() < self.mutation_prob:
                x[i] = not x[i]
        return x

    def _non_dominated_sort(self, objs: np.ndarray) -> np.ndarray:
        n = len(objs)
        fronts = np.full(n, -1, dtype=int)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions: List[Set[int]] = [set() for _ in range(n)]
        front1 = []

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(objs[p], objs[q]):
                    dominated_solutions[p].add(q)
                elif self._dominates(objs[q], objs[p]):
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

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, objs: np.ndarray, fronts: np.ndarray) -> np.ndarray:
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
                f_max = front_objs[sorted_idx[-1], m]
                f_min = front_objs[sorted_idx[0], m]
                denom = f_max - f_min
                if denom > 1e-9:
                    for k in range(1, len(indices) - 1):
                        distances[sorted_idx[k]] += (
                            front_objs[sorted_idx[k + 1], m] - front_objs[sorted_idx[k - 1], m]
                        ) / denom

            cd[indices] = distances

        return cd

    def _environmental_selection(
        self,
        combined: List[np.ndarray],
        fronts: np.ndarray,
        cd: np.ndarray,
    ) -> List[np.ndarray]:
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
        return selected[: self.pop_size]


# ==================== Momcozy 业务场景示例 ====================

def generate_momcozy_feedback() -> List[str]:
    """生成模拟的 Momcozy 吸奶器用户反馈 (跨市场)"""
    return [
        # 静音主题 -> Silent Motor Upgrade
        "The motor is too noisy, I can't use it at night without waking the baby",
        "Must be quiet, silence is the basic requirement for a breast pump motor upgrade",
        "Noise reduction motor is essential for nighttime pumping",
        "I hope the motor noise level can be improved significantly",
        # 便携主题 -> Travel Carrying Case
        "Very portable pump, easy to carry to work in a travel case",
        "Wish it came with a lightweight carrying case for travel",
        "Travel carrying case should be included as standard for on-the-go moms",
        # APP/智能主题 -> Smart App Control + Bluetooth Connectivity
        "Smart app control would be a pleasant surprise for tracking sessions",
        "Unexpectedly useful to have mobile app for remote control and session tracking",
        "The smart app needs bluetooth connectivity to work properly for app pairing",
        "Data sync through built-in bluetooth module is convenient for the app",
        # 充电主题 -> Fast Charging + USB-C Port
        "Fast charging support is highly desired, currently takes too long to charge",
        "USB-C port would make charging much easier with universal charging",
        "Type-C universal charging port is a must have for quick charge",
        # 模式/记忆主题 -> Memory Mode + Multi-Level Adjustment
        "Memory mode saves time, no need to adjust suction settings every time",
        "Multi-level suction adjustment is necessary before memory mode makes sense",
        "More adjustable suction strength levels please for better control",
        # 其他 -> LED Night Light + Color Shell Options
        "Color shell options are nice but not important for functionality",
        "I don't really care about the LED night light for low-light use",
    ]


def generate_momcozy_requirements() -> List[CandidateRequirement]:
    """生成 Momcozy 季度候选功能需求"""
    return [
        CandidateRequirement(
            req_id="R01",
            name="Silent Motor Upgrade",
            description="Upgrade to a quieter motor for nighttime pumping",
            estimated_cost=80.0,
        ),
        CandidateRequirement(
            req_id="R02",
            name="Travel Carrying Case",
            description="Lightweight portable case for on-the-go moms",
            estimated_cost=25.0,
        ),
        CandidateRequirement(
            req_id="R03",
            name="Smart App Control",
            description="Mobile app for remote control and session tracking",
            estimated_cost=120.0,
        ),
        CandidateRequirement(
            req_id="R04",
            name="Bluetooth Connectivity",
            description="Built-in bluetooth module for app pairing",
            estimated_cost=30.0,
        ),
        CandidateRequirement(
            req_id="R05",
            name="Fast Charging",
            description="Support quick charge technology",
            estimated_cost=40.0,
        ),
        CandidateRequirement(
            req_id="R06",
            name="USB-C Port",
            description="Replace proprietary charger with USB-C",
            estimated_cost=15.0,
        ),
        CandidateRequirement(
            req_id="R07",
            name="Memory Mode",
            description="Remember last used suction settings",
            estimated_cost=25.0,
        ),
        CandidateRequirement(
            req_id="R08",
            name="Multi-Level Adjustment",
            description="Finer suction strength levels",
            estimated_cost=20.0,
        ),
        CandidateRequirement(
            req_id="R09",
            name="LED Night Light",
            description="Soft LED light for low-light use",
            estimated_cost=10.0,
        ),
        CandidateRequirement(
            req_id="R10",
            name="Color Shell Options",
            description="Multiple color choices for outer shell",
            estimated_cost=10.0,
        ),
    ]


def demo():
    """iReFeed 完整演示"""
    print("=" * 70)
    print("iReFeed: Interconnected Requirements Prioritization")
    print("Momcozy Cross-Market Product Roadmap Demo")
    print("=" * 70)

    feedback_texts = generate_momcozy_feedback()
    requirements = generate_momcozy_requirements()

    irefeed = iReFeedPrioritizer(n_topics=4, pop_size=60, generations=50)

    # 1. 主题建模
    print("\n【Step 1】Topic Modeling (LDA)...")
    clusters = irefeed.fit_topics(feedback_texts)
    for c in clusters:
        print(f"  Cluster {c.cluster_id}: {len(c.feedbacks)} feedbacks, keywords={c.top_keywords[:5]}")

    # 2. 计算簇级优先级 (注入模拟情感/意图)
    print("\n【Step 2】Cluster-level Priority Scoring...")
    # 给反馈注入模拟的情感/意图值
    sentiment_map = {
        "noisy": (0.8, 0.0, 0.2),
        "quiet": (0.0, 0.9, 0.1),
        "portable": (0.0, 0.8, 0.2),
        "app": (0.0, 0.9, 0.1),
        "bluetooth": (0.0, 0.7, 0.1),
        "charging": (0.3, 0.4, 0.3),
        "memory": (0.0, 0.8, 0.1),
        "adjustment": (0.2, 0.5, 0.3),
        "color": (0.0, 0.3, 0.0),
        "light": (0.0, 0.2, 0.0),
    }
    for cluster in irefeed.clusters:
        for fb in cluster.feedbacks:
            for keyword, (neg, pos, inte) in sentiment_map.items():
                if keyword in fb.text.lower():
                    fb.sentiment_neg = neg
                    fb.sentiment_pos = pos
                    fb.intention = inte
                    break

    priorities = irefeed.compute_priorities(requirements, feedback_texts)
    for req in requirements:
        print(f"  {req.name}: P_r = {priorities[req.req_id]:.3f}")

    # 3. 依赖发现
    print("\n【Step 3】Dependency Discovery (LLM-simulated)...")
    deps = irefeed.discover_dependencies(requirements)
    for req_id, dep_set in deps.items():
        if dep_set:
            req_name = next(r.name for r in requirements if r.req_id == req_id)
            dep_names = [next(r.name for r in requirements if r.req_id == d) for d in dep_set]
            print(f"  {req_name} requires: {dep_names}")

    # 4. D-value
    print("\n【Step 4】D-value Computation...")
    d_values = irefeed.compute_d_values(requirements)
    for req in requirements:
        print(f"  {req.name}: D-value = {d_values[req.req_id]:.3f}")

    # 5. NSGA-II 优化
    print("\n【Step 5】NSGA-II Multi-Objective Optimization...")
    budget = 220.0
    selected, selection_vec = irefeed.optimize_selection(requirements, priorities, budget)
    total_cost = sum(r.estimated_cost for r in selected)
    total_value = sum(priorities[r.req_id] for r in selected)
    total_d = sum(d_values[r.req_id] for r in selected)

    print(f"  Budget: {budget}, Selected cost: {total_cost:.1f}")
    print(f"  Total stakeholder value: {total_value:.3f}")
    print(f"  Total dependency value: {total_d:.3f}")
    print("  Selected requirements:")
    for r in selected:
        print(f"    - {r.name} (cost={r.estimated_cost})")

    # 6. 路线图
    print("\n" + "=" * 70)
    print("Recommended Product Roadmap")
    print("=" * 70)
    print("Q1 Release:")
    for r in selected[:4]:
        print(f"  [Q1] {r.name}")
    if len(selected) > 4:
        print("Q2 Release:")
        for r in selected[4:]:
            print(f"  [Q2] {r.name}")

    print("\n✓ Cluster-level feedback association (vs. single-requirement)")
    print("✓ Interconnectedness-aware prioritization")
    print("✓ Dependency value integrated into NSGA-II optimization")
    print("✓ Data-driven quarterly roadmap for Momcozy product team")


if __name__ == "__main__":
    demo()
