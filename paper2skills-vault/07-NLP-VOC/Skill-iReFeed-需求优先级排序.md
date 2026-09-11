# Skill Card: iReFeed-需求优先级排序

---

## ① 算法原理

**核心思想**：将用户反馈驱动的需求优先级排序从"单需求独立评估"升级为"需求簇互联评估"。传统ReFeed将反馈与单个需求关联，忽略了需求间的相互依赖性；iReFeed通过topic modeling把用户反馈聚类为主题簇，将候选需求映射到簇中，在簇级别关联反馈并计算优先级，同时引入D-value衡量需求被依赖的价值，最终通过NSGA-II三目标优化求解Next Release Problem。

**数学直觉**：

1. **需求簇级反馈关联**
   传统ReFeed将反馈 $F$ 与单个需求 $r$ 关联：
   $$P_r = \frac{\sum_{i=1}^{|F|} [sim(r, F[i]) \times (neg_{F[i]} + pos_{F[i]} + int_{F[i]})]}{|F|}$$

   iReFeed改为将反馈与需求簇 $F_C$ 关联：
   $$P_r = \frac{\sum_{i=1}^{|F_C|} [sim(r, F_C[i]) \times (neg_{F_C[i]} + pos_{F_C[i]} + int_{F_C[i]})]}{|F_C|}$$

2. **簇内聚性增强**
   为奖励内部一致的需求簇，引入coherence factor：
   $$\alpha(F_C) = \min(1, \text{average pairwise similarity of } \forall r_i, r_j \in F_C)$$
   增强后的优先级：
   $$P_r = \frac{\sum_{i=1}^{|F_C|} [\alpha(F_C) \times sim(r, F_C[i]) \times (neg_{F_C[i]} + pos_{F_C[i]} + int_{F_C[i]})]}{|F_C|}$$

3. **依赖价值 D-value**
   利用LLM自动发现需求间的"requires"关系。对于需求 $i$，其依赖价值：
   $$D\text{-}value_i = \frac{count_i}{|\mathcal{D}|}$$
   其中 $count_i$ 是需求 $i$ 作为"requires"关系右侧（被依赖方）出现的次数，$|\mathcal{D}|$ 是发现的总关系数。被越多其他需求依赖的项，D-value越高。

4. **NSGA-II 三目标优化**
   将D-value作为第三目标集成到Next Release Problem的NSGA-II求解中：
   - Maximize: 利益相关者价值总和（$\sum P_r \cdot x_r$）
   - Minimize: 开发资源成本（$\sum cost_r \cdot x_r$）
   - Maximize: 需求依赖价值（$\sum D\text{-}value_r \cdot x_r$）
   - 约束: 总成本 $\leq$ 预算；若选 $r_i$ 且 $r_i \text{ requires } r_j$，则必须选 $r_j$

**反直觉洞察**：考虑需求间关联性的优先级排序比独立评估提升F1-score达35%，因为真实产品的功能是相互依赖的，孤立评估会导致路线图不可执行。

**关键假设**：用户反馈量足够支撑有意义的topic modeling（论文中每个应用有数万条评论）；需求可以被自然地按主题聚类；存在可用的LLM用于发现"requires"关系；需求的价值和开发成本可被量化评估。

---

## ② 母婴出海应用案例

### 场景1：Momcozy季度产品功能优先级排序

**业务问题**：Momcozy每季度从Amazon US/DE/Wayfair收集用户评论，结合内部产品规划产生30-50个candidate功能需求（如"降噪改进""APP远程控制""新配件兼容"）。传统方式靠产品委员会主观打分，导致高价值需求被遗漏、依赖关系被忽视、季度路线图频繁调整。

**数据要求**：
- 季度内跨市场用户评论（每个市场≥5000条，总数≥15000条）
- candidate功能需求清单（含描述、预估成本）
- 季度开发预算上限

**应用流程**：
1. 对用户评论做LDA topic modeling，识别15-20个主题簇
2. 将candidate功能映射到主题簇，计算簇级反馈优先级 $P_r$
3. 用LLM发现功能间依赖（如"APP升级" requires "蓝牙模块更新"）
4. 计算每个功能的D-value，识别基础底座型功能
5. 输入NSGA-II优化，在价值、成本、依赖关系约束下输出最优季度功能组合

**预期产出**：
- 各功能优先级得分 $P_r$（0-1区间）
- 依赖关系图和D-value排名
- NSGA-II帕累托前沿推荐的Q1/Q2/Q3功能清单
- 示例输出：
  ```
  Q1: 静音电机升级(P0) | Type-C接口(P0) | 蓝牙模块(P0)
  Q2: 智能APP控制(P1) | 快充支持(P1)
  Q3: 便携旅行包(P2) | 记忆模式(P2)
  ```

**业务价值**：将季度功能优先级决策从"主观委员会投票"升级为"数据驱动优化"，预计减少路线图返工50%，提升用户高反馈需求的落地率40%。

### 场景2：跨市场差异化需求整合

**业务问题**：美国市场用户高频反馈"便携性"，德国市场高频反馈"静音认证"，中国市场关注"清洗方便"。各区域团队各自为政 proposing 功能需求，总部难以判断哪些是"全球通用需求"、哪些是"区域特供"。

**数据要求**：
- 分市场的用户评论数据（US/DE/CN各≥3000条）
- 各市场提报的candidate需求清单

**应用流程**：
1. 将所有市场的用户反馈统一做topic modeling，构建跨市场共享主题空间
2. 将各区域candidate需求映射到统一topic空间
3. 识别能同时覆盖多个市场高反馈的主题簇，优先排序这些"全球通用需求"
4. 区域特定需求仅在局部季度计划中安排，避免全球SKU过度膨胀

**业务价值**：降低全球产品矩阵复杂度，将资源集中投入ROI最高的通用功能；预计减少15-20%的低价值区域特供功能开发。

---

## ③ 代码模板

代码路径：`paper2skills-code/nlp_voc/irefeed_priority_ranking/model.py`

```python
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

    def fit_topics(self, feedback_texts: List[str]) -> List[FeedbackCluster]:
        """LDA 主题建模"""
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
            clusters[dominant_topic].feedbacks.append(UserFeedback(text=feedback_texts[idx]))

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
```

---

## ④ 技能关联

- **前置技能**：
  - `Skill-Kano-需求分类与优先级` — Kano分类为iReFeed提供需求价值权重输入，基本型需求在NSGA-II中享有更高价值系数
  - `Skill-TopicImpact-观点单元画像抽取` — TopicImpact提取的功能主题和观点单元是iReFeed topic modeling的高质量输入，减少噪声反馈
- **延伸技能**：
  - `Skill-MAA-行动建议生成` — iReFeed生成的季度功能优先级和roadmap可直接输入MAA pipeline，转化为具体的产品改进行动建议
  - `Skill-AGRS-属性引导评论摘要` — AGRS季度摘要中的高频属性可作为iReFeed的candidate需求来源
- **可组合技能**：
  - 与 `Skill-Kano-需求分类与优先级` + `Skill-MAA-行动建议生成` 组合：Kano分类 → iReFeed排序 → MAA建议，形成"VOC洞察→产品决策→执行行动"完整闭环

---

## ⑤ 商业价值评估

- **ROI预估**：
  - 直接收益：将季度功能优先级决策从2周委员会讨论缩短至1天自动化分析，产品管理人力成本降低约70%；减少因忽视依赖关系导致的路线图返工，预计节约开发资源20-30%
  - 间接收益：提升用户高反馈需求的落地率，预计NPS提升5-8分；跨市场通用需求的识别可减少15-20%的低价值区域特供SKU开发
  - 综合ROI：首年投入约8万元（含NLP pipeline、LLM调用和NSGA-II优化模块），预期节省人力+降低返工+提升用户满意度带来的回报约50-70万元，**ROI约6-9倍**

- **实施难度**：⭐⭐⭐⭐☆（4/5）
  - 需要搭建完整的用户反馈topic modeling pipeline、LLM依赖关系抽取、以及NSGA-II多目标优化模块；论文已提供完整算法框架，但工程集成复杂度较高

- **优先级评分**：⭐⭐⭐⭐⭐（5/5）
  - iReFeed是VOC→产品决策链路的核心断点补齐技能，直接连接Kano分类与产品roadmap，是07-NLP-VOC领域闭环的最后一个关键节点

- **评估依据**：
  论文实验表明，考虑需求互联性的iReFeed相比独立评估的ReFeed在优先级排序F1-score上提升35%。对于产品线复杂、季度迭代密集的母婴出海业务，iReFeed能有效避免"孤立评估导致的功能孤岛"和"依赖被忽视导致的路线图延期"两大痛点。该技能与现有Kano、TopicImpact、MAA技能组合后，可形成从用户声音到产品执行的全链路数据驱动决策能力。
