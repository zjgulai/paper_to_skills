"""
知识图谱驱动的 Skills Management 系统
用于母婴出海数据科学团队的技能管理和推荐

功能：
1. Skills Knowledge Graph 构建和管理
2. 技能依赖关系查询
3. 学习路径规划
4. 知识缺口分析
5. 业务问题技能推荐

Author: paper2skills
Date: 2026-04-06
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class SkillsKnowledgeGraph:
    """
    技能知识图谱核心类

    管理 Skill 节点、关系，并提供图查询和推理能力
    """

    # 预定义的关系类型
    RELATION_TYPES = {
        'requires': '前置依赖',
        'extends': '延伸拓展',
        'combines_with': '可组合',
        'applies_to': '应用于',
        'produces': '产出',
        'measures': '衡量指标',
        'belongs_to': '属于领域'
    }

    def __init__(self):
        """初始化知识图谱"""
        self.graph = nx.DiGraph()  # 有向图
        self.skill_embeddings = {}  # 节点嵌入缓存
        self.domains = set()  # 领域集合

    def add_skill(self, skill_id: str, name: str, domain: str,
                  difficulty: int, business_value: int,
                  metadata: Dict = None):
        """
        添加 Skill 节点

        Args:
            skill_id: 技能唯一标识（如 'skill-uplift-modeling'）
            name: 技能名称
            domain: 所属领域（因果推断/A_B实验/时间序列/推荐系统/增长模型/供应链/NLP）
            difficulty: 实施难度（1-5星）
            business_value: 业务价值（1-5星）
            metadata: 其他元数据（描述、ROI等）
        """
        self.graph.add_node(
            skill_id,
            name=name,
            node_type='skill',
            domain=domain,
            difficulty=difficulty,
            business_value=business_value,
            **(metadata or {})
        )
        self.domains.add(domain)

    def add_concept(self, concept_id: str, name: str, concept_type: str):
        """
        添加概念节点

        Args:
            concept_id: 概念唯一标识
            name: 概念名称
            concept_type: 概念类型（algorithm/metric/scenario/domain）
        """
        self.graph.add_node(
            concept_id,
            name=name,
            node_type='concept',
            concept_type=concept_type
        )

    def add_relation(self, source: str, target: str, relation: str,
                     weight: float = 1.0):
        """
        添加关系边

        Args:
            source: 源节点ID
            target: 目标节点ID
            relation: 关系类型（requires/extends/combines_with/applies_to等）
            weight: 关系权重（0-1）
        """
        if relation not in self.RELATION_TYPES:
            raise ValueError(f"Unknown relation: {relation}")

        self.graph.add_edge(
            source, target,
            relation=relation,
            relation_name=self.RELATION_TYPES[relation],
            weight=weight
        )

    def get_prerequisites(self, skill_id: str) -> List[Tuple[str, float]]:
        """
        获取技能的前置依赖

        Returns:
            [(skill_id, weight), ...]
        """
        prerequisites = []
        for u, v, data in self.graph.edges(data=True):
            if v == skill_id and data['relation'] == 'requires':
                prerequisites.append((u, data['weight']))
        return prerequisites

    def get_extensions(self, skill_id: str) -> List[Tuple[str, float]]:
        """
        获取技能的延伸方向
        """
        extensions = []
        for u, v, data in self.graph.edges(data=True):
            if u == skill_id and data['relation'] == 'extends':
                extensions.append((v, data['weight']))
        return extensions

    def get_combinable_skills(self, skill_id: str) -> List[Tuple[str, float]]:
        """
        获取可组合的技能
        """
        combinables = []
        for u, v, data in self.graph.edges(data=True):
            if (u == skill_id or v == skill_id) and data['relation'] == 'combines_with':
                other = v if u == skill_id else u
                combinables.append((other, data['weight']))
        return combinables

    def compute_learning_path(self, start_skill: str, target_skill: str) -> List[str]:
        """
        计算从起始技能到目标技能的最短学习路径

        使用 Dijkstra 算法找到最短路径（考虑前置依赖关系）
        """
        try:
            # 构建前置依赖子图
            prereq_graph = nx.DiGraph()
            for u, v, data in self.graph.edges(data=True):
                if data['relation'] == 'requires':
                    prereq_graph.add_edge(u, v, weight=data['weight'])

            # 找到最短路径（目标技能的前置技能）
            path = nx.shortest_path(
                prereq_graph,
                source=start_skill,
                target=target_skill,
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    def find_skill_gaps(self) -> List[Dict]:
        """
        发现知识缺口

        Returns:
            [{'type': 'missing_prerequisite', 'skill': 'X', 'missing': 'Y'}, ...]
        """
        gaps = []

        # 1. 前置缺口检测
        for skill_id in self.graph.nodes():
            prereqs = self.get_prerequisites(skill_id)
            for prereq_id, _ in prereqs:
                if prereq_id not in self.graph:
                    gaps.append({
                        'type': 'missing_prerequisite',
                        'skill': skill_id,
                        'missing': prereq_id,
                        'priority': 'high'
                    })

        # 2. 延伸缺口检测（高价值但无延伸的技能）
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'skill':
                business_value = data.get('business_value', 0)
                extensions = self.get_extensions(node_id)
                if business_value >= 4 and len(extensions) == 0:
                    gaps.append({
                        'type': 'missing_extension',
                        'skill': node_id,
                        'skill_name': data['name'],
                        'priority': 'medium'
                    })

        # 3. 桥梁缺口（跨领域连接不足）
        domain_skills = defaultdict(list)
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'skill':
                domain_skills[data.get('domain')].append(node_id)

        # 检测领域间是否有连接
        domains = list(domain_skills.keys())
        for i, dom_a in enumerate(domains):
            for dom_b in domains[i+1:]:
                has_bridge = False
                for skill_a in domain_skills[dom_a]:
                    combinables = self.get_combinable_skills(skill_a)
                    for skill_b, _ in combinables:
                        if self.graph.nodes[skill_b].get('domain') == dom_b:
                            has_bridge = True
                            break
                    if has_bridge:
                        break
                if not has_bridge:
                    gaps.append({
                        'type': 'missing_bridge',
                        'between': [dom_a, dom_b],
                        'priority': 'low'
                    })

        return gaps

    def recommend_skills_for_problem(self, problem_keywords: List[str],
                                     current_skills: List[str] = None) -> List[Dict]:
        """
        针对业务问题推荐技能组合

        Args:
            problem_keywords: 问题关键词列表（如 ['获客成本', '复购率']）
            current_skills: 团队已掌握的技能列表

        Returns:
            [{'skill_id': 'X', 'score': 0.85, 'reason': '...'}, ...]
        """
        recommendations = []
        current_skills = current_skills or []

        # 1. 关键词匹配（简化的 TF-IDF 思路）
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') != 'skill':
                continue

            score = 0
            reasons = []

            # 匹配技能名称
            skill_name = data.get('name', '').lower()
            for kw in problem_keywords:
                if kw.lower() in skill_name:
                    score += 0.3
                    reasons.append(f"关键词'{kw}'匹配技能名称")

            # 匹配应用场景
            for u, v, edge_data in self.graph.edges(data=True):
                if edge_data['relation'] == 'applies_to' and u == node_id:
                    scenario = self.graph.nodes[v].get('name', '').lower()
                    for kw in problem_keywords:
                        if kw.lower() in scenario:
                            score += 0.4
                            reasons.append(f"应用于场景'{scenario}'")

            # 业务价值加权
            business_value = data.get('business_value', 3)
            score += (business_value / 5) * 0.2

            # 前置技能检查（如果已有前置技能，推荐分数更高）
            prereqs = self.get_prerequisites(node_id)
            if prereqs:
                prereq_ids = [p[0] for p in prereqs]
                matched_prereqs = set(prereq_ids) & set(current_skills)
                if matched_prereqs:
                    score += len(matched_prereqs) / len(prereqs) * 0.1
                    reasons.append(f"已有前置技能: {matched_prereqs}")

            if score > 0.3:
                recommendations.append({
                    'skill_id': node_id,
                    'skill_name': data['name'],
                    'score': score,
                    'reasons': reasons,
                    'difficulty': data.get('difficulty', 3),
                    'business_value': data.get('business_value', 3)
                })

        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:10]

    def visualize_subgraph(self, center_skill: str, depth: int = 2) -> Dict:
        """
        生成可视化子图数据

        Returns:
            {'nodes': [...], 'edges': [...]}
        """
        # BFS 遍历获取子图
        visited = {center_skill}
        current_level = {center_skill}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in visited:
                        next_level.add(predecessor)
                        visited.add(predecessor)
            current_level = next_level

        # 构建可视化数据
        nodes = []
        for node_id in visited:
            data = self.graph.nodes[node_id]
            nodes.append({
                'id': node_id,
                'name': data.get('name', node_id),
                'type': data.get('node_type', 'unknown'),
                'domain': data.get('domain', ''),
                'difficulty': data.get('difficulty', 0),
                'business_value': data.get('business_value', 0)
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in visited and v in visited:
                edges.append({
                    'source': u,
                    'target': v,
                    'relation': data['relation'],
                    'relation_name': data['relation_name'],
                    'weight': data['weight']
                })

        return {'nodes': nodes, 'edges': edges}

    def compute_centrality_metrics(self) -> Dict:
        """
        计算图谱的中心性指标

        Returns:
            {
                'degree_centrality': {...},
                'betweenness_centrality': {...},
                'pagerank': {...}
            }
        """
        return {
            'degree_centrality': nx.degree_centrality(self.graph),
            'betweenness_centrality': nx.betweenness_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph)
        }

    def find_skill_clusters(self) -> Dict:
        """
        发现技能聚类（社区检测）

        Returns:
            {cluster_id: [skill_ids], ...}
        """
        # 将图转换为无向图进行社区检测
        undirected = self.graph.to_undirected()
        communities = nx.community.greedy_modularity_communities(undirected)

        clusters = {}
        for i, community in enumerate(communities):
            clusters[f'cluster_{i}'] = list(community)

        return clusters

    def export_to_json(self, filepath: str):
        """导出图谱到 JSON 文件"""
        data = {
            'nodes': [
                {'id': n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ],
            'edges': [
                {'source': u, 'target': v, **d}
                for u, v, d in self.graph.edges(data=True)
            ]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'SkillsKnowledgeGraph':
        """从 JSON 文件加载图谱"""
        kg = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for node in data['nodes']:
            node_id = node.pop('id')
            kg.graph.add_node(node_id, **node)

        for edge in data['edges']:
            kg.graph.add_edge(
                edge['source'], edge['target'],
                relation=edge['relation'],
                relation_name=edge['relation_name'],
                weight=edge['weight']
            )

        return kg


def create_maternal_baby_skills_kg() -> SkillsKnowledgeGraph:
    """
    创建母婴出海数据科学团队的知识图谱示例
    """
    kg = SkillsKnowledgeGraph()

    # 添加领域节点
    domains = {
        'causal_inference': '因果推断',
        'ab_testing': 'A/B实验',
        'time_series': '时间序列',
        'supply_chain': '供应链',
        'recommendation': '推荐系统',
        'growth_model': '增长模型',
        'nlp_voc': 'NLP-VOC'
    }
    for dom_id, dom_name in domains.items():
        kg.add_concept(dom_id, dom_name, 'domain')

    # 添加技能节点（基于现有 Skill 卡片）
    skills_data = [
        # 因果推断领域
        ('skill-uplift', 'Uplift Modeling', 'causal_inference', 2, 5),
        ('skill-causal-forest', 'Causal Forest', 'causal_inference', 3, 5),
        ('skill-doubly-robust', 'Doubly Robust Estimation', 'causal_inference', 4, 4),
        ('skill-propensity', 'Propensity Score Matching', 'causal_inference', 2, 4),

        # A/B实验领域
        ('skill-mab', 'Multi-Armed Bandit', 'ab_testing', 2, 4),
        ('skill-thompson', 'Thompson Sampling', 'ab_testing', 2, 4),

        # 时间序列领域
        ('skill-tft', 'Temporal Fusion Transformer', 'time_series', 4, 5),
        ('skill-prophet', 'Prophet Forecasting', 'time_series', 2, 3),

        # 推荐系统领域
        ('skill-matrix-factor', 'Matrix Factorization', 'recommendation', 2, 4),
        ('skill-deep-rec', 'Deep Learning Recommendation', 'recommendation', 3, 5),
        ('skill-cold-start', 'Cold Start Product Recommendation', 'recommendation', 3, 4),

        # 增长模型领域
        ('skill-ziln-ltv', 'LTV Prediction (ZILN)', 'growth_model', 3, 5),
        ('skill-churn', 'Customer Churn Prediction', 'growth_model', 2, 4),
        ('skill-opportunity', 'New Product Opportunity Mining', 'growth_model', 3, 4),

        # 供应链领域
        ('skill-multi-echelon', 'Multi-Echelon Inventory', 'supply_chain', 3, 4),
        ('skill-drl-inventory', 'DRL for Inventory', 'supply_chain', 4, 4),

        # NLP-VOC 领域
        ('skill-absa', 'Aspect-Based Sentiment Analysis', 'nlp_voc', 3, 4),
        ('skill-sentiment-bert', 'BERT Sentiment Analysis', 'nlp_voc', 3, 4),
    ]

    for skill_id, name, domain, difficulty, bv in skills_data:
        kg.add_skill(skill_id, name, domain, difficulty, bv)

    # 添加技能关系
    relations = [
        # 前置依赖
        ('skill-uplift', 'requires', 'skill-propensity', 1.0),
        ('skill-causal-forest', 'requires', 'skill-uplift', 0.9),
        ('skill-doubly-robust', 'requires', 'skill-propensity', 0.9),
        ('skill-thompson', 'requires', 'skill-mab', 1.0),
        ('skill-tft', 'requires', 'skill-prophet', 0.6),
        ('skill-deep-rec', 'requires', 'skill-matrix-factor', 0.8),
        ('skill-ziln-ltv', 'requires', 'skill-churn', 0.6),
        ('skill-drl-inventory', 'requires', 'skill-multi-echelon', 0.8),
        ('skill-absa', 'requires', 'skill-sentiment-bert', 0.7),

        # 延伸拓展
        ('skill-uplift', 'extends', 'skill-causal-forest', 0.9),
        ('skill-uplift', 'extends', 'skill-doubly-robust', 0.8),
        ('skill-mab', 'extends', 'skill-thompson', 1.0),
        ('skill-prophet', 'extends', 'skill-tft', 0.7),
        ('skill-matrix-factor', 'extends', 'skill-deep-rec', 0.8),
        ('skill-churn', 'extends', 'skill-ziln-ltv', 0.8),
        ('skill-multi-echelon', 'extends', 'skill-drl-inventory', 0.8),

        # 可组合
        ('skill-uplift', 'combines_with', 'skill-ziln-ltv', 0.9),
        ('skill-tft', 'combines_with', 'skill-multi-echelon', 0.8),
        ('skill-matrix-factor', 'combines_with', 'skill-cold-start', 0.9),
        ('skill-absa', 'combines_with', 'skill-opportunity', 0.7),
        ('skill-causal-forest', 'combines_with', 'skill-deep-rec', 0.6),
    ]

    for source, rel, target, weight in relations:
        kg.add_relation(source, target, rel, weight)

    # 添加业务应用场景节点
    scenarios = [
        ('scenario-ad-attribution', '广告归因优化', 'ad_optimization'),
        ('scenario-pricing', '动态定价', 'pricing'),
        ('scenario-forecast', '销量预测', 'forecasting'),
        ('scenario-inventory', '库存优化', 'inventory'),
        ('scenario-recommend', '商品推荐', 'recommendation'),
        ('scenario-ltv', 'LTV分层运营', 'ltv_management'),
        ('scenario-churn', '流失预警', 'churn_prevention'),
        ('scenario-voc', 'VOC舆情分析', 'voc_analysis'),
    ]

    for scen_id, scen_name, scen_type in scenarios:
        kg.add_concept(scen_id, scen_name, 'scenario')

    # 技能-场景应用关系
    applies_to = [
        ('skill-uplift', 'scenario-ad-attribution', 1.0),
        ('skill-causal-forest', 'scenario-ad-attribution', 0.9),
        ('skill-mab', 'scenario-pricing', 0.9),
        ('skill-thompson', 'scenario-pricing', 0.9),
        ('skill-tft', 'scenario-forecast', 1.0),
        ('skill-prophet', 'scenario-forecast', 0.8),
        ('skill-multi-echelon', 'scenario-inventory', 1.0),
        ('skill-drl-inventory', 'scenario-inventory', 0.9),
        ('skill-matrix-factor', 'scenario-recommend', 1.0),
        ('skill-deep-rec', 'scenario-recommend', 1.0),
        ('skill-cold-start', 'scenario-recommend', 0.8),
        ('skill-ziln-ltv', 'scenario-ltv', 1.0),
        ('skill-churn', 'scenario-churn', 1.0),
        ('skill-absa', 'scenario-voc', 1.0),
        ('skill-opportunity', 'scenario-voc', 0.6),
    ]

    for skill, scenario, weight in applies_to:
        kg.add_relation(skill, scenario, 'applies_to', weight)

    return kg


def main():
    """主函数：演示知识图谱 Skills Management 系统"""
    print("=" * 80)
    print("母婴出海 - 知识图谱驱动的 Skills Management 系统")
    print("=" * 80)

    # 1. 创建知识图谱
    print("\n[1] 创建 Skills Knowledge Graph...")
    kg = create_maternal_baby_skills_kg()
    skill_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'skill']
    concept_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'concept']
    print(f"   技能节点数: {len(skill_nodes)}")
    print(f"   概念节点数: {len(concept_nodes)}")
    print(f"   关系边数: {kg.graph.number_of_edges()}")

    # 2. 查询技能依赖关系
    print("\n[2] 技能依赖关系查询示例...")
    skill_id = 'skill-causal-forest'
    prereqs = kg.get_prerequisites(skill_id)
    print(f"   {kg.graph.nodes[skill_id]['name']} 的前置技能:")
    for prereq, weight in prereqs:
        print(f"     - {kg.graph.nodes[prereq]['name']} (权重: {weight})")

    extensions = kg.get_extensions(skill_id)
    print(f"   延伸技能:")
    for ext, weight in extensions:
        print(f"     - {kg.graph.nodes[ext]['name']} (权重: {weight})")

    # 3. 计算学习路径
    print("\n[3] 学习路径规划...")
    path = kg.compute_learning_path('skill-propensity', 'skill-causal-forest')
    if path:
        print(f"   从 '倾向评分' 到 '因果森林' 的学习路径:")
        for i, node_id in enumerate(path):
            name = kg.graph.nodes[node_id]['name']
            print(f"     {i+1}. {name}")

    # 4. 发现知识缺口
    print("\n[4] 知识缺口分析...")
    gaps = kg.find_skill_gaps()
    high_priority_gaps = [g for g in gaps if g['priority'] == 'high']
    if high_priority_gaps:
        print(f"   发现 {len(high_priority_gaps)} 个高优先级缺口:")
        for gap in high_priority_gaps[:3]:
            print(f"     - {gap['type']}: {gap.get('skill', '')}")
    else:
        print("   暂无高优先级缺口")

    medium_gaps = [g for g in gaps if g['priority'] == 'medium']
    if medium_gaps:
        print(f"   发现 {len(medium_gaps)} 个中优先级缺口（高价值技能缺少延伸）")

    # 5. 业务问题技能推荐
    print("\n[5] 针对业务问题的技能推荐...")
    problem = ['获客成本', '广告', '归因']
    print(f"   业务问题: {problem}")
    recommendations = kg.recommend_skills_for_problem(problem, current_skills=['skill-propensity'])
    print(f"   推荐技能:")
    for rec in recommendations[:5]:
        print(f"     - {rec['skill_name']} (匹配度: {rec['score']:.2f})")
        if rec['reasons']:
            print(f"       原因: {rec['reasons'][0]}")

    # 6. 技能组合推荐
    print("\n[6] 技能组合推荐...")
    skill = 'skill-uplift'
    combinables = kg.get_combinable_skills(skill)
    print(f"   与 '{kg.graph.nodes[skill]['name']}' 可组合的技能:")
    for combo, weight in combinables:
        print(f"     - {kg.graph.nodes[combo]['name']} (协同度: {weight})")

    # 7. 中心性分析
    print("\n[7] 技能中心性分析...")
    centrality = kg.compute_centrality_metrics()
    pagerank = centrality['pagerank']
    top_skills = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    print("   PageRank Top 5（知识网络中的核心技能）:")
    for skill_id, score in top_skills:
        if kg.graph.nodes[skill_id].get('node_type') == 'skill':
            name = kg.graph.nodes[skill_id]['name']
            print(f"     - {name}: {score:.4f}")

    # 8. 社区检测
    print("\n[8] 技能聚类分析...")
    clusters = kg.find_skill_clusters()
    print(f"   发现 {len(clusters)} 个技能社区:")
    for cluster_id, members in clusters.items():
        skill_members = [m for m in members if kg.graph.nodes[m].get('node_type') == 'skill']
        if skill_members:
            names = [kg.graph.nodes[m]['name'] for m in skill_members[:3]]
            print(f"     - {cluster_id}: {', '.join(names)}{'...' if len(skill_members) > 3 else ''}")

    # 9. 导出图谱
    print("\n[9] 导出知识图谱...")
    output_path = '/tmp/skills_kg.json'
    kg.export_to_json(output_path)
    print(f"   已导出到: {output_path}")

    print("\n" + "=" * 80)
    print("Skills Knowledge Graph 系统演示完成!")
    print("=" * 80)

    return kg


if __name__ == '__main__':
    kg = main()
