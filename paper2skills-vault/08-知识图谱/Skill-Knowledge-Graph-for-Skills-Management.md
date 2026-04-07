# Skill Card: Knowledge Graph for Skills Management（知识图谱驱动的技能管理）

---

## ① 算法原理

### 核心思想
**知识图谱（Knowledge Graph, KG）** 是一种用图结构表示知识的方法，通过**实体-关系-实体**的三元组形式（如"Uplift Modeling -应用于- 广告投放"）将碎片化信息组织成可推理的知识网络。

在 Skills Graph 中引入知识图谱，可以解决以下痛点：
1. **技能孤岛**：现有 Skill 卡片之间的关系仅通过"前置/延伸/可组合"简单描述，缺乏语义化的关系定义
2. **知识断层**：难以发现跨领域的技能组合机会（如"Uplift Modeling + LTV预测"的协同效应）
3. **检索局限**：基于关键词的检索无法理解技能间的深层关联

### 数学直觉

**知识图谱形式化定义**：

$$G = (E, R, T)$$

其中：
- $E$：实体集合（如 Skill 节点、概念节点、应用场景节点）
- $R$：关系集合（如"前置_requires"、"延伸_extends"、"组合_combines_with"）
- $T$：三元组集合 $\{(h, r, t) | h, t \in E, r \in R\}$

**图嵌入表示（TransE 算法）**：

将实体和关系嵌入到同一向量空间：

$$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

目标是最小化：
$$\mathcal{L} = \sum_{(h,r,t) \in T} \sum_{(h',r,t') \in T'} \max(0, d(h+r, t) + \gamma - d(h'+r, t'))$$

其中 $d(\cdot, \cdot)$ 可以是 L1 或 L2 距离，$T'$ 是负采样三元组。

**技能相似度计算**：

基于图嵌入的余弦相似度：
$$\text{sim}(s_i, s_j) = \frac{\mathbf{s}_i \cdot \mathbf{s}_j}{||\mathbf{s}_i|| \cdot ||\mathbf{s}_j||}$$

### 关键假设
- **知识可结构化**：Skill 之间的关系可以用预定义的关系类型描述
- **图连通性**：大部分 Skill 节点应该与其他节点存在关联（避免孤立节点）
- **语义一致性**：相似技能的嵌入向量在空间中应该相近
- **可扩展性**：新 Skill 可以动态加入图谱而不需要重新构建

---

## ② 母婴出海应用案例

### 场景一：智能技能推荐系统

**业务问题**：
数据科学团队新入职一名分析师，需要快速掌握"母婴出海跨境电商"相关技能。现有 20+ 个 Skill 卡片分散在不同领域，新人不知道学习路径如何规划，也不清楚哪些技能组合能解决实际业务问题。

**数据要求**：
- 已有 Skill 卡片：20+ 个（涵盖因果推断、A/B实验、时间序列、推荐系统、增长模型、NLP等）
- 技能元数据：每个技能的领域、难度、业务价值、前置技能、延伸技能
- 业务场景库：典型的母婴出海业务问题与对应技能组合的映射
- 用户画像：团队成员的技能掌握程度、岗位职责、学习偏好

**预期产出**：
- **个性化学习路径**：根据当前技能水平推荐最优学习顺序
  ```
  基础统计 → 倾向评分 → Uplift Modeling → 因果森林
                    ↘
                      LTV预测 → 动态定价应用
  ```
- **技能组合推荐**：针对具体业务问题推荐技能组合
  - 问题"如何优化吸奶器广告投放？"→ 推荐 Uplift Modeling + 智能归因
  - 问题"如何预测新品销量？"→ 推荐 TFT + 多层级库存优化
- **知识缺口诊断**：识别团队技能短板并推荐补强方向

**业务价值**：
- 新人上手时间从 3 个月缩短至 1 个月
- 技能检索效率提升 60%+
- 跨领域项目（如"因果推断+推荐系统"）启动速度提升 40%

---

### 场景二：业务问题到技能方案的智能匹配

**业务问题**：
运营团队提出业务问题"我们如何降低吸奶器新客的获客成本，同时提升复购率？"，数据团队需要快速判断：
1. 这个问题涉及哪些技术领域？
2. 现有技能能否解决？还需要补充什么？
3. 不同技能组合的预期效果和投资回报率如何？

**数据要求**：
- 业务问题描述（自然语言）
- Skills Graph 知识图谱（实体：Skill、应用场景、业务指标）
- 历史项目数据：过往技能应用的业务效果记录
- 业务指标库：CAC、LTV、复购率、ROI 等关键指标

**预期产出**：
- **领域映射**：自动识别问题涉及的关键技术领域
  - "降低获客成本"→ 因果推断（广告归因）、A/B实验
  - "提升复购率"→ 推荐系统、增长模型（LTV预测）
- **技能匹配度评分**：
  | 技能组合 | 匹配度 | 预期ROI | 实施难度 |
  |---------|-------|---------|---------|
  | Uplift Modeling + LTV预测 | 95% | 15x | ⭐⭐⭐ |
  | 矩阵分解 + 冷启动推荐 | 80% | 12x | ⭐⭐ |
- **知识缺口提醒**：
  - "当前缺少'动态定价'技能，建议补充学习"
  - "推荐搜索论文: dynamic pricing breast pump e-commerce"

**业务价值**：
- 业务需求到技术方案匹配时间从 1 周缩短至 1 天
- 避免重复造轮子（先查知识图谱是否已有解决方案）
- 技术方案的业务相关性评估更准确

---

## ③ 代码模板

```python
"""
知识图谱驱动的 Skills Management 系统
用于母婴出海数据科学团队的技能管理和推荐
"""

import numpy as np
import pandas as pd
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
            node_type = node.get('node_type')
            if node_type == 'skill':
                kg.graph.add_node(node_id, **node)
            else:
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
    print(f"   技能节点数: {len([n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'skill'])}")
    print(f"   概念节点数: {len([n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'concept'])}")
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
    
    # 5. 业务问题技能推荐
    print("\n[5] 针对业务问题的技能推荐...")
    problem = ['获客成本', '广告', '归因']
    print(f"   业务问题: {problem}")
    recommendations = kg.recommend_skills_for_problem(problem, current_skills=['skill-propensity'])
    print(f"   推荐技能:")
    for rec in recommendations[:5]:
        print(f"     - {rec['skill_name']} (匹配度: {rec['score']:.2f})")
        print(f"       原因: {rec['reasons'][0] if rec['reasons'] else 'N/A'}")
    
    # 6. 技能组合推荐
    print("\n[6] 技能组合推荐...")
    skill = 'skill-uplift'
    combinables = kg.get_combinable_skills(skill)
    print(f"   与 '{kg.graph.nodes[skill]['name']}' 可组合的技能:")
    for combo, weight in combinables:
        print(f"     - {kg.graph.nodes[combo]['name']} (协同度: {weight})")
    
    # 7. 导出图谱
    print("\n[7] 导出知识图谱...")
    kg.export_to_json('/tmp/skills_kg.json')
    print("   已导出到: /tmp/skills_kg.json")
    
    print("\n" + "=" * 80)
    print("Skills Knowledge Graph 系统演示完成!")
    print("=" * 80)
    
    return kg


---

## ④ 技能关联

### 前置技能
- **图论基础**：理解图、节点、边、路径等基本概念
- **知识图谱基础**：了解 RDF、三元组、SPARQL 等知识图谱基本概念
- **图嵌入算法**：理解 Node2Vec、TransE 等图嵌入方法原理
- **Python网络分析**：熟悉 NetworkX 或类似图分析库

### 延伸技能
- **图神经网络 (GNN)**：使用 GCN、GAT 等深度学习方法进行图推理
- **GraphRAG**：基于知识图谱的检索增强生成系统
- **动态知识图谱**：支持图谱的动态更新和演化
- **多模态知识图谱**：整合文本、图像等多种模态信息

### 可组合技能
- **Skills Graph 分析**：结合知识图谱进行系统性技能缺口分析
- **个性化学习推荐**：基于知识图谱的个性化学习路径推荐
- **论文选题推荐**：利用知识图谱发现研究空白和选题方向
- **智能问答系统**：基于知识图谱的业务问题智能解答

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 新人培训加速 | 上手时间从3个月→1个月，节省人力成本50% | 开发2-3周 | 20-30x |
| 技能检索效率 | 技能查找时间减少60%，项目启动加速40% | 开发1-2周 | 15-20x |
| 跨领域创新 | 发现技能组合机会，提升方案创新性30% | 持续维护 | 10-15x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要整理现有 Skill 卡片的关系数据
- 技术门槛：中等，主要基于 NetworkX 和图算法
- 工程复杂度：中，需要设计图谱Schema和关系定义
- 维护成本：中，新技能加入时需要更新图谱

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- **战略价值高**：是 paper2skills 体系的基础设施，支撑长期发展
- **复利效应明显**：投入一次，持续受益，随技能库增长价值递增
- **团队赋能显著**：显著提升团队知识管理效率和新人培养速度
- **可扩展性强**：可延伸至智能推荐、问答系统等多个应用

### 评估依据
1. **知识管理是团队效能的瓶颈**：现有20+技能分散管理，检索和学习成本高
2. **技术成熟度高**：NetworkX、Neo4j等工具成熟，实现风险低
3. **与现有体系天然契合**：Skills Graph已有"前置/延伸/可组合"关系定义
4. **长期战略价值**：是构建AI驱动学习系统的基础架构

---

## 参考论文

1. **GAAMA: Graph Augmented Associative Memory for Agents** (2026)
   - arXiv:2603.27910v1
   - 核心贡献：将知识图谱用于Agent的长期记忆管理

2. **GraphWalker: Agentic Knowledge Graph Question Answering via Synthetic Trajectory Curriculum** (2026)
   - arXiv:2603.28533v1
   - 核心贡献：基于知识图谱的智能问答和推理

3. **TransE: Translating Embeddings for Modeling Multi-relational Data** (2013)
   - NIPS 2013
   - 核心贡献：知识图谱嵌入的经典算法

4. **ByteRover: Agent-Native Memory Through LLM-Curated Hierarchical Context** (2026)
   - arXiv:2604.01599v1
   - 核心贡献：LLM+知识图谱的层次化记忆架构

---

## 开源资源

- **NetworkX**: https://networkx.org/ - Python图分析库
- **Neo4j**: https://neo4j.com/ - 图数据库
- **DGL**: https://www.dgl.ai/ - 深度图学习库
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/ - 图神经网络库

---

## 后续演进方向

### Round 1: 静态知识图谱（当前）
- 基于现有 Skill 卡片构建静态图谱
- 支持基本的查询和路径计算

### Round 2: 智能推荐增强
- 引入图神经网络进行技能嵌入学习
- 基于业务问题自动推荐技能组合
- 集成 LLM 进行自然语言问答

### Round 3: 动态演化系统
- 技能图谱随新论文萃取自动更新
- 追踪团队成员技能掌握进度
- 基于业务目标动态调整学习路径推荐

if __name__ == '__main__':
    kg = main()
