"""
GraphRAG - 知识图谱增强检索生成系统
用于母婴出海电商的智能问答和推荐

功能：
1. 知识图谱构建（商品、用户、行为）
2. 结构化检索（实体匹配、邻居扩展、路径搜索）
3. 混合评分（语义相似度 + 图结构相关性）
4. 上下文组装（生成 LLM 可用的结构化上下文）

Author: paper2skills
Date: 2026-04-06
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, deque
import json
import re


class KnowledgeGraph:
    """知识图谱基础类"""

    def __init__(self):
        self.entities = {}  # entity_id -> {type, name, attributes}
        self.relations = []  # [(source, relation_type, target, weight)]
        self.entity_index = defaultdict(set)  # entity_type -> {entity_ids}
        self.adjacency = defaultdict(list)  # entity_id -> [(target, relation, weight)]

    def add_entity(self, entity_id: str, entity_type: str, name: str,
                   attributes: Dict = None):
        """添加实体"""
        self.entities[entity_id] = {
            'id': entity_id,
            'type': entity_type,
            'name': name,
            'attributes': attributes or {}
        }
        self.entity_index[entity_type].add(entity_id)

    def add_relation(self, source: str, relation_type: str, target: str,
                     weight: float = 1.0, attributes: Dict = None):
        """添加关系"""
        rel = {
            'source': source,
            'type': relation_type,
            'target': target,
            'weight': weight,
            'attributes': attributes or {}
        }
        self.relations.append(rel)
        self.adjacency[source].append((target, relation_type, weight))

    def get_neighbors(self, entity_id: str, relation_type: str = None,
                      max_depth: int = 1) -> Dict[str, List[Tuple]]:
        """
        获取邻居节点

        Returns:
            {depth: [(entity_id, relation, weight), ...]}
        """
        results = defaultdict(list)
        visited = {entity_id}
        queue = deque([(entity_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for target, rel_type, weight in self.adjacency[current]:
                if relation_type and rel_type != relation_type:
                    continue
                if target not in visited:
                    visited.add(target)
                    results[depth + 1].append((target, rel_type, weight))
                    queue.append((target, depth + 1))

        return dict(results)

    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[Tuple]]:
        """
        查找两实体间的路径

        Returns:
            [[(entity, relation, next_entity), ...], ...]
        """
        paths = []
        queue = deque([(source, [])])

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_length:
                continue

            for next_entity, relation, weight in self.adjacency[current]:
                new_path = path + [(current, relation, next_entity, weight)]
                if next_entity == target:
                    paths.append(new_path)
                elif next_entity not in [p[0] for p in path]:
                    queue.append((next_entity, new_path))

        return paths


class GraphRAG:
    """
    GraphRAG - 知识图谱增强检索生成
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.entity_embeddings = {}  # 简化的实体向量表示

    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        从查询中提取实体（简化版，实际可用NER模型）

        Returns:
            [entity_id, ...]
        """
        extracted = []
        query_lower = query.lower()

        # 简单的字符串匹配（实际项目中应使用NER模型）
        for entity_id, entity in self.kg.entities.items():
            if entity['name'].lower() in query_lower:
                extracted.append(entity_id)

        return extracted

    def semantic_similarity(self, query: str, entity_id: str) -> float:
        """
        计算查询与实体的语义相似度（简化实现）

        实际项目中应使用：
        - 预训练的实体嵌入（TransE/RotatE）
        - 或调用 embedding API
        """
        entity = self.kg.entities[entity_id]
        entity_name = entity['name'].lower()
        query_lower = query.lower()

        # 简化的相似度计算
        if entity_name in query_lower:
            return 1.0

        # 计算词语重叠
        entity_words = set(entity_name.split())
        query_words = set(query_lower.split())
        overlap = len(entity_words & query_words)

        return min(overlap / max(len(entity_words), 1), 0.8)

    def retrieve_local(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        本地检索：直接匹配查询中的实体
        """
        entities = self.extract_entities_from_query(query)

        results = []
        for entity_id in entities:
            score = self.semantic_similarity(query, entity_id)
            results.append({
                'entity_id': entity_id,
                'entity': self.kg.entities[entity_id],
                'score': score,
                'source': 'local'
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def retrieve_neighbors(self, query: str, entity_ids: List[str],
                          max_depth: int = 1, top_k: int = 10) -> List[Dict]:
        """
        邻居扩展检索：获取相关实体的一跳/多跳邻居
        """
        results = []

        for entity_id in entity_ids:
            base_score = self.semantic_similarity(query, entity_id)
            neighbors = self.kg.get_neighbors(entity_id, max_depth=max_depth)

            for depth, neighbor_list in neighbors.items():
                decay = 0.5 ** depth  # 距离衰减
                for neighbor_id, relation, weight in neighbor_list:
                    neighbor_score = self.semantic_similarity(query, neighbor_id)
                    combined_score = (base_score * 0.3 + neighbor_score * 0.4 + weight * 0.3) * decay

                    results.append({
                        'entity_id': neighbor_id,
                        'entity': self.kg.entities[neighbor_id],
                        'source_entity': entity_id,
                        'relation': relation,
                        'path_length': depth,
                        'score': combined_score,
                        'source': 'neighbor'
                    })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def retrieve_by_path(self, query: str, source_entity: str,
                        target_types: List[str], max_length: int = 2) -> List[Dict]:
        """
        路径检索：查找从源实体到目标类型实体的路径

        适用于："买了X的用户还买了什么？"这类问题
        """
        results = []

        # BFS 搜索到目标类型实体的路径
        visited = {source_entity}
        queue = deque([(source_entity, [])])

        while queue:
            current, path = queue.popleft()

            if len(path) >= max_length:
                continue

            for target, relation, weight in self.kg.adjacency[current]:
                new_path = path + [(current, relation, target, weight)]

                # 检查是否到达目标类型
                target_entity = self.kg.entities.get(target)
                if target_entity and target_entity['type'] in target_types:
                    path_score = np.mean([p[3] for p in new_path])
                    semantic_score = self.semantic_similarity(query, target)
                    combined_score = path_score * 0.6 + semantic_score * 0.4

                    results.append({
                        'entity_id': target,
                        'entity': target_entity,
                        'path': new_path,
                        'path_length': len(new_path),
                        'score': combined_score,
                        'source': 'path'
                    })

                if target not in visited:
                    visited.add(target)
                    queue.append((target, new_path))

        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def hybrid_retrieve(self, query: str,
                       alpha: float = 0.7,
                       max_neighbors_depth: int = 1) -> Dict[str, List[Dict]]:
        """
        混合检索策略

        Args:
            query: 用户查询
            alpha: 语义相似度权重
            max_neighbors_depth: 邻居扩展深度

        Returns:
            {
                'local': [...],
                'neighbors': [...],
                'combined': [...]  # 融合后的结果
            }
        """
        # 1. 本地检索
        local_results = self.retrieve_local(query)

        # 2. 邻居扩展
        entity_ids = [r['entity_id'] for r in local_results]
        neighbor_results = self.retrieve_neighbors(
            query, entity_ids, max_depth=max_neighbors_depth
        )

        # 3. 融合排序
        all_results = {}

        # 归一化分数
        for r in local_results:
            r['final_score'] = alpha * r['score'] + (1 - alpha) * 0.5
            all_results[r['entity_id']] = r

        for r in neighbor_results:
            if r['entity_id'] in all_results:
                # 更新已有实体的分数
                existing = all_results[r['entity_id']]
                existing['final_score'] = max(existing['final_score'],
                                             alpha * r['score'] + (1 - alpha) * 0.8)
            else:
                r['final_score'] = alpha * r['score'] + (1 - alpha) * 0.8
                all_results[r['entity_id']] = r

        combined = sorted(all_results.values(),
                         key=lambda x: x['final_score'], reverse=True)

        return {
            'local': local_results,
            'neighbors': neighbor_results,
            'combined': combined
        }

    def build_context(self, query: str, retrieval_results: List[Dict],
                     max_context_length: int = 2000) -> str:
        """
        将检索结果组装为 LLM 可用的上下文

        Returns:
            结构化的上下文字符串
        """
        context_parts = []
        current_length = 0

        context_parts.append("# 知识图谱检索结果\n")

        for i, result in enumerate(retrieval_results[:10], 1):
            entity = result['entity']
            entity_info = f"\n## {i}. {entity['name']} (类型: {entity['type']})\n"
            entity_info += f"- 相关性评分: {result['final_score']:.3f}\n"

            # 添加属性
            if entity['attributes']:
                entity_info += "- 属性:\n"
                for k, v in list(entity['attributes'].items())[:3]:
                    entity_info += f"  - {k}: {v}\n"

            # 添加关系上下文
            if 'relation' in result:
                entity_info += f"- 关联路径: {result.get('source_entity', '未知')} "
                entity_info += f"--[{result['relation']}]--> {entity['name']}\n"

            if 'path' in result:
                entity_info += "- 完整路径:\n"
                for step in result['path']:
                    entity_info += f"  - {self.kg.entities.get(step[0], {}).get('name', step[0])} "
                    entity_info += f"-> [{step[1]}] -> "
                    entity_info += f"{self.kg.entities.get(step[2], {}).get('name', step[2])}\n"

            if current_length + len(entity_info) < max_context_length:
                context_parts.append(entity_info)
                current_length += len(entity_info)
            else:
                break

        return ''.join(context_parts)

    def answer(self, query: str, return_context: bool = False) -> Dict:
        """
        完整的 GraphRAG 问答流程

        Returns:
            {
                'query': 原始查询,
                'context': 检索上下文,
                'retrieval': 检索结果详情
            }
        """
        # 1. 混合检索
        retrieval_results = self.hybrid_retrieve(query)

        # 2. 构建上下文
        context = self.build_context(query, retrieval_results['combined'])

        result = {
            'query': query,
            'context': context,
            'retrieval': retrieval_results
        }

        if return_context:
            return result

        # 实际应用中，这里会将 context 传递给 LLM 生成答案
        # result['answer'] = llm.generate(prompt_template.format(query=query, context=context))

        return result


def create_maternal_baby_kg() -> KnowledgeGraph:
    """
    创建母婴出海电商的示例知识图谱
    """
    kg = KnowledgeGraph()

    # 添加商品实体
    products = [
        ('prod_spectra_s1', 'product', 'Spectra S1 吸奶器', {
            'brand': 'Spectra', 'price': 199, 'category': '吸奶器',
            'features': ['电动', '双边', '内置电池', '静音'],
            'rating': 4.5
        }),
        ('prod_medela_pump', 'product', 'Medela Pump 吸奶器', {
            'brand': 'Medela', 'price': 249, 'category': '吸奶器',
            'features': ['电动', '双边', '医院级'],
            'rating': 4.2
        }),
        ('prod_lansinoh_bags', 'product', 'Lansinoh 储奶袋', {
            'brand': 'Lansinoh', 'price': 12.99, 'category': '储奶袋',
            'quantity': '100片', 'rating': 4.6
        }),
        ('prod_avent_warmer', 'product', 'Philips Avent 温奶器', {
            'brand': 'Philips Avent', 'price': 45, 'category': '温奶器',
            'features': ['快速加热', '均匀温控'], 'rating': 4.3
        }),
        ('prod_dr_brown_bottle', 'product', 'Dr. Brown 奶瓶', {
            'brand': 'Dr. Brown', 'price': 18, 'category': '奶瓶',
            'features': ['防胀气'], 'rating': 4.4
        }),
    ]

    for entity_id, entity_type, name, attrs in products:
        kg.add_entity(entity_id, entity_type, name, attrs)

    # 添加用户实体
    users = [
        ('user_001', 'user', '新手妈妈A', {'type': 'new_mom', 'region': '美国'}),
        ('user_002', 'user', '二胎妈妈B', {'type': 'experienced', 'region': '加拿大'}),
    ]

    for entity_id, entity_type, name, attrs in users:
        kg.add_entity(entity_id, entity_type, name, attrs)

    # 添加关系
    relations = [
        # 互补商品关系
        ('prod_spectra_s1', 'complementary', 'prod_lansinoh_bags', 0.9),
        ('prod_spectra_s1', 'complementary', 'prod_avent_warmer', 0.7),
        ('prod_medela_pump', 'complementary', 'prod_lansinoh_bags', 0.85),
        ('prod_avent_warmer', 'complementary', 'prod_dr_brown_bottle', 0.6),

        # 竞品关系
        ('prod_spectra_s1', 'competitor', 'prod_medela_pump', 0.95),

        # 用户购买行为
        ('user_001', 'purchased', 'prod_spectra_s1', 1.0, {'date': '2026-01-15'}),
        ('user_001', 'purchased', 'prod_lansinoh_bags', 1.0, {'date': '2026-01-20'}),
        ('user_002', 'purchased', 'prod_medela_pump', 1.0, {'date': '2026-02-01'}),
        ('user_002', 'purchased', 'prod_avent_warmer', 1.0, {'date': '2026-02-10'}),
        ('user_002', 'purchased', 'prod_dr_brown_bottle', 1.0, {'date': '2026-02-15'}),
    ]

    for rel in relations:
        if len(rel) == 4:
            kg.add_relation(rel[0], rel[1], rel[2], rel[3])
        else:
            kg.add_relation(rel[0], rel[1], rel[2], rel[3], rel[4])

    return kg


def main():
    """主函数：演示 GraphRAG 系统"""
    print("=" * 80)
    print("母婴出海 - GraphRAG 知识图谱增强检索生成系统")
    print("=" * 80)

    # 1. 创建知识图谱
    print("\n[1] 创建母婴电商知识图谱...")
    kg = create_maternal_baby_kg()
    print(f"   实体数量: {len(kg.entities)}")
    print(f"   关系数量: {len(kg.relations)}")

    # 2. 初始化 GraphRAG
    print("\n[2] 初始化 GraphRAG...")
    graph_rag = GraphRAG(kg)

    # 3. 测试查询
    queries = [
        "Spectra S1 吸奶器",
        "买了吸奶器还需要什么配件？",
        "吸奶器竞品对比",
    ]

    for query in queries:
        print(f"\n[3] 查询: \"{query}\"")
        result = graph_rag.answer(query, return_context=True)

        print(f"   本地检索结果:")
        for r in result['retrieval']['local'][:3]:
            print(f"     - {r['entity']['name']} (评分: {r['score']:.3f})")

        print(f"   邻居扩展结果:")
        for r in result['retrieval']['neighbors'][:3]:
            print(f"     - {r['entity']['name']} "
                  f"(关系: {r.get('relation', 'N/A')}, 评分: {r['score']:.3f})")

        print(f"   生成的上下文长度: {len(result['context'])} 字符")

    # 4. 路径检索演示
    print("\n[4] 购买路径分析...")
    paths = graph_rag.retrieve_by_path(
        "买了吸奶器的用户还买了什么？",
        'prod_spectra_s1',
        target_types=['product'],
        max_length=2
    )
    print(f"   从 Spectra S1 出发的推荐路径:")
    for p in paths[:3]:
        print(f"     - {p['entity']['name']} "
              f"(路径长度: {p['path_length']}, 评分: {p['score']:.3f})")

    print("\n" + "=" * 80)
    print("GraphRAG 演示完成!")
    print("=" * 80)

    return graph_rag


if __name__ == '__main__':
    graph_rag = main()
