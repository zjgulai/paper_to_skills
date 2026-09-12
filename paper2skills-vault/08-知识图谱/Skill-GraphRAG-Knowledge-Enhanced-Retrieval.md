# Skill Card: GraphRAG - 知识图谱增强检索生成

---

## ① 算法原理

### 核心思想
**GraphRAG（Graph Retrieval-Augmented Generation）** 将传统 RAG（检索增强生成）中的文本块检索升级为**知识图谱结构化检索**，通过图遍历获取与查询相关的实体、关系和子图，显著提升复杂推理场景的答案准确性和可解释性。

> ⚠️ **与 2608.28978 的负结果冲突，见 ①b 段**：上面这句「显著提升复杂推理场景的答案准确性」是
> 本卡的原有表述，**保留不改**，但必须按 ①b 的边界条件读 —— 该论文在 LongMemEval 上实测
> 基于抽取的图记忆 token F1 **0.417 < 0.468**（平铺向量基线），即「提升」并非无条件成立。

传统 RAG 的局限：
- 只能检索与查询字面相似的文本块
- 缺乏跨文档的关联推理能力
- 无法回答需要多跳推理的问题（如"买了吸奶器的用户中，哪些也购买了温奶器？"）

GraphRAG 的改进：
1. **结构化检索**：从知识图谱中检索实体和关系，而非原始文本
2. **多跳推理**：通过图遍历发现间接关联（用户→购买→商品→配件）
3. **可解释性**：答案可以追溯知识图谱中的具体路径

### 数学直觉

**检索评分函数**：

对于查询 $q$ 和候选实体 $e$，GraphRAG 计算相关性分数：

$$\text{score}(q, e) = \alpha \cdot \text{sim}_{\text{semantic}}(q, e) + (1-\alpha) \cdot \text{sim}_{\text{structural}}(q, e)$$

其中：
- $\text{sim}_{\text{semantic}}$：语义相似度（向量 embedding 余弦相似度）
- $\text{sim}_{\text{structural}}$：结构相似度（基于图距离）
- $\alpha$：权重参数（通常 0.6-0.8）

**多跳推理路径评分**：

对于从起点 $s$ 到终点 $t$ 的路径 $P = (e_0, e_1, ..., e_n)$：

$$\text{path\_score}(P) = \prod_{i=0}^{n-1} P(e_{i+1} | e_i, q) \cdot \text{rel}(e_i, q)$$

其中 $P(e_{i+1} | e_i, q)$ 是在查询 $q$ 条件下从 $e_i$ 跳转到 $e_{i+1}$ 的概率。

**混合检索策略**：

GraphRAG 通常采用三层检索架构：
1. **本地检索（Local）**：直接匹配查询中的实体
2. **邻居扩展（Neighbor）**：扩展一跳邻居获取上下文
3. **社区检索（Community）**：利用图社区发现获取主题相关子图

### 关键假设
- **知识图谱质量**：底层 KG 的完整性和准确性直接影响检索效果
- **查询可实体化**：用户查询可以映射到图谱中的实体或关系
- **图连通性**：相关实体在图中存在路径连接
- **语义一致性**：实体和关系的文本描述与查询语义一致

---

## ①b 反例与适用边界（负结果证据）

> **本节的证据等级说明**：本节全部边界来自一篇**负结果**论文 ——
> `2608.28978` *Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents*
> （arXiv preprint，在 LongMemEval 上评测，500 题）。它与本卡 ①②⑤ 段的正面主张**方向相反**，
> 故单列一节，**只用于限定本卡的适用范围**。
> ⚠️ 该论文把自己的结论严格限定在「基于抽取的流水线」这一范围内，
> **不能**读成「图结构记忆普遍无效」—— 见本节第 4 条，那是本节最重要的一处口径。

**1. 什么时候不要用（本卡最重要的一条边界）**

**当业务问题依赖「逐字回忆某条历史发言 / 某条具体推荐」时，不要用基于抽取的图记忆替代平铺文本检索。**
论文在 LongMemEval、**匹配 5 检索根候选生成预算**的条件下，图记忆**没有跑赢**平铺向量基线：
token F1 **0.417 vs 0.468**，配对 bootstrap（500 问）**Δ = −0.050，95% CI [−0.085, −0.016]**
（区间不含 0，差距显著）；judge 准确率 **0.454 vs 0.536**（⑥ Q1/Q5）。

差距**集中在**「单 session、需要回忆某条历史 assistant 发言」的问题上：
judge correctness 从 **0.911 掉到 0.607**（⑥ Q2/Q7）。
论文给的机制解释是：平铺基线能把原始 assistant 那一轮**逐字**取回，
而图抽取把一轮对话**分解成实体与关系**，于是丢掉了这类问题所依赖的
**表层形式（surface form）**（⑥ Q2/Q8）。

→ **对本卡场景一的直接含义**：客服 Agent 回答「你上次跟我说这个奶嘴不能高温消毒，对吗？」
这类**复述型**问题时，图记忆会系统性变差。这类场景**必须保留原文回溯通道**（原文 chunk 旁路），
不能只留图谱。

**2. 已知的失败模式（均有论文出处）**

| 失败模式 | 论文实测 | 出处 |
|---|---|---|
| 逐字 / 精确回忆（单 session·assistant） | judge **0.911 → 0.607** | ⑥ Q2 / Q7 |
| 知识更新 | F1 **0.456 vs 0.511**；无显式置信度时冲突消解**保留旧属性值**而不替换 | ⑥ Q16 |
| 整体 | token F1 **0.417 vs 0.468**；judge **0.454 vs 0.536** | ⑥ Q1 / Q5 |

**唯一的方向性例外**：时序推理是论文中图记忆**唯一**改善 judge 的题型（**0.293 vs 0.278**，⑥ Q6）——
事件可表示为带时序属性的 typed node 并连到参与实体。**本卡场景二（购买路径 / 时序）正落在这一例外里**，
而场景一（客服复述）落在最差的那一档。选型时请**按题型分流**，不要整站一刀切。

**3. 「剪枝 / 遗忘」那一半是正结果，可复用（本卡最该拿走的一条）**

同一篇论文的遗忘模块是**成功**的：对一张 **27,021 节点**的持久图应用**一次**，
移除 **9.8% 节点 / 9.5% 存储字节**（2,653 节点、2,560 边，440.6 MB → 398.6 MB），
token F1 **基本不变（+0.001，95% CI [−0.015, +0.016]）**；
judge 正确率下降 **1.6 个点**，95% 区间把任何损失的上界压在 **3.8 个点**（[−0.038, +0.006]）（⑥ Q4/Q12）。
打分口径：按 recency / access frequency / degree centrality / turn age 加权，
阈值以下的节点**连同关联边**一起删除，每 **400 轮**触发一次（⑥ Q13）。

→ **图记忆里可复用的是「重要度打分 + 尾部剪枝」，不是「抽取管线」。**
已经有一张图在跑的业务方，先上剪枝拿存储收益，比先重做抽取低风险得多。

**4. 结论的正确粒度（本卡强制口径）**

- ✅ 可以说：「这个**基于抽取的**图记忆流水线在 LongMemEval 上没跑赢平铺向量基线，
  且差距集中在逐字回忆类问题。」
- ❌ 不能说：「图结构记忆普遍不如向量检索」，也不能说「劝退所有投图记忆」。
  论文原话把结论限定为 **this extraction-based pipeline** 与 **at this model scale**，
  明确排除 **graph-structured memory in general**（⑥ Q3/Q10）；
  抽取器是**单个小模型（GPT-4o-mini）**、只评了**一个 benchmark**（⑥ Q15/Q3）。

**5. 论文自承、本卡无法消除的方法学缺口（照抄不加工）**

- **没有匹配压缩对照**：论文把「随机剪掉同样比例的节点」列为下一步**第一优先级**实验（⑥ Q11）——
  即「按重要度剪枝有效」尚未与「随便剪都一样」区分开。
- 只跑了 **4 次完整 run**（Exp1 处理 / 对照、Exp2 处理 / 对照），且**没有对保留参数做 sweep**（⑥ Q16/Q17）。
- **论文未讨论**母婴出海 / 跨境电商场景；**论文未报告**工程成本、抽取调用的 token 成本或端到端时延；
  **论文未报告**除 LongMemEval 之外的第二 benchmark。

**6. 本卡既有正面主张与本节的冲突（显式标注，未删除）**

① 核心思想中「复杂推理场景答案准确性」的正面表述，与 ⑤ 评估依据第 1 条援引的
微软 GraphRAG 准确率提升幅度 ——
⚠️ **这两处在本卡内都没有引文支撑，且与 2608.28978 的负结果方向相反。**
两者任务不同（微软 GraphRAG 是 query-focused summarization，2608.28978 是 agent 长效记忆），
**不能直接互相否定**；本卡保留原表述，但在此标注：**使用时不得当作已核验事实**。

---

## ② 母婴出海应用案例

### 场景一：智能客服问答系统

**业务问题**：
母婴出海电商的客服团队每天处理大量用户咨询，如"吸奶器A和吸奶器B有什么区别？"、"我买了吸奶器，还需要买什么配件？"。传统FAQ系统只能匹配关键词，无法处理需要关联推理的问题。

**数据要求**：
- 商品知识图谱：
  - 实体：商品（吸奶器、温奶器、储奶袋）、品牌、属性（价格、材质、功能）
  - 关系：配套购买、同类竞品、适用人群、使用场景
- 用户行为数据：购买记录、浏览记录、评价内容
- 客服历史记录：常见问题及标准答案

**预期产出**：
- **精准商品对比**：回答"吸奶器A和B的区别"时，自动提取两者在价格、功能、评价上的差异
  ```
  用户问：Spectra S1 和 Medela Pump 有什么区别？
  系统答：
  - 价格：Spectra S1 $199 vs Medela Pump $249
  - 便携性：Spectra内置电池（可充电）vs Medela需外接电源
  - 用户评价：Spectra静音评分4.5/5 vs Medela 4.2/5
  - 配套配件：Spectra兼容...（从图谱提取配套关系）
  ```
- **智能配件推荐**：回答"买了吸奶器还需要什么"时，基于图谱中的"配套购买"关系推荐
- **用户画像关联**：结合用户购买历史，提供个性化回答

**业务价值**：
- 客服响应时间从平均 5 分钟缩短至 30 秒
- 首次响应解决率从 60% 提升至 85%
- 客服人力成本节省 30-40%

---

### 场景二：用户购买路径分析与推荐

**业务问题**：
我们希望理解用户在母婴产品上的购买决策路径，例如"购买吸奶器的用户，通常还会在什么时间点购买什么配件？"这类多跳关联问题无法通过传统 SQL 查询或文本检索解决。

**数据要求**：
- 用户-商品交互图谱：
  - 用户节点： demographic 信息、会员等级
  - 商品节点：SKU、品类、品牌、价格带
  - 行为边：购买（时间、金额）、浏览、加购、评价
- 商品关系图谱：
  - 关系：互补商品（accessory）、替代商品（substitute）、升级路径（upgrade）

**预期产出**：
- **购买路径发现**：
  ```
  典型购买路径（从知识图谱遍历发现）：
  吸奶器 → 储奶袋（1周内购买率 45%）
        → 温奶器（2周内购买率 30%）
        → 消毒锅（1月内购买率 25%）
  ```
- **时机预测**：预测用户何时可能需要某个配件
- **个性化推荐**：基于用户当前购买阶段，推荐下一步可能需要的产品

**业务价值**：
- 配件交叉销售率提升 25-35%
- 用户生命周期价值（LTV）提升 15-20%
- 库存周转优化（基于预测提前备货）

---

## ③ 代码模板

```python
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
                    entity_info += f"→ [{step[1]}] → "
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
```

---

## ④ 技能关联

### 前置技能
- **知识图谱基础**：理解实体、关系、三元组等基本概念
- **RAG（检索增强生成）**：了解向量检索、上下文组装基本原理
- **图遍历算法**：理解 BFS、DFS、最短路径等图算法
- **文本向量化**：了解 Embedding 模型的基本原理

### 延伸技能
- **Graph Neural Networks (GNN)**：使用图神经网络学习更好的实体表示
- **多模态 GraphRAG**：整合文本、图像、视频等多模态信息
- **动态知识图谱检索**：支持时序知识图谱的检索
- **CausalRAG**：结合因果推理的检索增强生成

### 可组合技能
- **知识图谱构建**：与本技能共同构成完整的 KG-RAG 系统
- **智能客服系统**：GraphRAG 作为底层检索引擎
- **推荐系统**：基于图谱路径的跨域推荐
- **数据分析**：用户购买路径分析、关联规则挖掘

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 智能客服升级 | 响应时间减少90%，人力成本节省30-40% | 开发3-4周 | 15-20x |
| 交叉销售优化 | 配件销售提升25-35%，年增收50-100万 | 开发2-3周 | 10-15x |
| 用户洞察分析 | 购买路径发现驱动产品组合优化 | 开发2周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：需要构建完整的商品和用户知识图谱
- 技术门槛：较高，需要理解图算法和检索策略
- 工程复杂度：高，涉及知识图谱构建、检索优化、LLM集成
- 维护成本：中，知识图谱需要持续更新

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- **业务价值高**：客服和推荐是电商核心场景
- **技术前沿性**：GraphRAG 是 RAG 2.0 的发展方向
- **可落地性强**：可基于现有知识图谱技能扩展
- **竞争优势**：相比传统 RAG 有显著效果提升 ⚠️ **与 2608.28978 的负结果冲突，见 ①b 段** —— 该论文的图记忆在 LongMemEval 上并未跑赢平铺向量基线，「显著效果提升」不是无条件结论。

### 评估依据
1. **GraphRAG 相比传统 RAG 有明显优势**：微软研究显示 GraphRAG 在复杂推理任务上准确率提升 40%+ ⚠️ **本卡 ⑥ 段无对应引文支撑该数字，且与 2608.28978 的负结果方向相反，见 ①b 段第 6 条**（两者任务不同，不能直接互相否定，但此数字未经本卡核验）。
2. **母婴电商场景天然适合 KG**：商品关联、用户购买路径都有清晰的图结构
3. **基础设施复用**：可基于已萃取的 Knowledge Graph 技能快速构建
4. **长期演进路径清晰**：可向 GNN、多模态、CausalRAG 方向持续迭代

---

---

## ⑥ 原文引用

> **本段引文的论文与范围**：本段全部引文均来自 `2608.28978`
> *Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents*
> （arXiv preprint，全文存档：`papers/16-智能体工程/p2s-2026-0030/fulltext.md`）。
> 这是一篇**负结果**论文，因此本段引文主要用于**限定本卡的适用范围**（见 ①b 段），而非支持本卡的正面主张。
> 论文自设的范围限定见 Q3 与 Q10：结论仅适用于 **this extraction-based pipeline**，
> 不适用于 **graph-structured memory in general**。


**A. 负结果：图记忆没有跑赢平铺向量基线（本卡核心边界）**

> 原文:"On LongMemEval, the graph does not outperform a flat vector baseline at a matched candidate-generation budget of five retrieval roots: token F1 is $0.417$ against $0.468$, and a paired bootstrap over 500 questions gives $\Delta=-0.050$ (95% CI $[-0.085,-0.016]$)."
> 出处：2608.28978 §Abstract｜Q1

> 原文:"The gap is widest on questions that require recalling a specific prior assistant turn, where judged correctness falls from $0.911$ to $0.607$, suggesting that decomposing a turn into entities discards the surface form these questions depend on."
> 出处：2608.28978 §Abstract｜Q2

> 原文:"Because our extractor is a single small model evaluated on one benchmark, these results characterise this extraction-based pipeline rather than graph-structured memory in general."
> 出处：2608.28978 §Abstract｜Q3

> 原文:"The forgetting module is more successful. Applied once to a persistent 27,021-node graph, it removes 9.8% of nodes and 9.5% of stored bytes; token F1 is unchanged ($+0.001$, 95% CI $[-0.015,+0.016]$) and judged correctness falls by $1.6$ points, with the 95% interval bounding any loss at $3.8$ points ($[-0.038,+0.006]$)."
> 出处：2608.28978 §Abstract｜Q4

**B. 逐题型拆解：差距在哪里、唯一的例外在哪里**

> 原文:"In Experiment 1, the baseline RAG system outperforms Graph RAG overall, achieving a token F1 of 0.468 compared with 0.417 and an LLM-judge accuracy of 0.536 compared with 0.454."
> 出处：2608.28978 §5 Discussion and Limitations｜Q5

> 原文:"Graph RAG achieves its only improvement on the LLM-judge metric for temporal-reasoning questions (0.293 vs. 0.278)."
> 出处：2608.28978 §5 Discussion and Limitations｜Q6

> 原文:"The largest performance deficit occurs for single-session (assistant) questions (judge: 0.607 vs. 0.911). These questions often require recalling a specific recommendation or factual statement from a previous assistant response."
> 出处：2608.28978 §5 Discussion and Limitations｜Q7

> 原文:"This highlights an important limitation of extraction-based memory representations: structured abstraction can improve relational organization while simultaneously discarding information required for precise or verbatim recall."
> 出处：2608.28978 §5 Discussion and Limitations｜Q8

**C. 论文自设的适用范围（本卡强制口径的依据）**

> 原文:"Taken together, these results suggest that graph structure alone is not sufficient to improve long-term conversational memory. Its benefits are strongest when relationships and temporal structure are important, whereas flat text retrieval remains advantageous for precise or verbatim recall."
> 出处：2608.28978 §5 Discussion and Limitations｜Q9

> 原文:"Accordingly, our findings should be read as characterising this extraction-based graph memory pipeline at this model scale, not graph-structured memory in general."
> 出处：2608.28978 §A.1 Note to Reviewers on Experimental Scope and AI Use｜Q10

> 原文:"Given additional budget, our order of priority would be: a matched-compression control that prunes the same fraction of nodes at random, to isolate the contribution of the importance function; a second benchmark; and a stronger extraction model."
> 出处：2608.28978 §A.1 Note to Reviewers on Experimental Scope and AI Use｜Q11

**D.「剪枝 / 遗忘」那一半是正结果（同一篇论文，可复用）**

> 原文:"Applying the forgetting mechanism removes 2,653 nodes (9.8%) and 2,560 edges (5.5%), reducing the graph size from 440.6 MB to 398.6 MB, a 9.5% reduction."
> 出处：2608.28978 §5 Discussion and Limitations / Table 2｜Q12

> 原文:"Every 400 turns, the retention stage scores each node by recency, access frequency, centrality, and turn age, pruning nodes whose importance falls below the threshold together with their incident edges."
> 出处：2608.28978 §3.1 Architecture / Figure 1 caption｜Q13

> 原文:"At question time, the retrieval stage embeds the referenced entities, selects the top-5 matching nodes ranked by cosine similarity, and expands them via a 2-hop subgraph traversal to build the answer-generation context."
> 出处：2608.28978 §3.1 Architecture / Figure 1 caption｜Q14

> 原文:"Each conversational turn is processed by a single LLM extraction call (GPT-4o-mini) using a structured system prompt that defines the ontology, output schema, and extraction rules."
> 出处：2608.28978 §3.3 Extraction Pipeline｜Q15

**E. 知识更新失败模式（与冲突消解策略相关）**

> 原文:"Graph RAG also underperforms on knowledge-update questions (F1: 0.456 vs. 0.511). Inspection of failures indicates that the current conflict-resolution policy can retain an earlier attribute value instead of replacing it with a more recent value when no explicit confidence score is available."
> 出处：2608.28978 §5 Discussion and Limitations｜Q16

**F. 论文的实验规模与参数（用于判断上述结论的强度）**

> 原文:"We chose to spend it on four full runs (Experiment 1 treatment and control, Experiment 2 treatment and control) at $n=500$ with temperature $=0$, and to report paired bootstrap intervals over those runs, rather than on a larger number of partially evaluated configurations."
> 出处：2608.28978 §A.1 Note to Reviewers on Experimental Scope and AI Use｜Q17

> 原文:"We did not perform a full sweep over the retention parameters; the consequences of this are discussed at the end of this section."
> 出处：2608.28978 §A.2 Justification for parameter values｜Q18

<details><summary>Q 编号 ↔ 论文位置对照</summary>

| Q | 论文位置 | 行号 |
|---|---|---|
| Q1 | §Abstract | L15 |
| Q2 | §Abstract | L15 |
| Q3 | §Abstract | L15 |
| Q4 | §Abstract | L15 |
| Q5 | §5 Discussion and Limitations | L161 |
| Q6 | §5 Discussion and Limitations | L163 |
| Q7 | §5 Discussion and Limitations | L165 |
| Q8 | §5 Discussion and Limitations | L165 |
| Q9 | §5 Discussion and Limitations | L175 |
| Q10 | §A.1 Note to Reviewers on Experimental Scope and AI Use | L267 |
| Q11 | §A.1 Note to Reviewers on Experimental Scope and AI Use | L267 |
| Q12 | §5 Discussion and Limitations / Table 2 | L169 |
| Q13 | §3.1 Architecture / Figure 1 caption | L63 |
| Q14 | §3.1 Architecture / Figure 1 caption | L63 |
| Q15 | §3.3 Extraction Pipeline | L73 |
| Q16 | §5 Discussion and Limitations | L167 |
| Q17 | §A.1 Note to Reviewers on Experimental Scope and AI Use | L265 |
| Q18 | §A.2 Justification for parameter values | L275 |

</details>

---


---

## 参考论文

1. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization** (2024)
   - Microsoft Research
   - 核心贡献：提出 GraphRAG 框架，结合知识图谱社区发现和 LLM 生成

2. **Beyond the Parameters: A Technical Survey of Contextual Enrichment in LLMs** (2026)
   - arXiv:2604.03174v1
   - 核心贡献：RAG、GraphRAG、CausalRAG 的统一综述

3. **LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation** (2026)
   - arXiv:2604.02954v1
   - 核心贡献：GraphRAG 安全性研究，提供防护建议

4. **GraphWalk: Enabling Reasoning in LLMs through Tool-Based Graph Navigation** (2026)
   - arXiv:2604.01610v1
   - 核心贡献：基于工具调用的图遍历推理框架

---

## 开源资源

- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag
- **LangChain Graph RAG**: https://python.langchain.com/docs/use_cases/graph/
- **Neo4j LLM Knowledge Graph Builder**: https://github.com/neo4j-labs/llm-graph-builder
- **Hugging Face KG-RAG**: https://huggingface.co/papers/2404.16130

---

## 与 Knowledge Graph 技能的关联

本技能是 **Knowledge Graph for Skills Management** 的直接延伸：

```
Knowledge Graph 基础
        ↓
    图构建 + 图查询
        ↓
    GraphRAG 检索增强
        ↓
   （下一步）GNN 表示学习
```

**学习路径建议**：
1. 先掌握 Knowledge Graph 基础（图构建、图算法）
2. 再学习 GraphRAG（检索策略、上下文组装）
3. 最后可延伸至 GNN（图神经网络表示学习）

---

if __name__ == '__main__':
    graph_rag = main()
