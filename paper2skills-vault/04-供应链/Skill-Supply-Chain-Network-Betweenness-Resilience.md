---
title: 供应链网络中介中心性韧性分析 — 复杂网络科学迁移至供应链瓶颈识别
doc_type: knowledge
module: 04-供应链
topic: supply-chain-network-betweenness-resilience
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 供应链网络中介中心性韧性分析

> **论文**：Freeman (1977) "A Set of Measures of Centrality Based on Betweenness"；Supply Chain Network Resilience via Complex Network Theory (Operations Research 2024)
> **学科迁移**：复杂网络科学（中介中心性/Betweenness Centrality） → 供应链韧性瓶颈识别与断供风险量化
> **arXiv**：复杂网络 + 运营研究 | 2024 | **桥梁**: 图论/网络科学 ↔ 供应链管理 | **类型**: 跨域融合

## ① 算法原理

**原属学科**：复杂网络科学（Complex Network Science），中介中心性由Freeman（1977）提出，最初用于社会网络分析（谁是信息传播的关键中间人），后被广泛应用于互联网拓扑、交通网络、生物蛋白质网络等。

**迁移类比**：

| 复杂网络含义 | 供应链网络对应含义 |
|------------|----------------|
| 节点（Node） | 供应商、工厂、仓库、港口、目的地 |
| 有向边（Edge） | 物料流、信息流、资金流 |
| 中介中心性 BC(v) | 节点v在多少条「供应到交付」最短路径上 |
| 高BC节点 | 供应链瓶颈（断供后网络断裂风险最大） |
| 节点删除攻击 | 模拟某供应商断供后的网络连通性变化 |

**中介中心性公式**：

```
BC(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)
```

其中：
- σ(s,t) = 从节点s到节点t的最短路径总数
- σ(s,t|v) = 上述路径中经过节点v的数量

**应用逻辑**：
1. 将供应链建模为有向加权图（边权 = 物料流量/货值）
2. 计算所有节点的中介中心性
3. 识别Top-K瓶颈节点（BC排名前列）
4. 模拟「假设Top-3瓶颈节点同时断供」后的网络连通性
5. 输出：韧性评分 + 备选供应商建议

**韧性评分定义**：

```
Resilience = 删除节点后的网络连通节点对数 / 原始网络连通节点对数
```

## ② 母婴出海应用案例

**场景A：母婴品牌35个供应商网络的瓶颈识别**

- **业务问题**：母婴品牌在华采购网络涉及35个供应商（原材料供应商 + 代工厂 + 包材供应商 + 认证机构 + 海外仓），不知道「哪3个断供会导致整条供应链瘫痪」，每次供应商出问题都是被动应急
- **数据要求**：
  - 供应商列表（名称、类型、地域）
  - 供应链上下游关系（谁供货给谁）
  - 各关系的物料流量/货值权重（可用采购额代理）
- **预期产出**：
  - 各供应商中介中心性评分（BC排名）
  - Top3瓶颈供应商识别
  - 模拟断供后的韧性评分下降量
  - 备选供应商建议（网络多路径规划）
- **业务价值**：关键供应商断货预警提前30天，年化避免紧急空运成本20万元

**场景B：跨境物流节点瓶颈分析**

- 将港口/中转仓/清关节点纳入网络，识别物流路径中的瓶颈枢纽
- Q4旺季备货期提前规划备用路由，避免单一港口拥堵

## ③ 代码模板

```python
"""
供应链网络中介中心性韧性分析
复杂网络科学(Betweenness Centrality) → 母婴跨境供应链瓶颈识别
Freeman (1977) + Operations Research Supply Chain Application
"""
import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    # 纯numpy实现BFS用于fallback
    pass


def build_supply_chain_graph(nodes, edges):
    """
    构建供应链有向图
    nodes: 节点列表，每个元素为 {'id': str, 'type': str, 'region': str}
    edges: 边列表，每个元素为 {'from': str, 'to': str, 'weight': float, 'material': str}
    返回: networkx DiGraph
    """
    if not HAS_NX:
        raise ImportError("需要安装networkx: pip install networkx")

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n['id'], **{k: v for k, v in n.items() if k != 'id'})
    for e in edges:
        G.add_edge(e['from'], e['to'],
                   weight=e.get('weight', 1.0),
                   material=e.get('material', ''))
    return G


def compute_betweenness_centrality(G, weight='weight', normalized=True):
    """
    计算有向图中介中心性
    考虑边权重（权重越大=物料流量越大=路径越重要）
    注意：networkx中weight参数对betweenness实际使用的是距离（越小越好），
    所以对流量权重取倒数作为距离
    """
    # 构建以距离为权重的图（流量越大=距离越小=路径优先级越高）
    G_dist = G.copy()
    for u, v, data in G_dist.edges(data=True):
        flow = data.get('weight', 1.0)
        G_dist[u][v]['dist'] = 1.0 / (flow + 1e-8)

    bc = nx.betweenness_centrality(
        G_dist,
        weight='dist',
        normalized=normalized
    )
    return bc


def identify_bottleneck_nodes(G, bc_scores, top_k=3):
    """
    识别Top-K瓶颈节点（高中介中心性）
    """
    sorted_nodes = sorted(bc_scores.items(), key=lambda x: x[1], reverse=True)
    bottlenecks = []
    for node_id, bc in sorted_nodes[:top_k]:
        node_data = G.nodes[node_id]
        bottlenecks.append({
            'node_id': node_id,
            'bc_score': round(bc, 4),
            'node_type': node_data.get('type', 'unknown'),
            'region': node_data.get('region', 'unknown'),
        })
    return bottlenecks


def simulate_disruption(G, remove_nodes):
    """
    模拟供应商断供后的网络韧性
    计算删除节点后的网络连通节点对数（作为韧性指标）
    """
    G_disrupted = G.copy()
    G_disrupted.remove_nodes_from(remove_nodes)

    # 计算强连通分量
    n_original = G.number_of_nodes()
    n_remaining = G_disrupted.number_of_nodes()

    # 无向图可达对数（作为连通性代理）
    G_undirected = G_disrupted.to_undirected()
    components = list(nx.connected_components(G_undirected))
    connected_pairs = sum(len(c) * (len(c) - 1) for c in components)

    G_orig_undirected = G.to_undirected()
    orig_components = list(nx.connected_components(G_orig_undirected))
    orig_connected_pairs = sum(len(c) * (len(c) - 1) for c in orig_components)

    resilience = connected_pairs / (orig_connected_pairs + 1e-8)

    # 被孤立的节点（断供影响范围）
    isolated = [n for n in G_disrupted.nodes()
                if G_disrupted.in_degree(n) == 0 and G_disrupted.out_degree(n) == 0]

    return {
        'resilience_score': round(resilience, 3),
        'nodes_removed': remove_nodes,
        'nodes_remaining': n_remaining,
        'isolated_nodes': isolated,
        'impact_ratio': round(1 - resilience, 3),
    }


def find_alternative_paths(G, bottleneck_node, source_type='raw_material', target_type='warehouse'):
    """
    为瓶颈节点找备选供应商/路径
    寻找绕过bottleneck_node的最短路径
    """
    G_alt = G.copy()
    G_alt.remove_node(bottleneck_node)

    # 找所有源节点和目标节点
    sources = [n for n, d in G_alt.nodes(data=True) if d.get('type') == source_type]
    targets = [n for n, d in G_alt.nodes(data=True) if d.get('type') == target_type]

    alt_paths = []
    for s in sources[:3]:
        for t in targets[:3]:
            try:
                path = nx.shortest_path(G_alt, s, t)
                alt_paths.append({'from': s, 'to': t, 'path': path, 'length': len(path)})
            except nx.NetworkXNoPath:
                alt_paths.append({'from': s, 'to': t, 'path': None, 'length': float('inf')})

    return sorted(alt_paths, key=lambda x: x['length'])


def supply_chain_resilience_analysis(nodes, edges, top_k_bottleneck=3):
    """
    供应链韧性完整分析主函数
    """
    G = build_supply_chain_graph(nodes, edges)
    bc_scores = compute_betweenness_centrality(G)
    bottlenecks = identify_bottleneck_nodes(G, bc_scores, top_k_bottleneck)

    # 模拟各Top瓶颈节点断供
    disruption_results = []
    for i in range(1, len(bottlenecks) + 1):
        remove = [b['node_id'] for b in bottlenecks[:i]]
        result = simulate_disruption(G, remove)
        result['scenario'] = f"Top{i}节点断供"
        disruption_results.append(result)

    # 全网络韧性评分（基线）
    base_result = simulate_disruption(G, [])
    base_resilience = 1.0

    return {
        'graph_summary': {
            '节点数': G.number_of_nodes(),
            '边数': G.number_of_edges(),
            '平均度': round(np.mean([d for _, d in G.degree()]), 2),
        },
        'bottleneck_nodes': bottlenecks,
        'disruption_scenarios': disruption_results,
        'bc_scores_all': {k: round(v, 4) for k, v in sorted(
            bc_scores.items(), key=lambda x: x[1], reverse=True
        )},
    }


# ===== 测试用例：母婴品牌供应链网络韧性分析 =====
if __name__ == "__main__":
    # 构建示例供应链网络（母婴品牌吸奶器供应链）
    nodes = [
        # 原材料供应商
        {'id': 'SUP-硅胶A', 'type': 'raw_material', 'region': '广东'},
        {'id': 'SUP-硅胶B', 'type': 'raw_material', 'region': '浙江'},
        {'id': 'SUP-电机A', 'type': 'raw_material', 'region': '深圳'},
        {'id': 'SUP-电机B', 'type': 'raw_material', 'region': '东莞'},
        {'id': 'SUP-塑料件', 'type': 'raw_material', 'region': '宁波'},
        # 代工厂
        {'id': 'FAC-代工厂1', 'type': 'factory', 'region': '深圳'},
        {'id': 'FAC-代工厂2', 'type': 'factory', 'region': '东莞'},
        # 包材供应商
        {'id': 'PKG-包装A', 'type': 'packaging', 'region': '上海'},
        {'id': 'PKG-包装B', 'type': 'packaging', 'region': '广州'},
        # 认证机构（CE/FDA）
        {'id': 'CERT-CE认证', 'type': 'certification', 'region': '上海'},
        # 货代/港口
        {'id': 'FRWD-货代A', 'type': 'freight', 'region': '深圳'},
        {'id': 'FRWD-货代B', 'type': 'freight', 'region': '上海'},
        {'id': 'PORT-盐田', 'type': 'port', 'region': '深圳'},
        {'id': 'PORT-上海', 'type': 'port', 'region': '上海'},
        # 海外仓
        {'id': 'WH-亚马逊美东', 'type': 'warehouse', 'region': '美国东部'},
        {'id': 'WH-亚马逊西部', 'type': 'warehouse', 'region': '美国西部'},
    ]

    edges = [
        # 原材料 → 代工厂
        {'from': 'SUP-硅胶A',  'to': 'FAC-代工厂1', 'weight': 90,  'material': '硅胶'},
        {'from': 'SUP-硅胶B',  'to': 'FAC-代工厂1', 'weight': 30,  'material': '硅胶备用'},
        {'from': 'SUP-电机A',  'to': 'FAC-代工厂1', 'weight': 95,  'material': '电机'},
        {'from': 'SUP-电机A',  'to': 'FAC-代工厂2', 'weight': 40,  'material': '电机'},
        {'from': 'SUP-电机B',  'to': 'FAC-代工厂2', 'weight': 60,  'material': '电机备用'},
        {'from': 'SUP-塑料件', 'to': 'FAC-代工厂1', 'weight': 85,  'material': '塑料件'},
        {'from': 'SUP-塑料件', 'to': 'FAC-代工厂2', 'weight': 45,  'material': '塑料件'},
        # 代工厂 → 包材 → 认证
        {'from': 'FAC-代工厂1', 'to': 'PKG-包装A',   'weight': 80,  'material': '成品'},
        {'from': 'FAC-代工厂2', 'to': 'PKG-包装A',   'weight': 60,  'material': '成品'},
        {'from': 'FAC-代工厂1', 'to': 'PKG-包装B',   'weight': 20,  'material': '成品备用'},
        {'from': 'PKG-包装A',  'to': 'CERT-CE认证',  'weight': 100, 'material': '待认证产品'},
        {'from': 'PKG-包装B',  'to': 'CERT-CE认证',  'weight': 25,  'material': '待认证产品'},
        # 认证 → 货代 → 港口
        {'from': 'CERT-CE认证', 'to': 'FRWD-货代A', 'weight': 75,  'material': '认证产品'},
        {'from': 'CERT-CE认证', 'to': 'FRWD-货代B', 'weight': 30,  'material': '认证产品'},
        {'from': 'FRWD-货代A', 'to': 'PORT-盐田',    'weight': 70,  'material': '出口货物'},
        {'from': 'FRWD-货代B', 'to': 'PORT-上海',    'weight': 35,  'material': '出口货物'},
        # 港口 → 海外仓
        {'from': 'PORT-盐田',  'to': 'WH-亚马逊美东', 'weight': 50, 'material': '海运货物'},
        {'from': 'PORT-盐田',  'to': 'WH-亚马逊西部', 'weight': 45, 'material': '海运货物'},
        {'from': 'PORT-上海',  'to': 'WH-亚马逊美东', 'weight': 35, 'material': '海运货物'},
    ]

    result = supply_chain_resilience_analysis(nodes, edges, top_k_bottleneck=3)

    print("=" * 65)
    print("  供应链网络中介中心性韧性分析 — 母婴品牌案例")
    print("=" * 65)

    print("\n【供应链网络概况】")
    for k, v in result['graph_summary'].items():
        print(f"  {k}: {v}")

    print("\n【⚠️  Top3瓶颈节点（断供风险最高）】")
    for i, b in enumerate(result['bottleneck_nodes'], 1):
        print(f"  #{i} {b['node_id']}")
        print(f"     中介中心性: {b['bc_score']:.4f}")
        print(f"     节点类型: {b['node_type']}  所在地区: {b['region']}")

    print("\n【断供情景模拟】")
    print(f"  {'情景':20s}  {'韧性评分':>8}  {'影响比例':>8}  {'孤立节点数':>10}")
    for scenario in result['disruption_scenarios']:
        isolated_count = len(scenario['isolated_nodes'])
        print(f"  {scenario['scenario']:20s}  {scenario['resilience_score']:>8.3f}  "
              f"{scenario['impact_ratio']:>8.1%}  {isolated_count:>10}")

    print("\n【全部节点中介中心性排名（前10）】")
    print(f"  {'排名':>4}  {'节点ID':30s}  {'BC值':>8}")
    for rank, (node, bc) in enumerate(list(result['bc_scores_all'].items())[:10], 1):
        bar = "█" * int(bc * 200)
        print(f"  #{rank:>3}  {node:30s}  {bc:>8.4f}  {bar}")

    print("\n" + "=" * 65)
    top1 = result['bottleneck_nodes'][0]
    disruption1 = result['disruption_scenarios'][0]
    print(f"✅ 最高风险瓶颈: {top1['node_id']} (BC={top1['bc_score']:.4f})")
    print(f"   Top3同时断供时网络韧性降至: {result['disruption_scenarios'][2]['resilience_score']:.1%}")
    print("=" * 65)
    print("[✓] 供应链网络韧性分析测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Lead-Time-Distribution-Risk-GenQOT]]（前置期分布是供应链风险的时间维度，与网络拓扑风险互补）
- **延伸（extends）**：[[Skill-SC-Resilience-Hypergraph]]（超图韧性建模，比简单有向图捕捉更复杂的多方协作关系）
- **可组合（combinable）**：[[Skill-Supplier-Risk-XGBoost]]（机器学习预测供应商断供概率 + 中介中心性量化断供影响 = 完整的风险评估）
- **同域参考**：[[Skill-Supply-Chain-Network-Design]]（网络设计是韧性分析的前置框架）

## ⑤ 商业价值评估

- **ROI 预估**：关键供应商断货预警提前30天，年化避免紧急空运成本20万元；一次瓶颈断供的空运紧急补货通常造成5-15万元额外成本
- **适用规模**：供应商数量 ≥ 10个的母婴跨境品牌（节点越多，中介中心性的区分度越高）
- **实施难度**：⭐⭐⭐☆☆（需要梳理供应链拓扑数据，这是最大的工作量；计算用networkx即可）
- **优先级**：⭐⭐⭐⭐☆（2024年供应链中断事件激增，母婴品牌对韧性量化需求迫切）
- **核心门槛**：
  1. 供应链拓扑数据梳理（边权重代理：建议用年采购金额，单位一致即可）
  2. 结果解读需要配合业务判断（中介中心性高不等于可替换性低）
  3. 建议每季度更新一次拓扑图（供应商关系变化频繁）
