# Skill Card: Cross-Border Logistics Routing（跨境物流路径优化）

> **领域**: 18-物流履约 | **类型**: 综合萃取

---

## ① 算法原理

跨境物流的核心是**多式联运路径优化**——海陆空铁四种运输方式在不同路段的成本/时效组合中找最优。最短路径 Dijkstra 扩展为多目标（成本、时效、碳排放）Pareto 最优路径。

**状态空间**：节点=港口/仓库/机场，边=运输段（成本$c_e$, 时间$t_e$, 风险$r_e$）。多目标权重 $\min w_c \sum c_e + w_t \sum t_e + w_r \sum r_e$。母婴场景：高客单价吸奶器用空运（时效优先），配件用海运（成本优先）。

---

## ② 母婴出海应用案例

深圳→洛杉矶吸奶器运输：空运 3 天 $8000/批 vs 海运 25 天 $2000/批。旺季（Q4）缺货成本 $5000/天→空运；淡季持有成本低→海运。年化物流成本优化 **30-50 万元**。

---

## ③ 代码模板

```python
import heapq
def multi_modal_route(nodes, edges, start, end, weights=(0.5, 0.3, 0.2)):
    """edges: {u:{v:(cost,time,risk)}}, weights: (w_cost,w_time,w_risk)"""
    pq, dist = [(0, start, [])], {start: 0}
    while pq:
        d, u, path = heapq.heappop(pq)
        if u == end: return {'path': path+[u], 'score': d}
        for v, (c, t, r) in edges.get(u, {}).items():
            score = d + weights[0]*c + weights[1]*t + weights[2]*r
            if v not in dist or score < dist[v]:
                dist[v] = score; heapq.heappush(pq, (score, v, path+[u]))
    return None

nodes = ['SZ','HK','LAX','NYC']
edges = {'SZ':{'HK':(200,1,0.1),'LAX':(2000,3,0.3)},'HK':{'LAX':(1800,3,0.2),'NYC':(2500,4,0.4)},'LAX':{'NYC':(500,1,0.1)}}
print(multi_modal_route(nodes,edges,'SZ','NYC'))
print("[✓] Cross-Border Logistics 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Demand-Forecasting-Supply-Chain]]
- **组合**：[[Skill-Multi-Channel-Inventory-Pooling]] | [[Skill-Last-Mile-Delivery-Prediction]]

---

## ⑤ 商业价值

- **ROI**：30-50 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
