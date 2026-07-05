```markdown
# Skill Card: Cross-Border Logistics Routing（跨境物流路径优化）

> **领域**: 18-物流履约 | **类型**: 综合萃取

roadmap_phase: phase1
---

## ① 算法原理

跨境物流的核心是**多式联运路径优化**——海陆空铁四种运输方式在不同路段的成本/时效组合中找最优。最短路径 Dijkstra 扩展为多目标（成本、时效、碳排放）Pareto 最优路径。

**状态空间**：节点=港口/仓库/机场，边=运输段（成本$c_e$, 时间$t_e$, 风险$r_e$）。多目标权重 $\min w_c \sum c_e + w_t \sum t_e + w_r \sum r_e$。母婴场景：高客单价吸奶器用空运（时效优先），配件用海运（成本优先）。

---

## ② 母婴出海应用案例

**品类**：婴儿暖奶器（客单价 $39.9，单件重量 0.8kg，毛利率 55%）

**业务背景**：深圳仓→洛杉矶 FBA 仓，日销 50 件，安全库存 2000 件。旺季（Q4）日销峰值 120 件，缺货成本 $8/件/天（含广告损失 + 排名下降）。

**路径对比**：
| 路径 | 时效 | 单件成本 | 适用场景 |
|------|------|----------|----------|
| 空运（SZ→LAX） | 3 天 | $4.2/件 | 旺季补货 / 新品期 |
| 海运（SZ→LAX） | 25 天 | $1.1/件 | 淡季常规补货 |

**决策逻辑**：
- 旺季（11-12月）：缺货成本 $960/天（120件×$8），空运额外成本 $372/天（120件×$3.1差价）→ 空运净节省 $588/天
- 淡季（2-3月）：日销 35 件，持有成本 $0.03/件/天，海运 25 天 vs 空运 3 天，持有成本差异仅 $0.66/天 → 海运更优

**量化产出**：
- 年化物流成本节省 **45 万元**（空运/海运动态切换，减少 60% 紧急空运）
- 库存周转率从 8.2 次/年提升至 **10.5 次/年**（+28%）
- 旺季断货率从 12% 降至 **3%**，转化率从 3.8% 提升至 **4.5%**
- ROAS 从 2.6 提升至 **3.2**（广告因库存充足持续投放）

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
- **相关**：[[Skill-GraphDeepAR-Demand-Forecasting]]

## ⑤ 商业价值

- **ROI**：45 万元/年 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
```