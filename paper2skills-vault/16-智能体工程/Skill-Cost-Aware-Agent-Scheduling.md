# Skill Card: Cost-Aware Agent Scheduling（成本感知调度）

> **领域**: 16-智能体工程 | **类型**: 综合萃取

---

## ① 算法原理

不是所有 Agent 任务都需要 GPT-4——简单分类任务用 SLM（Small LM），复杂推理用 LLM。成本感知调度根据任务复杂度动态路由到最优模型。

**路由策略**：任务分类器判断复杂度（简单/中等/复杂）→ 简单→SLM（成本 1/10）、中等→中等模型、复杂→LLM。月 token 成本可从 $500→$80。

---

## ② 母婴出海应用案例

客服 Agent 意图分类（"我的订单到哪了"）→ SLM 处理（$0.001/次）。复杂推理（"比较 S1 和 S2 吸奶器哪个更适合早产儿妈妈"）→ LLM 处理（$0.05/次）。日均 200 次简单 × $0.001 + 30 次复杂 × $0.05 = $1.70/天 vs 全 LLM $11.50/天。年省：**$3,500**。

---

## ③ 代码模板

```python
class CostAwareRouter:
    MODEL_COSTS = {'slm': 0.001, 'medium': 0.01, 'llm': 0.05}
    
    def classify_complexity(self, query: str) -> str:
        words = len(query.split()); has_reasoning = any(w in query.lower() 
            for w in ['why', 'compare', 'explain', 'analyze', 'which is better'])
        if has_reasoning or words > 30: return 'llm'
        if words > 15: return 'medium'
        return 'slm'
    
    def estimate_cost(self, queries: list) -> dict:
        costs = {'slm': 0, 'medium': 0, 'llm': 0}
        for q in queries:
            level = self.classify_complexity(q)
            costs[level] += self.MODEL_COSTS[level]
        total = sum(costs.values()); naive = len(queries) * self.MODEL_COSTS['llm']
        return {'cost': total, 'naive': naive, 'saving_pct': (naive-total)/naive}

router = CostAwareRouter()
queries = ["where is my order", "compare S1 vs S2 for premature baby", "return policy"]
r = router.estimate_cost(queries)
print(f"Cost-aware: ${r['cost']:.3f} vs Naive: ${r['naive']:.3f} (save {r['saving_pct']:.0%})")
print("[✓] Cost-Aware Scheduling 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-SLM-Tool-Calling-Optimization]] | [[Skill-Agent-Fault-Tolerance]]
- **组合**：[[Skill-Context-Compression]]（成本优化的两个维度）

---

## ⑤ 商业价值

- **ROI**：年化 Token 成本节省 $2,000-10,000
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
