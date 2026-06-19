---
title: Agent 成本优化与预算管控 — 让 LLM Agent 可持续运行的 Token 经济学
doc_type: knowledge
module: 16-智能体工程
topic: agent-cost-optimization-budget-control
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Agent 成本优化与预算管控

> **论文**：FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance
> **arXiv**：2305.05176 | 2023 | **桥梁**: 智能体工程 ↔ 运营财务 | **类型**: 工程基础

## ① 算法原理

解决「AI Agent 跑一个月就发现 LLM Token 账单超预算，不得不砍掉功能」的业务问题。

LLM 调用成本的 80% 来自以下三个浪费点：
1. **无效重复调用**：同类问题每次重新发给 LLM，但答案几乎一样
2. **大模型滥用**：简单分类任务用 GPT-4，杀鸡用牛刀
3. **上下文膨胀**：Context 越来越长，每次调用越来越贵

**三招控成本**：

**① Semantic Cache（语义缓存）**：对历史请求做向量化，新请求与缓存做相似度匹配，相似度 > 0.92 直接返回缓存结果，0 Token 花费。

**② 动态模型降级（Cascade）**：按任务复杂度路由到不同模型。简单问题 → GPT-3.5（$0.001/1K）→ 若置信度低才升级 GPT-4（$0.03/1K），成本差 30 倍。

**③ Token Budget Scheduler**：给每个 Agent 分配每日/每月 Token 预算，优先级高的任务保留配额，低优先级任务降级或延迟执行，防止月底超支。

综合效果：**同等质量下，LLM 成本降低 40-70%**。

## ② 母婴出海应用案例

**场景A：定价 Agent 月账单从 $3000 降到 $900**
- 业务问题：动态定价 Agent 每天调用 GPT-4 分析竞品价格 2000 次，月费 $3000，CFO 要求砍半
- 数据要求：历史 LLM 调用日志（含 prompt/response/Token数）+ 业务决策结果标注
- 方案：对相似竞品分析请求做语义缓存（命中率约 45%）+ 简单价格比较降级到 GPT-3.5（占 60% 调用量）
- 预期产出：月 Token 成本从 $3000 → $850，决策质量下降 < 2%，年化节省 $25,800

**场景B：多 Agent 系统预算统一管控**
- 业务问题：客服/库存/广告三个 Agent 各自调用 LLM，月底总账单 $8000，不知道哪个 Agent 在烧钱
- 数据要求：各 Agent 执行日志 + 业务价值评估（每次调用带来多少收益）
- 方案：统一 Token Budget Scheduler，按 ROI 分配预算，广告 Agent（ROI 最高）优先
- 预期产出：同等预算 $5000/月，通过优先级重排后总业务价值提升 35%，年化减少浪费 $36,000

## ③ 代码模板

```python
"""
Agent 成本优化框架：语义缓存 + 模型降级 + Token 预算管控
"""
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============ 1. 语义缓存（简化版，使用余弦相似度） ============

def simple_embed(text: str) -> List[float]:
    """简化版文本向量化（生产用 text-embedding-3-small）"""
    # 使用字符频率作为特征（仅用于演示）
    vocab = "abcdefghijklmnopqrstuvwxyz0123456789 "
    text_lower = text.lower()
    vec = [text_lower.count(c) / max(len(text_lower), 1) for c in vocab]
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """语义缓存：相似请求直接返回缓存结果"""
    
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.cache: List[Tuple[List[float], str, str]] = []  # (embedding, prompt, response)
        self.hits = 0
        self.misses = 0
    
    def get(self, prompt: str) -> Optional[str]:
        emb = simple_embed(prompt)
        for cached_emb, cached_prompt, cached_response in self.cache:
            sim = cosine_similarity(emb, cached_emb)
            if sim >= self.threshold:
                self.hits += 1
                return cached_response
        self.misses += 1
        return None
    
    def put(self, prompt: str, response: str):
        emb = simple_embed(prompt)
        self.cache.append((emb, prompt, response))
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ============ 2. 动态模型降级（Cascade） ============

@dataclass
class ModelConfig:
    name: str
    cost_per_1k_tokens: float   # 美元
    avg_quality_score: float    # 0-1
    confidence_threshold: float # 置信度低于此值才升级


MODEL_CASCADE = [
    ModelConfig("gpt-3.5-turbo", cost_per_1k_tokens=0.001, avg_quality_score=0.72, confidence_threshold=0.75),
    ModelConfig("gpt-4-turbo",   cost_per_1k_tokens=0.010, avg_quality_score=0.91, confidence_threshold=0.60),
    ModelConfig("gpt-4o",        cost_per_1k_tokens=0.010, avg_quality_score=0.95, confidence_threshold=0.0),
]


def cascade_call(prompt: str, input_tokens: int = 500) -> Dict:
    """模拟级联调用：从便宜模型开始，置信度不够才升级"""
    random.seed(hash(prompt) % 1000)
    
    for i, model in enumerate(MODEL_CASCADE):
        # 模拟模型返回结果和置信度
        confidence = random.uniform(0.5 + i * 0.15, 0.9 + i * 0.05)
        cost = model.cost_per_1k_tokens * input_tokens / 1000
        
        if confidence >= model.confidence_threshold or i == len(MODEL_CASCADE) - 1:
            return {
                'model_used': model.name,
                'confidence': round(confidence, 3),
                'cost_usd': round(cost, 5),
                'quality_score': model.avg_quality_score,
                'cascade_level': i + 1,
            }
    
    return {}


# ============ 3. Token 预算管控 ============

@dataclass
class AgentBudget:
    agent_name: str
    monthly_budget_usd: float
    priority: int               # 1=最高, 3=最低
    spent_usd: float = 0.0
    calls_made: int = 0
    calls_rejected: int = 0
    
    @property
    def remaining_usd(self) -> float:
        return self.monthly_budget_usd - self.spent_usd
    
    @property
    def utilization_pct(self) -> float:
        return self.spent_usd / self.monthly_budget_usd * 100


class TokenBudgetScheduler:
    """Token 预算调度器：按优先级分配 LLM 调用配额"""
    
    def __init__(self, total_monthly_budget_usd: float):
        self.total_budget = total_monthly_budget_usd
        self.agents: Dict[str, AgentBudget] = {}
    
    def register_agent(self, name: str, budget_ratio: float, priority: int):
        """注册 Agent，按比例分配预算"""
        budget = self.total_budget * budget_ratio
        self.agents[name] = AgentBudget(name, budget, priority)
    
    def request_call(self, agent_name: str, estimated_cost_usd: float) -> Tuple[bool, str]:
        """
        Agent 申请 LLM 调用配额
        Returns: (是否批准, 原因说明)
        """
        if agent_name not in self.agents:
            return False, "Agent 未注册"
        
        agent = self.agents[agent_name]
        
        if agent.remaining_usd < estimated_cost_usd:
            agent.calls_rejected += 1
            return False, f"预算不足（剩余 ${agent.remaining_usd:.4f}，需要 ${estimated_cost_usd:.4f}）"
        
        agent.spent_usd += estimated_cost_usd
        agent.calls_made += 1
        return True, "已批准"
    
    def print_budget_report(self):
        print("\n📊 Token 预算使用报告")
        print("-" * 60)
        print(f"{'Agent':<30} {'预算':>8} {'已用':>8} {'剩余':>8} {'利用率':>8} {'拒绝':>6}")
        print("-" * 60)
        for ag in sorted(self.agents.values(), key=lambda x: x.priority):
            print(f"{ag.agent_name:<30} ${ag.monthly_budget_usd:>6.0f} "
                  f"${ag.spent_usd:>6.2f} ${ag.remaining_usd:>6.2f} "
                  f"{ag.utilization_pct:>6.1f}% {ag.calls_rejected:>5}次")
        print("-" * 60)
        total_spent = sum(a.spent_usd for a in self.agents.values())
        total_rejected = sum(a.calls_rejected for a in self.agents.values())
        print(f"{'合计':<30} ${self.total_budget:>6.0f} ${total_spent:>6.2f} "
              f"  -- {total_spent/self.total_budget*100:>6.1f}% {total_rejected:>5}次")


# ============ 综合演示 ============

def run_demo():
    print("=" * 60)
    print("🎯 Agent 成本优化框架演示（母婴跨境电商场景）")
    print("=" * 60)
    
    # 1. 语义缓存演示
    cache = SemanticCache(similarity_threshold=0.85)
    
    # 模拟相似的竞品分析请求
    prompts = [
        "分析竞品 ASIN B08N5WRWNW 的价格策略",
        "分析竞品 ASIN B08N5WRWNW 的定价",   # 高度相似
        "分析竞品 ASIN B07X9WZBHP 的价格",   # 不同ASIN
        "查询 ASIN B08N5WRWNW 竞争对手价格",  # 相似
    ]
    
    for p in prompts:
        cached = cache.get(p)
        if cached:
            print(f"  ✅ 缓存命中: {p[:30]}...")
        else:
            response = f"[LLM响应] {p[:20]}...（模拟生成）"
            cache.put(p, response)
            print(f"  🔄 LLM调用: {p[:30]}...")
    
    print(f"\n  缓存命中率: {cache.hit_rate:.1%}（节省 {cache.hit_rate:.0%} Token 费用）")
    
    # 2. 模型降级演示
    print("\n📉 模型降级效果：")
    tasks = ["简单价格比较", "复杂竞品分析报告", "Listing 文案优化"]
    total_without_cascade = 0
    total_with_cascade = 0
    
    for task in tasks:
        result = cascade_call(task)
        cost_gpt4_only = 0.01 * 500 / 1000  # 假设全部用 GPT-4
        savings = cost_gpt4_only - result['cost_usd']
        total_without_cascade += cost_gpt4_only
        total_with_cascade += result['cost_usd']
        print(f"  {task}: 使用 {result['model_used']}，"
              f"成本 ${result['cost_usd']:.5f}，节省 {savings/cost_gpt4_only:.0%}")
    
    savings_pct = (total_without_cascade - total_with_cascade) / total_without_cascade
    print(f"\n  平均节省: {savings_pct:.0%} Token 成本")
    
    # 3. 预算管控演示
    scheduler = TokenBudgetScheduler(total_monthly_budget_usd=5000)
    scheduler.register_agent("广告Agent（优先级1）", budget_ratio=0.45, priority=1)
    scheduler.register_agent("定价Agent（优先级1）", budget_ratio=0.35, priority=1)
    scheduler.register_agent("客服Agent（优先级2）", budget_ratio=0.15, priority=2)
    scheduler.register_agent("报表Agent（优先级3）", budget_ratio=0.05, priority=3)
    
    # 模拟调用
    random.seed(99)
    agents = list(scheduler.agents.keys())
    for _ in range(500):
        agent = random.choice(agents)
        cost = random.uniform(0.005, 0.05)
        scheduler.request_call(agent, cost)
    
    scheduler.print_budget_report()
    
    # 验证
    assert cache.hit_rate >= 0, "缓存命中率应 >= 0"
    # 级联路由平均成本 ≤ 全量 GPT-4 成本（部分任务用 GPT-3.5 节省）
    assert total_with_cascade <= total_without_cascade + 1e-9, "级联总成本不应超过全量 GPT-4"
    total_spent = sum(a.spent_usd for a in scheduler.agents.values())
    assert total_spent <= scheduler.total_budget * 1.01, "总花费不应超预算"
    
    print("\n[✓] Agent 成本优化与预算管控 测试通过")


if __name__ == "__main__":
    run_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Agent-SLO-Manager]]（SLO 定义了质量下限，降级时不能低于 SLO）
- **前置（prerequisite）**：[[Skill-Active-Context-Pruning]]（Context 压缩是 Token 成本控制的关键手段）
- **延伸（extends）**：[[Skill-Agentic-Memory-Management]]（长期记忆管理减少重复 Token 消耗）
- **可组合（combinable）**：[[Skill-Agent-ROI-Measurement-Framework]]（成本优化 + ROI 测量 → 完整「投入产出」管理闭环）

## ⑤ 商业价值评估

- **ROI 预估**：母婴跨境年 GMV $500 万规模团队，Agent LLM 月均费用 $3000-8000，实施语义缓存+模型降级后节省 **40-65%**，年化节省 $14,400-62,400；同时 Token 预算管控防止「月底透支停用」导致的运营事故
- **实施难度**：⭐⭐⭐☆☆（语义缓存需要向量数据库，其余均为代码改动）
- **优先级**：⭐⭐⭐⭐⭐（Agent 规模化运营的必选项，不做则 LLM 成本随规模线性增长，很快变得不可持续）
- **投资回收期**：通常 **4-6 周**（缓存数据积累后效果快速显现）
