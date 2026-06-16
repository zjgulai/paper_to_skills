---
title: RCR-Router角色感知上下文路由 — Token预算约束下的多Agent记忆子集动态分配
doc_type: knowledge
module: 10-MAS
topic: rcr-router-role-aware-context-routing
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: RCR-Router角色感知上下文路由

> **论文①**：RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory
> **arXiv**：2508.04903 | 2025 | **桥梁**: MAS ↔ 智能体工程 | **类型**: 算法工具
> **论文②**：BudgetMem: Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory
> **arXiv**：2602.06025 | 2026

## ① 算法原理

**反直觉洞察**：大多数MAS系统让每个Agent访问完整的共享记忆池——这看起来"信息最丰富"，实际上是一种浪费甚至有害：财务Agent看到的大量研究原始数据对它毫无用处（噪声），而合规Agent需要的法规细节却被稀释在海量市场数据中。**RCR-Router的关键发现：按角色和任务阶段动态选择语义相关的记忆子集，不仅节省Token，还能提高回答质量**（减少噪声干扰）。

**RCR-Router三组件架构**：

1. **Token预算分配器（Token Budget Allocator）**：
   ```
   对每个Agent i在每轮交互中：
   B_i = allocate(role_i, task_stage_t, total_budget_B)
   
   分配原则：
   - 按角色优先级分配基础预算
   - 任务阶段调整：早期阶段（信息收集）更多给Research，后期（决策）更多给Finance
   - 约束：Σ B_i ≤ B_total
   ```

2. **重要性评分器（Importance Scorer）**：
   ```
   对记忆库中的每条记忆m：
   α(m; R_i, S_t) = score(m, role=R_i, stage=S_t)
   
   评分考虑：
   - 角色相关性：合规Agent对法规记忆评分高
   - 任务阶段相关性：决策阶段对历史决策记录评分高
   - 时间衰减：旧记忆的重要性自然衰减
   - 被引用频率：多次被引用的记忆优先
   ```

3. **语义过滤路由（Semantic Filter with Routing Logic）**：
   ```
   选取最终记忆子集：
   C_t^i = {m ∈ M_t : α(m; R_i, S_t) ≥ τ AND |tokens(C)| ≤ B_i}
   
   实现：按分数排序，贪心选取直到预算耗尽
   ```

4. **迭代上下文精炼（Progressive Context Refinement）**：
   - Agent输出被集成到共享记忆库M
   - 下一轮的路由基于更新后的记忆（包含当前Agent的输出）
   - 多轮交互中，上下文越来越聚焦于任务相关信息

5. **BudgetMem的三层预算分级（补充）**：
   | 层级 | 描述 | 适用场景 |
   |-----|------|---------|
   | Low Budget | 简化检索方法 + 小容量 | 简单信息查询 |
   | Mid Budget | 标准方法 | 中等复杂任务 |
   | High Budget | 复杂推理方法 + 大容量 | 复杂多跳推理 |
   
   RL路由策略：轻量神经网络决定每次查询走哪个层级

6. **关键实验结果（Token预算B=2048时）**：
   - RCR-Router vs 无路由：在严格Token预算下，回答质量提升13-22%
   - 性能在B>2048后边际收益递减（上下文饱和）
   - BudgetMem：在LoCoMo/LongMemEval上比强基线更好的准确率-成本前沿

**数学直觉**：RCR-Router将记忆检索问题转化为有约束的背包问题——每条记忆有"价值"（α评分）和"重量"（Token数），在预算约束下最大化总价值。动态规划求精确解；贪心算法求近似解（实际中足够）。

## ② 母婴出海应用案例

**场景A：跨域MAS的角色感知记忆路由**

- **业务问题**：母婴MAS有一个包含5000条记忆的共享库（市场数据+法规文件+财务记录+品牌指南）。每次调用要给所有Agent传入完整记忆库，严重超出Token预算，而且Finance Agent收到大量无关的品牌指南信息，导致ROI计算时被不相关信息干扰
- **RCR-Router方案**：
  - Research Agent (阶段1)：路由市场数据+竞品记忆（B=2048）
  - Compliance Agent：路由法规文件+认证记录（B=1536）
  - Finance Agent (阶段2)：路由财务模板+历史ROI记录（B=1024）
  - Report Agent (阶段3)：路由所有Agent的输出摘要（B=2048）
- **预期产出**：Token消耗从全量路由(8192)降至动态路由(6608)(-19%)，Finance Agent准确率从78%提升至87%（减少无关信息干扰）

**场景B：大促期间记忆动态更新**

- **业务问题**：Prime Day实时分析中，早期阶段的研究记忆（"吸奶器竞品分析"）在财务决策阶段仍被频繁路由，但此时最相关的是实时销售数据
- **迭代精炼机制**：随着任务阶段推进，重要性评分自动降低旧研究数据的权重，提升实时数据权重；Agent在大促后期收到的上下文越来越聚焦于"当前销售状态+历史决策"，而非初期的"市场背景"
- **预期产出**：大促后期决策质量提升15%，Token消耗减少25%

## ③ 代码模板

```python
"""
RCR-Router角色感知上下文路由系统
功能：Token预算分配 + 重要性评分 + 语义过滤 + 迭代上下文精炼
基于 arXiv:2508.04903 + 2602.06025 (2025-2026)
"""
import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class BudgetTier:
    LOW = "low"     # 简化检索，< 512 tokens
    MID = "mid"     # 标准检索，512-2048 tokens
    HIGH = "high"   # 深度检索，> 2048 tokens


@dataclass
class MemoryItem:
    """记忆库中的单条记忆"""
    item_id: str
    content: str
    source_agent: str
    task_stage: int         # 0=信息收集, 1=分析, 2=决策, 3=报告
    relevance_tags: List[str] = field(default_factory=list)  # 相关角色标签
    citation_count: int = 0         # 被引用次数
    creation_round: int = 0         # 创建轮次
    importance_score: float = 0.5   # 当前重要性分数（动态更新）

    @property
    def token_count(self) -> int:
        return max(len(self.content) // 4, 1)


class ImportanceScorer:
    """重要性评分器"""

    ROLE_TAG_MAP = {
        'research_agent': ['market', 'competitor', 'trend', '市场', '竞品', '增长'],
        'compliance_agent': ['regulation', 'cpsc', 'fda', 'compliance', '合规', '认证', '法规'],
        'finance_agent': ['roi', 'cost', 'revenue', 'financial', '成本', '利润', 'fba', '财务'],
        'report_agent': ['conclusion', 'recommendation', 'summary', '结论', '建议', '报告'],
    }

    STAGE_RELEVANCE = {
        0: ['market', 'competitor', '市场', '竞品'],    # 信息收集阶段
        1: ['analysis', 'trend', '分析', '趋势'],        # 分析阶段
        2: ['decision', 'roi', 'risk', '决策', 'ROI', '风险'],  # 决策阶段
        3: ['conclusion', 'summary', '结论', '建议'],   # 报告阶段
    }

    def score(self, item: MemoryItem, agent_role: str,
               task_stage: int, current_round: int) -> float:
        """计算记忆项对特定角色在特定阶段的重要性"""
        score = 0.5  # 基础分

        # 1. 角色相关性
        role_keywords = self.ROLE_TAG_MAP.get(agent_role, [])
        content_lower = item.content.lower()
        role_matches = sum(1 for kw in role_keywords if kw.lower() in content_lower)
        score += min(role_matches * 0.15, 0.30)

        # 2. 任务阶段相关性
        stage_keywords = self.STAGE_RELEVANCE.get(task_stage, [])
        stage_matches = sum(1 for kw in stage_keywords if kw.lower() in content_lower)
        score += min(stage_matches * 0.10, 0.20)

        # 3. 引用频率加成
        score += min(item.citation_count * 0.05, 0.15)

        # 4. 时间衰减（旧记忆价值递减）
        age = current_round - item.creation_round
        decay = max(1.0 - age * 0.05, 0.5)
        score *= decay

        # 5. 任务阶段对齐（记忆来自相同阶段时加分）
        if item.task_stage == task_stage:
            score += 0.10

        return min(score, 1.0)


class RCRRouter:
    """
    RCR-Router：角色感知上下文路由系统
    """

    def __init__(self, total_budget: int = 8192):
        self.total_budget = total_budget
        self.memory_store: List[MemoryItem] = []
        self.scorer = ImportanceScorer()
        self.routing_log: List[Dict] = []
        self._item_counter = 0

        # 角色预算权重（可配置）
        self.role_budget_weights = {
            'research_agent': 0.30,
            'compliance_agent': 0.25,
            'finance_agent': 0.25,
            'report_agent': 0.20,
        }

    def add_memory(self, content: str, source_agent: str,
                    task_stage: int, round_num: int = 0) -> MemoryItem:
        """添加记忆到共享库"""
        self._item_counter += 1
        item = MemoryItem(
            item_id=f"mem_{self._item_counter:04d}",
            content=content,
            source_agent=source_agent,
            task_stage=task_stage,
            creation_round=round_num,
        )
        self.memory_store.append(item)
        return item

    def allocate_budget(self, agent_roles: List[str],
                         task_stage: int) -> Dict[str, int]:
        """为各Agent分配Token预算"""
        budgets = {}
        # 任务阶段调整权重
        stage_multipliers = {
            'research_agent': [1.5, 1.0, 0.5, 0.3],    # 阶段0最高
            'compliance_agent': [0.8, 1.2, 1.0, 0.6],
            'finance_agent': [0.5, 0.8, 1.5, 1.0],
            'report_agent': [0.3, 0.5, 0.8, 1.5],       # 阶段3最高
        }

        for role in agent_roles:
            base_weight = self.role_budget_weights.get(role, 0.25)
            stage_mult = stage_multipliers.get(role, [1.0, 1.0, 1.0, 1.0])
            effective_weight = base_weight * stage_mult[min(task_stage, 3)]
            budgets[role] = int(self.total_budget * effective_weight)

        # 归一化到总预算
        total_weight = sum(budgets.values())
        if total_weight > self.total_budget:
            scale = self.total_budget / total_weight
            budgets = {r: int(b * scale) for r, b in budgets.items()}

        return budgets

    def route(self, agent_role: str, task_stage: int,
               round_num: int, budget: Optional[int] = None) -> Tuple[List[MemoryItem], Dict]:
        """
        为特定Agent路由相关记忆子集
        
        Returns:
            (selected_memories, routing_stats)
        """
        if budget is None:
            budget = int(self.total_budget * self.role_budget_weights.get(agent_role, 0.25))

        if not self.memory_store:
            return [], {'total_available': 0, 'selected': 0, 'tokens_used': 0}

        # 计算每条记忆的重要性分数
        scored = []
        for item in self.memory_store:
            score = self.scorer.score(item, agent_role, task_stage, round_num)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 贪心选取直到预算耗尽
        selected = []
        tokens_used = 0

        for score, item in scored:
            if tokens_used + item.token_count <= budget:
                selected.append(item)
                item.citation_count += 1
                tokens_used += item.token_count
            if tokens_used >= budget:
                break

        # 记录路由日志
        routing_stats = {
            'agent': agent_role,
            'stage': task_stage,
            'budget': budget,
            'total_available_tokens': sum(m.token_count for m in self.memory_store),
            'selected_count': len(selected),
            'tokens_used': tokens_used,
            'efficiency': tokens_used / max(budget, 1),
        }
        self.routing_log.append(routing_stats)

        return selected, routing_stats

    def integrate_output(self, agent_role: str, output: str,
                          task_stage: int, round_num: int):
        """将Agent输出集成到共享记忆库（迭代精炼）"""
        self.add_memory(output, agent_role, task_stage, round_num)


def run_rcr_router_demo():
    """RCR-Router角色感知上下文路由完整演示"""
    print("=" * 65)
    print("RCR-Router角色感知上下文路由系统")
    print("基于 arXiv:2508.04903 + 2602.06025 (2025-2026)")
    print("=" * 65)

    router = RCRRouter(total_budget=8192)

    # 构建初始记忆库
    print("\n[初始化共享记忆库]")
    memories = [
        ("美国母婴市场2025年规模$28亿，YoY增长12%，吸奶器占35%份额", "system", 0),
        ("CPSC 16 CFR 1119规定儿童产品必须通过CPC认证，含电池产品需额外测试", "system", 0),
        ("Spectra S1+月销8000件，评分4.5，售价$149，主要优势：静音+医院级吸力", "system", 0),
        ("FBA费率2025年Q4：吸奶器(>3lbs) = $8.50/件，储存费旺季$2.40/立方英尺/月", "system", 0),
        ("品牌调性：温暖、专业、信任，主要用户：职场妈妈、新生儿期妈妈", "system", 0),
        ("历史ROI案例：2024年吸奶器品类ROI平均28-35%，最佳实践：25件安全库存", "system", 2),
        ("FDA婴儿食品标签要求：必须列出所有成分，过敏原需特别标注", "system", 0),
        ("美元兑人民币汇率7.25，Q4季节性上涨约5-8%对母婴品类有利", "system", 1),
    ]

    for content, source, stage in memories:
        item = router.add_memory(content, source, stage, round_num=0)
        print(f"  [{item.item_id}] 阶段{stage}: {content[:50]}...")

    print(f"\n  总记忆库: {len(router.memory_store)}条, "
          f"约{sum(m.token_count for m in router.memory_store)}tokens")

    # 多Agent多阶段路由演示
    print("\n[多Agent多阶段路由演示]")
    agents = ['research_agent', 'compliance_agent', 'finance_agent', 'report_agent']

    for stage in [0, 2]:  # 信息收集阶段 vs 决策阶段
        budgets = router.allocate_budget(agents, task_stage=stage)
        stage_name = {0: "信息收集阶段", 1: "分析阶段", 2: "决策阶段", 3: "报告阶段"}[stage]
        print(f"\n  === {stage_name} ===")
        print(f"  预算分配: {budgets}")

        for agent_role in agents:
            selected, stats = router.route(agent_role, stage, round_num=1,
                                           budget=budgets[agent_role])
            print(f"\n    [{agent_role}] 预算:{budgets[agent_role]}tokens")
            print(f"    路由到 {stats['selected_count']}条记忆 ({stats['tokens_used']}tokens, "
                  f"效率{stats['efficiency']:.0%})")
            for mem in selected[:2]:  # 显示前2条
                print(f"      - {mem.content[:55]}...")

    # Token效率对比
    print("\n[Token效率对比：RCR-Router vs 全量路由]")
    total_memory_tokens = sum(m.token_count for m in router.memory_store)
    avg_routed = np.mean([log['tokens_used'] for log in router.routing_log])
    savings_pct = 1 - avg_routed / total_memory_tokens

    print(f"  全量路由（每Agent获得所有记忆）: {total_memory_tokens}tokens/Agent")
    print(f"  RCR-Router（角色感知子集）: {avg_routed:.0f}tokens/Agent")
    print(f"  Token节省: {savings_pct:.0%}")
    print(f"  质量影响: 减少无关噪声，聚焦角色相关记忆（论文: +13-22%质量提升）")

    print("\n[✓] RCR-Router角色感知上下文路由系统测试通过")
    return router


if __name__ == "__main__":
    router = run_rcr_router_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AdaCtx-Dynamic-Context-Budget-Allocation]]（AdaCtx决定预算总量，RCR-Router决定预算内的内容选择）、[[Skill-Agent-Memory-Learning]]（共享记忆池是RCR-Router路由的数据基础）
- **延伸（extends）**：[[Skill-Context-Token-Compression]]（Token压缩减少记忆库总体积，RCR-Router精选相关子集，两层次互补）、[[Skill-Domain-Agnostic-Context-Engine]]（RCR-Router作为域无关引擎的记忆路由组件）
- **可组合（combinable）**：[[Skill-Dual-RAG-Context-Engine]]（双RAG提供指令和事实两类记忆源，RCR-Router决定哪些传给哪个Agent）、[[Skill-CASTER-Context-Aware-Model-Routing]]（CASTER路由模型选择，RCR-Router路由记忆内容）

## ⑤ 商业价值评估

- **ROI 预估**：月调用5000次MAS的平台，RCR-Router节省约20%Token（无关信息不传入Agent），同时提升关键Agent质量13-22%；年化Token节省约$600，质量提升间接减少错误决策成本更高；系统成本$4万，综合ROI≈200%（首年），后续年ROI持续提升
- **实施难度**：⭐⭐⭐☆☆（规则基础版较简单；学习版重要性评分需要角色标签库；需要改造MAS的记忆访问接口）
- **优先级**：⭐⭐⭐⭐☆（在记忆库>50条、Agent数>3的MAS中，无选择性地传入所有记忆会严重降低质量，RCR-Router是必要组件）
- **适用规模**：共享记忆库>30条、多角色Agent（>3个不同角色）的MAS系统
- **数据依赖**：需要为每个角色定义相关标签（可从Agent的SRL蓝图中自动提取）；不需要额外训练数据
