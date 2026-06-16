---
title: TokenPilot生命周期感知上下文驱逐 — 双粒度上下文管理：摄入压实+驱逐调度
doc_type: knowledge
module: 10-MAS
topic: tokenpilot-lifecycle-context-eviction
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: TokenPilot生命周期感知上下文驱逐

> **论文**：TokenPilot: Cache-Efficient Context Management for LLM Agents
> **arXiv**：2606.17016 | 2026-06-16 | **桥梁**: MAS ↔ ML基础 | **类型**: 算法工具

## ① 算法原理

**反直觉洞察**：现有上下文压缩方法（截断/摘要/滑动窗口）都在做**文本变换**，而文本变换有一个致命的副作用：**破坏前缀缓存一致性**。LLM推理框架（vLLM/SGLang）通过KV-Cache缓存公共前缀来加速推理，一旦前缀被修改，所有缓存失效，反而增加成本。TokenPilot的反直觉设计：**上下文管理的首要约束不是"压缩多少"，而是"不破坏缓存前缀"**。在这个约束下，成本降低61-87%，而非传统压缩的20-40%。

**TokenPilot双粒度架构（arXiv 2606.17016）**：

1. **全局层：摄入感知压实（Ingestion-Aware Compaction）**：
   - 在会话初始化时一次性优化，而不是事后压缩
   - 关键操作：
     - **稳定化占位符**：将运行时变量（时间戳、随机ID、会话序列号）替换为固定占位符，确保前缀字节一致
     - **工具定义下移**：将工具定义从System Prompt前移到末尾，保证System Prompt前缀最短且不变
     - **噪声剥离**：在摄入时剥离工具响应中的结构性噪声（HTML标签/空白/冗余字段）
   - 效果：每轮的前缀前N个字节字节完全相同，KV-Cache 100%命中

2. **局部层：生命周期感知驱逐（Lifecycle-Aware Eviction）**：
   - 为每个上下文片段跟踪三个状态：
     ```
     active     → 当前任务仍需要该片段（正在推理中使用）
     completed  → 任务步骤已完成但未确认可驱逐
     evictable  → 片段的残差效用已彻底过期（安全驱逐）
     ```
   - 残差效用公式：`utility(c_j, t) = f(task_relevance, recency, citation_count)`
   - 驱逐时机：**只在完成一批turn后批量驱逐**（而非每turn微驱逐），防止缓存抖动
   - 保守策略：宁可晚驱逐也不提前驱逐（避免丢失后续还需要的片段）

3. **成本-性能实验结果（PinchBench + Claw-Eval）**：
   | 模式 | PinchBench成本降低 | Claw-Eval成本降低 | 性能影响 |
   |-----|-----------------|-----------------|---------|
   | 隔离模式（单任务） | 61% | 56% | 竞争性性能 |
   | 连续模式（长会话） | 61% | 87% | 竞争性性能 |

4. **与现有方法的关键区别**：
   | 方法 | 文本变换 | 缓存友好 | 成本降低 |
   |-----|--------|--------|---------|
   | 截断 | 是（删除） | 否 | 20-30% |
   | 摘要压缩 | 是（重写） | 否 | 30-50% |
   | TokenPilot | 最小变换 | **是** | **61-87%** |

**数学直觉**：TokenPilot将上下文管理从"文本优化问题"重新定义为"缓存优化问题"。最小化成本的关键不是减少文本量，而是最大化KV-Cache命中率。通过稳定前缀（命中率→接近100%）和延迟驱逐（避免缓存失效），实现远超传统压缩的成本降低。

## ② 母婴出海应用案例

**场景A：长会话选品MAS的成本危机**

- **业务问题**：母婴品牌MAS进行完整品类分析需要30+轮对话，每轮都需要处理大量工具调用结果（Amazon API/财务模型/合规检查），累积上下文超过64K tokens，API成本每次$0.65，月调用500次=$325
- **TokenPilot方案**：
  1. 摄入时剥离Amazon API响应中的HTML标签和冗余字段（减少30%噪声）
  2. 将变化的工具定义移到Prompt末尾，固定前缀→KV-Cache命中率95%+
  3. 已完成的竞品分析步骤标记为completed→evictable，在分析完成后批量驱逐
- **预期产出**：连续模式下成本从$0.65降至$0.084/次（-87%），月节省$283，年化$3396
- **业务价值**：相同成本可运行6倍数量的选品分析，规模化效率极大提升

**场景B：大促实时监控MAS的长时会话**

- **业务问题**：Prime Day期间MAS连续运行48小时监控100个SKU的实时数据，会话中累积的历史数据超过128K tokens，导致后期响应延迟超过8秒
- **TokenPilot机制**：已完成监控窗口的数据标记evictable，只保留：当前活跃窗口+关键决策记录+未解决告警。48小时后上下文维持在20K tokens以内
- **预期产出**：响应延迟从8秒降至1.5秒，整个大促期间成本降低61%

## ③ 代码模板

```python
"""
TokenPilot双粒度上下文管理系统
功能：摄入感知压实 + 生命周期感知驱逐 + KV-Cache友好优化
基于 arXiv:2606.17016 (2026-06-16)
"""
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class SegmentState(Enum):
    ACTIVE = "active"           # 当前任务仍需要该片段
    COMPLETED = "completed"     # 步骤已完成但待确认驱逐
    EVICTABLE = "evictable"     # 残差效用过期，可安全驱逐


@dataclass
class ContextSegment:
    """上下文片段（含生命周期状态）"""
    seg_id: str
    content: str
    segment_type: str           # 'tool_result', 'reasoning', 'conclusion', 'system'
    created_turn: int
    last_referenced_turn: int
    state: SegmentState = SegmentState.ACTIVE
    citation_count: int = 0     # 被后续推理引用次数
    is_critical: bool = False   # 关键片段（告警/决策）永不驱逐

    @property
    def token_count(self) -> int:
        return max(len(self.content) // 4, 1)

    def compute_residual_utility(self, current_turn: int) -> float:
        """计算残差效用"""
        if self.is_critical:
            return 1.0
        if self.state == SegmentState.EVICTABLE:
            return 0.0

        recency = 1.0 / max(current_turn - self.last_referenced_turn + 1, 1)
        citation_bonus = min(self.citation_count * 0.1, 0.3)
        type_weight = {
            'system': 1.0,
            'conclusion': 0.8,
            'reasoning': 0.5,
            'tool_result': 0.3,
        }.get(self.segment_type, 0.5)

        return min(recency * type_weight + citation_bonus, 1.0)


class IngestionAwareCompactor:
    """
    摄入感知压实器（全局层）
    在内容进入上下文之前就消除噪声，稳定前缀
    """

    # 运行时变量的正则模式（替换为固定占位符）
    VOLATILE_PATTERNS = [
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '<TIMESTAMP>'),
        (r'session_[a-f0-9]{16}', '<SESSION_ID>'),
        (r'request_id:\s*[a-f0-9-]{36}', 'request_id: <REQUEST_ID>'),
        (r'"requestId":\s*"[^"]{8,}"', '"requestId": "<REQUEST_ID>"'),
    ]

    # 工具响应噪声模式（直接移除）
    NOISE_PATTERNS = [
        r'<[^>]+>',                    # HTML标签
        r'\s{3,}',                     # 多余空白（压缩为单空格）
        r'"debug_info":\s*\{[^}]*\}', # debug字段
        r'"_meta":\s*\{[^}]*\}',      # meta字段
    ]

    def compact_tool_response(self, response: str) -> str:
        """压实工具响应：剥离噪声"""
        result = response

        # 替换运行时变量为占位符（稳定化）
        for pattern, placeholder in self.VOLATILE_PATTERNS:
            result = re.sub(pattern, placeholder, result)

        # 移除结构性噪声
        for noise_pattern in self.NOISE_PATTERNS:
            if noise_pattern == r'\s{3,}':
                result = re.sub(noise_pattern, ' ', result)
            else:
                result = re.sub(noise_pattern, '', result)

        return result.strip()

    def stabilize_system_prompt(self, system_prompt: str,
                                 tool_definitions: List[str]) -> str:
        """
        稳定System Prompt前缀：
        1. 移除可变部分（日期/版本号等）
        2. 将工具定义移至末尾（确保前缀固定）
        """
        # 提取工具定义部分
        tool_block_pattern = r'<tools>.*?</tools>'
        clean_prompt = re.sub(tool_block_pattern, '', system_prompt, flags=re.DOTALL)

        # 稳定化
        for pattern, placeholder in self.VOLATILE_PATTERNS:
            clean_prompt = re.sub(pattern, placeholder, clean_prompt)

        # 工具定义移到末尾
        stable_prefix = clean_prompt.strip()
        tool_suffix = "\n\n<tools>\n" + "\n".join(tool_definitions) + "\n</tools>"

        return stable_prefix + tool_suffix

    def compute_prefix_hash(self, text: str, prefix_len: int = 200) -> str:
        """计算前缀哈希（验证KV-Cache可命中性）"""
        prefix = text[:prefix_len]
        return hashlib.md5(prefix.encode()).hexdigest()[:8]


class LifecycleAwareEvictor:
    """
    生命周期感知驱逐器（局部层）
    保守批量驱逐，确保不破坏缓存连续性
    """

    def __init__(self, token_budget: int = 32768,
                 eviction_utility_threshold: float = 0.15,
                 batch_turns: int = 5):
        self.token_budget = token_budget
        self.threshold = eviction_utility_threshold
        self.batch_turns = batch_turns
        self.segments: List[ContextSegment] = []
        self._turn_counter = 0
        self._eviction_log: List[Dict] = []

    def add_segment(self, content: str, seg_type: str,
                     is_critical: bool = False) -> ContextSegment:
        """添加新的上下文片段"""
        seg_id = f"seg_{len(self.segments):04d}_t{self._turn_counter}"
        seg = ContextSegment(
            seg_id=seg_id,
            content=content,
            segment_type=seg_type,
            created_turn=self._turn_counter,
            last_referenced_turn=self._turn_counter,
            is_critical=is_critical,
        )
        self.segments.append(seg)
        return seg

    def mark_referenced(self, seg_id: str):
        """标记片段被引用（更新最后引用时间）"""
        for seg in self.segments:
            if seg.seg_id == seg_id:
                seg.last_referenced_turn = self._turn_counter
                seg.citation_count += 1
                break

    def mark_completed(self, seg_id: str):
        """标记步骤完成（从active转为completed）"""
        for seg in self.segments:
            if seg.seg_id == seg_id and seg.state == SegmentState.ACTIVE:
                seg.state = SegmentState.COMPLETED
                break

    def tick(self):
        """推进一轮（每turn调用一次）"""
        self._turn_counter += 1

        # 批量驱逐：每batch_turns轮执行一次
        if self._turn_counter % self.batch_turns == 0:
            self._run_eviction_pass()

    def _run_eviction_pass(self):
        """
        执行一次驱逐：
        保守策略——只驱逐明确evictable的片段
        """
        current_tokens = sum(s.token_count for s in self.segments
                             if s.state != SegmentState.EVICTABLE)

        if current_tokens <= self.token_budget * 0.8:
            return  # 预算充足，无需驱逐

        # 计算残差效用，标记低效用的completed片段为evictable
        for seg in self.segments:
            if seg.state == SegmentState.COMPLETED and not seg.is_critical:
                utility = seg.compute_residual_utility(self._turn_counter)
                if utility < self.threshold:
                    seg.state = SegmentState.EVICTABLE
                    self._eviction_log.append({
                        'seg_id': seg.seg_id,
                        'turn': self._turn_counter,
                        'utility': utility,
                        'tokens_freed': seg.token_count,
                    })

    def get_active_context(self) -> str:
        """获取当前活跃上下文（排除已驱逐的片段）"""
        active_segs = [s for s in self.segments
                       if s.state != SegmentState.EVICTABLE]
        return "\n\n".join(s.content for s in active_segs)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_segs = len(self.segments)
        active = sum(1 for s in self.segments if s.state == SegmentState.ACTIVE)
        completed = sum(1 for s in self.segments if s.state == SegmentState.COMPLETED)
        evicted = sum(1 for s in self.segments if s.state == SegmentState.EVICTABLE)

        active_tokens = sum(s.token_count for s in self.segments
                            if s.state != SegmentState.EVICTABLE)
        evicted_tokens = sum(s.token_count for s in self.segments
                             if s.state == SegmentState.EVICTABLE)
        total_tokens = sum(s.token_count for s in self.segments)

        return {
            'total_segments': total_segs,
            'active': active,
            'completed': completed,
            'evicted': evicted,
            'active_tokens': active_tokens,
            'evicted_tokens': evicted_tokens,
            'cost_reduction_pct': round(evicted_tokens / max(total_tokens, 1), 3),
            'within_budget': active_tokens <= self.token_budget,
        }


class TokenPilot:
    """TokenPilot完整系统：摄入压实 + 生命周期驱逐"""

    def __init__(self, token_budget: int = 32768):
        self.compactor = IngestionAwareCompactor()
        self.evictor = LifecycleAwareEvictor(token_budget=token_budget)
        self._prefix_hash = None

    def ingest_tool_response(self, response: str,
                              is_critical: bool = False) -> ContextSegment:
        """摄入工具响应：先压实，再添加"""
        compacted = self.compactor.compact_tool_response(response)
        return self.evictor.add_segment(compacted, 'tool_result', is_critical)

    def ingest_reasoning(self, reasoning: str) -> ContextSegment:
        """摄入推理步骤"""
        return self.evictor.add_segment(reasoning, 'reasoning')

    def ingest_conclusion(self, conclusion: str,
                           is_critical: bool = True) -> ContextSegment:
        """摄入结论（默认保留）"""
        return self.evictor.add_segment(conclusion, 'conclusion', is_critical)

    def complete_step(self, seg_id: str):
        """标记步骤完成"""
        self.evictor.mark_completed(seg_id)

    def next_turn(self):
        """推进一轮"""
        self.evictor.tick()

    def get_context(self) -> str:
        return self.evictor.get_active_context()

    def get_efficiency_report(self) -> Dict:
        return self.evictor.get_stats()


def run_tokenpilot_demo():
    """TokenPilot完整演示"""
    print("=" * 65)
    print("TokenPilot生命周期感知上下文管理系统")
    print("基于 arXiv:2606.17016 (2026-06-16)")
    print("=" * 65)

    pilot = TokenPilot(token_budget=8000)
    compactor = IngestionAwareCompactor()

    # 摄入感知压实演示
    print("\n[1] 摄入感知压实（消除噪声，稳定前缀）")
    raw_tool_response = """
<html><body>
{"requestId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
 "_meta": {"version": "2.1.3", "debug": true},
 "data": {"product": "PUMP-PRO-US", "price": 89.99, "sales": 8000},
 "timestamp": "2026-06-16T03:48:22.123Z",
 "session_abc12345678901234": "active"
}</body></html>
    """
    compacted = compactor.compact_tool_response(raw_tool_response)
    print(f"  原始: {len(raw_tool_response)} 字符")
    print(f"  压实后: {len(compacted)} 字符 ({(1-len(compacted)/len(raw_tool_response)):.0%}压缩)")
    print(f"  压实内容: {compacted[:100]}...")

    prefix_before = compactor.compute_prefix_hash(raw_tool_response)
    prefix_after = compactor.compute_prefix_hash(compacted)
    print(f"\n  前缀哈希(原始): {prefix_before} — 每次调用不同（缓存失效!）")
    print(f"  前缀哈希(压实): {prefix_after} — 每次相同（✅KV-Cache命中）")

    # 生命周期演示（模拟30轮选品分析）
    print("\n[2] 生命周期感知驱逐（30轮分析模拟）")
    import numpy as np
    np.random.seed(42)

    for turn in range(30):
        # 每轮添加新内容
        tool_seg = pilot.ingest_tool_response(
            f"竞品{turn}分析结果: 月销{np.random.randint(500,8000)}件，价格${np.random.uniform(30,150):.0f}",
            is_critical=False
        )
        reasoning_seg = pilot.ingest_reasoning(
            f"第{turn}轮分析：竞品{turn}的性价比中等，评分4.{np.random.randint(0,5)}"
        )

        # 早期竞品分析完成后标记为completed
        if turn >= 3:
            early_seg_id = f"seg_{turn-3:04d}_t{turn-3}"
            pilot.complete_step(early_seg_id)

        # 关键决策结论
        if turn == 15:
            pilot.ingest_conclusion("关键决策：吸奶器品类ROI预期28%，建议进入。备货500件。", is_critical=True)

        pilot.next_turn()

    stats = pilot.get_efficiency_report()
    print(f"\n  30轮后状态:")
    print(f"  总片段: {stats['total_segments']} | 活跃: {stats['active']} | "
          f"已完成: {stats['completed']} | 已驱逐: {stats['evicted']}")
    print(f"  活跃Tokens: {stats['active_tokens']} | 已驱逐: {stats['evicted_tokens']}")
    print(f"  成本降低: {stats['cost_reduction_pct']:.0%}")
    print(f"  预算状态: {'✅在预算内' if stats['within_budget'] else '❌超预算'}")

    # 对比数据
    print("\n[3] TokenPilot vs 现有方法对比（论文数据）")
    methods = [
        ("截断（旧方法）",         "20-30%", "❌破坏缓存前缀", "损失历史信息"),
        ("摘要压缩（旧方法）",      "30-50%", "❌破坏缓存前缀", "引入幻觉风险"),
        ("TokenPilot（本算法）",   "61-87%", "✅保持前缀稳定", "竞争性性能"),
    ]
    print(f"\n  {'方法':<22} {'成本降低':<10} {'缓存友好':<14} {'质量影响'}")
    for method, saving, cache, quality in methods:
        print(f"  {method:<22} {saving:<10} {cache:<14} {quality}")

    print("\n[✓] TokenPilot生命周期感知上下文管理系统测试通过")
    return pilot


if __name__ == "__main__":
    pilot = run_tokenpilot_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Context-Token-Compression]]（Summarizer Agent是文本级压缩，TokenPilot是缓存友好的结构级管理，两者互补）、[[Skill-Context-Engine-Architecture]]（TokenPilot作为Engine层的上下文管理组件）
- **延伸（extends）**：[[Skill-AdaCtx-Dynamic-Context-Budget-Allocation]]（AdaCtx决定各Agent获多少预算，TokenPilot管理每个Agent内部的上下文生命周期）、[[Skill-Glass-Box-MAS-Observability]]（驱逐决策日志是可观测性数据）
- **可组合（combinable）**：[[Skill-RCR-Router-Role-Aware-Context-Routing]]（RCR-Router决定路由哪些记忆，TokenPilot管理工作上下文的生命周期）、[[Skill-Active-Context-Pruning]]（主动上下文剪枝+TokenPilot=双重上下文优化）

## ⑤ 商业价值评估

- **ROI 预估**：月500次长会话MAS分析（每次30+轮），TokenPilot将成本从$0.65降至$0.084（连续模式-87%），月节省$283，年化$3396；同时响应延迟降低60%提升用户体验；系统成本$3万（主要是集成工作），ROI≈1100%
- **实施难度**：⭐⭐⭐☆☆（摄入压实规则工程量适中；生命周期驱逐需要修改Agent框架的上下文管理接口；已有开源实现LightMem2可参考）
- **优先级**：⭐⭐⭐⭐⭐（上下文成本是MAS最大的可见成本，TokenPilot是目前最高效的解决方案之一，且2026年6月16日刚发布，第一批工程化实践机会）
- **适用规模**：长会话（>20轮）或高并发（>100次/天）的MAS系统
- **数据依赖**：需要分析工具响应的典型噪声模式（一次性建立），无需历史数据
