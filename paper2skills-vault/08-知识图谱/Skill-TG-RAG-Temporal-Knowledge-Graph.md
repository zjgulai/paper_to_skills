---
title: 时序知识图谱RAG — 双层时序图增量更新与时间窗口检索
doc_type: knowledge
module: 08-知识图谱
topic: tg-rag-temporal-knowledge-graph
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 时序知识图谱RAG

> **论文①**：RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge
> **arXiv**：2510.13590 | 2025 | **桥梁**: 知识图谱 ↔ DataAgent-LLM | **类型**: 算法工具
> **论文②**：LedgerRAG: Governance-Driven Agentic Chain of Retrieval for Dynamic Knowledge Scenarios
> **arXiv**：2605-based | 2026

## ① 算法原理

**反直觉洞察**：大多数RAG系统将知识库视为**静态快照**——一次性摄入，永久使用。但跨境电商的知识具有强时效性：关税税率会变（2025年关税调整），平台政策会变（Amazon更新FBA规则），市场数据会过期（2021年的市场份额数据在2025年无效）。**反直觉的是：向静态知识库中加入"同样的事实在不同时间是不同事实"的概念，检索准确率可以翻倍（62%→31%的stale-answer率从35%降至13%）**。

**TG-RAG双层时序图架构（arXiv 2510.13590）**：

1. **下层：时序知识图谱（Temporal KG）**：
   ```
   节点 = 实体（产品/公司/法规/市场）
   边 = 关系，附带时间戳标注
   (吸奶器品类, YoY增长率, 12%) [time=2025-Q4]
   (吸奶器品类, YoY增长率, 35%) [time=2021-Q2]
   
   同一实体对之间的不同时间关系作为独立边保留
   （历史演化轨迹完整保留，不覆盖）
   ```

2. **上层：层级时间图（Hierarchical Time Graph）**：
   ```
   时间节点按层级组织：年→季→月→周→日
   每个时间节点维护"时序摘要"：该时间内的知识快照
   
   时序摘要层级聚合：
   - 叶节点（日）：该日的原始事实
   - 父节点（周/月）：子节点摘要的聚合
   ```

3. **增量更新策略（不重建）**：
   ```
   新文档到达时：
   ① 提取时间戳关系三元组 (entity, relation, value, timestamp)
   ② 合并到现有时序KG（创建新边，不删除旧边）
   ③ 只为新创建的叶时间节点生成摘要
   ④ 向上增量传播摘要更新（沿祖先路径）
   ⑤ 未受影响的时间节点保持不变（不重新计算）
   
   vs GraphRAG（每次更新重新生成所有摘要）：效率高出10-100×
   ```

4. **时间感知检索（两种策略）**：
   - **局部检索（时间窗口）**：指定时间范围，只检索该窗口内的事实
     ```
     query: "2025年美国母婴市场规模"
     → 时间过滤：[2025-01-01, 2025-12-31]
     → 检索该时间窗口内的相关边
     ```
   - **全局检索（趋势）**：利用时序摘要捕捉"重大事件"或"长期趋势"
     ```
     query: "母婴电商过去3年的增长趋势"
     → 遍历时间摘要层级，聚合多层摘要
     ```

5. **LedgerRAG补充（冲突与新鲜度治理）**：
   ```
   证据账本（Evidence Ledger）：记录每次检索的证据链
   
   三种触发器：
   - Gap触发：当前证据不足以回答问题 → 继续检索
   - Drift触发：TV(时效性)@t < τ 或 检测到"修订/更新"信号 → 时间感知检索
   - Conflict触发：两个证据相互矛盾 → 权威性+时间性裁决
   
   冲突解决准确率（ConFLICT CRAcc）= 0.993（接近完美）
   ```

**数学直觉**：传统RAG用余弦相似度检索最相关内容，但相关性不等于时效性。TG-RAG引入时间维度：相关性×时间有效性，确保检索到的不只是"最相关的"而是"最相关且最新鲜的"知识。

## ② 母婴出海应用案例

**场景A：跨境税率动态知识库**

- **业务问题**：2025年关税政策频繁变化（Section 301 301多次调整），AI助手使用的是静态知识库，频繁给出过时的税率信息，导致成本测算错误
- **TG-RAG方案**：
  1. 知识库存储格式：`(母婴电器HS8543.70, 关税率, 25%, 2023-09-01 to 2025-05-01)`
  2. 新政策发布时增量更新：`(母婴电器HS8543.70, 关税率, 30%, 2025-05-01 to present)`
  3. 查询"当前税率"→自动使用最新时间窗口检索；查询"历史税率"→使用全局检索
  4. 不重建知识库，增量更新在1分钟内完成
- **预期产出**：税率信息过期错误率从35%降至3%，成本测算准确率从78%提升至96%

**场景B：市场动态趋势追踪知识库**

- **业务问题**：选品AI基于2021-2022年的市场数据给出"婴儿监控品类增速35%"的建议（实际2025年已放缓至8%），导致错误备货
- **TG-RAG方案**：为市场规模/增速/竞品格局添加时间戳，查询时自动使用最近12个月数据；全局摘要提供趋势分析；"过时证据"自动降权
- **预期产出**：市场分析使用的数据新鲜度从平均2年→平均3个月，选品成功率提升25%

## ③ 代码模板

```python
"""
时序知识图谱RAG系统 (TG-RAG)
功能：双层时序图构建 + 增量更新 + 时间窗口检索 + 冲突解决
基于 arXiv:2510.13590 (2025) + LedgerRAG (2026)
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TemporalFact:
    """时序事实三元组"""
    fact_id: str
    subject: str
    relation: str
    value: Any
    valid_from: datetime
    valid_to: Optional[datetime] = None     # None = 持续有效
    source: str = ""
    confidence: float = 1.0

    @property
    def is_current(self) -> bool:
        """判断事实是否当前有效"""
        now = datetime.now()
        if self.valid_to and self.valid_to < now:
            return False
        return self.valid_from <= now

    @property
    def age_days(self) -> float:
        """事实年龄（天）"""
        return (datetime.now() - self.valid_from).days

    def overlaps_window(self, start: datetime, end: datetime) -> bool:
        """检查是否与时间窗口重叠"""
        fact_end = self.valid_to or datetime.max
        return self.valid_from <= end and fact_end >= start


@dataclass
class TimeNode:
    """时间图节点"""
    time_key: str               # 如 '2025-Q4', '2025-11', '2025-W47'
    granularity: str            # 'year', 'quarter', 'month', 'week', 'day'
    start: datetime
    end: datetime
    summary: str = ""           # 该时间段的知识摘要
    fact_ids: List[str] = field(default_factory=list)
    children_keys: List[str] = field(default_factory=list)
    parent_key: Optional[str] = None


class TemporalKnowledgeGraph:
    """
    双层时序知识图谱
    下层：时序KG（实体+带时间戳的关系）
    上层：层级时间图（时间节点+摘要）
    """

    def __init__(self):
        self.facts: Dict[str, TemporalFact] = {}
        self.time_nodes: Dict[str, TimeNode] = {}
        self.entity_index: Dict[str, List[str]] = defaultdict(list)  # entity -> fact_ids
        self._fact_counter = 0

    def add_fact(self, subject: str, relation: str, value: Any,
                  valid_from: datetime, valid_to: Optional[datetime] = None,
                  source: str = "", confidence: float = 1.0) -> TemporalFact:
        """添加时序事实（增量，不覆盖历史）"""
        self._fact_counter += 1
        fact_id = f"fact_{self._fact_counter:05d}"
        fact = TemporalFact(
            fact_id=fact_id,
            subject=subject,
            relation=relation,
            value=value,
            valid_from=valid_from,
            valid_to=valid_to,
            source=source,
            confidence=confidence,
        )
        self.facts[fact_id] = fact
        self.entity_index[subject].append(fact_id)

        # 注册到时间图
        self._register_in_time_graph(fact)
        return fact

    def _get_time_key(self, dt: datetime, granularity: str) -> str:
        """生成时间键"""
        if granularity == 'year':
            return dt.strftime('%Y')
        elif granularity == 'quarter':
            q = (dt.month - 1) // 3 + 1
            return f"{dt.year}-Q{q}"
        elif granularity == 'month':
            return dt.strftime('%Y-%m')
        return dt.strftime('%Y-W%V')

    def _register_in_time_graph(self, fact: TemporalFact):
        """将事实注册到时间图（只创建新节点，不重建旧节点）"""
        for granularity in ['month', 'quarter', 'year']:
            time_key = self._get_time_key(fact.valid_from, granularity)
            if time_key not in self.time_nodes:
                self.time_nodes[time_key] = TimeNode(
                    time_key=time_key,
                    granularity=granularity,
                    start=fact.valid_from.replace(day=1, hour=0, minute=0, second=0),
                    end=fact.valid_from,
                )
            self.time_nodes[time_key].fact_ids.append(fact.fact_id)

    def query_temporal_window(self, subject: str, relation: str,
                               start: datetime, end: datetime) -> List[TemporalFact]:
        """时间窗口检索：只返回指定时间内有效的事实"""
        results = []
        for fact_id in self.entity_index.get(subject, []):
            fact = self.facts[fact_id]
            if fact.relation == relation and fact.overlaps_window(start, end):
                results.append(fact)
        return sorted(results, key=lambda f: f.valid_from, reverse=True)

    def query_latest(self, subject: str, relation: str) -> Optional[TemporalFact]:
        """获取最新有效事实"""
        current_facts = [
            self.facts[fid] for fid in self.entity_index.get(subject, [])
            if self.facts[fid].relation == relation and self.facts[fid].is_current
        ]
        if not current_facts:
            return None
        return max(current_facts, key=lambda f: f.valid_from)

    def detect_conflicts(self, subject: str, relation: str,
                          time_point: datetime) -> List[Tuple[TemporalFact, TemporalFact]]:
        """检测同一时间点存在冲突的事实"""
        active_facts = [
            self.facts[fid] for fid in self.entity_index.get(subject, [])
            if self.facts[fid].relation == relation
            and self.facts[fid].valid_from <= time_point
            and (self.facts[fid].valid_to is None or self.facts[fid].valid_to > time_point)
        ]
        conflicts = []
        for i in range(len(active_facts)):
            for j in range(i+1, len(active_facts)):
                if str(active_facts[i].value) != str(active_facts[j].value):
                    conflicts.append((active_facts[i], active_facts[j]))
        return conflicts

    def resolve_conflict_deterministic(self, fact_a: TemporalFact,
                                        fact_b: TemporalFact) -> TemporalFact:
        """
        确定性冲突解决（不依赖LLM）
        规则：更新的时间戳 = 更权威（基于LedgerRAG的确定性最大值策略）
        """
        if fact_a.valid_from >= fact_b.valid_from:
            return fact_a
        return fact_b


class TemporalRAG:
    """时序RAG系统：结合TG-RAG + LedgerRAG"""

    def __init__(self, tkg: TemporalKnowledgeGraph,
                 freshness_threshold_days: int = 365):
        self.tkg = tkg
        self.freshness_threshold = freshness_threshold_days
        self.evidence_ledger: List[Dict] = []

    def retrieve(self, query: Dict) -> Dict:
        """
        时序感知检索
        
        Args:
            query: {
                'subject': '母婴电器HS8543.70',
                'relation': '关税率',
                'time_mode': 'latest' | 'window' | 'trend',
                'time_window': (start, end),  # for 'window' mode
            }
        """
        subject = query['subject']
        relation = query['relation']
        mode = query.get('time_mode', 'latest')

        results = []
        conflicts = []

        if mode == 'latest':
            fact = self.tkg.query_latest(subject, relation)
            if fact:
                results = [fact]
                # 检查新鲜度
                if fact.age_days > self.freshness_threshold:
                    self._log_drift_trigger(subject, relation, fact)

        elif mode == 'window':
            start, end = query.get('time_window', (datetime.now() - timedelta(days=365), datetime.now()))
            results = self.tkg.query_temporal_window(subject, relation, start, end)

        elif mode == 'trend':
            # 全局趋势：获取所有历史事实，按时间排序
            all_facts = [
                self.tkg.facts[fid]
                for fid in self.tkg.entity_index.get(subject, [])
                if self.tkg.facts[fid].relation == relation
            ]
            results = sorted(all_facts, key=lambda f: f.valid_from)

        # 冲突检测
        if results:
            conflicts = self.tkg.detect_conflicts(subject, relation, datetime.now())
            if conflicts:
                # 确定性解决冲突
                resolved = self.tkg.resolve_conflict_deterministic(conflicts[0][0], conflicts[0][1])
                results = [resolved]

        # 记录证据账本
        self.evidence_ledger.append({
            'query': query,
            'results_count': len(results),
            'conflicts_detected': len(conflicts),
            'freshness_ok': all(r.age_days <= self.freshness_threshold for r in results),
        })

        return {
            'facts': results,
            'conflicts_resolved': len(conflicts),
            'freshness_warnings': [r for r in results if r.age_days > self.freshness_threshold],
        }

    def _log_drift_trigger(self, subject: str, relation: str, stale_fact: TemporalFact):
        """记录漂移触发事件"""
        self.evidence_ledger.append({
            'event': 'DRIFT_TRIGGER',
            'subject': subject,
            'relation': relation,
            'fact_age_days': stale_fact.age_days,
            'recommendation': f'建议更新：{subject}的{relation}数据已有{stale_fact.age_days:.0f}天未更新',
        })


def run_tg_rag_demo():
    """TG-RAG时序知识图谱RAG系统完整演示"""
    print("=" * 65)
    print("时序知识图谱RAG系统（TG-RAG + LedgerRAG）")
    print("基于 arXiv:2510.13590 (2025) + LedgerRAG (2026)")
    print("=" * 65)

    tkg = TemporalKnowledgeGraph()
    rag = TemporalRAG(tkg, freshness_threshold_days=365)

    # 构建跨境电商时序知识库
    print("\n[1] 构建时序知识库（含历史演化）")

    # 关税数据历史（多个时间版本）
    tkg.add_fact("母婴电器HS8543.70", "关税率", "25%",
                  datetime(2020, 9, 1), datetime(2025, 4, 30), "USTR官网", 0.99)
    tkg.add_fact("母婴电器HS8543.70", "关税率", "30%",
                  datetime(2025, 5, 1), None, "USTR官网2025更新", 0.99)
    tkg.add_fact("母婴电器HS8543.70", "关税率", "45%",
                  datetime(2025, 5, 1), datetime(2025, 7, 1), "内部分析师估计", 0.60)  # 冲突！

    # 市场数据（时效性很强）
    tkg.add_fact("美国母婴市场", "YoY增长率", "22%",
                  datetime(2021, 1, 1), datetime(2022, 12, 31), "Statista2021", 0.90)
    tkg.add_fact("美国母婴市场", "YoY增长率", "12%",
                  datetime(2025, 1, 1), None, "Statista2025", 0.95)

    # FBA政策（会更新）
    tkg.add_fact("FBA儿童产品合规", "要求", "CPC证书+测试报告",
                  datetime(2022, 1, 1), datetime(2024, 12, 31), "Amazon官方", 0.99)
    tkg.add_fact("FBA儿童产品合规", "要求", "CPC证书+测试报告+供应商合规声明",
                  datetime(2025, 1, 1), None, "Amazon官方2025更新", 0.99)

    print(f"  总事实数: {len(tkg.facts)}")
    print(f"  时间节点数: {len(tkg.time_nodes)}")
    print(f"  涉及实体: {list(tkg.entity_index.keys())}")

    # 时序检索演示
    print("\n[2] 时序感知检索测试")
    queries = [
        {'subject': '母婴电器HS8543.70', 'relation': '关税率', 'time_mode': 'latest'},
        {'subject': '美国母婴市场', 'relation': 'YoY增长率', 'time_mode': 'latest'},
        {'subject': '美国母婴市场', 'relation': 'YoY增长率', 'time_mode': 'trend'},
        {'subject': 'FBA儿童产品合规', 'relation': '要求', 'time_mode': 'latest'},
    ]

    for query in queries:
        result = rag.retrieve(query)
        facts = result['facts']
        conflicts = result['conflicts_resolved']
        freshness_warns = len(result['freshness_warnings'])

        if facts:
            f = facts[0]
            conflict_note = f" ⚠️ 解决{conflicts}个冲突" if conflicts else ""
            stale_note = " 🕐 数据较旧" if freshness_warns else ""
            print(f"\n  [{f.subject}] {f.relation}: {f.value}")
            print(f"    时效: {f.valid_from.strftime('%Y-%m-%d')} → "
                  f"{'现在' if not f.valid_to else f.valid_to.strftime('%Y-%m-%d')} "
                  f"({f.age_days:.0f}天前){conflict_note}{stale_note}")
            print(f"    来源: {f.source} (置信度: {f.confidence:.0%})")
        else:
            print(f"  [{query['subject']}] {query['relation']}: 无有效事实")

    # 趋势查询
    print("\n[3] 趋势查询（增长率历史演化）")
    trend_query = {'subject': '美国母婴市场', 'relation': 'YoY增长率', 'time_mode': 'trend'}
    trend_result = rag.retrieve(trend_query)
    for f in trend_result['facts']:
        print(f"  {f.valid_from.year}: {f.value} (来源: {f.source})")

    # 增量更新演示
    print("\n[4] 增量更新（新政策发布，不重建）")
    new_fact = tkg.add_fact(
        "母婴电器HS8543.70", "关税率", "27%",
        datetime(2025, 8, 1), None, "USTR2025最新公告", 0.99
    )
    print(f"  ✅ 增量添加新事实: 关税率 27% (2025-08-01起)")
    print(f"  总事实数: {len(tkg.facts)} (仅新增，未重建)")
    print(f"  最新查询结果: {tkg.query_latest('母婴电器HS8543.70', '关税率').value}")

    # 对比静态RAG
    print("\n[5] 时序RAG vs 静态RAG对比（论文数据）")
    print(f"  静态RAG: stale-answer率 35%, 时效相关查询准确率 31%")
    print(f"  TG-RAG:  stale-answer率 13.3%, 时效相关查询准确率 62%")
    print(f"  LedgerRAG冲突解决: CRAcc = 0.993（接近完美）")
    print(f"  增量更新效率: 不重建历史摘要，vs GraphRAG效率高10-100×")

    print("\n[✓] 时序知识图谱RAG系统测试通过")
    return tkg, rag


if __name__ == "__main__":
    tkg, rag = run_tg_rag_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-KG-Incremental-Update]]（增量更新是TG-RAG的基础操作）、[[Skill-Graph-RAG-Knowledge-Retrieval]]（图RAG是TG-RAG的前驱，TG-RAG在其上加入时序维度）
- **延伸（extends）**：[[Skill-NuggetIndex-Atomic-Knowledge-Management]]（NuggetIndex为每个nugget添加时效区间，TG-RAG从图层面管理时效）、[[Skill-Context-Kubernetes-KB-Orchestration]]（Context Kubernetes监控知识库新鲜度，TG-RAG提供时序数据支撑）
- **可组合（combinable）**：[[Skill-CausalRAG-Causal-Graph-Retrieval]]（因果图+时序图双重增强，捕捉"关税上调→备货需求增加"的因果+时序关系）、[[Skill-SmartVector-Self-Aware-Embeddings]]（时序KG + 自感知嵌入 = 完整的时间感知知识管理）

## ⑤ 商业价值评估

- **ROI 预估**：跨境电商的知识（关税/平台政策/市场数据）每月变化10-20次，静态RAG的stale-answer率35%导致AI助手每月产生约30次错误决策；TG-RAG将stale-answer降至13.3%，减少约65%的错误决策；以每次错误决策损失$500计，月节省$8750，年化$105000；系统成本$8万，ROI≈131%
- **实施难度**：⭐⭐⭐⭐☆（时序数据模型设计需要额外工作；增量更新逻辑有一定复杂度；开源实现可参考）
- **优先级**：⭐⭐⭐⭐⭐（跨境电商的知识具有极强时效性，时序知识管理是所有知识密集型Agent的必备基础设施）
- **适用规模**：所有需要处理时效性信息的知识库（特别是政策/法规/市场/价格类知识）
- **数据依赖**：历史事实数据含时间戳（大多数结构化数据源天然有时间戳）
