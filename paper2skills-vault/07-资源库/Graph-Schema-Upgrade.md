---
name: graph-schema-upgrade-v2
description: 技能图谱边权重 (Edge Weight) Schema 升维设计文档。引入 domain_relevance（跨域借用度）和 biz_confidence（商业验证置信度），将学术静态关联升级为实盘商业动态关联。
---

# 图谱升维设计：结构化非共识图网络升级方案

> **状态**：v2.0 Schema 设计
> **日期**：2026-06-20
> **目标**：解决图谱庞大但噪音多、权重无差异的问题，打造可对抗衰减的「母婴行业实战网络」。

## 一、为什么需要升维？ (The Hallucination Pruning)
当前 `paper2skills` 图谱拥有 13893 条边，但所有的边目前只是一维的静态 `weight=1.0`。
这导致：学术界认为相关性极强的两个领域，在母婴出海实战中可能毫无价值；而学术界毫无关联的两个点（如流行病学与流量运营），在真实世界却是高潜力的“跨界套利（Alpha）”。

**核心解法**：通过 `domain_relevance` 和 `biz_confidence` 两个新维度，对边进行「实盘赋权与噪音修建」。

## 二、新 Schema 设计：SkillEdge v2.0

在 `skills_graph_analyzer.py` 的 `SkillEdge` 类中引入两个浮点型字段：

```python
@dataclass
class SkillEdge:
    source: str
    target: str
    edge_type: str  # 'prerequisite', 'extension', 'combinable'
    weight: float = 1.0
    domain_relevance: float = 1.0  # 跨域关联度
    biz_confidence: float = 1.0    # 商业置信度
```

### 1. 跨域关联度 `domain_relevance` (寻找非共识)
- **定义**：如果 $Source$ 和 $Target$ 属于完全不同的底层 Domain（例如 04-供应链 和 10-MAS，或者 17-价格优化 和 01-因果推断），且被明确标记为 `combinable`，系统将给予惩罚或奖励。
- **提取逻辑**：
  - 默认跨域边基础衰减：`0.8`（大部分硬凑的跨域其实没有意义）
  - 如果匹配了 `BRIDGE_FOCUS_PAIRS`（强协同跨域白名单），权重提升至 `1.5`。
  - **终极加成**：如果在 `MasterPrompt.md` 中被识别出具有“降维打击”或“反直觉（Counter-intuitive）”特征，手动配置权重可达 `2.5 - 3.0`。

### 2. 商业验证置信度 `biz_confidence` (对抗学术幻觉)
- **定义**：将图谱与真实业务解绑。仅仅有学术论文支撑是不够的，必须在真实系统中运行并闭环。
- **提取逻辑 (RAG / DAG 反馈机制)**：
  - Base: `1.0`
  - 如果该条边（或相关的 Skill）出现在 `playbook/solutions/` 的顶级架构方案中（例如被 `sol-counterfactual-pricing` 引用）：置信度 `+1.0`。
  - 如果存在真实的 `skill_ps_override.yaml` 落地覆盖记录（证明它解决了业务人员的特定痛点）：置信度 `+0.5`。
  - 如果有大范围退货或负面 ROI 标签，触发红队审计：置信度 `-1.0`。

## 三、提取与计算逻辑实现路线

1. **图谱解析器升级**：在 `skills_graph_analyzer.py` 中，计算 Edge 时，动态检索 `solutions/*.md` 和 `skill_ps_override.yaml` 的命中情况。
2. **权重更新矩阵**：
   $$ W_{final} = W_{base} \times domain\_relevance \times biz\_confidence $$
3. **前端渲染更新**：当 $W_{final} > 2.0$ 时，在 `graph/overview.html` 中高亮显示（例如红色加粗线），表示这是“高价值金边”；如果 $W_{final} < 0.5$，将其弱化透明度（噪音修建）。

## 四、商业价值
这套升维方案把学术 RAG 升级成了**反馈闭环的 RLHF 网络**。老板看到的不再是一团乱麻的关系网，而是一张清晰的「母婴电商能力雷达图」，告诉他“哪里是别人没想到的高价值洼地（红边），哪里是鸡肋（灰边）”。
