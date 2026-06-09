---
title: MAA 多 Agent 行动建议 - 从评论到产品改进决策链
doc_type: knowledge
module: 14-用户分析
topic: review-to-action-multi-agent
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
paper: arXiv:2601.12024
roadmap_phase: phase2
---

# Skill: MAA — 多 Agent 评论到行动建议(规范性决策链路)

> 论文:**A Multi-Agent System for Generating Actionable Business Advice** (Bhandari et al., 2026-01) · arXiv:2601.12024
> 5 Agent 协作: Clustering → Issue → Recommendation → SRAC Evaluation → Ranking
> 实验:Yelp 三领域(汽车/餐饮/酒店),actionability/specificity 持续超越单模型

---

## ① 算法原理

### 核心思想

传统评论分析停留在"描述性"层面(情感/属性). MAA 将其升级为"**规范性决策链路**":评论 → 问题 → 建议 → 评估 → 排序. 通过 **5 个 Agent 分工**,把大规模评论语料蒸馏成企业可**直接执行的行动清单**,而非分析报告.

### 数学直觉

**Step 1 - Clustering Agent**(代表评论选择):
$$r^*_k = \arg\max_{r \in C_k} \cos(\mathbf{x}_r, \mathbf{c}_k)$$
对 K-Means 聚类后每个簇 $C_k$,选离质心 $\mathbf{c}_k$ 最近的评论作为代表,信息覆盖+去冗余.

**Step 2 - Issue Agent**:从代表评论提取 theme + concrete issue.

**Step 3 - Recommendation Agent**:对每个 issue 生成 3-4 条行动建议.

**Step 4 - SRAC 四维度评估**:
$$\text{Score} = 0.25 \cdot S + 0.25 \cdot R + 0.25 \cdot A + 0.25 \cdot C$$
- **S** Specificity (具体性 1-5)
- **R** Relevance (相关性 1-5)
- **A** Actionability (可执行性 1-5)
- **C** Concision (简洁性 1-5)

迭代终止条件: $\text{Score} \geq 3.5$. 不达标的建议返回 Recommendation Agent 重写.

**Step 5 - Ranking Agent**:按企业可行性视角(成本/周期/效果)排序,输出 Top-K 优先行动清单.

### 关键假设

1. **评论体量**:足够支持聚类的代表性(论文实验 1000-10000 条/品类)
2. **负面/中评驱动**:相比好评,负评更能驱动 actionable 改进
3. **企业约束已知**:能给 Ranking Agent 提供成本/周期/效果维度

### 关键效果数字

| 对比维度 | MAA 多 Agent vs 单 LLM 基线 |
|---|---|
| **Actionability** | 持续超越 |
| **Specificity** | 持续超越 |
| **Non-redundancy** | 持续超越 |
| **规模兼容性** | 中等规模模型(Gemini Flash) ≈ 大模型 ensemble |

---

## ② 母婴出海应用案例

### 场景一:Momcozy 跨市场吸奶器评论洞察

- **业务问题**:Momcozy M5 吸奶器在美国/德国/中国三市场销售,各市场用户痛点完全不同(美国关注续航便携、德国关注静音认证、中国关注清洗方便). 现有运营复盘只产出"差评列表",**无法直接驱动产品改进决策**——产品经理拿到差评列表还要花 1-2 周二次提炼
- **数据要求**:三市场 Amazon Review API + market 标签
- **MAA 配置**:
  - 按市场分别聚类(K=5,每市场 5 个主题簇)
  - Issue Agent 提取每个簇的核心痛点(如"夜间使用嘈杂吵醒宝宝")
  - Recommendation Agent 生成 3-4 条改进建议(如"加静音模式降至 35dB"/"加振动反馈替代蜂鸣音")
  - SRAC 评分筛选 ≥3.5 分,Ranking 按工程改造成本+生产周期+用户增长潜力排序
- **业务价值**:
  - 产品复盘从"评论汇总"升级为"可排期改进清单"
  - 三市场各 5 条 Top 改进 = 15 条/季度直接进产品 roadmap
  - 改进精准度提升带来 NPS 提升 5-10 分,复购率 +3-5%
  - 年化收益(以 5000 万 Momcozy 美亚 GMV 计):**复购增量 150-250 万元** + 改进减少差评带来转化率提升 = **总计 400-700 万/年**

### 场景二:Momcozy 消毒器/暖奶器季度复盘

- **业务问题**:Q1-Q4 季度复盘,人工归纳 3 个产品 × 4 季度 = 12 次,每次 3-5 PM 天,**总计 36-60 PM 天/年**;复盘结论高度依赖个人经验,**新人接手质量大幅波动**
- **数据要求**:季度 Amazon + Wayfair 评论合并
- **MAA 配置**: AGRS 摘要 → MAA 5 Agent 决策链 → 输出"季度 Top 3-5 优先改进项 + ROI 排序"
- **业务价值**:
  - 节省人工:36-60 PM 天 × 3000元/天 = **10-18 万/年**
  - 决策质量标准化(SRAC 评分客观可比),新人接手不衰退
  - 改进 ROI 提升 20-30% = **额外 100-200 万/年**

---

## ③ 代码模板

```python
"""
MAA Multi-Agent Actionable Advice 最小骨架
论文 arXiv:2601.12024 (Bhandari et al., 2026-01)
完整实现见 paper2skills-code/nlp_voc/maa_actionable_advice/model.py
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Review:
    text: str
    review_id: str = ""
    market: str = ""


@dataclass
class SRAC:
    S: int = 0
    R: int = 0
    A: int = 0
    C: int = 0

    @property
    def score(self) -> float:
        return 0.25 * (self.S + self.R + self.A + self.C)


def clustering_agent(reviews: List[Review], k: int = 5) -> Dict[int, List[Review]]:
    """阶段 1: 聚类 + 选代表(生产替换为 TF-IDF + K-Means)"""
    keyword_buckets = {
        0: ["noise", "loud", "quiet"],
        1: ["clean", "wash", "hygiene"],
        2: ["battery", "portable", "wireless"],
        3: ["price", "expensive", "value"],
        4: ["build", "sturdy", "broken"],
    }
    clusters = defaultdict(list)
    for r in reviews:
        text_low = r.text.lower()
        best_cluster, best_match = -1, 0
        for cid, kws in keyword_buckets.items():
            match = sum(1 for kw in kws if kw in text_low)
            if match > best_match:
                best_cluster, best_match = cid, match
        if best_cluster >= 0:
            clusters[best_cluster].append(r)
    return dict(clusters)


def issue_agent(cluster_reps: Dict[int, List[Review]]) -> List[Dict]:
    """阶段 2: 抽取主题 + 具体问题"""
    issues = []
    for cid, reps in cluster_reps.items():
        if not reps:
            continue
        themes = ["noise", "cleaning", "portability", "value", "build_quality"]
        theme = themes[cid] if cid < len(themes) else "general"
        sample_texts = [r.text for r in reps[:3]]
        issues.append({
            "cluster_id": cid,
            "theme": theme,
            "issue": f"Concerns around {theme}",
            "evidence": sample_texts,
        })
    return issues


def recommendation_agent(issues: List[Dict]) -> List[Dict]:
    """阶段 3: 生成行动建议(每个 issue 3-4 条)"""
    rec_templates = {
        "noise": ["Add silent mode <35dB", "Replace beep with vibration", "Use damping material"],
        "cleaning": ["Add detachable wash zone", "Use food-grade silicone", "Provide cleaning brush"],
        "portability": ["Increase battery to 1500mAh", "Wireless charging support", "Reduce weight <300g"],
        "value": ["Bundle pricing with replacement parts", "Subscription kit", "Launch entry-tier SKU"],
        "build_quality": ["Reinforce hinge", "Use PA66 frame", "QC sampling 100%"],
    }
    recs = []
    for issue in issues:
        templates = rec_templates.get(issue["theme"], ["Investigate further"])
        for r in templates:
            recs.append({**issue, "recommendation": r})
    return recs


def srac_evaluation_agent(rec: Dict, threshold: float = 3.5) -> SRAC:
    """阶段 4: SRAC 四维度评分(生产替换为 LLM judge)"""
    score = SRAC(S=4, R=4, A=4, C=4)
    text = rec["recommendation"]
    if len(text) > 60:
        score.C = 2
    if "investigate" in text.lower():
        score.A = 2
    return score


def ranking_agent(recs_with_score: List[Dict]) -> List[Dict]:
    """阶段 5: 按企业可行性排序(成本/周期/效果)"""
    cost_map = {"Add silent mode <35dB": 0.9, "Reduce weight <300g": 0.7, "Bundle pricing": 1.0}
    for r in recs_with_score:
        c = cost_map.get(r["recommendation"], 0.5)
        r["feasibility"] = r["srac"].score * c
    return sorted(recs_with_score, key=lambda x: -x["feasibility"])


def run_maa_pipeline(reviews: List[Review], k: int = 5) -> Dict:
    clusters = clustering_agent(reviews, k=k)
    issues = issue_agent(clusters)
    recs = recommendation_agent(issues)
    scored = []
    for rec in recs:
        srac = srac_evaluation_agent(rec)
        if srac.score >= 3.5:
            scored.append({**rec, "srac": srac})
    ranked = ranking_agent(scored)
    return {
        "clusters_count": len(clusters),
        "issues_count": len(issues),
        "recommendations_pass_srac": len(scored),
        "top_5_actions": ranked[:5],
    }


def main() -> None:
    sample = [
        Review("very loud at night, wakes my baby", "r1", "US"),
        Review("hard to clean inside corner", "r2", "DE"),
        Review("battery only lasts 30 min", "r3", "US"),
        Review("too expensive for what it offers", "r4", "CN"),
        Review("hinge broke after 1 month", "r5", "DE"),
    ]
    result = run_maa_pipeline(sample)
    print(f"Clusters: {result['clusters_count']}, Issues: {result['issues_count']}")
    print(f"Recommendations passed SRAC: {result['recommendations_pass_srac']}")
    for i, action in enumerate(result["top_5_actions"], 1):
        print(f"  {i}. {action['theme']}: {action['recommendation']} (feasibility={action['feasibility']:.2f})")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-AGRS-Aspect-Guided-Review-Summarization](./[[Skill-AGRS-Aspect-Guided-Review-Summarization]].md) — AGRS 摘要为 MAA Clustering Agent 提供高质量输入
- [Skill-MAS-Orchestrator](../10-MAS/[[Skill-MAS-Orchestrator]].md) — 5 Agent 调度依赖 MAS 编排框架

### 延伸技能
- [Skill-Root-Cause-Analysis-Agent](../09-DataAgent-LLM/[[Skill-Root-Cause-Analysis-Agent]].md) — MAA Top 改进建议进入 RCA Agent 验证根因
- [Skill-Customer-Journey-Decision-Tree](../09-DataAgent-LLM/[[Skill-Customer-Journey-Decision-Tree]].md) — MAA 建议 → 客服 FAQ 模板更新

### 可组合
- [Skill-RFM-Customer-Segmentation](../06-增长模型/[[Skill-RFM-Customer-Segmentation]].md) — 按 RFM 高价值用户 Review 加权聚类
- [Skill-DML-Cohort-Causal-Effect](../01-因果推断/[[Skill-DML-Cohort-Causal-Effect]].md) — 改进上线后用 DML 验证用户分群因果效应

---

## ⑤ 商业价值评估

### ROI 预估

**场景一(跨市场吸奶器)**:**400-700 万/年**(NPS+复购+转化率组合提升)

**场景二(季度复盘)**:**110-220 万/年**(人工节省 + 改进 ROI 提升)

合计两个场景潜在年化:**510-920 万元/年**

### 实施难度:⭐⭐⭐⭐☆ (4/5)

- 易处:论文 prompt 模板完整公开,5 Agent 模块清晰
- 难处:需要业务专家初始化 Ranking Agent 的成本/周期评分规则
- 难处:5 Agent 串联推理 token 成本较高(单产品估算 $0.05-0.15 GPT-4o-mini)

### 优先级评分:⭐⭐⭐⭐⭐ (5/5)

**评估依据**:
1. **规范性决策链路** vs 传统描述性分析,直接产出可执行清单
2. **跨市场天然适配**(论文实验 Yelp 三领域,本质就是不同市场/品类)
3. **AGRS + MAA 双 Skill 组合** = WF-E Review 健康度的核心闭环
4. **SRAC 评分客观可比**,降低对资深 PM 经验依赖,新人接手不衰退
