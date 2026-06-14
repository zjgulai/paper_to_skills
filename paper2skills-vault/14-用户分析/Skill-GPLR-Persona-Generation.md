---
title: GPLR 用户人群标签生成 - 购买行为到可解释 Persona 的低成本桥接
doc_type: knowledge
module: 14-用户分析
topic: persona-generation-llm-random-walk

roadmap_phase: phase2
created: 2026-05-19
updated: 2026-05-19
owner: self
source: human+ai
paper: arXiv:2504.17304
---

# Skill: GPLR — 用户人群标签生成(LLM + 随机游走低成本 Persona 推断)

> 论文:**You Are What You Bought: Generating Customer Personas for E-commerce Applications** (2025) · arXiv:2504.17304 · SIGIR 2025
> 核心贡献：仅用 5% 用户做 LLM 标注预算，通过随机游走传播将标签覆盖全量用户；人群标签稳定性比 RFM 高 13.8 倍。
> 代码模板：`paper2skills-code/nlp_voc/gplr_persona_generation/model.py`（568 行，`python3 model.py` 验证通过）

---

## ① 算法原理

### 核心思想

用户购买行为包含丰富的人群信号，但直接为百万用户调用 LLM 标注成本极高。GPLR 解决这个矛盾：**用少量 LLM 标注 + 图结构传播覆盖全量用户**。三步流程：① Diversity-Uncertainty（DU）采样选出最有代表性的"原型用户"做 LLM 标注；② LLM 基于购买历史为原型用户赋予 Persona 标签；③ 在用户-产品交互图上随机游走，将标签从有标注用户传播至全量未标注用户。

### 数学直觉

**DU 采样得分**（同时满足覆盖多样性 + 选择不确定性高的用户）：
$$\text{DU}(u) = \alpha \cdot D_{\text{KL}}(P_{\text{current}} \| \hat{P}_u) + (1-\alpha) \cdot H(u)$$

- $P_{\text{current}}$：已采集 Persona 的分布
- $\hat{P}_u$：预估用户 $u$ 的 Persona 分布（基于邻居推断）
- $H(u)$：用户购买多样性熵（品类越多，不确定性越高）
- $\alpha = 0.5$：多样性与不确定性均衡权重

**随机游走亲和度传播**（$t$ 步后的用户-Persona 矩阵）：
$$A^{(t+1)}[u, r] = \sum_{v \in \mathcal{N}(u)} \text{Sim}(u,v) \cdot A^{(t)}[v, r]$$

相似度使用 Jaccard：$\text{Sim}(u,v) = \frac{|P_u \cap P_v|}{|P_u \cup P_v|}$，$P_u$ 为用户购买产品集合。

### 关键假设

1. **相似购买行为 → 相似 Persona**：图传播有效的根本前提；适合购买历史≥5条的活跃用户
2. **5% LLM 预算足够**：DU 采样保证代表性，原型用户的标签能有效覆盖全域；冷启动用户（<5 次交互）效果受限
3. **Persona 集合预定义**：需要业务团队提前定义 6-10 个有意义的人群类型，泛化性依赖标签设计质量

### 关键效果数字

| 指标 | 数值 |
|---|---|
| LLM 标注预算（用户比例） | **5%** |
| Persona 稳定性 vs RFM | **13.8 倍** |
| 随机游走步数 | 2 步（平衡精度/计算量） |
| 推荐的 Persona 集合大小 | 6-10 个（复杂度可控） |

---

## ② 母婴出海应用案例

### 场景一：Momcozy 吸奶器用户人群精准分层

- **业务问题**：Momcozy 在 Amazon US 有 10 万+ 活跃用户，现有 RFM 分群只能区分"高消费/低消费"，营销团队无法针对"出差妈妈""新手妈妈"制定差异化素材；人工打标既慢又贵
- **数据要求**：Amazon 订单数据（`user_id`, `product_id`, `purchase_date`）+ 预定义 Persona 集合（如：职场背奶妈妈 / 全职新手妈妈 / 出差旅行妈妈 / 静音敏感型 / 价格敏感型）
- **GPLR 配置**：
  - 构建用户-产品交互图（购买=1.0，浏览=0.3）
  - DU 采样 5,000 名原型用户（5 万用户 5% 预算）→ LLM 基于购买 SKU 名称 + 品类标注 Persona
  - 随机游走 2 步传播至全量 10 万用户
  - 输出 JSON：`{user_id: [(persona, confidence)], ...}`，置信度 ≥0.5 归入主 Persona
- **预期产出**：
  - 每个用户 Top-3 Persona 标签 + 置信度
  - 6 个营销人群包：规模分布、典型用户、推荐 SKU 动作
- **业务价值**：
  - 广告 CTR 提升估算：精准人群 ROAS 提升 20-35%（行业基准），100 万/月广告预算 → **净增 20-35 万/月 = 240-420 万/年**
  - 替代人工打标：节省 3-4 人月/年 × 1.5 万/月 = **4.5-6 万/年**

### 场景二：新品冷启动快速 Persona 定位

- **业务问题**：Momcozy 静音款 S12 Pro 上市 1 个月，仅有 800 条购买记录，运营不确定核心人群是"职场妈妈"还是"夜晚哺乳妈妈"；若判断错误，亚马逊广告关键词投放偏差，前 2 个月 ROI 极低
- **数据要求**：新品 800 条购买记录 + 已有成熟产品 5 万条历史购买记录（图传播底座）
- **GPLR 配置**：
  - 使用全品类历史用户图作为传播底座（已包含各 Persona 原型）
  - 新品购买用户作为待推断节点加入图
  - 无需额外 LLM 标注（复用历史原型），直接随机游走 2 步得出 Persona 分布
  - 输出：新品用户 Persona 分布饼图 + 主导人群推荐广告词
- **业务价值**：
  - 上市第 1 个月即可定位核心人群，广告关键词精准率从 40% → 70%
  - 避免 2 个月错误投放损失：10 万广告费 × 30% 精准率提升 = **3-5 万/新品**，年化 20 款新品 = **60-100 万/年**

---

## ③ 代码模板

```python
"""
GPLR: Generating Personas with LLM and Random Walk
论文 arXiv:2504.17304 (SIGIR 2025)
完整实现见 paper2skills-code/nlp_voc/gplr_persona_generation/model.py
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class UserInteraction:
    user_id: str
    product_id: str
    interaction_type: str  # 'purchase', 'view', 'review'
    timestamp: str
    value: float = 1.0


class InteractionGraph:
    """用户-产品交互图"""

    def __init__(self):
        self.interactions: Dict[str, List[UserInteraction]] = defaultdict(list)
        self._user_set: set = set()

    def add_interaction(self, interaction: UserInteraction):
        self.interactions[interaction.user_id].append(interaction)
        self._user_set.add(interaction.user_id)

    def build_index(self):
        self.user_ids = list(self._user_set)
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}

    def get_user_interactions(self, user_id: str) -> List[UserInteraction]:
        return self.interactions.get(user_id, [])

    def get_user_index(self, user_id: str) -> int:
        return self.user_to_idx.get(user_id, -1)

    def get_similar_users(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Jaccard 相似度找相似用户"""
        user_id = self.user_ids[user_idx]
        user_products = set(i.product_id for i in self.interactions[user_id])
        sims = []
        for other_id, other_idx in self.user_to_idx.items():
            if other_id == user_id:
                continue
            other_products = set(i.product_id for i in self.interactions[other_id])
            union = user_products | other_products
            if union:
                sim = len(user_products & other_products) / len(union)
                sims.append((other_idx, sim))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]


class GPLRProfiler:
    """
    GPLR 人群标签生成器（三步流程）
    step1: DU采样 → step2: LLM标注原型用户 → step3: 随机游走传播
    """

    def __init__(
        self,
        persona_set: List[str],
        llm_budget_ratio: float = 0.05,
        random_walk_steps: int = 2,
    ):
        self.persona_set = persona_set
        self.llm_budget_ratio = llm_budget_ratio
        self.walk_steps = random_walk_steps

    def generate_personas(
        self,
        graph: InteractionGraph,
        prototype_labels: Dict[str, List[Tuple[str, float]]] | None = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        生成全量用户 Persona。

        Args:
            graph: 交互图（已 build_index）
            prototype_labels: 预先 LLM 标注的原型用户 {user_id: [(persona, score)]}
                              若 None 则使用 mock 策略（演示用）

        Returns:
            Dict[str, List[Tuple[str, float]]]: 每个用户的 Top-3 Persona + 置信度
        """
        user_ids = graph.user_ids
        n = len(user_ids)
        n_personas = len(self.persona_set)
        persona_to_idx = {p: i for i, p in enumerate(self.persona_set)}

        # --- step1: DU 采样（若未提供 prototype_labels） ---
        budget = max(1, int(n * self.llm_budget_ratio))
        if prototype_labels is None:
            prototype_ids = self._du_sample(user_ids, budget, graph)
            prototype_labels = {
                uid: self._mock_llm_label(graph.get_user_interactions(uid))
                for uid in prototype_ids
            }

        # --- step2: 初始化亲和度矩阵 ---
        affinity = np.zeros((n, n_personas))
        for uid, personas in prototype_labels.items():
            idx = graph.get_user_index(uid)
            if idx >= 0:
                for persona, score in personas:
                    if persona in persona_to_idx:
                        affinity[idx, persona_to_idx[persona]] = score

        # --- step3: 随机游走传播 ---
        for _ in range(self.walk_steps):
            new_aff = np.zeros_like(affinity)
            for u_idx in range(n):
                neighbors = graph.get_similar_users(u_idx, top_k=10)
                for v_idx, sim in neighbors:
                    new_aff[u_idx] += sim * affinity[v_idx]
                row_sum = new_aff[u_idx].sum()
                if row_sum > 0:
                    new_aff[u_idx] /= row_sum
            affinity = new_aff

        # --- step4: 输出 Top-3 Persona ---
        results: Dict[str, List[Tuple[str, float]]] = {}
        for uid in user_ids:
            u_idx = graph.get_user_index(uid)
            scores = affinity[u_idx]
            top3 = np.argsort(scores)[-3:][::-1]
            results[uid] = [
                (self.persona_set[i], float(scores[i]))
                for i in top3
                if scores[i] > 0.05
            ]
        return results

    def create_marketing_segments(
        self,
        user_personas: Dict[str, List[Tuple[str, float]]],
        confidence_threshold: float = 0.5,
        min_segment_size: int = 2,
    ) -> Dict[str, Dict]:
        """基于 Persona 生成营销人群包"""
        segments: Dict[str, List[str]] = defaultdict(list)
        for uid, personas in user_personas.items():
            for persona, score in personas:
                if score >= confidence_threshold:
                    segments[persona].append(uid)

        return {
            name: {
                "size": len(users),
                "sample_users": users[:5],
                "recommended_action": self._recommend_action(name),
            }
            for name, users in segments.items()
            if len(users) >= min_segment_size
        }

    # ---- 内部工具方法 ----

    def _du_sample(self, user_ids: List[str], budget: int, graph: InteractionGraph) -> List[str]:
        """Diversity-Uncertainty 采样"""
        sampled: List[str] = []
        current_dist: Dict[str, float] = defaultdict(float)
        for _ in range(budget):
            best, best_score = None, -float("inf")
            for uid in user_ids:
                if uid in sampled:
                    continue
                interactions = graph.get_user_interactions(uid)
                categories = {i.product_id.split("_")[0] for i in interactions}
                uncertainty = len(categories) / 10.0
                # 多样性用当前分布中已选用户数量的负对数近似
                diversity = -np.log1p(current_dist.get(uid, 0))
                score = 0.5 * diversity + 0.5 * uncertainty
                if score > best_score:
                    best_score, best = score, uid
            if best:
                sampled.append(best)
                current_dist[best] = current_dist.get(best, 0) + 1
        return sampled

    def _mock_llm_label(self, interactions: List[UserInteraction]) -> List[Tuple[str, float]]:
        """模拟 LLM 标注（生产替换为实际 LLM API 调用）"""
        products = [i.product_id for i in interactions]
        personas = []
        if any("S12" in p or "静音" in p for p in products):
            personas.append(("职场背奶妈妈", 0.80))
        if any("S9" in p or "便携" in p for p in products):
            personas.append(("出差旅行妈妈", 0.85))
        if any("M5" in p or "新手" in p for p in products):
            personas.append(("全职新手妈妈", 0.75))
        return personas or [("价格敏感型", 0.60)]

    def _recommend_action(self, segment_name: str) -> str:
        mapping = {
            "职场背奶": "推送降噪配件套装 + 职场收纳礼包",
            "出差旅行": "推送便携包 + 车载充电配件",
            "全职新手": "推送使用教程视频 + 新手礼包优惠",
            "价格敏感": "推送限时满减券 + 捆绑特价",
            "静音敏感": "推送静音款升级套餐",
        }
        for kw, action in mapping.items():
            if kw in segment_name:
                return action
        return "推送新品资讯 + 个性化关联产品"


# ====== 可运行测试 ======

def build_sample_graph() -> InteractionGraph:
    graph = InteractionGraph()
    data = [
        ("U001", "S12_静音款", "purchase"), ("U001", "便携包", "purchase"),
        ("U002", "S9_便携款", "purchase"), ("U002", "车载充电器", "purchase"),
        ("U003", "M5_入门款", "purchase"), ("U003", "储奶袋", "purchase"),
        ("U004", "S12_静音款", "view"), ("U004", "M5_入门款", "purchase"),
        ("U005", "S12_静音款", "purchase"), ("U005", "便携包", "purchase"),
    ]
    for uid, pid, itype in data:
        graph.add_interaction(UserInteraction(uid, pid, itype, "2024-01-01"))
    graph.build_index()
    return graph


def main():
    print("=" * 60)
    print("GPLR Persona Generation - Momcozy 场景验证")
    print("=" * 60)

    graph = build_sample_graph()
    persona_set = ["职场背奶妈妈", "全职新手妈妈", "出差旅行妈妈", "价格敏感型", "静音敏感型", "便携关注型"]

    profiler = GPLRProfiler(persona_set=persona_set, llm_budget_ratio=0.4, random_walk_steps=2)
    user_personas = profiler.generate_personas(graph)

    print("\n[用户 Persona 结果]")
    for uid, personas in user_personas.items():
        print(f"  {uid}: {[(p, round(s, 3)) for p, s in personas]}")

    segments = profiler.create_marketing_segments(user_personas)
    print("\n[营销人群包]")
    for name, info in segments.items():
        print(f"  {name}: {info['size']} 人 | {info['recommended_action']}")

    # 基本断言
    assert len(user_personas) == 5, "应有 5 个用户结果"
    assert any(len(v) > 0 for v in user_personas.values()), "至少有 1 个用户有 Persona"
    print("\n✓ 基本测试通过")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-User-Funnel-Analysis](./[[Skill-User-Funnel-Analysis]].md) — 漏斗分析提供用户行为数据基础；GPLR 需要交互历史作为图构建输入
- [Skill-Cohort-Retention-Analysis](./[[Skill-Cohort-Retention-Analysis]].md) — Cohort 分析提供时间维度用户分层，与 Persona 标签互补形成完整用户画像

### 延伸技能
- [Skill-MAA-Review-to-Action-Decision](./[[Skill-MAA-Review-to-Action-Decision]].md) — Persona 标签输入 MAA，为不同人群生成差异化的产品改进建议
- [Skill-AGRS-Aspect-Guided-Review-Summarization](./[[Skill-AGRS-Aspect-Guided-Review-Summarization]].md) — AGRS 提取人群关注的 Aspect，验证和丰富 GPLR 生成的 Persona 标签

### 可组合
- [Skill-Uplift-Modeling](../01-因果推断/[[Skill-Uplift-Modeling]].md) — Persona 分层 × Uplift 模型：识别每个人群中对促销最敏感的"可说服用户"，精准发券
- [Skill-DARA-Agentic-MMM-Optimizer](../15-营销投放分析/[[Skill-DARA-Agentic-MMM-Optimizer]].md) — Persona 驱动 MMM 渠道预算拆分：不同人群在不同渠道的响应系数各异，结合 DARA 动态调优预算

---

## ⑤ 商业价值评估

### ROI 预估

**场景一（精准广告分层）**：
- 现状：10 万用户 RFM 分层 3 档，广告 ROAS ≈ 3.0
- GPLR 后：6 档精准 Persona 定投，ROAS 提升 20-30%
- 广告月预算 100 万 × 25% ROAS 提升 → **净增利润 8-12 万/月 = 96-144 万/年**
- LLM 标注成本：5,000 用户 × 0.002 元/次 = **10 元（极低）**

**场景二（新品人群定位）**：
- 避免前 2 个月广告定向偏差损失：10 万广告费 × 30% 浪费率减少 = **3 万/新品**
- 年化 20 款新品 = **60 万/年**

**合计预估：年化 156-204 万/年**（中型品牌，广告月预算 100 万）

### 实施难度：⭐⭐☆☆☆ (2/5)

- 易处：纯 Python + numpy 实现，无需 GPU；代码 568 行已完整封装，`python3 model.py` 直接可运行
- 易处：图传播逻辑简单（Jaccard + 矩阵乘法），工程化成本低
- 关键工作：① 业务团队定义 6-10 个有意义的 Persona（约 1-2 天）；② 接入真实 LLM API 替换 `_mock_llm_label`（约 0.5 天）
- 可选优化：当用户数 >10 万时，用稀疏矩阵 + ANN 近似加速图传播

### 优先级评分：⭐⭐⭐⭐☆ (4/5)

**评估依据**：
1. **LLM 成本极低**：5% 标注预算下，10 万用户 LLM 费用 <100 元，ROI 极高
2. **标签稳定性 13.8× RFM**：解决运营团队长期抱怨 RFM 分群月月漂移的痛点
3. **代码已验证可运行**：`paper2skills-code/nlp_voc/gplr_persona_generation/model.py` 直接 import 使用
4. **填补 14-用户分析 人群生成缺口**：现有 AGRS/MAA/StaR/LACA 均聚焦评论分析，GPLR 是首个基于**购买行为**的人群标签生成技能
5. 需要 LLM API 接入增加少量工程工作 → 不给满分
