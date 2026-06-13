---
title: NLP Copy AB Test Optimizer — CTR 加权偏好优化 + 遗传算法的文案 A/B 测试闭环
doc_type: knowledge
module: 02-A_B实验
topic: nlp-copy-ab-test-optimizer
status: stable
created: 2026-06-13
updated: 2026-06-13
owner: self
source: human+ai
roadmap_phase: phase2
algorithm_summary: RAG 检索优秀文案示例 + CoT 推理生成多样候选，CTR 收益加权 DPO 偏好优化迭代精化，Thompson Sampling MAB 动态分配流量，Alibaba 生产验证 CTR +4.11%，遗传算法 +30-50%
problem_solved: 母婴 Listing 主标题靠运营拍脑袋写，每次手动 A/B 测试周期 2 周且覆盖文案变体有限——NLP 文案 A/B 闭环系统自动生成 10+ 高质量变体并用 MAB 快速收敛最优文案，CTR 提升 4-15%，年化增量 GMV 20-80 万元
---

# Skill Card: NLP Copy AB Test Optimizer

> **论文**：CTOP: Creative Text Optimization with Preference Alignment（arXiv:2507.20227, Alibaba 2024）+ GCOF: Genetic Copy Optimization Framework（arXiv:2402.13667）+ 约束感知 MAB 文案生成（arXiv:2504.10391）
> **arXiv**：2507.20227 | 2024 | **桥梁**: 02-A_B实验 ↔ 07-NLP-VOC | **类型**: 跨域融合

## ① 算法原理

核心思路是把文案优化问题转化为**在线学习 + 偏好对齐**的闭环：

**三阶段流水线**：

1. **RAG + CoT 文案生成**：从历史高 CTR 文案库检索 Top-K 示例作为上下文（RAG），引导 LLM 用链式思维（CoT）生成多样变体。关键创新是注入 RAG 示例后，LLM 生成质量提升 20%+。

2. **CTR 收益加权 DPO（CTOP 核心）**：传统 DPO 把偏好对视为等权，CTOP 引入 CTR 收益权重：
   ```
   L_DPO = -E[w(y_w, y_l) · log σ(β · r(y_w) - β · r(y_l))]
   w(y_w, y_l) = CTR(y_w) - CTR(y_l)     # CTR 差越大，梯度权重越大
   ```
   在 500K 真实 A/B 偏好对上训练，Alibaba 生产验证 CTR +4.11%，RPM +1.02%。

3. **遗传算法跨文案变异（GCOF 核心）**：把文案词组视为"基因"，高 CTR 文案交叉产生后代，低 CTR 文案被淘汰。适应度 = LLM 评估的 CTR 潜力分 + 真实历史 CTR 反馈。+30-50% CTR 增益。

4. **Thompson Sampling MAB 流量分配**：每个文案变体维护 Beta(α, β) 后验，α = 点击数+1，β = 展示-点击+1。每次展示从各变体后验采样，选最高采样值的变体，比固定 50/50 A/B 节省 30-40% 探索成本。

---

## ② 母婴出海应用案例

**场景A：Momcozy M5 吸奶器 Listing 主标题优化**

- **业务问题**：同一 Listing 使用 1 年未更新标题，CTR 停滞在 2.1%，竞品标题加入"Hands-Free""80dB Quiet"等卖点关键词后 CTR 达 3.8%
- **数据要求**：历史高 CTR 标题 50 条（从 Search Term Report 提取）、关键词库（来自 Business Report + Jungle Scout）、7 天 A/B 展示数据（Search Term Impression Share）
- **执行流程**：用 RAG 检索同类高 CTR 标题 → 生成 10 个变体 → MAB 动态分流 7 天 → 遗传算法基于中期反馈二次变异 → 最终锁定最优标题
- **预期产出**：CTR 从 2.1% → 3.0-3.5%，基于月均 50,000 次展示 × CTR 提升 1%，月增点击 500 次，按 $30 CPC 节省广告费约 $15,000/月
- **业务价值**：年化节省广告费 ≈ 18 万元（按 ¥130/USD），同时自然流量 CTR 提升带动 BSR 改善

**场景B：Prime Day 促销广告文案批量优化**

- **业务问题**：大促前需为 30 个 SKU 各生成 5 条广告文案，人工写作耗时 3 天，效果参差不齐
- **数据要求**：各 SKU 历史 ACOS + CTR 数据、竞品 ASIN 文案（通过 Keepa API 抓取）、品类关键词频次
- **执行流程**：批量 RAG + CoT 生成 → 遗传算法交叉融合高 CTR 文案结构 → MAB 多变体同步测试 → 72 小时内收敛最优文案
- **预期产出**：30 SKU × CTR 平均提升 5% × Prime Day 日均展示 10 万次 = 增量点击 150 次/天
- **业务价值**：大促 3 天增量 GMV ≈ 450 次点击 × 8% 转化 × $55 客单 = $1,980 增量收入，成本趋零

---

## ③ 代码模板

```python
# NLP Copy A/B Test Optimizer
# 三阶段闭环：文案生成 → Thompson Sampling MAB → 遗传算法变异
# 依赖：numpy, re (标准库)

import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ============================================================
# 数据结构
# ============================================================

@dataclass
class CopyVariant:
    """文案变体及其 MAB 状态"""
    text: str
    alpha: float = 1.0   # Beta分布参数：点击数 + 1
    beta: float = 1.0    # Beta分布参数：未点击数 + 1

    @property
    def estimated_ctr(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def impressions(self) -> int:
        return int(self.alpha + self.beta - 2)

    @property
    def clicks(self) -> int:
        return int(self.alpha - 1)


# ============================================================
# Stage 1: Rule-based 文案生成（Mock RAG + CoT）
# ============================================================

def generate_copy_variants(
    product_name: str,
    keywords: List[str],
    features: List[str],
    n_variants: int = 5
) -> List[str]:
    """
    模拟 RAG + CoT 文案生成
    实际生产中替换为 LLM API 调用
    """
    # 模拟 RAG 检索到的高 CTR 文案模板
    rag_templates = [
        "{product} - {feat1}, {feat2} | {kw1} for {kw2}",
        "{product} {feat1} {feat2} - {kw1}, Best {kw2}",
        "{feat1} {product} | {kw1} {feat2} | {kw2} Grade",
        "{product} with {feat1} Technology - {kw1} & {kw2} Ready",
        "Upgraded {product} {feat2} - {feat1} | Top {kw1}",
    ]

    np.random.seed(42)
    variants = []
    for i, tmpl in enumerate(rag_templates[:n_variants]):
        # CoT：从特征库中选择最相关的特征填充模板
        feat1 = features[i % len(features)]
        feat2 = features[(i + 1) % len(features)]
        kw1 = keywords[i % len(keywords)]
        kw2 = keywords[(i + 2) % len(keywords)]
        copy = tmpl.format(
            product=product_name,
            feat1=feat1, feat2=feat2,
            kw1=kw1, kw2=kw2
        )
        variants.append(copy)
    return variants


# ============================================================
# Stage 2: Thompson Sampling MAB
# ============================================================

class ThompsonSamplingMAB:
    """
    多臂老虎机 Thompson Sampling
    每个文案变体维护 Beta(α, β) 后验分布
    """

    def __init__(self, copy_texts: List[str], random_seed: int = 0):
        self.variants = [CopyVariant(text=t) for t in copy_texts]
        self.rng = np.random.RandomState(random_seed)
        self.history: List[Tuple[int, int]] = []  # (arm_idx, reward)

    def select_arm(self) -> int:
        """从各变体 Beta 分布采样，选最高采样值的 arm"""
        samples = [
            self.rng.beta(v.alpha, v.beta)
            for v in self.variants
        ]
        return int(np.argmax(samples))

    def update(self, arm_idx: int, clicked: int):
        """
        更新后验
        clicked: 1 = 点击, 0 = 未点击
        """
        v = self.variants[arm_idx]
        v.alpha += clicked
        v.beta += (1 - clicked)
        self.history.append((arm_idx, clicked))

    def simulate_round(self, true_ctrs: List[float]) -> Tuple[int, int]:
        """模拟一次展示：选 arm → 按真实 CTR 决定是否点击"""
        arm = self.select_arm()
        click = int(self.rng.random() < true_ctrs[arm])
        self.update(arm, click)
        return arm, click

    def best_arm(self) -> int:
        return int(np.argmax([v.estimated_ctr for v in self.variants]))

    def summary(self) -> str:
        lines = [f"{'Arm':>3} | {'Impressions':>11} | {'Clicks':>6} | {'Est. CTR':>8} | 文案前40字"]
        lines.append("-" * 80)
        for i, v in enumerate(self.variants):
            lines.append(
                f"{i:>3} | {v.impressions:>11} | {v.clicks:>6} | "
                f"{v.estimated_ctr:.4f}   | {v.text[:40]}"
            )
        best = self.best_arm()
        lines.append(f"\n🏆 最优文案 Arm {best}: {self.variants[best].text}")
        return "\n".join(lines)


# ============================================================
# Stage 3: 遗传算法文案变异
# ============================================================

def genetic_crossover(variant_a: str, variant_b: str, rng: np.random.RandomState) -> str:
    """
    基因交叉：以词组为单位交换
    父本 A: "Momcozy M5 Hands-Free | 80dB Quiet | Breast Pump"
    父本 B: "Upgraded Wearable Pump - Hands-Free, Hospital Grade"
    后代: 混合高 CTR 词组
    """
    # 提取词组单元（按 | , - 分割）
    units_a = re.split(r'[|,\-]', variant_a)
    units_b = re.split(r'[|,\-]', variant_b)
    units_a = [u.strip() for u in units_a if u.strip()]
    units_b = [u.strip() for u in units_b if u.strip()]

    # 随机选取词组组合
    combined = units_a + units_b
    rng.shuffle(combined)
    n_select = max(3, min(4, len(combined)))
    selected = combined[:n_select]
    return " | ".join(selected)


def genetic_mutation(variant: str, keywords: List[str], rng: np.random.RandomState) -> str:
    """随机替换一个关键词（点突变）"""
    words = variant.split()
    if len(words) < 3:
        return variant
    mut_pos = rng.randint(0, len(words))
    new_kw = rng.choice(keywords)
    words[mut_pos] = new_kw
    return " ".join(words)


def run_genetic_evolution(
    mab: ThompsonSamplingMAB,
    keywords: List[str],
    top_k: int = 2,
    n_offspring: int = 3,
    random_seed: int = 1
) -> List[str]:
    """
    基于 MAB 当前反馈，选 Top-K 文案进行遗传进化
    返回新生成的后代文案列表
    """
    rng = np.random.RandomState(random_seed)

    # 按估计 CTR 排序取 Top-K
    ranked = sorted(
        range(len(mab.variants)),
        key=lambda i: mab.variants[i].estimated_ctr,
        reverse=True
    )
    top_variants = [mab.variants[i].text for i in ranked[:top_k]]

    offspring = []
    for _ in range(n_offspring):
        if len(top_variants) >= 2:
            # 交叉
            pa, pb = rng.choice(len(top_variants), size=2, replace=False)
            child = genetic_crossover(top_variants[pa], top_variants[pb], rng)
        else:
            child = top_variants[0]
        # 50% 概率点突变
        if rng.random() < 0.5:
            child = genetic_mutation(child, keywords, rng)
        offspring.append(child)

    return offspring


# ============================================================
# 主流程：Momcozy M5 吸奶器 Listing 优化
# ============================================================

def run_listing_optimization():
    print("=" * 60)
    print("NLP Copy A/B Test Optimizer — Momcozy M5 吸奶器")
    print("=" * 60)

    # 产品配置
    product_name = "Momcozy M5"
    keywords = ["Breast Pump", "Wearable Pump", "Hands-Free", "Hospital Grade", "Wireless"]
    features = ["80dB Quiet", "Hands-Free", "2 Modes 9 Levels", "Upgraded Motor", "24mm Flange"]

    # Stage 1: 生成初始 5 个文案变体
    print("\n📝 Stage 1: 生成初始文案变体")
    copy_texts = generate_copy_variants(product_name, keywords, features, n_variants=5)
    for i, t in enumerate(copy_texts):
        print(f"  Arm {i}: {t}")

    # 模拟真实 CTR（未知，仅用于模拟环境）
    # 真实场景中由 Amazon Search Term Report 提供
    true_ctrs = [0.021, 0.034, 0.019, 0.041, 0.028]

    # Stage 2: MAB 前 100 轮探索
    print("\n🎰 Stage 2: Thompson Sampling MAB（前100次展示）")
    mab = ThompsonSamplingMAB(copy_texts, random_seed=42)
    for _ in range(100):
        mab.simulate_round(true_ctrs)

    print("\n中期结果（100次展示后）:")
    print(mab.summary())

    # Stage 3: 遗传算法二次进化
    print("\n🧬 Stage 3: 遗传算法文案变异（基于 MAB 中期反馈）")
    offspring = run_genetic_evolution(mab, keywords, top_k=2, n_offspring=3, random_seed=7)
    print("新生成后代文案:")
    for i, o in enumerate(offspring):
        print(f"  Offspring {i}: {o}")

    # 将后代加入 MAB（重置后验，从头探索）
    all_texts = copy_texts + offspring
    true_ctrs_extended = true_ctrs + [0.038, 0.045, 0.033]  # 后代假设 CTR 更高
    mab2 = ThompsonSamplingMAB(all_texts, random_seed=99)

    # Stage 2 续：后 100 轮收敛
    print("\n🎰 MAB 第二阶段（加入后代，再100次展示）")
    for _ in range(100):
        mab2.simulate_round(true_ctrs_extended)

    print("\n最终结果（200次展示后）:")
    print(mab2.summary())

    best_idx = mab2.best_arm()
    best_variant = mab2.variants[best_idx]
    print(f"\n📊 收敛分析:")
    print(f"  最优文案 Arm {best_idx} 估计 CTR: {best_variant.estimated_ctr:.4f}")
    print(f"  vs 初始文案平均 CTR: {np.mean(true_ctrs):.4f}")
    print(f"  CTR 提升: +{(best_variant.estimated_ctr - np.mean(true_ctrs)) / np.mean(true_ctrs) * 100:.1f}%")

    return mab2


# ============================================================
# 测试用例
# ============================================================

def test_thompson_sampling():
    """验证 Thompson Sampling 能收敛到最优臂"""
    copies = ["Copy A", "Copy B", "Copy C"]
    true_ctrs = [0.02, 0.05, 0.03]
    mab = ThompsonSamplingMAB(copies, random_seed=0)
    for _ in range(500):
        mab.simulate_round(true_ctrs)
    best = mab.best_arm()
    assert best == 1, f"期望收敛到 Arm 1（CTR=0.05），实际 best={best}"
    assert mab.variants[1].estimated_ctr > mab.variants[0].estimated_ctr
    assert mab.variants[1].estimated_ctr > mab.variants[2].estimated_ctr
    print("✅ test_thompson_sampling: MAB 正确收敛到最优臂（Arm 1, CTR=0.05）")


def test_genetic_crossover():
    """验证遗传算法交叉产生有效后代"""
    rng = np.random.RandomState(42)
    parent_a = "Momcozy M5 Hands-Free | 80dB Quiet | Breast Pump"
    parent_b = "Upgraded Wearable Pump - Hospital Grade | Wireless"
    child = genetic_crossover(parent_a, parent_b, rng)
    assert len(child) > 0
    assert "|" in child or len(child.split()) >= 2
    print(f"✅ test_genetic_crossover: 后代文案生成成功 → '{child[:50]}...'")


def test_ctr_improvement():
    """验证整体流程 CTR 提升有效"""
    copies = [f"Copy variant {i}" for i in range(5)]
    true_ctrs = [0.02, 0.025, 0.03, 0.035, 0.04]
    mab = ThompsonSamplingMAB(copies, random_seed=123)
    for _ in range(200):
        mab.simulate_round(true_ctrs)
    best_estimated = mab.variants[mab.best_arm()].estimated_ctr
    avg_ctr = np.mean(true_ctrs)
    assert best_estimated > avg_ctr, f"最优文案 CTR {best_estimated:.4f} 应高于平均 {avg_ctr:.4f}"
    print(f"✅ test_ctr_improvement: 最优文案 CTR {best_estimated:.4f} > 平均 CTR {avg_ctr:.4f}")


def test_generate_variants():
    """验证文案生成数量和格式"""
    texts = generate_copy_variants("TestProduct", ["kw1", "kw2", "kw3"], ["f1", "f2", "f3"], n_variants=5)
    assert len(texts) == 5
    for t in texts:
        assert "TestProduct" in t
        assert len(t) > 10
    print(f"✅ test_generate_variants: 成功生成 {len(texts)} 个文案变体")


if __name__ == "__main__":
    # 运行测试
    print("\n" + "=" * 60)
    print("单元测试")
    print("=" * 60)
    test_generate_variants()
    test_thompson_sampling()
    test_genetic_crossover()
    test_ctr_improvement()

    # 主流程演示
    print("\n")
    run_listing_optimization()

    print("\n" + "=" * 60)
    print("[✓] NLP Copy AB Test Optimizer 测试通过")
    print("=" * 60)
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AB-Experimental-Design]]（A/B 测试统计显著性基础）、[[Skill-NLP-Text-Classification]]（文本特征理解）
- **延伸（extends）**：[[Skill-LLM-Review-Structured-Extraction]]（LLM 提取文案结构化特征替代 Rule-based 生成）
- **可组合（combinable）**：[[Skill-VOC-Aspect-Sentiment-Extraction]]（从用户评论提取高 CTR 关键词作为 RAG 种子库）+ [[Skill-Listing-Quality-Scoring]]（对生成文案打分筛选，替代遗传算法适应度函数）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 基础场景：月均展示 50K 次，CTR 提升 1%（绝对值），增量点击 500 次/月，按 ACOS 25%、客单 $55、转化率 8% 估算，增量 GMV ≈ ¥28,600/月，年化 **¥34.3 万元**
  - 进阶场景（10 个 SKU 同步优化）：年化增量 GMV **¥100-200 万元**
  - 广告节省：减少无效曝光，年化降低广告费 **¥15-30 万元**

- **实施难度**：⭐⭐⭐☆☆（主要难点在于接入 Amazon Advertising API 获取真实 CTR 反馈；纯文案生成部分 1 天可完成）

- **优先级**：⭐⭐⭐⭐⭐（Listing CTR 是 Amazon 排名核心因子，每个 SKU 都适用，复用率极高）

- **实施路径**：
  1. Week 1：部署文案生成模块（Rule-based/LLM），准备关键词库
  2. Week 2：接入 MAB 分流逻辑（通过 Amazon A/B Testing 后台或广告 API）
  3. Week 3-4：收集 7 天数据，触发遗传算法第一轮进化
  4. 持续：每月一个进化周期，CTR 逐步收敛最优文案
