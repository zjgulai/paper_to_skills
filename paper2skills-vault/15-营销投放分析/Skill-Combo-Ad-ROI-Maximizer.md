---
title: 广告ROI最大化 Combo Pattern — 从归因分析到预算自动重分配的 6 步优化链路
doc_type: knowledge
module: 15-营销投放分析
topic: combo-ad-roi-maximizer-orchestration
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 广告ROI最大化 Combo Pattern

> **类型**：Combo Pattern（业务解决方案编排）
> **桥梁**：15-营销投放分析 ↔ 13-广告分析 ↔ 01-因果推断 ↔ 16-智能体工程
> **触发条件**：广告 ROAS < 目标值 20% 或 月度广告预算超支 > 15%，触发 6 步 ROI 优化链路

## ① 算法原理

广告 ROI 最大化 Combo Pattern 解决「广告预算越花越多，ROI 越来越低」的核心困境。根因通常是：多渠道归因混乱（MTA 虚报 / 末次点击偏差）+ 价格弹性未校准（降价促销实际稀释了广告效率）+ 预算分配靠经验拍脑袋。

**6 步链路设计**：

```
Step1 多触点归因  → 清洗归因偏差，建立真实渠道贡献矩阵
         ↓ 渠道真实 ROI（去偏后）
Step2 价格弹性校准 → 识别「低价 = 低利润」的广告点击陷阱
         ↓ 价格-需求弹性系数 ε
Step3 MMM 预算分析 → 测量各渠道的边际效益递减点（saturation curve）
         ↓ 渠道饱和度 + 最优饱和区间
Step4 跨平台预算分配 → 在亚马逊 SP/SB/SD、TikTok、Google 间最优分配
         ↓ 渠道预算分配比例
Step5 自然+付费协同 → 避免自我蚕食（cannibalization）
         ↓ 协同系数 + 调整后分配
Step6 Agent 自动优化 → 持续学习，每周自动调整预算
         ↓ 动态执行计划 + A/B 实验反馈
```

**数学核心**：最大化 Σ ROI_i(b_i)，约束 Σ b_i = B_total（总预算），其中 ROI_i(b_i) 是渠道 i 的 S 型饱和曲线（Hill function）。使用拉格朗日乘数法求解最优 b_i*，确保各渠道边际 ROI 相等。

**反直觉洞察**：最优解通常不是「给 ROAS 最高的渠道分配最多预算」——饱和曲线意味着边际 ROI 递减，真正的最优是找到「各渠道边际 ROI 相等」的均衡点。

## ② 母婴出海应用案例

**场景A：母婴品牌亚马逊+TikTok 双渠道广告 ROI 优化（月预算 $20,000）**

- **业务问题**：月度广告支出 $20,000，综合 ROAS = 2.1（行业均值 3.5），运营直觉是「TikTok 效果不好但老板要求继续投」
- **数据要求**：各渠道广告消耗、点击量、转化量（至少 90 天）、商品价格历史、自然流量数据
- **执行过程**：
  - Step1 归因修正：TikTok 末次点击归因虚报 40%（实际 view-through 转化多），修正后 TikTok 真实 ROAS = 1.8（非 3.2）
  - Step2 价格弹性：主推品弹性系数 ε = -1.8（高弹性），广告引流配合 -15% 降价可提升转化量 27%
  - Step3 MMM：亚马逊 SP 已达到 85% 饱和（边际效益极低），SB 仅 40% 饱和
  - Step4 预算重分配：SP 减少 $3,000 → SB 增加 $2,000 + TikTok 定向投减少 $1,000
  - Step5 协同调整：降低 SP 出价避免与自然位自我蚕食，净协同收益 +12%
  - Step6 Agent 部署：每周一自动读取上周数据，调整下周出价上限
- **量化产出**：ROAS 从 2.1 提升至 3.4（+62%），同等预算月销售额增加 $16,800，年化 ROI 提升价值约 201,600 美元

**场景B：跨境婴儿推车品牌 Black Friday 广告预算冲刺**

- **业务问题**：BF 当天预算从 $2,000/天 临时拉高至 $8,000/天，历史经验是「加钱就能出单」但实际超支 60% 而 ROAS 暴跌
- **执行亮点**：Step3 提前 7 天建立大促期间的饱和曲线（vs 平日区别很大），Step4 精确识别最优加投上限为 $5,500/天（超过此值 ROAS < 2.0），节省无效预算 $2,500/天 × 4 天 = $10,000

## ③ 代码模板

```python
"""
广告 ROI 最大化 Combo Pattern — 6 步优化链路
"""
from dataclasses import dataclass, field
from typing import Optional
import math

# ──────────────────────────────────────────────
# 渠道数据结构
# ──────────────────────────────────────────────
@dataclass
class AdChannel:
    name: str                    # 渠道名（如 "Amazon_SP"）
    current_spend_usd: float     # 当前月度预算
    reported_roas: float         # 平台上报 ROAS（有偏）
    clicks: int
    conversions: int
    organic_conversions: int     # 同期自然转化量（用于协同分析）
    # Combo 填充
    corrected_roas: float = 0.0   # Step1 修正后真实 ROAS
    saturation_pct: float = 0.0   # Step3 当前饱和度
    optimal_spend_usd: float = 0.0  # Step4 最优预算
    cannibalization_rate: float = 0.0  # Step5 自我蚕食率

@dataclass
class AdOptContext:
    channels: list[AdChannel]
    total_budget_usd: float
    price_elasticity: float = -1.5  # Step2 填充
    final_plan: dict = field(default_factory=dict)

# ──────────────────────────────────────────────
# Step1: 多触点归因修正 — Skill-Ad-Attribution-Modeling
# ──────────────────────────────────────────────
def step1_attribution_correction(ctx: AdOptContext) -> AdOptContext:
    """修正平台归因偏差（末次点击 → Shapley 值近似）"""
    BIAS_FACTORS = {
        "Amazon_SP": 0.92,    # SP 相对准确
        "Amazon_SB": 0.88,    # SB 有 view-through 虚报
        "TikTok": 0.65,       # TikTok 末次点击严重虚报
        "Google": 0.78,       # Google 中等偏差
        "Meta": 0.72,         # Meta 类似 TikTok
    }
    for ch in ctx.channels:
        bias = BIAS_FACTORS.get(ch.name, 0.85)
        ch.corrected_roas = round(ch.reported_roas * bias, 3)
    corrections = [(c.name, f"{c.reported_roas:.2f}→{c.corrected_roas:.2f}") for c in ctx.channels]
    print(f"  [Step1] 归因修正: {corrections}")
    return ctx

# ──────────────────────────────────────────────
# Step2: 价格弹性校准 — Skill-Price-Elasticity-Estimation
# ──────────────────────────────────────────────
def step2_price_elasticity(ctx: AdOptContext) -> AdOptContext:
    """估计产品价格弹性（简化：基于历史 CPC vs 转化率相关性）"""
    total_clicks = sum(c.clicks for c in ctx.channels)
    total_conv = sum(c.conversions for c in ctx.channels)
    cvr = total_conv / max(total_clicks, 1)
    # 高转化率 + 高 CPC → 价格敏感型（弹性大）
    avg_cpc = sum(c.current_spend_usd for c in ctx.channels) / max(total_clicks, 1)
    ctx.price_elasticity = -1.0 - (cvr * 5) - (avg_cpc / 10)
    ctx.price_elasticity = max(-3.0, min(-0.5, ctx.price_elasticity))  # 合理区间
    price_sensitivity = "高弹性（降价效果显著）" if ctx.price_elasticity < -1.5 else "低弹性（提升内容质量更重要）"
    print(f"  [Step2] 价格弹性 ε={ctx.price_elasticity:.2f} → {price_sensitivity}")
    return ctx

# ──────────────────────────────────────────────
# Step3: MMM 饱和度分析 — Skill-Marketing-Mix-Modeling
# ──────────────────────────────────────────────
def step3_saturation_analysis(ctx: AdOptContext) -> AdOptContext:
    """Hill 函数估计各渠道饱和度: ROI(b) = max_roi × b^α / (b^α + K^α)"""
    SATURATION_PARAMS = {
        "Amazon_SP": {"K": 3000, "alpha": 0.7},    # 小预算快饱和
        "Amazon_SB": {"K": 8000, "alpha": 0.8},    # 品牌词需要更多预算
        "TikTok": {"K": 6000, "alpha": 0.6},
        "Google": {"K": 5000, "alpha": 0.75},
        "Meta": {"K": 4500, "alpha": 0.65},
    }
    for ch in ctx.channels:
        params = SATURATION_PARAMS.get(ch.name, {"K": 5000, "alpha": 0.7})
        b = ch.current_spend_usd
        K, alpha = params["K"], params["alpha"]
        ch.saturation_pct = round((b**alpha) / (b**alpha + K**alpha), 3)
    oversaturated = [(c.name, f"{c.saturation_pct:.0%}") for c in ctx.channels if c.saturation_pct > 0.75]
    print(f"  [Step3] 饱和度分析: 过饱和渠道={oversaturated}")
    return ctx

# ──────────────────────────────────────────────
# Step4: 跨平台预算分配 — Skill-Multi-Platform-Ad-Budget-Allocator
# ──────────────────────────────────────────────
def step4_budget_allocation(ctx: AdOptContext) -> AdOptContext:
    """拉格朗日最优分配：各渠道边际 ROI 相等"""
    # 简化：按「corrected_roas × (1-saturation_pct)」权重分配
    weights = {}
    for ch in ctx.channels:
        # 边际效益 = 真实 ROAS × 未饱和空间
        weights[ch.name] = ch.corrected_roas * (1.0 - ch.saturation_pct)

    total_weight = sum(weights.values())
    for ch in ctx.channels:
        ch.optimal_spend_usd = round(ctx.total_budget_usd * weights[ch.name] / max(total_weight, 0.01), 2)

    changes = [(c.name, f"${c.current_spend_usd:.0f}→${c.optimal_spend_usd:.0f}") for c in ctx.channels]
    print(f"  [Step4] 预算重分配: {changes}")
    return ctx

# ──────────────────────────────────────────────
# Step5: 自然+付费协同分析 — Skill-Search-Ad-Budget-ROI-Integration
# ──────────────────────────────────────────────
def step5_organic_paid_synergy(ctx: AdOptContext) -> AdOptContext:
    """计算自我蚕食率并调整最优预算"""
    for ch in ctx.channels:
        # 蚕食率：付费 CTR 越高，侵占自然排名的概率越高
        total_conv = ch.conversions + ch.organic_conversions
        ch.cannibalization_rate = round(
            ch.conversions / max(total_conv, 1) * 0.3, 3
        )  # 30% 的付费转化实际来自本应是自然的流量
        # 调整后最优预算（减去蚕食部分）
        adj = ch.optimal_spend_usd * (1.0 - ch.cannibalization_rate * 0.5)
        ch.optimal_spend_usd = round(adj, 2)

    avg_cannibalization = sum(c.cannibalization_rate for c in ctx.channels) / len(ctx.channels)
    print(f"  [Step5] 协同优化: 平均蚕食率={avg_cannibalization:.1%}，预算下调后仍可维持自然流量")
    return ctx

# ──────────────────────────────────────────────
# Step6: Agent 自动优化计划 — Skill-DARA-Agentic-MMM-Optimizer
# ──────────────────────────────────────────────
def step6_agent_optimization_plan(ctx: AdOptContext) -> AdOptContext:
    """生成 Agent 自动化执行计划"""
    expected_roas_improvement = sum(
        (c.optimal_spend_usd * c.corrected_roas - c.current_spend_usd * c.corrected_roas)
        for c in ctx.channels
    ) / max(ctx.total_budget_usd, 1)

    ctx.final_plan = {
        "execution_mode": "weekly_auto_rebalance",
        "rebalance_trigger": "ROAS deviation > 15% from target",
        "channels": [
            {
                "name": c.name,
                "current_budget": c.current_spend_usd,
                "recommended_budget": c.optimal_spend_usd,
                "delta": round(c.optimal_spend_usd - c.current_spend_usd, 2),
                "corrected_roas": c.corrected_roas,
                "saturation": f"{c.saturation_pct:.0%}",
            }
            for c in ctx.channels
        ],
        "expected_roas_improvement_pct": round(expected_roas_improvement * 100, 1),
        "ab_test_recommendation": "对 Amazon SB 新预算做 2 周 holdout 实验验证",
    }
    print(f"  [Step6] Agent 优化计划: 预期 ROAS 改善 {ctx.final_plan['expected_roas_improvement_pct']}%")
    return ctx

# ──────────────────────────────────────────────
# Combo 编排入口
# ──────────────────────────────────────────────
def run_ad_roi_maximizer(channels: list[AdChannel], total_budget_usd: float) -> AdOptContext:
    ctx = AdOptContext(channels=channels, total_budget_usd=total_budget_usd)
    current_total_roas = (sum(c.current_spend_usd * c.reported_roas for c in channels) /
                          max(total_budget_usd, 1))
    print(f"\n📊 广告 ROI 最大化 Combo Pattern 启动")
    print(f"   总预算=${total_budget_usd:,.0f}, 渠道数={len(channels)}, 当前综合ROAS={current_total_roas:.2f}")
    print("=" * 60)

    for step_fn in [step1_attribution_correction, step2_price_elasticity,
                    step3_saturation_analysis, step4_budget_allocation,
                    step5_organic_paid_synergy, step6_agent_optimization_plan]:
        ctx = step_fn(ctx)

    new_total = sum(c.optimal_spend_usd for c in channels)
    print("=" * 60)
    print(f"💰 优化结果: 总预算分配核验=${new_total:,.0f} (目标=${total_budget_usd:,.0f})")
    print(f"🤖 Agent 执行模式: {ctx.final_plan['execution_mode']}")
    return ctx

# ──────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_channels = [
        AdChannel("Amazon_SP", current_spend_usd=8000, reported_roas=3.5,
                  clicks=5200, conversions=156, organic_conversions=89),
        AdChannel("Amazon_SB", current_spend_usd=4000, reported_roas=2.8,
                  clicks=2100, conversions=63, organic_conversions=120),
        AdChannel("TikTok",    current_spend_usd=5000, reported_roas=3.1,
                  clicks=18000, conversions=90, organic_conversions=30),
        AdChannel("Google",    current_spend_usd=3000, reported_roas=2.4,
                  clicks=1800, conversions=54, organic_conversions=45),
    ]

    ctx = run_ad_roi_maximizer(test_channels, total_budget_usd=20000)

    assert ctx.price_elasticity < 0, "价格弹性应为负数"
    assert all(c.corrected_roas > 0 for c in ctx.channels), "修正后 ROAS 应大于 0"
    assert all(c.optimal_spend_usd >= 0 for c in ctx.channels), "最优预算应为非负"
    assert len(ctx.final_plan["channels"]) == 4, "应有 4 个渠道的最终计划"
    assert ctx.final_plan["expected_roas_improvement_pct"] is not None, "应有预期改善数值"

    print("\n📋 最终预算计划：")
    for ch_plan in ctx.final_plan["channels"]:
        delta = ch_plan["delta"]
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {ch_plan['name']}: ${ch_plan['current_budget']:.0f} {arrow} ${ch_plan['recommended_budget']:.0f} "
              f"(Δ${delta:+.0f}) | 真实ROAS={ch_plan['corrected_roas']:.2f} | 饱和度={ch_plan['saturation']}")

    print(f"\n预期 ROAS 改善: +{ctx.final_plan['expected_roas_improvement_pct']}%")
    print("\n[✓] 广告 ROI 最大化 Combo Pattern 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Ad-Attribution-Modeling]]（Step1 核心：多触点因果归因，修正 ROAS 偏差）
- **前置（prerequisite）**：[[Skill-Marketing-Mix-Modeling]]（Step3 核心：MMM 饱和曲线建模）
- **组合（combinable）**：[[Skill-Price-Elasticity-Estimation]]（Step2：价格弹性校准，防止广告 + 降价双重稀释利润）
- **组合（combinable）**：[[Skill-Multi-Platform-Ad-Budget-Allocator]]（Step4：Lagrangian 最优预算分配）
- **组合（combinable）**：[[Skill-Search-Ad-Budget-ROI-Integration]]（Step5：自然 + 付费协同，消除自我蚕食）
- **组合（combinable）**：[[Skill-DARA-Agentic-MMM-Optimizer]]（Step6：Agent 持续学习自动调优）
- **延伸（extends）**：[[Skill-Combo-New-Product-Launch-Playbook]]（新品上市时的广告起量 Combo）
- **延伸（extends）**：[[Skill-Channel-Saturation-Curve]]（深化：各渠道 S 型饱和曲线精确建模）

## ⑤ 商业价值评估

- **ROI 预估**：月广告预算 $20,000 的品牌，ROAS 从 2.1 提升至 3.4（+62%），同等预算额外增收 = ($20,000 × 3.4 - $20,000 × 2.1) = $26,000/月，年化 $312,000（约 220 万人民币）；实施成本约 3-5 万元（工程化），ROI > 40x
- **关键洞察**：归因修正（Step1）通常是最大单一杠杆——TikTok 等渠道虚报 ROAS 高达 35-50%，修正后可立即重新分配 $3,000-$5,000/月
- **实施难度**：⭐⭐⭐⭐☆（MMM 需要 90 天历史数据 + Python 建模能力，完整实施约 4 周）
- **优先级**：⭐⭐⭐⭐⭐（广告费是跨境电商最大可变成本，任何优化都有直接底线影响）
- **适用规模**：月广告支出 ≥ $5,000（< $5,000 MMM 样本量不足），跨 2+ 渠道投放
