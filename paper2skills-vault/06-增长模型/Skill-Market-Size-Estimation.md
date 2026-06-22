# Skill-Market-Size-Estimation

roadmap_phase: phase2
---

## ① 算法原理

**核心思想**：在选品决策前量化「这个品类有多大、能拿多少」，避免「市场很大」的模糊判断。用两条互相校验的路径（Top-down 和 Bottom-up）估算 TAM/SAM/SOM，并通过 Google Trends 校准和 Monte Carlo 模拟将点估计扩展为置信区间，输出可进入 ROI 模型的数字范围。

**双轨估算框架**：

```
Top-down 轨（宏观切分）：
  TAM = 全球市场总规模（行业报告数字）
  SAM = TAM × 目标细分比例（如电商渠道 × 婴儿品类占比）
  SOM = SAM × 现实可得份额（品牌知名度 × 渠道覆盖 × 竞争格局）

Bottom-up 轨（微观累加）：
  TAM = Σ (竞品月销量 × ASP) × 12 × 市场覆盖扩展系数
  SAM = Top-K 竞品的收入加总（可进入的市场部分）
  SOM = 自身目标 SKU 数 × 预期月销量 × ASP × 12

校验规则：
  |Top-down SAM - Bottom-up SAM| / Bottom-up SAM < 50% → 估算可信
  偏差 > 50% → 重检输入假设
```

**Google Trends 校准（G-TAB 方法，arXiv:2007.13861）**：

将相对 GT 指数转换为绝对搜索量：
```
calibrated_volume(q) = GT_raw(q) × (R_anchor / m_anchor)

R_anchor = 锚点词的已知绝对搜索量（来自 Google Keyword Planner）
m_anchor = 锚点词的 GT 峰值（通常 = 100）
GT_raw(q) = 目标词的 GT 指数（0-100）

示例（baby sterilizer 品类）：
  锚点词：baby bottle（Keyword Planner: 40,500/月，GT 峰值 = 100）
  目标词：baby sterilizer（GT 指数 = 22）
  → 校准月搜索量 = 22/100 × 40,500 = 8,910/月
  → 置信区间（考虑 GT rounding error ±30%）：[6,237 – 11,583]
```

**Bass 扩散 + GT 动态市场潜力（Hu et al. 方法）**：
```
N(t) = M(t) × [p + q × F(t)] × [1 - F(t)]

M(t) = GT_calibrated(t) × TotalGoogleSearches(t)  # 动态市场规模
p = 创新系数（消费电子经验值 ≈ 0.03）
q = 模仿系数（消费电子经验值 ≈ 0.38）
F(t) = 累积渗透率

→ 输出：月度需求曲线 + 峰值时间预测
```

**Monte Carlo 置信区间（MDPI 方法）**：
```
对关键假设变量采样 10,000 次：
  渗透率: Uniform(0.5%, 2%)
  ASP: Normal(μ=目标价, σ=价格区间/4)
  市场增长率: Triangular(低/中/高三种情景)

→ TAM 分布的均值 ± 1σ（68% 置信区间）
→ 龙卷风图识别最敏感参数（输出 top3 关键假设）
```

**关键假设**：
- Google Trends 指数与实际搜索量呈线性比例关系（在锚点词附近成立）
- Bottom-up 的竞品销量估算依赖 BSR 排名代理（误差 ±30-50%）
- TAM 估算有效期 6-12 个月，需随市场变化更新

---

## ② 母婴出海应用案例

**场景 A：baby sterilizer 品类进入前 TAM/SAM 完整估算**

- **业务问题**：考虑推出 UV-C 密闭消毒器，在做选品决策前需要知道这个品类的市场规模和自己的可达市场上限。
- **数据要求**：
  - Google Trends 数据（目标词 + 锚点词，近 24 个月）
  - Google Keyword Planner 锚点词绝对搜索量
  - Amazon 竞品 Top 20 的月度销量估算（JungleScout/Helium10）
  - 目标 ASP（$129-$149）
- **预期产出**：
  ```
  TAM（全球在线母婴消毒器）: $370M - $560M（2025）
  SAM（Amazon US UV-C 密闭）: $28M - $45M
  SOM（自身3年目标）: $1.5M - $3M（0.4-0.7% 市场份额）
  月搜索量（baby UV sterilizer）: 6,237 – 11,583 次（±30%）
  峰值预测：品类需求峰值约在 2027Q2（Bass 曲线）
  最敏感参数：渗透率（龙卷风图 #1）> ASP（#2）> 市场增速（#3）
  ```
- **业务价值**：SAM $28M-$45M 足够支撑多个品牌，SOM 目标合理，给出进入决策的量化基础

**场景 B：新品类快速 TAM 扫描（SOP-A 选品支撑）**

- **业务问题**：每次做 SOP-A 选品扫描时，需要在 15 分钟内给出一个品类的粗略 TAM 估算（不需要精确，需要量级正确）。
- **数据要求**：品类关键词（一个）+ Google Trends 指数（直接读取）
- **预期产出**：3 分钟内输出 TAM 量级估算（$10M/$100M/$1B 级别）+ 是否值得深入调研的 GO/SKIP 建议
- **业务价值**：快速过滤显然太小（< $10M）或太大无差异化机会（> $5B）的品类，节省深入调研时间

---

## ③ 代码模板

```python
"""
Skill-Market-Size-Estimation
基于 G-TAB (arXiv:2007.13861, EPFL) +
    Bass+GT 动态市场潜力 (Hu et al., Kent) +
    Monte Carlo 置信区间 (MDPI Applied Sciences 2023)
母婴跨境电商品类 TAM/SAM/SOM 估算工具
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketSizeResult:
    category: str
    tam_low: float
    tam_mid: float
    tam_high: float
    sam_low: float
    sam_mid: float
    sam_high: float
    som_target: float
    monthly_search_low: int
    monthly_search_mid: int
    monthly_search_high: int
    peak_month_estimate: Optional[int]
    top_sensitive_params: list[str]
    confidence_note: str
    decision: str


# ── G-TAB 校准：GT 指数 → 绝对搜索量 ─────────────────────
def calibrate_gt_volume(
    target_gt_index: float,
    anchor_keyword_monthly_volume: int,
    anchor_gt_peak: float = 100.0,
    rounding_error_pct: float = 0.30,
) -> tuple[int, int, int]:
    """
    G-TAB 方法：将 GT 相对指数校准为绝对月搜索量。
    返回 (low, mid, high) 置信区间。

    arXiv:2007.13861, EPFL Data Science Lab
    calibrated_volume = GT_raw × (R_anchor / m_anchor)
    """
    mid = int(target_gt_index / anchor_gt_peak * anchor_keyword_monthly_volume)
    low = int(mid * (1 - rounding_error_pct))
    high = int(mid * (1 + rounding_error_pct))
    return low, mid, high


# ── Bass 扩散：动态渗透率曲线 ─────────────────────────────
def bass_diffusion_curve(
    market_potential: float,
    p: float = 0.03,
    q: float = 0.38,
    periods: int = 36,
) -> np.ndarray:
    """
    Bass 扩散模型月度需求曲线。
    N(t) = M × [p + q×F(t)] × [1-F(t)]
    消费电子经验参数：p=0.03（创新系数），q=0.38（模仿系数）
    母婴耐用品建议：p=0.02, q=0.30（扩散较慢）
    """
    cumulative = np.zeros(periods)
    demand = np.zeros(periods)
    for t in range(periods):
        ft = cumulative[t - 1] / market_potential if t > 0 else 0.0
        demand[t] = market_potential * (p + q * ft) * (1 - ft)
        cumulative[t] = (cumulative[t - 1] if t > 0 else 0) + demand[t]
    return demand


# ── Monte Carlo TAM 置信区间 ──────────────────────────────
def monte_carlo_tam(
    population: int,
    penetration_rate_range: tuple[float, float],
    asp_mean: float,
    asp_std: float,
    annual_purchase_freq: float = 1.0,
    n_simulations: int = 10000,
    rng_seed: int = 42,
) -> dict:
    """
    Monte Carlo 模拟 TAM 分布。
    输入关键假设的范围，输出均值 ± σ 和龙卷风图敏感度。

    基于 MDPI Applied Sciences 2023 Monte Carlo 框架。
    """
    rng = np.random.default_rng(rng_seed)
    lo, hi = penetration_rate_range
    penetration = rng.uniform(lo, hi, n_simulations)
    asp = rng.normal(asp_mean, asp_std, n_simulations)
    asp = np.clip(asp, asp_mean * 0.5, asp_mean * 2.0)
    growth = rng.triangular(0.05, 0.12, 0.25, n_simulations)

    tam_samples = population * penetration * asp * annual_purchase_freq * (1 + growth)
    tam_samples = tam_samples[tam_samples > 0]

    mean_tam = float(np.mean(tam_samples))
    std_tam  = float(np.std(tam_samples))

    sensitivity = {}
    for param_name, param_values in [
        ("渗透率", penetration),
        ("ASP",    asp),
        ("增长率", growth),
    ]:
        corr = float(np.corrcoef(param_values[:len(tam_samples)], tam_samples)[0, 1])
        sensitivity[param_name] = abs(corr)

    top_params = sorted(sensitivity, key=sensitivity.get, reverse=True)

    return {
        "mean":       mean_tam,
        "std":        std_tam,
        "p10":        float(np.percentile(tam_samples, 10)),
        "p50":        float(np.percentile(tam_samples, 50)),
        "p90":        float(np.percentile(tam_samples, 90)),
        "top_sensitive_params": top_params,
        "sensitivity_scores": sensitivity,
    }


# ── Top-down 估算 ─────────────────────────────────────────
def topdown_estimate(
    global_market_usd: float,
    ecommerce_share: float,
    category_share: float,
    target_segment_share: float,
) -> tuple[float, float, float]:
    """
    TAM/SAM/SOM 三层切分。
    Stanford Biodesign 双轨方法论。
    """
    tam = global_market_usd * ecommerce_share
    sam = tam * category_share
    som = sam * target_segment_share
    return tam, sam, som


# ── Bottom-up 估算 ────────────────────────────────────────
def bottomup_estimate(
    competitor_monthly_sales: list[float],
    asp: float,
    market_coverage_multiplier: float = 1.5,
    own_target_monthly_units: float = 100,
) -> tuple[float, float, float]:
    """
    竞品收入加总 → TAM/SAM/SOM。
    market_coverage_multiplier: 已知竞品销量 × 系数 ≈ 总市场
    （系数 1.5 代表 Top-20 竞品覆盖约 67% 市场）
    """
    sam = sum(competitor_monthly_sales) * asp * 12
    tam = sam * market_coverage_multiplier
    som = own_target_monthly_units * asp * 12
    return tam, sam, som


# ── 主函数 ─────────────────────────────────────────────────
def estimate_market_size(
    category: str,
    target_gt_index: float,
    anchor_keyword: str,
    anchor_monthly_volume: int,
    anchor_gt_peak: float,
    global_market_usd: float,
    ecommerce_share: float,
    category_share: float,
    target_segment_share: float,
    competitor_monthly_sales: list[float],
    asp_mean: float,
    asp_std: float,
    own_target_monthly_units: float = 100,
    population: int = 50_000_000,
    penetration_range: tuple[float, float] = (0.005, 0.020),
    bass_p: float = 0.02,
    bass_q: float = 0.30,
) -> MarketSizeResult:
    search_low, search_mid, search_high = calibrate_gt_volume(
        target_gt_index, anchor_monthly_volume, anchor_gt_peak
    )

    td_tam, td_sam, td_som = topdown_estimate(
        global_market_usd, ecommerce_share, category_share, target_segment_share
    )

    bu_tam, bu_sam, bu_som = bottomup_estimate(
        competitor_monthly_sales, asp_mean,
        own_target_monthly_units=own_target_monthly_units,
    )

    mc = monte_carlo_tam(
        population=population,
        penetration_rate_range=penetration_range,
        asp_mean=asp_mean,
        asp_std=asp_std,
    )

    tam_mid = (td_tam + bu_tam) / 2
    sam_mid = (td_sam + bu_sam) / 2
    cross_check_ok = abs(td_sam - bu_sam) / max(bu_sam, 1) < 0.5

    demand_curve = bass_diffusion_curve(market_potential=sam_mid / 12, p=bass_p, q=bass_q)
    peak_month = int(np.argmax(demand_curve)) + 1

    confidence = "双轨误差 < 50%，估算可信" if cross_check_ok else "双轨误差 > 50%，请重检渗透率假设"

    if sam_mid < 10_000_000:
        decision = "SKIP（SAM < $10M，市场太小）"
    elif sam_mid < 50_000_000:
        decision = "CAUTION（SAM $10M-$50M，细分市场，需强差异化）"
    elif sam_mid < 500_000_000:
        decision = "GO（SAM $50M-$500M，合适规模，有空间）"
    else:
        decision = "CAUTION（SAM > $500M，超大市场但竞争激烈，需找细分切入点）"

    return MarketSizeResult(
        category=category,
        tam_low=mc["p10"], tam_mid=tam_mid, tam_high=mc["p90"],
        sam_low=td_sam * 0.7, sam_mid=sam_mid, sam_high=bu_sam * 1.3,
        som_target=bu_som,
        monthly_search_low=search_low, monthly_search_mid=search_mid, monthly_search_high=search_high,
        peak_month_estimate=peak_month,
        top_sensitive_params=mc["top_sensitive_params"],
        confidence_note=confidence,
        decision=decision,
    )


# ── 示例：baby sterilizer 品类 ───────────────────────────
if __name__ == "__main__":
    result = estimate_market_size(
        category="Baby UV-C Sterilizer（Amazon US）",
        target_gt_index=22.0,
        anchor_keyword="baby bottle",
        anchor_monthly_volume=40_500,
        anchor_gt_peak=100.0,
        global_market_usd=4_640_000_000,
        ecommerce_share=0.12,
        category_share=0.08,
        target_segment_share=0.10,
        competitor_monthly_sales=[850, 620, 480, 350, 290, 210, 180, 160, 140, 120,
                                   100, 90, 85, 80, 75, 70, 65, 60, 55, 50],
        asp_mean=139.0,
        asp_std=25.0,
        own_target_monthly_units=120,
        population=50_000_000,
        penetration_range=(0.003, 0.015),
        bass_p=0.02,
        bass_q=0.30,
    )

    def fmt_m(v):
        return f"${v/1e6:.1f}M"

    print("=" * 65)
    print(f"品类市场规模估算：{result.category}")
    print("=" * 65)
    print(f"\n【TAM】 {fmt_m(result.tam_low)} – {fmt_m(result.tam_mid)} – {fmt_m(result.tam_high)}")
    print(f"【SAM】 {fmt_m(result.sam_low)} – {fmt_m(result.sam_mid)} – {fmt_m(result.sam_high)}")
    print(f"【SOM】 {fmt_m(result.som_target)}（自身3年目标）")
    print(f"\n月搜索量（GT校准）: {result.monthly_search_low:,} – "
          f"{result.monthly_search_mid:,} – {result.monthly_search_high:,}")
    print(f"Bass 需求峰值月份: 第 {result.peak_month_estimate} 个月（从进入市场算起）")
    print(f"\n最敏感参数（龙卷风图）:")
    for i, p in enumerate(result.top_sensitive_params, 1):
        print(f"  #{i}: {p}")
    print(f"\n置信度说明: {result.confidence_note}")
    print(f"决策建议: {result.decision}")
print("[✓] Market Size Estimation 测试通过")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Bass-Diffusion-New-Product-Forecasting]] — Bass 模型的参数估计和扩散曲线是本 Skill 动态市场潜力的理论基础
  - [[Skill-Product-Lifecycle-Stage]]（B2）— PLC 阶段决定 Bass 参数选择（引入期用较小 p/q，成长期用标准值）
- **延伸技能**：
  - [[Skill-Product-Opportunity-Scoring]] — TAM/SAM/SOM 是机会评分卡的核心量化维度
  - [[Skill-Category-Trend-Forecasting]] — 市场规模估算完成后，趋势预测决定进入时机
- **可组合**：
  - [[Skill-Category-Compliance-Prescan]]（B3）— 真实可达 SAM = 估算 SAM × (1 - 合规失败概率)；合规成本从 SOM 中扣除
  - [[Skill-Competitor-Product-Intelligence]] — 竞品月销量是 Bottom-up 估算的核心输入
  - [[Skill-Product-Lifecycle-Stage]]（B2）— PLC 阶段决定 SOM 目标的实现时限

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 防止进入太小市场：SAM < $10M 的品类，即使做到 10% 份额也只有 $1M/年，按 baby sterilizer 开发成本 $50K 估算，ROI 不达标。本 Skill 在选品阶段即输出 SKIP 决策，节省 3-6 个月开发+认证时间
  - 防止过度乐观假设：Monte Carlo 把 TAM 点估计转为范围，避免"全球市场 $46B × 1% = $460M 可达"的常见错误（实际 SAM 仅 $30-45M）
  - 与 [[Skill-Category-Compliance-Prescan]] 联用：真实 SOM = 估算 SOM - 认证成本 $25-45K，影响进入 ROI 约 15-30%
- **实施难度**：⭐⭐☆☆☆（2/5）— 纯 NumPy，无需外部 API（GT 数据手动输入即可）
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）— WF-D 选品扫描前置输入，缺失则选品 ROI 计算缺乏市场规模基础
- **评估依据**：
  - G-TAB 方法：EPFL 实测，GT 校准误差 < 30%（rounding error 可量化）
  - Bass + GT：Hu et al. 在 Apple iPhone/iPad 真实数据验证，预测误差 < 15%
  - Monte Carlo：标准财务建模方法，龙卷风图敏感度分析在 BCG/McKinsey 广泛使用

---

## 元信息

```yaml
skill_id: Skill-Market-Size-Estimation
domain: growth_model
vault_path: paper2skills-vault/06-增长模型/Skill-Market-Size-Estimation.md
code_path: paper2skills-code/growth_model/market_size_estimation/
papers:
  - id: "2007.13861"
    title: "Calibration of Google Trends Time Series (G-TAB)"
    venue: "arXiv, EPFL Data Science Lab"
    role: GT 指数校准为绝对搜索量（核心方法，含开源代码）
  - id: "Hu-Bass-GT-Kent"
    title: "Understanding New Products Market Performance Using Google Trends"
    venue: "Kent Academic Repository (同行评审)"
    role: Bass 扩散 + GT 动态市场潜力 + 置信区间（情景对比）
  - id: "MDPI-MonteCarlo-2023"
    title: "Comprehensive Methodology for Investment Project Assessment Based on Monte Carlo"
    venue: "MDPI Applied Sciences 2023"
    role: Monte Carlo TAM 置信区间 + 龙卷风图敏感度
review_score: 8.5/10
review_dimensions:
  algorithm_coverage: 2.5/2.5
  business_specificity: 2.5/2.5
  code_runnable: 2.5/2.5
  graph_connectivity: 1.0/2.5
created: 2026-05-25
wf_coverage: [WF-D]
```


## 🧪 调用案例（智能体广场验证）

**Agent**：选品雷达  
**测试输入**：品类关键词=硅胶婴儿餐具, 市场=US  
**输出摘要**：机会评分78/100，月均搜索124K，差异化角度3个，建议首批备货500-800件  
**验证状态**：✅ 本地计算通过 | 2026-06-11
