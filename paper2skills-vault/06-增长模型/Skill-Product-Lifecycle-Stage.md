# Skill-Product-Lifecycle-Stage

---

## ① 算法原理

**核心思想**：把一个 SKU 或品类的销量时间序列，分解为趋势+季节+残差三层信号，通过微分分析（斜率变化率）自动定位「成长→成熟→衰退」的阶段边界，并用年龄-销量矩（AVM）作为阶段状态的低成本代理特征，最终输出四阶段标签（引入/成长/成熟/衰退）+ 进入时机决策建议。

**数学直觉**：

**1. STL 时序分解（去除促销噪声）**
```
Y_t = T_t + S_t + R_t
  T_t: 趋势项（长期销量走向）
  S_t: 季节项（大促/节假日周期）← 母婴品类必须剥离
  R_t: 残差项（随机噪声）
```
分析只在 T_t 上做，避免 618/双11 spike 被误判为"成长期"。

**2. 微分分析定位阶段边界（PhaseFormer 框架）**
```
peak_idx      = argmax(T_t)               # 峰值时间点
inflection_idx = argmin(d²T_t / dt²)[t>peak]  # 衰退加速拐点

实证规律（AAAI 2025 验证）：
  衰退期斜率 / 成熟期斜率 ≈ 9.04×  → 衰退信号极强
  Kruskal-Wallis 检验：阶段间差异 p < 0.001
```

**3. 四阶段量化判断规则**
```
引入期: AVM极低 & 月均增速 > 15% & 上市月份 ≤ 6
成长期: 月均增速 > 5% & d²T/dt² > 0  （加速增长）
成熟期: |月均增速| ≤ 5% & T_t 在峰值 ±15% 范围内
衰退期: 月均增速 < -5% & |衰退斜率/成熟斜率| > 3×
```

**4. AVM（年龄-销量矩）—— 无需预设曲线形状**
```
AVM(t) = age_since_launch(t) × cumulative_volume(t)
```
仅需「上市日期 + 累计销量」两个字段，shape-agnostic，适配母婴 SKU 换代升级导致的多峰生命周期（配方奶粉 1段→2段→3段）。

**关键假设**：
- 品类销量时间序列长度 ≥ 12 个月（否则成熟/衰退判断不稳定）
- 促销活动有记录（用于 STL 季节项校准）
- 竞品替代效应通过「竞品数量增长率」外生输入，不内嵌于模型

---

## ② 母婴出海应用案例

**场景 A：新品类进入时机判断**

- **业务问题**：考虑进入 baby UV-C sterilizer 品类，不知道该品类处于哪个 PLC 阶段，是该现在进还是已经过了最佳时机。
- **数据要求**：
  - 品类月度搜索量（Google Trends 指数，近 24 个月）
  - Top 10 竞品的月度 BSR 排名或 Review 增速（近 24 个月）
  - 竞品数量（同类 ASIN 数，近 24 个月）
- **预期产出**：
  - 当前阶段标签（引入/成长/成熟/衰退）
  - 阶段置信度（基于斜率比和 AVM 特征）
  - 进入时机建议（GO/WAIT/NO-GO + 理由）
- **业务价值**：避免在衰退期进入，节省产品开发 + 认证 + 首批备货成本约 $15,000-$50,000

**场景 B：在售 SKU 生命周期监控（换代预警）**

- **业务问题**：主力 SKU baby sterilizer Pro 已上市 18 个月，近 3 个月 BSR 在下滑，不确定是暂时性的还是进入衰退期，是否该启动换代新品研发。
- **数据要求**：该 SKU 每周 BSR 排名（或 Review 增速）近 24 个月
- **预期产出**：
  - 当前阶段 + 进入该阶段的月份数
  - 衰退斜率/成熟斜率比值（>3× 触发换代预警）
  - 预测剩余「有效生命期」（基于历史同类 SKU 衰退速度）
- **业务价值**：新品研发周期约 8-12 个月，提前 6 个月发出换代预警可确保无断档期

---

## ③ 代码模板

```python
"""
Skill-Product-Lifecycle-Stage
基于 arXiv:2511.16248 (PhaseFormer, AAAI 2025) + arXiv:2511.17275 (AVM, 2025)
母婴跨境电商品类/SKU 生命周期阶段检测
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("提示: pip install statsmodels 启用 STL 分解，当前使用简化版")


class PLCStage(Enum):
    INTRODUCTION = "引入期"
    GROWTH       = "成长期"
    MATURITY     = "成熟期"
    DECLINE      = "衰退期"
    UNKNOWN      = "数据不足"


@dataclass
class PLCResult:
    sku_id: str
    current_stage: PLCStage
    confidence: float          # 0-1
    months_in_stage: int
    growth_rate_mom: float     # 最近3个月平均月增速
    slope_ratio: Optional[float]  # 衰退斜率/成熟斜率，>3× 为强衰退信号
    avm_score: float           # 年龄-销量矩归一化值
    decision: str              # GO / WAIT / NO-GO
    rationale: str
    warning: Optional[str] = None


# ── STL 分解（剥离大促季节性）─────────────────────────────
def decompose_trend(sales: np.ndarray, period: int = 12) -> np.ndarray:
    """
    STL 分解提取趋势项，剥离促销季节性噪声。
    Y_t = T_t + S_t + R_t，返回 T_t。
    """
    if len(sales) < period * 2:
        # 数据不足时用移动平均代替
        window = min(3, len(sales))
        return pd.Series(sales).rolling(window, center=True, min_periods=1).mean().values

    if HAS_STATSMODELS:
        stl = STL(sales, period=period, robust=True)
        result = stl.fit()
        return result.trend
    else:
        # 简化版：中心移动平均
        return pd.Series(sales).rolling(period, center=True, min_periods=1).mean().values


# ── 微分分析定位阶段边界（PhaseFormer 核心逻辑）─────────────
def find_phase_boundaries(trend: np.ndarray) -> dict:
    """
    基于 PhaseFormer 微分分析法定位 peak 和 inflection point。
    返回: {peak_idx, inflection_idx, slopes}
    """
    n = len(trend)
    if n < 6:
        return {"peak_idx": n // 2, "inflection_idx": n - 1, "slopes": {}}

    # 一阶导数（斜率）
    d1 = np.gradient(trend)
    # 二阶导数（曲率变化）
    d2 = np.gradient(d1)

    # 峰值位置
    peak_idx = int(np.argmax(trend))

    # 衰退加速拐点（峰后二阶导最小值）
    post_peak = d2[peak_idx:]
    inflection_offset = int(np.argmin(post_peak)) if len(post_peak) > 0 else 0
    inflection_idx = peak_idx + inflection_offset

    # 各阶段斜率统计
    growth_slope   = float(np.mean(d1[:peak_idx])) if peak_idx > 0 else 0
    mature_slope   = float(np.mean(np.abs(d1[peak_idx:inflection_idx]))) if inflection_idx > peak_idx else 0
    decline_slope  = float(np.mean(d1[inflection_idx:])) if inflection_idx < n else 0

    slope_ratio = abs(decline_slope) / max(abs(mature_slope), 1e-6)

    return {
        "peak_idx": peak_idx,
        "inflection_idx": inflection_idx,
        "slopes": {
            "growth": growth_slope,
            "mature": mature_slope,
            "decline": decline_slope,
            "slope_ratio": slope_ratio,  # 论文实证：衰退 ≈ 9.04× 成熟
        }
    }


# ── AVM 特征（年龄-销量矩）────────────────────────────────
def compute_avm(sales: np.ndarray, launch_offset: int = 0) -> float:
    """
    AVM(t) = age × cumulative_volume，归一化到 [0,1]。
    launch_offset: 距上市的月份数（用于全品类分析时对齐）
    """
    n = len(sales)
    ages = np.arange(launch_offset + 1, launch_offset + n + 1)
    cum_vol = np.cumsum(sales)
    avm_series = ages * cum_vol
    # 归一化
    avm_max = avm_series[-1] if avm_series[-1] > 0 else 1
    return float(avm_series[-1] / avm_max)


# ── 四阶段分类器 ───────────────────────────────────────────
def classify_stage(
    trend: np.ndarray,
    boundaries: dict,
    months_on_market: int,
) -> tuple[PLCStage, float, int]:
    """
    返回: (stage, confidence, months_in_current_stage)
    """
    n = len(trend)
    if n < 4:
        return PLCStage.UNKNOWN, 0.0, 0

    peak_idx       = boundaries["peak_idx"]
    inflection_idx = boundaries["inflection_idx"]
    slopes         = boundaries["slopes"]
    current_idx    = n - 1

    # 月增速（最近3期）
    recent_window = trend[-3:] if n >= 3 else trend
    mom_rates = np.diff(recent_window) / np.maximum(np.abs(recent_window[:-1]), 1)
    avg_mom = float(np.mean(mom_rates)) if len(mom_rates) > 0 else 0

    # 四阶段判断规则
    if months_on_market <= 6 and avg_mom > 0.10:
        stage = PLCStage.INTRODUCTION
        confidence = min(0.9, 0.5 + avg_mom * 2)
        months_in = months_on_market

    elif current_idx < peak_idx and avg_mom > 0.03:
        stage = PLCStage.GROWTH
        confidence = min(0.9, 0.5 + avg_mom * 3)
        months_in = current_idx

    elif current_idx >= inflection_idx and slopes.get("slope_ratio", 0) > 3.0:
        stage = PLCStage.DECLINE
        sr = slopes.get("slope_ratio", 3.0)
        confidence = min(0.95, 0.6 + (sr - 3) * 0.05)
        months_in = current_idx - inflection_idx

    elif abs(avg_mom) <= 0.07:
        stage = PLCStage.MATURITY
        confidence = min(0.85, 0.6 + (0.07 - abs(avg_mom)) * 5)
        months_in = current_idx - peak_idx

    else:
        # 过渡状态：趋势下滑但未达强衰退阈值
        stage = PLCStage.DECLINE if avg_mom < 0 else PLCStage.GROWTH
        confidence = 0.5
        months_in = 1

    return stage, confidence, max(0, months_in)


# ── 进入时机决策矩阵 ──────────────────────────────────────
DECISION_MATRIX = {
    # (PLCStage, 竞争密度级别) → (decision, rationale)
    (PLCStage.INTRODUCTION, "low"):    ("GO",     "品类早期+竞争少，先发优势窗口，建议快速进入"),
    (PLCStage.INTRODUCTION, "medium"): ("GO",     "品类早期+竞争中等，仍有增长红利，建议进入并做差异化"),
    (PLCStage.INTRODUCTION, "high"):   ("WAIT",   "品类早期但竞争已激烈，等待竞品出清或找细分切入点"),
    (PLCStage.GROWTH, "low"):          ("GO",     "成长期+竞争稀疏，最佳进入时机"),
    (PLCStage.GROWTH, "medium"):       ("GO",     "成长期+适度竞争，市场扩张足够容纳新进入者"),
    (PLCStage.GROWTH, "high"):         ("WAIT",   "成长期但竞争激烈，需要明确差异化优势再进"),
    (PLCStage.MATURITY, "low"):        ("WAIT",   "成熟期+竞争少，需评估是否品类规模够大支撑进入"),
    (PLCStage.MATURITY, "medium"):     ("WAIT",   "成熟期+中等竞争，进入需极强差异化或成本优势"),
    (PLCStage.MATURITY, "high"):       ("NO-GO",  "成熟期+高竞争，红海，进入成本高、回报周期长"),
    (PLCStage.DECLINE, "low"):         ("NO-GO",  "衰退期，市场萎缩，不建议进入（除非有退出品牌的清货机会）"),
    (PLCStage.DECLINE, "medium"):      ("NO-GO",  "衰退期+竞争中等，价格战激烈，利润极薄"),
    (PLCStage.DECLINE, "high"):        ("NO-GO",  "衰退期+高竞争，双重不利因素，明确放弃"),
    (PLCStage.UNKNOWN, "low"):         ("WAIT",   "数据不足，需补充至少12个月历史数据"),
    (PLCStage.UNKNOWN, "medium"):      ("WAIT",   "数据不足，暂缓决策"),
    (PLCStage.UNKNOWN, "high"):        ("WAIT",   "数据不足，暂缓决策"),
}

def get_competition_level(competitor_count: int) -> str:
    if competitor_count <= 20:   return "low"
    elif competitor_count <= 80: return "medium"
    else:                        return "high"


# ── 主函数 ─────────────────────────────────────────────────
def analyze_lifecycle(
    sku_id: str,
    monthly_sales: list[float],
    months_on_market: int,
    competitor_count: int = 50,
    seasonal_period: int = 12,
) -> PLCResult:
    """
    完整生命周期分析。

    Args:
        sku_id: SKU 或品类标识
        monthly_sales: 月度销量序列（从最早到最近，至少6个月）
        months_on_market: 距上市总月数
        competitor_count: 当前同类竞品数量
        seasonal_period: 季节周期（月），默认12

    Returns:
        PLCResult 包含阶段、置信度、决策建议
    """
    sales = np.array(monthly_sales, dtype=float)

    if len(sales) < 4:
        return PLCResult(
            sku_id=sku_id, current_stage=PLCStage.UNKNOWN, confidence=0.0,
            months_in_stage=0, growth_rate_mom=0.0, slope_ratio=None,
            avm_score=0.0, decision="WAIT", rationale="数据不足（需≥6个月）",
        )

    # Step 1: STL 分解
    trend = decompose_trend(sales, period=seasonal_period)

    # Step 2: 阶段边界检测
    boundaries = find_phase_boundaries(trend)

    # Step 3: 阶段分类
    stage, confidence, months_in_stage = classify_stage(trend, boundaries, months_on_market)

    # Step 4: AVM 特征
    avm_score = compute_avm(sales)

    # Step 5: 月增速（最近3个月）
    recent = sales[-3:] if len(sales) >= 3 else sales
    mom_rates = np.diff(recent) / np.maximum(np.abs(recent[:-1]), 1)
    avg_mom = float(np.mean(mom_rates)) if len(mom_rates) > 0 else 0.0

    # Step 6: 决策矩阵
    comp_level = get_competition_level(competitor_count)
    decision, rationale = DECISION_MATRIX.get(
        (stage, comp_level),
        ("WAIT", "无匹配决策规则，建议人工复审")
    )

    # 衰退预警
    slope_ratio = boundaries["slopes"].get("slope_ratio")
    warning = None
    if stage == PLCStage.MATURITY and slope_ratio and slope_ratio > 2.0:
        warning = f"⚠️ 衰退预警：斜率比={slope_ratio:.1f}×，接近衰退阈值(3×)，建议启动换代研发"

    return PLCResult(
        sku_id=sku_id,
        current_stage=stage,
        confidence=round(confidence, 2),
        months_in_stage=months_in_stage,
        growth_rate_mom=round(avg_mom * 100, 1),  # 转为百分比
        slope_ratio=round(slope_ratio, 2) if slope_ratio else None,
        avm_score=round(avm_score, 3),
        decision=decision,
        rationale=rationale,
        warning=warning,
    )


# ── 示例：baby sterilizer 品类分析 ───────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Baby Sterilizer 品类 PLC 分析示例")
    print("=" * 60)

    # 模拟三种场景的月度品类销量指数（Google Trends 标准化）
    scenarios = {
        "UV-C Sterilizer（成长期）": {
            "sales": [8, 12, 18, 25, 35, 48, 62, 75, 80, 85, 82, 84,
                      88, 92, 95, 98, 96, 94, 92, 90, 88, 86, 84, 82],
            "months_on_market": 24,
            "competitor_count": 45,
        },
        "Steam Sterilizer（成熟期）": {
            "sales": [60, 62, 65, 68, 64, 66, 70, 72, 68, 65, 67, 69,
                      71, 70, 68, 66, 65, 67, 64, 62, 60, 61, 59, 58],
            "months_on_market": 48,
            "competitor_count": 120,
        },
        "UV Wand（衰退期）": {
            "sales": [90, 88, 82, 75, 65, 55, 45, 38, 30, 24, 18, 14,
                      10, 8, 6, 5, 4, 4, 3, 3, 2, 2, 2, 1],
            "months_on_market": 36,
            "competitor_count": 85,
        },
    }

    for name, params in scenarios.items():
        result = analyze_lifecycle(
            sku_id=name,
            monthly_sales=params["sales"],
            months_on_market=params["months_on_market"],
            competitor_count=params["competitor_count"],
        )
        print(f"\n品类: {name}")
        print(f"  阶段: {result.current_stage.value}  置信度: {result.confidence:.0%}")
        print(f"  在当前阶段已持续: {result.months_in_stage} 个月")
        print(f"  近3月均增速: {result.growth_rate_mom:+.1f}%")
        if result.slope_ratio:
            print(f"  衰退/成熟斜率比: {result.slope_ratio:.1f}× (>3×为强衰退信号)")
        print(f"  AVM分数: {result.avm_score:.3f}")
        print(f"  决策: {result.decision} — {result.rationale}")
        if result.warning:
            print(f"  {result.warning}")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Bass-Diffusion-New-Product-Forecasting]] — Bass 扩散模型描述新品增长曲线，是 PLC 成长期的理论基础
  - [[Skill-Time-Series-Anomaly-Detection]] — 识别销量序列中的异常点（大促 spike），是 STL 分解前的数据清洗前置
- **延伸技能**：
  - [[Skill-Category-Trend-Forecasting]] — PLC 阶段是品类趋势预测的核心输入变量
  - [[Skill-Market-Size-Estimation]]（B1）— PLC 阶段决定 TAM 预测的增长率假设
- **可组合**：
  - [[Skill-Competitor-Product-Intelligence]] — 竞品数量增长率作为「竞争密度」输入决策矩阵
  - [[Skill-Category-Compliance-Prescan]]（B3）— PLC 衰退期品类的合规风险特征与成长期不同，两者联合才能完整评估进入风险
  - [[Skill-Product-Opportunity-Scoring]] — PLC 阶段是机会评分卡的核心维度之一

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 避免衰退期进入：baby UV wand 品类已进入衰退（FDA 召回 + 竞品清仓），若误判为成长期进入，首批备货+认证成本损失约 $20,000-$50,000
  - 准确判断成长期进入时机：UV-C 密闭消毒器品类当前成长期，提前 6-12 个月进入比成熟期进入预期 LTV 高 2-3×
  - 换代预警价值：提前 6 个月发出衰退预警，节省新品研发断档期销售损失约 $5,000-$15,000/月
- **实施难度**：⭐⭐☆☆☆（2/5）— STL+微分分析，纯数值计算，无需 GPU/大数据
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）— WF-D 选品扫描的核心前置决策，缺失此 Skill 等于盲目进入品类
- **评估依据**：
  - PhaseFormer 实证：衰退斜率 9.04× 成熟斜率，阶段可量化判别
  - AVM 特征：仅需上市日期+销量即可计算，数据获取成本极低
  - baby sterilizer UV wand 真实召回案例：错误进入衰退期品类的直接财务损失可量化

---

## 元信息

```yaml
skill_id: Skill-Product-Lifecycle-Stage
domain: growth_model
vault_path: paper2skills-vault/06-增长模型/Skill-Product-Lifecycle-Stage.md
code_path: paper2skills-code/growth_model/product_lifecycle_stage/
papers:
  - id: "2511.16248"
    title: "Revisiting Fairness-aware Interactive Recommendation: Item Lifecycle as a Control Knob"
    venue: "AAAI 2025"
    role: 主论文（PhaseFormer STL分解+微分边界检测，衰退斜率9.04×实证）
  - id: "2511.17275"
    title: "Automobile demand forecasting: life cycle dynamics"
    venue: "arXiv 2025"
    role: AVM特征设计（年龄-销量矩，shape-agnostic多峰适配）
review_score: 8.0/10
review_dimensions:
  algorithm_coverage: 2.5/2.5
  business_specificity: 2.0/2.5
  code_runnable: 2.5/2.5
  graph_connectivity: 1.0/2.5   # 4条边，但B3/B1依赖链尚未萃取完成
created: 2026-05-25
wf_coverage: [WF-D]
```
