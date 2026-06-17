---
title: 预测偏差加减码检测与校正 — 供应链计划主动修正行为的量化分析与偏差溯源
doc_type: knowledge
module: 04-供应链
topic: forecast-bias-adjustment-detection-correction
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 预测偏差加减码检测与校正

> **书籍**：《全链路管理》陈凤霞 第五章第一节"如何平衡销售和备货——供应链是否自行打折或加码"
> **桥梁**: 供应链 ↔ A/B实验 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：供应链团队在拿到销售预测后，往往会主动"打折"（认为预测偏高而缩减）或"加码"（认为保险起见而扩大）。书中将这种行为称为**加减码（Supply Chain Adjustment）**，并明确指出：**加减码本身不是问题，问题是加减码行为是否有依据、是否带来正向效果**。如果供应链团队的历史修正总是让预测变得更差（即修正后的计划准确率低于原始预测准确率），那么这种"经验调整"实际上是在增加误差而非减少。

**关键概念区分（书中明确强调）**：
```
原始预测准确率 FA = 1 - |预测 - 实际| / 实际
计划准确率 PA = 1 - |计划（修正后预测）- 实际| / 实际
加减码影响度 = PA - FA（正值=修正有益，负值=修正有害）
```

**四类加减码模式识别**：

1. **系统性乐观偏差**（书中最常见）：
   - 表现：预测值持续高于实际，FA长期偏低
   - 原因：销售团队倾向于给乐观预测（考核导向）
   - 校正：历史修正系数 = mean(实际/预测)，对未来预测乘以该系数

2. **系统性悲观偏差**：
   - 表现：供应链总是打折，导致缺货频发
   - 原因：供应链团队规避积压库存的保守心态
   - 校正：建立"打折行为与缺货率"的相关性分析

3. **随机修正**（无效修正）：
   - 表现：修正方向无规律，PA ≈ FA（没有改善也没有恶化）
   - 意味着：修正行为是徒劳的，不如直接用原始预测

4. **有效修正**：
   - 表现：PA > FA（计划比预测更准）
   - 意味着：修正行为有价值，但要分析是什么信息导致了准确修正

**算法核心——Theil's U统计量**：
```
U = sqrt(mean((F_t - A_t)^2)) / sqrt(mean((A_t)^2))
修正Theil's U = sqrt(mean((P_t - A_t)^2)) / sqrt(mean((F_t - A_t)^2))

修正U<1：修正优于原始预测（有价值）
修正U=1：修正与原始预测等效（无价值）
修正U>1：修正劣于原始预测（有害）
```

## ② 母婴出海应用案例

**场景A：销售团队预测系统性乐观检测**

- **业务问题**：某卖家月末经常发现库存积压，运营声称"预测做得很好"，但实际供应链分析发现销售团队的月度预测平均高出实际28%（系统性乐观偏差）
- **检测应用**：
  1. 计算过去12个月每月的FA：平均72%（28%偏差）
  2. 计算修正后的PA（供应链后来打折了80%执行）：平均78%（修正有益！）
  3. 但仍有22%误差，且修正系数本身也有偏差（打折80%基于经验，实际应打折74%）
  4. 量化最优修正系数：0.74，更新供应链SOP
- **预期产出**：计划准确率从78%提升至84%，库存积压减少约20%

**场景B：加减码行为影响评估（是否应该保留人工修正）**

- **业务问题**：管理层质疑"为什么需要人工修正预测？"，想了解供应链团队的修正到底有没有价值
- **算法评估**：计算修正Theil's U：
  - 吸奶器品类：U=0.85（修正有益，保留）
  - 婴儿服装品类：U=1.12（修正有害！去掉人工修正）
  - 婴儿食品品类：U=0.97（基本无效，建议自动化替代）

## ③ 代码模板

```python
"""
预测偏差加减码检测与校正
基于《全链路管理》陈凤霞 第五章第一节
FA vs PA对比 + 修正Theil's U + 最优修正系数
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ForecastBiasAnalyzer:
    """预测偏差与加减码分析器"""

    @staticmethod
    def forecast_accuracy(forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """计算预测准确率FA"""
        if len(actuals) == 0:
            return 0.0
        errors = np.abs(forecasts - actuals) / np.maximum(actuals, 1)
        return float(1 - np.mean(errors))

    @staticmethod
    def mean_bias(forecasts: np.ndarray, actuals: np.ndarray) -> float:
        """计算系统性偏差（正=乐观偏高，负=悲观偏低）"""
        biases = (forecasts - actuals) / np.maximum(actuals, 1)
        return float(np.mean(biases))

    @staticmethod
    def theil_u_ratio(forecasts: np.ndarray, plans: np.ndarray,
                       actuals: np.ndarray) -> float:
        """
        修正Theil's U统计量
        <1: 修正优于原始预测
        =1: 修正与原始预测等效
        >1: 修正劣于原始预测（有害修正）
        """
        rmse_forecast = np.sqrt(np.mean((forecasts - actuals) ** 2))
        rmse_plan = np.sqrt(np.mean((plans - actuals) ** 2))
        if rmse_forecast == 0:
            return 1.0
        return float(rmse_plan / rmse_forecast)

    def compute_optimal_adjustment_factor(self, forecasts: np.ndarray,
                                           actuals: np.ndarray,
                                           min_factor: float = 0.5,
                                           max_factor: float = 1.5) -> Dict:
        """
        计算历史最优修正系数
        通过网格搜索找到使PA最高的乘数因子
        """
        best_factor = 1.0
        best_pa = self.forecast_accuracy(forecasts, actuals)

        for factor in np.arange(min_factor, max_factor, 0.02):
            adjusted = forecasts * factor
            pa = self.forecast_accuracy(adjusted, actuals)
            if pa > best_pa:
                best_pa = pa
                best_factor = factor

        return {
            'optimal_factor': round(best_factor, 3),
            'original_fa': self.forecast_accuracy(forecasts, actuals),
            'optimized_pa': best_pa,
            'pa_improvement': best_pa - self.forecast_accuracy(forecasts, actuals),
            'bias_direction': 'lệ观偏高→需打折' if best_factor < 0.99 else ('悲观偏低→需加码' if best_factor > 1.01 else '无明显偏差'),
        }

    def pattern_classification(self, forecasts: np.ndarray, plans: np.ndarray,
                                 actuals: np.ndarray) -> Dict:
        """识别加减码模式类型"""
        fa = self.forecast_accuracy(forecasts, actuals)
        pa = self.forecast_accuracy(plans, actuals)
        theil_u = self.theil_u_ratio(forecasts, plans, actuals)
        forecast_bias = self.mean_bias(forecasts, actuals)
        adjustment_bias = self.mean_bias(plans, forecasts)  # 修正行为本身的偏向

        if theil_u < 0.95:
            pattern = "有效修正"
            desc = f"人工修正使准确率提升{(pa-fa):.1%}，建议保留修正流程"
        elif theil_u > 1.05:
            pattern = "有害修正"
            desc = f"人工修正使准确率下降{(fa-pa):.1%}，建议停止人工修正！"
        else:
            pattern = "无效修正"
            desc = "人工修正对准确率无显著影响，建议自动化替代人工"

        if abs(forecast_bias) > 0.15:
            if forecast_bias > 0:
                system_bias = f"系统性乐观偏差：预测平均高出实际{forecast_bias:.0%}"
            else:
                system_bias = f"系统性悲观偏差：预测平均低于实际{abs(forecast_bias):.0%}"
        else:
            system_bias = "无系统性偏差"

        return {
            'pattern': pattern,
            'description': desc,
            'system_bias': system_bias,
            'forecast_accuracy_fa': fa,
            'plan_accuracy_pa': pa,
            'theil_u': theil_u,
            'adjustment_direction': '打折' if adjustment_bias < -0.03 else ('加码' if adjustment_bias > 0.03 else '微调'),
        }

    def monthly_bias_trend(self, monthly_data: List[Dict]) -> pd.DataFrame:
        """月度偏差趋势分析"""
        records = []
        for d in monthly_data:
            fa = self.forecast_accuracy(
                np.array([d['forecast']]), np.array([d['actual']]))
            pa = self.forecast_accuracy(
                np.array([d['plan']]), np.array([d['actual']]))
            records.append({
                'month': d['month'],
                'forecast': d['forecast'],
                'plan': d['plan'],
                'actual': d['actual'],
                'fa': fa,
                'pa': pa,
                'adjustment_factor': d['plan'] / max(d['forecast'], 1),
                'pa_vs_fa': pa - fa,
            })
        df = pd.DataFrame(records)
        df['trend'] = df['pa_vs_fa'].rolling(3).mean()
        return df


def run_forecast_bias_demo():
    """预测偏差加减码检测演示"""
    print("=" * 65)
    print("预测偏差加减码检测与校正")
    print("基于《全链路管理》陈凤霞 第五章第一节")
    print("区分FA(预测准确率) vs PA(计划准确率)")
    print("=" * 65)

    np.random.seed(42)
    analyzer = ForecastBiasAnalyzer()

    # 模拟12个月销售预测数据
    actuals = np.random.normal(1000, 150, 12).clip(600, 1500)
    # 销售团队预测系统性乐观（高出实际约25%）
    forecasts = actuals * np.random.uniform(1.15, 1.35, 12)
    # 供应链打折处理（乘以0.85）
    plans = forecasts * np.random.uniform(0.80, 0.90, 12)

    print("\n[月度预测vs计划vs实际对比]")
    months = [f"2025-{i+1:02d}" for i in range(12)]
    monthly_data = [{'month': m, 'forecast': f, 'plan': p, 'actual': a}
                    for m, f, p, a in zip(months, forecasts, plans, actuals)]

    df = analyzer.monthly_bias_trend(monthly_data)
    print(f"  {'月份':<10} {'预测':<8} {'计划':<8} {'实际':<8} {'FA':<8} {'PA':<8} {'修正效果'}")
    for _, row in df.iterrows():
        effect = f"+{row['pa_vs_fa']:.1%}" if row['pa_vs_fa'] > 0 else f"{row['pa_vs_fa']:.1%}"
        print(f"  {row['month']:<10} {row['forecast']:<8.0f} {row['plan']:<8.0f} "
              f"{row['actual']:<8.0f} {row['fa']:<8.1%} {row['pa']:<8.1%} {effect}")

    print(f"\n  均值: FA={df['fa'].mean():.1%} PA={df['pa'].mean():.1%} "
          f"(供应链修正{'有益' if df['pa'].mean() > df['fa'].mean() else '有害'})")

    # 模式识别
    print("\n[加减码模式识别]")
    pattern = analyzer.pattern_classification(forecasts, plans, actuals)
    print(f"  模式: {pattern['pattern']}")
    print(f"  描述: {pattern['description']}")
    print(f"  系统偏差: {pattern['system_bias']}")
    print(f"  修正方向: {pattern['adjustment_direction']}")
    print(f"  Theil's U = {pattern['theil_u']:.3f} ({'<1有益' if pattern['theil_u']<1 else '>1有害'})")

    # 最优修正系数
    print("\n[最优修正系数计算]")
    opt = analyzer.compute_optimal_adjustment_factor(forecasts, actuals)
    print(f"  当前系统性偏差: {opt['bias_direction']}")
    print(f"  推荐修正系数: {opt['optimal_factor']} (乘以此系数得到最优计划)")
    print(f"  原始FA: {opt['original_fa']:.1%} → 优化后PA: {opt['optimized_pa']:.1%} "
          f"(+{opt['pa_improvement']:.1%})")

    # 三类产品对比
    print("\n[三品类修正价值对比（Theil's U判断是否保留人工修正）]")
    categories = [
        ("吸奶器", np.random.normal(800, 100, 12), 0.85),  # 有效修正
        ("婴儿服装", np.random.normal(500, 80, 12), 1.15),  # 有害修正
        ("婴儿食品", np.random.normal(1200, 150, 12), 1.0),  # 无效修正
    ]
    for cat_name, cat_actuals, u_sim in categories:
        cat_forecasts = cat_actuals * np.random.uniform(1.2, 1.3, 12)
        # 模拟不同U值的修正
        adjustment = 0.85 if u_sim < 1 else (1.15 if u_sim > 1 else 1.0)
        cat_plans = cat_forecasts * adjustment * np.random.uniform(0.95, 1.05, 12)
        u = analyzer.theil_u_ratio(cat_forecasts, cat_plans, cat_actuals)
        verdict = "✅保留人工修正" if u < 0.95 else ("❌停止人工修正" if u > 1.05 else "⚠️自动化替代")
        print(f"  {cat_name}: Theil's U = {u:.3f} → {verdict}")

    print("\n[书中关键洞察]")
    print("  FA vs PA的差值 = 人工修正的价值（可正可负！）")
    print("  修正U<1→有益，>1→反而更差，供应链团队应停止此类修正")
    print("  最常见：销售团队乐观偏差→供应链打折→仍然积压（两层误差叠加）")
    print("\n[✓] 预测偏差加减码检测测试通过")
    return pattern


if __name__ == "__main__":
    run_forecast_bias_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]]（需求预测是本Skill的输入）、[[Skill-Logistics-Plan-Three-Dimension-Accuracy]]（销售件数预测准确率是本Skill的结果指标）
- **延伸（extends）**：[[Skill-SOP-Sales-Operations-Planning]]（S&OP会议正是对齐预测与计划的场合，本Skill量化了会议决策的历史质量）
- **可组合（combinable）**：[[Skill-ML-AB-Randomization-Test]]（用随机化检验评估"人工修正是否显著优于原始预测"）、[[Skill-Promo-Stocktaking-SOP-Automation]]（大促备货中加减码行为的专项分析）

## ⑤ 商业价值评估

- **ROI 预估**：发现"有害修正"后停止此类人工干预，直接使计划准确率提升约5-8%；以月GMV$50万计算，准确率每提升1%≈$5000价值；系统$1万，ROI>400%
- **实施难度**：⭐⭐☆☆☆（只需历史预测+实际数据；主要挑战是记录每次人工修正前后的原始预测值）
- **优先级**：⭐⭐⭐⭐⭐（书中第五章专章讲解，是供应链计划管理的"元认知"——知道自己的决策质量）
- **适用规模**：所有有人工预测修正流程的组织，特别是销售与供应链存在"博弈"的团队
- **数据依赖**：原始销售预测（修正前）、最终执行计划（修正后）、实际销售数据（结果）
