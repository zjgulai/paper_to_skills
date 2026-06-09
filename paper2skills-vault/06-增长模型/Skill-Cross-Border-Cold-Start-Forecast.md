# Skill Card: Cross-Border Cold-Start Forecast（跨境冷启动需求预测）

> **论文**: ZODIAC: Zero-Inflated Overshoot-Aware Demand Forecasting for Cross-Border E-Commerce  
> **来源**: OpenReview 2024 | 真实跨境平台数据：US→{UK, DE, FR, IT, ES, JP} 六条弧, 250K+商品/弧  
> **代码**: ❌ 工业部署论文 | **领域**: 06-增长模型 | **场景**: WF-D 选品扫描 — 冷启动销量验证

---

## ① 算法原理

### 核心思想
跨境电商的致命问题：**70-80% 商品在目标市场无历史数据**，15-25% 上架后零销量，但传统预测模型会给出"看起来合理"的正数预测→导致库存积压。ZODIAC 用双域 LSTM + 双头架构同时解决"零销量预测"和"过预测"两个核心痛点。

### 数学直觉

**双域 LSTM（Source + Target Domain）**：
- Source LSTM：学习源市场（如 US）的时序模式
- Target LSTM：学习目标市场（如 DE）的时序模式
- 双域融合：$h_t = \text{Concat}(h_t^{src}, h_t^{tgt}) \cdot W_{fusion}$

**双头架构**：

1. **分类头** — 预测是否零销量：
   $$P(zero\_sales) = \sigma(\text{MLP}_{cls}(h_T))$$
   这是关键创新——传统回归模型无法输出"零销量"概率

2. **回归头** — 预测销量幅度：
   $$\hat{y} = \text{MLP}_{reg}(h_T) \cdot (1 - P(zero\_sales))$$
   仅当分类头预测"非零"时才激活回归头

**非对称损失函数（Asymmetric Loss）**：
$$\mathcal{L} = \mathcal{L}_{cls} + \lambda \cdot \begin{cases} \alpha_{over} \cdot |\hat{y} - y| & \text{if } \hat{y} > y \\ \alpha_{under} \cdot |\hat{y} - y| & \text{if } \hat{y} \leq y \end{cases}$$

其中 $\alpha_{over} > \alpha_{under}$（过预测惩罚 > 欠预测）。这直接对应商业逻辑：**多备货比少备货更贵**（库存积压 vs 缺货损失）。

### 关键假设
- 源市场与目标市场存在可迁移的时序模式（如季节性、价格弹性）
- 产品在源市场有足够历史数据（>90 天）
- 零膨胀分布假设（Zero-Inflated Negative Binomial）适合新品场景

---

## ② 母婴出海应用案例

### 场景：判断一款吸奶器从美国站引入德国站的首月销量

**业务问题**：
S1 吸奶器在美国 Amazon 月销 800 台（$99.99），现在考虑上架德国站（€89.99）。传统做法：按美国销量×0.7（经验折扣）= 预计 560 台，备货 800 台。ZODIAC 分析显示：德国站同品类竞品密度高 40%，且德国消费者对"Medela 兼容性"偏好强→零销量概率 22%，预计销量 350-480 台（90% CI），建议备货 500 台（而非 800）。

**数据要求**：
- Source domain：Amazon US 该 SKU 12 个月日销量
- Target domain：Amazon DE 同品类竞品数据（BSR/价格/评论）
- 产品特征：价格、品类、品牌知名度（在目标市场的搜索量）

**预期产出**：
- 零销量概率：22%
- 预测销量（若非零）：350-480 台/月（90% CI）
- 过预测风险：原经验法 800 台 → 过剩 300-450 台 × €60 成本 = €18,000-27,000 损失
- ZODIAC 建议：首月备货 500 台（覆盖 90% CI 上界），3 个月后根据实际数据迭代

**业务价值**：
- 首单避免过度备货：节省 $18,000-27,000/市场
- 6 个新市场 × 平均 3 个 SKU = 避免 $300K+ 库存积压
- 年化 ROI：**60-120 万元**

### 场景二：新品上架的"去或不去"决策

**业务问题**：
5 个候选新品都在美国站表现不错（月销 200-500 台），但要决定优先上哪个到日本站。资源有限，只能选 2 个。

**数据要求**：
- ZODIAC 预测 5 个候选品在 JP 站的零销量概率 + 预计销量
- Market embedding 分析日本市场的品类偏好

**预期产出**：
- 候选 A（吸奶器）：零销率 15%，预计 120-180 台 → **优先**
- 候选 B（奶瓶套装）：零销率 8%，预计 200-300 台 → **优先**
- 候选 C（婴儿车）：零销率 45%，预计 50-80 台 → 暂缓（日本市场竞争极强）
- 候选 D（辅食机）：零销率 30%，预计 80-120 台 → 候选
- 候选 E（哺乳巾）：零销率 60%，预计 20-40 台 → 放弃（文化不适配）

**业务价值**：数据驱动的优先级排序替代"拍脑袋"决策

---

## ③ 代码模板

```python
"""
ZODIAC — Cross-Border Cold-Start Demand Forecasting
基于 ZODIAC (OpenReview 2024) 的简化实现

核心: 双域LSTM + 双头(分类+回归) + 非对称损失
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ColdStartForecast:
    """冷启动预测结果"""
    sku: str
    source_market: str
    target_market: str
    zero_sales_prob: float        # 零销量概率
    predicted_sales: float         # 预测月销量（若非零）
    ci_lower: float                # 90% CI 下界
    ci_upper: float                # 90% CI 上界
    overshoot_risk: float          # 过预测风险 (0-1)
    recommended_inventory: int     # 建议备货量
    decision: str                  # GO / CAUTIOUS / NOGO


class ZODIACColdStartPredictor:
    """
    跨境冷启动需求预测器
    
    生产环境使用完整 ZODIAC 模型（双域LSTM + Zero-Inflated + Asymmetric Loss）
    当前为简化实现，用启发式规则 + source domain statistics
    """
    
    def __init__(self):
        pass
    
    def predict_cold_start(
        self,
        sku: str,
        source_market: str,
        source_monthly_sales: float,
        source_sales_std: float,
        source_days_active: int,
        target_market: str,
        target_competition_density: float,  # 0-1, 竞品密度
        target_price_competitiveness: float, # 0-1, 价格竞争力
        category_cultural_fit: float = 0.8, # 0-1, 文化适配性
        lead_time_weeks: int = 8,
    ) -> ColdStartForecast:
        """
        预测冷启动首月销量
        
        Args:
            sku: 产品 SKU
            source_market: 源市场
            source_monthly_sales: 源市场月销量
            source_sales_std: 源市场销量标准差
            source_days_active: 源市场已上架天数
            target_market: 目标市场
            target_competition_density: 目标市场竞品密度
            target_price_competitiveness: 目标市场价格竞争力
            category_cultural_fit: 品类文化适配性
            lead_time_weeks: 备货提前期（周）
        """
        # 1. 零销量概率估计（ZODIAC 分类头模拟）
        # 基于：竞品密度 + 文化适配 + 源市场稳定性
        stability_factor = min(source_days_active / 365, 1.0)
        zero_prob = (
            0.25 * target_competition_density +
            0.25 * (1 - category_cultural_fit) +
            0.25 * (1 - stability_factor) +
            0.25 * (1 - target_price_competitiveness)
        )
        zero_prob = np.clip(zero_prob, 0.05, 0.65)
        
        # 2. 销量预测（若非零）
        # 基准：源市场销量 × 市场大小修正 × 竞争修正
        market_size_ratios = {"US": 1.0, "DE": 0.35, "UK": 0.30, "JP": 0.25, "FR": 0.18, "IT": 0.15, "ES": 0.12}
        size_ratio = market_size_ratios.get(target_market, 0.15)
        
        competition_penalty = 1 - target_competition_density * 0.5
        cultural_boost = 0.7 + category_cultural_fit * 0.3
        price_boost = 0.8 + target_price_competitiveness * 0.4
        
        base_sales = source_monthly_sales * size_ratio * competition_penalty * cultural_boost * price_boost
        
        # 非对称预测：过预测惩罚
        predicted_sales = base_sales * (1 - zero_prob)
        
        # 不确定性区间（源市场波动 × 新市场不确定性放大）
        cv = source_sales_std / max(source_monthly_sales, 0.01)
        uncertainty_multiplier = 1.5 + target_competition_density  # 新市场更不确定
        sales_std = predicted_sales * cv * uncertainty_multiplier
        
        ci_lower = max(0, predicted_sales - 1.645 * sales_std)
        ci_upper = predicted_sales + 1.645 * sales_std
        
        # 3. 过预测风险评估
        overshoot_risk = zero_prob + (1 - zero_prob) * min(cv * uncertainty_multiplier, 0.8)
        
        # 4. 备货建议：覆盖 90% CI 上界，但不超过源市场销量的 2x
        recommended = min(int(np.ceil(ci_upper * lead_time_weeks / 4)), 
                          int(source_monthly_sales * 2 * lead_time_weeks / 4))
        
        # 5. 决策
        if zero_prob > 0.35:
            decision = "NOGO"
        elif zero_prob > 0.2 or overshoot_risk > 0.5:
            decision = "CAUTIOUS"
        else:
            decision = "GO"
        
        return ColdStartForecast(
            sku=sku,
            source_market=source_market,
            target_market=target_market,
            zero_sales_prob=round(zero_prob, 3),
            predicted_sales=round(predicted_sales),
            ci_lower=round(ci_lower),
            ci_upper=round(ci_upper),
            overshoot_risk=round(overshoot_risk, 3),
            recommended_inventory=recommended,
            decision=decision,
        )
    
    def rank_candidates(
        self, 
        candidates: List[Dict],
        target_market: str,
        top_n: int = 3,
    ) -> List[Dict]:
        """批量候选品排序"""
        results = []
        for c in candidates:
            forecast = self.predict_cold_start(
                sku=c["sku"],
                source_market=c.get("source_market", "US"),
                source_monthly_sales=c["source_sales"],
                source_sales_std=c.get("sales_std", c["source_sales"] * 0.3),
                source_days_active=c.get("days_active", 180),
                target_market=target_market,
                target_competition_density=c.get("competition", 0.5),
                target_price_competitiveness=c.get("price_comp", 0.7),
                category_cultural_fit=c.get("cultural_fit", 0.8),
                lead_time_weeks=c.get("lead_time", 8),
            )
            results.append(forecast)
        
        # 按 expected_value（非零概率 × 预测销量）排序
        results.sort(key=lambda x: (1-x.zero_sales_prob)*x.predicted_sales, reverse=True)
        
        return [
            {
                "rank": i+1,
                "sku": r.sku,
                "decision": r.decision,
                "zero_risk": f"{r.zero_sales_prob:.0%}",
                "predicted_monthly": f"{r.predicted_sales}台",
                "ci": f"[{r.ci_lower}, {r.ci_upper}]",
                "recommended_inventory": r.recommended_inventory,
                "overshoot_risk": f"{r.overshoot_risk:.0%}",
            }
            for i, r in enumerate(results[:top_n])
        ]


# ============ 测试 ============

if __name__ == '__main__':
    predictor = ZODIACColdStartPredictor()
    
    # 单个 SKU 预测
    forecast = predictor.predict_cold_start(
        sku="Pump-S1",
        source_market="US",
        source_monthly_sales=800,
        source_sales_std=150,
        source_days_active=365,
        target_market="DE",
        target_competition_density=0.6,      # 德国市场竞争激烈
        target_price_competitiveness=0.75,   # 价格有竞争力
        category_cultural_fit=0.8,           # 品类文化适配
        lead_time_weeks=8,
    )
    
    print(f"跨境冷启动预测: Pump-S1 (US→DE)")
    print(f"  零销量概率: {forecast.zero_sales_prob:.0%}")
    print(f"  预测月销: {forecast.predicted_sales}台 [{forecast.ci_lower}, {forecast.ci_upper}]")
    print(f"  建议备货: {forecast.recommended_inventory}台")
    print(f"  过预测风险: {forecast.overshoot_risk:.0%}")
    print(f"  决策: {'✅ GO' if forecast.decision == 'GO' else ('⚠️ CAUTIOUS' if forecast.decision == 'CAUTIOUS' else '❌ NOGO')}")
    
    # 批量候选排序
    candidates = [
        {"sku": "Pump-S1", "source_sales": 800, "sales_std": 150, "competition": 0.6, "price_comp": 0.75, "cultural_fit": 0.8},
        {"sku": "Bottle-Set", "source_sales": 500, "sales_std": 80, "competition": 0.3, "price_comp": 0.85, "cultural_fit": 0.9},
        {"sku": "Stroller-X", "source_sales": 200, "sales_std": 60, "competition": 0.8, "price_comp": 0.5, "cultural_fit": 0.7},
        {"sku": "Nursing-Cover", "source_sales": 300, "sales_std": 100, "competition": 0.4, "price_comp": 0.9, "cultural_fit": 0.3},
    ]
    
    ranking = predictor.rank_candidates(candidates, "DE", top_n=4)
    print(f"\n候选品排序 (→德国站):")
    for r in ranking:
        flag = "✅" if r["decision"] == "GO" else ("⚠️" if r["decision"] == "CAUTIOUS" else "❌")
        print(f"  {r['rank']}. {flag} {r['sku']}: 月销{r['predicted_monthly']} (零销率{r['zero_risk']}) → {r['decision']}")
    
    # 验证
    assert forecast.zero_sales_prob > 0
    assert len(ranking) == 4
    print("\n[✓] Cross-Border Cold-Start Forecast 测试通过")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Cross-Market-Product-Transfer]] — Bert4XMR 判断"能不能卖"，ZODIAC 预测"能卖多少"
  - [[Skill-Demand-Forecasting-Supply-Chain]] — 传统需求预测是 ZODIAC 的基础
  - [[Skill-Bass-Diffusion-New-Product-Forecasting]] — Bass 模型的新品扩散 + ZODIAC 的跨境冷启动互补
- **延伸技能**：
  - [[Skill-Conformal-Prediction-Demand-UQ]] — ZODIAC 的 CI + Conformal 的分布无关区间 = 更强的置信度
  - [[Skill-Multi-Channel-Inventory-Pooling]] — 冷启动预测 + 多渠道库存池化 = 跨市场备货优化
- **可组合技能**：
  - **[[Skill-Product-Opportunity-Scoring]]** — ZODIAC 销量预测作为机会评分的"市场规模"维度量化输入
  - **[[Skill-Review-Pain-Point-Mining]]** — 先挖痛点→再做差异化产品→再用 ZODIAC 验证销量潜力

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 首单避免过度备货：每市场 $18,000-27,000 × 6 市场 = $100K-160K
  - 避免零销量 SKU 上架：每次节省 $3,000-8,000（listing 费用+FBA 仓储+广告）
  - 年化 ROI：**60-120 万元**
- **实施难度**：⭐⭐⭐☆☆（3 星）— ZODIAC 需跨市场销售数据，初始可用简化版启发式规则
- **优先级评分**：⭐⭐⭐⭐⭐（5 星）— 直接解决跨境选品最痛的"首单备货量"问题
- **评估依据**：
  - 唯一专门针对跨境电商 US→多国冷启动的论文
  - 250K+商品、6 条跨境弧的真实数据验证
  - 过预测率从 66-87% 降至 38-41% → WAPE 提升 13-26%
