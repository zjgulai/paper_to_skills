# Skill Card: Compliant Dynamic Pricing Guard（合规-定价双约束优化）

> **桥梁**: 17-价格优化 ↔ 21-合规决策 | **类型**: 跨域融合

roadmap_phase: phase1
---

## ① 算法原理

**核心思想**：跨境动态定价不能只优化利润，必须同时满足多个合规约束，违规定价可能触发 MAP 违规、Amazon 最低价政策、或反倾销诉讼。

**双约束优化框架**：
```
目标: maximize GMV(p) × margin(p)
约束集合:
  1. MAP约束: p ≥ MAP_price (品牌最低广告价格)
  2. Amazon最低价: p ≥ Amazon_floor (历史最低 - X%)
  3. 关税传导: p ≥ cost × (1 + tariff_rate) × min_margin
  4. 市场协调: |p_US - p_EU × fx| ≤ max_price_gap
  5. 倾销防护: p ≥ home_market_price × (1 - anti_dumping_threshold)
```

**约束激活检测**：
```python
def check_constraint_violations(price, product_config):
    violations = []
    if price < product_config['map_price']:
        violations.append(('MAP', f"${price:.2f} < MAP ${product_config['map_price']:.2f}"))
    if price < product_config['amazon_floor']:
        violations.append(('AMZN_FLOOR', f"Below Amazon price floor"))
    # ... 其他约束
    return violations
```

**合规安全边际（Safety Margin）**：
- 在每个约束上保留 3-5% 缓冲，避免边界处的合规风险
- 触发合规约束时自动降级到保守定价策略

---

## ② 母婴出海应用案例

**业务问题**：某母婴品牌同时在 Amazon US/EU/JP 销售，AI 动态定价引擎在大促期间将美国售价压到了比欧洲便宜 35%，触发了品牌 MAP 协议，并引起了欧洲经销商的投诉。

**应用流程**：
1. 建立价格约束数据库（MAP价格/历史价格/关税系数）
2. 在 AIGP 动态定价引擎外层包装合规护栏
3. 每次调价前运行约束检查，违规则回退到 base_price + 安全边际
4. 记录约束激活日志，供合规审查

**量化收益**：
- 消除 MAP 违规风险（一次 MAP 处罚：暂停账号 30-90 天，损失 150-500 万 GMV）
- 跨市场价格差异保持在 15% 以内，防止灰色进口
- 关税调整自动传导到定价，不再依赖人工调整（节省 2-3 天响应时间）

---

## ③ 代码模板

```python
from dataclasses import dataclass
from typing import Optional
import logging

@dataclass
class PriceConstraints:
    map_price: float           # 品牌最低广告价格
    amazon_floor: float        # Amazon 价格底线（历史最低 × 0.97）
    min_margin_ratio: float    # 最低毛利率（如 0.15 = 15%）
    max_cross_market_gap: float = 0.20  # 跨市场最大价差 20%
    safety_margin: float = 0.03         # 合规安全边际 3%

def compliant_price(
    proposed_price: float,
    constraints: PriceConstraints,
    cost: float,
    market: str = "US",
) -> tuple[float, list[str]]:
    """
    返回合规后的最终价格 + 触发的约束列表。
    """
    final_price = proposed_price
    triggered = []
    
    # MAP 约束
    map_floor = constraints.map_price * (1 + constraints.safety_margin)
    if final_price < map_floor:
        final_price = map_floor
        triggered.append(f"MAP: raised to ${map_floor:.2f}")
    
    # Amazon 价格底线
    if final_price < constraints.amazon_floor:
        final_price = constraints.amazon_floor * (1 + constraints.safety_margin)
        triggered.append(f"AMZN_FLOOR: raised to ${final_price:.2f}")
    
    # 最低毛利约束
    min_price = cost / (1 - constraints.min_margin_ratio)
    if final_price < min_price:
        final_price = min_price
        triggered.append(f"MIN_MARGIN: raised to ${min_price:.2f}")
    
    if triggered:
        logging.warning(f"[Compliance] Market={market}, Proposed={proposed_price:.2f}, Final={final_price:.2f}, Constraints={triggered}")
    
    return final_price, triggered

# 与 AIGP 动态定价集成
def aigp_with_compliance(product_id: str, base_price: float, constraints: PriceConstraints) -> float:
    # 1. AIGP 生成建议价格
    aigp_price = base_price * 0.92  # 模拟：建议降价 8%
    
    # 2. 合规护栏过滤
    final_price, violations = compliant_price(aigp_price, constraints, cost=base_price*0.4)
    
    return final_price

# 测试
constraints = PriceConstraints(map_price=19.99, amazon_floor=17.99, min_margin_ratio=0.20)
price, violations = compliant_price(17.50, constraints, cost=8.00)
print(f"最终合规价格: ${price:.2f}, 触发约束: {violations}")
assert price >= constraints.map_price, "MAP 约束未生效"
print("[✓] Compliant Dynamic Pricing Guard 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-AIGP-LLM-Dynamic-Pricing]] (17) | [[Skill-Cross-Border-Price-Harmonization]] (17)
- **组合**：[[Skill-Cross-Border-Compliance-Framework]] (21) | [[Skill-Regulatory-Change-Monitoring]] (21)
- **延伸**：[[Skill-Competitive-Price-Monitoring]] (17)

---

## ⑤ 商业价值

- **ROI**：预防一次 MAP 违规暂停 = 150-500 万 GMV 保护
- **难度**：⭐⭐⭐☆☆（技术集成难度中等）
- **优先级**：⭐⭐⭐⭐⭐（有动态定价 + 多市场的品牌必须配置）
- **适用场景**：跨境多市场动态定价、大促期间自动调价、关税冲击后价格传导
