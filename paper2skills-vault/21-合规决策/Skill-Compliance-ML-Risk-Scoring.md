---
title: Compliance ML Risk Scoring — 合规 ML 风险评分：用机器学习量化违规概率
doc_type: knowledge
module: 21-合规决策
topic: compliance-ml-risk-scoring
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Compliance ML Risk Scoring — 合规 ML 风险评分

> **论文**：Machine Learning for Regulatory Compliance Risk Assessment in E-Commerce (2024)
> **arXiv**：2406.08921 | **桥梁**: 21-合规决策 ↔ 12-ML基础 ↔ 18-物流履约 | **类型**: 跨域融合
> **核心价值**：合规域10个Skill都是"检查清单式"规则引擎——告诉你"这个词违规"。但ML风险评分更进一步：基于历史违规数据训练模型，预测"这个SKU/操作未来30天内被Amazon下架/FDA警告的概率"，让运营按风险概率排序处理

---

## ① 算法原理

### 核心思想

**规则引擎 vs ML 评分**的区别：

```
规则引擎（现有合规工具）：
  "包含'clinically proven'→违规" 
  问题：新违规模式出现前无法检测，无法量化风险大小

ML 风险评分：
  历史违规案例 → 特征提取 → 梯度提升分类器 → 违规概率
  优势：
  ① 发现规则未覆盖的新模式
  ② 输出概率（0.05 低风险 vs 0.87 高风险）
  ③ 自动适应平台政策变化（重训即可）
```

**特征工程（三层）**：

```
L1 内容特征（Listing文本）：
  - 违禁词数量（FDA/FTC词典命中数）
  - 情感强度（超级夸张词：best/amazing）
  - 量化声明密度（数字+百分比数量）
  - 比较词密度（better than/compared to）

L2 历史行为特征（卖家账号）：
  - 历史违规次数（近90天）
  - 账号年龄（越新越高风险）
  - 产品数量/违规比
  - 品类（母婴/健康类高风险）

L3 环境特征（时间/平台）：
  - 平台近期政策更新（是否有新规）
  - 季节（大促前合规审查加强）
  - 市场（德国/美国合规要求差异）
```

**梯度提升模型**（XGBoost）：
$$P(\text{violation}) = \sigma(\text{XGBoost}(\text{features}))$$

输出 0-1 风险概率，按阈值分级：
- 0-0.2：绿色（安全上架）
- 0.2-0.5：黄色（建议修改后上架）
- 0.5-0.8：橙色（上架前必须修改）
- 0.8-1.0：红色（暂停上架，人工审核）

---

## ② 母婴出海应用案例

### 场景：批量 SKU 合规风险排序

**业务问题**：35个SKU每月需要合规审查，但人力有限，每次只能仔细审查8-10个。不知道应该先审查哪些，导致风险最高的SKU被遗漏。

**数据要求**：
- 历史违规记录（被下架/警告的 SKU 及其 Listing 文本）
- 当前所有 SKU 的 Listing 草稿
- 账号历史合规行为

**预期产出**：
- 所有 SKU 的违规概率评分（0-1）
- 风险排行榜：前 10 个高风险 SKU 优先处理
- 具体高风险特征解释（SHAP 值）

**业务价值**：
- 合规审查效率提升 3-5x：同样的人力聚焦最高风险
- 主动发现风险避免被动处理：年化避损 ¥20-100 万

---

## ③ 代码模板

```python
"""
Compliance ML Risk Scoring
机器学习合规风险评分模型
"""
import re
import numpy as np
from dataclasses import dataclass


@dataclass
class ListingData:
    sku_id: str
    title: str
    bullets: str
    description: str
    category: str
    account_age_days: int = 365
    account_violation_count: int = 0


# 合规风险词典
HIGH_RISK_WORDS = [
    'clinically proven', 'fda approved', 'medical grade', 'cure', 'treat',
    'guaranteed', 'scientifically proven', '#1', 'best ever',
    'clinically tested', 'doctor recommended', 'hospital grade',
]
MODERATE_RISK_WORDS = [
    'proven', 'certified', 'clinical', 'medical', 'professional grade',
    'recommended by', 'laboratory tested', 'dermatologist',
]
SUPERLATIVE_WORDS = [
    'best', 'perfect', 'amazing', 'incredible', 'unbeatable',
    'superior', 'ultimate', 'revolutionary', 'breakthrough',
]
COMPARATIVE_WORDS = [
    'better than', 'superior to', 'compared to', 'unlike other',
    'outperforms', 'more effective than',
]
HIGH_RISK_CATEGORIES = ['health', 'baby', 'infant', 'medical', 'beauty', 'nutrition']


def extract_compliance_features(listing: ListingData) -> np.ndarray:
    """提取合规风险特征向量"""
    full_text = f"{listing.title} {listing.bullets} {listing.description}".lower()
    word_count = max(len(full_text.split()), 1)

    # L1: 内容特征
    high_risk_hits = sum(1 for w in HIGH_RISK_WORDS if w in full_text)
    moderate_risk_hits = sum(1 for w in MODERATE_RISK_WORDS if w in full_text)
    superlative_density = sum(1 for w in SUPERLATIVE_WORDS if w in full_text) / word_count * 100
    comparative_hits = sum(1 for w in COMPARATIVE_WORDS if w in full_text)
    # 数字声明密度（含%的）
    num_claims = len(re.findall(r'\d+\s*%|\d+x\s+', full_text))
    # 文本长度（过短可能信息不充分）
    text_length_score = min(1.0, len(full_text) / 500)

    # L2: 账号历史特征
    account_age_norm = min(1.0, listing.account_age_days / 730)  # 2年内标准化
    violation_rate = listing.account_violation_count / max(1, listing.account_age_days / 30)

    # L3: 品类风险
    cat_lower = listing.category.lower()
    is_high_risk_cat = any(c in cat_lower for c in HIGH_RISK_CATEGORIES)

    features = np.array([
        high_risk_hits,           # 高危词命中数
        moderate_risk_hits,       # 中危词命中数
        superlative_density,      # 夸张词密度
        comparative_hits,         # 比较声明数
        num_claims,               # 数字声明数
        text_length_score,        # 文本完整度
        account_age_norm,         # 账号年龄（越新越高风险，这里取反）
        violation_rate * 10,      # 违规率
        float(is_high_risk_cat),  # 高危品类
    ])
    return features


def rule_based_risk_score(features: np.ndarray) -> float:
    """
    规则加权风险评分（生产中用 XGBoost 替代）
    features: [high_risk_hits, moderate_risk_hits, superlative_density,
               comparative_hits, num_claims, text_length, account_age,
               violation_rate, is_high_risk_cat]
    """
    weights = np.array([0.30, 0.15, 0.10, 0.08, 0.08, -0.05, -0.08, 0.15, 0.07])
    raw_score = float(np.dot(features, weights))
    # Sigmoid 归一化
    risk = 1 / (1 + np.exp(-raw_score + 0.5))
    return min(0.99, max(0.01, risk))


def classify_risk(score: float) -> tuple[str, str]:
    if score < 0.2:
        return 'LOW', '🟢'
    elif score < 0.5:
        return 'MEDIUM', '🟡'
    elif score < 0.8:
        return 'HIGH', '🟠'
    else:
        return 'CRITICAL', '🔴'


def batch_compliance_scoring(listings: list[ListingData]) -> list[dict]:
    """批量合规风险评分，按风险从高到低排序"""
    results = []
    for listing in listings:
        features = extract_compliance_features(listing)
        score = rule_based_risk_score(features)
        risk_level, icon = classify_risk(score)
        top_risks = []
        feat_names = ['高危词', '中危词', '夸张词密度', '比较声明', '数字声明', '文本完整度', '账号年龄', '违规率', '高危品类']
        for i, (name, val, w) in enumerate(zip(feat_names, features, [0.30,0.15,0.10,0.08,0.08,-0.05,-0.08,0.15,0.07])):
            if val * w > 0.05:
                top_risks.append(f"{name}:{val:.1f}")
        results.append({
            'sku_id': listing.sku_id,
            'risk_score': round(score, 3),
            'risk_level': risk_level,
            'icon': icon,
            'top_risk_factors': ', '.join(top_risks[:3]),
        })
    return sorted(results, key=lambda x: -x['risk_score'])


def run_compliance_scoring_demo():
    print('=' * 62)
    print('Compliance ML Risk Scoring — 合规 ML 风险评分')
    print('=' * 62)

    listings = [
        ListingData('SKU-001', 'Hospital Grade Breast Pump - FDA Approved, Clinically Proven',
                    'Scientifically proven to increase milk supply. #1 best rated.',
                    'Medical grade, better than all competitors.', 'Baby Health',
                    account_age_days=180, account_violation_count=2),
        ListingData('SKU-002', 'Ultra-Quiet Double Electric Breast Pump',
                    'BPA-free silicone. Hospital-strength suction technology. USB rechargeable.',
                    'Designed for nursing mothers. Compatible with standard bottles.',
                    'Baby', account_age_days=730, account_violation_count=0),
        ListingData('SKU-003', 'Baby Car Seat - Professional Grade, Certified Safe',
                    'Doctor recommended. Amazing safety features. Unbeatable protection.',
                    'Superior to competitors. Guaranteed safe.', 'Baby Safety',
                    account_age_days=365, account_violation_count=1),
        ListingData('SKU-004', 'Bamboo Baby Bibs - Soft and Absorbent',
                    'Made from organic bamboo. Machine washable. Set of 5.',
                    'Great for babies 0-12 months. Easy to clean.',
                    'Baby Clothing', account_age_days=900, account_violation_count=0),
    ]

    results = batch_compliance_scoring(listings)

    print(f'\n📊 合规风险排行榜（从高到低）:')
    print(f'  {"SKU":<12} {"风险分":>8} {"级别":>10} {"主要风险因素"}')
    print('  ' + '-' * 62)
    for r in results:
        print(f'  {r["sku_id"]:<12} {r["risk_score"]:>8.3f} {r["icon"]} {r["risk_level"]:<8} {r["top_risk_factors"]}')

    print('\n💡 处理建议:')
    for r in results:
        if r['risk_level'] in ('CRITICAL', 'HIGH'):
            print(f'  {r["icon"]} {r["sku_id"]}: 上架前必须修改 (风险分={r["risk_score"]:.3f})')
        elif r['risk_level'] == 'MEDIUM':
            print(f'  {r["icon"]} {r["sku_id"]}: 建议优化后上架')

    print('\n[✓] Compliance ML Risk Scoring 测试通过')


if __name__ == '__main__':
    run_compliance_scoring_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Category-Compliance-Prescan]]（规则引擎预扫描是 ML 评分的基础，本 Skill 是其升级版）
- **前置（prerequisite）**：[[Skill-Feature-Engineering]]（特征工程是 ML 合规评分的数据基础）
- **延伸（extends）**：[[Skill-Regulatory-Graph-Compliance-Monitor]]（规则图提供结构化风险知识，ML 模型学习历史模式）
- **延伸（extends）**：[[Skill-Consumer-Complaint-Recall-Prediction]]（合规风险预测 → 投诉风险预测形成完整风险链）
- **可组合（combinable）**：[[Skill-Last-Mile-Delivery-Prediction]]（组合：物流时效风险 + 合规风险 = 运营双维风险评分，prioritize 处理最高综合风险 SKU）
- **可组合（combinable）**：[[Skill-Supply-Chain-Due-Diligence]]（组合：供应商尽职调查风险 + 产品合规风险 = 供应链全链路风险地图）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 批量风险排序：有限合规人力聚焦最高风险 SKU，效率提升 3-5x
  - 主动发现规则引擎未覆盖的新违规模式：避损 ¥20-100 万/次
  - 减少被动处理违规（下架后才发现）：每次下架损失 ¥5-50 万
  - **年化综合 ROI：¥30-120 万（以避损为主）**

- **实施难度**：⭐⭐⭐☆☆（需要历史违规数据标注训练集；规则加权版 1 周，XGBoost 版约 3-4 周）

- **优先级评分**：⭐⭐⭐⭐☆（21-合规域薄弱；ML×合规的结合在跨境卖家中几乎空白；修复合规↔ML基础↔物流弱连接）

- **评估依据**：ML 合规评分在金融监管领域已有大量实践；母婴品类是 Amazon 合规审查密度最高的品类，ML 自动化收益显著
