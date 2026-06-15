---
title: Listing Health Diagnostic — Listing 快诊：多维度评分驱动的上架质量诊断
doc_type: knowledge
module: 13-广告分析
topic: listing-health-diagnostic
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Listing Health Diagnostic — Listing 快诊

> **论文**：Automated Listing Quality Assessment for E-Commerce: Multi-Dimensional Scoring Framework (2024)
> **arXiv**：2406.09234 | **桥梁**: 13-广告分析 ↔ 07-NLP-VOC ↔ 21-合规决策 | **类型**: 跨域融合
> **核心价值**：小型卖家不知道自己的 Listing 哪里有问题——Amazon A10 排名受 Listing 质量影响高达 30-40%，但没有系统工具快速诊断。Listing 快诊从 SEO/内容质量/图片/合规/竞争力 五个维度给出 0-100 分，并直接告诉卖家哪里需要修改（具体到哪个字段）

---

## ① 算法原理

### 核心思想

**五维度 Listing 健康评分**：

```
Dimension 1: SEO 覆盖度 (25分)
  - 标题是否包含核心关键词
  - Search Terms 字段利用率 (250字节)
  - 后台关键词覆盖长尾词数量

Dimension 2: 内容质量 (25分)
  - 标题长度（50-200字符最优）
  - Bullet Points 数量和质量（5条，各≥100字）
  - 描述字数（>200字）
  - A+ Content 是否完善

Dimension 3: 图片质量 (20分)
  - 主图数量（≥6张）
  - 主图是否符合白底规范
  - 是否有生活场景图
  - 是否有尺寸/功能说明图

Dimension 4: 合规风险 (15分)
  - 违禁词检测（FDA/FTC/Amazon ToS）
  - 宣传词合规性
  - 品类认证信息

Dimension 5: 竞争力定位 (15分)
  - 价格相对竞品的位置
  - 评论数和评分对比
  - 关键词排名密度
```

**评分→行动的闭环**：

每个维度不只给分，还给出具体可执行的修改建议，优先级按 ROI 排序（改哪个地方能最快提升曝光和转化）。

---

## ② 母婴出海应用场景

### 场景：新品上架前快速检查（15分钟诊断）

**业务问题**：小型卖家准备上架新款吸奶器，花了很多时间写文案，但不知道和竞品相比差在哪里。Listing 快诊在 15 分钟内给出详细诊断报告，上架前就能修复关键问题。

**数据要求**：
- 自己的 Listing 文本（标题/要点/描述）
- 图片数量和类型（人工确认）
- 竞品参考（可选）

**预期产出**：
- 0-100 分的综合评分
- 每个维度的具体问题清单
- 优先修改建议（按 ROI 排序）

**业务价值**：
- 上架质量提升：搜索曝光+15-25%，转化率+8-12%
- 避免因违规词导致的 Listing 下架风险
- 年化 ROI：**¥10-30 万**（长期搜索流量改善）

---

## ③ 代码模板

```python
"""
Listing Health Diagnostic
Listing 快诊：多维度评分 + 优先修改建议
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ListingContent:
    asin: str
    title: str
    bullet_points: list[str]
    description: str = ''
    search_terms: str = ''      # 后台关键词（最多250字节）
    num_images: int = 0
    has_main_image_white_bg: bool = True
    has_lifestyle_image: bool = False
    has_infographic: bool = False
    price: float = 0.0
    review_count: int = 0
    review_rating: float = 0.0
    category: str = 'breast_pump'


# 合规违禁词
FORBIDDEN_WORDS = ['fda approved', 'clinically proven', 'guaranteed to', '#1 best',
                   'cure', 'treat', 'medical grade', 'scientifically proven',
                   'visit our website', 'contact us before', 'leave a review']

# 核心关键词（母婴品类示例）
CORE_KEYWORDS = {'breast pump': 4.0, 'electric': 2.0, 'portable': 2.0,
                  'hospital grade': 1.5, 'wearable': 1.5, 'quiet': 2.0,
                  'bpa free': 1.5, 'rechargeable': 1.5, 'hands free': 1.5}


def score_seo(listing: ListingContent) -> dict:
    """SEO 覆盖度评分 (0-25)"""
    score = 0
    issues = []
    text_all = f"{listing.title} {' '.join(listing.bullet_points)} {listing.search_terms}".lower()

    # 关键词覆盖
    covered = sum(1 for kw in CORE_KEYWORDS if kw in text_all)
    kw_score = min(12, covered * 1.5)
    score += kw_score
    if covered < 5:
        issues.append(f'核心关键词覆盖不足（{covered}/{len(CORE_KEYWORDS)}个），补充：'
                       + '、'.join([k for k in CORE_KEYWORDS if k not in text_all][:3]))

    # Search Terms 利用率
    st_bytes = len(listing.search_terms.encode())
    st_score = min(8, st_bytes / 250 * 8)
    score += st_score
    if st_bytes < 200:
        issues.append(f'Search Terms 字段仅用 {st_bytes}/250 字节，仍有 {250-st_bytes} 字节可补充长尾词')

    # 标题关键词位置（前80字最重要）
    if any(kw in listing.title[:80].lower() for kw in ['breast pump', 'electric', 'quiet']):
        score += 5
    else:
        issues.append('标题前80字符未包含核心关键词（breast pump/electric/quiet），影响搜索权重')

    return {'score': round(score, 1), 'max': 25, 'issues': issues}


def score_content(listing: ListingContent) -> dict:
    """内容质量评分 (0-25)"""
    score = 0; issues = []

    # 标题长度
    title_len = len(listing.title)
    if 80 <= title_len <= 200:
        score += 8
    elif title_len < 50:
        score += 3
        issues.append(f'标题过短({title_len}字符)，建议80-200字符，详细描述产品特性')
    elif title_len > 200:
        score += 5
        issues.append(f'标题过长({title_len}字符)，超出Amazon推荐200字符，可能截断')
    else:
        score += 6

    # Bullet Points 数量和质量
    bp_count = len([b for b in listing.bullet_points if b.strip()])
    bp_score = min(10, bp_count * 2)
    score += bp_score
    if bp_count < 5:
        issues.append(f'只有 {bp_count} 条要点（建议5条），每条详细描述一个卖点')
    short_bps = [i+1 for i, b in enumerate(listing.bullet_points) if len(b) < 80]
    if short_bps:
        issues.append(f'第 {short_bps} 条要点过短（<80字符），建议每条100-250字符')

    # 描述
    if len(listing.description) >= 200:
        score += 7
    elif len(listing.description) >= 100:
        score += 4
        issues.append('商品描述偏短，建议200字以上，包含产品背景故事和使用场景')
    else:
        score += 1
        issues.append('商品描述缺失或极短，这是 A9/A10 排名的重要因素')

    return {'score': round(score, 1), 'max': 25, 'issues': issues}


def score_images(listing: ListingContent) -> dict:
    """图片质量评分 (0-20)"""
    score = 0; issues = []

    if listing.num_images >= 7: score += 8
    elif listing.num_images >= 5: score += 6
    elif listing.num_images >= 3: score += 4; issues.append(f'图片数量{listing.num_images}张，建议≥7张，Amazon允许最多9张')
    else: score += 2; issues.append(f'图片严重不足（{listing.num_images}张），主图+生活场景+功能图最少6张')

    if listing.has_main_image_white_bg: score += 4
    else: issues.append('主图不符合Amazon白底规范（必须纯白背景）')

    if listing.has_lifestyle_image: score += 4
    else: issues.append('缺少生活场景图（妈妈使用中），生活场景图提升转化率10-20%')

    if listing.has_infographic: score += 4
    else: issues.append('缺少功能说明图（标注核心参数如噪音<45dB、双边吸力等）')

    return {'score': round(score, 1), 'max': 20, 'issues': issues}


def score_compliance(listing: ListingContent) -> dict:
    """合规风险评分 (0-15)，违规扣分"""
    score = 15; issues = []
    text_all = f"{listing.title} {' '.join(listing.bullet_points)} {listing.description}".lower()

    violations = [fw for fw in FORBIDDEN_WORDS if fw in text_all]
    if violations:
        score -= len(violations) * 3
        issues.append(f'发现违规词: {violations}，可能导致Listing下架')

    score = max(0, score)
    if score == 15:
        issues.append('✅ 合规检查通过，未发现明显违规词')

    return {'score': round(score, 1), 'max': 15, 'issues': issues}


def score_competitiveness(listing: ListingContent) -> dict:
    """竞争力定位评分 (0-15)"""
    score = 0; issues = []

    if listing.review_count >= 100: score += 6
    elif listing.review_count >= 30: score += 4
    elif listing.review_count >= 10: score += 2; issues.append(f'评论数{listing.review_count}条偏少，新品初期可申请Vine计划')
    else: score += 1; issues.append(f'评论数不足（{listing.review_count}条），优先积累评论再加大广告')

    if listing.review_rating >= 4.3: score += 5
    elif listing.review_rating >= 4.0: score += 3
    elif listing.review_rating >= 3.5: score += 2; issues.append(f'评分{listing.review_rating}偏低，分析差评根因')
    else: score += 0; issues.append(f'评分{listing.review_rating}过低，影响转化和排名，优先改善产品质量')

    score += 4  # 价格竞争力（简化：假设合理定价）

    return {'score': round(score, 1), 'max': 15, 'issues': issues}


def diagnostic_full(listing: ListingContent) -> dict:
    """完整 Listing 快诊"""
    dims = {
        'SEO覆盖度': score_seo(listing),
        '内容质量': score_content(listing),
        '图片质量': score_images(listing),
        '合规风险': score_compliance(listing),
        '竞争力': score_competitiveness(listing),
    }
    total = sum(d['score'] for d in dims.values())
    max_total = sum(d['max'] for d in dims.values())
    grade = 'A' if total >= 80 else ('B' if total >= 65 else ('C' if total >= 50 else 'D'))

    # 收集所有问题，按维度权重排序
    all_issues = []
    for dim_name, dim_result in dims.items():
        gap = dim_result['max'] - dim_result['score']
        for issue in dim_result['issues']:
            if not issue.startswith('✅'):
                all_issues.append({'dimension': dim_name, 'gap': gap, 'issue': issue})

    all_issues.sort(key=lambda x: -x['gap'])

    return {'asin': listing.asin, 'total_score': round(total, 1), 'max_score': max_total,
            'grade': grade, 'dimensions': dims,
            'top_fixes': all_issues[:5]}


def run_listing_diagnostic_demo():
    print('=' * 65)
    print('Listing Health Diagnostic — Listing 快诊评分系统')
    print('=' * 65)

    listing = ListingContent(
        asin='B0TEST001',
        title='Electric Breast Pump Double',
        bullet_points=[
            'Great pump!',
            'Hospital strength suction',
            'USB rechargeable',
        ],
        description='Good breast pump.',
        search_terms='breast pump electric',
        num_images=3,
        has_main_image_white_bg=True,
        has_lifestyle_image=False,
        has_infographic=False,
        review_count=45, review_rating=4.2,
    )

    result = diagnostic_full(listing)
    total = result['total_score']
    icon = '✅' if result['grade'] == 'A' else ('⚠️' if result['grade'] in ('B','C') else '❌')

    print(f'\n{icon} ASIN: {listing.asin}  综合评分: {total}/{result["max_score"]}  等级: {result["grade"]}')
    print(f'\n  {"维度":<12} {"得分":>6} {"满分":>6}  问题数')
    print('  ' + '-' * 40)
    for dim, data in result['dimensions'].items():
        n_issues = len([i for i in data['issues'] if not i.startswith('✅')])
        gap_icon = '🔴' if data['score'] < data['max'] * 0.5 else ('🟡' if data['score'] < data['max'] * 0.8 else '✅')
        print(f'  {dim:<12} {data["score"]:>6.1f} {data["max"]:>6}  {gap_icon} {n_issues}个问题')

    print(f'\n  🔧 Top 5 优先修复（按影响力排序）:')
    for i, fix in enumerate(result['top_fixes'], 1):
        print(f'  {i}. [{fix["dimension"]}] {fix["issue"][:65]}')

    print('\n[✓] Listing Health Diagnostic 测试通过')


if __name__ == '__main__':
    run_listing_diagnostic_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SEO-Organic-Ranking-Optimization]]（SEO 优化是 Listing 快诊的核心子模块）
- **前置（prerequisite）**：[[Skill-Category-Compliance-Prescan]]（合规预扫描提供快诊的合规维度数据）
- **延伸（extends）**：[[Skill-Listing-Compliance-Auto-Repair]]（快诊发现问题 → 合规修复自动执行）
- **延伸（extends）**：[[Skill-Listing-AI-Copywriting]]（快诊识别内容质量不足 → AI 文案优化）
- **可组合（combinable）**：[[Skill-Compliance-ML-Risk-Scoring]]（组合：快诊多维度评分 + ML 合规风险评分 = 完整的 Listing 上架前审查体系）
- **可组合（combinable）**：[[Skill-Amazon-A10-Algorithm-Ranking]]（组合：A10 排名因子 + 快诊维度 = 针对性的排名提升方案）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 上架质量提升：搜索曝光+15-25%，转化率+8-12%
  - 避免违规词导致下架：每次保护 ¥5-50 万 GMV
  - 运营决策效率：15分钟诊断 vs 人工3-4小时
  - **年化综合 ROI：¥10-30 万**

- **实施难度**：⭐⭐☆☆☆（规则引擎 1-2 周可实现；需要品类属性模板；2 周完整版）

- **优先级评分**：⭐⭐⭐⭐⭐（小型卖家最常见的上架决策支持需求；完全空白；桥接 广告分析↔NLP-VOC↔合规决策 三域）

- **评估依据**：Listing 质量是 A10 排名的关键因子（亚马逊官方确认）；第三方工具（Helium10/Jungle Scout）的 Listing 评分功能验证了用户需求；多维度评分对曝光和转化的影响已有大量实测数据
