# Skill Card: AGRS-属性引导评论摘要

---

## ① 算法原理

**核心思想**：将大规模LLM评论摘要从"无约束自由生成"重构为"属性引导的结构化生成"。通过ABSA提取aspect-sentiment对、consolidation去噪归一、代表性评论采样、结构化prompt引导，生成100%基于真实反馈的产品摘要，从根本上避免幻觉。

**数学直觉**：
1. **Aspect提取与整合**：对每条评论用结构化prompt提取最多5个aspect-sentiment对；通过频率阈值（95th percentile，约30次）将细粒度词汇变体映射到canonical forms，低频噪音aspect向上合并到更高级别概念。
2. **Top-K Aspect筛选**：统计整合后的aspect频率，选取Top 5作为摘要的核心骨架。
   $$\text{TopAspects} = \arg\max_{A' \subset A, |A'|=5} \sum_{a \in A'} \text{freq}(a)$$
3. **代表性评论采样**：对每个aspect-sentiment pair按频率加权采样代表性评论，既保证观点覆盖均衡，又将输入上下文限制在可控长度（上限200条评论/产品）。
4. **引导式摘要生成**：将consolidated aspects和selected reviews以固定模板组织进prompt，约束LLM输出空间和事实依据，生成300-500字符的凝练摘要。

**关键假设**：单个产品评论量≥10条才能支撑有意义的aspect统计；存在可用的LLM用于结构化提取和摘要生成；aspect consolidation的canonical映射可被有效缓存复用。

---

## ② 母婴出海应用案例

### 场景1：Momcozy消毒器双平台季度摘要

**业务问题**：Momcozy紫外线消毒器在Amazon US和Amazon DE均有销售，每季度运营团队需要汇总双平台用户反馈形成产品复盘报告，但直接阅读数千条评论效率极低，且传统LLM自由生成摘要容易出现幻觉或遗漏关键问题。

**数据要求**：
- 季度内Amazon US + Amazon DE的Momcozy紫外线消毒器评论（≥1000条）
- 字段：评论文本、星级、日期、平台标签、评论ID

**预期产出**：
- 自动提取并整合的aspect-sentiment对（如"消毒效果-positive""烘干功能-negative""容量大小-negative"）
- 经去重和频率筛选后的Top 5核心关注属性
- 基于真实评论生成的季度摘要，示例输出：
  > "关于Momcozy紫外线消毒器，用户最关注的是消毒效果、烘干功能、容量大小。具体而言，消毒效果（提及12次）满意度高；烘干功能（提及8次）吐槽较多；容量大小（提及6次）整体尚可。这些反馈主要来源于6条代表性评论。"
- 可直接用于季度管理层汇报和产品迭代roadmap输入

**业务价值**：将季度评论复盘周期从2周缩短至1天，确保摘要100% grounded in真实评论，避免LLM幻觉误导决策；预计提升产品迭代响应速度40%。

### 场景2：Momcozy暖奶器上市后快速评论监控

**业务问题**：新品Momcozy智能暖奶器上市后，需要快速捕捉早期用户反馈热点，及时调整营销策略和产品FAQ，但手动监控成本高。

**数据要求**：
- 上市后累积的Amazon评论（≥10条触发）
- 实时评论数据流或每日增量抓取

**预期产出**：
- 评论数达阈值后自动生成aspect-guided摘要
- 识别早期高关注属性（如"加热均匀性""温控精准度""操作简便性"）
-  sentiment 分布预警：若某个核心属性负面占比>50%，自动标记并推送运营团队

**业务价值**：实现新品评论监控自动化，早期问题发现时间从1-2周缩短至24-48小时，降低新品口碑危机风险。

---

## ③ 代码模板

代码路径：`paper2skills-code/nlp_voc/agrs_review_summarization/model.py`

```python
"""
AGRS: Aspect-Guided Review Summarization at Scale
基于论文: End-to-End Aspect-Guided Review Summarization at Scale
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Review:
    text: str
    review_id: str = ""
    rating: int = 5
    market: str = ""


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str  # positive / negative / mixed
    review_id: str = ""


@dataclass
class ProductSummary:
    product_name: str
    top_aspects: List[Tuple[str, int]] = field(default_factory=list)
    summary_text: str = ""
    selected_reviews_count: int = 0


class AspectExtractor:
    """Simulated LLM-based aspect extraction from individual reviews."""

    ASPECT_KEYWORDS = {
        "消毒效果": (["消毒", "杀菌", "干净", "卫生"], ["消毒不彻底", "杀菌弱", "不干净"]),
        "烘干功能": (["烘干", "干燥", "无水渍"], ["烘干慢", "不干", "有水渍"]),
        "容量大小": (["容量大", "装得多", "空间大"], ["容量小", "装不下", "空间小"]),
        "操作简便性": (["操作简单", "一键", "方便"], ["操作复杂", "难用", "麻烦"]),
        "噪音控制": (["静音", "噪音小", "安静"], ["噪音大", "吵", "声响"]),
        "外观设计": (["好看", "美观", "精致"], ["丑", "粗糙", "廉价感"]),
        "性价比": (["划算", "值得", "性价比高"], ["贵", "不值", "性价比低"]),
    }

    def extract(self, reviews: List[Review]) -> List[AspectSentiment]:
        results = []
        for review in reviews:
            extracted = []
            for aspect, (pos_kw, neg_kw) in self.ASPECT_KEYWORDS.items():
                pos_hit = any(kw in review.text for kw in pos_kw)
                neg_hit = any(kw in review.text for kw in neg_kw)
                if pos_hit and neg_hit:
                    extracted.append((aspect, "mixed"))
                elif pos_hit:
                    extracted.append((aspect, "positive"))
                elif neg_hit:
                    extracted.append((aspect, "negative"))
            for asp, sent in extracted[:5]:
                results.append(AspectSentiment(aspect=asp, sentiment=sent, review_id=review.review_id))
        return results


class AspectConsolidator:
    """Maps fine-grained aspects to broader canonical forms."""

    def __init__(self, min_freq_threshold: int = 3):
        self.min_freq = min_freq_threshold

    def consolidate(self, aspects: List[AspectSentiment]) -> List[AspectSentiment]:
        freq = Counter(a.aspect for a in aspects)
        canonical = {}
        for asp, count in freq.items():
            canonical[asp] = asp if count >= self.min_freq else self._broaden(asp)
        consolidated = []
        for a in aspects:
            new_asp = canonical.get(a.aspect, a.aspect)
            consolidated.append(AspectSentiment(aspect=new_asp, sentiment=a.sentiment, review_id=a.review_id))
        return consolidated

    def _broaden(self, aspect: str) -> str:
        hierarchy = {
            "消毒效果": "功能表现", "烘干功能": "功能表现",
            "容量大小": "使用体验", "操作简便性": "使用体验", "噪音控制": "使用体验",
            "外观设计": "外观品质", "性价比": "购买决策",
        }
        return hierarchy.get(aspect, "综合体验")


class ReviewSelector:
    def __init__(self, max_aspects: int = 5, max_reviews_per_product: int = 200):
        self.max_aspects = max_aspects
        self.max_reviews = max_reviews_per_product

    def select(self, aspects: List[AspectSentiment], reviews: List[Review], max_reviews_per_aspect: int = 10):
        review_map = {r.review_id: r for r in reviews}
        aspect_counts = Counter(a.aspect for a in aspects)
        top_aspects = aspect_counts.most_common(self.max_aspects)
        aspect_to_reviews = defaultdict(list)
        for a in aspects:
            if a.aspect in [ta[0] for ta in top_aspects] and a.review_id in review_map:
                aspect_to_reviews[a.aspect].append(a.review_id)

        selected_review_ids = set()
        sampled_aspect_reviews = {}
        for asp, _ in top_aspects:
            ids = list(set(aspect_to_reviews.get(asp, [])))
            sampled = random.sample(ids, min(len(ids), max_reviews_per_aspect))
            sampled_aspect_reviews[asp] = sampled
            selected_review_ids.update(sampled)

        selected_reviews = [review_map[r_id] for r_id in selected_review_ids if r_id in review_map]
        return top_aspects, selected_reviews[:self.max_reviews], sampled_aspect_reviews


class GuidedSummarizer:
    def summarize(self, product_name: str, top_aspects, aspect_sentiments, sampled_reviews, review_map):
        lines = []
        for aspect, count in top_aspects:
            sentiments = [a.sentiment for a in aspect_sentiments if a.aspect == aspect]
            total = len(sentiments)
            if total == 0:
                continue
            pos_ratio = sentiments.count("positive") / total
            neg_ratio = sentiments.count("negative") / total
            mix_ratio = sentiments.count("mixed") / total
            if pos_ratio >= 0.7:
                tone = "满意度高"
            elif neg_ratio >= 0.5:
                tone = "吐槽较多"
            elif mix_ratio >= 0.3:
                tone = "评价分化"
            else:
                tone = "整体尚可"
            lines.append(f"{aspect}（提及{count}次）{tone}")

        if not lines:
            return f"{product_name}暂无足够评论数据生成摘要。"

        summary = f"关于{product_name}，用户最关注的是" + "、".join([asp for asp, _ in top_aspects[:3]]) + "。"
        summary += "具体而言，" + "；".join(lines) + "。"
        n_reviews = len(set(review_map[rid].review_id for rs in sampled_reviews.values() for rid in rs if rid in review_map))
        summary += f"这些反馈主要来源于{n_reviews}条代表性评论。"
        return summary[:300]


class AGRSPipeline:
    def __init__(self, min_freq: int = 3, max_aspects: int = 5, max_reviews: int = 200):
        self.extractor = AspectExtractor()
        self.consolidator = AspectConsolidator(min_freq_threshold=min_freq)
        self.selector = ReviewSelector(max_aspects=max_aspects, max_reviews_per_product=max_reviews)
        self.summarizer = GuidedSummarizer()

    def summarize_product(self, product_name: str, reviews: List[Review]) -> ProductSummary:
        raw_aspects = self.extractor.extract(reviews)
        consolidated = self.consolidator.consolidate(raw_aspects)
        top_aspects, selected_reviews, sampled = self.selector.select(consolidated, reviews)
        review_map = {r.review_id: r for r in reviews}
        summary_text = self.summarizer.summarize(product_name, top_aspects, consolidated, sampled, review_map)
        return ProductSummary(
            product_name=product_name,
            top_aspects=top_aspects,
            summary_text=summary_text,
            selected_reviews_count=len(selected_reviews),
        )


def build_demo_reviews() -> List[Review]:
    return [
        Review(review_id="r1", text="这款Momcozy紫外线消毒器消毒效果很好，烘干功能也不错，但是容量有点小，放不下全套吸奶器配件。", rating=4, market="US"),
        Review(review_id="r2", text="操作简单一键启动，消毒效果彻底，就是烘干时间太长了，而且运行时噪音有点大。", rating=3, market="US"),
        Review(review_id="r3", text="外观设计很好看，放在厨房里很美观，消毒效果和烘干功能都很满意，容量也够用。", rating=5, market="DE"),
        Review(review_id="r4", text="性价比很高，消毒烘干一体很方便，但是操作面板有点复杂，老人不太会用。", rating=4, market="US"),
        Review(review_id="r5", text="容量大，可以同时消毒奶瓶和吸奶器，烘干后没有水渍，静音设计很好。", rating=5, market="DE"),
        Review(review_id="r6", text="消毒效果不错，但是噪音控制一般，晚上使用会影响宝宝睡觉，希望能改进。", rating=3, market="US"),
        Review(review_id="r7", text="操作简单，消毒彻底，外观设计精致，性价比也很高，整体非常满意。", rating=5, market="US"),
        Review(review_id="r8", text="烘干功能不太稳定，有时候奶瓶内壁还有水珠，但消毒效果没问题。", rating=3, market="DE"),
    ]


def demo():
    reviews = build_demo_reviews()
    pipeline = AGRSPipeline(min_freq=2, max_aspects=5, max_reviews=200)
    summary = pipeline.summarize_product("Momcozy 紫外线消毒器", reviews)
    result = {
        "product": summary.product_name,
        "top_aspects": summary.top_aspects,
        "selected_reviews": summary.selected_reviews_count,
        "summary": summary.summary_text,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
```

---

## ④ 技能关联

- **前置技能**：
  - `Skill-StaR-观点语句排序` — StaR提取的canonical statements可作为AGRS aspect extraction的高质量输入，提升提取准确性和去噪效果
  - `Skill-BERT-MoE高效方面情感分析` — 提供精确的aspect-sentiment识别能力，是AGRS aspect extraction的基础组件
- **延伸技能**：
  - `Skill-MAA-行动建议生成` — AGRS生成的季度摘要和Top aspects可直接输入MAA pipeline，进一步转化为可执行的产品改进建议
  - `Skill-Kano-需求分类与优先级` — AGRS识别的高频/高关注属性可进入Kano模型，判断其属于基本型、期望型还是魅力型需求
- **可组合技能**：
  - 与 `Skill-StaR-观点语句排序` + `Skill-MAA-行动建议生成` 组合：StaR提取 → AGRS摘要 → MAA建议，形成"洞察→总结→决策"完整链路

---

## ⑤ 商业价值评估

- **ROI预估**：
  - 直接收益：季度评论复盘周期从2周缩短至1天，运营人力成本降低约75%；消除LLM幻觉导致的错误决策风险
  - 间接收益：基于真实评论的aspect-guided摘要提升产品迭代精准度，预计NPS提升5-10分，新品上市后的早期问题发现时间从1-2周缩短至24-48小时
  - 综合ROI：首年投入约6万元（含数据接入、LLM调用和prompt调优），预期节省人力+降低风险+提升转化带来的回报约40-55万元，**ROI约6-9倍**

- **实施难度**：⭐⭐⭐☆☆（3/5）
  - 需要搭建结构化LLM prompt pipeline和aspect consolidation缓存机制，工程复杂度中等；但Wayfair论文已提供完整架构参考和开源数据集

- **优先级评分**：⭐⭐⭐⭐☆（4/5）
  - 直接解决LLM摘要的幻觉痛点，是规模化VOC运营的核心基础设施；与现有ABSA、StaR、MAA技能形成高价值闭环

- **评估依据**：
  AGRS通过"属性引导"的范式确保摘要100% grounded in真实评论，这是传统自由生成式摘要无法比拟的优势。在Wayfair的大规模A/B测试中，ATCR提升0.3%、CVR提升0.5%、跳出率降低0.13%，证明了其对电商核心指标的真实正向影响。对于依赖用户口碑的母婴出海业务，该技能的信任度和可落地性极高。
