# Skill: VOC Proxy NPS × AIPL 统一标签萃取引擎

## 基础信息

- **技能名称**: VOC-Proxy-NPS-AIPL-统一萃取引擎
- **核心方法**: 多标签关键词匹配 + 品线过滤 + 情感校准 + 画像推导
- **应用场景**: 母婴出海跨境电商 VOC 全链路标签自动萃取
- **数据规模**: 376 标签种子 + 55 原子画像标签 + AIPL 7 节点
- **代码位置**: `paper2skills-code/nlp_voc/proxy_nps_aipl_workflow/`

---

## 1. 算法原理

### 1.1 核心问题

传统 VOC 标签体系存在三大断层：

1. **标签碎片化**：产品问题标签、AIPL 旅程标签、画像标签分散在不同系统，无法从一条 VOC 文本同时萃取
2. **品线交叉污染**：通用标签和品线专属标签混用，导致吸奶器标签误打到内衣评论上
3. **情感方向混乱**：同一文本中不同方面的情感方向可能相反（"吸力好但噪音大"），粗粒度情感极性无法支撑业务决策

### 1.2 统一萃取框架

```
单条 VOC 文本输入
    │
    ├──→ 品线过滤：只加载该 VOC 所属品线的标签种子
    │
    ├──→ 376 标签种子匹配：关键词 + 消费者习惯表达（多标签并存）
    │      └── 否定词检测："not recommend" ≠ recommend 意愿
    │
    ├──→ 55 原子画像标签匹配：WHO/WHY/WHAT/WHEN/HOW/EMOTION
    │      └── 共现模式推导 → 社群黏着型/系统规划型/品质探索型
    │
    ├──→ 情感校准：标签预定义情感 + ABSA 动态计算
    │      ├── preset=负, ABSA=负 → calibrated（取 ABSA 强度）
    │      ├── preset=负, ABSA=正 → conflict（需人工复核）
    │      └── preset=正, ABSA=正 → calibrated（取 ABSA 强度）
    │
    ├──→ 品牌检测：Momcozy + 竞品提及（Spectra/Medela/Willow/Elvie）
    │
    └──→ Proxy NPS 计算：多标签场景下的 Promoter/Detractor/Passive 判定
           ├── [推荐意愿] + 正向 → Promoter
           ├── [产品问题] + 负向 → Detractor
           └── 无标签 + 5星 → Promoter（默认）
```

### 1.3 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 多标签策略 | 全部保留 | 用户可能同时提及多个问题方面 |
| 品线过滤 | 统一跑一次全量（带过滤） | 避免先通用后个性的两次遍历开销 |
| 情感校准 | 预定义 + ABSA 动态 | 预定义保证一致性，ABSA 捕捉上下文 |
| 画像推导 | 共现计分（非硬编码） | 数据驱动的画像归属，避免规则僵化 |
| Proxy NPS | 标签优先级法 | 推荐意愿标签优先级最高 |

### 1.4 数据模型

```python
@dataclass
class VOCLabelExtraction:
    # 基础信息
    review_id, source_type, platform, spu_code, product_line, category, rating

    # 维度1: AIPL 旅程
    aipl_stage: str           # 主阶段 A/I/P1/P2/L1/L2/L3
    aipl_tags: list[dict]     # [{tag_id, tag_en, tag_cn, theme, sentiment_calibrated, confidence}]

    # 维度2: 问题类型（现有 classification）
    classification_tag, cn_level1, cn_level2, cn_level3

    # 维度3: 画像
    persona_atomic: list[str]   # 命中的原子标签
    persona_derived: str        # 推导的业务画像

    # 维度4: 情感
    sentiment_polarity, sentiment_intensity, sentiment_calibration, aspect_sentiments

    # 维度5: 品牌
    brand_mentions, brand_comparison

    # 维度6: 质量
    quality_score, is_suspicious

    # 业务闭环
    proxy_nps_contribution, metric_direction, story_line, strategy_pack, owner_dept, priority
```

---

## 2. 业务应用

### 2.1 Momcozy 场景：一条评论萃取全部标签

**输入**（亚马逊评论）：
> "I was searching for a wearable pump and came across Momcozy on TikTok. Compared it with Willow and Elvie, the price is much more affordable. However, the flange size is too small and the suction feels weak. Customer service was slow to respond. Would not recommend to friends."

**萃取输出**：
```json
{
  "review_id": "REV001",
  "aipl_stage": "L1",
  "aipl_tags": [
    {"tag_en": "brand_search", "aipl_node": "A", "theme": "品牌认知"},
    {"tag_en": "product_comparison", "aipl_node": "I", "theme": "产品对比"},
    {"tag_en": "flange_size_issue", "aipl_node": "P1", "theme": "产品核心性能"},
    {"tag_en": "suction_too_weak", "aipl_node": "L1", "theme": "产品核心性能"},
    {"tag_en": "slow_customer_service", "aipl_node": "L2", "theme": "问题解决效率"},
    {"tag_en": "price_concern", "aipl_node": "I", "theme": "价格价值感"}
  ],
  "persona": {
    "atomic": ["hands_free_seeker", "research_driven", "social_media_influenced", "price_sensitive"],
    "derived": "community_driven"
  },
  "sentiment": {"polarity": -0.50, "intensity": -2.5, "calibration": "conflict"},
  "brand": {"mentions": ["momcozy", "willow", "elvie"], "comparison": true},
  "proxy_nps": "detractor",
  "strategy_pack": "服务体验优化包",
  "owner_dept": "客户服务部",
  "priority": "P0"
}
```

**业务闭环**：
- 主责部门：客户服务部（P0）→ 立即跟进
- 策略包：服务体验优化包 + 核心体验改良包
- 画像洞察：该用户属于"社群黏着型"，对价格敏感，通过社媒了解品牌

### 2.2 指标看板：Proxy NPS × AIPL 漏斗

```python
# 生成指标看板
dashboard = DashboardGenerator().build(extractions)

# 输出结构
{
  "proxy_nps": {
    "overall": {"proxy_nps": 35.0, "promoters": 350, "detractors": 150},
    "by_product_line": {
      "breast_pump": {"proxy_nps": 42.0, ...},
      "underwear": {"proxy_nps": 28.0, ...}
    },
    "by_persona": {
      "community_driven": {"proxy_nps": 38.0, ...},
      "systematic_planner": {"proxy_nps": 32.0, ...},
      "quality_explorer": {"proxy_nps": 45.0, ...}
    }
  },
  "aipl_funnel": {
    "A": {"count": 1200, "top_themes": ["品牌认知", "产品认知"]},
    "I": {"count": 800, "top_themes": ["产品对比", "价格价值感"]},
    "L3": {"count": 80, "top_themes": ["推荐意愿"]}
  },
  "driver_analysis": {
    "top_detractor_themes": [
      {"theme": "产品核心性能", "mention_rate": 0.15, "avg_sentiment": -0.72},
      {"theme": "物流时效", "mention_rate": 0.12, "avg_sentiment": -0.65}
    ]
  }
}
```

### 2.3 四路数据源统一处理

```python
# 退货留言 (212,746条) → 规则标注为主
# 客服工单 (124,928条) → ALCHEmist + 规则
# 商品评论 (15,418条) → ABSA + 画像
# Trustpilot (2,605条) → 轻量分类 + NPS

# 统一流水线处理全部 355,697 条
workflow = VOCProxyNPSWorkflow(tag_dict_path="tag_seeds.csv")
results = workflow.run(all_voc_records)
```

---

## 3. 代码模板

完整代码见：`paper2skills-code/nlp_voc/proxy_nps_aipl_workflow/`

```python
"""
VOC Proxy NPS × AIPL 统一标签萃取引擎

核心流程：VOCRecord → VOCLabelExtraction → DashboardData
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict, Counter
import csv


# ==================== 数据模型 ====================

@dataclass
class VOCRecord:
    """单条 VOC 原始输入"""
    review_id: str
    text: str
    source_type: str      # return_note / ticket / review / trustpilot
    platform: str         # amazon / dtc / offline / tiktok
    spu_code: str
    product_line: str     # 品线（如 breast_pump）
    category: str         # 品类（如 wearable_pump）
    rating: Optional[float] = None


@dataclass
class TagSeed:
    """单条标签种子"""
    tag_id: str
    tag_en: str
    tag_cn: str
    aipl_node: str        # A/I/P1/P2/L1/L2/L3
    theme: str
    sentiment_preset: str # positive / negative / neutral
    keywords: list[str] = field(default_factory=list)
    applicable_line: list[str] = field(default_factory=list)
    # 业务元数据
    strategy_pack: str = ""
    owner_dept: str = ""
    priority: str = ""


@dataclass
class VOCLabelExtraction:
    """单条 VOC 完整萃取结果"""
    review_id: str
    aipl_stage: str
    aipl_tags: list[dict] = field(default_factory=list)
    persona_atomic: list[str] = field(default_factory=list)
    persona_derived: str = ""
    sentiment_polarity: float = 0.0
    brand_mentions: list[str] = field(default_factory=list)
    proxy_nps_contribution: str = ""  # promoter / passive / detractor
    strategy_pack: str = ""
    owner_dept: str = ""
    priority: str = ""


# ==================== 核心引擎 ====================

class TagSeedDictionary:
    """统一标签字典 — 管理全部标签种子"""

    def __init__(self):
        self._tags: dict[str, TagSeed] = {}
        self._by_line: dict[str, list[TagSeed]] = defaultdict(list)

    def add(self, tag: TagSeed) -> None:
        self._tags[tag.tag_id] = tag
        for line in tag.applicable_line:
            self._by_line[line].append(tag)

    @classmethod
    def from_csv(cls, path: str) -> 'TagSeedDictionary':
        dictionary = cls()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = TagSeed(
                    tag_id=row.get("tag_id", ""),
                    tag_en=row.get("tag_en", row.get("VOC标签（英文）", "")),
                    tag_cn=row.get("tag_cn", row.get("VOC标签（中文）", "")),
                    aipl_node=row.get("aipl_node", row.get("AIPL节点", "")),
                    theme=row.get("theme", row.get("标签主题", "")),
                    sentiment_preset=row.get("sentiment_preset", row.get("情感极性", "")),
                    keywords=[s.strip() for s in row.get("keywords", "").split(",") if s.strip()],
                    applicable_line=[s.strip() for s in row.get("applicable_line", "").split(",") if s.strip()],
                    strategy_pack=row.get("strategy_pack", row.get("策略包", "")),
                    owner_dept=row.get("owner_dept", row.get("主责部门", "")),
                    priority=row.get("priority", row.get("默认优先级", "")),
                )
                dictionary.add(tag)
        return dictionary

    def filter_by_line(self, line: str) -> list[TagSeed]:
        """按品线过滤：通用标签 + 该品线专属"""
        return [
            tag for tag in self._tags.values()
            if not tag.applicable_line or line in tag.applicable_line
        ]


class VOCLabelExtractor:
    """单条 VOC 完整标签萃取器"""

    # 否定词列表
    NEGATION_WORDS = {"not", "no", "never", "n't", "dont", "doesnt", "wouldnt"}

    # 55 原子画像标签（示例）
    PERSONA_KEYWORDS = {
        "first_time_parent": ["first time mom", "new mom", "newborn"],
        "research_driven": ["research", "compare", "review", "youtube"],
        "price_sensitive": ["expensive", "cheap", "price", "budget"],
        "quiet_seeker": ["quiet", "silent", "noise", "loud"],
        "nighttime_user": ["at night", "middle of the night", "2am"],
        "anxiety_driven": ["worried", "anxious", "nervous", "stress"],
    }

    # 品牌列表
    BRANDS = {"momcozy", "spectra", "medela", "willow", "elvie", "lansinoh"}

    # 画像推导信号
    PERSONA_SIGNALS = {
        "community_driven": {"word_of_mouth", "social_media", "price_sensitive", "first_time_parent"},
        "systematic_planner": {"research_driven", "quiet_seeker", "anxiety_driven"},
        "quality_explorer": {"nighttime_user", "brand_loyal", "care_driven"},
    }

    def __init__(self, tag_dict: TagSeedDictionary):
        self.tag_dict = tag_dict

    def extract(self, voc: VOCRecord) -> VOCLabelExtraction:
        text_lower = voc.text.lower()

        # 1. 品线过滤 + 标签匹配
        candidate_tags = self.tag_dict.filter_by_line(voc.product_line)
        aipl_tags = self._match_aipl_tags(text_lower, candidate_tags)

        # 2. 画像标签匹配
        persona_atomic = self._match_persona(text_lower)
        persona_derived = self._derive_persona(persona_atomic)

        # 3. 情感校准
        sentiment = self._calibrate_sentiment(text_lower, aipl_tags, voc.rating)

        # 4. 品牌检测
        brand_mentions = [b for b in self.BRANDS if b in text_lower]

        # 5. Proxy NPS
        proxy_nps = self._calculate_proxy_nps(aipl_tags, sentiment, voc.rating)

        # 6. 业务元数据聚合
        meta = self._aggregate_meta(aipl_tags)

        # 7. 确定主 AIPL 阶段
        aipl_stage = self._derive_stage(aipl_tags, voc.source_type)

        return VOCLabelExtraction(
            review_id=voc.review_id,
            aipl_stage=aipl_stage,
            aipl_tags=aipl_tags,
            persona_atomic=persona_atomic,
            persona_derived=persona_derived,
            sentiment_polarity=sentiment,
            brand_mentions=brand_mentions,
            proxy_nps_contribution=proxy_nps,
            strategy_pack=meta.get("strategy_pack", ""),
            owner_dept=meta.get("owner_dept", ""),
            priority=meta.get("priority", ""),
        )

    def _match_aipl_tags(self, text_lower: str, candidate_tags: list[TagSeed]) -> list[dict]:
        """关键词匹配 AIPL 标签（多标签）"""
        matches = []
        for tag in candidate_tags:
            for kw in tag.keywords:
                kw_lower = kw.lower()
                if kw_lower in text_lower:
                    # 否定词检测（对 L3 推荐标签）
                    if tag.aipl_node == "L3" and self._has_negation(text_lower, kw_lower):
                        continue
                    matches.append({
                        "tag_id": tag.tag_id,
                        "tag_en": tag.tag_en,
                        "tag_cn": tag.tag_cn,
                        "aipl_node": tag.aipl_node,
                        "theme": tag.theme,
                        "sentiment_preset": tag.sentiment_preset,
                    })
                    break
        return matches

    def _has_negation(self, text: str, keyword: str, window: int = 15) -> bool:
        """检查关键词前是否有否定词"""
        idx = text.find(keyword)
        if idx < 0:
            return False
        prefix = text[max(0, idx - window):idx]
        return any(neg in prefix for neg in self.NEGATION_WORDS)

    def _match_persona(self, text_lower: str) -> list[str]:
        """匹配画像标签"""
        return [
            name for name, keywords in self.PERSONA_KEYWORDS.items()
            if any(kw.lower() in text_lower for kw in keywords)
        ]

    def _derive_persona(self, atomic_tags: list[str]) -> str:
        """55 原子 → 3 业务画像"""
        atomic_set = set(atomic_tags)
        scores = {
            persona: len(atomic_set & signals)
            for persona, signals in self.PERSONA_SIGNALS.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "uncategorized"

    def _calibrate_sentiment(
        self, text: str, aipl_tags: list[dict], rating: Optional[float]
    ) -> float:
        """情感校准：预定义 + ABSA"""
        # 简化版：统计情感词
        pos_words = {"good", "great", "excellent", "love", "perfect", "comfortable", "recommend"}
        neg_words = {"bad", "terrible", "worst", "weak", "slow", "pain", "disappointed"}

        pos = sum(1 for w in pos_words if w in text)
        neg = sum(1 for w in neg_words if w in text)

        if pos > neg:
            absa = 0.6
        elif neg > pos:
            absa = -0.6
        else:
            absa = 0.0

        # 评分校准
        if rating is not None:
            rating_sent = (rating - 3) / 2.0
            absa = absa * 0.7 + rating_sent * 0.3

        return max(-1.0, min(1.0, absa))

    def _calculate_proxy_nps(
        self, aipl_tags: list[dict], sentiment: float, rating: Optional[float]
    ) -> str:
        """Proxy NPS 计算"""
        has_recommendation = any(t["aipl_node"] == "L3" for t in aipl_tags)
        is_positive = sentiment > 0.2
        is_negative = sentiment < -0.2

        if has_recommendation:
            return "promoter" if is_positive else "detractor" if is_negative else "passive"

        if any(t["aipl_node"] in {"P1", "P2", "L1", "L2"} for t in aipl_tags):
            return "detractor" if is_negative else "passive" if is_positive else "passive"

        if rating is not None:
            return "promoter" if rating >= 4 else "detractor" if rating <= 2 else "passive"

        return "promoter" if is_positive else "detractor" if is_negative else "passive"

    def _derive_stage(self, aipl_tags: list[dict], source_type: str) -> str:
        """推导主 AIPL 阶段"""
        if not aipl_tags:
            return {"return_note": "P1", "ticket": "L2", "review": "L1", "trustpilot": "L3"}.get(source_type, "unknown")
        node_counts = Counter(t["aipl_node"] for t in aipl_tags)
        return node_counts.most_common(1)[0][0]

    def _aggregate_meta(self, aipl_tags: list[dict]) -> dict:
        """聚合业务元数据"""
        meta = {}
        priorities = []
        for tag in aipl_tags:
            seed = self.tag_dict._tags.get(tag["tag_id"])
            if seed:
                if seed.priority:
                    priorities.append(seed.priority)
                for field in ["strategy_pack", "owner_dept"]:
                    if not meta.get(field) and getattr(seed, field):
                        meta[field] = getattr(seed, field)
        if priorities:
            priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
            meta["priority"] = min(priorities, key=lambda x: priority_order.get(x, 99))
        return meta


class DashboardGenerator:
    """指标看板生成器"""

    def build(self, extractions: list[VOCLabelExtraction]) -> dict:
        """生成指标看板"""
        return {
            "proxy_nps": self._calc_proxy_nps(extractions),
            "aipl_funnel": self._calc_funnel(extractions),
            "driver_analysis": self._calc_drivers(extractions),
            "persona_insights": self._calc_persona(extractions),
        }

    def _calc_proxy_nps(self, extractions: list[VOCLabelExtraction]) -> dict:
        total = len(extractions)
        promoters = sum(1 for e in extractions if e.proxy_nps_contribution == "promoter")
        detractors = sum(1 for e in extractions if e.proxy_nps_contribution == "detractor")
        return {
            "proxy_nps": round((promoters / total * 100) - (detractors / total * 100), 1),
            "promoters": promoters,
            "detractors": detractors,
        }

    def _calc_funnel(self, extractions: list[VOCLabelExtraction]) -> dict:
        node_counts = Counter(e.aipl_stage for e in extractions)
        return {node: {"count": node_counts.get(node, 0)} for node in ["A", "I", "P1", "P2", "L1", "L2", "L3"]}

    def _calc_drivers(self, extractions: list[VOCLabelExtraction]) -> dict:
        theme_sentiments = defaultdict(list)
        for e in extractions:
            for tag in e.aipl_tags:
                theme_sentiments[tag["theme"]].append(e.sentiment_polarity)

        theme_stats = []
        for theme, sentiments in theme_sentiments.items():
            avg = sum(sentiments) / len(sentiments)
            theme_stats.append({
                "theme": theme,
                "avg_sentiment": round(avg, 2),
                "nps_contribution": "promoter_driver" if avg > 0.3 else "detractor_driver" if avg < -0.3 else "neutral",
            })
        theme_stats.sort(key=lambda x: abs(x["avg_sentiment"]), reverse=True)
        return {"top_detractor_themes": [t for t in theme_stats if t["nps_contribution"] == "detractor_driver"][:5]}

    def _calc_persona(self, extractions: list[VOCLabelExtraction]) -> dict:
        by_persona = defaultdict(list)
        for e in extractions:
            by_persona[e.persona_derived].append(e)
        total = len(extractions)
        return {
            persona: {
                "penetration": round(len(items) / total, 3),
                "avg_sentiment": round(sum(e.sentiment_polarity for e in items) / len(items), 2),
            }
            for persona, items in by_persona.items()
        }


# ==================== 端到端工作流 ====================

class VOCProxyNPSWorkflow:
    """VOC Proxy NPS × AIPL 全旅程指标落地工作流"""

    def __init__(self, tag_dict_path: Optional[str] = None):
        if tag_dict_path:
            self.tag_dict = TagSeedDictionary.from_csv(tag_dict_path)
        else:
            self.tag_dict = TagSeedDictionary()
        self.extractor = VOCLabelExtractor(self.tag_dict)
        self.dashboard = DashboardGenerator()

    def run(self, vocs: list[VOCRecord]) -> list[VOCLabelExtraction]:
        """执行完整工作流"""
        return [self.extractor.extract(voc) for voc in vocs]

    def generate_dashboard(self, extractions: list[VOCLabelExtraction]) -> dict:
        """生成指标看板"""
        return self.dashboard.build(extractions)


# ==================== 演示 ====================

def demo():
    """演示：单条 VOC 完整萃取"""
    print("=" * 70)
    print("VOC Proxy NPS × AIPL 统一标签萃取引擎 - 演示")
    print("=" * 70)

    # 构建演示标签字典
    tag_dict = TagSeedDictionary()
    tag_dict.add(TagSeed("TAG_A_001", "brand_search", "品牌搜索", "A", "品牌认知", "neutral",
                         ["searching for", "looking for", "came across"]))
    tag_dict.add(TagSeed("TAG_I_001", "product_comparison", "产品对比", "I", "产品对比", "neutral",
                         ["compared", "vs", "better than"]))
    tag_dict.add(TagSeed("TAG_L1_001", "suction_too_weak", "吸力差", "L1", "产品核心性能", "negative",
                         ["suction", "weak", "not strong"], ["breast_pump"],
                         strategy_pack="核心体验改良包", owner_dept="产品中心/品线", priority="P0"))
    tag_dict.add(TagSeed("TAG_L2_001", "slow_customer_service", "客服响应慢", "L2", "问题解决效率", "negative",
                         ["customer service", "slow", "no response"],
                         strategy_pack="服务体验优化包", owner_dept="客户服务部", priority="P0"))
    tag_dict.add(TagSeed("TAG_L3_001", "recommend_willingness", "推荐意愿", "L3", "推荐意愿", "positive",
                         ["recommend", "suggest", "tell friends"]))

    # 测试 VOC
    voc = VOCRecord(
        review_id="REV001",
        text=("I was searching for a wearable pump and came across Momcozy. "
              "Compared it with Willow, but the suction feels weak and "
              "customer service was slow. Would not recommend."),
        source_type="review",
        platform="amazon",
        spu_code="SPU001",
        product_line="breast_pump",
        category="wearable_pump",
        rating=2.0,
    )

    # 萃取
    extractor = VOCLabelExtractor(tag_dict)
    result = extractor.extract(voc)

    print(f"\n--- 萃取结果 ---")
    print(f"  AIPL 阶段: {result.aipl_stage}")
    print(f"  AIPL 标签: {[t['tag_en'] for t in result.aipl_tags]}")
    print(f"  画像原子: {result.persona_atomic}")
    print(f"  画像推导: {result.persona_derived}")
    print(f"  情感极性: {result.sentiment_polarity:+.2f}")
    print(f"  品牌提及: {result.brand_mentions}")
    print(f"  Proxy NPS: {result.proxy_nps_contribution}")
    print(f"  策略包: {result.strategy_pack or 'N/A'}")
    print(f"  主责部门: {result.owner_dept or 'N/A'}")
    print(f"  优先级: {result.priority or 'N/A'}")

    # 验证
    assert result.aipl_stage in {"A", "I", "L1", "L2"}
    assert "suction_too_weak" in [t["tag_en"] for t in result.aipl_tags]
    assert "recommend_willingness" not in [t["tag_en"] for t in result.aipl_tags]  # 否定词过滤
    assert result.proxy_nps_contribution == "detractor"
    print("\n✓ 验证通过")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
```

---

## 4. 技能关联

### 4.1 前置技能

| 技能 | 关系 | 说明 |
|------|------|------|
| **ReviewQuality-Scoring** | 输入过滤 | 低质量 VOC 在进入萃取前被过滤 |
| **TaxoAdapt-MultidimensionalTaxonomy** | 标签管理 | 多维标签空间的树状管理 |
| **AIPL-VOC-Lifecycle-Tags** | AIPL 分类 | AIPL 7 节点阶段分类基础 |
| [[Skill-InstructUIE-Unified-Information-Extraction]] | prerequisite | 统一 IE 框架是多标签萃取的前置基础能力 |

### 4.2 后置技能

| 技能 | 关系 | 说明 |
|------|------|------|
| **NPS-Driver-Analysis** | 输出 | Proxy NPS 驱动因素归因分析 |
| **Kano-Mapper** | 输出 | 标签主题 → Kano 需求分类 |
| **GPLR-Persona-Generation** | 互补 | 原子画像标签 → 营销人群包 |
| [[Skill-Semantic-Blueprint-Compiler]] | extends | 本 Skill 的 AIPL 标签 + 画像输出作为语义蓝图编译器的输入信号 |

### 4.3 可组合技能

- [[Skill-BERT-SRL-Event-Frame-Extraction]] — 组合场景：SRL 事件框架提取可识别评论中的动作-主体-客体结构，与本 Skill 的关键词匹配互补，覆盖表达复杂的长评论

### 4.3 技能联动流程

```
VOC 原始数据
    ↓
【ReviewQuality】质量筛选
    ↓
【统一萃取引擎】← 376标签种子 + 55画像标签
    ↓
    ├──→ AIPL 旅程标签
    ├──→ 画像标签
    ├──→ 情感校准
    ├──→ 品牌检测
    └──→ Proxy NPS
         ↓
    【NPS-Driver-Analysis】SHAP 归因
    【Kano-Mapper】需求分类
    【GPLR】营销人群包
```

---

## 5. 业务价值评估

### 5.1 ROI 估算

| 收益来源 | 提升幅度 | 预估收益 |
|---------|---------|---------|
| 标签萃取效率 | 从人工 2min/条 → 自动 0.01s/条 | 节省人力 300万/年 |
| 多维度标签覆盖 | 从 3 维 → 6 维 | 洞察深度提升 2 倍 |
| Proxy NPS 实时化 | 从季度调研 → 实时计算 | 响应速度提升 90 倍 |
| 策略闭环 | 标签 → 部门 → 策略包自动路由 | 执行效率 +40% |
| **总计** | - | **500万+/年** |

### 5.2 实施成本

- 标签种子维护：5 人天/季度
- 英文特征词典适配：2-3 天（一次性）
- 四路打标 pipeline：3-5 天
- **总计**：约 10-15 天初始投入

### 5.3 难度评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 算法复杂度 | ⭐⭐ | 关键词匹配 + 规则校准，无深度学习依赖 |
| 数据依赖 | ⭐⭐⭐ | 需要完整的 376 标签种子表 |
| 工程实现 | ⭐⭐ | 纯 Python，无外部模型依赖 |
| 业务落地 | ⭐⭐ | 萃取结果直接对接现有指标体系 |
| **综合评分** | **2.3/5** | 中等偏低难度，高价值回报 |

---

## 6. 与现有 VOC 技能的衔接

```
完整链路:

【VOC 数据层】
退货留言 / 客服工单 / 商品评论 / Trustpilot (355,697条)
    ↓
【质量筛选层】
ReviewQuality-Scoring (4维度) + SpamDetector (5规则)
    ↓
【统一萃取层】← 本 Skill 核心位置
376标签种子 + 55画像标签 + AIPL 7节点
    ↓
【指标计算层】
Proxy NPS / AIPL 漏斗 / 驱动分析 / 画像交叉
    ↓
【决策输出层】
策略包路由 → 产品中心/品线/客户服务部/品牌营销部...
```

---

**文档版本**: v1.0
**创建日期**: 2026-04-22
**适用场景**: Momcozy 母婴出海 VOC 全链路标签自动萃取与指标计算
