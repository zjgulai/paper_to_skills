---
title: LLM多语言Listing批量生成 — 1小时完成5语言新品入驻
doc_type: knowledge
module: 20-AI视频生成
topic: multilingual-listing-generation
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: LLM多语言Listing批量生成

> **论文**：PPLM: Plug and Play Language Models（ICLR 2020）+ BLOOM Multilingual LLM 工程实践（2023）
> **arXiv**：1912.02164 | 2020 | **桥梁**: 20-AI视频生成 ↔ 07-NLP-VOC | **类型**: 工程基础

## ① 算法原理

**核心思想**：基于大语言模型（LLM）的 Few-shot Prompt 工程 + 结构化约束生成，将商品核心属性自动转化为符合各平台规范的多语言 Listing（标题/五点/描述）。关键创新是**品类词库注入**（Category Keyword Injection）和**关键词密度控制**（Keyword Density Control）。

**数学直觉**：
```
P(Listing | product_info, category_keywords, lang) = 
    LLM(system_prompt + few_shot_examples + product_template)
```

- **Few-shot Prompt**：提供 2-3 个同品类高分 Listing 作为示例，引导模型学习结构和语气
- **关键词密度**：标题关键词覆盖率 = |target_keywords ∩ generated_title_words| / |target_keywords|
- **模板约束**：硬约束标题字数（Amazon: ≤200 字节；DE 站：≤80 字节）

**关键假设**：
- 商品核心属性（品类/材质/适用年龄/功能/认证）已结构化录入
- Few-shot 示例来自当前品类 Best Seller，质量有保障
- 不同语言 Listing 的 SEO 关键词需提前准备（可用品类关键词库）

## ② 母婴出海应用案例

**场景A：新品同步上架 5 国站点**
- 业务问题：一款吸奶器新品需同时上架 US/DE/FR/JP/UK 5 站，人工翻译+撰写需 3-5 天，错过节日首发窗口
- 数据要求：
  - 商品基础信息：品名、材质、功能、认证（CE/FDA）、适用年龄
  - 各语言种子关键词：3-5 个高搜索量词（可从 Helium10 导出）
  - 竞品 Best Seller Listing 样本：各站各 2-3 个（作为 Few-shot 示例）
- 预期产出：5 语言 × (标题 + 五点 × 5 + 描述) = 55 条文案，1 小时内生成
- 业务价值：**节省翻译费用 $800-1,200/SKU**（按专业母语翻译计），首发提前 3 天，节日期间 GMV 增量约 15%

**场景B：存量 SKU Listing 关键词刷新**
- 业务问题：旺季前需对 50 个核心 SKU 刷新 Listing，嵌入当季热词（如"Christmas gift for baby"）
- 数据要求：原有 Listing + 当季热词列表 + 平台关键词搜索量数据
- 预期产出：关键词密度报告 + 优化后 Listing（替换低效词段）
- 业务价值：旺季 ACoS 降低约 8%，自然流量词命中率提升 20%

## ③ 代码模板

```python
"""
多语言 Listing 批量生成引擎
基于 LLM Few-shot + 品类词库注入 + 关键词密度控制
生产环境接入 OpenAI / DeepSeek / Claude API（当前为 Mock 演示）
"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProductInfo:
    """商品基础信息结构"""
    product_name: str               # 商品名（中文或英文）
    category: str                   # 品类（如 breast pump）
    material: str                   # 材质（如 BPA-free silicone）
    age_range: str                  # 适用年龄（如 0-24 months）
    key_features: List[str]         # 核心功能，3-5 条
    certifications: List[str]       # 认证（CE, FDA, BPA-free）
    brand: str                      # 品牌名


@dataclass
class ListingResult:
    """单语言 Listing 生成结果"""
    language: str
    title: str
    bullet_points: List[str]        # 五点描述，5 条
    description: str
    keyword_coverage: float         # 关键词覆盖率 0-1
    char_count_title: int           # 标题字符数


# 各站点 Listing 规范约束
PLATFORM_CONSTRAINTS = {
    "US": {"title_max_bytes": 200, "bullet_count": 5, "desc_max_words": 2000},
    "DE": {"title_max_bytes": 80,  "bullet_count": 5, "desc_max_words": 1500},
    "FR": {"title_max_bytes": 80,  "bullet_count": 5, "desc_max_words": 1500},
    "JP": {"title_max_bytes": 120, "bullet_count": 5, "desc_max_words": 1000},
    "UK": {"title_max_bytes": 200, "bullet_count": 5, "desc_max_words": 2000},
}

# 品类种子关键词库（生产环境从 Helium10 / DataDive 同步）
CATEGORY_KEYWORDS = {
    "breast pump": {
        "US": ["breast pump", "electric breast pump", "portable breast pump",
               "wearable breast pump", "hands free breast pump"],
        "DE": ["Milchpumpe", "elektrische Milchpumpe", "tragbare Milchpumpe",
               "kabellose Milchpumpe", "Doppelmilchpumpe"],
        "FR": ["tire-lait", "tire-lait électrique", "tire-lait portable",
               "tire-lait sans fil", "extracteur de lait"],
        "JP": ["搾乳器", "電動搾乳器", "ハンズフリー搾乳器", "携帯搾乳器", "静音搾乳器"],
        "UK": ["breast pump", "electric breast pump", "portable breast pump",
               "hands free breast pump", "silent breast pump"],
    },
    "baby bottle": {
        "US": ["baby bottle", "anti-colic bottle", "wide neck bottle",
               "BPA free bottle", "newborn bottle"],
        "DE": ["Babyflasche", "Anti-Kolik-Flasche", "Weithalsflasche",
               "BPA-freie Flasche", "Neugeborenen-Flasche"],
        "FR": ["biberon", "biberon anti-colique", "biberon col large",
               "biberon sans BPA", "biberon nouveau-né"],
        "JP": ["哺乳瓶", "耐熱哺乳瓶", "広口哺乳瓶", "BPAフリー哺乳瓶", "新生児哺乳瓶"],
        "UK": ["baby bottle", "anti-colic bottle", "wide neck bottle",
               "BPA free bottle", "feeding bottle"],
    },
}

# Few-shot 示例模板（生产环境从竞品 Best Seller 动态采集）
FEW_SHOT_EXAMPLES = {
    "US": """
Title: Momcozy S12 Pro Wearable Breast Pump Hands Free, Electric Portable Breast Pump...
Bullets:
• HANDS-FREE FREEDOM: Wearable design fits in your bra...
• HOSPITAL GRADE SUCTION: 4 modes x 9 levels...
""",
    "DE": """
Title: Milchpumpe Elektrisch Doppel Tragbar, Handsfree Milchpumpe...
Bullets:
• HANDSFREI DESIGN: Passt direkt in den BH...
• KLINISCHE SAUGLEISTUNG: 4 Modi x 9 Stufen...
""",
}


class MockLLMClient:
    """
    LLM API Mock（生产环境替换为真实 API 调用）
    模拟 GPT-4 / Claude / DeepSeek 的结构化输出
    """
    def generate(self, prompt: str, max_tokens: int = 800) -> str:
        """
        生产环境示例（OpenAI）：
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        """
        # Mock 返回结构化 JSON
        return json.dumps({
            "title": "Premium Electric Breast Pump - Hands Free Wearable, 4 Modes 9 Levels, BPA Free, Portable Quiet for Travel",
            "bullet_points": [
                "HANDS-FREE WEARABLE DESIGN: Fits discreetly in nursing bra, no tubes or wires needed for complete freedom",
                "HOSPITAL GRADE SUCTION: 4 stimulation + expression modes with 9 adjustable levels, mimic natural nursing",
                "ULTRA QUIET & PORTABLE: <45dB whisper-quiet motor, 150ml capacity per side, rechargeable 2500mAh battery",
                "BPA-FREE FOOD GRADE: All milk-contact parts FDA approved BPA-free silicone, safe for baby",
                "SMART LCD DISPLAY: Real-time display of mode, level and battery; includes carry bag, spare parts",
            ],
            "description": "Designed for modern moms who refuse to slow down. The wearable breast pump fits seamlessly into your daily routine - pump at work, in the car, or while caring for your baby. With hospital-grade suction technology and whisper-quiet operation, you get efficient pumping without compromise. Backed by 12-month warranty and dedicated customer support.",
        }, ensure_ascii=False)


def build_listing_prompt(
    product: ProductInfo,
    language: str,
    keywords: List[str],
    few_shot: str,
    constraints: Dict,
) -> str:
    """构建 Few-shot Prompt"""
    lang_instruction = {
        "US": "Write in American English",
        "DE": "Write in German (Deutsch)",
        "FR": "Write in French (Français)",
        "JP": "Write in Japanese (日本語)",
        "UK": "Write in British English",
    }.get(language, "Write in English")

    return f"""You are an Amazon listing copywriter specializing in baby products.
{lang_instruction}. Return valid JSON only.

PRODUCT INFO:
- Name: {product.product_name}
- Category: {product.category}
- Material: {product.material}
- Age Range: {product.age_range}
- Features: {', '.join(product.key_features)}
- Certifications: {', '.join(product.certifications)}
- Brand: {product.brand}

TARGET KEYWORDS (must include ≥3 in title): {', '.join(keywords[:5])}

PLATFORM CONSTRAINTS:
- Title: max {constraints['title_max_bytes']} bytes
- Bullet Points: exactly {constraints['bullet_count']} bullets
- Each bullet: start with ALL CAPS keyword phrase

REFERENCE EXAMPLES:
{few_shot}

OUTPUT FORMAT (JSON):
{{
  "title": "...",
  "bullet_points": ["...", "...", "...", "...", "..."],
  "description": "..."
}}"""


def calculate_keyword_coverage(text: str, keywords: List[str]) -> float:
    """计算关键词覆盖率（大小写不敏感）"""
    text_lower = text.lower()
    covered = sum(1 for kw in keywords if kw.lower() in text_lower)
    return covered / len(keywords) if keywords else 0.0


def generate_listing(
    product: ProductInfo,
    language: str,
    llm_client: MockLLMClient,
    custom_keywords: Optional[List[str]] = None,
) -> ListingResult:
    """
    为指定语言生成 Listing

    Args:
        product: 商品信息
        language: 目标语言（US/DE/FR/JP/UK）
        llm_client: LLM 客户端
        custom_keywords: 自定义关键词（覆盖品类默认词库）

    Returns:
        ListingResult: 完整 Listing + 质量指标
    """
    constraints = PLATFORM_CONSTRAINTS[language]

    # 获取关键词（优先使用自定义词，回退到品类词库）
    if custom_keywords:
        keywords = custom_keywords
    else:
        cat_key = product.category.lower()
        keywords = CATEGORY_KEYWORDS.get(cat_key, {}).get(language, [])

    # 获取 Few-shot 示例
    few_shot = FEW_SHOT_EXAMPLES.get(language, FEW_SHOT_EXAMPLES.get("US", ""))

    # 构建 Prompt
    prompt = build_listing_prompt(product, language, keywords, few_shot, constraints)

    # 调用 LLM
    raw_response = llm_client.generate(prompt)

    # 解析 JSON 响应
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        # 容错：尝试提取 JSON 块
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else {}

    title = data.get("title", "")
    bullets = data.get("bullet_points", [])
    description = data.get("description", "")

    # 计算关键词覆盖率（以标题为主）
    coverage = calculate_keyword_coverage(title + " " + " ".join(bullets), keywords)

    return ListingResult(
        language=language,
        title=title,
        bullet_points=bullets,
        description=description,
        keyword_coverage=coverage,
        char_count_title=len(title.encode("utf-8")),
    )


def batch_generate_multilingual(
    product: ProductInfo,
    target_languages: List[str],
    llm_client: MockLLMClient,
) -> Dict[str, ListingResult]:
    """批量生成多语言 Listing"""
    results = {}
    for lang in target_languages:
        if lang not in PLATFORM_CONSTRAINTS:
            print(f"⚠️  不支持的站点: {lang}，跳过")
            continue
        results[lang] = generate_listing(product, lang, llm_client)
    return results


# ===== 测试用例 =====
if __name__ == "__main__":
    # 商品信息
    sample_product = ProductInfo(
        product_name="MomFlow Pro 双边吸奶器",
        category="breast pump",
        material="BPA-free silicone, food-grade PP",
        age_range="0-24 months",
        key_features=[
            "双边同时吸奶，节省50%时间",
            "4档吸力模式 x 9级强度",
            "免提穿戴设计，适合工作场合",
            "超静音<45dB，可办公室使用",
            "USB-C充电，续航8小时",
        ],
        certifications=["CE", "FDA", "BPA-free", "ISO 9001"],
        brand="MomFlow",
    )

    llm = MockLLMClient()
    target_langs = ["US", "DE", "FR", "JP", "UK"]

    print("=" * 65)
    print("多语言 Listing 批量生成报告")
    print("=" * 65)

    all_results = batch_generate_multilingual(sample_product, target_langs, llm)

    for lang, result in all_results.items():
        constraints = PLATFORM_CONSTRAINTS[lang]
        title_ok = result.char_count_title <= constraints["title_max_bytes"]
        bullet_ok = len(result.bullet_points) == constraints["bullet_count"]

        print(f"\n{'='*20} {lang} 站 {'='*20}")
        print(f"标题（{result.char_count_title}/{constraints['title_max_bytes']}B）"
              f"{'✅' if title_ok else '⚠️ 超长'}: {result.title[:60]}...")
        print(f"五点 {'✅' if bullet_ok else '❌'}: {len(result.bullet_points)} 条")
        print(f"关键词覆盖率: {result.keyword_coverage:.0%}")
        print(f"描述字数: {len(result.description.split())} words")

    # 断言验证
    assert len(all_results) == len(target_langs), "生成语言数量不符"
    for lang, result in all_results.items():
        assert result.title, f"{lang} 标题为空"
        assert len(result.bullet_points) > 0, f"{lang} 五点为空"
        assert result.description, f"{lang} 描述为空"
        assert 0.0 <= result.keyword_coverage <= 1.0, f"{lang} 覆盖率超出范围"

    total_time_saved = len(target_langs) * 3  # 每语言人工平均 3 小时
    print(f"\n估算节省翻译时间: {total_time_saved} 小时")
    print(f"估算节省翻译费用: ${len(target_langs) * 200}-${len(target_langs) * 250}")

    print("\n[✓] LLM多语言Listing批量生成 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-NLP-Text-Classification]]（理解文本结构化分析基础）
- **前置（prerequisite）**：[[Skill-VOC-Aspect-Sentiment-Extraction]]（从评论提取用户关注点作为 Listing 输入）
- **延伸（extends）**：[[Skill-GEO-Generative-Engine-Optimization]]（Listing 生成后进行 AI 搜索引擎优化）
- **可组合（combinable）**：[[Skill-A-Plus-Content-Template-Engine]]（Listing 生成后自动产出 A+ Content，形成内容全套）

## ⑤ 商业价值评估

- **ROI 预估**：
  - 直接节省翻译费：$200-250/语言/SKU × 5 语言 = **$1,000-1,250/SKU**
  - 时间节省：3 天 → 1 小时，节假日首发提前对应 GMV 增量约 **$5,000/活动**（保守估计）
  - 按月均 20 个新品计：月节省成本 **$20,000-25,000**
- **实施难度**：⭐⭐☆☆☆（2/5）— 调用 LLM API 即可，工程复杂度低
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 直接替代高频人工操作，ROI 清晰，1 周可上线
- **评估依据**：多语言扩张是母婴出海核心战略，Listing 质量直接影响搜索排名和转化，自动化空间极大
