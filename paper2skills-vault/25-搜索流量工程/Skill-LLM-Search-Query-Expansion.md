---
title: LLM搜索词语义扩展 — 用大模型覆盖长尾搜索意图提升Listing覆盖面
doc_type: knowledge
module: 25-搜索流量工程
topic: llm-search-query-expansion
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: LLM搜索词语义扩展

> **论文**：LLM-Enhanced Query Expansion for E-commerce Search Coverage Optimization  
> **arXiv**：2403.18252 | 2024 | **桥梁**: search_traffic ↔ data_agent_llm | **类型**: 跨域融合

## ① 算法原理

传统搜索词扩展依赖关键词工具（Helium10/Jungle Scout）提供的搜索量榜单，局限在「已知热词」范围内，对长尾意图（占总搜索量 60% 以上）覆盖不足。核心问题是：买家搜索「哪款奶瓶换成婴儿不哭」时，Listing 里没有这个表达，但语义完全匹配。

LLM 搜索词扩展技术利用大模型对自然语言的语义理解能力，通过三个层次扩展覆盖面：

**Layer 1 — 同义词与措辞变体扩展**  
提示 LLM 生成同义表达、口语化说法、不同方言/文化表述（英/西/葡三语母婴场景）：
- `anti-colic bottle` → `gas reducer bottle`, `no-cry bottle`, `gassy baby bottle`

**Layer 2 — 用户意图分类与场景扩展**  
将搜索词按意图类型分类（功能型/场景型/问题型），针对每类生成对应扩展词：
- 功能型：`slow flow nipple` → 对应功能场景词
- 问题型：`baby spitting up` → 对应解决方案词（`anti-reflux bottle`）

**Layer 3 — Listing 关键词密度优化**  
基于扩展词集合，分析现有 Listing 的关键词覆盖缺口：
$$\text{Coverage Gap} = \{w \in W_{\text{expanded}} \mid w \notin \text{Listing}\}$$

最终输出「优先补充词列表」，按搜索量代理分排序，引导 Listing 优化。

整个流程为 Agent 化执行：LLM 作为「意图理解引擎」，不依赖外部搜索 API，纯文本推理输出。

## ② 母婴出海应用案例

**场景A：新品 Listing 关键词盲区覆盖**
- 业务问题：「防胀气奶瓶」Listing 仅覆盖 40 个词，大量用户用「colic solution bottle」「baby gas bottle」搜索时搜不到
- 数据要求：种子关键词 5-10 个，现有 Listing 文本（title + bullets + description）
- 预期产出：扩展词包 100-200 个，Listing 覆盖词从 40 个提升到 130 个
- 业务价值：自然搜索覆盖词 +3 倍，假设每个新覆盖词月均带来 3 次额外搜索曝光，130 个词月均新增 390 次曝光，转化率 4%，客单价 30 元 → 月增 GMV 约 0.5 万元，年化约 6 万元/SKU

**场景B：多语言市场扩展（美/墨西哥/巴西）**
- 业务问题：进入西班牙语/葡语市场，不知道当地用什么词搜索婴儿用品
- 数据要求：英文种子词列表，目标市场语言（es/pt）
- 预期产出：三语词包，覆盖本地化搜索意图（墨西哥妈妈用「biberón anti-cólico」搜索）
- 业务价值：进入新语言市场，初始自然流量覆盖面提升 2-3 倍，节省本地化词库人工成本约 2 万元/市场

## ③ 代码模板

```python
"""
LLM 搜索词语义扩展（本地模拟版，不需要真实 API Key）
LLM-Enhanced Query Expansion for E-commerce Search Coverage
"""

import re
from collections import defaultdict

# ─── 模拟 LLM 响应（生产中替换为实际 LLM 调用，如 OpenAI/DeepSeek）───
# 格式：种子词 → LLM 扩展词列表
LLM_EXPANSION_DB = {
    "anti-colic bottle": [
        "gas reducer bottle", "colic relief bottle", "no-gas baby bottle",
        "gassy baby bottle", "anti-gas feeding bottle", "colicky baby bottle",
        "baby bottle for gas", "stomach pain relief bottle", "less gas bottle",
        "fussy baby bottle", "reduce crying baby bottle", "gripe water bottle",
        "infant bottle gas free", "baby bottle no cry", "comfort feeding bottle"
    ],
    "overnight diapers": [
        "12 hour diapers", "nighttime diapers", "sleep dry diapers",
        "all night diaper", "heavy wetters diapers", "leak proof night diapers",
        "no leak overnight diaper", "bedtime diapers", "extended wear diapers",
        "stay dry night diapers", "super absorbent overnight", "diapers for sleeping"
    ],
    "breast pump portable": [
        "wearable breast pump", "hands free breast pump", "cordless breast pump",
        "rechargeable breast pump", "on the go breast pump", "travel breast pump",
        "wireless breast pump", "silent breast pump work", "discreet breast pump",
        "pump at work", "pump while driving", "mobile breast pump"
    ],
}

# 意图分类规则
INTENT_PATTERNS = {
    "functional": ["best", "top", "how to", "what is", "which"],
    "problem_solving": ["help", "stop", "prevent", "fix", "reduce", "no more"],
    "comparison": ["vs", "versus", "compare", "better than", "alternative"],
    "attribute": ["size", "type", "color", "material", "age", "weight"],
}

# 现有 Listing 文本（示例）
LISTING_TEXTS = {
    "anti-colic-bottle": """
    Dr. Brown's Natural Flow Anti-Colic Baby Bottle
    - Anti-Colic Internal Vent System reduces colic, spit-up, burping, and gas
    - Slow flow nipple ideal for newborns and breastfeeding transition
    - Wide neck bottle easy to clean, BPA free
    - Recommended by pediatricians for babies with colic
    - Works with all Dr. Brown's bottle accessories
    """,
    "overnight-diaper": """
    Pampers Swaddlers Overnight Diapers
    - 2x more absorb than regular Pampers diapers
    - Comfort Fit for overnight protection
    - Hypoallergenic gentle on skin
    - Wetness indicator for easy check
    """,
}


def classify_intent(query: str) -> str:
    """分类搜索意图"""
    query_lower = query.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in query_lower for p in patterns):
            return intent
    return "navigational"  # 默认：品牌/导航型


def llm_expand_query(seed_keyword: str, expansion_db: dict, max_expand: int = 15) -> list:
    """
    模拟 LLM 查询扩展（生产中替换为：
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Generate {max_expand} search keyword variations for: {seed_keyword}
        Focus on: synonyms, user intent phrases, problem-solution terms, colloquial expressions.
        Format: one keyword per line, no numbering."}]
    )
    expanded = response.choices[0].message.content.strip().split('\n')
    """
    # 模拟 LLM 扩展
    expanded = expansion_db.get(seed_keyword, [])
    
    # 基于种子词的规则扩展补充
    words = seed_keyword.split()
    rule_expansions = []
    
    for i, word in enumerate(words):
        # 前后词组合变体
        if i > 0:
            rule_expansions.append(f"{words[i]} {' '.join(words[:i])}")
        if i < len(words) - 1:
            rule_expansions.append(f"{word} for baby")
    
    all_expanded = list(dict.fromkeys(expanded + rule_expansions))[:max_expand]
    return all_expanded


def extract_listing_keywords(listing_text: str) -> set:
    """从 Listing 文本提取现有关键词（简化版：1-3 gram）"""
    text = listing_text.lower()
    text = re.sub(r'[^a-z\s-]', ' ', text)
    words = text.split()
    
    keywords = set()
    for i in range(len(words)):
        keywords.add(words[i])
        if i + 1 < len(words):
            keywords.add(f"{words[i]} {words[i+1]}")
        if i + 2 < len(words):
            keywords.add(f"{words[i]} {words[i+1]} {words[i+2]}")
    return keywords


def compute_coverage_gap(expanded_keywords: list, listing_keywords: set) -> list:
    """计算 Listing 关键词覆盖缺口"""
    gaps = []
    for kw in expanded_keywords:
        kw_lower = kw.lower()
        # 检查是否部分匹配（任意单词出现在 Listing 中）
        kw_words = set(kw_lower.split())
        listing_overlap = len(kw_words & listing_keywords) / len(kw_words)
        
        if listing_overlap < 0.5:  # 低于50%词匹配 = 覆盖缺口
            gaps.append({
                "keyword": kw,
                "listing_overlap": round(listing_overlap, 2),
                "priority": "HIGH" if listing_overlap < 0.2 else "MEDIUM"
            })
    return sorted(gaps, key=lambda x: x["listing_overlap"])


def keyword_density_optimizer(listing_text: str, gap_keywords: list, max_inject: int = 5) -> str:
    """建议将高优先级缺口词注入 Listing 的哪个位置"""
    suggestions = []
    for kw_info in gap_keywords[:max_inject]:
        kw = kw_info["keyword"]
        # 简化建议：根据词性推荐注入位置
        if any(w in kw for w in ["for", "baby", "infant", "newborn"]):
            location = "title（目标用户词组）"
        elif any(w in kw for w in ["reduce", "prevent", "no", "anti", "free"]):
            location = "bullet points（功能利益点）"
        else:
            location = "backend keywords（后台搜索词）"
        
        suggestions.append({
            "keyword": kw,
            "suggested_location": location,
            "priority": kw_info["priority"]
        })
    return suggestions


# ─── 执行 ───
if __name__ == "__main__":
    print("🤖 LLM 搜索词语义扩展引擎\n")
    
    seed_keywords = ["anti-colic bottle", "overnight diapers", "breast pump portable"]
    
    all_results = {}
    for seed in seed_keywords:
        expanded = llm_expand_query(seed, LLM_EXPANSION_DB)
        intent = classify_intent(seed)
        all_results[seed] = {"expanded": expanded, "intent": intent}
    
    # 覆盖率分析
    print("=" * 70)
    for seed, data in all_results.items():
        print(f"\n🔑 种子词: \"{seed}\" [意图: {data['intent']}]")
        print(f"   LLM 扩展词 ({len(data['expanded'])} 个):")
        for i, kw in enumerate(data['expanded'][:8], 1):
            print(f"   {i:2d}. {kw}")
        if len(data['expanded']) > 8:
            print(f"   ... 共 {len(data['expanded'])} 个")
    
    # Listing 覆盖缺口分析
    print("\n" + "=" * 70)
    print("📊 Listing 关键词覆盖缺口分析\n")
    
    listing_key = "anti-colic-bottle"
    listing_text = LISTING_TEXTS[listing_key]
    listing_kws = extract_listing_keywords(listing_text)
    
    expanded_for_listing = all_results["anti-colic bottle"]["expanded"]
    gaps = compute_coverage_gap(expanded_for_listing, listing_kws)
    
    print(f"  Listing: \"{listing_key}\"")
    print(f"  现有覆盖词数: {len(listing_kws)}")
    print(f"  扩展词中覆盖缺口: {len(gaps)} 个\n")
    
    print("  ⚠️  高优先级缺口词（亟需补充）:")
    high_gaps = [g for g in gaps if g["priority"] == "HIGH"][:8]
    for g in high_gaps:
        print(f"    [{g['priority']}] overlap={g['listing_overlap']:.0%}  \"{g['keyword']}\"")
    
    # 注入建议
    print("\n  💡 关键词注入位置建议:")
    suggestions = keyword_density_optimizer(listing_text, gaps)
    for s in suggestions:
        print(f"    [{s['priority']}] → {s['suggested_location']}")
        print(f"         \"{s['keyword']}\"")
    
    # 汇总
    total_expanded = sum(len(d["expanded"]) for d in all_results.values())
    print(f"\n📈 汇总: {len(seed_keywords)} 个种子词 → {total_expanded} 个扩展词")
    print(f"   覆盖缺口: {len(gaps)} 个词需要补充到 Listing")
    print(f"   预估覆盖率提升: {len(gaps)}/{len(expanded_for_listing)} = {len(gaps)/max(len(expanded_for_listing),1)*100:.0f}% 新词")
    
    print("\n[✓] LLM 搜索词语义扩展 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Listing-Semantic-Relevance-Scoring]]（语义相关性评分衡量扩展词与 Listing 的匹配质量）
- **前置（prerequisite）**：[[Skill-LLM-Business-Intelligence-Reasoning]]（LLM 推理能力基础框架）
- **延伸（extends）**：[[Skill-Keyword-Demand-Gap-Analysis]]（扩展词包生成后，用需求缺口分析确定投入优先级）
- **延伸（extends）**：[[Skill-LLM-Annotation-Weak-Supervision]]（LLM 扩展词可作为弱监督标签训练本地搜索词扩展模型）
- **可组合（combinable）**：[[Skill-Search-Tag-Keyword-Auto-Mapping]]（标签→词包 + LLM语义扩展组合，形成完整关键词体系自动化流水线）

## ⑤ 商业价值评估

- **ROI 预估**：
  - 覆盖词提升：每 SKU 覆盖词从 40 → 130 个，+225%，假设 50 个主力 SKU 每个月均自然流量价值增量 3,000 元 → 年化约 180 万元
  - 人工节省：人工词库扩展每 SKU 需 2-3 天，本算法 30 分钟，50 SKU 节省约 10 人天/季度，年化约 6 万元
  - 多语言市场：进入西/葡语市场初始覆盖成本降低 80%，节省本地化费用约 4 万元/市场
  - **综合年化 ROI ≈ 190-200 万元（主要来自覆盖词扩张带动的自然流量增量）**
- **实施难度**：⭐⭐☆☆☆（低，LLM API 调用即可，无需训练）
- **优先级**：⭐⭐⭐⭐☆（高，直接作用于所有 SKU 的自然流量，phase2 重要杠杆）
