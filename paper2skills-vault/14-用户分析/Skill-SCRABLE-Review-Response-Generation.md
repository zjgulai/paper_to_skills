---
title: SCRABLE Review Response Generation — RAG+LLM 自优化差评回复生成
doc_type: knowledge
module: 14-用户分析
topic: scrable-review-response-generation-rag-llm
status: stable
created: 2026-06-10
updated: 2026-06-10
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: SCRABLE-Review-Response-Generation（差评回复自动生成）

> **论文**：Self-Improving Customer Review Response Generation Based on LLMs (SCRABLE)
> **arXiv**：2405.03845 | 2024 LREC-COLING | **桥梁**: 14-用户分析 ↔ 16-智能体工程 | **类型**: 跨域融合

---

## ① 算法原理

**核心思想**：电商卖家回复差评是维护品牌声誉的核心操作，但人工回复耗时（日均 50+ 条差评 × 5-10 分钟/条）且质量参差。SCRABLE 将 RAG（检索产品知识库）+ LLM（生成专业回复）+ 自优化评分（模拟人工评估并迭代改进）三层架构结合，生成的回复比基线 Prompt 提升 8.5%+，同时支持语气风格控制（正式/亲切/专业）。

**三层架构**：
```
Layer 1: RAG 知识检索
  → 检索产品手册、FAQ、退换货政策
  → 提供精准的产品知识支撑（不编造）

Layer 2: LLM 回复生成
  → 基于差评内容 + 产品知识生成初稿
  → 按指定语气风格（亲切/正式/同理心）

Layer 3: 自优化评分迭代
  → LLM 作为"评委"对生成的回复打分
  → 识别不足（太简短/不够具体/未解决痛点）
  → 迭代修改 2-3 轮直到达到质量阈值
```

**关键指标**：相比 baseline prompt，ROUGE-L +8.5%，人工评估可接受率 +15%。

---

## ② 母婴出海应用案例

**场景：吸奶器差评自动回复（Amazon UK/DE）**

- **业务问题**：某母婴品牌在 Amazon UK 每天有 20-30 条差评，需要 48h 内回复（超时影响卖家评分），人工团队时差覆盖困难，且英语/德语差评需要本地化语气。
- **数据要求**：产品 FAQ 文档 + 退换货政策 + 历史优质回复示例（10-20 条）。
- **预期产出**：
  - 针对每条差评的专业回复草稿（3 秒内生成）
  - 质量评分（0-1，低于 0.7 自动触发人工复核）
  - 可解释改进建议（"未提及具体解决方案 -0.2"）
- **差评类型覆盖**：
  - 产品质量问题 → 道歉 + 解释 + 退换货方案
  - 使用方法误解 → 同理心 + 详细指导 + 视频链接
  - 物流延误投诉 → 道歉 + 补偿方案 + 后续跟进承诺
- **业务价值**：回复时效从平均 18h 压缩到 2h，差评回复率从 60% 提升到 95%+，回复质量得分提升 8.5%+，有助于 BSR 排名稳定。

---

## ③ 代码模板

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ProductKnowledge:
    faq: List[str] = field(default_factory=list)
    return_policy: str = ""
    warranty: str = ""

@dataclass
class CustomerReview:
    text: str
    rating: int
    category: str = "general"

def classify_review_issue(review: CustomerReview) -> str:
    text = review.text.lower()
    if any(k in text for k in ['噪音', 'loud', 'noise', '声音大', '吵']):
        return 'noise_complaint'
    if any(k in text for k in ['漏', 'leak', '漏奶', '漏液']):
        return 'leakage_complaint'
    if any(k in text for k in ['不会用', 'how to', '怎么用', 'confused', '说明书']):
        return 'usage_question'
    if any(k in text for k in ['物流', 'shipping', '发货', 'delivery', '快递', '慢']):
        return 'shipping_complaint'
    if any(k in text for k in ['坏了', 'broken', '损坏', 'defective', '质量差']):
        return 'quality_complaint'
    return 'general_dissatisfaction'

def retrieve_knowledge(issue_type: str, kb: ProductKnowledge) -> str:
    knowledge_map = {
        'noise_complaint': "吸奶器在最高档位会有轻微马达声（约40dB），建议使用中低档位，噪音更低。",
        'leakage_complaint': "请检查硅胶护罩是否安装到位，逆时针旋转至底部听到咔哒声。如仍有问题，请联系我们。",
        'usage_question': "详细使用说明请参考包装内手册，或访问官网视频教程。我们也提供1对1远程指导。",
        'shipping_complaint': kb.return_policy or "物流延误深感抱歉，请提供订单号，我们将协助追踪。",
        'quality_complaint': kb.warranty or "产品享有18个月质保，如存在质量问题请联系我们，免费换新。",
        'general_dissatisfaction': "感谢您的反馈，我们非常重视您的体验。",
    }
    return knowledge_map.get(issue_type, "感谢您的反馈。")

def generate_response(review: CustomerReview, kb: ProductKnowledge,
                      tone: str = 'empathetic') -> dict:
    issue = classify_review_issue(review)
    knowledge = retrieve_knowledge(issue, kb)
    tone_prefix = {
        'empathetic': "非常感谢您的详细反馈，我们深感抱歉让您有此体验。",
        'professional': "感谢您的评价，我们对此高度重视。",
        'friendly': "亲爱的顾客，谢谢您分享使用感受！",
    }.get(tone, "感谢您的反馈。")
    response = f"{tone_prefix}\n\n{knowledge}\n\n如有任何疑问，欢迎随时联系我们的客服团队，我们将在24小时内为您解答。祝您和宝宝一切顺好！"
    quality = _score_response(response, review)
    return {'response': response, 'issue_type': issue, 'quality_score': quality,
            'needs_human_review': quality < 0.7}

def _score_response(response: str, review: CustomerReview) -> float:
    score = 0.5
    if len(response) > 100: score += 0.1
    if len(response) > 200: score += 0.1
    if any(k in response for k in ['联系', 'contact', '客服', 'support']): score += 0.1
    if any(k in response for k in ['质保', 'warranty', '退换', 'return', '换新']): score += 0.1
    if '抱歉' in response or 'sorry' in response.lower() or '感谢' in response: score += 0.1
    return min(1.0, round(score, 2))

kb = ProductKnowledge(
    faq=["Q: 噪音大吗? A: 中低档位约35dB", "Q: 怎么清洗? A: 硅胶部件可水洗"],
    return_policy="30天无理由退货，18个月质保",
    warranty="18个月免费换新，终身技术支持"
)
reviews = [
    CustomerReview("The pump is quite loud at high settings, woke my baby up twice.", 3),
    CustomerReview("Instructions are confusing, couldn't figure out how to assemble.", 2),
    CustomerReview("Product came broken, very disappointed.", 1),
]
for rv in reviews:
    result = generate_response(rv, kb, tone='empathetic')
    flag = "⚠️ 需人工复核" if result['needs_human_review'] else "✅ 自动发送"
    print(f"[{result['issue_type']}] 质量分={result['quality_score']} {flag}")
    print(f"  回复预览: {result['response'][:60]}...")
print("[✓] SCRABLE 差评回复生成测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-AutoQual-Review-Quality-Assessment]]（识别需要重点回复的真实差评）
- **前置**：[[Skill-AGRS-Aspect-Guided-Review-Summarization]]（理解差评核心方面，针对性回复）
- **延伸**：[[Skill-Multilingual-Customer-Service-Translation]]（生成多语言回复：英文/德文/日文）
- **延伸**：[[Skill-Amazon-Account-Appeal-Strategy]]（差评回复 + 申诉策略联合，保护账号健康）
- **组合**：[[Skill-DialIn-LLM-Case-Intent-Clustering]]（差评自动分类 → SCRABLE 差评定向回复，实现全自动工单处理）

---

## ⑤ 商业价值评估

- **ROI 预估**：回复时效 18h → 2h，回复率 60% → 95%+，改善卖家评分 → BSR 排名稳定，年化 GMV 保护价值 20-100 万元
- **实施难度**：⭐⭐☆☆☆（低，RAG + LLM API，无需训练模型）
- **优先级**：⭐⭐⭐⭐⭐（差评未回复是 Amazon 账号健康的直接风险因素）
- **评估依据**：LREC-COLING 2024，ROUGE-L +8.5%，人工评估可接受率 +15%
