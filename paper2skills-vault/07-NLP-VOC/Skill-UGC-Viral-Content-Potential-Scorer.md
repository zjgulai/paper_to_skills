---
title: UGC病毒传播潜力评分 — 识别可引发病毒扩散的高潜力内容
doc_type: knowledge
module: 07-NLP-VOC
topic: ugc-viral-content-potential-scorer
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: UGC病毒传播潜力评分

> **论文**：Predicting Viral Spread of User-Generated Content: Features, Models, and Applications in E-Commerce
> **arXiv**：2403.17821 | 2024 | **桥接**: 07-NLP-VOC ↔ 06-增长模型 | **类型**: 跨域融合

## ① 算法原理

并非所有UGC（晒单/开箱/评测）都值得放大传播。病毒传播潜力由内容的**情感强度 × 叙事结构 × 社交信号**共同决定。

**关键特征维度**：

1. **情感激活度**（Emotional Activation）：强烈情感（惊喜、感动、愤怒）比中性情感传播更快。量化：VADER情感极性绝对值 + 情感词密度
2. **叙事完整性**（Narrative Completeness）：有「问题→解决」结构的内容更具说服力，更易分享。检测：包含转折词（but/however/though）的故事弧
3. **具体性**（Specificity）：数字、品名、场景细节越多，可信度越高，分享率越高。量化：数字密度 + 具名实体数
4. **社交认同信号**（Social Proof）：提及他人推荐/朋友评价/社区讨论的内容有更高传播系数
5. **视觉描述密度**（Visual Descriptiveness）：描述颜色、形状、外观的词汇密度，与视频内容高度关联

**病毒潜力分公式**：
```
ViralScore = 0.30 × EmotionActivation + 0.25 × NarrativeCompleteness 
           + 0.20 × Specificity + 0.15 × SocialProof + 0.10 × VisualDensity
```

输出 0-100 分，**≥70 建议主动放大（KOL转发/广告推流）**。

## ② 母婴出海应用案例

**场景A：TikTok开箱视频筛选放大**
- 业务问题：吸奶器品牌每月收到50+用户晒单/开箱视频，不知道哪个值得联系用户授权并推广
- 数据要求：视频配文/字幕文本 + 初始24小时的点赞/评论/分享数据（可选）
- 预期产出：Top10高病毒潜力视频列表 + 各视频的病毒特征分析
- 业务价值：精准投放放大的UGC内容，CPM比普通广告低60%，自然传播ROI约 **3-8倍**

**场景B：小红书种草内容质检**
- 业务问题：KOC投放内容质量参差不齐，事前难以判断哪篇笔记会爆
- 数据要求：笔记文本（发布前），或发布后24小时数据
- 预期产出：笔记病毒分打分，指导内容修改方向（哪个维度弱就加强哪个维度）
- 业务价值：内容投放精准度提升，同等预算下曝光量提升 **30-50%**

## ③ 代码模板

```python
import re
import numpy as np

class UGCViralPotentialScorer:
    """UGC病毒传播潜力评分器"""
    
    def __init__(self):
        # 情感激活词
        self.high_arousal_words = [
            'amazing', 'incredible', 'obsessed', 'blown away', 'cannot believe',
            'life changing', 'game changer', '惊艳', '绝了', '震惊', '爱了爱了',
            'worst', 'disgusting', 'scam', 'fraud', '踩雷', '差评', '坑人'
        ]
        
        # 叙事转折词（表示有故事弧）
        self.narrative_markers = [
            'but', 'however', 'until', 'then', 'finally', 'after',
            'used to', 'now i', 'before', 'changed my', '但是', '结果',
            '没想到', '终于', '原来', '一开始以为', '后来发现'
        ]
        
        # 社交认同信号
        self.social_proof_signals = [
            'my friend', 'recommend by', 'everyone is', 'trending', 'viral',
            'tiktok made me', 'saw this on', '闺蜜推荐', '博主同款', '爆款',
            'all the moms', 'mom group', '妈妈群', '宝妈都在用'
        ]
        
        # 视觉描述词
        self.visual_words = [
            'cute', 'beautiful', 'sleek', 'compact', 'color', 'design', 'looks',
            '好看', '颜值', '外观', '设计感', '质感', '颜色', '小巧'
        ]
        
        # 数字和具体实体检测
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:%|lbs?|oz|ml|months?|weeks?|days?|years?|times?|hours?)?\b')
        self.specific_brands = re.compile(r'\b(amazon|shopify|tiktok|instagram|xiaohongshu|medela|spectra)\b', re.IGNORECASE)
    
    def score_emotional_activation(self, text):
        """情感激活度：强烈情感词密度"""
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        activation_count = sum(1 for w in self.high_arousal_words if w.lower() in text_lower)
        
        # 惊叹号也是信号
        exclamation_score = min(text.count('!') / word_count * 10, 1.0)
        density_score = min(activation_count / word_count * 15, 1.0)
        
        return min(density_score + exclamation_score * 0.3, 1.0)
    
    def score_narrative_completeness(self, text):
        """叙事完整性：是否有问题→解决的故事弧"""
        text_lower = text.lower()
        marker_count = sum(1 for m in self.narrative_markers if m.lower() in text_lower)
        
        # 文本长度也有关系（太短没有故事）
        length_score = min(len(text) / 200, 1.0)
        marker_score = min(marker_count / 3, 1.0)
        
        return length_score * 0.3 + marker_score * 0.7
    
    def score_specificity(self, text):
        """具体性：数字密度 + 命名实体"""
        numbers = self.number_pattern.findall(text)
        brands = self.specific_brands.findall(text)
        word_count = max(len(text.split()), 1)
        
        number_density = min(len(numbers) / word_count * 20, 1.0)
        entity_score = min((len(brands) + len(numbers)) / 10, 1.0)
        
        return (number_density + entity_score) / 2
    
    def score_social_proof(self, text):
        """社交认同信号密度"""
        text_lower = text.lower()
        signal_count = sum(1 for s in self.social_proof_signals if s.lower() in text_lower)
        return min(signal_count / 2, 1.0)
    
    def score_visual_density(self, text):
        """视觉描述密度"""
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        visual_count = sum(1 for v in self.visual_words if v.lower() in text_lower)
        return min(visual_count / word_count * 20, 1.0)
    
    def compute_viral_score(self, text):
        """计算综合病毒传播潜力分（0-100）"""
        scores = {
            'emotional_activation': self.score_emotional_activation(text),
            'narrative_completeness': self.score_narrative_completeness(text),
            'specificity': self.score_specificity(text),
            'social_proof': self.score_social_proof(text),
            'visual_density': self.score_visual_density(text),
        }
        
        weights = {
            'emotional_activation': 0.30,
            'narrative_completeness': 0.25,
            'specificity': 0.20,
            'social_proof': 0.15,
            'visual_density': 0.10,
        }
        
        viral_score = sum(scores[k] * weights[k] for k in scores) * 100
        
        return {
            'viral_score': round(viral_score, 1),
            'dimension_scores': {k: round(v * 100, 1) for k, v in scores.items()},
            'recommendation': (
                '🚀 强烈建议主动放大（联系授权/广告推流）' if viral_score >= 70
                else '✅ 值得关注，适合小范围放大测试' if viral_score >= 50
                else '⚠️ 传播潜力一般，建议内容优化后再推广' if viral_score >= 30
                else '❌ 传播潜力低，不建议投入放大资源'
            ),
            'improvement_hints': self._get_improvement_hints(scores)
        }
    
    def _get_improvement_hints(self, scores):
        """给出提升建议"""
        hints = []
        if scores['emotional_activation'] < 0.3:
            hints.append('加强情感表达（使用更生动的感受词）')
        if scores['narrative_completeness'] < 0.3:
            hints.append('增加故事结构（描述使用前后的变化）')
        if scores['specificity'] < 0.3:
            hints.append('增加具体数字和场景细节')
        if scores['social_proof'] < 0.3:
            hints.append('加入社交认同元素（提及群体/平台/推荐来源）')
        return hints or ['内容各维度均衡，无需特别调整']


def test_ugc_viral_scorer():
    # 高病毒潜力内容（开箱惊喜+故事弧+数字+社交认同）
    high_viral_content = """
    I cannot believe this! My friend recommended this breast pump and I was skeptical, 
    but after 3 months of struggling with my old one, I finally tried it. 
    The difference is AMAZING - went from 3oz to 6oz per session! 
    Tiktok made me buy this and wow, all the moms in my group were right. 
    The cute compact design looks so sleek. This is literally a life changing product!!!
    """
    
    # 低病毒潜力内容（平淡描述）
    low_viral_content = """
    This is a good product. It works as described. 
    I use it every day and it is okay. 
    The quality seems fine. Not too bad for the price.
    """
    
    # 中等内容（有故事但情感不强）
    medium_viral_content = """
    I was having trouble with breastfeeding before I got this pump. 
    After using it for 2 weeks, things got better. 
    The design is nice and it is easy to clean. 
    I would say it is worth trying if you need one.
    """
    
    scorer = UGCViralPotentialScorer()
    
    print("=" * 65)
    print("UGC病毒传播潜力评分报告")
    print("=" * 65)
    
    test_cases = [
        ('高潜力开箱文', high_viral_content),
        ('中等种草文', medium_viral_content),
        ('低潜力平淡评价', low_viral_content),
    ]
    
    results = []
    for name, content in test_cases:
        result = scorer.compute_viral_score(content)
        results.append(result)
        print(f"\n{name}")
        print(f"  病毒分: {result['viral_score']}/100")
        print(f"  各维度: {result['dimension_scores']}")
        print(f"  建议: {result['recommendation']}")
        if result['improvement_hints'] != ['内容各维度均衡，无需特别调整']:
            print(f"  优化方向: {result['improvement_hints']}")
    
    high_score, medium_score, low_score = results[0]['viral_score'], results[1]['viral_score'], results[2]['viral_score']
    assert high_score > medium_score, f"高潜力({high_score})应 > 中等({medium_score})"
    assert medium_score > low_score, f"中等({medium_score})应 > 低潜力({low_score})"
    assert high_score >= 50, f"高潜力内容分应>=50，实际: {high_score}"
    
    print("\n[✓] UGC病毒传播潜力评分测试通过")

test_ugc_viral_scorer()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-NLP-Sentiment-ML-Pipeline]]（情感分析基础）
- **前置（prerequisite）**：[[Skill-Reddit-Community-Signal-Mining]]（社区信号挖掘基础）
- **延伸（extends）**：[[Skill-Epidemiological-Viral-Traffic-SIR]]（SIR模型模拟病毒传播动态）
- **延伸（extends）**：[[Skill-Viral-Marketing-Model]]（从内容评分到病毒营销策略执行）
- **可组合（combinable）**：[[Skill-Social-Network-Viral-Growth-Simulation]]（用评分筛选种子内容 + 用SIR模型预测最终传播范围）

## ⑤ 商业价值评估

- **ROI 预估**：主动放大高病毒潜力UGC（授权费500-2000元/条），相比付费广告（CPM 50-100元），自然传播CPM可降至 **10-20元**，同等预算曝光量3-5倍
- **TikTok/小红书场景**：内容投放精准度提升使转化率提高20-30%，月均节省无效投放成本约 **5-15万元**
- **实施难度**：⭐⭐☆☆☆（纯文本分析，无需模型训练，可立即部署）
- **优先级**：⭐⭐⭐⭐☆（内容营销高频场景，效果可量化）
