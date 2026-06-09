"""
AIGC Content Detection — AI生成内容鉴别：母婴评论真实性保护
paper2skills-code: 11-AI人文 | 母婴出海跨境电商

纯 Python 标准库实现（无外部依赖）
Python 3.14 兼容
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# 枚举：检测结论
# ──────────────────────────────────────────────

class ContentLabel(Enum):
    """AIGC 鉴别结论"""
    HUMAN = "human"          # 人类写作
    AI_GENERATED = "ai"      # AI 生成
    UNCERTAIN = "uncertain"  # 无法判定


# ──────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────

@dataclass
class TextFeatures:
    """从文本提取的统计特征"""
    token_count: int
    unique_token_ratio: float        # 词汇多样性：唯一词 / 总词数
    sentence_lengths: list[int]      # 各句字数
    avg_sentence_length: float
    sentence_length_variance: float  # 句长均匀性（低 = AI 特征）
    avg_word_length: float           # 平均词长
    repetition_rate: float           # 词汇重复率（高 = AI 特征）
    approx_entropy: float            # 近似困惑度（基于 bigram 分布）
    punctuation_density: float       # 标点密度


@dataclass
class DetectionResult:
    """AIGC 检测结果"""
    label: ContentLabel
    confidence: float                # 0.0 ~ 1.0
    features: TextFeatures
    reasons: list[str] = field(default_factory=list)

    def is_ai(self) -> bool:
        return self.label == ContentLabel.AI_GENERATED


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """简单分词：按空白和标点切割，全部小写"""
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return tokens


def _split_sentences(text: str) -> list[str]:
    """按句号/感叹号/问号分句"""
    sentences = re.split(r"[。！？.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _calc_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _bigram_entropy(tokens: list[str]) -> float:
    """
    基于 bigram 分布的近似文本熵。
    AI 生成文本通常 bigram 分布更均匀，熵值偏高但缺乏突发性。
    人类写作常有局部重复（话题词），熵值相对较低。
    """
    if len(tokens) < 2:
        return 0.0
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    counts = Counter(bigrams)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy


# ──────────────────────────────────────────────
# 特征提取器
# ──────────────────────────────────────────────

class TextFeatureExtractor:
    """
    从原始文本提取 AIGC 鉴别所需的统计特征。

    AI 生成文本的核心统计特征：
    1. 句长均匀性高（方差低）：AI 倾向于生成长度相近的句子
    2. 词汇多样性中等偏高，但缺少口语化/错别字/情绪化词汇
    3. 词汇重复率低（AI 倾向回避重复，但偶有模板性重复）
    4. bigram 熵偏高（分布均匀，缺乏人类写作的突发性）
    5. 标点使用规则，极少连续感叹号/省略号等情绪性标点
    """

    def extract(self, text: str) -> TextFeatures:
        tokens = _tokenize(text)
        sentences = _split_sentences(text)

        token_count = len(tokens)
        unique_tokens = set(tokens)
        unique_token_ratio = len(unique_tokens) / max(token_count, 1)

        sent_lengths = [len(_tokenize(s)) for s in sentences]
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        sent_variance = _calc_variance([float(l) for l in sent_lengths])

        avg_word_len = (
            sum(len(t) for t in tokens) / max(token_count, 1)
        )

        # 词汇重复率：出现 2+ 次的词占总词数的比例
        freq = Counter(tokens)
        repeated = sum(c for c in freq.values() if c > 1)
        repetition_rate = repeated / max(token_count, 1)

        entropy = _bigram_entropy(tokens)

        # 标点密度：标点字符数 / 总字符数
        punct_count = sum(1 for c in text if c in "，。！？,!?;；…·")
        punctuation_density = punct_count / max(len(text), 1)

        return TextFeatures(
            token_count=token_count,
            unique_token_ratio=unique_token_ratio,
            sentence_lengths=sent_lengths,
            avg_sentence_length=avg_sent_len,
            sentence_length_variance=sent_variance,
            avg_word_length=avg_word_len,
            repetition_rate=repetition_rate,
            approx_entropy=entropy,
            punctuation_density=punctuation_density,
        )


# ──────────────────────────────────────────────
# AIGC 检测器
# ──────────────────────────────────────────────

class AIGCDetector:
    """
    基于特征阈值的 AIGC 鉴别分类器。

    零样本检测（无需训练数据），基于语言学统计特征评分。
    每个 AI 特征贡献一定权重；累积分数超过阈值判定为 AI 生成。

    权衡：
    - 零样本检测：召回率高，精确率中等，适合粗筛
    - 监督检测：精确率高，但需要标注数据集
    本实现采用零样本策略，适用于母婴评论初步过滤。
    """

    # 各特征的 AI 判定阈值
    THRESHOLDS = {
        "low_sent_variance": 4.0,     # 句长方差 < 4 → AI 特征（句长均匀）
        "high_unique_ratio": 0.75,    # 词汇多样性 > 0.75 → AI 特征（回避重复）
        "low_repetition": 0.15,       # 重复率 < 15% → AI 特征
        "high_entropy": 5.0,          # bigram 熵 > 5 → AI 特征
        "low_punct_density": 0.03,    # 标点密度 < 3% → AI 特征（缺少情绪性标点）
        "long_avg_sent": 12.0,        # 平均句长 > 12 词 → AI 特征
    }

    # 各特征的权重（满分 1.0）
    WEIGHTS = {
        "low_sent_variance": 0.25,
        "high_unique_ratio": 0.20,
        "low_repetition": 0.15,
        "high_entropy": 0.25,
        "low_punct_density": 0.10,
        "long_avg_sent": 0.05,
    }

    AI_SCORE_THRESHOLD = 0.55   # 超过此分数判定为 AI 生成
    UNCERTAIN_THRESHOLD = 0.35  # 介于此分数和 AI 阈值之间为不确定

    # 人类写作特征信号（匹配任意一条则降低 AI 分）
    HUMAN_SIGNALS = [
        r"[！!～~]{2,}",          # 连续情绪性标点（口语化）
        r"[哈嗯哦诶呀哇嘛呢吧啦]{2,}",  # 语气词连用
        r"(就是|反正|总体|不过|但是|只是|感觉|好像|其实)",  # 口语转折词
        r"(还行|还好|一般般|挺好的|不错哦|值得买|推荐)",    # 口语评价语
        r"(宝宝|娃|孩子).{0,5}(喜欢|爱|不喜|哭|笑)",       # 口语亲子描述
        r"下次(还来|再买|再囤)",   # 口语购买意向
    ]

    def __init__(self) -> None:
        self._extractor = TextFeatureExtractor()
        self._human_signal_re = [re.compile(p) for p in self.HUMAN_SIGNALS]

    def detect(self, text: str) -> DetectionResult:
        """
        对输入文本执行 AIGC 鉴别。

        Args:
            text: 待鉴别的文本（评论、UGC 内容等）

        Returns:
            DetectionResult，包含标签、置信度、特征和判断依据
        """
        features = self._extractor.extract(text)
        ai_score = 0.0
        reasons: list[str] = []

        # 规则 1：句长均匀性（方差低 = AI）
        if features.sentence_length_variance < self.THRESHOLDS["low_sent_variance"]:
            ai_score += self.WEIGHTS["low_sent_variance"]
            reasons.append(
                f"句长均匀（方差={features.sentence_length_variance:.2f}<{self.THRESHOLDS['low_sent_variance']}）"
            )

        # 规则 2：词汇多样性（高唯一词比 = AI）
        if features.unique_token_ratio > self.THRESHOLDS["high_unique_ratio"]:
            ai_score += self.WEIGHTS["high_unique_ratio"]
            reasons.append(
                f"词汇多样性高（{features.unique_token_ratio:.2f}>{self.THRESHOLDS['high_unique_ratio']}）"
            )

        # 规则 3：词汇重复率（低重复 = AI）
        if features.repetition_rate < self.THRESHOLDS["low_repetition"]:
            ai_score += self.WEIGHTS["low_repetition"]
            reasons.append(
                f"词汇重复率低（{features.repetition_rate:.2f}<{self.THRESHOLDS['low_repetition']}）"
            )

        # 规则 4：bigram 熵（高熵 = AI）
        if features.approx_entropy > self.THRESHOLDS["high_entropy"]:
            ai_score += self.WEIGHTS["high_entropy"]
            reasons.append(
                f"文本熵高（{features.approx_entropy:.2f}>{self.THRESHOLDS['high_entropy']}）"
            )

        # 规则 5：标点密度（低密度 = AI）
        if features.punctuation_density < self.THRESHOLDS["low_punct_density"]:
            ai_score += self.WEIGHTS["low_punct_density"]
            reasons.append(
                f"标点密度低（{features.punctuation_density:.3f}<{self.THRESHOLDS['low_punct_density']}）"
            )

        # 规则 6：平均句长（长句 = AI）
        if features.avg_sentence_length > self.THRESHOLDS["long_avg_sent"]:
            ai_score += self.WEIGHTS["long_avg_sent"]
            reasons.append(
                f"平均句长较长（{features.avg_sentence_length:.1f}>{self.THRESHOLDS['long_avg_sent']}词）"
            )

        # 人类信号折扣：检测到口语/情绪/语气词则降低 AI 分
        human_hits = [p.pattern for p in self._human_signal_re if p.search(text)]
        if human_hits:
            discount = min(len(human_hits) * 0.15, 0.35)
            ai_score = max(ai_score - discount, 0.0)
            reasons.append(f"检测到人类语言信号（-{discount:.2f}折扣）: {human_hits[:3]}")

        # 判定标签
        if ai_score >= self.AI_SCORE_THRESHOLD:
            label = ContentLabel.AI_GENERATED
            confidence = min(ai_score / 1.0, 0.95)
        elif ai_score >= self.UNCERTAIN_THRESHOLD:
            label = ContentLabel.UNCERTAIN
            confidence = 0.5
        else:
            label = ContentLabel.HUMAN
            confidence = 1.0 - ai_score

        return DetectionResult(
            label=label,
            confidence=round(confidence, 3),
            features=features,
            reasons=reasons,
        )

    def batch_detect(self, texts: list[str]) -> list[DetectionResult]:
        """批量检测，过滤掉 AI 生成内容，返回所有结果"""
        return [self.detect(t) for t in texts]

    def filter_human_only(self, texts: list[str]) -> list[str]:
        """过滤列表，仅保留判定为人类写作的文本"""
        results = self.batch_detect(texts)
        return [
            text for text, r in zip(texts, results)
            if r.label == ContentLabel.HUMAN
        ]


# ──────────────────────────────────────────────
# 测试：3条真实评论 + 3条AI生成评论
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 60)
    print("AIGC Content Detection — AI生成内容鉴别测试")
    print("=" * 60)

    detector = AIGCDetector()

    # 真实用户评论（有错别字/情绪/短句/口语化）
    real_reviews = [
        (
            "买了两罐了！宝宝超爱喝，冲开没啥腥味。就是价格有点贵，下次看看有没有活动再囤货。",
            "real_1",
        ),
        (
            "用了一个月，娃大便正常，没有上火。但包装有点难拆，每次都要剪刀。总体还行吧。",
            "real_2",
        ),
        (
            "客服很有耐心！！我问了好多问题，都一一回答了。发货也快，两天就到了。下次还来买～",
            "real_3",
        ),
    ]

    # AI生成评论（结构工整/无口语/句长均匀/无情绪标点）
    ai_reviews = [
        (
            "该产品品质优良，营养成分全面均衡，适合各年龄段婴幼儿食用。"
            "配方科学合理，充分满足婴幼儿生长发育的需求。"
            "包装设计精美大方，使用方便快捷，性价比较为突出。"
            "综合评估后认为该产品值得信赖和推荐给有需要的家庭使用。",
            "ai_1",
        ),
        (
            "本产品采用优质原料精心配制，严格执行国家婴幼儿食品安全标准。"
            "经过专业检测机构认证，确保产品质量符合各项要求。"
            "服用后婴幼儿消化吸收效果良好，体重身高发育均达到正常水平。"
            "建议广大消费者根据自身需求合理选购本类产品。",
            "ai_2",
        ),
        (
            "产品各项指标均达到行业领先水平，配方合理科学，营养物质丰富全面。"
            "适用于不同生长阶段的婴幼儿群体，满足多样化的营养补充需求。"
            "用户反馈总体良好，使用体验符合预期效果，综合性价比较高。"
            "该产品值得在婴幼儿营养品市场上获得广泛推广与应用。",
            "ai_3",
        ),
    ]

    print("\n【真实评论检测】")
    real_ai_count = 0
    for text, name in real_reviews:
        result = detector.detect(text)
        status = "✓ 正确(HUMAN)" if result.label == ContentLabel.HUMAN else f"✗ 误判({result.label.value})"
        print(f"  {name}: {status} | 置信度={result.confidence:.2f} | AI分={sum(detector.WEIGHTS[k] for k in ['low_sent_variance','high_unique_ratio','low_repetition','high_entropy','low_punct_density','long_avg_sent'] if k in [r.split('（')[0].strip() for r in []]):.2f}")
        print(f"    标签={result.label.value} | 句长方差={result.features.sentence_length_variance:.2f} | 熵={result.features.approx_entropy:.2f}")
        if result.label == ContentLabel.AI_GENERATED:
            real_ai_count += 1

    print("\n【AI生成评论检测】")
    ai_human_count = 0
    for text, name in ai_reviews:
        result = detector.detect(text)
        status = "✓ 正确(AI)" if result.label == ContentLabel.AI_GENERATED else f"△ 未检出({result.label.value})"
        print(f"  {name}: {status} | 置信度={result.confidence:.2f}")
        print(f"    触发规则: {result.reasons}")
        if result.label == ContentLabel.HUMAN:
            ai_human_count += 1

    # 批量过滤演示
    print("\n【批量过滤演示】")
    all_texts = [t for t, _ in real_reviews] + [t for t, _ in ai_reviews]
    human_only = detector.filter_human_only(all_texts)
    print(f"  总输入: {len(all_texts)} 条 | 过滤后保留: {len(human_only)} 条")

    print("\n" + "=" * 60)
    print("[✓] AIGC Content Detection 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
