"""
大规模消费者评论方面情感分析
Hybrid ABSA: LLM for aspect identification + ML for sentiment classification

论文: Beyond the Star Rating: A Scalable Framework for Aspect-Based Sentiment Analysis Using LLMs and Text Classification
arXiv: 2602.21082
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

# ============================================================
# 1. 数据类型定义
# ============================================================

@dataclass
class Review:
    """评论数据结构"""
    text: str           # 评论文本
    rating: int         # 整体评分 1-5
    platform: str       # 来源平台
    timestamp: str     # 评论时间


@dataclass
class AspectSentiment:
    """方面情感结果"""
    aspect: str         # 方面名称
    sentiment: str      # 情感: positive/negative/neutral
    confidence: float   # 置信度


# ============================================================
# 2. LLM 方面识别模块 (示例: 使用 GPT API)
# ============================================================

class AspectIdentifier:
    """使用 LLM 从少量样本中识别关键方面"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self._aspects = None

    def identify_aspects(self, sample_reviews: List[Review], n_aspects: int = 10) -> List[str]:
        """
        从样本评论中识别关键方面

        Args:
            sample_reviews: 样本文本（建议 50-100 条）
            n_aspects: 需要识别的主要方面数量

        Returns:
            方面列表，如 ["产品材质", "使用体验", "包装设计"]
        """
        # 步骤1: 随机抽样
        samples = sample_reviews[:min(20, len(sample_reviews))]

        # 步骤2: 构建 prompt
        prompt = self._build_aspect_prompt(samples, n_aspects)

        # 步骤3: 调用 LLM (示例代码)
        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}]
        # )

        # 步骤4: 解析结果
        # aspects = self._parse_aspects(response)

        # 模拟输出
        aspects = [
            "产品材质安全", "使用舒适度", "包装设计",
            "性价比", "物流时效", "产品颜值",
            "做工质量", "功能实用性", "售后服务", "尺寸合适度"
        ]
        self._aspects = aspects
        return aspects

    def _build_aspect_prompt(self, reviews: List[Review], n: int) -> str:
        """构建方面识别 prompt"""
        review_texts = "\n".join([r.text[:200] for r in reviews])
        return f"""从以下母婴产品评论中识别最关键的{n}个方面（方面应该是用户常评论的维度）:

评论示例:
{review_texts}

输出格式:
["方面1", "方面2", ...]"""

    def _parse_aspects(self, response) -> List[str]:
        """解析 LLM 返回的方面"""
        # 实际实现需要解析 JSON
        pass


# ============================================================
# 3. 传统 ML 情感分类器
# ============================================================

class SentimentClassifier:
    """基于 BERT 的方面情感分类器"""

    def __init__(self, model_path: str = "bert-base-chinese"):
        self.model_path = model_path
        self.model = None
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.is_trained = False

    def train(self, train_data: List[Tuple[str, str, int]]) -> None:
        """
        训练情感分类模型

        Args:
            train_data: [(review_text, aspect, sentiment_label), ...]
                        sentiment_label: 0=negative, 1=neutral, 2=positive
        """
        # 示例: 使用 transformers 训练
        # from transformers import BertForSequenceClassification, Trainer
        # ... 训练代码 ...

        print(f"训练数据量: {len(train_data)} 条")
        self.is_trained = True

    def predict(self, reviews: List[str], aspects: List[str]) -> List[List[AspectSentiment]]:
        """
        批量预测方面情感

        Args:
            reviews: 评论文本列表
            aspects: 方面列表

        Returns:
            每条评论的方面情感结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用 train()")

        results = []
        for review in reviews:
            review_results = []
            for aspect in aspects:
                # 模拟预测结果
                sentiment = np.random.choice(
                    ["positive", "negative", "neutral"],
                    p=[0.6, 0.2, 0.2]
                )
                confidence = np.random.uniform(0.7, 0.95)
                review_results.append(AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment,
                    confidence=confidence
                ))
            results.append(review_results)

        return results

    def predict_batch(self, df: pd.DataFrame, aspects: List[str]) -> pd.DataFrame:
        """
        批量处理 DataFrame

        Args:
            df: 包含 'review_text' 列的 DataFrame
            aspects: 方面列表

        Returns:
            添加了方面情感列的 DataFrame
        """
        predictions = self.predict(df['review_text'].tolist(), aspects)

        for aspect in aspects:
            df[f'{aspect}_sentiment'] = [
                next((p.sentiment for p in preds if p.aspect == aspect), None)
                for preds in predictions
            ]
            df[f'{aspect}_confidence'] = [
                next((p.confidence for p in preds if p.aspect == aspect), 0)
                for preds in predictions
            ]

        return df


# ============================================================
# 4. 主 Pipeline
# ============================================================

class ABSAPipeline:
    """方面情感分析完整 Pipeline"""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.aspect_identifier = AspectIdentifier() if use_llm else None
        self.sentiment_classifier = SentimentClassifier()
        self.aspects = None

    def fit(self, reviews: List[Review], labeled_data: List[Tuple[str, str, int]] = None):
        """
        训练 Pipeline

        Args:
            reviews: 全部评论数据（用于 LLM 识别方面）
            labeled_data: 标注数据 [(review, aspect, label), ...]
        """
        print("Step 1: 识别关键方面...")
        if self.use_llm and self.aspect_identifier:
            self.aspects = self.aspect_identifier.identify_aspects(reviews)
        else:
            # 默认方面
            self.aspects = ["产品材质", "使用体验", "包装设计", "性价比"]

        print(f"识别到 {len(self.aspects)} 个方面: {self.aspects}")

        print("Step 2: 训练情感分类器...")
        if labeled_data:
            self.sentiment_classifier.train(labeled_data)
        else:
            # 使用小样本提示学习
            print("使用小样本提示学习 (Few-shot)")

        print("训练完成!")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理新数据

        Args:
            df: 包含 'review_text' 列的 DataFrame

        Returns:
            添加了方面情感列的 DataFrame
        """
        return self.sentiment_classifier.predict_batch(df, self.aspects)


# ============================================================
# 5. 示例数据和测试
# ============================================================

def generate_sample_data(n: int = 100) -> pd.DataFrame:
    """生成模拟评论数据"""
    np.random.seed(42)

    aspects = ["产品材质", "使用体验", "包装设计", "性价比"]
    sentiments = ["positive", "negative", "neutral"]

    data = []
    for i in range(n):
        review = {
            "review_id": f"R{i:04d}",
            "review_text": f"这个产品真{np.random.choice(['好', '不错', '一般', '差', '非常差'])}，" +
                          f"材质{np.random.choice(['安全', '放心', '可以', '一般', '有问题'])}，" +
                          f"用起来{np.random.choice(['舒服', '顺手', '还行', '不方便'])}",
            "rating": np.random.randint(1, 6),
            "platform": np.random.choice(["Amazon", "Shopee", "TikTok"]),
            "timestamp": f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"
        }
        data.append(review)

    return pd.DataFrame(data)


def run_pipeline_demo():
    """运行 Pipeline 演示"""
    print("=" * 60)
    print("大规模消费者评论方面情感分析 - Demo")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    df = generate_sample_data(100)
    print(f"生成 {len(df)} 条评论")
    print(df.head(3))

    # 2. 初始化 Pipeline
    print("\n[2] 初始化 Pipeline...")
    pipeline = ABSAPipeline(use_llm=True)

    # 3. 训练 (使用小样本)
    print("\n[3] 训练模型...")
    # 模拟训练
    pipeline.aspects = ["产品材质", "使用体验", "包装设计", "性价比"]
    pipeline.sentiment_classifier.is_trained = True

    # 4. 预测
    print("\n[4] 预测方面情感...")
    result_df = pipeline.transform(df)

    # 5. 统计结果
    print("\n[5] 统计结果...")
    for aspect in pipeline.aspects:
        sentiment_col = f"{aspect}_sentiment"
        if sentiment_col in result_df.columns:
            counts = result_df[sentiment_col].value_counts()
            print(f"{aspect}: {counts.to_dict()}")

    print("\n[6] 方面情感趋势分析...")
    result_df['month'] = pd.to_datetime(result_df['timestamp']).dt.month
    monthly_sentiment = result_df.groupby('month')['产品材质_sentiment'].apply(
        lambda x: (x == 'positive').mean()
    )
    print("月度正面率趋势:")
    print(monthly_sentiment)

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)

    return result_df


if __name__ == "__main__":
    result = run_pipeline_demo()
    print("\n输出数据预览:")
    print(result.head())