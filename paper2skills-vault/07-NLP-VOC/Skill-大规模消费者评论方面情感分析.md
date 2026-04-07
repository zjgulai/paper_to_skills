# Skill Card: 大规模消费者评论方面情感分析

---

## ① 算法原理

### 核心思想
混合使用大语言模型（LLM）和传统机器学习构建可扩展的方面情感分析系统：先用 LLM 从少量样本中识别评论的关键方面（aspect），再用传统分类器对海量评论进行情感分类。兼顾 LLM 的语义理解能力和传统方法的计算效率。

### 数学直觉
**两步Pipeline**：
1. **方面识别**：用 LLM（如 ChatGPT）分析样本评论，提取关键方面（如"服务态度"、"食材新鲜度"、"环境氛围"）
2. **情感分类**：用传统ML模型（如 BERT + 分类器）在标注数据上训练，对海量评论进行方面级情感预测

关键公式：回归分析验证方面对评分的解释力
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### 关键假设
- LLM 识别的方面能覆盖主要业务关注点
- 方面与整体评分存在可量化的统计关系
- 标注数据能覆盖主要方面类别

---

## ② 吸奶器出海应用案例

### 场景1：产品评论智能分析
- **业务问题**：母婴产品评论量巨大（每天上千条），人工标注无法覆盖，无法快速发现产品问题
- **数据要求**：
  - 原始评论文本（中文/英文）
  - 整体评分（1-5星）
  - 少量人工标注样本（每方面30-50条）
- **预期产出**：
  - 自动识别评论中的关键方面（材质安全、舒适度、性价比等）
  - 输出每条评论的方面级情感（正面/负面/中性）
  - 按时间维度统计方面情感趋势
- **业务价值**：
  - 快速定位产品问题（如"吸管易碎"、"尺寸偏小"）
  - 按方面维度聚类差评，优先解决高频问题
  - 预计提升客服响应效率 60%

### 场景2：竞品监控与差异化定位
- **业务问题**：想了解自家产品与竞品在各维度的优劣，但竞品评论分散且量大
- **数据要求**：
  - 竞品评论数据（Amazon、Shopee 等平台）
  - 预定义的竞争维度
- **预期产出**：
  - 竞品各维度的情感得分对比
  - 差异化机会点（如"竞品在物流时效上差评集中"）
- **业务价值**：
  - 指导产品改进优先级
  - 营销文案突出差异化卖点

---

## ③ 代码模板

```python
"""
大规模消费者评论方面情感分析
Hybrid ABSA: LLM for aspect identification + ML for sentiment classification
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
    platform: str      # 来源平台
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
        self._ Aspects = None

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
```

---

## ④ 技能关联

### 前置技能
- **基础技能**：Python 文本处理（正则表达式、字符串操作）
- **数据技能**：Pandas 数据清洗与统计分析

### 延伸技能
- **增长模型**：与 Churn Prediction 组合，分析差评用户的流失风险
- **推荐系统**：将方面情感作为特征加入推荐模型
- **A/B实验**：验证产品改进后的情感指标提升

### 可组合技能
- **VOC + Churn Prediction**：识别高负面情感用户，预测流失概率
- **VOC + Demand Forecasting**：结合评论情感变化预测产品需求波动

---

## ⑤ 商业价值评估

### ROI 预估
- **实施成本**：LLM API 调用 + 模型训练，约 $500-1000/月
- **收益**：
  - 客服效率提升 60%（减少人工分析时间）
  - 产品问题发现周期从天级缩短到小时级
  - 预计每月节省 20-30 人力小时
- **ROI**：300-500%（投入 1 元产生 3-5 元价值）

### 实施难度：⭐⭐☆☆☆ (2/5)
- 技术栈成熟，有开源实现
- 主要挑战：中文评论的分词和方面识别

### 优先级评分：⭐⭐⭐⭐☆ (4/5)
- 业务需求迫切（评论量大，难以人工处理）
- 见效快，1-2 周可上线 MVP
- 可复用到多个产品线

### 评估依据
- 母婴产品评论增长快，人工分析无法覆盖
- 方面级情感能精准定位问题，比整体评分更有价值
- 混合方法兼顾精度和效率，适合业务落地