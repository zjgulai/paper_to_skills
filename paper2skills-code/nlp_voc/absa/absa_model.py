"""
Aspect-Based Sentiment Analysis (ABSA)
方面级情感分析 - 母婴产品评论分析

Reference: Patil et al. (2026) - Beyond the Star Rating
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')


# 模拟 LLM 提取的方面（实际项目中使用 GPT-4 API）
ASPECTS = [
    '产品质量', '物流速度', '客服服务', '性价比', 
    '包装设计', '使用体验', '安全性', '适龄性'
]


def generate_maternity_reviews(n_samples=1000):
    """生成母婴产品评论数据"""
    np.random.seed(42)
    
    # 模拟产品类型
    products = np.random.choice(
        ['奶粉', '尿布', '辅食', '婴儿服装', '玩具', '奶瓶', '婴儿车'],
        n_samples
    )
    
    # 方面相关模板（正面/负面/中性）
    templates = {
        '产品质量': {
            'positive': ['质量非常好', '品质过硬', '做工精细', '用料讲究', '正品保障'],
            'negative': ['质量一般', '做工粗糙', '有异味', '质量不稳定', '怀疑是假货'],
            'neutral': ['质量还可以', '一般般', '符合预期', '正常水平']
        },
        '物流速度': {
            'positive': ['物流很快', '第二天就到了', '配送及时', '快递小哥很给力'],
            'negative': ['物流太慢', '等了半个月', '配送延误', '包装破损'],
            'neutral': ['物流正常', '一般速度', '还可以']
        },
        '客服服务': {
            'positive': ['客服态度好', '回复及时', '解决问题快', '很有耐心'],
            'negative': ['客服态度差', '不理人', '推诿责任', '售后麻烦'],
            'neutral': ['客服一般', '正常回复']
        },
        '性价比': {
            'positive': ['性价比高', '物美价廉', '很划算', '值得买'],
            'negative': ['太贵了', '不值这个价', '性价比低', '有 cheaper 的替代品'],
            'neutral': ['价格正常', '还可以接受']
        },
        '包装设计': {
            'positive': ['包装精美', '密封性好', '方便携带', '设计人性化'],
            'negative': ['包装简陋', '密封不好', '漏粉', '不方便'],
            'neutral': ['包装正常', '普通包装']
        },
        '使用体验': {
            'positive': ['宝宝很喜欢', '使用方便', '效果很明显', '一直在用'],
            'negative': ['宝宝不吃', '用着不方便', '没效果', '过敏了'],
            'neutral': ['用着还行', '正常吧']
        },
        '安全性': {
            'positive': ['安全可靠', '无添加', '有机认证', '放心使用'],
            'negative': ['担心安全问题', '有化学成分', '不够天然'],
            'neutral': ['应该安全吧']
        },
        '适龄性': {
            'positive': ['正好适合', '尺码标准', '阶段合适'],
            'negative': ['尺码偏小', '不适合这个月龄', '阶段不匹配'],
            'neutral': ['基本合适']
        }
    }
    
    reviews = []
    labels = []
    
    for i in range(n_samples):
        product = products[i]
        # 随机选择 2-4 个方面进行评价
        n_aspects = np.random.randint(2, 5)
        selected_aspects = np.random.choice(ASPECTS, n_aspects, replace=False)
        
        review_parts = []
        aspect_labels = {}
        
        for aspect in selected_aspects:
            sentiment = np.random.choice(['positive', 'negative', 'neutral'], 
                                         p=[0.5, 0.3, 0.2])
            phrase = np.random.choice(templates[aspect][sentiment])
            review_parts.append(phrase)
            aspect_labels[aspect] = sentiment
        
        # 添加产品相关上下文
        review = f"购买了{product}，" + "。".join(review_parts) + "。"
        reviews.append(review)
        labels.append(aspect_labels)
    
    return pd.DataFrame({
        'review': reviews,
        'product': products,
        'aspect_labels': labels
    })


class AspectExtractor:
    """方面提取器（简化版，实际使用 LLM API）"""
    
    def __init__(self):
        self.aspect_keywords = {
            '产品质量': ['质量', '品质', '做工', '用料', '正品'],
            '物流速度': ['物流', '快递', '配送', '发货', '速度'],
            '客服服务': ['客服', '售后', '服务', '态度', '回复'],
            '性价比': ['价格', '性价比', '便宜', '贵', '划算', '值得'],
            '包装设计': ['包装', '密封', '设计', '外观'],
            '使用体验': ['使用', '体验', '方便', '效果', '喜欢'],
            '安全性': ['安全', '放心', '无添加', '有机', '天然'],
            '适龄性': ['适合', '月龄', '阶段', '尺码', '年龄']
        }
    
    def extract_aspects(self, text: str) -> List[str]:
        """提取评论中涉及的方面"""
        mentioned_aspects = []
        for aspect, keywords in self.aspect_keywords.items():
            if any(kw in text for kw in keywords):
                mentioned_aspects.append(aspect)
        return mentioned_aspects


class TwoStageABSA:
    """两阶段方面级情感分析"""
    
    def __init__(self, aspects: List[str]):
        self.aspects = aspects
        # Stage 1: 相关性分类器
        self.relevance_classifiers = {
            aspect: LogisticRegression(max_iter=1000, class_weight='balanced')
            for aspect in aspects
        }
        # Stage 2: 情感分类器
        self.sentiment_classifiers = {
            aspect: LogisticRegression(max_iter=1000, class_weight='balanced')
            for aspect in aspects
        }
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """准备训练数据"""
        X = self.vectorizer.fit_transform(df['review']).toarray()
        
        # 为每个方面准备标签
        y = {aspect: {'relevance': [], 'sentiment': []} for aspect in self.aspects}
        
        for _, row in df.iterrows():
            labels = row['aspect_labels']
            for aspect in self.aspects:
                if aspect in labels:
                    y[aspect]['relevance'].append(1)
                    y[aspect]['sentiment'].append(labels[aspect])
                else:
                    y[aspect]['relevance'].append(0)
                    y[aspect]['sentiment'].append('neutral')  # 占位
        
        return X, y
    
    def fit(self, df: pd.DataFrame):
        """训练两阶段分类器"""
        print("训练两阶段 ABSA 模型...")
        X, y = self._prepare_data(df)
        
        for aspect in self.aspects:
            # Stage 1: 训练相关性分类器
            rel_labels = np.array(y[aspect]['relevance'])
            self.relevance_classifiers[aspect].fit(X, rel_labels)
            
            # Stage 2: 训练情感分类器（仅使用相关样本）
            relevant_mask = rel_labels == 1
            if relevant_mask.sum() > 10:
                X_rel = X[relevant_mask]
                sent_labels = np.array(y[aspect]['sentiment'])[relevant_mask]
                self.sentiment_classifiers[aspect].fit(X_rel, sent_labels)
        
        print("训练完成!")
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """预测新评论的方面情感"""
        X = self.vectorizer.transform(texts).toarray()
        results = []
        
        for i in range(len(texts)):
            result = {}
            for aspect in self.aspects:
                # Stage 1: 判断是否相关
                is_relevant = self.relevance_classifiers[aspect].predict(X[i:i+1])[0]
                
                if is_relevant == 1:
                    # Stage 2: 判断情感极性
                    sentiment = self.sentiment_classifiers[aspect].predict(X[i:i+1])[0]
                    proba = self.sentiment_classifiers[aspect].predict_proba(X[i:i+1])[0]
                    result[aspect] = {
                        'sentiment': sentiment,
                        'confidence': float(max(proba))
                    }
            results.append(result)
        
        return results


def analyze_aspect_sentiment_distribution(results: List[Dict]) -> pd.DataFrame:
    """分析方面情感分布"""
    aspect_stats = {aspect: {'positive': 0, 'negative': 0, 'neutral': 0} 
                    for aspect in ASPECTS}
    
    for result in results:
        for aspect, info in result.items():
            sentiment = info['sentiment']
            if sentiment in aspect_stats[aspect]:
                aspect_stats[aspect][sentiment] += 1
    
    df_stats = pd.DataFrame(aspect_stats).T
    df_stats['total'] = df_stats.sum(axis=1)
    df_stats['positive_pct'] = (df_stats['positive'] / df_stats['total'] * 100).round(1)
    df_stats['negative_pct'] = (df_stats['negative'] / df_stats['total'] * 100).round(1)
    
    return df_stats


def generate_improvement_suggestions(aspect_stats: pd.DataFrame) -> List[str]:
    """生成改进建议"""
    suggestions = []
    
    for aspect, row in aspect_stats.iterrows():
        if row['negative_pct'] > 30:
            suggestions.append(f"⚠️ {aspect}: 负面评价占比 {row['negative_pct']:.1f}%，建议重点关注")
        elif row['positive_pct'] > 70:
            suggestions.append(f"✅ {aspect}: 好评率 {row['positive_pct']:.1f}%，保持优势")
    
    return suggestions


def main():
    """主函数：母婴产品评论 ABSA 分析"""
    print("=" * 60)
    print("Aspect-Based Sentiment Analysis for Maternity E-commerce")
    print("母婴产品评论方面级情感分析")
    print("=" * 60)
    
    # 1. 生成模拟数据
    print("\n[1] 生成模拟评论数据...")
    df = generate_maternity_reviews(n_samples=2000)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_df)} 条评论")
    print(f"测试集: {len(test_df)} 条评论")
    
    # 2. 训练模型
    print("\n[2] 训练两阶段 ABSA 模型...")
    absa = TwoStageABSA(ASPECTS)
    absa.fit(train_df)
    
    # 3. 预测
    print("\n[3] 分析测试集评论...")
    test_texts = test_df['review'].tolist()
    predictions = absa.predict(test_texts)
    
    # 4. 分析结果
    print("\n[4] 方面情感分布分析:")
    stats = analyze_aspect_sentiment_distribution(predictions)
    print(stats[['total', 'positive_pct', 'negative_pct']].to_string())
    
    # 5. 生成建议
    print("\n[5] 改进建议:")
    suggestions = generate_improvement_suggestions(stats)
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    # 6. 展示示例
    print("\n[6] 评论分析示例:")
    for i in range(min(3, len(test_df))):
        print(f"\n  评论 {i+1}: {test_df.iloc[i]['review'][:60]}...")
        pred = predictions[i]
        for aspect, info in pred.items():
            emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(info['sentiment'], '')
            print(f"    {emoji} {aspect}: {info['sentiment']} (置信度: {info['confidence']:.2f})")
    
    print("\n" + "=" * 60)
    print("ABSA 分析完成!")
    print("=" * 60)
    
    return absa, stats


if __name__ == "__main__":
    absa_model, aspect_stats = main()
