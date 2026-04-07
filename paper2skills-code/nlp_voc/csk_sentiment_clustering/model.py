"""
CSK: Cuckoo Search + K-means for Customer Sentiment Clustering
CSK客户情感聚类算法

论文: Customer Sentiment Analysis with Cuckoo Search and K-means Clustering
arXiv: 2311.11250 (2023)

核心创新:
- 结合布谷鸟搜索(Cuckoo Search)全局优化和K-means局部聚类
- 解决K-means初始质心敏感问题
- 实现情感驱动的用户分群
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import math
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CustomerReview:
    """客户评论数据"""
    review_id: str
    user_id: str
    text: str
    rating: float  # 1-5星评分
    timestamp: pd.Timestamp
    product_category: str  # 'milk', 'diaper', 'food', 'toy', 'clothes'
    

class TextFeatureExtractor:
    """
    文本特征提取器
    将评论文本转换为数值特征向量
    """
    
    def __init__(self):
        # 情感词典 (简化版，实际应用应使用更完整的词典)
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best',
            'happy', 'satisfied', 'recommend', 'quality', 'comfortable', 'safe',
            'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best',
            'happy', 'satisfied', 'recommend', 'quality', 'comfortable', 'safe',
            'soft', 'gentle', 'healthy', 'convenient', 'fast', 'reliable'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'poor',
            'unhappy', 'unsatisfied', 'avoid', 'cheap', 'uncomfortable', 'dangerous',
            'hard', 'rough', 'unhealthy', 'inconvenient', 'slow', 'unreliable',
            'problem', 'issue', 'defect', 'broken', 'damaged', 'wrong'
        }
        
        # 母婴特定方面词
        self.aspect_keywords = {
            'quality': ['quality', 'material', 'texture', 'durable'],
            'safety': ['safe', 'safety', 'secure', 'protect', 'harmless'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
            'service': ['service', 'delivery', 'shipping', 'support', 'return'],
            'usability': ['easy', 'convenient', 'simple', 'difficult', 'complicated']
        }
        
    def extract_features(self, review: CustomerReview) -> np.ndarray:
        """
        提取评论特征向量
        
        特征维度:
        - 情感词计数 (2维: pos, neg)
        - 评分 (1维)
        - 文本长度 (1维)
        - 方面情感 (5维: quality, safety, price, service, usability)
        - 情感强度 (1维)
        
        总维度: 10维
        """
        text = review.text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # 1. 情感词计数
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        # 2. 评分 (归一化到0-1)
        rating_norm = (review.rating - 1) / 4.0
        
        # 3. 文本长度 (归一化)
        text_length = min(len(words) / 100.0, 1.0)
        
        # 4. 方面情感
        aspect_scores = []
        for aspect, keywords in self.aspect_keywords.items():
            aspect_pos = sum(1 for w in words if w in keywords and w in self.positive_words)
            aspect_neg = sum(1 for w in words if w in keywords and w in self.negative_words)
            aspect_score = (aspect_pos - aspect_neg) / max(len(keywords), 1)
            aspect_scores.append(aspect_score)
        
        # 5. 情感强度 (情感词占比)
        sentiment_intensity = (pos_count + neg_count) / max(len(words), 1)
        
        features = np.array([
            pos_count / 10.0,  # 归一化
            neg_count / 10.0,
            rating_norm,
            text_length,
            *aspect_scores,
            sentiment_intensity
        ])
        
        return features
    
    def extract_batch(self, reviews: List[CustomerReview]) -> np.ndarray:
        """批量提取特征"""
        return np.array([self.extract_features(r) for r in reviews])


class CuckooSearch:
    """
    布谷鸟搜索算法
    用于优化K-means初始质心选择
    """
    
    def __init__(self, n_nests: int = 25, n_iterations: int = 100, 
                 pa: float = 0.25, beta: float = 1.5):
        """
        Args:
            n_nests: 鸟巢数量 (种群大小)
            n_iterations: 迭代次数
            pa: 发现概率 (被宿主发现后丢弃鸟蛋的概率)
            beta: Levy飞行参数
        """
        self.n_nests = n_nests
        self.n_iterations = n_iterations
        self.pa = pa
        self.beta = beta
        
    def levy_flight(self, n: int, dim: int) -> np.ndarray:
        """Levy飞行随机步长生成"""
        sigma = (math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2) /
                (math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        
        u = np.random.normal(0, sigma, (n, dim))
        v = np.random.normal(0, 1, (n, dim))
        step = u / (np.abs(v) ** (1 / self.beta))
        
        return step
    
    def optimize(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        使用布谷鸟搜索优化聚类质心
        
        Returns:
            优化的初始质心 [n_clusters, n_features]
        """
        n_samples, n_features = data.shape
        
        # 初始化鸟巢 (随机选择数据点作为质心)
        nests = []
        for _ in range(self.n_nests):
            indices = np.random.choice(n_samples, n_clusters, replace=False)
            nests.append(data[indices].copy())
        nests = np.array(nests)  # [n_nests, n_clusters, n_features]
        
        # 计算初始适应度 (负的inertia，越小越好)
        fitness = np.array([self._calculate_inertia(data, nest) for nest in nests])
        
        # 最优解
        best_nest_idx = np.argmin(fitness)
        best_nest = nests[best_nest_idx].copy()
        best_fitness = fitness[best_nest_idx]
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            # 1. Levy飞行生成新解
            for i in range(self.n_nests):
                if i == best_nest_idx:
                    continue  # 保留最优解
                
                # Levy飞行
                step = self.levy_flight(n_clusters, n_features) * 0.01
                new_nest = nests[i] + step
                
                # 边界处理
                new_nest = np.clip(new_nest, data.min(axis=0), data.max(axis=0))
                
                # 评估新解
                new_fitness = self._calculate_inertia(data, new_nest)
                
                # 贪婪选择
                if new_fitness < fitness[i]:
                    nests[i] = new_nest
                    fitness[i] = new_fitness
                    
                    # 更新全局最优
                    if new_fitness < best_fitness:
                        best_nest = new_nest.copy()
                        best_fitness = new_fitness
            
            # 2. 随机丢弃被发现概率pa的巢，并随机重建
            for i in range(self.n_nests):
                if np.random.random() < self.pa and i != best_nest_idx:
                    indices = np.random.choice(n_samples, n_clusters, replace=False)
                    nests[i] = data[indices].copy()
                    fitness[i] = self._calculate_inertia(data, nests[i])
        
        return best_nest
    
    def _calculate_inertia(self, data: np.ndarray, centroids: np.ndarray) -> float:
        """计算聚类inertia (所有点到最近质心的距离平方和)"""
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        min_distances = distances.min(axis=1)
        return np.sum(min_distances ** 2)


class CSKClustering:
    """
    CSK聚类算法
    Cuckoo Search + K-means
    """
    
    def __init__(self, n_clusters: int = 5, max_iter: int = 300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.cuckoo_search = CuckooSearch(n_nests=25, n_iterations=50)
        
    def fit(self, data: np.ndarray) -> 'CSKClustering':
        """
        训练CSK聚类模型
        
        步骤:
        1. 使用布谷鸟搜索找到最优初始质心
        2. 使用K-means进行精细聚类
        """
        print(f"CSK聚类: 数据量 {len(data)}, 聚类数 {self.n_clusters}")
        
        # 步骤1: 布谷鸟搜索优化初始质心
        print("步骤1: 布谷鸟搜索优化初始质心...")
        self.centroids = self.cuckoo_search.optimize(data, self.n_clusters)
        print(f"初始inertia: {self.cuckoo_search._calculate_inertia(data, self.centroids):.2f}")
        
        # 步骤2: K-means迭代优化
        print("步骤2: K-means精细聚类...")
        for iteration in range(self.max_iter):
            # 分配样本到最近质心
            distances = np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
            labels = distances.argmin(axis=1)
            
            # 更新质心
            new_centroids = np.array([data[labels == k].mean(axis=0) 
                                     for k in range(self.n_clusters)])
            
            # 处理空簇
            for k in range(self.n_clusters):
                if np.isnan(new_centroids[k]).any():
                    new_centroids[k] = self.centroids[k]
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids, rtol=1e-4):
                print(f"K-means在第{iteration}轮收敛")
                break
            
            self.centroids = new_centroids
        
        # 最终标签
        distances = np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        self.labels = distances.argmin(axis=1)
        
        inertia = self.cuckoo_search._calculate_inertia(data, self.centroids)
        print(f"最终inertia: {inertia:.2f}")
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """预测样本所属簇"""
        distances = np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return distances.argmin(axis=1)
    
    def get_cluster_stats(self, data: np.ndarray) -> Dict:
        """获取聚类统计信息"""
        stats = {}
        for k in range(self.n_clusters):
            cluster_data = data[self.labels == k]
            stats[f'cluster_{k}'] = {
                'size': len(cluster_data),
                'centroid': self.centroids[k],
                'mean_distance': np.mean(np.sqrt(((cluster_data - self.centroids[k]) ** 2).sum(axis=1)))
            }
        return stats


class CustomerSentimentAnalyzer:
    """
    客户情感分析器
    整合特征提取、CSK聚类、标签解读
    """
    
    # 预定义的 sentiment cluster 标签
    CLUSTER_LABELS = {
        0: '高满意-推荐型',
        1: '价格敏感型',
        2: '质量关注型',
        3: '服务抱怨型',
        4: '中性观望型'
    }
    
    CLUSTER_STRATEGIES = {
        '高满意-推荐型': {
            'description': '高度满意，愿意推荐',
            'action': '邀请成为品牌大使，奖励推荐',
            'priority': '低'
        },
        '价格敏感型': {
            'description': '关注价格和性价比',
            'action': '推送优惠券、促销活动',
            'priority': '中'
        },
        '质量关注型': {
            'description': '重点关注产品质量和安全',
            'action': '展示质检报告、安全认证',
            'priority': '中'
        },
        '服务抱怨型': {
            'description': '对服务不满意，有抱怨',
            'action': '立即客服介入，解决问题',
            'priority': '高'
        },
        '中性观望型': {
            'description': '态度中性，正在观望',
            'action': '教育内容，增加信任',
            'priority': '中'
        }
    }
    
    def __init__(self, n_clusters: int = 5):
        self.feature_extractor = TextFeatureExtractor()
        self.clustering = CSKClustering(n_clusters=n_clusters)
        self.is_fitted = False
        
    def fit(self, reviews: List[CustomerReview]):
        """训练情感分析模型"""
        print(f"训练客户情感分析模型: {len(reviews)} 条评论")
        
        # 特征提取
        features = self.feature_extractor.extract_batch(reviews)
        
        # CSK聚类
        self.clustering.fit(features)
        self.is_fitted = True
        
        # 为每个簇分配标签 (基于簇的特征)
        self._assign_cluster_labels(features, reviews)
        
        print("模型训练完成")
        return self
    
    def _assign_cluster_labels(self, features: np.ndarray, reviews: List[CustomerReview]):
        """基于簇特征自动分配标签"""
        self.cluster_label_map = {}
        
        for k in range(self.clustering.n_clusters):
            cluster_mask = self.clustering.labels == k
            cluster_features = features[cluster_mask]
            cluster_reviews = [r for i, r in enumerate(reviews) if cluster_mask[i]]
            
            if len(cluster_features) == 0:
                self.cluster_label_map[k] = '中性观望型'
                continue
            
            # 计算簇的平均特征
            avg_pos = cluster_features[:, 0].mean()
            avg_neg = cluster_features[:, 1].mean()
            avg_rating = cluster_features[:, 2].mean()
            
            # 根据特征分配标签
            if avg_pos > avg_neg and avg_rating > 0.7:
                label = '高满意-推荐型'
            elif avg_neg > avg_pos and avg_neg > 0.3:
                label = '服务抱怨型'
            elif cluster_features[:, 4].mean() < -0.2:  # price aspect negative
                label = '价格敏感型'
            elif cluster_features[:, 3].mean() < -0.2:  # quality aspect negative
                label = '质量关注型'
            else:
                label = '中性观望型'
            
            self.cluster_label_map[k] = label
    
    def analyze(self, review: CustomerReview) -> Dict:
        """
        分析单条评论
        
        Returns:
            {
                'review_id': str,
                'sentiment_cluster': int,
                'cluster_label': str,
                'sentiment_score': float,  # -1 to 1
                'aspect_scores': Dict[str, float],
                'recommended_action': str,
                'priority': str
            }
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()")
        
        # 特征提取
        features = self.feature_extractor.extract_features(review)
        
        # 预测簇
        cluster = self.clustering.predict(features.reshape(1, -1))[0]
        label = self.cluster_label_map.get(cluster, '未知')
        
        # 计算情感分数
        pos_score = features[0]
        neg_score = features[1]
        sentiment_score = (pos_score - neg_score) / (pos_score + neg_score + 0.1)
        
        # 方面分数
        aspect_names = ['quality', 'safety', 'price', 'service', 'usability']
        aspect_scores = {name: features[4 + i] for i, name in enumerate(aspect_names)}
        
        # 策略
        strategy = self.CLUSTER_STRATEGIES.get(label, {})
        
        return {
            'review_id': review.review_id,
            'sentiment_cluster': cluster,
            'cluster_label': label,
            'sentiment_score': float(sentiment_score),
            'rating': review.rating,
            'aspect_scores': aspect_scores,
            'recommended_action': strategy.get('action', '观察'),
            'priority': strategy.get('priority', '低')
        }
    
    def analyze_batch(self, reviews: List[CustomerReview]) -> List[Dict]:
        """批量分析"""
        return [self.analyze(r) for r in reviews]
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """获取聚类摘要"""
        summary = []
        for k, label in self.cluster_label_map.items():
            count = np.sum(self.clustering.labels == k)
            summary.append({
                'cluster_id': k,
                'label': label,
                'count': count,
                'percentage': f"{count/len(self.clustering.labels)*100:.1f}%"
            })
        return pd.DataFrame(summary)


# ==================== 测试用例 ====================

def create_sample_reviews(n_samples: int = 200) -> List[CustomerReview]:
    """创建示例评论数据"""
    np.random.seed(42)
    
    # 各类别的模板评论
    templates = {
        'high_satisfaction': [
            "Great product! Love it and highly recommend to all moms.",
            "Excellent quality, my baby is very comfortable. Will buy again!",
            "Perfect! Fast delivery and amazing customer service.",
            "Best purchase ever! Safe and reliable product."
        ],
        'price_sensitive': [
            "Good product but a bit expensive compared to alternatives.",
            "Quality is okay but price is too high for what you get.",
            "Would buy more if there were discounts or promotions.",
            "Nice but looking for better value options."
        ],
        'quality_concern': [
            "Concerned about the material safety for my baby.",
            "Quality is not as expected, feels cheap.",
            "Not sure if this is safe enough for newborns.",
            "Material could be softer and more gentle."
        ],
        'service_complaint': [
            "Terrible service! Delivery was late and support was unhelpful.",
            "Product arrived damaged and return process is complicated.",
            "Very disappointed with customer service response.",
            "Wrong item sent and hard to get refund."
        ],
        'neutral': [
            "It's okay, nothing special but does the job.",
            "Average product, as expected.",
            "Haven't used it enough to form a strong opinion yet.",
            "Decent quality, fair price."
        ]
    }
    
    categories = ['milk', 'diaper', 'food', 'toy', 'clothes']
    base_time = pd.Timestamp('2024-01-01')
    
    reviews = []
    for i in range(n_samples):
        # 随机选择类别
        sentiment_type = np.random.choice(list(templates.keys()))
        text = np.random.choice(templates[sentiment_type])
        
        # 根据类型设置评分
        if sentiment_type == 'high_satisfaction':
            rating = np.random.choice([4, 5])
        elif sentiment_type == 'service_complaint':
            rating = np.random.choice([1, 2])
        elif sentiment_type == 'price_sensitive':
            rating = np.random.choice([3, 4])
        else:
            rating = np.random.choice([2, 3, 4])
        
        reviews.append(CustomerReview(
            review_id=f'R{i:05d}',
            user_id=f'U{i//2:04d}',
            text=text,
            rating=rating,
            timestamp=base_time + pd.Timedelta(hours=i),
            product_category=np.random.choice(categories)
        ))
    
    return reviews


def test_csk_sentiment_clustering():
    """测试CSK情感聚类"""
    print("=" * 70)
    print("CSK客户情感聚类测试")
    print("=" * 70)
    
    # 1. 创建数据
    reviews = create_sample_reviews(200)
    print(f"\n[OK] 生成 {len(reviews)} 条示例评论")
    
    # 评分分布
    ratings = [r.rating for r in reviews]
    print(f"[OK] 评分分布: 1星:{ratings.count(1)} 2星:{ratings.count(2)} 3星:{ratings.count(3)} 4星:{ratings.count(4)} 5星:{ratings.count(5)}")
    
    # 2. 训练模型
    print("\n" + "=" * 70)
    print("训练CSK模型")
    print("=" * 70)
    
    analyzer = CustomerSentimentAnalyzer(n_clusters=5)
    analyzer.fit(reviews)
    
    # 3. 聚类摘要
    print("\n" + "=" * 70)
    print("聚类分布")
    print("=" * 70)
    summary = analyzer.get_cluster_summary()
    print(summary.to_string(index=False))
    
    # 4. 单条分析
    print("\n" + "=" * 70)
    print("单条评论分析示例")
    print("=" * 70)
    
    test_reviews = [
        CustomerReview('T001', 'U0001', "Excellent quality! Love this product and recommend to all moms.", 5.0, pd.Timestamp.now(), 'diaper'),
        CustomerReview('T002', 'U0002', "Too expensive for what you get. Looking for discounts.", 3.0, pd.Timestamp.now(), 'milk'),
        CustomerReview('T003', 'U0003', "Terrible service! Delivery late and support unhelpful.", 1.0, pd.Timestamp.now(), 'toy'),
    ]
    
    for review in test_reviews:
        result = analyzer.analyze(review)
        print(f"\n评论: {review.text[:50]}...")
        print(f"  聚类: {result['cluster_label']} (ID:{result['sentiment_cluster']})")
        print(f"  情感分数: {result['sentiment_score']:.2f}")
        print(f"  评分: {result['rating']}")
        print(f"  建议动作: {result['recommended_action']}")
        print(f"  优先级: {result['priority']}")
    
    # 5. 批量分析
    print("\n" + "=" * 70)
    print("批量分析 (前10条)")
    print("=" * 70)
    
    results = analyzer.analyze_batch(reviews[:10])
    for r in results:
        print(f"评论{r['review_id']}: {r['cluster_label']:12s} | "
              f"情感:{r['sentiment_score']:+.2f} | "
              f"评分:{r['rating']} | "
              f"优先级:{r['priority']}")
    
    print("\n" + "=" * 70)
    print("测试完成 [OK]")
    print("=" * 70)
    
    return analyzer


if __name__ == "__main__":
    analyzer = test_csk_sentiment_clustering()
