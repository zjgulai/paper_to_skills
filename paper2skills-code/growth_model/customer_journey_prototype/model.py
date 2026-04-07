"""
Customer Journey Prototype Detection with Counterfactual Explanations
客户旅程序列原型检测与反事实解释

论文: Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations
arXiv: 2505.11086 (2025)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')


@dataclass
class JourneyEvent:
    """客户旅程事件"""
    channel: str  # 'app', 'web', 'mini_program', 'offline'
    action: str   # 'browse', 'search', 'click', 'cart', 'purchase', 'review'
    category: str # 'milk', 'diaper', 'food', 'toy', 'clothes'
    timestamp: pd.Timestamp
    duration: float = 0.0  # 停留时长(秒)


@dataclass
class CustomerJourney:
    """客户完整旅程"""
    user_id: str
    events: List[JourneyEvent]
    has_purchase: bool = False
    purchase_amount: float = 0.0


class SequenceDistance:
    """
    序列距离计算器
    支持多种距离度量方式
    """
    
    @staticmethod
    def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
        """
        编辑距离 (Levenshtein Distance)
        计算两个序列之间的最小编辑操作数
        """
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return int(dp[m][n])
    
    @staticmethod
    def normalized_distance(seq1: List[str], seq2: List[str]) -> float:
        """归一化编辑距离 (0-1范围)"""
        raw_dist = SequenceDistance.levenshtein_distance(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return raw_dist / max_len if max_len > 0 else 0.0
    
    @staticmethod
    def journey_distance(journey1: CustomerJourney, journey2: CustomerJourney) -> float:
        """
        客户旅程距离
        综合考虑渠道、行为和品类的序列相似度
        """
        # 将旅程编码为字符串序列
        seq1 = [f"{e.channel}:{e.action}:{e.category}" for e in journey1.events]
        seq2 = [f"{e.channel}:{e.action}:{e.category}" for e in journey2.events]
        
        return SequenceDistance.normalized_distance(seq1, seq2)


class PrototypeDetector:
    """
    原型序列检测器
    识别代表性的客户旅程序列
    """
    
    def __init__(self, n_prototypes: int = 5):
        self.n_prototypes = n_prototypes
        self.prototypes: List[CustomerJourney] = []
        self.prototype_labels: List[str] = []
        
    def fit(self, journeys: List[CustomerJourney]) -> 'PrototypeDetector':
        """
        检测原型序列
        使用基于距离的最大化覆盖算法
        """
        if len(journeys) <= self.n_prototypes:
            self.prototypes = journeys
            return self
        
        # 计算所有两两距离
        n = len(journeys)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = SequenceDistance.journey_distance(journeys[i], journeys[j])
                distances[i][j] = distances[j][i] = dist
        
        # 选择原型：最大化相互距离
        selected = [0]  # 从第一个开始
        while len(selected) < self.n_prototypes:
            max_min_dist = -1
            best_idx = -1
            
            for i in range(n):
                if i in selected:
                    continue
                # 计算到已选原型的最小距离
                min_dist = min(distances[i][j] for j in selected)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            selected.append(best_idx)
        
        self.prototypes = [journeys[i] for i in selected]
        self._generate_labels()
        return self
    
    def _generate_labels(self):
        """为原型生成业务标签"""
        labels = [
            "App深度浏览型",
            "跨渠道比价型",
            "小程序冲动型",
            "线下体验型",
            "搜索精准型"
        ]
        self.prototype_labels = [labels[i % len(labels)] for i in range(len(self.prototypes))]
    
    def assign_to_prototype(self, journey: CustomerJourney) -> Tuple[int, float]:
        """
        将旅程分配到最近的原型
        
        Returns:
            (原型索引, 距离)
        """
        min_dist = float('inf')
        best_idx = 0
        
        for i, proto in enumerate(self.prototypes):
            dist = SequenceDistance.journey_distance(journey, proto)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        return best_idx, min_dist
    
    def get_prototype_summary(self) -> List[Dict]:
        """获取原型摘要统计"""
        summaries = []
        for i, proto in enumerate(self.prototypes):
            event_types = defaultdict(int)
            channels = defaultdict(int)
            for e in proto.events:
                event_types[e.action] += 1
                channels[e.channel] += 1
            
            summaries.append({
                'prototype_id': i,
                'label': self.prototype_labels[i],
                'n_events': len(proto.events),
                'event_distribution': dict(event_types),
                'channel_distribution': dict(channels),
                'has_purchase': proto.has_purchase
            })
        return summaries


class PurchasePredictor:
    """
    购买概率预测器
    基于到原型的距离预测购买可能性
    """
    
    def __init__(self, prototype_detector: PrototypeDetector):
        self.detector = prototype_detector
        self.prototype_conversion_rates: Dict[int, float] = {}
        
    def fit(self, journeys: List[CustomerJourney]):
        """
        基于历史数据计算各原型的转化率
        """
        # 将每个旅程分配到原型
        proto_purchases = defaultdict(lambda: {'total': 0, 'purchases': 0})
        
        for journey in journeys:
            proto_idx, _ = self.detector.assign_to_prototype(journey)
            proto_purchases[proto_idx]['total'] += 1
            if journey.has_purchase:
                proto_purchases[proto_idx]['purchases'] += 1
        
        # 计算转化率
        for idx, stats in proto_purchases.items():
            self.prototype_conversion_rates[idx] = (
                stats['purchases'] / stats['total'] if stats['total'] > 0 else 0
            )
    
    def predict(self, journey: CustomerJourney) -> Dict:
        """
        预测购买概率
        
        Returns:
            {
                'purchase_probability': float,
                'assigned_prototype': int,
                'prototype_label': str,
                'distance_to_prototype': float,
                'confidence': str  # 'high', 'medium', 'low'
            }
        """
        proto_idx, distance = self.detector.assign_to_prototype(journey)
        base_rate = self.prototype_conversion_rates.get(proto_idx, 0.5)
        
        # 根据距离调整概率（距离越远，置信度越低）
        adjusted_prob = base_rate * (1 - distance * 0.3)
        adjusted_prob = np.clip(adjusted_prob, 0, 1)
        
        # 置信度判断
        if distance < 0.3:
            confidence = 'high'
        elif distance < 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'purchase_probability': float(adjusted_prob),
            'assigned_prototype': proto_idx,
            'prototype_label': self.detector.prototype_labels[proto_idx],
            'distance_to_prototype': float(distance),
            'confidence': confidence
        }


class CounterfactualRecommender:
    """
    反事实序列推荐器
    推荐修改以提升购买概率
    """
    
    def __init__(self, prototype_detector: PrototypeDetector, 
                 purchase_predictor: PurchasePredictor):
        self.detector = prototype_detector
        self.predictor = purchase_predictor
    
    def recommend(self, journey: CustomerJourney) -> Dict:
        """
        生成反事实推荐
        
        Returns:
            {
                'current_probability': float,
                'recommended_changes': List[str],
                'target_prototype': int,
                'expected_probability': float,
                'action_plan': List[str]
            }
        """
        current = self.predictor.predict(journey)
        current_prob = current['purchase_probability']
        
        # 如果概率已很高，无需推荐
        if current_prob > 0.7:
            return {
                'current_probability': current_prob,
                'recommended_changes': [],
                'message': '当前旅程购买概率已较高，无需优化'
            }
        
        # 找到转化率最高的原型
        best_proto = max(self.predictor.prototype_conversion_rates.items(),
                        key=lambda x: x[1])
        target_idx, target_rate = best_proto
        
        target_proto = self.detector.prototypes[target_idx]
        
        # 生成改进建议
        recommendations = self._generate_recommendations(journey, target_proto)
        
        return {
            'current_probability': current_prob,
            'target_prototype': target_idx,
            'target_label': self.detector.prototype_labels[target_idx],
            'expected_probability': target_rate,
            'recommended_changes': recommendations['changes'],
            'action_plan': recommendations['actions']
        }
    
    def _generate_recommendations(self, current: CustomerJourney, 
                                   target: CustomerJourney) -> Dict:
        """生成具体的改进建议"""
        changes = []
        actions = []
        
        # 分析渠道差异
        current_channels = [e.channel for e in current.events]
        target_channels = [e.channel for e in target.events]
        
        if 'app' in target_channels and 'app' not in current_channels:
            changes.append("增加App端深度浏览")
            actions.append("推送App专属优惠券引导下载")
        
        if 'offline' in target_channels and 'offline' not in current_channels:
            changes.append("增加线下门店体验")
            actions.append("推送附近门店地址和体验活动")
        
        # 分析行为序列差异
        current_actions = [e.action for e in current.events]
        
        if 'search' not in current_actions:
            changes.append("增加主动搜索行为")
            actions.append("优化搜索推荐，展示热门搜索词")
        
        if current_actions.count('cart') == 0:
            changes.append("增加加购行为")
            actions.append("商品详情页突出限时优惠，促进加购")
        
        if 'review' not in current_actions and len(current_actions) > 5:
            changes.append("增加评价浏览")
            actions.append("在商品页突出好评内容和买家秀")
        
        # 如果变化太少，添加通用建议
        if len(changes) < 2:
            changes.append("增加跨品类浏览")
            actions.append("首页推荐相关品类商品")
        
        return {'changes': changes, 'actions': actions}


class CustomerJourneyAnalyzer:
    """
    客户旅程分析系统
    整合原型检测、购买预测、反事实推荐
    """
    
    def __init__(self, n_prototypes: int = 5):
        self.detector = PrototypeDetector(n_prototypes)
        self.predictor = None
        self.recommender = None
        
    def fit(self, journeys: List[CustomerJourney]):
        """训练分析模型"""
        print(f"训练数据: {len(journeys)} 条客户旅程")
        
        # 1. 检测原型
        self.detector.fit(journeys)
        print(f"检测到 {len(self.detector.prototypes)} 个原型序列")
        
        # 2. 训练购买预测器
        self.predictor = PurchasePredictor(self.detector)
        self.predictor.fit(journeys)
        
        # 3. 初始化推荐器
        self.recommender = CounterfactualRecommender(self.detector, self.predictor)
        
        print("模型训练完成")
        
    def analyze_journey(self, journey: CustomerJourney) -> Dict:
        """
        分析单个客户旅程
        
        Returns:
            完整的分析报告
        """
        # 购买预测
        prediction = self.predictor.predict(journey)
        
        # 反事实推荐
        recommendation = self.recommender.recommend(journey)
        
        # 旅程统计
        stats = self._journey_stats(journey)
        
        return {
            'user_id': journey.user_id,
            'journey_stats': stats,
            'purchase_prediction': prediction,
            'optimization_recommendation': recommendation
        }
    
    def _journey_stats(self, journey: CustomerJourney) -> Dict:
        """计算旅程统计信息"""
        events = journey.events
        channels = [e.channel for e in events]
        actions = [e.action for e in events]
        
        return {
            'total_events': len(events),
            'unique_channels': len(set(channels)),
            'channel_sequence': channels,
            'action_distribution': dict(pd.Series(actions).value_counts()),
            'total_duration': sum(e.duration for e in events),
            'has_purchase': journey.has_purchase
        }
    
    def get_prototype_report(self) -> pd.DataFrame:
        """获取原型分析报告"""
        summaries = self.detector.get_prototype_summary()
        
        # 添加转化率
        for s in summaries:
            s['conversion_rate'] = self.predictor.prototype_conversion_rates.get(
                s['prototype_id'], 0
            )
        
        return pd.DataFrame(summaries)


# ==================== 测试用例 ====================

def create_sample_journeys(n_samples: int = 100) -> List[CustomerJourney]:
    """创建示例客户旅程数据"""
    np.random.seed(42)
    journeys = []
    
    channels = ['app', 'web', 'mini_program', 'offline']
    actions = ['browse', 'search', 'click', 'cart', 'purchase', 'review']
    categories = ['milk', 'diaper', 'food', 'toy', 'clothes']
    
    base_time = pd.Timestamp('2024-01-01')
    
    for i in range(n_samples):
        n_events = np.random.randint(3, 15)
        events = []
        
        # 模拟不同类型旅程
        journey_type = np.random.choice(['browser', 'searcher', 'buyer'], p=[0.4, 0.3, 0.3])
        
        for j in range(n_events):
            timestamp = base_time + pd.Timedelta(minutes=j*10 + np.random.randint(0, 5))
            
            if journey_type == 'browser':
                action = np.random.choice(['browse', 'click', 'cart'], p=[0.6, 0.3, 0.1])
                channel = np.random.choice(['app', 'web'], p=[0.7, 0.3])
            elif journey_type == 'searcher':
                action = np.random.choice(['search', 'click', 'browse', 'cart'], p=[0.4, 0.3, 0.2, 0.1])
                channel = np.random.choice(['app', 'web', 'mini_program'], p=[0.4, 0.4, 0.2])
            else:  # buyer
                action = np.random.choice(['browse', 'click', 'cart', 'purchase', 'review'], 
                                         p=[0.2, 0.2, 0.2, 0.3, 0.1])
                channel = np.random.choice(['app', 'web', 'offline'], p=[0.5, 0.3, 0.2])
            
            events.append(JourneyEvent(
                channel=channel,
                action=action,
                category=np.random.choice(categories),
                timestamp=timestamp,
                duration=np.random.uniform(10, 300)
            ))
        
        has_purchase = journey_type == 'buyer' or (journey_type == 'searcher' and np.random.random() > 0.7)
        
        journeys.append(CustomerJourney(
            user_id=f'U{i:04d}',
            events=events,
            has_purchase=has_purchase,
            purchase_amount=np.random.uniform(100, 1000) if has_purchase else 0
        ))
    
    return journeys


def test_customer_journey_analysis():
    """测试客户旅程分析系统"""
    print("=" * 70)
    print("客户旅程序列原型检测与反事实推荐测试")
    print("=" * 70)
    
    # 1. 创建数据
    journeys = create_sample_journeys(100)
    print(f"\n[OK] 生成 {len(journeys)} 条示例客户旅程")
    
    # 统计
    purchase_count = sum(1 for j in journeys if j.has_purchase)
    print(f"[OK] 购买转化: {purchase_count}/{len(journeys)} ({purchase_count/len(journeys):.1%})")
    
    # 2. 训练模型
    print("\n" + "=" * 70)
    print("训练模型")
    print("=" * 70)
    analyzer = CustomerJourneyAnalyzer(n_prototypes=5)
    analyzer.fit(journeys)
    
    # 3. 原型报告
    print("\n" + "=" * 70)
    print("原型序列报告")
    print("=" * 70)
    proto_report = analyzer.get_prototype_report()
    print(proto_report.to_string(index=False))
    
    # 4. 单用户分析
    print("\n" + "=" * 70)
    print("单用户旅程分析示例")
    print("=" * 70)
    
    # 找一个未购买的例子
    non_purchase_journeys = [j for j in journeys if not j.has_purchase]
    if non_purchase_journeys:
        sample_journey = non_purchase_journeys[0]
        analysis = analyzer.analyze_journey(sample_journey)
        
        print(f"\n用户ID: {analysis['user_id']}")
        print(f"旅程事件数: {analysis['journey_stats']['total_events']}")
        print(f"渠道序列: {' → '.join(analysis['journey_stats']['channel_sequence'])}")
        
        print(f"\n购买预测:")
        pred = analysis['purchase_prediction']
        print(f"  - 购买概率: {pred['purchase_probability']:.2%}")
        print(f"  - 归属原型: {pred['prototype_label']} (ID: {pred['assigned_prototype']})")
        print(f"  - 预测置信度: {pred['confidence']}")
        
        print(f"\n优化建议:")
        rec = analysis['optimization_recommendation']
        if rec.get('recommended_changes'):
            print(f"  当前购买概率: {rec['current_probability']:.2%}")
            print(f"  目标原型: {rec['target_label']}")
            print(f"  预期购买概率: {rec['expected_probability']:.2%}")
            print(f"\n  建议改进:")
            for change in rec['recommended_changes']:
                print(f"    • {change}")
            print(f"\n  运营动作:")
            for action in rec['action_plan']:
                print(f"    → {action}")
        else:
            print(f"  {rec.get('message', '无需优化')}")
    
    # 5. 批量分析
    print("\n" + "=" * 70)
    print("批量分析 (10个用户)")
    print("=" * 70)
    
    for i, journey in enumerate(journeys[:10]):
        analysis = analyzer.analyze_journey(journey)
        pred = analysis['purchase_prediction']
        status = "已购买" if journey.has_purchase else "未购买"
        print(f"用户{journey.user_id}: {pred['prototype_label']:12s} "
              f"| 购买概率: {pred['purchase_probability']:5.1%} "
              f"| 实际: {status}")
    
    print("\n" + "=" * 70)
    print("测试完成 [OK]")
    print("=" * 70)
    
    return analyzer


if __name__ == "__main__":
    analyzer = test_customer_journey_analysis()
