"""
Auto-extracted from: paper2skills-vault/14-用户分析/Skill-Abandoned-Cart-Recovery-ML.md
Skill: Skill-Abandoned-Cart-Recovery-ML
Domain: 14-用户分析
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class RecoveryAction:
    channel: str        # 'email', 'sms', 'push'
    delay_hours: float  # 触达时机
    content_type: str   # 'reminder', 'social_proof', 'discount', 'friction_resolve'
    discount_pct: float # 0.0 = 无折扣


class CartAbandonmentClassifier:
    """
    弃购意图分类器：Gradient Boosting多分类
    """
    
    ABANDONMENT_TYPES = ['price_comparison', 'payment_friction', 'decision_hesitant', 'accidental']
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.le = LabelEncoder()
        self.feature_cols = [
            'session_duration_min', 'scroll_depth_pct', 'image_clicks',
            'cart_items', 'cart_value', 'is_first_purchase',
            'days_since_last_order', 'device_mobile', 'hour_of_day',
            'viewed_competitors', 'payment_page_reached'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        features = df[self.feature_cols].copy()
        features['cart_value_log'] = np.log1p(features['cart_value'])
        features['session_engagement'] = (
            features['scroll_depth_pct'] * features['session_duration_min'] / 100
        )
        return features
    
    def train(self, df: pd.DataFrame, labels: pd.Series):
        X = self.prepare_features(df)
        y = self.le.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=self.le.classes_))
        return self
    
    def predict(self, session_data: Dict) -> Tuple[str, float]:
        """预测弃购类型和概率"""
        df = pd.DataFrame([session_data])
        X = self.prepare_features(df)
        proba = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        return self.le.classes_[pred_idx], proba[pred_idx]


class RecoverySequenceOptimizer:
    """
    挽回序列决策器：基于弃购类型的规则+RL混合策略
    """
    
    # 基线规则（可被RL优化覆盖）
    BASE_STRATEGIES = {
        'accidental': [
            RecoveryAction('push', 0.5, 'reminder', 0.0),
            RecoveryAction('email', 24, 'reminder', 0.0),
        ],
        'price_comparison': [
            RecoveryAction('email', 1, 'social_proof', 0.0),
            RecoveryAction('email', 24, 'scarcity', 0.0),
            RecoveryAction('email', 72, 'last_chance', 0.0),
        ],
        'decision_hesitant': [
            RecoveryAction('email', 4, 'social_proof', 0.0),
            RecoveryAction('sms', 24, 'discount', 0.05),
            RecoveryAction('email', 72, 'discount_expiry', 0.05),
        ],
        'payment_friction': [
            RecoveryAction('email', 1, 'friction_resolve', 0.0),
            RecoveryAction('email', 48, 'discount', 0.0),
        ],
    }
    
    def get_recovery_plan(self, 
                           abandonment_type: str, 
                           cart_value: float,
                           days_to_promo: int = 99) -> List[RecoveryAction]:
        """生成个性化挽回计划"""
        plan = self.BASE_STRATEGIES.get(abandonment_type, []).copy()
        
        # 大促前抑制折扣（对比价型尤其重要）
        if days_to_promo <= 14 and abandonment_type == 'price_comparison':
            plan = [a for a in plan if a.discount_pct == 0]
        
        # 高客单价：增加额外挽回步骤
        if cart_value > 200 and abandonment_type == 'decision_hesitant':
            plan.append(RecoveryAction('sms', 96, 'vip_offer', 0.10))
        
        return plan


def generate_synthetic_data(n=5000):
    """生成模拟弃购数据"""
    np.random.seed(42)
    data = {
        'session_duration_min': np.random.exponential(8, n),
        'scroll_depth_pct': np.random.beta(2, 3, n) * 100,
        'image_clicks': np.random.poisson(3, n),
        'cart_items': np.random.poisson(2, n) + 1,
        'cart_value': np.random.lognormal(4.5, 0.8, n),
        'is_first_purchase': np.random.binomial(1, 0.4, n),
        'days_since_last_order': np.random.exponential(30, n),
        'device_mobile': np.random.binomial(1, 0.65, n),
        'hour_of_day': np.random.randint(0, 24, n),
        'viewed_competitors': np.random.binomial(1, 0.3, n),
        'payment_page_reached': np.random.binomial(1, 0.25, n),
    }
    df = pd.DataFrame(data)
    
    # 生成标签（基于规则）
    labels = []
    for _, row in df.iterrows():
        if row['payment_page_reached'] > 0.5:
            labels.append('payment_friction')
        elif row['viewed_competitors'] > 0.5 and row['session_duration_min'] < 3:
            labels.append('price_comparison')
        elif row['scroll_depth_pct'] > 80 and row['session_duration_min'] > 10:
            labels.append('decision_hesitant')
        elif row['session_duration_min'] < 1:
            labels.append('accidental')
        else:
            labels.append(np.random.choice(
                ['price_comparison', 'decision_hesitant', 'accidental'],
                p=[0.6, 0.3, 0.1]
            ))
    
    return df, pd.Series(labels)


if __name__ == '__main__':
    print("=== 弃购挽回ML系统 ===\n")
    
    # 训练分类器
    df, labels = generate_synthetic_data(5000)
    classifier = CartAbandonmentClassifier()
    classifier.train(df, labels)
    
    # 预测示例
    optimizer = RecoverySequenceOptimizer()
    
    test_sessions = [
        {'session_duration_min': 2, 'scroll_depth_pct': 30, 'image_clicks': 1,
         'cart_items': 1, 'cart_value': 180, 'is_first_purchase': 0,
         'days_since_last_order': 45, 'device_mobile': 1, 'hour_of_day': 21,
         'viewed_competitors': 1, 'payment_page_reached': 0},
        {'session_duration_min': 15, 'scroll_depth_pct': 95, 'image_clicks': 8,
         'cart_items': 2, 'cart_value': 320, 'is_first_purchase': 1,
         'days_since_last_order': 999, 'device_mobile': 0, 'hour_of_day': 20,
         'viewed_competitors': 0, 'payment_page_reached': 0},
    ]
    
    print("\n--- 个性化挽回计划 ---")
    for i, session in enumerate(test_sessions, 1):
        atype, conf = classifier.predict(session)
        plan = optimizer.get_recovery_plan(atype, session['cart_value'], days_to_promo=30)
        
        print(f"\n用户{i}（购物车${session['cart_value']:.0f}）:")
        print(f"  弃购类型: {atype} (置信度: {conf:.2f})")
        print(f"  挽回计划:")
        for step, action in enumerate(plan, 1):
            discount_str = f" | 折扣{action.discount_pct*100:.0f}%" if action.discount_pct > 0 else ""
            print(f"    Step {step}: T+{action.delay_hours}h → {action.channel} | {action.content_type}{discount_str}")
