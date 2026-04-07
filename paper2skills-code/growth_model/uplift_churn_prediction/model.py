"""
Uplift Modeling for Churn Prediction
Uplift建模在用户流失预测中的应用

论文: A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling
arXiv: 2312.07206 (2023)
会议: ECML PKDD 2023 Workshop

核心方法:
- Uplift Modeling (个体干预效应估计 ITE)
- T-Learner: 分别训练处理组和对照组模型
- S-Learner: 将干预作为特征输入单一模型
- X-Learner: 结合T-Learner和S-Learner优势
- Causal Forest: 基于树的非参数ITE估计
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CustomerData:
    """客户数据"""
    customer_id: str
    features: np.ndarray  # 特征向量
    treatment: int  # 0=对照组, 1=处理组(收到干预)
    outcome: int  # 0=未流失, 1=流失
    lifecycle_stage: str  # 'Awareness', 'Interest', 'Purchase', 'Loyalty'
    sentiment_cluster: str  # 来自CSK的情感分群


class BaseUpliftModel:
    """Uplift建模基类"""
    
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """训练模型"""
        raise NotImplementedError
    
    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """预测个体干预效应 (ITE)"""
        raise NotImplementedError
    
    def predict_uplift_score(self, X: np.ndarray) -> np.ndarray:
        """预测Uplift分数"""
        return self.predict_ite(X)


class TLearner(BaseUpliftModel):
    """
    T-Learner (Two-Learner)
    分别训练处理组和对照组的模型，然后相减得到ITE
    
    ITE(x) = μ₁(x) - μ₀(x)
    其中 μ₁: 处理组模型, μ₀: 对照组模型
    """
    
    def __init__(self, base_model=None):
        super().__init__()
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model_treated = base_model
        self.model_control = base_model.__class__(**base_model.get_params())
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """训练T-Learner"""
        # 分割处理组和对照组
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        print(f"训练T-Learner: 处理组 {len(X_treated)}, 对照组 {len(X_control)}")
        
        # 分别训练两个模型
        if len(X_treated) > 0:
            self.model_treated.fit(X_treated, y_treated)
        if len(X_control) > 0:
            self.model_control.fit(X_control, y_control)
        
        self.is_fitted = True
        return self
    
    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """预测ITE: 处理组概率 - 对照组概率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 预测处理组和对照组的流失概率
        prob_treated = self.model_treated.predict_proba(X)[:, 1]  # 假设1是流失
        prob_control = self.model_control.predict_proba(X)[:, 1]
        
        # ITE = P(Y=1|T=1,X) - P(Y=1|T=0,X)
        # 负值表示干预降低流失概率（好效果）
        ite = prob_treated - prob_control
        
        return -ite  # 负号：我们希望看到负的ITE（干预降低流失）


class SLearner(BaseUpliftModel):
    """
    S-Learner (Single-Learner)
    将干预作为特征输入单一模型
    
    ITE(x) = μ(x, T=1) - μ(x, T=0)
    """
    
    def __init__(self, base_model=None):
        super().__init__()
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model = base_model
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """训练S-Learner"""
        # 将干预作为特征
        X_with_treatment = np.column_stack([X, treatment])
        
        print(f"训练S-Learner: 样本数 {len(X)}")
        
        self.model.fit(X_with_treatment, y)
        self.is_fitted = True
        return self
    
    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """预测ITE"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 构造处理组和对照组输入
        X_treated = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])
        
        # 预测
        prob_treated = self.model.predict_proba(X_treated)[:, 1]
        prob_control = self.model.predict_proba(X_control)[:, 1]
        
        ite = prob_treated - prob_control
        return -ite


class XLearner(BaseUpliftModel):
    """
    X-Learner
    结合T-Learner和S-Learner的优势
    步骤:
    1. 训练处理组和对照组模型
    2. 计算imputed treatment effects
    3. 用另一个模型预测ITE
    """
    
    def __init__(self, base_model=None):
        super().__init__()
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model_treated = base_model
        self.model_control = base_model.__class__(**base_model.get_params())
        # ITE预测需要回归模型，因为imputed_effects是连续值
        self.model_ite = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """训练X-Learner"""
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        print(f"训练X-Learner: 处理组 {len(X_treated)}, 对照组 {len(X_control)}")
        
        # 步骤1: 训练基础模型
        self.model_treated.fit(X_treated, y_treated)
        self.model_control.fit(X_control, y_control)
        
        # 步骤2: 计算imputed treatment effects
        # 对于处理组: D = Y - μ₀(X)
        # 对于对照组: D = μ₁(X) - Y
        imputed_effects = np.zeros(len(X))
        
        if len(X_treated) > 0:
            prob_control_for_treated = self.model_control.predict_proba(X_treated)[:, 1]
            imputed_effects[treatment == 1] = y_treated - prob_control_for_treated
        
        if len(X_control) > 0:
            prob_treated_for_control = self.model_treated.predict_proba(X_control)[:, 1]
            imputed_effects[treatment == 0] = prob_treated_for_control - y_control
        
        # 步骤3: 训练ITE预测模型
        self.model_ite.fit(X, imputed_effects)
        
        self.is_fitted = True
        return self
    
    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """预测ITE"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        return self.model_ite.predict(X)


class UpliftMetrics:
    """Uplift模型评估指标"""
    
    @staticmethod
    def qini_curve(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算Qini曲线
        
        Returns:
            (thresholds, qini_values)
        """
        # 按uplift分数降序排序
        order = np.argsort(-uplift)
        y_true = y_true[order]
        treatment = treatment[order]
        
        n = len(y_true)
        n_t = np.sum(treatment)
        n_c = n - n_t
        
        y_t = y_true[treatment == 1]
        y_c = y_true[treatment == 0]
        
        # 累计统计
        cumsum_y_t = np.cumsum(treatment * y_true)
        cumsum_y_c = np.cumsum((1 - treatment) * y_true)
        cumsum_n_t = np.cumsum(treatment)
        cumsum_n_c = np.cumsum(1 - treatment)
        
        # 避免除零
        cumsum_n_t = np.maximum(cumsum_n_t, 1)
        cumsum_n_c = np.maximum(cumsum_n_c, 1)
        
        # Qini值
        qini = cumsum_y_t / cumsum_n_t - cumsum_y_c / cumsum_n_c
        qini = qini * (cumsum_n_t + cumsum_n_c)  # 加权
        
        thresholds = np.arange(1, n + 1) / n
        
        return thresholds, qini
    
    @staticmethod
    def auuc(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
        """
        计算AUUC (Area Under the Uplift Curve)
        """
        thresholds, qini = UpliftMetrics.qini_curve(y_true, uplift, treatment)
        
        # 计算曲线下的面积
        auuc = np.trapz(qini, thresholds)
        
        return auuc
    
    @staticmethod
    def qini_coefficient(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
        """
        Qini系数 (归一化的AUUC)
        """
        thresholds, qini = UpliftMetrics.qini_curve(y_true, uplift, treatment)
        
        # 计算随机线的Qini
        n = len(y_true)
        n_t = np.sum(treatment)
        n_c = n - n_t
        
        y_t_sum = np.sum(y_true[treatment == 1])
        y_c_sum = np.sum(y_true[treatment == 0])
        
        # 随机线
        random_qini = (y_t_sum / n_t - y_c_sum / n_c) * thresholds * n
        
        # Qini系数
        qini_auc = np.trapz(qini, thresholds)
        random_auc = np.trapz(random_qini, thresholds)
        
        if random_auc == 0:
            return 0
        
        return (qini_auc - random_auc) / random_auc


class CustomerUpliftAnalyzer:
    """
    客户Uplift分析器
    整合多种Uplift模型，提供业务洞察
    """
    
    USER_SEGMENTS = {
        'Persuadables': 'Sure Things',
        '可说服': '必然转化',
        'Lost Causes': 'Sleeping Dogs',
        '无法挽回': '不要打扰',
        'Sure Things': 'Persuadables',
        '必然转化': '可说服',
        'Sleeping Dogs': 'Lost Causes',
        '不要打扰': '无法挽回'
    }
    
    def __init__(self, model_type: str = 'xlearner'):
        """
        Args:
            model_type: 'tlearner', 'slearner', 'xlearner'
        """
        if model_type == 'tlearner':
            self.model = TLearner()
        elif model_type == 'slearner':
            self.model = SLearner()
        elif model_type == 'xlearner':
            self.model = XLearner()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        self.metrics = UpliftMetrics()
    
    def fit(self, customers: List[CustomerData]):
        """训练Uplift模型"""
        X = np.array([c.features for c in customers])
        treatment = np.array([c.treatment for c in customers])
        y = np.array([c.outcome for c in customers])
        
        print(f"\n训练 {self.model_type.upper()} Uplift模型")
        print(f"总样本: {len(customers)}")
        print(f"处理组: {np.sum(treatment)}, 对照组: {len(treatment) - np.sum(treatment)}")
        print(f"流失: {np.sum(y)}, 未流失: {len(y) - np.sum(y)}")
        
        self.model.fit(X, treatment, y)
        
        # 保存训练数据用于评估
        self.train_customers = customers
        
        return self
    
    def analyze_customer(self, customer: CustomerData) -> Dict:
        """
        分析单个客户的Uplift
        
        Returns:
            {
                'customer_id': str,
                'uplift_score': float,  # 干预效果分数
                'segment': str,  # 用户分群
                'recommendation': str  # 建议
            }
        """
        X = customer.features.reshape(1, -1)
        uplift = self.model.predict_ite(X)[0]
        
        # 用户分群
        if uplift > 0.1:
            segment = '可说服 (Persuadables)'
            recommendation = '发送优惠券，预计显著降低流失'
        elif uplift < -0.1:
            segment = '不要打扰 (Sleeping Dogs)'
            recommendation = '避免干预，干预可能增加流失'
        elif uplift > 0:
            segment = '必然转化 (Sure Things)'
            recommendation = '无需干预，用户会自然留存'
        else:
            segment = '无法挽回 (Lost Causes)'
            recommendation = '放弃干预，节省营销预算'
        
        return {
            'customer_id': customer.customer_id,
            'uplift_score': float(uplift),
            'segment': segment,
            'recommendation': recommendation,
            'lifecycle_stage': customer.lifecycle_stage,
            'sentiment_cluster': customer.sentiment_cluster
        }
    
    def evaluate(self, customers: List[CustomerData]) -> Dict:
        """评估模型性能"""
        X = np.array([c.features for c in customers])
        treatment = np.array([c.treatment for c in customers])
        y = np.array([c.outcome for c in customers])
        
        # 预测Uplift
        uplift = self.model.predict_ite(X)
        
        # 计算Qini曲线
        thresholds, qini = self.metrics.qini_curve(y, uplift, treatment)
        
        # 计算AUUC
        auuc = self.metrics.auuc(y, uplift, treatment)
        
        # Qini系数
        qini_coef = self.metrics.qini_coefficient(y, uplift, treatment)
        
        return {
            'auuc': auuc,
            'qini_coefficient': qini_coef,
            'thresholds': thresholds,
            'qini_values': qini
        }


# ==================== 测试用例 ====================

def create_sample_customers(n_samples: int = 1000) -> List[CustomerData]:
    """创建示例客户数据"""
    np.random.seed(42)
    
    customers = []
    
    for i in range(n_samples):
        # 生成特征
        tenure = np.random.randint(1, 72)  # 在网时长(月)
        monthly_charges = np.random.uniform(20, 120)  # 月消费
        total_charges = monthly_charges * tenure * np.random.uniform(0.8, 1.2)
        num_support_calls = np.random.poisson(2)  # 客服通话次数
        num_products = np.random.randint(1, 5)  # 产品数量
        
        # 构建特征向量
        features = np.array([
            tenure / 72.0,  # 归一化
            monthly_charges / 120.0,
            total_charges / 8000.0,
            num_support_calls / 10.0,
            num_products / 5.0
        ])
        
        # 生成处理组标签 (随机分配干预)
        treatment = np.random.binomial(1, 0.5)
        
        # 生成流失标签 (处理组流失率更低)
        base_churn_prob = 0.2
        if treatment == 1:
            churn_prob = base_churn_prob * 0.6  # 干预降低40%流失
        else:
            churn_prob = base_churn_prob
        
        # 特征影响流失概率
        if tenure < 12:
            churn_prob += 0.2
        if num_support_calls > 3:
            churn_prob += 0.15
        if monthly_charges > 80:
            churn_prob += 0.1
        
        churn_prob = min(churn_prob, 0.9)
        outcome = np.random.binomial(1, churn_prob)
        
        # 生命周期阶段
        if tenure < 6:
            stage = 'Awareness'
        elif tenure < 24:
            stage = 'Interest'
        elif outcome == 0:
            stage = 'Loyalty'
        else:
            stage = 'Purchase'
        
        # 情感分群
        sentiment = np.random.choice(['高满意', '价格敏感', '质量关注', '服务抱怨', '中性'])
        
        customers.append(CustomerData(
            customer_id=f'C{i:05d}',
            features=features,
            treatment=treatment,
            outcome=outcome,
            lifecycle_stage=stage,
            sentiment_cluster=sentiment
        ))
    
    return customers


def test_uplift_modeling():
    """测试Uplift建模"""
    print("=" * 70)
    print("Uplift Modeling for Churn Prediction 测试")
    print("=" * 70)
    
    # 1. 创建数据
    customers = create_sample_customers(1000)
    train_customers = customers[:700]
    test_customers = customers[700:]
    
    print(f"\n[OK] 生成 {len(customers)} 条客户数据")
    print(f"[OK] 训练集: {len(train_customers)}, 测试集: {len(test_customers)}")
    
    # 统计
    train_treatment = sum(c.treatment for c in train_customers)
    train_churn = sum(c.outcome for c in train_customers)
    print(f"[OK] 训练集 - 处理组: {train_treatment}, 流失: {train_churn}")
    
    # 2. 训练三种模型
    models = {
        'T-Learner': TLearner(),
        'S-Learner': SLearner(),
        'X-Learner': XLearner()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"训练 {name}")
        print(f"{'='*70}")
        
        # 准备数据
        X_train = np.array([c.features for c in train_customers])
        treatment_train = np.array([c.treatment for c in train_customers])
        y_train = np.array([c.outcome for c in train_customers])
        
        # 训练
        model.fit(X_train, treatment_train, y_train)
        
        # 预测测试集
        X_test = np.array([c.features for c in test_customers])
        uplift_scores = model.predict_ite(X_test)
        
        # 评估
        analyzer = CustomerUpliftAnalyzer()
        analyzer.model = model
        metrics = analyzer.evaluate(test_customers)
        
        results[name] = {
            'model': model,
            'uplift_scores': uplift_scores,
            'auuc': metrics['auuc'],
            'qini_coef': metrics['qini_coefficient']
        }
        
        print(f"\n评估结果:")
        print(f"  AUUC: {metrics['auuc']:.4f}")
        print(f"  Qini系数: {metrics['qini_coefficient']:.4f}")
    
    # 3. 对比模型
    print(f"\n{'='*70}")
    print("模型对比")
    print(f"{'='*70}")
    
    for name, result in results.items():
        print(f"{name:15s}: AUUC={result['auuc']:.4f}, Qini={result['qini_coef']:.4f}")
    
    # 4. 用户分群分析
    print(f"\n{'='*70}")
    print("用户分群分析 (使用X-Learner)")
    print(f"{'='*70}")
    
    best_model = results['X-Learner']['model']
    analyzer = CustomerUpliftAnalyzer('xlearner')
    analyzer.model = best_model
    
    # 分析前10个测试客户
    segments = defaultdict(list)
    
    for customer in test_customers[:20]:
        result = analyzer.analyze_customer(customer)
        segments[result['segment']].append(result)
    
    for segment, customers_list in segments.items():
        avg_uplift = np.mean([c['uplift_score'] for c in customers_list])
        print(f"\n{segment}: {len(customers_list)}人")
        print(f"  平均Uplift: {avg_uplift:.4f}")
        print(f"  示例: {customers_list[0]['recommendation']}")
    
    # 5. 业务价值计算
    print(f"\n{'='*70}")
    print("业务价值预估")
    print(f"{'='*70}")
    
    # 假设
    n_customers = 10000
    intervention_cost = 10  # 每次干预成本(元)
    customer_value = 500  # 客户价值(元)
    
    # 使用Uplift模型 targeting
    persuadables_ratio = 0.2  # 20%是可说服的
    persuadables = int(n_customers * persuadables_ratio)
    churn_reduction = 0.4  # 干预降低40%流失
    
    # 节省的价值
    saved_customers = int(persuadables * churn_reduction)
    saved_value = saved_customers * customer_value
    intervention_total_cost = persuadables * intervention_cost
    net_value = saved_value - intervention_total_cost
    roi = net_value / intervention_total_cost
    
    print(f"假设场景: {n_customers}客户, 干预成本{intervention_cost}元, 客户价值{customer_value}元")
    print(f"可说服客户: {persuadables}人 ({persuadables_ratio*100}%)")
    print(f"预计挽回: {saved_customers}人")
    print(f"挽回价值: {saved_value:,}元")
    print(f"干预成本: {intervention_total_cost:,}元")
    print(f"净收益: {net_value:,}元")
    print(f"ROI: {roi:.1f}倍")
    
    print(f"\n{'='*70}")
    print("测试完成 [OK]")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    results = test_uplift_modeling()
