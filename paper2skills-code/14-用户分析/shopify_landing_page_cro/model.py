"""
Auto-extracted from: paper2skills-vault/14-用户分析/Skill-Shopify-Landing-Page-CRO.md
Skill: Skill-Shopify-Landing-Page-CRO
Domain: 14-用户分析
"""
import numpy as np
from scipy.stats import beta
from itertools import product
from typing import List, Dict, Tuple
import random

class LandingPageBayesianOptimizer:
    """
    贝叶斯优化落地页CRO：Thompson Sampling多臂老虎机
    适用于Shopify等独立站的多元素配置优化
    """
    
    def __init__(self, elements: Dict[str, List[str]]):
        """
        elements: {'headline': ['A', 'B', 'C'], 'cta': ['X', 'Y'], ...}
        """
        self.elements = elements
        self.element_names = list(elements.keys())
        
        # 生成所有配置组合
        values = [elements[k] for k in self.element_names]
        self.configs = list(product(*values))
        
        # Beta分布参数（先验：alpha=1, beta=1 即均匀分布）
        self.alpha = {c: 1.0 for c in self.configs}
        self.beta_param = {c: 1.0 for c in self.configs}
        self.impressions = {c: 0 for c in self.configs}
        self.conversions = {c: 0 for c in self.configs}
    
    def select_config(self) -> tuple:
        """Thompson Sampling: 从后验Beta分布采样，选择最高期望配置"""
        samples = {}
        for config in self.configs:
            # 从Beta后验采样
            samples[config] = np.random.beta(
                self.alpha[config], 
                self.beta_param[config]
            )
        return max(samples, key=samples.get)
    
    def update(self, config: tuple, converted: bool):
        """观测结果后更新Beta分布参数"""
        self.impressions[config] += 1
        if converted:
            self.alpha[config] += 1
            self.conversions[config] += 1
        else:
            self.beta_param[config] += 1
    
    def get_conversion_rate(self, config: tuple) -> Dict:
        """获取配置的转化率估计及置信区间"""
        a = self.alpha[config]
        b = self.beta_param[config]
        mean = a / (a + b)
        # 95% HDI (Highest Density Interval)
        from scipy.stats import beta as beta_dist
        ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)
        return {
            'mean': mean,
            'ci_95': (ci_low, ci_high),
            'impressions': self.impressions[config],
            'conversions': self.conversions[config]
        }
    
    def get_best_config(self) -> Tuple[tuple, Dict]:
        """返回当前最优配置"""
        best = max(self.configs, 
                   key=lambda c: self.alpha[c] / (self.alpha[c] + self.beta_param[c]))
        stats = self.get_conversion_rate(best)
        return best, stats
    
    def get_ranking(self, top_n: int = 5) -> List:
        """返回Top N配置排名"""
        ranked = sorted(
            self.configs,
            key=lambda c: self.alpha[c] / (self.alpha[c] + self.beta_param[c]),
            reverse=True
        )[:top_n]
        return [(c, self.get_conversion_rate(c)) for c in ranked]


class ShopifyUTMPersonalizer:
    """
    基于UTM参数的落地页动态内容个性化
    不同流量来源展示不同版本
    """
    
    SOURCE_CONFIGS = {
        'tiktok': {
            'headline': '夜晚安心，妈妈放心睡',
            'trust_signal': '前30天无理由退换',
            'urgency': '限时优惠，今日下单立减$20',
            'hero_style': 'lifestyle'
        },
        'google_brand': {
            'headline': '专业级婴儿监视器，医院同款',
            'trust_signal': '儿科医生推荐 | FDA认证',
            'urgency': '',
            'hero_style': 'product_specs'
        },
        'email_loyalty': {
            'headline': '老客专属升级优惠',
            'trust_signal': '积分3倍加速，会员专属价',
            'urgency': '48小时专属优惠',
            'hero_style': 'upgrade'
        }
    }
    
    def get_config(self, utm_source: str, utm_campaign: str = '') -> Dict:
        """根据UTM参数返回个性化配置"""
        source_key = utm_source.lower()
        if 'email' in source_key or 'loyalty' in utm_campaign.lower():
            return self.SOURCE_CONFIGS['email_loyalty']
        elif 'tiktok' in source_key or 'instagram' in source_key:
            return self.SOURCE_CONFIGS['tiktok']
        elif 'google' in source_key:
            return self.SOURCE_CONFIGS['google_brand']
        else:
            return self.SOURCE_CONFIGS['tiktok']  # 默认冷流量配置


# 使用示例
def run_cro_simulation():
    """模拟1000次落地页访问的贝叶斯优化过程"""
    
    # 定义落地页元素
    elements = {
        'headline': ['守护宝宝每一刻', '夜晚安心妈妈放心睡', '1080P实时监控'],
        'hero_image': ['product', 'lifestyle', 'baby_closeup'],
        'cta': ['立即购买', '限时优惠', '免费试用30天'],
        'social_proof': ['star_rating', 'media_endorsement', 'user_count']
    }
    
    # 模拟真实转化率（实际中由A/B平台记录）
    true_rates = {}
    random.seed(42)
    for config in LandingPageBayesianOptimizer(elements).configs:
        # 模拟：场景图+免费试用+用户数量组合最优
        boost = 0.02 if 'lifestyle' in config and '免费试用30天' in config else 0
        boost += 0.01 if 'user_count' in config else 0
        true_rates[config] = max(0.01, min(0.08, random.gauss(0.025, 0.01) + boost))
    
    optimizer = LandingPageBayesianOptimizer(elements)
    
    print("=== 贝叶斯优化落地页CRO 模拟 ===\n")
    
    # 模拟1000次访问
    for i in range(1000):
        config = optimizer.select_config()
        # 根据真实转化率决定是否转化
        converted = random.random() < true_rates[config]
        optimizer.update(config, converted)
    
    # 输出结果
    best_config, best_stats = optimizer.get_best_config()
    config_dict = dict(zip(elements.keys(), best_config))
    
    print(f"最优配置：")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print(f"\n预测转化率: {best_stats['mean']:.3f}")
    print(f"95% 置信区间: [{best_stats['ci_95'][0]:.3f}, {best_stats['ci_95'][1]:.3f}]")
    print(f"观测样本数: {best_stats['impressions']}")
    
    print("\nTop 3 配置排名：")
    for rank, (config, stats) in enumerate(optimizer.get_ranking(3), 1):
        cfg = dict(zip(elements.keys(), config))
        print(f"  #{rank}: CVR={stats['mean']:.3f} | {cfg['headline']} + {cfg['cta']}")
    
    # UTM个性化示例
    print("\n=== UTM流量个性化配置 ===")
    personalizer = ShopifyUTMPersonalizer()
    for source in ['tiktok', 'google_brand', 'email_loyalty']:
        cfg = personalizer.get_config(source)
        print(f"  {source}: {cfg['headline']} | CTA紧迫感: {cfg['urgency'][:20] if cfg['urgency'] else '无'}")
    
    return optimizer


if __name__ == '__main__':
    optimizer = run_cro_simulation()
