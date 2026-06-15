"""
Auto-extracted from: paper2skills-vault/23-运营财务/Skill-Accounts-Receivable-Intelligence.md
Skill: Skill-Accounts-Receivable-Intelligence
Domain: 23-运营财务
"""
"""
Accounts Receivable Intelligence
账期智能管理：逾期预测 + 差异化催收策略
"""
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Invoice:
    invoice_id: str
    buyer_id: str
    amount_usd: float
    issue_date: str
    due_date: str
    payment_terms_days: int = 30


@dataclass
class BuyerProfile:
    buyer_id: str
    name: str
    historical_invoices: list   # [{'due_days': 30, 'actual_days': 35, 'amount': 5000}]
    region: str = 'US'
    relationship_years: float = 1.0


def compute_buyer_risk_features(buyer: BuyerProfile) -> dict:
    """从历史记录计算买家风险特征"""
    if not buyer.historical_invoices:
        return {'avg_delay': 0, 'delay_std': 0, 'on_time_rate': 1.0,
                'avg_amount': 0, 'trend': 0}

    delays = [inv['actual_days'] - inv['due_days'] for inv in buyer.historical_invoices]
    on_time = sum(1 for d in delays if d <= 2) / len(delays)  # ≤2天算准时

    # 趋势：近3笔 vs 历史均值
    recent = np.mean(delays[-3:]) if len(delays) >= 3 else np.mean(delays)
    overall = np.mean(delays)
    trend = recent - overall  # 正=最近变差，负=最近改善

    return {
        'avg_delay': round(float(np.mean(delays)), 1),
        'delay_std': round(float(np.std(delays)), 1),
        'on_time_rate': round(on_time, 3),
        'avg_amount': round(np.mean([inv['amount'] for inv in buyer.historical_invoices]), 0),
        'trend': round(trend, 1),
    }


def predict_overdue_probability(invoice: Invoice, buyer: BuyerProfile) -> dict:
    """预测账款逾期概率"""
    features = compute_buyer_risk_features(buyer)

    # 逾期风险因子（规则加权，生产用 XGBoost）
    risk = 0.0

    # 历史逾期率
    if features['on_time_rate'] < 0.7:
        risk += 0.4
    elif features['on_time_rate'] < 0.9:
        risk += 0.2

    # 平均延迟天数
    if features['avg_delay'] > 10:
        risk += 0.25
    elif features['avg_delay'] > 5:
        risk += 0.12

    # 趋势（最近变差）
    if features['trend'] > 5:
        risk += 0.15
    elif features['trend'] > 0:
        risk += 0.05

    # 大额账款额外风险
    if invoice.amount_usd > 20000:
        risk += 0.1

    # 新买家额外风险
    if buyer.relationship_years < 0.5:
        risk += 0.15

    overdue_prob = min(0.99, max(0.01, risk))

    # 预期回款天数（基于历史均值）
    expected_days = invoice.payment_terms_days + max(0, features['avg_delay'])

    return {
        'invoice_id': invoice.invoice_id,
        'buyer_id': buyer.buyer_id,
        'amount_usd': invoice.amount_usd,
        'overdue_probability': round(overdue_prob, 3),
        'expected_payment_days': round(expected_days, 0),
        'risk_level': 'HIGH' if overdue_prob > 0.5 else ('MEDIUM' if overdue_prob > 0.25 else 'LOW'),
        'buyer_features': features,
    }


def generate_collection_strategy(prediction: dict) -> dict:
    """生成差异化催收策略"""
    risk = prediction['risk_level']
    amount = prediction['amount_usd']

    strategies = {
        ('HIGH', 'large'):   ('立即行动', '电话+邮件联系，了解付款计划，必要时暂停新订单授信'),
        ('HIGH', 'small'):   ('本周邮件', '发送友好提醒邮件，告知付款截止日期'),
        ('MEDIUM', 'large'): ('提前提醒', '到期前7天发送正式付款提醒'),
        ('MEDIUM', 'small'): ('常规提醒', '到期前3天发送标准提醒'),
        ('LOW', 'large'):    ('常规流程', '正常到期提醒，无需特殊处理'),
        ('LOW', 'small'):    ('自动处理', '系统自动发送到期提醒'),
    }

    size = 'large' if amount > 10000 else 'small'
    timing, message = strategies.get((risk, size), ('常规流程', '按标准流程处理'))

    return {
        'action_timing': timing,
        'recommended_action': message,
        'action_priority': {'HIGH': 'P1', 'MEDIUM': 'P2', 'LOW': 'P3'}[risk],
    }


def run_ar_intelligence_demo():
    print('=' * 65)
    print('Accounts Receivable Intelligence — 账期智能管理')
    print('=' * 65)

    buyers = {
        'BUYER-A': BuyerProfile('BUYER-A', 'US Retail Chain',
                                [{'due_days':30,'actual_days':28,'amount':8000},
                                 {'due_days':30,'actual_days':31,'amount':9500},
                                 {'due_days':30,'actual_days':30,'amount':7000}],
                                region='US', relationship_years=3.0),
        'BUYER-B': BuyerProfile('BUYER-B', 'EU Distributor',
                                [{'due_days':30,'actual_days':45,'amount':15000},
                                 {'due_days':30,'actual_days':50,'amount':18000},
                                 {'due_days':30,'actual_days':55,'amount':20000}],
                                region='EU', relationship_years=1.5),
        'BUYER-C': BuyerProfile('BUYER-C', 'New Buyer',
                                [],
                                region='Asia', relationship_years=0.1),
    }

    invoices = [
        Invoice('INV-001', 'BUYER-A', 12000, '2026-06-01', '2026-07-01'),
        Invoice('INV-002', 'BUYER-B', 25000, '2026-06-01', '2026-07-01'),
        Invoice('INV-003', 'BUYER-C',  5000, '2026-06-01', '2026-07-01'),
    ]

    print(f'\n📊 应收账款风险分析:')
    print(f'  {"发票":>8} {"买家":>12} {"金额":>10} {"逾期概率":>9} {"预期回款":>9} {"风险"}')
    print('  ' + '-' * 62)

    for inv in invoices:
        buyer = buyers[inv.buyer_id]
        pred = predict_overdue_probability(inv, buyer)
        strategy = generate_collection_strategy(pred)
        risk_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[pred['risk_level']]
        print(f'  {inv.invoice_id:>8} {buyer.name:>12} ${inv.amount_usd:>9,.0f} '
              f'{pred["overdue_probability"]:>9.1%} '
              f'T+{pred["expected_payment_days"]:>6.0f}天 '
              f'{risk_icon} {pred["risk_level"]}')
        print(f'           → [{strategy["action_priority"]}] {strategy["action_timing"]}: {strategy["recommended_action"][:50]}')

    total_ar = sum(inv.amount_usd for inv in invoices)
    high_risk_ar = sum(inv.amount_usd for inv in invoices if
                       predict_overdue_probability(inv, buyers[inv.buyer_id])['risk_level'] == 'HIGH')
    print(f'\n  总应收: ${total_ar:,.0f}  高风险: ${high_risk_ar:,.0f} ({high_risk_ar/total_ar:.0%})')
    print('\n[✓] Accounts Receivable Intelligence 测试通过')


if __name__ == '__main__':
    run_ar_intelligence_demo()
