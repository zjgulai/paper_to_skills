---
title: Social Engineering Attack Detection — 社会工程攻击检测钓鱼邮件/虚假供应商识别
doc_type: knowledge
module: 19-风控反欺诈
topic: social-engineering-attack-detection
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Social-Engineering-Attack-Detection

## ① 算法原理（≤300字）

**核心问题**：母婴跨境电商卖家是社会工程攻击的高价值目标——攻击者伪装成 Amazon 官方、物流公司、税务机构发送钓鱼邮件，或伪装成供应商骗取预付款/样品费。每次成功攻击的损失从数千到数十万美元不等。

**钓鱼邮件检测特征**：

1. **发件人域名异常**：
   - 合法 Amazon 邮件：@amazon.com / @marketplace.amazon.com
   - 钓鱼变体：@amazon-notice.com / @amaz0n.com（数字替换字母）
   - 域名注册时间 < 30 天（新注册域名高风险）

2. **内容特征**：
   - 紧迫性词汇：「immediate action」「account suspended」「within 24 hours」
   - 外链指向非官方域名
   - 要求通过邮件提供账号密码/银行信息

3. **虚假供应商检测**：
   - 价格异常低（> 30% 低于市场价）
   - 公司成立时间 < 6 个月
   - 要求只接受电汇 / 加密货币 / 非正规支付方式
   - 沟通英语错误率异常高（非母语特征）

**机器学习分类**：基于上述特征训练逻辑回归或朴素贝叶斯分类器，对收到的邮件/供应商信息给出欺诈概率分。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：运营人员收到「来自 Amazon」的邮件，要求 48 小时内验证账号（「Account Verification Required」），链接指向 amaz0n-verification.com。同月，新供应商报价奶瓶模具低于市场价 40%，要求先付 5 万元样品费。

**数据要求**：邮件头信息（发件人域名、DMARC/SPF 状态）、邮件内容文本，供应商信息（公司名、成立时间、支付要求）。

**检测应用**：邮件欺诈评分 92 分（域名新注册 + 紧迫词 + 非官方链接），供应商欺诈评分 78 分（价格异常 + 可疑支付）。两起均为钓鱼/诈骗，运营人员拒绝操作，避免损失。

**量化产出**：每次成功识别钓鱼攻击平均避免损失 5-50 万元，年化减少社会工程攻击损失 **50-100 万元**。

## ③ 代码模板

```python
import re
import numpy as np

# 合法邮件域名白名单
LEGITIMATE_DOMAINS = {
    'amazon': ['amazon.com', 'marketplace.amazon.com', 'seller.amazon.com',
               'amazon.co.uk', 'amazon.de', 'amazon.co.jp'],
    'ebay': ['ebay.com', 'ebay.co.uk'],
    'paypal': ['paypal.com', 'paypal.me'],
}

# 钓鱼指标词汇
PHISHING_KEYWORDS = [
    'immediate action', 'account suspended', 'verify now', 'click here',
    'within 24 hours', 'urgent', 'account disabled', 'payment failed',
    'confirm your identity', 'update payment', 'security alert'
]

# 供应商高风险词汇
SUPPLIER_RED_FLAGS = [
    'wire transfer only', 'cryptocurrency', 'western union', 'moneygram',
    'no refund', 'advance payment required', 'factory price',
]

def analyze_email_phishing(
    sender_domain: str,
    subject: str,
    body: str,
    registration_age_days: int = None
) -> dict:
    """邮件钓鱼检测"""
    score = 0
    signals = []

    # 域名检测
    is_legit_domain = any(
        sender_domain.endswith(d)
        for domains in LEGITIMATE_DOMAINS.values()
        for d in domains
    )

    # 检测混淆域名（数字替换字母）
    leet_pattern = re.sub(r'[0-9]', lambda m: {'0': 'o', '1': 'i', '3': 'e', '4': 'a'}.get(m.group(), m.group()), sender_domain)
    is_leet = leet_pattern != sender_domain and any(
        leet_pattern.endswith(d) for domains in LEGITIMATE_DOMAINS.values() for d in domains
    )

    if not is_legit_domain:
        score += 30
        signals.append('非官方发件域名')
    if is_leet:
        score += 40
        signals.append('数字混淆域名（疑似仿冒官方）')
    if registration_age_days is not None and registration_age_days < 30:
        score += 25
        signals.append(f'域名注册仅 {registration_age_days} 天')

    # 内容关键词
    full_text = (subject + ' ' + body).lower()
    matched_keywords = [kw for kw in PHISHING_KEYWORDS if kw in full_text]
    score += len(matched_keywords) * 5
    if matched_keywords:
        signals.append(f'钓鱼关键词: {matched_keywords[:3]}')

    # 外链检测（简化：检测非官方 URL）
    urls = re.findall(r'https?://([^/\s]+)', body)
    suspicious_urls = [u for u in urls if not any(u.endswith(d) for domains in LEGITIMATE_DOMAINS.values() for d in domains)]
    if suspicious_urls:
        score += 20
        signals.append(f'可疑外链: {suspicious_urls[:2]}')

    score = min(100, score)
    return {
        'phishing_score': score,
        'is_phishing': score > 60,
        'signals': signals,
        'risk_level': 'HIGH' if score > 70 else 'MEDIUM' if score > 40 else 'LOW'
    }

def analyze_supplier_risk(
    price_vs_market: float,  # 负值=低于市场价，如 -0.35 = 低 35%
    company_age_months: int,
    payment_methods: list,
    has_business_license: bool = True,
    communication_error_rate: float = 0.0
) -> dict:
    """供应商欺诈风险评估"""
    score = 0
    signals = []

    if price_vs_market < -0.30:
        score += 30
        signals.append(f'价格低于市场 {abs(price_vs_market):.0%}（异常低价）')
    if company_age_months < 6:
        score += 25
        signals.append(f'公司成立仅 {company_age_months} 个月')
    if not has_business_license:
        score += 35
        signals.append('无营业执照')

    risky_payments = [m for m in payment_methods if any(rf in m.lower() for rf in ['wire', 'crypto', 'western', 'western union', 'moneygram'])]
    if risky_payments:
        score += 20
        signals.append(f'高风险支付方式: {risky_payments}')
    if communication_error_rate > 0.15:
        score += 10
        signals.append(f'沟通错误率 {communication_error_rate:.0%}（疑似非原厂）')

    score = min(100, score)
    return {
        'fraud_score': score,
        'is_suspicious': score > 50,
        'signals': signals,
        'risk_level': 'HIGH' if score > 70 else 'MEDIUM' if score > 40 else 'LOW'
    }

# 测试
# 钓鱼邮件测试
email_result = analyze_email_phishing(
    sender_domain='amaz0n-verification.com',
    subject='Urgent: Account Verification Required Within 24 Hours',
    body='Your seller account requires immediate action. Click here to verify: https://amaz0n-verification.com/verify',
    registration_age_days=7
)
assert email_result['is_phishing'], f"应识别为钓鱼，分数: {email_result['phishing_score']}"

# 可疑供应商测试
supplier_result = analyze_supplier_risk(
    price_vs_market=-0.40,
    company_age_months=3,
    payment_methods=['wire transfer only', 'no credit card'],
    has_business_license=False,
    communication_error_rate=0.25
)
assert supplier_result['is_suspicious'], f"应识别为可疑供应商，分数: {supplier_result['fraud_score']}"

print(f"邮件欺诈评分: {email_result['phishing_score']} - {email_result['risk_level']}")
print(f"钓鱼信号: {email_result['signals']}")
print(f"\n供应商欺诈评分: {supplier_result['fraud_score']} - {supplier_result['risk_level']}")
print(f"风险信号: {supplier_result['signals']}")
print("[✓] Social-Engineering-Attack-Detection 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Identity-Fraud-Detection]]
- 前置技能：[[Skill-Transaction-Anomaly-Detection]]
- 延伸技能：[[Skill-Agent-Payment-Security-Red-Team]]
- 延伸技能：[[Skill-MAS-Adversarial-Defense]]
- 可组合：[[Skill-AgentTrust-Runtime-Safety-Interception]]
- 可组合：[[Skill-Cross-Platform-Account-Linkage-Risk]]

## ⑤ 商业价值评估

- **ROI量化**: 每次识别钓鱼/诈骗避免损失 5-50 万元，年化减少损失 50-100 万元
- **实施难度**: ⭐（规则库建设简单，主要是人员意识培训）
- **优先级**: ⭐⭐⭐⭐⭐（每个团队的基础安全防线，零技术门槛）
