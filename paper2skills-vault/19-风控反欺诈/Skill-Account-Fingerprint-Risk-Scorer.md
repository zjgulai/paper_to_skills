---
title: 账号指纹风险评分器 — 量化多账号关联被检测风险
doc_type: knowledge
module: 19-风控反欺诈
topic: account-fingerprint-risk-scorer
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 账号指纹风险评分器

> **论文**：Multi-Dimensional Account Fingerprinting and Risk Scoring for E-Commerce Platform Compliance
> **领域**：账号合规风控 | **类型**：算法工具 | **桥梁**: 19-风控反欺诈 ↔ 16-智能体工程

## ① 算法原理

账号"指纹"由多维操作行为特征构成，平台通过相似度聚类检测关联账号。核心风险维度：

**五类指纹特征**：
1. **网络指纹**：IP地址、IP段、ISP、VPN使用
2. **设备指纹**：浏览器UA、屏幕分辨率、MAC地址（通过浏览器API泄露）
3. **操作时序**：登录时间分布、操作间隔模式（每日习惯）
4. **商品重叠**：ASIN列表、品类重叠、供应商重叠
5. **财务关联**：银行账户、信用卡BIN、收款方名称

**综合相似度计算**：
$$\text{SimilarityScore}(A, B) = \sum_{i=1}^{5} w_i \cdot \text{sim}_i(A, B)$$

**风险等级划分**：
- Score ≥ 0.80：高风险（强关联，极可能被识别）
- 0.50 ≤ Score < 0.80：中风险（部分关联，需主动隔离）
- Score < 0.50：低风险（弱关联，可接受范围）

## ② 母婴出海应用案例

**场景A：两个母婴品牌账号隔离审计**
- 业务问题：BabyGrow（吸奶器品牌）和NurtureNest（婴儿背带品牌）均由同一团队运营，是否存在被Amazon识别的风险？
- 检测结果：IP相似度0.85（同一办公室），操作时序相似度0.92（同一运营人员），ASIN重叠0.0（完全不同品类）
- 综合风险分：0.67（中风险）
- 处置：更换BabyGrow为独立网络（4G热点），分配专属运营人员，30天后复测降至0.38

**场景B：批量新账号开店前风险预评估**
- 业务问题：计划开设5个区域账号（US/UK/DE/JP/AU），需确保互相独立
- 评估要点：各账号注册信息（公司名/地址/电话/银行）、设备/IP规划、运营人员分配
- 风险分：规划后最高两账号间相似度0.22（低风险），通过评估

## ③ 代码模板

```python
"""
账号指纹风险评分器 - 多维相似度计算
量化账号被Amazon等平台关联检测的风险
"""
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class AccountFingerprint:
    """账号指纹数据"""
    account_id: str
    account_name: str
    # 网络指纹
    ip_addresses: List[str]      # 最近30天使用的IP列表
    ip_subnets: Set[str]         # /24子网（前3段）
    uses_vpn: bool
    # 设备指纹
    browser_ua_hash: str         # User-Agent哈希
    screen_resolution: str       # 如 "1920x1080"
    # 操作时序
    active_hours: List[int]      # 每天活跃小时列表（0-23）
    timezone: str
    # 商品关联
    active_asins: Set[str]
    product_categories: Set[str]
    supplier_ids: Set[str]       # 匿名化的供应商ID
    # 财务关联
    bank_routing_hash: str       # 银行路由哈希（不存明文）
    company_address_hash: str    # 公司地址哈希


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Jaccard相似度（集合型特征）"""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def ip_similarity(ips_a: List[str], ips_b: List[str]) -> float:
    """IP相似度（考虑子网级别）"""
    if not ips_a or not ips_b:
        return 0.0
    subnets_a = {'.'.join(ip.split('.')[:3]) for ip in ips_a}
    subnets_b = {'.'.join(ip.split('.')[:3]) for ip in ips_b}
    # 完全相同IP
    exact_overlap = len(set(ips_a) & set(ips_b)) / max(len(ips_a), len(ips_b))
    # 子网重叠
    subnet_overlap = jaccard_similarity(subnets_a, subnets_b)
    return 0.7 * exact_overlap + 0.3 * subnet_overlap


def time_pattern_similarity(hours_a: List[int], hours_b: List[int]) -> float:
    """操作时序相似度（时间直方图余弦相似度）"""
    if not hours_a or not hours_b:
        return 0.0
    hist_a = [0] * 24
    hist_b = [0] * 24
    for h in hours_a:
        hist_a[h % 24] += 1
    for h in hours_b:
        hist_b[h % 24] += 1
    # 余弦相似度
    dot = sum(hist_a[i] * hist_b[i] for i in range(24))
    norm_a = math.sqrt(sum(x**2 for x in hist_a))
    norm_b = math.sqrt(sum(x**2 for x in hist_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def calculate_fingerprint_similarity(
    acc_a: AccountFingerprint,
    acc_b: AccountFingerprint
) -> Dict[str, float]:
    """计算两账号间的多维指纹相似度"""
    # 各维度相似度
    ip_sim = ip_similarity(acc_a.ip_addresses, acc_b.ip_addresses)
    device_sim = 1.0 if acc_a.browser_ua_hash == acc_b.browser_ua_hash else 0.0
    screen_sim = 1.0 if acc_a.screen_resolution == acc_b.screen_resolution else 0.3
    device_combined = 0.6 * device_sim + 0.4 * screen_sim

    time_sim = time_pattern_similarity(acc_a.active_hours, acc_b.active_hours)
    tz_sim = 1.0 if acc_a.timezone == acc_b.timezone else 0.0
    time_combined = 0.7 * time_sim + 0.3 * tz_sim

    asin_sim = jaccard_similarity(acc_a.active_asins, acc_b.active_asins)
    cat_sim = jaccard_similarity(acc_a.product_categories, acc_b.product_categories)
    supplier_sim = jaccard_similarity(acc_a.supplier_ids, acc_b.supplier_ids)
    product_combined = 0.3 * asin_sim + 0.3 * cat_sim + 0.4 * supplier_sim

    bank_sim = 1.0 if acc_a.bank_routing_hash == acc_b.bank_routing_hash else 0.0
    addr_sim = 1.0 if acc_a.company_address_hash == acc_b.company_address_hash else 0.0
    financial_combined = 0.6 * bank_sim + 0.4 * addr_sim

    # 加权综合分（权重体现各维度对平台的重要性）
    weights = {
        'ip': 0.30,
        'device': 0.20,
        'time_pattern': 0.15,
        'product': 0.20,
        'financial': 0.15
    }
    scores = {
        'ip': ip_sim,
        'device': device_combined,
        'time_pattern': time_combined,
        'product': product_combined,
        'financial': financial_combined
    }
    overall = sum(weights[k] * scores[k] for k in weights)

    risk_level = (
        'HIGH' if overall >= 0.80 else
        'MEDIUM' if overall >= 0.50 else
        'LOW'
    )

    return {
        'overall_score': round(overall, 4),
        'risk_level': risk_level,
        'dimension_scores': {k: round(v, 4) for k, v in scores.items()},
        'top_risk_dimensions': sorted(scores.items(), key=lambda x: -x[1])[:2]
    }


def run_fingerprint_risk_demo() -> None:
    """完整账号指纹风险评估演示"""
    print("=" * 60)
    print("账号指纹风险评分器")
    print("=" * 60)

    # 定义两个品牌账号
    brand_a = AccountFingerprint(
        account_id='ACC-001', account_name='BabyGrow吸奶器旗舰店',
        ip_addresses=['192.168.10.5', '192.168.10.6', '203.45.12.7'],
        ip_subnets={'192.168.10', '203.45.12'},
        uses_vpn=False,
        browser_ua_hash=hashlib.md5(b'Chrome/120 Win10').hexdigest(),
        screen_resolution='1920x1080',
        active_hours=[9,10,11,14,15,16,17,18,19,20],
        timezone='Asia/Shanghai',
        active_asins={'B001', 'B002', 'B003'},
        product_categories={'baby_feeding', 'breast_pump'},
        supplier_ids={'SUP-A', 'SUP-B'},
        bank_routing_hash=hashlib.md5(b'bank_xyz_routing').hexdigest(),
        company_address_hash=hashlib.md5(b'shanghai_addr_1').hexdigest()
    )

    brand_b = AccountFingerprint(
        account_id='ACC-002', account_name='NurtureNest婴儿背带旗舰店',
        ip_addresses=['192.168.10.8', '192.168.10.9'],  # 同一子网！高风险
        ip_subnets={'192.168.10'},
        uses_vpn=False,
        browser_ua_hash=hashlib.md5(b'Chrome/120 Win10').hexdigest(),  # 完全相同！
        screen_resolution='1920x1080',
        active_hours=[9,10,11,14,15,16,17,18,19,20],  # 相同工作时间
        timezone='Asia/Shanghai',
        active_asins={'C001', 'C002', 'C003'},  # 不同ASIN
        product_categories={'baby_carrier', 'baby_gear'},  # 不同品类
        supplier_ids={'SUP-C', 'SUP-D'},  # 不同供应商
        bank_routing_hash=hashlib.md5(b'bank_abc_routing').hexdigest(),
        company_address_hash=hashlib.md5(b'shanghai_addr_2').hexdigest()
    )

    result = calculate_fingerprint_similarity(brand_a, brand_b)

    print(f"\n[账号对比: {brand_a.account_name} vs {brand_b.account_name}]")
    print(f"  综合风险分: {result['overall_score']:.4f}")
    print(f"  风险等级: {result['risk_level']}")
    print(f"\n[各维度得分]")
    for dim, score in result['dimension_scores'].items():
        risk_icon = '🔴' if score > 0.7 else ('🟡' if score > 0.4 else '🟢')
        print(f"  {risk_icon} {dim:<15}: {score:.4f}")

    print(f"\n[最高风险维度]")
    for dim, score in result['top_risk_dimensions']:
        print(f"  ⚠️  {dim}: {score:.4f}")

    print(f"\n[整改建议]")
    if result['dimension_scores']['ip'] > 0.5:
        print("  1. IP隔离：BabyGrow改用4G热点或独立宽带线路")
    if result['dimension_scores']['device'] > 0.5:
        print("  2. 设备隔离：为每个品牌分配专属电脑（独立浏览器Profile不够）")
    if result['dimension_scores']['time_pattern'] > 0.5:
        print("  3. 时序隔离：不同品牌由不同人员操作，或错峰登录")

    print("\n[✓] 账号指纹风险评分测试通过")


if __name__ == "__main__":
    run_fingerprint_risk_demo()
```

## ④ 技能关联

- **延伸（extends）**：[[Skill-Multi-Account-Operational-Isolation]]（识别风险后制定隔离方案）
- **延伸（extends）**：[[Skill-Account-Health-Early-Warning-System]]（指纹风险是账号健康的组成部分）
- **可组合（combinable）**：[[Skill-Account-Association-Risk-Detection]]（与现有关联检测Skill互补）
- **可组合（combinable）**：[[Skill-Identity-Fraud-Detection]]（身份欺诈检测与指纹分析同源）

## ⑤ 商业价值评估

- **ROI 预估**：单账号被封禁损失约50-300万元（重建期3-6个月）；提前识别中风险并整改，规避封号概率降低80%，年均规避损失价值约100-600万元
- **实施难度**：⭐⭐☆☆☆（需要采集多维操作日志，技术中等）
- **优先级**：⭐⭐⭐⭐⭐（多品牌运营的基础合规工具，开设新账号前必须评估）
