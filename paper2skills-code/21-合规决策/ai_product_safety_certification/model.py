"""
Auto-extracted from: paper2skills-vault/21-合规决策/Skill-AI-Product-Safety-Certification.md
Skill: Skill-AI-Product-Safety-Certification
Domain: 21-合规决策
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class Market(Enum):
    US = "美国"
    EU = "欧盟"
    UK = "英国"
    AU = "澳大利亚"
    CA = "加拿大"

@dataclass
class CertificationRequirement:
    standard: str           # e.g., "ASTM F963"
    description: str
    cost_usd: float
    lead_time_weeks: float
    lab_options: List[str]  # e.g., ["SGS", "BV", "TÜV"]
    can_share_with: List[str] = field(default_factory=list)  # 可共享报告的其他认证

@dataclass
class ProductProfile:
    name: str
    category: str           # "婴儿玩具", "儿童家具", etc.
    materials: List[str]
    age_group_months: tuple # (min, max)
    has_electronics: bool
    has_bluetooth: bool
    target_markets: List[Market]

# 认证规则知识库（简化版）
CERTIFICATION_RULES: Dict[str, Dict] = {
    "婴儿玩具": {
        Market.US: [
            CertificationRequirement(
                "ASTM F963", "美国玩具安全标准", 800, 4, ["SGS", "BV", "Intertek"],
                can_share_with=[]
            ),
            CertificationRequirement(
                "CPSIA Lead+Phthalates", "铅和邻苯二甲酸盐检测", 300, 2, ["SGS", "BV"],
                can_share_with=["EN 71-3"]  # 与欧盟化学测试共用
            ),
        ],
        Market.EU: [
            CertificationRequirement(
                "EN 71-1", "机械和物理特性", 400, 3, ["TÜV", "SGS", "Intertek"],
                can_share_with=["AS/NZS 8124-1"]
            ),
            CertificationRequirement(
                "EN 71-2", "燃烧性", 250, 2, ["TÜV", "SGS"],
                can_share_with=[]
            ),
            CertificationRequirement(
                "EN 71-3", "特定元素迁移", 350, 3, ["TÜV", "SGS"],
                can_share_with=["CPSIA Lead+Phthalates"]
            ),
        ],
        Market.AU: [
            CertificationRequirement(
                "AS/NZS 8124-1", "澳洲玩具安全-机械", 350, 3, ["SAI Global"],
                can_share_with=["EN 71-1"]  # 可复用EN测试数据
            ),
        ],
    },
    "婴儿摇椅": {
        Market.US: [
            CertificationRequirement("ASTM F2194", "婴儿座椅安全", 900, 5, ["SGS", "Intertek"]),
        ],
        Market.EU: [
            CertificationRequirement("EN 1930", "儿童安全屏障", 700, 5, ["TÜV", "SGS"]),
        ],
    }
}

ELECTRONICS_CERT = {
    Market.US: CertificationRequirement("FCC Part 15", "无线设备认证", 400, 3, ["SGS", "UL"]),
    Market.EU: CertificationRequirement("RED Directive", "无线电设备指令", 500, 4, ["TÜV"]),
}


class CertificationPathPlanner:
    """
    AI认证路径规划器：找到覆盖所有目标市场要求的最低成本认证组合
    """
    
    def __init__(self, rules: Dict = CERTIFICATION_RULES):
        self.rules = rules
    
    def plan(self, product: ProductProfile) -> Dict:
        """规划认证路径，识别共享测试机会"""
        required = []
        
        # 收集所有市场的认证要求
        category_rules = self.rules.get(product.category, {})
        for market in product.target_markets:
            market_reqs = category_rules.get(market, [])
            required.extend(market_reqs)
        
        # 电子产品额外要求
        if product.has_electronics or product.has_bluetooth:
            for market in product.target_markets:
                if market in ELECTRONICS_CERT:
                    required.append(ELECTRONICS_CERT[market])
        
        # 识别可以共享报告的认证（去重优化）
        shared_pairs = []
        seen_standards = set()
        optimized = []
        
        for req in required:
            if req.standard not in seen_standards:
                optimized.append(req)
                seen_standards.add(req.standard)
                for shared in req.can_share_with:
                    if shared in seen_standards:
                        shared_pairs.append((req.standard, shared))
        
        # 计算费用和时间
        total_cost = sum(r.cost_usd for r in optimized)
        max_lead_time = max((r.lead_time_weeks for r in optimized), default=0)
        
        # 推荐实验室（选每项认证中评分最高的）
        lab_recommendations = {}
        preferred_labs = {"SGS": 3, "TÜV": 3, "BV": 2, "Intertek": 2, "SAI Global": 2, "UL": 2}
        for req in optimized:
            best_lab = max(req.lab_options, key=lambda l: preferred_labs.get(l, 1))
            lab_recommendations[req.standard] = best_lab
        
        return {
            "product": product.name,
            "target_markets": [m.value for m in product.target_markets],
            "certifications": [
                {
                    "standard": r.standard,
                    "description": r.description,
                    "cost_usd": r.cost_usd,
                    "lead_time_weeks": r.lead_time_weeks,
                    "recommended_lab": lab_recommendations.get(r.standard, r.lab_options[0])
                }
                for r in optimized
            ],
            "shared_tests": shared_pairs,
            "total_cost_usd": total_cost,
            "total_lead_time_weeks": max_lead_time,
            "savings_from_sharing": len(shared_pairs) * 300  # 估算每次共享节省$300
        }
    
    def generate_cpc_draft(self, product: ProductProfile, plan: Dict) -> str:
        """生成美国CPC合规声明草稿"""
        standards = [c["standard"] for c in plan["certifications"]
                     if Market.US in product.target_markets]
        
        return f"""
CERTIFICATE OF PRODUCT COMPLIANCE (CPC)
========================================
Product Name: {product.name}
Category: {product.category}
Intended Age Group: {product.age_group_months[0]}-{product.age_group_months[1]} months

The manufacturer/importer certifies that this product complies with:
{chr(10).join(f'  • {s}' for s in standards)}

Test Laboratory: [Fill in after receiving reports]
Test Report Numbers: [Fill in after receiving reports]
Date of Certification: [Fill in]
Manufacturer: [Your Company Name]
Importer: [If applicable]

This certificate is based on third-party testing conducted by a CPSC-accepted lab.
"""


# 使用示例
if __name__ == '__main__':
    # 创建产品档案
    product = ProductProfile(
        name="SmartSoothe婴儿电动摇椅",
        category="婴儿摇椅",
        materials=["钢架", "聚酯棉", "ABS", "蓝牙模块"],
        age_group_months=(0, 12),
        has_electronics=True,
        has_bluetooth=True,
        target_markets=[Market.US, Market.EU, Market.AU]
    )
    
    planner = CertificationPathPlanner()
    plan = planner.plan(product)
    
    print("=== AI认证路径规划报告 ===\n")
    print(f"产品: {plan['product']}")
    print(f"目标市场: {', '.join(plan['target_markets'])}")
    print(f"\n认证清单（{len(plan['certifications'])}项）:")
    
    for cert in plan['certifications']:
        print(f"  ✓ {cert['standard']}: ${cert['cost_usd']} | {cert['lead_time_weeks']}周 | 推荐实验室: {cert['recommended_lab']}")
    
    if plan['shared_tests']:
        print(f"\n共享测试优化（节省约${plan['savings_from_sharing']}）:")
        for a, b in plan['shared_tests']:
            print(f"  {a} ↔ {b} 可共享测试数据")
    
    print(f"\n合计费用: ${plan['total_cost_usd']}")
    print(f"最长周期: {plan['total_lead_time_weeks']}周（并行送测）")
    
    # 生成CPC草稿
    cpc = planner.generate_cpc_draft(product, plan)
    print("\n=== CPC声明草稿（前5行）===")
    print('\n'.join(cpc.strip().split('\n')[:6]))
