---
title: GPSR EU Risk Assessment Auto — 欧盟GPSR风险评估自动化
doc_type: knowledge
module: 21-合规决策
topic: gpsr-eu-risk-assessment-auto
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: GPSR EU Risk Assessment Auto

> **论文/方法来源**：EU Regulation 2023/988（GPSR）+ 风险评估矩阵（Risk Assessment Matrix）方法论
> **领域**：合规决策 ↔ 风控反欺诈 | **类型**: 工程基础

## ① 算法原理

欧盟《通用产品安全法规》（GPSR, EU 2023/988）于2024年12月13日正式生效，要求所有在欧销售消费品提供：①风险评估报告 ②欧盟责任人信息 ③产品安全联系点（24小时响应）④数字召回系统接入。未符合触发Amazon Error 5995，Listing被暂停，库存30天后自动销毁。

**技术核心**：基于风险矩阵（Risk Matrix）的结构化评估引擎：
1. **危害识别**：基于产品类别的预设危害清单（机械/电气/化学/生物/人体工程学5类），结合产品描述关键词自动补全
2. **风险量化**：2×2矩阵（发生可能性 × 严重程度），每个危害输出风险等级（Negligible/Low/Medium/High/Critical）
3. **控制措施匹配**：风险等级→标准控制措施（EN标准/CE认证/欧盟测试报告）的自动映射
4. **GPSR文档生成**：基于评估结果自动生成符合GPSR附件I格式要求的风险评估报告草稿

关键假设：产品已有CE认证和相关EN标准测试报告，GPSR评估是在此基础上的额外合规层，不取代CE认证。

## ② 母婴出海应用案例

**场景A：婴儿床欧盟市场准入风险评估（德国/法国站）**
- 业务问题：销售婴儿床的卖家收到Amazon Error 5995，要求提交GPSR风险评估报告，不知道如何撰写（传统做法：委托第三方机构出具，费用5000-15000元，周期3-4周）
- 数据要求：产品规格表、现有CE认证报告（EN 1130婴儿床标准）、欧盟责任人信息
- 预期产出：GPSR风险评估报告草稿（符合附件I格式）+ 欧盟责任人声明模板，可直接上传到Amazon合规门户
- 业务价值：报告生成时间从3周→2小时（草稿）+1天（律师审查），成本从1.5万元→3000元，欧盟准入周期缩短40%

**场景B：吸奶器多欧盟站点合规矩阵构建**
- 业务问题：在DE/FR/IT/ES/NL五个欧盟站销售的吸奶器，每个站点的GPSR细节要求有差异（如语言要求、责任人本地化），手动维护5份文档极易出错
- 数据要求：5个站点的产品合规要求清单 + 现有风险评估基础文档
- 预期产出：统一的风险评估基础版本 + 各站点本地化差异对照表
- 业务价值：多站点合规管理效率提升60%，防止因文档不一致导致部分站点Listing被暂停（欧盟市场月GMV约20万元）

## ③ 代码模板

```python
"""
GPSR EU Risk Assessment Auto
欧盟通用产品安全法规(EU 2023/988)风险评估自动化
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re


# 风险等级定义（ISO 31000）
RISK_LEVELS = {
    "Negligible": {"score": 1, "action": "记录在案，无需额外措施"},
    "Low": {"score": 2, "action": "标准控制措施足够"},
    "Medium": {"score": 4, "action": "需要额外防护措施或警告标签"},
    "High": {"score": 8, "action": "需要设计改进或强制安全认证"},
    "Critical": {"score": 16, "action": "产品可能无法满足GPSR要求，建议暂停上市"},
}

# 危害类型库（母婴产品相关）
HAZARD_CATALOG = {
    "mechanical": {
        "name": "机械危害",
        "description": "刺伤、夹伤、倒塌、绞入",
        "relevant_categories": ["婴儿车", "安全座椅", "婴儿床", "高脚椅", "玩具"],
        "default_controls": ["ISO 8124-1", "EN 1888（婴儿车）", "EN 1130（婴儿床）", "材料强度测试"],
    },
    "choking": {
        "name": "窒息/误吞危害",
        "description": "小零件吞咽、绳索绕颈",
        "relevant_categories": ["玩具", "婴儿服装", "喂养用品", "安抚奶嘴"],
        "default_controls": ["小零件测试（直径≥31.7mm）", "绳索长度限制（≤220mm）", "警告标签"],
    },
    "chemical": {
        "name": "化学危害",
        "description": "重金属（铅/镉/汞）、邻苯二甲酸盐、双酚A",
        "relevant_categories": ["玩具", "婴儿服装", "喂养用品", "奶瓶", "床垫"],
        "default_controls": ["REACH法规检测", "RoHS合规", "EN 71-3重金属测试", "邻苯二甲酸盐<0.1%"],
    },
    "electrical": {
        "name": "电气危害",
        "description": "触电、过热、起火",
        "relevant_categories": ["电动吸奶器", "婴儿监视器", "电热垫", "智能玩具"],
        "default_controls": ["CE认证（LVD低电压指令）", "IEC 60335测试", "UL认证（美国）"],
    },
    "ergonomic": {
        "name": "人体工程学危害",
        "description": "不正确支撑导致脊柱发育问题、窒息体位",
        "relevant_categories": ["婴儿车", "安全座椅", "婴儿背带", "婴儿躺椅"],
        "default_controls": ["EN 1888姿态测试", "婴儿头颈支撑测试", "使用说明书警告"],
    },
    "biological": {
        "name": "生物/卫生危害",
        "description": "细菌滋生、过敏原、霉菌",
        "relevant_categories": ["奶瓶", "安抚奶嘴", "婴儿食品容器", "床上用品"],
        "default_controls": ["食品接触材料认证（EU 10/2011）", "抗菌测试", "清洁说明书"],
    },
}

# 产品类别 → 适用危害类型映射
CATEGORY_HAZARD_MAP = {
    "婴儿车": ["mechanical", "choking", "ergonomic"],
    "安全座椅": ["mechanical", "ergonomic"],
    "婴儿床": ["mechanical", "choking", "ergonomic"],
    "吸奶器": ["electrical", "biological", "chemical"],
    "玩具": ["mechanical", "choking", "chemical"],
    "婴儿服装": ["choking", "chemical"],
    "奶瓶": ["chemical", "biological"],
    "婴儿背带": ["mechanical", "ergonomic"],
}

# GPSR欧盟站点特定要求
EU_COUNTRY_REQUIREMENTS = {
    "DE": {"language": "德语", "local_rep_required": True, "specific_standards": ["DIN"]},
    "FR": {"language": "法语", "local_rep_required": True, "specific_standards": []},
    "IT": {"language": "意大利语", "local_rep_required": True, "specific_standards": []},
    "ES": {"language": "西班牙语", "local_rep_required": True, "specific_standards": []},
    "NL": {"language": "荷兰语", "local_rep_required": True, "specific_standards": []},
    "UK": {"language": "英语", "local_rep_required": True, "specific_standards": ["UKCA（脱欧后）"]},
}


@dataclass
class HazardAssessment:
    hazard_type: str
    hazard_name: str
    likelihood: int      # 1-4: Rare/Unlikely/Possible/Probable
    severity: int        # 1-4: Minor/Moderate/Major/Critical
    risk_score: int      # likelihood * severity
    risk_level: str
    current_controls: List[str]
    additional_measures: List[str]
    residual_risk: str


def calculate_risk_level(likelihood: int, severity: int) -> str:
    """风险矩阵：可能性(1-4) × 严重程度(1-4) = 风险分数"""
    score = likelihood * severity
    if score <= 2:
        return "Negligible"
    elif score <= 4:
        return "Low"
    elif score <= 8:
        return "Medium"
    elif score <= 12:
        return "High"
    else:
        return "Critical"


def assess_product_risks(
    product_name: str,
    product_category: str,
    target_age: str,
    existing_certifications: List[str],
    eu_markets: List[str]
) -> Dict:
    """
    为产品生成GPSR风险评估报告
    """
    # 确定适用危害类型
    applicable_hazards = CATEGORY_HAZARD_MAP.get(product_category, ["mechanical"])
    
    # 婴儿产品（0-36个月）自动提升危害严重程度
    age_multiplier = 1.5 if "0" in target_age and "month" in target_age.lower() else 1.0
    
    hazard_assessments = []
    
    for hazard_type in applicable_hazards:
        hazard_info = HAZARD_CATALOG[hazard_type]
        
        # 基础风险评估（简化版，实际应结合具体产品设计）
        base_likelihood = 2  # Unlikely（有控制措施时）
        base_severity = 3 if hazard_type in ["choking", "electrical"] else 2  # 窒息/电气危害严重
        
        # 婴儿产品提升严重程度
        adjusted_severity = min(4, int(base_severity * age_multiplier))
        
        risk_level = calculate_risk_level(base_likelihood, adjusted_severity)
        
        # 已有认证覆盖的控制措施
        covered_controls = []
        additional_measures = []
        for ctrl in hazard_info["default_controls"]:
            cert_covered = any(cert.upper() in ctrl.upper() for cert in existing_certifications)
            if cert_covered or any(keyword in ctrl for keyword in ["CE", "测试", "标准"]):
                covered_controls.append(ctrl)
            else:
                additional_measures.append(ctrl)
        
        # 残余风险（有控制措施后）
        residual_likelihood = max(1, base_likelihood - 1) if covered_controls else base_likelihood
        residual_risk_level = calculate_risk_level(residual_likelihood, adjusted_severity)
        
        hazard_assessments.append(HazardAssessment(
            hazard_type=hazard_type,
            hazard_name=hazard_info["name"],
            likelihood=base_likelihood,
            severity=adjusted_severity,
            risk_score=base_likelihood * adjusted_severity,
            risk_level=risk_level,
            current_controls=covered_controls,
            additional_measures=additional_measures,
            residual_risk=residual_risk_level,
        ))
    
    # GPSR合规状态判断
    critical_hazards = [h for h in hazard_assessments if h.residual_risk in ("Critical", "High")]
    gpsr_ready = len(critical_hazards) == 0
    
    # 生成报告
    report = {
        "document_type": "GPSR Risk Assessment Report",
        "regulation": "EU Regulation 2023/988",
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "product_info": {
            "name": product_name,
            "category": product_category,
            "target_age": target_age,
            "existing_certifications": existing_certifications,
        },
        "gpsr_compliance_status": "READY" if gpsr_ready else "REQUIRES_ACTION",
        "hazard_assessments": [
            {
                "hazard": h.hazard_name,
                "initial_risk": h.risk_level,
                "current_controls": h.current_controls,
                "additional_measures_needed": h.additional_measures,
                "residual_risk": h.residual_risk,
            }
            for h in hazard_assessments
        ],
        "critical_issues": [
            f"{h.hazard_name}：残余风险仍为{h.residual_risk}级，需要额外措施"
            for h in critical_hazards
        ],
        "eu_market_requirements": {
            market: EU_COUNTRY_REQUIREMENTS.get(market, {"language": "English"})
            for market in eu_markets
        },
        "required_gpsr_documents": [
            "产品风险评估报告（本文档）",
            "欧盟责任人（EU Responsible Person）声明",
            "产品安全联系点（24小时响应邮件/电话）",
            "召回计划说明（含Amazon Recall Portal接入方式）",
            f"产品标签（需包含{'/'.join([EU_COUNTRY_REQUIREMENTS.get(m,{}).get('language','English') for m in eu_markets])}语言）",
        ],
        "amazon_error_5995_prevention": {
            "status": "满足" if gpsr_ready else "不满足",
            "missing_items": [h.hazard_name for h in critical_hazards],
            "submission_checklist": [
                "✓ 风险评估报告（PDF，附签名）",
                "✓ 欧盟责任人信息（姓名+地址+联系方式）",
                "✓ 产品型号与GPSR报告一致",
                "✓ 报告日期在12个月内",
            ]
        }
    }
    
    return report


def generate_responsible_person_declaration(
    company_name: str,
    contact_email: str,
    product_name: str,
    eu_address: str
) -> str:
    """生成欧盟责任人声明模板（GPSR要求）"""
    today = datetime.now().strftime("%Y年%m月%d日")
    return f"""
EU RESPONSIBLE PERSON DECLARATION
(In accordance with EU Regulation 2023/988 on General Product Safety)

Company: {company_name}
EU Address: {eu_address}
Safety Contact: {contact_email} (24-hour response commitment)

We, {company_name}, hereby declare that we act as the Responsible Person in the 
European Union for the following product(s):

Product Name: {product_name}

We confirm that:
1. The product complies with all applicable EU product safety legislation
2. A risk assessment has been conducted and documented
3. Technical documentation is maintained and available to market surveillance authorities
4. Corrective actions will be taken immediately upon identification of any safety issue
5. We will cooperate fully with market surveillance authorities

Date: {today}
Signature: _____________________ (Authorized Representative)
""".strip()


# ========== 测试用例 ==========
if __name__ == "__main__":
    # 测试：婴儿车欧盟市场准入
    report = assess_product_risks(
        product_name="Pro Baby Stroller A300",
        product_category="婴儿车",
        target_age="0-36 months",
        existing_certifications=["CE", "EN 1888", "ASTM F833"],
        eu_markets=["DE", "FR", "NL"]
    )
    
    print("=== GPSR EU Risk Assessment Auto 测试结果 ===")
    print(f"产品: {report['product_info']['name']}")
    print(f"GPSR合规状态: {report['gpsr_compliance_status']}")
    print(f"Error 5995预防状态: {report['amazon_error_5995_prevention']['status']}")
    
    print("\n--- 危害评估摘要 ---")
    for h_info in report["hazard_assessments"]:
        print(f"  {h_info['hazard']}: 初始风险={h_info['initial_risk']} → 残余风险={h_info['residual_risk']}")
    
    if report["critical_issues"]:
        print(f"\n⚠️  关键问题: {report['critical_issues']}")
    else:
        print("\n✅ 无关键风险问题")
    
    print(f"\n需要的GPSR文档数量: {len(report['required_gpsr_documents'])}份")
    
    # 测试责任人声明生成
    declaration = generate_responsible_person_declaration(
        company_name="Baby World Ltd.",
        contact_email="safety@babyworld.eu",
        product_name="Pro Baby Stroller A300",
        eu_address="123 EU Street, Berlin 10115, Germany"
    )
    assert "EU RESPONSIBLE PERSON DECLARATION" in declaration, "声明模板生成失败"
    assert "EU Regulation 2023/988" in declaration, "法规引用缺失"
    
    # 断言验证
    assert "hazard_assessments" in report, "缺少危害评估"
    assert len(report["hazard_assessments"]) >= 2, "婴儿车应有≥2种危害"
    assert "required_gpsr_documents" in report, "缺少文档清单"
    assert len(report["eu_market_requirements"]) == 3, "欧盟市场数量不匹配"
    
    print("\n[✓] GPSR EU Risk Assessment Auto 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-HTS-Code-Risk-Classifier]]（确认产品是否属于GPSR受管制商品）
- **前置（prerequisite）**：[[Skill-Cross-Border-Compliance-Framework]]（理解欧盟合规整体框架）
- **延伸（extends）**：[[Skill-Amazon-Compliance-Error-Auto-Resolver]]（GPSR不合规触发Error 5995，需要配合错误码修复）
- **延伸（extends）**：[[Skill-Regulatory-Graph-Compliance-Monitor]]（监控GPSR实施细则更新，自动触发重新评估）
- **可组合（combinable）**：[[Skill-AI-Product-Safety-Certification]]（AI辅助识别GPSR文档缺失项，与本Skill生成的评估报告配合使用）

## ⑤ 商业价值评估

- ROI预估：替代第三方机构出具GPSR报告（5000-15000元/次），年化节省6-18万元（按3次/年计）；防止Error 5995触发的Listing暂停（欧盟站月GMV损失5-20万元）
- 实施难度：⭐⭐☆☆☆（规则引擎+模板生成，草稿仍需律师或合规顾问审核）
- 优先级：⭐⭐⭐⭐⭐（时间窗口紧迫）
- 评估依据：GPSR 2024-12-13已生效，Amazon已开始执行Error 5995，欧盟市场库存30天内自动销毁风险为高概率事件
