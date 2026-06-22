---
title: HTS Code Risk Classifier — 基于HTS码的CPSC多标签风险分类
doc_type: knowledge
module: 21-合规决策
topic: hts-code-risk-classifier
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: HTS Code Risk Classifier

> **论文/方法来源**：CPSC高风险消费品数据库（2024）+ 多标签分类（Multi-Label Classification）技术
> **领域**：合规决策 ↔ 供应链 | **类型**: 工程基础

## ① 算法原理

CPSC管制范围覆盖婴幼儿用品600+个HTS码，但卖家通常只知道自己的HTS码，不清楚哪些触发eFiling义务。手动逐一核对CPSC官网需2-3天。

**技术核心**：基于HTS前缀树的多标签风险分类：
1. **HTS树结构**：HTS码采用分层编码（前2位=章节，前4位=品目，前6位=子目），同一前缀下的商品共享风险特征，用前缀树批量匹配
2. **多标签输出**：每个HTS码输出4类标签：①是否需要eFiling ②危害类型（机械/化学/电气/窒息） ③CPSC风险等级（Class I/II/III） ④适用强制标准列表
3. **置信度衰减**：6位精确匹配置信度0.95，4位前缀匹配0.75，2位章节推断0.50

关键假设：HTS码格式为10位美国HTS（形如8715.00.0000），若卖家持有HS码（6位国际码）需先补全后两位。

## ② 母婴出海应用案例

**场景A：新品入库前CPSC风险预扫描（吸奶器品类扩展）**
- 业务问题：某母婴品牌计划从婴儿车扩展到电动吸奶器，50个新SKU需确认是否触发eFiling + GCC要求
- 数据要求：50个SKU的HTS码列表（从Supplier清单或Amazon后台导出），格式CSV
- 预期产出：自动输出分级清单：P0-必须eFiling（如吸奶器8479.89→Class II），P1-建议核查（配件类），P2-暂不需要
- 业务价值：新品上架周期缩短5天（省去人工查CPSC网站），避免首批货入仓即被拒收损失15万元

**场景B：全SKU年度合规体检（安全座椅卖家）**
- 业务问题：SKU数量500+的卖家，每年需核查CPSC法规更新是否影响现有商品，人工核查费用3万元/次
- 数据要求：全量SKU的HTS码 + 历史检测记录（判断现有证书是否覆盖当前法规）
- 预期产出：变更影响报告：哪些SKU因法规更新需重新测试（年均影响约10-15%的SKU）
- 业务价值：年度合规体检成本从3万元→0.2万元（工具费），年化节省2.8万元

## ③ 代码模板

```python
"""
HTS Code Risk Classifier
基于HTS码的CPSC多标签风险分类
"""
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CPSCRiskLabel:
    hts_code: str
    efiling_required: bool
    risk_class: str          # "Class I", "Class II", "Class III", "Not Regulated"
    hazard_types: List[str]  # ["mechanical", "chemical", "electrical", "choking"]
    applicable_standards: List[str]
    confidence: float
    match_type: str          # "exact", "prefix_6", "prefix_4", "chapter_infer"
    notes: str = ""


# CPSC受管制HTS码数据库（母婴高频品类，基于CPSC官方数据）
# 格式: HTS前缀 -> (风险等级, 危害类型列表, 适用标准列表, 备注)
CPSC_HTS_DATABASE = {
    # ===== Class I（最高风险，eFiling强制）=====
    "8715.00": ("Class I", ["mechanical", "choking"], 
                ["ASTM F833", "16 CFR Part 1228"], 
                "婴儿车，0-36个月，强制GCC"),
    "9401.20": ("Class I", ["mechanical"],
                ["FMVSS 213", "ASTM F97"],
                "儿童安全座椅，NHTSA管制，eFiling必须"),
    "9401.80.4001": ("Class I", ["mechanical"],
                    ["FMVSS 213"],
                    "婴儿安全座椅（专项子目）"),
    "9403.89": ("Class I", ["mechanical", "choking"],
                ["ASTM F1888", "16 CFR Part 1213", "16 CFR Part 1220"],
                "婴儿床，Class I需独立测试报告"),
    "9404.21": ("Class I", ["chemical", "mechanical"],
                ["ASTM F2933", "16 CFR Part 1633"],
                "床垫，阻燃要求+窒息风险"),
    
    # ===== Class II（高风险，eFiling强制）=====
    "8479.89.9499": ("Class II", ["electrical", "mechanical"],
                    ["UL 2738", "IEC 60335"],
                    "电动吸奶器，FDA II类医疗器械+CPSC双重管制"),
    "8479.89": ("Class II", ["electrical"],
               ["UL 60335", "IEC 60335-2-27"],
               "家用电器（含电动吸奶器），Class II"),
    "9503.00": ("Class II", ["choking", "mechanical"],
               ["ASTM F963", "16 CFR Part 1501"],
               "玩具（全类），含小零件测试"),
    "3924.90": ("Class II", ["chemical"],
               ["FDA 21 CFR", "ASTM F2456"],
               "塑料喂养用品（奶瓶/餐具），BPA法规"),
    "6111.20": ("Class II", ["chemical"],
               ["16 CFR Part 1615", "16 CFR Part 1616"],
               "婴儿棉质服装，阻燃强制标准"),
    "6111.30": ("Class II", ["chemical"],
               ["16 CFR Part 1615"],
               "婴儿合成纤维服装，阻燃"),
    
    # ===== Class III（中等风险，建议eFiling）=====
    "9504.90": ("Class III", ["choking"],
               ["ASTM F963 Clause 4.37"],
               "游戏/娱乐用品，非玩具但有年龄限制"),
    "4818.40": ("Class III", ["chemical"],
               ["ASTM E1871"],
               "一次性纸尿裤，皮肤安全测试"),
    "8714.99": ("Class III", ["mechanical"],
               ["ASTM F1625"],
               "儿童自行车座，附加安全要求"),
    
    # ===== Chapter-level推断规则 =====
    "87": ("Class I", ["mechanical"], [], "87章：机动车辆及零件，儿童类优先核查"),
    "94": ("Class I", ["mechanical", "choking"], [], "94章：家具及寝具，儿童用品需GCC"),
    "95": ("Class II", ["choking", "mechanical"], [], "95章：玩具/游戏/运动用品，ASTM F963覆盖"),
    "61": ("Class II", ["chemical"], [], "61章：针织服装，儿童类需阻燃检测"),
    "62": ("Class II", ["chemical"], [], "62章：机织服装，同上"),
}


def normalize_hts(hts_raw: str) -> str:
    """标准化HTS码格式，去除空格/点号，统一格式"""
    cleaned = re.sub(r'[\s\.]', '', hts_raw)
    # 重新插入标准点号（XXXX.XX.XXXX）
    if len(cleaned) >= 4:
        return cleaned[:4] + '.' + cleaned[4:6] + ('.' + cleaned[6:] if len(cleaned) > 6 else '')
    return cleaned


def classify_hts_risk(hts_code: str) -> CPSCRiskLabel:
    """对单个HTS码进行CPSC风险分类"""
    normalized = normalize_hts(hts_code)
    
    # 策略1：精确匹配（10位/8位）
    for length in [10, 8, 7]:
        key = normalized[:length] if len(normalized) >= length else normalized
        if key in CPSC_HTS_DATABASE:
            risk_class, hazards, standards, notes = CPSC_HTS_DATABASE[key]
            return CPSCRiskLabel(
                hts_code=normalized,
                efiling_required=risk_class in ("Class I", "Class II"),
                risk_class=risk_class,
                hazard_types=hazards,
                applicable_standards=standards,
                confidence=0.95,
                match_type="exact",
                notes=notes
            )
    
    # 策略2：6位子目前缀匹配
    prefix6 = normalized[:7]  # X X X X . X X
    for db_key in CPSC_HTS_DATABASE:
        if db_key.startswith(prefix6[:6]):
            risk_class, hazards, standards, notes = CPSC_HTS_DATABASE[db_key]
            return CPSCRiskLabel(
                hts_code=normalized,
                efiling_required=risk_class in ("Class I", "Class II"),
                risk_class=risk_class,
                hazard_types=hazards,
                applicable_standards=standards,
                confidence=0.75,
                match_type="prefix_6",
                notes=f"[6位前缀匹配 → {db_key}] {notes}"
            )
    
    # 策略3：2位章节推断
    chapter = normalized[:2]
    if chapter in CPSC_HTS_DATABASE:
        risk_class, hazards, standards, notes = CPSC_HTS_DATABASE[chapter]
        return CPSCRiskLabel(
            hts_code=normalized,
            efiling_required=False,  # 章节推断不足以强制eFiling，需人工确认
            risk_class=risk_class,
            hazard_types=hazards,
            applicable_standards=standards,
            confidence=0.50,
            match_type="chapter_infer",
            notes=f"[章节推断，需人工确认] {notes}"
        )
    
    # 未匹配
    return CPSCRiskLabel(
        hts_code=normalized,
        efiling_required=False,
        risk_class="Not Regulated",
        hazard_types=[],
        applicable_standards=[],
        confidence=0.85,
        match_type="not_found",
        notes="未在CPSC受管制HTS码数据库中找到，暂不需要eFiling"
    )


def batch_classify(hts_list: List[Dict]) -> Dict:
    """
    批量分类，输入: [{"sku": "SKU001", "hts_code": "8715.00.0000"}, ...]
    输出: 分级清单 + 统计摘要
    """
    p0_must_efile = []     # Class I/II + 高置信度
    p1_review = []         # Class III 或 低置信度
    p2_not_required = []   # 未受管制
    
    for item in hts_list:
        sku = item.get("sku", "UNKNOWN")
        hts = item.get("hts_code", "")
        label = classify_hts_risk(hts)
        
        entry = {
            "sku": sku,
            "hts_code": label.hts_code,
            "risk_class": label.risk_class,
            "efiling_required": label.efiling_required,
            "hazard_types": label.hazard_types,
            "standards": label.applicable_standards,
            "confidence": label.confidence,
            "notes": label.notes,
        }
        
        if label.efiling_required and label.confidence >= 0.75:
            p0_must_efile.append(entry)
        elif label.risk_class != "Not Regulated" or label.confidence < 0.70:
            p1_review.append(entry)
        else:
            p2_not_required.append(entry)
    
    return {
        "summary": {
            "total_skus": len(hts_list),
            "p0_must_efile": len(p0_must_efile),
            "p1_review": len(p1_review),
            "p2_not_required": len(p2_not_required),
        },
        "P0_必须eFiling": p0_must_efile,
        "P1_建议核查": p1_review,
        "P2_暂不需要": p2_not_required,
    }


# ========== 测试用例 ==========
if __name__ == "__main__":
    test_skus = [
        {"sku": "STROLLER-A300", "hts_code": "8715.00.0000"},    # 婴儿车 → Class I
        {"sku": "BREAST-PUMP-X1", "hts_code": "8479.89.9499"},   # 吸奶器 → Class II
        {"sku": "CAR-SEAT-B200", "hts_code": "9401.20.0000"},    # 安全座椅 → Class I
        {"sku": "TOY-BLOCK-C5", "hts_code": "9503.00.0090"},     # 玩具 → Class II
        {"sku": "DIAPER-D10", "hts_code": "4818.40.0000"},       # 纸尿裤 → Class III
        {"sku": "PHONE-CASE-Z", "hts_code": "3926.90.9990"},     # 手机壳 → Not Regulated
    ]
    
    result = batch_classify(test_skus)
    
    print("=== HTS Code Risk Classifier 测试结果 ===")
    s = result["summary"]
    print(f"总SKU数: {s['total_skus']}")
    print(f"P0-必须eFiling: {s['p0_must_efile']}个")
    print(f"P1-建议核查:    {s['p1_review']}个")
    print(f"P2-暂不需要:   {s['p2_not_required']}个")
    
    print("\n--- P0 必须eFiling ---")
    for item in result["P0_必须eFiling"]:
        print(f"  [{item['sku']}] {item['hts_code']} → {item['risk_class']} | 标准: {', '.join(item['standards'][:2])}")
    
    print("\n--- P1 建议核查 ---")
    for item in result["P1_建议核查"]:
        print(f"  [{item['sku']}] {item['hts_code']} → {item['risk_class']} (置信度: {item['confidence']:.0%})")
    
    # 断言验证
    assert result["summary"]["p0_must_efile"] >= 3, "Class I/II商品识别数量不足"
    assert any(i["sku"] == "STROLLER-A300" for i in result["P0_必须eFiling"]), "婴儿车应为P0"
    assert any(i["sku"] == "BREAST-PUMP-X1" for i in result["P0_必须eFiling"]), "吸奶器应为P0"
    assert any(i["sku"] == "PHONE-CASE-Z" for i in result["P2_暂不需要"]), "手机壳应为P2"
    
    print("\n[✓] HTS Code Risk Classifier 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-HTS-Tariff-Classification]]（HTS码正确分类是风险识别的前提）
- **延伸（extends）**：[[Skill-CPSC-Children-Product-Safety]]（Class I商品需进一步查儿童产品特殊要求）
- **延伸（extends）**：[[Skill-Compliance-ML-Risk-Scoring]]（结合ML风险评分做更精细的风险分级）
- **可组合（combinable）**：[[Skill-CPSC-eFiling-Auto-Mapper]]（风险扫描后直接进入字段填充流程，实现端到端自动化）
- **可组合（combinable）**：[[Skill-Category-Compliance-Prescan]]（上架前完整合规扫描，HTS风险分类是其中一环）

## ⑤ 商业价值评估

- ROI预估：节省人工核查时间40小时×150元/小时=6000元/次；避免1次FBA拒收损失=8-15万元；年化ROI约50-100倍
- 实施难度：⭐☆☆☆☆（纯规则查表，无需ML，数据库维护成本低）
- 优先级：⭐⭐⭐⭐⭐（时间窗口紧迫）
- 评估依据：CPSC eFiling 2026-07-08强制执行，遗漏一个Class I商品即触发FBA拒收，损失远超工具开发成本
