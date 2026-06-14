# Skill Card: CPSC 儿童产品安全合规（美国强制认证）

> **桥梁**: 21-合规决策 ↔ 04-供应链 | **类型**: 合规基础设施

roadmap_phase: phase1
---

## ① 算法原理

**核心思想**：美国消费品安全委员会（CPSC）对儿童产品实施强制第三方认证（3PTC），未通过不得在美销售。

母婴跨境电商必须理解三层合规框架：

**Layer 1：法规识别**
```
产品品类 → 对应联邦标准
  吸奶器/哺乳辅助 → 21 CFR (FDA管辖，非CPSC)
  婴儿玩具(<3岁) → ASTM F963 强制
  婴儿床/摇篮   → 16 CFR 1219-1220
  儿童服装      → CPSC flammability 标准
  汽车座椅      → FMVSS 213 (DOT管辖)
```

**Layer 2：测试路径**
```
选认证实验室(CPSC认可) → 样品测试 → CPC证书
  常见实验室: SGS/BV/Intertek/UL
  测试周期: 2-6周
  有效期: 产品无重大变更时持续有效
```

**Layer 3：标签与报告要求**
```
GCC/CPC证书 → 随货附带或存档备查
  亚马逊要求: 上传到 Seller Central > Documents
  海关要求: 进口时可能被抽查
```

---

## ② 母婴出海应用案例

**业务问题**：Momcozy 婴儿玩具品类扩张，新产品上架 Amazon 美国站被要求提供 CPSC 合规文件，不知道需要哪些认证、找哪家实验室、成本多少。

**应用流程**：
1. 产品归类（婴儿玩具 → ASTM F963）
2. 选定 SGS 实验室，提交样品
3. 4周后获取测试报告 + CPC 证书
4. 上传到 Amazon 后台，listing 通过审核
5. 建立内部合规文件管理系统（每款产品对应证书档案）

**年化收益**：
- 避免一次因合规被下架：保护 30-80 万 GMV
- 新市场进入合规准备从 3 个月缩短至 2-3 周
- 建立合规护城河，竞品难以快速跟进

---

## ③ 代码模板

```python
# CPSC 合规预检查工具
CPSC_REQUIREMENTS = {
    "infant_toy": {
        "standard": "ASTM F963",
        "third_party": True,
        "cert_required": "Children's Product Certificate (CPC)",
        "lab_options": ["SGS", "Bureau Veritas", "Intertek", "UL"],
        "typical_cost_usd": "800-2500",
        "typical_weeks": "3-6",
        "amazon_upload": "Seller Central > Catalog > Documents",
    },
    "nursing_pump": {
        "standard": "FDA 21 CFR (非CPSC)",
        "third_party": True,
        "cert_required": "FDA 510k clearance or 513(f)(2) De Novo",
        "note": "医疗器械，CPSC 不管辖",
    },
    "baby_clothing": {
        "standard": "16 CFR 1615/1616 (flammability)",
        "third_party": True,
        "cert_required": "General Certificate of Conformity (GCC)",
        "typical_cost_usd": "500-1500",
    },
    "baby_carrier": {
        "standard": "ASTM F2236",
        "third_party": True,
        "cert_required": "CPC",
        "typical_cost_usd": "1200-3000",
    },
}

def check_compliance_requirements(product_category: str, target_market: str = "US") -> dict:
    """
    输入产品品类，输出合规要求清单。
    """
    cat = product_category.lower().replace(" ", "_")
    req = CPSC_REQUIREMENTS.get(cat, {
        "note": f"未找到 '{product_category}' 的预置规则，请查询 CPSC.gov",
        "cpsc_url": "https://www.cpsc.gov/Business--Manufacturing/Business-Education/Business-Guidance"
    })
    
    result = {
        "product": product_category,
        "market": target_market,
        "requirements": req,
        "action_items": [],
    }
    
    if req.get("third_party"):
        result["action_items"].append(f"联系认证实验室: {req.get('lab_options', ['SGS'])[0]}")
        result["action_items"].append(f"预算: ${req.get('typical_cost_usd', 'TBD')}")
        result["action_items"].append(f"周期: {req.get('typical_weeks', 'TBD')} 周")
    
    return result

# 测试
result = check_compliance_requirements("infant_toy", "US")
import json
print(json.dumps(result, ensure_ascii=False, indent=2))
print("[✓] CPSC 合规预检查工具测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Category-Compliance-Prescan]] (21)
- **组合**：[[Skill-Cross-Border-Compliance-Framework]] (21) | [[Skill-Supply-Chain-Due-Diligence]] (21)
- **延伸**：[[Skill-Consumer-Complaint-Recall-Prediction]] (21)

---

## ⑤ 商业价值

- **ROI**：避免一次下架 = 30-80 万 GMV 保护；认证成本仅 $800-2500
- **难度**：⭐⭐☆☆☆（流程固定，主要是执行管理）
- **优先级**：⭐⭐⭐⭐⭐（P0 合规，必须满足才能在美销售）
- **适用场景**：新品上架前合规预检、供应商资质审查、亚马逊合规文件准备
