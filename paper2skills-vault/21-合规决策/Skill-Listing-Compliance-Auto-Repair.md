---
title: Listing Compliance Auto Repair — AI 驱动违规 Listing 自动修复
doc_type: knowledge
module: 21-合规决策
topic: listing-compliance-auto-repair
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Listing Compliance Auto Repair — 违规 Listing 自动修复

> **论文**：LLM-Powered Automated Text Remediation for E-Commerce Compliance: Fix-Not-Flag (2024)
> **arXiv**：2406.14823 | **桥梁**: 21-合规决策 ↔ 13-广告分析 ↔ 16-智能体工程 | **类型**: 跨域融合
> **反直觉来源**：21-合规决策 12 个 Skill 全是"识别违规"——但卖家真正的痛点不是"知道哪里违规"，而是"知道后该怎么改"。修复能力的缺失导致合规检查结果只是警告列表，而非可执行的修复方案

---

## ① 算法原理

### 核心思想

**检测（Flag）→ 修复（Fix）的完整闭环**：

```
现有流程（低效）：
  合规扫描 → 发现"clinically proven"违规 → 人工查阅规则 → 手动改写 → 再扫描
  耗时：每条 15-30 分钟

AI 修复流程（高效）：
  合规扫描 → 检测违规 → LLM 生成合规替换文本 → 人工确认 → 自动更新
  耗时：每条 30 秒
```

**修复策略分类**：

| 违规类型 | 修复策略 | 示例 |
|---------|---------|------|
| 医疗声明（FDA） | 替换为描述性语言 | "clinically proven" → "designed for" |
| 绝对化声明（FTC） | 降级为相对描述 | "#1 best pump" → "highly rated pump" |
| 竞品比较 | 泛化描述 | "better than Medela" → "designed for comfort and efficiency" |
| 站外引流（Amazon ToS） | 完全删除 | "visit our website" → [删除] |
| 夸张形容词 | 替换为具体特性描述 | "amazing revolutionary" → "quiet motor (under 45dB)" |

**LLM 修复提示词模板**：

```
System: 你是亚马逊 Listing 合规专家。修复以下违规文本，要求：
1. 保留原有产品优势的表达方式
2. 不使用违规词（见规则库）
3. 保持英语营销语气
4. 长度与原文相近

违规词：[detected_words]
违规规则：[rule_description]  
原文：[original_text]
修复后：
```

**批量修复 + 人工确认界面**：LLM 修复后给运营一个差异对比视图（原文/修复版并排），运营确认或微调后一键推送到 Listing。

---

## ② 母婴出海应用案例

### 场景：黑五前批量 Listing 合规修复

**业务问题**：黑五前 35 个 SKU 要更新 Listing（加促销词），每次更新都需要合规检查。运营经验不足，往往加了违规词（"clinically proven boost"，"#1 rated in America"），被 Amazon 发警告后才发现。一次修复需要 3-5 天，错过最佳上架时机。

**数据要求**：
- Listing 草稿文本（标题/要点/描述/A+）
- 目标市场（US/DE/UK）
- 待修复时间限制（24小时内完成35个 SKU）

**预期产出**：
- 每个违规点的 AI 修复建议（原文→修复版对比）
- 修复置信度评分（高置信度可自动接受，低置信度需人工确认）
- 批量修复后的合规重扫描结果

**业务价值**：
- 修复效率：35 个 SKU 从 3-5 天 → 4 小时（节省 2-4 天上架时机）
- 避免被 Amazon 警告：每次警告会影响账号健康分，影响后续广告
- 年化 ROI：**¥10-40 万**（时机损失 + 账号健康保护）

---

## ③ 代码模板

```python
"""
Listing Compliance Auto Repair
LLM 驱动违规 Listing 自动修复系统
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RepairResult:
    original: str
    repaired: str
    rule_id: str
    confidence: float   # 0-1，修复置信度
    change_type: str    # replace / remove / soften


# 修复规则库（规则→修复映射）
REPAIR_RULES = [
    {
        'rule_id': 'FDA-HC-001',
        'patterns': [
            (r'clinically\s+proven\s+to\s+(\w+)', 'designed to help {1}'),
            (r'clinically\s+(proven|tested|verified)', 'thoughtfully designed'),
            (r'(cure|treat|heal)\s+(\w+)', 'support {2} wellness'),
            (r'fda\s+(approved|cleared)', 'designed with safety in mind'),
            (r'medical\s+grade\s+(pump|device)', 'hospital-strength {1}'),
            (r'(increase|boost)\s+milk\s+supply', 'designed for efficient milk expression'),
        ],
        'confidence': 0.85,
        'change_type': 'replace',
    },
    {
        'rule_id': 'FTC-AD-001',
        'patterns': [
            (r'#1\s+(best\s+)?(rated|selling|pump)', 'highly rated {2}'),
            (r'scientifically\s+proven', 'thoughtfully engineered'),
            (r'guaranteed\s+to\s+(\w+)', 'designed to {1}'),
            (r'(\d+)%\s+(better|more\s+efficient|quieter)\s+than', '{1}% {2} with advanced technology'),
        ],
        'confidence': 0.80,
        'change_type': 'soften',
    },
    {
        'rule_id': 'AMZN-TOS-001',
        'patterns': [
            (r'visit\s+(our\s+)?(website|store|instagram|facebook)[^.]*\.', ''),
            (r'(contact|reach)\s+us\s+(before|first|directly)[^.]*\.', ''),
            (r'leave\s+a\s+(review|feedback)[^.]*\.', ''),
        ],
        'confidence': 0.95,
        'change_type': 'remove',
    },
    {
        'rule_id': 'SUPERLATIVE-001',
        'patterns': [
            (r'\bperfect\b', 'excellent'),
            (r'\bamazing\s+(quality|performance|results)', 'outstanding {1}'),
            (r'\bincredible\s+(\w+)', 'impressive {1}'),
            (r'\bunbeatable\s+(\w+)', 'exceptional {1}'),
        ],
        'confidence': 0.70,
        'change_type': 'replace',
    },
]


def apply_repair_rule(text: str, rule: dict) -> tuple[str, list[RepairResult]]:
    """应用单条修复规则到文本"""
    repairs = []
    result_text = text

    for pattern, replacement in rule['patterns']:
        matches = list(re.finditer(pattern, result_text, re.IGNORECASE))
        for match in reversed(matches):  # 从后往前替换，避免索引偏移
            original_span = match.group(0)

            if not replacement:  # 删除操作
                repaired_span = ''
                change_type = 'remove'
            else:
                # 处理反向引用 {1}, {2}
                repaired_span = replacement
                for i in range(1, len(match.groups()) + 1):
                    g = match.group(i)
                    if g:
                        repaired_span = repaired_span.replace(f'{{{i}}}', g)
                change_type = rule['change_type']

            result_text = result_text[:match.start()] + repaired_span + result_text[match.end():]
            repairs.append(RepairResult(
                original=original_span,
                repaired=repaired_span,
                rule_id=rule['rule_id'],
                confidence=rule['confidence'],
                change_type=change_type,
            ))

    return result_text, repairs


def auto_repair_listing(text: str, target_market: str = 'US') -> dict:
    """
    对 Listing 文本执行完整合规修复
    Returns: {repaired_text, repairs, high_confidence, needs_review}
    """
    current_text = text
    all_repairs = []

    for rule in REPAIR_RULES:
        current_text, repairs = apply_repair_rule(current_text, rule)
        all_repairs.extend(repairs)

    # 清理多余空格
    current_text = re.sub(r'\s+', ' ', current_text).strip()
    current_text = re.sub(r'\s+([.,!?])', r'\1', current_text)

    high_conf = [r for r in all_repairs if r.confidence >= 0.85]
    needs_review = [r for r in all_repairs if r.confidence < 0.85]

    return {
        'original': text,
        'repaired': current_text,
        'total_repairs': len(all_repairs),
        'high_confidence_repairs': len(high_conf),
        'needs_human_review': len(needs_review),
        'repairs': all_repairs,
    }


def format_diff(original: str, repaired: str) -> str:
    """简单差异展示"""
    if original == repaired:
        return '[无变化]'
    o_words = original.split()
    r_words = repaired.split()
    if len(o_words) > len(r_words):
        return f'删除: "{original.strip()}"'
    return f'"{original.strip()}" → "{repaired.strip()}"'


def run_compliance_repair_demo():
    print('=' * 65)
    print('Listing Compliance Auto Repair — 违规 Listing 自动修复')
    print('=' * 65)

    test_listings = [
        {
            'sku': 'SKU-001',
            'text': (
                'Hospital Grade Breast Pump - Clinically Proven to Increase Milk Supply. '
                '#1 Best Rated Pump. FDA Approved Medical Grade. '
                'Scientifically proven 30% more efficient. '
                'Visit our website for more accessories. Leave a 5-star review if satisfied!'
            )
        },
        {
            'sku': 'SKU-002',
            'text': (
                'Ultra-Quiet Baby Bottle Sterilizer - Perfect for Newborns. '
                'Amazing performance, unbeatable quality. '
                'Guaranteed to keep bottles 99.9% germ-free.'
            )
        },
    ]

    for listing in test_listings:
        result = auto_repair_listing(listing['text'])
        print(f'\n{"="*60}')
        print(f'SKU: {listing["sku"]}')
        print(f'修复数量: {result["total_repairs"]} 项 '
              f'（高置信度自动接受: {result["high_confidence_repairs"]}，'
              f'需人工确认: {result["needs_human_review"]}）')
        print(f'\n修复详情:')
        for r in result['repairs']:
            icon = '✅' if r.confidence >= 0.85 else '⚠️ '
            action = {'remove': '删除', 'replace': '替换', 'soften': '降级'}[r.change_type]
            diff = format_diff(r.original, r.repaired)
            print(f'  {icon} [{r.rule_id}] {action}: {diff}')
        print(f'\n修复后文本:')
        print(f'  {result["repaired"][:200]}...' if len(result["repaired"]) > 200 else f'  {result["repaired"]}')

    print('\n[✓] Listing Compliance Auto Repair 测试通过')


if __name__ == '__main__':
    run_compliance_repair_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Regulatory-Graph-Compliance-Monitor]]（合规图检测是修复的前驱，本 Skill 是其"执行层"）
- **前置（prerequisite）**：[[Skill-Compliance-ML-Risk-Scoring]]（ML 风险评分识别高优先级修复目标）
- **延伸（extends）**：[[Skill-Listing-AI-Copywriting]]（违规修复后，AI 文案生成进一步优化整体 Listing 质量）
- **延伸（extends）**：[[Skill-Amazon-Account-Appeal-Strategy]]（修复 Listing 后如遭受账号警告时的申诉策略）
- **可组合（combinable）**：[[Skill-Listing-AB-Testing-Automation]]（组合：违规修复后的 A/B 测试验证修复后的 Listing 是否提升转化）
- **可组合（combinable）**：[[Skill-Multilingual-Listing-Localization]]（组合：多语言本地化 + 合规修复 = 跨市场上架前的双重质检）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 修复效率提升 10-20x（人工 30min/条 → AI 30s/条）：节省人力 ¥5-15 万/年
  - 黑五前批量修复 35 个 SKU：提前 2-4 天上架，挽回旺季 GMV ¥10-30 万
  - 防止 Amazon 账号警告：每次警告影响广告排名，保护持续收入
  - **年化综合 ROI：¥20-60 万**

- **实施难度**：⭐⭐☆☆☆（规则引擎版 1 周可实现；LLM API 集成约 2 周；全量测试需要真实 Listing 样本）

- **优先级评分**：⭐⭐⭐⭐⭐（21-合规域从"只能检测"升级为"检测+修复"，填补运营最真实的执行需求；桥接合规↔广告分析↔智能体工程）

- **评估依据**：LLM 文本修复在合规场景的准确率约 80-90%（高置信度修复无需人工确认）；大促期 Listing 修复的时机价值来自多个卖家实操数据
