---
title: Brand Registry Infringement Tracker — 品牌注册侵权追踪自动监控 EUIPO/USPTO
doc_type: knowledge
module: 19-风控反欺诈
topic: brand-registry-infringement-tracker
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Brand-Registry-Infringement-Tracker

## ① 算法原理（≤300字）

**核心问题**：母婴品牌的商标侵权分为两类：① 他人在电商平台使用近似商标（品牌仿冒）；② 他人在 EUIPO/USPTO 注册相同或近似商标（商标抢注）。前者发现已造成销售损失，后者若未监控可能导致商标无效化。

**近似商标检测算法**：

**文本相似度（品牌名检测）**：
- Levenshtein 编辑距离：字符级相似度，$\text{lev}(\text{a}, \text{b}) \leq 2$ 认为高度近似
- Jaro-Winkler 相似度：对前缀更敏感，$\text{jw} > 0.88$ 触发告警
- 发音相似（Soundex/Metaphone）：相同发音但不同拼写（如 KoolBaby vs CoolBaby）

**图形商标相似度**（视觉指纹）：
- Logo pHash 比对（参考 Skill-Supply-Chain-Counterfeit-Detection）
- 颜色分布直方图余弦相似度

**监控数据源**：
- Amazon Brand Registry 新上架品牌每日增量
- EUIPO/USPTO API（已申请商标公告）
- 平台 Listing 爬取（新竞品品牌名扫描）

**优先级分级**：
- P0（紧急）：完全相同或编辑距离 1 的商标在官方局注册申请
- P1（高）：近似商标出现在同类目 Listing，Jaro-Winkler > 0.90
- P2（观察）：相似度 0.80-0.90，列入监控

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某母婴品牌「MomzCare」在美国注册 USPTO 商标，监控系统发现竞品在欧洲以「MomsCare」申请 EUIPO 商标（编辑距离 2，Jaro-Winkler 0.95），提前 3 个月发现并提交 Opposition（异议）。

**数据要求**：品牌名称、分类（NICE Classification）、图形 Logo，EUIPO/USPTO 申请公告数据。

**应用**：自动每周扫描 EUIPO 相关类目新申请，识别近似商标，3 个月内成功提交 Opposition 并获批，避免商标注册侵权既成事实。

**量化产出**：成功阻止竞品商标注册，保护欧洲市场独家品牌权，年化保护品牌价值 **100-500 万元**（避免分裂权利或市场份额被蚕食）。

## ③ 代码模板

```python
import numpy as np

def levenshtein_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """计算 Jaro-Winkler 相似度"""
    if s1 == s2:
        return 1.0

    len_s1, len_s2 = len(s1), len(s2)
    match_dist = max(len_s1, len_s2) // 2 - 1
    match_dist = max(0, match_dist)

    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2
    matches = 0
    transpositions = 0

    for i in range(len_s1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len_s2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len_s1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches/len_s1 + matches/len_s2 + (matches - transpositions/2)/matches) / 3
    # Winkler 前缀加分
    prefix = 0
    for i in range(min(4, len_s1, len_s2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)

def scan_trademark_infringement(
    my_brand: str,
    candidate_brands: list,  # [{'name': str, 'region': str, 'application_date': str, 'class': str}]
    lev_threshold: int = 3,
    jw_threshold: float = 0.85
) -> dict:
    """
    商标侵权扫描
    """
    my_brand_lower = my_brand.lower()
    alerts = []

    for candidate in candidate_brands:
        name = candidate['name'].lower()
        lev = levenshtein_distance(my_brand_lower, name)
        jw = jaro_winkler_similarity(my_brand_lower, name)

        risk_level = 'NONE'
        if lev <= 1 or jw >= 0.95:
            risk_level = 'P0'
        elif lev <= 2 or jw >= 0.90:
            risk_level = 'P1'
        elif lev <= lev_threshold or jw >= jw_threshold:
            risk_level = 'P2'

        if risk_level != 'NONE':
            alerts.append({
                **candidate,
                'levenshtein': lev,
                'jaro_winkler': round(jw, 3),
                'risk_level': risk_level
            })

    alerts.sort(key=lambda x: {'P0': 0, 'P1': 1, 'P2': 2}[x['risk_level']])
    return {
        'my_brand': my_brand,
        'alerts': alerts,
        'p0_count': sum(1 for a in alerts if a['risk_level'] == 'P0'),
        'p1_count': sum(1 for a in alerts if a['risk_level'] == 'P1'),
        'total_alerts': len(alerts)
    }

# 测试
my_brand = "MomzCare"
candidates = [
    {'name': 'MomsCare', 'region': 'EUIPO', 'application_date': '2026-05-01', 'class': 'Class 21'},
    {'name': 'MomCare', 'region': 'USPTO', 'application_date': '2026-04-15', 'class': 'Class 21'},
    {'name': 'BabyCare', 'region': 'EUIPO', 'application_date': '2026-03-01', 'class': 'Class 21'},
    {'name': 'MomzKare', 'region': 'Amazon', 'application_date': '2026-06-01', 'class': 'N/A'},
    {'name': 'NurtureBrand', 'region': 'EUIPO', 'application_date': '2026-06-01', 'class': 'Class 10'},
]

result = scan_trademark_infringement(my_brand, candidates)
assert result['total_alerts'] >= 2, f"应发现至少2个近似商标，实际: {result['total_alerts']}"
assert result['p0_count'] + result['p1_count'] >= 1

print(f"我的品牌: {result['my_brand']}")
print(f"总告警数: {result['total_alerts']} (P0: {result['p0_count']}, P1: {result['p1_count']})")
for alert in result['alerts']:
    print(f"  [{alert['risk_level']}] {alert['name']} ({alert['region']}) - Lev:{alert['levenshtein']}, JW:{alert['jaro_winkler']}")
print("[✓] Brand-Registry-Infringement-Tracker 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-Hijacker-Seller-Network-Analysis]]（卖家身份识别）
> 延伸: [[Skill-Supply-Chain-Counterfeit-Detection]]（视觉商标比对）
> 可组合: [[Skill-Competitor-Negative-Campaign-Detection]]（全链路品牌防御）

## ⑤ 商业价值评估

- **ROI量化**: 阻止商标抢注，年化保护品牌价值 100-500 万元
- **实施难度**: ⭐⭐（商标文字比对算法简单，图形比对需额外工具）
- **优先级**: ⭐⭐⭐⭐（品牌注册后必须持续监控，非注册卖家优先注册）
