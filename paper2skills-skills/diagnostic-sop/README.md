# paper2skills 通用业务诊断 SOP

**把 1037 个 Skill 的知识，封装成任何人 30 分钟内可执行的诊断流程。**

---

## 快速开始

```bash
cd paper2skills-skills/diagnostic-sop

# 交互模式（推荐新手）
python3 diagnose.py

# 参数模式（快速执行）
python3 diagnose.py --sku "暖奶器" --channel amazon --problem repurchase_drop

# 真实数据模式（最准确）
python3 diagnose.py \
  --sku "暖奶器" --channel amazon --problem repurchase_drop \
  --reviews data/reviews.csv \
  --prices  data/prices.csv \
  --users   data/users.csv
```

---

## 支持的问题类型

| 参数 | 问题描述 | 核心 Skill |
|------|---------|-----------|
| `repurchase_drop` | 复购率下降 | VOC + 竞品 + 月龄时钟 + 因果 |
| `roas_decline` | 广告 ROAS 下滑 | 创意疲劳 + 竞品 + 归因去偏 |
| `traffic_drop` | ASIN 流量异常 | A10算法 + VOC + 竞品情报 |
| `new_cvr_low` | 新客转化率低 | Listing健康 + VOC + 竞品 |
| `inventory_issue` | 库存积压/断货 | 库存健康 + 月龄节律预测 |
| `review_attack` | 差评攻击/评分下滑 | 虚假评价检测 + VOC + 修复 |

---

## 输入数据格式

### reviews.csv（评价数据）
```
rating,period,review_text
4,before,质量不错
1,after,漏水严重
```
- `period`: `before`（对照期）/ `after`（问题期）
- `rating`: 1-5 星

### prices.csv（竞品价格）
```
day,own_price,comp_a,comp_b
0,38.99,39.50,44.99
1,38.99,39.20,44.80
```
- 前 N/2 行为问题前，后 N/2 行为问题后

### users.csv（用户月龄，仅母婴品类需要）
```
baby_age_months,churned
3,0
8,1
```
- `churned`: 1=已流失，0=未流失

---

## 输出结构

```
诊断报告
├── 综合结论（整体严重程度 + 主要驱动因素 + VOID洞察）
├── 诊断层（VOC / 竞品价格 / 婴儿月龄时钟 / 因果归因）
├── 处置层（优先级行动队列，含 Skill 直达链接）
├── 预防层（长效机制 Skill）
└── 参考链接（相关 Skill 卡片 URL）
```

---

## 四个诊断模块说明

### 1. VOC 评价分析
**Skill**: `Skill-VOC-Aspect-Sentiment-Extraction`

比较问题前后的评分分布、差评率变化、高频痛点词。
识别是否存在安全类严重差评（漏水/故障/安全隐患）。

### 2. 竞品价格监控
**Skill**: `Skill-Competitor-Price-Intelligence`

分析竞品近期价格变化，计算自身溢价水平。
自动标记降价幅度 > $2 的竞品行动。

### 3. 婴儿月龄时钟 ⚡ VOID 框架
**Skill**: `Skill-Baby-Age-Clock-RFM-Enhancement`

仅对母婴品类激活（奶粉/暖奶器/辅食/尿布等关键词触发）。

**核心洞察**：区分"月龄自然退出"和"真实流失"——
宝宝 7 个月以上的用户对暖奶器的需求自然消失，
传统 RFM 会将这部分用户误判为流失并浪费干预资源。

### 4. 因果归因
**Skill**: `Skill-Causal-Churn-Retention-Attribution`

综合前三个模块的信号，估算各因素对问题的贡献比例。
输出：主要驱动因素排名 + 建议优先处置顺序。

---

## 扩展新问题类型

在 `diagnose.py` 的 `PROBLEM_CONFIG` 字典中添加：

```python
"my_new_problem": {
    "name": "问题的中文名称",
    "modules": ["voc", "competitor_price"],   # 激活哪些诊断模块
    "primary_skills": ["Skill-XXX", ...],     # 诊断层 Skill
    "treatment_skills": ["Skill-YYY", ...],   # 处置层 Skill
    "prevention_skills": ["Skill-ZZZ", ...],  # 预防层 Skill
},
```

---

## 设计原则

1. **输入极简**：只需 SKU + 渠道 + 问题类型，无需懂 AI
2. **数据可选**：无真实数据时自动使用模拟数据，有数据则切换到真实模式
3. **结论明确**：每个模块输出一句话结论，不需要自己解读
4. **行动可执行**：处置层按天排序，含 Skill 直达 URL
5. **VOID 感知**：自动识别是否为母婴品类，激活月龄时钟模块
