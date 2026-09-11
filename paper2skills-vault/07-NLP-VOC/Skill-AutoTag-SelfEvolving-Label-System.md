# Skill Card: VOC 自动打标签与自进化标签体系
# AutoTag Self-Evolving Label System for VOC

**论文来源**: InsightNet — Structured Insight Mining from Customer Feedback (Amazon, arXiv:2405.07195, 2024)
**理论基础**: 层级 Topic Classification + 开放世界标签发现 + 持续学习
**适用领域**: 消费者评价分析、客服反馈结构化、产品痛点追踪、运营标签自动化

---

## ① 算法原理

### 核心思想
消费者反馈是非结构化的文本（评论、客服记录、社媒帖子），传统人工打标签成本高、覆盖率低、更新滞后。本技能构建一个**生产-评估-进化**三阶段闭环系统：先通过层级分类器自动打标签，再用一致性评估筛选高质量标签，最后基于新数据自动扩展标签体系。

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    标签生产 (Label Production)                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ L1 品类   │ → │ L2 问题域 │ → │ L3 细分类 │ → L4 具体点 │
│  │  (粗)    │    │  (中)    │    │  (细)    │    (最细)   │
│  └──────────┘    └──────────┘    └──────────┘              │
│       多任务模型: Topic + Sentiment + Verbatim 联合预测       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    标签评估 (Label Assessment)                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ cleanlab   │  │ 覆盖率监控  │  │ 业务匹配度  │            │
│  │ 噪声检测   │  │ (标签分布)  │  │ (人工抽检)  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    标签进化 (Label Evolution)                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ 新标签发现  │  │ 低频标签淘汰│  │ 层级关系更新│            │
│  │ (聚类+LLM) │  │ (阈值过滤)  │  │ (父子重计算)│            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 层级标签体系设计

采用 L1→L2→L3→L4 四级架构，**每一层都是下一层的父标签**，形成树状结构：

| 层级 | 粒度 | 示例（母婴出海） | 标签数量 |
|------|------|-----------------|---------|
| L1 | 品类维度 | 纸尿裤 / 奶粉 / 童装 | ~20 |
| L2 | 问题域 | 质量 / 物流 / 价格 / 服务 | ~10/品类 |
| L3 | 细分类别 | 材质舒适度 / 尺码偏差 / 漏尿 | ~30/问题域 |
| L4 | 具体痛点 | "腰贴太硬划伤皮肤" / "夜间侧漏" | 动态扩展 |

**关键设计**: L1-L3 由业务专家预定义，L4 由模型自动发现。L4 标签积累到一定量后，经人工确认可提升为 L3。

### 多任务联合预测

对每条反馈文本 $x$，同时预测三个输出：

$$
\hat{y} = f_{MT}(x) = \{ \underbrace{\hat{t}}_{\text{topic}}, \underbrace{\hat{s}}_{\text{sentiment}}, \underbrace{\hat{v}}_{\text{verbatim}} \}
$$

其中：
- **Topic** $\hat{t} \in \{1, ..., K\}$: 标签分类（L1-L4 层级预测）
- **Sentiment** $\hat{s} \in \{-1, 0, +1\}$: 情感极性（负/中/正）
- **Verbatim** $\hat{v}$: 原文中支撑该标签的关键短语（可解释性）

联合损失函数：

$$
\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{\text{CE}}(t, \hat{t}) + \lambda_2 \cdot \mathcal{L}_{\text{CE}}(s, \hat{s}) + \lambda_3 \cdot \mathcal{L}_{\text{span}}(v, \hat{v})
$$

### 新标签发现（开放世界识别）

当模型对某条文本的 Topic 置信度低于阈值 $\tau$ 时，触发**新标签发现流程**：

1. **语义聚类**: 将低置信度文本用 sentence embedding 聚类
2. **候选命名**: 用 LLM 为每个聚类生成标签名称和描述
3. **相似度去重**: 与现有标签计算语义相似度，过滤重复
4. **人工确认**: 运营人员审核后入库（半自动）

新标签发现率经验值约 **12-18%**（Amazon 数据），即每批新数据中约有 15% 的反馈无法被现有标签覆盖，需要新增标签。

### 标签进化策略

**触发条件**（任一满足即触发进化）：
- 时间窗口：每 30 天或每收到 10,000 条新反馈
- 覆盖率下降：现有标签覆盖率 < 85%
- 新标签候选积累：未确认候选标签 ≥ 20 个

**进化动作**：
1. **新增**: 高频新候选标签通过阈值（出现次数 ≥ 50）后入库
2. **合并**: 语义相似度 > 0.85 的两个标签合并
3. **淘汰**: 连续 90 天无命中的标签标记为"休眠"
4. **升级**: L4 标签下子标签数量 ≥ 10 时，可抽取共同上位概念提升为 L3

### 关键假设

1. **文本质量**: 输入文本长度 ≥ 5 个中文字符，过短文本（如"还行"）难以准确分类
2. **领域稳定**: L1-L3 层级相对稳定，不频繁变动；L4 为动态扩展层
3. **反馈密度**: 需要一定数据量支撑聚类发现新标签（建议 ≥ 1000 条/月）
4. **人工参与**: 新标签入库需要人工确认，全自动化会导致标签膨胀

---

## ② 母婴出海应用案例

### 场景1：跨平台评价标签统一化

**业务问题**
母婴出海商家在 Amazon、Shopee、TikTok Shop、独立站等多平台销售，每个平台有自己的评价体系和标签。运营团队需要人工阅读评价来汇总产品问题，耗时且标准不统一。一个纸尿裤的"侧漏"问题，在 Amazon 可能被描述为 "leaks at night"，在 Shopee 是 "bocor malam"（印尼语），运营团队需要跨语言、跨平台统一追踪。

**数据要求**
| 数据源 | 字段 | 格式 |
|--------|------|------|
| Amazon 评价 | rating, text, date, asin | JSON/CSV |
| Shopee 评价 | rating, comment, item_id | API/CSV |
| TikTok 评价 | content, create_time | API/JSON |
| 客服工单 | content, category, solved | 数据库 |

**预期产出**
- 统一标签看板：各平台评价按 L1-L4 标签聚合，实时显示 "质量-尺码偏差-偏大" 问题在 Amazon 有 23 条、Shopee 有 17 条
- 新标签预警：自动发现 "腰贴 adhesive 过敏" 这一新痛点，运营尚未关注
- 趋势追踪："物流-清关延迟" 标签占比从 3% 上升至 12%，触发预警

**业务价值**
- 运营效率：人工阅读评价时间从每人每天 4 小时降至 30 分钟（仅审核自动标签）
- 问题发现速度：新痛点发现周期从 2-4 周缩短至 3-7 天
- 跨平台协同：统一标签语言，产品、供应链、客服团队使用同一套问题描述

### 场景2：季节性产品标签自进化

**业务问题**
母婴产品有强季节性（如夏季防蚊、冬季保暖）。每年季节更替时，新一批消费者反馈会出现去年没有的标签（如"驱蚊效果持续时间"、"保暖内衣起球"）。静态标签体系无法覆盖这些季节性新问题，导致运营漏掉关键反馈。

**数据要求**
- 历史评价数据：至少 2 个完整季节周期
- 实时评价流：每日新增评价接入系统
- 产品属性：品类、上市时间、季节属性

**预期产出**
- 春季自动发现新标签："防蚊贴粘性不足（出汗后脱落）"
- 冬季自动发现新标签："睡袋拉链夹到宝宝皮肤"
- 季节对比报告：同一品类在不同季节的标签分布变化

**业务价值**
- 产品迭代：新标签直接指导下一季产品改进（如发现"出汗后脱落"→改进 adhesive 配方）
- 库存决策：标签趋势与退货率关联，提前预警需调整库存的产品
- 客服培训：新标签自动同步到客服知识库，提升首次解决率

---

## ③ 代码模板

见 `paper2skills-code/nlp_voc/autotag_self_evolving/` 目录：
- `model.py` — 多任务层级分类器 + 新标签发现
- `label_system.py` — 标签体系管理（CRUD + 层级关系）
- `evolution.py` — 标签进化引擎（触发判断 + 进化动作）
- `label_quality.py` — 标签质量评估（cleanlab 集成 + 降级方案）
- `sentiment_intensity.py` — 情感强度量化（-5~+5 细粒度）
- `kano_bridge.py` — Aspect-Kano 桥接（标签→需求分类→优先级）
- `__init__.py` — 模块导出

运行测试：
```bash
cd paper2skills-code/nlp_voc/autotag_self_evolving
python model.py                # 测试分类器
python label_system.py         # 测试标签体系管理
python evolution.py            # 测试进化引擎
python label_quality.py        # 测试标签质量评估
python sentiment_intensity.py  # 测试情感强度量化
python kano_bridge.py          # 测试 Kano 桥接
```

### 标签质量评估增强

`label_quality.py` 集成 cleanlab 的 Confident Learning，自动检测标签噪声：

```python
from autotag_self_evolving.label_quality import LabelQualityAssessor, assess_autotag_predictions

# 方式1: 直接评估（需 pred_probs）
assessor = LabelQualityAssessor(labels=["尺码偏差", "漏尿", "物流延迟", ...])
report = assessor.assess(texts, given_labels, pred_probs)
assessor.print_report(report, top_k=10)

# 方式2: 与 AutoTag 预测结果直接集成
report = assess_autotag_predictions(
    texts=texts,
    predictions=autotag_results,  # PredictionResult.to_dict() 列表
    labels=all_labels,
    pred_probs=pred_probs,  # 可选，不提供则构造简化概率矩阵
)
```

**检测能力**：
- `label_error`: 给定标签与模型高置信度预测不一致
- `ambiguous`: 模型对所有标签置信度均低，标签可能模糊
- `outlier`: 样本语义偏离训练分布

**降级方案**：cleanlab 未安装时自动切换为规则检测（预测不一致 + 低置信度阈值），确保生产环境可用性。

### 情感强度量化增强

`sentiment_intensity.py` 将粗粒度情感(-1/0/+1)量化为细粒度强度(-5~+5)，直接对接 Kano 分类：

```python
from autotag_self_evolving.sentiment_intensity import SentimentIntensityQuantifier

quantifier = SentimentIntensityQuantifier()
result = quantifier.quantify(text, prediction, pred_probs)  # pred_probs 可选
print(result.intensity)        # -4.2 (极度负面)
print(result.intensity_level)  # "extreme_neg"
```

**双模式设计**：
- **概率模式**：有 pred_probs 时，从 softmax 分布的确定性推导强度
- **规则模式**：无 pred_probs 时，从情感词密度 + 程度副词 + 否定词推导

### Kano 桥接增强

`kano_bridge.py` 将标签 + 情感强度映射到 Kano 需求分类，输出可执行的行动建议：

```python
from autotag_self_evolving.kano_bridge import KanoMapper, bridge_autotag_to_kano

mapper = KanoMapper()
mapping = mapper.map_label("质量", "漏尿", intensity=-4.2)
print(mapping.kano_type)       # "must_be" (基本型)
print(mapping.priority_score)  # 9.3 (最高优先级)
print(mapping.action)          # "立即整改: 漏尿 是基本安全需求..."

# 一键桥接 AutoTag → Kano 报告
report = bridge_autotag_to_kano(predictions, intensity_results)
```

**Kano 五类映射规则**（按 L4 关键词匹配）：
- **基本型**: 漏、过敏、异味、甲醛、安全隐患
- **期望型**: 柔软、透气、吸水量、物流、价格
- **兴奋型**: 赠品、试用装、夜光、智能、环保
- **反向型**: 太厚、太重、香味太浓、包装过度
- **无差异型**: 颜色、款式（默认 fallback）

---

## ④ 技能关联

### 前置技能
- **Skill-Aspect-Based-Sentiment-Analysis** — 方面情感分析是本技能的底层能力，需要先理解如何提取产品属性-情感对
- **Skill-AIPL-VOC-Lifecycle-Tags** — 生命周期标签体系提供了标签设计的方法论，本技能是其自动化实现

### 延伸技能
- **Skill-CSK-Customer-Sentiment-Clustering** — 情感聚类可用于新标签发现的预处理步骤，两者可组合使用
- **Skill-TopicImpact-观点单元画像抽取** — TopicImpact 抽取的观点单元可作为本技能的输入文本单元

### 可组合
- **Skill-Kano-需求分类与优先级** + 本技能：自动标签产出后，用 Kano 模型对标签对应的需求进行分类和优先级排序
- **Skill-iReFeed-需求优先级排序** + 本技能：标签体系作为输入，自动计算各标签对应需求的优先级得分
- **Skill-GPLR-人群标签生成** + 本技能：VOC 标签与用户行为标签交叉，生成更精准的人群画像
- **cleanlab 标签质量评估** + 本技能：AutoTag 产出的标签通过 `label_quality.py` 自动检测噪声标签，问题样本回流至人工审核或 LLM 重标注，形成"生产→评估→修正"闭环

---

## ⑤ 商业价值评估

### ROI 预估

| 指标 | 现状（人工） | 实施后（自动+审核） | 节省/提升 |
|------|-------------|-------------------|----------|
| 单条反馈处理时间 | 2-3 分钟 | 0.1 秒（模型）+ 10 秒（审核） | **95%↓** |
| 新痛点发现周期 | 2-4 周 | 3-7 天 | **80%↓** |
| 标签覆盖率 | 60-70% | 85-92% | **30%↑** |
| 跨平台标签一致性 | 低（各平台独立） | 高（统一标签树） | 质的变化 |
| 运营人力（10万条/月） | 3-4 人全职 | 0.5 人审核 | **人力成本 85%↓** |

**年化收益估算**（以月处理 10 万条反馈的出海商家为例）：
- 人力节省：3.5 人 × 15 万/年 = **52.5 万/年**
- 问题发现提前：退货率降低 1-2% × 年 GMV（假设 5000 万）= **50-100 万/年**
- 产品迭代提速：季节性产品改进周期缩短，预估增收 **30-50 万/年**
- **合计年化收益：130-200 万**

### 实施难度

⭐⭐⭐⭐☆（4/5星）

**难点分析**：
1. 多语言处理（出海涉及英/西/印尼/泰等）需要多语言模型支持
2. 初始标签体系设计需要业务专家深度参与（2-3 周）
3. LLM 调用成本（新标签发现阶段）需控制频率
4. 标签进化的人工审核流程需要运营配合

### 优先级评分

⭐⭐⭐⭐⭐（5/5星）

**评估依据**：
- **痛点强度**: VOC 是母婴出海最核心的决策输入之一，当前人工处理瓶颈明显
- **可落地性**: 技术方案成熟（InsightNet 已在 Amazon 生产环境验证）
- **复利效应**: 标签体系越用越准，数据飞轮效应强
- **组合价值**: 与现有 5+ 个 Skill 可组合，产生协同价值

---

## 附录：论文信息

| 项目 | 内容 |
|------|------|
| **主论文** | InsightNet: Structured Insight Mining from Customer Feedback |
| **作者** | Sandeep Sricharan Mukku, Manan Soni, Jitenkumar Rana, et al. (Amazon Science) |
| **arXiv** | 2405.07195 |
| **年份** | 2024 |
| **核心指标** | Topic Classification F1 = 0.85（较此前最优提升 11%） |
| **验证规模** | 43 个品类、2200+ 产品类型、1200+ 标签 |
| **开源状态** | 未开源，但方法可复现 |

**辅助参考论文**：
- SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue (美团, arXiv:2602.03548, 2026) — 自进化框架
- Can LLMs Extract Customer Needs as Well as Professional Analysts? (MIT, arXiv:2503.01870, 2025) — LLM 标注方法
