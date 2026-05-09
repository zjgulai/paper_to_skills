# Skill Card: 弱监督自动标注 — LLM 生成标注程序
# ALCHEmist: Automated Labeling via Program Generation

**论文来源**: The ALCHEmist: Automated Labeling 500x CHEaper Than LLM Data Annotators (NeurIPS 2024 Spotlight, arXiv:2407.11004)
**理论基础**: 弱监督学习 (Weak Supervision) + LLM 程序合成 (Program Synthesis) + 数据编程 (Data Programming)
**适用领域**: 文本分类数据标注、可审计标注规则生成、大规模语料自动标注、标注成本敏感场景

---

## ① 算法原理

### 核心思想

传统"LLM-as-Annotator"让大模型逐条标注文本——虽然比人工快，但每条都要调一次 API，成本高、不可复现、无法审计。ALCHEmist 的核心洞察是：**与其让 LLM 回答"这条评论是什么标签"，不如让 LLM 写一段"能自动判断标签"的程序**。一次 LLM 调用生成一个标注程序（label function），这个程序可以在本地无限次运行，零后续成本，还能被审查和修改。

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    阶段1: LLM 生成标注程序                     │
│                                                             │
│   LLM Prompt: "写一个 Python 函数，判断评论是否关于尺码问题"   │
│        ↓                                                    │
│   输出: def lf_size_issue(text):                            │
│            return "尺码" in text or "size" in text.lower()  │
│                                                             │
│   成本: 1 次 LLM API 调用 (~$0.01)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    阶段2: 程序批量标注                         │
│                                                             │
│   for text in unlabeled_data:                               │
│       label = lf_size_issue(text)  # 本地执行，零 API 成本    │
│                                                             │
│   成本: $0 (程序本地运行)                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    阶段3: 多程序投票聚合                       │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│   │ 程序A    │  │ 程序B    │  │ 程序C    │                │
│   │ 关键词匹配│  │ 正则规则  │  │ 语义模式  │                │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│        │             │             │                       │
│        └─────────────┼─────────────┘                       │
│                      ↓                                      │
│              多数投票 → 最终标签                            │
│                                                             │
│   成本: $0 (本地聚合)                                       │
└─────────────────────────────────────────────────────────────┘
```

### Label Function (标注程序)

ALCHEmist 的核心是 **Label Function**——一段轻量代码，输入文本，输出标签或弃权（abstain）。

```python
def lf_size_issue(text: str) -> str | None:
    """判断评论是否关于尺码问题"""
    keywords = ["尺码", "大小", "偏大", "偏小", "size", "sizing", "fit"]
    if any(kw in text.lower() for kw in keywords):
        return "尺码偏差"
    return None  # 弃权：不确定
```

**弃权机制 (Abstain)** 是关键设计——当程序无法确定时返回 None，不参与投票，避免硬猜导致噪声。

### LLM 程序生成策略

ALCHEmist 用 LLM 生成 label functions，而非直接标注：

**1. 任务描述**: 告诉 LLM 标签定义和示例
   ```
   标签: "尺码偏差" — 用户反馈产品尺码不合适
   示例正例: "纸尿裤太小了，宝宝穿上松松垮垮"
   示例反例: "质量很好，就是物流慢"
   请写一个 Python 函数，输入文本，返回是否匹配该标签。
   ```

**2. 程序生成**: LLM 输出可执行的 Python 函数

**3. 程序验证**: 在少量验证集上测试函数准确率，过滤低质量程序

**4. 程序迭代**: 对覆盖不足的数据子集，要求 LLM 生成补充程序

### 数据编程聚合 (Data Programming)

多个 label functions 对同一样本可能给出不同标签。ALCHEmist 使用 **多数投票 + 置信度加权**：

$$\hat{y} = \arg\max_{y} \sum_{i=1}^{M} \mathbb{1}[\text{lf}_i(x) = y] \cdot w_i$$

其中 $w_i$ 是第 $i$ 个 label function 的历史准确率权重。

### 与传统方法的对比

| 方法 | 每千条成本 | 可审计 | 可复用 | 准确率 |
|------|-----------|--------|--------|--------|
| 人工标注 | $500-2000 | ✓ | ✗ | 85-90% |
| LLM 逐条标注 | $10-30 | ✗ | ✗ | 88-93% |
| **ALCHEmist** | **$0.02-0.05** | **✓** | **✓** | **85-92%** |

### 关键假设

1. **LLM 程序生成能力**: LLM 能根据少量示例写出合理的文本判断程序（文本分类场景通常满足）
2. **标签可规则化**: 标签有明确的文本模式可被程序捕获（如关键词、正则、简单语义）
3. **多程序覆盖**: 每个标签需要 3-10 个互补的 label functions 才能稳定覆盖
4. **验证集存在**: 需要少量人工标注的验证集来筛选和加权程序
5. **程序可本地执行**: 生成的代码在安全沙箱中运行，避免注入风险

---

## ② 母婴出海应用案例

### 场景1：新标签快速生产可审计标注规则

**业务问题**

AutoTag 进化引擎发现新痛点"腰贴 adhesive 过敏"，需要：
1. 快速生产 500 条标注样本训练分类器
2. 运营团队需要理解"为什么这些评论被标为过敏"（可审计性）
3. 下季度产品改进后，标签定义可能微调，标注规则需要可修改

传统 LLM 逐条标注的问题：
- 500 条 × $0.01 = $5，成本尚可，但**无法解释**为什么某条被标为过敏
- 标签定义微调后，500 条全部要重新标注

**ALCHEmist 方案**

1. **LLM 生成 5 个 label functions**（1 次 API 调用，~$0.02）：
   ```python
   # lf1: 关键词匹配
   def lf_allergy_keyword(text):
       if any(kw in text for kw in ["过敏", "红疹", "发红", "allergy", "rash"]):
           return "过敏反应"
       return None

   # lf2: 身体部位 + 颜色变化
   def lf_skin_reaction(text):
       body_parts = ["皮肤", "屁股", "大腿", "腰部", "肚子"]
       reactions = ["红", "疹", "肿", "痒"]
       if any(b in text for b in body_parts) and any(r in text for r in reactions):
           return "过敏反应"
       return None

   # lf3: 排除非过敏场景（否定规则）
   def lf_not_allergy(text):
       if any(kw in text for kw in ["勒", "紧", "小", "摩擦"]):
           return None  # 弃权：可能是尺码问题而非过敏
       return None
   ```

2. **本地批量标注**: 5 个程序在 5000 条评论上本地运行，零 API 成本

3. **投票聚合**: 3 个程序一致 → 高置信度；2:1 → 中置信度；分歧 → 送人工

**预期产出**

- 5000 条评论的自动标注结果，其中 ~4000 条高置信度直接采用
- 5 个可审计的标注程序（运营人员可读、可修改）
- 当标签定义微调时（如"发红超过 3 天"才算过敏），直接修改程序即可

**业务价值**

- **成本**: 5 次 LLM 调用 ≈ $0.05 vs LLM 逐条 5000 × $0.01 = $50，**成本降低 99%**
- **可审计**: 运营人员看到"这条被标为过敏是因为 lf2 匹配了'皮肤'+'红'"，信任度提升
- **可维护**: 标签定义变化时，修改程序即可，无需重新标注

### 场景2：跨境多平台评论统一标注规则库

**业务问题**

母婴出海商家在 Amazon US、Amazon DE、Shopee ID、乐天日本销售。每个平台的评论语言不同，但产品问题本质相同。传统方案是每个语言单独标注团队，或用一个 LLM 逐条处理多语言文本。

ALCHEmist 方案：**为每个标签生成语言无关的标注程序集**。

**多语言 label function 设计**

```python
# 尺码问题 — 跨语言通用
def lf_size_issue_multilingual(text):
    # 中文
    if any(kw in text for kw in ["尺码", "大小", "偏大", "偏小"]):
        return "尺码偏差"
    # 英语
    if any(kw in text.lower() for kw in ["size", "sizing", "too big", "too small", "tight", "loose"]):
        return "尺码偏差"
    # 德语
    if any(kw in text.lower() for kw in ["größe", "grösse", "zu groß", "zu klein"]):
        return "尺码偏差"
    # 日语
    if any(kw in text for kw in ["サイズ", "大きい", "小さい", "きつい"]):
        return "尺码偏差"
    return None
```

**ALCHEmist 流程**

1. 用 LLM 为每个标签生成**多语言合一**的 label function
2. 在各平台评论上统一运行
3. 输出统一的中文标签，便于跨市场对比

**业务价值**

- 一套标注规则覆盖所有语言，维护成本低
- 跨市场问题对比成为可能（如"漏尿"在美/日/德的发生率）
- 新市场扩展时，只需在 label function 中新增该语言关键词

---

## ③ 代码模板

见 `paper2skills-code/nlp_voc/alchemist_weak_supervision/` 目录：
- `label_function.py` — Label Function 定义 + 生成器接口
- `program_generator.py` — LLM 程序生成器（模拟模式）
- `aggregator.py` — 多程序投票聚合 + 置信度估计
- `pipeline.py` — 完整流水线：生成 → 验证 → 标注 → 聚合
- `__init__.py` — 模块导出

运行测试：
```bash
cd paper2skills-code/nlp_voc/alchemist_weak_supervision
python label_function.py     # 测试 label function 执行
python program_generator.py  # 测试程序生成
python aggregator.py         # 测试投票聚合
python pipeline.py           # 测试完整流水线
```

---

## ④ 技能关联

### 前置技能
- **Skill-AutoTag-SelfEvolving-Label-System** — 提供标签体系定义，ALCHEmist 为每个标签生产标注程序
- **Skill-Aspect-Based-Sentiment-Analysis** — ABSA 提供标签边界划分的方法论

### 延伸技能
- **Skill-Active-Learning-Annotation** — 主动学习是 ALCHEmist 的互补方案：ALCHEmist 适合规则可捕获的标签，主动学习适合需要模型理解的复杂标签

### 可组合
- **ALCHEmist + AutoTag**: AutoTag 发现新标签 → ALCHEmist 生成标注程序 → 批量生产训练数据 → 训练 AutoTag 分类器
- **ALCHEmist + cleanlab**: ALCHEmist 产出 probabilistic labels，cleanlab 检测其中程序错误导致的系统性噪声

---

## ⑤ 商业价值评估

### ROI 预估

| 指标 | 人工标注 | LLM 逐条 | ALCHEmist |
|------|---------|---------|-----------|
| 5000 条标注成本 | $2500 | $30-50 | **$0.05** |
| 可审计性 | ✓ | ✗ | **✓** |
| 标签变更后重标成本 | $2500 | $30-50 | **$0** |
| 新语言扩展成本 | 新团队 | $30-50 | **修改程序** |

**年化收益估算**（月处理 5 万条评论、8 个新标签的商家）：

| 成本项 | 纯人工 | ALCHEmist | 节省 |
|--------|--------|-----------|------|
| 日常标注 | 5万 × 12 × $0.5 = **30万** | 程序生成($10) + 验证($500) = **510** | **~30万/年** |
| 新标签标注 | 8 × 500 × $0.5 × 12 = **2.4万** | 8 × $0.02 × 12 = **$2** | **~2.4万/年** |
| **合计** | **32.4万/年** | **~510/年** | **~32万/年** |

> 注：ALCHEmist 日常标注假设已有稳定 label functions，仅需少量验证和程序迭代。

### 实施难度

⭐⭐⭐☆☆（3/5星）

**难点分析**：
1. LLM 生成的程序需要语法校验和安全沙箱执行
2. 复杂语义标签（如讽刺、隐含抱怨）难以用规则捕获
3. 多程序投票阈值需调参
4. 程序库需要版本管理（标签定义变化时程序如何演进）

**降低难度的路径**：
- 从关键词/正则类标签开始（最易规则化）
- 先用 ALCHEmist 覆盖 70-80% 的"简单标签"，复杂标签留给主动学习或 LLM 逐条

### 优先级评分

⭐⭐⭐⭐⭐（5/5星）

**评估依据**：
- **痛点强度**: 标注成本是 NLP 落地的最大障碍，ALCHEmist 的 500x 成本削减是质的飞跃
- **独特价值**: 可审计性和可复用性是其他方法无法提供的
- **可落地性**: 方法成熟（NeurIPS Spotlight，已有开源），不依赖复杂训练
- **通用性**: 任何有明确文本模式的分类任务都可应用

---

## 附录：论文信息

| 项目 | 内容 |
|------|------|
| **主论文** | The ALCHEmist: Automated Labeling 500x CHEaper Than LLM Data Annotators |
| **作者** | Tzu-Heng Huang, Catherine Cao, Vaishnavi Bhargava, Frederic Sala |
| **单位** | University of Wisconsin-Madison |
| **会议** | NeurIPS 2024 Spotlight |
| **arXiv** | [2407.11004](https://arxiv.org/abs/2407.11004) |
| **核心指标** | 成本降低 500x，性能提升 12.9%（相对直接 LLM 标注） |
| **开源状态** | **开源** [github.com/SprocketLab/Alchemist](https://github.com/SprocketLab/Alchemist) |

**相关论文**：
- Snorkel: Rapid Training Data Creation with Weak Supervision (Stanford, VLDB 2018) — 数据编程框架
- ScriptoriumWS (ICLR Workshop 2023) — 同一团队的代码生成辅助弱监督
