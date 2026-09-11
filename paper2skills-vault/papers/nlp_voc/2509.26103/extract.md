# 萃取记录: AGRS - 属性引导评论摘要

## 论文信息
- **arXiv ID**: 2509.26103
- **标题**: End-to-End Aspect-Guided Review Summarization at Scale
- **萃取日期**: 2026-04-14
- **领域**: 07-NLP-VOC

## 核心算法提炼

### 算法名称
AGRS (Aspect-Guided Review Summarization) Pipeline

### 核心思想
将大规模LLM摘要从"无约束自由生成"转变为"属性引导的结构化生成"。通过ABSA提取aspect-sentiment对、consolidation去噪归一、代表性评论采样、结构化prompt引导，生成100%基于真实反馈的产品摘要，从根本上避免幻觉。

### 数学直觉
1. **Aspect频率筛选**：对每个产品统计consolidated aspects的出现频次，取Top-K（论文中为Top 5）作为摘要的核心骨架。
   $$\text{TopAspects} = \arg\max_{A' \subset A, |A'|=K} \sum_{a \in A'} \text{freq}(a)$$
2. **代表性评论采样**：对每个aspect-sentiment pair按频率加权随机采样，既保证不同观点的平衡覆盖，又将输入上下文限制在可控长度内（≤200条评论）。
3. **结构化摘要生成**：将aspects和reviews以固定模板组织进prompt，约束LLM的输出空间和事实依据，使生成长度稳定在300-500字符。

### 关键假设
- 评论量足够支撑有意义的aspect统计（论文中新品触发阈值为≥10条评论）
- 存在可用的LLM用于aspect提取和摘要生成
- aspect consolidation的canonical映射可以被有效缓存和复用

## 业务适配设计

### 场景1：Momcozy消毒器双平台季度摘要
汇总Amazon US和Amazon DE的Momcozy紫外线消毒器季度评论，提取并consolidate aspect-sentiment对，生成季度产品摘要，直接用于管理层季度复盘和供应链/产品团队的迭代输入。

### 场景2：新品上市后的快速评论监控
Momcozy新品暖奶器上市后，当评论累积超过10条时自动触发AGRS pipeline，每周生成一次aspect-guided摘要，帮助运营团队快速捕捉早期用户反馈热点。

## 代码实现
- **路径**: `paper2skills-code/nlp_voc/agrs_review_summarization/model.py`
- **设计**: 用规则/启发式模拟LLM aspect extraction，用字典映射实现aspect consolidation，用模板方法生成guided summary
- **验证状态**: ✅ 通过
