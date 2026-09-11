# 论文阅读笔记: End-to-End Aspect-Guided Review Summarization at Scale

## 基本信息
- **arXiv ID**: 2509.26103
- **标题**: End-to-End Aspect-Guided Review Summarization at Scale
- **作者**: Ilya Boytsov, Vinny DeGenova, Mikhail Balyasin, Joseph Walt, Caitlin Eusden, Marie-Claire Rochat, Margaret Pierson
- **发表时间**: 2025-09-30
- **机构**: Wayfair

## 核心贡献
提出一个工业级可扩展的LLM-based pipeline，将ABSA与引导式摘要结合，为电商平台生成简洁、可解释的产品评论摘要。该系统已在Wayfair平台通过大规模在线A/B测试验证，并开源了1180万匿名评论的数据集。

## 算法流程（4个阶段）
1. **Aspect Extraction**: 用结构化prompt引导LLM从单条评论中提取最多5个aspect-sentiment对，输出JSON格式
2. **Aspect Consolidation**: 将细粒度、词汇变体多的aspect映射到canonical forms。以95th percentile（约30次）为阈值，高频保留，低频合并到更高级别概念
3. **Aspect-based Review Selection**: 为每个产品选Top 5高频aspects，对每个aspect-sentiment pair按频率加权采样代表性评论，每产品输入上限200条评论
4. **Aspect-Guided Summarization**: 用结构化prompt（包含consolidated aspects + selected reviews）引导LLM生成300-500字符的产品级摘要

## 关键设计
- **Caching机制**: aspect consolidation的映射表会被缓存复用，避免对新数据重复进行consolidation
- **实时部署**: 新品累计≥10条评论时触发pipeline；已有摘要的产品在评论增长≥10%时自动刷新
- **模型无关**: 使用Gemini 1.5 Flash，但pipeline可替换任意LLM backbone

## 实验结果
- **离线评估**: 341个产品，50,000条评论。84%的摘要无错误，11%有minor问题，5%有major问题
- **在线A/B测试**: 2025年3月，493,208个产品，2,329个品类，持续3周
  - Add-to-Cart Rate (ATCR): +0.3% (p=0.10)
  - Conversion Rate (CVR): +0.5%
  - Bounce Rate: -0.13%
  - 无显著负面影响

## 数据集/代码可用性
- **开源数据集**: HuggingFace `leBoytsov/review-summaries-68dab02e7b6a5bc8e29e81fa`
  - 1180万匿名评论
  - 92,000个产品
  - 包含extracted aspects和generated summaries

## 母婴出海适配点
- 直接适用于Momcozy多平台（Amazon/Wayfair/Temu）评论汇总，生成季度产品摘要
- 通过aspect-guided方式确保摘要100% grounded in真实评论，避免LLM幻觉
- 与StaR（观点语句排序）和MAA（行动建议生成）形成完整链路
