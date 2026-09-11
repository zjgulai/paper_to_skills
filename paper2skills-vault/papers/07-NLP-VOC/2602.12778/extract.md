# Paper Extract: BERT-MoE for ABSA

## 论文信息
- **arXiv ID**: 2602.12778
- **标题**: Aspect-Based Sentiment Analysis for Future Tourism Experiences: A BERT-MoE Framework for Persian User Reviews
- **日期**: 2026-02-13
- **摘要**: 提出了 BERT-MoE 混合模型用于方面情感分析，在旅游评论数据集上达到 90.6% F1-score，GPU 功耗降低 39%

## 核心贡献
1. **BERT-MoE 架构**: 混合专家模型优化 BERT 情感分类
2. **动态路由**: 只激活 top-k 专家，大幅降低计算量
3. **效率提升**: GPU 功耗降低 39%，F1-score 提升至 90.6%

## 业务价值
- 适合多语言电商场景
- 资源效率高，适合小团队
- 与第一个 VOC 技能形成梯度（基础版 → 高效版）

## 生成 Skill
- Skill-BERT-MoE高效方面情感分析.md