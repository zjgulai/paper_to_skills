# 论文阅读笔记: Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation

## 基本信息
- **arXiv ID**: 2604.03724
- **标题**: Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation
- **作者**: Ben Kabongo, Arthur Satouf, Vincent Guigue
- **发表时间**: 2026-04-04

## 核心贡献
提出将可解释推荐从"生成段落"转变为"排序语句"（rank, don't generate）。通过从评论中提取explanatory（解释性）、atomic（原子性）、unique（唯一性）的statements并排序，从根本上消除LLM幻觉，并实现标准化的细粒度评估。

## 算法流程
1. **Statement Candidate Extraction**: LLM从原始评论中提取候选statements并标注sentiment
2. **Statement Verification**: 第二个LLM过滤non-explanatory（非解释性）、non-atomic（非原子性）、redundant（冗余）的候选
3. **Statement Clustering**: 三阶段语义聚类去重
   - Approximate nearest-neighbor search (用dense embedding)
   - Pairwise filtering (cross-encoder重评估，只保留高置信匹配)
   - Refinement (连通分量形成初始簇，再按cohesion split低内聚簇)
4. **Statement Ranking**: global-level（全库statements排序）和item-level（限定目标商品的statements排序），用Precision@k, Recall@k, NDCG@k评估

## 关键公式
- NDCG@k: 标准信息检索排序指标
-  relevance: rel_j = δ(π_ui(j) ∈ S_ui)，判断排名j的statement是否属于ground-truth

## 实验设计
- 数据集：Amazon Reviews 2014（Toys, Clothes, Beauty, Sports四个品类）
- 自建StaR benchmark：115K-294K interactions，718K-1.3M user-item-statement triplets
- 发现：简单 popularity baselines在global-level很有竞争力，但在item-level上SOTA模型反而不如popularity baselines，暴露了个性化解释排序的持久性局限

## 数据集/代码可用性
- 公开benchmark数据集（Amazon Reviews衍生）
- GitHub: https://github.com/BenKabongo25/Statement_Ranking_Explainable_Recommendation

## 母婴出海适配点
- 可将Momcozy各品类评论提取为atomic statements（如"加热均匀""操作简单""清洗方便"），再做跨市场排序对比
- 消除LLM幻觉对商业决策的干扰，确保每个建议都有真实评论支撑
- 与MAA技能完美衔接：StaR提取高质量statements → MAA生成行动建议
