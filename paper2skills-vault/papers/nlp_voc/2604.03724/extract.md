# 萃取记录: StaR - 观点语句排序

## 论文信息
- **arXiv ID**: 2604.03724
- **标题**: Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation
- **萃取日期**: 2026-04-14
- **领域**: 07-NLP-VOC

## 核心算法提炼

### 算法名称
StaR (Statement-level Ranking) Pipeline

### 核心思想
将可解释推荐从"生成自由文本段落"重构为"排序候选语句"。通过提取满足三要素（explanatory解释性、atomic原子性、unique唯一性）的statements并排序，从根本上消除LLM幻觉，实现可标准化评估的细粒度解释。

### 数学直觉
1. **语句质量三要素**：
   - Explanatoriness: statement必须描述影响用户体验的产品事实
   - Atomicity: 一个statement只表达一个aspect的一个观点
   - Uniqueness: 通过语义聚类合并同义paraphrases，每个簇只保留canonical representative
2. **排序评估**：用经典IR指标评估statement ranking质量
   - P@k, R@k, NDCG@k
   - relevance_j = δ(π_ui(j) ∈ S_ui)，判断排名j的statement是否在ground-truth集合中
3. **两阶段提取**：Candidate Extraction + Verification，先提取再过滤，保证statement质量
4. **语义聚类三步法**：ANN近邻搜索 → pairwise cross-encoder过滤 → 连通分量+cohesion refinement

### 关键假设
- 用户评论中包含足够的解释性证据
- 存在可用的dense embedding模型和cross-encoder用于语义匹配
- 对于item-level ranking，历史交互数据能提供足够的item-specific signal

## 业务适配设计

### 场景1：Momcozy暖奶器Amazon评论atomic观点提取与排序
从美国、德国市场Momcozy暖奶器评论中提取"加热均匀""温控精准""操作简单"等原子观点，经语义去重后按市场相关性排序，识别各市场最突出的用户关注点。

### 场景2：跨市场属性偏好对比基座
将StaR pipeline作为跨市场对比的前置步骤：先对不同国家的评论分别跑statement extraction和ranking，再对比各市场Top-K statements的差异，支撑选品和营销策略制定。

## 代码实现
- **路径**: `paper2skills-code/nlp_voc/star_statement_ranking/model.py`
- **设计**: 用规则模拟LLM extraction/verification，用TF-IDF向量+余弦相似度模拟语义聚类
- **验证状态**: ✅ 通过
