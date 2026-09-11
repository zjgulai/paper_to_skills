# 萃取记录: iReFeed - 互联需求优先级排序

## 论文信息

- **arXiv ID**: 2603.28677
- **标题**: Enhancing User-Feedback Driven Requirements Prioritization
- **作者**: Aurek Chattopadhyay, Nan Niu, Hui Liu, Jianzhang Zhang
- **萃取日期**: 2026-04-16
- **领域**: 07-NLP-VOC

## 核心算法提炼

### 算法名称
iReFeed (interconnected ReFeed) Pipeline

### 核心思想
将用户反馈驱动的需求优先级排序从"单需求独立评估"升级为"需求簇互联评估"。通过 topic modeling 把用户反馈聚类为主题，再将 candidate requirements 映射到主题簇中，在簇级别关联反馈并计算优先级。核心洞察是：**需求的 interconnectedness 显著影响优先级决策**，相关需求应该一起考虑而非孤立评估。

### 数学直觉

1. **需求簇级反馈关联**
   传统 ReFeed 将反馈 $F$ 与单个需求 $r$ 关联：
   $$P_r = \frac{\sum_{i=1}^{|F|} [sim(r, F[i]) \times (neg_{F[i]} + pos_{F[i]} + int_{F[i]})]}{|F|}$$

   iReFeed 改为将反馈与需求簇 $F_C$ 关联：
   $$P_r = \frac{\sum_{i=1}^{|F_C|} [sim(r, F_C[i]) \times (neg_{F_C[i]} + pos_{F_C[i]} + int_{F_C[i]})]}{|F_C|}$$

2. **簇内聚性增强 (LDA-C / BERTopic-C)**
   为奖励内部一致的需求簇，引入 coherence factor：
   $$\alpha(F_C) = \min(1, \text{average pairwise similarity of } \forall r_i, r_j \in F_C)$$
   增强后的优先级：
   $$P_r = \frac{\sum_{i=1}^{|F_C|} [\alpha(F_C) \times sim(r, F_C[i]) \times (neg_{F_C[i]} + pos_{F_C[i]} + int_{F_C[i]})]}{|F_C|}$$

3. **依赖价值 D-value**
   利用 LLM (ChatGPT) 自动发现需求间的 "requires" 关系。对于需求 $i$，其依赖价值：
   $$D\text{-}value_i = \frac{count_i}{|\mathcal{D}|}$$
   其中 $count_i$ 是需求 $i$ 作为 "requires" 关系右侧（被依赖方）出现的次数，$|\mathcal{D}|$ 是发现的总关系数。

4. **NSGA-II 三目标优化**
   将 D-value 作为第三目标集成到 Next Release Problem (NRP) 的 NSGA-II 求解中：
   - Maximize: 利益相关者价值总和
   - Minimize: 开发资源成本
   - Maximize: 需求依赖价值 (D-value)

### 关键假设
- 用户反馈量足够支撑有意义的 topic modeling（论文中每个应用有数万条评论）
- 需求可以被自然地按主题聚类（跨市场/跨平台的需求具有一定主题重叠）
- 存在可用的 LLM 用于发现 "requires" 关系（论文使用 ChatGPT 4.5）
- 需求的价值和开发成本可以被量化评估

## 业务适配设计

### 场景1: Momcozy 季度产品功能优先级排序
Momcozy 每季度从 Amazon US/DE/Wayfair 收集用户评论，结合内部产品规划产生 30-50 个 candidate 功能需求（如"降噪改进""APP远程控制""新配件兼容"）。使用 iReFeed：
1. 对用户评论做 topic modeling，识别 15-20 个主题簇
2. 将 candidate 功能映射到主题簇
3. 计算每个功能的用户反馈驱动优先级
4. 用 LLM 发现功能间的依赖关系（如"APP升级" requires "蓝牙模块更新"）
5. 输入 NSGA-II 优化，在价值、成本、依赖关系约束下输出最优季度功能组合

### 场景2: 跨市场差异化需求整合
美国市场用户高频反馈"便携性"，德国市场高频反馈"静音认证"，中国市场关注"清洗方便"。iReFeed 可以将这些跨市场的 candidate 需求聚类到统一的 topic 空间中，识别全球通用需求 vs 区域特定需求，优先排序那些能同时覆盖多个市场的功能改进。

## 代码实现

- **路径**: `paper2skills-code/nlp_voc/irefeed_priority_ranking/model.py`
- **设计**: 
  - 使用 sklearn LDA 模拟 topic modeling
  - 用 TF-IDF + cosine similarity 模拟反馈-需求关联
  - 用规则/启发式模拟 sentiment、intention 提取
  - 用模板方法模拟 LLM 的 "requires" 关系发现
  - 用 pymoo 库实现 NSGA-II 三目标优化
- **验证状态**: 待验证
