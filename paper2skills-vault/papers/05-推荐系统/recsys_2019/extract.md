# 论文信息

## Paper 6: Matrix Factorization for Recommendation

### 论文信息
- **标题**: Matrix Factorization Techniques for Recommender Systems
- **领域**: 推荐系统 / 协同过滤

### 核心算法
1. **SVD**: 奇异值分解
2. **ALS**: 交替最小二乘法
3. **BPR**: Bayesian Personalized Ranking

### 关键公式
- 评分预测: r_ui = q_i^T p_u + b_u + b_i
- 损失函数: min sum(r_ui - q_i^T p_u)^2 + λ(||p_u||^2 + ||q_i||^2)
- 矩阵分解: R ≈ P × Q^T

---

# 论文摘要

We present matrix factorization methods for recommender systems. The key insight is to represent users and items as latent vectors in a low-dimensional space, where the dot product between user and item vectors predicts the rating. We introduce several methods including SVD, ALS, and BPR, showing that matrix factorization achieves state-of-the-art performance on benchmark datasets. Our methods are particularly effective for e-commerce platforms with explicit or implicit feedback.
