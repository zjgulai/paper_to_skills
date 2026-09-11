# Skill Card: Matrix Factorization for Recommendation (矩阵分解推荐)

---

## ① 算法原理

### 核心思想
矩阵分解解决的核心问题是：**根据用户历史行为，预测用户对未交互商品的兴趣程度**。与简单统计方法不同，矩阵分解将用户和商品映射到低维隐向量空间，通过向量内积预测评分或购买概率。

### 数学直觉

**评分预测模型**：
$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$$

- $\mu$：全局平均评分
- $b_u$：用户偏差（某些用户打分偏高/偏低）
- $b_i$：商品偏差（某些商品天然好评/差评）
- $q_i$：商品隐向量
- $p_u$：用户隐向量
- $q_i^T p_u$：用户-商品匹配度

**优化目标**：
$$\min \sum_{(u,i)} (r_{ui} - \hat{r}_{ui})^2 + \lambda(||p_u||^2 + ||q_i||^2)$$

- L2 正则化防止过拟合

**ALS（交替最小二乘法）**：
1. 固定用户矩阵 P，优化商品矩阵 Q
2. 固定商品矩阵 Q，优化用户矩阵 P
3. 交替迭代直至收敛

### 关键假设
- **独立性假设**：用户对不同商品的评分相互独立
- **低秩假设**：用户-商品交互矩阵可由低维隐向量近似
- **平稳性**：用户偏好不随时间剧烈变化

---

## ② 吸奶器出海应用案例

### 场景一：吸奶器配件复购推荐

**业务问题**：
购买吸奶器的妈妈用户（如定期更换配件：喇叭罩、鸭嘴阀、储奶袋）需要"下次买什么"的推荐。传统的热销榜单无法满足个性化需求，需要基于用户历史购买记录推荐配件。

**数据要求**：
- 用户-商品交互矩阵：购买记录（隐式反馈）
- 商品特征：品类（喇叭罩、鸭嘴阀、储奶袋）、品牌、价格带
- 用户特征：历史购买品类、购买频次

**预期产出**：
- 每个用户的商品推荐列表（top N）
- 推荐理由（"买了XX的用户也买了YY"）
- 推荐分数

**业务价值**：
- 复购率提升 15-25%
- 客单价提升 5-10%
- 用户体验提升

---

### 场景二：新品冷启动推荐

**业务问题**：
新品上架时没有历史销量，需要决定推荐给哪些用户。使用矩阵分解可以：
- 找到与新品相似的已有商品
- 将新品推荐给购买过相似商品的用户

**数据要求**：
- 新品特征向量
- 历史商品隐向量
- 用户隐向量

**预期产出**：
- 新品潜在用户列表
- 推荐优先级
- 曝光策略

**业务价值**：
- 新品推广效率提升 30%+
- 库存周转提升
- 新品销量占比提升

---

## ③ 代码模板

```python
"""
Matrix Factorization for Recommendation
用于母婴出海电商商品推荐
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class MatrixFactorization:
    """矩阵分解推荐算法"""

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        """
        初始化

        Args:
            n_factors: 隐向量维度
            n_epochs: 迭代轮数
            lr: 学习率
            reg: 正则化系数
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def fit(self, interactions):
        """
        训练模型

        Args:
            interactions: 用户-商品交互矩阵 (scipy sparse matrix)
        """
        self.n_users, self.n_items = interactions.shape

        # 初始化参数
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        # 偏差项
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.mu = interactions.sum() / (interactions > 0).sum()

        # SGD 训练
        interactions = interactions.tocoo()
        ratings = interactions.data
        user_indices = interactions.row
        item_indices = interactions.col

        for epoch in range(self.n_epochs):
            for u, i, r in zip(user_indices, item_indices, ratings):
                # 预测
                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])

                # 误差
                error = r - pred

                # 更新偏差
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])

                # 更新隐向量
                P_u = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u - self.reg * self.Q[i])

            # 打印进度
            if (epoch + 1) % 5 == 0:
                mse = self._compute_mse(interactions)
                print(f"   Epoch {epoch+1}/{self.n_epochs}, MSE: {mse:.4f}")

        return self

    def _compute_mse(self, interactions):
        """计算 MSE"""
        interactions = interactions.tocoo()
        total_error = 0
        for u, i, r in zip(interactions.row, interactions.col, interactions.data):
            pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
            total_error += (r - pred) ** 2
        return np.sqrt(total_error / len(interactions.data))

    def recommend(self, user_id, n_items=10, exclude_known=True):
        """
        为用户推荐商品

        Args:
            user_id: 用户 ID
            n_items: 推荐数量
            exclude_known: 是否排除已知商品

        Returns:
            recommendations: [(item_id, score), ...]
        """
        # 计算用户对所有商品的评分
        scores = self.mu + self.b_u[user_id] + self.b_i + np.dot(self.P[user_id], self.Q.T)

        # 排序
        top_items = np.argsort(scores)[::-1]

        # 排除已知商品
        if exclude_known:
            known_items = set()  # 可以传入已知商品列表
            top_items = [i for i in top_items if i not in known_items]

        return [(i, scores[i]) for i in top_items[:n_items]]

    def get_similar_items(self, item_id, n=5):
        """
        获取相似商品

        Args:
            item_id: 商品 ID
            n: 相似商品数量

        Returns:
            similar_items: [(item_id, similarity), ...]
        """
        # 使用余弦相似度
        item_vector = self.Q[item_id]
        similarities = np.dot(self.Q, item_vector) / (
            np.linalg.norm(self.Q, axis=1) * np.linalg.norm(item_vector) + 1e-8
        )

        # 排除自身
        similarities[item_id] = -1

        top_items = np.argsort(similarities)[::-1][:n]
        return [(i, similarities[i]) for i in top_items]


class ALS:
    """交替最小二乘法（更高效的实现）"""

    def __init__(self, n_factors=50, n_iterations=10, reg=0.1):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg

    def fit(self, interactions):
        """训练 ALS 模型"""
        self.n_users, self.n_items = interactions.shape
        interactions = interactions.tocsc()

        # 初始化
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.mu = interactions.sum() / (interactions > 0).sum()

        # 构建索引
        self.user_items = {}
        self.item_users = {}
        for u in range(self.n_users):
            self.user_items[u] = list(interactions[u].nonzero()[1])
        for i in range(self.n_items):
            self.item_users[i] = list(interactions[:, i].nonzero()[0])

        # 迭代
        for iteration in range(self.n_iterations):
            # 更新用户矩阵
            for u in range(self.n_users):
                items = self.user_items[u]
                if not items:
                    continue

                Q_u = self.Q[items]
                ratings = np.array([interactions[u, i] for i in items])
                residuals = ratings - self.mu - self.b_u[u] - self.b_i[items]

                A = Q_u.T @ Q_u + self.reg * np.eye(self.n_factors)
                b = Q_u.T @ residuals
                self.P[u] = np.linalg.solve(A, b)

                self.b_u[u] = np.mean(residuals - Q_u @ self.P[u])

            # 更新商品矩阵
            for i in range(self.n_items):
                users = self.item_users[i]
                if not users:
                    continue

                P_i = self.P[users]
                ratings = np.array([interactions[u, i] for u in users])
                residuals = ratings - self.mu - self.b_u[users] - self.b_i[i]

                A = P_i.T @ P_i + self.reg * np.eye(self.n_factors)
                b = P_i.T @ residuals
                self.Q[i] = np.linalg.solve(A, b)

                self.b_i[i] = np.mean(residuals - P_i @ self.Q[i])

            if (iteration + 1) % 3 == 0:
                print(f"   ALS Iteration {iteration+1}/{self.n_iterations}")

        return self


# ==================== 示例代码 ====================

def generate_sample_data(n_users=1000, n_items=500, density=0.01):
    """生成模拟数据"""
    np.random.seed(42)

    # 随机生成用户-商品交互
    data = []
    for u in range(n_users):
        for i in range(n_items):
            if np.random.random() < density:
                # 评分 1-5
                rating = np.random.randint(1, 6)
                data.append((u, i, rating))

    # 构建稀疏矩阵
    rows = [d[0] for d in data]
    cols = [d[1] for d in data]
    vals = [d[2] for d in data]

    interactions = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    return interactions, data


def main():
    """主函数"""
    print("=" * 60)
    print("Matrix Factorization 推荐系统测试")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    interactions, data = generate_sample_data(n_users=500, n_items=200, density=0.02)
    print(f"   用户数: {interactions.shape[0]}")
    print(f"   商品数: {interactions.shape[1]}")
    print(f"   交互数: {len(data)}")

    # 2. 训练模型
    print("\n[2] 训练矩阵分解模型...")
    model = MatrixFactorization(n_factors=30, n_epochs=20, lr=0.005, reg=0.02)
    model.fit(interactions)

    # 3. 推荐
    print("\n[3] 为用户生成推荐...")
    user_id = 0
    recommendations = model.recommend(user_id, n_items=10)
    print(f"   用户 {user_id} 的 Top 10 推荐:")
    for item_id, score in recommendations:
        print(f"     商品 {item_id}: {score:.2f}")

    # 4. 相似商品
    print("\n[4] 查找相似商品...")
    item_id = 100
    similar_items = model.get_similar_items(item_id, n=5)
    print(f"   商品 {item_id} 的相似商品:")
    for sim_item_id, similarity in similar_items:
        print(f"     商品 {sim_item_id}: {similarity:.4f}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return model


if __name__ == '__main__':
    model = main()
