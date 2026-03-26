"""
Matrix Factorization for Recommendation
用于母婴出海电商商品推荐
"""

import numpy as np
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class MatrixFactorization:
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def fit(self, interactions):
        self.n_users, self.n_items = interactions.shape
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.mu = interactions.sum() / max(1, (interactions > 0).sum())

        interactions = interactions.tocoo()
        for epoch in range(self.n_epochs):
            for u, i, r in zip(interactions.row, interactions.col, interactions.data):
                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                error = r - pred
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])
                P_u = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u - self.reg * self.Q[i])
        return self

    def recommend(self, user_id, n_items=10):
        scores = self.mu + self.b_u[user_id] + self.b_i + np.dot(self.P[user_id], self.Q.T)
        top_items = np.argsort(scores)[::-1][:n_items]
        return [(i, scores[i]) for i in top_items]

    def get_similar_items(self, item_id, n=5):
        item_vector = self.Q[item_id]
        similarities = np.dot(self.Q, item_vector) / (
            np.linalg.norm(self.Q, axis=1) * np.linalg.norm(item_vector) + 1e-8
        )
        similarities[item_id] = -1
        top_items = np.argsort(similarities)[::-1][:n]
        return [(i, similarities[i]) for i in top_items]


def generate_sample_data(n_users=500, n_items=200, density=0.02):
    np.random.seed(42)
    data = []
    for u in range(n_users):
        for i in range(n_items):
            if np.random.random() < density:
                rating = np.random.randint(1, 6)
                data.append((u, i, rating))
    rows = [d[0] for d in data]
    cols = [d[1] for d in data]
    vals = [d[2] for d in data]
    interactions = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return interactions, data


def main():
    print("=" * 60)
    print("Matrix Factorization 推荐系统测试")
    print("=" * 60)

    print("\n[1] 生成模拟数据...")
    interactions, data = generate_sample_data()
    print(f"   用户数: {interactions.shape[0]}, 商品数: {interactions.shape[1]}")

    print("\n[2] 训练模型...")
    model = MatrixFactorization(n_factors=30, n_epochs=10)
    model.fit(interactions)

    print("\n[3] 为用户推荐...")
    recommendations = model.recommend(0, n_items=5)
    for item_id, score in recommendations:
        print(f"   商品 {item_id}: {score:.2f}")

    print("\n" + "=" * 60)
    print("测试完成!")
    return model


if __name__ == '__main__':
    main()
