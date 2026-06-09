"""
Semantic ID Retrieval — RPG-inspired implementation
Skill: Skill-Semantic-ID-Retrieval-RPG.md
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticIDEncoder:
    """语义ID编码器"""

    def __init__(self, max_tokens=8):
        self.max_tokens = max_tokens
        self.token_vocab = {}
        self.token_idx = 0

    def encode(self, item_attributes):
        tokens = []
        for key, value in item_attributes.items():
            token = f"{key}={value}"
            if token not in self.token_vocab:
                self.token_vocab[token] = self.token_idx
                self.token_idx += 1
            tokens.append(self.token_vocab[token])
        while len(tokens) < self.max_tokens:
            tokens.append(-1)
        return tokens[:self.max_tokens]


class SemanticRetrieval:
    """语义ID检索系统"""

    def __init__(self, embedding_dim=128):
        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.item_ids = []
        self.item_embeddings = None

    def index_items(self, items):
        self.item_ids = [item['id'] for item in items]
        texts = [' '.join([f"{k}_{v}" for k, v in item['attributes'].items()]) for item in items]
        self.item_embeddings = self.vectorizer.fit_transform(texts)

    def search(self, query_attributes, top_k=10):
        query_text = ' '.join([f"{k}_{v}" for k, v in query_attributes.items()])
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.item_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.item_ids[i], sims[i]) for i in top_idx]


if __name__ == '__main__':
    items = [
        {'id': 'sku_001', 'attributes': {'category': '奶粉', 'stage': '3段', 'brand': '爱他美'}},
        {'id': 'sku_002', 'attributes': {'category': '奶粉', 'stage': '2段', 'brand': '爱他美'}},
        {'id': 'sku_003', 'attributes': {'category': '纸尿裤', 'size': 'M', 'brand': '花王'}},
    ]
    retriever = SemanticRetrieval()
    retriever.index_items(items)
    print("Search '3段奶粉':", retriever.search({'category': '奶粉', 'stage': '3段'}, top_k=2))
