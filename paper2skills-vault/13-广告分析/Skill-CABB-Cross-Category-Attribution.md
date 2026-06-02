---
title: Click A Buy B 跨品类归因去偏 - 点击与购买商品不一致的归因修正
doc_type: knowledge
module: 13-广告分析
topic: click-a-buy-b-attribution
status: stable
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: arXiv:2507.15113 (2025)
---

# Skill: CABB Cross-Category Attribution — Click A Buy B 跨品类归因去偏

> 论文：**Click A, Buy B: Rethinking Conversion Attribution in E-Commerce Recommendations** · arXiv:2507.15113 (2025)
> 作者：Xiangyu Zeng, Amit Jaspal, Bin Liu, Goutham Panneeru, Kevin Huang, Nicolas Bievre, Mohit Jaggi, Prathap Maniraju, Ankur Jain
> 应用：电商广告中用户点击商品 A 但购买商品 B 导致的归因偏差修正

---

## ① 算法原理

### 核心思想

电商广告平台大量 Session 存在 **CABB（Click A, Buy B）** 现象——用户点击了商品 A 的广告，但最终购买的是商品 B。传统的 Last-Click 归因模型将这些"点击-购买不一致"的会话视为无意义噪声，强行将转化归零，导致系统性学习偏差：模型被迫"奖励那些与购买只是巧合相关、而非真正驱动转化的商品展示"。

论文的核心洞见是：CABB 并非全是噪声，其中相当一部分是 **替代品购买（Substitution）** 或 **互补品购买（Complementary）**——广告确实促成了购买，只是用户在浏览中发现了更合适的选择。因此不应丢弃这些信号，而应基于**品类相似度**给予差异化权重。

方案将转化预测重构为**多任务学习**：
- **CABA Head（Click A Buy A）**：点击与购买是同一商品，信号纯净，直接建模。
- **CABB Head（Click A Buy B）**：点击与购买不一致，通过 taxonomy-aware 协同过滤相似度加权，区分"有效 CABB"（品类相近，广告真实贡献）与"巧合 CABB"（跨类随机购买，降权）。

**离线评估**：归一化熵（Normalized Entropy）降低 **13.9%**（vs Last-Click 基线）。  
**线上 A/B 测试**：主业务指标（转化率 CVR）提升 **+0.25%**。

### 数学直觉

**Step 1 — Taxonomy 映射**：将每件商品 $p$ 映射到产品分类树中的叶节点：

$$\text{cat}(p) = \text{TaxonomyLeaf}(p)$$

**Step 2 — 品类级协同过滤相似度矩阵**：

从大规模用户共同参与日志（co-engagement logs）中学习品类间相似度矩阵 $S \in \mathbb{R}^{C \times C}$，其中 $C$ 为品类总数：

$$S_{c_i, c_j} = \text{CosineSim}\left(\text{Embed}(c_i),\ \text{Embed}(c_j)\right)$$

**Step 3 — CABB 样本加权**：

对于一条 CABB 会话（点击商品 A，购买商品 B），其归因权重为：

$$w_{\text{CABB}}(A, B) = S_{\text{cat}(A),\ \text{cat}(B)}$$

权重高 → 品类相近（替代/互补品），保留为有效正样本；  
权重低 → 品类差异大（巧合购买），降权甚至剔除。

**Step 4 — 多任务联合训练**：

$$\mathcal{L} = \mathcal{L}_{\text{CABA}} + \lambda \cdot \sum_{(A,B) \in \text{CABB}} w_{\text{CABB}}(A,B) \cdot \mathcal{L}_{\text{CABB}}(A, B)$$

两个 Head 共享底层表示，独立输出转化概率，最终线上使用 CABA + 加权 CABB 的合并信号进行排序。

### 关键假设

1. **Taxonomy 质量**：产品分类树必须足够细粒度，才能区分真正相近品类 vs 无关品类。
2. **共参与日志充足**：品类相似度矩阵需要大规模用户行为日志学习，冷门品类（长尾）覆盖不足时相似度可靠性下降。
3. **跨品类购买有规律**：替代品/互补品的跨类购买需要在数据中有统计显著的共现规律。
4. **CABB 比例非边缘情况**：如果平台 CABB 比例极低（< 1%），提升幅度有限；论文场景中该比例"significant"（显著）。

### 关键效果数字

| 指标 | 提升 | 说明 |
|------|------|------|
| Normalized Entropy | **-13.9%** | 离线评估，vs Last-Click 基线 |
| 主业务指标 CVR | **+0.25%** | 在线 A/B 测试，live traffic |

---

## ② 母婴出海应用案例

### 场景 1：吸奶器广告 → 奶瓶购买的归因恢复

**业务问题**：用户点击吸奶器广告，进入品牌店铺后转而购买了储奶袋和奶瓶。Last-Click 归因将吸奶器广告标记为"零转化"→ 该广告系列预算在下一轮自动竞价中被砍。但实际上，用户通过吸奶器广告建立了品牌信任，最终在母婴品类内完成了高价值购买。

这是典型的**品类内交叉购买 CABB**：吸奶器与奶瓶/储奶袋的品类相似度高（同属哺乳周边），应恢复该广告的归因权重。

**数据要求**：
- 用户 Session 数据：`session_id, clicked_product_id, purchased_product_id, timestamp`
- 商品 Taxonomy 树：产品类目层级（至少三级：大类 → 中类 → 小类），如"母婴 → 哺乳用品 → 吸奶器"
- 共参与日志：用户在同一 session/7 天窗口内共同点击/收藏/购买的商品对，用于训练品类相似度矩阵
- 广告展示日志：`ad_id, product_id, user_id, impression/click/conversion`

**预期产出**：
- 修正后的广告归因权重（区分 CABA/CABB，CABB 按品类相似度加权）
- 重新评分的广告 ROAS，恢复被低估渠道的预算贡献
- 品类间相似度热力图（运营可用于理解母婴用户交叉购买规律）

**业务价值**：恢复 10-30% 被错误归零的广告转化信号，避免高效广告被错误砍预算，预估可提升整体广告 ROAS 5-15%（中型母婴品牌年广告支出 100 万元，约合 5-15 万元增量收益/年）。


### 场景 2：搜索广告 → 关联母婴产品购买的跨类归因

**业务问题**：用户搜索"婴儿推车"并点击了推车广告，最终购买的是汽车安全座椅（同属出行安全品类）。Last-Click 模型认为推车广告未转化，但用户是通过推车广告进入品牌生态后发现更高客单价商品的典型 CABB 路径。

**数据要求**：搜索关键词 → 点击商品 → 购买商品的 Session 三元组；商品品类 Embedding（可从产品知识图谱生成）。

**预期产出**：出行品类内（推车/安全座椅/婴儿背带）的跨品归因权重矩阵；调整后的关键词竞价出价建议。

**业务价值**：降低跨品类高客单价商品的广告获客成本，优化搜索广告的关键词出价策略。

---

## ③ 代码模板

```python
"""
CABB (Click A, Buy B) 跨品类归因去偏框架
=========================================
基于 arXiv:2507.15113 论文实现：
- Session CABB 识别与分类
- Taxonomy-aware 品类协同过滤相似度学习
- CABA/CABB 双头多任务转化预测
- 归因权重修正

适用场景：母婴出海电商广告归因偏差修正
依赖：numpy, pandas, torch, sklearn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 数据结构与 Taxonomy 映射
# ============================================================

class ProductTaxonomy:
    """商品品类树管理器
    
    维护商品 → 品类映射，计算品类路径相似度
    """
    def __init__(self):
        # 母婴品类树示例（三级：大类/中类/小类）
        self.taxonomy = {
            # 哺乳周边品类
            "breast_pump": ["maternal_infant", "nursing", "breast_pump"],
            "milk_bag":    ["maternal_infant", "nursing", "milk_storage"],
            "bottle":      ["maternal_infant", "nursing", "feeding_bottle"],
            "nipple":      ["maternal_infant", "nursing", "nipple_shield"],
            # 出行品类
            "stroller":    ["maternal_infant", "travel", "stroller"],
            "car_seat":    ["maternal_infant", "travel", "car_seat"],
            "baby_carrier":["maternal_infant", "travel", "baby_carrier"],
            # 睡眠品类
            "crib":        ["maternal_infant", "sleep", "crib"],
            "sleep_sack":  ["maternal_infant", "sleep", "sleep_sack"],
            "night_light": ["maternal_infant", "sleep", "night_light"],
            # 喂养品类
            "baby_food":   ["maternal_infant", "feeding", "solid_food"],
            "formula":     ["maternal_infant", "feeding", "formula"],
        }
    
    def get_category(self, product_id: str, level: int = 2) -> str:
        """获取商品指定层级的品类标签"""
        path = self.taxonomy.get(product_id, ["unknown", "unknown", "unknown"])
        return "/".join(path[:level+1])
    
    def get_leaf_category(self, product_id: str) -> str:
        """获取叶节点品类"""
        path = self.taxonomy.get(product_id, ["unknown", "unknown", product_id])
        return path[-1]
    
    def path_similarity(self, prod_a: str, prod_b: str) -> float:
        """基于分类树路径计算商品品类相似度
        
        共享路径越深（越接近叶节点），相似度越高
        """
        path_a = self.taxonomy.get(prod_a, [])
        path_b = self.taxonomy.get(prod_b, [])
        
        if not path_a or not path_b:
            return 0.0
        
        # 计算最长公共前缀深度
        common_depth = 0
        for i, (a, b) in enumerate(zip(path_a, path_b)):
            if a == b:
                common_depth = i + 1
            else:
                break
        
        max_depth = max(len(path_a), len(path_b))
        return common_depth / max_depth


# ============================================================
# 2. 品类协同过滤相似度矩阵
# ============================================================

class CategoryCoEmbedding:
    """品类协同过滤 Embedding 学习器
    
    从用户共参与日志（co-engagement logs）学习品类间相似度矩阵
    用于区分"有效 CABB"（品类相近）vs "巧合 CABB"（品类无关）
    """
    def __init__(self, embedding_dim: int = 32, taxonomy: Optional[ProductTaxonomy] = None):
        self.embedding_dim = embedding_dim
        self.taxonomy = taxonomy or ProductTaxonomy()
        self.category_encoder = LabelEncoder()
        self.embeddings: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.categories: List[str] = []
    
    def build_cooccurrence_matrix(
        self, 
        sessions: pd.DataFrame,
        window_days: int = 7
    ) -> np.ndarray:
        """从 Session 数据构建品类共现矩阵
        
        Args:
            sessions: 列包含 [session_id, product_id, action_type, timestamp]
            window_days: 共参与时间窗口（天）
        Returns:
            co_matrix: 品类×品类共现频次矩阵
        """
        # 提取每个 session 的品类集合
        sessions["category"] = sessions["product_id"].apply(
            lambda p: self.taxonomy.get_category(p, level=1)
        )
        
        # 统计品类对共现次数
        all_categories = sessions["category"].unique().tolist()
        self.categories = sorted(all_categories)
        n_cat = len(self.categories)
        cat_idx = {c: i for i, c in enumerate(self.categories)}
        
        co_matrix = np.zeros((n_cat, n_cat), dtype=float)
        
        for sid, grp in sessions.groupby("session_id"):
            cats = grp["category"].unique()
            for i, c1 in enumerate(cats):
                for c2 in cats[i+1:]:
                    idx1, idx2 = cat_idx.get(c1, -1), cat_idx.get(c2, -1)
                    if idx1 >= 0 and idx2 >= 0:
                        co_matrix[idx1, idx2] += 1
                        co_matrix[idx2, idx1] += 1
        
        # 归一化（PPMI，正点互信息）
        row_sum = co_matrix.sum(axis=1, keepdims=True)
        col_sum = co_matrix.sum(axis=0, keepdims=True)
        total = co_matrix.sum()
        
        with np.errstate(divide="ignore", invalid="ignore"):
            ppmi = np.log((co_matrix * total) / (row_sum @ col_sum + 1e-9) + 1e-9)
            ppmi = np.maximum(ppmi, 0)
        
        return ppmi
    
    def fit(self, sessions: pd.DataFrame) -> "CategoryCoEmbedding":
        """从 Session 数据学习品类 Embedding 与相似度矩阵"""
        co_matrix = self.build_cooccurrence_matrix(sessions)
        
        # SVD 降维得到品类 Embedding
        U, s, Vt = np.linalg.svd(co_matrix, full_matrices=False)
        k = min(self.embedding_dim, len(s))
        self.embeddings = U[:, :k] * np.sqrt(s[:k])
        
        # 计算品类余弦相似度矩阵
        self.similarity_matrix = cosine_similarity(self.embeddings)
        np.fill_diagonal(self.similarity_matrix, 1.0)
        return self
    
    def get_similarity(self, cat_a: str, cat_b: str) -> float:
        """获取两个品类的相似度分数"""
        if self.similarity_matrix is None:
            raise RuntimeError("请先调用 fit() 训练品类 Embedding")
        
        cat_idx = {c: i for i, c in enumerate(self.categories)}
        idx_a = cat_idx.get(cat_a, -1)
        idx_b = cat_idx.get(cat_b, -1)
        
        if idx_a < 0 or idx_b < 0:
            # Fallback：使用 Taxonomy 路径相似度
            return self.taxonomy.path_similarity(cat_a.split("/")[-1], cat_b.split("/")[-1])
        
        return float(self.similarity_matrix[idx_a, idx_b])


# ============================================================
# 3. CABB Session 分类与加权
# ============================================================

class CABBSessionProcessor:
    """CABB Session 识别与归因权重分配器"""
    
    def __init__(self, taxonomy: ProductTaxonomy, cat_embedding: CategoryCoEmbedding):
        self.taxonomy = taxonomy
        self.cat_embedding = cat_embedding
    
    def classify_sessions(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """将 Session 分类为 CABA / CABB-informative / CABB-spurious
        
        Args:
            sessions: 列包含 [session_id, clicked_product, purchased_product]
        Returns:
            sessions: 添加 [session_type, attribution_weight] 列
        """
        results = []
        for _, row in sessions.iterrows():
            clicked = row["clicked_product"]
            purchased = row["purchased_product"]
            
            if pd.isna(purchased) or purchased == "":
                session_type = "no_conversion"
                weight = 0.0
            elif clicked == purchased:
                session_type = "CABA"
                weight = 1.0
            else:
                # CABB：计算品类相似度决定权重
                cat_clicked = self.taxonomy.get_category(clicked, level=1)
                cat_purchased = self.taxonomy.get_category(purchased, level=1)
                
                similarity = self.cat_embedding.get_similarity(cat_clicked, cat_purchased)
                
                if similarity >= 0.5:
                    session_type = "CABB_informative"  # 有效 CABB（替代/互补品）
                    weight = similarity
                elif similarity >= 0.2:
                    session_type = "CABB_weak"          # 弱相关 CABB
                    weight = similarity * 0.5
                else:
                    session_type = "CABB_spurious"      # 巧合 CABB（降权剔除）
                    weight = 0.0
            
            results.append({**row.to_dict(), "session_type": session_type, "attribution_weight": weight})
        
        return pd.DataFrame(results)
    
    def compute_attribution_correction(
        self, 
        attributed_sessions: pd.DataFrame
    ) -> pd.DataFrame:
        """计算归因修正前后的广告 ROAS 对比
        
        Args:
            attributed_sessions: 含 [ad_id, revenue, session_type, attribution_weight] 的 DF
        Returns:
            summary: 广告级别的归因修正对比表
        """
        # Last-Click 归因（仅计 CABA，CABB 视为零转化）
        last_click = attributed_sessions[
            attributed_sessions["session_type"] == "CABA"
        ].groupby("ad_id").agg(
            last_click_revenue=("revenue", "sum"),
            last_click_conversions=("session_id", "count")
        ).reset_index()
        
        # CABB 修正归因（CABA + 加权 CABB）
        attributed_sessions["weighted_revenue"] = (
            attributed_sessions["revenue"] * attributed_sessions["attribution_weight"]
        )
        cabb_corrected = attributed_sessions[
            attributed_sessions["session_type"].isin(["CABA", "CABB_informative", "CABB_weak"])
        ].groupby("ad_id").agg(
            cabb_corrected_revenue=("weighted_revenue", "sum"),
            cabb_total_sessions=("session_id", "count")
        ).reset_index()
        
        summary = last_click.merge(cabb_corrected, on="ad_id", how="outer").fillna(0)
        summary["revenue_recovery_rate"] = (
            (summary["cabb_corrected_revenue"] - summary["last_click_revenue"]) 
            / (summary["last_click_revenue"] + 1e-9)
        )
        return summary


# ============================================================
# 4. CABA/CABB 双头多任务转化预测模型
# ============================================================

class CABBMultiTaskDataset(Dataset):
    """CABB 多任务训练数据集"""
    
    def __init__(self, sessions: pd.DataFrame, feature_dim: int = 16):
        self.sessions = sessions.reset_index(drop=True)
        # 简化：用商品 ID 哈希作为特征
        self._build_features()
        # feature_dim 以实际商品数量为准（确保与模型输入维度一致）
        self.feature_dim = len(self.prod_enc.classes_)
    
    def _build_features(self):
        all_products = pd.concat([
            self.sessions["clicked_product"],
            self.sessions["purchased_product"].fillna("unknown")
        ]).unique()
        self.prod_enc = LabelEncoder()
        self.prod_enc.fit(all_products)
    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        row = self.sessions.iloc[idx]
        
        clicked_id = self.prod_enc.transform([row["clicked_product"]])[0]
        purchased_id = self.prod_enc.transform(
            [row["purchased_product"] if pd.notna(row["purchased_product"]) else "unknown"]
        )[0]
        
        # One-Hot 特征（维度 = 商品数量）
        n_products = len(self.prod_enc.classes_)
        clicked_feat = np.zeros(n_products, dtype=np.float32)
        clicked_feat[clicked_id] = 1.0
        
        # 标签
        is_converted = float(row["session_type"] in ["CABA", "CABB_informative", "CABB_weak"])
        is_caba = float(row["session_type"] == "CABA")
        weight = float(row["attribution_weight"])
        
        return {
            "features": torch.tensor(clicked_feat),
            "is_converted": torch.tensor(is_converted),
            "is_caba": torch.tensor(is_caba),
            "attribution_weight": torch.tensor(weight)
        }


class CABBMultiTaskModel(nn.Module):
    """CABA/CABB 双头多任务转化预测模型
    
    共享底层表示，独立 CABA/CABB 预测头
    """
    
    def __init__(self, feature_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        
        # 共享 Backbone
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # CABA Head：预测点击与购买一致的转化
        self.caba_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # CABB Head：预测跨品类转化（加权）
        self.cabb_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor):
        shared = self.backbone(features)
        caba_prob = self.caba_head(shared).squeeze(-1)
        cabb_prob = self.cabb_head(shared).squeeze(-1)
        return caba_prob, cabb_prob
    
    def compute_loss(
        self,
        features: torch.Tensor,
        is_caba: torch.Tensor,
        is_converted: torch.Tensor,
        attribution_weight: torch.Tensor,
        lambda_cabb: float = 0.5
    ) -> torch.Tensor:
        """多任务损失：L_CABA + λ * weighted_L_CABB"""
        caba_prob, cabb_prob = self.forward(features)
        
        # CABA 损失：标准 BCE
        loss_caba = nn.functional.binary_cross_entropy(caba_prob, is_caba)
        
        # CABB 损失：用品类相似度加权的 BCE
        cabb_label = is_converted * (1 - is_caba)  # CABB 正样本
        loss_cabb_raw = nn.functional.binary_cross_entropy(
            cabb_prob, cabb_label, reduction="none"
        )
        loss_cabb = (attribution_weight * loss_cabb_raw).mean()
        
        return loss_caba + lambda_cabb * loss_cabb
    
    def predict_attribution_score(self, features: torch.Tensor) -> torch.Tensor:
        """线上预测：CABA 概率 + CABB 概率（后者已通过训练时加权去偏）"""
        with torch.no_grad():
            caba_prob, cabb_prob = self.forward(features)
        return caba_prob + cabb_prob  # 合并归因分数


def train_cabb_model(
    sessions: pd.DataFrame,
    feature_dim: int = 16,
    hidden_dim: int = 64,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_cabb: float = 0.5
) -> CABBMultiTaskModel:
    """训练 CABB 多任务模型"""
    dataset = CABBMultiTaskDataset(sessions, feature_dim=feature_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 使用数据集的实际特征维度（商品数量），确保与模型输入一致
    actual_feature_dim = dataset.feature_dim
    model = CABBMultiTaskModel(feature_dim=actual_feature_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model.compute_loss(
                batch["features"],
                batch["is_caba"],
                batch["is_converted"],
                batch["attribution_weight"],
                lambda_cabb=lambda_cabb
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")
    
    return model


# ============================================================
# 5. 端到端 Pipeline 与测试用例
# ============================================================

def generate_mock_data(n_sessions: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成模拟母婴电商 Session 数据"""
    np.random.seed(seed)
    
    products = [
        "breast_pump", "milk_bag", "bottle", "nipple",
        "stroller", "car_seat", "baby_carrier",
        "crib", "sleep_sack", "night_light",
        "baby_food", "formula"
    ]
    
    sessions = []
    for i in range(n_sessions):
        session_id = f"sess_{i:05d}"
        clicked = np.random.choice(products)
        
        # 模拟转化规律：
        # 40% 不转化，30% CABA，20% 品类内 CABB，10% 跨类 CABB
        rand = np.random.random()
        if rand < 0.40:
            purchased = None
        elif rand < 0.70:
            purchased = clicked  # CABA
        elif rand < 0.90:
            # 品类内 CABB（哺乳/出行/睡眠内部交叉）
            taxonomy = ProductTaxonomy()
            cat = taxonomy.get_category(clicked, level=1)
            same_cat = [p for p in products if taxonomy.get_category(p, level=1) == cat and p != clicked]
            purchased = np.random.choice(same_cat) if same_cat else clicked
        else:
            # 跨品类 CABB（随机选不同品类商品）
            purchased = np.random.choice([p for p in products if p != clicked])
        
        revenue = np.random.uniform(20, 200) if purchased else 0.0
        ad_id = f"ad_{np.random.randint(1, 11):02d}"
        
        sessions.append({
            "session_id": session_id,
            "clicked_product": clicked,
            "purchased_product": purchased,
            "revenue": revenue,
            "ad_id": ad_id
        })
    
    return pd.DataFrame(sessions)


def run_cabb_pipeline(sessions_df: pd.DataFrame) -> Dict:
    """CABB 完整归因修正 Pipeline
    
    Returns:
        dict: {
            "attribution_summary": 广告级归因修正对比,
            "session_distribution": CABA/CABB 类型分布,
            "model": 训练好的 CABB 多任务模型
        }
    """
    print("=" * 60)
    print("CABB 跨品类归因修正 Pipeline 启动")
    print("=" * 60)
    
    # Step 1: 初始化 Taxonomy
    print("\n[Step 1] 初始化商品 Taxonomy...")
    taxonomy = ProductTaxonomy()
    
    # Step 2: 训练品类协同过滤 Embedding
    print("[Step 2] 学习品类协同过滤相似度矩阵...")
    cat_embedding = CategoryCoEmbedding(embedding_dim=16, taxonomy=taxonomy)
    # 构造 session 长格式（每个 session 包含点击和购买商品）
    long_format = []
    for _, row in sessions_df.iterrows():
        long_format.append({"session_id": row["session_id"], "product_id": row["clicked_product"]})
        if pd.notna(row["purchased_product"]):
            long_format.append({"session_id": row["session_id"], "product_id": row["purchased_product"]})
    cat_embedding.fit(pd.DataFrame(long_format))
    print(f"  ✓ 学习到 {len(cat_embedding.categories)} 个品类的相似度矩阵")
    
    # Step 3: Session 分类与加权
    print("[Step 3] CABB Session 分类与归因权重分配...")
    processor = CABBSessionProcessor(taxonomy, cat_embedding)
    attributed = processor.classify_sessions(sessions_df)
    
    type_dist = attributed["session_type"].value_counts()
    print(f"  Session 类型分布:\n{type_dist.to_string()}")
    
    # Step 4: 训练 CABB 多任务模型
    print("\n[Step 4] 训练 CABA/CABB 双头多任务模型...")
    conv_sessions = attributed[attributed["session_type"] != "no_conversion"].copy()
    if len(conv_sessions) > 0:
        model = train_cabb_model(conv_sessions, n_epochs=10)
    else:
        model = CABBMultiTaskModel()
        print("  ⚠ 无转化 Session，跳过模型训练")
    
    # Step 5: 归因修正对比
    print("\n[Step 5] 计算归因修正前后 ROAS 对比...")
    summary = processor.compute_attribution_correction(attributed)
    top_recovery = summary.nlargest(5, "revenue_recovery_rate")[
        ["ad_id", "last_click_revenue", "cabb_corrected_revenue", "revenue_recovery_rate"]
    ]
    print("\n归因修正 Top-5 广告（按收益恢复率排序）:")
    print(top_recovery.to_string(index=False))
    
    # 汇总统计
    total_last_click = summary["last_click_revenue"].sum()
    total_corrected = summary["cabb_corrected_revenue"].sum()
    overall_recovery = (total_corrected - total_last_click) / (total_last_click + 1e-9) * 100
    print(f"\n📊 整体归因修正:")
    print(f"  Last-Click 归因总收益: ¥{total_last_click:.2f}")
    print(f"  CABB 修正归因总收益:   ¥{total_corrected:.2f}")
    print(f"  归因恢复率:            +{overall_recovery:.1f}%")
    
    return {
        "attribution_summary": summary,
        "session_distribution": type_dist,
        "model": model
    }


# ============================================================
# 测试用例
# ============================================================

def test_taxonomy():
    """测试品类树路径相似度计算"""
    print("\n🧪 Test 1: Taxonomy 路径相似度")
    tax = ProductTaxonomy()
    
    # 同品类（哺乳周边内部）：吸奶器 vs 奶瓶
    sim1 = tax.path_similarity("breast_pump", "bottle")
    assert sim1 > 0.5, f"同品类相似度应 > 0.5，实际 {sim1:.2f}"
    
    # 跨大类（哺乳 vs 出行）：吸奶器 vs 推车
    sim2 = tax.path_similarity("breast_pump", "stroller")
    assert sim2 < sim1, f"跨品类相似度应 < 同品类，实际 {sim2:.2f} vs {sim1:.2f}"
    
    # 完全相同
    sim3 = tax.path_similarity("breast_pump", "breast_pump")
    assert sim3 == 1.0, f"相同商品相似度应为 1.0，实际 {sim3:.2f}"
    
    print(f"  ✓ breast_pump vs bottle (同类):  {sim1:.3f}")
    print(f"  ✓ breast_pump vs stroller (跨类): {sim2:.3f}")
    print(f"  ✓ breast_pump vs breast_pump (同): {sim3:.3f}")
    print("  测试通过 ✓")


def test_session_classification():
    """测试 Session CABB 分类逻辑"""
    print("\n🧪 Test 2: Session CABB 分类")
    
    tax = ProductTaxonomy()
    # 用 Taxonomy 相似度作为 fallback（跳过 Embedding 训练）
    cat_embedding = CategoryCoEmbedding(taxonomy=tax)
    cat_embedding.categories = ["maternal_infant/nursing", "maternal_infant/travel", "maternal_infant/sleep"]
    cat_embedding.embeddings = np.eye(3, dtype=float)
    cat_embedding.similarity_matrix = np.array([
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.1],
        [0.3, 0.1, 1.0]
    ])
    
    processor = CABBSessionProcessor(tax, cat_embedding)
    
    test_sessions = pd.DataFrame([
        {"session_id": "s1", "clicked_product": "breast_pump", "purchased_product": "breast_pump"},   # CABA
        {"session_id": "s2", "clicked_product": "breast_pump", "purchased_product": "bottle"},         # CABB（同类哺乳）
        {"session_id": "s3", "clicked_product": "breast_pump", "purchased_product": "stroller"},       # CABB（跨类）
        {"session_id": "s4", "clicked_product": "breast_pump", "purchased_product": None},             # 无转化
    ])
    
    result = processor.classify_sessions(test_sessions)
    
    assert result.iloc[0]["session_type"] == "CABA", "CABA 分类错误"
    assert result.iloc[0]["attribution_weight"] == 1.0, "CABA 权重应为 1.0"
    assert result.iloc[3]["session_type"] == "no_conversion", "无转化分类错误"
    assert result.iloc[3]["attribution_weight"] == 0.0, "无转化权重应为 0.0"
    
    print(f"  Session 类型: {result[['session_id', 'session_type', 'attribution_weight']].to_string(index=False)}")
    print("  测试通过 ✓")


def test_full_pipeline():
    """端到端 Pipeline 集成测试"""
    print("\n🧪 Test 3: 端到端 Pipeline")
    sessions = generate_mock_data(n_sessions=200, seed=42)
    print(f"  生成 {len(sessions)} 条 Session 数据")
    print(f"  转化率: {(sessions['purchased_product'].notna()).mean():.1%}")
    
    result = run_cabb_pipeline(sessions)
    
    assert "attribution_summary" in result
    assert "session_distribution" in result
    assert len(result["attribution_summary"]) > 0
    
    print("\n  测试通过 ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("CABB 跨品类归因去偏 — 测试套件")
    print("=" * 60)
    
    test_taxonomy()
    test_session_classification()
    test_full_pipeline()
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Skill-Ad-Attribution-Modeling (Shapley)]([[Skill-Ad-Attribution-Modeling]].md) | Shapley 归因是基础，CABB 补充 Shapley 无法处理的跨品类 Session 场景 |
| 组合 | [Skill-FrontDoor-Causal-MTA]([[Skill-FrontDoor-Causal-MTA]].md) | CABB 修正 Session 偏差后，FrontDoor 进一步处理渠道选择性偏差，两者串联消偏更彻底 |
| 组合 | [Skill-PIE-Experimental-MTA]([[Skill-PIE-Experimental-MTA]].md) | CABB 补充 PIE 框架中未覆盖的跨品类购买路径，PIE RCT 校准后可叠加 CABB 权重修正 |
| 延伸 | [Skill-ROAS-Budget-Optimization]([[Skill-ROAS-Budget-Optimization]].md) | 归因修正后输入更准确的 ROAS 信号，提升预算优化质量 |
| 延伸 | [Skill-PVM-Attribution-Window-Harmonization]([[Skill-PVM-Attribution-Window-Harmonization]].md) | CABB 解决"购买商品不匹配"偏差，PVM 解决"归因时间窗口不一致"偏差，两类偏差正交互补 |

---

- **前置技能**：[[Skill-Ad-Attribution-Modeling]] | [[Skill-ROAS-Budget-Optimization]]
- **延伸技能**：[[Skill-TESLA-NetCVR-Cascade]]
- **可组合技能**：[[Skill-Hierarchical-Search-Intent-Classification]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| ROI 预估 | ⭐⭐⭐⭐☆ | 恢复 10-30% 被错误归零的广告转化；线上验证 CVR +0.25%；中型品牌年广告 100 万元，预估增量贡献 5-15 万元/年 |
| 实施难度 | ⭐⭐⭐☆☆ | 需要商品 Taxonomy 树（通常已有）+ 用户 Session 日志（标准数据）+ 多任务模型改造（工程量中等） |
| 优先级 | ⭐⭐⭐⭐☆ | CABB 现象在母婴品类内天然高发（品类内互补品购买频繁）；Last-Click 归因偏差已造成可量化损失；P1 级建议落地 |

### 适用场景判断

| 场景 | 建议 |
|------|------|
| 母婴品类内（吸奶器/奶瓶/储奶袋）跨品转化高 | ✅ 强烈推荐，品类相似度天然高 |
| 多品类跨境店铺（母婴 + 家居 + 宠物） | ⚠️ 需谨慎，跨大类 CABB 信号噪声多 |
| 单品类 / SKU 少的精品店 | ❌ 不适用，缺乏 CABB 多样性 |
| 独立站（点击与购买归因窗口一致） | ✅ 适用，且工程复杂度更低（无跨平台 ID 问题） |
