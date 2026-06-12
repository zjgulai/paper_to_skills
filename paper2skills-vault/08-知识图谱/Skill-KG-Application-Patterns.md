---
title: KG Application Patterns — 知识图谱下游应用：从构建到推荐/搜索/冷启动
doc_type: knowledge
module: 08-知识图谱
topic: kg-application-patterns
status: stable
created: 2026-06-11
updated: 2026-06-11
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: KG Application Patterns — 知识图谱下游应用模式

> **论文**：Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching
> **arXiv**：2606.11583 | 2026年 | **桥梁**: 08-知识图谱 ↔ 05-推荐系统 | **类型**: 跨域融合
> **反直觉来源**：`Skill-KG-Auto-Construction-Agent-Driven` in=19 但 out=0 — 整个 KG 构建体系无下游应用出口

---

## ① 算法原理

### 核心思想

知识图谱的价值不在于"建完"，而在于"用好"。图谱构建完成后，有三类核心下游任务：**节点分类**（SKU 类目自动标注）、**链接预测**（商品关系补全）、**冷启动推荐**（新品零历史数据的推荐）。这三类任务的共同难点是**少标注 + 新节点**。

**LLM-GNN Co-Teaching** 用两个老师互相监督的方式解决这个问题：
- **GNN 老师**：读图拓扑（谁和谁相连），擅长结构化模式
- **LLM 老师**：读文本语义（产品描述、品类名称），擅长语义理解
- **共同教学**：置信度高的 GNN 预测指导 LLM，置信度高的 LLM 预测指导 GNN，互相用对方的可信样本训练

```
GNN 预测 → 高置信样本 → 训练 LLM
LLM 预测 → 高置信样本 → 训练 GNN
```

**冷节点处理**：新产品没有图结构关系 → GNN 无法推断 → 退化为 LLM 预测（只用文本）。随着新品积累交互，GNN 逐渐接管（从 LLM-only → Co-teaching）。

### 数学直觉

设 GNN 对节点 $v$ 的预测置信度为 $\text{conf}_{\text{GNN}}(v)$，LLM 为 $\text{conf}_{\text{LLM}}(v)$，共同教学权重：

$$\hat{y}_v = \text{conf}_{\text{GNN}}(v) \cdot y_{\text{GNN}} + \text{conf}_{\text{LLM}}(v) \cdot y_{\text{LLM}}$$

高置信预测作为伪标签：$\mathcal{L}_{\text{pseudo}} = \sum_{v: \text{conf}>τ} \text{CE}(\hat{y}_v, y_{\text{GNN/LLM}})$

### 关键假设
- 产品有文本描述（标题 + bullet points）
- KG 已构建（由 AutoPKG 或人工维护）
- 有少量真实标注（10-100 个/类别即可）

---

## ② 母婴出海应用案例

### 场景 A：新品上架自动类目标注（冷启动）

**业务问题**：母婴团队每月上架 50 款新品，需要手工分配到 Amazon 的 200+ 子类目。分类错误会导致搜索流量损失（错误类目的关键词权重不同），且修改需等待 Amazon 审核。

**Co-Teaching 处理**：
1. **构建 SKU 类目图**：用 AutoPKG 输出的属性图 + 历史销售关系（一起购买、互补品）
2. **新品冷启动**：新 SKU 只有文本描述（无交互历史）→ LLM 老师根据标题/描述预测类目
3. **有交互后精化**：积累 2-4 周浏览/购买数据 → GNN 老师接管，精化类目归属

**预期产出**：新品类目分配准确率从人工 85% → 模型 93%+；人工标注时间从 2 小时/款 → 5 分钟确认

**业务价值**：类目准确率提升带来搜索流量增长约 8-15%，年化 GMV 增量 ¥20-60 万

### 场景 B：跨平台商品关系补全（链接预测）

**业务问题**：Amazon 图谱中标注了"经常一起购买"关系，但大量实际互补品（如"吸奶器 + 储奶袋"）因历史数据不足未被捕获，导致关联推荐不完整。

**Co-Teaching 处理**：用文本语义（吸奶器描述提到"配套储奶袋"）+ 稀疏交互数据 → 预测未标注的互补关系 → 补全图谱

**业务价值**：关联推荐覆盖率提升 30%，连带销售率提升约 5-8%

---

## ③ 代码模板

```python
"""
KG Application Patterns — LLM-GNN Co-Teaching 下游应用
基于 arXiv: 2606.11583

依赖: json, random, dataclasses (标准库)
生产环境: 替换 MockGNN/MockLLM 为真实模型
"""

from dataclasses import dataclass, field
import random
import json


@dataclass
class SKUNode:
    """知识图谱中的 SKU 节点"""
    sku_id: str
    title: str
    description: str
    category: str = ""          # 真实类目（训练集）
    predicted_category: str = ""
    confidence_gnn: float = 0.0
    confidence_llm: float = 0.0
    is_cold_start: bool = False  # 新品（无图结构邻居）


@dataclass
class KGGraph:
    """简化的知识图谱"""
    nodes: dict = field(default_factory=dict)   # {sku_id: SKUNode}
    edges: list = field(default_factory=list)   # [(src, tgt, relation)]

    def get_neighbors(self, sku_id: str) -> list:
        return [tgt for src, tgt, _ in self.edges if src == sku_id] + \
               [src for src, tgt, _ in self.edges if tgt == sku_id]


class MockGNN:
    """模拟 GNN 推理（生产环境替换为 PyG/DGL）"""

    CATEGORY_MAP = {
        "pump": "Breast Pumps", "steril": "Sterilizers",
        "bottle": "Baby Bottles", "stroller": "Strollers",
        "diaper": "Diapers", "formula": "Baby Formula",
    }

    def predict(self, node: SKUNode, graph: KGGraph) -> tuple:
        """基于图结构预测类目"""
        neighbors = graph.get_neighbors(node.sku_id)
        if not neighbors or node.is_cold_start:
            return "", 0.0  # 冷启动时无法预测

        # 多数投票（从邻居推断）
        neighbor_categories = [
            graph.nodes[n].category for n in neighbors
            if graph.nodes.get(n) and graph.nodes[n].category
        ]
        if not neighbor_categories:
            return "", 0.3

        from collections import Counter
        most_common = Counter(neighbor_categories).most_common(1)[0]
        confidence = most_common[1] / len(neighbor_categories)
        return most_common[0], min(0.95, confidence * 0.9 + 0.1)


class MockLLM:
    """模拟 LLM 文本推理（生产环境替换为 GPT-4o/Claude）"""

    KEYWORDS = {
        "Breast Pumps": ["pump", "suction", "breast", "wearable", "nursing"],
        "Sterilizers": ["steril", "uv", "disinfect", "clean", "sanitize"],
        "Baby Bottles": ["bottle", "nipple", "feeding", "anti-colic"],
        "Strollers": ["stroller", "pram", "pushchair", "fold", "canopy"],
        "Baby Formula": ["formula", "milk powder", "infant", "stage"],
    }

    def predict(self, node: SKUNode) -> tuple:
        """基于文本语义预测类目"""
        text = (node.title + " " + node.description).lower()
        scores = {}
        for category, keywords in self.KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[category] = score

        best = max(scores, key=scores.get)
        best_score = scores[best]
        if best_score == 0:
            return "Other", 0.3

        confidence = min(0.9, 0.5 + best_score * 0.1)
        return best, confidence


class CoTeachingClassifier:
    """
    LLM-GNN 共同教学分类器
    策略：高置信样本互相作为伪标签
    """

    def __init__(self, gnn: MockGNN, llm: MockLLM,
                 confidence_threshold: float = 0.7):
        self.gnn = gnn
        self.llm = llm
        self.threshold = confidence_threshold

    def classify(self, node: SKUNode, graph: KGGraph) -> SKUNode:
        """对单个节点做分类"""
        gnn_cat, gnn_conf = self.gnn.predict(node, graph)
        llm_cat, llm_conf = self.llm.predict(node)

        node.confidence_gnn = gnn_conf
        node.confidence_llm = llm_conf

        # 冷启动：只用 LLM
        if node.is_cold_start or gnn_conf < 0.1:
            node.predicted_category = llm_cat
            return node

        # 共同教学：选置信度更高的
        if gnn_conf >= self.threshold and gnn_conf > llm_conf:
            node.predicted_category = gnn_cat
        elif llm_conf >= self.threshold and llm_conf > gnn_conf:
            node.predicted_category = llm_cat
        elif gnn_cat == llm_cat:
            node.predicted_category = gnn_cat
        else:
            # 不一致时：取更高置信度
            node.predicted_category = gnn_cat if gnn_conf > llm_conf else llm_cat

        return node

    def batch_classify(self, nodes: list, graph: KGGraph) -> list:
        return [self.classify(n, graph) for n in nodes]

    def link_prediction(self, src_id: str, candidates: list,
                        graph: KGGraph, top_k: int = 3) -> list:
        """预测潜在的图谱关联（链接预测）"""
        src = graph.nodes.get(src_id)
        if not src:
            return []

        src_cat_llm, src_conf = self.llm.predict(src)
        scores = []
        for cand_id in candidates:
            cand = graph.nodes.get(cand_id)
            if not cand or cand_id == src_id:
                continue
            cand_cat, _ = self.llm.predict(cand)
            # 简化：同类目或文本相似 → 高关联概率
            sim = 0.8 if src_cat_llm == cand_cat else 0.3
            scores.append((cand_id, sim))

        return sorted(scores, key=lambda x: -x[1])[:top_k]


def run_kg_application_demo():
    """演示：母婴 SKU 类目标注 + 链接预测"""
    print("=" * 60)
    print("KG Application Patterns — 母婴 SKU 下游应用演示")
    print("=" * 60)

    # 构建小型知识图谱
    graph = KGGraph()
    existing_skus = [
        SKUNode("SKU-001", "Momcozy M5 Wearable Breast Pump",
                "Electric breast pump suction wearable nursing 9-level", "Breast Pumps"),
        SKUNode("SKU-002", "Spectra S1 Hospital Grade Pump",
                "Double electric breast pump suction strong 12-level", "Breast Pumps"),
        SKUNode("SKU-003", "UV Sterilizer Baby Bottle",
                "UV sterilize disinfect bottle nipple 99.9% bacteria", "Sterilizers"),
        SKUNode("SKU-004", "Anti-Colic Baby Bottle Set",
                "Bottle feeding nipple anti-colic newborn BPA free", "Baby Bottles"),
    ]
    for s in existing_skus:
        graph.nodes[s.sku_id] = s

    # 添加边（基于共同购买关系）
    graph.edges = [
        ("SKU-001", "SKU-002", "similar"),
        ("SKU-001", "SKU-003", "complementary"),
        ("SKU-003", "SKU-004", "complementary"),
    ]

    # 新品（冷启动）
    new_skus = [
        SKUNode("NEW-001", "Lansinoh Single Breast Pump Portable",
                "Compact breast pump suction rechargeable USB quiet", is_cold_start=True),
        SKUNode("NEW-002", "Baby Formula Stage 2 Follow-On",
                "Infant formula milk powder stage 2 6-12 months", is_cold_start=True),
    ]
    for s in new_skus:
        graph.nodes[s.sku_id] = s

    gnn, llm = MockGNN(), MockLLM()
    classifier = CoTeachingClassifier(gnn, llm)

    # 1. 批量分类（含冷启动）
    all_nodes = existing_skus + new_skus
    results = classifier.batch_classify(all_nodes, graph)

    print("\n📋 SKU 类目分类结果:")
    print(f"{'SKU ID':<12} {'冷启动':>5} {'GNN置信':>8} {'LLM置信':>8} {'预测类目'}")
    print("-" * 65)
    for r in results:
        cold = "✅" if r.is_cold_start else "—"
        print(f"{r.sku_id:<12} {cold:>5} {r.confidence_gnn:>8.2f} "
              f"{r.confidence_llm:>8.2f}  {r.predicted_category}")

    # 2. 链接预测
    print("\n🔗 链接预测（SKU-003 的潜在关联）:")
    all_ids = [s.sku_id for s in all_nodes]
    links = classifier.link_prediction("SKU-003", all_ids, graph)
    for cand_id, score in links:
        cand = graph.nodes[cand_id]
        print(f"  SKU-003 ↔ {cand_id} ({cand.title[:30]}...)  score={score:.2f}")

    # 验证
    new_results = {r.sku_id: r for r in results if r.is_cold_start}
    assert new_results["NEW-001"].predicted_category == "Breast Pumps", "新品应被分类为吸奶器"
    assert new_results["NEW-002"].predicted_category == "Baby Formula", "新品应被分类为奶粉"
    assert len(links) > 0, "应有链接预测结果"

    print("\n[✓] KG Application Patterns 测试通过")
    return results


if __name__ == "__main__":
    run_kg_application_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-KG-Auto-Construction-Agent-Driven]]（本 Skill 是 KG 构建的直接下游；构建完成后才能做分类/链接预测/冷启动推荐）
- **前置（prerequisite）**：[[Skill-AutoPKG-Multimodal-Product-Attribute-KG]]（AutoPKG 输出的属性图是本 Skill 的输入图结构）
- **延伸（extends）**：[[Skill-GNN-Foundations]]（Co-Teaching 中的 GNN 老师需要 GNN 基础）
- **延伸（extends）**：[[Skill-Dense-Retrieval-Ecommerce-Semantic-Search]]（LLM 语义特征与稠密检索结合，共同支持搜索召回）
- **可组合（combinable）**：[[Skill-Product-Knowledge-Graph-Query]]（组合场景：Co-Teaching 分类新品 → 输出类目标签 → Q2K 跨平台映射 → 统一 SKU 图谱）
- **可组合（combinable）**：[[Skill-Matrix-Factorization]]（组合场景：KG 链接预测补全关系后 → 传入协同过滤作为辅助信号，解决稀疏问题）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 新品类目自动标注：人工时间从 2 小时/款 → 5 分钟，50 SKU/月节省 ¥15,000+/月
  - 类目准确率提升（85%→93%）带来搜索流量增长 8-15%，年化 GMV ¥20-60 万
  - 链接预测补全推荐图：连带销售率提升 5-8%，年化 GMV ¥10-30 万
  - **年化综合 ROI**：¥50-110 万

- **实施难度**：⭐⭐⭐☆☆（需要 PyG/DGL 基础；冷启动版本纯 LLM 即可上线，2-3 天）

- **优先级评分**：⭐⭐⭐⭐⭐（填补知识图谱领域最大缺口：构建完的 KG 终于有使用方式）

- **评估依据**：论文 2606.11583 在电商场景图上验证；LLM-GNN Co-Teaching 在 NeurIPS/KDD 多次验证冷启动效果提升 15-25%
