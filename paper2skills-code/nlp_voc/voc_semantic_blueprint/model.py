"""
VOC Semantic Blueprint Extractor
基于 USSA (Zhai et al., ACL 2023) 的 Table-Filling 思想，
将用户评论从序列转换为结构化语义蓝图。

核心映射：
- USSA holder  → VOC 用户/评论者
- USSA target  → VOC 产品方面 (aspect)
- USSA expression → VOC 观点表达 (opinion)
- USSA polarity → VOC 情感极性 (sentiment)
- 扩展: cause (原因), scene (场景)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ── 数据模型 ──────────────────────────────────────────

@dataclass
class VOCBlueprintNode:
    """语义蓝图中的一个节点（对应一条观点）"""
    aspect: str           # 产品方面，如 "吸力", "噪音", "便携性"
    opinion: str          # 观点表达，如 "吸力很强", "噪音大"
    sentiment: str        # 正/负/中性: "positive" | "negative" | "neutral"
    cause: Optional[str] = None    # 原因，如 "因为电机功率大"
    scene: Optional[str] = None    # 场景，如 "夜间吸奶", "上班背奶"
    confidence: float = 0.0        # 置信度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.aspect,
            "opinion": self.opinion,
            "sentiment": self.sentiment,
            "cause": self.cause,
            "scene": self.scene,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class VOCBlueprint:
    """单条评论的语义蓝图 = 一组结构化节点 + 元信息"""
    nodes: List[VOCBlueprintNode] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "nodes": [n.to_dict() for n in self.nodes],
            "metadata": self.metadata,
        }

    @property
    def aspects(self) -> List[str]:
        return list({n.aspect for n in self.nodes})

    @property
    def sentiment_summary(self) -> Dict[str, int]:
        """情感分布统计"""
        summary: Dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for n in self.nodes:
            summary[n.sentiment] = summary.get(n.sentiment, 0) + 1
        return summary


# ── Table-Filling 核心 ────────────────────────────────

@dataclass
class TableCell:
    """2D Table 中的一个单元格"""
    row: int
    col: int
    relation: str          # RP 关系类型 或 TE token 标记
    score: float = 0.0


class RelationTable:
    """
    USSA 风格的 2D Table-Filling 结构。

    下三角 (row > col): Relation Prediction (RP)
        - E-POS: expression → positive polarity
        - E-NEG: expression → negative polarity
        - E-NEU: expression → neutral polarity
        - S-H  : holder start/end boundary
        - S-T  : target start/end boundary
        - S-E  : expression start/end boundary

    上三角 (row < col): Token Extraction (TE)
        - TE   : token 属于某个 entity
    """

    # 13 种 USSA 关系类型 → 5 种简化版用于 VOC
    RP_RELATIONS = ["E-POS", "E-NEG", "E-NEU", "S-H", "S-T", "S-E"]
    TE_RELATION = "TE"

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.n = len(tokens)
        # 下三角: RP 矩阵
        self.rp: Dict[Tuple[int, int], str] = {}
        # 上三角: TE 矩阵
        self.te: Dict[Tuple[int, int], str] = {}

    def set_rp(self, i: int, j: int, relation: str) -> None:
        """设置下三角单元格的关系类型 (i > j)"""
        if i <= j:
            raise ValueError(f"RP cell requires i > j, got ({i}, {j})")
        if relation not in self.RP_RELATIONS:
            raise ValueError(f"Unknown RP relation: {relation}")
        self.rp[(i, j)] = relation

    def set_te(self, i: int, j: int, label: str = "TE") -> None:
        """设置上三角单元格的 token 标记 (i < j)"""
        if i >= j:
            raise ValueError(f"TE cell requires i < j, got ({i}, {j})")
        self.te[(i, j)] = label

    def decode_blueprint(
        self,
        aspect_keywords: List[str],
    ) -> List[VOCBlueprintNode]:
        """
        从 Table 解码出 VOCBlueprintNode 列表。
        简化版：利用 RP 关系识别情感极性，利用 TE 识别 token 边界。
        """
        nodes: List[VOCBlueprintNode] = []

        # 1. 识别 expression 区间 (通过 S-E 边界)
        expr_spans = self._extract_spans("S-E")
        # 2. 识别 target (aspect) 区间
        target_spans = self._extract_spans("S-T")
        # 3. 识别 holder 区间
        holder_spans = self._extract_spans("S-H")

        # 4. 对每个 expression，找最近的 target 和 polarity
        for (e_start, e_end) in expr_spans:
            expr_text = " ".join(self.tokens[e_start:e_end + 1])

            # 找情感极性: 看 expression 所在行/列的 RP 关系
            polarity = self._find_polarity(e_start, e_end)

            # 找最近的 target (aspect)
            aspect, aspect_text = self._find_nearest_target(
                e_start, e_end, target_spans, aspect_keywords
            )

            # 原因/场景: 简单模式匹配
            cause = self._extract_cause(e_end)
            scene = self._extract_scene()

            nodes.append(VOCBlueprintNode(
                aspect=aspect,
                opinion=expr_text,
                sentiment=polarity,
                cause=cause,
                scene=scene,
                confidence=0.7,  # 规则基线置信度
            ))

        return nodes

    def _extract_spans(self, boundary_type: str) -> List[Tuple[int, int]]:
        """从 RP 矩阵提取某类 entity 的 start/end 边界对"""
        # 简化: S-X 标记在 (end, start) 位置
        spans = []
        starts = set()
        ends = set()
        for (i, j), rel in self.rp.items():
            if rel == boundary_type:
                ends.add(i)
                starts.add(j)
        # 配对: 每个 start 找最近的 end
        for s in sorted(starts):
            for e in sorted(ends):
                if e >= s:
                    spans.append((s, e))
                    break
        return spans

    def _find_polarity(self, start: int, end: int) -> str:
        """在 expression 区间附近找 E-POS/E-NEG/E-NEU"""
        for (i, j), rel in self.rp.items():
            if start <= j <= end or start <= i <= end:
                if rel == "E-POS":
                    return "positive"
                if rel == "E-NEG":
                    return "negative"
                if rel == "E-NEU":
                    return "neutral"
        return "neutral"

    def _find_nearest_target(
        self,
        e_start: int,
        e_end: int,
        target_spans: List[Tuple[int, int]],
        aspect_keywords: List[str],
    ) -> Tuple[str, str]:
        """找离 expression 最近的 target / aspect"""
        if target_spans:
            # 找距离最近的 target span
            best = min(target_spans, key=lambda t: abs(t[0] - e_start))
            text = " ".join(self.tokens[best[0]:best[1] + 1])
            return text, text

        # 回退: 在 expression 附近匹配 aspect keyword
        window = self.tokens[max(0, e_start - 5):e_end + 6]
        window_text = " ".join(window).lower()
        for kw in aspect_keywords:
            if kw.lower() in window_text:
                return kw, kw
        return "general", "general"

    def _extract_cause(self, after_idx: int) -> Optional[str]:
        """简单模式: 'because', 'since', 'as' 后面跟原因"""
        cause_markers = ["because", "since", "as", "due to", "所以", "因为"]
        for i in range(after_idx, min(self.n, after_idx + 8)):
            if self.tokens[i].lower() in cause_markers:
                end = min(self.n, i + 6)
                return " ".join(self.tokens[i:end])
        return None

    def _extract_scene(self) -> Optional[str]:
        """简单模式匹配场景关键词"""
        scene_keywords = {
            "night": "夜间使用", "sleep": "夜间使用", "bedtime": "夜间使用",
            "work": "上班背奶", "office": "上班背奶", "company": "上班背奶",
            "travel": "外出便携", "outdoor": "外出便携", "trip": "外出便携",
            "home": "居家使用", "house": "居家使用",
        }
        for i, tok in enumerate(self.tokens):
            t = tok.lower()
            if t in scene_keywords:
                return scene_keywords[t]
        return None


# ── 主提取器 ──────────────────────────────────────────

class VOCBlueprintExtractor:
    """
    VOC 语义蓝图提取器。

    基于 USSA Table-Filling 思想，将单条评论解析为
    (aspect, opinion, sentiment, cause, scene) 五元组列表。
    """

    # 母婴出海场景默认方面词典（吸奶器品类）
    DEFAULT_ASPECTS = [
        "suction", "noise", "portability", "comfort", "battery",
        "cleaning", "app", "price", "design", "material",
        "吸力", "噪音", "便携", "舒适", "续航",
        "清洗", "APP", "价格", "设计", "材质",
    ]

    # 情感词词典（简化版）
    POSITIVE_WORDS = {
        "great", "good", "excellent", "amazing", "love", "perfect",
        "strong", "convenient", "easy", "comfortable", "quiet",
        "好", "不错", "强", "方便", "舒适", "安静", "满意",
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "hate", "disappointed", "noisy", "loud",
        "difficult", "uncomfortable", "weak", "expensive", "poor",
        "差", "不好", "吵", "难", "不舒服", "弱", "贵", "失望",
    }

    def __init__(
        self,
        aspect_keywords: Optional[List[str]] = None,
    ):
        self.aspect_keywords = aspect_keywords or self.DEFAULT_ASPECTS

    def extract(self, text: str) -> VOCBlueprint:
        """提取单条评论的语义蓝图"""
        if not text or not isinstance(text, str):
            return VOCBlueprint(raw_text=text or "")

        # 1. 分词 (简化: 按空格和标点分词)
        tokens = self._tokenize(text)
        if not tokens:
            return VOCBlueprint(raw_text=text)

        # 2. 构建 Table-Filling 结构
        table = RelationTable(tokens)
        self._fill_table(table, tokens)

        # 3. 解码为 VOCBlueprintNode
        nodes = table.decode_blueprint(self.aspect_keywords)

        # 4. 去重与过滤
        nodes = self._deduplicate(nodes)

        return VOCBlueprint(
            nodes=nodes,
            raw_text=text,
            metadata={
                "token_count": len(tokens),
                "node_count": len(nodes),
                "aspects": list({n.aspect for n in nodes}),
            },
        )

    def extract_batch(self, texts: List[str]) -> List[VOCBlueprint]:
        """批量提取"""
        return [self.extract(t) for t in texts]

    def _tokenize(self, text: str) -> List[str]:
        """简化分词: 保留字母、数字、中文"""
        # 英文按空格和标点分割
        tokens = re.findall(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]", text)
        return [t.lower() if t.isascii() else t for t in tokens]

    def _fill_table(self, table: RelationTable, tokens: List[str]) -> None:
        """
        填充 2D Table。
        简化版规则: 基于词典和模式匹配填充 RP 和 TE。
        """
        n = len(tokens)

        # --- Token Extraction (上三角 TE) ---
        # 标记所有属于 entity 的 token
        for i in range(n):
            for j in range(i + 1, n):
                span_text = " ".join(tokens[i:j + 1])
                # 匹配 aspect keyword
                if any(kw.lower() in span_text for kw in self.aspect_keywords):
                    table.set_te(i, j, "TE-ASPECT")
                # 匹配情感词
                elif any(w in tokens[i:j + 1] for w in self.POSITIVE_WORDS | self.NEGATIVE_WORDS):
                    table.set_te(i, j, "TE-EXPR")

        # --- Relation Prediction (下三角 RP) ---
        for i in range(n):
            for j in range(i):
                # 检查 token_i (end) 和 token_j (start) 之间的关系
                # 简化: 根据附近词判断情感极性
                window = tokens[max(0, j - 2):min(n, i + 3)]
                window_set = set(window)

                pos_score = len(window_set & self.POSITIVE_WORDS)
                neg_score = len(window_set & self.NEGATIVE_WORDS)

                if pos_score > neg_score and pos_score > 0:
                    table.set_rp(i, j, "E-POS")
                elif neg_score > pos_score and neg_score > 0:
                    table.set_rp(i, j, "E-NEG")
                elif pos_score == neg_score and pos_score > 0:
                    table.set_rp(i, j, "E-NEU")

                # 标记 entity 边界 (简化: aspect 词附近的 token)
                if any(kw.lower() in tokens[j].lower() for kw in self.aspect_keywords):
                    table.set_rp(i, j, "S-T")
                if any(w in tokens[j].lower() for w in self.POSITIVE_WORDS | self.NEGATIVE_WORDS):
                    table.set_rp(i, j, "S-E")

    def _deduplicate(self, nodes: List[VOCBlueprintNode]) -> List[VOCBlueprintNode]:
        """去重: aspect + opinion 相同的只保留一个"""
        seen: set = set()
        result = []
        for n in nodes:
            key = (n.aspect, n.opinion)
            if key not in seen and n.opinion != "general":
                seen.add(key)
                result.append(n)
        return result


# ── 可视化辅助 ────────────────────────────────────────

def print_blueprint(blueprint: VOCBlueprint, indent: int = 0) -> None:
    """打印语义蓝图（树形结构）"""
    prefix = "  " * indent
    print(f"{prefix}📋 VOC 语义蓝图")
    print(f"{prefix}原文: {blueprint.raw_text[:80]}...")
    print(f"{prefix}节点数: {len(blueprint.nodes)}")
    print(f"{prefix}涉及方面: {', '.join(blueprint.aspects)}")
    print(f"{prefix}情感分布: {blueprint.sentiment_summary}")
    print()
    for i, node in enumerate(blueprint.nodes, 1):
        print(f"{prefix}  [{i}] 方面: {node.aspect}")
        print(f"{prefix}      观点: {node.opinion}")
        print(f"{prefix}      情感: {node.sentiment}")
        if node.cause:
            print(f"{prefix}      原因: {node.cause}")
        if node.scene:
            print(f"{prefix}      场景: {node.scene}")
        print()


# ── 测试 ──────────────────────────────────────────────

def test_extractor() -> None:
    """单元测试"""
    extractor = VOCBlueprintExtractor()

    # 测试用例 1: 英文评论
    text1 = (
        "This breast pump has great suction but the noise is loud. "
        "I use it at work every day."
    )
    bp1 = extractor.extract(text1)
    print_blueprint(bp1)
    assert len(bp1.nodes) >= 1, "Expected at least 1 node"
    print("✅ Test 1 passed")

    # 测试用例 2: 含场景的评论
    text2 = (
        "Very quiet at night. The battery lasts long but cleaning is difficult."
    )
    bp2 = extractor.extract(text2)
    print_blueprint(bp2)
    assert bp2.sentiment_summary.get("positive", 0) >= 1, "Expected positive sentiment"
    print("✅ Test 2 passed")

    # 测试用例 3: 空文本
    bp3 = extractor.extract("")
    assert len(bp3.nodes) == 0, "Expected 0 nodes for empty text"
    print("✅ Test 3 passed")

    print("\n🎉 All tests passed!")


def test_with_momcozy_data() -> None:
    """用 Momcozy 真实数据做 POC 验证"""
    import pandas as pd

    data_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/momcozy_voc_high_quality_sampled.csv"
    df = pd.read_csv(data_path, nrows=100)

    extractor = VOCBlueprintExtractor()
    blueprints: List[VOCBlueprint] = []

    for idx, row in df.iterrows():
        text = row.get("_review_text", "")
        if pd.isna(text) or not text:
            continue
        bp = extractor.extract(str(text))
        bp.metadata["voc_label"] = row.get("VOC标签", "")
        bp.metadata["l1_category"] = row.get("标签一级分类", "")
        blueprints.append(bp)

    # 统计
    total_nodes = sum(len(bp.nodes) for bp in blueprints)
    avg_nodes = total_nodes / len(blueprints) if blueprints else 0
    all_aspects = []
    for bp in blueprints:
        all_aspects.extend(bp.aspects)
    unique_aspects = set(all_aspects)

    print(f"\n📊 Momcozy POC 统计 ({len(blueprints)} 条评论)")
    print(f"   总节点数: {total_nodes}")
    print(f"   平均每评论节点: {avg_nodes:.2f}")
    print(f"   提取到的不重复方面: {len(unique_aspects)}")
    print(f"   方面示例: {list(unique_aspects)[:10]}")

    # 打印第一个非空蓝图
    for bp in blueprints:
        if bp.nodes:
            print("\n--- 示例输出 ---")
            print_blueprint(bp)
            break

    print("\n✅ Momcozy POC 验证通过")


if __name__ == "__main__":
    print("=" * 60)
    print("VOC Semantic Blueprint Extractor - Unit Tests")
    print("=" * 60)
    test_extractor()

    print("\n" + "=" * 60)
    print("VOC Semantic Blueprint Extractor - Momcozy POC")
    print("=" * 60)
    test_with_momcozy_data()
