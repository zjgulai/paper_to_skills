"""Persona × AIPL 指标体系矩阵

将 VOC 萃取结果按 6维画像 × 7节点AIPL 组织为结构化矩阵，
支撑监控、分析、决策三层价值洞察。

Usage:
    from persona_aipl_matrix import PersonaAIPLMatrixBuilder

    builder = PersonaAIPLMatrixBuilder()
    matrix = builder.build(extractions)

    # 监控层: 查看全量矩阵
    print(matrix.to_dict())

    # 分析层: 下钻到特定画像维度
    who_matrix = matrix.get_dimension_slice("WHO")

    # 决策层: 获取异常格子
    anomalies = matrix.detect_anomalies()
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from unified_label_extraction import VOCLabelExtraction


# ---------------------------------------------------------------------------
# 1. 常量定义
# ---------------------------------------------------------------------------

PERSONA_DIMENSIONS = ["WHO", "WHY", "WHAT", "WHEN", "HOW", "EMOTION"]
AIPL_NODES = ["A", "I", "P1", "P2", "L1", "L2", "L3"]

# 画像维度中文映射
DIMENSION_NAMES = {
    "WHO": "人群身份",
    "WHY": "决策动机",
    "WHAT": "关注方面",
    "WHEN": "使用场景",
    "HOW": "行为模式",
    "EMOTION": "情感状态",
}

# AIPL 节点中文映射
AIPL_NODE_NAMES = {
    "A": "认知 Awareness",
    "I": "兴趣 Interest",
    "P1": "首购 Purchase-1st",
    "P2": "复购 Purchase-Repeat",
    "L1": "活跃 Loyalty-Engage",
    "L2": "推荐 Loyalty-Advocacy",
    "L3": "超级 Loyalty-Champion",
}

# 矩阵单元指标定义（每个交叉格子的口径）
CELL_METRICS = {
    "count": {"type": "int", "desc": "该画像维度在该AIPL阶段的VOC条数"},
    "mention_rate": {"type": "float", "desc": "提及率 = count / 总VOC数", "unit": "%"},
    "avg_sentiment": {"type": "float", "desc": "平均情感极性", "range": "[-1.0, +1.0]"},
    "proxy_nps": {"type": "float", "desc": "Proxy NPS = Promoter% - Detractor%", "range": "[-100, +100]"},
    "top_themes": {"type": "list", "desc": "Top 3 主题及提及次数"},
    "sentiment_distribution": {"type": "dict", "desc": "情感分布: positive/neutral/negative"},
}


# ---------------------------------------------------------------------------
# 2. 数据模型
# ---------------------------------------------------------------------------

@dataclass
class MatrixCell:
    """画像×AIPL矩阵中的一个单元格"""

    persona_dim: str          # WHO/WHY/WHAT/WHEN/HOW/EMOTION
    aipl_node: str            # A/I/P1/P2/L1/L2/L3
    count: int = 0
    sentiments: list[float] = field(default_factory=list)
    themes: Counter[str] = field(default_factory=Counter)
    proxy_nps_counts: dict[str, int] = field(default_factory=lambda: {"promoter": 0, "passive": 0, "detractor": 0})
    # 子维度细分（如 WHO 下细分为 family_role/parenting_stage）
    sub_dimensions: dict[str, int] = field(default_factory=dict)

    def avg_sentiment(self) -> float:
        return round(sum(self.sentiments) / len(self.sentiments), 2) if self.sentiments else 0.0

    def mention_rate(self, total: int) -> float:
        return round(self.count / total, 3) if total > 0 else 0.0

    def proxy_nps(self) -> float:
        p = self.proxy_nps_counts["promoter"]
        d = self.proxy_nps_counts["detractor"]
        n = self.proxy_nps_counts["passive"]
        total = p + d + n
        return round((p / total * 100) - (d / total * 100), 1) if total > 0 else 0.0

    def sentiment_distribution(self) -> dict[str, int]:
        pos = sum(1 for s in self.sentiments if s > 0.2)
        neg = sum(1 for s in self.sentiments if s < -0.2)
        neu = len(self.sentiments) - pos - neg
        return {"positive": pos, "neutral": neu, "negative": neg}

    def top_themes(self, k: int = 3) -> list[dict[str, Any]]:
        return [{"theme": t, "count": c} for t, c in self.themes.most_common(k)]

    def to_dict(self, total: int = 1) -> dict[str, Any]:
        return {
            "count": self.count,
            "mention_rate": self.mention_rate(total),
            "avg_sentiment": self.avg_sentiment(),
            "proxy_nps": self.proxy_nps(),
            "sentiment_distribution": self.sentiment_distribution(),
            "top_themes": self.top_themes(),
            "sub_dimensions": dict(self.sub_dimensions.most_common()) if hasattr(self.sub_dimensions, 'most_common') else self.sub_dimensions,
        }


@dataclass
class PersonaAIPLMatrix:
    """画像×AIPL完整矩阵"""

    # 矩阵数据: matrix[persona_dim][aipl_node] = MatrixCell
    cells: dict[str, dict[str, MatrixCell]] = field(default_factory=dict)
    total_voc: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 确保所有格子都存在
        for dim in PERSONA_DIMENSIONS:
            if dim not in self.cells:
                self.cells[dim] = {}
            for node in AIPL_NODES:
                if node not in self.cells[dim]:
                    self.cells[dim][node] = MatrixCell(persona_dim=dim, aipl_node=node)

    def get_cell(self, persona_dim: str, aipl_node: str) -> MatrixCell:
        return self.cells.get(persona_dim, {}).get(aipl_node, MatrixCell(persona_dim, aipl_node))

    def get_dimension_slice(self, persona_dim: str) -> dict[str, MatrixCell]:
        """获取某个画像维度在所有AIPL节点的切片"""
        return self.cells.get(persona_dim, {})

    def get_aipl_slice(self, aipl_node: str) -> dict[str, MatrixCell]:
        """获取某个AIPL节点在所有画像维度的切片"""
        return {dim: self.cells[dim][aipl_node] for dim in PERSONA_DIMENSIONS if aipl_node in self.cells[dim]}

    def get_row_totals(self, persona_dim: str) -> dict[str, Any]:
        """某画像维度的行汇总"""
        cells = self.get_dimension_slice(persona_dim).values()
        total_count = sum(c.count for c in cells)
        all_sentiments = [s for c in cells for s in c.sentiments]
        return {
            "dimension": persona_dim,
            "dimension_name": DIMENSION_NAMES.get(persona_dim, ""),
            "total_count": total_count,
            "avg_sentiment": round(sum(all_sentiments) / len(all_sentiments), 2) if all_sentiments else 0.0,
            "by_aipl_node": {node: cell.to_dict(self.total_voc) for node, cell in self.get_dimension_slice(persona_dim).items()},
        }

    def get_column_totals(self, aipl_node: str) -> dict[str, Any]:
        """某AIPL节点的列汇总"""
        cells = self.get_aipl_slice(aipl_node).values()
        total_count = sum(c.count for c in cells)
        all_sentiments = [s for c in cells for s in c.sentiments]
        return {
            "aipl_node": aipl_node,
            "node_name": AIPL_NODE_NAMES.get(aipl_node, ""),
            "total_count": total_count,
            "avg_sentiment": round(sum(all_sentiments) / len(all_sentiments), 2) if all_sentiments else 0.0,
            "by_persona_dim": {dim: cell.to_dict(self.total_voc) for dim, cell in self.get_aipl_slice(aipl_node).items()},
        }

    def detect_anomalies(self, thresholds: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
        """检测异常格子（监控层）

        异常规则:
        1. avg_sentiment < -0.5 且 count > 5: 严重负面信号
        2. proxy_nps < 0 且 mention_rate > 0.05: NPS异常
        3. count 在该维度所有节点中占比 > 50%: 阶段集中度过高
        """
        thresholds = thresholds or {}
        sentiment_threshold = thresholds.get("sentiment", -0.5)
        nps_threshold = thresholds.get("proxy_nps", 0.0)
        mention_rate_threshold = thresholds.get("mention_rate", 0.05)

        anomalies = []

        for dim in PERSONA_DIMENSIONS:
            dim_cells = list(self.get_dimension_slice(dim).values())
            dim_total = sum(c.count for c in dim_cells)

            for cell in dim_cells:
                reasons = []

                # 规则1: 严重负面
                if cell.avg_sentiment() < sentiment_threshold and cell.count > 5:
                    reasons.append(f"情感极性{cell.avg_sentiment():.2f}低于阈值{sentiment_threshold}")

                # 规则2: NPS异常
                if cell.proxy_nps() < nps_threshold and cell.mention_rate(self.total_voc) > mention_rate_threshold:
                    reasons.append(f"Proxy NPS {cell.proxy_nps():.1f}低于阈值{nps_threshold}")

                # 规则3: 阶段集中度过高
                if dim_total > 0 and cell.count / dim_total > 0.5 and cell.count > 10:
                    reasons.append(f"该维度{cell.count/dim_total*100:.0f}%集中在该阶段")

                if reasons:
                    anomalies.append({
                        "persona_dim": dim,
                        "aipl_node": cell.aipl_node,
                        "dimension_name": DIMENSION_NAMES.get(dim, ""),
                        "node_name": AIPL_NODE_NAMES.get(cell.aipl_node, ""),
                        "count": cell.count,
                        "mention_rate": cell.mention_rate(self.total_voc),
                        "avg_sentiment": cell.avg_sentiment(),
                        "proxy_nps": cell.proxy_nps(),
                        "severity": "high" if cell.avg_sentiment() < -0.7 else "medium",
                        "reasons": reasons,
                        "top_themes": cell.top_themes(),
                    })

        # 按严重程度排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anomalies.sort(key=lambda x: (severity_order.get(x["severity"], 3), -x["count"]))
        return anomalies

    def find_opportunities(self) -> list[dict[str, Any]]:
        """发现机会格子（分析层）

        机会规则:
        1. avg_sentiment > 0.5 且 mention_rate < 0.03: 高满意度但低渗透，扩大推广
        2. count 快速增长（需时序数据）
        3. proxy_nps > 50 但 count 小: 高口碑但小众，破圈潜力
        """
        opportunities = []

        for dim in PERSONA_DIMENSIONS:
            for node in AIPL_NODES:
                cell = self.get_cell(dim, node)
                reasons = []

                # 规则1: 高满意度低渗透
                if cell.avg_sentiment() > 0.5 and 0 < cell.mention_rate(self.total_voc) < 0.03:
                    reasons.append("高满意度但低渗透率，存在扩大推广空间")

                # 规则2: 高口碑小众
                if cell.proxy_nps() > 50 and 3 < cell.count < 20:
                    reasons.append("高NPS但规模小，有破圈潜力")

                # 规则3: 正向情感但NPS一般（被动者多，激活空间大）
                if cell.avg_sentiment() > 0.3 and -10 < cell.proxy_nps() < 20 and cell.count > 10:
                    reasons.append("情感正面但推荐意愿不足，激活空间大")

                if reasons:
                    opportunities.append({
                        "persona_dim": dim,
                        "aipl_node": node,
                        "dimension_name": DIMENSION_NAMES.get(dim, ""),
                        "node_name": AIPL_NODE_NAMES.get(node, ""),
                        "count": cell.count,
                        "mention_rate": cell.mention_rate(self.total_voc),
                        "avg_sentiment": cell.avg_sentiment(),
                        "proxy_nps": cell.proxy_nps(),
                        "reasons": reasons,
                        "top_themes": cell.top_themes(),
                    })

        # 按潜力排序（mention_rate 越低但 sentiment 越高 = 潜力越大）
        opportunities.sort(key=lambda x: x["avg_sentiment"] - x["mention_rate"] * 10, reverse=True)
        return opportunities

    def compare_dimensions(self, dim_a: str, dim_b: str) -> dict[str, Any]:
        """对比两个画像维度在各AIPL节点的差异（分析层）"""
        comparison = {}
        for node in AIPL_NODES:
            cell_a = self.get_cell(dim_a, node)
            cell_b = self.get_cell(dim_b, node)

            comparison[node] = {
                "node_name": AIPL_NODE_NAMES.get(node, ""),
                dim_a: {
                    "count": cell_a.count,
                    "avg_sentiment": cell_a.avg_sentiment(),
                    "proxy_nps": cell_a.proxy_nps(),
                },
                dim_b: {
                    "count": cell_b.count,
                    "avg_sentiment": cell_b.avg_sentiment(),
                    "proxy_nps": cell_b.proxy_nps(),
                },
                "sentiment_diff": round(cell_a.avg_sentiment() - cell_b.avg_sentiment(), 2),
                "nps_diff": round(cell_a.proxy_nps() - cell_b.proxy_nps(), 1),
            }
        return comparison

    def to_dict(self) -> dict[str, Any]:
        """全量矩阵输出"""
        return {
            "metadata": {
                "total_voc": self.total_voc,
                **self.metadata,
            },
            "dimensions": {dim: DIMENSION_NAMES[dim] for dim in PERSONA_DIMENSIONS},
            "aipl_nodes": {node: AIPL_NODE_NAMES[node] for node in AIPL_NODES},
            "matrix": {
                dim: {
                    node: cell.to_dict(self.total_voc)
                    for node, cell in nodes.items()
                }
                for dim, nodes in self.cells.items()
            },
            "row_totals": {dim: self.get_row_totals(dim) for dim in PERSONA_DIMENSIONS},
            "column_totals": {node: self.get_column_totals(node) for node in AIPL_NODES},
        }


# ---------------------------------------------------------------------------
# 3. 矩阵构建器
# ---------------------------------------------------------------------------

class PersonaAIPLMatrixBuilder:
    """从 VOCLabelExtraction 列表构建画像×AIPL矩阵"""

    def build(
        self,
        extractions: list[VOCLabelExtraction],
        metadata: Optional[dict[str, Any]] = None,
    ) -> PersonaAIPLMatrix:
        """构建完整矩阵"""
        valid = [e for e in extractions if not e.is_suspicious]
        total = len(valid)

        # 初始化矩阵
        cells: dict[str, dict[str, MatrixCell]] = {}
        for dim in PERSONA_DIMENSIONS:
            cells[dim] = {}
            for node in AIPL_NODES:
                cells[dim][node] = MatrixCell(persona_dim=dim, aipl_node=node)

        # 填充数据
        for e in valid:
            for dim in PERSONA_DIMENSIONS:
                dim_tags = e.persona_dimensions.get(dim, [])
                if not dim_tags:
                    continue

                cell = cells[dim][e.aipl_stage]
                cell.count += 1
                cell.sentiments.append(e.sentiment_polarity)

                for tag in e.aipl_tags:
                    cell.themes[tag.theme] += 1

                # Proxy NPS
                if e.proxy_nps_contribution == "promoter":
                    cell.proxy_nps_counts["promoter"] += 1
                elif e.proxy_nps_contribution == "detractor":
                    cell.proxy_nps_counts["detractor"] += 1
                else:
                    cell.proxy_nps_counts["passive"] += 1

                # 子维度统计
                # 从 atomic tags 的子维度信息中提取
                for tag_name in dim_tags:
                    cell.sub_dimensions[tag_name] = cell.sub_dimensions.get(tag_name, 0) + 1

        return PersonaAIPLMatrix(
            cells=cells,
            total_voc=total,
            metadata=metadata or {},
        )

    def build_by_slice(
        self,
        extractions: list[VOCLabelExtraction],
        slice_by: str,  # "product_line" / "platform" / "time_period"
    ) -> dict[str, PersonaAIPLMatrix]:
        """按维度分片构建多个矩阵"""
        slices: dict[str, list[VOCLabelExtraction]] = defaultdict(list)

        for e in extractions:
            key = getattr(e, slice_by, "unknown")
            slices[key].append(e)

        return {
            key: self.build(items, metadata={"slice_by": slice_by, "slice_value": key})
            for key, items in slices.items()
        }


# ---------------------------------------------------------------------------
# 4. 演示
# ---------------------------------------------------------------------------

def demo():
    """演示：从模拟数据构建矩阵并输出洞察"""
    from unified_label_extraction import VOCLabelExtraction, AIPLTagMatch

    print("=" * 70)
    print("Persona × AIPL 指标体系矩阵 - 演示")
    print("=" * 70)

    # 构造模拟萃取结果
    extractions = []

    # 模拟: 20条 VOC，覆盖不同画像维度 × AIPL阶段
    demo_data = [
        # WHO=working_parent × A/I: 搜索/对比阶段的职场妈妈
        ("A", {"WHO": ["working_parent"]}, -0.1, "passive"),
        ("A", {"WHO": ["working_parent"]}, 0.2, "passive"),
        ("I", {"WHO": ["working_parent"], "WHAT": ["quiet_seeker"]}, 0.3, "passive"),
        ("I", {"WHO": ["working_parent"], "WHAT": ["quiet_seeker", "portable_seeker"]}, 0.5, "promoter"),
        ("P1", {"WHO": ["working_parent"], "WHAT": ["quiet_seeker"]}, -0.6, "detractor"),  # 异常: 买了但噪音不满
        ("P1", {"WHO": ["working_parent"]}, 0.4, "promoter"),
        ("L1", {"WHO": ["working_parent"], "WHAT": ["easy_clean_seeker"]}, 0.6, "promoter"),

        # WHO=first_time_parent × A/I/P1: 新手妈妈
        ("A", {"WHO": ["first_time_parent"], "WHY": ["anxiety_driven"]}, 0.1, "passive"),
        ("A", {"WHO": ["first_time_parent"]}, 0.0, "passive"),
        ("I", {"WHO": ["first_time_parent"], "HOW": ["research_driven"]}, 0.4, "passive"),
        ("P1", {"WHO": ["first_time_parent"], "EMOTION": ["anxiety_driven"]}, -0.3, "detractor"),
        ("P1", {"WHO": ["first_time_parent"]}, 0.5, "promoter"),
        ("L1", {"WHO": ["first_time_parent"]}, 0.7, "promoter"),

        # WHAT=quiet_seeker × 各阶段: 静音敏感型贯穿全链路
        ("I", {"WHAT": ["quiet_seeker"], "WHEN": ["workplace_user"]}, 0.2, "passive"),
        ("P1", {"WHAT": ["quiet_seeker"]}, -0.4, "detractor"),
        ("P1", {"WHAT": ["quiet_seeker"]}, -0.5, "detractor"),  # 又一个负面
        ("L2", {"WHAT": ["quiet_seeker", "hands_free_seeker"]}, 0.8, "promoter"),

        # WHY=price_sensitive × I: 价格敏感型在兴趣阶段
        ("I", {"WHY": ["price_sensitive", "budget_conscious"]}, 0.3, "passive"),
        ("I", {"WHY": ["price_sensitive"]}, 0.1, "passive"),
        ("P1", {"WHY": ["price_sensitive"]}, -0.2, "detractor"),
    ]

    for i, (stage, dims, sentiment, nps) in enumerate(demo_data):
        tag = AIPLTagMatch(
            tag_id=f"TAG_{i}",
            tag_en="demo_tag",
            tag_cn="演示标签",
            theme="产品核心性能",
            aipl_node=stage,
            sentiment_preset="neutral",
            sentiment_calibrated=sentiment,
            confidence=0.8,
        )
        extractions.append(VOCLabelExtraction(
            review_id=f"REV_{i:03d}",
            source_type="review",
            platform="amazon",
            spu_code="SPU001",
            product_line="breast_pump",
            category="wearable_pump",
            rating=3.0 + sentiment,
            aipl_stage=stage,
            aipl_tags=[tag],
            persona_dimensions=dims,
            sentiment_polarity=sentiment,
            proxy_nps_contribution=nps,
        ))

    # 构建矩阵
    builder = PersonaAIPLMatrixBuilder()
    matrix = builder.build(extractions, metadata={"product_line": "breast_pump", "platform": "amazon"})

    print(f"\n总VOC数: {matrix.total_voc}")

    # 监控层: 异常检测
    print("\n--- 监控层: 异常检测 ---")
    anomalies = matrix.detect_anomalies()
    for a in anomalies[:5]:
        print(f"  [{a['severity'].upper()}] {a['dimension_name']} × {a['node_name']}")
        print(f"    count={a['count']}, sentiment={a['avg_sentiment']}, NPS={a['proxy_nps']}")
        for r in a['reasons']:
            print(f"    → {r}")

    # 分析层: 机会发现
    print("\n--- 分析层: 机会识别 ---")
    opportunities = matrix.find_opportunities()
    for o in opportunities[:3]:
        print(f"  {o['dimension_name']} × {o['node_name']}")
        print(f"    count={o['count']}, sentiment={o['avg_sentiment']}, NPS={o['proxy_nps']}")
        for r in o['reasons']:
            print(f"    → {r}")

    # 分析层: 维度对比
    print("\n--- 分析层: WHO维度对比 ---")
    who_slice = matrix.get_dimension_slice("WHO")
    for node, cell in who_slice.items():
        if cell.count > 0:
            print(f"  {AIPL_NODE_NAMES[node]}: count={cell.count}, sentiment={cell.avg_sentiment():+.2f}, NPS={cell.proxy_nps():+.1f}")

    # 输出全量矩阵
    print("\n--- 矩阵快照 (WHAT维度) ---")
    what_slice = matrix.get_dimension_slice("WHAT")
    for node, cell in what_slice.items():
        if cell.count > 0:
            print(f"  {node}: count={cell.count}, sentiment={cell.avg_sentiment():+.2f}, NPS={cell.proxy_nps():+.1f}")

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    demo()
