"""三层价值洞察引擎

基于 Persona × AIPL 矩阵，实现:
- 监控层 (Monitoring): KPI看板、异常告警、阈值监控
- 分析层 (Analysis): 下钻分析、根因分析、对比分析、趋势追踪
- 决策层 (Decision): 策略推荐、自动路由、优先级排序、行动建议

Usage:
    from insight_engine import ValueInsightEngine

    engine = ValueInsightEngine()

    # 监控层
    monitoring = engine.monitoring(matrix)

    # 分析层
    analysis = engine.analysis(matrix, drill_down={"persona_dim": "WHO", "aipl_node": "P1"})

    # 决策层
    decisions = engine.decision(matrix, extractions)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from persona_aipl_matrix import PersonaAIPLMatrix, MatrixCell, PERSONA_DIMENSIONS, AIPL_NODES
from unified_label_extraction import VOCLabelExtraction


# ---------------------------------------------------------------------------
# 1. 监控层 (Monitoring Layer)
# ---------------------------------------------------------------------------

@dataclass
class MonitoringReport:
    """监控层报告"""

    kpi_snapshot: dict[str, Any] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    health_score: float = 0.0  # 0-100


class MonitoringLayer:
    """监控层: 实时监控画像×AIPL指标体系的健康状态"""

    # 默认阈值配置
    DEFAULT_THRESHOLDS = {
        "sentiment_critical": -0.5,
        "sentiment_warning": -0.3,
        "proxy_nps_critical": 0,
        "proxy_nps_warning": 20,
        "mention_rate_min": 0.02,
        "count_min": 5,
    }

    def __init__(self, thresholds: Optional[dict[str, float]] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def run(self, matrix: PersonaAIPLMatrix) -> MonitoringReport:
        """执行监控扫描"""
        kpi = self._kpi_snapshot(matrix)
        anomalies = matrix.detect_anomalies(self.thresholds)
        alerts = self._generate_alerts(matrix, anomalies)
        health = self._calc_health_score(matrix, anomalies)

        return MonitoringReport(
            kpi_snapshot=kpi,
            anomalies=anomalies,
            alerts=alerts,
            health_score=health,
        )

    def _kpi_snapshot(self, matrix: PersonaAIPLMatrix) -> dict[str, Any]:
        """生成KPI快照"""
        # 全量聚合
        all_sentiments = []
        all_promoters = all_detractors = all_passive = 0

        for dim in PERSONA_DIMENSIONS:
            for node in AIPL_NODES:
                cell = matrix.get_cell(dim, node)
                all_sentiments.extend(cell.sentiments)
                all_promoters += cell.proxy_nps_counts["promoter"]
                all_detractors += cell.proxy_nps_counts["detractor"]
                all_passive += cell.proxy_nps_counts["passive"]

        total_nps = all_promoters + all_detractors + all_passive
        proxy_nps = round((all_promoters / total_nps * 100) - (all_detractors / total_nps * 100), 1) if total_nps > 0 else 0.0

        return {
            "total_voc": matrix.total_voc,
            "overall_proxy_nps": proxy_nps,
            "overall_avg_sentiment": round(sum(all_sentiments) / len(all_sentiments), 2) if all_sentiments else 0.0,
            "promoter_rate": round(all_promoters / total_nps, 3) if total_nps > 0 else 0.0,
            "detractor_rate": round(all_detractors / total_nps, 3) if total_nps > 0 else 0.0,
            "aipl_distribution": {
                node: sum(matrix.get_cell(dim, node).count for dim in PERSONA_DIMENSIONS)
                for node in AIPL_NODES
            },
            "persona_distribution": {
                dim: sum(matrix.get_cell(dim, node).count for node in AIPL_NODES)
                for dim in PERSONA_DIMENSIONS
            },
        }

    def _generate_alerts(
        self,
        matrix: PersonaAIPLMatrix,
        anomalies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """生成结构化告警"""
        alerts = []

        for a in anomalies:
            alert = {
                "level": "critical" if a["severity"] == "high" else "warning",
                "category": "sentiment_drop" if "情感" in str(a["reasons"]) else "nps_drop",
                "target": f"{a['dimension_name']} × {a['node_name']}",
                "metric_value": a["avg_sentiment"] if "情感" in str(a["reasons"]) else a["proxy_nps"],
                "threshold": self.thresholds["sentiment_critical"] if "情感" in str(a["reasons"]) else self.thresholds["proxy_nps_critical"],
                "description": "; ".join(a["reasons"]),
                "recommended_action": self._alert_action(a),
                "top_themes": a.get("top_themes", []),
            }
            alerts.append(alert)

        # 按优先级排序
        alerts.sort(key=lambda x: (0 if x["level"] == "critical" else 1, x["metric_value"]))
        return alerts

    def _alert_action(self, anomaly: dict[str, Any]) -> str:
        """根据异常类型推荐监控动作"""
        dim = anomaly["persona_dim"]
        node = anomaly["aipl_node"]

        actions = {
            ("WHAT", "P1"): "立即排查产品核心体验问题",
            ("WHAT", "P2"): "关注复购驱动因素，检查配件/服务",
            ("WHO", "P1"): "针对该人群优化首购体验",
            ("WHY", "I"): "调整营销信息，匹配用户决策动机",
            ("EMOTION", "L1"): "启动用户关怀计划，防止流失",
            ("HOW", "A"): "优化触达渠道和内容形式",
        }
        return actions.get((dim, node), f"关注 {anomaly['dimension_name']} 在 {anomaly['node_name']} 阶段的表现")

    def _calc_health_score(self, matrix: PersonaAIPLMatrix, anomalies: list[dict[str, Any]]) -> float:
        """计算整体健康度分数 (0-100)"""
        base_score = 100.0

        # 异常扣分
        high_count = sum(1 for a in anomalies if a["severity"] == "high")
        medium_count = sum(1 for a in anomalies if a["severity"] == "medium")
        base_score -= high_count * 8
        base_score -= medium_count * 3

        # NPS加权
        kpi = self._kpi_snapshot(matrix)
        nps = kpi.get("overall_proxy_nps", 0)
        if nps < 0:
            base_score -= 15
        elif nps < 20:
            base_score -= 5
        elif nps > 50:
            base_score += 5

        return max(0.0, min(100.0, round(base_score, 1)))


# ---------------------------------------------------------------------------
# 2. 分析层 (Analysis Layer)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisReport:
    """分析层报告"""

    drill_down: dict[str, Any] = field(default_factory=dict)
    root_cause: dict[str, Any] = field(default_factory=dict)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    opportunities: list[dict[str, Any]] = field(default_factory=list)
    trend_signals: list[dict[str, Any]] = field(default_factory=list)


class AnalysisLayer:
    """分析层: 深度分析矩阵数据，支撑业务洞察"""

    def run(
        self,
        matrix: PersonaAIPLMatrix,
        extractions: list[VOCLabelExtraction],
        drill_down: Optional[dict[str, str]] = None,
    ) -> AnalysisReport:
        """执行分析"""
        return AnalysisReport(
            drill_down=self._drill_down(matrix, drill_down),
            root_cause=self._root_cause_analysis(matrix, extractions, drill_down),
            comparisons=self._cross_comparisons(matrix),
            opportunities=matrix.find_opportunities(),
            trend_signals=self._trend_signals(matrix),
        )

    def _drill_down(
        self,
        matrix: PersonaAIPLMatrix,
        drill_down: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """下钻分析: 从矩阵格子深入到具体主题和子维度"""
        if not drill_down:
            return {"status": "no_drill_down_specified"}

        dim = drill_down.get("persona_dim")
        node = drill_down.get("aipl_node")

        if not dim or not node:
            return {"status": "invalid_drill_down"}

        cell = matrix.get_cell(dim, node)

        return {
            "target": f"{dim} × {node}",
            "count": cell.count,
            "mention_rate": cell.mention_rate(matrix.total_voc),
            "avg_sentiment": cell.avg_sentiment(),
            "proxy_nps": cell.proxy_nps(),
            "sentiment_distribution": cell.sentiment_distribution(),
            "top_themes": cell.top_themes(5),
            "sub_dimensions": dict(Counter(cell.sub_dimensions).most_common(10)) if isinstance(cell.sub_dimensions, dict) else cell.sub_dimensions,
            "diagnosis": self._diagnose_cell(cell),
        }

    def _diagnose_cell(self, cell: MatrixCell) -> str:
        """对单个格子做诊断"""
        if cell.avg_sentiment() < -0.5:
            return f"严重负面: 该画像维度在{cell.aipl_node}阶段出现系统性不满"
        elif cell.avg_sentiment() < -0.2:
            return f"轻度负面: 存在改进空间，建议关注Top主题"
        elif cell.avg_sentiment() > 0.5 and cell.mention_rate(cell.count) < 0.03:
            return f"高满意度低渗透: 有扩大推广的潜力"
        elif cell.proxy_nps() > 50:
            return f"口碑优秀: 可作为案例传播"
        return "表现正常"

    def _root_cause_analysis(
        self,
        matrix: PersonaAIPLMatrix,
        extractions: list[VOCLabelExtraction],
        drill_down: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """根因分析: 找出负面指标背后的具体原因"""
        if not drill_down:
            return {"status": "specify_drill_down_for_root_cause"}

        dim = drill_down.get("persona_dim")
        node = drill_down.get("aipl_node")

        # 过滤出目标格子的原始VOC
        target_vocs = [
            e for e in extractions
            if e.aipl_stage == node and dim in e.persona_dimensions and e.persona_dimensions[dim]
        ]

        # 主题归因
        theme_sentiments: dict[str, list[float]] = defaultdict(list)
        for e in target_vocs:
            for tag in e.aipl_tags:
                theme_sentiments[tag.theme].append(e.sentiment_polarity)

        theme_analysis = []
        for theme, sentiments in sorted(theme_sentiments.items(), key=lambda x: len(x[1]), reverse=True):
            avg = sum(sentiments) / len(sentiments)
            theme_analysis.append({
                "theme": theme,
                "count": len(sentiments),
                "avg_sentiment": round(avg, 2),
                "impact": "high" if abs(avg) > 0.5 and len(sentiments) > 3 else "medium" if len(sentiments) > 3 else "low",
            })

        # 情感原因归因（从VOC语义蓝图角度）
        negative_vocs = [e for e in target_vocs if e.sentiment_polarity < -0.2]
        negative_themes = Counter()
        for e in negative_vocs:
            for tag in e.aipl_tags:
                negative_themes[tag.theme] += 1

        return {
            "target": f"{dim} × {node}",
            "sample_size": len(target_vocs),
            "theme_breakdown": theme_analysis[:5],
            "top_negative_themes": [{"theme": t, "count": c} for t, c in negative_themes.most_common(5)],
            "root_cause_summary": self._summarize_root_cause(theme_analysis[:3]),
        }

    def _summarize_root_cause(self, top_themes: list[dict[str, Any]]) -> str:
        """总结根因"""
        if not top_themes:
            return "暂无足够数据"

        negative_themes = [t for t in top_themes if t["avg_sentiment"] < -0.3]
        if negative_themes:
            themes_str = ", ".join([t["theme"] for t in negative_themes])
            return f"主要由 [{themes_str}] 方面的负面体验驱动"

        positive_themes = [t for t in top_themes if t["avg_sentiment"] > 0.3]
        if positive_themes:
            themes_str = ", ".join([t["theme"] for t in positive_themes])
            return f"主要由 [{themes_str}] 方面的正面体验驱动"

        return "情感分布较为分散，无单一主导因素"

    def _cross_comparisons(self, matrix: PersonaAIPLMatrix) -> list[dict[str, Any]]:
        """跨维度对比分析"""
        comparisons = []

        # 对比各维度在 P1 阶段的表现（首购体验是关键）
        p1_cells = {dim: matrix.get_cell(dim, "P1") for dim in PERSONA_DIMENSIONS}
        p1_sentiments = {dim: c.avg_sentiment() for dim, c in p1_cells.items() if c.count > 0}

        if len(p1_sentiments) >= 2:
            best_dim = max(p1_sentiments, key=p1_sentiments.get)
            worst_dim = min(p1_sentiments, key=p1_sentiments.get)
            comparisons.append({
                "type": "p1_experience_gap",
                "description": f"首购体验差距: {best_dim}({p1_sentiments[best_dim]:+.2f}) vs {worst_dim}({p1_sentiments[worst_dim]:+.2f})",
                "best": {"dim": best_dim, "sentiment": p1_sentiments[best_dim]},
                "worst": {"dim": worst_dim, "sentiment": p1_sentiments[worst_dim]},
                "gap": round(p1_sentiments[best_dim] - p1_sentiments[worst_dim], 2),
            })

        # 对比同一画像维度在 A→I→P1 的转化率信号
        for dim in PERSONA_DIMENSIONS:
            a_cell = matrix.get_cell(dim, "A")
            i_cell = matrix.get_cell(dim, "I")
            p1_cell = matrix.get_cell(dim, "P1")

            if a_cell.count > 0 and i_cell.count > 0:
                a_to_i = i_cell.count / (a_cell.count + i_cell.count)
                i_to_p1 = p1_cell.count / (i_cell.count + p1_cell.count) if (i_cell.count + p1_cell.count) > 0 else 0

                if a_to_i < 0.3 and a_cell.count > 5:
                    comparisons.append({
                        "type": "conversion_bottleneck",
                        "description": f"{dim} 从认知到兴趣的转化偏低 ({a_to_i:.1%})",
                        "dim": dim,
                        "stage": "A→I",
                        "rate": round(a_to_i, 3),
                    })

        return comparisons

    def _trend_signals(self, matrix: PersonaAIPLMatrix) -> list[dict[str, Any]]:
        """趋势信号（基于当前快照的结构性趋势推断）"""
        signals = []

        # 信号1: AIPL漏斗形态分析
        a_count = sum(matrix.get_cell(dim, "A").count for dim in PERSONA_DIMENSIONS)
        i_count = sum(matrix.get_cell(dim, "I").count for dim in PERSONA_DIMENSIONS)
        p1_count = sum(matrix.get_cell(dim, "P1").count for dim in PERSONA_DIMENSIONS)
        l1_count = sum(matrix.get_cell(dim, "L1").count for dim in PERSONA_DIMENSIONS)

        if a_count > 0 and i_count / a_count < 0.5:
            signals.append({
                "type": "funnel_leak",
                "stage": "A→I",
                "severity": "high",
                "description": "大量认知期用户未进入兴趣阶段，需优化内容触达",
            })

        if i_count > 0 and p1_count / i_count < 0.3:
            signals.append({
                "type": "funnel_leak",
                "stage": "I→P1",
                "severity": "high",
                "description": "兴趣期用户转化率低，需优化促销/信任建设",
            })

        if p1_count > 0 and l1_count / p1_count < 0.2:
            signals.append({
                "type": "loyalty_risk",
                "stage": "P1→L1",
                "severity": "medium",
                "description": "首购后活跃率低，需加强 onboarding",
            })

        # 信号2: 画像维度集中度
        for dim in PERSONA_DIMENSIONS:
            dim_total = sum(matrix.get_cell(dim, node).count for node in AIPL_NODES)
            if dim_total > 0:
                max_node = max(AIPL_NODES, key=lambda n: matrix.get_cell(dim, n).count)
                max_ratio = matrix.get_cell(dim, max_node).count / dim_total
                if max_ratio > 0.6:
                    signals.append({
                        "type": "stage_concentration",
                        "dim": dim,
                        "node": max_node,
                        "severity": "low",
                        "description": f"{dim} 画像 {max_ratio:.0%} 集中在 {max_node} 阶段",
                    })

        return signals


# ---------------------------------------------------------------------------
# 3. 决策层 (Decision Layer)
# ---------------------------------------------------------------------------

@dataclass
class DecisionReport:
    """决策层报告"""

    strategies: list[dict[str, Any]] = field(default_factory=list)
    action_queue: list[dict[str, Any]] = field(default_factory=list)
    priority_ranking: list[dict[str, Any]] = field(default_factory=list)
    expected_impact: dict[str, Any] = field(default_factory=dict)


class DecisionLayer:
    """决策层: 基于矩阵洞察生成可执行策略"""

    def run(
        self,
        matrix: PersonaAIPLMatrix,
        extractions: list[VOCLabelExtraction],
        monitoring: MonitoringReport,
        analysis: AnalysisReport,
    ) -> DecisionReport:
        """生成决策建议"""
        strategies = self._generate_strategies(matrix, monitoring, analysis)
        action_queue = self._build_action_queue(strategies)
        priority = self._priority_ranking(strategies)
        impact = self._estimate_impact(strategies, matrix)

        return DecisionReport(
            strategies=strategies,
            action_queue=action_queue,
            priority_ranking=priority,
            expected_impact=impact,
        )

    def _generate_strategies(
        self,
        matrix: PersonaAIPLMatrix,
        monitoring: MonitoringReport,
        analysis: AnalysisReport,
    ) -> list[dict[str, Any]]:
        """生成策略列表"""
        strategies = []

        # 策略1: 针对异常格子的修复策略
        for anomaly in monitoring.anomalies[:5]:
            strategies.append({
                "id": f"STRAT_FIX_{anomaly['persona_dim']}_{anomaly['aipl_node']}",
                "type": "fix",
                "target": f"{anomaly['dimension_name']} × {anomaly['node_name']}",
                "problem": "; ".join(anomaly["reasons"]),
                "strategy": self._fix_strategy(anomaly),
                "owner": self._suggest_owner(anomaly),
                "urgency": "high" if anomaly["severity"] == "high" else "medium",
                "expected_outcome": f"sentiment提升0.3-0.5，NPS提升10-20",
            })

        # 策略2: 针对机会的放大策略
        for opp in analysis.opportunities[:3]:
            strategies.append({
                "id": f"STRAT_GROW_{opp['persona_dim']}_{opp['aipl_node']}",
                "type": "grow",
                "target": f"{opp['dimension_name']} × {opp['node_name']}",
                "opportunity": "; ".join(opp["reasons"]),
                "strategy": self._grow_strategy(opp),
                "owner": "营销增长部",
                "urgency": "medium",
                "expected_outcome": f"mention_rate提升2-3倍，count增长50%+",
            })

        # 策略3: 针对漏斗漏损的转化策略
        for signal in analysis.trend_signals:
            if signal["type"] == "funnel_leak":
                strategies.append({
                    "id": f"STRAT_CONV_{signal['stage']}",
                    "type": "convert",
                    "target": signal["stage"],
                    "problem": signal["description"],
                    "strategy": self._convert_strategy(signal),
                    "owner": "用户增长部",
                    "urgency": signal["severity"],
                    "expected_outcome": f"{signal['stage']}转化率提升15-25%",
                })

        return strategies

    def _fix_strategy(self, anomaly: dict[str, Any]) -> str:
        """生成修复策略"""
        dim = anomaly["persona_dim"]
        node = anomaly["aipl_node"]
        themes = [t["theme"] for t in anomaly.get("top_themes", [])]
        theme_hint = themes[0] if themes else "核心体验"

        strategies = {
            ("WHAT", "P1"): f"针对 [{theme_hint}] 启动产品体验优化专项",
            ("WHAT", "P2"): f"优化 [{theme_hint}] 相关的配件/服务体验",
            ("WHO", "P1"): f"为该人群定制首购 onboarding 流程，重点解决 [{theme_hint}]",
            ("WHY", "I"): f"调整营销信息，强化 [{theme_hint}] 的价值传达",
            ("EMOTION", "L1"): f"启动用户关怀计划，主动触达解决 [{theme_hint}] 困扰",
            ("HOW", "A"): f"优化触达渠道，增加 [{theme_hint}] 相关的种草内容",
        }
        return strategies.get((dim, node), f"关注并改善 [{theme_hint}] 方面的体验")

    def _grow_strategy(self, opportunity: dict[str, Any]) -> str:
        """生成增长策略"""
        dim = opportunity["persona_dim"]
        node = opportunity["aipl_node"]

        strategies = {
            ("WHAT", "I"): "扩大该功能点的内容营销覆盖，增加KOL测评",
            ("WHO", "A"): "针对该人群特征扩大广告投放，精准触达",
            ("WHY", "I"): "在营销素材中强化该决策动机的痛点共鸣",
            ("WHEN", "P1"): "在该场景下增加使用案例和场景化推荐",
        }
        return strategies.get((dim, node), "扩大推广覆盖，提升渗透率")

    def _convert_strategy(self, signal: dict[str, Any]) -> str:
        """生成转化策略"""
        stage = signal["stage"]

        strategies = {
            "A→I": "增加互动式内容（测评视频/对比工具），降低认知到兴趣的门槛",
            "I→P1": "推出限时优惠+免邮活动，降低首购决策成本",
            "P1→L1": "建立首购后7天关怀流程，主动解决使用问题",
            "P1→L2": "启动推荐返利计划，激励老用户推荐",
        }
        return strategies.get(stage, "优化该阶段的用户体验")

    def _suggest_owner(self, anomaly: dict[str, Any]) -> str:
        """建议主责部门"""
        dim = anomaly["persona_dim"]
        node = anomaly["aipl_node"]

        owners = {
            ("WHAT", "P1"): "产品研发部",
            ("WHAT", "P2"): "产品运营部",
            ("WHO", "P1"): "用户运营部",
            ("WHY", "I"): "品牌营销部",
            ("EMOTION", "L1"): "客户服务部",
            ("HOW", "A"): "数字营销部",
        }
        return owners.get((dim, node), "用户研究部")

    def _build_action_queue(self, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """构建行动队列（按紧急度排序的具体行动）"""
        actions = []
        for s in strategies:
            if s["urgency"] == "high":
                actions.append({
                    "action_id": s["id"],
                    "action": f"【紧急】{s['strategy']}",
                    "owner": s["owner"],
                    "deadline": "3个工作日内",
                    "status": "pending",
                })
            else:
                actions.append({
                    "action_id": s["id"],
                    "action": s["strategy"],
                    "owner": s["owner"],
                    "deadline": "2周内",
                    "status": "pending",
                })

        # 按紧急度排序
        actions.sort(key=lambda x: 0 if "紧急" in x["action"] else 1)
        return actions

    def _priority_ranking(self, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """策略优先级排序"""
        scored = []
        for s in strategies:
            score = 0
            if s["urgency"] == "high":
                score += 30
            elif s["urgency"] == "medium":
                score += 15
            if s["type"] == "fix":
                score += 20  # 修复优先于增长
            elif s["type"] == "convert":
                score += 25  # 转化最优先

            scored.append({
                **s,
                "priority_score": score,
            })

        scored.sort(key=lambda x: x["priority_score"], reverse=True)
        return [{"rank": i + 1, "strategy_id": s["id"], "target": s["target"],
                 "type": s["type"], "score": s["priority_score"], "owner": s["owner"]}
                for i, s in enumerate(scored)]

    def _estimate_impact(self, strategies: list[dict[str, Any]], matrix: PersonaAIPLMatrix) -> dict[str, Any]:
        """预估策略影响"""
        fix_count = sum(1 for s in strategies if s["type"] == "fix")
        grow_count = sum(1 for s in strategies if s["type"] == "grow")
        convert_count = sum(1 for s in strategies if s["type"] == "convert")

        # 预估NPS提升
        current_nps = 0
        all_promoters = all_detractors = 0
        for dim in PERSONA_DIMENSIONS:
            for node in AIPL_NODES:
                cell = matrix.get_cell(dim, node)
                all_promoters += cell.proxy_nps_counts["promoter"]
                all_detractors += cell.proxy_nps_counts["detractor"]
        total = all_promoters + all_detractors
        if total > 0:
            current_nps = (all_promoters / total * 100) - (all_detractors / total * 100)

        # 假设每个fix策略可提升NPS 5-10分
        estimated_nps_gain = fix_count * 5 + convert_count * 8

        return {
            "strategy_count": len(strategies),
            "by_type": {"fix": fix_count, "grow": grow_count, "convert": convert_count},
            "current_proxy_nps": round(current_nps, 1),
            "estimated_nps_gain": estimated_nps_gain,
            "estimated_nps_after": round(current_nps + estimated_nps_gain, 1),
            "estimated_roi": f"{estimated_nps_gain * 2}万-" if estimated_nps_gain > 0 else "需更多数据",
        }


# ---------------------------------------------------------------------------
# 4. 统一引擎入口
# ---------------------------------------------------------------------------

class ValueInsightEngine:
    """价值洞察引擎统一入口"""

    def __init__(self, thresholds: Optional[dict[str, float]] = None):
        self.monitoring_layer = MonitoringLayer(thresholds)
        self.analysis_layer = AnalysisLayer()
        self.decision_layer = DecisionLayer()

    def run(
        self,
        matrix: PersonaAIPLMatrix,
        extractions: list[VOCLabelExtraction],
        drill_down: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """执行完整的三层洞察流程"""
        # 监控层
        monitoring = self.monitoring_layer.run(matrix)

        # 分析层
        analysis = self.analysis_layer.run(matrix, extractions, drill_down)

        # 决策层
        decisions = self.decision_layer.run(matrix, extractions, monitoring, analysis)

        return {
            "monitoring": {
                "health_score": monitoring.health_score,
                "kpi_snapshot": monitoring.kpi_snapshot,
                "anomaly_count": len(monitoring.anomalies),
                "alert_count": len(monitoring.alerts),
                "alerts": monitoring.alerts[:5],
            },
            "analysis": {
                "drill_down": analysis.drill_down,
                "root_cause": analysis.root_cause,
                "comparison_count": len(analysis.comparisons),
                "comparisons": analysis.comparisons[:3],
                "opportunity_count": len(analysis.opportunities),
                "opportunities": analysis.opportunities[:3],
                "trend_signals": analysis.trend_signals[:3],
            },
            "decision": {
                "strategy_count": len(decisions.strategies),
                "strategies": decisions.strategies[:5],
                "action_queue": decisions.action_queue[:5],
                "priority_ranking": decisions.priority_ranking[:5],
                "expected_impact": decisions.expected_impact,
            },
        }

    def monitoring(self, matrix: PersonaAIPLMatrix) -> MonitoringReport:
        """仅运行监控层"""
        return self.monitoring_layer.run(matrix)

    def analysis(
        self,
        matrix: PersonaAIPLMatrix,
        extractions: list[VOCLabelExtraction],
        drill_down: Optional[dict[str, str]] = None,
    ) -> AnalysisReport:
        """仅运行分析层"""
        return self.analysis_layer.run(matrix, extractions, drill_down)

    def decision(
        self,
        matrix: PersonaAIPLMatrix,
        extractions: list[VOCLabelExtraction],
    ) -> DecisionReport:
        """仅运行决策层"""
        monitoring = self.monitoring_layer.run(matrix)
        analysis = self.analysis_layer.run(matrix, extractions)
        return self.decision_layer.run(matrix, extractions, monitoring, analysis)


# ---------------------------------------------------------------------------
# 5. 演示
# ---------------------------------------------------------------------------

def demo():
    """演示：完整三层洞察流程"""
    from unified_label_extraction import VOCLabelExtraction, AIPLTagMatch
    from persona_aipl_matrix import PersonaAIPLMatrixBuilder

    print("=" * 70)
    print("三层价值洞察引擎 - 演示")
    print("=" * 70)

    # 构造模拟数据（与 persona_aipl_matrix.py 的 demo 相同）
    extractions = []
    demo_data = [
        ("A", {"WHO": ["working_parent"]}, -0.1, "passive"),
        ("A", {"WHO": ["working_parent"]}, 0.2, "passive"),
        ("I", {"WHO": ["working_parent"], "WHAT": ["quiet_seeker"]}, 0.3, "passive"),
        ("I", {"WHO": ["working_parent"], "WHAT": ["quiet_seeker", "portable_seeker"]}, 0.5, "promoter"),
        ("P1", {"WHO": ["working_parent"], "WHAT": ["quiet_seeker"]}, -0.6, "detractor"),
        ("P1", {"WHO": ["working_parent"]}, 0.4, "promoter"),
        ("L1", {"WHO": ["working_parent"], "WHAT": ["easy_clean_seeker"]}, 0.6, "promoter"),
        ("A", {"WHO": ["first_time_parent"], "WHY": ["anxiety_driven"]}, 0.1, "passive"),
        ("A", {"WHO": ["first_time_parent"]}, 0.0, "passive"),
        ("I", {"WHO": ["first_time_parent"], "HOW": ["research_driven"]}, 0.4, "passive"),
        ("P1", {"WHO": ["first_time_parent"], "EMOTION": ["anxiety_driven"]}, -0.3, "detractor"),
        ("P1", {"WHO": ["first_time_parent"]}, 0.5, "promoter"),
        ("L1", {"WHO": ["first_time_parent"]}, 0.7, "promoter"),
        ("I", {"WHAT": ["quiet_seeker"], "WHEN": ["workplace_user"]}, 0.2, "passive"),
        ("P1", {"WHAT": ["quiet_seeker"]}, -0.4, "detractor"),
        ("P1", {"WHAT": ["quiet_seeker"]}, -0.5, "detractor"),
        ("L2", {"WHAT": ["quiet_seeker", "hands_free_seeker"]}, 0.8, "promoter"),
        ("I", {"WHY": ["price_sensitive", "budget_conscious"]}, 0.3, "passive"),
        ("I", {"WHY": ["price_sensitive"]}, 0.1, "passive"),
        ("P1", {"WHY": ["price_sensitive"]}, -0.2, "detractor"),
    ]

    for i, (stage, dims, sentiment, nps) in enumerate(demo_data):
        tag = AIPLTagMatch(
            tag_id=f"TAG_{i}", tag_en="demo_tag", tag_cn="演示标签",
            theme="产品核心性能", aipl_node=stage,
            sentiment_preset="neutral", sentiment_calibrated=sentiment, confidence=0.8,
        )
        extractions.append(VOCLabelExtraction(
            review_id=f"REV_{i:03d}", source_type="review", platform="amazon",
            spu_code="SPU001", product_line="breast_pump", category="wearable_pump",
            rating=3.0 + sentiment, aipl_stage=stage, aipl_tags=[tag],
            persona_dimensions=dims, sentiment_polarity=sentiment,
            proxy_nps_contribution=nps,
        ))

    # 构建矩阵
    matrix = PersonaAIPLMatrixBuilder().build(extractions)

    # 运行三层洞察
    engine = ValueInsightEngine()
    report = engine.run(matrix, extractions, drill_down={"persona_dim": "WHAT", "aipl_node": "P1"})

    # 输出监控层
    print("\n--- 监控层 ---")
    print(f"  健康度分数: {report['monitoring']['health_score']}/100")
    print(f"  异常数: {report['monitoring']['anomaly_count']}")
    print(f"  告警数: {report['monitoring']['alert_count']}")
    print(f"  整体Proxy NPS: {report['monitoring']['kpi_snapshot']['overall_proxy_nps']}")

    for alert in report['monitoring']['alerts'][:3]:
        print(f"  [{alert['level'].upper()}] {alert['target']}: {alert['description']}")

    # 输出分析层
    print("\n--- 分析层 ---")
    root = report['analysis']['root_cause']
    if 'root_cause_summary' in root:
        print(f"  根因分析 ({root.get('target', '')}):")
        print(f"    → {root['root_cause_summary']}")
    for comp in report['analysis']['comparisons'][:2]:
        print(f"  对比发现: {comp['description']}")

    # 输出决策层
    print("\n--- 决策层 ---")
    print(f"  策略总数: {report['decision']['strategy_count']}")
    print(f"  预估NPS提升: {report['decision']['expected_impact']['estimated_nps_gain']}")
    print(f"  预估NPS从 {report['decision']['expected_impact']['current_proxy_nps']} → {report['decision']['expected_impact']['estimated_nps_after']}")

    print("\n  Top 3 优先策略:")
    for s in report['decision']['priority_ranking'][:3]:
        print(f"    #{s['rank']} [{s['type']}] {s['target']} (得分{s['score']}) → {s['owner']}")

    print("\n  行动队列:")
    for a in report['decision']['action_queue'][:3]:
        print(f"    → {a['action'][:50]}... ({a['owner']}, {a['deadline']})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
