"""SkillRegistry: 按业务领域 lazy-load Skill 卡片为 Agent 工具.

设计:
  - 内部 Tool 数据结构: name + description + invoke(args)->dict
  - 实际生产环境会替换为 LangChain Tool 或 MCP Tool
  - 当前阶段使用纯函数式 stub,验证 MAS 骨架可跑通
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SkillTool:
    name: str
    description: str
    domain: str
    invoke: Callable[..., Dict[str, Any]] = field(default=lambda **_: {"status": "stub"})


_DOMAIN_TOOLS: Dict[str, List[SkillTool]] = {}


def register_tool(tool: SkillTool) -> None:
    _DOMAIN_TOOLS.setdefault(tool.domain, []).append(tool)


def _bootstrap_default_tools() -> None:
    if _DOMAIN_TOOLS:
        return

    _bootstrap_supply_chain()
    _bootstrap_time_series()
    _bootstrap_causal_inference()
    _bootstrap_marketing()
    _bootstrap_advertising()
    _bootstrap_ab_testing()
    _bootstrap_growth_model()
    _bootstrap_recommendation()
    _bootstrap_user_analytics()
    _bootstrap_data_agent_llm()
    _bootstrap_knowledge_graph()
    _bootstrap_ml_fundamentals()
    _bootstrap_pricing()
    _bootstrap_selection()


def _bootstrap_pricing() -> None:
    items = [
        ("pricing_dynamic_elasticity", "动态定价+需求弹性,DRL最优价格决策"),
        ("pricing_competitive_monitoring", "竞品价格监测,PCI指数+交叉弹性响应"),
        ("pricing_markdown_optimization", "折扣清仓优化,动态折扣路径最大化回收价值"),
        ("pricing_bundle_strategy", "捆绑定价,混合捆绑+估值分布优化"),
        ("pricing_cross_border_harmonization", "跨境价格协调,PPP归一化+汇率缓冲带"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="pricing", invoke=_stub_invoke(name)))


def _bootstrap_selection() -> None:
    items = [
        ("selection_market_space", "市场空间评估,月销/BSR趋势/评论数/评分四维过滤"),
        ("selection_gross_margin", "毛利可行性测算,售价-FBA-头程-关税≥40%门槛"),
        ("selection_compliance_risk", "母婴合规预判,高风险关键词+品类认证要求"),
        ("selection_kgqa_attributes", "KGQA最畅销属性匹配,类目最优属性命中率"),
        ("selection_causal_lift", "因果需求提升估计,BSR趋势+季节性+竞争度综合"),
        ("selection_composite_score", "候选品综合评分排序,五维加权composite_score"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="selection", invoke=_stub_invoke(name)))


def _stub_invoke(skill_name: str) -> Callable[..., Dict[str, Any]]:
    def _fn(**kwargs: Any) -> Dict[str, Any]:
        return {
            "skill": skill_name,
            "status": "stub_ok",
            "received_args_keys": list(kwargs.keys()),
        }
    return _fn


def _bootstrap_supply_chain() -> None:
    items = [
        ("supply_demand_forecast", "需求预测(供应链版),输入历史销售/季节性返回各 SKU 未来 N 周需求"),
        ("supply_multi_echelon_inventory", "多级库存优化,跨仓位补货分配建模"),
        ("supply_two_echelon_drl", "两级库存 DRL 策略,自动给出补货量与时点"),
        ("supply_safety_stock_replenishment", "安全库存与再订货点计算"),
        ("supply_monodense_price_elasticity", "单 SKU 价格弹性估计"),
        ("supply_hiforead_reconciliation", "HiFoReAd 分层需求预测调和(SKU×仓×市场)"),
        ("supply_genqot_lead_time_risk", "Gen-QOT 提前期分布风险,海运波动+旺季动态安全库存"),
        ("supply_multi_channel_inventory_pooling", "多渠道库存池化,跨Amazon/独立站/TikTok调拨"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="supply_chain", invoke=_stub_invoke(name)))


def _bootstrap_time_series() -> None:
    items = [
        ("ts_demand_forecasting", "通用时序需求预测"),
        ("ts_tft", "Temporal Fusion Transformer 多变量预测"),
        ("ts_prophet", "Prophet 季节分解预测"),
        ("ts_causal_gcf", "GCF 反事实需求预测(处理促销/缺货干预)"),
        ("ts_doubly_robust", "双重稳健干预效应估计"),
        ("ts_anomaly_detection", "时间序列异常检测"),
        ("ts_intelligent_prediction_dr", "智能反事实预测"),
        ("ts_conformal_demand_uq", "Conformal PID 自适应预测区间,需求不确定性量化"),
        ("ts_conformal_ts_intervals", "时序 Conformal 预测区间 EnbPI"),
        ("ts_multivariate_cointegration", "多变量协整 VECM,联动补货预测"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="time_series", invoke=_stub_invoke(name)))


def _bootstrap_causal_inference() -> None:
    items = [
        ("causal_uplift_modeling", "Uplift 模型,估计干预对个体的因果效应"),
        ("causal_attribution_forest", "Causal Forest 智能归因"),
        ("causal_dml_cohort", "DML 群体异质 CATE 估计"),
        ("causal_did", "DiD 双重差分估计"),
        ("causal_iv", "IV 工具变量估计"),
        ("causal_mediation", "中介机制分析"),
        ("causal_discovery_pc", "PC 算法因果发现"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="causal_inference", invoke=_stub_invoke(name)))


def _bootstrap_marketing() -> None:
    items = [
        ("marketing_mmm", "Marketing Mix Modeling"),
        ("marketing_dara_optimizer", "DARA 双阶段 LLM+RL 预算分配 Agent"),
        ("marketing_promotion_effectiveness", "促销因果效应估计"),
        ("marketing_channel_saturation_curve", "渠道饱和曲线,Hill函数拟合+边际回报"),
        ("marketing_competitive_response", "竞争响应建模,CI指数+反制策略"),
        ("marketing_multi_objective_budget", "多目标预算分配,Pareto ROI/Brand/NewProduct"),
        ("marketing_geo_level_effectiveness", "地理级营销效果,Geo Lift DiD估计"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="marketing", invoke=_stub_invoke(name)))


def _bootstrap_advertising() -> None:
    items = [
        ("ad_attribution_modeling", "多触点广告归因"),
        ("ad_roas_budget_optimization", "ROAS 目标预算优化"),
        ("ad_negative_keyword_safe_guard", "负向关键词安全守卫,Bayesian小样本保护"),
        ("ad_creative_fatigue_detection", "创意疲劳检测,双任务生存+PathSignature预警"),
        ("ad_tiktok_shop_attribution", "TikTok兴趣电商内容归因,Shapley+CABB跨品"),
        ("ad_compliance_guardrail", "Amazon ToS合规护栏,三层过滤(医疗/安全/价格声明)"),
        ("ad_to_behavior_funnel", "广告→用户行为漏斗,马尔可夫链触点建模"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="advertising", invoke=_stub_invoke(name)))


def _bootstrap_ab_testing() -> None:
    items = [
        ("ab_experimental_design", "A/B 实验设计,样本量/分流配比"),
        ("ab_multi_armed_bandit", "MAB 多臂老虎机"),
        ("ab_thompson_sampling", "Thompson Sampling Bandit"),
        ("ab_test_result_interpretation", "A/B 实验结果解读"),
        ("ab_power_analysis", "样本量与功效分析"),
        ("ab_switchback_design", "Switchback 双边市场实验设计"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="ab_testing", invoke=_stub_invoke(name)))


def _bootstrap_growth_model() -> None:
    items = [
        ("growth_churn_prediction", "客户流失预测"),
        ("growth_ltv_ziln", "LTV 长尾分布预测"),
        ("growth_new_product_opportunity", "新品机会挖掘"),
        ("growth_user_lifecycle_stan", "用户生命周期 STAN"),
        ("growth_cold_start_product_rec", "冷启动新品推荐"),
        ("growth_dl_churn_prediction", "深度学习流失预测"),
        ("growth_customer_journey_prototype", "客户旅程原型聚类"),
        ("growth_dqn_purchase_prediction", "DQN 购买预测"),
        ("growth_uplift_churn", "Uplift Churn 干预挽留"),
        ("growth_rfm_segmentation", "RFM 用户分群"),
        ("growth_category_trend_forecasting", "品类趋势预测,多信号融合+MK趋势检验"),
        ("growth_competitor_product_intelligence", "竞品选品监测,新SKU成功概率+价格战预警"),
        ("growth_product_opportunity_scoring", "新品机会评分卡,六维加权综合评估"),
        ("growth_supplier_evaluation", "供应商评估,TOPSIS多准则决策"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="growth_model", invoke=_stub_invoke(name)))


def _bootstrap_recommendation() -> None:
    items = [
        ("rec_matrix_factorization", "矩阵分解协同过滤"),
        ("rec_deep_learning_hi", "深度学习异构推荐"),
        ("rec_neural_ndcg_l2r", "Neural NDCG Learning-to-Rank"),
        ("rec_session_based_sr_gnn", "Session-based GNN 推荐"),
        ("rec_cold_start_meta", "元学习冷启动推荐"),
        ("rec_counterfactual_dce", "反事实推荐 DCE 双校准估计"),
        ("rec_semantic_id_retrieval", "语义 ID 检索"),
        ("rec_diversity_reranking", "多样性重排序"),
        ("rec_explainable", "可解释推荐"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="recommendation", invoke=_stub_invoke(name)))


def _bootstrap_user_analytics() -> None:
    items = [
        ("user_funnel_analysis", "用户漏斗分析"),
        ("user_cohort_retention", "队列留存分析"),
        ("user_agrs_review_summary", "AGRS 属性引导评论摘要,结构化Review洞察"),
        ("user_maa_actionable_advice", "MAA 多Agent行动建议,Review→运营决策"),
        ("user_star_statement_ranking", "StaR 观点语句排序,Review原子句排序"),
        ("user_laca_crosslingual_absa", "LACA 跨语言ABSA,零样本多语种情感分析"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="user_analytics", invoke=_stub_invoke(name)))


def _bootstrap_data_agent_llm() -> None:
    items = [
        ("data_sql_agent", "Text-to-SQL Agent"),
        ("data_argos_anomaly", "Argos Agentic 异常检测"),
        ("data_deepanalyze", "DeepAnalyze 自治数据科学 Agent"),
        ("data_root_cause_analysis", "Root Cause Analysis Agent"),
        ("data_dashboard_visualization", "数据到 Dashboard 可视化"),
        ("data_customer_journey_tree", "客服决策树"),
        ("data_dialin_llm_intent", "Dial-In LLM 客服意图聚类"),
        ("data_rag_enhanced_analysis", "RAG增强数据分析,历史报告检索+验证"),
        ("data_nl2dashboard", "NL2Dashboard,自然语言→BI仪表盘自动生成"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="data_agent_llm", invoke=_stub_invoke(name)))


def _bootstrap_knowledge_graph() -> None:
    items = [
        ("kg_hgt", "异构图 Transformer"),
        ("kg_hgcn", "双曲图卷积"),
        ("kg_dense_retrieval_ecommerce", "电商稠密语义检索"),
        ("kg_graphrag", "GraphRAG 知识增强检索"),
        ("kg_auto_construction", "KG 自动构建 Agent"),
        ("kg_hierarchical_product", "层级产品 KG 构建(图片→KG)"),
        ("kg_augmented_recommendation_colakg", "CoLaKG LLM 增强 KG 推荐"),
        ("kg_relation_completion", "KG 关系补全"),
        ("kg_kgqa", "KGQA 问答"),
        ("kg_multilingual_ner", "多语种 NER"),
        ("kg_skills_management", "Skill 元数据 KG"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="knowledge_graph", invoke=_stub_invoke(name)))


def _bootstrap_ml_fundamentals() -> None:
    items = [
        ("ml_feature_engineering", "通用特征工程,数值/类别/时间特征构造"),
        ("ml_model_evaluation", "模型评估体系,ROC/AUC/PR/Calibration"),
        ("ml_cross_validation", "交叉验证策略,K-Fold/Stratified/TimeSeries/Group"),
        ("ml_imbalanced_data", "不平衡数据处理,SMOTE/ClassWeight/Threshold调优"),
        ("ml_ensemble_methods", "集成学习,Bagging/Boosting/Stacking/Blending"),
        ("ml_feature_selection", "特征选择,SHAP/Boruta/Permutation/RFE"),
        ("ml_hyperparameter_optimization", "超参调优,Optuna/Bayesian/Hyperband"),
    ]
    for name, desc in items:
        register_tool(SkillTool(name=name, description=desc, domain="ml_fundamentals", invoke=_stub_invoke(name)))


class SkillRegistry:
    def __init__(self) -> None:
        _bootstrap_default_tools()

    def get_tools_for_domains(self, domains: List[str]) -> List[SkillTool]:
        result: List[SkillTool] = []
        for d in domains:
            result.extend(_DOMAIN_TOOLS.get(d, []))
        return result

    def get_tool(self, name: str) -> Optional[SkillTool]:
        for tools in _DOMAIN_TOOLS.values():
            for t in tools:
                if t.name == name:
                    return t
        return None

    def all_domains(self) -> List[str]:
        return sorted(_DOMAIN_TOOLS.keys())

    def total_tools(self) -> int:
        return sum(len(v) for v in _DOMAIN_TOOLS.values())

    def get_mcp_client(self) -> "MultiServerMCPClient":
        from mas.skills.mcp_servers.client import MultiServerMCPClient
        return MultiServerMCPClient()
