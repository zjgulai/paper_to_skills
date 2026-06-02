# paper2skills-code

将前沿学术论文转化为可落地的商业决策技能代码模板。

## 目录结构

```
├── causal_inference/      # 因果推断
│   ├── uplift_model/      # Uplift Model
│   ├── causal_forest/     # Causal Forest
│   ├── dml_cohort_cate/   # DML Cohort CATE
│   ├── did/               # DiD 双重差分
│   ├── iv/                # IV 工具变量
│   ├── mediation/         # 中介分析
│   ├── causal_discovery_pc/# PC 因果发现
│   └── doubly_robust/     # 双鲁棒估计
├── ab_testing/            # A/B实验
├── time_series/           # 时间序列预测
├── supply_chain/          # 供应链优化
├── recommendation/        # 推荐系统
├── growth_model/          # 增长模型
├── nlp_voc/               # NLP 舆情分析 (已迁至 ../ai_nlp_voc/)
├── knowledge_graph/       # 知识图谱 (HGT / HGCN)
├── data_agent_llm/        # DataAgent & LLM 分析
├── mas/                   # 多智能体系统
├── llm_agent_engineering/ # 智能体工程
├── ml_fundamentals/       # ML基础 (评估/CV/不平衡/集成/特征选择/HPO)
├── advertising/           # 广告分析 (归因/ROAS/关键词/素材/合规)
├── user_analytics/        # 用户分析 (漏斗/留存/ABSA)
├── marketing/             # 营销投放分析 (MMM/饱和/竞争/Geo)
├── pricing/               # 价格优化 (动态/竞品/清仓/捆绑/跨境)
├── logistics/             # 物流履约 (跨境路径/配送/退货)
├── risk_fraud/            # 风控反欺诈 (虚假评论/异常交易/刷量)
└── visual_content/        # AI视频生成 (虚拟主播/商品展示/品牌/UGC)
    ├── anchor_demo/       # AnchorCrafter 虚拟主播带货
    ├── product_showcase/  # Phantom 商品展示 I2V
    ├── brand_video/       # Aquarius 品牌视频
    └── ugc_talking_head/  # DAWN 口播 Review
```

## 安装
```bash
pip install -r requirements.txt
```

## 使用
每个模块包含：
- `model.py` - 核心算法实现
- `example.py` - 使用示例
- `__init__.py` - 模块导出
