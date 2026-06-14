# Auto-extracted from build_playbook.py — DO NOT EDIT MANUALLY
# Source of truth: this file. Edit here, not in build_playbook.py
# fmt: off
# ruff: noqa

AGENT_CATALOG = [
    {
        "id": "agent-product-radar", "icon": "RA", "name": "选品雷达",
        "category": "选品分析", "cat_key": "selection", "cat_class": "cat-supply",
        "desc": "输入品类关键词，输出 Amazon/速卖通市场机会评分、竞争密度、需求趋势和推荐切入角度。",
        "roi": "选品决策周期 14天→2天",
        "linked_skills": ["Skill-Market-Size-Estimation", "Skill-New-Product-Opportunity-Mining", "Skill-Category-Trend-Forecasting"],
        "inputs": [
            {"id": "keyword", "label": "品类关键词", "type": "text", "placeholder": "例：硅胶婴儿餐具（或点击下方快速选择）"},
            {"id": "market", "label": "目标市场", "type": "select", "options": ["US", "UK", "DE", "AU", "JP"]},
            {"id": "budget", "label": "预算区间", "type": "select", "options": ["<$5k", "$5-20k", ">$20k"]},
        ],
        "demo_output": """[OK] 机会评分: 78/100（强力推荐）

市场数据
月均搜索量: 124,000（YoY +23%）
BSR TOP10 均价: $18.9 | 您的成本带: $6-8
头部 CR（前3卖家）: 合计占比 41%——仍有切入空间

差异化切入角度
1. 食品级硅胶+麦秸秆混合材质（情感溢价 +$4）
2. 带 OEM 定制礼盒（B2B 批发线索）
3. 月龄分段套装（提升 AOV 至 $35+）

竞争分析
竞品平均评论数: 847 | 最低切入评论数: ~200
新品窗口: ⭐⭐⭐⭐ 良好

[+] GO — 搜索量健康，价格带有利润空间，3个差异化方向均有评论验证。
建议首批备货: 500-800 件""",
    },
    {
        "id": "agent-listing-doctor", "icon": "LD", "name": "Listing 医生",
        "category": "Listing 优化", "cat_key": "listing", "cat_class": "cat-ad",
        "desc": "粘贴现有 Listing，输出逐字诊断报告和重写版本，精准命中 Amazon A10 算法关键因子。",
        "roi": "Listing 优化平均带动 GMV +12%",
        "linked_skills": ["Skill-Listing-AI-Copywriting", "Skill-Listing-Quality-Scoring", "Skill-Negative-Keyword-Safe-Guard"],
        "inputs": [
            {"id": "title", "label": "当前 Title", "type": "textarea", "placeholder": "粘贴当前商品标题..."},
            {"id": "bullets", "label": "Bullet Points", "type": "textarea", "placeholder": "粘贴5条 Bullet（每行一条）..."},
            {"id": "keywords", "label": "目标核心词 Top3", "type": "text", "placeholder": "例：silicone baby plate, BPA free"},
        ],
        "demo_output": """[!] 当前 Listing 诊断（62/100）

Title 评分: C（62/100）
问题 ①: 缺少核心词 "BPA-free"（搜索量 89K/月）
问题 ②: 字符仅 89 个，远低于 200 字符上限——损失关键词密度
问题 ③: 无场景词（6M+ / toddler / starter kit）

[~] Bullet #3 诊断
原文: "好用耐用，宝宝喜欢"
问题: 缺乏量化证明，过于主观
建议: "经 10,000 次弯折测试，FDA 认证食品级硅胶，安全耐用"

[OK] 重写后 Title（197字符）
[2024 Upgrade] BPA-Free Silicone Baby Plate Set — Suction Bowl+Spoon+Fork for Toddlers 6M+, Dishwasher Safe, Anti-Slip Self-Feeding Starter Kit (Gray)

[OK] 重写后 Bullet #3
Tested 10,000+ bends without cracking — Made from 100% FDA-compliant food-grade silicone, completely free of BPA, PVC, and phthalates. Safe for baby's first foods.

预估 CTR 提升: +18-25%（基于同类 A/B 测试基准）""",
    },
    {
        "id": "agent-voc-decoder", "icon": "VC", "name": "用户之声解码器",
        "category": "VOC 分析", "cat_key": "voc", "cat_class": "cat-voc",
        "desc": "批量导入竞品评论，自动聚类痛点/爽点，输出产品迭代优先级矩阵和广告素材金句库。",
        "roi": "VOC 驱动迭代，退货率平均下降 2-4%",
        "linked_skills": ["Skill-Review-Pain-Point-Mining", "Skill-LACA-CrossLingual-ABSA", "Skill-AGRS-Aspect-Guided-Review-Summarization"],
        "inputs": [
            {"id": "reviews", "label": "评论文本（每行一条）", "type": "textarea", "placeholder": "粘贴 20-200 条评论..."},
            {"id": "asin", "label": "竞品 ASIN（可选）", "type": "text", "placeholder": "例：B08XYZ1234"},
            {"id": "lang", "label": "语言", "type": "select", "options": ["英语", "英语+德语", "英语+日语", "多语言"]},
        ],
        "demo_output": """分析了 147 条评论（1-3星: 52条 | 4-5星: 95条）

[!] TOP3 痛点（高频）
1. 吸盘失效（38次提及）
   代表原话: "suction doesn't hold after 2 months of use"
2. 颜色褪色（29次）
   代表原话: "faded after dishwasher, looks cheap now"
3. 尺寸偏小（21次）
   代表原话: "not big enough for 18mo+, she outgrew it fast"

[+] TOP3 爽点（高频）
1. 好清洗（61次）— "easiest to clean baby product I own"
2. 防摔耐用（44次）— "dropped 100 times still perfect"
3. 颜色好看（38次）— "great minimalist colors"

[TIP] 广告金句（真实用户语言，高转化潜力）
"The only plate that actually stays on the table"

[FIX] 产品迭代建议（优先级排序）
P0: 吸盘结构升级 → 直接影响复购率和退货率
P1: 推出 18M+ 加大版 → 延长产品生命周期
P2: 加强洗碗机染色防护工艺""",
    },
    {
        "id": "agent-ad-attribution", "icon": "AA", "name": "广告归因侦探",
        "category": "广告归因", "cat_key": "attribution", "cat_class": "cat-ad",
        "desc": "上传广告报告，自动识别无效花费、归因漏洞、预算分配错误，输出可立即执行的调优清单。",
        "roi": "识别 25-35% 广告浪费，$10k预算节省 $2,500-3,500/月",
        "linked_skills": ["Skill-PVM-Attribution-Window-Harmonization", "Skill-Identified-Bayesian-MMM", "Skill-DARA-Agentic-MMM-Optimizer"],
        "inputs": [
            {"id": "platform", "label": "广告平台", "type": "select", "options": ["Amazon SP", "Amazon SB/SD", "TikTok Ads", "Meta Ads", "Google Ads"]},
            {"id": "spend", "label": "月广告花费（$）", "type": "text", "placeholder": "例：12400"},
            {"id": "target_acos", "label": "目标 ACoS/ROAS", "type": "text", "placeholder": "例：ACoS 18% 或 ROAS 5x"},
            {"id": "data", "label": "广告数据（可选，粘贴 CSV）", "type": "textarea", "placeholder": "粘贴关键词报告数据..."},
        ],
        "demo_output": """广告浪费诊断（过去30天）
总花费: $12,400 | 有效转化花费: $8,100
估算浪费: $4,300（34.7%）[WARN]

[!] 无效关键词 TOP3（建议立即否定）
1. "baby plate cheap" — 花费 $380，转化 0，点击 214
2. "kids dinnerware set" — ACoS 187%，花费 $520
3. "silicone bowl wholesale" — B2B 意图，转化率 0.3%

[~] 归因漏洞
SB广告 impression 12万 → 无再营销链路，损失中端漏斗流量

当前 ACoS: 26.1%（目标 18%，超标 8.1pp）

[OK] 行动清单（本周执行，预期节省 $900+/月）
1. 否定以上3个关键词 → 立即节省 ~$900/月
2. 开启 SP 动态竞价-仅降低（流量质量提升，ACoS 预计 -3pp）
3. 新增否定词组: "wholesale" "bulk" "cheap" "set of 10"
4. SB 广告新增 Retargeting 受众（覆盖已浏览未购买用户）""",
    },
    {
        "id": "agent-competitor-radar", "icon": "CR", "name": "竞品雷达站",
        "category": "竞品监控", "cat_key": "competitor", "cat_class": "cat-ops",
        "desc": "输入竞品 ASIN 列表，追踪价格/排名/评论/Listing 变化，异常时生成智能预警和响应建议。",
        "roi": "广告截流策略平均提升转化率 8-12%",
        "linked_skills": ["Skill-Competitive-Price-Monitoring", "Skill-Review-Fraud-Detection", "Skill-Competitive-Response-Modeling"],
        "inputs": [
            {"id": "asins", "label": "竞品 ASIN 列表（每行一个）", "type": "textarea", "placeholder": "B08XYZ1234\nB09ABC5678\nB07DEF9012"},
            {"id": "period", "label": "监控周期", "type": "select", "options": ["过去7天", "过去14天", "过去30天"]},
            {"id": "metrics", "label": "监控维度", "type": "select", "options": ["全部", "价格+BSR", "评论动态", "Listing变更"]},
        ],
        "demo_output": """[ALERT] 竞品异动报告（过去7天）

B08XYZ1234（竞品A — 头部卖家）
├─ 价格: $21.99 → $17.99（-18%）[WARN] 降价促销迹象
├─ BSR: #342 → #89（大幅上升）
└─ 新增评论: +47条（含3条1星，投诉发货问题）

B09ABC5678（竞品B）
├─ Title 已修改（新增 "BPA-Free" 关键词）
└─ A+ 页面上线（新增产品对比表格）

7天数据汇总
竞品均价变化: -8.3% | 你的价格竞争力: 中等

[TIP] 建议响应（按优先级）
P0: 竞品A大促进行中 → 考虑同步降价 $1-2 或强化差异化广告
P1: 竞品A发货差评爆发 → 可针对竞品词做广告截流（时间窗口约2周）
P2: 竞品B更新Listing → 检查是否使用你的核心卖点词汇""",
    },
    {
        "id": "agent-supply-sentinel", "icon": "SC", "name": "供应链哨兵",
        "category": "供应链预警", "cat_key": "supply", "cat_class": "cat-supply",
        "desc": "接入库存/销速数据，预测断货风险和过库存风险，给出补货建议时间表和海运/空运决策。",
        "roi": "避免一次断货可保护 $4,000-15,000 BSR 回弹成本",
        "linked_skills": ["Skill-Safety-Stock-Replenishment", "Skill-Lead-Time-Distribution-Risk-GenQOT", "Skill-Promotion-Logistics-Surge-Forecast"],
        "inputs": [
            {"id": "stock", "label": "当前库存量（件）", "type": "text", "placeholder": "例：340"},
            {"id": "velocity", "label": "日均销速（件/天）", "type": "text", "placeholder": "例：28"},
            {"id": "lead_time", "label": "供货周期（天）", "type": "text", "placeholder": "例：21（海运）或7（空运）"},
            {"id": "channel", "label": "渠道类型", "type": "select", "options": ["Amazon FBA", "自发货", "FBA+海外仓混合"]},
        ],
        "demo_output": """[!] 断货风险评级: 高危

当前库存: 340件
日均销速: 28件/天（近7天均值）
剩余可售天数: 12.1天

[WARN] FBA 入库周期: 14-18天
结论: 已进入断货窗口，需立即行动！

补货建议
├─ 建议补货量: 1,200件（含安全库存30天）
├─ 最迟下单日期: 今日（已超过海运安全窗口）
├─ 推荐方案: 空运600件（应急，+$0.8/件）+ 海运600件（补充）
└─ 预估断货损失（若不补货）: ~$4,200（按当前BSR#234计算）

[~] Q4旺季预警（60天后）
历史数据显示11月销速 ×2.8 → 建议提前备货至2,500件
最迟开始备货时间: 9月15日（海运）

成本对比
海运方案: 成本+$0, 风险高
空运方案: 成本+$480, 断货风险消除""",
    },
    {
        "id": "agent-cs-triage", "icon": "CS", "name": "客服分诊台",
        "category": "客服售后", "cat_key": "cs", "cat_class": "cat-voc",
        "desc": "批量导入工单，自动分类优先级、识别高风险工单（A-to-Z/差评威胁），生成文化适配回复模板。",
        "roi": "处理效率提升 3x，A-to-Z 索赔率降低 40%",
        "linked_skills": ["Skill-DialIn-LLM-Case-Intent-Clustering", "Skill-Customer-Journey-Decision-Tree", "Skill-Emotional-AI-Customer-Care"],
        "inputs": [
            {"id": "tickets", "label": "工单文本（每行一条）", "type": "textarea", "placeholder": "粘贴 10-100 条客服工单..."},
            {"id": "platform", "label": "平台来源", "type": "select", "options": ["Amazon", "Shopify", "eBay", "混合"]},
            {"id": "sla", "label": "SLA 要求", "type": "select", "options": ["24小时", "48小时", "72小时"]},
        ],
        "demo_output": """分诊报告（63条工单）

分类分布
退货退款请求: 18条（28.6%）
产品质量问题: 14条（22.2%）
物流查询: 19条（30.2%）
使用咨询: 12条（19.0%）

[ALERT] 高优先级（需24h内处理）
工单#2847: "file A-to-Z claim if no response by tomorrow"
工单#2851: "going to leave 1-star review, terrible quality"
工单#2863: 情绪值: ANGRY（检测到高风险用户）

[OK] 标准回复模板 #1（物流查询）
"Hi [Name], Thank you for reaching out! Your order [ORDER_ID] is currently in transit. Expected delivery: [DATE]. Tracking: [LINK]
If you haven't received it by [DATE+3], please reply and we'll send a replacement immediately."

[!] 产品缺陷信号
14条工单提及 "strap breaks" → 可能存在结构性质量问题
建议: 立即联系工厂复查该批次（批号: 请提供）""",
    },
    {
        "id": "agent-pricing-advisor", "icon": "PA", "name": "动态定价顾问",
        "category": "价格策略", "cat_key": "pricing", "cat_class": "cat-ops",
        "desc": "分析竞品价格带、成本结构和当前排名，给出最优定价策略和分季节的促销节奏建议。",
        "roi": "合理溢价策略平均提升净利润率 4-8个百分点",
        "linked_skills": ["Skill-AIGP-LLM-Dynamic-Pricing", "Skill-Dynamic-Pricing-Elasticity", "Skill-Markdown-Optimization"],
        "inputs": [
            {"id": "price", "label": "当前售价（$）", "type": "text", "placeholder": "例：19.99"},
            {"id": "cost", "label": "综合成本（货值+头程+FBA，$）", "type": "text", "placeholder": "例：7.80"},
            {"id": "comp_range", "label": "竞品价格区间", "type": "text", "placeholder": "例：$15-$22"},
            {"id": "bsr", "label": "当前 BSR", "type": "text", "placeholder": "例：234"},
        ],
        "demo_output": """定价策略分析

当前状态
售价: $19.99 | 成本: $7.80 | 毛利率: 61% | BSR: #234

最优价格区间: $21.99 - $23.99
理由: 竞品头部集中在 $22-25，您的 Review 数量（847条）
和评分（4.6）支持高于均值的溢价定位。

涨价路径（建议分步执行）
Week 1: $19.99 → $20.99（观察转化率变化）
Week 2: 若转化率降幅 <15%，升至 $21.99
预估毛利率: 61% → 69%（+$1.60/件，月增益约 $1,400）

促销节奏建议
├─ 每月1次 Coupon 15%（维持搜索权重）
├─ Prime Day 前2周: $18.99（冲BSR，接受短期利润压缩）
└─ Q4 旺季（11-12月）: 维持 $22.99（需求刚性，不降价）

[WARN] 涨价风险提示
若7天内转化率下降 >25%，回退至 $20.99 并检查竞品动态""",
    },
    {
        "id": "agent-account-guardian", "icon": "AG", "name": "账号风险卫士",
        "category": "合规风控", "cat_key": "risk", "cat_class": "cat-risk",
        "desc": "扫描账号操作记录和 Listing 合规性，提前识别封号/下架风险，生成整改清单和申诉模板。",
        "roi": "预防式合规管理，避免封号损失（平均 $20,000-50,000）",
        "linked_skills": ["Skill-Amazon-ToS-Compliance-Guardrail", "Skill-Consumer-Complaint-Recall-Prediction", "Skill-Compliance-Scored-Guardrail-Orchestration"],
        "inputs": [
            {"id": "notice", "label": "近期异常通知（粘贴邮件内容）", "type": "textarea", "placeholder": "粘贴 Amazon 警告邮件或 Account Health 异常通知..."},
            {"id": "asins", "label": "需检查的 ASIN 列表", "type": "textarea", "placeholder": "每行一个 ASIN"},
            {"id": "health", "label": "当前账号健康状态", "type": "select", "options": ["绿色（正常）", "黄色（预警）", "红色（高危）"]},
        ],
        "demo_output": """账号风险评分: 6.8/10（中等风险）

[!] 高风险项（立即处理）
1. 检测到 Review Manipulation 风险信号
   3条评论来自同一IP段 → 可能触发 Amazon 检测
   建议: 停止所有站外导流至评论页的操作

2. Listing B08XYZ: Title 包含竞品品牌词 "SimilarBrand"
   → 商标侵权风险，建议24h内删除

[~] 中等风险项
3. ODR（订单缺陷率）本月: 1.08%（红线1%）
   → 轻微超标，需本周内处理所有未回复差评工单

[OK] 整改清单（按优先级）
P0（今日）: 删除 Title 中的侵权词
P0（今日）: 联系3名疑似刷单买家要求删除评论
P1（本周）: 回复全部差评工单，目标 ODR<0.9%
P2（下月）: 申请 Brand Registry 加强品牌保护

申诉模板已就绪
如收到 Account Health 警告，可使用以下 POA 模板框架:
"Root Cause: ... Corrective Actions: ... Preventive Measures: ..." """,
    },
    {
        "id": "agent-pnl-analyzer", "icon": "PL", "name": "P&L 透视镜",
        "category": "数据分析", "cat_key": "analytics", "cat_class": "cat-ops",
        "desc": "输入销售数据，自动计算真实净利润率（含所有隐性成本），识别利润漏洞并给出量化改善路径。",
        "roi": "平均识别 35-50% 的利润改善空间",
        "linked_skills": ["Skill-DeepAnalyze-Autonomous-Data-Science-Agent", "Skill-ProRCA-Business-Analysis", "Skill-NL2Dashboard-Automation"],
        "inputs": [
            {"id": "revenue", "label": "月销售额（$）", "type": "text", "placeholder": "例：32400"},
            {"id": "cogs", "label": "商品成本（$）", "type": "text", "placeholder": "例：9200"},
            {"id": "fba", "label": "FBA 费用（$）", "type": "text", "placeholder": "例：5800"},
            {"id": "ads", "label": "广告花费（$）", "type": "text", "placeholder": "例：6500"},
            {"id": "return_rate", "label": "退货率（%）", "type": "text", "placeholder": "例：4"},
        ],
        "demo_output": """P&L 透视报告（月度）

收入: $32,400
├─ 商品成本: -$9,200（28.4%）
├─ FBA 费用: -$5,800（17.9%）
├─ 广告花费: -$6,500（20.1%）[!] 偏高
├─ 头程物流: -$1,900（5.9%）
├─ 退货成本: -$1,296（4.0%）[~] 需关注
├─ 平台佣金: -$4,860（15.0%）
└─ 净利润: $2,844（净利率 8.8%）[!] 低于行业均值 15%

[!] 利润漏洞识别（TOP3）
1. ACoS 26.1% → 行业均值 18% → 优化空间: +$2,700/月
2. 退货率 4% → 行业优秀 3% → 每降1% = +$324/月
3. 头程走空运 → 改海运可节省 $600/月

改善后利润模拟（执行以上3项）
预计净利润: $6,144（净利率 19.0%）
利润提升: +116%

最优先行动: 优化广告 ACoS（ROI最高，可在30天内见效）""",
    },
    {
        "id": "agent-brand-guardian", "icon": "BG", "name": "品牌合规卫士",
        "category": "合规风控", "cat_key": "risk", "cat_class": "cat-risk",
        "desc": "扫描品牌文案，进行 FDA/FTC/Amazon TOS 三轨合规检查，输出违规词清单和逐句合规改写建议。",
        "roi": "预防 FTC 警告和产品下架，单次违规处罚可达 $50,000",
        "linked_skills": ["Skill-Compliance-Scored-Guardrail-Orchestration", "Skill-Amazon-ToS-Compliance-Guardrail", "Skill-Cross-Border-Compliance-Framework"],
        "inputs": [
            {"id": "copy", "label": "品牌文案（Listing/广告语/包装文字）", "type": "textarea", "placeholder": "粘贴需要检查的文案内容..."},
            {"id": "category", "label": "产品品类", "type": "select", "options": ["母婴", "健康保健", "食品饮料", "消费电子", "美妆个护"]},
            {"id": "market", "label": "目标市场", "type": "select", "options": ["US", "UK/EU", "AU", "全球"]},
        ],
        "demo_output": """合规扫描报告（母婴品类 US 市场）
综合合规评分: 64/100 → 整改后预计: 94/100

[!] 禁用词（立即删除，3处违规）
1. "clinically proven" → 需 FDA 认证才可使用
   合规改写: "designed with safety in mind"

2. "prevents colic" → 医疗声明，违反 FTC
   合规改写: "designed for comfortable feeding"

3. "100% safe" → 绝对化表述，FTC 违规
   合规改写: "made with food-grade materials tested to US safety standards"

[~] 慎用词（需补充证明文件，2处）
4. "BPA-free" → 需有第三方检测报告支撑
5. "FDA approved" → 应改为 "FDA registered facility"

[OK] 合规文案建议
将 "clinically proven to reduce fussiness" 改写为:
"Thoughtfully designed for baby's comfort — made with soft, food-grade silicone that moms trust"

所需证明文件清单
□ SGS/Intertek 第三方检测报告
□ CPSIA 认证（儿童产品必需）
□ BPA-Free 声明（实验室报告）""",
    },
    {
        "id": "agent-tiktok-content", "icon": "TC", "name": "TikTok 内容官",
        "category": "内容营销", "cat_key": "content", "cat_class": "cat-ad",
        "desc": "输入产品和受众画像，输出 TikTok/Reels 爆款选题矩阵、脚本框架和话题标签策略，降低内容生产成本。",
        "roi": "系统化内容输出降低 CPM 40%，自然流量占比提升至 30%+",
        "linked_skills": ["Skill-DAWN-Talking-Head-Review", "Skill-AnchorCrafter-Virtual-Anchor-Demo", "Skill-Creative-Fatigue-Detection"],
        "inputs": [
            {"id": "product", "label": "产品名称/描述", "type": "text", "placeholder": "例：硅胶婴儿餐具套装"},
            {"id": "audience", "label": "目标受众画像", "type": "text", "placeholder": "例：0-2岁宝妈，关注辅食/育儿"},
            {"id": "style", "label": "内容风格偏好", "type": "select", "options": ["教程/攻略", "痛点反转", "生活记录", "对比测评", "UGC种草"]},
            {"id": "freq", "label": "周更新频次", "type": "select", "options": ["3条/周", "5条/周", "每日更新"]},
        ],
        "demo_output": """本周 TikTok 选题矩阵（硅胶婴儿餐具）

Day 1（周一）— 痛点反转
Hook: "妈妈们最崩溃的吃饭时刻是这个→"
核心内容: 展示宝宝把普通盘子扫落地的10秒混剪
转折: 切换到吸盘盘子完全无法被扫落
CTA: "这个改变让我重获自由"
预测完播率: 68%+（情感共鸣强）

Day 3（周三）— 教程攻略
Hook: "6个月宝宝开始辅食? 3个工具够了"
话题标签: #babyfood #momhack #toddlermom #BLW #辅食
预算: $0（自拍）

Day 5（周五）— UGC 素人合作
策略: 找 3 个粉丝量 1-5k 的素人妈妈
换货方式: 寄送产品换 1条真实使用视频
预算: 产品成本约 $25×3=$75

最佳发布时间: 周一/三/五 晚7-9PM（目标市场时区）

爆款公式（适合你的品类）
情绪触发（共鸣）+ 意外反转 + 简单CTA = 完播率 65%+""",
    },
]
