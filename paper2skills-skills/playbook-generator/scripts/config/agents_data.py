# Auto-extracted from build_playbook.py — DO NOT EDIT MANUALLY
# Source of truth: this file. Edit here, not in build_playbook.py
# fmt: off
# ruff: noqa

AGENT_CATALOG = [
    {
        "id": "agent-product-radar",
        "icon": "RA",
        "name": "选品雷达",
        "category": "选品分析",
        "cat_key": "selection",
        "cat_class": "cat-supply",
        "desc": "输入品类关键词，输出 Amazon/速卖通市场机会评分、竞争密度、需求趋势和推荐切入角度。",
        "roi": "选品决策周期 14天→2天",
        "linked_skills": ['Skill-Market-Size-Estimation', 'Skill-New-Product-Opportunity-Mining', 'Skill-Category-Trend-Forecasting'],
        "inputs": [{'id': 'keyword', 'label': '品类关键词', 'type': 'text', 'placeholder': '例：硅胶婴儿餐具（或点击下方快速选择）'}, {'id': 'market', 'label': '目标市场', 'type': 'select', 'options': ['US', 'UK', 'DE', 'AU', 'JP']}, {'id': 'budget', 'label': '预算区间', 'type': 'select', 'options': ['<$5k', '$5-20k', '>$20k']}],
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
        "id": "agent-listing-doctor",
        "icon": "LD",
        "name": "Listing 医生",
        "category": "Listing 优化",
        "cat_key": "listing",
        "cat_class": "cat-ad",
        "desc": "粘贴现有 Listing，输出逐字诊断报告和重写版本，精准命中 Amazon A10 算法关键因子。",
        "roi": "Listing 优化平均带动 GMV +12%",
        "linked_skills": ['Skill-Listing-AI-Copywriting', 'Skill-Listing-Quality-Scoring', 'Skill-Negative-Keyword-Safe-Guard'],
        "inputs": [{'id': 'title', 'label': '当前 Title', 'type': 'textarea', 'placeholder': '粘贴当前商品标题...'}, {'id': 'bullets', 'label': 'Bullet Points', 'type': 'textarea', 'placeholder': '粘贴5条 Bullet（每行一条）...'}, {'id': 'keywords', 'label': '目标核心词 Top3', 'type': 'text', 'placeholder': '例：silicone baby plate, BPA free'}],
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
        "id": "agent-voc-decoder",
        "icon": "VC",
        "name": "用户之声解码器",
        "category": "VOC 分析",
        "cat_key": "voc",
        "cat_class": "cat-voc",
        "desc": "批量导入竞品评论，自动聚类痛点/爽点，输出产品迭代优先级矩阵和广告素材金句库。",
        "roi": "VOC 驱动迭代，退货率平均下降 2-4%",
        "linked_skills": ['Skill-Review-Pain-Point-Mining', 'Skill-LACA-CrossLingual-ABSA', 'Skill-AGRS-Aspect-Guided-Review-Summarization'],
        "inputs": [{'id': 'reviews', 'label': '评论文本（每行一条）', 'type': 'textarea', 'placeholder': '粘贴 20-200 条评论...'}, {'id': 'asin', 'label': '竞品 ASIN（可选）', 'type': 'text', 'placeholder': '例：B08XYZ1234'}, {'id': 'lang', 'label': '语言', 'type': 'select', 'options': ['英语', '英语+德语', '英语+日语', '多语言']}],
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
        "id": "agent-ad-attribution",
        "icon": "AA",
        "name": "广告归因侦探",
        "category": "广告归因",
        "cat_key": "attribution",
        "cat_class": "cat-ad",
        "desc": "上传广告报告，自动识别无效花费、归因漏洞、预算分配错误，输出可立即执行的调优清单。",
        "roi": "识别 25-35% 广告浪费，$10k预算节省 $2,500-3,500/月",
        "linked_skills": ['Skill-PVM-Attribution-Window-Harmonization', 'Skill-Identified-Bayesian-MMM', 'Skill-DARA-Agentic-MMM-Optimizer'],
        "inputs": [{'id': 'platform', 'label': '广告平台', 'type': 'select', 'options': ['Amazon SP', 'Amazon SB/SD', 'TikTok Ads', 'Meta Ads', 'Google Ads']}, {'id': 'spend', 'label': '月广告花费（$）', 'type': 'text', 'placeholder': '例：12400'}, {'id': 'target_acos', 'label': '目标 ACoS/ROAS', 'type': 'text', 'placeholder': '例：ACoS 18% 或 ROAS 5x'}, {'id': 'data', 'label': '广告数据（可选，粘贴 CSV）', 'type': 'textarea', 'placeholder': '粘贴关键词报告数据...'}],
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
        "id": "agent-competitor-radar",
        "icon": "CR",
        "name": "竞品雷达站",
        "category": "竞品监控",
        "cat_key": "competitor",
        "cat_class": "cat-ops",
        "desc": "输入竞品 ASIN 列表，追踪价格/排名/评论/Listing 变化，异常时生成智能预警和响应建议。",
        "roi": "广告截流策略平均提升转化率 8-12%",
        "linked_skills": ['Skill-Competitive-Price-Monitoring', 'Skill-Review-Fraud-Detection', 'Skill-Competitive-Response-Modeling'],
        "inputs": [{'id': 'asins', 'label': '竞品 ASIN 列表（每行一个）', 'type': 'textarea', 'placeholder': 'B08XYZ1234\nB09ABC5678\nB07DEF9012'}, {'id': 'period', 'label': '监控周期', 'type': 'select', 'options': ['过去7天', '过去14天', '过去30天']}, {'id': 'metrics', 'label': '监控维度', 'type': 'select', 'options': ['全部', '价格+BSR', '评论动态', 'Listing变更']}],
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
        "id": "agent-supply-sentinel",
        "icon": "SC",
        "name": "供应链哨兵",
        "category": "供应链预警",
        "cat_key": "supply",
        "cat_class": "cat-supply",
        "desc": "接入库存/销速数据，预测断货风险和过库存风险，给出补货建议时间表和海运/空运决策。",
        "roi": "避免一次断货可保护 $4,000-15,000 BSR 回弹成本",
        "linked_skills": ['Skill-Safety-Stock-Replenishment', 'Skill-Lead-Time-Distribution-Risk-GenQOT', 'Skill-Promotion-Logistics-Surge-Forecast'],
        "inputs": [{'id': 'stock', 'label': '当前库存量（件）', 'type': 'text', 'placeholder': '例：340'}, {'id': 'velocity', 'label': '日均销速（件/天）', 'type': 'text', 'placeholder': '例：28'}, {'id': 'lead_time', 'label': '供货周期（天）', 'type': 'text', 'placeholder': '例：21（海运）或7（空运）'}, {'id': 'channel', 'label': '渠道类型', 'type': 'select', 'options': ['Amazon FBA', '自发货', 'FBA+海外仓混合']}],
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
        "id": "agent-cs-triage",
        "icon": "CS",
        "name": "客服分诊台",
        "category": "客服售后",
        "cat_key": "cs",
        "cat_class": "cat-voc",
        "desc": "批量导入工单，自动分类优先级、识别高风险工单（A-to-Z/差评威胁），生成文化适配回复模板。",
        "roi": "处理效率提升 3x，A-to-Z 索赔率降低 40%",
        "linked_skills": ['Skill-DialIn-LLM-Case-Intent-Clustering', 'Skill-Customer-Journey-Decision-Tree', 'Skill-Emotional-AI-Customer-Care'],
        "inputs": [{'id': 'tickets', 'label': '工单文本（每行一条）', 'type': 'textarea', 'placeholder': '粘贴 10-100 条客服工单...'}, {'id': 'platform', 'label': '平台来源', 'type': 'select', 'options': ['Amazon', 'Shopify', 'eBay', '混合']}, {'id': 'sla', 'label': 'SLA 要求', 'type': 'select', 'options': ['24小时', '48小时', '72小时']}],
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
        "id": "agent-pricing-advisor",
        "icon": "PA",
        "name": "动态定价顾问",
        "category": "价格策略",
        "cat_key": "pricing",
        "cat_class": "cat-ops",
        "desc": "分析竞品价格带、成本结构和当前排名，给出最优定价策略和分季节的促销节奏建议。",
        "roi": "合理溢价策略平均提升净利润率 4-8个百分点",
        "linked_skills": ['Skill-AIGP-LLM-Dynamic-Pricing', 'Skill-Dynamic-Pricing-Elasticity', 'Skill-Markdown-Optimization'],
        "inputs": [{'id': 'price', 'label': '当前售价（$）', 'type': 'text', 'placeholder': '例：19.99'}, {'id': 'cost', 'label': '综合成本（货值+头程+FBA，$）', 'type': 'text', 'placeholder': '例：7.80'}, {'id': 'comp_range', 'label': '竞品价格区间', 'type': 'text', 'placeholder': '例：$15-$22'}, {'id': 'bsr', 'label': '当前 BSR', 'type': 'text', 'placeholder': '例：234'}],
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
        "id": "agent-account-guardian",
        "icon": "AG",
        "name": "账号风险卫士",
        "category": "合规风控",
        "cat_key": "risk",
        "cat_class": "cat-risk",
        "desc": "扫描账号操作记录和 Listing 合规性，提前识别封号/下架风险，生成整改清单和申诉模板。",
        "roi": "预防式合规管理，避免封号损失（平均 $20,000-50,000）",
        "linked_skills": ['Skill-Amazon-ToS-Compliance-Guardrail', 'Skill-Consumer-Complaint-Recall-Prediction', 'Skill-Compliance-Scored-Guardrail-Orchestration'],
        "inputs": [{'id': 'notice', 'label': '近期异常通知（粘贴邮件内容）', 'type': 'textarea', 'placeholder': '粘贴 Amazon 警告邮件或 Account Health 异常通知...'}, {'id': 'asins', 'label': '需检查的 ASIN 列表', 'type': 'textarea', 'placeholder': '每行一个 ASIN'}, {'id': 'health', 'label': '当前账号健康状态', 'type': 'select', 'options': ['绿色（正常）', '黄色（预警）', '红色（高危）']}],
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
        "id": "agent-pnl-analyzer",
        "icon": "PL",
        "name": "P&L 透视镜",
        "category": "数据分析",
        "cat_key": "analytics",
        "cat_class": "cat-ops",
        "desc": "输入销售数据，自动计算真实净利润率（含所有隐性成本），识别利润漏洞并给出量化改善路径。",
        "roi": "平均识别 35-50% 的利润改善空间",
        "linked_skills": ['Skill-DeepAnalyze-Autonomous-Data-Science-Agent', 'Skill-ProRCA-Business-Analysis', 'Skill-NL2Dashboard-Automation'],
        "inputs": [{'id': 'revenue', 'label': '月销售额（$）', 'type': 'text', 'placeholder': '例：32400'}, {'id': 'cogs', 'label': '商品成本（$）', 'type': 'text', 'placeholder': '例：9200'}, {'id': 'fba', 'label': 'FBA 费用（$）', 'type': 'text', 'placeholder': '例：5800'}, {'id': 'ads', 'label': '广告花费（$）', 'type': 'text', 'placeholder': '例：6500'}, {'id': 'return_rate', 'label': '退货率（%）', 'type': 'text', 'placeholder': '例：4'}],
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
        "id": "agent-brand-guardian",
        "icon": "BG",
        "name": "品牌合规卫士",
        "category": "合规风控",
        "cat_key": "risk",
        "cat_class": "cat-risk",
        "desc": "扫描品牌文案，进行 FDA/FTC/Amazon TOS 三轨合规检查，输出违规词清单和逐句合规改写建议。",
        "roi": "预防 FTC 警告和产品下架，单次违规处罚可达 $50,000",
        "linked_skills": ['Skill-Compliance-Scored-Guardrail-Orchestration', 'Skill-Amazon-ToS-Compliance-Guardrail', 'Skill-Cross-Border-Compliance-Framework'],
        "inputs": [{'id': 'copy', 'label': '品牌文案（Listing/广告语/包装文字）', 'type': 'textarea', 'placeholder': '粘贴需要检查的文案内容...'}, {'id': 'category', 'label': '产品品类', 'type': 'select', 'options': ['母婴', '健康保健', '食品饮料', '消费电子', '美妆个护']}, {'id': 'market', 'label': '目标市场', 'type': 'select', 'options': ['US', 'UK/EU', 'AU', '全球']}],
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
        "id": "agent-tiktok-content",
        "icon": "TC",
        "name": "TikTok 内容官",
        "category": "内容营销",
        "cat_key": "content",
        "cat_class": "cat-ad",
        "desc": "输入产品和受众画像，输出 TikTok/Reels 爆款选题矩阵、脚本框架和话题标签策略，降低内容生产成本。",
        "roi": "系统化内容输出降低 CPM 40%，自然流量占比提升至 30%+",
        "linked_skills": ['Skill-DAWN-Talking-Head-Review', 'Skill-AnchorCrafter-Virtual-Anchor-Demo', 'Skill-Creative-Fatigue-Detection'],
        "inputs": [{'id': 'product', 'label': '产品名称/描述', 'type': 'text', 'placeholder': '例：硅胶婴儿餐具套装'}, {'id': 'audience', 'label': '目标受众画像', 'type': 'text', 'placeholder': '例：0-2岁宝妈，关注辅食/育儿'}, {'id': 'style', 'label': '内容风格偏好', 'type': 'select', 'options': ['教程/攻略', '痛点反转', '生活记录', '对比测评', 'UGC种草']}, {'id': 'freq', 'label': '周更新频次', 'type': 'select', 'options': ['3条/周', '5条/周', '每日更新']}],
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
    {
        "id": "agent-dml-counterfactual-pricing",
        "icon": "CP",
        "name": "反事实定价策略 Agent",
        "category": "价格策略",
        "cat_key": "pricing",
        "cat_class": "cat-ops",
        "desc": "基于 DML (双重机器学习) 计算反事实基线，对抗大盘流量的降价内卷幻觉。支持 MAS 跨部门财务/供应链博弈测算，给出真实的净利润率最优解。",
        "roi": "告别无效价格战，通过反事实预判提升净利润率 15%-40%",
        "linked_skills": ['Skill-Counterfactual-Price-Elasticity', 'Skill-Cross-Domain-Orthogonal-Signals', 'Skill-Agentic-Nash-Equilibrium-Debate'],
        "inputs": [{'id': 'current_price', 'label': '我方当前售价（$）', 'type': 'text', 'placeholder': '例：49.0'}, {'id': 'comp_price', 'label': '主要竞品最新售价（$）', 'type': 'text', 'placeholder': '例：38.0 (跳水降价)'}, {'id': 'comp_stockout_risk', 'label': '竞品断货风险系数 (0-1)', 'type': 'select', 'options': ['0.1 (库存充足)', '0.5 (库存预警)', '0.9 (极度高危/即将断货)']}, {'id': 'seasonality', 'label': '大盘流量因子', 'type': 'select', 'options': ['1.0 (平季)', '1.4 (旺季/黑五)']}],
        "demo_output": """[执行阶段] 启动 DML - Reality Checker 测算引擎
--------------------------------------------------
[+] 输入参数: 我方 .0 | 竞品 .0 | 断货风险 0.9 | 流量因子 1.4
[+] 正在剥离大盘流量混杂效应... 
-> 测算真实价格弹性: -5.12 (单/美元)

[三轨对抗推演 - MAS 博弈]
 
🔥 方案 A - 传统竞价 (跟进降价至 .0)
- 预测单量: 189 单
- 毛利润: ,409.46
- [供应链Agent 预警]: 爆单将击穿安全库存线，触发 2 周全价断货真空期。
- [风险合规核验]: PASS (不违规，但属高风险价格战)

💡 方案 B - 反直觉高价 (维持 .0)
- 预测单量: 134 单
- 毛利润: ,876.31
- [财务Agent 评估]: 剥离“旺季降价必卖爆”的幻觉后，降价()换来的增量(55单)无法弥补巨额的单件毛利损失。
- [战略机会点]: 竞品高危断货(0.9)，我方将具有长尾议价权。

--------------------------------------------------
[决议指令] 阻断自动降价！建议执行【方案 B】，预期多赚取净利润 .85 (+13.7%)""",
    },
    {
        "id": "agent-sku-tag-scanner",
        "icon": "TQ",
        "name": "SKU标签质量扫描器",
        "category": "标签工程",
        "cat_key": "tag",
        "cat_class": "cat-supply",
        "desc": "输入SKU列表和目标市场，扫描标签覆盖率/时效性/准确率，识别质量缺口并生成修复优先级清单。",
        "roi": "标签质量提升后断货识别延迟 8h→15min",
        "linked_skills": ['Skill-Tag-Quality-Coverage-KPI', 'Skill-Tag-Schema-Engineering-Lifecycle', 'Skill-Auto-Tagging-Pipeline-Rule-ML-LLM'],
        "inputs": [{'id': 'sku_list', 'label': 'SKU列表（每行一个）', 'type': 'textarea', 'placeholder': 'SKU-001\nSKU-002\nSKU-003\n...'}, {'id': 'tag_types', 'label': '重点检查标签类型', 'type': 'select', 'options': ['全部标签', '库存状态标签', '合规认证标签', '预测风险标签', '财务利润标签']}, {'id': 'market', 'label': '目标市场', 'type': 'select', 'options': ['US', 'EU', 'JP', 'AU', '全球多市场']}],
        "demo_output": """[SKU标签质量扫描器] 分析结果

━━ 扫描概览 ━━
扫描SKU数: 50  目标市场: US
检查标签维度: 库存状态 / 合规认证 / 预测风险 / 财务利润

━━ 标签覆盖率报告 ━━
stockout_risk（断货风险）: 98% ✅ (目标≥99%)
compliance.fda_registered: 62% 🔴 (目标=100%)
abc_class（ABC分类）: 100% ✅
predicted_stockout_7d（7日断货预测）: 78% ⚠️ (目标≥95%)
sku.margin_tier（利润层级）: 45% 🔴 (目标≥90%)

━━ 时效性检查 ━━
stockout_risk 超时（>4h未更新）: 3个SKU ⚠️
predicted_stockout_7d 超时（>24h未更新）: 8个SKU 🔴

━━ 质量缺口TOP5（修复优先级）━━
1. 🔴 FDA认证标签缺失 — 19个SKU无合规标签
   → 影响: 合规审查无法自动化，风险人工遗漏
   → 修复: 从供应商档案自动传播认证标签（Tag传播算法）

2. 🔴 利润标签覆盖不足 — 27个SKU无margin_tier
   → 影响: AI无法识别亏损SKU，无法触发调价Action
   → 修复: 接入FBA费用数据，运行P&L计算流水线

3. ⚠️ 预测标签时效滞后 — 8个SKU 7日断货预测超期
   → 影响: 断货预警不准，可能误报或漏报
   → 修复: 检查预测调度任务是否正常运行

4. ⚠️ 断货风险标签超时 — 3个SKU超过4h未更新
   → 修复: 检查ERP数据源连接是否正常

5. 📝 ABC分类全覆盖但月度更新已超期7天
   → 修复: 触发下次月度重分类任务

━━ 综合质量评分 ━━
当前: 67/100  目标: ≥90/100
主要扣分项: 合规标签缺失(-20) + 预测标签超时(-8) + 利润标签不足(-5)

[!] 建议行动: 优先补充FDA/CE认证标签（1天内完成，接入供应商认证台账即可）""",
    },
    {
        "id": "agent-compliance-matrix",
        "icon": "CM",
        "name": "多市场合规矩阵",
        "category": "合规风控",
        "cat_key": "risk",
        "cat_class": "cat-ad",
        "desc": "输入产品信息和目标市场，自动扫描US/EU/JP/AU合规要求缺口，生成上市准入评估和整改优先级。",
        "roi": "新品上市合规扫描 2周→10分钟，防扣押损失5-15万/次",
        "linked_skills": ['Skill-Multi-Market-Compliance-Matrix-Ontology', 'Skill-EPR-Extended-Producer-Responsibility-Tag', 'Skill-Regulatory-Change-Impact-Propagation'],
        "inputs": [{'id': 'product_name', 'label': '产品名称', 'type': 'text', 'placeholder': '例：Momcozy S12 Pro 双边吸奶器'}, {'id': 'product_type', 'label': '产品类型', 'type': 'select', 'options': ['母婴电子设备', '配方奶粉/食品', '婴儿洗护用品', '婴儿玩具', '其他母婴用品']}, {'id': 'target_markets', 'label': '目标市场（多选）', 'type': 'text', 'placeholder': '例：US, DE, FR, JP（逗号分隔）'}],
        "demo_output": """[多市场合规矩阵] 扫描结果 — Momcozy S12 Pro 双边吸奶器

━━ 合规矩阵概览 ━━
目标市场: US / DE / FR / JP
产品类型: 母婴电子设备

市场     FCC   CE    RoHS  EPR注册  PSE   合规状态
US       ✅    N/A   N/A   N/A     N/A   ✅ 可销售
DE       N/A   ✅    ✅    ❌缺失   N/A   ⚠️ 有缺口
FR       N/A   ✅    ✅    ❌缺失   N/A   ⚠️ 有缺口
JP       N/A   N/A   N/A   N/A    ❌缺失 🔴 不可销售

━━ 关键缺口详情 ━━

🔴 [JP] PSE认证缺失（日本特定电器强制认证）
   → 违规后果: 进口被拒，货物扣押
   → 估算整改周期: 60-90天
   → 整改机构: 日本电器安全与环境研究院（J-QAC）
   → 费用预估: ¥15,000-30,000

⚠️ [DE/FR] EPR包装注册未完成
   → 违规后果: 最高罚款€100,000/SKU + 下架
   → 截止日期: 2025年1月1日（已过期，立即处理）
   → 整改周期: 7-14天（在线注册）
   → 德国: 在 LUCID平台注册（₁ <lucid.verpackungsregister.org>）
   → 法国: 在 CITEO平台注册

━━ 上市准入评分 ━━
US: 100/100 ✅ 可立即销售
DE: 72/100  ⚠️ 需补EPR注册（预计14天完成）
FR: 72/100  ⚠️ 需补EPR注册（预计14天完成）
JP: 35/100  🔴 需PSE认证（预计60-90天）

━━ 行动优先级 ━━
P0（立即）: DE/FR EPR注册（成本低，影响大）
P1（30天内）: 启动JP PSE认证申请流程
P2（持续）: 监控EU REACH成分变化通知""",
    },
    {
        "id": "agent-return-analyzer",
        "icon": "RT",
        "name": "退货根因分析师",
        "category": "客服售后",
        "cat_key": "cs",
        "cat_class": "cat-voc",
        "desc": "输入退货记录和评论数据，三层归因分析（客诉→运营原因→供应链根因），输出改善闭环行动方案。",
        "roi": "根因修复后退货率降低40-60%，年化减少退货成本8万+",
        "linked_skills": ['Skill-Return-Root-Cause-Attribution-Graph', 'Skill-Returnformer-Returns-Prediction', 'Skill-Cross-Border-Return-Rate-By-Country-KPI'],
        "inputs": [{'id': 'return_data', 'label': '退货记录（原因 + 数量，每行一条）', 'type': 'textarea', 'placeholder': '到货破损, 23\n与描述不符, 45\n质量问题, 18\n改变主意, 12'}, {'id': 'sku_id', 'label': 'SKU ID / 产品名称', 'type': 'text', 'placeholder': '例：SKU-S12Pro 或 吸奶器旗舰款'}, {'id': 'market', 'label': '市场', 'type': 'select', 'options': ['US', 'DE', 'UK', 'JP', '全部市场']}],
        "demo_output": """[退货根因分析师] 三层归因报告 — SKU-S12Pro / DE市场

━━ 退货概览 ━━
总退货量: 98件  退货率: 14.2%（行业基准 DE市场: 10-18%）
主要退货原因分布:

━━ 第一层：表层原因 ━━
1. 与描述不符  45件 (45.9%) ← 主要问题
2. 到货破损    23件 (23.5%)
3. 质量问题    18件 (18.4%)
4. 改变主意    12件 (12.2%)

━━ 第二层：运营原因归因 ━━
「与描述不符」45件 → 归因分析:
  ├─ 德文翻译质量差 (68%): 产品页面机器翻译，3处关键功能描述有误
  ├─ 主图与实物不符 (22%): 展示图为旧款，实物已更新设计
  └─ 使用说明不完整 (10%): 缺少德语操作视频

「到货破损」23件 → 归因分析:
  ├─ 包材防护不足 (70%): 内衬泡棉厚度3mm→2mm（供应商降本导致）
  └─ 运输方式问题 (30%): DHL偏远区域处理粗糙

━━ 第三层：根本原因 ━━
🔴 根因1: 德文Listing本地化质量差（直接导致45%退货）
   → 证据: 评论词云中高频出现"nicht wie beschrieben"（与描述不符）
   → 行动: 找母语译者重写德文页面，预算€500，预计7天完成
   → 预期效果: 退货率从14.2%降至9%（近DE基准线）

🟡 根因2: 包材供应商降本导致防护下降（导致23%退货）
   → 证据: 破损退货集中在2月后（与包材变更时间吻合）
   → 行动: 恢复3mm泡棉标准，成本+¥0.3/件
   → 预期效果: 破损退货降低60%

━━ 改善ROI估算 ━━
德文Listing优化: 投入€500 → 年化节省退货成本约€8,000
包材恢复标准: 成本+¥0.3/件 → 年化节省破损退货约¥15,000

[!] 结论: 优先修复德文Listing（ROI最高，1周内可完成）""",
    },
    {
        "id": "agent-margin-calculator",
        "icon": "PL",
        "name": "SKU利润归因计算器",
        "category": "数据分析",
        "cat_key": "analytics",
        "cat_class": "cat-ad",
        "desc": "输入SKU销售和成本数据，生成全链路P&L瀑布图，识别利润漏点，给出可操作的提利行动建议。",
        "roi": "发现亏损SKU后调价/优化，年化利润率提升3-8pp",
        "linked_skills": ['Skill-SKU-Level-Margin-Attribution-Ontology', 'Skill-Supply-Chain-Total-Cost-TCO-Model', 'Skill-GMROI-Inventory-Investment-Efficiency'],
        "inputs": [{'id': 'sku_id', 'label': 'SKU / 产品名称', 'type': 'text', 'placeholder': '例：吸奶器S12 Pro'}, {'id': 'gmv', 'label': '月销售额（元）', 'type': 'text', 'placeholder': '例：150000'}, {'id': 'costs', 'label': '成本明细（格式: 项目=金额，每行一条）', 'type': 'textarea', 'placeholder': '采购成本=45000\nFBA费用=18000\n广告费用=22500\n退货成本=6000\n物流头程=9000'}],
        "demo_output": """[SKU利润归因计算器] P&L 瀑布分析 — 吸奶器S12 Pro

━━ 月度 P&L 瀑布 ━━
GMV（销售收入）           ¥150,000  100.0%
 └─ 平台佣金（12%）       -¥18,000   12.0%
净销售额（NSV）           ¥132,000   88.0%
 └─ 采购成本（COGS）      -¥45,000   30.0%
毛利润                    ¥87,000    58.0%
 └─ FBA/仓储费用          -¥18,000   12.0%
 └─ 广告费用（ACoS 15%）  -¥22,500   15.0%
 └─ 退货成本（4%）        -¥6,000     4.0%
 └─ 头程物流              -¥9,000     6.0%
产品贡献利润              ¥31,500    21.0% ✅

━━ 成本结构诊断 ━━
广告费率 15%：⚠️ 偏高（行业优秀: 10-12%）
退货率导致成本 4%：⚠️ 中等（优化目标: <2.5%）
FBA费率 12%：✅ 正常
头程费率 6%：✅ 正常

━━ 与同类产品对标 ━━
当前净贡献率: 21%  行业P75: 25%  行业P90: 32%
差距: -4pp（约¥6,000/月提升空间）

━━ 提利行动建议 ━━
🎯 P0 降广告ACoS: 15%→12%
   → 操作: 暂停ROI<0.8的关键词，强化精准匹配
   → 预期节省: ¥4,500/月  难度: ⭐⭐☆☆☆

🎯 P1 降退货成本: 4%→2.5%
   → 操作: 优化德文Listing（见退货分析报告）
   → 预期节省: ¥2,250/月  难度: ⭐⭐☆☆☆

🎯 P2 谈判FBA费优化 或 部分切换海外仓
   → 适用SKU: 月销>300件的稳定品
   → 预期节省: ¥1,800/月  难度: ⭐⭐⭐☆☆

━━ 综合提利潜力 ━━
可实现年化净利润提升: ¥102,600（≈ GMV的5.7%）""",
    },
    {
        "id": "agent-geopolitical-risk",
        "icon": "GR",
        "name": "地缘风险评估仪",
        "category": "合规风控",
        "cat_key": "risk",
        "cat_class": "cat-supply",
        "desc": "输入供应商信息和物流路线，评估关税/港口/汇率/出口管制多维地缘风险，生成应急预案建议。",
        "roi": "提前14天预警延误，避免空运附加费$8,000+/次",
        "linked_skills": ['Skill-Geopolitical-Risk-Tag-Supply-Impact', 'Skill-Black-Swan-Scenario-Simulation-Tag', 'Skill-SC-Resilience-Hypergraph'],
        "inputs": [{'id': 'supplier_country', 'label': '供应商所在国家/地区', 'type': 'select', 'options': ['中国大陆', '台湾', '越南', '印度', '马来西亚', '墨西哥', '其他']}, {'id': 'target_market', 'label': '目标销售市场', 'type': 'select', 'options': ['美国', '欧盟', '英国', '日本', '澳大利亚', '多市场']}, {'id': 'logistics_route', 'label': '主要物流路线', 'type': 'text', 'placeholder': '例：上海→洛杉矶（苏伊士运河）或 宁波→汉堡（红海航线）'}],
        "demo_output": """[地缘风险评估仪] 风险评估报告

━━ 风险概览 ━━
供应商来源: 中国大陆  目标市场: 美国
主要路线: 上海→洛杉矶（太平洋直航）
综合地缘风险评分: 62/100（中等偏高）

━━ 五维风险评估 ━━
关税风险     🔴 HIGH (75/100)
  现状: US对华进口关税已额外+7.5-25%（品类差异）
  预警: 2025年贸易政策不确定性高，追加关税概率45%
  应对: 评估越南/墨西哥供应商转移可行性

港口/航运风险 🟡 MEDIUM (50/100)
  现状: 太平洋航线相对稳定（苏伊士未影响此线路）
  预警: 洛杉矶港口拥堵指数本月+15%，ETA延误概率约30%
  应对: 提前备货窗口，考虑西雅图/温哥华备用港口

汇率风险     🟡 MEDIUM (45/100)
  现状: CNY/USD近3月波动率±2.8%
  采购成本影响: 汇率每波动1% = 成本±¥0.5/件（以售价$59计）
  应对: 锁定3个月远期汇率，降低不确定性

出口管制风险  🟢 LOW (25/100)
  现状: 吸奶器不在出口管制清单（非半导体/AI芯片类）
  持续监控: 锂电池组件是否纳入新一轮管制

供应商集中度  🔴 HIGH (80/100)
  现状: 主要供应商（宁波精工）占采购额68%
  风险: 单一供应商故障 = 全量断供
  应对: 立即启动第二供应商开发（目标6个月内完成）

━━ 最高风险情景模拟 ━━
情景: 关税额外+15% + 洛杉矶港口罢工（概率15%，2025年Q3）
影响估算: 利润率下降4-8pp + 延误2-4周
应急预案:
  → 提前3个月备货（当前情景下务必执行）
  → 激活越南备用供应商（建议今年内完成认证）
  → 评估墨西哥前置仓可行性（美国关税减免）

[!] 结论: 关税和供应商集中度是最需立即行动的风险点""",
    },
    {
        "id": "agent-epr-calculator",
        "icon": "EP",
        "name": "EPR合规费用测算",
        "category": "合规风控",
        "cat_key": "risk",
        "cat_class": "cat-ad",
        "desc": "输入产品包装信息，自动计算欧盟德/法/奥等市场EPR注册要求和年度费用，生成注册优先级清单。",
        "roi": "EPR合规成本仅违规罚款的0.2%，提前注册ROI无限大",
        "linked_skills": ['Skill-EPR-Extended-Producer-Responsibility-Tag', 'Skill-Multi-Market-Compliance-Matrix-Ontology', 'Skill-Climate-ESG-Supply-Chain-Tag'],
        "inputs": [{'id': 'product_name', 'label': '产品名称', 'type': 'text', 'placeholder': '例：吸奶器S12 Pro'}, {'id': 'packaging_weight', 'label': '包装总重量（克）', 'type': 'text', 'placeholder': '例：480（含内衬+外箱+泡棉）'}, {'id': 'packaging_material', 'label': '主要包装材料', 'type': 'select', 'options': ['纸板+泡棉', '纸板+塑料', '全纸板', '混合材料']}, {'id': 'eu_markets', 'label': 'EU目标市场', 'type': 'text', 'placeholder': '例：DE, FR, AT, NL（逗号分隔）'}, {'id': 'annual_units', 'label': '年销售量（件）', 'type': 'text', 'placeholder': '例：5000'}],
        "demo_output": """[EPR合规费用测算] 报告 — 吸奶器S12 Pro

━━ 输入参数 ━━
包装重量: 480g  材料: 纸板+泡棉
EU目标市场: DE, FR, AT  年销量: 5,000件

━━ EPR费率计算 ━━
包装材料分解（估算）:
  纸板外箱 280g × €0.30/kg = €0.084/件
  泡棉内衬 120g × €0.90/kg = €0.108/件
  封口带   80g × €1.20/kg  = €0.096/件
单件EPR材料费: €0.288/件

━━ 各市场年度EPR费用 ━━
德国（LUCID注册）:
  材料费: 5,000件 × €0.288 = €1,440
  固定行政费: €200（年度注册费）
  德国年度合计: €1,640

法国（CITEO注册）:
  材料费: 5,000件 × €0.240 = €1,200
  固定行政费: €150
  法国年度合计: €1,350

奥地利（ARA注册）:
  材料费: 5,000件 × €0.265 = €1,325
  固定行政费: €100
  奥地利年度合计: €1,425

━━ 三市场EPR总费用 ━━
年度合计: €4,415（约¥34,000）
占EU市场GMV比例: 约1.1%（可接受）
单件分摊: €0.88/件 → 建议纳入定价成本

━━ 注册优先级 ━━
🔴 P0 德国（LUCID）: 2023年起已强制，违规罚款最高€100,000/SKU
   注册链接: lucid.verpackungsregister.org
   预计完成: 7天  材料: 营业执照+产品清单+包装规格

🔴 P0 法国（CITEO）: 2022年起已强制
   注册链接: citeo.com/pro
   预计完成: 10天

🟡 P1 奥地利（ARA）: 建议在前两个完成后处理

━━ EPR单件成本 vs 违规风险 ━━
合规成本: €0.88/件  vs  违规风险: €100,000/SKU（5000件以上）
结论: 立即注册，ROI无限大

[!] 行动: 本周内启动德国LUCID注册（最高风险市场）""",
    },
    {
        "id": "agent-cold-start-advisor",
        "icon": "CS",
        "name": "新品冷启动助手",
        "category": "选品增长",
        "cat_key": "selection",
        "cat_class": "cat-supply",
        "desc": "新品上架前72小时：需求预测 + 备货建议 + 关键词矩阵 + 冷启动打法一站式决策",
        "roi": "冷启动备货准确率+30%，首批资金浪费降低50%",
        "linked_skills": ['Skill-Real-Options-Product-Launch-Timing', 'Skill-Cross-Border-Cold-Start-Forecast', 'Skill-SIR-Viral-Product-Adoption-Forecasting', 'Skill-Long-Tail-Keyword-Mining', 'Skill-Index-Health-Monitoring'],
        "inputs": [{'id': 'category', 'label': '品类关键词', 'type': 'text', 'placeholder': '例：electric breast pump'}, {'id': 'launch_budget', 'label': '首批备货预算（USD）', 'type': 'text', 'placeholder': '例：5000'}, {'id': 'target_market', 'label': '目标市场', 'type': 'select', 'options': ['Amazon US', 'Amazon EU', 'TikTok Shop', 'Shopee']}, {'id': 'competitors', 'label': 'TOP3竞品ASIN（逗号分隔）', 'type': 'text', 'placeholder': '例：B0XXXXX,B0YYYYY'}],
        "demo_output": """[OK] 新品冷启动决策报告 — electric breast pump

━━ 最优上架时机（实物期权分析）━━
当前时机评分: 82/100（推荐执行）
等待成本: 每延迟1个月，市场份额损失约3.2%
关键触发条件: 竞品库存低谷期（当前满足）+ 季节需求上升期（T+3周入场）

━━ 首批备货建议 ━━
预算: $5,000  推荐备货量: 320件（P50预测）
  保守场景 (P25): 210件，资金占用 $3,280
  基准场景 (P50): 320件，资金占用 $5,000 ✅
  激进场景 (P75): 480件，资金占用 $7,500（超预算）
安全库存天数建议: 45天（含2周前置期缓冲）

━━ 关键词矩阵 ━━
核心词 (高搜索量/高竞争):
  electric breast pump (124K/月) | hospital grade pump (89K/月) | wearable breast pump (67K/月)
长尾词 (低竞争/高转化):
  hands free breast pump for working moms (8.2K/月) ⭐
  quiet breast pump double electric (6.1K/月) ⭐
  breast pump with app bluetooth (4.8K/月) ⭐

━━ 冷启动前4周行动计划 ━━
Week 1: Vine计划申请（目标15个初始评论）+ 关键词广告冷启动（$50/天）
Week 2: A+页面上线 + 精准匹配出价提升20%
Week 3: 评论数≥8后开启自动广告 + 追踪BSR排名变化
Week 4: 评估点击率/转化率，调整出价策略

[+] 建议执行：市场窗口良好，3周内上架最优""",
    },
    {
        "id": "agent-festival-replenishment",
        "icon": "FR",
        "name": "大促备货决策Agent",
        "category": "供应链",
        "cat_key": "supply",
        "cat_class": "cat-supply",
        "desc": "Prime Day/黑五/双十一：需求预测+库存风险+补货时间线+资金占用优化的全链路大促备货方案",
        "roi": "大促备货准确率+20%，断货损失年化减少30-50万元",
        "linked_skills": ['Skill-CVaR-Inventory-Risk-Portfolio', 'Skill-Demand-Forecasting-Supply-Chain', 'Skill-Safety-Stock-Replenishment', 'Skill-Seasonal-Search-Trend-Modeling', 'Skill-Lead-Time-Safety-Stock-Auto-Adjuster'],
        "inputs": [{'id': 'festival', 'label': '大促类型', 'type': 'select', 'options': ['Prime Day', 'Black Friday', '双十一', 'Prime Big Deal Days']}, {'id': 'skus', 'label': '核心SKU列表（ASIN，逗号分隔）', 'type': 'text', 'placeholder': '例：B0XXXXX,B0YYYYY'}, {'id': 'current_inventory', 'label': '当前库存天数', 'type': 'text', 'placeholder': '例：45'}, {'id': 'budget_constraint', 'label': '备货预算上限（USD）', 'type': 'text', 'placeholder': '例：50000'}],
        "demo_output": """[OK] 大促备货决策方案 — Prime Day

━━ CVaR风险分析（置信度95%）━━
当前库存天数: 45天  大促距今: 52天
预期大促倍增系数: 3.2x（历史数据P50）
CVaR(95%) 断货风险: 在极端需求下（P95），库存将在大促第2天耗尽
CVaR(95%) 积压风险: 若需求低于P25，大促后剩余库存率42%

━━ 最优备货量计算 ━━
基准备货量（P50需求）: 当前库存 × 3.2 = 1,440件
风险调整备货量（CVaR优化）: 1,680件（+17%安全垫）
预算约束下的最优方案: 1,580件（$47,400，在预算内）

━━ T-8周到大促当日补货时间线 ━━
T-8周 (现在): 确认供应商交货档期，下达50%采购订单（790件）
T-6周: 追踪搜索趋势，需求信号超P60则追加20%（316件）
T-4周: FBA入仓截止日，剩余订单必须到库
T-2周: 大促前最后一次库存盘点，不足则启动航空紧急补货
大促当日: 实时监控库存消耗速率，触发补货预警阈值

━━ 资金使用计划 ━━
总备货资金: $47,400
  分批采购: T-8周 $23,700（50%）+ T-6周 $9,480（20%）+ T-4周 $14,220（30%）
资金周转率: 大促后15天回款，占用周期约70天
ROI预测: 大促期间额外GMV $142,000，净利润率18%，投资回报率5.4x

[+] 建议：按分批采购计划执行，降低单次资金压力""",
    },
]
