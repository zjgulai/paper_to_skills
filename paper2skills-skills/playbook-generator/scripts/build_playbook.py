#!/usr/bin/env python3
"""Build a static HTML Playbook for paper2skills Skill cards."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from paper2skills_common.assets import detect_code_path, iter_skill_files  # noqa: E402
from paper2skills_common.domains import load_domain_registry  # noqa: E402

GRAPH_SCRIPT_DIR = BASE_DIR / "paper2skills-skills" / "paper-skills-graph" / "scripts"
if str(GRAPH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_SCRIPT_DIR))

from skills_graph_analyzer import SkillsGraph  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Section name normalizer: strips leading circled-number (①②③…) and Chinese
# ordinal prefixes (一、二、三…) so fuzzy matching always works.
# ---------------------------------------------------------------------------
_ORDINAL_PREFIX = re.compile(
    r"^[①②③④⑤⑥⑦⑧⑨⑩\u2460-\u2473]"   # ①–⑳  circled digits
    r"|^[一二三四五六七八九十]+[\.、]\s*"   # 一、二、…
    r"|^\d+[\.、]\s*"                      # 1. 2、…
)

def _norm_title(raw: str) -> str:
    """Normalise a section title: strip ordinal prefix, lowercase."""
    return _ORDINAL_PREFIX.sub("", raw).strip().lower()


# Section key → list of normalised name fragments to match against.
# All entries are already lower-case; matching is substring.
SECTION_KEYS: dict[str, list[str]] = {
    "algorithm": ["算法原理", "核心算法", "算法逻辑", "核心思想"],
    "scenario":  ["母婴出海应用案例", "应用案例", "业务应用", "业务场景", "应用场景",
                  "吸奶器出海应用案例"],
    "code":      ["代码模板", "完整可运行"],
    "guide":     ["使用指南"],
    "value":     ["商业价值", "业务价值", "量化 roi", "量化roi"],
    "relations": ["skill relations", "技能关联", "技能关系", "4. 技能关系", "四、技能关联"],
}

TOPIC_RULES = {
    "广告与投放": ["广告", "roas", "attribution", "tiktok", "keyword", "creative", "marketing", "mmm"],
    "供应链与补货": ["供应链", "库存", "补货", "forecast", "demand", "logistics", "fulfillment", "lead-time"],
    "客服与VOC": ["客服", "review", "voc", "absa", "sentiment", "translation", "customer"],
    "推荐与搜索": ["recommend", "推荐", "search", "retrieval", "ranking", "rerank", "embedding"],
    "知识图谱与RAG": ["rag", "graphrag", "knowledge graph", "知识图谱", "kg", "chunk", "hyde", "raptor", "ontology"],
    "数据采集与治理": ["data collection", "数据采集", "crawl", "quality", "provenance", "dedup", "signal"],
    "MAS与智能体工程": ["mas", "agent", "mcp", "orchestr", "tool", "memory", "trust"],
    "定价与利润": ["pricing", "price", "价格", "elasticity", "margin"],
    "风控与合规": ["fraud", "risk", "compliance", "合规", "风控", "fake"],
    "视觉内容生成": ["video", "visual", "image", "avatar", "ai视频", "multimodal"],
}

WORKFLOW_RULES = {
    "WF-A 智能补货": ["供应链", "库存", "补货", "demand", "forecast", "lead-time", "safety-stock", "logistics"],
    "WF-B 广告优化": ["广告", "roas", "attribution", "tiktok", "keyword", "creative", "mmm", "marketing"],
    "WF-C 客服分诊": ["客服", "review", "voc", "absa", "translation", "customer", "sentiment"],
    "WF-D 选品扫描": ["选品", "product", "market", "competitive", "signal", "data collection", "knowledge graph"],
    "WF-E Review监控": ["review", "fake-review", "sentiment", "absa", "dedup", "quality"],
    "WF-F 动态定价": ["pricing", "price", "价格", "elasticity", "markdown", "定价", "竞价", "discount"],
    "WF-G Listing内容优化": ["listing", "content", "copywriting", "主图", "视频", "a/b", "creative", "文案"],
    "WF-H 复购增长": ["churn", "ltv", "retention", "复购", "流失", "rfm", "lifecycle", "cohort"],
    "WF-I 智能体工程": ["agent", "智能体", "mas", "mcp", "llm agent", "workflow", "tool use", "safety guard", "监控", "部署"],
}

KNOWN_SKILL_IDS: set[str] = set()

ALGO_TAG_RULES = {
    "causal": ["causal", "因果", "uplift", "dml", "did", "iv"],
    "experiment": ["ab", "a/b", "experiment", "bandit", "实验"],
    "forecasting": ["forecast", "time series", "预测", "demand"],
    "optimization": ["optimization", "优化", "allocation", "scheduling"],
    "recommendation": ["recommend", "推荐", "ranking"],
    "rag": ["rag", "retrieval", "chunk", "hyde", "raptor", "rerank"],
    "knowledge_graph": ["knowledge graph", "知识图谱", "kg", "ontology", "entity resolution"],
    "multi_agent": ["mas", "multi-agent", "agent", "orchestr"],
    "data_collection": ["data collection", "数据采集", "crawl", "signal"],
    "fraud_detection": ["fraud", "risk", "anomaly"],
    "pricing": ["pricing", "price", "价格"],
    "visual_generation": ["video", "visual", "image", "multimodal"],
}

# ---------------------------------------------------------------------------
# Domain → Business Context mapping (22 domains × role/trigger/outcome/pain)
# Used to inject a "business perspective panel" on every skill detail page.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Skill-level config overrides — loaded from config/ YAML files at startup
# ---------------------------------------------------------------------------
def _load_config_dir() -> tuple[dict, dict, dict]:
    """Load SKILL_PS_OVERRIDE, SKILL_BIZ_CONTEXT_OVERRIDE, SKILL_HANDBOOK_MAP from YAML."""
    import yaml as _yaml

    _config_dir = Path(__file__).parent / "config"

    def _load(fname: str) -> dict:
        p = _config_dir / fname
        if not p.exists():
            return {}
        return _yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    ps = _load("skill_ps_override.yaml")
    biz = _load("skill_biz_context_override.yaml")
    # SKILL_HANDBOOK_MAP: convert {id, name} dicts back to (id, name) tuples
    hb_raw = _load("skill_handbook_map.yaml")
    hb = {sid: [(item["id"], item["name"]) for item in lst]
          for sid, lst in hb_raw.items()}
    return ps, biz, hb


SKILL_PS_OVERRIDE, SKILL_BIZ_CONTEXT_OVERRIDE, SKILL_HANDBOOK_MAP = _load_config_dir()


DOMAIN_BUSINESS_CONTEXT: dict[str, dict[str, Any]] = {
    "01-因果推断": {
        "role": "增长负责人 / CMO",
        "role2": "数据分析师 · 广告优化师",
        "trigger": "广告预算花了，但不确定哪个渠道真的带来新客；做了大促，不知道销量增长是促销效果还是季节规律",
        "outcome": "能区分「真实增量」和「自然购买」，砍掉虚假归因渠道后同等预算 ROI 提升 20-40%",
        "pain": "钱花出去了不知道有没有用 · 各渠道报告都说自己贡献最大 · 怎么向老板证明这笔钱值得花",
        "platform": "Amazon · TikTok Shop · Meta Ads · DTC 独立站",
    },
    "02-A_B实验": {
        "role": "运营负责人 / 产品经理",
        "role2": "广告优化师 · 选品负责人",
        "trigger": "改了主图/标题/价格，不确定销量变化是改动导致的还是流量波动；两个方案团队各持己见，需要数据裁决",
        "outcome": "每次改动都有 ≥95% 置信度的数据结论，好的改动快速全量，坏的及时止损",
        "pain": "改了主图感觉好多了但不确定 · 小范围测试结果好全量后没效果 · 测试周期短结论不可靠",
        "platform": "Amazon Listing · TikTok 广告素材 · DTC 落地页",
    },
    "03-时间序列": {
        "role": "供应链负责人 / 采购负责人",
        "role2": "运营负责人 · 财务负责人",
        "trigger": "大促前备货总是不是多了就是少了；新品上线第一个月断货，再补又积压；年底预算不知道各月目标怎么定",
        "outcome": "提前 4-8 周准确预判各 SKU 需求峰值，库存积压减少 30%，断货率降低 50%",
        "pain": "备货总是压货或断货 · 旺季淡季波动太大预测不准 · 补货周期 30 天但预测只看 7 天",
        "platform": "Amazon FBA · 海外仓 · 多市场多仓",
    },
    "04-供应链": {
        "role": "供应链负责人",
        "role2": "采购负责人 · CEO / 运营 VP",
        "trigger": "库存周转率低，资金压在海外仓出不来；SKU 断货紧急空运，物流成本吃掉毛利；多仓库存分布不均",
        "outcome": "库存周转天数从 90 天降到 60 天，断货率 <3%，海外仓综合成本降低 15-25%",
        "pain": "库存周转天数太长资金压死了 · 断货了只能空运救急成本爆了 · 多市场库存分配不均",
        "platform": "Amazon FBA · 海外仓 · 多国仓位（美/欧/日）",
    },
    "05-推荐系统": {
        "role": "运营负责人 / 选品负责人",
        "role2": "产品经理 · 广告优化师",
        "trigger": "老客来了只买一件就走，相关产品没被推出去；Bundle 商品连带销售做不起来；站内推荐位点击率低",
        "outcome": "老客连带购买率提升 20-35%，客单价提升，品类交叉销售做起来",
        "pain": "老客复购率上不去 · 相关产品没有被看到 · Bundle 凑单没人用 · 新品没有曝光机会",
        "platform": "Amazon · DTC 独立站 · 邮件/SMS 个性化",
    },
    "06-增长模型": {
        "role": "CEO / 增长负责人",
        "role2": "CMO · 财务负责人",
        "trigger": "公司增长放缓，不知道是市场饱和还是产品问题还是获客太贵；老板要 12 个月 GMV 预测，只能靠感觉",
        "outcome": "建立增长拆解模型找到瓶颈，预测未来 6-12 个月营收区间，支撑融资/战略会议",
        "pain": "增长放缓不知道问题在哪 · CAC 越来越高已经高于 LTV · 新市场要不要进没有数据支撑",
        "platform": "Amazon · TikTok Shop · DTC 独立站 · 多市场",
    },
    "07-NLP-VOC": {
        "role": "产品运营负责人 / 选品负责人",
        "role2": "客服负责人 · 品牌负责人",
        "trigger": "每月几千条差评和 Q&A 没有人力一条条看，但痛点都在里面；新品开发不知道做什么功能、改什么问题",
        "outcome": "自动提取 Top 10 高频痛点，新品开发有用户数据背书，每月出竞品用户洞察报告",
        "pain": "差评太多看不过来 · 不知道用户真正在意什么 · 竞品评论没有系统分析过 · 新品开发靠拍脑袋",
        "platform": "Amazon Reviews / Q&A · TikTok 评论区 · Reddit 母婴社区",
    },
    "08-知识图谱": {
        "role": "选品负责人 / 运营负责人",
        "role2": "数据分析师 · 供应链负责人",
        "trigger": "品类很多，不清楚品类间的关联，没法做系统性类目扩张规划；竞品矩阵太复杂，品牌/SKU/渠道理不清",
        "outcome": "建立品类知识图谱，清晰看到哪些是入口品/引流品/利润品，指导下一步选品扩张方向",
        "pain": "品类太多不知道先做哪个 · 竞品关系理不清楚 · 不知道用户买了奶瓶还会买什么 · 类目扩张没有逻辑",
        "platform": "Amazon 品类体系 · 竞品 ASIN 网络分析",
    },
    "09-DataAgent-LLM": {
        "role": "数据分析师 / 运营负责人",
        "role2": "CEO · 供应链负责人",
        "trigger": "数据需求太多，数据团队排期 2 周；非技术人员（采购/客服/运营）有数据问题但不会 SQL；重复报表占用大量时间",
        "outcome": "业务方用自然语言自助查数据，常规报表自动化，数据驱动决策响应速度从「天」变「分钟」",
        "pain": "数据需求排期太长 · 不会 SQL 只能等数据团队 · 老板临时要数据没法马上出 · 分析师时间都花在取数上",
        "platform": "Amazon SP API · Shopify · TikTok Ads API · 多平台数据整合",
    },
    "10-MAS": {
        "role": "运营负责人 / CTO",
        "role2": "产品经理 · CEO",
        "trigger": "运营任务太碎，选品/定价/广告/客服同时跑，人手严重不足；重复性运营动作需要 7×24 响应但没有足够人力",
        "outcome": "多个 AI Agent 协作自动完成跨系统运营任务，运营团队人效提升 3-5 倍，7×24 无人值守运营",
        "pain": "运营人手不够任务太多 · 价格变化没有及时响应 · 重复性工作占据太多时间 · 想做 7×24 监控但没人盯",
        "platform": "Amazon PPC + 库存 + 定价 多 Agent 协作 · TikTok 内容运营流水线",
    },
    "11-AI人文": {
        "role": "品牌负责人 / 内容运营",
        "role2": "CEO · 社媒运营",
        "trigger": "品牌内容同质化，想在母婴赛道建立有温度有记忆点的品牌人设；海外用户文化差异大，本地化内容难以真正有共鸣",
        "outcome": "品牌内容从「产品介绍」升级为「情感共鸣的故事」，海外用户分享率和评论互动率提升",
        "pain": "内容没有灵魂用户不爱看 · AI 写的东西太像 AI · 不同文化的妈妈怎么打动 · 品牌故事讲不出来",
        "platform": "TikTok · Instagram · DTC 品牌站 · 母婴社媒内容",
    },
    "12-ML基础": {
        "role": "数据分析师 / 数据工程师",
        "role2": "运营负责人 · 产品经理",
        "trigger": "想用机器学习解决业务问题，但不知道该选什么模型；模型上线后效果越来越差不知道为什么",
        "outcome": "选对算法工具减少 50% 试错时间，模型上线后可监控可解释，数据团队和业务团队建立共同语言",
        "pain": "不知道该用什么模型 · 模型准确率不稳定 · 业务不相信模型结果 · 模型黑盒说不清为什么这么预测",
        "platform": "选品评分 · 差评预测 · 用户流失预警 · 广告出价预测",
    },
    "13-广告分析": {
        "role": "广告优化师 / 投放负责人",
        "role2": "CMO · 运营负责人",
        "trigger": "广告账户几十个系列，不知道哪个在真正赚钱；ROAS 看起来好看但实际利润没有提升；预算有限想集中打高价值用户",
        "outcome": "每分广告预算有明确 ROI 追踪，砍掉低效渠道后同等预算 ROAS 提升 30-50%",
        "pain": "ROAS 好看但利润没有涨 · 不知道哪个素材真的有效 · 归因窗口期不同数据打架 · TikTok/Meta/Amazon 广告数据整合不了",
        "platform": "Amazon PPC（SP/SB/SD）· TikTok Ads · Meta 广告 · 多平台归因",
    },
    "14-用户分析": {
        "role": "运营负责人 / 用户增长负责人",
        "role2": "CMO · 产品经理",
        "trigger": "有大量老客户，但不知道谁是高价值客户、谁快要流失；新客获取成本越来越高，老客复购却上不去",
        "outcome": "用户按 RFM/LTV 分层精准触达，高价值用户留存率提升，老客贡献收入占比从 30% 提升到 50%",
        "pain": "老客复购率上不去 · 不知道哪些用户要流失了 · 所有用户用同一套活动 · 买过一次就不见了",
        "platform": "Amazon 买家分层 · DTC 站 LTV 预测 · Klaviyo/Brevo 邮件分群",
    },
    "15-营销投放分析": {
        "role": "CMO / 营销负责人",
        "role2": "广告优化师 · CEO",
        "trigger": "同时跑 Amazon 广告/TikTok/网红投放/邮件，不知道整体预算怎么分配最高效；网红投放花了大钱但不知道带来多少真实 GMV",
        "outcome": "建立全渠道营销归因模型（MMM），每个渠道真实 ROI 可量化，大促前做预算优化模拟",
        "pain": "多渠道预算分配靠感觉 · 网红带货效果不知道怎么量化 · 渠道之间互相抢功劳数据打架 · 整体营销 ROI 算不清楚",
        "platform": "Amazon + TikTok + Meta + KOL 四渠道 · Prime Day / Black Friday 预算前置",
    },
    "16-智能体工程": {
        "role": "CTO / 技术负责人",
        "role2": "产品经理 · 数据工程师",
        "trigger": "想把 AI 集成到业务系统，但 LLM 稳定性差、幻觉问题、成本控制都是挑战；Agent 任务失败了不知道哪步出了问题",
        "outcome": "AI Agent 在生产环境稳定运行，失败可追踪，成本可控，复杂任务完成率 >85%",
        "pain": "LLM 返回结果不稳定不可靠 · AI 幻觉导致业务决策错误 · Agent 任务失败了不知道哪步出问题 · AI 调用成本控制不住",
        "platform": "跨境运营 AI Agent 工程落地 · Amazon SP API + LLM 集成 · 多平台数据采集 Agent",
    },
    "17-价格优化": {
        "role": "定价负责人 / 运营负责人",
        "role2": "选品负责人 · CEO",
        "trigger": "竞品突然降价，不知道该不该跟，跟了怕伤利润不跟怕丢 BSR；大促期间不知道折扣给多少，给多了利润没了",
        "outcome": "实时监控竞品价格并自动触发调价，毛利率保持在目标区间，BSR 排名和利润同时兼顾",
        "pain": "竞品降价了不知道要不要跟 · 大促折扣给多少没有依据 · 手动盯价格太累反应不及时 · 新品上线定价高了还是低了",
        "platform": "Amazon Buy Box 竞价策略 · 多市场价格协调 · Prime Day / Coupon 折扣优化",
    },
    "18-物流履约": {
        "role": "物流负责人 / 供应链负责人",
        "role2": "客服负责人 · 运营负责人",
        "trigger": "物流时效不稳定，差评里大量「收货太慢」，影响 DSR 评分；退货率高，处理成本吃掉大量利润；旺季物流爆仓",
        "outcome": "物流时效提升 20-30%，物流相关差评减少 40%，退货成本可控，旺季履约稳定不崩溃",
        "pain": "物流超时差评太多 · 旺季爆仓订单积压 · 退货处理成本太高 · 头程运费太贵压缩了毛利",
        "platform": "FBA vs FBM vs 第三方海外仓 · 美国本土最后一公里 · 跨境退货逆向物流",
    },
    "19-风控反欺诈": {
        "role": "运营负责人 / 合规负责人",
        "role2": "品牌负责人 · CEO",
        "trigger": "竞品刷单刷好评，自己的 BSR 和评分被打压；账号/ASIN 被恶意投诉删除；店铺有异常订单不确定是真实买家",
        "outcome": "识别过滤刷评/恶意竞争行为，账号风险提前预警，维权有数据证据，降低封号风险",
        "pain": "竞品刷评打压我们 · 我们的好评被恶意举报删除 · 不知道差评是真实的还是恶意的 · 如何证明竞品恶意行为",
        "platform": "Amazon 刷评检测与举报 · TikTok Shop 刷单识别 · 竞品 Listing 攻击溯源",
    },
    "20-AI视频生成": {
        "role": "内容运营 / 品牌负责人",
        "role2": "社媒运营 · CMO",
        "trigger": "TikTok/Reels 需要大量视频，拍摄成本高周期长产能跟不上；想做直播带货但真人主播成本高语言是障碍",
        "outcome": "视频内容产能提升 5-10 倍，单条视频成本降低 80%，多语言市场内容本地化快速覆盖",
        "pain": "视频内容来不及做 · 拍视频成本太高 · 主播太贵或不稳定 · 多语言内容没有人拍 · TikTok 更新频率要求太高",
        "platform": "TikTok Shop LIVE · Instagram Reels · 多语言虚拟主播（英/西/阿/日）",
    },
    "21-合规决策": {
        "role": "合规负责人 / 选品负责人",
        "role2": "CEO · 供应链负责人",
        "trigger": "新品上架前不确定在美国/欧盟是否需要认证，怕因合规问题被下架；产品被平台下架但不清楚哪里出了问题",
        "outcome": "上架前自动完成合规预扫描，0 合规下架事故，新市场合规准备时间从 3 个月缩短到 2 周",
        "pain": "产品被下架说是合规问题 · 不知道目标市场需要什么认证 · EU/US 合规要求不一样怎么处理 · 母婴产品安全标准太严怕踩雷",
        "platform": "美国 CPSC/ASTM · 欧盟 CE/EN71 · Amazon 类目合规要求 · 德国/英国/中东市场",
    },
    "22-数据采集工程": {
        "role": "数据工程师 / 技术负责人",
        "role2": "运营负责人 · 选品负责人",
        "trigger": "想监控竞品价格/评论/排名但没有稳定采集能力，手动太慢；多平台数据分散整合成本极高；数据管道不稳定经常断",
        "outcome": "竞品价格/评论数据每日自动更新，多平台数据统一入仓，数据管道稳定性 >99%，取数时间从小时降到分钟",
        "pain": "竞品数据要手动收集太慢 · 平台 API 限制抓不到数据 · 多系统数据整合不起来 · 报表用的数据是过期的",
        "platform": "Amazon SP API + Keepa · TikTok Shop API · 跨境多平台数据湖",
    },
    "23-运营财务": {
        "role": "CFO / 财务负责人",
        "role2": "CEO · 运营负责人",
        "trigger": "月度 FBA 账单 15 万但不知道哪些 SKU 在亏损；大促备货资金不够但不知道缺口多少；整体利润率 18% 但不知道是哪条产品线在拖累",
        "outcome": "SKU 级 P&L 实时可见，FBA 费用长库龄提前预警，大促现金流缺口提前识别，融资窗口精准规划",
        "pain": "FBA 费用算不清楚 · 现金流紧张不知道哪里漏了 · 哪个 SKU 真正赚钱看不见 · 财务数据滞后一个月才出来",
        "platform": "Amazon Seller Central · Amazon SP API · FBA 报告 · 多货币财务系统",
    },
}

# ---------------------------------------------------------------------------
# Business-problem → workflow quick-entry (for home page C-redesign)
# ---------------------------------------------------------------------------
BUSINESS_ENTRIES = [
    {
        "icon": "AG",
        "label": "防御竞品攻击 / 平台封号预防",
        "desc": "广告刷量、虚假差评、AI 推荐注入、合规封号——四条战线主动防御",
        "href": "playbooks/pb-risk-defense.html",
        "tag": "风险防御",
    },
    {
        "icon": "TR",
        "label": "关税冲击 / 贸易政策应对",
        "desc": "72 小时内输出完整行动清单：定价调整 + 库存处置 + 供应链转移方案",
        "href": "playbooks/pb-tariff-response.html",
        "tag": "关税响应",
    },
    {
        "icon": "CL",
        "label": "上架合规 / 关税编码优化",
        "desc": "新品上架前合规预扫描 + HTS 关税编码精准分类 + 封号风险防御三合一",
        "href": "playbooks/pb-compliance.html",
        "tag": "合规手册",
    },
    {
        "icon": "PL",
        "label": "提升广告 ROI / 归因准确性",
        "desc": "识别无效预算、纠正渠道归因偏差、实现因果驱动的广告优化",
        "href": "workflows/wf-b-广告优化.html",
        "tag": "WF-B 广告优化",
    },
    {
        "icon": "SC",
        "label": "FBA 库存健康 / 头程优化",
        "desc": "长库龄清仓 + 头程路线成本优化 + 旺季备货计划，库存周转天数降 30%",
        "href": "playbooks/pb-fba-operations.html",
        "tag": "FBA 运营",
    },
    {
        "icon": "VP",
        "label": "竞品差评 → 新品机会挖掘",
        "desc": "竞品 1-3 星差评是最好的免费 R&D，新品成功率从 30% 提升到 50%",
        "href": "playbooks/pb-voc-product-loop.html",
        "tag": "竞品情报",
    },
    {
        "icon": "CS",
        "label": "客服 24h 自动化 / 差评防御",
        "desc": "70% 工单全自动处理，多语言覆盖，INR 欺诈退货从 35% 降至 5%",
        "href": "playbooks/pb-customer-service-agent.html",
        "tag": "客服售后",
    },
    {
        "icon": "VO",
        "label": "分析用户评价 / 发现产品痛点",
        "desc": "多语言 VOC 挖掘、差评根因归类、产品改进信号提取",
        "href": "workflows/wf-c-客服分诊.html",
        "tag": "WF-C 客服",
    },
    {
        "icon": "NP",
        "label": "评估新品 / 新市场机会",
        "desc": "市场规模估算、竞品情报采集、选品可行性综合评分",
        "href": "workflows/wf-d-选品扫描.html",
        "tag": "WF-D 选品",
    },
    {
        "icon": "UG",
        "label": "预测用户流失 / 提升 LTV",
        "desc": "Uplift 建模识别可干预用户，精准发券减少无效留存成本",
        "href": "domains/14-用户分析.html",
        "tag": "用户分析",
    },
    {
        "icon": "AI",
        "label": "AI Agent 替代重复性岗位",
        "desc": "供应链对账、数据分析提数、广告出价——三类岗位 70% 重复工作 Agent 覆盖",
        "href": "playbooks/pb-agent-replace.html",
        "tag": "Agent 替人",
    },
    {
        "icon": "PR",
        "label": "动态定价 / A/B 实测 GMV +13%",
        "desc": "LLM 动态定价引擎，定价是乘数——精准定价 1% 比多投广告 15% 更高效",
        "href": "playbooks/pb-pricing-engine.html",
        "tag": "定价引擎",
    },
    {
        "icon": "NP",
        "label": "新品冷启动备货 / 预测",
        "desc": "零历史数据下的扩散曲线预测，跨市场迁移学习",
        "href": "playbooks/pb-new-product-launch.html",
        "tag": "新品冷启动",
    },
    {
        "icon": "CR",
        "label": "广告归因打架 / 渠道预算分配",
        "desc": "PVM 窗口统一 480万/年，Bayesian MMM 1000万——让四份报告说一个真相",
        "href": "playbooks/pb-attribution-unification.html",
        "tag": "全渠道归因",
    },
]


@dataclass
class PlaybookSkill:
    skill_id: str
    title: str
    domain_key: str
    domain_dir: str
    path: str
    algorithm_summary: str = ""
    problem_solved: str = ""
    business_scenarios: list[str] = field(default_factory=list)
    scenario_paragraphs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    roi: list[str] = field(default_factory=list)
    roi_figure: str = ""          # e.g. "10-20 万元"
    difficulty: str = ""          # e.g. "⭐⭐⭐☆☆"
    priority: str = ""            # e.g. "⭐⭐⭐⭐☆"
    papers: list[str] = field(default_factory=list)
    code_path: str | None = None
    code_blocks: int = 0
    code_preview: str = ""
    relations: dict[str, list[str]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    workflows: list[str] = field(default_factory=list)
    biz_role: str = ""
    biz_role2: str = ""
    biz_trigger: str = ""
    biz_outcome: str = ""
    biz_pain: str = ""
    biz_platform: str = ""


def slugify(value: str) -> str:
    value = re.sub(r"^Skill-", "", value)
    value = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff_-]+", "-", value)
    return value.strip("-").lower() or "item"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[end + 4:]
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data, body


def section_map(body: str) -> dict[str, str]:
    """Build section_key → content dict using normalised title matching."""
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", body, re.MULTILINE))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        raw_title = match.group(1).strip()
        norm = _norm_title(raw_title)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        for key, names in SECTION_KEYS.items():
            if any(name in norm for name in names):
                # Keep first match per key (highest in document)
                if key not in sections:
                    sections[key] = content
    return sections


def first_nonempty_line(text: str, fallback: str = "") -> str:
    for line in text.splitlines():
        clean = re.sub(r"[#>*`\-]+", "", line).strip()
        if clean and not clean.startswith("|") and len(clean) > 8:
            return clean[:220]
    return fallback


def _clean_problem_solved(text: str, skill_id: str = "") -> str:
    text = re.sub(r"^[：:。\s]+", "", text).strip()
    text = re.sub(r"^\*\*\s*[：:]?\s*", "", text).strip()
    text = re.sub(r"^是\s*[：:]\s*", "", text).strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text).strip()
    text = re.sub(r"\*\*\.?$", "", text).strip()
    text = re.sub(r"\*\*", "", text).strip()
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"[：:]\s*$", "", text).strip()
    return text[:280]


def first_bold_sentence(text: str, fallback: str = "") -> str:
    """Extract problem statement: prefer labelled 核心问题/业务问题 paragraph, then first clean sentence."""
    for marker in ("核心问题", "业务问题", "核心挑战", "解决的核心问题", "核心痛点"):
        m = re.search(r"(?:" + marker + r")[：:\s]*([^\n。！？]{20,300}[。！？\n]?)", text)
        if m:
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()
            if len(clean) > 20:
                return clean[:250]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```") or stripped.startswith("|"):
            continue
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        clean = re.sub(r"\*(.+?)\*", r"\1", clean)
        clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", clean)
        clean = re.sub(r"^\$\$.*", "", clean).strip()
        clean = re.sub(r"^[-*>]+\s*", "", clean).strip()
        if len(clean) > 30:
            return clean[:250]
    return fallback


def extract_title(frontmatter: dict[str, str], body: str, skill_id: str) -> str:
    if frontmatter.get("title"):
        return frontmatter["title"]
    for line in body.splitlines()[:30]:
        if line.startswith("# Skill Card:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("# "):
            return line[2:].strip()
    return skill_id


def extract_list_snippets(text: str, limit: int = 6) -> list[str]:
    snippets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", stripped[2:]).strip()
            # skip relation / paper reference lines
            if len(clean) > 8 and not clean.startswith("**前置") and not clean.startswith("**组合"):
                snippets.append(clean[:180])
        if len(snippets) >= limit:
            break
    return snippets


def extract_scenario_paragraphs(text: str, limit: int = 3) -> list[str]:
    """Extract meaningful prose paragraphs from the scenario section."""
    paras: list[str] = []
    # Split on double newline, skip headers and code blocks
    in_code = False
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.startswith("#"):
            continue
        if line.strip() == "":
            chunk = " ".join(buf).strip()
            if len(chunk) > 20:
                # clean markdown formatting
                chunk = re.sub(r"\[\[([^\]]+)\]\]", r"\1", chunk)
                chunk = re.sub(r"\*\*(.+?)\*\*", r"\1", chunk)
                chunk = re.sub(r"\*(.+?)\*", r"\1", chunk)
                chunk = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", chunk)
                paras.append(chunk[:300])
            buf = []
        else:
            buf.append(line.strip())
    if buf:
        chunk = " ".join(buf).strip()
        if len(chunk) > 20:
            paras.append(chunk[:300])
    return [p for p in paras if len(p) > 20][:limit]


def extract_roi(value_text: str) -> tuple[str, str, str]:
    """
    Parse the ⑤ 商业价值 section.
    Returns (roi_figure, difficulty_stars, priority_stars).
    """
    roi_figure = ""
    difficulty = ""
    priority = ""

    # Format A: "- **ROI**：10-20 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐⭐☆"
    inline = re.search(
        r"(?:ROI|年化)[：:]\s*([^\|*\n]{3,30})"
        r"(?:[^|]*\|[^|]*难度[：:]\s*([⭐☆]{3,6}))?",
        value_text,
    )
    if inline:
        roi_figure = inline.group(1).strip().rstrip("|").strip()
        if inline.group(2):
            difficulty = inline.group(2).strip()

    # Format B: "年化：**XX万元**"
    if not roi_figure:
        m = re.search(r"(?:年化|增收)[：:]\s*\*?\*?([^\n*]{3,30})", value_text)
        if m:
            roi_figure = m.group(1).strip()

    # Format C: bullet "- 避免损失：xxx，节省 30 万元"
    if not roi_figure:
        m = re.search(r"-\s[^\n]{0,40}(?:节省|节约|增收|创造)[^\n]{0,20}(\d[\d,\.\-]*\s*万元)", value_text)
        if m:
            roi_figure = m.group(1).strip()

    # Format D: bare "¥XX万" anywhere
    if not roi_figure:
        m = re.search(r"(\d[\d,\.\-]+\s*万(?:元|\/月|\/年)?)", value_text)
        if m:
            roi_figure = m.group(1).strip()

    # Format E: table row "| ROI预估 | xxx |" or "| 核心收益 | xxx |"
    if not roi_figure:
        m = re.search(r"\|\s*(?:ROI预估|年化收益|核心收益|节省成本)[^\|]*\|\s*([^\|\n]{3,60})\|", value_text)
        if m:
            roi_figure = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()[:40]

    # Difficulty: table "| 实现难度 | ⭐⭐⭐ |"
    if not difficulty:
        m = re.search(r"(?:实现)?难度[^|]*\|\s*([⭐☆]{3,6})", value_text)
        if m:
            difficulty = m.group(1).strip()
    # Difficulty: bold "**实施难度**：⭐⭐⭐☆☆"
    if not difficulty:
        m = re.search(r"\*\*(?:实施|实现)?难度\*\*[：:]\s*([⭐☆]{3,6})", value_text)
        if m:
            difficulty = m.group(1).strip()
    # Difficulty: unbolded "实施难度：⭐⭐☆☆☆（1/5星）"
    if not difficulty:
        m = re.search(r"(?:实施|实现)?难度[：:]\s*([⭐☆]{2,6})", value_text)
        if m:
            difficulty = m.group(1).strip()
    # Difficulty: "评分：⭐☆☆☆☆" pattern
    if not difficulty:
        m = re.search(r"评分[：:]\s*([⭐☆]{3,6})", value_text)
        if m:
            difficulty = m.group(1).strip()

    # Priority — four formats:
    # A: "优先级：⭐⭐⭐⭐☆"
    # B: "**评分：⭐⭐⭐⭐⭐（5/5星）**" or "**评分:⭐⭐⭐⭐☆(4/5 星)**"
    # C: "⭐⭐⭐⭐⭐ (5/5)" bare stars with fraction
    # D: table "** | ⭐⭐⭐☆☆ |" (bold in table cell)
    if not priority:
        m = re.search(r"(?:商业)?优先级[：:*|\s]*([⭐☆]{3,6})", value_text)
        if m:
            priority = m.group(1).strip()
    if not priority:
        m = re.search(r"\*\*评分[：:]\s*([⭐☆]{3,6})", value_text)
        if m:
            priority = m.group(1).strip()
    if not priority:
        m = re.search(r"([⭐☆]{3,6})\s*\([1-5]/5", value_text)
        if m:
            priority = m.group(1).strip()
    if not priority:
        m = re.search(r"\*\*\s*\|\s*([⭐☆]{3,6})\s*\|", value_text)
        if m:
            priority = m.group(1).strip()

    return roi_figure, difficulty, priority


def extract_roi_from_scenario(scenario_text: str) -> str:
    """Pull ROI figure from the scenario/case section when value section has none."""
    # Pattern 1: 年化/增收/节省：¥XX 万元
    m = re.search(r"(?:年化|增收|节省|节约)[：:]\s*\*?\*?([^\n*]{3,40})", scenario_text)
    if m:
        return m.group(1).strip()
    # Pattern 2: bare XX万元 following a benefit phrase
    m = re.search(
        r"(?:减少|节省|节约|增加|提升)[^\n，。]{0,20}(\d[\d,\.\-]*\s*万元)",
        scenario_text,
    )
    if m:
        return m.group(1).strip()
    # Pattern 3: ¥XX万 or XX万元 anywhere prominent
    m = re.search(r"\*\*([¥￥]?\d[\d,\.\-]*\s*万元)\*\*", scenario_text)
    if m:
        return m.group(1).strip()
    # Pattern 4: +XX% 提升 as last-resort summary
    m = re.search(r"(\+\d[\d\.]*%[^，。\n]{0,20}(?:提升|增长|增加))", scenario_text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_first_code_block(code_section: str) -> str:
    m = re.search(r"```(?:python|bash|sql)?\n(.*?)```", code_section, re.DOTALL)
    if m:
        raw = m.group(1)
        lines = raw.splitlines()
        if len(lines) > 60:
            lines = lines[:60]
        return "\n".join(lines)
    return ""


def extract_papers(text: str) -> list[str]:
    ids = sorted(set(re.findall(r"\b\d{4}\.\d{4,5}\b", text)))
    return ids[:12]


def classify(text: str, rules: dict[str, list[str]]) -> list[str]:
    lower = text.lower()
    labels = []
    for label, needles in rules.items():
        if any(needle.lower() in lower for needle in needles):
            labels.append(label)
    return labels


def build_graph(vault: Path) -> SkillsGraph:
    graph = SkillsGraph(str(vault))
    graph.build_graph()
    return graph


def build_skills(root: Path, vault: Path, graph: SkillsGraph) -> list[PlaybookSkill]:
    registry = load_domain_registry(root)
    domain_by_dir = {entry.vault_dir: entry.key for entry in registry.entries}
    skills: list[PlaybookSkill] = []

    node_by_id = graph.nodes
    for file_path in iter_skill_files(root):
        if not file_path.name.startswith("Skill-"):
            continue
        rel_path = file_path.relative_to(root).as_posix()
        skill_id = file_path.stem
        text = read_text(file_path)
        fm, body = parse_frontmatter(text)
        sections = section_map(body)
        domain_dir = file_path.parent.name
        domain_key = domain_by_dir.get(domain_dir, slugify(domain_dir))
        node = node_by_id.get(skill_id)
        relations = {
            "prerequisite": sorted(node.prerequisites) if node else [],
            "extends": sorted(node.extensions) if node else [],
            "combinable": sorted(node.combinable) if node else [],
        }
        full_text_for_classify = "\n".join([skill_id, domain_dir, text[:5000]])
        code_path = detect_code_path(root, skill_id, domain_key)
        code_path_str = None
        if code_path:
            normalized_code_path = code_path if code_path.is_absolute() else root / code_path
            try:
                code_path_str = normalized_code_path.relative_to(root).as_posix()
            except ValueError:
                code_path_str = code_path.as_posix()

        algo_text = sections.get("algorithm", "")
        scenario_text = sections.get("scenario", "")
        value_text = sections.get("value", "")

        roi_figure, difficulty, priority = extract_roi(value_text)
        if not roi_figure:
            roi_figure = extract_roi_from_scenario(scenario_text)

        skill = PlaybookSkill(
            skill_id=skill_id,
            title=extract_title(fm, body, skill_id),
            domain_key=domain_key,
            domain_dir=domain_dir,
            path=rel_path,
            algorithm_summary=first_nonempty_line(algo_text, first_nonempty_line(body, skill_id)),
            problem_solved=_clean_problem_solved(
                first_bold_sentence(algo_text, first_nonempty_line(scenario_text, "")),
                skill_id,
            ),
            business_scenarios=extract_list_snippets(scenario_text, 8),
            scenario_paragraphs=extract_scenario_paragraphs(scenario_text, 3),
            inputs=extract_list_snippets(
                re.sub(r"输出.*", "", sections.get("guide", ""), flags=re.DOTALL), 5
            ),
            outputs=extract_list_snippets(sections.get("guide", ""), 5),
            roi=extract_list_snippets(value_text, 6),
            roi_figure=roi_figure,
            difficulty=difficulty,
            priority=priority,
            papers=extract_papers(text),
            code_path=code_path_str,
            code_blocks=len(re.findall(r"```(?:python)?", text)) // 2,
            code_preview=_extract_first_code_block(sections.get("code", "")),
            relations=relations,
            tags=classify(full_text_for_classify, ALGO_TAG_RULES),
            topics=classify(full_text_for_classify, TOPIC_RULES),
            workflows=classify(full_text_for_classify, WORKFLOW_RULES),
        )
        if skill_id in SKILL_PS_OVERRIDE:
            skill.problem_solved = _clean_problem_solved(SKILL_PS_OVERRIDE[skill_id], skill_id)
        if skill.problem_solved and skill.algorithm_summary and \
                skill.problem_solved[:60] == skill.algorithm_summary[:60]:
            import sys
            print(f"WARN dup_ps: {skill_id} — problem_solved==algorithm_summary", file=sys.stderr)
        biz = SKILL_BIZ_CONTEXT_OVERRIDE.get(skill_id) or DOMAIN_BUSINESS_CONTEXT.get(domain_dir, {})
        skill.biz_role     = biz.get("role", "")
        skill.biz_role2    = biz.get("role2", "")
        skill.biz_trigger  = biz.get("trigger", "")
        skill.biz_outcome  = biz.get("outcome", "")
        skill.biz_pain     = biz.get("pain", "")
        skill.biz_platform = biz.get("platform", "")
        if not skill.tags:
            skill.tags = [domain_key]
        if not skill.topics:
            skill.topics = ["其他"]
        skills.append(skill)
    return sorted(skills, key=lambda item: (item.domain_dir, item.skill_id))


# ---------------------------------------------------------------------------
# Workflow YAML loader (Phase 2B)
# ---------------------------------------------------------------------------

def load_workflow_defs(root: Path) -> dict[str, Any]:
    """
    Load structured workflow YAML definitions from
    paper2skills-skills/paper-workflow/definitions/*.yaml
    Returns dict keyed by workflow id (e.g. "wf-b").
    Falls back to empty dict if PyYAML unavailable or no files found.
    """
    wf_dir = root / "paper2skills-skills" / "paper-workflow" / "definitions"
    if not wf_dir.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    defs: dict[str, Any] = {}
    for yf in sorted(wf_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yf.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "id" in data:
                defs[data["id"]] = data
        except Exception as e:
            import sys
            print(f"WARN: failed to load workflow YAML {yf.name}: {e}", file=sys.stderr)
    return defs


def render_workflow_step(step: dict[str, Any], skill_lookup: dict[str, "PlaybookSkill"], depth: int = 0) -> str:
    """Recursively render a workflow step as nested HTML decision tree."""
    parts: list[str] = []
    indent = "  " * depth
    name = html.escape(step.get("name", ""))
    question = html.escape(step.get("question", ""))
    context = html.escape(step.get("context", ""))

    parts.append(f'<div class="wf-step" style="--depth:{depth}">')
    if name:
        parts.append(f'  <div class="wf-step-name">{name}</div>')
    if question:
        parts.append(f'  <div class="wf-question">{question}</div>')
    if context:
        parts.append(f'  <div class="wf-context">{context}</div>')

    branches = step.get("branches", [])
    if branches:
        parts.append('  <div class="wf-branches">')
        for branch in branches:
            cond = html.escape(branch.get("condition", ""))
            parts.append(f'    <details class="wf-branch" open>')
            parts.append(f'      <summary class="wf-condition">{cond}</summary>')
            branch_skills = branch.get("skills", [])
            if branch_skills:
                parts.append('      <div class="wf-branch-skills">')
                for bs in branch_skills:
                    sid = bs.get("id", "")
                    role = html.escape(bs.get("role", ""))
                    sk = skill_lookup.get(sid)
                    if sk:
                        parts.append(
                            f'        <a class="wf-skill-chip" href="../skills/{html.escape(sid)}.html">'
                            f'<span class="chip-name">{html.escape(sk.title)}</span>'
                            f'<span class="chip-role">{role}</span></a>'
                        )
                    else:
                        parts.append(
                            f'        <span class="wf-skill-chip missing">'
                            f'<span class="chip-name">{html.escape(sid)}</span>'
                            f'<span class="chip-role">{role}</span></span>'
                        )
                parts.append('      </div>')
            parts.append('    </details>')
        parts.append('  </div>')

    # Inline (non-branching) skills
    inline_skills = step.get("skills", [])
    if inline_skills and not branches:
        parts.append('  <div class="wf-branch-skills">')
        for bs in inline_skills:
            sid = bs.get("id", "")
            role = html.escape(bs.get("role", ""))
            sk = skill_lookup.get(sid)
            if sk:
                parts.append(
                    f'    <a class="wf-skill-chip" href="../skills/{html.escape(sid)}.html">'
                    f'<span class="chip-name">{html.escape(sk.title)}</span>'
                    f'<span class="chip-role">{role}</span></a>'
                )
        parts.append('  </div>')

    parts.append('</div>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# toB Scene Playbooks (Phase F)
# ---------------------------------------------------------------------------

TOB_PLAYBOOKS: list[dict[str, Any]] = [
    {
        "id": "pb-tiktok-shop",
        "icon": "RA",
        "name": "TikTok Shop 运营决策手册",
        "tag": "广告 · 内容 · 归因",
        "desc": "从内容归因到智能竞价的 TikTok Shop 全链路数据决策指南",
        "intro": "TikTok Shop 运营面临三大核心挑战：内容归因不准、竞价策略不清、冷启动缺数据。今天大多数团队的做法是：靠经验调出价、靠感觉判断哪条视频有效、大促前拍脑袋备货——这三个「靠感觉」每年合计造成预算错配 10-20 万元、断货损失 5-10 万元，以及无法复制的投放经验。本手册提供从内容归因到智能竞价的 TikTok Shop 全链路数据决策指南。",
        "steps": [
            {
                "step": "Step 1 — 内容归因（上线第 1 周）",
                "problem": "视频带货归因不准：直播、短视频、商品卡三类流量的真实增量贡献各是多少？",
                "skills": [
                    {"id": "Skill-Causal-Attribution-Bridge", "why": "将 naive 归因（点击→购买）替换为因果 ITE，识别「无论有没有这条视频都会买」的用户"},
                    {"id": "Skill-TikTok-Shop-Content-Attribution", "why": "TikTok 场域专用的内容-转化路径建模"},
                ],
                "data": "需要：用户曝光日志（impression）、点击日志、订单数据，至少 7 天",
                "output": "每条内容的真实增量 ITE 值，用于下周预算分配",
            },
            {
                "step": "Step 2 — 受众精准化（第 2-4 周）",
                "problem": "如何找到「这个视频对谁最有效」，而不是全量推送？",
                "skills": [
                    {"id": "Skill-Uplift-Modeling", "why": "识别可说服者（Persuadables），避免对必然购买者和无法说服者浪费预算"},
                    {"id": "Skill-Guardrailed-Uplift-Targeting", "why": "加入预算护栏约束，自动输出最优干预名单"},
                ],
                "data": "需要：A/B 实验数据（发/未发内容对照组）或历史随机分组数据，≥5000 用户",
                "output": "按 CATE 排序的用户分群，精准定向投放",
            },
            {
                "step": "Step 3 — 智能竞价（第 4 周起）",
                "problem": "如何在有限预算下最大化 ROAS，同时避免渠道过度饱和？",
                "skills": [
                    {"id": "Skill-LLM-AutoBidding-MAS", "why": "LLM 驱动的层次化竞价，支持多目标（GMV、ROAS、品牌曝光）动态平衡"},
                    {"id": "Skill-Channel-Saturation-Curve", "why": "识别投放饱和拐点，防止边际 ROAS 持续下滑"},
                    {"id": "Skill-Creative-Fatigue-Detection", "why": "监控创意疲劳，自动触发素材轮换"},
                ],
                "data": "需要：每日投放日志、ROAS 数据，实时流可选",
                "output": "自动竞价策略 + 创意轮换提醒",
            },
        ],
        "outcomes": ["内容归因准确率 +30%，预算错配减少", "可说服用户定向精准率 +25%", "竞价 ROAS 提升 15-25%"],
    },
    {
        "id": "pb-inventory-festival",
        "icon": "SC",
        "name": "大促备货决策手册",
        "tag": "供应链 · 补货 · 预测",
        "desc": "双十一 / Prime Day / Black Friday 前 8 周的库存决策完整路线图",
        "intro": "大促备货最大的风险：备多了积压，备少了断货。今天大多数团队的备货方式是：拿历史销量乘以经验系数——这个方法在大促销速是平日 3-10 倍的情况下，误差率高达 40%。两种错误的年化损失都可能超过 300 万。本手册提供从 T-8 周到大促当日的分阶段决策节点。",
        "steps": [
            {
                "step": "T-8 周 — 需求预测基线",
                "problem": "大促期间需求是平日的 3-10 倍，历史数据如何泛化到极端场景？",
                "skills": [
                    {"id": "Skill-Demand-Forecasting-Supply-Chain", "why": "结合促销日历、竞品价格、渠道库存的供应链专用预测"},
                    {"id": "Skill-Promotion-Demand-Decomposition", "why": "将大促需求拆解为：基础需求 + 促销增量 + 渠道转移"},
                    {"id": "Skill-Conformal-Prediction-Demand-UQ", "why": "输出置信区间（P10/P50/P90），为保守/激进备货方案提供概率支撑"},
                ],
                "data": "需要：近 2 年同类大促历史销售、促销方案草案、竞品去年大促价格",
                "output": "SKU 级需求预测 + 90% 置信区间",
            },
            {
                "step": "T-4 周 — 补货策略锁定",
                "problem": "如何在 MOQ 约束和海外仓容积限制下，制定最优补货计划？",
                "skills": [
                    {"id": "Skill-Safety-Stock-Replenishment", "why": "动态安全库存计算，考虑供应商交期波动"},
                    {"id": "Skill-Dynamic-Lot-Sizing-MOQ", "why": "最小起订量约束下的动态批量优化，避免超额采购"},
                    {"id": "Skill-Multi-Channel-Inventory-Pooling", "why": "跨 Amazon/独立站/TikTok 动态调拨，防止 A 仓过剩 B 仓断货"},
                ],
                "data": "需要：供应商 lead time 分布、各仓容积上限、各渠道历史分单比例",
                "output": "分渠道补货计划 + 调拨策略",
            },
            {
                "step": "大促实时 — 库存健康监控",
                "problem": "大促期间如何实时发现异常并快速响应？",
                "skills": [
                    {"id": "Skill-Inventory-Health-Aging-Attribution", "why": "实时 FSN 分级监控，识别哪些 SKU 正在快速耗尽"},
                    {"id": "Skill-Argos-Agentic-Anomaly-Detection", "why": "Agentic 异常检测，自动预警库存骤降"},
                ],
                "data": "需要：实时库存快照（每小时）、销售流水",
                "output": "实时库存预警 + 自动调拨建议",
            },
        ],
        "outcomes": ["需求预测 MAPE 降低 20-35%", "库存积压减少 15-25%", "断货率从 8% → 3%", "年化节省 50-200 万元"],
    },
    {
        "id": "pb-new-product-launch",
        "icon": "NP",
        "name": "新品冷启动手册",
        "tag": "选品 · 预测 · 增长",
        "desc": "零历史数据下的新品上市全链路决策：从选品验证到首批备货到冷启动推广",
        "intro": "母婴跨境每年 20-30 款新品上市，前 8 周零销售记录——你对首批该备多少货完全没有数据依据。今天大多数团队的做法是：参考相似品的月销，乘以一个「保守系数」。结果是：30% 的新品因备货不足在 launch 窗口断货，50% 的新品因备货过多造成长库龄积压，两种错误的总损失年化超过 300 万元。本手册用 Bass 扩散模型 + 相似品迁移学习，让首批备货有数据背书。"
                 "本手册用数据替代直觉，从选品验证开始就建立可量化的决策依据。",
        "steps": [
            {
                "step": "上市前 12 周 — 选品验证",
                "problem": "这个品在目标市场有多大空间？竞争格局如何？",
                "skills": [
                    {"id": "Skill-Market-Size-Estimation", "why": "TAM/SAM/SOM 双路径估算 + Monte Carlo 置信区间"},
                    {"id": "Skill-Cross-Market-Product-Transfer", "why": "预测国内爆品在海外的适配性，避免负迁移"},
                    {"id": "Skill-Category-Compliance-Prescan", "why": "目标市场合规预扫描（FDA/CE），提前发现上市障碍"},
                    {"id": "Skill-AutoPKG-Multimodal-Product-Attribute-KG", "why": "自动扫描竞品 Listing 的属性图谱：发现竞品已标注但你的 SKU 缺失的属性字段，在新品上架前补齐属性差距（Search GMV+5.32% 参考基准）"},
                ],
                "data": "需要：类似品近 1 年 Amazon BSR、Google Trends、同类竞品价格带、竞品主图",
                "output": "市场空间评分 + GO/NO-GO 建议 + 竞品属性差距矩阵",
            },
            {
                "step": "上市前 8 周 — 首批备货预测",
                "problem": "没有历史数据，首批备多少货？",
                "skills": [
                    {"id": "Skill-Bass-Diffusion-New-Product-Forecasting", "why": "Bass 扩散模型 + 相似品参数迁移，输出 8 周扩散曲线"},
                    {"id": "Skill-Cold-Start-Product-Recommendation", "why": "LLM 模拟用户行为，预测冷启动期的需求信号"},
                ],
                "data": "需要：3 个以上相似品的历史销售曲线、产品定价方案",
                "output": "首批备货建议量（P25/P50/P75 三档）",
            },
            {
                "step": "上市后 1-4 周 — 冷启动加速",
                "problem": "如何在数据稀疏期快速学习并调整投放策略？",
                "skills": [
                    {"id": "Skill-BCCB-Causal-Bandits", "why": "预算约束因果 Bandit，Day 1 就开始在线学习，无需等待历史数据"},
                    {"id": "Skill-Thompson-Sampling-MAB", "why": "MAB 动态分配流量，快速识别高转化渠道"},
                    {"id": "Skill-Category-Trend-Forecasting", "why": "实时监测品类趋势，判断新品是否踩上上升风口"},
                ],
                "data": "需要：实时用户行为流（曝光→点击→加购→转化）",
                "output": "最优渠道分配 + 动态竞价策略",
            },
        ],
        "outcomes": ["首批备货准确率提升，积压/断货损失减少 60%", "冷启动学习周期从 4 周压缩至 1 周", "选品 GO/NO-GO 决策有数据支撑"],
    },
    {
        "id": "pb-user-growth",
        "icon": "UG",
        "name": "用户增长决策手册",
        "tag": "LTV · 流失 · 分层运营",
        "desc": "从用户价值分层到精准干预的全链路用户增长决策指南",
        "intro": "母婴跨境用户运营的核心矛盾：发券 80% 的预算打给了「本来就会复购」的用户，真正流失的那批人没被识别出来。今天大多数团队的做法是：按 RFM 分层发统一优惠券，或者对全体用户做 EDM 触达。这种「广撒网」的方式造成每月 15-25 万元的无效促销成本，且老客复购率始终上不去。本手册提供从用户分层到精准干预的端到端增长路径，让每一分促销预算花在真正值得干预的用户身上。",
        "steps": [
            {
                "step": "Step 1 — 用户价值分层",
                "problem": "谁是你的高价值用户？谁即将流失？谁从未真正激活？",
                "skills": [
                    {"id": "Skill-RFM-Customer-Segmentation", "why": "用 R/F/M 三维把用户分成 8 类（冠军→流失→沉睡），每类对应不同运营策略"},
                    {"id": "Skill-LTV-Prediction-ZILN", "why": "零膨胀对数正态模型预测用户生命周期价值，识别潜力高价值用户"},
                    {"id": "Skill-User-Lifecycle-STAN", "why": "时空注意力网络建模用户生命周期阶段（新客→成熟→衰退），判断当前阶段"},
                ],
                "data": "需要：用户历史订单（近 1 年）、注册时间、品类购买记录",
                "output": "用户价值分层标签 + 各层用户规模与贡献占比",
            },
            {
                "step": "Step 2 — 流失预警与精准干预",
                "problem": "哪些用户会流失？发券对谁真正有效？",
                "skills": [
                    {"id": "Skill-Uplift-Churn-Prediction", "why": "Uplift 流失预测：识别「可说服者」，避免对必然流失和必然留存的用户浪费资源"},
                    {"id": "Skill-Guardrailed-CATE-NBA", "why": "带预算护栏的最优行动决策：在预算约束下输出最优干预名单"},
                    {"id": "Skill-Customer-Churn-Prediction", "why": "深度学习流失预测，捕捉复杂的行为序列模式"},
                ],
                "data": "需要：历史 A/B 干预数据（发/未发券对照）或 RCT 数据，≥ 5000 用户",
                "output": "可干预用户名单（按 CATE 排序）+ 干预方式推荐（券面值/文案/渠道）",
            },
            {
                "step": "Step 3 — 复购周期与 LTV 提升",
                "problem": "如何在正确的时机触达用户，提升复购率？",
                "skills": [
                    {"id": "Skill-Cohort-Retention-Analysis", "why": "队列留存分析，找出复购的关键时间窗口（如首单后 14 天是黄金窗口）"},
                    {"id": "Skill-Long-Term-Preference-Memory", "why": "长期偏好记忆模型，捕捉用户跨品类的兴趣演变"},
                    {"id": "Skill-User-Profile-Long-Memory", "why": "用户长记忆画像，支持个性化触达内容生成"},
                ],
                "data": "需要：用户行为序列（浏览/加购/购买）、触达记录与响应结果",
                "output": "个人化复购提醒时机 + 触达内容模板",
            },
            {
                "step": "Step 4 — 新用户冷启动推荐",
                "problem": "新用户没有历史数据，如何实现个性化推荐？",
                "skills": [
                    {"id": "Skill-Cold-Start-Product-Recommendation", "why": "LLM 模拟新用户行为，生成合成交互数据填补冷启动空白"},
                    {"id": "Skill-Cold-Start-Meta-Learning-PAM", "why": "元学习框架，用少量交互数据快速适配新用户偏好"},
                ],
                "data": "需要：注册信息、首次浏览 session 行为（即使只有 3-5 次点击）",
                "output": "新用户首屏个性化推荐列表",
            },
        ],
        "outcomes": [
            "可说服用户识别精准率 +25%，优惠券 ROI 提升 3-5x",
            "复购率提升 10-15%，LTV 增加",
            "新用户首单转化率提升 15-20%",
            "年化节省无效促销成本 25-50 万元",
        ],
    },
    {
        "id": "pb-data-foundation",
        "icon": "DB",
        "name": "数据治理基础手册",
        "tag": "数据质量 · KG · Agent",
        "desc": "中小跨境电商团队从零建立 AI 可用数据基础设施的分阶段路线图",
        "intro": "AI 决策的上限是数据质量。大多数中小团队在部署 AI 时遭遇的「效果差」，根因不是模型不好，而是数据没有准备好。今天大多数团队的数据现状是：Amazon/TikTok/Shopify 三套数据各自为政，手动导出对账每月耗时 2-3 天；SKU 命名不统一导致数据拼不起来；AI 工具喂进去的是脏数据，输出必然是垃圾。本手册提供从数据采集、清洗到知识图谱的分阶段建设路线，让 AI 有干净数据可用。",
        "steps": [
            {
                "step": "Step 1 — 数据采集与质量基线",
                "problem": "数据从哪来？质量怎么保证？",
                "skills": [
                    {"id": "Skill-Data-Collection-Agent-Pipeline", "why": "Agent 驱动的自动化数据采集流水线，覆盖 Amazon/社媒/竞品多源"},
                    {"id": "Skill-Ecommerce-Data-Quality-Assessment", "why": "电商数据质量综合评估，建立数据质量基线（完整性/准确性/时效性）"},
                    {"id": "Skill-Data-Provenance-Lineage", "why": "数据血缘追踪，知道每条数据从哪来、经过什么处理"},
                ],
                "data": "需要：现有数据源清单（ERP/平台API/手工Excel）",
                "output": "数据源地图 + 质量评分报告 + 优先修复清单",
            },
            {
                "step": "Step 2 — 数据清洗与治理",
                "problem": "如何系统性消除脏数据、孤岛数据、重复数据？",
                "skills": [
                    {"id": "Skill-Review-Dedup-Quality-Filter", "why": "评论/工单去重与质量过滤，建立标准化文本数据集"},
                    {"id": "Skill-Entity-Resolution-KG-Dedup", "why": "跨系统实体解析去重（同一 SKU 在不同系统有不同编码）"},
                    {"id": "Skill-Data-Drift-Detection", "why": "数据漂移检测，发现数据分布变化（如季节性导致的分布偏移）"},
                ],
                "data": "需要：各系统导出的原始数据文件",
                "output": "清洗后的标准化数据集 + 实体映射表",
            },
            {
                "step": "Step 3 — 产品知识图谱构建",
                "problem": "如何把产品目录、用户行为、竞品关系结构化为 AI 可查询的知识库？",
                "skills": [
                    {"id": "Skill-Ontology-Schema-Design", "why": "母婴电商本体设计（品牌→系列→产品→成分→适用年龄），是 KG 的地图"},
                    {"id": "Skill-Hierarchical-Product-KG-Construction", "why": "层次化产品知识图谱自动构建"},
                    {"id": "Skill-KG-Incremental-Update", "why": "图谱增量更新，新品上架/下架自动同步，不用每次全量重建"},
                ],
                "data": "需要：产品目录（SKU/类目/属性）、历史订单中的同购关系",
                "output": "可查询的产品知识图谱（支持 KGQA + GraphRAG）",
            },
            {
                "step": "Step 4 — AI 可用数据接口",
                "problem": "如何让 AI Agent 能直接查询业务数据，而不依赖数据团队提数？",
                "skills": [
                    {"id": "Skill-SQL-Agent-Text-to-SQL", "why": "自然语言转 SQL，业务同学直接用中文查数据库"},
                    {"id": "Skill-NL2Dashboard-Automation", "why": "自然语言→仪表盘自动生成，运营自助分析无需等排期"},
                    {"id": "Skill-RAG-Enhanced-Data-Analysis", "why": "RAG 增强数据分析，结合知识库回答「为什么」类问题"},
                ],
                "data": "需要：已完成 Step 1-3 的数据基础设施",
                "output": "可供 AI Agent 调用的数据查询接口 + 自助分析工具",
            },
        ],
        "outcomes": [
            "数据质量分从基线提升，AI 模型效果直接受益",
            "运营自助分析比例从 20% → 80%，数据团队提数工作量 -60%",
            "知识图谱建成后 KGQA 查询召回率 > 90%",
            "新品上架数据同步从 2 天 → 实时",
        ],
    },
    {
        "id": "pb-agent-replace",
        "icon": "AI",
        "name": "AI Agent 替人手册",
        "tag": "供应链 · 数据分析 · 广告优化",
        "desc": "三类岗位的核心重复性工作，逐步交给 AI Agent 执行，释放人力专注高价值决策",
        "intro": "AI 替代的不是岗位，是岗位里的「重复性决策」——每天都要做、做法固定、但规模超出人力上限的那部分。今天的现状是：供应链负责人每月花 2-3 天纯人工对账三层 Excel（SKU→仓库→市场），数据分析师 80% 时间在取数而非分析，客服团队因无法覆盖德语/日语工单导致旺季 Buy Box 丢失 35%。这三条业务线加起来消耗了 5-8 名运营人员的大量精力，但仍然有盲区、有滞后、有错误。本手册从供应链出发，提供三条独立的替代路径，每条路径都可以单独落地，合计年化 ROI 可达 2000 万+。",
        "steps": [
            {
                "step": "Chapter 1：供应链全链路 Agent（主角）",
                "problem": "15 个 Excel 联动地狱：SKU×仓库×市场三层预测加总不一致，月底 2-3 天纯人工对账；大促定价和补货是两个团队各自决策，无法实时联动",
                "skills": [
                    {"id": "Skill-Hierarchical-Demand-Forecasting-Reconciliation", "why": "分层预测调和，数学保证 SKU→仓库→市场三层一致，月底对账人力清零（ROI 800-1500万）"},
                    {"id": "Skill-Event-Driven-Demand-MAS", "why": "事件感知补货 MAS：大促信号触发自动备货，告别「排期跑批已来不及」（ROI 5000万）"},
                    {"id": "Skill-FSDA-DRL", "why": "快慢双 Agent 联动定价与补货：大促全周期利润最优，替代供应链+运营协调会（ROI 5000万）"},
                    {"id": "Skill-Lead-Time-Distribution-Risk-GenQOT", "why": "海运延误动态安全库存：苏伊士/巴拿马运河异常时自动预警并提前采购（ROI 200-500万）"},
                    {"id": "Skill-Multi-Channel-Inventory-Pooling", "why": "跨渠道动态调拨：Amazon缺货+独立站积压同时存在时自动平衡（ROI 200-400万）"},
                ],
                "data": "需要：60天+销售历史、多仓库存快照、大促日历、供应商交货记录",
                "output": "自动化补货计划 + 大促定价策略 + 海运风险预警 + 跨渠道调拨指令",
            },
            {
                "step": "Chapter 2：数据分析师 Agent",
                "problem": "提数需求排期 2-3 天，分析师 80% 时间在取数不在分析；GMV 异常时人工排查根因需要 1-2 小时，节假日无人监控",
                "skills": [
                    {"id": "Skill-SQL-Agent-Text-to-SQL", "why": "自然语言→SQL：业务同学直接用中文查数据库，消灭 BI 提数排期"},
                    {"id": "Skill-NL2Dashboard-Automation", "why": "自然语言→仪表盘自动生成，运营自助分析，节省 BI 开发人力"},
                    {"id": "Skill-DeepAnalyze-Autonomous-Data-Science-Agent", "why": "自主数据科学 Agent：多平台数据周报 4-6 小时人工 → 5 分钟 Agent 自动生成（ROI 20万）"},
                    {"id": "Skill-ProRCA-Business-Analysis", "why": "因果图根因分析：GMV 暴跌 1-2 小时人工排查 → 0.5 秒锁定根因路径（ROI 100万）"},
                    {"id": "Skill-CAR-Agent-Causal-Shapley", "why": "Agent 步骤因果 Shapley 归因：当 Agent 系统决策失效时，通过结构因果模型精确定位哪步贡献最大，MTTR 从 3 天降至 0.5 天（ROI 50-150万）"},
                ],
                "data": "需要：数据仓库连接权限、历史报表模板、异常阈值定义、Agent 执行轨迹日志",
                "output": "自助查询接口 + 自动周报 + 实时异常根因诊断 + Agent 决策因果归因报告",
            },
            {
                "step": "Chapter 3：广告优化师 Agent",
                "problem": "人工出价精度上限是工作时长：凌晨 2 点平台出新流量包，人在睡觉；新品冷启动 3-5 周无数据，靠经验猜测出价",
                "skills": [
                    {"id": "Skill-Negative-Keyword-Safe-Guard", "why": "贝叶斯小样本负关键词过滤：无关消耗从 18% → 3.2%，自动替代人工整词/否词"},
                    {"id": "Skill-Creative-Fatigue-Detection", "why": "广告素材疲劳检测：CTR/CVR 持续衰减时自动触发素材更新信号，替代人工监控"},
                    {"id": "Skill-DARA-Agentic-MMM-Optimizer", "why": "LLM+RL 双阶段广告预算分配：自动日预算分配，冷启动 ROAS 提升 15-30%（ROI 360-720万）"},
                    {"id": "Skill-Identified-Bayesian-MMM", "why": "贝叶斯 MMM 归因：告诉你广告钱真正浪费在哪里，1000万 ROI"},
                ],
                "data": "需要：各平台广告 API 数据、历史出价记录、素材表现数据",
                "output": "自动出价策略 + 素材更换信号 + 预算分配建议 + 归因报告",
            },
        ],
        "outcomes": [
            "供应链人工对账工作量从 2-3 天/月 → 接近零",
            "数据分析提数响应从 2-3 天 → 5 分钟",
            "广告无关消耗从 18% → 3.2%，冷启动 ROAS +15-30%",
            "三类岗位重复性工作 Agent 覆盖率 > 70%，人力聚焦高价值决策",
        ],
        "roi_calculator": {
            "title": "计算你的 ROI",
            "subtitle": "填入你的业务数字，实时看 AI Agent 的年化收益估算",
            "sections": [
                {
                    "id": "sc",
                    "label": "供应链",
                    "color": "#2563eb",
                    "inputs": [
                        {"id": "sku_count",    "label": "管理 SKU 数量",        "unit": "个",    "default": 60,      "min": 1,    "max": 5000,   "step": 10},
                        {"id": "stockout_rate","label": "当前断货率",            "unit": "%",     "default": 8,       "min": 0,    "max": 50,     "step": 0.5},
                        {"id": "monthly_gmv",  "label": "月均 GMV",             "unit": "万元",  "default": 200,     "min": 10,   "max": 50000,  "step": 10},
                        {"id": "overstock_pct","label": "库存积压占总库存比例", "unit": "%",     "default": 20,      "min": 0,    "max": 80,     "step": 1},
                        {"id": "pm_days",      "label": "月底对账人力",         "unit": "人天/月","default": 3,      "min": 0,    "max": 30,     "step": 0.5},
                        {"id": "pm_cost",      "label": "人力日均成本",         "unit": "元/天", "default": 800,     "min": 200,  "max": 5000,   "step": 100},
                    ],
                    "formula": """
                        const stockout_loss = (stockout_rate/100) * monthly_gmv * 12 * 0.4;
                        const overstock_cost = (overstock_pct/100) * monthly_gmv * 2 * 0.15;
                        const labor_save = pm_days * pm_cost * 12 / 10000;
                        const improvement_stockout = stockout_loss * 0.55;
                        const improvement_overstock = overstock_cost * 0.25;
                        return Math.round(improvement_stockout + improvement_overstock + labor_save);
                    """,
                    "items": [
                        {"label": "断货损失减少（断货率 8%→3.5%）", "key": "improvement_stockout"},
                        {"label": "库存积压成本降低（25%）",         "key": "improvement_overstock"},
                        {"label": "对账人力节省（年化）",             "key": "labor_save"},
                    ],
                },
                {
                    "id": "da",
                    "label": "数据分析师",
                    "color": "#7c3aed",
                    "inputs": [
                        {"id": "analyst_count",   "label": "数据分析师人数",      "unit": "人",    "default": 3,    "min": 1,   "max": 50,    "step": 1},
                        {"id": "analyst_salary",  "label": "分析师年均成本",      "unit": "万元/人","default": 40,  "min": 10,  "max": 200,   "step": 5},
                        {"id": "fetch_hours_pct", "label": "取数占工作时间比例",  "unit": "%",     "default": 60,   "min": 10,  "max": 90,    "step": 5},
                        {"id": "anomaly_per_month","label": "月均 GMV 异常事件",  "unit": "次",    "default": 2,    "min": 0,   "max": 50,    "step": 1},
                        {"id": "anomaly_cost",    "label": "平均每次异常损失",    "unit": "万元",  "default": 5,    "min": 1,   "max": 500,   "step": 1},
                    ],
                    "formula": """
                        const labor_save = analyst_count * analyst_salary * (fetch_hours_pct/100) * 0.7;
                        const anomaly_save = anomaly_per_month * anomaly_cost * 12 * 0.6;
                        return Math.round(labor_save + anomaly_save);
                    """,
                    "items": [
                        {"label": "取数人力节省（覆盖 70% 取数工作）", "key": "labor_save"},
                        {"label": "异常响应加速带来的损失减少",         "key": "anomaly_save"},
                    ],
                },
                {
                    "id": "ad",
                    "label": "广告优化师",
                    "color": "#059669",
                    "inputs": [
                        {"id": "monthly_adspend", "label": "月广告投放额",         "unit": "万元",  "default": 50,    "min": 1,   "max": 10000, "step": 5},
                        {"id": "wasted_pct",      "label": "估算无效消耗比例",     "unit": "%",     "default": 18,    "min": 0,   "max": 60,    "step": 1},
                        {"id": "cold_start_sku",  "label": "月均新品冷启动 SKU 数","unit": "个",    "default": 3,     "min": 0,   "max": 100,   "step": 1},
                        {"id": "cold_start_spend","label": "每 SKU 冷启动广告费",  "unit": "万元",  "default": 5,     "min": 0,   "max": 200,   "step": 1},
                    ],
                    "formula": """
                        const keyword_save = monthly_adspend * 12 * ((wasted_pct - 3.2) / 100) * 0.8;
                        const roas_lift = cold_start_sku * cold_start_spend * 12 * 0.20;
                        return Math.round(Math.max(0, keyword_save) + roas_lift);
                    """,
                    "items": [
                        {"label": "负关键词过滤：无关消耗从 " + "wasted_pct" + "% → 3.2%", "key": "keyword_save"},
                        {"label": "冷启动 ROAS 提升 20%（保守估算）",                        "key": "roas_lift"},
                    ],
                },
            ],
        },
    },
    {
        "id": "pb-content-factory",
        "icon": "TC",
        "name": "AI 内容工厂手册",
        "tag": "素材采集 · 视频生成",
        "desc": "从人工内容团队到 AI 批量生产的 4 步迁移路线图，部分能力今日可用，视频生成接入中",
        "intro": "内容工厂的核心矛盾：进入德国、日本市场需要本地化内容，但人工方案每条视频 $2000+、周期 3-4 周、语言门槛高。今天大多数团队的做法是：先只做英语，等「资源够了」再做多语言——结果是永远没有足够资源，非英语市场的转化率长期低 25-35%。本手册提供分步落地路径：Step 1/3/4 今天就能带来 ROI，Step 2（视频生成）接入中，逐步释放内容产能 5-10 倍。",
        "steps": [
            {
                "step": "Step 1 【今日可用】素材智能采集",
                "problem": "内容创作前需要竞品素材、用户 VOC、爆品视频——人工采集耗时且遗漏率高",
                "skills": [
                    {"id": "Skill-Visual-Data-Collection", "why": "电商图文视频批量采集，为 AI 生成构建原材料库（ROI 380万）"},
                    {"id": "Skill-Review-Dedup-Quality-Filter", "why": "多平台评论去重净化，提取高质量用户 VOC 作为内容脚本素材（ROI 10-50万）"},
                ],
                "data": "需要：目标平台账号、竞品 ASIN/URL 列表",
                "output": "结构化素材库（图/视频/UGC 评论）",
            },
            {
                "step": "Step 2 【实验接入中】AI 批量内容生成",
                "problem": "进入德/日/韩市场需要本地化视频，人工方案成本 $2000+/条、周期 3-4 周",
                "skills": [
                    {"id": "Skill-AnchorCrafter-Virtual-Anchor-Demo", "why": "虚拟主播带货视频生成，HOI 交互保持商品真实感（ROI 50-100万）— 实验接入中"},
                    {"id": "Skill-DAWN-Talking-Head-Review", "why": "AI 口播 Review 视频，批量生成不同语言/形象的真人测评风格（ROI 30-60万）— 实验接入中"},
                    {"id": "Skill-Virbo-Multilingual-Avatar-UGC", "why": "多语言虚拟人 UGC 批量生产，100+ 语言 TTS + 对口型（ROI 35-60万）— 实验接入中"},
                    {"id": "Skill-Aquarius-Brand-Video-Generation", "why": "品牌营销视频生成，2B 参数模型，多主题批量（ROI 80-150万）— 实验接入中"},
                ],
                "data": "需要：品牌素材包（Logo/色调/产品图）、脚本模板、目标语言列表",
                "output": "批量本地化营销视频（预计 Q3 完整上线）",
            },
            {
                "step": "Step 3 【今日可用】内容质量评估",
                "problem": "AI 生成视频质量参差不齐，商品保真度和品牌一致性无客观评分标准",
                "skills": [
                    {"id": "Skill-E-Commerce-Video-Benchmark", "why": "电商域专用 Benchmark，PCF/LTP/MN 三维度量化评估，驱动工具选型与质检 SOP"},
                    {"id": "Skill-Creative-Fatigue-Detection", "why": "素材疲劳生命周期监测，生存分析识别 CTR/CVR 衰减，触发素材更新信号"},
                ],
                "data": "需要：已生成视频文件、历史广告表现数据",
                "output": "视频质量评分报告 + 素材更换优先级",
            },
            {
                "step": "Step 4 【今日可用】投放归因闭环",
                "problem": "不知道哪类内容真正带来了转化，内容创作无数据反馈，靠感觉迭代",
                "skills": [
                    {"id": "Skill-TikTok-Shop-Content-Attribution", "why": "TikTok Shop 短视频带货因果归因：识别哪类内容元素效率最高，指导下期制作"},
                    {"id": "Skill-DARA-Agentic-MMM-Optimizer", "why": "内容分发预算自动分配到最优渠道/时段（ROI 360-720万）"},
                ],
                "data": "需要：TikTok 广告数据 API、用户行为序列（曝光→点击→转化）",
                "output": "内容效果归因报告 + 下期制作方向 + 渠道预算自动分配",
            },
        ],
        "outcomes": [
            "Step 1+3+4 今天就能带来 ROI，Step 2 是未来的乘数",
            "本地化视频成本降低 80%，生产周期 3 周 → 3 天（Step 2 接入后）",
            "内容创作从「靠感觉迭代」升级为「数据驱动选题」",
            "素材库规模 10x，多语言市场同步覆盖",
        ],
    },
    {
        "id": "pb-pricing-engine",
        "icon": "PA",
        "name": "AI 定价引擎手册",
        "tag": "竞品监控 · 弹性估算",
        "desc": "定价是乘数，广告是加法——A/B 实测 GMV +13%，定价科学化比多投广告更高效",
        "intro": "大多数跨境品牌的定价策略是「跟感觉」+「盯竞品手动调」。今天的定价现状是：运营每天手动对比竞品价格，遇到竞品降价不知道该不该跟——跟了怕伤利润，不跟怕丢 BSR 排名；大促折扣拍脑袋给，给多了利润归零，给少了没效果。这种「经验驱动定价」的代价是：A/B 实测显示系统化定价可多获 GMV +13%，对应年化 1,321 万元。本手册提供从竞品监控到动态定价的完整飞轮，每步都有可量化 ROI。",
        "steps": [
            {
                "step": "Step 1：竞品价格实时感知",
                "problem": "竞品降价后 47 分钟才响应，Buy Box 已丢失——每 30 分钟延迟损失 GMV 约 $1,600",
                "skills": [
                    {"id": "Skill-Price-Signal-Collection", "why": "竞品价格信号实时采集，Buy Box 获得率 41%→79%，响应延迟从 47 分钟降至 18 分钟（ROI 73.2万）"},
                    {"id": "Skill-Competitive-Price-Monitoring", "why": "因果竞争响应模型：量化「不跟降损失多少」，驱动有依据的响应决策（ROI 5-60万）"},
                ],
                "data": "需要：竞品 ASIN 列表、自身历史定价与销量数据",
                "output": "实时竞品价格监控面板 + 响应建议",
            },
            {
                "step": "Step 2：需求弹性估算",
                "problem": "「这个 SKU 降 $2 能多卖多少件」——没有弹性数据，定价靠猜",
                "skills": [
                    {"id": "Skill-Dynamic-Pricing-Elasticity", "why": "需求价格弹性估计 + 最优价格公式 P*=ε/(ε+1)·MC，利润率 +8-12%（ROI 50万）"},
                    {"id": "Skill-DML-Cohort-Causal-Effect", "why": "DML 分群弹性差异：7-12 月龄用户对价格敏感度是 0-3 月龄的 2.3 倍，精准定价（ROI 1500-2500万）"},
                ],
                "data": "需要：历史价格变动记录（至少 6 次调价）、同期销量数据",
                "output": "各 SKU 价格弹性曲线 + 最优定价区间",
            },
            {
                "step": "Step 3：动态定价执行",
                "problem": "人工定价最大缺陷：只能优化当前销量，无法同时考虑长期品牌溢价和库存健康",
                "skills": [
                    {"id": "Skill-AIGP-LLM-Dynamic-Pricing", "why": "LLM 跨周期 GMV 对齐定价，A/B 实测 GMV +13%，这是真实实验数据不是预测（ROI 1321万）"},
                    {"id": "Skill-Markdown-Optimization", "why": "清仓折扣优化：库存生命周期内最大化总回收价值，清仓多回收 15-40%（ROI 20-50万）"},
                    {"id": "Skill-Bundle-Pricing-Strategy", "why": "捆绑定价提升 AOV：吸奶器+法兰配件组合定价，配件复购率 +25%（ROI 10-15万）"},
                ],
                "data": "需要：SKU 成本结构、库存水位、竞品定价实时数据",
                "output": "动态定价策略 + 每日价格执行建议",
            },
            {
                "step": "Step 4：定价效果度量与迭代",
                "problem": "调价后「效果好不好」靠感觉判断，无法区分是定价作用还是季节波动",
                "skills": [
                    {"id": "Skill-DiD-Difference-in-Differences", "why": "双重差分估计调价因果效应：区分真实影响与自然波动（ROI 50万）"},
                    {"id": "Skill-Identified-Bayesian-MMM", "why": "分离价格效应与广告效应：告诉 CMO 预算浪费在哪里（ROI 1000万）"},
                    {"id": "Skill-Causal-Cohort-Analysis", "why": "促销长期 LTV 追踪：6 个月后用户复购是否真的提升了？（ROI 200-500万）"},
                ],
                "data": "需要：调价前后各 30 天销售数据、广告投放数据",
                "output": "调价效果因果报告 + 下一轮定价优化方向",
            },
        ],
        "outcomes": [
            "Buy Box 获得率从 41% 恢复至 79%，月均 GMV +$32,000",
            "AIGP 动态定价 A/B 实测 GMV +13%（真实实验，非预测）",
            "清仓效率提升，资金回收速度 +15-40%",
            "「定价是乘数，广告是加法」——精准定价 1% 比广告多投 15% 更高效",
        ],
    },
    {
        "id": "pb-risk-defense",
        "icon": "AG",
        "name": "跨境风险防御作战室",
        "tag": "欺诈反制 · 合规预警",
        "desc": "竞品用 AI 攻击你的广告、评分和排名——你需要 AI 来守门。封号预防 800 万/年 vs 防御投入 30 万",
        "roi_callout": [
            {"label": "不防御：一次封号", "value": "30-80 万 GMV 损失"},
            {"label": "防御投入：年度成本", "value": "约 30 万"},
            {"label": "净收益差值", "value": "> 800 万/年 · ROI 26:1"},
        ],
        "intro": "跨境电商的竞争已经进入「AI 对抗」阶段。今天大多数卖家还在用人工方式应对：人工监控评分、人工处理差评、人工核查合规——这三件事加起来每月消耗 2-3 名运营人员的大量精力，但仍然有盲区。竞品已在用 AI 对你发动三条战线：① 广告刷量消耗你的预算；② 虚假差评拉低你的评分；③ AI 注入攻击你的推荐排名。一次封号损失 30-80 万 GMV，一次召回损失 500 万+，但防御投入只需约 30 万——ROI 26:1。本手册提供从欺诈信号采集到合规预警的完整防御体系。",
        "steps": [
            {
                "step": "Step 1：欺诈信号采集与基线建立",
                "problem": "不知道自己正在被攻击——竞品刷评、广告 IVT、刷单行为悄无声息地侵蚀 ROI，发现时已损失数周",
                "skills": [
                    {"id": "Skill-Fraud-Signal-Collection",
                     "why": "主动采集刷单行为、虚假评论、异常流量信号，建立欺诈监控基线，同时向平台举报竞品（ROI 48万）"},
                    {"id": "Skill-Transaction-Anomaly-Detection",
                     "why": "Isolation Forest 检测异常交易模式（订单金额/IP/支付方式异常组合），拦截盗刷订单（ROI 3-8万/月）"},
                ],
                "data": "需要：订单流水、广告点击日志、评论数据（近 90 天）",
                "output": "欺诈信号基线报告 + 异常事件告警规则",
            },
            {
                "step": "Step 2：广告 IVT 实时过滤",
                "problem": "月广告 30 万，8% 是无效点击 = 2.4 万/月浪费。Bot 点击、竞品恶意点击无法靠平台自动过滤完全解决",
                "skills": [
                    {"id": "Skill-Click-Fraud-Detection",
                     "why": "时序异常 + 行为模式识别 IVT 攻击，向 FB/Google 申请退款，月均挽回 6-15 万"},
                    {"id": "Skill-Identity-Fraud-Detection",
                     "why": "设备+行为+网络三重验证账号欺诈，识别刷单账号防止 Amazon 卖家账户关联封号"},
                ],
                "data": "需要：广告平台 API（点击日志、IP、设备指纹）",
                "output": "IVT 过滤规则 + 月度退款申请报告",
            },
            {
                "step": "Step 3：Listing 评论生态保护",
                "problem": "新品上架 6 小时内被刷评团伙集中攻击；或竞品 ChatGPT 批量生成高质量虚假好评拉高自己",
                "skills": [
                    {"id": "Skill-Review-Fraud-Detection",
                     "why": "评论者-产品-评分关系图检测刷评团伙，向 Amazon 举报并保护自身 listing（ROI 5-15万/月）"},
                    {"id": "Skill-DS-DGA-GCN-Fake-Review-Group",
                     "why": "动态图 GCN 检测冷启动新品上架 6 小时内的刷评冲击，防止新品评分被操控"},
                    {"id": "Skill-FraudSquad-LLM-Review-Detection",
                     "why": "LM 嵌入 + 门控图变换器检测 ChatGPT 生成的高质量虚假好评（2025 年最新攻击方式）"},
                    {"id": "Skill-AIGC-Content-Detection",
                     "why": "鉴别 AI 生成内容，保护自身 Review 生态可信度，防止 VOC 分析被污染"},
                ],
                "data": "需要：Listing 评论历史、评论者行为序列",
                "output": "刷评团伙举报名单 + 评论质量评分 + 告警规则",
            },
            {
                "step": "Step 4：AI 推荐排名防御",
                "problem": "竞品在商品描述中嵌入恶意 Prompt 指令，劫持 AI 导购排名，某品牌自营商品曝光量下降 30-50%（2025 年真实攻击）",
                "skills": [
                    {"id": "Skill-Agent-Payment-Security-Red-Team",
                     "why": "检测 Prompt 注入攻击，保护 AI 推荐系统不被竞品操控（防御价值 > 5000万）"},
                    {"id": "Skill-MAS-Adversarial-Defense",
                     "why": "多智能体对抗防御，应对竞品协同攻击（多个假账号+多个被操控 listing 联合操作）"},
                ],
                "data": "需要：竞品 ASIN 列表、自身 AI 搜索排名监控数据",
                "output": "注入攻击检测告警 + 竞品操控行为报告",
            },
            {
                "step": "Step 5：合规预警与封号防御",
                "problem": "平台政策每季度更新，人工合规检查平均滞后 90 天，一次封号损失 30-80 万 GMV + BSR 排名恢复 2-6 周",
                "skills": [
                    {"id": "Skill-Regulatory-Change-Monitoring",
                     "why": "监管机构（FDA/CPSC/EU GPS）法规更新自动映射到受影响 SKU，提前 90 天预警"},
                    {"id": "Skill-Cross-Border-Compliance-Framework",
                     "why": "US+EU+UK 三维合规矩阵自动映射，新市场进入合规核查从 3 个月压缩到 2 周"},
                    {"id": "Skill-Product-Safety-Testing-Requirements",
                     "why": "品类×市场安全测试需求自动生成，选品阶段前置合规成本估算，避免选错品"},
                    {"id": "Skill-Compliance-Scored-Guardrail-Orchestration",
                     "why": "AI 生成 Listing 文案的合规门控（Best-of-N 评分），防止 ChatGPT 写出违规声明被 Amazon 下架"},
                ],
                "data": "需要：全部在售 SKU 信息、目标销售市场列表",
                "output": "合规风险评分矩阵 + 高风险 SKU 预警 + Listing 文案合规检测",
            },
            {
                "step": "Step 6：召回风险预测与供应链尽调",
                "problem": "被动等到消费者集中投诉才发现产品安全问题，主动召回 vs 被动召回成本相差 10 倍",
                "skills": [
                    {"id": "Skill-Consumer-Complaint-Recall-Prediction",
                     "why": "投诉信号驱动的召回风险预测，提前 12 个月预警，主动召回比被动召回节省 80% 成本"},
                    {"id": "Skill-Supply-Chain-Due-Diligence",
                     "why": "供应商劳工+环境+产品三维合规评估（德国供应链法 LkSG2023），满足欧洲 B2B 买家准入要求"},
                ],
                "data": "需要：客服工单历史、产品投诉数据、供应商信息",
                "output": "召回风险评分 + 供应商合规评级报告",
            },
        ],
        "outcomes": [
            "广告 IVT 从 8% 降至 2%，月均挽回 6-15 万",
            "竞品刷评攻击检测率 > 90%，Listing 评分保护",
            "合规检查从滞后 90 天→实时预警，封号风险降低 70%",
            "主动召回比被动召回节省 80% 成本，一次预防 = 500 万保障",
            "AI 推荐注入攻击防御：保护自然流量不被操控",
        ],
    },
    {
        "id": "pb-tariff-response",
        "icon": "TR",
        "name": "关税冲击 72h 响应手册",
        "tag": "关税应对 · 定价重估",
        "desc": "关税涨 10 个点 = 利润腰斩。你有 72 小时决定怎么做——AI 给你完整行动清单",
        "intro": "2025 年跨境电商面临的最大不确定性：关税政策随时变动。今天大多数团队面对关税冲击的响应方式是：开会讨论、人工拉数据、各部门各自决策——这个流程通常需要 3-5 天，而关税政策生效后的前 72 小时是争夺市场份额的黄金窗口。某母婴品牌在 2025 年关税调整中，因响应慢了 2 周，损失了 BSR 前 10 排名并花了 3 个月才恢复。没有 AI 的团队靠开会讨论，有 AI 的团队 72 小时内拿到完整行动清单并已在执行。",
        "steps": [
            {
                "step": "Step 1 【触发后 0-4h】冲击量化",
                "problem": "关税变动后，不知道哪些 SKU 利润率归零、哪些还有空间——凭感觉操作容易误伤优质品",
                "skills": [
                    {"id": "Skill-DML-Cohort-Causal-Effect",
                     "why": "DML 双机器学习分群估计：不同市场/用户群对关税引发的价格变化弹性差异（ROI 1500-2500万）"},
                    {"id": "Skill-Supply-Chain-Causal-SCM-Attribution",
                     "why": "供应链因果 SCM 根因归因：区分「关税直接影响」vs「市场自然波动」，避免把季节性下滑误归因于关税"},
                ],
                "data": "需要：全 SKU 成本结构（含关税比例）、近 6 个月销售数据、竞品价格",
                "output": "SKU 级利润率冲击矩阵（三种关税假设场景）",
            },
            {
                "step": "Step 2 【4-24h】定价响应决策",
                "problem": "吸收多少关税、转嫁多少给消费者——不同 SKU 的价格弹性差异巨大，统一处理必然错误",
                "skills": [
                    {"id": "Skill-Cross-Border-Price-Harmonization",
                     "why": "跨境价格协调：US/EU/JP 三市场同时调价时防止价差过大引发 Amazon 最低价政策违规（ROI 8-15万）"},
                    {"id": "Skill-AIGP-LLM-Dynamic-Pricing",
                     "why": "LLM 跨周期定价优化：考虑关税冲击期间品牌溢价保护 vs 短期销量最大化的权衡（A/B 实测 GMV +13%）"},
                ],
                "data": "需要：各 SKU 价格弹性估算、竞品实时价格监控数据",
                "output": "SKU 级定价调整方案（吸收 / 部分转嫁 / 全转嫁 三档建议）",
            },
            {
                "step": "Step 3 【24-48h】库存与广告决策",
                "problem": "现有库存是清仓套现还是维价等待？广告是暂停还是继续投？两个决策相互影响",
                "skills": [
                    {"id": "Skill-Markdown-Optimization",
                     "why": "库存生命周期清仓定价优化：关税冲击后高速清库存，多回收 15-40%（ROI 20-50万）"},
                    {"id": "Skill-Channel-Saturation-Curve",
                     "why": "渠道饱和曲线：价格波动期广告效率评估，判断是否应暂停广告等价格稳定后再恢复（ROI 18-25万）"},
                ],
                "data": "需要：各 SKU 库存水位、广告当前 ACOS/TACOS、历史清仓速度",
                "output": "库存处置方案（清仓/维价/转渠道）+ 广告预算调整建议",
            },
            {
                "step": "Step 4 【48-72h】供应链转移可行性评估",
                "problem": "「把订单转移到越南工厂」说起来容易，但不知道实际需要多长时间、成本差多少、合规认证能否复用",
                "skills": [
                    {"id": "Skill-Supplier-Capacity-Planning",
                     "why": "供应商产能规划：评估备选工厂（越南/墨西哥/印尼）的实际产能上限和爬坡周期"},
                    {"id": "Skill-Lead-Time-Distribution-Risk-GenQOT",
                     "why": "转移供应商后的交期分布重建：新工厂 lead time 从均值 30 天→不确定分布，动态安全库存防断货（ROI 200-500万/年）"},
                ],
                "data": "需要：备选供应商资质清单、现有认证（CPSC/CE）可复用性评估",
                "output": "供应链转移可行性报告（时间轴 + 成本差 + 合规风险）",
            },
        ],
        "outcomes": [
            "72 小时内输出完整行动清单，替代 3 天团队讨论",
            "定价响应精准化：弹性低的 SKU 维价，弹性高的提前清仓",
            "供应链转移决策有数据支撑：不靠直觉，知道转移到哪里、什么时候、成本差多少",
            "关税冲击期间广告预算不盲目暂停，基于饱和曲线做有依据的调整",
            "一次关税冲击响应提速 = 保护 3-6 个月 BSR 排名稳定",
        ],
    },
    {
        "id": "pb-compliance",
        "icon": "CL",
        "name": "跨境合规全链路手册",
        "tag": "产品合规 · HTS关税",
        "desc": "新品上架前合规预扫描 + HTS 关税编码节税 + Amazon 申诉策略三合一。合规失误一次 = 损失 30-500 万",
        "roi_callout": [
            {"label": "HTS 编码优化节税", "value": "年均 20-200 万元"},
            {"label": "封号损失（不防御）", "value": "30-80 万 GMV/次"},
            {"label": "主动召回 vs 被动召回", "value": "成本差 10 倍"},
        ],
        "intro": "跨境母婴品牌面临三类「隐形合规炸弹」：① 产品认证缺失导致 Amazon 强制下架（一次损失 30-80 万 GMV）；② HTS 关税编码错误导致多缴 7.5-25% 关税（年均损失 20-200 万）；③ Listing 被封号后不知道如何写 POA 申诉（恢复周期从 2 周拖到 2 个月）。这三类风险有一个共同特点：大多数团队在事后才意识到——因为没有系统性的事前排查机制。本手册从选品阶段就建立合规防线，让你在 listing 上线前发现问题，而不是等平台下架通知。",
        "steps": [
            {
                "step": "Step 1：新品上架前合规预扫描（T-4 周）",
                "problem": "不知道新品在目标市场是否需要特定认证——等 listing 上线被 Amazon 要求补文件，已浪费 2-4 周窗口期",
                "skills": [
                    {"id": "Skill-Category-Compliance-Prescan",
                     "why": "上架前自动扫描目标市场合规要求，识别 FDA/CPSC/CE/FCC 认证门控，提前安排实验室测试（节省 2-4 周窗口期）"},
                    {"id": "Skill-Cross-Border-Compliance-Framework",
                     "why": "US + EU + UK 三维合规矩阵自动映射，新市场进入合规核查从 3 个月压缩到 2 周"},
                    {"id": "Skill-Product-Safety-Testing-Requirements",
                     "why": "品类 × 市场安全测试需求自动生成：哪个实验室 + 多少费用 + 多长周期，选品阶段前置合规成本估算"},
                ],
                "data": "需要：产品品类、目标销售市场、产品材质/功能描述",
                "output": "合规门控清单（BLOCKING / MANDATORY / LABELING 三级）+ 认证机构推荐 + 时间轴估算",
            },
            {
                "step": "Step 2：CPSC 儿童产品强制认证（美国市场）",
                "problem": "母婴产品需要 CPSC 强制第三方认证（3PTC），不通过不得在美销售——但大多数卖家不清楚具体哪些产品需要哪些认证",
                "skills": [
                    {"id": "Skill-CPSC-Children-Product-Safety",
                     "why": "ASTM F963/16 CFR 法规映射 + 认证实验室推荐（SGS/BV/Intertek/UL）+ CPC 证书申请流程（ROI：保护 30-80 万 GMV）"},
                ],
                "data": "需要：产品品类（玩具/婴儿床/服装/汽车座椅等）、是否含电子组件",
                "output": "CPC/GCC 证书需求确认 + 实验室对接清单 + 预算估算（$500-3000 + 3-6 周周期）",
            },
            {
                "step": "Step 3：HTS 关税编码精准分类（节税合规双赢）",
                "problem": "同一件产品因 HTS 编码不同，税率可能相差 0%-25%——大多数卖家沿用报关行默认编码，每年多缴数十万关税",
                "skills": [
                    {"id": "Skill-HTS-Tariff-Classification",
                     "why": "AI 驱动的 HTS 精准分类：识别节税机会（吸奶器 0% vs 3%，睡袋 0% vs 17.5%）+ Section 301 排除申请指导 + CBP Binding Ruling 路径"},
                ],
                "data": "需要：全 SKU 产品描述、材质、功能、原产地",
                "output": "SKU 级 HTS 编码建议 + 关税率对比 + 年化节税估算 + 行动优先级排序",
            },
            {
                "step": "Step 4：Listing 文案合规审查（AI 写稿防违规声明）",
                "problem": "ChatGPT 生成的 Listing 文案容易出现违规声明（如「治疗/预防疾病」类措辞），被 Amazon 算法扫描后下架",
                "skills": [
                    {"id": "Skill-Compliance-Scored-Guardrail-Orchestration",
                     "why": "AI 生成 Listing 文案的合规门控（Best-of-N 评分）：自动检测违规声明、医疗类措辞、夸大功效表述，防止 ChatGPT 写出被 Amazon 下架的内容"},
                ],
                "data": "需要：产品 Listing 文案草稿（标题/Bullet/Description）",
                "output": "合规评分矩阵 + 违规措辞定位 + 合规替换建议",
            },
            {
                "step": "Step 5：法规动态监控（合规状态持续维护）",
                "problem": "平台政策每季度更新，FDA/CPSC/EU GPS 法规每年都有修订——人工跟踪平均滞后 90 天，一次漏网 = 批量下架",
                "skills": [
                    {"id": "Skill-Regulatory-Change-Monitoring",
                     "why": "监管机构法规更新自动映射到受影响 SKU，提前 90 天预警，Amazon 政策变更 24h 内推送相关 listing 影响评估"},
                ],
                "data": "需要：全部在售 SKU 信息、当前认证文件状态",
                "output": "法规变更预警 + 受影响 SKU 清单 + 更新行动计划",
            },
            {
                "step": "Step 6：Amazon 账号申诉策略（封号后快速恢复）",
                "problem": "Listing 或账号被封后，多数卖家写的 POA 申诉成功率只有 20-30%——不是问题没解决，而是 POA 写法不对",
                "skills": [
                    {"id": "Skill-Amazon-Account-Appeal-Strategy",
                     "why": "POA 三段式结构（根因/纠正/预防）+ 按封号类型的差异化策略（ODR/ASIN 违规/知识产权/Review 操纵）+ 升级路径（案例 ID → Executive Relations）"},
                ],
                "data": "需要：Amazon 封号通知邮件、受影响 ASIN、封号日期",
                "output": "结构化 POA 草稿 + 证明文件清单 + 申诉提交路径 + 预计恢复时间",
            },
        ],
        "outcomes": [
            "新品上架前合规扫描：0 因合规被强制下架（vs 平均每季度 1-2 次被动处理）",
            "HTS 编码优化：年均节税 20-200 万元（视 SKU 数量和进口额）",
            "合规文案审查：AI 生成内容违规率从 15% → < 2%",
            "法规变更响应从滞后 90 天 → 提前 90 天预警",
            "POA 申诉成功率从 20-30%（模板）→ 65-80%（结构化策略）",
            "封号恢复周期从 2-4 周 → 3-7 天",
        ],
    },
    {
        "id": "pb-voc-product-loop",
        "icon": "VP",
        "name": "竞品情报→产品迭代加速器",
        "tag": "VOC挖掘 · 痛点归因",
        "desc": "新品从洞察到上架 18 个月 → 6 个月。竞品差评是你最好的免费 R&D 数据",
        "intro": "你花 18 个月开发的新品，竞品 3 个月前就上了类似款，价格还低 30%。今天大多数团队的选品情报方式是：运营人工浏览竞品页面、看销量排名、偶尔读几条评论——每月最多处理 3-5 个竞品、几十条评论，完全跟不上市场变化速度。而竞品的 1-3 星差评里藏着最高密度的产品洞察，用户在告诉你「市场缺什么」。差距不在执行力，在情报速度。本手册让你的情报处理速度提升 10 倍。",
        "steps": [
            {
                "step": "Step 1：竞品差评多语言采集与净化",
                "problem": "手动分析竞品评论：人工处理 1 万条评论需 2-3 周，且只能做英语市场，德语/日语市场的洞察完全缺失",
                "skills": [
                    {"id": "Skill-Review-Pain-Point-Mining",
                     "why": "无监督竞品差评痛点挖掘，自动聚类「漏液」「噪音大」「难清洗」等产品缺陷维度（ROI 50-100万）"},
                    {"id": "Skill-Multilingual-NER-Universal-v2",
                     "why": "22 种语言命名实体识别，从德语/日语评论抽取「品牌/产品/症状」实体，覆盖非英语市场洞察"},
                    {"id": "Skill-Cultural-Data-Collection",
                     "why": "跨文化 UGC 采集：量化「美国妈妈要便利」vs「日本妈妈要安全」的消费偏好差异（ROI 280万）"},
                ],
                "data": "需要：竞品 ASIN 列表（1-3 星差评）、目标市场语言范围",
                "output": "多语言痛点矩阵（按功能维度聚类，按频次排序）",
            },
            {
                "step": "Step 2：痛点归因与产品差距识别",
                "problem": "知道「用户说漏液」还不够——漏液是哪个具体设计特征导致的？竞品有哪些功能是你没有的？",
                "skills": [
                    {"id": "Skill-AGRS-Aspect-Guided-Review-Summarization",
                     "why": "方面引导评论摘要：将「漏液投诉」归因到「密封圈设计/材质/安装方式」具体特征（ROI 1.5万/月）"},
                    {"id": "Skill-LACA-CrossLingual-ABSA",
                     "why": "跨语言方面级情感分析：同一功能在不同市场的情感极性对比，识别市场特有痛点（ROI 300-600万）"},
                ],
                "data": "需要：Step 1 输出的痛点矩阵、产品规格说明书",
                "output": "「功能差距矩阵」：竞品有/你没有的功能列表 + 各市场用户优先级排序",
            },
            {
                "step": "Step 3：洞察转化为产品需求",
                "problem": "从「用户说什么」到「供应商要改什么」之间有巨大鸿沟——数据团队的报告无法直接交给工厂",
                "skills": [
                    {"id": "Skill-StaR-Review-Statement-Ranking",
                     "why": "评论声明重要性排序：找出「最影响购买决策的 5 个改进点」，聚焦有限的产品迭代资源（ROI 80-150万/年）"},
                    {"id": "Skill-MAA-Review-to-Action-Decision",
                     "why": "多 Agent 评论→行动建议：自动生成供应商沟通文档（改良规格 + 测试要求），从洞察直达执行（ROI 510-920万/年）"},
                    {"id": "Skill-AutoPKG-Multimodal-Product-Attribute-KG",
                     "why": "多模态属性图谱自动构建：将评论洞察中提及的属性缺失问题直接映射到产品规格库，自动检测「竞品有但我们没有」的属性字段，Search GMV+5.32%（Lazada A/B实测）"},
                ],
                "data": "需要：Step 2 的功能差距矩阵、现有产品规格、商品主图",
                "output": "可直接交工厂的「产品改良规格书」+ 属性完整度报告 + 优先级排序的改进清单",
            },
            {
                "step": "Step 4：新品上线后效果因果追踪",
                "problem": "改良版新品上架后，不知道销量提升是来自产品改进还是自然市场增长——下次无法复制成功",
                "skills": [
                    {"id": "Skill-DiD-Difference-in-Differences",
                     "why": "双重差分：对比改良品 vs 未改版品在同期的销量变化，量化产品迭代的真实因果效应（ROI 50万）"},
                ],
                "data": "需要：改良品和对照品的销售数据（上架前后各 30 天）",
                "output": "产品迭代 ROI 归因报告 + 下一轮迭代优先级建议",
            },
        ],
        "outcomes": [
            "竞品差评分析从 2-3 周人工 → 实时自动，覆盖 22 种语言",
            "新品开发周期从 18 个月压缩到 6 个月（情报速度提升 3x）",
            "新品成功率从 30% 提升到 50%，年增量 GMV 400万+",
            "产品迭代有数据追踪，每次改动的 ROI 可量化",
            "跨市场文化差异洞察：知道日本和德国要什么，不用靠猜",
        ],
    },
    {
        "id": "pb-customer-service-agent",
        "icon": "CS",
        "name": "客服售后智能体手册",
        "tag": "多语言客服 · 退货优化",
        "desc": "跨境客服三大成本：人力、时效、差评——AI 覆盖 70% 工单，差评率从 3% 降至 1.5%",
        "intro": "跨境客服有三个独特难点：时区、语言、平台惩罚。今天大多数团队的客服现状是：中国团队只能覆盖北京时间工作日，德语/日语差评看不懂无法回复，高峰期工单积压平均响应时间超 48 小时。这直接导致 ODR（订单缺陷率）超标、差评堆积、A-to-Z 索赔风险上升，某团队因此在旺季丢失了 35% 的 Buy Box 份额。本手册提供从工单分流到退货闭环的完整 AI 客服体系，不需要人工值守即可 24h 响应。",
        "steps": [
            {
                "step": "Step 1：工单意图自动分类与路由",
                "problem": "日均 500 条工单，40% 是重复问题（物流时效/使用方法/退货流程）；人工分流每条耗时 3-5 分钟，高峰期积压严重",
                "skills": [
                    {"id": "Skill-DialIn-LLM-Case-Intent-Clustering",
                     "why": "无监督层次化意图聚类：自动发现客服意图树（退款/换货/咨询/投诉），无需人工标注，70% 工单自动路由（ROI 200-400万）"},
                    {"id": "Skill-Customer-Journey-Decision-Tree",
                     "why": "从历史日志自学决策树：70% 标准工单完全自动化处理，客服人力节省，释放人力专注高价值案例（ROI 600万）"},
                ],
                "data": "需要：近 6 个月客服工单历史（含工单文本、处理结果、处理时长）",
                "output": "意图分类体系 + 自动路由规则 + 工单自动回复模板",
            },
            {
                "step": "Step 2：多语言实时响应",
                "problem": "德语/日语差评无人响应，导致 Amazon 账号健康分下降；人工翻译后回复不够专业，文化语气不对",
                "skills": [
                    {"id": "Skill-Multilingual-Customer-Service-Translation",
                     "why": "多语言客服自动翻译与回复生成：覆盖德/日/法/西/葡语，A-to-Z 投诉文化适配回复，响应从 48h 压缩到 2h"},
                    {"id": "Skill-Emotional-AI-Customer-Care",
                     "why": "情感感知客服：高压场景（召回恐慌/宝宝安全问题）识别情绪强度，ANGRY/FRIGHTENED 时自动升级人工，避免 AI 误判激化矛盾"},
                ],
                "data": "需要：目标市场语言列表、历史回复模板、升级阈值配置",
                "output": "多语言 24h 自动回复 + 情绪升级告警",
            },
            {
                "step": "Step 3：差评根因归因与主动干预",
                "problem": "差评出现才处理是被动模式——研究表明 70% 的差评根因是可预防的（说明书不清晰/预期管理失败/物流时效误判）",
                "skills": [
                    {"id": "Skill-AGRS-Aspect-Guided-Review-Summarization",
                     "why": "方面引导评论摘要：将「宝宝用了皮肤发红」聚类到「材质/成分」问题，驱动产品改进而非仅回复差评（ROI 1.5万/月）"},
                    {"id": "Skill-Review-Pain-Point-Mining",
                     "why": "差评痛点挖掘：识别高频重复投诉点（「说明书看不懂」「充电口设计差」），生成预防性改进优先级（ROI 50-100万）"},
                    {"id": "Skill-LACA-CrossLingual-ABSA",
                     "why": "跨语言方面级情感分析：德语/日语差评的情感极性识别，跨市场差评根因对比（ROI 300-600万）"},
                ],
                "data": "需要：全部市场评论数据（含评分、语言、日期）",
                "output": "差评根因矩阵 + 高频问题改进清单 + 主动干预触发规则",
            },
            {
                "step": "Step 4：退货预测与欺诈拦截",
                "problem": "退货率 12%，其中约 35% 的 PayPal/信用卡纠纷（Chargeback）是欺诈性退货（INR 欺诈），人工无法区分",
                "skills": [
                    {"id": "Skill-Returns-Reverse-Logistics",
                     "why": "退货概率预测（XGBoost，按品类/价格/历史退货率）+ 退货处理路径优化（FBA退货 vs 海外仓 vs 销毁），年化 ROI 6-10万"},
                    {"id": "Skill-Logistics-Fraud-Detection",
                     "why": "物流链路欺诈检测：虚假收货/地址篡改/刷单物流识别，INR 欺诈从 35% 降至 5% 以下，月均挽回 $3,200+"},
                ],
                "data": "需要：订单数据、物流轨迹、历史退货记录、支付纠纷记录",
                "output": "退货风险评分 + 欺诈退货拦截规则 + 退货处理路径建议",
            },
        ],
        "outcomes": [
            "70% 标准工单全自动处理，客服人力节省 60%（ROI 600-800万/年）",
            "多语言覆盖：德/日/法/西/葡，响应时效从 48h → 2h",
            "差评率从 3% 降至 1.5%，对应转化率提升约 8-12%",
            "INR 欺诈退货从 35% 降至 5%，月均挽回 $3,200+",
            "差评根因转化为产品改进清单，形成「客服→R&D」反馈闭环",
        ],
    },
    {
        "id": "pb-fba-operations",
        "icon": "SC",
        "name": "FBA 运营全链路手册",
        "tag": "库存健康 · 头程优化",
        "desc": "FBA 仓储成本占 GMV 8-15%，库存周转天数中位数 95 天 vs 行业标杆 60 天——差距就是现金",
        "intro": "FBA 是跨境卖家最大的「看不见的成本中心」。今天大多数团队的 FBA 管理方式是：靠感觉判断补货时机、靠经验选择头程方式、月底才发现有长库龄费账单——行业均值库存周转天数 95 天（标杆 60 天），这 35 天的差距意味着资金效率损失 30%+。长库龄费、移仓费、头程超支，每一项单独看都不大，加总起来可能吃掉全年利润的一半。本手册提供数据驱动的 FBA 全链路优化：库存健康诊断、头程路线决策、旺季备货计划。",
        "steps": [
            {
                "step": "Step 1：库存健康诊断与长库龄清仓",
                "problem": "长库龄（>180 天）SKU 每月额外收费，但「应该清仓还是降价促销还是转移到海外仓」没有系统性决策框架",
                "skills": [
                    {"id": "Skill-Inventory-Health-Aging-Attribution",
                     "why": "业务指标驱动的库存预测：按库存健康分层，识别「高风险滞销品」并给出清仓 vs 维价 vs 转仓的量化建议"},
                    {"id": "Skill-Markdown-Optimization",
                     "why": "清仓折扣优化：长库龄 SKU 在剩余库存寿命内最大化回收价值，多回收 15-40%（ROI 20-50万）"},
                ],
                "data": "需要：全 SKU 库存年龄报告、销售速度（Sales Velocity）、FBA 仓储费率",
                "output": "SKU 库存健康评分 + 清仓/维价/转仓优先级清单",
            },
            {
                "step": "Step 2：需求预测驱动补货计划",
                "problem": "月底补货计划靠拍脑袋：要么缺货（BSR 排名跌落，广告 ACOS 飙升），要么积压（长库龄费开始计算）",
                "skills": [
                    {"id": "Skill-Hierarchical-Demand-Forecasting-Reconciliation",
                     "why": "分层预测调和：SKU→ASIN→市场三层预测数学保证一致，月度补货计划自动生成（ROI 800-1500万）"},
                    {"id": "Skill-Promotion-Logistics-Surge-Forecast",
                     "why": "大促物流爆仓预测：Prime Day/黑五前 3-7 天预测 FBA 入库峰值，提前锁定仓位防爆仓（ROI 20-40万）"},
                ],
                "data": "需要：历史销售数据（90天+）、大促日历、当前库存水位",
                "output": "月度 FBA 补货计划 + 大促前置入库时间表",
            },
            {
                "step": "Step 3：头程路线成本优化",
                "problem": "海运 vs 空运 vs 铁运的选择靠货代报价，不知道「多花 X 美元空运」是否值得——没有时效×成本的系统性对比框架",
                "skills": [
                    {"id": "Skill-Cross-Border-Logistics-Routing",
                     "why": "多式联运帕累托最优路径：成本/时效/碳排放三目标同时优化，识别「当前库存水位下空运是否合算」（ROI 30-50万）"},
                    {"id": "Skill-Lead-Time-Distribution-Risk-GenQOT",
                     "why": "头程交期分布建模：海运延误不是均值问题而是分布问题，动态安全库存防止「准时到港但仍然断货」（ROI 200-500万/年）"},
                ],
                "data": "需要：历史头程时效数据（按货代/路线/季节）、当前各 SKU 安全库存",
                "output": "头程路线推荐（当前最优选择 + 成本差分析）+ 动态安全库存设定",
            },
            {
                "step": "Step 4：多渠道库存调拨",
                "problem": "Amazon FBA 库存充足，但独立站缺货；或者 FBA 某仓库积压，另一仓库缺货——渠道间库存无法实时平衡",
                "skills": [
                    {"id": "Skill-Multi-Channel-Inventory-Pooling",
                     "why": "跨渠道动态调拨：Amazon FBA + 独立站 + TikTok Shop 库存实时平衡，防止单渠道缺货同时另一渠道积压（ROI 200-400万）"},
                    {"id": "Skill-Returns-Reverse-Logistics",
                     "why": "退货库存再利用：FBA 退货品质检后按「可售/翻新/销毁」三路分流，退货库存回收率提升（ROI 6-10万）"},
                ],
                "data": "需要：各渠道实时库存、销售速度、调拨成本矩阵",
                "output": "跨渠道库存调拨指令 + 退货品分流建议",
            },
        ],
        "outcomes": [
            "库存周转天数从 95 天降至 65 天，释放压占资金 200-400万",
            "长库龄费支出减少 60%（提前清仓 + 补货节奏优化）",
            "头程成本优化 15-25%（路线选择 + 安全库存精准设定）",
            "大促前不爆仓：Prime Day/黑五提前 7 天完成入库",
            "跨渠道缺货率从 8% → 2%，保护 BSR 排名稳定",
        ],
    },
    {
        "id": "pb-attribution-unification",
        "icon": "CR",
        "name": "全渠道归因统一手册",
        "tag": "多渠道归因 · MMM预算",
        "desc": "Amazon/TikTok/Meta/Google 四份报告数字打架——PVM 统一归因窗口 480万/年，Bayesian MMM 1000万/年",
        "intro": "每个月运营拿到四份报告：Amazon 说 TikTok 贡献 15%，TikTok 说自己贡献 45%，Meta 说 30%，加起来超过 100%。今天大多数团队面对这个数字时的做法是：取平均、凭感觉调预算、或者直接信任「效果最好看」的平台数据。这导致每月 10-20% 的广告预算被浪费在归因错误上，高价值渠道（如 TikTok 品牌建设）被系统性低估和砍预算。本手册提供从跨设备追踪到贝叶斯 MMM 的完整「真相还原」链路，让每一分广告费的真实贡献可量化、可对比、可优化。",
        "steps": [
            {
                "step": "Step 1：数据管道统一（所有归因的前提）",
                "problem": "Amazon/TikTok/Meta/Google 数据在四个孤岛里，手动导出对齐每月耗费 2-3 天人工——这一步不解决，后续所有归因都是沙堡",
                "skills": [
                    {"id": "Skill-Marketing-Data-Pipeline",
                     "why": "多渠道归因数据采集管道：统一接入 Meta/TikTok/Amazon 广告 API，标准化时间戳和事件定义，消灭「数据格式不统一」的根因（ROI 12万）"},
                ],
                "data": "需要：各平台广告账户 API 权限（Amazon ADS API / TikTok Ads API / Meta Graph API）",
                "output": "统一格式的多渠道广告数据仓库（每日自动更新）",
            },
            {
                "step": "Step 2：跨设备用户路径拼接",
                "problem": "TikTok 手机种草 → Safari 桌面购买的链路断裂，投手看不到 TikTok 的真实转化，误判削减预算",
                "skills": [
                    {"id": "Skill-GraphTrack-Cross-Device-Tracking",
                     "why": "图基跨设备追踪：无监督 IP-Domain 图谱拼接手机端和桌面端的同一用户，恢复被断裂链路遮蔽的真实 ROAS（ROI 600-1200万）"},
                    {"id": "Skill-CDA-Privacy-Causal-Attribution",
                     "why": "GDPR/CCPA 隐私合规版：欧洲市场无 Cookie 环境下的因果归因，不依赖第三方追踪（必须有，EU 市场合规要求）"},
                ],
                "data": "需要：用户 IP 日志、设备指纹、点击 ID（各平台原始日志）",
                "output": "跨设备用户旅程图谱 + 修正后的各渠道真实转化数",
            },
            {
                "step": "Step 3：归因窗口统一化",
                "problem": "Amazon 14 天归因窗口系统性「抢走」TikTok/Meta 的功劳——同一笔订单，Amazon 说是自己的，TikTok 也说是自己的，导致 ROAS 虚高",
                "skills": [
                    {"id": "Skill-PVM-Attribution-Window-Harmonization",
                     "why": "PVM 跨平台归因窗口统一化：消除 Amazon 14d vs TikTok 7d vs Meta 1d 的窗口差异，让三平台 ROAS 真正可比（ROI 480万/年）"},
                    {"id": "Skill-Causal-Attribution-Bridge",
                     "why": "因果归因桥梁：将 naive「点击→购买」相关性归因升级为反事实「如果没有这个广告还会买吗」，识别 13% 的虚假归因贡献（ROI 10-20万）"},
                ],
                "data": "需要：Step 2 输出的跨设备用户图谱 + 各平台原始归因报告",
                "output": "统一归因窗口后的各渠道真实 ROAS 对比表",
            },
            {
                "step": "Step 4：多触点归因建模",
                "problem": "用户购买前平均接触 6-8 个广告触点（TikTok 种草 → Google 搜索 → Amazon 直接购买），末次点击模型把 100% 功劳给最后一步，严重低估上游渠道价值",
                "skills": [
                    {"id": "Skill-FrontDoor-Causal-MTA",
                     "why": "前门准则多触点因果归因：用因果图正确拆分「TikTok 内容」→「品牌搜索」→「购买」的链路贡献，识别 TikTok 的真实品牌建设价值（ROI 150-300万）"},
                    {"id": "Skill-Promotion-Effectiveness",
                     "why": "因果 ML 促销效果评估：区分「本来就会买的用户」和「被广告说服的用户」，防止把自然购买误算为广告功劳"},
                ],
                "data": "需要：用户完整触点序列（曝光→点击→加购→购买，含时间戳）",
                "output": "各触点渠道的真实边际贡献分配 + MTA 归因报告",
            },
            {
                "step": "Step 5：宏观预算分配（MMM）",
                "problem": "微观归因解决「哪个广告带来了这笔订单」，但无法回答「我应该把 100 万预算怎么分配才能最大化 GMV」——这是 MMM 的问题",
                "skills": [
                    {"id": "Skill-Identified-Bayesian-MMM",
                     "why": "无混淆贝叶斯 MMM：用高斯过程消除识别危机，输出各渠道真实饱和曲线，让 CMO 预算分配决策可信（ROI 1000万）"},
                    {"id": "Skill-Channel-Saturation-Curve",
                     "why": "渠道饱和曲线建模：量化「这个渠道再多投 1 万边际回报是多少」，找到各渠道的最优投放点（ROI 18-25万）"},
                    {"id": "Skill-Geo-Level-Marketing-Effectiveness",
                     "why": "地理级营销效果：美国加州 ROI 2.8x vs 全国均值 1.9x，区域差异化投放释放 20-40万增量（ROI 20-40万）"},
                    {"id": "Skill-DARA-Agentic-MMM-Optimizer",
                     "why": "LLM+RL 双阶段广告预算分配 Agent：基于 MMM 结果自动执行日预算分配，冷启动 ROAS +15-30%（ROI 360-720万）"},
                ],
                "data": "需要：近 12 个月各渠道投放额 + GMV 数据（周粒度），各渠道饱和曲线基础数据",
                "output": "各渠道最优预算分配方案 + 饱和曲线 + 自动执行 Agent",
            },
        ],
        "outcomes": [
            "归因窗口统一后各渠道 ROAS 真正可比，消除「平台各自抢功劳」（ROI 480万/年）",
            "跨设备路径拼接恢复 TikTok 真实贡献，防止因误判而削减高价值渠道（ROI 600-1200万）",
            "MTA 多触点归因识别品牌建设渠道的真实价值（ROI 150-300万）",
            "Bayesian MMM 预算分配：每 100 万预算多出 15-20 万 GMV（ROI 1000万）",
            "「广告数据对账」从每月 2-3 天人工 → 全自动实时更新",
        ],
    },
]



AGENT_CATALOG: list[dict[str, Any]] = [
    {
        "id": "agent-product-radar", "icon": "RA", "name": "选品雷达",
        "category": "选品分析", "cat_key": "selection", "cat_class": "cat-supply",
        "desc": "输入品类关键词，输出 Amazon/速卖通市场机会评分、竞争密度、需求趋势和推荐切入角度。",
        "roi": "选品决策周期 14天→2天",
        "linked_skills": ["Skill-Market-Size-Estimation", "Skill-New-Product-Opportunity-Mining", "Skill-Category-Trend-Forecasting"],
        "inputs": [
            {"id": "keyword", "label": "品类关键词", "type": "text", "placeholder": "例：硅胶婴儿餐具"},
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


def render_agents_page(skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    """Render the Agent Marketplace page with 12 callable demo agents."""

    cats = {"全部": "", "选品分析": "selection", "Listing优化": "listing",
            "广告归因": "attribution", "VOC分析": "voc", "供应链预警": "supply",
            "客服售后": "cs", "价格策略": "pricing", "合规风控": "risk",
            "数据分析": "analytics", "内容营销": "content", "竞品监控": "competitor"}

    cat_pills = "".join(
        f"<button class='cat-pill{'  active' if k == '全部' else ''}' data-cat='{v}'>{k}</button>"
        for k, v in cats.items()
    )

    def _chip(sid: str) -> str:
        sk = skill_lookup.get(sid)
        label = sk.title[:24] + "…" if sk and len(sk.title) > 24 else (sk.title if sk else sid[-20:])
        href = f"skills/{sid}.html"
        return f"<a class='agent-skill-chip' href='{html.escape(href)}'>{html.escape(label)}</a>"

    def _input_field(inp: dict[str, Any], agent_id: str) -> str:
        fid = f"{html.escape(agent_id)}__{html.escape(inp['id'])}"
        label = html.escape(inp['label'])
        placeholder = html.escape(inp.get('placeholder', ''))
        if inp['type'] == 'textarea':
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<textarea class='modal-input' id='{fid}' rows='4' placeholder='{placeholder}'></textarea></div>")
        elif inp['type'] == 'select':
            opts = "".join(f"<option value='{html.escape(o)}'>{html.escape(o)}</option>" for o in inp.get('options', []))
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<select class='modal-input' id='{fid}'>{opts}</select></div>")
        else:
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<input class='modal-input' type='text' id='{fid}' placeholder='{placeholder}'></div>")

    cards_html = ""
    modals_html = ""
    for ag in AGENT_CATALOG:
        sid_chips = "".join(_chip(s) for s in ag.get("linked_skills", [])[:3])
        cards_html += f"""
<div class='agent-card' data-cat='{html.escape(ag["cat_key"])}' onclick='openAgent("{html.escape(ag["id"])}")'>
  <div class='agent-card-top'>
    <div class='agent-icon-wrap {html.escape(ag["cat_class"])}'>{ag["icon"]}</div>
    <div class='agent-card-info'>
      <div class='agent-name'>{html.escape(ag["name"])}</div>
      <span class='agent-cat-badge'>{html.escape(ag["category"])}</span>
    </div>
  </div>
  <div class='agent-status'>
    <span class='status-dot demo'></span>
    <span style='color:#92400e;font-size:12px;font-weight:600'>演示模式</span>
    &nbsp;·&nbsp;
    <span style='font-size:12px;color:#64748b'>可调用</span>
  </div>
  <p class='agent-desc'>{html.escape(ag["desc"])}</p>
  <div class='agent-skills'>{sid_chips}</div>
  <div class='agent-roi'>{html.escape(ag["roi"])}</div>
  <button class='agent-invoke-btn'>立即调用</button>
</div>"""

        input_fields = "".join(_input_field(inp, ag["id"]) for inp in ag.get("inputs", []))
        demo_out_escaped = html.escape(ag.get("demo_output", ""))
        modals_html += f"""
<div id='modal-{html.escape(ag["id"])}' class='agent-modal-overlay' role='dialog' aria-modal='true' aria-label='{html.escape(ag["name"])}'>
  <div class='agent-modal'>
    <div class='modal-header'>
      <span class='modal-icon'>{ag["icon"]}</span>
      <div class='modal-header-info'>
        <h2>{html.escape(ag["name"])}</h2>
        <div style='display:flex;gap:8px;align-items:center'>
          <span class='agent-cat-badge'>{html.escape(ag["category"])}</span>
          <span class='agent-status'><span class='status-dot demo'></span> <span style='font-size:12px;color:#92400e;font-weight:600'>演示模式</span></span>
        </div>
      </div>
      <button class='modal-close' onclick='closeAgent("{html.escape(ag["id"])}")'>×</button>
    </div>
    <div class='modal-body'>
      <div class='modal-section'>
        <h3>输入参数</h3>
        <div style='display:flex;flex-direction:column;gap:14px'>{input_fields}</div>
        <div style='margin-top:8px'>
          <button class='btn-secondary' style='font-size:12px;padding:5px 12px' onclick='fillExample("{html.escape(ag["id"])}")'>填入示例数据</button>
        </div>
      </div>
      <button class='modal-run-btn' id='run-{html.escape(ag["id"])}' onclick='runAgent("{html.escape(ag["id"])}")'>
         <span id='run-label-{html.escape(ag["id"])}'>开始分析</span>
      </button>
      <div class='modal-output' id='output-{html.escape(ag["id"])}'>
        <div class='output-thinking' id='thinking-{html.escape(ag["id"])}' style='display:none'>
                    <span>Agent 正在分析</span>
          <span class='thinking-dots'><span>·</span><span>·</span><span>·</span></span>
        </div>
        <pre class='output-content' id='content-{html.escape(ag["id"])}' style='margin:0;font-family:inherit;white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.7'></pre>
      </div>
      <div class='modal-footer-skills' id='footer-skills-{html.escape(ag["id"])}' style='margin-top:16px'>
        <span style='font-size:12px;color:#64748b;font-weight:600'>关联 Skills：</span>
        {sid_chips}
      </div>
    </div>
  </div>
</div>"""

    demo_data_js = json.dumps(
        {ag["id"]: {"output": ag.get("demo_output", ""), "inputs": ag.get("inputs", [])} for ag in AGENT_CATALOG},
        ensure_ascii=False,
    )

    body = f"""
<div class='agent-hero'>
  <div class='agent-hero-text'>
    <h1 style='font-size:32px;font-weight:900;letter-spacing:-.03em;margin:0 0 10px'>
      智能体广场
    </h1>
    <p class='lead'>12 个专业 AI Agent，覆盖选品→Listing→广告→客服→合规全链路</p>
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:4px'>
      <span style='font-size:13px;background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:999px;font-weight:600'>演示模式</span>
      <span style='font-size:13px;color:#64748b'>所有 Agent 均可在线试用，输入你的数据即可获得个性化分析</span>
    </div>
  </div>
  <div class='agent-hero-stats'>
    <div class='agent-stat'><strong>12</strong><span>个专业 Agent</span></div>
    <div class='agent-stat'><strong>7</strong><span>业务场景</span></div>
    <div class='agent-stat'><strong>30+</strong><span>关联 Skills</span></div>
  </div>
</div>

<div class='agent-cat-filter' id='catFilter'>
  {cat_pills}
</div>

<div class='agent-grid' id='agentGrid'>
  {cards_html}
</div>

{modals_html}

<script>
const DEMO_DATA = {demo_data_js};

function openAgent(id) {{
  const overlay = document.getElementById('modal-' + id);
  if (!overlay) return;
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
  const firstInput = overlay.querySelector('.modal-input');
  if (firstInput) setTimeout(() => firstInput.focus(), 200);
}}
function closeAgent(id) {{
  const overlay = document.getElementById('modal-' + id);
  if (overlay) overlay.classList.remove('open');
  document.body.style.overflow = '';
  resetOutput(id);
}}
function resetOutput(id) {{
  const out = document.getElementById('output-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const content = document.getElementById('content-' + id);
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  if (out) out.classList.remove('visible');
  if (thinking) thinking.style.display = 'none';
  if (content) content.textContent = '';
  if (btn) btn.disabled = false;
  if (label) label.textContent = '开始分析';
}}
function fillExample(id) {{
  const data = DEMO_DATA[id];
  if (!data || !data.inputs) return;
  data.inputs.forEach(inp => {{
    const el = document.getElementById(id + '__' + inp.id);
    if (!el) return;
    if (inp.type === 'textarea') {{
      el.value = inp.placeholder || '';
    }} else if (inp.type === 'select') {{
      if (inp.options && inp.options.length > 0) el.value = inp.options[0];
    }} else {{
      el.value = inp.placeholder ? inp.placeholder.replace(/^例：/, '') : '';
    }}
  }});
}}
async function runAgent(id) {{
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const out = document.getElementById('output-' + id);
  const content = document.getElementById('content-' + id);
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  if (label) label.textContent = '分析中...';
  if (content) content.textContent = '';
  if (out) out.classList.add('visible');
  if (thinking) thinking.style.display = 'flex';
  await sleep(1400);
  if (thinking) thinking.style.display = 'none';
  const text = (DEMO_DATA[id] || {{}}).output || '演示输出暂不可用';
  await streamText(content, text);
  if (btn) btn.disabled = false;
  if (label) label.textContent = '重新分析';
}}
async function streamText(el, text) {{
  let i = 0;
  const chunk = 2;
  while (i < text.length) {{
    el.textContent += text.slice(i, i + chunk);
    el.parentElement && (el.parentElement.scrollTop = el.parentElement.scrollHeight);
    const c = text[i];
    await sleep(c === '\\n' ? 60 : c === '。' || c === '，' ? 30 : 12);
    i += chunk;
  }}
}}
function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

document.querySelectorAll('.agent-modal-overlay').forEach(ov => {{
  ov.addEventListener('click', e => {{
    if (e.target === ov) {{
      const id = ov.id.replace('modal-', '');
      closeAgent(id);
    }}
  }});
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    document.querySelectorAll('.agent-modal-overlay.open').forEach(ov => {{
      const id = ov.id.replace('modal-', '');
      closeAgent(id);
    }});
  }}
}});

const catBtns = document.querySelectorAll('.cat-pill');
const cards = document.querySelectorAll('.agent-card');
catBtns.forEach(btn => {{
  btn.addEventListener('click', () => {{
    catBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const cat = btn.dataset.cat;
    cards.forEach(c => {{
      c.style.display = (cat === '' || c.dataset.cat === cat) ? '' : 'none';
    }});
  }});
}});
</script>

"""
    return html_page("智能体广场", body, active_nav="agents")


def render_roadmap_page(skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    """CEO-facing AI capability roadmap whitepaper. Designed for B2B sales, print-ready via @media print."""

    PHASES = [
        {
            "id": "phase1",
            "label": "Phase 1",
            "period": "第 1-3 个月",
            "theme": "立竿见影（Month 1-3）",
            "color": "#2563eb",
            "bg": "#eff6ff",
            "tagline": "30 天内出数字，首月可见 ROI",
            "roi": "800 - 1,900 万/年",
            "items": [
                {
                    "icon": "SC",
                    "title": "供应链预测基线",
                    "story": "某母婴品牌 60+ SKU × 多仓 × 多市场，每月底 2-3 名 PM 纯人工对账三层预测数字——SKU 求和对不上仓库数，仓库数对不上市场总量，相差最高 50%。",
                    "result": "HiFoReAd 分层调和后：对账人力清零，补货计划冲突率降至 < 5%",
                    "roi": "800-1,500 万/年",
                    "skills": ["Skill-Hierarchical-Demand-Forecasting-Reconciliation", "Skill-Demand-Forecasting-Supply-Chain"],
                },
                {
                    "icon": "AI",
                    "title": "客服智能路由 + 70% 工单自动化",
                    "story": "某品牌日均 5 万条跨领域工单，人工路由正确率 61%，每天约 1 万条工单需二次转单；新手妈妈咨询「宝宝 3 月夜醒频繁」，人工客服响应平均 72 小时，且有医疗合规风险。",
                    "result": "AgentRouter 路由正确率 61% → 82%；客服决策树从历史日志自学，70% 工单实现自动化处理，年节省运营成本",
                    "roi": "1,900 万/年（路由）+ 600 万/年（自动化）",
                    "skills": ["Skill-AgentRouter-KG-Guided", "Skill-Customer-Journey-Decision-Tree"],
                },
                {
                    "icon": "RA",
                    "title": "广告归因修正",
                    "story": "某品牌 TikTok 渠道被 naive 归因分配 45% 贡献，但因果 ITE 分析显示真实增量只有 32%——13% 的购买是用户自然意愿，和广告无关。",
                    "result": "纠正后 TikTok 预算从 $40K 调至 $30K，节省的 $10K 转投 Google（ITE 更高），预算效率提升",
                    "roi": "10-20 万/年",
                    "skills": ["Skill-Causal-Attribution-Bridge"],
                },
            ],
        },
        {
            "id": "phase2",
            "label": "Phase 2",
            "period": "第 4-6 个月",
            "theme": "让快赢可持续（Month 4-6）",
            "color": "#7c3aed",
            "bg": "#f5f3ff",
            "tagline": "让 Phase 1 的效果可持续、可复制",
            "roi": "200 - 500 万/年（新增）",
            "items": [
                {
                    "icon": "KG",
                    "title": "产品知识图谱",
                    "story": "AI 在没有结构化产品知识的情况下，给用户推荐「买了吸奶器的用户还需要什么」——它不知道硅胶法兰和乳头霜属于哺乳期刚需配件，推荐准确率极低。",
                    "result": "构建产品 KG 后：KGQA 查询召回率 52% → 92%，跨品类推荐 CTR +18%",
                    "roi": "20-35 万/年（推荐层增量）",
                    "skills": ["Skill-Hierarchical-Product-KG-Construction", "Skill-Ontology-Schema-Design"],
                },
                {
                    "icon": "VP",
                    "title": "A/B 实验平台",
                    "story": "某品牌每次调整定价策略或上新 listing，无法区分效果是真实改进还是季节波动。团队争论持续数周，决策依赖「感觉」。",
                    "result": "Switchback 实验体系搭建后：物流/双边市场实验可信，决策从争论变为数据裁决",
                    "roi": "1,500 万/年（错误决策避免）",
                    "skills": ["Skill-Switchback-Experiment-Design", "Skill-CUPED-Variance-Reduction"],
                },
                {
                    "icon": "AA",
                    "title": "多渠道库存池化",
                    "story": "Amazon FBA 仓吸奶器缺货（超卖），独立站海外仓还有 200 件积压，TikTok Shop 慢速消化——三渠道不互通，总库存 800 件但某渠道已断货。",
                    "result": "跨渠道动态调拨后：总库存减少 15-25%，缺货率 8% → 3%",
                    "roi": "200-400 万/年",
                    "skills": ["Skill-Multi-Channel-Inventory-Pooling"],
                },
            ],
        },
        {
            "id": "phase3",
            "label": "Phase 3",
            "period": "第 7-12 个月",
            "theme": "建立不对称优势（Month 7-12）",
            "color": "#059669",
            "bg": "#f0fdf4",
            "tagline": "竞争对手需要 18 个月才能追上这里",
            "roi": "5,000 万+ 潜力（战略级）",
            "items": [
                {
                    "icon": "PR",
                    "title": "AI 定价引擎",
                    "story": "某品牌大促前手动跟价——降太多伤利润，降太少丢份额。运营靠经验感知，每次大促前都是高压决策，无法同时优化当前销量和品牌长期溢价。",
                    "result": "AIGP 动态定价 A/B 实测：GMV +13%，实验数据非预测值",
                    "roi": "1,321 万/年（A/B 实测）",
                    "skills": ["Skill-AIGP-LLM-Dynamic-Pricing", "Skill-Dynamic-Pricing-Elasticity"],
                },
                {
                    "icon": "TC",
                    "title": "AI 内容工厂",
                    "story": "某品牌进入德国/日本市场，需要本地化口播 Review 视频。人工方案：雇本地 KOL 拍摄，周期 3-4 周，单条成本 $2,000+。批量生产 20 个 SKU 的测评视频需要 6 个月预算。",
                    "result": "Virbo 多语言虚拟人：同等内容量成本降低 80%，生产周期 3 周→ 3 天（实验接入中）",
                    "roi": "35-150 万/年（视接入程度）",
                    "skills": ["Skill-Virbo-Multilingual-Avatar-UGC", "Skill-AnchorCrafter-Virtual-Anchor-Demo"],
                },
                {
                    "icon": "TR",
                    "title": "MAS 多智能体联动",
                    "story": "大促首日某品牌吸奶器打 7 折卖出 5,000 件，第 3 天库存告急被迫涨价，剩余 7 天流量白白浪费——整个大促周期总利润反而低于平销期。根因：定价和补货是两个团队各自决策，无法实时联动。",
                    "result": "FSDA-DRL 快慢双 Agent：定价与补货实时联动，大促周期利润最优化，中小卖家（月 GMV 100-500 万）保守估计年化 225-300 万",
                    "roi": "225-300 万/年（中小规模）· 5,000 万+（GMV > 2 亿规模）",
                    "skills": ["Skill-FSDA-DRL", "Skill-Event-Driven-Demand-MAS"],
                },
                {
                    "icon": "AG",
                    "title": "防御性 AI：保护推荐系统不被竞品劫持",
                    "story": "竞品卖家在商品描述中嵌入恶意 prompt 指令，劫持 AI 导购排名，导致某品牌自营商品在 AI 搜索中的曝光量下降 30-50%——这是 2025 年已出现的真实攻击方式。",
                    "result": "Agent 支付安全红队：自动检测注入攻击并拦截，保护 AI 推荐系统不被操控",
                    "roi": "防御价值 > 5,000 万（以被攻击时的流量损失计）",
                    "skills": ["Skill-Agent-Payment-Security-Red-Team", "Skill-MAS-Adversarial-Defense"],
                },
            ],
        },
    ]

    def _phase_html(phase: dict[str, Any]) -> str:
        items_html = ""
        for item in phase["items"]:
            skill_chips = ""
            for sid in item.get("skills", []):
                sk = skill_lookup.get(sid)
                if sk:
                    skill_chips += (
                        f"<a class='rm-chip' href='skills/{html.escape(sid)}.html'>"
                        f"{html.escape(sk.title[:36])}{'…' if len(sk.title) > 36 else ''}</a>"
                    )
            items_html += f"""
<div class="rm-item">
  <div class="rm-item-icon">{item['icon']}</div>
  <div class="rm-item-body">
    <h4 class="rm-item-title">{html.escape(item['title'])}</h4>
    <div class="rm-story">
      <span class="rm-story-label">真实案例</span>
      {html.escape(item['story'])}
    </div>
    <div class="rm-result">
      <span class="rm-result-label">[OK] 结果</span>
      {html.escape(item['result'])}
    </div>
    <div class="rm-roi-line">年化 ROI：<strong>{html.escape(item['roi'])}</strong></div>
    <div class="rm-chips">{skill_chips}</div>
  </div>
</div>"""

        return f"""
<div class="rm-phase" id="{phase['id']}" style="--phase-color:{phase['color']};--phase-bg:{phase['bg']}">
  <div class="rm-phase-header">
    <div class="rm-phase-badge">{html.escape(phase['label'])}</div>
    <div class="rm-phase-meta">
      <span class="rm-phase-period">{html.escape(phase['period'])}</span>
      <h3 class="rm-phase-theme">{html.escape(phase['theme'])}</h3>
      <p class="rm-phase-tagline">{html.escape(phase['tagline'])}</p>
    </div>
    <div class="rm-phase-roi">
      <span class="rm-phase-roi-label">阶段可验证 ROI</span>
      <strong>{html.escape(phase['roi'])}</strong>
    </div>
  </div>
  <div class="rm-items">{items_html}</div>
</div>"""

    phases_html = "".join(_phase_html(p) for p in PHASES)

    body = f"""
<div class="rm-scqa">
  <div class="rm-scqa-s"><span class="rm-scqa-label">现状</span>2025年，母婴跨境出海品牌平均每月仍有 3 名运营人员全职处理重复性决策——手工对账、等待提数、人工盯价，每天消耗 72 小时以上的宝贵人力。</div>
  <div class="rm-scqa-c"><span class="rm-scqa-label">冲突</span>而先行品牌已在用 AI 将这些成本清零——AgentRouter 年节省 1,900 万元运营成本，AIGP 定价 A/B 实测 GMV +13%。一旦错过这个窗口，差距将在 18-24 个月内变得不可追赶。</div>
  <div class="rm-scqa-q"><span class="rm-scqa-label">问题</span>你的品牌应该从哪里开始，才能在首月就看到可验证的 ROI？</div>
</div>
<div class="rm-hero">
  <div class="rm-hero-eyebrow">唯一把顶会 ML 论文翻译为跨境运营决策的平台 · 2025-2026</div>
  <h1 class="rm-hero-title">12 个月，3 阶段，AI 替代 3 类岗位的重复性决策</h1>
  <p class="rm-hero-sub">首月可见 ROI，全年可验证收益 > 3,000 万元 | NeurIPS · KDD · ICML 论文背书</p>
  <div class="rm-hero-cta">
    <button class="rm-btn-primary" onclick="window.print()">下载 PDF</button>
    <a class="rm-btn-sec" href="playbooks/index.html">查看场景手册 →</a>
    <a class="rm-btn-sec" href="mailto:skills@lute-tlz-dddd.top?subject=预约Demo-AI能力路线图" style="background:#9c5455;color:#fff;border:none">预约 Demo</a>
  </div>
  <p class="rm-hero-note">所有 ROI 数字来源于真实 A/B 实验或匿名客户案例，非模型预测 | 与 Northbeam / Jungle Scout / 纯咨询公司的核心差异：我们给你的是「决策算法」，不是「数据报表」</p>
</div>

<div class="rm-summary-bar">
  <div class="rm-summary-item">
    <span class="rm-summary-num">3</span>
    <span class="rm-summary-label">阶段</span>
  </div>
  <div class="rm-summary-sep">→</div>
  <div class="rm-summary-item">
    <span class="rm-summary-num">10</span>
    <span class="rm-summary-label">核心场景</span>
  </div>
  <div class="rm-summary-sep">→</div>
  <div class="rm-summary-item">
    <span class="rm-summary-num">3,000万+</span>
    <span class="rm-summary-label">可验证年化 ROI</span>
  </div>
  <div class="rm-summary-sep">→</div>
  <div class="rm-summary-item">
    <span class="rm-summary-num">3</span>
    <span class="rm-summary-label">岗位重复性工作被替代</span>
  </div>
</div>

<div class="rm-roles-bar">
  <div class="rm-role">
    
    <div>
      <strong>供应链全链路</strong>
      <p>从 15 个 Excel 联动 → 1 个 MAS 自动执行</p>
      <span class="rm-role-roi">ROI 5,000-8,000 万</span>
    </div>
  </div>
  <div class="rm-role">
    
    <div>
      <strong>数据分析师</strong>
      <p>提数从 72 小时 → 5 分钟，报告从做 → 审</p>
      <span class="rm-role-roi">ROI 1,600-3,000 万</span>
    </div>
  </div>
  <div class="rm-role">
    
    <div>
      <strong>广告优化师</strong>
      <p>凌晨 2 点平台调整，Agent 已完成出价</p>
      <span class="rm-role-roi">ROI 3,000-5,000 万</span>
    </div>
  </div>
</div>

<div class="rm-phases">
  {phases_html}
</div>

<div class="rm-footer">
  <div class="rm-footer-left">
     <h3>从哪里开始？</h3>
     <p>根据你的当前痛点选择入口——每个场景手册包含完整操作步骤、所需数据和 ROI 计算模板。</p>
     <div class="rm-footer-links">
        <a href="playbooks/pb-risk-defense.html">跨境风险防御作战室</a>
        <a href="playbooks/pb-agent-replace.html">AI Agent 替人手册</a>
        <a href="playbooks/pb-tariff-response.html">关税冲击 72h 响应</a>
        <a href="playbooks/pb-compliance.html">跨境合规全链路</a>
        <a href="playbooks/pb-voc-product-loop.html">竞品情报 → 产品迭代</a>
       <a href="playbooks/pb-customer-service-agent.html">客服售后智能体</a>
       <a href="playbooks/pb-fba-operations.html">FBA 运营全链路</a>
       <a href="playbooks/pb-pricing-engine.html">AI 定价引擎手册</a>
       <a href="playbooks/pb-inventory-festival.html">大促备货决策手册</a>
     </div>
  </div>
  <div class="rm-footer-right">
    <div class="rm-footer-cta">
      <p>获取 PDF + 预约 30 分钟 ROI 测算</p>
      <button class="rm-btn-primary" onclick="window.print()" style="margin-bottom:10px">下载 PDF</button>
      <form class="rm-lead-form" action="mailto:skills@lute-tlz-dddd.top" method="GET">
        <input type="email" name="email" placeholder="your@company.com" required style="width:100%;padding:8px 12px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#f1f5f9;font-size:13px;margin-bottom:8px;box-sizing:border-box">
        <input type="hidden" name="subject" value="paper2skills ROI测算申请">
        <button type="submit" style="width:100%;padding:8px;background:#9c5455;color:#fff;border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer">获取定制 ROI 测算（邮件联系）</button>
      </form>
    </div>
    <p class="rm-footer-note">
      数据来源：350 个从顶会论文萃取的业务 Skills，包含真实 A/B 实验与匿名客户案例。<br>
      所有案例均已脱敏处理，以「某跨境母婴品牌」表述。
    </p>
  </div>
</div>
"""


    return html_page("AI 能力建设路线图", body)


def _render_roi_calculator(calc: dict[str, Any] | None) -> str:
    """Render an interactive ROI calculator as self-contained HTML+JS. No backend needed."""
    if not calc:
        return ""

    sections = calc.get("sections", [])

    tabs_html = "".join(
        "<button class='calc-tab{active}' data-sec='{sid}' style='--tc:{color}'>{label}</button>".format(
            active=" active" if i == 0 else "",
            sid=html.escape(sec["id"]),
            color=html.escape(sec["color"]),
            label=html.escape(sec["label"]),
        )
        for i, sec in enumerate(sections)
    )

    panels_html = ""
    for i, sec in enumerate(sections):
        inputs_html = ""
        for inp in sec["inputs"]:
            inputs_html += f"""
<div class='calc-row'>
  <label class='calc-label' for='{html.escape(sec["id"])}_{html.escape(inp["id"])}'>{html.escape(inp["label"])}</label>
  <div class='calc-input-wrap'>
    <input class='calc-input' type='range'
      id='{html.escape(sec["id"])}_{html.escape(inp["id"])}'
      data-sec='{html.escape(sec["id"])}' data-var='{html.escape(inp["id"])}'
      min='{inp["min"]}' max='{inp["max"]}' step='{inp["step"]}' value='{inp["default"]}'>
    <span class='calc-val' id='v_{html.escape(sec["id"])}_{html.escape(inp["id"])}'>{inp["default"]}</span>
    <span class='calc-unit'>{html.escape(inp["unit"])}</span>
  </div>
</div>"""

        active_cls = " active" if i == 0 else ""
        panels_html += (
            "<div class='calc-panel{active}' id='panel_{sid}' data-sec='{sid}' style='--tc:{color}'>"
            "<div class='calc-inputs'>{inputs}</div>"
            "<div class='calc-result'>"
            "<div class='calc-result-label'>年化 ROI 估算</div>"
            "<div class='calc-result-num' id='result_{sid}'>—</div>"
            "<div class='calc-result-unit'>万元/年</div>"
            "<div class='calc-disclaimer'>基于行业平均改善率保守估算，实际收益因业务规模与实施深度而异</div>"
            "</div></div>"
        ).format(
            active=active_cls,
            sid=html.escape(sec["id"]),
            color=html.escape(sec["color"]),
            inputs=inputs_html,
        )

    formulas_js = "{\n"
    for sec in sections:
        var_names = [inp["id"] for inp in sec["inputs"]]
        defaults = {inp["id"]: inp["default"] for inp in sec["inputs"]}
        formulas_js += f"  '{html.escape(sec['id'])}': {{\n"
        formulas_js += f"    vars: {json.dumps(var_names)},\n"
        formulas_js += f"    defaults: {json.dumps(defaults)},\n"
        formula_body = sec["formula"].strip().replace("\n", "\n    ")
        formulas_js += f"    compute: function({', '.join(var_names)}) {{\n    {formula_body}\n    }},\n"
        formulas_js += f"  }},\n"
    formulas_js += "}"

    js = f"""
<script>
(function(){{
  var CALC = {formulas_js};
  var state = {{}};
  Object.keys(CALC).forEach(function(sec){{
    state[sec] = Object.assign({{}}, CALC[sec].defaults);
  }});

  function compute(sec){{
    var cfg = CALC[sec];
    var args = cfg.vars.map(function(v){{ return state[sec][v]; }});
    try {{
      var result = cfg.compute.apply(null, args);
      var el = document.getElementById('result_' + sec);
      if(el) el.textContent = isNaN(result) || result < 0 ? '—' : result.toLocaleString('zh-CN');
    }} catch(e) {{}}
  }}

  document.querySelectorAll('.calc-input').forEach(function(inp){{
    var sec = inp.dataset.sec, v = inp.dataset.var;
    var valEl = document.getElementById('v_' + sec + '_' + v);
    inp.addEventListener('input', function(){{
      state[sec][v] = parseFloat(this.value);
      if(valEl) valEl.textContent = this.value;
      compute(sec);
    }});
    compute(sec);
  }});

  document.querySelectorAll('.calc-tab').forEach(function(btn){{
    btn.addEventListener('click', function(){{
      var sec = this.dataset.sec;
      document.querySelectorAll('.calc-tab').forEach(function(b){{ b.classList.remove('active'); }});
      document.querySelectorAll('.calc-panel').forEach(function(p){{ p.classList.remove('active'); }});
      this.classList.add('active');
      var panel = document.getElementById('panel_' + sec);
      if(panel){{ panel.classList.add('active'); compute(sec); }}
    }});
  }});

  Object.keys(CALC).forEach(compute);
}})();
</script>"""

    return f"""
<div class='calc-wrapper'>
  <div class='calc-header'>
    <h2>{html.escape(calc.get('title', 'ROI 计算器'))}</h2>
    <p class='muted'>{html.escape(calc.get('subtitle', ''))}</p>
  </div>
  <div class='calc-tabs'>{tabs_html}</div>
  <div class='calc-body'>{panels_html}</div>
</div>
{js}"""


def render_tob_playbook(pb: dict[str, Any], skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    nav = "../"
    steps_html = ""
    for i, step in enumerate(pb.get("steps", []), 1):
        skills_html = ""
        for bs in step.get("skills", []):
            sid = bs["id"]
            why = html.escape(bs.get("why", ""))
            sk = skill_lookup.get(sid)
            if sk:
                roi = f"<span class='roi-badge'>{html.escape(sk.roi_figure)}</span>" if sk.roi_figure else ""
                diff = f"<span class='diff-badge'>{html.escape(sk.difficulty)}</span>" if sk.difficulty else ""
                skills_html += (
                    f"<div class='pb-skill'>"
                    f"<div class='pb-skill-header'>"
                    f"<a href='../skills/{html.escape(sid)}.html' class='pb-skill-name'>{html.escape(sk.title)}</a>"
                    f"<div class='pb-skill-badges'>{roi}{diff}</div>"
                    f"</div>"
                    f"<p class='pb-skill-why'>→ {why}</p>"
                    f"</div>"
                )
            else:
                skills_html += f"<div class='pb-skill'><span class='muted'>{html.escape(sid)}</span></div>"

        data_req = html.escape(step.get("data", ""))
        output = html.escape(step.get("output", ""))
        steps_html += f"""
<div class='pb-step'>
  <div class='pb-step-num'>Step {i}</div>
  <div class='pb-step-body'>
    <h3 class='pb-step-title'>{html.escape(step['step'])}</h3>
    <p class='pb-problem'>{html.escape(step['problem'])}</p>
    <div class='pb-skills'>{skills_html}</div>
    {'<div class="pb-data"><strong>所需数据：</strong>' + data_req + '</div>' if data_req else ''}
    {'<div class="pb-output"><strong>输出结果：</strong>' + output + '</div>' if output else ''}
  </div>
</div>"""

    outcomes = "".join(f"<li>[OK] {html.escape(o)}</li>" for o in pb.get("outcomes", []))
    calc_html = _render_roi_calculator(pb.get("roi_calculator")) if pb.get("roi_calculator") else ""
    pb_name_safe = pb['name'].replace(' ', '%20').replace('&', '%26')
    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../playbooks/index.html">场景手册</a> / {html.escape(pb['name'])}</nav>
<div class='pb-hero'>
  <span class='pb-icon'>{pb['icon']}</span>
  <div>
    <h1>{html.escape(pb['name'])}</h1>
    <p class='lead'>{html.escape(pb['desc'])}</p>
    <span class='biz-tag'>{html.escape(pb['tag'])}</span>
  </div>
</div>
{''.join([f"<div class='pb-roi-callout'>{html.escape(item['label'])}<span class='pb-roi-val'>{html.escape(item['value'])}</span></div>" for item in pb.get('roi_callout', [])])}
<div class='pb-intro'>{html.escape(pb['intro'])}</div>
{'<div class="wf-outcomes"><h3>预期收益</h3><ul>' + outcomes + '</ul></div>' if outcomes else ''}
<div class='pb-steps'>{steps_html}</div>
{calc_html}
<div class='pb-lead-capture'>
  <div class='pb-lead-inner'>
    <div class='pb-lead-text'>
      <h3>想了解这套方案如何落地你的业务？</h3>
      <p>预约 30 分钟免费 ROI 测算 — 基于你的 SKU 数量、广告预算和当前痛点，给出定制化收益估算。</p>
      <ul class='pb-lead-bullets'>
        <li>✓ 结合你的实际数据，不是通用模板</li>
        <li>✓ 明确哪 1-2 个 Skill 优先落地 ROI 最高</li>
        <li>✓ 30 分钟，结束后你有一份行动清单</li>
      </ul>
    </div>
    <div class='pb-lead-action'>
      <a href='mailto:skills@lute-tlz-dddd.top?subject=预约ROI测算-{pb_name_safe}&body=手册:{pb_name_safe}%0A公司规模:%0A主要痛点:%0A当前月GMV:' class='pb-lead-btn'>预约 30 分钟 ROI 测算 →</a>
      <p class='pb-lead-note'>发送邮件后 24h 内回复确认时间</p>
    </div>
  </div>
</div>
"""
    return html_page(pb["name"], body, nav)


def render_workflow_page(wf_def: dict[str, Any], skill_lookup: dict[str, "PlaybookSkill"]) -> str:
    """Render a full decision-tree workflow page from YAML definition."""
    nav = "../"
    name = html.escape(wf_def.get("name", ""))
    description = html.escape(wf_def.get("description", "按业务流程推荐的 Skill 链。"))
    entry_q = html.escape(wf_def.get("entry_question", ""))
    target_users = wf_def.get("target_users", [])
    outcomes = wf_def.get("outcomes", [])
    steps = wf_def.get("steps", [])

    user_tags = "".join(
        f"<span class='tag'>{html.escape(u)}</span>" for u in target_users
    )
    outcome_items = "".join(
        f"<li>[OK] {html.escape(o)}</li>" for o in outcomes
    )
    step_html = "".join(render_workflow_step(s, skill_lookup) for s in steps)

    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../workflows/index.html">工作流</a> / {name}</nav>
<h1>{name}</h1>
<p class="lead">{description}</p>
<div class="wf-meta">
  <div><strong>适用角色</strong>{user_tags}</div>
  {'<div class="wf-entry-question"><strong>入口问题：</strong>' + entry_q + '</div>' if entry_q else ''}
</div>
{'<div class="wf-outcomes"><h3>预期收益</h3><ul>' + outcome_items + '</ul></div>' if outcomes else ''}
<div class="wf-tree">
  {step_html}
</div>
"""
    return html_page(wf_def.get("name", "工作流"), body, nav)


# ---------------------------------------------------------------------------
# HTML page scaffold
# ---------------------------------------------------------------------------

def html_page(title: str, body: str, nav: str = "", active_nav: str = "") -> str:
    def sidebar_link(href: str, label: str, key: str = "", icon: str = "") -> str:
        active = ' aria-current="page" class="active"' if key and key == active_nav else ""
        icon_html = f'<span class="sbl-icon">{icon}</span>' if icon else ""
        return f'<a href="{nav}{href}"{active}>{icon_html}<span class="sbl-text">{label}</span></a>'

    def sidebar_section(label: str, links: str) -> str:
        return f'<div class="sb-section"><p class="sb-label">{label}</p><div class="sb-links">{links}</div></div>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)} · paper2skills</title>
  <link rel="stylesheet" href="{nav}assets/style.css">
</head>
<body>
  <header class="topbar">
    <button class="hamburger" id="hamburger" aria-label="菜单" aria-expanded="false">
      <span></span><span></span><span></span>
    </button>
    <a class="brand" href="{nav}index.html">
      <span class="brand-icon">P</span>
      <span class="brand-name">paper2skills<span class="brand-tag">Playbook</span></span>
    </a>
    <div class="topbar-right">
      <input id="global-search" placeholder="搜索技能 / 场景…" autocomplete="off" role="search" aria-label="搜索">
      <a href="{nav}ai-roadmap.html" class="topbar-cta{'  active' if active_nav == 'roadmap' else ''}">AI 路线图 →</a>
    </div>
  </header>
  <div id="search-results" class="search-results hidden" role="listbox"></div>
  <div class="mobile-nav-overlay" id="mobile-overlay"></div>
  <main class="layout">
    <aside class="sidebar" id="sidebar">
      <div class="sb-top">
        {sidebar_section('主导航', 
          sidebar_link('index.html', '总览', 'index', '⊞') +
          sidebar_link('chat.html', 'AI 知识库对话', 'chat', '✦') +
          sidebar_link('playbooks/index.html', '场景手册', 'playbooks', '◧') +
          sidebar_link('agents.html', '智能体广场', 'agents', '◈') +
          sidebar_link('ai-roadmap.html', 'AI 能力路线图', 'roadmap', '◉')
        )}
        {sidebar_section('知识图谱',
          sidebar_link('domains/index.html', '按领域浏览', 'domains', '◫') +
          sidebar_link('topics/index.html', '按主题浏览', 'topics', '◪') +
          sidebar_link('workflows/index.html', '业务工作流', 'workflows', '◳') +
          sidebar_link('graph/overview.html', '技能关系图谱', 'graph', '◉') +
          sidebar_link('skills/index.html', '全部 Skills', 'skills', '≡')
        )}
      </div>
    </aside>
    <section class="content">{body}</section>
  </main>

  <!-- AI Chat Panel -->
  <script src="{nav}assets/playbook-data.js"></script>
  <script src="{nav}assets/search.js"></script>
  <script>
  const hbtn = document.getElementById('hamburger');
  const overlay = document.getElementById('mobile-overlay');
  const sidebar = document.getElementById('sidebar');
  function toggleMenu(open) {{
    hbtn.setAttribute('aria-expanded', open);
    hbtn.classList.toggle('open', open);
    sidebar.classList.toggle('open', open);
    overlay.classList.toggle('show', open);
    document.body.style.overflow = open ? 'hidden' : '';
  }}
  hbtn.addEventListener('click', () => toggleMenu(hbtn.getAttribute('aria-expanded') !== 'true'));
  overlay.addEventListener('click', () => toggleMenu(false));
  </script>
</body>
</html>"""


def skill_url(skill_id: str, nav: str = "") -> str:
    return f"{nav}skills/{skill_id}.html"


def render_skill_card(skill: PlaybookSkill, nav: str = "") -> str:
    roi_html = (
        f"<span class='sc-roi'>{html.escape(skill.roi_figure)}</span>"
        if skill.roi_figure else ""
    )
    diff_html = (
        f"<span class='sc-diff'>{html.escape(skill.difficulty)}</span>"
        if skill.difficulty else ""
    )
    footer_html = f"<div class='sc-footer'>{roi_html}{diff_html}</div>" if (roi_html or diff_html) else ""
    desc = html.escape(skill.problem_solved or skill.algorithm_summary)
    data_domain = html.escape(skill.domain_dir)
    data_diff   = html.escape(skill.difficulty or "")
    return f"""<a class="card skill-card" href="{skill_url(skill.skill_id, nav)}" data-domain="{data_domain}" data-diff="{data_diff}">
  <div class="sc-domain">{html.escape(skill.domain_dir)}</div>
  <h3 class="sc-title">{html.escape(skill.title)}</h3>
  <p class="sc-desc">{desc}</p>
  {footer_html}
</a>"""


def link_list(items: list[str], nav: str = "", skill_ids: set[str] | None = None) -> str:
    if not items:
        return "<p class='muted'>暂无</p>"
    _ids = skill_ids if skill_ids is not None else KNOWN_SKILL_IDS
    rows = []
    for item in items:
        escaped = html.escape(item)
        if item in _ids:
            rows.append(f"<li><a href='{skill_url(item, nav)}'>{escaped}</a></li>")
        else:
            rows.append(f"<li><span class='muted'>{escaped}</span></li>")
    return "<ul>" + "".join(rows) + "</ul>"


def render_skill_page(skill: PlaybookSkill) -> str:
    nav = "../"

    # Handbook uplinks: show which handbooks use this skill
    hb_refs = SKILL_HANDBOOK_MAP.get(skill.skill_id, [])
    handbook_uplinks = ""
    if hb_refs:
        chips = "".join(
            f"<a class='hb-uplink' href='../playbooks/{html.escape(pb_id)}.html'>{html.escape(pb_name)}</a>"
            for pb_id, pb_name in hb_refs
        )
        handbook_uplinks = f"<div class='hb-uplinks'><span class='hb-uplinks-label'>收录于</span>{chips}</div>"

    # Scenario section: prefer prose paragraphs; fall back to bullet list
    if skill.scenario_paragraphs:
        scenario_html = "".join(f"<p>{html.escape(p)}</p>" for p in skill.scenario_paragraphs)
    elif skill.business_scenarios:
        scenario_html = render_items(skill.business_scenarios)
    else:
        scenario_html = "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>"

    # ROI / value panel
    roi_meta = ""
    if skill.roi_figure or skill.difficulty or skill.priority:
        parts = []
        if skill.roi_figure:
            parts.append(f"<div class='roi-item'><span class='roi-label'>年化 ROI</span><span class='roi-value'>{html.escape(skill.roi_figure)}</span></div>")
        if skill.difficulty:
            parts.append(f"<div class='roi-item'><span class='roi-label'>实现难度</span><span class='roi-value'>{html.escape(skill.difficulty)}</span></div>")
        if skill.priority:
            parts.append(f"<div class='roi-item'><span class='roi-label'>业务优先级</span><span class='roi-value'>{html.escape(skill.priority)}</span></div>")
        roi_meta = "<div class='roi-panel'>" + "".join(parts) + "</div>"

    # Business context panel (injected from DOMAIN_BUSINESS_CONTEXT)
    biz_panel = ""
    if skill.biz_role or skill.biz_trigger:
        role_html = (
            f"<div class='biz-ctx-item'>"
            f"<span class='biz-ctx-label'>适用角色</span>"
            f"<span class='biz-ctx-value'>{html.escape(skill.biz_role)}"
            + (f"<span class='biz-ctx-secondary'> · {html.escape(skill.biz_role2)}</span>" if skill.biz_role2 else "")
            + f"</span></div>"
        )
        trigger_html = (
            f"<div class='biz-ctx-item biz-ctx-full'>"
            f"<span class='biz-ctx-label'>什么情况下用</span>"
            f"<span class='biz-ctx-value'>{html.escape(skill.biz_trigger)}</span>"
            f"</div>"
        ) if skill.biz_trigger else ""
        outcome_html = (
            f"<div class='biz-ctx-item biz-ctx-full'>"
            f"<span class='biz-ctx-label'>成功是什么样的</span>"
            f"<span class='biz-ctx-value biz-ctx-outcome'>{html.escape(skill.biz_outcome)}</span>"
            f"</div>"
        ) if skill.biz_outcome else ""
        pain_html = (
            f"<div class='biz-ctx-item biz-ctx-full'>"
            f"<span class='biz-ctx-label'>业务痛点</span>"
            f"<div class='biz-pain-tags'>"
            + "".join(
                f"<span class='biz-pain-tag'>{html.escape(p.strip())}</span>"
                for p in skill.biz_pain.split("·") if p.strip()
            )
            + f"</div></div>"
        ) if skill.biz_pain else ""
        platform_html = (
            f"<div class='biz-ctx-item'>"
            f"<span class='biz-ctx-label'>适用平台</span>"
            f"<span class='biz-ctx-value'>{html.escape(skill.biz_platform)}</span>"
            f"</div>"
        ) if skill.biz_platform else ""
        biz_panel = (
            f"<div class='biz-ctx-panel'>"
            f"<div class='biz-ctx-header'>业务视角</div>"
            f"<div class='biz-ctx-grid'>"
            f"{role_html}{platform_html}{trigger_html}{outcome_html}{pain_html}"
            f"</div></div>"
        )

    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../domains/{slugify(skill.domain_dir)}.html">{html.escape(skill.domain_dir)}</a> / {html.escape(skill.skill_id)}</nav>
<h1>{html.escape(skill.title)}</h1>
<p class="muted">{html.escape(skill.skill_id)} · {html.escape(skill.domain_dir)}</p>
<div class="tag-row">{''.join(f"<span class='tag'>{html.escape(t)}</span>" for t in skill.tags + skill.topics + skill.workflows)}</div>
{handbook_uplinks}
{roi_meta}
{biz_panel}
<div class="two-col">
  <section>
    <h2>1. 解决的问题</h2>
    <p>{html.escape(skill.problem_solved or skill.algorithm_summary)}</p>
    <h2>2. 核心算法逻辑</h2>
    <p>{html.escape(skill.algorithm_summary)}</p>
    <h2>3. 业务应用场景</h2>
    {scenario_html}
    <h2>4. 输入数据要求</h2>{render_items(skill.inputs) if skill.inputs else "<p class='muted'>请查看原始代码模板获取输入规格。</p>"}
    <h2>5. 输出结果</h2>{render_items(skill.outputs) if skill.outputs else "<p class='muted'>请查看原始代码模板获取输出规格。</p>"}
    <h2>6. 业务价值 / ROI</h2>{render_items(skill.roi) if skill.roi else ("<p>" + html.escape(skill.roi_figure) + "</p>" if skill.roi_figure else "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>")}
    <h2>7. 代码模板</h2>
    <p class="muted">代码块数量：{skill.code_blocks} · 路径：{html.escape(skill.code_path or '未检测到')}</p>
    {_render_code_preview(skill.code_preview)}
    <h2>8. 论文来源</h2>{render_items(skill.papers)}
  </section>
  <aside class="relation-panel">
    <h2>Skill Relations</h2>
    <svg id="ego-graph" data-skill="{html.escape(skill.skill_id)}" width="280" height="220"></svg>
    <div id="ego-legend" class="ego-legend">
      <span class="edge-dot prereq"></span>前置
      <span class="edge-dot combo" style="margin-left:8px"></span>组合
      <span class="edge-dot ext" style="margin-left:8px"></span>延伸
    </div>
    <h3>前置技能</h3>{link_list(skill.relations.get('prerequisite', []), nav)}
    <h3>延伸技能</h3>{link_list(skill.relations.get('extends', []), nav)}
    <h3>可组合技能</h3>{link_list(skill.relations.get('combinable', []), nav)}
  </aside>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="../assets/ego-graph.js"></script>"""
    return html_page(skill.title, body, nav)


def _md_inline(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', text)
    return text


def render_items(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>"
    return "<ul>" + "".join(f"<li>{_md_inline(html.escape(item))}</li>" for item in items) + "</ul>"


def _render_code_preview(code: str) -> str:
    if not code:
        return "<p class='muted'>请查看原始 Skill 卡片获取完整代码。</p>"
    escaped = html.escape(code)
    return f"<pre class='code-preview'><code>{escaped}</code></pre>"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Index page (Phase 3C — three-audience redesign)
# ---------------------------------------------------------------------------

def render_index(skill_count: int, domain_count: int, edge_count: int, domains: list[dict[str, Any]], skills: list[PlaybookSkill]) -> str:
    domain_cards = "".join(
        f"<a class='metric-card domain-card' href='domains/{slugify(d['vault_dir'])}.html'>"
        f"<strong>{html.escape(d['vault_dir'])}</strong>"
        f"<span>{d.get('skill_count', 0)} Skills</span></a>"
        for d in domains
    )

    # Top 5 skills by relation count (degree centrality proxy)
    skill_degree = {s.skill_id: len(s.relations.get("prerequisite", [])) + len(s.relations.get("combinable", [])) for s in skills}
    hot_skills = sorted(skills, key=lambda s: skill_degree.get(s.skill_id, 0), reverse=True)[:5]
    hot_items = "".join(
        f"<li><a href='skills/{s.skill_id}.html'>{html.escape(s.title)}</a>"
        f"{'<span class=roi-badge>' + html.escape(s.roi_figure) + '</span>' if s.roi_figure else ''}</li>"
        for s in hot_skills
    )

    business_cards = "".join(
        f"<a class='biz-card' href='{e['href']}'>"
        f"<div class='biz-card-header'>"
        f"<span class='biz-icon'>{e['icon']}</span>"
        f"<div class='biz-body'>"
        f"<div class='biz-card-meta'>"
        f"<strong>{html.escape(e['label'])}</strong>"
        f"<span class='biz-tag'>{html.escape(e['tag'])}</span>"
        f"</div>"
        f"<p>{html.escape(e['desc'])}</p>"
        f"</div>"
        f"</div>"
        f"</a>"
        for e in BUSINESS_ENTRIES
    )

    return f"""
<div class="hero">
  <p class="hero-badge">唯一把顶会 ML 论文翻译为跨境运营决策的平台</p>
  <h1>母婴跨境品牌用这里的 AI 技能，每年多赚 3,000 万</h1>
  <p class="lead">350 个从 NeurIPS / KDD / ICML 萃取的可落地决策技能——每个技能有真实 ROI 数字、可运行代码、和跨境电商业务场景。这是任何咨询公司和 SaaS 工具都无法复制的能力。</p>
  <div class="hero-primary-cta">
    <a class="btn-primary accent" href="ai-roadmap.html">查看 AI 能力路线图</a>
    <a class="btn-secondary" href="mailto:skills@lute-tlz-dddd.top?subject=预约Demo-paper2skills" >预约 30 分钟 Demo</a>
  </div>
  <div class="hero-tabs" id="heroTabs">
    <button class="tab-btn active" data-tab="biz">业务专家 / 运营</button>
    <button class="tab-btn" data-tab="ds">数据科学家</button>
    <button class="tab-btn" data-tab="ceo">CEO / 决策层</button>
    <button class="tab-btn" data-tab="explore">技术 / 算法研究者</button>
  </div>
</div>

<div class="tab-panel active" id="tab-biz">
  <h2>从业务问题出发</h2>
  <p class="muted">选择你正在面对的挑战，直达对应的 Skill 路径与工作流。</p>
  <div class="biz-grid">
    {business_cards}
  </div>
</div>

<div class="tab-panel" id="tab-ds">
  <h2>数据科学家视角</h2>
  <div class="ds-grid">
    <div class="ds-card">
      <h3>高连接度 Skills</h3>
      <p class="muted">被最多 Skill 依赖的核心算法，学习回报最高。</p>
      <ul class="hot-list">{hot_items}</ul>
    </div>
    <div class="ds-card">
      <h3>按算法类型</h3>
      <div class="algo-tags">
        <a class="tag" href="topics/广告与投放.html">广告与投放</a>
        <a class="tag" href="topics/供应链与补货.html">供应链与补货</a>
        <a class="tag" href="topics/知识图谱与rag.html">知识图谱&amp;RAG</a>
        <a class="tag" href="topics/mas与智能体工程.html">MAS&amp;智能体</a>
        <a class="tag" href="topics/推荐与搜索.html">推荐与搜索</a>
        <a class="tag" href="topics/定价与利润.html">定价与利润</a>
        <a class="tag" href="topics/风控与合规.html">风控与合规</a>
        <a class="tag" href="topics/视觉内容生成.html">视觉内容生成</a>
      </div>
    </div>
    <div class="ds-card">
      <h3>vs 竞品对比</h3>
      <table style="font-size:12px;margin-top:8px">
        <thead><tr><th>维度</th><th>纯咨询</th><th>SaaS工具</th><th>paper2skills</th></tr></thead>
        <tbody>
          <tr><td>证据级别</td><td>经验判断</td><td>平台数据</td><td><strong>顶会论文 + A/B实测</strong></td></tr>
          <tr><td>ROI可溯源</td><td>无</td><td>部分</td><td><strong>每个Skill有ROI数字</strong></td></tr>
          <tr><td>跨境场景</td><td>通用</td><td>通用</td><td><strong>母婴跨境专属</strong></td></tr>
          <tr><td>可执行代码</td><td>无</td><td>无</td><td><strong>350个可运行模板</strong></td></tr>
          <tr><td>知识更新</td><td>项目制</td><td>产品迭代</td><td><strong>持续萃取顶会论文</strong></td></tr>
        </tbody>
      </table>
      <div class="algo-tags">
        <a class="tag" href="topics/广告与投放.html">广告与投放</a>
        <a class="tag" href="topics/供应链与补货.html">供应链与补货</a>
        <a class="tag" href="topics/知识图谱与rag.html">知识图谱&amp;RAG</a>
        <a class="tag" href="topics/mas与智能体工程.html">MAS&amp;智能体</a>
        <a class="tag" href="topics/推荐与搜索.html">推荐与搜索</a>
        <a class="tag" href="topics/定价与利润.html">定价与利润</a>
        <a class="tag" href="topics/风控与合规.html">风控与合规</a>
        <a class="tag" href="topics/视觉内容生成.html">视觉内容生成</a>
      </div>
    </div>
    <div class="ds-card">
      <h3>Skills Graph</h3>
      <p class="muted">{skill_count} 节点 · {edge_count} 关系边的知识图谱可视化。</p>
      <a class="btn-primary" href="graph/overview.html">打开图谱 →</a>
    </div>
  </div>
</div>

<div class="tab-panel" id="tab-ceo">
  <h2>AI 能力建设路线图</h2>
  <p class="muted">12 个月 3 阶段，替代 3 类岗位重复性工作，可验证 ROI > 3,000 万/年</p>
  <div class="ceo-entry">
    <div class="ceo-entry-body">
      <h3>「大促首日打 7 折清空库存，第 3 天涨价流量白白浪费」</h3>
      <p>这是供应链没有联动决策的代价。AI 路线图从这里开始。</p>
      <a class="btn-primary" href="ai-roadmap.html">查看完整路线图 →</a>
      <a class="btn-primary" href="ai-roadmap.html" onclick="window.open('ai-roadmap.html','_blank').print();return false;" style="margin-left:8px;background:#475569">下载 PDF</a>
    </div>
    <div class="ceo-phases">
      <div class="ceo-phase" style="border-color:#2563eb">
        <span style="color:#2563eb;font-weight:700">Phase 1</span> 快赢
        <p>HiFoReAd + AgentRouter + 归因修正</p>
        <strong style="color:#2563eb">800-1,900 万/年</strong>
      </div>
      <div class="ceo-phase" style="border-color:#7c3aed">
        <span style="color:#7c3aed;font-weight:700">Phase 2</span> 基础设施
        <p>产品 KG + 实验平台 + 库存池化</p>
        <strong style="color:#7c3aed">200-500 万/年（新增）</strong>
      </div>
      <div class="ceo-phase" style="border-color:#059669">
        <span style="color:#059669;font-weight:700">Phase 3</span> 护城河
        <p>AI 定价 + 内容工厂 + MAS 联动</p>
        <strong style="color:#059669">5,000 万+ 潜力</strong>
      </div>
    </div>
  </div>
</div>

<div class="tab-panel" id="tab-explore">
  <h2>按领域浏览</h2>
  <div class="metrics">
    <div><strong>{skill_count}</strong><span>Skills</span></div>
    <div><strong>{domain_count}</strong><span>领域</span></div>
    <div><strong>{edge_count}</strong><span>关系边</span></div>
    <div><strong>5</strong><span>工作流</span></div>
  </div>
  <div class="grid">{domain_cards}</div>
</div>

<script>
(function(){{
  const btns = document.querySelectorAll('#heroTabs .tab-btn');
  const panels = document.querySelectorAll('.tab-panel');
  btns.forEach(btn => btn.addEventListener('click', () => {{
    btns.forEach(b => b.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }}));
}})();
</script>
"""


# ---------------------------------------------------------------------------
# Skills Graph D3 page (Phase 3D)
# ---------------------------------------------------------------------------

def render_graph_page(skill_count: int, edge_count: int, build_ts: str = "") -> str:
    body = f"""
<h1>Skills Graph</h1>
<p class="muted">节点 {skill_count} · 边 {edge_count}　　点击节点查看详情，悬停高亮邻居，滚轮缩放。</p>
<div class="graph-controls">
  <label><input type="checkbox" id="cb-prerequisite" checked> <span class="edge-dot prereq"></span> 前置 (prerequisite)</label>
  <label><input type="checkbox" id="cb-combinable" checked> <span class="edge-dot combo"></span> 可组合 (combinable)</label>
  <label><input type="checkbox" id="cb-extension"> <span class="edge-dot ext"></span> 延伸 (extension)</label>
  <input id="graph-search" placeholder="搜索节点..." style="margin-left:16px;padding:6px 10px;border:1px solid #e5e7eb;border-radius:8px;width:200px">
</div>
<div id="graph-info" class="graph-info hidden">
  <button id="graph-info-close" style="float:right;background:none;border:none;cursor:pointer;font-size:18px">×</button>
  <h3 id="gi-title"></h3>
  <p id="gi-domain" class="muted"></p>
  <p id="gi-summary"></p>
  <a id="gi-link" href="#" class="btn-primary" target="_self">查看详情 →</a>
</div>
<svg id="graph-svg"></svg>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="../assets/graph.js?v={build_ts}"></script>
"""
    return html_page("Skills Graph", body, "../")


def build_ego_graph_js() -> str:
    """Ego graph for Skill detail pages: renders 1-hop neighbourhood in the relation panel."""
    return r"""
(function () {
  const svg = document.getElementById('ego-graph');
  if (!svg || typeof d3 === 'undefined') return;
  const centerId = svg.dataset.skill;
  if (!centerId) return;

  const W = +svg.getAttribute('width')  || 280;
  const H = +svg.getAttribute('height') || 220;

  function load(cb) {
    if (window._EGO_DATA) { cb(window._EGO_DATA); return; }
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '../assets/graph-data.json');
    xhr.onload = () => {
      try { window._EGO_DATA = JSON.parse(xhr.responseText); cb(window._EGO_DATA); }
      catch (e) { cb(null); }
    };
    xhr.onerror = () => cb(null);
    xhr.send();
  }

  load(function (raw) {
    if (!raw) return;

    const edgeCfg = {
      prerequisite: '#3b82f6',
      combinable:   '#10b981',
      extension:    '#f59e0b',
    };

    const neighborIds = new Set([centerId]);
    const egoLinks = raw.links.filter(l => {
      if (l.source === centerId || l.target === centerId) {
        neighborIds.add(l.source);
        neighborIds.add(l.target);
        return true;
      }
      return false;
    });

    if (neighborIds.size <= 1) {
      d3.select(svg).append('text')
        .attr('x', W / 2).attr('y', H / 2)
        .attr('text-anchor', 'middle').attr('fill', '#9ca3af').attr('font-size', 12)
        .text('无关联 Skill');
      return;
    }

    const egoNodes = raw.nodes
      .filter(n => neighborIds.has(n.id))
      .map(n => ({ ...n }));

    const sel = d3.select(svg).attr('viewBox', `0 0 ${W} ${H}`);

    const sim = d3.forceSimulation(egoNodes)
      .force('link', d3.forceLink(egoLinks.map(l => ({ ...l }))).id(d => d.id).distance(65).strength(0.6))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(18));

    const linkEl = sel.append('g').selectAll('line')
      .data(egoLinks)
      .join('line')
      .attr('stroke', d => edgeCfg[d.type] || '#94a3b8')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.7);

    const nodeEl = sel.append('g').selectAll('g')
      .data(egoNodes)
      .join('g')
      .attr('cursor', d => d.id === centerId ? 'default' : 'pointer')
      .on('click', (e, d) => {
        if (d.id !== centerId) window.location.href = `${d.id}.html`;
      });

    nodeEl.append('circle')
      .attr('r', d => d.id === centerId ? 10 : 7)
      .attr('fill', d => d.id === centerId ? '#2563eb' : '#7c3aed')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('fill-opacity', d => d.id === centerId ? 1 : 0.75);

    nodeEl.append('text')
      .attr('dy', d => d.id === centerId ? -13 : -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', d => d.id === centerId ? 11 : 9)
      .attr('fill', d => d.id === centerId ? '#1e40af' : '#374151')
      .attr('font-weight', d => d.id === centerId ? '700' : '400')
      .text(d => {
        const label = d.id.replace(/^Skill-/, '').replace(/-/g, ' ');
        return label.length > 18 ? label.slice(0, 17) + '…' : label;
      });

    nodeEl.append('title').text(d => d.id);

    sim.on('tick', () => {
      linkEl
        .attr('x1', d => Math.max(10, Math.min(W - 10, d.source.x)))
        .attr('y1', d => Math.max(10, Math.min(H - 10, d.source.y)))
        .attr('x2', d => Math.max(10, Math.min(W - 10, d.target.x)))
        .attr('y2', d => Math.max(10, Math.min(H - 10, d.target.y)));
      nodeEl.attr('transform', d =>
        `translate(${Math.max(10, Math.min(W - 10, d.x))},${Math.max(12, Math.min(H - 8, d.y))})`
      );
    });

    sim.stop();
    for (let i = 0; i < 120; i++) sim.tick();
    sim.on('tick', () => {
      linkEl
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeEl.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    sim.restart();
  });
})();
""".strip()


def build_graph_js() -> str:
    """Return the D3 force graph JS bundle."""
    return r"""
document.addEventListener('DOMContentLoaded', function () {
(function () {
  const DATA = window.PLAYBOOK_DATA || {};
  const skills = DATA.skills || [];
  const skillMap = {};
  skills.forEach(s => { skillMap[s.skill_id] = s; });

  // Load graph-data.json via XHR (works for file:// and http://)
  function loadGraphData(cb) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '../assets/graph-data.json');
    xhr.onload = () => {
      try { cb(JSON.parse(xhr.responseText)); } catch (e) { cb(null); }
    };
    xhr.onerror = () => cb(null);
    xhr.send();
  }

  loadGraphData(function (raw) {
    if (!raw) { document.getElementById('graph-svg').insertAdjacentHTML('beforebegin', '<p class="muted">无法加载图谱数据。</p>'); return; }

    const nodes = raw.nodes.map(n => ({ ...n }));
    const nodeIdSet = new Set(nodes.map(n => n.id));
    const links = raw.links
      .map(l => ({ ...l }))
      .filter(l => nodeIdSet.has(l.source) && nodeIdSet.has(l.target));

    // Domain colour palette (Tableau-10 extended)
    const domains = [...new Set(nodes.map(n => n.domain))].sort();
    const colour = d3.scaleOrdinal(d3.schemeTableau10.concat(d3.schemePastel1)).domain(domains);

    // Degree map for node sizing
    const degree = {};
    links.forEach(l => {
      degree[l.source] = (degree[l.source] || 0) + 1;
      degree[l.target] = (degree[l.target] || 0) + 1;
    });
    const maxDeg = Math.max(...Object.values(degree), 1);
    const rScale = d3.scaleSqrt().domain([0, maxDeg]).range([4, 14]);

    const svg = d3.select('#graph-svg');
    const W = svg.node().parentElement.clientWidth || 1100;
    const H = Math.max(600, window.innerHeight - 240);
    svg.attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`);

    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom().scaleExtent([0.1, 6]).on('zoom', e => g.attr('transform', e.transform)));

    // Edge type → display config
    const edgeCfg = {
      prerequisite: { stroke: '#3b82f6', dasharray: null, width: 1.5 },
      combinable:   { stroke: '#10b981', dasharray: '5,3',   width: 1 },
      extension:    { stroke: '#f59e0b', dasharray: '2,4',   width: 1 },
    };

    // Visibility state
    const visible = { prerequisite: true, combinable: true, extension: false };
    document.querySelectorAll('.graph-controls input[type=checkbox]').forEach(cb => {
      cb.addEventListener('change', () => {
        visible[cb.id.replace('cb-', '')] = cb.checked;
        updateEdgeVisibility();
      });
    });

    function updateEdgeVisibility() {
      linkEl.style('display', d => visible[d.type] ? null : 'none');
    }

    // Simulation — only prerequisite + combinable edges by default for perf
    const activeLinks = links.filter(l => l.type === 'prerequisite' || l.type === 'combinable');
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(activeLinks).id(d => d.id).distance(60).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide().radius(d => rScale(degree[d.id] || 0) + 4));

    // Draw all edges (extension hidden initially)
    const linkEl = g.append('g').selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => (edgeCfg[d.type] || edgeCfg.prerequisite).stroke)
      .attr('stroke-width', d => (edgeCfg[d.type] || edgeCfg.prerequisite).width)
      .attr('stroke-dasharray', d => (edgeCfg[d.type] || edgeCfg.prerequisite).dasharray)
      .attr('stroke-opacity', 0.5)
      .style('display', d => visible[d.type] ? null : 'none');

    // Draw nodes
    const nodeEl = g.append('g').selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => rScale(degree[d.id] || 0))
      .attr('fill', d => colour(d.domain))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Hover: highlight 1-hop neighbourhood
    const neighborSet = new Set();
    nodeEl
      .on('mouseover', (e, d) => {
        neighborSet.clear();
        neighborSet.add(d.id);
        links.forEach(l => {
          const src = typeof l.source === 'object' ? l.source.id : l.source;
          const tgt = typeof l.target === 'object' ? l.target.id : l.target;
          if (src === d.id || tgt === d.id) { neighborSet.add(src); neighborSet.add(tgt); }
        });
        nodeEl.attr('opacity', n => neighborSet.has(n.id) ? 1 : 0.15);
        linkEl.attr('stroke-opacity', l => {
          const src = typeof l.source === 'object' ? l.source.id : l.source;
          const tgt = typeof l.target === 'object' ? l.target.id : l.target;
          return (neighborSet.has(src) && neighborSet.has(tgt)) ? 0.8 : 0.05;
        });
      })
      .on('mouseout', () => {
        nodeEl.attr('opacity', 1);
        linkEl.attr('stroke-opacity', 0.5);
      })
      .on('click', (e, d) => showInfo(d));

    // Tick
    sim.on('tick', () => {
      linkEl
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeEl.attr('cx', d => d.x).attr('cy', d => d.y);
    });

    // Info panel
    const infoPanel = document.getElementById('graph-info');
    document.getElementById('graph-info-close').addEventListener('click', () => infoPanel.classList.add('hidden'));

    function showInfo(d) {
      const sk = skillMap[d.id];
      document.getElementById('gi-title').textContent = sk ? sk.title : d.id;
      document.getElementById('gi-domain').textContent = d.domain || '';
      document.getElementById('gi-summary').textContent = sk ? (sk.problem_solved || sk.algorithm_summary || '') : '';
      const link = document.getElementById('gi-link');
      link.href = `../skills/${d.id}.html`;
      infoPanel.classList.remove('hidden');
    }

    // Search
    document.getElementById('graph-search').addEventListener('input', function () {
      const q = this.value.trim().toLowerCase();
      if (!q) { nodeEl.attr('opacity', 1); return; }
      nodeEl.attr('opacity', d => (d.id.toLowerCase().includes(q) || (skillMap[d.id] && skillMap[d.id].title.toLowerCase().includes(q))) ? 1 : 0.1);
    });

    updateEdgeVisibility();
  });
})();
});
"""


# ---------------------------------------------------------------------------
# CSS + JS assets (Phase 3C/D additions merged)
# ---------------------------------------------------------------------------

def build_css() -> str:
    return """
/* ═══════════════════════════════════════════════════════
   paper2skills Playbook — Design System v4
   Apple Standard · SF Pro · PingFang SC · Warm White
   ═══════════════════════════════════════════════════════ */

/* ── Design Tokens ── */
:root {
  /* Page backgrounds */
  --bg:          #FAF7F5;
  --bg-warm:     #F5EFE9;
  --panel:       #FFFFFF;
  --panel-2:     #F9F7F5;
  --panel-3:     #F2EDE8;

  /* Text — Apple standard */
  --ink:         #1d1d1f;
  --ink-2:       #3d3d3f;
  --muted:       #86868b;

  /* Borders */
  --line:        #EDE6DF;
  --line-strong: #E0D8CF;

  /* Brand — deep rose / coral */
  --accent:      #C25B6E;
  --accent-dark: #A34759;
  --accent-light:#F9E8EC;
  --accent-bg:   #F9E8EC;
  --accent2:     #7c6f64;
  --accent2-bg:  #f0ede9;

  /* Semantic — iOS colors */
  --green:       #34C759;
  --green-bg:    #F0FBF3;
  --green-dark:  #1a6b34;
  --amber:       #FF9500;
  --amber-bg:    #FFF8EE;
  --amber-dark:  #7a4a1e;
  --red:         #FF3B30;
  --red-bg:      #FFF0EF;

  /* Phase — warm triad */
  --phase-1:     #C25B6E; --phase-1-bg: #F9E8EC; --phase-1-muted: #d9909c;
  --phase-2:     #7c6f64; --phase-2-bg: #f0ede9; --phase-2-muted: #b0a69e;
  --phase-3:     #1a6b34; --phase-3-bg: #F0FBF3; --phase-3-muted: #5fa87a;

  /* Tag */
  --tag-bg:      #F2EDE8;
  --tag-ink:     #3d3d3f;
  --tag-topic-bg:#F0FBF3;
  --tag-topic-ink:#1a6b34;

  /* Navigation */
  --nav-bg:         #FFFFFF;
  --nav-border:     #EDE6DF;
  --nav-text:       #5a5450;
  --nav-text-hover: #1d1d1f;
  --nav-active-bg:  #F9E8EC;
  --nav-active-text:#C25B6E;
  --nav-highlight:  #C25B6E;
  --topbar-height:  56px;

  /* Radius */
  --r-xs:  4px;
  --r-sm:  6px;
  --r-md:  9px;
  --r-lg:  12px;
  --r-xl:  18px;
  --r-2xl: 20px;
  --r-full:999px;

  /* Shadow — very light, Apple style */
  --shadow-xs:    0 1px 4px rgba(0,0,0,0.04);
  --shadow-sm:    0 2px 8px rgba(0,0,0,0.06);
  --shadow-md:    0 1px 4px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.06);
  --shadow-lg:    0 8px 32px rgba(0,0,0,0.12);
  --shadow-hover: 0 8px 32px rgba(0,0,0,0.12);
  --shadow-accent:0 4px 16px rgba(194,91,110,.20);

  /* Motion — cubic-bezier Apple easing */
  --t:     .15s ease;
  --t-card:.25s cubic-bezier(0.4,0,0.2,1);
  --t-slow:.25s ease;

  /* Typography — system font stack (SF Pro + PingFang SC, no CDN) */
  --font: -apple-system, "SF Pro Display", "SF Pro Text", BlinkMacSystemFont,
          "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", "Microsoft YaHei",
          sans-serif;
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font);
  font-size: 15px;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  letter-spacing: 0;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
/* Card links must never show underline — override global hover */
.card:hover, .skill-card:hover, .biz-card:hover, .domain-card:hover,
.metric-card:hover, .wf-card:hover, .agent-card:hover, .ds-card:hover,
a.card:hover, a.skill-card:hover, a.biz-card:hover, a.domain-card:hover,
a.metric-card:hover, a.wf-card:hover {
  text-decoration: none;
}
img { max-width: 100%; }
strong { font-weight: 600; }
p { margin: 0 0 14px; }
p:last-child { margin-bottom: 0; }

/* ── Top Bar — minimalist brand bar ── */
.topbar {
  position: sticky; top: 0; z-index: 200;
  display: flex; align-items: center;
  height: var(--topbar-height);
  padding: 0 24px 0 0;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(12px) saturate(180%);
  -webkit-backdrop-filter: blur(12px) saturate(180%);
  border-bottom: 1px solid var(--nav-border);
  box-shadow: 0 1px 0 rgba(0,0,0,0.05);
}
.brand {
  display: flex; align-items: center; gap: 10px;
  text-decoration: none; color: var(--ink);
  flex-shrink: 0;
  padding: 0 24px;
  height: 100%;
  border-right: 1px solid var(--nav-border);
}
.brand:hover { text-decoration: none; }
.brand-icon {
  width: 30px; height: 30px; border-radius: 8px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
  color: #fff; font-weight: 800; font-size: 13px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; letter-spacing: -.02em;
}
.brand-name {
  font-weight: 700; font-size: 14.5px; letter-spacing: -.025em;
  color: var(--ink); line-height: 1;
}
.brand-tag {
  display: block; font-size: 10px; font-weight: 500;
  color: var(--muted); letter-spacing: .04em;
  text-transform: uppercase; margin-top: 2px;
}
.topbar-right {
  margin-left: auto; display: flex; align-items: center; gap: 12px;
}
#global-search {
  width: min(320px, 28vw); padding: 7px 16px 7px 36px;
  border-radius: var(--r-full);
  border: 1.5px solid var(--line);
  background: var(--panel-2);
  color: var(--ink); font-size: 13px;
  font-family: var(--font);
  transition: border-color var(--t), background var(--t), box-shadow var(--t);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2386868b' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: 12px center;
}
#global-search::placeholder { color: var(--muted); }
#global-search:hover { border-color: var(--line-strong); background: var(--panel); }
#global-search:focus {
  outline: none; border-color: var(--accent); background: var(--panel);
  box-shadow: 0 0 0 3px rgba(194,91,110,.10);
}
.topbar-cta {
  display: inline-flex; align-items: center;
  padding: 7px 16px; border-radius: var(--r-full);
  background: var(--accent); color: #fff !important;
  font-size: 13px; font-weight: 600; letter-spacing: -.01em;
  text-decoration: none !important;
  transition: background var(--t), box-shadow var(--t), transform var(--t);
  white-space: nowrap; flex-shrink: 0;
}
.topbar-cta:hover { background: var(--accent-dark); box-shadow: var(--shadow-accent); transform: translateY(-1px); }
.topbar-cta.active { background: var(--accent-dark); }

/* ── Topbar AI Button ── */
/* ── Hamburger ── */
.hamburger {
  display: none; flex-direction: column; justify-content: center;
  gap: 5px; width: 40px; height: var(--topbar-height); padding: 0 10px;
  background: none; border: none; border-right: 1px solid var(--nav-border);
  cursor: pointer; flex-shrink: 0;
}
.hamburger span {
  display: block; height: 1.5px; background: var(--muted);
  border-radius: 2px; transition: transform var(--t), opacity var(--t);
}
.hamburger.open span:nth-child(1) { transform: translateY(6.5px) rotate(45deg); }
.hamburger.open span:nth-child(2) { opacity: 0; }
.hamburger.open span:nth-child(3) { transform: translateY(-6.5px) rotate(-45deg); }
.hamburger:hover span { background: var(--ink); }

/* ── Layout ── */
.layout {
  display: grid;
  grid-template-columns: 240px 1fr;
  min-height: calc(100vh - var(--topbar-height));
}

/* ── Sidebar — premium left nav ── */
.sidebar {
  display: flex; flex-direction: column;
  background: var(--panel);
  border-right: 1px solid var(--line);
  position: sticky; top: var(--topbar-height);
  height: calc(100vh - var(--topbar-height));
  overflow-y: auto; overflow-x: hidden;
}
.sidebar::-webkit-scrollbar { width: 3px; }
.sidebar::-webkit-scrollbar-track { background: transparent; }
.sidebar::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 3px; }

/* Sidebar top (nav items) */
.sb-top { flex: 1; padding: 16px 10px 12px; display: flex; flex-direction: column; gap: 4px; }

/* Sidebar section group */
.sb-section { margin-bottom: 4px; }
.sb-label {
  font-size: 10.5px; font-weight: 700; letter-spacing: .07em;
  text-transform: uppercase; color: var(--muted);
  padding: 6px 10px 4px; margin: 0; user-select: none;
}
.sb-links { display: flex; flex-direction: column; gap: 1px; }

/* Sidebar link */
.sidebar a {
  display: flex; align-items: center; gap: 9px;
  color: var(--ink-2); text-decoration: none;
  padding: 8px 10px; border-radius: var(--r-lg);
  font-size: 13.5px; font-weight: 450;
  line-height: 1.3;
  transition: background var(--t), color var(--t);
  position: relative;
}
.sidebar a:hover {
  background: var(--panel-2); color: var(--ink);
  text-decoration: none;
}
.sidebar a.active, .sidebar a[aria-current="page"] {
  background: var(--accent-light);
  color: var(--accent);
  font-weight: 600;
}
.sidebar a.active::before, .sidebar a[aria-current="page"]::before {
  content: '';
  position: absolute; left: 0; top: 4px; bottom: 4px;
  width: 3px; border-radius: 0 3px 3px 0;
  background: var(--accent);
}
.sbl-icon {
  width: 20px; height: 20px; border-radius: 5px;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; flex-shrink: 0; opacity: 0.5;
  transition: opacity var(--t);
}
.sidebar a:hover .sbl-icon, .sidebar a.active .sbl-icon, .sidebar a[aria-current="page"] .sbl-icon {
  opacity: 1;
}
.sbl-text { flex: 1; }

/* Sidebar bottom — AI Chat */
.sb-bottom {
  padding: 12px 10px;
  border-top: 1px solid var(--line);
}

/* ── Content area ── */
.content { padding: 36px 44px; max-width: 1400px; overflow-x: hidden; }
.mobile-nav-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(29,29,27,.5); backdrop-filter: blur(4px); z-index: 190;
}
.mobile-nav-overlay.show { display: block; }

/* ── Typography — 4-level Apple scale ── */
.content h1 {
  font-size: 30px; font-weight: 700; letter-spacing: -.04em;
  line-height: 1.1; margin: 0 0 12px; color: var(--ink);
}
.content h2 {
  font-size: 18px; font-weight: 650; letter-spacing: -.02em;
  margin: 28px 0 12px; color: var(--ink); line-height: 1.3;
}
.content h2:first-child { margin-top: 0; }
.content h3 {
  font-size: 16px; font-weight: 600; letter-spacing: -.01em;
  margin: 24px 0 8px; color: var(--ink); line-height: 1.3;
}
.content h4 {
  font-size: 13.5px; font-weight: 600; letter-spacing: 0;
  margin: 18px 0 6px; color: var(--ink-2);
}
.lead {
  font-size: 15.5px; color: var(--muted); margin: 0 0 24px;
  line-height: 1.75; letter-spacing: -.005em;
}
.muted { color: var(--muted); }
.section-eyebrow {
  font-size: 11px; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted);
  margin: 0 0 8px; display: block;
}
/* Section head container */
.section-head { margin-bottom: 24px; }
.section-head h2 { margin: 0 0 6px; border: none; padding: 0; }
.section-head p { margin: 0; font-size: 14px; color: var(--muted); line-height: 1.6; }

/* ── Hero / Tabs ── */
.hero { margin-bottom: 8px; }
.hero h1 { margin: 0 0 10px; }
.hero-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 24px; }
.tab-btn {
  padding: 8px 20px;
  border: 1.5px solid var(--line-strong);
  background: var(--panel);
  border-radius: var(--r-full);
  font-size: 13px; font-weight: 500;
  font-family: var(--font);
  cursor: pointer; color: var(--muted);
  transition: background var(--t), color var(--t), border-color var(--t), box-shadow var(--t);
}
.tab-btn:hover:not(.active) {
  background: rgba(0,0,0,0.04); color: var(--ink); border-color: var(--line-strong);
}
.tab-btn.active {
  background: var(--accent); border-color: var(--accent);
  color: #fff; font-weight: 600; box-shadow: var(--shadow-accent);
}
.tab-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ── Buttons ── */
.btn-primary {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  padding: 10px 22px;
  background: var(--ink); color: #fff;
  border-radius: var(--r-md); text-decoration: none;
  font-size: 13px; font-weight: 600; font-family: var(--font);
  letter-spacing: .02em; text-transform: uppercase;
  border: none; cursor: pointer;
  transition: background var(--t-card), box-shadow var(--t-card), transform var(--t-card);
}
.btn-primary:hover { background: var(--ink-2); box-shadow: var(--shadow-lg); transform: translateY(-1px); }
.btn-primary:active { transform: translateY(0); box-shadow: none; }
.btn-primary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.btn-primary.accent { background: var(--accent); }
.btn-primary.accent:hover { background: var(--accent-dark); box-shadow: var(--shadow-accent); }
.btn-secondary {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 10px 20px;
  background: transparent; color: var(--accent);
  border: 1.5px solid var(--accent);
  border-radius: var(--r-full); text-decoration: none;
  font-size: 14px; font-weight: 600; font-family: var(--font);
  letter-spacing: -.01em; cursor: pointer;
  transition: background var(--t), color var(--t), box-shadow var(--t);
}
.btn-secondary:hover {
  background: var(--accent-light);
  text-decoration: none;
  box-shadow: var(--shadow-xs);
}
.btn-secondary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ── Text icon badge ── */
.icon-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 44px; height: 44px;
  background: var(--accent-light); color: var(--accent);
  border-radius: var(--r-md); font-size: 10px; font-weight: 700;
  letter-spacing: .04em; flex-shrink: 0; font-family: var(--font);
  text-transform: uppercase;
}
.icon-badge.warm { background: var(--amber-bg); color: var(--amber-dark); }
.icon-badge.green { background: var(--green-bg); color: var(--green-dark); }
.icon-badge.red   { background: var(--red-bg); color: var(--red); }
.icon-badge.dark  { background: var(--panel-3); color: var(--ink); }
.icon-badge.lg {
  width: 48px; height: 48px; font-size: 11px; border-radius: var(--r-lg);
}

/* ── Business Entry Cards — Apple card layout ── */
.biz-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
  align-items: stretch;
}
.biz-card {
  display: flex;
  flex-direction: column;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 24px;
  text-decoration: none; color: var(--ink);
  box-shadow: var(--shadow-md);
  transition: transform .22s cubic-bezier(0.4,0,0.2,1),
              box-shadow .22s cubic-bezier(0.4,0,0.2,1),
              border-color .15s ease;
}
.biz-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 36px rgba(0,0,0,.13);
  border-color: var(--accent);
  text-decoration: none;
}
.biz-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.biz-icon {
  width: 44px; height: 44px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-light); color: var(--accent);
  border-radius: var(--r-md); font-size: 10px; font-weight: 700;
  letter-spacing: .04em; text-transform: uppercase; font-family: var(--font);
}
.biz-card-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}
.biz-body { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.biz-body strong {
  display: block;
  font-size: 15px; font-weight: 600;
  letter-spacing: -.01em; line-height: 1.4;
  color: var(--ink);
  word-break: keep-all;
  overflow-wrap: break-word;
}
.biz-body p {
  margin: 10px 0 0; font-size: 13.5px;
  color: var(--muted); line-height: 1.65;
  flex: 1;
  word-break: keep-all; overflow-wrap: break-word;
  display: -webkit-box; -webkit-line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden;
}
.biz-tag {
  flex-shrink: 1;
  min-width: 0;
  font-size: 11px; font-weight: 600;
  background: var(--panel-3); color: var(--ink-2);
  padding: 3px 10px; border-radius: var(--r-full);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 190px;
}

/* ── DS Cards ── */
.ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin: 16px 0; align-items: stretch; }
.ds-card { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 22px; box-shadow: var(--shadow-md); }
.ds-card h3 { margin: 0 0 10px; font-size: 14.5px; font-weight: 700; }
.hot-list { padding: 0; margin: 8px 0 0; list-style: none; display: flex; flex-direction: column; gap: 5px; }
.hot-list li { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 13.5px; }
.hot-list a { color: var(--ink-2); text-decoration: none; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
.hot-list a:hover { color: var(--accent); text-decoration: none; }
.hot-list .roi-badge { flex-shrink: 0; }
.algo-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.algo-tags .tag { text-decoration: none; }
.ceo-entry { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin: 20px 0; align-items: start; }
.ceo-entry-body h3 { margin: 0 0 10px; font-size: 17px; font-weight: 700; }
.ceo-entry-body p { color: var(--muted); margin: 0 0 16px; font-size: 13.5px; line-height: 1.65; }
.ceo-phases { display: flex; flex-direction: column; gap: 10px; }
.ceo-phase { background: var(--panel); border-left: 3px solid; border-radius: 0 var(--r-lg) var(--r-lg) 0; padding: 12px 16px; font-size: 13px; box-shadow: var(--shadow-xs); }
.ceo-phase p { margin: 4px 0; color: var(--muted); }

/* ── Metrics ── */
.metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 14px; margin: 16px 0; }
.metrics > div { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 20px; box-shadow: var(--shadow-md); }
.metrics strong { display: block; font-size: 32px; font-weight: 700; letter-spacing: -.04em; color: var(--accent); }
.metrics span { color: var(--muted); font-size: 12.5px; }

/* ── Domain / Topic Grids ── */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; margin: 20px 0; align-items: stretch; }
.metric-card, .domain-card {
  display: flex; flex-direction: column;
  background: var(--panel);
  border: 1px solid var(--line); border-radius: 20px;
  padding: 18px 20px; text-decoration: none; color: var(--ink);
  box-shadow: var(--shadow-sm); min-height: 72px;
  transition: transform .22s cubic-bezier(0.4,0,0.2,1),
              box-shadow .22s cubic-bezier(0.4,0,0.2,1),
              border-color .15s ease;
}
.metric-card:hover, .domain-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 36px rgba(0,0,0,.13);
  border-color: var(--accent);
  text-decoration: none;
}
.metric-card strong { display: block; font-weight: 600; font-size: 14px; letter-spacing: -.01em; }
.metric-card span { color: var(--muted); font-size: 12.5px; }
.domain-card strong {
  font-size: 14px; font-weight: 650; letter-spacing: -.01em;
  color: var(--ink); display: block; margin-bottom: 4px;
}
.domain-card span { font-size: 12px; color: var(--muted); }

/* ── Skill Cards — Apple card style ── */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; align-items: stretch; }
.skill-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  border-top: 3px solid var(--accent);
  padding: 22px;
  display: flex; flex-direction: column;
  box-shadow: var(--shadow-sm);
  transition: transform .22s cubic-bezier(0.4,0,0.2,1),
              box-shadow .22s cubic-bezier(0.4,0,0.2,1),
              border-color .15s ease;
  color: var(--ink); text-decoration: none;
}
.skill-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 36px rgba(0,0,0,.13);
  border-color: var(--accent);
  text-decoration: none;
}
.skill-card h3 {
  margin: 0; font-size: 15px; font-weight: 600; letter-spacing: -.01em;
  line-height: 1.35;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.skill-card h3 a { color: var(--ink); text-decoration: none; }
.skill-card h3 a:hover { color: var(--accent); }
.skill-card p { margin: 0; font-size: 13.5px; color: var(--muted); line-height: 1.7; }
.card-badges { display: flex; gap: 6px; flex-wrap: wrap; }
/* .card generic */
.card {
  transition: transform .22s cubic-bezier(0.4,0,0.2,1),
              box-shadow .22s cubic-bezier(0.4,0,0.2,1),
              border-color .15s ease;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 36px rgba(0,0,0,.13);
  border-color: rgba(194,91,110,.5);
  text-decoration: none;
}
/* .wf-card inherits .card hover; explicit for specificity */
.wf-card {
  transition: transform .22s cubic-bezier(0.4,0,0.2,1),
              box-shadow .22s cubic-bezier(0.4,0,0.2,1),
              border-color .15s ease;
}
.wf-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 36px rgba(0,0,0,.13);
  border-color: rgba(194,91,110,.5);
  text-decoration: none;
}

/* ── sc-* sub-elements for new skill-card design ── */
.sc-domain {
  font-size: 11px; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; color: var(--muted);
  margin-bottom: 2px; flex-shrink: 0;
}
.sc-title {
  font-size: 14.5px; font-weight: 650; line-height: 1.4;
  letter-spacing: -.01em; color: var(--ink);
  margin: 0 0 6px; flex-shrink: 0;
  display: -webkit-box; -webkit-line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden;
}
.sc-desc {
  font-size: 13px; color: var(--muted); line-height: 1.65;
  margin: 0 0 10px; flex: 1;
  display: -webkit-box; -webkit-line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden;
}
.sc-footer {
  display: flex; align-items: center; gap: 6px; margin-top: auto; flex-shrink: 0;
  min-height: 22px;
}
.sc-roi {
  font-size: 11px; font-weight: 700;
  background: var(--green-bg); color: var(--green-dark);
  padding: 2px 8px; border-radius: var(--r-full);
  white-space: nowrap; max-width: 140px;
  overflow: hidden; text-overflow: ellipsis;
}
.sc-diff {
  font-size: 11px; color: var(--muted);
  padding: 2px 8px; background: var(--panel-2);
  border-radius: var(--r-full); white-space: nowrap;
}

/* ── Badges — Apple pill style ── */
.roi-badge {
  display: inline-block; padding: 3px 9px; border-radius: var(--r-full);
  font-size: 11px; font-weight: 700; letter-spacing: .02em;
  background: var(--green-bg); color: var(--green-dark);
  border: none; white-space: nowrap;
}
.diff-badge {
  display: inline-block; padding: 3px 9px; border-radius: var(--r-full);
  font-size: 11px; font-weight: 500; letter-spacing: .02em;
  background: var(--panel-2); color: var(--muted);
  border: none; white-space: nowrap;
}

/* ── Tags ── */
.tag {
  display: inline-block; padding: 3px 9px;
  background: var(--tag-bg); border-radius: var(--r-full);
  font-size: 11.5px; color: var(--tag-ink);
  text-decoration: none; font-weight: 500;
  white-space: nowrap; letter-spacing: .01em;
}
.tag:hover { background: var(--line-strong); text-decoration: none; }
.tag.topic { background: var(--tag-topic-bg); color: var(--tag-topic-ink); }
.tag.topic:hover { background: #d0eedd; }
.tag-row { margin: 8px 0 16px; }

/* ── Skill Detail Page ── */
.breadcrumbs { color: var(--muted); margin-bottom: 14px; font-size: 12.5px; }
.breadcrumbs a { color: var(--accent); text-decoration: none; }
.breadcrumbs a:hover { text-decoration: underline; }
.two-col { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 28px; margin-top: 20px; }
.relation-panel {
  position: sticky; top: calc(var(--topbar-height) + 16px); align-self: start;
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 20px; padding: 20px; box-shadow: var(--shadow-md);
}
.relation-panel h2 { margin: 0 0 12px; font-size: 14px; font-weight: 700; }
.relation-panel h3 { margin: 16px 0 6px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }
.relation-panel ul { padding: 0; margin: 0; list-style: none; display: flex; flex-direction: column; }
.relation-panel ul li { border-bottom: 1px solid var(--line); }
.relation-panel ul li:last-child { border-bottom: none; }
.relation-panel ul li a { display: block; padding: 5px 0; font-size: 12.5px; color: var(--accent); text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.relation-panel ul li a:hover { text-decoration: underline; }
#ego-graph { display: block; width: 100%; height: auto; min-height: 180px; border-radius: var(--r-md); background: var(--panel-2); border: 1px solid var(--line); margin-bottom: 8px; }
.ego-legend { font-size: 11.5px; color: var(--muted); display: flex; align-items: center; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }
.roi-panel { display: flex; gap: 20px; flex-wrap: wrap; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 14px 18px; margin: 12px 0 20px; }
.roi-item { display: flex; flex-direction: column; }
.roi-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.roi-value { font-size: 16px; font-weight: 700; margin-top: 2px; color: var(--ink); }

/* ── Search ── */
.search-results {
  position: absolute; top: var(--topbar-height); left: 0; right: 0; z-index: 300;
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 0 0 var(--r-xl) var(--r-xl);
  box-shadow: var(--shadow-lg);
  max-height: 480px; overflow-y: auto; margin: 0;
}
.search-results.hidden { display: none; }
.search-results .result {
  display: block; padding: 11px 20px;
  text-decoration: none; color: var(--ink);
  border-bottom: 1px solid var(--line); font-size: 13.5px;
  transition: background var(--t);
}
.search-results .result:hover { background: var(--accent-bg); }
.search-results .result:last-child { border-bottom: none; }
.search-results .result strong { color: var(--accent); font-weight: 600; }

/* ── Workflow ── */
.wf-meta { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 12px 0; }
.wf-entry-question { font-size: 15px; font-weight: 600; }
.wf-outcomes { background: var(--green-bg); border: 1px solid #b8dfc9; border-radius: var(--r-lg); padding: 14px 18px; margin: 16px 0; }
.wf-outcomes h3 { margin: 0 0 8px; font-size: 13px; color: var(--green-dark); font-weight: 700; }
.wf-outcomes ul { margin: 0; padding-left: 18px; }
.wf-outcomes li { font-size: 13.5px; margin-bottom: 4px; }
.wf-tree { margin-top: 24px; }
.wf-step {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--r-lg); padding: 18px 20px; margin-bottom: 12px;
  border-left: 3px solid var(--accent);
}
.wf-step-name { font-weight: 700; font-size: 14.5px; margin-bottom: 8px; }
.wf-question { font-size: 14px; margin: 8px 0; color: var(--accent); font-weight: 500; }
.wf-context { font-size: 13px; color: var(--muted); margin-bottom: 12px; line-height: 1.6; }
.wf-branches { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
.wf-branch { border: 1px solid var(--line); border-radius: var(--r-md); overflow: hidden; }
.wf-branch > summary {
  padding: 10px 14px; cursor: pointer; font-size: 13.5px; font-weight: 500;
  background: var(--panel-2); list-style: none;
  position: relative; padding-left: 36px;
  transition: background var(--t);
}
.wf-branch > summary:hover { background: var(--accent-bg); }
.wf-branch > summary::before {
  content: ''; position: absolute; left: 14px; top: 50%;
  width: 6px; height: 6px;
  border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted);
  transform: translateY(-65%) rotate(-45deg); transition: transform var(--t);
}
.wf-branch[open] > summary::before { transform: translateY(-35%) rotate(45deg); }
.wf-condition { color: var(--ink); }
.wf-branch-skills { display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 14px; background: var(--panel); }
.wf-skill-chip {
  display: flex; flex-direction: column;
  background: var(--accent-bg); border-radius: var(--r-md);
  padding: 8px 12px; text-decoration: none; color: var(--ink);
  min-width: 160px;
  border: 1px solid transparent;
  transition: box-shadow var(--t), transform var(--t), border-color var(--t);
}
.wf-skill-chip:hover { box-shadow: var(--shadow-sm); transform: translateY(-1px); border-color: var(--accent); }
.wf-skill-chip.missing { opacity: .45; cursor: default; }
.chip-name { font-size: 12.5px; font-weight: 700; color: var(--accent); }
.chip-role { font-size: 12px; color: var(--muted); margin-top: 2px; }

/* ── Graph ── */
#graph-svg { width: 100%; display: block; background: var(--bg); border-radius: var(--r-xl); border: 1px solid var(--line); }
.graph-controls { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; font-size: 13.5px; }
.graph-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; font-weight: 500; }
.edge-dot { width: 14px; height: 4px; border-radius: 2px; display: inline-block; }
.edge-dot.prereq { background: #C25B6E; }
.edge-dot.combo  { background: #1a6b34; }
.edge-dot.ext    { background: #FF9500; }
.graph-info {
  position: fixed; top: 80px; right: 24px; z-index: 20;
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--r-xl); padding: 20px; width: 280px;
  box-shadow: var(--shadow-lg);
}
.graph-info.hidden { display: none; }
.graph-info h3 { margin: 0 28px 10px 0; font-size: 14px; font-weight: 700; }
.graph-info p { margin: 4px 0; font-size: 13px; }

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--line); }
th { background: var(--panel-2); font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
tr:hover td { background: var(--bg); }

/* ── Code Preview ── */
.code-preview {
  background: #1a1916; color: #e8e0d4;
  border-radius: var(--r-lg); padding: 18px 20px;
  overflow-x: auto; overflow-y: auto;
  font-size: 12.5px; line-height: 1.65;
  font-family: "JetBrains Mono", "Fira Code", Menlo, monospace;
  max-height: 420px; margin: 12px 0; white-space: pre;
  border: 1px solid rgba(255,255,255,.06);
}

/* ── Business Context Panel ── */
.biz-ctx-panel {
  background: var(--panel);
  border: 1.5px solid var(--line);
  border-left: 4px solid var(--accent);
  border-radius: var(--r-xl);
  padding: 18px 22px;
  margin: 14px 0 22px;
}
.biz-ctx-header {
  font-size: 11px; font-weight: 800; letter-spacing: .08em;
  text-transform: uppercase; color: var(--accent);
  margin-bottom: 14px;
}
.biz-ctx-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px 24px;
}
.biz-ctx-item { display: flex; flex-direction: column; gap: 3px; }
.biz-ctx-full { grid-column: 1 / -1; }
.biz-ctx-label {
  font-size: 10.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted);
}
.biz-ctx-value { font-size: 13.5px; color: var(--ink-2); line-height: 1.55; }
.biz-ctx-secondary { color: var(--muted); font-weight: 400; }
.biz-ctx-outcome { color: var(--green-dark); font-weight: 500; }
.biz-pain-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.biz-pain-tag {
  font-size: 12px; padding: 3px 10px;
  background: var(--panel-2);
  border: 1px solid var(--line-strong);
  border-radius: var(--r-full);
  color: var(--ink-2);
  font-weight: 500;
}
@media (max-width: 600px) {
  .biz-ctx-grid { grid-template-columns: 1fr; }
}

/* ── Filter Bar ── */
.filter-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 16px 0 8px; }
/* ── Handbook Uplinks ── */
.hb-uplinks { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 8px 0 12px; }
.hb-uplinks-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.hb-uplink {
  font-size: 12px; font-weight: 600;
  padding: 4px 12px; border-radius: var(--r-full);
  background: var(--accent-bg); color: var(--accent);
  text-decoration: none; border: 1px solid rgba(156,84,85,.2);
  transition: background var(--t), box-shadow var(--t);
}
.hb-uplink:hover { background: #f0e0e0; box-shadow: var(--shadow-xs); }
.filter-select {
  padding: 7px 12px; border: 1.5px solid var(--line-strong);
  border-radius: var(--r-md); font-size: 13px;
  background: var(--panel); color: var(--ink); cursor: pointer;
  font-family: var(--font);
  transition: border-color var(--t), box-shadow var(--t);
}
.filter-select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(156,84,85,.12); }
.filter-hint { font-size: 13px; color: var(--muted); }

/* ── Playbook Pages ── */
.pb-hero { display: flex; gap: 18px; align-items: flex-start; margin-bottom: 16px; }
.pb-icon {
  width: 52px; height: 52px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-light); color: var(--accent);
  border-radius: var(--r-lg); font-size: 11px; font-weight: 700;
  letter-spacing: .04em; text-transform: uppercase; font-family: var(--font);
  margin-top: 4px; border: 1px solid rgba(194,91,110,.15);
}
.pb-hero-body h1 { margin: 0 0 6px; }
.biz-tag { display: inline-block; font-size: 11px; font-weight: 600; background: var(--panel-3); color: var(--ink-2); padding: 3px 10px; border-radius: var(--r-full); margin-bottom: 8px; }
.pb-roi-callout { display: inline-flex; align-items: center; gap: 12px; background: var(--red-bg); border: 1px solid rgba(255,59,48,.2); border-radius: var(--r-lg); padding: 10px 16px; margin: 4px 8px 4px 0; font-size: 13px; font-weight: 500; }
.pb-roi-val { font-weight: 700; color: var(--red); font-size: 14px; }
.hero-badge { display: inline-block; font-size: 11.5px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); background: var(--accent-light); padding: 4px 12px; border-radius: var(--r-full); margin: 0 0 12px; }
.hero-primary-cta { display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 24px; }
.rm-scqa { background: linear-gradient(135deg, var(--bg-warm) 0%, var(--accent-light) 100%); border: 1px solid var(--line); border-radius: 20px; padding: 24px 28px; margin: 0 0 32px; max-width: 860px; margin-left: auto; margin-right: auto; }
.rm-scqa-s, .rm-scqa-c, .rm-scqa-q { display: flex; gap: 12px; margin-bottom: 12px; font-size: 14px; line-height: 1.65; color: var(--ink-2); }
.rm-scqa-q { margin-bottom: 0; font-weight: 600; color: var(--ink); }
.rm-scqa-label { flex-shrink: 0; width: 36px; height: 20px; background: var(--accent); color: #fff; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; display: flex; align-items: center; justify-content: center; margin-top: 2px; }
.rm-scqa-c .rm-scqa-label { background: var(--red); }
.rm-scqa-q .rm-scqa-label { background: var(--amber-dark); }
.pb-intro {
  background: var(--panel-2);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--r-lg) var(--r-lg) 0;
  padding: 14px 20px; margin: 16px 0;
  font-size: 14px; color: var(--ink-2); line-height: 1.7;
}
.pb-steps { margin-top: 24px; display: flex; flex-direction: column; gap: 14px; }
.pb-lead-capture {
  margin-top: 40px; padding: 32px; border-radius: 20px;
  background: linear-gradient(135deg, #1e3a5f 0%, #0f2340 100%);
  border: 1px solid rgba(37,99,235,.27);
}
.pb-lead-inner { display: flex; gap: 32px; align-items: flex-start; flex-wrap: wrap; }
.pb-lead-text { flex: 1; min-width: 240px; }
.pb-lead-text h3 { color: #f1f5f9; font-size: 18px; margin: 0 0 10px; }
.pb-lead-text p { color: #94a3b8; font-size: 14px; margin: 0 0 12px; }
.pb-lead-bullets { color: #94a3b8; font-size: 13px; padding-left: 0; list-style: none; margin: 0; }
.pb-lead-bullets li { margin-bottom: 6px; }
.pb-lead-action { flex-shrink: 0; text-align: center; }
.pb-lead-btn {
  display: inline-block; padding: 14px 28px;
  background: #2563eb; color: #fff; border-radius: 10px;
  font-weight: 700; font-size: 15px; text-decoration: none;
  transition: background 0.2s;
}
.pb-lead-btn:hover { background: #1d4ed8; }
.pb-lead-note { color: #64748b; font-size: 12px; margin-top: 10px; }
.pb-step {
  display: flex; gap: 18px;
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 20px; padding: 22px;
  box-shadow: var(--shadow-md);
  transition: transform var(--t-card), box-shadow var(--t-card);
}
.pb-step:hover { transform: translateY(-2px); box-shadow: var(--shadow-hover); }
.pb-step-num {
  width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
  color: #fff; display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; letter-spacing: -.01em;
  margin-top: 4px; font-family: var(--font);
}
.pb-step-body { flex: 1; min-width: 0; }
.pb-step-title { margin: 0 0 8px; font-size: 15px; font-weight: 600; letter-spacing: -.01em; }
.pb-problem {
  font-size: 13px; color: var(--accent); margin: 0 0 14px;
  font-weight: 500; padding: 8px 12px;
  background: var(--accent-light); border-radius: var(--r-sm);
  border-left: 3px solid var(--accent);
}
.pb-skills { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.pb-skill { background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-md); padding: 10px 14px; }
.pb-skill-header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.pb-skill-name { font-weight: 700; font-size: 13.5px; color: var(--accent); text-decoration: none; }
.pb-skill-name:hover { text-decoration: underline; }
.pb-skill-badges { display: flex; gap: 6px; }
.pb-skill-why { margin: 5px 0 0; font-size: 12.5px; color: var(--muted); line-height: 1.55; }
.pb-data, .pb-output { font-size: 12.5px; margin-top: 10px; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-sm); padding: 8px 12px; }
.pb-outcomes { background: var(--green-bg); border: 1px solid rgba(52,199,89,.2); border-radius: 20px; padding: 18px 22px; margin-top: 24px; }
.pb-outcomes h2 { margin: 0 0 10px; font-size: 14px; color: var(--green-dark); font-weight: 700; }
.pb-outcomes ul { margin: 0; padding-left: 18px; }
.pb-outcomes li { font-size: 13.5px; margin-bottom: 5px; color: var(--green-dark); }

/* ── ROI Calculator ── */
.calc-wrapper { margin: 40px 0 0; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-2xl); overflow: hidden; box-shadow: var(--shadow-sm); }
.calc-header { padding: 24px 28px 18px; border-bottom: 1px solid var(--line); }
.calc-header h2 { margin: 0 0 5px; font-size: 20px; font-weight: 800; }
.calc-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--line); background: var(--panel-2); }
.calc-tab { flex: 1; padding: 13px 8px; border: none; background: none; cursor: pointer; font-size: 13px; font-weight: 500; font-family: var(--font); color: var(--muted); border-bottom: 2px solid transparent; transition: color var(--t), background var(--t), border-color var(--t); }
.calc-tab:hover { color: var(--ink); background: var(--panel-3); }
.calc-tab.active { color: var(--tc, var(--accent)); border-bottom-color: var(--tc, var(--accent)); background: #fff; font-weight: 700; }
.calc-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.calc-body { padding: 0; }
.calc-panel { display: none; grid-template-columns: 1fr 260px; gap: 0; }
.calc-panel.active { display: grid; }
.calc-inputs { padding: 24px 28px; display: flex; flex-direction: column; gap: 20px; }
.calc-row { display: flex; flex-direction: column; gap: 6px; }
.calc-label { font-size: 13px; font-weight: 600; color: var(--ink-2); }
.calc-input-wrap { display: flex; align-items: center; gap: 10px; }
.calc-input { flex: 1; accent-color: var(--tc, var(--accent)); height: 5px; cursor: pointer; }
.calc-val { font-size: 16px; font-weight: 800; color: var(--tc, var(--accent)); min-width: 58px; text-align: right; font-variant-numeric: tabular-nums; }
.calc-unit { font-size: 12px; color: var(--muted); min-width: 52px; }
.calc-result { background: var(--tc, var(--accent)); padding: 36px 24px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
.calc-result-label { font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: rgba(255,255,255,.7); margin-bottom: 10px; }
.calc-result-num { font-size: 50px; font-weight: 900; color: #fff; line-height: 1; font-variant-numeric: tabular-nums; }
.calc-result-unit { font-size: 15px; color: rgba(255,255,255,.85); margin-top: 6px; font-weight: 600; }
.calc-disclaimer { font-size: 11px; color: rgba(255,255,255,.5); margin-top: 20px; line-height: 1.6; max-width: 200px; }

/* ── CEO Tab content ── */
.ceo-entry { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 20px 0; align-items: start; }
.ceo-entry-body h3 { margin: 0 0 8px; font-size: 17px; font-weight: 700; }
.ceo-entry-body p { color: var(--muted); margin: 0 0 16px; font-size: 13.5px; line-height: 1.65; }
.ceo-phases { display: flex; flex-direction: column; gap: 10px; }
.ceo-phase { background: var(--panel); border-left: 3px solid; border-radius: 0 var(--r-lg) var(--r-lg) 0; padding: 12px 16px; font-size: 13px; box-shadow: var(--shadow-xs); }
.ceo-phase p { margin: 4px 0; color: var(--muted); }

/* ── Agent Marketplace ── */
.agent-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; margin: 20px 0; align-items: stretch; }
.agent-card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 20px;
  padding: 22px; display: flex; flex-direction: column; gap: 10px;
  box-shadow: var(--shadow-md);
  transition: transform var(--t-card), box-shadow var(--t-card), border-color var(--t-card);
  cursor: pointer;
}
.agent-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-hover); border-color: var(--accent); }
.agent-card-top { display: flex; align-items: flex-start; gap: 14px; }
.agent-icon-wrap {
  width: 44px; height: 44px; border-radius: var(--r-lg);
  background: var(--accent-light); color: var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700; flex-shrink: 0;
  letter-spacing: .04em; text-transform: uppercase; font-family: var(--font);
}
.agent-icon-wrap.cat-supply { background: var(--amber-bg); color: var(--amber-dark); }
.agent-icon-wrap.cat-ad     { background: var(--accent-light); color: var(--accent); }
.agent-icon-wrap.cat-risk   { background: var(--red-bg); color: var(--red); }
.agent-icon-wrap.cat-voc    { background: var(--green-bg); color: var(--green-dark); }
.agent-icon-wrap.cat-ops    { background: var(--panel-3); color: var(--ink); }
.agent-card-info { flex: 1; min-width: 0; }
.agent-name { font-size: 14.5px; font-weight: 700; margin: 0 0 4px; letter-spacing: -.01em; }
.agent-cat-badge {
  display: inline-block; font-size: 10.5px; font-weight: 600;
  padding: 2px 8px; border-radius: var(--r-full);
  background: var(--panel-3); color: var(--ink-2);
}
.agent-status { display: flex; align-items: center; gap: 5px; font-size: 12px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse-dot 2.5s ease-in-out infinite; }
.status-dot.demo { background: var(--amber); }
@keyframes pulse-dot { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.4);opacity:.7} }
.agent-desc { font-size: 13px; color: var(--muted); line-height: 1.55; margin: 0; }
.agent-roi { font-size: 11.5px; font-weight: 600; color: var(--green-dark); background: var(--green-bg); padding: 3px 10px; border-radius: var(--r-full); align-self: flex-start; }
.agent-skills { display: flex; flex-wrap: wrap; gap: 4px; }
.agent-skill-chip { font-size: 11px; background: var(--accent-light); color: var(--accent); padding: 2px 7px; border-radius: var(--r-full); font-weight: 500; text-decoration: none; }
.agent-invoke-btn {
  margin-top: auto; width: 100%; padding: 10px;
  background: var(--ink); color: #fff;
  border: none; border-radius: var(--r-md);
  font-size: 13px; font-weight: 600; font-family: var(--font);
  letter-spacing: .03em; text-transform: uppercase;
  cursor: pointer; transition: background var(--t), box-shadow var(--t);
  display: flex; align-items: center; justify-content: center; gap: 6px;
}
.agent-invoke-btn:hover { background: var(--ink-2); box-shadow: var(--shadow-lg); }
.agent-invoke-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.agent-cat-filter { display: flex; gap: 6px; flex-wrap: wrap; margin: 16px 0 8px; }
.cat-pill {
  padding: 6px 16px; border-radius: var(--r-full);
  border: 1.5px solid var(--line-strong);
  background: var(--panel); font-size: 12.5px; font-weight: 500;
  font-family: var(--font); color: var(--muted); cursor: pointer;
  transition: all var(--t);
}
.cat-pill:hover { border-color: var(--accent); color: var(--accent); }
.cat-pill.active { background: var(--accent); border-color: var(--accent); color: #fff; }

/* Agent Modal */
.agent-modal-overlay {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(29,29,27,.55); backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center;
  padding: 20px; opacity: 0; pointer-events: none;
  transition: opacity var(--t-slow);
}
.agent-modal-overlay.open { opacity: 1; pointer-events: all; }
.agent-modal {
  background: var(--panel); border-radius: 20px;
  width: 100%; max-width: 680px; max-height: 88vh;
  overflow-y: auto; box-shadow: var(--shadow-hover);
  transform: translateY(16px) scale(.98);
  transition: transform var(--t-slow);
}
.agent-modal-overlay.open .agent-modal { transform: translateY(0) scale(1); }
.modal-header {
  position: sticky; top: 0; z-index: 1;
  display: flex; align-items: center; gap: 14px;
  padding: 18px 22px; background: var(--panel);
  border-bottom: 1px solid var(--line);
}
.modal-icon {
  width: 44px; height: 44px; border-radius: var(--r-lg);
  background: var(--accent-light); color: var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; flex-shrink: 0;
  text-transform: uppercase; font-family: var(--font);
}
.modal-header-info { flex: 1; }
.modal-header-info h2 { margin: 0 0 4px; font-size: 16px; font-weight: 700; }
.modal-close {
  width: 30px; height: 30px; border-radius: 50%;
  background: var(--panel-2); border: 1px solid var(--line);
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  font-size: 14px; transition: background var(--t), color var(--t);
  color: var(--muted); font-weight: 500;
}
.modal-close:hover { background: var(--red-bg); color: var(--red); }
.modal-body { padding: 22px; }
.modal-section { margin-bottom: 22px; }
.modal-section h3 { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin: 0 0 12px; }
.modal-input-group { display: flex; flex-direction: column; gap: 10px; }
.modal-input {
  width: 100%; padding: 9px 13px;
  border: 1.5px solid var(--line-strong); border-radius: var(--r-md);
  font-size: 13.5px; background: var(--panel); color: var(--ink);
  font-family: var(--font);
  transition: border-color var(--t), box-shadow var(--t);
}
.modal-input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(194,91,110,.12); }
.modal-input::placeholder { color: var(--muted); }
.modal-run-btn {
  width: 100%; padding: 12px; background: var(--ink); color: #fff;
  border: none; border-radius: var(--r-md);
  font-size: 13.5px; font-weight: 700; font-family: var(--font);
  letter-spacing: .03em; text-transform: uppercase;
  cursor: pointer; transition: background var(--t), box-shadow var(--t);
  display: flex; align-items: center; justify-content: center; gap: 8px;
}
.modal-run-btn:hover { background: var(--ink-2); box-shadow: var(--shadow-lg); }
.modal-run-btn:disabled { background: var(--line-strong); cursor: not-allowed; box-shadow: none; }
.modal-output {
  background: var(--panel-2); border: 1px solid var(--line);
  border-radius: var(--r-lg); padding: 16px 18px; margin-top: 14px;
  min-height: 100px; display: none;
}
.modal-output.visible { display: block; }
.output-thinking { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 13.5px; }
.thinking-dots span { animation: blink 1.2s infinite; font-size: 18px; letter-spacing: 2px; }
.thinking-dots span:nth-child(2) { animation-delay: .2s; }
.thinking-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,100%{opacity:.2} 50%{opacity:1} }
.output-content { font-size: 13.5px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; font-family: var(--font); }
.modal-footer-skills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }

/* ── Responsive ── */
@media (max-width: 1024px) {
  .layout { grid-template-columns: 190px 1fr; }
  .content { padding: 24px 28px; }
  .two-col { grid-template-columns: 1fr 300px; }
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .hamburger { display: flex; }
  .sidebar {
    display: none; position: fixed;
    top: var(--topbar-height); left: 0;
    width: 260px; height: calc(100vh - var(--topbar-height));
    z-index: 195; transform: translateX(-100%);
    transition: transform var(--t-slow);
    box-shadow: var(--shadow-lg);
  }
  .sidebar.open { display: flex; flex-direction: column; transform: translateX(0); }
  #global-search { width: 160px; }
  .content { padding: 20px 18px; }
}
@media (max-width: 768px) {
  .two-col { grid-template-columns: 1fr; }
  .relation-panel { position: static; }
  .ceo-entry { grid-template-columns: 1fr; }
  .biz-grid { grid-template-columns: 1fr; }
  .hero-tabs { flex-wrap: wrap; }
  .calc-panel.active { grid-template-columns: 1fr; }
  .calc-result { padding: 24px; }
  .calc-result-num { font-size: 40px; }
  .agent-grid { grid-template-columns: 1fr; }
  .agent-modal { max-height: 96vh; }
}
@media (max-width: 480px) {
  .content h1 { font-size: 22px; }
  .pb-step { flex-direction: column; }
  .pb-step-num { width: 36px; height: 36px; font-size: 10px; }
  .topbar { padding: 0 14px; }
  #global-search { width: 80px; font-size: 12px; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .ds-grid { grid-template-columns: 1fr; }
  .biz-grid { grid-template-columns: 1fr; }
}

/* ── Agent Marketplace Hero ── */
.agent-hero {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 24px; margin-bottom: 28px; flex-wrap: wrap;
}
.agent-hero-text { flex: 1; min-width: 280px; }
.agent-hero-stats { display: flex; gap: 16px; flex-shrink: 0; }
.agent-stat {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--r-xl); padding: 16px 20px; text-align: center; min-width: 80px;
}
.agent-stat strong { display: block; font-size: 28px; font-weight: 900; color: var(--accent); letter-spacing: -.03em; }
.agent-stat span { font-size: 12px; color: var(--muted); font-weight: 600; }
@media(max-width: 600px) {
  .agent-hero { flex-direction: column; }
  .agent-hero-stats { width: 100%; justify-content: space-around; }
}

/* ── AI Roadmap (rm-*) ── */
.rm-hero{text-align:center;padding:60px 40px 40px;max-width:860px;margin:0 auto}
.rm-hero-eyebrow{font-size:13px;font-weight:600;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.rm-hero-title{font-size:36px;font-weight:800;line-height:1.2;margin:0 0 16px;color:#0f172a}
.rm-hero-sub{font-size:18px;color:#475569;margin:0 0 28px}
.rm-hero-cta{display:flex;gap:12px;justify-content:center;margin-bottom:14px}
.rm-hero-note{font-size:12px;color:var(--muted)}
.rm-btn-primary{padding:12px 28px;background:#2563eb;color:#fff;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block}
.rm-btn-primary:hover{background:#1d4ed8}
.rm-btn-sec{padding:12px 28px;background:#f1f5f9;color:#334155;border-radius:10px;font-size:15px;font-weight:600;text-decoration:none;display:inline-block}

.rm-summary-bar{display:flex;align-items:center;justify-content:center;gap:0;background:#0f172a;color:#fff;padding:24px 40px;margin:0 -36px;flex-wrap:wrap;gap:8px}
.rm-summary-item{text-align:center;padding:0 24px}
.rm-summary-num{display:block;font-size:28px;font-weight:800;color:#60a5fa}
.rm-summary-label{font-size:12px;color:#94a3b8}
.rm-summary-sep{font-size:24px;color:#334155;padding:0 8px}

.rm-roles-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0;padding:0}
.rm-role{display:flex;gap:16px;align-items:flex-start;background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:20px}
.rm-role-icon{font-size:32px;flex-shrink:0}
.rm-role strong{display:block;font-size:15px;margin-bottom:4px}
.rm-role p{margin:0 0 8px;font-size:13px;color:#64748b}
.rm-role-roi{font-size:13px;font-weight:700;color:#2563eb;background:#eff6ff;padding:3px 10px;border-radius:999px}

.rm-phases{display:flex;flex-direction:column;gap:24px;margin:32px 0}
.rm-phase{border-radius:16px;overflow:hidden;border:1px solid #e2e8f0}
.rm-phase-header{display:flex;align-items:flex-start;gap:20px;padding:24px 28px;background:var(--phase-bg);border-bottom:1px solid #e2e8f0}
.rm-phase-badge{width:72px;height:72px;border-radius:50%;background:var(--phase-color);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;flex-shrink:0;text-align:center;line-height:1.2}
.rm-phase-meta{flex:1}
.rm-phase-period{font-size:12px;font-weight:600;color:var(--phase-color);text-transform:uppercase;letter-spacing:.06em}
.rm-phase-theme{margin:4px 0;font-size:22px;font-weight:800}
.rm-phase-tagline{margin:0;font-size:14px;color:#64748b}
.rm-phase-roi{text-align:right;flex-shrink:0}
.rm-phase-roi-label{display:block;font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}
.rm-phase-roi strong{font-size:18px;font-weight:800;color:var(--phase-color)}

.rm-items{display:flex;flex-direction:column;gap:0}
.rm-item{display:flex;gap:20px;padding:24px 28px;border-bottom:1px solid #f1f5f9}
.rm-item:last-child{border-bottom:none}
.rm-item-icon{font-size:28px;flex-shrink:0;width:40px;text-align:center;margin-top:2px}
.rm-item-body{flex:1}
.rm-item-title{margin:0 0 12px;font-size:16px;font-weight:700}
.rm-story{background:#fefce8;border-left:3px solid #f59e0b;padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;color:#78350f;margin-bottom:8px}
.rm-story-label,.rm-result-label{font-weight:700;margin-right:6px}
.rm-result{background:#f0fdf4;border-left:3px solid #10b981;padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;color:#14532d;margin-bottom:8px}
.rm-roi-line{font-size:13px;margin-bottom:8px;color:#374151}
.rm-chips{display:flex;flex-wrap:wrap;gap:6px}
.rm-chip{font-size:11px;background:#eff6ff;color:#1e40af;padding:3px 10px;border-radius:999px;text-decoration:none;border:1px solid #bfdbfe}
.rm-chip:hover{background:#dbeafe}

.rm-footer{display:grid;grid-template-columns:1fr 340px;gap:40px;margin-top:40px;padding:36px;background:#0f172a;border-radius:16px;color:#e2e8f0}
.rm-footer h3{margin:0 0 10px;font-size:18px;color:#fff}
.rm-footer p{font-size:14px;color:#94a3b8;margin:0 0 16px}
.rm-footer-links{display:flex;flex-direction:column;gap:8px}
.rm-footer-links a{color:#60a5fa;text-decoration:none;font-size:14px}
.rm-footer-links a:hover{text-decoration:underline}
.rm-footer-right{display:flex;flex-direction:column;justify-content:space-between}
.rm-footer-cta{text-align:center;background:#1e293b;border-radius:12px;padding:24px;margin-bottom:16px}
.rm-footer-cta p{color:#94a3b8;font-size:13px;margin-bottom:12px}
.rm-footer-note{font-size:11px;color:#475569;line-height:1.6}

@media print {
  .topbar,.sidebar,.rm-hero-cta,.rm-footer-cta button{display:none!important}
  body{background:#fff}
  .content{padding:0!important;max-width:100%!important}
  .rm-summary-bar{margin:0!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .rm-phase,.rm-footer{break-inside:avoid}
  .rm-phases{gap:16px}
  @page{margin:20mm 15mm;size:A4}
}
@media(max-width:900px){
  .rm-roles-bar{grid-template-columns:1fr}
  .rm-footer{grid-template-columns:1fr}
  .rm-phase-header{flex-wrap:wrap}
}
""".strip()

def render_chat_page(nav: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI 知识库对话 · paper2skills</title>
  <link rel="stylesheet" href="{nav}assets/style.css">
  <style>
    body {{ overflow: hidden; }}
    .chat-layout {{
      display: flex; height: 100vh; flex-direction: column;
    }}
    .chat-topbar {{
      display: flex; align-items: center; gap: 0;
      height: var(--topbar-height); flex-shrink: 0;
      padding: 0 20px;
      background: rgba(255,255,255,0.92);
      backdrop-filter: blur(12px) saturate(180%);
      -webkit-backdrop-filter: blur(12px) saturate(180%);
      border-bottom: 1px solid var(--nav-border);
      box-shadow: 0 1px 0 rgba(0,0,0,0.05);
    }}
    .chat-back {{
      display: flex; align-items: center; gap: 6px;
      color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 500;
      padding: 6px 10px; border-radius: var(--r-md);
      transition: background var(--t);
      flex-shrink: 0;
    }}
    .chat-back:hover {{ background: var(--accent-light); text-decoration: none; }}
    .chat-title-area {{
      flex: 1; display: flex; align-items: center; justify-content: center;
      gap: 10px;
    }}
    .chat-title-icon {{
      font-size: 20px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .chat-title-text {{
      font-size: 15px; font-weight: 650; letter-spacing: -.02em; color: var(--ink);
    }}
    .chat-title-sub {{
      font-size: 11.5px; color: var(--muted); font-weight: 400;
      padding: 3px 9px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }}
    .chat-ctrl {{ flex-shrink: 0; display: flex; align-items: center; gap: 10px; }}
    .web-search-toggle {{
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 12px; color: var(--muted); cursor: pointer;
      padding: 5px 10px; border-radius: var(--r-full);
      border: 1.5px solid var(--line); background: transparent;
      transition: all var(--t); user-select: none; flex-shrink: 0;
      font-family: var(--font); white-space: nowrap;
    }}
    .web-search-toggle.on {{
      color: var(--accent); border-color: var(--accent);
      background: var(--accent-light); font-weight: 600;
    }}
    .web-search-toggle:hover {{ border-color: var(--line-strong); color: var(--ink); }}
    .web-search-toggle.on:hover {{ border-color: var(--accent-dark); }}
    .web-search-toggle-icon {{ font-size: 13px; line-height: 1; }}
    .chat-body {{
      flex: 1; display: flex; flex-direction: column;
      max-width: 760px; width: 100%; margin: 0 auto;
      padding: 0 16px; overflow: hidden;
    }}
    .chat-messages {{
      flex: 1; overflow-y: auto; padding: 28px 0 12px;
      display: flex; flex-direction: column; gap: 20px;
    }}
    .chat-messages::-webkit-scrollbar {{ width: 4px; }}
    .chat-messages::-webkit-scrollbar-thumb {{ background: var(--line-strong); border-radius: 4px; }}
    .cmsg {{ display: flex; gap: 12px; align-items: flex-start; }}
    .cmsg-user {{ flex-direction: row-reverse; }}
    .cmsg-avatar {{
      width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      color: #fff; display: flex; align-items: center; justify-content: center;
      font-size: 13px; font-weight: 700; margin-top: 2px;
    }}
    .cmsg-user .cmsg-avatar {{
      background: var(--panel-3); color: var(--muted); font-size: 11px;
    }}
    .cmsg-body {{ flex: 1; min-width: 0; }}
    .cmsg-name {{
      font-size: 11px; font-weight: 600; letter-spacing: .02em;
      text-transform: uppercase; color: var(--muted); margin-bottom: 5px;
    }}
    .cmsg-user .cmsg-name {{ text-align: right; }}
    .cmsg-bubble {{
      display: inline-block; max-width: 100%;
      padding: 12px 16px; border-radius: 4px 18px 18px 18px;
      background: var(--panel); border: 1px solid var(--line);
      font-size: 14.5px; line-height: 1.72; color: var(--ink);
      box-shadow: var(--shadow-xs);
    }}
    .cmsg-user .cmsg-bubble {{
      background: var(--accent); color: #fff; border-color: transparent;
      border-radius: 18px 4px 18px 18px; box-shadow: none;
    }}
    .cmsg-bubble strong {{ font-weight: 700; }}
    .cmsg-bubble code {{
      background: rgba(0,0,0,.06); padding: 2px 6px;
      border-radius: 5px; font-size: 13px; font-family: 'SF Mono', 'Menlo', monospace;
    }}
    .cmsg-bubble br {{ margin: 0; }}
    .cmsg-web-badge {{
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: var(--muted); margin-bottom: 6px;
      padding: 2px 8px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }}
    .cmsg-typing .cmsg-bubble::after {{
      content: ''; display: inline-block; width: 40px; height: 10px;
      background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 10'%3E%3Ccircle cx='5' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0s' repeatCount='indefinite'/%3E%3C/circle%3E%3Ccircle cx='20' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0.2s' repeatCount='indefinite'/%3E%3C/circle%3E%3Ccircle cx='35' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0.4s' repeatCount='indefinite'/%3E%3C/circle%3E%3C/svg%3E") no-repeat center;
      vertical-align: middle; margin-left: 4px;
    }}
    .chat-welcome {{
      text-align: center; padding: 40px 20px 20px; color: var(--muted);
    }}
    .chat-welcome-icon {{
      font-size: 48px; display: block; margin-bottom: 16px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .chat-welcome h2 {{
      font-size: 22px; font-weight: 700; letter-spacing: -.03em; color: var(--ink);
      margin: 0 0 8px; border: none; padding: 0;
    }}
    .chat-welcome p {{ font-size: 14.5px; color: var(--muted); margin: 0 0 24px; }}
    .chat-suggestions {{
      display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px;
    }}
    .chat-sug-btn {{
      padding: 8px 16px; border-radius: var(--r-full);
      border: 1.5px solid var(--line); background: var(--panel);
      font-size: 13px; color: var(--ink-2); cursor: pointer; font-family: var(--font);
      transition: border-color var(--t), background var(--t), color var(--t);
    }}
    .chat-sug-btn:hover {{
      border-color: var(--accent); color: var(--accent); background: var(--accent-light);
    }}
    .chat-input-area {{
      flex-shrink: 0; padding: 10px 0 18px;
      border-top: 1px solid var(--line);
    }}
    .chat-input-wrap {{
      display: flex; align-items: flex-end; gap: 8px;
      background: var(--panel); border: 1.5px solid var(--line);
      border-radius: 18px; padding: 8px 10px 8px 12px;
      transition: border-color var(--t), box-shadow var(--t);
    }}
    .chat-input-wrap:focus-within {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(194,91,110,.10);
    }}
    .chat-input-wrap .web-search-toggle {{
      align-self: flex-end; margin-bottom: 1px;
    }}
    .chat-textarea {{
      flex: 1; border: none; outline: none; resize: none;
      font-family: var(--font); font-size: 14.5px; line-height: 1.6;
      color: var(--ink); background: transparent;
      min-height: 24px; max-height: 160px;
      overflow-y: auto;
    }}
    .chat-textarea::placeholder {{ color: var(--muted); }}
    .chat-send-btn {{
      width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
      background: var(--accent); border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-size: 16px;
      transition: background var(--t), transform var(--t);
    }}
    .chat-send-btn:hover {{ background: var(--accent-dark); transform: scale(1.06); }}
    .chat-send-btn:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
    .chat-hint {{
      text-align: center; font-size: 11.5px; color: var(--muted);
      margin-top: 8px;
    }}
    @media (max-width: 600px) {{
      .chat-title-sub {{ display: none; }}
      .chat-body {{ padding: 0 12px; }}
    }}
  </style>
</head>
<body>
  <div class="chat-layout">
    <header class="chat-topbar">
      <a class="chat-back" href="{nav}index.html">← 返回</a>
      <div class="chat-title-area">
        <span class="chat-title-icon">✦</span>
        <span class="chat-title-text">AI 知识库对话</span>
        <span class="chat-title-sub">360 Skills · DeepSeek V3</span>
      </div>
    </header>

    <div class="chat-body">
      <div class="chat-messages" id="chat-messages">
        <div class="chat-welcome" id="chat-welcome">
          <span class="chat-welcome-icon">✦</span>
          <h2>paper2skills 知识库助手</h2>
          <p>基于 360 个从顶会论文萃取的跨境电商 AI 决策技能，为你提供专业问答</p>
          <div class="chat-suggestions">
            <button class="chat-sug-btn">如何提升广告 ROI？</button>
            <button class="chat-sug-btn">大促备货如何预测需求？</button>
            <button class="chat-sug-btn">供应链 AI 有哪些关键技能？</button>
            <button class="chat-sug-btn">KOL 投放效果怎么归因？</button>
            <button class="chat-sug-btn">如何预防封号和合规风险？</button>
            <button class="chat-sug-btn">用户流失预警方法有哪些？</button>
          </div>
        </div>
      </div>

      <div class="chat-input-area">
        <div class="chat-input-wrap">
          <button class="web-search-toggle" id="web-search-toggle" title="开启联网搜索">
            <span class="web-search-toggle-icon">🌐</span>
            <span id="web-search-label">联网</span>
          </button>
          <textarea class="chat-textarea" id="chat-input"
            placeholder="问我关于跨境电商 AI 决策技能的任何问题…"
            rows="1" autocomplete="off"></textarea>
          <button class="chat-send-btn" id="chat-send" title="发送 (Enter)">↑</button>
        </div>
        <p class="chat-hint">Enter 发送 · Shift+Enter 换行</p>
      </div>
    </div>
  </div>

  <script src="{nav}assets/playbook-data.js"></script>
  <script src="{nav}assets/chat-page.js"></script>
</body>
</html>"""


def build_chat_page_js() -> str:
    return r"""
(function () {
  const msgsEl   = document.getElementById('chat-messages');
  const welcome  = document.getElementById('chat-welcome');
  const textarea = document.getElementById('chat-input');
  const sendBtn  = document.getElementById('chat-send');
  const webToggle= document.getElementById('web-search-toggle');
  const webLabel = document.getElementById('web-search-label');

  let webSearchOn = false;
  let history = [];

  webToggle.addEventListener('click', () => {
    webSearchOn = !webSearchOn;
    webToggle.classList.toggle('on', webSearchOn);
    webLabel.textContent = webSearchOn ? '已开启联网' : '联网搜索';
  });

  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  });
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend(); }
  });
  sendBtn.addEventListener('click', doSend);

  document.querySelectorAll('.chat-sug-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.textContent.trim();
      textarea.dispatchEvent(new Event('input'));
      doSend();
    });
  });

  function md(text) {
    return text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/\*\*(.+?)\*\*/gs,'<strong>$1</strong>')
      .replace(/\*([^*\n]+)\*/g,'<em>$1</em>')
      .replace(/`([^`\n]+)`/g,'<code>$1</code>')
      .replace(/^#{1,3}\s+(.+)$/gm,'<strong style="font-size:15px">$1</strong>')
      .replace(/^[-•]\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">• $1</span>')
      .replace(/^\d+\.\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">$&</span>')
      .replace(/\n\n+/g,'<br><br>').replace(/\n/g,'<br>');
  }

  function addMsg(text, role, webBadge) {
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-' + role;
    const avatar = document.createElement('div');
    avatar.className = 'cmsg-avatar';
    avatar.textContent = role === 'bot' ? '\u2726' : 'U';
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    const name = document.createElement('div');
    name.className = 'cmsg-name';
    name.textContent = role === 'bot' ? 'AI 助手' : '你';
    body.appendChild(name);
    if (webBadge) {
      const badge = document.createElement('div');
      badge.className = 'cmsg-web-badge';
      badge.innerHTML = '🌐 联网搜索';
      body.appendChild(badge);
    }
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    if (role === 'bot') { bubble.innerHTML = md(text); }
    else { bubble.textContent = text; }
    body.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(body);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    return { row, bubble };
  }

  function addTyping() {
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-bot cmsg-typing';
    const avatar = document.createElement('div');
    avatar.className = 'cmsg-avatar';
    avatar.textContent = '\u2726';
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    const name = document.createElement('div');
    name.className = 'cmsg-name';
    name.textContent = 'AI 助手';
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    body.appendChild(name);
    body.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(body);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    return row;
  }

  function buildContext() {
    const DATA = window.PLAYBOOK_DATA || {};
    return (DATA.skills || []).slice(0, 80).map(s =>
      s.skill_id + ': ' + (s.problem_solved || s.algorithm_summary || '').slice(0, 140)
    ).join('\n');
  }

  async function doSend() {
    const text = textarea.value.trim();
    if (!text || sendBtn.disabled) return;
    textarea.value = '';
    textarea.style.height = 'auto';
    sendBtn.disabled = true;

    addMsg(text, 'user');
    history.push({ role: 'user', content: text });

    const typing = addTyping();
    const ctx = buildContext();
    const systemPrompt = `你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策技能。知识库收录了360个从顶会论文（NeurIPS/KDD/ICML/WWW）萃取的可落地业务技能，涵盖供应链优化、广告归因、用户分析、KOL投放、合规决策、智能体工程等领域。请用清晰、结构化的中文回答，优先引用知识库中的具体Skill，给出可操作建议。当前时间：${new Date().toLocaleDateString('zh-CN', {year:'numeric',month:'long',day:'numeric'})}。`;

    const messages = [
      { role: 'system', content: systemPrompt + '\n\n知识库摘要（前80条Skill）：\n' + ctx },
      ...history.slice(-6)
    ];

    try {
      const body = {
        model: 'deepseek-chat',
        messages,
        max_tokens: 1200,
        temperature: 0.6,
        stream: false
      };
      if (webSearchOn) {
        body.tools = [{ type: 'function', function: { name: 'web_search', description: 'Search the web for current information', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } } }];
        body.tool_choice = 'auto';
      }
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      const choice = data?.choices?.[0];
      let answer = choice?.message?.content?.trim();
      if (!answer && choice?.finish_reason === 'tool_calls') {
        answer = '（已触发联网搜索，DeepSeek 正在整合结果…）\n\n' + (choice?.message?.tool_calls?.[0]?.function?.arguments || '');
      }
      answer = answer || '抱歉，暂时无法获取回答，请稍后重试。';
      typing.remove();
      addMsg(answer, 'bot', webSearchOn);
      history.push({ role: 'assistant', content: answer });
    } catch (e) {
      typing.remove();
      addMsg('网络请求失败，请检查连接后重试。', 'bot');
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  }
})();
"""


def build_search_js() -> str:
    return r"""
(function(){
  const input = document.getElementById('global-search');
  const box   = document.getElementById('search-results');
  if (!input || !box || !window.PLAYBOOK_DATA) return;
  const skills = window.PLAYBOOK_DATA.skills || [];

  function applyFilters(list) {
    const diff  = (document.getElementById('filter-diff')  || {}).value || '';
    const roi   = (document.getElementById('filter-roi')   || {}).value || '';
    const dom   = (document.getElementById('filter-domain') || {}).value || '';
    return list.filter(s => {
      if (dom  && s.domain_dir !== dom) return false;
      if (diff && s.difficulty !== diff) return false;
      if (roi) {
        const stars = (s.difficulty || '').split('⭐').length - 1;
        if (roi === 'easy'   && stars > 2) return false;
        if (roi === 'medium' && (stars < 3 || stars > 3)) return false;
        if (roi === 'hard'   && stars < 4) return false;
      }
      return true;
    });
  }

  function doSearch() {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { box.classList.add('hidden'); box.innerHTML = ''; return; }
    let hits = skills.filter(s =>
      [s.skill_id, s.title, s.domain_dir,
       (s.tags||[]).join(' '), (s.topics||[]).join(' '),
       s.algorithm_summary, s.problem_solved, s.roi_figure
      ].join(' ').toLowerCase().includes(q)
    );
    hits = applyFilters(hits).slice(0, 24);
    box.innerHTML = hits.map(s =>
      `<a class="result" href="${rootPrefix()}skills/${s.skill_id}.html">` +
      `<strong>${esc(s.title)}</strong>` +
      `<br><span>${esc(s.domain_dir)}` +
      `${s.roi_figure ? ' · ' + esc(s.roi_figure) : ''}` +
      `${s.difficulty ? ' · ' + esc(s.difficulty) : ''}</span></a>`
    ).join('') || '<p class="muted" style="padding:12px">无结果</p>';
    box.classList.remove('hidden');
  }

  input.addEventListener('input', doSearch);
  ['filter-diff','filter-roi','filter-domain'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', doSearch);
  });
  document.addEventListener('click', e => {
    if (e.target !== input && !box.contains(e.target)) box.classList.add('hidden');
  });
  function rootPrefix() {
    const p = location.pathname;
    return (p.includes('/skills/') || p.includes('/domains/') || p.includes('/topics/') ||
            p.includes('/workflows/') || p.includes('/playbooks/') || p.includes('/graph/')) ? '../' : '';
  }
  function esc(s) {
    return String(s||'').replace(/[&<>"']/g, c =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
})();
""".strip()


# ---------------------------------------------------------------------------
# render_pages orchestrator
# ---------------------------------------------------------------------------

def render_pages(
    out: Path,
    skills: list[PlaybookSkill],
    domains: list[dict[str, Any]],
    graph: SkillsGraph,
    wf_defs: dict[str, Any],
) -> dict[str, Any]:
    known_skill_ids: set[str] = {skill.skill_id for skill in skills}
    KNOWN_SKILL_IDS.clear()
    KNOWN_SKILL_IDS.update(known_skill_ids)
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)

    skill_count  = len(skills)
    edge_count   = len(graph.edges)
    domain_count = len({s.domain_dir for s in skills})

    # Data assets
    build_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {"skill_count": skill_count, "domain_count": domain_count, "edge_count": edge_count},
        "domains": domains,
        "skills": [skill.__dict__ for skill in skills],
    }
    write_file(out / "assets" / "playbook-data.json", json.dumps(data, ensure_ascii=False, indent=2))
    write_file(out / "assets" / "playbook-data.js",  "window.PLAYBOOK_DATA = " + json.dumps(data, ensure_ascii=False) + ";")

    _skill_title_map = {s.skill_id: s.title for s in skills}
    graph_node_ids = {n.id for n in graph.nodes.values()}
    graph_json = {
        "nodes": [
            {
                "id": n.id,
                "domain": n.domain,
                "title": _skill_title_map.get(n.id, n.id),
            }
            for n in graph.nodes.values()
        ],
        "links": [
            {"source": e.source, "target": e.target, "type": e.edge_type}
            for e in graph.edges
            if e.source in graph_node_ids and e.target in graph_node_ids
        ],
    }
    write_file(out / "assets" / "graph-data.json", json.dumps(graph_json, ensure_ascii=False, indent=2))
    write_file(out / "assets" / "style.css",  build_css())
    write_file(out / "assets" / "search.js",  build_search_js())
    write_file(out / "assets" / "graph.js",   build_graph_js())
    write_file(out / "assets" / "ego-graph.js", build_ego_graph_js())
    write_file(out / "assets" / "chat-page.js", build_chat_page_js())
    write_file(out / "chat.html", render_chat_page())

    # ── Index (Phase 3C) ──
    write_file(out / "index.html", html_page(
        "总览",
        render_index(skill_count, domain_count, edge_count, domains, skills),
        active_nav="index",
    ))

    # ── Skill pages ──
    for skill in skills:
        write_file(out / "skills" / f"{skill.skill_id}.html", render_skill_page(skill))
    all_cards = "".join(render_skill_card(skill, "../") for skill in skills)
    domain_opts = "".join(
        f"<option value='{html.escape(d)}'>{html.escape(d)}</option>"
        for d in sorted({s.domain_dir for s in skills})
    )
    filter_bar = f"""
<div class="filter-bar">
  <select id="filter-domain" class="filter-select">
    <option value="">全部领域</option>{domain_opts}
  </select>
  <select id="filter-diff" class="filter-select">
    <option value="">全部难度</option>
    <option value="⭐☆☆☆☆">⭐ 入门</option>
    <option value="⭐⭐☆☆☆">⭐⭐ 简单</option>
    <option value="⭐⭐⭐☆☆">⭐⭐⭐ 中等</option>
    <option value="⭐⭐⭐⭐☆">⭐⭐⭐⭐ 较难</option>
    <option value="⭐⭐⭐⭐⭐">⭐⭐⭐⭐⭐ 专家</option>
  </select>
  <span class="filter-hint muted" id="filter-count"></span>
</div>
<script>
(function(){{
  function applyCardFilters(){{
    var domSel  = document.getElementById('filter-domain');
    var diffSel = document.getElementById('filter-diff');
    var dom  = domSel  ? domSel.value  : '';
    var diff = diffSel ? diffSel.value : '';
    var cards = document.querySelectorAll('#skill-card-grid .skill-card');
    var shown = 0;
    cards.forEach(function(c){{
      var matchDom  = !dom  || c.dataset.domain === dom;
      var matchDiff = !diff || c.dataset.diff   === diff;
      var visible = matchDom && matchDiff;
      c.style.display = visible ? '' : 'none';
      if(visible) shown++;
    }});
    var hint = document.getElementById('filter-count');
    if(hint){{
      hint.textContent = (dom||diff) ? ('\u663e\u793a '+shown+' / '+cards.length+' \u4e2a') : '';
    }}
  }}
  ['filter-domain','filter-diff'].forEach(function(id){{
    var el = document.getElementById(id);
    if(el) el.addEventListener('change', applyCardFilters);
  }});
}})();
</script>"""
    write_file(out / "skills" / "index.html", html_page(
        "全部 Skills",
        f"<h1>全部 Skills</h1>{filter_bar}<div class='cards' id='skill-card-grid'>{all_cards}</div>",
        "../",
        active_nav="skills",
    ))

    # ── Domain pages ──
    domain_index_cards: list[str] = []
    for domain in domains:
        domain_skills = [s for s in skills if s.domain_dir == domain["vault_dir"]]
        cards = "".join(render_skill_card(s, "../") for s in domain_skills)
        title = domain["vault_dir"]
        write_file(
            out / "domains" / f"{slugify(title)}.html",
            html_page(title,
                      f"<h1>{html.escape(title)}</h1>"
                      f"<p>{html.escape(domain.get('description',''))}</p>"
                      f"<div class='cards'>{cards}</div>",
                      "../"),
        )
        domain_index_cards.append(
            f"<a class='metric-card domain-card' href='{slugify(title)}.html'>"
            f"<strong>{html.escape(title)}</strong>"
            f"<span>{len(domain_skills)} Skills</span></a>"
        )
    write_file(out / "domains" / "index.html", html_page(
        "按领域",
        "<h1>按领域</h1><div class='grid'>" + "".join(domain_index_cards) + "</div>",
        "../",
        active_nav="domains",
    ))

    # ── Topic pages ──
    all_topics = sorted({topic for s in skills for topic in s.topics})
    topic_cards: list[str] = []
    for topic in all_topics:
        topic_skills = [s for s in skills if topic in s.topics]
        cards = "".join(render_skill_card(s, "../") for s in topic_skills)
        path = f"{slugify(topic)}.html"
        write_file(out / "topics" / path, html_page(
            topic,
            f"<h1>{html.escape(topic)}</h1><div class='cards'>{cards}</div>",
            "../",
        ))
        topic_cards.append(f"<a class='metric-card' href='{path}'>{html.escape(topic)}<span>{len(topic_skills)} Skills</span></a>")
    write_file(out / "topics" / "index.html", html_page(
        "按主题",
        "<h1>按主题</h1><div class='grid'>" + "".join(topic_cards) + "</div>",
        "../",
    ))

    # ── Workflow pages (Phase 2B: YAML-first, keyword fallback) ──
    skill_lookup = {s.skill_id: s for s in skills}
    workflow_cards: list[str] = []
    for workflow_name in WORKFLOW_RULES:
        slug_path = f"{slugify(workflow_name)}.html"
        wf_id = slugify(workflow_name).split("-")[0] + "-" + slugify(workflow_name).split("-")[1]  # e.g. "wf-a"

        # Check if a structured YAML definition exists
        if wf_defs and wf_id in wf_defs:
            page_html = render_workflow_page(wf_defs[wf_id], skill_lookup)
        else:
            # Fallback: keyword-matched skill list (original behaviour)
            wf_skills = [s for s in skills if workflow_name in s.workflows]
            cards = "".join(render_skill_card(s, "../") for s in wf_skills)
            page_html = html_page(
                workflow_name,
                f"<h1>{html.escape(workflow_name)}</h1>"
                f"<p class='muted'>按业务流程推荐的 Skill 链。</p>"
                f"<div class='cards'>{cards}</div>",
                "../",
            )
            wf_skills_for_count = wf_skills

        write_file(out / "workflows" / slug_path, page_html)
        wf_skill_count = len(wf_defs.get(wf_id, {}).get("steps", [])) or len([s for s in skills if workflow_name in s.workflows])
        workflow_cards.append(
            f"<a class='metric-card' href='{slug_path}'>"
            f"<strong>{html.escape(workflow_name)}</strong>"
            f"<span>{wf_skill_count} 步骤/Skills</span></a>"
        )
    write_file(out / "workflows" / "index.html", html_page(
        "工作流",
        "<h1>工作流</h1><p class='muted'>端到端业务决策路径，每条工作流包含分步决策树和推荐 Skill 组合。</p>"
        "<div class='grid'>" + "".join(workflow_cards) + "</div>",
        "../",
    ))

    # ── Skills Graph (Phase 3D: D3 visualisation) ──
    write_file(out / "graph" / "overview.html", render_graph_page(skill_count, edge_count, data["generated_at"].replace("-", "").replace(":", "").replace("T", "")))

    # ── CEO Roadmap whitepaper ──
    write_file(out / "ai-roadmap.html", render_roadmap_page(skill_lookup))
    write_file(out / "agents.html", render_agents_page(skill_lookup))

    # ── toB Scene Playbooks (Phase F) ──
    for pb in TOB_PLAYBOOKS:
        write_file(
            out / "playbooks" / f"{pb['id']}.html",
            render_tob_playbook(pb, skill_lookup),
        )
    tob_index_cards = "".join(
        f"<a class='biz-card' href='{pb['id']}.html'>"
        f"<div class='biz-card-header'>"
        f"<span class='biz-icon'>{pb['icon']}</span>"
        f"<div class='biz-body'>"
        f"<div class='biz-card-meta'>"
        f"<strong>{html.escape(pb['name'])}</strong>"
        f"<span class='biz-tag'>{html.escape(pb['tag'])}</span>"
        f"</div>"
        f"<p>{html.escape(pb['desc'])}</p>"
        f"</div>"
        f"</div>"
        f"</a>"
        for pb in TOB_PLAYBOOKS
    )
    write_file(out / "playbooks" / "index.html", html_page(
        "场景手册",
        "<h1>场景手册</h1>"
        "<p class='muted'>针对运营部门的开箱即用决策指南，每本手册包含完整操作步骤、所需数据和预期收益。</p>"
        f"<div class='biz-grid'>{tob_index_cards}</div>",
        "../",
    ))

    write_file(out / "README.md", "# paper2skills Playbook\n\n打开 `index.html` 浏览。\n")

    for html_file in out.rglob("*.html"):
        try:
            content = html_file.read_text(encoding="utf-8")
            new_content = content.replace(
                'playbook-data.js"',
                f'playbook-data.js?v={build_ts}"'
            ).replace(
                "playbook-data.js'",
                f"playbook-data.js?v={build_ts}'"
            )
            if new_content != content:
                html_file.write_text(new_content, encoding="utf-8")
        except Exception:
            pass
    report = {
        "skill_pages": skill_count,
        "domains": domain_count,
        "edges": edge_count,
        "generated_at": data["generated_at"],
    }
    write_file(out / "build-report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def build(root: Path, vault: Path, out: Path) -> dict[str, Any]:
    graph  = build_graph(vault)
    skills = build_skills(root, vault, graph)
    domains = domain_dicts(root)
    wf_defs = load_workflow_defs(root)
    return render_pages(out, skills, domains, graph, wf_defs)


def domain_dicts(root: Path) -> list[dict[str, Any]]:
    registry = load_domain_registry(root)
    return [
        {
            "key": entry.key,
            "vault_dir": entry.vault_dir,
            "description": entry.description,
            "skill_count": entry.skill_count,
            "code_status": entry.code_status,
        }
        for entry in registry.entries
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper2skills static Playbook")
    parser.add_argument("--root",  default=str(BASE_DIR))
    parser.add_argument("--vault", default="paper2skills-vault")
    parser.add_argument("--out",   default="playbook")
    args = parser.parse_args()

    root  = Path(args.root).resolve()
    vault = (root / args.vault).resolve() if not Path(args.vault).is_absolute() else Path(args.vault)
    out   = (root / args.out).resolve()   if not Path(args.out).is_absolute()   else Path(args.out)
    report = build(root, vault, out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
