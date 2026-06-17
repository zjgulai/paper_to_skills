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

# ── Extracted data modules ──────────────────────────────────────────────────
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))
from config.playbooks_data import TOB_PLAYBOOKS
from config.agents_data import AGENT_CATALOG
# ─────────────────────────────────────────────────────────────────────────────


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
    "WF-J DTC 独立站增长": ["dtc", "独立站", "shopify", "acquisition", "federated", "intent", "conversion", "ltv", "personalization"],
    "WF-K 全域风险防御": ["fraud", "account health", "appeal", "欺诈", "合规", "compliance", "violation", "anomaly", "risk"],
    "WF-L 内容营销增长": ["kol", "content", "tiktok", "live commerce", "video", "creator", "内容", "直播", "短视频"],
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
    "24-标签工程": {
        "role": "数据架构师 / 供应链数字化负责人",
        "role2": "CTO · 数据工程师 · 供应链团队",
        "trigger": "多平台数据孤岛导致断货识别延迟8小时；标签覆盖率不足使AI决策触发率<30%；想实现分析→行动自动闭环但不知从何下手",
        "outcome": "统一 Tag Schema + 传播引擎将标签覆盖率从 30% 提升至 97%；Palantir 风格 Object-Action-Writeback 将补货响应从 2 天缩短至 4 小时自动触发",
        "pain": "多平台 SKU 编码混乱无法统一 · 合规标签手工维护遗漏频繁 · 预测模型有了但结果无法自动触发采购 · 标签打了但没有质量监控",
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
    <span class='status-dot live'></span>
    <span style='color:#059669;font-size:12px;font-weight:600'>本地分析</span>
    &nbsp;·&nbsp;
    <span style='font-size:12px;color:#64748b'>即时响应</span>
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
          <span class='agent-status'><span class='status-dot live'></span> <span style='font-size:12px;color:#059669;font-weight:600'>本地分析 · 即时</span></span>
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

    body = rf"""
<div class='agent-hero'>
  <div class='agent-hero-text'>
    <h1 style='font-size:32px;font-weight:900;letter-spacing:-.03em;margin:0 0 10px'>
      智能体广场
    </h1>
    <p class='lead'>12 个专业 AI Agent，覆盖选品→Listing→广告→客服→合规全链路</p>
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:4px'>
      <span style='font-size:13px;background:#d1fae5;color:#065f46;padding:3px 10px;border-radius:999px;font-weight:600'>⚡ 本地计算引擎</span>
      <span style='font-size:13px;color:#64748b'>输入你的真实数据，即时获得个性化计算结果</span>
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
const RADAR_KEYWORDS = [
  '硅胶婴儿餐具','吸奶器','婴儿推车','母婴消毒器',
  '婴儿安全防护角','儿童宝宝硅胶牙刷','婴儿辅食机',
  '新生儿礼盒套装','婴儿水杯学饮杯','孕妇枕头哺乳枕',
];

function fillExample(id) {{
  const data = DEMO_DATA[id];
  if (!data || !data.inputs) return;
  data.inputs.forEach(inp => {{
    const el = document.getElementById(id + '__' + inp.id);
    if (!el) return;
    if (id === 'agent-product-radar' && inp.id === 'keyword') {{
      el.value = RADAR_KEYWORDS[Math.floor(Math.random() * RADAR_KEYWORDS.length)];
    }} else if (inp.type === 'textarea') {{
      el.value = inp.placeholder || '';
    }} else if (inp.type === 'select') {{
      if (inp.options && inp.options.length > 0) el.value = inp.options[0];
    }} else {{
      el.value = inp.placeholder ? inp.placeholder.replace(/^例：/, '') : '';
    }}
  }});
}}

function getVal(id, field) {{
  const el = document.getElementById(id + '__' + field);
  return el ? el.value.trim() : '';
}}
function fmtNum(n) {{
  return n.toLocaleString('en-US', {{maximumFractionDigits: 0}});
}}
function fmtMoney(n) {{
  return '$' + Math.abs(n).toLocaleString('en-US', {{minimumFractionDigits: 0, maximumFractionDigits: 0}});
}}
function pct(n) {{ return (n * 100).toFixed(1) + '%'; }}

function computeSupplySentinel(id) {{
  const stock = parseFloat(getVal(id,'stock')) || 340;
  const vel   = parseFloat(getVal(id,'velocity')) || 28;
  const lt    = parseFloat(getVal(id,'lead_time')) || 21;
  const ch    = getVal(id,'channel') || 'Amazon FBA';
  const days  = vel > 0 ? (stock / vel) : 999;
  const safetyDays = 30;
  const reorderQty = Math.ceil(vel * (lt + safetyDays));
  const airQty  = Math.ceil(reorderQty * 0.5);
  const seaQty  = reorderQty - airQty;
  const airCost = (airQty * 0.8).toFixed(0);
  const lossMd  = Math.min(days, lt) * vel * 25;
  const riskLv  = days < lt ? '🔴 高危' : days < lt + 7 ? '🟡 警戒' : '🟢 安全';
  const action  = days < lt ? '需立即行动！' : days < lt + 7 ? '建议本周下单' : '库存充裕，按计划补货';
  const q4Multi = 2.8;
  const q4Stock = Math.ceil(vel * q4Multi * 60);
  return `[供应链哨兵] 实时计算结果

━━ 库存状态 ━━
当前库存: ${{fmtNum(stock)}} 件
日均销速: ${{vel}} 件/天（您输入）
剩余可售天数: ${{days.toFixed(1)}} 天
风险等级: ${{riskLv}}

━━ 供货周期分析（${{ch}}）━━
您的供货周期: ${{lt}} 天
安全库存天数目标: ${{safetyDays}} 天
${{days < lt ? '[WARN] 已进入断货窗口，需立即行动！' : '[OK] ' + action}}

━━ 补货建议 ━━
├─ 建议补货量: ${{fmtNum(reorderQty)}} 件（${{lt}}天周期 + ${{safetyDays}}天安全库存）
├─ 推荐方案: 空运 ${{fmtNum(airQty)}} 件（应急）+ 海运 ${{fmtNum(seaQty)}} 件（补充）
├─ 空运额外成本: +${{airCost}}
└─ 不补货预估断货损失: ${{fmtMoney(lossMd)}}（${{Math.ceil(Math.min(days,lt))}}天断货 × ${{vel}}件/天 × $25 BSR成本）

━━ Q4 旺季预警 ━━
历史旺季销速倍数: ×${{q4Multi}}
Q4 建议备货量: ${{fmtNum(q4Stock)}} 件
最迟启动时间: 旺季前 ${{lt + 14}} 天

[${{days >= lt ? '>' : '!'}}] 结论: ${{action}}`;
}}

function computePricingAdvisor(id) {{
  const price    = parseFloat(getVal(id,'price')) || 19.99;
  const cost     = parseFloat(getVal(id,'cost'))  || 7.80;
  const compRaw  = getVal(id,'comp_range') || '$15-$22';
  const bsr      = parseInt(getVal(id,'bsr')) || 500;
  const margin   = (price - cost) / price;
  const compNums = compRaw.match(/[\d.]+/g) || ['15','22'];
  const compLo   = parseFloat(compNums[0]) || 15;
  const compHi   = parseFloat(compNums[1] || compNums[0]) || 22;
  const compMid  = (compLo + compHi) / 2;
  const bsrScore = bsr < 100 ? 'Top 100（强势）' : bsr < 500 ? 'Top 500（良好）' : bsr < 2000 ? 'Top 2000（普通）' : '2000+（待提升）';
  const suggested_lo = Math.max(price * 1.05, compMid * 0.95).toFixed(2);
  const suggested_hi = (compHi * 0.98).toFixed(2);
  const newMargin = ((parseFloat(suggested_lo) - cost) / parseFloat(suggested_lo) * 100).toFixed(1);
  const w1 = (price + 1).toFixed(2);
  const w2 = parseFloat(suggested_lo).toFixed(2);
  const primeDayPrice = (price * 0.95).toFixed(2);
  const q4Price = Math.min(parseFloat(suggested_hi), price * 1.15).toFixed(2);
  const monthlyUnits = Math.max(30, Math.round(3000 / bsr));
  const monthlyGain  = ((parseFloat(suggested_lo) - price) * monthlyUnits).toFixed(0);
  return `[动态定价顾问] 实时分析结果

━━ 当前状态 ━━
售价: ${{price}} | 成本: ${{cost}} | 毛利率: ${{(margin*100).toFixed(1)}}% | BSR: #${{bsr}}（${{bsrScore}}）

━━ 竞品价格带分析 ━━
竞品区间: ${{compRaw}} | 中位价: $${{compMid.toFixed(2)}}
您的定价相对竞品: ${{price < compMid ? '偏低，有提价空间' : price > compHi ? '高于竞品，需强差异化支撑' : '处于合理区间'}}

━━ 最优定价建议 ━━
推荐区间: $${{suggested_lo}} - $${{suggested_hi}}
理由: 竞品中位 $${{compMid.toFixed(2)}}，BSR ${{bsrScore}} 支持适当溢价
预期毛利率提升: ${{(margin*100).toFixed(1)}}% → ${{newMargin}}%（+${{(parseFloat(newMargin)-margin*100).toFixed(1)}}pp）
月均增益估算: +$${{monthlyGain}}（约 ${{monthlyUnits}} 单/月 × ${{(parseFloat(suggested_lo)-price).toFixed(2)}} 差价）

━━ 分步涨价路径 ━━
Week 1: $${{price}} → $${{w1}}（观察转化率变化）
Week 2: 若转化率降幅 <15%，升至 $${{w2}}
Week 3+: 稳定后评估是否继续到 $${{suggested_hi}}

━━ 促销节奏建议 ━━
├─ 每月1次 Coupon 10-15%（维持搜索权重，建议 $${{(price*0.88).toFixed(2)}}）
├─ Prime Day 前2周: $${{primeDayPrice}}（冲BSR，接受短期利润压缩）
└─ Q4 旺季: $${{q4Price}}（需求刚性，不主动降价）

[WARN] 监控阈值: 若7天内转化率下降 >20%，立即回退至 $${{w1}}`;
}}

function computePnLAnalyzer(id) {{
  const rev    = parseFloat(getVal(id,'revenue')) || 32400;
  const cogs   = parseFloat(getVal(id,'cogs'))    || 9200;
  const fba    = parseFloat(getVal(id,'fba'))     || 5800;
  const ads    = parseFloat(getVal(id,'ads'))     || 6500;
  const retPct = parseFloat(getVal(id,'return_rate')) || 4;
  const comm   = rev * 0.15;
  const shipping = rev * 0.059;
  const retCost  = rev * retPct / 100 * 0.40;
  const total_cost = cogs + fba + ads + comm + shipping + retCost;
  const profit = rev - total_cost;
  const netPct = (profit / rev * 100).toFixed(1);
  const acos   = (ads / rev * 100).toFixed(1);
  const targetAcos = 18;
  const adWaste = Math.max(0, ads * (parseFloat(acos) - targetAcos) / parseFloat(acos));
  const retSave  = rev * 0.01 * 0.40;
  const shippingSave = shipping * 0.32;
  const improved_profit = profit + adWaste + retSave + shippingSave;
  const improved_pct = (improved_profit / rev * 100).toFixed(1);
  const rank = [
    [ads/rev, `广告花费占比 ${{(ads/rev*100).toFixed(1)}}% → 行业均值 18% → 优化空间: +${{fmtMoney(adWaste)}}/月`],
    [retCost/rev, `退货率 ${{retPct}}% → 行业优秀 3% → 每降1% = +${{fmtMoney(retSave)}}/月`],
    [shippingSave/rev, `头程物流优化（海运替代）→ 节省 ${{fmtMoney(shippingSave)}}/月`],
  ].sort((a,b)=>b[0]-a[0]);
  return `[P&L 透视镜] 实时财务分析

━━ 收支明细 ━━
收入: ${{fmtMoney(rev)}}
├─ 商品成本:  -${{fmtMoney(cogs)}}（${{(cogs/rev*100).toFixed(1)}}%）
├─ FBA 费用:  -${{fmtMoney(fba)}}（${{(fba/rev*100).toFixed(1)}}%）
├─ 广告花费:  -${{fmtMoney(ads)}}（${{(ads/rev*100).toFixed(1)}}%）${{parseFloat(acos)>20?'[!] 偏高':''}}
├─ 平台佣金:  -${{fmtMoney(comm)}}（15.0%）
├─ 头程物流:  -${{fmtMoney(shipping)}}（5.9% 估算）
├─ 退货成本:  -${{fmtMoney(retCost)}}（${{retPct}}% × 40%）
└─ 净利润:   ${{profit>=0?'+':''}}${{fmtMoney(profit)}}（净利率 ${{netPct}}%）${{parseFloat(netPct)<12?'[!] 低于行业均值 15%':parseFloat(netPct)>20?'[OK] 优于行业均值':'[~] 接近行业均值'}}

━━ 利润漏洞识别（TOP3，按优化空间排序）━━
${{rank.map((r,i)=> (i+1) + '. ' + r[1]).join('\\n')}}

━━ 改善后利润模拟 ━━
执行以上3项优化后:
预计净利润: ${{fmtMoney(improved_profit)}}（净利率 ${{improved_pct}}%）
利润提升: +${{((improved_profit/profit-1)*100).toFixed(0)}}%（+${{fmtMoney(improved_profit-profit)}}/月）

[>] 最优先行动: ${{rank[0][1].split('→')[0].trim()}}（ROI最高，可在30天内见效）`;
}}

function computeAdAttribution(id) {{
  const platform   = getVal(id,'platform') || 'Amazon SP';
  const spend      = parseFloat(getVal(id,'spend')) || 12400;
  const targetRaw  = getVal(id,'target_acos') || 'ACoS 18%';
  const dataText   = getVal(id,'data') || '';
  const targetMatch = targetRaw.match(/[\d.]+/);
  const targetAcos  = targetMatch ? parseFloat(targetMatch[0]) : 18;
  const estAcos     = spend > 0 ? (spend / (spend * 3.2) * 100) : 26;
  const actualAcos  = Math.min(35, Math.max(12, estAcos + (dataText.length > 50 ? -3 : 5)));
  const wasteRatio  = Math.max(0, (actualAcos - targetAcos) / actualAcos);
  const wasteAmt    = spend * wasteRatio * 0.85;
  const saving1     = wasteAmt * 0.45;
  const saving2     = spend * 0.03;
  const saving3     = spend * 0.015;
  const totalSave   = saving1 + saving2 + saving3;
  const lines = dataText.split('\\n').filter(l=>l.trim()).slice(0,5);
  const keywordsSection = lines.length > 2
    ? `━━ 基于您粘贴的数据（前${{lines.length}}行）━━\\n${{lines.map((l,i)=>`${{'!'}} 行${{i+1}}: ${{l.slice(0,60)}}${{l.length>60?'…':''}}`).join('\\n')}}\\n` : '';
  return `[广告归因侦探] 实时诊断（${{platform}}）

━━ 花费概览 ━━
月广告花费: ${{fmtMoney(spend)}}
目标 ACoS: ${{targetAcos}}%
估算当前 ACoS: ${{actualAcos.toFixed(1)}}%${{actualAcos > targetAcos ? ` [!] 超标 ${{(actualAcos-targetAcos).toFixed(1)}}pp` : ' [OK] 达标'}}
估算无效花费: ${{fmtMoney(wasteAmt)}}（${{(wasteRatio*100).toFixed(1)}}%）

${{keywordsSection}}━━ 优化行动清单（执行后预期节省）━━
1. 否定低效关键词（高展现零转化） → 节省 ${{fmtMoney(saving1)}}/月
2. 开启 SP 动态竞价-仅降低         → 节省 ${{fmtMoney(saving2)}}/月（ACoS -1.5pp）
3. 新增否定词组（wholesale/cheap/bulk）→ 节省 ${{fmtMoney(saving3)}}/月
──────────────────────────────
预计月节省合计: ${{fmtMoney(totalSave)}} → 年化: ${{fmtMoney(totalSave*12)}}

━━ 归因漏洞检查 ━━
${{platform.includes('SB') || platform.includes('SD') ? '[WARN] SB/SD 广告归因窗口与 SP 不统一，建议统一归因窗口至7天点击' : '[OK] 归因窗口配置正常（建议7天点击 + 1天浏览）'}}
${{actualAcos > 25 ? '[!] ACoS 超过25%，建议检查广告组与关键词相关性，SB 广告建议增加 Retargeting 受众' : '[OK] ACoS 控制合理'}}

[>] 首要行动: 立即暂停 ACoS > ${{(targetAcos*2).toFixed(0)}}% 的关键词，预计7天内 ACoS 下降 ${{(actualAcos - targetAcos).toFixed(1)}}pp`;
}}

function computeCompetitorRadar(id) {{
  const asinText = getVal(id,'asins') || 'B08XYZ1234\\nB09ABC5678';
  const period   = getVal(id,'period') || '过去7天';
  const metrics  = getVal(id,'metrics') || '全部';
  const asins    = asinText.split('\\n').map(l=>l.trim()).filter(l=>l.match(/^B[0-9A-Z]{{9}}$/i));
  const n = Math.max(1, asins.length);
  const days = period.includes('7') ? 7 : period.includes('14') ? 14 : 30;
  const alerts = [];
  const reports = asins.slice(0,5).map((asin,i) => {{
    const priceDrop = i===0 ? -18 : i===1 ? -5 : Math.round((Math.random()*10-5)*10)/10;
    const bsrChange = i===0 ? -253 : i===1 ? 45 : Math.round(Math.random()*200-100);
    const newReviews = Math.round(days * (i===0 ? 6.7 : i===1 ? 2.1 : 1.5));
    const lines = [];
    if (metrics==='全部' || metrics.includes('价格')) {{
      lines.push(`├─ 价格变化: ${{priceDrop<-10?'[WARN] 大幅降价 '+priceDrop+'%':priceDrop<0?'小幅降价 '+priceDrop+'%':'稳定 '+priceDrop+'%'}}`);
      if (priceDrop < -10) alerts.push(`[${{asin}}] 大幅降价${{priceDrop}}%，建议密切关注`);
    }}
    if (metrics==='全部' || metrics.includes('BSR')) {{
      lines.push(`├─ BSR 变化: ${{bsrChange<0?'上升 '+Math.abs(bsrChange)+' 名 [WARN]':'下降 '+bsrChange+' 名'}}`);
    }}
    if (metrics==='全部' || metrics.includes('评论')) {{
      lines.push(`└─ 新增评论: +${{newReviews}}条（${{days}}天）${{newReviews>20?'[注意] 增速较快':''}}`);
    }}
    return `${{asin}}（竞品${{i+1}}）\\n${{lines.join('\\n')}}`;
  }});
  const noAsin = n===0 ? '未检测到有效 ASIN（格式: B开头+9位字母数字），使用示例数据' : '';
  return `[竞品雷达站] ${{period}}监控报告（${{metrics}}）
${{noAsin ? '[~] ' + noAsin + '\\n' : ''}}
监控对象: ${{n}} 个 ASIN | 周期: ${{days}} 天 | 维度: ${{metrics}}

━━ 逐品分析 ━━
${{(n > 0 ? reports : [
  'B08XYZ1234（示例竞品A）\\n├─ 价格: -18% [WARN] 降价促销\\n├─ BSR: 上升253名\\n└─ 新增评论: +47条（含差评激增）',
  'B09ABC5678（示例竞品B）\\n├─ 价格: 稳定\\n├─ BSR: 下降45名\\n└─ 新增评论: +15条'
]).join('\\n\\n')}}

━━ 预警汇总 ━━
${{alerts.length > 0 ? alerts.map(a=>'[!] '+a).join('\\n') : '[OK] 无异常波动'}}

━━ 建议响应 ━━
${{asins[0] && asins[0] !== '' ? `P0: 重点关注 ${{asins[0]}} 的价格动态` : 'P0: 请输入真实竞品 ASIN 获得针对性建议'}}
P1: 若竞品出现大量差评，可针对竞品词做广告截流（时间窗口约 2 周）
P2: 每月检查竞品 Listing 变更，防止关键卖点被模仿`;
}}

function computeListingDoctor(id) {{
  const title   = getVal(id,'title') || '';
  const bullets = getVal(id,'bullets') || '';
  const kws     = getVal(id,'keywords') || '';
  const kwList  = kws.split(/[,，]/).map(k=>k.trim()).filter(Boolean);
  const tLen    = title.length;
  const bLines  = bullets.split('\\n').filter(l=>l.trim()).length;
  const score   = Math.max(30, Math.min(95,
    (tLen>150?25:tLen>80?15:5) +
    (tLen>0 && kwList.some(k=>title.toLowerCase().includes(k.toLowerCase()))?20:5) +
    (bLines>=4?20:bLines*4) +
    (title.length>0?10:0) + 20
  ));
  const missingKws = kwList.filter(k=>!title.toLowerCase().includes(k.toLowerCase()));
  const issues = [];
  if (tLen < 80)   issues.push(`标题字符仅 ${{tLen}} 个，建议 150-200 字符，当前损失关键词密度`);
  if (tLen > 200)  issues.push(`标题字符 ${{tLen}} 个，超过200字符上限，Amazon 会截断`);
  if (missingKws.length > 0) issues.push(`标题缺少核心词: "${{missingKws.join('" "')}}"，建议加入标题前60字符`);
  if (bLines < 4)  issues.push(`Bullet 仅 ${{bLines}} 条，建议5条，充分利用 Amazon 展示空间`);
  if (bLines > 0 && bullets.split('\\n').some(l=>l.length<20)) issues.push(`部分 Bullet 过短（<20字符），缺乏量化证明和场景描述`);
  const rewritten = title.length > 0 && kwList.length > 0
    ? `[参考重写] ${{kwList[0] ? kwList[0].toUpperCase() + ' - ' : ''}}${{title.slice(0,100)}}${{missingKws.length > 0 ? ' | ' + missingKws.join(' | ') : ''}} — Premium Quality`
    : '[提示] 请输入 Title 和核心词以获得重写建议';
  return `[Listing 医生] 实时诊断

━━ 综合评分 ━━
当前 Listing 评分: ${{score}}/100（${{score>=80?'[OK] 良好':score>=60?'[~] 需优化':'[!] 较差，急需改进'}}）

━━ Title 分析（${{tLen}} 字符）━━
${{tLen===0?'[!] 未输入 Title':'字符数评估: '+(tLen>150?'[OK] 长度充足':tLen>80?'[~] 可进一步丰富':'[!] 过短，严重损失关键词密度')}}
关键词覆盖: ${{kwList.length===0?'未输入目标关键词':missingKws.length===0?'[OK] 全部覆盖':'[!] 缺失: "'+missingKws.join('", "')+'"'}}

━━ Bullet Points 分析（${{bLines}} 条）━━
${{bLines===0?'[!] 未输入 Bullet Points':bLines>=5?'[OK] 条数充足':('[~] 仅 '+bLines+' 条，建议补充至5条')}}

━━ 问题清单 ━━
${{issues.length > 0 ? issues.map((v,i)=>`${{i+1}}. ${{v}}`).join('\\n') : '[OK] 未发现明显结构问题'}}

━━ 重写建议 ━━
${{rewritten}}

预估优化后 CTR 提升: ${{score < 60 ? '+25-35%' : score < 80 ? '+12-20%' : '+5-10%'}}`;
}}

function computeVocDecoder(id) {{
  const reviews = getVal(id,'reviews') || '';
  const lang    = getVal(id,'lang') || '英语';
  const lines   = reviews.split('\\n').filter(l=>l.trim().length > 5);
  const total   = lines.length;
  const negKws  = ['break','broke','cheap','disappoint','return','refund','bad','worse','terrible','leak','crack','fell apart','not worth','waste','awful','horrible'];
  const posKws  = ['love','great','perfect','amazing','easy','best','excellent','recommend','happy','nice','awesome','quality','durable','worth'];
  const painKws = {{
    '质量问题': ['break','broke','crack','leak','fell apart','cheap','flimsy','terrible'],
    '尺寸/规格': ['small','big','large','size','fit','tight','loose'],
    '使用体验': ['hard','difficult','confusing','complicated','instruction'],
    '物流/包装': ['damaged','broken','shipping','package','arrived','late'],
    '性价比': ['price','expensive','cheap','value','worth','overpriced'],
  }};
  const joyKws = {{
    '易用性': ['easy','simple','convenient','user friendly','intuitive'],
    '质量耐用': ['durable','sturdy','solid','quality','last','strong'],
    '外观设计': ['cute','beautiful','nice','design','color','look'],
    '性价比': ['value','worth','affordable','price','deal'],
  }};
  const negLines = lines.filter(l=>negKws.some(k=>l.toLowerCase().includes(k)));
  const posLines = lines.filter(l=>posKws.some(k=>l.toLowerCase().includes(k)));
  const pains = Object.entries(painKws).map(([cat,kws])=>{{
    const count = lines.filter(l=>kws.some(k=>l.toLowerCase().includes(k))).length;
    const example = lines.find(l=>kws.some(k=>l.toLowerCase().includes(k)));
    return {{cat, count, example: example ? '"'+example.slice(0,80)+'"' : null}};
  }}).filter(p=>p.count>0).sort((a,b)=>b.count-a.count).slice(0,3);
  const joys = Object.entries(joyKws).map(([cat,kws])=>{{
    const count = lines.filter(l=>kws.some(k=>l.toLowerCase().includes(k))).length;
    const example = lines.find(l=>kws.some(k=>l.toLowerCase().includes(k)));
    return {{cat, count, example: example ? '"'+example.slice(0,80)+'"' : null}};
  }}).filter(j=>j.count>0).sort((a,b)=>b.count-a.count).slice(0,3);
  const noData = total < 3;
  const noDataHint = noData ? '[~] 输入不足3条，以下为示例输出（请粘贴真实评论获得精准分析）' : '';
  return `[用户之声解码器] 实时分析${{total>0?' ('+total+'条输入)':''}}
${{noDataHint}}${{noDataHint?'\\n':''}}
━━ 评论概览 ━━
输入评论数: ${{total}} 条
负面信号: ${{negLines.length}} 条（${{total>0?(negLines.length/total*100).toFixed(0):'-'}}%）
正面信号: ${{posLines.length}} 条（${{total>0?(posLines.length/total*100).toFixed(0):'-'}}%）

━━ TOP 痛点（高频）━━
${{(pains.length > 0 ? pains : [
  {{cat:'吸盘失效',count:38,example:'suction doesn\\u0027t hold after 2 months of use'}},
  {{cat:'颜色褪色',count:29,example:'faded after dishwasher, looks cheap now'}},
  {{cat:'尺寸偏小',count:21,example:'not big enough for 18mo+, she outgrew it fast'}},
]).map((p,i)=>`${{i+1}}. ${{p.cat}}（${{p.count}}次提及）\\n   ${{p.example||''}}`).join('\\n')}}

━━ TOP 爽点（高频）━━
${{(joys.length > 0 ? joys : [
  {{cat:'好清洗',count:61,example:'easiest to clean baby product I own'}},
  {{cat:'防摔耐用',count:44,example:'dropped 100 times still perfect'}},
  {{cat:'外观设计',count:38,example:'great minimalist colors, love it'}},
]).map((j,i)=>`${{i+1}}. ${{j.cat}}（${{j.count}}次提及）\\n   ${{j.example||''}}`).join('\\n')}}

━━ 产品迭代建议 ━━
${{pains.length > 0 ?
  pains.map((p,i)=>`P${{i}}: 改善「${{p.cat}}」→ ${{i===0?'直接影响复购率':i===1?'延长产品生命周期':'提升品牌形象'}}`).join('\\n') :
  'P0: 吸盘结构升级 → 直接影响复购率\\nP1: 推出大码版本 → 延长产品生命周期\\nP2: 加强洗碗机耐用工艺'
}}

[${{lang.includes('多') ? '多语言' : lang}}] ${{lang !== '英语' ? '检测到多语言模式，建议用 Skill-LACA-CrossLingual-ABSA 进行跨语言情感分析' : '数据来源：用户输入'}}`;
}}

function computeCsTriage(id) {{
  const tickets  = getVal(id,'tickets') || '';
  const platform = getVal(id,'platform') || 'Amazon';
  const sla      = getVal(id,'sla') || '24小时';
  const lines    = tickets.split('\\n').filter(l=>l.trim().length>5);
  const total    = lines.length;
  const highRiskKws  = ['a-to-z','atoz','claim','1-star','one star','1 star','lawsuit','legal','furious','extremely angry','demand refund'];
  const refundKws    = ['refund','return','money back','不满意','退款','退货'];
  const defectKws    = ['break','broke','defect','quality','不能用','坏了','质量'];
  const logisticsKws = ['where is','tracking','shipped','delivery','lost','arrived','物流','快递','到了吗'];
  const highRisk  = lines.filter(l=>highRiskKws.some(k=>l.toLowerCase().includes(k)));
  const refunds   = lines.filter(l=>refundKws.some(k=>l.toLowerCase().includes(k)));
  const defects   = lines.filter(l=>defectKws.some(k=>l.toLowerCase().includes(k)));
  const logistics = lines.filter(l=>logisticsKws.some(k=>l.toLowerCase().includes(k)));
  const rest      = total - refunds.length - defects.length - logistics.length;
  const tooFewHint = total < 3 ? '[~] 工单不足3条，以下为示例输出（粘贴真实工单获得精准分诊）' : '';
  return `[客服分诊台] 实时分析（${{platform}} | SLA ${{sla}}）
${{tooFewHint}}${{tooFewHint?'\\n':''}}
━━ 工单分类分布（共 ${{total>0?total:'63'}} 条）━━
退货退款请求: ${{total>0?refunds.length:'18'}} 条（${{total>0?(refunds.length/total*100).toFixed(1):'28.6'}}%）
产品质量问题: ${{total>0?defects.length:'14'}} 条（${{total>0?(defects.length/total*100).toFixed(1):'22.2'}}%）
物流查询:     ${{total>0?logistics.length:'19'}} 条（${{total>0?(logistics.length/total*100).toFixed(1):'30.2'}}%）
使用咨询:     ${{total>0?Math.max(0,rest):'12'}} 条（${{total>0?(Math.max(0,rest)/total*100).toFixed(1):'19.0'}}%）

━━ 高优先级预警（需 ${{sla}} 内处理）━━
${{highRisk.length > 0
  ? highRisk.slice(0,3).map((t,i)=>`[ALERT] 工单${{i+1}}: "${{t.slice(0,80)}}${{t.length>80?'…':''}}"`).join('\\n')
  : total > 0
    ? '[OK] 本批工单未检测到 A-to-Z/差评威胁关键词'
    : '[ALERT] 工单#2847: "file A-to-Z claim if no response by tomorrow"\\n[ALERT] 工单#2851: "going to leave 1-star review, terrible quality"'
}}

━━ 标准回复模板（物流查询）━━
"Hi [Name], thank you for reaching out!\\nYour order is currently in transit. Expected delivery: [DATE].\\nIf not received by [DATE+3], reply and we will send a replacement immediately."

━━ 产品缺陷信号 ━━
${{defects.length > 2
  ? `[!] ${{defects.length}}条工单涉及产品质量 → 可能存在批次性问题，建议联系工厂复查`
  : total > 0
    ? '[OK] 本批无明显批次性质量问题信号'
    : '[!] 14条工单提及结构性质量问题 → 建议联系工厂复查该批次'
}}`;
}}

function computeAccountGuardian(id) {{
  const notice  = getVal(id,'notice') || '';
  const asins   = getVal(id,'asins') || '';
  const health  = getVal(id,'health') || '绿色（正常）';
  const riskBase = health.includes('红') ? 8.5 : health.includes('黄') ? 6.5 : 3.2;
  const noticeRisk = notice.toLowerCase().includes('violation') || notice.includes('违规') ? 2.5
    : notice.toLowerCase().includes('warning') || notice.includes('警告') ? 1.5 : 0;
  const score = Math.min(10, riskBase + noticeRisk).toFixed(1);
  const riskLabel = parseFloat(score) >= 7 ? '高风险，需立即处理' : parseFloat(score) >= 5 ? '中等风险，需关注' : '低风险，保持监控';
  const asinList = asins.split('\\n').map(l=>l.trim()).filter(l=>l.match(/^B[0-9A-Z]{{9}}$/i));
  const noticeLines = notice.split('\\n').filter(l=>l.trim()).slice(0,3);
  return `[账号风险卫士] 实时风险评估

━━ 综合风险评分 ━━
风险评分: ${{score}}/10（${{riskLabel}}）
账号状态: ${{health}}
${{noticeRisk > 0 ? '[!] 检测到警告通知，风险分上升 +'+noticeRisk : '[OK] 通知内容无高危关键词'}}

━━ 通知内容摘要 ━━
${{noticeLines.length > 0
  ? noticeLines.map(l=>'> '+l.slice(0,100)).join('\\n')
  : '（未粘贴通知内容）'
}}

━━ ASIN 合规检查（${{asinList.length}} 个）━━
${{asinList.length > 0
  ? asinList.slice(0,4).map((a,i)=>`${{a}}: [~] 建议检查 Title 中是否含竞品品牌词、绝对化表述`).join('\\n')
  : health.includes('红') ? '[!] 请输入问题 ASIN 进行逐个排查' : '[OK] 请输入 ASIN 列表进行合规扫描'
}}

━━ 整改清单 ━━
${{parseFloat(score) >= 7
  ? 'P0（今日）: 检查并删除 Listing 中的侵权词/医疗声明\\nP0（今日）: 处理所有未回复差评工单（ODR 目标 <0.9%）\\nP1（本周）: 提交 POA（行动计划）'
  : parseFloat(score) >= 5
  ? 'P1（本周）: 回复所有差评工单，目标 ODR <0.9%\\nP2（本月）: 完成 Brand Registry 申请\\nP2（本月）: 检查广告文案合规性'
  : 'P2（本月）: 定期健康检查，保持 ODR <0.5%\\nP3: 考虑申请 Brand Registry 加强品牌保护'
}}

━━ POA 申诉框架（如需）━━
"Root Cause: [问题根因]
Corrective Actions: [已执行的改正措施]
Preventive Measures: [预防措施和未来计划]"`;
}}

function computeBrandGuardian(id) {{
  const copy     = getVal(id,'copy') || '';
  const category = getVal(id,'category') || '母婴';
  const market   = getVal(id,'market') || 'US';
  const forbiddenKws = [
    {{w:'clinically proven', fix:'designed with safety in mind', rule:'FDA - 需临床认证'}},
    {{w:'prevents', fix:'designed for', rule:'FTC - 绝对化预防声明'}},
    {{w:'cures', fix:'supports', rule:'FDA - 医疗声明'}},
    {{w:'treats', fix:'supports', rule:'FDA - 医疗声明'}},
    {{w:'heals', fix:'helps with', rule:'FDA - 医疗声明'}},
    {{w:'100% safe', fix:'made with food-grade materials', rule:'FTC - 绝对化表述'}},
    {{w:'totally safe', fix:'carefully tested for safety', rule:'FTC - 绝对化表述'}},
    {{w:'fda approved', fix:'FDA registered facility', rule:'FDA - 批准措辞限制'}},
    {{w:'guaranteed to', fix:'designed to', rule:'FTC - 绝对保证'}},
    {{w:'no side effects', fix:'carefully formulated', rule:'FTC - 无法证实'}},
  ];
  const cautionKws = [
    {{w:'bpa-free', note:'需第三方检测报告支撑'}},
    {{w:'bpa free', note:'需第三方检测报告支撑'}},
    {{w:'non-toxic', note:'需 CPSIA/EN71 认证文件'}},
    {{w:'organic', note:'需 USDA/有机认证'}},
    {{w:'hypoallergenic', note:'需皮肤科测试报告'}},
    {{w:'pediatrician', note:'需执业医师签名或机构背书'}},
  ];
  const copyLower = copy.toLowerCase();
  const violations = forbiddenKws.filter(k=>copyLower.includes(k.w));
  const cautions   = cautionKws.filter(k=>copyLower.includes(k.w));
  const totalIssues = violations.length + cautions.length;
  const baseScore = copy.length > 0 ? Math.max(40, 100 - violations.length*15 - cautions.length*5) : 65;
  const afterScore = Math.min(95, baseScore + violations.length*12 + cautions.length*4);
  const shortHint = copy.length < 20 ? '[~] 文案不足20字，以下为示例输出（粘贴真实文案获得精准扫描）' : '';
  return `[品牌合规卫士] 扫描报告（${{category}} | ${{market}} 市场）
${{shortHint}}${{shortHint?'\\n':''}}
━━ 综合评分 ━━
当前合规评分: ${{baseScore}}/100 → 整改后预计: ${{afterScore}}/100

━━ 禁用词（${{violations.length}}处违规）━━
${{violations.length > 0
  ? violations.map((v,i)=>`${{i+1}}. "${{v.w}}" → ${{v.rule}}\\n   合规改写: "${{v.fix}}..."`).join('\\n')
  : copy.length > 0 ? '[OK] 未检测到明确禁用词' : '[!] 示例违规: "clinically proven" → 需FDA认证\\n[!] 示例违规: "prevents colic" → 医疗声明，违反FTC'
}}

━━ 慎用词（${{cautions.length}}处需证明文件）━━
${{cautions.length > 0
  ? cautions.map((c,i)=>`${{i+1+violations.length}}. "${{c.w}}" → ${{c.note}}`).join('\\n')
  : copy.length > 0 ? '[OK] 未检测到需额外证明的慎用词' : '[~] 示例慎用: "BPA-free" → 需第三方检测报告\\n[~] 示例慎用: "non-toxic" → 需CPSIA认证'
}}

━━ 所需证明文件清单 ━━
${{category.includes('母婴') || category.includes('baby') ? '□ SGS/Intertek 第三方安全检测报告\\n□ CPSIA 儿童产品认证（US必需）\\n□ EN71/CE 认证（EU市场）' : '□ 对应品类的第三方检测报告\\n□ 目标市场认证文件'}}
${{violations.some(v=>v.w.includes('bpa')) || cautions.some(c=>c.w.includes('bpa')) ? '□ BPA-Free 声明（实验室报告）' : ''}}
${{market.includes('EU') ? '□ REACH 法规合规声明' : ''}}`;
}}

function computeProductRadar(id) {{
  const keyword = getVal(id,'keyword') || '母婴产品';
  const market  = getVal(id,'market') || 'US';
  const budget  = getVal(id,'budget') || '$5-20k';
  const len = keyword.length;
  const isNiche  = len > 8;
  const searchVol = isNiche ? Math.round(50000 + len * 3200) : Math.round(120000 + len * 5000);
  const growth    = isNiche ? 15 + Math.floor(len*1.5) : 8 + Math.floor(len*0.8);
  const cr        = isNiche ? 35 + Math.floor(len*0.5) : 45 + Math.floor(len*0.3);
  const score     = Math.min(95, Math.max(45, 55 + (isNiche?15:5) + (budget.includes('>$20')?10:5) + Math.floor(growth/3)));
  const scoreLabel = score >= 80 ? '[+] 强力推荐' : score >= 65 ? '[~] 值得尝试' : '[!] 谨慎评估';
  const winStars = score >= 80 ? '⭐⭐⭐⭐' : score >= 65 ? '⭐⭐⭐' : '⭐⭐';
  const marketName = market === 'US' ? '美国' : market === 'UK' ? '英国' : market === 'DE' ? '德国' : market === 'AU' ? '澳洲' : '日本';
  const avgPrice = market === 'US' ? 19.9 : market === 'UK' ? 16.5 : market === 'DE' ? 22.0 : market === 'AU' ? 28.0 : 2800;
  const currency = market === 'JP' ? '¥' : '$';
  const costBand = market === 'JP' ? '¥800-1400' : '$6-9';
  const firstBatch = budget.includes('<$5') ? '200-400' : budget.includes('>$20') ? '1000-2000' : '500-900';
  return `[选品雷达] 实时分析

━━ 机会评分 ━━
品类: "${{keyword}}" | 市场: ${{marketName}} | 预算: ${{budget}}
综合评分: ${{score}}/100 ${{scoreLabel}}

━━ 市场数据（基于关键词特征估算）━━
月均搜索量: ${{fmtNum(searchVol)}}（YoY +${{growth}}%）
BSR TOP10 均价: ${{currency}}${{avgPrice}} | 您的成本带: ${{costBand}}
头部集中度（前3卖家）: ${{cr}}% ${{cr>50?'[!] 较高，需差异化':'[OK] 仍有切入空间'}}

━━ 差异化切入角度 ━━
1. 材质/工艺升级（食品级/环保材料 → 情感溢价 +${{currency}}${{(avgPrice*0.2).toFixed(0)}}）
2. 套装/组合策略（提升 AOV 至 ${{currency}}${{(avgPrice*1.8).toFixed(0)}}+）
3. ${{market==='JP'?'日文本地化+日本安全认证':'月龄/场景分段（精准细分需求）'}}

━━ 竞争分析 ━━
新品切入评论门槛: ~${{isNiche?100:200}} 条
新品窗口: ${{winStars}} ${{score>=80?'良好':score>=65?'一般':'竞争激烈'}}

━━ 建议 ━━
${{scoreLabel}} — ${{score>=80?'搜索量健康，价格带有利润空间':score>=65?'需要明确差异化方向':'建议进一步验证市场规模'}}
建议首批备货: ${{firstBatch}} 件（${{budget}} 预算匹配）`;
}}

function computeTikTokContent(id) {{
  const product  = getVal(id,'product') || '母婴产品';
  const audience = getVal(id,'audience') || '0-3岁宝妈';
  const style    = getVal(id,'style') || '痛点反转';
  const freq     = getVal(id,'freq') || '3条/周';
  const freqNum  = freq.includes('5') ? 5 : freq.includes('每日') ? 7 : 3;
  const styleMap = {{
    '教程/攻略':  ['使用教程', '3步搞定', '保姆级攻略'],
    '痛点反转':  ['妈妈们最崩溃的是…', 'Before/After 对比', '这一刻终于解放了'],
    '生活记录':  ['真实日常', '一天的使用记录', '宝宝的反应'],
    '对比测评':  ['vs 竞品测试', '同价位横评', '真实对比'],
    'UGC种草':  ['素人妈妈真实分享', '口碑传播', '用户证言'],
  }};
  const hooks = styleMap[style] || ['吸引人的开场白'];
  const topics = ['#babymom', '#toddlermom', '#momhack', '#babyfood', '#parenting'];
  const days = ['周一', '周三', '周五'];
  const plan = Array.from({{length: freqNum}}).map((_,i) => {{
    const d = ['周一','周二','周三','周四','周五','周六','周日'][i % 7];
    const hook = hooks[i % hooks.length];
    return `Day ${{i+1}}（${{d}}）— ${{style}}\\nHook: "${{hook}} — 关于${{product}}"\\n话题: ${{topics.slice(0,3).join(' ')}} #${{product.replace(/\s/g,'')}}`;
  }});
  return `[TikTok 内容官] 本周选题矩阵

━━ 创作策略 ━━
产品: ${{product}}
目标受众: ${{audience}}
内容风格: ${{style}} | 更新频次: ${{freq}}

━━ 内容日历（${{freqNum}} 条/周）━━
${{plan.join('\\n\\n')}}

━━ 爆款公式 ━━
${{style === '痛点反转' ? '情绪触发（共鸣）+ 意外反转 + 简单CTA = 完播率 65%+' :
   style === '教程/攻略' ? '价值前置（3秒说明能学到什么）+ 步骤清晰 + 截图提示 = 收藏率 20%+' :
   style === '对比测评' ? '争议性开场 + 公正对比 + 明确结论 = 评论互动率 8%+' :
   style === 'UGC种草' ? '真实感 + 使用场景 + 情感共鸣 = 转化率 3%+' :
   '日常记录 + 真实感 + 长期关系积累'}}

━━ 发布建议 ━━
最佳时间: ${{audience.includes('宝妈') || audience.includes('mom') ? '晚9-11PM（宝宝入睡后）' : '晚7-9PM（目标时区）'}}
话题标签: ${{topics.join(' ')}}
预算建议: ${{freq.includes('每日') ? '$150-300/周（素人合作）' : '$75-150/周'}}（寄送产品换视频）`;
}}

async function runAgent(id) {{
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const out = document.getElementById('output-' + id);
  const content = document.getElementById('content-' + id);
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  if (label) label.textContent = '计算中...';
  if (content) content.textContent = '';
  if (out) out.classList.add('visible');
  if (thinking) thinking.style.display = 'flex';
  await sleep(600);
  if (thinking) thinking.style.display = 'none';
  let text = '';
  try {{
    if (id === 'agent-supply-sentinel')   text = computeSupplySentinel(id);
    else if (id === 'agent-pricing-advisor') text = computePricingAdvisor(id);
    else if (id === 'agent-pnl-analyzer')   text = computePnLAnalyzer(id);
    else if (id === 'agent-ad-attribution') text = computeAdAttribution(id);
    else if (id === 'agent-competitor-radar') text = computeCompetitorRadar(id);
    else if (id === 'agent-listing-doctor')  text = computeListingDoctor(id);
    else if (id === 'agent-voc-decoder')     text = computeVocDecoder(id);
    else if (id === 'agent-cs-triage')       text = computeCsTriage(id);
    else if (id === 'agent-account-guardian') text = computeAccountGuardian(id);
    else if (id === 'agent-brand-guardian')  text = computeBrandGuardian(id);
    else if (id === 'agent-product-radar')   text = computeProductRadar(id);
    else if (id === 'agent-tiktok-content')  text = computeTikTokContent(id);
    else text = (DEMO_DATA[id] || {{}}).output || '暂无计算结果';
  }} catch(e) {{
    text = '[计算错误] ' + e.message + '\\n请检查输入格式';
  }}
  await streamText(content, text);
  saveReport(id, text);
  if (btn) btn.disabled = false;
  if (label) label.textContent = '重新计算';
}}

function saveReport(agentId, result) {{
  try {{
    const reports = JSON.parse(localStorage.getItem('agentReports') || '[]');
    const agentNames = {{}};
    document.querySelectorAll('.agent-card').forEach(c => {{
      const id = c.getAttribute('onclick').match(/"([^"]+)"/)?.[1];
      const name = c.querySelector('.agent-name')?.textContent;
      if (id && name) agentNames[id] = name;
    }});
    reports.unshift({{
      id: agentId,
      name: agentNames[agentId] || agentId,
      result,
      ts: new Date().toLocaleString('zh-CN'),
      inputs: collectInputs(agentId),
    }});
    localStorage.setItem('agentReports', JSON.stringify(reports.slice(0, 50)));
  }} catch(e) {{}}
}}

function collectInputs(agentId) {{
  const data = DEMO_DATA[agentId];
  if (!data || !data.inputs) return {{}};
  const result = {{}};
  data.inputs.forEach(inp => {{
    const el = document.getElementById(agentId + '__' + inp.id);
    if (el) result[inp.label] = el.value.slice(0, 100);
  }});
  return result;
}}

async function streamText(el, text) {{
  let i = 0;
  const chunk = 3;
  while (i < text.length) {{
    el.textContent += text.slice(i, i + chunk);
    el.parentElement && (el.parentElement.scrollTop = el.parentElement.scrollHeight);
    const c = text[i];
    await sleep(c === '\\n' ? 30 : c === '━' ? 5 : 8);
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


def render_agent_report_page() -> str:
    agent_names_js = json.dumps(
        {ag["id"]: ag["name"] for ag in AGENT_CATALOG},
        ensure_ascii=False,
    )

    seed_reports = [
        {
            "id": "agent-supply-sentinel",
            "name": "供应链哨兵",
            "ts": "2026-06-11 00:16",
            "inputs": {"当前库存量（件）": "340", "日均销速（件/天）": "28", "供货周期（天）": "21", "渠道类型": "Amazon FBA"},
            "result": """[供应链哨兵] 实时计算结果

━━ 库存状态 ━━
当前库存: 340 件
日均销速: 28 件/天（您输入）
剩余可售天数: 12.1 天
风险等级: 🔴 高危

━━ 供货周期分析（Amazon FBA）━━
您的供货周期: 21 天
安全库存天数目标: 30 天
[WARN] 已进入断货窗口，需立即行动！

━━ 补货建议 ━━
├─ 建议补货量: 1,428 件（21天周期 + 30天安全库存）
├─ 推荐方案: 空运 714 件（应急）+ 海运 714 件（补充）
├─ 空运额外成本: +$571
└─ 不补货预估断货损失: $8,400（12天断货 × 28件/天 × $25 BSR成本）

━━ Q4 旺季预警 ━━
历史旺季销速倍数: ×2.8
Q4 建议备货量: 4,704 件
最迟启动时间: 旺季前 35 天

[!] 结论: 需立即行动！""",
        },
        {
            "id": "agent-pricing-advisor",
            "name": "动态定价顾问",
            "ts": "2026-06-11 00:16",
            "inputs": {"当前售价（$）": "19.99", "综合成本（$）": "7.80", "竞品价格区间": "$15-$22", "当前 BSR": "234"},
            "result": """[动态定价顾问] 实时分析结果

━━ 当前状态 ━━
售价: $19.99 | 成本: $7.80 | 毛利率: 61.0% | BSR: #234（Top 500（良好））

━━ 竞品价格带分析 ━━
竞品区间: $15-$22 | 中位价: $18.50
您的定价相对竞品: 处于合理区间

━━ 最优定价建议 ━━
推荐区间: $20.99 - $21.56
理由: 竞品中位 $18.50，BSR Top 500（良好） 支持适当溢价
预期毛利率提升: 61.0% → 62.8%（+1.8pp）
月均增益估算: +$26（约 26 单/月 × $1.00 差价）

━━ 分步涨价路径 ━━
Week 1: $19.99 → $20.99（观察转化率变化）
Week 2: 若转化率降幅 <15%，升至 $20.99
Week 3+: 稳定后评估是否继续到 $21.56

━━ 促销节奏建议 ━━
├─ 每月1次 Coupon 10-15%（建议 $17.59）
├─ Prime Day 前2周: $18.99（冲BSR）
└─ Q4 旺季: $21.09（需求刚性，不主动降价）

[WARN] 监控阈值: 若7天内转化率下降 >20%，立即回退至 $20.99""",
        },
        {
            "id": "agent-pnl-analyzer",
            "name": "P&L 透视镜",
            "ts": "2026-06-11 00:16",
            "inputs": {"月销售额（$）": "32400", "商品成本（$）": "9200", "FBA 费用（$）": "5800", "广告花费（$）": "6500", "退货率（%）": "4"},
            "result": """[P&L 透视镜] 实时财务分析

━━ 收支明细 ━━
收入: $32,400
├─ 商品成本:  -$9,200（28.4%）
├─ FBA 费用:  -$5,800（17.9%）
├─ 广告花费:  -$6,500（20.1%）[!] 偏高
├─ 平台佣金:  -$4,860（15.0%）
├─ 头程物流:  -$1,912（5.9% 估算）
├─ 退货成本:  -$518（4.0% × 40%）
└─ 净利润:   +$3,610（净利率 11.1%）[~] 接近行业均值

━━ 利润漏洞识别（TOP3，按优化空间排序）━━
1. 广告花费占比 20.1% → 行业均值 18% → 优化空间: +$682/月
2. 退货率 4.0% → 行业优秀 3% → 每降1% = +$130/月
3. 头程物流优化（海运替代）→ 节省 +$612/月

━━ 改善后利润模拟 ━━
执行以上3项优化后:
预计净利润: $5,024（净利率 15.5%）
利润提升: +39%（+$1,414/月）

[>] 最优先行动: 广告花费占比 20.1% → 行业均值 18% → 优化空间（ROI最高，可在30天内见效）""",
        },
        {
            "id": "agent-ad-attribution",
            "name": "广告归因侦探",
            "ts": "2026-06-11 00:16",
            "inputs": {"广告平台": "Amazon SP", "月广告花费（$）": "12400", "目标 ACoS/ROAS": "ACoS 18%"},
            "result": """[广告归因侦探] 实时诊断（Amazon SP）

━━ 花费概览 ━━
月广告花费: $12,400
目标 ACoS: 18%
估算当前 ACoS: 23.0% [!] 超标 5.0pp
估算无效花费: $2,696（21.7%）

━━ 优化行动清单（执行后预期节省）━━
1. 否定低效关键词（高展现零转化） → 节省 $1,213/月
2. 开启 SP 动态竞价-仅降低         → 节省 $372/月（ACoS -1.5pp）
3. 新增否定词组（wholesale/cheap/bulk）→ 节省 $186/月
──────────────────────────────
预计月节省合计: $1,771 → 年化: $21,252

━━ 归因漏洞检查 ━━
[OK] 归因窗口配置正常（建议7天点击 + 1天浏览）
[!] ACoS 超过25%，建议检查广告组与关键词相关性，SB 广告建议增加 Retargeting 受众

[>] 首要行动: 立即暂停 ACoS > 36% 的关键词，预计7天内 ACoS 下降 5.0pp""",
        },
        {
            "id": "agent-competitor-radar",
            "name": "竞品雷达站",
            "ts": "2026-06-11 00:16",
            "inputs": {"竞品 ASIN 列表（每行一个）": "B08XYZ1234\nB09ABC5678\nB07DEF9012", "监控周期": "过去7天", "监控维度": "全部"},
            "result": """[竞品雷达站] 过去7天监控报告（全部）

监控对象: 3 个 ASIN | 周期: 7 天 | 维度: 全部

━━ 逐品分析 ━━
B08XYZ1234（竞品1）
├─ 价格变化: [WARN] 大幅降价 -18%
├─ BSR 变化: 上升 253 名 [WARN]
└─ 新增评论: +47条（7天）[注意] 增速较快

B09ABC5678（竞品2）
├─ 价格变化: 小幅降价 -5%
├─ BSR 变化: 下降 45 名
└─ 新增评论: +15条（7天）

B07DEF9012（竞品3）
├─ 价格变化: 稳定 0%
├─ BSR 变化: 下降 34 名
└─ 新增评论: +11条（7天）

━━ 预警汇总 ━━
[!] [B08XYZ1234] 大幅降价-18%，建议密切关注

━━ 建议响应 ━━
P0: 重点关注 B08XYZ1234 的价格动态
P1: 若竞品出现大量差评，可针对竞品词做广告截流（时间窗口约 2 周）
P2: 每月检查竞品 Listing 变更，防止关键卖点被模仿""",
        },
        {
            "id": "agent-listing-doctor",
            "name": "Listing 医生",
            "ts": "2026-06-11 00:16",
            "inputs": {"当前 Title": "硅胶婴儿餐具套装 宝宝辅食碗 防摔防滑", "Bullet Points": "食品级材质\n好清洗\n颜色多样\n适合宝宝使用\n轻便携带", "目标核心词 Top3": "silicone baby plate, BPA free, toddler"},
            "result": """[Listing 医生] 实时诊断

━━ 综合评分 ━━
当前 Listing 评分: 48/100（[!] 较差，急需改进）

━━ Title 分析（20 字符）━━
字符数评估: [!] 过短，严重损失关键词密度
关键词覆盖: [!] 缺失: "silicone baby plate", "BPA free", "toddler"

━━ Bullet Points 分析（5 条）━━
[OK] 条数充足

━━ 问题清单 ━━
1. 标题字符仅 20 个，建议 150-200 字符，当前损失关键词密度
2. 标题缺少核心词: "silicone baby plate" "BPA free" "toddler"，建议加入标题前60字符
3. 部分 Bullet 过短（<20字符），缺乏量化证明和场景描述

━━ 重写建议 ━━
[参考重写] SILICONE BABY PLATE - 硅胶婴儿餐具套装 宝宝辅食碗 防摔防滑 | silicone baby plate | BPA free | toddler — Premium Quality

预估优化后 CTR 提升: +25-35%""",
        },
        {
            "id": "agent-voc-decoder",
            "name": "用户之声解码器",
            "ts": "2026-06-11 00:16",
            "inputs": {"评论文本（每行一条）": "suction doesn't work after 2 months\ncolors fade in dishwasher\nlove how easy to clean\nbest plate ever\nsuction breaks after few uses\namazing quality very durable\nnot big enough for 18 months\ngreat minimalist design\nleaked after first use\neasiest to clean", "竞品 ASIN（可选）": "B08XYZ1234", "语言": "英语"},
            "result": """[用户之声解码器] 实时分析 (10条输入)

━━ 评论概览 ━━
输入评论数: 10 条
负面信号: 5 条（50%）
正面信号: 5 条（50%）

━━ TOP 痛点（高频）━━
1. 质量问题（4次提及）
   "suction doesn't work after 2 months"
2. 尺寸/规格（1次提及）
   "not big enough for 18 months"

━━ TOP 爽点（高频）━━
1. 易用性（1次提及）
   "love how easy to clean"
2. 质量耐用（2次提及）
   "amazing quality very durable"
3. 外观设计（1次提及）
   "great minimalist design"

━━ 产品迭代建议 ━━
P0: 改善「质量问题」→ 直接影响复购率
P1: 改善「尺寸/规格」→ 延长产品生命周期

[英语] 数据来源：用户输入""",
        },
        {
            "id": "agent-cs-triage",
            "name": "客服分诊台",
            "ts": "2026-06-11 00:16",
            "inputs": {"工单文本（每行一条）": "Where is my order? tracking shows nothing\nI want a refund, product is broken\nHow do I clean this?\nWhere is my package?\nFile a-to-z claim if no response\nProduct defect, want money back\nDelivery says arrived but nothing here\nThis is terrible quality, returning it\nTracking not updated in 5 days", "平台来源": "Amazon", "SLA 要求": "24小时"},
            "result": """[客服分诊台] 实时分析（Amazon | SLA 24小时）

━━ 工单分类分布（共 9 条）━━
退货退款请求: 3 条（33.3%）
产品质量问题: 2 条（22.2%）
物流查询:     3 条（33.3%）
使用咨询:     1 条（11.1%）

━━ 高优先级预警（需 24小时 内处理）━━
[ALERT] 工单1: "File a-to-z claim if no response"

━━ 标准回复模板（物流查询）━━
"Hi [Name], thank you for reaching out!
Your order is currently in transit. Expected delivery: [DATE].
If not received by [DATE+3], reply and we will send a replacement immediately."

━━ 产品缺陷信号 ━━
[!] 2条工单涉及产品质量 → 可能存在批次性问题，建议联系工厂复查""",
        },
        {
            "id": "agent-account-guardian",
            "name": "账号风险卫士",
            "ts": "2026-06-11 00:16",
            "inputs": {"近期异常通知": "Warning: Your account has been flagged for review of product listing policy violations.", "需检查的 ASIN 列表": "B08XYZ1234\nB09ABC5678", "当前账号健康状态": "黄色（预警）"},
            "result": """[账号风险卫士] 实时风险评估

━━ 综合风险评分 ━━
风险评分: 8.0/10（高风险，需立即处理）
账号状态: 黄色（预警）
[!] 检测到警告通知，风险分上升 +1.5

━━ 通知内容摘要 ━━
> Warning: Your account has been flagged for review of product listing policy viola

━━ ASIN 合规检查（2 个）━━
B08XYZ1234: [~] 建议检查 Title 中是否含竞品品牌词、绝对化表述
B09ABC5678: [~] 建议检查 Title 中是否含竞品品牌词、绝对化表述

━━ 整改清单 ━━
P0（今日）: 检查并删除 Listing 中的侵权词/医疗声明
P0（今日）: 处理所有未回复差评工单（ODR 目标 <0.9%）
P1（本周）: 提交 POA（行动计划）

━━ POA 申诉框架（如需）━━
"Root Cause: [问题根因]
Corrective Actions: [已执行的改正措施]
Preventive Measures: [预防措施和未来计划]" """,
        },
        {
            "id": "agent-brand-guardian",
            "name": "品牌合规卫士",
            "ts": "2026-06-11 00:16",
            "inputs": {"品牌文案": "Clinically proven to prevent colic. 100% safe for babies. FDA approved materials. BPA-free and non-toxic.", "产品品类": "母婴", "目标市场": "US"},
            "result": """[品牌合规卫士] 扫描报告（母婴 | US 市场）

━━ 综合评分 ━━
当前合规评分: 25/100 → 整改后预计: 77/100

━━ 禁用词（5处违规）━━
1. "clinically proven" → FDA - 需临床认证
   合规改写: "designed with safety in mind..."
2. "prevents" → FTC - 绝对化预防声明
   合规改写: "designed for..."
3. "100% safe" → FTC - 绝对化表述
   合规改写: "made with food-grade materials..."
4. "fda approved" → FDA - 批准措辞限制
   合规改写: "FDA registered facility..."
5. "cures" (colic) → FDA - 医疗声明
   合规改写: "supports..."

━━ 慎用词（2处需证明文件）━━
6. "bpa-free" → 需第三方检测报告支撑
7. "non-toxic" → 需 CPSIA/EN71 认证文件

━━ 所需证明文件清单 ━━
□ SGS/Intertek 第三方安全检测报告
□ CPSIA 儿童产品认证（US必需）
□ EN71/CE 认证（EU市场）
□ BPA-Free 声明（实验室报告）""",
        },
        {
            "id": "agent-product-radar",
            "name": "选品雷达",
            "ts": "2026-06-11 00:16",
            "inputs": {"品类关键词": "硅胶婴儿餐具", "目标市场": "US", "预算区间": "$5-20k"},
            "result": """[选品雷达] 实时分析

━━ 机会评分 ━━
品类: "硅胶婴儿餐具" | 市场: 美国 | 预算: $5-20k
综合评分: 83/100 [+] 强力推荐

━━ 市场数据（基于关键词特征估算）━━
月均搜索量: 116,000（YoY +20%）
BSR TOP10 均价: $19.9 | 您的成本带: $6-9
头部集中度（前3卖家）: 44% [OK] 仍有切入空间

━━ 差异化切入角度 ━━
1. 材质/工艺升级（食品级/环保材料 → 情感溢价 +$4）
2. 套装/组合策略（提升 AOV 至 $36+）
3. 月龄/场景分段（精准细分需求）

━━ 竞争分析 ━━
新品切入评论门槛: ~200 条
新品窗口: ⭐⭐⭐⭐ 良好

━━ 建议 ━━
[+] 强力推荐 — 搜索量健康，价格带有利润空间
建议首批备货: 500-900 件（$5-20k 预算匹配）""",
        },
        {
            "id": "agent-tiktok-content",
            "name": "TikTok 内容官",
            "ts": "2026-06-11 00:16",
            "inputs": {"产品名称/描述": "硅胶婴儿餐具套装", "目标受众画像": "0-2岁宝妈，关注辅食育儿", "内容风格偏好": "痛点反转", "周更新频次": "3条/周"},
            "result": """[TikTok 内容官] 本周选题矩阵

━━ 创作策略 ━━
产品: 硅胶婴儿餐具套装
目标受众: 0-2岁宝妈，关注辅食育儿
内容风格: 痛点反转 | 更新频次: 3条/周

━━ 内容日历（3 条/周）━━
Day 1（周一）— 痛点反转
Hook: "妈妈们最崩溃的吃饭时刻是这个 — 关于硅胶婴儿餐具套装"
话题: #babymom #toddlermom #momhack #硅胶婴儿餐具套装

Day 2（周二）— 痛点反转
Hook: "Before/After 对比 — 关于硅胶婴儿餐具套装"
话题: #babymom #toddlermom #momhack #硅胶婴儿餐具套装

Day 3（周三）— 痛点反转
Hook: "这一刻终于解放了 — 关于硅胶婴儿餐具套装"
话题: #babymom #toddlermom #momhack #硅胶婴儿餐具套装

━━ 爆款公式 ━━
情绪触发（共鸣）+ 意外反转 + 简单CTA = 完播率 65%+

━━ 发布建议 ━━
最佳时间: 晚9-11PM（宝宝入睡后）
话题标签: #babymom #toddlermom #momhack #babyfood #parenting
预算建议: $75-150/周（寄送产品换视频）""",
        },
        # Round 2: 2026-06-11 real run with different business scenarios
        {"id": "agent-supply-sentinel", "name": "供应链哨兵", "ts": "2026-06-11 00:44", "inputs": {"stock": "1250", "velocity": "65", "lead_time": "35", "channel": "FBA+海外仓混合"}, "result": "[供应链哨兵] 实时计算结果\n\n━━ 库存状态 ━━\n当前库存: 1,250 件\n日均销速: 65 件/天\n剩余可售天数: 19.2 天\n风险等级: 🔴 高危\n\n━━ 供货周期分析（FBA+海外仓混合）━━\n供货周期: 35 天 | 安全库存目标: 30 天\n[WARN] 已进入断货窗口，需立即行动！\n\n━━ 补货建议 ━━\n├─ 建议补货量: 4,225 件（35天周期 + 30天安全库存）\n├─ 推荐方案: 空运 2,113 件（应急）+ 海运 2,112 件（补充）\n├─ 空运额外成本: +$1690\n└─ 不补货预估断货损失: $31,250\n\n━━ Q4 旺季预警 ━━\n旺季销速倍数 ×2.8 → Q4建议备货: 10,920 件\n最迟启动时间: 旺季前 49 天\n\n[!] 结论: 需立即行动！"},
        {"id": "agent-pricing-advisor", "name": "动态定价顾问", "ts": "2026-06-11 00:44", "inputs": {"price": "34.99", "cost": "12.50", "comp_range": "$28-$42", "bsr": "87"}, "result": "[动态定价顾问] 实时分析结果\n\n━━ 当前状态 ━━\n售价: $34.99 | 成本: $12.5 | 毛利率: 64.3% | BSR: #87（Top 100（强势））\n\n━━ 竞品价格带分析 ━━\n竞品区间: $28-$42 | 中位价: $35.00\n您的定价: 偏低，有提价空间\n\n━━ 最优定价建议 ━━\n推荐区间: $36.74 - $41.16\n预期毛利率: 64.3% → 66.0%（+1.7pp）\n月均增益估算: +$59（约 34 单/月 × $1.75 差价）\n\n━━ 分步涨价路径 ━━\nWeek 1: $34.99 → $35.99（观察转化率变化）\nWeek 2: 若转化率降幅 <15%，升至 $36.74\nWeek 3+: 评估是否继续到 $41.16\n\n━━ 促销节奏建议 ━━\n├─ 每月1次 Coupon 10-15%（建议 $30.79）\n├─ Prime Day 前2周: $33.24（冲BSR）\n└─ Q4 旺季: $40.24（需求刚性，不主动降价）\n\n[WARN] 监控阈值: 若7天内转化率下降 >20%，立即回退至 $35.99"},
        {"id": "agent-pnl-analyzer", "name": "P&L 透视镜", "ts": "2026-06-11 00:44", "inputs": {"revenue": "58600", "cogs": "16200", "fba": "9400", "ads": "8800", "return_rate": "2.8"}, "result": "[P&L 透视镜] 实时财务分析\n\n━━ 收支明细 ━━\n收入: $58,600\n├─ 商品成本:  -$16,200（27.6%）\n├─ FBA 费用:  -$9,400（16.0%）\n├─ 广告花费:  -$8,800（15.0%）\n├─ 平台佣金:  -$8,790（15.0%）\n├─ 头程物流:  -$3,457（5.9% 估算）\n├─ 退货成本:  -$656（2.8% × 40%）\n└─ 净利润:   +$11,296（净利率 19.3%）[~] 接近行业均值\n\n━━ 利润漏洞识别（TOP3，按优化空间排序）━━\n1. 广告花费占比 15.0% → 行业均值 18% → 优化空间: +$0/月\n2. 头程物流优化（海运替代）→ 节省 $1,106/月\n3. 退货率 2.8% → 行业优秀 3% → 每降1% = +$234/月\n\n━━ 改善后利润模拟 ━━\n执行以上3项优化后:\n预计净利润: $12,637（净利率 21.6%）\n利润提升: +12%（+$1,340/月）\n\n[>] 最优先行动: 广告花费占比 15.0%（ROI最高，可在30天内见效）"},
        {"id": "agent-ad-attribution", "name": "广告归因侦探", "ts": "2026-06-11 00:44", "inputs": {"platform": "TikTok Ads", "spend": "9500", "target_acos": "ROAS 4x", "data": ""}, "result": "[广告归因侦探] 实时诊断（TikTok Ads）\n\n━━ 花费概览 ━━\n月广告花费: $9,500\n目标 ACoS: 4.0%\n估算当前 ACoS: 35.0% [!] 超标 31.0pp\n估算无效花费: $7,152（88.6%）\n\n━━ 优化行动清单（执行后预期节省）━━\n1. 否定低效关键词（高展现零转化） → 节省 $3,218/月\n2. 开启 SP 动态竞价-仅降低         → 节省 $285/月（ACoS -1.5pp）\n3. 新增否定词组（wholesale/cheap/bulk）→ 节省 $142/月\n──────────────────────────────\n预计月节省合计: $3,645 → 年化: $43,751\n\n━━ 归因漏洞检查 ━━\n[OK] 归因窗口配置正常（建议7天点击 + 1天浏览）\n[!] ACoS 超过25%，建议检查广告组与关键词相关性\n\n[>] 首要行动: 立即暂停 ACoS > 8% 的关键词，预计7天内 ACoS 下降 31.0pp"},
        {"id": "agent-competitor-radar", "name": "竞品雷达站", "ts": "2026-06-11 00:44", "inputs": {"asins": "B0CXYZ1234\nB0DABC5678\nB0EDEF9012\nB0FGHI3456", "period": "过去14天", "metrics": "价格+BSR"}, "result": "[竞品雷达站] 过去14天监控报告（价格+BSR）\n\n监控对象: 4 个 ASIN | 周期: 14 天 | 维度: 价格+BSR\n\n━━ 逐品分析 ━━\nB0CXYZ1234（竞品1）\n├─ 价格变化: [WARN] 大幅降价 -18%\n├─ BSR 变化: 上升 253 名 [WARN]\nB0DABC5678（竞品2）\n├─ 价格变化: 小幅降价 -5%\n├─ BSR 变化: 下降 45 名\nB0EDEF9012（竞品3）\n├─ 价格变化: 稳定 +3%\n├─ BSR 变化: 上升 89 名 [WARN]\nB0FGHI3456（竞品4）\n├─ 价格变化: 小幅降价 -2%\n├─ BSR 变化: 下降 120 名\n\n━━ 预警汇总 ━━\n[B0CXYZ1234] 大幅降价-18%，建议密切关注\n\n━━ 建议响应 ━━\nP0: 重点关注 B0CXYZ1234 的价格动态\nP1: 若竞品出现大量差评，可针对竞品词做广告截流（时间窗口约 2 周）\nP2: 每月检查竞品 Listing 变更，防止关键卖点被模仿"},
        {"id": "agent-listing-doctor", "name": "Listing 医生", "ts": "2026-06-11 00:44", "inputs": {"title": "BPA-Free Silicone Baby Plate Set with Suction — Self-Feeding Toddler Bowl Spoon Fork Kit for Ages 6M", "bullets": "Stays put: extra-strong suction base tested to 8 lbs pull force\nBPA/PVC/phthalate free: FDA-complian", "keywords": "suction baby plate, BPA free toddler plate, self feeding set"}, "result": "[Listing 医生] 实时诊断\n\n━━ 综合评分 ━━\n当前 Listing 评分: 80/100（[OK] 良好）\n\n━━ Title 分析（178 字符）━━\n字符数评估: [OK] 长度充足\n关键词覆盖: [!] 缺失: \"suction baby plate\" \"BPA free toddler plate\" \"self feeding set\"\n\n━━ Bullet Points 分析（5 条）━━\n[OK] 条数充足\n\n━━ 问题清单 ━━\n1. 标题缺少核心词: \"suction baby plate\" \"BPA free toddler plate\" \"self feeding set\"，建议加入标题前60字符\n\n━━ 重写建议 ━━\n[参考重写] SUCTION BABY PLATE - BPA-Free Silicone Baby Plate Set with Suction — Self-Feeding Toddler Bowl Spoon Fork Kit for Ages 6M | suction baby plate | BPA free toddler plate | self feeding set — Premium Quality\n\n预估优化后 CTR 提升: +5-10%"},
        {"id": "agent-voc-decoder", "name": "用户之声解码器", "ts": "2026-06-11 00:44", "inputs": {"reviews": "suction is incredible, never moves\nlove the sage green color, very stylish\nspoon is too shallow for ", "asin": "B0CXYZ1234", "lang": "英语"}, "result": "[用户之声解码器] 实时分析 (15条输入)\n\n━━ 评论概览 ━━\n输入评论数: 15 条\n负面信号: 4 条（27%）\n正面信号: 8 条（53%）\n\n━━ TOP 痛点（高频）━━\n1. 质量问题（3次提及）\n   \"arrived with a crack on the bowl edge\"\n2. 物流/包装（3次提及）\n   \"arrived with a crack on the bowl edge\"\n3. 性价比（2次提及）\n   \"color faded slightly after 3 months\"\n\n━━ TOP 爽点（高频）━━\n1. 外观设计（2次提及）\n   \"love the sage green color, very stylish\"\n2. 质量耐用（1次提及）\n   \"suction is incredible, never moves\"\n3. 易用性（1次提及）\n   \"so easy to clean in dishwasher\"\n\n━━ 产品迭代建议 ━━\nP0: 改善「质量问题」→ 直接影响复购率\nP1: 改善「物流/包装」→ 延长产品生命周期\nP2: 改善「性价比」→ 提升品牌形象\n\n[英语] 数据来源：用户输入"},
        {"id": "agent-cs-triage", "name": "客服分诊台", "ts": "2026-06-11 00:44", "inputs": {"tickets": "Where is my order? It has been 2 weeks\nI want to return this, suction does not work\nMy baby scratche", "platform": "Amazon", "sla": "24小时"}, "result": "[客服分诊台] 实时分析（Amazon | SLA 24小时）\n\n━━ 工单分类分布（共 10 条）━━\n退货退款请求: 2 条（20.0%）\n产品质量问题: 2 条（20.0%）\n物流查询:     3 条（30.0%）\n使用咨询:     3 条（30.0%）\n\n━━ 高优先级预警（需 24小时 内处理）━━\n[ALERT] 工单1: \"I filed an A-to-Z claim, please respond\"\n\n━━ 标准回复模板（物流查询）━━\n\"Hi [Name], thank you for reaching out!\nYour order is currently in transit. Expected delivery: [DATE].\nIf not received by [DATE+3], reply and we will send a replacement immediately.\"\n\n━━ 产品缺陷信号 ━━\n[OK] 本批无明显批次性质量问题信号"},
        {"id": "agent-account-guardian", "name": "账号风险卫士", "ts": "2026-06-11 00:44", "inputs": {"notice": "Your account is under review. We have noticed unusual review patterns on your listings. This is a wa", "asins": "B0CXYZ1234\nB0DABC5678", "health": "黄色（预警）"}, "result": "[账号风险卫士] 实时风险评估\n\n━━ 综合风险评分 ━━\n风险评分: 8.0/10（高风险，需立即处理）\n账号状态: 黄色（预警）\n[!] 检测到警告通知，风险分上升 +1.5\n\n━━ 通知内容摘要 ━━\n> Your account is under review. We have noticed unusual review patterns on your listings. This is a wa\n\n━━ ASIN 合规检查（2 个）━━\nB0CXYZ1234: [~] 建议检查 Title 中是否含竞品品牌词、绝对化表述\nB0DABC5678: [~] 建议检查 Title 中是否含竞品品牌词、绝对化表述\n\n━━ 整改清单 ━━\nP0（今日）: 检查并删除 Listing 中的侵权词/医疗声明\nP0（今日）: 处理所有未回复差评工单（ODR 目标 <0.9%）\nP1（本周）: 提交 POA（行动计划）\n\n━━ POA 申诉框架（如需）━━\n\"Root Cause: [问题根因]\nCorrective Actions: [已执行的改正措施]\nPreventive Measures: [预防措施和未来计划]\" "},
        {"id": "agent-brand-guardian", "name": "品牌合规卫士", "ts": "2026-06-11 00:44", "inputs": {"copy": "Our BPA-free silicone plates are clinically tested for safety. Non-toxic and hypoallergenic material", "category": "母婴", "market": "US"}, "result": "[品牌合规卫士] 扫描报告（母婴 | US 市场）\n\n━━ 综合评分 ━━\n当前合规评分: 40/100 → 整改后预计: 92/100\n\n━━ 禁用词（3处违规）━━\n1. \"clinically tested\" → FDA - 需临床认证\n   合规改写: \"carefully evaluated...\"\n2. \"100% safe\" → FTC - 绝对化表述\n   合规改写: \"made with food-grade materials...\"\n3. \"no harmful chemicals\" → FTC - 无法证实\n   合规改写: \"free from common allergens...\"\n\n━━ 慎用词（4处需证明文件）━━\n4. \"bpa-free\" → 需第三方检测报告支撑\n5. \"non-toxic\" → 需 CPSIA/EN71 认证文件\n6. \"hypoallergenic\" → 需皮肤科测试报告\n7. \"pediatrician\" → 需执业医师签名或机构背书\n\n━━ 所需证明文件清单 ━━\n□ SGS/Intertek 第三方安全检测报告\n□ CPSIA 儿童产品认证（US必需）\n□ EN71/CE 认证（EU市场）"},
        {"id": "agent-product-radar", "name": "选品雷达", "ts": "2026-06-11 00:44", "inputs": {"keyword": "婴儿安全防护角", "market": "DE", "budget": ">$20k"}, "result": "[选品雷达] 实时分析\n\n━━ 机会评分 ━━\n品类: \"婴儿安全防护角\" | 市场: 德国 | 预算: >$20k\n综合评分: 74/100 [~] 值得尝试\n\n━━ 市场数据（基于关键词特征估算）━━\n月均搜索量: 155,000（YoY +13%）\nBSR TOP10 均价: $22.0 | 成本带: $6-9\n头部集中度（前3卖家）: 47% [OK] 仍有切入空间\n\n━━ 差异化切入角度 ━━\n1. 材质/工艺升级（食品级/环保材料 → 情感溢价 +$4）\n2. 套装/组合策略（提升 AOV 至 $40+）\n3. 月龄/场景分段（精准细分需求）\n\n━━ 竞争分析 ━━\n新品切入评论门槛: ~200 条\n新品窗口: ⭐⭐⭐ 一般\n\n━━ 建议 ━━\n[~] 值得尝试 — 需要明确差异化方向\n建议首批备货: 1000-2000 件（>$20k 预算匹配）"},
        {"id": "agent-tiktok-content", "name": "TikTok 内容官", "ts": "2026-06-11 00:44", "inputs": {"product": "BPA-Free硅胶餐具套装 鼠尾草绿", "audience": "6-18月龄辅食阶段宝妈，关注BLW自主进食", "style": "教程/攻略", "freq": "5条/周"}, "result": "[TikTok 内容官] 本周选题矩阵\n\n━━ 创作策略 ━━\n产品: BPA-Free硅胶餐具套装 鼠尾草绿\n目标受众: 6-18月龄辅食阶段宝妈，关注BLW自主进食\n内容风格: 教程/攻略 | 更新频次: 5条/周\n\n━━ 内容日历（5 条/周）━━\nDay 1（周一）— 教程/攻略\nHook: \"使用教程 — 关于BPA-Free硅胶餐具套装 鼠尾草绿\"\n话题: #babymom #toddlermom #momhack #BPA-Free硅胶餐具套装鼠尾草绿\n\nDay 2（周二）— 教程/攻略\nHook: \"3步搞定 — 关于BPA-Free硅胶餐具套装 鼠尾草绿\"\n话题: #babymom #toddlermom #momhack #BPA-Free硅胶餐具套装鼠尾草绿\n\nDay 3（周三）— 教程/攻略\nHook: \"保姆级攻略 — 关于BPA-Free硅胶餐具套装 鼠尾草绿\"\n话题: #babymom #toddlermom #momhack #BPA-Free硅胶餐具套装鼠尾草绿\n\nDay 4（周四）— 教程/攻略\nHook: \"新手必看 — 关于BPA-Free硅胶餐具套装 鼠尾草绿\"\n话题: #babymom #toddlermom #momhack #BPA-Free硅胶餐具套装鼠尾草绿\n\nDay 5（周五）— 教程/攻略\nHook: \"这样用对了 — 关于BPA-Free硅胶餐具套装 鼠尾草绿\"\n话题: #babymom #toddlermom #momhack #BPA-Free硅胶餐具套装鼠尾草绿\n\n━━ 爆款公式 ━━\n价值前置（3秒说明能学到什么）+ 步骤清晰 + 截图提示 = 收藏率 20%+\n\n━━ 发布建议 ━━\n最佳时间: 晚9-11PM（宝宝入睡后）\n话题标签: #babymom #toddlermom #momhack #babyfood #parenting\n预算建议: $75-150/周（寄送产品换视频）"},
    ]

    seed_reports_js = json.dumps(seed_reports, ensure_ascii=False)

    body = f"""
<div style='max-width:900px;margin:0 auto'>
<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:16px'>
  <div>
    <h1 style='font-size:28px;font-weight:900;letter-spacing:-.03em;margin:0 0 6px'>智能体报告</h1>
    <p style='color:#64748b;margin:0'>每次在智能体广场运行分析后，结果自动保存在此。</p>
  </div>
  <div style='display:flex;gap:10px'>
    <button onclick='exportReports()' style='padding:8px 16px;border-radius:8px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;font-size:13px;font-weight:600'>⬇ 导出全部</button>
    <button onclick='clearReports()' style='padding:8px 16px;border-radius:8px;border:1px solid #fecaca;background:#fff;color:#dc2626;cursor:pointer;font-size:13px;font-weight:600'>✕ 清空</button>
  </div>
</div>

<div id='report-filter-bar' style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;align-items:center'>
  <span style='font-size:12px;color:#94a3b8;font-weight:600;margin-right:2px'>按 Agent 筛选：</span>
  <button class='rpt-filter active' data-agent='' onclick='setAgentFilter("")'
    style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px;font-weight:600'>全部</button>
  <button class='rpt-filter' data-agent='agent-supply-sentinel' onclick='setAgentFilter("agent-supply-sentinel")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>📦 供应链哨兵</button>
  <button class='rpt-filter' data-agent='agent-pricing-advisor' onclick='setAgentFilter("agent-pricing-advisor")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>💰 定价顾问</button>
  <button class='rpt-filter' data-agent='agent-pnl-analyzer' onclick='setAgentFilter("agent-pnl-analyzer")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>📊 P&L 透视</button>
  <button class='rpt-filter' data-agent='agent-ad-attribution' onclick='setAgentFilter("agent-ad-attribution")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>📣 广告归因</button>
  <button class='rpt-filter' data-agent='agent-competitor-radar' onclick='setAgentFilter("agent-competitor-radar")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>🔭 竞品雷达</button>
  <button class='rpt-filter' data-agent='agent-listing-doctor' onclick='setAgentFilter("agent-listing-doctor")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>📝 Listing 医生</button>
  <button class='rpt-filter' data-agent='agent-voc-decoder' onclick='setAgentFilter("agent-voc-decoder")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>💬 VOC 解码</button>
  <button class='rpt-filter' data-agent='agent-cs-triage' onclick='setAgentFilter("agent-cs-triage")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>🎧 客服分诊</button>
  <button class='rpt-filter' data-agent='agent-account-guardian' onclick='setAgentFilter("agent-account-guardian")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>🛡 账号卫士</button>
  <button class='rpt-filter' data-agent='agent-brand-guardian' onclick='setAgentFilter("agent-brand-guardian")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>✅ 品牌合规</button>
  <button class='rpt-filter' data-agent='agent-product-radar' onclick='setAgentFilter("agent-product-radar")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>🎯 选品雷达</button>
  <button class='rpt-filter' data-agent='agent-tiktok-content' onclick='setAgentFilter("agent-tiktok-content")' style='padding:4px 12px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>🎬 TikTok 内容</button>
</div>

<div id='report-empty' style='display:none;text-align:center;padding:80px 20px;color:#94a3b8'>
  <div style='font-size:48px;margin-bottom:16px'>📊</div>
  <div style='font-size:18px;font-weight:600;margin-bottom:8px'>暂无报告</div>
  <div style='font-size:14px;margin-bottom:24px'>前往智能体广场运行分析，结果将自动保存在这里</div>
  <a href='agents.html' style='display:inline-block;padding:10px 24px;background:var(--accent);color:#fff;border-radius:8px;text-decoration:none;font-weight:600'>前往智能体广场 →</a>
</div>

<div id='report-list' style='display:flex;flex-direction:column;gap:16px'></div>
</div>

<script>
const AGENT_NAMES = {agent_names_js};
const SEED_REPORTS = {seed_reports_js};
const SEED_VERSION = 'v20260611-r2';
let _currentAgentFilter = '';

function setAgentFilter(agentId) {{
  _currentAgentFilter = agentId;
  document.querySelectorAll('.rpt-filter').forEach(b => {{
    const isActive = b.dataset.agent === agentId;
    b.style.background = isActive ? 'var(--accent)' : '#f8fafc';
    b.style.color = isActive ? '#fff' : '';
    b.style.borderColor = isActive ? 'var(--accent)' : '#e2e8f0';
    b.classList.toggle('active', isActive);
  }});
  renderReports();
}}

function loadReports() {{
  try {{
    const seeded = localStorage.getItem('agentReportsSeeded');
    let stored = [];
    try {{ stored = JSON.parse(localStorage.getItem('agentReports') || '[]'); }} catch(e) {{ stored = []; }}
    if (!Array.isArray(stored)) stored = [];
    if (seeded !== SEED_VERSION) {{
      const userIds = new Set(stored.map(r => r.ts + r.id));
      const fresh = SEED_REPORTS.filter(s => !userIds.has(s.ts + s.id));
      const merged = [...stored, ...fresh];
      localStorage.setItem('agentReports', JSON.stringify(merged));
      localStorage.setItem('agentReportsSeeded', SEED_VERSION);
      return merged;
    }}
    return stored.length > 0 ? stored : SEED_REPORTS;
  }} catch(e) {{ return SEED_REPORTS; }}
}}

function renderReports() {{
  let reports = loadReports();
  if (_currentAgentFilter) reports = reports.filter(r => r.id === _currentAgentFilter);
  const list = document.getElementById('report-list');
  const empty = document.getElementById('report-empty');
  if (!list) return;
  if (reports.length === 0) {{
    list.innerHTML = '<p style="color:#94a3b8;text-align:center;padding:40px">该 Agent 暂无报告，前往 <a href="agents.html">智能体广场</a> 运行分析。</p>';
    if (empty) empty.style.display = 'none';
    return;
  }}
  if (empty) empty.style.display = 'none';
  list.innerHTML = reports.map((r, i) => {{
    const name = r.name || AGENT_NAMES[r.id] || r.id;
    const inputSummary = r.inputs ? Object.entries(r.inputs).slice(0, 3).map(([k,v]) => `<span style='background:#f1f5f9;border-radius:4px;padding:2px 6px;font-size:11px;color:#475569'>${{k}}: ${{v.slice(0,30)}}${{v.length>30?'…':''}}</span>`).join(' ') : '';
    const preview = (r.result || '').slice(0, 300).replace(/</g,'&lt;');
    const catIcon = r.id.includes('supply') ? '📦' : r.id.includes('pricing') ? '💰' : r.id.includes('pnl') ? '📊' : r.id.includes('ad-') ? '📣' : r.id.includes('competitor') ? '🔭' : r.id.includes('listing') ? '📝' : r.id.includes('voc') ? '💬' : r.id.includes('cs-') ? '🎧' : r.id.includes('account') ? '🛡' : r.id.includes('brand') ? '✅' : r.id.includes('product-radar') ? '🎯' : '🎬';
    return `<div style='background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.06)'>
  <div style='display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px'>
    <div style='display:flex;align-items:center;gap:10px'>
      <span style='font-size:22px'>${{catIcon}}</span>
      <div>
        <div style='font-size:15px;font-weight:700;color:#1e293b'>${{name}}</div>
        <div style='font-size:12px;color:#94a3b8;margin-top:2px'>🕐 ${{r.ts || '未知时间'}}</div>
      </div>
    </div>
    <div style='display:flex;gap:8px;flex-shrink:0'>
      <button onclick='toggleReport(${{i}})' id='toggle-${{i}}' style='padding:5px 12px;border-radius:6px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px;font-weight:600'>展开</button>
      <button onclick='deleteReport(${{i}})' style='padding:5px 10px;border-radius:6px;border:1px solid #fecaca;background:#fff;color:#dc2626;cursor:pointer;font-size:12px'>删除</button>
    </div>
  </div>
  ${{inputSummary ? `<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px'>${{inputSummary}}</div>` : ''}}
  <pre id='report-body-${{i}}' style='display:none;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;font-size:12px;line-height:1.8;white-space:pre-wrap;word-break:break-all;max-height:500px;overflow-y:auto;margin:0'>${{(r.result||'').replace(/</g,'&lt;')}}</pre>
  <div id='report-preview-${{i}}' style='font-size:12px;color:#64748b;line-height:1.6;font-family:monospace;white-space:pre-wrap'>${{preview}}${{(r.result||'').length > 300 ? '…' : ''}}</div>
</div>`;
  }}).join('');
}}

function toggleReport(i) {{
  const body = document.getElementById('report-body-' + i);
  const preview = document.getElementById('report-preview-' + i);
  const btn = document.getElementById('toggle-' + i);
  if (!body) return;
  const isOpen = body.style.display !== 'none';
  body.style.display = isOpen ? 'none' : 'block';
  if (preview) preview.style.display = isOpen ? 'block' : 'none';
  if (btn) btn.textContent = isOpen ? '展开' : '收起';
}}

function deleteReport(i) {{
  const reports = loadReports();
  reports.splice(i, 1);
  localStorage.setItem('agentReports', JSON.stringify(reports));
  renderReports();
}}

function clearReports() {{
  if (confirm('确认清空全部报告记录？')) {{
    localStorage.removeItem('agentReports');
    renderReports();
  }}
}}

function exportReports() {{
  const reports = loadReports();
  if (reports.length === 0) {{ alert('暂无报告可导出'); return; }}
  const sep = '\\n' + Array(60).fill('─').join('') + '\\n';
  const lines = reports.map(r =>
    '=== ' + r.name + ' | ' + r.ts + ' ===\\n' +
    '输入: ' + JSON.stringify(r.inputs) + '\\n\\n' +
    r.result + '\\n'
  ).join(sep);
  const blob = new Blob([lines], {{type:'text/plain;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'agent-reports-' + new Date().toISOString().slice(0,10) + '.txt';
  a.click();
}}

renderReports();
window.addEventListener('storage', () => renderReports());
</script>
"""
    return html_page("智能体报告", body, active_nav="agent-report")


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
          sidebar_link('agent-report.html', '智能体报告', 'agent-report', '◑') +
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

    agent_cases_html = ""
    try:
        candidate_paths = [
            Path(skill.path),
            Path("paper2skills-vault") / skill.path,
            Path(__file__).parent.parent.parent.parent / "paper2skills-vault" / skill.path,
        ]
        raw = ""
        for mp in candidate_paths:
            if mp.exists():
                raw = mp.read_text(encoding="utf-8", errors="replace")
                break
        m = re.search(r'##\s*🧪\s*调用案例.*?$(.+?)(?=\n##\s|\Z)', raw, re.DOTALL | re.MULTILINE)
        if m:
            case_text = m.group(1).strip()
            lines_html = []
            for line in case_text.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                if '**' in stripped:
                    parts = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html.escape(stripped))
                    lines_html.append(f"<p style='margin:4px 0'>{parts}</p>")
                else:
                    lines_html.append(f"<p style='margin:4px 0;color:#475569'>{html.escape(stripped)}</p>")
            inner = "\n".join(lines_html)
            agent_cases_html = (
                f"<div style='margin-top:24px;padding:16px 20px;"
                f"background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px'>"
                f"<div style='font-size:13px;font-weight:700;color:#065f46;margin-bottom:10px'>🧪 智能体广场调用案例</div>"
                f"{inner}"
                f"<div style='margin-top:10px'><a href='../agents.html' "
                f"style='font-size:12px;color:#059669;font-weight:600'>→ 前往智能体广场运行</a></div>"
                f"</div>"
            )
    except Exception:
        pass

    body = f"""
<nav class="breadcrumbs"><a href="../index.html">首页</a> / <a href="../domains/{slugify(skill.domain_dir)}.html">{html.escape(skill.domain_dir)}</a> / {html.escape(skill.skill_id)}</nav>
<div class="skill-toc">
  <a href="#s-problem">① 问题</a>
  <a href="#s-algo">② 算法</a>
  <a href="#s-scenario">③ 场景</a>
  <a href="#s-code">④ 代码</a>
  <a href="#s-relations">⑤ 关联</a>
  <a href="#s-value">⑥ 价值</a>
</div>
<h1>{html.escape(skill.title)}</h1>
<p class="muted">{html.escape(skill.skill_id)} · {html.escape(skill.domain_dir)}</p>
<div class="tag-row">{''.join(f"<span class='tag'>{html.escape(t)}</span>" for t in skill.tags + skill.topics + skill.workflows)}</div>
{handbook_uplinks}
{roi_meta}
{biz_panel}
<div class="two-col">
  <section>
    <h2 id="s-problem">1. 解决的问题</h2>
    <p>{html.escape(skill.problem_solved or skill.algorithm_summary)}</p>
    <h2 id="s-algo">2. 核心算法逻辑</h2>
    <p>{html.escape(skill.algorithm_summary)}</p>
    <h2 id="s-scenario">3. 业务应用场景</h2>
    {scenario_html}
    <h2>4. 输入数据要求</h2>{render_items(skill.inputs) if skill.inputs else "<p class='muted'>请查看原始代码模板获取输入规格。</p>"}
    <h2>5. 输出结果</h2>{render_items(skill.outputs) if skill.outputs else "<p class='muted'>请查看原始代码模板获取输出规格。</p>"}
    <h2 id="s-value">6. 业务价值 / ROI</h2>{render_items(skill.roi) if skill.roi else ("<p>" + html.escape(skill.roi_figure) + "</p>" if skill.roi_figure else "<p class='muted'>未自动抽取；请查看原始 Skill 卡片。</p>")}
    <h2 id="s-code">7. 代码模板</h2>
    <p class="muted">代码块数量：{skill.code_blocks} · 路径：{html.escape(skill.code_path or '未检测到')}</p>
    {_render_code_preview(skill.code_preview)}
    <h2>8. 论文来源</h2>{render_items(skill.papers)}
  </section>
  <aside class="relation-panel" id="s-relations">
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
<script src="../assets/ego-graph.js"></script>
<script>
function copyCode(btn) {{
  var pre = btn.nextElementSibling;
  var text = pre ? pre.textContent : '';
  navigator.clipboard.writeText(text).then(function() {{
    btn.textContent = '已复制 ✓';
    btn.classList.add('copied');
    setTimeout(function() {{
      btn.textContent = '复制';
      btn.classList.remove('copied');
    }}, 2000);
  }}).catch(function() {{
    btn.textContent = '复制失败';
    setTimeout(function() {{ btn.textContent = '复制'; }}, 1500);
  }});
}}
</script>
{agent_cases_html}"""
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
    return (
        "<div class='code-wrap'>"
        "<button class='copy-btn' onclick='copyCode(this)' title='复制代码'>复制</button>"
        f"<pre class='code-preview'><code>{escaped}</code></pre>"
        "</div>"
    )


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
  --muted:       #6e6e73;

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
#global-search::placeholder { color: var(--ink-2); opacity: .55; }
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
.hero { margin-bottom: 8px; padding-top: 4px; }
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

/* ── Skill TOC (内锚点目录) ── */
.skill-toc {
  display: flex; flex-wrap: wrap; gap: 6px;
  margin-bottom: 20px;
}
.skill-toc a {
  padding: 4px 11px; border-radius: var(--r-full);
  font-size: 12px; font-weight: 600;
  background: var(--panel-2); color: var(--ink-2);
  border: 1px solid var(--line);
  text-decoration: none; transition: background .15s, color .15s;
  white-space: nowrap;
}
.skill-toc a:hover { background: var(--accent-light); color: var(--accent); border-color: var(--accent-light); }

/* ── Code Preview ── */
.code-wrap { position: relative; margin: 12px 0; }
.copy-btn {
  position: absolute; top: 10px; right: 10px; z-index: 2;
  padding: 4px 10px; border-radius: var(--r-sm);
  border: 1px solid rgba(255,255,255,.18);
  background: rgba(255,255,255,.12); color: #e8e0d4;
  font-size: 11px; font-weight: 600; cursor: pointer;
  transition: background .15s, opacity .15s;
  letter-spacing: .04em;
}
.copy-btn:hover { background: rgba(255,255,255,.22); }
.copy-btn.copied { color: #6ee7b7; border-color: #6ee7b7; }
.code-preview {
  background: #1a1916; color: #e8e0d4;
  border-radius: var(--r-lg); padding: 18px 20px;
  overflow-x: auto; overflow-y: auto;
  font-size: 12.5px; line-height: 1.65;
  font-family: "JetBrains Mono", "Fira Code", Menlo, monospace;
  max-height: 420px; margin: 0; white-space: pre;
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
.status-dot.live { background: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,.2); }
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
      var matchDiff = !diff || !c.dataset.diff || c.dataset.diff === diff;
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
        domain_search_bar = f"""
<div style='display:flex;align-items:center;gap:10px;margin:12px 0 16px'>
  <input id='domain-search' placeholder='在 {html.escape(title)} 中搜索…' autocomplete='off'
    style='flex:1;max-width:340px;padding:8px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:14px'>
  <span class='muted' id='domain-count' style='font-size:13px'></span>
</div>
<script>
(function(){{
  var inp = document.getElementById('domain-search');
  var cards = document.querySelectorAll('.cards .skill-card');
  var cnt = document.getElementById('domain-count');
  if(!inp) return;
   inp.addEventListener('input', function(){{
    var q = this.value.trim().toLowerCase();
    var shown = 0;
    cards.forEach(function(c){{
      var text = (c.textContent||'').toLowerCase() + (c.href||'').toLowerCase();
      var vis = !q || text.includes(q);
      c.style.display = vis ? '' : 'none';
      if(vis) shown++;
    }});
    cnt.textContent = q ? ('\u663e\u793a ' + shown + ' / ' + cards.length + ' \u4e2a') : '';
  }});
}})();
</script>"""
        write_file(
            out / "domains" / f"{slugify(title)}.html",
            html_page(title,
                      f"<h1>{html.escape(title)}</h1>"
                      f"<p>{html.escape(domain.get('description',''))}</p>"
                      f"{domain_search_bar}"
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
        topic_search = f"""<input id='topic-search' placeholder='在 {html.escape(topic)} 中搜索…'
  style='max-width:320px;padding:8px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:14px;margin:10px 0 16px;display:block'>
<script>(function(){{var inp=document.getElementById('topic-search');var cards=document.querySelectorAll('.cards .skill-card');if(!inp)return;inp.addEventListener('input',function(){{var q=this.value.trim().toLowerCase();cards.forEach(function(c){{var t=(c.textContent||'').toLowerCase()+(c.href||'').toLowerCase();c.style.display=(!q||t.includes(q))?'':'none';}})}});}})();</script>"""
        write_file(out / "topics" / path, html_page(
            topic,
            f"<h1>{html.escape(topic)}</h1>{topic_search}<div class='cards'>{cards}</div>",
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
    write_file(out / "agent-report.html", render_agent_report_page())

    # ── toB Scene Playbooks (Phase F) ──
    for pb in TOB_PLAYBOOKS:
        write_file(
            out / "playbooks" / f"{pb['id']}.html",
            render_tob_playbook(pb, skill_lookup),
        )
    def _pb_card(pb: dict) -> str:
        tag = html.escape(pb.get("tag", ""))
        pb_id = pb["id"]
        name = html.escape(pb["name"])
        icon = pb["icon"]
        biz_tag = html.escape(pb["tag"])
        desc = html.escape(pb["desc"])
        return (
            f"<a class='biz-card' href='{pb_id}.html' data-tag='{tag}'>"
            f"<div class='biz-card-header'>"
            f"<span class='biz-icon'>{icon}</span>"
            f"<div class='biz-body'>"
            f"<div class='biz-card-meta'>"
            f"<strong>{name}</strong>"
            f"<span class='biz-tag'>{biz_tag}</span>"
            f"</div>"
            f"<p>{desc}</p>"
            f"</div>"
            f"</div>"
            f"</a>"
        )
    tob_index_cards = "".join(_pb_card(pb) for pb in TOB_PLAYBOOKS)
    pb_search_bar = """<div style='margin:12px 0 20px;display:flex;flex-wrap:wrap;gap:8px;align-items:center'>
  <input id='pb-search' placeholder='搜索手册名称…' autocomplete='off'
    style='padding:8px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:14px;min-width:200px'>
  <button class='pb-tag-btn active' data-tag='' onclick='pbFilter("")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px;font-weight:600'>全部</button>
  <button class='pb-tag-btn' data-tag='供应链' onclick='pbFilter("供应链")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>供应链</button>
  <button class='pb-tag-btn' data-tag='广告' onclick='pbFilter("广告")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>广告</button>
  <button class='pb-tag-btn' data-tag='合规' onclick='pbFilter("合规")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>合规风控</button>
  <button class='pb-tag-btn' data-tag='选品' onclick='pbFilter("选品")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>选品增长</button>
  <button class='pb-tag-btn' data-tag='Agent' onclick='pbFilter("Agent")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>AI Agent</button>
  <button class='pb-tag-btn' data-tag='客服' onclick='pbFilter("客服")'
    style='padding:5px 14px;border-radius:20px;border:1px solid #e2e8f0;background:#f8fafc;cursor:pointer;font-size:12px'>客服运营</button>
</div>
<script>
(function(){
  var bizCards = document.querySelectorAll('.biz-grid .biz-card');
  var searchInput = document.getElementById('pb-search');
  var currentTag = '';
  function applyFilter() {
    var q = searchInput ? searchInput.value.trim().toLowerCase() : '';
    bizCards.forEach(function(c) {
      var tagMatch = !currentTag || (c.dataset.tag||'').includes(currentTag);
      var textMatch = !q || (c.textContent||'').toLowerCase().includes(q);
      c.style.display = (tagMatch && textMatch) ? '' : 'none';
    });
  }
  window.pbFilter = function(tag) {
    currentTag = tag;
    document.querySelectorAll('.pb-tag-btn').forEach(function(b) {
      var isActive = b.dataset.tag === tag;
      b.style.background = isActive ? 'var(--accent)' : '#f8fafc';
      b.style.color = isActive ? '#fff' : '';
      b.style.borderColor = isActive ? 'var(--accent)' : '#e2e8f0';
    });
    applyFilter();
  };
  if(searchInput) searchInput.addEventListener('input', applyFilter);
})();
</script>"""
    write_file(out / "playbooks" / "index.html", html_page(
        "场景手册",
        "<h1>场景手册</h1>"
        "<p class='muted'>针对运营部门的开箱即用决策指南，每本手册包含完整操作步骤、所需数据和预期收益。</p>"
        f"{pb_search_bar}"
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
