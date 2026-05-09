"""标签字典 v3.8 生成器

将 Phase 4 代码标签同步到 v3.7 字典，生成 v3.8 版本。

同步范围：
1. TAG_GEN_N001~N008（通用负面标签，对应 E001~E008 的互斥负面）
2. TAG_ZEN_R001~R012（Zendesk 极简规则）
3. TAG_DEF_N001~N008（负面缺陷标签）
4. BRAND_*（品牌提及标签）

字段补全策略：
- 核心字段（ID/中英文/关键词/情感/AIPL）直接从代码标签提取
- 业务字段（主责部门/策略包/原子指标）按规则推导
- 评分字段填默认值，标记为"待审核"
"""

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Phase 4 标签定义 ───────────────────────────────────────────────

# Import from the code modules
import sys
_SCRIPT_DIR = Path(__file__).parent.resolve()
_DATA_PROC_DIR = _SCRIPT_DIR.parent / "04-数据处理"
for d in (_SCRIPT_DIR, _DATA_PROC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from general_tag_labeler import GENERAL_TAGS, POS_TO_NEG
from zendesk_minimal_rules import ZENDESK_RULES
from brand_label_functions import BRAND_KEYWORD_LIBRARY

# 负面缺陷标签（内联，与 unified_labeler 保持一致）
NEGATIVE_DEFECT_TAGS = [
    {"tag_id": "TAG_DEF_N001", "tag_en": "functional_failure", "tag_cn": "功能失效", "aipl": "L1", "sentiment": "negative",
     "keywords": ["doesn't work", "doesnt work", "not working", "stopped working", "won't turn on", "wont turn on", "no power", "dead", "malfunction", "faulty", "defective", "broken", "broke", "not charging", "not charge", "no suction", "weak suction", "low suction", "not strong", "too weak", "no response", "kaputt", "defekt", "nicht funktioniert", "ne fonctionne pas", "défectueux", "cassé"]},
    {"tag_id": "TAG_DEF_N002", "tag_en": "leakage_issue", "tag_cn": "泄漏问题", "aipl": "L1", "sentiment": "negative",
     "keywords": ["leaking", "leak", "leaks", "leaked", "spill", "spills", "spilling", "drip", "dripping", "drips"]},
    {"tag_id": "TAG_DEF_N003", "tag_en": "surface_damage", "tag_cn": "表面损伤", "aipl": "L1", "sentiment": "negative",
     "keywords": ["crack", "cracked", "cracking", "scratch", "scratched", "dented", "stain", "stained", "discolor", "discolored", "fading", "faded", "peel", "peeling", "tear", "torn", "rip", "ripped", "fray", "fraying", "frayed"]},
    {"tag_id": "TAG_DEF_N004", "tag_en": "odor_overheating", "tag_cn": "异味过热", "aipl": "L1", "sentiment": "negative",
     "keywords": ["smell", "smells", "smelly", "odor", "stink", "stinky", "burning smell", "chemical smell", "plastic smell", "overheat", "overheating", "overheated", "too hot", "burning", "melt", "melted", "melting", "warp", "warped"]},
    {"tag_id": "TAG_DEF_N005", "tag_en": "structural_looseness", "tag_cn": "结构松动", "aipl": "L1", "sentiment": "negative",
     "keywords": ["loose", "loosen", "fall off", "falls off", "falling off", "detach", "detached", "come off", "comes off", "coming off", "wobble", "wobbly", "not secure", "not stable"]},
    {"tag_id": "TAG_DEF_N006", "tag_en": "noise_issue", "tag_cn": "噪音问题", "aipl": "L1", "sentiment": "negative",
     "keywords": ["noisy", "loud", "squeak", "squeaking", "rattling", "vibrating", "buzzing", "clicking", "grinding", "whining"]},
    {"tag_id": "TAG_DEF_N007", "tag_en": "missing_parts", "tag_cn": "缺少配件", "aipl": "L1", "sentiment": "negative",
     "keywords": ["missing", "not included", "incomplete", "no adapter", "no cable", "no charger", "no manual", "no instructions", "parts missing", "missing part"]},
    {"tag_id": "TAG_DEF_N008", "tag_en": "wear_aging", "tag_cn": "磨损老化", "aipl": "L1", "sentiment": "negative",
     "keywords": ["wear", "worn", "wearing out", "wore out", "deteriorate", "deteriorated", "aging", "aged", "old", "not durable"]},
]


# ── 字段推导规则 ───────────────────────────────────────────────────

def derive_tag_theme(tag_id: str, tag_en: str) -> str:
    """推导标签主题"""
    if tag_id.startswith("TAG_GEN_N"):
        return "产品体验负面"
    elif tag_id.startswith("TAG_ZEN_R"):
        return "客服售后"
    elif tag_id.startswith("TAG_DEF_N"):
        return "产品质量缺陷"
    elif tag_id.startswith("BRAND_"):
        return "品牌提及"
    return "通用"


def derive_sentiment_text(sentiment: str) -> str:
    """情感极性文本"""
    mapping = {
        "positive": "正向",
        "negative": "负向",
        "neutral": "中性",
    }
    return mapping.get(sentiment, "中性")


def derive_metric_direction(sentiment: str) -> str:
    """MetricDirection"""
    return "positive" if sentiment == "positive" else "negative"


def derive_proxy_nps(sentiment: str) -> str:
    """Proxy NPS贡献"""
    mapping = {
        "positive": "Promoter驱动",
        "negative": "Detractor驱动",
        "neutral": "Passive影响",
    }
    return mapping.get(sentiment, "Passive影响")


def derive_atomic_metric(tag_id: str, tag_en: str) -> str:
    """推导对应原子指标"""
    if tag_id.startswith("TAG_GEN_N"):
        num = tag_id.split("_")[-1]
        return f"COM_N_{num}"
    elif tag_id.startswith("TAG_ZEN_R"):
        num = tag_id.split("_")[-1]
        return f"ZEN_{num}"
    elif tag_id.startswith("TAG_DEF_N"):
        num = tag_id.split("_")[-1]
        return f"DEF_{num}"
    elif tag_id.startswith("BRAND_"):
        brand = tag_id.replace("BRAND_", "").replace("_", " ")
        return f"BRD_{brand[:10].upper()}"
    return ""


def derive_strategy_pack(tag_theme: str, sentiment: str) -> str:
    """推导策略包"""
    if tag_theme == "产品体验负面":
        return "产品体验优化包"
    elif tag_theme == "客服售后":
        return "售后服务提升包"
    elif tag_theme == "产品质量缺陷":
        return "质量缺陷闭环包"
    elif tag_theme == "品牌提及":
        return "品牌声量监测包"
    return "通用策略包"


def derive_owner_dept(tag_theme: str) -> tuple[str, str, str]:
    """推导主责部门、协同部门、业务动作"""
    mapping = {
        "产品体验负面": ("产品部", "用户体验部", "产品部：优化产品体验，降低负面反馈率"),
        "客服售后": ("客服部", "运营部", "客服部：建立售后工单快速响应机制"),
        "产品质量缺陷": ("品控部", "研发部; 供应链部", "品控部：建立缺陷闭环追踪，推动供应商改进"),
        "品牌提及": ("市场部", "品牌运营部", "市场部：监测品牌声量，分析竞品对比趋势"),
    }
    return mapping.get(tag_theme, ("产品部", "通用", "产品部：持续监控标签覆盖率"))


def derive_storyline(tag_theme: str, tag_cn: str) -> str:
    """推导故事线关联"""
    if tag_theme == "产品体验负面":
        return f"{tag_cn}体验断层"
    elif tag_theme == "客服售后":
        return "售后触点效率"
    elif tag_theme == "产品质量缺陷":
        return "质量信任危机"
    elif tag_theme == "品牌提及":
        return "品牌竞争格局"
    return "通用体验故事"


def derive_applicable_voc_carrier(tag_theme: str) -> str:
    """推导适用VOC载体"""
    mapping = {
        "产品体验负面": "评论; 工单; 社媒; 问卷",
        "客服售后": "工单; 邮件; 在线聊天",
        "产品质量缺陷": "评论; 工单; 退货申请",
        "品牌提及": "评论; 社媒; 论坛; 新闻",
    }
    return mapping.get(tag_theme, "评论; 社媒")


def derive_applicable_persona(tag_theme: str) -> str:
    """推导适用用户画像"""
    if tag_theme == "客服售后":
        return "售后求助型"
    return "通用"


def derive_definition(tag_cn: str, tag_en: str, keywords: list[str]) -> str:
    """推导标签定义"""
    kw_sample = "; ".join(keywords[:5])
    return f"消费者表达{tag_cn}相关体验（{tag_en}），典型表达包括：{kw_sample}等"


# ── 行生成器 ───────────────────────────────────────────────────────

def build_dict_row(tag: dict, row_template: dict) -> dict:
    """根据代码标签和模板生成字典行"""
    tag_id = tag["tag_id"]
    tag_en = tag.get("tag_en", "")
    tag_cn = tag.get("tag_cn", "")
    aipl = tag.get("aipl", tag.get("aipl_node", ""))
    sentiment = tag.get("sentiment", tag.get("sentiment_base", "neutral"))
    keywords = tag.get("keywords", [])

    theme = derive_tag_theme(tag_id, tag_en)
    kw_str = "; ".join(keywords)
    owner, co_owner, action = derive_owner_dept(theme)

    row = deepcopy(row_template)
    row.update({
        "标签ID": tag_id,
        "AIPL节点": aipl,
        "标签主题": theme,
        "VOC标签（中文）": tag_cn,
        "VOC标签（英文）": tag_en,
        "英文关键词/典型表达": kw_str,
        "消费者习惯关键词/原话短语": kw_str,
        "标签定义": derive_definition(tag_cn, tag_en, keywords),
        "情感极性": derive_sentiment_text(sentiment),
        "是否AI可抽取": "是",
        "来源类型": "Phase4新增",
        "适用产品品线": "通用",
        "适用VOC载体": derive_applicable_voc_carrier(theme),
        "适用用户画像": derive_applicable_persona(theme),
        "对应原子指标": derive_atomic_metric(tag_id, tag_en),
        "MetricDirection": derive_metric_direction(sentiment),
        "Proxy NPS贡献": derive_proxy_nps(sentiment),
        "是否通用标签": "是",
        "故事线关联": derive_storyline(theme, tag_cn),
        "策略包": derive_strategy_pack(theme, sentiment),
        "业务动作/责任部门": action,
        "主责部门": owner,
        "协同部门": co_owner,
        "默认优先级": "P2",
        "备注": f"Phase 4 自动生成（{datetime.now().strftime('%Y-%m-%d')}）",
        "合理性评分": 50.0,
        "风险等级": "中风险",
        "问题诊断": "新标签，待验证覆盖率和区分度",
        "品类特异性指数": 0.0,
        "共性/特性分类": "强共性标签",
        "主导品类": "通用",
        "优化建议": "运行Phase 4流水线验证实际覆盖率后调整关键词",
        "优化优先级": "P2",
        "审核状态": "待审核",
        "v3.6_AIPL动态规则": f"关键词匹配:{tag_en}",
        "v3.6_安全等级": "中-常规监控",
    })
    return row


def build_brand_dict_row(brand_name: str, config, row_template: dict) -> dict:
    """为品牌生成字典行"""
    tag_id = f"BRAND_{brand_name.replace(' ', '_')}"
    tag_en = brand_name.lower().replace(" ", "_")
    tag_cn = brand_name

    sentiment_map = {"own_brand": "neutral", "direct_competitor": "negative", "indirect_competitor": "negative"}
    sentiment = sentiment_map.get(config.brand_type, "neutral")

    kw_str = "; ".join(config.keywords)
    theme = "品牌提及"
    owner, co_owner, action = derive_owner_dept(theme)

    row = deepcopy(row_template)
    row.update({
        "标签ID": tag_id,
        "AIPL节点": "B1",
        "标签主题": theme,
        "VOC标签（中文）": tag_cn,
        "VOC标签（英文）": tag_en,
        "英文关键词/典型表达": kw_str,
        "消费者习惯关键词/原话短语": kw_str,
        "标签定义": f"消费者在VOC中提及{brand_name}品牌（{config.brand_type}），品类范围：{', '.join(config.categories)}",
        "情感极性": derive_sentiment_text(sentiment),
        "是否AI可抽取": "是",
        "来源类型": "Phase4新增",
        "适用产品品线": "; ".join(config.categories),
        "适用VOC载体": "评论; 社媒; 论坛; 新闻",
        "适用用户画像": "通用",
        "对应原子指标": f"BRD_{brand_name[:10].upper().replace(' ', '_')}",
        "MetricDirection": derive_metric_direction(sentiment),
        "Proxy NPS贡献": derive_proxy_nps(sentiment),
        "是否通用标签": "是",
        "故事线关联": "品牌竞争格局",
        "策略包": "品牌声量监测包",
        "业务动作/责任部门": action,
        "主责部门": owner,
        "协同部门": co_owner,
        "默认优先级": "P1" if config.brand_type == "own_brand" else "P2",
        "备注": f"Phase 4 自动生成; 品牌类型:{config.brand_type}; 优先级:{config.priority}",
        "合理性评分": 60.0,
        "风险等级": "低风险",
        "问题诊断": "品牌识别标签，区分度取决于关键词覆盖",
        "品类特异性指数": 0.0,
        "共性/特性分类": "强共性标签",
        "主导品类": "; ".join(config.categories) if config.categories else "通用",
        "优化建议": "根据实际品牌提及覆盖率扩展拼写变体",
        "优化优先级": "P2",
        "审核状态": "待审核",
        "v3.6_AIPL动态规则": f"品牌关键词匹配:{brand_name}",
        "v3.6_安全等级": "高-关键监控" if config.brand_type == "own_brand" else "中-常规监控",
    })
    return row


# ── 主函数 ─────────────────────────────────────────────────────────

def generate_v38():
    """生成 v3.8 字典"""
    print("=" * 70)
    print("标签字典 v3.8 生成器")
    print("=" * 70)

    v37_path = Path(__file__).parent.parent.parent / "04-输出结果/01-字典版本/tag_dictionary_v3.7.xlsx"
    v38_path = Path(__file__).parent.parent.parent / "04-输出结果/01-字典版本/tag_dictionary_v3.8.xlsx"

    if not v37_path.exists():
        print(f"⚠️ v3.7 字典不存在: {v37_path}")
        return

    print(f"\n读取: {v37_path}")
    xl = pd.ExcelFile(v37_path)
    print(f"Sheets: {xl.sheet_names}")

    # 读取主表作为模板
    main_df = pd.read_excel(v37_path, sheet_name="01_通用标签主表")
    print(f"\n原通用标签主表: {len(main_df)} 行, {len(main_df.columns)} 列")

    # 获取一行作为模板（用最后一行或空值填充）
    row_template = {col: None for col in main_df.columns}
    # 用第一行填充非空值作为默认值
    for col in main_df.columns:
        if col in main_df.columns and len(main_df) > 0 and pd.notna(main_df[col].iloc[0]):
            row_template[col] = main_df[col].iloc[0]

    # ── 生成新行 ──────────────────────────────────────────────
    new_rows: list[dict] = []

    # 1. 通用负面标签 N001~N008
    print("\n--- 生成通用负面标签 ---")
    neg_tags = [t for t in GENERAL_TAGS if t["tag_id"].startswith("TAG_GEN_N")]
    for tag in sorted(neg_tags, key=lambda t: t["tag_id"]):
        new_rows.append(build_dict_row(tag, row_template))
        print(f"  {tag['tag_id']}: {tag['tag_cn']}")

    # 2. Zendesk 规则 R001~R012
    print("\n--- 生成 Zendesk 极简规则标签 ---")
    for rule in sorted(ZENDESK_RULES, key=lambda r: r["tag_id"]):
        tag = {
            "tag_id": rule["tag_id"],
            "tag_en": rule["tag_en"],
            "tag_cn": rule["tag_cn"],
            "aipl": rule["aipl"],
            "sentiment": rule["sentiment"],
            "keywords": rule["keywords"],
        }
        new_rows.append(build_dict_row(tag, row_template))
        print(f"  {rule['tag_id']}: {rule['tag_cn']}")

    # 3. 负面缺陷标签 DEF_N001~N008
    print("\n--- 生成负面缺陷标签 ---")
    for tag in sorted(NEGATIVE_DEFECT_TAGS, key=lambda t: t["tag_id"]):
        new_rows.append(build_dict_row(tag, row_template))
        print(f"  {tag['tag_id']}: {tag['tag_cn']}")

    # 4. 品牌标签
    print("\n--- 生成品牌标签 ---")
    for brand_name, config in sorted(BRAND_KEYWORD_LIBRARY.items()):
        new_rows.append(build_brand_dict_row(brand_name, config, row_template))
        print(f"  BRAND_{brand_name.replace(' ', '_')}: {brand_name} ({config.brand_type})")

    # ── 合并数据 ──────────────────────────────────────────────
    print(f"\n--- 合并数据 ---")
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([main_df, new_df], ignore_index=True)
    print(f"  原行数: {len(main_df)}")
    print(f"  新增行: {len(new_df)}")
    print(f"  合并后: {len(combined_df)}")

    # ── 写入 Excel ────────────────────────────────────────────
    print(f"\n--- 写入 v3.8 ---")
    v38_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(v38_path, engine="openpyxl") as writer:
        # 写入所有原始 sheet
        for sheet_name in xl.sheet_names:
            if sheet_name == "01_通用标签主表":
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df = pd.read_excel(v37_path, sheet_name=sheet_name)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  已保存: {v38_path}")

    # ── 审计报告 ──────────────────────────────────────────────
    audit = {
        "version": "3.8",
        "generated": datetime.now().isoformat(),
        "base_version": "3.7",
        "base_rows": len(main_df),
        "new_rows": len(new_df),
        "total_rows": len(combined_df),
        "new_tags_by_category": {
            "通用负面": len(neg_tags),
            "Zendesk规则": len(ZENDESK_RULES),
            "负面缺陷": len(NEGATIVE_DEFECT_TAGS),
            "品牌标签": len(BRAND_KEYWORD_LIBRARY),
        },
        "new_tag_ids": [r["标签ID"] for r in new_rows],
        "output_path": str(v38_path),
    }

    audit_path = v38_path.parent / "tag_dictionary_v3.8_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"  审计: {audit_path}")

    # ── 自证 ──────────────────────────────────────────────────
    print("\n--- 自证验证 ---")

    # 验证1: 所有新标签ID不重复
    all_ids = combined_df["标签ID"].astype(str).tolist()
    dupes = [tid for tid, count in pd.Series(all_ids).value_counts().items() if count > 1]
    id_check = "PASS" if not dupes else f"FAIL (dupes: {dupes})"
    print(f"  [{id_check}] 标签ID唯一性")

    # 验证2: 核心字段无空值
    core_cols = ["标签ID", "VOC标签（中文）", "VOC标签（英文）", "情感极性", "AIPL节点"]
    null_counts = {col: combined_df[col].isna().sum() for col in core_cols}
    null_check = "PASS" if all(c == 0 for c in null_counts.values()) else f"FAIL (nulls: {null_counts})"
    print(f"  [{null_check}] 核心字段完整性")

    # 验证3: 新标签情感极性正确
    new_id_set = {r["标签ID"] for r in new_rows}
    new_in_combined = combined_df[combined_df["标签ID"].isin(new_id_set)]
    valid_sentiments = {"正向", "负向", "中性"}
    invalid_sent = new_in_combined[~new_in_combined["情感极性"].isin(valid_sentiments)]
    sent_check = "PASS" if len(invalid_sent) == 0 else f"FAIL ({len(invalid_sent)} invalid)"
    print(f"  [{sent_check}] 情感极性有效性")

    # 验证4: 关键词非空
    empty_kw = new_in_combined[new_in_combined["英文关键词/典型表达"].isna() | (new_in_combined["英文关键词/典型表达"] == "")]
    kw_check = "PASS" if len(empty_kw) == 0 else f"FAIL ({len(empty_kw)} empty)"
    print(f"  [{kw_check}] 关键词非空性")

    print("\n" + "=" * 70)
    return audit


if __name__ == "__main__":
    generate_v38()
