"""VOC Dashboard 导出到 Obsidian + Feishu CSV

从 dashboard_and_insights.json 生成:
1. Obsidian Markdown 报告
2. Feishu 多维表格 CSV
"""

import json
import csv
from pathlib import Path
from datetime import datetime

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/精选打标voc/labeling_output")
DASHBOARD_FILE = OUTPUT_BASE / "dashboard_and_insights.json"
EXPORT_BASE = OUTPUT_BASE / "exports"


def load_dashboard():
    with open(DASHBOARD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_obsidian_report(data: dict) -> str:
    """生成 Obsidian 格式 Markdown 报告"""
    gd = data["global_dashboard"]
    sources = data["source_dashboards"]
    plan = data["reverse_improvement_plan"]

    nps = gd["proxy_nps"]
    funnel = gd["aipl_funnel"]
    drivers = gd["driver_analysis"]
    personas = gd["persona_insights"]
    brand = gd["brand_analysis"]
    coverage = gd["tag_coverage"]

    lines = [
        "# VOC Proxy NPS × AIPL 全旅程看板报告",
        f"",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 数据源: Amazon / Trustpilot / Reddit / Zendesk",
        f"> 总样本: {coverage['total_voc']:,} 条",
        f"",
        "---",
        f"",
        "## 1. 全局指标概览",
        f"",
        "### Proxy NPS",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| **Proxy NPS** | {nps['proxy_nps']:.1f} |",
        f"| 推荐者 | {nps['promoters']:,} ({nps['promoter_pct']}%) |",
        f"| 被动者 | {nps['passives']:,} |",
        f"| 贬损者 | {nps['detractors']:,} ({nps['detractor_pct']}%) |",
        f"",
        "### AIPL 旅程漏斗",
        f"",
        "| 节点 | VOC 数 | Top 主题 |",
        "|------|--------|----------|",
    ]

    for node in ["A", "I", "P1", "P2", "L1", "L2", "L3"]:
        info = funnel[node]
        themes = ", ".join(f"{t['theme']}({t['count']})" for t in info["top_themes"])
        lines.append(f"| **{node}** | {info['count']:,} | {themes} |")

    lines.extend([
        f"",
        "### 标签覆盖率",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总 VOC | {coverage['total_voc']:,} |",
        f"| 命中标签 | {coverage['matched_voc']:,} |",
        f"| 覆盖率 | {coverage['coverage_rate']*100:.1f}% |",
        f"| 命中标签种类 | {coverage['unique_tags_matched']} |",
        f"",
        "---",
        f"",
        "## 2. 各数据源对比",
        f"",
        "| 数据源 | NPS | 推荐者% | 贬损者% | 覆盖率 | 冲突率 |",
        "|--------|-----|---------|---------|--------|--------|",
    ])

    for src, sd in sources.items():
        n = sd["proxy_nps"]
        c = sd["tag_coverage"]
        # 计算冲突率
        conflict_rate = sd.get("conflict_rate", 0)  # 可能需要重新计算
        lines.append(
            f"| {src.title()} | {n['proxy_nps']:.1f} | {n['promoter_pct']}% | "
            f"{n['detractor_pct']}% | {c['coverage_rate']*100:.1f}% | - |"
        )

    lines.extend([
        f"",
        "---",
        f"",
        "## 3. 画像洞察",
        f"",
        "| 画像 | 数量 | 渗透率 | NPS | 平均情感 | Top 主题 |",
        "|------|------|--------|-----|----------|----------|",
    ])

    for name, info in personas.items():
        if not name:
            continue
        themes = ", ".join(info["top_themes"][:3])
        lines.append(
            f"| {name.replace('_', ' ').title()} | {info['count']:,} | "
            f"{info['penetration']*100:.1f}% | {info['proxy_nps']['proxy_nps']:.1f} | "
            f"{info['avg_sentiment']:+.2f} | {themes} |"
        )

    lines.extend([
        f"",
        "---",
        f"",
        "## 4. 品牌分析",
        f"",
        f"- 总品牌提及: {brand['total_mentions']:,}",
        f"- 独特品牌: {brand['unique_brands']}",
        f"- 竞品对比: {brand['comparison_count']} ({brand['comparison_rate']*100:.1f}%)",
        f"",
        "| 品牌 | 提及次数 |",
        "|------|----------|",
    ])

    for b, c in brand["brand_distribution"].items():
        lines.append(f"| {b.title()} | {c:,} |")

    lines.extend([
        f"",
        "---",
        f"",
        "## 5. 驱动因素分析",
        f"",
        "### Promoter 驱动 (Top 5)",
        f"",
        "| 主题 | 提及率 | 平均情感 |",
        "|------|--------|----------|",
    ])

    for t in drivers["top_promoter_themes"][:5]:
        lines.append(f"| {t['theme']} | {t['mention_rate']*100:.1f}% | {t['avg_sentiment']:+.2f} |")

    lines.extend([
        f"",
        "### Detractor 驱动 (Top 5)",
        f"",
        "| 主题 | 提及率 | 平均情感 |",
        "|------|--------|----------|",
    ])

    for t in drivers["top_detractor_themes"][:5]:
        lines.append(f"| {t['theme']} | {t['mention_rate']*100:.1f}% | {t['avg_sentiment']:+.2f} |")

    lines.extend([
        f"",
        "---",
        f"",
        "## 6. 逆向完善清单",
        f"",
        "### 零标签高频词 (Top 20)",
        f"",
        "| 词 | 出现次数 | 建议 |",