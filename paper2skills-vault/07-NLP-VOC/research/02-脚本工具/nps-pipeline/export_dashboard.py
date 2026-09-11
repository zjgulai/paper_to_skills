"""VOC Dashboard 导出到 Obsidian + Feishu CSV

从 dashboard_and_insights.json 生成:
1. Obsidian Markdown 报告
2. Feishu 多维表格 CSV
"""

import json
import csv
from pathlib import Path
from datetime import datetime

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/00-归档资料/labeling-outputs/v3.3")
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
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "> 数据源: Amazon / Trustpilot / Reddit / Zendesk",
        f"> 总样本: {coverage['total_voc']:,} 条",
        "",
        "---",
        "",
        "## 1. 全局指标概览",
        "",
        "### Proxy NPS",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| **Proxy NPS** | {nps['proxy_nps']:.1f} |",
        f"| 推荐者 | {nps['promoters']:,} ({nps['promoter_pct']}%) |",
        f"| 被动者 | {nps['passives']:,} |",
        f"| 贬损者 | {nps['detractors']:,} ({nps['detractor_pct']}%) |",
        "",
        "### AIPL 旅程漏斗",
        "",
        "| 节点 | VOC 数 | Top 主题 |",
        "|------|--------|----------|",
    ]

    for node in ["A", "I", "P1", "P2", "L1", "L2", "L3"]:
        info = funnel[node]
        themes = ", ".join(f"{t['theme']}({t['count']})" for t in info["top_themes"])
        lines.append(f"| **{node}** | {info['count']:,} | {themes} |")

    lines.extend([
        "",
        "### 标签覆盖率",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 总 VOC | {coverage['total_voc']:,} |",
        f"| 命中标签 | {coverage['matched_voc']:,} |",
        f"| 覆盖率 | {coverage['coverage_rate']*100:.1f}% |",
        f"| 命中标签种类 | {coverage['unique_tags_matched']} |",
        "",
        "---",
        "",
        "## 2. 各数据源对比",
        "",
        "| 数据源 | NPS | 推荐者% | 贬损者% | 覆盖率 |",
        "|--------|-----|---------|---------|--------|",
    ])

    for src, sd in sources.items():
        n = sd["proxy_nps"]
        c = sd["tag_coverage"]
        lines.append(
            f"| {src.title()} | {n['proxy_nps']:.1f} | {n['promoter_pct']}% | "
            f"{n['detractor_pct']}% | {c['coverage_rate']*100:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. 画像洞察",
        "",
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
        "",
        "---",
        "",
        "## 4. 品牌分析",
        "",
        f"- 总品牌提及: {brand['total_mentions']:,}",
        f"- 独特品牌: {brand['unique_brands']}",
        f"- 竞品对比: {brand['comparison_count']} ({brand['comparison_rate']*100:.1f}%)",
        "",
        "| 品牌 | 提及次数 |",
        "|------|----------|",
    ])

    for b, c in brand["brand_distribution"].items():
        lines.append(f"| {b.title()} | {c:,} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. 驱动因素分析",
        "",
        "### Promoter 驱动 (Top 5)",
        "",
        "| 主题 | 提及率 | 平均情感 |",
        "|------|--------|----------|",
    ])

    for t in drivers["top_promoter_themes"][:5]:
        lines.append(f"| {t['theme']} | {t['mention_rate']*100:.1f}% | {t['avg_sentiment']:+.2f} |")

    lines.extend([
        "",
        "### Detractor 驱动 (Top 5)",
        "",
        "| 主题 | 提及率 | 平均情感 |",
        "|------|--------|----------|",
    ])

    for t in drivers["top_detractor_themes"][:5]:
        lines.append(f"| {t['theme']} | {t['mention_rate']*100:.1f}% | {t['avg_sentiment']:+.2f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. 逆向完善清单",
        "",
        "### 零标签高频词 (Top 20)",
        "",
        "| 词 | 出现次数 | 建议 |",
        "|----|----------|------|",
    ])

    for w, c in plan["zero_label_top_words"][:20]:
        lines.append(f"| {w} | {c} | 映射到现有标签或创建新标签 |")

    lines.extend([
        "",
        "### 高冲突标签 (>20% 冲突率)",
        "",
        "| 标签 | 命中 | 冲突 | 冲突率 |",
        "|------|------|------|--------|",
    ])

    for t in plan["high_conflict_tags"][:10]:
        lines.append(f"| {t['tag']} | {t['hits']} | {t['conflicts']} | {t['conflict_rate']}% |")

    lines.extend([
        "",
        "### 完善建议",
        "",
    ])

    for s in plan["suggestions"]:
        lines.append(f"**[{s['priority']}优先级] {s['issue']}**")
        lines.append(f"- 证据: {s['evidence']}")
        lines.append(f"- 行动: {s['action']}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 附录: 可视化图表",
        "",
        "- `viz_dashboard.png` — 全局看板 (6图表)",
        "- `viz_dashboard_sources.png` — 数据源对比 (4图表)",
        "",
        "> 图表位于同目录下，可直接在 Obsidian 中嵌入查看。",
        "",
    ])

    return "\n".join(lines)


def generate_feishu_csvs(data: dict, export_dir: Path):
    """生成 Feishu 多维表格 CSV"""
    export_dir.mkdir(parents=True, exist_ok=True)

    gd = data["global_dashboard"]
    sources = data["source_dashboards"]
    plan = data["reverse_improvement_plan"]

    # 1. 数据源指标表
    csv_path = export_dir / "source_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["数据源", "总VOC", "Proxy NPS", "推荐者%", "贬损者%", "覆盖率%", "命中标签数"])
        for src, sd in sources.items():
            n = sd["proxy_nps"]
            c = sd["tag_coverage"]
            writer.writerow([
                src.title(), c["total_voc"], n["proxy_nps"],
                n["promoter_pct"], n["detractor_pct"],
                round(c["coverage_rate"] * 100, 1), c["unique_tags_matched"],
            ])
    print(f"  CSV: {csv_path}")

    # 2. AIPL 漏斗表
    csv_path = export_dir / "aipl_funnel.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["节点", "VOC数", "Top主题1", "Top主题2", "Top主题3"])
        for node in ["A", "I", "P1", "P2", "L1", "L2", "L3"]:
            info = gd["aipl_funnel"][node]
            themes = [t["theme"] for t in info["top_themes"]] + ["", "", ""]
            writer.writerow([node, info["count"], themes[0], themes[1], themes[2]])
    print(f"  CSV: {csv_path}")

    # 3. 画像洞察表
    csv_path = export_dir / "persona_insights.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["画像", "数量", "渗透率%", "NPS", "平均情感", "Top主题"])
        for name, info in gd["persona_insights"].items():
            if not name:
                continue
            themes = ", ".join(info["top_themes"][:3])
            writer.writerow([
                name.replace("_", " ").title(), info["count"],
                round(info["penetration"] * 100, 1),
                info["proxy_nps"]["proxy_nps"],
                info["avg_sentiment"],
                themes,
            ])
    print(f"  CSV: {csv_path}")

    # 4. 驱动因素表
    csv_path = export_dir / "driver_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["类型", "主题", "提及率%", "平均情感", "数量"])
        for t in gd["driver_analysis"]["top_promoter_themes"][:10]:
            writer.writerow(["Promoter", t["theme"], round(t["mention_rate"] * 100, 2), t["avg_sentiment"], t["count"]])
        for t in gd["driver_analysis"]["top_detractor_themes"][:10]:
            writer.writerow(["Detractor", t["theme"], round(t["mention_rate"] * 100, 2), t["avg_sentiment"], t["count"]])
    print(f"  CSV: {csv_path}")

    # 5. 品牌分析表
    csv_path = export_dir / "brand_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["品牌", "提及次数"])
        for brand, count in gd["brand_analysis"]["brand_distribution"].items():
            writer.writerow([brand.title(), count])
    print(f"  CSV: {csv_path}")

    # 6. 零标签高频词表
    csv_path = export_dir / "zero_label_words.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["词", "出现次数", "优先级"])
        for i, (w, c) in enumerate(plan["zero_label_top_words"][:30]):
            priority = "高" if i < 5 else "中" if i < 15 else "低"
            writer.writerow([w, c, priority])
    print(f"  CSV: {csv_path}")

    # 7. 高冲突标签表
    csv_path = export_dir / "high_conflict_tags.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["标签", "命中次数", "冲突次数", "冲突率%"])
        for t in plan["high_conflict_tags"][:20]:
            writer.writerow([t["tag"], t["hits"], t["conflicts"], t["conflict_rate"]])
    print(f"  CSV: {csv_path}")


def main():
    print("=" * 60)
    print("VOC Dashboard 导出")
    print("=" * 60)

    data = load_dashboard()
    EXPORT_BASE.mkdir(parents=True, exist_ok=True)

    # 1. Obsidian Markdown
    md_content = generate_obsidian_report(data)
    md_path = EXPORT_BASE / "voc_dashboard_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[Obsidian] Markdown 报告: {md_path}")

    # 2. Feishu CSVs
    print("\n[Feishu] 多维表格 CSV:")
    generate_feishu_csvs(data, EXPORT_BASE / "feishu_csvs")

    # 3. 复制图片到导出目录
    for img_name in ["viz_dashboard.png", "viz_dashboard_sources.png"]:
        src = OUTPUT_BASE / img_name
        if src.exists():
            dst = EXPORT_BASE / img_name
            import shutil
            shutil.copy2(src, dst)
            print(f"\n[图片] {img_name} -> {dst}")

    print(f"\n{'=' * 60}")
    print(f"导出完成: {EXPORT_BASE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
